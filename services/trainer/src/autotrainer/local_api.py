"""Loopback-only JSON API used by the AutoTrainer human interface.

The API is intentionally thin: every operation delegates to the same Python
services used by the CLI. This keeps the browser from becoming a second source
of model, training, or evaluation policy and avoids hidden GUI-only state.
"""

from __future__ import annotations

from contextlib import suppress
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
from threading import RLock
from typing import Any, Mapping
from urllib.parse import parse_qs, urlsplit

from .config import ConfigError
from .evaluation_service import EvaluationJobManager
from .github_service import GitHubSearchError, search_repositories
from .hosting_service import HostingManager
from .history_service import (
    get_history_workspace,
    retire_stale_reviews,
    review_history_candidate,
)
from .model_cache import inspect_reference_model, ModelCacheError
from .model_service import (
    get_model,
    list_models,
    ModelSearchError,
    model_status,
    search_models,
    select_model,
)
from .model_download_service import ModelDownloadManager
from .project_service import prepare_project
from .project_gate import (
    ProjectBusyError,
    acquire_project_lease,
    assert_project_available,
)
from .source_service import add_source, list_sources, remove_source
from .training_service import TrainingJobManager
from .workspace_service import ProjectWorkspace


API_PREFIX = "/api/v1"
ALLOWED_ORIGINS = {
    "http://127.0.0.1:3000",
    "http://localhost:3000",
}
_SECRET_PATTERNS = (
    re.compile(r"\bhf_[A-Za-z0-9]{8,}\b"),
    re.compile(r"(?i)\b(bearer\s+)[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(
        r"(?i)\b(api[-_ ]?key|token|password|secret)\s*[:=]\s*[^\s,;]+"
    ),
)


def _safe_message(value: object) -> str:
    """Bound expected service errors and remove common credential forms."""

    text = str(value).replace("\r", " ").replace("\n", " ")[:1_000]
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(
            lambda match: (
                f"{match.group(1)}[redacted]" if match.lastindex else "[redacted]"
            ),
            text,
        )
    return text


def _require_keys(
    payload: Mapping[str, Any],
    *,
    allowed: set[str],
    required: set[str] | None = None,
) -> None:
    """Keep every browser operation on one explicit, versioned shape."""

    keys = set(payload)
    if not keys <= allowed:
        raise ConfigError("request contains unsupported fields")
    if required is not None and not required <= keys:
        raise ConfigError("request is missing required fields")


def _query_values(query: str, *, allowed: set[str]) -> dict[str, list[str]]:
    values = parse_qs(query, keep_blank_values=True, strict_parsing=False)
    if not set(values) <= allowed or any(len(items) != 1 for items in values.values()):
        raise ConfigError("query contains unsupported or repeated fields")
    return values


def _event_cursor(query: str) -> int:
    values = _query_values(query, allowed={"after"})
    raw = values.get("after", ["0"])[0]
    if not re.fullmatch(r"[0-9]+", raw):
        raise ConfigError("after must be a non-negative integer")
    return int(raw)


def _required_text(payload: Mapping[str, Any], key: str) -> str:
    """Read one required string without coercing arrays or objects to text."""

    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{key} is required and must be text")
    return value.strip()


def _optional_text(payload: Mapping[str, Any], key: str) -> str | None:
    """Read one optional non-empty string while preserving absence as ``None``."""

    if key not in payload:
        return None
    value = payload[key]
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{key} must be non-empty text")
    return value.strip()


def _optional_text_list(
    payload: Mapping[str, Any],
    key: str,
) -> list[str] | None:
    """Validate bounded browser string lists before source policy sees them."""

    if key not in payload:
        return None
    value = payload[key]
    if not isinstance(value, list) or len(value) > 100:
        raise ConfigError(f"{key} must be a list of at most 100 text values")
    normalized: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip() or len(item) > 1_000:
            raise ConfigError(f"{key} must contain non-empty text values")
        normalized.append(item.strip())
    return normalized


def _download_job_for_kind(
    manager: ModelDownloadManager,
    kind: str,
) -> dict[str, Any] | None:
    """Keep project and fixed-reference transfer state in the correct panel."""

    job = manager.snapshot()
    job_kind = job.get("kind")
    if job_kind == kind or (kind == "project" and job_kind is None):
        return job
    return None


class LocalApiServer(ThreadingHTTPServer):
    """HTTP server carrying one safely switchable local project workspace."""

    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        handler: type[BaseHTTPRequestHandler],
        config_path: Path,
        projects_root: Path,
    ) -> None:
        super().__init__(address, handler)
        self.context_lock = RLock()
        self.workspace = ProjectWorkspace(projects_root, config_path)
        self.model_download = ModelDownloadManager(config_path)
        self.training = TrainingJobManager(config_path)
        self.evaluation = EvaluationJobManager(config_path)
        self.hosting = HostingManager(config_path)

    @property
    def config_path(self) -> Path:
        """Expose the active config for existing service calls and clients."""

        return self.workspace.active_config

    def _assert_context_idle(self) -> None:
        """Reject a context change before it can leave partial project state."""

        # Managers acquire the cross-process lease before exposing a queued
        # job. Their public states also close the short model-host startup
        # window before the child process has fully published its lease.
        if (
            self.model_download.snapshot().get("status") in {"queued", "downloading"}
            or self.training.snapshot().get("status") in {"queued", "running"}
            or self.evaluation.snapshot().get("status") in {"queued", "running"}
            or self.hosting.snapshot().get("status") in {"loading", "live"}
        ):
            raise ProjectBusyError(
                "Stop the active project job or model host before switching projects."
            )

    def _install_project_context(self, project_id: str, target: Path) -> dict[str, Any]:
        """Construct every manager before publishing a new active project."""

        constructed: list[object] = []
        try:
            model_download = ModelDownloadManager(target)
            constructed.append(model_download)
            training = TrainingJobManager(target)
            constructed.append(training)
            evaluation = EvaluationJobManager(target)
            constructed.append(evaluation)
            hosting = HostingManager(target)
            constructed.append(hosting)
            selected = self.workspace.select_project(project_id)
        except Exception:
            # A constructor can fail after earlier managers opened resources.
            # Close only those new managers; the current context stays intact.
            for manager in reversed(constructed):
                with suppress(Exception):
                    manager.close()  # type: ignore[attr-defined]
            raise
        old_download, old_training, old_evaluation, old_hosting = (
            self.model_download,
            self.training,
            self.evaluation,
            self.hosting,
        )
        self.model_download, self.training, self.evaluation, self.hosting = (
            model_download,
            training,
            evaluation,
            hosting,
        )
        # The new context is already published. Cleanup failures in an idle
        # old manager must not turn a successful switch into a failed POST.
        for manager in (old_hosting, old_evaluation, old_training, old_download):
            with suppress(Exception):
                manager.close()
        return selected

    def create_project(self, name: str) -> dict[str, Any]:
        """Create and activate a project as one lease-protected API action."""

        with self.context_lock:
            self._assert_context_idle()
            current_lease = acquire_project_lease(self.config_path)
            target_lease = None
            created: dict[str, Any] | None = None
            try:
                created = self.workspace.create_project(name)
                target = self.workspace.resolve_project(str(created["id"]))
                # The new direct child is locked before it becomes discoverable
                # as the active GUI context. No agent can mutate it mid-switch.
                target_lease = acquire_project_lease(target)
                return self._install_project_context(str(created["id"]), target)
            except Exception:
                if created is not None:
                    # Roll back the exact inactive record created by this
                    # request. Existing projects and unexpected files remain.
                    self.workspace.discard_created_project(
                        str(created["id"]),
                        str(created["config_path"]),
                    )
                raise
            finally:
                if target_lease is not None:
                    target_lease.release()
                current_lease.release()

    def select_project(self, project_id: str) -> dict[str, Any]:
        """Switch managers only after both project leases prove inactive."""

        with self.context_lock:
            target = self.workspace.resolve_project(project_id)
            current = self.config_path
            self._assert_context_idle()
            assert_project_available(current)
            if target != current:
                assert_project_available(target)
            return self._install_project_context(project_id, target)

    def server_close(self) -> None:
        """Close the socket, then let an active non-daemon training write finish."""

        try:
            super().server_close()
        finally:
            # At most one long project job can own the shared lease.  Joining
            # both managers keeps shutdown honest whichever lifecycle owns it.
            self.hosting.close()
            self.evaluation.close()
            self.training.close()
            self.model_download.close()


class LocalApiHandler(BaseHTTPRequestHandler):
    """Serve the versioned model lifecycle without accepting arbitrary commands."""

    server: LocalApiServer

    def log_message(self, format: str, *args: object) -> None:
        # Retain the standard concise access log while making its product source
        # explicit when the backend is embedded in another terminal.
        super().log_message("AutoTrainer API: " + format, *args)

    def _cors_origin(self) -> str | None:
        origin = self.headers.get("Origin")
        return origin if origin in ALLOWED_ORIGINS else None

    def _send_json(self, status: HTTPStatus, payload: Mapping[str, Any]) -> None:
        body = (json.dumps(dict(payload), default=str) + "\n").encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        origin = self._cors_origin()
        if origin:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        raw_length = self.headers.get("Content-Length", "0")
        try:
            length = int(raw_length)
        except ValueError as error:
            raise ConfigError("invalid Content-Length") from error
        if length < 0 or length > 65_536:
            raise ConfigError("request body exceeds 64 KiB")
        if length == 0:
            return {}
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ConfigError("request body must be valid UTF-8 JSON") from error
        if not isinstance(payload, dict):
            raise ConfigError("request body must be a JSON object")
        return payload

    def _allow_mutation(self, *, require_json: bool) -> bool:
        """Reject cross-site writes before reading any attacker-controlled body."""

        origin = self.headers.get("Origin")
        if origin is not None and origin not in ALLOWED_ORIGINS:
            self._send_json(
                HTTPStatus.FORBIDDEN,
                {"error": {"code": "origin_denied", "message": "origin is not allowed"}},
            )
            return False
        if require_json:
            media_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
            if media_type != "application/json":
                self._send_json(
                    HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                    {
                        "error": {
                            "code": "invalid_content_type",
                            "message": "POST requests require application/json",
                        }
                    },
                )
                return False
        return True

    def _handle(self, operation: Any) -> None:
        try:
            operation()
        except ProjectBusyError as error:
            self._send_json(
                HTTPStatus.CONFLICT,
                {"error": {"code": "project_busy", "message": _safe_message(error)}},
            )
        except ModelSearchError as error:
            self._send_json(
                HTTPStatus.SERVICE_UNAVAILABLE,
                {
                    "error": {
                        "code": "model_search_unavailable",
                        "message": _safe_message(error),
                    }
                },
            )
        except GitHubSearchError as error:
            self._send_json(
                HTTPStatus.SERVICE_UNAVAILABLE,
                {
                    "error": {
                        "code": "repository_search_unavailable",
                        "message": _safe_message(error),
                    }
                },
            )
        except (ConfigError, ValueError) as error:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"error": {"code": "invalid_request", "message": _safe_message(error)}},
            )
        except ModelCacheError as error:
            self._send_json(
                HTTPStatus.CONFLICT,
                {"error": {"code": "model_unavailable", "message": _safe_message(error)}},
            )
        except Exception:
            # Do not leak local paths, credentials, or upstream response bodies
            # for unexpected failures. The server terminal retains diagnostics.
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": {"code": "internal_error", "message": "unexpected local backend failure"}},
            )

    def do_OPTIONS(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract
        if self._cors_origin() is None:
            self._send_json(HTTPStatus.FORBIDDEN, {"error": {"code": "origin_denied", "message": "origin is not allowed"}})
            return
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", self._cors_origin() or "")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Max-Age", "600")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract
        def operation() -> None:
            split = urlsplit(self.path)
            path, query = split.path, split.query
            if path == f"{API_PREFIX}/models/search":
                # Hub discovery is independent of the active project and can
                # take seconds on a slow network. Do not pause live training or
                # evaluation event polling while the picker searches.
                values = _query_values(query, allowed={"q", "limit"})
                query_value = values.get("q", [""])[0]
                if not query_value:
                    raise ConfigError("q is required")
                raw_limit = values.get("limit", ["12"])[0]
                if not re.fullmatch(r"[0-9]+", raw_limit):
                    raise ConfigError("limit must be a positive integer")
                self._send_json(
                    HTTPStatus.OK,
                    {
                        "models": search_models(
                            query_value,
                            limit=int(raw_limit),
                        )
                    },
                )
                return
            if path == f"{API_PREFIX}/repositories/search":
                # Discovery performs no project mutation, so live telemetry
                # polling does not wait on a slow GitHub request.
                values = _query_values(query, allowed={"q", "limit"})
                query_value = values.get("q", [""])[0]
                if not query_value:
                    raise ConfigError("q is required")
                raw_limit = values.get("limit", ["8"])[0]
                if not re.fullmatch(r"[0-9]+", raw_limit):
                    raise ConfigError("limit must be a positive integer")
                self._send_json(
                    HTTPStatus.OK,
                    {
                        "repositories": search_repositories(
                            query_value,
                            limit=int(raw_limit),
                        )
                    },
                )
                return
            with self.server.context_lock:
                if path == f"{API_PREFIX}/health":
                    _query_values(query, allowed=set())
                    workspace = self.server.workspace.list_projects()
                    active = next(
                        project
                        for project in workspace["projects"]
                        if project["id"] == workspace["active_id"]
                    )
                    self._send_json(
                        HTTPStatus.OK,
                        {
                            "status": "ok",
                            "config": str(self.server.config_path),
                            "active_project": active,
                        },
                    )
                elif path == f"{API_PREFIX}/projects":
                    _query_values(query, allowed=set())
                    self._send_json(
                        HTTPStatus.OK,
                        self.server.workspace.list_projects(),
                    )
                elif path == f"{API_PREFIX}/models":
                    _query_values(query, allowed=set())
                    self._send_json(HTTPStatus.OK, {"models": list_models()})
                elif path == f"{API_PREFIX}/model":
                    _query_values(query, allowed=set())
                    self._send_json(
                        HTTPStatus.OK,
                        {
                            "model": get_model(self.server.config_path),
                            "cache": model_status(self.server.config_path),
                            "download_job": _download_job_for_kind(
                                self.server.model_download,
                                "project",
                            ),
                        },
                    )
                elif path == f"{API_PREFIX}/model/status":
                    _query_values(query, allowed=set())
                    self._send_json(
                        HTTPStatus.OK,
                        {
                            **model_status(self.server.config_path),
                            "download_job": _download_job_for_kind(
                                self.server.model_download,
                                "project",
                            ),
                        },
                    )
                elif path == f"{API_PREFIX}/reference-model":
                    _query_values(query, allowed=set())
                    self._send_json(
                        HTTPStatus.OK,
                        {
                            **inspect_reference_model(self.server.config_path),
                            "download_job": _download_job_for_kind(
                                self.server.model_download,
                                "reference",
                            ),
                        },
                    )
                elif path == f"{API_PREFIX}/sources":
                    _query_values(query, allowed=set())
                    self._send_json(
                        HTTPStatus.OK,
                        {"sources": list_sources(self.server.config_path)},
                    )
                elif path == f"{API_PREFIX}/history":
                    _query_values(query, allowed=set())
                    self._send_json(
                        HTTPStatus.OK,
                        get_history_workspace(self.server.config_path),
                    )
                elif path == f"{API_PREFIX}/training":
                    _query_values(query, allowed=set())
                    self._send_json(
                        HTTPStatus.OK,
                        self.server.training.snapshot(),
                    )
                elif path == f"{API_PREFIX}/training/events":
                    self._send_json(
                        HTTPStatus.OK,
                        self.server.training.events(_event_cursor(query)),
                    )
                elif path == f"{API_PREFIX}/evaluation":
                    _query_values(query, allowed=set())
                    self._send_json(
                        HTTPStatus.OK,
                        self.server.evaluation.workspace(),
                    )
                elif path == f"{API_PREFIX}/evaluation/events":
                    self._send_json(
                        HTTPStatus.OK,
                        self.server.evaluation.events(_event_cursor(query)),
                    )
                elif path == f"{API_PREFIX}/hosting":
                    _query_values(query, allowed=set())
                    self._send_json(
                        HTTPStatus.OK,
                        self.server.hosting.snapshot(),
                    )
                else:
                    self._send_json(
                        HTTPStatus.NOT_FOUND,
                        {
                            "error": {
                                "code": "not_found",
                                "message": "endpoint not found",
                            }
                        },
                    )

        self._handle(operation)

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract
        if not self._allow_mutation(require_json=True):
            return

        def operation() -> None:
            split = urlsplit(self.path)
            path = split.path
            _query_values(split.query, allowed=set())
            payload = self._read_json()
            with self.server.context_lock:
                if path == f"{API_PREFIX}/projects":
                    _require_keys(payload, allowed={"name"}, required={"name"})
                    self._send_json(
                        HTTPStatus.CREATED,
                        self.server.create_project(
                            _required_text(payload, "name")
                        ),
                    )
                elif path == f"{API_PREFIX}/projects/select":
                    _require_keys(
                        payload,
                        allowed={"project_id"},
                        required={"project_id"},
                    )
                    self._send_json(
                        HTTPStatus.OK,
                        self.server.select_project(
                            _required_text(payload, "project_id")
                        ),
                    )
                elif path == f"{API_PREFIX}/model/select":
                    _require_keys(
                        payload,
                        allowed={"model", "revision"},
                        required={"model"},
                    )
                    result = select_model(
                        self.server.config_path,
                        _required_text(payload, "model"),
                        revision=_optional_text(payload, "revision"),
                    )
                    self._send_json(
                        HTTPStatus.OK,
                        {
                            **result,
                            "cache": model_status(self.server.config_path),
                        },
                    )
                elif path == f"{API_PREFIX}/model/download":
                    _require_keys(payload, allowed=set())
                    self._send_json(
                        HTTPStatus.ACCEPTED,
                        self.server.model_download.start(),
                    )
                elif path == f"{API_PREFIX}/reference-model/download":
                    _require_keys(payload, allowed=set())
                    self._send_json(
                        HTTPStatus.ACCEPTED,
                        self.server.model_download.start_reference(),
                    )
                elif path == f"{API_PREFIX}/sources":
                    _require_keys(
                        payload,
                        allowed={
                            "value",
                            "modes",
                            "revision",
                            "include",
                            "exclude",
                            "license_spdx",
                            "license_attribution",
                        },
                        required={"value"},
                    )
                    self._send_json(
                        HTTPStatus.OK,
                        add_source(
                            self.server.config_path,
                            _required_text(payload, "value"),
                            modes=_optional_text_list(payload, "modes"),
                            require_modes=True,
                            revision=_optional_text(payload, "revision"),
                            include=_optional_text_list(payload, "include"),
                            exclude=_optional_text_list(payload, "exclude"),
                            license_spdx=_optional_text(
                                payload,
                                "license_spdx",
                            ),
                            license_attribution=_optional_text(
                                payload,
                                "license_attribution",
                            ),
                        ),
                    )
                elif path == f"{API_PREFIX}/history/review":
                    _require_keys(
                        payload,
                        allowed={
                            "candidate_id",
                            "decision",
                            "instruction",
                            "rights_confirmed",
                        },
                        required={"candidate_id", "decision"},
                    )
                    rights_value = payload.get("rights_confirmed", False)
                    if not isinstance(rights_value, bool):
                        raise ConfigError("rights_confirmed must be true or false")
                    self._send_json(
                        HTTPStatus.OK,
                        review_history_candidate(
                            self.server.config_path,
                            candidate_id=_required_text(payload, "candidate_id"),
                            decision=_required_text(payload, "decision"),
                            instruction=_optional_text(payload, "instruction"),
                            rights_confirmed=rights_value,
                        ),
                    )
                elif path == f"{API_PREFIX}/history/retire-stale":
                    _require_keys(payload, allowed=set())
                    self._send_json(
                        HTTPStatus.OK,
                        retire_stale_reviews(self.server.config_path),
                    )
                elif path == f"{API_PREFIX}/prepare":
                    _require_keys(payload, allowed=set())
                    # Preparation calls shared Python operations directly; the
                    # API never turns browser input into a shell command.
                    self._send_json(
                        HTTPStatus.OK,
                        prepare_project(self.server.config_path),
                    )
                elif path == f"{API_PREFIX}/training/start":
                    _require_keys(payload, allowed=set())
                    # The manager serializes local jobs and invokes Python stage
                    # runners directly; request data never becomes a command.
                    self._send_json(
                        HTTPStatus.ACCEPTED,
                        self.server.training.start(),
                    )
                elif path == f"{API_PREFIX}/evaluation/plan":
                    _require_keys(payload, allowed=set())
                    self._send_json(
                        HTTPStatus.OK,
                        self.server.evaluation.plan(),
                    )
                elif path == f"{API_PREFIX}/evaluation/start":
                    _require_keys(payload, allowed={"suite"})
                    suite = _optional_text(payload, "suite")
                    if suite is None:
                        workspace = self.server.evaluation.plan()
                        suites = workspace.get("suites")
                        if not isinstance(suites, list):
                            raise ConfigError(
                                "evaluation plan returned no runnable local suite"
                            )
                        suite = next(
                            (
                                str(item.get("id", "")).strip()
                                for item in suites
                                if isinstance(item, Mapping)
                                and item.get("runner_type") in {"builtin", "command"}
                                and str(item.get("id", "")).strip()
                            ),
                            None,
                        )
                        if suite is None:
                            raise ConfigError(
                                "evaluation plan returned no runnable local suite"
                            )
                    self._send_json(
                        HTTPStatus.ACCEPTED,
                        self.server.evaluation.start(suite),
                    )
                elif path == f"{API_PREFIX}/hosting/start":
                    _require_keys(payload, allowed={"adapter", "port"})
                    adapter = _optional_text(payload, "adapter") or "auto"
                    port = payload.get("port", 8791)
                    if not isinstance(port, int) or isinstance(port, bool):
                        raise ConfigError("port must be an integer")
                    self._send_json(
                        HTTPStatus.ACCEPTED,
                        self.server.hosting.start(
                            adapter=adapter,
                            host="127.0.0.1",
                            port=port,
                        ),
                    )
                elif path == f"{API_PREFIX}/hosting/stop":
                    _require_keys(payload, allowed=set())
                    self._send_json(
                        HTTPStatus.OK,
                        self.server.hosting.stop(),
                    )
                elif path == f"{API_PREFIX}/hosting/test":
                    _require_keys(
                        payload,
                        allowed={"prompt"},
                        required={"prompt"},
                    )
                    self._send_json(
                        HTTPStatus.OK,
                        self.server.hosting.test(
                            _required_text(payload, "prompt")
                        ),
                    )
                else:
                    self._send_json(
                        HTTPStatus.NOT_FOUND,
                        {
                            "error": {
                                "code": "not_found",
                                "message": "endpoint not found",
                            }
                        },
                    )

        self._handle(operation)

    def do_DELETE(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract
        if not self._allow_mutation(require_json=False):
            return

        def operation() -> None:
            split = urlsplit(self.path)
            path = split.path
            _query_values(split.query, allowed=set())
            with self.server.context_lock:
                prefix = f"{API_PREFIX}/sources/"
                if not path.startswith(prefix) or not path[len(prefix) :]:
                    self._send_json(
                        HTTPStatus.NOT_FOUND,
                        {
                            "error": {
                                "code": "not_found",
                                "message": "endpoint not found",
                            }
                        },
                    )
                    return
                source_id = path[len(prefix) :]
                # IDs are single slugs; rejecting separators keeps routing and
                # filesystem cleanup independent from user-controlled paths.
                if "/" in source_id or not re.fullmatch(
                    r"[a-z0-9][a-z0-9._-]*",
                    source_id,
                ):
                    raise ConfigError("source id is invalid")
                self._send_json(
                    HTTPStatus.OK,
                    remove_source(self.server.config_path, source_id),
                )

        self._handle(operation)


def create_local_api_server(
    config_path: str | Path,
    host: str,
    port: int,
    projects_root: str | Path | None = None,
) -> LocalApiServer:
    """Create a loopback server without starting its blocking serve loop."""

    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ConfigError("the V1 local API may only bind to a loopback host")
    if not 0 <= port <= 65_535:
        raise ConfigError("port must be between 0 and 65535")
    resolved_config = Path(config_path).expanduser().resolve()
    resolved_projects_root = (
        Path(projects_root).expanduser().resolve()
        if projects_root is not None
        else (resolved_config.parent / ".autotrainer" / "projects").resolve()
    )
    # Fail before binding a socket if the selected project is invalid.
    get_model(resolved_config)
    return LocalApiServer(
        (host, port),
        LocalApiHandler,
        resolved_config,
        resolved_projects_root,
    )


def serve_local_api(config_path: str | Path, host: str = "127.0.0.1", port: int = 8765) -> None:
    """Run the human-interface backend until interrupted by the operator."""

    server = create_local_api_server(config_path, host, port)
    address, bound_port = server.server_address[:2]
    print(f"AutoTrainer local API: http://{address}:{bound_port}{API_PREFIX}")
    print(f"Project config: {server.config_path}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


__all__ = ["API_PREFIX", "create_local_api_server", "serve_local_api"]
