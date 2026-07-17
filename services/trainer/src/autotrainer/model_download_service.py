"""Durable background model downloads for the human interface.

Agent CLI downloads remain synchronous and script-friendly.  A browser should
not hold a multi-gigabyte Hub request open, so this manager reserves the same
project lease, writes a small durable receipt, and finishes the shared model
operation in a non-daemon worker.  It never invents byte percentages when the
underlying Hub transfer has not reported them.
"""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import re
from threading import Lock, Thread, current_thread
from typing import Any, Mapping
from uuid import uuid4

from .config import ConfigError, load_config
from .model_cache import (
    ModelCacheError,
    materialize_model_owned,
    materialize_reference_model_owned,
)
from .project_gate import ProjectLease, acquire_project_lease


_SCHEMA_VERSION = 1
_STATUSES = frozenset({"idle", "queued", "downloading", "completed", "failed", "interrupted"})
_LIVE = frozenset({"queued", "downloading"})
_JOB_ID = re.compile(r"^[0-9a-f]{32}$")


class ModelDownloadError(ConfigError):
    """Raised when a second or invalid human download is requested."""


def _idle() -> dict[str, Any]:
    return {
        "id": None,
        "status": "idle",
        "message": "No model download has started.",
        "model_id": None,
        "revision": None,
        "result": None,
        "kind": None,
    }


def _safe_result(value: object) -> dict[str, Any] | None:
    """Retain useful transfer evidence without upstream errors or credentials."""

    if not isinstance(value, Mapping):
        return None
    result: dict[str, Any] = {}
    for field in ("status", "model_id", "revision", "snapshot_path", "cache_dir", "receipt"):
        item = value.get(field)
        if isinstance(item, str) and len(item) <= 4_096:
            result[field] = item
    for field in ("file_count", "logical_bytes"):
        item = value.get(field)
        if isinstance(item, int) and not isinstance(item, bool) and item >= 0:
            result[field] = item
    return result or None


def _normalize(value: object) -> dict[str, Any] | None:
    if not isinstance(value, Mapping) or value.get("status") not in _STATUSES:
        return None
    if value.get("status") == "idle":
        return _idle()
    job_id = value.get("id")
    if not isinstance(job_id, str) or _JOB_ID.fullmatch(job_id) is None:
        return None
    model_id = value.get("model_id")
    revision = value.get("revision")
    if not isinstance(model_id, str) or not isinstance(revision, str):
        return None
    message = str(value.get("message", "Model download status is available."))
    return {
        "id": job_id,
        "status": str(value["status"]),
        "message": message.replace("\r", " ").replace("\n", " ")[:1_000],
        "model_id": model_id[:500],
        "revision": revision[:128],
        "result": _safe_result(value.get("result")),
        "kind": str(value.get("kind", "project")),
    }


def _write(path: Path, job: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(
                {"schema_version": _SCHEMA_VERSION, "job": dict(job)},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, Mapping) or payload.get("schema_version") != _SCHEMA_VERSION:
        return None
    return _normalize(payload.get("job"))


class ModelDownloadManager:
    """Own at most one background Hub snapshot job for one project."""

    def __init__(self, config_path: str | Path) -> None:
        self._config_path = Path(config_path).expanduser().resolve()
        config = load_config(self._config_path)
        self._record_path = config.artifact_dir / "models" / "download-job.json"
        self._lock = Lock()
        self._worker: Thread | None = None
        self._job = _read(self._record_path) or _idle()
        if self._job["status"] in _LIVE:
            # A manager is process-local and downloads are not resumable.  If a
            # new manager reads a live receipt, the prior backend disappeared.
            self._job.update(
                status="interrupted",
                message="The previous model download was interrupted. Start it again to verify the snapshot.",
                result=None,
            )
            _write(self._record_path, self._job)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return deepcopy(self._job)

    def start(self) -> dict[str, Any]:
        """Reserve the project immediately, then return a queued receipt."""

        return self._start(kind="project", model_name=None)

    def start_reference(
        self, model_name: str = "qwythos-9b-reference"
    ) -> dict[str, Any]:
        """Queue the one catalogued V1 benchmark reference in the shared cache."""

        from .models import resolve_model

        profile = resolve_model(model_name)
        if profile.get("purpose") != "benchmark_reference":
            raise ModelDownloadError("model is not a catalogued V1 benchmark reference")
        return self._start(kind="reference", model_name=model_name)

    def _start(self, *, kind: str, model_name: str | None) -> dict[str, Any]:
        """Reserve the project and dispatch one selected or reference transfer."""

        with self._lock:
            if self._job["status"] in _LIVE:
                raise ModelDownloadError("A model download is already active for this project.")
            config = load_config(self._config_path)
            if kind == "reference":
                from .models import resolve_model

                profile = resolve_model(str(model_name))
                model_id = str(profile.get("id", "")).strip()
                revision = str(profile.get("default_revision", "")).strip()
            else:
                model_id = str(config.model.get("id", "")).strip()
                revision = str(config.model.get("revision", "")).strip()
            if not model_id or not revision:
                raise ModelDownloadError("Select a Hugging Face model and revision before downloading.")
            lease = acquire_project_lease(self._config_path)
            job_id = uuid4().hex
            previous = self._job
            self._job = {
                "id": job_id,
                "status": "queued",
                "message": "The exact Hugging Face snapshot is queued for download.",
                "model_id": model_id,
                "revision": revision,
                "result": None,
                "kind": kind,
            }
            try:
                _write(self._record_path, self._job)
                worker = Thread(
                    target=self._run,
                    args=(job_id, lease, kind, model_name),
                    name=f"autotrainer-model-download-{job_id[:8]}",
                    daemon=False,
                )
                self._worker = worker
                worker.start()
            except Exception:
                self._job = previous
                self._worker = None
                lease.release()
                raise
            return deepcopy(self._job)

    def _update(self, job_id: str, **values: Any) -> None:
        with self._lock:
            if self._job.get("id") != job_id:
                return
            self._job.update(values)
            _write(self._record_path, self._job)

    def _run(
        self,
        job_id: str,
        lease: ProjectLease,
        kind: str,
        model_name: str | None,
    ) -> None:
        try:
            with lease.activate("run"):
                self._update(
                    job_id,
                    status="downloading",
                    message=(
                        "Downloading the exact snapshot. AutoTrainer reports completion, "
                        "not an estimated percentage."
                    ),
                )
                try:
                    result = (
                        materialize_reference_model_owned(
                            self._config_path,
                            str(model_name or "qwythos-9b-reference"),
                        )
                        if kind == "reference"
                        else materialize_model_owned(self._config_path)
                    )
                except ConfigError as error:
                    message = str(error).replace("\r", " ").replace("\n", " ")[:1_000]
                    self._update(
                        job_id,
                        status="failed",
                        message=message,
                        result=None,
                    )
                except (ModelCacheError, RuntimeError):
                    # Hub exception strings can include endpoints, local paths,
                    # or authentication detail. Keep the chain in the backend
                    # and publish one stable recovery instruction.
                    self._update(
                        job_id,
                        status="failed",
                        message=(
                            "Model download failed. Check disk space, network access, "
                            "model access, and Hugging Face login, then retry."
                        ),
                        result=None,
                    )
                except Exception:
                    self._update(
                        job_id,
                        status="failed",
                        message="The model download stopped after an unexpected local backend failure.",
                        result=None,
                    )
                else:
                    self._update(
                        job_id,
                        status="completed",
                        message="The exact model snapshot is downloaded and ready for offline use.",
                        result=_safe_result(result),
                    )
        finally:
            lease.release()

    def close(self) -> None:
        """Let an active non-daemon transfer finish during graceful shutdown."""

        with self._lock:
            worker = self._worker
        if worker is not None and worker is not current_thread() and worker.is_alive():
            worker.join()


__all__ = ["ModelDownloadError", "ModelDownloadManager"]
