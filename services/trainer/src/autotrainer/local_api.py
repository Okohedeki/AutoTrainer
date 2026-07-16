"""Loopback-only JSON API used by the AutoTrainer human interface.

The API is intentionally thin: every operation delegates to model_service,
which is also used by the CLI. This keeps the browser from becoming a second
source of model policy or hidden project state.
"""

from __future__ import annotations

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit

from .config import ConfigError
from .model_cache import ModelCacheError
from .model_service import (
    download_model,
    get_model,
    list_models,
    model_status,
    select_model,
)


API_PREFIX = "/api/v1"
ALLOWED_ORIGINS = {
    "http://127.0.0.1:3000",
    "http://localhost:3000",
}


class LocalApiServer(ThreadingHTTPServer):
    """HTTP server carrying the one project configuration selected at startup."""

    daemon_threads = True

    def __init__(self, address: tuple[str, int], handler: type[BaseHTTPRequestHandler], config_path: Path):
        super().__init__(address, handler)
        self.config_path = config_path


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

    def _handle(self, operation: Any) -> None:
        try:
            operation()
        except (ConfigError, ValueError) as error:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"error": {"code": "invalid_request", "message": str(error)}},
            )
        except ModelCacheError as error:
            self._send_json(
                HTTPStatus.CONFLICT,
                {"error": {"code": "model_unavailable", "message": str(error)}},
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
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Max-Age", "600")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract
        def operation() -> None:
            path = urlsplit(self.path).path
            if path == f"{API_PREFIX}/health":
                self._send_json(HTTPStatus.OK, {"status": "ok", "config": str(self.server.config_path)})
            elif path == f"{API_PREFIX}/models":
                self._send_json(HTTPStatus.OK, {"models": list_models()})
            elif path == f"{API_PREFIX}/model":
                self._send_json(
                    HTTPStatus.OK,
                    {"model": get_model(self.server.config_path), "cache": model_status(self.server.config_path)},
                )
            elif path == f"{API_PREFIX}/model/status":
                self._send_json(HTTPStatus.OK, model_status(self.server.config_path))
            else:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": {"code": "not_found", "message": "endpoint not found"}})

        self._handle(operation)

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract
        def operation() -> None:
            path = urlsplit(self.path).path
            payload = self._read_json()
            if path == f"{API_PREFIX}/model/select":
                model_name = str(payload.get("model", "")).strip()
                if not model_name:
                    raise ConfigError("model is required")
                result = select_model(
                    self.server.config_path,
                    model_name,
                    revision=str(payload["revision"]).strip() if payload.get("revision") else None,
                    cache_dir=str(payload["cache_dir"]).strip() if payload.get("cache_dir") else None,
                )
                self._send_json(HTTPStatus.OK, {**result, "cache": model_status(self.server.config_path)})
            elif path == f"{API_PREFIX}/model/download":
                self._send_json(HTTPStatus.OK, download_model(self.server.config_path))
            else:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": {"code": "not_found", "message": "endpoint not found"}})

        self._handle(operation)


def create_local_api_server(config_path: str | Path, host: str, port: int) -> LocalApiServer:
    """Create a loopback server without starting its blocking serve loop."""

    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ConfigError("the V1 local API may only bind to a loopback host")
    if not 0 <= port <= 65_535:
        raise ConfigError("port must be between 0 and 65535")
    resolved_config = Path(config_path).expanduser().resolve()
    # Fail before binding a socket if the selected project is invalid.
    get_model(resolved_config)
    return LocalApiServer((host, port), LocalApiHandler, resolved_config)


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
