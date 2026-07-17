"""Durable control plane for the separate local model-host process.

The model process owns the project lease for its entire lifetime.  That makes
Train, Evaluate, and Host mutually exclusive on the same single GPU even when
an agent starts one of them from another terminal.  This module stores only a
small local process receipt; tokens and raw generation logs never enter the
browser response.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
from threading import Lock
import time
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4

from .config import ConfigError, load_config
from .model_host import ModelHostError, resolve_host_spec
from .project_gate import assert_project_available


_RECORD_SCHEMA_VERSION = 1
_ACTIVE = frozenset({"loading", "live"})
_STATUSES = frozenset({"not_ready", "ready", "loading", "live", "stopped", "failed"})


class HostingServiceError(ConfigError):
    """Raised when the local callable model lifecycle cannot continue."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _record_path(config_path: str | Path) -> Path:
    return load_config(config_path).artifact_dir / "hosting" / "current.json"


def _log_path(config_path: str | Path) -> Path:
    return load_config(config_path).artifact_dir / "hosting" / "model-host.log"


def _write_record(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(
                {"schema_version": _RECORD_SCHEMA_VERSION, "host": dict(record)},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_record(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, Mapping) or payload.get("schema_version") != _RECORD_SCHEMA_VERSION:
        return None
    record = payload.get("host")
    if not isinstance(record, Mapping) or record.get("status") not in _STATUSES:
        return None
    return dict(record)


def _pid_alive(pid: object) -> bool:
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError, PermissionError):
        return False
    return True


def _available_port(host: str, port: int) -> None:
    """Fail before model loading if another local service owns the port."""

    family = socket.AF_INET6 if host == "::1" else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as candidate:
        candidate.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            candidate.bind((host, port))
        except OSError as error:
            raise HostingServiceError(f"Local port {port} is already in use.") from error


def _json_request(
    url: str,
    *,
    payload: Mapping[str, Any] | None = None,
    token: str | None = None,
    timeout: float = 1.0,
) -> dict[str, Any]:
    body = None
    headers: dict[str, str] = {"Accept": "application/json"}
    method = "GET"
    if payload is not None:
        body = json.dumps(dict(payload)).encode("utf-8")
        headers["Content-Type"] = "application/json"
        method = "POST"
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(url, data=body, headers=headers, method=method)
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310 - loopback URL is constructed locally
            value = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        try:
            detail = json.loads(error.read().decode("utf-8")).get("error", {}).get("message")
        except Exception:
            detail = None
        raise HostingServiceError(str(detail or f"Local host returned HTTP {error.code}.")) from error
    except (URLError, TimeoutError, OSError) as error:
        raise HostingServiceError("The local model host is not responding yet.") from error
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HostingServiceError("The local model host returned an invalid response.") from error
    if not isinstance(value, dict):
        raise HostingServiceError("The local model host returned an invalid response.")
    return value


def _public(record: Mapping[str, Any]) -> dict[str, Any]:
    """Remove process-control fields before the record reaches localhost UI."""

    allowed = (
        "status",
        "message",
        "endpoint",
        "model",
        "base_model",
        "revision",
        "adapter",
        "pid",
        "started_at",
        "stopped_at",
    )
    return {key: deepcopy(record.get(key)) for key in allowed if key in record}


def _ready_snapshot(config_path: str | Path) -> dict[str, Any]:
    try:
        spec = resolve_host_spec(config_path, "auto")
    except (ConfigError, ModelHostError) as error:
        return {
            "status": "not_ready",
            "message": str(error),
            "endpoint": None,
            "adapter": None,
        }
    return {
        "status": "ready",
        "message": "The downloaded model is ready to host locally.",
        "endpoint": None,
        "model": spec.display_name,
        "base_model": spec.model_id,
        "revision": spec.revision,
        "adapter": spec.adapter_name,
    }


class HostingManager:
    """Start, inspect, test, and stop one project-owned inference process."""

    def __init__(self, config_path: str | Path) -> None:
        self._config_path = Path(config_path).expanduser().resolve()
        self._path = _record_path(self._config_path)
        self._lock = Lock()
        self._process: subprocess.Popen[bytes] | None = None
        self._log_handle: Any = None

    @property
    def config_path(self) -> Path:
        return self._config_path

    def _refresh_owned_process(self) -> None:
        if self._process is None or self._process.poll() is None:
            return
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None

    def snapshot(self) -> dict[str, Any]:
        """Probe the durable process receipt without loading model weights."""

        with self._lock:
            self._refresh_owned_process()
            record = _read_record(self._path)
            if record is None or record.get("status") not in _ACTIVE:
                return _public(record) if record is not None else _ready_snapshot(self._config_path)

            endpoint = str(record.get("endpoint", ""))
            try:
                health = _json_request(f"{endpoint}/health", timeout=0.35)
            except HostingServiceError:
                if _pid_alive(record.get("pid")):
                    record.update(
                        status="loading",
                        message="The model process is loading weights onto the local GPU.",
                    )
                else:
                    record.update(
                        status="failed",
                        message=(
                            "The model process stopped before becoming ready. "
                            "Review the local model-host log, then retry."
                        ),
                        stopped_at=_now(),
                    )
                _write_record(self._path, record)
                return _public(record)

            if health.get("status") != "ready":
                record.update(status="loading", message="The model process is still loading.")
            else:
                record.update(
                    status="live",
                    message="The trained model is callable on this computer.",
                    model=health.get("model", record.get("model")),
                    adapter=health.get("adapter", record.get("adapter")),
                )
            _write_record(self._path, record)
            return _public(record)

    def start(
        self,
        *,
        adapter: str = "auto",
        host: str = "127.0.0.1",
        port: int = 8791,
    ) -> dict[str, Any]:
        """Spawn a model process and return immediately while weights load."""

        if host not in {"127.0.0.1", "localhost", "::1"}:
            raise HostingServiceError("the V1 model host may only bind to loopback")
        if not isinstance(port, int) or isinstance(port, bool) or not 1 <= port <= 65_535:
            raise HostingServiceError("port must be between 1 and 65535")
        with self._lock:
            current = _read_record(self._path)
            if current is not None and current.get("status") in _ACTIVE and _pid_alive(current.get("pid")):
                raise HostingServiceError("The local model host is already active.")

            spec = resolve_host_spec(self._config_path, adapter)
            assert_project_available(self._config_path)
            _available_port(host, port)
            token = secrets.token_urlsafe(48)
            log_path = _log_path(self._config_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = log_path.open("ab", buffering=0)
            environment = os.environ.copy()
            environment["AUTOTRAINER_HOST_TOKEN"] = token
            command = [
                sys.executable,
                "-m",
                "autotrainer.model_host",
                "--config",
                str(self._config_path),
                "--host",
                host,
                "--port",
                str(port),
                "--adapter",
                adapter,
            ]
            creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            try:
                process = subprocess.Popen(  # noqa: S603 - fixed interpreter/module argv
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    env=environment,
                    creationflags=creationflags,
                )
            except Exception:
                log_handle.close()
                raise
            self._process = process
            self._log_handle = log_handle
            endpoint = f"http://{host}:{port}"
            record = {
                "status": "loading",
                "message": "The model process is loading weights onto the local GPU.",
                "endpoint": endpoint,
                "model": spec.display_name,
                "base_model": spec.model_id,
                "revision": spec.revision,
                "adapter": spec.adapter_name,
                "pid": process.pid,
                "control_token": token,
                "started_at": _now(),
                "stopped_at": None,
                "log_path": str(log_path),
            }
            _write_record(self._path, record)
            return _public(record)

    def stop(self) -> dict[str, Any]:
        """Ask the loopback process to release GPU memory and its project lease."""

        with self._lock:
            record = _read_record(self._path)
            if record is None or record.get("status") not in _ACTIVE:
                stopped = record or {
                    "status": "stopped",
                    "message": "The local model host is stopped.",
                    "endpoint": None,
                }
                return _public(stopped)
            endpoint = str(record.get("endpoint", ""))
            token = str(record.get("control_token", ""))
            try:
                _json_request(
                    f"{endpoint}/_autotrainer/shutdown",
                    payload={},
                    token=token,
                    timeout=2,
                )
            except HostingServiceError:
                # A process that already exited needs only a truthful receipt.
                # We never terminate an unknown PID from a stale project file.
                if _pid_alive(record.get("pid")):
                    raise
            process = self._process
            if process is not None:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    process.wait(timeout=5)
            else:
                deadline = time.monotonic() + 10
                while _pid_alive(record.get("pid")) and time.monotonic() < deadline:
                    time.sleep(0.1)
                if _pid_alive(record.get("pid")):
                    raise HostingServiceError("The model host did not stop; retry from its owning terminal.")
            self._refresh_owned_process()
            record.update(
                status="stopped",
                message="The local model host is stopped and GPU memory was released.",
                stopped_at=_now(),
            )
            record.pop("control_token", None)
            _write_record(self._path, record)
            return _public(record)

    def test(self, prompt: str) -> dict[str, Any]:
        """Run one real completion through the same public endpoint users call."""

        text = str(prompt).strip()
        if not text or len(text) > 8_000:
            raise HostingServiceError("test prompt must contain between 1 and 8,000 characters")
        snapshot = self.snapshot()
        if snapshot.get("status") != "live":
            raise HostingServiceError("Wait for the local model host to become live before testing it.")
        endpoint = str(snapshot["endpoint"])
        result = _json_request(
            f"{endpoint}/v1/chat/completions",
            payload={
                "model": snapshot["model"],
                "messages": [{"role": "user", "content": text}],
                "max_tokens": 512,
                "temperature": 0.2,
                "stream": False,
            },
            timeout=300,
        )
        choices = result.get("choices")
        content = None
        if isinstance(choices, list) and choices and isinstance(choices[0], Mapping):
            message = choices[0].get("message")
            if isinstance(message, Mapping) and isinstance(message.get("content"), str):
                content = message["content"]
        if content is None:
            raise HostingServiceError("The local model host returned no assistant message.")
        return {
            "status": "completed",
            "model": result.get("model"),
            "content": content,
            "usage": result.get("usage") if isinstance(result.get("usage"), Mapping) else {},
        }

    def close(self) -> None:
        """Stop only the process this manager owns during dashboard shutdown."""

        with self._lock:
            owned = self._process is not None and self._process.poll() is None
        if owned:
            self.stop()


__all__ = ["HostingManager", "HostingServiceError"]
