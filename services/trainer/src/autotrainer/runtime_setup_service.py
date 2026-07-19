"""Guided, fixed-command setup for the local training runtime.

Doctor remains read-only. This module turns its blockers into explicit actions
that a human can start from the GUI or an agent can invoke through the CLI.
No request field is ever interpreted as a command or package specification.
"""

from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from threading import Lock, Thread, current_thread
from typing import Any, Mapping
from uuid import uuid4

from .config import ConfigError, load_config
from .doctor import REFERENCE_PACKAGES, run_doctor
from .project_gate import ProjectLease, acquire_project_lease


_LIVE = frozenset({"queued", "running"})
_PYTORCH_CUDA_TAG = "cu130"
_PYTORCH_CUDA_INDEX = "https://download.pytorch.org/whl/cu130"
_ACTION_IDS = frozenset(
    {
        "install_training_packages",
        "install_wsl_ubuntu",
        "install_docker_desktop",
        "build_runtime_image",
    }
)


class RuntimeSetupError(ConfigError):
    """Raised when a setup action is unknown, unavailable, or already active."""


def _command_result(command: str, arguments: list[str]) -> dict[str, Any]:
    executable = shutil.which(command)
    if executable is None:
        return {"status": "missing", "detail": f"{command} is not installed"}
    try:
        completed = subprocess.run(
            [executable, *arguments],
            capture_output=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"status": "error", "detail": str(error)[:240]}
    output = completed.stdout or completed.stderr
    # wsl.exe emits UTF-16LE when its stdout is redirected on Windows.
    if b"\x00" in output:
        detail = output.decode("utf-16le", errors="replace")
    else:
        detail = output.decode(errors="replace")
    return {
        "status": "ready" if completed.returncode == 0 else "error",
        "detail": detail.strip()[:2_000],
    }


def _checkout_root() -> Path | None:
    candidate = Path(__file__).resolve().parents[4]
    dockerfile = candidate / "infra" / "frontend-runtime" / "Dockerfile"
    fixture = candidate / "examples" / "frontend-expert" / "fixture-site"
    if dockerfile.is_file() and (fixture / "package-lock.json").is_file():
        return candidate
    return None


def _windows_host() -> dict[str, Any]:
    if platform.system() != "Windows":
        return {
            "applicable": False,
            "wsl_status": "not-applicable",
            "ubuntu_installed": False,
            "winget_available": False,
        }
    wsl = _command_result("wsl.exe", ["--status"])
    distributions = _command_result("wsl.exe", ["--list", "--quiet"])
    names = [
        line.strip().replace("\x00", "")
        for line in str(distributions.get("detail", "")).splitlines()
        if line.strip().replace("\x00", "")
    ]
    return {
        "applicable": True,
        "wsl_status": wsl["status"],
        "wsl_detail": wsl.get("detail"),
        "distributions": names,
        "ubuntu_installed": any(name.casefold().startswith("ubuntu") for name in names),
        "winget_available": shutil.which("winget.exe") is not None,
    }


def _action(
    action_id: str,
    title: str,
    detail: str,
    *,
    status: str,
    command: list[str],
    requires_admin: bool = False,
    restart_required: bool = False,
) -> dict[str, Any]:
    return {
        "id": action_id,
        "title": title,
        "detail": detail,
        "status": status,
        "command": command,
        "requires_admin": requires_admin,
        "restart_required": restart_required,
    }


def inspect_runtime_setup(config_path: str | Path) -> dict[str, Any]:
    """Return Doctor evidence plus only the fixed actions relevant now."""

    config = load_config(config_path)
    environment = config.data["environment"]
    backend = str(environment["backend"])
    image = str(environment.get("image", ""))
    doctor = run_doctor(
        environment_backend=backend,
        environment_image=image,
    )
    windows = _windows_host()
    checkout = _checkout_root()
    packages_ready = all(
        item.get("status") == "ready" for item in doctor.get("packages", [])
    )
    gpu = doctor.get("gpu", {})
    driver = gpu.get("driver", {}) if isinstance(gpu, Mapping) else {}
    cuda_wheel_needed = (
        isinstance(gpu, Mapping)
        and gpu.get("status") != "ready"
        and isinstance(driver, Mapping)
        and driver.get("status") == "ready"
    )
    actions: list[dict[str, Any]] = []
    if windows["applicable"] and not windows["ubuntu_installed"]:
        actions.append(
            _action(
                "install_wsl_ubuntu",
                "Install WSL2 and Ubuntu",
                "Install the Linux distribution used for CUDA and container training.",
                status="available" if shutil.which("wsl.exe") else "blocked",
                command=["wsl.exe", "--install", "-d", "Ubuntu", "--no-launch"],
                requires_admin=True,
                restart_required=True,
            )
        )
    if not packages_ready or cuda_wheel_needed:
        specifications = [
            (
                f"torch=={version}+{_PYTORCH_CUDA_TAG}"
                if name == "torch"
                else f"{name}=={version}"
            )
            for name, version in REFERENCE_PACKAGES.items()
        ]
        actions.append(
            _action(
                "install_training_packages",
                "Install the pinned training stack",
                "Install or correct the exact package matrix checked by Doctor.",
                status="available",
                command=[
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--upgrade",
                    "--extra-index-url",
                    _PYTORCH_CUDA_INDEX,
                    *specifications,
                ],
                restart_required=True,
            )
        )
    if doctor["sandbox"].get("status") != "ready":
        actions.append(
            _action(
                "install_docker_desktop",
                "Install Docker Desktop",
                "Install the configured Docker backend with WSL2 integration.",
                status=(
                    "available"
                    if windows["applicable"] and windows["winget_available"]
                    else "blocked"
                ),
                command=[
                    "winget.exe",
                    "install",
                    "--id",
                    "Docker.DockerDesktop",
                    "--exact",
                    "--accept-package-agreements",
                    "--accept-source-agreements",
                    "--silent",
                    "--disable-interactivity",
                ],
                requires_admin=True,
                restart_required=True,
            )
        )
    if doctor["environment_image"].get("status") != "ready":
        actions.append(
            _action(
                "build_runtime_image",
                "Build the pinned rollout image",
                "Build the configured network-disabled frontend task runtime.",
                status=(
                    "available"
                    if doctor["sandbox"].get("status") == "ready" and checkout
                    else "blocked"
                ),
                command=[
                    backend,
                    "build",
                    "-t",
                    image,
                    "-f",
                    "infra/frontend-runtime/Dockerfile",
                    ".",
                ],
            )
        )
    next_action = next(
        (item for item in actions if item["status"] == "available"),
        actions[0] if actions else None,
    )
    return {
        "status": "ready" if doctor["rl_ready"] else "action_needed",
        "doctor": doctor,
        "host": {
            "platform": platform.platform(),
            "python_executable": sys.executable,
            "windows": windows,
            "checkout_root": str(checkout) if checkout else None,
        },
        "actions": actions,
        "next_action": next_action,
    }


def _action_from_workspace(
    workspace: Mapping[str, Any],
    action_id: str,
) -> dict[str, Any]:
    if action_id not in _ACTION_IDS:
        raise RuntimeSetupError("runtime setup action is invalid")
    selected = next(
        (action for action in workspace["actions"] if action["id"] == action_id),
        None,
    )
    if selected is None:
        raise RuntimeSetupError("that runtime component is already ready")
    if selected["status"] != "available":
        raise RuntimeSetupError(selected["detail"])
    return selected


def _selected_action(config_path: str | Path, action_id: str) -> dict[str, Any]:
    return _action_from_workspace(inspect_runtime_setup(config_path), action_id)


def apply_runtime_setup_action_owned(
    config_path: str | Path,
    action_id: str,
) -> dict[str, Any]:
    """Execute one predeclared action while the caller owns the project lease."""

    selected = _selected_action(config_path, action_id)
    command = list(selected["command"])
    checkout = _checkout_root()
    cwd = checkout if action_id == "build_runtime_image" else None
    executable = shutil.which(command[0])
    if executable is None:
        raise RuntimeSetupError(f"{command[0]} is not installed")
    command[0] = executable
    environment = os.environ.copy()
    if action_id == "install_training_packages":
        environment["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=7_200,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeSetupError(f"runtime setup could not finish: {error}") from error
    if completed.returncode != 0:
        guidance = (
            "Run the AutoTrainer backend as Administrator and retry."
            if selected["requires_admin"]
            else "Review the setup output and retry."
        )
        raise RuntimeSetupError(
            f"{selected['title']} failed with exit {completed.returncode}. "
            f"{guidance}"
        )
    return {
        "action_id": action_id,
        "status": "completed",
        "message": f"{selected['title']} completed.",
        "restart_required": selected["restart_required"],
        "detail": "The fixed setup command completed successfully.",
    }


def apply_runtime_setup_action(
    config_path: str | Path,
    action_id: str,
) -> dict[str, Any]:
    """Synchronously apply one action for CLI automation."""

    lease = acquire_project_lease(config_path)
    try:
        with lease.activate("run"):
            return apply_runtime_setup_action_owned(config_path, action_id)
    finally:
        lease.release()


def _idle() -> dict[str, Any]:
    return {
        "id": None,
        "action_id": None,
        "status": "idle",
        "message": "No runtime setup action is active.",
        "result": None,
    }


class RuntimeSetupManager:
    """Run at most one long machine setup action outside the HTTP request."""

    def __init__(self, config_path: str | Path) -> None:
        self._config_path = Path(config_path).expanduser().resolve()
        self._lock = Lock()
        self._job = _idle()
        self._worker: Thread | None = None
        self._cached_workspace: dict[str, Any] | None = None

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return deepcopy(self._job)

    def workspace(self) -> dict[str, Any]:
        job = self.snapshot()
        with self._lock:
            cached = deepcopy(self._cached_workspace)
        if job["status"] in _LIVE and cached is not None:
            return {**cached, "job": job}
        inspected = inspect_runtime_setup(self._config_path)
        with self._lock:
            self._cached_workspace = deepcopy(inspected)
        return {**inspected, "job": job}

    def start(self, action_id: str) -> dict[str, Any]:
        with self._lock:
            if self._job["status"] in _LIVE:
                raise RuntimeSetupError("A runtime setup action is already active.")
            inspected = inspect_runtime_setup(self._config_path)
            selected = _action_from_workspace(inspected, action_id)
            lease = acquire_project_lease(self._config_path)
            job_id = uuid4().hex
            self._job = {
                "id": job_id,
                "action_id": action_id,
                "status": "queued",
                "message": f"{selected['title']} is queued.",
                "result": None,
            }
            self._cached_workspace = deepcopy(inspected)
            try:
                worker = Thread(
                    target=self._run,
                    args=(job_id, action_id, lease),
                    name=f"autotrainer-runtime-setup-{job_id[:8]}",
                    daemon=False,
                )
                self._worker = worker
                worker.start()
            except Exception:
                self._job = _idle()
                self._worker = None
                lease.release()
                raise
            return deepcopy(self._job)

    def _update(self, job_id: str, **values: Any) -> None:
        with self._lock:
            if self._job.get("id") == job_id:
                self._job.update(values)

    def _run(self, job_id: str, action_id: str, lease: ProjectLease) -> None:
        try:
            with lease.activate("run"):
                self._update(
                    job_id,
                    status="running",
                    message="AutoTrainer is applying the selected runtime setup action.",
                )
                try:
                    result = apply_runtime_setup_action_owned(
                        self._config_path,
                        action_id,
                    )
                except RuntimeSetupError as error:
                    self._update(
                        job_id,
                        status="failed",
                        message=str(error).replace("\r", " ").replace("\n", " ")[:1_000],
                        result=None,
                    )
                except Exception:
                    self._update(
                        job_id,
                        status="failed",
                        message="Runtime setup stopped after an unexpected local failure.",
                        result=None,
                    )
                else:
                    self._update(
                        job_id,
                        status="completed",
                        message=str(result["message"]),
                        result=result,
                    )
        finally:
            lease.release()

    def close(self) -> None:
        with self._lock:
            worker = self._worker
        if worker is not None and worker is not current_thread() and worker.is_alive():
            worker.join()


__all__ = [
    "RuntimeSetupError",
    "RuntimeSetupManager",
    "apply_runtime_setup_action",
    "inspect_runtime_setup",
]
