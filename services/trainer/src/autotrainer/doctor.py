"""Read-only host and Python runtime checks."""

from __future__ import annotations

import importlib.metadata
import platform
import shutil
import subprocess
import sys
from typing import Any


REFERENCE_PACKAGES = {
    "torch": "2.13.0",
    "transformers": "5.13.1",
    "trl": "1.8.0",
    "peft": "0.19.1",
    "accelerate": "1.14.0",
    "datasets": "5.0.0",
    "jmespath": "1.1.0",
    "bitsandbytes": "0.49.2",
}


def _package_check(name: str, expected: str) -> dict[str, Any]:
    try:
        installed = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return {"name": name, "status": "missing", "expected": expected}
    return {
        "name": name,
        "status": "ready" if installed == expected else "version-mismatch",
        "installed": installed,
        "expected": expected,
    }


def _command_check(command: str, arguments: list[str]) -> dict[str, Any]:
    executable = shutil.which(command)
    if not executable:
        return {"name": command, "status": "missing"}
    try:
        completed = subprocess.run(
            [executable, *arguments],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        first_line = (completed.stdout or completed.stderr).strip().splitlines()
        return {
            "name": command,
            "status": "ready" if completed.returncode == 0 else "error",
            "detail": first_line[0][:240] if first_line else f"exit {completed.returncode}",
        }
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"name": command, "status": "error", "detail": str(error)}


def run_doctor(*, environment_backend: str = "docker") -> dict[str, Any]:
    packages = [_package_check(name, version) for name, version in REFERENCE_PACKAGES.items()]
    gpu = _command_check("nvidia-smi", ["--query-gpu=name,memory.total", "--format=csv,noheader"])
    sandbox = _command_check(environment_backend, ["version", "--format", "{{.Server.Version}}"])
    return {
        "python": {
            "status": "ready" if sys.version_info[:2] == (3, 11) else "unsupported",
            "version": platform.python_version(),
            "expected": "3.11.x",
        },
        "platform": platform.platform(),
        "gpu": gpu,
        "sandbox": sandbox,
        "packages": packages,
        "sft_ready": gpu["status"] == "ready" and all(item["status"] == "ready" for item in packages),
        "rl_ready": gpu["status"] == "ready"
        and sandbox["status"] == "ready"
        and all(item["status"] == "ready" for item in packages),
    }
