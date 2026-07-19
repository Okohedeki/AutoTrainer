"""Read-only host and Python runtime checks."""

from __future__ import annotations

import importlib
import importlib.metadata
import platform
import shutil
import subprocess
import sys
from typing import Any

from .training.common import validate_single_gpu


REFERENCE_PACKAGES = {
    "torch": "2.13.0",
    # Transformers probes torchvision while importing model classes. Keep the
    # companion wheel aligned with PyTorch so an unrelated host installation
    # cannot make the otherwise pinned training stack fail at import time.
    "torchvision": "0.28.0",
    "transformers": "5.13.1",
    "trl": "1.8.0",
    "peft": "0.19.1",
    "accelerate": "1.14.0",
    "datasets": "5.0.0",
    "regex": "2025.10.22",
    "jmespath": "1.1.0",
    "bitsandbytes": "0.49.2",
}

# Check the public objects imported by the real runners, not only wheel
# metadata. A correctly pinned package can still be unusable because of a
# broken CUDA/native-library installation.
REFERENCE_IMPORTS: dict[str, tuple[tuple[str, tuple[str, ...]], ...]] = {
    "torch": (("torch", ()),),
    "torchvision": (("torchvision", ()),),
    "transformers": (
        (
            "transformers",
            ("AutoModelForCausalLM", "AutoTokenizer", "BitsAndBytesConfig"),
        ),
    ),
    "trl": (
        ("trl", ("SFTConfig", "SFTTrainer", "GRPOConfig", "GRPOTrainer")),
        ("trl.chat_template_utils", ("get_training_chat_template",)),
    ),
    "peft": (
        (
            "peft",
            ("LoraConfig", "PeftModel", "get_peft_model", "prepare_model_for_kbit_training"),
        ),
    ),
    "accelerate": (("accelerate", ()),),
    "datasets": (("datasets", ("load_dataset",)),),
    "regex": (("regex", ()),),
    "jmespath": (("jmespath", ()),),
    "bitsandbytes": (("bitsandbytes", ()),),
}


def _package_check(name: str, expected: str) -> dict[str, Any]:
    try:
        installed = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return {"name": name, "status": "missing", "expected": expected}
    if installed.split("+", 1)[0] != expected:
        return {
            "name": name,
            "status": "version-mismatch",
            "installed": installed,
            "expected": expected,
        }
    try:
        for module_name, attributes in REFERENCE_IMPORTS[name]:
            module = importlib.import_module(module_name)
            missing = [attribute for attribute in attributes if not hasattr(module, attribute)]
            if missing:
                raise ImportError(
                    f"{module_name} is missing: {', '.join(sorted(missing))}"
                )
    except Exception as error:
        return {
            "name": name,
            "status": "import-error",
            "installed": installed,
            "expected": expected,
            "detail": str(error)[:240],
        }
    return {
        "name": name,
        "status": "ready",
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


def _torch_gpu_check() -> dict[str, Any]:
    """Run the same no-weight CUDA guard used immediately before training."""

    try:
        torch_module = importlib.import_module("torch")
        details = validate_single_gpu(torch_module)
    except Exception as error:
        return {"status": "blocked", "detail": str(error)[:240]}
    return {"status": "ready", **details}


def run_doctor(
    *,
    environment_backend: str = "docker",
    environment_image: str = "",
) -> dict[str, Any]:
    """Inspect every prerequisite a selected local V1 run can verify early."""

    packages = [_package_check(name, version) for name, version in REFERENCE_PACKAGES.items()]
    gpu = _torch_gpu_check()
    # Driver output is diagnostic only. The readiness decision uses PyTorch's
    # CUDA visibility because it honors CUDA_VISIBLE_DEVICES exactly as the
    # stage runner will.
    gpu["driver"] = _command_check(
        "nvidia-smi",
        ["--query-gpu=driver_version", "--format=csv,noheader"],
    )
    sandbox = _command_check(environment_backend, ["version", "--format", "{{.Server.Version}}"])
    image = (
        _command_check(
            environment_backend,
            ["image", "inspect", environment_image, "--format", "{{.Id}}"],
        )
        if environment_image.strip()
        else {
            "name": f"{environment_backend} image",
            "status": "missing",
            "detail": "environment.image is not configured",
        }
    )
    python = {
        "status": "ready" if sys.version_info[:2] == (3, 11) else "unsupported",
        "version": platform.python_version(),
        "expected": "3.11.x",
    }
    packages_ready = all(item["status"] == "ready" for item in packages)
    base_ready = python["status"] == "ready" and gpu["status"] == "ready" and packages_ready
    return {
        "python": python,
        "platform": platform.platform(),
        "gpu": gpu,
        "sandbox": sandbox,
        "environment_image": image,
        "packages": packages,
        "sft_ready": base_ready,
        "rl_ready": base_ready
        and sandbox["status"] == "ready"
        and image["status"] == "ready",
    }
