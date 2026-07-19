from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch


SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE_ROOT / "src"))

from autotrainer.cli import main  # noqa: E402
from autotrainer.config import default_config, write_config  # noqa: E402
from autotrainer.doctor import _package_check, _torch_gpu_check, run_doctor  # noqa: E402


class DoctorTests(unittest.TestCase):
    def test_pinned_but_unimportable_package_is_not_ready(self) -> None:
        with (
            patch("autotrainer.doctor.importlib.metadata.version", return_value="2.13.0"),
            patch(
                "autotrainer.doctor.importlib.import_module",
                side_effect=OSError("native CUDA library is unavailable"),
            ),
        ):
            result = _package_check("torch", "2.13.0")

        self.assertEqual(result["status"], "import-error")
        self.assertIn("native CUDA", result["detail"])

    def test_gpu_check_uses_the_same_exact_one_gpu_guard_as_training(self) -> None:
        class TwoGpuCuda:
            @staticmethod
            def is_available() -> bool:
                return True

            @staticmethod
            def device_count() -> int:
                return 2

        class FakeTorch:
            cuda = TwoGpuCuda()

        with patch("autotrainer.doctor.importlib.import_module", return_value=FakeTorch()):
            result = _torch_gpu_check()

        self.assertEqual(result["status"], "blocked")
        self.assertIn("exactly one visible CUDA GPU", result["detail"])

    def test_sft_can_be_ready_while_rl_waits_for_container_image(self) -> None:
        package = {"name": "package", "status": "ready"}
        commands = [
            {"name": "nvidia-smi", "status": "ready"},
            {"name": "docker", "status": "ready"},
            {"name": "docker", "status": "error", "detail": "image not found"},
        ]
        with (
            patch(
                "autotrainer.doctor._probe_python_runtime",
                return_value={
                    "packages": [package],
                    "gpu": {
                        "status": "ready",
                        "device_count": 1,
                        "vram_gib": 24.0,
                        "bf16_supported": True,
                    },
                },
            ),
            patch("autotrainer.doctor._command_check", side_effect=commands),
        ):
            result = run_doctor(
                environment_backend="docker",
                environment_image="autotrainer/frontend-runtime:0.1",
            )

        self.assertTrue(result["sft_ready"])
        self.assertFalse(result["rl_ready"])
        self.assertEqual(result["environment_image"]["status"], "error")

    def test_python_version_is_part_of_readiness(self) -> None:
        package = {"name": "package", "status": "ready"}
        with (
            patch(
                "autotrainer.doctor._probe_python_runtime",
                return_value={"packages": [package], "gpu": {"status": "ready"}},
            ),
            patch(
                "autotrainer.doctor._command_check",
                return_value={"name": "command", "status": "ready"},
            ),
            patch("autotrainer.doctor.sys.version_info", (3, 12, 0)),
            patch("autotrainer.doctor.platform.python_version", return_value="3.12.0"),
        ):
            result = run_doctor(environment_image="runtime:latest")

        self.assertFalse(result["sft_ready"])
        self.assertFalse(result["rl_ready"])
        self.assertEqual(result["python"]["status"], "unsupported")

    def test_native_runtime_checks_run_in_a_short_lived_child(self) -> None:
        payload = {
            "packages": [{"name": "torch", "status": "ready"}],
            "gpu": {"status": "ready", "device_count": 1},
        }
        completed = subprocess.CompletedProcess(
            [sys.executable, "-m", "autotrainer.runtime_probe"],
            0,
            json.dumps(payload),
            "",
        )
        with patch("autotrainer.doctor.subprocess.run", return_value=completed) as run:
            from autotrainer.doctor import _probe_python_runtime

            result = _probe_python_runtime()

        self.assertEqual(result, payload)
        self.assertEqual(
            run.call_args.args[0],
            [sys.executable, "-m", "autotrainer.runtime_probe"],
        )

    def test_cli_requires_rl_readiness_when_grpo_is_enabled(self) -> None:
        doctor = {"sft_ready": True, "rl_ready": False}
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "autotrainer.yaml"
            payload = default_config()
            write_config(config_path, payload, overwrite=False)
            with (
                patch("autotrainer.cli.run_doctor", return_value=doctor),
                redirect_stdout(StringIO()),
            ):
                self.assertEqual(
                    main(["doctor", "--config", str(config_path), "--json"]),
                    3,
                )

            payload["grpo"]["enabled"] = False
            write_config(config_path, payload, overwrite=True)
            with (
                patch("autotrainer.cli.run_doctor", return_value=doctor),
                redirect_stdout(StringIO()),
            ):
                self.assertEqual(
                    main(["doctor", "--config", str(config_path), "--json"]),
                    0,
                )


if __name__ == "__main__":
    unittest.main()
