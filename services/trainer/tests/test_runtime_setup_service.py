from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE_ROOT / "src"))

from autotrainer.config import default_config, write_config  # noqa: E402
from autotrainer.runtime_setup_service import (  # noqa: E402
    apply_runtime_setup_action_owned,
    inspect_runtime_setup,
    RuntimeSetupError,
)


class RuntimeSetupServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.config_path = self.root / "autotrainer.yaml"
        write_config(self.config_path, default_config(), overwrite=False)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_inspection_turns_doctor_blockers_into_fixed_actions(self) -> None:
        doctor = {
            "python": {"status": "ready", "version": "3.11.9", "expected": "3.11.x"},
            "gpu": {"status": "ready"},
            "packages": [
                {"name": "torch", "status": "version-mismatch", "expected": "2.13.0"}
            ],
            "sandbox": {"status": "missing"},
            "environment_image": {"status": "missing"},
            "sft_ready": False,
            "rl_ready": False,
        }
        windows = {
            "applicable": True,
            "wsl_status": "ready",
            "ubuntu_installed": False,
            "winget_available": True,
        }
        with (
            patch("autotrainer.runtime_setup_service.run_doctor", return_value=doctor),
            patch("autotrainer.runtime_setup_service._windows_host", return_value=windows),
            patch("autotrainer.runtime_setup_service._checkout_root", return_value=self.root),
            patch("autotrainer.runtime_setup_service.shutil.which", return_value="tool.exe"),
        ):
            result = inspect_runtime_setup(self.config_path)

        self.assertEqual(result["status"], "action_needed")
        actions = {action["id"]: action for action in result["actions"]}
        self.assertEqual(
            set(actions),
            {
                "install_training_packages",
                "install_wsl_ubuntu",
                "install_docker_desktop",
                "build_runtime_image",
            },
        )
        self.assertIn("torch==2.13.0", actions["install_training_packages"]["command"])
        self.assertTrue(actions["install_wsl_ubuntu"]["requires_admin"])
        self.assertEqual(actions["build_runtime_image"]["status"], "blocked")

    def test_apply_uses_only_the_predeclared_command_without_a_shell(self) -> None:
        selected = {
            "id": "build_runtime_image",
            "title": "Build image",
            "detail": "Build it.",
            "status": "available",
            "command": ["docker", "build", "-t", "runtime:0.1", "."],
            "requires_admin": False,
            "restart_required": False,
        }
        completed = subprocess.CompletedProcess(selected["command"], 0, "built\n", "")
        with (
            patch("autotrainer.runtime_setup_service._selected_action", return_value=selected),
            patch("autotrainer.runtime_setup_service._checkout_root", return_value=self.root),
            patch("autotrainer.runtime_setup_service.shutil.which", return_value="docker.exe"),
            patch("autotrainer.runtime_setup_service.subprocess.run", return_value=completed) as run,
        ):
            result = apply_runtime_setup_action_owned(
                self.config_path,
                "build_runtime_image",
            )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(run.call_args.args[0][0], "docker.exe")
        self.assertNotIn("shell", run.call_args.kwargs)
        self.assertEqual(run.call_args.kwargs["cwd"], self.root)

    def test_apply_rejects_an_unknown_action_before_execution(self) -> None:
        with self.assertRaisesRegex(RuntimeSetupError, "invalid"):
            apply_runtime_setup_action_owned(self.config_path, "run_anything")


if __name__ == "__main__":
    unittest.main()
