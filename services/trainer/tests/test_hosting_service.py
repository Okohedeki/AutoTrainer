from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE_ROOT / "src"))

from autotrainer.config import default_config, write_config  # noqa: E402
from autotrainer.hosting_service import HostingManager, HostingServiceError  # noqa: E402


class FakeProcess:
    def __init__(self, pid: int = 424242) -> None:
        self.pid = pid
        self.returncode: int | None = None

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = 0
        return 0

    def terminate(self):
        self.returncode = -1


class HostingServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.config_path = self.root / "autotrainer.yaml"
        self.revision = "b" * 40
        payload = default_config(revision=self.revision)
        payload["model"]["cache_dir"] = "./model-cache"
        write_config(self.config_path, payload, overwrite=False)
        snapshot_path = (
            self.root
            / "model-cache"
            / "models--Qwen--Qwen3.5-9B"
            / "snapshots"
            / self.revision
        )
        snapshot_path.mkdir(parents=True)
        # Match the usable immutable Hub layout required before a process can
        # be advertised as ready; no model weights are loaded by this fixture.
        (snapshot_path / "config.json").write_text(
            json.dumps({"model_type": "qwen3_5"}), encoding="utf-8"
        )
        (snapshot_path / "tokenizer.json").write_text("{}", encoding="utf-8")
        (snapshot_path / "model.safetensors").write_bytes(b"weights")
        receipt_path = self.root / ".autotrainer" / "models" / "current.json"
        receipt_path.parent.mkdir(parents=True)
        receipt_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "model_id": "Qwen/Qwen3.5-9B",
                    "requested_revision": self.revision,
                    "revision": self.revision,
                    "snapshot_path": str(snapshot_path),
                    "cache_dir": str((self.root / "model-cache").resolve()),
                    "file_count": 3,
                    "logical_bytes": sum(
                        item.stat().st_size for item in snapshot_path.iterdir()
                    ),
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_snapshot_is_ready_before_a_process_starts(self) -> None:
        snapshot = HostingManager(self.config_path).snapshot()

        self.assertEqual(snapshot["status"], "ready")
        self.assertEqual(snapshot["base_model"], "Qwen/Qwen3.5-9B")
        self.assertEqual(snapshot["adapter"], "base")
        self.assertIsNone(snapshot["endpoint"])

    def test_start_spawns_fixed_module_and_hides_control_fields(self) -> None:
        manager = HostingManager(self.config_path)
        process = FakeProcess()
        with (
            patch("autotrainer.hosting_service._available_port"),
            patch("autotrainer.hosting_service.subprocess.Popen", return_value=process) as popen,
        ):
            result = manager.start(adapter="base", port=9876)

        self.assertEqual(result["status"], "loading")
        self.assertEqual(result["endpoint"], "http://127.0.0.1:9876")
        self.assertNotIn("control_token", result)
        self.assertNotIn("log_path", result)
        command = popen.call_args.args[0]
        self.assertEqual(command[1:3], ["-m", "autotrainer.model_host"])
        self.assertIn(str(self.config_path.resolve()), command)

        process.returncode = 0
        with patch("autotrainer.hosting_service._pid_alive", return_value=False):
            self.assertEqual(manager.snapshot()["status"], "failed")

    def test_stop_uses_private_token_and_releases_owned_process(self) -> None:
        manager = HostingManager(self.config_path)
        process = FakeProcess()
        with (
            patch("autotrainer.hosting_service._available_port"),
            patch("autotrainer.hosting_service.subprocess.Popen", return_value=process),
        ):
            manager.start(adapter="base", port=9876)

        with patch("autotrainer.hosting_service._json_request", return_value={"status": "stopping"}) as request:
            result = manager.stop()

        self.assertEqual(result["status"], "stopped")
        self.assertNotIn("control_token", result)
        self.assertEqual(request.call_args.args[0], "http://127.0.0.1:9876/_autotrainer/shutdown")
        self.assertGreaterEqual(len(request.call_args.kwargs["token"]), 32)
        self.assertEqual(process.returncode, 0)

    def test_test_completion_returns_endpoint_content(self) -> None:
        manager = HostingManager(self.config_path)
        live = {
            "status": "live",
            "endpoint": "http://127.0.0.1:9876",
            "model": "project-grpo",
        }
        completion = {
            "model": "project-grpo",
            "choices": [{"message": {"role": "assistant", "content": "It works."}}],
            "usage": {"total_tokens": 12},
        }
        with (
            patch.object(manager, "snapshot", return_value=live),
            patch("autotrainer.hosting_service._json_request", return_value=completion) as request,
        ):
            result = manager.test("Explain this repository.")

        self.assertEqual(result["content"], "It works.")
        self.assertEqual(result["usage"]["total_tokens"], 12)
        self.assertEqual(request.call_args.args[0], "http://127.0.0.1:9876/v1/chat/completions")

    def test_test_rejects_empty_or_oversized_prompts(self) -> None:
        manager = HostingManager(self.config_path)
        with self.assertRaises(HostingServiceError):
            manager.test("   ")
        with self.assertRaises(HostingServiceError):
            manager.test("x" * 8_001)


if __name__ == "__main__":
    unittest.main()
