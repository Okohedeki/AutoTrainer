from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch


SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE_ROOT / "src"))

from autotrainer.config import default_config, write_config  # noqa: E402
from autotrainer.model_download_service import (  # noqa: E402
    ModelDownloadError,
    ModelDownloadManager,
)


class ModelDownloadServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.config_path = self.root / "autotrainer.yaml"
        write_config(
            self.config_path,
            default_config(revision="c" * 40),
            overwrite=False,
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_background_download_transitions_to_completed_and_persists(self) -> None:
        completed = {
            "status": "downloaded",
            "model_id": "Qwen/Qwen3.5-9B",
            "revision": "c" * 40,
            "snapshot_path": str(self.root / "snapshot"),
            "file_count": 12,
            "logical_bytes": 34,
        }
        manager = ModelDownloadManager(self.config_path)
        with patch(
            "autotrainer.model_download_service.materialize_model_owned",
            return_value=completed,
        ):
            queued = manager.start()
            manager.close()

        self.assertEqual(queued["status"], "queued")
        snapshot = manager.snapshot()
        self.assertEqual(snapshot["status"], "completed")
        self.assertEqual(snapshot["result"]["file_count"], 12)
        self.assertEqual(ModelDownloadManager(self.config_path).snapshot(), snapshot)

    def test_second_click_is_rejected_while_download_owns_project(self) -> None:
        entered = threading.Event()
        release = threading.Event()

        def download(_path):
            entered.set()
            release.wait(timeout=3)
            return {"status": "downloaded"}

        manager = ModelDownloadManager(self.config_path)
        with patch(
            "autotrainer.model_download_service.materialize_model_owned",
            side_effect=download,
        ):
            manager.start()
            self.assertTrue(entered.wait(timeout=3))
            with self.assertRaisesRegex(ModelDownloadError, "already active"):
                manager.start()
            release.set()
            manager.close()

    def test_public_failure_is_terminal_and_does_not_store_exception_chain(self) -> None:
        manager = ModelDownloadManager(self.config_path)
        with patch(
            "autotrainer.model_download_service.materialize_model_owned",
            side_effect=RuntimeError("network unavailable"),
        ):
            manager.start()
            manager.close()

        snapshot = manager.snapshot()
        self.assertEqual(snapshot["status"], "failed")
        self.assertEqual(
            snapshot["message"],
            "Model download failed. Check disk space, network access, model access, "
            "and Hugging Face login, then retry.",
        )
        self.assertIsNone(snapshot["result"])

    def test_live_record_from_previous_backend_becomes_interrupted(self) -> None:
        record_path = self.root / ".autotrainer" / "models" / "download-job.json"
        record_path.parent.mkdir(parents=True)
        record_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "job": {
                        "id": "d" * 32,
                        "status": "downloading",
                        "message": "active",
                        "model_id": "Qwen/Qwen3.5-9B",
                        "revision": "c" * 40,
                        "result": None,
                    },
                }
            ),
            encoding="utf-8",
        )

        snapshot = ModelDownloadManager(self.config_path).snapshot()

        self.assertEqual(snapshot["status"], "interrupted")
        self.assertIn("Start it again", snapshot["message"])


if __name__ == "__main__":
    unittest.main()
