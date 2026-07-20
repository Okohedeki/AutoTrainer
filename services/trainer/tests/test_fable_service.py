from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE_ROOT / "src"))

from autotrainer.config import default_config, load_config, write_config  # noqa: E402
from autotrainer.fable_service import (  # noqa: E402
    _run_owned,
    _verified_receipt,
    FableServiceError,
    inspect_fable_workflow,
    pin_fable_runner,
)


class FableServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.config_path = self.root / "autotrainer.yaml"
        write_config(self.config_path, default_config(), overwrite=False)
        self.runtime = self.root / "fable-runtime"
        self.runtime.mkdir()
        (self.runtime / "orchestration.json").write_text(
            '{"tools":["read","patch"],"sampling":{"temperature":0.2}}\n',
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_pin_hashes_runtime_and_updates_external_runner(self) -> None:
        pointer = (
            load_config(self.config_path).artifact_dir
            / "evaluation"
            / "current-plan.json"
        )
        pointer.parent.mkdir(parents=True)
        pointer.write_text('{"plan_id":"stale","path":"stale"}\n', encoding="utf-8")
        result = pin_fable_runner(
            self.config_path,
            version="1.4.2",
            runtime_path=self.runtime,
        )

        self.assertEqual(result["status"], "in_progress")
        self.assertTrue(result["runner"]["pinned"])
        self.assertTrue(result["runner"]["receipt_matches"])
        self.assertEqual(result["runner"]["runtime_path"], str(self.runtime.resolve()))
        digest = result["runner"]["orchestration_sha256"]
        self.assertRegex(digest, r"^sha256:[0-9a-f]{64}$")
        runner = load_config(self.config_path).data["evaluation"]["suites"][
            "fable_ab"
        ]["runner"]
        self.assertEqual(runner["version"], "1.4.2")
        self.assertEqual(runner["orchestration_sha256"], digest)
        self.assertEqual(result["next_action"]["id"], "export")
        self.assertFalse(pointer.exists())

    def test_export_verification_rejects_runtime_bytes_that_changed_after_pin(self) -> None:
        pin_fable_runner(
            self.config_path,
            version="commit-abc123",
            runtime_path=self.runtime,
        )
        (self.runtime / "orchestration.json").write_text(
            '{"tools":["unsafe-new-tool"]}\n',
            encoding="utf-8",
        )

        with self.assertRaisesRegex(FableServiceError, "bytes changed"):
            _verified_receipt(load_config(self.config_path))

    def test_rejects_placeholder_version_and_missing_runtime(self) -> None:
        with self.assertRaisesRegex(FableServiceError, "concrete Fable"):
            pin_fable_runner(
                self.config_path,
                version="REPLACE_WITH_FABLE_VERSION",
                runtime_path=self.runtime,
            )
        with self.assertRaisesRegex(FableServiceError, "existing file or directory"):
            pin_fable_runner(
                self.config_path,
                version="1.0.0",
                runtime_path=self.root / "missing",
            )

    def test_default_project_reports_fable_as_optional(self) -> None:
        result = inspect_fable_workflow(self.config_path)

        self.assertEqual(result["status"], "optional")
        self.assertFalse(result["runner"]["pinned"])
        self.assertIsNone(result["next_action"])
        self.assertEqual(result["actions"][0]["id"], "export")
        self.assertEqual(result["actions"][0]["status"], "blocked")

    def test_export_freezes_plan_and_uses_managed_exchange_directory(self) -> None:
        config = SimpleNamespace(
            data={"evaluation": "fixture"},
            root=self.root,
            artifact_dir=self.root / ".autotrainer",
        )
        expected = {"plan_id": "plan-one", "request_count": 10}
        with (
            patch("autotrainer.fable_service.load_config", return_value=config),
            patch("autotrainer.fable_service._verified_receipt"),
            patch(
                "autotrainer.fable_service.write_evaluation_plan",
                return_value={"plan_id": "plan-one"},
            ) as freeze,
            patch(
                "autotrainer.fable_service.export_evaluation_suite",
                return_value=expected,
            ) as export,
        ):
            result = _run_owned(self.config_path, "export", None)

        self.assertEqual(result, expected)
        freeze.assert_called_once_with(config.data, config.root)
        export.assert_called_once_with(
            config.data,
            config.root,
            "fable_ab",
            config.artifact_dir
            / "evaluation"
            / "fable-exchange"
            / "plan-one"
            / "requests",
        )


if __name__ == "__main__":
    unittest.main()
