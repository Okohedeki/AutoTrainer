from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from threading import Event
from unittest.mock import patch


SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE_ROOT / "src"))

from autotrainer.config import default_config, load_config, write_config  # noqa: E402
from autotrainer.cli import main  # noqa: E402
from autotrainer.model_service import select_model  # noqa: E402
from autotrainer.project_gate import ProjectBusyError  # noqa: E402
from autotrainer.training_service import (  # noqa: E402
    TrainingJobManager,
    TrainingServiceError,
    run_project_training,
)


def _prepared(recipe: str, *, ready: bool = True) -> dict[str, object]:
    return {
        "status": "ready" if ready else "blocked",
        "recipe": recipe,
        "summary": "ready" if ready else "fix the source",
        "next_action": None if ready else {"detail": "fix the source"},
    }


class TrainingServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.config_path = self.root / "autotrainer.yaml"
        write_config(self.config_path, default_config(), overwrite=False)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_teach_runs_only_sft(self) -> None:
        sft_result = {"status": "completed", "stage": "sft"}
        with (
            patch("autotrainer.training_service.prepare_project", return_value=_prepared("teach")),
            patch("autotrainer.training_service.run_sft", return_value=sft_result) as run_sft,
            patch("autotrainer.training_service.run_grpo") as run_grpo,
        ):
            result = run_project_training(self.config_path)

        self.assertEqual(result, {"status": "completed", "recipe": "teach", "stages": [sft_result]})
        run_sft.assert_called_once()
        run_grpo.assert_not_called()
        stage_config = run_sft.call_args.args[0]
        self.assertTrue(stage_config["sft"]["enabled"])
        self.assertFalse(stage_config["grpo"]["enabled"])

    def test_practice_runs_only_verified_rl(self) -> None:
        rl_result = {"status": "completed", "stage": "grpo"}
        with (
            patch("autotrainer.training_service.prepare_project", return_value=_prepared("practice")),
            patch("autotrainer.training_service.run_sft") as run_sft,
            patch("autotrainer.training_service.run_grpo", return_value=rl_result) as run_grpo,
        ):
            result = run_project_training(self.config_path)

        self.assertEqual(result, {"status": "completed", "recipe": "practice", "stages": [rl_result]})
        run_sft.assert_not_called()
        run_grpo.assert_called_once()
        stage_config = run_grpo.call_args.args[0]
        self.assertFalse(stage_config["sft"]["enabled"])
        self.assertEqual(stage_config["grpo"]["start_from"], "base")

    def test_both_runs_sft_before_rl(self) -> None:
        order: list[str] = []
        with (
            patch("autotrainer.training_service.prepare_project", return_value=_prepared("both")),
            patch("autotrainer.training_service.run_sft", side_effect=lambda *args, **kwargs: order.append("sft") or {"stage": "sft"}),
            patch("autotrainer.training_service.run_grpo", side_effect=lambda *args, **kwargs: order.append("grpo") or {"stage": "grpo"}),
        ):
            result = run_project_training(self.config_path)

        self.assertEqual(order, ["sft", "grpo"])
        self.assertEqual(result["recipe"], "both")

    def test_practice_preserves_an_explicit_adapter_when_sft_is_disabled(self) -> None:
        config = default_config()
        config["sft"]["enabled"] = False
        config["grpo"]["start_from"] = "selected-adapter"
        write_config(self.config_path, config, overwrite=True)
        with (
            patch("autotrainer.training_service.prepare_project", return_value=_prepared("practice")),
            patch("autotrainer.training_service.run_grpo", return_value={"stage": "grpo"}) as run_grpo,
        ):
            run_project_training(self.config_path)

        self.assertEqual(
            run_grpo.call_args.args[0]["grpo"]["start_from"],
            "selected-adapter",
        )

    def test_blocked_preparation_starts_no_stage(self) -> None:
        with (
            patch("autotrainer.training_service.prepare_project", return_value=_prepared("needs_training_data", ready=False)),
            patch("autotrainer.training_service.run_sft") as run_sft,
            patch("autotrainer.training_service.run_grpo") as run_grpo,
            self.assertRaisesRegex(TrainingServiceError, "fix the source"),
        ):
            run_project_training(self.config_path)
        run_sft.assert_not_called()
        run_grpo.assert_not_called()

    def test_stage_rereads_the_configuration_after_prepare(self) -> None:
        """Stage selection uses the snapshot Prepare just finished producing."""

        prepared_rate = 0.000321

        def prepare(path: Path) -> dict[str, object]:
            config = load_config(path)
            config.data["sft"]["learning_rate"] = prepared_rate
            write_config(config.path, config.data, overwrite=True)
            return _prepared("teach")

        with (
            patch("autotrainer.training_service.prepare_project", side_effect=prepare),
            patch(
                "autotrainer.training_service.run_sft",
                return_value={"stage": "sft"},
            ) as run_sft,
        ):
            run_project_training(self.config_path)

        self.assertEqual(
            run_sft.call_args.args[0]["sft"]["learning_rate"], prepared_rate
        )

    def test_job_manager_serializes_jobs_and_reaches_completion(self) -> None:
        manager = TrainingJobManager(self.config_path)
        secret = "hf_this_must_never_reach_the_job_record"
        stage_result = {
            "status": "completed",
            "stage": "sft",
            "output_dir": str(self.root / ".autotrainer" / "adapters" / "sft"),
            "metrics": {"train_loss": 0.25, "note": secret},
            "dependencies": {"HF_TOKEN": secret},
            "recipe": {"token": secret},
        }

        def run(_path: Path, *, on_progress: object) -> dict[str, object]:
            on_progress("sft", "Teaching from approved examples.")  # type: ignore[operator]
            time.sleep(0.03)
            return {"status": "completed", "recipe": "teach", "stages": [stage_result]}

        with patch("autotrainer.training_service.run_project_training", side_effect=run):
            queued = manager.start()
            # The worker may leave the queue before start() returns; both states
            # prove the same single job was accepted.
            self.assertIn(queued["status"], {"queued", "running"})
            with self.assertRaisesRegex(TrainingServiceError, "already running"):
                manager.start()
            deadline = time.monotonic() + 2
            while manager.snapshot()["status"] not in {"completed", "failed"}:
                self.assertLess(time.monotonic(), deadline)
                time.sleep(0.01)
        manager.close()

        completed = manager.snapshot()
        self.assertEqual(completed["status"], "completed")
        self.assertEqual(completed["recipe"], "teach")
        self.assertEqual(completed["stage"], "sft")
        self.assertEqual(
            completed["result"],
            {
                "status": "completed",
                "recipe": "teach",
                "stages": [
                    {
                        "status": "completed",
                        "stage": "sft",
                        "output_dir": str(
                            self.root / ".autotrainer" / "adapters" / "sft"
                        ),
                        "metrics": {"train_loss": 0.25},
                    }
                ],
            },
        )

        record_text = manager.record_path.read_text(encoding="utf-8")
        record = json.loads(record_text)
        self.assertEqual(record, {"schema_version": 1, "job": completed})
        self.assertNotIn(secret, record_text)
        self.assertEqual(list(manager.record_path.parent.glob("*.tmp")), [])

        # A new backend process restores the terminal result instead of hiding
        # the output as the old in-memory-only manager did.
        restored = TrainingJobManager(self.config_path)
        self.assertEqual(restored.snapshot(), completed)

    def test_queued_job_reserves_project_before_worker_execution(self) -> None:
        """The API cannot expose a mutable queued window before its worker."""

        leases: list[object] = []

        class DeferredThread:
            daemon = False

            def __init__(self, *, args: tuple[object, ...], **_values: object) -> None:
                leases.append(args[1])

            def start(self) -> None:
                return None

            def is_alive(self) -> bool:
                return False

        manager = TrainingJobManager(self.config_path)
        with patch("autotrainer.training_service.Thread", DeferredThread):
            queued = manager.start()
        try:
            self.assertEqual(queued["status"], "queued")
            with self.assertRaisesRegex(ProjectBusyError, "project is busy"):
                select_model(self.config_path, "qwen3.5-9b-text")
        finally:
            leases[0].release()  # type: ignore[attr-defined]

    def test_live_saved_job_becomes_interrupted_and_a_retry_can_start(self) -> None:
        record_path = self.root / ".autotrainer" / "training" / "current-job.json"
        record_path.parent.mkdir(parents=True)
        for status in ("queued", "running"):
            with self.subTest(status=status):
                record_path.write_text(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "job": {
                                "id": "a" * 32,
                                "status": status,
                                "recipe": None,
                                "stage": "prepare",
                                "message": "Old backend owned this thread.",
                                "result": None,
                            },
                        }
                    ),
                    encoding="utf-8",
                )
                manager = TrainingJobManager(self.config_path)
                interrupted = manager.snapshot()
                self.assertEqual(interrupted["status"], "interrupted")
                self.assertIn("backend stopped", interrupted["message"])
                self.assertEqual(
                    json.loads(record_path.read_text(encoding="utf-8"))["job"],
                    interrupted,
                )

        with patch(
            "autotrainer.training_service.run_project_training",
            return_value={"status": "completed", "recipe": "teach", "stages": []},
        ):
            accepted = manager.start()
            self.assertNotEqual(accepted["id"], "a" * 32)
            manager.close()
        self.assertEqual(manager.snapshot()["status"], "completed")

    def test_worker_is_non_daemon_and_close_waits_for_it(self) -> None:
        manager = TrainingJobManager(self.config_path)
        entered = Event()
        release = Event()

        def run(_path: Path, *, on_progress: object) -> dict[str, object]:
            entered.set()
            release.wait(timeout=2)
            return {"status": "completed", "recipe": "teach", "stages": []}

        try:
            with patch("autotrainer.training_service.run_project_training", side_effect=run):
                manager.start()
                self.assertTrue(entered.wait(timeout=1))
                self.assertIsNotNone(manager._worker)
                self.assertFalse(manager._worker.daemon)  # type: ignore[union-attr]
                release.set()
                manager.close()
        finally:
            release.set()
        self.assertEqual(manager.snapshot()["status"], "completed")

    def test_agent_cli_auto_uses_the_same_training_pipeline(self) -> None:
        completed = {"status": "completed", "recipe": "teach", "stages": []}
        output = StringIO()
        with (
            patch("autotrainer.training_service.run_project_training", return_value=completed) as run,
            redirect_stdout(output),
        ):
            exit_code = main(["train", "auto", "--config", str(self.config_path), "--json"])

        self.assertEqual(exit_code, 0)
        self.assertIn('"recipe": "teach"', output.getvalue())
        run.assert_called_once_with(self.config_path)


if __name__ == "__main__":
    unittest.main()
