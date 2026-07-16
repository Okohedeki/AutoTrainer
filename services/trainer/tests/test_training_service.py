from __future__ import annotations

import sys
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch


SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE_ROOT / "src"))

from autotrainer.config import default_config, write_config  # noqa: E402
from autotrainer.cli import main  # noqa: E402
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

    def test_job_manager_serializes_jobs_and_reaches_completion(self) -> None:
        manager = TrainingJobManager()

        def run(_path: Path, *, on_progress: object) -> dict[str, object]:
            on_progress("sft", "Teaching from approved examples.")  # type: ignore[operator]
            time.sleep(0.03)
            return {"status": "completed", "recipe": "teach", "stages": []}

        with patch("autotrainer.training_service.run_project_training", side_effect=run):
            queued = manager.start(self.config_path)
            # The worker may leave the queue before start() returns; both states
            # prove the same single job was accepted.
            self.assertIn(queued["status"], {"queued", "running"})
            with self.assertRaisesRegex(TrainingServiceError, "already running"):
                manager.start(self.config_path)
            deadline = time.monotonic() + 2
            while manager.snapshot()["status"] not in {"completed", "failed"}:
                self.assertLess(time.monotonic(), deadline)
                time.sleep(0.01)

        completed = manager.snapshot()
        self.assertEqual(completed["status"], "completed")
        self.assertEqual(completed["recipe"], "teach")
        self.assertEqual(completed["stage"], "sft")

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
