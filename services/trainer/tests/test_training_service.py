from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
from contextlib import nullcontext, redirect_stdout
from io import StringIO
from pathlib import Path
from threading import Event
from unittest.mock import patch


SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE_ROOT / "src"))

from autotrainer.config import ConfigError, default_config, load_config, write_config  # noqa: E402
from autotrainer.cli import main  # noqa: E402
from autotrainer.device_gate import DeviceBusyError  # noqa: E402
from autotrainer.model_service import select_model  # noqa: E402
from autotrainer.project_gate import ProjectBusyError  # noqa: E402
from autotrainer.training_service import (  # noqa: E402
    TrainingJobManager,
    TrainingServiceError,
    prepare_managed_training,
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
        self.frozen_dataset = patch(
            "autotrainer.training_service.require_frozen_dataset",
            return_value={"status": "frozen"},
        ).start()
        self.addCleanup(self.frozen_dataset.stop)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_managed_readiness_selects_disposable_output_preflight(self) -> None:
        prepared = {"status": "ready", "recipe": "teach"}
        with patch(
            "autotrainer.training_service.prepare_project",
            return_value=prepared,
        ) as prepare:
            result = prepare_managed_training(self.config_path)

        self.assertEqual(result, prepared)
        prepare.assert_called_once_with(self.config_path, managed_readiness=True)

    def test_teach_runs_only_sft(self) -> None:
        sft_result = {"status": "completed", "stage": "sft"}
        events: list[dict[str, object]] = []
        with (
            patch("autotrainer.training_service.prepare_project", return_value=_prepared("teach")),
            patch("autotrainer.training_service.run_sft", return_value=sft_result) as run_sft,
            patch("autotrainer.training_service.run_grpo") as run_grpo,
        ):
            result = run_project_training(
                self.config_path,
                on_event=lambda event: events.append(dict(event)),
            )

        self.assertEqual(result, {"status": "completed", "recipe": "teach", "stages": [sft_result]})
        run_sft.assert_called_once()
        run_grpo.assert_not_called()
        stage_config = run_sft.call_args.args[0]
        self.assertTrue(stage_config["sft"]["enabled"])
        self.assertFalse(stage_config["grpo"]["enabled"])
        self.assertEqual(
            events,
            [
                {"type": "stage_started", "stage": "prepare"},
                {"type": "stage_completed", "stage": "prepare"},
                {"type": "stage_started", "stage": "sft"},
                {"type": "stage_completed", "stage": "sft"},
            ],
        )

    def test_unfrozen_dataset_starts_no_preparation_or_gpu_stage(self) -> None:
        self.frozen_dataset.side_effect = ConfigError(
            "freeze and inspect the local dataset before training"
        )
        with (
            patch("autotrainer.training_service.prepare_project") as prepare,
            patch("autotrainer.training_service.run_sft") as run_sft,
            patch("autotrainer.training_service.run_grpo") as run_grpo,
            self.assertRaisesRegex(TrainingServiceError, "freeze and inspect"),
        ):
            run_project_training(self.config_path)

        prepare.assert_not_called()
        run_sft.assert_not_called()
        run_grpo.assert_not_called()

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

    def test_practice_events_bind_to_the_compiled_curriculum(self) -> None:
        events: list[dict[str, object]] = []
        digest = "a" * 64
        fingerprint = "b" * 64
        with (
            patch("autotrainer.training_service.prepare_project", return_value=_prepared("practice")),
            patch("autotrainer.training_service.run_grpo", return_value={"stage": "grpo"}),
            patch(
                "autotrainer.curriculum_service.load_compiled_catalog",
                return_value={
                    "status": "compiled",
                    "fingerprint": fingerprint,
                    "dataset_sha256": digest,
                    "blockers": [],
                },
            ),
        ):
            run_project_training(
                self.config_path,
                on_event=lambda event: events.append(dict(event)),
            )

        self.assertEqual(
            events[1],
            {
                "type": "stage_completed",
                "stage": "prepare",
                "catalog_fingerprint": fingerprint,
                "dataset_sha256": digest,
            },
        )

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

    def test_start_rejects_an_unvalidated_budget_before_claiming_the_gpu_or_outputs(
        self,
    ) -> None:
        payload = default_config()
        payload["refinement"]["vram"] = {
            "max_gib": 5,
            "enforcement": "hard",
        }
        original_outputs = {
            "sft": payload["sft"]["output_dir"],
            "grpo": payload["grpo"]["output_dir"],
            "start_from": payload["grpo"]["start_from"],
        }
        write_config(self.config_path, payload, overwrite=True)
        manager = TrainingJobManager(self.config_path)

        with (
            patch("autotrainer.training_service.acquire_device_lease") as device_lease,
            patch("autotrainer.training_service._allocate_run_outputs") as allocate,
            self.assertRaisesRegex(TrainingServiceError, "requires at least 20 GiB"),
        ):
            manager.start()

        device_lease.assert_not_called()
        allocate.assert_not_called()
        self.assertEqual(manager.snapshot()["status"], "idle")
        unchanged = load_config(self.config_path).data
        self.assertEqual(unchanged["sft"]["output_dir"], original_outputs["sft"])
        self.assertEqual(unchanged["grpo"]["output_dir"], original_outputs["grpo"])
        self.assertEqual(unchanged["grpo"]["start_from"], original_outputs["start_from"])

    def test_start_rejects_out_of_range_or_nonfinite_legacy_budgets_early(self) -> None:
        for limit in (193, float("inf")):
            with self.subTest(limit=limit):
                payload = default_config()
                payload["refinement"]["vram"]["max_gib"] = limit
                write_config(self.config_path, payload, overwrite=True)
                manager = TrainingJobManager(self.config_path)

                with (
                    patch("autotrainer.training_service.acquire_device_lease") as device_lease,
                    patch("autotrainer.training_service._allocate_run_outputs") as allocate,
                    self.assertRaisesRegex(
                        TrainingServiceError,
                        "finite number between 4 and 192",
                    ),
                ):
                    manager.start()

                device_lease.assert_not_called()
                allocate.assert_not_called()
                self.assertEqual(manager.snapshot()["status"], "idle")

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
            "performance": {
                "profile": {
                    "clock": "monotonic_wall_time",
                    "phase_seconds": {"training": 10.0, "bad phase": 99.0},
                    "total_seconds": 12.0,
                },
                "telemetry": {
                    "vram_peak_allocated_gib": 8.0,
                    "private_note": secret,
                },
                "receipt_path": str(self.root / "training_receipt.json"),
            },
        }

        def run(
            _path: Path, *, on_progress: object, on_event: object
        ) -> dict[str, object]:
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
                        "performance": {
                            "profile": {
                                "clock": "monotonic_wall_time",
                                "phase_seconds": {"training": 10.0},
                                "total_seconds": 12.0,
                            },
                            "telemetry": {"vram_peak_allocated_gib": 8.0},
                            "receipt_path": str(self.root / "training_receipt.json"),
                        },
                    }
                ],
            },
        )

        record_text = manager.record_path.read_text(encoding="utf-8")
        record = json.loads(record_text)
        self.assertEqual(record, {"schema_version": 1, "job": completed})
        self.assertNotIn(secret, record_text)
        self.assertEqual(list(manager.record_path.parent.glob("*.tmp")), [])
        run_record = (
            self.root
            / ".autotrainer"
            / "training"
            / "runs"
            / completed["id"]
            / "run.json"
        )
        self.assertEqual(json.loads(run_record.read_text(encoding="utf-8")), record)

        # A new backend process restores the terminal result instead of hiding
        # the output as the old in-memory-only manager did.
        restored = TrainingJobManager(self.config_path)
        self.assertEqual(restored.snapshot(), completed)

    def test_live_events_are_cursor_ordered_sanitized_and_restored(self) -> None:
        manager = TrainingJobManager(self.config_path)
        secret = "hf_event_secret_must_not_persist"

        def run(
            _path: Path, *, on_progress: object, on_event: object
        ) -> dict[str, object]:
            on_progress("grpo", "Practicing against verified tasks.")  # type: ignore[operator]
            on_event(  # type: ignore[operator]
                {
                    "type": "stage_completed",
                    "stage": "prepare",
                    "catalog_fingerprint": "a" * 64,
                    "dataset_sha256": "b" * 64,
                }
            )
            on_event({"type": "stage_started", "stage": "grpo"})  # type: ignore[operator]
            on_event(  # type: ignore[operator]
                {
                    "type": "calibration_round_started",
                    "stage": "grpo",
                    "round": 1,
                    "total_rounds": 2,
                    "prompt": secret,
                }
            )
            on_event(  # type: ignore[operator]
                {
                    "type": "trainer_log",
                    "stage": "grpo",
                    "step": 5,
                    "epoch": 0.5,
                    "metrics": {
                        "loss": 0.25,
                        "reward": 0.75,
                        "token_secret": 99,
                        "text": secret,
                    },
                }
            )
            on_event(  # type: ignore[operator]
                {
                    "type": "episode_started",
                    "episode_id": "0123456789ab",
                    "task_id": "pricing-task",
                    "task_family_id": "pricing-family",
                    "prompt": secret,
                }
            )
            on_event(  # type: ignore[operator]
                {
                    "type": "episode_scored",
                    "stage": "grpo",
                    "episode_id": "0123456789ab",
                    "task_id": "pricing-task",
                    "reward": 0.91,
                    "hard_gate_passed": True,
                    "gate_reason": None,
                    "tool_call_count": 7,
                    "tool_calls_by_name": {"read_file": 4, "apply_patch": 3, "shell": 99},
                    "patch_applied_count": 1,
                    "patch_rejections_by_reason": {
                        "context_mismatch": 2,
                        "secret_reason": 99,
                    },
                    "changed_file_count": 2,
                    "elapsed_seconds": 12.5,
                    "rubric": {
                        "design_rules": 0.8,
                        "patch_quality": 0.9,
                        "regression_safety": 1.0,
                        "responsive_rules": 0.75,
                        "task_tests": 1.0,
                    },
                    "patch": secret,
                }
            )
            on_event(  # type: ignore[operator]
                {
                    "type": "calibration_round_completed",
                    "stage": "grpo",
                    "round": 1,
                    "total_rounds": 2,
                    "rewards": secret,
                }
            )
            on_event({"type": "stage_completed", "stage": "grpo"})  # type: ignore[operator]
            return {"status": "completed", "recipe": "practice", "stages": []}

        with patch(
            "autotrainer.training_service.run_project_training", side_effect=run
        ):
            manager.start()
            manager.close()

        page = manager.events()
        self.assertEqual(
            [event["type"] for event in page["events"]],
            [
                "stage_completed",
                "stage_started",
                "calibration_round_started",
                "trainer_log",
                "episode_started",
                "episode_scored",
                "calibration_round_completed",
                "stage_completed",
                "job_completed",
            ],
        )
        self.assertEqual(
            [event["sequence"] for event in page["events"]],
            list(range(1, 10)),
        )
        self.assertEqual(page["events"][0]["catalog_fingerprint"], "a" * 64)
        calibration_started = page["events"][2]
        self.assertEqual(calibration_started["round"], 1)
        self.assertEqual(calibration_started["total_rounds"], 2)
        trainer_log = page["events"][3]
        self.assertEqual(trainer_log["metrics"], {"loss": 0.25, "reward": 0.75})
        episode = page["events"][5]
        self.assertEqual(episode["rubric"]["task_tests"], 1.0)
        self.assertEqual(episode["episode_id"], "0123456789ab")
        self.assertEqual(
            episode["tool_calls_by_name"],
            {"apply_patch": 3, "read_file": 4},
        )
        self.assertEqual(episode["changed_file_count"], 2)
        self.assertEqual(episode["patch_applied_count"], 1)
        self.assertEqual(
            episode["patch_rejections_by_reason"],
            {"context_mismatch": 2},
        )
        self.assertEqual(episode["elapsed_seconds"], 12.5)
        serialized = json.dumps(page)
        self.assertNotIn(secret, serialized)
        self.assertNotIn('"patch":', serialized)

        tail = manager.events(after=2)
        self.assertEqual([event["sequence"] for event in tail["events"]], [3, 4, 5, 6, 7, 8, 9])
        self.assertEqual(tail["cursor"], 9)
        self.assertFalse(tail["truncated"])

        restored = TrainingJobManager(self.config_path)
        self.assertEqual(restored.events(), page)

    def test_event_storage_is_bounded_and_reports_a_stale_cursor(self) -> None:
        manager = TrainingJobManager(self.config_path)
        with patch("autotrainer.training_service._EVENT_STORAGE_LIMIT", 3):
            with patch("autotrainer.training_service.Thread") as thread:
                thread.return_value.start.return_value = None
                thread.return_value.is_alive.return_value = False
                job = manager.start()
            try:
                for step in range(5):
                    manager._append_event(  # type: ignore[attr-defined]
                        job["id"],
                        {
                            "type": "trainer_log",
                            "stage": "sft",
                            "step": step,
                            "metrics": {"loss": 1.0 / (step + 1)},
                        },
                    )
            finally:
                thread.call_args.kwargs["args"][1].release()
                thread.call_args.kwargs["args"][2].release()

        page = manager.events(after=0)
        self.assertEqual(len(page["events"]), 3)
        self.assertEqual([event["sequence"] for event in page["events"]], [3, 4, 5])
        self.assertTrue(page["truncated"])

    def test_queued_job_reserves_project_before_worker_execution(self) -> None:
        """The API cannot expose a mutable queued window before its worker."""

        leases: list[object] = []

        class DeferredThread:
            daemon = False

            def __init__(self, *, args: tuple[object, ...], **_values: object) -> None:
                leases.extend((args[1], args[2]))

            def start(self) -> None:
                return None

            def is_alive(self) -> bool:
                return False

        manager = TrainingJobManager(self.config_path)
        with patch("autotrainer.training_service.Thread", DeferredThread):
            queued = manager.start()
        try:
            self.assertEqual(queued["status"], "queued")
            updated = load_config(self.config_path).data
            run_root = f".autotrainer/training/runs/{queued['id']}/checkpoints"
            self.assertEqual(updated["sft"]["output_dir"], f"{run_root}/sft")
            self.assertEqual(updated["grpo"]["output_dir"], f"{run_root}/grpo")
            self.assertEqual(updated["grpo"]["start_from"], f"{run_root}/sft")
            with self.assertRaisesRegex(ProjectBusyError, "project is busy"):
                select_model(self.config_path, "qwen3.5-9b-text")
        finally:
            leases[0].release()  # type: ignore[attr-defined]
            leases[1].release()  # type: ignore[attr-defined]

    def test_two_projects_cannot_queue_models_on_gpu_zero(self) -> None:
        """A project-specific lease alone must not allow a second 9B load."""

        second_root = self.root / "second-project"
        second_root.mkdir()
        second_config = second_root / "autotrainer.yaml"
        write_config(second_config, default_config(name="second"), overwrite=False)
        manager_one = TrainingJobManager(self.config_path)
        manager_two = TrainingJobManager(second_config)
        captured: list[tuple[object, ...]] = []

        class DeferredThread:
            daemon = False

            def __init__(self, *, args: tuple[object, ...], **_values: object) -> None:
                captured.append(args)

            def start(self) -> None:
                return None

            def is_alive(self) -> bool:
                return False

        with patch("autotrainer.training_service.Thread", DeferredThread):
            manager_one.start()
            try:
                with self.assertRaisesRegex(DeviceBusyError, "GPU 0 is already in use"):
                    manager_two.start()
            finally:
                # The deferred test worker never reaches its normal finally.
                captured[0][2].release()  # type: ignore[attr-defined]
                captured[0][1].release()  # type: ignore[attr-defined]

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
                run_record = (
                    self.root
                    / ".autotrainer"
                    / "training"
                    / "runs"
                    / interrupted["id"]
                    / "run.json"
                )
                self.assertEqual(
                    json.loads(run_record.read_text(encoding="utf-8"))["job"],
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

    def test_terminal_retry_allocates_fresh_immutable_checkpoint_paths(self) -> None:
        record_path = self.root / ".autotrainer" / "training" / "current-job.json"
        record_path.parent.mkdir(parents=True)
        record_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "job": {
                        "id": "a" * 32,
                        "status": "failed",
                        "recipe": "both",
                        "stage": "grpo",
                        "message": "old run stopped",
                        "result": None,
                    },
                }
            ),
            encoding="utf-8",
        )
        captured: list[tuple[object, ...]] = []

        class DeferredThread:
            daemon = False

            def __init__(self, *, args: tuple[object, ...], **_values: object) -> None:
                captured.append(args)

            def start(self) -> None:
                return None

            def is_alive(self) -> bool:
                return False

        manager = TrainingJobManager(self.config_path)
        with patch("autotrainer.training_service.Thread", DeferredThread):
            queued = manager.start()
        try:
            updated = load_config(self.config_path).data
            run_root = f".autotrainer/training/runs/{queued['id']}/checkpoints"
            self.assertEqual(updated["sft"]["output_dir"], f"{run_root}/sft")
            self.assertEqual(updated["grpo"]["output_dir"], f"{run_root}/grpo")
            self.assertEqual(updated["grpo"]["start_from"], f"{run_root}/sft")
        finally:
            captured[0][2].release()  # type: ignore[attr-defined]
            captured[0][1].release()  # type: ignore[attr-defined]

    def test_completed_runs_bind_the_candidate_and_preserve_run_history(self) -> None:
        config = default_config()
        evaluation = config["evaluation"]
        candidate = evaluation["arms"].pop("autotrainer")
        candidate["adapter"]["sha256"] = "a" * 64
        evaluation["arms"]["trained_candidate"] = candidate
        evaluation["candidates"] = [
            "trained_candidate" if arm == "autotrainer" else arm
            for arm in evaluation["candidates"]
        ]
        evaluation["suites"]["model_benchmark"]["arms"] = [
            "trained_candidate" if arm == "autotrainer" else arm
            for arm in evaluation["suites"]["model_benchmark"]["arms"]
        ]
        evaluation["decisions"]["model_benchmark"]["candidate"] = (
            "trained_candidate"
        )
        write_config(self.config_path, config, overwrite=True)

        plan_pointer = (
            self.root / ".autotrainer" / "evaluation" / "current-plan.json"
        )
        plan_pointer.parent.mkdir(parents=True, exist_ok=True)
        plan_pointer.write_text('{}\n', encoding="utf-8")
        manager = TrainingJobManager(self.config_path)

        with patch(
            "autotrainer.training_service.run_project_training",
            return_value={"status": "completed", "recipe": "teach", "stages": []},
        ):
            first = manager.start()
            manager.close()
        self.assertEqual(manager.snapshot()["status"], "completed")
        first_root = f".autotrainer/training/runs/{first['id']}"
        updated = load_config(self.config_path).data
        self.assertEqual(
            updated["evaluation"]["arms"]["trained_candidate"]["adapter"],
            {
                "path": f"{first_root}/checkpoints/sft",
                "stage": "sft",
            },
        )
        self.assertFalse(plan_pointer.exists())
        first_record = self.root / first_root / "run.json"
        self.assertEqual(
            json.loads(first_record.read_text(encoding="utf-8"))["job"]["status"],
            "completed",
        )

        plan_pointer.write_text('{}\n', encoding="utf-8")
        with patch(
            "autotrainer.training_service.run_project_training",
            return_value={
                "status": "completed",
                "recipe": "practice",
                "stages": [],
            },
        ):
            second = manager.start()
            manager.close()
        second_root = f".autotrainer/training/runs/{second['id']}"
        updated = load_config(self.config_path).data
        self.assertEqual(
            updated["evaluation"]["arms"]["trained_candidate"]["adapter"],
            {
                "path": f"{second_root}/checkpoints/grpo",
                "stage": "grpo",
            },
        )
        self.assertFalse(plan_pointer.exists())
        self.assertTrue(first_record.exists())
        self.assertTrue((self.root / second_root / "run.json").is_file())

    def test_worker_is_non_daemon_and_close_waits_for_it(self) -> None:
        manager = TrainingJobManager(self.config_path)
        entered = Event()
        release = Event()

        def run(
            _path: Path, *, on_progress: object, on_event: object
        ) -> dict[str, object]:
            del on_progress, on_event
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

    def test_agent_cli_auto_uses_the_same_durable_training_manager(self) -> None:
        completed = {"status": "completed", "recipe": "teach", "stages": []}
        output = StringIO()
        with (
            patch("autotrainer.training_service.run_project_training", return_value=completed) as run,
            redirect_stdout(output),
        ):
            exit_code = main(["train", "auto", "--config", str(self.config_path), "--json"])

        self.assertEqual(exit_code, 0)
        self.assertIn('"recipe": "teach"', output.getvalue())
        self.assertIn('"status": "completed"', output.getvalue())
        self.assertIn('"job": {', output.getvalue())
        run.assert_called_once()
        self.assertEqual(run.call_args.args, (self.config_path,))
        self.assertTrue(callable(run.call_args.kwargs["on_progress"]))
        self.assertTrue(callable(run.call_args.kwargs["on_event"]))

    def test_direct_cli_sft_owns_the_project_and_gpu(self) -> None:
        loaded = load_config(self.config_path)
        completed = {"status": "completed", "stage": "sft"}
        with (
            patch("autotrainer.cli.load_config", return_value=loaded),
            patch(
                "autotrainer.project_gate.project_run_gate",
                return_value=nullcontext(),
            ) as project_gate,
            patch(
                "autotrainer.device_gate.device_run_gate",
                return_value=nullcontext(),
            ) as device_gate,
            patch("autotrainer.training.run_sft", return_value=completed) as run_sft,
            redirect_stdout(StringIO()),
        ):
            exit_code = main(
                ["train", "sft", "--config", str(self.config_path), "--json"]
            )

        self.assertEqual(exit_code, 0)
        project_gate.assert_called_once_with(self.config_path)
        device_gate.assert_called_once_with()
        run_sft.assert_called_once()

    def test_direct_cli_dry_run_does_not_reserve_the_gpu(self) -> None:
        loaded = load_config(self.config_path)
        resolved = {"status": "dry_run", "dry_run": True, "recipe": {}}
        with (
            patch("autotrainer.cli.load_config", return_value=loaded),
            patch(
                "autotrainer.project_gate.project_run_gate",
                return_value=nullcontext(),
            ) as project_gate,
            patch("autotrainer.device_gate.device_run_gate") as device_gate,
            patch("autotrainer.training.run_grpo", return_value=resolved) as run_grpo,
            redirect_stdout(StringIO()),
        ):
            exit_code = main(
                [
                    "train",
                    "rl",
                    "--dry-run",
                    "--config",
                    str(self.config_path),
                    "--json",
                ]
            )

        self.assertEqual(exit_code, 0)
        project_gate.assert_called_once_with(self.config_path)
        device_gate.assert_not_called()
        run_grpo.assert_called_once()


if __name__ == "__main__":
    unittest.main()
