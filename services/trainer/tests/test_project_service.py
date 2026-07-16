from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE_ROOT / "src"))

from autotrainer.cli import main  # noqa: E402
from autotrainer.config import default_config, write_config  # noqa: E402
from autotrainer.model_cache import ModelCacheError  # noqa: E402
from autotrainer.project_service import prepare_project  # noqa: E402


def scan_result(
    examples: int,
    tasks: int,
    *,
    approved_history: int = 0,
    pending_history: int = 0,
) -> dict:
    return {
        "errors": [],
        "warnings": [],
        "sources": [],
        "summary": {
            "valid_sft_record_count": examples,
            "train_ready_task_count": tasks,
            "approved_history_record_count": approved_history,
            "pending_history_review_count": pending_history,
        },
    }


def plan_result() -> dict:
    return {
        "status": "blocked",
        "errors": ["evaluation: runner pins remain placeholders"],
        "model": {"blockers": []},
        "evidence": {"blockers": []},
        "stages": {
            "sft": {"blockers": []},
            "grpo": {"blockers": []},
            "evaluation": {"blockers": ["runner pins remain placeholders"]},
        },
    }


READY_DOCTOR = {"sft_ready": True, "rl_ready": True}


class ProjectServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.config_path = self.root / "autotrainer.yaml"
        write_config(self.config_path, default_config(), overwrite=False)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def prepare_with(
        self,
        examples: int,
        tasks: int,
        doctor: dict | None = None,
        *,
        approved_history: int = 0,
        pending_history: int = 0,
        recipe_error: Exception | None = None,
        model_error: Exception | None = None,
    ) -> dict:
        full = scan_result(
            examples,
            tasks,
            approved_history=approved_history,
            pending_history=pending_history,
        )
        training = scan_result(
            examples,
            tasks,
            approved_history=approved_history,
            pending_history=pending_history,
        )
        model = {
            "id": "Qwen/Qwen3.5-9B",
            "revision": "a" * 40,
            "cache_dir": str((self.root / ".autotrainer" / "model-cache").resolve()),
        }
        with (
            patch("autotrainer.project_service.scan_sources", side_effect=[full, training]),
            patch(
                "autotrainer.project_service.compile_data",
                return_value={"errors": [], "warnings": [], "artifacts": {"sft": "train.jsonl"}},
            ),
            patch("autotrainer.project_service.build_plan", return_value=plan_result()),
            patch("autotrainer.project_service.run_doctor", return_value=doctor or READY_DOCTOR),
            patch(
                "autotrainer.project_service.resolve_sft_recipe",
                side_effect=recipe_error,
                return_value={"stage": "sft", "model": model},
            ),
            patch(
                "autotrainer.project_service.resolve_grpo_recipe",
                side_effect=recipe_error,
                return_value={
                    "stage": "grpo",
                    "model": model,
                    "environment": {
                        "factory": "autotrainer.environments.frontend:FrontendEnvironment"
                    },
                },
            ),
            patch("autotrainer.project_service.import_factory"),
            patch(
                "autotrainer.project_service.require_materialized_model",
                side_effect=model_error,
            ),
        ):
            return prepare_project(self.config_path)

    def test_recipe_recommendations_are_input_driven(self) -> None:
        cases = [
            (1, 1, "both", "ready"),
            (1, 0, "teach", "ready"),
            (0, 1, "practice", "ready"),
            (0, 0, "needs_training_data", "blocked"),
        ]
        for examples, tasks, recipe, status in cases:
            with self.subTest(recipe=recipe):
                result = self.prepare_with(examples, tasks)
                self.assertEqual(result["recipe"], recipe)
                self.assertEqual(result["status"], status)
                self.assertEqual(
                    [step["id"] for step in result["steps"]],
                    ["validate", "sources", "compile", "runtime"],
                )
                self.assertEqual(
                    set(result["details"]),
                    {"validation", "scan", "compile", "plan", "preflight", "doctor"},
                )

    def test_prepare_resolves_selected_recipe_and_exact_local_model(self) -> None:
        result = self.prepare_with(1, 0)

        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["details"]["preflight"]["status"], "ready")
        self.assertEqual(
            result["details"]["preflight"]["model"]["status"],
            "materialized",
        )

    def test_missing_model_snapshot_blocks_before_doctor_claims_ready(self) -> None:
        result = self.prepare_with(
            1,
            0,
            model_error=ModelCacheError("exact model snapshot is not complete"),
        )

        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["next_action"]["title"], "Download the base model")
        self.assertEqual(result["details"]["doctor"]["status"], "skipped")

    def test_unresolvable_selected_recipe_blocks_start(self) -> None:
        result = self.prepare_with(
            0,
            1,
            recipe_error=ValueError("grpo generation batch is invalid"),
        )

        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["next_action"]["title"], "Fix the training recipe")
        self.assertIn("generation batch", result["next_action"]["detail"])

    def test_evaluation_only_validation_and_plan_work_are_deferred(self) -> None:
        payload = default_config()
        payload["evaluation"]["task_pack"] = ""
        write_config(self.config_path, payload, overwrite=True)

        result = self.prepare_with(1, 0)

        self.assertEqual(result["status"], "ready")
        self.assertFalse(result["details"]["validation"]["valid"])
        self.assertTrue(result["details"]["validation"]["training_valid"])
        self.assertIn(
            "evaluation.task_pack is required",
            result["details"]["validation"]["later_proof"],
        )
        self.assertIsNone(result["next_action"])

    def test_approved_history_counts_as_teach_data(self) -> None:
        result = self.prepare_with(0, 0, approved_history=1)

        self.assertEqual(result["recipe"], "teach")
        self.assertEqual(result["status"], "ready")

    def test_pending_history_is_the_next_action_before_adding_data(self) -> None:
        result = self.prepare_with(0, 0, pending_history=2)

        self.assertEqual(result["recipe"], "needs_training_data")
        self.assertEqual(result["next_action"]["title"], "Review accepted changes")

    def test_source_failure_invalidates_prior_compiled_provenance(self) -> None:
        report = self.root / ".autotrainer" / "compiled" / "compile-report.json"
        report.parent.mkdir(parents=True)
        report.write_text('{"errors":[],"fingerprint":"previous-success"}\n', encoding="utf-8")
        failed_scan = {
            "errors": ["history: an approved review is stale"],
            "warnings": [],
            "sources": [],
            "summary": {},
        }
        with (
            patch("autotrainer.project_service.scan_sources", return_value=failed_scan),
            patch("autotrainer.project_service.build_plan", return_value=plan_result()),
            patch("autotrainer.project_service.run_doctor", return_value=READY_DOCTOR),
        ):
            result = prepare_project(self.config_path)

        invalidated = report.read_text(encoding="utf-8")
        self.assertEqual(result["next_action"]["title"], "Fix the first source")
        self.assertNotIn("previous-success", invalidated)
        self.assertIn("previous provenance was invalidated", invalidated)

    def test_returns_only_the_first_configuration_blocker(self) -> None:
        payload = default_config()
        payload["model"]["provider"] = "local"
        write_config(self.config_path, payload, overwrite=True)
        with (
            patch("autotrainer.project_service.scan_sources", return_value=scan_result(1, 0)),
            patch("autotrainer.project_service.build_plan", return_value=plan_result()),
            patch("autotrainer.project_service.run_doctor", return_value=READY_DOCTOR),
        ):
            result = prepare_project(self.config_path)

        self.assertEqual(result["steps"][0]["status"], "blocked")
        self.assertEqual(result["next_action"]["title"], "Fix project configuration")
        self.assertIsInstance(result["next_action"]["detail"], str)

    def test_runtime_is_the_last_blocker(self) -> None:
        result = self.prepare_with(1, 0, {"sft_ready": False, "rl_ready": False})

        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["steps"][-1]["status"], "blocked")
        self.assertEqual(result["next_action"]["title"], "Prepare the local runtime")

    def test_runtime_action_surfaces_doctors_first_concrete_blocker(self) -> None:
        doctor = {
            "sft_ready": False,
            "rl_ready": False,
            "python": {"status": "ready"},
            "gpu": {"status": "blocked", "detail": "CUDA is not available."},
            "packages": [],
        }

        result = self.prepare_with(1, 0, doctor)

        self.assertEqual(result["next_action"]["detail"], "CUDA is not available.")

    def test_cli_calls_the_shared_prepare_service(self) -> None:
        prepared = {
            "status": "ready",
            "recipe": "teach",
            "summary": "ready",
            "next_action": None,
            "steps": [],
            "details": {},
        }
        output = StringIO()
        with (
            patch("autotrainer.project_service.prepare_project", return_value=prepared),
            redirect_stdout(output),
        ):
            code = main(["prepare", "--config", str(self.config_path), "--json"])

        self.assertEqual(code, 0)
        self.assertIn('"recipe": "teach"', output.getvalue())


if __name__ == "__main__":
    unittest.main()
