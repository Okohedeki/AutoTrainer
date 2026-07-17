from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE_ROOT / "src"))

from autotrainer.config import default_config, validate_mapping  # noqa: E402
from autotrainer.planner import build_plan  # noqa: E402


TRAIN_REPOSITORY_IDENTITY = "sha256:" + "1" * 64
EVALUATION_REPOSITORY_IDENTITY = "sha256:" + "2" * 64
TRAIN_COMMIT = "c" * 40
EVALUATION_COMMIT = "d" * 40


def evaluation_config() -> dict:
    payload = default_config(revision="a" * 40)
    reference = payload["evaluation"]["arms"]["reference_9b"]["model"]
    reference["id"] = "example/reference-9b"
    reference["revision"] = "b" * 40
    model_runner = payload["evaluation"]["suites"]["model_benchmark"]["runner"]
    # Most validation cases below exercise the extension point for a custom
    # command producer. New projects use the code-owned built-in runner.
    model_runner.clear()
    model_runner.update(
        {
            "type": "command",
            "producer": "local-agent",
            "version": "1.0.0",
            "orchestration_sha256": "sha256:" + "c" * 64,
            "argv": ["model-agent", "--request", "{request}", "--result", "{result}"],
        }
    )
    fable_runner = payload["evaluation"]["suites"]["fable_ab"]["runner"]
    fable_runner["version"] = "1.0.0"
    fable_runner["orchestration_sha256"] = "sha256:" + "d" * 64
    return payload


def evaluation_scan() -> dict:
    return {
        "errors": [],
        "warnings": [],
        "sources": [
            {
                "id": "training-repository-1",
                "kind": "repository",
                "partition": "train",
                "status": "ready",
                "eligible_file_count": 1,
                "repository_identity": TRAIN_REPOSITORY_IDENTITY,
                "commit": TRAIN_COMMIT,
            },
            {
                "id": "evaluation-repository-1",
                "kind": "repository",
                "partition": "evaluation",
                "status": "ready",
                "eligible_file_count": 1,
                "repository_identity": EVALUATION_REPOSITORY_IDENTITY,
                "commit": EVALUATION_COMMIT,
            },
            {
                "id": "held-out-frontend",
                "kind": "task_pack",
                "partition": "evaluation",
                "status": "ready",
                "tasks": [
                    {
                        "ready": True,
                        "split": "evaluation",
                        "task_id": "evaluation-task-1",
                        "snapshot_source_id": "evaluation-repository-1",
                    }
                ],
            }
        ],
    }


def write_compiled_evaluation(
    root: Path, relative: str = ".autotrainer/compiled/rl/evaluation.jsonl"
) -> Path:
    destination = root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({"task_id": "evaluation-task-1"}) + "\n", encoding="utf-8")
    return destination


def write_adapter(root: Path) -> Path:
    destination = root / ".autotrainer" / "checkpoints" / "grpo"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    (destination / "adapter_model.safetensors").write_bytes(b"adapter")
    return destination


class EvaluationConfigTests(unittest.TestCase):
    def test_default_declares_two_distinct_suites_and_three_roles(self) -> None:
        payload = default_config()
        report = validate_mapping(payload)

        self.assertEqual(report.errors, ())
        evaluation = payload["evaluation"]
        self.assertEqual(set(evaluation["suites"]), {"model_benchmark", "fable_ab"})
        self.assertEqual(
            {arm["role"] for arm in evaluation["arms"].values()},
            {"reference", "control", "candidate"},
        )
        self.assertEqual(set(evaluation["candidates"]), set(evaluation["arms"]))
        self.assertEqual(
            evaluation["suites"]["model_benchmark"]["runner"],
            {"type": "builtin"},
        )
        reference = evaluation["arms"]["reference_9b"]["model"]
        self.assertEqual(
            reference["id"],
            "empero-ai/Qwythos-9B-Claude-Mythos-5-1M",
        )
        self.assertEqual(reference["revision"], "14a29bae5143091aeaf87ad37120de4cd57d592c")

    def test_rejects_seed_repetition_mismatch(self) -> None:
        payload = evaluation_config()
        payload["evaluation"]["seeds"] = [1, 2]

        report = validate_mapping(payload)

        self.assertTrue(any("seeds length" in error for error in report.errors))

    def test_rejects_candidate_list_that_does_not_match_arms(self) -> None:
        payload = evaluation_config()
        payload["evaluation"]["candidates"][-1] = "unknown"

        report = validate_mapping(payload)

        self.assertTrue(any("declared arm ids" in error for error in report.errors))

    def test_rejects_mutable_reference_model(self) -> None:
        payload = evaluation_config()
        payload["evaluation"]["arms"]["reference_9b"]["model"]["revision"] = "main"

        report = validate_mapping(payload)

        self.assertTrue(any("immutable" in error for error in report.errors))

    def test_candidate_requires_a_stage_adapter(self) -> None:
        payload = evaluation_config()
        del payload["evaluation"]["arms"]["autotrainer"]["adapter"]

        report = validate_mapping(payload)

        self.assertTrue(any("adapter.path" in error for error in report.errors))
        self.assertTrue(any("adapter.stage" in error for error in report.errors))

    def test_rejects_unknown_suite_arm_and_unblinded_fable_review(self) -> None:
        payload = evaluation_config()
        payload["evaluation"]["suites"]["model_benchmark"]["arms"][1] = "unknown"
        payload["evaluation"]["suites"]["fable_ab"]["review"]["blind"] = False

        report = validate_mapping(payload)

        self.assertTrue(any("unknown arms" in error for error in report.errors))
        self.assertTrue(any("review.blind" in error for error in report.errors))

    def test_rejects_malformed_command_and_external_runners(self) -> None:
        payload = evaluation_config()
        payload["evaluation"]["suites"]["model_benchmark"]["runner"]["argv"] = []
        del payload["evaluation"]["suites"]["fable_ab"]["runner"]["result_schema"]

        report = validate_mapping(payload)

        self.assertTrue(any("runner.argv" in error for error in report.errors))
        self.assertTrue(any("runner.result_schema" in error for error in report.errors))

    def test_rejects_unpinned_runners_and_missing_command_placeholders(self) -> None:
        payload = evaluation_config()
        model_runner = payload["evaluation"]["suites"]["model_benchmark"]["runner"]
        model_runner["orchestration_sha256"] = "not-a-digest"
        model_runner["argv"] = ["model-agent"]
        del payload["evaluation"]["suites"]["fable_ab"]["runner"]["version"]

        report = validate_mapping(payload)

        self.assertTrue(any("orchestration_sha256" in error for error in report.errors))
        self.assertTrue(any("{request}" in error for error in report.errors))
        self.assertTrue(any("{result}" in error for error in report.errors))
        self.assertTrue(any("runner.version" in error for error in report.errors))

    def test_rejects_relaxed_fairness_and_mismatched_decisions(self) -> None:
        payload = evaluation_config()
        payload["evaluation"]["fairness"]["same_verifier"] = False
        payload["evaluation"]["decisions"]["fable_ab"]["control"] = "reference_9b"
        payload["evaluation"]["decisions"]["model_benchmark"]["metric"] = "build_rate"
        payload["evaluation"]["decisions"]["model_benchmark"]["minimum_tasks"] = 1

        report = validate_mapping(payload)

        self.assertTrue(any("fairness.same_verifier" in error for error in report.errors))
        self.assertTrue(any("fable_ab.control" in error for error in report.errors))
        self.assertTrue(any("model_benchmark.metric" in error for error in report.errors))
        self.assertTrue(any("model_benchmark.minimum_tasks" in error for error in report.errors))

    def test_rejects_training_validation_that_reuses_final_benchmark(self) -> None:
        payload = evaluation_config()
        payload["grpo"]["eval_dataset"] = ".autotrainer/data/../compiled/rl/evaluation.jsonl"

        report = validate_mapping(payload)

        self.assertTrue(
            any(
                "grpo.eval_dataset must be separate from evaluation.dataset" in error
                for error in report.errors
            )
        )


class EvaluationPlannerTests(unittest.TestCase):
    def test_uses_declared_final_dataset_instead_of_artifact_convention(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = evaluation_config()
            payload["evaluation"]["dataset"] = "benchmarks/final.jsonl"
            payload["grpo"]["eval_dataset"] = "training/grpo-validation.jsonl"
            final_dataset = write_compiled_evaluation(root, "benchmarks/final.jsonl")
            write_adapter(root)

            plan = build_plan(payload, root, evaluation_scan())

            evaluation = plan["stages"]["evaluation"]
            self.assertEqual(evaluation["status"], "inputs_ready")
            self.assertEqual(evaluation["compiled_dataset"], str(final_dataset))

    def test_runtime_inputs_are_ready_only_with_compiled_tasks_and_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = evaluation_config()
            write_compiled_evaluation(root)
            write_adapter(root)

            plan = build_plan(payload, root, evaluation_scan())

            evaluation = plan["stages"]["evaluation"]
            self.assertEqual(evaluation["status"], "inputs_ready")
            self.assertEqual(evaluation["ready_task_count"], 1)
            self.assertEqual(evaluation["repetitions"], 3)
            self.assertEqual(evaluation["suites"]["model_benchmark"]["pair_count"], 3)
            self.assertEqual(evaluation["suites"]["model_benchmark"]["arm_run_count"], 6)
            self.assertEqual(
                evaluation["suites"]["fable_ab"]["runner_status"],
                "awaiting_external_results",
            )
            self.assertFalse(
                any("repository holdout" in item for item in evaluation["blockers"])
            )

    def test_default_fable_placeholders_are_deferred_without_blocking_local_eval(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = evaluation_config()
            payload["evaluation"]["suites"]["fable_ab"]["runner"] = dict(
                default_config()["evaluation"]["suites"]["fable_ab"]["runner"]
            )
            write_compiled_evaluation(root)
            write_adapter(root)

            plan = build_plan(payload, root, evaluation_scan())

            evaluation = plan["stages"]["evaluation"]
            self.assertEqual(evaluation["status"], "inputs_ready")
            self.assertEqual(
                evaluation["suites"]["model_benchmark"]["runner_status"],
                "declared",
            )
            self.assertEqual(
                evaluation["suites"]["fable_ab"]["runner_status"],
                "deferred_configuration",
            )
            self.assertTrue(evaluation["suites"]["fable_ab"]["blockers"])
            self.assertFalse(
                any("fable_ab" in item for item in evaluation["blockers"])
            )

    def test_repository_holdout_blocks_source_aliases_and_exact_commit_clones(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = evaluation_config()
            write_compiled_evaluation(root)
            write_adapter(root)

            alias_scan = evaluation_scan()
            alias_scan["sources"][1]["repository_identity"] = TRAIN_REPOSITORY_IDENTITY
            alias_plan = build_plan(payload, root, alias_scan)
            self.assertTrue(
                any(
                    "share repository identity" in item
                    for item in alias_plan["stages"]["evaluation"]["blockers"]
                )
            )

            clone_scan = evaluation_scan()
            clone_scan["sources"][1]["commit"] = TRAIN_COMMIT
            clone_plan = build_plan(payload, root, clone_scan)
            self.assertTrue(
                any(
                    "share exact commit" in item
                    for item in clone_plan["stages"]["evaluation"]["blockers"]
                )
            )

    def test_repository_holdout_fails_closed_without_scanner_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = evaluation_config()
            write_compiled_evaluation(root)
            write_adapter(root)
            scan = evaluation_scan()
            del scan["sources"][1]["repository_identity"]

            plan = build_plan(payload, root, scan)

            self.assertTrue(
                any(
                    "lack repository_identity" in item
                    for item in plan["stages"]["evaluation"]["blockers"]
                )
            )

    def test_missing_compiled_evaluation_dataset_blocks_only_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = evaluation_config()
            payload["sft"]["enabled"] = False
            payload["grpo"]["enabled"] = False
            write_adapter(root)

            plan = build_plan(payload, root, evaluation_scan())

            self.assertEqual(plan["stages"]["sft"]["status"], "not_requested")
            self.assertEqual(plan["stages"]["grpo"]["status"], "not_requested")
            self.assertEqual(plan["stages"]["evaluation"]["status"], "blocked")
            self.assertTrue(
                any(
                    "compiled evaluation task dataset is missing" in blocker
                    for blocker in plan["stages"]["evaluation"]["blockers"]
                )
            )

    def test_missing_adapter_and_mutable_reference_pin_block_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = evaluation_config()
            payload["evaluation"]["arms"]["reference_9b"]["model"]["revision"] = "main"
            write_compiled_evaluation(root)

            plan = build_plan(payload, root, evaluation_scan())

            blockers = plan["stages"]["evaluation"]["blockers"]
            self.assertEqual(plan["stages"]["evaluation"]["status"], "blocked")
            self.assertTrue(any("candidate adapter does not exist" in item for item in blockers))
            self.assertTrue(any("immutable commit SHA" in item for item in blockers))

    def test_placeholder_runner_fingerprint_blocks_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = evaluation_config()
            payload["evaluation"]["suites"]["model_benchmark"]["runner"][
                "orchestration_sha256"
            ] = "sha256:" + "0" * 64
            write_compiled_evaluation(root)
            write_adapter(root)

            plan = build_plan(payload, root, evaluation_scan())

            blockers = plan["stages"]["evaluation"]["blockers"]
            self.assertEqual(plan["stages"]["evaluation"]["status"], "blocked")
            self.assertTrue(any("orchestration_sha256" in item for item in blockers))


if __name__ == "__main__":
    unittest.main()
