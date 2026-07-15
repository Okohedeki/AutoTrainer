from __future__ import annotations

import json
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE_ROOT / "src"))

from autotrainer.evaluation import (  # noqa: E402
    EvaluationError,
    build_evaluation_plan,
    build_evaluation_reports,
    export_blind_review,
    export_evaluation_suite,
    import_blind_reviews,
    ingest_evaluation_results,
    load_current_plan,
    write_evaluation_plan,
)
from autotrainer.cli import main as cli_main  # noqa: E402
from autotrainer.config import default_config, write_config  # noqa: E402
from autotrainer.packaging import PackagingError, build_adapter_package  # noqa: E402


REVISION = "a" * 40
ORCHESTRATION = "sha256:" + "b" * 64


def _task(task_id: str) -> dict:
    manifest = {
        "version": "1.0",
        "task": {
            "id": task_id,
            "instruction": f"Improve {task_id} without regressions.",
            "sourceId": f"source-{task_id}",
            "startingRevision": "locked",
            "split": "evaluation",
            "groupId": f"group-{task_id}",
        },
        "runtime": {
            "workingDirectory": ".",
            "build": "npm run build",
            "tests": "npm test",
            "browserTests": "npm run test:browser",
        },
        "limits": {
            "toolCalls": 20,
            "tokenBudget": 2048,
            "commandTimeoutSeconds": 120,
            "episodeTimeoutSeconds": 900,
            "networkAccess": False,
        },
        "tools": ["list_files", "read_file", "apply_patch", "run_check"],
        "verifier": {
            "bundle": "hidden/verifier",
            "command": "node /autotrainer-verifier/verify.mjs",
            "reportPath": ".autotrainer-report.json",
        },
        "rewards": {
            "buildGate": True,
            "regressionGate": True,
            "taskTests": 0.35,
            "regressionSafety": 0.20,
            "responsiveRules": 0.20,
            "designRules": 0.15,
            "patchQuality": 0.10,
        },
    }
    return {
        "task_id": task_id,
        "manifest": manifest,
        "source_path": f"sources/{task_id}",
        "source_revision": f"tree:{task_id}-locked",
        "environment_backend": "docker",
        "environment_image": "autotrainer/frontend-runtime:0.1",
    }


def _config(root: Path) -> dict:
    adapter = root / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "Qwen/Qwen3.5-9B"}),
        encoding="utf-8",
    )
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    dataset = root / ".artifacts" / "compiled" / "rl" / "evaluation.jsonl"
    dataset.parent.mkdir(parents=True)
    dataset.write_text(
        "".join(json.dumps(_task(task_id)) + "\n" for task_id in ("checkout", "pricing")),
        encoding="utf-8",
    )
    arms = {
        "reference_9b": {
            "label": "Declared 9B reference",
            "role": "reference",
            "parameter_class": "9b",
            "model": "project",
        },
        "base_fable": {
            "label": "Base 9B + Fable",
            "role": "control",
            "parameter_class": "9b",
            "model": "project",
        },
        "autotrainer": {
            "label": "AutoTrainer 9B",
            "role": "candidate",
            "parameter_class": "9b",
            "model": "project",
            "adapter": {"path": "adapter", "stage": "grpo"},
        },
    }
    suites = {
        "model_benchmark": {
            "kind": "model_benchmark",
            "arms": ["reference_9b", "autotrainer"],
            "runner": {
                "type": "external",
                "producer": "local-agent",
                "version": "1.0.0",
                "orchestration_sha256": ORCHESTRATION,
            },
        },
        "fable_ab": {
            "kind": "fable_ab",
            "arms": ["base_fable", "autotrainer"],
            "runner": {
                "type": "external",
                "producer": "fable",
                "version": "1.0.0",
                "orchestration_sha256": ORCHESTRATION,
            },
            "review": {
                "type": "manual",
                "blind": True,
                "reviewers_per_pair": 1,
                "viewports": ["375x812", "1440x900"],
            },
        },
    }
    return {
        "schema_version": 1,
        "project": {"name": "evaluation-test", "seed": 42, "artifact_dir": ".artifacts"},
        "model": {
            "provider": "huggingface",
            "id": "Qwen/Qwen3.5-9B",
            "revision": REVISION,
            "loader": "qwen3_5_text",
            "trust_remote_code": False,
        },
        "grpo": {"eval_dataset": ".artifacts/compiled/rl/evaluation.jsonl"},
        "environment": {
            "factory": "autotrainer.environments.frontend:FrontendEnvironment",
            "backend": "docker",
            "image": "autotrainer/frontend-runtime:0.1",
            "network": "none",
            "max_tool_output_chars": 12000,
            "episode_timeout_seconds": 900,
        },
        "evaluation": {
            "task_pack": "held-out",
            "task_split": "evaluation",
            "repetitions": 2,
            "seeds": [1701, 1702],
            "primary_metric": "verified_task_success",
            "candidates": list(arms),
            "arms": arms,
            "suites": suites,
            "fairness": {
                "paired_by": ["task_id", "repetition", "seed"],
                "same_task_snapshot": True,
                "same_instruction": True,
                "same_tools_and_limits": True,
                "same_verifier": True,
                "same_runner_within_suite": True,
                "same_sampling": True,
                "require_seed_control": True,
                "immutable_models_and_adapter": True,
                "randomize_arm_order": True,
                "failures_score_zero": True,
                "allow_unplanned_reruns": False,
            },
            "decisions": {
                "model_benchmark": {
                    "candidate": "autotrainer",
                    "control": "reference_9b",
                    "metric": "verified_task_success",
                    "minimum_delta": 0.0,
                    "minimum_tasks": 2,
                },
                "fable_ab": {
                    "candidate": "autotrainer",
                    "control": "base_fable",
                    "metric": "blind_preference_rate",
                    "minimum_rate": 0.5,
                    "minimum_tasks": 2,
                },
            },
        },
    }


def _result(plan: dict, trial: dict, directory: Path) -> Path:
    directory.mkdir(parents=True)
    patch = directory / "patch.diff"
    patch.write_text(f"arm={trial['arm_id']}\n", encoding="utf-8")
    review = directory / "website.html"
    review.write_text(f"<h1>{trial['arm_id']}</h1>", encoding="utf-8")
    arm = plan["arms"][trial["arm_id"]]
    runner = plan["suites"][trial["suite_id"]]["runner"]
    value = {
        "schema_version": 1,
        "plan_id": plan["plan_id"],
        "trial_id": trial["trial_id"],
        "suite_id": trial["suite_id"],
        "arm_id": trial["arm_id"],
        "task_id": trial["task_id"],
        "repetition": trial["repetition"],
        "seed": trial["seed"],
        "status": "completed",
        "producer": {
            "name": runner["producer"],
            "version": runner["version"],
            "orchestration_sha256": runner["orchestration_sha256"],
            "model_revision": arm["model"]["revision"],
            "adapter_sha256": arm["adapter"]["sha256"] if arm["adapter"] else None,
            "seed_honored": True,
            "fallback_models_used": False,
        },
        "usage": {
            "input_tokens": 10,
            "output_tokens": 20,
            "tool_calls": 2,
            "wall_time_seconds": 1.5,
        },
        "output": {
            "patch": patch.name,
            "review_artifact": review.name,
        },
    }
    result_path = directory / "result.json"
    result_path.write_text(json.dumps(value), encoding="utf-8")
    return result_path


def _scorer(_: dict, patch: str) -> dict:
    passed = "arm=autotrainer" in patch
    score = 1.0 if passed else 0.25
    return {
        "gated": False,
        "gate_reason": None,
        "reward": score,
        "signals": {
            "design_rules": score,
            "patch_quality": score,
            "regression_safety": 1.0,
            "responsive_rules": score,
            "task_tests": 1.0 if passed else 0.0,
        },
        "elapsed_seconds": 1.0,
    }


class EvaluationPlanTests(unittest.TestCase):
    def test_plan_is_deterministic_paired_and_sensitive_to_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = _config(root)
            first = build_evaluation_plan(config, root)
            second = build_evaluation_plan(config, root)
            self.assertEqual(first, second)
            self.assertEqual(len(first["trials"]), 16)
            self.assertEqual(len({trial["trial_id"] for trial in first["trials"]}), 16)
            changed = json.loads(json.dumps(config))
            changed["environment"]["image"] = "autotrainer/frontend-runtime:0.2"
            self.assertNotEqual(
                first["plan_id"], build_evaluation_plan(changed, root)["plan_id"]
            )

    def test_external_export_never_contains_hidden_verifier(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = _config(root)
            write_evaluation_plan(config, root)
            output = root / "export"
            manifest = export_evaluation_suite(config, root, "fable_ab", output)
            self.assertEqual(manifest["request_count"], 8)
            request = json.loads(Path(manifest["requests"][0]).read_text(encoding="utf-8"))
            self.assertNotIn("verifier", request["task"]["manifest"])
            self.assertNotIn("rewards", request["task"]["manifest"])

    def test_cli_writes_plan_and_exports_fable_requests(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = default_config(revision=REVISION)
            config["project"]["artifact_dir"] = ".artifacts"
            config["grpo"]["eval_dataset"] = ".artifacts/compiled/rl/evaluation.jsonl"
            config["evaluation"]["arms"]["reference_9b"]["model"]["id"] = "Qwen/Qwen3.5-9B"
            config["evaluation"]["arms"]["reference_9b"]["model"]["revision"] = REVISION
            for suite in config["evaluation"]["suites"].values():
                suite["runner"]["version"] = "1.0.0"
                suite["runner"]["orchestration_sha256"] = ORCHESTRATION
            config["evaluation"]["suites"]["model_benchmark"]["runner"]["argv"] = [
                "model-agent",
                "--request",
                "{request}",
                "--result",
                "{result}",
            ]
            adapter = root / ".autotrainer" / "checkpoints" / "grpo"
            adapter.mkdir(parents=True)
            (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
            (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
            dataset = root / ".artifacts" / "compiled" / "rl" / "evaluation.jsonl"
            dataset.parent.mkdir(parents=True)
            dataset.write_text(json.dumps(_task("cli-task")) + "\n", encoding="utf-8")
            config_path = write_config(root / "autotrainer.yaml", config)

            with redirect_stdout(io.StringIO()):
                plan_status = cli_main(
                    [
                        "evaluate",
                        "plan",
                        "--write",
                        "--config",
                        str(config_path),
                        "--json",
                    ]
                )
            self.assertEqual(plan_status, 0)
            export = root / "fable-export"
            with redirect_stdout(io.StringIO()):
                export_status = cli_main(
                    [
                        "benchmark",
                        "export",
                        "--suite",
                        "fable_ab",
                        "--output",
                        str(export),
                        "--config",
                        str(config_path),
                        "--json",
                    ]
                )
            self.assertEqual(export_status, 0)
            self.assertTrue((export / "export-manifest.json").is_file())


class EvaluationWorkflowTests(unittest.TestCase):
    def test_ingest_report_and_blind_review_complete_both_v1_decisions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = _config(root)
            write_evaluation_plan(config, root)
            plan, run_dir = load_current_plan(config, root)
            for index, trial in enumerate(plan["trials"]):
                result_path = _result(plan, trial, root / "incoming-results" / str(index))
                ingest_evaluation_results(
                    config,
                    root,
                    trial["suite_id"],
                    result_path,
                    scorer=_scorer,
                )

            review_export = export_blind_review(config, root, "fable_ab", root / "review")
            blind_map = json.loads(Path(review_export["sealed_map"]).read_text(encoding="utf-8"))
            review_rows = []
            for pair_id, mapping in blind_map["pairs"].items():
                choice = "left" if mapping["left"] == "autotrainer" else "right"
                review_rows.append(
                    {"pair_id": pair_id, "reviewer_id": "reviewer-1", "choice": choice}
                )
            reviews = root / "reviews.jsonl"
            reviews.write_text(
                "".join(json.dumps(row) + "\n" for row in review_rows),
                encoding="utf-8",
            )
            import_blind_reviews(config, root, "fable_ab", reviews)

            summary = build_evaluation_reports(config, root)
            self.assertTrue(summary["v1_success_criteria_verified"])
            self.assertTrue(
                summary["suites"]["model_benchmark"]["decision"]["verified_better"]
            )
            self.assertTrue(summary["suites"]["fable_ab"]["decision"]["verified_better"])
            self.assertTrue((run_dir / "reports" / "model_benchmark.md").is_file())
            package = build_adapter_package(
                config,
                root,
                output_dir=root / "package",
            )
            self.assertEqual(package["status"], "verified_winner")
            manifest = json.loads(Path(package["manifest"]).read_text(encoding="utf-8"))
            self.assertTrue(manifest["v1_success_criteria_verified"])
            self.assertEqual(manifest["candidate_id"], "autotrainer")

    def test_missing_trials_remain_failures_and_duplicate_ingest_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = _config(root)
            write_evaluation_plan(config, root)
            plan, _ = load_current_plan(config, root)
            trial = plan["trials"][0]
            result_path = _result(plan, trial, root / "result")
            ingest_evaluation_results(
                config, root, trial["suite_id"], result_path, scorer=_scorer
            )
            with self.assertRaisesRegex(EvaluationError, "duplicate result"):
                ingest_evaluation_results(
                    config, root, trial["suite_id"], result_path, scorer=_scorer
                )
            summary = build_evaluation_reports(config, root)
            suite = summary["suites"][trial["suite_id"]]
            self.assertLess(suite["completeness"]["rate"], 1.0)
            self.assertFalse(suite["decision"]["verified_better"])
            with self.assertRaisesRegex(PackagingError, "success criteria"):
                build_adapter_package(config, root, output_dir=root / "refused-package")


if __name__ == "__main__":
    unittest.main()
