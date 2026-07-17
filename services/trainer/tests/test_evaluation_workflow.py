from __future__ import annotations

import json
import hashlib
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE_ROOT / "src"))

from autotrainer.evaluation import (  # noqa: E402
    EvaluationError,
    MAX_RESULT_ARTIFACT_BYTES,
    _build_trial_matrix,
    _copy_evidence,
    _digest,
    _paired_delta,
    _result_documents,
    _review_summary,
    build_evaluation_plan,
    build_evaluation_reports,
    export_blind_review,
    export_evaluation_suite,
    import_blind_reviews,
    ingest_evaluation_results,
    load_current_plan,
    run_command_suite,
    write_evaluation_plan,
)
from autotrainer.cli import main as cli_main  # noqa: E402
from autotrainer.compiler import compile_data  # noqa: E402
from autotrainer.config import default_config, write_config  # noqa: E402
from autotrainer.packaging import PackagingError, build_adapter_package  # noqa: E402


REVISION = "a" * 40
ORCHESTRATION = "sha256:" + "b" * 64
TRAIN_REPOSITORY_IDENTITY = "sha256:" + "1" * 64
EVALUATION_REPOSITORY_IDENTITY = "sha256:" + "2" * 64
TRAIN_SOURCE_REVISION = "c" * 40
EVALUATION_SOURCE_REVISION = "d" * 40


def _task(task_id: str, *, split: str = "evaluation") -> dict:
    source_id = "training-source" if split == "train" else f"source-{task_id}"
    manifest = {
        "version": "1.0",
        "task": {
            "id": task_id,
            "instruction": f"Improve {task_id} without regressions.",
            "sourceId": source_id,
            "startingRevision": "locked",
            "split": split,
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
        "source_repository_identity": (
            TRAIN_REPOSITORY_IDENTITY
            if split == "train"
            else EVALUATION_REPOSITORY_IDENTITY
        ),
        "source_revision": (
            TRAIN_SOURCE_REVISION if split == "train" else EVALUATION_SOURCE_REVISION
        ),
        "environment_backend": "docker",
        "environment_image": "autotrainer/frontend-runtime:0.1",
    }


def _write_compile_provenance(
    root: Path, *, exposures: list[dict[str, str]] | None = None
) -> Path:
    compiled = root / ".artifacts" / "compiled"
    training = compiled / "rl" / "train.jsonl"
    evaluation = compiled / "rl" / "evaluation.jsonl"
    report = {
        "schema_version": 1,
        "fingerprint": "fixture-compile-fingerprint",
        "errors": [],
        "artifacts": {
            "rl_train": str(training),
            "rl_evaluation": str(evaluation),
        },
        "artifact_sha256": {
            "rl_train": hashlib.sha256(training.read_bytes()).hexdigest(),
            "rl_evaluation": hashlib.sha256(evaluation.read_bytes()).hexdigest(),
        },
        "repository_exposures": exposures
        or [
            {
                "source_id": "training-source",
                "partition": "train",
                "repository_identity": TRAIN_REPOSITORY_IDENTITY,
                "commit": TRAIN_SOURCE_REVISION,
            },
            {
                "source_id": "evaluation-source",
                "partition": "evaluation",
                "repository_identity": EVALUATION_REPOSITORY_IDENTITY,
                "commit": EVALUATION_SOURCE_REVISION,
            },
        ],
    }
    destination = compiled / "compile-report.json"
    destination.write_text(json.dumps(report, sort_keys=True) + "\n", encoding="utf-8")
    return destination


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
    training_dataset = root / ".artifacts" / "compiled" / "rl" / "train.jsonl"
    training_dataset.write_text(
        json.dumps(_task("training", split="train")) + "\n", encoding="utf-8"
    )
    _write_compile_provenance(root)
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
        "grpo": {"dataset": ".artifacts/compiled/rl/train.jsonl"},
        "environment": {
            "factory": "autotrainer.environments.frontend:FrontendEnvironment",
            "backend": "docker",
            "image": "autotrainer/frontend-runtime:0.1",
            "network": "none",
            "max_tool_output_chars": 12000,
            "episode_timeout_seconds": 900,
        },
        "evaluation": {
            "dataset": ".artifacts/compiled/rl/evaluation.jsonl",
            "task_pack": "held-out",
            "task_split": "evaluation",
            "repetitions": 2,
            "seeds": [1701, 1702],
            "holdout_unit": "repository",
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
                "pair_position_policy": "deterministic_counterbalance",
                "execution_order_policy": "frozen_per_suite",
                "per_trial_arm_randomization": False,
                "failures_score_zero": True,
                "allow_unplanned_reruns": False,
            },
            "decisions": {
                "confidence": 0.95,
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
            self.assertEqual(first["holdout"]["unit"], "repository")
            self.assertEqual(
                first["holdout"]["training_repository_identities"],
                [TRAIN_REPOSITORY_IDENTITY],
            )
            self.assertEqual(len(first["trials"]), 16)
            self.assertEqual(len({trial["trial_id"] for trial in first["trials"]}), 16)
            changed = json.loads(json.dumps(config))
            changed["environment"]["image"] = "autotrainer/frontend-runtime:0.2"
            self.assertNotEqual(
                first["plan_id"], build_evaluation_plan(changed, root)["plan_id"]
            )

    def test_default_fable_placeholders_defer_only_the_external_suite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = _config(root)
            defaults = default_config()
            config["evaluation"]["suites"]["model_benchmark"]["runner"] = {
                "type": "builtin"
            }
            config["evaluation"]["suites"]["fable_ab"]["runner"] = dict(
                defaults["evaluation"]["suites"]["fable_ab"]["runner"]
            )

            plan = build_evaluation_plan(config, root)

            local_suite = plan["suites"]["model_benchmark"]
            fable_suite = plan["suites"]["fable_ab"]
            self.assertTrue(local_suite["runnable"])
            self.assertEqual(local_suite["execution_policy"]["type"], "grouped_by_arm")
            self.assertFalse(fable_suite["runnable"])
            self.assertEqual(fable_suite["runner"]["status"], "deferred")
            self.assertTrue(fable_suite["blockers"])
            self.assertEqual(
                {trial["suite_id"] for trial in plan["trials"]},
                {"model_benchmark"},
            )
            write_evaluation_plan(config, root)
            reports = build_evaluation_reports(config, root)
            self.assertEqual(reports["suites"]["fable_ab"]["status"], "deferred")
            self.assertFalse(reports["v1_success_criteria_verified"])

    def test_load_current_plan_rejects_task_and_trial_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = _config(root)
            written = write_evaluation_plan(config, root)
            plan_path = Path(written["artifact"])
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            plan["task_rows"]["checkout"]["manifest"]["task"][
                "instruction"
            ] = "tampered"
            plan_path.write_text(json.dumps(plan), encoding="utf-8")

            with self.assertRaisesRegex(EvaluationError, "content digest"):
                load_current_plan(config, root)

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = _config(root)
            written = write_evaluation_plan(config, root)
            plan_path = Path(written["artifact"])
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            plan["trials"][0]["sequence"] = 99
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            trials_path = plan_path.parent / "trials.jsonl"
            trials_path.write_text(
                "".join(json.dumps(item) + "\n" for item in plan["trials"]),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(EvaluationError, "canonical derivation"):
                load_current_plan(config, root)

    def test_load_current_plan_rebuilds_resolved_arms_after_a_rehashed_forgery(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = _config(root)
            original = write_evaluation_plan(config, root)
            forged = json.loads(Path(original["artifact"]).read_text(encoding="utf-8"))
            forged["arms"]["autotrainer"]["label"] = "forged candidate"
            identity = {
                key: value
                for key, value in forged.items()
                if key not in {"plan_id", "trials"}
            }
            forged["plan_id"] = f"sha256:{_digest(identity)}"
            task_map = {item["task_id"]: item for item in forged["tasks"]}
            forged["trials"] = _build_trial_matrix(
                forged["plan_id"],
                forged["suites"],
                task_map,
                forged["seeds"],
            )
            run_dir = (
                root
                / ".artifacts"
                / "evaluation"
                / forged["plan_id"].removeprefix("sha256:")
            )
            run_dir.mkdir(parents=True)
            forged_path = run_dir / "evaluation-plan.json"
            forged_path.write_text(json.dumps(forged), encoding="utf-8")
            (run_dir / "trials.jsonl").write_text(
                "".join(json.dumps(item) + "\n" for item in forged["trials"]),
                encoding="utf-8",
            )
            pointer = root / ".artifacts" / "evaluation" / "current-plan.json"
            pointer.write_text(
                json.dumps({"plan_id": forged["plan_id"], "path": str(forged_path)}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(EvaluationError, "resolved arms"):
                load_current_plan(config, root)

    def test_plan_rejects_repository_alias_across_compiled_splits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = _config(root)
            dataset = root / ".artifacts" / "compiled" / "rl" / "evaluation.jsonl"
            rows = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines()]
            for row in rows:
                row["source_repository_identity"] = TRAIN_REPOSITORY_IDENTITY
            dataset.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            _write_compile_provenance(root)

            with self.assertRaisesRegex(EvaluationError, "repository identity"):
                build_evaluation_plan(config, root)

    def test_plan_rejects_same_commit_from_differently_named_clones(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = _config(root)
            dataset = root / ".artifacts" / "compiled" / "rl" / "evaluation.jsonl"
            rows = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines()]
            for row in rows:
                row["source_revision"] = TRAIN_SOURCE_REVISION
            dataset.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            _write_compile_provenance(root)

            with self.assertRaisesRegex(EvaluationError, "exact source revision"):
                build_evaluation_plan(config, root)

    def test_plan_fails_closed_when_compiled_identity_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = _config(root)
            dataset = root / ".artifacts" / "compiled" / "rl" / "evaluation.jsonl"
            rows = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines()]
            rows[0].pop("source_repository_identity")
            dataset.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )

            with self.assertRaisesRegex(EvaluationError, "lacks source_repository_identity"):
                build_evaluation_plan(config, root)

    def test_direct_plan_rejects_final_dataset_as_grpo_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = _config(root)
            config["grpo"]["eval_dataset"] = config["evaluation"]["dataset"]

            with self.assertRaisesRegex(EvaluationError, "training validation"):
                build_evaluation_plan(config, root)

    def test_plan_rejects_style_exposure_absent_from_grpo_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = _config(root)
            _write_compile_provenance(
                root,
                exposures=[
                    {
                        "source_id": "training-task-source",
                        "partition": "train",
                        "repository_identity": TRAIN_REPOSITORY_IDENTITY,
                        "commit": TRAIN_SOURCE_REVISION,
                    },
                    {
                        "source_id": "style-only-training-source",
                        "partition": "train",
                        "repository_identity": EVALUATION_REPOSITORY_IDENTITY,
                        "commit": "e" * 40,
                    },
                    {
                        "source_id": "evaluation-source",
                        "partition": "evaluation",
                        "repository_identity": EVALUATION_REPOSITORY_IDENTITY,
                        "commit": EVALUATION_SOURCE_REVISION,
                    },
                ],
            )

            with self.assertRaisesRegex(
                EvaluationError, "compiler-frozen repository exposure"
            ):
                build_evaluation_plan(config, root)

    def test_failed_recompile_invalidates_prior_success_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = _config(root)
            report_path = root / ".artifacts" / "compiled" / "compile-report.json"
            self.assertFalse(json.loads(report_path.read_text(encoding="utf-8"))["errors"])

            failed = compile_data(
                config,
                root,
                {"errors": ["later source scan failed"], "sources": []},
            )

            on_disk = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(on_disk, failed)
            self.assertTrue(on_disk["errors"])
            # Dataset bytes may remain useful for diagnosis, but they cannot be
            # paired with provenance from the earlier successful compilation.
            with self.assertRaisesRegex(EvaluationError, "compilation errors"):
                build_evaluation_plan(config, root)

    def test_cli_invalidates_provenance_before_source_scan_exception(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = _config(root)
            config_path = write_config(root / "autotrainer.yaml", config)
            report_path = root / ".artifacts" / "compiled" / "compile-report.json"

            loaded = SimpleNamespace(data=config, root=root)
            with patch("autotrainer.cli.load_config", return_value=loaded):
                with patch(
                    "autotrainer.sources.scan_sources",
                    side_effect=PermissionError("simulated scan artifact failure"),
                ):
                    with self.assertRaisesRegex(PermissionError, "simulated scan"):
                        cli_main(["compile", "--config", str(config_path), "--json"])

            on_disk = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertTrue(on_disk["errors"])
            self.assertIn("previous provenance was invalidated", on_disk["errors"][0])
            with self.assertRaisesRegex(EvaluationError, "compilation errors"):
                build_evaluation_plan(config, root)

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
            self.assertIn(
                "relative to the result envelope",
                request["result_contract"]["completed_output"],
            )
            self.assertEqual(
                request["result_contract"]["directory_envelope_names"],
                ["result.json", "*.result.json"],
            )

    def test_cli_writes_plan_and_exports_fable_requests(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = default_config(revision=REVISION)
            config["project"]["artifact_dir"] = ".artifacts"
            config["grpo"]["dataset"] = ".artifacts/compiled/rl/train.jsonl"
            config["evaluation"]["dataset"] = ".artifacts/compiled/rl/evaluation.jsonl"
            config["evaluation"]["arms"]["reference_9b"]["model"]["id"] = "Qwen/Qwen3.5-9B"
            config["evaluation"]["arms"]["reference_9b"]["model"]["revision"] = REVISION
            fable_runner = config["evaluation"]["suites"]["fable_ab"]["runner"]
            fable_runner["version"] = "1.0.0"
            fable_runner["orchestration_sha256"] = ORCHESTRATION
            adapter = root / ".autotrainer" / "checkpoints" / "grpo"
            adapter.mkdir(parents=True)
            (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
            (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
            dataset = root / ".artifacts" / "compiled" / "rl" / "evaluation.jsonl"
            dataset.parent.mkdir(parents=True)
            dataset.write_text(json.dumps(_task("cli-task")) + "\n", encoding="utf-8")
            (dataset.parent / "train.jsonl").write_text(
                json.dumps(_task("training", split="train")) + "\n", encoding="utf-8"
            )
            _write_compile_provenance(root)
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
    def test_builtin_suite_preflights_then_loads_trials_in_arm_groups(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = _config(root)
            config["evaluation"]["suites"]["model_benchmark"]["runner"] = {
                "type": "builtin"
            }
            plan = write_evaluation_plan(config, root)
            preflighted: list[str] = []
            produced: list[str] = []
            closed: list[bool] = []

            class FakeProducer:
                def preflight(self, arm_ids: list[str]) -> None:
                    preflighted.extend(arm_ids)

                def produce(self, request: dict, result_path: Path) -> None:
                    arm_id = request["arm_id"]
                    produced.append(arm_id)
                    patch_path = result_path.parent / "patch.diff"
                    patch_path.write_text(f"arm={arm_id}\n", encoding="utf-8")
                    arm = plan["arms"][arm_id]
                    runner = plan["suites"]["model_benchmark"]["runner"]
                    result_path.write_text(
                        json.dumps(
                            {
                                "schema_version": 1,
                                "plan_id": request["plan_id"],
                                "trial_id": request["trial_id"],
                                "suite_id": request["suite_id"],
                                "arm_id": arm_id,
                                "task_id": request["task_id"],
                                "repetition": request["repetition"],
                                "seed": request["seed"],
                                "status": "completed",
                                "producer": {
                                    "name": runner["producer"],
                                    "version": runner["version"],
                                    "orchestration_sha256": runner[
                                        "orchestration_sha256"
                                    ],
                                    "model_revision": arm["model"]["revision"],
                                    "adapter_sha256": (
                                        arm["adapter"]["sha256"]
                                        if arm["adapter"]
                                        else None
                                    ),
                                    "seed_honored": True,
                                    "fallback_models_used": False,
                                },
                                "usage": {"tool_calls": 0},
                                "output": {"patch": patch_path.name},
                            }
                        ),
                        encoding="utf-8",
                    )

                def close(self) -> None:
                    closed.append(True)

            outcome = run_command_suite(
                config,
                root,
                "model_benchmark",
                scorer=_scorer,
                producer_factory=lambda _config, _root, _plan: FakeProducer(),
            )

            suite_arms = plan["suites"]["model_benchmark"]["arms"]
            self.assertFalse(plan["fairness"]["per_trial_arm_randomization"])
            self.assertEqual(
                plan["suites"]["model_benchmark"]["execution_policy"],
                outcome["execution_policy"],
            )
            self.assertEqual(outcome["execution_policy"]["type"], "grouped_by_arm")
            self.assertEqual(preflighted, suite_arms)
            self.assertEqual(
                produced,
                [suite_arms[0]] * 4 + [suite_arms[1]] * 4,
            )
            self.assertEqual(outcome["completed"], 8)
            self.assertEqual(closed, [True])

    def test_command_suite_reports_only_observed_trial_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = _config(root)
            config["evaluation"]["suites"]["model_benchmark"]["runner"] = {
                "type": "command",
                "producer": "local-agent",
                "version": "1.0.0",
                "orchestration_sha256": ORCHESTRATION,
                "argv": ["model-agent", "{request}", "{result}"],
            }
            plan = write_evaluation_plan(config, root)
            events: list[dict] = []

            def command(argv: list[str], **_: object) -> SimpleNamespace:
                request = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
                trial = next(
                    item for item in plan["trials"] if item["trial_id"] == request["trial_id"]
                )
                result_path = Path(argv[2])
                producer_dir = result_path.parent / "producer"
                produced = _result(plan, trial, producer_dir)
                for artifact in producer_dir.iterdir():
                    artifact.replace(result_path.parent / artifact.name)
                producer_dir.rmdir()
                self.assertEqual(produced.name, "result.json")
                return SimpleNamespace(stdout="runner output", stderr="", returncode=0)

            with patch("autotrainer.evaluation.subprocess.run", side_effect=command):
                outcome = run_command_suite(
                    config,
                    root,
                    "model_benchmark",
                    scorer=_scorer,
                    on_progress=events.append,
                )

            self.assertEqual(outcome["total"], 8)
            self.assertEqual(outcome["completed"], 8)
            self.assertEqual(events[0]["phase"], "queued")
            self.assertEqual(events[-1]["phase"], "completed")
            self.assertEqual(events[-1]["completed"], 8)
            self.assertEqual(
                [event["phase"] for event in events].count("generating"), 8
            )
            self.assertEqual(
                [event["phase"] for event in events].count("verifying"), 8
            )
            # Opaque generation exposes no fabricated fractional percentage;
            # progress advances only after a trusted scored artifact exists.
            completed = [event["completed"] for event in events]
            self.assertEqual(completed, sorted(completed))

    def test_resume_scores_the_existing_result_without_regenerating_it(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = _config(root)
            config["evaluation"]["suites"]["model_benchmark"]["runner"] = {
                "type": "command",
                "producer": "local-agent",
                "version": "1.0.0",
                "orchestration_sha256": ORCHESTRATION,
                "argv": ["model-agent", "{request}", "{result}"],
            }
            plan = write_evaluation_plan(config, root)
            generated: list[str] = []
            score_calls = 0
            scored_patches: list[str] = []

            def command(argv: list[str], **_: object) -> SimpleNamespace:
                request = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
                trial = next(
                    item for item in plan["trials"] if item["trial_id"] == request["trial_id"]
                )
                generated.append(trial["trial_id"])
                result_path = Path(argv[2])
                producer_dir = result_path.parent / "producer"
                _result(plan, trial, producer_dir)
                for artifact in producer_dir.iterdir():
                    artifact.replace(result_path.parent / artifact.name)
                producer_dir.rmdir()
                return SimpleNamespace(stdout="", stderr="", returncode=0)

            def fails_once(task: dict, patch_text: str) -> dict:
                nonlocal score_calls
                score_calls += 1
                scored_patches.append(patch_text)
                if score_calls == 1:
                    raise RuntimeError("trusted scorer temporarily unavailable")
                return _scorer(task, patch_text)

            with patch("autotrainer.evaluation.subprocess.run", side_effect=command):
                with self.assertRaisesRegex(RuntimeError, "temporarily unavailable"):
                    run_command_suite(
                        config,
                        root,
                        "model_benchmark",
                        scorer=fails_once,
                    )
                first_model_trial = next(
                    item
                    for item in plan["trials"]
                    if item["suite_id"] == "model_benchmark"
                )
                generated_patch = (
                    Path(plan["artifact"]).parent
                    / "incoming"
                    / first_model_trial["trial_id"]
                    / "patch.diff"
                )
                generated_patch.write_text("producer mutated its patch\n", encoding="utf-8")
                outcome = run_command_suite(
                    config,
                    root,
                    "model_benchmark",
                    resume=True,
                    scorer=fails_once,
                )

            first_trial = next(
                item["trial_id"]
                for item in plan["trials"]
                if item["suite_id"] == "model_benchmark"
            )
            self.assertEqual(generated.count(first_trial), 1)
            self.assertEqual(len(generated), outcome["total"])
            self.assertEqual(outcome["completed"], outcome["total"])
            self.assertEqual(score_calls, outcome["total"] + 1)
            self.assertEqual(scored_patches[0], scored_patches[1])
            self.assertNotIn("mutated", scored_patches[1])

    def test_resume_fails_closed_on_a_mismatched_existing_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = _config(root)
            config["evaluation"]["suites"]["model_benchmark"]["runner"] = {
                "type": "command",
                "producer": "local-agent",
                "version": "1.0.0",
                "orchestration_sha256": ORCHESTRATION,
                "argv": ["model-agent", "{request}", "{result}"],
            }
            plan = write_evaluation_plan(config, root)

            def command(argv: list[str], **_: object) -> SimpleNamespace:
                request = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
                trial = next(
                    item for item in plan["trials"] if item["trial_id"] == request["trial_id"]
                )
                result_path = Path(argv[2])
                producer_dir = result_path.parent / "producer"
                _result(plan, trial, producer_dir)
                for artifact in producer_dir.iterdir():
                    artifact.replace(result_path.parent / artifact.name)
                producer_dir.rmdir()
                return SimpleNamespace(stdout="", stderr="", returncode=0)

            command_patch = patch(
                "autotrainer.evaluation.subprocess.run", side_effect=command
            )
            with command_patch as command_mock:
                with self.assertRaisesRegex(RuntimeError, "scorer failed"):
                    run_command_suite(
                        config,
                        root,
                        "model_benchmark",
                        scorer=lambda _task, _patch: (_ for _ in ()).throw(
                            RuntimeError("scorer failed")
                        ),
                    )
                first = next(
                    item
                    for item in plan["trials"]
                    if item["suite_id"] == "model_benchmark"
                )
                result_path = (
                    Path(plan["artifact"]).parent
                    / "incoming"
                    / first["trial_id"]
                    / "result.json"
                )
                result = json.loads(result_path.read_text(encoding="utf-8"))
                result["arm_id"] = "base_fable"
                result_path.write_text(json.dumps(result), encoding="utf-8")

                with self.assertRaisesRegex(EvaluationError, "changed planned field arm_id"):
                    run_command_suite(
                        config,
                        root,
                        "model_benchmark",
                        resume=True,
                        scorer=_scorer,
                    )
                self.assertEqual(command_mock.call_count, 1)

    def test_evidence_copy_hashes_the_private_bounded_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "patch.diff"
            original = b"diff --git a/a b/a\n+verified\n"
            source.write_bytes(original)

            evidence = _copy_evidence(source, root / "evidence")
            source.write_bytes(b"producer changed its original after ingestion\n")

            stored = Path(evidence["path"])
            self.assertEqual(stored.read_bytes(), original)
            self.assertEqual(evidence["bytes"], len(original))
            self.assertEqual(evidence["sha256"], hashlib.sha256(original).hexdigest())
            self.assertEqual(hashlib.sha256(stored.read_bytes()).hexdigest(), evidence["sha256"])

            oversized = root / "oversized.bin"
            with oversized.open("wb") as handle:
                handle.seek(MAX_RESULT_ARTIFACT_BYTES)
                handle.write(b"x")
            with self.assertRaisesRegex(EvaluationError, "10 MiB"):
                _copy_evidence(oversized, root / "evidence")
            self.assertEqual(
                list((root / "evidence").glob(".incoming-evidence.*.tmp")),
                [],
            )

    def test_paired_bootstrap_uses_configurable_deterministic_quantiles(self) -> None:
        # Two repetitions per task create several distinct paired differences,
        # making the narrower interval observable instead of a metadata-only test.
        rates = {
            "minus-one": ((False, False), (True, True)),
            "minus-half": ((False, False), (True, False)),
            "zero": ((True, False), (True, False)),
            "plus-half": ((True, False), (False, False)),
            "plus-one": ((True, True), (False, False)),
        }
        runs = []
        for task_id, (candidate_results, control_results) in rates.items():
            for arm_id, results in (
                ("candidate", candidate_results),
                ("control", control_results),
            ):
                runs.extend(
                    {
                        "candidate_id": arm_id,
                        "task_id": task_id,
                        "hard_gate_passed": passed,
                    }
                    for passed in results
                )
        payload = {"runs": runs}

        interval_80 = _paired_delta(payload, "candidate", "control", "plan", 0.80)
        interval_95 = _paired_delta(payload, "candidate", "control", "plan", 0.95)

        self.assertEqual(
            interval_80,
            _paired_delta(payload, "candidate", "control", "plan", 0.80),
        )
        narrow = interval_80["confidence_interval"]
        wide = interval_95["confidence_interval"]
        self.assertEqual(narrow["confidence"], 0.8)
        self.assertEqual((narrow["lower_quantile"], narrow["upper_quantile"]), (0.1, 0.9))
        self.assertEqual(narrow["quantile_method"], "R-7 linear interpolation")
        self.assertGreaterEqual(narrow["low"], wide["low"])
        self.assertLessEqual(narrow["high"], wide["high"])
        self.assertNotEqual((narrow["low"], narrow["high"]), (wide["low"], wide["high"]))

    def test_result_ingest_rejects_unknown_and_mistyped_schema_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = _config(root)
            write_evaluation_plan(config, root)
            plan, _ = load_current_plan(config, root)
            trial = plan["trials"][0]

            unknown_path = _result(plan, trial, root / "unknown-result")
            unknown = json.loads(unknown_path.read_text(encoding="utf-8"))
            unknown["producer_score"] = 1.0
            unknown_path.write_text(json.dumps(unknown), encoding="utf-8")
            with self.assertRaisesRegex(EvaluationError, "unknown field"):
                ingest_evaluation_results(
                    config, root, trial["suite_id"], unknown_path, scorer=_scorer
                )

            mistyped_path = _result(plan, trial, root / "mistyped-result")
            mistyped = json.loads(mistyped_path.read_text(encoding="utf-8"))
            mistyped["usage"]["input_tokens"] = "ten"
            mistyped_path.write_text(json.dumps(mistyped), encoding="utf-8")
            with self.assertRaisesRegex(EvaluationError, "non-negative integer"):
                ingest_evaluation_results(
                    config, root, trial["suite_id"], mistyped_path, scorer=_scorer
                )

            malformed_enum_path = _result(plan, trial, root / "malformed-enum-result")
            malformed_enum = json.loads(malformed_enum_path.read_text(encoding="utf-8"))
            malformed_enum["status"] = []
            malformed_enum_path.write_text(json.dumps(malformed_enum), encoding="utf-8")
            with self.assertRaisesRegex(EvaluationError, "status must be a non-empty string"):
                ingest_evaluation_results(
                    config, root, trial["suite_id"], malformed_enum_path, scorer=_scorer
                )

    def test_directory_ingest_ignores_json_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = _config(root)
            write_evaluation_plan(config, root)
            plan, _ = load_current_plan(config, root)
            trial = plan["trials"][0]
            incoming = root / "incoming-result"
            result_path = _result(plan, trial, incoming)
            transcript = incoming / "transcript.json"
            transcript.write_text('{"events":[]}\n', encoding="utf-8")
            result = json.loads(result_path.read_text(encoding="utf-8"))
            result["output"]["transcript"] = transcript.name
            result_path.write_text(json.dumps(result), encoding="utf-8")

            ingested = ingest_evaluation_results(
                config, root, trial["suite_id"], incoming, scorer=_scorer
            )

            self.assertEqual(ingested["ingested_count"], 1)

    def test_review_import_rejects_unknown_and_mistyped_schema_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = _config(root)
            write_evaluation_plan(config, root)
            plan, _ = load_current_plan(config, root)
            for index, trial in enumerate(
                trial for trial in plan["trials"] if trial["suite_id"] == "fable_ab"
            ):
                ingest_evaluation_results(
                    config,
                    root,
                    "fable_ab",
                    _result(plan, trial, root / "fable-results" / str(index)),
                    scorer=_scorer,
                )
            exported = export_blind_review(config, root, "fable_ab", root / "review")
            blind_map = json.loads(Path(exported["sealed_map"]).read_text(encoding="utf-8"))
            pair_id = next(iter(blind_map["pairs"]))
            reviews = root / "reviews.jsonl"
            reviews.write_text(
                json.dumps(
                    {
                        "pair_id": pair_id,
                        "reviewer_id": "reviewer-1",
                        "choice": "left",
                        "arm_id": "autotrainer",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(EvaluationError, "unknown field"):
                import_blind_reviews(config, root, "fable_ab", reviews)

            reviews.write_text(
                json.dumps(
                    {"pair_id": pair_id, "reviewer_id": 1, "choice": "left"}
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(EvaluationError, "non-empty string"):
                import_blind_reviews(config, root, "fable_ab", reviews)

            reviews.write_text(
                json.dumps(
                    {"pair_id": pair_id, "reviewer_id": "reviewer-1", "choice": []}
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(EvaluationError, "choice must be a non-empty string"):
                import_blind_reviews(config, root, "fable_ab", reviews)

    def test_result_document_discovery_rejects_symlink_envelopes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            outside = root / "outside.json"
            outside.write_text("{}\n", encoding="utf-8")
            incoming = root / "incoming"
            incoming.mkdir()
            linked_result = incoming / "linked.result.json"
            try:
                linked_result.symlink_to(outside)
            except OSError as error:
                self.skipTest(f"file symlinks are unavailable: {error}")

            # Directory discovery must reject a reserved-name link rather than
            # following it or treating it as unrelated JSON evidence.
            with self.assertRaisesRegex(EvaluationError, "must not be a symlink"):
                _result_documents(incoming)
            # Explicit-file ingestion keeps arbitrary names, but not links.
            with self.assertRaisesRegex(EvaluationError, "must not be a symlink"):
                _result_documents(linked_result)

    def test_fable_reviews_penalize_both_fail_and_require_exact_reviewers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_dir = Path(temporary_directory)
            review_root = run_dir / "reviews" / "fable_ab"
            review_root.mkdir(parents=True)
            blind_map = {
                "pairs": {
                    "pair-one": {"left": "autotrainer", "right": "base_fable"},
                    "pair-two": {"left": "autotrainer", "right": "base_fable"},
                }
            }
            (review_root / "blind-map.json").write_text(
                json.dumps(blind_map), encoding="utf-8"
            )
            rows = [
                {"pair_id": "pair-one", "reviewer_id": "reviewer-1", "choice": "left"},
                {
                    "pair_id": "pair-two",
                    "reviewer_id": "reviewer-1",
                    "choice": "both_fail",
                },
            ]
            rows_path = review_root / "reviews.jsonl"
            rows_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            plan = {
                "suites": {"fable_ab": {"review": {"reviewers_per_pair": 1}}},
                "decisions": {
                    "fable_ab": {"candidate": "autotrainer", "control": "base_fable"}
                },
                "trials": [
                    {"suite_id": "fable_ab", "task_id": "task-one"},
                    {"suite_id": "fable_ab", "task_id": "task-two"},
                ],
            }

            summary = _review_summary(plan, run_dir, "fable_ab")
            self.assertEqual(summary["blind_preference_rate"], 0.5)
            self.assertTrue(summary["complete"])

            # A second reviewer on only one pair must invalidate verified
            # completeness instead of weighting that pair twice.
            rows.append(
                {"pair_id": "pair-one", "reviewer_id": "reviewer-2", "choice": "left"}
            )
            rows_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            summary = _review_summary(plan, run_dir, "fable_ab")
            self.assertFalse(summary["complete"])

    def test_fable_minimum_tasks_counts_unique_tasks_not_repetitions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = _config(root)
            dataset = root / ".artifacts" / "compiled" / "rl" / "evaluation.jsonl"
            only_task = dataset.read_text(encoding="utf-8").splitlines()[0]
            dataset.write_text(only_task + "\n", encoding="utf-8")
            _write_compile_provenance(root)
            write_evaluation_plan(config, root)
            plan, _ = load_current_plan(config, root)
            for index, trial in enumerate(plan["trials"]):
                ingest_evaluation_results(
                    config,
                    root,
                    trial["suite_id"],
                    _result(plan, trial, root / "one-task-results" / str(index)),
                    scorer=_scorer,
                )
            exported = export_blind_review(config, root, "fable_ab", root / "review")
            blind_map = json.loads(Path(exported["sealed_map"]).read_text(encoding="utf-8"))
            reviews = root / "reviews.jsonl"
            reviews.write_text(
                "".join(
                    json.dumps(
                        {
                            "pair_id": pair_id,
                            "reviewer_id": "reviewer-1",
                            "choice": (
                                "left"
                                if mapping["left"] == "autotrainer"
                                else "right"
                            ),
                        }
                    )
                    + "\n"
                    for pair_id, mapping in blind_map["pairs"].items()
                ),
                encoding="utf-8",
            )
            import_blind_reviews(config, root, "fable_ab", reviews)

            decision = build_evaluation_reports(config, root)["suites"]["fable_ab"][
                "decision"
            ]

            self.assertTrue(decision["observed_better"])
            self.assertEqual(decision["review"]["task_count"], 1)
            self.assertEqual(decision["review"]["pair_count"], 2)
            self.assertFalse(decision["verified_better"])

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
            interval = summary["suites"]["model_benchmark"]["decision"][
                "confidence_interval"
            ]
            self.assertEqual(interval["confidence"], 0.95)
            self.assertEqual(summary["metadata"]["model_benchmark_confidence"], 0.95)
            self.assertEqual(
                summary["suites"]["model_benchmark"]["metadata"][
                    "paired_bootstrap_confidence"
                ],
                0.95,
            )
            self.assertNotIn("metadata", summary["suites"]["fable_ab"])
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
