from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import nullcontext, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch


SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE_ROOT / "src"))

from autotrainer.config import ConfigError, default_config, load_config, write_config  # noqa: E402
from autotrainer.cli import main as cli_main  # noqa: E402
from autotrainer.evaluation import (  # noqa: E402
    _producer_fairness,
    _seal_scored_result,
    load_current_plan,
)
from autotrainer.evaluation_service import (  # noqa: E402
    EvaluationJobManager,
    EvaluationServiceError,
    _readiness,
    _refresh_frozen_compiler_provenance,
    plan_project_evaluation,
    run_project_evaluation,
)

# Reuse the independently tested held-out task and compiler-provenance fixtures.
from test_evaluation_workflow import (  # noqa: E402
    EVALUATION_TASK_IDS,
    IMAGE_ID,
    ORCHESTRATION,
    REVISION,
    _task,
    _write_compile_provenance,
)


def _language_repository(root: Path, name: str, filename: str) -> tuple[Path, str]:
    repository = root / name
    repository.mkdir()
    subprocess.run(["git", "init", str(repository)], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.name", "AutoTrainer Tests"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.email", "tests@example.invalid"],
        check=True,
    )
    (repository / filename).write_text("# language fixture\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "commit", "-m", "fixture"],
        check=True,
        capture_output=True,
    )
    revision = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return repository, revision


def _project(root: Path, *, evaluation_filename: str = "held_out.py") -> Path:
    train_repository, train_revision = _language_repository(
        root, "language-train", "training.py"
    )
    evaluation_repository, evaluation_revision = _language_repository(
        root, "language-evaluation", evaluation_filename
    )
    config = default_config(revision=REVISION)
    config["project"]["artifact_dir"] = ".artifacts"
    config["grpo"]["dataset"] = ".artifacts/compiled/rl/train.jsonl"
    config["evaluation"]["dataset"] = ".artifacts/compiled/rl/evaluation.jsonl"
    config["evaluation"]["repetitions"] = 1
    config["evaluation"]["seeds"] = [1701]
    config["sft"]["enabled"] = False
    config["environment"]["image"] = IMAGE_ID
    reference = config["evaluation"]["arms"]["reference_9b"]["model"]
    reference["id"] = "Qwen/Qwen3.5-9B"
    reference["revision"] = REVISION
    config["evaluation"].pop("language", None)
    config["sources"] = [
        {
            "id": "language-train",
            "kind": "repository",
            "license": {"spdx": "MIT"},
            "partition": "train",
            "revision": train_revision,
            "roles": ["style"],
            "uri": str(train_repository),
        },
        {
            "id": "language-evaluation",
            "kind": "repository",
            "license": {"spdx": "MIT"},
            "partition": "evaluation",
            "revision": evaluation_revision,
            "roles": ["evaluation"],
            "uri": str(evaluation_repository),
        },
    ]
    config["evaluation"]["candidates"].insert(1, "base_fable")
    config["evaluation"]["arms"]["base_fable"] = {
        "label": "Base 9B + Fable",
        "role": "control",
        "parameter_class": "9b",
        "model": "project",
    }
    config["evaluation"]["suites"]["fable_ab"] = {
        "kind": "fable_ab",
        "arms": ["base_fable", "autotrainer"],
        "runner": {
            "type": "external",
            "producer": "fable",
            "version": "1.0.0",
            "orchestration_sha256": ORCHESTRATION,
            "result_schema": "autotrainer-evaluation-result-v1",
        },
        "review": {"type": "manual", "blind": True, "reviewers_per_pair": 1},
    }
    config["evaluation"]["decisions"]["fable_ab"] = {
        "candidate": "autotrainer",
        "control": "base_fable",
        "metric": "blind_preference_rate",
        "minimum_rate": 0.5,
        "minimum_tasks": 5,
    }
    fable_runner = config["evaluation"]["suites"]["fable_ab"]["runner"]
    fable_runner["version"] = "1.0.0"
    fable_runner["orchestration_sha256"] = ORCHESTRATION

    adapter_value = config["evaluation"]["arms"]["autotrainer"]["adapter"]["path"]
    adapter = (root / adapter_value).resolve()
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "Qwen/Qwen3.5-9B"}),
        encoding="utf-8",
    )
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    (adapter / "resolved_recipe.json").write_text(
        json.dumps(
            {
                "stage": "grpo",
                "model": {"id": "Qwen/Qwen3.5-9B", "revision": REVISION},
            }
        ),
        encoding="utf-8",
    )

    compiled = root / ".artifacts" / "compiled" / "rl"
    compiled.mkdir(parents=True)
    (compiled / "evaluation.jsonl").write_text(
        "".join(
            json.dumps(_task(task_id, root=root)) + "\n"
            for task_id in EVALUATION_TASK_IDS
        ),
        encoding="utf-8",
    )
    (compiled / "train.jsonl").write_text(
        json.dumps(_task("training", split="train", root=root)) + "\n",
        encoding="utf-8",
    )
    _write_compile_provenance(root)
    return write_config(root / "autotrainer.yaml", config)


def _write_scored_results(
    config_path: Path,
    suite_id: str,
    *,
    resume: bool = True,
    on_progress: object,
) -> dict:
    del resume
    config = load_config(config_path, check_paths=True)
    plan, run_dir = load_current_plan(config.data, config.root)
    trials = [trial for trial in plan["trials"] if trial["suite_id"] == suite_id]
    callback = on_progress
    scored_dir = run_dir / "scored-trials"
    scored_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = run_dir / "raw" / suite_id
    raw_dir.mkdir(parents=True, exist_ok=True)
    for index, trial in enumerate(trials, start=1):
        callback(
            {
                "phase": "generating",
                "trial": trial,
                "completed": index - 1,
                "total": len(trials),
            }
        )
        callback(
            {
                "phase": "verifying",
                "trial": trial,
                "completed": index - 1,
                "total": len(trials),
            }
        )
        arm = plan["arms"][trial["arm_id"]]
        runner = plan["suites"][suite_id]["runner"]
        raw = {
            "schema_version": 1,
            "plan_id": plan["plan_id"],
            "trial_id": trial["trial_id"],
            "suite_id": suite_id,
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
            "usage": {},
            "output": {},
        }
        unsealed = {
            "schema_version": 1,
            "plan_id": plan["plan_id"],
            "trial_id": trial["trial_id"],
            "suite_id": suite_id,
            "candidate_id": trial["arm_id"],
            "task_id": trial["task_id"],
            "repetition": trial["repetition"],
            "seed": trial["seed"],
            "status": "completed",
            "hard_gate_passed": False,
            "gate_reason": "token=supersecret",
            "reward": 0.0,
            "components": {
                "design_rules": 1.0,
                "patch_quality": 1.0,
                "regression_safety": 1.0,
                "responsive_rules": 1.0,
                "task_tests": 1.0,
            },
            "metadata": {
                "fairness": _producer_fairness(raw, trial, plan),
                "usage": {},
                "evidence": {},
            },
        }
        value = _seal_scored_result(unsealed, plan, raw)
        (raw_dir / f"{trial['trial_id']}.json").write_text(
            json.dumps(raw), encoding="utf-8"
        )
        (scored_dir / f"{trial['trial_id']}.json").write_text(
            json.dumps(value), encoding="utf-8"
        )
        callback(
            {
                "phase": "trial_completed",
                "trial": trial,
                "completed": index,
                "total": len(trials),
            }
        )
    callback(
        {
            "phase": "completed",
            "trial": None,
            "completed": len(trials),
            "total": len(trials),
        }
    )
    return {
        "plan_id": plan["plan_id"],
        "suite_id": suite_id,
        "completed": len(trials),
        "skipped": 0,
        "total": len(trials),
    }


class EvaluationServiceTests(unittest.TestCase):
    def test_missing_language_key_cannot_bypass_frontend_holdout_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = _project(
                Path(temporary_directory), evaluation_filename="build.mjs"
            )
            config = load_config(config_path, check_paths=True)
            self.assertNotIn("language", config.data["evaluation"])

            with self.assertRaisesRegex(ConfigError, "do not contain Python code"):
                plan_project_evaluation(config_path)

    def test_frozen_evaluation_restores_only_matching_full_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = _project(Path(temporary_directory))
            config = load_config(config_path, check_paths=True)
            freeze_path = config.artifact_dir / "dataset" / "freeze.json"
            freeze_path.parent.mkdir(parents=True)
            freeze_path.write_text("{}\n", encoding="utf-8")
            receipt = {
                "artifact_sha256": {"rl_evaluation": "held-out-sha"},
                "compiler_fingerprint": "full-fingerprint",
                "counts": {"rl_evaluation": 5, "rl_train": 0, "sft_train": 0},
            }
            compiled = {
                "artifact_sha256": dict(receipt["artifact_sha256"]),
                "counts": dict(receipt["counts"]),
                "errors": [],
                "fingerprint": receipt["compiler_fingerprint"],
            }
            scan = {"errors": [], "sources": []}

            with patch(
                "autotrainer.evaluation_service.require_frozen_dataset",
                return_value=receipt,
            ), patch(
                "autotrainer.evaluation_service.scan_sources", return_value=scan
            ) as source_scan, patch(
                "autotrainer.evaluation_service.compile_data", return_value=compiled
            ) as compiler:
                _refresh_frozen_compiler_provenance(config)

            source_scan.assert_called_once_with(config.data, config.root, write=True)
            compiler.assert_called_once_with(config.data, config.root, scan)

            mismatched = {**compiled, "artifact_sha256": {"rl_evaluation": "changed"}}
            with patch(
                "autotrainer.evaluation_service.require_frozen_dataset",
                return_value=receipt,
            ), patch(
                "autotrainer.evaluation_service.scan_sources", return_value=scan
            ), patch(
                "autotrainer.evaluation_service.compile_data", return_value=mismatched
            ), self.assertRaisesRegex(EvaluationServiceError, "does not match"):
                _refresh_frozen_compiler_provenance(config)

    def test_readiness_rejects_a_plan_for_an_old_project_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = _project(Path(temporary_directory))
            config = load_config(config_path, check_paths=True)
            plan_path = config.artifact_dir / "plan.json"
            plan_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "config_fingerprint": "sha256:" + "0" * 64,
                        "stages": {
                            "evaluation": {
                                "status": "blocked",
                                "blockers": ["obsolete runner blocker"],
                                "warnings": ["obsolete Fable warning"],
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            readiness = _readiness(config)

            self.assertEqual(readiness["status"], "stale")
            self.assertEqual(readiness["ready_task_count"], 0)
            self.assertEqual(
                readiness["blockers"],
                ["Project inputs changed since the last Prepare; check readiness again."],
            )
            self.assertEqual(readiness["warnings"], [])

    def test_external_suite_is_waiting_and_cannot_be_started_locally(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = _project(Path(temporary_directory))
            manager = EvaluationJobManager(config_path)
            try:
                workspace = manager.plan()
                fable = next(item for item in workspace["suites"] if item["id"] == "fable_ab")
                self.assertEqual(fable["runner_type"], "external")
                self.assertEqual(fable["phase"], "awaiting_external_results")
                self.assertEqual(fable["completed"], 0)
                self.assertEqual(fable["total"], 10)
                self.assertEqual(fable["trials"], [])
                self.assertEqual(fable["results"], [])
                self.assertIn("no local run is being simulated", fable["message"])

                with self.assertRaisesRegex(EvaluationServiceError, "will not pretend Fable"):
                    manager.start("fable_ab")
                self.assertEqual(manager.snapshot()["status"], "idle")
            finally:
                manager.close()

    def test_frozen_command_plan_exposes_only_sanitized_planned_trials(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = _project(Path(temporary_directory))
            manager = EvaluationJobManager(config_path)
            try:
                workspace = manager.plan()
                benchmark = next(
                    item for item in workspace["suites"] if item["id"] == "model_benchmark"
                )

                self.assertEqual(len(benchmark["trials"]), benchmark["total"])
                self.assertFalse(benchmark["trials_truncated"])
                self.assertEqual(
                    set(benchmark["trials"][0]),
                    {
                        "trial_id",
                        "task_id",
                        "arm_id",
                        "repetition",
                        "seed",
                        "status",
                    },
                )
                self.assertEqual(
                    {trial["status"] for trial in benchmark["trials"]},
                    {"planned"},
                )
                serialized = json.dumps(benchmark["trials"])
                self.assertNotIn("prompt", serialized)
                self.assertNotIn("path", serialized)

                # Workspace state follows only an observed manager boundary; it
                # never estimates partial progress inside model generation.
                current = benchmark["trials"][0]
                live_job = {
                    "id": "a" * 32,
                    "status": "running",
                    "plan_id": workspace["plan"]["plan_id"],
                    "suite": "model_benchmark",
                    "phase": "generating",
                    "message": "Generating.",
                    "completed": 0,
                    "total": benchmark["total"],
                    "current_trial": current,
                    "results": [],
                    "results_truncated": False,
                }
                for phase in ("generating", "verifying"):
                    live_job["phase"] = phase
                    with patch.object(manager, "snapshot", return_value=live_job):
                        live = manager.workspace()
                    live_benchmark = next(
                        item for item in live["suites"] if item["id"] == "model_benchmark"
                    )
                    states = {
                        trial["trial_id"]: trial["status"]
                        for trial in live_benchmark["trials"]
                    }
                    self.assertEqual(states[current["trial_id"]], phase)
                    self.assertEqual(list(states.values()).count(phase), 1)
            finally:
                manager.close()

    def test_command_job_persists_sanitized_observed_results(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = _project(Path(temporary_directory))
            manager = EvaluationJobManager(config_path)
            manager.plan()
            try:
                with patch(
                    "autotrainer.evaluation_service.run_project_evaluation",
                    side_effect=_write_scored_results,
                ):
                    queued = manager.start("model_benchmark")
                    self.assertEqual(queued["status"], "queued")
                    manager.close()

                completed = manager.snapshot()
                self.assertEqual(completed["status"], "completed")
                self.assertEqual(completed["phase"], "completed")
                self.assertEqual(completed["completed"], completed["total"])
                self.assertEqual(len(completed["results"]), completed["total"])
                serialized = json.dumps(completed)
                self.assertNotIn("supersecret", serialized)
                self.assertNotIn("must-not-reach-localhost", serialized)
                self.assertIn("[redacted]", serialized)

                event_page = manager.events(0)
                events = event_page["events"]
                phases = [event["phase"] for event in events]
                self.assertEqual(phases[0], "queued")
                self.assertEqual(phases[-1], "completed")
                self.assertEqual(phases.count("generating"), completed["total"])
                self.assertEqual(phases.count("verifying"), completed["total"])
                self.assertEqual(phases.count("trial_completed"), completed["total"])
                self.assertEqual(phases.count("completed"), 1)
                self.assertEqual(
                    [event["sequence"] for event in events],
                    list(range(1, len(events) + 1)),
                )
                completed_trials = [
                    event for event in events if event["phase"] == "trial_completed"
                ]
                self.assertTrue(all(event["result"] for event in completed_trials))
                self.assertTrue(all(event["rubric"] for event in completed_trials))
                self.assertTrue(
                    all(
                        set(event["rubric"]["components"]) == {
                            "design_rules",
                            "patch_quality",
                            "regression_safety",
                            "responsive_rules",
                            "task_tests",
                        }
                        for event in completed_trials
                    )
                )
                serialized_events = json.dumps(event_page)
                self.assertNotIn("supersecret", serialized_events)
                self.assertNotIn("must-not-reach-localhost", serialized_events)
                self.assertNotIn("prompt", serialized_events)
                self.assertNotIn("gate_reason", serialized_events)
                cursor = events[2]["sequence"]
                self.assertEqual(
                    manager.events(cursor)["events"],
                    [event for event in events if event["sequence"] > cursor],
                )

                completed_workspace = manager.workspace()
                benchmark = next(
                    item
                    for item in completed_workspace["suites"]
                    if item["id"] == "model_benchmark"
                )
                self.assertEqual(
                    {trial["status"] for trial in benchmark["trials"]},
                    {"completed"},
                )
                self.assertEqual(
                    {arm["role"] for arm in benchmark["arms"]},
                    {"candidate", "reference"},
                )
                report = benchmark["report"]
                self.assertIsNotNone(report)
                self.assertEqual(report["completeness"]["completed_trials"], completed["total"])
                self.assertEqual(
                    {item["candidate_id"] for item in report["comparison"]["candidates"]},
                    {"autotrainer", "reference_9b"},
                )
                self.assertEqual(report["decision"]["delta"], 0.0)
                self.assertEqual(
                    report["decision"]["confidence_interval"]["confidence"],
                    0.95,
                )
                fable = next(
                    item
                    for item in completed_workspace["suites"]
                    if item["id"] == "fable_ab"
                )
                self.assertIsNone(fable["report"])

                # A new backend process reads the same bounded record instead
                # of turning a completed job back into an in-memory fiction.
                config = load_config(config_path)
                _plan, run_dir = load_current_plan(config.data, config.root)
                report_path = run_dir / "reports" / "model_benchmark.json"
                forged_report = json.loads(report_path.read_text(encoding="utf-8"))
                forged_report["decision"]["delta"] = 1.0
                forged_report["decision"]["verified_better"] = True
                report_path.write_text(json.dumps(forged_report), encoding="utf-8")

                restored = EvaluationJobManager(config_path)
                try:
                    self.assertEqual(restored.snapshot(), completed)
                    self.assertEqual(restored.events(0), event_page)
                    restored_benchmark = next(
                        item
                        for item in restored.workspace()["suites"]
                        if item["id"] == "model_benchmark"
                    )
                    # The dashboard recomputes from sealed scores; it neither
                    # trusts nor rewrites the forged derived report on GET.
                    self.assertEqual(restored_benchmark["report"]["decision"]["delta"], 0.0)
                    self.assertFalse(
                        restored_benchmark["report"]["decision"]["verified_better"]
                    )
                    self.assertEqual(
                        json.loads(report_path.read_text(encoding="utf-8"))["decision"]["delta"],
                        1.0,
                    )
                finally:
                    restored.close()
            finally:
                manager.close()

    def test_workspace_rejects_a_locally_edited_sealed_score(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = _project(Path(temporary_directory))
            manager = EvaluationJobManager(config_path)
            manager.plan()
            try:
                _write_scored_results(
                    config_path,
                    "model_benchmark",
                    on_progress=lambda _event: None,
                )
                config = load_config(config_path)
                plan, run_dir = load_current_plan(config.data, config.root)
                trial = next(
                    trial
                    for trial in plan["trials"]
                    if trial["suite_id"] == "model_benchmark"
                )
                path = run_dir / "scored-trials" / f"{trial['trial_id']}.json"
                edited = json.loads(path.read_text(encoding="utf-8"))
                edited["reward"] = 0.5
                path.write_text(json.dumps(edited), encoding="utf-8")

                with self.assertRaisesRegex(
                    EvaluationServiceError, "content digest does not match"
                ):
                    manager.workspace()
            finally:
                manager.close()

    def test_synchronous_evaluation_uses_both_shared_run_gates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = _project(Path(temporary_directory))
            expected = {"suite_id": "model_benchmark", "total": 4}
            with (
                patch(
                    "autotrainer.evaluation_service.project_run_gate",
                    return_value=nullcontext(),
                ) as project_gate,
                patch(
                    "autotrainer.evaluation_service.device_run_gate",
                    return_value=nullcontext(),
                ) as device_gate,
                patch(
                    "autotrainer.evaluation_service.run_command_suite",
                    return_value=expected,
                ) as runner,
            ):
                result = run_project_evaluation(
                    config_path,
                    "model_benchmark",
                    resume=True,
                )

            self.assertEqual(result, expected)
            project_gate.assert_called_once_with(config_path)
            device_gate.assert_called_once_with()
            runner.assert_called_once()

    def test_agent_cli_runs_evaluation_through_the_durable_manager(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = _project(Path(temporary_directory))
            manager = MagicMock()
            manager.snapshot.return_value = {
                "id": "a" * 32,
                "status": "completed",
                "phase": "completed",
                "completed": 4,
                "total": 4,
            }
            with (
                patch(
                    "autotrainer.evaluation_service.EvaluationJobManager",
                    return_value=manager,
                ) as manager_type,
                redirect_stdout(StringIO()),
            ):
                exit_code = cli_main(
                    [
                        "evaluate",
                        "run",
                        "--suite",
                        "model_benchmark",
                        "--resume",
                        "--config",
                        str(config_path),
                        "--json",
                    ]
                )

            self.assertEqual(exit_code, 0)
            manager_type.assert_called_once_with(config_path.resolve())
            manager.start.assert_called_once_with("model_benchmark", resume=True)
            self.assertGreaterEqual(manager.close.call_count, 1)

    def test_failed_jobs_emit_a_redacted_durable_terminal_event(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = _project(Path(temporary_directory))
            manager = EvaluationJobManager(config_path)
            manager.plan()
            try:
                with patch(
                    "autotrainer.evaluation_service.run_project_evaluation",
                    side_effect=EvaluationServiceError("token=supersecret local failure"),
                ):
                    manager.start("model_benchmark")
                    manager.close()

                self.assertEqual(manager.snapshot()["status"], "failed")
                events = manager.events(0)["events"]
                self.assertEqual(events[-1]["phase"], "failed")
                self.assertIn("[redacted]", events[-1]["message"])
                self.assertNotIn("supersecret", json.dumps(events))
            finally:
                manager.close()

    def test_event_log_is_bounded_and_reports_a_stale_cursor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = _project(Path(temporary_directory))
            with patch("autotrainer.evaluation_service._EVENT_LIMIT", 5):
                manager = EvaluationJobManager(config_path)
                manager.plan()
                try:
                    with patch(
                        "autotrainer.evaluation_service.run_project_evaluation",
                        side_effect=_write_scored_results,
                    ):
                        manager.start("model_benchmark")
                        manager.close()

                    page = manager.events(0)
                    self.assertEqual(len(page["events"]), 5)
                    self.assertGreater(page["oldest_sequence"], 1)
                    self.assertTrue(page["cursor_reset"])
                    self.assertEqual(
                        page["events"][-1]["sequence"], page["latest_sequence"]
                    )

                    restored = EvaluationJobManager(config_path)
                    try:
                        self.assertEqual(restored.events(0), page)
                    finally:
                        restored.close()
                finally:
                    manager.close()


if __name__ == "__main__":
    unittest.main()
