from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE_ROOT / "src"))

from autotrainer.config import default_config, load_config, write_config  # noqa: E402
from autotrainer.evaluation import load_current_plan  # noqa: E402
from autotrainer.evaluation_service import (  # noqa: E402
    EvaluationJobManager,
    EvaluationServiceError,
)

# Reuse the independently tested held-out task and compiler-provenance fixtures.
from test_evaluation_workflow import (  # noqa: E402
    ORCHESTRATION,
    REVISION,
    _task,
    _write_compile_provenance,
)


def _project(root: Path) -> Path:
    config = default_config(revision=REVISION)
    config["project"]["artifact_dir"] = ".artifacts"
    config["grpo"]["dataset"] = ".artifacts/compiled/rl/train.jsonl"
    config["evaluation"]["dataset"] = ".artifacts/compiled/rl/evaluation.jsonl"
    config["evaluation"]["repetitions"] = 1
    config["evaluation"]["seeds"] = [1701]
    reference = config["evaluation"]["arms"]["reference_9b"]["model"]
    reference["id"] = "Qwen/Qwen3.5-9B"
    reference["revision"] = REVISION
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

    adapter_value = config["evaluation"]["arms"]["autotrainer"]["adapter"]["path"]
    adapter = (root / adapter_value).resolve()
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")

    compiled = root / ".artifacts" / "compiled" / "rl"
    compiled.mkdir(parents=True)
    (compiled / "evaluation.jsonl").write_text(
        "".join(json.dumps(_task(task_id)) + "\n" for task_id in ("checkout", "pricing")),
        encoding="utf-8",
    )
    (compiled / "train.jsonl").write_text(
        json.dumps(_task("training", split="train")) + "\n",
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
    callback(
        {
            "phase": "generating",
            "trial": trials[0],
            "completed": 0,
            "total": len(trials),
        }
    )
    scored_dir = run_dir / "scored-trials"
    scored_dir.mkdir(parents=True, exist_ok=True)
    for index, trial in enumerate(trials, start=1):
        value = {
            "schema_version": 1,
            "plan_id": plan["plan_id"],
            "trial_id": trial["trial_id"],
            "suite_id": suite_id,
            "candidate_id": trial["arm_id"],
            "task_id": trial["task_id"],
            "repetition": trial["repetition"],
            "seed": trial["seed"],
            "status": "completed",
            "hard_gate_passed": True,
            "gate_reason": "token=supersecret",
            "reward": 1.0,
            "components": {
                "design_rules": 1.0,
                "patch_quality": 1.0,
                "regression_safety": 1.0,
                "responsive_rules": 1.0,
                "task_tests": 1.0,
            },
            "metadata": {"evidence": {"patch": "must-not-reach-localhost"}},
        }
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
                self.assertEqual(fable["total"], 4)
                self.assertEqual(fable["results"], [])
                self.assertIn("no local run is being simulated", fable["message"])

                with self.assertRaisesRegex(EvaluationServiceError, "will not pretend Fable"):
                    manager.start("fable_ab")
                self.assertEqual(manager.snapshot()["status"], "idle")
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

                # A new backend process reads the same bounded record instead
                # of turning a completed job back into an in-memory fiction.
                restored = EvaluationJobManager(config_path)
                try:
                    self.assertEqual(restored.snapshot(), completed)
                finally:
                    restored.close()
            finally:
                manager.close()


if __name__ == "__main__":
    unittest.main()
