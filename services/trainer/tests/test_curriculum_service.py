from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE_ROOT / "src"))

from autotrainer.config import default_config, write_config  # noqa: E402
from autotrainer.curriculum_service import get_curriculum_workspace  # noqa: E402


def _manifest(*, task_id: str, split: str, instruction: str) -> dict:
    return {
        "version": "1.0",
        "task": {
            "id": task_id,
            "instruction": instruction,
            "sourceId": "site",
            "startingRevision": "locked",
            "split": split,
            "groupId": "pricing-family",
        },
        "runtime": {
            "workingDirectory": ".",
            "install": "",
            "build": "npm run build",
            "tests": "npm test",
            "browserTests": "npm run test:browser",
        },
        "tools": ["list_files", "read_file", "search_code", "apply_patch", "run_check"],
        "verifier": {
            "bundle": "verifier",
            "command": "node /autotrainer-verifier/verify.mjs",
            "reportPath": ".autotrainer-verifier-report.json",
        },
        "rewards": {
            "buildGate": True,
            "regressionGate": True,
            "regressionSafety": 0.2,
            "taskTests": 0.35,
            "responsiveRules": 0.2,
            "designRules": 0.15,
            "patchQuality": 0.1,
        },
        "limits": {
            "toolCalls": 20,
            "commandTimeoutSeconds": 30,
            "episodeTimeoutSeconds": 300,
            "networkAccess": False,
        },
    }


class CurriculumServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.config_path = self.root / "autotrainer.yaml"
        self.train_manifest = _manifest(
            task_id="pricing-task",
            split="train",
            instruction="Repair the compiled pricing card.",
        )
        self.holdout_manifest = _manifest(
            task_id="pricing-holdout",
            split="evaluation",
            instruction="Repair the protected pricing card.",
        )
        train_dir = self.root / "tasks" / "train"
        evaluation_dir = self.root / "tasks" / "evaluation"
        train_dir.mkdir(parents=True)
        evaluation_dir.mkdir(parents=True)
        self.train_path = train_dir / "task.json"
        self.train_path.write_text(json.dumps(self.train_manifest), encoding="utf-8")
        (evaluation_dir / "task.json").write_text(
            json.dumps(self.holdout_manifest), encoding="utf-8"
        )

        config = default_config()
        config["sources"] = [
            {
                "id": "train-tasks",
                "kind": "task_pack",
                "uri": "tasks/train",
                "partition": "train",
            },
            {
                "id": "holdouts",
                "kind": "task_pack",
                "uri": "tasks/evaluation",
                "partition": "evaluation",
            },
        ]
        write_config(self.config_path, config, overwrite=False)

        self.dataset = self.root / config["grpo"]["dataset"]
        self.dataset.parent.mkdir(parents=True)
        row = {
            "task_id": "pricing-task",
            "manifest": self.train_manifest,
            "manifest_path": str(self.train_path),
            "source_revision": "1" * 40,
        }
        self.dataset.write_text(json.dumps(row) + "\n", encoding="utf-8")
        self.dataset_digest = hashlib.sha256(self.dataset.read_bytes()).hexdigest()
        self.fingerprint = "a" * 64
        self.report_path = self.root / ".autotrainer" / "compiled" / "compile-report.json"
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_report()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _write_report(self, *, digest: str | None = None, dataset: Path | None = None) -> None:
        report = {
            "schema_version": 1,
            "fingerprint": self.fingerprint,
            "artifacts": {"rl_train": str(dataset or self.dataset)},
            "artifact_sha256": {"rl_train": digest or self.dataset_digest},
            "errors": [],
        }
        self.report_path.write_text(json.dumps(report), encoding="utf-8")

    def _activity(self, rewards: list[float], *, fingerprint: str | None = None) -> dict:
        events: list[dict] = [
            {
                "type": "stage_completed",
                "stage": "prepare",
                "sequence": 1,
                "observed_at": "2026-07-17T00:00:00Z",
                "catalog_fingerprint": fingerprint or self.fingerprint,
                "dataset_sha256": self.dataset_digest,
            }
        ]
        for index, reward in enumerate(rewards, start=2):
            events.append(
                {
                    "type": "episode_scored",
                    "stage": "grpo",
                    "sequence": index,
                    "observed_at": f"2026-07-17T00:00:{index:02d}Z",
                    "episode_id": f"{index:012x}",
                    "task_id": "pricing-task",
                    "reward": reward,
                    "hard_gate_passed": reward > 0,
                    "gate_reason": None if reward > 0 else "tests_failed",
                    "tool_call_count": index,
                    "tool_calls_by_name": {"read_file": index - 1, "apply_patch": 1},
                    "changed_file_count": 1,
                    "elapsed_seconds": 10.0 + index,
                    "rubric": {
                        "design_rules": reward,
                        "patch_quality": reward,
                        "regression_safety": reward,
                        "responsive_rules": reward,
                        "task_tests": reward,
                    },
                }
            )
        return {
            "job_id": "b" * 32,
            "status": "running",
            "stage": "grpo",
            "events": events,
            "window": {
                "scope": "current_job_retained_window",
                "first_sequence": 1,
                "last_sequence": len(events),
                "retained_event_count": len(events),
                "observed_event_count": len(events),
                "truncated": False,
            },
        }

    def test_compiled_definition_wins_when_the_declared_file_changes(self) -> None:
        changed = json.loads(json.dumps(self.train_manifest))
        changed["task"]["instruction"] = "This draft changed after Prepare."
        self.train_path.write_text(json.dumps(changed), encoding="utf-8")

        result = get_curriculum_workspace(self.config_path)

        self.assertEqual(result["catalog"]["status"], "compiled")
        self.assertEqual(result["tasks"][0]["instruction"], "Repair the compiled pricing card.")
        self.assertEqual(result["tasks"][0]["declaration_state"], "changed_since_prepare")
        self.assertEqual(result["summary"]["protected_holdout_count"], 1)
        self.assertEqual({task["split"] for task in result["tasks"]}, {"train"})

    def test_dataset_sha_mismatch_fails_closed(self) -> None:
        self._write_report(digest="f" * 64)

        result = get_curriculum_workspace(self.config_path)

        self.assertEqual(result["catalog"]["status"], "blocked")
        self.assertEqual(result["summary"]["compiled_task_count"], 0)
        self.assertNotEqual(result["tasks"][0]["status"], "compiled")

    def test_matched_retained_rollouts_are_aggregated_without_empty_zeroes(self) -> None:
        empty = get_curriculum_workspace(self.config_path)
        self.assertIsNone(empty["tasks"][0]["observed"]["reward_mean"])
        self.assertIsNone(empty["summary"]["reward_mean"])

        result = get_curriculum_workspace(
            self.config_path,
            activity=self._activity([0.0, 0.25, 0.5, 0.75]),
        )

        self.assertEqual(result["run"]["catalog_alignment"], "matched")
        self.assertEqual(result["summary"]["rollout_count"], 4)
        self.assertEqual(result["tasks"][0]["observed"]["outcome_mix"], "varied")
        self.assertEqual(result["tasks"][0]["observed"]["gate_pattern"], "mixed")
        self.assertEqual(result["rollouts"][-1]["tool_calls_by_name"]["apply_patch"], 1)
        self.assertEqual(result["run"]["window"]["scope"], "current_job_retained_window")

    def test_catalog_mismatch_keeps_old_rollouts_unmatched(self) -> None:
        result = get_curriculum_workspace(
            self.config_path,
            activity=self._activity([0.5], fingerprint="c" * 64),
        )

        self.assertEqual(result["run"]["catalog_alignment"], "mismatch")
        self.assertEqual(result["summary"]["rollout_count"], 0)
        self.assertEqual(result["rollouts"], [])
        self.assertEqual(result["unmatched_observations"][0]["reason"], "catalog_mismatch")


if __name__ == "__main__":
    unittest.main()
