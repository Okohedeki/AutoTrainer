from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

SERVICE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = SERVICE_ROOT.parents[1]
sys.path.insert(0, str(SERVICE_ROOT / "src"))

from autotrainer import (  # noqa: E402
    ManifestError,
    RolloutVerifierReport,
    TaskManifest,
    score_rollout,
)


class RewardTests(unittest.TestCase):
    def test_scores_verified_rollout(self) -> None:
        reward = score_rollout(RolloutVerifierReport(True, 1, 1, 0.75, 0.8, 0.9))
        self.assertFalse(reward.gated)
        self.assertEqual(reward.total, 0.91)

    def test_build_failure_is_hard_gated(self) -> None:
        reward = score_rollout(RolloutVerifierReport(False, 1, 1, 1, 1, 1))
        self.assertTrue(reward.gated)
        self.assertEqual(reward.gate_reason, "build_failed")
        self.assertEqual(reward.total, 0)

    def test_regression_is_hard_gated(self) -> None:
        reward = score_rollout(RolloutVerifierReport(True, 0.99, 1, 1, 1, 1))
        self.assertEqual(reward.gate_reason, "regression_failed")
        self.assertEqual(reward.total, 0)


class ManifestTests(unittest.TestCase):
    def load_example(self) -> dict:
        path = REPOSITORY_ROOT / "examples" / "tasks" / "responsive-pricing" / "task.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def test_loads_example_manifest(self) -> None:
        task = TaskManifest.from_mapping(self.load_example())
        self.assertEqual(task.task_id, "responsive-pricing-001")
        self.assertFalse(task.network_access)

    def test_rejects_network_access(self) -> None:
        payload = self.load_example()
        payload["limits"]["networkAccess"] = True
        with self.assertRaisesRegex(ManifestError, "disable network access"):
            TaskManifest.from_mapping(payload)


if __name__ == "__main__":
    unittest.main()
