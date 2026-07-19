from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE_ROOT / "src"))

from autotrainer.training import (  # noqa: E402
    TrainingRuntimeError,
    run_grpo_environment_canary,
)


def check(*, configured: bool = True, passed: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        configured=configured,
        passed=passed,
        status="passed" if passed else "failed",
        stdout="",
        stderr="broken command" if not passed else "",
    )


class FakeEnvironment:
    def __init__(
        self,
        *,
        task_pass_rate: float = 0.0,
        failed_check: str | None = None,
        reset_error: Exception | None = None,
    ) -> None:
        self.task_pass_rate = task_pass_rate
        self.failed_check = failed_check
        self.reset_error = reset_error
        self.last_result = None

    def reset(self, **row: object) -> str:
        if self.reset_error is not None:
            raise self.reset_error
        return f"Task: {row['task_id']}"

    def get_reward(self) -> float:
        checks = {
            name: check(passed=name != self.failed_check)
            for name in ("install", "build", "tests", "verifier")
        }
        hard_gate = f"{self.failed_check}_failed" if self.failed_check else None
        self.last_result = SimpleNamespace(
            hard_gate_reason=hard_gate,
            checks=checks,
            raw_verifier_rates={"task_tests": self.task_pass_rate},
        )
        return 0.2


class TrainingPreflightTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.dataset = self.root / "grpo.jsonl"
        self.dataset.write_text(
            json.dumps(
                {
                    "task_id": "pricing-001",
                    "prompt": [{"role": "user", "content": "Repair pricing."}],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        self.recipe = {
            "environment": {"factory": "unused.module:create_environment"},
            "grpo": {"dataset": {"path": str(self.dataset)}},
        }

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_canary_returns_executable_runtime_and_signal_evidence(self) -> None:
        result = run_grpo_environment_canary(
            self.recipe,
            factory=lambda: FakeEnvironment(task_pass_rate=0.25),
        )

        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["task_id"], "pricing-001")
        self.assertEqual(result["task_pass_rate"], 0.25)
        self.assertEqual(result["checks"]["verifier"], "passed")

    def test_canary_surfaces_environment_initialization_failure(self) -> None:
        with self.assertRaisesRegex(TrainingRuntimeError, "container unavailable"):
            run_grpo_environment_canary(
                self.recipe,
                factory=lambda: FakeEnvironment(
                    reset_error=RuntimeError("container unavailable")
                ),
            )

    def test_canary_rejects_failed_verifier(self) -> None:
        with self.assertRaisesRegex(TrainingRuntimeError, "verifier_failed"):
            run_grpo_environment_canary(
                self.recipe,
                factory=lambda: FakeEnvironment(failed_check="verifier"),
            )

    def test_canary_rejects_a_task_already_solved_at_its_starting_revision(self) -> None:
        with self.assertRaisesRegex(TrainingRuntimeError, "already passes every task test"):
            run_grpo_environment_canary(
                self.recipe,
                factory=lambda: FakeEnvironment(task_pass_rate=1.0),
            )


if __name__ == "__main__":
    unittest.main()
