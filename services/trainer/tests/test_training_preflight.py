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
from autotrainer.training.common import dataset_file_identity  # noqa: E402
from autotrainer.integrity import IntegrityError  # noqa: E402


IMAGE_DIGEST = "sha256:" + "b" * 64
IMAGE_IDENTITY = {
    "backend": "docker",
    "reference": "autotrainer-rollout:latest",
    "digest": IMAGE_DIGEST,
    "runtime_reference": IMAGE_DIGEST,
    "resolution": "local_image_id",
}


def resolve_image(backend: str, image: str) -> dict[str, str]:
    if backend != "docker" or image != "autotrainer-rollout:latest":
        raise AssertionError("unexpected image resolution request")
    return dict(IMAGE_IDENTITY)


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
        separated_observation: bool = True,
        observed_tasks: list[str] | None = None,
        observed_images: list[str] | None = None,
    ) -> None:
        self.task_pass_rate = task_pass_rate
        self.failed_check = failed_check
        self.reset_error = reset_error
        self.separated_observation = separated_observation
        self.observed_tasks = observed_tasks
        self.observed_images = observed_images
        self.last_result = None

    def reset(self, **row: object) -> str:
        if self.reset_error is not None:
            raise self.reset_error
        if self.observed_tasks is not None:
            self.observed_tasks.append(str(row["task_id"]))
        if self.observed_images is not None:
            self.observed_images.append(str(row["environment_image_identity"]))
        prefix = "\n\n" if self.separated_observation else ""
        return f"{prefix}Environment state: {row['task_id']}"

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
                    "environment_backend": "docker",
                    "environment_image": "autotrainer-rollout:latest",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        self.recipe = {
            "environment": {
                "factory": "unused.module:create_environment",
                "backend": "docker",
                "image": "autotrainer-rollout:latest",
            },
            "grpo": {
                "dataset": {
                    "path": str(self.dataset),
                    **dataset_file_identity(self.dataset),
                }
            },
        }

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_canary_returns_executable_runtime_and_signal_evidence(self) -> None:
        observed_images: list[str] = []
        result = run_grpo_environment_canary(
            self.recipe,
            factory=lambda: FakeEnvironment(
                task_pass_rate=0.25,
                observed_images=observed_images,
            ),
            image_resolver=resolve_image,
        )

        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["task_id"], "pricing-001")
        self.assertEqual(result["task_pass_rate"], 0.25)
        self.assertEqual(result["task_count"], 1)
        self.assertEqual([task["task_id"] for task in result["tasks"]], ["pricing-001"])
        self.assertEqual(result["checks"]["verifier"], "passed")
        self.assertEqual(result["container_image"], IMAGE_IDENTITY)
        self.assertEqual(observed_images, [IMAGE_DIGEST])

    def test_canary_executes_every_compiled_task_before_reporting_ready(self) -> None:
        with self.dataset.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "task_id": "checkout-002",
                        "prompt": [{"role": "user", "content": "Repair checkout."}],
                        "environment_backend": "docker",
                        "environment_image": "autotrainer-rollout:latest",
                    }
                )
                + "\n"
            )
        self.recipe["grpo"]["dataset"].update(dataset_file_identity(self.dataset))
        observed: list[str] = []

        result = run_grpo_environment_canary(
            self.recipe,
            factory=lambda: FakeEnvironment(observed_tasks=observed),
            image_resolver=resolve_image,
        )

        self.assertEqual(observed, ["pricing-001", "checkout-002"])
        self.assertEqual(result["task_count"], 2)

    def test_canary_refuses_a_broken_later_task(self) -> None:
        with self.dataset.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "task_id": "broken-002",
                        "prompt": [{"role": "user", "content": "Repair checkout."}],
                        "environment_backend": "docker",
                        "environment_image": "autotrainer-rollout:latest",
                    }
                )
                + "\n"
            )
        self.recipe["grpo"]["dataset"].update(dataset_file_identity(self.dataset))
        created = 0

        def factory() -> FakeEnvironment:
            nonlocal created
            created += 1
            return FakeEnvironment(failed_check="verifier" if created == 2 else None)

        with self.assertRaisesRegex(
            TrainingRuntimeError, "broken-002.*verifier_failed"
        ):
            run_grpo_environment_canary(
                self.recipe,
                factory=factory,
                image_resolver=resolve_image,
            )
        self.assertEqual(created, 2)

    def test_canary_surfaces_environment_initialization_failure(self) -> None:
        with self.assertRaisesRegex(TrainingRuntimeError, "container unavailable"):
            run_grpo_environment_canary(
                self.recipe,
                factory=lambda: FakeEnvironment(
                    reset_error=RuntimeError("container unavailable")
                ),
                image_resolver=resolve_image,
            )

    def test_canary_rejects_an_observation_glued_to_the_compiled_prompt(self) -> None:
        with self.assertRaisesRegex(TrainingRuntimeError, "blank-line separator"):
            run_grpo_environment_canary(
                self.recipe,
                factory=lambda: FakeEnvironment(separated_observation=False),
                image_resolver=resolve_image,
            )

    def test_canary_rejects_failed_verifier(self) -> None:
        with self.assertRaisesRegex(TrainingRuntimeError, "verifier_failed"):
            run_grpo_environment_canary(
                self.recipe,
                factory=lambda: FakeEnvironment(failed_check="verifier"),
                image_resolver=resolve_image,
            )

    def test_canary_rejects_a_task_already_solved_at_its_starting_revision(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            TrainingRuntimeError, "already passes every task test"
        ):
            run_grpo_environment_canary(
                self.recipe,
                factory=lambda: FakeEnvironment(task_pass_rate=1.0),
                image_resolver=resolve_image,
            )

    def test_canary_rejects_compiled_runtime_divergence_before_image_resolution(
        self,
    ) -> None:
        row = json.loads(self.dataset.read_text(encoding="utf-8"))
        row["environment_image"] = "other-rollout:latest"
        self.dataset.write_text(json.dumps(row) + "\n", encoding="utf-8")
        self.recipe["grpo"]["dataset"].update(dataset_file_identity(self.dataset))

        with self.assertRaisesRegex(TrainingRuntimeError, "re-run compilation"):
            run_grpo_environment_canary(
                self.recipe,
                factory=FakeEnvironment,
                image_resolver=lambda *_args: self.fail("resolver must not be called"),
            )

    def test_canary_fails_closed_when_mutable_image_cannot_be_resolved(self) -> None:
        def fail_resolution(_backend: str, _image: str) -> dict[str, str]:
            raise IntegrityError("image tag is unavailable")

        with self.assertRaisesRegex(TrainingRuntimeError, "image tag is unavailable"):
            run_grpo_environment_canary(
                self.recipe,
                factory=FakeEnvironment,
                image_resolver=fail_resolution,
            )


if __name__ == "__main__":
    unittest.main()
