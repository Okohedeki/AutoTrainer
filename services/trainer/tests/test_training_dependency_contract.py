from __future__ import annotations

from importlib import metadata
import inspect
import os
from pathlib import Path
import sys
import unittest


SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE_ROOT / "src"))

from autotrainer.training.common import (  # noqa: E402
    REFERENCE_DEPENDENCIES,
    validate_reference_dependencies,
)


class TrainingDependencyContractTests(unittest.TestCase):
    """Catch upstream API drift without allocating a model or requiring CUDA."""

    @classmethod
    def setUpClass(cls) -> None:
        missing = []
        for distribution in REFERENCE_DEPENDENCIES:
            try:
                metadata.version(distribution)
            except metadata.PackageNotFoundError:
                missing.append(distribution)
        if missing:
            message = "pinned training stack is not installed: " + ", ".join(missing)
            if os.environ.get("AUTOTRAINER_REQUIRE_TRAINING_STACK") == "1":
                raise AssertionError(message)
            raise unittest.SkipTest(message)

    def test_all_reference_versions_are_exact(self) -> None:
        self.assertEqual(validate_reference_dependencies(), REFERENCE_DEPENDENCIES)

    def test_grpo_environment_factory_api_is_still_available(self) -> None:
        from trl import GRPOConfig, GRPOTrainer

        trainer_parameters = inspect.signature(GRPOTrainer.__init__).parameters
        self.assertIn("environment_factory", trainer_parameters)
        fields = getattr(GRPOConfig, "__dataclass_fields__", {})
        for name in (
            "generation_batch_size",
            "max_tool_calling_iterations",
            "loss_type",
            "use_vllm",
        ):
            self.assertIn(name, fields)

    def test_grpo_adds_the_response_schema_before_deriving_training_template(self) -> None:
        from trl import GRPOTrainer

        source = inspect.getsource(GRPOTrainer.__init__)
        response_schema = source.find("add_response_schema")
        training_template = source.find("get_training_chat_template")
        self.assertGreaterEqual(response_schema, 0)
        self.assertGreaterEqual(training_template, 0)
        self.assertLess(response_schema, training_template)

        # AutoTrainer must pass the original pinned tokenizer into that order;
        # pre-mutating it to a *_training template makes response-schema
        # detection fail in the pinned TRL release.
        from autotrainer.training.grpo import run_grpo

        autotrainer_source = inspect.getsource(run_grpo)
        before_trainer = autotrainer_source.split("trainer = GRPOTrainer", 1)[0]
        self.assertNotIn("get_training_chat_template", before_trainer)

    def test_peft_can_record_the_immutable_base_revision(self) -> None:
        from peft import LoraConfig

        fields = getattr(LoraConfig, "__dataclass_fields__", {})
        self.assertIn("revision", fields)


if __name__ == "__main__":
    unittest.main()
