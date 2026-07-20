from __future__ import annotations

from importlib import metadata
import inspect
import os
from pathlib import Path
import sys
import tempfile
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
        installed = validate_reference_dependencies()
        self.assertEqual(set(installed), set(REFERENCE_DEPENDENCIES))
        for distribution, expected in REFERENCE_DEPENDENCIES.items():
            with self.subTest(distribution=distribution):
                # CUDA wheels carry a PEP 440 local build suffix such as
                # +cu130; the pinned public version remains exact.
                self.assertEqual(installed[distribution].split("+", 1)[0], expected)

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

    def test_grpo_keeps_kv_cache_for_autoregressive_generation(self) -> None:
        from trl import GRPOTrainer

        from autotrainer.training.grpo import run_grpo

        autotrainer_source = inspect.getsource(run_grpo)
        before_trainer = autotrainer_source.split("trainer = GRPOTrainer", 1)[0]
        self.assertIn("use_cache=True", before_trainer)

        # The pinned TRL runtime still disables caching for the differentiable
        # policy forward pass, so enabling the trainer-level setting affects
        # generation without retaining inference caches during optimization.
        scored_forward = inspect.getsource(GRPOTrainer._get_last_hidden_state)
        self.assertIn('model_inputs["use_cache"] = False', scored_forward)

    def test_real_grpo_trainer_initializes_and_renders_every_environment_tool(self) -> None:
        """Exercise TRL's actual environment probe and tool-schema rendering path."""

        from datasets import Dataset
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import (
            GPT2Config,
            GPT2LMHeadModel,
            PreTrainedTokenizerFast,
        )
        from trl import GRPOConfig, GRPOTrainer
        from trl.chat_template_utils import qwen3_5_nothink_chat_template

        from autotrainer.environments.frontend import FrontendEnvironment

        backend = Tokenizer(
            WordLevel(
                vocab={"<unk>": 0, "<pad>": 1, "<eos>": 2},
                unk_token="<unk>",
            )
        )
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend,
            unk_token="<unk>",
            pad_token="<pad>",
            eos_token="<eos>",
        )
        tokenizer.chat_template = qwen3_5_nothink_chat_template
        tokenizer.padding_side = "left"
        model = GPT2LMHeadModel(
            GPT2Config(
                vocab_size=len(tokenizer),
                n_positions=64,
                n_embd=16,
                n_layer=1,
                n_head=1,
                bos_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        )
        prompt = [
            {"role": "system", "content": "Use the tools."},
            {"role": "user", "content": "Inspect and repair the fixture."},
        ]
        dataset = Dataset.from_list(
            [{"prompt": prompt, "task_id": "environment-contract"}]
        )

        with tempfile.TemporaryDirectory() as output:
            trainer = GRPOTrainer(
                model=model,
                args=GRPOConfig(
                    output_dir=str(Path(output) / "trainer"),
                    per_device_train_batch_size=1,
                    gradient_accumulation_steps=2,
                    num_generations=2,
                    generation_batch_size=2,
                    max_completion_length=8,
                    max_steps=1,
                    use_vllm=False,
                    bf16=False,
                    fp16=False,
                    report_to="none",
                    remove_unused_columns=False,
                ),
                train_dataset=dataset,
                processing_class=tokenizer,
                environment_factory=FrontendEnvironment,
            )
            self.assertEqual(
                {tool.__name__ for tool in trainer.tools},
                {
                    "apply_patch",
                    "list_files",
                    "read_file",
                    "replace_text",
                    "run_check",
                    "search_code",
                },
            )
            # Schema conversion is lazy in TRL 1.8. Force the exact path used
            # immediately before generation; malformed Args documentation used
            # to raise DocstringParsingException here.
            trainer._batch_environments = [None]
            prompt_ids, images, multimodal_fields = trainer._tokenize_prompts([prompt])
            self.assertTrue(prompt_ids[0])
            self.assertIsNone(images)
            self.assertEqual(multimodal_fields, {})

    def test_peft_can_record_the_immutable_base_revision(self) -> None:
        from peft import LoraConfig

        fields = getattr(LoraConfig, "__dataclass_fields__", {})
        self.assertIn("revision", fields)


if __name__ == "__main__":
    unittest.main()
