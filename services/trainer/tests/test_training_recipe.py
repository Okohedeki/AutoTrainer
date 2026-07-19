from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE_ROOT / "src"))

from autotrainer.training import (  # noqa: E402
    TrainingConfigurationError,
    TrainingRuntimeError,
    resolve_grpo_recipe,
    run_grpo,
    run_sft,
)
from autotrainer.training.common import (  # noqa: E402
    claim_fresh_output_directory,
    validate_sft_token_lengths,
    verify_adapter_tree_identity,
)
from autotrainer.training.grpo import (  # noqa: E402
    _bind_environment_image_identity,
    _load_json_dataset as load_grpo_json_dataset,
    _save_grpo_processing_class,
)
from autotrainer.training.sft import (  # noqa: E402
    _load_json_dataset as load_sft_json_dataset,
)


IMAGE_DIGEST = "sha256:" + "b" * 64


class TrainingRecipeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.project_root = Path(self.temporary_directory.name)
        self.revision = "a" * 40
        data_directory = self.project_root / "data"
        data_directory.mkdir()
        self.sft_dataset = data_directory / "sft.jsonl"
        self.sft_dataset.write_text(
            json.dumps(
                {
                    "prompt": [{"role": "user", "content": "Create a pricing page."}],
                    "completion": [
                        {
                            "role": "assistant",
                            "content": "I will inspect the repository.",
                        }
                    ],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        self.grpo_dataset = data_directory / "grpo.jsonl"
        self.grpo_dataset.write_text(
            json.dumps(
                {
                    "prompt": [{"role": "user", "content": "Create a pricing page."}],
                    "task_id": "pricing-page-001",
                    "task_path": "tasks/pricing-page",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        self.adapter = self.project_root / "artifacts" / "sft-adapter"
        self.adapter.mkdir(parents=True)
        (self.adapter / "adapter_config.json").write_text(
            json.dumps(
                {
                    "base_model_name_or_path": "Qwen/Qwen3.5-9B",
                    "revision": self.revision,
                    "peft_type": "LORA",
                }
            ),
            encoding="utf-8",
        )
        (self.adapter / "adapter_model.safetensors").write_bytes(b"adapter-weights")

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def config(self) -> dict:
        return {
            "schema_version": 1,
            "model": {
                "id": "Qwen/Qwen3.5-9B",
                "revision": self.revision,
                "text_only": True,
                "trust_remote_code": False,
                "thinking": False,
                "dtype": "bfloat16",
            },
            "qlora": {
                "load_in_4bit": True,
                "quant_type": "nf4",
                "double_quant": True,
                "compute_dtype": "bfloat16",
                "rank": 32,
                "alpha": 32,
                "dropout": 0.0,
                "bias": "none",
                "target_modules": "all-linear",
            },
            "sft": {"dataset": "data/sft.jsonl"},
            "grpo": {
                "dataset": "data/grpo.jsonl",
                "sft_adapter": "artifacts/sft-adapter",
            },
            "environment": {
                "factory": "my_project.environment:create_environment",
                "backend": "docker",
                "image": "autotrainer-rollout:latest",
            },
        }

    def test_sft_dry_run_resolves_local_data_and_effective_batch(self) -> None:
        result = run_sft(
            self.config(),
            project_root=self.project_root,
            output_dir=Path("artifacts/sft-output"),
            dry_run=True,
        )
        self.assertEqual(result["status"], "dry_run")
        self.assertEqual(result["recipe"]["stage"], "sft")
        self.assertEqual(result["recipe"]["sft"]["effective_batch_size"], 8)
        self.assertEqual(result["recipe"]["sft"]["per_device_eval_batch_size"], 1)
        self.assertEqual(
            result["recipe"]["sft"]["dataset"]["path"], str(self.sft_dataset)
        )
        dataset = result["recipe"]["sft"]["dataset"]
        self.assertEqual(dataset["record_count"], 1)
        self.assertEqual(dataset["bytes"], self.sft_dataset.stat().st_size)
        self.assertEqual(len(dataset["sha256"]), 64)
        self.assertTrue(result["recipe"]["model"]["local_files_only"])
        self.assertEqual(
            result["recipe"]["model"]["cache_dir"],
            str((self.project_root / ".autotrainer" / "model-cache").resolve()),
        )

    def test_sft_dry_run_rejects_an_uncompiled_messages_record(self) -> None:
        self.sft_dataset.write_text(
            json.dumps(
                {
                    "messages": [
                        {"role": "user", "content": "Create a pricing page."},
                        {"role": "assistant", "content": "Here is the result."},
                    ]
                }
            )
            + "\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(
            TrainingConfigurationError, "compiled conversational prompt and completion"
        ):
            run_sft(
                self.config(),
                project_root=self.project_root,
                output_dir=Path("artifacts/sft-output"),
                dry_run=True,
            )

    def test_sft_dry_run_validates_later_records_before_model_loading(self) -> None:
        invalid_second = {
            "messages": [
                {"role": "user", "content": "Repair the checkout."},
                {"role": "assistant", "content": "Done."},
            ]
        }
        self.sft_dataset.write_text(
            self.sft_dataset.read_text(encoding="utf-8")
            + json.dumps(invalid_second)
            + "\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(
            TrainingConfigurationError,
            "record 2 must use compiled conversational prompt and completion",
        ):
            run_sft(
                self.config(),
                project_root=self.project_root,
                output_dir=Path("artifacts/sft-output"),
                dry_run=True,
            )

    def test_sft_checks_every_full_conversation_with_the_real_tokenizer_boundary(
        self,
    ) -> None:
        second = {
            "prompt": [{"role": "user", "content": "short task"}],
            "completion": [{"role": "assistant", "content": "x" * 80}],
        }
        self.sft_dataset.write_text(
            self.sft_dataset.read_text(encoding="utf-8") + json.dumps(second) + "\n",
            encoding="utf-8",
        )

        class LengthTokenizer:
            def apply_chat_template(
                self, messages: list[dict], **_: object
            ) -> list[int]:
                length = sum(len(message["content"]) for message in messages)
                return list(range(length))

        with self.assertRaisesRegex(TrainingConfigurationError, "record 2 uses"):
            validate_sft_token_lengths(
                LengthTokenizer(), self.sft_dataset, max_length=60
            )

        report = validate_sft_token_lengths(
            LengthTokenizer(), self.sft_dataset, max_length=200
        )
        self.assertEqual(report["record_count"], 2)
        self.assertGreater(report["longest_token_count"], 60)

    def test_real_sft_rejects_mutable_model_before_dependency_imports(self) -> None:
        config = self.config()
        config["model"]["revision"] = "main"
        with self.assertRaisesRegex(RuntimeError, "immutable downloaded"):
            run_sft(
                config,
                project_root=self.project_root,
                output_dir=Path("artifacts/sft-output"),
                dry_run=False,
            )

    def test_real_grpo_runs_executable_canary_before_training_imports(self) -> None:
        with (
            patch("autotrainer.training.grpo.require_materialized_model"),
            patch(
                "autotrainer.training.grpo.validate_reference_dependencies",
                return_value={},
            ),
            patch(
                "autotrainer.training.grpo.run_grpo_environment_canary",
                side_effect=RuntimeError("verifier canary failed"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "verifier canary failed"):
                run_grpo(
                    self.config(),
                    project_root=self.project_root,
                    output_dir=Path("artifacts/grpo-output"),
                    dry_run=False,
                )

    def test_grpo_dry_run_requires_and_records_same_sft_adapter(self) -> None:
        result = run_grpo(
            self.config(),
            project_root=self.project_root,
            output_dir=Path("artifacts/grpo-output"),
            dry_run=True,
        )
        recipe = result["recipe"]
        self.assertEqual(recipe["stage"], "grpo")
        self.assertEqual(recipe["grpo"]["sft_adapter"]["path"], str(self.adapter))
        self.assertEqual(
            recipe["environment"]["factory"],
            "my_project.environment:create_environment",
        )
        self.assertEqual(recipe["environment"]["backend"], "docker")
        self.assertEqual(
            recipe["environment"]["image"],
            "autotrainer-rollout:latest",
        )
        self.assertEqual(recipe["grpo"]["effective_batch_size"], 2)
        self.assertEqual(recipe["grpo"]["per_device_eval_batch_size"], 1)
        self.assertEqual(recipe["grpo"]["num_generations"], 2)
        self.assertEqual(recipe["grpo"]["dataset"]["record_count"], 1)
        self.assertEqual(
            recipe["grpo"]["dataset"]["bytes"], self.grpo_dataset.stat().st_size
        )
        self.assertEqual(len(recipe["grpo"]["dataset"]["sha256"]), 64)
        self.assertEqual(recipe["grpo"]["sft_adapter"]["tree"]["file_count"], 2)
        self.assertEqual(len(recipe["grpo"]["sft_adapter"]["tree"]["sha256"]), 64)

    def test_grpo_runtime_dataset_is_bound_to_the_canary_image(self) -> None:
        class Dataset:
            def __init__(self) -> None:
                self.update: dict[str, str] | None = None
                self.description: str | None = None

            def map(self, transform: object, *, desc: str) -> Dataset:
                self.update = transform({})  # type: ignore[operator]
                self.description = desc
                return self

        dataset = Dataset()
        result = _bind_environment_image_identity(dataset, IMAGE_DIGEST)

        self.assertIs(result, dataset)
        self.assertEqual(dataset.update, {"environment_image_identity": IMAGE_DIGEST})
        self.assertEqual(dataset.description, "Binding immutable rollout image")

    def test_grpo_persists_the_template_created_by_the_trainer(self) -> None:
        class ProcessingClass:
            chat_template = "original-qwen-template"

            def __init__(self) -> None:
                self.saved_to: str | None = None

            def save_pretrained(self, destination: str) -> None:
                self.saved_to = destination

        processing_class = ProcessingClass()
        trainer = MagicMock(
            processing_class=processing_class,
            chat_template="trl-response-aware-training-template",
        )
        destination = self.project_root / "saved-tokenizer"

        _save_grpo_processing_class(trainer, object(), destination)

        self.assertEqual(
            processing_class.chat_template,
            "trl-response-aware-training-template",
        )
        self.assertEqual(processing_class.saved_to, str(destination))

    def test_grpo_refuses_to_publish_without_a_persistent_trainer_template(
        self,
    ) -> None:
        processing_class = MagicMock(chat_template=None)
        trainer = MagicMock(processing_class=processing_class, chat_template=None)

        with self.assertRaisesRegex(
            TrainingRuntimeError, "persistent training chat template"
        ):
            _save_grpo_processing_class(
                trainer,
                object(),
                self.project_root / "saved-tokenizer",
            )

    def test_stage_loaders_reject_dataset_mutation_after_resolution(self) -> None:
        sft_recipe = run_sft(
            self.config(),
            project_root=self.project_root,
            output_dir=Path("artifacts/sft-output"),
            dry_run=True,
        )["recipe"]
        grpo_recipe = run_grpo(
            self.config(),
            project_root=self.project_root,
            output_dir=Path("artifacts/grpo-output"),
            dry_run=True,
        )["recipe"]
        self.sft_dataset.write_text(
            self.sft_dataset.read_text(encoding="utf-8") + "\n",
            encoding="utf-8",
        )
        self.grpo_dataset.write_text(
            self.grpo_dataset.read_text(encoding="utf-8") + "\n",
            encoding="utf-8",
        )

        loader = MagicMock()
        with self.assertRaisesRegex(RuntimeError, "Dataset changed"):
            load_sft_json_dataset(loader, sft_recipe["sft"]["dataset"])
        with self.assertRaisesRegex(RuntimeError, "Dataset changed"):
            load_grpo_json_dataset(loader, grpo_recipe["grpo"]["dataset"])
        loader.assert_not_called()

    def test_adapter_tree_mutation_is_rejected_before_peft_load(self) -> None:
        recipe = run_grpo(
            self.config(),
            project_root=self.project_root,
            output_dir=Path("artifacts/grpo-output"),
            dry_run=True,
        )["recipe"]
        description = recipe["grpo"]["start_from"]["adapter"]
        (self.adapter / "adapter_model.safetensors").write_bytes(b"changed")

        with self.assertRaisesRegex(RuntimeError, "Adapter changed"):
            verify_adapter_tree_identity(description)

    def test_adapter_revision_provenance_is_mandatory(self) -> None:
        config_path = self.adapter / "adapter_config.json"
        adapter_config = json.loads(config_path.read_text(encoding="utf-8"))
        adapter_config.pop("revision")
        config_path.write_text(json.dumps(adapter_config), encoding="utf-8")

        with self.assertRaisesRegex(
            TrainingConfigurationError, "immutable base-model revision"
        ):
            resolve_grpo_recipe(
                self.config(),
                project_root=self.project_root,
                output_dir=Path("artifacts/grpo-output"),
            )

    def test_nonempty_output_directory_cannot_be_reused(self) -> None:
        destination = self.project_root / "artifacts" / "existing-output"
        destination.mkdir(parents=True)
        (destination / "checkpoint.bin").write_bytes(b"old run")

        with self.assertRaisesRegex(TrainingConfigurationError, "must be empty"):
            run_sft(
                self.config(),
                project_root=self.project_root,
                output_dir=destination,
                dry_run=True,
            )

    def test_empty_output_directory_receives_an_exclusive_run_claim(self) -> None:
        destination = self.project_root / "artifacts" / "new-output"
        destination.mkdir(parents=True)
        claim_fresh_output_directory(destination)

        claim = destination / ".autotrainer-run-claim.json"
        self.assertEqual(
            json.loads(claim.read_text(encoding="utf-8"))["status"], "running"
        )
        with self.assertRaisesRegex(TrainingConfigurationError, "must be empty"):
            claim_fresh_output_directory(destination)

    def test_grpo_recipe_validates_every_compiled_row(self) -> None:
        malformed = {
            "task_id": "pricing-page-002",
            "prompt": "this must remain a conversational message list",
        }
        self.grpo_dataset.write_text(
            self.grpo_dataset.read_text(encoding="utf-8")
            + json.dumps(malformed)
            + "\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(TrainingConfigurationError, "record 2.prompt"):
            resolve_grpo_recipe(
                self.config(),
                project_root=self.project_root,
                output_dir=Path("artifacts/grpo-output"),
            )

    def test_grpo_recipe_requires_supported_roles_and_a_final_user_message(
        self,
    ) -> None:
        cases = (
            (
                [{"role": "developer", "content": "hidden policy"}],
                "role must be system, user, assistant, or tool",
            ),
            (
                [{"role": "assistant", "content": "Already answered."}],
                "must end with a user message",
            ),
        )
        for index, (prompt, message) in enumerate(cases, 2):
            with self.subTest(prompt=prompt):
                self.grpo_dataset.write_text(
                    self.grpo_dataset.read_text(encoding="utf-8")
                    + json.dumps({"task_id": f"invalid-{index}", "prompt": prompt})
                    + "\n",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(TrainingConfigurationError, message):
                    resolve_grpo_recipe(
                        self.config(),
                        project_root=self.project_root,
                        output_dir=Path("artifacts/grpo-output"),
                    )
                # Each subtest starts from the valid first row so one invalid
                # case cannot shadow the validation branch exercised by next.
                lines = self.grpo_dataset.read_text(encoding="utf-8").splitlines()
                self.grpo_dataset.write_text(lines[0] + "\n", encoding="utf-8")

    def test_grpo_recipe_rejects_duplicate_task_ids(self) -> None:
        duplicate = {
            "task_id": "pricing-page-001",
            "prompt": [{"role": "user", "content": "A different task."}],
        }
        self.grpo_dataset.write_text(
            self.grpo_dataset.read_text(encoding="utf-8")
            + json.dumps(duplicate)
            + "\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(TrainingConfigurationError, "duplicate task_id"):
            resolve_grpo_recipe(
                self.config(),
                project_root=self.project_root,
                output_dir=Path("artifacts/grpo-output"),
            )

    def test_practice_dry_run_creates_a_fresh_qlora_policy_from_base(self) -> None:
        config = self.config()
        config["sft"]["enabled"] = False
        config["grpo"].pop("sft_adapter")
        config["grpo"]["start_from"] = "base"

        result = run_grpo(
            config,
            project_root=self.project_root,
            output_dir=Path("artifacts/grpo-output"),
            dry_run=True,
        )

        self.assertEqual(result["recipe"]["grpo"]["start_from"]["type"], "base")
        self.assertIsNone(result["recipe"]["grpo"]["sft_adapter"])

    def test_both_recipe_cannot_skip_its_sft_adapter(self) -> None:
        config = self.config()
        config["grpo"].pop("sft_adapter")
        config["grpo"]["start_from"] = "base"

        with self.assertRaisesRegex(TrainingConfigurationError, "both-stage"):
            run_grpo(
                config,
                project_root=self.project_root,
                output_dir=Path("artifacts/grpo-output"),
                dry_run=True,
            )

    def test_disabled_stage_runners_fail_closed(self) -> None:
        sft_config = self.config()
        sft_config["sft"]["enabled"] = False
        with self.assertRaisesRegex(TrainingConfigurationError, "SFT is disabled"):
            run_sft(
                sft_config,
                project_root=self.project_root,
                output_dir=Path("artifacts/sft-output"),
                dry_run=True,
            )

        grpo_config = self.config()
        grpo_config["grpo"]["enabled"] = False
        with self.assertRaisesRegex(TrainingConfigurationError, "GRPO is disabled"):
            run_grpo(
                grpo_config,
                project_root=self.project_root,
                output_dir=Path("artifacts/grpo-output"),
                dry_run=True,
            )

    def test_grpo_rejects_invalid_generation_arithmetic(self) -> None:
        config = self.config()
        config["grpo"]["gradient_accumulation_steps"] = 3
        with self.assertRaisesRegex(
            TrainingConfigurationError, "effective batch size must be divisible"
        ):
            resolve_grpo_recipe(
                config,
                project_root=self.project_root,
                output_dir=Path("artifacts/grpo-output"),
            )

    def test_grpo_rejects_output_that_would_overwrite_sft_adapter(self) -> None:
        with self.assertRaisesRegex(TrainingConfigurationError, "must differ"):
            resolve_grpo_recipe(
                self.config(),
                project_root=self.project_root,
                output_dir=Path("artifacts/sft-adapter"),
            )

    def test_sft_rejects_multimodal_records(self) -> None:
        self.sft_dataset.write_text(
            json.dumps(
                {
                    "messages": [
                        {"role": "user", "content": "Describe this image."},
                        {"role": "assistant", "content": "No."},
                    ],
                    "image": "screen.png",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(TrainingConfigurationError, "text-only"):
            run_sft(
                self.config(),
                project_root=self.project_root,
                output_dir=Path("artifacts/sft-output"),
                dry_run=True,
            )

    def test_grpo_rejects_non_qwen_sft_adapter(self) -> None:
        adapter_config_path = self.adapter / "adapter_config.json"
        adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
        adapter_config["base_model_name_or_path"] = "some/other-model"
        adapter_config_path.write_text(json.dumps(adapter_config), encoding="utf-8")
        with self.assertRaisesRegex(TrainingConfigurationError, "does not match"):
            run_grpo(
                self.config(),
                project_root=self.project_root,
                output_dir=Path("artifacts/grpo-output"),
                dry_run=True,
            )

    def test_environment_factory_must_be_importable_path_syntax(self) -> None:
        config = self.config()
        config["environment"]["factory"] = "not a path"
        with self.assertRaisesRegex(TrainingConfigurationError, "valid dotted path"):
            run_grpo(
                config,
                project_root=self.project_root,
                output_dir=Path("artifacts/grpo-output"),
                dry_run=True,
            )


if __name__ == "__main__":
    unittest.main()
