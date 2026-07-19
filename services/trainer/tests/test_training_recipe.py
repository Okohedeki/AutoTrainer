from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE_ROOT / "src"))

from autotrainer.training import (  # noqa: E402
    TrainingConfigurationError,
    resolve_grpo_recipe,
    run_grpo,
    run_sft,
)
from autotrainer.training.common import validate_sft_token_lengths  # noqa: E402


class TrainingRecipeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.project_root = Path(self.temporary_directory.name)
        data_directory = self.project_root / "data"
        data_directory.mkdir()
        self.sft_dataset = data_directory / "sft.jsonl"
        self.sft_dataset.write_text(
            json.dumps(
                {
                    "prompt": [{"role": "user", "content": "Create a pricing page."}],
                    "completion": [
                        {"role": "assistant", "content": "I will inspect the repository."}
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
                    "revision": "main",
                    "peft_type": "LORA",
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def config(self) -> dict:
        return {
            "schema_version": 1,
            "model": {
                "id": "Qwen/Qwen3.5-9B",
                "revision": "main",
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
            "environment": {"factory": "my_project.environment:create_environment"},
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
        self.assertEqual(
            result["recipe"]["sft"]["dataset"]["path"], str(self.sft_dataset)
        )
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

    def test_sft_checks_every_full_conversation_with_the_real_tokenizer_boundary(self) -> None:
        second = {
            "prompt": [{"role": "user", "content": "short task"}],
            "completion": [{"role": "assistant", "content": "x" * 80}],
        }
        self.sft_dataset.write_text(
            self.sft_dataset.read_text(encoding="utf-8") + json.dumps(second) + "\n",
            encoding="utf-8",
        )

        class LengthTokenizer:
            def apply_chat_template(self, messages: list[dict], **_: object) -> list[int]:
                length = sum(len(message["content"]) for message in messages)
                return list(range(length))

        with self.assertRaisesRegex(TrainingConfigurationError, "record 2 uses"):
            validate_sft_token_lengths(LengthTokenizer(), self.sft_dataset, max_length=60)

        report = validate_sft_token_lengths(
            LengthTokenizer(), self.sft_dataset, max_length=200
        )
        self.assertEqual(report["record_count"], 2)
        self.assertGreater(report["longest_token_count"], 60)

    def test_real_sft_rejects_mutable_model_before_dependency_imports(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "immutable downloaded"):
            run_sft(
                self.config(),
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
        self.assertEqual(recipe["grpo"]["effective_batch_size"], 2)
        self.assertEqual(recipe["grpo"]["num_generations"], 2)
        self.assertEqual(recipe["grpo"]["dataset"]["record_count"], 1)

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
