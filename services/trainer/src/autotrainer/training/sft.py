"""Guarded text-only QLoRA supervised fine-tuning."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from ..model_cache import require_materialized_model
from .telemetry import TrainingEventCallback, make_trainer_log_callback
from .common import (
    SUPPORTED_MODEL_CLASS,
    TrainingConfigurationError,
    TrainingDependencyError,
    TrainingRuntimeError,
    assert_adapter_only,
    as_serializable,
    base_recipe,
    bool_value,
    claim_fresh_output_directory,
    configure_vram_budget,
    float_value,
    get_section,
    inspect_sft_dataset,
    int_value,
    mark_output_directory_complete,
    model_max_memory,
    resolve_input_file,
    validate_reference_dependencies,
    validate_fresh_output_directory,
    validate_sft_token_lengths,
    verify_dataset_identity,
    verify_saved_adapter_provenance,
)


def resolve_sft_recipe(
    config: Mapping[str, Any], *, project_root: Path, output_dir: Path
) -> dict[str, Any]:
    """Resolve and validate an SFT recipe without importing training libraries."""

    recipe = base_recipe(config, project_root=project_root, output_dir=output_dir)
    section = get_section(config, "sft")
    if section.get("enabled", True) is False:
        raise TrainingConfigurationError("SFT is disabled for the selected training recipe")
    root = Path(recipe["project_root"])

    dataset_path = resolve_input_file(section.get("dataset"), root, "sft.dataset")
    eval_value = section.get("eval_dataset")
    eval_path = (
        resolve_input_file(eval_value, root, "sft.eval_dataset")
        if eval_value is not None
        else None
    )
    batch_size = int_value(section, "per_device_train_batch_size", 1, "sft")
    accumulation = int_value(section, "gradient_accumulation_steps", 8, "sft")
    effective_batch_size = batch_size * accumulation
    if effective_batch_size >= 32:
        raise TrainingConfigurationError(
            "sft effective batch size must remain below 32 for the supported LoRA recipe; "
            f"got {batch_size} * {accumulation} = {effective_batch_size}"
        )

    max_length = int_value(section, "max_length", 2048, "sft")
    if max_length > 4096:
        raise TrainingConfigurationError(
            "sft.max_length must be <= 4096 on the supported 24 GB runtime"
        )
    epochs = float_value(
        section,
        "num_train_epochs",
        1.0,
        "sft",
        minimum=0.0,
        minimum_inclusive=False,
    )
    learning_rate = float_value(
        section,
        "learning_rate",
        1.0e-4,
        "sft",
        minimum=0.0,
        maximum=1.0e-2,
        minimum_inclusive=False,
    )
    bf16 = bool_value(section, "bf16", True, "sft")
    tf32 = bool_value(section, "tf32", True, "sft")
    gradient_checkpointing = bool_value(section, "gradient_checkpointing", True, "sft")
    completion_only_loss = bool_value(section, "completion_only_loss", True, "sft")
    assistant_only_loss = bool_value(section, "assistant_only_loss", True, "sft")
    packing = bool_value(section, "packing", False, "sft")
    if not bf16 or not gradient_checkpointing:
        raise TrainingConfigurationError(
            "sft.bf16 and sft.gradient_checkpointing must remain true in V1"
        )
    if not completion_only_loss or not assistant_only_loss:
        raise TrainingConfigurationError(
            "sft.completion_only_loss and sft.assistant_only_loss must remain true "
            "to avoid training on prompts and tool observations"
        )
    if packing:
        raise TrainingConfigurationError(
            "sft.packing must remain false until conversational loss masks are verified"
        )

    dataset = inspect_sft_dataset(dataset_path)
    eval_dataset = inspect_sft_dataset(eval_path) if eval_path is not None else None
    recipe["stage"] = "sft"
    recipe["sft"] = {
        "dataset": dataset,
        "eval_dataset": eval_dataset,
        "max_length": max_length,
        "per_device_train_batch_size": batch_size,
        # Transformers defaults evaluation to eight rows per device. Keep
        # optional validation inside the same 24 GB single-GPU envelope.
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": accumulation,
        "effective_batch_size": effective_batch_size,
        "learning_rate": learning_rate,
        "num_train_epochs": epochs,
        "bf16": bf16,
        "tf32": tf32,
        "gradient_checkpointing": gradient_checkpointing,
        "completion_only_loss": completion_only_loss,
        "assistant_only_loss": assistant_only_loss,
        "packing": packing,
        "seed": int_value(section, "seed", 42, "sft", minimum=0),
        "logging_steps": int_value(section, "logging_steps", 5, "sft"),
        "save_steps": int_value(section, "save_steps", 50, "sft"),
        "save_total_limit": int_value(section, "save_total_limit", 2, "sft"),
    }
    validate_fresh_output_directory(Path(recipe["output_dir"]))
    return as_serializable(recipe)


def _load_json_dataset(load_dataset: Any, description: Mapping[str, Any]) -> Any:
    # Re-hash at the last boundary before Datasets opens the path.  A recipe is
    # provenance only if mutation after dry-run is rejected at execution time.
    verify_dataset_identity(description)
    return load_dataset("json", data_files=description["path"], split="train")


def run_sft(
    config: Mapping[str, Any],
    *,
    project_root: Path,
    output_dir: Path,
    dry_run: bool = False,
    on_event: TrainingEventCallback | None = None,
) -> dict[str, Any]:
    """Validate and optionally execute text-only 9B QLoRA SFT.

    A dry run performs the complete static validation path, including checking
    every dataset record, without importing PyTorch or Hugging Face modules.
    """

    recipe = resolve_sft_recipe(config, project_root=project_root, output_dir=output_dir)
    if dry_run:
        return {"status": "dry_run", "dry_run": True, "recipe": recipe}

    require_materialized_model(recipe["model"])
    installed_versions = validate_reference_dependencies()

    # Heavy imports stay behind validation and dry-run handling by design.
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            TrainerCallback,
        )
        from trl import SFTConfig, SFTTrainer
        from trl.chat_template_utils import get_training_chat_template
    except (ImportError, OSError) as error:
        raise TrainingDependencyError(
            "The pinned training packages are installed but could not be imported. "
            "Verify the WSL/CUDA PyTorch and bitsandbytes installation: "
            f"{error}"
        ) from error

    runtime = configure_vram_budget(torch, recipe["refinement"])
    destination = claim_fresh_output_directory(Path(recipe["output_dir"]))
    model_recipe = recipe["model"]
    qlora = recipe["qlora"]
    stage = recipe["sft"]

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_recipe["id"],
            revision=model_recipe["revision"],
            cache_dir=model_recipe["cache_dir"],
            local_files_only=True,
            trust_remote_code=False,
        )
        patched_template = get_training_chat_template(processing_class=tokenizer)
        if patched_template is not None:
            tokenizer.chat_template = patched_template
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Validate every complete prompt/answer before allocating the 9B model.
        # Static character limits are only a prefilter; tokenizer length is the
        # boundary that prevents SFT from silently dropping part of an answer.
        validate_sft_token_lengths(
            tokenizer,
            Path(stage["dataset"]["path"]),
            stage["max_length"],
        )

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=qlora["quant_type"],
            bnb_4bit_use_double_quant=qlora["double_quant"],
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            model_recipe["id"],
            revision=model_recipe["revision"],
            cache_dir=model_recipe["cache_dir"],
            local_files_only=True,
            dtype=torch.bfloat16,
            quantization_config=quantization_config,
            device_map={"": 0},
            max_memory=model_max_memory(recipe["refinement"]),
            trust_remote_code=False,
        )
        if base_model.__class__.__name__ != SUPPORTED_MODEL_CLASS:
            raise TrainingRuntimeError(
                "Text-only model guard failed: expected "
                f"{SUPPORTED_MODEL_CLASS}, loaded {base_model.__class__.__name__}. "
                "The vision-language model must not be used in V1."
            )
        base_model.config.use_cache = False
        base_model = prepare_model_for_kbit_training(
            base_model,
            use_gradient_checkpointing=stage["gradient_checkpointing"],
        )
        model = get_peft_model(
            base_model,
            LoraConfig(
                base_model_name_or_path=model_recipe["id"],
                revision=model_recipe["revision"],
                task_type=qlora["task_type"],
                target_modules=qlora["target_modules"],
                r=qlora["rank"],
                lora_alpha=qlora["alpha"],
                lora_dropout=qlora["dropout"],
                bias=qlora["bias"],
            ),
        )
        trainable_parameters = assert_adapter_only(model, stage="SFT")

        train_dataset = _load_json_dataset(load_dataset, stage["dataset"])
        eval_dataset = (
            _load_json_dataset(load_dataset, stage["eval_dataset"])
            if stage["eval_dataset"] is not None
            else None
        )
        training_args = SFTConfig(
            output_dir=str(destination),
            max_length=stage["max_length"],
            per_device_train_batch_size=stage["per_device_train_batch_size"],
            per_device_eval_batch_size=stage["per_device_eval_batch_size"],
            gradient_accumulation_steps=stage["gradient_accumulation_steps"],
            learning_rate=stage["learning_rate"],
            num_train_epochs=stage["num_train_epochs"],
            bf16=stage["bf16"],
            tf32=stage["tf32"],
            gradient_checkpointing=stage["gradient_checkpointing"],
            completion_only_loss=stage["completion_only_loss"],
            assistant_only_loss=stage["assistant_only_loss"],
            packing=stage["packing"],
            use_cache=False,
            seed=stage["seed"],
            data_seed=stage["seed"],
            logging_steps=stage["logging_steps"],
            save_strategy="steps",
            save_steps=stage["save_steps"],
            save_total_limit=stage["save_total_limit"],
            eval_strategy="steps" if eval_dataset is not None else "no",
            eval_steps=stage["save_steps"] if eval_dataset is not None else None,
            report_to="none",
        )
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            # This callback publishes only numeric values that the trainer
            # actually logged; it never reads batches, prompts, or tokens.
            callbacks=[
                make_trainer_log_callback(
                    TrainerCallback,
                    stage="sft",
                    on_event=on_event,
                    torch_module=torch,
                    vram_limit_gib=recipe["refinement"]["vram"]["max_gib"],
                )
            ],
        )
        train_output = trainer.train()
        trainer.save_model(str(destination))
        tokenizer.save_pretrained(str(destination))
        # PEFT's adapter_config.json is the handoff contract for GRPO.  Refuse
        # to publish an adapter if the exact base snapshot was not persisted.
        verify_saved_adapter_provenance(
            destination,
            model_recipe["id"],
            model_recipe["revision"],
        )
        (destination / "resolved_recipe.json").write_text(
            json.dumps(recipe, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        mark_output_directory_complete(destination)
    except TrainingRuntimeError:
        raise
    except torch.cuda.OutOfMemoryError as error:
        raise TrainingRuntimeError(
            "SFT exhausted GPU memory. Keep batch size at 1 and reduce sft.max_length "
            "before changing the validated QLoRA settings."
        ) from error
    except Exception as error:
        raise TrainingRuntimeError(f"SFT failed after recipe validation: {error}") from error

    metrics = as_serializable(getattr(train_output, "metrics", {}))
    return {
        "status": "completed",
        "dry_run": False,
        "stage": "sft",
        "output_dir": str(destination),
        "recipe": recipe,
        "dependencies": installed_versions,
        "runtime": runtime,
        "trainable_adapter_parameters": trainable_parameters,
        "metrics": metrics,
    }
