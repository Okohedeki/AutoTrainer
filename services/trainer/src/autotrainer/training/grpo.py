"""Guarded text-only GRPO from the selected base or a compatible LoRA adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from ..model_cache import require_materialized_model
from .common import (
    SUPPORTED_MODEL_CLASS,
    TrainingConfigurationError,
    TrainingDependencyError,
    TrainingRuntimeError,
    as_serializable,
    base_recipe,
    bool_value,
    choice_value,
    float_value,
    get_section,
    import_factory,
    inspect_adapter,
    inspect_grpo_dataset,
    int_value,
    resolve_input_directory,
    resolve_input_file,
    validate_factory_path,
    validate_reference_dependencies,
    validate_single_gpu,
)


def resolve_grpo_recipe(
    config: Mapping[str, Any], *, project_root: Path, output_dir: Path
) -> dict[str, Any]:
    """Resolve a GRPO recipe without importing PyTorch or Hugging Face packages.

    ``grpo.start_from`` is explicit: ``base`` creates a fresh QLoRA policy for
    practice-only RL, while a path continues a verified compatible PEFT adapter.
    The legacy ``sft_adapter`` key remains readable for existing projects.
    """

    recipe = base_recipe(config, project_root=project_root, output_dir=output_dir)
    section = get_section(config, "grpo")
    if section.get("enabled", True) is False:
        raise TrainingConfigurationError("GRPO is disabled for the selected training recipe")
    environment_section = get_section(config, "environment")
    root = Path(recipe["project_root"])

    dataset_path = resolve_input_file(section.get("dataset"), root, "grpo.dataset")
    eval_value = section.get("eval_dataset")
    eval_path = (
        resolve_input_file(eval_value, root, "grpo.eval_dataset")
        if eval_value is not None
        else None
    )
    start_value = section.get("start_from", section.get("sft_adapter"))
    if not isinstance(start_value, str) or not start_value.strip():
        raise TrainingConfigurationError(
            "grpo.start_from must be 'base' or point to a compatible LoRA adapter"
        )
    start_value = start_value.strip()
    sft_section = config.get("sft", {})
    sft_enabled = not isinstance(sft_section, Mapping) or sft_section.get("enabled", True) is not False
    if start_value == "base" and sft_enabled:
        raise TrainingConfigurationError(
            "both-stage training requires GRPO to continue the SFT adapter, not base"
        )
    adapter_path = (
        None
        if start_value == "base"
        else resolve_input_directory(start_value, root, "grpo.start_from")
    )
    destination = Path(recipe["output_dir"])
    if adapter_path is not None and destination == adapter_path:
        raise TrainingConfigurationError(
            "output_dir must differ from grpo.start_from so the input adapter is not overwritten"
        )

    factory_path = validate_factory_path(environment_section.get("factory"))
    batch_size = int_value(section, "per_device_train_batch_size", 1, "grpo")
    accumulation = int_value(section, "gradient_accumulation_steps", 2, "grpo")
    num_generations = int_value(section, "num_generations", 2, "grpo", minimum=2)
    effective_batch_size = batch_size * accumulation
    if effective_batch_size % num_generations:
        raise TrainingConfigurationError(
            "grpo effective batch size must be divisible by grpo.num_generations; "
            f"got {batch_size} * {accumulation} = {effective_batch_size}, with "
            f"num_generations={num_generations}"
        )
    if effective_batch_size >= 32:
        raise TrainingConfigurationError(
            "grpo effective batch size must remain below 32 on the supported runtime; "
            f"got {effective_batch_size}"
        )

    generation_batch_size = int_value(
        section, "generation_batch_size", 2, "grpo", minimum=2
    )
    if generation_batch_size < num_generations:
        raise TrainingConfigurationError(
            "grpo.generation_batch_size must be at least grpo.num_generations"
        )
    if generation_batch_size % num_generations:
        raise TrainingConfigurationError(
            "grpo.generation_batch_size must be divisible by grpo.num_generations"
        )

    max_completion_length = int_value(
        section, "max_completion_length", 2048, "grpo"
    )
    if max_completion_length > 4096:
        raise TrainingConfigurationError(
            "grpo.max_completion_length must be <= 4096 on the supported 24 GB runtime"
        )
    epochs = float_value(
        section,
        "num_train_epochs",
        1.0,
        "grpo",
        minimum=0.0,
        minimum_inclusive=False,
    )
    max_steps = int_value(section, "max_steps", 100, "grpo", minimum=1)
    learning_rate = float_value(
        section,
        "learning_rate",
        1.0e-5,
        "grpo",
        minimum=0.0,
        maximum=1.0e-3,
        minimum_inclusive=False,
    )
    beta = float_value(section, "beta", 0.0, "grpo", minimum=0.0)
    loss_type = choice_value(section, "loss_type", "dapo", {"dapo"}, "grpo")
    bf16 = bool_value(section, "bf16", True, "grpo")
    tf32 = bool_value(section, "tf32", True, "grpo")
    gradient_checkpointing = bool_value(
        section, "gradient_checkpointing", True, "grpo"
    )
    use_vllm = bool_value(section, "use_vllm", False, "grpo")
    if not bf16 or not gradient_checkpointing:
        raise TrainingConfigurationError(
            "grpo.bf16 and grpo.gradient_checkpointing must remain true in V1"
        )
    if use_vllm:
        raise TrainingConfigurationError(
            "grpo.use_vllm must remain false in V1; colocated vLLM is not in the "
            "validated single-GPU memory budget"
        )
    if beta != 0.0:
        raise TrainingConfigurationError(
            "grpo.beta must remain 0.0 in the validated memory-efficient recipe"
        )

    dataset = inspect_grpo_dataset(dataset_path)
    eval_dataset = inspect_grpo_dataset(eval_path) if eval_path is not None else None
    adapter = (
        inspect_adapter(
            adapter_path,
            expected_model_id=recipe["model"]["id"],
            expected_revision=recipe["model"]["revision"],
        )
        if adapter_path is not None
        else None
    )
    start_from = (
        {"type": "adapter", "adapter": adapter}
        if adapter is not None
        else {"type": "base", "adapter": None}
    )
    recipe["stage"] = "grpo"
    recipe["environment"] = {
        "factory": factory_path,
        "network_access": False,
        "contract": {
            "factory_arguments": 0,
            "reset_receives_dataset_row": True,
            "reward_method": "get_reward",
        },
    }
    recipe["grpo"] = {
        "dataset": dataset,
        "eval_dataset": eval_dataset,
        "start_from": start_from,
        # Compatibility projection for clients that still display this field.
        "sft_adapter": adapter,
        "per_device_train_batch_size": batch_size,
        "gradient_accumulation_steps": accumulation,
        "effective_batch_size": effective_batch_size,
        "num_generations": num_generations,
        "generation_batch_size": generation_batch_size,
        "max_completion_length": max_completion_length,
        "max_tool_calling_iterations": int_value(
            section, "max_tool_calling_iterations", 8, "grpo"
        ),
        "learning_rate": learning_rate,
        "num_train_epochs": epochs,
        "max_steps": max_steps,
        "beta": beta,
        "loss_type": loss_type,
        "bf16": bf16,
        "tf32": tf32,
        "gradient_checkpointing": gradient_checkpointing,
        "use_vllm": use_vllm,
        "temperature": float_value(
            section,
            "temperature",
            0.7,
            "grpo",
            minimum=0.0,
            maximum=2.0,
            minimum_inclusive=False,
        ),
        "top_p": float_value(
            section,
            "top_p",
            0.8,
            "grpo",
            minimum=0.0,
            maximum=1.0,
            minimum_inclusive=False,
        ),
        "top_k": int_value(section, "top_k", 20, "grpo", minimum=0),
        "seed": int_value(section, "seed", 42, "grpo", minimum=0),
        "logging_steps": int_value(section, "logging_steps", 5, "grpo"),
        "save_steps": int_value(section, "save_steps", 50, "grpo"),
        "save_total_limit": int_value(section, "save_total_limit", 2, "grpo"),
    }
    return as_serializable(recipe)


def _load_json_dataset(load_dataset: Any, description: Mapping[str, Any]) -> Any:
    return load_dataset("json", data_files=description["path"], split="train")


def run_grpo(
    config: Mapping[str, Any],
    *,
    project_root: Path,
    output_dir: Path,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Validate and optionally train a QLoRA policy with verified tool-use GRPO."""

    recipe = resolve_grpo_recipe(config, project_root=project_root, output_dir=output_dir)
    if dry_run:
        return {"status": "dry_run", "dry_run": True, "recipe": recipe}

    require_materialized_model(recipe["model"])
    installed_versions = validate_reference_dependencies()

    # Heavy imports stay behind static validation and dry-run handling.
    try:
        import torch
        from datasets import load_dataset
        from peft import (
            LoraConfig,
            PeftModel,
            get_peft_model,
            prepare_model_for_kbit_training,
        )
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import GRPOConfig, GRPOTrainer
        from trl.chat_template_utils import get_training_chat_template
    except (ImportError, OSError) as error:
        raise TrainingDependencyError(
            "The pinned training packages are installed but could not be imported. "
            "Verify the WSL/CUDA PyTorch and bitsandbytes installation: "
            f"{error}"
        ) from error

    runtime = validate_single_gpu(torch)
    environment_factory = import_factory(recipe["environment"]["factory"])
    destination = Path(recipe["output_dir"])
    destination.mkdir(parents=True, exist_ok=True)
    model_recipe = recipe["model"]
    qlora = recipe["qlora"]
    stage = recipe["grpo"]

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
        # GRPO batches variable-length prompts for generation; TRL requires
        # left padding so prompt endings line up with the generation boundary.
        tokenizer.padding_side = "left"

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
        if stage["start_from"]["type"] == "base":
            # Practice-only RL still trains a small LoRA policy; "base" means
            # no supervised warm-up, never full-parameter 4-bit training.
            policy = get_peft_model(
                base_model,
                LoraConfig(
                    task_type=qlora["task_type"],
                    target_modules=qlora["target_modules"],
                    r=qlora["rank"],
                    lora_alpha=qlora["alpha"],
                    lora_dropout=qlora["dropout"],
                    bias=qlora["bias"],
                ),
            )
        else:
            policy = PeftModel.from_pretrained(
                base_model,
                stage["start_from"]["adapter"]["path"],
                is_trainable=True,
            )
        trainable_parameters = sum(
            parameter.numel() for parameter in policy.parameters() if parameter.requires_grad
        )
        if trainable_parameters == 0:
            raise TrainingRuntimeError(
                "The GRPO policy has no trainable adapter parameters"
            )

        train_dataset = _load_json_dataset(load_dataset, stage["dataset"])
        eval_dataset = (
            _load_json_dataset(load_dataset, stage["eval_dataset"])
            if stage["eval_dataset"] is not None
            else None
        )
        training_args = GRPOConfig(
            output_dir=str(destination),
            per_device_train_batch_size=stage["per_device_train_batch_size"],
            gradient_accumulation_steps=stage["gradient_accumulation_steps"],
            learning_rate=stage["learning_rate"],
            num_train_epochs=stage["num_train_epochs"],
            max_steps=stage["max_steps"],
            bf16=stage["bf16"],
            tf32=stage["tf32"],
            gradient_checkpointing=stage["gradient_checkpointing"],
            use_cache=False,
            seed=stage["seed"],
            data_seed=stage["seed"],
            logging_steps=stage["logging_steps"],
            save_strategy="steps",
            save_steps=stage["save_steps"],
            save_total_limit=stage["save_total_limit"],
            eval_strategy="steps" if eval_dataset is not None else "no",
            eval_steps=stage["save_steps"] if eval_dataset is not None else None,
            num_generations=stage["num_generations"],
            generation_batch_size=stage["generation_batch_size"],
            max_completion_length=stage["max_completion_length"],
            max_tool_calling_iterations=stage["max_tool_calling_iterations"],
            beta=stage["beta"],
            loss_type=stage["loss_type"],
            use_vllm=stage["use_vllm"],
            temperature=stage["temperature"],
            top_p=stage["top_p"],
            top_k=stage["top_k"],
            chat_template_kwargs={"enable_thinking": False},
            remove_unused_columns=False,
            report_to="none",
        )
        trainer = GRPOTrainer(
            model=policy,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            environment_factory=environment_factory,
        )
        train_output = trainer.train()
        trainer.save_model(str(destination))
        tokenizer.save_pretrained(str(destination))
        (destination / "resolved_recipe.json").write_text(
            json.dumps(recipe, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except TrainingRuntimeError:
        raise
    except torch.cuda.OutOfMemoryError as error:
        raise TrainingRuntimeError(
            "GRPO exhausted GPU memory. Keep generation and training batch sizes at 2 "
            "and reduce grpo.max_completion_length before changing QLoRA settings."
        ) from error
    except Exception as error:
        raise TrainingRuntimeError(f"GRPO failed after recipe validation: {error}") from error

    metrics = as_serializable(getattr(train_output, "metrics", {}))
    return {
        "status": "completed",
        "dry_run": False,
        "stage": "grpo",
        "output_dir": str(destination),
        "recipe": recipe,
        "dependencies": installed_versions,
        "runtime": runtime,
        "trainable_adapter_parameters": trainable_parameters,
        "metrics": metrics,
    }
