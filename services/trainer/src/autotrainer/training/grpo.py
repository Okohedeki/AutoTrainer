"""Guarded text-only GRPO from the selected base or a compatible LoRA adapter."""

from __future__ import annotations

from collections import defaultdict
import json
from math import isfinite
from numbers import Real
from pathlib import Path
from typing import Any, Mapping

from ..model_cache import require_materialized_model
from .experiment import (
    PhaseProfiler,
    build_training_receipt,
    write_training_receipt,
)
from .telemetry import TrainingEventCallback, make_trainer_log_callback
from .preflight import run_grpo_environment_canary
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
    choice_value,
    configure_vram_budget,
    float_value,
    get_section,
    import_factory,
    inspect_adapter,
    inspect_grpo_dataset,
    int_value,
    mark_output_directory_complete,
    model_max_memory,
    resolve_input_directory,
    resolve_input_file,
    resolve_stage_optimization,
    string_value,
    validate_factory_path,
    validate_fresh_output_directory,
    validate_reference_dependencies,
    verify_adapter_tree_identity,
    verify_dataset_identity,
    verify_effective_attention_backend,
    verify_saved_adapter_provenance,
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
        raise TrainingConfigurationError(
            "GRPO is disabled for the selected training recipe"
        )
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
    sft_enabled = (
        not isinstance(sft_section, Mapping)
        or sft_section.get("enabled", True) is not False
    )
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
    environment_backend = choice_value(
        environment_section,
        "backend",
        "docker",
        {"docker", "podman"},
        "environment",
    )
    environment_image = string_value(
        environment_section,
        "image",
        None,
        "environment",
    )
    batch_size = int_value(section, "per_device_train_batch_size", 1, "grpo")
    accumulation = int_value(section, "gradient_accumulation_steps", 2, "grpo")
    num_generations = int_value(section, "num_generations", 2, "grpo", minimum=2)
    calibration_generations = int_value(
        section,
        "calibration_generations",
        4,
        "grpo",
        minimum=4,
    )
    if calibration_generations % num_generations:
        raise TrainingConfigurationError(
            "grpo.calibration_generations must be divisible by grpo.num_generations; "
            f"got {calibration_generations} and {num_generations}"
        )
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

    max_completion_length = int_value(section, "max_completion_length", 2048, "grpo")
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
    gradient_checkpointing = bool_value(section, "gradient_checkpointing", True, "grpo")
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
        "backend": environment_backend,
        "image": environment_image,
        "network_access": False,
        "contract": {
            "factory_arguments": 0,
            "reset_receives_dataset_row": True,
            "reward_method": "get_reward",
            # V1 readiness requires auditable gate evidence, not just a scalar.
            "result_attribute": "last_result",
        },
    }
    recipe["grpo"] = {
        "dataset": dataset,
        "eval_dataset": eval_dataset,
        "start_from": start_from,
        # Compatibility projection for clients that still display this field.
        "sft_adapter": adapter,
        "per_device_train_batch_size": batch_size,
        # TRL's evaluation sampler emits one contiguous reward-normalization
        # group per prompt.  Keep the whole group in a single batch: smaller
        # batches fail when TRL reshapes rewards by num_generations, while the
        # inherited Transformers default of eight can OOM a 9B QLoRA.
        "per_device_eval_batch_size": num_generations,
        "gradient_accumulation_steps": accumulation,
        "effective_batch_size": effective_batch_size,
        "num_generations": num_generations,
        "calibration_generations": calibration_generations,
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
        **resolve_stage_optimization(section, "grpo"),
    }
    validate_fresh_output_directory(Path(recipe["output_dir"]))
    return as_serializable(recipe)


def _load_json_dataset(load_dataset: Any, description: Mapping[str, Any]) -> Any:
    # The static recipe locks exact bytes; enforce that lock immediately before
    # the dataset library receives the path.
    verify_dataset_identity(description)
    return load_dataset("json", data_files=description["path"], split="train")


def _bind_environment_image_identity(dataset: Any, runtime_reference: str) -> Any:
    """Inject the canary-resolved image into every row handed to TRL."""

    map_dataset = getattr(dataset, "map", None)
    if not callable(map_dataset):
        raise TrainingRuntimeError(
            "loaded GRPO dataset cannot bind immutable container image identity"
        )
    return map_dataset(
        lambda _row: {"environment_image_identity": runtime_reference},
        desc="Binding immutable rollout image",
    )


def _summarize_starting_policy_calibration(
    events: list[Mapping[str, Any]],
    *,
    task_ids: list[str],
    repetitions: int,
    num_generations: int,
) -> dict[str, Any]:
    """Require sampled reward spread before any optimizer step is allowed."""

    expected_tasks = set(task_ids)
    if not expected_tasks or len(expected_tasks) != len(task_ids):
        raise TrainingRuntimeError(
            "starting-policy calibration requires unique executable task ids"
        )
    grouped: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    observed_events: dict[str, dict[int, list[Mapping[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    gate_passes: dict[str, int] = defaultdict(int)
    for event in events:
        task_id = str(event.get("task_id", "")).strip()
        round_number = event.get("calibration_round")
        reward = event.get("reward")
        if task_id not in expected_tasks:
            raise TrainingRuntimeError(
                f"starting-policy calibration returned unknown task {task_id!r}"
            )
        if (
            isinstance(round_number, bool)
            or not isinstance(round_number, int)
            or not 1 <= round_number <= repetitions
        ):
            raise TrainingRuntimeError(
                "starting-policy calibration returned an invalid round number"
            )
        if (
            isinstance(reward, bool)
            or not isinstance(reward, Real)
            or not isfinite(float(reward))
        ):
            raise TrainingRuntimeError(
                f"starting-policy calibration returned invalid reward for {task_id!r}"
            )
        grouped[task_id][round_number].append(float(reward))
        observed_events[task_id][round_number].append(event)
        if event.get("hard_gate_passed") is True:
            gate_passes[task_id] += 1

    tasks: list[dict[str, Any]] = []
    blockers: list[str] = []
    for task_id in task_ids:
        round_summaries: list[dict[str, Any]] = []
        varied_rounds = 0
        all_rewards: list[float] = []
        changed_rollout_count = 0
        tool_call_count = 0
        tool_calls_by_name: dict[str, int] = defaultdict(int)
        patch_applied_count = 0
        patch_rejections_by_reason: dict[str, int] = defaultdict(int)
        for round_number in range(1, repetitions + 1):
            rewards = grouped[task_id].get(round_number, [])
            if len(rewards) != num_generations:
                blockers.append(
                    f"task {task_id!r} produced {len(rewards)} of {num_generations} "
                    f"required rewards in calibration round {round_number}"
                )
                continue
            reward_range = max(rewards) - min(rewards)
            if reward_range > 1e-8:
                varied_rounds += 1
            all_rewards.extend(rewards)
            round_events = observed_events[task_id].get(round_number, [])
            round_changed_count = sum(
                1
                for event in round_events
                if isinstance(event.get("changed_file_count"), int)
                and event["changed_file_count"] > 0
            )
            round_tool_count = sum(
                event.get("tool_call_count", 0)
                for event in round_events
                if isinstance(event.get("tool_call_count"), int)
                and not isinstance(event.get("tool_call_count"), bool)
            )
            round_tools: dict[str, int] = defaultdict(int)
            round_patch_applied = 0
            round_patch_rejections: dict[str, int] = defaultdict(int)
            for event in round_events:
                named_calls = event.get("tool_calls_by_name")
                if isinstance(named_calls, Mapping):
                    for name, count in named_calls.items():
                        if (
                            isinstance(name, str)
                            and isinstance(count, int)
                            and not isinstance(count, bool)
                            and count >= 0
                        ):
                            round_tools[name] += count
                            tool_calls_by_name[name] += count
                applied_count = event.get("patch_applied_count")
                if (
                    isinstance(applied_count, int)
                    and not isinstance(applied_count, bool)
                    and applied_count >= 0
                ):
                    round_patch_applied += applied_count
                    patch_applied_count += applied_count
                raw_rejections = event.get("patch_rejections_by_reason")
                if isinstance(raw_rejections, Mapping):
                    for name, count in raw_rejections.items():
                        if (
                            isinstance(name, str)
                            and isinstance(count, int)
                            and not isinstance(count, bool)
                            and count >= 0
                        ):
                            round_patch_rejections[name] += count
                            patch_rejections_by_reason[name] += count
            changed_rollout_count += round_changed_count
            tool_call_count += round_tool_count
            round_summaries.append(
                {
                    "round": round_number,
                    "reward_min": round(min(rewards), 6),
                    "reward_max": round(max(rewards), 6),
                    "reward_range": round(reward_range, 6),
                    "changed_rollout_count": round_changed_count,
                    "tool_call_count": round_tool_count,
                    "tool_calls_by_name": dict(sorted(round_tools.items())),
                    "patch_applied_count": round_patch_applied,
                    "patch_rejections_by_reason": dict(
                        sorted(round_patch_rejections.items())
                    ),
                }
            )
        if len(round_summaries) == repetitions and varied_rounds == 0:
            blockers.append(
                f"task {task_id!r} produced no within-group reward variation across "
                f"{repetitions} frozen starting-policy calibration rounds"
            )
        tasks.append(
            {
                "task_id": task_id,
                "rollout_count": len(all_rewards),
                "hard_gate_pass_count": gate_passes[task_id],
                "varied_round_count": varied_rounds,
                "changed_rollout_count": changed_rollout_count,
                "tool_call_count": tool_call_count,
                "tool_calls_by_name": dict(sorted(tool_calls_by_name.items())),
                "patch_applied_count": patch_applied_count,
                "patch_rejections_by_reason": dict(
                    sorted(patch_rejections_by_reason.items())
                ),
                "rounds": round_summaries,
            }
        )
    return {
        "status": "ready" if not blockers else "blocked",
        "policy_frozen": True,
        "optimizer_steps": 0,
        "task_count": len(task_ids),
        "num_generations": num_generations,
        "round_count": repetitions,
        "generations_per_task": repetitions * num_generations,
        "tasks": tasks,
        "blockers": blockers,
    }


def _save_grpo_processing_class(trainer: Any, fallback: Any, destination: Path) -> None:
    """Persist the exact tokenizer/template contract established by TRL.

    With an environment factory, GRPOTrainer must see the original supported
    Qwen template so it can add its response schema before deriving the training
    template. The trainer may retain that final template separately, so copy it
    onto the processing class that is saved with the adapter.
    """

    processing_class = getattr(trainer, "processing_class", None) or fallback
    save_pretrained = getattr(processing_class, "save_pretrained", None)
    if not callable(save_pretrained):
        raise TrainingRuntimeError(
            "GRPOTrainer did not retain a saveable tokenizer processing class"
        )
    trainer_template = getattr(trainer, "chat_template", None)
    if isinstance(trainer_template, str) and trainer_template.strip():
        processing_class.chat_template = trainer_template
    saved_template = getattr(processing_class, "chat_template", None)
    if not isinstance(saved_template, str) or not saved_template.strip():
        raise TrainingRuntimeError(
            "GRPOTrainer did not produce a persistent training chat template"
        )
    save_pretrained(str(destination))


def run_grpo(
    config: Mapping[str, Any],
    *,
    project_root: Path,
    output_dir: Path,
    dry_run: bool = False,
    on_event: TrainingEventCallback | None = None,
) -> dict[str, Any]:
    """Validate and optionally train a QLoRA policy with verified tool-use GRPO."""

    recipe = resolve_grpo_recipe(
        config, project_root=project_root, output_dir=output_dir
    )
    if dry_run:
        return {"status": "dry_run", "dry_run": True, "recipe": recipe}

    profiler = PhaseProfiler()
    require_materialized_model(recipe["model"])
    installed_versions = validate_reference_dependencies()
    profiler.checkpoint("preflight")

    # Exercise the actual repository snapshot, container gates, and hidden
    # verifier before allocating several gigabytes of model memory. Direct CLI
    # callers receive the same fail-closed contract as project Prepare.
    environment_canary = run_grpo_environment_canary(recipe)
    container_image = environment_canary.get("container_image")
    if (
        not isinstance(container_image, Mapping)
        or not str(container_image.get("runtime_reference", "")).strip()
    ):
        raise TrainingRuntimeError(
            "GRPO executable canary did not return immutable container image evidence"
        )
    runtime_image = str(container_image["runtime_reference"]).strip()
    recipe["environment"]["image_identity"] = dict(container_image)
    profiler.checkpoint("environment_canary")

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
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            TrainerCallback,
        )
        from trl import GRPOConfig, GRPOTrainer
    except (ImportError, OSError) as error:
        raise TrainingDependencyError(
            "The pinned training packages are installed but could not be imported. "
            "Verify the WSL/CUDA PyTorch and bitsandbytes installation: "
            f"{error}"
        ) from error

    runtime = configure_vram_budget(torch, recipe["refinement"])
    base_environment_factory = import_factory(recipe["environment"]["factory"])
    calibration_round: int | None = None
    calibration_events: list[dict[str, Any]] = []

    def environment_factory() -> Any:
        """Attach private phase observers without changing the TRL tool surface."""

        environment = base_environment_factory()
        setter = getattr(environment, "_set_episode_callback", None)
        if callable(setter):

            def observe_episode(event: Mapping[str, Any]) -> None:
                if calibration_round is not None:
                    if event.get("type") == "episode_scored":
                        calibration_events.append(
                            {
                                **dict(event),
                                "calibration_round": calibration_round,
                            }
                        )
                    return
                if on_event is not None:
                    on_event({"stage": "grpo", **dict(event)})

            setter(observe_episode)
        return environment

    destination = claim_fresh_output_directory(Path(recipe["output_dir"]))
    profiler.checkpoint("runtime_setup")
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
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        # GRPO batches variable-length prompts for generation; TRL requires
        # left padding so prompt endings line up with the generation boundary.
        tokenizer.padding_side = "left"
        profiler.checkpoint("tokenizer_setup")

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
            attn_implementation=model_recipe["attn_implementation"],
        )
        model_recipe["effective_attn_implementation"] = (
            verify_effective_attention_backend(
                base_model, model_recipe["attn_implementation"]
            )
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
        else:
            # Re-hash the complete adapter at the last boundary before PEFT
            # opens it.  Matching adapter_config.json alone cannot bind weights.
            verify_adapter_tree_identity(stage["start_from"]["adapter"])
            policy = PeftModel.from_pretrained(
                base_model,
                stage["start_from"]["adapter"]["path"],
                is_trainable=True,
            )
        trainable_parameters = assert_adapter_only(policy, stage="GRPO")
        profiler.checkpoint("model_adapter_setup")

        train_dataset = _bind_environment_image_identity(
            _load_json_dataset(load_dataset, stage["dataset"]),
            runtime_image,
        )
        eval_dataset = (
            _bind_environment_image_identity(
                _load_json_dataset(load_dataset, stage["eval_dataset"]),
                runtime_image,
            )
            if stage["eval_dataset"] is not None
            else None
        )
        training_args = GRPOConfig(
            output_dir=str(destination),
            per_device_train_batch_size=stage["per_device_train_batch_size"],
            per_device_eval_batch_size=stage["per_device_eval_batch_size"],
            gradient_accumulation_steps=stage["gradient_accumulation_steps"],
            learning_rate=stage["learning_rate"],
            optim=stage["optim"],
            lr_scheduler_type=stage["lr_scheduler_type"],
            warmup_steps=stage["warmup_steps"],
            weight_decay=stage["weight_decay"],
            max_grad_norm=stage["max_grad_norm"],
            use_liger_kernel=stage["use_liger_kernel"],
            num_train_epochs=stage["num_train_epochs"],
            max_steps=stage["max_steps"],
            bf16=stage["bf16"],
            tf32=stage["tf32"],
            gradient_checkpointing=stage["gradient_checkpointing"],
            # Transformers copies this value onto model.config. TRL disables
            # the cache explicitly for gradient-scored forward passes, while
            # autoregressive generation needs the KV cache to avoid quadratic
            # recomputation at every token.
            use_cache=True,
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
        telemetry_callback = make_trainer_log_callback(
            TrainerCallback,
            stage="grpo",
            on_event=on_event,
            torch_module=torch,
            vram_limit_gib=recipe["refinement"]["vram"]["max_gib"],
        )
        trainer = GRPOTrainer(
            model=policy,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            environment_factory=environment_factory,
            callbacks=[telemetry_callback],
        )
        profiler.checkpoint("trainer_setup")
        calibration_metrics: list[dict[str, Any]] = []
        calibration_repetitions = (
            stage["calibration_generations"] // stage["num_generations"]
        )
        for round_index in range(calibration_repetitions):
            calibration_round = round_index + 1
            if on_event is not None:
                on_event(
                    {
                        "type": "calibration_round_started",
                        "stage": "grpo",
                        "round": calibration_round,
                        "total_rounds": calibration_repetitions,
                    }
                )
            metrics = trainer.evaluate(
                eval_dataset=train_dataset,
                metric_key_prefix=f"starting_policy_calibration_{calibration_round}",
            )
            calibration_metrics.append(as_serializable(metrics))
            if on_event is not None:
                on_event(
                    {
                        "type": "calibration_round_completed",
                        "stage": "grpo",
                        "round": calibration_round,
                        "total_rounds": calibration_repetitions,
                    }
                )
        calibration_round = None
        canary_tasks = environment_canary.get("tasks")
        if not isinstance(canary_tasks, list):
            raise TrainingRuntimeError(
                "GRPO executable canary did not return task identities for calibration"
            )
        task_ids = [
            str(task.get("task_id", "")).strip()
            for task in canary_tasks
            if isinstance(task, Mapping)
        ]
        starting_policy_calibration = _summarize_starting_policy_calibration(
            calibration_events,
            task_ids=task_ids,
            repetitions=calibration_repetitions,
            num_generations=stage["num_generations"],
        )
        starting_policy_calibration.update(
            {
                "model": {
                    "id": model_recipe["id"],
                    "revision": model_recipe["revision"],
                },
                "dataset": {
                    key: stage["dataset"][key]
                    for key in ("sha256", "bytes", "record_count")
                },
                "container_image": dict(container_image),
                "trainer_metrics": calibration_metrics,
            }
        )
        (destination / "starting_policy_calibration.json").write_text(
            json.dumps(starting_policy_calibration, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if starting_policy_calibration["status"] != "ready":
            raise TrainingRuntimeError(
                "GRPO starting-policy calibration rejected the curriculum: "
                + str(starting_policy_calibration["blockers"][0])
            )
        profiler.checkpoint("starting_policy_calibration")
        train_output = trainer.train()
        profiler.checkpoint("training")
        trainer.save_model(str(destination))
        _save_grpo_processing_class(trainer, tokenizer, destination)
        verify_saved_adapter_provenance(
            destination,
            model_recipe["id"],
            model_recipe["revision"],
        )
        (destination / "resolved_recipe.json").write_text(
            json.dumps(recipe, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        profiler.checkpoint("artifact_save")
        metrics = as_serializable(getattr(train_output, "metrics", {}))
        telemetry = telemetry_callback.observed_summary()
        profile = profiler.summary()
        receipt = build_training_receipt(
            stage="grpo",
            recipe=recipe,
            dependencies=installed_versions,
            runtime=runtime,
            trainable_adapter_parameters=trainable_parameters,
            metrics=metrics,
            telemetry=telemetry,
            profile=profile,
        )
        receipt_path = write_training_receipt(destination, receipt)
        mark_output_directory_complete(destination)
    except TrainingRuntimeError:
        raise
    except torch.cuda.OutOfMemoryError as error:
        raise TrainingRuntimeError(
            "GRPO exhausted GPU memory. Keep generation and training batch sizes at 2 "
            "and reduce grpo.max_completion_length before changing QLoRA settings."
        ) from error
    except Exception as error:
        raise TrainingRuntimeError(
            f"GRPO failed after recipe validation: {error}"
        ) from error

    return {
        "status": "completed",
        "dry_run": False,
        "stage": "grpo",
        "output_dir": str(destination),
        "recipe": recipe,
        "dependencies": installed_versions,
        "runtime": runtime,
        "environment_canary": environment_canary,
        "starting_policy_calibration": starting_policy_calibration,
        "trainable_adapter_parameters": trainable_parameters,
        "metrics": metrics,
        "performance": {
            "profile": profile,
            "telemetry": telemetry,
            "receipt_path": str(receipt_path),
        },
    }
