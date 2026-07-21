"""Shared validation and runtime guards for local single-GPU training.

This module intentionally depends only on the Python standard library.  The
CUDA and Hugging Face stack is imported by the stage runners after a recipe has
been validated, so configuration inspection remains useful on machines that do
not have the training extras installed.
"""

from __future__ import annotations

import hashlib
import hmac
import importlib
import inspect
import json
import re
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, Mapping

from ..sources import normalize_sft_record


SUPPORTED_SCHEMA_VERSION = 1
SUPPORTED_MODEL_ID = "Qwen/Qwen3.5-9B"
SUPPORTED_MODEL_CLASS = "Qwen3_5ForCausalLM"
IMMUTABLE_REVISION = re.compile(r"[0-9a-fA-F]{40,64}")

REFERENCE_DEPENDENCIES: dict[str, str] = {
    "torch": "2.13.0",
    "transformers": "5.13.1",
    "trl": "1.8.0",
    "peft": "0.19.1",
    "accelerate": "1.14.0",
    "datasets": "5.0.0",
    "bitsandbytes": "0.49.2",
    # TRL's environment_factory tool loop imports this at trainer startup.
    "jmespath": "1.1.0",
}


class TrainingConfigurationError(ValueError):
    """Raised before training when a recipe is unsafe or inconsistent."""


class TrainingDependencyError(RuntimeError):
    """Raised when the validated training dependency matrix is unavailable."""


class TrainingRuntimeError(RuntimeError):
    """Raised when the host cannot safely execute the requested training run."""


def as_serializable(value: Any) -> Any:
    """Convert nested runtime values into JSON-compatible primitives."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): as_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [as_serializable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return as_serializable(value.item())
        except (TypeError, ValueError):
            pass
    return str(value)


def validate_root_config(config: Mapping[str, Any]) -> None:
    if not isinstance(config, Mapping):
        raise TrainingConfigurationError("config must be a mapping")
    schema_version = config.get("schema_version", SUPPORTED_SCHEMA_VERSION)
    if schema_version != SUPPORTED_SCHEMA_VERSION:
        raise TrainingConfigurationError(
            f"schema_version must be {SUPPORTED_SCHEMA_VERSION!r}; got {schema_version!r}"
        )


def get_section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = config.get(name, {})
    if not isinstance(value, Mapping):
        raise TrainingConfigurationError(f"{name} must be a mapping")
    return value


def bool_value(section: Mapping[str, Any], key: str, default: bool, prefix: str) -> bool:
    value = section.get(key, default)
    if not isinstance(value, bool):
        raise TrainingConfigurationError(f"{prefix}.{key} must be a boolean")
    return value


def int_value(
    section: Mapping[str, Any],
    key: str,
    default: int,
    prefix: str,
    *,
    minimum: int = 1,
) -> int:
    value = section.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise TrainingConfigurationError(
            f"{prefix}.{key} must be an integer greater than or equal to {minimum}"
        )
    return value


def float_value(
    section: Mapping[str, Any],
    key: str,
    default: float,
    prefix: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_inclusive: bool = True,
) -> float:
    value = section.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrainingConfigurationError(f"{prefix}.{key} must be a number")
    result = float(value)
    if minimum is not None:
        too_small = result < minimum if minimum_inclusive else result <= minimum
        if too_small:
            comparison = ">=" if minimum_inclusive else ">"
            raise TrainingConfigurationError(
                f"{prefix}.{key} must be {comparison} {minimum}"
            )
    if maximum is not None and result > maximum:
        raise TrainingConfigurationError(f"{prefix}.{key} must be <= {maximum}")
    return result


def string_value(
    section: Mapping[str, Any], key: str, default: str | None, prefix: str
) -> str:
    value = section.get(key, default)
    if not isinstance(value, str) or not value.strip():
        raise TrainingConfigurationError(f"{prefix}.{key} must be a non-empty string")
    return value.strip()


def choice_value(
    section: Mapping[str, Any],
    key: str,
    default: str,
    choices: set[str],
    prefix: str,
) -> str:
    value = string_value(section, key, default, prefix)
    if value not in choices:
        options = ", ".join(sorted(choices))
        raise TrainingConfigurationError(f"{prefix}.{key} must be one of: {options}")
    return value


def resolve_project_root(project_root: Path) -> Path:
    root = Path(project_root).expanduser().resolve()
    if not root.is_dir():
        raise TrainingConfigurationError(f"project_root is not a directory: {root}")
    return root


def resolve_output_dir(output_dir: Path, project_root: Path) -> Path:
    path = Path(output_dir).expanduser()
    if not path.is_absolute():
        path = project_root / path
    path = path.resolve()
    if path.exists() and not path.is_dir():
        raise TrainingConfigurationError(f"output_dir is not a directory: {path}")
    return path


def validate_fresh_output_directory(path: Path) -> None:
    """Statically reject output that already contains another run's bytes."""

    destination = Path(path)
    if not destination.exists():
        return
    if not destination.is_dir():
        raise TrainingConfigurationError(f"output_dir is not a directory: {destination}")
    try:
        has_entries = next(destination.iterdir(), None) is not None
    except OSError as error:
        raise TrainingConfigurationError(
            f"Could not inspect output_dir before training: {destination}: {error}"
        ) from error
    if has_entries:
        raise TrainingConfigurationError(
            "output_dir must be empty for an immutable training run; "
            f"choose a fresh path instead of reusing {destination}"
        )


def claim_fresh_output_directory(path: Path) -> Path:
    """Claim an empty destination so a run can never reuse stale checkpoints.

    Existing empty directories are accepted because project setup may create
    them in advance.  The exclusive claim file closes the race between that
    emptiness check and the trainer's first checkpoint write.  A failed run is
    intentionally left claimed; callers must choose a new run directory rather
    than accidentally resume a partial adapter with different inputs.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination.mkdir()
    except FileExistsError:
        if not destination.is_dir():
            raise TrainingConfigurationError(
                f"output_dir is not a directory: {destination}"
            )
        try:
            has_entries = next(destination.iterdir(), None) is not None
        except OSError as error:
            raise TrainingRuntimeError(
                f"Could not inspect output_dir before training: {destination}: {error}"
            ) from error
        if has_entries:
            raise TrainingConfigurationError(
                "output_dir must be empty for an immutable training run; "
                f"choose a fresh path instead of reusing {destination}"
            )

    claim = destination / ".autotrainer-run-claim.json"
    payload = {
        "policy": "immutable-fresh-run-v1",
        "status": "running",
    }
    try:
        # Exclusive creation makes simultaneous direct API callers fail closed,
        # even when they bypass the CLI's project and GPU leases.
        with claim.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError as error:
        raise TrainingConfigurationError(
            "output_dir is already claimed by another or an incomplete run; "
            f"choose a fresh path instead of reusing {destination}"
        ) from error
    return destination


def mark_output_directory_complete(path: Path) -> None:
    """Atomically mark a claimed output as a completed immutable artifact."""

    destination = Path(path)
    claim = destination / ".autotrainer-run-claim.json"
    if not claim.is_file():
        raise TrainingRuntimeError(f"Training output lost its run claim: {claim}")
    temporary = claim.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(
            {"policy": "immutable-fresh-run-v1", "status": "completed"},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(claim)


def resolve_input_file(value: Any, project_root: Path, field: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise TrainingConfigurationError(f"{field} must be a local JSON or JSONL path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    path = path.resolve()
    if not path.is_file():
        raise TrainingConfigurationError(f"{field} does not exist or is not a file: {path}")
    if path.suffix.lower() not in {".json", ".jsonl"}:
        raise TrainingConfigurationError(f"{field} must end in .json or .jsonl: {path}")
    return path


def resolve_input_directory(value: Any, project_root: Path, field: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise TrainingConfigurationError(f"{field} must be a local directory path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    path = path.resolve()
    if not path.is_dir():
        raise TrainingConfigurationError(f"{field} does not exist or is not a directory: {path}")
    return path


def resolve_model_recipe(config: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    section = get_section(config, "model")
    model_id = string_value(section, "id", SUPPORTED_MODEL_ID, "model")
    if model_id != SUPPORTED_MODEL_ID:
        raise TrainingConfigurationError(
            f"V1 supports only the text backbone of {SUPPORTED_MODEL_ID}; got {model_id}"
        )
    revision = string_value(section, "revision", "main", "model")
    text_only = bool_value(section, "text_only", True, "model")
    trust_remote_code = bool_value(section, "trust_remote_code", False, "model")
    thinking = bool_value(section, "thinking", False, "model")
    dtype = choice_value(section, "dtype", "bfloat16", {"bfloat16"}, "model")
    attn_implementation = choice_value(
        section,
        "attn_implementation",
        "sdpa",
        {"eager", "sdpa"},
        "model",
    )
    if not text_only:
        raise TrainingConfigurationError("model.text_only must remain true in V1")
    if trust_remote_code:
        raise TrainingConfigurationError("model.trust_remote_code must remain false in V1")
    if thinking:
        raise TrainingConfigurationError(
            "model.thinking must remain false in the supported 24 GB recipe"
        )
    warnings: list[str] = []
    if not IMMUTABLE_REVISION.fullmatch(revision):
        warnings.append(
            "model.revision is mutable; replace it with a resolved Hugging Face commit SHA "
            "before a benchmark or published run"
        )
    return (
        {
            "id": model_id,
            "revision": revision,
            "class": SUPPORTED_MODEL_CLASS,
            "text_only": True,
            "trust_remote_code": False,
            "thinking": False,
            "dtype": dtype,
            "attn_implementation": attn_implementation,
        },
        warnings,
    )


def resolve_stage_optimization(
    section: Mapping[str, Any], prefix: str
) -> dict[str, Any]:
    """Resolve the explicit, behavior-preserving Transformers optimizer baseline."""

    use_liger_kernel = bool_value(section, "use_liger_kernel", False, prefix)
    if use_liger_kernel:
        raise TrainingConfigurationError(
            f"{prefix}.use_liger_kernel must remain false until AutoTrainer ships a "
            "pinned Qwen3.5 QLoRA numerical contract for Liger"
        )
    return {
        "optim": choice_value(
            section,
            "optim",
            "adamw_torch_fused",
            {"adamw_torch", "adamw_torch_fused"},
            prefix,
        ),
        "lr_scheduler_type": choice_value(
            section,
            "lr_scheduler_type",
            "linear",
            {"cosine", "linear"},
            prefix,
        ),
        "warmup_steps": int_value(section, "warmup_steps", 0, prefix, minimum=0),
        "weight_decay": float_value(
            section, "weight_decay", 0.0, prefix, minimum=0.0, maximum=1.0
        ),
        "max_grad_norm": float_value(
            section, "max_grad_norm", 1.0, prefix, minimum=0.0, maximum=100.0
        ),
        "use_liger_kernel": False,
    }


def verify_effective_attention_backend(model: Any, requested: str) -> str:
    """Refuse a silent attention fallback that would invalidate comparisons."""

    config = getattr(model, "config", None)
    effective = getattr(config, "_attn_implementation", None)
    if effective != requested:
        raise TrainingRuntimeError(
            "The loaded model did not honor model.attn_implementation: "
            f"requested={requested!r}, effective={effective!r}"
        )
    return str(effective)


def resolve_qlora_recipe(config: Mapping[str, Any]) -> dict[str, Any]:
    section = get_section(config, "qlora")
    model_section = get_section(config, "model")
    quantization = model_section.get("quantization", {})
    if not isinstance(quantization, Mapping):
        raise TrainingConfigurationError("model.quantization must be a mapping")
    method = quantization.get("method", "bitsandbytes-4bit")
    if method != "bitsandbytes-4bit":
        raise TrainingConfigurationError("model.quantization.method must be bitsandbytes-4bit")
    load_in_4bit = bool_value(section, "load_in_4bit", True, "qlora")
    quant_type = choice_value(
        section,
        "quant_type",
        str(quantization.get("quant_type", "nf4")),
        {"nf4"},
        "qlora",
    )
    double_quant = bool_value(
        section,
        "double_quant",
        bool(quantization.get("double_quant", True)),
        "qlora",
    )
    compute_dtype = choice_value(
        section,
        "compute_dtype",
        str(quantization.get("compute_dtype", "bfloat16")),
        {"bfloat16"},
        "qlora",
    )
    rank = int_value(section, "rank", 32, "qlora", minimum=1)
    alpha = int_value(section, "alpha", 32, "qlora", minimum=1)
    dropout = float_value(
        section, "dropout", 0.0, "qlora", minimum=0.0, maximum=1.0
    )
    bias = choice_value(section, "bias", "none", {"none"}, "qlora")
    target_modules = choice_value(
        section, "target_modules", "all-linear", {"all-linear"}, "qlora"
    )
    if not load_in_4bit:
        raise TrainingConfigurationError("qlora.load_in_4bit must remain true in V1")
    if not double_quant:
        raise TrainingConfigurationError("qlora.double_quant must remain true in V1")
    if rank > 256:
        raise TrainingConfigurationError("qlora.rank must be <= 256 on the supported GPU")
    return {
        "load_in_4bit": load_in_4bit,
        "quant_type": quant_type,
        "double_quant": double_quant,
        "compute_dtype": compute_dtype,
        "rank": rank,
        "alpha": alpha,
        "dropout": dropout,
        "bias": bias,
        "target_modules": target_modules,
        "task_type": "CAUSAL_LM",
    }


def resolve_refinement_policy(config: Mapping[str, Any]) -> dict[str, Any]:
    section = config.get(
        "refinement",
        {"mode": "adapter_only", "vram": {"max_gib": 20, "enforcement": "hard"}},
    )
    if not isinstance(section, Mapping):
        raise TrainingConfigurationError("refinement must be a mapping")
    if section.get("mode", "adapter_only") != "adapter_only":
        raise TrainingConfigurationError(
            "refinement.mode must be adapter_only; full-model training is not supported"
        )
    vram = section.get("vram", {})
    if not isinstance(vram, Mapping):
        raise TrainingConfigurationError("refinement.vram must be a mapping")
    max_gib = float_value(
        vram,
        "max_gib",
        20.0,
        "refinement.vram",
        minimum=4.0,
        maximum=192.0,
    )
    enforcement = choice_value(
        vram,
        "enforcement",
        "hard",
        {"hard", "soft"},
        "refinement.vram",
    )
    return {
        "mode": "adapter_only",
        "vram": {"max_gib": max_gib, "enforcement": enforcement},
    }


def base_recipe(
    config: Mapping[str, Any], *, project_root: Path, output_dir: Path
) -> dict[str, Any]:
    validate_root_config(config)
    root = resolve_project_root(project_root)
    destination = resolve_output_dir(output_dir, root)
    model, warnings = resolve_model_recipe(config)
    model_section = get_section(config, "model")
    cache_value = string_value(
        model_section,
        "cache_dir",
        ".autotrainer/model-cache",
        "model",
    )
    cache_dir = Path(cache_value).expanduser()
    if not cache_dir.is_absolute():
        cache_dir = root / cache_dir
    model["cache_dir"] = str(cache_dir.resolve())
    # Real runs are explicitly offline after the separate materialization step.
    model["local_files_only"] = True
    return {
        "schema_version": SUPPORTED_SCHEMA_VERSION,
        "project_root": str(root),
        "output_dir": str(destination),
        "model": model,
        "qlora": resolve_qlora_recipe(config),
        "refinement": resolve_refinement_policy(config),
        "dependency_matrix": dict(REFERENCE_DEPENDENCIES),
        "runtime_requirements": {
            "visible_cuda_devices": 1,
            "configured_vram_gib": resolve_refinement_policy(config)["vram"]["max_gib"],
            "bf16": True,
            "network_during_episode": False,
        },
        "warnings": warnings,
    }


def _first_json_record(path: Path) -> Mapping[str, Any]:
    """Read the first object for executable preflights that need one task."""

    try:
        if path.suffix.lower() == ".jsonl":
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    value = json.loads(line)
                    if not isinstance(value, Mapping):
                        raise TrainingConfigurationError(
                            f"{path}:{line_number} must contain a JSON object"
                        )
                    return value
            raise TrainingConfigurationError(f"dataset is empty: {path}")

        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, list):
            if not value:
                raise TrainingConfigurationError(f"dataset is empty: {path}")
            value = value[0]
        if not isinstance(value, Mapping):
            raise TrainingConfigurationError(
                f"JSON dataset must contain an object or a list of objects: {path}"
            )
        return value
    except json.JSONDecodeError as error:
        raise TrainingConfigurationError(
            f"dataset contains invalid JSON at {path}:{error.lineno}:{error.colno}: {error.msg}"
        ) from error


def _reject_multimodal(value: Any, path: str = "record") -> None:
    if isinstance(value, Mapping):
        forbidden_keys = {"image", "images", "video", "videos", "pixel_values"}
        found = forbidden_keys.intersection(str(key) for key in value)
        if found:
            raise TrainingConfigurationError(
                f"V1 datasets are text-only; found {sorted(found)!r} at {path}"
            )
        content_type = value.get("type")
        if content_type in {"image", "image_url", "video", "video_url"}:
            raise TrainingConfigurationError(
                f"V1 datasets are text-only; found content type {content_type!r} at {path}"
            )
        for key, item in value.items():
            _reject_multimodal(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_multimodal(item, f"{path}[{index}]")


def _validate_messages(value: Any, field: str, *, require_assistant: bool) -> None:
    if not isinstance(value, list) or not value:
        raise TrainingConfigurationError(f"{field} must be a non-empty message list")
    roles: set[str] = set()
    for index, message in enumerate(value):
        if not isinstance(message, Mapping):
            raise TrainingConfigurationError(f"{field}[{index}] must be a message object")
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not role:
            raise TrainingConfigurationError(f"{field}[{index}].role must be a string")
        if role not in {"system", "user", "assistant", "tool"}:
            raise TrainingConfigurationError(
                f"{field}[{index}].role must be system, user, assistant, or tool"
            )
        if not isinstance(content, str):
            raise TrainingConfigurationError(
                f"{field}[{index}].content must be text in V1"
            )
        roles.add(role)
    if require_assistant and "assistant" not in roles:
        raise TrainingConfigurationError(f"{field} must contain an assistant message")


def inspect_sft_dataset(path: Path) -> dict[str, Any]:
    records = _json_records(path)
    first_record = records[0][1]
    for position, record in records:
        field = f"sft.dataset record {position}"
        _reject_multimodal(record, field)
        try:
            normalized = normalize_sft_record(record)
        except ValueError as error:
            raise TrainingConfigurationError(f"{field} {error}") from error
        if (
            "messages" in record
            or record.get("prompt") != normalized["prompt"]
            or record.get("completion") != normalized["completion"]
        ):
            raise TrainingConfigurationError(
                f"{field} must use compiled conversational prompt and completion message lists"
            )
    return {
        "path": str(path),
        "format": "conversational-prompt-completion",
        **dataset_file_identity(path, record_count=len(records)),
        "first_record_fields": sorted(str(key) for key in first_record),
    }


def _json_records(path: Path) -> list[tuple[int, Mapping[str, Any]]]:
    """Read deterministic training records with useful source positions."""

    records: list[tuple[int, Mapping[str, Any]]] = []
    try:
        if path.suffix.lower() == ".jsonl":
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise TrainingConfigurationError(
                        f"{path}:{line_number} must contain a JSON object"
                    )
                records.append((line_number, value))
        else:
            value = json.loads(path.read_text(encoding="utf-8"))
            values = value if isinstance(value, list) else [value]
            for index, record in enumerate(values, 1):
                if not isinstance(record, Mapping):
                    raise TrainingConfigurationError(
                        f"{path}: record {index} must be a JSON object"
                    )
                records.append((index, record))
    except json.JSONDecodeError as error:
        raise TrainingConfigurationError(
            f"dataset contains invalid JSON at {path}:{error.lineno}:{error.colno}: {error.msg}"
        ) from error
    if not records:
        raise TrainingConfigurationError(f"dataset is empty: {path}")
    return records


def dataset_file_identity(path: Path, *, record_count: int | None = None) -> dict[str, Any]:
    """Return the byte-level identity bound into a resolved training recipe."""

    try:
        payload = path.read_bytes()
    except OSError as error:
        raise TrainingConfigurationError(f"Could not read dataset {path}: {error}") from error
    if record_count is None:
        record_count = len(_json_records(path))
    return {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "record_count": record_count,
    }


def verify_dataset_identity(description: Mapping[str, Any]) -> None:
    """Fail when dataset bytes changed after the recipe was resolved.

    Stage loaders call this immediately before handing the path to Datasets.
    This turns the recipe digest into an enforced input lock instead of merely
    descriptive provenance.
    """

    path = Path(str(description.get("path", "")))
    required = ("sha256", "bytes", "record_count")
    missing = [field for field in required if field not in description]
    if missing:
        raise TrainingRuntimeError(
            f"Resolved dataset identity is incomplete for {path}: missing {missing}"
        )
    try:
        current = dataset_file_identity(path)
    except TrainingConfigurationError as error:
        raise TrainingRuntimeError(str(error)) from error
    expected_sha = str(description["sha256"])
    unchanged = (
        hmac.compare_digest(current["sha256"], expected_sha)
        and current["bytes"] == description["bytes"]
        and current["record_count"] == description["record_count"]
    )
    if not unchanged:
        raise TrainingRuntimeError(
            "Dataset changed after recipe resolution; refusing to train on unbound bytes: "
            f"{path}"
        )


def validate_sft_token_lengths(
    tokenizer: Any,
    path: Path,
    max_length: int,
) -> dict[str, int]:
    """Reject any full conversation that the real tokenizer would truncate."""

    longest = 0
    records = _json_records(path)
    for position, record in records:
        _reject_multimodal(record)
        try:
            normalized = normalize_sft_record(record)
        except ValueError as error:
            raise TrainingConfigurationError(
                f"sft.dataset record {position}: {error}"
            ) from error
        if (
            "messages" in record
            or record.get("prompt") != normalized["prompt"]
            or record.get("completion") != normalized["completion"]
        ):
            raise TrainingConfigurationError(
                f"sft.dataset record {position} must use compiled conversational "
                "prompt and completion message lists"
            )
        messages = [*normalized["prompt"], *normalized["completion"]]

        token_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
        )
        if isinstance(token_ids, Mapping):
            token_ids = token_ids.get("input_ids")
        if not isinstance(token_ids, list):
            raise TrainingConfigurationError(
                "the selected tokenizer did not return inspectable input_ids"
            )
        if token_ids and isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        length = len(token_ids)
        longest = max(longest, length)
        if length > max_length:
            raise TrainingConfigurationError(
                f"sft.dataset record {position} uses {length} tokens; "
                f"sft.max_length is {max_length}. Shorten or split the example."
            )
    return {"record_count": len(records), "longest_token_count": longest}


def inspect_grpo_dataset(path: Path) -> dict[str, Any]:
    """Validate every rollout row before a model or environment is loaded.

    Checking only the first row made a late malformed task look like a runtime
    failure after an expensive model load. Task IDs are optional for generic
    environment integrations, but compiled AutoTrainer rows always include
    them and must remain unique.
    """

    records = _json_records(path)
    task_ids: set[str] = set()
    first_record = records[0][1]
    for position, record in records:
        _reject_multimodal(record, f"grpo.dataset record {position}")
        if "prompt" not in record:
            raise TrainingConfigurationError(
                f"grpo.dataset record {position} must contain a prompt field"
            )
        _validate_messages(
            record["prompt"],
            f"grpo.dataset record {position}.prompt",
            require_assistant=False,
        )
        if record["prompt"][-1].get("role") != "user":
            raise TrainingConfigurationError(
                f"grpo.dataset record {position}.prompt must end with a user message "
                "because TRL appends the environment reset observation there"
            )
        if "task_id" in record:
            task_id = record.get("task_id")
            if not isinstance(task_id, str) or not task_id.strip():
                raise TrainingConfigurationError(
                    f"grpo.dataset record {position}.task_id must be non-empty text"
                )
            if task_id in task_ids:
                raise TrainingConfigurationError(
                    f"grpo.dataset contains duplicate task_id {task_id!r}"
                )
            task_ids.add(task_id)
    return {
        "path": str(path),
        "format": "conversational-prompts",
        **dataset_file_identity(path, record_count=len(records)),
        "first_record_fields": sorted(str(key) for key in first_record),
    }


def validate_factory_path(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TrainingConfigurationError(
            "environment.factory must be a dotted path such as package.module:create_environment"
        )
    path = value.strip()
    if ":" in path:
        module_name, attribute_name = path.split(":", 1)
    else:
        module_name, separator, attribute_name = path.rpartition(".")
        if not separator:
            module_name = ""
    valid_module = bool(module_name) and all(
        component.isidentifier() for component in module_name.split(".")
    )
    if not valid_module or not attribute_name.isidentifier():
        raise TrainingConfigurationError(
            "environment.factory must be a valid dotted path such as "
            "package.module:create_environment"
        )
    return f"{module_name}:{attribute_name}"


def import_factory(path: str) -> Callable[[], Any]:
    canonical_path = validate_factory_path(path)
    module_name, attribute_name = canonical_path.split(":", 1)
    try:
        module = importlib.import_module(module_name)
    except Exception as error:
        raise TrainingRuntimeError(
            f"Could not import environment factory module {module_name!r}: {error}"
        ) from error
    try:
        factory = getattr(module, attribute_name)
    except AttributeError as error:
        raise TrainingRuntimeError(
            f"Environment factory attribute {attribute_name!r} does not exist in {module_name!r}"
        ) from error
    if not callable(factory):
        raise TrainingRuntimeError(f"Environment factory {canonical_path!r} is not callable")
    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):
        return factory
    required = [
        parameter.name
        for parameter in signature.parameters.values()
        if parameter.default is inspect.Parameter.empty
        and parameter.kind
        in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }
    ]
    if required:
        raise TrainingRuntimeError(
            f"Environment factory {canonical_path!r} must take no required arguments; "
            f"found {required}"
        )
    return factory


def validate_reference_dependencies() -> dict[str, str]:
    missing: list[str] = []
    mismatched: list[str] = []
    installed: dict[str, str] = {}
    for distribution, expected in REFERENCE_DEPENDENCIES.items():
        try:
            found = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            missing.append(distribution)
            continue
        installed[distribution] = found
        normalized = found.split("+", 1)[0]
        if normalized != expected:
            mismatched.append(f"{distribution}=={expected} (found {found})")
    if missing or mismatched:
        details: list[str] = []
        if missing:
            details.append("missing: " + ", ".join(sorted(missing)))
        if mismatched:
            details.append("unsupported versions: " + ", ".join(mismatched))
        pins = " ".join(f"{name}=={version}" for name, version in REFERENCE_DEPENDENCIES.items())
        raise TrainingDependencyError(
            "The validated Hugging Face training stack is unavailable ("
            + "; ".join(details)
            + f"). Install the project training extra with the reference pins: {pins}"
        )
    return installed


def validate_single_gpu(
    torch_module: Any,
    vram_policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not torch_module.cuda.is_available():
        raise TrainingRuntimeError(
            "CUDA is not available. AutoTrainer V1 requires one visible NVIDIA GPU."
        )
    count = int(torch_module.cuda.device_count())
    if count != 1:
        raise TrainingRuntimeError(
            f"AutoTrainer V1 requires exactly one visible CUDA GPU; found {count}. "
            "Set CUDA_VISIBLE_DEVICES to one device before training."
        )
    if hasattr(torch_module.cuda, "is_bf16_supported") and not torch_module.cuda.is_bf16_supported():
        raise TrainingRuntimeError("The visible CUDA GPU does not support bfloat16 training")
    properties = torch_module.cuda.get_device_properties(0)
    total_bytes = int(properties.total_memory)
    policy = dict(vram_policy or {"max_gib": 20.0, "enforcement": "hard"})
    limit_gib = float(policy.get("max_gib", 20.0))
    if limit_gib < 4:
        raise TrainingConfigurationError("refinement.vram.max_gib must be at least 4")
    limit_bytes = int(limit_gib * 1024**3)
    if total_bytes < limit_bytes:
        raise TrainingRuntimeError(
            f"The configured {limit_gib:g} GiB VRAM budget exceeds the visible GPU's "
            f"{total_bytes / 1024**3:.1f} GiB capacity"
        )
    return {
        "device_count": count,
        "device_name": str(properties.name),
        "vram_gib": round(total_bytes / 1024**3, 2),
        "vram_limit_gib": limit_gib,
        "vram_enforcement": str(policy.get("enforcement", "hard")),
        "bf16_supported": True,
    }


def configure_vram_budget(
    torch_module: Any,
    refinement: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the selected CUDA allocator policy before model allocation."""

    vram = refinement.get("vram", {})
    if not isinstance(vram, Mapping):
        raise TrainingConfigurationError("refinement.vram must be a mapping")
    runtime = validate_single_gpu(torch_module, vram)
    limit_gib = float(runtime["vram_limit_gib"])
    enforcement = str(runtime["vram_enforcement"])
    if enforcement == "hard":
        fraction = min(1.0, limit_gib / float(runtime["vram_gib"]))
        try:
            torch_module.cuda.set_per_process_memory_fraction(fraction, 0)
        except (AttributeError, RuntimeError) as error:
            raise TrainingRuntimeError(
                "The hard VRAM limit could not be installed before model loading"
            ) from error
        runtime["allocator_fraction"] = fraction
    return runtime


def model_max_memory(refinement: Mapping[str, Any]) -> dict[int, str]:
    vram = refinement.get("vram", {})
    if not isinstance(vram, Mapping):
        raise TrainingConfigurationError("refinement.vram must be a mapping")
    limit_gib = float(vram.get("max_gib", 20.0))
    return {0: f"{limit_gib:g}GiB"}


def assert_adapter_only(model: Any, *, stage: str) -> int:
    """Reject any refinement stage that exposes base weights to optimization."""

    trainable: list[tuple[str, int]] = [
        (str(name), int(parameter.numel()))
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if not trainable:
        raise TrainingRuntimeError(f"The {stage} policy has no trainable adapter parameters")
    unexpected = [name for name, _count in trainable if "lora_" not in name.casefold()]
    if unexpected:
        raise TrainingRuntimeError(
            f"{stage} exposed non-adapter parameters; full-model training is forbidden"
        )
    return sum(count for _name, count in trainable)


def inspect_adapter(path: Path, expected_model_id: str, expected_revision: str) -> dict[str, Any]:
    config_path = path / "adapter_config.json"
    if not config_path.is_file():
        raise TrainingConfigurationError(
            f"grpo.start_from is not a PEFT adapter; missing {config_path}"
        )
    try:
        adapter_config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TrainingConfigurationError(
            f"Could not read PEFT adapter metadata from {config_path}: {error}"
        ) from error
    if not isinstance(adapter_config, Mapping):
        raise TrainingConfigurationError(f"Invalid PEFT adapter metadata in {config_path}")
    base_model = adapter_config.get("base_model_name_or_path")
    if base_model != expected_model_id:
        raise TrainingConfigurationError(
            "The GRPO input adapter base model does not match model.id: "
            f"adapter={base_model!r}, recipe={expected_model_id!r}"
        )
    peft_type = adapter_config.get("peft_type")
    if peft_type != "LORA":
        raise TrainingConfigurationError(
            f"grpo.start_from must be a LoRA adapter; found peft_type={peft_type!r}"
        )
    adapter_revision = adapter_config.get("revision")
    if not isinstance(adapter_revision, str) or not IMMUTABLE_REVISION.fullmatch(
        adapter_revision
    ):
        raise TrainingConfigurationError(
            "The GRPO input adapter must record an immutable base-model revision"
        )
    if not IMMUTABLE_REVISION.fullmatch(expected_revision):
        raise TrainingConfigurationError(
            "model.revision must be immutable before continuing a saved adapter"
        )
    if adapter_revision.lower() != expected_revision.lower():
        raise TrainingConfigurationError(
            "The GRPO input adapter base revision does not match model.revision: "
            f"adapter={adapter_revision!r}, recipe={expected_revision!r}"
        )
    if not any(
        (path / filename).is_file()
        for filename in ("adapter_model.safetensors", "adapter_model.bin")
    ):
        raise TrainingConfigurationError(
            "grpo.start_from is missing adapter_model.safetensors or adapter_model.bin"
        )
    return {
        "path": str(path),
        "base_model_name_or_path": base_model,
        "revision": adapter_revision,
        "peft_type": peft_type,
        "tree": adapter_tree_identity(path),
    }


def adapter_tree_identity(path: Path) -> dict[str, Any]:
    """Hash every adapter artifact with deterministic path and size framing."""

    root = Path(path)
    hasher = hashlib.sha256(b"autotrainer-adapter-tree-v1\0")
    file_count = 0
    byte_count = 0
    try:
        entries = sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix())
        for entry in entries:
            relative = entry.relative_to(root).as_posix()
            if entry.is_symlink():
                raise TrainingConfigurationError(
                    f"Adapter trees cannot contain symbolic links: {entry}"
                )
            if entry.is_dir():
                continue
            if not entry.is_file():
                raise TrainingConfigurationError(
                    f"Adapter trees may contain only regular files: {entry}"
                )
            path_bytes = relative.encode("utf-8")
            size = entry.stat().st_size
            hasher.update(len(path_bytes).to_bytes(8, "big"))
            hasher.update(path_bytes)
            hasher.update(size.to_bytes(8, "big"))
            bytes_read = 0
            with entry.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    bytes_read += len(chunk)
                    hasher.update(chunk)
            if bytes_read != size:
                raise TrainingConfigurationError(
                    f"Adapter file changed while its identity was computed: {entry}"
                )
            file_count += 1
            byte_count += bytes_read
    except OSError as error:
        raise TrainingConfigurationError(
            f"Could not compute adapter tree identity for {root}: {error}"
        ) from error
    if file_count == 0:
        raise TrainingConfigurationError(f"Adapter directory is empty: {root}")
    return {
        "sha256": hasher.hexdigest(),
        "bytes": byte_count,
        "file_count": file_count,
    }


def verify_adapter_tree_identity(description: Mapping[str, Any]) -> None:
    """Reject an input adapter changed after the GRPO recipe was resolved."""

    path = Path(str(description.get("path", "")))
    expected = description.get("tree")
    if not isinstance(expected, Mapping):
        raise TrainingRuntimeError(
            f"Resolved adapter identity is missing for {path}"
        )
    try:
        current = adapter_tree_identity(path)
    except TrainingConfigurationError as error:
        raise TrainingRuntimeError(str(error)) from error
    required = ("sha256", "bytes", "file_count")
    if any(field not in expected for field in required):
        raise TrainingRuntimeError(
            f"Resolved adapter identity is incomplete for {path}"
        )
    unchanged = (
        hmac.compare_digest(str(current["sha256"]), str(expected["sha256"]))
        and current["bytes"] == expected["bytes"]
        and current["file_count"] == expected["file_count"]
    )
    if not unchanged:
        raise TrainingRuntimeError(
            "Adapter changed after recipe resolution; refusing to load unbound bytes: "
            f"{path}"
        )


def verify_saved_adapter_provenance(
    path: Path, expected_model_id: str, expected_revision: str
) -> dict[str, Any]:
    """Require PEFT to persist the immutable base identity after saving."""

    try:
        return inspect_adapter(path, expected_model_id, expected_revision)
    except TrainingConfigurationError as error:
        raise TrainingRuntimeError(
            f"Saved LoRA adapter provenance is incomplete or inconsistent: {error}"
        ) from error
