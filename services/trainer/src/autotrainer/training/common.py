"""Shared validation and runtime guards for local single-GPU training.

This module intentionally depends only on the Python standard library.  The
CUDA and Hugging Face stack is imported by the stage runners after a recipe has
been validated, so configuration inspection remains useful on machines that do
not have the training extras installed.
"""

from __future__ import annotations

import importlib
import inspect
import json
import re
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, Mapping


SUPPORTED_SCHEMA_VERSION = 1
SUPPORTED_MODEL_ID = "Qwen/Qwen3.5-9B"
SUPPORTED_MODEL_CLASS = "Qwen3_5ForCausalLM"

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
    if not text_only:
        raise TrainingConfigurationError("model.text_only must remain true in V1")
    if trust_remote_code:
        raise TrainingConfigurationError("model.trust_remote_code must remain false in V1")
    if thinking:
        raise TrainingConfigurationError(
            "model.thinking must remain false in the supported 24 GB recipe"
        )
    warnings: list[str] = []
    if not re.fullmatch(r"[0-9a-fA-F]{40,64}", revision):
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
        },
        warnings,
    )


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
        "dependency_matrix": dict(REFERENCE_DEPENDENCIES),
        "runtime_requirements": {
            "visible_cuda_devices": 1,
            "minimum_vram_gib": 20,
            "bf16": True,
            "network_during_episode": False,
        },
        "warnings": warnings,
    }


def _first_json_record(path: Path) -> Mapping[str, Any]:
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
        if not isinstance(content, str):
            raise TrainingConfigurationError(
                f"{field}[{index}].content must be text in V1"
            )
        roles.add(role)
    if require_assistant and "assistant" not in roles:
        raise TrainingConfigurationError(f"{field} must contain an assistant message")


def inspect_sft_dataset(path: Path) -> dict[str, Any]:
    record = _first_json_record(path)
    _reject_multimodal(record)
    if "messages" in record:
        _validate_messages(record["messages"], "sft.dataset.messages", require_assistant=True)
        dataset_format = "conversational-messages"
    elif "prompt" in record and "completion" in record:
        _validate_messages(record["prompt"], "sft.dataset.prompt", require_assistant=False)
        _validate_messages(
            record["completion"], "sft.dataset.completion", require_assistant=True
        )
        dataset_format = "conversational-prompt-completion"
    else:
        raise TrainingConfigurationError(
            "sft.dataset records must contain conversational messages, or prompt and completion"
        )
    return {
        "path": str(path),
        "format": dataset_format,
        "first_record_fields": sorted(str(key) for key in record),
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
        if "messages" in record:
            messages = record["messages"]
            _validate_messages(messages, "sft.dataset.messages", require_assistant=True)
        elif "prompt" in record and "completion" in record:
            prompt = record["prompt"]
            completion = record["completion"]
            _validate_messages(prompt, "sft.dataset.prompt", require_assistant=False)
            _validate_messages(completion, "sft.dataset.completion", require_assistant=True)
            messages = [*prompt, *completion]
        else:
            raise TrainingConfigurationError(
                f"sft.dataset record {position} must contain messages or prompt and completion"
            )

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
    record = _first_json_record(path)
    _reject_multimodal(record)
    if "prompt" not in record:
        raise TrainingConfigurationError("grpo.dataset records must contain a prompt field")
    _validate_messages(record["prompt"], "grpo.dataset.prompt", require_assistant=False)
    return {
        "path": str(path),
        "format": "conversational-prompts",
        "first_record_fields": sorted(str(key) for key in record),
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


def validate_single_gpu(torch_module: Any) -> dict[str, Any]:
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
    minimum_bytes = 20 * 1024**3
    if total_bytes < minimum_bytes:
        raise TrainingRuntimeError(
            "The supported 9B QLoRA+GRPO recipe requires at least 20 GiB of VRAM; "
            f"the visible GPU reports {total_bytes / 1024**3:.1f} GiB"
        )
    return {
        "device_count": count,
        "device_name": str(properties.name),
        "vram_gib": round(total_bytes / 1024**3, 2),
        "bf16_supported": True,
    }


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
    if peft_type not in {None, "LORA"}:
        raise TrainingConfigurationError(
            f"grpo.start_from must be a LoRA adapter; found peft_type={peft_type!r}"
        )
    adapter_revision = adapter_config.get("revision")
    if adapter_revision and adapter_revision != expected_revision:
        raise TrainingConfigurationError(
            "The GRPO input adapter base revision does not match model.revision: "
            f"adapter={adapter_revision!r}, recipe={expected_revision!r}"
        )
    return {
        "path": str(path),
        "base_model_name_or_path": base_model,
        "revision": adapter_revision,
        "peft_type": peft_type or "LORA",
    }
