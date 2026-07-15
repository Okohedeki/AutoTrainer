"""Project configuration for the AutoTrainer command line.

The YAML file is deliberately the source of truth.  The dashboard must eventually
read and write this same contract rather than keeping a second model catalogue.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Mapping

import yaml


class ConfigError(ValueError):
    """Raised when an AutoTrainer project configuration is invalid."""


ALLOWED_SOURCE_KINDS = {"repository", "sft_jsonl", "task_pack"}
ALLOWED_PARTITIONS = {"train", "evaluation"}


@dataclass(frozen=True, slots=True)
class ValidationReport:
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not self.errors


@dataclass(frozen=True, slots=True)
class ProjectConfig:
    path: Path
    data: dict[str, Any]

    @property
    def root(self) -> Path:
        return self.path.parent

    @property
    def artifact_dir(self) -> Path:
        configured = self.data.get("project", {}).get("artifact_dir", ".autotrainer")
        candidate = Path(str(configured)).expanduser()
        return candidate if candidate.is_absolute() else (self.root / candidate).resolve()

    @property
    def model(self) -> Mapping[str, Any]:
        return self.data.get("model", {})

    @property
    def sources(self) -> list[dict[str, Any]]:
        return list(self.data.get("sources", []))

    def resolve_path(self, value: str | Path) -> Path:
        candidate = Path(value).expanduser()
        return candidate.resolve() if candidate.is_absolute() else (self.root / candidate).resolve()


def _mapping(value: Any, name: str, errors: list[str]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        errors.append(f"{name} must be a mapping")
        return {}
    return dict(value)


def _positive_int(section: Mapping[str, Any], key: str, errors: list[str]) -> None:
    value = section.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        errors.append(f"{key} must be a positive integer")


def validate_mapping(data: Mapping[str, Any], *, root: Path | None = None) -> ValidationReport:
    """Validate the cross-stage configuration without importing ML libraries."""

    errors: list[str] = []
    warnings: list[str] = []
    allowed_top_level = {
        "schema_version",
        "project",
        "model",
        "sources",
        "qlora",
        "sft",
        "grpo",
        "environment",
        "evaluation",
        "package",
    }
    unknown = sorted(set(data) - allowed_top_level)
    if unknown:
        errors.append(f"unknown top-level fields: {', '.join(unknown)}")

    if data.get("schema_version") != 1:
        errors.append("schema_version must be 1")

    project = _mapping(data.get("project"), "project", errors)
    if not str(project.get("name", "")).strip():
        errors.append("project.name is required")
    seed = project.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        errors.append("project.seed must be a non-negative integer")

    model = _mapping(data.get("model"), "model", errors)
    if model.get("provider") != "huggingface":
        errors.append("model.provider must be huggingface in V1")
    if not str(model.get("id", "")).strip():
        errors.append("model.id must name a Hugging Face model or local model path")
    revision = str(model.get("revision", "")).strip()
    if not revision:
        errors.append("model.revision is required")
    elif not re.fullmatch(r"[0-9a-fA-F]{40,64}", revision):
        warnings.append("model.revision is mutable; `autotrainer lock` must resolve it before a real run")
    if model.get("trust_remote_code") is not False:
        errors.append("model.trust_remote_code must be false in V1")
    if model.get("loader") not in {"auto_text_causal_lm", "qwen3_5_text"}:
        errors.append("model.loader must be auto_text_causal_lm or qwen3_5_text")
    _positive_int(model, "max_sequence_length", errors)

    quantization = _mapping(model.get("quantization"), "model.quantization", errors)
    if quantization.get("method") != "bitsandbytes-4bit":
        errors.append("model.quantization.method must be bitsandbytes-4bit")
    if quantization.get("quant_type") != "nf4":
        errors.append("model.quantization.quant_type must be nf4")
    if quantization.get("compute_dtype") != "bfloat16":
        errors.append("model.quantization.compute_dtype must be bfloat16")

    sources = data.get("sources")
    if not isinstance(sources, list):
        errors.append("sources must be a list")
        sources = []
    source_ids: set[str] = set()
    for index, source_value in enumerate(sources):
        label = f"sources[{index}]"
        source = _mapping(source_value, label, errors)
        source_id = str(source.get("id", "")).strip()
        if not source_id:
            errors.append(f"{label}.id is required")
        elif source_id in source_ids:
            errors.append(f"duplicate source id: {source_id}")
        source_ids.add(source_id)
        kind = source.get("kind")
        if kind not in ALLOWED_SOURCE_KINDS:
            errors.append(f"{label}.kind must be repository, sft_jsonl, or task_pack")
        uri = str(source.get("uri", "")).strip()
        if not uri:
            errors.append(f"{label}.uri is required")
        partition = source.get("partition")
        if partition not in ALLOWED_PARTITIONS:
            errors.append(f"{label}.partition must be train or evaluation")
        if kind == "repository" and not source.get("roles"):
            warnings.append(f"{label} has no roles; it will be scanned as reference evidence only")
        if root is not None and uri and "://" not in uri and not any(ch in uri for ch in "*?["):
            candidate = Path(uri).expanduser()
            candidate = candidate if candidate.is_absolute() else root / candidate
            if not candidate.exists():
                errors.append(f"{label}.uri does not exist: {candidate.resolve()}")

    qlora = _mapping(data.get("qlora"), "qlora", errors)
    _positive_int(qlora, "rank", errors)
    _positive_int(qlora, "alpha", errors)
    dropout = qlora.get("dropout")
    if not isinstance(dropout, (int, float)) or isinstance(dropout, bool) or not 0 <= dropout < 1:
        errors.append("qlora.dropout must be between 0 (inclusive) and 1 (exclusive)")
    target_modules = qlora.get("target_modules")
    if target_modules != "all-linear" and not (
        isinstance(target_modules, list) and target_modules and all(isinstance(item, str) for item in target_modules)
    ):
        errors.append("qlora.target_modules must be all-linear or a non-empty list")

    sft = _mapping(data.get("sft"), "sft", errors)
    if sft.get("enabled") is not False:
        _positive_int(sft, "per_device_train_batch_size", errors)
        _positive_int(sft, "gradient_accumulation_steps", errors)
        _positive_int(sft, "max_length", errors)
        if not isinstance(sft.get("learning_rate"), (int, float)) or sft.get("learning_rate", 0) <= 0:
            errors.append("sft.learning_rate must be positive")

    grpo = _mapping(data.get("grpo"), "grpo", errors)
    if grpo.get("enabled") is not False:
        for key in (
            "per_device_train_batch_size",
            "gradient_accumulation_steps",
            "num_generations",
            "max_completion_length",
            "max_tool_calling_iterations",
            "max_steps",
        ):
            _positive_int(grpo, key, errors)
        if grpo.get("algorithm") != "grpo":
            errors.append("grpo.algorithm must be grpo")
        effective_batch = grpo.get("per_device_train_batch_size", 0) * grpo.get(
            "gradient_accumulation_steps", 0
        )
        generations = grpo.get("num_generations", 0)
        if isinstance(effective_batch, int) and isinstance(generations, int) and generations > 0:
            if effective_batch % generations:
                errors.append(
                    "GRPO effective batch (per_device_train_batch_size × gradient_accumulation_steps) "
                    "must be divisible by num_generations"
                )
        if grpo.get("sft_adapter") in {None, "", "base"}:
            errors.append("grpo.sft_adapter must point to the SFT adapter; RL continues the QLoRA adapter")
        if grpo.get("use_vllm") is not False:
            warnings.append("grpo.use_vllm is not supported by the reference one-GPU V1 profile")

    environment = _mapping(data.get("environment"), "environment", errors)
    if environment.get("backend") not in {"docker", "podman"}:
        errors.append("environment.backend must be docker or podman")
    if environment.get("network") != "none":
        errors.append("environment.network must be none for RL rollouts")
    if not str(environment.get("factory", "")).strip():
        errors.append("environment.factory must be a dotted environment factory path")

    evaluation = _mapping(data.get("evaluation"), "evaluation", errors)
    candidates = evaluation.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        errors.append("evaluation.candidates must be a non-empty list")
    if not any(
        isinstance(source, Mapping)
        and source.get("partition") == "evaluation"
        and source.get("kind") == "task_pack"
        for source in sources
    ):
        warnings.append("no held-out evaluation task_pack is declared")

    return ValidationReport(tuple(errors), tuple(warnings))


def load_config(path: str | Path = "autotrainer.yaml", *, check_paths: bool = False) -> ProjectConfig:
    config_path = Path(path).expanduser().resolve()
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ConfigError(f"configuration not found: {config_path}") from error
    except yaml.YAMLError as error:
        raise ConfigError(f"invalid YAML in {config_path}: {error}") from error
    if not isinstance(payload, Mapping):
        raise ConfigError("configuration root must be a YAML mapping")
    data = dict(payload)
    report = validate_mapping(data, root=config_path.parent if check_paths else None)
    if report.errors:
        raise ConfigError("\n".join(report.errors))
    return ProjectConfig(config_path, data)


def write_config(path: str | Path, data: Mapping[str, Any], *, overwrite: bool = False) -> Path:
    destination = Path(path).expanduser().resolve()
    if destination.exists() and not overwrite:
        raise ConfigError(f"refusing to overwrite existing configuration: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(yaml.safe_dump(dict(data), sort_keys=False, width=100), encoding="utf-8")
    temporary.replace(destination)
    return destination


def default_config(
    *,
    name: str = "frontend-expert-9b",
    model_id: str = "Qwen/Qwen3.5-9B",
    revision: str = "main",
) -> dict[str, Any]:
    """Return the documented one-RTX-4090 smoke recipe."""

    return {
        "schema_version": 1,
        "project": {"name": name, "seed": 42, "artifact_dir": ".autotrainer"},
        "model": {
            "provider": "huggingface",
            "id": model_id,
            "revision": revision,
            "loader": "qwen3_5_text" if model_id == "Qwen/Qwen3.5-9B" else "auto_text_causal_lm",
            "trust_remote_code": False,
            "dtype": "bfloat16",
            "max_sequence_length": 2048,
            "quantization": {
                "method": "bitsandbytes-4bit",
                "quant_type": "nf4",
                "double_quant": True,
                "compute_dtype": "bfloat16",
            },
        },
        "sources": [],
        "qlora": {
            "rank": 32,
            "alpha": 64,
            "dropout": 0.0,
            "target_modules": "all-linear",
            "bias": "none",
        },
        "sft": {
            "enabled": True,
            "dataset": ".autotrainer/compiled/sft/train.jsonl",
            "output_dir": ".autotrainer/checkpoints/sft",
            "num_train_epochs": 1,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "learning_rate": 0.0001,
            "max_length": 2048,
            "gradient_checkpointing": True,
            "assistant_only_loss": True,
            "completion_only_loss": True,
            "packing": False,
            "bf16": True,
            "tf32": True,
            "seed": 42,
            "logging_steps": 5,
            "save_steps": 50,
            "save_total_limit": 2,
        },
        "grpo": {
            "enabled": True,
            "algorithm": "grpo",
            "dataset": ".autotrainer/compiled/rl/train.jsonl",
            "sft_adapter": ".autotrainer/checkpoints/sft",
            "output_dir": ".autotrainer/checkpoints/grpo",
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 2,
            "num_generations": 2,
            "generation_batch_size": 2,
            "learning_rate": 0.00001,
            "max_steps": 100,
            "max_completion_length": 2048,
            "max_tool_calling_iterations": 8,
            "beta": 0.0,
            "loss_type": "dapo",
            "use_vllm": False,
            "gradient_checkpointing": True,
            "bf16": True,
            "tf32": True,
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "seed": 42,
            "logging_steps": 5,
            "save_steps": 50,
            "save_total_limit": 2,
        },
        "environment": {
            "factory": "autotrainer.environments.frontend:FrontendEnvironment",
            "backend": "docker",
            "image": "autotrainer/frontend-runtime:0.1",
            "network": "none",
            "max_tool_output_chars": 12000,
            "episode_timeout_seconds": 900,
        },
        "evaluation": {
            "candidates": ["base", "sft:best", "rl:best"],
            "primary_metric": "verified_task_success",
        },
        "package": {"type": "lora_adapter", "merge_base_weights": False},
    }
