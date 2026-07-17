"""Project configuration for the AutoTrainer command line.

The YAML file is deliberately the source of truth.  The dashboard must eventually
read and write this same contract rather than keeping a second model catalogue.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import re
import threading
from typing import Any, Iterator, Mapping

import yaml


class ConfigError(ValueError):
    """Raised when an AutoTrainer project configuration is invalid."""


# GUI requests can finish in a different order than they started.  Every
# read/modify/write operation uses the lock for its project so one request
# cannot replace unrelated YAML fields written by another request.
_CONFIG_LOCKS_GUARD = threading.Lock()
_CONFIG_MUTATION_LOCKS: dict[Path, threading.RLock] = {}


@contextmanager
def project_config_mutation(path: str | Path) -> Iterator[Path]:
    """Serialize one project's in-process YAML read/modify/write operations.

    Long work that does not mutate YAML (for example downloading model blobs)
    should happen before entering this context.  Callers must re-read the
    returned path inside the context and merge only the fields they own.
    """

    config_path = Path(path).expanduser().resolve()
    with _CONFIG_LOCKS_GUARD:
        lock = _CONFIG_MUTATION_LOCKS.setdefault(config_path, threading.RLock())
    with lock:
        yield config_path


ALLOWED_SOURCE_KINDS = {"repository", "sft_jsonl", "task_pack"}
ALLOWED_PARTITIONS = {"train", "evaluation"}
EVALUATION_ROLES = {"reference", "control", "candidate"}
EVALUATION_SUITES = {"model_benchmark", "fable_ab"}
IMMUTABLE_REVISION = re.compile(r"[0-9a-fA-F]{40,64}")
FAIRNESS_TRUE_FIELDS = (
    "same_task_snapshot",
    "same_instruction",
    "same_tools_and_limits",
    "same_verifier",
    "same_runner_within_suite",
    "same_sampling",
    "require_seed_control",
    "immutable_models_and_adapter",
    "failures_score_zero",
)


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


def _resolved_config_path(value: Any, root: Path | None) -> Path | None:
    """Return one canonical identity for cross-section path comparisons."""

    if not isinstance(value, (str, Path)) or not str(value).strip():
        return None
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        # Validation can run before a config file has been written. Using one
        # shared anchor still detects equivalent relative spellings such as
        # ``data/eval.jsonl`` and ``data/./eval.jsonl``.
        candidate = (root if root is not None else Path.cwd()) / candidate
    return candidate.resolve()


def _evaluation_model_ref(value: Any, label: str, errors: list[str]) -> str:
    if value == "project":
        return "project"
    model = _mapping(value, f"{label}.model", errors)
    if model.get("provider") != "huggingface":
        errors.append(f"{label}.model.provider must be huggingface")
    if not str(model.get("id", "")).strip():
        errors.append(f"{label}.model.id is required")
    revision = str(model.get("revision", "")).strip()
    if not IMMUTABLE_REVISION.fullmatch(revision):
        errors.append(f"{label}.model.revision must be an immutable 40-64 character commit SHA")
    if model.get("loader") not in {"auto_text_causal_lm", "qwen3_5_text"}:
        errors.append(f"{label}.model.loader must be auto_text_causal_lm or qwen3_5_text")
    if model.get("trust_remote_code", False) is not False:
        errors.append(f"{label}.model.trust_remote_code must be false")
    if model.get("dtype") != "bfloat16":
        errors.append(f"{label}.model.dtype must be bfloat16")
    max_sequence_length = model.get("max_sequence_length")
    if (
        not isinstance(max_sequence_length, int)
        or isinstance(max_sequence_length, bool)
        or max_sequence_length < 256
    ):
        errors.append(f"{label}.model.max_sequence_length must be at least 256")
    if model.get("quantization") != "project":
        errors.append(f"{label}.model.quantization must be project")
    return "huggingface"


def _validate_evaluation(evaluation: Mapping[str, Any], errors: list[str]) -> None:
    dataset = evaluation.get("dataset")
    if not isinstance(dataset, (str, Path)) or not str(dataset).strip():
        errors.append("evaluation.dataset is required")
    elif Path(dataset).suffix.lower() != ".jsonl":
        errors.append("evaluation.dataset must end in .jsonl")
    if not str(evaluation.get("task_pack", "")).strip():
        errors.append("evaluation.task_pack is required")
    if evaluation.get("task_split") != "evaluation":
        errors.append("evaluation.task_split must be evaluation")
    if evaluation.get("holdout_unit") != "repository":
        errors.append("evaluation.holdout_unit must be repository")
    if evaluation.get("primary_metric") != "verified_task_success":
        errors.append("evaluation.primary_metric must be verified_task_success")

    repetitions = evaluation.get("repetitions")
    if not isinstance(repetitions, int) or isinstance(repetitions, bool) or repetitions < 1:
        errors.append("evaluation.repetitions must be a positive integer")
        repetitions = 0
    seeds = evaluation.get("seeds")
    if not isinstance(seeds, list) or not seeds or any(
        not isinstance(seed, int) or isinstance(seed, bool) or seed < 0 for seed in seeds
    ):
        errors.append("evaluation.seeds must be a non-empty list of non-negative integers")
        seeds = []
    elif len(set(seeds)) != len(seeds):
        errors.append("evaluation.seeds must be unique")
    if repetitions and isinstance(seeds, list) and len(seeds) != repetitions:
        errors.append("evaluation.seeds length must equal evaluation.repetitions")

    arms_value = evaluation.get("arms")
    if not isinstance(arms_value, Mapping):
        errors.append("evaluation.arms must be a mapping")
        arms: dict[str, Any] = {}
    else:
        arms = dict(arms_value)
    if len(arms) != 3:
        errors.append("evaluation.arms must declare exactly reference, control, and candidate arms")

    candidates = evaluation.get("candidates")
    if not isinstance(candidates, list) or not candidates or any(
        not isinstance(candidate, str) or not candidate.strip() for candidate in candidates
    ):
        errors.append("evaluation.candidates must be a non-empty list of arm ids")
        candidate_ids: list[str] = []
    else:
        candidate_ids = list(candidates)
        if len(set(candidate_ids)) != len(candidate_ids):
            errors.append("evaluation.candidates must be unique")
        if set(candidate_ids) != set(arms):
            errors.append("evaluation.candidates must contain exactly the declared arm ids")

    arm_roles: dict[str, str] = {}
    for arm_id, arm_value in arms.items():
        label = f"evaluation.arms.{arm_id}"
        if not isinstance(arm_id, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", arm_id):
            errors.append(f"{label} id must be a lowercase slug")
        arm = _mapping(arm_value, label, errors)
        if not str(arm.get("label", "")).strip():
            errors.append(f"{label}.label is required")
        if arm.get("parameter_class") != "9b":
            errors.append(f"{label}.parameter_class must be 9b in V1")
        role = arm.get("role")
        if role not in EVALUATION_ROLES:
            errors.append(f"{label}.role must be reference, control, or candidate")
            role = ""
        arm_roles[str(arm_id)] = str(role)
        model_kind = _evaluation_model_ref(arm.get("model"), label, errors)
        adapter = arm.get("adapter")
        if role == "reference":
            if model_kind != "huggingface":
                errors.append(f"{label} reference arm must use an immutable Hugging Face model")
            if adapter is not None:
                errors.append(f"{label} reference arm must not declare an adapter")
        elif role == "control":
            if model_kind != "project":
                errors.append(f"{label} control arm must use model: project")
            if adapter is not None:
                errors.append(f"{label} control arm must not declare an adapter")
        elif role == "candidate":
            if model_kind != "project":
                errors.append(f"{label} candidate arm must use model: project")
            adapter_mapping = _mapping(adapter, f"{label}.adapter", errors)
            if not str(adapter_mapping.get("path", "")).strip():
                errors.append(f"{label}.adapter.path is required")
            if adapter_mapping.get("stage") not in {"sft", "grpo"}:
                errors.append(f"{label}.adapter.stage must be sft or grpo")
            digest = adapter_mapping.get("sha256")
            if digest is not None and not re.fullmatch(r"[0-9a-fA-F]{64}", str(digest)):
                errors.append(f"{label}.adapter.sha256 must be a 64 character digest")
    role_counts = {role: list(arm_roles.values()).count(role) for role in EVALUATION_ROLES}
    if any(role_counts[role] != 1 for role in EVALUATION_ROLES):
        errors.append("evaluation.arms must contain exactly one reference, one control, and one candidate role")

    suites_value = evaluation.get("suites")
    if not isinstance(suites_value, Mapping):
        errors.append("evaluation.suites must be a mapping")
        suites: dict[str, Any] = {}
    else:
        suites = dict(suites_value)
    if set(suites) != EVALUATION_SUITES:
        errors.append("evaluation.suites must contain exactly model_benchmark and fable_ab")

    suite_arms: dict[str, list[str]] = {}
    for suite_id in sorted(EVALUATION_SUITES):
        suite = _mapping(suites.get(suite_id), f"evaluation.suites.{suite_id}", errors)
        if suite.get("kind") != suite_id:
            errors.append(f"evaluation.suites.{suite_id}.kind must be {suite_id}")
        members = suite.get("arms")
        if not isinstance(members, list) or not members or any(
            not isinstance(member, str) or not member for member in members
        ):
            errors.append(f"evaluation.suites.{suite_id}.arms must be a non-empty list")
            members = []
        elif len(set(members)) != len(members):
            errors.append(f"evaluation.suites.{suite_id}.arms must be unique")
        unknown_members = sorted(set(members) - set(arms))
        if unknown_members:
            errors.append(
                f"evaluation.suites.{suite_id} references unknown arms: {', '.join(unknown_members)}"
            )
        suite_arms[suite_id] = list(members)
        runner = _mapping(suite.get("runner"), f"evaluation.suites.{suite_id}.runner", errors)
        runner_type = runner.get("type")
        if runner_type not in {"builtin", "command", "external"}:
            errors.append(
                f"evaluation.suites.{suite_id}.runner.type must be builtin, command, or external"
            )
        runner_label = f"evaluation.suites.{suite_id}.runner"
        if runner_type == "builtin":
            # Built-in identity is derived from the installed code when the
            # plan freezes. Users cannot accidentally claim a different prompt
            # or producer version for the runner AutoTrainer actually invokes.
            unknown = sorted(set(runner) - {"type"})
            if unknown:
                errors.append(
                    f"{runner_label} builtin runner accepts only type; remove: "
                    + ", ".join(unknown)
                )
        else:
            if not str(runner.get("producer", "")).strip():
                errors.append(f"{runner_label}.producer is required")
            if not str(runner.get("version", "")).strip():
                errors.append(f"{runner_label}.version is required")
            orchestration_sha256 = str(runner.get("orchestration_sha256", "")).strip()
            if not re.fullmatch(r"sha256:[0-9a-fA-F]{64}", orchestration_sha256):
                errors.append(
                    f"{runner_label}.orchestration_sha256 must be sha256:<64 hex characters>"
                )
        if runner_type == "command":
            argv = runner.get("argv")
            if not isinstance(argv, list) or not argv or any(
                not isinstance(part, str) or not part.strip() for part in argv
            ):
                errors.append(
                    f"{runner_label}.argv must be a non-empty argument list"
                )
            else:
                if not any("{request}" in part for part in argv):
                    errors.append(f"{runner_label}.argv must include a {{request}} placeholder")
                if not any("{result}" in part for part in argv):
                    errors.append(f"{runner_label}.argv must include a {{result}} placeholder")
        elif runner_type == "external":
            if not str(runner.get("result_schema", "")).strip():
                errors.append(f"{runner_label}.result_schema is required")

    model_roles = {arm_roles.get(member) for member in suite_arms.get("model_benchmark", [])}
    if len(suite_arms.get("model_benchmark", [])) != 2 or model_roles != {"reference", "candidate"}:
        errors.append("model_benchmark must contain exactly the reference and candidate arms")
    model_suite = _mapping(
        suites.get("model_benchmark"), "evaluation.suites.model_benchmark", errors
    )
    model_runner = _mapping(
        model_suite.get("runner"), "evaluation.suites.model_benchmark.runner", errors
    )
    if model_runner.get("type") not in {"builtin", "command"}:
        errors.append("model_benchmark runner must use type: builtin or command")

    fable_members = suite_arms.get("fable_ab", [])
    fable_roles = {arm_roles.get(member) for member in fable_members}
    if len(fable_members) != 2 or fable_roles != {"control", "candidate"}:
        errors.append("fable_ab must contain exactly the control and candidate arms")
    fable_suite = _mapping(suites.get("fable_ab"), "evaluation.suites.fable_ab", errors)
    fable_runner = _mapping(
        fable_suite.get("runner"), "evaluation.suites.fable_ab.runner", errors
    )
    if fable_runner.get("type") != "external":
        errors.append("fable_ab runner must use type: external")
    review = _mapping(fable_suite.get("review"), "evaluation.suites.fable_ab.review", errors)
    if review.get("type") != "manual":
        errors.append("evaluation.suites.fable_ab.review.type must be manual")
    if review.get("blind") is not True:
        errors.append("evaluation.suites.fable_ab.review.blind must be true")
    _positive_int(review, "reviewers_per_pair", errors)

    fairness = _mapping(evaluation.get("fairness"), "evaluation.fairness", errors)
    if fairness.get("paired_by") != ["task_id", "repetition", "seed"]:
        errors.append("evaluation.fairness.paired_by must be [task_id, repetition, seed]")
    for key in FAIRNESS_TRUE_FIELDS:
        if fairness.get(key) is not True:
            errors.append(f"evaluation.fairness.{key} must be true")
    policy_fields = {
        "pair_position_policy",
        "execution_order_policy",
        "per_trial_arm_randomization",
    }
    legacy_policy = fairness.get("randomize_arm_order") is True
    declared_policy = any(key in fairness for key in policy_fields)
    if legacy_policy and declared_policy:
        errors.append(
            "evaluation.fairness cannot combine legacy randomize_arm_order with the V1 execution policy"
        )
    elif not legacy_policy:
        if fairness.get("pair_position_policy") != "deterministic_counterbalance":
            errors.append(
                "evaluation.fairness.pair_position_policy must be deterministic_counterbalance"
            )
        if fairness.get("execution_order_policy") != "frozen_per_suite":
            errors.append(
                "evaluation.fairness.execution_order_policy must be frozen_per_suite"
            )
        if fairness.get("per_trial_arm_randomization") is not False:
            errors.append("evaluation.fairness.per_trial_arm_randomization must be false")
    if fairness.get("allow_unplanned_reruns") is not False:
        errors.append("evaluation.fairness.allow_unplanned_reruns must be false")

    decisions = _mapping(evaluation.get("decisions"), "evaluation.decisions", errors)
    confidence = decisions.get("confidence")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not 0.5 < confidence < 1:
        errors.append("evaluation.decisions.confidence must be between 0.5 and 1")
    model_decision = _mapping(
        decisions.get("model_benchmark"), "evaluation.decisions.model_benchmark", errors
    )
    if model_decision.get("candidate") not in suite_arms.get("model_benchmark", []):
        errors.append("evaluation.decisions.model_benchmark.candidate must reference a suite arm")
    elif arm_roles.get(str(model_decision.get("candidate"))) != "candidate":
        errors.append("evaluation.decisions.model_benchmark.candidate must use the candidate role")
    if model_decision.get("control") not in suite_arms.get("model_benchmark", []):
        errors.append("evaluation.decisions.model_benchmark.control must reference a suite arm")
    elif arm_roles.get(str(model_decision.get("control"))) != "reference":
        errors.append("evaluation.decisions.model_benchmark.control must use the reference role")
    if model_decision.get("metric") != "verified_task_success":
        errors.append("evaluation.decisions.model_benchmark.metric must be verified_task_success")
    minimum_delta = model_decision.get("minimum_delta")
    if not isinstance(minimum_delta, (int, float)) or isinstance(minimum_delta, bool) or not -1 <= minimum_delta <= 1:
        errors.append("evaluation.decisions.model_benchmark.minimum_delta must be between -1 and 1")
    model_minimum_tasks = model_decision.get("minimum_tasks")
    if (
        not isinstance(model_minimum_tasks, int)
        or isinstance(model_minimum_tasks, bool)
        or model_minimum_tasks < 2
    ):
        errors.append("evaluation.decisions.model_benchmark.minimum_tasks must be at least 2")

    fable_decision = _mapping(decisions.get("fable_ab"), "evaluation.decisions.fable_ab", errors)
    if fable_decision.get("candidate") not in fable_members:
        errors.append("evaluation.decisions.fable_ab.candidate must reference a suite arm")
    elif arm_roles.get(str(fable_decision.get("candidate"))) != "candidate":
        errors.append("evaluation.decisions.fable_ab.candidate must use the candidate role")
    if fable_decision.get("control") not in fable_members:
        errors.append("evaluation.decisions.fable_ab.control must reference a suite arm")
    elif arm_roles.get(str(fable_decision.get("control"))) != "control":
        errors.append("evaluation.decisions.fable_ab.control must use the control role")
    if fable_decision.get("metric") != "blind_preference_rate":
        errors.append("evaluation.decisions.fable_ab.metric must be blind_preference_rate")
    minimum_rate = fable_decision.get("minimum_rate")
    if not isinstance(minimum_rate, (int, float)) or isinstance(minimum_rate, bool) or not 0 <= minimum_rate <= 1:
        errors.append("evaluation.decisions.fable_ab.minimum_rate must be between 0 and 1")
    fable_minimum_tasks = fable_decision.get("minimum_tasks")
    if (
        not isinstance(fable_minimum_tasks, int)
        or isinstance(fable_minimum_tasks, bool)
        or fable_minimum_tasks < 2
    ):
        errors.append("evaluation.decisions.fable_ab.minimum_tasks must be at least 2")


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
    sft_enabled = sft.get("enabled", True)
    if not isinstance(sft_enabled, bool):
        errors.append("sft.enabled must be a boolean")
        sft_enabled = True
    if sft_enabled:
        _positive_int(sft, "per_device_train_batch_size", errors)
        _positive_int(sft, "gradient_accumulation_steps", errors)
        _positive_int(sft, "max_length", errors)
        if not isinstance(sft.get("learning_rate"), (int, float)) or sft.get("learning_rate", 0) <= 0:
            errors.append("sft.learning_rate must be positive")

    grpo = _mapping(data.get("grpo"), "grpo", errors)
    grpo_enabled = grpo.get("enabled", True)
    if not isinstance(grpo_enabled, bool):
        errors.append("grpo.enabled must be a boolean")
        grpo_enabled = True
    if grpo_enabled:
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
        start_from = grpo.get("start_from", grpo.get("sft_adapter"))
        if start_from in {None, ""}:
            errors.append("grpo.start_from must be 'base' or point to a LoRA adapter")
        # When both stages are requested, RL must actually continue the adapter
        # produced by this SFT stage. Practice-only RL may instead start from the
        # selected base model or a separately supplied compatible adapter.
        if sft_enabled:
            if start_from == "base":
                errors.append("both-stage training requires GRPO to continue sft.output_dir, not base")
            else:
                sft_output = _resolved_config_path(sft.get("output_dir"), root)
                grpo_input = _resolved_config_path(start_from, root)
                if sft_output is not None and grpo_input is not None and sft_output != grpo_input:
                    errors.append("both-stage training requires grpo.start_from to equal sft.output_dir")
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
    _validate_evaluation(evaluation, errors)
    grpo_eval_path = _resolved_config_path(grpo.get("eval_dataset"), root)
    final_eval_path = _resolved_config_path(evaluation.get("dataset"), root)
    if grpo_eval_path is not None and grpo_eval_path == final_eval_path:
        errors.append(
            "grpo.eval_dataset must be separate from evaluation.dataset; "
            "training validation cannot reuse the final benchmark"
        )
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

    # The reference arm is a product-level benchmark choice, not setup the
    # user should have to rediscover in every new project. Keep it aligned with
    # the same immutable catalog record used by the downloader and GUI.
    from .models import MODEL_CATALOG

    reference_model = MODEL_CATALOG["qwythos-9b-reference"]

    return {
        "schema_version": 1,
        "project": {"name": name, "seed": 42, "artifact_dir": ".autotrainer"},
        "model": {
            "provider": "huggingface",
            "id": model_id,
            "revision": revision,
            "cache_dir": ".autotrainer/model-cache",
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
            "start_from": ".autotrainer/checkpoints/sft",
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
            "dataset": ".autotrainer/compiled/rl/evaluation.jsonl",
            "task_pack": "held-out-frontend",
            "task_split": "evaluation",
            "repetitions": 3,
            "seeds": [1701, 1702, 1703],
            "holdout_unit": "repository",
            "primary_metric": "verified_task_success",
            # ``candidates`` is retained as the ordered, display-facing arm list.
            # Runtime identity and comparison roles live in ``arms`` and ``suites``.
            "candidates": ["reference_9b", "base_fable", "autotrainer"],
            "arms": {
                "reference_9b": {
                    "label": "Qwythos 9B reference",
                    "role": "reference",
                    "parameter_class": "9b",
                    "model": {
                        "provider": "huggingface",
                        "id": reference_model["id"],
                        "revision": reference_model["default_revision"],
                        "loader": "auto_text_causal_lm",
                        "trust_remote_code": False,
                        "dtype": "bfloat16",
                        "max_sequence_length": 2048,
                        "quantization": "project",
                    },
                },
                "base_fable": {
                    "label": "Base 9B + Fable",
                    "role": "control",
                    "parameter_class": "9b",
                    "model": "project",
                },
                "autotrainer": {
                    "label": "AutoTrainer 9B",
                    "role": "candidate",
                    "parameter_class": "9b",
                    "model": "project",
                    "adapter": {
                        "path": ".autotrainer/checkpoints/grpo",
                        "stage": "grpo",
                    },
                },
            },
            "suites": {
                "model_benchmark": {
                    "kind": "model_benchmark",
                    "arms": ["reference_9b", "autotrainer"],
                    "runner": {
                        # AutoTrainer owns the exact prompt and local 4-bit
                        # loader. Its immutable identity is frozen into the
                        # evaluation plan from the installed code.
                        "type": "builtin",
                    },
                },
                "fable_ab": {
                    "kind": "fable_ab",
                    "arms": ["base_fable", "autotrainer"],
                    "runner": {
                        "type": "external",
                        "producer": "fable",
                        "version": "REPLACE_WITH_FABLE_VERSION",
                        "orchestration_sha256": "sha256:" + "0" * 64,
                        "result_schema": "autotrainer-evaluation-result-v1",
                    },
                    "review": {
                        "type": "manual",
                        "blind": True,
                        "reviewers_per_pair": 3,
                    },
                },
            },
            "fairness": {
                "paired_by": ["task_id", "repetition", "seed"],
                "same_task_snapshot": True,
                "same_instruction": True,
                "same_tools_and_limits": True,
                "same_verifier": True,
                "same_runner_within_suite": True,
                "same_sampling": True,
                "require_seed_control": True,
                "immutable_models_and_adapter": True,
                # Pair positions are counterbalanced for analysis, while the
                # built-in runner groups 9B arms so only one occupies GPU 0.
                "pair_position_policy": "deterministic_counterbalance",
                "execution_order_policy": "frozen_per_suite",
                "per_trial_arm_randomization": False,
                "failures_score_zero": True,
                "allow_unplanned_reruns": False,
            },
            "decisions": {
                "confidence": 0.95,
                "model_benchmark": {
                    "candidate": "autotrainer",
                    "control": "reference_9b",
                    "metric": "verified_task_success",
                    "minimum_delta": 0.0,
                    "minimum_tasks": 2,
                },
                "fable_ab": {
                    "candidate": "autotrainer",
                    "control": "base_fable",
                    "metric": "blind_preference_rate",
                    "minimum_rate": 0.5,
                    "minimum_tasks": 2,
                },
            },
        },
        "package": {"type": "lora_adapter", "merge_base_weights": False},
    }
