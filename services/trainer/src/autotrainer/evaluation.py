"""Reproducible two-suite evaluation workflow for AutoTrainer V1.

Evaluation is intentionally separated from model generation.  AutoTrainer
freezes a paired trial matrix, exports public task envelopes to a local agent or
Fable, ingests unified patches, and re-scores every patch in the trusted
frontend environment.  Producer supplied scores are never accepted.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import stat
import subprocess
import tempfile
from typing import Any, Callable, Iterable, Mapping, Sequence

from .benchmark import compare_benchmark, render_benchmark_markdown


SCHEMA_VERSION = 1
RESULT_COMPONENTS = (
    "design_rules",
    "patch_quality",
    "regression_safety",
    "responsive_rules",
    "task_tests",
)
RESULT_ENVELOPE_NAMES = ("result.json",)
RESULT_ENVELOPE_SUFFIX = ".result.json"
MAX_RESULT_ARTIFACT_BYTES = 10 * 1024 * 1024


class EvaluationError(ValueError):
    """Raised when an evaluation plan or result is incomplete or inconsistent."""


def _canonical(value: Any) -> bytes:
    # Every identity in the evaluation tree is content-addressed. Compact,
    # sorted JSON keeps hashes stable when mappings arrive in a different order.
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_tree(path: Path) -> str:
    if not path.is_dir():
        raise EvaluationError(f"adapter directory does not exist: {path}")
    # Hash relative paths as well as bytes: renaming an adapter file changes the
    # load contract even when its contents happen to be identical.
    entries: list[dict[str, str]] = []
    for candidate in sorted(path.rglob("*")):
        if candidate.is_symlink():
            raise EvaluationError(f"adapter trees must not contain symlinks: {candidate}")
        if candidate.is_file():
            entries.append(
                {
                    "path": candidate.relative_to(path).as_posix(),
                    "sha256": _sha256_file(candidate),
                }
            )
    if not entries:
        raise EvaluationError(f"adapter directory is empty: {path}")
    return _digest(entries)


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise EvaluationError(f"{field} must be a mapping")
    return dict(value)


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvaluationError(f"{field} must be a non-empty string")
    return value.strip()


def _exact_mapping(
    value: Any,
    field: str,
    *,
    required: Iterable[str],
    optional: Iterable[str] = (),
) -> dict[str, Any]:
    """Apply the published schemas' closed-object contract without a runtime dependency."""

    result = _mapping(value, field)
    required_keys = set(required)
    allowed_keys = required_keys | set(optional)
    missing = sorted(required_keys - set(result))
    unknown = sorted(set(result) - allowed_keys)
    if missing:
        raise EvaluationError(f"{field} is missing required field(s): {', '.join(missing)}")
    if unknown:
        raise EvaluationError(f"{field} contains unknown field(s): {', '.join(unknown)}")
    return result


def _schema_string(value: Any, field: str, pattern: str | None = None) -> str:
    if not isinstance(value, str) or not value:
        raise EvaluationError(f"{field} must be a non-empty string")
    if pattern is not None and re.fullmatch(pattern, value) is None:
        raise EvaluationError(f"{field} does not match the published schema")
    return value


def _schema_non_negative_integer(value: Any, field: str) -> int:
    # JSON Schema treats 1 and 1.0 as the same integer value. Preserve that
    # behavior while rejecting booleans and non-finite Python JSON extensions.
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or value < 0
        or not float(value).is_integer()
    ):
        raise EvaluationError(f"{field} must be a non-negative integer")
    return int(value)


def _validate_result_envelope(value: Any, field: str) -> dict[str, Any]:
    """Mirror ``schemas/evaluation-result.schema.json`` at the ingestion boundary."""

    result = _exact_mapping(
        value,
        field,
        required=(
            "schema_version",
            "plan_id",
            "trial_id",
            "suite_id",
            "arm_id",
            "task_id",
            "repetition",
            "seed",
            "status",
            "producer",
            "usage",
            "output",
        ),
    )
    if isinstance(result["schema_version"], bool) or result["schema_version"] != SCHEMA_VERSION:
        raise EvaluationError(f"{field}.schema_version must be {SCHEMA_VERSION}")
    _schema_string(result["plan_id"], f"{field}.plan_id", r"sha256:[0-9a-f]{64}")
    for key in ("trial_id", "arm_id", "task_id"):
        _schema_string(result[key], f"{field}.{key}")
    suite_id = _schema_string(result["suite_id"], f"{field}.suite_id")
    if suite_id not in {"model_benchmark", "fable_ab"}:
        raise EvaluationError(f"{field}.suite_id must be model_benchmark or fable_ab")
    _schema_non_negative_integer(result["repetition"], f"{field}.repetition")
    _schema_non_negative_integer(result["seed"], f"{field}.seed")
    status = _schema_string(result["status"], f"{field}.status")
    if status not in {"completed", "failed", "timeout"}:
        raise EvaluationError(f"{field}.status must be completed, failed, or timeout")

    producer = _exact_mapping(
        result["producer"],
        f"{field}.producer",
        required=(
            "name",
            "version",
            "orchestration_sha256",
            "model_revision",
            "adapter_sha256",
            "seed_honored",
            "fallback_models_used",
        ),
    )
    _schema_string(producer["name"], f"{field}.producer.name")
    _schema_string(producer["version"], f"{field}.producer.version")
    _schema_string(
        producer["orchestration_sha256"],
        f"{field}.producer.orchestration_sha256",
        r"sha256:[0-9a-fA-F]{64}",
    )
    _schema_string(
        producer["model_revision"],
        f"{field}.producer.model_revision",
        r"[0-9a-fA-F]{40,64}",
    )
    adapter_digest = producer["adapter_sha256"]
    if adapter_digest is not None:
        _schema_string(
            adapter_digest,
            f"{field}.producer.adapter_sha256",
            r"[0-9a-fA-F]{64}",
        )
    if producer["seed_honored"] is not True:
        raise EvaluationError(f"{field}.producer.seed_honored must be true")
    if producer["fallback_models_used"] is not False:
        raise EvaluationError(f"{field}.producer.fallback_models_used must be false")

    usage = _exact_mapping(
        result["usage"],
        f"{field}.usage",
        required=(),
        optional=("input_tokens", "output_tokens", "tool_calls", "wall_time_seconds"),
    )
    for key in ("input_tokens", "output_tokens", "tool_calls"):
        if key in usage:
            _schema_non_negative_integer(usage[key], f"{field}.usage.{key}")
    if "wall_time_seconds" in usage:
        duration = usage["wall_time_seconds"]
        if (
            not isinstance(duration, (int, float))
            or isinstance(duration, bool)
            or not math.isfinite(float(duration))
            or duration < 0
        ):
            raise EvaluationError(f"{field}.usage.wall_time_seconds must be non-negative")

    output = _exact_mapping(
        result["output"],
        f"{field}.output",
        required=(),
        optional=("patch", "transcript", "review_artifact"),
    )
    for key, path_value in output.items():
        path_text = _schema_string(path_value, f"{field}.output.{key}")
        # Match the schema using both slash styles even on non-Windows hosts.
        if re.search(r"^(?:[A-Za-z]:|[/\\])", path_text) or re.search(
            r"(?:^|[/\\])\.\.(?:[/\\]|$)", path_text
        ):
            raise EvaluationError(f"{field}.output.{key} must be a relative file path")
    return result


def _validate_review_row(value: Any, field: str) -> dict[str, Any]:
    """Mirror ``schemas/blind-review-row.schema.json`` before normalization."""

    row = _exact_mapping(
        value,
        field,
        required=("pair_id", "reviewer_id", "choice"),
    )
    _schema_string(row["pair_id"], f"{field}.pair_id", r"pair-[0-9a-f]{24}")
    _schema_string(row["reviewer_id"], f"{field}.reviewer_id")
    choice = _schema_string(row["choice"], f"{field}.choice")
    if choice not in {"left", "right", "tie", "both_fail"}:
        raise EvaluationError(f"{field}.choice is not a supported blind-review choice")
    return row


def _artifact_dir(config: Mapping[str, Any], root: Path) -> Path:
    project = _mapping(config.get("project", {}), "project")
    value = Path(str(project.get("artifact_dir", ".autotrainer"))).expanduser()
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def _resolve_local(root: Path, value: Any, field: str) -> Path:
    text = _text(value, field)
    path = Path(text).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _relative_or_text(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _read_jsonl(path: Path, field: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise EvaluationError(f"{field} does not exist; run `autotrainer compile` first: {path}")
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise EvaluationError(f"{path}:{line_number} must contain a JSON object")
                rows.append(dict(value))
    except json.JSONDecodeError as error:
        raise EvaluationError(
            f"invalid JSON at {path}:{error.lineno}:{error.colno}: {error.msg}"
        ) from error
    if not rows:
        raise EvaluationError(f"{field} contains no records: {path}")
    return rows


def _evaluation_dataset(config: Mapping[str, Any], root: Path) -> Path:
    evaluation = _mapping(config.get("evaluation", {}), "evaluation")
    configured = evaluation.get("dataset")
    if not configured:
        raise EvaluationError("evaluation.dataset is required")
    return _resolve_local(root, configured, "evaluation.dataset")


def _training_dataset(config: Mapping[str, Any], root: Path) -> Path:
    grpo = _mapping(config.get("grpo", {}), "grpo")
    configured = grpo.get("dataset")
    if not configured:
        raise EvaluationError(
            "grpo.dataset is required to prove repository holdout before evaluation"
        )
    return _resolve_local(root, configured, "grpo.dataset")


def _repository_keys(
    rows: Sequence[Mapping[str, Any]], field: str
) -> tuple[set[str], set[str]]:
    """Extract compiler-attested repository identities and locked revisions."""

    identities: set[str] = set()
    revisions: set[str] = set()
    for index, row in enumerate(rows):
        task_id = str(row.get("task_id", f"row-{index + 1}"))
        identity = row.get("source_repository_identity")
        if not isinstance(identity, str) or not identity.strip():
            # Evaluation may be invoked directly, without the static planner.
            # Missing compiler provenance therefore fails closed here.
            raise EvaluationError(
                f"{field} task {task_id!r} lacks source_repository_identity; "
                "re-run source scan and compilation"
            )
        revision = row.get("source_revision")
        if not isinstance(revision, str) or not revision.strip():
            raise EvaluationError(
                f"{field} task {task_id!r} lacks a locked source_revision; "
                "re-run source scan and compilation"
            )
        identities.add(identity.strip())
        revisions.add(revision.strip().lower())
    return identities, revisions


def _compiler_repository_provenance(
    config: Mapping[str, Any],
    root: Path,
    training_dataset: Path,
    evaluation_dataset: Path,
) -> dict[str, Any]:
    """Verify the compiler ledger for every declared repository exposure."""

    report_path = _artifact_dir(config, root) / "compiled" / "compile-report.json"
    if not report_path.is_file():
        raise EvaluationError(
            "compiler provenance report is missing; run `autotrainer compile` before evaluation: "
            f"{report_path}"
        )
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EvaluationError(f"compiler provenance report is unreadable: {report_path}: {error}") from error
    if not isinstance(report, Mapping) or report.get("errors"):
        raise EvaluationError("compiler provenance report is invalid or records compilation errors")

    artifacts = report.get("artifacts")
    hashes = report.get("artifact_sha256")
    if not isinstance(artifacts, Mapping) or not isinstance(hashes, Mapping):
        raise EvaluationError("compiler provenance report lacks artifact paths or SHA-256 digests")
    for key, expected_path in (
        ("rl_train", training_dataset),
        ("rl_evaluation", evaluation_dataset),
    ):
        reported_value = artifacts.get(key)
        if not isinstance(reported_value, str) or not reported_value.strip():
            raise EvaluationError(f"compiler provenance report lacks {key} artifact")
        reported_path = Path(reported_value).expanduser()
        if not reported_path.is_absolute():
            reported_path = root / reported_path
        if reported_path.resolve() != expected_path.resolve():
            raise EvaluationError(
                f"configured {key} dataset does not match compiler provenance: "
                f"{expected_path} != {reported_path.resolve()}"
            )
        expected_digest = hashes.get(key)
        actual_digest = _sha256_file(expected_path)
        if not isinstance(expected_digest, str) or expected_digest != actual_digest:
            raise EvaluationError(
                f"compiled {key} dataset bytes do not match compiler provenance"
            )

    raw_exposures = report.get("repository_exposures")
    if not isinstance(raw_exposures, list) or not raw_exposures:
        raise EvaluationError("compiler provenance report has no repository exposure ledger")
    exposures: list[dict[str, str]] = []
    for index, raw in enumerate(raw_exposures):
        if not isinstance(raw, Mapping):
            raise EvaluationError(f"compiler repository exposure {index} must be a mapping")
        partition = raw.get("partition")
        identity = raw.get("repository_identity")
        commit = raw.get("commit")
        source_id = raw.get("source_id")
        if partition not in {"train", "evaluation"}:
            raise EvaluationError(
                f"compiler repository exposure {index} has invalid partition {partition!r}"
            )
        if not isinstance(identity, str) or not identity.strip():
            raise EvaluationError(
                f"compiler repository exposure {index} lacks repository_identity"
            )
        if not isinstance(commit, str) or not commit.strip():
            raise EvaluationError(f"compiler repository exposure {index} lacks a locked commit")
        exposures.append(
            {
                "source_id": str(source_id or f"source-{index + 1}"),
                "partition": str(partition),
                "repository_identity": identity.strip(),
                "commit": commit.strip().lower(),
            }
        )

    train = [item for item in exposures if item["partition"] == "train"]
    held_out = [item for item in exposures if item["partition"] == "evaluation"]
    collisions: set[str] = set()
    for train_item in train:
        for evaluation_item in held_out:
            shared_identity = (
                train_item["repository_identity"]
                == evaluation_item["repository_identity"]
            )
            shared_commit = train_item["commit"] == evaluation_item["commit"]
            if shared_identity or shared_commit:
                reason = "repository identity" if shared_identity else "exact commit"
                collisions.add(
                    f"{train_item['source_id']} and {evaluation_item['source_id']} share {reason}"
                )
    if collisions:
        raise EvaluationError(
            "repository holdout is violated by compiler-frozen repository exposure: "
            + "; ".join(sorted(collisions))
        )

    # Sorting makes the plan independent of scan declaration order while the
    # report digest below still detects any byte-level provenance replacement.
    exposures.sort(
        key=lambda item: (
            item["partition"],
            item["repository_identity"],
            item["commit"],
            item["source_id"],
        )
    )
    return {
        "path": _relative_or_text(report_path, root),
        "sha256": _sha256_file(report_path),
        "fingerprint": report.get("fingerprint"),
        "repository_exposures": exposures,
    }


def _task_identity(row: Mapping[str, Any], index: int) -> tuple[str, dict[str, Any]]:
    manifest = _mapping(row.get("manifest", {}), f"evaluation task {index}.manifest")
    task = _mapping(manifest.get("task", {}), f"evaluation task {index}.manifest.task")
    task_id = _text(row.get("task_id", task.get("id")), f"evaluation task {index}.task_id")
    if task.get("id") != task_id:
        raise EvaluationError(f"evaluation task row {task_id!r} disagrees with manifest.task.id")
    if task.get("split") != "evaluation":
        raise EvaluationError(f"evaluation task {task_id!r} must use task.split=\"evaluation\"")
    if not row.get("source_revision"):
        raise EvaluationError(f"evaluation task {task_id!r} has no locked source revision")
    return task_id, manifest


def _resolved_arm(
    arm_id: str,
    arm: Mapping[str, Any],
    config: Mapping[str, Any],
    root: Path,
) -> dict[str, Any]:
    # "project" means the exact locked base model at the top level. Reference
    # arms may instead pin a different 9B quality bar without changing training.
    model_value = arm.get("model")
    if model_value == "project":
        model = _mapping(config.get("model", {}), "model")
    else:
        model = _mapping(model_value, f"evaluation.arms.{arm_id}.model")
    revision = _text(model.get("revision"), f"evaluation.arms.{arm_id}.model.revision")
    if not all(character in "0123456789abcdefABCDEF" for character in revision) or not 40 <= len(revision) <= 64:
        raise EvaluationError(
            f"evaluation arm {arm_id!r} must resolve to an immutable 40-64 character model revision"
        )
    model_id = _text(model.get("id"), f"evaluation.arms.{arm_id}.model.id")
    if model_id.startswith("REPLACE_WITH_") or set(revision.lower()) == {"0"}:
        raise EvaluationError(f"evaluation arm {arm_id!r} still contains a placeholder model pin")
    resolved: dict[str, Any] = {
        "id": arm_id,
        "label": _text(arm.get("label", arm_id), f"evaluation.arms.{arm_id}.label"),
        "role": _text(arm.get("role"), f"evaluation.arms.{arm_id}.role"),
        "parameter_class": _text(
            arm.get("parameter_class", "9b"), f"evaluation.arms.{arm_id}.parameter_class"
        ),
        "model": {
            "provider": model.get("provider", "huggingface"),
            "id": model_id,
            "revision": revision,
            "loader": model.get("loader", "auto_text_causal_lm"),
            "trust_remote_code": False,
        },
        "adapter": None,
    }
    # The adapter digest, not merely its directory name, is part of the plan.
    # Re-training therefore creates a new plan even if output paths are reused.
    adapter_value = arm.get("adapter")
    if adapter_value is not None:
        adapter = _mapping(adapter_value, f"evaluation.arms.{arm_id}.adapter")
        adapter_path = _resolve_local(
            root, adapter.get("path"), f"evaluation.arms.{arm_id}.adapter.path"
        )
        resolved["adapter"] = {
            "path": _relative_or_text(adapter_path, root),
            "stage": _text(
                adapter.get("stage", "grpo"), f"evaluation.arms.{arm_id}.adapter.stage"
            ),
            "sha256": _sha256_tree(adapter_path),
        }
    return resolved


def _resolved_runner(suite_id: str, suite: Mapping[str, Any]) -> dict[str, Any]:
    runner = _mapping(suite.get("runner", {}), f"evaluation.suites.{suite_id}.runner")
    runner_type = _text(runner.get("type"), f"evaluation.suites.{suite_id}.runner.type")
    if runner_type not in {"command", "external"}:
        raise EvaluationError(f"evaluation suite {suite_id!r} runner type must be command or external")
    producer = _text(
        runner.get("producer", "local-command"),
        f"evaluation.suites.{suite_id}.runner.producer",
    )
    version = _text(runner.get("version"), f"evaluation.suites.{suite_id}.runner.version")
    if version.startswith("REPLACE_WITH_"):
        raise EvaluationError(f"evaluation suite {suite_id!r} still contains a placeholder runner version")
    # Model weights alone do not define an agent run. The orchestrator digest
    # freezes prompts, tool routing, fallbacks, and other behavior around them.
    orchestration = _text(
        runner.get("orchestration_sha256"),
        f"evaluation.suites.{suite_id}.runner.orchestration_sha256",
    )
    orchestration_hex = orchestration.removeprefix("sha256:")
    if len(orchestration_hex) != 64 or any(
        character not in "0123456789abcdefABCDEF" for character in orchestration_hex
    ):
        raise EvaluationError(
            f"evaluation suite {suite_id!r} runner orchestration_sha256 must be a SHA-256 digest"
        )
    if set(orchestration_hex.lower()) == {"0"}:
        raise EvaluationError(
            f"evaluation suite {suite_id!r} still contains a placeholder orchestration digest"
        )
    resolved: dict[str, Any] = {
        "type": runner_type,
        "producer": producer,
        "version": version,
        "orchestration_sha256": f"sha256:{orchestration_hex.lower()}",
    }
    if runner_type == "command":
        argv = runner.get("argv")
        if not isinstance(argv, list) or not argv or not all(
            isinstance(item, str) and item for item in argv
        ):
            raise EvaluationError(
                f"evaluation suite {suite_id!r} command runner requires a non-empty argv list"
            )
        resolved["argv"] = list(argv)
    return resolved


def build_evaluation_plan(config: Mapping[str, Any], project_root: Path) -> dict[str, Any]:
    """Freeze the exact tasks, arms, suites, seeds, and runtime fingerprints."""

    root = Path(project_root).expanduser().resolve()
    evaluation = _mapping(config.get("evaluation", {}), "evaluation")
    arms_value = _mapping(evaluation.get("arms", {}), "evaluation.arms")
    suites_value = _mapping(evaluation.get("suites", {}), "evaluation.suites")
    fairness = _mapping(evaluation.get("fairness", {}), "evaluation.fairness")
    decisions = _mapping(evaluation.get("decisions", {}), "evaluation.decisions")
    confidence = decisions.get("confidence")
    if (
        not isinstance(confidence, (int, float))
        or isinstance(confidence, bool)
        or not 0.5 < float(confidence) < 1.0
    ):
        raise EvaluationError("evaluation.decisions.confidence must be between 0.5 and 1")
    repetitions = evaluation.get("repetitions")
    seeds = evaluation.get("seeds")
    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions < 1:
        raise EvaluationError("evaluation.repetitions must be a positive integer")
    if not isinstance(seeds, list) or len(seeds) != repetitions or not all(
        isinstance(seed, int) and not isinstance(seed, bool) and seed >= 0 for seed in seeds
    ):
        raise EvaluationError("evaluation.seeds must contain one non-negative integer per repetition")

    if evaluation.get("holdout_unit") != "repository":
        raise EvaluationError("evaluation.holdout_unit must be repository")

    # The static planner is advisory and can be bypassed by invoking evaluation
    # directly. Compare the two compiled datasets again at this execution gate.
    # IDs are labels; scanner-derived identity and exact revision are provenance.
    training_dataset_path = _training_dataset(config, root)
    dataset_path = _evaluation_dataset(config, root)
    grpo = _mapping(config.get("grpo", {}), "grpo")
    training_eval_value = grpo.get("eval_dataset")
    if training_eval_value:
        training_eval_path = _resolve_local(root, training_eval_value, "grpo.eval_dataset")
        if training_eval_path == dataset_path:
            # Public library callers can bypass semantic config validation. The
            # execution gate therefore repeats this anti-leakage invariant.
            raise EvaluationError(
                "grpo.eval_dataset must be separate from evaluation.dataset; "
                "training validation cannot reuse the final benchmark"
            )
    training_rows = _read_jsonl(training_dataset_path, "compiled GRPO training dataset")
    task_rows = _read_jsonl(dataset_path, "compiled evaluation dataset")
    training_identities, training_revisions = _repository_keys(
        training_rows, "compiled GRPO training dataset"
    )
    evaluation_identities, evaluation_revisions = _repository_keys(
        task_rows, "compiled evaluation dataset"
    )
    compiler_provenance = _compiler_repository_provenance(
        config, root, training_dataset_path, dataset_path
    )
    shared_identities = sorted(training_identities & evaluation_identities)
    shared_revisions = sorted(training_revisions & evaluation_revisions)
    if shared_identities or shared_revisions:
        reasons: list[str] = []
        if shared_identities:
            reasons.append("repository identity " + ", ".join(shared_identities))
        if shared_revisions:
            reasons.append("exact source revision " + ", ".join(shared_revisions))
        raise EvaluationError(
            "repository holdout is violated; training and evaluation datasets share "
            + " and ".join(reasons)
        )

    tasks: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(task_rows):
        task_id, manifest = _task_identity(row, index)
        if task_id in tasks:
            raise EvaluationError(f"duplicate evaluation task id: {task_id}")
        tasks[task_id] = {
            "task_id": task_id,
            "group_id": manifest["task"].get("groupId"),
            "source_id": manifest["task"].get("sourceId"),
            "source_repository_identity": row.get("source_repository_identity"),
            "source_revision": row.get("source_revision"),
            "fingerprint": f"sha256:{_digest(row)}",
            "row": row,
        }

    arms = {
        arm_id: _resolved_arm(arm_id, _mapping(value, f"evaluation.arms.{arm_id}"), config, root)
        for arm_id, value in sorted(arms_value.items())
    }
    suites: dict[str, dict[str, Any]] = {}
    for suite_id, value in sorted(suites_value.items()):
        suite = _mapping(value, f"evaluation.suites.{suite_id}")
        suite_arms = suite.get("arms")
        if not isinstance(suite_arms, list) or len(suite_arms) != 2 or len(set(suite_arms)) != 2:
            raise EvaluationError(f"evaluation suite {suite_id!r} must declare exactly two arms")
        missing = [arm_id for arm_id in suite_arms if arm_id not in arms]
        if missing:
            raise EvaluationError(
                f"evaluation suite {suite_id!r} refers to unknown arms: {', '.join(missing)}"
            )
        suites[suite_id] = {
            "kind": _text(suite.get("kind"), f"evaluation.suites.{suite_id}.kind"),
            "arms": list(suite_arms),
            "runner": _resolved_runner(suite_id, suite),
            "review": suite.get("review"),
        }

    environment = _mapping(config.get("environment", {}), "environment")
    plan_input: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "project": _mapping(config.get("project", {}), "project").get("name"),
        "task_source": {
            "path": _relative_or_text(dataset_path, root),
            "sha256": _sha256_file(dataset_path),
        },
        # Freeze the evidence used for the holdout decision into the plan ID.
        # Changing either compiled dataset invalidates the paired trial matrix.
        "holdout": {
            "unit": "repository",
            "training_source": {
                "path": _relative_or_text(training_dataset_path, root),
                "sha256": _sha256_file(training_dataset_path),
            },
            "training_repository_identities": sorted(training_identities),
            "training_revisions": sorted(training_revisions),
            "evaluation_repository_identities": sorted(evaluation_identities),
            "evaluation_revisions": sorted(evaluation_revisions),
            "compiler_provenance": compiler_provenance,
        },
        "repetitions": repetitions,
        "seeds": list(seeds),
        "environment": {
            key: environment.get(key)
            for key in (
                "factory",
                "backend",
                "image",
                "network",
                "max_tool_output_chars",
                "episode_timeout_seconds",
            )
        },
        "fairness": fairness,
        "arms": arms,
        "suites": suites,
        "tasks": [
            {key: value for key, value in tasks[task_id].items() if key != "row"}
            for task_id in sorted(tasks)
        ],
        # Full rows (including hidden verifier paths) remain only in the local
        # plan. External exports are derived through _public_task below.
        "task_rows": {task_id: tasks[task_id]["row"] for task_id in sorted(tasks)},
        "decisions": decisions,
    }
    plan_id = f"sha256:{_digest(plan_input)}"
    trials: list[dict[str, Any]] = []
    for suite_index, (suite_id, suite) in enumerate(sorted(suites.items())):
        suite_arms = list(suite["arms"])
        for task_index, task_id in enumerate(sorted(tasks)):
            for repetition, seed in enumerate(seeds):
                # Counterbalance left/first position deterministically. This is
                # predictable for reproduction but alternates across pairs.
                ordered_arms = (
                    suite_arms
                    if (suite_index + task_index + repetition) % 2 == 0
                    else list(reversed(suite_arms))
                )
                for sequence, arm_id in enumerate(ordered_arms):
                    identity = {
                        "plan_id": plan_id,
                        "suite_id": suite_id,
                        "arm_id": arm_id,
                        "task_id": task_id,
                        "repetition": repetition,
                        "seed": seed,
                    }
                    trials.append(
                        {
                            **identity,
                            "trial_id": f"trial-{_digest(identity)[:24]}",
                            "sequence": sequence,
                            "task_fingerprint": tasks[task_id]["fingerprint"],
                        }
                    )
    plan = {**plan_input, "plan_id": plan_id, "trials": trials}
    return plan


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8", newline="\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, values: Iterable[Mapping[str, Any]]) -> None:
    _atomic_text(path, "".join(json.dumps(value, sort_keys=True) + "\n" for value in values))


def evaluation_run_dir(config: Mapping[str, Any], root: Path, plan_id: str) -> Path:
    digest = plan_id.removeprefix("sha256:")
    if len(digest) != 64:
        raise EvaluationError(f"invalid evaluation plan id: {plan_id}")
    return _artifact_dir(config, root) / "evaluation" / digest


def write_evaluation_plan(config: Mapping[str, Any], project_root: Path) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    plan = build_evaluation_plan(config, root)
    run_dir = evaluation_run_dir(config, root, plan["plan_id"])
    plan_path = run_dir / "evaluation-plan.json"
    # A plan ID is immutable. Reusing its directory with different bytes is
    # treated as tampering instead of an invitation to overwrite evidence.
    if plan_path.exists():
        existing = json.loads(plan_path.read_text(encoding="utf-8"))
        if existing != plan:
            raise EvaluationError(f"existing evaluation plan was modified: {plan_path}")
    else:
        _write_json(plan_path, plan)
        _write_jsonl(run_dir / "trials.jsonl", plan["trials"])
    pointer = _artifact_dir(config, root) / "evaluation" / "current-plan.json"
    _write_json(pointer, {"plan_id": plan["plan_id"], "path": str(plan_path)})
    return {**plan, "artifact": str(plan_path)}


def load_current_plan(config: Mapping[str, Any], project_root: Path) -> tuple[dict[str, Any], Path]:
    root = Path(project_root).expanduser().resolve()
    pointer = _artifact_dir(config, root) / "evaluation" / "current-plan.json"
    if not pointer.is_file():
        raise EvaluationError("no evaluation plan exists; run `autotrainer evaluate plan --write`")
    try:
        pointer_value = json.loads(pointer.read_text(encoding="utf-8"))
        plan_id = _text(pointer_value.get("plan_id"), "current plan id")
        plan_path = Path(_text(pointer_value.get("path"), "current plan path")).resolve()
        expected_dir = evaluation_run_dir(config, root, plan_id).resolve()
        plan_path.relative_to(expected_dir)
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise EvaluationError(f"current evaluation plan is unreadable: {error}") from error
    if plan.get("plan_id") != plan_id:
        raise EvaluationError("current evaluation plan pointer does not match the plan document")
    return plan, expected_dir


def _public_task(row: Mapping[str, Any]) -> dict[str, Any]:
    public = json.loads(json.dumps(row))
    manifest = public.get("manifest")
    if isinstance(manifest, dict):
        # An external producer receives the brief and bounded tools, never the
        # hidden verifier location or reward weights it could optimize directly.
        manifest.pop("verifier", None)
        manifest.pop("rewards", None)
    public.pop("manifest_path", None)
    public.pop("task_root", None)
    return public


def _request_for(plan: Mapping[str, Any], trial: Mapping[str, Any]) -> dict[str, Any]:
    task_rows = _mapping(plan.get("task_rows", {}), "evaluation plan task_rows")
    task_row = _mapping(
        task_rows.get(trial["task_id"]),
        f"evaluation plan task_rows.{trial['task_id']}",
    )
    suite = plan["suites"][trial["suite_id"]]
    return {
        "schema_version": SCHEMA_VERSION,
        "plan_id": plan["plan_id"],
        "trial_id": trial["trial_id"],
        "suite_id": trial["suite_id"],
        "arm_id": trial["arm_id"],
        "task_id": trial["task_id"],
        "repetition": trial["repetition"],
        "seed": trial["seed"],
        "sequence": trial["sequence"],
        "candidate": plan["arms"][trial["arm_id"]],
        "runner": suite["runner"],
        "task": _public_task(task_row),
        "result_contract": {
            "format": "autotrainer-evaluation-result-v1",
            "scores_are_ignored": True,
            "completed_output": "output.patch names a unified Git patch relative to the result envelope",
            "directory_envelope_names": ["result.json", "*.result.json"],
            "failed_statuses": ["failed", "timeout"],
        },
    }


def export_evaluation_suite(
    config: Mapping[str, Any],
    project_root: Path,
    suite_id: str,
    output_dir: Path,
) -> dict[str, Any]:
    plan, _ = load_current_plan(config, project_root)
    if suite_id not in plan["suites"]:
        raise EvaluationError(f"unknown evaluation suite: {suite_id}")
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    exported: list[str] = []
    for trial in plan["trials"]:
        if trial["suite_id"] != suite_id:
            continue
        request = _request_for(plan, trial)
        path = destination / f"{trial['trial_id']}.json"
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing != request:
                raise EvaluationError(f"refusing to overwrite changed request: {path}")
        else:
            _write_json(path, request)
        exported.append(str(path))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "plan_id": plan["plan_id"],
        "suite_id": suite_id,
        "request_count": len(exported),
        "requests": exported,
    }
    _write_json(destination / "export-manifest.json", manifest)
    return manifest


def _result_documents(path: Path) -> list[tuple[dict[str, Any], Path]]:
    unresolved_source = Path(path).expanduser()
    # Keep the unresolved input long enough to detect an explicitly supplied
    # symlink; resolving first would erase the only portable identity check.
    if unresolved_source.is_symlink():
        raise EvaluationError(f"result envelope input must not be a symlink: {unresolved_source}")
    source = unresolved_source.resolve()
    if source.is_dir():
        # JSON evidence is valid, so extension-only discovery is ambiguous.
        # Reserve two explicit names for directory ingestion while preserving
        # arbitrary filenames when the caller supplies one file directly.
        files = []
        for candidate in sorted(source.rglob("*.json")):
            if not (
                candidate.name in RESULT_ENVELOPE_NAMES
                or candidate.name.endswith(RESULT_ENVELOPE_SUFFIX)
            ):
                continue
            # Reserved envelope names are authoritative: reject a link or a
            # junction escape instead of silently ignoring or following it.
            if candidate.is_symlink():
                raise EvaluationError(f"result envelope must not be a symlink: {candidate}")
            resolved_candidate = candidate.resolve()
            try:
                resolved_candidate.relative_to(source)
            except ValueError as error:
                raise EvaluationError(
                    f"result envelope escapes the supplied directory: {candidate}"
                ) from error
            if not resolved_candidate.is_file():
                raise EvaluationError(f"result envelope must be a regular file: {candidate}")
            files.append(resolved_candidate)
    else:
        files = [source]
    if not files:
        raise EvaluationError(
            "no result envelopes were found; directory inputs require "
            f"result.json or *{RESULT_ENVELOPE_SUFFIX}: {source}"
        )
    documents: list[tuple[dict[str, Any], Path]] = []
    for result_path in files:
        try:
            value = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise EvaluationError(f"could not read result {result_path}: {error}") from error
        if not isinstance(value, Mapping):
            raise EvaluationError(f"result must contain one JSON object: {result_path}")
        documents.append((dict(value), result_path))
    if not documents:
        raise EvaluationError(f"no result envelopes were found in {source}")
    return documents


def _safe_result_file(result_path: Path, value: Any, field: str) -> Path:
    relative = Path(_text(value, field))
    if relative.is_absolute() or ".." in relative.parts:
        raise EvaluationError(f"{field} must be relative to the result envelope")
    # Resolve before reading so `..`, directory junctions, and symlinks cannot
    # turn an envelope into an arbitrary host-file reader.
    candidate = (result_path.parent / relative).resolve()
    try:
        candidate.relative_to(result_path.parent.resolve())
    except ValueError as error:
        raise EvaluationError(f"{field} escapes the result directory") from error
    if candidate.is_symlink() or not candidate.is_file():
        raise EvaluationError(f"{field} must name a regular file: {candidate}")
    if candidate.stat().st_size > MAX_RESULT_ARTIFACT_BYTES:
        raise EvaluationError(f"{field} exceeds the 10 MiB artifact limit: {candidate}")
    return candidate


def _is_link_or_reparse(path: Path) -> bool:
    """Detect links, including Windows reparse points, without following them."""

    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return False
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    file_attributes = getattr(metadata, "st_file_attributes", 0)
    return stat.S_ISLNK(metadata.st_mode) or bool(file_attributes & reparse_flag)


def _verify_evidence_target(target: Path, digest: str, size: int) -> None:
    """Accept an existing content address only when its regular bytes match."""

    if _is_link_or_reparse(target) or not target.is_file():
        raise EvaluationError(f"content-addressed evidence target is not a regular file: {target}")
    if target.stat().st_size != size or _sha256_file(target) != digest:
        raise EvaluationError(f"content-addressed evidence was modified: {target}")


def _copy_evidence(source: Path, destination: Path) -> dict[str, Any]:
    """Copy once while hashing the exact bounded bytes that will be scored."""

    destination.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination,
        prefix=".incoming-evidence.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    digest = hashlib.sha256()
    copied_bytes = 0
    try:
        # The producer may still mutate its path. Holding one input descriptor
        # and hashing bytes as they enter our private file guarantees the label,
        # stored artifact, and bytes later scored all describe the same copy.
        with os.fdopen(descriptor, "wb") as output_handle:
            with source.open("rb") as input_handle:
                if not stat.S_ISREG(os.fstat(input_handle.fileno()).st_mode):
                    raise EvaluationError(f"result evidence must be a regular file: {source}")
                for chunk in iter(lambda: input_handle.read(1024 * 1024), b""):
                    copied_bytes += len(chunk)
                    if copied_bytes > MAX_RESULT_ARTIFACT_BYTES:
                        raise EvaluationError(
                            f"result evidence exceeds the 10 MiB artifact limit: {source}"
                        )
                    digest.update(chunk)
                    output_handle.write(chunk)
            output_handle.flush()
            os.fsync(output_handle.fileno())

        digest_hex = digest.hexdigest()
        # Preserve ordinary viewer hints without allowing an untrusted filename
        # to introduce an alternate-data-stream marker or an oversized target.
        suffix = source.suffix.lower()
        if re.fullmatch(r"\.[a-z0-9]{1,16}", suffix) is None:
            suffix = ".bin"
        target = destination / f"sha256-{digest_hex}{suffix}"
        if _is_link_or_reparse(target):
            raise EvaluationError(
                f"content-addressed evidence target must not be a link: {target}"
            )
        try:
            # A same-directory hard link installs the fully written private file
            # without an overwrite window. Concurrent identical ingestion may
            # win the name, but any pre-existing bytes are verified below.
            os.link(temporary, target)
        except FileExistsError:
            _verify_evidence_target(target, digest_hex, copied_bytes)
        else:
            _verify_evidence_target(target, digest_hex, copied_bytes)
        return {
            "path": str(target),
            "sha256": digest_hex,
            "bytes": copied_bytes,
        }
    finally:
        temporary.unlink(missing_ok=True)


def _producer_fairness(
    result: Mapping[str, Any],
    trial: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    producer = _mapping(result.get("producer", {}), "result.producer")
    runner = plan["suites"][trial["suite_id"]]["runner"]
    arm = plan["arms"][trial["arm_id"]]
    expected_adapter = arm.get("adapter")
    # Producer metadata is evidence, not a score. Every claimed runtime pin must
    # match the preregistered plan before local verification is considered fair.
    checks = {
        "producer": producer.get("name") == runner["producer"],
        "producer_version": producer.get("version") == runner["version"],
        "orchestration": producer.get("orchestration_sha256")
        == runner["orchestration_sha256"],
        "model_revision": producer.get("model_revision") == arm["model"]["revision"],
        "adapter": producer.get("adapter_sha256")
        == (expected_adapter["sha256"] if expected_adapter else None),
        "seed_honored": producer.get("seed_honored") is True,
        "no_fallback_models": producer.get("fallback_models_used") is False,
    }
    return {"passed": all(checks.values()), "checks": checks}


def _episode_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "to_mapping") and callable(value.to_mapping):
        mapped = value.to_mapping()
        if isinstance(mapped, Mapping):
            return dict(mapped)
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    raise EvaluationError("environment scorer returned an unsupported episode result")


def _normalized_episode(value: Any) -> tuple[bool, str | None, float, dict[str, float], dict[str, Any]]:
    episode = _episode_mapping(value)
    reward_value = episode.get("reward", episode.get("total", 0.0))
    reward = float(reward_value)
    if not math.isfinite(reward) or not 0 <= reward <= 1:
        raise EvaluationError("environment reward must be finite and between 0 and 1")
    gate_reason = episode.get("gate_reason", episode.get("hard_gate_reason"))
    gated = bool(episode.get("gated", gate_reason is not None))
    raw_signals = episode.get(
        "raw_verifier_rates",
        episode.get("signals", episode.get("weighted_signals", episode.get("components", {}))),
    )
    if not isinstance(raw_signals, Mapping):
        raw_signals = {}
    components = {
        name: float(raw_signals.get(name, 0.0)) for name in RESULT_COMPONENTS
    }
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in components.values()):
        raise EvaluationError("environment component scores must be finite rates")
    verified = not gated and components["task_tests"] >= 1.0
    if not verified and not gate_reason:
        gate_reason = "verification_incomplete"
    # Evaluation uses verified task success as the hard gate. Partial verifier
    # signals remain visible in components, but an unverified trial contributes
    # zero reward to the comparison and can never outrank a completed task.
    if not verified:
        reward = 0.0
    metadata = {
        key: value
        for key, value in episode.items()
        if key
        not in {
            "reward",
            "total",
            "gated",
            "gate_reason",
            "hard_gate_reason",
            "raw_verifier_rates",
            "signals",
            "weighted_signals",
            "components",
        }
    }
    return verified, str(gate_reason) if gate_reason else None, reward, components, metadata


def _default_patch_scorer(task_row: Mapping[str, Any], patch: str) -> Any:
    from .environments.frontend import evaluate_patch

    return evaluate_patch(task_row, patch)


def _zero_scored_result(
    trial: Mapping[str, Any],
    *,
    reason: str,
    status: str,
    fairness: Mapping[str, Any],
    usage: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "plan_id": trial["plan_id"],
        "trial_id": trial["trial_id"],
        "suite_id": trial["suite_id"],
        "candidate_id": trial["arm_id"],
        "task_id": trial["task_id"],
        "repetition": trial["repetition"],
        "seed": trial["seed"],
        "status": status,
        "hard_gate_passed": False,
        "gate_reason": reason,
        "reward": 0.0,
        "components": {name: 0.0 for name in RESULT_COMPONENTS},
        "metadata": {
            "fairness": dict(fairness),
            "usage": dict(usage),
            "evidence": dict(evidence),
        },
    }


def ingest_evaluation_results(
    config: Mapping[str, Any],
    project_root: Path,
    suite_id: str,
    input_path: Path,
    *,
    scorer: Callable[[Mapping[str, Any], str], Any] | None = None,
) -> dict[str, Any]:
    """Validate external envelopes and locally score their patches exactly once."""

    plan, run_dir = load_current_plan(config, project_root)
    if suite_id not in plan["suites"]:
        raise EvaluationError(f"unknown evaluation suite: {suite_id}")
    trial_by_id = {
        trial["trial_id"]: trial
        for trial in plan["trials"]
        if trial["suite_id"] == suite_id
    }
    active_scorer = scorer or _default_patch_scorer
    ingested: list[str] = []
    for result, result_path in _result_documents(Path(input_path)):
        # Keep runtime behavior in lockstep with the published producer schema;
        # silently normalizing extra or mistyped fields would make that contract
        # advisory precisely where untrusted data crosses into trusted reports.
        result = _validate_result_envelope(result, f"result {result_path}")
        trial_id = _text(result.get("trial_id"), "result.trial_id")
        trial = trial_by_id.get(trial_id)
        if trial is None:
            raise EvaluationError(f"result refers to an unplanned trial in {suite_id}: {trial_id}")
        for field in ("plan_id", "suite_id", "arm_id", "task_id", "repetition", "seed"):
            if result.get(field) != trial.get(field):
                raise EvaluationError(
                    f"result {trial_id} changed planned field {field}: "
                    f"expected {trial.get(field)!r}, got {result.get(field)!r}"
                )
        raw_path = run_dir / "raw" / suite_id / f"{trial_id}.json"
        scored_path = run_dir / "scored-trials" / f"{trial_id}.json"
        if raw_path.exists() or scored_path.exists():
            raise EvaluationError(f"duplicate result refused for immutable trial: {trial_id}")

        fairness = _producer_fairness(result, trial, plan)
        usage = _mapping(result["usage"], "result.usage")
        status = result.get("status")
        output = _mapping(result["output"], "result.output")
        # Copy submitted artifacts under content hashes before executing any
        # patch. The original envelope and evidence remain auditable afterward.
        evidence: dict[str, Any] = {}
        evidence_dir = run_dir / "evidence"
        for field in ("patch", "transcript", "review_artifact"):
            if output.get(field):
                source = _safe_result_file(result_path, output[field], f"result.output.{field}")
                evidence[field] = _copy_evidence(source, evidence_dir)

        if not fairness["passed"]:
            scored = _zero_scored_result(
                trial,
                reason="fairness_failed",
                status="rejected",
                fairness=fairness,
                usage=usage,
                evidence=evidence,
            )
        elif status != "completed":
            scored = _zero_scored_result(
                trial,
                reason=f"producer_{status}",
                status=status,
                fairness=fairness,
                usage=usage,
                evidence=evidence,
            )
        elif "patch" not in evidence:
            scored = _zero_scored_result(
                trial,
                reason="missing_patch",
                status="failed",
                fairness=fairness,
                usage=usage,
                evidence=evidence,
            )
        else:
            patch_text = Path(evidence["patch"]["path"]).read_text(encoding="utf-8")
            task_row = plan["task_rows"][trial["task_id"]]
            episode = active_scorer(task_row, patch_text)
            verified, gate_reason, reward, components, episode_metadata = _normalized_episode(
                episode
            )
            scored = {
                "schema_version": SCHEMA_VERSION,
                "plan_id": trial["plan_id"],
                "trial_id": trial_id,
                "suite_id": suite_id,
                "candidate_id": trial["arm_id"],
                "task_id": trial["task_id"],
                "repetition": trial["repetition"],
                "seed": trial["seed"],
                "status": "completed",
                "hard_gate_passed": verified,
                "gate_reason": gate_reason,
                "reward": reward,
                "components": components,
                "metadata": {
                    "fairness": fairness,
                    "usage": usage,
                    "evidence": evidence,
                    "episode": episode_metadata,
                },
            }
        # Write the immutable raw/scored pair only after local scoring succeeds.
        # A missing container runtime can therefore be fixed and retried safely.
        _write_json(raw_path, result)
        _write_json(scored_path, scored)
        ingested.append(str(scored_path))
    return {
        "plan_id": plan["plan_id"],
        "suite_id": suite_id,
        "ingested_count": len(ingested),
        "scored_results": ingested,
    }


def _load_scored(run_dir: Path) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    directory = run_dir / "scored-trials"
    if not directory.exists():
        return results
    for path in sorted(directory.glob("*.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise EvaluationError(f"could not read scored trial {path}: {error}") from error
        if not isinstance(value, dict) or value.get("trial_id") in results:
            raise EvaluationError(f"invalid or duplicate scored trial: {path}")
        results[value["trial_id"]] = value
    return results


def _suite_payload(
    plan: Mapping[str, Any], suite_id: str, scored: Mapping[str, Mapping[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    suite = plan["suites"][suite_id]
    candidates = [
        {
            "id": arm_id,
            "label": plan["arms"][arm_id]["label"],
            "metadata": {
                "model": plan["arms"][arm_id]["model"],
                "adapter": plan["arms"][arm_id].get("adapter"),
                "role": plan["arms"][arm_id]["role"],
            },
        }
        for arm_id in suite["arms"]
    ]
    runs: list[dict[str, Any]] = []
    expected = 0
    present = 0
    fairness_passed = True
    for trial in plan["trials"]:
        if trial["suite_id"] != suite_id:
            continue
        expected += 1
        result = scored.get(trial["trial_id"])
        if result is None:
            # Missing trials stay in the denominator as explicit zeroes. Silent
            # row dropping would reward flaky runners and bias the comparison.
            result = _zero_scored_result(
                trial,
                reason="missing_result",
                status="missing",
                fairness={"passed": False, "checks": {}},
                usage={},
                evidence={},
            )
            fairness_passed = False
        else:
            present += 1
            fairness_passed = fairness_passed and bool(
                result.get("metadata", {}).get("fairness", {}).get("passed")
            )
        runs.append(
            {
                "candidate_id": result["candidate_id"],
                "task_id": result["task_id"],
                "repetition": result["repetition"],
                "seed": result["seed"],
                "hard_gate_passed": result["hard_gate_passed"],
                "gate_reason": result["gate_reason"],
                "reward": result["reward"],
                "components": result["components"],
                "metadata": result.get("metadata", {}),
            }
        )
    payload = {
        "schema_version": "1.0",
        "benchmark_id": f"{plan['plan_id']}:{suite_id}",
        "metadata": {
            "plan_id": plan["plan_id"],
            "suite_id": suite_id,
            "suite_kind": suite["kind"],
        },
        "candidates": candidates,
        "runs": runs,
    }
    completeness = {
        "expected_trials": expected,
        "completed_trials": present,
        "rate": round(present / expected, 6) if expected else 0.0,
        "fairness_passed": fairness_passed and present == expected,
    }
    return payload, completeness


def _deterministic_quantile(values: Sequence[float], probability: float) -> float:
    """Select an R-7 quantile without depending on library/version defaults."""

    position = probability * (len(values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    fraction = position - lower
    return values[lower] + (values[upper] - values[lower]) * fraction


def _paired_delta(
    payload: Mapping[str, Any],
    candidate: str,
    control: str,
    plan_id: str,
    confidence: float,
) -> dict[str, Any]:
    by_arm_task: dict[tuple[str, str], list[float]] = {}
    for run in payload["runs"]:
        key = (run["candidate_id"], run["task_id"])
        by_arm_task.setdefault(key, []).append(1.0 if run["hard_gate_passed"] else 0.0)
    tasks = sorted({run["task_id"] for run in payload["runs"]})
    differences = [
        sum(by_arm_task[(candidate, task)]) / len(by_arm_task[(candidate, task)])
        - sum(by_arm_task[(control, task)]) / len(by_arm_task[(control, task)])
        for task in tasks
    ]
    point = sum(differences) / len(differences)
    interval = None
    if len(differences) >= 2:
        # Resample whole tasks, not individual repetitions. Repetitions of one
        # website are correlated and must not masquerade as independent tasks.
        bootstrap_samples = 5000
        seed = int(
            _digest(
                {
                    "plan": plan_id,
                    "comparison": [candidate, control],
                }
            )[:16],
            16,
        )
        generator = random.Random(seed)
        samples = []
        for _ in range(bootstrap_samples):
            drawn = [generator.choice(differences) for _ in differences]
            samples.append(sum(drawn) / len(drawn))
        samples.sort()
        tail_probability = (1.0 - confidence) / 2.0
        interval = {
            "confidence": round(confidence, 6),
            "low": round(_deterministic_quantile(samples, tail_probability), 6),
            "high": round(
                _deterministic_quantile(samples, 1.0 - tail_probability), 6
            ),
            "lower_quantile": round(tail_probability, 6),
            "upper_quantile": round(1.0 - tail_probability, 6),
            "method": "task-clustered deterministic bootstrap",
            "bootstrap_samples": bootstrap_samples,
            "quantile_method": "R-7 linear interpolation",
            "seed": seed,
        }
    return {
        "candidate": candidate,
        "control": control,
        "metric": "verified_task_success",
        "task_count": len(tasks),
        "delta": round(point, 6),
        "confidence_interval": interval,
    }


def _review_summary(plan: Mapping[str, Any], run_dir: Path, suite_id: str) -> dict[str, Any] | None:
    review_root = run_dir / "reviews" / suite_id
    map_path = review_root / "blind-map.json"
    rows_path = review_root / "reviews.jsonl"
    if not map_path.is_file() or not rows_path.is_file():
        return None
    blind_map = json.loads(map_path.read_text(encoding="utf-8"))
    rows = _read_jsonl(rows_path, "blind reviews")
    suite = plan["suites"][suite_id]
    review = _mapping(suite.get("review", {}), f"evaluation.suites.{suite_id}.review")
    required = int(review.get("reviewers_per_pair", 1))
    decisions = plan.get("decisions", {}).get(suite_id, {})
    candidate = decisions.get("candidate")
    control = decisions.get("control")
    counts = {"candidate": 0, "control": 0, "tie": 0, "both_fail": 0}
    reviews_by_pair: dict[str, set[str]] = {}
    for row in rows:
        pair_id = row["pair_id"]
        mapping = blind_map["pairs"].get(pair_id)
        if mapping is None:
            raise EvaluationError(f"review refers to unknown blind pair: {pair_id}")
        reviews_by_pair.setdefault(pair_id, set()).add(str(row["reviewer_id"]))
        choice = row["choice"]
        if choice in {"left", "right"}:
            winner = mapping[choice]
            counts["candidate" if winner == candidate else "control"] += 1
        else:
            counts[choice] += 1
    # An exact reviewer count prevents one pair with extra votes from receiving
    # more weight while still satisfying the verified-completeness gate.
    complete = all(
        len(reviews_by_pair.get(pair_id, set())) == required
        for pair_id in blind_map["pairs"]
    )
    # Repetitions improve stability but do not create additional held-out tasks.
    # Keep both counts visible so minimum_tasks can never be met by reruns alone.
    task_count = len(
        {
            trial["task_id"]
            for trial in plan["trials"]
            if trial["suite_id"] == suite_id
        }
    )
    # A failed result is evidence against a preference claim, not an abstention.
    denominator = sum(counts.values())
    rate = (
        (counts["candidate"] + 0.5 * counts["tie"]) / denominator
        if denominator
        else 0.0
    )
    return {
        "candidate": candidate,
        "control": control,
        "required_reviewers_per_pair": required,
        "pair_count": len(blind_map["pairs"]),
        "task_count": task_count,
        "complete": complete,
        "counts": counts,
        "blind_preference_rate": round(rate, 6),
    }


def build_evaluation_reports(
    config: Mapping[str, Any], project_root: Path
) -> dict[str, Any]:
    """Report each suite separately and gate every verified improvement claim."""

    plan, run_dir = load_current_plan(config, project_root)
    scored = _load_scored(run_dir)
    suite_reports: dict[str, Any] = {}
    all_verified = True
    confidence = float(
        _mapping(plan.get("decisions", {}), "evaluation.decisions").get("confidence")
    )
    # Each suite has its own runner, so each receives an independent comparison
    # and decision. Local-agent reward is never pooled with Fable A/B results.
    for suite_id in sorted(plan["suites"]):
        payload, completeness = _suite_payload(plan, suite_id, scored)
        comparison = compare_benchmark(payload)
        decision_config = _mapping(
            plan.get("decisions", {}).get(suite_id, {}),
            f"evaluation.decisions.{suite_id}",
        )
        candidate = _text(decision_config.get("candidate"), f"decision {suite_id}.candidate")
        control = _text(decision_config.get("control"), f"decision {suite_id}.control")
        metric = decision_config.get("metric", "verified_task_success")
        minimum_tasks = int(decision_config.get("minimum_tasks", 2))
        if metric == "blind_preference_rate":
            review = _review_summary(plan, run_dir, suite_id)
            minimum_rate = float(decision_config.get("minimum_rate", 0.5))
            observed = bool(review and review["blind_preference_rate"] > minimum_rate)
            verified = bool(
                review
                and review["complete"]
                and completeness["rate"] == 1.0
                and completeness["fairness_passed"]
                and review["task_count"] >= minimum_tasks
                and observed
            )
            decision = {
                "metric": metric,
                "minimum_rate": minimum_rate,
                "minimum_tasks": minimum_tasks,
                "review": review,
                "observed_better": observed,
                "verified_better": verified,
            }
        else:
            delta = _paired_delta(
                payload,
                candidate,
                control,
                plan["plan_id"],
                confidence,
            )
            threshold = float(decision_config.get("minimum_delta", 0.0))
            observed = delta["delta"] > threshold
            interval = delta["confidence_interval"]
            verified = bool(
                completeness["rate"] == 1.0
                and completeness["fairness_passed"]
                and delta["task_count"] >= minimum_tasks
                and interval is not None
                and interval["low"] > threshold
            )
            decision = {
                **delta,
                "minimum_delta": threshold,
                "minimum_tasks": minimum_tasks,
                "observed_better": observed,
                "verified_better": verified,
            }
        all_verified = all_verified and bool(decision["verified_better"])
        suite_report = {
            "suite_id": suite_id,
            "kind": plan["suites"][suite_id]["kind"],
            "completeness": completeness,
            "comparison": comparison,
            "decision": decision,
        }
        if metric == "verified_task_success":
            # Confidence config drives only the paired task bootstrap. Fable's
            # point-preference rule must not be mislabeled as a confidence gate.
            suite_report["metadata"] = {
                "paired_bootstrap_confidence": confidence
            }
        suite_reports[suite_id] = suite_report
        _write_json(run_dir / "reports" / f"{suite_id}.json", suite_report)
        markdown = render_benchmark_markdown(comparison)
        markdown += (
            "\n## Decision\n\n"
            f"- Observed better: **{str(decision['observed_better']).lower()}**\n"
            f"- Verified better: **{str(decision['verified_better']).lower()}**\n"
            f"- Complete trials: {completeness['completed_trials']}/{completeness['expected_trials']}\n"
            f"- Fairness passed: **{str(completeness['fairness_passed']).lower()}**\n"
        )
        _atomic_text(run_dir / "reports" / f"{suite_id}.md", markdown)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "plan_id": plan["plan_id"],
        "metadata": {"model_benchmark_confidence": confidence},
        "v1_success_criteria_verified": all_verified and bool(suite_reports),
        "suites": suite_reports,
    }
    _write_json(run_dir / "summary.json", summary)
    return {**summary, "artifact": str(run_dir / "summary.json")}


def export_blind_review(
    config: Mapping[str, Any],
    project_root: Path,
    suite_id: str,
    output_dir: Path,
) -> dict[str, Any]:
    plan, run_dir = load_current_plan(config, project_root)
    suite = plan["suites"].get(suite_id)
    if not suite or suite.get("kind") != "fable_ab":
        raise EvaluationError("blind review export requires a fable_ab suite")
    scored = _load_scored(run_dir)
    by_key: dict[tuple[str, int, int], dict[str, Mapping[str, Any]]] = {}
    for trial in plan["trials"]:
        if trial["suite_id"] != suite_id:
            continue
        result = scored.get(trial["trial_id"])
        if result is None:
            raise EvaluationError("all Fable trials must be ingested before blind review export")
        artifact = result.get("metadata", {}).get("evidence", {}).get("review_artifact")
        if not artifact:
            raise EvaluationError(
                f"Fable trial {trial['trial_id']} has no output.review_artifact"
            )
        key = (trial["task_id"], trial["repetition"], trial["seed"])
        by_key.setdefault(key, {})[trial["arm_id"]] = result

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    pairs: list[dict[str, Any]] = []
    sealed: dict[str, Any] = {}
    for key in sorted(by_key):
        arms = by_key[key]
        if set(arms) != set(suite["arms"]):
            raise EvaluationError(f"blind pair is incomplete for {key}")
        pair_id = f"pair-{_digest({'plan': plan['plan_id'], 'suite': suite_id, 'key': key})[:24]}"
        # Arm identity is stored only in the sealed map. Reviewer exports expose
        # content-addressed left/right artifacts with deterministic randomization.
        ordered = list(suite["arms"])
        if int(_digest(pair_id)[:2], 16) % 2:
            ordered.reverse()
        left, right = ordered
        pairs.append(
            {
                "pair_id": pair_id,
                "task_id": key[0],
                "repetition": key[1],
                "seed": key[2],
                "left": arms[left]["metadata"]["evidence"]["review_artifact"],
                "right": arms[right]["metadata"]["evidence"]["review_artifact"],
                "choices": ["left", "right", "tie", "both_fail"],
            }
        )
        sealed[pair_id] = {"left": left, "right": right}
    _write_jsonl(destination / "blind-pairs.jsonl", pairs)
    review_root = run_dir / "reviews" / suite_id
    _write_json(
        review_root / "blind-map.json",
        {"plan_id": plan["plan_id"], "suite_id": suite_id, "pairs": sealed},
    )
    return {
        "plan_id": plan["plan_id"],
        "suite_id": suite_id,
        "pair_count": len(pairs),
        "artifact": str(destination / "blind-pairs.jsonl"),
        "sealed_map": str(review_root / "blind-map.json"),
    }


def import_blind_reviews(
    config: Mapping[str, Any],
    project_root: Path,
    suite_id: str,
    input_path: Path,
) -> dict[str, Any]:
    plan, run_dir = load_current_plan(config, project_root)
    review_root = run_dir / "reviews" / suite_id
    map_path = review_root / "blind-map.json"
    if not map_path.is_file():
        raise EvaluationError("export blind review pairs before importing reviews")
    blind_map = json.loads(map_path.read_text(encoding="utf-8"))
    rows = _read_jsonl(Path(input_path).expanduser().resolve(), "blind review import")
    seen: set[tuple[str, str]] = set()
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        row = _validate_review_row(row, f"reviews[{index}]")
        pair_id = _text(row.get("pair_id"), f"reviews[{index}].pair_id")
        reviewer_id = _text(row.get("reviewer_id"), f"reviews[{index}].reviewer_id")
        choice = row.get("choice")
        if pair_id not in blind_map["pairs"]:
            raise EvaluationError(f"review refers to unknown pair: {pair_id}")
        if choice not in {"left", "right", "tie", "both_fail"}:
            raise EvaluationError(f"invalid review choice for {pair_id}: {choice!r}")
        key = (pair_id, reviewer_id)
        if key in seen:
            raise EvaluationError(f"duplicate reviewer/pair row: {reviewer_id}/{pair_id}")
        seen.add(key)
        normalized.append(
            {"pair_id": pair_id, "reviewer_id": reviewer_id, "choice": choice}
        )
    destination = review_root / "reviews.jsonl"
    if destination.exists():
        raise EvaluationError("blind reviews are immutable; remove the plan and start again to replace them")
    _write_jsonl(destination, sorted(normalized, key=lambda row: (row["pair_id"], row["reviewer_id"])))
    return {
        "plan_id": plan["plan_id"],
        "suite_id": suite_id,
        "review_count": len(normalized),
        "artifact": str(destination),
    }


def run_command_suite(
    config: Mapping[str, Any],
    project_root: Path,
    suite_id: str,
    *,
    resume: bool = False,
    scorer: Callable[[Mapping[str, Any], str], Any] | None = None,
) -> dict[str, Any]:
    """Run an explicitly declared argv adapter without invoking a shell."""

    plan, run_dir = load_current_plan(config, project_root)
    suite = plan["suites"].get(suite_id)
    if not suite:
        raise EvaluationError(f"unknown evaluation suite: {suite_id}")
    runner = suite["runner"]
    if runner["type"] != "command":
        raise EvaluationError(
            f"suite {suite_id!r} is external; use evaluate export and evaluate ingest"
        )
    completed = 0
    skipped = 0
    for trial in plan["trials"]:
        if trial["suite_id"] != suite_id:
            continue
        scored_path = run_dir / "scored-trials" / f"{trial['trial_id']}.json"
        if scored_path.exists() and resume:
            skipped += 1
            continue
        if scored_path.exists():
            raise EvaluationError(
                f"trial already exists: {trial['trial_id']}; use --resume to skip it"
            )
        incoming = run_dir / "incoming" / trial["trial_id"]
        incoming.mkdir(parents=True, exist_ok=True)
        request_path = incoming / "request.json"
        result_path = incoming / "result.json"
        _write_json(request_path, _request_for(plan, trial))
        substitutions = {
            "request": str(request_path),
            "result": str(result_path),
            "trial_id": trial["trial_id"],
            "arm_id": trial["arm_id"],
        }
        try:
            argv = [item.format(**substitutions) for item in runner["argv"]]
        except KeyError as error:
            raise EvaluationError(f"unknown command runner placeholder: {error}") from error
        timeout = int(plan["environment"].get("episode_timeout_seconds") or 900)
        # argv execution is explicit and shell-free. This prevents values in a
        # task or artifact path from being interpreted as shell syntax.
        completed_process = subprocess.run(
            argv,
            cwd=str(Path(project_root).resolve()),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            shell=False,
        )
        _atomic_text(incoming / "stdout.txt", completed_process.stdout)
        _atomic_text(incoming / "stderr.txt", completed_process.stderr)
        if not result_path.is_file():
            raise EvaluationError(
                f"command runner did not write {result_path} (exit {completed_process.returncode})"
            )
        ingest_evaluation_results(
            config,
            project_root,
            suite_id,
            result_path,
            scorer=scorer,
        )
        completed += 1
    return {
        "plan_id": plan["plan_id"],
        "suite_id": suite_id,
        "completed": completed,
        "skipped": skipped,
    }


__all__ = [
    "EvaluationError",
    "RESULT_COMPONENTS",
    "build_evaluation_plan",
    "build_evaluation_reports",
    "evaluation_run_dir",
    "export_evaluation_suite",
    "export_blind_review",
    "import_blind_reviews",
    "ingest_evaluation_results",
    "load_current_plan",
    "run_command_suite",
    "write_evaluation_plan",
]
