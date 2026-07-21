"""Shared evaluation lifecycle for the human GUI and agent CLI.

The evaluation core owns plan integrity, runners, trusted re-scoring, blind
review, and reports.  This module adds only project ownership and a durable,
sanitized view of observable progress.  It never estimates opaque model work
and never represents the external Fable producer as a local running job.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
import os
from pathlib import Path
import re
from threading import Lock, Thread, current_thread
from typing import Any, Mapping
from uuid import uuid4

from .compiler import compile_data
from .config import ConfigError, ProjectConfig, load_config
from .dataset_service import require_frozen_dataset
from .device_gate import (
    DeviceLease,
    acquire_device_lease,
    clear_cuda_memory,
    device_run_gate,
)
from .evaluation import (
    EvaluationError,
    EvaluationProgressCallback,
    RESULT_COMPONENTS,
    build_evaluation_reports,
    load_current_plan,
    load_validated_scored_results,
    run_command_suite,
    write_evaluation_plan,
)
from .language_evaluation import require_language_matched_evaluation
from .planner import config_fingerprint
from .project_gate import (
    ProjectLease,
    acquire_project_lease,
    project_is_busy,
    project_run_gate,
)
from .sources import scan_sources


_JOB_SCHEMA_VERSION = 1
_JOB_STATUSES = frozenset(
    {"idle", "queued", "running", "completed", "failed", "interrupted"}
)
_LIVE_JOB_STATUSES = frozenset({"queued", "running"})
_PHASES = frozenset(
    {
        "idle",
        "queued",
        "resuming",
        "generating",
        "verifying",
        "trial_completed",
        "completed",
        "failed",
        "interrupted",
        "running",
    }
)
_PUBLIC_RESULT_LIMIT = 500
_PUBLIC_TRIAL_LIMIT = 1000
_EVENT_SCHEMA_VERSION = 1
_EVENT_LIMIT = 1000
_EVENT_PHASES = frozenset(
    {"queued", "generating", "verifying", "trial_completed", "completed", "failed"}
)
_SAFE_ID = re.compile(r"^[A-Za-z0-9_.:-]{1,200}$")
_PLAN_ID = re.compile(r"^sha256:[0-9a-f]{64}$")
_SECRET_PATTERNS = (
    re.compile(r"\bhf_[A-Za-z0-9]{8,}\b"),
    re.compile(r"(?i)\b(bearer\s+)[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"(?i)\b(api[-_ ]?key|token|password|secret)\s*[:=]\s*[^\s,;]+"),
)


class EvaluationServiceError(ConfigError):
    """Raised when a local evaluation action cannot be represented honestly."""


def _redact(value: object, *, limit: int = 1000) -> str:
    """Bound public text and remove common credential forms defensively."""

    text = str(value).replace("\r", " ").replace("\n", " ")[:limit]
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(
            lambda match: f"{match.group(1)}[redacted]" if match.lastindex else "[redacted]",
            text,
        )
    return text


def _safe_integer(value: object, *, default: int = 0) -> int:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return default


def _safe_trial(value: object) -> dict[str, Any] | None:
    """Keep only immutable trial identity fields useful to the progress UI."""

    if not isinstance(value, Mapping):
        return None
    text_fields = ("trial_id", "task_id", "arm_id")
    trial: dict[str, Any] = {}
    for field in text_fields:
        item = value.get(field)
        if not isinstance(item, str) or not _SAFE_ID.fullmatch(item):
            return None
        trial[field] = item
    for field in ("repetition", "seed"):
        item = value.get(field)
        if not isinstance(item, int) or isinstance(item, bool) or item < 0:
            return None
        trial[field] = item
    return trial


def _safe_components(value: object) -> dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    components: dict[str, float] = {}
    for name in RESULT_COMPONENTS:
        item = value.get(name)
        if isinstance(item, (int, float)) and not isinstance(item, bool):
            number = float(item)
            if math.isfinite(number) and 0 <= number <= 1:
                components[name] = number
    return components


def _safe_result(value: object, trial: Mapping[str, Any] | None = None) -> dict[str, Any] | None:
    """Whitelist trusted score fields; evidence paths and transcripts stay local."""

    if not isinstance(value, Mapping):
        return None
    identity_value = dict(value)
    # Scored artifacts use candidate_id; the public lifecycle uses arm_id to
    # match the frozen trial vocabulary without exposing the entire artifact.
    if "arm_id" not in identity_value:
        identity_value["arm_id"] = identity_value.get("candidate_id")
    identity = _safe_trial(identity_value)
    if identity is None:
        return None
    if trial is not None and any(identity[key] != trial.get(key) for key in identity):
        return None
    reward = value.get("reward")
    if not isinstance(reward, (int, float)) or isinstance(reward, bool):
        return None
    reward_number = float(reward)
    if not math.isfinite(reward_number) or not 0 <= reward_number <= 1:
        return None
    status = value.get("status")
    if not isinstance(status, str) or not _SAFE_ID.fullmatch(status):
        return None
    gate_reason = value.get("gate_reason")
    return {
        **identity,
        "status": status,
        "hard_gate_passed": value.get("hard_gate_passed") is True,
        "gate_reason": _redact(gate_reason, limit=300) if gate_reason else None,
        "reward": reward_number,
        "components": _safe_components(value.get("components")),
    }


def _suite_results(
    plan: Mapping[str, Any], run_dir: Path, suite_id: str
) -> list[dict[str, Any]]:
    """Project seal-validated scored rows in deterministic trial order."""

    results: list[dict[str, Any]] = []
    try:
        validated = load_validated_scored_results(plan, run_dir)
    except EvaluationError as error:
        raise EvaluationServiceError(str(error)) from error
    for raw_trial in plan.get("trials", []):
        if not isinstance(raw_trial, Mapping) or raw_trial.get("suite_id") != suite_id:
            continue
        trial = _safe_trial(raw_trial)
        if trial is None:
            raise EvaluationServiceError("The frozen evaluation plan contains an invalid trial.")
        value = validated.get(trial["trial_id"])
        if value is None:
            continue
        result = _safe_result(value, trial)
        if (
            result is None
            or value.get("plan_id") != plan.get("plan_id")
            or value.get("suite_id") != suite_id
            or set(result["components"]) != set(RESULT_COMPONENTS)
        ):
            raise EvaluationServiceError(
                "A scored evaluation result does not match its frozen trial: "
                f"{trial['trial_id']}"
            )
        results.append(result)
    return results


def _planned_trials(
    trials: list[Mapping[str, Any]],
    scored: list[Mapping[str, Any]],
    job: Mapping[str, Any],
    suite_id: str,
) -> tuple[list[dict[str, Any]], bool]:
    """Expose plan identity and only lifecycle state observed by AutoTrainer."""

    completed_ids = {str(result.get("trial_id")) for result in scored}
    current = (
        _safe_trial(job.get("current_trial"))
        if job.get("suite") == suite_id and job.get("status") in _LIVE_JOB_STATUSES
        else None
    )
    active_status = (
        str(job.get("phase"))
        if job.get("phase") in {"generating", "verifying"}
        else None
    )
    public: list[dict[str, Any]] = []
    for raw_trial in trials[:_PUBLIC_TRIAL_LIMIT]:
        trial = _safe_trial(raw_trial)
        if trial is None:
            raise EvaluationServiceError("The frozen evaluation plan contains an invalid trial.")
        if trial["trial_id"] in completed_ids:
            status = "completed"
        elif current is not None and trial["trial_id"] == current["trial_id"] and active_status:
            status = active_status
        else:
            status = "planned"
        public.append({**trial, "status": status})
    return public, len(trials) > len(public)


def _safe_event(value: object) -> dict[str, Any] | None:
    """Validate one persisted event and derive its rubric from a safe result."""

    if not isinstance(value, Mapping):
        return None
    sequence = value.get("sequence")
    job_id = value.get("job_id")
    plan_id = value.get("plan_id")
    suite = value.get("suite")
    phase = value.get("phase")
    if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 1:
        return None
    if not isinstance(job_id, str) or re.fullmatch(r"[0-9a-f]{32}", job_id) is None:
        return None
    if not isinstance(plan_id, str) or _PLAN_ID.fullmatch(plan_id) is None:
        return None
    if not isinstance(suite, str) or _SAFE_ID.fullmatch(suite) is None:
        return None
    if phase not in _EVENT_PHASES:
        return None
    completed = _safe_integer(value.get("completed"))
    total = _safe_integer(value.get("total"))
    if completed > total:
        return None
    trial = _safe_trial(value.get("trial"))
    safe_result = (
        _safe_result(value.get("result"), trial)
        if phase == "trial_completed" and trial is not None
        else None
    )
    result = (
        {
            **{key: safe_result[key] for key in ("trial_id", "task_id", "arm_id", "repetition", "seed")},
            "status": safe_result["status"],
            "hard_gate_passed": safe_result["hard_gate_passed"],
            "reward": safe_result["reward"],
            "components": dict(safe_result["components"]),
        }
        if safe_result is not None
        else None
    )
    # A rubric is never accepted independently from disk. It is reconstructed
    # from the already-whitelisted trusted result so paths, prompts, and model
    # transcripts cannot enter the browser event stream.
    rubric = (
        {
            "hard_gate_passed": result["hard_gate_passed"],
            "reward": result["reward"],
            "components": dict(result["components"]),
        }
        if result is not None
        else None
    )
    return {
        "sequence": sequence,
        "job_id": job_id,
        "plan_id": plan_id,
        "suite": suite,
        "phase": str(phase),
        "message": _redact(value.get("message", "Evaluation status changed.")),
        "completed": completed,
        "total": total,
        "trial": trial,
        "result": result,
        "rubric": rubric,
    }


def _idle_job(*, plan_id: str | None = None) -> dict[str, Any]:
    return {
        "id": None,
        "status": "idle",
        "plan_id": plan_id,
        "suite": None,
        "phase": "idle",
        "message": "No local evaluation job has started.",
        "completed": 0,
        "total": 0,
        "current_trial": None,
        "results": [],
        "results_truncated": False,
    }


def _normalize_saved_job(value: object) -> dict[str, Any] | None:
    """Validate every persisted field before returning it through localhost."""

    if not isinstance(value, Mapping):
        return None
    status = value.get("status")
    if status not in _JOB_STATUSES:
        return None
    plan_id_value = value.get("plan_id")
    plan_id = str(plan_id_value) if plan_id_value is not None else None
    if plan_id is not None and not _PLAN_ID.fullmatch(plan_id):
        return None
    if status == "idle":
        return _idle_job(plan_id=plan_id)

    job_id = value.get("id")
    suite = value.get("suite")
    phase = value.get("phase")
    if not isinstance(job_id, str) or re.fullmatch(r"[0-9a-f]{32}", job_id) is None:
        return None
    if not isinstance(suite, str) or not _SAFE_ID.fullmatch(suite):
        return None
    if phase not in _PHASES:
        return None
    completed = _safe_integer(value.get("completed"))
    total = _safe_integer(value.get("total"))
    if completed > total:
        return None
    raw_results = value.get("results")
    results = []
    if isinstance(raw_results, list):
        for raw_result in raw_results[:_PUBLIC_RESULT_LIMIT]:
            result = _safe_result(raw_result)
            if result is not None:
                results.append(result)
    return {
        "id": job_id,
        "status": status,
        "plan_id": plan_id,
        "suite": suite,
        "phase": phase,
        "message": _redact(value.get("message", "Evaluation status is available.")),
        "completed": completed,
        "total": total,
        "current_trial": _safe_trial(value.get("current_trial")),
        "results": results,
        "results_truncated": bool(value.get("results_truncated")),
    }


def _write_job_record(destination: Path, job: Mapping[str, Any]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(
                {"schema_version": _JOB_SCHEMA_VERSION, "job": dict(job)},
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _read_job_record(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, Mapping) or payload.get("schema_version") != _JOB_SCHEMA_VERSION:
        return None
    return _normalize_saved_job(payload.get("job"))


def _write_event_record(
    destination: Path,
    events: list[Mapping[str, Any]],
    next_sequence: int,
) -> None:
    """Atomically persist the bounded cursor log beside the durable job."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(
                {
                    "schema_version": _EVENT_SCHEMA_VERSION,
                    "next_sequence": next_sequence,
                    "events": list(events),
                },
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _read_event_record(path: Path) -> tuple[list[dict[str, Any]], int]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeDecodeError, json.JSONDecodeError):
        return [], 1
    if not isinstance(payload, Mapping) or payload.get("schema_version") != _EVENT_SCHEMA_VERSION:
        return [], 1
    raw_events = payload.get("events")
    if not isinstance(raw_events, list):
        return [], 1
    events: list[dict[str, Any]] = []
    previous = 0
    for raw_event in raw_events[-_EVENT_LIMIT:]:
        event = _safe_event(raw_event)
        if event is None or event["sequence"] <= previous:
            return [], 1
        previous = event["sequence"]
        events.append(event)
    next_sequence = payload.get("next_sequence")
    if (
        not isinstance(next_sequence, int)
        or isinstance(next_sequence, bool)
        or next_sequence <= previous
    ):
        return [], 1
    return events, next_sequence


def _load_project(config_path: str | Path) -> ProjectConfig:
    return load_config(config_path, check_paths=True)


def _refresh_frozen_compiler_provenance(config: ProjectConfig) -> None:
    """Restore full train/evaluation provenance after train-only Prepare.

    Prepare deliberately compiles only training sources. Evaluation rebuilds
    the full compiler ledger, then proves that it is byte-for-byte equivalent
    to the operator-inspected dataset freeze before freezing any trials.
    """

    freeze_path = config.artifact_dir / "dataset" / "freeze.json"
    if not freeze_path.is_file():
        # Compatibility for direct library/CLI projects that predate the local
        # first-class dataset workflow; their existing compiler report remains
        # subject to the evaluator's strict provenance checks.
        return
    receipt = require_frozen_dataset(config.path)
    scan = scan_sources(config.data, config.root, write=True)
    scan_errors = scan.get("errors")
    if isinstance(scan_errors, list) and scan_errors:
        raise EvaluationServiceError(
            "held-out source inspection failed: " + str(scan_errors[0])
        )
    compiled = compile_data(config.data, config.root, scan)
    compile_errors = compiled.get("errors")
    if isinstance(compile_errors, list) and compile_errors:
        raise EvaluationServiceError(
            "held-out dataset compilation failed: " + str(compile_errors[0])
        )
    comparisons = (
        ("compiler fingerprint", receipt.get("compiler_fingerprint"), compiled.get("fingerprint")),
        ("record counts", receipt.get("counts"), compiled.get("counts")),
        ("artifact hashes", receipt.get("artifact_sha256"), compiled.get("artifact_sha256")),
    )
    mismatched = [label for label, expected, observed in comparisons if expected != observed]
    if mismatched:
        raise EvaluationServiceError(
            "the rebuilt held-out provenance does not match the frozen dataset "
            f"({', '.join(mismatched)}); review and lock Data again"
        )


def plan_project_evaluation(config_path: str | Path) -> dict[str, Any]:
    """Freeze a plan under the same project lease used by local training."""

    with project_run_gate(config_path), device_run_gate():
        config = _load_project(config_path)
        _refresh_frozen_compiler_provenance(config)
        # Missing ``evaluation.language`` has always meant ``auto`` in config
        # validation.  Enforce the same language gate for legacy projects so
        # omitting the key cannot bypass held-out task matching.
        require_language_matched_evaluation(config.path)
        return write_evaluation_plan(config.data, config.root)


def run_project_evaluation(
    config_path: str | Path,
    suite_id: str,
    *,
    resume: bool = True,
    on_progress: EvaluationProgressCallback | None = None,
) -> dict[str, Any]:
    """Run one suite with the same project and GPU ownership in every client.

    The GUI reserves transferable leases before it publishes a queued job. The
    CLI enters this function directly. Both paths therefore pass through these
    re-entrant gates, so neither can load a second 9B model onto GPU 0.
    """

    with project_run_gate(config_path), device_run_gate():
        config = _load_project(config_path)
        return run_command_suite(
            config.data,
            config.root,
            suite_id,
            resume=resume,
            on_progress=on_progress,
        )


def _readiness(config: ProjectConfig) -> dict[str, Any]:
    """Expose the last prepared evaluation blockers without re-running Prepare."""

    path = config.artifact_dir / "plan.json"
    if not path.is_file():
        return {
            "status": "not_prepared",
            "ready_task_count": 0,
            "blockers": [],
            "warnings": [],
        }
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, AttributeError):
        return {
            "status": "invalid",
            "ready_task_count": 0,
            "blockers": ["The prepared evaluation snapshot is unreadable."],
            "warnings": [],
        }
    if not isinstance(value, Mapping):
        return {
            "status": "invalid",
            "ready_task_count": 0,
            "blockers": ["The prepared evaluation snapshot is invalid."],
            "warnings": [],
        }
    if value.get("config_fingerprint") != config_fingerprint(config.data):
        return {
            "status": "stale",
            "ready_task_count": 0,
            "blockers": [
                "Project inputs changed since the last Prepare; check readiness again."
            ],
            "warnings": [],
        }
    stages = value.get("stages")
    stage = stages.get("evaluation", {}) if isinstance(stages, Mapping) else None
    if not isinstance(stage, Mapping):
        return {
            "status": "invalid",
            "ready_task_count": 0,
            "blockers": ["The prepared evaluation snapshot is invalid."],
            "warnings": [],
        }
    blockers = stage.get("blockers") if isinstance(stage.get("blockers"), list) else []
    warnings = stage.get("warnings") if isinstance(stage.get("warnings"), list) else []
    return {
        "status": _redact(stage.get("status", "unknown"), limit=80),
        "ready_task_count": _safe_integer(stage.get("ready_task_count")),
        "blockers": [_redact(item) for item in blockers[:100]],
        "warnings": [_redact(item) for item in warnings[:100]],
    }


def _review_counts(run_dir: Path, suite_id: str, expected_pairs: int, reviewers: int) -> dict[str, Any]:
    review_root = run_dir / "reviews" / suite_id
    map_exists = (review_root / "blind-map.json").is_file()
    rows_path = review_root / "reviews.jsonl"
    count = 0
    if rows_path.is_file():
        try:
            count = sum(
                1
                for line in rows_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        except (OSError, UnicodeDecodeError):
            count = 0
    required = expected_pairs * reviewers
    return {
        "pairs_exported": map_exists,
        "review_count": count,
        "required_reviews": required,
        "complete": map_exists and count == required,
    }


def _safe_signed_rate(value: object) -> float | None:
    """Accept a finite comparison delta while rejecting arbitrary report data."""

    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    number = float(value)
    return number if math.isfinite(number) and -1 <= number <= 1 else None


def _safe_rate(value: object) -> float | None:
    number = _safe_signed_rate(value)
    return number if number is not None and number >= 0 else None


def _safe_report_summary(
    value: object,
    suite_id: str,
    *,
    plan_id: str,
    completed: int,
    total: int,
) -> dict[str, Any] | None:
    """Project the immutable comparison report into a small localhost contract.

    Reports contain model metadata and verifier details that the evaluation UI
    does not need. This whitelist exposes only descriptive arm summaries and
    the already-computed paired decision; the browser never reimplements the
    statistical test or receives evidence paths.
    """

    if not isinstance(value, Mapping) or value.get("suite_id") != suite_id:
        return None
    completeness = value.get("completeness")
    if not isinstance(completeness, Mapping):
        return None
    expected = _safe_integer(completeness.get("expected_trials"))
    observed = _safe_integer(completeness.get("completed_trials"))
    rate = _safe_rate(completeness.get("rate"))
    if expected != total or observed != completed or rate is None:
        return None

    public_candidates: list[dict[str, Any]] = []
    comparison = value.get("comparison")
    if isinstance(comparison, Mapping):
        raw_candidates = comparison.get("candidates")
        if isinstance(raw_candidates, list):
            for raw_candidate in raw_candidates:
                if not isinstance(raw_candidate, Mapping):
                    continue
                candidate_id = raw_candidate.get("candidate_id")
                label = raw_candidate.get("label")
                rank = raw_candidate.get("rank")
                hard_gate = raw_candidate.get("hard_gate")
                reward = raw_candidate.get("reward")
                if (
                    not isinstance(candidate_id, str)
                    or not _SAFE_ID.fullmatch(candidate_id)
                    or not isinstance(label, str)
                    or not isinstance(rank, int)
                    or isinstance(rank, bool)
                    or rank < 1
                    or not isinstance(hard_gate, Mapping)
                    or not isinstance(reward, Mapping)
                ):
                    continue
                pass_rate = _safe_rate(hard_gate.get("pass_rate"))
                reward_mean = _safe_rate(reward.get("mean"))
                if pass_rate is None or reward_mean is None:
                    continue
                public_candidates.append(
                    {
                        "candidate_id": candidate_id,
                        "label": _redact(label, limit=120),
                        "rank": rank,
                        "hard_gate_pass_rate": pass_rate,
                        "reward_mean": reward_mean,
                    }
                )

    decision_value = value.get("decision")
    if not isinstance(decision_value, Mapping):
        return None
    decision: dict[str, Any] = {
        "metric": _redact(decision_value.get("metric", "unknown"), limit=80),
        "observed_better": decision_value.get("observed_better") is True,
        "verified_better": decision_value.get("verified_better") is True,
    }
    for name in ("candidate", "control"):
        item = decision_value.get(name)
        if isinstance(item, str) and _SAFE_ID.fullmatch(item):
            decision[name] = item
    for name in ("task_count", "minimum_tasks"):
        item = decision_value.get(name)
        if isinstance(item, int) and not isinstance(item, bool) and item >= 0:
            decision[name] = item
    for name in ("delta", "minimum_delta"):
        item = _safe_signed_rate(decision_value.get(name))
        if item is not None:
            decision[name] = item

    interval_value = decision_value.get("confidence_interval")
    interval = None
    if isinstance(interval_value, Mapping):
        low = _safe_signed_rate(interval_value.get("low"))
        high = _safe_signed_rate(interval_value.get("high"))
        confidence = _safe_rate(interval_value.get("confidence"))
        method = interval_value.get("method")
        if (
            low is not None
            and high is not None
            and low <= high
            and confidence is not None
            and isinstance(method, str)
        ):
            interval = {
                "low": low,
                "high": high,
                "confidence": confidence,
                "method": _redact(method, limit=120),
            }
    decision["confidence_interval"] = interval
    return {
        "plan_id": plan_id,
        "completeness": {
            "expected_trials": expected,
            "completed_trials": observed,
            "rate": rate,
            "fairness_passed": completeness.get("fairness_passed") is True,
        },
        "comparison": {
            "winner_candidate_id": (
                comparison.get("winner_candidate_id")
                if isinstance(comparison, Mapping)
                and isinstance(comparison.get("winner_candidate_id"), str)
                and _SAFE_ID.fullmatch(str(comparison.get("winner_candidate_id")))
                else None
            ),
            "candidates": public_candidates,
        },
        "decision": decision,
    }


def _suite_snapshot(
    plan: Mapping[str, Any],
    run_dir: Path,
    suite_id: str,
    job: Mapping[str, Any],
    reports: Mapping[str, Any],
) -> dict[str, Any]:
    suite = plan.get("suites", {}).get(suite_id, {})
    if not isinstance(suite, Mapping):
        raise EvaluationServiceError(f"The frozen evaluation suite is invalid: {suite_id}")
    runner = suite.get("runner", {})
    runner_type = runner.get("type") if isinstance(runner, Mapping) else None
    kind = str(suite.get("kind", ""))
    trials = [
        trial
        for trial in plan.get("trials", [])
        if isinstance(trial, Mapping) and trial.get("suite_id") == suite_id
    ]
    total = len(trials)
    scored = _suite_results(plan, run_dir, suite_id)
    completed = len(scored)
    report = _safe_report_summary(
        reports.get(suite_id),
        suite_id,
        plan_id=str(plan.get("plan_id", "")),
        completed=completed,
        total=total,
    )
    report_exists = report is not None

    review = None
    if kind == "fable_ab":
        review_config = suite.get("review", {})
        reviewers = (
            _safe_integer(review_config.get("reviewers_per_pair"), default=1)
            if isinstance(review_config, Mapping)
            else 1
        )
        review = _review_counts(run_dir, suite_id, total // 2, max(1, reviewers))
        # Building the local model report may also write an incomplete Fable
        # report. Do not let that administrative artifact reveal arm identity
        # before the configured blind review has actually completed.
        if not review["complete"]:
            report = None
            report_exists = False

    if total == 0:
        phase = "blocked"
        blockers = suite.get("blockers", [])
        if suite.get("runnable") is False and isinstance(blockers, list) and blockers:
            message = "Deferred until configured: " + "; ".join(
                str(item) for item in blockers
            )
        else:
            message = "The frozen suite contains no trials. Freeze a valid evaluation plan."
    elif job.get("status") in _LIVE_JOB_STATUSES and job.get("suite") == suite_id:
        phase = str(job.get("phase"))
        message = str(job.get("message"))
    elif runner_type == "external":
        if completed < total:
            phase = "awaiting_external_results"
            message = (
                f"{completed} of {total} trusted results are ingested. "
                "Fable runs outside AutoTrainer; no local run is being simulated."
            )
        elif review and not review["pairs_exported"]:
            phase = "ready_for_blind_review"
            message = "All Fable results are verified locally. Export the blind review pairs."
        elif review and not review["complete"]:
            phase = "awaiting_blind_reviews"
            message = (
                f"{review['review_count']} of {review['required_reviews']} blind reviews are imported."
            )
        elif report_exists:
            phase = "reported"
            message = "The external comparison report is ready."
        else:
            phase = "ready_to_report"
            message = "All external results and blind reviews are complete."
    elif completed == total and total > 0:
        phase = "reported" if report_exists else "ready_to_report"
        message = (
            "The local benchmark report is ready."
            if report_exists
            else "All local benchmark trials are complete."
        )
    elif completed > 0:
        phase = "paused"
        message = f"{completed} of {total} local benchmark trials are complete; resume is available."
    else:
        phase = "ready"
        message = "The frozen local benchmark is ready to start."

    # Fable results stay arm-blind in the localhost response.  The reviewed
    # report, not per-arm trial telemetry, is the point where identity returns.
    public_results = [] if kind == "fable_ab" else scored[-_PUBLIC_RESULT_LIMIT:]
    public_trials: list[dict[str, Any]] = []
    trials_truncated = False
    if runner_type in {"builtin", "command"}:
        public_trials, trials_truncated = _planned_trials(trials, scored, job, suite_id)
    public_arms: list[dict[str, str]] = []
    if kind != "fable_ab":
        plan_arms = plan.get("arms", {})
        for arm_id in suite.get("arms", []):
            arm = plan_arms.get(arm_id) if isinstance(plan_arms, Mapping) else None
            if (
                not isinstance(arm_id, str)
                or not _SAFE_ID.fullmatch(arm_id)
                or not isinstance(arm, Mapping)
            ):
                continue
            label = arm.get("label")
            role = arm.get("role")
            public_arms.append(
                {
                    "id": arm_id,
                    "label": _redact(label if isinstance(label, str) else arm_id, limit=120),
                    "role": _redact(role if isinstance(role, str) else "unknown", limit=80),
                }
            )
    return {
        "id": suite_id,
        "kind": kind,
        "runner_type": runner_type,
        "runnable": suite.get("runnable") is not False,
        "blockers": list(suite.get("blockers", []))
        if isinstance(suite.get("blockers"), list)
        else [],
        "execution_policy": suite.get("execution_policy"),
        "phase": phase,
        "message": message,
        "completed": completed,
        "total": total,
        # Local plans are safe to show before execution because this
        # projection contains identity and observed lifecycle only. External
        # blind-review arms stay withheld.
        "trials": public_trials,
        "trials_truncated": trials_truncated,
        "arms": public_arms,
        "results": public_results,
        "results_truncated": len(scored) > len(public_results) and kind != "fable_ab",
        "results_withheld_for_blind_review": kind == "fable_ab" and bool(scored),
        "review": review,
        "report": report,
    }


def _reports_published(run_dir: Path) -> bool:
    """A report remains an explicit action, even though its bytes are untrusted."""

    reports = run_dir / "reports"
    return reports.is_dir() and any(reports.glob("*.json"))


def _report_inputs_key(plan: Mapping[str, Any], run_dir: Path) -> tuple[Any, ...]:
    """Cheap cache identity; content is still fully validated on every cache miss."""

    paths = list((run_dir / "scored-trials").glob("*.json"))
    review_root = run_dir / "reviews"
    if review_root.is_dir():
        paths.extend(path for path in review_root.rglob("*") if path.is_file())
    entries: list[tuple[str, int, int]] = []
    for path in sorted(paths):
        try:
            stat = path.stat()
            entries.append(
                (path.relative_to(run_dir).as_posix(), stat.st_size, stat.st_mtime_ns)
            )
        except (OSError, ValueError) as error:
            raise EvaluationServiceError(
                "Evaluation report inputs changed while they were being inspected."
            ) from error
    return (str(plan.get("plan_id", "")), *entries)


class EvaluationJobManager:
    """Own one durable command-suite job and expose all suite phases honestly."""

    def __init__(self, config_path: str | Path) -> None:
        self._lock = Lock()
        self._config_path = Path(config_path).expanduser().resolve()
        config = _load_project(self._config_path)
        self._record_path = config.artifact_dir / "evaluation" / "current-job.json"
        self._events_path = config.artifact_dir / "evaluation" / "events.json"
        self._worker: Thread | None = None
        self._reports_lock = Lock()
        self._reports_cache_key: tuple[Any, ...] | None = None
        self._reports_cache: dict[str, Any] = {}
        self._events, self._next_event_sequence = _read_event_record(self._events_path)
        self._job = _read_job_record(self._record_path) or _idle_job()
        if self._job["status"] in _LIVE_JOB_STATUSES and not project_is_busy(
            self._config_path
        ):
            self._job.update(
                status="interrupted",
                phase="interrupted",
                message="Evaluation was interrupted when the local backend stopped.",
                current_trial=None,
            )
            _write_job_record(self._record_path, self._job)
            self._append_event_locked("failed")

    @property
    def record_path(self) -> Path:
        return self._record_path

    @property
    def events_path(self) -> Path:
        return self._events_path

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return deepcopy(self._job)

    def events(self, after_sequence: int = 0) -> dict[str, Any]:
        """Return a cursor page from the bounded durable event history."""

        if (
            not isinstance(after_sequence, int)
            or isinstance(after_sequence, bool)
            or after_sequence < 0
        ):
            raise EvaluationServiceError("after_sequence must be a non-negative integer")
        with self._lock:
            oldest = self._events[0]["sequence"] if self._events else 0
            latest = self._next_event_sequence - 1
            return {
                "events": deepcopy(
                    [event for event in self._events if event["sequence"] > after_sequence]
                ),
                "oldest_sequence": oldest,
                "latest_sequence": latest,
                # A stale browser cursor can explicitly replace its local list
                # instead of silently missing events trimmed by the bound.
                "cursor_reset": bool(self._events and after_sequence < oldest - 1),
            }

    def workspace(self) -> dict[str, Any]:
        """Combine the durable job with immutable artifacts; never mutate on GET."""

        config = _load_project(self._config_path)
        job = self.snapshot()
        pointer = config.artifact_dir / "evaluation" / "current-plan.json"
        if not pointer.is_file():
            return {
                "readiness": _readiness(config),
                "plan": None,
                "job": job,
                "suites": [],
            }
        plan, run_dir = load_current_plan(config.data, config.root)
        plan_id = str(plan["plan_id"])
        reports: dict[str, Any] = {}
        if _reports_published(run_dir):
            key = _report_inputs_key(plan, run_dir)
            with self._reports_lock:
                if key != self._reports_cache_key:
                    # Never trust the derived JSON file. Recompute the UI
                    # projection from sealed scored rows and frozen plan bytes;
                    # the read-only mode leaves the workspace untouched.
                    computed = build_evaluation_reports(
                        config.data,
                        config.root,
                        write_artifacts=False,
                    )
                    raw_reports = computed.get("suites", {})
                    if not isinstance(raw_reports, Mapping):
                        raise EvaluationServiceError(
                            "Evaluation report computation returned invalid suites."
                        )
                    self._reports_cache = deepcopy(dict(raw_reports))
                    self._reports_cache_key = key
                reports = deepcopy(self._reports_cache)
        suites = [
            _suite_snapshot(plan, run_dir, suite_id, job, reports)
            for suite_id in sorted(plan.get("suites", {}))
        ]
        return {
            "readiness": _readiness(config),
            "plan": {
                "plan_id": plan_id,
                "task_count": len(plan.get("tasks", [])),
                "repetitions": _safe_integer(plan.get("repetitions")),
                "seeds": [
                    seed
                    for seed in plan.get("seeds", [])
                    if isinstance(seed, int) and not isinstance(seed, bool) and seed >= 0
                ],
            },
            "job": job,
            "suites": suites,
        }

    def _append_event_locked(
        self,
        phase: str,
        *,
        trial: Mapping[str, Any] | None = None,
        result: Mapping[str, Any] | None = None,
    ) -> None:
        if phase not in _EVENT_PHASES:
            return
        raw_event = {
            "sequence": self._next_event_sequence,
            "job_id": self._job.get("id"),
            "plan_id": self._job.get("plan_id"),
            "suite": self._job.get("suite"),
            "phase": phase,
            "message": self._job.get("message"),
            "completed": self._job.get("completed"),
            "total": self._job.get("total"),
            "trial": dict(trial) if isinstance(trial, Mapping) else None,
            "result": dict(result) if isinstance(result, Mapping) else None,
        }
        event = _safe_event(raw_event)
        if event is None:
            raise EvaluationServiceError("Evaluation produced an invalid public event.")
        self._events.append(event)
        if len(self._events) > _EVENT_LIMIT:
            self._events = self._events[-_EVENT_LIMIT:]
        self._next_event_sequence += 1
        _write_event_record(self._events_path, self._events, self._next_event_sequence)

    def _update(
        self,
        job_id: str,
        *,
        event_phase: str | None = None,
        event_trial: Mapping[str, Any] | None = None,
        event_result: Mapping[str, Any] | None = None,
        **values: Any,
    ) -> None:
        with self._lock:
            if self._job.get("id") != job_id:
                return
            self._job.update(values)
            _write_job_record(self._record_path, self._job)
            if event_phase is not None:
                self._append_event_locked(
                    event_phase,
                    trial=event_trial,
                    result=event_result,
                )

    def plan(self) -> dict[str, Any]:
        """Freeze the configured proof matrix without starting any producer."""

        with self._lock:
            if self._job["status"] in _LIVE_JOB_STATUSES:
                raise EvaluationServiceError("An evaluation job is already running.")
            lease = acquire_project_lease(self._config_path)
            try:
                with lease.activate("run"):
                    plan = plan_project_evaluation(self._config_path)
                with self._reports_lock:
                    self._reports_cache_key = None
                    self._reports_cache = {}
                self._job = _idle_job(plan_id=str(plan["plan_id"]))
                _write_job_record(self._record_path, self._job)
            finally:
                lease.release()
        return self.workspace()

    def start(self, suite_id: str, *, resume: bool = True) -> dict[str, Any]:
        """Queue a local built-in/command producer; external suites cannot start here."""

        suite_name = str(suite_id).strip()
        if not suite_name or not _SAFE_ID.fullmatch(suite_name):
            raise EvaluationServiceError("suite is required")
        with self._lock:
            if self._job["status"] in _LIVE_JOB_STATUSES:
                raise EvaluationServiceError("An evaluation job is already running.")
            lease = acquire_project_lease(self._config_path)
            device_lease: DeviceLease | None = None
            try:
                config = _load_project(self._config_path)
                plan, run_dir = load_current_plan(config.data, config.root)
                suite = plan.get("suites", {}).get(suite_name)
                if not isinstance(suite, Mapping):
                    raise EvaluationServiceError(f"Unknown evaluation suite: {suite_name}")
                runner = suite.get("runner", {})
                if not isinstance(runner, Mapping) or runner.get("type") not in {
                    "builtin",
                    "command",
                }:
                    raise EvaluationServiceError(
                        f"Suite {suite_name!r} is external. Export its frozen requests and "
                        "ingest the returned results; AutoTrainer will not pretend Fable started."
                    )
                # Reserve the physical device before publishing `queued` so a
                # different project cannot accept a second 9B job in the same
                # browser/CLI race window.
                device_lease = acquire_device_lease()
                scored = _suite_results(plan, run_dir, suite_name)
                total = sum(
                    1
                    for trial in plan.get("trials", [])
                    if isinstance(trial, Mapping) and trial.get("suite_id") == suite_name
                )
                job_id = uuid4().hex
                self._job = {
                    "id": job_id,
                    "status": "queued",
                    "plan_id": str(plan["plan_id"]),
                    "suite": suite_name,
                    "phase": "queued",
                    "message": "The local benchmark is queued.",
                    "completed": len(scored),
                    "total": total,
                    "current_trial": None,
                    "results": scored[-_PUBLIC_RESULT_LIMIT:],
                    "results_truncated": len(scored) > _PUBLIC_RESULT_LIMIT,
                }
                _write_job_record(self._record_path, self._job)
                self._append_event_locked("queued")
                worker = Thread(
                    target=self._run,
                    args=(job_id, suite_name, resume, lease, device_lease),
                    name=f"autotrainer-eval-{job_id[:8]}",
                    daemon=False,
                )
                self._worker = worker
                try:
                    worker.start()
                except RuntimeError as error:
                    self._job.update(
                        status="failed",
                        phase="failed",
                        message="Evaluation could not start its local worker.",
                    )
                    _write_job_record(self._record_path, self._job)
                    self._append_event_locked("failed")
                    self._worker = None
                    raise EvaluationServiceError(self._job["message"]) from error
            except Exception:
                if self._worker is None or not self._worker.is_alive():
                    if device_lease is not None:
                        device_lease.release()
                    lease.release()
                raise
        return self.snapshot()

    def _run(
        self,
        job_id: str,
        suite_id: str,
        resume: bool,
        lease: ProjectLease,
        device_lease: DeviceLease,
    ) -> None:
        def progress(event: Mapping[str, Any]) -> None:
            phase = str(event.get("phase", "running"))
            trial = _safe_trial(event.get("trial"))
            completed = _safe_integer(event.get("completed"))
            total = _safe_integer(event.get("total"))
            messages = {
                "queued": "The local benchmark is starting.",
                "resuming": "Completed immutable trials are being retained.",
                "generating": "The configured model runner is generating a patch.",
                "verifying": "AutoTrainer is verifying the patch in the trusted environment.",
                "trial_completed": "One benchmark trial completed trusted verification.",
                "completed": "All local benchmark trials are complete.",
            }
            values: dict[str, Any] = {
                # The final callback means every trial boundary was observed,
                # but the worker still validates the complete scored set before
                # publishing a terminal completed state.
                "status": "running",
                "phase": phase if phase in _PHASES else "running",
                "message": messages.get(phase, "The local benchmark is running."),
                "completed": completed,
                "total": total,
                "current_trial": trial,
            }
            event_result: Mapping[str, Any] | None = None
            if phase in {"trial_completed", "completed"}:
                config = _load_project(self._config_path)
                plan, run_dir = load_current_plan(config.data, config.root)
                results = _suite_results(plan, run_dir, suite_id)
                values["results"] = results[-_PUBLIC_RESULT_LIMIT:]
                values["results_truncated"] = len(results) > _PUBLIC_RESULT_LIMIT
                if phase == "trial_completed" and trial is not None:
                    event_result = next(
                        (
                            result
                            for result in results
                            if result.get("trial_id") == trial["trial_id"]
                        ),
                        None,
                    )
            self._update(
                job_id,
                # The core's completed callback precedes this service's final
                # scored-set validation. Publish completed only after that
                # validation below, never optimistically here.
                event_phase=(
                    phase
                    if phase in {"generating", "verifying", "trial_completed"}
                    else None
                ),
                event_trial=trial,
                event_result=event_result,
                **values,
            )

        try:
            with lease.activate("run"), device_lease.activate():
                try:
                    result = run_project_evaluation(
                        self._config_path,
                        suite_id,
                        resume=resume,
                        on_progress=progress,
                    )
                    config = _load_project(self._config_path)
                    plan, run_dir = load_current_plan(config.data, config.root)
                    results = _suite_results(plan, run_dir, suite_id)
                    total = _safe_integer(result.get("total"))
                    if len(results) != total:
                        raise EvaluationServiceError(
                            "The local producer returned without one trusted result per trial."
                        )
                    # The comparison core remains the only owner of paired
                    # bootstrap statistics. Publish its current report before
                    # declaring the durable job complete so every client sees
                    # the same final decision on its next read.
                    build_evaluation_reports(config.data, config.root)
                    with self._reports_lock:
                        self._reports_cache_key = None
                        self._reports_cache = {}
                    self._update(
                        job_id,
                        event_phase="completed",
                        status="completed",
                        phase="completed",
                        message="The local benchmark completed.",
                        completed=total,
                        total=total,
                        current_trial=None,
                        results=results[-_PUBLIC_RESULT_LIMIT:],
                        results_truncated=len(results) > _PUBLIC_RESULT_LIMIT,
                    )
                except Exception as error:
                    public = (
                        str(error)
                        if isinstance(error, (ConfigError, EvaluationError))
                        else "Evaluation stopped after an unexpected local backend failure."
                    )
                    self._update(
                        job_id,
                        event_phase="failed",
                        status="failed",
                        phase="failed",
                        message=_redact(public),
                        current_trial=None,
                    )
        finally:
            # Producer/model locals have unwound at this boundary; clear any
            # unused allocator blocks before publishing GPU 0 as available.
            clear_cuda_memory()
            device_lease.release()
            lease.release()

    def close(self) -> None:
        """Wait for this manager's non-daemon worker during backend shutdown."""

        with self._lock:
            worker = self._worker
        if worker is not None and worker is not current_thread() and worker.is_alive():
            worker.join()


__all__ = [
    "EvaluationJobManager",
    "EvaluationServiceError",
    "plan_project_evaluation",
    "run_project_evaluation",
]
