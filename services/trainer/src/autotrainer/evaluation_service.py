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

from .config import ConfigError, ProjectConfig, load_config
from .evaluation import (
    EvaluationError,
    EvaluationProgressCallback,
    RESULT_COMPONENTS,
    load_current_plan,
    run_command_suite,
    write_evaluation_plan,
)
from .project_gate import (
    ProjectLease,
    acquire_project_lease,
    project_is_busy,
    project_run_gate,
)


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
    """Read only valid, planned scored rows in deterministic trial order."""

    results: list[dict[str, Any]] = []
    scored_dir = run_dir / "scored-trials"
    for raw_trial in plan.get("trials", []):
        if not isinstance(raw_trial, Mapping) or raw_trial.get("suite_id") != suite_id:
            continue
        trial = _safe_trial(raw_trial)
        if trial is None:
            raise EvaluationServiceError("The frozen evaluation plan contains an invalid trial.")
        path = scored_dir / f"{trial['trial_id']}.json"
        if not path.is_file():
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise EvaluationServiceError(
                f"A scored evaluation result is unreadable: {path.name}"
            ) from error
        result = _safe_result(value, trial)
        if (
            result is None
            or value.get("plan_id") != plan.get("plan_id")
            or value.get("suite_id") != suite_id
            or set(result["components"]) != set(RESULT_COMPONENTS)
        ):
            raise EvaluationServiceError(
                f"A scored evaluation result does not match its frozen trial: {path.name}"
            )
        results.append(result)
    return results


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


def _load_project(config_path: str | Path) -> ProjectConfig:
    return load_config(config_path, check_paths=True)


def plan_project_evaluation(config_path: str | Path) -> dict[str, Any]:
    """Freeze a plan under the same project lease used by local training."""

    with project_run_gate(config_path):
        config = _load_project(config_path)
        return write_evaluation_plan(config.data, config.root)


def run_project_evaluation(
    config_path: str | Path,
    suite_id: str,
    *,
    resume: bool = True,
    on_progress: EvaluationProgressCallback | None = None,
) -> dict[str, Any]:
    """Run one command-backed suite while holding the project snapshot stable."""

    with project_run_gate(config_path):
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
        stage = value.get("stages", {}).get("evaluation", {})
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, AttributeError):
        return {
            "status": "invalid",
            "ready_task_count": 0,
            "blockers": ["The prepared evaluation snapshot is unreadable."],
            "warnings": [],
        }
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


def _report_is_current(
    run_dir: Path, suite_id: str, *, completed: int, total: int
) -> bool:
    """Accept a report marker only when its recorded completeness is current."""

    path = run_dir / "reports" / f"{suite_id}.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        completeness = value.get("completeness", {})
    except (FileNotFoundError, OSError, UnicodeDecodeError, json.JSONDecodeError, AttributeError):
        return False
    return bool(
        isinstance(completeness, Mapping)
        and completeness.get("completed_trials") == completed
        and completeness.get("expected_trials") == total
    )


def _suite_snapshot(
    plan: Mapping[str, Any],
    run_dir: Path,
    suite_id: str,
    job: Mapping[str, Any],
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
    report_exists = _report_is_current(
        run_dir, suite_id, completed=completed, total=total
    )

    review = None
    if kind == "fable_ab":
        review_config = suite.get("review", {})
        reviewers = (
            _safe_integer(review_config.get("reviewers_per_pair"), default=1)
            if isinstance(review_config, Mapping)
            else 1
        )
        review = _review_counts(run_dir, suite_id, total // 2, max(1, reviewers))

    if total == 0:
        phase = "blocked"
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
        message = "The frozen command-backed benchmark is ready to start."

    # Fable results stay arm-blind in the localhost response.  The reviewed
    # report, not per-arm trial telemetry, is the point where identity returns.
    public_results = [] if kind == "fable_ab" else scored[-_PUBLIC_RESULT_LIMIT:]
    return {
        "id": suite_id,
        "kind": kind,
        "runner_type": runner_type,
        "phase": phase,
        "message": message,
        "completed": completed,
        "total": total,
        "results": public_results,
        "results_truncated": len(scored) > len(public_results) and kind != "fable_ab",
        "results_withheld_for_blind_review": kind == "fable_ab" and bool(scored),
        "review": review,
    }


class EvaluationJobManager:
    """Own one durable command-suite job and expose all suite phases honestly."""

    def __init__(self, config_path: str | Path) -> None:
        self._lock = Lock()
        self._config_path = Path(config_path).expanduser().resolve()
        config = _load_project(self._config_path)
        self._record_path = config.artifact_dir / "evaluation" / "current-job.json"
        self._worker: Thread | None = None
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

    @property
    def record_path(self) -> Path:
        return self._record_path

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return deepcopy(self._job)

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
        suites = [
            _suite_snapshot(plan, run_dir, suite_id, job)
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

    def _update(self, job_id: str, **values: Any) -> None:
        with self._lock:
            if self._job.get("id") != job_id:
                return
            self._job.update(values)
            _write_job_record(self._record_path, self._job)

    def plan(self) -> dict[str, Any]:
        """Freeze the configured proof matrix without starting any producer."""

        with self._lock:
            if self._job["status"] in _LIVE_JOB_STATUSES:
                raise EvaluationServiceError("An evaluation job is already running.")
            lease = acquire_project_lease(self._config_path)
            try:
                with lease.activate("run"):
                    plan = plan_project_evaluation(self._config_path)
                self._job = _idle_job(plan_id=str(plan["plan_id"]))
                _write_job_record(self._record_path, self._job)
            finally:
                lease.release()
        return self.workspace()

    def start(self, suite_id: str) -> dict[str, Any]:
        """Queue only an explicit command runner; external suites cannot start here."""

        suite_name = str(suite_id).strip()
        if not suite_name or not _SAFE_ID.fullmatch(suite_name):
            raise EvaluationServiceError("suite is required")
        with self._lock:
            if self._job["status"] in _LIVE_JOB_STATUSES:
                raise EvaluationServiceError("An evaluation job is already running.")
            lease = acquire_project_lease(self._config_path)
            try:
                config = _load_project(self._config_path)
                plan, run_dir = load_current_plan(config.data, config.root)
                suite = plan.get("suites", {}).get(suite_name)
                if not isinstance(suite, Mapping):
                    raise EvaluationServiceError(f"Unknown evaluation suite: {suite_name}")
                runner = suite.get("runner", {})
                if not isinstance(runner, Mapping) or runner.get("type") != "command":
                    raise EvaluationServiceError(
                        f"Suite {suite_name!r} is external. Export its frozen requests and "
                        "ingest the returned results; AutoTrainer will not pretend Fable started."
                    )
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
                worker = Thread(
                    target=self._run,
                    args=(job_id, suite_name, lease),
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
                    self._worker = None
                    raise EvaluationServiceError(self._job["message"]) from error
            except Exception:
                if self._worker is None or not self._worker.is_alive():
                    lease.release()
                raise
        return self.snapshot()

    def _run(self, job_id: str, suite_id: str, lease: ProjectLease) -> None:
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
            if phase in {"trial_completed", "completed"}:
                config = _load_project(self._config_path)
                plan, run_dir = load_current_plan(config.data, config.root)
                results = _suite_results(plan, run_dir, suite_id)
                values["results"] = results[-_PUBLIC_RESULT_LIMIT:]
                values["results_truncated"] = len(results) > _PUBLIC_RESULT_LIMIT
            self._update(job_id, **values)

        try:
            with lease.activate("run"):
                try:
                    result = run_project_evaluation(
                        self._config_path,
                        suite_id,
                        resume=True,
                        on_progress=progress,
                    )
                    config = _load_project(self._config_path)
                    plan, run_dir = load_current_plan(config.data, config.root)
                    results = _suite_results(plan, run_dir, suite_id)
                    total = _safe_integer(result.get("total"))
                    if len(results) != total:
                        raise EvaluationServiceError(
                            "The command runner returned without one trusted result per trial."
                        )
                    self._update(
                        job_id,
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
                        status="failed",
                        phase="failed",
                        message=_redact(public),
                        current_trial=None,
                    )
        finally:
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
