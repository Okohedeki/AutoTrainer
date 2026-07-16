"""Shared, single-GPU training orchestration for the GUI and agent CLI.

The stage runners remain the source of model and recipe policy.  This module
only chooses the stages justified by prepared data and serializes local jobs so
two browser clicks cannot compete for the same GPU.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
import os
from pathlib import Path
import re
from threading import Lock, Thread, current_thread
from typing import Any, Callable, Mapping
from uuid import uuid4

from .config import ConfigError
from .model_cache import ModelCacheError
from .project_service import prepare_project, read_project_config
from .project_gate import (
    ProjectLease,
    acquire_project_lease,
    project_is_busy,
    project_run_gate,
)
from .training import run_grpo, run_sft
from .training.common import (
    TrainingConfigurationError,
    TrainingDependencyError,
    TrainingRuntimeError,
)
from .training.selection import TRAINING_RECIPES, select_stage_config


ProgressCallback = Callable[[str, str], None]

_JOB_SCHEMA_VERSION = 1
_JOB_STATUSES = frozenset(
    {"idle", "queued", "running", "completed", "failed", "interrupted"}
)
_LIVE_JOB_STATUSES = frozenset({"queued", "running"})
_RECIPES = frozenset(TRAINING_RECIPES)
_STAGES = frozenset({"prepare", "sft", "grpo"})
_METRIC_KEY = re.compile(r"^[A-Za-z0-9_.-]{1,80}$")
_SECRET_PATTERNS = (
    re.compile(r"\bhf_[A-Za-z0-9]{8,}\b"),
    re.compile(r"(?i)\b(bearer\s+)[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(
        r"(?i)\b(api[-_ ]?key|token|password|secret)\s*[:=]\s*[^\s,;]+"
    ),
)


class TrainingServiceError(ConfigError):
    """Raised when prepared project state cannot start an honest training run."""


def _notify(callback: ProgressCallback | None, stage: str, message: str) -> None:
    if callback is not None:
        callback(stage, message)


def _run_project_training_owned(
    config_path: str | Path,
    *,
    on_progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Run exactly the stages recommended by the project's prepared evidence."""

    _notify(on_progress, "prepare", "Checking the project.")
    preparation = prepare_project(config_path)
    if preparation.get("status") != "ready":
        next_action = preparation.get("next_action")
        detail = (
            str(next_action.get("detail", ""))
            if isinstance(next_action, Mapping)
            else str(preparation.get("summary", ""))
        )
        raise TrainingServiceError(detail or "Prepare the project before training.")

    recipe = str(preparation.get("recipe", ""))
    if recipe not in TRAINING_RECIPES:
        raise TrainingServiceError("Prepared data does not select a training recipe.")

    # Load the exact file again after preparation because preparation may write
    # deterministic compiled artifacts that the guarded stage runners consume.
    config = read_project_config(config_path)
    stage_config = select_stage_config(config.data, recipe)
    stages: list[dict[str, Any]] = []
    if recipe in {"teach", "both"}:
        _notify(on_progress, "sft", "Teaching from approved examples.")
        stages.append(
            run_sft(
                stage_config,
                project_root=config.root,
                output_dir=config.resolve_path(config.data["sft"]["output_dir"]),
                dry_run=False,
            )
        )
    if recipe in {"practice", "both"}:
        _notify(on_progress, "grpo", "Practicing against verified tasks.")
        stages.append(
            run_grpo(
                stage_config,
                project_root=config.root,
                output_dir=config.resolve_path(config.data["grpo"]["output_dir"]),
                dry_run=False,
            )
        )

    return {
        "status": "completed",
        "recipe": recipe,
        "stages": stages,
    }


def run_project_training(
    config_path: str | Path,
    *,
    on_progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Hold the project snapshot stable from Prepare through every stage."""

    with project_run_gate(config_path):
        return _run_project_training_owned(config_path, on_progress=on_progress)


def _public_error(error: BaseException) -> str:
    """Return useful local errors without reflecting arbitrary exception text."""

    expected = (
        ConfigError,
        ModelCacheError,
        TrainingConfigurationError,
        TrainingDependencyError,
        TrainingRuntimeError,
    )
    if isinstance(error, expected):
        return str(error)
    return "Training stopped after an unexpected local backend failure. Check the backend terminal."


def _redact_secrets(value: object, *, limit: int) -> str:
    """Bound persisted text and remove common credential forms defensively."""

    text = str(value).replace("\r", " ").replace("\n", " ")[:limit]
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(lambda match: f"{match.group(1)}[redacted]" if match.lastindex else "[redacted]", text)
    return text


def _sanitize_metrics(value: object) -> dict[str, int | float | bool]:
    """Keep useful scalar trainer metrics without persisting arbitrary text."""

    if not isinstance(value, Mapping):
        return {}
    metrics: dict[str, int | float | bool] = {}
    for raw_key, raw_value in value.items():
        key = str(raw_key)
        if not _METRIC_KEY.fullmatch(key):
            continue
        if isinstance(raw_value, bool):
            metrics[key] = raw_value
        elif isinstance(raw_value, int):
            metrics[key] = raw_value
        elif isinstance(raw_value, float) and math.isfinite(raw_value):
            metrics[key] = raw_value
    return metrics


def _sanitize_result(value: object) -> dict[str, Any] | None:
    """Whitelist completion evidence safe to retain in the token-free record."""

    if not isinstance(value, Mapping):
        return None
    recipe = str(value.get("recipe", ""))
    if recipe not in _RECIPES:
        return None

    stages: list[dict[str, Any]] = []
    raw_stages = value.get("stages")
    if isinstance(raw_stages, list):
        for raw_stage in raw_stages:
            if not isinstance(raw_stage, Mapping):
                continue
            stage_name = str(raw_stage.get("stage", ""))
            if stage_name not in {"sft", "grpo"}:
                continue
            stage: dict[str, Any] = {
                "status": "completed",
                "stage": stage_name,
            }
            output_dir = raw_stage.get("output_dir")
            if output_dir:
                stage["output_dir"] = _redact_secrets(output_dir, limit=4096)
            metrics = _sanitize_metrics(raw_stage.get("metrics"))
            if metrics:
                stage["metrics"] = metrics
            trainable_parameters = raw_stage.get("trainable_adapter_parameters")
            if isinstance(trainable_parameters, int) and not isinstance(
                trainable_parameters, bool
            ):
                stage["trainable_adapter_parameters"] = trainable_parameters
            stages.append(stage)

    return {
        "status": "completed",
        "recipe": recipe,
        "stages": stages,
    }


def _idle_job() -> dict[str, Any]:
    return {
        "id": None,
        "status": "idle",
        "recipe": None,
        "stage": None,
        "message": "No training job has started.",
        "result": None,
    }


def _normalize_saved_job(value: object) -> dict[str, Any] | None:
    """Validate an on-disk record before returning any part of it to the API."""

    if not isinstance(value, Mapping):
        return None
    status = str(value.get("status", ""))
    if status not in _JOB_STATUSES:
        return None
    if status == "idle":
        return _idle_job()

    job_id = value.get("id")
    if not isinstance(job_id, str) or not re.fullmatch(r"[0-9a-f]{32}", job_id):
        return None
    recipe_value = value.get("recipe")
    recipe = str(recipe_value) if recipe_value is not None else None
    if recipe not in _RECIPES | {None}:
        recipe = None
    stage_value = value.get("stage")
    stage = str(stage_value) if stage_value is not None else None
    if stage not in _STAGES | {None}:
        stage = None
    message = _redact_secrets(value.get("message", "Training status is available."), limit=1000)
    return {
        "id": job_id,
        "status": status,
        "recipe": recipe,
        "stage": stage,
        "message": message,
        "result": _sanitize_result(value.get("result")),
    }


def _write_job_record(destination: Path, job: Mapping[str, Any]) -> None:
    """Atomically replace the one small project-local lifecycle record."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    payload = {"schema_version": _JOB_SCHEMA_VERSION, "job": dict(job)}
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
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


class TrainingJobManager:
    """Own and durably report one local single-GPU training job."""

    def __init__(self, config_path: str | Path) -> None:
        self._lock = Lock()
        self._config_path = Path(config_path).expanduser().resolve()
        config = read_project_config(self._config_path)
        self._record_path = config.artifact_dir / "training" / "current-job.json"
        self._worker: Thread | None = None
        self._job = _read_job_record(self._record_path) or _idle_job()
        if self._job["status"] in _LIVE_JOB_STATUSES and not project_is_busy(
            self._config_path
        ):
            # A fresh Python process cannot own the thread recorded by the old
            # process. Calling it interrupted is honest and permits an explicit retry.
            self._job.update(
                status="interrupted",
                message="Training was interrupted when the local backend stopped.",
                result=None,
            )
            _write_job_record(self._record_path, self._job)

    @property
    def record_path(self) -> Path:
        """Expose the inspectable artifact location without exposing its contents."""

        return self._record_path

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return deepcopy(self._job)

    def _update(self, job_id: str, **values: Any) -> None:
        with self._lock:
            if self._job.get("id") != job_id:
                return
            self._job.update(values)
            _write_job_record(self._record_path, self._job)

    def start(self) -> dict[str, Any]:
        """Queue the project unless this backend already owns a live GPU job."""

        with self._lock:
            if self._job["status"] in _LIVE_JOB_STATUSES:
                raise TrainingServiceError("A training job is already running.")
            # Reserve the cross-process lease before exposing `queued`. This
            # closes the short API window in which a setup request could have
            # changed YAML before the worker thread reached Prepare.
            lease = acquire_project_lease(self._config_path)
            job_id = uuid4().hex
            self._job = {
                "id": job_id,
                "status": "queued",
                "recipe": None,
                "stage": "prepare",
                "message": "Training is queued.",
                "result": None,
            }
            try:
                _write_job_record(self._record_path, self._job)
            except Exception:
                lease.release()
                raise

            # The API remains responsive while the single worker owns model
            # loading and the GPU. A non-daemon thread lets server shutdown wait
            # for the active adapter write instead of abandoning it mid-file.
            worker = Thread(
                target=self._run,
                args=(job_id, lease),
                name=f"autotrainer-{job_id[:8]}",
                daemon=False,
            )
            self._worker = worker
            try:
                worker.start()
            except RuntimeError as error:
                self._job.update(
                    status="failed",
                    message="Training could not start its local worker.",
                )
                _write_job_record(self._record_path, self._job)
                self._worker = None
                lease.release()
                raise TrainingServiceError(self._job["message"]) from error
        return self.snapshot()

    def _run(self, job_id: str, lease: ProjectLease) -> None:
        def progress(stage: str, message: str) -> None:
            self._update(
                job_id,
                status="running",
                stage=stage,
                message=_redact_secrets(message, limit=1000),
            )

        try:
            # The request thread reserved this lease before queueing. Context
            # activation transfers the narrow Prepare bypass to this worker.
            with lease.activate("run"):
                try:
                    result = run_project_training(
                        self._config_path, on_progress=progress
                    )
                    recipe = str(result["recipe"])
                    last_stage = (
                        "grpo" if recipe in {"practice", "both"} else "sft"
                    )
                    self._update(
                        job_id,
                        status="completed",
                        recipe=recipe,
                        stage=last_stage,
                        message="Training completed. The adapter is ready.",
                        result=_sanitize_result(result),
                    )
                except Exception as error:  # worker boundary reaches a terminal state
                    self._update(
                        job_id,
                        status="failed",
                        message=_redact_secrets(_public_error(error), limit=1000),
                        result=None,
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
    "TrainingJobManager",
    "TrainingServiceError",
    "run_project_training",
]
