"""Shared, single-GPU training orchestration for the GUI and agent CLI.

The stage runners remain the source of model and recipe policy.  This module
only chooses the stages justified by prepared data and serializes local jobs so
two browser clicks cannot compete for the same GPU.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from threading import Lock, Thread
from typing import Any, Callable, Mapping
from uuid import uuid4

from .config import ConfigError
from .model_cache import ModelCacheError
from .project_service import prepare_project, read_project_config
from .training import run_grpo, run_sft
from .training.common import (
    TrainingConfigurationError,
    TrainingDependencyError,
    TrainingRuntimeError,
)
from .training.selection import TRAINING_RECIPES, select_stage_config


ProgressCallback = Callable[[str, str], None]


class TrainingServiceError(ConfigError):
    """Raised when prepared project state cannot start an honest training run."""


def _notify(callback: ProgressCallback | None, stage: str, message: str) -> None:
    if callback is not None:
        callback(stage, message)


def run_project_training(
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


class TrainingJobManager:
    """Own one in-process job; closing a browser tab does not cancel it."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._job: dict[str, Any] = {
            "id": None,
            "status": "idle",
            "recipe": None,
            "stage": None,
            "message": "No training job has started.",
        }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return deepcopy(self._job)

    def _update(self, **values: Any) -> None:
        with self._lock:
            self._job.update(values)

    def start(self, config_path: str | Path) -> dict[str, Any]:
        """Queue the project unless this backend already owns a live GPU job."""

        with self._lock:
            if self._job["status"] in {"queued", "running"}:
                raise TrainingServiceError("A training job is already running.")
            job_id = uuid4().hex
            self._job = {
                "id": job_id,
                "status": "queued",
                "recipe": None,
                "stage": "prepare",
                "message": "Training is queued.",
            }

        # The API remains responsive while the single worker owns model loading
        # and the GPU. It deliberately does not start more than one worker.
        worker = Thread(
            target=self._run,
            args=(Path(config_path).expanduser().resolve(), job_id),
            name=f"autotrainer-{job_id[:8]}",
            daemon=True,
        )
        worker.start()
        return self.snapshot()

    def _run(self, config_path: Path, job_id: str) -> None:
        def progress(stage: str, message: str) -> None:
            self._update(status="running", stage=stage, message=message)

        try:
            result = run_project_training(config_path, on_progress=progress)
            recipe = str(result["recipe"])
            last_stage = "grpo" if recipe in {"practice", "both"} else "sft"
            self._update(
                status="completed",
                recipe=recipe,
                stage=last_stage,
                message="Training completed. The adapter is ready.",
            )
        except Exception as error:  # worker boundary must always reach a terminal state
            self._update(
                status="failed",
                message=_public_error(error),
            )


__all__ = [
    "TrainingJobManager",
    "TrainingServiceError",
    "run_project_training",
]
