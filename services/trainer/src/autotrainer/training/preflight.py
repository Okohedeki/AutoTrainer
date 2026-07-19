"""Executable GRPO task preflight shared by Prepare and real training.

Static manifest checks cannot prove that the pinned container can install the
repository, execute its gates, or produce a valid hidden-verifier report. This
module runs a deterministic no-edit baseline episode for every task before model loading.
It is intentionally not a model rollout and makes no claim about reward
variance across sampled completions.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from math import isfinite
from numbers import Real
from pathlib import Path
from typing import Any

from ..integrity import IntegrityError, resolve_container_image
from .common import (
    TrainingRuntimeError,
    _json_records,
    import_factory,
    verify_dataset_identity,
)


_REQUIRED_CHECKS = ("build", "tests", "verifier")
_OPTIONAL_CHECKS = ("install", "browserTests")


def _validate_environment_rows(
    records: list[tuple[int, Mapping[str, Any]]],
    *,
    backend: str,
    image: str,
    dataset_name: str,
) -> None:
    """Reject compiled rows that diverge from the resolved recipe runtime."""

    for position, row in records:
        row_backend = str(row.get("environment_backend", "")).strip()
        row_image = str(row.get("environment_image", "")).strip()
        if row_backend != backend or row_image != image:
            raise TrainingRuntimeError(
                f"{dataset_name} record {position} does not use the configured container "
                "runtime; re-run compilation before training"
            )


def _bind_environment_rows(
    records: list[tuple[int, Mapping[str, Any]]],
    runtime_reference: str,
) -> list[tuple[int, Mapping[str, Any]]]:
    """Copy compiled rows and bind them to the already-resolved image identity."""

    bound: list[tuple[int, Mapping[str, Any]]] = []
    for position, row in records:
        runtime_row = dict(row)
        runtime_row["environment_image_identity"] = runtime_reference
        bound.append((position, runtime_row))
    return bound


def _freeze_environment_image(
    environment: Mapping[str, Any],
    records: list[tuple[int, Mapping[str, Any]]],
    image_resolver: Callable[[str, str], Mapping[str, str]],
) -> tuple[list[tuple[int, Mapping[str, Any]]], dict[str, str]]:
    """Resolve the configured tag once and bind every executable task row to it."""

    backend = str(environment.get("backend", "")).strip()
    image = str(environment.get("image", "")).strip()
    if backend not in {"docker", "podman"}:
        raise TrainingRuntimeError(
            "resolved GRPO recipe environment backend must be docker or podman"
        )
    if not image:
        raise TrainingRuntimeError(
            "resolved GRPO recipe environment image must be non-empty"
        )
    _validate_environment_rows(
        records,
        backend=backend,
        image=image,
        dataset_name="grpo.dataset",
    )
    try:
        resolved = image_resolver(backend, image)
    except IntegrityError as error:
        raise TrainingRuntimeError(str(error)) from error
    except Exception as error:
        raise TrainingRuntimeError(
            f"could not resolve container image {image!r} with {backend}: {error}"
        ) from error
    if not isinstance(resolved, Mapping):
        raise TrainingRuntimeError("container image resolver returned invalid evidence")
    identity = {str(key): str(value) for key, value in resolved.items()}
    runtime_reference = identity.get("runtime_reference", "").strip()
    digest = identity.get("digest", "").strip()
    if (
        identity.get("backend") != backend
        or identity.get("reference") != image
        or not runtime_reference
        or not digest.startswith("sha256:")
    ):
        raise TrainingRuntimeError("container image resolver returned invalid evidence")
    return _bind_environment_rows(records, runtime_reference), identity


def _check_detail(check: Any) -> str:
    detail = str(getattr(check, "stderr", "") or getattr(check, "stdout", "")).strip()
    return f": {detail[:500]}" if detail else ""


def _run_task_canary(
    row: Mapping[str, Any],
    task_id: str,
    factory: Callable[[], Any],
) -> dict[str, Any]:
    """Run and validate one no-edit task episode, always cleaning its workspace."""

    instance: Any | None = None
    try:
        instance = factory()
        reset = getattr(instance, "reset", None)
        get_reward = getattr(instance, "get_reward", None)
        if not callable(reset) or not callable(get_reward):
            raise TrainingRuntimeError(
                "GRPO environment must expose callable reset and get_reward methods"
            )
        observation = reset(**dict(row))
        if not isinstance(observation, str) or not observation.strip():
            raise TrainingRuntimeError(
                "GRPO environment reset must return a text observation"
            )
        # TRL appends reset output directly to the last prompt message. Require
        # a visible boundary so custom environments cannot glue observation
        # text onto the operator's instruction or repeat it ambiguously.
        if not observation.startswith("\n\n"):
            raise TrainingRuntimeError(
                "GRPO environment reset observation must begin with a blank-line separator"
            )
        reward = get_reward()

        if (
            isinstance(reward, bool)
            or not isinstance(reward, Real)
            or not isfinite(float(reward))
        ):
            raise TrainingRuntimeError(
                f"GRPO executable task canary {task_id!r} returned a non-finite scalar reward"
            )

        result = getattr(instance, "last_result", None)
        if result is None:
            raise TrainingRuntimeError(
                f"GRPO executable task canary {task_id!r} produced no structured episode result"
            )
        hard_gate_reason = getattr(result, "hard_gate_reason", None)
        if hard_gate_reason:
            raise TrainingRuntimeError(
                f"GRPO executable task canary {task_id!r} failed {hard_gate_reason}"
            )

        checks = getattr(result, "checks", None)
        if not isinstance(checks, Mapping):
            raise TrainingRuntimeError(
                f"GRPO executable task canary {task_id!r} produced no check evidence"
            )
        for name in _REQUIRED_CHECKS:
            check = checks.get(name)
            if check is None or getattr(check, "configured", None) is not True:
                raise TrainingRuntimeError(
                    f"GRPO executable task canary {task_id!r} did not configure {name}"
                )
            if getattr(check, "passed", None) is not True:
                raise TrainingRuntimeError(
                    f"GRPO executable task canary {task_id!r} failed {name}"
                    f"{_check_detail(check)}"
                )
        for name in _OPTIONAL_CHECKS:
            check = checks.get(name)
            if check is not None and getattr(check, "configured", False):
                if getattr(check, "passed", None) is not True:
                    raise TrainingRuntimeError(
                        f"GRPO executable task canary {task_id!r} failed {name}"
                        f"{_check_detail(check)}"
                    )

        raw_rates = getattr(result, "raw_verifier_rates", None)
        if not isinstance(raw_rates, Mapping) or "task_tests" not in raw_rates:
            raise TrainingRuntimeError(
                f"GRPO executable task canary {task_id!r} produced no task-test signal"
            )
        raw_task_pass_rate = raw_rates["task_tests"]
        if (
            isinstance(raw_task_pass_rate, bool)
            or not isinstance(raw_task_pass_rate, Real)
            or not isfinite(float(raw_task_pass_rate))
            or not 0.0 <= float(raw_task_pass_rate) <= 1.0
        ):
            raise TrainingRuntimeError(
                f"GRPO executable task canary {task_id!r} produced an invalid task-test signal"
            )
        task_pass_rate = float(raw_task_pass_rate)
        if task_pass_rate >= 1.0:
            raise TrainingRuntimeError(
                f"GRPO executable task canary {task_id!r} already passes every task test at "
                "the starting revision. Tighten the verifier or choose an unsolved task."
            )

        return {
            "task_id": task_id,
            "baseline_reward": round(float(reward), 6),
            "task_pass_rate": task_pass_rate,
            "signal": "unsaturated_baseline",
            "checks": {
                name: str(getattr(check, "status", "unknown"))
                for name, check in sorted(checks.items())
            },
        }
    except TrainingRuntimeError:
        raise
    except Exception as error:
        raise TrainingRuntimeError(
            f"GRPO executable task canary {task_id!r} failed: {error}"
        ) from error
    finally:
        cleanup = getattr(instance, "_cleanup", None)
        if callable(cleanup):
            cleanup()


def run_grpo_environment_canary(
    recipe: Mapping[str, Any],
    *,
    factory: Callable[[], Any] | None = None,
    image_resolver: Callable[[str, str], Mapping[str, str]] = resolve_container_image,
) -> dict[str, Any]:
    """Execute every compiled task through the real RL environment without edits.

    Prepare and direct GRPO execution share this full-dataset gate. A successful
    result proves every selected source snapshot, runtime command, and hidden
    verifier can run together, and rejects rows whose starting state already
    satisfies all task-specific checks.

    Args:
        recipe: Fully resolved GRPO recipe.
        factory: Optional environment factory injection used by unit tests.
        image_resolver: Optional immutable-image resolver injection used by unit tests.

    Returns:
        JSON-compatible evidence for every exercised task and baseline.

    Raises:
        TrainingRuntimeError: If the executable task contract is unavailable,
            any runtime gate fails, the verifier report is invalid, or the
            baseline task checks are already saturated.
    """

    stage = recipe.get("grpo")
    environment = recipe.get("environment")
    if not isinstance(stage, Mapping) or not isinstance(environment, Mapping):
        raise TrainingRuntimeError(
            "resolved GRPO recipe is missing stage or environment data"
        )
    dataset = stage.get("dataset")
    if not isinstance(dataset, Mapping) or not dataset.get("path"):
        raise TrainingRuntimeError("resolved GRPO recipe is missing its dataset path")

    verify_dataset_identity(dataset)
    dataset_path = Path(str(dataset["path"]))
    records = _json_records(dataset_path)
    records, container_image = _freeze_environment_image(
        environment,
        records,
        image_resolver,
    )
    eval_dataset = stage.get("eval_dataset")
    if eval_dataset is not None:
        if not isinstance(eval_dataset, Mapping) or not eval_dataset.get("path"):
            raise TrainingRuntimeError(
                "resolved GRPO recipe has invalid eval dataset evidence"
            )
        verify_dataset_identity(eval_dataset)
        eval_records = _json_records(Path(str(eval_dataset["path"])))
        _validate_environment_rows(
            eval_records,
            backend=container_image["backend"],
            image=container_image["reference"],
            dataset_name="grpo.eval_dataset",
        )
    active_factory = factory or import_factory(str(environment.get("factory", "")))
    task_results = [
        _run_task_canary(
            row,
            str(row.get("task_id") or f"record {position}"),
            active_factory,
        )
        for position, row in records
    ]
    first = task_results[0]
    return {
        "status": "ready",
        "dataset_path": str(dataset_path),
        "task_count": len(task_results),
        "tasks": task_results,
        "container_image": container_image,
        # Keep the first-row summary for existing GUI clients while the full
        # evidence list makes readiness truthful for every compiled task.
        **first,
    }


__all__ = ["run_grpo_environment_canary"]
