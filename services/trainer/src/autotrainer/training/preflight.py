"""Executable GRPO task preflight shared by Prepare and real training.

Static manifest checks cannot prove that the pinned container can install the
repository, execute its gates, or produce a valid hidden-verifier report. This
module runs one deterministic no-edit baseline episode before model loading.
It is intentionally not a model rollout and makes no claim about reward
variance across sampled completions.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from math import isfinite
from numbers import Real
from pathlib import Path
from typing import Any

from .common import TrainingRuntimeError, _first_json_record, import_factory


_REQUIRED_CHECKS = ("build", "tests", "verifier")
_OPTIONAL_CHECKS = ("install", "browserTests")


def _check_detail(check: Any) -> str:
    detail = str(getattr(check, "stderr", "") or getattr(check, "stdout", "")).strip()
    return f": {detail[:500]}" if detail else ""


def run_grpo_environment_canary(
    recipe: Mapping[str, Any],
    *,
    factory: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    """Execute a deterministic no-edit task through the real RL environment.

    The first sorted compiled row is the canary used by both Prepare and direct
    GRPO execution. A successful canary proves the selected source snapshot,
    container image, install/build/regression commands, and hidden verifier can
    run together. It also rejects a task whose starting state already satisfies
    every task-specific check, since that row offers no positive task signal.

    Args:
        recipe: Fully resolved GRPO recipe.
        factory: Optional environment factory injection used by unit tests.

    Returns:
        JSON-compatible evidence describing the exercised task and baseline.

    Raises:
        TrainingRuntimeError: If the executable task contract is unavailable,
            any runtime gate fails, the verifier report is invalid, or the
            baseline task checks are already saturated.
    """

    stage = recipe.get("grpo")
    environment = recipe.get("environment")
    if not isinstance(stage, Mapping) or not isinstance(environment, Mapping):
        raise TrainingRuntimeError("resolved GRPO recipe is missing stage or environment data")
    dataset = stage.get("dataset")
    if not isinstance(dataset, Mapping) or not dataset.get("path"):
        raise TrainingRuntimeError("resolved GRPO recipe is missing its dataset path")

    dataset_path = Path(str(dataset["path"]))
    row = dict(_first_json_record(dataset_path))
    task_id = str(row.get("task_id") or "first compiled task")
    active_factory = factory or import_factory(str(environment.get("factory", "")))

    try:
        instance = active_factory()
        reset = getattr(instance, "reset", None)
        get_reward = getattr(instance, "get_reward", None)
        if not callable(reset) or not callable(get_reward):
            raise TrainingRuntimeError(
                "GRPO environment must expose callable reset and get_reward methods"
            )
        observation = reset(**row)
        if not isinstance(observation, str) or not observation.strip():
            raise TrainingRuntimeError("GRPO environment reset must return a text observation")
        # TRL appends reset output directly to the last prompt message. Require
        # a visible boundary so custom environments cannot glue observation
        # text onto the operator's instruction or repeat it ambiguously.
        if not observation.startswith("\n\n"):
            raise TrainingRuntimeError(
                "GRPO environment reset observation must begin with a blank-line separator"
            )
        reward = get_reward()
    except TrainingRuntimeError:
        raise
    except Exception as error:
        raise TrainingRuntimeError(
            f"GRPO executable task canary {task_id!r} failed: {error}"
        ) from error

    if isinstance(reward, bool) or not isinstance(reward, Real) or not isfinite(float(reward)):
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
    task_pass_rate = float(raw_rates["task_tests"])
    if not isfinite(task_pass_rate) or not 0.0 <= task_pass_rate <= 1.0:
        raise TrainingRuntimeError(
            f"GRPO executable task canary {task_id!r} produced an invalid task-test signal"
        )
    if task_pass_rate >= 1.0:
        raise TrainingRuntimeError(
            f"GRPO executable task canary {task_id!r} already passes every task test at "
            "the starting revision. Tighten the verifier or choose an unsolved task."
        )

    return {
        "status": "ready",
        "task_id": task_id,
        "dataset_path": str(dataset_path),
        "baseline_reward": round(float(reward), 6),
        "task_pass_rate": task_pass_rate,
        "signal": "unsaturated_baseline",
        "checks": {
            name: str(getattr(check, "status", "unknown"))
            for name, check in sorted(checks.items())
        },
    }


__all__ = ["run_grpo_environment_canary"]
