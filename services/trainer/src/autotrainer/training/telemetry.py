"""Small, dependency-free adapters for observed trainer telemetry.

The training modules import this file before PyTorch or Transformers exist, so
dry runs retain their lightweight validation boundary.  The callback adapter
accepts Transformers' base callback class at runtime instead of importing it
here.
"""

from __future__ import annotations

import math
import re
import time
from typing import Any, Callable, Mapping


TrainingEventCallback = Callable[[Mapping[str, Any]], None]

_METRIC_KEY = re.compile(r"^[A-Za-z0-9_.:/-]{1,80}$")
_SECRET_KEY = re.compile(r"(?i)(?:api[-_]?key|password|secret|token)")
_MAX_METRICS = 48
_MAX_ABSOLUTE_METRIC = 10**100


def numeric_metrics(value: object) -> dict[str, int | float]:
    """Return bounded finite scalar metrics and discard all text payloads."""

    if not isinstance(value, Mapping):
        return {}
    metrics: dict[str, int | float] = {}
    for raw_key, raw_value in value.items():
        key = str(raw_key)
        if (
            len(metrics) >= _MAX_METRICS
            or not _METRIC_KEY.fullmatch(key)
            or _SECRET_KEY.search(key)
            or isinstance(raw_value, bool)
        ):
            continue
        if isinstance(raw_value, int) and abs(raw_value) <= _MAX_ABSOLUTE_METRIC:
            metrics[key] = raw_value
        elif (
            isinstance(raw_value, float)
            and math.isfinite(raw_value)
            and abs(raw_value) <= _MAX_ABSOLUTE_METRIC
        ):
            metrics[key] = raw_value
    return metrics


def make_trainer_log_callback(
    base_class: type,
    *,
    stage: str,
    on_event: TrainingEventCallback | None,
    torch_module: Any | None = None,
    vram_limit_gib: float | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> object:
    """Create a real TrainerCallback without importing Transformers eagerly."""

    class ObservedLogCallback(base_class):
        def __init__(self) -> None:
            self._training_started_at: float | None = None
            self._last_observed_at: float | None = None
            self._last_step = 0
            self._final_summary: dict[str, int | float] = {}

        def _memory_metrics(self) -> dict[str, int | float]:
            if torch_module is None or vram_limit_gib is None:
                return {}
            try:
                values: dict[str, int | float] = {
                    "vram_allocated_gib": round(
                        float(torch_module.cuda.memory_allocated(0)) / 1024**3,
                        3,
                    ),
                    "vram_limit_gib": float(vram_limit_gib),
                    "vram_reserved_gib": round(
                        float(torch_module.cuda.memory_reserved(0)) / 1024**3,
                        3,
                    ),
                }
                max_allocated = getattr(torch_module.cuda, "max_memory_allocated", None)
                max_reserved = getattr(torch_module.cuda, "max_memory_reserved", None)
                if callable(max_allocated):
                    values["vram_peak_allocated_gib"] = round(
                        float(max_allocated(0)) / 1024**3,
                        3,
                    )
                if callable(max_reserved):
                    values["vram_peak_reserved_gib"] = round(
                        float(max_reserved(0)) / 1024**3,
                        3,
                    )
                return values
            except (AttributeError, RuntimeError, TypeError, ValueError):
                # The hard policy is enforced by CUDA's allocator. Missing
                # counters affect observability, not the installed cap.
                return {}

        def on_train_begin(
            self,
            _args: object,
            state: object,
            control: object,
            **_kwargs: object,
        ) -> object:
            observed = clock()
            self._training_started_at = observed
            self._last_observed_at = observed
            step = getattr(state, "global_step", 0)
            self._last_step = step if isinstance(step, int) and not isinstance(step, bool) else 0
            reset_peak = (
                getattr(torch_module.cuda, "reset_peak_memory_stats", None)
                if torch_module is not None
                else None
            )
            if callable(reset_peak):
                try:
                    reset_peak(0)
                except (RuntimeError, TypeError, ValueError):
                    pass
            return control

        # Transformers supplies these values by keyword. Keeping a permissive
        # signature also works with the pinned TRL trainers' callback handler.
        def on_log(
            self,
            _args: object,
            state: object,
            control: object,
            logs: Mapping[str, Any] | None = None,
            **_kwargs: object,
        ) -> object:
            if on_event is None:
                return control
            metrics = numeric_metrics(logs)
            metrics.update(self._memory_metrics())
            step = getattr(state, "global_step", None)
            if (
                self._last_observed_at is not None
                and isinstance(step, int)
                and not isinstance(step, bool)
                and step >= self._last_step
            ):
                observed = clock()
                elapsed = max(0.0, observed - self._last_observed_at)
                step_delta = step - self._last_step
                if elapsed > 0 and step_delta > 0:
                    metrics["observed_window_seconds"] = round(elapsed, 6)
                    metrics["observed_steps_per_second"] = round(step_delta / elapsed, 6)
                    metrics["observed_seconds_per_step"] = round(elapsed / step_delta, 6)
                self._last_observed_at = observed
                self._last_step = step
            if not metrics:
                return control
            event: dict[str, Any] = {
                "type": "trainer_log",
                "stage": stage,
                "metrics": metrics,
            }
            if (
                isinstance(step, int)
                and not isinstance(step, bool)
                and 0 <= step <= 10**15
            ):
                event["step"] = step
            epoch = getattr(state, "epoch", None)
            if isinstance(epoch, (int, float)) and not isinstance(epoch, bool):
                epoch_number = float(epoch)
                if math.isfinite(epoch_number) and 0 <= epoch_number <= 10**15:
                    event["epoch"] = epoch_number
            on_event(event)
            return control

        def on_train_end(
            self,
            _args: object,
            state: object,
            control: object,
            **_kwargs: object,
        ) -> object:
            summary = self._memory_metrics()
            if self._training_started_at is not None:
                summary["observed_train_seconds"] = round(
                    max(0.0, clock() - self._training_started_at),
                    6,
                )
            step = getattr(state, "global_step", None)
            if isinstance(step, int) and not isinstance(step, bool) and step >= 0:
                summary["observed_train_steps"] = step
            self._final_summary = summary
            return control

        def observed_summary(self) -> dict[str, int | float]:
            if self._final_summary:
                return dict(self._final_summary)
            summary = self._memory_metrics()
            if self._training_started_at is not None:
                summary["observed_train_seconds"] = round(
                    max(0.0, clock() - self._training_started_at),
                    6,
                )
                summary["observed_train_steps"] = self._last_step
            return summary

    return ObservedLogCallback()


__all__ = ["TrainingEventCallback", "make_trainer_log_callback", "numeric_metrics"]
