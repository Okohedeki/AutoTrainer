"""Small, dependency-free adapters for observed trainer telemetry.

The training modules import this file before PyTorch or Transformers exist, so
dry runs retain their lightweight validation boundary.  The callback adapter
accepts Transformers' base callback class at runtime instead of importing it
here.
"""

from __future__ import annotations

import math
import re
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
) -> object:
    """Create a real TrainerCallback without importing Transformers eagerly."""

    class ObservedLogCallback(base_class):
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
            if torch_module is not None and vram_limit_gib is not None:
                try:
                    metrics.update(
                        {
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
                    )
                except (AttributeError, RuntimeError, TypeError, ValueError):
                    # The hard policy is enforced by CUDA's allocator. Missing
                    # counters affect observability, not the installed cap.
                    pass
            if not metrics:
                return control
            event: dict[str, Any] = {
                "type": "trainer_log",
                "stage": stage,
                "metrics": metrics,
            }
            step = getattr(state, "global_step", None)
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

    return ObservedLogCallback()


__all__ = ["TrainingEventCallback", "make_trainer_log_callback", "numeric_metrics"]
