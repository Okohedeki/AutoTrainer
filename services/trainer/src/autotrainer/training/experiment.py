"""Dependency-free timing and durable receipts for refinement experiments."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Any, Callable, Mapping
from uuid import uuid4

from .common import as_serializable


RECEIPT_SCHEMA_VERSION = 1
_PHASE_NAME = re.compile(r"[a-z][a-z0-9_]{0,63}")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class PhaseProfiler:
    """Record non-overlapping coarse wall-time phases with negligible overhead."""

    def __init__(self, *, clock: Callable[[], float] = time.perf_counter) -> None:
        self._clock = clock
        self._started = clock()
        self._previous = self._started
        self._phases: dict[str, float] = {}

    def checkpoint(self, name: str) -> float:
        if _PHASE_NAME.fullmatch(name) is None:
            raise ValueError(f"invalid training profile phase: {name!r}")
        observed = self._clock()
        elapsed = max(0.0, observed - self._previous)
        self._previous = observed
        self._phases[name] = round(self._phases.get(name, 0.0) + elapsed, 6)
        return elapsed

    def summary(self) -> dict[str, Any]:
        total = max(0.0, self._clock() - self._started)
        return {
            "clock": "monotonic_wall_time",
            "phase_seconds": dict(self._phases),
            "total_seconds": round(total, 6),
        }


def build_training_receipt(
    *,
    stage: str,
    recipe: Mapping[str, Any],
    dependencies: Mapping[str, str],
    runtime: Mapping[str, Any],
    trainable_adapter_parameters: int,
    metrics: Mapping[str, Any],
    telemetry: Mapping[str, Any],
    profile: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a token-free receipt from already validated stage evidence."""

    if stage not in {"sft", "grpo"}:
        raise ValueError("training receipt stage must be sft or grpo")
    normalized = as_serializable(
        {
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "status": "completed",
            "stage": stage,
            "completed_at": _now(),
            "recipe": dict(recipe),
            "dependencies": dict(sorted(dependencies.items())),
            "runtime": dict(runtime),
            "trainable_adapter_parameters": trainable_adapter_parameters,
            "metrics": dict(metrics),
            "telemetry": dict(telemetry),
            "profile": dict(profile),
        }
    )
    # Receipts are benchmark inputs. Reject NaN/Infinity instead of relying on
    # the JSON encoder's non-standard representations.
    def validate(value: Any) -> None:
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("training receipt contains a non-finite number")
        if isinstance(value, Mapping):
            for item in value.values():
                validate(item)
        elif isinstance(value, list):
            for item in value:
                validate(item)

    validate(normalized)
    return normalized


def write_training_receipt(destination: Path, receipt: Mapping[str, Any]) -> Path:
    """Atomically publish one immutable stage receipt beside its adapter."""

    output = Path(destination) / "training_receipt.json"
    temporary = output.with_name(f".{output.name}.{os.getpid()}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(
                dict(receipt),
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


__all__ = [
    "PhaseProfiler",
    "RECEIPT_SCHEMA_VERSION",
    "build_training_receipt",
    "write_training_receipt",
]
