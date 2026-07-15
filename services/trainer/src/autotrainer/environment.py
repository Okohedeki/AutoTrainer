"""Deterministic reward calculation for frontend RL episodes."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any


@dataclass(frozen=True, slots=True)
class RolloutVerifierReport:
    build_passed: bool
    regression_pass_rate: float
    task_pass_rate: float
    responsive_pass_rate: float
    design_rule_pass_rate: float
    code_quality_pass_rate: float


@dataclass(frozen=True, slots=True)
class RewardBreakdown:
    gated: bool
    gate_reason: str | None
    total: float
    signals: dict[str, float]


@dataclass(frozen=True, slots=True)
class CheckResult:
    """Structured outcome for one trusted evaluation command."""

    name: str
    configured: bool
    status: str
    passed: bool | None
    returncode: int | None
    timed_out: bool
    duration_seconds: float
    stdout: str
    stderr: str

    def to_mapping(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""

        return {
            "name": self.name,
            "configured": self.configured,
            "status": self.status,
            "passed": self.passed,
            "returncode": self.returncode,
            "timed_out": self.timed_out,
            "duration_seconds": self.duration_seconds,
            "stdout": self.stdout,
            "stderr": self.stderr,
        }


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    """Complete, evaluation-safe record captured before workspace cleanup."""

    task_id: str
    hard_gate_reason: str | None
    raw_verifier_rates: dict[str, float]
    weighted_signals: dict[str, float]
    reward: float
    checks: dict[str, CheckResult]
    tool_call_count: int
    elapsed_seconds: float
    unified_diff: str

    @property
    def gated(self) -> bool:
        """Whether a hard gate prevented a positive evaluation outcome."""

        return self.hard_gate_reason is not None

    def to_mapping(self) -> dict[str, Any]:
        """Return a stable JSON-compatible representation."""

        return {
            "task_id": self.task_id,
            "gated": self.gated,
            "hard_gate_reason": self.hard_gate_reason,
            "raw_verifier_rates": dict(self.raw_verifier_rates),
            "weighted_signals": dict(self.weighted_signals),
            "reward": self.reward,
            "checks": {
                name: result.to_mapping() for name, result in sorted(self.checks.items())
            },
            "tool_call_count": self.tool_call_count,
            "elapsed_seconds": self.elapsed_seconds,
            "unified_diff": self.unified_diff,
        }


WEIGHTS = {
    "regression_safety": 0.20,
    "task_tests": 0.35,
    "responsive_rules": 0.20,
    "design_rules": 0.15,
    "patch_quality": 0.10,
}


def _validate_rate(name: str, value: float) -> None:
    if not isfinite(value) or value < 0 or value > 1:
        raise ValueError(f"{name} must be between 0 and 1")


def score_rollout(
    report: RolloutVerifierReport,
    weights: dict[str, float] | None = None,
) -> RewardBreakdown:
    """Convert verifier signals into the scalar reward consumed by GRPO."""

    active_weights = WEIGHTS if weights is None else weights
    if set(active_weights) != set(WEIGHTS):
        raise ValueError(f"weights must contain exactly: {', '.join(sorted(WEIGHTS))}")
    if any(not isfinite(value) or value < 0 or value > 1 for value in active_weights.values()):
        raise ValueError("reward weights must be finite and between 0 and 1")
    if abs(sum(active_weights.values()) - 1) > 1e-8:
        raise ValueError("reward weights must sum to 1")

    rates = {
        "regression_safety": report.regression_pass_rate,
        "task_tests": report.task_pass_rate,
        "responsive_rules": report.responsive_pass_rate,
        "design_rules": report.design_rule_pass_rate,
        "patch_quality": report.code_quality_pass_rate,
    }
    for name, value in rates.items():
        _validate_rate(name, value)

    signals = {
        name: round(value * active_weights[name], 4)
        for name, value in rates.items()
    }

    if not report.build_passed:
        return RewardBreakdown(True, "build_failed", 0.0, signals)
    if report.regression_pass_rate < 1:
        return RewardBreakdown(True, "regression_failed", 0.0, signals)

    return RewardBreakdown(False, None, round(sum(signals.values()), 4), signals)
