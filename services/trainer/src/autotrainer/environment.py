"""Deterministic reward calculation for frontend RL episodes."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


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


def score_rollout(report: RolloutVerifierReport) -> RewardBreakdown:
    """Convert verifier signals into the scalar reward consumed by GRPO."""

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
        name: round(value * WEIGHTS[name], 4)
        for name, value in rates.items()
    }

    if not report.build_passed:
        return RewardBreakdown(True, "build_failed", 0.0, signals)
    if report.regression_pass_rate < 1:
        return RewardBreakdown(True, "regression_failed", 0.0, signals)

    return RewardBreakdown(False, None, round(sum(signals.values()), 4), signals)
