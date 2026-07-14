"""Core contracts for the AutoTrainer single-GPU training runtime."""

from .environment import RewardBreakdown, RolloutVerifierReport, score_rollout
from .manifest import ManifestError, TaskManifest

__all__ = [
    "ManifestError",
    "RewardBreakdown",
    "RolloutVerifierReport",
    "TaskManifest",
    "score_rollout",
]
