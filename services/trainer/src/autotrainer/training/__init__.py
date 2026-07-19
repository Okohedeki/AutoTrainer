"""Public training recipe and execution APIs for AutoTrainer."""

from .common import (
    REFERENCE_DEPENDENCIES,
    SUPPORTED_MODEL_CLASS,
    SUPPORTED_MODEL_ID,
    SUPPORTED_SCHEMA_VERSION,
    TrainingConfigurationError,
    TrainingDependencyError,
    TrainingRuntimeError,
)
from .grpo import resolve_grpo_recipe, run_grpo
from .preflight import run_grpo_environment_canary
from .sft import resolve_sft_recipe, run_sft

__all__ = [
    "REFERENCE_DEPENDENCIES",
    "SUPPORTED_MODEL_CLASS",
    "SUPPORTED_MODEL_ID",
    "SUPPORTED_SCHEMA_VERSION",
    "TrainingConfigurationError",
    "TrainingDependencyError",
    "TrainingRuntimeError",
    "resolve_grpo_recipe",
    "resolve_sft_recipe",
    "run_grpo_environment_canary",
    "run_grpo",
    "run_sft",
]
