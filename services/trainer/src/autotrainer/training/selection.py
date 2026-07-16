"""Select the data-driven training stages without mutating project YAML.

The GUI preparation path and the agent-facing training runner both call this
module.  Keeping the conditional stage rewrite in one place prevents Prepare
from validating a different recipe than the one Start will execute.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from .common import TrainingConfigurationError


TRAINING_RECIPES = {"teach", "practice", "both"}


def select_stage_config(config: Mapping[str, Any], recipe: str) -> dict[str, Any]:
    """Return the inferred stage configuration for one prepared-data recipe."""

    data = deepcopy(dict(config))
    sft = data.setdefault("sft", {})
    grpo = data.setdefault("grpo", {})
    if not isinstance(sft, dict) or not isinstance(grpo, dict):
        raise TrainingConfigurationError("The SFT and GRPO project sections must be mappings.")

    if recipe == "teach":
        sft["enabled"] = True
        grpo["enabled"] = False
    elif recipe == "practice":
        # A default both-stage project points GRPO at the not-yet-created SFT
        # output. Tasks-only practice instead creates a fresh QLoRA policy.
        original_sft_enabled = sft.get("enabled", True) is not False
        selected = grpo.get("start_from", grpo.get("sft_adapter"))
        if original_sft_enabled or not selected:
            selected = "base"
        sft["enabled"] = False
        grpo["enabled"] = True
        grpo["start_from"] = selected
        grpo.pop("sft_adapter", None)
    elif recipe == "both":
        output = sft.get("output_dir")
        if not isinstance(output, str) or not output.strip():
            raise TrainingConfigurationError("Both-stage training requires sft.output_dir.")
        sft["enabled"] = True
        grpo["enabled"] = True
        grpo["start_from"] = output
        grpo.pop("sft_adapter", None)
    else:
        raise TrainingConfigurationError("Prepared data does not select a training recipe.")
    return data


__all__ = ["TRAINING_RECIPES", "select_stage_config"]
