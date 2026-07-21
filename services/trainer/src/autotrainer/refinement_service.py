"""User-owned adapter-refinement and VRAM budget settings."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from pathlib import Path
from typing import Any

from .config import (
    ConfigError,
    load_config,
    project_config_mutation,
    validate_mapping,
    write_config,
)
from .models import minimum_vram_gib, vram_budget_error


DEFAULT_VRAM_GIB = 20.0


def _model_id(data: Mapping[str, Any]) -> str:
    model = data.get("model", {})
    return str(model.get("id", "")).strip() if isinstance(model, Mapping) else ""


def refinement_vram_error(data: Mapping[str, Any]) -> str | None:
    """Return a model-aware budget blocker without rejecting legacy YAML reads."""

    refinement = data.get("refinement", {})
    vram = refinement.get("vram", {}) if isinstance(refinement, Mapping) else {}
    configured = (
        vram.get("max_gib", DEFAULT_VRAM_GIB)
        if isinstance(vram, Mapping)
        else DEFAULT_VRAM_GIB
    )
    if isinstance(configured, bool) or not isinstance(configured, (int, float)):
        return "refinement.vram.max_gib must be a number"
    limit = float(configured)
    if not isfinite(limit) or not 4 <= limit <= 192:
        return "refinement.vram.max_gib must be a finite number between 4 and 192"
    return vram_budget_error(_model_id(data), limit)


def get_refinement_settings(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    section = config.data.get("refinement", {})
    vram = section.get("vram", {}) if isinstance(section, dict) else {}
    model_id = _model_id(config.data)
    return {
        "mode": "adapter_only",
        "model_id": model_id,
        "minimum_vram_gib": minimum_vram_gib(model_id),
        "vram": {
            "max_gib": float(vram.get("max_gib", DEFAULT_VRAM_GIB)),
            "enforcement": str(vram.get("enforcement", "hard")),
        },
    }


def set_refinement_settings(
    config_path: str | Path,
    *,
    max_vram_gib: float,
    enforcement: str,
) -> dict[str, Any]:
    if isinstance(max_vram_gib, bool) or not 4 <= float(max_vram_gib) <= 192:
        raise ConfigError("max_vram_gib must be between 4 and 192")
    selected = str(enforcement).strip().casefold()
    if selected not in {"hard", "soft"}:
        raise ConfigError("enforcement must be hard or soft")
    with project_config_mutation(config_path):
        config = load_config(config_path)
        budget_error = vram_budget_error(_model_id(config.data), float(max_vram_gib))
        if budget_error is not None:
            raise ConfigError(budget_error)
        updated = dict(config.data)
        updated["refinement"] = {
            "mode": "adapter_only",
            "vram": {
                "max_gib": float(max_vram_gib),
                "enforcement": selected,
            },
        }
        report = validate_mapping(updated, root=config.root)
        if report.errors:
            raise ConfigError("\n".join(report.errors))
        write_config(config.path, updated, overwrite=True)
    return get_refinement_settings(config_path)


__all__ = [
    "get_refinement_settings",
    "refinement_vram_error",
    "set_refinement_settings",
]
