"""User-owned adapter-refinement and VRAM budget settings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import (
    ConfigError,
    load_config,
    project_config_mutation,
    validate_mapping,
    write_config,
)


DEFAULT_VRAM_GIB = 20.0


def get_refinement_settings(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    section = config.data.get("refinement", {})
    vram = section.get("vram", {}) if isinstance(section, dict) else {}
    return {
        "mode": "adapter_only",
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


__all__ = ["get_refinement_settings", "set_refinement_settings"]
