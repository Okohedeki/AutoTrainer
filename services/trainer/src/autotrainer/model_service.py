"""Shared model lifecycle operations for the GUI and agent-facing CLI.

This module is the product boundary for model selection and materialization.
The CLI and local HTTP API call these functions directly so model policy,
configuration writes, and cache semantics cannot drift between interfaces.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import ConfigError, load_config, project_config_mutation, write_config
from .model_cache import inspect_model_cache, materialize_model
from .models import MODEL_CATALOG, resolve_model
from .project_gate import project_mutation_gate


def list_models() -> dict[str, dict[str, Any]]:
    """Return detached catalogue records safe for JSON or CLI serialization."""

    return {alias: dict(details) for alias, details in MODEL_CATALOG.items()}


def get_model(config_path: str | Path) -> dict[str, Any]:
    """Return the model declaration currently stored in project YAML."""

    return dict(load_config(config_path).model)


def select_model(
    config_path: str | Path,
    model_name: str,
    *,
    revision: str | None = None,
    cache_dir: str | None = None,
) -> dict[str, Any]:
    """Select a training base and persist the same declaration both clients use."""

    resolved = resolve_model(model_name)
    if model_name in MODEL_CATALOG and not resolved.get("trainable_v1", False):
        raise ConfigError(
            f"{model_name} is a {resolved.get('purpose', 'reference')} model, "
            "not a validated V1 training base"
        )
    selected_revision = (revision or resolved.get("default_revision") or "").strip()
    if not selected_revision:
        raise ConfigError("an explicit model revision is required")

    with project_mutation_gate(config_path):
        with project_config_mutation(config_path):
            # Selection owns only model fields. Re-read while holding the
            # config lock so unrelated simultaneous edits are never replaced.
            config = load_config(config_path)
            model = config.data["model"]
            model["id"] = resolved["id"]
            model["revision"] = selected_revision
            model["loader"] = resolved["loader"]
            if cache_dir is not None:
                model["cache_dir"] = cache_dir
            write_config(config.path, config.data, overwrite=True)
    return {
        "model": dict(model),
        "catalog_key": model_name if model_name in MODEL_CATALOG else None,
        "catalog": resolved,
    }


def model_status(config_path: str | Path) -> dict[str, Any]:
    """Inspect the exact configured snapshot without making a network request."""

    return inspect_model_cache(config_path)


def download_model(config_path: str | Path) -> dict[str, Any]:
    """Resolve and materialize the exact configured Hugging Face snapshot."""

    return materialize_model(config_path)


__all__ = [
    "download_model",
    "get_model",
    "list_models",
    "model_status",
    "select_model",
]
