"""Shared model lifecycle operations for the GUI and agent-facing CLI.

This module is the product boundary for model selection and materialization.
The CLI and local HTTP API call these functions directly so model policy,
configuration writes, and cache semantics cannot drift between interfaces.
"""

from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path
import re
from typing import Any

from .config import ConfigError, load_config, project_config_mutation, write_config
from .model_cache import (
    adopt_local_model as _adopt_local_model,
    discover_local_models as _discover_local_models,
    inspect_model_cache,
    materialize_model,
)
from .models import MODEL_CATALOG, resolve_model
from .project_gate import project_mutation_gate


_IMMUTABLE_REVISION = re.compile(r"^[0-9a-fA-F]{40,64}$")
_MIN_SEARCH_LENGTH = 2
_MAX_SEARCH_LENGTH = 100
_DEFAULT_SEARCH_LIMIT = 12
_MAX_SEARCH_LIMIT = 25


class ModelSearchError(RuntimeError):
    """Raised with a stable message when remote model discovery is unavailable."""


def _catalog_match(value: str) -> tuple[str | None, dict[str, Any]]:
    """Resolve aliases and exact catalog IDs through the same policy path."""

    if value in MODEL_CATALOG:
        return value, dict(MODEL_CATALOG[value])
    for alias, details in MODEL_CATALOG.items():
        if details.get("id") == value:
            return alias, dict(details)
    return None, resolve_model(value)


def _hub_api() -> Any:
    """Construct the Hub client lazily so offline commands stay lightweight."""

    try:
        from huggingface_hub import HfApi
    except (ImportError, OSError) as error:
        raise ModelSearchError(
            "Hugging Face model search requires the installed AutoTrainer package."
        ) from error
    # Authentication remains process-local. Search results are normalized
    # below and can never contain this token.
    return HfApi(token=os.environ.get("HF_TOKEN") or None)


def _field(record: object, name: str, default: object = None) -> object:
    if isinstance(record, Mapping):
        return record.get(name, default)
    return getattr(record, name, default)


def _nonnegative_integer(value: object) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _search_record(record: object) -> dict[str, Any] | None:
    """Reduce a Hub record to bounded, display-safe compatibility evidence."""

    model_id = str(_field(record, "id", _field(record, "modelId", ""))).strip()
    if not model_id:
        return None
    raw_revision = str(_field(record, "sha", "")).strip()
    revision = (
        raw_revision.lower()
        if _IMMUTABLE_REVISION.fullmatch(raw_revision)
        else None
    )
    catalog_key, details = _catalog_match(model_id)
    if catalog_key and details.get("trainable_v1") is True:
        compatibility = "supported"
        reason = "Validated by AutoTrainer's single-GPU V1 profile."
    elif catalog_key and details.get("purpose") == "benchmark_reference":
        compatibility = "reference_only"
        reason = "Pinned for evaluation; not a validated V1 training base."
    else:
        compatibility = "unverified"
        reason = "Architecture, chat template, license, and GPU fit are not validated for V1."

    pipeline_tag = _field(record, "pipeline_tag")
    library_name = _field(record, "library_name")
    return {
        "id": model_id,
        # Never substitute `main`: a missing SHA remains visibly unresolved.
        "revision": revision,
        "pipeline_tag": str(pipeline_tag) if pipeline_tag else None,
        "library_name": str(library_name) if library_name else None,
        "downloads": _nonnegative_integer(_field(record, "downloads", 0)),
        "likes": _nonnegative_integer(_field(record, "likes", 0)),
        "gated": bool(_field(record, "gated", False)),
        "private": bool(_field(record, "private", False)),
        "compatibility": compatibility,
        "catalog_key": catalog_key,
        "trainable_v1": compatibility == "supported",
        "reason": reason,
    }


def search_models(
    query: str,
    *,
    limit: int = _DEFAULT_SEARCH_LIMIT,
) -> list[dict[str, Any]]:
    """Search Hugging Face while keeping trainability an explicit local claim."""

    if not isinstance(query, str):
        raise ConfigError("model search query must be text")
    normalized_query = query.strip()
    if not _MIN_SEARCH_LENGTH <= len(normalized_query) <= _MAX_SEARCH_LENGTH:
        raise ConfigError(
            f"model search query must be {_MIN_SEARCH_LENGTH}-{_MAX_SEARCH_LENGTH} characters"
        )
    if (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or not 1 <= limit <= _MAX_SEARCH_LIMIT
    ):
        raise ConfigError(f"model search limit must be between 1 and {_MAX_SEARCH_LIMIT}")

    try:
        # `full=True` asks the Hub for the current immutable commit SHA along
        # with the small metadata needed by the picker. No model files are read.
        records = list(
            _hub_api().list_models(
                search=normalized_query,
                sort="downloads",
                direction=-1,
                limit=limit,
                full=True,
            )
        )
    except ModelSearchError:
        raise
    except Exception as error:
        # Upstream exception text can include local paths, endpoints, or auth
        # details. Keep it in the exception chain, never the public message.
        raise ModelSearchError(
            "Hugging Face model search is unavailable; check network access and login."
        ) from error

    normalized: list[dict[str, Any]] = []
    for record in records[:limit]:
        item = _search_record(record)
        if item is not None:
            normalized.append(item)
    return normalized


def list_models() -> dict[str, dict[str, Any]]:
    """Return detached catalogue records safe for JSON or CLI serialization."""

    return {alias: dict(details) for alias, details in MODEL_CATALOG.items()}


def discover_local_models(config_path: str | Path) -> dict[str, Any]:
    """Discover supported local snapshots through the shared offline policy."""

    return _discover_local_models(config_path)


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

    catalog_match, resolved = _catalog_match(model_name)
    if catalog_match is not None and not resolved.get("trainable_v1", False):
        raise ConfigError(
            f"{catalog_match} is a {resolved.get('purpose', 'reference')} model, "
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
        # Preserve the existing response contract: direct Hub IDs remain
        # direct selections even when they match a known catalog profile.
        "catalog_key": model_name if model_name in MODEL_CATALOG else None,
        "catalog": resolved,
    }


def model_status(config_path: str | Path) -> dict[str, Any]:
    """Inspect the exact configured snapshot without making a network request."""

    return inspect_model_cache(config_path)


def download_model(config_path: str | Path) -> dict[str, Any]:
    """Resolve and materialize the exact configured Hugging Face snapshot."""

    return materialize_model(config_path)


def use_local_model(config_path: str | Path, candidate_id: str) -> dict[str, Any]:
    """Adopt one opaque discovery candidate after server-side revalidation."""

    return _adopt_local_model(config_path, candidate_id)


__all__ = [
    "discover_local_models",
    "download_model",
    "get_model",
    "list_models",
    "model_status",
    "ModelSearchError",
    "search_models",
    "select_model",
    "use_local_model",
]
