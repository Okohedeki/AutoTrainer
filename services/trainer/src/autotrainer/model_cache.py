"""Materialize immutable Hugging Face model snapshots for offline training.

The YAML remains the model declaration, while the Hugging Face cache stores the
large shared blobs. A small project-local receipt records which exact snapshot
AutoTrainer finished downloading; tokens are read from the process environment
or the normal Hugging Face login and are never written to project artifacts.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping

from .config import ConfigError, load_config, project_config_mutation, write_config
from .locking import _resolve_huggingface_revision
from .project_gate import project_mutation_gate


IMMUTABLE_REVISION = re.compile(r"^[0-9a-fA-F]{40,64}$")


class ModelCacheError(RuntimeError):
    """Raised when an immutable model snapshot cannot be verified or downloaded."""


def _hub_functions() -> tuple[Any, Any]:
    """Import Hub functionality lazily so static CLI commands stay lightweight."""

    try:
        from huggingface_hub import scan_cache_dir, snapshot_download
    except (ImportError, OSError) as error:
        raise ModelCacheError(
            "Model downloads require huggingface-hub. Install the AutoTrainer "
            "package before using autotrainer model download."
        ) from error
    return snapshot_download, scan_cache_dir


def _receipt_path(artifact_dir: Path) -> Path:
    return artifact_dir / "models" / "current.json"


def _configured_cache_dir(config: Any) -> Path:
    """Resolve the cache in the same project/runtime context that will train."""

    value = config.model.get("cache_dir", ".autotrainer/model-cache")
    return config.resolve_path(str(value))


def _read_receipt(artifact_dir: Path) -> dict[str, Any] | None:
    path = _receipt_path(artifact_dir)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return payload if isinstance(payload, dict) else None


def _write_receipt(artifact_dir: Path, payload: Mapping[str, Any]) -> Path:
    destination = _receipt_path(artifact_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def _snapshot_size(snapshot_path: Path) -> tuple[int, int]:
    """Return logical file count and bytes for a completed cached snapshot."""

    file_count = 0
    logical_bytes = 0
    for item in snapshot_path.rglob("*"):
        if not item.is_file():
            continue
        file_count += 1
        try:
            logical_bytes += item.stat().st_size
        except OSError:
            # The receipt remains useful even when an external cache cleanup
            # races with one optional cached link.
            continue
    return file_count, logical_bytes


def inspect_model_cache(config_path: str | Path) -> dict[str, Any]:
    """Describe whether the YAML's exact model revision is available offline."""

    config = load_config(config_path)
    model_id = str(config.model.get("id", "")).strip()
    revision = str(config.model.get("revision", "")).strip()
    result: dict[str, Any] = {
        "model_id": model_id,
        "revision": revision,
        "immutable": bool(IMMUTABLE_REVISION.fullmatch(revision)),
        "status": "not_downloaded",
        "snapshot_path": None,
        "receipt": None,
        "hf_token_configured": bool(os.environ.get("HF_TOKEN")),
        "cache_dir": str(_configured_cache_dir(config)),
    }
    if not result["immutable"]:
        result["status"] = "revision_unresolved"
        return result

    configured_cache_dir = _configured_cache_dir(config).resolve()
    receipt = _read_receipt(config.artifact_dir)
    receipt_cache_dir: Path | None = None
    if receipt and str(receipt.get("cache_dir", "")).strip():
        receipt_cache_dir = Path(str(receipt["cache_dir"])).expanduser().resolve()
    if (
        receipt
        and receipt.get("model_id") == model_id
        and receipt.get("revision") == revision.lower()
        # A receipt proves only the cache it was written for. If the user
        # relocates model.cache_dir, the old snapshot must not make the new
        # empty folder look downloaded.
        and receipt_cache_dir == configured_cache_dir
        and Path(str(receipt.get("snapshot_path", ""))).is_dir()
    ):
        result.update(
            status="downloaded",
            snapshot_path=receipt["snapshot_path"],
            receipt=str(_receipt_path(config.artifact_dir)),
            file_count=receipt.get("file_count"),
            logical_bytes=receipt.get("logical_bytes"),
        )
        return result

    # An exact snapshot may already exist in the shared cache because another
    # project downloaded it. local_files_only guarantees no network request.
    try:
        snapshot_download, _scan_cache_dir = _hub_functions()
        snapshot = Path(
            snapshot_download(
                repo_id=model_id,
                revision=revision,
                cache_dir=_configured_cache_dir(config),
                local_files_only=True,
            )
        ).resolve()
    except ModelCacheError:
        result["status"] = "dependency_missing"
        return result
    except Exception:
        return result

    file_count, logical_bytes = _snapshot_size(snapshot)
    result.update(
        status="cached_unverified",
        snapshot_path=str(snapshot),
        file_count=file_count,
        logical_bytes=logical_bytes,
    )
    return result


def materialize_model_owned(config_path: str | Path) -> dict[str, Any]:
    """Download one immutable snapshot while the caller owns the project lease.

    The synchronous CLI uses :func:`materialize_model` below.  The human API
    reserves the lease before queueing a background worker and calls this
    narrower entry point so the browser can disconnect without releasing the
    model or configuration boundary mid-download.
    """

    config = load_config(config_path)
    model_id = str(config.model.get("id", "")).strip()
    requested_revision = str(config.model.get("revision", "")).strip()
    if not model_id or not requested_revision:
        raise ConfigError("model.id and model.revision are required before download")

    try:
        resolved_revision = _resolve_huggingface_revision(model_id, requested_revision)
    except RuntimeError as error:
        raise ModelCacheError(str(error)) from error

    snapshot_download, _scan_cache_dir = _hub_functions()
    token = os.environ.get("HF_TOKEN") or None
    cache_dir = _configured_cache_dir(config)
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        snapshot = Path(
            snapshot_download(
                repo_id=model_id,
                revision=resolved_revision,
                cache_dir=cache_dir,
                token=token,
            )
        ).resolve()
    except Exception as error:
        raise ModelCacheError(
            f"cannot download {model_id}@{resolved_revision}; check disk space, "
            "network access, model access, and Hugging Face login: "
            f"{error}"
        ) from error

    file_count, logical_bytes = _snapshot_size(snapshot)
    with project_config_mutation(config.path):
        # Re-read even though the project lease is held. This keeps the commit
        # merge-safe with direct callers using the lower-level config helper.
        current = load_config(config.path)
        current_model = current.data["model"]
        current_id = str(current_model.get("id", "")).strip()
        current_revision = str(current_model.get("revision", "")).strip()
        current_cache_dir = _configured_cache_dir(current).resolve()
        if current_id != model_id or current_revision not in {
            requested_revision,
            resolved_revision,
        }:
            raise ModelCacheError(
                "The selected model changed while its snapshot downloaded; "
                "the new selection was preserved. Start its download when ready."
            )
        if current_cache_dir != cache_dir.resolve():
            raise ModelCacheError(
                "The model cache folder changed while its snapshot downloaded; "
                "the new folder was preserved. Start the download again."
            )
        current_model["revision"] = resolved_revision
        write_config(current.path, current.data, overwrite=True)
        receipt = {
            "schema_version": 1,
            "model_id": model_id,
            "requested_revision": requested_revision,
            "revision": resolved_revision,
            "snapshot_path": str(snapshot),
            "cache_dir": str(cache_dir),
            "file_count": file_count,
            "logical_bytes": logical_bytes,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        receipt_path = _write_receipt(current.artifact_dir, receipt)
    return {
        "status": "downloaded",
        **receipt,
        "receipt": str(receipt_path),
    }


def materialize_model(config_path: str | Path) -> dict[str, Any]:
    """Hold the project lease for the complete model download and commit."""

    with project_mutation_gate(config_path):
        return materialize_model_owned(config_path)


def require_materialized_model(model: Mapping[str, Any]) -> None:
    """Reject mutable or unavailable model snapshots before heavy ML imports."""

    model_id = str(model.get("id", "")).strip()
    revision = str(model.get("revision", "")).strip()
    if not IMMUTABLE_REVISION.fullmatch(revision):
        raise ModelCacheError(
            "Real training requires an immutable downloaded model revision. Run "
            "autotrainer model download first; dry runs remain available."
        )
    cache_dir = Path(str(model.get("cache_dir", ""))).expanduser()
    if not model_id or not cache_dir.is_absolute():
        raise ModelCacheError("The resolved model recipe must contain an absolute cache directory.")
    snapshot_download, _scan_cache_dir = _hub_functions()
    try:
        snapshot_download(
            repo_id=model_id,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=True,
        )
    except Exception as error:
        raise ModelCacheError(
            f"The exact model snapshot {model_id}@{revision} is not complete in "
            f"{cache_dir}. Run autotrainer model download before training."
        ) from error


__all__ = [
    "ModelCacheError",
    "inspect_model_cache",
    "materialize_model",
    "materialize_model_owned",
    "require_materialized_model",
]
