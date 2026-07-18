"""Materialize immutable Hugging Face model snapshots for offline training.

The YAML remains the model declaration, while the Hugging Face cache stores the
large shared blobs. A small project-local receipt records which exact snapshot
AutoTrainer finished downloading; tokens are read from the process environment
or the normal Hugging Face login and are never written to project artifacts.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from itertools import islice
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import re
from typing import Any, Mapping

from .config import ConfigError, load_config, project_config_mutation, write_config
from .locking import _resolve_huggingface_revision
from .models import MODEL_CATALOG
from .project_gate import project_mutation_gate


IMMUTABLE_REVISION = re.compile(r"^[0-9a-fA-F]{40,64}$")
LOCAL_CANDIDATE_ID = re.compile(r"^[0-9a-f]{64}$")

# Discovery runs on the request path, so every directory and JSON boundary is
# explicit. The catalogue is intentionally tiny in V1, but the limits keep a
# damaged or unexpectedly large cache from monopolizing the local control API.
_MAX_CACHE_ROOTS = 8
_MAX_CATALOG_MODELS = 32
_MAX_REVISIONS_PER_MODEL = 32
_MAX_LOCAL_CANDIDATES = 100
_MAX_INDEX_BYTES = 16 * 1024 * 1024
_MAX_CONFIG_BYTES = 4 * 1024 * 1024
_MAX_INDEX_ENTRIES = 250_000
_MAX_REQUIRED_FILES = 4_096

_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer.model",
    "spiece.model",
    "vocab.json",
    "vocab.txt",
)
_WEIGHT_INDEX_FILES = (
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)
_SINGLE_WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")


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


def _reference_receipt_path(artifact_dir: Path, alias: str) -> Path:
    return artifact_dir / "models" / "references" / f"{alias}.json"


def _configured_cache_dir(config: Any) -> Path:
    """Resolve the cache in the same project/runtime context that will train."""

    value = config.model.get("cache_dir", ".autotrainer/model-cache")
    return config.resolve_path(str(value))


def _expanded_path(value: str) -> Path | None:
    """Normalize one trusted local cache declaration without creating it."""

    if not value or "\x00" in value or len(value) > 4_096:
        return None
    try:
        return Path(os.path.expandvars(value)).expanduser().resolve()
    except (OSError, RuntimeError, ValueError):
        return None


def _cache_root_records(config: Any) -> list[dict[str, Any]]:
    """Return existing configured and standard Hub roots in priority order.

    Environment values are read for every scan instead of relying on constants
    captured when ``huggingface_hub`` was imported. This also makes discovery
    deterministic in tests and keeps it completely offline.
    """

    inputs: list[tuple[Path | None, str, str]] = [
        (_configured_cache_dir(config).resolve(), "project_cache", "Project cache")
    ]
    default_home = Path.home() / ".cache"
    xdg_home = _expanded_path(os.environ.get("XDG_CACHE_HOME", "")) or default_home
    hf_home = _expanded_path(os.environ.get("HF_HOME", "")) or (xdg_home / "huggingface")
    legacy_hub = _expanded_path(os.environ.get("HUGGINGFACE_HUB_CACHE", ""))
    effective_hub = (
        _expanded_path(os.environ.get("HF_HUB_CACHE", ""))
        or legacy_hub
        or (hf_home / "hub").resolve()
    )
    inputs.append((effective_hub, "huggingface_cache", "Hugging Face cache"))
    # If the new variable overrides an explicitly configured legacy cache, the
    # older location can still contain a valid snapshot worth adopting.
    if legacy_hub is not None:
        inputs.append((legacy_hub, "huggingface_cache", "Hugging Face cache"))
    for variable in (
        "TRANSFORMERS_CACHE",
        "PYTORCH_TRANSFORMERS_CACHE",
        "PYTORCH_PRETRAINED_BERT_CACHE",
    ):
        value = _expanded_path(os.environ.get(variable, ""))
        if value is not None:
            inputs.append((value, "transformers_cache", "Transformers cache"))

    roots: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path, source, label in inputs:
        if path is None or len(roots) >= _MAX_CACHE_ROOTS:
            continue
        key = os.path.normcase(str(path))
        if key in seen:
            continue
        seen.add(key)
        try:
            if not path.is_dir():
                continue
        except OSError:
            continue
        roots.append({"path": path, "source": source, "cache_label": label})
    return roots


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _bounded_children(path: Path, limit: int) -> tuple[list[Path], bool]:
    """Read at most ``limit + 1`` children; never sort an unbounded directory."""

    try:
        children = list(islice(path.iterdir(), limit + 1))
    except OSError:
        return [], False
    return children[:limit], len(children) > limit


def _safe_snapshot_file(
    snapshot: Path,
    repo_root: Path,
    relative_name: str,
) -> tuple[Path, int] | None:
    """Resolve one snapshot file while rejecting traversal and escaped links."""

    if not relative_name or "\\" in relative_name or len(relative_name) > 1_024:
        return None
    parts = relative_name.split("/")
    relative = PurePosixPath(relative_name)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in parts):
        return None
    try:
        target = snapshot.joinpath(*relative.parts).resolve(strict=True)
        if not _within(target, repo_root) or not target.is_file():
            return None
        size = target.stat().st_size
    except (OSError, RuntimeError, ValueError):
        return None
    if size <= 0:
        return None
    return target, size


def _bounded_json(
    snapshot: Path,
    repo_root: Path,
    relative_name: str,
    maximum_bytes: int,
) -> tuple[Mapping[str, Any], tuple[Path, int]] | None:
    file_info = _safe_snapshot_file(snapshot, repo_root, relative_name)
    if file_info is None or file_info[1] > maximum_bytes:
        return None
    try:
        value = json.loads(file_info[0].read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(value, Mapping):
        return None
    return value, file_info


def _inspect_local_snapshot(snapshot_path: Path, repo_root: Path) -> dict[str, int] | None:
    """Prove the local files required by the V1 Transformers loader are present.

    This is structural evidence, not a remote repository comparison or tensor
    checksum. No weight bytes are opened. Hub downloads expose only completed
    blobs through snapshot links, while missing/broken links and missing shards
    are rejected here before AutoTrainer writes a receipt.
    """

    try:
        repo = repo_root.resolve(strict=True)
        snapshot = snapshot_path.resolve(strict=True)
    except (OSError, RuntimeError, ValueError):
        return None
    if not snapshot.is_dir() or not _within(snapshot, repo):
        return None

    config_result = _bounded_json(
        snapshot,
        repo,
        "config.json",
        _MAX_CONFIG_BYTES,
    )
    if config_result is None:
        return None
    config, config_file = config_result
    if not isinstance(config.get("model_type"), str) or not str(config["model_type"]).strip():
        return None

    tokenizer_file = next(
        (
            info
            for name in _TOKENIZER_FILES
            if (info := _safe_snapshot_file(snapshot, repo, name)) is not None
        ),
        None,
    )
    if tokenizer_file is None:
        return None

    required: dict[Path, int] = {
        config_file[0]: config_file[1],
        tokenizer_file[0]: tokenizer_file[1],
    }
    found_index = False
    for index_name in _WEIGHT_INDEX_FILES:
        unresolved = snapshot / index_name
        try:
            index_declared = unresolved.exists() or unresolved.is_symlink()
        except OSError:
            index_declared = True
        if not index_declared:
            continue
        found_index = True
        index_result = _bounded_json(snapshot, repo, index_name, _MAX_INDEX_BYTES)
        if index_result is None:
            return None
        index, index_file = index_result
        weight_map = index.get("weight_map")
        if (
            not isinstance(weight_map, Mapping)
            or not weight_map
            or len(weight_map) > _MAX_INDEX_ENTRIES
        ):
            return None
        shard_names: set[str] = set()
        for shard_name in weight_map.values():
            if not isinstance(shard_name, str):
                return None
            shard_names.add(shard_name)
            if len(shard_names) > _MAX_REQUIRED_FILES:
                return None
        required[index_file[0]] = index_file[1]
        for shard_name in shard_names:
            shard = _safe_snapshot_file(snapshot, repo, shard_name)
            if shard is None:
                return None
            required[shard[0]] = shard[1]
        break

    if not found_index:
        weight_file = next(
            (
                info
                for name in _SINGLE_WEIGHT_FILES
                if (info := _safe_snapshot_file(snapshot, repo, name)) is not None
            ),
            None,
        )
        if weight_file is None:
            return None
        required[weight_file[0]] = weight_file[1]

    return {
        "file_count": len(required),
        "logical_bytes": sum(required.values()),
    }


def _candidate_id(cache_root: Path, model_id: str, revision: str) -> str:
    identity = "\x00".join(
        (os.path.normcase(str(cache_root)), model_id, revision.lower())
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _local_model_records(config: Any) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Scan only V1-trainable catalogue repositories across known cache roots."""

    roots = _cache_root_records(config)
    configured_root = _configured_cache_dir(config).resolve()
    selected_id = str(config.model.get("id", "")).strip()
    selected_revision = str(config.model.get("revision", "")).strip().lower()
    candidates: dict[tuple[str, str], dict[str, Any]] = {}
    private_records: dict[str, dict[str, Any]] = {}
    ignored_incomplete = 0
    truncated = False

    catalogue = [
        (alias, dict(profile))
        for alias, profile in MODEL_CATALOG.items()
        if profile.get("trainable_v1") is True
    ]
    if len(catalogue) > _MAX_CATALOG_MODELS:
        catalogue = catalogue[:_MAX_CATALOG_MODELS]
        truncated = True

    for root_record in roots:
        cache_root = root_record["path"]
        for alias, profile in catalogue:
            model_id = str(profile.get("id", "")).strip()
            if not model_id:
                continue
            repo_name = "models--" + model_id.replace("/", "--")
            repo_input = cache_root / repo_name
            try:
                repo_root = repo_input.resolve(strict=True)
            except (FileNotFoundError, OSError, RuntimeError, ValueError):
                continue
            if not repo_root.is_dir() or not _within(repo_root, cache_root):
                continue
            snapshots = repo_root / "snapshots"
            try:
                snapshots_root = snapshots.resolve(strict=True)
            except (FileNotFoundError, OSError, RuntimeError, ValueError):
                continue
            if not snapshots_root.is_dir() or not _within(snapshots_root, repo_root):
                continue
            revisions, revisions_truncated = _bounded_children(
                snapshots_root,
                _MAX_REVISIONS_PER_MODEL,
            )
            truncated = truncated or revisions_truncated
            for revision_input in revisions:
                revision = revision_input.name.lower()
                if IMMUTABLE_REVISION.fullmatch(revision) is None:
                    continue
                try:
                    snapshot = revision_input.resolve(strict=True)
                except (FileNotFoundError, OSError, RuntimeError, ValueError):
                    ignored_incomplete += 1
                    continue
                evidence = _inspect_local_snapshot(snapshot, repo_root)
                if evidence is None:
                    ignored_incomplete += 1
                    continue
                identity = (model_id, revision)
                # Root order is policy: an already configured cache wins over
                # global and legacy locations, avoiding duplicate GUI rows.
                if identity in candidates:
                    continue
                identifier = _candidate_id(cache_root, model_id, revision)
                selected = (
                    selected_id == model_id
                    and selected_revision == revision
                    and os.path.normcase(str(configured_root))
                    == os.path.normcase(str(cache_root))
                )
                public = {
                    "candidate_id": identifier,
                    "catalog_key": alias,
                    "model_id": model_id,
                    "revision": revision,
                    "availability": "available",
                    "selected": selected,
                    "source": root_record["source"],
                    "cache_label": root_record["cache_label"],
                    **evidence,
                }
                internal = {
                    **public,
                    "cache_dir": cache_root,
                    "snapshot_path": snapshot,
                    "catalog": profile,
                }
                candidates[identity] = public
                private_records[identifier] = internal
                if len(candidates) >= _MAX_LOCAL_CANDIDATES:
                    truncated = True
                    break
            if len(candidates) >= _MAX_LOCAL_CANDIDATES:
                break
        if len(candidates) >= _MAX_LOCAL_CANDIDATES:
            break

    models = sorted(
        candidates.values(),
        key=lambda item: (
            not bool(item["selected"]),
            str(item["model_id"]).lower(),
            str(item["revision"]),
        ),
    )
    workspace: dict[str, Any] = {
        "models": models,
        "scanned_cache_count": len(roots),
        "ignored_incomplete_count": ignored_incomplete,
    }
    if truncated:
        workspace["truncated"] = True
    return workspace, private_records


def discover_local_models(config_path: str | Path) -> dict[str, Any]:
    """List structurally ready supported snapshots without network or paths."""

    workspace, _private = _local_model_records(load_config(config_path))
    return workspace


def adopt_local_model(config_path: str | Path, candidate_id: str) -> dict[str, Any]:
    """Select one discovered snapshot after re-scanning inside the project gate."""

    if not isinstance(candidate_id, str) or LOCAL_CANDIDATE_ID.fullmatch(candidate_id) is None:
        raise ConfigError("local model candidate is invalid; scan local models again")
    with project_mutation_gate(config_path):
        with project_config_mutation(config_path):
            # Candidate IDs never grant path access. The server reconstructs
            # the allowlisted roots and repeats every structural check while
            # setup is locked, closing the discovery-to-selection race.
            current = load_config(config_path)
            _workspace, candidates = _local_model_records(current)
            candidate = candidates.get(candidate_id)
            if candidate is None:
                raise ConfigError(
                    "local model candidate is unavailable; scan local models again"
                )
            cache_dir = Path(candidate["cache_dir"])
            snapshot = Path(candidate["snapshot_path"])
            evidence = _inspect_local_snapshot(snapshot, snapshot.parent.parent)
            if evidence is None:
                raise ConfigError(
                    "local model candidate is incomplete; scan local models again"
                )

            model = current.data["model"]
            model["id"] = candidate["model_id"]
            model["revision"] = candidate["revision"]
            model["loader"] = candidate["catalog"]["loader"]
            model["cache_dir"] = str(cache_dir)
            write_config(current.path, current.data, overwrite=True)

            completed_at = datetime.now(timezone.utc).isoformat()
            receipt = {
                "schema_version": 1,
                "source": "adopted_local_cache",
                "model_id": candidate["model_id"],
                "requested_revision": candidate["revision"],
                "revision": candidate["revision"],
                "snapshot_path": str(snapshot),
                "cache_dir": str(cache_dir),
                **evidence,
                "completed_at": completed_at,
            }
            _write_receipt(current.artifact_dir, receipt)

    public_candidate = {
        key: value
        for key, value in candidate.items()
        if key not in {"cache_dir", "snapshot_path", "catalog"}
    }
    public_candidate["selected"] = True
    return {
        "model": dict(model),
        "catalog_key": candidate["catalog_key"],
        "catalog": dict(candidate["catalog"]),
        "candidate": public_candidate,
    }


def _read_receipt(artifact_dir: Path) -> dict[str, Any] | None:
    path = _receipt_path(artifact_dir)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return payload if isinstance(payload, dict) else None


def _receipt_snapshot_evidence(
    receipt: Mapping[str, Any],
    cache_dir: Path,
    model_id: str,
    revision: str,
) -> dict[str, int] | None:
    """Revalidate a receipt against the exact expected Hub cache location."""

    try:
        snapshot = Path(str(receipt.get("snapshot_path", ""))).expanduser().resolve(
            strict=True
        )
        expected_repo = (
            cache_dir / ("models--" + model_id.replace("/", "--"))
        ).resolve(strict=True)
        expected_snapshot = (expected_repo / "snapshots" / revision.lower()).resolve(
            strict=True
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError):
        return None
    if snapshot != expected_snapshot or not _within(expected_repo, cache_dir):
        return None
    return _inspect_local_snapshot(snapshot, expected_repo)


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


def _reference_profile(model_name: str) -> tuple[str, dict[str, Any]]:
    """Resolve only catalogued benchmark references, never arbitrary disk pulls."""

    if model_name in MODEL_CATALOG:
        alias = model_name
    else:
        alias = next(
            (
                key
                for key, details in MODEL_CATALOG.items()
                if details.get("id") == model_name
            ),
            "",
        )
    profile = MODEL_CATALOG.get(alias)
    if not alias or not profile or profile.get("purpose") != "benchmark_reference":
        raise ConfigError("model is not a catalogued V1 benchmark reference")
    revision = str(profile.get("default_revision", ""))
    if IMMUTABLE_REVISION.fullmatch(revision) is None:
        raise ConfigError("benchmark reference must have an immutable catalog revision")
    return alias, dict(profile)


def inspect_reference_model(
    config_path: str | Path,
    model_name: str = "qwythos-9b-reference",
) -> dict[str, Any]:
    """Inspect the pinned benchmark snapshot without making a network request."""

    config = load_config(config_path)
    alias, profile = _reference_profile(model_name)
    model_id = str(profile["id"])
    revision = str(profile["default_revision"]).lower()
    cache_dir = _configured_cache_dir(config)
    result: dict[str, Any] = {
        "alias": alias,
        "model_id": model_id,
        "revision": revision,
        "status": "not_downloaded",
        "snapshot_path": None,
        "cache_dir": str(cache_dir),
        "receipt": None,
    }
    receipt_path = _reference_receipt_path(config.artifact_dir, alias)
    receipt = _read_json_receipt(receipt_path)
    if (
        receipt
        and receipt.get("model_id") == model_id
        and receipt.get("revision") == revision
        and Path(str(receipt.get("cache_dir", ""))).expanduser().resolve()
        == cache_dir.resolve()
    ):
        evidence = _receipt_snapshot_evidence(
            receipt,
            cache_dir.resolve(),
            model_id,
            revision,
        )
        if evidence is not None:
            result.update(
                status="downloaded",
                snapshot_path=receipt["snapshot_path"],
                receipt=str(receipt_path),
                **evidence,
            )
            return result
    try:
        snapshot_download, _scan_cache_dir = _hub_functions()
        snapshot = Path(
            snapshot_download(
                repo_id=model_id,
                revision=revision,
                cache_dir=cache_dir,
                local_files_only=True,
            )
        ).resolve()
    except Exception:
        return result
    evidence = _receipt_snapshot_evidence(
        {"snapshot_path": str(snapshot)},
        cache_dir.resolve(),
        model_id,
        revision,
    )
    if evidence is None:
        return result
    result.update(
        status="cached_unverified",
        snapshot_path=str(snapshot),
        **evidence,
    )
    return result


def _read_json_receipt(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def materialize_reference_model_owned(
    config_path: str | Path,
    model_name: str = "qwythos-9b-reference",
) -> dict[str, Any]:
    """Download one catalogued reference while the caller owns the project."""

    config = load_config(config_path)
    alias, profile = _reference_profile(model_name)
    model_id = str(profile["id"])
    revision = str(profile["default_revision"]).lower()
    snapshot_download, _scan_cache_dir = _hub_functions()
    cache_dir = _configured_cache_dir(config)
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        snapshot = Path(
            snapshot_download(
                repo_id=model_id,
                revision=revision,
                cache_dir=cache_dir,
                token=os.environ.get("HF_TOKEN") or None,
            )
        ).resolve()
    except Exception as error:
        raise ModelCacheError(
            "cannot download the pinned benchmark reference; check disk space, "
            "network access, model access, and Hugging Face login"
        ) from error
    file_count, logical_bytes = _snapshot_size(snapshot)
    receipt = {
        "schema_version": 1,
        "alias": alias,
        "model_id": model_id,
        "revision": revision,
        "snapshot_path": str(snapshot),
        "cache_dir": str(cache_dir.resolve()),
        "file_count": file_count,
        "logical_bytes": logical_bytes,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    destination = _reference_receipt_path(config.artifact_dir, alias)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return {"status": "downloaded", **receipt, "receipt": str(destination)}


def materialize_reference_model(
    config_path: str | Path,
    model_name: str = "qwythos-9b-reference",
) -> dict[str, Any]:
    """Synchronously cache a benchmark reference for agent CLI callers."""

    with project_mutation_gate(config_path):
        return materialize_reference_model_owned(config_path, model_name)


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
    ):
        evidence = _receipt_snapshot_evidence(
            receipt,
            configured_cache_dir,
            model_id,
            revision,
        )
        if evidence is not None:
            result.update(
                status="downloaded",
                snapshot_path=receipt["snapshot_path"],
                receipt=str(_receipt_path(config.artifact_dir)),
                **evidence,
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

    evidence = _receipt_snapshot_evidence(
        {"snapshot_path": str(snapshot)},
        configured_cache_dir,
        model_id,
        revision,
    )
    if evidence is None:
        return result
    result.update(
        status="cached_unverified",
        snapshot_path=str(snapshot),
        **evidence,
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
        snapshot = Path(
            snapshot_download(
                repo_id=model_id,
                revision=revision,
                cache_dir=cache_dir,
                local_files_only=True,
            )
        ).resolve()
        evidence = _receipt_snapshot_evidence(
            {"snapshot_path": str(snapshot)},
            cache_dir.resolve(),
            model_id,
            revision,
        )
        if evidence is None:
            raise ModelCacheError("snapshot is structurally incomplete")
    except Exception as error:
        raise ModelCacheError(
            f"The exact model snapshot {model_id}@{revision} is not complete in "
            f"{cache_dir}. Run autotrainer model download before training."
        ) from error


__all__ = [
    "adopt_local_model",
    "discover_local_models",
    "ModelCacheError",
    "inspect_model_cache",
    "inspect_reference_model",
    "materialize_model",
    "materialize_model_owned",
    "materialize_reference_model",
    "materialize_reference_model_owned",
    "require_materialized_model",
]
