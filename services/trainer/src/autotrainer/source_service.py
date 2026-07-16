"""Shared source onboarding for the human API and agent-facing CLI.

The normal V1 path accepts one value and infers the source contract.  Advanced
callers may still supply the older declaration fields, but neither client gets
to maintain a second source policy beside this module.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
from typing import Any
from urllib.parse import unquote, urlsplit

from .config import ConfigError, load_config, validate_mapping, write_config
from .sources import materialize_repository


_SOURCE_MUTATION_LOCK = threading.RLock()
_SOURCE_ID_PATTERN = re.compile(r"[^a-z0-9._-]+")
_GITHUB_PART_PATTERN = re.compile(r"[A-Za-z0-9_.-]+")
_GITHUB_SCP_PATTERN = re.compile(r"^(?:[^@\s]+@)?github\.com:(?P<path>[^?#]+)$", re.IGNORECASE)


def _resolve_local(value: str, root: Path) -> Path:
    candidate = Path(value).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()


def _display_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _github_parts(value: str, *, allow_shorthand: bool = True) -> tuple[str, str] | None:
    """Return owner/repository without retaining credentials or URL metadata."""

    text = value.strip()
    if not text:
        return None
    scp_match = _GITHUB_SCP_PATTERN.fullmatch(text)
    if scp_match:
        path = scp_match.group("path")
    elif "://" in text:
        parsed = urlsplit(text)
        if (parsed.hostname or "").rstrip(".").casefold() != "github.com":
            return None
        path = unquote(parsed.path)
    elif allow_shorthand:
        shorthand = re.fullmatch(r"(?:github\.com/)?(?P<path>[^/\s]+/[^/\s]+)", text, re.IGNORECASE)
        if shorthand is None:
            return None
        path = shorthand.group("path")
    else:
        return None

    parts = [part for part in path.strip("/").split("/") if part]
    if len(parts) != 2:
        raise ConfigError("GitHub sources must identify one owner/repository pair")
    owner, repository = parts
    if repository.casefold().endswith(".git"):
        repository = repository[:-4]
    if not owner or not repository or not _GITHUB_PART_PATTERN.fullmatch(owner) or not _GITHUB_PART_PATTERN.fullmatch(repository):
        raise ConfigError("GitHub owner and repository names contain unsupported characters")
    return owner, repository


def _github_url(owner: str, repository: str) -> str:
    # Only this credential-free form reaches git argv or persisted responses.
    return f"https://github.com/{owner}/{repository}.git"


def _git(repository: Path, *arguments: str) -> str | None:
    environment = {**os.environ, "GIT_OPTIONAL_LOCKS": "0", "GIT_TERMINAL_PROMPT": "0"}
    try:
        completed = subprocess.run(
            ["git", "-c", f"safe.directory={repository.as_posix()}", "-C", str(repository), *arguments],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
            env=environment,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def _task_partition(path: Path) -> str:
    manifests = [path] if path.is_file() else sorted(path.rglob("task.json"))
    if not manifests:
        raise ConfigError("task-pack directories must contain at least one task.json")
    partitions: set[str] = set()
    for manifest_path in manifests:
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            task = payload.get("task", {}) if isinstance(payload, Mapping) else {}
            split = task.get("split") if isinstance(task, Mapping) else None
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ConfigError(f"cannot read task manifest {manifest_path.name}") from error
        if split not in {"train", "evaluation"}:
            raise ConfigError(f"task manifest {manifest_path.name} must declare task.split")
        partitions.add(str(split))
    if len(partitions) != 1:
        raise ConfigError("one task-pack source cannot mix train and evaluation tasks")
    return next(iter(partitions))


def _looks_like_task_pack(path: Path) -> bool:
    return (path.is_file() and path.name.casefold() == "task.json") or (
        path.is_dir() and any(path.rglob("task.json"))
    )


def _slug(value: str) -> str:
    candidate = _SOURCE_ID_PATTERN.sub("-", value.casefold()).strip("-.")
    if not candidate or not candidate[0].isalnum():
        candidate = f"source-{candidate}".strip("-")
    return candidate or "source"


def _unique_id(base: str, value: str, sources: Sequence[Mapping[str, Any]]) -> str:
    used = {str(source.get("id", "")) for source in sources}
    candidate = _slug(base)
    if candidate not in used:
        return candidate
    digest = hashlib.sha256(value.casefold().encode("utf-8")).hexdigest()[:8]
    candidate = f"{candidate}-{digest}"
    if candidate not in used:
        return candidate
    for suffix in range(2, 10_000):
        numbered = f"{candidate}-{suffix}"
        if numbered not in used:
            return numbered
    raise ConfigError("could not allocate a unique source id")


def _infer_source(value: str, root: Path) -> tuple[dict[str, Any], str, str]:
    local = _resolve_local(value, root)
    if local.exists():
        if local.is_file() and local.suffix.casefold() == ".jsonl":
            return (
                {"kind": "sft_jsonl", "uri": _display_path(local, root), "partition": "train", "roles": ["demonstrations"]},
                local.stem,
                "local",
            )
        if local.is_dir():
            git_root = _git(local, "rev-parse", "--show-toplevel")
            revision = _git(local, "rev-parse", "HEAD") if git_root else None
            if (
                git_root
                and Path(git_root).resolve() == local.resolve()
                and revision
                and re.fullmatch(r"[0-9a-fA-F]{40,64}", revision)
            ):
                return (
                    {
                        "kind": "repository",
                        "uri": _display_path(local, root),
                        "partition": "train",
                        "roles": ["style", "history"],
                        "revision": revision.lower(),
                        "license": {"spdx": "UNDECLARED"},
                    },
                    local.name,
                    "local",
                )
        if _looks_like_task_pack(local):
            partition = _task_partition(local)
            return (
                {
                    "kind": "task_pack",
                    "uri": _display_path(local, root),
                    "partition": partition,
                    "roles": ["evaluation" if partition == "evaluation" else "rl_tasks"],
                },
                local.parent.name if local.is_file() else local.name,
                "local",
            )
        raise ConfigError("local sources must be a Git directory, .jsonl file, or task-pack directory/task.json")

    github = _github_parts(value)
    if github:
        owner, repository = github
        return (
            {
                "kind": "repository",
                "uri": _github_url(owner, repository),
                "partition": "train",
                "roles": ["style", "history"],
                "revision": "HEAD",
                "license": {"spdx": "UNDECLARED"},
            },
            f"{owner}-{repository}",
            "github",
        )
    raise ConfigError("source must be a GitHub owner/repository, GitHub URL, or supported local path")


def _advanced_source(
    value: str,
    *,
    kind: str,
    partition: str | None,
    roles: Sequence[str] | None,
    revision: str | None,
    license_spdx: str | None,
) -> tuple[dict[str, Any], str]:
    selected_partition = partition or "train"
    source: dict[str, Any] = {
        "kind": kind,
        "uri": value,
        "partition": selected_partition,
    }
    if kind == "repository":
        source["roles"] = list(roles or ("style",))
        source["revision"] = revision or "HEAD"
        source["license"] = {"spdx": license_spdx or "UNDECLARED"}
    elif kind == "sft_jsonl":
        source["roles"] = ["demonstrations"]
    elif kind == "task_pack":
        source["roles"] = ["evaluation" if selected_partition == "evaluation" else "rl_tasks"]
    else:
        raise ConfigError("source kind must be repository, sft_jsonl, or task_pack")
    return source, Path(value.rstrip("/\\")).stem or kind


def _managed_source_root(config: Any) -> Path:
    return (config.artifact_dir / "sources").resolve()


def _managed_source_path(config: Any, source: Mapping[str, Any]) -> Path | None:
    uri = str(source.get("uri", "")).strip()
    if not uri or "://" in uri or uri.startswith("git@"):
        return None
    path = config.resolve_path(uri)
    root = _managed_source_root(config)
    try:
        relative = path.resolve().relative_to(root)
    except (OSError, ValueError):
        return None
    # A managed source is always one child named from its declaration.  This
    # prevents source removal from recursively deleting an arbitrary artifact.
    return path if len(relative.parts) == 1 and relative.name == str(source.get("id", "")) else None


def _canonical_locator(config: Any, source: Mapping[str, Any]) -> str:
    """Identify duplicate inputs without relying on user-selected source IDs."""

    kind = str(source.get("kind", ""))
    uri = str(source.get("uri", "")).strip()
    if kind == "repository":
        github = (
            _github_parts(uri, allow_shorthand=True)
            if "://" in uri or uri.startswith("git@") or uri.casefold().startswith("github.com/")
            else None
        )
        local: Path | None = None
        if github is None and uri:
            local = config.resolve_path(uri)
            if local.is_dir():
                remote = _git(local, "remote", "get-url", "origin")
                github = _github_parts(remote, allow_shorthand=False) if remote else None
        if github:
            return f"github:{github[0].casefold()}/{github[1].casefold()}"
        if local is not None and local.is_dir():
            git_root = _git(local, "rev-parse", "--show-toplevel")
            if git_root:
                return "git:" + os.path.normcase(str(Path(git_root).resolve()))
        if "://" in uri or uri.startswith("git@"):
            return "remote:" + uri.casefold()
    if uri:
        return f"{kind}:" + os.path.normcase(str(config.resolve_path(uri)))
    return ""


def _serialize_source(config: Any, source: Mapping[str, Any]) -> dict[str, Any]:
    source_id = str(source.get("id", ""))
    kind = str(source.get("kind", ""))
    uri = str(source.get("uri", ""))
    managed_path = _managed_source_path(config, source) if kind == "repository" else None
    github: tuple[str, str] | None = None
    if managed_path is not None:
        remote = _git(managed_path, "remote", "get-url", "origin")
        github = _github_parts(remote, allow_shorthand=False) if remote else None
    elif kind == "repository" and ("://" in uri or uri.startswith("git@")):
        github = _github_parts(uri, allow_shorthand=False)

    if github:
        owner, repository = github
        origin = "github"
        value = _github_url(owner, repository)
        label = f"{owner}/{repository}"
    else:
        origin = "local"
        value = uri
        local_name = Path(uri.rstrip("/\\")).name
        label = local_name or source_id
    purpose = {"repository": "work", "sft_jsonl": "examples", "task_pack": "tasks"}.get(kind, "work")
    record: dict[str, Any] = {
        "id": source_id,
        "kind": kind,
        "label": label,
        "value": value,
        "origin": origin,
        "purpose": purpose,
        "status": "ready",
    }
    revision = str(source.get("revision", "")).strip()
    if revision:
        record["revision"] = revision
    return record


def list_sources(config_path: str | Path) -> list[dict[str, Any]]:
    """Return the compact source shape consumed by both clients."""

    config = load_config(config_path)
    return [_serialize_source(config, source) for source in config.sources]


def add_source(
    config_path: str | Path,
    value: str,
    *,
    name: str | None = None,
    kind: str | None = None,
    partition: str | None = None,
    roles: Sequence[str] | None = None,
    revision: str | None = None,
    license_spdx: str | None = None,
) -> dict[str, Any]:
    """Infer, persist, and when needed securely materialize one source."""

    supplied = str(value).strip()
    if not supplied:
        raise ConfigError("source value is required")
    with _SOURCE_MUTATION_LOCK:
        config = load_config(config_path)
        inferred_origin = "local"
        if kind is None:
            declared, base_name, inferred_origin = _infer_source(supplied, config.root)
            if partition is not None:
                declared["partition"] = partition
                if declared["kind"] == "task_pack":
                    declared["roles"] = ["evaluation" if partition == "evaluation" else "rl_tasks"]
        else:
            declared, base_name = _advanced_source(
                supplied,
                kind=kind,
                partition=partition,
                roles=roles,
                revision=revision,
                license_spdx=license_spdx,
            )
            github = _github_parts(supplied, allow_shorthand=True) if kind == "repository" else None
            if github:
                declared["uri"] = _github_url(*github)
                inferred_origin = "github"

        locator = _canonical_locator(config, declared)
        if locator and any(_canonical_locator(config, source) == locator for source in config.sources):
            raise ConfigError("source is already added")
        source_id = _slug(name) if name else _unique_id(base_name, str(declared["uri"]), config.sources)
        if any(str(source.get("id")) == source_id for source in config.sources):
            raise ConfigError(f"source id already exists: {source_id}")
        declared["id"] = source_id
        trial = dict(config.data)
        trial["sources"] = [*config.sources, declared]
        report = validate_mapping(trial, root=config.root)
        if report.errors:
            raise ConfigError("\n".join(report.errors))

        materialized_path: Path | None = None
        try:
            if inferred_origin == "github":
                try:
                    materialized = materialize_repository(trial, config.root, source_id)
                except (RuntimeError, ValueError) as error:
                    raise ConfigError(
                        "could not clone and pin the GitHub repository; check access, network, and repository name"
                    ) from error
                declared = dict(materialized["updated_source"])
                materialized_path = Path(str(materialized["local_path"])).resolve()
                trial["sources"][-1] = declared
            write_config(config.path, trial, overwrite=True)
        except Exception:
            if materialized_path is not None and materialized_path.exists():
                root = _managed_source_root(config)
                try:
                    materialized_path.relative_to(root)
                except ValueError:
                    pass
                else:
                    shutil.rmtree(materialized_path)
            raise

        refreshed = load_config(config.path)
        serialized = _serialize_source(refreshed, declared)
        return {"source": serialized, "sources": [_serialize_source(refreshed, source) for source in refreshed.sources]}


def remove_source(config_path: str | Path, source_id: str) -> dict[str, Any]:
    """Remove a declaration and only clean up clones managed by AutoTrainer."""

    requested_id = str(source_id).strip()
    if not requested_id:
        raise ConfigError("source id is required")
    with _SOURCE_MUTATION_LOCK:
        config = load_config(config_path)
        matching = [source for source in config.sources if str(source.get("id")) == requested_id]
        if not matching:
            raise ConfigError(f"source does not exist: {requested_id}")
        source = matching[0]
        removed = _serialize_source(config, source)
        managed_path = _managed_source_path(config, source)
        updated = dict(config.data)
        updated["sources"] = [item for item in config.sources if str(item.get("id")) != requested_id]
        write_config(config.path, updated, overwrite=True)

        if managed_path is not None and managed_path.exists():
            root = _managed_source_root(config)
            resolved = managed_path.resolve()
            # Re-check the final absolute path immediately before recursive deletion.
            if resolved.parent == root and resolved.name == requested_id:
                shutil.rmtree(resolved)
        refreshed = load_config(config.path)
        return {"removed": removed, "sources": [_serialize_source(refreshed, item) for item in refreshed.sources]}


__all__ = ["add_source", "list_sources", "remove_source"]
