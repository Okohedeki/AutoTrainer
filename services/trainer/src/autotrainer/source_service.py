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
from typing import Any
from urllib.parse import unquote, urlsplit

from .config import (
    ConfigError,
    load_config,
    project_config_mutation,
    validate_mapping,
    write_config,
)
from .project_gate import project_mutation_gate
from .sources import materialize_repository


_SOURCE_ID_PATTERN = re.compile(r"[^a-z0-9._-]+")
_GITHUB_PART_PATTERN = re.compile(r"[A-Za-z0-9_.-]+")
_GITHUB_SCP_PATTERN = re.compile(r"^(?:[^@\s]+@)?github\.com:(?P<path>[^?#]+)$", re.IGNORECASE)

# Human-facing modes say why a repository was added.  They intentionally map
# onto the same low-level roles used by YAML and the agent CLI, so the GUI
# cannot create a second source policy or imply that raw code is training data.
_MODE_TO_ROLE = {
    "accepted_changes": "history",
    "practice_tasks": "rl_seed",
    "reference_only": "style",
    "evaluation_holdout": "evaluation",
}
_MODE_ORDER = tuple(_MODE_TO_ROLE)


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


def _strings(value: Sequence[str] | None, field: str) -> list[str]:
    """Normalize an optional list without treating one string as characters."""

    if value is None:
        return []
    if isinstance(value, (str, bytes, bytearray)):
        raise ConfigError(f"{field} must be a list of strings")
    normalized = [str(item).strip() for item in value]
    if any(not item for item in normalized):
        raise ConfigError(f"{field} cannot contain empty values")
    return normalized


def _normalize_modes(value: Sequence[str] | None) -> list[str] | None:
    if value is None:
        return None
    supplied = _strings(value, "modes")
    invalid = sorted(set(supplied) - set(_MODE_TO_ROLE))
    if invalid:
        raise ConfigError(
            "source modes must be accepted_changes, practice_tasks, "
            "reference_only, or evaluation_holdout"
        )
    if len(set(supplied)) != len(supplied):
        raise ConfigError("source modes must be unique")
    selected = [mode for mode in _MODE_ORDER if mode in supplied]
    if "evaluation_holdout" in selected and len(selected) != 1:
        raise ConfigError(
            "evaluation_holdout cannot be combined with training source modes"
        )
    return selected


def _apply_repository_modes(
    source: dict[str, Any],
    modes: Sequence[str],
    *,
    requested_partition: str | None,
) -> None:
    """Project product modes into the canonical repository declaration."""

    if source.get("kind") != "repository":
        raise ConfigError("source modes are only supported for Git repositories")
    if not modes:
        raise ConfigError("at least one source mode is required for a repository")
    evaluation = "evaluation_holdout" in modes
    partition = "evaluation" if evaluation else "train"
    if requested_partition is not None and requested_partition != partition:
        raise ConfigError(
            f"source partition {requested_partition!r} conflicts with the selected modes"
        )
    source["partition"] = partition
    source["roles"] = [_MODE_TO_ROLE[mode] for mode in modes]


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
        selected_roles = _strings(roles, "roles") if roles is not None else []
        source["roles"] = selected_roles or ["style"]
        source["revision"] = revision or "HEAD"
        source["license"] = {"spdx": license_spdx or "UNDECLARED"}
    elif kind == "sft_jsonl":
        source["roles"] = ["demonstrations"]
    elif kind == "task_pack":
        source["roles"] = ["evaluation" if selected_partition == "evaluation" else "rl_tasks"]
    else:
        raise ConfigError("source kind must be repository, sft_jsonl, or task_pack")
    return source, Path(value.rstrip("/\\")).stem or kind


def _apply_optional_fields(
    source: dict[str, Any],
    *,
    revision: str | None,
    include: Sequence[str] | None,
    exclude: Sequence[str] | None,
    license_spdx: str | None,
    license_attribution: str | None,
) -> None:
    """Retain reviewed source scope regardless of inference or advanced input."""

    kind = str(source.get("kind", ""))
    if revision is not None:
        selected_revision = str(revision).strip()
        if kind != "repository":
            raise ConfigError("revision is only supported for Git repositories")
        if not selected_revision:
            raise ConfigError("revision must be non-empty")
        source["revision"] = selected_revision

    include_values = _strings(include, "include") if include is not None else None
    exclude_values = _strings(exclude, "exclude") if exclude is not None else None
    if (include_values is not None or exclude_values is not None) and kind != "repository":
        raise ConfigError("include and exclude filters are only supported for Git repositories")
    if include_values is not None:
        source["include"] = include_values
    if exclude_values is not None:
        source["exclude"] = exclude_values

    if license_spdx is not None or license_attribution is not None:
        selected_spdx = str(license_spdx or "UNDECLARED").strip()
        selected_attribution = (
            str(license_attribution).strip() if license_attribution is not None else None
        )
        if not selected_spdx:
            raise ConfigError("license SPDX value must be non-empty")
        if license_attribution is not None and not selected_attribution:
            raise ConfigError("license attribution must be non-empty")
        license_value: dict[str, str] = {"spdx": selected_spdx}
        if selected_attribution is not None:
            license_value["attribution"] = selected_attribution
        source["license"] = license_value


def _managed_source_root(config: Any) -> Path:
    return (config.artifact_dir / "sources").resolve()


def _public_materialization_error(error: Exception) -> str:
    """Translate credential-free Git failures into useful localhost guidance."""

    detail = str(error).casefold()
    if "timed out" in detail:
        return "GitHub repository download timed out; check the connection and retry"
    if any(
        marker in detail
        for marker in (
            "repository not found",
            "authentication failed",
            "could not read username",
            "access denied",
        )
    ):
        return (
            "GitHub repository was not found or requires access; use owner/repository "
            "and verify Git credentials"
        )
    if "cannot resolve declared revision" in detail:
        return "GitHub repository does not contain the requested branch, tag, or commit"
    return (
        "could not clone and pin the GitHub repository; check access, network, "
        "and repository name"
    )


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


def _declared_modes(source: Mapping[str, Any]) -> list[str]:
    """Reconstruct product modes for old YAML and new human declarations."""

    if source.get("kind") != "repository":
        return []
    roles = set(_strings(source.get("roles"), "roles"))
    if source.get("partition") == "evaluation" or "evaluation" in roles:
        return ["evaluation_holdout"]
    return [mode for mode in _MODE_ORDER if _MODE_TO_ROLE[mode] in roles]


def _next_action(kind: str, modes: Sequence[str], partition: str) -> dict[str, str]:
    """Explain what must happen before a configured source can teach anything."""

    selected = set(modes)
    if "evaluation_holdout" in selected:
        return {
            "title": "Add held-out tasks",
            "detail": (
                "This repository is isolated from training. Add held-out tasks and "
                "verifiers before it can evaluate the model."
            ),
        }
    if "accepted_changes" in selected and "practice_tasks" in selected:
        return {
            "title": "Review changes and add tasks",
            "detail": (
                "Approve useful Git changes as demonstrations, then create or import "
                "executable practice tasks with verifiers."
            ),
        }
    if "accepted_changes" in selected:
        return {
            "title": "Review accepted changes",
            "detail": (
                "Raw code is not a demonstration. Approve useful Git changes and "
                "supply the instruction each change answered."
            ),
        }
    if "practice_tasks" in selected:
        return {
            "title": "Add practice tasks",
            "detail": (
                "This repository supplies starting states only. Create or import "
                "executable tasks and verifiers before reinforcement learning."
            ),
        }
    if "reference_only" in selected:
        return {
            "title": "Reference configured",
            "detail": "Reference code is inspectable evidence and does not train the model by itself.",
        }
    if kind == "sft_jsonl":
        return {
            "title": "Validate examples",
            "detail": "Prepare the project to validate and compile these authored demonstrations.",
        }
    if kind == "task_pack":
        return {
            "title": "Validate held-out tasks" if partition == "evaluation" else "Validate practice tasks",
            "detail": (
                "Prepare the project to validate each task, starting state, and verifier "
                "before execution."
            ),
        }
    return {
        "title": "Choose a learning purpose",
        "detail": "A repository needs an explicit purpose before it can contribute training data.",
    }


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
    partition = str(source.get("partition", "train"))
    roles = _strings(source.get("roles"), "roles")
    modes = _declared_modes(source)
    include = _strings(source.get("include"), "include") if source.get("include") is not None else []
    exclude = _strings(source.get("exclude"), "exclude") if source.get("exclude") is not None else []
    license_value = source.get("license")
    serialized_license = dict(license_value) if isinstance(license_value, Mapping) else None
    record: dict[str, Any] = {
        "id": source_id,
        "kind": kind,
        "label": label,
        "value": value,
        "origin": origin,
        "purpose": purpose,
        "modes": modes,
        "partition": partition,
        "roles": roles,
        "filters": {"include": include, "exclude": exclude},
        "license": serialized_license,
        # Configured means the declaration is persisted. Readiness is resolved
        # later by scan/compile; a repository alone is never called training data.
        "status": "configured",
        "next_action": _next_action(kind, modes, partition),
    }
    revision = str(source.get("revision", "")).strip()
    if revision:
        record["revision"] = revision
    return record


def list_sources(config_path: str | Path) -> list[dict[str, Any]]:
    """Return the compact source shape consumed by both clients."""

    config = load_config(config_path)
    return [_serialize_source(config, source) for source in config.sources]


def _add_source_owned(
    config_path: str | Path,
    value: str,
    *,
    name: str | None = None,
    kind: str | None = None,
    partition: str | None = None,
    roles: Sequence[str] | None = None,
    modes: Sequence[str] | None = None,
    require_modes: bool = False,
    revision: str | None = None,
    include: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
    license_spdx: str | None = None,
    license_attribution: str | None = None,
) -> dict[str, Any]:
    """Infer, persist, and when needed securely materialize one source."""

    supplied = str(value).strip()
    if not supplied:
        raise ConfigError("source value is required")
    normalized_modes = _normalize_modes(modes)
    if normalized_modes is not None and roles is not None:
        raise ConfigError("use source modes or repository roles, not both")
    with project_config_mutation(config_path):
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

        if normalized_modes is not None:
            _apply_repository_modes(
                declared,
                normalized_modes,
                requested_partition=partition,
            )
        elif require_modes and declared.get("kind") == "repository":
            # Deterministic files retain their intrinsic purpose. Repositories
            # are ambiguous and the human/API path must say how they contribute.
            raise ConfigError("at least one source mode is required for a repository")

        _apply_optional_fields(
            declared,
            revision=revision,
            include=include,
            exclude=exclude,
            license_spdx=license_spdx,
            license_attribution=license_attribution,
        )

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
                    raise ConfigError(_public_materialization_error(error)) from error
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


def _remove_source_owned(config_path: str | Path, source_id: str) -> dict[str, Any]:
    """Remove a declaration and only clean up clones managed by AutoTrainer."""

    requested_id = str(source_id).strip()
    if not requested_id:
        raise ConfigError("source id is required")
    with project_config_mutation(config_path):
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


def add_source(
    config_path: str | Path,
    value: str,
    *,
    name: str | None = None,
    kind: str | None = None,
    partition: str | None = None,
    roles: Sequence[str] | None = None,
    modes: Sequence[str] | None = None,
    require_modes: bool = False,
    revision: str | None = None,
    include: Sequence[str] | None = None,
    exclude: Sequence[str] | None = None,
    license_spdx: str | None = None,
    license_attribution: str | None = None,
) -> dict[str, Any]:
    """Add one source only while no training snapshot is active.

    ``require_modes`` is the human/API guard. Agent and YAML-compatible callers
    may omit it and continue supplying explicit low-level roles as before.
    """

    with project_mutation_gate(config_path):
        return _add_source_owned(
            config_path,
            value,
            name=name,
            kind=kind,
            partition=partition,
            roles=roles,
            modes=modes,
            require_modes=require_modes,
            revision=revision,
            include=include,
            exclude=exclude,
            license_spdx=license_spdx,
            license_attribution=license_attribution,
        )


def remove_source(config_path: str | Path, source_id: str) -> dict[str, Any]:
    """Remove one source only while no training snapshot is active."""

    with project_mutation_gate(config_path):
        return _remove_source_owned(config_path, source_id)


__all__ = ["add_source", "list_sources", "remove_source"]
