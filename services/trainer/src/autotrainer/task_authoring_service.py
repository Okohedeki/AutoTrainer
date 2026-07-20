"""Guided authoring for executable GRPO and held-out evaluation tasks.

A repository is only a resettable starting state.  This service deliberately
does not turn source files into training rows or invent correctness checks.  It
persists a task only after the operator supplies an instruction, runtime gates,
and an existing hidden verifier outside the policy-visible repository.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
from typing import Any
from uuid import uuid4

from .config import (
    ConfigError,
    load_config,
    project_config_mutation,
    validate_mapping,
    write_config,
)
from .manifest import TaskManifest
from .project_gate import project_mutation_gate


_IMMUTABLE_REVISION = re.compile(r"[0-9a-fA-F]{40,64}")
_SAFE_ID = re.compile(r"[^a-z0-9._-]+")
_PACK_IDS = {
    "train": "authored-practice-tasks",
    "evaluation": "authored-evaluation-tasks",
}
_PACK_ROLES = {"train": ["rl_tasks"], "evaluation": ["evaluation"]}
_DEFAULT_WEIGHTS = {
    "buildGate": True,
    "regressionGate": True,
    "regressionSafety": 0.20,
    "taskTests": 0.35,
    "responsiveRules": 0.20,
    "designRules": 0.15,
    "patchQuality": 0.10,
}
_DEFAULT_TOOLS = [
    "list_files",
    "read_file",
    "search_code",
    "apply_patch",
    "replace_text",
    "run_check",
]


def _text(value: object, field: str, *, minimum: int = 1, maximum: int = 4_000) -> str:
    if not isinstance(value, str):
        raise ConfigError(f"{field} must be text")
    selected = value.strip()
    if len(selected) < minimum:
        raise ConfigError(f"{field} must contain at least {minimum} characters")
    if len(selected) > maximum:
        raise ConfigError(f"{field} must contain at most {maximum} characters")
    if "\x00" in selected:
        raise ConfigError(f"{field} cannot contain a null byte")
    return selected


def _command(value: object, field: str, *, required: bool) -> str:
    if value is None or (isinstance(value, str) and not value.strip()):
        if required:
            raise ConfigError(f"{field} is required")
        return ""
    selected = _text(value, field, maximum=1_000)
    # Runtime commands are explicit shell contracts. Keeping each one on a
    # single line makes the generated manifest reviewable in both clients.
    if "\n" in selected or "\r" in selected:
        raise ConfigError(f"{field} must be one command on one line")
    return selected


def _slug(value: str, *, fallback: str) -> str:
    selected = _SAFE_ID.sub("-", value.casefold()).strip("-.")[:72]
    if not selected or not selected[0].isalnum():
        selected = f"{fallback}-{selected}".strip("-.")
    return selected or fallback


def _display_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _git(repository: Path, *arguments: str) -> str | None:
    environment = {
        **os.environ,
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_TERMINAL_PROMPT": "0",
    }
    try:
        completed = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={repository.as_posix()}",
                "-C",
                str(repository),
                *arguments,
            ],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
            env=environment,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def _repository_source(config: Any, source_id: str) -> tuple[Mapping[str, Any], Path, str, str]:
    selected = next(
        (
            source
            for source in config.sources
            if str(source.get("id", "")) == source_id
        ),
        None,
    )
    if selected is None or selected.get("kind") != "repository":
        raise ConfigError("source_id must identify a configured Git repository")

    partition = str(selected.get("partition", "train"))
    roles = {str(role) for role in selected.get("roles", [])}
    if partition == "train" and "rl_seed" not in roles:
        raise ConfigError(
            "the selected repository is not configured for executable practice tasks"
        )
    if partition == "evaluation" and "evaluation" not in roles:
        raise ConfigError(
            "the selected repository is not configured as an isolated evaluation holdout"
        )
    if partition not in _PACK_IDS:
        raise ConfigError("the selected repository partition must be train or evaluation")

    repository = config.resolve_path(str(selected.get("uri", "")))
    if not repository.is_dir() or not (repository / ".git").exists():
        raise ConfigError(
            "the selected repository must be downloaded locally before authoring a task"
        )
    declared_revision = str(selected.get("revision", "")).strip()
    if _IMMUTABLE_REVISION.fullmatch(declared_revision) is None:
        raise ConfigError(
            "the selected repository must be pinned to a full Git commit before authoring a task"
        )
    resolved_revision = _git(repository, "rev-parse", "--verify", f"{declared_revision}^{{commit}}")
    if resolved_revision is None or resolved_revision.casefold() != declared_revision.casefold():
        raise ConfigError("the selected repository's locked commit is not available locally")
    return selected, repository.resolve(), declared_revision.casefold(), partition


def _working_directory(repository: Path, revision: str, value: object) -> str:
    selected = _text(value, "working_directory", maximum=500).replace("\\", "/")
    relative = PurePosixPath(selected)
    if relative.is_absolute() or ".." in relative.parts or ".git" in relative.parts:
        raise ConfigError("working_directory must stay inside the locked repository and outside .git")
    normalized = relative.as_posix()
    if normalized in {"", "."}:
        return "."
    kind = _git(repository, "cat-file", "-t", f"{revision}:{normalized}")
    if kind != "tree":
        raise ConfigError(
            "working_directory must be a tracked directory at the repository's locked commit"
        )
    return normalized


def _verifier_bundle(config: Any, repository: Path, value: object) -> Path:
    supplied = Path(_text(value, "verifier_bundle", maximum=1_000)).expanduser()
    bundle = supplied.resolve() if supplied.is_absolute() else config.resolve_path(supplied)
    if not bundle.is_dir():
        raise ConfigError("verifier_bundle must be an existing local directory")
    try:
        bundle.relative_to(repository)
    except ValueError:
        pass
    else:
        raise ConfigError(
            "verifier_bundle must be outside the editable repository so the policy cannot inspect it"
        )
    if any(path.is_symlink() for path in bundle.rglob("*")):
        raise ConfigError("verifier_bundle cannot contain symbolic links")
    files = [path for path in bundle.rglob("*") if path.is_file()]
    if not files:
        raise ConfigError("verifier_bundle must contain at least one verifier file")
    if len(files) > 500:
        raise ConfigError("verifier_bundle exceeds the 500 file authoring limit")
    try:
        total_bytes = sum(path.stat().st_size for path in files)
    except OSError as error:
        raise ConfigError(f"verifier_bundle cannot be inspected: {error}") from error
    if total_bytes > 25 * 1024 * 1024:
        raise ConfigError("verifier_bundle exceeds the 25 MiB authoring limit")
    return bundle


def _report_path(value: object) -> str:
    selected = _text(value, "verifier_report_path", maximum=500).replace("\\", "/")
    relative = PurePosixPath(selected)
    if relative.is_absolute() or ".." in relative.parts or ".git" in relative.parts:
        raise ConfigError("verifier_report_path must stay inside the disposable workspace")
    return relative.as_posix()


def _managed_root(config: Any, split: str) -> Path:
    return (config.artifact_dir / "authored-tasks" / split).resolve()


def _existing_task_ids(pack_root: Path) -> set[str]:
    task_ids: set[str] = set()
    if not pack_root.is_dir():
        return task_ids
    for manifest_path in pack_root.glob("*/task.json"):
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            task = payload.get("task", {}) if isinstance(payload, Mapping) else {}
            task_id = str(task.get("id", "")) if isinstance(task, Mapping) else ""
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if task_id:
            task_ids.add(task_id)
    return task_ids


def _allocate_task_id(pack_root: Path, instruction: str, requested: str | None) -> str:
    base = _slug(requested or instruction[:72], fallback="task")
    used = _existing_task_ids(pack_root)
    if base not in used and not (pack_root / base).exists():
        return base
    for suffix in range(2, 10_000):
        candidate = f"{base[:68]}-{suffix}"
        if candidate not in used and not (pack_root / candidate).exists():
            return candidate
    raise ConfigError("could not allocate a unique task id")


def _task_pack_declaration(config: Any, split: str, pack_root: Path) -> tuple[str, bool]:
    """Return a stable task-pack ID and whether YAML needs a new declaration."""

    for source in config.sources:
        if source.get("kind") != "task_pack":
            continue
        uri = str(source.get("uri", ""))
        if uri and config.resolve_path(uri) == pack_root:
            if str(source.get("partition", "")) != split:
                raise ConfigError("the managed task pack has a conflicting partition")
            return str(source.get("id", "")), False

    base = _PACK_IDS[split]
    used = {str(source.get("id", "")) for source in config.sources}
    if base not in used:
        return base, True
    for suffix in range(2, 10_000):
        candidate = f"{base}-{suffix}"
        if candidate not in used:
            return candidate, True
    raise ConfigError("could not allocate a managed task-pack id")


def _updated_config(config: Any, split: str, pack_root: Path, pack_id: str) -> dict[str, Any]:
    updated = dict(config.data)
    sources = list(config.sources)
    sources.append(
        {
            "id": pack_id,
            "kind": "task_pack",
            "uri": _display_path(pack_root, config.root),
            "partition": split,
            "roles": _PACK_ROLES[split],
        }
    )
    updated["sources"] = sources
    if split == "evaluation":
        evaluation = dict(updated.get("evaluation", {}))
        current = str(evaluation.get("task_pack", "")).strip()
        declared_ids = {str(source.get("id", "")) for source in config.sources}
        # Preserve an existing explicit holdout choice. Replace only the empty
        # or default placeholder that cannot resolve to a declared source.
        if not current or current == "held-out-frontend" or current not in declared_ids:
            evaluation["task_pack"] = pack_id
        updated["evaluation"] = evaluation
    report = validate_mapping(updated, root=config.root)
    if report.errors:
        raise ConfigError("\n".join(report.errors))
    return updated


def _remove_created_directory(directory: Path, pack_root: Path) -> None:
    """Rollback only the exact child allocated by this service."""

    resolved = directory.resolve()
    root = pack_root.resolve()
    if resolved.parent == root and resolved.exists():
        shutil.rmtree(resolved)


def _serialize_manifest(
    path: Path,
    payload: Mapping[str, Any],
    config: Any,
) -> dict[str, Any]:
    manifest = TaskManifest.from_mapping(payload)
    bundle = Path(manifest.verifier_bundle or "").expanduser().resolve()
    blockers: list[str] = []
    if not bundle.is_dir():
        blockers.append("The authored hidden verifier directory is no longer available.")
    locked_revision = manifest.starting_revision
    if locked_revision == "locked":
        source = next(
            (
                item
                for item in config.sources
                if str(item.get("id", "")) == manifest.source_id
                and item.get("kind") == "repository"
            ),
            None,
        )
        resolved = str(source.get("revision", "")) if source is not None else ""
        if _IMMUTABLE_REVISION.fullmatch(resolved):
            locked_revision = resolved.casefold()
        else:
            blockers.append("The authored task no longer has its locked repository source.")
    return {
        "id": manifest.task_id,
        "instruction": manifest.instruction,
        "source_id": manifest.source_id,
        "locked_revision": locked_revision,
        "split": manifest.split,
        "group_id": manifest.group_id,
        "working_directory": manifest.working_directory,
        "runtime": dict(manifest.runtime_commands),
        "verifier": {
            "bundle": str(bundle),
            "command": manifest.verifier_command,
            "report_path": manifest.verifier_report_path,
        },
        "manifest_path": str(path.resolve()),
        # Static authoring cannot claim executability. Prepare runs the actual
        # container canary and is the only operation that can promote readiness.
        "status": "blocked" if blockers else "declared",
        "blockers": blockers,
        "next_action": {
            "title": "Run Prepare",
            "detail": (
                "Prepare will execute the install, build, tests, and hidden verifier "
                "against the locked starting state before GRPO can start."
            ),
        },
    }


def list_authored_tasks(config_path: str | Path) -> dict[str, Any]:
    """List only tasks created by this guided local authoring flow."""

    config = load_config(config_path)
    tasks: list[dict[str, Any]] = []
    for split in ("train", "evaluation"):
        root = _managed_root(config, split)
        if not root.is_dir():
            continue
        for manifest_path in sorted(root.glob("*/task.json")):
            try:
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                if not isinstance(payload, Mapping):
                    raise ValueError("manifest root is not an object")
                tasks.append(_serialize_manifest(manifest_path, payload, config))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as error:
                tasks.append(
                    {
                        "id": manifest_path.parent.name,
                        "split": split,
                        "manifest_path": str(manifest_path.resolve()),
                        "status": "blocked",
                        "blockers": [f"The authored task manifest is invalid: {error}"],
                    }
                )
    train_tasks = [task for task in tasks if task.get("split") == "train"]
    evaluation_tasks = [
        task for task in tasks if task.get("split") == "evaluation"
    ]
    evaluation_groups = {
        str(task.get("group_id", "")).strip()
        for task in evaluation_tasks
        if str(task.get("group_id", "")).strip()
    }
    required_evaluation_groups = 5
    return {
        "tasks": tasks,
        "summary": {
            "train_task_count": len(train_tasks),
            "evaluation_task_count": len(evaluation_tasks),
            "evaluation_group_count": len(evaluation_groups),
            "required_evaluation_groups": required_evaluation_groups,
            "evaluation_groups_remaining": max(
                0, required_evaluation_groups - len(evaluation_groups)
            ),
        },
    }


def create_authored_task(
    config_path: str | Path,
    *,
    source_id: str,
    instruction: str,
    working_directory: str,
    build: str,
    tests: str,
    verifier_bundle: str,
    verifier_command: str,
    verifier_report_path: str = ".autotrainer-verifier-report.json",
    install: str | None = None,
    browser_tests: str | None = None,
    task_id: str | None = None,
    group_id: str | None = None,
) -> dict[str, Any]:
    """Create one declared task and connect its managed pack to project YAML.

    This operation performs static checks only. It never runs repository code,
    a verifier, or a model; callers must run Prepare before claiming readiness.
    """

    selected_source_id = _text(source_id, "source_id", maximum=200)
    selected_instruction = _text(instruction, "instruction", minimum=20)
    selected_build = _command(build, "build", required=True)
    selected_tests = _command(tests, "tests", required=True)
    selected_install = _command(install, "install", required=False)
    selected_browser_tests = _command(browser_tests, "browser_tests", required=False)
    selected_verifier_command = _command(
        verifier_command, "verifier_command", required=True
    )
    if "/autotrainer-verifier" not in selected_verifier_command.replace("\\", "/"):
        raise ConfigError(
            "verifier_command must read the hidden bundle from /autotrainer-verifier"
        )
    selected_report_path = _report_path(verifier_report_path)
    selected_requested_id = (
        _text(task_id, "task_id", maximum=100) if task_id is not None else None
    )
    selected_group_id = (
        _text(group_id, "group_id", maximum=200) if group_id is not None else None
    )

    with project_mutation_gate(config_path):
        with project_config_mutation(config_path):
            config = load_config(config_path)
            _source, repository, revision, split = _repository_source(
                config, selected_source_id
            )
            selected_working_directory = _working_directory(
                repository, revision, working_directory
            )
            bundle = _verifier_bundle(config, repository, verifier_bundle)
            pack_root = _managed_root(config, split)
            pack_id, add_pack = _task_pack_declaration(config, split, pack_root)
            allocated_id = _allocate_task_id(
                pack_root, selected_instruction, selected_requested_id
            )
            payload: dict[str, Any] = {
                "version": "1.0",
                "task": {
                    "id": allocated_id,
                    "instruction": selected_instruction,
                    "sourceId": selected_source_id,
                    # The source declaration already contains the exact commit.
                    # "locked" makes compile attest that same project lock.
                    "startingRevision": "locked",
                    "split": split,
                    "groupId": selected_group_id or selected_source_id,
                },
                "runtime": {
                    "workingDirectory": selected_working_directory,
                    "install": selected_install,
                    "build": selected_build,
                    "tests": selected_tests,
                    "browserTests": selected_browser_tests,
                },
                "tools": list(_DEFAULT_TOOLS),
                "verifier": {
                    # The path is intentionally operator-authored and remains
                    # outside the editable source. AutoTrainer does not create
                    # a verifier that could silently encode the wrong goal.
                    "bundle": str(bundle),
                    "command": selected_verifier_command,
                    "reportPath": selected_report_path,
                },
                "rewards": dict(_DEFAULT_WEIGHTS),
                "limits": {
                    "toolCalls": 40,
                    "commandTimeoutSeconds": 120,
                    "episodeTimeoutSeconds": 900,
                    "networkAccess": False,
                },
            }
            # Use the runtime parser before writing any artifact. This catches
            # drift between the guided form and the manifest actually consumed.
            TaskManifest.from_mapping(payload)

            pack_root.mkdir(parents=True, exist_ok=True)
            task_directory = (pack_root / allocated_id).resolve()
            if task_directory.parent != pack_root or task_directory.exists():
                raise ConfigError("the allocated task directory is not safe to create")
            task_directory.mkdir()
            manifest_path = task_directory / "task.json"
            try:
                manifest_path.write_text(
                    json.dumps(payload, indent=2, sort_keys=False) + "\n",
                    encoding="utf-8",
                )
                # Config validation resolves local task-pack paths, so the
                # manifest must exist before validating the new declaration.
                updated = (
                    _updated_config(config, split, pack_root, pack_id)
                    if add_pack
                    else None
                )
                if updated is not None:
                    write_config(config.path, updated, overwrite=True)
            except Exception:
                _remove_created_directory(task_directory, pack_root)
                raise

    refreshed = load_config(config_path)
    return {
        "task": _serialize_manifest(manifest_path, payload, refreshed),
        **list_authored_tasks(config_path),
    }


def remove_authored_task(
    config_path: str | Path,
    *,
    split: str,
    task_id: str,
) -> dict[str, Any]:
    """Remove one guided task without accepting an arbitrary filesystem path."""

    selected_split = _text(split, "split", maximum=20)
    if selected_split not in _PACK_IDS:
        raise ConfigError("split must be train or evaluation")
    selected_id = _text(task_id, "task_id", maximum=100)
    if _slug(selected_id, fallback="task") != selected_id:
        raise ConfigError("task_id is invalid")

    with project_mutation_gate(config_path):
        with project_config_mutation(config_path):
            config = load_config(config_path)
            pack_root = _managed_root(config, selected_split)
            task_directory = (pack_root / selected_id).resolve()
            if task_directory.parent != pack_root:
                raise ConfigError("task_id is invalid")
            manifest_path = task_directory / "task.json"
            if not manifest_path.is_file():
                raise ConfigError(f"authored task does not exist: {selected_id}")
            try:
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                if not isinstance(payload, Mapping):
                    raise ValueError("manifest root is not an object")
                removed = _serialize_manifest(manifest_path, payload, config)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as error:
                raise ConfigError(f"authored task cannot be read: {error}") from error

            remaining = [
                path
                for path in pack_root.glob("*/task.json")
                if path.parent.resolve() != task_directory
            ]
            updated: dict[str, Any] | None = None
            if not remaining:
                managed_ids = {
                    str(source.get("id", ""))
                    for source in config.sources
                    if source.get("kind") == "task_pack"
                    and str(source.get("uri", ""))
                    and config.resolve_path(str(source.get("uri"))) == pack_root
                }
                if managed_ids:
                    updated = dict(config.data)
                    updated["sources"] = [
                        source
                        for source in config.sources
                        if str(source.get("id", "")) not in managed_ids
                    ]
                    if selected_split == "evaluation":
                        evaluation = dict(updated.get("evaluation", {}))
                        if str(evaluation.get("task_pack", "")) in managed_ids:
                            evaluation["task_pack"] = "held-out-frontend"
                        updated["evaluation"] = evaluation
                    report = validate_mapping(updated, root=config.root)
                    if report.errors:
                        raise ConfigError("\n".join(report.errors))

            # Rename first so rollback is lossless if the YAML update fails.
            # The project lease keeps Prepare from scanning the short-lived
            # tombstone directory as a second task.
            tombstone = pack_root / f".{selected_id}.deleting-{uuid4().hex[:8]}"
            task_directory.replace(tombstone)
            try:
                if updated is not None:
                    write_config(config.path, updated, overwrite=True)
            except Exception:
                tombstone.replace(task_directory)
                raise
            _remove_created_directory(tombstone, pack_root)

    return {"removed": removed, **list_authored_tasks(config_path)}


__all__ = ["create_authored_task", "list_authored_tasks", "remove_authored_task"]
