"""Static source inspection and deterministic source materialization.

This module deliberately does not import the training stack, download remote
content, or execute commands from a repository.  It answers the narrower and
more useful question: what evidence was declared, can it be read safely, and is
it shaped like data that a later SFT or RL stage can consume?
"""

from __future__ import annotations

import fnmatch
import glob
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from .manifest import TaskManifest


MAX_SOURCE_FILE_BYTES = 512 * 1024
REMOTE_PREFIXES = ("http://", "https://", "ssh://", "git://", "git@")
DEFAULT_FRONTEND_SUFFIXES = {
    ".css",
    ".html",
    ".js",
    ".json",
    ".jsonc",
    ".jsx",
    ".less",
    ".md",
    ".mdx",
    ".sass",
    ".scss",
    ".svelte",
    ".ts",
    ".tsx",
    ".vue",
}
DEFAULT_EXCLUDED_DIRECTORIES = {
    ".autotrainer",
    ".git",
    ".next",
    ".nuxt",
    ".output",
    ".turbo",
    ".vite",
    ".yarn",
    "artifacts",
    "build",
    "checkpoints",
    "coverage",
    "datasets",
    "dist",
    "model-cache",
    "node_modules",
    "rollouts",
    "runs",
    "vendor",
}
SECRET_PRONE_NAMES = {
    ".env",
    ".env.local",
    ".env.production",
    ".npmrc",
    ".pypirc",
    "credentials",
    "credentials.json",
    "id_dsa",
    "id_ed25519",
    "id_rsa",
    "secrets.json",
}
SECRET_PRONE_SUFFIXES = {".key", ".p12", ".pfx", ".pem"}


def _is_remote(uri: str) -> bool:
    lowered = uri.strip().lower()
    return lowered.startswith(REMOTE_PREFIXES)


def _as_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [str(item) for item in value if str(item).strip()]
    return []


def _source_specs(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return the canonical flat list, while tolerating early grouped drafts."""

    raw = config.get("sources", [])
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        return [dict(item) if isinstance(item, Mapping) else {"_invalid": item} for item in raw]
    if not isinstance(raw, Mapping):
        return [{"_invalid": raw}]

    grouped: list[dict[str, Any]] = []
    aliases = {
        "repositories": "repository",
        "sft": "sft_jsonl",
        "datasets": "sft_jsonl",
        "rlTasks": "task_pack",
        "rl_tasks": "task_pack",
        "taskPacks": "task_pack",
        "task_packs": "task_pack",
        "tasks": "task_pack",
    }
    for key, kind in aliases.items():
        values = raw.get(key, [])
        if isinstance(values, Mapping):
            values = [values]
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
            continue
        for value in values:
            if not isinstance(value, Mapping):
                grouped.append({"_invalid": value})
                continue
            item = dict(value)
            item.setdefault("kind", kind)
            if "uri" not in item:
                item["uri"] = item.get("path", item.get("url", item.get("glob", "")))
            grouped.append(item)
    return grouped


def _resolve_local(uri: str, project_root: Path) -> Path:
    candidate = Path(uri).expanduser()
    if not candidate.is_absolute():
        candidate = project_root / candidate
    return candidate.resolve()


def _display_path(path: Path, project_root: Path) -> str:
    try:
        return path.relative_to(project_root).as_posix()
    except ValueError:
        return str(path)


def _artifact_dir(config: Mapping[str, Any], project_root: Path) -> Path:
    project = config.get("project", {})
    if not isinstance(project, Mapping):
        project = {}
    configured = project.get(
        "artifact_dir",
        project.get("output_dir", config.get("artifact_dir", ".autotrainer")),
    )
    candidate = Path(str(configured)).expanduser()
    if not candidate.is_absolute():
        candidate = project_root / candidate
    return candidate.resolve()


def _git(repository: Path, *arguments: str) -> tuple[bool, str]:
    environment = os.environ.copy()
    environment.update({"GIT_OPTIONAL_LOCKS": "0", "GIT_TERMINAL_PROMPT": "0"})
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
    except FileNotFoundError:
        return False, "git executable was not found"
    except subprocess.TimeoutExpired:
        return False, "git command timed out"
    if completed.returncode:
        detail = (completed.stderr or completed.stdout).strip().replace("\r", " ").replace("\n", " ")
        return False, detail[-800:] or f"git exited with status {completed.returncode}"
    return True, completed.stdout.strip()


def _canonical_remote_host(hostname: str, port: int | None, scheme: str) -> str:
    """Normalize a remote host while retaining meaningful non-default ports."""

    host = hostname.rstrip(".").casefold()
    try:
        host = host.encode("idna").decode("ascii")
    except UnicodeError:
        # Hashing the case-folded Unicode host remains credential-safe. Git will
        # report an unusable URL later if the host itself is malformed.
        pass
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    defaults = {"http": 80, "https": 443, "ssh": 22, "git": 9418}
    return host if port is None or defaults.get(scheme) == port else f"{host}:{port}"


def _canonical_remote_path(value: str) -> str:
    """Normalize the repository portion shared by HTTPS and SSH spellings."""

    path = re.sub(r"/+", "/", unquote(value).replace("\\", "/")).strip("/")
    if path.casefold().endswith(".git"):
        path = path[:-4]
    return path.rstrip("/").casefold()


def _canonical_repository_locator(value: str, git_root: Path) -> str:
    """Return a credential-free locator shared by equivalent Git URL forms.

    Git has no repository UUID. V1 therefore treats the same normalized remote
    host/path as one repository across HTTPS, SSH URL, and SCP syntax. Userinfo,
    query strings, and fragments are intentionally discarded so access tokens
    cannot alter holdout identity or enter persisted source locks.
    """

    text = value.strip()
    parsed = urlsplit(text)
    scheme = parsed.scheme.casefold()
    if scheme in {"http", "https", "ssh", "git"} and parsed.hostname:
        try:
            port = parsed.port
        except ValueError:
            port = None
        host = _canonical_remote_host(parsed.hostname, port, scheme)
        return f"remote:{host}/{_canonical_remote_path(parsed.path)}"

    # SCP-style Git URLs have no `//`, so urllib treats their host as a scheme.
    # Check them explicitly after URL forms and before interpreting a relative
    # local path. A Windows drive path is excluded by its slash/backslash tail.
    scp = re.fullmatch(
        r"(?:(?:[^@/:\\]+)@)?(?P<host>\[[^\]]+\]|[^:/\\]+):(?P<path>[^\\].*)",
        text,
    )
    if scp and not re.fullmatch(r"[A-Za-z]", scp.group("host")):
        host_value = scp.group("host").strip("[]")
        host = _canonical_remote_host(host_value, None, "ssh")
        return f"remote:{host}/{_canonical_remote_path(scp.group('path'))}"

    if scheme == "file":
        path_text = unquote(parsed.path)
        if re.match(r"^/[A-Za-z]:/", path_text):
            path_text = path_text[1:]
        if parsed.netloc and parsed.netloc.casefold() != "localhost":
            path_text = f"//{parsed.netloc}{path_text}"
        local = Path(path_text).expanduser()
    else:
        local = Path(text).expanduser()
    if not local.is_absolute():
        local = git_root / local
    return "local:" + local.resolve().as_posix().casefold().rstrip("/")


def _repository_identity(repository: Path, git_root: Path) -> str:
    """Return one credential-safe identity for aliases or clones of a repository."""

    ok, origin = _git(repository, "remote", "get-url", "origin")
    if ok and origin.strip():
        value = origin.strip()
    else:
        # Linked worktrees have different working roots but one common object
        # store; treating them as distinct would let a path alias evade holdout.
        common_ok, common_directory = _git(
            repository,
            "rev-parse",
            "--path-format=absolute",
            "--git-common-dir",
        )
        value = (
            common_directory.strip()
            if common_ok and common_directory.strip()
            else git_root.resolve().as_posix()
        )
    canonical = _canonical_repository_locator(value, git_root)
    # Persist only the digest. Even if a future URL form is not recognized,
    # credentials and host paths never appear in the public scan artifact.
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _matches(path: str, patterns: Sequence[str]) -> bool:
    normalized = path.replace("\\", "/")
    for pattern in patterns:
        candidate = str(pattern).replace("\\", "/").lstrip("./")
        variants = {candidate}
        # ``fnmatch`` treats ``**`` like ``*`` and therefore requires at least
        # one directory for patterns such as ``src/**/*.tsx``.  Git-style
        # source globs conventionally allow that segment to match zero levels.
        pending = [candidate]
        while pending:
            value = pending.pop()
            if "/**/" in value:
                collapsed = value.replace("/**/", "/", 1)
                if collapsed not in variants:
                    variants.add(collapsed)
                    pending.append(collapsed)
        if candidate.startswith("**/"):
            variants.add(candidate[3:])
        if any(fnmatch.fnmatchcase(normalized, value) for value in variants):
            return True
        if candidate.endswith("/**") and normalized == candidate[:-3].rstrip("/"):
            return True
    return False


def _secret_prone(path: Path) -> bool:
    lowered = path.name.lower()
    return (
        lowered in SECRET_PRONE_NAMES
        or lowered.startswith(".env.")
        or path.suffix.lower() in SECRET_PRONE_SUFFIXES
    )


def _language(path: Path) -> str:
    return {
        ".css": "css",
        ".html": "html",
        ".js": "javascript",
        ".jsx": "jsx",
        ".md": "markdown",
        ".mdx": "mdx",
        ".scss": "scss",
        ".ts": "typescript",
        ".tsx": "tsx",
    }.get(path.suffix.lower(), path.suffix.lower().lstrip(".") or "text")


def _scan_repository_files(
    repository: Path,
    source: Mapping[str, Any],
    source_id: str,
    commit: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    include = _as_string_list(source.get("include"))
    exclude = _as_string_list(source.get("exclude"))
    metadata: list[dict[str, Any]] = []
    documents: list[dict[str, Any]] = []
    skipped = {
        "binary": 0,
        "excluded": 0,
        "oversized": 0,
        "secret_prone": 0,
        "symlink": 0,
        "unreadable": 0,
        "unsupported_extension": 0,
    }

    for current, directory_names, file_names in os.walk(repository, followlinks=False):
        current_path = Path(current)
        kept_directories: list[str] = []
        for name in sorted(directory_names):
            directory = current_path / name
            relative = directory.relative_to(repository).as_posix()
            if directory.is_symlink():
                skipped["symlink"] += 1
            elif name in DEFAULT_EXCLUDED_DIRECTORIES or _matches(relative, exclude):
                skipped["excluded"] += 1
            else:
                kept_directories.append(name)
        directory_names[:] = kept_directories

        for name in sorted(file_names):
            path = current_path / name
            relative = path.relative_to(repository).as_posix()
            if path.is_symlink():
                skipped["symlink"] += 1
                continue
            if _matches(relative, exclude):
                skipped["excluded"] += 1
                continue
            if include:
                if not _matches(relative, include):
                    skipped["excluded"] += 1
                    continue
            elif path.suffix.lower() not in DEFAULT_FRONTEND_SUFFIXES:
                skipped["unsupported_extension"] += 1
                continue
            if _secret_prone(path):
                skipped["secret_prone"] += 1
                continue
            try:
                size = path.stat().st_size
                if size > MAX_SOURCE_FILE_BYTES:
                    skipped["oversized"] += 1
                    continue
                raw = path.read_bytes()
            except OSError:
                skipped["unreadable"] += 1
                continue
            if b"\x00" in raw:
                skipped["binary"] += 1
                continue
            try:
                content = raw.decode("utf-8")
            except UnicodeDecodeError:
                skipped["binary"] += 1
                continue
            digest = hashlib.sha256(raw).hexdigest()
            item = {
                "bytes": len(raw),
                "language": _language(path),
                "path": relative,
                "sha256": digest,
            }
            metadata.append(item)
            document = {
                **item,
                "commit": commit,
                "source_id": source_id,
                "text": content,
            }
            if "license" in source:
                document["license"] = source["license"]
            documents.append(document)

    metadata.sort(key=lambda item: item["path"])
    documents.sort(key=lambda item: item["path"])
    return metadata, documents, skipped


def _base_result(source: Mapping[str, Any], index: int) -> dict[str, Any]:
    source_id = str(source.get("id", "")).strip() or f"source-{index + 1}"
    kind = str(source.get("kind", "")).strip()
    return {
        "errors": [],
        "id": source_id,
        "kind": kind,
        "partition": str(source.get("partition", "train")),
        "roles": _as_string_list(source.get("roles")),
        "status": "pending",
        "uri": str(source.get("uri", "")).strip(),
        "warnings": [],
    }


def _finish(result: dict[str, Any]) -> dict[str, Any]:
    result["errors"] = sorted(dict.fromkeys(str(item) for item in result["errors"]))
    result["warnings"] = sorted(dict.fromkeys(str(item) for item in result["warnings"]))
    if result["errors"]:
        result["status"] = "blocked"
    elif result.get("needs_materialization"):
        result["status"] = "needs_materialization"
    elif result["warnings"]:
        result["status"] = "warning"
    else:
        result["status"] = "ready"
    return result


def _scan_repository(
    source: Mapping[str, Any], index: int, project_root: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    result = _base_result(source, index)
    result.update(
        {
            "commit": None,
            "dirty": None,
            "dirty_entry_count": 0,
            "eligible_bytes": 0,
            "eligible_file_count": 0,
            "files": [],
            "repository_identity": None,
            "requested_revision": str(source.get("revision", "HEAD")),
            "skipped": {},
        }
    )
    uri = result["uri"]
    if not uri:
        result["errors"].append("repository uri is required")
        return _finish(result), []
    if _is_remote(uri):
        result["needs_materialization"] = True
        result["warnings"].append(
            "remote repository is declared but has not been cloned; run materialization with network access"
        )
        return _finish(result), []

    repository = _resolve_local(uri, project_root)
    result["resolved_uri"] = _display_path(repository, project_root)
    if not repository.is_dir():
        result["errors"].append(f"repository does not exist or is not a directory: {repository}")
        return _finish(result), []

    ok, git_root_value = _git(repository, "rev-parse", "--show-toplevel")
    if not ok:
        result["errors"].append(f"cannot inspect repository with git: {git_root_value}")
        return _finish(result), []
    git_root = Path(git_root_value).resolve()
    result["git_root"] = _display_path(git_root, project_root)
    result["repository_identity"] = _repository_identity(repository, git_root)

    revision = result["requested_revision"] or "HEAD"
    ok, commit = _git(repository, "rev-parse", "--verify", f"{revision}^{{commit}}")
    if not ok:
        result["errors"].append(f"cannot resolve revision {revision!r}: {commit}")
        return _finish(result), []
    result["commit"] = commit
    ok, head = _git(repository, "rev-parse", "HEAD")
    if not ok:
        result["errors"].append(f"cannot resolve checked-out HEAD: {head}")
        return _finish(result), []
    if head != commit:
        result["errors"].append(
            f"requested revision resolves to {commit}, but the local checkout is {head}; check out the requested revision first"
        )
        return _finish(result), []

    ok, porcelain = _git(repository, "status", "--porcelain", "--untracked-files=normal")
    if not ok:
        result["errors"].append(f"cannot inspect repository status: {porcelain}")
        return _finish(result), []
    dirty_entries = [line for line in porcelain.splitlines() if line.strip()]
    result["dirty"] = bool(dirty_entries)
    result["dirty_entry_count"] = len(dirty_entries)
    if dirty_entries:
        result["warnings"].append(
            "repository has uncommitted or untracked files; its commit alone cannot reproduce this scan"
        )
    if revision in {"HEAD", "main", "master", "latest"} or not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        result["warnings"].append(
            f"requested revision {revision!r} is mutable or abbreviated; the lock records resolved commit {commit}"
        )

    files, documents, skipped = _scan_repository_files(repository, source, result["id"], commit)
    result["files"] = files
    result["eligible_file_count"] = len(files)
    result["eligible_bytes"] = sum(item["bytes"] for item in files)
    result["skipped"] = skipped
    if skipped["secret_prone"]:
        result["warnings"].append(
            f"skipped {skipped['secret_prone']} secret-prone file(s); this is not a complete secret scan"
        )
    if not files:
        result["errors"].append("repository contains no eligible frontend text files")
    return _finish(result), documents


def _validate_message_list(value: Any, *, require_assistant: bool) -> str | None:
    if not isinstance(value, list) or not value:
        return "must be a non-empty message list"
    roles: set[str] = set()
    for index, message in enumerate(value):
        if not isinstance(message, Mapping):
            return f"message {index} must be an object"
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant", "tool"}:
            return f"message {index}.role is invalid"
        if not isinstance(content, str) or not content.strip():
            return f"message {index}.content must be non-empty text"
        roles.add(str(role))
    if "user" not in roles:
        return "must contain a user message"
    if require_assistant and "assistant" not in roles:
        return "must contain an assistant message"
    return None


def _validate_sft_record(record: Any) -> tuple[str | None, str | None]:
    if not isinstance(record, Mapping):
        return None, "record must be a JSON object"
    if "messages" in record:
        error = _validate_message_list(record["messages"], require_assistant=True)
        return "messages", error
    if "prompt" in record and "completion" in record:
        prompt = record["prompt"]
        completion = record["completion"]
        if isinstance(prompt, str) and isinstance(completion, str):
            if not prompt.strip() or not completion.strip():
                return "prompt_completion", "prompt and completion must be non-empty text"
            return "prompt_completion", None
        prompt_error = _validate_message_list(prompt, require_assistant=False)
        if prompt_error:
            return "conversational_prompt_completion", f"prompt {prompt_error}"
        completion_error = _validate_message_list(completion, require_assistant=True)
        if completion_error:
            return "conversational_prompt_completion", f"completion {completion_error}"
        return "conversational_prompt_completion", None
    return None, "record must contain messages or prompt and completion"


def _scan_sft_jsonl(source: Mapping[str, Any], index: int, project_root: Path) -> dict[str, Any]:
    result = _base_result(source, index)
    result.update(
        {
            "bytes": 0,
            "format_counts": {},
            "invalid_record_count": 0,
            "sha256": None,
            "valid_record_count": 0,
        }
    )
    uri = result["uri"]
    if not uri:
        result["errors"].append("SFT JSONL uri is required")
        return _finish(result)
    if _is_remote(uri):
        result["needs_materialization"] = True
        result["warnings"].append("remote SFT data is not downloaded by static source inspection")
        return _finish(result)
    path = _resolve_local(uri, project_root)
    result["resolved_uri"] = _display_path(path, project_root)
    if not path.is_file():
        result["errors"].append(f"SFT JSONL does not exist or is not a file: {path}")
        return _finish(result)
    if path.suffix.lower() != ".jsonl":
        result["errors"].append("SFT data must use the .jsonl extension")
        return _finish(result)
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        result["errors"].append(f"SFT JSONL is not UTF-8 text: byte {error.start}")
        return _finish(result)
    except OSError as error:
        result["errors"].append(f"cannot read SFT JSONL: {error}")
        return _finish(result)

    result["bytes"] = len(raw)
    result["sha256"] = hashlib.sha256(raw).hexdigest()
    format_counts: dict[str, int] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            result["invalid_record_count"] += 1
            result["errors"].append(
                f"{path}:{line_number}:{error.colno}: invalid JSON: {error.msg}"
            )
            continue
        record_format, validation_error = _validate_sft_record(record)
        if validation_error:
            result["invalid_record_count"] += 1
            result["errors"].append(f"{path}:{line_number}: {validation_error}")
            continue
        result["valid_record_count"] += 1
        assert record_format is not None
        format_counts[record_format] = format_counts.get(record_format, 0) + 1
    result["format_counts"] = dict(sorted(format_counts.items()))
    if not result["valid_record_count"]:
        result["errors"].append("SFT JSONL contains no valid training records")
    if result["partition"] != "train":
        result["warnings"].append("SFT JSONL is not in the train partition")
    return _finish(result)


def _task_files(uri: str, project_root: Path) -> tuple[list[Path], str | None]:
    if any(character in uri for character in "*?["):
        pattern = Path(uri).expanduser()
        if not pattern.is_absolute():
            pattern = project_root / pattern
        paths = [Path(value).resolve() for value in glob.glob(str(pattern), recursive=True)]
        return sorted(path for path in paths if path.is_file() and path.suffix.lower() == ".json"), None
    path = _resolve_local(uri, project_root)
    if path.is_dir():
        return sorted(item.resolve() for item in path.rglob("*.json") if item.is_file()), None
    if path.is_file() and path.suffix.lower() == ".json":
        return [path], None
    return [], f"task pack does not resolve to a JSON file, directory, or JSON glob: {path}"


def _explicit_verifier(payload: Mapping[str, Any]) -> Any:
    for container in (payload, payload.get("task", {})):
        if not isinstance(container, Mapping):
            continue
        for key in ("verifier", "verification", "hiddenTests", "hidden_tests"):
            if key in container:
                return container[key]
    return None


def _verifier_state(
    payload: Mapping[str, Any], source: Mapping[str, Any], manifest_path: Path, project_root: Path
) -> tuple[bool, bool | None, str, str | None]:
    explicit = _explicit_verifier(payload)
    if explicit is not None:
        value = explicit
        if isinstance(explicit, Mapping):
            value = explicit.get("path", explicit.get("uri", explicit.get("bundle")))
        if isinstance(value, str) and value.strip():
            candidate = Path(value).expanduser()
            if not candidate.is_absolute():
                local_candidate = (manifest_path.parent / candidate).resolve()
                project_candidate = (project_root / candidate).resolve()
                candidate = local_candidate if local_candidate.exists() else project_candidate
            else:
                candidate = candidate.resolve()
            exists = candidate.exists()
            return True, exists, "resolved" if exists else "missing", str(candidate)
        return True, False, "invalid", None

    commands: list[Any] = []
    for runtime in (payload.get("runtime", {}), source.get("runtime", {})):
        if not isinstance(runtime, Mapping):
            continue
        commands.extend(
            runtime.get(key)
            for key in ("tests", "browserTests", "test", "browser_test")
            if runtime.get(key)
        )
    if commands:
        return True, None, "declared_command", None
    return False, None, "missing", None


def _resolve_snapshot(
    payload: Mapping[str, Any], snapshot: str, repositories: Mapping[str, dict[str, Any]]
) -> tuple[str | None, str | None, bool, str | None]:
    task = payload.get("task", {})
    source_id: str | None = None
    revision: str | None = None
    if isinstance(task, Mapping):
        declared_source = task.get("sourceId", task.get("source_id"))
        declared_revision = task.get("startingRevision", task.get("starting_revision"))
        if declared_source:
            source_id = str(declared_source)
            revision = str(declared_revision or snapshot)
    if source_id is None and "@" in snapshot:
        source_id, revision = snapshot.rsplit("@", 1)
    if not source_id or not revision:
        return source_id, revision, False, "startingSnapshot must be <repository-source-id>@<git-revision>"
    repository = repositories.get(source_id)
    if repository is None:
        return source_id, revision, False, f"startingSnapshot references undeclared repository source {source_id!r}"
    if repository.get("needs_materialization"):
        return source_id, revision, False, f"repository source {source_id!r} needs materialization"
    resolved_uri = repository.get("_absolute_uri")
    if not resolved_uri:
        return source_id, revision, False, f"repository source {source_id!r} is not locally resolvable"
    if revision == "locked":
        commit = repository.get("commit")
        if commit:
            return source_id, str(commit), True, None
        return source_id, revision, False, f"repository source {source_id!r} has no resolved commit"
    ok, detail = _git(Path(resolved_uri), "rev-parse", "--verify", f"{revision}^{{commit}}")
    if not ok:
        return source_id, revision, False, f"cannot resolve snapshot revision {revision!r}: {detail}"
    return source_id, detail, True, None


def _scan_task_pack(
    source: Mapping[str, Any],
    index: int,
    project_root: Path,
    repositories: Mapping[str, dict[str, Any]],
) -> dict[str, Any]:
    result = _base_result(source, index)
    result.update(
        {
            "evaluation_task_count": 0,
            "ready_task_count": 0,
            "task_count": 0,
            "tasks": [],
            "train_task_count": 0,
        }
    )
    uri = result["uri"]
    if not uri:
        result["errors"].append("task pack uri is required")
        return _finish(result)
    if _is_remote(uri):
        result["needs_materialization"] = True
        result["warnings"].append("remote task packs are not downloaded by static source inspection")
        return _finish(result)
    paths, discovery_error = _task_files(uri, project_root)
    result["resolved_uri"] = _display_path(_resolve_local(uri, project_root), project_root)
    if discovery_error:
        result["errors"].append(discovery_error)
        return _finish(result)
    if not paths:
        result["errors"].append("task pack contains no JSON task manifests")
        return _finish(result)

    for path in paths:
        item: dict[str, Any] = {
            "errors": [],
            "manifest": _display_path(path, project_root),
            "ready": False,
            "warnings": [],
        }
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, Mapping):
                raise ValueError("manifest root must be a JSON object")
            manifest = TaskManifest.from_mapping(payload)
        except (OSError, json.JSONDecodeError, TypeError, ValueError, AttributeError) as error:
            item["errors"].append(f"invalid task manifest: {error}")
            result["errors"].append(f"{path}: {item['errors'][0]}")
            result["tasks"].append(item)
            continue

        item.update(
            {
                "split": manifest.split,
                "starting_snapshot": manifest.starting_snapshot,
                "task_id": manifest.task_id,
            }
        )
        result["task_count"] += 1
        if manifest.split == "train":
            result["train_task_count"] += 1
        else:
            result["evaluation_task_count"] += 1
        if manifest.split != result["partition"]:
            item["errors"].append(
                f"task split {manifest.split!r} does not match source partition {result['partition']!r}"
            )

        source_id, resolved_revision, snapshot_resolved, snapshot_error = _resolve_snapshot(
            payload, manifest.starting_snapshot, repositories
        )
        item["snapshot_source_id"] = source_id
        item["snapshot_revision"] = resolved_revision
        item["snapshot_resolved"] = snapshot_resolved
        if snapshot_error:
            item["errors"].append(snapshot_error)

        verifier_declared, verifier_resolved, verifier_status, verifier_path = _verifier_state(
            payload, source, path, project_root
        )
        item["verifier_declared"] = verifier_declared
        item["verifier_resolved"] = verifier_resolved
        item["verifier_status"] = verifier_status
        if verifier_path:
            item["verifier_path"] = verifier_path
        if not verifier_declared:
            item["errors"].append("no verifier path or test/browser-test command is declared")
        elif verifier_resolved is False:
            item["errors"].append(f"declared verifier is not resolvable: {verifier_path or 'invalid value'}")
        elif verifier_resolved is None:
            item["warnings"].append(
                "verifier command is declared but static inspection does not execute it"
            )

        item["errors"] = sorted(dict.fromkeys(item["errors"]))
        item["warnings"] = sorted(dict.fromkeys(item["warnings"]))
        item["ready"] = not item["errors"]
        if item["ready"]:
            result["ready_task_count"] += 1
        else:
            result["errors"].extend(f"{path}: {message}" for message in item["errors"])
        result["warnings"].extend(f"{path}: {message}" for message in item["warnings"])
        result["tasks"].append(item)

    return _finish(result)


def _safe_artifact_name(source_id: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", source_id).strip("-.")
    return name or hashlib.sha256(source_id.encode("utf-8")).hexdigest()[:12]


def _clone_repository(uri: str, destination: Path) -> tuple[bool, str]:
    environment = os.environ.copy()
    environment.update({"GIT_OPTIONAL_LOCKS": "0", "GIT_TERMINAL_PROMPT": "0"})
    try:
        completed = subprocess.run(
            [
                "git",
                "clone",
                "--quiet",
                "--no-checkout",
                "--no-hardlinks",
                "--no-recurse-submodules",
                uri,
                str(destination),
            ],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=300,
            env=environment,
        )
    except FileNotFoundError:
        return False, "git executable was not found"
    except subprocess.TimeoutExpired:
        return False, "git clone timed out"
    if completed.returncode:
        detail = (completed.stderr or completed.stdout).strip().replace("\r", " ").replace("\n", " ")
        return False, detail[-1000:] or f"git clone exited with status {completed.returncode}"
    return True, ""


def _remove_materialization(path: Path) -> None:
    """Remove only a destination created for the current failed operation."""

    def retry_read_only(function: Any, value: str, _error: Any) -> None:
        if os.path.islink(value):
            os.unlink(value)
            return
        mode = stat.S_IRUSR | stat.S_IWUSR
        if os.path.isdir(value):
            mode |= stat.S_IXUSR
        os.chmod(value, mode)
        function(value)

    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.exists():
        shutil.rmtree(path, onerror=retry_read_only)


def _materialization_destination(
    config: Mapping[str, Any], project_root: Path, source_id: str, destination: Path | None
) -> tuple[Path, Path]:
    source_root = (_artifact_dir(config, project_root) / "sources").resolve()
    if destination is None:
        candidate = source_root / _safe_artifact_name(source_id)
    else:
        supplied = Path(destination).expanduser()
        candidate = supplied if supplied.is_absolute() else source_root / supplied
    # Keep a lexical absolute path long enough to detect destination symlinks.
    # Resolving first would silently turn ``sources/name -> another-place`` into
    # the target path and defeat the explicit no-symlink/no-overwrite checks.
    candidate = Path(os.path.abspath(candidate))
    try:
        candidate.relative_to(source_root)
    except ValueError as error:
        raise ValueError(
            f"materialization destination must stay inside {source_root}: {candidate}"
        ) from error
    if candidate == source_root:
        raise ValueError("materialization destination must be below the artifact sources directory")

    relative = candidate.relative_to(source_root)
    current = source_root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"materialization destination must not traverse a symlink: {current}")

    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(source_root)
    except ValueError as error:
        raise ValueError(
            f"materialization destination must stay inside {source_root}: {resolved}"
        ) from error
    if resolved != candidate:
        raise ValueError(f"materialization destination must not traverse a symlink: {candidate}")
    return source_root, resolved


def materialize_repository(
    config: Mapping[str, Any],
    project_root: Path,
    source_id: str,
    destination: Path | None = None,
) -> dict[str, Any]:
    """Clone and pin one declared repository source into the project artifacts.

    Relative ``destination`` values are resolved below
    ``project.artifact_dir/sources``.  The clone is performed with Git argv and
    no shell, submodules are not initialized, repository symlinks are rejected,
    and the checkout is detached at the resolved commit.  The returned
    ``updated_source`` remains compatible with the source declaration schema so
    a caller can persist it explicitly.
    """

    if not isinstance(config, Mapping):
        raise ValueError("config must be a mapping")
    root = Path(project_root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"project root does not exist or is not a directory: {root}")
    requested_id = str(source_id).strip()
    if not requested_id:
        raise ValueError("source_id must be a non-empty string")

    matching = [
        source
        for source in _source_specs(config)
        if "_invalid" not in source and str(source.get("id", "")).strip() == requested_id
    ]
    if not matching:
        raise ValueError(f"repository source is not declared: {requested_id}")
    if len(matching) > 1:
        raise ValueError(f"source id is declared more than once: {requested_id}")
    source = matching[0]
    if source.get("kind") != "repository":
        raise ValueError(f"source {requested_id!r} is not a repository")
    uri = str(source.get("uri", "")).strip()
    if not uri:
        raise ValueError(f"repository source {requested_id!r} has no uri")
    revision = str(source.get("revision", "")).strip()
    if not revision:
        raise ValueError(
            f"repository source {requested_id!r} must declare a revision before materialization"
        )

    clone_uri = uri
    if not _is_remote(uri) and not uri.lower().startswith("file://"):
        supplied_source = Path(uri).expanduser()
        if not supplied_source.is_absolute():
            supplied_source = root / supplied_source
        supplied_source = Path(os.path.abspath(supplied_source))
        if supplied_source.is_symlink():
            raise ValueError(f"repository source uri must not be a symlink: {supplied_source}")
        local_source = supplied_source.resolve()
        if not local_source.is_dir():
            raise ValueError(f"repository source does not exist or is not a directory: {local_source}")
        clone_uri = str(local_source)

    source_root, local_path = _materialization_destination(
        config, root, requested_id, destination
    )
    if local_path.is_symlink():
        raise ValueError(f"refusing to overwrite symlink destination: {local_path}")
    if local_path.exists():
        raise ValueError(f"refusing to overwrite materialized repository: {local_path}")

    source_root.mkdir(parents=True, exist_ok=True)
    clone_started = False
    try:
        clone_started = True
        ok, detail = _clone_repository(clone_uri, local_path)
        if not ok:
            raise RuntimeError(f"cannot clone repository source {requested_id!r}: {detail}")

        ok, commit = _git(local_path, "rev-parse", "--verify", f"{revision}^{{commit}}")
        if not ok:
            raise RuntimeError(
                f"cannot resolve declared revision {revision!r} for source {requested_id!r}: {commit}"
            )
        if not re.fullmatch(r"[0-9a-fA-F]{40,64}", commit):
            raise RuntimeError(f"git returned a non-immutable commit identifier: {commit!r}")

        ok, tree = _git(local_path, "ls-tree", "-r", "-z", commit)
        if not ok:
            raise RuntimeError(f"cannot inspect repository tree at {commit}: {tree}")
        symlinks = []
        for entry in tree.split("\x00"):
            if not entry:
                continue
            metadata, separator, path_value = entry.partition("\t")
            if separator and metadata.startswith("120000 "):
                symlinks.append(path_value)
        if symlinks:
            preview = ", ".join(sorted(symlinks)[:5])
            raise ValueError(
                "repository tree contains symlinks, which are not supported by V1 "
                f"materialization: {preview}"
            )

        ok, checkout_detail = _git(
            local_path,
            "-c",
            "core.autocrlf=false",
            "checkout",
            "--quiet",
            "--detach",
            "--force",
            commit,
        )
        if not ok:
            raise RuntimeError(f"cannot check out resolved commit {commit}: {checkout_detail}")
        ok, head = _git(local_path, "rev-parse", "HEAD")
        if not ok or head != commit:
            raise RuntimeError(
                f"detached checkout verification failed: expected {commit}, got {head or 'unresolved'}"
            )
        ok, _branch = _git(local_path, "symbolic-ref", "--quiet", "--short", "HEAD")
        if ok:
            raise RuntimeError("materialized repository checkout is not detached")

        updated_source = dict(source)
        updated_source["uri"] = _display_path(local_path, root)
        updated_source["revision"] = commit.lower()
        return {
            "source_id": requested_id,
            "local_path": str(local_path),
            "commit": commit.lower(),
            "requested_revision": revision,
            "updated_source": updated_source,
        }
    except Exception:
        if clone_started and (local_path.exists() or local_path.is_symlink()):
            _remove_materialization(local_path)
        raise


def _write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        # Unique same-directory files keep simultaneous read-only validation and
        # artifact-producing commands from contending for one predictable name.
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        for attempt in range(12):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                # Windows can transiently deny two simultaneous replacements of
                # the same target even after both writers closed their handles.
                if attempt == 11:
                    raise
                time.sleep(0.01)
    finally:
        temporary.unlink(missing_ok=True)


def _write_artifacts(
    config: Mapping[str, Any],
    project_root: Path,
    scan: dict[str, Any],
    documents: Mapping[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    artifact_dir = _artifact_dir(config, project_root)
    ingested_dir = artifact_dir / "ingested"
    artifacts: dict[str, Any] = {"documents": {}}
    for source_id in sorted(documents):
        destination = ingested_dir / f"{_safe_artifact_name(source_id)}.documents.jsonl"
        lines = [
            json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            for item in documents[source_id]
        ]
        _write_text_atomic(destination, "\n".join(lines) + ("\n" if lines else ""))
        artifacts["documents"][source_id] = str(destination)

    lock = {
        "schema_version": 1,
        "sources": scan["sources"],
        "summary": scan["summary"],
    }
    lock_path = artifact_dir / "sources.lock.json"
    _write_text_atomic(
        lock_path,
        json.dumps(lock, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    artifacts["lock"] = str(lock_path)
    return artifacts


def scan_sources(
    config: Mapping[str, Any], project_root: Path, write: bool = False
) -> dict[str, Any]:
    """Inspect declared sources without importing ML libraries or executing repo code.

    When ``write`` is true, deterministic inventories are written below
    ``project.artifact_dir`` (``.autotrainer`` by default).  Remote sources are
    intentionally not downloaded here; they are reported as needing
    materialization so network access remains an explicit operation.
    """

    root = Path(project_root).expanduser().resolve()
    scan: dict[str, Any] = {
        "errors": [],
        "project_root": str(root),
        "schema_version": 1,
        "sources": [],
        "summary": {},
        "warnings": [],
    }
    if not isinstance(config, Mapping):
        scan["errors"].append("config must be a mapping")
        return scan
    if not root.is_dir():
        scan["errors"].append(f"project root does not exist or is not a directory: {root}")
        return scan

    specs = _source_specs(config)
    if not specs:
        scan["warnings"].append("no sources are declared")
    results: list[dict[str, Any] | None] = [None] * len(specs)
    documents: dict[str, list[dict[str, Any]]] = {}

    for index, source in enumerate(specs):
        if "_invalid" in source:
            result = _base_result(source, index)
            result["errors"].append("source declaration must be a mapping")
            results[index] = _finish(result)
            continue
        kind = str(source.get("kind", ""))
        if kind == "repository":
            result, source_documents = _scan_repository(source, index, root)
            if result.get("resolved_uri"):
                result["_absolute_uri"] = str(_resolve_local(result["uri"], root))
            results[index] = result
            if source_documents:
                documents[result["id"]] = source_documents
        elif kind == "sft_jsonl":
            results[index] = _scan_sft_jsonl(source, index, root)
        elif kind != "task_pack":
            result = _base_result(source, index)
            result["errors"].append(
                "unsupported source kind; expected repository, sft_jsonl, or task_pack"
            )
            results[index] = _finish(result)

    repositories = {
        str(result["id"]): result
        for result in results
        if result is not None and result.get("kind") == "repository"
    }
    for index, source in enumerate(specs):
        if results[index] is None and str(source.get("kind", "")) == "task_pack":
            results[index] = _scan_task_pack(source, index, root, repositories)

    finalized: list[dict[str, Any]] = []
    for result in results:
        assert result is not None
        clean = dict(result)
        clean.pop("_absolute_uri", None)
        finalized.append(clean)
        scan["errors"].extend(f"{clean['id']}: {message}" for message in clean["errors"])
        scan["warnings"].extend(f"{clean['id']}: {message}" for message in clean["warnings"])
    scan["sources"] = finalized

    repositories_found = [item for item in finalized if item["kind"] == "repository"]
    sft_found = [item for item in finalized if item["kind"] == "sft_jsonl"]
    tasks_found = [item for item in finalized if item["kind"] == "task_pack"]
    scan["summary"] = {
        "blocked_source_count": sum(item["status"] == "blocked" for item in finalized),
        "eligible_repository_file_count": sum(
            int(item.get("eligible_file_count", 0)) for item in repositories_found
        ),
        "evaluation_ready_task_count": sum(
            sum(
                bool(task.get("ready")) and task.get("split") == "evaluation"
                for task in item.get("tasks", [])
            )
            for item in tasks_found
        ),
        "needs_materialization_count": sum(
            item["status"] == "needs_materialization" for item in finalized
        ),
        "repository_count": len(repositories_found),
        "sft_source_count": len(sft_found),
        "source_count": len(finalized),
        "task_pack_count": len(tasks_found),
        "train_ready_task_count": sum(
            sum(
                bool(task.get("ready")) and task.get("split") == "train"
                for task in item.get("tasks", [])
            )
            for item in tasks_found
        ),
        "valid_sft_record_count": sum(
            int(item.get("valid_record_count", 0)) for item in sft_found
        ),
    }
    scan["errors"] = sorted(dict.fromkeys(scan["errors"]))
    scan["warnings"] = sorted(dict.fromkeys(scan["warnings"]))
    if write:
        scan["artifacts"] = _write_artifacts(config, root, scan, documents)
    return scan


__all__ = ["materialize_repository", "scan_sources"]
