"""Turn reviewed Git changes into deterministic supervised-data candidates.

Repository contents are context, not labels.  This module therefore stops at
the narrow boundary that V1 can defend: inspect a pinned repository's
first-parent history, retain small text-only changes, and require an explicit
human review before exposing an accepted patch to the later compiler.

No repository command, hook, text conversion, external diff, or network
operation is executed.  Generated candidates and review decisions remain
under the project's ignored artifact directory.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import re
import tempfile
import threading
import time
import unicodedata
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import unquote, urlsplit


SCHEMA_VERSION = 1
EXTRACTOR_VERSION = 1
MAX_HISTORY_COMMITS = 200
MAX_REVIEWABLE_CANDIDATES = 50
MAX_CHANGED_FILES = 4
MAX_CHANGED_LINES = 160
MAX_TRAINING_CHARS = 4_096
MAX_INSTRUCTION_CHARS = 500
MIN_INSTRUCTION_CHARS = 12
MAX_GIT_OUTPUT_BYTES = 4 * 1024 * 1024

SUPPORTED_SUFFIXES = {
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
EXCLUDED_DIRECTORIES = {
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
GENERATED_NAMES = {
    "bun.lock",
    "bun.lockb",
    "cargo.lock",
    "composer.lock",
    "package-lock.json",
    "pnpm-lock.yaml",
    "poetry.lock",
    "yarn.lock",
}
GENERATED_SUFFIXES = (".map", ".min.css", ".min.js", ".snap")
GENERIC_INSTRUCTIONS = {
    "change",
    "changes",
    "cleanup",
    "fix",
    "fixed",
    "misc",
    "stuff",
    "update",
    "updates",
    "wip",
}
TRAILER_PATTERN = re.compile(
    r"^(?:co-authored-by|signed-off-by|reviewed-by|acked-by|tested-by|"
    r"reported-by|helped-by|cc|fixes|closes):\s*.+$",
    re.IGNORECASE,
)
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40,64}$", re.IGNORECASE)
REMOTE_PREFIXES = ("http://", "https://", "ssh://", "git://", "git@")

# These patterns intentionally target well-known credential shapes rather than
# guessing at every high-entropy string in source code.  The result is a
# conservative safety gate, not a claim of complete secret detection.
SECRET_PATTERNS = (
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{30,}\b"),
    re.compile(
        r"(?i)[\"']?(?:api[_-]?key|access[_-]?token|auth[_-]?token|password|secret)"
        r"[\"']?\s*[:=]\s*[\"']?[A-Za-z0-9_./+=-]{12,}"
    ),
)

SYSTEM_PROMPT = (
    "Return only a focused unified diff for the supplied task and repository context."
)

_REVIEW_LOCK = threading.RLock()


class HistoryError(ValueError):
    """A user-correctable history discovery or review error."""


def _coerce_project(
    config_or_path: Mapping[str, Any] | str | Path,
    project_root: Path | None,
) -> tuple[dict[str, Any], Path]:
    if isinstance(config_or_path, Mapping):
        if project_root is None:
            raise HistoryError("project_root is required when config is a mapping")
        data = dict(config_or_path)
        root = Path(project_root).expanduser().resolve()
    else:
        # The import stays local so the history core has no configuration-module
        # import cycle when source scanning is wired to it later.
        from .config import load_config

        loaded = load_config(config_or_path)
        data = dict(loaded.data)
        root = loaded.root.resolve()
    if not root.is_dir():
        raise HistoryError("project root does not exist or is not a directory")
    return data, root


def _artifact_dir(config: Mapping[str, Any], root: Path) -> Path:
    project = config.get("project", {})
    value = project.get("artifact_dir", ".autotrainer") if isinstance(project, Mapping) else ".autotrainer"
    candidate = Path(str(value)).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()


def _safe_artifact_name(value: str) -> str:
    candidate = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-.")
    return candidate or hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def _source_specs(config: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    value = config.get("sources", [])
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _training_char_limit(config: Mapping[str, Any]) -> int:
    sft = config.get("sft", {})
    raw = sft.get("max_length", 2_048) if isinstance(sft, Mapping) else 2_048
    try:
        max_length = int(raw)
    except (TypeError, ValueError):
        max_length = 2_048
    # Character count is a dependency-free conservative prefilter. The SFT
    # runtime must still perform exact tokenizer-length validation later.
    return min(MAX_TRAINING_CHARS, max(1, max_length) * 2)


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [str(item) for item in value if str(item).strip()]
    return []


def _resolve_repository(source: Mapping[str, Any], root: Path) -> Path:
    uri = str(source.get("uri", "")).strip()
    if not uri:
        raise HistoryError("repository uri is required")
    if uri.casefold().startswith(REMOTE_PREFIXES):
        raise HistoryError("repository must be materialized locally before history review")
    candidate = Path(uri).expanduser()
    repository = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    if not repository.is_dir():
        raise HistoryError("repository does not exist or is not a directory")
    return repository


def _git_bytes(
    repository: Path,
    *arguments: str,
    max_output_bytes: int = MAX_GIT_OUTPUT_BYTES,
) -> bytes:
    """Run a read-only Git command without shell, hooks, pagers, or diff drivers."""

    environment = os.environ.copy()
    environment.update(
        {
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_PAGER": "cat",
            "GIT_TERMINAL_PROMPT": "0",
            "LANG": "C",
            "LC_ALL": "C",
            "PAGER": "cat",
        }
    )
    try:
        import subprocess

        # Temporary files prevent a malicious repository from forcing Python to
        # retain unbounded Git output in memory before the size check runs.
        with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
            completed = subprocess.run(
                [
                    "git",
                    "-c",
                    f"safe.directory={repository.as_posix()}",
                    "-c",
                    "color.ui=false",
                    "-c",
                    "core.quotepath=false",
                    "-c",
                    "i18n.logOutputEncoding=utf-8",
                    "-C",
                    str(repository),
                    *arguments,
                ],
                check=False,
                stdout=stdout,
                stderr=stderr,
                timeout=20,
                env=environment,
            )
            stdout_size = stdout.tell()
            if stdout_size > max_output_bytes:
                raise HistoryError("git history output exceeded the V1 safety limit")
            stdout.seek(0)
            output = stdout.read()
            stderr.seek(0)
            error_output = stderr.read(64 * 1024)
    except FileNotFoundError as error:
        raise HistoryError("git executable was not found") from error
    except subprocess.TimeoutExpired as error:
        raise HistoryError("git history inspection timed out") from error
    if completed.returncode:
        detail = error_output.decode("utf-8", errors="replace").strip()
        detail = detail.replace("\r", " ").replace("\n", " ")[-600:]
        raise HistoryError(detail or f"git exited with status {completed.returncode}")
    return output


def _git_text(repository: Path, *arguments: str) -> str:
    return _git_bytes(repository, *arguments).decode("utf-8", errors="replace").strip()


def _canonical_remote_host(hostname: str, port: int | None, scheme: str) -> str:
    host = hostname.rstrip(".").casefold()
    try:
        host = host.encode("idna").decode("ascii")
    except UnicodeError:
        pass
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    defaults = {"http": 80, "https": 443, "ssh": 22, "git": 9418}
    return host if port is None or defaults.get(scheme) == port else f"{host}:{port}"


def _canonical_remote_path(value: str) -> str:
    path = re.sub(r"/+", "/", unquote(value).replace("\\", "/")).strip("/")
    if path.casefold().endswith(".git"):
        path = path[:-4]
    return path.rstrip("/").casefold()


def _canonical_locator(value: str, git_root: Path) -> str:
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
    scp = re.fullmatch(
        r"(?:(?:[^@/:\\]+)@)?(?P<host>\[[^\]]+\]|[^:/\\]+):(?P<path>[^\\].*)",
        text,
    )
    if scp and not re.fullmatch(r"[A-Za-z]", scp.group("host")):
        host = _canonical_remote_host(scp.group("host").strip("[]"), None, "ssh")
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
    try:
        locator = _git_text(repository, "remote", "get-url", "origin")
    except HistoryError:
        locator = ""
    if not locator:
        try:
            locator = _git_text(
                repository,
                "rev-parse",
                "--path-format=absolute",
                "--git-common-dir",
            )
        except HistoryError:
            locator = git_root.as_posix()
    canonical = _canonical_locator(locator, git_root)
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _match_glob_parts(path_parts: Sequence[str], pattern_parts: Sequence[str]) -> bool:
    if not pattern_parts:
        return not path_parts
    pattern = pattern_parts[0]
    if pattern == "**":
        return _match_glob_parts(path_parts, pattern_parts[1:]) or bool(
            path_parts and _match_glob_parts(path_parts[1:], pattern_parts)
        )
    return bool(
        path_parts
        and fnmatch.fnmatchcase(path_parts[0], pattern)
        and _match_glob_parts(path_parts[1:], pattern_parts[1:])
    )


def _matches(path: str, patterns: Sequence[str]) -> bool:
    """Match Git-style globs without allowing ``*`` to cross directories."""

    path_parts = tuple(path.replace("\\", "/").split("/"))
    for raw_pattern in patterns:
        candidate = str(raw_pattern).replace("\\", "/")
        while candidate.startswith("./"):
            candidate = candidate[2:]
        if candidate.startswith("/"):
            candidate = candidate[1:]
        if candidate and _match_glob_parts(path_parts, tuple(candidate.split("/"))):
            return True
    return False


def _portable_path(value: str) -> bool:
    if not value or "\\" in value or "\0" in value or "\ufffd" in value:
        return False
    if any(ord(character) < 32 for character in value):
        return False
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        return False
    reserved = {"CON", "PRN", "AUX", "NUL"} | {
        f"{prefix}{number}"
        for prefix in ("COM", "LPT")
        for number in range(1, 10)
    }
    for part in path.parts:
        stem = part.split(".", 1)[0].upper()
        if (
            unicodedata.normalize("NFC", part) != part
            or part.casefold() == ".git"
            or part.endswith((" ", "."))
            or ":" in part
            or stem in reserved
        ):
            return False
    return True


def _path_blocker(
    value: str,
    *,
    include: Sequence[str],
    exclude: Sequence[str],
) -> str | None:
    if not _portable_path(value):
        return "nonportable_path"
    path = PurePosixPath(value)
    lowered_parts = {part.casefold() for part in path.parts}
    lowered_name = path.name.casefold()
    if (
        lowered_name in SECRET_PRONE_NAMES
        or lowered_name.startswith(".env.")
        or path.suffix.casefold() in SECRET_PRONE_SUFFIXES
    ):
        return "secret_prone_path"
    if lowered_parts.intersection(EXCLUDED_DIRECTORIES) or _matches(value, exclude):
        return "excluded_path"
    if include and not _matches(value, include):
        return "outside_include_scope"
    if lowered_name in GENERATED_NAMES or lowered_name.endswith(GENERATED_SUFFIXES):
        return "generated_path"
    if not include and path.suffix.casefold() not in SUPPORTED_SUFFIXES:
        return "unsupported_extension"
    return None


def _contains_secret(value: str) -> bool:
    return any(pattern.search(value) for pattern in SECRET_PATTERNS)


def _parse_raw_changes(raw: bytes) -> list[tuple[str, str, str, str]]:
    """Parse ``git diff-tree --raw -z`` without quote or delimiter ambiguity."""

    fields = raw.decode("utf-8", errors="replace").split("\0")
    fields = [field for field in fields if field]
    if len(fields) % 2:
        raise HistoryError("git returned malformed raw change data")
    changes: list[tuple[str, str, str, str]] = []
    for index in range(0, len(fields), 2):
        metadata = fields[index].split()
        path = fields[index + 1]
        if len(metadata) != 5 or not metadata[0].startswith(":"):
            raise HistoryError("git returned malformed raw change metadata")
        changes.append((metadata[4], path, metadata[0][1:], metadata[1]))
    return changes


def _file_stats(
    repository: Path,
    parent: str,
    commit: str,
    paths: Sequence[str],
) -> dict[str, tuple[int, int]] | None:
    raw = _git_bytes(
        repository,
        "diff",
        "--numstat",
        "-z",
        "--no-renames",
        parent,
        commit,
        "--",
        *paths,
    )
    result: dict[str, tuple[int, int]] = {}
    for entry in raw.decode("utf-8", errors="replace").split("\0"):
        if not entry:
            continue
        parts = entry.split("\t", 2)
        if len(parts) != 3 or "-" in parts[:2]:
            return None
        try:
            result[parts[2]] = (int(parts[0]), int(parts[1]))
        except ValueError:
            return None
    return result if set(result) == set(paths) else None


def _strip_trailers(message: str) -> list[str]:
    lines = message.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    while lines and not lines[-1].strip():
        lines.pop()
    while lines and TRAILER_PATTERN.fullmatch(lines[-1].strip()):
        lines.pop()
        while lines and not lines[-1].strip():
            lines.pop()
    return lines


def _proposed_instruction(message: str, parent_count: int) -> tuple[str, list[str]]:
    lines = _strip_trailers(message)
    paragraphs = [
        " ".join(line.strip() for line in paragraph.split("\n") if line.strip())
        for paragraph in "\n".join(lines).split("\n\n")
        if paragraph.strip()
    ]
    instruction = paragraphs[0] if paragraphs else ""
    if parent_count == 2 and instruction.casefold().startswith("merge pull request"):
        instruction = paragraphs[1] if len(paragraphs) > 1 else ""
    flags: list[str] = []
    if len(instruction) > MAX_INSTRUCTION_CHARS:
        instruction = instruction[:MAX_INSTRUCTION_CHARS].rstrip()
        flags.append("instruction_needs_rewrite")
    normalized = re.sub(r"[^a-z0-9]+", " ", instruction.casefold()).strip()
    if (
        len(instruction) < MIN_INSTRUCTION_CHARS
        or normalized in GENERIC_INSTRUCTIONS
        or "\ufffd" in instruction
    ):
        flags.append("instruction_needs_rewrite")
    return instruction, sorted(set(flags))


def _patch_added_lines(patch: str) -> list[str]:
    in_hunk = False
    values: list[str] = []
    for line in patch.splitlines():
        if line.startswith("@@"):
            in_hunk = True
            continue
        if line.startswith("diff --git "):
            in_hunk = False
            continue
        if in_hunk and line.startswith("+") and not line.startswith("+++"):
            candidate = re.sub(r"\s+", " ", line[1:].strip())
            if len(candidate) >= 24:
                values.append(candidate.casefold())
    return values


def _instruction_flags(instruction: str, patch: str) -> list[str]:
    flags: list[str] = []
    if "```" in instruction or re.search(r"(?m)^(?:diff --git|@@|\+\+\+|---)", instruction):
        flags.append("instruction_may_leak_solution")
    normalized = re.sub(r"\s+", " ", instruction.strip()).casefold()
    if any(added in normalized for added in _patch_added_lines(patch)):
        flags.append("instruction_may_leak_solution")
    return flags


def _before_context(patch: str, files: Sequence[Mapping[str, Any]]) -> str:
    """Build prompt context from pre-image lines without marking the answer."""

    sections: list[str] = []
    current_path: str | None = None
    current: list[str] = []
    in_hunk = False
    added_paths = {str(item["path"]) for item in files if item["status"] == "A"}

    def finish() -> None:
        nonlocal current
        if current_path is not None and current:
            sections.append("\n".join([f"File: {current_path}", *current]))
        current = []

    for line in patch.splitlines():
        if line.startswith("diff --git "):
            finish()
            current_path = None
            in_hunk = False
            continue
        if line.startswith("+++ b/"):
            current_path = line[6:]
            continue
        if line.startswith("@@"):
            if current_path in added_paths:
                in_hunk = False
                continue
            if current_path is not None and current:
                finish()
            current.append(line.split("@@", 2)[1].strip())
            in_hunk = True
            continue
        if not in_hunk or current_path is None or line == "\\ No newline at end of file":
            continue
        if line.startswith("+"):
            continue
        if line.startswith((" ", "-")):
            # Strip diff markers. A leading '-' in the prompt would tell the
            # model which line to remove and leak part of the accepted answer.
            current.append(line[1:])
    finish()

    existing = {section.splitlines()[0][6:] for section in sections if section.startswith("File: ")}
    for item in files:
        path = str(item["path"])
        if item["status"] == "A" and path not in existing:
            sections.append(f"File: {path}\n(file does not exist yet)")
    return "\n\n".join(sections)


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _candidate_for_commit(
    repository: Path,
    source: Mapping[str, Any],
    repository_identity: str,
    commit: str,
    parents: Sequence[str],
    message: str,
    training_char_limit: int,
) -> tuple[dict[str, Any] | None, str | None]:
    if not parents:
        return None, "root_commit"
    if len(parents) > 2:
        return None, "octopus_merge"
    parent = parents[0]
    if _contains_secret(message):
        return None, "secret_detected"
    subject = next((line.strip() for line in message.splitlines() if line.strip()), "")
    if subject.casefold().startswith("revert"):
        return None, "revert_commit"

    entries = _parse_raw_changes(
        _git_bytes(
            repository,
            "diff-tree",
            "--no-commit-id",
            "--raw",
            "-r",
            "-z",
            "--no-renames",
            parent,
            commit,
        )
    )
    if not entries:
        return None, "empty_change"
    if len(entries) > MAX_CHANGED_FILES:
        return None, "too_many_files"

    include = _strings(source.get("include"))
    exclude = _strings(source.get("exclude"))
    normalized_paths: set[str] = set()
    files: list[dict[str, Any]] = []
    total_lines = 0
    paths: list[str] = []
    for status, path, parent_mode, commit_mode in entries:
        if status not in {"A", "M"}:
            return None, "unsupported_status"
        if _contains_secret(path):
            return None, "secret_detected"
        blocker = _path_blocker(path, include=include, exclude=exclude)
        if blocker:
            return None, blocker
        path_key = unicodedata.normalize("NFC", path).casefold()
        if path_key in normalized_paths:
            return None, "path_collision"
        normalized_paths.add(path_key)

        if commit_mode not in {"100644", "100755"}:
            return None, "non_regular_file"
        if status == "M" and (parent_mode not in {"100644", "100755"} or parent_mode != commit_mode):
            return None, "file_mode_change"
        paths.append(path)
        files.append(
            {
                "path": path,
                "status": status,
            }
        )

    paths.sort()
    stats = _file_stats(repository, parent, commit, paths)
    if stats is None:
        return None, "binary_or_unreadable_diff"
    for item in files:
        additions, deletions = stats[str(item["path"])]
        item["additions"] = additions
        item["deletions"] = deletions
        total_lines += additions + deletions
    if total_lines > MAX_CHANGED_LINES:
        return None, "too_many_changed_lines"

    patch = _git_bytes(
        repository,
        "diff",
        "--no-color",
        "--no-ext-diff",
        "--no-textconv",
        "--no-renames",
        "--full-index",
        "--src-prefix=a/",
        "--dst-prefix=b/",
        "--diff-algorithm=myers",
        "--no-indent-heuristic",
        "--inter-hunk-context=0",
        "--unified=3",
        parent,
        commit,
        "--",
        *paths,
        max_output_bytes=256 * 1024,
    ).decode("utf-8", errors="replace")
    if "\ufffd" in patch:
        return None, "binary_or_unreadable_diff"
    if patch and not patch.endswith("\n"):
        patch += "\n"
    if not patch:
        return None, "empty_change"
    if _contains_secret(patch):
        return None, "secret_detected"

    instruction, flags = _proposed_instruction(message, len(parents))
    flags.extend(_instruction_flags(instruction, patch))
    before_context = _before_context(patch, files)
    training_chars = len(SYSTEM_PROMPT) + len(instruction) + len(before_context) + len(patch)
    if training_chars > training_char_limit:
        return None, "training_example_too_large"

    candidate: dict[str, Any] = {
        "before_context": before_context,
        "commit": commit.lower(),
        "extractor_version": EXTRACTOR_VERSION,
        "files": sorted(files, key=lambda item: item["path"]),
        "flags": sorted(set(flags)),
        "parent": parent.lower(),
        "patch": patch,
        "patch_sha256": "sha256:" + hashlib.sha256(patch.encode("utf-8")).hexdigest(),
        "proposed_instruction": instruction,
        "repository_identity": repository_identity,
        "schema_version": SCHEMA_VERSION,
        "source_id": str(source.get("id", "")),
        "training_char_count": training_chars,
    }
    digest = hashlib.sha256(_canonical_json(candidate).encode("utf-8")).hexdigest()
    candidate["candidate_id"] = f"sha256:{digest}"
    return candidate, None


def _discover_source(
    source: Mapping[str, Any],
    root: Path,
    training_char_limit: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_id = str(source.get("id", "")).strip()
    if not source_id:
        raise HistoryError("repository source id is required")
    revision = str(source.get("revision", "")).strip().lower()
    if not COMMIT_PATTERN.fullmatch(revision):
        raise HistoryError("history review requires an immutable repository revision")
    repository = _resolve_repository(source, root)
    git_root = Path(_git_text(repository, "rev-parse", "--show-toplevel")).resolve()
    if git_root != repository:
        raise HistoryError("repository uri must point at the Git worktree root")
    resolved = _git_text(repository, "rev-parse", "--verify", f"{revision}^{{commit}}")
    if resolved.casefold() != revision.casefold():
        raise HistoryError("repository revision did not resolve to the declared immutable commit")
    identity = _repository_identity(repository, git_root)
    lineage_rows = _git_text(
        repository,
        "rev-list",
        "--first-parent",
        "--parents",
        f"--max-count={MAX_HISTORY_COMMITS}",
        revision,
    ).splitlines()
    lineages = {
        values[0]: values[1:]
        for row in lineage_rows
        if (values := row.split())
    }
    message_fields = _git_bytes(
        repository,
        "log",
        "--first-parent",
        "-z",
        f"--max-count={MAX_HISTORY_COMMITS}",
        "--format=%H%x00%B",
        revision,
    ).decode("utf-8", errors="replace").split("\0")
    if message_fields and not message_fields[-1]:
        message_fields.pop()
    if len(message_fields) % 2:
        raise HistoryError("git returned malformed commit-message data")
    messages = {
        message_fields[index].strip(): message_fields[index + 1].strip()
        for index in range(0, len(message_fields), 2)
    }
    commits = [row.split()[0] for row in lineage_rows if row.split()]
    if set(commits) != set(messages):
        raise HistoryError("git history changed during candidate discovery")

    excluded: dict[str, int] = {}
    candidates: list[dict[str, Any]] = []
    seen_patches: set[str] = set()
    for index, commit in enumerate(commits):
        candidate, reason = _candidate_for_commit(
            repository,
            source,
            identity,
            commit,
            lineages[commit],
            messages[commit],
            training_char_limit,
        )
        if candidate is None:
            assert reason is not None
            excluded[reason] = excluded.get(reason, 0) + 1
            continue
        if candidate["patch_sha256"] in seen_patches:
            excluded["duplicate_patch"] = excluded.get("duplicate_patch", 0) + 1
            continue
        seen_patches.add(candidate["patch_sha256"])
        if len(candidates) >= MAX_REVIEWABLE_CANDIDATES:
            # The newest reviewable integrations are the V1 surface. Once that
            # surface is full, avoid hundreds of needless Git subprocesses.
            excluded["candidate_limit"] = excluded.get("candidate_limit", 0) + len(commits) - index
            break
        candidates.append(candidate)

    report = {
        "considered": len(commits),
        "excluded": dict(sorted(excluded.items())),
        "extractor_version": EXTRACTOR_VERSION,
        "locked_commit": revision,
        "repository_identity": identity,
        "reviewable": len(candidates),
        "schema_version": SCHEMA_VERSION,
        "source_id": source_id,
    }
    return candidates, report


def _write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        for attempt in range(12):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if attempt == 11:
                    raise
                time.sleep(0.01)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def _review_file_lock(reviews_path: Path) -> Any:
    """Serialize short review updates across API and CLI processes."""

    lock_path = reviews_path.with_suffix(reviews_path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + 5.0
    descriptor: int | None = None
    while descriptor is None:
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                stale = time.time() - lock_path.stat().st_mtime > 60.0
            except FileNotFoundError:
                continue
            if stale:
                try:
                    lock_path.unlink()
                except FileNotFoundError:
                    pass
                continue
            if time.monotonic() >= deadline:
                raise HistoryError("history review store is busy; try again")
            time.sleep(0.05)
    try:
        os.write(descriptor, f"{os.getpid()}\n".encode("ascii"))
        os.close(descriptor)
        descriptor = None
        yield
    finally:
        if descriptor is not None:
            os.close(descriptor)
        lock_path.unlink(missing_ok=True)


def _read_reviews(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HistoryError("history review file is invalid") from error
    if not isinstance(payload, Mapping) or payload.get("schema_version") != SCHEMA_VERSION:
        raise HistoryError("history review file has an unsupported schema")
    reviews = payload.get("reviews", {})
    if not isinstance(reviews, Mapping):
        raise HistoryError("history review file must contain a reviews object")
    return {
        str(key): dict(value)
        for key, value in reviews.items()
        if isinstance(value, Mapping)
    }


def _write_source_artifacts(
    history_root: Path,
    source_id: str,
    candidates: Sequence[Mapping[str, Any]],
    report: Mapping[str, Any],
) -> None:
    source_root = history_root / _safe_artifact_name(source_id)
    candidate_lines = [_canonical_json(candidate) for candidate in candidates]
    _write_atomic(
        source_root / "candidates.jsonl",
        "\n".join(candidate_lines) + ("\n" if candidate_lines else ""),
    )
    _write_atomic(
        source_root / "history-report.json",
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _review_has_stale_authority(
    review: Mapping[str, Any],
    declared_source_ids: set[str],
    current_identities: Mapping[str, str],
) -> bool:
    if review.get("decision") != "approved":
        return False
    source_id = str(review.get("source_id", ""))
    identity = str(review.get("repository_identity", ""))
    # Legacy reviews without source metadata stay fail-closed. New reviews stop
    # carrying authority after their repository is deliberately removed or
    # replaced by a different repository under the same display ID.
    if not source_id:
        return True
    if source_id not in declared_source_ids:
        return False
    current_identity = current_identities.get(source_id)
    return not identity or current_identity is None or identity == current_identity


def list_history(
    config_or_path: Mapping[str, Any] | str | Path,
    project_root: Path | None = None,
    *,
    write: bool = True,
) -> dict[str, Any]:
    """Discover current candidates and merge their separately stored reviews.

    The return shape is directly serializable by the loopback API.  Per-source
    failures are reported without exposing repository paths or remote URLs.
    """

    config, root = _coerce_project(config_or_path, project_root)
    training_char_limit = _training_char_limit(config)
    history_root = _artifact_dir(config, root) / "history"
    reviews_path = history_root / "reviews.json"
    reviews = _read_reviews(reviews_path)
    candidates: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    errors: list[str] = []
    excluded: dict[str, int] = {}

    sources = [
        source
        for source in _source_specs(config)
        if source.get("kind") == "repository"
        and source.get("partition", "train") == "train"
        and "history" in _strings(source.get("roles"))
    ]
    for source in sorted(sources, key=lambda item: str(item.get("id", ""))):
        source_id = str(source.get("id", "<unknown>"))
        try:
            discovered, report = _discover_source(source, root, training_char_limit)
        except HistoryError as error:
            errors.append(f"{source_id}: {error}")
            continue
        reports.append(report)
        for reason, count in report["excluded"].items():
            excluded[reason] = excluded.get(reason, 0) + int(count)
        if write:
            _write_source_artifacts(history_root, source_id, discovered, report)
        for candidate in discovered:
            item = dict(candidate)
            review = reviews.get(item["candidate_id"])
            if review and review.get("decision") == "retired":
                # Retiring authority does not reject the underlying change. If
                # that commit becomes current again it needs a fresh decision.
                item["decision"] = "pending"
            elif review and review.get("decision") == "rejected":
                item["decision"] = "rejected"
            elif review and review.get("decision") == "approved":
                try:
                    if review.get("rights_confirmed") is not True:
                        raise HistoryError("rights confirmation is missing")
                    item["instruction"] = _validate_approved_instruction(
                        item,
                        str(review.get("instruction", "")),
                        training_char_limit,
                    )
                except HistoryError:
                    # A hand-edited or obsolete approval must not look valid in
                    # the GUI or silently reach compilation.
                    errors.append(f"{source_id}: an approved history review is invalid; review it again")
                    item["decision"] = "pending"
                else:
                    item["decision"] = "approved"
                    item["rights_confirmed"] = True
            elif review:
                errors.append(f"{source_id}: a history review has an invalid decision")
                item["decision"] = "pending"
            else:
                item["decision"] = "pending"
            candidates.append(item)

    # Sources are already sorted and each source yields newest-first mainline
    # changes, which is both deterministic and the useful review order.
    current_ids = {str(item["candidate_id"]) for item in candidates}
    declared_source_ids = {str(source.get("id", "")) for source in sources}
    current_identities = {
        str(report["source_id"]): str(report["repository_identity"])
        for report in reports
    }

    stale_reviews = sum(
        candidate_id not in current_ids
        and _review_has_stale_authority(
            review, declared_source_ids, current_identities
        )
        for candidate_id, review in reviews.items()
    )
    orphaned_reviews = sum(
        candidate_id not in current_ids
        and review.get("decision") == "approved"
        and not _review_has_stale_authority(
            review, declared_source_ids, current_identities
        )
        for candidate_id, review in reviews.items()
    )
    summary = {
        "approved": sum(item["decision"] == "approved" for item in candidates),
        "considered": sum(int(report["considered"]) for report in reports),
        "excluded": sum(excluded.values()),
        "pending": sum(item["decision"] == "pending" for item in candidates),
        "rejected": sum(item["decision"] == "rejected" for item in candidates),
        "retired_reviews": sum(
            review.get("decision") == "retired" for review in reviews.values()
        ),
        "reviewable": len(candidates),
        "source_count": len(reports),
        "orphaned_reviews": orphaned_reviews,
        # Rejected candidates are not training authority. Only an approval that
        # left the locked ancestry must block later compilation.
        "stale_reviews": stale_reviews,
    }
    return {
        "candidates": candidates,
        "errors": sorted(set(errors)),
        "excluded": dict(sorted(excluded.items())),
        "schema_version": SCHEMA_VERSION,
        "sources": reports,
        "summary": summary,
    }


def _validate_approved_instruction(
    candidate: Mapping[str, Any], instruction: str, training_char_limit: int
) -> str:
    value = instruction.strip()
    normalized = re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()
    if len(value) < MIN_INSTRUCTION_CHARS or normalized in GENERIC_INSTRUCTIONS:
        raise HistoryError("approved instruction must describe the requested behavior")
    if len(value) > MAX_INSTRUCTION_CHARS:
        raise HistoryError(f"approved instruction must be at most {MAX_INSTRUCTION_CHARS} characters")
    if _contains_secret(value):
        raise HistoryError("approved instruction appears to contain a credential")
    if _instruction_flags(value, str(candidate["patch"])):
        raise HistoryError("approved instruction appears to contain the accepted solution")
    training_chars = (
        len(SYSTEM_PROMPT)
        + len(value)
        + len(str(candidate["before_context"]))
        + len(str(candidate["patch"]))
    )
    if training_chars > training_char_limit:
        raise HistoryError("approved instruction makes the training example too large")
    return value


def review_history(
    config_or_path: Mapping[str, Any] | str | Path,
    project_root: Path | None = None,
    *,
    candidate_id: str,
    decision: str,
    instruction: str | None = None,
    rights_confirmed: bool = False,
) -> dict[str, Any]:
    """Atomically approve or reject one current candidate."""

    selected_decision = str(decision).strip().casefold()
    if selected_decision not in {"approved", "rejected"}:
        raise HistoryError("decision must be approved or rejected")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", str(candidate_id)):
        raise HistoryError("candidate_id is invalid")
    config, root = _coerce_project(config_or_path, project_root)
    history = list_history(config, root, write=True)
    matching = [item for item in history["candidates"] if item["candidate_id"] == candidate_id]
    if not matching:
        raise HistoryError("candidate is missing or stale; refresh history before reviewing")
    candidate = matching[0]

    # Source identity lets a later source removal retire this authority without
    # deleting the audit record. It is a digest, never a path or remote URL.
    review: dict[str, Any] = {
        "decision": selected_decision,
        "repository_identity": candidate["repository_identity"],
        "source_id": candidate["source_id"],
    }
    if selected_decision == "approved":
        if rights_confirmed is not True:
            raise HistoryError("confirm the right to train on this change before approval")
        review["instruction"] = _validate_approved_instruction(
            candidate,
            instruction or "",
            _training_char_limit(config),
        )
        review["rights_confirmed"] = True

    history_root = _artifact_dir(config, root) / "history"
    reviews_path = history_root / "reviews.json"
    with _REVIEW_LOCK, _review_file_lock(reviews_path):
        reviews = _read_reviews(reviews_path)
        reviews[candidate_id] = review
        payload = {"reviews": dict(sorted(reviews.items())), "schema_version": SCHEMA_VERSION}
        _write_atomic(
            reviews_path,
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
    return {
        "history": list_history(config, root, write=True),
        "review": {"candidate_id": candidate_id, **review},
    }


def retire_stale_history_reviews(
    config_or_path: Mapping[str, Any] | str | Path,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Explicitly retire missing approvals while preserving their audit rows."""

    config, root = _coerce_project(config_or_path, project_root)
    history = list_history(config, root, write=True)
    current_ids = {str(item["candidate_id"]) for item in history["candidates"]}
    sources = [
        source
        for source in _source_specs(config)
        if source.get("kind") == "repository"
        and source.get("partition", "train") == "train"
        and "history" in _strings(source.get("roles"))
    ]
    declared_source_ids = {str(source.get("id", "")) for source in sources}
    current_identities = {
        str(report["source_id"]): str(report["repository_identity"])
        for report in history["sources"]
    }
    history_root = _artifact_dir(config, root) / "history"
    reviews_path = history_root / "reviews.json"
    retired_count = 0
    with _REVIEW_LOCK, _review_file_lock(reviews_path):
        reviews = _read_reviews(reviews_path)
        for candidate_id, review in list(reviews.items()):
            if candidate_id in current_ids or not _review_has_stale_authority(
                review, declared_source_ids, current_identities
            ):
                continue
            replacement = dict(review)
            replacement["decision"] = "retired"
            replacement["retired_reason"] = "stale_source_revision"
            reviews[candidate_id] = replacement
            retired_count += 1
        if retired_count:
            payload = {
                "reviews": dict(sorted(reviews.items())),
                "schema_version": SCHEMA_VERSION,
            }
            _write_atomic(
                reviews_path,
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            )
    return {
        "history": list_history(config, root, write=True),
        "retired_count": retired_count,
    }


def approved_history_records(
    config_or_path: Mapping[str, Any] | str | Path,
    project_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Return compiler-ready prompt/completion rows for current approvals."""

    config, root = _coerce_project(config_or_path, project_root)
    history = list_history(config, root, write=True)
    if history["errors"]:
        raise HistoryError("cannot compile approved history while discovery has errors")
    if int(history["summary"].get("stale_reviews", 0)):
        raise HistoryError(
            "cannot compile approved history while a prior approval is stale; review history again"
        )
    records: list[dict[str, Any]] = []
    for candidate in history["candidates"]:
        if candidate.get("decision") != "approved" or candidate.get("rights_confirmed") is not True:
            continue
        instruction = _validate_approved_instruction(
            candidate,
            str(candidate.get("instruction", "")),
            _training_char_limit(config),
        )
        records.append(
            {
                "candidate_id": candidate["candidate_id"],
                "completion": [{"content": candidate["patch"], "role": "assistant"}],
                "patch_sha256": candidate["patch_sha256"],
                "prompt": [
                    {"content": SYSTEM_PROMPT, "role": "system"},
                    {
                        "content": (
                            f"{instruction}\n\nPre-change context:\n{candidate['before_context']}"
                        ),
                        "role": "user",
                    },
                ],
                "source_id": candidate["source_id"],
                "source_parent_revision": candidate["parent"],
                "source_repository_identity": candidate["repository_identity"],
                "source_revision": candidate["commit"],
                "source_type": "approved_git_change",
            }
        )
    records.sort(key=lambda item: str(item["candidate_id"]))
    return records


__all__ = [
    "HistoryError",
    "approved_history_records",
    "list_history",
    "retire_stale_history_reviews",
    "review_history",
]
