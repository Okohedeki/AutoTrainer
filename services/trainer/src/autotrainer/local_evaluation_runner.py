"""Built-in, local-only agent producer for the model benchmark.

The producer runs the same bounded repository tools used by GRPO.  A model can
inspect the locked checkout, apply a patch, and run only manifest-declared
checks. :mod:`autotrainer.evaluation` remains the trust boundary: it replays
the resulting patch in a second disposable environment and computes every
score from the frozen verifier.

One producer instance keeps a model loaded while all pending trials for that
arm run.  The suite orchestrator groups trials by arm, then calls ``close``
before loading the next 9B model, so V1 never requires two models on one GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
import gc
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import time
from typing import Any, Callable, Mapping, Protocol, Sequence
from uuid import uuid4

from .model_host import HostSpec, TextGenerator
from .training.common import REFERENCE_DEPENDENCIES, import_factory


PRODUCER_NAME = "autotrainer-local-patch"
PRODUCER_VERSION = "1.1.0"
MAX_SOURCE_FILES = 40
MAX_SOURCE_PATHS = 400
MAX_SOURCE_CHARS = 32_000
MAX_FILE_CHARS = 10_000
MAX_SOURCE_BLOB_BYTES = 256_000
CONTEXT_TOKENS = 8192
MAX_NEW_TOKENS = 2048
MAX_INPUT_TOKENS = CONTEXT_TOKENS - MAX_NEW_TOKENS
TEMPERATURE = 0.2
TOP_P = 0.9

MAX_TOOL_CALLING_ITERATIONS = 8

# Keep this schema code-owned and frozen in the runner identity. It mirrors the
# six public methods on FrontendEnvironment, which TRL exposes during GRPO.
_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List editable repository files below a relative directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Repository-relative directory to inspect. Use ``.`` for the "
                            "editable repository root."
                        ),
                    }
                },
            },
            "return": {
                "type": "string",
                "description": (
                    "A bounded newline-delimited list of repository-relative file paths."
                ),
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read an inclusive, one-indexed line range from a UTF-8 text file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Repository-relative path to the text file.",
                    },
                    "start": {
                        "type": "integer",
                        "description": "First one-indexed line to return.",
                    },
                    "end": {
                        "type": "integer",
                        "description": "Last one-indexed line to return, inclusive.",
                    },
                },
                "required": ["path"],
            },
            "return": {
                "type": "string",
                "description": (
                    "Bounded numbered lines, or a short validation error visible to the\n"
                    "    policy."
                ),
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": (
                "Search text files for a literal query and return bounded line matches."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Literal case-insensitive text to find.",
                    },
                    "path": {
                        "type": "string",
                        "description": "Repository-relative file or directory to search.",
                    },
                },
                "required": ["query"],
            },
            "return": {
                "type": "string",
                "description": "Bounded ``path:line: text`` matches, or ``no matches``.",
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_patch",
            "description": "Apply a unified diff after path and Git safety checks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patch": {
                        "type": "string",
                        "description": (
                            "Complete unified diff whose paths stay inside the editable "
                            "repository."
                        ),
                    }
                },
                "required": ["patch"],
            },
            "return": {
                "type": "string",
                "description": (
                    "A bounded success message or the reason the patch was rejected."
                ),
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "replace_text",
            "description": (
                "Replace one exact UTF-8 text occurrence in an editable file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Repository-relative path to the text file.",
                    },
                    "old": {
                        "type": "string",
                        "description": "Exact text that must occur once in the current file.",
                    },
                    "new": {
                        "type": "string",
                        "description": (
                            "Replacement text. It may be empty when deleting the occurrence."
                        ),
                    },
                },
                "required": ["path", "old", "new"],
            },
            "return": {
                "type": "string",
                "description": (
                    "A bounded success message or a short rejection reason visible to the\n"
                    "    policy. The operation preserves the file's existing newline style."
                ),
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_check",
            "description": "Run one trusted check declared by the task author.",
            "parameters": {
                "type": "object",
                "properties": {
                    "check": {
                        "type": "string",
                        "description": (
                            "Check name: ``build``, ``tests``, or ``browserTests``."
                        ),
                    }
                },
                "required": ["check"],
            },
            "return": {
                "type": "string",
                "description": "A bounded pass/fail summary with captured command output.",
            },
        },
    },
]

# This document is the human-readable meaning of the runner fingerprint.  The
# final identity also hashes the installed implementation files and declared
# runtime dependency pins below, so code drift creates a new evaluation plan
# even when a developer forgets to edit this prose-level specification.
_ORCHESTRATION_SPEC = {
    "producer": PRODUCER_NAME,
    "version": PRODUCER_VERSION,
    "strategy": "native-qwen-multiturn-environment-tools",
    "source": "locked-disposable-environment",
    "prompt": "frozen-compiled-conversation-plus-trl-reset-observation",
    "model_residency": "one-arm-at-a-time-suite-arm-groups",
    "generation": {
        "context_tokens": CONTEXT_TOKENS,
        "max_input_tokens": "context-minus-frozen-completion-budget",
        "max_new_tokens": "frozen-from-grpo-recipe",
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "seed": "frozen-trial-seed-plus-turn-index",
        "tools": _TOOL_SCHEMAS,
        "max_tool_calling_iterations": "frozen-from-grpo-recipe",
        "thinking": False,
        "tool_errors": "bounded-observation-not-suite-failure",
    },
    "output": "environment-applied-unified-git-diff",
}

_SOURCE_PROTOCOL_FILES = (
    "environment.py",
    "environments/frontend.py",
    "evaluation.py",
    "local_evaluation_runner.py",
    "model_host.py",
)


def _stable_identity_digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _source_protocol_identity(
    sources: Mapping[str, bytes] | None = None,
) -> dict[str, Any]:
    """Describe relevant installed code without recording machine paths.

    The injectable mapping keeps the digest algorithm independently testable.
    Production calls read the exact package files that Python is executing and
    label them by stable package-relative names only.
    """

    if sources is None:
        package_root = Path(__file__).resolve().parent
        sources = {
            f"autotrainer/{name}": (package_root / name).read_bytes()
            for name in _SOURCE_PROTOCOL_FILES
        }
    entries = [
        {
            "path": str(name).replace("\\", "/"),
            "sha256": hashlib.sha256(content).hexdigest(),
            "bytes": len(content),
        }
        for name, content in sorted(sources.items())
    ]
    return {
        "files": entries,
        "sha256": _stable_identity_digest(entries),
    }


SOURCE_PROTOCOL_IDENTITY = _source_protocol_identity()
DECLARED_RUNTIME_DEPENDENCIES = dict(sorted(REFERENCE_DEPENDENCIES.items()))
ORCHESTRATION_SHA256 = _stable_identity_digest(
    {
        "orchestration": _ORCHESTRATION_SPEC,
        "source_protocol_sha256": SOURCE_PROTOCOL_IDENTITY["sha256"],
        "runtime_dependencies": DECLARED_RUNTIME_DEPENDENCIES,
    }
)

_IMMUTABLE_REVISION = re.compile(r"^[0-9a-fA-F]{40,64}$")
_TEXT_SUFFIXES = frozenset(
    {
        ".css",
        ".graphql",
        ".htm",
        ".html",
        ".js",
        ".json",
        ".jsx",
        ".md",
        ".mjs",
        ".scss",
        ".svg",
        ".ts",
        ".tsx",
        ".txt",
        ".vue",
        ".yaml",
        ".yml",
    }
)
_SKIPPED_NAMES = frozenset(
    {"package-lock.json", "pnpm-lock.yaml", "yarn.lock", "bun.lock", "bun.lockb"}
)


class LocalEvaluationRunnerError(RuntimeError):
    """Raised when a frozen local model trial cannot run truthfully."""


@dataclass(frozen=True, slots=True)
class ArmRuntime:
    """Resolved local files for one immutable arm in the frozen plan."""

    arm_id: str
    model_id: str
    revision: str
    snapshot_path: Path
    adapter_name: str
    adapter_path: Path | None
    adapter_sha256: str | None


class Producer(Protocol):
    """Interface used by the evaluation orchestrator and lightweight tests."""

    def preflight(self, arm_ids: Sequence[str]) -> None: ...

    def produce(self, request: Mapping[str, Any], result_path: Path) -> None: ...

    def close(self) -> None: ...


ModelLoader = Callable[[HostSpec], TextGenerator]
RuntimeResolver = Callable[[Mapping[str, Any], Path, Mapping[str, Any]], ArmRuntime]
EnvironmentFactory = Callable[[], Any]


def builtin_runner_identity() -> dict[str, Any]:
    """Return code-owned identity fields inserted into every frozen plan."""

    return {
        "producer": PRODUCER_NAME,
        "version": PRODUCER_VERSION,
        "orchestration_sha256": ORCHESTRATION_SHA256,
        # Return detached mappings so callers cannot mutate module-level
        # identity state and make a later plan rebuild bless edited evidence.
        "source_protocol": json.loads(json.dumps(SOURCE_PROTOCOL_IDENTITY)),
        "runtime_dependencies": dict(DECLARED_RUNTIME_DEPENDENCIES),
    }


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_tree(path: Path) -> str:
    """Match the evaluation plan's adapter tree identity exactly."""

    if path.is_symlink() or not path.is_dir():
        raise LocalEvaluationRunnerError(f"adapter directory is missing: {path}")
    entries: list[dict[str, str]] = []
    for candidate in sorted(path.rglob("*")):
        if candidate.is_symlink():
            raise LocalEvaluationRunnerError(
                f"adapter trees must not contain symlinks: {candidate}"
            )
        if candidate.is_file():
            entries.append(
                {
                    "path": candidate.relative_to(path).as_posix(),
                    "sha256": _sha256_file(candidate),
                }
            )
    if not entries:
        raise LocalEvaluationRunnerError(f"adapter directory is empty: {path}")
    return hashlib.sha256(_canonical(entries)).hexdigest()


def _project_path(root: Path, value: object) -> Path:
    supplied = Path(str(value)).expanduser()
    return supplied.resolve() if supplied.is_absolute() else (root / supplied).resolve()


def _snapshot_download() -> Callable[..., str]:
    """Import the Hub client lazily; evaluation never performs a download."""

    try:
        from huggingface_hub import snapshot_download
    except (ImportError, OSError) as error:
        raise LocalEvaluationRunnerError(
            "The built-in benchmark requires the pinned huggingface-hub dependency."
        ) from error
    return snapshot_download


def resolve_arm_runtime(
    config: Mapping[str, Any],
    project_root: Path,
    arm: Mapping[str, Any],
) -> ArmRuntime:
    """Resolve one arm from local cache and verify its optional adapter.

    ``local_files_only=True`` is not an optimization; it is the policy boundary
    that keeps Evaluate separate from the explicit model-download step.
    """

    root = Path(project_root).expanduser().resolve()
    arm_id = str(arm.get("id", "")).strip()
    model = arm.get("model")
    if not arm_id or not isinstance(model, Mapping):
        raise LocalEvaluationRunnerError("the frozen evaluation arm is invalid")
    model_id = str(model.get("id", "")).strip()
    revision = str(model.get("revision", "")).strip().lower()
    if not model_id or _IMMUTABLE_REVISION.fullmatch(revision) is None:
        raise LocalEvaluationRunnerError(
            f"evaluation arm {arm_id!r} does not have an immutable model pin"
        )
    if model.get("trust_remote_code", False) is not False:
        raise LocalEvaluationRunnerError(
            f"evaluation arm {arm_id!r} must set trust_remote_code to false"
        )

    project_model = config.get("model", {})
    if not isinstance(project_model, Mapping):
        raise LocalEvaluationRunnerError("project model configuration is missing")
    cache_dir = _project_path(
        root, project_model.get("cache_dir", ".autotrainer/model-cache")
    )
    try:
        snapshot = Path(
            _snapshot_download()(
                repo_id=model_id,
                revision=revision,
                cache_dir=cache_dir,
                local_files_only=True,
            )
        ).resolve()
    except LocalEvaluationRunnerError:
        raise
    except Exception as error:
        raise LocalEvaluationRunnerError(
            f"Evaluation arm {arm_id!r} is not downloaded: {model_id}@{revision}. "
            "Download that exact Hugging Face revision first; Evaluate never downloads weights."
        ) from error
    if not snapshot.is_dir():
        raise LocalEvaluationRunnerError(
            f"The cached snapshot for evaluation arm {arm_id!r} is missing: {snapshot}"
        )

    adapter = arm.get("adapter")
    adapter_path: Path | None = None
    adapter_name = "base"
    adapter_sha256: str | None = None
    if adapter is not None:
        if not isinstance(adapter, Mapping):
            raise LocalEvaluationRunnerError(
                f"evaluation arm {arm_id!r} has an invalid adapter declaration"
            )
        adapter_path = _project_path(root, adapter.get("path"))
        adapter_name = str(adapter.get("stage", "adapter"))
        adapter_sha256 = _sha256_tree(adapter_path)
        expected_digest = str(adapter.get("sha256", "")).lower()
        if adapter_sha256 != expected_digest:
            raise LocalEvaluationRunnerError(
                f"Evaluation arm {arm_id!r} adapter changed after its plan was frozen. "
                "Freeze a new evaluation plan before running it."
            )
        try:
            adapter_config = json.loads(
                (adapter_path / "adapter_config.json").read_text(encoding="utf-8")
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise LocalEvaluationRunnerError(
                f"Evaluation arm {arm_id!r} adapter metadata is unreadable: {error}"
            ) from error
        if not isinstance(adapter_config, Mapping) or adapter_config.get(
            "base_model_name_or_path"
        ) != model_id:
            raise LocalEvaluationRunnerError(
                f"Evaluation arm {arm_id!r} adapter does not belong to {model_id!r}"
            )
        adapter_revision = adapter_config.get("revision")
        if adapter_revision is not None and str(adapter_revision).lower() != revision:
            raise LocalEvaluationRunnerError(
                f"Evaluation arm {arm_id!r} adapter base revision does not match "
                "the frozen model revision"
            )
        provenance = adapter.get("training_provenance")
        if isinstance(provenance, Mapping) and provenance.get("status") == "verified":
            recipe_path = _project_path(root, provenance.get("path"))
            expected_recipe_digest = str(provenance.get("sha256", "")).lower()
            if not recipe_path.is_file() or _sha256_file(recipe_path) != expected_recipe_digest:
                raise LocalEvaluationRunnerError(
                    f"Evaluation arm {arm_id!r} training provenance changed after plan creation"
                )

    return ArmRuntime(
        arm_id=arm_id,
        model_id=model_id,
        revision=revision,
        snapshot_path=snapshot,
        adapter_name=adapter_name,
        adapter_path=adapter_path,
        adapter_sha256=adapter_sha256,
    )


def _git_environment() -> dict[str, str]:
    """Prevent user Git config, replace refs, or lazy fetching from changing input."""

    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.upper().startswith("GIT_")
    }
    environment.update(
        {
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_NO_LAZY_FETCH": "1",
            "GIT_TERMINAL_PROMPT": "0",
        }
    )
    return environment


def _git(
    repository: Path, *arguments: str, binary: bool = False
) -> subprocess.CompletedProcess[Any]:
    command = [
        "git",
        "--no-replace-objects",
        "-c",
        "protocol.file.allow=never",
        "-c",
        f"safe.directory={repository.as_posix()}",
        "-C",
        str(repository),
        *arguments,
    ]
    return subprocess.run(
        command,
        capture_output=True,
        text=not binary,
        timeout=60,
        check=False,
        shell=False,
        env=_git_environment(),
    )


def _working_directory(task: Mapping[str, Any]) -> PurePosixPath:
    manifest = task.get("manifest")
    runtime = manifest.get("runtime") if isinstance(manifest, Mapping) else None
    value = runtime.get("workingDirectory", ".") if isinstance(runtime, Mapping) else "."
    text = str(value).replace("\\", "/").strip() or "."
    relative = PurePosixPath(text)
    if relative.is_absolute() or any(part in {"..", ".git"} for part in relative.parts):
        raise LocalEvaluationRunnerError(
            "evaluation runtime.workingDirectory must stay inside the locked repository"
        )
    return relative


def _source_priority(path: str) -> tuple[int, int, str]:
    """Prefer editable app code before docs/config when context is constrained."""

    normalized = path.lower()
    name = PurePosixPath(path).name.lower()
    if "/src/" in f"/{normalized}" or normalized.startswith(("src/", "app/")):
        group = 0
    elif name in {"index.html", "package.json", "vite.config.ts", "vite.config.js"}:
        group = 1
    elif Path(name).suffix.lower() in {".tsx", ".ts", ".jsx", ".js", ".css", ".html"}:
        group = 2
    else:
        group = 3
    return group, len(PurePosixPath(path).parts), normalized


def _locked_source_context(task: Mapping[str, Any]) -> str:
    """Read bounded UTF-8 files from the exact Git object named by the task."""

    source_value = task.get("source_path")
    revision = str(task.get("source_revision", "")).strip().lower()
    if not source_value or _IMMUTABLE_REVISION.fullmatch(revision) is None:
        raise LocalEvaluationRunnerError(
            "the evaluation task lacks a local Git source and immutable revision"
        )
    repository = Path(str(source_value)).expanduser().resolve()
    if not repository.is_dir() or not (repository / ".git").exists():
        raise LocalEvaluationRunnerError(
            f"the evaluation source is not a local Git repository: {repository}"
        )

    resolved = _git(repository, "rev-parse", "--verify", f"{revision}^{{commit}}")
    if resolved.returncode or resolved.stdout.strip().lower() != revision:
        raise LocalEvaluationRunnerError(
            f"the frozen source revision is unavailable locally: {revision}"
        )

    working_directory = _working_directory(task)
    prefix = "" if str(working_directory) == "." else working_directory.as_posix()
    arguments = ["ls-tree", "-r", "-z", "--long", revision]
    if prefix:
        # Git pathspec metacharacters are meaningful even after ``--``.  The
        # explicit literal signature makes a directory such as ``apps/[ui]``
        # select that exact subtree instead of the similarly named ``apps/u``.
        arguments.extend(["--", f":(literal){prefix}"])
    listing = _git(repository, *arguments, binary=True)
    if listing.returncode:
        detail = listing.stderr.decode("utf-8", errors="replace").strip()
        raise LocalEvaluationRunnerError(f"cannot read the frozen source tree: {detail}")

    candidates: list[tuple[str, str, int]] = []
    for raw_entry in listing.stdout.split(b"\0"):
        if not raw_entry:
            continue
        metadata, separator, raw_path = raw_entry.partition(b"\t")
        if not separator:
            raise LocalEvaluationRunnerError("the frozen Git tree contains invalid metadata")
        try:
            mode, object_type, object_id, size_text = metadata.decode("ascii").split()
            path = raw_path.decode("utf-8")
            size = int(size_text)
        except (UnicodeDecodeError, ValueError) as error:
            raise LocalEvaluationRunnerError(
                "the frozen Git tree contains a non-portable entry"
            ) from error
        path_value = PurePosixPath(path)
        if (
            path_value.is_absolute()
            or any(part in {"..", ".git"} for part in path_value.parts)
            or (
                prefix
                and path_value.parts[: len(working_directory.parts)]
                != working_directory.parts
            )
        ):
            raise LocalEvaluationRunnerError(
                f"frozen source entry escaped workingDirectory: {path!r}"
            )
        if mode not in {"100644", "100755"} or object_type != "blob":
            raise LocalEvaluationRunnerError(
                f"the frozen source contains an unsupported Git entry: {path}"
            )
        name = PurePosixPath(path).name.lower()
        suffix = PurePosixPath(path).suffix.lower()
        if (
            name in _SKIPPED_NAMES
            or suffix not in _TEXT_SUFFIXES
            or size > MAX_SOURCE_BLOB_BYTES
        ):
            continue
        candidates.append((path, object_id, size))

    candidates.sort(key=lambda item: _source_priority(item[0]))
    visible_paths = sorted(path for path, _object_id, _size in candidates)[:MAX_SOURCE_PATHS]
    sections: list[str] = []
    used_chars = 0
    for path, object_id, _size in candidates:
        if len(sections) >= MAX_SOURCE_FILES or used_chars >= MAX_SOURCE_CHARS:
            break
        blob = _git(repository, "cat-file", "blob", object_id, binary=True)
        if blob.returncode:
            detail = blob.stderr.decode("utf-8", errors="replace").strip()
            raise LocalEvaluationRunnerError(f"cannot read frozen source file {path}: {detail}")
        try:
            content = blob.stdout.decode("utf-8")
        except UnicodeDecodeError:
            continue
        remaining = MAX_SOURCE_CHARS - used_chars
        file_limit = min(MAX_FILE_CHARS, remaining)
        if file_limit <= 0:
            break
        truncated = len(content) > file_limit
        content = content[:file_limit]
        header = f"\n--- FILE: {path} ---\n"
        section = header + content
        if truncated:
            section += "\n[AutoTrainer truncated this file at the frozen context limit]\n"
        sections.append(section)
        used_chars += len(content)

    if not sections:
        raise LocalEvaluationRunnerError(
            "the frozen working directory contains no supported UTF-8 source files"
        )
    path_manifest = "\n".join(f"- {path}" for path in visible_paths)
    if len(candidates) > len(visible_paths):
        path_manifest += f"\n- ... {len(candidates) - len(visible_paths)} more paths omitted"
    # Put editable contents first so tokenizer fitting cannot spend the entire
    # budget on a large path manifest before the model sees any code.
    return (
        "Selected frozen file contents (deterministic and bounded):"
        + "".join(sections)
        + "\n\nOther tracked source paths:\n"
        + path_manifest
    )


def _unified_patch(text: str) -> str | None:
    """Extract one unified diff without treating explanation text as evidence."""

    if not isinstance(text, str) or "\0" in text:
        return None
    candidate = text
    for match in re.finditer(
        r"```(?:diff|patch)?\s*\n(.*?)```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    ):
        if "diff --git " in match.group(1):
            candidate = match.group(1)
            break
    start = candidate.find("diff --git ")
    if start < 0:
        return None
    patch = candidate[start:].strip()
    if not patch or len(patch.encode("utf-8")) > 10 * 1024 * 1024:
        return None
    return patch + "\n"


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _default_model_loader(spec: HostSpec) -> TextGenerator:
    # Reuse the exact 4-bit, local-files-only loader used by the callable host.
    # Importing here keeps ordinary setup and plan inspection CUDA-free.
    from .model_host import _load_generator

    return _load_generator(spec)


@dataclass(frozen=True, slots=True)
class _ToolCall:
    """One parsed native Qwen tool request; reasoning is intentionally absent."""

    name: str
    arguments: dict[str, Any]


_TOOL_BLOCK = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_FUNCTION_BLOCK = re.compile(
    r"\s*<function=([A-Za-z_][A-Za-z0-9_]*)>\s*(.*?)\s*</function>\s*",
    re.DOTALL,
)
_PARAMETER_BLOCK = re.compile(
    r"<parameter=([A-Za-z_][A-Za-z0-9_]*)>\s*\n?(.*?)\n?\s*</parameter>",
    re.DOTALL,
)

_TOOL_ARGUMENTS: dict[str, tuple[set[str], set[str]]] = {
    "list_files": (set(), {"path"}),
    "read_file": ({"path"}, {"start", "end"}),
    "search_code": ({"query"}, {"path"}),
    "apply_patch": ({"patch"}, set()),
    "replace_text": ({"path", "old", "new"}, set()),
    "run_check": ({"check"}, set()),
}


def _parse_tool_calls(text: str) -> tuple[str, list[_ToolCall]] | None:
    """Parse Qwen's native XML call format without retaining model reasoning."""

    matches = list(_TOOL_BLOCK.finditer(text))
    if not matches:
        if "<tool_call" in text or "</tool_call>" in text:
            raise LocalEvaluationRunnerError("the model emitted a malformed tool call")
        return None
    if text[matches[-1].end() :].strip():
        raise LocalEvaluationRunnerError("the model added text after a tool call")
    for previous, current in zip(matches, matches[1:]):
        if text[previous.end() : current.start()].strip():
            raise LocalEvaluationRunnerError("the model added text between tool calls")

    calls: list[_ToolCall] = []
    for match in matches:
        function = _FUNCTION_BLOCK.fullmatch(match.group(1))
        if function is None:
            raise LocalEvaluationRunnerError("the model emitted a malformed function call")
        name = function.group(1)
        contract = _TOOL_ARGUMENTS.get(name)
        if contract is None:
            raise LocalEvaluationRunnerError(f"the model requested unknown tool {name!r}")

        body = function.group(2)
        arguments: dict[str, Any] = {}
        cursor = 0
        for parameter in _PARAMETER_BLOCK.finditer(body):
            if body[cursor : parameter.start()].strip():
                raise LocalEvaluationRunnerError(
                    f"tool {name!r} contains malformed parameter markup"
                )
            key = parameter.group(1)
            if key in arguments:
                raise LocalEvaluationRunnerError(
                    f"tool {name!r} repeats parameter {key!r}"
                )
            raw = parameter.group(2).strip()
            if key in {"start", "end"}:
                try:
                    arguments[key] = int(raw)
                except ValueError as error:
                    raise LocalEvaluationRunnerError(
                        f"tool {name!r} parameter {key!r} must be an integer"
                    ) from error
            else:
                # Native templates usually emit strings without JSON quotes,
                # but accepting a properly quoted string avoids passing quote
                # characters into repository paths.
                if len(raw) >= 2 and raw[0] == raw[-1] == '"':
                    try:
                        decoded = json.loads(raw)
                    except json.JSONDecodeError:
                        decoded = raw
                    arguments[key] = decoded if isinstance(decoded, str) else raw
                else:
                    arguments[key] = raw
            cursor = parameter.end()
        if body[cursor:].strip():
            raise LocalEvaluationRunnerError(
                f"tool {name!r} contains malformed parameter markup"
            )
        required, optional = contract
        missing = sorted(required - set(arguments))
        unknown = sorted(set(arguments) - required - optional)
        if missing or unknown:
            detail = []
            if missing:
                detail.append("missing " + ", ".join(missing))
            if unknown:
                detail.append("unknown " + ", ".join(unknown))
            raise LocalEvaluationRunnerError(
                f"tool {name!r} has invalid arguments ({'; '.join(detail)})"
            )
        if name == "run_check" and arguments["check"] not in {
            "build",
            "tests",
            "browserTests",
        }:
            raise LocalEvaluationRunnerError("run_check requested an unsupported check")
        calls.append(_ToolCall(name=name, arguments=arguments))

    # The prefix may contain native thinking text. It is needed only to render
    # the next turn and is never written to the evidence directory.
    return text[: matches[0].start()].strip(), calls


def _tool_messages_for_context(
    generator: TextGenerator,
    messages: list[dict[str, Any]],
    *,
    max_input_tokens: int = MAX_INPUT_TOKENS,
) -> tuple[list[dict[str, Any]], int]:
    """Fit tool history using exact tokenizer counts and deterministic pruning."""

    counter = getattr(generator, "count_tokens_with_tools", None)
    if not callable(counter):
        raise LocalEvaluationRunnerError(
            "the loaded model runtime does not support the GRPO tool chat template"
        )
    fitted = json.loads(json.dumps(messages))
    token_count = counter(fitted, _TOOL_SCHEMAS)
    if token_count <= max_input_tokens:
        return fitted, token_count

    tool_indexes = [
        index for index, message in enumerate(fitted) if message.get("role") == "tool"
    ]
    marker = "[Earlier bounded tool output omitted to fit the frozen context.]"
    for index in tool_indexes[:-1]:
        fitted[index]["content"] = marker
        token_count = counter(fitted, _TOOL_SCHEMAS)
        if token_count <= max_input_tokens:
            return fitted, token_count

    if tool_indexes:
        index = tool_indexes[-1]
        original = str(fitted[index].get("content", ""))
        low, high = 0, len(original)
        best: tuple[list[dict[str, Any]], int] | None = None
        while low <= high:
            middle = (low + high) // 2
            fitted[index]["content"] = original[:middle] + "\n[Tool output truncated.]"
            candidate_tokens = counter(fitted, _TOOL_SCHEMAS)
            if candidate_tokens <= max_input_tokens:
                best = (json.loads(json.dumps(fitted)), candidate_tokens)
                low = middle + 1
            else:
                high = middle - 1
        if best is not None:
            return best
    raise LocalEvaluationRunnerError(
        "the task and native tool history exceed the frozen 8K evaluation context"
    )


def _tool_call_message(prefix: str, calls: Sequence[_ToolCall]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": prefix,
        "tool_calls": [
            {
                "type": "function",
                "function": {"name": call.name, "arguments": call.arguments},
            }
            for call in calls
        ],
    }


def _tool_error_observation(error: Exception) -> str:
    """Turn a model-caused rejected tool call into a bounded safe observation."""

    # Training treats tool exceptions as observations instead of aborting the
    # trainer. Evaluation does the same, but does not echo exception text that
    # could contain a host path or other private runtime detail.
    return (
        f"Tool error ({type(error).__name__}): the bounded environment rejected "
        "this request. Correct the arguments or choose another declared tool."
    )


def _prompt_with_environment_observation(
    task_row: Mapping[str, Any], observation: str
) -> list[dict[str, Any]]:
    """Mirror TRL's environment reset append for the frozen compiled prompt."""

    prompt = task_row.get("prompt")
    if not isinstance(prompt, list) or not prompt:
        raise LocalEvaluationRunnerError(
            "the frozen task row has no conversational prompt"
        )
    messages = json.loads(json.dumps(prompt))
    if not all(
        isinstance(message, dict)
        and isinstance(message.get("role"), str)
        and isinstance(message.get("content"), str)
        for message in messages
    ):
        raise LocalEvaluationRunnerError(
            "the frozen task row contains an invalid conversational prompt"
        )
    if messages[-1]["role"] != "user":
        raise LocalEvaluationRunnerError(
            "the frozen task prompt must end with a user message"
        )
    if not observation.startswith("\n\n"):
        raise LocalEvaluationRunnerError(
            "the environment reset observation must begin with a message separator"
        )
    # TRL 1.8 appends reset output to the final message. Reproducing that exact
    # operation keeps held-out evaluation aligned with GRPO prompt semantics.
    messages[-1]["content"] += observation
    return messages


class BuiltinEvaluationProducer:
    """Run the GRPO tool surface with one persistent, locally cached arm."""

    def __init__(
        self,
        config: Mapping[str, Any],
        project_root: Path,
        plan: Mapping[str, Any],
        *,
        model_loader: ModelLoader | None = None,
        runtime_resolver: RuntimeResolver = resolve_arm_runtime,
        environment_factory: EnvironmentFactory | None = None,
    ) -> None:
        self._config = config
        self._root = Path(project_root).expanduser().resolve()
        self._plan = plan
        self._model_loader = model_loader or _default_model_loader
        self._runtime_resolver = runtime_resolver
        self._environment_factory = environment_factory or self._resolve_environment_factory()
        environment = plan.get("environment")
        iterations = (
            environment.get("max_tool_calling_iterations", MAX_TOOL_CALLING_ITERATIONS)
            if isinstance(environment, Mapping)
            else MAX_TOOL_CALLING_ITERATIONS
        )
        if (
            not isinstance(iterations, int)
            or isinstance(iterations, bool)
            or not 1 <= iterations <= 32
        ):
            raise LocalEvaluationRunnerError(
                "the frozen evaluation tool-iteration limit is invalid"
            )
        self._max_tool_calling_iterations = iterations
        completion_tokens = (
            environment.get("max_completion_tokens", MAX_NEW_TOKENS)
            if isinstance(environment, Mapping)
            else MAX_NEW_TOKENS
        )
        if (
            not isinstance(completion_tokens, int)
            or isinstance(completion_tokens, bool)
            or not 1 <= completion_tokens <= 4096
        ):
            raise LocalEvaluationRunnerError(
                "the frozen evaluation completion-token limit is invalid"
            )
        # The completion budget is frozen from the GRPO recipe. Reserving it
        # before fitting each tool turn keeps training and evaluation on the
        # same 8K context contract instead of silently using a fixed default.
        self._max_completion_tokens = completion_tokens
        self._max_input_tokens = CONTEXT_TOKENS - completion_tokens
        self._runtimes: dict[str, ArmRuntime] = {}
        self._active_arm: str | None = None
        self._generator: TextGenerator | None = None

    def _resolve_environment_factory(self) -> EnvironmentFactory:
        environment = self._plan.get("environment")
        if not isinstance(environment, Mapping):
            environment = self._config.get("environment")
        if not isinstance(environment, Mapping):
            raise LocalEvaluationRunnerError(
                "the frozen evaluation plan has no environment contract"
            )
        factory_path = environment.get("factory")
        try:
            return import_factory(str(factory_path or ""))
        except Exception as error:
            raise LocalEvaluationRunnerError(
                f"the frozen evaluation environment factory is unavailable: {error}"
            ) from error

    @staticmethod
    def _validate_environment(environment: Any) -> dict[str, Callable[..., str]]:
        tools: dict[str, Callable[..., str]] = {}
        for schema in _TOOL_SCHEMAS:
            name = str(schema["function"]["name"])
            method = getattr(environment, name, None)
            if not callable(method):
                raise LocalEvaluationRunnerError(
                    f"the evaluation environment is missing GRPO tool {name!r}"
                )
            tools[name] = method
        if not callable(getattr(environment, "reset", None)):
            raise LocalEvaluationRunnerError("the evaluation environment has no reset method")
        if not callable(getattr(environment, "_export_patch_for_evaluation", None)):
            raise LocalEvaluationRunnerError(
                "the evaluation environment cannot export an untrusted patch safely"
            )
        return tools

    def _private_task_row(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        task_id = str(request.get("task_id", ""))
        rows = self._plan.get("task_rows")
        if isinstance(rows, Mapping) and isinstance(rows.get(task_id), Mapping):
            return rows[task_id]
        # This fallback exists for injected unit environments only. Canonical
        # plans always retain the private row locally while requests stay public.
        task = request.get("task")
        if isinstance(task, Mapping):
            return task
        raise LocalEvaluationRunnerError(f"frozen task row is missing for {task_id!r}")

    def preflight(self, arm_ids: Sequence[str]) -> None:
        """Resolve every exact snapshot before producing any partial benchmark."""

        arms = self._plan.get("arms")
        if not isinstance(arms, Mapping):
            raise LocalEvaluationRunnerError("the frozen evaluation plan has no arms")
        for arm_id in arm_ids:
            arm = arms.get(arm_id)
            if not isinstance(arm, Mapping):
                raise LocalEvaluationRunnerError(f"unknown frozen evaluation arm: {arm_id}")
            self._runtimes[arm_id] = self._runtime_resolver(
                self._config, self._root, arm
            )

    def _select_arm(self, arm_id: str) -> TextGenerator:
        if self._active_arm == arm_id and self._generator is not None:
            return self._generator
        self._release_model()
        runtime = self._runtimes.get(arm_id)
        if runtime is None:
            self.preflight([arm_id])
            runtime = self._runtimes[arm_id]
        # Preflight may be separated from GPU loading by other arm groups or a
        # user pause. Re-hash at the last safe boundary so a replaced adapter
        # is refused before PEFT reads any mutable bytes.
        if runtime.adapter_path is not None:
            current_digest = _sha256_tree(runtime.adapter_path)
            if current_digest != runtime.adapter_sha256:
                raise LocalEvaluationRunnerError(
                    f"Evaluation arm {arm_id!r} adapter changed after preflight. "
                    "Freeze a new evaluation plan before loading it."
                )
        host_spec = HostSpec(
            config_path=self._root / "autotrainer.yaml",
            model_id=runtime.model_id,
            revision=runtime.revision,
            snapshot_path=runtime.snapshot_path,
            adapter_name=runtime.adapter_name,
            adapter_path=runtime.adapter_path,
            display_name=arm_id,
        )
        try:
            self._generator = self._model_loader(host_spec)
        except Exception as error:
            raise LocalEvaluationRunnerError(
                f"Could not load evaluation arm {arm_id!r} on the local GPU: {error}"
            ) from error
        self._active_arm = arm_id
        return self._generator

    def produce(self, request: Mapping[str, Any], result_path: Path) -> None:
        """Run one native tool episode and write the schema-v1 result envelope."""

        arm_id = str(request.get("arm_id", ""))
        task = request.get("task")
        runner = request.get("runner")
        if not arm_id or not isinstance(task, Mapping) or not isinstance(runner, Mapping):
            raise LocalEvaluationRunnerError("the built-in evaluation request is invalid")
        if runner.get("type") != "builtin":
            raise LocalEvaluationRunnerError("the request is not assigned to the built-in runner")
        expected_identity = builtin_runner_identity()
        if any(runner.get(key) != value for key, value in expected_identity.items()):
            raise LocalEvaluationRunnerError(
                "the frozen built-in runner identity does not match this AutoTrainer version"
            )

        runtime = self._runtimes.get(arm_id)
        if runtime is None:
            self.preflight([arm_id])
            runtime = self._runtimes[arm_id]
        generator = self._select_arm(arm_id)
        generate_turn = getattr(generator, "generate_with_tools", None)
        if not callable(generate_turn):
            raise LocalEvaluationRunnerError(
                "the loaded model runtime does not support native GRPO tools"
            )

        environment = self._environment_factory()
        tools = self._validate_environment(environment)
        task_row = self._private_task_row(request)
        started = time.monotonic()
        try:
            observation = environment.reset(**dict(task_row))
            if not isinstance(observation, str):
                raise LocalEvaluationRunnerError(
                    "the evaluation environment reset did not return text"
                )
            messages = _prompt_with_environment_observation(task_row, observation)
            total_input_tokens = 0
            total_output_tokens = 0
            tool_calls = 0
            environment_hard_failed = False
            for turn in range(self._max_tool_calling_iterations):
                fitted, expected_input = _tool_messages_for_context(
                    generator,
                    messages,
                    max_input_tokens=self._max_input_tokens,
                )
                remaining = self._max_completion_tokens - total_output_tokens
                if remaining <= 0:
                    break
                try:
                    output_text, input_tokens, output_tokens = generate_turn(
                        fitted,
                        _TOOL_SCHEMAS,
                        max_tokens=remaining,
                        temperature=TEMPERATURE,
                        top_p=TOP_P,
                        seed=int(request.get("seed", 0)) + turn,
                    )
                except Exception as error:
                    raise LocalEvaluationRunnerError(
                        f"Local generation failed for evaluation arm {arm_id!r}: {error}"
                    ) from error
                if (
                    not isinstance(input_tokens, int)
                    or isinstance(input_tokens, bool)
                    or not isinstance(output_tokens, int)
                    or isinstance(output_tokens, bool)
                    or input_tokens < 0
                    or output_tokens < 0
                    or input_tokens != expected_input
                    or input_tokens > self._max_input_tokens
                    or input_tokens + remaining > CONTEXT_TOKENS
                    or output_tokens > remaining
                ):
                    raise LocalEvaluationRunnerError(
                        "the local generator exceeded the frozen 8K evaluation context budget"
                    )
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens
                try:
                    parsed = _parse_tool_calls(output_text)
                except LocalEvaluationRunnerError:
                    # A malformed or unavailable tool request is model output,
                    # not an infrastructure outage. Preserve failures-score-zero
                    # semantics and export only work already applied safely.
                    break
                if parsed is None:
                    break
                prefix, calls = parsed
                messages = fitted
                messages.append(_tool_call_message(prefix, calls))
                for call in calls:
                    tool_calls += 1
                    try:
                        result = tools[call.name](**call.arguments)
                    except Exception as error:
                        # Invalid paths, disabled tools, and exhausted tool
                        # budgets are policy behavior. Keep them inside this
                        # trial instead of misreporting them as suite outages.
                        result = _tool_error_observation(error)
                        last_result = getattr(environment, "last_result", None)
                        environment_hard_failed = bool(
                            getattr(last_result, "hard_gate_reason", None)
                        )
                    messages.append({"role": "tool", "content": str(result)})
            try:
                exported = environment._export_patch_for_evaluation()
            except Exception:
                # A deadline finalizes and cleans the supported environment
                # before raising to the tool loop. Its structured result still
                # retains the exact bounded diff; use that failed-trial evidence
                # instead of turning model behavior into a suite outage.
                last_result = getattr(environment, "last_result", None)
                captured_diff = getattr(last_result, "unified_diff", None)
                if not isinstance(captured_diff, str):
                    raise
                exported = captured_diff
                environment_hard_failed = bool(
                    getattr(last_result, "hard_gate_reason", None)
                )
            patch = _unified_patch(exported)
        finally:
            cleanup = getattr(environment, "_cleanup", None)
            if callable(cleanup):
                cleanup()

        elapsed = round(time.monotonic() - started, 6)
        output: dict[str, str] = {}
        if patch is not None:
            Path(result_path).parent.mkdir(parents=True, exist_ok=True)
            patch_path = Path(result_path).parent / "patch.diff"
            patch_path.write_text(patch, encoding="utf-8", newline="\n")
            output["patch"] = patch_path.name

        result = {
            "schema_version": 1,
            "plan_id": request.get("plan_id"),
            "trial_id": request.get("trial_id"),
            "suite_id": request.get("suite_id"),
            "arm_id": arm_id,
            "task_id": request.get("task_id"),
            "repetition": request.get("repetition"),
            "seed": request.get("seed"),
            "status": (
                "completed"
                if patch is not None and not environment_hard_failed
                else "failed"
            ),
            "producer": {
                "name": PRODUCER_NAME,
                "version": PRODUCER_VERSION,
                "orchestration_sha256": ORCHESTRATION_SHA256,
                "model_revision": runtime.revision,
                "adapter_sha256": runtime.adapter_sha256,
                "seed_honored": True,
                "fallback_models_used": False,
            },
            "usage": {
                "input_tokens": int(total_input_tokens),
                "output_tokens": int(total_output_tokens),
                "tool_calls": tool_calls,
                "wall_time_seconds": elapsed,
            },
            "output": output,
        }
        _atomic_json(Path(result_path), result)

    def _release_model(self) -> None:
        self._generator = None
        self._active_arm = None
        gc.collect()
        # Torch is already imported after a real model load.  Clearing the CUDA
        # allocator here is what lets the next 9B arm fit on the same GPU.
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except (ImportError, OSError):
            pass

    def close(self) -> None:
        self._release_model()


def create_builtin_producer(
    config: Mapping[str, Any],
    project_root: Path,
    plan: Mapping[str, Any],
) -> BuiltinEvaluationProducer:
    """Default factory kept separate so evaluation tests can inject a fake."""

    return BuiltinEvaluationProducer(config, project_root, plan)


__all__ = [
    "ArmRuntime",
    "BuiltinEvaluationProducer",
    "LocalEvaluationRunnerError",
    "ORCHESTRATION_SHA256",
    "PRODUCER_NAME",
    "PRODUCER_VERSION",
    "SOURCE_PROTOCOL_IDENTITY",
    "Producer",
    "DECLARED_RUNTIME_DEPENDENCIES",
    "builtin_runner_identity",
    "create_builtin_producer",
    "resolve_arm_runtime",
]
