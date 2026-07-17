"""Built-in, local-only patch producer for the model benchmark.

The producer is deliberately smaller than a general coding agent.  It reads a
bounded view of the exact Git tree frozen by the evaluation plan, asks one
locally cached model for a unified patch, and writes the existing untrusted
result envelope.  :mod:`autotrainer.evaluation` remains the trust boundary: it
replays that patch in the disposable environment and computes every score.

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
from .training.common import REFERENCE_DEPENDENCIES


PRODUCER_NAME = "autotrainer-local-patch"
PRODUCER_VERSION = "1.0.0"
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

_SYSTEM_PROMPT = (
    "You are AutoTrainer's local held-out code-patch producer. "
    "Return only one valid unified Git diff beginning with 'diff --git'. "
    "Use repository-relative paths, make the smallest focused change that satisfies "
    "the instruction, and preserve unrelated behavior. You have no network and no "
    "interactive tools in this V1 runner; use only the frozen source shown below. "
    "Do not wrap the diff in Markdown and do not add prose."
)
_USER_PROMPT_TEMPLATE = (
    "Task instruction:\n{instruction}\n\n"
    "Public runtime commands (these run only after you submit the patch):\n"
    "{runtime}\n\n{source_context}"
)

# This document is the human-readable meaning of the runner fingerprint.  The
# final identity also hashes the installed implementation files and declared
# runtime dependency pins below, so code drift creates a new evaluation plan
# even when a developer forgets to edit this prose-level specification.
_ORCHESTRATION_SPEC = {
    "producer": PRODUCER_NAME,
    "version": PRODUCER_VERSION,
    "strategy": "single-pass-full-context-unified-diff",
    "source": {
        "git_object_database_only": True,
        "max_files": MAX_SOURCE_FILES,
        "max_paths": MAX_SOURCE_PATHS,
        "max_chars": MAX_SOURCE_CHARS,
        "max_file_chars": MAX_FILE_CHARS,
        "max_blob_bytes": MAX_SOURCE_BLOB_BYTES,
    },
    "prompt": {
        "system": _SYSTEM_PROMPT,
        "user_template": _USER_PROMPT_TEMPLATE,
    },
    "model_residency": "one-arm-at-a-time-suite-arm-groups",
    "generation": {
        "context_tokens": CONTEXT_TOKENS,
        "max_input_tokens": MAX_INPUT_TOKENS,
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "seed": "frozen-trial-seed",
        "tools": [],
    },
    "output": "unified-git-diff-only",
}

_SOURCE_PROTOCOL_FILES = (
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
ContextBuilder = Callable[[Mapping[str, Any]], str]


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


def _prompt_messages(task: Mapping[str, Any], source_context: str) -> list[dict[str, str]]:
    """Build the public prompt without verifier paths, weights, or hidden tests."""

    manifest = task.get("manifest")
    if not isinstance(manifest, Mapping):
        raise LocalEvaluationRunnerError("the evaluation request has no task manifest")
    task_section = manifest.get("task")
    runtime = manifest.get("runtime")
    if not isinstance(task_section, Mapping) or not isinstance(runtime, Mapping):
        raise LocalEvaluationRunnerError("the evaluation request has an invalid task manifest")
    instruction = str(task_section.get("instruction", "")).strip()
    if not instruction:
        raise LocalEvaluationRunnerError("the evaluation request has no instruction")

    public_runtime = {
        key: str(runtime.get(key, ""))
        for key in ("workingDirectory", "install", "build", "tests", "browserTests")
        if runtime.get(key) is not None
    }
    user_content = _USER_PROMPT_TEMPLATE.format(
        instruction=instruction,
        runtime=json.dumps(public_runtime, indent=2, sort_keys=True),
        source_context=source_context,
    )
    return [
        {
            "role": "system",
            "content": _SYSTEM_PROMPT,
        },
        {"role": "user", "content": user_content},
    ]


def _message_token_count(
    generator: TextGenerator, messages: list[dict[str, str]]
) -> int:
    """Count exactly when supported, with a conservative byte fallback.

    Byte length is an upper bound for ordinary byte-backed tokenizers, while a
    fixed allowance covers the chat template's role and control tokens. The
    fallback intentionally admits less source rather than risking a silent
    context overflow with a custom injected generator.
    """

    counter = getattr(generator, "count_tokens", None)
    if callable(counter):
        value = counter(messages)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise LocalEvaluationRunnerError(
                "the local tokenizer returned an invalid prompt token count"
            )
        return value
    encoded_bytes = sum(
        len(message["role"].encode("utf-8"))
        + len(message["content"].encode("utf-8"))
        for message in messages
    )
    return encoded_bytes + 256


def _fit_prompt_messages(
    generator: TextGenerator,
    task: Mapping[str, Any],
    source_context: str,
) -> tuple[list[dict[str, str]], int]:
    """Fit the public source to the explicit 8K input-plus-output budget."""

    messages = _prompt_messages(task, source_context)
    token_count = _message_token_count(generator, messages)
    if token_count <= MAX_INPUT_TOKENS:
        return messages, token_count

    marker = "\n[AutoTrainer truncated source context to fit the frozen token budget]\n"
    empty_messages = _prompt_messages(task, marker)
    if _message_token_count(generator, empty_messages) > MAX_INPUT_TOKENS:
        raise LocalEvaluationRunnerError(
            "the task instruction and runner prompt exceed the 8K evaluation context budget"
        )

    # Tokenization is not perfectly proportional to characters. Binary search
    # the deterministic prefix using the exact loaded tokenizer, then verify
    # once more before generation.
    low = 0
    high = len(source_context)
    best_messages = empty_messages
    best_tokens = _message_token_count(generator, empty_messages)
    while low <= high:
        middle = (low + high) // 2
        candidate = source_context[:middle] + marker
        candidate_messages = _prompt_messages(task, candidate)
        candidate_tokens = _message_token_count(generator, candidate_messages)
        if candidate_tokens <= MAX_INPUT_TOKENS:
            best_messages = candidate_messages
            best_tokens = candidate_tokens
            low = middle + 1
        else:
            high = middle - 1
    if best_tokens + MAX_NEW_TOKENS > CONTEXT_TOKENS:
        raise LocalEvaluationRunnerError(
            "the fitted evaluation prompt exceeds the frozen model context budget"
        )
    return best_messages, best_tokens


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


class BuiltinEvaluationProducer:
    """Generate untrusted patches with one persistent, locally cached arm."""

    def __init__(
        self,
        config: Mapping[str, Any],
        project_root: Path,
        plan: Mapping[str, Any],
        *,
        model_loader: ModelLoader | None = None,
        runtime_resolver: RuntimeResolver = resolve_arm_runtime,
        context_builder: ContextBuilder = _locked_source_context,
    ) -> None:
        self._config = config
        self._root = Path(project_root).expanduser().resolve()
        self._plan = plan
        self._model_loader = model_loader or _default_model_loader
        self._runtime_resolver = runtime_resolver
        self._context_builder = context_builder
        self._runtimes: dict[str, ArmRuntime] = {}
        self._active_arm: str | None = None
        self._generator: TextGenerator | None = None

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
        """Generate a patch and write the existing schema-v1 result envelope."""

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
        source_context = self._context_builder(task)
        messages, _input_budget = _fit_prompt_messages(
            generator, task, source_context
        )
        started = time.monotonic()
        try:
            output_text, input_tokens, output_tokens = generator.generate(
                messages,
                max_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                seed=int(request.get("seed", 0)),
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
            or input_tokens > MAX_INPUT_TOKENS
            or input_tokens + output_tokens > CONTEXT_TOKENS
        ):
            raise LocalEvaluationRunnerError(
                "the local generator exceeded the frozen 8K evaluation context budget"
            )
        elapsed = round(time.monotonic() - started, 6)
        patch = _unified_patch(output_text)
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
            "status": "completed" if patch is not None else "failed",
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
                "input_tokens": int(input_tokens),
                "output_tokens": int(output_tokens),
                "tool_calls": 0,
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
