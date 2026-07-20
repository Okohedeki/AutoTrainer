"""First-class, local dataset curation and freeze workflow.

Remote PR metadata is only discovery evidence. An operator-selected LLM may
propose a dataset design, a human approves or rejects each candidate, and the
compiler writes a versioned local dataset before training can consume it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from .compiler import compile_data
from .config import ConfigError, load_config
from .github_pr_service import (
    read_merged_pull_request_catalog,
    sync_merged_pull_requests,
)
from .history import HistoryError, list_history, validate_history_instruction
from .project_gate import project_mutation_gate
from .sources import scan_sources


DATASET_SCHEMA_VERSION = 1
MAX_LLM_RESPONSE_BYTES = 128 * 1024
DEFAULT_LOCAL_ENDPOINT = "http://127.0.0.1:8000/v1/chat/completions"
LANGUAGES = ("python", "typescript_react", "csharp", "cpp")
METHODS = ("qlora", "grpo")
_SAFE_MODEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/+-]{0,199}")
_LANGUAGE_SUFFIXES = {
    ".py": "python",
    ".pyi": "python",
    ".js": "typescript_react",
    ".jsx": "typescript_react",
    ".ts": "typescript_react",
    ".tsx": "typescript_react",
    ".cs": "csharp",
    ".c": "cpp",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".h": "cpp",
    ".hh": "cpp",
    ".hpp": "cpp",
    ".hxx": "cpp",
}


class DatasetDesignError(RuntimeError):
    """Stable public error for a selected dataset designer."""


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
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


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _dataset_root(config: Any) -> Path:
    return (config.artifact_dir / "dataset").resolve()


def _design_path(config: Any) -> Path:
    return _dataset_root(config) / "designs.json"


def _freeze_path(config: Any) -> Path:
    return _dataset_root(config) / "freeze.json"


def _read_mapping(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return dict(value) if isinstance(value, Mapping) else None


def _read_designs(config: Any) -> dict[str, dict[str, Any]]:
    payload = _read_mapping(_design_path(config))
    designs = payload.get("designs", {}) if payload else {}
    if payload and payload.get("schema_version") != DATASET_SCHEMA_VERSION:
        return {}
    if not isinstance(designs, Mapping):
        return {}
    return {
        str(key): dict(value)
        for key, value in designs.items()
        if isinstance(value, Mapping)
    }


def _candidate_languages(candidate: Mapping[str, Any]) -> list[str]:
    languages: set[str] = set()
    files = candidate.get("files", [])
    if not isinstance(files, Sequence) or isinstance(files, (str, bytes, bytearray)):
        return []
    for value in files:
        if not isinstance(value, Mapping):
            continue
        suffix = Path(str(value.get("path", ""))).suffix.casefold()
        language = _LANGUAGE_SUFFIXES.get(suffix)
        if language:
            languages.add(language)
    return [value for value in LANGUAGES if value in languages]


def _language_summary(candidates: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = {language: 0 for language in LANGUAGES}
    for candidate in candidates:
        for language in _candidate_languages(candidate):
            counts[language] += 1
    return {key: value for key, value in counts.items() if value}


def _hash_file(digest: Any, path: Path, label: str) -> None:
    digest.update(label.encode("utf-8"))
    digest.update(b"\0")
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        digest.update(b"<missing>")
    digest.update(b"\0")


def _input_fingerprint(config: Any) -> str:
    """Bind a freeze to config, reviews, catalogs, and explicit dataset files."""

    digest = hashlib.sha256()
    data_binding = {
        "artifact_dir": config.data.get("project", {}).get("artifact_dir", ".autotrainer"),
        "grpo_datasets": {
            key: config.data.get("grpo", {}).get(key)
            for key in ("dataset", "eval_dataset")
        },
        "sft_datasets": {
            key: config.data.get("sft", {}).get(key)
            for key in ("dataset", "eval_dataset")
        },
        "sources": config.sources,
    }
    digest.update(_canonical_json(data_binding))
    digest.update(b"\0")
    for path in sorted((_dataset_root(config) / "github-prs").glob("*.json")):
        _hash_file(digest, path, f"catalog:{path.name}")
    _hash_file(digest, config.artifact_dir / "history" / "reviews.json", "reviews")
    for source in config.sources:
        if source.get("kind") not in {"sft_jsonl", "task_pack"}:
            continue
        source_id = str(source.get("id", ""))
        path = config.resolve_path(str(source.get("uri", "")))
        if path.is_file():
            _hash_file(digest, path, f"source:{source_id}")
        elif path.is_dir():
            files = sorted(value for value in path.rglob("*") if value.is_file())
            if len(files) > 10_000:
                raise ConfigError(f"dataset source {source_id!r} contains too many files")
            for value in files:
                relative = value.relative_to(path).as_posix()
                if ".git" not in value.relative_to(path).parts:
                    _hash_file(digest, value, f"source:{source_id}:{relative}")
        else:
            digest.update(f"source:{source_id}:<missing>\0".encode("utf-8"))
    return "sha256:" + digest.hexdigest()


def _extract_json(text: str) -> Mapping[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.IGNORECASE)
        candidate = re.sub(r"\s*```$", "", candidate)
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        if start < 0:
            raise DatasetDesignError("The selected designer did not return JSON.") from None
        try:
            value, _end = json.JSONDecoder().raw_decode(candidate[start:])
        except json.JSONDecodeError as error:
            raise DatasetDesignError("The selected designer did not return valid JSON.") from error
    if not isinstance(value, Mapping):
        raise DatasetDesignError("The selected designer returned an invalid dataset design.")
    return value


def _request_json(url: str, headers: Mapping[str, str], payload: Mapping[str, Any]) -> Mapping[str, Any]:
    request = Request(
        url,
        data=_canonical_json(payload),
        headers={"Content-Type": "application/json", **dict(headers)},
        method="POST",
    )
    try:
        with urlopen(request, timeout=90) as response:
            body = response.read(MAX_LLM_RESPONSE_BYTES + 1)
    except HTTPError as error:
        raise DatasetDesignError(
            "The selected dataset designer rejected the request; check its model and credentials."
        ) from error
    except (URLError, TimeoutError, OSError) as error:
        raise DatasetDesignError(
            "The selected dataset designer is unavailable; check its endpoint or network access."
        ) from error
    if len(body) > MAX_LLM_RESPONSE_BYTES:
        raise DatasetDesignError("The selected dataset designer returned too much data.")
    try:
        value = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DatasetDesignError("The selected dataset designer returned invalid data.") from error
    if not isinstance(value, Mapping):
        raise DatasetDesignError("The selected dataset designer returned invalid data.")
    return value


def _designer_response(
    *,
    provider: str,
    model: str,
    endpoint: str | None,
    prompt: str,
) -> str:
    if not _SAFE_MODEL.fullmatch(model):
        raise ConfigError("designer model is invalid")
    if provider == "local":
        url = endpoint or DEFAULT_LOCAL_ENDPOINT
        parsed = urlsplit(url)
        if (
            parsed.scheme != "http"
            or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ConfigError("local designer endpoint must be a loopback HTTP URL")
        response = _request_json(
            url,
            {},
            {
                "max_tokens": 700,
                "messages": [{"content": prompt, "role": "user"}],
                "model": model,
                "temperature": 0.1,
            },
        )
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
            raise DatasetDesignError("The local designer returned no completion.")
        message = choices[0].get("message")
        content = message.get("content") if isinstance(message, Mapping) else None
        if not isinstance(content, str):
            raise DatasetDesignError("The local designer returned no completion.")
        return content
    if provider == "anthropic":
        key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not key:
            raise DatasetDesignError("ANTHROPIC_API_KEY is required for the Anthropic designer.")
        response = _request_json(
            "https://api.anthropic.com/v1/messages",
            {
                "anthropic-version": "2023-06-01",
                "x-api-key": key,
            },
            {
                "max_tokens": 700,
                "messages": [{"content": prompt, "role": "user"}],
                "model": model,
                "temperature": 0.1,
            },
        )
        blocks = response.get("content")
        if not isinstance(blocks, list):
            raise DatasetDesignError("The Anthropic designer returned no completion.")
        text = "\n".join(
            str(block.get("text", ""))
            for block in blocks
            if isinstance(block, Mapping) and block.get("type") == "text"
        ).strip()
        if not text:
            raise DatasetDesignError("The Anthropic designer returned no completion.")
        return text
    raise ConfigError("designer provider must be local or anthropic")


def _design_prompt(candidate: Mapping[str, Any]) -> str:
    pull_request = candidate.get("pull_request", {})
    pr_context = (
        f"PR #{pull_request.get('number')}: {pull_request.get('title')}"
        if isinstance(pull_request, Mapping)
        else "Reviewed local change"
    )
    return (
        "You are designing a local code-refinement dataset from an accepted change. "
        "Infer the request answered by the patch without copying the solution into the request. "
        "Choose QLoRA when the patch is a useful supervised response. Choose GRPO only when the "
        "change can become a resettable task with an executable verifier. Return JSON only with "
        "keys instruction, language, recommended_method, reason, and grpo_task. language must be "
        "python, typescript_react, csharp, or cpp. recommended_method must be qlora or grpo. "
        "grpo_task must be null for QLoRA, or an object with instruction and verifier_focus for GRPO.\n\n"
        f"{pr_context}\n\nChanged files:\n"
        + "\n".join(str(value.get("path", "")) for value in candidate.get("files", []))
        + f"\n\nPre-change context:\n{candidate.get('before_context', '')}"
        + f"\n\nAccepted patch:\n{candidate.get('patch', '')}"
    )


def _validate_design(candidate: Mapping[str, Any], value: Mapping[str, Any]) -> dict[str, Any]:
    language = str(value.get("language", "")).strip().casefold()
    method = str(value.get("recommended_method", "")).strip().casefold()
    if language not in LANGUAGES:
        raise DatasetDesignError("The selected designer returned an unsupported language.")
    if method not in METHODS:
        raise DatasetDesignError("The selected designer returned an unsupported refinement method.")
    instruction = validate_history_instruction(candidate, str(value.get("instruction", "")))
    reason = " ".join(str(value.get("reason", "")).split())[:500]
    if not reason:
        raise DatasetDesignError("The selected designer did not explain its recommendation.")
    grpo_task = value.get("grpo_task")
    normalized_task: dict[str, str] | None = None
    if method == "grpo":
        if not isinstance(grpo_task, Mapping):
            raise DatasetDesignError("A GRPO design must include an executable-task proposal.")
        task_instruction = " ".join(str(grpo_task.get("instruction", "")).split())[:1_000]
        verifier_focus = " ".join(str(grpo_task.get("verifier_focus", "")).split())[:1_000]
        if not task_instruction or not verifier_focus:
            raise DatasetDesignError("A GRPO design must describe its task and verifier focus.")
        normalized_task = {
            "instruction": task_instruction,
            "verifier_focus": verifier_focus,
        }
    return {
        "grpo_task": normalized_task,
        "instruction": instruction,
        "language": language,
        "reason": reason,
        "recommended_method": method,
    }


def _history(config: Any) -> dict[str, Any]:
    try:
        return list_history(config.data, config.root, write=True)
    except HistoryError as error:
        return {
            "candidates": [],
            "errors": [str(error)],
            "summary": {},
        }


def _freeze_status(config: Any) -> dict[str, Any]:
    receipt = _read_mapping(_freeze_path(config))
    if not receipt or receipt.get("schema_version") != DATASET_SCHEMA_VERSION:
        return {"status": "not_frozen"}
    current = _input_fingerprint(config)
    if receipt.get("input_fingerprint") != current:
        return {"status": "stale", "receipt": receipt}
    return {"status": "ready", "receipt": receipt}


def get_dataset_workspace(config_path: str | Path) -> dict[str, Any]:
    """Inspect local dataset sources, designs, reviews, and frozen artifacts."""

    config = load_config(config_path)
    catalogs = read_merged_pull_request_catalog(config.path)
    history = _history(config)
    designs = _read_designs(config)
    candidates: list[dict[str, Any]] = []
    for value in history.get("candidates", []):
        if not isinstance(value, Mapping):
            continue
        item = {
            "candidate_id": value.get("candidate_id"),
            "decision": value.get("decision"),
            "files": value.get("files", []),
            "flags": value.get("flags", []),
            "languages": _candidate_languages(value),
            "patch": value.get("patch", ""),
            "proposed_instruction": value.get("proposed_instruction", ""),
            "pull_request": value.get("pull_request"),
        }
        design = designs.get(str(value.get("candidate_id", "")))
        if design:
            item["design"] = design
        candidates.append(item)
    approved = [value for value in history.get("candidates", []) if value.get("decision") == "approved"]
    summary = history.get("summary", {})
    return {
        "catalog": catalogs,
        "candidates": candidates,
        "designers": {
            "anthropic": {
                "configured": bool(os.environ.get("ANTHROPIC_API_KEY", "").strip()),
                "models": ["claude-haiku-4-5", "claude-sonnet-4-5"],
            },
            "local": {"default_endpoint": DEFAULT_LOCAL_ENDPOINT},
        },
        "errors": [str(value) for value in history.get("errors", [])],
        "freeze": _freeze_status(config),
        "policy": {
            "allowed_base_branches": ["main", "master"],
            "license_required": True,
            "quality_rule": "merged_pull_request",
            "storage": "local_only",
        },
        "summary": {
            "approved_count": int(summary.get("approved", 0) or 0),
            "language_counts": _language_summary(approved),
            "pending_count": int(summary.get("pending", 0) or 0),
            "rejected_count": int(summary.get("rejected", 0) or 0),
            "reviewable_count": int(summary.get("reviewable", 0) or 0),
            "stale_review_count": int(summary.get("stale_reviews", 0) or 0),
        },
    }


def sync_dataset_sources(config_path: str | Path) -> dict[str, Any]:
    """Refresh licensed merged PRs and return the local dataset workspace."""

    with project_mutation_gate(config_path):
        sync_merged_pull_requests(config_path)
        return get_dataset_workspace(config_path)


def design_dataset_candidate(
    config_path: str | Path,
    *,
    candidate_id: str,
    provider: str,
    model: str,
    endpoint: str | None = None,
) -> dict[str, Any]:
    """Ask the operator-selected model to propose a reviewable dataset design."""

    with project_mutation_gate(config_path):
        config = load_config(config_path)
        history = list_history(config.data, config.root, write=True)
        matches = [
            value
            for value in history.get("candidates", [])
            if value.get("candidate_id") == candidate_id and value.get("decision") == "pending"
        ]
        if not matches:
            raise ConfigError("dataset candidate is missing, already reviewed, or stale")
        candidate = matches[0]
        raw = _designer_response(
            provider=provider.strip().casefold(),
            model=model.strip(),
            endpoint=endpoint.strip() if endpoint else None,
            prompt=_design_prompt(candidate),
        )
        design = _validate_design(candidate, _extract_json(raw))
        designs = _read_designs(config)
        designs[candidate_id] = {
            **design,
            "designer": {"model": model.strip(), "provider": provider.strip().casefold()},
        }
        _atomic_json(
            _design_path(config),
            {
                "designs": dict(sorted(designs.items())),
                "schema_version": DATASET_SCHEMA_VERSION,
            },
        )
        return get_dataset_workspace(config.path)


def freeze_dataset(config_path: str | Path) -> dict[str, Any]:
    """Compile inspected inputs and bind training to their local immutable receipt."""

    with project_mutation_gate(config_path):
        config = load_config(config_path)
        catalogs = read_merged_pull_request_catalog(config.path)
        if catalogs["status"] == "needs_sync":
            raise ConfigError("sync merged pull requests before freezing the dataset")
        history = list_history(config.data, config.root, write=True)
        if history.get("errors"):
            raise ConfigError(str(history["errors"][0]))
        summary = history.get("summary", {})
        if int(summary.get("stale_reviews", 0) or 0):
            raise ConfigError("retire stale reviews before freezing the dataset")
        if int(summary.get("pending", 0) or 0):
            raise ConfigError("approve or reject every dataset candidate before freezing")

        scan = scan_sources(config.data, config.root, write=True)
        if scan.get("errors"):
            raise ConfigError(str(scan["errors"][0]))
        compiled = compile_data(config.data, config.root, scan)
        if compiled.get("errors"):
            raise ConfigError(str(compiled["errors"][0]))
        counts = compiled.get("counts", {})
        if not any(int(value or 0) for value in counts.values()):
            raise ConfigError("the inspected dataset is empty")
        approved = [
            value
            for value in history.get("candidates", [])
            if isinstance(value, Mapping) and value.get("decision") == "approved"
        ]
        receipt = {
            "artifact_sha256": dict(compiled.get("artifact_sha256", {})),
            "compiler_fingerprint": compiled.get("fingerprint"),
            "counts": dict(counts),
            "input_fingerprint": _input_fingerprint(config),
            "language_counts": _language_summary(approved),
            "schema_version": DATASET_SCHEMA_VERSION,
            "status": "frozen",
        }
        _atomic_json(_freeze_path(config), receipt)
        # A new dataset freeze can change the primary training language even
        # when an older benchmark run remains useful audit evidence. Retire
        # only the mutable pointer so evaluation must resolve and freeze the
        # matching shipped language profile again.
        (config.artifact_dir / "evaluation" / "current-plan.json").unlink(
            missing_ok=True
        )
        return get_dataset_workspace(config.path)


def require_frozen_dataset(config_path: str | Path) -> Mapping[str, Any]:
    """Fail closed before refinement if the inspected local dataset changed."""

    config = load_config(config_path)
    status = _freeze_status(config)
    if status["status"] == "not_frozen":
        raise ConfigError("freeze and inspect the local dataset before training")
    if status["status"] == "stale":
        raise ConfigError("the local dataset changed after it was frozen; freeze it again")
    receipt = status.get("receipt")
    if not isinstance(receipt, Mapping):
        raise ConfigError("the local dataset freeze receipt is invalid")
    return receipt


__all__ = [
    "DatasetDesignError",
    "design_dataset_candidate",
    "freeze_dataset",
    "get_dataset_workspace",
    "require_frozen_dataset",
    "sync_dataset_sources",
]
