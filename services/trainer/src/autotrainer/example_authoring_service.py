"""Guided, provenance-aware authoring for supervised training examples."""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
import re
from typing import Any

from .config import (
    ConfigError,
    load_config,
    project_config_mutation,
    validate_mapping,
    write_config,
)
from .project_gate import project_mutation_gate
from .sources import normalize_sft_record


_MANAGED_SOURCE_ID = "authored-sft-examples"
_SAFE_ID = re.compile(r"[^a-z0-9._-]+")


def _text(value: object, field: str, *, minimum: int, maximum: int) -> str:
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


def _slug(value: str) -> str:
    selected = _SAFE_ID.sub("-", value.casefold()).strip("-.")[:72]
    if not selected or not selected[0].isalnum():
        selected = f"example-{selected}".strip("-.")
    return selected or "example"


def _managed_path(config: Any) -> Path:
    return (config.artifact_dir / "authored-examples" / "sft.jsonl").resolve()


def _display_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    try:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ConfigError(
                    f"managed SFT example {line_number} must be a JSON object"
                )
            normalized = normalize_sft_record(value)
            records.append(dict(normalized))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ConfigError(f"managed SFT examples cannot be read: {error}") from error
    return records


def _write_records(path: Path, records: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = "".join(
        json.dumps(dict(record), sort_keys=True, separators=(",", ":")) + "\n"
        for record in sorted(records, key=lambda item: str(item.get("example_id", "")))
    )
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def _repository_source(config: Any, source_id: str) -> Mapping[str, Any]:
    source = next(
        (
            item
            for item in config.sources
            if str(item.get("id", "")) == source_id
        ),
        None,
    )
    if source is None or source.get("kind") != "repository":
        raise ConfigError("source_id must identify a configured Git repository")
    if source.get("partition") != "train":
        raise ConfigError("supervised examples cannot use an evaluation holdout repository")
    revision = str(source.get("revision", "")).strip()
    if re.fullmatch(r"[0-9a-fA-F]{40,64}", revision) is None:
        raise ConfigError(
            "the selected repository must be pinned to an immutable commit"
        )
    return source


def _source_declaration(config: Any, path: Path) -> tuple[str, bool]:
    for source in config.sources:
        if source.get("kind") != "sft_jsonl":
            continue
        uri = str(source.get("uri", ""))
        if uri and config.resolve_path(uri) == path:
            return str(source.get("id", "")), False
    used = {str(source.get("id", "")) for source in config.sources}
    if _MANAGED_SOURCE_ID not in used:
        return _MANAGED_SOURCE_ID, True
    for suffix in range(2, 10_000):
        candidate = f"{_MANAGED_SOURCE_ID}-{suffix}"
        if candidate not in used:
            return candidate, True
    raise ConfigError("could not allocate a managed SFT source id")


def _updated_config(config: Any, path: Path, source_id: str) -> dict[str, Any]:
    updated = dict(config.data)
    sources = list(config.sources)
    sources.append(
        {
            "id": source_id,
            "kind": "sft_jsonl",
            "uri": _display_path(path, config.root),
            "partition": "train",
            "roles": ["demonstrations"],
        }
    )
    updated["sources"] = sources
    report = validate_mapping(updated, root=config.root)
    if report.errors:
        raise ConfigError("\n".join(report.errors))
    return updated


def _serialize(record: Mapping[str, Any]) -> dict[str, Any]:
    prompt = record.get("prompt", [])
    completion = record.get("completion", [])
    instruction = (
        str(prompt[-1].get("content", ""))
        if isinstance(prompt, list) and prompt and isinstance(prompt[-1], Mapping)
        else ""
    )
    response = (
        str(completion[-1].get("content", ""))
        if isinstance(completion, list)
        and completion
        and isinstance(completion[-1], Mapping)
        else ""
    )
    return {
        "id": str(record.get("example_id", "")),
        "source_id": str(record.get("source_id", "")),
        "instruction": instruction,
        "response_preview": response[:240],
        "rights_confirmed": record.get("rights_confirmed") is True,
        "status": "declared",
        "next_action": {
            "title": "Review and lock the dataset",
            "detail": (
                "This example is ready to inspect in Data. Lock the dataset when "
                "you are satisfied with every training record."
            ),
        },
    }


def list_authored_examples(config_path: str | Path) -> dict[str, Any]:
    """List supervised examples created through the managed authoring flow."""

    config = load_config(config_path)
    records = _records(_managed_path(config))
    return {"examples": [_serialize(record) for record in records]}


def create_authored_example(
    config_path: str | Path,
    *,
    source_id: str,
    instruction: str,
    accepted_response: str,
    rights_confirmed: bool,
    example_id: str | None = None,
) -> dict[str, Any]:
    """Persist one rights-confirmed demonstration tied to a locked repository."""

    selected_source_id = _text(source_id, "source_id", minimum=1, maximum=200)
    selected_instruction = _text(
        instruction, "instruction", minimum=20, maximum=8_000
    )
    selected_response = _text(
        accepted_response, "accepted_response", minimum=20, maximum=40_000
    )
    if rights_confirmed is not True:
        raise ConfigError(
            "rights_confirmed must be true before an accepted response can be used"
        )
    requested_id = (
        _text(example_id, "example_id", minimum=1, maximum=100)
        if example_id is not None
        else None
    )

    with project_mutation_gate(config_path):
        with project_config_mutation(config_path):
            config = load_config(config_path)
            _repository_source(config, selected_source_id)
            path = _managed_path(config)
            records = _records(path)
            used = {str(record.get("example_id", "")) for record in records}
            base = _slug(requested_id or selected_instruction)
            allocated_id = base
            for suffix in range(2, 10_000):
                if allocated_id not in used:
                    break
                allocated_id = f"{base[:68]}-{suffix}"
            else:
                raise ConfigError("could not allocate a unique example id")
            record = normalize_sft_record(
                {
                    "example_id": allocated_id,
                    "source_id": selected_source_id,
                    "rights_confirmed": True,
                    "prompt": selected_instruction,
                    "completion": selected_response,
                }
            )
            managed_source_id, add_source = _source_declaration(config, path)
            original = path.read_bytes() if path.is_file() else None
            _write_records(path, [*records, record])
            try:
                if add_source:
                    write_config(
                        config.path,
                        _updated_config(config, path, managed_source_id),
                        overwrite=True,
                    )
            except Exception:
                if original is None:
                    path.unlink(missing_ok=True)
                else:
                    path.write_bytes(original)
                raise

    return {"example": _serialize(record), **list_authored_examples(config_path)}


def remove_authored_example(
    config_path: str | Path,
    *,
    example_id: str,
) -> dict[str, Any]:
    """Remove one managed example and disconnect its empty JSONL source."""

    selected_id = _text(example_id, "example_id", minimum=1, maximum=100)
    with project_mutation_gate(config_path):
        with project_config_mutation(config_path):
            config = load_config(config_path)
            path = _managed_path(config)
            records = _records(path)
            selected = next(
                (
                    record
                    for record in records
                    if str(record.get("example_id", "")) == selected_id
                ),
                None,
            )
            if selected is None:
                raise ConfigError(f"authored example does not exist: {selected_id}")
            remaining = [record for record in records if record is not selected]
            updated: dict[str, Any] | None = None
            if not remaining:
                updated = dict(config.data)
                updated["sources"] = [
                    source
                    for source in config.sources
                    if not (
                        source.get("kind") == "sft_jsonl"
                        and str(source.get("uri", ""))
                        and config.resolve_path(str(source.get("uri"))) == path
                    )
                ]
                report = validate_mapping(updated, root=config.root)
                if report.errors:
                    raise ConfigError("\n".join(report.errors))
            original = path.read_bytes()
            _write_records(path, remaining)
            try:
                if updated is not None:
                    write_config(config.path, updated, overwrite=True)
            except Exception:
                path.write_bytes(original)
                raise
            if not remaining:
                path.unlink(missing_ok=True)

    return {
        "removed": _serialize(selected),
        **list_authored_examples(config_path),
    }


__all__ = [
    "create_authored_example",
    "list_authored_examples",
    "remove_authored_example",
]
