"""Compile declared demonstrations and task packs into trainer-ready JSONL.

Raw repository files are never promoted to SFT records here.  A repository is
only attached to an executable task row or retained as inspectable evidence.
"""

from __future__ import annotations

import glob
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .manifest import TaskManifest


def _resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _task_files(root: Path, uri: str) -> list[Path]:
    if any(character in uri for character in "*?["):
        pattern = Path(uri).expanduser()
        if not pattern.is_absolute():
            pattern = root / pattern
        return sorted(
            Path(value).resolve()
            for value in glob.glob(str(pattern), recursive=True)
            if Path(value).is_file() and Path(value).suffix.lower() == ".json"
        )
    path = _resolve(root, uri)
    if path.is_dir():
        return sorted(item.resolve() for item in path.rglob("*.json") if item.is_file())
    return [path] if path.is_file() and path.suffix.lower() == ".json" else []


def _atomic_jsonl(path: Path, records: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    content = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for record in records
    )
    temporary.write_text(content, encoding="utf-8", newline="\n")
    temporary.replace(path)


def _repository_lock(scan: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(source["id"]): source
        for source in scan.get("sources", [])
        if isinstance(source, Mapping) and source.get("kind") == "repository"
    }


def compile_data(
    config: Mapping[str, Any],
    project_root: Path,
    scan: Mapping[str, Any],
) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    artifact_value = config.get("project", {}).get("artifact_dir", ".autotrainer")
    artifact_dir = _resolve(root, str(artifact_value))
    compiled_dir = artifact_dir / "compiled"
    source_specs = config.get("sources", [])
    repositories = {
        str(source["id"]): source
        for source in source_specs
        if isinstance(source, Mapping) and source.get("kind") == "repository"
    }
    repository_locks = _repository_lock(scan)
    sft_records: dict[str, list[Mapping[str, Any]]] = {"train": [], "evaluation": []}
    rl_records: dict[str, list[Mapping[str, Any]]] = {"train": [], "evaluation": []}
    errors: list[str] = []
    warnings: list[str] = []
    task_ids: set[str] = set()
    group_partitions: dict[str, str] = {}

    for source in source_specs:
        if not isinstance(source, Mapping):
            continue
        kind = source.get("kind")
        source_id = str(source.get("id", "<unknown>"))
        uri = str(source.get("uri", ""))
        partition = str(source.get("partition", "train"))
        if kind == "sft_jsonl":
            path = _resolve(root, uri)
            if not path.is_file():
                errors.append(f"{source_id}: SFT source is missing: {path}")
                continue
            line_number = 0
            try:
                for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                    if not line.strip():
                        continue
                    value = json.loads(line)
                    if not isinstance(value, Mapping):
                        raise ValueError("record must be an object")
                    sft_records[partition].append(dict(value))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
                errors.append(f"{source_id}: cannot compile {path}:{line_number}: {error}")
        elif kind == "task_pack":
            for manifest_path in _task_files(root, uri):
                try:
                    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                    manifest = TaskManifest.from_mapping(payload)
                except (OSError, json.JSONDecodeError, ValueError, TypeError) as error:
                    errors.append(f"{source_id}: invalid task {manifest_path}: {error}")
                    continue
                if manifest.version != "1.0":
                    errors.append(f"{manifest.task_id}: compile requires task manifest version 1.0")
                    continue
                if manifest.task_id in task_ids:
                    errors.append(f"duplicate task id: {manifest.task_id}")
                    continue
                task_ids.add(manifest.task_id)
                previous_partition = group_partitions.setdefault(manifest.group_id, manifest.split)
                if previous_partition != manifest.split:
                    errors.append(
                        f"group leakage: {manifest.group_id!r} appears in both train and evaluation"
                    )
                    continue
                repository = repositories.get(manifest.source_id)
                locked = repository_locks.get(manifest.source_id)
                if repository is None or locked is None:
                    errors.append(
                        f"{manifest.task_id}: sourceId {manifest.source_id!r} is not a scanned repository"
                    )
                    continue
                if locked.get("status") == "blocked" or not locked.get("commit"):
                    errors.append(f"{manifest.task_id}: repository {manifest.source_id!r} is not locked")
                    continue
                source_path = _resolve(root, str(repository["uri"]))
                revision = (
                    str(locked["commit"])
                    if manifest.starting_revision == "locked"
                    else manifest.starting_revision
                )
                row = {
                    "prompt": [
                        {
                            "role": "system",
                            "content": (
                                "You are editing a disposable frontend repository. Use only the provided "
                                "bounded tools. Finish with a verified focused patch; never request network access."
                            ),
                        },
                        {"role": "user", "content": manifest.instruction},
                    ],
                    "task_id": manifest.task_id,
                    "manifest": payload,
                    "manifest_path": str(manifest_path),
                    "task_root": str(manifest_path.parent),
                    "source_path": str(source_path),
                    "source_revision": revision,
                    "environment_backend": config["environment"]["backend"],
                    "environment_image": config["environment"]["image"],
                    "max_tool_output_chars": config["environment"].get(
                        "max_tool_output_chars", 12_000
                    ),
                }
                rl_records[manifest.split].append(row)

    for records in (*sft_records.values(), *rl_records.values()):
        records.sort(key=lambda row: str(row.get("task_id", json.dumps(row, sort_keys=True))))

    artifacts: dict[str, str] = {}
    for partition, records in sft_records.items():
        if records:
            path = compiled_dir / "sft" / f"{partition}.jsonl"
            _atomic_jsonl(path, records)
            artifacts[f"sft_{partition}"] = str(path)
    for partition, records in rl_records.items():
        if records:
            path = compiled_dir / "rl" / f"{partition}.jsonl"
            _atomic_jsonl(path, records)
            artifacts[f"rl_{partition}"] = str(path)

    if not sft_records["train"]:
        warnings.append("no SFT demonstrations were compiled; repository code is not a substitute")
    if not rl_records["train"]:
        warnings.append("no executable training tasks were compiled for GRPO")
    if not rl_records["evaluation"]:
        warnings.append("no held-out executable evaluation tasks were compiled")

    fingerprint_payload = {
        "sft": sft_records,
        "rl": rl_records,
        "source_commits": {
            source_id: source.get("commit") for source_id, source in repository_locks.items()
        },
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    report = {
        "schema_version": 1,
        "fingerprint": fingerprint,
        "counts": {
            "sft_train": len(sft_records["train"]),
            "sft_evaluation": len(sft_records["evaluation"]),
            "rl_train": len(rl_records["train"]),
            "rl_evaluation": len(rl_records["evaluation"]),
        },
        "artifacts": artifacts,
        "errors": sorted(set(errors)),
        "warnings": sorted(set(warnings)),
    }
    report_path = compiled_dir / "compile-report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report["artifacts"]["report"] = str(report_path)
    return report


__all__ = ["compile_data"]
