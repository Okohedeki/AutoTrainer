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


def _within(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def _output_destinations(
    config: Mapping[str, Any], root: Path, artifact_dir: Path
) -> tuple[dict[tuple[str, str], Path], list[str]]:
    """Resolve declared trainer outputs and enforce the artifact boundary."""

    declarations = {
        ("sft", "train"): ("sft", "dataset", True),
        ("sft", "evaluation"): ("sft", "eval_dataset", False),
        ("rl", "train"): ("grpo", "dataset", True),
        ("rl", "evaluation"): ("grpo", "eval_dataset", False),
    }
    destinations: dict[tuple[str, str], Path] = {}
    errors: list[str] = []
    for key, (section_name, field_name, required) in declarations.items():
        section = config.get(section_name, {})
        if not isinstance(section, Mapping):
            errors.append(f"{section_name} must be a mapping")
            continue
        value = section.get(field_name)
        if value is None or value == "":
            if required and section.get("enabled", True) is not False:
                errors.append(f"{section_name}.{field_name} is required for compilation")
            continue
        if not isinstance(value, (str, Path)):
            errors.append(f"{section_name}.{field_name} must be a path")
            continue
        destination = _resolve(root, str(value))
        if destination.suffix.lower() != ".jsonl":
            errors.append(f"{section_name}.{field_name} must end in .jsonl: {destination}")
            continue
        if not _within(destination, artifact_dir):
            errors.append(
                f"{section_name}.{field_name} must stay inside project.artifact_dir "
                f"({artifact_dir}): {destination}"
            )
            continue
        destinations[key] = destination

    paths: dict[Path, list[str]] = {}
    for key, destination in destinations.items():
        paths.setdefault(destination, []).append(f"{key[0]}.{key[1]}")
    for destination, owners in sorted(paths.items(), key=lambda item: str(item[0])):
        if len(owners) > 1:
            errors.append(
                f"compiled dataset destinations collide at {destination}: {', '.join(sorted(owners))}"
            )
    return destinations, errors


def _source_input_paths(config: Mapping[str, Any], root: Path) -> set[Path]:
    paths: set[Path] = set()
    sources = config.get("sources", [])
    if not isinstance(sources, list):
        return paths
    for source in sources:
        if not isinstance(source, Mapping):
            continue
        uri = source.get("uri")
        if not isinstance(uri, str) or not uri or "://" in uri or uri.startswith("git@"):
            continue
        if any(character in uri for character in "*?["):
            continue
        paths.add(_resolve(root, uri))
    return paths


def _fingerprint(
    sft_records: Mapping[str, list[Mapping[str, Any]]],
    rl_records: Mapping[str, list[Mapping[str, Any]]],
    repository_locks: Mapping[str, Mapping[str, Any]],
    destinations: Mapping[tuple[str, str], Path],
    root: Path,
) -> str:
    destination_values = {
        f"{kind}_{partition}": (
            destination.relative_to(root).as_posix()
            if _within(destination, root)
            else str(destination)
        )
        for (kind, partition), destination in sorted(destinations.items())
    }
    payload = {
        "sft": sft_records,
        "rl": rl_records,
        "source_commits": {
            source_id: source.get("commit") for source_id, source in repository_locks.items()
        },
        "destinations": destination_values,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _report(
    *,
    sft_records: Mapping[str, list[Mapping[str, Any]]],
    rl_records: Mapping[str, list[Mapping[str, Any]]],
    repository_locks: Mapping[str, Mapping[str, Any]],
    destinations: Mapping[tuple[str, str], Path],
    root: Path,
    errors: list[str],
    warnings: list[str],
    artifacts: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "fingerprint": _fingerprint(
            sft_records, rl_records, repository_locks, destinations, root
        ),
        "counts": {
            "sft_train": len(sft_records["train"]),
            "sft_evaluation": len(sft_records["evaluation"]),
            "rl_train": len(rl_records["train"]),
            "rl_evaluation": len(rl_records["evaluation"]),
        },
        "artifacts": dict(artifacts or {}),
        "errors": sorted(set(errors)),
        "warnings": sorted(set(warnings)),
    }


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
    empty_sft: dict[str, list[Mapping[str, Any]]] = {"train": [], "evaluation": []}
    empty_rl: dict[str, list[Mapping[str, Any]]] = {"train": [], "evaluation": []}
    repository_locks = _repository_lock(scan)
    scan_errors = [f"source scan: {value}" for value in scan.get("errors", [])]
    if scan_errors:
        return _report(
            sft_records=empty_sft,
            rl_records=empty_rl,
            repository_locks=repository_locks,
            destinations={},
            root=root,
            errors=scan_errors,
            warnings=["compilation aborted before writing artifacts because source inspection failed"],
        )

    destinations, destination_errors = _output_destinations(config, root, artifact_dir)
    report_path = compiled_dir / "compile-report.json"
    source_paths = _source_input_paths(config, root)
    for key, destination in destinations.items():
        if destination == report_path:
            destination_errors.append(
                f"{key[0]}.{key[1]} collides with the compile report: {destination}"
            )
        if destination in source_paths:
            destination_errors.append(
                f"{key[0]}.{key[1]} would overwrite a declared source input: {destination}"
            )
    if destination_errors:
        return _report(
            sft_records=empty_sft,
            rl_records=empty_rl,
            repository_locks=repository_locks,
            destinations=destinations,
            root=root,
            errors=destination_errors,
            warnings=["compilation aborted before writing artifacts because output paths are unsafe"],
        )

    source_specs = config.get("sources", [])
    repositories = {
        str(source["id"]): source
        for source in source_specs
        if isinstance(source, Mapping) and source.get("kind") == "repository"
    }
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

    if not sft_records["train"]:
        warnings.append("no SFT demonstrations were compiled; repository code is not a substitute")
    if not rl_records["train"]:
        warnings.append("no executable training tasks were compiled for GRPO")
    if not rl_records["evaluation"]:
        warnings.append("no held-out executable evaluation tasks were compiled")

    for key, records in (
        (("sft", "train"), sft_records["train"]),
        (("sft", "evaluation"), sft_records["evaluation"]),
        (("rl", "train"), rl_records["train"]),
        (("rl", "evaluation"), rl_records["evaluation"]),
    ):
        if records and key not in destinations:
            section = "sft" if key[0] == "sft" else "grpo"
            field = "dataset" if key[1] == "train" else "eval_dataset"
            errors.append(
                f"compiled {key[0]} {key[1]} records require {section}.{field}"
            )

    if errors:
        warnings.append("compilation produced no dataset artifacts because validation failed")
        return _report(
            sft_records=sft_records,
            rl_records=rl_records,
            repository_locks=repository_locks,
            destinations=destinations,
            root=root,
            errors=errors,
            warnings=warnings,
        )

    artifacts: dict[str, str] = {}
    for key, records in (
        (("sft", "train"), sft_records["train"]),
        (("sft", "evaluation"), sft_records["evaluation"]),
        (("rl", "train"), rl_records["train"]),
        (("rl", "evaluation"), rl_records["evaluation"]),
    ):
        if not records:
            continue
        path = destinations[key]
        _atomic_jsonl(path, records)
        artifacts[f"{key[0]}_{key[1]}"] = str(path)

    artifacts["report"] = str(report_path)
    report = _report(
        sft_records=sft_records,
        rl_records=rl_records,
        repository_locks=repository_locks,
        destinations=destinations,
        root=root,
        errors=[],
        warnings=warnings,
        artifacts=artifacts,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


__all__ = ["compile_data"]
