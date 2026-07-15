"""Compile declared demonstrations and task packs into trainer-ready JSONL.

Raw repository files are never promoted to SFT records here.  A repository is
only attached to an executable task row or retained as inspectable evidence.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import tempfile
import time
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
    content = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for record in records
    )
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        # Each compiler owns its same-directory temporary file, so concurrent
        # attempts cannot truncate or delete one another's in-progress dataset.
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        _replace_with_retry(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _replace_with_retry(temporary: Path, destination: Path) -> None:
    """Finish an atomic replacement despite transient Windows sharing locks."""

    for attempt in range(12):
        try:
            os.replace(temporary, destination)
            return
        except PermissionError:
            # Retrying only the closed-file replacement preserves atomicity; it
            # never exposes or rewrites the destination in place.
            if attempt == 11:
                raise
            time.sleep(0.01)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    """Replace a JSON document without exposing a partially written report."""

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
        _replace_with_retry(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _invalidate_previous_report(path: Path) -> bool:
    """Fail closed before replacing artifacts from an earlier successful run."""

    if not path.exists() and not path.is_symlink():
        return False
    # If compilation crashes or returns an error after this point, direct
    # evaluation sees this explicit invalid state instead of trusting stale
    # dataset hashes and repository exposure from the preceding run.
    _atomic_json(
        path,
        {
            "schema_version": 1,
            "artifacts": {},
            "artifact_sha256": {},
            "repository_exposures": [],
            "errors": ["compilation is in progress; previous provenance was invalidated"],
            "warnings": [],
        },
    )
    return True


def invalidate_compile_provenance(
    config: Mapping[str, Any],
    project_root: Path,
) -> bool:
    """Invalidate a prior success before CLI source scanning can fail."""

    root = Path(project_root).expanduser().resolve()
    artifact_value = config.get("project", {}).get("artifact_dir", ".autotrainer")
    report_path = _resolve(root, str(artifact_value)) / "compiled" / "compile-report.json"
    return _invalidate_previous_report(report_path)


def _return_failed_report(
    path: Path,
    report: Mapping[str, Any],
    *,
    persist: bool,
) -> dict[str, Any]:
    """Persist a failed retry only when it must replace prior provenance."""

    result = dict(report)
    if persist:
        _atomic_json(path, result)
    return result


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
        # Held-out task records are final benchmark inputs. ``grpo.eval_dataset``
        # is deliberately absent because it belongs to the training loop and may
        # only point at a separately supplied validation set.
        ("rl", "evaluation"): ("evaluation", "dataset", True),
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
        "repository_exposures": _repository_exposures(repository_locks),
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
    artifact_sha256: Mapping[str, str] | None = None,
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
        "artifact_sha256": dict(artifact_sha256 or {}),
        # This is the compiler-frozen exposure ledger used by direct evaluation.
        # IDs aid diagnostics; identity and commit enforce the holdout boundary.
        "repository_exposures": _repository_exposures(repository_locks),
        "errors": sorted(set(errors)),
        "warnings": sorted(set(warnings)),
    }


def _repository_lock(scan: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(source["id"]): source
        for source in scan.get("sources", [])
        if isinstance(source, Mapping) and source.get("kind") == "repository"
    }


def _repository_exposures(
    repository_locks: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "source_id": source_id,
            "partition": source.get("partition"),
            "repository_identity": source.get("repository_identity"),
            "commit": source.get("commit"),
        }
        for source_id, source in sorted(repository_locks.items())
    ]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repository_exposure_warnings(
    repository_locks: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    """Warn when declared evidence alone invalidates repository holdout."""

    exposures = _repository_exposures(repository_locks)
    train = [item for item in exposures if item.get("partition") == "train"]
    evaluation = [item for item in exposures if item.get("partition") == "evaluation"]
    collisions: set[str] = set()
    for train_item in train:
        for held_out in evaluation:
            shared_identity = bool(train_item.get("repository_identity")) and (
                train_item.get("repository_identity") == held_out.get("repository_identity")
            )
            shared_commit = bool(train_item.get("commit")) and str(
                train_item.get("commit")
            ).lower() == str(held_out.get("commit", "")).lower()
            if shared_identity or shared_commit:
                reason = "repository identity" if shared_identity else "exact commit"
                collisions.add(
                    f"{train_item['source_id']} (train) and "
                    f"{held_out['source_id']} (evaluation) share {reason}"
                )
    if not collisions:
        return []
    return [
        "REPOSITORY HOLDOUT VIOLATION: declared repository exposure overlaps: "
        + "; ".join(sorted(collisions))
        + "; datasets were written for authoring only, and evaluation planning will refuse them"
    ]


def _task_source_id(row: Mapping[str, Any]) -> str:
    """Read the declared source ID from a compiled task row for diagnostics."""

    manifest = row.get("manifest", {})
    task = manifest.get("task", {}) if isinstance(manifest, Mapping) else {}
    return str(task.get("sourceId", "<unknown>")) if isinstance(task, Mapping) else "<unknown>"


def _repository_holdout_warnings(
    rl_records: Mapping[str, list[Mapping[str, Any]]],
) -> list[str]:
    """Flag authoring datasets that cannot support a repository-held-out claim.

    Compilation deliberately still writes these rows: the bundled fixture is
    useful for exercising training and environment contracts. The planner and
    evaluation planner are the hard gates that prevent publishing its result as
    a repository-held-out benchmark.
    """

    train_rows = rl_records.get("train", [])
    evaluation_rows = rl_records.get("evaluation", [])
    if not train_rows or not evaluation_rows:
        return []

    missing = sorted(
        {
            _task_source_id(row)
            for row in (*train_rows, *evaluation_rows)
            if not str(row.get("source_repository_identity", "")).strip()
        }
    )
    warnings: list[str] = []
    if missing:
        warnings.append(
            "REPOSITORY HOLDOUT UNVERIFIED: compiled task source(s) lack "
            "source_repository_identity: "
            + ", ".join(missing)
            + "; evaluation planning will refuse these rows"
        )

    collisions: set[str] = set()
    for train in train_rows:
        for held_out in evaluation_rows:
            train_identity = str(train.get("source_repository_identity", "")).strip()
            held_out_identity = str(
                held_out.get("source_repository_identity", "")
            ).strip()
            train_revision = str(train.get("source_revision", "")).strip().lower()
            held_out_revision = str(held_out.get("source_revision", "")).strip().lower()
            shared_identity = bool(train_identity) and train_identity == held_out_identity
            shared_revision = bool(train_revision) and train_revision == held_out_revision
            if shared_identity or shared_revision:
                reason = "repository identity" if shared_identity else "exact source revision"
                collisions.add(
                    f"{_task_source_id(train)} (train) and "
                    f"{_task_source_id(held_out)} (evaluation) share {reason}"
                )
    if collisions:
        warnings.append(
            "REPOSITORY HOLDOUT VIOLATION: "
            + "; ".join(sorted(collisions))
            + "; datasets were written for authoring only, and evaluation planning will refuse them"
        )
    return warnings


def compile_data(
    config: Mapping[str, Any],
    project_root: Path,
    scan: Mapping[str, Any],
) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    artifact_value = config.get("project", {}).get("artifact_dir", ".autotrainer")
    artifact_dir = _resolve(root, str(artifact_value))
    compiled_dir = artifact_dir / "compiled"
    report_path = compiled_dir / "compile-report.json"
    had_previous_report = _invalidate_previous_report(report_path)
    empty_sft: dict[str, list[Mapping[str, Any]]] = {"train": [], "evaluation": []}
    empty_rl: dict[str, list[Mapping[str, Any]]] = {"train": [], "evaluation": []}
    repository_locks = _repository_lock(scan)
    scan_errors = [f"source scan: {value}" for value in scan.get("errors", [])]
    # A failed retry may leave old dataset bytes for diagnosis, but the success
    # report was already invalidated, so no evaluation plan can trust them.
    if scan_errors:
        return _return_failed_report(
            report_path,
            _report(
                sft_records=empty_sft,
                rl_records=empty_rl,
                repository_locks=repository_locks,
                destinations={},
                root=root,
                errors=scan_errors,
                warnings=[
                    "compilation aborted before writing artifacts because source inspection failed"
                ],
            ),
            persist=had_previous_report,
        )

    destinations, destination_errors = _output_destinations(config, root, artifact_dir)
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
        return _return_failed_report(
            report_path,
            _report(
                sft_records=empty_sft,
                rl_records=empty_rl,
                repository_locks=repository_locks,
                destinations=destinations,
                root=root,
                errors=destination_errors,
                warnings=[
                    "compilation aborted before writing artifacts because output paths are unsafe"
                ],
            ),
            persist=had_previous_report,
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

    # Phase one builds and validates every record in memory. No declared trainer
    # dataset is touched until all sources and cross-split invariants pass.
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
                # Related task variants move as one group. Splitting variants
                # across train/evaluation would leak the same project family.
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
                    # Source IDs are user-selected labels. The scanner-derived
                    # identity is what downstream holdout gates compare.
                    "source_repository_identity": locked.get("repository_identity"),
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
    warnings.extend(_repository_exposure_warnings(repository_locks))
    warnings.extend(_repository_holdout_warnings(rl_records))

    for key, records in (
        (("sft", "train"), sft_records["train"]),
        (("sft", "evaluation"), sft_records["evaluation"]),
        (("rl", "train"), rl_records["train"]),
        (("rl", "evaluation"), rl_records["evaluation"]),
    ):
        if records and key not in destinations:
            if key == ("rl", "evaluation"):
                section, field = "evaluation", "dataset"
            else:
                section = "sft" if key[0] == "sft" else "grpo"
                field = "dataset" if key[1] == "train" else "eval_dataset"
            errors.append(
                f"compiled {key[0]} {key[1]} records require {section}.{field}"
            )

    if errors:
        warnings.append("compilation produced no dataset artifacts because validation failed")
        return _return_failed_report(
            report_path,
            _report(
                sft_records=sft_records,
                rl_records=rl_records,
                repository_locks=repository_locks,
                destinations=destinations,
                root=root,
                errors=errors,
                warnings=warnings,
            ),
            persist=had_previous_report,
        )

    # Phase two writes the complete validated set using per-file atomic replace.
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

    # Hash dataset bytes before writing the report. Direct evaluation verifies
    # these digests so a stale exposure ledger cannot bless replaced JSONL.
    artifact_sha256 = {
        key: _sha256_file(Path(path))
        for key, path in sorted(artifacts.items())
    }
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
        artifact_sha256=artifact_sha256,
    )
    _atomic_json(report_path, report)
    return report


__all__ = ["compile_data", "invalidate_compile_provenance"]
