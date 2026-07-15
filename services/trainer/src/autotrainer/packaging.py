"""Build a portable, auditable LoRA adapter package after evaluation."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

import yaml

from .evaluation import EvaluationError, load_current_plan


class PackagingError(ValueError):
    """Raised when a verified adapter package cannot be assembled safely."""


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise PackagingError(f"{field} must be a mapping")
    return dict(value)


def _resolve(root: Path, value: Any, field: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise PackagingError(f"{field} must be a non-empty local path")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_tree(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise PackagingError(f"adapter directory does not exist: {source}")
    for candidate in source.rglob("*"):
        if candidate.is_symlink():
            raise PackagingError(f"packages do not accept symlinks: {candidate}")
    shutil.copytree(source, destination)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _artifact_dir(config: Mapping[str, Any], root: Path) -> Path:
    project = _mapping(config.get("project", {}), "project")
    return _resolve(root, project.get("artifact_dir", ".autotrainer"), "project.artifact_dir")


def _candidate_arm(plan: Mapping[str, Any]) -> str:
    candidates = {
        decision.get("candidate")
        for decision in plan.get("decisions", {}).values()
        if isinstance(decision, Mapping) and decision.get("candidate")
    }
    if len(candidates) != 1:
        raise PackagingError(
            "evaluation decisions must identify one shared candidate arm before packaging"
        )
    return str(next(iter(candidates)))


def _source_license_manifest(config: Mapping[str, Any]) -> dict[str, Any]:
    sources = config.get("sources", [])
    rows = []
    if isinstance(sources, list):
        for source in sources:
            if not isinstance(source, Mapping):
                continue
            license_value = source.get("license", {})
            license_mapping = license_value if isinstance(license_value, Mapping) else {}
            rows.append(
                {
                    "id": source.get("id"),
                    "kind": source.get("kind"),
                    "uri": source.get("uri"),
                    "revision": source.get("revision"),
                    "partition": source.get("partition"),
                    "license": {
                        "spdx": license_mapping.get("spdx", "UNDECLARED"),
                        "attribution": license_mapping.get("attribution"),
                    },
                }
            )
    return {"schema_version": 1, "sources": rows}


def _first_evaluation_task(plan: Mapping[str, Any]) -> Mapping[str, Any] | None:
    rows = plan.get("task_rows", {})
    if not isinstance(rows, Mapping) or not rows:
        return None
    first_key = sorted(str(key) for key in rows)[0]
    value = rows[first_key]
    return value if isinstance(value, Mapping) else None


def _file_manifest(root: Path) -> list[dict[str, Any]]:
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "package-manifest.json":
            files.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    return files


def build_adapter_package(
    config: Mapping[str, Any],
    project_root: Path,
    *,
    config_path: Path | None = None,
    output_dir: Path | None = None,
    allow_unverified: bool = False,
) -> dict[str, Any]:
    """Assemble an immutable adapter package and refuse unsupported winner claims."""

    root = Path(project_root).expanduser().resolve()
    try:
        plan, evaluation_dir = load_current_plan(config, root)
    except EvaluationError as error:
        raise PackagingError(str(error)) from error
    summary_path = evaluation_dir / "summary.json"
    if not summary_path.is_file():
        raise PackagingError("evaluation summary is missing; run `autotrainer evaluate report`")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    verified = summary.get("v1_success_criteria_verified") is True
    if not verified and not allow_unverified:
        raise PackagingError(
            "both V1 success criteria must be verified before packaging a winner; "
            "use --allow-unverified only for a clearly marked development artifact"
        )

    package = _mapping(config.get("package", {}), "package")
    package_name = str(package.get("name") or config.get("project", {}).get("name") or "autotrainer-adapter")
    destination = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else _resolve(
            root,
            package.get("output_dir", _artifact_dir(config, root) / "packages" / package_name),
            "package.output_dir",
        )
    )
    if destination.exists():
        raise PackagingError(f"refusing to overwrite an existing package: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    candidate_id = _candidate_arm(plan)
    candidate = _mapping(plan.get("arms", {}).get(candidate_id), f"evaluation arm {candidate_id}")
    adapter = _mapping(candidate.get("adapter"), f"evaluation arm {candidate_id}.adapter")
    adapter_path = _resolve(root, adapter.get("path"), "candidate adapter path")
    include = package.get(
        "include",
        ["adapter", "evaluation_report", "source_license_manifest", "resolved_recipe"],
    )
    if not isinstance(include, list) or not all(isinstance(item, str) for item in include):
        raise PackagingError("package.include must be a list of artifact names")

    with tempfile.TemporaryDirectory(prefix=f".{package_name}-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "package"
        staging.mkdir()
        if "adapter" in include:
            _copy_tree(adapter_path, staging / "adapter")
        if "evaluation_report" in include:
            reports = evaluation_dir / "reports"
            if not reports.is_dir():
                raise PackagingError("evaluation reports are missing")
            _copy_tree(reports, staging / "evaluation" / "reports")
            shutil.copyfile(summary_path, staging / "evaluation" / "summary.json")
        if "source_license_manifest" in include:
            _write_json(staging / "source-license-manifest.json", _source_license_manifest(config))
        if "resolved_recipe" in include:
            recipe_path = adapter_path / "resolved_recipe.json"
            if recipe_path.is_file():
                shutil.copyfile(recipe_path, staging / "resolved-recipe.json")
            else:
                _write_json(
                    staging / "resolved-recipe.json",
                    {
                        "model": candidate["model"],
                        "adapter": adapter,
                        "qlora": config.get("qlora"),
                        "sft": config.get("sft"),
                        "grpo": config.get("grpo"),
                    },
                )
        task_row = _first_evaluation_task(plan)
        if "system_prompt" in include:
            prompt = ""
            if task_row:
                messages = task_row.get("prompt", [])
                if isinstance(messages, list):
                    prompt = next(
                        (
                            str(message.get("content"))
                            for message in messages
                            if isinstance(message, Mapping) and message.get("role") == "system"
                        ),
                        "",
                    )
            (staging / "system-prompt.md").write_text(prompt + "\n", encoding="utf-8")
        if "tool_schema" in include:
            manifest = task_row.get("manifest", {}) if task_row else {}
            _write_json(staging / "tool-schema.json", {"tools": manifest.get("tools", [])})
        if config_path is not None and Path(config_path).is_file():
            shutil.copyfile(Path(config_path), staging / "autotrainer.yaml")
        else:
            (staging / "autotrainer.yaml").write_text(
                yaml.safe_dump(dict(config), sort_keys=False, width=100), encoding="utf-8"
            )
        lock_path = _artifact_dir(config, root) / "autotrainer.lock.json"
        if lock_path.is_file():
            shutil.copyfile(lock_path, staging / "autotrainer.lock.json")
        manifest = {
            "schema_version": 1,
            "package_name": package_name,
            "status": "verified_winner" if verified else "unverified_development_artifact",
            "candidate_id": candidate_id,
            "base_model": candidate["model"],
            "adapter_sha256": adapter["sha256"],
            "evaluation_plan_id": plan["plan_id"],
            "v1_success_criteria_verified": verified,
            "merge_base_weights": False,
            "files": _file_manifest(staging),
        }
        _write_json(staging / "package-manifest.json", manifest)
        shutil.copytree(staging, destination)
    return {
        "status": manifest["status"],
        "candidate_id": candidate_id,
        "output_dir": str(destination),
        "manifest": str(destination / "package-manifest.json"),
        "file_count": len(manifest["files"]) + 1,
    }


__all__ = ["PackagingError", "build_adapter_package"]
