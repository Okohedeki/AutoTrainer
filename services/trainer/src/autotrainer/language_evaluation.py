"""Shipped language-matched evaluation profiles for local code refinement."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
import subprocess
from typing import Any

from .config import (
    ConfigError,
    load_config,
    project_config_mutation,
    validate_mapping,
    write_config,
)


LANGUAGE_ORDER = ("python", "typescript_react", "csharp", "cpp")
_SUFFIX_LANGUAGE = {
    ".py": "python",
    ".pyi": "python",
    ".js": "typescript_react",
    ".jsx": "typescript_react",
    ".cjs": "typescript_react",
    ".mjs": "typescript_react",
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

LANGUAGE_SUITES: dict[str, dict[str, Any]] = {
    "python": {
        "label": "Python",
        "benchmark_inspirations": ["HumanEval", "MBPP"],
        "checks": ["python -m compileall", "python -m pytest"],
        "metrics": [
            "build_passed",
            "regression_pass_rate",
            "task_pass_rate",
            "code_quality_pass_rate",
            "pass_at_1",
        ],
    },
    "typescript_react": {
        "label": "TypeScript / React",
        "benchmark_inspirations": ["MultiPL-E", "HumanEval-X"],
        "checks": ["typecheck/build", "unit tests", "component/browser tests"],
        "metrics": [
            "build_passed",
            "regression_pass_rate",
            "task_pass_rate",
            "responsive_pass_rate",
            "design_rule_pass_rate",
            "code_quality_pass_rate",
            "pass_at_1",
        ],
    },
    "csharp": {
        "label": "C#",
        "benchmark_inspirations": ["MultiPL-E", "HumanEval-X"],
        "checks": ["dotnet build", "dotnet test"],
        "metrics": [
            "build_passed",
            "regression_pass_rate",
            "task_pass_rate",
            "code_quality_pass_rate",
            "pass_at_1",
        ],
    },
    "cpp": {
        "label": "C++",
        "benchmark_inspirations": ["MultiPL-E", "HumanEval-X"],
        "checks": ["compiler/build", "ctest or project test runner"],
        "metrics": [
            "build_passed",
            "regression_pass_rate",
            "task_pass_rate",
            "code_quality_pass_rate",
            "pass_at_1",
        ],
    },
}


def _artifact_dir(config: Any) -> Path:
    return config.artifact_dir.resolve()


def _counts_from_paths(paths: Sequence[str]) -> dict[str, int]:
    counts = {language: 0 for language in LANGUAGE_ORDER}
    for value in paths:
        language = _SUFFIX_LANGUAGE.get(Path(value).suffix.casefold())
        if language:
            counts[language] += 1
    return {key: value for key, value in counts.items() if value}


def _repository_paths(repository: Path, revision: str) -> list[str]:
    environment = {
        **os.environ,
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_TERMINAL_PROMPT": "0",
    }
    try:
        completed = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={repository.as_posix()}",
                "-C",
                str(repository),
                "ls-tree",
                "-r",
                "--name-only",
                revision,
            ],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            env=environment,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return []
    if completed.returncode:
        return []
    return completed.stdout.splitlines()[:100_000]


def _repository_language_counts(config: Any, partition: str) -> dict[str, int]:
    counts = {language: 0 for language in LANGUAGE_ORDER}
    for source in config.sources:
        if source.get("kind") != "repository" or source.get("partition") != partition:
            continue
        repository = config.resolve_path(str(source.get("uri", "")))
        revision = str(source.get("revision", ""))
        if not repository.is_dir() or not revision:
            continue
        for language, count in _counts_from_paths(
            _repository_paths(repository, revision)
        ).items():
            counts[language] += count
    return {key: value for key, value in counts.items() if value}


def _training_language_counts(config: Any) -> dict[str, int]:
    freeze_path = _artifact_dir(config) / "dataset" / "freeze.json"
    if freeze_path.is_file():
        try:
            receipt = json.loads(freeze_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            receipt = None
        counts = receipt.get("language_counts") if isinstance(receipt, Mapping) else None
        if isinstance(counts, Mapping):
            normalized: dict[str, int] = {}
            for language in LANGUAGE_ORDER:
                try:
                    count = int(counts.get(language, 0) or 0)
                except (TypeError, ValueError):
                    count = 0
                if count > 0:
                    normalized[language] = count
            if normalized:
                return normalized
    return _repository_language_counts(config, "train")


def _primary_language(counts: Mapping[str, int]) -> str | None:
    if not counts:
        return None
    return max(LANGUAGE_ORDER, key=lambda language: (int(counts.get(language, 0)), -LANGUAGE_ORDER.index(language)))


def get_language_evaluation_workspace(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    evaluation = config.data.get("evaluation", {})
    configured = (
        str(evaluation.get("language", "auto"))
        if isinstance(evaluation, Mapping)
        else "auto"
    )
    training_counts = _training_language_counts(config)
    evaluation_counts = _repository_language_counts(config, "evaluation")
    inferred = _primary_language(training_counts)
    selected = inferred if configured == "auto" else configured
    blockers: list[str] = []
    if inferred is None:
        blockers.append(
            "The frozen training dataset does not identify a supported primary language."
        )
    if selected not in LANGUAGE_SUITES:
        blockers.append(
            "Select the language used by the frozen training dataset before evaluation."
        )
    else:
        if inferred is not None and selected != inferred:
            blockers.append(
                f"The selected {LANGUAGE_SUITES[selected]['label']} suite does not match "
                f"the primary {LANGUAGE_SUITES[inferred]['label']} training language."
            )
        if not evaluation_counts:
            blockers.append(
                "Held-out repositories do not contain detectable Python, TypeScript / "
                "React, C#, or C++ code."
            )
        elif selected not in evaluation_counts:
            blockers.append(
                f"Held-out repositories do not contain {LANGUAGE_SUITES[selected]['label']} code."
            )
    return {
        "available": [
            {"id": language, **LANGUAGE_SUITES[language]}
            for language in LANGUAGE_ORDER
        ],
        "blockers": blockers,
        "configured": configured,
        "evaluation_language_counts": evaluation_counts,
        "inferred_training_language": inferred,
        "selected": selected,
        "selected_suite": LANGUAGE_SUITES.get(str(selected)),
        "status": "ready" if selected in LANGUAGE_SUITES and not blockers else "blocked",
        "training_language_counts": training_counts,
    }


def set_evaluation_language(config_path: str | Path, language: str) -> dict[str, Any]:
    selected = str(language).strip().casefold()
    if selected not in {"auto", *LANGUAGE_SUITES}:
        raise ConfigError(
            "language must be auto, python, typescript_react, csharp, or cpp"
        )
    with project_config_mutation(config_path):
        config = load_config(config_path)
        updated = dict(config.data)
        evaluation = dict(updated.get("evaluation", {}))
        evaluation["language"] = selected
        updated["evaluation"] = evaluation
        report = validate_mapping(updated, root=config.root)
        if report.errors:
            raise ConfigError("\n".join(report.errors))
        write_config(config.path, updated, overwrite=True)
        # The immutable run directories remain audit evidence. Only their
        # active pointer loses authority after the suite selection changes.
        (config.artifact_dir / "evaluation" / "current-plan.json").unlink(missing_ok=True)
    return get_language_evaluation_workspace(config_path)


def require_language_matched_evaluation(config_path: str | Path) -> dict[str, Any]:
    workspace = get_language_evaluation_workspace(config_path)
    if workspace["status"] != "ready":
        raise ConfigError(str(workspace["blockers"][0]))
    return workspace


__all__ = [
    "LANGUAGE_SUITES",
    "get_language_evaluation_workspace",
    "require_language_matched_evaluation",
    "set_evaluation_language",
]
