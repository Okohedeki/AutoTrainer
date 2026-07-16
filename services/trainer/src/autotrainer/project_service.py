"""Shared project preparation used by the human API and agent CLI.

Preparation is deliberately bounded: it validates and materializes deterministic
inputs, writes the static plan, and checks the host.  It never loads a model or
starts a training process.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import json
from pathlib import Path
import re
from typing import Any

import yaml

from .compiler import compile_data
from .config import ConfigError, ProjectConfig, validate_mapping
from .doctor import run_doctor
from .planner import build_plan
from .sources import scan_sources


STEP_LABELS = {
    "validate": "Validate inputs",
    "sources": "Scan sources",
    "compile": "Compile training data",
    "runtime": "Check local runtime",
}


def _read_project(config_path: str | Path) -> ProjectConfig:
    """Read YAML even when only the later proof section is incomplete."""

    path = Path(config_path).expanduser().resolve()
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ConfigError(f"configuration not found: {path}") from error
    except yaml.YAMLError as error:
        raise ConfigError(f"invalid YAML in {path}: {error}") from error
    if not isinstance(payload, Mapping):
        raise ConfigError("configuration root must be a YAML mapping")
    return ProjectConfig(path=path, data=dict(payload))


def _evaluation_source_indexes(data: Mapping[str, Any]) -> set[int]:
    sources = data.get("sources", [])
    if not isinstance(sources, list):
        return set()
    return {
        index
        for index, source in enumerate(sources)
        if isinstance(source, Mapping) and source.get("partition") == "evaluation"
    }


def _proof_validation_error(error: str, data: Mapping[str, Any]) -> bool:
    """Separate unfinished final proof wiring from train-input correctness."""

    if error.startswith(("evaluation.", "model_benchmark ", "fable_ab ")):
        return True
    match = re.match(r"sources\[(\d+)\]", error)
    return bool(match and int(match.group(1)) in _evaluation_source_indexes(data))


def _proof_scan_error(error: str, data: Mapping[str, Any]) -> bool:
    source_id = error.split(":", 1)[0]
    sources = data.get("sources", [])
    return any(
        isinstance(source, Mapping)
        and str(source.get("id")) == source_id
        and source.get("partition") == "evaluation"
        for source in sources if isinstance(sources, list)
    )


def _training_config(data: Mapping[str, Any]) -> dict[str, Any]:
    """Compile only train-partition inputs; final proof is a later checkpoint."""

    result = deepcopy(dict(data))
    sources = result.get("sources", [])
    if isinstance(sources, list):
        result["sources"] = [
            source
            for source in sources
            if isinstance(source, Mapping) and source.get("partition") == "train"
        ]
    return result


def _recipe(scan: Mapping[str, Any]) -> str:
    summary = scan.get("summary", {})
    if not isinstance(summary, Mapping):
        summary = {}
    has_examples = int(summary.get("valid_sft_record_count", 0) or 0) > 0
    has_tasks = int(summary.get("train_ready_task_count", 0) or 0) > 0
    if has_examples and has_tasks:
        return "both"
    if has_examples:
        return "teach"
    if has_tasks:
        return "practice"
    return "needs_training_data"


def _training_plan_blockers(plan: Mapping[str, Any], recipe: str) -> list[str]:
    blockers: list[str] = []
    model = plan.get("model", {})
    evidence = plan.get("evidence", {})
    stages = plan.get("stages", {})
    for section in (model, evidence):
        if isinstance(section, Mapping):
            blockers.extend(str(value) for value in section.get("blockers", []))
    if isinstance(stages, Mapping):
        selected = []
        if recipe in {"teach", "both"}:
            selected.append(stages.get("sft", {}))
        if recipe in {"practice", "both"}:
            selected.append(stages.get("grpo", {}))
        for stage in selected:
            if isinstance(stage, Mapping):
                blockers.extend(str(value) for value in stage.get("blockers", []))
    return list(dict.fromkeys(blockers))


def _write_plan(config: ProjectConfig, plan: dict[str, Any]) -> None:
    config.artifact_dir.mkdir(parents=True, exist_ok=True)
    destination = config.artifact_dir / "plan.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)
    plan["artifact"] = str(destination)


def _doctor_ready(doctor: Mapping[str, Any], recipe: str) -> bool:
    if recipe == "teach":
        return doctor.get("sft_ready") is True
    if recipe in {"practice", "both"}:
        return doctor.get("rl_ready") is True
    return False


def _step(step_id: str, status: str) -> dict[str, str]:
    return {"id": step_id, "label": STEP_LABELS[step_id], "status": status}


def prepare_project(config_path: str | Path) -> dict[str, Any]:
    """Prepare deterministic training inputs and return one actionable blocker."""

    config = _read_project(config_path)
    validation_report = validate_mapping(config.data, root=config.root)
    validation_errors = list(validation_report.errors)
    later_validation = [
        error for error in validation_errors if _proof_validation_error(error, config.data)
    ]
    blocking_validation = [error for error in validation_errors if error not in later_validation]
    validation = {
        "valid": not validation_errors,
        "training_valid": not blocking_validation,
        "errors": validation_errors,
        "blocking_errors": blocking_validation,
        "later_proof": later_validation,
        "warnings": list(validation_report.warnings),
    }

    # A full read-only scan preserves proof diagnostics. The lock and compiled
    # datasets intentionally contain train sources only, so an unfinished Fable
    # or held-out source cannot prevent preparation of usable training inputs.
    try:
        full_scan = scan_sources(config.data, config.root, write=False)
    except Exception as error:  # defensive product boundary around filesystem/git inspection
        full_scan = {"errors": [str(error)], "warnings": [], "sources": [], "summary": {}}
    scan_errors = [str(value) for value in full_scan.get("errors", [])]
    later_scan = [error for error in scan_errors if _proof_scan_error(error, config.data)]
    blocking_scan = [error for error in scan_errors if error not in later_scan]

    training_data = _training_config(config.data)
    training_scan: dict[str, Any] = {"errors": [], "warnings": [], "summary": {}}
    compile_result: dict[str, Any] = {
        "status": "skipped",
        "errors": [],
        "warnings": ["compile waits for valid training inputs"],
    }
    if not blocking_validation and not blocking_scan:
        try:
            training_scan = scan_sources(training_data, config.root, write=True)
            if not training_scan.get("errors"):
                compile_result = compile_data(training_data, config.root, training_scan)
        except Exception as error:  # return a blocker instead of a false completed state
            compile_result = {"status": "blocked", "errors": [str(error)], "warnings": []}
    scan_detail = dict(full_scan)
    scan_detail["blocking_errors"] = blocking_scan
    scan_detail["later_proof"] = later_scan
    scan_detail["training"] = training_scan

    recipe = _recipe(training_scan if training_scan.get("summary") else full_scan)

    try:
        plan = build_plan(config.data, config.root, full_scan)
        _write_plan(config, plan)
    except Exception as error:
        plan = {"status": "blocked", "errors": [str(error)], "blockers": [str(error)]}
    plan_blockers = _training_plan_blockers(plan, recipe)

    environment = config.data.get("environment", {})
    backend = str(environment.get("backend", "docker")) if isinstance(environment, Mapping) else "docker"
    try:
        doctor = run_doctor(environment_backend=backend)
    except Exception as error:
        doctor = {"sft_ready": False, "rl_ready": False, "errors": [str(error)]}

    compile_errors = [str(value) for value in compile_result.get("errors", [])]
    steps: list[dict[str, str]] = []
    next_action: dict[str, str] | None = None
    if blocking_validation:
        steps = [_step("validate", "blocked"), _step("sources", "waiting"), _step("compile", "waiting"), _step("runtime", "waiting")]
        next_action = {"title": "Fix project configuration", "detail": blocking_validation[0]}
    elif blocking_scan or training_scan.get("errors"):
        source_error = blocking_scan or [str(value) for value in training_scan.get("errors", [])]
        steps = [_step("validate", "complete"), _step("sources", "blocked"), _step("compile", "waiting"), _step("runtime", "waiting")]
        next_action = {"title": "Fix the first source", "detail": source_error[0]}
    elif recipe == "needs_training_data":
        steps = [_step("validate", "complete"), _step("sources", "complete"), _step("compile", "blocked"), _step("runtime", "waiting")]
        next_action = {"title": "Add training data", "detail": "Add accepted examples, executable tasks, or both."}
    elif compile_errors:
        steps = [_step("validate", "complete"), _step("sources", "complete"), _step("compile", "blocked"), _step("runtime", "waiting")]
        next_action = {"title": "Fix compiled training data", "detail": compile_errors[0]}
    elif plan_blockers:
        steps = [_step("validate", "complete"), _step("sources", "complete"), _step("compile", "blocked"), _step("runtime", "waiting")]
        next_action = {"title": "Complete the training plan", "detail": plan_blockers[0]}
    elif not _doctor_ready(doctor, recipe):
        steps = [_step("validate", "complete"), _step("sources", "complete"), _step("compile", "complete"), _step("runtime", "blocked")]
        next_action = {"title": "Prepare the local runtime", "detail": "Resolve the first GPU, Python package, or sandbox blocker reported by Doctor."}
    else:
        steps = [_step("validate", "complete"), _step("sources", "complete"), _step("compile", "complete"), _step("runtime", "complete")]

    status = "ready" if next_action is None else "blocked"
    summary = (
        "Training inputs and the local runtime are ready."
        if status == "ready"
        else next_action["detail"]
    )
    return {
        "status": status,
        "recipe": recipe,
        "summary": summary,
        "next_action": next_action,
        "steps": steps,
        "details": {
            "validation": validation,
            "scan": scan_detail,
            "compile": compile_result,
            "plan": plan,
            "doctor": doctor,
        },
    }


__all__ = ["prepare_project"]
