"""Truthful GRPO curriculum catalogue and retained rollout observations.

The catalogue has three deliberately separate kinds of evidence:

* declared task files are editable drafts;
* compiled rows are the immutable definitions a trainer will consume; and
* rollout events belong to one job bound to one compiled dataset digest.

Keeping those boundaries explicit prevents a dashboard refresh from making a
changed task file, stale job, or empty metric look like training proof.
"""

from __future__ import annotations

from collections import Counter
import glob
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any, Mapping, Sequence

from .manifest import REWARD_KEYS, V1_TOOLS, TaskManifest
from .project_service import read_project_config


_RUBRIC_COMPONENTS = (
    "design_rules",
    "patch_quality",
    "regression_safety",
    "responsive_rules",
    "task_tests",
)
_TASK_FILE_LIMIT = 5_000
_TASK_FILE_BYTES = 1_000_000
_COMPILED_DATASET_BYTES = 256 * 1024 * 1024
_MIN_STABLE_OBSERVATIONS = 4
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _display_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        # Do not send arbitrary absolute host paths to the browser.
        return path.name


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_digest(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _task_files(root: Path, uri: str) -> list[Path]:
    """Discover a bounded set of JSON manifests from one task-pack URI."""

    if not uri.strip():
        return []
    if any(character in uri for character in "*?["):
        pattern = Path(uri).expanduser()
        if not pattern.is_absolute():
            pattern = root / pattern
        candidates = (Path(value).resolve() for value in glob.glob(str(pattern), recursive=True))
    else:
        path = _resolve(root, uri)
        candidates = path.rglob("*.json") if path.is_dir() else [path]
    return sorted(
        candidate.resolve()
        for candidate in candidates
        if candidate.is_file() and candidate.suffix.lower() == ".json"
    )[:_TASK_FILE_LIMIT]


def _blocked_catalog(message: str) -> dict[str, Any]:
    return {
        "status": "blocked",
        "fingerprint": None,
        "dataset_sha256": None,
        "dataset": None,
        "task_count": 0,
        "tasks": [],
        "blockers": [message],
    }


def load_compiled_catalog(config: Any, *, report: object | None = None) -> dict[str, Any]:
    """Verify compiler provenance and load exact embedded train manifests.

    This function fails closed. Task IDs from an arbitrary JSONL file are not
    enough: the report path, configured dataset path, SHA-256, and every row's
    embedded manifest must agree before the word ``compiled`` is used.
    """

    report_path = config.artifact_dir / "compiled" / "compile-report.json"
    if report is None:
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {
                **_blocked_catalog("Prepare the project to compile executable GRPO tasks."),
                "status": "not_prepared",
            }
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return _blocked_catalog("The compiler provenance report cannot be verified.")
    if not isinstance(report, Mapping) or report.get("schema_version") != 1:
        return _blocked_catalog("The compiler provenance report has an unsupported schema.")
    errors = report.get("errors")
    if not isinstance(errors, list) or errors:
        return _blocked_catalog("The compiler provenance report records compilation errors.")

    fingerprint = str(report.get("fingerprint", "")).lower()
    artifacts = report.get("artifacts")
    hashes = report.get("artifact_sha256")
    if (
        _SHA256.fullmatch(fingerprint) is None
        or not isinstance(artifacts, Mapping)
        or not isinstance(hashes, Mapping)
    ):
        return _blocked_catalog("The compiler report lacks a valid fingerprint or artifact ledger.")

    reported_value = artifacts.get("rl_train")
    expected_digest = str(hashes.get("rl_train", "")).lower()
    grpo = config.data.get("grpo", {})
    dataset_value = grpo.get("dataset") if isinstance(grpo, Mapping) else None
    if not isinstance(reported_value, str) or not isinstance(dataset_value, str):
        return _blocked_catalog("No compiled GRPO train artifact is recorded.")
    if _SHA256.fullmatch(expected_digest) is None:
        return _blocked_catalog("The compiled GRPO train artifact has no valid SHA-256 digest.")

    dataset = _resolve(config.root, reported_value)
    configured_dataset = config.resolve_path(dataset_value)
    if dataset != configured_dataset:
        return _blocked_catalog("The configured GRPO dataset does not match compiler provenance.")
    try:
        size = dataset.stat().st_size
        if size > _COMPILED_DATASET_BYTES:
            return _blocked_catalog("The compiled GRPO dataset exceeds the inspection limit.")
        if _sha256_file(dataset) != expected_digest:
            return _blocked_catalog("The compiled GRPO dataset bytes do not match provenance.")
    except OSError:
        return _blocked_catalog("The compiled GRPO dataset is missing or unreadable.")

    tasks: list[dict[str, Any]] = []
    seen: set[str] = set()
    try:
        with dataset.open("r", encoding="utf-8") as handle:
            for index, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                if len(tasks) >= _TASK_FILE_LIMIT:
                    return _blocked_catalog("The compiled GRPO dataset exceeds the task limit.")
                row = json.loads(line)
                if not isinstance(row, Mapping) or not isinstance(row.get("manifest"), Mapping):
                    return _blocked_catalog(f"Compiled GRPO row {index} has no embedded manifest.")
                manifest_payload = dict(row["manifest"])
                manifest = TaskManifest.from_mapping(manifest_payload)
                if manifest.split != "train":
                    return _blocked_catalog(f"Compiled GRPO row {index} is not a training task.")
                if row.get("task_id") != manifest.task_id or manifest.task_id in seen:
                    return _blocked_catalog(f"Compiled GRPO row {index} has an invalid or duplicate task ID.")
                seen.add(manifest.task_id)
                raw_manifest_path = str(row.get("manifest_path", ""))
                manifest_path = _resolve(config.root, raw_manifest_path) if raw_manifest_path else dataset
                tasks.append(
                    {
                        "manifest": manifest,
                        "manifest_payload": manifest_payload,
                        "manifest_path": _display_path(manifest_path, config.root),
                        "source_revision": str(
                            row.get("source_revision", manifest.starting_revision)
                        ),
                    }
                )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError, KeyError) as error:
        return _blocked_catalog(f"The compiled GRPO dataset is invalid: {error}")

    return {
        "status": "compiled",
        "fingerprint": fingerprint,
        "dataset_sha256": expected_digest,
        "dataset": _display_path(dataset, config.root),
        "task_count": len(tasks),
        "tasks": tasks,
        "blockers": [],
    }


def _declared_tasks(config: Any) -> list[dict[str, Any]]:
    """Read editable declarations without implying that Prepare consumed them."""

    declared: list[dict[str, Any]] = []
    raw_sources = config.data.get("sources", [])
    sources = raw_sources if isinstance(raw_sources, list) else []
    for source in sources:
        if not isinstance(source, Mapping) or source.get("kind") != "task_pack":
            continue
        source_id = str(source.get("id", "tasks"))
        source_partition = str(source.get("partition", "train"))
        for manifest_path in _task_files(config.root, str(source.get("uri", ""))):
            if len(declared) >= _TASK_FILE_LIMIT:
                break
            try:
                if manifest_path.stat().st_size > _TASK_FILE_BYTES:
                    raise ValueError("task manifest exceeds the 1 MB inspection limit")
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                if not isinstance(payload, Mapping):
                    raise ValueError("manifest root must be a JSON object")
                manifest = TaskManifest.from_mapping(payload)
                blockers = []
                if manifest.split != source_partition:
                    blockers.append(
                        f"task split {manifest.split!r} does not match source partition {source_partition!r}"
                    )
                declared.append(
                    {
                        "manifest": manifest,
                        "manifest_payload": dict(payload),
                        "manifest_path": _display_path(manifest_path, config.root),
                        "source_id": source_id,
                        "source_partition": source_partition,
                        "blockers": blockers,
                    }
                )
            except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError, KeyError) as error:
                declared.append(
                    {
                        "manifest": None,
                        "manifest_payload": None,
                        "manifest_path": _display_path(manifest_path, config.root),
                        "source_id": source_id,
                        "source_partition": source_partition,
                        "blockers": [str(error)],
                    }
                )
    return declared


def _safe_unit(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if math.isfinite(number) and 0.0 <= number <= 1.0:
            return number
    return None


def _safe_count(value: object, *, maximum: int = 1_000_000) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= maximum:
        return value
    return None


def _mean(values: Sequence[float]) -> float | None:
    return round(statistics.fmean(values), 4) if values else None


def _rollout_record(value: Mapping[str, Any]) -> dict[str, Any] | None:
    task_id = value.get("task_id")
    reward = _safe_unit(value.get("reward"))
    raw_rubric = value.get("rubric")
    sequence = value.get("sequence")
    if (
        not isinstance(task_id, str)
        or not task_id
        or reward is None
        or not isinstance(raw_rubric, Mapping)
        or not isinstance(sequence, int)
        or isinstance(sequence, bool)
    ):
        return None
    rubric = {name: _safe_unit(raw_rubric.get(name)) for name in _RUBRIC_COMPONENTS}
    if any(component is None for component in rubric.values()):
        return None
    result: dict[str, Any] = {
        "sequence": sequence,
        "observed_at": str(value.get("observed_at", "")),
        "episode_id": str(value["episode_id"]) if value.get("episode_id") else None,
        "task_id": task_id,
        "reward": reward,
        "hard_gate_passed": value.get("hard_gate_passed") is True,
        "gate_reason": str(value["gate_reason"]) if value.get("gate_reason") else None,
        "rubric": {name: float(rubric[name]) for name in _RUBRIC_COMPONENTS},
    }
    for field in ("tool_call_count", "changed_file_count"):
        count = _safe_count(value.get(field))
        result[field] = count
    elapsed = value.get("elapsed_seconds")
    result["elapsed_seconds"] = (
        round(float(elapsed), 4)
        if isinstance(elapsed, (int, float))
        and not isinstance(elapsed, bool)
        and math.isfinite(float(elapsed))
        and 0 <= float(elapsed) <= 7 * 24 * 60 * 60
        else None
    )
    raw_tools = value.get("tool_calls_by_name")
    tools: dict[str, int] = {}
    if isinstance(raw_tools, Mapping):
        for raw_name, raw_count in raw_tools.items():
            name = str(raw_name)
            count = _safe_count(raw_count, maximum=100_000)
            if name in V1_TOOLS and count is not None:
                tools[name] = count
    result["tool_calls_by_name"] = dict(sorted(tools.items()))
    return result


def _observations(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rewards = [float(event["reward"]) for event in events]
    passes = [event.get("hard_gate_passed") is True for event in events]
    count = len(events)
    reward_range = round(max(rewards) - min(rewards), 4) if rewards else None
    if count == 0:
        outcome_mix = "unobserved"
        recommendation = "Collect verified rollouts before judging this task."
    elif count < _MIN_STABLE_OBSERVATIONS:
        outcome_mix = "uncalibrated"
        recommendation = f"Collect at least {_MIN_STABLE_OBSERVATIONS} rollouts before reading the pattern."
    elif reward_range is not None and reward_range <= 1e-8:
        outcome_mix = "flat"
        recommendation = "This retained window has no relative reward spread; inspect task difficulty and verifier sensitivity."
    else:
        outcome_mix = "varied"
        recommendation = "This retained window contains relative reward variation. Confirm it against a frozen base-policy calibration."

    if not passes:
        gate_pattern = "unobserved"
    elif all(passes):
        gate_pattern = "all_passed"
    elif not any(passes):
        gate_pattern = "all_gated"
    else:
        gate_pattern = "mixed"
    gate_reasons = Counter(
        str(event["gate_reason"]) for event in events if event.get("gate_reason")
    )
    return {
        "rollout_count": count,
        "hard_gate_pass_count": sum(passes),
        "hard_gate_pass_rate": round(sum(passes) / count, 4) if count else None,
        "reward_mean": _mean(rewards),
        "reward_min": round(min(rewards), 4) if rewards else None,
        "reward_max": round(max(rewards), 4) if rewards else None,
        "reward_range": reward_range,
        "reward_variance": (
            round(statistics.pvariance(rewards), 6) if rewards else None
        ),
        "rubric_means": {
            name: _mean([float(event["rubric"][name]) for event in events])
            for name in _RUBRIC_COMPONENTS
        },
        "gate_reasons": dict(sorted(gate_reasons.items())),
        "outcome_mix": outcome_mix,
        "gate_pattern": gate_pattern,
        "recommendation": recommendation,
    }


def _task_record(
    manifest: TaskManifest,
    *,
    manifest_path: str,
    status: str,
    blockers: Sequence[str],
    source_revision: str,
    declaration_state: str,
    observations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "id": manifest.task_id,
        "instruction": manifest.instruction,
        "task_family_id": manifest.group_id,
        "source_id": manifest.source_id,
        "source_revision": source_revision,
        "split": manifest.split,
        "manifest": manifest_path,
        "status": status,
        "declaration_state": declaration_state,
        "blockers": list(blockers),
        "working_directory": manifest.working_directory,
        "tools": list(manifest.tools),
        "checks": {
            "build": bool(manifest.runtime_commands.get("build")),
            "tests": bool(manifest.runtime_commands.get("tests")),
            "browser_tests": bool(manifest.runtime_commands.get("browserTests")),
            "hidden_verifier": bool(manifest.verifier_bundle and manifest.verifier_command),
        },
        "limits": {
            "tool_calls": manifest.tool_call_limit,
            "command_timeout_seconds": manifest.command_timeout_seconds,
            "episode_timeout_seconds": manifest.episode_timeout_seconds,
            "network_access": manifest.network_access,
        },
        "reward_weights": {
            "regression_safety": manifest.reward_weights["regressionSafety"],
            "task_tests": manifest.reward_weights["taskTests"],
            "responsive_rules": manifest.reward_weights["responsiveRules"],
            "design_rules": manifest.reward_weights["designRules"],
            "patch_quality": manifest.reward_weights["patchQuality"],
        },
        "aspects": {
            "instruction": bool(manifest.instruction),
            "locked_snapshot": bool(manifest.source_id and source_revision),
            "bounded_tools": bool(manifest.tools and manifest.tool_call_limit),
            "hidden_verifier": bool(manifest.verifier_bundle and manifest.verifier_command),
            "reward_contract": set(manifest.reward_weights) == REWARD_KEYS,
        },
        "observed": _observations(observations),
    }


def _empty_activity() -> dict[str, Any]:
    return {
        "job_id": None,
        "status": "idle",
        "stage": None,
        "events": [],
        "window": {
            "scope": "current_job_retained_window",
            "first_sequence": None,
            "last_sequence": None,
            "retained_event_count": 0,
            "observed_event_count": 0,
            "truncated": False,
        },
    }


def _run_evidence(
    activity: Mapping[str, Any], catalog: Mapping[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw_events = activity.get("events", [])
    events = raw_events if isinstance(raw_events, list) else []
    binding: Mapping[str, Any] | None = None
    for value in events:
        if (
            isinstance(value, Mapping)
            and value.get("type") == "stage_completed"
            and value.get("stage") == "prepare"
            and value.get("catalog_fingerprint")
            and value.get("dataset_sha256")
        ):
            binding = value

    job_id = activity.get("job_id")
    if not job_id:
        alignment = "not_applicable"
    elif catalog.get("status") != "compiled":
        alignment = "unavailable"
    elif binding is None:
        alignment = "unknown"
    elif (
        binding.get("catalog_fingerprint") == catalog.get("fingerprint")
        and binding.get("dataset_sha256") == catalog.get("dataset_sha256")
    ):
        alignment = "matched"
    else:
        alignment = "mismatch"

    binding_sequence = int(binding.get("sequence", 0)) if binding is not None else 0
    rollouts: list[dict[str, Any]] = []
    active: dict[str, dict[str, Any]] = {}
    for value in events:
        if not isinstance(value, Mapping):
            continue
        sequence = value.get("sequence")
        if not isinstance(sequence, int) or sequence <= binding_sequence:
            continue
        if value.get("type") == "episode_started" and value.get("episode_id"):
            active[str(value["episode_id"])] = {
                "sequence": sequence,
                "observed_at": str(value.get("observed_at", "")),
                "episode_id": str(value["episode_id"]),
                "task_id": str(value.get("task_id", "unknown-task")),
                "task_family_id": (
                    str(value["task_family_id"])
                    if value.get("task_family_id")
                    else None
                ),
            }
        elif value.get("type") == "episode_scored":
            rollout = _rollout_record(value)
            if rollout is not None:
                rollouts.append(rollout)
                if rollout.get("episode_id"):
                    active.pop(str(rollout["episode_id"]), None)

    # Evidence is visible but never merged into the current catalogue unless
    # the current job explicitly recorded the exact compiler fingerprint.
    run = {
        "job_id": job_id,
        "status": str(activity.get("status", "idle")),
        "stage": activity.get("stage"),
        "catalog_alignment": alignment,
        "bound_catalog_fingerprint": (
            binding.get("catalog_fingerprint") if binding is not None else None
        ),
        "bound_dataset_sha256": (
            binding.get("dataset_sha256") if binding is not None else None
        ),
        "window": dict(activity.get("window", {}))
        if isinstance(activity.get("window"), Mapping)
        else _empty_activity()["window"],
        "active_episodes": sorted(active.values(), key=lambda item: int(item["sequence"])),
    }
    return run, rollouts


def get_curriculum_workspace(
    config_path: str | Path,
    *,
    activity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return one catalogue for Overview, Tasks, and Rollouts GUI lenses."""

    config = read_project_config(config_path)
    compiled = load_compiled_catalog(config)
    declared = _declared_tasks(config)
    run, observed_rollouts = _run_evidence(activity or _empty_activity(), compiled)

    compiled_ids = {
        item["manifest"].task_id for item in compiled["tasks"]
    } if compiled["status"] == "compiled" else set()
    declared_by_id: dict[str, dict[str, Any]] = {}
    protected_holdout_count = 0
    invalid_train: list[dict[str, Any]] = []
    for item in declared:
        manifest = item.get("manifest")
        if not isinstance(manifest, TaskManifest):
            if item.get("source_partition") == "train":
                invalid_train.append(item)
            continue
        if manifest.split == "evaluation":
            protected_holdout_count += 1
            continue
        declared_by_id.setdefault(manifest.task_id, item)

    known_ids = compiled_ids | set(declared_by_id)
    rollouts_by_task: dict[str, list[dict[str, Any]]] = {}
    unmatched: list[dict[str, Any]] = []
    for rollout in observed_rollouts:
        if run["catalog_alignment"] != "matched":
            unmatched.append({**rollout, "reason": f"catalog_{run['catalog_alignment']}"})
        elif rollout["task_id"] not in known_ids:
            unmatched.append({**rollout, "reason": "task_not_in_current_catalog"})
        else:
            rollouts_by_task.setdefault(str(rollout["task_id"]), []).append(rollout)

    tasks: list[dict[str, Any]] = []
    if compiled["status"] == "compiled":
        for item in compiled["tasks"]:
            manifest = item["manifest"]
            declared_item = declared_by_id.get(manifest.task_id)
            if declared_item is None:
                declaration_state = "missing"
            elif _canonical_digest(item["manifest_payload"]) == _canonical_digest(
                declared_item["manifest_payload"]
            ):
                declaration_state = "matches_compiled"
            else:
                declaration_state = "changed_since_prepare"
            tasks.append(
                _task_record(
                    manifest,
                    manifest_path=str(item["manifest_path"]),
                    status="compiled",
                    blockers=[],
                    source_revision=str(item["source_revision"]),
                    declaration_state=declaration_state,
                    observations=rollouts_by_task.get(manifest.task_id, []),
                )
            )

    for task_id, item in declared_by_id.items():
        if task_id in compiled_ids:
            continue
        manifest = item["manifest"]
        blockers = list(item.get("blockers", []))
        tasks.append(
            _task_record(
                manifest,
                manifest_path=str(item["manifest_path"]),
                status="blocked" if blockers else "declared",
                blockers=blockers,
                source_revision=manifest.starting_revision,
                declaration_state="not_compiled",
                observations=[],
            )
        )

    for index, item in enumerate(invalid_train, start=1):
        tasks.append(
            {
                "id": f"invalid-task-{index}",
                "instruction": "This task manifest could not be read.",
                "task_family_id": None,
                "source_id": item.get("source_id"),
                "source_revision": None,
                "split": "train",
                "manifest": item.get("manifest_path"),
                "status": "blocked",
                "declaration_state": "invalid",
                "blockers": list(item.get("blockers", [])),
                "working_directory": None,
                "tools": [],
                "checks": {},
                "limits": {},
                "reward_weights": {},
                "aspects": {
                    "instruction": False,
                    "locked_snapshot": False,
                    "bounded_tools": False,
                    "hidden_verifier": False,
                    "reward_contract": False,
                },
                "observed": _observations([]),
            }
        )
    tasks.sort(key=lambda item: str(item["id"]))

    matched_rollouts = [
        rollout
        for task_rollouts in rollouts_by_task.values()
        for rollout in task_rollouts
    ]
    rewards = [float(rollout["reward"]) for rollout in matched_rollouts]
    passes = [rollout["hard_gate_passed"] is True for rollout in matched_rollouts]
    outcome_states = Counter(str(task["observed"]["outcome_mix"]) for task in tasks)
    summary = {
        "train_task_count": len(tasks),
        "compiled_task_count": sum(task["status"] == "compiled" for task in tasks),
        "declared_task_count": sum(task["status"] == "declared" for task in tasks),
        "blocked_task_count": sum(task["status"] == "blocked" for task in tasks),
        "changed_since_prepare_count": sum(
            task["declaration_state"] == "changed_since_prepare" for task in tasks
        ),
        "protected_holdout_count": protected_holdout_count,
        "task_family_count": len(
            {task["task_family_id"] for task in tasks if task.get("task_family_id")}
        ),
        "source_count": len({task["source_id"] for task in tasks if task.get("source_id")}),
        "observed_task_count": sum(task["observed"]["rollout_count"] > 0 for task in tasks),
        "rollout_count": len(matched_rollouts),
        "active_episode_count": len(run["active_episodes"]),
        "hard_gate_pass_rate": round(sum(passes) / len(passes), 4) if passes else None,
        "reward_mean": _mean(rewards),
        "rubric_means": {
            name: _mean([float(rollout["rubric"][name]) for rollout in matched_rollouts])
            for name in _RUBRIC_COMPONENTS
        },
        "outcome_states": {
            state: int(outcome_states.get(state, 0))
            for state in ("unobserved", "uncalibrated", "varied", "flat")
        },
    }

    if not tasks:
        status = "empty"
        next_action = {
            "title": "Add executable tasks",
            "detail": "Add a local task pack with an instruction, locked snapshot, bounded tools, hidden verifier, and reward contract.",
        }
    elif summary["blocked_task_count"] or compiled["status"] == "blocked":
        status = "blocked"
        detail = (
            str(compiled["blockers"][0])
            if compiled.get("blockers")
            else str(next(task for task in tasks if task["status"] == "blocked")["blockers"][0])
        )
        next_action = {"title": "Fix curriculum preparation", "detail": detail}
    elif compiled["status"] != "compiled":
        status = "declared"
        next_action = {
            "title": "Prepare the project",
            "detail": "Compile the declared tasks before starting GRPO.",
        }
    elif run["catalog_alignment"] in {"mismatch", "unknown"} and observed_rollouts:
        status = "ready"
        next_action = {
            "title": "Start a current run",
            "detail": "Stored rollout evidence is not bound to the current compiled curriculum and is shown separately.",
        }
    elif matched_rollouts:
        status = "observed"
        next_action = {
            "title": "Read the retained signal",
            "detail": "Use Tasks for per-task patterns and Rollouts for verified episode detail; calibrate difficulty against a frozen base policy before pruning tasks.",
        }
    else:
        status = "ready"
        next_action = {
            "title": "Collect real outcomes",
            "detail": "Start GRPO when Doctor is ready. This view will show only observed, verifier-scored work.",
        }

    return {
        "schema_version": 1,
        "status": status,
        "catalog": {
            key: compiled.get(key)
            for key in (
                "status",
                "fingerprint",
                "dataset_sha256",
                "dataset",
                "task_count",
                "blockers",
            )
        },
        "run": run,
        "summary": summary,
        "tasks": tasks,
        "rollouts": sorted(matched_rollouts, key=lambda item: int(item["sequence"])),
        "unmatched_observations": unmatched,
        "policy": {
            "minimum_pattern_observations": _MIN_STABLE_OBSERVATIONS,
            "scope": "current_job_retained_window",
            "principle": "Prefer diverse tasks with verifier-backed reward variation. Calibrate task difficulty against a frozen base policy before changing the curriculum.",
        },
        "next_action": next_action,
    }


__all__ = ["get_curriculum_workspace", "load_compiled_catalog"]
