"""Build an honest static experiment plan from a source scan.

The planner describes input readiness.  It never downloads a model, imports a
training library, starts a sandbox, or claims that an SFT/GRPO run succeeded.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import re
from typing import Any


def config_fingerprint(config: Mapping[str, Any]) -> str:
    """Return a stable identity for the inputs represented by a static plan."""

    payload = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _requested(section: Mapping[str, Any]) -> bool:
    return bool(section) and section.get("enabled", True) is not False


def _sources(scan: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    value = scan.get("sources", [])
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _reference_matches(
    source: Mapping[str, Any], reference: Any, project_root: Path
) -> bool:
    if isinstance(reference, Mapping):
        reference = reference.get("id", reference.get("uri", reference.get("path")))
    text = _string(reference)
    if not text:
        return False
    if text == _string(source.get("id")) or text == _string(source.get("uri")):
        return True
    if "://" in text or text.startswith("git@"):
        return False
    candidate = Path(text).expanduser()
    if not candidate.is_absolute():
        candidate = project_root / candidate
    try:
        reference_path = candidate.resolve()
    except OSError:
        return False
    resolved = _string(source.get("resolved_uri"))
    if not resolved:
        return False
    source_path = Path(resolved).expanduser()
    if not source_path.is_absolute():
        source_path = project_root / source_path
    try:
        return source_path.resolve() == reference_path
    except OSError:
        return False


def _select_source(
    sources: Sequence[Mapping[str, Any]],
    reference: Any,
    kind: str,
    project_root: Path,
) -> Mapping[str, Any] | None:
    for source in sources:
        if source.get("kind") == kind and _reference_matches(source, reference, project_root):
            return source
    return None


def _stage_status(requested: bool, blockers: Sequence[str]) -> str:
    if not requested:
        return "not_requested"
    return "blocked" if blockers else "inputs_ready"


def _task_rows(sources: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for source in sources:
        if source.get("kind") != "task_pack":
            continue
        tasks = source.get("tasks", [])
        if isinstance(tasks, Sequence) and not isinstance(tasks, (str, bytes, bytearray)):
            rows.extend(item for item in tasks if isinstance(item, Mapping))
    return rows


def _repository_holdout_blockers(
    sources: Sequence[Mapping[str, Any]],
    task_rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Prove repository isolation from scanner-derived identities.

    A source ID is only a user-facing label, so it cannot be the holdout key.
    Repository declarations count as exposure in their declared partition, and
    task references count as exposure in the task split. This also catches an
    evaluation task that accidentally points back to a train-partition source.
    """

    repositories = {
        _string(source.get("id")): source
        for source in sources
        if source.get("kind") == "repository" and _string(source.get("id"))
    }
    exposures: dict[str, set[str]] = {
        source_id: {_string(source.get("partition"))}
        for source_id, source in repositories.items()
        if _string(source.get("partition")) in {"train", "evaluation"}
    }
    missing_sources: set[str] = set()
    for task in task_rows:
        source_id = _string(task.get("snapshot_source_id"))
        split = _string(task.get("split"))
        if not source_id or split not in {"train", "evaluation"}:
            continue
        if source_id not in repositories:
            missing_sources.add(source_id)
            continue
        exposures.setdefault(source_id, set()).add(split)

    blockers: list[str] = []
    if missing_sources:
        blockers.append(
            "repository holdout cannot be verified because task source(s) have no scanned "
            "repository identity: "
            + ", ".join(sorted(missing_sources))
        )

    exposed_ids = {
        source_id
        for source_id, partitions in exposures.items()
        if partitions & {"train", "evaluation"}
    }
    missing_identities = sorted(
        source_id
        for source_id in exposed_ids
        if not _string(repositories[source_id].get("repository_identity"))
    )
    if missing_identities:
        # Fail closed: an unresolved/legacy scan must never silently downgrade
        # repository holdout back to comparing user-controlled source IDs.
        blockers.append(
            "repository holdout cannot be verified because scanned source(s) lack "
            "repository_identity: "
            + ", ".join(missing_identities)
        )

    train_ids = sorted(
        source_id for source_id, partitions in exposures.items() if "train" in partitions
    )
    evaluation_ids = sorted(
        source_id
        for source_id, partitions in exposures.items()
        if "evaluation" in partitions
    )
    collisions: set[str] = set()
    for train_id in train_ids:
        train = repositories[train_id]
        for evaluation_id in evaluation_ids:
            held_out = repositories[evaluation_id]
            train_identity = _string(train.get("repository_identity"))
            evaluation_identity = _string(held_out.get("repository_identity"))
            train_commit = _string(train.get("commit")).lower()
            evaluation_commit = _string(held_out.get("commit")).lower()
            shared_identity = bool(train_identity) and train_identity == evaluation_identity
            shared_commit = bool(train_commit) and train_commit == evaluation_commit
            if shared_identity or shared_commit:
                reason = "repository identity" if shared_identity else "exact commit"
                collisions.add(f"{train_id} and {evaluation_id} share {reason}")
    if collisions:
        blockers.append(
            "repository holdout is violated: " + "; ".join(sorted(collisions))
        )
    return blockers


def _deduplicate(values: Sequence[str]) -> list[str]:
    return sorted(dict.fromkeys(value for value in values if value))


def _project_path(project_root: Path, value: Any) -> Path | None:
    text = _string(value)
    if not text:
        return None
    candidate = Path(text).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (project_root / candidate).resolve()


def _adapter_artifact_blockers(path: Path) -> list[str]:
    if not path.exists():
        return [f"candidate adapter does not exist: {path}"]
    if path.is_file():
        if path.suffix not in {".bin", ".safetensors"}:
            return [f"candidate adapter path is not a PEFT weight file or directory: {path}"]
        return []
    missing: list[str] = []
    if not (path / "adapter_config.json").is_file():
        missing.append("adapter_config.json")
    if not any((path / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin")):
        missing.append("adapter_model.safetensors or adapter_model.bin")
    if missing:
        return [f"candidate adapter is incomplete at {path}; missing " + ", ".join(missing)]
    return []


def build_plan(
    config: Mapping[str, Any], project_root: Path, scan: Mapping[str, Any]
) -> dict[str, Any]:
    """Describe whether model, evidence, SFT, RL, and evaluation inputs exist.

    ``inputs_ready`` means only that static inputs passed inspection.  Runtime,
    GPU, dependency, model-access, verifier-execution, and benchmark checks are
    deliberately outside this function and are called out in the returned plan.
    """

    root = Path(project_root).expanduser().resolve()
    plan_errors = [str(item) for item in scan.get("errors", [])]
    plan_warnings = [str(item) for item in scan.get("warnings", [])]
    notes = [
        "Repository files are evidence/reference material, not SFT examples or RL environments by themselves.",
        "This is a static input plan: it does not download a model, import CUDA libraries, execute repository code, run verifiers, or start training.",
    ]
    declared_sources = _sources(scan)
    history = _mapping(scan.get("history"))
    history_summary = _mapping(history.get("summary"))
    scan_summary = _mapping(scan.get("summary"))
    approved_history_count = int(
        scan_summary.get(
            "approved_history_record_count", history_summary.get("approved", 0)
        )
        or 0
    )
    approved_history_source_ids = sorted(
        {
            _string(value)
            for value in history.get("approved_source_ids", [])
            if _string(value)
        }
    )

    model_config = _mapping(config.get("model"))
    model_blockers: list[str] = []
    model_warnings: list[str] = []
    model_id = _string(model_config.get("id"))
    model_revision = _string(model_config.get("revision"))
    model_provider = _string(model_config.get("provider"))
    if not model_id:
        model_blockers.append("model.id is required")
    elif model_id != "Qwen/Qwen3.5-9B":
        model_blockers.append(
            "the guarded V1 training backend currently supports only Qwen/Qwen3.5-9B"
        )
    if not model_revision:
        model_blockers.append("model.revision is required")
    elif not re.fullmatch(r"[0-9a-fA-F]{40,64}", model_revision):
        model_warnings.append(
            "model revision is mutable; resolve and lock an immutable upstream commit before training"
        )
    if model_provider and model_provider != "huggingface":
        model_blockers.append("V1 static planning supports only the huggingface model provider")
    if model_config.get("trust_remote_code") is True:
        model_blockers.append("model.trust_remote_code must remain false in V1")
    plan_errors.extend(f"model: {message}" for message in model_blockers)
    plan_warnings.extend(f"model: {message}" for message in model_warnings)
    model_plan = {
        "id": model_id or None,
        "provider": model_provider or None,
        "revision": model_revision or None,
        "status": "blocked" if model_blockers else "declared_unresolved",
        "blockers": model_blockers,
        "warnings": model_warnings,
    }

    repository_sources = [
        source for source in declared_sources if source.get("kind") == "repository"
    ]
    usable_repositories = [
        source
        for source in repository_sources
        if source.get("status") in {"ready", "warning"}
        and int(source.get("eligible_file_count", 0)) > 0
    ]
    evidence_blockers: list[str] = []
    evidence_warnings: list[str] = []
    if repository_sources and not usable_repositories:
        evidence_blockers.append("no declared repository has a usable frontend source inventory")
    if not repository_sources:
        evidence_warnings.append("no repository evidence is declared")
    for source in usable_repositories:
        if source.get("dirty"):
            evidence_warnings.append(
                f"repository {source.get('id')!r} is dirty; its commit does not fully identify the scanned evidence"
            )
    plan_errors.extend(f"evidence: {message}" for message in evidence_blockers)
    plan_warnings.extend(f"evidence: {message}" for message in evidence_warnings)
    evidence_plan = {
        "status": (
            "blocked"
            if evidence_blockers
            else "ready"
            if usable_repositories
            else "not_declared"
        ),
        "repository_count": len(repository_sources),
        "usable_repository_count": len(usable_repositories),
        "eligible_file_count": sum(
            int(source.get("eligible_file_count", 0)) for source in usable_repositories
        ),
        "source_ids": [str(source.get("id")) for source in usable_repositories],
        "blockers": evidence_blockers,
        "warnings": evidence_warnings,
        "training_ready": False,
    }

    sft_config = _mapping(config.get("sft"))
    sft_requested = _requested(sft_config)
    sft_blockers: list[str] = []
    sft_warnings: list[str] = []
    sft_reference = sft_config.get("dataset")
    sft_source: Mapping[str, Any] | None = None
    sft_sources: list[Mapping[str, Any]] = []
    included_history_count = 0
    included_history_source_ids: list[str] = []
    if sft_requested:
        if not _string(sft_reference):
            sft_blockers.append(
                "sft.dataset must point to a declared train sft_jsonl source"
            )
        else:
            sft_source = _select_source(
                declared_sources, sft_reference, "sft_jsonl", root
            )
            if sft_source is None:
                if _string(sft_reference).replace("\\", "/").endswith(
                    ".autotrainer/compiled/sft/train.jsonl"
                ):
                    # The compiler merges all explicit demonstrations with
                    # approved history only at its managed SFT destination.
                    included_history_count = approved_history_count
                    included_history_source_ids = approved_history_source_ids
                    sft_sources = [
                        source
                        for source in declared_sources
                        if source.get("kind") == "sft_jsonl"
                        and source.get("partition") == "train"
                    ]
                    if not sft_sources and included_history_count < 1:
                        sft_blockers.append(
                            "no train sft_jsonl source or approved history example is available for compile"
                        )
                else:
                    sft_blockers.append(
                        f"sft.dataset {_string(sft_reference)!r} is neither a declared source nor the compiled SFT path"
                    )
            else:
                sft_sources = [sft_source]
            for candidate in sft_sources:
                if candidate.get("partition") != "train":
                    sft_blockers.append("the SFT dataset source must use the train partition")
                if candidate.get("status") not in {"ready", "warning"}:
                    sft_blockers.append(
                        f"SFT source {candidate.get('id')!r} did not pass inspection"
                    )
            if (
                sft_sources
                and sum(int(item.get("valid_record_count", 0)) for item in sft_sources)
                + included_history_count
                < 1
            ):
                sft_blockers.append("the SFT datasets contain no valid examples")
    # A reviewed Git change is an explicit prompt/completion example; raw files
    # and pending candidates still contribute zero to training readiness.
    sft_example_count = (
        sum(int(item.get("valid_record_count", 0)) for item in sft_sources)
        + included_history_count
    )
    sft_source_ids = sorted(
        {
            *(str(source.get("id")) for source in sft_sources),
            *included_history_source_ids,
        }
    )
    plan_errors.extend(f"sft: {message}" for message in sft_blockers)
    plan_warnings.extend(f"sft: {message}" for message in sft_warnings)
    sft_plan = {
        "requested": sft_requested,
        "status": _stage_status(sft_requested, sft_blockers),
        "dataset_reference": _string(sft_reference) or None,
        "source_id": str(sft_source.get("id")) if sft_source else None,
        "source_ids": sft_source_ids,
        "approved_history_example_count": included_history_count,
        "approved_history_source_ids": included_history_source_ids,
        "valid_example_count": sft_example_count,
        "blockers": sft_blockers,
        "warnings": sft_warnings,
    }

    grpo_config = _mapping(config.get("grpo"))
    grpo_requested = _requested(grpo_config)
    grpo_blockers: list[str] = []
    grpo_warnings: list[str] = []
    grpo_reference = grpo_config.get("dataset")
    grpo_source: Mapping[str, Any] | None = None
    grpo_sources: list[Mapping[str, Any]] = []
    if grpo_requested:
        if not _string(grpo_reference):
            grpo_blockers.append(
                "grpo.dataset must point to a declared train task_pack source"
            )
        else:
            grpo_source = _select_source(
                declared_sources, grpo_reference, "task_pack", root
            )
            if grpo_source is None:
                if _string(grpo_reference).replace("\\", "/").endswith(
                    ".autotrainer/compiled/rl/train.jsonl"
                ):
                    grpo_sources = [
                        source
                        for source in declared_sources
                        if source.get("kind") == "task_pack"
                        and source.get("partition") == "train"
                    ]
                    if not grpo_sources:
                        grpo_blockers.append("no train task_pack source is declared for compile")
                else:
                    grpo_blockers.append(
                        f"grpo.dataset {_string(grpo_reference)!r} is neither a declared source nor the compiled RL path"
                    )
            else:
                grpo_sources = [grpo_source]
            for candidate in grpo_sources:
                if candidate.get("partition") != "train":
                    grpo_blockers.append("the GRPO task pack must use the train partition")
                if candidate.get("status") not in {"ready", "warning"}:
                    grpo_blockers.append(
                        f"GRPO task source {candidate.get('id')!r} did not pass inspection"
                    )
            ready_train_tasks = sum(
                bool(task.get("ready")) and task.get("split") == "train"
                for candidate in grpo_sources
                for task in candidate.get("tasks", [])
                if isinstance(task, Mapping)
            )
            if grpo_sources and ready_train_tasks < 1:
                grpo_blockers.append("the GRPO task packs have no statically ready train tasks")
        start_reference = grpo_config.get(
            "start_from", grpo_config.get("sft_adapter")
        )
        if not _string(start_reference):
            grpo_blockers.append(
                "grpo.start_from must be 'base' or identify a compatible LoRA adapter"
            )
            start_from_plan: dict[str, Any] | None = None
        elif _string(start_reference) == "base":
            start_from_plan = {"type": "base", "path": None}
            if sft_requested:
                grpo_blockers.append(
                    "both-stage training requires GRPO to continue sft.output_dir, not base"
                )
        else:
            adapter_path = _project_path(root, start_reference)
            start_from_plan = {
                "type": "adapter",
                "path": str(adapter_path) if adapter_path is not None else None,
            }
            if not sft_requested and adapter_path is not None:
                grpo_blockers.extend(_adapter_artifact_blockers(adapter_path))
        if sft_requested and _string(start_reference) != "base":
            expected_output = _string(sft_config.get("output_dir"))
            expected_path = _project_path(root, expected_output)
            selected_path = _project_path(root, start_reference)
            if expected_path is not None and selected_path is not None and expected_path != selected_path:
                grpo_blockers.append(
                    "both-stage training requires grpo.start_from to equal sft.output_dir"
                )
        environment = _mapping(config.get("environment"))
        factory = _string(environment.get("factory"))
        if not factory:
            grpo_blockers.append("environment.factory is required for agentic RL")
        elif ":" not in factory and "." not in factory:
            grpo_blockers.append(
                "environment.factory must be a dotted or module:attribute path"
            )
    else:
        start_reference = None
        start_from_plan = None
        factory = None
    grpo_ready_tasks = (
        sum(
            bool(task.get("ready")) and task.get("split") == "train"
            for candidate in grpo_sources
            for task in candidate.get("tasks", [])
            if isinstance(task, Mapping)
        )
    )
    plan_errors.extend(f"grpo: {message}" for message in grpo_blockers)
    plan_warnings.extend(f"grpo: {message}" for message in grpo_warnings)
    grpo_plan = {
        "requested": grpo_requested,
        "status": _stage_status(grpo_requested, grpo_blockers),
        "dataset_reference": _string(grpo_reference) or None,
        "source_id": str(grpo_source.get("id")) if grpo_source else None,
        "source_ids": [str(source.get("id")) for source in grpo_sources],
        "ready_task_count": grpo_ready_tasks,
        # ``sft_adapter`` remains as a compatibility projection for older
        # clients; ``start_from`` is the actual stage-neutral contract.
        "start_from": start_from_plan,
        "sft_adapter": (
            _string(start_reference)
            if _string(start_reference) and _string(start_reference) != "base"
            else None
        ),
        "environment_factory": _string(factory) or None,
        "blockers": grpo_blockers,
        "warnings": grpo_warnings,
    }

    evaluation_config = _mapping(config.get("evaluation"))
    evaluation_requested = bool(evaluation_config)
    evaluation_blockers: list[str] = []
    evaluation_warnings: list[str] = []
    evaluation_reference = evaluation_config.get("task_pack")
    evaluation_source: Mapping[str, Any] | None = None
    evaluation_candidates = evaluation_config.get("candidates")
    evaluation_arms_value = evaluation_config.get("arms")
    evaluation_arms = (
        evaluation_arms_value if isinstance(evaluation_arms_value, Mapping) else {}
    )
    evaluation_suites_value = evaluation_config.get("suites")
    evaluation_suites = (
        evaluation_suites_value if isinstance(evaluation_suites_value, Mapping) else {}
    )
    evaluation_repetitions = evaluation_config.get("repetitions")
    evaluation_seeds = evaluation_config.get("seeds")
    evaluation_dataset_reference = evaluation_config.get("dataset")
    # The final benchmark path is an explicit contract, not an artifact-dir
    # convention and never the GRPO trainer's optional validation dataset.
    compiled_evaluation_path = _project_path(root, evaluation_dataset_reference)
    arm_plans: dict[str, dict[str, Any]] = {}
    suite_plans: dict[str, dict[str, Any]] = {}
    if evaluation_requested:
        if evaluation_reference:
            evaluation_source = _select_source(
                declared_sources, evaluation_reference, "task_pack", root
            )
            if evaluation_source is None:
                evaluation_blockers.append(
                    f"evaluation.task_pack {_string(evaluation_reference)!r} is not declared"
                )
        else:
            held_out = [
                source
                for source in declared_sources
                if source.get("kind") == "task_pack"
                and source.get("partition") == "evaluation"
            ]
            if len(held_out) == 1:
                evaluation_source = held_out[0]
                evaluation_reference = held_out[0].get("id")
                evaluation_warnings.append(
                    "evaluation.task_pack was inferred from the only evaluation task source"
                )
            elif not held_out:
                evaluation_blockers.append("no held-out evaluation task_pack is declared")
            else:
                evaluation_blockers.append(
                    "multiple held-out task packs exist; set evaluation.task_pack explicitly"
                )
        if not isinstance(evaluation_candidates, list) or not evaluation_candidates:
            evaluation_blockers.append("evaluation.candidates must be a non-empty list")
        if not evaluation_arms:
            evaluation_blockers.append("evaluation.arms must declare runtime model identities")
        elif isinstance(evaluation_candidates, list) and set(evaluation_candidates) != set(
            evaluation_arms
        ):
            evaluation_blockers.append(
                "evaluation.candidates must contain exactly the declared arm ids"
            )
        if "model_benchmark" not in evaluation_suites:
            evaluation_blockers.append(
                "evaluation.suites must declare model_benchmark"
            )
        if not isinstance(evaluation_repetitions, int) or isinstance(
            evaluation_repetitions, bool
        ) or evaluation_repetitions < 1:
            evaluation_blockers.append("evaluation.repetitions must be a positive integer")
        if not isinstance(evaluation_seeds, list) or not evaluation_seeds:
            evaluation_blockers.append("evaluation.seeds must be a non-empty list")
        elif (
            isinstance(evaluation_repetitions, int)
            and not isinstance(evaluation_repetitions, bool)
            and len(evaluation_seeds) != evaluation_repetitions
        ):
            evaluation_blockers.append(
                "evaluation.seeds length must equal evaluation.repetitions"
            )

        for arm_id, arm_value in evaluation_arms.items():
            arm = _mapping(arm_value)
            role = _string(arm.get("role"))
            model = arm.get("model")
            adapter = _mapping(arm.get("adapter"))
            arm_plan: dict[str, Any] = {
                "label": _string(arm.get("label")) or str(arm_id),
                "role": role or None,
                "parameter_class": _string(arm.get("parameter_class")) or None,
                "model": dict(model) if isinstance(model, Mapping) else model,
                "adapter": None,
            }
            if isinstance(model, Mapping):
                external_id = _string(model.get("id"))
                external_revision = _string(model.get("revision"))
                if not external_id or external_id.upper().startswith("REPLACE_"):
                    evaluation_blockers.append(
                        f"arm {arm_id!r} has an unresolved external model id"
                    )
                if not re.fullmatch(r"[0-9a-fA-F]{40,64}", external_revision) or not set(
                    external_revision
                ) - {"0"}:
                    evaluation_blockers.append(
                        f"arm {arm_id!r} external model revision must be a non-placeholder immutable commit SHA"
                    )
            if role == "candidate":
                adapter_reference = adapter.get("path")
                adapter_path = _project_path(root, adapter_reference)
                arm_plan["adapter"] = {
                    "path": str(adapter_path) if adapter_path is not None else None,
                    "stage": _string(adapter.get("stage")) or None,
                    "sha256": _string(adapter.get("sha256")) or None,
                }
                if adapter_path is None:
                    evaluation_blockers.append(
                        f"candidate arm {arm_id!r} must declare adapter.path"
                    )
                else:
                    evaluation_blockers.extend(_adapter_artifact_blockers(adapter_path))
            arm_plans[str(arm_id)] = arm_plan

        if compiled_evaluation_path is None:
            evaluation_blockers.append(
                "evaluation.dataset is required to locate compiled evaluation tasks"
            )
        elif not compiled_evaluation_path.is_file():
            evaluation_blockers.append(
                f"compiled evaluation task dataset is missing: {compiled_evaluation_path}"
            )
        else:
            try:
                compiled_size = compiled_evaluation_path.stat().st_size
            except OSError:
                compiled_size = 0
            if compiled_size < 1:
                evaluation_blockers.append(
                    f"compiled evaluation task dataset is empty: {compiled_evaluation_path}"
                )
        if evaluation_source is not None:
            if evaluation_source.get("partition") != "evaluation":
                evaluation_blockers.append(
                    "evaluation.task_pack must use the evaluation partition"
                )
            if evaluation_source.get("status") not in {"ready", "warning"}:
                evaluation_blockers.append("the held-out task pack did not pass inspection")
            ready_evaluation_tasks = sum(
                bool(task.get("ready")) and task.get("split") == "evaluation"
                for task in evaluation_source.get("tasks", [])
                if isinstance(task, Mapping)
            )
            if ready_evaluation_tasks < 1:
                evaluation_blockers.append(
                    "the held-out task pack has no statically ready evaluation tasks"
                )
    evaluation_ready_tasks = (
        sum(
            bool(task.get("ready")) and task.get("split") == "evaluation"
            for task in evaluation_source.get("tasks", [])
            if isinstance(task, Mapping)
        )
        if evaluation_source
        else 0
    )

    task_rows = _task_rows(declared_sources)
    train_rows = [task for task in task_rows if task.get("split") == "train"]
    evaluation_rows = [task for task in task_rows if task.get("split") == "evaluation"]
    train_ids = {_string(task.get("task_id")) for task in train_rows if task.get("task_id")}
    evaluation_ids = {
        _string(task.get("task_id")) for task in evaluation_rows if task.get("task_id")
    }
    duplicate_task_ids = sorted(train_ids & evaluation_ids)
    if duplicate_task_ids:
        evaluation_blockers.append(
            "task ids appear in both train and evaluation: " + ", ".join(duplicate_task_ids)
        )
    holdout_unit = _string(evaluation_config.get("holdout_unit"))
    if holdout_unit == "repository":
        evaluation_blockers.extend(
            _repository_holdout_blockers(declared_sources, task_rows)
        )

    repetitions_for_plan = (
        evaluation_repetitions
        if isinstance(evaluation_repetitions, int)
        and not isinstance(evaluation_repetitions, bool)
        and evaluation_repetitions > 0
        else 0
    )
    for suite_id, suite_value in evaluation_suites.items():
        suite = _mapping(suite_value)
        members_value = suite.get("arms")
        members = (
            list(members_value)
            if isinstance(members_value, Sequence)
            and not isinstance(members_value, (str, bytes, bytearray))
            else []
        )
        runner = _mapping(suite.get("runner"))
        runner_type = _string(runner.get("type"))
        runner_blockers: list[str] = []
        if evaluation_requested:
            unknown_members = sorted(
                _string(member) for member in members if _string(member) not in evaluation_arms
            )
            if unknown_members:
                evaluation_blockers.append(
                    f"suite {suite_id!r} references unknown arms: "
                    + ", ".join(unknown_members)
                )
            if runner_type == "builtin":
                # The built-in runner identity is code-owned and frozen later;
                # there are no executable paths or editable prompt hashes for
                # the project author to fill in.
                from .local_evaluation_runner import builtin_runner_identity

                runner = {"type": "builtin", **builtin_runner_identity()}
            else:
                runner_producer = _string(runner.get("producer"))
                runner_version = _string(runner.get("version"))
                runner_fingerprint = _string(runner.get("orchestration_sha256"))
                if not runner_producer:
                    runner_blockers.append(
                        f"suite {suite_id!r} runner requires producer"
                    )
                if not runner_version or runner_version.upper().startswith("REPLACE_"):
                    runner_blockers.append(
                        f"suite {suite_id!r} runner requires a concrete version"
                    )
                fingerprint_match = re.fullmatch(
                    r"sha256:([0-9a-fA-F]{64})", runner_fingerprint
                )
                if fingerprint_match is None or not set(fingerprint_match.group(1)) - {"0"}:
                    runner_blockers.append(
                        f"suite {suite_id!r} runner requires a non-placeholder immutable orchestration_sha256"
                    )
            if runner_type == "command":
                argv = runner.get("argv")
                if not isinstance(argv, list) or not argv:
                    runner_blockers.append(
                        f"suite {suite_id!r} command runner has no argv"
                    )
                else:
                    if any("REPLACE_WITH" in _string(part).upper() for part in argv):
                        runner_blockers.append(
                            f"suite {suite_id!r} command runner argv is still a placeholder"
                        )
                    if not any("{request}" in _string(part) for part in argv):
                        runner_blockers.append(
                            f"suite {suite_id!r} command runner argv requires {{request}}"
                        )
                    if not any("{result}" in _string(part) for part in argv):
                        runner_blockers.append(
                            f"suite {suite_id!r} command runner argv requires {{result}}"
                        )
            elif runner_type == "external":
                if not _string(runner.get("result_schema")):
                    runner_blockers.append(
                        f"suite {suite_id!r} external runner requires result_schema"
                    )
            elif runner_type != "builtin":
                runner_blockers.append(
                    f"suite {suite_id!r} runner must be builtin, command, or external"
                )
            if runner_type == "external" and runner_blockers:
                evaluation_warnings.append(
                    f"suite {suite_id!r} is deferred until its external runner is pinned"
                )
            else:
                evaluation_blockers.extend(runner_blockers)
        pair_count = evaluation_ready_tasks * repetitions_for_plan
        suite_plans[str(suite_id)] = {
            "kind": _string(suite.get("kind")) or None,
            "arms": members,
            "runner": dict(runner),
            "runner_status": (
                "ready_local"
                if runner_type == "builtin"
                else "declared"
                if runner_type == "command"
                else "deferred_configuration"
                if runner_type == "external" and runner_blockers
                else "awaiting_external_results"
                if runner_type == "external"
                else "invalid"
            ),
            "blockers": runner_blockers,
            "pair_count": pair_count,
            "arm_run_count": pair_count * len(members),
        }

    evaluation_blockers = _deduplicate(evaluation_blockers)
    evaluation_warnings = _deduplicate(evaluation_warnings)
    plan_errors.extend(f"evaluation: {message}" for message in evaluation_blockers)
    plan_warnings.extend(f"evaluation: {message}" for message in evaluation_warnings)
    evaluation_plan = {
        "requested": evaluation_requested,
        "status": _stage_status(evaluation_requested, evaluation_blockers),
        "task_pack_reference": _string(evaluation_reference) or None,
        "source_id": str(evaluation_source.get("id")) if evaluation_source else None,
        "ready_task_count": evaluation_ready_tasks,
        "candidates": list(evaluation_candidates) if isinstance(evaluation_candidates, list) else [],
        "repetitions": repetitions_for_plan,
        "seeds": list(evaluation_seeds) if isinstance(evaluation_seeds, list) else [],
        "compiled_dataset": str(compiled_evaluation_path) if compiled_evaluation_path else None,
        "arms": arm_plans,
        "suites": suite_plans,
        "fairness": dict(_mapping(evaluation_config.get("fairness"))),
        "decisions": dict(_mapping(evaluation_config.get("decisions"))),
        "holdout_unit": holdout_unit or None,
        "blockers": evaluation_blockers,
        "warnings": evaluation_warnings,
    }

    plan_errors = _deduplicate(plan_errors)
    plan_warnings = _deduplicate(plan_warnings)
    project = _mapping(config.get("project"))
    return {
        "schema_version": 1,
        "config_fingerprint": config_fingerprint(config),
        "project_root": str(root),
        "project": {
            "name": _string(project.get("name")) or None,
            "artifact_dir": _string(
                project.get("artifact_dir", project.get("output_dir", ".autotrainer"))
            ),
        },
        "status": "blocked" if plan_errors else "inputs_ready",
        "static_only": True,
        "runtime_checked": False,
        "model": model_plan,
        "evidence": evidence_plan,
        "stages": {
            "sft": sft_plan,
            "grpo": grpo_plan,
            "evaluation": evaluation_plan,
        },
        "source_summary": dict(_mapping(scan.get("summary"))),
        "errors": plan_errors,
        "blockers": plan_errors,
        "warnings": plan_warnings,
        "notes": notes,
    }


__all__ = ["build_plan", "config_fingerprint"]
