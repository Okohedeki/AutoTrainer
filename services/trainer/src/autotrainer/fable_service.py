"""Managed setup and exchange workflow for the external Fable comparison."""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import re
from threading import Lock, Thread, current_thread
from typing import Any, Mapping
from uuid import uuid4

from .config import (
    ConfigError,
    load_config,
    project_config_mutation,
    validate_mapping,
    write_config,
)
from .evaluation import (
    EvaluationError,
    build_evaluation_reports,
    export_blind_review,
    export_evaluation_suite,
    import_blind_reviews,
    ingest_evaluation_results,
    write_evaluation_plan,
)
from .integrity import IntegrityError, sha256_file, tree_identity
from .project_gate import (
    ProjectLease,
    acquire_project_lease,
    project_mutation_gate,
)


_PLACEHOLDER_VERSION = "REPLACE_WITH_FABLE_VERSION"
_PLACEHOLDER_DIGEST = "sha256:" + "0" * 64
_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+:-]{0,99}$")
_ACTIONS = frozenset(
    {"export", "ingest", "review_export", "review_import", "report"}
)
_INPUT_ACTIONS = frozenset({"ingest", "review_import"})
_LIVE = frozenset({"queued", "running"})


class FableServiceError(ConfigError):
    """Raised for invalid pins or unavailable Fable workflow actions."""


def _runner(config: Any) -> Mapping[str, Any] | None:
    evaluation = config.data.get("evaluation", {})
    suites = evaluation.get("suites", {}) if isinstance(evaluation, Mapping) else {}
    suite = suites.get("fable_ab") if isinstance(suites, Mapping) else None
    runner = suite.get("runner") if isinstance(suite, Mapping) else None
    return runner if isinstance(runner, Mapping) else None


def _is_pinned(runner: Mapping[str, Any]) -> bool:
    version = str(runner.get("version", "")).strip()
    digest = str(runner.get("orchestration_sha256", "")).strip().casefold()
    return bool(
        version
        and version != _PLACEHOLDER_VERSION
        and re.fullmatch(r"sha256:[0-9a-f]{64}", digest)
        and digest != _PLACEHOLDER_DIGEST
    )


def _receipt_path(config: Any) -> Path:
    return config.artifact_dir / "evaluation" / "fable-runner-pin.json"


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return dict(value) if isinstance(value, Mapping) else None


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(dict(value), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _runtime_identity(path: Path) -> dict[str, Any]:
    candidate = Path(os.path.abspath(Path(path).expanduser()))
    try:
        if candidate.is_file():
            return {
                "kind": "file",
                "path": str(candidate),
                "sha256": f"sha256:{sha256_file(candidate)}",
                "file_count": 1,
                "byte_count": candidate.stat().st_size,
            }
        if candidate.is_dir():
            identity = tree_identity(candidate)
            files = identity["files"]
            return {
                "kind": "directory",
                "path": str(candidate),
                "sha256": identity["sha256"],
                "file_count": len(files),
                "byte_count": sum(int(item["bytes"]) for item in files),
            }
    except (OSError, IntegrityError) as error:
        raise FableServiceError(f"Fable runtime cannot be pinned: {error}") from error
    raise FableServiceError("Fable runtime path must be an existing file or directory")


def _updated_config(config: Any, version: str, digest: str) -> dict[str, Any]:
    updated = deepcopy(config.data)
    evaluation = updated["evaluation"]
    suites = evaluation["suites"]
    if "fable_ab" not in suites:
        arms = evaluation["arms"]
        if len(arms) >= 3 or any(
            isinstance(value, Mapping) and value.get("role") == "control"
            for value in arms.values()
        ):
            raise FableServiceError(
                "Fable cannot be enabled while another optional evaluation control is configured"
            )
        arms["base_fable"] = {
            "label": "Base 9B + Fable",
            "role": "control",
            "parameter_class": "9b",
            "model": "project",
        }
        candidates = evaluation["candidates"]
        candidates.insert(max(len(candidates) - 1, 0), "base_fable")
        suites["fable_ab"] = {
            "kind": "fable_ab",
            "arms": ["base_fable", "autotrainer"],
            "runner": {
                "type": "external",
                "producer": "fable",
                "version": _PLACEHOLDER_VERSION,
                "orchestration_sha256": _PLACEHOLDER_DIGEST,
                "result_schema": "autotrainer-evaluation-result-v1",
            },
            "review": {
                "type": "manual",
                "blind": True,
                "reviewers_per_pair": 3,
            },
        }
        evaluation["decisions"]["fable_ab"] = {
            "candidate": "autotrainer",
            "control": "base_fable",
            "metric": "blind_preference_rate",
            "minimum_rate": 0.5,
            "minimum_tasks": 5,
        }
    runner = suites["fable_ab"]["runner"]
    runner["producer"] = "fable"
    runner["version"] = version
    runner["orchestration_sha256"] = digest
    runner["result_schema"] = "autotrainer-evaluation-result-v1"
    report = validate_mapping(updated, root=config.root)
    if report.errors:
        raise FableServiceError("\n".join(report.errors))
    return updated


def pin_fable_runner(
    config_path: str | Path,
    *,
    version: str,
    runtime_path: str | Path,
) -> dict[str, Any]:
    """Hash a supplied Fable bundle and bind that identity into project YAML."""

    selected_version = str(version).strip()
    if _VERSION.fullmatch(selected_version) is None or selected_version.startswith(
        "REPLACE_WITH_"
    ):
        raise FableServiceError(
            "version must be a concrete Fable release or immutable revision"
        )
    identity = _runtime_identity(Path(runtime_path))
    with project_mutation_gate(config_path):
        with project_config_mutation(config_path):
            config = load_config(config_path)
            receipt_path = _receipt_path(config)
            previous = receipt_path.read_bytes() if receipt_path.is_file() else None
            plan_pointer = config.artifact_dir / "evaluation" / "current-plan.json"
            previous_pointer = (
                plan_pointer.read_bytes() if plan_pointer.is_file() else None
            )
            receipt = {
                "schema_version": 1,
                "producer": "fable",
                "version": selected_version,
                "orchestration_sha256": identity["sha256"],
                "runtime": identity,
            }
            _write_json(receipt_path, receipt)
            try:
                # A frozen plan includes the external runner identity. Preserve
                # every prior run directory, but retire the small mutable pointer
                # before changing that identity so the GUI never loads a stale plan.
                plan_pointer.unlink(missing_ok=True)
                write_config(
                    config.path,
                    _updated_config(
                        config,
                        selected_version,
                        str(identity["sha256"]),
                    ),
                    overwrite=True,
                )
            except Exception:
                if previous is None:
                    receipt_path.unlink(missing_ok=True)
                else:
                    receipt_path.write_bytes(previous)
                if previous_pointer is not None:
                    plan_pointer.parent.mkdir(parents=True, exist_ok=True)
                    plan_pointer.write_bytes(previous_pointer)
                raise
    return inspect_fable_workflow(config_path)


def _verified_receipt(config: Any) -> dict[str, Any]:
    runner = _runner(config)
    if runner is None or not _is_pinned(runner):
        raise FableServiceError("Pin a concrete Fable runtime before exporting requests")
    receipt = _read_json(_receipt_path(config))
    if receipt is None:
        raise FableServiceError("The Fable pin receipt is missing; pin the runtime again")
    expected = str(runner["orchestration_sha256"]).casefold()
    if (
        receipt.get("producer") != "fable"
        or receipt.get("version") != runner.get("version")
        or str(receipt.get("orchestration_sha256", "")).casefold() != expected
    ):
        raise FableServiceError("The Fable pin receipt no longer matches project YAML")
    runtime = receipt.get("runtime")
    if not isinstance(runtime, Mapping) or not isinstance(runtime.get("path"), str):
        raise FableServiceError("The Fable pin receipt has no runtime path")
    current = _runtime_identity(Path(runtime["path"]))
    if str(current["sha256"]).casefold() != expected:
        raise FableServiceError(
            "The pinned Fable runtime bytes changed; pin the new identity before export"
        )
    return receipt


def _pointer_state(config: Any) -> dict[str, Any] | None:
    pointer = _read_json(config.artifact_dir / "evaluation" / "current-plan.json")
    if (
        pointer is None
        or not isinstance(pointer.get("path"), str)
        or not isinstance(pointer.get("plan_id"), str)
    ):
        return None
    digest = str(pointer["plan_id"]).removeprefix("sha256:")
    if re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None:
        return None
    plan_path = Path(pointer["path"]).expanduser().resolve()
    expected_path = (
        config.artifact_dir / "evaluation" / digest / "evaluation-plan.json"
    ).resolve()
    if plan_path != expected_path:
        return None
    plan = _read_json(plan_path)
    if plan is None or plan.get("plan_id") != pointer.get("plan_id"):
        return None
    runner = plan.get("suites", {}).get("fable_ab", {}).get("runner", {})
    configured = _runner(config)
    if (
        not isinstance(runner, Mapping)
        or configured is None
        or runner.get("version") != configured.get("version")
        or str(runner.get("orchestration_sha256", "")).casefold()
        != str(configured.get("orchestration_sha256", "")).casefold()
    ):
        return None
    run_dir = plan_path.parent
    trials = [
        trial
        for trial in plan.get("trials", [])
        if isinstance(trial, Mapping) and trial.get("suite_id") == "fable_ab"
    ]
    trial_ids = {str(trial.get("trial_id", "")) for trial in trials}
    scored_count = sum(
        (run_dir / "scored-trials" / f"{trial_id}.json").is_file()
        for trial_id in trial_ids
        if trial_id
    )
    exchange_root = config.artifact_dir / "evaluation" / "fable-exchange" / str(
        plan["plan_id"]
    )
    request_manifest = _read_json(exchange_root / "requests" / "export-manifest.json")
    blind_pairs = exchange_root / "blind-review" / "blind-pairs.jsonl"
    reviews = run_dir / "reviews" / "fable_ab" / "reviews.jsonl"
    report = run_dir / "reports" / "fable_ab.json"
    return {
        "plan_id": str(plan["plan_id"]),
        "trial_count": len(trials),
        "scored_count": scored_count,
        "requests_exported": request_manifest is not None,
        "request_path": (
            str(exchange_root / "requests") if request_manifest is not None else None
        ),
        "blind_pairs_exported": blind_pairs.is_file(),
        "blind_review_path": str(blind_pairs) if blind_pairs.is_file() else None,
        "reviews_imported": reviews.is_file(),
        "report_ready": report.is_file(),
        "report_path": str(report) if report.is_file() else None,
    }


def _workflow_actions(pinned: bool, plan: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    trial_count = int(plan.get("trial_count", 0)) if plan else 0
    scored_count = int(plan.get("scored_count", 0)) if plan else 0
    return [
        {
            "id": "export",
            "title": "Freeze and export Fable requests",
            "detail": "Write verifier-free requests into the managed project exchange folder.",
            "status": (
                "complete"
                if plan and plan.get("requests_exported")
                else "available"
                if pinned
                else "blocked"
            ),
            "input_required": False,
        },
        {
            "id": "ingest",
            "title": "Ingest returned Fable results",
            "detail": "Validate producer identity and re-score patches with the trusted local verifier.",
            "status": (
                "complete"
                if trial_count > 0 and scored_count == trial_count
                else "available"
                if plan
                and plan.get("requests_exported")
                and scored_count < trial_count
                else "blocked"
            ),
            "input_required": True,
        },
        {
            "id": "review_export",
            "title": "Export blind review pairs",
            "detail": "Create counterbalanced left/right artifacts with model identities sealed away.",
            "status": (
                "complete"
                if plan and plan.get("blind_pairs_exported")
                else "available"
                if trial_count > 0
                and scored_count == trial_count
                else "blocked"
            ),
            "input_required": False,
        },
        {
            "id": "review_import",
            "title": "Import blind reviewer choices",
            "detail": "Validate immutable reviewer rows against the sealed pair map.",
            "status": (
                "complete"
                if plan and plan.get("reviews_imported")
                else "available"
                if plan
                and plan.get("blind_pairs_exported")
                else "blocked"
            ),
            "input_required": True,
        },
        {
            "id": "report",
            "title": "Build separate Fable report",
            "detail": "Publish the Fable decision without pooling it with the model benchmark.",
            "status": (
                "complete"
                if plan and plan.get("report_ready")
                else "available"
                if plan and plan.get("reviews_imported")
                else "blocked"
            ),
            "input_required": False,
        },
    ]


def inspect_fable_workflow(config_path: str | Path) -> dict[str, Any]:
    """Inspect the pin and managed exchange without executing external code."""

    config = load_config(config_path)
    runner = _runner(config)
    if runner is None:
        actions = _workflow_actions(False, None)
        return {
            "status": "optional",
            "runner": {
                "producer": "fable",
                "version": None,
                "orchestration_sha256": None,
                "pinned": False,
                "receipt_matches": False,
                "runtime_path": None,
            },
            "exchange": None,
            "actions": actions,
            "next_action": None,
        }
    pinned = _is_pinned(runner)
    receipt = _read_json(_receipt_path(config))
    receipt_matches = bool(
        pinned
        and receipt
        and receipt.get("version") == runner.get("version")
        and str(receipt.get("orchestration_sha256", "")).casefold()
        == str(runner.get("orchestration_sha256", "")).casefold()
    )
    plan = _pointer_state(config)
    actions = _workflow_actions(pinned and receipt_matches, plan)
    next_action = next(
        (action for action in actions if action["status"] == "available"),
        None,
    )
    return {
        "status": (
            "report_ready"
            if plan and plan.get("report_ready")
            else "in_progress"
            if pinned and receipt_matches
            else "needs_pin"
        ),
        "runner": {
            "producer": "fable",
            "version": runner.get("version"),
            "orchestration_sha256": runner.get("orchestration_sha256"),
            "pinned": pinned,
            "receipt_matches": receipt_matches,
            "runtime_path": (
                receipt.get("runtime", {}).get("path")
                if isinstance(receipt, Mapping)
                and isinstance(receipt.get("runtime"), Mapping)
                else None
            ),
        },
        "exchange": plan,
        "actions": actions,
        "next_action": next_action,
    }


def _run_owned(
    config_path: str | Path,
    action_id: str,
    input_path: str | Path | None,
) -> dict[str, Any]:
    if action_id not in _ACTIONS:
        raise FableServiceError("Fable workflow action is invalid")
    if action_id in _INPUT_ACTIONS and not str(input_path or "").strip():
        raise FableServiceError("input_path is required for this Fable action")
    if action_id not in _INPUT_ACTIONS and input_path is not None:
        raise FableServiceError("input_path is not accepted for this Fable action")
    config = load_config(config_path, check_paths=True)
    if action_id == "export":
        _verified_receipt(config)
        plan = write_evaluation_plan(config.data, config.root)
        destination = (
            config.artifact_dir
            / "evaluation"
            / "fable-exchange"
            / str(plan["plan_id"])
            / "requests"
        )
        return export_evaluation_suite(
            config.data,
            config.root,
            "fable_ab",
            destination,
        )
    if action_id == "ingest":
        return ingest_evaluation_results(
            config.data,
            config.root,
            "fable_ab",
            Path(str(input_path)),
        )
    if action_id == "review_export":
        plan = _pointer_state(config)
        if plan is None:
            raise FableServiceError("Export Fable requests before blind review")
        destination = (
            config.artifact_dir
            / "evaluation"
            / "fable-exchange"
            / str(plan["plan_id"])
            / "blind-review"
        )
        return export_blind_review(
            config.data,
            config.root,
            "fable_ab",
            destination,
        )
    if action_id == "review_import":
        return import_blind_reviews(
            config.data,
            config.root,
            "fable_ab",
            Path(str(input_path)),
        )
    return build_evaluation_reports(config.data, config.root, write_artifacts=True)


def run_fable_action(
    config_path: str | Path,
    action_id: str,
    *,
    input_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run one managed exchange action synchronously for agent automation."""

    lease = acquire_project_lease(config_path)
    try:
        with lease.activate("run"):
            return _run_owned(config_path, action_id, input_path)
    finally:
        lease.release()


def _idle() -> dict[str, Any]:
    return {
        "id": None,
        "action_id": None,
        "status": "idle",
        "message": "No Fable workflow action is active.",
        "result": None,
    }


def _safe_job_result(value: object) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    result: dict[str, Any] = {}
    for field in ("plan_id", "suite_id", "artifact"):
        item = value.get(field)
        if isinstance(item, str) and len(item) <= 4_096:
            result[field] = item
    for field in ("request_count", "ingested_count", "pair_count", "review_count"):
        item = value.get(field)
        if isinstance(item, int) and not isinstance(item, bool) and item >= 0:
            result[field] = item
    return result or None


class FableWorkflowManager:
    """Run long external-result verification without holding an HTTP request."""

    def __init__(self, config_path: str | Path) -> None:
        self._config_path = Path(config_path).expanduser().resolve()
        self._lock = Lock()
        self._job = _idle()
        self._worker: Thread | None = None

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return deepcopy(self._job)

    def workspace(self) -> dict[str, Any]:
        return {**inspect_fable_workflow(self._config_path), "job": self.snapshot()}

    def start(
        self,
        action_id: str,
        *,
        input_path: str | Path | None = None,
    ) -> dict[str, Any]:
        if action_id not in _ACTIONS:
            raise FableServiceError("Fable workflow action is invalid")
        if action_id in _INPUT_ACTIONS and not str(input_path or "").strip():
            raise FableServiceError("input_path is required for this Fable action")
        if action_id not in _INPUT_ACTIONS and input_path is not None:
            raise FableServiceError("input_path is not accepted for this Fable action")
        workspace = inspect_fable_workflow(self._config_path)
        selected = next(
            (action for action in workspace["actions"] if action["id"] == action_id),
            None,
        )
        if selected is None or selected["status"] != "available":
            raise FableServiceError("Complete the prior Fable workflow step first")
        with self._lock:
            if self._job["status"] in _LIVE:
                raise FableServiceError("A Fable workflow action is already active.")
            lease = acquire_project_lease(self._config_path)
            job_id = uuid4().hex
            self._job = {
                "id": job_id,
                "action_id": action_id,
                "status": "queued",
                "message": f"{selected['title']} is queued.",
                "result": None,
            }
            try:
                worker = Thread(
                    target=self._run,
                    args=(job_id, action_id, input_path, lease),
                    name=f"autotrainer-fable-{job_id[:8]}",
                    daemon=False,
                )
                self._worker = worker
                worker.start()
            except Exception:
                self._job = _idle()
                self._worker = None
                lease.release()
                raise
            return deepcopy(self._job)

    def _update(self, job_id: str, **values: Any) -> None:
        with self._lock:
            if self._job.get("id") == job_id:
                self._job.update(values)

    def _run(
        self,
        job_id: str,
        action_id: str,
        input_path: str | Path | None,
        lease: ProjectLease,
    ) -> None:
        try:
            with lease.activate("run"):
                self._update(
                    job_id,
                    status="running",
                    message="AutoTrainer is processing the managed Fable exchange.",
                )
                try:
                    result = _run_owned(
                        self._config_path,
                        action_id,
                        input_path,
                    )
                except (ConfigError, EvaluationError, FableServiceError) as error:
                    self._update(
                        job_id,
                        status="failed",
                        message=str(error).replace("\r", " ").replace("\n", " ")[:1_000],
                        result=None,
                    )
                except Exception:
                    self._update(
                        job_id,
                        status="failed",
                        message="The Fable workflow stopped after an unexpected local failure.",
                        result=None,
                    )
                else:
                    self._update(
                        job_id,
                        status="completed",
                        message="The managed Fable workflow action completed.",
                        result=_safe_job_result(result),
                    )
        finally:
            lease.release()

    def close(self) -> None:
        with self._lock:
            worker = self._worker
        if worker is not None and worker is not current_thread() and worker.is_alive():
            worker.join()


__all__ = [
    "FableServiceError",
    "FableWorkflowManager",
    "inspect_fable_workflow",
    "pin_fable_runner",
    "run_fable_action",
]
