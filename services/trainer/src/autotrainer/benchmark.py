"""Deterministic, offline comparison of normalized website-task run results.

The benchmark core never imports a model runtime and never executes a task. It
accepts already-normalized verifier results, checks that every candidate was
evaluated on the same task/repetition/seed matrix, and produces descriptive
statistics suitable for both automation and a concise human report.

The normalized input is one JSON object with ``schema_version``,
``benchmark_id``, optional ``metadata``, named ``candidates``, and ``runs``.
Each run declares ``candidate_id``, ``task_id``, ``repetition``, ``seed``,
``hard_gate_passed``, ``gate_reason``, ``reward``, and normalized
``components`` whose values are finite rates in the inclusive range [0, 1].
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


INPUT_SCHEMA_VERSION = "1.0"
REPORT_SCHEMA_VERSION = "1.0"


class BenchmarkError(ValueError):
    """Raised when benchmark inputs are invalid or not directly comparable."""


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BenchmarkError(f"{field} must be an object")
    return value


def _sequence(value: Any, field: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise BenchmarkError(f"{field} must be an array")
    return value


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BenchmarkError(f"{field} must be a non-empty string")
    return value.strip()


def _nonnegative_integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BenchmarkError(f"{field} must be a non-negative integer")
    return value


def _rate(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BenchmarkError(f"{field} must be a number between 0 and 1")
    result = float(value)
    if not math.isfinite(result) or result < 0 or result > 1:
        raise BenchmarkError(f"{field} must be a finite number between 0 and 1")
    return result


def _json_value(value: Any, field: str) -> Any:
    """Validate and copy metadata into a deterministic JSON-compatible value."""

    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise BenchmarkError(f"{field} must not contain NaN or infinity")
        return value
    if isinstance(value, list):
        return [_json_value(item, f"{field}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise BenchmarkError(f"{field} object keys must be strings")
        normalized: dict[str, Any] = {}
        for key in sorted(value):
            normalized[key] = _json_value(value[key], f"{field}.{key}")
        return normalized
    raise BenchmarkError(f"{field} contains a non-JSON value of type {type(value).__name__}")


def _reject_unknown_fields(value: Mapping[str, Any], allowed: set[str], field: str) -> None:
    unknown = sorted(str(key) for key in value if key not in allowed)
    if unknown:
        raise BenchmarkError(f"{field} contains unknown fields: {', '.join(unknown)}")


def load_benchmark_input(path: str | Path) -> dict[str, Any]:
    """Load a normalized benchmark input document from a local JSON file."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise BenchmarkError(f"benchmark input does not exist or is not a file: {source}")
    if source.suffix.lower() != ".json":
        raise BenchmarkError(f"benchmark input must be a .json file: {source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except OSError as error:
        raise BenchmarkError(f"could not read benchmark input {source}: {error}") from error
    except json.JSONDecodeError as error:
        raise BenchmarkError(
            f"invalid benchmark JSON at {source}:{error.lineno}:{error.colno}: {error.msg}"
        ) from error
    if not isinstance(payload, dict):
        raise BenchmarkError("benchmark input must contain one JSON object")
    return payload


def _normalize_input(payload: Mapping[str, Any]) -> dict[str, Any]:
    root = _mapping(payload, "benchmark")
    _reject_unknown_fields(
        root,
        {"schema_version", "benchmark_id", "metadata", "candidates", "runs"},
        "benchmark",
    )
    schema_version = root.get("schema_version")
    if schema_version != INPUT_SCHEMA_VERSION:
        raise BenchmarkError(
            f"schema_version must be {INPUT_SCHEMA_VERSION!r}; got {schema_version!r}"
        )
    benchmark_id = _text(root.get("benchmark_id"), "benchmark_id")
    metadata = _json_value(_mapping(root.get("metadata", {}), "metadata"), "metadata")

    raw_candidates = _sequence(root.get("candidates"), "candidates")
    if len(raw_candidates) < 2:
        raise BenchmarkError("candidates must contain at least two named candidates")
    candidates: list[dict[str, Any]] = []
    candidate_ids: set[str] = set()
    candidate_labels: set[str] = set()
    for index, raw_candidate in enumerate(raw_candidates):
        candidate = _mapping(raw_candidate, f"candidates[{index}]")
        _reject_unknown_fields(
            candidate, {"id", "label", "metadata"}, f"candidates[{index}]"
        )
        candidate_id = _text(candidate.get("id"), f"candidates[{index}].id")
        label = _text(candidate.get("label"), f"candidates[{index}].label")
        if candidate_id in candidate_ids:
            raise BenchmarkError(f"candidate id is duplicated: {candidate_id}")
        if label in candidate_labels:
            raise BenchmarkError(f"candidate label is duplicated: {label}")
        candidate_ids.add(candidate_id)
        candidate_labels.add(label)
        candidates.append(
            {
                "id": candidate_id,
                "label": label,
                "metadata": _json_value(
                    _mapping(candidate.get("metadata", {}), f"candidates[{index}].metadata"),
                    f"candidates[{index}].metadata",
                ),
            }
        )

    raw_runs = _sequence(root.get("runs"), "runs")
    if not raw_runs:
        raise BenchmarkError("runs must contain at least one normalized run result")
    runs: list[dict[str, Any]] = []
    run_keys: set[tuple[str, str, int]] = set()
    component_names: set[str] | None = None
    for index, raw_run in enumerate(raw_runs):
        field = f"runs[{index}]"
        run = _mapping(raw_run, field)
        _reject_unknown_fields(
            run,
            {
                "candidate_id",
                "task_id",
                "repetition",
                "seed",
                "hard_gate_passed",
                "gate_reason",
                "reward",
                "components",
                "metadata",
            },
            field,
        )
        candidate_id = _text(run.get("candidate_id"), f"{field}.candidate_id")
        if candidate_id not in candidate_ids:
            raise BenchmarkError(
                f"{field}.candidate_id refers to undeclared candidate {candidate_id!r}"
            )
        task_id = _text(run.get("task_id"), f"{field}.task_id")
        repetition = _nonnegative_integer(run.get("repetition"), f"{field}.repetition")
        seed = _nonnegative_integer(run.get("seed"), f"{field}.seed")
        key = (candidate_id, task_id, repetition)
        if key in run_keys:
            raise BenchmarkError(
                "duplicate run for candidate/task/repetition: "
                f"{candidate_id}/{task_id}/{repetition}"
            )
        run_keys.add(key)

        hard_gate_passed = run.get("hard_gate_passed")
        if not isinstance(hard_gate_passed, bool):
            raise BenchmarkError(f"{field}.hard_gate_passed must be a boolean")
        gate_reason_value = run.get("gate_reason")
        if hard_gate_passed:
            if gate_reason_value is not None:
                raise BenchmarkError(f"{field}.gate_reason must be null when the gate passes")
            gate_reason = None
        else:
            gate_reason = _text(gate_reason_value, f"{field}.gate_reason")
        reward = _rate(run.get("reward"), f"{field}.reward")
        if not hard_gate_passed and reward != 0.0:
            raise BenchmarkError(f"{field}.reward must be 0 when the hard gate fails")

        raw_components = _mapping(run.get("components"), f"{field}.components")
        if not raw_components:
            raise BenchmarkError(f"{field}.components must not be empty")
        if any(not isinstance(name, str) for name in raw_components):
            raise BenchmarkError(f"{field}.components keys must be strings")
        components: dict[str, float] = {}
        for name in sorted(raw_components):
            component_name = _text(name, f"{field}.components key")
            components[component_name] = _rate(
                raw_components[name], f"{field}.components.{component_name}"
            )
        names = set(components)
        if component_names is None:
            component_names = names
        elif names != component_names:
            missing = sorted(component_names - names)
            extra = sorted(names - component_names)
            raise BenchmarkError(
                f"{field}.components must match every other run; "
                f"missing={missing}, extra={extra}"
            )

        runs.append(
            {
                "candidate_id": candidate_id,
                "task_id": task_id,
                "repetition": repetition,
                "seed": seed,
                "hard_gate_passed": hard_gate_passed,
                "gate_reason": gate_reason,
                "reward": reward,
                "components": components,
                "metadata": _json_value(
                    _mapping(run.get("metadata", {}), f"{field}.metadata"),
                    f"{field}.metadata",
                ),
            }
        )

    return {
        "schema_version": schema_version,
        "benchmark_id": benchmark_id,
        "metadata": metadata,
        "candidates": sorted(candidates, key=lambda item: item["id"]),
        "runs": sorted(
            runs,
            key=lambda item: (
                item["candidate_id"],
                item["task_id"],
                item["repetition"],
            ),
        ),
        "component_names": sorted(component_names or ()),
    }


def _validate_comparison_matrix(normalized: Mapping[str, Any]) -> dict[str, Any]:
    candidate_ids = [candidate["id"] for candidate in normalized["candidates"]]
    runs_by_candidate: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for run in normalized["runs"]:
        runs_by_candidate[run["candidate_id"]].append(run)

    reference_id = candidate_ids[0]
    if not runs_by_candidate[reference_id]:
        raise BenchmarkError(f"candidate has no runs: {reference_id}")
    reference_keys = {
        (run["task_id"], run["repetition"]) for run in runs_by_candidate[reference_id]
    }
    for candidate_id in candidate_ids:
        candidate_keys = {
            (run["task_id"], run["repetition"])
            for run in runs_by_candidate[candidate_id]
        }
        if candidate_keys != reference_keys:
            missing = sorted(reference_keys - candidate_keys)
            extra = sorted(candidate_keys - reference_keys)
            raise BenchmarkError(
                "every candidate must use an identical task/repetition matrix; "
                f"candidate={candidate_id!r}, missing={missing}, extra={extra}"
            )

    task_ids = sorted({task_id for task_id, _ in reference_keys})
    repetitions_by_task = {
        task_id: sorted(
            repetition for candidate_task, repetition in reference_keys if candidate_task == task_id
        )
        for task_id in task_ids
    }
    expected_repetitions = repetitions_by_task[task_ids[0]]
    inconsistent_tasks = {
        task_id: repetitions
        for task_id, repetitions in repetitions_by_task.items()
        if repetitions != expected_repetitions
    }
    if inconsistent_tasks:
        raise BenchmarkError(
            "every task must use the same repetition identifiers; "
            f"expected={expected_repetitions}, mismatched={inconsistent_tasks}"
        )

    seed_by_key = {
        (run["task_id"], run["repetition"]): run["seed"]
        for run in runs_by_candidate[reference_id]
    }
    for candidate_id in candidate_ids[1:]:
        for run in runs_by_candidate[candidate_id]:
            key = (run["task_id"], run["repetition"])
            if run["seed"] != seed_by_key[key]:
                raise BenchmarkError(
                    "every candidate must use the same seed for each task/repetition; "
                    f"candidate={candidate_id!r}, task={key[0]!r}, repetition={key[1]}, "
                    f"expected={seed_by_key[key]}, got={run['seed']}"
                )

    return {
        "task_ids": task_ids,
        "repetitions": expected_repetitions,
        "seeds": [
            {
                "task_id": task_id,
                "repetition": repetition,
                "seed": seed_by_key[(task_id, repetition)],
            }
            for task_id in task_ids
            for repetition in expected_repetitions
        ],
        "runs_per_candidate": len(reference_keys),
    }


def _rounded(value: float) -> float:
    return round(value, 6)


def _normal_task_mean_ci(task_means: Sequence[float]) -> dict[str, float] | None:
    if len(task_means) < 2:
        return None
    mean = statistics.fmean(task_means)
    standard_error = statistics.stdev(task_means) / math.sqrt(len(task_means))
    z_value = statistics.NormalDist().inv_cdf(0.975)
    return {
        "low": _rounded(max(0.0, mean - z_value * standard_error)),
        "high": _rounded(min(1.0, mean + z_value * standard_error)),
    }


def _distribution(values: Sequence[float], task_means: Sequence[float]) -> dict[str, Any]:
    run_std_dev = statistics.stdev(values) if len(values) > 1 else None
    task_std_dev = statistics.stdev(task_means) if len(task_means) > 1 else None
    task_standard_error = (
        task_std_dev / math.sqrt(len(task_means)) if task_std_dev is not None else None
    )
    return {
        "mean": _rounded(statistics.fmean(values)),
        "median": _rounded(statistics.median(values)),
        "minimum": _rounded(min(values)),
        "maximum": _rounded(max(values)),
        "run_sample_std_dev": _rounded(run_std_dev) if run_std_dev is not None else None,
        "task_mean_sample_std_dev": (
            _rounded(task_std_dev) if task_std_dev is not None else None
        ),
        "task_mean_standard_error": (
            _rounded(task_standard_error) if task_standard_error is not None else None
        ),
        "task_mean_ci95": _normal_task_mean_ci(task_means),
    }


def _wilson_interval(successes: int, total: int) -> dict[str, float]:
    z_value = statistics.NormalDist().inv_cdf(0.975)
    proportion = successes / total
    denominator = 1 + z_value**2 / total
    center = (proportion + z_value**2 / (2 * total)) / denominator
    margin = (
        z_value
        * math.sqrt(
            proportion * (1 - proportion) / total + z_value**2 / (4 * total**2)
        )
        / denominator
    )
    return {
        "low": _rounded(max(0.0, center - margin)),
        "high": _rounded(min(1.0, center + margin)),
    }


def _candidate_summary(
    candidate: Mapping[str, Any],
    runs: Sequence[Mapping[str, Any]],
    task_ids: Sequence[str],
    repetitions: Sequence[int],
    component_names: Sequence[str],
) -> tuple[dict[str, Any], tuple[float, float, float, str]]:
    runs_by_task: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for run in runs:
        runs_by_task[run["task_id"]].append(run)
    for task_runs in runs_by_task.values():
        task_runs.sort(key=lambda item: item["repetition"])

    passed = sum(1 for run in runs if run["hard_gate_passed"])
    reward_values = [run["reward"] for run in runs]
    reward_task_means = [
        statistics.fmean(run["reward"] for run in runs_by_task[task_id])
        for task_id in task_ids
    ]
    component_summaries: dict[str, Any] = {}
    for component_name in component_names:
        values = [run["components"][component_name] for run in runs]
        task_means = [
            statistics.fmean(
                run["components"][component_name] for run in runs_by_task[task_id]
            )
            for task_id in task_ids
        ]
        component_summaries[component_name] = _distribution(values, task_means)

    per_task = []
    for task_id in task_ids:
        task_runs = runs_by_task[task_id]
        task_passed = sum(1 for run in task_runs if run["hard_gate_passed"])
        per_task.append(
            {
                "task_id": task_id,
                "hard_gate_pass_rate": _rounded(task_passed / len(task_runs)),
                "reward_mean": _rounded(
                    statistics.fmean(run["reward"] for run in task_runs)
                ),
                "component_means": {
                    component_name: _rounded(
                        statistics.fmean(
                            run["components"][component_name] for run in task_runs
                        )
                    )
                    for component_name in component_names
                },
            }
        )

    hard_gate_pass_rate = passed / len(runs)
    reward_mean = statistics.fmean(reward_values)
    reward_median = statistics.median(reward_values)
    summary = {
        "candidate_id": candidate["id"],
        "label": candidate["label"],
        "metadata": candidate["metadata"],
        "sample_size": {
            "tasks": len(task_ids),
            "repetitions_per_task": len(repetitions),
            "runs": len(runs),
        },
        "hard_gate": {
            "passed": passed,
            "failed": len(runs) - passed,
            "pass_rate": _rounded(hard_gate_pass_rate),
            "wilson_ci95": _wilson_interval(passed, len(runs)),
            "failure_reasons": dict(
                sorted(
                    Counter(
                        run["gate_reason"]
                        for run in runs
                        if not run["hard_gate_passed"]
                    ).items()
                )
            ),
        },
        "reward": _distribution(reward_values, reward_task_means),
        "components": component_summaries,
        "per_task": per_task,
    }
    ranking_key = (
        -hard_gate_pass_rate,
        -reward_mean,
        -reward_median,
        candidate["id"],
    )
    return summary, ranking_key


def compare_benchmark(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and compare normalized candidate results without executing models.

    Ranking is descriptive and deterministic: hard-gate pass rate, mean reward,
    median reward, then candidate id. Confidence intervals and dispersion are
    reported so consumers do not have to treat the point ranking as certainty.
    """

    normalized = _normalize_input(payload)
    matrix = _validate_comparison_matrix(normalized)
    digest_payload = {
        key: value for key, value in normalized.items() if key != "component_names"
    }
    canonical_input = json.dumps(
        digest_payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    input_digest = hashlib.sha256(canonical_input.encode("utf-8")).hexdigest()

    runs_by_candidate: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for run in normalized["runs"]:
        runs_by_candidate[run["candidate_id"]].append(run)
    ranked: list[tuple[tuple[float, float, float, str], dict[str, Any]]] = []
    for candidate in normalized["candidates"]:
        summary, ranking_key = _candidate_summary(
            candidate,
            runs_by_candidate[candidate["id"]],
            matrix["task_ids"],
            matrix["repetitions"],
            normalized["component_names"],
        )
        ranked.append((ranking_key, summary))
    ranked.sort(key=lambda item: item[0])
    ranking = []
    for rank, (_, summary) in enumerate(ranked, start=1):
        ranking.append({"rank": rank, **summary})

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "benchmark_id": normalized["benchmark_id"],
        "input_sha256": input_digest,
        "metadata": normalized["metadata"],
        "comparison_matrix": matrix,
        "component_names": normalized["component_names"],
        "ranking_policy": {
            "ordered_metrics": [
                "hard_gate.pass_rate descending",
                "reward.mean descending",
                "reward.median descending",
            ],
            "deterministic_tie_breaker": "candidate_id ascending",
        },
        "uncertainty_policy": {
            "hard_gate_pass_rate": "two-sided Wilson score interval at 95%",
            "reward_and_components": (
                "two-sided 95% normal interval over per-task means"
            ),
            "dispersion": "sample standard deviation across runs and per-task means",
            "interpretation": (
                "descriptive comparison only; interval overlap is exposed and no "
                "statistical-significance claim is made"
            ),
        },
        "winner_candidate_id": ranking[0]["candidate_id"],
        "candidates": ranking,
    }


def render_benchmark_markdown(report: Mapping[str, Any]) -> str:
    """Render a concise Markdown view of a comparison report."""

    benchmark_id = _text(report.get("benchmark_id"), "report.benchmark_id")
    candidates = _sequence(report.get("candidates"), "report.candidates")
    matrix = _mapping(report.get("comparison_matrix"), "report.comparison_matrix")
    component_names = _sequence(report.get("component_names"), "report.component_names")

    def escape(value: Any) -> str:
        return str(value).replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")

    def percent(value: float) -> str:
        return f"{100 * value:.1f}%"

    lines = [
        f"# Website benchmark: {escape(benchmark_id)}",
        "",
        (
            f"Compared {len(candidates)} candidates on {len(matrix['task_ids'])} tasks × "
            f"{len(matrix['repetitions'])} repetitions "
            f"({matrix['runs_per_candidate']} runs per candidate)."
        ),
        "",
        "| Rank | Candidate | Hard-gate pass rate (95% CI) | Reward mean ± run SD | Task-mean 95% CI |",
        "| ---: | --- | ---: | ---: | ---: |",
    ]
    for candidate in candidates:
        gate = candidate["hard_gate"]
        gate_ci = gate["wilson_ci95"]
        reward = candidate["reward"]
        reward_sd = reward["run_sample_std_dev"]
        reward_ci = reward["task_mean_ci95"]
        reward_sd_text = "n/a" if reward_sd is None else f"{reward_sd:.4f}"
        reward_ci_text = (
            "n/a"
            if reward_ci is None
            else f"{reward_ci['low']:.4f}–{reward_ci['high']:.4f}"
        )
        lines.append(
            "| "
            f"{candidate['rank']} | {escape(candidate['label'])} | "
            f"{gate['passed']}/{candidate['sample_size']['runs']} "
            f"({percent(gate['pass_rate'])}; "
            f"{percent(gate_ci['low'])}–{percent(gate_ci['high'])}) | "
            f"{reward['mean']:.4f} ± {reward_sd_text} | {reward_ci_text} |"
        )

    lines.extend(
        [
            "",
            "## Component means",
            "",
            "| Candidate | "
            + " | ".join(escape(name) for name in component_names)
            + " |",
            "| --- | " + " | ".join("---:" for _ in component_names) + " |",
        ]
    )
    for candidate in candidates:
        lines.append(
            f"| {escape(candidate['label'])} | "
            + " | ".join(
                f"{candidate['components'][name]['mean']:.4f}"
                for name in component_names
            )
            + " |"
        )

    lines.extend(
        [
            "",
            (
                "Ranking order: hard-gate pass rate, mean reward, median reward, then "
                "candidate id. Intervals are descriptive; overlap is not a significance test."
            ),
            "",
            f"Input SHA-256: `{escape(report['input_sha256'])}`",
            "",
        ]
    )
    return "\n".join(lines)


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8", newline="\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_benchmark_reports(
    report: Mapping[str, Any], *, json_path: str | Path, markdown_path: str | Path
) -> dict[str, str]:
    """Write machine-readable JSON and concise Markdown reports atomically."""

    json_output = Path(json_path).expanduser().resolve()
    markdown_output = Path(markdown_path).expanduser().resolve()
    if json_output == markdown_output:
        raise BenchmarkError("json_path and markdown_path must be different files")
    if json_output.suffix.lower() != ".json":
        raise BenchmarkError("json_path must end in .json")
    if markdown_output.suffix.lower() not in {".md", ".markdown"}:
        raise BenchmarkError("markdown_path must end in .md or .markdown")
    normalized_report = _json_value(_mapping(report, "report"), "report")
    json_text = json.dumps(
        normalized_report,
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    markdown_text = render_benchmark_markdown(normalized_report)
    _atomic_write(json_output, json_text)
    _atomic_write(markdown_output, markdown_text)
    return {"json": str(json_output), "markdown": str(markdown_output)}


def build_benchmark_reports(
    input_path: str | Path,
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> dict[str, Any]:
    """Load normalized result JSON, compare candidates, and write both reports."""

    report = compare_benchmark(load_benchmark_input(input_path))
    write_benchmark_reports(report, json_path=json_path, markdown_path=markdown_path)
    return report


__all__ = [
    "BenchmarkError",
    "INPUT_SCHEMA_VERSION",
    "REPORT_SCHEMA_VERSION",
    "build_benchmark_reports",
    "compare_benchmark",
    "load_benchmark_input",
    "render_benchmark_markdown",
    "write_benchmark_reports",
]
