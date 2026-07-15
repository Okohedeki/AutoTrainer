from __future__ import annotations

import json
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE_ROOT / "src"))

from autotrainer.benchmark import (  # noqa: E402
    BenchmarkError,
    build_benchmark_reports,
    compare_benchmark,
    render_benchmark_markdown,
)


COMPONENTS = [
    "design_rules",
    "patch_quality",
    "regression_safety",
    "responsive_rules",
    "task_tests",
]


def _components(score: float) -> dict[str, float]:
    return {name: score for name in COMPONENTS}


def _payload() -> dict:
    candidates = [
        {
            "id": "stronger-reference",
            "label": "Stronger 9B reference",
            "metadata": {"model": "reference/model", "revision": "abc123"},
        },
        {
            "id": "base-fable",
            "label": "Base 9B + Fable",
            "metadata": {"model": "Qwen/Qwen3.5-9B", "orchestrator": "fable"},
        },
        {
            "id": "autotrainer-trained",
            "label": "AutoTrainer-trained 9B",
            "metadata": {"adapter": "sha256:trained", "orchestrator": "fable"},
        },
    ]
    outcomes = {
        "stronger-reference": {
            ("pricing", 0): (True, None, 0.82, 0.82),
            ("pricing", 1): (True, None, 0.84, 0.84),
            ("dashboard", 0): (True, None, 0.80, 0.80),
            ("dashboard", 1): (False, "build_failed", 0.0, 0.50),
        },
        "base-fable": {
            ("pricing", 0): (True, None, 0.72, 0.72),
            ("pricing", 1): (False, "regression_failed", 0.0, 0.60),
            ("dashboard", 0): (True, None, 0.70, 0.70),
            ("dashboard", 1): (False, "build_failed", 0.0, 0.40),
        },
        "autotrainer-trained": {
            ("pricing", 0): (True, None, 0.90, 0.90),
            ("pricing", 1): (True, None, 0.88, 0.88),
            ("dashboard", 0): (True, None, 0.86, 0.86),
            ("dashboard", 1): (True, None, 0.84, 0.84),
        },
    }
    seeds = {("pricing", 0): 100, ("pricing", 1): 101, ("dashboard", 0): 200, ("dashboard", 1): 201}
    runs = []
    for candidate_id, candidate_outcomes in outcomes.items():
        for (task_id, repetition), (passed, reason, reward, component) in candidate_outcomes.items():
            runs.append(
                {
                    "candidate_id": candidate_id,
                    "task_id": task_id,
                    "repetition": repetition,
                    "seed": seeds[(task_id, repetition)],
                    "hard_gate_passed": passed,
                    "gate_reason": reason,
                    "reward": reward,
                    "components": _components(component),
                    "metadata": {"artifact": f"{candidate_id}/{task_id}/{repetition}"},
                }
            )
    return {
        "schema_version": "1.0",
        "benchmark_id": "website-design-v1",
        "metadata": {"task_suite_revision": "locked-task-suite"},
        "candidates": candidates,
        "runs": runs,
    }


class BenchmarkComparisonTests(unittest.TestCase):
    def test_ranks_three_candidates_and_exposes_uncertainty(self) -> None:
        report = compare_benchmark(_payload())
        self.assertEqual(report["winner_candidate_id"], "autotrainer-trained")
        self.assertEqual(
            [candidate["candidate_id"] for candidate in report["candidates"]],
            ["autotrainer-trained", "stronger-reference", "base-fable"],
        )
        winner = report["candidates"][0]
        self.assertEqual(winner["hard_gate"]["pass_rate"], 1.0)
        self.assertEqual(winner["reward"]["mean"], 0.87)
        self.assertIsNotNone(winner["hard_gate"]["wilson_ci95"])
        self.assertIsNotNone(winner["reward"]["task_mean_ci95"])
        self.assertGreater(winner["reward"]["run_sample_std_dev"], 0)
        self.assertEqual(winner["components"]["design_rules"]["mean"], 0.87)

    def test_semantically_identical_input_order_has_same_digest(self) -> None:
        payload = _payload()
        reordered = deepcopy(payload)
        reordered["candidates"].reverse()
        reordered["runs"].reverse()
        self.assertEqual(
            compare_benchmark(payload)["input_sha256"],
            compare_benchmark(reordered)["input_sha256"],
        )

    def test_rejects_mismatched_task_matrix(self) -> None:
        payload = _payload()
        payload["runs"] = [
            run
            for run in payload["runs"]
            if not (
                run["candidate_id"] == "base-fable"
                and run["task_id"] == "dashboard"
                and run["repetition"] == 1
            )
        ]
        with self.assertRaisesRegex(BenchmarkError, "identical task/repetition matrix"):
            compare_benchmark(payload)

    def test_rejects_different_repetitions_between_tasks(self) -> None:
        payload = _payload()
        payload["runs"] = [
            run
            for run in payload["runs"]
            if not (run["task_id"] == "dashboard" and run["repetition"] == 1)
        ]
        with self.assertRaisesRegex(BenchmarkError, "same repetition identifiers"):
            compare_benchmark(payload)

    def test_rejects_seed_mismatch(self) -> None:
        payload = _payload()
        target = next(
            run
            for run in payload["runs"]
            if run["candidate_id"] == "base-fable" and run["task_id"] == "pricing"
        )
        target["seed"] += 1
        with self.assertRaisesRegex(BenchmarkError, "same seed"):
            compare_benchmark(payload)

    def test_rejects_nonzero_reward_after_gate_failure(self) -> None:
        payload = _payload()
        target = next(run for run in payload["runs"] if not run["hard_gate_passed"])
        target["reward"] = 0.5
        with self.assertRaisesRegex(BenchmarkError, "must be 0"):
            compare_benchmark(payload)

    def test_rejects_component_schema_drift(self) -> None:
        payload = _payload()
        del payload["runs"][0]["components"]["design_rules"]
        with self.assertRaisesRegex(BenchmarkError, "components must match"):
            compare_benchmark(payload)

    def test_ties_use_candidate_id_for_deterministic_order(self) -> None:
        payload = _payload()
        template = [
            run for run in payload["runs"] if run["candidate_id"] == "autotrainer-trained"
        ]
        for candidate_id in ("base-fable", "stronger-reference"):
            replacements = []
            for run in template:
                replacement = deepcopy(run)
                replacement["candidate_id"] = candidate_id
                replacements.append(replacement)
            payload["runs"] = [
                run for run in payload["runs"] if run["candidate_id"] != candidate_id
            ] + replacements
        report = compare_benchmark(payload)
        self.assertEqual(
            [candidate["candidate_id"] for candidate in report["candidates"]],
            ["autotrainer-trained", "base-fable", "stronger-reference"],
        )

    def test_markdown_is_concise_and_contains_component_table(self) -> None:
        markdown = render_benchmark_markdown(compare_benchmark(_payload()))
        self.assertIn("# Website benchmark: website-design-v1", markdown)
        self.assertIn("AutoTrainer-trained 9B", markdown)
        self.assertIn("## Component means", markdown)
        self.assertIn("Input SHA-256", markdown)


class BenchmarkFileTests(unittest.TestCase):
    def test_builds_json_and_markdown_reports_from_normalized_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "runs.json"
            json_path = root / "reports" / "comparison.json"
            markdown_path = root / "reports" / "comparison.md"
            input_path.write_text(json.dumps(_payload()), encoding="utf-8")

            report = build_benchmark_reports(
                input_path,
                json_path=json_path,
                markdown_path=markdown_path,
            )

            written = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(written, report)
            self.assertEqual(written["winner_candidate_id"], "autotrainer-trained")
            self.assertIn(
                "Hard-gate pass rate",
                markdown_path.read_text(encoding="utf-8"),
            )


if __name__ == "__main__":
    unittest.main()
