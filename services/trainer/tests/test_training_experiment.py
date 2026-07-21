from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE_ROOT / "src"))

from autotrainer.training.experiment import (  # noqa: E402
    PhaseProfiler,
    build_training_receipt,
    write_training_receipt,
)


class TrainingExperimentTests(unittest.TestCase):
    def test_phase_profiler_records_non_overlapping_wall_time(self) -> None:
        observed = iter((10.0, 12.0, 17.5, 20.0))
        profiler = PhaseProfiler(clock=lambda: next(observed))

        profiler.checkpoint("preflight")
        profiler.checkpoint("training")

        self.assertEqual(
            profiler.summary(),
            {
                "clock": "monotonic_wall_time",
                "phase_seconds": {"preflight": 2.0, "training": 5.5},
                "total_seconds": 10.0,
            },
        )

    def test_training_receipt_is_atomic_and_contains_fixed_evidence(self) -> None:
        receipt = build_training_receipt(
            stage="sft",
            recipe={
                "model": {
                    "id": "Qwen/Qwen3.5-9B",
                    "revision": "a" * 40,
                    "attn_implementation": "sdpa",
                    "effective_attn_implementation": "sdpa",
                },
                "sft": {"dataset": {"sha256": "b" * 64}, "seed": 42},
            },
            dependencies={"torch": "2.13.0+cu130"},
            runtime={"device_name": "Test GPU", "vram_limit_gib": 12.0},
            trainable_adapter_parameters=128,
            metrics={"train_steps_per_second": 2.5},
            telemetry={"vram_peak_allocated_gib": 8.0},
            profile={"phase_seconds": {"training": 4.0}, "total_seconds": 5.0},
        )

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            path = write_training_receipt(destination, receipt)
            stored = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(stored["schema_version"], 1)
        self.assertEqual(stored["recipe"]["model"]["effective_attn_implementation"], "sdpa")
        self.assertEqual(stored["telemetry"]["vram_peak_allocated_gib"], 8.0)
        self.assertEqual(stored["profile"]["phase_seconds"]["training"], 4.0)


if __name__ == "__main__":
    unittest.main()
