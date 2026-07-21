from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE_ROOT / "src"))

from autotrainer.training.telemetry import (  # noqa: E402
    make_trainer_log_callback,
    numeric_metrics,
)


class TrainingTelemetryTests(unittest.TestCase):
    def test_numeric_metrics_excludes_text_secrets_booleans_and_non_finite_values(self) -> None:
        self.assertEqual(
            numeric_metrics(
                {
                    "loss": 0.25,
                    "global_step": 7,
                    "ready": True,
                    "note": "private prompt",
                    "token_secret": 42,
                    "nan": math.nan,
                }
            ),
            {"loss": 0.25, "global_step": 7},
        )

    def test_trainer_callback_emits_only_observed_numeric_logs(self) -> None:
        events: list[dict] = []

        class TrainerCallback:
            pass

        callback = make_trainer_log_callback(
            TrainerCallback,
            stage="sft",
            on_event=lambda event: events.append(dict(event)),
        )
        control = object()
        returned = callback.on_log(  # type: ignore[attr-defined]
            object(),
            SimpleNamespace(global_step=12, epoch=0.75),
            control,
            logs={"loss": 0.125, "learning_rate": 1e-4, "prompt": "hidden"},
        )

        self.assertIs(returned, control)
        self.assertEqual(
            events,
            [
                {
                    "type": "trainer_log",
                    "stage": "sft",
                    "step": 12,
                    "epoch": 0.75,
                    "metrics": {"loss": 0.125, "learning_rate": 1e-4},
                }
            ],
        )

    def test_trainer_callback_reports_observed_vram_against_user_limit(self) -> None:
        events: list[dict] = []

        class TrainerCallback:
            pass

        cuda = SimpleNamespace(
            memory_allocated=lambda _device: 6 * 1024**3,
            memory_reserved=lambda _device: 7 * 1024**3,
        )
        callback = make_trainer_log_callback(
            TrainerCallback,
            stage="grpo",
            on_event=lambda event: events.append(dict(event)),
            torch_module=SimpleNamespace(cuda=cuda),
            vram_limit_gib=12.0,
        )
        callback.on_log(  # type: ignore[attr-defined]
            object(),
            SimpleNamespace(global_step=2, epoch=None),
            object(),
            logs={"loss": 0.5},
        )

        self.assertEqual(
            events[0]["metrics"],
            {
                "loss": 0.5,
                "vram_allocated_gib": 6.0,
                "vram_limit_gib": 12.0,
                "vram_reserved_gib": 7.0,
            },
        )

    def test_trainer_callback_reports_window_throughput_and_peak_vram(self) -> None:
        events: list[dict] = []
        observed = iter((10.0, 12.0, 15.0))

        class TrainerCallback:
            pass

        cuda = SimpleNamespace(
            memory_allocated=lambda _device: 6 * 1024**3,
            memory_reserved=lambda _device: 7 * 1024**3,
            max_memory_allocated=lambda _device: 8 * 1024**3,
            max_memory_reserved=lambda _device: 9 * 1024**3,
            reset_peak_memory_stats=lambda _device: None,
        )
        callback = make_trainer_log_callback(
            TrainerCallback,
            stage="sft",
            on_event=lambda event: events.append(dict(event)),
            torch_module=SimpleNamespace(cuda=cuda),
            vram_limit_gib=12.0,
            clock=lambda: next(observed),
        )
        control = object()
        callback.on_train_begin(  # type: ignore[attr-defined]
            object(), SimpleNamespace(global_step=0), control
        )
        callback.on_log(  # type: ignore[attr-defined]
            object(),
            SimpleNamespace(global_step=4, epoch=0.5),
            control,
            logs={"loss": 0.5},
        )
        callback.on_train_end(  # type: ignore[attr-defined]
            object(), SimpleNamespace(global_step=4), control
        )

        self.assertEqual(events[0]["metrics"]["observed_steps_per_second"], 2.0)
        self.assertEqual(events[0]["metrics"]["observed_seconds_per_step"], 0.5)
        self.assertEqual(events[0]["metrics"]["vram_peak_allocated_gib"], 8.0)
        self.assertEqual(
            callback.observed_summary(),  # type: ignore[attr-defined]
            {
                "observed_train_seconds": 5.0,
                "observed_train_steps": 4,
                "vram_allocated_gib": 6.0,
                "vram_limit_gib": 12.0,
                "vram_peak_allocated_gib": 8.0,
                "vram_peak_reserved_gib": 9.0,
                "vram_reserved_gib": 7.0,
            },
        )


if __name__ == "__main__":
    unittest.main()
