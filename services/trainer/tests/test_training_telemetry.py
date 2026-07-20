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


if __name__ == "__main__":
    unittest.main()
