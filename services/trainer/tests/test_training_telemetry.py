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


if __name__ == "__main__":
    unittest.main()
