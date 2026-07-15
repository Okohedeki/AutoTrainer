from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

import sys

SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE_ROOT / "src"))

from autotrainer.cli import main  # noqa: E402
from autotrainer.config import default_config, load_config, validate_mapping, write_config  # noqa: E402


class ConfigTests(unittest.TestCase):
    def test_default_is_valid_and_explicit_about_qlora_to_grpo(self) -> None:
        payload = default_config()
        report = validate_mapping(payload)
        self.assertEqual(report.errors, ())
        self.assertEqual(payload["model"]["id"], "Qwen/Qwen3.5-9B")
        self.assertEqual(payload["model"]["quantization"]["quant_type"], "nf4")
        self.assertEqual(payload["grpo"]["sft_adapter"], ".autotrainer/checkpoints/sft")

    def test_rejects_group_size_that_does_not_divide_effective_batch(self) -> None:
        payload = default_config()
        payload["grpo"]["num_generations"] = 3
        report = validate_mapping(payload)
        self.assertTrue(any("divisible" in error for error in report.errors))

    def test_round_trips_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "autotrainer.yaml"
            write_config(path, default_config(), overwrite=False)
            loaded = load_config(path)
            self.assertEqual(loaded.data["project"]["seed"], 42)
            self.assertEqual(loaded.root, Path(directory).resolve())


class CliTests(unittest.TestCase):
    def test_init_and_model_use(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(main(["init", directory, "--name", "test-project"]), 0)
            path = Path(directory) / "autotrainer.yaml"
            self.assertTrue(path.exists())
            self.assertEqual(
                main(
                    [
                        "model",
                        "use",
                        "qwen3.5-9b-text",
                        "--revision",
                        "abc123",
                        "--config",
                        str(path),
                    ]
                ),
                0,
            )
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["model"]["revision"], "abc123")


if __name__ == "__main__":
    unittest.main()
