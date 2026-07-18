from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE_ROOT / "src"))

from autotrainer.config import ConfigError, default_config, load_config, write_config  # noqa: E402
from autotrainer.model_cache import (  # noqa: E402
    adopt_local_model,
    discover_local_models,
    inspect_model_cache,
)
from autotrainer.models import MODEL_CATALOG  # noqa: E402


REVISION = "a" * 40
MODEL_ID = str(MODEL_CATALOG["qwen3.5-9b-text"]["id"])


def _snapshot(cache_root: Path, revision: str = REVISION) -> Path:
    """Create the smallest structurally usable Transformers cache snapshot."""

    snapshot = (
        cache_root
        / "models--Qwen--Qwen3.5-9B"
        / "snapshots"
        / revision
    )
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text(
        json.dumps({"model_type": "qwen3_5"}),
        encoding="utf-8",
    )
    (snapshot / "tokenizer.json").write_text("{}", encoding="utf-8")
    (snapshot / "model.safetensors").write_bytes(b"local weights")
    return snapshot


class LocalModelDiscoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.config_path = self.root / "autotrainer.yaml"
        payload = default_config(revision="main")
        payload["model"]["cache_dir"] = str(self.root / "project-cache")
        write_config(self.config_path, payload, overwrite=False)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _isolated_environment(self, **values: str):
        # Keep home/temp available on Windows while excluding real developer
        # cache overrides from deterministic discovery tests.
        environment = {
            "HOME": str(self.root / "home"),
            "USERPROFILE": str(self.root / "home"),
            "TEMP": str(self.root / "temp"),
            "TMP": str(self.root / "temp"),
            **values,
        }
        return patch.dict(os.environ, environment, clear=True)

    def test_discovers_effective_hub_cache_offline_without_exposing_paths(self) -> None:
        hub_cache = self.root / "hub-cache"
        _snapshot(hub_cache)

        with (
            self._isolated_environment(HF_HUB_CACHE=str(hub_cache)),
            patch(
                "autotrainer.model_cache._hub_functions",
                side_effect=AssertionError("discovery must not call Hub functions"),
            ),
        ):
            workspace = discover_local_models(self.config_path)

        self.assertEqual(workspace["scanned_cache_count"], 1)
        self.assertEqual(workspace["ignored_incomplete_count"], 0)
        self.assertEqual(len(workspace["models"]), 1)
        candidate = workspace["models"][0]
        self.assertEqual(candidate["model_id"], MODEL_ID)
        self.assertEqual(candidate["revision"], REVISION)
        self.assertEqual(candidate["availability"], "available")
        self.assertEqual(candidate["source"], "huggingface_cache")
        self.assertFalse(candidate["selected"])
        serialized = json.dumps(workspace)
        self.assertNotIn(str(hub_cache), serialized)
        self.assertNotIn("snapshot_path", serialized)
        self.assertNotIn("cache_dir", serialized)

    def test_incomplete_weight_index_is_counted_but_not_selectable(self) -> None:
        cache = self.root / "project-cache"
        snapshot = _snapshot(cache)
        (snapshot / "model.safetensors").unlink()
        (snapshot / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "weight_map": {
                        "layer.one": "model-00001-of-00002.safetensors",
                        "layer.two": "model-00002-of-00002.safetensors",
                    }
                }
            ),
            encoding="utf-8",
        )
        (snapshot / "model-00001-of-00002.safetensors").write_bytes(b"first")

        with self._isolated_environment():
            workspace = discover_local_models(self.config_path)

        self.assertEqual(workspace["models"], [])
        self.assertEqual(workspace["ignored_incomplete_count"], 1)

    def test_adoption_revalidates_then_persists_selection_and_normal_receipt(self) -> None:
        hub_cache = self.root / "hub-cache"
        snapshot = _snapshot(hub_cache)
        with self._isolated_environment(HF_HUB_CACHE=str(hub_cache)):
            candidate = discover_local_models(self.config_path)["models"][0]
            result = adopt_local_model(self.config_path, candidate["candidate_id"])
            status = inspect_model_cache(self.config_path)

        model = load_config(self.config_path).model
        self.assertEqual(model["id"], MODEL_ID)
        self.assertEqual(model["revision"], REVISION)
        self.assertEqual(Path(str(model["cache_dir"])).resolve(), hub_cache.resolve())
        self.assertEqual(model["loader"], "qwen3_5_text")
        self.assertTrue(result["candidate"]["selected"])
        self.assertNotIn("snapshot_path", result["candidate"])
        self.assertNotIn("cache_dir", result["candidate"])
        self.assertEqual(status["status"], "downloaded")
        receipt = json.loads(
            (self.root / ".autotrainer" / "models" / "current.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(receipt["source"], "adopted_local_cache")
        self.assertEqual(Path(receipt["snapshot_path"]), snapshot.resolve())

    def test_removed_candidate_is_rejected_and_stale_receipt_is_not_trusted(self) -> None:
        hub_cache = self.root / "hub-cache"
        snapshot = _snapshot(hub_cache)
        with self._isolated_environment(HF_HUB_CACHE=str(hub_cache)):
            candidate_id = discover_local_models(self.config_path)["models"][0][
                "candidate_id"
            ]
            (snapshot / "model.safetensors").unlink()
            with self.assertRaisesRegex(ConfigError, "scan local models again"):
                adopt_local_model(self.config_path, candidate_id)

            # Restore, adopt, then simulate an external cache cleanup. A stale
            # receipt must no longer make an incomplete snapshot look ready.
            (snapshot / "model.safetensors").write_bytes(b"restored")
            candidate_id = discover_local_models(self.config_path)["models"][0][
                "candidate_id"
            ]
            adopt_local_model(self.config_path, candidate_id)
            (snapshot / "model.safetensors").unlink()
            with patch(
                "autotrainer.model_cache._hub_functions",
                return_value=(lambda **_kwargs: str(snapshot), object()),
            ):
                status = inspect_model_cache(self.config_path)

        self.assertEqual(status["status"], "not_downloaded")


if __name__ == "__main__":
    unittest.main()
