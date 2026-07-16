from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE_ROOT / "src"))

from autotrainer.config import default_config, load_config, write_config  # noqa: E402
from autotrainer.model_cache import (  # noqa: E402
    ModelCacheError,
    inspect_model_cache,
    materialize_model,
    require_materialized_model,
)


class ModelCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.config_path = self.root / "autotrainer.yaml"
        write_config(self.config_path, default_config(), overwrite=False)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_mutable_revision_is_never_reported_as_downloaded(self) -> None:
        state = inspect_model_cache(self.config_path)
        self.assertEqual(state["status"], "revision_unresolved")
        self.assertFalse(state["immutable"])
        self.assertEqual(
            state["cache_dir"],
            str((self.root / ".autotrainer" / "model-cache").resolve()),
        )

    def test_materialize_pins_yaml_and_writes_token_free_receipt(self) -> None:
        snapshot = self.root / "hub" / "snapshot"
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_text("{}", encoding="utf-8")
        resolved = "a" * 40

        def fake_download(**kwargs: object) -> str:
            self.assertEqual(kwargs["revision"], resolved)
            # Authentication is passed to the Hub client in memory. The
            # security boundary is persistence: YAML and receipts must never
            # contain the credential after the download completes.
            self.assertEqual(kwargs["token"], "secret-token")
            return str(snapshot)

        with (
            patch(
                "autotrainer.model_cache._resolve_huggingface_revision",
                return_value=resolved,
            ),
            patch(
                "autotrainer.model_cache._hub_functions",
                return_value=(fake_download, object()),
            ),
            patch.dict("os.environ", {"HF_TOKEN": "secret-token"}),
        ):
            result = materialize_model(self.config_path)

        self.assertEqual(result["status"], "downloaded")
        self.assertEqual(load_config(self.config_path).model["revision"], resolved)
        self.assertNotIn(
            "secret-token",
            self.config_path.read_text(encoding="utf-8"),
        )
        receipt_text = Path(result["receipt"]).read_text(encoding="utf-8")
        self.assertNotIn("secret-token", receipt_text)
        self.assertEqual(json.loads(receipt_text)["snapshot_path"], str(snapshot.resolve()))

    def test_exact_snapshot_check_is_local_only(self) -> None:
        calls: list[dict[str, object]] = []

        def fake_download(**kwargs: object) -> str:
            calls.append(dict(kwargs))
            return str(self.root)

        with patch(
            "autotrainer.model_cache._hub_functions",
            return_value=(fake_download, object()),
        ):
            require_materialized_model(
                {
                    "id": "Qwen/Qwen3.5-9B",
                    "revision": "b" * 40,
                    "cache_dir": str(self.root),
                }
            )

        self.assertTrue(calls[0]["local_files_only"])
        self.assertEqual(calls[0]["revision"], "b" * 40)

    def test_real_training_rejects_mutable_revision(self) -> None:
        with self.assertRaisesRegex(ModelCacheError, "immutable downloaded"):
            require_materialized_model(
                {
                    "id": "Qwen/Qwen3.5-9B",
                    "revision": "main",
                    "cache_dir": str(self.root),
                }
            )


if __name__ == "__main__":
    unittest.main()
