from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE_ROOT / "src"))

from autotrainer.config import default_config, write_config  # noqa: E402
from autotrainer.model_cache import (  # noqa: E402
    inspect_reference_model,
    materialize_reference_model,
)


class ReferenceModelCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.config_path = self.root / "autotrainer.yaml"
        payload = default_config(revision="e" * 40)
        payload["model"]["cache_dir"] = "./shared-cache"
        write_config(self.config_path, payload, overwrite=False)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_reference_status_is_local_only_and_separate_from_training_base(self) -> None:
        with patch(
            "autotrainer.model_cache._hub_functions",
            return_value=(unittest.mock.Mock(side_effect=FileNotFoundError), unittest.mock.Mock()),
        ):
            status = inspect_reference_model(self.config_path)

        self.assertEqual(status["status"], "not_downloaded")
        self.assertEqual(status["alias"], "qwythos-9b-reference")
        self.assertEqual(
            status["model_id"],
            "empero-ai/Qwythos-9B-Claude-Mythos-5-1M",
        )

    def test_download_writes_a_distinct_token_free_reference_receipt(self) -> None:
        snapshot = self.root / "snapshot"
        snapshot.mkdir()
        (snapshot / "config.json").write_text("{}", encoding="utf-8")
        download = unittest.mock.Mock(return_value=str(snapshot))
        with (
            patch("autotrainer.model_cache._hub_functions", return_value=(download, unittest.mock.Mock())),
            patch.dict("os.environ", {"HF_TOKEN": "hf_private"}),
        ):
            result = materialize_reference_model(self.config_path)

        self.assertEqual(result["status"], "downloaded")
        receipt = Path(result["receipt"])
        value = json.loads(receipt.read_text(encoding="utf-8"))
        self.assertNotIn("token", json.dumps(value).lower())
        self.assertIn("references", receipt.parts)
        download.assert_called_once()
        self.assertEqual(download.call_args.kwargs["token"], "hf_private")


if __name__ == "__main__":
    unittest.main()
