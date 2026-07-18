from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import sys
import tempfile
import threading
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
from autotrainer.model_service import select_model  # noqa: E402
from autotrainer.project_gate import ProjectBusyError  # noqa: E402
from autotrainer.source_service import add_source  # noqa: E402


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

    def test_download_holds_the_project_lease_until_commit(self) -> None:
        """Setup cannot change underneath a slow snapshot download."""

        resolved = "c" * 40
        snapshot = self.root / ".autotrainer" / "model-cache" / "snapshot"
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_text("{}", encoding="utf-8")
        examples = self.root / "accepted.jsonl"
        examples.write_text(
            json.dumps(
                {
                    "messages": [
                        {"role": "user", "content": "Build the card"},
                        {"role": "assistant", "content": "Done"},
                    ]
                }
            )
            + "\n",
            encoding="utf-8",
        )
        download_started = threading.Event()
        finish_download = threading.Event()

        def slow_download(**_kwargs: object) -> str:
            download_started.set()
            if not finish_download.wait(timeout=5):
                raise AssertionError("test did not release the model download")
            return str(snapshot)

        with (
            patch(
                "autotrainer.model_cache._resolve_huggingface_revision",
                return_value=resolved,
            ),
            patch(
                "autotrainer.model_cache._hub_functions",
                return_value=(slow_download, object()),
            ),
            ThreadPoolExecutor(max_workers=1) as executor,
        ):
            future = executor.submit(materialize_model, self.config_path)
            self.assertTrue(download_started.wait(timeout=2))
            try:
                with self.assertRaisesRegex(ProjectBusyError, "project is busy"):
                    add_source(self.config_path, str(examples))
            finally:
                # Always unblock the worker so a failed assertion cannot hang
                # the suite inside ThreadPoolExecutor.__exit__.
                finish_download.set()
            result = future.result(timeout=5)

        added = add_source(self.config_path, str(examples))

        config = load_config(self.config_path)
        self.assertEqual(result["revision"], resolved)
        self.assertEqual(config.model["revision"], resolved)
        self.assertEqual([source["id"] for source in config.sources], [added["source"]["id"]])

    def test_relocated_cache_does_not_trust_the_old_receipt(self) -> None:
        resolved = "d" * 40
        old_cache = self.root / ".autotrainer" / "model-cache"
        snapshot = old_cache / "snapshot"
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_text("{}", encoding="utf-8")

        with (
            patch(
                "autotrainer.model_cache._resolve_huggingface_revision",
                return_value=resolved,
            ),
            patch(
                "autotrainer.model_cache._hub_functions",
                return_value=(lambda **_kwargs: str(snapshot), object()),
            ),
        ):
            materialize_model(self.config_path)

        relocated = self.root / "relocated-cache"
        config = load_config(self.config_path)
        config.data["model"]["cache_dir"] = str(relocated)
        write_config(config.path, config.data, overwrite=True)

        def missing_from_new_cache(**kwargs: object) -> str:
            self.assertEqual(Path(str(kwargs["cache_dir"])).resolve(), relocated.resolve())
            raise FileNotFoundError("new cache is empty")

        with patch(
            "autotrainer.model_cache._hub_functions",
            return_value=(missing_from_new_cache, object()),
        ):
            state = inspect_model_cache(self.config_path)

        self.assertEqual(state["status"], "not_downloaded")
        self.assertIsNone(state["snapshot_path"])
        self.assertEqual(state["cache_dir"], str(relocated.resolve()))

    def test_download_rejects_a_new_model_selection_until_commit(self) -> None:
        original_resolution = "e" * 40
        new_revision = "f" * 40
        snapshot = self.root / ".autotrainer" / "model-cache" / "snapshot"
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_text("{}", encoding="utf-8")
        download_started = threading.Event()
        finish_download = threading.Event()

        def slow_download(**_kwargs: object) -> str:
            download_started.set()
            if not finish_download.wait(timeout=5):
                raise AssertionError("test did not release the model download")
            return str(snapshot)

        with (
            patch(
                "autotrainer.model_cache._resolve_huggingface_revision",
                return_value=original_resolution,
            ),
            patch(
                "autotrainer.model_cache._hub_functions",
                return_value=(slow_download, object()),
            ),
            ThreadPoolExecutor(max_workers=1) as executor,
        ):
            future = executor.submit(materialize_model, self.config_path)
            self.assertTrue(download_started.wait(timeout=2))
            try:
                with self.assertRaisesRegex(ProjectBusyError, "project is busy"):
                    select_model(
                        self.config_path,
                        "qwen3.5-9b-text",
                        revision=new_revision,
                    )
            finally:
                finish_download.set()
            result = future.result(timeout=5)

        self.assertEqual(result["revision"], original_resolution)
        self.assertEqual(
            load_config(self.config_path).model["revision"], original_resolution
        )
        select_model(
            self.config_path,
            "qwen3.5-9b-text",
            revision=new_revision,
        )
        self.assertEqual(load_config(self.config_path).model["revision"], new_revision)

    def test_exact_snapshot_check_is_local_only(self) -> None:
        calls: list[dict[str, object]] = []
        revision = "b" * 40
        snapshot = (
            self.root
            / "models--Qwen--Qwen3.5-9B"
            / "snapshots"
            / revision
        )
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_text(
            json.dumps({"model_type": "qwen3_5"}), encoding="utf-8"
        )
        (snapshot / "tokenizer.json").write_text("{}", encoding="utf-8")
        (snapshot / "model.safetensors").write_bytes(b"weights")

        def fake_download(**kwargs: object) -> str:
            calls.append(dict(kwargs))
            return str(snapshot)

        with patch(
            "autotrainer.model_cache._hub_functions",
            return_value=(fake_download, object()),
        ):
            require_materialized_model(
                {
                    "id": "Qwen/Qwen3.5-9B",
                    "revision": revision,
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
