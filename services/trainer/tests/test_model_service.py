from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE_ROOT / "src"))

from autotrainer.config import ConfigError, default_config, load_config, write_config  # noqa: E402
from autotrainer.model_service import (  # noqa: E402
    ModelSearchError,
    _hub_api,
    search_models,
    select_model,
)
from autotrainer.models import MODEL_CATALOG  # noqa: E402


class FakeHubApi:
    def __init__(self, records: list[object] | None = None, error: Exception | None = None) -> None:
        self.records = records or []
        self.error = error
        self.calls: list[dict[str, object]] = []

    def list_models(self, **kwargs: object) -> list[object]:
        self.calls.append(dict(kwargs))
        if self.error is not None:
            raise self.error
        return list(self.records)


class ModelSearchTests(unittest.TestCase):
    def test_search_normalizes_exact_revisions_and_catalog_compatibility(self) -> None:
        qwen = MODEL_CATALOG["qwen3.5-9b-text"]
        reference = MODEL_CATALOG["qwythos-9b-reference"]
        hub = FakeHubApi(
            [
                SimpleNamespace(
                    id=qwen["id"],
                    sha="A" * 40,
                    pipeline_tag="text-generation",
                    library_name="transformers",
                    downloads=120,
                    likes=9,
                    gated=False,
                    private=False,
                ),
                SimpleNamespace(
                    id=reference["id"],
                    sha="b" * 40,
                    pipeline_tag="text-generation",
                    downloads=20,
                    likes=2,
                    gated="manual",
                    private=False,
                ),
                {
                    "id": "someone/another-9b",
                    "sha": "c" * 64,
                    "pipeline_tag": "text-generation",
                    "downloads": -4,
                    "likes": "3",
                },
                {"id": "someone/unpinned", "sha": "main"},
            ]
        )

        with patch("autotrainer.model_service._hub_api", return_value=hub):
            results = search_models("  qwen  ", limit=4)

        self.assertEqual(
            hub.calls,
            [
                {
                    "search": "qwen",
                    "sort": "downloads",
                    "direction": -1,
                    "limit": 4,
                    "full": True,
                }
            ],
        )
        self.assertEqual(results[0]["revision"], "a" * 40)
        self.assertEqual(results[0]["compatibility"], "supported")
        self.assertTrue(results[0]["trainable_v1"])
        self.assertEqual(results[1]["compatibility"], "reference_only")
        self.assertTrue(results[1]["gated"])
        self.assertEqual(results[2]["compatibility"], "unverified")
        self.assertEqual(results[2]["downloads"], 0)
        self.assertEqual(results[2]["likes"], 3)
        self.assertIsNone(results[3]["revision"])

    def test_search_bounds_query_and_limit_before_network_access(self) -> None:
        hub = FakeHubApi()
        with patch("autotrainer.model_service._hub_api", return_value=hub):
            for query in ("x", "x" * 101):
                with self.subTest(query_length=len(query)), self.assertRaises(ConfigError):
                    search_models(query)
            with self.assertRaises(ConfigError):
                search_models(42)  # type: ignore[arg-type]
            for limit in (0, 26, True, 1.5):
                with self.subTest(limit=limit), self.assertRaises(ConfigError):
                    search_models("qwen", limit=limit)  # type: ignore[arg-type]
        self.assertEqual(hub.calls, [])

    def test_upstream_errors_and_results_never_expose_hf_token(self) -> None:
        secret = "hf_private_token"
        hub = FakeHubApi(error=RuntimeError(f"Authorization: Bearer {secret}"))
        with (
            patch("autotrainer.model_service._hub_api", return_value=hub),
            patch.dict(os.environ, {"HF_TOKEN": secret}),
            self.assertRaises(ModelSearchError) as raised,
        ):
            search_models("qwen")
        self.assertNotIn(secret, str(raised.exception))

        safe_hub = FakeHubApi([{"id": "someone/model", "sha": "d" * 40}])
        with (
            patch("autotrainer.model_service._hub_api", return_value=safe_hub),
            patch.dict(os.environ, {"HF_TOKEN": secret}),
        ):
            payload = search_models("model")
        self.assertNotIn(secret, json.dumps(payload))

    def test_hub_dependency_is_imported_lazily_and_receives_env_token(self) -> None:
        captured: list[str | None] = []
        fake_module = ModuleType("huggingface_hub")

        class CapturingApi:
            def __init__(self, *, token: str | None = None) -> None:
                captured.append(token)

        fake_module.HfApi = CapturingApi  # type: ignore[attr-defined]
        with (
            patch.dict(sys.modules, {"huggingface_hub": fake_module}),
            patch.dict(os.environ, {"HF_TOKEN": "in-memory-only"}),
        ):
            client = _hub_api()

        self.assertIsInstance(client, CapturingApi)
        self.assertEqual(captured, ["in-memory-only"])


class ModelSelectionPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.config_path = Path(self.temporary_directory.name) / "autotrainer.yaml"
        write_config(self.config_path, default_config(), overwrite=False)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_exact_reference_repo_id_cannot_bypass_catalog_policy(self) -> None:
        reference_id = str(MODEL_CATALOG["qwythos-9b-reference"]["id"])
        with self.assertRaisesRegex(ConfigError, "not a validated V1 training base"):
            select_model(self.config_path, reference_id)

    def test_unknown_custom_id_remains_declarable_for_advanced_authoring(self) -> None:
        revision = "e" * 40
        result = select_model(
            self.config_path,
            "someone/custom-model",
            revision=revision,
        )
        self.assertIsNone(result["catalog_key"])
        self.assertEqual(load_config(self.config_path).model["id"], "someone/custom-model")


if __name__ == "__main__":
    unittest.main()
