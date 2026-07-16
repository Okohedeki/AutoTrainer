from __future__ import annotations

from http.client import HTTPConnection
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch


SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE_ROOT / "src"))

from autotrainer.config import ConfigError, default_config, load_config, write_config  # noqa: E402
from autotrainer.local_api import create_local_api_server  # noqa: E402


class LocalApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.config_path = self.root / "autotrainer.yaml"
        write_config(self.config_path, default_config(), overwrite=False)
        self.server = create_local_api_server(self.config_path, "127.0.0.1", 0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.connection = HTTPConnection("127.0.0.1", self.server.server_port, timeout=3)

    def tearDown(self) -> None:
        self.connection.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3)
        self.temporary_directory.cleanup()

    def request(self, method: str, path: str, payload: dict[str, object] | None = None) -> tuple[int, dict[str, object]]:
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        headers = {"Content-Type": "application/json"} if body is not None else {}
        self.connection.request(method, path, body=body, headers=headers)
        response = self.connection.getresponse()
        return response.status, json.loads(response.read().decode("utf-8"))

    def test_model_catalog_and_current_status_share_one_contract(self) -> None:
        status, catalog = self.request("GET", "/api/v1/models")
        self.assertEqual(status, 200)
        self.assertIn("qwen3.5-9b-text", catalog["models"])

        status, current = self.request("GET", "/api/v1/model")
        self.assertEqual(status, 200)
        self.assertEqual(current["model"]["id"], "Qwen/Qwen3.5-9B")
        self.assertEqual(current["cache"]["status"], "revision_unresolved")

    def test_model_selection_persists_the_same_yaml_used_by_cli(self) -> None:
        status, result = self.request(
            "POST",
            "/api/v1/model/select",
            {"model": "qwen3.5-9b-text", "cache_dir": "D:/models"},
        )
        self.assertEqual(status, 200)
        self.assertEqual(result["model"]["cache_dir"], "D:/models")
        selected = load_config(self.config_path).model
        self.assertEqual(selected["revision"], "c202236235762e1c871ad0ccb60c8ee5ba337b9a")

    def test_download_endpoint_calls_the_shared_model_operation(self) -> None:
        completed = {"status": "downloaded", "model_id": "Qwen/Qwen3.5-9B"}
        with patch("autotrainer.local_api.download_model", return_value=completed):
            status, result = self.request("POST", "/api/v1/model/download", {})
        self.assertEqual(status, 200)
        self.assertEqual(result, completed)

    def test_source_endpoints_share_the_persisted_source_contract(self) -> None:
        demonstrations = self.root / "accepted.jsonl"
        demonstrations.write_text(
            '{"messages":[{"role":"assistant","content":"accepted"}]}\n',
            encoding="utf-8",
        )

        status, initial = self.request("GET", "/api/v1/sources")
        self.assertEqual(status, 200)
        self.assertEqual(initial, {"sources": []})

        status, added = self.request(
            "POST", "/api/v1/sources", {"value": str(demonstrations)}
        )
        self.assertEqual(status, 200)
        source = added["source"]
        self.assertEqual(
            set(source),
            {"id", "kind", "label", "value", "origin", "purpose", "status"},
        )
        self.assertEqual(source["purpose"], "examples")
        self.assertEqual(source["status"], "ready")
        self.assertEqual(len(load_config(self.config_path).sources), 1)

        status, removed = self.request("DELETE", f"/api/v1/sources/{source['id']}")
        self.assertEqual(status, 200)
        self.assertEqual(removed["removed"], source)
        self.assertEqual(removed["sources"], [])
        self.assertEqual(load_config(self.config_path).sources, [])

    def test_source_endpoint_rejects_missing_value(self) -> None:
        status, result = self.request("POST", "/api/v1/sources", {})

        self.assertEqual(status, 400)
        self.assertEqual(result["error"]["code"], "invalid_request")

    def test_prepare_endpoint_returns_the_shared_project_result_directly(self) -> None:
        prepared = {
            "status": "blocked",
            "recipe": "needs_training_data",
            "summary": "Add training data.",
            "next_action": {"title": "Add training data", "detail": "Add examples or tasks."},
            "steps": [],
            "details": {},
        }
        with patch("autotrainer.local_api.prepare_project", return_value=prepared):
            status, result = self.request("POST", "/api/v1/prepare", {})

        self.assertEqual(status, 200)
        self.assertEqual(result, prepared)

    def test_server_rejects_non_loopback_binding(self) -> None:
        with self.assertRaisesRegex(ConfigError, "loopback"):
            create_local_api_server(self.config_path, "0.0.0.0", 8765)


if __name__ == "__main__":
    unittest.main()
