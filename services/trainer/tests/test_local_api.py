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
from autotrainer.project_gate import project_run_gate  # noqa: E402


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

    def test_history_endpoints_share_the_review_service_contract(self) -> None:
        workspace = {
            "summary": {"reviewable_count": 1, "approved_count": 0},
            "candidates": [{"candidate_id": "sha256:" + "a" * 64}],
        }
        with patch("autotrainer.local_api.get_history_workspace", return_value=workspace):
            status, result = self.request("GET", "/api/v1/history")
        self.assertEqual(status, 200)
        self.assertEqual(result, workspace)

        refreshed = {
            "summary": {"reviewable_count": 0, "approved_count": 1},
            "candidates": [],
        }
        with patch(
            "autotrainer.local_api.review_history_candidate", return_value=refreshed
        ) as review:
            status, result = self.request(
                "POST",
                "/api/v1/history/review",
                {
                    "candidate_id": "sha256:" + "a" * 64,
                    "decision": "approved",
                    "instruction": "Keep the narrow layout usable.",
                    "rights_confirmed": True,
                },
            )
        self.assertEqual(status, 200)
        self.assertEqual(result, refreshed)
        review.assert_called_once_with(
            self.config_path.resolve(),
            candidate_id="sha256:" + "a" * 64,
            decision="approved",
            instruction="Keep the narrow layout usable.",
            rights_confirmed=True,
        )

        with patch(
            "autotrainer.local_api.retire_stale_reviews", return_value=refreshed
        ) as retire:
            status, result = self.request(
                "POST", "/api/v1/history/retire-stale", {}
            )
        self.assertEqual(status, 200)
        self.assertEqual(result, refreshed)
        retire.assert_called_once_with(self.config_path.resolve())

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

    def test_active_training_returns_a_project_busy_conflict(self) -> None:
        with project_run_gate(self.config_path):
            status, result = self.request("POST", "/api/v1/prepare", {})

        self.assertEqual(status, 409)
        self.assertEqual(result["error"]["code"], "project_busy")

    def test_training_endpoints_use_the_server_owned_single_job_manager(self) -> None:
        status, idle = self.request("GET", "/api/v1/training")
        self.assertEqual(status, 200)
        self.assertEqual(
            idle,
            {
                "id": None,
                "status": "idle",
                "recipe": None,
                "stage": None,
                "message": "No training job has started.",
                "result": None,
            },
        )

        queued = {
            "id": "job-1",
            "status": "queued",
            "recipe": None,
            "stage": "prepare",
            "message": "Training is queued.",
            "result": None,
        }
        with patch.object(self.server.training, "start", return_value=queued) as start:
            status, result = self.request("POST", "/api/v1/training/start", {})

        self.assertEqual(status, 202)
        self.assertEqual(result, queued)
        start.assert_called_once_with()

        status, rejected = self.request(
            "POST", "/api/v1/training/start", {"learning_rate": 1}
        )
        self.assertEqual(status, 400)
        self.assertEqual(rejected["error"]["code"], "invalid_request")

    def test_evaluation_endpoints_use_the_server_owned_job_manager(self) -> None:
        workspace = {
            "readiness": {"status": "ready", "blockers": []},
            "plan": {"plan_id": "sha256:" + "a" * 64},
            "job": {"status": "idle", "phase": "idle"},
            "suites": [
                {
                    "id": "fable_ab",
                    "runner_type": "external",
                    "phase": "awaiting_external_results",
                }
            ],
        }
        with patch.object(self.server.evaluation, "workspace", return_value=workspace):
            status, result = self.request("GET", "/api/v1/evaluation")
        self.assertEqual(status, 200)
        self.assertEqual(result, workspace)

        with patch.object(self.server.evaluation, "plan", return_value=workspace) as plan:
            status, result = self.request("POST", "/api/v1/evaluation/plan", {})
        self.assertEqual(status, 200)
        self.assertEqual(result, workspace)
        plan.assert_called_once_with()

        queued = {
            "id": "a" * 32,
            "status": "queued",
            "suite": "model_benchmark",
            "phase": "queued",
        }
        with patch.object(self.server.evaluation, "start", return_value=queued) as start:
            status, result = self.request(
                "POST",
                "/api/v1/evaluation/start",
                {"suite": "model_benchmark"},
            )
        self.assertEqual(status, 202)
        self.assertEqual(result, queued)
        start.assert_called_once_with("model_benchmark")

        status, rejected = self.request(
            "POST",
            "/api/v1/evaluation/start",
            {"suite": "model_benchmark", "percent": 50},
        )
        self.assertEqual(status, 400)
        self.assertEqual(rejected["error"]["code"], "invalid_request")

    def test_server_rejects_non_loopback_binding(self) -> None:
        with self.assertRaisesRegex(ConfigError, "loopback"):
            create_local_api_server(self.config_path, "0.0.0.0", 8765)


if __name__ == "__main__":
    unittest.main()
