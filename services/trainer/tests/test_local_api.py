from __future__ import annotations

from http.client import HTTPConnection
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE_ROOT / "src"))

from autotrainer.config import ConfigError, default_config, load_config, write_config  # noqa: E402
from autotrainer.github_service import GitHubSearchError  # noqa: E402
from autotrainer.local_api import create_local_api_server  # noqa: E402
from autotrainer.model_service import ModelSearchError  # noqa: E402
from autotrainer.project_gate import (  # noqa: E402
    acquire_project_lease,
    ProjectBusyError,
    project_run_gate,
)


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

    def request(
        self,
        method: str,
        path: str,
        payload: dict[str, object] | None = None,
        *,
        headers: dict[str, str] | None = None,
        include_content_type: bool = True,
    ) -> tuple[int, dict[str, object]]:
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        request_headers = dict(headers or {})
        if body is not None and include_content_type:
            request_headers.setdefault("Content-Type", "application/json")
        self.connection.request(method, path, body=body, headers=request_headers)
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
        self.assertEqual(current["download_job"]["status"], "idle")

    def test_projects_are_created_active_and_can_be_selected_without_browser_paths(self) -> None:
        status, initial = self.request("GET", "/api/v1/projects")
        self.assertEqual(status, 200)
        self.assertEqual(initial["active_id"], "startup")
        self.assertEqual([item["id"] for item in initial["projects"]], ["startup"])

        old_managers = (
            self.server.model_download,
            self.server.training,
            self.server.evaluation,
            self.server.hosting,
        )
        status, created = self.request(
            "POST",
            "/api/v1/projects",
            {"name": "Frontend Specialist"},
        )
        self.assertEqual(status, 201)
        self.assertEqual(created["id"], "frontend-specialist")
        self.assertTrue(created["active"])
        selected_config = Path(created["config_path"]).resolve()
        self.assertEqual(self.server.config_path, selected_config)
        self.assertEqual(self.server.model_download._config_path, selected_config)
        self.assertEqual(self.server.training._config_path, selected_config)
        self.assertEqual(self.server.evaluation._config_path, selected_config)
        self.assertEqual(self.server.hosting.config_path, selected_config)
        for old, current in zip(
            old_managers,
            (
                self.server.model_download,
                self.server.training,
                self.server.evaluation,
                self.server.hosting,
            ),
        ):
            self.assertIsNot(old, current)

        status, health = self.request("GET", "/api/v1/health")
        self.assertEqual(status, 200)
        self.assertEqual(health["active_project"]["id"], created["id"])
        self.assertEqual(Path(health["config"]), selected_config)

        status, listed = self.request("GET", "/api/v1/projects")
        self.assertEqual(status, 200)
        self.assertEqual(listed["active_id"], created["id"])

        status, selected = self.request(
            "POST",
            "/api/v1/projects/select",
            {"project_id": "startup"},
        )
        self.assertEqual(status, 200)
        self.assertTrue(selected["active"])
        self.assertEqual(self.server.workspace.active_id, "startup")

    def test_project_creation_rejects_an_active_project_lease_without_a_ghost(self) -> None:
        with project_run_gate(self.config_path):
            status, result = self.request(
                "POST",
                "/api/v1/projects",
                {"name": "Second Project"},
            )

        self.assertEqual(status, 409)
        self.assertEqual(result["error"]["code"], "project_busy")
        self.assertEqual(self.server.workspace.active_id, "startup")
        self.assertEqual(
            [item["id"] for item in self.server.workspace.list_projects()["projects"]],
            ["startup"],
        )

    def test_project_creation_rolls_back_when_the_target_lease_is_unavailable(self) -> None:
        marker = (
            self.server.workspace.projects_root
            / "lease-contended"
            / "keep.txt"
        )

        def acquire_or_reject(path: str | Path):
            if Path(path).resolve() == self.config_path.resolve():
                return acquire_project_lease(path)
            # Simulate an unexpected file appearing after creation. Rollback
            # must hide the config without recursively deleting that file.
            marker.write_text("preserve me", encoding="utf-8")
            raise ProjectBusyError("target project is busy")

        with patch(
            "autotrainer.local_api.acquire_project_lease",
            side_effect=acquire_or_reject,
        ):
            status, result = self.request(
                "POST",
                "/api/v1/projects",
                {"name": "Lease Contended"},
            )

        self.assertEqual(status, 409)
        self.assertEqual(result["error"]["code"], "project_busy")
        self.assertTrue(marker.is_file())
        self.assertFalse((marker.parent / "autotrainer.yaml").exists())
        self.assertEqual(
            [item["id"] for item in self.server.workspace.list_projects()["projects"]],
            ["startup"],
        )

    def test_project_creation_rolls_back_after_partial_manager_construction(self) -> None:
        original_managers = (
            self.server.model_download,
            self.server.training,
            self.server.evaluation,
            self.server.hosting,
        )
        partial_download = Mock()
        with (
            patch(
                "autotrainer.local_api.ModelDownloadManager",
                return_value=partial_download,
            ),
            patch(
                "autotrainer.local_api.TrainingJobManager",
                side_effect=RuntimeError("manager setup failed"),
            ),
        ):
            status, result = self.request(
                "POST",
                "/api/v1/projects",
                {"name": "Broken Context"},
            )

        self.assertEqual(status, 500)
        self.assertEqual(result["error"]["code"], "internal_error")
        partial_download.close.assert_called_once_with()
        self.assertEqual(self.server.workspace.active_id, "startup")
        self.assertEqual(
            (
                self.server.model_download,
                self.server.training,
                self.server.evaluation,
                self.server.hosting,
            ),
            original_managers,
        )
        self.assertEqual(
            [item["id"] for item in self.server.workspace.list_projects()["projects"]],
            ["startup"],
        )

    def test_server_uses_a_safe_default_or_explicit_projects_root(self) -> None:
        self.assertEqual(
            self.server.workspace.projects_root,
            (self.config_path.parent / ".autotrainer" / "projects").resolve(),
        )
        explicit = self.root / "bounded-projects"
        second = create_local_api_server(
            self.config_path,
            "127.0.0.1",
            0,
            projects_root=explicit,
        )
        try:
            self.assertEqual(second.workspace.projects_root, explicit.resolve())
        finally:
            second.server_close()

    def test_hugging_face_search_is_bounded_and_sanitizes_upstream_errors(self) -> None:
        models = [
            {
                "id": "example/model",
                "revision": "a" * 40,
                "compatibility": "unverified",
            }
        ]
        with patch(
            "autotrainer.local_api.search_models",
            return_value=models,
        ) as search:
            status, result = self.request(
                "GET",
                "/api/v1/models/search?q=qwen&limit=5",
            )
        self.assertEqual(status, 200)
        self.assertEqual(result, {"models": models})
        search.assert_called_once_with("qwen", limit=5)

        status, result = self.request(
            "GET",
            "/api/v1/models/search?q=qwen&limit=5&limit=6",
        )
        self.assertEqual(status, 400)
        self.assertEqual(result["error"]["code"], "invalid_request")

        with patch(
            "autotrainer.local_api.search_models",
            side_effect=ModelSearchError("token=hf_supersecretvalue upstream failed"),
        ):
            status, result = self.request(
                "GET",
                "/api/v1/models/search?q=qwen",
            )
        self.assertEqual(status, 503)
        self.assertNotIn("hf_supersecretvalue", result["error"]["message"])

    def test_local_model_discovery_and_adoption_use_path_free_contracts(self) -> None:
        candidate_id = "a" * 64
        workspace = {
            "models": [
                {
                    "candidate_id": candidate_id,
                    "catalog_key": "qwen3.5-9b-text",
                    "model_id": "Qwen/Qwen3.5-9B",
                    "revision": "b" * 40,
                    "availability": "available",
                    "selected": False,
                    "source": "huggingface_cache",
                    "cache_label": "Hugging Face cache",
                    "file_count": 6,
                    "logical_bytes": 10,
                }
            ],
            "scanned_cache_count": 2,
            "ignored_incomplete_count": 1,
        }
        with patch(
            "autotrainer.local_api.discover_local_models",
            return_value=workspace,
        ) as discover:
            status, result = self.request("GET", "/api/v1/models/local")
        self.assertEqual(status, 200)
        self.assertEqual(result, workspace)
        self.assertNotIn("cache_dir", json.dumps(result))
        self.assertNotIn("snapshot_path", json.dumps(result))
        discover.assert_called_once_with(self.config_path.resolve())

        adopted = {
            "model": {"id": "Qwen/Qwen3.5-9B", "revision": "b" * 40},
            "catalog_key": "qwen3.5-9b-text",
        }
        cache = {"status": "downloaded", "model_id": "Qwen/Qwen3.5-9B"}
        with (
            patch(
                "autotrainer.local_api.use_local_model",
                return_value=adopted,
            ) as use_local,
            patch("autotrainer.local_api.model_status", return_value=cache),
        ):
            status, result = self.request(
                "POST",
                "/api/v1/model/use-local",
                {"candidate_id": candidate_id},
            )
        self.assertEqual(status, 200)
        self.assertEqual(result, {**adopted, "cache": cache})
        use_local.assert_called_once_with(self.config_path.resolve(), candidate_id)

        with patch("autotrainer.local_api.use_local_model") as use_local:
            status, result = self.request(
                "POST",
                "/api/v1/model/use-local",
                {"candidate_id": candidate_id, "cache_dir": "C:/private"},
            )
        self.assertEqual(status, 400)
        self.assertEqual(result["error"]["code"], "invalid_request")
        use_local.assert_not_called()

    def test_github_repository_search_is_bounded_and_sanitized(self) -> None:
        repositories = [
            {
                "full_name": "apache/airflow",
                "clone_url": "https://github.com/apache/airflow.git",
                "stars": 42000,
            }
        ]
        with patch(
            "autotrainer.local_api.search_repositories",
            return_value=repositories,
        ) as search:
            status, result = self.request(
                "GET",
                "/api/v1/repositories/search?q=airflow&limit=5",
            )
        self.assertEqual(status, 200)
        self.assertEqual(result, {"repositories": repositories})
        search.assert_called_once_with("airflow", limit=5)

        status, result = self.request(
            "GET",
            "/api/v1/repositories/search?q=airflow&limit=5&limit=6",
        )
        self.assertEqual(status, 400)
        self.assertEqual(result["error"]["code"], "invalid_request")

        with patch(
            "autotrainer.local_api.search_repositories",
            side_effect=GitHubSearchError("token=github_supersecretvalue upstream failed"),
        ):
            status, result = self.request(
                "GET",
                "/api/v1/repositories/search?q=airflow",
            )
        self.assertEqual(status, 503)
        self.assertEqual(result["error"]["code"], "repository_search_unavailable")
        self.assertNotIn("github_supersecretvalue", result["error"]["message"])

    def test_model_selection_persists_the_same_yaml_used_by_cli(self) -> None:
        original_cache = load_config(self.config_path).model["cache_dir"]
        status, result = self.request(
            "POST",
            "/api/v1/model/select",
            {"model": "qwen3.5-9b-text"},
        )
        self.assertEqual(status, 200)
        self.assertEqual(result["model"]["cache_dir"], original_cache)
        selected = load_config(self.config_path).model
        self.assertEqual(selected["revision"], "c202236235762e1c871ad0ccb60c8ee5ba337b9a")

        status, rejected = self.request(
            "POST",
            "/api/v1/model/select",
            {"model": "qwen3.5-9b-text", "cache_dir": "D:/models"},
        )
        self.assertEqual(status, 400)
        self.assertEqual(rejected["error"]["code"], "invalid_request")

    def test_download_endpoint_queues_the_server_owned_background_manager(self) -> None:
        queued = {
            "id": "a" * 32,
            "status": "queued",
            "message": "The exact Hugging Face snapshot is queued for download.",
            "model_id": "Qwen/Qwen3.5-9B",
            "revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
            "result": None,
            "kind": "project",
        }
        with patch.object(self.server.model_download, "start", return_value=queued) as start:
            status, result = self.request("POST", "/api/v1/model/download", {})
        self.assertEqual(status, 202)
        self.assertEqual(result, queued)
        start.assert_called_once_with()

        with patch.object(
            self.server.model_download,
            "snapshot",
            return_value=queued,
        ):
            status, result = self.request("GET", "/api/v1/model/status")
        self.assertEqual(status, 200)
        self.assertEqual(result["download_job"], queued)

    def test_fixed_reference_model_has_a_separate_inspection_and_download_route(self) -> None:
        reference = {
            "alias": "qwythos-9b-reference",
            "model_id": "empero-ai/Qwythos-9B-Claude-Mythos-5-1M",
            "revision": "b" * 40,
            "status": "not_downloaded",
            "snapshot_path": None,
            "cache_dir": str(self.root / "model-cache"),
            "receipt": None,
        }
        queued = {
            "id": "c" * 32,
            "status": "queued",
            "message": "The exact Hugging Face snapshot is queued for download.",
            "model_id": reference["model_id"],
            "revision": reference["revision"],
            "result": None,
            "kind": "reference",
        }
        with (
            patch(
                "autotrainer.local_api.inspect_reference_model",
                return_value=reference,
            ) as inspect,
            patch.object(
                self.server.model_download,
                "snapshot",
                return_value=queued,
            ),
        ):
            status, result = self.request("GET", "/api/v1/reference-model")
        self.assertEqual(status, 200)
        self.assertEqual(result, {**reference, "download_job": queued})
        inspect.assert_called_once_with(self.config_path.resolve())

        with patch.object(
            self.server.model_download,
            "start_reference",
            return_value=queued,
        ) as start_reference:
            status, result = self.request(
                "POST",
                "/api/v1/reference-model/download",
                {},
            )
        self.assertEqual(status, 202)
        self.assertEqual(result, queued)
        start_reference.assert_called_once_with()

        # A reference transfer never masquerades as the selected base-model
        # transfer in the neighboring model panel.
        with patch.object(
            self.server.model_download,
            "snapshot",
            return_value=queued,
        ):
            status, result = self.request("GET", "/api/v1/model/status")
        self.assertEqual(status, 200)
        self.assertIsNone(result["download_job"])

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
        self.assertTrue(
            {
                "id",
                "kind",
                "label",
                "value",
                "origin",
                "purpose",
                "modes",
                "partition",
                "roles",
                "filters",
                "license",
                "status",
                "next_action",
            }
            <= set(source)
        )
        self.assertEqual(source["purpose"], "examples")
        self.assertEqual(source["status"], "configured")
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

    def test_source_endpoint_passes_explicit_repository_learning_scope(self) -> None:
        response = {"source": {"id": "repo"}, "sources": [{"id": "repo"}]}
        with patch(
            "autotrainer.local_api.add_source",
            return_value=response,
        ) as add:
            status, result = self.request(
                "POST",
                "/api/v1/sources",
                {
                    "value": "owner/repository",
                    "modes": ["accepted_changes", "practice_tasks"],
                    "revision": "a" * 40,
                    "include": ["src/**"],
                    "exclude": ["vendor/**"],
                    "license_spdx": "MIT",
                    "license_attribution": "Example authors",
                },
            )

        self.assertEqual(status, 200)
        self.assertEqual(result, response)
        add.assert_called_once_with(
            self.config_path.resolve(),
            "owner/repository",
            modes=["accepted_changes", "practice_tasks"],
            require_modes=True,
            revision="a" * 40,
            include=["src/**"],
            exclude=["vendor/**"],
            license_spdx="MIT",
            license_attribution="Example authors",
        )

    def test_repository_source_requires_modes_and_rejects_unsupported_fields(self) -> None:
        status, result = self.request(
            "POST",
            "/api/v1/sources",
            {"value": "owner/repository"},
        )
        self.assertEqual(status, 400)
        self.assertIn("mode", result["error"]["message"])

        status, result = self.request(
            "POST",
            "/api/v1/sources",
            {
                "value": "owner/repository",
                "modes": ["reference_only"],
                "kind": "repository",
            },
        )
        self.assertEqual(status, 400)
        self.assertEqual(result["error"]["code"], "invalid_request")

    def test_task_endpoints_share_guided_authoring_contract(self) -> None:
        workspace = {
            "tasks": [
                {
                    "id": "repair-form",
                    "split": "train",
                    "status": "declared",
                }
            ]
        }
        with patch(
            "autotrainer.local_api.list_authored_tasks",
            return_value=workspace,
        ) as listed:
            status, result = self.request("GET", "/api/v1/tasks")
        self.assertEqual(status, 200)
        self.assertEqual(result, workspace)
        listed.assert_called_once_with(self.config_path.resolve())

        with patch(
            "autotrainer.local_api.create_authored_task",
            return_value=workspace,
        ) as create:
            status, result = self.request(
                "POST",
                "/api/v1/tasks",
                {
                    "source_id": "workspace",
                    "instruction": "Repair the form while preserving existing submit behavior.",
                    "working_directory": "app",
                    "build": "npm run build",
                    "tests": "npm test",
                    "verifier_bundle": "C:\\hidden\\form-verifier",
                    "verifier_command": "node /autotrainer-verifier/verify.mjs",
                },
            )
        self.assertEqual(status, 201)
        self.assertEqual(result, workspace)
        create.assert_called_once_with(
            self.config_path.resolve(),
            source_id="workspace",
            instruction="Repair the form while preserving existing submit behavior.",
            working_directory="app",
            install=None,
            build="npm run build",
            tests="npm test",
            browser_tests=None,
            verifier_bundle="C:\\hidden\\form-verifier",
            verifier_command="node /autotrainer-verifier/verify.mjs",
            verifier_report_path=".autotrainer-verifier-report.json",
            task_id=None,
            group_id=None,
        )

        empty = {"removed": workspace["tasks"][0], "tasks": []}
        with patch(
            "autotrainer.local_api.remove_authored_task",
            return_value=empty,
        ) as remove:
            status, result = self.request(
                "DELETE",
                "/api/v1/tasks/train/repair-form",
            )
        self.assertEqual(status, 200)
        self.assertEqual(result, empty)
        remove.assert_called_once_with(
            self.config_path.resolve(),
            split="train",
            task_id="repair-form",
        )

    def test_task_endpoint_rejects_missing_runtime_and_verifier_fields(self) -> None:
        status, result = self.request(
            "POST",
            "/api/v1/tasks",
            {
                "source_id": "workspace",
                "instruction": "Repair one concrete behavior with an executable check.",
                "working_directory": ".",
            },
        )

        self.assertEqual(status, 400)
        self.assertEqual(result["error"]["code"], "invalid_request")

    def test_example_endpoints_share_guided_authoring_contract(self) -> None:
        workspace = {
            "examples": [
                {
                    "id": "repair-form-response",
                    "source_id": "workspace",
                    "status": "declared",
                }
            ]
        }
        with patch(
            "autotrainer.local_api.list_authored_examples",
            return_value=workspace,
        ) as listed:
            status, result = self.request("GET", "/api/v1/examples")
        self.assertEqual(status, 200)
        self.assertEqual(result, workspace)
        listed.assert_called_once_with(self.config_path.resolve())

        with patch(
            "autotrainer.local_api.create_authored_example",
            return_value=workspace,
        ) as create:
            status, result = self.request(
                "POST",
                "/api/v1/examples",
                {
                    "source_id": "workspace",
                    "instruction": "Repair the form while preserving submit behavior.",
                    "accepted_response": (
                        "Implemented the validation and retained the existing submit path."
                    ),
                    "rights_confirmed": True,
                },
            )
        self.assertEqual(status, 201)
        self.assertEqual(result, workspace)
        create.assert_called_once_with(
            self.config_path.resolve(),
            source_id="workspace",
            instruction="Repair the form while preserving submit behavior.",
            accepted_response=(
                "Implemented the validation and retained the existing submit path."
            ),
            rights_confirmed=True,
            example_id=None,
        )

        empty = {"removed": workspace["examples"][0], "examples": []}
        with patch(
            "autotrainer.local_api.remove_authored_example",
            return_value=empty,
        ) as remove:
            status, result = self.request(
                "DELETE",
                "/api/v1/examples/repair-form-response",
            )
        self.assertEqual(status, 200)
        self.assertEqual(result, empty)
        remove.assert_called_once_with(
            self.config_path.resolve(),
            example_id="repair-form-response",
        )

    def test_example_endpoint_requires_boolean_rights_confirmation(self) -> None:
        status, result = self.request(
            "POST",
            "/api/v1/examples",
            {
                "source_id": "workspace",
                "instruction": "Repair the form while preserving submit behavior.",
                "accepted_response": (
                    "Implemented the validation and retained the existing submit path."
                ),
                "rights_confirmed": "yes",
            },
        )

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

    def test_training_events_accept_only_one_nonnegative_cursor(self) -> None:
        page = {
            "job_id": "a" * 32,
            "cursor": 7,
            "events": [{"sequence": 7, "type": "rubric_scored"}],
            "truncated": False,
            "has_more": False,
        }
        with patch.object(self.server.training, "events", return_value=page) as events:
            status, result = self.request(
                "GET",
                "/api/v1/training/events?after=6",
            )
        self.assertEqual(status, 200)
        self.assertEqual(result, page)
        events.assert_called_once_with(6)

        status, result = self.request(
            "GET",
            "/api/v1/training/events?after=-1",
        )
        self.assertEqual(status, 400)
        self.assertEqual(result["error"]["code"], "invalid_request")

    def test_curriculum_uses_the_server_owned_validated_event_window(self) -> None:
        activity = {
            "job_id": "a" * 32,
            "status": "running",
            "stage": "grpo",
            "events": [],
            "window": {"scope": "current_job_retained_window"},
        }
        workspace = {"schema_version": 1, "status": "ready"}
        with (
            patch.object(
                self.server.training,
                "rollout_snapshot",
                return_value=activity,
            ) as snapshot,
            patch(
                "autotrainer.local_api.get_curriculum_workspace",
                return_value=workspace,
            ) as curriculum,
        ):
            status, result = self.request("GET", "/api/v1/curriculum")

        self.assertEqual(status, 200)
        self.assertEqual(result, workspace)
        snapshot.assert_called_once_with()
        curriculum.assert_called_once_with(self.config_path.resolve(), activity=activity)

        status, rejected = self.request("GET", "/api/v1/curriculum?view=fake")
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

    def test_evaluation_start_can_plan_and_choose_the_first_local_suite(self) -> None:
        planned = {
            "readiness": {"status": "ready", "blockers": []},
            "plan": {"plan_id": "sha256:" + "a" * 64},
            "job": {"status": "idle", "phase": "idle"},
            "suites": [
                {"id": "fable_ab", "runner_type": "external"},
                {"id": "model_benchmark", "runner_type": "builtin"},
                {"id": "secondary", "runner_type": "command"},
            ],
        }
        queued = {
            "id": "b" * 32,
            "status": "queued",
            "suite": "model_benchmark",
            "phase": "queued",
        }
        with (
            patch.object(self.server.evaluation, "plan", return_value=planned) as plan,
            patch.object(self.server.evaluation, "start", return_value=queued) as start,
        ):
            status, result = self.request(
                "POST",
                "/api/v1/evaluation/start",
                {},
            )

        self.assertEqual(status, 202)
        self.assertEqual(result, queued)
        plan.assert_called_once_with()
        start.assert_called_once_with("model_benchmark")

    def test_evaluation_events_accept_a_reconnect_cursor(self) -> None:
        page = {
            "events": [
                {
                    "sequence": 4,
                    "phase": "trial_scored",
                    "result": {"reward": 0.8},
                }
            ],
            "oldest_sequence": 1,
            "latest_sequence": 4,
            "cursor_reset": False,
        }
        with patch.object(self.server.evaluation, "events", return_value=page) as events:
            status, result = self.request(
                "GET",
                "/api/v1/evaluation/events?after=3",
            )
        self.assertEqual(status, 200)
        self.assertEqual(result, page)
        events.assert_called_once_with(3)

        status, result = self.request(
            "GET",
            "/api/v1/evaluation/events?after=3&unexpected=true",
        )
        self.assertEqual(status, 400)
        self.assertEqual(result["error"]["code"], "invalid_request")

    def test_hosting_endpoints_use_one_loopback_only_manager(self) -> None:
        ready = {
            "status": "ready",
            "message": "An adapter is ready to host.",
            "endpoint": None,
        }
        with patch.object(self.server.hosting, "snapshot", return_value=ready) as snapshot:
            status, result = self.request("GET", "/api/v1/hosting")
        self.assertEqual(status, 200)
        self.assertEqual(result, ready)
        snapshot.assert_called_once_with()

        loading = {
            "status": "loading",
            "message": "The model process is loading weights.",
            "endpoint": "http://127.0.0.1:9911",
        }
        with patch.object(self.server.hosting, "start", return_value=loading) as start:
            status, result = self.request(
                "POST",
                "/api/v1/hosting/start",
                {"adapter": "sft", "port": 9911},
            )
        self.assertEqual(status, 202)
        self.assertEqual(result, loading)
        start.assert_called_once_with(adapter="sft", host="127.0.0.1", port=9911)

        stopped = {
            "status": "stopped",
            "message": "The local model host is stopped.",
            "endpoint": None,
        }
        with patch.object(self.server.hosting, "stop", return_value=stopped) as stop:
            status, result = self.request(
                "POST",
                "/api/v1/hosting/stop",
                {},
            )
        self.assertEqual(status, 200)
        self.assertEqual(result, stopped)
        stop.assert_called_once_with()

        completed = {
            "status": "completed",
            "model": "frontend-specialist",
            "content": "A narrow implementation.",
            "usage": {},
        }
        with patch.object(self.server.hosting, "test", return_value=completed) as test:
            status, result = self.request(
                "POST",
                "/api/v1/hosting/test",
                {"prompt": "Build the page."},
            )
        self.assertEqual(status, 200)
        self.assertEqual(result, completed)
        test.assert_called_once_with("Build the page.")

    def test_mutations_require_json_and_reject_foreign_origins(self) -> None:
        with patch.object(self.server.training, "start") as start:
            status, result = self.request(
                "POST",
                "/api/v1/training/start",
                {},
                headers={"Origin": "https://attacker.example"},
            )
        self.assertEqual(status, 403)
        self.assertEqual(result["error"]["code"], "origin_denied")
        start.assert_not_called()

        status, result = self.request(
            "POST",
            "/api/v1/training/start",
            {},
            include_content_type=False,
        )
        self.assertEqual(status, 415)
        self.assertEqual(result["error"]["code"], "invalid_content_type")

        with patch("autotrainer.local_api.remove_source") as remove:
            status, result = self.request(
                "DELETE",
                "/api/v1/sources/example",
                headers={"Origin": "https://attacker.example"},
            )
        self.assertEqual(status, 403)
        self.assertEqual(result["error"]["code"], "origin_denied")
        remove.assert_not_called()

    def test_post_operations_reject_extra_payload_fields(self) -> None:
        status, result = self.request(
            "POST",
            "/api/v1/model/download",
            {"force": True},
        )
        self.assertEqual(status, 400)
        self.assertEqual(result["error"]["code"], "invalid_request")

    def test_server_rejects_non_loopback_binding(self) -> None:
        with self.assertRaisesRegex(ConfigError, "loopback"):
            create_local_api_server(self.config_path, "0.0.0.0", 8765)


if __name__ == "__main__":
    unittest.main()
