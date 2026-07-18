from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

import sys

SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE_ROOT / "src"))

from autotrainer.cli import main  # noqa: E402
from autotrainer.config import default_config, load_config, validate_mapping, write_config  # noqa: E402
from autotrainer.model_service import select_model  # noqa: E402


class ConfigTests(unittest.TestCase):
    def test_default_is_valid_and_explicit_about_qlora_to_grpo(self) -> None:
        payload = default_config()
        report = validate_mapping(payload)
        self.assertEqual(report.errors, ())
        self.assertEqual(payload["model"]["id"], "Qwen/Qwen3.5-9B")
        self.assertEqual(payload["model"]["quantization"]["quant_type"], "nf4")
        self.assertEqual(payload["grpo"]["start_from"], ".autotrainer/checkpoints/sft")

    def test_conditional_stage_recipes_validate_honestly(self) -> None:
        teach = default_config()
        teach["grpo"] = {"enabled": False}
        self.assertEqual(validate_mapping(teach).errors, ())

        practice = default_config()
        practice["sft"] = {"enabled": False}
        practice["grpo"]["start_from"] = "base"
        self.assertEqual(validate_mapping(practice).errors, ())

        both_from_base = default_config()
        both_from_base["grpo"]["start_from"] = "base"
        self.assertTrue(
            any("both-stage" in error for error in validate_mapping(both_from_base).errors)
        )

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
    def _json_command(self, arguments: list[str]) -> tuple[int, object]:
        output = io.StringIO()
        with redirect_stdout(output):
            status = main(arguments)
        return status, json.loads(output.getvalue())

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

    def test_curriculum_cli_uses_the_same_validated_activity_contract(self) -> None:
        activity = {"job_id": None, "status": "idle", "events": []}
        workspace = {"schema_version": 1, "status": "empty"}
        with (
            patch(
                "autotrainer.training_service.read_training_activity",
                return_value=activity,
            ) as read_activity,
            patch(
                "autotrainer.curriculum_service.get_curriculum_workspace",
                return_value=workspace,
            ) as curriculum,
        ):
            status, result = self._json_command(
                ["curriculum", "--config", "project.yaml", "--json"]
            )

        self.assertEqual(status, 0)
        self.assertEqual(result, workspace)
        read_activity.assert_called_once_with(Path("project.yaml"))
        curriculum.assert_called_once_with(Path("project.yaml"), activity=activity)

    def test_catalog_selection_uses_the_pinned_default_revision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "autotrainer.yaml"
            write_config(path, default_config(), overwrite=False)
            result = select_model(path, "qwen3.5-9b-text")
            self.assertEqual(
                result["model"]["revision"],
                "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
            )

    def test_agent_cli_uses_the_same_pinned_catalog_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "autotrainer.yaml"
            write_config(path, default_config(revision="main"), overwrite=False)
            self.assertEqual(
                main(["model", "use", "qwen3.5-9b-text", "--config", str(path)]),
                0,
            )
            self.assertEqual(
                load_config(path).model["revision"],
                "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
            )

    def test_reference_model_cannot_be_selected_as_the_training_base(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "autotrainer.yaml"
            write_config(path, default_config(), overwrite=False)
            with self.assertRaisesRegex(ValueError, "not a validated V1 training base"):
                select_model(path, "qwythos-9b-reference")

    def test_projects_commands_create_list_and_resolve_explicit_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "current" / "autotrainer.yaml"
            projects_root = root / "projects"
            projects_root.mkdir()
            write_config(
                config_path,
                default_config(name="Current"),
                overwrite=False,
            )
            common = [
                "--projects-root",
                str(projects_root),
                "--config",
                str(config_path),
                "--json",
            ]

            status, created = self._json_command(
                ["projects", "create", "CLI Project", *common]
            )
            self.assertEqual(status, 0)
            self.assertEqual(created["id"], "cli-project")

            status, listed = self._json_command(["projects", "list", *common])
            self.assertEqual(status, 0)
            self.assertEqual(
                [project["id"] for project in listed["projects"]],
                ["startup", "cli-project"],
            )

            status, selected = self._json_command(
                ["projects", "select", "cli-project", *common]
            )
            self.assertEqual(status, 0)
            self.assertTrue(selected["active"])
            self.assertEqual(
                Path(selected["config_path"]).resolve(),
                projects_root / "cli-project" / "autotrainer.yaml",
            )

    def test_models_search_uses_shared_service_and_json_contract(self) -> None:
        result = [{"id": "Qwen/Qwen3.5-9B", "compatibility": "supported"}]
        with patch(
            "autotrainer.model_service.search_models",
            return_value=result,
        ) as search:
            status, payload = self._json_command(
                ["models", "search", "qwen", "--limit", "5", "--json"]
            )

        self.assertEqual(status, 0)
        self.assertEqual(payload, {"models": result})
        search.assert_called_once_with("qwen", limit=5)

    def test_local_model_commands_share_discovery_and_adoption_services(self) -> None:
        config_path = Path("project/autotrainer.yaml")
        candidate_id = "a" * 64
        workspace = {
            "models": [{"candidate_id": candidate_id, "availability": "available"}],
            "scanned_cache_count": 1,
            "ignored_incomplete_count": 0,
        }
        adopted = {"model": {"id": "Qwen/Qwen3.5-9B"}}
        with patch(
            "autotrainer.model_service.discover_local_models",
            return_value=workspace,
        ) as discover:
            code, payload = self._json_command(
                ["models", "local", "--config", str(config_path), "--json"]
            )
        self.assertEqual(code, 0)
        self.assertEqual(payload, workspace)
        discover.assert_called_once_with(config_path)

        with patch(
            "autotrainer.model_service.use_local_model",
            return_value=adopted,
        ) as use_local:
            code, payload = self._json_command(
                [
                    "model",
                    "use-local",
                    candidate_id,
                    "--config",
                    str(config_path),
                    "--json",
                ]
            )
        self.assertEqual(code, 0)
        self.assertEqual(payload, adopted)
        use_local.assert_called_once_with(config_path, candidate_id)

    def test_reference_model_commands_share_the_pinned_cache_service(self) -> None:
        config_path = Path("project/autotrainer.yaml")
        status_result = {"status": "not_downloaded", "alias": "qwythos-9b-reference"}
        download_result = {"status": "downloaded", "alias": "qwythos-9b-reference"}
        with patch(
            "autotrainer.model_cache.inspect_reference_model",
            return_value=status_result,
        ) as status:
            code, payload = self._json_command(
                ["model", "reference-status", "--config", str(config_path), "--json"]
            )
        self.assertEqual(code, 0)
        self.assertEqual(payload, status_result)
        status.assert_called_once_with(config_path)

        with patch(
            "autotrainer.model_cache.materialize_reference_model",
            return_value=download_result,
        ) as download:
            code, payload = self._json_command(
                ["model", "reference-download", "--config", str(config_path), "--json"]
            )
        self.assertEqual(code, 0)
        self.assertEqual(payload, download_result)
        download.assert_called_once_with(config_path)

    def test_source_add_flattens_repeatable_intent_and_scope_flags(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "autotrainer.yaml"
            expected = {"source": {"id": "owner-repository"}, "sources": []}
            with patch(
                "autotrainer.source_service.add_source",
                return_value=expected,
            ) as add:
                status, payload = self._json_command(
                    [
                        "source",
                        "add",
                        "owner/repository",
                        "--mode",
                        "accepted_changes,practice_tasks",
                        "--include",
                        "src/**,tests/**",
                        "--include",
                        "scripts/**",
                        "--exclude",
                        "dist/**",
                        "--exclude",
                        "coverage/**,tmp/**",
                        "--license",
                        "Apache-2.0",
                        "--license-attribution",
                        "Copyright Example",
                        "--config",
                        str(config_path),
                        "--json",
                    ]
                )

        self.assertEqual(status, 0)
        self.assertEqual(payload, expected)
        add.assert_called_once_with(
            config_path,
            "owner/repository",
            name=None,
            kind=None,
            partition=None,
            roles=None,
            modes=["accepted_changes", "practice_tasks"],
            revision=None,
            include=["src/**", "tests/**", "scripts/**"],
            exclude=["dist/**", "coverage/**", "tmp/**"],
            license_spdx="Apache-2.0",
            license_attribution="Copyright Example",
        )

    def test_host_commands_delegate_without_blocking_on_model_load(self) -> None:
        config_path = Path("project/autotrainer.yaml")
        with patch("autotrainer.hosting_service.HostingManager") as manager_type:
            manager = manager_type.return_value
            manager.start.return_value = {"status": "loading", "endpoint": "http://127.0.0.1:9000"}
            status, started = self._json_command(
                [
                    "host",
                    "start",
                    "--adapter",
                    "grpo",
                    "--port",
                    "9000",
                    "--config",
                    str(config_path),
                    "--json",
                ]
            )
            self.assertEqual(status, 0)
            self.assertEqual(started["status"], "loading")
            manager.start.assert_called_once_with(
                adapter="grpo",
                host="127.0.0.1",
                port=9000,
            )

            manager.snapshot.return_value = {"status": "live"}
            status, snapshot = self._json_command(
                ["host", "status", "--config", str(config_path), "--json"]
            )
            self.assertEqual(status, 0)
            self.assertEqual(snapshot["status"], "live")

            manager.test.return_value = {"status": "completed", "content": "Hello"}
            status, tested = self._json_command(
                ["host", "test", "Hello?", "--config", str(config_path), "--json"]
            )
            self.assertEqual(status, 0)
            self.assertEqual(tested["content"], "Hello")
            manager.test.assert_called_once_with("Hello?")

            manager.stop.return_value = {"status": "stopped"}
            status, stopped = self._json_command(
                ["host", "stop", "--config", str(config_path), "--json"]
            )
            self.assertEqual(status, 0)
            self.assertEqual(stopped["status"], "stopped")

        self.assertEqual(manager_type.call_count, 4)
        manager_type.assert_called_with(config_path)


if __name__ == "__main__":
    unittest.main()
