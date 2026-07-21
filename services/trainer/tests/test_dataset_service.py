from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE_ROOT / "src"))

from autotrainer.config import ConfigError, default_config, write_config  # noqa: E402
from autotrainer.dataset_service import (  # noqa: E402
    DatasetDesignError,
    _LANGUAGE_SUFFIXES,
    design_dataset_candidate,
    freeze_dataset,
    get_dataset_workspace,
    require_frozen_dataset,
)
from autotrainer.history import EXTRACTOR_VERSION, SUPPORTED_SUFFIXES  # noqa: E402


def run_git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


class DatasetServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.config_path = self.root / "autotrainer.yaml"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def write_local_history_project(self) -> None:
        repository = self.root / "workspace"
        repository.mkdir()
        run_git(repository, "init")
        run_git(repository, "config", "user.name", "AutoTrainer Tests")
        run_git(repository, "config", "user.email", "tests@example.invalid")
        (repository / "App.tsx").write_text(
            "export const label = 'Before';\n",
            encoding="utf-8",
        )
        run_git(repository, "add", ".")
        run_git(repository, "commit", "-m", "Initial fixture")
        (repository / "App.tsx").write_text(
            "export const label = 'After';\n",
            encoding="utf-8",
        )
        run_git(repository, "add", ".")
        run_git(repository, "commit", "-m", "Clarify the primary interface label")
        revision = run_git(repository, "rev-parse", "HEAD")
        config = default_config()
        config["sources"] = [
            {
                "id": "workspace",
                "kind": "repository",
                "license": {"spdx": "LicenseRef-Internal"},
                "partition": "train",
                "revision": revision,
                "roles": ["history"],
                "uri": "workspace",
            }
        ]
        write_config(self.config_path, config, overwrite=True)

    def test_airflow_python_pull_request_is_reviewable_and_classified(self) -> None:
        repository = self.root / ".autotrainer" / "sources" / "airflow"
        repository.mkdir(parents=True)
        run_git(repository, "init", "-q")
        run_git(repository, "config", "user.name", "AutoTrainer Tests")
        run_git(repository, "config", "user.email", "tests@example.invalid")
        run_git(
            repository,
            "remote",
            "add",
            "origin",
            "https://github.com/apache/airflow.git",
        )
        dags = repository / "dags"
        dags.mkdir()
        dag_path = dags / "etl_pipeline.py"
        dag_path.write_text(
            "def extract():\n"
            "    return 'raw records'\n",
            encoding="utf-8",
        )
        run_git(repository, "add", ".")
        run_git(repository, "commit", "-q", "-m", "Initial Airflow fixture")
        dag_path.write_text(
            "from airflow import DAG\n"
            "from airflow.operators.python import PythonOperator\n\n"
            "def extract():\n"
            "    return 'raw records'\n\n"
            "def transform():\n"
            "    return 'clean records'\n\n"
            "def load():\n"
            "    return 'loaded records'\n\n"
            "with DAG(dag_id='etl_pipeline', schedule=None) as dag:\n"
            "    extract_task = PythonOperator(task_id='extract', python_callable=extract)\n"
            "    transform_task = PythonOperator(task_id='transform', python_callable=transform)\n"
            "    load_task = PythonOperator(task_id='load', python_callable=load)\n"
            "    extract_task >> transform_task >> load_task\n",
            encoding="utf-8",
        )
        run_git(repository, "add", ".")
        run_git(
            repository,
            "commit",
            "-q",
            "-m",
            "Define the ETL pipeline task dependency order",
        )
        revision = run_git(repository, "rev-parse", "HEAD")

        config = default_config()
        config["sources"] = [
            {
                "id": "airflow",
                "kind": "repository",
                "license": {"spdx": "Apache-2.0"},
                "partition": "train",
                "revision": revision,
                "roles": ["history"],
                "uri": ".autotrainer/sources/airflow",
            }
        ]
        write_config(self.config_path, config, overwrite=True)
        catalog = self.root / ".autotrainer" / "dataset" / "github-prs" / "airflow.json"
        catalog.parent.mkdir(parents=True)
        catalog.write_text(
            json.dumps(
                {
                    "license_spdx": "Apache-2.0",
                    "pull_requests": [
                        {
                            "base_branch": "main",
                            "body": "",
                            "merge_commit": revision,
                            "merged_at": "2026-07-20T12:00:00Z",
                            "number": 42,
                            "title": "Define the ETL pipeline task dependency order",
                        }
                    ],
                    "repository": "apache/airflow",
                    "schema_version": 1,
                    "source_id": "airflow",
                    "source_revision": revision,
                }
            )
            + "\n",
            encoding="utf-8",
        )

        workspace = get_dataset_workspace(self.config_path)

        self.assertEqual(workspace["catalog"]["status"], "ready")
        self.assertEqual(workspace["summary"]["reviewable_count"], 1)
        candidate = workspace["candidates"][0]
        self.assertEqual(candidate["languages"], ["python"])
        self.assertEqual(candidate["pull_request"]["number"], 42)
        self.assertEqual(candidate["files"][0]["path"], "dags/etl_pipeline.py")
        self.assertIn("extract_task >> transform_task >> load_task", candidate["patch"])

    def test_history_supports_every_declared_v1_language_extension(self) -> None:
        self.assertLessEqual(2, EXTRACTOR_VERSION)
        self.assertTrue(set(_LANGUAGE_SUFFIXES).issubset(SUPPORTED_SUFFIXES))

    def test_selected_llm_design_is_local_reviewable_metadata(self) -> None:
        self.write_local_history_project()
        workspace = get_dataset_workspace(self.config_path)
        candidate_id = workspace["candidates"][0]["candidate_id"]
        completion = json.dumps(
            {
                "grpo_task": None,
                "instruction": "Make the primary interface label clearer for users.",
                "language": "typescript_react",
                "reason": "The accepted patch is a focused supervised code response.",
                "recommended_method": "qlora",
            }
        )

        with patch(
            "autotrainer.dataset_service._designer_response",
            return_value=completion,
        ) as designer:
            result = design_dataset_candidate(
                self.config_path,
                candidate_id=candidate_id,
                provider="local",
                model="qwen-local",
            )

        designer.assert_called_once()
        design = result["candidates"][0]["design"]
        self.assertEqual(design["recommended_method"], "qlora")
        self.assertEqual(design["language"], "typescript_react")
        self.assertEqual(design["designer"], {"model": "qwen-local", "provider": "local"})
        stored = (
            self.root / ".autotrainer" / "dataset" / "designs.json"
        ).read_text(encoding="utf-8")
        self.assertNotIn("endpoint", stored)

    def test_invalid_llm_design_cannot_enter_the_review_queue(self) -> None:
        self.write_local_history_project()
        candidate_id = get_dataset_workspace(self.config_path)["candidates"][0][
            "candidate_id"
        ]
        with patch(
            "autotrainer.dataset_service._designer_response",
            return_value=json.dumps(
                {
                    "grpo_task": None,
                    "instruction": "Make the primary interface label clearer for users.",
                    "language": "ruby",
                    "reason": "Unsupported.",
                    "recommended_method": "qlora",
                }
            ),
        ), self.assertRaisesRegex(DatasetDesignError, "unsupported language"):
            design_dataset_candidate(
                self.config_path,
                candidate_id=candidate_id,
                provider="anthropic",
                model="claude-haiku-4-5",
            )

    def test_freeze_binds_compiled_local_data_and_detects_changes(self) -> None:
        examples = self.root / "accepted.jsonl"
        examples.write_text(
            json.dumps(
                {
                    "prompt": [{"content": "Write a greeting.", "role": "user"}],
                    "completion": [{"content": "Hello", "role": "assistant"}],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        config = default_config()
        config["sources"] = [
            {
                "id": "accepted",
                "kind": "sft_jsonl",
                "license": {"spdx": "LicenseRef-Internal"},
                "partition": "train",
                "roles": ["demonstrations"],
                "uri": "accepted.jsonl",
            }
        ]
        write_config(self.config_path, config, overwrite=True)
        current_plan = (
            self.root / ".autotrainer" / "evaluation" / "current-plan.json"
        )
        current_plan.parent.mkdir(parents=True)
        current_plan.write_text(
            json.dumps({"path": "old", "plan_id": "old"}) + "\n",
            encoding="utf-8",
        )

        frozen = freeze_dataset(self.config_path)

        self.assertEqual(frozen["freeze"]["status"], "ready")
        self.assertEqual(frozen["freeze"]["receipt"]["counts"]["sft_train"], 1)
        self.assertTrue(require_frozen_dataset(self.config_path))
        self.assertFalse(current_plan.exists())

        examples.write_text(examples.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        self.assertEqual(get_dataset_workspace(self.config_path)["freeze"]["status"], "stale")
        with self.assertRaisesRegex(ConfigError, "changed after it was frozen"):
            require_frozen_dataset(self.config_path)

    def test_pending_reviews_must_be_resolved_before_freeze(self) -> None:
        self.write_local_history_project()

        with self.assertRaisesRegex(ConfigError, "approve or reject every"):
            freeze_dataset(self.config_path)

    def test_freeze_detects_compiled_artifact_tampering(self) -> None:
        examples = self.root / "accepted.jsonl"
        examples.write_text(
            json.dumps(
                {
                    "prompt": [{"content": "Write a greeting.", "role": "user"}],
                    "completion": [{"content": "Hello", "role": "assistant"}],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        config = default_config()
        config["sources"] = [
            {
                "id": "accepted",
                "kind": "sft_jsonl",
                "license": {"spdx": "LicenseRef-Internal"},
                "partition": "train",
                "roles": ["demonstrations"],
                "uri": "accepted.jsonl",
            }
        ]
        write_config(self.config_path, config, overwrite=True)
        freeze_dataset(self.config_path)

        compiled = self.root / config["sft"]["dataset"]
        compiled.write_text(
            compiled.read_text(encoding="utf-8") + "\n",
            encoding="utf-8",
        )

        status = get_dataset_workspace(self.config_path)["freeze"]
        self.assertEqual(status["status"], "stale")
        self.assertEqual(status["reason"], "compiled_artifacts_changed")
        with self.assertRaisesRegex(ConfigError, "compiled dataset bytes changed"):
            require_frozen_dataset(self.config_path)


if __name__ == "__main__":
    unittest.main()
