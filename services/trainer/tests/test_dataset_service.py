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
    design_dataset_candidate,
    freeze_dataset,
    get_dataset_workspace,
    require_frozen_dataset,
)


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


if __name__ == "__main__":
    unittest.main()
