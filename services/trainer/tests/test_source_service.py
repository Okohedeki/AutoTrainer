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

from autotrainer.config import ConfigError, default_config, load_config, write_config  # noqa: E402
from autotrainer.source_service import add_source, list_sources, remove_source  # noqa: E402


def run_git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


class SourceServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.config_path = self.root / "autotrainer.yaml"
        write_config(self.config_path, default_config(), overwrite=False)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def create_repository(self, name: str = "workspace") -> tuple[Path, str]:
        repository = self.root / name
        repository.mkdir()
        (repository / "app.py").write_text("print('ready')\n", encoding="utf-8")
        run_git(repository, "init")
        run_git(repository, "config", "user.name", "AutoTrainer Tests")
        run_git(repository, "config", "user.email", "tests@example.invalid")
        run_git(repository, "add", ".")
        run_git(repository, "commit", "-m", "fixture")
        return repository, run_git(repository, "rev-parse", "HEAD")

    def test_infers_local_git_before_task_files_and_rejects_duplicate_identity(self) -> None:
        repository, commit = self.create_repository()
        (repository / "task.json").write_text(
            json.dumps({"task": {"split": "train"}}), encoding="utf-8"
        )

        result = add_source(self.config_path, str(repository))

        self.assertEqual(result["source"]["kind"], "repository")
        self.assertEqual(result["source"]["origin"], "local")
        self.assertEqual(result["source"]["purpose"], "work")
        self.assertEqual(result["source"]["revision"], commit)
        with self.assertRaisesRegex(ConfigError, "already added"):
            add_source(self.config_path, str(repository / "."))

    def test_infers_demonstrations_and_task_partition(self) -> None:
        demonstrations = self.root / "accepted.jsonl"
        demonstrations.write_text(
            json.dumps({"messages": [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]})
            + "\n",
            encoding="utf-8",
        )
        tasks = self.root / "held-out-tasks"
        tasks.mkdir()
        (tasks / "task.json").write_text(
            json.dumps({"task": {"split": "evaluation"}}), encoding="utf-8"
        )

        examples_result = add_source(self.config_path, str(demonstrations))
        tasks_result = add_source(self.config_path, str(tasks))

        self.assertEqual(examples_result["source"]["purpose"], "examples")
        self.assertEqual(examples_result["source"]["kind"], "sft_jsonl")
        self.assertEqual(tasks_result["source"]["purpose"], "tasks")
        declared = load_config(self.config_path).sources
        task_source = next(source for source in declared if source["id"] == tasks_result["source"]["id"])
        self.assertEqual(task_source["partition"], "evaluation")
        self.assertEqual(task_source["roles"], ["evaluation"])

    def test_stable_ids_disambiguate_same_local_label(self) -> None:
        first = self.root / "first"
        second = self.root / "second"
        first.mkdir()
        second.mkdir()
        for directory in (first, second):
            (directory / "accepted.jsonl").write_text(
                '{"messages":[{"role":"assistant","content":"ok"}]}\n',
                encoding="utf-8",
            )

        first_id = add_source(self.config_path, str(first / "accepted.jsonl"))["source"]["id"]
        second_id = add_source(self.config_path, str(second / "accepted.jsonl"))["source"]["id"]

        self.assertEqual(first_id, "accepted")
        self.assertRegex(second_id, r"^accepted-[0-9a-f]{8}$")

    def test_github_shorthand_is_normalized_materialized_and_safely_removed(self) -> None:
        def fake_materialize(config: dict, root: Path, source_id: str) -> dict:
            local = root / ".autotrainer" / "sources" / source_id
            local.mkdir(parents=True)
            run_git(local, "init")
            run_git(local, "remote", "add", "origin", "https://github.com/Owner/Repo.git")
            updated = dict(config["sources"][-1])
            updated["uri"] = f".autotrainer/sources/{source_id}"
            updated["revision"] = "a" * 40
            return {
                "local_path": str(local),
                "updated_source": updated,
                "commit": "a" * 40,
            }

        with patch("autotrainer.source_service.materialize_repository", side_effect=fake_materialize):
            result = add_source(self.config_path, "github.com/Owner/Repo")

        source = result["source"]
        self.assertEqual(source["origin"], "github")
        self.assertEqual(source["value"], "https://github.com/Owner/Repo.git")
        self.assertEqual(source["label"], "Owner/Repo")
        managed = self.root / ".autotrainer" / "sources" / source["id"]
        self.assertTrue(managed.is_dir())

        removed = remove_source(self.config_path, source["id"])

        self.assertEqual(removed["removed"], source)
        self.assertEqual(removed["sources"], [])
        self.assertFalse(managed.exists())

    def test_failed_github_materialization_does_not_change_config(self) -> None:
        before = self.config_path.read_bytes()
        with patch(
            "autotrainer.source_service.materialize_repository",
            side_effect=RuntimeError("clone failed"),
        ):
            with self.assertRaisesRegex(ConfigError, "could not clone and pin"):
                add_source(self.config_path, "owner/repository")

        self.assertEqual(self.config_path.read_bytes(), before)
        self.assertEqual(list_sources(self.config_path), [])


if __name__ == "__main__":
    unittest.main()
