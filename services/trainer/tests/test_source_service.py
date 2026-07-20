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
        self.assertEqual(
            result["source"]["modes"],
            ["accepted_changes", "reference_only"],
        )
        self.assertEqual(result["source"]["status"], "configured")
        with self.assertRaisesRegex(ConfigError, "already added"):
            add_source(self.config_path, str(repository / "."))

    def test_human_modes_define_repository_roles_and_preserve_reviewed_scope(self) -> None:
        repository, commit = self.create_repository()

        result = add_source(
            self.config_path,
            str(repository),
            # Input order is intentionally reversed; persisted product modes and
            # low-level roles have one canonical order for stable clients.
            modes=["practice_tasks", "accepted_changes"],
            require_modes=True,
            revision=commit,
            include=["app.py", "tests/**"],
            exclude=["vendor/**"],
            license_spdx="Apache-2.0",
            license_attribution="https://example.invalid/workspace",
        )

        source = result["source"]
        self.assertEqual(source["modes"], ["accepted_changes", "practice_tasks"])
        self.assertEqual(source["partition"], "train")
        self.assertEqual(source["roles"], ["history", "rl_seed"])
        self.assertEqual(
            source["filters"],
            {"include": ["app.py", "tests/**"], "exclude": ["vendor/**"]},
        )
        self.assertEqual(
            source["license"],
            {
                "spdx": "Apache-2.0",
                "attribution": "https://example.invalid/workspace",
            },
        )
        self.assertEqual(source["status"], "configured")
        self.assertEqual(source["next_action"]["title"], "Review changes and add tasks")

        declared = load_config(self.config_path).sources[0]
        self.assertEqual(declared["roles"], ["history", "rl_seed"])
        self.assertEqual(declared["revision"], commit)
        self.assertEqual(declared["include"], ["app.py", "tests/**"])
        self.assertEqual(declared["exclude"], ["vendor/**"])

    def test_evaluation_mode_is_isolated_and_cannot_mix_with_training_modes(self) -> None:
        repository, _commit = self.create_repository()

        result = add_source(
            self.config_path,
            str(repository),
            modes=["evaluation_holdout"],
            require_modes=True,
        )

        source = result["source"]
        self.assertEqual(source["modes"], ["evaluation_holdout"])
        self.assertEqual(source["partition"], "evaluation")
        self.assertEqual(source["roles"], ["evaluation"])
        self.assertEqual(source["next_action"]["title"], "Add held-out tasks")

        with self.assertRaisesRegex(ConfigError, "cannot be combined"):
            add_source(
                self.config_path,
                str(repository),
                modes=["evaluation_holdout", "accepted_changes"],
            )

    def test_human_mode_guard_applies_only_to_ambiguous_repositories(self) -> None:
        repository, _commit = self.create_repository()
        examples = self.root / "accepted.jsonl"
        examples.write_text(
            '{"messages":[{"role":"assistant","content":"ok"}]}\n',
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ConfigError, "source mode is required"):
            add_source(self.config_path, str(repository), require_modes=True)

        # A demonstration file already has one intrinsic purpose, so requiring
        # repository intent does not invent a misleading mode for it.
        result = add_source(
            self.config_path,
            str(examples),
            require_modes=True,
            license_spdx="LicenseRef-Internal",
        )
        self.assertEqual(result["source"]["modes"], [])
        self.assertEqual(result["source"]["license"], {"spdx": "LicenseRef-Internal"})
        self.assertEqual(result["source"]["next_action"]["title"], "Validate examples")

    def test_modes_reject_conflicting_low_level_roles_and_partition(self) -> None:
        repository, _commit = self.create_repository()

        with self.assertRaisesRegex(ConfigError, "modes or repository roles"):
            add_source(
                self.config_path,
                str(repository),
                modes=["accepted_changes"],
                roles=["history"],
            )
        with self.assertRaisesRegex(ConfigError, "conflicts with the selected modes"):
            add_source(
                self.config_path,
                str(repository),
                modes=["practice_tasks"],
                partition="evaluation",
            )

    def test_advanced_role_callers_remain_backward_compatible_without_modes(self) -> None:
        repository, commit = self.create_repository()

        result = add_source(
            self.config_path,
            str(repository),
            kind="repository",
            partition="train",
            roles=["rl_seed"],
            revision=commit,
            include=["app.py"],
        )

        self.assertEqual(result["source"]["modes"], ["practice_tasks"])
        self.assertEqual(result["source"]["roles"], ["rl_seed"])
        self.assertEqual(result["source"]["filters"]["include"], ["app.py"])

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
            result = add_source(
                self.config_path,
                "github.com/Owner/Repo",
                license_spdx="MIT",
            )

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
                add_source(
                    self.config_path,
                    "owner/repository",
                    license_spdx="MIT",
                )

        self.assertEqual(self.config_path.read_bytes(), before)
        self.assertEqual(list_sources(self.config_path), [])

    def test_github_materialization_reports_safe_actionable_failures(self) -> None:
        before = self.config_path.read_bytes()
        cases = (
            (RuntimeError("git clone timed out"), "download timed out"),
            (RuntimeError("fatal: repository not found"), "not found or requires access"),
            (
                RuntimeError("cannot resolve declared revision 'missing'"),
                "does not contain the requested branch",
            ),
        )
        for error, expected in cases:
            with self.subTest(expected=expected), patch(
                "autotrainer.source_service.materialize_repository",
                side_effect=error,
            ):
                with self.assertRaisesRegex(ConfigError, expected):
                    add_source(
                        self.config_path,
                        "owner/repository",
                        license_spdx="MIT",
                    )

        self.assertEqual(self.config_path.read_bytes(), before)
        self.assertEqual(list_sources(self.config_path), [])

    def test_github_accepted_changes_require_a_declared_license(self) -> None:
        before = self.config_path.read_bytes()
        with patch("autotrainer.source_service.materialize_repository") as materialize:
            with self.assertRaisesRegex(ConfigError, "declared SPDX license"):
                add_source(self.config_path, "owner/repository")

        materialize.assert_not_called()
        self.assertEqual(self.config_path.read_bytes(), before)
        self.assertEqual(list_sources(self.config_path), [])


if __name__ == "__main__":
    unittest.main()
