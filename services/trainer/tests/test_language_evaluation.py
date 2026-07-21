from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE_ROOT / "src"))

from autotrainer.config import (  # noqa: E402
    ConfigError,
    default_config,
    load_config,
    write_config,
)
from autotrainer.language_evaluation import (  # noqa: E402
    LANGUAGE_SUITES,
    get_language_evaluation_workspace,
    require_language_matched_evaluation,
    set_evaluation_language,
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


class LanguageEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.config_path = self.root / "autotrainer.yaml"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def repository(self, name: str, filename: str) -> tuple[Path, str]:
        repository = self.root / name
        repository.mkdir()
        run_git(repository, "init")
        run_git(repository, "config", "user.name", "AutoTrainer Tests")
        run_git(repository, "config", "user.email", "tests@example.invalid")
        (repository / filename).write_text("# held-out code\n", encoding="utf-8")
        run_git(repository, "add", ".")
        run_git(repository, "commit", "-m", "fixture")
        return repository, run_git(repository, "rev-parse", "HEAD")

    def write_project(self, evaluation_filename: str = "test_feature.py") -> None:
        train, train_revision = self.repository("train-repo", "feature.py")
        held_out, held_out_revision = self.repository("eval-repo", evaluation_filename)
        config = default_config(revision="a" * 40)
        config["sources"] = [
            {
                "id": "train-repo",
                "kind": "repository",
                "license": {"spdx": "MIT"},
                "partition": "train",
                "revision": train_revision,
                "roles": ["style"],
                "uri": train.name,
            },
            {
                "id": "eval-repo",
                "kind": "repository",
                "license": {"spdx": "MIT"},
                "partition": "evaluation",
                "revision": held_out_revision,
                "roles": ["evaluation"],
                "uri": held_out.name,
            },
        ]
        write_config(self.config_path, config, overwrite=True)
        receipt = self.root / ".autotrainer" / "dataset" / "freeze.json"
        receipt.parent.mkdir(parents=True)
        receipt.write_text(
            json.dumps({"language_counts": {"python": 7}, "schema_version": 1}) + "\n",
            encoding="utf-8",
        )

    def test_auto_selection_matches_primary_training_and_held_out_language(self) -> None:
        self.write_project()

        workspace = get_language_evaluation_workspace(self.config_path)

        self.assertEqual(workspace["status"], "ready")
        self.assertEqual(workspace["selected"], "python")
        self.assertEqual(workspace["inferred_training_language"], "python")
        self.assertGreater(workspace["evaluation_language_counts"]["python"], 0)
        self.assertEqual(require_language_matched_evaluation(self.config_path)["selected"], "python")

    def test_explicit_mismatch_blocks_evaluation(self) -> None:
        self.write_project()

        workspace = set_evaluation_language(self.config_path, "csharp")

        self.assertEqual(workspace["status"], "blocked")
        self.assertTrue(
            any("does not match the primary Python" in value for value in workspace["blockers"])
        )
        with self.assertRaisesRegex(ConfigError, "does not match"):
            require_language_matched_evaluation(self.config_path)

    def test_explicit_secondary_language_does_not_override_primary_language(self) -> None:
        self.write_project()
        receipt = self.root / ".autotrainer" / "dataset" / "freeze.json"
        receipt.write_text(
            json.dumps(
                {"language_counts": {"python": 7, "csharp": 2}, "schema_version": 1}
            )
            + "\n",
            encoding="utf-8",
        )

        workspace = set_evaluation_language(self.config_path, "csharp")

        self.assertEqual(workspace["status"], "blocked")
        self.assertEqual(workspace["inferred_training_language"], "python")

    def test_module_javascript_holdout_blocks_python_training(self) -> None:
        self.write_project("build.mjs")

        workspace = get_language_evaluation_workspace(self.config_path)

        self.assertEqual(workspace["status"], "blocked")
        self.assertEqual(
            workspace["evaluation_language_counts"], {"typescript_react": 1}
        )
        self.assertTrue(
            any("do not contain Python code" in value for value in workspace["blockers"])
        )

    def test_undetectable_held_out_language_fails_closed(self) -> None:
        self.write_project("page.html")

        workspace = get_language_evaluation_workspace(self.config_path)

        self.assertEqual(workspace["status"], "blocked")
        self.assertEqual(workspace["evaluation_language_counts"], {})
        self.assertTrue(
            any("do not contain detectable" in value for value in workspace["blockers"])
        )

    def test_unknown_training_primary_fails_closed_for_explicit_suite(self) -> None:
        self.write_project()
        receipt = self.root / ".autotrainer" / "dataset" / "freeze.json"
        receipt.write_text(
            json.dumps({"language_counts": {}, "schema_version": 1}) + "\n",
            encoding="utf-8",
        )
        train = self.root / "train-repo"
        (train / "feature.py").unlink()
        (train / "README.md").write_text("# training context\n", encoding="utf-8")
        run_git(train, "add", "-A")
        run_git(train, "commit", "-m", "remove supported training language")
        config = load_config(self.config_path)
        updated = dict(config.data)
        updated["sources"] = [dict(source) for source in config.sources]
        updated["sources"][0]["revision"] = run_git(train, "rev-parse", "HEAD")
        updated["evaluation"] = dict(updated["evaluation"])
        updated["evaluation"]["language"] = "python"
        write_config(self.config_path, updated, overwrite=True)

        workspace = get_language_evaluation_workspace(self.config_path)

        self.assertEqual(workspace["status"], "blocked")
        self.assertIsNone(workspace["inferred_training_language"])
        self.assertTrue(
            any("does not identify" in value for value in workspace["blockers"])
        )

    def test_shipped_profiles_cover_the_initial_four_language_families(self) -> None:
        self.assertEqual(
            set(LANGUAGE_SUITES),
            {"python", "typescript_react", "csharp", "cpp"},
        )
        for suite in LANGUAGE_SUITES.values():
            self.assertIn("build_passed", suite["metrics"])
            self.assertIn("task_pass_rate", suite["metrics"])
            self.assertIn("pass_at_1", suite["metrics"])
            self.assertTrue(suite["benchmark_inspirations"])

    def test_unknown_language_is_rejected_without_changing_config(self) -> None:
        self.write_project()
        before = self.config_path.read_bytes()

        with self.assertRaisesRegex(ConfigError, "language must be"):
            set_evaluation_language(self.config_path, "ruby")

        self.assertEqual(self.config_path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
