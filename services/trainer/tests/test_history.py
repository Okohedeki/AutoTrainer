from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE_ROOT / "src"))

from autotrainer.history import (  # noqa: E402
    HistoryError,
    approved_history_records,
    list_history,
    review_history,
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


class HistoryServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def create_repository(self, name: str = "workspace") -> Path:
        repository = self.root / name
        repository.mkdir()
        run_git(repository, "init", "-q")
        # Personal metadata is deliberately distinctive so persistence tests
        # can prove that candidate artifacts do not retain it.
        run_git(repository, "config", "user.name", "Private Person")
        run_git(repository, "config", "user.email", "private@example.invalid")
        return repository

    def commit(self, repository: Path, message: str) -> str:
        run_git(repository, "add", ".")
        run_git(repository, "commit", "-q", "-m", message)
        return run_git(repository, "rev-parse", "HEAD")

    def config(self, repository: Path, revision: str, **source_changes: object) -> dict:
        source = {
            "id": "work",
            "kind": "repository",
            "uri": repository.name,
            "partition": "train",
            "roles": ["style", "history"],
            "revision": revision,
        }
        source.update(source_changes)
        return {
            "project": {"artifact_dir": ".autotrainer"},
            "sft": {"max_length": 2_048},
            "sources": [source],
        }

    def create_focused_history(self) -> tuple[Path, str]:
        repository = self.create_repository()
        source = repository / "src"
        source.mkdir()
        (source / "App.tsx").write_text(
            "export const label = 'Before';\n",
            encoding="utf-8",
        )
        self.commit(repository, "Initial fixture")
        (source / "App.tsx").write_text(
            "export const label = 'After';\n",
            encoding="utf-8",
        )
        revision = self.commit(repository, "Make the application label easier to understand")
        return repository, revision

    def test_discovers_deterministic_first_parent_candidate_without_worktree_or_pii(self) -> None:
        repository, revision = self.create_focused_history()
        # Discovery reads Git objects at the locked revision. A dirty checkout
        # must not silently become part of an accepted demonstration.
        (repository / "src" / "App.tsx").write_text(
            "export const label = 'Dirty and uncommitted';\n",
            encoding="utf-8",
        )
        config = self.config(repository, revision)

        first = list_history(config, self.root)
        candidate_path = self.root / ".autotrainer" / "history" / "work" / "candidates.jsonl"
        report_path = self.root / ".autotrainer" / "history" / "work" / "history-report.json"
        first_candidate_bytes = candidate_path.read_bytes()
        first_report_bytes = report_path.read_bytes()
        second = list_history(config, self.root)

        self.assertEqual(first, second)
        self.assertEqual(candidate_path.read_bytes(), first_candidate_bytes)
        self.assertEqual(report_path.read_bytes(), first_report_bytes)
        self.assertEqual(first["summary"]["reviewable"], 1)
        candidate = first["candidates"][0]
        self.assertEqual(candidate["commit"], revision)
        self.assertEqual(candidate["decision"], "pending")
        self.assertIn("export const label = 'Before';", candidate["before_context"])
        self.assertNotIn("After", candidate["before_context"])
        self.assertIn("+export const label = 'After';", candidate["patch"])
        self.assertNotIn("Dirty and uncommitted", json.dumps(first))

        persisted = first_candidate_bytes.decode("utf-8") + first_report_bytes.decode("utf-8")
        self.assertNotIn("Private Person", persisted)
        self.assertNotIn("private@example.invalid", persisted)
        self.assertNotIn(str(self.root), persisted)
        self.assertNotIn("origin", persisted.casefold())
        self.assertRegex(candidate["repository_identity"], r"^sha256:[0-9a-f]{64}$")

    def test_first_parent_merge_becomes_one_integration_candidate(self) -> None:
        repository = self.create_repository()
        (repository / "App.tsx").write_text("export const navigation = 'Menu';\n", encoding="utf-8")
        self.commit(repository, "Initial fixture")
        main_branch = run_git(repository, "branch", "--show-current")

        run_git(repository, "checkout", "-q", "-b", "feature")
        (repository / "App.tsx").write_text(
            "export const navigation = 'Open navigation';\n",
            encoding="utf-8",
        )
        child = self.commit(repository, "Change the internal navigation label")
        run_git(repository, "checkout", "-q", main_branch)
        run_git(
            repository,
            "merge",
            "-q",
            "--no-ff",
            "feature",
            "-m",
            "Merge pull request #7 from example/feature",
            "-m",
            "Give the navigation control a clearer accessible label",
        )
        merge = run_git(repository, "rev-parse", "HEAD")

        result = list_history(self.config(repository, merge), self.root)

        self.assertEqual(result["summary"]["reviewable"], 1)
        candidate = result["candidates"][0]
        self.assertEqual(candidate["commit"], merge)
        self.assertNotEqual(candidate["commit"], child)
        self.assertEqual(
            candidate["proposed_instruction"],
            "Give the navigation control a clearer accessible label",
        )

    def test_source_service_style_managed_github_checkout_is_supported_without_url_leakage(self) -> None:
        repository = self.root / ".autotrainer" / "sources" / "github-work"
        repository.mkdir(parents=True)
        run_git(repository, "init", "-q")
        run_git(repository, "config", "user.name", "Private Person")
        run_git(repository, "config", "user.email", "private@example.invalid")
        run_git(
            repository,
            "remote",
            "add",
            "origin",
            "https://oauth2:private-token@github.com/Owner/Repo.git",
        )
        (repository / "App.tsx").write_text("export const state = 1;\n", encoding="utf-8")
        self.commit(repository, "Initial fixture")
        (repository / "App.tsx").write_text("export const state = 2;\n", encoding="utf-8")
        revision = self.commit(repository, "Update the application state default safely")
        config = self.config(repository, revision)
        config["sources"][0]["uri"] = ".autotrainer/sources/github-work"

        result = list_history(config, self.root)

        self.assertEqual(result["summary"]["reviewable"], 1)
        persisted = (
            self.root / ".autotrainer" / "history" / "work" / "history-report.json"
        ).read_text(encoding="utf-8")
        self.assertNotIn("private-token", persisted)
        self.assertNotIn("github.com", persisted)
        self.assertRegex(result["candidates"][0]["repository_identity"], r"^sha256:[0-9a-f]{64}$")

    def test_strict_filters_keep_secrets_binary_generated_and_revert_content_out(self) -> None:
        repository = self.create_repository()
        (repository / "App.tsx").write_text("export const value = 1;\n", encoding="utf-8")
        self.commit(repository, "Initial fixture")

        (repository / "App.tsx").write_text("export const value = 2;\n", encoding="utf-8")
        self.commit(repository, "Make the displayed value reflect the new default")

        (repository / ".env").write_text(
            "ACCESS_TOKEN=ghp_abcdefghijklmnopqrstuvwxyz123456\n",
            encoding="utf-8",
        )
        self.commit(repository, "Temporarily add local credentials")

        (repository / "payload.json").write_bytes(b"{\x00binary}")
        self.commit(repository, "Add generated binary payload")

        (repository / "package-lock.json").write_text("{}\n", encoding="utf-8")
        self.commit(repository, "Refresh generated dependency lock")

        (repository / "App.tsx").write_text("export const value = 3;\n", encoding="utf-8")
        revision = self.commit(repository, "Revert unwanted display experiment")

        result = list_history(self.config(repository, revision), self.root)

        self.assertEqual(result["summary"]["reviewable"], 1)
        self.assertEqual(result["excluded"]["secret_prone_path"], 1)
        self.assertEqual(result["excluded"]["binary_or_unreadable_diff"], 1)
        self.assertEqual(result["excluded"]["generated_path"], 1)
        self.assertEqual(result["excluded"]["revert_commit"], 1)
        persisted = (
            self.root / ".autotrainer" / "history" / "work" / "candidates.jsonl"
        ).read_text(encoding="utf-8")
        self.assertNotIn("ghp_", persisted)
        self.assertNotIn("credentials", persisted.casefold())

    def test_include_scope_rejects_partial_commits_instead_of_trimming_the_answer(self) -> None:
        repository = self.create_repository()
        (repository / "src").mkdir()
        (repository / "docs").mkdir()
        (repository / "src" / "App.tsx").write_text("export const App = 1;\n", encoding="utf-8")
        (repository / "docs" / "guide.md").write_text("Before\n", encoding="utf-8")
        self.commit(repository, "Initial fixture")
        (repository / "src" / "App.tsx").write_text("export const App = 2;\n", encoding="utf-8")
        (repository / "docs" / "guide.md").write_text("After\n", encoding="utf-8")
        revision = self.commit(repository, "Update the component and its guide together")

        result = list_history(
            self.config(repository, revision, include=["src/**"]),
            self.root,
        )

        self.assertEqual(result["summary"]["reviewable"], 0)
        self.assertEqual(result["excluded"]["outside_include_scope"], 1)

    def test_git_style_globs_preserve_dot_directories_and_single_segment_stars(self) -> None:
        repository = self.create_repository()
        (repository / "src" / "private").mkdir(parents=True)
        (repository / ".github").mkdir()
        (repository / "src" / "private" / "App.tsx").write_text(
            "export const nested = 1;\n",
            encoding="utf-8",
        )
        (repository / ".github" / "policy.yml").write_text("enabled: false\n", encoding="utf-8")
        self.commit(repository, "Initial fixture")
        (repository / "src" / "private" / "App.tsx").write_text(
            "export const nested = 2;\n",
            encoding="utf-8",
        )
        nested_revision = self.commit(repository, "Update the nested application component")

        shallow = list_history(
            self.config(repository, nested_revision, include=["src/*.tsx"]),
            self.root,
            write=False,
        )
        recursive = list_history(
            self.config(repository, nested_revision, include=["src/**/*.tsx"]),
            self.root,
            write=False,
        )
        self.assertEqual(shallow["summary"]["reviewable"], 0)
        self.assertEqual(recursive["summary"]["reviewable"], 1)

        (repository / ".github" / "policy.yml").write_text("enabled: true\n", encoding="utf-8")
        dot_revision = self.commit(repository, "Enable the repository policy workflow")
        dot_directory = list_history(
            self.config(repository, dot_revision, include=[".github/**"]),
            self.root,
            write=False,
        )
        self.assertEqual(dot_directory["summary"]["reviewable"], 1)
        self.assertEqual(dot_directory["candidates"][0]["files"][0]["path"], ".github/policy.yml")

    def test_json_credentials_are_blocked_before_candidate_persistence(self) -> None:
        repository = self.create_repository()
        (repository / "config.json").write_text('{"enabled":false}\n', encoding="utf-8")
        self.commit(repository, "Initial fixture")
        (repository / "config.json").write_text(
            '{"password":"supersecretvalue123"}\n',
            encoding="utf-8",
        )
        revision = self.commit(repository, "Update the local service configuration")

        result = list_history(self.config(repository, revision), self.root)

        self.assertEqual(result["summary"]["reviewable"], 0)
        self.assertEqual(result["excluded"]["secret_detected"], 1)
        persisted = (
            self.root / ".autotrainer" / "history" / "work" / "candidates.jsonl"
        ).read_text(encoding="utf-8")
        self.assertNotIn("supersecretvalue123", persisted)

    def test_diff_and_added_file_context_ignore_ambient_git_preferences(self) -> None:
        repository = self.create_repository()
        (repository / "App.tsx").write_text("export const App = 1;\n", encoding="utf-8")
        self.commit(repository, "Initial fixture")
        (repository / "New.ts").write_text("export const created = true;\n", encoding="utf-8")
        revision = self.commit(repository, "Add the shared creation marker module")
        run_git(repository, "config", "diff.noprefix", "true")
        run_git(repository, "config", "core.abbrev", "7")
        run_git(repository, "config", "diff.algorithm", "histogram")

        result = list_history(self.config(repository, revision), self.root)

        candidate = result["candidates"][0]
        self.assertIn("--- /dev/null", candidate["patch"])
        self.assertIn("+++ b/New.ts", candidate["patch"])
        self.assertIn("(file does not exist yet)", candidate["before_context"])
        index_line = next(line for line in candidate["patch"].splitlines() if line.startswith("index "))
        self.assertRegex(index_line, r"^index [0-9a-f]{40}\.\.[0-9a-f]{40}")

    def test_reviews_are_separate_atomic_idempotent_and_compile_to_prompt_completion(self) -> None:
        repository, revision = self.create_focused_history()
        config = self.config(repository, revision)
        candidate = list_history(config, self.root)["candidates"][0]

        with self.assertRaisesRegex(HistoryError, "right to train"):
            review_history(
                config,
                self.root,
                candidate_id=candidate["candidate_id"],
                decision="approved",
                instruction="Make the application label easier to understand.",
            )
        with self.assertRaisesRegex(HistoryError, "requested behavior"):
            review_history(
                config,
                self.root,
                candidate_id=candidate["candidate_id"],
                decision="approved",
                instruction="fix",
                rights_confirmed=True,
            )

        approved = review_history(
            config,
            self.root,
            candidate_id=candidate["candidate_id"],
            decision="approved",
            instruction="Clarify the application label without changing its public API.",
            rights_confirmed=True,
        )
        reviews_path = self.root / ".autotrainer" / "history" / "reviews.json"
        first_bytes = reviews_path.read_bytes()
        repeated = review_history(
            config,
            self.root,
            candidate_id=candidate["candidate_id"],
            decision="approved",
            instruction="Clarify the application label without changing its public API.",
            rights_confirmed=True,
        )

        self.assertEqual(approved, repeated)
        self.assertEqual(reviews_path.read_bytes(), first_bytes)
        self.assertEqual(list(reviews_path.parent.glob(".reviews.json.*.tmp")), [])
        self.assertFalse(reviews_path.with_suffix(".json.lock").exists())
        self.assertEqual(approved["history"]["summary"]["approved"], 1)
        records = approved_history_records(config, self.root)
        self.assertEqual(len(records), 1)
        row = records[0]
        self.assertEqual(row["source_type"], "approved_git_change")
        self.assertEqual(row["source_revision"], revision)
        self.assertEqual(row["prompt"][0]["role"], "system")
        self.assertIn("Pre-change context", row["prompt"][1]["content"])
        self.assertIn("Before", row["prompt"][1]["content"])
        self.assertNotIn("After", row["prompt"][1]["content"])
        self.assertEqual(row["completion"][0]["role"], "assistant")
        self.assertIn("+export const label = 'After';", row["completion"][0]["content"])

        rejected = review_history(
            config,
            self.root,
            candidate_id=candidate["candidate_id"],
            decision="rejected",
        )
        self.assertEqual(rejected["history"]["summary"]["rejected"], 1)
        self.assertEqual(approved_history_records(config, self.root), [])

    def test_stale_review_is_not_reused_after_locked_history_changes(self) -> None:
        repository, revision = self.create_focused_history()
        config = self.config(repository, revision)
        candidate = list_history(config, self.root)["candidates"][0]
        review_history(
            config,
            self.root,
            candidate_id=candidate["candidate_id"],
            decision="approved",
            instruction="Clarify the application label without changing its public API.",
            rights_confirmed=True,
        )

        # A rewritten history has a different candidate digest. The old review
        # remains auditable but cannot silently approve the new answer.
        run_git(repository, "reset", "-q", "--hard", f"{revision}^")
        (repository / "src" / "App.tsx").write_text(
            "export const label = 'Different accepted answer';\n",
            encoding="utf-8",
        )
        replacement = self.commit(repository, "Make the application label easier to understand")
        replacement_config = self.config(repository, replacement)

        result = list_history(replacement_config, self.root)

        self.assertEqual(result["summary"]["approved"], 0)
        self.assertEqual(result["summary"]["stale_reviews"], 1)
        self.assertEqual(result["candidates"][0]["decision"], "pending")
        with self.assertRaisesRegex(HistoryError, "prior approval is stale"):
            approved_history_records(replacement_config, self.root)

    def test_only_train_history_sources_are_discovered(self) -> None:
        repository, revision = self.create_focused_history()
        evaluation = self.config(repository, revision, partition="evaluation")
        style_only = self.config(repository, revision, roles=["style"])

        self.assertEqual(list_history(evaluation, self.root)["summary"]["source_count"], 0)
        self.assertEqual(list_history(style_only, self.root)["summary"]["source_count"], 0)


if __name__ == "__main__":
    unittest.main()
