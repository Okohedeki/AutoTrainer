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
from autotrainer.github_pr_service import (  # noqa: E402
    _git_is_ancestor,
    GitHubPullRequestError,
    read_merged_pull_request_catalog,
    sync_merged_pull_requests,
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


class GitHubPullRequestServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.config_path = self.root / "autotrainer.yaml"
        self.repository = self.root / ".autotrainer" / "sources" / "owner-repo"
        self.repository.mkdir(parents=True)
        run_git(self.repository, "init")
        run_git(self.repository, "config", "user.name", "AutoTrainer Tests")
        run_git(self.repository, "config", "user.email", "tests@example.invalid")
        run_git(
            self.repository,
            "remote",
            "add",
            "origin",
            "https://github.com/owner/repo.git",
        )
        (self.repository / "app.py").write_text("print('ready')\n", encoding="utf-8")
        run_git(self.repository, "add", ".")
        run_git(self.repository, "commit", "-m", "Merge accepted change")
        self.revision = run_git(self.repository, "rev-parse", "HEAD")
        tree = run_git(self.repository, "rev-parse", "HEAD^{tree}")
        self.unrelated_revision = run_git(
            self.repository,
            "commit-tree",
            tree,
            "-m",
            "Unrelated change",
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def write_source(self, *, license_spdx: str = "MIT", revision: str | None = None) -> None:
        config = default_config()
        source: dict[str, object] = {
            "id": "owner-repo",
            "kind": "repository",
            "partition": "train",
            "revision": revision or self.revision,
            "roles": ["history"],
            "uri": ".autotrainer/sources/owner-repo",
        }
        if license_spdx:
            source["license"] = {"spdx": license_spdx}
        config["sources"] = [source]
        write_config(self.config_path, config, overwrite=True)

    def shallow_history_fixture(self) -> tuple[Path, str, str]:
        worktree = self.root / "history-worktree"
        worktree.mkdir()
        run_git(worktree, "init")
        run_git(worktree, "config", "user.name", "AutoTrainer Tests")
        run_git(worktree, "config", "user.email", "tests@example.invalid")
        (worktree / "app.py").write_text("print('first')\n", encoding="utf-8")
        run_git(worktree, "add", ".")
        run_git(worktree, "commit", "-m", "Accepted change")
        accepted = run_git(worktree, "rev-parse", "HEAD")
        (worktree / "app.py").write_text("print('tip')\n", encoding="utf-8")
        run_git(worktree, "add", ".")
        run_git(worktree, "commit", "-m", "Pinned tip")
        tip = run_git(worktree, "rev-parse", "HEAD")

        remote = self.root / "history.git"
        subprocess.run(
            ["git", "clone", "--quiet", "--bare", str(worktree), str(remote)],
            check=True,
            capture_output=True,
        )
        shallow = self.root / "history-shallow"
        subprocess.run(
            [
                "git",
                "clone",
                "--quiet",
                "--depth=1",
                remote.as_uri(),
                str(shallow),
            ],
            check=True,
            capture_output=True,
        )
        self.assertEqual(run_git(shallow, "rev-parse", "--is-shallow-repository"), "true")
        return shallow, accepted, tip

    def pull_request(
        self,
        *,
        number: int,
        branch: str = "main",
        merge_commit: str | None = None,
        merged_at: str | None = "2026-07-18T12:00:00Z",
    ) -> dict[str, object]:
        return {
            "base": {"ref": branch},
            "body": "Explain why the implementation is correct.",
            "html_url": "https://github.com/owner/repo/pull/1",
            "merge_commit_sha": merge_commit or self.revision,
            "merged_at": merged_at,
            "number": number,
            "title": "Implement the accepted change",
            "user": {"login": "must-not-be-persisted"},
        }

    def test_sync_persists_only_merged_main_or_master_prs_in_the_pinned_history(self) -> None:
        self.write_source()
        response = [
            self.pull_request(number=1),
            self.pull_request(number=2, branch="master"),
            self.pull_request(number=3, branch="develop"),
            self.pull_request(number=4, merged_at=None),
            self.pull_request(number=5, merge_commit=self.unrelated_revision),
            self.pull_request(number=1),
        ]

        with patch("autotrainer.github_pr_service._github_request", return_value=response):
            result = sync_merged_pull_requests(self.config_path)

        self.assertEqual(result["status"], "synced")
        self.assertEqual(result["source_count"], 1)
        self.assertEqual(result["sources"][0]["merged_pull_request_count"], 2)
        self.assertEqual(result["sources"][0]["skipped_count"], 4)
        status = read_merged_pull_request_catalog(self.config_path)
        self.assertEqual(status["status"], "ready")
        self.assertEqual(status["merged_pull_request_count"], 2)

        catalog_path = (
            self.root
            / ".autotrainer"
            / "dataset"
            / "github-prs"
            / "owner-repo.json"
        )
        payload = json.loads(catalog_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["license_spdx"], "MIT")
        self.assertEqual({item["base_branch"] for item in payload["pull_requests"]}, {"main", "master"})
        serialized = catalog_path.read_text(encoding="utf-8")
        self.assertNotIn("must-not-be-persisted", serialized)
        self.assertNotIn("html_url", serialized)

    def test_sync_repairs_legacy_shallow_clone_before_ancestry_check(self) -> None:
        repository, accepted, tip = self.shallow_history_fixture()
        config = default_config()
        source = {
            "id": "owner-repo",
            "kind": "repository",
            "partition": "train",
            "revision": tip,
            "roles": ["history"],
            "uri": repository.name,
            "license": {"spdx": "MIT"},
        }
        config["sources"] = [source]
        write_config(self.config_path, config, overwrite=True)

        with patch(
            "autotrainer.github_pr_service._github_training_sources",
            return_value=[(source, {"label": "owner/repo"})],
        ), patch(
            "autotrainer.github_pr_service._github_request",
            return_value=[
                self.pull_request(
                    number=1,
                    branch="master",
                    merge_commit=accepted,
                )
            ],
        ):
            result = sync_merged_pull_requests(self.config_path)

        self.assertEqual(result["sources"][0]["merged_pull_request_count"], 1)
        self.assertEqual(run_git(repository, "rev-parse", "--is-shallow-repository"), "false")
        with patch(
            "autotrainer.github_pr_service._github_training_sources",
            return_value=[(source, {"label": "owner/repo"})],
        ):
            catalog = read_merged_pull_request_catalog(self.config_path)
        self.assertEqual(catalog["merged_pull_request_count"], 1)

    def test_ancestry_check_fails_closed_for_incomplete_shallow_history(self) -> None:
        repository, accepted, tip = self.shallow_history_fixture()

        with self.assertRaisesRegex(
            GitHubPullRequestError,
            "incomplete shallow history",
        ):
            _git_is_ancestor(repository, accepted, tip)

    def test_catalog_becomes_stale_when_the_pinned_revision_changes(self) -> None:
        self.write_source()
        with patch(
            "autotrainer.github_pr_service._github_request",
            return_value=[self.pull_request(number=1)],
        ):
            sync_merged_pull_requests(self.config_path)

        self.write_source(revision="a" * 40)

        status = read_merged_pull_request_catalog(self.config_path)
        self.assertEqual(status["status"], "needs_sync")
        self.assertEqual(status["merged_pull_request_count"], 0)

    def test_missing_license_blocks_remote_discovery(self) -> None:
        self.write_source(license_spdx="")
        with patch("autotrainer.github_pr_service._github_request") as request:
            with self.assertRaisesRegex(ConfigError, "declared SPDX license"):
                sync_merged_pull_requests(self.config_path)
        request.assert_not_called()

    def test_empty_allowlist_is_explicit(self) -> None:
        write_config(self.config_path, default_config(), overwrite=True)

        self.assertEqual(
            sync_merged_pull_requests(self.config_path)["status"],
            "no_github_training_sources",
        )
        self.assertEqual(
            read_merged_pull_request_catalog(self.config_path)["status"],
            "no_github_training_sources",
        )


if __name__ == "__main__":
    unittest.main()
