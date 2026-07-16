from __future__ import annotations

import inspect
import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any, Mapping

SERVICE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = SERVICE_ROOT.parents[1]
sys.path.insert(0, str(SERVICE_ROOT / "src"))

from autotrainer import (  # noqa: E402
    ManifestError,
    RolloutVerifierReport,
    TaskManifest,
    score_rollout,
)
from autotrainer.environment import CheckResult, EpisodeResult  # noqa: E402
from autotrainer.environments.frontend import (  # noqa: E402
    EpisodeTimeoutError,
    FrontendEnvironment,
)


class PolicyToolSurfaceTests(unittest.TestCase):
    def test_only_declared_policy_tools_are_public_callables(self) -> None:
        # TRL reserves reset/get_reward and exposes the remaining public
        # callables as tools. Evaluation helpers must therefore stay private.
        public = {
            name
            for name, value in inspect.getmembers(FrontendEnvironment, callable)
            if not name.startswith("_")
        }
        self.assertEqual(
            public,
            {
                "reset",
                "get_reward",
                "list_files",
                "read_file",
                "search_code",
                "apply_patch",
                "run_check",
            },
        )


def executable_manifest(*, browser_tests: str = "npm run test:browser") -> dict[str, Any]:
    return {
        "version": "1.0",
        "task": {
            "id": "pricing-001",
            "instruction": "Repair the responsive pricing page.",
            "sourceId": "storefront",
            "startingRevision": "locked",
            "split": "evaluation",
            "groupId": "storefront",
        },
        "runtime": {
            "workingDirectory": ".",
            "install": "npm ci",
            "build": "npm run build",
            "tests": "npm test",
            "browserTests": browser_tests,
        },
        "tools": ["list_files", "read_file", "search_code", "apply_patch", "run_check"],
        "verifier": {
            "bundle": "verifier",
            "command": "hidden verify",
            "reportPath": "verifier-report.json",
        },
        "rewards": {
            "buildGate": True,
            "regressionGate": True,
            "regressionSafety": 0.2,
            "taskTests": 0.35,
            "responsiveRules": 0.2,
            "designRules": 0.15,
            "patchQuality": 0.1,
        },
        "limits": {
            "toolCalls": 40,
            "commandTimeoutSeconds": 120,
            "episodeTimeoutSeconds": 900,
            "networkAccess": False,
        },
    }


class ScriptedFrontendEnvironment(FrontendEnvironment):
    def __init__(
        self,
        manifest: TaskManifest,
        workspace: Path,
        *,
        failures: dict[str, int] | None = None,
        timeouts: set[str] | None = None,
    ) -> None:
        super().__init__()
        self._manifest = manifest
        self._workspace = workspace
        self._temporary_root = workspace.parent
        self._task_root = workspace.parent
        self._backend = "docker"
        self._image = "test-image"
        self._started_at = time.monotonic()
        self._deadline = self._started_at + manifest.episode_timeout_seconds
        self.failures = failures or {}
        self.timeouts = timeouts or set()
        self.commands: list[tuple[str, bool]] = []
        self.diff = "diff --git a/site.css b/site.css\n--- a/site.css\n+++ b/site.css\n"

    def _capture_unified_diff(self) -> str:
        return self.diff

    def _run_container_command(
        self, command: str, *, include_verifier: bool = False
    ) -> subprocess.CompletedProcess[str]:
        self.commands.append((command, include_verifier))
        if command in self.timeouts:
            raise subprocess.TimeoutExpired(command, 1, output="partial output")
        returncode = self.failures.get(command, 0)
        if include_verifier and returncode == 0:
            (self._require_workspace() / "verifier-report.json").write_text(
                json.dumps(
                    {
                        "regression_pass_rate": 1.0,
                        "task_pass_rate": 1.0,
                        "responsive_pass_rate": 0.75,
                        "design_rule_pass_rate": 0.8,
                        "code_quality_pass_rate": 0.9,
                    }
                ),
                encoding="utf-8",
            )
        return subprocess.CompletedProcess(
            [command], returncode, f"{command} stdout", f"{command} stderr"
        )


class RewardTests(unittest.TestCase):
    def test_scores_verified_rollout(self) -> None:
        reward = score_rollout(RolloutVerifierReport(True, 1, 1, 0.75, 0.8, 0.9))
        self.assertFalse(reward.gated)
        self.assertEqual(reward.total, 0.91)

    def test_build_failure_is_hard_gated(self) -> None:
        reward = score_rollout(RolloutVerifierReport(False, 1, 1, 1, 1, 1))
        self.assertTrue(reward.gated)
        self.assertEqual(reward.gate_reason, "build_failed")
        self.assertEqual(reward.total, 0)

    def test_regression_is_hard_gated(self) -> None:
        reward = score_rollout(RolloutVerifierReport(True, 0.99, 1, 1, 1, 1))
        self.assertEqual(reward.gate_reason, "regression_failed")
        self.assertEqual(reward.total, 0)


class ManifestTests(unittest.TestCase):
    def load_example(self) -> dict:
        path = (
            REPOSITORY_ROOT
            / "examples"
            / "frontend-expert"
            / "tasks"
            / "train"
            / "responsive-pricing"
            / "task.json"
        )
        return json.loads(path.read_text(encoding="utf-8"))

    def test_loads_example_manifest(self) -> None:
        task = TaskManifest.from_mapping(self.load_example())
        self.assertEqual(task.task_id, "responsive-pricing-train-001")
        self.assertFalse(task.network_access)

    def test_rejects_network_access(self) -> None:
        payload = self.load_example()
        payload["limits"]["networkAccess"] = True
        with self.assertRaisesRegex(ManifestError, "disable network access"):
            TaskManifest.from_mapping(payload)

    def test_loads_executable_v1_manifest(self) -> None:
        payload = {
            "version": "1.0",
            "task": {
                "id": "pricing-001",
                "instruction": "Repair the pricing layout and preserve existing interactions.",
                "sourceId": "storefront",
                "startingRevision": "locked",
                "split": "train",
                "groupId": "storefront",
            },
            "runtime": {
                "workingDirectory": ".",
                "install": "npm ci",
                "build": "npm run build",
                "tests": "npm test",
                "browserTests": "npm run test:browser",
            },
            "tools": ["list_files", "read_file", "search_code", "apply_patch", "run_check"],
            "verifier": {
                "bundle": "verifier",
                "command": "node /autotrainer-verifier/verify.mjs",
                "reportPath": ".autotrainer-verifier-report.json",
            },
            "rewards": {
                "buildGate": True,
                "regressionGate": True,
                "regressionSafety": 0.2,
                "taskTests": 0.35,
                "responsiveRules": 0.2,
                "designRules": 0.15,
                "patchQuality": 0.1,
            },
            "limits": {
                "toolCalls": 40,
                "commandTimeoutSeconds": 120,
                "episodeTimeoutSeconds": 900,
                "networkAccess": False,
            },
        }
        task = TaskManifest.from_mapping(payload)
        self.assertEqual(task.source_id, "storefront")
        self.assertEqual(task.starting_revision, "locked")
        self.assertEqual(task.verifier_report_path, ".autotrainer-verifier-report.json")


class SnapshotIsolationTests(unittest.TestCase):
    @staticmethod
    def _git(repository: Path, *arguments: str) -> str:
        completed = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={repository.as_posix()}",
                "-C",
                str(repository),
                *arguments,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode:
            raise AssertionError(completed.stderr or completed.stdout)
        return completed.stdout.strip()

    def test_verifier_bundle_must_be_outside_editable_project(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            source = Path(temporary_directory) / "source"
            (source / "project" / "hidden-verifier").mkdir(parents=True)
            payload = executable_manifest()
            payload["runtime"]["workingDirectory"] = "project"
            payload["verifier"]["bundle"] = "project/hidden-verifier"
            manifest = TaskManifest.from_mapping(payload)

            with self.assertRaisesRegex(RuntimeError, "must not overlap"):
                FrontendEnvironment()._validate_verifier_boundary(
                    manifest,
                    source,
                    source,
                )

            payload["verifier"]["bundle"] = "."
            ancestor_manifest = TaskManifest.from_mapping(payload)
            with self.assertRaisesRegex(RuntimeError, "must not overlap"):
                FrontendEnvironment()._validate_verifier_boundary(
                    ancestor_manifest,
                    source,
                    source,
                )

            payload["verifier"]["bundle"] = "hidden-verifier"
            external_manifest = TaskManifest.from_mapping(payload)
            FrontendEnvironment()._validate_verifier_boundary(
                external_manifest,
                source,
                source,
            )

    def test_materialize_exports_project_without_sibling_files_or_git_objects(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source"
            (source / "project").mkdir(parents=True)
            (source / "hidden-verifier").mkdir()
            (source / "project" / "app.txt").write_text("editable\n", encoding="utf-8")
            secret = source / "hidden-verifier" / "secret.txt"
            secret.write_text("never policy visible\n", encoding="utf-8")
            subprocess.run(
                ["git", "init", "--quiet", str(source)],
                capture_output=True,
                text=True,
                check=True,
            )
            self._git(source, "add", "--all")
            self._git(
                source,
                "-c",
                "user.name=Fixture",
                "-c",
                "user.email=fixture@example.test",
                "commit",
                "--quiet",
                "-m",
                "fixture",
            )
            revision = self._git(source, "rev-parse", "HEAD")
            secret_object = self._git(source, "rev-parse", "HEAD:hidden-verifier/secret.txt")
            destination = root / "episode" / "workspace"
            destination.parent.mkdir()

            FrontendEnvironment()._materialize(
                source,
                revision,
                destination,
                "project",
            )

            self.assertEqual(
                (destination / "project" / "app.txt").read_text(encoding="utf-8"),
                "editable\n",
            )
            self.assertFalse((destination / "hidden-verifier").exists())
            self.assertFalse((destination.parent / "locked-source").exists())
            self.assertEqual(self._git(destination, "status", "--porcelain"), "")
            object_probe = subprocess.run(
                ["git", "-C", str(destination), "cat-file", "-e", secret_object],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(object_probe.returncode, 0)

    def test_materialize_uses_only_the_locked_tree_and_builds_a_fresh_history(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source"
            project = source / "project"
            verifier = source / "hidden-verifier"
            retired = source / "retired-verifier"
            project.mkdir(parents=True)
            verifier.mkdir()
            retired.mkdir()
            # These attributes would mutate a normal checkout/index round trip.
            # The episode exporter must preserve the committed blob bytes instead.
            (project / ".gitattributes").write_text("app.txt ident\n", encoding="utf-8")
            (project / "app.txt").write_text("committed $Id$\n", encoding="utf-8")
            (verifier / "current-secret.txt").write_text(
                "current verifier secret\n",
                encoding="utf-8",
            )
            (retired / "historical-secret.txt").write_text(
                "deleted verifier secret\n",
                encoding="utf-8",
            )
            subprocess.run(
                ["git", "init", "--quiet", str(source)],
                capture_output=True,
                text=True,
                check=True,
            )
            self._git(source, "add", "--all")
            self._git(
                source,
                "-c",
                "user.name=Fixture",
                "-c",
                "user.email=fixture@example.test",
                "commit",
                "--quiet",
                "-m",
                "add current and historical verifier fixtures",
            )
            historical_secret_object = self._git(
                source,
                "rev-parse",
                "HEAD:retired-verifier/historical-secret.txt",
            )
            (retired / "historical-secret.txt").unlink()
            retired.rmdir()
            self._git(source, "add", "--all")
            self._git(
                source,
                "-c",
                "user.name=Fixture",
                "-c",
                "user.email=fixture@example.test",
                "commit",
                "--quiet",
                "-m",
                "retire old verifier",
            )
            revision = self._git(source, "rev-parse", "HEAD")
            current_secret_object = self._git(
                source,
                "rev-parse",
                "HEAD:hidden-verifier/current-secret.txt",
            )

            # A locked revision, not the caller's dirty checkout, defines the
            # exact policy-visible input to a reproducible training episode.
            (project / "app.txt").write_text("dirty worktree value\n", encoding="utf-8")
            (project / "untracked.txt").write_text("do not export\n", encoding="utf-8")
            destinations = [root / "episode-a" / "workspace", root / "episode-b" / "workspace"]
            environment = FrontendEnvironment()
            for destination in destinations:
                destination.parent.mkdir()
                environment._materialize(source, revision, destination, "project")

                self.assertEqual(
                    (destination / "project" / "app.txt").read_bytes(),
                    b"committed $Id$\n",
                )
                self.assertFalse((destination / "project" / "untracked.txt").exists())
                self.assertFalse((destination / "hidden-verifier").exists())
                self.assertFalse((destination / "retired-verifier").exists())
                self.assertEqual(self._git(destination, "status", "--porcelain"), "")
                # A single parentless baseline cannot reveal source history.
                head_with_parents = self._git(
                    destination,
                    "rev-list",
                    "--parents",
                    "-n",
                    "1",
                    "HEAD",
                )
                self.assertEqual(len(head_with_parents.split()), 1)
                self.assertEqual(self._git(destination, "remote"), "")
                alternates = destination / ".git" / "objects" / "info" / "alternates"
                self.assertFalse(alternates.exists())
                config = (destination / ".git" / "config").read_text(encoding="utf-8")
                self.assertNotIn(str(source), config)
                for secret_object in (current_secret_object, historical_secret_object):
                    probe = subprocess.run(
                        ["git", "-C", str(destination), "cat-file", "-e", secret_object],
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    self.assertNotEqual(probe.returncode, 0)

            # Fixed metadata and exact input blobs make equivalent episode
            # baselines byte-for-byte reproducible across temporary locations.
            self.assertEqual(
                self._git(destinations[0], "rev-parse", "HEAD"),
                self._git(destinations[1], "rev-parse", "HEAD"),
            )

    def test_materialize_rejects_a_git_symlink_entry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source"
            (source / "project").mkdir(parents=True)
            (source / "project" / "app.txt").write_text("safe\n", encoding="utf-8")
            subprocess.run(
                ["git", "init", "--quiet", str(source)],
                capture_output=True,
                text=True,
                check=True,
            )
            self._git(source, "add", "--all")
            link_object = subprocess.run(
                ["git", "-C", str(source), "hash-object", "-w", "--stdin"],
                input="../hidden-verifier",
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            self._git(
                source,
                "update-index",
                "--add",
                "--cacheinfo",
                "120000",
                link_object,
                "project/verifier-link",
            )
            self._git(
                source,
                "-c",
                "user.name=Fixture",
                "-c",
                "user.email=fixture@example.test",
                "commit",
                "--quiet",
                "-m",
                "add a Git symlink entry",
            )

            with self.assertRaisesRegex(RuntimeError, "reject symlinks"):
                FrontendEnvironment()._materialize(
                    source,
                    self._git(source, "rev-parse", "HEAD"),
                    root / "episode" / "workspace",
                    "project",
                )

    def test_materialize_rejects_non_git_sources_instead_of_trusting_a_tree_label(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source"
            (source / "project").mkdir(parents=True)
            (source / "project" / "app.txt").write_text("live tree\n", encoding="utf-8")

            # A caller-provided digest is not proof that these mutable bytes
            # came from that digest, so V1 fails closed on non-Git sources.
            with self.assertRaisesRegex(RuntimeError, "Git-backed"):
                FrontendEnvironment()._materialize(
                    source,
                    f"tree:{'0' * 64}",
                    root / "episode" / "workspace",
                    "project",
                )

    def test_materialize_bounds_file_count_blob_size_and_total_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source"
            (source / "project").mkdir(parents=True)
            (source / "project" / "one.txt").write_text("one!\n", encoding="utf-8")
            (source / "project" / "two.txt").write_text("two!\n", encoding="utf-8")
            subprocess.run(
                ["git", "init", "--quiet", str(source)],
                capture_output=True,
                text=True,
                check=True,
            )
            self._git(source, "add", "--all")
            self._git(
                source,
                "-c",
                "user.name=Fixture",
                "-c",
                "user.email=fixture@example.test",
                "commit",
                "--quiet",
                "-m",
                "bounded fixture",
            )
            revision = self._git(source, "rev-parse", "HEAD")
            cases = (
                ("_SNAPSHOT_FILE_LIMIT", 1, "file snapshot limit"),
                ("_SNAPSHOT_BLOB_BYTES", 4, "blob exceeds"),
                ("_SNAPSHOT_TOTAL_BYTES", 7, "byte snapshot limit"),
            )
            for index, (attribute, limit, message) in enumerate(cases):
                with self.subTest(limit=attribute):
                    environment = FrontendEnvironment()
                    setattr(environment, attribute, limit)
                    with self.assertRaisesRegex(RuntimeError, message):
                        environment._materialize(
                            source,
                            revision,
                            root / f"episode-{index}" / "workspace",
                            "project",
                        )


class EpisodeFinalizationTests(unittest.TestCase):
    def prepare_environment(
        self,
        temporary_directory: str,
        *,
        browser_tests: str = "npm run test:browser",
        failures: dict[str, int] | None = None,
        timeouts: set[str] | None = None,
    ) -> ScriptedFrontendEnvironment:
        episode_root = Path(temporary_directory) / "episode"
        workspace = episode_root / "workspace"
        workspace.mkdir(parents=True)
        (workspace / "verifier-report.json").write_text(
            json.dumps(
                {
                    "regression_pass_rate": 1.0,
                    "task_pass_rate": 1.0,
                    "responsive_pass_rate": 0.75,
                    "design_rule_pass_rate": 0.8,
                    "code_quality_pass_rate": 0.9,
                }
            ),
            encoding="utf-8",
        )
        manifest = TaskManifest.from_mapping(
            executable_manifest(browser_tests=browser_tests)
        )
        return ScriptedFrontendEnvironment(
            manifest,
            workspace,
            failures=failures,
            timeouts=timeouts,
        )

    def test_finalize_captures_complete_evaluation_before_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            environment = self.prepare_environment(temporary_directory)
            environment._tool_calls = 3

            result = environment._finalize()

            self.assertIsInstance(result, EpisodeResult)
            self.assertFalse(result.gated)
            self.assertEqual(result.reward, 0.91)
            self.assertEqual(result.raw_verifier_rates["responsive_rules"], 0.75)
            self.assertEqual(result.weighted_signals["task_tests"], 0.35)
            self.assertEqual(result.tool_call_count, 3)
            self.assertEqual(result.unified_diff, environment.diff)
            self.assertGreaterEqual(result.elapsed_seconds, 0)
            self.assertEqual(result.checks["browserTests"].status, "passed")
            self.assertTrue(result.checks["verifier"].passed)
            self.assertIsNone(environment._workspace)
            self.assertIs(environment.last_result, result)
            self.assertEqual(environment._finalize(), result)

    def test_browser_tests_are_a_hard_gate_when_declared(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            environment = self.prepare_environment(
                temporary_directory,
                failures={"npm run test:browser": 1},
            )

            result = environment._finalize()

            self.assertTrue(result.gated)
            self.assertEqual(result.hard_gate_reason, "browser_tests_failed")
            self.assertEqual(result.reward, 0.0)
            self.assertEqual(result.checks["browserTests"].status, "failed")
            self.assertNotIn(("hidden verify", True), environment.commands)
            self.assertIsNone(environment._workspace)

    def test_explicitly_omitted_browser_tests_do_not_create_a_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            environment = self.prepare_environment(
                temporary_directory,
                browser_tests="",
            )

            result = environment._finalize()

            self.assertFalse(result.gated)
            self.assertEqual(result.checks["browserTests"].status, "not_configured")
            self.assertIn(("hidden verify", True), environment.commands)

    def test_command_timeout_is_structured_and_always_cleans_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            environment = self.prepare_environment(
                temporary_directory,
                timeouts={"npm run build"},
            )

            result = environment._finalize()

            self.assertEqual(result.hard_gate_reason, "build_timeout")
            self.assertTrue(result.checks["build"].timed_out)
            self.assertEqual(result.unified_diff, environment.diff)
            self.assertIsNone(environment._workspace)

    def test_hidden_verifier_cannot_reuse_a_policy_supplied_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            environment = self.prepare_environment(
                temporary_directory,
                failures={"hidden verify": 1},
            )

            result = environment._finalize()

            self.assertEqual(result.hard_gate_reason, "verifier_failed")
            self.assertEqual(result.raw_verifier_rates, {})
            self.assertEqual(result.weighted_signals, {})
            self.assertEqual(result.reward, 0.0)

    def test_episode_deadline_applies_to_policy_tools_and_cleans(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            environment = self.prepare_environment(temporary_directory)
            environment._latest_diff = environment.diff
            environment._deadline = time.monotonic() - 1

            with self.assertRaises(EpisodeTimeoutError):
                environment.list_files()

            self.assertIsNone(environment._workspace)
            self.assertIsNotNone(environment.last_result)
            self.assertEqual(environment.last_result.hard_gate_reason, "episode_timeout")
            self.assertEqual(environment.last_result.unified_diff, environment.diff)

    def test_get_reward_delegates_to_structured_finalize(self) -> None:
        checks = {
            "build": CheckResult(
                "build", True, "passed", True, 0, False, 0.1, "", ""
            )
        }
        expected = EpisodeResult(
            task_id="task",
            hard_gate_reason=None,
            raw_verifier_rates={},
            weighted_signals={},
            reward=0.75,
            checks=checks,
            tool_call_count=0,
            elapsed_seconds=0.1,
            unified_diff="",
        )

        class DelegatingEnvironment(FrontendEnvironment):
            def _finalize(self) -> EpisodeResult:
                return expected

        self.assertEqual(DelegatingEnvironment().get_reward(), 0.75)


class EvaluatePatchTests(unittest.TestCase):
    def test_replays_patch_before_install_without_policy_tool_count(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "episode"
            workspace = root / "workspace"
            workspace.mkdir(parents=True)
            (workspace / "verifier-report.json").write_text(
                json.dumps(
                    {
                        "regression_pass_rate": 1.0,
                        "task_pass_rate": 1.0,
                        "responsive_pass_rate": 1.0,
                        "design_rule_pass_rate": 1.0,
                        "code_quality_pass_rate": 1.0,
                    }
                ),
                encoding="utf-8",
            )
            manifest = TaskManifest.from_mapping(executable_manifest())

            class ReplayEnvironment(ScriptedFrontendEnvironment):
                def __init__(self) -> None:
                    super().__init__(manifest, workspace)
                    self.events: list[str] = []

                def _initialize(self, task_row: Mapping[str, Any]) -> tuple[TaskManifest, str]:
                    self.events.append("initialize")
                    return manifest, "locked-revision"

                def _apply_unified_diff(self, patch: str) -> tuple[bool, str]:
                    self.events.append("apply_patch")
                    self.diff = patch
                    return True, "patch applied"

                def _run_container_command(
                    self, command: str, *, include_verifier: bool = False
                ) -> subprocess.CompletedProcess[str]:
                    self.events.append(command)
                    return super()._run_container_command(
                        command, include_verifier=include_verifier
                    )

            environment = ReplayEnvironment()
            patch = "diff --git a/site.css b/site.css\n--- a/site.css\n+++ b/site.css\n"

            result = environment._evaluate_patch({}, patch)

            self.assertEqual(result.tool_call_count, 0)
            self.assertEqual(result.unified_diff, patch)
            self.assertLess(
                environment.events.index("apply_patch"),
                environment.events.index("npm ci"),
            )
            self.assertEqual(result.checks["patch"].status, "passed")
            self.assertIsNone(environment._workspace)


if __name__ == "__main__":
    unittest.main()
