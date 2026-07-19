"""Container-isolated frontend environment for TRL's environment factory API.

The policy never receives a general shell tool.  It can inspect and patch a
disposable checkout and invoke only commands named by the trusted task manifest.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
import time
import unicodedata
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping
from uuid import uuid4

from ..environment import CheckResult, EpisodeResult, RolloutVerifierReport, score_rollout
from ..integrity import IntegrityError, tree_identity
from ..manifest import TaskManifest


class EpisodeTimeoutError(TimeoutError):
    """Raised when the trusted episode-wide deadline is exhausted."""


_RUBRIC_COMPONENTS = (
    "design_rules",
    "patch_quality",
    "regression_safety",
    "responsive_rules",
    "task_tests",
)

EpisodeEventCallback = Callable[[Mapping[str, Any]], None]


class FrontendEnvironment:
    """One disposable, network-disabled frontend coding episode."""

    # Snapshot limits apply before a repository enters the container. They
    # bound host memory/disk use from an accidentally huge or hostile Git tree.
    _SNAPSHOT_FILE_LIMIT = 20_000
    _SNAPSHOT_BLOB_BYTES = 64 * 1024 * 1024
    _SNAPSHOT_TOTAL_BYTES = 2 * 1024 * 1024 * 1024

    def __init__(self) -> None:
        self._manifest: TaskManifest | None = None
        self._workspace: Path | None = None
        self._temporary_root: Path | None = None
        self._task_root: Path | None = None
        self._backend = "docker"
        self._image = ""
        self._verifier_identity: dict[str, Any] | None = None
        self._tool_calls = 0
        self._tool_calls_by_name: dict[str, int] = {}
        self._max_output_chars = 12_000
        self._install_result: CheckResult | None = None
        self._check_results: dict[str, CheckResult] = {}
        self._started_at: float | None = None
        self._deadline: float | None = None
        self._latest_diff = ""
        self._last_result: EpisodeResult | None = None
        self._episode_callback: EpisodeEventCallback | None = None
        self._episode_id: str | None = None

    def _set_episode_callback(
        self, callback: EpisodeEventCallback | None
    ) -> None:
        """Attach GRPO telemetry without adding a policy-visible tool method."""

        self._episode_callback = callback

    def reset(self, **task_row: Any) -> str:
        """Materialize the task row and return the policy's first observation."""

        try:
            manifest, revision = self._initialize(task_row)
            if self._episode_callback is not None:
                # Expose only bounded episode structure. Prompts, source text,
                # model reasoning, and verifier content stay private.
                self._episode_callback(
                    {
                        "type": "episode_started",
                        "episode_id": self._episode_id,
                        "task_id": manifest.task_id,
                        # A manifest group is a task family, not a batch of
                        # sibling completions sampled by GRPO.
                        "task_family_id": manifest.group_id,
                    }
                )
            install = manifest.runtime_commands.get("install", "").strip()
            self._install_result = self._run_named_check("install", install)
            if self._install_result.timed_out:
                self._finish_timeout(self._install_result, "install")
            install_status = self._install_result.status.replace("_", " ")
            return (
                f"Task: {manifest.instruction}\n"
                f"Source: {manifest.source_id}@{revision}\n"
                f"Install: {install_status}. Network access is disabled. "
                "Inspect the repository, apply a focused patch, and run named checks."
            )
        except Exception:
            self._cleanup()
            raise

    @property
    def last_result(self) -> EpisodeResult | None:
        """Most recent finalized result, retained after disposable cleanup."""

        return self._last_result

    def list_files(self, path: str = ".") -> str:
        """List editable repository files below a relative directory.

        Args:
            path: Repository-relative directory to inspect. Use ``.`` for the
                editable repository root.

        Returns:
            A bounded newline-delimited list of repository-relative file paths.
        """

        self._use_tool("list_files")
        root = self._safe_path(path)
        if not root.is_dir():
            return f"not a directory: {path}"
        files: list[str] = []
        for candidate in sorted(root.rglob("*")):
            self._check_deadline()
            if candidate.is_file() and ".git" not in candidate.parts:
                files.append(candidate.relative_to(self._require_workspace()).as_posix())
            if len(files) >= 500:
                files.append("... truncated at 500 files")
                break
        return self._bounded("\n".join(files))

    def read_file(self, path: str, start: int = 1, end: int = 200) -> str:
        """Read an inclusive, one-indexed line range from a UTF-8 text file.

        Args:
            path: Repository-relative path to the text file.
            start: First one-indexed line to return.
            end: Last one-indexed line to return, inclusive.

        Returns:
            Bounded numbered lines, or a short validation error visible to the
            policy.
        """

        self._use_tool("read_file")
        if start < 1 or end < start or end - start > 500:
            return "invalid range: use 1-based lines and request at most 501 lines"
        candidate = self._safe_path(path)
        if not candidate.is_file():
            return f"not a file: {path}"
        if candidate.stat().st_size > 1_000_000:
            return "file is larger than the 1 MB tool limit"
        try:
            lines = candidate.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            return "binary or non-UTF-8 file"
        self._check_deadline()
        selected = [f"{number}: {lines[number - 1]}" for number in range(start, min(end, len(lines)) + 1)]
        return self._bounded("\n".join(selected))

    def search_code(self, query: str, path: str = ".") -> str:
        """Search text files for a literal query and return bounded line matches.

        Args:
            query: Literal case-insensitive text to find.
            path: Repository-relative file or directory to search.

        Returns:
            Bounded ``path:line: text`` matches, or ``no matches``.
        """

        self._use_tool("search_code")
        if not query or len(query) > 200:
            return "query must contain between 1 and 200 characters"
        root = self._safe_path(path)
        matches: list[str] = []
        lowered = query.casefold()
        candidates = [root] if root.is_file() else sorted(root.rglob("*"))
        for candidate in candidates:
            self._check_deadline()
            if not candidate.is_file() or ".git" in candidate.parts or candidate.stat().st_size > 1_000_000:
                continue
            try:
                for number, line in enumerate(candidate.read_text(encoding="utf-8").splitlines(), 1):
                    if lowered in line.casefold():
                        relative = candidate.relative_to(self._require_workspace()).as_posix()
                        matches.append(f"{relative}:{number}: {line[:300]}")
                        if len(matches) >= 200:
                            matches.append("... truncated at 200 matches")
                            return self._bounded("\n".join(matches))
            except (UnicodeDecodeError, OSError):
                continue
        return self._bounded("\n".join(matches) if matches else "no matches")

    def apply_patch(self, patch: str) -> str:
        """Apply a unified diff after path and Git safety checks.

        Args:
            patch: Complete unified diff whose paths stay inside the editable
                repository.

        Returns:
            A bounded success message or the reason the patch was rejected.
        """

        self._use_tool("apply_patch")
        try:
            applied, message = self._apply_unified_diff(patch)
            if applied:
                self._latest_diff = self._capture_unified_diff()
            return self._bounded(message)
        except subprocess.TimeoutExpired as error:
            result = CheckResult(
                name="patch",
                configured=True,
                status="timed_out",
                passed=False,
                returncode=None,
                timed_out=True,
                duration_seconds=0.0,
                stdout=self._truncate(error.stdout if isinstance(error.stdout, str) else ""),
                stderr=self._truncate(error.stderr if isinstance(error.stderr, str) else ""),
            )
            self._check_results["patch"] = result
            reason = (
                "episode_timeout"
                if self._deadline is not None and time.monotonic() >= self._deadline
                else "patch_timeout"
            )
            self._complete_result(reason)
            self._cleanup()
            raise EpisodeTimeoutError("patch application exceeded its trusted timeout") from error
        except EpisodeTimeoutError:
            raise
        except Exception:
            self._cleanup()
            raise

    def run_check(self, check: str) -> str:
        """Run one trusted check declared by the task author.

        Args:
            check: Check name: ``build``, ``tests``, or ``browserTests``.

        Returns:
            A bounded pass/fail summary with captured command output.
        """

        self._use_tool("run_check")
        if check not in {"build", "tests", "browserTests"}:
            return "unknown check; choose build, tests, or browserTests"
        command = self._require_manifest().runtime_commands.get(check, "").strip()
        if not command:
            return f"check is not configured: {check}"
        result = self._run_named_check(check, command)
        if result.timed_out:
            self._finish_timeout(result, check)
        label = "passed" if result.passed else f"failed (exit {result.returncode})"
        return self._bounded(f"{check} {label}\n{result.stdout}\n{result.stderr}")

    def get_reward(self) -> float:
        """Delegate the TRL reward hook to the structured finalization path."""

        return self._finalize().reward

    def _finalize(self) -> EpisodeResult:
        """Evaluate the current patch, capture evidence, and clean the workspace."""

        if self._manifest is None:
            if self._last_result is not None:
                return self._last_result
            raise RuntimeError("environment reset must run before finalization")
        manifest = self._manifest
        try:
            self._check_deadline()
            self._prime_check_results(manifest)
            try:
                self._latest_diff = self._capture_unified_diff()
            except subprocess.TimeoutExpired as error:
                self._check_results["diff"] = CheckResult(
                    name="diff",
                    configured=True,
                    status="timed_out",
                    passed=False,
                    returncode=None,
                    timed_out=True,
                    duration_seconds=0.0,
                    stdout=self._truncate(
                        error.stdout if isinstance(error.stdout, str) else ""
                    ),
                    stderr=self._truncate(
                        error.stderr if isinstance(error.stderr, str) else ""
                    ),
                )
                reason = (
                    "episode_timeout"
                    if self._deadline is not None and time.monotonic() >= self._deadline
                    else "diff_capture_timeout"
                )
                return self._complete_result(reason)
            except EpisodeTimeoutError:
                raise
            except (OSError, RuntimeError) as error:
                self._check_results["diff"] = CheckResult(
                    name="diff",
                    configured=True,
                    status="failed",
                    passed=False,
                    returncode=1,
                    timed_out=False,
                    duration_seconds=0.0,
                    stdout="",
                    stderr=self._truncate(str(error)),
                )
                return self._complete_result("diff_capture_failed")

            install = self._install_result
            if install is None:
                install = self._run_named_check(
                    "install", manifest.runtime_commands.get("install", "").strip()
                )
                self._install_result = install
            gate_reason = self._failed_check_reason("install", install, "install_failed")
            if gate_reason:
                return self._complete_result(gate_reason)

            # Gate order is deliberate: cheap deterministic regressions run
            # before browser work, and the hidden verifier is always last.
            checks = (
                ("build", manifest.runtime_commands["build"], "build_failed", False),
                ("tests", manifest.runtime_commands["tests"], "regression_failed", False),
                (
                    "browserTests",
                    manifest.runtime_commands.get("browserTests", "").strip(),
                    "browser_tests_failed",
                    False,
                ),
                ("verifier", manifest.verifier_command or "", "verifier_failed", True),
            )
            for name, command, failed_reason, include_verifier in checks:
                if include_verifier:
                    try:
                        report_path = self._safe_path(
                            manifest.verifier_report_path or ""
                        )
                        if report_path.exists():
                            # The editable policy can write inside the workspace.
                            # Remove any forged report before mounting/running the
                            # trusted verifier bundle.
                            if not report_path.is_file():
                                return self._complete_result("invalid_verifier_report")
                            report_path.unlink()
                    except (OSError, RuntimeError):
                        return self._complete_result("invalid_verifier_report")
                result = self._run_named_check(
                    name, command, include_verifier=include_verifier
                )
                gate_reason = self._failed_check_reason(name, result, failed_reason)
                if gate_reason:
                    return self._complete_result(gate_reason)

            try:
                report_path = self._safe_path(manifest.verifier_report_path or "")
                payload = json.loads(report_path.read_text(encoding="utf-8"))
                raw_rates = {
                    "regression_safety": float(payload["regression_pass_rate"]),
                    "task_tests": float(payload["task_pass_rate"]),
                    "responsive_rules": float(payload["responsive_pass_rate"]),
                    "design_rules": float(payload["design_rule_pass_rate"]),
                    "patch_quality": float(payload["code_quality_pass_rate"]),
                }
                report = RolloutVerifierReport(
                    build_passed=True,
                    regression_pass_rate=raw_rates["regression_safety"],
                    task_pass_rate=raw_rates["task_tests"],
                    responsive_pass_rate=raw_rates["responsive_rules"],
                    design_rule_pass_rate=raw_rates["design_rules"],
                    code_quality_pass_rate=raw_rates["patch_quality"],
                )
                weights = {
                    "regression_safety": manifest.reward_weights["regressionSafety"],
                    "task_tests": manifest.reward_weights["taskTests"],
                    "responsive_rules": manifest.reward_weights["responsiveRules"],
                    "design_rules": manifest.reward_weights["designRules"],
                    "patch_quality": manifest.reward_weights["patchQuality"],
                }
                reward = score_rollout(report, weights)
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                return self._complete_result("invalid_verifier_report")
            return self._complete_result(
                reward.gate_reason,
                raw_rates=raw_rates,
                weighted_signals=reward.signals,
                reward=reward.total,
            )
        except EpisodeTimeoutError:
            if self._last_result is not None:
                return self._last_result
            return self._complete_result("episode_timeout")
        finally:
            self._cleanup()

    def _evaluate_patch(self, task_row: Mapping[str, Any], patch: str) -> EpisodeResult:
        """Replay a supplied diff into a fresh locked snapshot and evaluate it.

        Applying this externally supplied patch is evaluation setup, not a
        policy action, so it never increments ``tool_call_count``.
        """

        try:
            manifest, _ = self._initialize(task_row)
            self._prime_check_results(manifest)
            started = time.monotonic()
            applied, message = self._apply_unified_diff(patch)
            duration = round(time.monotonic() - started, 6)
            patch_check = CheckResult(
                name="patch",
                configured=True,
                status="passed" if applied else "failed",
                passed=applied,
                returncode=0 if applied else 1,
                timed_out=False,
                duration_seconds=duration,
                stdout=message if applied else "",
                stderr="" if applied else message,
            )
            self._check_results["patch"] = patch_check
            if not applied:
                result = self._complete_result("patch_apply_failed")
                self._cleanup()
                return result
            self._latest_diff = self._capture_unified_diff()
            return self._finalize()
        except Exception:
            self._cleanup()
            raise

    def _initialize(self, task_row: Mapping[str, Any]) -> tuple[TaskManifest, str]:
        self._cleanup()
        manifest_payload, task_root = self._load_manifest(task_row)
        manifest = TaskManifest.from_mapping(manifest_payload)
        if manifest.version != "1.0":
            raise RuntimeError("executable RL requires a compiled task manifest at version 1.0")
        source_path_value = task_row.get("source_path")
        if not source_path_value:
            raise RuntimeError("compiled task row is missing source_path")
        source_path = Path(str(source_path_value)).expanduser().resolve()
        if not source_path.is_dir():
            raise RuntimeError(f"task source_path is not a directory: {source_path}")

        # Task bundles may live beside a source repository in an authoring
        # monorepo. The verifier can be a sibling, but never part of the project
        # snapshot exported into the policy-visible episode.
        self._validate_verifier_boundary(manifest, task_root, source_path)

        # Never evaluate a moving branch or an un-fingerprinted directory. This
        # is what makes the same task row replayable across all candidate arms.
        revision = manifest.starting_revision
        if revision == "locked":
            revision = str(task_row.get("source_revision", ""))
        if not revision:
            raise RuntimeError("starting revision was not resolved by compile")
        # V1 repository declarations are Git-backed. A former `tree:<sha>`
        # escape hatch copied live directories without an independently stored
        # snapshot, so accepting its digest would overstate reproducibility.
        if not (source_path / ".git").exists():
            raise RuntimeError(
                "V1 executable evaluation requires a Git-backed repository source"
            )
        if re.fullmatch(r"(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})", revision) is None:
            raise RuntimeError(
                "evaluation requires source_revision to be a full immutable Git commit"
            )

        self._backend = str(task_row.get("environment_backend", "docker"))
        if self._backend not in {"docker", "podman"}:
            raise RuntimeError("environment_backend must be docker or podman")
        if shutil.which(self._backend) is None:
            raise RuntimeError(f"{self._backend} is required for frontend RL episodes")
        # Evaluation plans replace mutable image tags with an immutable image
        # ID or repository digest. GRPO rows created directly by compile retain
        # the tag because training has its own Doctor preflight.
        self._image = str(
            task_row.get("environment_image_identity")
            or task_row.get("environment_image", "")
        )
        if not self._image:
            raise RuntimeError("compiled task row is missing environment_image")
        verifier_identity = task_row.get("verifier_identity")
        if verifier_identity is not None:
            if not isinstance(verifier_identity, Mapping):
                raise RuntimeError("compiled verifier_identity must be a mapping")
            self._verifier_identity = dict(verifier_identity)
        else:
            self._verifier_identity = None
        self._max_output_chars = int(task_row.get("max_tool_output_chars", 12_000))

        self._manifest = manifest
        self._task_root = task_root
        self._episode_id = uuid4().hex[:12]
        self._tool_calls = 0
        self._tool_calls_by_name = {}
        self._install_result = None
        self._check_results = {}
        self._latest_diff = ""
        self._last_result = None
        self._started_at = time.monotonic()
        self._deadline = self._started_at + manifest.episode_timeout_seconds

        # Track the exact temporary root separately; cleanup later verifies this
        # relationship before performing any recursive removal.
        temporary_root = Path(tempfile.mkdtemp(prefix="autotrainer-episode-"))
        workspace = temporary_root / "workspace"
        self._temporary_root = temporary_root.resolve()
        self._workspace = workspace
        try:
            self._materialize(
                source_path,
                revision,
                workspace,
                manifest.working_directory,
            )
            self._check_deadline()
        except Exception:
            self._cleanup()
            raise
        return manifest, revision

    def _prime_check_results(self, manifest: TaskManifest) -> None:
        commands = {
            "install": manifest.runtime_commands.get("install", "").strip(),
            "build": manifest.runtime_commands.get("build", "").strip(),
            "tests": manifest.runtime_commands.get("tests", "").strip(),
            "browserTests": manifest.runtime_commands.get("browserTests", "").strip(),
            "verifier": (manifest.verifier_command or "").strip(),
        }
        for name, command in commands.items():
            if name in self._check_results:
                continue
            configured = bool(command)
            self._check_results[name] = CheckResult(
                name=name,
                configured=configured,
                status="not_run" if configured else "not_configured",
                passed=None,
                returncode=None,
                timed_out=False,
                duration_seconds=0.0,
                stdout="",
                stderr="",
            )

    def _run_named_check(
        self, name: str, command: str, *, include_verifier: bool = False
    ) -> CheckResult:
        if not command:
            result = CheckResult(
                name=name,
                configured=False,
                status="not_configured",
                passed=None,
                returncode=None,
                timed_out=False,
                duration_seconds=0.0,
                stdout="",
                stderr="",
            )
            self._check_results[name] = result
            return result
        started = time.monotonic()
        try:
            completed = self._run_container_command(
                command, include_verifier=include_verifier
            )
        except EpisodeTimeoutError:
            raise
        except subprocess.TimeoutExpired as error:
            duration = round(time.monotonic() - started, 6)
            stdout = error.stdout if isinstance(error.stdout, str) else ""
            stderr = error.stderr if isinstance(error.stderr, str) else ""
            result = CheckResult(
                name=name,
                configured=True,
                status="timed_out",
                passed=False,
                returncode=None,
                timed_out=True,
                duration_seconds=duration,
                stdout=self._truncate(stdout),
                stderr=self._truncate(stderr),
            )
            self._check_results[name] = result
            return result
        except (OSError, RuntimeError) as error:
            duration = round(time.monotonic() - started, 6)
            result = CheckResult(
                name=name,
                configured=True,
                status="failed",
                passed=False,
                returncode=1,
                timed_out=False,
                duration_seconds=duration,
                stdout="",
                stderr=self._truncate(str(error)),
            )
            self._check_results[name] = result
            return result
        duration = round(time.monotonic() - started, 6)
        passed = completed.returncode == 0
        result = CheckResult(
            name=name,
            configured=True,
            status="passed" if passed else "failed",
            passed=passed,
            returncode=completed.returncode,
            timed_out=False,
            duration_seconds=duration,
            stdout=self._truncate(completed.stdout or ""),
            stderr=self._truncate(completed.stderr or ""),
        )
        self._check_results[name] = result
        return result

    def _failed_check_reason(
        self, name: str, result: CheckResult, failed_reason: str
    ) -> str | None:
        if not result.configured:
            return None
        if result.timed_out:
            if self._deadline is not None and time.monotonic() >= self._deadline:
                return "episode_timeout"
            timeout_names = {
                "browserTests": "browser_tests_timeout",
                "patch": "patch_timeout",
            }
            return timeout_names.get(name, f"{name}_timeout")
        return None if result.passed else failed_reason

    def _complete_result(
        self,
        hard_gate_reason: str | None,
        *,
        raw_rates: Mapping[str, float] | None = None,
        weighted_signals: Mapping[str, float] | None = None,
        reward: float = 0.0,
    ) -> EpisodeResult:
        manifest = self._manifest
        task_id = manifest.task_id if manifest is not None else ""
        result = EpisodeResult(
            task_id=task_id,
            hard_gate_reason=hard_gate_reason,
            raw_verifier_rates=dict(raw_rates or {}),
            weighted_signals=dict(weighted_signals or {}),
            reward=round(float(reward), 4),
            checks=dict(self._check_results),
            tool_call_count=self._tool_calls,
            elapsed_seconds=self._elapsed_seconds(),
            unified_diff=self._latest_diff,
        )
        self._last_result = result
        if self._episode_callback is not None:
            # Only verifier scores cross this boundary. The editable patch,
            # prompts, check output, and tool observations remain private.
            self._episode_callback(
                {
                    "type": "episode_scored",
                    "episode_id": self._episode_id,
                    "task_id": result.task_id,
                    "reward": result.reward,
                    "hard_gate_passed": not result.gated,
                    "gate_reason": result.hard_gate_reason,
                    "tool_call_count": result.tool_call_count,
                    "tool_calls_by_name": dict(sorted(self._tool_calls_by_name.items())),
                    "elapsed_seconds": result.elapsed_seconds,
                    "changed_file_count": sum(
                        line.startswith("diff --git ")
                        for line in result.unified_diff.splitlines()
                    ),
                    "rubric": {
                        name: float(result.raw_verifier_rates.get(name, 0.0))
                        for name in _RUBRIC_COMPONENTS
                    },
                }
            )
        return result

    def _finish_timeout(self, result: CheckResult, name: str) -> None:
        reason = self._failed_check_reason(name, result, f"{name}_timeout")
        self._complete_result(reason or f"{name}_timeout")
        self._cleanup()
        raise EpisodeTimeoutError(f"{name} exceeded its trusted timeout")

    def _elapsed_seconds(self) -> float:
        if self._started_at is None:
            return 0.0
        return round(max(0.0, time.monotonic() - self._started_at), 6)

    def _check_deadline(self) -> None:
        if self._deadline is None or time.monotonic() < self._deadline:
            return
        self._complete_result("episode_timeout")
        self._cleanup()
        raise EpisodeTimeoutError("episodeTimeoutSeconds exceeded")

    def _timeout_for(self, maximum_seconds: float) -> float:
        self._check_deadline()
        if self._deadline is None:
            return maximum_seconds
        remaining = self._deadline - time.monotonic()
        if remaining <= 0:
            self._check_deadline()
        return max(0.001, min(float(maximum_seconds), remaining))

    def _apply_unified_diff(self, patch: str) -> tuple[bool, str]:
        if not isinstance(patch, str) or not patch.strip() or len(patch) > 200_000:
            return False, "patch must contain between 1 and 200000 characters"
        try:
            self._validate_patch_paths(patch)
        except (RuntimeError, ValueError) as error:
            return False, f"rejected patch: {error}"
        workspace = self._require_workspace()
        check = self._run_git(
            workspace,
            "apply",
            "--check",
            "--whitespace=error-all",
            "-",
            input_value=patch,
        )
        if check.returncode:
            return False, self._truncate(f"patch check failed:\n{check.stderr or check.stdout}")
        applied = self._run_git(
            workspace,
            "apply",
            "--whitespace=nowarn",
            "-",
            input_value=patch,
        )
        if applied.returncode:
            return False, self._truncate(
                f"patch apply failed:\n{applied.stderr or applied.stdout}"
            )
        self._check_deadline()
        return True, "patch applied"

    def _capture_unified_diff(self) -> str:
        workspace = self._require_workspace()
        intent = self._run_git(
            workspace,
            "add",
            "--intent-to-add",
            "--all",
        )
        if intent.returncode:
            raise RuntimeError(f"could not prepare unified diff: {intent.stderr.strip()}")
        diff = self._run_git(
            workspace,
            "diff",
            "--binary",
            "--no-ext-diff",
            "HEAD",
            "--",
        )
        if diff.returncode:
            raise RuntimeError(f"could not capture unified diff: {diff.stderr.strip()}")
        self._check_deadline()
        return diff.stdout

    def _load_manifest(self, row: Mapping[str, Any]) -> tuple[Mapping[str, Any], Path]:
        if isinstance(row.get("manifest"), Mapping):
            root = Path(str(row.get("task_root", "."))).expanduser().resolve()
            return row["manifest"], root
        if isinstance(row.get("manifest_json"), str):
            root = Path(str(row.get("task_root", "."))).expanduser().resolve()
            return json.loads(row["manifest_json"]), root
        if row.get("manifest_path"):
            path = Path(str(row["manifest_path"])).expanduser().resolve()
            return json.loads(path.read_text(encoding="utf-8")), path.parent
        if row.get("version"):
            root = Path(str(row.get("task_root", "."))).expanduser().resolve()
            return row, root
        raise RuntimeError("task row must contain manifest, manifest_json, manifest_path, or manifest fields")

    @staticmethod
    def _working_directory_path(value: str) -> Path:
        relative = Path(value or ".")
        if relative.is_absolute() or ".." in relative.parts or ".git" in relative.parts:
            raise RuntimeError(
                "runtime.workingDirectory must stay inside the locked source and outside .git"
            )
        return relative

    def _validate_verifier_boundary(
        self,
        manifest: TaskManifest,
        task_root: Path,
        source: Path,
    ) -> None:
        relative = self._working_directory_path(manifest.working_directory)
        source_root = source.resolve()
        editable_source = (source_root / relative).resolve()
        try:
            editable_source.relative_to(source_root)
        except ValueError as error:
            raise RuntimeError("runtime.workingDirectory escapes the locked source") from error
        bundle_value = manifest.verifier_bundle
        if not bundle_value:
            return
        bundle = Path(bundle_value)
        bundle = bundle.resolve() if bundle.is_absolute() else (task_root / bundle).resolve()
        for child, parent in (
            (bundle, editable_source),
            (editable_source, bundle),
        ):
            try:
                child.relative_to(parent)
            except ValueError:
                continue
            raise RuntimeError(
                "verifier bundle must not overlap the policy-visible working directory"
            )

    def _materialize(
        self,
        source: Path,
        revision: str,
        destination: Path,
        working_directory: str,
    ) -> None:
        relative = self._working_directory_path(working_directory)
        if not (source / ".git").exists():
            raise RuntimeError(
                "V1 executable evaluation requires a Git-backed repository source"
            )
        files = self._export_locked_git_tree(
            source,
            revision,
            destination,
            relative,
        )

        # Never carry the source repository's object database into an episode.
        # A fresh parentless baseline makes deleted or sibling verifier blobs
        # unavailable even to policy code that invokes `git cat-file` itself.
        self._initialize_baseline_repository(destination, files)

    @staticmethod
    def _isolated_git_environment() -> dict[str, str]:
        environment = dict(os.environ)
        # Git accepts configuration and object-store overrides through many
        # environment variables. Clear them so a user's templates, filters,
        # hooks, alternates, or replacement refs cannot change trusted setup.
        for key in list(environment):
            if key.startswith("GIT_CONFIG_") or key in {
                "GIT_ALTERNATE_OBJECT_DIRECTORIES",
                "GIT_COMMON_DIR",
                "GIT_DEFAULT_HASH",
                "GIT_DIR",
                "GIT_INDEX_FILE",
                "GIT_OBJECT_DIRECTORY",
                "GIT_QUARANTINE_PATH",
                "GIT_REPLACE_REF_BASE",
                "GIT_TEMPLATE_DIR",
                "GIT_WORK_TREE",
            }:
                environment.pop(key, None)
        environment["GIT_CONFIG_NOSYSTEM"] = "1"
        environment["GIT_CONFIG_GLOBAL"] = os.devnull
        # A local partial clone may otherwise contact its promisor remote when
        # cat-file encounters a missing blob. Snapshot export must stay local.
        environment["GIT_NO_LAZY_FETCH"] = "1"
        environment["GIT_NO_REPLACE_OBJECTS"] = "1"
        return environment

    def _run_git(
        self,
        repository: Path,
        *arguments: str,
        binary: bool = False,
        input_value: str | None = None,
        timeout: float = 30,
    ) -> subprocess.CompletedProcess[Any]:
        return subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={repository.resolve().as_posix()}",
                "-C",
                str(repository),
                *arguments,
            ],
            input=None if binary else input_value,
            capture_output=True,
            text=not binary,
            env=self._isolated_git_environment(),
            timeout=self._timeout_for(timeout),
            check=False,
        )

    def _export_locked_git_tree(
        self,
        source: Path,
        revision: str,
        destination: Path,
        relative: Path,
    ) -> list[tuple[str, str]]:
        relative_posix = relative.as_posix()
        if relative != Path("."):
            object_type = self._run_git(
                source,
                "cat-file",
                "-t",
                f"{revision}:{relative_posix}",
            )
            if object_type.returncode or object_type.stdout.strip() != "tree":
                raise RuntimeError(
                    "runtime.workingDirectory does not exist as a directory "
                    f"at the locked revision: {relative_posix}"
                )

        arguments = ["ls-tree", "-r", "-z", "-l", "--full-tree", revision]
        if relative != Path("."):
            # Literal pathspecs prevent metacharacters in a project directory
            # from selecting files outside the intended subtree.
            arguments.extend(["--", f":(literal){relative_posix}"])
        listing = self._run_git(source, *arguments, binary=True, timeout=120)
        if listing.returncode:
            detail = listing.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"cannot read locked source tree: {detail}")

        destination.mkdir(parents=True, exist_ok=True)
        files: list[tuple[str, str]] = []
        file_keys: set[str] = set()
        directory_keys: set[str] = set()
        total_bytes = 0
        for raw_entry in listing.stdout.split(b"\0"):
            if not raw_entry:
                continue
            metadata, separator, raw_path = raw_entry.partition(b"\t")
            if not separator:
                raise RuntimeError("locked source tree contains an invalid Git entry")
            try:
                mode, object_type, object_id, size_value = metadata.decode("ascii").split()
                path = raw_path.decode("utf-8")
            except (UnicodeDecodeError, ValueError) as error:
                raise RuntimeError(
                    "locked source paths and tree metadata must be valid UTF-8"
                ) from error
            if object_type != "blob" or mode not in {"100644", "100755"}:
                raise RuntimeError(
                    "editable task snapshots reject symlinks, gitlinks, "
                    f"and special entries: {path}"
                )
            try:
                blob_bytes = int(size_value)
            except ValueError as error:
                raise RuntimeError(f"locked source blob has no valid size: {path}") from error
            if blob_bytes > self._SNAPSHOT_BLOB_BYTES:
                raise RuntimeError(
                    f"locked source blob exceeds the snapshot limit: {path} ({blob_bytes} bytes)"
                )
            if len(files) + 1 > self._SNAPSHOT_FILE_LIMIT:
                raise RuntimeError(
                    f"locked source exceeds the {self._SNAPSHOT_FILE_LIMIT} file snapshot limit"
                )
            total_bytes += blob_bytes
            if total_bytes > self._SNAPSHOT_TOTAL_BYTES:
                raise RuntimeError(
                    "locked source exceeds the "
                    f"{self._SNAPSHOT_TOTAL_BYTES} byte snapshot limit"
                )
            target = self._export_target(
                destination,
                path,
                relative,
                file_keys,
                directory_keys,
            )
            blob = self._run_git(source, "cat-file", "blob", object_id, binary=True)
            if blob.returncode:
                detail = blob.stderr.decode("utf-8", errors="replace").strip()
                raise RuntimeError(f"cannot read locked blob {object_id}: {detail}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(blob.stdout)
            if mode == "100755":
                target.chmod(target.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
            files.append((path, mode))
        if not files:
            raise RuntimeError("runtime.workingDirectory contains no tracked files")
        return files

    @staticmethod
    def _export_target(
        destination: Path,
        value: str,
        relative: Path,
        file_keys: set[str],
        directory_keys: set[str],
    ) -> Path:
        if not value or "\\" in value or "\0" in value:
            raise RuntimeError(f"locked source contains a non-portable path: {value!r}")
        path = PurePosixPath(value)
        if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
            raise RuntimeError(f"locked source path escapes its snapshot: {value!r}")
        prefix = PurePosixPath(relative.as_posix()).parts if relative != Path(".") else ()
        if prefix and path.parts[: len(prefix)] != prefix:
            raise RuntimeError(f"locked source entry escaped workingDirectory: {value!r}")

        reserved = {"CON", "PRN", "AUX", "NUL"} | {
            f"{name}{number}"
            for name in ("COM", "LPT")
            for number in range(1, 10)
        }
        normalized_parts: list[str] = []
        for part in path.parts:
            normalized = unicodedata.normalize("NFC", part).casefold()
            stem = part.split(".", 1)[0].upper()
            if (
                normalized == ".git"
                or part.endswith((" ", "."))
                or ":" in part
                or stem in reserved
            ):
                raise RuntimeError(f"locked source contains a non-portable path: {value!r}")
            normalized_parts.append(normalized)

        for index in range(1, len(normalized_parts)):
            directory_key = "/".join(normalized_parts[:index])
            if directory_key in file_keys:
                raise RuntimeError(
                    f"locked source has a case or Unicode path collision: {value!r}"
                )
            directory_keys.add(directory_key)
        file_key = "/".join(normalized_parts)
        if file_key in file_keys or file_key in directory_keys:
            raise RuntimeError(f"locked source has a case or Unicode path collision: {value!r}")
        file_keys.add(file_key)

        target = (destination / Path(*path.parts)).resolve()
        try:
            target.relative_to(destination.resolve())
        except ValueError as error:
            raise RuntimeError(f"locked source path escapes its snapshot: {value!r}") from error
        return target

    def _initialize_baseline_repository(
        self,
        destination: Path,
        files: list[tuple[str, str]],
    ) -> None:
        empty_template = destination.parent / "empty-git-template"
        empty_template.mkdir(exist_ok=True)
        environment = self._isolated_git_environment()
        initialized = subprocess.run(
            [
                "git",
                "init",
                "--quiet",
                f"--template={empty_template}",
                str(destination),
            ],
            capture_output=True,
            text=True,
            env=environment,
            timeout=self._timeout_for(30),
            check=False,
        )
        if initialized.returncode:
            raise RuntimeError(f"failed to initialize task snapshot: {initialized.stderr.strip()}")

        # The highest-precedence attributes file neutralizes repository-provided
        # clean/smudge, encoding, and ident transforms. The baseline indexes the
        # exact bytes exported from the locked tree and never executes filters.
        info = destination / ".git" / "info"
        info.mkdir(exist_ok=True)
        (info / "attributes").write_text(
            "* -text -filter -ident -working-tree-encoding\n"
            "**/* -text -filter -ident -working-tree-encoding\n",
            encoding="utf-8",
        )

        for path, mode in files:
            hashed = self._run_git(
                destination,
                "hash-object",
                "-w",
                "--no-filters",
                "--",
                path,
            )
            if hashed.returncode:
                raise RuntimeError(
                    f"failed to hash task snapshot file {path}: {hashed.stderr.strip()}"
                )
            indexed = self._run_git(
                destination,
                "update-index",
                "--add",
                "--cacheinfo",
                mode,
                hashed.stdout.strip(),
                path,
            )
            if indexed.returncode:
                raise RuntimeError(
                    f"failed to index task snapshot file {path}: {indexed.stderr.strip()}"
                )

        tree = self._run_git(destination, "write-tree")
        if tree.returncode:
            raise RuntimeError(f"failed to write task snapshot tree: {tree.stderr.strip()}")
        commit_environment = self._isolated_git_environment()
        commit_environment.update(
            {
                "GIT_AUTHOR_NAME": "AutoTrainer",
                "GIT_AUTHOR_EMAIL": "autotrainer@localhost",
                "GIT_AUTHOR_DATE": "2000-01-01T00:00:00+00:00",
                "GIT_COMMITTER_NAME": "AutoTrainer",
                "GIT_COMMITTER_EMAIL": "autotrainer@localhost",
                "GIT_COMMITTER_DATE": "2000-01-01T00:00:00+00:00",
            }
        )
        committed = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={destination.resolve().as_posix()}",
                "-C",
                str(destination),
                "commit-tree",
                tree.stdout.strip(),
            ],
            input="Locked evaluation snapshot\n",
            capture_output=True,
            text=True,
            env=commit_environment,
            timeout=self._timeout_for(30),
            check=False,
        )
        if committed.returncode:
            raise RuntimeError(f"failed to commit task snapshot: {committed.stderr.strip()}")
        updated = self._run_git(
            destination,
            "update-ref",
            "HEAD",
            committed.stdout.strip(),
        )
        if updated.returncode:
            raise RuntimeError(f"failed to lock task snapshot: {updated.stderr.strip()}")

        status = self._run_git(destination, "status", "--porcelain")
        if status.returncode or status.stdout:
            raise RuntimeError(
                "fresh task snapshot baseline is not clean: "
                f"{status.stderr.strip() or status.stdout.strip()}"
            )

    @staticmethod
    def _is_link_or_reparse(path: Path) -> bool:
        if path.is_symlink():
            return True
        try:
            attributes = getattr(path.lstat(), "st_file_attributes", 0)
        except OSError:
            return False
        return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))

    @classmethod
    def _reject_workspace_escape_links(cls, root: Path) -> None:
        root = root.resolve()
        for current, directory_names, file_names in os.walk(root, followlinks=False):
            base = Path(current)
            safe_directories: list[str] = []
            for name in directory_names:
                candidate = base / name
                if not cls._is_link_or_reparse(candidate):
                    safe_directories.append(name)
                    continue
                try:
                    candidate.resolve().relative_to(root)
                except (OSError, ValueError) as error:
                    raise RuntimeError(
                        "episode contains a link that escapes the editable workspace"
                    ) from error
                # Do not descend through even an internal junction. On Windows,
                # os.walk may otherwise traverse a reparse point despite
                # followlinks=False, which can produce cycles or duplicate scans.
            directory_names[:] = safe_directories
            for name in file_names:
                candidate = base / name
                if not cls._is_link_or_reparse(candidate):
                    continue
                try:
                    candidate.resolve().relative_to(root)
                except (OSError, ValueError) as error:
                    raise RuntimeError(
                        "episode contains a link that escapes the editable workspace"
                    ) from error

    def _run_container_command(
        self,
        command: str,
        *,
        include_verifier: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        if not command:
            return subprocess.CompletedProcess([], 0, "", "")
        manifest = self._require_manifest()
        workspace = self._require_workspace()
        working = self._safe_path(manifest.working_directory)
        relative_working = working.relative_to(workspace).as_posix()
        # Package scripts and unified diffs can create links. Reject links that
        # escape the workspace before a verifier mount gives
        # `/autotrainer-verifier` any meaning inside the container.
        self._reject_workspace_escape_links(workspace)
        # The workspace is writable because the policy must edit it. Everything
        # else is constrained: no network, no capabilities, bounded CPU/RAM/PIDs,
        # a no-new-privileges bit, and an ephemeral tmpfs.
        arguments = [
            self._backend,
            "run",
            "--rm",
            "--network",
            "none",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            "512",
            "--memory",
            "12g",
            "--cpus",
            "8",
            "--shm-size",
            "1g",
            "--tmpfs",
            "/tmp:rw,nosuid,size=1g",
            # Read-only Git metadata still supports status/diff when Git skips
            # optional index refresh locks.
            "-e",
            "GIT_OPTIONAL_LOCKS=0",
            "-v",
            f"{workspace}:/workspace",
            # Keep the private baseline readable for project tools but prevent
            # package scripts from changing Git config, refs, or object data.
            "-v",
            f"{workspace / '.git'}:/workspace/.git:ro",
            "-w",
            f"/workspace/{relative_working}" if relative_working != "." else "/workspace",
        ]
        if include_verifier:
            # Hidden verifier code is mounted read-only only for the verifier
            # command; policy-visible checks never receive this mount.
            bundle = self._resolve_verifier_bundle()
            report_path = self._safe_path(manifest.verifier_report_path or "")
            report_relative = report_path.relative_to(workspace).as_posix()
            arguments.extend(
                [
                    "-v",
                    f"{bundle}:/autotrainer-verifier:ro",
                    "-e",
                    "AUTOTRAINER_WORKSPACE=/workspace",
                    "-e",
                    f"AUTOTRAINER_REPORT_PATH=/workspace/{report_relative}",
                ]
            )
        arguments.extend([self._image, "sh", "-lc", command])
        return subprocess.run(
            arguments,
            capture_output=True,
            text=True,
            timeout=self._timeout_for(manifest.command_timeout_seconds),
            check=False,
        )

    def _resolve_verifier_bundle(self) -> Path:
        manifest = self._require_manifest()
        root = self._task_root or Path.cwd()
        candidate = Path(manifest.verifier_bundle or "")
        candidate = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
        if not candidate.exists():
            raise RuntimeError(f"verifier bundle does not exist: {candidate}")
        expected = self._verifier_identity
        if expected is not None:
            try:
                actual = tree_identity(candidate)
            except IntegrityError as error:
                raise RuntimeError(f"verifier identity cannot be checked: {error}") from error
            # Compare both the aggregate hash and entry ledger. The latter
            # identifies renames and size changes instead of accepting an
            # underspecified digest-only task row.
            if actual.get("sha256") != expected.get("sha256") or actual.get(
                "files"
            ) != expected.get("files"):
                raise RuntimeError(
                    "hidden verifier files changed after the evaluation plan was frozen"
                )
        workspace = self._require_workspace()
        try:
            candidate.relative_to(workspace)
        except ValueError:
            return candidate
        raise RuntimeError("verifier bundle must be outside the editable workspace")

    def _safe_path(self, value: str) -> Path:
        workspace = self._require_workspace()
        relative = Path(value)
        if relative.is_absolute() or ".." in relative.parts or ".git" in relative.parts:
            raise RuntimeError("path must stay inside the editable workspace and outside .git")
        candidate = (workspace / relative).resolve()
        try:
            candidate.relative_to(workspace.resolve())
        except ValueError as error:
            raise RuntimeError("path escapes the editable workspace") from error
        return candidate

    def _validate_patch_paths(self, patch: str) -> None:
        headers = [line[4:].strip() for line in patch.splitlines() if line.startswith(("+++ ", "--- "))]
        if not headers:
            raise ValueError("unified diff headers were not found")
        for header in headers:
            path = header.split("\t", 1)[0]
            if path == "/dev/null":
                continue
            if path.startswith(("a/", "b/")):
                path = path[2:]
            self._safe_path(path)

    def _use_tool(self, name: str) -> None:
        self._check_deadline()
        manifest = self._require_manifest()
        if name not in manifest.tools:
            raise RuntimeError(f"tool is disabled for this task: {name}")
        self._tool_calls += 1
        self._tool_calls_by_name[name] = self._tool_calls_by_name.get(name, 0) + 1
        if self._tool_calls > manifest.tool_call_limit:
            raise RuntimeError("task tool-call limit exceeded")
        # Per-tool events would evict more useful scored episodes from the
        # bounded event store. The score event publishes aggregate names and
        # counts, never arguments, output, source, or model reasoning.

    def _bounded(self, value: str) -> str:
        self._check_deadline()
        return self._truncate(value)

    def _truncate(self, value: str) -> str:
        if len(value) <= self._max_output_chars:
            return value
        return value[: self._max_output_chars] + "\n... output truncated"

    def _require_manifest(self) -> TaskManifest:
        if self._manifest is None:
            raise RuntimeError("environment reset must run before tools")
        return self._manifest

    def _require_workspace(self) -> Path:
        if self._workspace is None:
            raise RuntimeError("environment reset must run before tools")
        return self._workspace

    @staticmethod
    def _remove_tree(path: Path) -> None:
        if not path.exists():
            return

        def remove_readonly(function: Any, value: str, _error: Any) -> None:
            os.chmod(value, stat.S_IRWXU)
            function(value)

        shutil.rmtree(path, onerror=remove_readonly)

    def _cleanup(self) -> None:
        if self._workspace is not None:
            workspace = self._workspace.resolve()
            root = self._temporary_root.resolve() if self._temporary_root else workspace.parent
            # Recursive deletion is permitted only for the exact temp root this
            # instance created, never for a path supplied by a task manifest.
            if workspace.name != "workspace" or workspace.parent != root:
                raise RuntimeError(
                    "refusing to clean an environment outside its verified temporary root"
                )

            self._workspace = None
            self._temporary_root = None
            self._remove_tree(root)
        self._manifest = None
        self._task_root = None
        self._verifier_identity = None
        self._episode_id = None
        self._tool_calls_by_name = {}
        self._install_result = None
        self._started_at = None
        self._deadline = None


def evaluate_patch(task_row: Mapping[str, Any], patch: str) -> EpisodeResult:
    """Evaluate a supplied patch in a fresh isolated environment instance."""

    return FrontendEnvironment()._evaluate_patch(task_row, patch)


__all__ = [
    "EpisodeResult",
    "EpisodeTimeoutError",
    "FrontendEnvironment",
    "evaluate_patch",
]
