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
from pathlib import Path
from typing import Any, Mapping

from ..environment import CheckResult, EpisodeResult, RolloutVerifierReport, score_rollout
from ..manifest import TaskManifest


class EpisodeTimeoutError(TimeoutError):
    """Raised when the trusted episode-wide deadline is exhausted."""


class FrontendEnvironment:
    """One disposable, network-disabled frontend coding episode."""

    def __init__(self) -> None:
        self._manifest: TaskManifest | None = None
        self._workspace: Path | None = None
        self._temporary_root: Path | None = None
        self._task_root: Path | None = None
        self._backend = "docker"
        self._image = ""
        self._tool_calls = 0
        self._max_output_chars = 12_000
        self._install_result: CheckResult | None = None
        self._check_results: dict[str, CheckResult] = {}
        self._started_at: float | None = None
        self._deadline: float | None = None
        self._latest_diff = ""
        self._last_result: EpisodeResult | None = None

    def reset(self, **task_row: Any) -> str:
        """Materialize the task row and return the policy's first observation."""

        try:
            manifest, revision = self._initialize(task_row)
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
        """List editable repository files below a relative directory."""

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
        """Read an inclusive, one-indexed line range from a UTF-8 text file."""

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
        """Search text files for a literal query and return bounded line matches."""

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
        """Apply a unified diff after path and Git safety checks."""

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
        """Run one trusted named check: build, tests, or browserTests."""

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

        return self.finalize().reward

    def finalize(self) -> EpisodeResult:
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

    def evaluate_patch(self, task_row: Mapping[str, Any], patch: str) -> EpisodeResult:
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
            return self.finalize()
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

        revision = manifest.starting_revision
        if revision == "locked":
            revision = str(task_row.get("source_revision", ""))
        if not revision:
            raise RuntimeError("starting revision was not resolved by compile")
        if (source_path / ".git").exists():
            if re.fullmatch(r"(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})", revision) is None:
                raise RuntimeError(
                    "evaluation requires source_revision to be a full immutable Git commit"
                )
        elif re.fullmatch(r"tree:[0-9a-fA-F]{64}", revision) is None:
            raise RuntimeError(
                "non-Git evaluation requires an immutable tree:<sha256> source_revision"
            )

        self._backend = str(task_row.get("environment_backend", "docker"))
        if self._backend not in {"docker", "podman"}:
            raise RuntimeError("environment_backend must be docker or podman")
        if shutil.which(self._backend) is None:
            raise RuntimeError(f"{self._backend} is required for frontend RL episodes")
        self._image = str(task_row.get("environment_image", ""))
        if not self._image:
            raise RuntimeError("compiled task row is missing environment_image")
        self._max_output_chars = int(task_row.get("max_tool_output_chars", 12_000))

        self._manifest = manifest
        self._task_root = task_root
        self._tool_calls = 0
        self._install_result = None
        self._check_results = {}
        self._latest_diff = ""
        self._last_result = None
        self._started_at = time.monotonic()
        self._deadline = self._started_at + manifest.episode_timeout_seconds

        temporary_root = Path(tempfile.mkdtemp(prefix="autotrainer-episode-"))
        workspace = temporary_root / "workspace"
        self._temporary_root = temporary_root.resolve()
        self._workspace = workspace
        try:
            self._materialize(source_path, revision, workspace)
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
        check = subprocess.run(
            ["git", "-C", str(workspace), "apply", "--check", "--whitespace=error-all", "-"],
            input=patch,
            capture_output=True,
            text=True,
            timeout=self._timeout_for(30),
            check=False,
        )
        if check.returncode:
            return False, self._truncate(f"patch check failed:\n{check.stderr or check.stdout}")
        applied = subprocess.run(
            ["git", "-C", str(workspace), "apply", "--whitespace=nowarn", "-"],
            input=patch,
            capture_output=True,
            text=True,
            timeout=self._timeout_for(30),
            check=False,
        )
        if applied.returncode:
            return False, self._truncate(
                f"patch apply failed:\n{applied.stderr or applied.stdout}"
            )
        self._check_deadline()
        return True, "patch applied"

    def _capture_unified_diff(self) -> str:
        workspace = self._require_workspace()
        intent = subprocess.run(
            ["git", "-C", str(workspace), "add", "--intent-to-add", "--all"],
            capture_output=True,
            text=True,
            timeout=self._timeout_for(30),
            check=False,
        )
        if intent.returncode:
            raise RuntimeError(f"could not prepare unified diff: {intent.stderr.strip()}")
        diff = subprocess.run(
            ["git", "-C", str(workspace), "diff", "--binary", "--no-ext-diff", "HEAD", "--"],
            capture_output=True,
            text=True,
            timeout=self._timeout_for(30),
            check=False,
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

    def _materialize(self, source: Path, revision: str, destination: Path) -> None:
        if (source / ".git").exists():
            clone = subprocess.run(
                ["git", "clone", "--quiet", "--no-hardlinks", str(source), str(destination)],
                capture_output=True,
                text=True,
                timeout=self._timeout_for(120),
                check=False,
            )
            if clone.returncode:
                raise RuntimeError(f"failed to clone task source: {clone.stderr.strip()}")
            checkout = subprocess.run(
                ["git", "-C", str(destination), "checkout", "--quiet", "--detach", revision],
                capture_output=True,
                text=True,
                timeout=self._timeout_for(60),
                check=False,
            )
            if checkout.returncode:
                raise RuntimeError(f"cannot resolve starting revision {revision}: {checkout.stderr.strip()}")
            return
        if not revision.startswith("tree:"):
            raise RuntimeError("non-Git task sources require a locked tree:<sha256> revision")
        shutil.copytree(
            source,
            destination,
            symlinks=False,
            ignore=shutil.ignore_patterns("node_modules", "dist", "build", ".autotrainer", ".git"),
        )
        initialized = subprocess.run(
            ["git", "init", "--quiet", str(destination)],
            capture_output=True,
            text=True,
            timeout=self._timeout_for(30),
            check=False,
        )
        if initialized.returncode:
            raise RuntimeError(f"failed to initialize task snapshot: {initialized.stderr.strip()}")
        staged = subprocess.run(
            ["git", "-C", str(destination), "add", "--all"],
            capture_output=True,
            text=True,
            timeout=self._timeout_for(30),
            check=False,
        )
        if staged.returncode:
            raise RuntimeError(f"failed to index task snapshot: {staged.stderr.strip()}")
        committed = subprocess.run(
            [
                "git",
                "-C",
                str(destination),
                "-c",
                "user.name=AutoTrainer",
                "-c",
                "user.email=autotrainer@localhost",
                "commit",
                "--quiet",
                "-m",
                "Locked evaluation snapshot",
            ],
            capture_output=True,
            text=True,
            timeout=self._timeout_for(30),
            check=False,
        )
        if committed.returncode:
            raise RuntimeError(f"failed to lock task snapshot: {committed.stderr.strip()}")

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
            "-v",
            f"{workspace}:/workspace",
            "-w",
            f"/workspace/{relative_working}" if relative_working != "." else "/workspace",
        ]
        if include_verifier:
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
        if self._tool_calls > manifest.tool_call_limit:
            raise RuntimeError("task tool-call limit exceeded")

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

    def _cleanup(self) -> None:
        if self._workspace is not None:
            workspace = self._workspace.resolve()
            root = self._temporary_root.resolve() if self._temporary_root else workspace.parent
            if workspace.name != "workspace" or workspace.parent != root:
                raise RuntimeError(
                    "refusing to clean an environment outside its verified temporary root"
                )

            def remove_readonly(function: Any, path: str, _error: Any) -> None:
                os.chmod(path, stat.S_IRWXU)
                function(path)

            self._workspace = None
            self._temporary_root = None
            shutil.rmtree(root, onerror=remove_readonly)
        self._manifest = None
        self._task_root = None
        self._install_result = None
        self._started_at = None
        self._deadline = None


def evaluate_patch(task_row: Mapping[str, Any], patch: str) -> EpisodeResult:
    """Evaluate a supplied patch in a fresh isolated environment instance."""

    return FrontendEnvironment().evaluate_patch(task_row, patch)


__all__ = [
    "EpisodeResult",
    "EpisodeTimeoutError",
    "FrontendEnvironment",
    "evaluate_patch",
]
