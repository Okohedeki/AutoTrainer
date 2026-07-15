"""Container-isolated frontend environment for TRL's environment factory API.

The policy never receives a general shell tool.  It can inspect and patch a
disposable checkout and invoke only commands named by the trusted task manifest.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping

from ..environment import RolloutVerifierReport, score_rollout
from ..manifest import TaskManifest


class FrontendEnvironment:
    """One disposable, network-disabled frontend coding episode."""

    def __init__(self) -> None:
        self._manifest: TaskManifest | None = None
        self._workspace: Path | None = None
        self._task_root: Path | None = None
        self._backend = "docker"
        self._image = ""
        self._tool_calls = 0
        self._max_output_chars = 12_000
        self._install_result: subprocess.CompletedProcess[str] | None = None
        self._last_reward: dict[str, Any] | None = None

    def reset(self, **task_row: Any) -> str:
        """Materialize the task row and return the policy's first observation."""

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

        self._backend = str(task_row.get("environment_backend", "docker"))
        if self._backend not in {"docker", "podman"}:
            raise RuntimeError("environment_backend must be docker or podman")
        if shutil.which(self._backend) is None:
            raise RuntimeError(f"{self._backend} is required for frontend RL episodes")
        self._image = str(task_row.get("environment_image", ""))
        if not self._image:
            raise RuntimeError("compiled task row is missing environment_image")
        self._max_output_chars = int(task_row.get("max_tool_output_chars", 12_000))

        temporary_root = Path(tempfile.mkdtemp(prefix="autotrainer-episode-"))
        workspace = temporary_root / "workspace"
        self._materialize(source_path, revision, workspace)
        self._manifest = manifest
        self._workspace = workspace
        self._task_root = task_root
        self._tool_calls = 0
        self._last_reward = None

        install = manifest.runtime_commands.get("install", "").strip()
        self._install_result = self._run_container_command(install) if install else None
        install_status = "not requested"
        if self._install_result is not None:
            install_status = "passed" if self._install_result.returncode == 0 else "failed"
        return (
            f"Task: {manifest.instruction}\n"
            f"Source: {manifest.source_id}@{revision}\n"
            f"Install: {install_status}. Network access is disabled. "
            "Inspect the repository, apply a focused patch, and run named checks."
        )

    def list_files(self, path: str = ".") -> str:
        """List editable repository files below a relative directory."""

        self._use_tool("list_files")
        root = self._safe_path(path)
        if not root.is_dir():
            return f"not a directory: {path}"
        files: list[str] = []
        for candidate in sorted(root.rglob("*")):
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
        if not patch.strip() or len(patch) > 200_000:
            return "patch must contain between 1 and 200000 characters"
        try:
            self._validate_patch_paths(patch)
        except ValueError as error:
            return f"rejected patch: {error}"
        workspace = self._require_workspace()
        check = subprocess.run(
            ["git", "-C", str(workspace), "apply", "--check", "--whitespace=error-all", "-"],
            input=patch,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if check.returncode:
            return self._bounded(f"patch check failed:\n{check.stderr or check.stdout}")
        applied = subprocess.run(
            ["git", "-C", str(workspace), "apply", "--whitespace=nowarn", "-"],
            input=patch,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if applied.returncode:
            return self._bounded(f"patch apply failed:\n{applied.stderr or applied.stdout}")
        return "patch applied"

    def run_check(self, check: str) -> str:
        """Run one trusted named check: build, tests, or browserTests."""

        self._use_tool("run_check")
        if check not in {"build", "tests", "browserTests"}:
            return "unknown check; choose build, tests, or browserTests"
        command = self._require_manifest().runtime_commands.get(check, "").strip()
        if not command:
            return f"check is not configured: {check}"
        completed = self._run_container_command(command)
        label = "passed" if completed.returncode == 0 else f"failed (exit {completed.returncode})"
        return self._bounded(f"{check} {label}\n{completed.stdout}\n{completed.stderr}")

    def get_reward(self) -> float:
        """Run trusted gates and hidden verifier, then return the scalar reward."""

        manifest = self._require_manifest()
        if self._install_result is not None and self._install_result.returncode:
            self._last_reward = {"gated": True, "gate_reason": "install_failed", "total": 0.0}
            self._cleanup()
            return 0.0
        build = self._run_container_command(manifest.runtime_commands["build"])
        if build.returncode:
            self._last_reward = {"gated": True, "gate_reason": "build_failed", "total": 0.0}
            self._cleanup()
            return 0.0
        tests = self._run_container_command(manifest.runtime_commands["tests"])
        if tests.returncode:
            self._last_reward = {"gated": True, "gate_reason": "regression_failed", "total": 0.0}
            self._cleanup()
            return 0.0
        verifier = self._run_container_command(manifest.verifier_command or "", include_verifier=True)
        if verifier.returncode:
            self._last_reward = {"gated": True, "gate_reason": "verifier_failed", "total": 0.0}
            self._cleanup()
            return 0.0
        report_path = self._safe_path(manifest.verifier_report_path or "")
        try:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            report = RolloutVerifierReport(
                build_passed=True,
                regression_pass_rate=float(payload["regression_pass_rate"]),
                task_pass_rate=float(payload["task_pass_rate"]),
                responsive_pass_rate=float(payload["responsive_pass_rate"]),
                design_rule_pass_rate=float(payload["design_rule_pass_rate"]),
                code_quality_pass_rate=float(payload["code_quality_pass_rate"]),
            )
            weights = {
                "regression_safety": manifest.reward_weights["regressionSafety"],
                "task_tests": manifest.reward_weights["taskTests"],
                "responsive_rules": manifest.reward_weights["responsiveRules"],
                "design_rules": manifest.reward_weights["designRules"],
                "patch_quality": manifest.reward_weights["patchQuality"],
            }
            reward = score_rollout(report, weights)
            self._last_reward = {
                "gated": reward.gated,
                "gate_reason": reward.gate_reason,
                "total": reward.total,
                "signals": reward.signals,
            }
            return reward.total
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            self._last_reward = {"gated": True, "gate_reason": "invalid_verifier_report", "total": 0.0}
            return 0.0
        finally:
            self._cleanup()

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
                timeout=120,
                check=False,
            )
            if clone.returncode:
                raise RuntimeError(f"failed to clone task source: {clone.stderr.strip()}")
            checkout = subprocess.run(
                ["git", "-C", str(destination), "checkout", "--quiet", "--detach", revision],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            if checkout.returncode:
                raise RuntimeError(f"cannot resolve starting revision {revision}: {checkout.stderr.strip()}")
            return
        if not (revision.startswith("tree:") or revision == "WORKTREE"):
            raise RuntimeError("non-Git task sources require a locked tree:<sha256> revision")
        shutil.copytree(
            source,
            destination,
            symlinks=False,
            ignore=shutil.ignore_patterns("node_modules", "dist", "build", ".autotrainer", ".git"),
        )
        subprocess.run(["git", "init", "--quiet", str(destination)], capture_output=True, check=False)

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
            timeout=manifest.command_timeout_seconds,
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
        manifest = self._require_manifest()
        if name not in manifest.tools:
            raise RuntimeError(f"tool is disabled for this task: {name}")
        self._tool_calls += 1
        if self._tool_calls > manifest.tool_call_limit:
            raise RuntimeError("task tool-call limit exceeded")

    def _bounded(self, value: str) -> str:
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
            root = self._workspace.parent
            self._workspace = None
            shutil.rmtree(root, ignore_errors=True)
        self._manifest = None
        self._task_root = None


__all__ = ["FrontendEnvironment"]
