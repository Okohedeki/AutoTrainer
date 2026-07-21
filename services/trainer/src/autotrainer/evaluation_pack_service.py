"""Install first-party, language-matched held-out evaluation packs.

The shipped packs are generated from constants in this module so installation
does not need a package manager or network access.  Each task gets its own Git
repository and a verifier in a sibling directory that is never mounted into the
editable policy workspace.  The installer swaps one complete staging tree into
place before updating YAML, and rolls both changes back if any later step fails.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
from typing import Any
from uuid import uuid4

from .config import (
    ConfigError,
    load_config,
    project_config_mutation,
    validate_mapping,
    write_config,
)
from .manifest import TaskManifest
from .project_gate import project_mutation_gate


PACK_ID = "python-core-v1"
PACK_LABEL = "Python workflow repairs"
PACK_LANGUAGE = "python"
PACK_LICENSE = "Apache-2.0"
RUNTIME_IMAGE = "autotrainer/frontend-runtime:0.1"
PACK_DESCRIPTION = (
    "Five independent, offline Python workflow-repair tasks with hidden verifiers. "
    "The tasks are fully original and draw only on benchmark design patterns from "
    "HumanEval, MBPP, and SWE-bench."
)
PACK_CHECKS = (
    "python3 -m compileall -q .",
    "python3 -m unittest discover -s tests -v",
    "python3 /autotrainer-verifier/verify.py",
)

_GENERATOR = "autotrainer.evaluation_pack_service"
_PACK_FORMAT_VERSION = 1
_FIXED_GIT_DATE = "2024-01-01T00:00:00+00:00"
_REPORT_PATH = ".autotrainer-verifier-report.json"
_TOOLS = (
    "list_files",
    "read_file",
    "search_code",
    "apply_patch",
    "replace_text",
    "run_check",
)
_LICENSE_TEXT = """Apache License
Version 2.0, January 2004
https://www.apache.org/licenses/LICENSE-2.0

SPDX-License-Identifier: Apache-2.0

Copyright 2026 AutoTrainer contributors

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License. You may obtain a copy of the
License at the URL above. Unless required by applicable law or agreed to in
writing, software distributed under the License is distributed on an "AS IS"
BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations.
"""
_NOTICE_TEXT = """AutoTrainer Python Core Evaluation Pack

This product contains five original workflow-domain Python repair tasks shipped
by AutoTrainer. Their benchmark structure is inspired by HumanEval, MBPP, and
SWE-bench; no task text, tests, or repository code is copied from those works.
The task repositories and hidden verifier bundles are licensed under Apache-2.0.
"""


@dataclass(frozen=True, slots=True)
class _TaskSpec:
    slug: str
    task_id: str
    module_name: str
    instruction: str
    module_source: str
    public_test_source: str
    hidden_test_source: str

    @property
    def source_id(self) -> str:
        return f"{PACK_ID}-{self.slug}"


_TASKS = (
    _TaskSpec(
        slug="ready-order",
        task_id="python-core-stable-ready-order",
        module_name="workflow_order.py",
        instruction=(
            "Repair downstream_order so tasks that become runnable at the same time "
            "retain the caller's declared order. Keep dependency validation and cycle "
            "detection working, and do not add third-party dependencies."
        ),
        module_source='''"""Small dependency-ordering helper used by a local DAG runner."""\n\n\ndef downstream_order(tasks, dependencies):\n    """Return a valid downstream order or raise ValueError for an invalid graph."""\n    declared = list(tasks)\n    if len(set(declared)) != len(declared):\n        raise ValueError("task names must be unique")\n    known = set(declared)\n    graph = {name: set(dependencies.get(name, ())) for name in declared}\n    if any(parent not in known for parents in graph.values() for parent in parents):\n        raise ValueError("dependency names must identify declared tasks")\n\n    completed = set()\n    ordered = []\n    while len(ordered) < len(declared):\n        ready = sorted(\n            name for name in declared if name not in completed and graph[name] <= completed\n        )\n        if not ready:\n            raise ValueError("workflow contains a cycle")\n        ordered.extend(ready)\n        completed.update(ready)\n    return ordered\n''',
        public_test_source='''import unittest\n\nfrom workflow_order import downstream_order\n\n\nclass DownstreamOrderTests(unittest.TestCase):\n    def test_orders_a_simple_pipeline(self):\n        tasks = ["extract", "transform", "load"]\n        graph = {"transform": {"extract"}, "load": {"transform"}}\n        self.assertEqual(downstream_order(tasks, graph), tasks)\n\n    def test_rejects_an_unknown_dependency(self):\n        with self.assertRaises(ValueError):\n            downstream_order(["extract"], {"extract": {"missing"}})\n\n\nif __name__ == "__main__":\n    unittest.main()\n''',
        hidden_test_source='''from workflow_order import downstream_order\n\ntasks = ["validate", "extract", "publish"]\ngraph = {"publish": {"validate", "extract"}}\nassert downstream_order(tasks, graph) == tasks\n''',
    ),
    _TaskSpec(
        slug="trigger-rule",
        task_id="python-core-none-failed-trigger",
        module_name="trigger_rules.py",
        instruction=(
            "Add correct support for the none_failed_min_one_success trigger rule. It "
            "must wait while an upstream task is unfinished, reject failed and "
            "upstream_failed states, and require at least one success even when other "
            "upstreams were skipped. Preserve the existing all_success behavior."
        ),
        module_source='''"""Trigger-rule predicates for an offline workflow scheduler."""\n\n_TERMINAL = {"success", "failed", "upstream_failed", "skipped"}\n\n\ndef should_run(trigger_rule, upstream_states):\n    states = list(upstream_states)\n    if not states or any(state not in _TERMINAL for state in states):\n        return False\n    if trigger_rule == "all_success":\n        return all(state == "success" for state in states)\n    if trigger_rule == "none_failed_min_one_success":\n        return all(state != "failed" for state in states)\n    raise ValueError(f"unknown trigger rule: {trigger_rule}")\n''',
        public_test_source='''import unittest\n\nfrom trigger_rules import should_run\n\n\nclass TriggerRuleTests(unittest.TestCase):\n    def test_all_success(self):\n        self.assertTrue(should_run("all_success", ["success", "success"]))\n        self.assertFalse(should_run("all_success", ["success", "skipped"]))\n\n    def test_unknown_rule(self):\n        with self.assertRaises(ValueError):\n            should_run("sometimes", ["success"])\n\n\nif __name__ == "__main__":\n    unittest.main()\n''',
        hidden_test_source='''from trigger_rules import should_run\n\nassert should_run("none_failed_min_one_success", ["skipped", "success"]) is True\nassert should_run("none_failed_min_one_success", ["skipped", "skipped"]) is False\nassert should_run("none_failed_min_one_success", ["success", "upstream_failed"]) is False\nassert should_run("none_failed_min_one_success", ["success", "running"]) is False\n''',
    ),
    _TaskSpec(
        slug="retry-backoff",
        task_id="python-core-exponential-retry-backoff",
        module_name="retry_policy.py",
        instruction=(
            "Fix retry_delay_seconds to use capped exponential backoff: attempt 1 uses "
            "base_seconds, each later attempt doubles the prior delay, and max_seconds "
            "is a hard ceiling. Continue rejecting booleans, non-positive attempts, and "
            "invalid delay bounds."
        ),
        module_source='''"""Retry timing policy for task attempts."""\n\n\ndef retry_delay_seconds(base_seconds, attempt, max_seconds):\n    values = (base_seconds, attempt, max_seconds)\n    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):\n        raise TypeError("retry values must be integers")\n    if base_seconds < 1 or attempt < 1 or max_seconds < base_seconds:\n        raise ValueError("invalid retry bounds")\n    return min(base_seconds * attempt, max_seconds)\n''',
        public_test_source='''import unittest\n\nfrom retry_policy import retry_delay_seconds\n\n\nclass RetryPolicyTests(unittest.TestCase):\n    def test_first_two_attempts(self):\n        self.assertEqual(retry_delay_seconds(10, 1, 100), 10)\n        self.assertEqual(retry_delay_seconds(10, 2, 100), 20)\n\n    def test_rejects_boolean_attempt(self):\n        with self.assertRaises(TypeError):\n            retry_delay_seconds(10, True, 100)\n\n\nif __name__ == "__main__":\n    unittest.main()\n''',
        hidden_test_source='''from retry_policy import retry_delay_seconds\n\nassert retry_delay_seconds(10, 3, 100) == 40\nassert retry_delay_seconds(10, 8, 100) == 100\nassert retry_delay_seconds(3, 4, 50) == 24\n''',
    ),
    _TaskSpec(
        slug="context-values",
        task_id="python-core-preserve-context-values",
        module_name="task_context.py",
        instruction=(
            "Repair serializable_context so it omits only values that are None. Preserve "
            "valid false, zero, and empty-string values because templates distinguish "
            "them from missing values. Keep deterministic key ordering and reject "
            "non-string keys."
        ),
        module_source='''"""Build deterministic task-template context dictionaries."""\n\n\ndef serializable_context(values):\n    if any(not isinstance(key, str) for key in values):\n        raise TypeError("context keys must be strings")\n    return {key: values[key] for key in sorted(values) if values[key]}\n''',
        public_test_source='''import unittest\n\nfrom task_context import serializable_context\n\n\nclass TaskContextTests(unittest.TestCase):\n    def test_sorts_populated_values(self):\n        self.assertEqual(\n            list(serializable_context({"run_id": "42", "dag_id": "etl"})),\n            ["dag_id", "run_id"],\n        )\n\n    def test_rejects_non_string_keys(self):\n        with self.assertRaises(TypeError):\n            serializable_context({1: "invalid"})\n\n\nif __name__ == "__main__":\n    unittest.main()\n''',
        hidden_test_source='''from task_context import serializable_context\n\nvalues = {"enabled": False, "retries": 0, "note": "", "owner": None}\nassert serializable_context(values) == {"enabled": False, "note": "", "retries": 0}\n''',
    ),
    _TaskSpec(
        slug="pool-slots",
        task_id="python-core-pool-slot-allocation",
        module_name="pool_scheduler.py",
        instruction=(
            "Fix allocate_pool_slots so a queued task requesting more slots than the "
            "pool's total capacity is ignored instead of blocking later runnable work. "
            "Schedule by descending priority, retain FIFO order for ties, never exceed "
            "capacity, and reject malformed task records."
        ),
        module_source='''"""Select queued workflow tasks for a fixed-capacity worker pool."""\n\n\ndef allocate_pool_slots(tasks, capacity):\n    if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:\n        raise ValueError("capacity must be a positive integer")\n    queued = []\n    for position, task in enumerate(tasks):\n        if set(task) != {"id", "priority", "slots"}:\n            raise ValueError("tasks require id, priority, and slots")\n        if not isinstance(task["id"], str) or not task["id"]:\n            raise ValueError("task id must be non-empty text")\n        if any(isinstance(task[key], bool) or not isinstance(task[key], int) for key in ("priority", "slots")):\n            raise ValueError("priority and slots must be integers")\n        if task["slots"] < 1:\n            raise ValueError("slots must be positive")\n        queued.append((position, task))\n\n    remaining = capacity\n    selected = []\n    for _, task in sorted(queued, key=lambda item: (-item[1]["priority"], item[0])):\n        if task["slots"] > remaining:\n            break\n        selected.append(task["id"])
        remaining -= task["slots"]\n    return selected\n''',
        public_test_source='''import unittest\n\nfrom pool_scheduler import allocate_pool_slots\n\n\nclass PoolSchedulerTests(unittest.TestCase):\n    def test_priority_and_capacity(self):\n        tasks = [\n            {"id": "low", "priority": 1, "slots": 1},\n            {"id": "high", "priority": 5, "slots": 2},\n        ]\n        self.assertEqual(allocate_pool_slots(tasks, 3), ["high", "low"])\n\n    def test_rejects_zero_capacity(self):\n        with self.assertRaises(ValueError):\n            allocate_pool_slots([], 0)\n\n\nif __name__ == "__main__":\n    unittest.main()\n''',
        hidden_test_source='''from pool_scheduler import allocate_pool_slots\n\ntasks = [\n    {"id": "oversized", "priority": 100, "slots": 9},\n    {"id": "extract", "priority": 10, "slots": 1},\n    {"id": "validate", "priority": 10, "slots": 1},\n]\nassert allocate_pool_slots(tasks, 2) == ["extract", "validate"]\n''',
    ),
)


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def _write_file(path: Path, value: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = value if isinstance(value, bytes) else value.encode("utf-8")
    path.write_bytes(payload)


def _display_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _run_git(repository: Path, *arguments: str) -> str:
    environment = {
        **os.environ,
        "GIT_AUTHOR_DATE": _FIXED_GIT_DATE,
        "GIT_COMMITTER_DATE": _FIXED_GIT_DATE,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_TERMINAL_PROMPT": "0",
    }
    # Ignore the operator's Git identity and signing settings. The fixed tree,
    # identity, dates, and message make each repository commit reproducible.
    environment["GIT_CONFIG_GLOBAL"] = os.devnull
    try:
        completed = subprocess.run(
            [
                "git",
                "-c",
                "core.autocrlf=false",
                "-c",
                "commit.gpgsign=false",
                "-c",
                "user.name=AutoTrainer Evaluation Packs",
                "-c",
                "user.email=evaluation-packs@autotrainer.invalid",
                "-C",
                str(repository),
                *arguments,
            ],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            env=environment,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired) as error:
        raise ConfigError(f"could not create the shipped evaluation repository: {error}") from error
    if completed.returncode:
        detail = (completed.stderr or completed.stdout).strip().replace("\n", " ")
        raise ConfigError(
            "could not create the shipped evaluation repository: "
            + (detail[-1_000:] or f"git exited {completed.returncode}")
        )
    return completed.stdout.strip()


def _create_repository(path: Path, spec: _TaskSpec) -> str:
    path.mkdir(parents=True)
    _write_file(path / spec.module_name, spec.module_source)
    _write_file(path / "tests" / "test_public.py", spec.public_test_source)
    _write_file(path / "README.md", f"# {spec.task_id}\n\n{spec.instruction}\n")
    _write_file(path / ".gitignore", "__pycache__/\n*.py[cod]\n.autotrainer-verifier-report.json\n")
    _write_file(path / "LICENSE", _LICENSE_TEXT)
    _write_file(path / "NOTICE", _NOTICE_TEXT)
    _run_git(path, "init", "--quiet")
    _run_git(path, "symbolic-ref", "HEAD", "refs/heads/main")
    _run_git(path, "add", "--all")
    _run_git(path, "commit", "--quiet", "-m", f"Ship {spec.task_id} baseline")
    revision = _run_git(path, "rev-parse", "HEAD").casefold()
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise ConfigError("git returned an invalid immutable revision for a shipped task")
    if _run_git(path, "status", "--porcelain"):
        raise ConfigError("the generated evaluation repository is not clean")
    return revision


def _verifier_source(spec: _TaskSpec) -> str:
    hidden = repr(spec.hidden_test_source)
    module = repr(spec.module_name)
    return f'''"""Hidden verifier for {spec.task_id}."""\n\nimport ast\nimport json\nimport os\nfrom pathlib import Path\nimport subprocess\nimport sys\n\nworkspace = Path(os.environ["AUTOTRAINER_WORKSPACE"]).resolve()\nreport_path = Path(os.environ["AUTOTRAINER_REPORT_PATH"]).resolve()\n\ndef run(*arguments):\n    return subprocess.run(\n        [sys.executable, *arguments], cwd=workspace, capture_output=True, text=True, timeout=60\n    )\n\nbuild = run("-m", "compileall", "-q", ".")\nregression = run("-m", "unittest", "discover", "-s", "tests", "-v")\nhidden = run("-c", {hidden})\ntry:\n    source = (workspace / {module}).read_text(encoding="utf-8")\n    ast.parse(source)\n    quality = float("TODO" not in source and "pass  #" not in source)\nexcept (OSError, SyntaxError, UnicodeError):\n    quality = 0.0\n\nreport = {{\n    "build_passed": build.returncode == 0,\n    "regression_pass_rate": float(regression.returncode == 0),\n    "task_pass_rate": float(hidden.returncode == 0),\n    "responsive_pass_rate": 1.0,\n    "design_rule_pass_rate": 1.0,\n    "code_quality_pass_rate": quality,\n}}\nreport_path.write_text(json.dumps(report, sort_keys=True) + "\\n", encoding="utf-8")\nprint(json.dumps({{"task": {spec.task_id!r}, "hidden_passed": hidden.returncode == 0}}))\nsys.exit(0)\n'''


def _manifest(
    spec: _TaskSpec,
    revision: str,
    verifier_bundle: Path,
) -> dict[str, Any]:
    return {
        "version": "1.0",
        "task": {
            "id": spec.task_id,
            "instruction": spec.instruction,
            "sourceId": spec.source_id,
            "startingRevision": revision,
            "split": "evaluation",
            "groupId": spec.source_id,
        },
        "runtime": {
            "stack": "python-stdlib",
            "workingDirectory": ".",
            "install": "",
            "build": PACK_CHECKS[0],
            "tests": PACK_CHECKS[1],
            "browserTests": "",
        },
        "tools": list(_TOOLS),
        "verifier": {
            "bundle": str(verifier_bundle.resolve()),
            "command": PACK_CHECKS[2],
            "reportPath": _REPORT_PATH,
        },
        "rewards": {
            "buildGate": True,
            "regressionGate": True,
            "regressionSafety": 0.25,
            "taskTests": 0.55,
            "responsiveRules": 0.0,
            "designRules": 0.0,
            "patchQuality": 0.20,
        },
        "limits": {
            "toolCalls": 32,
            "commandTimeoutSeconds": 120,
            "episodeTimeoutSeconds": 900,
            "networkAccess": False,
        },
        "metadata": {
            "language": PACK_LANGUAGE,
            "license": PACK_LICENSE,
            "runtimeImage": RUNTIME_IMAGE,
            "benchmarkInspirations": ["HumanEval", "MBPP", "SWE-bench"],
            "originalWork": True,
        },
    }


def _tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _materialize_pack(staging: Path, final: Path) -> dict[str, str]:
    revisions: dict[str, str] = {}
    verifier_hashes: dict[str, str] = {}
    for spec in _TASKS:
        repository = staging / "repositories" / spec.slug
        revision = _create_repository(repository, spec)
        revisions[spec.source_id] = revision

        verifier = staging / "verifiers" / spec.task_id
        _write_file(verifier / "verify.py", _verifier_source(spec))
        _write_file(verifier / "LICENSE", _LICENSE_TEXT)
        _write_file(verifier / "NOTICE", _NOTICE_TEXT)
        verifier_hashes[spec.task_id] = _tree_hash(verifier)

        final_verifier = final / "verifiers" / spec.task_id
        manifest = _manifest(spec, revision, final_verifier)
        # Parse before publication so malformed shipped constants never reach a
        # project, even if config validation would accept their source paths.
        TaskManifest.from_mapping(manifest)
        _write_file(staging / "tasks" / spec.task_id / "task.json", _json_bytes(manifest))

    _write_file(staging / "LICENSE", _LICENSE_TEXT)
    _write_file(staging / "NOTICE", _NOTICE_TEXT)
    metadata = {
        "format_version": _PACK_FORMAT_VERSION,
        "generator": _GENERATOR,
        "id": PACK_ID,
        "label": PACK_LABEL,
        "language": PACK_LANGUAGE,
        "license": PACK_LICENSE,
        "runtime_image": RUNTIME_IMAGE,
        "task_count": len(_TASKS),
        "independent_group_count": len(_TASKS),
        "repository_revisions": revisions,
        "verifier_sha256": verifier_hashes,
        "benchmark_inspirations": ["HumanEval", "MBPP", "SWE-bench"],
        "original_work": True,
    }
    _write_file(staging / "pack.json", _json_bytes(metadata))
    return revisions


def _remove_tree(path: Path) -> None:
    def make_writable(function: Any, name: str, _error: Any) -> None:
        os.chmod(name, stat.S_IRUSR | stat.S_IWUSR | (stat.S_IXUSR if os.path.isdir(name) else 0))
        function(name)

    if path.is_symlink():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path, onerror=make_writable)


def _read_metadata(pack_root: Path) -> Mapping[str, Any] | None:
    try:
        value = json.loads((pack_root / "pack.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(value, Mapping):
        return None
    if (
        value.get("id") != PACK_ID
        or value.get("generator") != _GENERATOR
        or value.get("format_version") != _PACK_FORMAT_VERSION
    ):
        return None
    return value


def _inspect_materialized_pack(pack_root: Path) -> dict[str, str] | None:
    metadata = _read_metadata(pack_root)
    if metadata is None:
        return None
    revisions_value = metadata.get("repository_revisions")
    verifier_hashes = metadata.get("verifier_sha256")
    if not isinstance(revisions_value, Mapping) or not isinstance(verifier_hashes, Mapping):
        return None
    revisions: dict[str, str] = {}
    try:
        for spec in _TASKS:
            repository = pack_root / "repositories" / spec.slug
            revision = str(revisions_value[spec.source_id]).casefold()
            if _run_git(repository, "rev-parse", "HEAD").casefold() != revision:
                return None
            if _run_git(repository, "status", "--porcelain"):
                return None
            manifest_path = pack_root / "tasks" / spec.task_id / "task.json"
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest = TaskManifest.from_mapping(payload)
            expected_bundle = (pack_root / "verifiers" / spec.task_id).resolve()
            if (
                manifest.task_id != spec.task_id
                or manifest.source_id != spec.source_id
                or manifest.starting_revision.casefold() != revision
                or Path(manifest.verifier_bundle or "").resolve() != expected_bundle
                or _tree_hash(expected_bundle) != str(verifier_hashes[spec.task_id])
            ):
                return None
            revisions[spec.source_id] = revision
    except (ConfigError, KeyError, OSError, UnicodeError, ValueError, json.JSONDecodeError):
        return None
    return revisions if len(revisions) == len(_TASKS) else None


def _repository_declaration(
    config: Any,
    pack_root: Path,
    spec: _TaskSpec,
    revision: str,
) -> dict[str, Any]:
    return {
        "id": spec.source_id,
        "kind": "repository",
        "uri": _display_path(pack_root / "repositories" / spec.slug, config.root),
        "revision": revision,
        "partition": "evaluation",
        "roles": ["evaluation"],
        "include": ["*.py", "tests/**/*.py", "README.md", "LICENSE", "NOTICE"],
        "license": PACK_LICENSE,
    }


def _pack_declaration(config: Any, pack_root: Path) -> dict[str, Any]:
    return {
        "id": PACK_ID,
        "kind": "task_pack",
        "uri": _display_path(pack_root / "tasks", config.root),
        "partition": "evaluation",
        "roles": ["evaluation"],
        "license": PACK_LICENSE,
    }


def _source_uri_matches(config: Any, source: Mapping[str, Any], path: Path) -> bool:
    try:
        return config.resolve_path(str(source.get("uri", ""))) == path.resolve()
    except (OSError, ValueError):
        return False


def _merge_sources(
    config: Any,
    pack_root: Path,
    revisions: Mapping[str, str],
) -> list[dict[str, Any]]:
    canonical = {
        spec.source_id: _repository_declaration(
            config, pack_root, spec, revisions[spec.source_id]
        )
        for spec in _TASKS
    }
    canonical[PACK_ID] = _pack_declaration(config, pack_root)
    expected_paths = {
        spec.source_id: pack_root / "repositories" / spec.slug for spec in _TASKS
    }
    expected_paths[PACK_ID] = pack_root / "tasks"

    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in config.sources:
        source_id = str(source.get("id", ""))
        if source_id not in canonical:
            merged.append(dict(source))
            continue
        expected_kind = canonical[source_id]["kind"]
        if source.get("kind") != expected_kind or not _source_uri_matches(
            config, source, expected_paths[source_id]
        ):
            raise ConfigError(
                f"source id {source_id!r} is already used outside the shipped {PACK_ID} pack"
            )
        merged.append(canonical[source_id])
        seen.add(source_id)
    merged.extend(canonical[source_id] for source_id in canonical if source_id not in seen)
    return merged


def _config_has_pack(config: Any, pack_root: Path, revisions: Mapping[str, str]) -> bool:
    sources = {str(source.get("id", "")): source for source in config.sources}
    if len(sources) != len(config.sources):
        return False
    for spec in _TASKS:
        source = sources.get(spec.source_id)
        if not isinstance(source, Mapping):
            return False
        expected_path = pack_root / "repositories" / spec.slug
        if (
            source.get("kind") != "repository"
            or source.get("partition") != "evaluation"
            or "evaluation" not in source.get("roles", [])
            or str(source.get("revision", "")).casefold() != revisions[spec.source_id]
            or not _source_uri_matches(config, source, expected_path)
        ):
            return False
    pack_source = sources.get(PACK_ID)
    return bool(
        isinstance(pack_source, Mapping)
        and pack_source.get("kind") == "task_pack"
        and pack_source.get("partition") == "evaluation"
        and "evaluation" in pack_source.get("roles", [])
        and _source_uri_matches(config, pack_source, pack_root / "tasks")
    )


def _selected_pack(config: Any) -> str | None:
    evaluation = config.data.get("evaluation", {})
    if isinstance(evaluation, Mapping) and evaluation.get("task_pack") == PACK_ID:
        return PACK_ID
    return None


def _descriptor(config: Any) -> dict[str, Any]:
    pack_root = config.artifact_dir.resolve() / "evaluation-packs" / PACK_ID
    revisions = _inspect_materialized_pack(pack_root) if pack_root.is_dir() else None
    installed = bool(revisions and _config_has_pack(config, pack_root, revisions))
    selected = _selected_pack(config) == PACK_ID
    return {
        "id": PACK_ID,
        "label": PACK_LABEL,
        "language": PACK_LANGUAGE,
        "license": PACK_LICENSE,
        "task_count": len(_TASKS),
        "independent_group_count": len(_TASKS),
        "description": PACK_DESCRIPTION,
        "status": "installed" if installed else "available",
        "selected": selected,
        "installed": installed,
        "runtime_image": RUNTIME_IMAGE,
        "checks": list(PACK_CHECKS),
    }


def list_evaluation_packs(config_path: str | Path) -> dict[str, Any]:
    """Return the stable GUI/API catalogue for the configured project."""

    config = load_config(config_path)
    selected = _selected_pack(config)
    return {"packs": [_descriptor(config)], "selected_pack": selected}


def _atomic_bytes(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_bytes(payload)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_optional_file(path: Path) -> bytes | None:
    if path.is_symlink():
        raise ConfigError(f"refusing to invalidate a symbolic-link artifact: {path}")
    if not path.exists():
        return None
    if not path.is_file():
        raise ConfigError(f"expected an artifact file but found another file type: {path}")
    return path.read_bytes()


def _restore_optional_file(path: Path, payload: bytes | None) -> None:
    if payload is None:
        path.unlink(missing_ok=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_bytes(path, payload)


def install_evaluation_pack(config_path: str | Path, pack_id: str) -> dict[str, Any]:
    """Install and select one shipped pack without touching immutable run evidence."""

    selected_id = str(pack_id).strip()
    if selected_id != PACK_ID:
        raise ConfigError(f"unknown evaluation pack: {selected_id or '<empty>'}")

    resolved_config = Path(config_path).expanduser().resolve()
    with project_mutation_gate(resolved_config):
        with project_config_mutation(resolved_config):
            config = load_config(resolved_config)
            artifact_dir = config.artifact_dir.resolve()
            pack_parent = artifact_dir / "evaluation-packs"
            pack_root = pack_parent / PACK_ID
            if pack_root.is_symlink():
                raise ConfigError(f"refusing to replace a symbolic-link evaluation pack: {pack_root}")
            pack_parent.mkdir(parents=True, exist_ok=True)

            original_config = resolved_config.read_bytes()
            freeze_path = artifact_dir / "dataset" / "freeze.json"
            plan_path = artifact_dir / "evaluation" / "current-plan.json"
            original_freeze = _read_optional_file(freeze_path)
            original_plan = _read_optional_file(plan_path)

            revisions = _inspect_materialized_pack(pack_root) if pack_root.is_dir() else None
            staging: Path | None = None
            backup: Path | None = None
            promoted = False
            try:
                if revisions is None:
                    if pack_root.exists() and _read_metadata(pack_root) is None:
                        raise ConfigError(
                            f"refusing to replace an unmanaged directory at {pack_root}"
                        )
                    staging = pack_parent / f".{PACK_ID}.{uuid4().hex}.staging"
                    staging.mkdir()
                    revisions = _materialize_pack(staging, pack_root)
                    if pack_root.exists():
                        backup = pack_parent / f".{PACK_ID}.{uuid4().hex}.backup"
                        pack_root.replace(backup)
                    staging.replace(pack_root)
                    promoted = True

                updated = dict(config.data)
                updated["sources"] = _merge_sources(config, pack_root, revisions)
                evaluation = dict(updated.get("evaluation", {}))
                evaluation["language"] = PACK_LANGUAGE
                evaluation["task_pack"] = PACK_ID
                updated["evaluation"] = evaluation
                environment = dict(updated.get("environment", {}))
                environment["image"] = RUNTIME_IMAGE
                updated["environment"] = environment
                report = validate_mapping(updated, root=config.root)
                if report.errors:
                    raise ConfigError("\n".join(report.errors))

                write_config(resolved_config, updated, overwrite=True)
                freeze_path.unlink(missing_ok=True)
                plan_path.unlink(missing_ok=True)
            except Exception:
                # Restore exact pre-call bytes. Immutable run directories are
                # never moved or removed by this service.
                _atomic_bytes(resolved_config, original_config)
                _restore_optional_file(freeze_path, original_freeze)
                _restore_optional_file(plan_path, original_plan)
                if promoted and pack_root.exists():
                    _remove_tree(pack_root)
                if backup is not None and backup.exists():
                    backup.replace(pack_root)
                if staging is not None and staging.exists():
                    _remove_tree(staging)
                raise
            else:
                if backup is not None and backup.exists():
                    _remove_tree(backup)

    return list_evaluation_packs(resolved_config)


__all__ = ["install_evaluation_pack", "list_evaluation_packs"]
