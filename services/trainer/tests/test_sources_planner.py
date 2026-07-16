from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE_ROOT / "src"))

from autotrainer.planner import build_plan  # noqa: E402
from autotrainer.sources import (  # noqa: E402
    _write_text_atomic,
    materialize_repository,
    scan_sources,
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


def create_repository(root: Path, name: str = "frontend") -> tuple[Path, str]:
    repository = root / name
    (repository / "src").mkdir(parents=True)
    (repository / "src" / "App.tsx").write_text(
        "export function App() { return <main>Hello</main>; }\n", encoding="utf-8"
    )
    (repository / "src" / "styles.css").write_text("main { display: grid; }\n", encoding="utf-8")
    (repository / ".gitignore").write_text(".autotrainer/\n", encoding="utf-8")
    run_git(repository, "init")
    run_git(repository, "config", "user.name", "AutoTrainer Tests")
    run_git(repository, "config", "user.email", "tests@example.invalid")
    run_git(repository, "add", ".")
    run_git(repository, "commit", "-m", "fixture")
    return repository, run_git(repository, "rev-parse", "HEAD")


def clone_repository(source: Path, destination: Path) -> Path:
    subprocess.run(
        ["git", "clone", "--quiet", str(source), str(destination)],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return destination


def create_bare_remote(
    root: Path, name: str = "remote", *, include_tree_symlink: bool = False
) -> tuple[Path, str]:
    worktree, commit = create_repository(root, f"{name}-worktree")
    if include_tree_symlink:
        link_target = root / f"{name}-link-target.txt"
        link_target.write_text("../outside\n", encoding="utf-8")
        blob = run_git(worktree, "hash-object", "-w", str(link_target))
        run_git(
            worktree,
            "update-index",
            "--add",
            "--cacheinfo",
            "120000",
            blob,
            "escape-link",
        )
        run_git(worktree, "commit", "-m", "add symlink fixture")
        commit = run_git(worktree, "rev-parse", "HEAD")

    remote = root / f"{name}.git"
    subprocess.run(
        ["git", "clone", "--quiet", "--bare", str(worktree), str(remote)],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return remote, commit


def repository_source(repository: Path, commit: str, source_id: str = "frontend") -> dict:
    return {
        "id": source_id,
        "kind": "repository",
        "uri": str(repository),
        "revision": commit,
        "partition": "train",
        "roles": ["style", "history", "rl_seed"],
        "include": ["src/**/*.tsx", "src/**/*.css"],
    }


def task_payload(source_id: str, commit: str, split: str = "train", task_id: str = "task-1") -> dict:
    return {
        "version": "0.1",
        "task": {
            "id": task_id,
            "instruction": "Improve the frontend layout while preserving existing behavior.",
            "startingSnapshot": f"{source_id}@{commit}",
            "split": split,
        },
        "runtime": {
            "stack": "react-vite-tailwind",
            "install": "pnpm install --frozen-lockfile",
            "build": "pnpm build",
            "tests": "pnpm test",
            "browserTests": "pnpm playwright test",
        },
        "tools": ["read_file", "apply_patch", "run_command"],
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
            "toolCalls": 20,
            "commandTimeoutSeconds": 60,
            "networkAccess": False,
        },
    }


class SourceScanTests(unittest.TestCase):
    def test_atomic_artifact_writes_use_unique_temporary_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "artifact.txt"
            contents = [f"complete payload {index}\n" for index in range(12)]

            with ThreadPoolExecutor(max_workers=6) as executor:
                list(
                    executor.map(
                        lambda content: _write_text_atomic(destination, content),
                        contents,
                    )
                )

            self.assertIn(destination.read_text(encoding="utf-8"), contents)
            self.assertEqual(list(root.glob(".artifact.txt.*.tmp")), [])

    def test_scans_committed_frontend_text_without_executing_repository_code(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository, commit = create_repository(root)
            marker = root / "must-not-exist"
            (repository / "run-me.ps1").write_text(
                f"Set-Content -LiteralPath '{marker}' -Value unsafe\n", encoding="utf-8"
            )
            config = {"sources": [repository_source(repository, commit)]}

            scan = scan_sources(config, root)

            source = scan["sources"][0]
            self.assertEqual(source["commit"], commit)
            self.assertRegex(source["repository_identity"], r"^sha256:[0-9a-f]{64}$")
            self.assertEqual(source["eligible_file_count"], 2)
            self.assertEqual([item["path"] for item in source["files"]], ["src/App.tsx", "src/styles.css"])
            self.assertTrue(source["dirty"])
            self.assertFalse(marker.exists())

    def test_repository_aliases_share_a_credential_safe_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository, commit = create_repository(root)
            first = repository_source(repository, commit, "training-site")
            second = repository_source(repository / ".", commit, "evaluation-site")

            scan = scan_sources({"sources": [first, second]}, root)

            identities = {
                source["id"]: source["repository_identity"]
                for source in scan["sources"]
            }
            self.assertEqual(identities["training-site"], identities["evaluation-site"])
            # The scan artifact records a digest, never a local path or a
            # credential-bearing remote URL.
            self.assertNotIn(str(repository), identities["training-site"])

    def test_equivalent_https_ssh_and_scp_origins_share_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            upstream, commit = create_repository(root, "upstream")
            variants = (
                "https://token:secret@GitHub.com/Org/Frontend.git/?access_token=hidden",
                "ssh://git@github.com:22/org/frontend.git/",
                "git@github.com:ORG/FRONTEND.git",
                "https://github.com/org/frontend/",
            )
            sources = []
            for index, origin in enumerate(variants):
                clone = clone_repository(upstream, root / f"clone-{index}")
                run_git(clone, "remote", "set-url", "origin", origin)
                sources.append(repository_source(clone, commit, f"source-{index}"))

            scan = scan_sources({"sources": sources}, root)

            identities = {source["repository_identity"] for source in scan["sources"]}
            self.assertEqual(len(identities), 1)
            self.assertRegex(next(iter(identities)), r"^sha256:[0-9a-f]{64}$")
            serialized = json.dumps(scan)
            self.assertNotIn("token:secret", serialized)
            self.assertNotIn("access_token=hidden", serialized)

    def test_separate_clones_at_different_commits_share_remote_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            upstream, first_commit = create_repository(root, "upstream")
            first_clone = clone_repository(upstream, root / "first-clone")
            (upstream / "src" / "App.tsx").write_text(
                "export function App() { return <main>Second</main>; }\n",
                encoding="utf-8",
            )
            run_git(upstream, "add", ".")
            run_git(upstream, "commit", "-m", "second fixture")
            second_commit = run_git(upstream, "rev-parse", "HEAD")
            second_clone = clone_repository(upstream, root / "second-clone")

            scan = scan_sources(
                {
                    "sources": [
                        repository_source(first_clone, first_commit, "training-site"),
                        {
                            **repository_source(
                                second_clone, second_commit, "evaluation-site"
                            ),
                            "partition": "evaluation",
                        },
                    ]
                },
                root,
            )

            first, second = scan["sources"]
            self.assertNotEqual(first["commit"], second["commit"])
            self.assertEqual(first["repository_identity"], second["repository_identity"])

    def test_remote_repository_requires_explicit_materialization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = {
                "sources": [
                    {
                        "id": "remote",
                        "kind": "repository",
                        "uri": "https://github.com/example/frontend.git",
                        "revision": "abc123",
                        "partition": "train",
                        "roles": ["style"],
                    }
                ]
            }
            scan = scan_sources(config, Path(directory))
            self.assertEqual(scan["sources"][0]["status"], "needs_materialization")
            self.assertEqual(scan["summary"]["needs_materialization_count"], 1)

    def test_validates_every_sft_jsonl_line_and_reports_line_numbers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "examples.jsonl"
            rows = [
                {"messages": [{"role": "user", "content": "Build a card"}, {"role": "assistant", "content": "Here is the patch"}]},
                {"prompt": "Build a navbar", "completion": "Here is the implementation"},
            ]
            dataset.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n{not-json}\n", encoding="utf-8"
            )
            config = {
                "sources": [
                    {
                        "id": "demos",
                        "kind": "sft_jsonl",
                        "uri": str(dataset),
                        "partition": "train",
                        "roles": ["demonstrations"],
                    }
                ]
            }

            source = scan_sources(config, root)["sources"][0]

            self.assertEqual(source["valid_record_count"], 2)
            self.assertEqual(source["invalid_record_count"], 1)
            self.assertEqual(source["status"], "blocked")
            self.assertTrue(any(":3:" in error for error in source["errors"]))

    def test_task_pack_resolves_repository_snapshot_and_declared_verifier(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository, commit = create_repository(root)
            tasks = root / "tasks"
            tasks.mkdir()
            (tasks / "task.json").write_text(
                json.dumps(task_payload("frontend", commit)), encoding="utf-8"
            )
            config = {
                "sources": [
                    {
                        "id": "tasks",
                        "kind": "task_pack",
                        "uri": str(tasks),
                        "partition": "train",
                        "roles": ["rl_tasks"],
                    },
                    repository_source(repository, commit),
                ]
            }

            scan = scan_sources(config, root)
            task_source = scan["sources"][0]
            task = task_source["tasks"][0]

            self.assertTrue(task["snapshot_resolved"])
            self.assertEqual(task["snapshot_revision"], commit)
            self.assertEqual(task["verifier_status"], "declared_command")
            self.assertTrue(task["ready"])
            self.assertEqual(task_source["ready_task_count"], 1)

    def test_task_pack_blocks_an_undeclared_snapshot_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tasks = root / "task.json"
            tasks.write_text(json.dumps(task_payload("missing", "abc123")), encoding="utf-8")
            config = {
                "sources": [
                    {
                        "id": "tasks",
                        "kind": "task_pack",
                        "uri": str(tasks),
                        "partition": "train",
                        "roles": ["rl_tasks"],
                    }
                ]
            }
            source = scan_sources(config, root)["sources"][0]
            self.assertEqual(source["status"], "blocked")
            self.assertTrue(any("undeclared repository" in error for error in source["errors"]))

    def test_write_produces_deterministic_lock_and_reference_documents(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository, commit = create_repository(root)
            config = {
                "project": {"artifact_dir": str(root / "artifacts")},
                "sources": [repository_source(repository, commit)],
            }

            first = scan_sources(config, root, write=True)
            lock = Path(first["artifacts"]["lock"])
            documents = Path(first["artifacts"]["documents"]["frontend"])
            first_lock = lock.read_bytes()
            first_documents = documents.read_bytes()
            second = scan_sources(config, root, write=True)

            self.assertEqual(first_lock, Path(second["artifacts"]["lock"]).read_bytes())
            self.assertEqual(first_documents, documents.read_bytes())
            records = [json.loads(line) for line in documents.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(records), 2)
            self.assertIn("text", records[0])


class RepositoryMaterializationTests(unittest.TestCase):
    @staticmethod
    def config(remote: Path, revision: str) -> dict:
        return {
            "project": {"artifact_dir": "artifacts"},
            "sources": [
                {
                    "id": "remote",
                    "kind": "repository",
                    "uri": str(remote),
                    "revision": revision,
                    "partition": "train",
                    "roles": ["style", "rl_seed"],
                }
            ],
        }

    def test_clones_declared_repository_to_deterministic_detached_commit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            remote, commit = create_bare_remote(root)
            config = self.config(remote, commit)
            original_uri = config["sources"][0]["uri"]

            result = materialize_repository(config, root, "remote")

            local = (root / "artifacts" / "sources" / "remote").resolve()
            self.assertEqual(Path(result["local_path"]), local)
            self.assertEqual(result["commit"], commit)
            self.assertEqual(run_git(local, "rev-parse", "HEAD"), commit)
            self.assertEqual(run_git(local, "branch", "--show-current"), "")
            self.assertEqual(result["updated_source"]["uri"], "artifacts/sources/remote")
            self.assertEqual(result["updated_source"]["revision"], commit)
            self.assertEqual(config["sources"][0]["uri"], original_uri)

            with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                materialize_repository(config, root, "remote")

    def test_rejects_undeclared_non_repository_and_unpinned_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            remote, _commit = create_bare_remote(root)

            with self.assertRaisesRegex(ValueError, "is not declared"):
                materialize_repository({"sources": []}, root, "remote")

            non_repository = {
                "sources": [
                    {
                        "id": "remote",
                        "kind": "sft_jsonl",
                        "uri": str(root / "examples.jsonl"),
                    }
                ]
            }
            with self.assertRaisesRegex(ValueError, "is not a repository"):
                materialize_repository(non_repository, root, "remote")

            unpinned = self.config(remote, "")
            with self.assertRaisesRegex(ValueError, "must declare a revision"):
                materialize_repository(unpinned, root, "remote")

    def test_rejects_path_escape_and_preserves_existing_destination(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            remote, commit = create_bare_remote(root)
            config = self.config(remote, commit)

            with self.assertRaisesRegex(ValueError, "must stay inside"):
                materialize_repository(config, root, "remote", Path("..") / "escape")

            occupied = root / "artifacts" / "sources" / "occupied"
            occupied.mkdir(parents=True)
            sentinel = occupied / "keep.txt"
            sentinel.write_text("do not replace\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                materialize_repository(config, root, "remote", Path("occupied"))
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "do not replace\n")

    def test_failed_revision_resolution_removes_partial_clone(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            remote, _commit = create_bare_remote(root)
            config = self.config(remote, "refs/heads/does-not-exist")
            local = root / "artifacts" / "sources" / "remote"

            with self.assertRaisesRegex(RuntimeError, "cannot resolve declared revision"):
                materialize_repository(config, root, "remote")

            self.assertFalse(local.exists())

    def test_rejects_repository_tree_symlinks_before_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            remote, commit = create_bare_remote(root, include_tree_symlink=True)
            config = self.config(remote, commit)
            local = root / "artifacts" / "sources" / "remote"

            with self.assertRaisesRegex(ValueError, "repository tree contains symlinks"):
                materialize_repository(config, root, "remote")

            self.assertFalse(local.exists())


class PlannerTests(unittest.TestCase):
    def test_repository_evidence_does_not_count_as_sft_or_rl_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository, commit = create_repository(root)
            config = {
                "project": {"name": "frontend-expert", "artifact_dir": ".autotrainer"},
                "model": {
                    "provider": "huggingface",
                    "id": "example/coder-9b",
                    "revision": "main",
                    "trust_remote_code": False,
                },
                "sources": [repository_source(repository, commit)],
                "sft": {"enabled": True, "output_dir": ".autotrainer/checkpoints/sft"},
                "grpo": {
                    "enabled": True,
                    "start_from": ".autotrainer/checkpoints/sft",
                },
                "environment": {"factory": "package.environment:create"},
                "evaluation": {"candidates": ["base", "sft", "rl"]},
            }
            scan = scan_sources(config, root)

            plan = build_plan(config, root, scan)

            self.assertEqual(plan["evidence"]["status"], "ready")
            self.assertFalse(plan["evidence"]["training_ready"])
            self.assertEqual(plan["stages"]["sft"]["status"], "blocked")
            self.assertEqual(plan["stages"]["grpo"]["status"], "blocked")
            self.assertTrue(any("sft.dataset" in blocker for blocker in plan["blockers"]))
            self.assertTrue(any("grpo.dataset" in blocker for blocker in plan["blockers"]))
            self.assertTrue(plan["static_only"])

    def test_sft_stage_is_inputs_ready_for_a_valid_declared_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "sft.jsonl"
            dataset.write_text(
                json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": "Build the component"},
                            {"role": "assistant", "content": "Implemented the component"},
                        ]
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            config = {
                "project": {"name": "frontend-expert"},
                "model": {
                    "provider": "huggingface",
                    "id": "Qwen/Qwen3.5-9B",
                    "revision": "a" * 40,
                    "trust_remote_code": False,
                },
                "sources": [
                    {
                        "id": "demos",
                        "kind": "sft_jsonl",
                        "uri": str(dataset),
                        "partition": "train",
                        "roles": ["demonstrations"],
                    }
                ],
                "sft": {"enabled": True, "dataset": str(dataset)},
                "grpo": {"enabled": False},
                "evaluation": {},
            }
            scan = scan_sources(config, root)
            plan = build_plan(config, root, scan)
            self.assertEqual(plan["stages"]["sft"]["status"], "inputs_ready")
            self.assertEqual(plan["stages"]["sft"]["valid_example_count"], 1)
            self.assertEqual(plan["status"], "inputs_ready")


if __name__ == "__main__":
    unittest.main()
