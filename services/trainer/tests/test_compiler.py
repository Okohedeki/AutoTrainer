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

from autotrainer.compiler import _atomic_jsonl, compile_data  # noqa: E402
from autotrainer.config import default_config  # noqa: E402
from autotrainer.history import list_history, review_history  # noqa: E402
from autotrainer.sources import scan_sources  # noqa: E402
from autotrainer.training.common import (  # noqa: E402
    inspect_sft_dataset,
    validate_sft_token_lengths,
)


def _task_manifest(
    *, task_id: str, split: str, group_id: str, source_id: str = "site"
) -> dict:
    return {
        "version": "1.0",
        "task": {
            "id": task_id,
            "instruction": "Repair the component while preserving its existing public behavior.",
            "sourceId": source_id,
            "startingRevision": "locked",
            "split": split,
            "groupId": group_id,
        },
        "runtime": {
            "workingDirectory": ".",
            "install": "",
            "build": "npm run build",
            "tests": "npm test",
            "browserTests": "",
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
            "toolCalls": 20,
            "commandTimeoutSeconds": 30,
            "episodeTimeoutSeconds": 300,
            "networkAccess": False,
        },
    }


def _fixture(root: Path, *, evaluation: bool = False) -> tuple[dict, dict, Path]:
    repository = root / "site"
    (repository / "src").mkdir(parents=True)
    (repository / "src" / "App.tsx").write_text(
        "export const App = () => null;\n", encoding="utf-8"
    )
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "-c",
            "user.name=AutoTrainer Test",
            "-c",
            "user.email=test@autotrainer.local",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )

    train_sft = root / "accepted-train.jsonl"
    train_sft.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "Create a component."},
                    {"role": "assistant", "content": "Here is the verified patch."},
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )
    task_root = root / "tasks" / "train" / "one"
    verifier = task_root / "verifier"
    verifier.mkdir(parents=True)
    (verifier / "verify.mjs").write_text("// hidden verifier\n", encoding="utf-8")
    task_path = task_root / "task.json"
    task_path.write_text(
        json.dumps(_task_manifest(task_id="train-task", split="train", group_id="train-family")),
        encoding="utf-8",
    )

    sources = [
        {
            "id": "site",
            "kind": "repository",
            "uri": "site",
            "revision": "HEAD",
            "partition": "train",
            "roles": ["style", "rl_seed"],
            "include": ["src/**"],
        },
        {
            "id": "accepted-train",
            "kind": "sft_jsonl",
            "uri": train_sft.name,
            "partition": "train",
        },
        {
            "id": "tasks-train",
            "kind": "task_pack",
            "uri": "tasks/train",
            "partition": "train",
        },
    ]
    if evaluation:
        evaluation_sft = root / "accepted-evaluation.jsonl"
        evaluation_sft.write_text(train_sft.read_text(encoding="utf-8"), encoding="utf-8")
        evaluation_task_root = root / "tasks" / "evaluation" / "one"
        evaluation_verifier = evaluation_task_root / "verifier"
        evaluation_verifier.mkdir(parents=True)
        (evaluation_verifier / "verify.mjs").write_text(
            "// hidden verifier\n", encoding="utf-8"
        )
        (evaluation_task_root / "task.json").write_text(
            json.dumps(
                _task_manifest(
                    task_id="evaluation-task",
                    split="evaluation",
                    group_id="evaluation-family",
                    source_id="evaluation-site",
                )
            ),
            encoding="utf-8",
        )
        sources.extend(
            [
                {
                    "id": "evaluation-site",
                    "kind": "repository",
                    "uri": "site",
                    "revision": "HEAD",
                    "partition": "evaluation",
                    "roles": ["evaluation"],
                    "include": ["src/**"],
                },
                {
                    "id": "accepted-evaluation",
                    "kind": "sft_jsonl",
                    "uri": evaluation_sft.name,
                    "partition": "evaluation",
                },
                {
                    "id": "tasks-evaluation",
                    "kind": "task_pack",
                    "uri": "tasks/evaluation",
                    "partition": "evaluation",
                },
            ]
        )

    config = default_config(name="compiler-test", revision="a" * 40)
    config["sources"] = sources
    if evaluation:
        config["evaluation"]["task_pack"] = "tasks-evaluation"
        config["sft"]["eval_dataset"] = ".autotrainer/compiled/sft/evaluation.jsonl"
    return config, scan_sources(config, root), task_path


def _add_evaluation_task_pack(
    root: Path,
    config: dict,
    *,
    source_id: str,
    task_id: str,
    group_id: str,
    repository_source_id: str = "evaluation-site",
) -> None:
    task_root = root / "tasks" / source_id / "one"
    verifier = task_root / "verifier"
    verifier.mkdir(parents=True)
    (verifier / "verify.mjs").write_text("// hidden verifier\n", encoding="utf-8")
    (task_root / "task.json").write_text(
        json.dumps(
            _task_manifest(
                task_id=task_id,
                split="evaluation",
                group_id=group_id,
                source_id=repository_source_id,
            )
        ),
        encoding="utf-8",
    )
    config["sources"].append(
        {
            "id": source_id,
            "kind": "task_pack",
            "uri": task_root.parent.relative_to(root).as_posix(),
            "partition": "evaluation",
        }
    )


def _add_evaluation_repository(root: Path, config: dict, *, source_id: str) -> Path:
    repository = root / source_id
    (repository / "src").mkdir(parents=True)
    (repository / "src" / "App.tsx").write_text(
        "export const Secondary = () => null;\n", encoding="utf-8"
    )
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "-c",
            "user.name=AutoTrainer Test",
            "-c",
            "user.email=test@autotrainer.local",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    config["sources"].append(
        {
            "id": source_id,
            "kind": "repository",
            "uri": source_id,
            "revision": "HEAD",
            "partition": "evaluation",
            "roles": ["evaluation"],
            "include": ["src/**"],
        }
    )
    return repository


def _reviewable_history_fixture(root: Path) -> tuple[dict, Path, str]:
    repository = root / "history-site"
    (repository / "src").mkdir(parents=True)
    app = repository / "src" / "App.tsx"
    app.write_text("export const label = 'Before';\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "-c",
            "user.name=AutoTrainer Test",
            "-c",
            "user.email=test@autotrainer.local",
            "commit",
            "-qm",
            "Initial fixture",
        ],
        check=True,
    )
    parent = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()
    app.write_text("export const label = 'After';\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "-c",
            "user.name=AutoTrainer Test",
            "-c",
            "user.email=test@autotrainer.local",
            "commit",
            "-qm",
            "Make the application label easier to understand",
        ],
        check=True,
    )
    revision = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()
    config = default_config(name="history-compiler", revision="a" * 40)
    config["sources"] = [
        {
            "id": "history-site",
            "kind": "repository",
            "uri": repository.name,
            "revision": revision,
            "partition": "train",
            "roles": ["style", "history"],
            "include": ["src/**"],
        }
    ]
    return config, repository, parent


class CompilerTests(unittest.TestCase):
    def test_atomic_dataset_writes_use_unique_temporary_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "compiled.jsonl"
            record_sets = [[{"writer": index, "complete": True}] for index in range(12)]

            with ThreadPoolExecutor(max_workers=6) as executor:
                list(
                    executor.map(
                        lambda records: _atomic_jsonl(destination, records),
                        record_sets,
                    )
                )

            final_records = [
                json.loads(line)
                for line in destination.read_text(encoding="utf-8").splitlines()
            ]
            self.assertIn(final_records, record_sets)
            self.assertEqual(list(root.glob(".compiled.jsonl.*.tmp")), [])

    def test_compiles_explicit_sft_and_executable_tasks_but_not_raw_code(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = root / "site"
            (repository / "src").mkdir(parents=True)
            (repository / "src" / "App.tsx").write_text("export const App = () => null;\n", encoding="utf-8")
            subprocess.run(["git", "init", "-q", str(repository)], check=True)
            subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(repository),
                    "-c",
                    "user.name=AutoTrainer Test",
                    "-c",
                    "user.email=test@autotrainer.local",
                    "commit",
                    "-qm",
                    "fixture",
                ],
                check=True,
            )

            sft = root / "accepted.jsonl"
            sft.write_text(
                json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": "Create a component."},
                            {"role": "assistant", "content": "Here is the verified patch."},
                        ]
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            task_root = root / "tasks" / "one"
            verifier = task_root / "verifier"
            verifier.mkdir(parents=True)
            (verifier / "verify.mjs").write_text("// hidden verifier\n", encoding="utf-8")
            manifest = {
                "version": "1.0",
                "task": {
                    "id": "task-1",
                    "instruction": "Repair the component while preserving its existing public behavior.",
                    "sourceId": "site",
                    "startingRevision": "locked",
                    "split": "train",
                    "groupId": "site-family",
                },
                "runtime": {
                    "workingDirectory": ".",
                    "install": "",
                    "build": "npm run build",
                    "tests": "npm test",
                    "browserTests": "",
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
                    "toolCalls": 20,
                    "commandTimeoutSeconds": 30,
                    "episodeTimeoutSeconds": 300,
                    "networkAccess": False,
                },
            }
            (task_root / "task.json").write_text(json.dumps(manifest), encoding="utf-8")

            config = default_config(name="compiler-test", revision="a" * 40)
            config["sources"] = [
                {
                    "id": "site",
                    "kind": "repository",
                    "uri": "site",
                    "revision": "HEAD",
                    "partition": "train",
                    "roles": ["style", "rl_seed"],
                    "include": ["src/**"],
                },
                {
                    "id": "accepted",
                    "kind": "sft_jsonl",
                    "uri": "accepted.jsonl",
                    "partition": "train",
                },
                {
                    "id": "tasks",
                    "kind": "task_pack",
                    "uri": "tasks",
                    "partition": "train",
                },
            ]
            scan = scan_sources(config, root)
            self.assertEqual(scan["errors"], [])
            compiled = compile_data(config, root, scan)
            self.assertEqual(compiled["errors"], [])
            self.assertEqual(compiled["counts"]["sft_train"], 1)
            self.assertEqual(compiled["counts"]["rl_train"], 1)
            sft_rows = (root / ".autotrainer" / "compiled" / "sft" / "train.jsonl").read_text()
            self.assertNotIn("export const App", sft_rows)
            rl_row = json.loads(
                (root / ".autotrainer" / "compiled" / "rl" / "train.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()[0]
            )
            self.assertEqual(rl_row["source_revision"], scan["sources"][0]["commit"])
            system_prompt = rl_row["prompt"][0]["content"]
            self.assertIn("disposable code repository", system_prompt)
            self.assertNotIn("disposable frontend repository", system_prompt)
            self.assertIn("Inspection is not completion", system_prompt)
            self.assertIn("apply_patch", system_prompt)
            self.assertIn("run the named checks", system_prompt)
            self.assertEqual(
                rl_row["source_repository_identity"],
                scan["sources"][0]["repository_identity"],
            )
            self.assertEqual(
                compiled["repository_exposures"],
                [
                    {
                        "source_id": "site",
                        "partition": "train",
                        "repository_identity": scan["sources"][0][
                            "repository_identity"
                        ],
                        "commit": scan["sources"][0]["commit"],
                    }
                ],
            )
            verifier_identity = rl_row["verifier_identity"]
            self.assertEqual(verifier_identity["path"], str(verifier.resolve()))
            self.assertRegex(verifier_identity["sha256"], r"^sha256:[0-9a-f]{64}$")
            self.assertEqual(verifier_identity["files"][0]["path"], "verify.mjs")
            self.assertEqual(
                verifier_identity["files"][0]["bytes"],
                len((verifier / "verify.mjs").read_bytes()),
            )
            self.assertRegex(
                verifier_identity["files"][0]["sha256"], r"^[0-9a-f]{64}$"
            )

    def test_selected_evaluation_task_pack_controls_rows_and_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, _scan, _task_path = _fixture(root, evaluation=True)
            _add_evaluation_task_pack(
                root,
                config,
                source_id="tasks-evaluation-secondary",
                task_id="evaluation-task-secondary",
                group_id="evaluation-family-secondary",
            )

            first = compile_data(config, root, scan_sources(config, root))
            first_rows = [
                json.loads(line)
                for line in Path(first["artifacts"]["rl_evaluation"])
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            first_train_rows = [
                json.loads(line)
                for line in Path(first["artifacts"]["rl_train"])
                .read_text(encoding="utf-8")
                .splitlines()
            ]

            config["evaluation"]["task_pack"] = "tasks-evaluation-secondary"
            second = compile_data(config, root, scan_sources(config, root))
            second_rows = [
                json.loads(line)
                for line in Path(second["artifacts"]["rl_evaluation"])
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            second_train_rows = [
                json.loads(line)
                for line in Path(second["artifacts"]["rl_train"])
                .read_text(encoding="utf-8")
                .splitlines()
            ]

            self.assertEqual(first["errors"], [])
            self.assertEqual(second["errors"], [])
            self.assertEqual(first["counts"]["rl_evaluation"], 1)
            self.assertEqual(second["counts"]["rl_evaluation"], 1)
            self.assertEqual([row["task_id"] for row in first_rows], ["evaluation-task"])
            self.assertEqual(
                [row["task_id"] for row in second_rows],
                ["evaluation-task-secondary"],
            )
            self.assertEqual(
                [row["task_id"] for row in first_train_rows], ["train-task"]
            )
            self.assertEqual(
                [row["task_id"] for row in second_train_rows], ["train-task"]
            )
            self.assertNotEqual(first["fingerprint"], second["fingerprint"])

    def test_unselected_evaluation_pack_repository_does_not_affect_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, _scan, _task_path = _fixture(root, evaluation=True)

            baseline = compile_data(config, root, scan_sources(config, root))
            baseline_fingerprint = baseline["fingerprint"]
            baseline_exposures = baseline["repository_exposures"]

            secondary_repository = _add_evaluation_repository(
                root, config, source_id="secondary-evaluation-site"
            )
            _add_evaluation_task_pack(
                root,
                config,
                source_id="tasks-evaluation-secondary",
                task_id="evaluation-task-secondary",
                group_id="evaluation-family-secondary",
                repository_source_id="secondary-evaluation-site",
            )
            with_unselected_pack = compile_data(
                config, root, scan_sources(config, root)
            )

            self.assertEqual(with_unselected_pack["errors"], [])
            self.assertEqual(with_unselected_pack["fingerprint"], baseline_fingerprint)
            self.assertEqual(
                with_unselected_pack["repository_exposures"], baseline_exposures
            )
            self.assertNotIn(
                "secondary-evaluation-site",
                {
                    exposure["source_id"]
                    for exposure in with_unselected_pack["repository_exposures"]
                },
            )

            (secondary_repository / "src" / "App.tsx").write_text(
                "export const Secondary = () => 'changed';\n", encoding="utf-8"
            )
            subprocess.run(
                ["git", "-C", str(secondary_repository), "add", "."], check=True
            )
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(secondary_repository),
                    "-c",
                    "user.name=AutoTrainer Test",
                    "-c",
                    "user.email=test@autotrainer.local",
                    "commit",
                    "-qm",
                    "change unselected repository",
                ],
                check=True,
            )
            after_unselected_change = compile_data(
                config, root, scan_sources(config, root)
            )

            self.assertEqual(after_unselected_change["errors"], [])
            self.assertEqual(after_unselected_change["fingerprint"], baseline_fingerprint)
            self.assertEqual(
                after_unselected_change["repository_exposures"], baseline_exposures
            )

    def test_missing_or_wrong_evaluation_task_pack_selection_blocks_compile(self) -> None:
        cases = (
            (
                "not-declared",
                "evaluation.task_pack 'not-declared' is not a declared evaluation task_pack source",
            ),
            (
                "tasks-train",
                "evaluation.task_pack 'tasks-train' must name a task_pack source in the evaluation partition",
            ),
        )
        for selection, expected_error in cases:
            with self.subTest(selection=selection), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                config, _scan, _task_path = _fixture(root, evaluation=True)
                config["evaluation"]["task_pack"] = selection

                report = compile_data(config, root, scan_sources(config, root))

                self.assertIn(expected_error, report["errors"])
                self.assertEqual(report["artifacts"], {})

    def test_approved_history_compiles_alone_and_combines_with_explicit_jsonl(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, repository, _parent = _reviewable_history_fixture(root)
            candidate = list_history(config, root, write=False)["candidates"][0]
            review_history(
                config,
                root,
                candidate_id=candidate["candidate_id"],
                decision="approved",
                instruction="Make the application label clearer for people using the interface.",
                rights_confirmed=True,
            )
            accepted_revision = candidate["commit"]
            (repository / "src" / "extra.ts").write_text(
                "export const extra = true;\n", encoding="utf-8"
            )
            subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(repository),
                    "-c",
                    "user.name=AutoTrainer Test",
                    "-c",
                    "user.email=test@autotrainer.local",
                    "commit",
                    "-qm",
                    "Add a separate later change",
                ],
                check=True,
            )
            locked_revision = subprocess.run(
                ["git", "-C", str(repository), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
            ).stdout.strip()
            config["sources"][0]["revision"] = locked_revision

            history_only = compile_data(config, root, scan_sources(config, root))

            self.assertEqual(history_only["errors"], [])
            self.assertEqual(history_only["counts"]["sft_train"], 1)
            history_rows = [
                json.loads(line)
                for line in Path(history_only["artifacts"]["sft_train"])
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(history_rows[0]["source_type"], "approved_git_change")
            repository_identity = history_rows[0]["source_repository_identity"]
            self.assertNotEqual(accepted_revision, locked_revision)
            self.assertEqual(
                history_only["repository_exposures"],
                sorted(
                    [
                        {
                            "source_id": "history-site",
                            "partition": "train",
                            "repository_identity": repository_identity,
                            "commit": accepted_revision,
                        },
                        {
                            "source_id": "history-site",
                            "partition": "train",
                            "repository_identity": repository_identity,
                            "commit": locked_revision,
                        },
                    ],
                    key=lambda item: json.dumps(
                        item, sort_keys=True, separators=(",", ":")
                    ),
                ),
            )

            explicit = root / "accepted.jsonl"
            explicit.write_text(
                json.dumps({"prompt": "Build a card", "completion": "Verified result"})
                + "\n"
                + json.dumps(
                    {
                        "messages": [
                            {"role": "system", "content": "Follow the local style."},
                            {"role": "user", "content": "Build a navbar"},
                            {"role": "assistant", "content": "Verified navbar result"},
                        ]
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            config["sources"].append(
                {
                    "id": "accepted",
                    "kind": "sft_jsonl",
                    "uri": explicit.name,
                    "partition": "train",
                    "roles": ["demonstrations"],
                }
            )
            combined = compile_data(config, root, scan_sources(config, root))
            combined_rows = [
                json.loads(line)
                for line in Path(combined["artifacts"]["sft_train"])
                .read_text(encoding="utf-8")
                .splitlines()
            ]

            self.assertEqual(combined["errors"], [])
            self.assertEqual(combined["counts"]["sft_train"], 3)
            self.assertEqual(
                sum(row.get("source_type") == "approved_git_change" for row in combined_rows),
                1,
            )
            self.assertTrue(
                any(row["prompt"][0]["content"] == "Build a card" for row in combined_rows)
            )
            self.assertTrue(all("messages" not in row for row in combined_rows))
            self.assertTrue(
                all(
                    isinstance(row["prompt"], list)
                    and isinstance(row["completion"], list)
                    and all(message["role"] == "assistant" for message in row["completion"])
                    for row in combined_rows
                )
            )

            dataset_path = Path(combined["artifacts"]["sft_train"])
            self.assertEqual(
                inspect_sft_dataset(dataset_path)["format"],
                "conversational-prompt-completion",
            )

            class CountingTokenizer:
                def apply_chat_template(self, messages: list[dict], **_: object) -> list[int]:
                    return list(range(sum(len(message["content"]) for message in messages)))

            self.assertEqual(
                validate_sft_token_lengths(CountingTokenizer(), dataset_path, max_length=20_000)[
                    "record_count"
                ],
                3,
            )

    def test_pending_history_repository_compiles_zero_sft_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, _repository, _parent = _reviewable_history_fixture(root)

            report = compile_data(config, root, scan_sources(config, root))

            self.assertEqual(report["errors"], [])
            self.assertEqual(report["counts"]["sft_train"], 0)
            self.assertNotIn("sft_train", report["artifacts"])
            self.assertFalse(
                (root / ".autotrainer" / "compiled" / "sft" / "train.jsonl").exists()
            )

    def test_stale_history_approval_blocks_scan_and_compilation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, repository, parent = _reviewable_history_fixture(root)
            candidate = list_history(config, root, write=False)["candidates"][0]
            review_history(
                config,
                root,
                candidate_id=candidate["candidate_id"],
                decision="approved",
                instruction="Make the application label clearer for people using the interface.",
                rights_confirmed=True,
            )
            subprocess.run(
                ["git", "-C", str(repository), "checkout", "-q", "--detach", parent],
                check=True,
            )
            config["sources"][0]["revision"] = parent

            scan = scan_sources(config, root)
            report = compile_data(config, root, scan)

            self.assertEqual(scan["summary"]["stale_history_review_count"], 1)
            self.assertTrue(any("stale" in error for error in scan["errors"]))
            self.assertTrue(report["errors"])
            self.assertEqual(report["counts"]["sft_train"], 0)
            self.assertEqual(report["artifacts"], {})

    def test_aborts_before_writing_when_source_scan_has_errors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = default_config(name="scan-error", revision="a" * 40)
            config["project"]["artifact_dir"] = ".artifacts"

            report = compile_data(
                config,
                root,
                {"errors": ["broken source"], "sources": []},
            )

            self.assertTrue(any("source scan" in error for error in report["errors"]))
            self.assertEqual(report["artifacts"], {})
            self.assertFalse((root / ".artifacts").exists())

    def test_writes_all_partitions_to_declared_dataset_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, scan, _ = _fixture(root, evaluation=True)
            config["project"]["artifact_dir"] = ".artifacts"
            config["sft"]["dataset"] = ".artifacts/data/custom-sft-train.jsonl"
            config["sft"]["eval_dataset"] = ".artifacts/data/custom-sft-eval.jsonl"
            config["grpo"]["dataset"] = ".artifacts/data/custom-grpo-train.jsonl"
            config["evaluation"]["dataset"] = ".artifacts/data/custom-final-eval.jsonl"
            # This is a separately supplied training-validation input. The
            # compiler must neither replace nor report it as the final holdout.
            training_eval = root / ".artifacts" / "data" / "grpo-validation.jsonl"
            training_eval.parent.mkdir(parents=True)
            training_eval.write_text('{"task_id":"training-validation"}\n', encoding="utf-8")
            config["grpo"]["eval_dataset"] = ".artifacts/data/grpo-validation.jsonl"

            report = compile_data(config, root, scan)

            self.assertEqual(report["errors"], [])
            self.assertTrue(
                any("REPOSITORY HOLDOUT VIOLATION" in item for item in report["warnings"])
            )
            for key in (
                "sft_train",
                "sft_evaluation",
                "rl_train",
                "rl_evaluation",
            ):
                self.assertTrue(Path(report["artifacts"][key]).is_file())
            self.assertEqual(
                Path(report["artifacts"]["rl_evaluation"]),
                root / ".artifacts" / "data" / "custom-final-eval.jsonl",
            )
            self.assertEqual(
                training_eval.read_text(encoding="utf-8"),
                '{"task_id":"training-validation"}\n',
            )
            self.assertFalse((root / ".artifacts" / "compiled" / "sft" / "train.jsonl").exists())
            on_disk = json.loads(Path(report["artifacts"]["report"]).read_text(encoding="utf-8"))
            self.assertEqual(on_disk, report)

    def test_rejects_colliding_dataset_destinations_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, scan, _ = _fixture(root)
            config["project"]["artifact_dir"] = ".artifacts"
            config["sft"]["dataset"] = ".artifacts/data/train.jsonl"
            config["grpo"]["dataset"] = ".artifacts/data/train.jsonl"

            report = compile_data(config, root, scan)

            self.assertTrue(any("collide" in error for error in report["errors"]))
            self.assertEqual(report["artifacts"], {})
            self.assertFalse((root / ".artifacts").exists())

    def test_rejects_dataset_destination_outside_artifact_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, scan, _ = _fixture(root)
            config["project"]["artifact_dir"] = ".artifacts"
            config["sft"]["dataset"] = "outside/sft.jsonl"
            config["grpo"]["dataset"] = ".artifacts/data/grpo.jsonl"

            report = compile_data(config, root, scan)

            self.assertTrue(any("artifact_dir" in error for error in report["errors"]))
            self.assertEqual(report["artifacts"], {})
            self.assertFalse((root / ".artifacts").exists())

    def test_compile_error_does_not_leave_valid_source_as_partial_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, scan, task_path = _fixture(root)
            config["project"]["artifact_dir"] = ".artifacts"
            config["sft"]["dataset"] = ".artifacts/data/sft.jsonl"
            config["grpo"]["dataset"] = ".artifacts/data/grpo.jsonl"
            config["evaluation"]["dataset"] = ".artifacts/data/final-evaluation.jsonl"
            task_path.write_text("{invalid-json", encoding="utf-8")

            report = compile_data(config, root, scan)

            self.assertTrue(report["errors"])
            self.assertEqual(report["counts"]["sft_train"], 1)
            self.assertEqual(report["artifacts"], {})
            self.assertFalse((root / ".artifacts").exists())


if __name__ == "__main__":
    unittest.main()
