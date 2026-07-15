from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE_ROOT / "src"))

from autotrainer.compiler import compile_data  # noqa: E402
from autotrainer.config import default_config  # noqa: E402
from autotrainer.sources import scan_sources  # noqa: E402


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
            "tokenBudget": 2000,
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
    return config, scan_sources(config, root), task_path


class CompilerTests(unittest.TestCase):
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
                    "tokenBudget": 2000,
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
            self.assertEqual(
                rl_row["source_repository_identity"],
                scan["sources"][0]["repository_identity"],
            )

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
