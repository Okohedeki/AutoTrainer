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


if __name__ == "__main__":
    unittest.main()
