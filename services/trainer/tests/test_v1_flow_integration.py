from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE_ROOT / "src"))

from autotrainer.cli import main  # noqa: E402
from autotrainer.config import default_config, write_config  # noqa: E402
from autotrainer.project_service import prepare_project  # noqa: E402
from autotrainer.training_service import run_project_training  # noqa: E402


READY_DOCTOR = {
    "status": "ready",
    "sft_ready": True,
    "rl_ready": True,
    "python": {"status": "ready", "version": "3.11.9"},
    "gpu": {
        "status": "ready",
        "device_count": 1,
        "vram_gib": 24,
        "bf16_supported": True,
    },
    "packages": [],
    "sandbox": {"status": "ready"},
    "environment_image": {"status": "ready"},
    "errors": [],
}


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


def _task_manifest() -> dict[str, object]:
    """Return the smallest complete V1 task, including a resolvable verifier."""

    return {
        "version": "1.0",
        "task": {
            "id": "repair-homepage",
            "instruction": "Repair the homepage while preserving its public behavior.",
            "sourceId": "site",
            "startingRevision": "locked",
            "split": "train",
            "groupId": "homepage-family",
        },
        "runtime": {
            "workingDirectory": ".",
            "install": "",
            "build": "npm run build",
            "tests": "npm test",
            "browserTests": "",
        },
        "tools": [
            "list_files",
            "read_file",
            "search_code",
            "apply_patch",
            "run_check",
        ],
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


class V1FlowIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _create_repository(self, root: Path) -> str:
        repository = root / "site"
        (repository / "src").mkdir(parents=True)
        (repository / "src" / "App.tsx").write_text(
            "export function App() { return <main>Ready</main>; }\n",
            encoding="utf-8",
        )
        _git(repository, "init", "-q")
        _git(repository, "config", "user.name", "AutoTrainer Tests")
        _git(repository, "config", "user.email", "tests@example.invalid")
        _git(repository, "add", ".")
        _git(repository, "commit", "-qm", "Create the frontend fixture")
        return _git(repository, "rev-parse", "HEAD")

    def _create_project(self, recipe: str) -> Path:
        root = self.root / recipe
        root.mkdir()
        sources: list[dict[str, object]] = []

        if recipe in {"teach", "both"}:
            # Plain strings are an accepted human input, but the real compiler
            # must normalize them before the SFT recipe inspects the dataset.
            (root / "accepted.jsonl").write_text(
                json.dumps(
                    {
                        "prompt": "Make the status card easier to understand.",
                        "completion": "Updated the label and kept the existing behavior.",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            sources.append(
                {
                    "id": "accepted",
                    "kind": "sft_jsonl",
                    "uri": "accepted.jsonl",
                    "partition": "train",
                    "roles": ["demonstrations"],
                }
            )

        if recipe in {"practice", "both"}:
            commit = self._create_repository(root)
            verifier = root / "tasks" / "one" / "verifier"
            verifier.mkdir(parents=True)
            (verifier / "verify.mjs").write_text(
                "// Trusted fixture verifier; integration tests never execute it.\n",
                encoding="utf-8",
            )
            (verifier.parent / "task.json").write_text(
                json.dumps(_task_manifest()),
                encoding="utf-8",
            )
            sources.extend(
                [
                    {
                        "id": "site",
                        "kind": "repository",
                        "uri": "site",
                        "revision": commit,
                        "partition": "train",
                        "roles": ["style", "rl_seed"],
                        "include": ["src/**"],
                    },
                    {
                        "id": "tasks",
                        "kind": "task_pack",
                        "uri": "tasks",
                        "partition": "train",
                        "roles": ["rl_tasks"],
                    },
                ]
            )

        config = default_config(
            name=f"v1-{recipe}-integration",
            # A fixed immutable-looking revision lets the real recipe resolver
            # enforce the offline model identity without contacting Hugging Face.
            revision="a" * 40,
        )
        config["sources"] = sources
        config_path = root / "autotrainer.yaml"
        write_config(config_path, config, overwrite=False)
        return config_path

    def test_real_prepare_compile_and_auto_train_routes_all_v1_recipes(self) -> None:
        expected_counts = {
            "teach": {"sft_train": 1, "rl_train": 0},
            "practice": {"sft_train": 0, "rl_train": 1},
            "both": {"sft_train": 1, "rl_train": 1},
        }
        expected_stages = {
            "teach": ["sft"],
            "practice": ["grpo"],
            "both": ["sft", "grpo"],
        }

        for recipe in ("teach", "practice", "both"):
            with self.subTest(recipe=recipe):
                config_path = self._create_project(recipe)
                root = config_path.parent
                events: list[tuple[str, dict[str, object]]] = []

                def fake_sft(config: dict, **_: object) -> dict[str, object]:
                    events.append(("sft", config))
                    self.assertTrue(
                        (root / ".autotrainer" / "compiled" / "sft" / "train.jsonl").is_file()
                    )
                    return {"status": "completed", "stage": "sft"}

                def fake_grpo(config: dict, **_: object) -> dict[str, object]:
                    events.append(("grpo", config))
                    self.assertTrue(
                        (root / ".autotrainer" / "compiled" / "rl" / "train.jsonl").is_file()
                    )
                    return {"status": "completed", "stage": "grpo"}

                # These are the only machine-specific boundaries. Everything
                # before the stage runners remains the real V1 application path.
                with (
                    patch("autotrainer.project_service.require_materialized_model") as model_guard,
                    patch(
                        "autotrainer.project_service.run_doctor",
                        return_value=READY_DOCTOR,
                    ) as doctor,
                    patch(
                        "autotrainer.project_service.run_grpo_environment_canary",
                        return_value={
                            "status": "ready",
                            "task_id": "repair-homepage",
                            "baseline_reward": 0.2,
                            "task_pass_rate": 0.0,
                        },
                    ) as environment_canary,
                    patch(
                        "autotrainer.training_service.run_sft",
                        side_effect=fake_sft,
                    ),
                    patch(
                        "autotrainer.training_service.run_grpo",
                        side_effect=fake_grpo,
                    ),
                ):
                    prepared = prepare_project(config_path)

                    self.assertEqual(prepared["status"], "ready")
                    self.assertEqual(prepared["recipe"], recipe)
                    self.assertEqual(prepared["details"]["scan"]["blocking_errors"], [])
                    self.assertEqual(prepared["details"]["compile"]["errors"], [])
                    for key, count in expected_counts[recipe].items():
                        self.assertEqual(prepared["details"]["compile"]["counts"][key], count)
                    self.assertEqual(
                        sorted(prepared["details"]["preflight"]["recipes"]),
                        sorted(expected_stages[recipe]),
                    )

                    if recipe in {"practice", "both"}:
                        scanned_sources = prepared["details"]["scan"]["training"]["sources"]
                        scanned_repository = next(
                            source for source in scanned_sources if source["id"] == "site"
                        )
                        scanned_tasks = next(
                            source for source in scanned_sources if source["id"] == "tasks"
                        )
                        compiled_rl = json.loads(
                            (root / ".autotrainer" / "compiled" / "rl" / "train.jsonl")
                            .read_text(encoding="utf-8")
                            .splitlines()[0]
                        )
                        self.assertEqual(scanned_repository["eligible_file_count"], 1)
                        self.assertEqual(scanned_tasks["ready_task_count"], 1)
                        self.assertEqual(
                            compiled_rl["source_revision"],
                            scanned_repository["commit"],
                        )
                        self.assertEqual(compiled_rl["task_id"], "repair-homepage")

                    if recipe == "practice":
                        # Exercise an agent-facing entrypoint without replacing
                        # the shared preparation or auto-training service.
                        output = StringIO()
                        with redirect_stdout(output):
                            exit_code = main(
                                [
                                    "train",
                                    "auto",
                                    "--config",
                                    str(config_path),
                                    "--json",
                                ]
                            )
                        self.assertEqual(exit_code, 0)
                        trained = json.loads(output.getvalue())
                    else:
                        trained = run_project_training(config_path)

                self.assertGreaterEqual(model_guard.call_count, 2)
                self.assertEqual(doctor.call_count, 2)
                self.assertEqual(
                    environment_canary.call_count,
                    2 if recipe in {"practice", "both"} else 0,
                )
                self.assertEqual(trained["status"], "completed")
                self.assertEqual(trained["recipe"], recipe)
                self.assertEqual([stage for stage, _config in events], expected_stages[recipe])
                self.assertEqual(
                    [stage["stage"] for stage in trained["stages"]],
                    expected_stages[recipe],
                )

                for stage, stage_config in events:
                    self.assertEqual(
                        stage_config["sft"]["enabled"], recipe in {"teach", "both"}
                    )
                    self.assertEqual(
                        stage_config["grpo"]["enabled"], recipe in {"practice", "both"}
                    )
                    if stage == "grpo":
                        expected_start = (
                            "base"
                            if recipe == "practice"
                            else ".autotrainer/checkpoints/sft"
                        )
                        self.assertEqual(stage_config["grpo"]["start_from"], expected_start)

                if recipe in {"teach", "both"}:
                    compiled_sft = json.loads(
                        (root / ".autotrainer" / "compiled" / "sft" / "train.jsonl")
                        .read_text(encoding="utf-8")
                        .splitlines()[0]
                    )
                    self.assertIsInstance(compiled_sft["prompt"], list)
                    self.assertIsInstance(compiled_sft["completion"], list)
                    self.assertNotIn("messages", compiled_sft)


if __name__ == "__main__":
    unittest.main()
