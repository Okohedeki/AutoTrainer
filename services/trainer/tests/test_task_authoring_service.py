from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE_ROOT / "src"))

from autotrainer.config import ConfigError, default_config, load_config, write_config  # noqa: E402
from autotrainer.compiler import compile_data  # noqa: E402
from autotrainer.source_service import add_source  # noqa: E402
from autotrainer.sources import scan_sources  # noqa: E402
from autotrainer.task_authoring_service import (  # noqa: E402
    create_authored_task,
    list_authored_tasks,
    remove_authored_task,
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


class TaskAuthoringServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.config_path = self.root / "autotrainer.yaml"
        write_config(self.config_path, default_config(), overwrite=False)
        self.repository = self.root / "workspace"
        (self.repository / "app").mkdir(parents=True)
        (self.repository / "app" / "package.json").write_text(
            '{"scripts":{"build":"echo build","test":"echo test"}}\n',
            encoding="utf-8",
        )
        run_git(self.repository, "init")
        run_git(self.repository, "config", "user.name", "AutoTrainer Tests")
        run_git(self.repository, "config", "user.email", "tests@example.invalid")
        run_git(self.repository, "add", ".")
        run_git(self.repository, "commit", "-m", "locked fixture")
        self.revision = run_git(self.repository, "rev-parse", "HEAD")
        self.verifier = self.root / "hidden-verifier"
        self.verifier.mkdir()
        (self.verifier / "verify.mjs").write_text(
            "// Test-owned verifier fixture.\n",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def add_repository(self, modes: list[str]) -> str:
        return add_source(
            self.config_path,
            str(self.repository),
            modes=modes,
            require_modes=True,
        )["source"]["id"]

    def create_task(self, source_id: str, **overrides: str) -> dict:
        arguments = {
            "source_id": source_id,
            "instruction": (
                "Change the account form so invalid email addresses show an inline error "
                "without breaking the existing submit behavior."
            ),
            "working_directory": "app",
            "build": "npm run build",
            "tests": "npm test",
            "verifier_bundle": str(self.verifier),
            "verifier_command": "node /autotrainer-verifier/verify.mjs",
        }
        arguments.update(overrides)
        return create_authored_task(self.config_path, **arguments)

    def test_creates_declared_train_manifest_and_connects_managed_task_pack(self) -> None:
        source_id = self.add_repository(["practice_tasks"])

        result = self.create_task(source_id)

        task = result["task"]
        self.assertEqual(task["status"], "declared")
        self.assertEqual(task["split"], "train")
        self.assertEqual(task["source_id"], source_id)
        self.assertEqual(task["locked_revision"], self.revision)
        self.assertEqual(task["working_directory"], "app")
        self.assertIn("Prepare will execute", task["next_action"]["detail"])
        manifest = json.loads(Path(task["manifest_path"]).read_text(encoding="utf-8"))
        self.assertEqual(manifest["task"]["startingRevision"], "locked")
        self.assertEqual(manifest["runtime"]["build"], "npm run build")
        self.assertEqual(manifest["verifier"]["bundle"], str(self.verifier.resolve()))
        self.assertEqual(
            manifest["tools"],
            ["list_files", "read_file", "search_code", "apply_patch", "run_check"],
        )
        task_pack = next(
            source
            for source in load_config(self.config_path).sources
            if source["kind"] == "task_pack"
        )
        self.assertEqual(task_pack["id"], "authored-practice-tasks")
        self.assertEqual(task_pack["partition"], "train")
        self.assertEqual(list_authored_tasks(self.config_path)["tasks"], result["tasks"])

        # The guided artifact is not merely display state: it traverses the
        # same scan and compile path consumed by the GRPO trainer.
        config = load_config(self.config_path).data
        scan = scan_sources(config, self.root)
        self.assertEqual(scan["errors"], [])
        compiled = compile_data(config, self.root, scan)
        self.assertEqual(compiled["errors"], [])
        self.assertEqual(compiled["counts"]["rl_train"], 1)

    def test_evaluation_source_authors_isolated_task_and_selects_holdout_pack(self) -> None:
        source_id = self.add_repository(["evaluation_holdout"])

        result = self.create_task(source_id)

        self.assertEqual(result["task"]["split"], "evaluation")
        config = load_config(self.config_path)
        self.assertEqual(config.data["evaluation"]["task_pack"], "authored-evaluation-tasks")
        task_pack = next(source for source in config.sources if source["kind"] == "task_pack")
        self.assertEqual(task_pack["roles"], ["evaluation"])

    def test_rejects_reference_only_repo_moving_worktree_and_visible_verifier(self) -> None:
        source_id = self.add_repository(["reference_only"])
        with self.assertRaisesRegex(ConfigError, "not configured for executable practice"):
            self.create_task(source_id)

        # Reconfigure a fresh project for practice to exercise task boundaries.
        write_config(self.config_path, default_config(), overwrite=True)
        source_id = self.add_repository(["practice_tasks"])
        with self.assertRaisesRegex(ConfigError, "tracked directory"):
            self.create_task(source_id, working_directory="not-committed")
        visible_verifier = self.repository / "visible-verifier"
        visible_verifier.mkdir()
        (visible_verifier / "verify.mjs").write_text("// visible\n", encoding="utf-8")
        with self.assertRaisesRegex(ConfigError, "outside the editable repository"):
            self.create_task(source_id, verifier_bundle=str(visible_verifier))

    def test_requires_operator_authored_verifier_and_does_not_claim_readiness(self) -> None:
        source_id = self.add_repository(["practice_tasks"])
        with self.assertRaisesRegex(ConfigError, "existing local directory"):
            self.create_task(source_id, verifier_bundle=str(self.root / "missing"))
        with self.assertRaisesRegex(ConfigError, "/autotrainer-verifier"):
            self.create_task(source_id, verifier_command="node verify.mjs")
        with self.assertRaisesRegex(ConfigError, "at least 20"):
            self.create_task(source_id, instruction="Fix it")

    def test_removes_authored_task_and_its_empty_managed_pack_declaration(self) -> None:
        source_id = self.add_repository(["practice_tasks"])
        created = self.create_task(source_id)["task"]

        result = remove_authored_task(
            self.config_path,
            split="train",
            task_id=created["id"],
        )

        self.assertEqual(result["removed"]["id"], created["id"])
        self.assertEqual(result["tasks"], [])
        self.assertFalse(Path(created["manifest_path"]).exists())
        self.assertEqual(
            [source["kind"] for source in load_config(self.config_path).sources],
            ["repository"],
        )


if __name__ == "__main__":
    unittest.main()
