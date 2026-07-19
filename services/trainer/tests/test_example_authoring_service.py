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
from autotrainer.config import ConfigError, default_config, load_config, write_config  # noqa: E402
from autotrainer.example_authoring_service import (  # noqa: E402
    create_authored_example,
    list_authored_examples,
    remove_authored_example,
)
from autotrainer.source_service import add_source  # noqa: E402
from autotrainer.sources import scan_sources  # noqa: E402


def run_git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


class ExampleAuthoringServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.config_path = self.root / "autotrainer.yaml"
        write_config(self.config_path, default_config(), overwrite=False)
        self.repository = self.root / "workspace"
        self.repository.mkdir()
        (self.repository / "app.tsx").write_text(
            "export const App = () => <main>Original</main>;\n",
            encoding="utf-8",
        )
        run_git(self.repository, "init")
        run_git(self.repository, "config", "user.name", "AutoTrainer Tests")
        run_git(self.repository, "config", "user.email", "tests@example.invalid")
        run_git(self.repository, "add", ".")
        run_git(self.repository, "commit", "-m", "locked fixture")

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def add_repository(self, mode: str = "accepted_changes") -> str:
        return add_source(
            self.config_path,
            str(self.repository),
            modes=[mode],
            require_modes=True,
        )["source"]["id"]

    def create_example(self, source_id: str, **overrides: object) -> dict:
        arguments: dict[str, object] = {
            "source_id": source_id,
            "instruction": (
                "Repair the account form so invalid email addresses have an inline error."
            ),
            "accepted_response": (
                "I updated the form validation and added an accessible inline error message."
            ),
            "rights_confirmed": True,
        }
        arguments.update(overrides)
        return create_authored_example(self.config_path, **arguments)  # type: ignore[arg-type]

    def test_creates_managed_example_and_compiles_repository_provenance(self) -> None:
        source_id = self.add_repository()

        result = self.create_example(source_id)

        example = result["example"]
        self.assertEqual(example["source_id"], source_id)
        self.assertTrue(example["rights_confirmed"])
        self.assertEqual(list_authored_examples(self.config_path), {"examples": result["examples"]})
        config = load_config(self.config_path)
        managed = next(
            source for source in config.sources if source["kind"] == "sft_jsonl"
        )
        self.assertEqual(managed["id"], "authored-sft-examples")

        scan = scan_sources(config.data, self.root)
        self.assertEqual(scan["errors"], [])
        compiled = compile_data(config.data, self.root, scan)
        self.assertEqual(compiled["errors"], [])
        self.assertEqual(compiled["counts"]["sft_train"], 1)
        row = json.loads(
            config.resolve_path(config.data["sft"]["dataset"]).read_text(
                encoding="utf-8"
            )
        )
        repository = next(
            item for item in scan["sources"] if item["id"] == source_id
        )
        self.assertEqual(row["source_id"], source_id)
        self.assertEqual(
            row["source_repository_identity"], repository["repository_identity"]
        )
        self.assertEqual(row["source_revision"], repository["commit"])

    def test_rejects_missing_rights_and_evaluation_repository(self) -> None:
        train_source = self.add_repository()
        with self.assertRaisesRegex(ConfigError, "rights_confirmed must be true"):
            self.create_example(train_source, rights_confirmed=False)

        write_config(self.config_path, default_config(), overwrite=True)
        evaluation_source = self.add_repository("evaluation_holdout")
        with self.assertRaisesRegex(ConfigError, "evaluation holdout"):
            self.create_example(evaluation_source)

    def test_removes_last_example_and_managed_source_declaration(self) -> None:
        source_id = self.add_repository()
        created = self.create_example(source_id)["example"]

        result = remove_authored_example(
            self.config_path,
            example_id=created["id"],
        )

        self.assertEqual(result["removed"]["id"], created["id"])
        self.assertEqual(result["examples"], [])
        self.assertEqual(
            [source["kind"] for source in load_config(self.config_path).sources],
            ["repository"],
        )


if __name__ == "__main__":
    unittest.main()
