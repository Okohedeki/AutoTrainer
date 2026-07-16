from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE_ROOT / "src"))

from autotrainer.config import default_config, write_config  # noqa: E402
from autotrainer.cli import main  # noqa: E402
from autotrainer.history_service import (  # noqa: E402
    retire_stale_reviews,
    review_history_candidate,
)
from autotrainer.model_cache import materialize_model  # noqa: E402
from autotrainer.model_service import select_model  # noqa: E402
from autotrainer.project_gate import (  # noqa: E402
    ProjectBusyError,
    project_run_gate,
)
from autotrainer.project_service import prepare_project  # noqa: E402
from autotrainer.source_service import add_source, remove_source  # noqa: E402


class ProjectGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.config_path = self.root / "autotrainer.yaml"
        write_config(self.config_path, default_config(), overwrite=False)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_active_run_rejects_all_setup_mutations_and_external_prepare(self) -> None:
        """Every GUI/CLI setup boundary fails before touching project state."""

        operations = (
            lambda: select_model(self.config_path, "qwen3.5-9b-text"),
            lambda: materialize_model(self.config_path),
            lambda: add_source(self.config_path, str(self.root / "examples.jsonl")),
            lambda: remove_source(self.config_path, "missing"),
            lambda: review_history_candidate(
                self.config_path,
                candidate_id="sha256:" + "a" * 64,
                decision="rejected",
            ),
            lambda: retire_stale_reviews(self.config_path),
        )
        with project_run_gate(self.config_path):
            for operation in operations:
                with self.subTest(operation=operation):
                    with self.assertRaisesRegex(ProjectBusyError, "project is busy"):
                        operation()
            # Context variables distinguish the run owner's internal Prepare
            # from an unrelated API/CLI request arriving on another thread.
            with ThreadPoolExecutor(max_workers=1) as executor:
                external_prepare = executor.submit(prepare_project, self.config_path)
                with self.assertRaisesRegex(ProjectBusyError, "project is busy"):
                    external_prepare.result(timeout=2)

    def test_run_owner_can_call_prepare_without_reacquiring_the_lease(self) -> None:
        prepared = {"status": "ready", "recipe": "teach"}
        with (
            project_run_gate(self.config_path),
            patch(
                "autotrainer.project_service._prepare_project_owned",
                return_value=prepared,
            ) as inner,
        ):
            self.assertEqual(prepare_project(self.config_path), prepared)
        inner.assert_called_once_with(self.config_path)

    def test_legacy_cli_artifact_mutation_rejects_an_active_run(self) -> None:
        error_output = StringIO()
        with project_run_gate(self.config_path), redirect_stderr(error_output):
            exit_code = main(
                ["compile", "--config", str(self.config_path), "--json"]
            )

        self.assertEqual(exit_code, 2)
        self.assertIn("project is busy", error_output.getvalue())

    def test_os_lease_rejects_a_second_python_process(self) -> None:
        """The guard is process-wide, not merely a lock between GUI threads."""

        script = (
            "from autotrainer.project_gate import ProjectBusyError, acquire_project_lease\n"
            "import sys\n"
            "try:\n"
            "    acquire_project_lease(sys.argv[1])\n"
            "except ProjectBusyError:\n"
            "    raise SystemExit(0)\n"
            "raise SystemExit(3)\n"
        )
        environment = dict(os.environ)
        existing_pythonpath = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = os.pathsep.join(
            value
            for value in (str(SERVICE_ROOT / "src"), existing_pythonpath)
            if value
        )
        with project_run_gate(self.config_path):
            completed = subprocess.run(
                [sys.executable, "-c", script, str(self.config_path)],
                check=False,
                capture_output=True,
                env=environment,
                text=True,
                timeout=10,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
