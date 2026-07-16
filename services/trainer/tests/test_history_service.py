from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE_ROOT / "src"))

from autotrainer.cli import main  # noqa: E402
from autotrainer.config import default_config, write_config  # noqa: E402
from autotrainer.history_service import (  # noqa: E402
    get_history_workspace,
    retire_stale_reviews,
    review_history_candidate,
)


class HistoryServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.config_path = self.root / "autotrainer.yaml"
        write_config(self.config_path, default_config(), overwrite=False)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def history(self) -> dict[str, object]:
        return {
            "errors": [],
            "excluded": {"generated_path": 2},
            "summary": {"pending": 1, "approved": 1, "stale_reviews": 0},
            "candidates": [
                {
                    "candidate_id": "pending",
                    "decision": "pending",
                    "proposed_instruction": "Improve the form label.",
                    "files": [],
                    "patch": "diff",
                    "flags": [],
                },
                {
                    "candidate_id": "approved",
                    "decision": "approved",
                    "proposed_instruction": "Approved.",
                    "files": [],
                    "patch": "private accepted diff",
                    "flags": [],
                },
            ],
        }

    def test_workspace_returns_only_pending_diffs_and_aggregate_counts(self) -> None:
        with patch("autotrainer.history_service.list_history", return_value=self.history()):
            result = get_history_workspace(self.config_path)

        self.assertEqual(result["summary"]["reviewable_count"], 1)
        self.assertEqual(result["summary"]["approved_count"], 1)
        self.assertEqual(result["summary"]["stale_review_count"], 0)
        self.assertEqual(result["summary"]["blocked_counts"], {"generated_path": 2})
        self.assertEqual(
            [item["candidate_id"] for item in result["candidates"]],
            ["pending"],
        )

    def test_review_returns_the_refreshed_queue(self) -> None:
        reviewed = {"history": self.history(), "review": {"decision": "approved"}}
        with patch("autotrainer.history_service.review_history", return_value=reviewed) as review:
            result = review_history_candidate(
                self.config_path,
                candidate_id="sha256:" + "a" * 64,
                decision="approved",
                instruction="Improve the form label.",
                rights_confirmed=True,
            )
        self.assertEqual(result["summary"]["approved_count"], 1)
        review.assert_called_once()

    def test_retire_returns_the_refreshed_queue(self) -> None:
        history = self.history()
        history["summary"]["stale_reviews"] = 0  # type: ignore[index]
        retired = {"history": history, "retired_count": 1}
        with patch(
            "autotrainer.history_service.retire_stale_history_reviews",
            return_value=retired,
        ) as retire:
            result = retire_stale_reviews(self.config_path)
        self.assertEqual(result["summary"]["stale_review_count"], 0)
        retire.assert_called_once()

    def test_agent_cli_uses_the_same_history_workspace(self) -> None:
        workspace = {
            "summary": {"reviewable_count": 0, "approved_count": 0},
            "candidates": [],
        }
        output = StringIO()
        with (
            patch(
                "autotrainer.history_service.get_history_workspace",
                return_value=workspace,
            ) as get,
            redirect_stdout(output),
        ):
            code = main(
                ["history", "list", "--config", str(self.config_path), "--json"]
            )
        self.assertEqual(code, 0)
        self.assertIn('"reviewable_count": 0', output.getvalue())
        get.assert_called_once_with(self.config_path)


if __name__ == "__main__":
    unittest.main()
