from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE_ROOT / "src"))

from autotrainer.config import ConfigError  # noqa: E402
from autotrainer.github_service import GitHubSearchError, search_repositories  # noqa: E402


class GitHubSearchTests(unittest.TestCase):
    def test_search_returns_only_bounded_clone_safe_metadata(self) -> None:
        payload = {
            "items": [
                {
                    "full_name": "apache/airflow",
                    "description": "  Platform   for workflows  ",
                    "language": "Python",
                    "stargazers_count": 42000,
                    "fork": False,
                    "archived": False,
                    "private": False,
                    "default_branch": "main",
                    "license": {"spdx_id": "Apache-2.0"},
                    "html_url": "https://attacker.invalid/not-returned",
                },
                {"full_name": "bad name/not safe", "stargazers_count": 1},
            ]
        }
        with patch("autotrainer.github_service._github_request", return_value=payload) as request:
            results = search_repositories("airflow", limit=8)

        request.assert_called_once_with("airflow", 8)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["full_name"], "apache/airflow")
        self.assertEqual(results[0]["clone_url"], "https://github.com/apache/airflow.git")
        self.assertEqual(results[0]["description"], "Platform for workflows")
        self.assertEqual(results[0]["license_spdx"], "Apache-2.0")
        self.assertNotIn("html_url", results[0])

    def test_search_validates_query_and_limit_before_network(self) -> None:
        with patch("autotrainer.github_service._github_request") as request:
            for query in ("", "a", "x" * 101):
                with self.subTest(query=query), self.assertRaises(ConfigError):
                    search_repositories(query)
            for limit in (0, 13, True):
                with self.subTest(limit=limit), self.assertRaises(ConfigError):
                    search_repositories("airflow", limit=limit)  # type: ignore[arg-type]
        request.assert_not_called()

    def test_invalid_upstream_shape_is_not_forwarded(self) -> None:
        with patch(
            "autotrainer.github_service._github_request",
            return_value={"items": "not-a-list"},
        ):
            with self.assertRaisesRegex(GitHubSearchError, "invalid data"):
                search_repositories("airflow")


if __name__ == "__main__":
    unittest.main()
