"""Bounded GitHub repository discovery for the human source picker.

The GUI needs names and small public metadata, never arbitrary API responses or
credentials. Clone/pin remains owned by :mod:`autotrainer.source_service` after
the operator chooses a result and declares how it may be used.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
import os
import re
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen

from .config import ConfigError


_MIN_QUERY_LENGTH = 2
_MAX_QUERY_LENGTH = 100
_DEFAULT_LIMIT = 8
_MAX_LIMIT = 12
_MAX_RESPONSE_BYTES = 2 * 1024 * 1024
_FULL_NAME = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")


class GitHubSearchError(RuntimeError):
    """Stable public failure for remote repository discovery."""


def _nonnegative_integer(value: object) -> int:
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return 0
    return max(0, parsed)


def _github_request(query: str, limit: int) -> Mapping[str, Any]:
    """Read one bounded response from GitHub's fixed search endpoint."""

    url = "https://api.github.com/search/repositories?" + urlencode(
        {
            "q": query,
            "sort": "stars",
            "order": "desc",
            "per_page": limit,
        }
    )
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "AutoTrainer/0.1",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if token:
        # Authentication is process-local and is never returned or persisted.
        headers["Authorization"] = f"Bearer {token}"
    request = Request(url, headers=headers, method="GET")
    try:
        with urlopen(request, timeout=10) as response:
            final = urlsplit(response.geturl())
            if final.scheme != "https" or final.hostname != "api.github.com":
                raise GitHubSearchError("GitHub repository search returned an unsafe redirect.")
            body = response.read(_MAX_RESPONSE_BYTES + 1)
    except HTTPError as error:
        if error.code in {403, 429}:
            raise GitHubSearchError(
                "GitHub repository search is rate-limited; retry shortly or set GH_TOKEN."
            ) from error
        if error.code == 401:
            raise GitHubSearchError(
                "GitHub repository search authentication failed; refresh GH_TOKEN."
            ) from error
        raise GitHubSearchError(
            "GitHub repository search is unavailable; check network access."
        ) from error
    except (URLError, TimeoutError, OSError) as error:
        raise GitHubSearchError(
            "GitHub repository search is unavailable; check network access."
        ) from error
    if len(body) > _MAX_RESPONSE_BYTES:
        raise GitHubSearchError("GitHub repository search returned too much data.")
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GitHubSearchError("GitHub repository search returned invalid data.") from error
    if not isinstance(payload, Mapping):
        raise GitHubSearchError("GitHub repository search returned invalid data.")
    return payload


def _search_record(record: object) -> dict[str, Any] | None:
    if not isinstance(record, Mapping):
        return None
    full_name = str(record.get("full_name", "")).strip()
    if not _FULL_NAME.fullmatch(full_name):
        return None
    owner, repository = full_name.split("/", 1)
    description = " ".join(str(record.get("description") or "").split())[:280]
    language = str(record.get("language") or "").strip()[:80]
    license_value = record.get("license")
    spdx = (
        str(license_value.get("spdx_id") or "").strip()
        if isinstance(license_value, Mapping)
        else ""
    )
    if spdx in {"", "NOASSERTION", "OTHER"}:
        spdx = "UNDECLARED"
    return {
        "full_name": full_name,
        "clone_url": f"https://github.com/{owner}/{repository}.git",
        "description": description,
        "language": language or None,
        "stars": _nonnegative_integer(record.get("stargazers_count")),
        "fork": bool(record.get("fork", False)),
        "archived": bool(record.get("archived", False)),
        "private": bool(record.get("private", False)),
        "default_branch": str(record.get("default_branch") or "").strip()[:200],
        "license_spdx": spdx,
    }


def search_repositories(
    query: str,
    *,
    limit: int = _DEFAULT_LIMIT,
) -> list[dict[str, Any]]:
    """Search GitHub while returning only clone-safe repository identities."""

    if not isinstance(query, str):
        raise ConfigError("repository search query must be text")
    normalized = query.strip()
    if not _MIN_QUERY_LENGTH <= len(normalized) <= _MAX_QUERY_LENGTH:
        raise ConfigError(
            f"repository search query must be {_MIN_QUERY_LENGTH}-{_MAX_QUERY_LENGTH} characters"
        )
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= _MAX_LIMIT:
        raise ConfigError(f"repository search limit must be between 1 and {_MAX_LIMIT}")

    payload = _github_request(normalized, limit)
    items = payload.get("items", [])
    if not isinstance(items, list):
        raise GitHubSearchError("GitHub repository search returned invalid data.")
    results: list[dict[str, Any]] = []
    for record in items[:limit]:
        normalized_record = _search_record(record)
        if normalized_record is not None:
            results.append(normalized_record)
    return results


__all__ = ["GitHubSearchError", "search_repositories"]
