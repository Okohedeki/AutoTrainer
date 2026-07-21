"""Bounded merged-pull-request discovery for licensed GitHub training sources.

Configured repository sources are the project's allowlist. This service reads
only PRs merged into ``main`` or ``master``, proves their merge commit belongs
to the pinned local checkout, and stores a credential-free local catalog. The
catalog is later consumed by dataset design; no remote response becomes
training data directly.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen

from .config import ConfigError, load_config
from .source_service import list_sources


CATALOG_SCHEMA_VERSION = 1
MAX_PULL_REQUESTS = 100
MAX_RESPONSE_BYTES = 4 * 1024 * 1024
MAX_BODY_CHARS = 4_000
MERGE_BRANCHES = {"main", "master"}
_COMMIT = re.compile(r"[0-9a-f]{40,64}", re.IGNORECASE)
_FULL_NAME = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")


class GitHubPullRequestError(RuntimeError):
    """Stable public error for PR dataset discovery."""


def _artifact_dir(config: Any) -> Path:
    return (config.artifact_dir / "dataset" / "github-prs").resolve()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _bounded_text(value: object, limit: int) -> str:
    return " ".join(str(value or "").replace("\x00", "").split())[:limit]


def _github_request(owner: str, repository: str) -> Sequence[object]:
    query = urlencode(
        {
            "state": "closed",
            "sort": "updated",
            "direction": "desc",
            "per_page": MAX_PULL_REQUESTS,
        }
    )
    url = f"https://api.github.com/repos/{owner}/{repository}/pulls?{query}"
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "AutoTrainer/0.1",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(url, headers=headers, method="GET")
    try:
        with urlopen(request, timeout=20) as response:
            final = urlsplit(response.geturl())
            if final.scheme != "https" or final.hostname != "api.github.com":
                raise GitHubPullRequestError("GitHub PR discovery returned an unsafe redirect.")
            body = response.read(MAX_RESPONSE_BYTES + 1)
    except HTTPError as error:
        if error.code in {403, 429}:
            raise GitHubPullRequestError(
                "GitHub PR discovery is rate-limited; retry shortly or set GH_TOKEN."
            ) from error
        if error.code == 401:
            raise GitHubPullRequestError(
                "GitHub PR discovery authentication failed; refresh GH_TOKEN."
            ) from error
        if error.code == 404:
            raise GitHubPullRequestError(
                "The allowlisted GitHub repository was not found or is not accessible."
            ) from error
        raise GitHubPullRequestError(
            "GitHub PR discovery is unavailable; check network access."
        ) from error
    except (URLError, TimeoutError, OSError) as error:
        raise GitHubPullRequestError(
            "GitHub PR discovery is unavailable; check network access."
        ) from error
    if len(body) > MAX_RESPONSE_BYTES:
        raise GitHubPullRequestError("GitHub PR discovery returned too much data.")
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GitHubPullRequestError("GitHub PR discovery returned invalid data.") from error
    if not isinstance(payload, list):
        raise GitHubPullRequestError("GitHub PR discovery returned invalid data.")
    return payload


def _git_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_TERMINAL_PROMPT": "0",
        }
    )
    return environment


def _is_shallow_repository(repository: Path) -> bool:
    try:
        completed = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={repository.as_posix()}",
                "-C",
                str(repository),
                "rev-parse",
                "--is-shallow-repository",
            ],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
            env=_git_environment(),
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as error:
        raise GitHubPullRequestError(
            "The pinned repository could not inspect its Git history."
        ) from error
    value = completed.stdout.strip().casefold()
    if completed.returncode or value not in {"true", "false"}:
        raise GitHubPullRequestError(
            "The pinned repository could not inspect its Git history."
        )
    return value == "true"


def _head_commit(repository: Path) -> str:
    try:
        completed = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={repository.as_posix()}",
                "-C",
                str(repository),
                "rev-parse",
                "--verify",
                "HEAD^{commit}",
            ],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
            env=_git_environment(),
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as error:
        raise GitHubPullRequestError(
            "The pinned repository could not verify its checked-out revision."
        ) from error
    commit = completed.stdout.strip().casefold()
    if completed.returncode or not _COMMIT.fullmatch(commit):
        raise GitHubPullRequestError(
            "The pinned repository could not verify its checked-out revision."
        )
    return commit


def _ensure_complete_history(repository: Path, revision: str) -> None:
    """Upgrade legacy depth-one clones before proving merged-PR ancestry."""

    if _head_commit(repository) != revision.casefold():
        raise GitHubPullRequestError(
            "The pinned repository checkout no longer matches its configured revision."
        )
    if not _is_shallow_repository(repository):
        return
    try:
        completed = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={repository.as_posix()}",
                "-C",
                str(repository),
                "fetch",
                "--quiet",
                "--filter=blob:none",
                "--unshallow",
                "--no-tags",
                "origin",
            ],
            check=False,
            capture_output=True,
            timeout=300,
            env=_git_environment(),
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as error:
        raise GitHubPullRequestError(
            "The pinned repository has incomplete shallow history and AutoTrainer "
            "could not load the commit graph required to verify merged pull requests."
        ) from error
    if completed.returncode or _is_shallow_repository(repository):
        raise GitHubPullRequestError(
            "The pinned repository has incomplete shallow history and AutoTrainer "
            "could not load the commit graph required to verify merged pull requests."
        )
    if _head_commit(repository) != revision.casefold():
        raise GitHubPullRequestError(
            "The pinned repository checkout changed while its history was refreshed."
        )


def _git_is_ancestor(repository: Path, commit: str, revision: str) -> bool:
    if _is_shallow_repository(repository):
        raise GitHubPullRequestError(
            "The pinned repository has incomplete shallow history; refresh the source "
            "before importing merged pull requests."
        )
    try:
        object_check = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={repository.as_posix()}",
                "-C",
                str(repository),
                "cat-file",
                "-e",
                f"{commit}^{{commit}}",
            ],
            check=False,
            capture_output=True,
            timeout=60,
            env=_git_environment(),
        )
        if object_check.returncode:
            raise GitHubPullRequestError(
                "The pinned repository does not contain the commit graph required "
                "to verify a merged pull request."
            )
        completed = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={repository.as_posix()}",
                "-C",
                str(repository),
                "merge-base",
                "--is-ancestor",
                commit,
                revision,
            ],
            check=False,
            capture_output=True,
            timeout=20,
            env=_git_environment(),
        )
    except GitHubPullRequestError:
        raise
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as error:
        raise GitHubPullRequestError(
            "The pinned repository could not verify merged PR commits."
        ) from error
    if completed.returncode == 0:
        return True
    if completed.returncode == 1:
        return False
    raise GitHubPullRequestError(
        "The pinned repository could not verify merged PR ancestry from its commit graph."
    )


def _pull_request_record(value: object) -> dict[str, Any] | None:
    if not isinstance(value, Mapping) or value.get("merged_at") is None:
        return None
    base = value.get("base")
    base_ref = str(base.get("ref", "")).strip() if isinstance(base, Mapping) else ""
    if base_ref.casefold() not in MERGE_BRANCHES:
        return None
    merge_commit = str(value.get("merge_commit_sha", "")).strip().lower()
    if not _COMMIT.fullmatch(merge_commit):
        return None
    number = value.get("number")
    if isinstance(number, bool) or not isinstance(number, int) or number < 1:
        return None
    merged_at = str(value.get("merged_at", "")).strip()
    try:
        datetime.fromisoformat(merged_at.replace("Z", "+00:00"))
    except ValueError:
        return None
    title = _bounded_text(value.get("title"), 500)
    if not title:
        return None
    return {
        "base_branch": base_ref,
        "body": _bounded_text(value.get("body"), MAX_BODY_CHARS),
        "merge_commit": merge_commit,
        "merged_at": merged_at,
        "number": number,
        "title": title,
    }


def _github_training_sources(config_path: str | Path) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    config = load_config(config_path)
    serialized = {item["id"]: item for item in list_sources(config.path)}
    selected: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for source in config.sources:
        source_id = str(source.get("id", ""))
        public = serialized.get(source_id, {})
        roles = source.get("roles", [])
        if (
            source.get("kind") == "repository"
            and source.get("partition", "train") == "train"
            and isinstance(roles, Sequence)
            and not isinstance(roles, (str, bytes, bytearray))
            and "history" in roles
            and public.get("origin") == "github"
        ):
            selected.append((source, public))
    return selected


def _require_license(source: Mapping[str, Any]) -> str:
    license_value = source.get("license")
    spdx = (
        str(license_value.get("spdx", "")).strip()
        if isinstance(license_value, Mapping)
        else ""
    )
    if spdx.casefold() in {"", "undeclared", "noassertion", "other"}:
        raise ConfigError(
            "GitHub training sources require a declared SPDX license before PR discovery."
        )
    return spdx


def sync_merged_pull_requests(config_path: str | Path) -> dict[str, Any]:
    """Refresh local PR catalogs for every licensed allowlisted GitHub source."""

    config = load_config(config_path)
    reports: list[dict[str, Any]] = []
    for source, public in _github_training_sources(config.path):
        source_id = str(source["id"])
        spdx = _require_license(source)
        full_name = str(public.get("label", ""))
        if not _FULL_NAME.fullmatch(full_name):
            raise GitHubPullRequestError("GitHub source identity is invalid.")
        owner, repository_name = full_name.split("/", 1)
        repository = config.resolve_path(str(source.get("uri", "")))
        revision = str(source.get("revision", "")).strip().lower()
        if not repository.is_dir() or not _COMMIT.fullmatch(revision):
            raise GitHubPullRequestError(
                "GitHub PR discovery requires a pinned local repository checkout."
            )
        _ensure_complete_history(repository, revision)

        records: list[dict[str, Any]] = []
        skipped = 0
        seen: set[tuple[int, str]] = set()
        for raw in _github_request(owner, repository_name)[:MAX_PULL_REQUESTS]:
            record = _pull_request_record(raw)
            if record is None or not _git_is_ancestor(
                repository, record["merge_commit"], revision
            ):
                skipped += 1
                continue
            identity = (int(record["number"]), str(record["merge_commit"]))
            if identity in seen:
                skipped += 1
                continue
            seen.add(identity)
            records.append(record)
        records.sort(key=lambda item: (str(item["merged_at"]), int(item["number"])), reverse=True)
        catalog = {
            "license_spdx": spdx,
            "pull_requests": records,
            "repository": full_name,
            "schema_version": CATALOG_SCHEMA_VERSION,
            "source_id": source_id,
            "source_revision": revision,
        }
        _atomic_json(_artifact_dir(config) / f"{source_id}.json", catalog)
        reports.append(
            {
                "merged_pull_request_count": len(records),
                "repository": full_name,
                "skipped_count": skipped,
                "source_id": source_id,
                "status": "synced",
            }
        )
    return {
        "source_count": len(reports),
        "sources": reports,
        "status": "synced" if reports else "no_github_training_sources",
    }


def read_merged_pull_request_catalog(config_path: str | Path) -> dict[str, Any]:
    """Inspect cached PR catalogs without making a network request."""

    config = load_config(config_path)
    sources: list[dict[str, Any]] = []
    for source, public in _github_training_sources(config.path):
        source_id = str(source["id"])
        path = _artifact_dir(config) / f"{source_id}.json"
        record: dict[str, Any] = {
            "license_spdx": _require_license(source),
            "merged_pull_request_count": 0,
            "repository": str(public.get("label", "")),
            "source_id": source_id,
            "status": "needs_sync",
        }
        if path.is_file():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                payload = None
            if (
                isinstance(payload, Mapping)
                and payload.get("schema_version") == CATALOG_SCHEMA_VERSION
                and payload.get("source_id") == source_id
                and payload.get("source_revision") == source.get("revision")
                and payload.get("repository") == public.get("label")
                and isinstance(payload.get("pull_requests"), list)
            ):
                record["status"] = "synced"
                record["merged_pull_request_count"] = len(payload["pull_requests"])
        sources.append(record)
    return {
        "merged_pull_request_count": sum(
            int(item["merged_pull_request_count"]) for item in sources
        ),
        "source_count": len(sources),
        "sources": sources,
        "status": (
            "ready"
            if sources and all(item["status"] == "synced" for item in sources)
            else "needs_sync"
            if sources
            else "no_github_training_sources"
        ),
    }


__all__ = [
    "GitHubPullRequestError",
    "read_merged_pull_request_catalog",
    "sync_merged_pull_requests",
]
