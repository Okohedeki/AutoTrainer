"""Small shared view over the deterministic Git-history review core."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .history import HistoryError, list_history, review_history
from .project_service import read_project_config


def _workspace(history: Mapping[str, Any]) -> dict[str, Any]:
    errors = [str(value) for value in history.get("errors", [])]
    if errors:
        raise HistoryError(errors[0])
    summary = history.get("summary", {})
    if not isinstance(summary, Mapping):
        summary = {}
    # Rejected and approved diffs remain in deterministic local artifacts but
    # leave the human queue. The GUI receives only the next pending decisions.
    candidates = [
        {
            "candidate_id": item["candidate_id"],
            "proposed_instruction": item["proposed_instruction"],
            "files": item["files"],
            "patch": item["patch"],
            "flags": item.get("flags", []),
        }
        for item in history.get("candidates", [])
        if isinstance(item, Mapping) and item.get("decision") == "pending"
    ]
    return {
        "summary": {
            "reviewable_count": int(summary.get("pending", 0) or 0),
            "approved_count": int(summary.get("approved", 0) or 0),
            "blocked_counts": dict(history.get("excluded", {})),
        },
        "candidates": candidates,
    }


def get_history_workspace(config_path: str | Path) -> dict[str, Any]:
    """Return the privacy-limited review queue used by both clients."""

    config = read_project_config(config_path)
    return _workspace(list_history(config.data, config.root, write=True))


def review_history_candidate(
    config_path: str | Path,
    *,
    candidate_id: str,
    decision: str,
    instruction: str | None = None,
    rights_confirmed: bool = False,
) -> dict[str, Any]:
    """Persist one decision and return the refreshed shared review queue."""

    config = read_project_config(config_path)
    result = review_history(
        config.data,
        config.root,
        candidate_id=candidate_id,
        decision=decision,
        instruction=instruction,
        rights_confirmed=rights_confirmed,
    )
    return _workspace(result["history"])


__all__ = ["get_history_workspace", "review_history_candidate"]
