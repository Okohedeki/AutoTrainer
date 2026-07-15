"""Resolve mutable experiment inputs into an inspectable JSON lock."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping


COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40,64}$", re.IGNORECASE)


def _resolve_huggingface_revision(model_id: str, revision: str) -> str:
    if COMMIT_PATTERN.fullmatch(revision):
        return revision.lower()
    local = Path(model_id).expanduser()
    if local.is_dir() and (local / ".git").exists():
        result = subprocess.run(
            ["git", "-C", str(local), "rev-parse", f"{revision}^{{commit}}"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if result.returncode:
            raise RuntimeError(f"cannot resolve local model revision: {result.stderr.strip()}")
        return result.stdout.strip()
    url = f"https://huggingface.co/api/models/{model_id}/revision/{revision}"
    headers = {"User-Agent": "AutoTrainer/0.1"}
    token = os.environ.get("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=30) as response:
            payload = json.load(response)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as error:
        raise RuntimeError(
            f"cannot resolve {model_id}@{revision}; check network access, model access, and HF_TOKEN: {error}"
        ) from error
    sha = payload.get("sha")
    if not isinstance(sha, str) or not COMMIT_PATTERN.fullmatch(sha):
        raise RuntimeError("Hugging Face did not return an immutable model commit")
    return sha.lower()


def build_lock(
    config: Mapping[str, Any],
    project_root: Path,
    source_scan: Mapping[str, Any],
    *,
    resolve_model: bool = True,
) -> dict[str, Any]:
    model = config["model"]
    revision = str(model["revision"])
    resolved_revision = (
        _resolve_huggingface_revision(str(model["id"]), revision) if resolve_model else revision
    )
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    sources = source_scan.get("sources", source_scan.get("results", []))
    return {
        "schema_version": 1,
        "project": config["project"]["name"],
        "config_sha256": hashlib.sha256(canonical).hexdigest(),
        "model": {
            "provider": model["provider"],
            "id": model["id"],
            "requested_revision": revision,
            "resolved_revision": resolved_revision,
            "loader": model["loader"],
            "trust_remote_code": model["trust_remote_code"],
        },
        "sources": sources,
        "environment": {
            "backend": config["environment"]["backend"],
            "image": config["environment"]["image"],
            "network": config["environment"]["network"],
        },
        "seed": config["project"]["seed"],
        "project_root": str(project_root.resolve()),
    }


def write_lock(destination: Path, lock: Mapping[str, Any]) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return destination
