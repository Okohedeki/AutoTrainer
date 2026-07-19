"""Content identities shared by evaluation planning and trusted runtimes.

These helpers deliberately avoid Git metadata and mutable container tags.  An
evaluation plan must name the bytes that will execute, not merely the friendly
path or tag that happened to point at them when the plan was created.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
from typing import Any, Iterable


class IntegrityError(ValueError):
    """Raised when content cannot be given a safe, immutable identity."""


def canonical_json(value: Any) -> bytes:
    """Return the single JSON representation used by all integrity hashes."""

    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def digest_json(value: Any) -> str:
    """Hash a JSON-compatible value using :func:`canonical_json`."""

    return hashlib.sha256(canonical_json(value)).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash one regular file without following a link or reparse point."""

    candidate = Path(path)
    try:
        metadata = candidate.lstat()
    except OSError as error:
        raise IntegrityError(f"identity file is unavailable: {candidate}: {error}") from error
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    attributes = getattr(metadata, "st_file_attributes", 0)
    if stat.S_ISLNK(metadata.st_mode) or bool(attributes & reparse_flag):
        raise IntegrityError(f"identity files must not be links: {candidate}")
    if not stat.S_ISREG(metadata.st_mode):
        raise IntegrityError(f"identity path must be a regular file: {candidate}")
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_identity(path: Path) -> dict[str, Any]:
    """Describe a directory tree by every relative path, size, and file hash.

    The entry list is retained in the plan so an audit can identify which file
    changed; the aggregate digest is the compact identity used by run records.
    """

    root = Path(path)
    try:
        metadata = root.lstat()
    except OSError as error:
        raise IntegrityError(f"identity directory is unavailable: {root}: {error}") from error
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    attributes = getattr(metadata, "st_file_attributes", 0)
    if stat.S_ISLNK(metadata.st_mode) or bool(attributes & reparse_flag):
        raise IntegrityError(f"identity directories must not be links: {root}")
    if not stat.S_ISDIR(metadata.st_mode):
        raise IntegrityError(f"identity path must be a directory: {root}")

    entries: list[dict[str, Any]] = []
    for candidate in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        candidate_metadata = candidate.lstat()
        attributes = getattr(candidate_metadata, "st_file_attributes", 0)
        if stat.S_ISLNK(candidate_metadata.st_mode) or bool(attributes & reparse_flag):
            raise IntegrityError(f"identity trees must not contain links: {candidate}")
        if stat.S_ISDIR(candidate_metadata.st_mode):
            continue
        if not stat.S_ISREG(candidate_metadata.st_mode):
            raise IntegrityError(f"identity trees contain a non-regular file: {candidate}")
        entries.append(
            {
                "path": candidate.relative_to(root).as_posix(),
                "bytes": int(candidate_metadata.st_size),
                "sha256": sha256_file(candidate),
            }
        )
    if not entries:
        raise IntegrityError(f"identity directory contains no files: {root}")
    return {"sha256": f"sha256:{digest_json(entries)}", "files": entries}


def source_identity(paths: Iterable[tuple[str, Path]]) -> dict[str, Any]:
    """Freeze a named set of trusted Python implementation files."""

    entries = [
        {"path": name, "sha256": sha256_file(Path(path))}
        for name, path in sorted(paths, key=lambda item: item[0])
    ]
    if not entries:
        raise IntegrityError("trusted implementation identity contains no files")
    return {"sha256": f"sha256:{digest_json(entries)}", "files": entries}


_SHA256_REFERENCE = re.compile(r"^sha256:([0-9a-fA-F]{64})$")
_DIGEST_REFERENCE = re.compile(r"^.+@sha256:([0-9a-fA-F]{64})$")


def resolve_container_image(backend: str, reference: str) -> dict[str, str]:
    """Resolve a mutable local image reference to an immutable runtime value.

    A bare image ID and a repository digest are already immutable. Tags require
    a local ``image inspect``; this never pulls and therefore cannot silently
    change the machine while an evaluation plan is being frozen.
    """

    backend_value = str(backend).strip()
    reference_value = str(reference).strip()
    if backend_value not in {"docker", "podman"}:
        raise IntegrityError("container backend must be docker or podman")
    if not reference_value:
        raise IntegrityError("container image reference is required")

    image_id_match = _SHA256_REFERENCE.fullmatch(reference_value)
    digest_match = _DIGEST_REFERENCE.fullmatch(reference_value)
    if image_id_match:
        digest = image_id_match.group(1).lower()
        return {
            "backend": backend_value,
            "reference": reference_value,
            "digest": f"sha256:{digest}",
            "runtime_reference": f"sha256:{digest}",
            "resolution": "image_id",
        }
    if digest_match:
        digest = digest_match.group(1).lower()
        return {
            "backend": backend_value,
            "reference": reference_value,
            "digest": f"sha256:{digest}",
            "runtime_reference": reference_value,
            "resolution": "repository_digest",
        }

    try:
        completed = subprocess.run(
            [backend_value, "image", "inspect", reference_value, "--format", "{{.Id}}"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
            shell=False,
            env=dict(os.environ),
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise IntegrityError(
            f"could not resolve container image {reference_value!r} with {backend_value}: {error}"
        ) from error
    image_id = completed.stdout.strip().splitlines()[0] if completed.stdout.strip() else ""
    match = _SHA256_REFERENCE.fullmatch(image_id)
    if completed.returncode != 0 or match is None:
        detail = completed.stderr.strip() or completed.stdout.strip() or "image is unavailable"
        raise IntegrityError(
            f"could not freeze container image {reference_value!r}; build it first: {detail}"
        )
    digest = match.group(1).lower()
    return {
        "backend": backend_value,
        "reference": reference_value,
        "digest": f"sha256:{digest}",
        # Docker and Podman both accept a local image ID directly. Running this
        # value cannot follow a tag that was retargeted after plan creation.
        "runtime_reference": f"sha256:{digest}",
        "resolution": "local_inspect",
    }


__all__ = [
    "IntegrityError",
    "canonical_json",
    "digest_json",
    "resolve_container_image",
    "sha256_file",
    "source_identity",
    "tree_identity",
]
