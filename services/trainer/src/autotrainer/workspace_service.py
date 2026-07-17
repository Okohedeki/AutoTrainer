"""Safe local project discovery and creation for the human workspace.

The CLI can always address an arbitrary project with ``--config``.  The GUI,
however, needs a bounded directory it can list without accepting filesystem
paths from browser requests.  This service keeps that boundary small: one
startup project plus projects created directly below a server-owned root.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import re
from threading import RLock
import unicodedata
from typing import Any

from .config import ConfigError, default_config, load_config, write_config
from .models import MODEL_CATALOG


_CONFIG_NAME = "autotrainer.yaml"
_STARTUP_ID = "startup"
_PROJECT_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
_WINDOWS_RESERVED_NAMES = {
    "con",
    "prn",
    "aux",
    "nul",
    *(f"com{index}" for index in range(1, 10)),
    *(f"lpt{index}" for index in range(1, 10)),
}


def _display_name(value: object) -> str:
    """Validate a human name before it can influence a local directory."""

    if not isinstance(value, str):
        raise ConfigError("project name must be text")
    name = value.strip()
    if not name:
        raise ConfigError("project name is required")
    if len(name) > 80:
        raise ConfigError("project name must be 80 characters or fewer")
    if any(ord(character) < 32 for character in name):
        raise ConfigError("project name cannot contain control characters")
    # Names are labels, not paths. Reject separators and traversal explicitly
    # instead of silently turning a dangerous-looking value into a safe slug.
    if "/" in name or "\\" in name or ".." in name:
        raise ConfigError("project name cannot contain a path or traversal")
    return name


def _slug(name: str) -> str:
    """Create a portable ASCII identifier used as exactly one child folder."""

    normalized = unicodedata.normalize("NFKD", name)
    ascii_name = normalized.encode("ascii", "ignore").decode("ascii").lower()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_name).strip("-")[:64].rstrip("-")
    if not slug or not _PROJECT_ID.fullmatch(slug):
        raise ConfigError("project name must contain at least one letter or number")
    if slug == _STARTUP_ID or slug in _WINDOWS_RESERVED_NAMES:
        raise ConfigError("project name resolves to a reserved local identifier")
    return slug


class ProjectWorkspace:
    """List, create, resolve, and select projects inside one trusted root."""

    def __init__(self, projects_root: str | Path, startup_config: str | Path) -> None:
        root_input = Path(projects_root).expanduser()
        if root_input.exists() and root_input.is_symlink():
            raise ConfigError("projects root cannot be a symbolic link")
        self._root = root_input.resolve()
        if self._root.exists() and not self._root.is_dir():
            raise ConfigError(f"projects root is not a directory: {self._root}")

        self._startup_config = Path(startup_config).expanduser().resolve()
        # Validate the initial project before a server can advertise the
        # workspace. Path checks remain Prepare's responsibility.
        load_config(self._startup_config)
        self._active_id = _STARTUP_ID
        self._lock = RLock()

    @property
    def projects_root(self) -> Path:
        return self._root

    @property
    def active_id(self) -> str:
        with self._lock:
            return self._active_id

    @property
    def active_config(self) -> Path:
        with self._lock:
            return self.resolve_project(self._active_id)

    @property
    def shared_model_cache(self) -> Path:
        """Return the cache new projects share without creating it eagerly."""

        return (self._root / ".autotrainer" / "model-cache").resolve()

    def _managed_config(self, project_id: str) -> Path:
        if not _PROJECT_ID.fullmatch(project_id) or project_id == _STARTUP_ID:
            raise ConfigError("project id is invalid")
        directory = self._root / project_id
        if directory.is_symlink():
            raise ConfigError("project directory cannot be a symbolic link")
        config_path = directory / _CONFIG_NAME
        if config_path.is_symlink():
            raise ConfigError("project configuration cannot be a symbolic link")
        # The identifier is already one safe segment; the containment check is
        # retained as a defense if that representation changes later.
        resolved = config_path.resolve()
        try:
            resolved.relative_to(self._root)
        except ValueError as error:
            raise ConfigError("project configuration escapes the projects root") from error
        return resolved

    def resolve_project(self, project_id: str) -> Path:
        """Resolve only the startup project or a managed direct child."""

        if not isinstance(project_id, str):
            raise ConfigError("project id must be text")
        if project_id == _STARTUP_ID:
            path = self._startup_config
        else:
            path = self._managed_config(project_id)
        if not path.is_file():
            raise ConfigError(f"unknown project: {project_id}")
        load_config(path)
        return path

    def _record(self, project_id: str, config_path: Path) -> dict[str, Any]:
        config = load_config(config_path)
        project = config.data.get("project", {})
        model = config.data.get("model", {})
        return {
            "id": project_id,
            "name": str(project.get("name", project_id)),
            "config_path": str(config.path),
            "active": project_id == self._active_id,
            "managed": project_id != _STARTUP_ID,
            "model": {
                "id": str(model.get("id", "")),
                "revision": str(model.get("revision", "")),
            },
        }

    def list_projects(self) -> dict[str, Any]:
        """Return the startup project and valid managed child projects."""

        with self._lock:
            records = [self._record(_STARTUP_ID, self._startup_config)]
            if self._root.is_dir():
                for directory in sorted(self._root.iterdir(), key=lambda path: path.name.lower()):
                    if directory.is_symlink() or not directory.is_dir():
                        continue
                    if not _PROJECT_ID.fullmatch(directory.name) or directory.name == _STARTUP_ID:
                        continue
                    try:
                        config_path = self._managed_config(directory.name)
                        if config_path == self._startup_config or not config_path.is_file():
                            continue
                        records.append(self._record(directory.name, config_path))
                    except ConfigError:
                        # An edited or linked project is not selectable. The GUI
                        # must not receive an unsafe path merely so it can display
                        # a broken entry.
                        continue
            return {"active_id": self._active_id, "projects": deepcopy(records)}

    def create_project(self, name: str, *, activate: bool = False) -> dict[str, Any]:
        """Create one pinned project without following or replacing paths."""

        display_name = _display_name(name)
        project_id = _slug(display_name)
        with self._lock:
            if self._root.exists() and self._root.is_symlink():
                raise ConfigError("projects root cannot be a symbolic link")
            self._root.mkdir(parents=True, exist_ok=True)
            if self._root.is_symlink():
                raise ConfigError("projects root cannot be a symbolic link")
            directory = self._root / project_id
            if directory.exists() or directory.is_symlink():
                raise ConfigError(f"project already exists: {project_id}")
            try:
                directory.mkdir(exist_ok=False)
            except FileExistsError as error:
                # A second local process may win after the existence check.
                raise ConfigError(f"project already exists: {project_id}") from error

            profile = MODEL_CATALOG["qwen3.5-9b-text"]
            payload = default_config(
                name=display_name,
                model_id=str(profile["id"]),
                revision=str(profile["default_revision"]),
            )
            payload["model"]["cache_dir"] = str(self.shared_model_cache)
            config_path = directory / _CONFIG_NAME
            try:
                write_config(config_path, payload, overwrite=False)
                # Load the completed file before exposing or activating it.
                load_config(config_path)
            except Exception:
                # Only the directory created above is removed, and only while
                # empty. Existing user files are never part of this rollback.
                if config_path.exists():
                    config_path.unlink()
                directory.rmdir()
                raise
            if activate:
                self._active_id = project_id
            return self._record(project_id, config_path)

    def discard_created_project(self, project_id: str, config_path: str | Path) -> None:
        """Hide only the exact, inactive project created by a failed API call."""

        with self._lock:
            if project_id == _STARTUP_ID or self._active_id == project_id:
                raise ConfigError("cannot discard the startup or active project")
            expected = self._managed_config(project_id)
            supplied = Path(config_path).expanduser().resolve()
            if supplied != expected:
                raise ConfigError("created project path does not match its identifier")

            # Removing the one configuration is enough to make the failed
            # project undiscoverable. rmdir is deliberately non-recursive: a
            # lease scaffold or any unexpected user file is never destroyed.
            if expected.is_file():
                expected.unlink()
            try:
                expected.parent.rmdir()
            except OSError:
                pass

    def select_project(self, project_id: str) -> dict[str, Any]:
        """Select a validated record; long-job policy remains the API's job."""

        with self._lock:
            config_path = self.resolve_project(project_id)
            self._active_id = project_id
            return self._record(project_id, config_path)


__all__ = ["ProjectWorkspace"]
