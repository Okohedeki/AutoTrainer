"""Cross-process ownership for one project's mutable training lifecycle.

Training consumes compiled artifacts and a configuration snapshot for hours.
The small non-blocking lease in this module keeps Prepare and setup mutations
from changing either input between stage selection and execution.  The OS owns
the byte-range lock, so an interrupted Python process releases it automatically.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import os
from pathlib import Path
import threading
from typing import BinaryIO, Iterator, Literal

from .config import ConfigError


class ProjectBusyError(ConfigError):
    """Raised instead of waiting behind a long project operation."""


_BUSY_MESSAGE = (
    "This project is busy with an active training or preparation operation. "
    "Wait for it to finish before changing or preparing the project."
)
_LOCAL_GUARD = threading.Lock()
_LOCAL_LEASES: set[Path] = set()
_OwnerPurpose = Literal["run", "prepare"]
_CURRENT_OWNER: ContextVar[tuple[Path, _OwnerPurpose] | None] = ContextVar(
    "autotrainer_project_owner", default=None
)


def _project_key(config_path: str | Path) -> Path:
    return Path(config_path).expanduser().resolve()


def _lock_path(config_path: Path) -> Path:
    # Keep the stable coordination file in AutoTrainer's already-ignored local
    # workspace. It must not follow configurable artifact_dir while locked.
    return config_path.parent / ".autotrainer" / "training" / "project-run.lock"


def _lock_file(handle: BinaryIO) -> None:
    """Acquire byte zero without blocking on Windows or POSIX."""

    handle.seek(0, os.SEEK_END)
    if handle.tell() == 0:
        handle.write(b"\0")
        handle.flush()
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock_file(handle: BinaryIO) -> None:
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@dataclass(slots=True)
class ProjectLease:
    """A transferable lease held from API queueing through worker completion."""

    config_path: Path
    _handle: BinaryIO
    _released: bool = False

    @contextmanager
    def activate(self, purpose: _OwnerPurpose) -> Iterator[None]:
        """Mark this execution context as the owner of the held OS lease."""

        if self._released:
            raise RuntimeError("cannot activate a released project lease")
        token = _CURRENT_OWNER.set((self.config_path, purpose))
        try:
            yield
        finally:
            _CURRENT_OWNER.reset(token)

    def release(self) -> None:
        """Release exactly once, including after worker or process failures."""

        if self._released:
            return
        try:
            _unlock_file(self._handle)
        finally:
            self._handle.close()
            with _LOCAL_GUARD:
                _LOCAL_LEASES.discard(self.config_path)
            self._released = True


def acquire_project_lease(config_path: str | Path) -> ProjectLease:
    """Acquire immediately or report that the project is already in use."""

    key = _project_key(config_path)
    with _LOCAL_GUARD:
        if key in _LOCAL_LEASES:
            raise ProjectBusyError(_BUSY_MESSAGE)
        _LOCAL_LEASES.add(key)

    path = _lock_path(key)
    handle: BinaryIO | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = path.open("a+b")
        _lock_file(handle)
    except (OSError, BlockingIOError) as error:
        if handle is not None:
            handle.close()
        with _LOCAL_GUARD:
            _LOCAL_LEASES.discard(key)
        raise ProjectBusyError(_BUSY_MESSAGE) from error
    except Exception:
        if handle is not None:
            handle.close()
        with _LOCAL_GUARD:
            _LOCAL_LEASES.discard(key)
        raise
    return ProjectLease(config_path=key, _handle=handle)


@contextmanager
def project_run_gate(config_path: str | Path) -> Iterator[None]:
    """Own one synchronous run, or reuse the GUI worker's reserved lease."""

    key = _project_key(config_path)
    if _CURRENT_OWNER.get() == (key, "run"):
        yield
        return
    lease = acquire_project_lease(key)
    try:
        with lease.activate("run"):
            yield
    finally:
        lease.release()


@contextmanager
def project_prepare_gate(config_path: str | Path) -> Iterator[None]:
    """Protect external Prepare while allowing the active run's own Prepare."""

    key = _project_key(config_path)
    if _CURRENT_OWNER.get() == (key, "run"):
        yield
        return
    lease = acquire_project_lease(key)
    try:
        with lease.activate("prepare"):
            yield
    finally:
        lease.release()


@contextmanager
def project_mutation_gate(config_path: str | Path) -> Iterator[None]:
    """Protect a setup mutation; run ownership never bypasses this guard."""

    lease = acquire_project_lease(config_path)
    try:
        yield
    finally:
        lease.release()


def assert_project_available(config_path: str | Path) -> None:
    """Fail fast before starting long work, without retaining the lease."""

    lease = acquire_project_lease(config_path)
    lease.release()


def project_is_busy(config_path: str | Path) -> bool:
    """Probe a saved job owner without waiting or changing its lock file."""

    try:
        lease = acquire_project_lease(config_path)
    except ProjectBusyError:
        return True
    lease.release()
    return False


__all__ = [
    "ProjectBusyError",
    "ProjectLease",
    "acquire_project_lease",
    "assert_project_available",
    "project_is_busy",
    "project_mutation_gate",
    "project_prepare_gate",
    "project_run_gate",
]
