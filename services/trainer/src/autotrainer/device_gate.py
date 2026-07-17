"""Cross-project ownership for AutoTrainer's single local CUDA device.

Project leases keep one project's configuration stable, but they do not stop
two different projects from loading separate 9B models onto GPU 0.  This small
OS-backed lease is shared by training, evaluation, and the callable model host
so every entry point obeys the same one-GPU V1 contract.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import tempfile
import threading
from typing import BinaryIO, Iterator

from .project_gate import ProjectBusyError


DEFAULT_DEVICE = "cuda:0"
_LOCAL_GUARD = threading.Lock()
_LOCAL_LEASES: set[str] = set()
_CURRENT_DEVICE: ContextVar[str | None] = ContextVar(
    "autotrainer_gpu_owner", default=None
)
_USER_SCOPE = hashlib.sha256(
    str(Path.home().expanduser().resolve()).encode("utf-8")
).hexdigest()[:16]
_LOCK_ROOT = Path(tempfile.gettempdir()) / "autotrainer-gpu-leases" / _USER_SCOPE


class DeviceBusyError(ProjectBusyError):
    """Raised instead of oversubscribing the single V1 CUDA device."""


def _device_name(value: str) -> str:
    device = str(value).strip().lower()
    if device != DEFAULT_DEVICE:
        raise ValueError(f"AutoTrainer V1 supports only {DEFAULT_DEVICE}")
    return device


def _lock_path(device: str) -> Path:
    return _LOCK_ROOT / f"{device.replace(':', '-')}.lock"


def _lock_file(handle: BinaryIO) -> None:
    """Acquire byte zero immediately on Windows or POSIX."""

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
class DeviceLease:
    """A transferable device lease held until model memory is released."""

    device: str
    _handle: BinaryIO
    _released: bool = False

    @contextmanager
    def activate(self) -> Iterator[None]:
        if self._released:
            raise RuntimeError("cannot activate a released GPU lease")
        token = _CURRENT_DEVICE.set(self.device)
        try:
            yield
        finally:
            _CURRENT_DEVICE.reset(token)

    def release(self) -> None:
        if self._released:
            return
        try:
            _unlock_file(self._handle)
        finally:
            self._handle.close()
            with _LOCAL_GUARD:
                _LOCAL_LEASES.discard(self.device)
            self._released = True


def acquire_device_lease(device: str = DEFAULT_DEVICE) -> DeviceLease:
    """Reserve the one supported GPU immediately or return a useful conflict."""

    normalized = _device_name(device)
    with _LOCAL_GUARD:
        if normalized in _LOCAL_LEASES:
            raise DeviceBusyError(
                "GPU 0 is already in use by another AutoTrainer training, "
                "evaluation, or model-host process. Stop it before starting this job."
            )
        _LOCAL_LEASES.add(normalized)

    path = _lock_path(normalized)
    handle: BinaryIO | None = None
    try:
        if _LOCK_ROOT.exists() and _LOCK_ROOT.is_symlink():
            raise OSError("GPU lease directory cannot be a symbolic link")
        _LOCK_ROOT.mkdir(parents=True, exist_ok=True)
        if _LOCK_ROOT.is_symlink() or path.is_symlink():
            raise OSError("GPU lease path cannot be a symbolic link")
        handle = path.open("a+b")
        _lock_file(handle)
    except (OSError, BlockingIOError) as error:
        if handle is not None:
            handle.close()
        with _LOCAL_GUARD:
            _LOCAL_LEASES.discard(normalized)
        raise DeviceBusyError(
            "GPU 0 is already in use by another AutoTrainer training, "
            "evaluation, or model-host process. Stop it before starting this job."
        ) from error
    except Exception:
        if handle is not None:
            handle.close()
        with _LOCAL_GUARD:
            _LOCAL_LEASES.discard(normalized)
        raise
    return DeviceLease(device=normalized, _handle=handle)


@contextmanager
def device_run_gate(device: str = DEFAULT_DEVICE) -> Iterator[None]:
    """Own the device for a synchronous run or reuse a transferred lease."""

    normalized = _device_name(device)
    if _CURRENT_DEVICE.get() == normalized:
        yield
        return
    lease = acquire_device_lease(normalized)
    try:
        with lease.activate():
            yield
    finally:
        lease.release()


__all__ = [
    "DEFAULT_DEVICE",
    "DeviceBusyError",
    "DeviceLease",
    "acquire_device_lease",
    "device_run_gate",
]
