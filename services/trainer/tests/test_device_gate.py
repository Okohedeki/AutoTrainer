from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch


SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE_ROOT / "src"))

from autotrainer.device_gate import (  # noqa: E402
    DeviceBusyError,
    acquire_device_lease,
    clear_cuda_memory,
    device_run_gate,
)


class DeviceGateTests(unittest.TestCase):
    def test_only_one_project_can_own_gpu_zero(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch(
            "autotrainer.device_gate._LOCK_ROOT", Path(directory)
        ):
            first = acquire_device_lease()
            try:
                with self.assertRaisesRegex(DeviceBusyError, "GPU 0 is already in use"):
                    acquire_device_lease()
            finally:
                first.release()

            second = acquire_device_lease()
            second.release()

    def test_transferred_owner_can_enter_nested_runtime_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch(
            "autotrainer.device_gate._LOCK_ROOT", Path(directory)
        ):
            lease = acquire_device_lease()
            try:
                with lease.activate(), device_run_gate():
                    pass
            finally:
                lease.release()

    def test_runtime_gate_clears_cuda_before_releasing_after_failure(self) -> None:
        events: list[str] = []

        class FakeLease:
            @contextmanager
            def activate(self):
                events.append("activate")
                try:
                    yield
                finally:
                    events.append("deactivate")

            def release(self) -> None:
                events.append("release")

        with (
            patch("autotrainer.device_gate.acquire_device_lease", return_value=FakeLease()),
            patch(
                "autotrainer.device_gate.clear_cuda_memory",
                side_effect=lambda: events.append("cleanup"),
            ),
            self.assertRaisesRegex(RuntimeError, "training failed"),
        ):
            with device_run_gate():
                events.append("run")
                raise RuntimeError("training failed")

        self.assertEqual(
            events,
            ["activate", "run", "deactivate", "cleanup", "release"],
        )

    def test_cuda_cleanup_is_best_effort_for_available_and_broken_torch(self) -> None:
        cuda = MagicMock()
        cuda.is_available.return_value = True
        with (
            patch("autotrainer.device_gate.gc.collect") as collect,
            patch.dict(sys.modules, {"torch": SimpleNamespace(cuda=cuda)}),
        ):
            clear_cuda_memory()
        collect.assert_called_once_with()
        cuda.empty_cache.assert_called_once_with()

        broken_cuda = MagicMock()
        broken_cuda.is_available.side_effect = RuntimeError("broken driver")
        with (
            patch("autotrainer.device_gate.gc.collect", side_effect=RuntimeError("broken gc")),
            patch.dict(sys.modules, {"torch": SimpleNamespace(cuda=broken_cuda)}),
        ):
            clear_cuda_memory()
        broken_cuda.empty_cache.assert_not_called()

        with (
            patch("autotrainer.device_gate.gc.collect"),
            patch.dict(sys.modules, {"torch": None}),
        ):
            clear_cuda_memory()


if __name__ == "__main__":
    unittest.main()
