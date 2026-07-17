from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE_ROOT / "src"))

from autotrainer.device_gate import (  # noqa: E402
    DeviceBusyError,
    acquire_device_lease,
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


if __name__ == "__main__":
    unittest.main()
