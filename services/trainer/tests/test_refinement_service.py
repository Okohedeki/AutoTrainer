from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE_ROOT / "src"))

from autotrainer.config import ConfigError, default_config, load_config, write_config  # noqa: E402
from autotrainer.refinement_service import (  # noqa: E402
    get_refinement_settings,
    set_refinement_settings,
)


class RefinementServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.config_path = Path(self.temporary_directory.name) / "autotrainer.yaml"
        write_config(self.config_path, default_config(), overwrite=False)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_user_can_select_hard_or_soft_vram_budget(self) -> None:
        hard = set_refinement_settings(
            self.config_path,
            max_vram_gib=14.5,
            enforcement="hard",
        )
        self.assertEqual(hard["mode"], "adapter_only")
        self.assertEqual(hard["vram"], {"max_gib": 14.5, "enforcement": "hard"})

        soft = set_refinement_settings(
            self.config_path,
            max_vram_gib=12,
            enforcement="soft",
        )
        self.assertEqual(soft["vram"], {"max_gib": 12.0, "enforcement": "soft"})
        self.assertEqual(load_config(self.config_path).data["refinement"], soft)

    def test_invalid_budget_cannot_change_configuration(self) -> None:
        before = self.config_path.read_bytes()
        for limit, enforcement in ((3.5, "hard"), (24, "maybe")):
            with self.subTest(limit=limit, enforcement=enforcement), self.assertRaises(ConfigError):
                set_refinement_settings(
                    self.config_path,
                    max_vram_gib=limit,
                    enforcement=enforcement,
                )
        self.assertEqual(self.config_path.read_bytes(), before)

    def test_legacy_project_reads_safe_defaults(self) -> None:
        config = default_config()
        config.pop("refinement")
        write_config(self.config_path, config, overwrite=True)

        self.assertEqual(
            get_refinement_settings(self.config_path),
            {
                "mode": "adapter_only",
                "vram": {"max_gib": 20.0, "enforcement": "hard"},
            },
        )


if __name__ == "__main__":
    unittest.main()
