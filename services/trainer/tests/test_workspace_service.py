from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE_ROOT / "src"))

from autotrainer.config import ConfigError, default_config, load_config, write_config  # noqa: E402
from autotrainer.models import MODEL_CATALOG  # noqa: E402
from autotrainer.workspace_service import ProjectWorkspace  # noqa: E402


class ProjectWorkspaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.projects_root = self.root / "projects"
        self.projects_root.mkdir()
        self.startup_config = self.root / "current" / "autotrainer.yaml"
        write_config(
            self.startup_config,
            default_config(name="Current project"),
            overwrite=False,
        )
        self.workspace = ProjectWorkspace(self.projects_root, self.startup_config)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_lists_startup_and_persisted_managed_projects(self) -> None:
        initial = self.workspace.list_projects()
        self.assertEqual(initial["active_id"], "startup")
        self.assertEqual([project["id"] for project in initial["projects"]], ["startup"])

        created = self.workspace.create_project("Frontend Specialist")
        reconstructed = ProjectWorkspace(self.projects_root, self.startup_config)
        records = reconstructed.list_projects()

        self.assertEqual(created["id"], "frontend-specialist")
        self.assertEqual(
            [project["id"] for project in records["projects"]],
            ["startup", "frontend-specialist"],
        )

    def test_creation_uses_pinned_default_and_one_shared_cache(self) -> None:
        created = self.workspace.create_project("Customer Support", activate=True)
        config = load_config(Path(created["config_path"]))
        profile = MODEL_CATALOG["qwen3.5-9b-text"]

        self.assertEqual(config.data["project"]["name"], "Customer Support")
        self.assertEqual(config.model["id"], profile["id"])
        self.assertEqual(config.model["revision"], profile["default_revision"])
        self.assertEqual(
            Path(str(config.model["cache_dir"])).resolve(),
            self.workspace.shared_model_cache,
        )
        self.assertEqual(self.workspace.active_id, "customer-support")
        self.assertEqual(self.workspace.active_config, config.path)

    def test_select_resolves_only_known_safe_records(self) -> None:
        created = self.workspace.create_project("Coding Agent")
        selected = self.workspace.select_project(created["id"])

        self.assertTrue(selected["active"])
        self.assertEqual(self.workspace.resolve_project("startup"), self.startup_config.resolve())
        with self.assertRaisesRegex(ConfigError, "invalid"):
            self.workspace.resolve_project("../outside")
        with self.assertRaisesRegex(ConfigError, "unknown"):
            self.workspace.select_project("missing")

    def test_creation_rejects_paths_reserved_names_and_overwrite(self) -> None:
        for value in ("../outside", r"folder\outside", "..", "startup", "CON"):
            with self.subTest(value=value), self.assertRaises(ConfigError):
                self.workspace.create_project(value)

        existing = self.projects_root / "existing"
        existing.mkdir()
        marker = existing / "keep.txt"
        marker.write_text("user-owned", encoding="utf-8")
        with self.assertRaisesRegex(ConfigError, "already exists"):
            self.workspace.create_project("Existing")
        self.assertEqual(marker.read_text(encoding="utf-8"), "user-owned")
        self.assertFalse((existing / "autotrainer.yaml").exists())

    def test_symbolic_project_directory_is_never_followed(self) -> None:
        outside = self.root / "outside"
        outside.mkdir()
        link = self.projects_root / "linked"
        try:
            link.symlink_to(outside, target_is_directory=True)
        except OSError as error:
            self.skipTest(f"symbolic links are unavailable: {error}")

        with self.assertRaisesRegex(ConfigError, "symbolic link"):
            self.workspace.resolve_project("linked")
        with self.assertRaisesRegex(ConfigError, "already exists"):
            self.workspace.create_project("Linked")
        self.assertNotIn(
            "linked",
            [project["id"] for project in self.workspace.list_projects()["projects"]],
        )


if __name__ == "__main__":
    unittest.main()
