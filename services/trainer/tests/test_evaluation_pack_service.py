from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE_ROOT / "src"))

from autotrainer.config import ConfigError, default_config, load_config, write_config  # noqa: E402
from autotrainer.compiler import compile_data  # noqa: E402
from autotrainer.evaluation_pack_service import (  # noqa: E402
    install_evaluation_pack,
    list_evaluation_packs,
)
from autotrainer.manifest import TaskManifest  # noqa: E402
from autotrainer.sources import scan_sources  # noqa: E402


PACK_ID = "python-core-v1"


class EvaluationPackServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.config_path = self.root / "autotrainer.yaml"
        write_config(self.config_path, default_config(), overwrite=False)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_lists_the_one_shipped_pack_before_installation(self) -> None:
        result = list_evaluation_packs(self.config_path)

        self.assertEqual(result["selected_pack"], None)
        self.assertEqual(len(result["packs"]), 1)
        pack = result["packs"][0]
        self.assertEqual(
            set(pack),
            {
                "id",
                "label",
                "language",
                "license",
                "task_count",
                "independent_group_count",
                "description",
                "status",
                "selected",
                "installed",
                "runtime_image",
                "checks",
            },
        )
        self.assertEqual(pack["id"], PACK_ID)
        self.assertEqual(pack["language"], "python")
        self.assertEqual(pack["license"], "Apache-2.0")
        self.assertEqual(pack["task_count"], 5)
        self.assertEqual(pack["independent_group_count"], 5)
        self.assertEqual(pack["status"], "available")
        self.assertFalse(pack["installed"])
        self.assertFalse(pack["selected"])
        self.assertEqual(pack["runtime_image"], "autotrainer/frontend-runtime:0.1")

    def test_installs_five_independent_repositories_and_hidden_verifiers(self) -> None:
        existing = self.root / "existing.jsonl"
        existing.write_text(
            '{"prompt":"Keep me","completion":"Preserved"}\n', encoding="utf-8"
        )
        payload = load_config(self.config_path).data
        payload["sources"].append(
            {
                "id": "existing-examples",
                "kind": "sft_jsonl",
                "uri": "existing.jsonl",
                "partition": "train",
                "roles": ["demonstrations"],
            }
        )
        write_config(self.config_path, payload, overwrite=True)
        freeze = self.root / ".autotrainer" / "dataset" / "freeze.json"
        plan = self.root / ".autotrainer" / "evaluation" / "current-plan.json"
        training_evidence = self.root / ".autotrainer" / "training" / "runs" / "immutable"
        evaluation_evidence = self.root / ".autotrainer" / "evaluation" / "runs" / "immutable"
        for path in (freeze, plan, training_evidence / "evidence.json", evaluation_evidence / "evidence.json"):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}\n", encoding="utf-8")

        result = install_evaluation_pack(self.config_path, PACK_ID)

        self.assertEqual(result["selected_pack"], PACK_ID)
        pack = result["packs"][0]
        self.assertTrue(pack["installed"])
        self.assertTrue(pack["selected"])
        self.assertEqual(pack["status"], "installed")
        self.assertFalse(freeze.exists())
        self.assertFalse(plan.exists())
        self.assertTrue((training_evidence / "evidence.json").is_file())
        self.assertTrue((evaluation_evidence / "evidence.json").is_file())

        config = load_config(self.config_path)
        self.assertEqual(config.data["evaluation"]["language"], "python")
        self.assertEqual(config.data["evaluation"]["task_pack"], PACK_ID)
        self.assertEqual(
            config.data["environment"]["image"], "autotrainer/frontend-runtime:0.1"
        )
        self.assertIn("existing-examples", {source["id"] for source in config.sources})
        repositories = [
            source
            for source in config.sources
            if source.get("kind") == "repository" and source.get("id", "").startswith(PACK_ID)
        ]
        task_packs = [source for source in config.sources if source.get("id") == PACK_ID]
        self.assertEqual(len(repositories), 5)
        self.assertEqual(len(task_packs), 1)
        self.assertEqual(len({source["revision"] for source in repositories}), 5)

        pack_root = self.root / ".autotrainer" / "evaluation-packs" / PACK_ID
        self.assertTrue((pack_root / "LICENSE").is_file())
        self.assertTrue((pack_root / "NOTICE").is_file())
        manifests = sorted((pack_root / "tasks").glob("*/task.json"))
        self.assertEqual(len(manifests), 5)
        groups: set[str] = set()
        for manifest_path in manifests:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest = TaskManifest.from_mapping(payload)
            groups.add(manifest.group_id)
            repository = next(
                source for source in repositories if source["id"] == manifest.source_id
            )
            repository_path = config.resolve_path(repository["uri"])
            bundle = Path(manifest.verifier_bundle or "")
            self.assertFalse(bundle.is_relative_to(repository_path))
            self.assertTrue(bundle.is_dir())
            self.assertEqual(manifest.starting_revision, repository["revision"])
            self.assertEqual(manifest.runtime_commands["build"], "python3 -m compileall -q .")
            self.assertEqual(
                manifest.runtime_commands["tests"],
                "python3 -m unittest discover -s tests -v",
            )
        self.assertEqual(len(groups), 5)

        scan = scan_sources(config.data, self.root)
        self.assertEqual(scan["errors"], [])
        self.assertEqual(scan["summary"]["evaluation_ready_task_count"], 5)
        compiled = compile_data(config.data, self.root, scan)
        self.assertEqual(compiled["errors"], [])
        self.assertEqual(compiled["counts"]["rl_evaluation"], 5)

    def test_every_baseline_passes_public_checks_and_fails_its_hidden_task(self) -> None:
        install_evaluation_pack(self.config_path, PACK_ID)
        config = load_config(self.config_path)
        pack_root = self.root / ".autotrainer" / "evaluation-packs" / PACK_ID

        for manifest_path in sorted((pack_root / "tasks").glob("*/task.json")):
            manifest = TaskManifest.from_mapping(
                json.loads(manifest_path.read_text(encoding="utf-8"))
            )
            source = next(item for item in config.sources if item["id"] == manifest.source_id)
            repository = config.resolve_path(source["uri"])
            public = subprocess.run(
                [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
                cwd=repository,
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(public.returncode, 0, public.stderr)

            report = repository / ".autotrainer-verifier-report.json"
            environment = {
                **os.environ,
                "AUTOTRAINER_WORKSPACE": str(repository),
                "AUTOTRAINER_REPORT_PATH": str(report),
            }
            verifier = subprocess.run(
                [sys.executable, str(Path(manifest.verifier_bundle or "") / "verify.py")],
                cwd=repository,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(verifier.returncode, 0, verifier.stderr)
            scores = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(
                set(scores),
                {
                    "build_passed",
                    "regression_pass_rate",
                    "task_pass_rate",
                    "responsive_pass_rate",
                    "design_rule_pass_rate",
                    "code_quality_pass_rate",
                },
            )
            self.assertTrue(scores["build_passed"])
            self.assertEqual(scores["regression_pass_rate"], 1.0)
            self.assertEqual(scores["task_pass_rate"], 0.0)

    def test_install_is_idempotent_and_rejects_unknown_or_conflicting_ids(self) -> None:
        first = install_evaluation_pack(self.config_path, PACK_ID)
        first_config = load_config(self.config_path)
        first_sources = first_config.sources
        first_revisions = {
            source["id"]: source.get("revision")
            for source in first_sources
            if source.get("kind") == "repository"
        }

        second = install_evaluation_pack(self.config_path, PACK_ID)

        second_config = load_config(self.config_path)
        self.assertEqual(first, second)
        self.assertEqual(first_sources, second_config.sources)
        self.assertEqual(
            first_revisions,
            {
                source["id"]: source.get("revision")
                for source in second_config.sources
                if source.get("kind") == "repository"
            },
        )
        with self.assertRaisesRegex(ConfigError, "unknown evaluation pack"):
            install_evaluation_pack(self.config_path, "not-shipped")

        other_root = self.root / "conflict"
        other_root.mkdir()
        other_config = other_root / "autotrainer.yaml"
        payload = default_config()
        payload["sources"] = [
            {
                "id": "python-core-v1-ready-order",
                "kind": "repository",
                "uri": ".",
                "revision": "0" * 40,
                "partition": "evaluation",
                "roles": ["evaluation"],
            }
        ]
        write_config(other_config, payload, overwrite=False)
        with self.assertRaisesRegex(ConfigError, "already used outside"):
            install_evaluation_pack(other_config, PACK_ID)


if __name__ == "__main__":
    unittest.main()
