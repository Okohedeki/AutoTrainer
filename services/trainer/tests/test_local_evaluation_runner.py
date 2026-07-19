from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from typing import Any
from unittest.mock import patch


SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE_ROOT / "src"))

from autotrainer.local_evaluation_runner import (  # noqa: E402
    ArmRuntime,
    BuiltinEvaluationProducer,
    CONTEXT_TOKENS,
    DECLARED_RUNTIME_DEPENDENCIES,
    MAX_INPUT_TOKENS,
    MAX_NEW_TOKENS,
    SOURCE_PROTOCOL_IDENTITY,
    LocalEvaluationRunnerError,
    _locked_source_context,
    _TOOL_SCHEMAS,
    _sha256_tree,
    _source_protocol_identity,
    builtin_runner_identity,
    resolve_arm_runtime,
)


REVISION = "a" * 40


def _request(arm_id: str = "candidate") -> dict:
    return {
        "schema_version": 1,
        "plan_id": "sha256:" + "c" * 64,
        "trial_id": f"trial-{arm_id}",
        "suite_id": "model_benchmark",
        "arm_id": arm_id,
        "task_id": "held-out-task",
        "repetition": 0,
        "seed": 1701,
        "runner": {"type": "builtin", **builtin_runner_identity()},
        "task": {
            "source_path": "unused-by-injected-context",
            "source_revision": "d" * 40,
            "manifest": {
                "task": {
                    "id": "held-out-task",
                    "instruction": "Repair the responsive card without changing its copy.",
                },
                "runtime": {
                    "workingDirectory": ".",
                    "build": "npm run build",
                    "tests": "npm test",
                    "browserTests": "npm run test:browser",
                },
                # Even a malformed direct caller cannot add trusted internals to
                # the producer prompt; prompt construction selects public keys.
                "verifier": {"command": "hidden-secret-verifier"},
                "rewards": {"hidden_weight": 1.0},
            },
        },
    }


class _FakeGenerator:
    def __init__(self) -> None:
        self.messages: list[list[dict[str, Any]]] = []
        self.max_token_budgets: list[int] = []
        self.calls = 0

    def count_tokens(self, messages: list[dict[str, str]]) -> int:
        # Deterministic stand-in for a real tokenizer; two characters per token
        # makes an oversized injected context exercise binary-search fitting.
        return 32 + sum(len(item["content"]) for item in messages) // 2

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        seed: int | None,
    ) -> tuple[str, int, int]:
        input_tokens = self.count_tokens(messages)
        if input_tokens > MAX_INPUT_TOKENS:
            raise AssertionError("producer passed an over-context prompt to generate")
        self.messages.append(messages)
        self.calls += 1
        return (
            "diff --git a/src/card.css b/src/card.css\n"
            "--- a/src/card.css\n"
            "+++ b/src/card.css\n"
            "@@ -1 +1 @@\n"
            "-.card{display:block}\n"
            "+.card{display:grid}\n",
            input_tokens,
            64,
        )

    def count_tokens_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> int:
        del tools
        return 64 + len(json.dumps(messages, sort_keys=True)) // 8

    def generate_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        seed: int | None,
    ) -> tuple[str, int, int]:
        del temperature, top_p, seed
        input_tokens = self.count_tokens_with_tools(messages, tools)
        self.messages.append(messages)
        self.max_token_budgets.append(max_tokens)
        call = self.calls
        self.calls += 1
        if call % 2:
            return "DONE", input_tokens, 1
        return (
            "<tool_call>\n"
            "<function=apply_patch>\n"
            "<parameter=patch>\n"
            "diff --git a/src/card.css b/src/card.css\n"
            "--- a/src/card.css\n"
            "+++ b/src/card.css\n"
            "@@ -1 +1 @@\n"
            "-.card{display:block}\n"
            "+.card{display:grid}\n"
            "</parameter>\n"
            "</function>\n"
            "</tool_call>",
            input_tokens,
            64,
        )


class _FakeEnvironment:
    def __init__(self) -> None:
        self.patch = ""

    def reset(self, **task_row: object) -> str:
        del task_row
        return "Task: repair the responsive card. Inspect, patch, and check it."

    def list_files(self, path: str = ".") -> str:
        del path
        return "src/card.css"

    def read_file(self, path: str, start: int = 1, end: int = 200) -> str:
        del path, start, end
        return "1: .card{display:block}"

    def search_code(self, query: str, path: str = ".") -> str:
        del query, path
        return "src/card.css:1: .card{display:block}"

    def apply_patch(self, patch: str) -> str:
        self.patch = patch
        return "patch applied"

    def run_check(self, check: str) -> str:
        return f"{check} passed"

    def _export_patch_for_evaluation(self) -> str:
        return self.patch

    def _cleanup(self) -> None:
        return None


class LocalEvaluationRunnerTests(unittest.TestCase):
    def test_evaluation_uses_the_exact_grpo_tool_schemas(self) -> None:
        try:
            from transformers.utils import get_json_schema
        except ImportError:
            self.skipTest("transformers is available in the pinned training extra")

        from autotrainer.environments.frontend import FrontendEnvironment

        environment = FrontendEnvironment()
        generated = [
            get_json_schema(getattr(environment, name))
            for name in (
                "list_files",
                "read_file",
                "search_code",
                "apply_patch",
                "run_check",
            )
        ]
        self.assertEqual(generated, _TOOL_SCHEMAS)

    def test_source_context_reads_the_frozen_commit_not_the_working_tree(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory) / "repository"
            source = repository / "src" / "App.tsx"
            source.parent.mkdir(parents=True)
            source.write_text("export const value = 'locked';\n", encoding="utf-8")
            subprocess.run(["git", "init", "-q", str(repository)], check=True)
            subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(repository),
                    "-c",
                    "user.name=AutoTrainer Test",
                    "-c",
                    "user.email=tests@autotrainer.local",
                    "commit",
                    "-q",
                    "-m",
                    "fixture",
                ],
                check=True,
            )
            revision = subprocess.run(
                ["git", "-C", str(repository), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            source.write_text("export const value = 'moving';\n", encoding="utf-8")

            context = _locked_source_context(
                {
                    "source_path": str(repository),
                    "source_revision": revision,
                    "manifest": {"runtime": {"workingDirectory": "."}},
                }
            )

            self.assertIn("export const value = 'locked'", context)
            self.assertNotIn("export const value = 'moving'", context)

    def test_working_directory_uses_a_literal_git_pathspec_and_stays_in_subtree(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory) / "repository"
            exact = repository / "apps" / "[ui]" / "App.tsx"
            decoy = repository / "apps" / "u" / "Secret.tsx"
            exact.parent.mkdir(parents=True)
            decoy.parent.mkdir(parents=True)
            exact.write_text("export const exact = 'visible';\n", encoding="utf-8")
            decoy.write_text("export const decoy = 'must-not-leak';\n", encoding="utf-8")
            subprocess.run(["git", "init", "-q", str(repository)], check=True)
            subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(repository),
                    "-c",
                    "user.name=AutoTrainer Test",
                    "-c",
                    "user.email=tests@autotrainer.local",
                    "commit",
                    "-q",
                    "-m",
                    "fixture",
                ],
                check=True,
            )
            revision = subprocess.run(
                ["git", "-C", str(repository), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()

            context = _locked_source_context(
                {
                    "source_path": str(repository),
                    "source_revision": revision,
                    "manifest": {
                        "runtime": {"workingDirectory": "apps/[ui]"}
                    },
                }
            )

            self.assertIn("export const exact = 'visible'", context)
            self.assertNotIn("must-not-leak", context)
            self.assertNotIn("apps/u/Secret.tsx", context)

    def test_runner_fingerprint_has_a_stable_golden_and_detects_source_changes(self) -> None:
        sources = {
            "autotrainer/evaluation.py": b"evaluation-v1\n",
            "autotrainer/local_evaluation_runner.py": b"runner-v1\n",
        }
        identity = _source_protocol_identity(sources)

        self.assertEqual(
            identity["sha256"],
            "sha256:9213146bba82812bd947c7a825eaffffd89799c6965d0c28ac7932422f06c84e",
        )
        changed = dict(sources)
        changed["autotrainer/evaluation.py"] = b"evaluation-v2\n"
        self.assertNotEqual(
            identity["sha256"], _source_protocol_identity(changed)["sha256"]
        )
        self.assertTrue(SOURCE_PROTOCOL_IDENTITY["files"])
        self.assertTrue(
            all(
                not Path(item["path"]).is_absolute()
                for item in SOURCE_PROTOCOL_IDENTITY["files"]
            )
        )
        self.assertEqual(
            builtin_runner_identity()["runtime_dependencies"],
            DECLARED_RUNTIME_DEPENDENCIES,
        )
        detached = builtin_runner_identity()
        detached["source_protocol"]["sha256"] = "sha256:" + "0" * 64
        detached["runtime_dependencies"]["torch"] = "forged"
        fresh = builtin_runner_identity()
        self.assertEqual(fresh["source_protocol"], SOURCE_PROTOCOL_IDENTITY)
        self.assertEqual(
            fresh["runtime_dependencies"], DECLARED_RUNTIME_DEPENDENCIES
        )

    def test_runtime_resolution_is_local_only_and_never_uses_a_token(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "snapshot"
            snapshot.mkdir()
            calls: list[dict[str, object]] = []

            def local_snapshot(**kwargs: object) -> str:
                calls.append(dict(kwargs))
                return str(snapshot)

            config = {"model": {"cache_dir": "model-cache"}}
            arm = {
                "id": "reference",
                "model": {
                    "id": "org/reference-9b",
                    "revision": REVISION,
                    "trust_remote_code": False,
                },
                "adapter": None,
            }
            with patch(
                "autotrainer.local_evaluation_runner._snapshot_download",
                return_value=local_snapshot,
            ):
                runtime = resolve_arm_runtime(config, root, arm)

            self.assertEqual(runtime.snapshot_path, snapshot.resolve())
            self.assertEqual(len(calls), 1)
            self.assertIs(calls[0]["local_files_only"], True)
            self.assertNotIn("token", calls[0])
            self.assertEqual(calls[0]["revision"], REVISION)

    def test_missing_reference_snapshot_is_a_clear_blocker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def missing(**kwargs: object) -> str:
                self.assertIs(kwargs["local_files_only"], True)
                raise FileNotFoundError("not cached")

            arm = {
                "id": "reference",
                "model": {
                    "id": "org/reference-9b",
                    "revision": REVISION,
                    "trust_remote_code": False,
                },
                "adapter": None,
            }
            with (
                patch(
                    "autotrainer.local_evaluation_runner._snapshot_download",
                    return_value=missing,
                ),
                self.assertRaisesRegex(
                    LocalEvaluationRunnerError,
                    "not downloaded.*never downloads weights",
                ),
            ):
                resolve_arm_runtime({"model": {}}, root, arm)

    def test_producer_uses_grpo_tools_and_reuses_one_loaded_arm(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter_path = root / "adapter"
            adapter_path.mkdir()
            (adapter_path / "adapter_config.json").write_text("{}", encoding="utf-8")
            (adapter_path / "adapter_model.safetensors").write_bytes(b"adapter")
            adapter_digest = _sha256_tree(adapter_path)
            runtimes = {
                arm_id: ArmRuntime(
                    arm_id=arm_id,
                    model_id=f"org/{arm_id}-9b",
                    revision=REVISION,
                    snapshot_path=root / arm_id,
                    adapter_name="grpo" if arm_id == "candidate" else "base",
                    adapter_path=adapter_path if arm_id == "candidate" else None,
                    adapter_sha256=adapter_digest if arm_id == "candidate" else None,
                )
                for arm_id in ("reference", "candidate")
            }
            loaded: list[str] = []
            generators: dict[str, _FakeGenerator] = {}

            def resolve(
                _config: dict, _root: Path, arm: dict
            ) -> ArmRuntime:
                return runtimes[str(arm["id"])]

            def load(spec: object) -> _FakeGenerator:
                arm_id = str(getattr(spec, "display_name"))
                loaded.append(arm_id)
                generator = _FakeGenerator()
                generators[arm_id] = generator
                return generator

            plan = {
                "arms": {
                    arm_id: {"id": arm_id, "model": {}}
                    for arm_id in ("reference", "candidate")
                }
            }
            producer = BuiltinEvaluationProducer(
                {"model": {}},
                root,
                plan,
                model_loader=load,
                runtime_resolver=resolve,
                environment_factory=_FakeEnvironment,
            )
            producer.preflight(["reference", "candidate"])

            first = root / "first" / "result.json"
            second = root / "second" / "result.json"
            producer.produce(_request("candidate"), first)
            producer.produce(_request("candidate"), second)

            self.assertEqual(loaded, ["candidate"])
            self.assertEqual(generators["candidate"].calls, 4)
            messages = generators["candidate"].messages[0]
            token_count = generators["candidate"].count_tokens_with_tools(messages, [])
            self.assertLessEqual(token_count + MAX_NEW_TOKENS, CONTEXT_TOKENS)
            rendered = json.dumps(messages)
            self.assertIn("repair the responsive card", rendered)
            self.assertNotIn("hidden-secret-verifier", rendered)
            self.assertNotIn("hidden_weight", rendered)
            result = json.loads(first.read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["producer"]["adapter_sha256"], adapter_digest)
            self.assertFalse(result["producer"]["fallback_models_used"])
            self.assertEqual(result["usage"]["tool_calls"], 1)
            self.assertTrue((first.parent / "patch.diff").is_file())

            # Switching arms releases the candidate before the reference load;
            # it never keeps two 9B generators resident on the single GPU.
            producer.produce(_request("reference"), root / "third" / "result.json")
            self.assertEqual(loaded, ["candidate", "reference"])
            producer.close()

    def test_producer_honors_frozen_grpo_turn_and_completion_budgets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = ArmRuntime(
                arm_id="candidate",
                model_id="org/candidate-9b",
                revision=REVISION,
                snapshot_path=root,
                adapter_name="base",
                adapter_path=None,
                adapter_sha256=None,
            )
            generator = _FakeGenerator()
            producer = BuiltinEvaluationProducer(
                {"model": {}},
                root,
                {
                    "arms": {"candidate": {"id": "candidate", "model": {}}},
                    "environment": {
                        "max_tool_calling_iterations": 1,
                        "max_completion_tokens": 128,
                    },
                },
                model_loader=lambda _spec: generator,
                runtime_resolver=lambda _config, _root, _arm: runtime,
                environment_factory=_FakeEnvironment,
            )

            producer.produce(_request("candidate"), root / "result.json")

            # One GRPO tool turn is permitted and the generation call receives
            # the exact completion budget frozen into the evaluation plan.
            self.assertEqual(generator.calls, 1)
            self.assertEqual(generator.max_token_budgets, [128])
            self.assertEqual(
                json.loads((root / "result.json").read_text(encoding="utf-8"))["status"],
                "completed",
            )

    def test_adapter_mutation_after_preflight_is_refused_before_model_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter = root / "adapter"
            adapter.mkdir()
            weights = adapter / "adapter_model.safetensors"
            weights.write_bytes(b"planned")
            runtime = ArmRuntime(
                arm_id="candidate",
                model_id="org/candidate-9b",
                revision=REVISION,
                snapshot_path=root,
                adapter_name="grpo",
                adapter_path=adapter,
                adapter_sha256=_sha256_tree(adapter),
            )
            loads: list[object] = []
            producer = BuiltinEvaluationProducer(
                {"model": {}},
                root,
                {"arms": {"candidate": {"id": "candidate", "model": {}}}},
                model_loader=lambda spec: loads.append(spec) or _FakeGenerator(),
                runtime_resolver=lambda _config, _root, _arm: runtime,
                environment_factory=_FakeEnvironment,
            )
            producer.preflight(["candidate"])
            weights.write_bytes(b"mutated")

            with self.assertRaisesRegex(
                LocalEvaluationRunnerError, "adapter changed after preflight"
            ):
                producer.produce(_request("candidate"), root / "result.json")
            self.assertEqual(loads, [])

    def test_generator_usage_cannot_exceed_frozen_context(self) -> None:
        class MisreportingGenerator(_FakeGenerator):
            def generate_with_tools(
                self, *args: object, **kwargs: object
            ) -> tuple[str, int, int]:
                return "diff --git a/a b/a\n", MAX_INPUT_TOKENS + 1, 1

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = ArmRuntime(
                arm_id="candidate",
                model_id="org/candidate-9b",
                revision=REVISION,
                snapshot_path=root,
                adapter_name="base",
                adapter_path=None,
                adapter_sha256=None,
            )
            producer = BuiltinEvaluationProducer(
                {"model": {}},
                root,
                {"arms": {"candidate": {"id": "candidate", "model": {}}}},
                model_loader=lambda _spec: MisreportingGenerator(),
                runtime_resolver=lambda _config, _root, _arm: runtime,
                environment_factory=_FakeEnvironment,
            )
            producer.preflight(["candidate"])
            with self.assertRaisesRegex(
                LocalEvaluationRunnerError, "exceeded the frozen 8K"
            ):
                producer.produce(_request("candidate"), root / "result.json")


if __name__ == "__main__":
    unittest.main()
