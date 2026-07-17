from __future__ import annotations

from http.client import HTTPConnection
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest


SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE_ROOT / "src"))

from autotrainer.config import default_config, write_config  # noqa: E402
from autotrainer.model_host import (  # noqa: E402
    ModelHostError,
    _require_context_fit,
    create_model_host_server,
    resolve_host_spec,
)


class FakeGenerator:
    def __init__(self) -> None:
        self.requests: list[dict[str, object]] = []

    def generate(self, messages, **options):
        self.requests.append({"messages": messages, **options})
        return "A local answer.", 7, 4


class ModelHostTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.config_path = self.root / "autotrainer.yaml"
        self.revision = "a" * 40
        payload = default_config(revision=self.revision)
        payload["model"]["cache_dir"] = "./model-cache"
        write_config(self.config_path, payload, overwrite=False)
        self.snapshot = self.root / "model-cache" / "snapshot"
        self.snapshot.mkdir(parents=True)
        receipt = self.root / ".autotrainer" / "models" / "current.json"
        receipt.parent.mkdir(parents=True)
        receipt.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "model_id": "Qwen/Qwen3.5-9B",
                    "requested_revision": self.revision,
                    "revision": self.revision,
                    "snapshot_path": str(self.snapshot),
                    "cache_dir": str((self.root / "model-cache").resolve()),
                    "file_count": 1,
                    "logical_bytes": 1,
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_auto_adapter_prefers_completed_grpo_then_sft(self) -> None:
        sft = self.root / ".autotrainer" / "checkpoints" / "sft"
        grpo = self.root / ".autotrainer" / "checkpoints" / "grpo"
        sft.mkdir(parents=True)
        grpo.mkdir(parents=True)
        (sft / "adapter_config.json").write_text("{}", encoding="utf-8")

        self.assertEqual(resolve_host_spec(self.config_path).adapter_name, "sft")

        (grpo / "adapter_config.json").write_text("{}", encoding="utf-8")
        selected = resolve_host_spec(self.config_path)
        self.assertEqual(selected.adapter_name, "grpo")
        self.assertEqual(selected.adapter_path, grpo.resolve())

    def test_explicit_missing_adapter_is_not_presented_as_deployed(self) -> None:
        with self.assertRaisesRegex(ModelHostError, "not complete"):
            resolve_host_spec(self.config_path, "grpo")

    def test_context_limit_rejects_generation_before_cuda_overrun(self) -> None:
        self.assertEqual(_require_context_fit(4_000, 2_000, 32_768), 8_192)
        self.assertEqual(_require_context_fit(1_000, 500, 2_048), 2_048)
        with self.assertRaisesRegex(ModelHostError, "8192-token context limit"):
            _require_context_fit(7_000, 2_000, 32_768)

    def test_chat_endpoint_runs_the_injected_generator_and_reports_usage(self) -> None:
        generator = FakeGenerator()
        server = create_model_host_server(
            self.config_path,
            host="127.0.0.1",
            port=0,
            adapter="base",
            control_token="t" * 32,
            generator=generator,
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        connection = HTTPConnection("127.0.0.1", server.server_port, timeout=3)
        try:
            connection.request(
                "POST",
                "/v1/chat/completions",
                body=json.dumps(
                    {
                        "model": server.spec.display_name,
                        "messages": [{"role": "user", "content": "What changed?"}],
                        "max_tokens": 64,
                        "temperature": 0,
                        "stream": False,
                    }
                ),
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            payload = json.loads(response.read().decode("utf-8"))
        finally:
            connection.close()
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)

        self.assertEqual(response.status, 200)
        self.assertEqual(payload["object"], "chat.completion")
        self.assertEqual(payload["choices"][0]["message"]["content"], "A local answer.")
        self.assertIsNone(payload["choices"][0]["logprobs"])
        self.assertEqual(payload["choices"][0]["finish_reason"], "stop")
        self.assertEqual(payload["usage"]["total_tokens"], 11)
        self.assertEqual(generator.requests[0]["max_tokens"], 64)

    def test_streaming_and_unknown_fields_are_rejected_clearly(self) -> None:
        generator = FakeGenerator()
        server = create_model_host_server(
            self.config_path,
            host="127.0.0.1",
            port=0,
            adapter="base",
            control_token="t" * 32,
            generator=generator,
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        connection = HTTPConnection("127.0.0.1", server.server_port, timeout=3)
        try:
            connection.request(
                "POST",
                "/v1/chat/completions",
                body=json.dumps(
                    {
                        "messages": [{"role": "user", "content": "hello"}],
                        "stream": True,
                    }
                ),
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            payload = json.loads(response.read().decode("utf-8"))
        finally:
            connection.close()
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)

        self.assertEqual(response.status, 400)
        self.assertIn("streaming", payload["error"]["message"])
        self.assertEqual(generator.requests, [])

    def test_health_identifies_exact_base_and_adapter(self) -> None:
        generator = FakeGenerator()
        server = create_model_host_server(
            self.config_path,
            host="127.0.0.1",
            port=0,
            adapter="base",
            control_token="t" * 32,
            generator=generator,
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        connection = HTTPConnection("127.0.0.1", server.server_port, timeout=3)
        try:
            connection.request("GET", "/health")
            response = connection.getresponse()
            payload = json.loads(response.read().decode("utf-8"))
        finally:
            connection.close()
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)

        self.assertEqual(response.status, 200)
        self.assertEqual(payload["base_model"], "Qwen/Qwen3.5-9B")
        self.assertEqual(payload["revision"], self.revision)
        self.assertEqual(payload["adapter"], "base")


if __name__ == "__main__":
    unittest.main()
