"""Loopback-only inference host for one downloaded AutoTrainer model.

The human GUI starts this process through :mod:`autotrainer.hosting_service`.
Agents can use the same process through ``autotrainer host``.  Keeping model
loading in a separate process releases GPU memory reliably when the operator
stops hosting and prevents the dashboard backend from importing the large ML
stack during ordinary setup work.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import secrets
import threading
import time
from typing import Any, Mapping, Protocol
from uuid import uuid4

from .config import ConfigError, load_config
from .device_gate import DeviceLease, acquire_device_lease
from .model_cache import inspect_model_cache
from .project_gate import ProjectLease, acquire_project_lease


MAX_REQUEST_BYTES = 256 * 1024
MAX_MESSAGES = 200
MAX_MESSAGE_CHARS = 200_000
MAX_NEW_TOKENS = 4096
MAX_CONTEXT_TOKENS = 8192


class ModelHostError(ConfigError):
    """Raised when a model cannot be exposed through the local host."""


@dataclass(frozen=True, slots=True)
class HostSpec:
    """Everything the model process needs after project paths are resolved."""

    config_path: Path
    model_id: str
    revision: str
    snapshot_path: Path
    adapter_name: str
    adapter_path: Path | None
    display_name: str


class TextGenerator(Protocol):
    """Small seam that keeps HTTP validation testable without loading CUDA."""

    def count_tokens(self, messages: list[dict[str, str]]) -> int:
        """Return the exact chat-template input length for context budgeting."""
        ...

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        seed: int | None,
    ) -> tuple[str, int, int]: ...


def _require_context_fit(
    prompt_tokens: int,
    max_tokens: int,
    native_limit: object,
) -> int:
    """Return the V1 context limit or reject a request before CUDA generation."""

    model_limit = MAX_CONTEXT_TOKENS
    if isinstance(native_limit, int) and not isinstance(native_limit, bool) and native_limit > 0:
        model_limit = min(model_limit, native_limit)
    if prompt_tokens + max_tokens > model_limit:
        raise ModelHostError(
            f"prompt plus max_tokens exceeds the local {model_limit}-token context limit"
        )
    return model_limit


def _adapter_output(config_path: Path, stage: str) -> Path:
    config = load_config(config_path)
    section = config.data.get(stage, {})
    if not isinstance(section, Mapping):
        raise ModelHostError(f"{stage} configuration is missing")
    value = section.get("output_dir")
    if not isinstance(value, str) or not value.strip():
        raise ModelHostError(f"{stage}.output_dir is missing")
    return config.resolve_path(value)


def _is_adapter(path: Path) -> bool:
    """Accept only a completed PEFT root, never a half-written run folder."""

    return path.is_dir() and (path / "adapter_config.json").is_file()


def resolve_host_spec(config_path: str | Path, adapter: str = "auto") -> HostSpec:
    """Resolve the immutable base snapshot and an optional trained adapter."""

    path = Path(config_path).expanduser().resolve()
    config = load_config(path)
    cache = inspect_model_cache(path)
    if cache.get("status") not in {"downloaded", "cached_unverified"}:
        raise ModelHostError(
            "Download the selected Hugging Face model before starting the local host."
        )
    snapshot_value = cache.get("snapshot_path")
    snapshot = Path(str(snapshot_value)).expanduser().resolve()
    if not snapshot.is_dir():
        raise ModelHostError("The downloaded model snapshot is no longer available.")

    choice = str(adapter).strip().lower()
    if choice not in {"auto", "grpo", "sft", "base"}:
        raise ModelHostError("adapter must be auto, grpo, sft, or base")

    selected_name = "base"
    selected_path: Path | None = None
    if choice == "auto":
        for stage in ("grpo", "sft"):
            candidate = _adapter_output(path, stage)
            if _is_adapter(candidate):
                selected_name, selected_path = stage, candidate
                break
    elif choice != "base":
        candidate = _adapter_output(path, choice)
        if not _is_adapter(candidate):
            raise ModelHostError(
                f"The {choice.upper()} adapter is not complete yet. Finish training or host the base model."
            )
        selected_name, selected_path = choice, candidate

    project_name = str(config.data.get("project", {}).get("name") or "autotrainer")
    display_name = f"{project_name}-{selected_name}"
    return HostSpec(
        config_path=path,
        model_id=str(config.model.get("id", "")),
        revision=str(config.model.get("revision", "")),
        snapshot_path=snapshot,
        adapter_name=selected_name,
        adapter_path=selected_path,
        display_name=display_name,
    )


def _load_generator(spec: HostSpec) -> TextGenerator:
    """Import and load the optional training stack only inside the host process."""

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except (ImportError, OSError) as error:
        raise ModelHostError(
            "Local hosting requires the pinned AutoTrainer training dependencies."
        ) from error

    if not torch.cuda.is_available():
        raise ModelHostError("Local hosting requires one CUDA GPU.")

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(spec.snapshot_path),
        local_files_only=True,
        trust_remote_code=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(spec.snapshot_path),
        local_files_only=True,
        trust_remote_code=False,
        device_map={"": 0},
        quantization_config=quantization,
        torch_dtype=torch.bfloat16,
    )
    if spec.adapter_path is not None:
        try:
            from peft import PeftModel
        except (ImportError, OSError) as error:
            raise ModelHostError("Loading a trained adapter requires PEFT.") from error
        model = PeftModel.from_pretrained(model, str(spec.adapter_path), is_trainable=False)
    model.eval()

    class TransformersGenerator:
        """Thin synchronous generator guarded by the HTTP server's GPU lock."""

        def count_tokens(self, messages: list[dict[str, str]]) -> int:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tokenizer(prompt, return_tensors="pt")
            return int(inputs["input_ids"].shape[-1])

        def generate(
            self,
            messages: list[dict[str, str]],
            *,
            max_tokens: int,
            temperature: float,
            top_p: float,
            seed: int | None,
        ) -> tuple[str, int, int]:
            if seed is not None:
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            prompt_tokens = int(inputs["input_ids"].shape[-1])
            _require_context_fit(
                prompt_tokens,
                max_tokens,
                getattr(model.config, "max_position_embeddings", None),
            )
            generate_options: dict[str, Any] = {
                "max_new_tokens": max_tokens,
                "do_sample": temperature > 0,
                "use_cache": True,
                "pad_token_id": tokenizer.eos_token_id,
            }
            if temperature > 0:
                generate_options.update(temperature=temperature, top_p=top_p)
            with torch.inference_mode():
                output = model.generate(**inputs, **generate_options)
            generated = output[0][prompt_tokens:]
            completion_tokens = int(generated.shape[-1])
            return (
                tokenizer.decode(generated, skip_special_tokens=True),
                prompt_tokens,
                completion_tokens,
            )

    return TransformersGenerator()


def _number(
    value: object,
    *,
    field: str,
    minimum: float,
    maximum: float,
    default: float,
) -> float:
    if value is None:
        return default
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ModelHostError(f"{field} must be a number")
    number = float(value)
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise ModelHostError(f"{field} must be between {minimum} and {maximum}")
    return number


def _chat_request(payload: object, *, expected_model: str) -> dict[str, Any]:
    """Validate the documented non-streaming chat subset served by V1."""

    if not isinstance(payload, Mapping):
        raise ModelHostError("request body must be a JSON object")
    allowed = {"model", "messages", "max_tokens", "temperature", "top_p", "seed", "stream"}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ModelHostError(f"unsupported chat fields: {', '.join(unknown)}")
    if payload.get("stream", False) is not False:
        raise ModelHostError("streaming is not available in V1")
    requested_model = str(payload.get("model", "")).strip()
    if requested_model and requested_model != expected_model:
        raise ModelHostError(f"unknown model: {requested_model}")
    raw_messages = payload.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages or len(raw_messages) > MAX_MESSAGES:
        raise ModelHostError(f"messages must contain between 1 and {MAX_MESSAGES} items")
    messages: list[dict[str, str]] = []
    character_count = 0
    for item in raw_messages:
        if not isinstance(item, Mapping) or set(item) != {"role", "content"}:
            raise ModelHostError("each message requires exactly role and content")
        role = item.get("role")
        content = item.get("content")
        if role not in {"system", "user", "assistant"} or not isinstance(content, str):
            raise ModelHostError("message roles are system, user, or assistant and content must be text")
        character_count += len(content)
        messages.append({"role": str(role), "content": content})
    if character_count > MAX_MESSAGE_CHARS:
        raise ModelHostError("messages exceed the local context request limit")

    max_tokens_value = payload.get("max_tokens", 512)
    if (
        not isinstance(max_tokens_value, int)
        or isinstance(max_tokens_value, bool)
        or not 1 <= max_tokens_value <= MAX_NEW_TOKENS
    ):
        raise ModelHostError(f"max_tokens must be between 1 and {MAX_NEW_TOKENS}")
    seed_value = payload.get("seed")
    if seed_value is not None and (
        not isinstance(seed_value, int) or isinstance(seed_value, bool) or seed_value < 0
    ):
        raise ModelHostError("seed must be a non-negative integer")
    return {
        "messages": messages,
        "max_tokens": max_tokens_value,
        "temperature": _number(
            payload.get("temperature"), field="temperature", minimum=0, maximum=2, default=0.2
        ),
        "top_p": _number(payload.get("top_p"), field="top_p", minimum=0, maximum=1, default=0.9),
        "seed": seed_value,
    }


class ModelHostServer(ThreadingHTTPServer):
    """HTTP host carrying one model plus project and physical-device leases."""

    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        spec: HostSpec,
        generator: TextGenerator,
        project_lease: ProjectLease,
        device_lease: DeviceLease,
        control_token: str,
    ) -> None:
        super().__init__(address, ModelHostHandler)
        self.spec = spec
        self.generator = generator
        self.project_lease = project_lease
        self.device_lease = device_lease
        self.control_token = control_token
        self.generation_lock = threading.Lock()

    def server_close(self) -> None:
        try:
            super().server_close()
        finally:
            # Wait for the one in-flight generation, drop the final model
            # reference, and clear the allocator before another project may
            # acquire GPU 0. Releasing the file lock first would recreate the
            # exact cross-project OOM race this lease prevents.
            with self.generation_lock:
                self.generator = None  # type: ignore[assignment]
                gc.collect()
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except (ImportError, OSError):
                    pass
                self.device_lease.release()
                self.project_lease.release()


class ModelHostHandler(BaseHTTPRequestHandler):
    """Serve the small OpenAI-compatible inference surface on loopback."""

    server: ModelHostServer

    def log_message(self, format: str, *args: object) -> None:
        super().log_message("AutoTrainer model host: " + format, *args)

    def _send(self, status: HTTPStatus, payload: Mapping[str, Any]) -> None:
        body = (json.dumps(dict(payload), ensure_ascii=False) + "\n").encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: HTTPStatus, message: str) -> None:
        self._send(status, {"error": {"message": message, "type": "invalid_request_error"}})

    def _read_json(self) -> object:
        content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
        if content_type != "application/json":
            raise ModelHostError("Content-Type must be application/json")
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as error:
            raise ModelHostError("invalid Content-Length") from error
        if not 1 <= length <= MAX_REQUEST_BYTES:
            raise ModelHostError("request body is empty or too large")
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ModelHostError("request body must be valid UTF-8 JSON") from error

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract
        if self.path == "/health":
            self._send(
                HTTPStatus.OK,
                {
                    "status": "ready",
                    "model": self.server.spec.display_name,
                    "base_model": self.server.spec.model_id,
                    "revision": self.server.spec.revision,
                    "adapter": self.server.spec.adapter_name,
                },
            )
        elif self.path == "/v1/models":
            self._send(
                HTTPStatus.OK,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": self.server.spec.display_name,
                            "object": "model",
                            "created": 0,
                            "owned_by": "autotrainer-local",
                        }
                    ],
                },
            )
        else:
            self._error(HTTPStatus.NOT_FOUND, "endpoint not found")

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract
        try:
            if self.path == "/_autotrainer/shutdown":
                supplied = self.headers.get("Authorization", "")
                expected = f"Bearer {self.server.control_token}"
                if not secrets.compare_digest(supplied, expected):
                    self._error(HTTPStatus.FORBIDDEN, "control token is invalid")
                    return
                self._send(HTTPStatus.OK, {"status": "stopping"})
                threading.Thread(target=self.server.shutdown, daemon=True).start()
                return
            if self.path != "/v1/chat/completions":
                self._error(HTTPStatus.NOT_FOUND, "endpoint not found")
                return
            request = _chat_request(
                self._read_json(), expected_model=self.server.spec.display_name
            )
            # The V1 host intentionally serializes GPU generation.  A second
            # request receives a clear retry response rather than competing for
            # the same single-GPU memory budget as the active request.
            if not self.server.generation_lock.acquire(blocking=False):
                self._error(HTTPStatus.TOO_MANY_REQUESTS, "the local GPU is serving another request")
                return
            try:
                text, prompt_tokens, completion_tokens = self.server.generator.generate(**request)
            finally:
                self.server.generation_lock.release()
            finish_reason = (
                "length"
                if completion_tokens >= int(request["max_tokens"])
                else "stop"
            )
            self._send(
                HTTPStatus.OK,
                {
                    "id": f"chatcmpl-{uuid4().hex}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": self.server.spec.display_name,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": text},
                            "logprobs": None,
                            "finish_reason": finish_reason,
                        }
                    ],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                },
            )
        except ModelHostError as error:
            self._error(HTTPStatus.BAD_REQUEST, str(error))
        except Exception:
            # Details remain in the host terminal.  The callable API never
            # returns paths, model internals, or environment values.
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "local generation failed")


def create_model_host_server(
    config_path: str | Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8791,
    adapter: str = "auto",
    control_token: str,
    generator: TextGenerator | None = None,
) -> ModelHostServer:
    """Create and load a host while holding the project GPU lease."""

    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ModelHostError("the V1 model host may only bind to loopback")
    # Port zero is useful for embedded tests and direct Python callers.  The
    # dashboard control plane requires an explicit non-zero port because it
    # must publish the endpoint before the child process finishes loading.
    if not 0 <= port <= 65_535:
        raise ModelHostError("port must be between 0 and 65535")
    if len(control_token) < 32:
        raise ModelHostError("a private control token is required")
    spec = resolve_host_spec(config_path, adapter)
    project_lease = acquire_project_lease(spec.config_path)
    device_lease = None
    try:
        device_lease = acquire_device_lease()
        loaded = generator or _load_generator(spec)
        return ModelHostServer(
            (host, port),
            spec,
            loaded,
            project_lease,
            device_lease,
            control_token,
        )
    except Exception:
        if device_lease is not None:
            device_lease.release()
        project_lease.release()
        raise


def serve_model(
    config_path: str | Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8791,
    adapter: str = "auto",
    control_token: str | None = None,
) -> None:
    """Load the selected weights and serve them until stopped."""

    token = control_token or os.environ.pop("AUTOTRAINER_HOST_TOKEN", "")
    server = create_model_host_server(
        config_path,
        host=host,
        port=port,
        adapter=adapter,
        control_token=token,
    )
    address, bound_port = server.server_address[:2]
    print(f"AutoTrainer model host: http://{address}:{bound_port}/v1")
    print(f"Model: {server.spec.display_name}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AutoTrainer local model host process")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8791)
    parser.add_argument("--adapter", choices=("auto", "grpo", "sft", "base"), default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        serve_model(
            arguments.config,
            host=arguments.host,
            port=arguments.port,
            adapter=arguments.adapter,
        )
    except (ConfigError, RuntimeError) as error:
        print(f"error: {error}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "HostSpec",
    "ModelHostError",
    "ModelHostServer",
    "create_model_host_server",
    "resolve_host_spec",
    "serve_model",
]
