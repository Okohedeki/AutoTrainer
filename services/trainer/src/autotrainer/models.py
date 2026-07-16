"""Small, explicit model catalogue for known V1 profiles."""

from __future__ import annotations

from typing import Any


MODEL_CATALOG: dict[str, dict[str, Any]] = {
    "qwen3.5-9b-text": {
        "id": "Qwen/Qwen3.5-9B",
        "default_revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        "loader": "qwen3_5_text",
        "parameters": "9B",
        "license": "Apache-2.0",
        "purpose": "training_base",
        "trainable_v1": True,
        "expected_class": "Qwen3_5ForCausalLM",
        "minimum_vram_gib": 20,
        "v1_mode": "text-only causal LM; no processor, image inputs, or vision encoder",
        "notes": "Validated single-GPU training profile.",
    },
    "qwythos-9b-reference": {
        "id": "empero-ai/Qwythos-9B-Claude-Mythos-5-1M",
        "default_revision": "14a29bae5143091aeaf87ad37120de4cd57d592c",
        "loader": "auto_text_causal_lm",
        "parameters": "9B",
        "license": "verify upstream",
        "purpose": "benchmark_reference",
        "trainable_v1": False,
        "expected_class": None,
        "minimum_vram_gib": 20,
        "v1_mode": "reference inference only until its runner profile is verified",
        "notes": "Pinned deferred benchmark reference; not a V1 training base.",
    },
}


def resolve_model(value: str) -> dict[str, Any]:
    """Resolve a catalogue alias or return a custom Hugging Face declaration."""

    if value in MODEL_CATALOG:
        return dict(MODEL_CATALOG[value])
    for details in MODEL_CATALOG.values():
        if details["id"] == value:
            return dict(details)
    return {
        "id": value,
        "loader": "auto_text_causal_lm",
        "parameters": "unknown",
        "license": "verify upstream",
        "purpose": "custom",
        "trainable_v1": False,
        "expected_class": None,
        "minimum_vram_gib": None,
        "v1_mode": "text-only causal LM",
        "notes": "Custom model: verify architecture, license, chat template, and memory use.",
    }
