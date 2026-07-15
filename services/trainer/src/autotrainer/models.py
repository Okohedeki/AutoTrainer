"""Small, explicit model catalogue for known V1 profiles."""

from __future__ import annotations

from typing import Any


MODEL_CATALOG: dict[str, dict[str, Any]] = {
    "qwen3.5-9b-text": {
        "id": "Qwen/Qwen3.5-9B",
        "loader": "qwen3_5_text",
        "parameters": "9B",
        "license": "Apache-2.0",
        "v1_mode": "text-only causal LM; no processor, image inputs, or vision encoder",
        "notes": "Reference single-GPU profile. Resolve `main` to an immutable commit before training.",
    }
}


def resolve_model(value: str) -> dict[str, Any]:
    """Resolve a catalogue alias or return a custom Hugging Face model declaration."""

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
        "v1_mode": "text-only causal LM",
        "notes": "Custom model: verify architecture, license, chat template, and 24 GB memory use.",
    }
