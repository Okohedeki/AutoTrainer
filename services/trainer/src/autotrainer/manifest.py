"""Versioned task-manifest contract shared by rollout environments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


class ManifestError(ValueError):
    """Raised when a task manifest violates the v0.1 contract."""


@dataclass(frozen=True, slots=True)
class TaskManifest:
    task_id: str
    instruction: str
    starting_snapshot: str
    split: str
    stack: str
    tools: tuple[str, ...]
    tool_call_limit: int
    token_budget: int
    command_timeout_seconds: int
    network_access: bool

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "TaskManifest":
        if payload.get("version") != "0.1":
            raise ManifestError("Only task manifest version 0.1 is supported")

        try:
            task = payload["task"]
            runtime = payload["runtime"]
            limits = payload["limits"]
            tools = tuple(payload["tools"])
        except (KeyError, TypeError) as error:
            raise ManifestError(f"Missing or invalid manifest field: {error}") from error

        if task.get("split") not in {"train", "evaluation"}:
            raise ManifestError("task.split must be train or evaluation")
        if runtime.get("stack") != "react-vite-tailwind":
            raise ManifestError("v0.1 supports only react-vite-tailwind")
        if limits.get("networkAccess") is not False:
            raise ManifestError("v0.1 environments must disable network access")
        if not tools:
            raise ManifestError("At least one controlled tool is required")

        return cls(
            task_id=str(task["id"]),
            instruction=str(task["instruction"]),
            starting_snapshot=str(task["startingSnapshot"]),
            split=str(task["split"]),
            stack=str(runtime["stack"]),
            tools=tools,
            tool_call_limit=int(limits["toolCalls"]),
            token_budget=int(limits["tokenBudget"]),
            command_timeout_seconds=int(limits["commandTimeoutSeconds"]),
            network_access=bool(limits["networkAccess"]),
        )
