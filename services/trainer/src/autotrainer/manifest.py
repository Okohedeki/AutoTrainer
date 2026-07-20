"""Versioned task-manifest contract shared by rollout environments."""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose
from typing import Any, Mapping


class ManifestError(ValueError):
    """Raised when a task manifest violates a supported contract."""


V1_TOOLS = {
    "list_files",
    "read_file",
    "search_code",
    "apply_patch",
    "replace_text",
    "run_check",
}
REWARD_KEYS = {
    "regressionSafety",
    "taskTests",
    "responsiveRules",
    "designRules",
    "patchQuality",
}


@dataclass(frozen=True, slots=True)
class TaskManifest:
    version: str
    task_id: str
    instruction: str
    source_id: str
    starting_revision: str
    split: str
    group_id: str
    working_directory: str
    runtime_commands: dict[str, str]
    tools: tuple[str, ...]
    verifier_bundle: str | None
    verifier_command: str | None
    verifier_report_path: str | None
    reward_weights: dict[str, float]
    tool_call_limit: int
    command_timeout_seconds: int
    episode_timeout_seconds: int
    network_access: bool
    stack: str

    @property
    def starting_snapshot(self) -> str:
        """Compatibility name for the original v0.1 Python API."""

        return f"{self.source_id}@{self.starting_revision}"

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "TaskManifest":
        version = payload.get("version")
        if version == "0.1":
            return cls._from_v01(payload)
        if version == "1.0":
            return cls._from_v1(payload)
        raise ManifestError("task manifest version must be 0.1 or 1.0")

    @staticmethod
    def _common(payload: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], tuple[str, ...]]:
        try:
            task = payload["task"]
            runtime = payload["runtime"]
            limits = payload["limits"]
            tools = tuple(payload["tools"])
        except (KeyError, TypeError) as error:
            raise ManifestError(f"missing or invalid manifest field: {error}") from error
        if not isinstance(task, Mapping) or not isinstance(runtime, Mapping) or not isinstance(limits, Mapping):
            raise ManifestError("task, runtime, and limits must be mappings")
        if task.get("split") not in {"train", "evaluation"}:
            raise ManifestError("task.split must be train or evaluation")
        if limits.get("networkAccess") is not False:
            raise ManifestError("environments must disable network access")
        if not tools:
            raise ManifestError("at least one controlled tool is required")
        return task, runtime, limits, tools

    @staticmethod
    def _weights(payload: Mapping[str, Any]) -> dict[str, float]:
        rewards = payload.get("rewards")
        if not isinstance(rewards, Mapping):
            raise ManifestError("rewards must be a mapping")
        if rewards.get("buildGate") is not True or rewards.get("regressionGate") is not True:
            raise ManifestError("buildGate and regressionGate must be true")
        try:
            weights = {key: float(rewards[key]) for key in REWARD_KEYS}
        except (KeyError, TypeError, ValueError) as error:
            raise ManifestError(f"invalid reward weights: {error}") from error
        if any(value < 0 or value > 1 for value in weights.values()):
            raise ManifestError("reward weights must be between 0 and 1")
        if not isclose(sum(weights.values()), 1.0, abs_tol=1e-8):
            raise ManifestError("reward weights must sum to 1")
        return weights

    @classmethod
    def _from_v01(cls, payload: Mapping[str, Any]) -> "TaskManifest":
        task, runtime, limits, tools = cls._common(payload)
        if runtime.get("stack") != "react-vite-tailwind":
            raise ManifestError("v0.1 supports only react-vite-tailwind")
        snapshot = str(task.get("startingSnapshot", ""))
        source_id, separator, revision = snapshot.partition("@")
        if not separator:
            source_id, revision = snapshot, "unresolved"
        commands = {
            key: str(runtime.get(key, ""))
            for key in ("install", "build", "tests", "browserTests")
        }
        return cls(
            version="0.1",
            task_id=str(task["id"]),
            instruction=str(task["instruction"]),
            source_id=source_id,
            starting_revision=revision,
            split=str(task["split"]),
            group_id=source_id,
            working_directory=".",
            runtime_commands=commands,
            tools=tools,
            verifier_bundle=None,
            verifier_command=None,
            verifier_report_path=None,
            reward_weights=cls._weights(payload),
            tool_call_limit=int(limits["toolCalls"]),
            command_timeout_seconds=int(limits["commandTimeoutSeconds"]),
            episode_timeout_seconds=int(limits.get("episodeTimeoutSeconds", 900)),
            network_access=False,
            stack="react-vite-tailwind",
        )

    @classmethod
    def _from_v1(cls, payload: Mapping[str, Any]) -> "TaskManifest":
        task, runtime, limits, tools = cls._common(payload)
        unknown_tools = sorted(set(tools) - V1_TOOLS)
        if unknown_tools:
            raise ManifestError(f"unsupported tools: {', '.join(unknown_tools)}")
        required_task = ("id", "instruction", "sourceId", "startingRevision", "split", "groupId")
        missing = [key for key in required_task if not str(task.get(key, "")).strip()]
        if missing:
            raise ManifestError(f"missing task fields: {', '.join(missing)}")
        verifier = payload.get("verifier")
        if not isinstance(verifier, Mapping):
            raise ManifestError("verifier must be a mapping")
        required_verifier = ("bundle", "command", "reportPath")
        missing_verifier = [key for key in required_verifier if not str(verifier.get(key, "")).strip()]
        if missing_verifier:
            raise ManifestError(f"missing verifier fields: {', '.join(missing_verifier)}")
        commands = {
            key: str(runtime.get(key, ""))
            for key in ("install", "build", "tests", "browserTests")
        }
        if not commands["build"] or not commands["tests"]:
            raise ManifestError("runtime.build and runtime.tests are required")
        for key in ("toolCalls", "commandTimeoutSeconds", "episodeTimeoutSeconds"):
            value = limits.get(key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ManifestError(f"limits.{key} must be a positive integer")
        return cls(
            version="1.0",
            task_id=str(task["id"]),
            instruction=str(task["instruction"]),
            source_id=str(task["sourceId"]),
            starting_revision=str(task["startingRevision"]),
            split=str(task["split"]),
            group_id=str(task["groupId"]),
            working_directory=str(runtime.get("workingDirectory", ".")),
            runtime_commands=commands,
            tools=tools,
            verifier_bundle=str(verifier["bundle"]),
            verifier_command=str(verifier["command"]),
            verifier_report_path=str(verifier["reportPath"]),
            reward_weights=cls._weights(payload),
            tool_call_limit=int(limits["toolCalls"]),
            command_timeout_seconds=int(limits["commandTimeoutSeconds"]),
            episode_timeout_seconds=int(limits["episodeTimeoutSeconds"]),
            network_access=False,
            stack=str(runtime.get("stack", "frontend")),
        )
