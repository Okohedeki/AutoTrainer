"""Small local utilities for developing environment manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .manifest import TaskManifest


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m autotrainer")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate", help="validate a task manifest")
    validate.add_argument("manifest", type=Path)
    arguments = parser.parse_args()

    if arguments.command == "validate":
        payload = json.loads(arguments.manifest.read_text(encoding="utf-8"))
        task = TaskManifest.from_mapping(payload)
        print(f"valid: {task.task_id} ({task.split})")


if __name__ == "__main__":
    main()
