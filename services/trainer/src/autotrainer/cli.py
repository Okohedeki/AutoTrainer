"""Command-line entry point for the declarative AutoTrainer workflow."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from .config import ConfigError, default_config, load_config, validate_mapping, write_config
from .doctor import run_doctor
from .models import resolve_model


def _emit(payload: Any, *, as_json: bool = False) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    elif isinstance(payload, str):
        print(payload)
    else:
        print(yaml.safe_dump(payload, sort_keys=False, width=100).rstrip())


def _save_loaded(config: Any) -> None:
    write_config(config.path, config.data, overwrite=True)


def _config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=Path("autotrainer.yaml"))
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="autotrainer",
        description="Configure, inspect, and run a single-GPU QLoRA -> GRPO experiment.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init", help="create an autotrainer.yaml project")
    init.add_argument("directory", nargs="?", type=Path, default=Path("."))
    init.add_argument("--name", default=None)
    init.add_argument("--model", default="Qwen/Qwen3.5-9B")
    init.add_argument("--revision", default="main")
    init.add_argument("--force", action="store_true")

    models = subparsers.add_parser("models", help="inspect the small supported model catalogue")
    models_sub = models.add_subparsers(dest="models_command", required=True)
    models_list = models_sub.add_parser("list")
    models_list.add_argument("--json", action="store_true")
    models_show = models_sub.add_parser("show")
    models_show.add_argument("model")
    models_show.add_argument("--json", action="store_true")

    model = subparsers.add_parser("model", help="show or update the configured base model")
    model_sub = model.add_subparsers(dest="model_command", required=True)
    model_show = model_sub.add_parser("show")
    _config_argument(model_show)
    model_status = model_sub.add_parser("status", help="inspect the exact offline model snapshot")
    _config_argument(model_status)
    model_download = model_sub.add_parser(
        "download", help="resolve, download, and record the configured model snapshot"
    )
    _config_argument(model_download)
    model_use = model_sub.add_parser("use")
    model_use.add_argument("model")
    model_use.add_argument(
        "--revision",
        default=None,
        help="immutable revision; supported catalogue models use their pinned default",
    )
    model_use.add_argument(
        "--cache-dir",
        default=None,
        help="Hugging Face cache path used by both download and training",
    )
    _config_argument(model_use)

    source = subparsers.add_parser("source", help="declare and inspect repositories or datasets")
    source_sub = source.add_subparsers(dest="source_command", required=True)
    source_list = source_sub.add_parser("list")
    _config_argument(source_list)
    source_add = source_sub.add_parser("add")
    source_add.add_argument("uri")
    source_add.add_argument("--name", required=True)
    source_add.add_argument("--kind", required=True, choices=["repository", "sft_jsonl", "task_pack"])
    source_add.add_argument("--partition", choices=["train", "evaluation"], default="train")
    source_add.add_argument("--roles", default="", help="comma-separated repository roles")
    source_add.add_argument("--revision", default=None)
    source_add.add_argument("--license", dest="source_license", default="UNDECLARED")
    _config_argument(source_add)
    source_scan = source_sub.add_parser("scan")
    source_scan.add_argument("--write", action="store_true", help="write lock and raw reference artifacts")
    _config_argument(source_scan)
    source_materialize = source_sub.add_parser(
        "materialize", help="clone a declared repository and pin its detached commit"
    )
    source_materialize.add_argument("source_id")
    source_materialize.add_argument("--destination", type=Path, default=None)
    source_materialize.add_argument(
        "--no-update",
        action="store_true",
        help="leave the source URI unchanged after cloning",
    )
    _config_argument(source_materialize)

    validate = subparsers.add_parser("validate", help="validate config, paths, recipes, and declared sources")
    _config_argument(validate)

    compile_command = subparsers.add_parser(
        "compile", help="materialize deterministic source inventories and data readiness"
    )
    _config_argument(compile_command)

    lock = subparsers.add_parser("lock", help="resolve model and source revisions into an immutable lock")
    lock.add_argument(
        "--offline",
        action="store_true",
        help="do not resolve a mutable Hugging Face revision (useful only for inspecting lock shape)",
    )
    _config_argument(lock)

    plan = subparsers.add_parser("plan", help="show which experiment stages are ready or blocked")
    plan.add_argument("--write", action="store_true")
    _config_argument(plan)

    doctor = subparsers.add_parser("doctor", help="check GPU, sandbox, Python, and pinned ML packages")
    _config_argument(doctor)

    train = subparsers.add_parser("train", help="run a declared training stage")
    train_sub = train.add_subparsers(dest="train_command", required=True)
    train_sft = train_sub.add_parser("sft", help="train the 4-bit LoRA adapter on demonstrations")
    train_sft.add_argument("--dry-run", action="store_true")
    _config_argument(train_sft)
    train_rl = train_sub.add_parser("rl", help="continue the SFT adapter with GRPO environments")
    train_rl.add_argument("--dry-run", action="store_true")
    _config_argument(train_rl)

    evaluate = subparsers.add_parser(
        "evaluate",
        aliases=["benchmark"],
        help="plan, execute, ingest, and report held-out comparisons",
    )
    evaluate_sub = evaluate.add_subparsers(dest="evaluate_command", required=True)
    evaluate_plan = evaluate_sub.add_parser("plan", help="freeze the paired evaluation matrix")
    evaluate_plan.add_argument("--write", action="store_true")
    _config_argument(evaluate_plan)
    evaluate_run = evaluate_sub.add_parser(
        "run", help="execute a suite whose runner is an explicit argv command"
    )
    evaluate_run.add_argument("--suite", required=True)
    evaluate_run.add_argument("--resume", action="store_true")
    _config_argument(evaluate_run)
    evaluate_export = evaluate_sub.add_parser(
        "export", help="export verifier-free requests for an external runner such as Fable"
    )
    evaluate_export.add_argument("--suite", required=True)
    evaluate_export.add_argument("--output", type=Path, required=True)
    _config_argument(evaluate_export)
    evaluate_ingest = evaluate_sub.add_parser(
        "ingest", help="ingest patches and re-score them in the trusted local environment"
    )
    evaluate_ingest.add_argument("input", type=Path)
    evaluate_ingest.add_argument("--suite", required=True)
    _config_argument(evaluate_ingest)
    evaluate_report = evaluate_sub.add_parser(
        "report", help="write separate model-benchmark and Fable A/B reports"
    )
    _config_argument(evaluate_report)
    evaluate_review = evaluate_sub.add_parser("review", help="manage blind website reviews")
    review_sub = evaluate_review.add_subparsers(dest="review_command", required=True)
    review_export = review_sub.add_parser("export", help="create deterministic blind pairs")
    review_export.add_argument("--suite", required=True)
    review_export.add_argument("--output", type=Path, required=True)
    _config_argument(review_export)
    review_import = review_sub.add_parser("import", help="import immutable blind choices")
    review_import.add_argument("input", type=Path)
    review_import.add_argument("--suite", required=True)
    _config_argument(review_import)

    package = subparsers.add_parser(
        "package", help="assemble the evaluated LoRA adapter and audit artifacts"
    )
    package.add_argument("--output", type=Path, default=None)
    package.add_argument(
        "--allow-unverified",
        action="store_true",
        help="build a clearly marked development artifact without a verified winner claim",
    )
    _config_argument(package)

    serve = subparsers.add_parser("serve", help="run the loopback backend used by the human GUI")
    serve.add_argument("--config", type=Path, default=Path("autotrainer.yaml"))
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)

    return parser


def _run_init(arguments: argparse.Namespace) -> int:
    destination = arguments.directory.expanduser().resolve() / "autotrainer.yaml"
    project_name = arguments.name or arguments.directory.resolve().name or "frontend-expert-9b"
    write_config(
        destination,
        default_config(name=project_name, model_id=arguments.model, revision=arguments.revision),
        overwrite=arguments.force,
    )
    print(f"created {destination}")
    print("next: declare sources with `autotrainer source add ...`, then run `autotrainer plan`")
    return 0


def _run_models(arguments: argparse.Namespace) -> int:
    from .model_service import list_models

    if arguments.models_command == "list":
        payload = list_models()
    else:
        payload = resolve_model(arguments.model)
    _emit(payload, as_json=arguments.json)
    return 0


def _run_model(arguments: argparse.Namespace) -> int:
    from .model_service import download_model, get_model, model_status, select_model

    if arguments.model_command == "show":
        _emit(get_model(arguments.config), as_json=arguments.json)
        return 0
    if arguments.model_command == "status":
        _emit(model_status(arguments.config), as_json=arguments.json)
        return 0
    if arguments.model_command == "download":
        _emit(download_model(arguments.config), as_json=arguments.json)
        return 0
    result = select_model(
        arguments.config,
        arguments.model,
        revision=arguments.revision,
        cache_dir=arguments.cache_dir,
    )
    _emit(result, as_json=arguments.json)
    return 0


def _run_source(arguments: argparse.Namespace) -> int:
    config = load_config(arguments.config)
    if arguments.source_command == "list":
        _emit(config.sources, as_json=arguments.json)
        return 0
    if arguments.source_command == "add":
        if any(source.get("id") == arguments.name for source in config.sources):
            raise ConfigError(f"source id already exists: {arguments.name}")
        declared: dict[str, Any] = {
            "id": arguments.name,
            "kind": arguments.kind,
            "uri": arguments.uri,
            "partition": arguments.partition,
        }
        if arguments.kind == "repository":
            declared["roles"] = [role.strip() for role in arguments.roles.split(",") if role.strip()] or ["style"]
            declared["revision"] = arguments.revision or "HEAD"
            declared["license"] = {"spdx": arguments.source_license}
        elif arguments.kind == "sft_jsonl":
            declared["roles"] = ["demonstrations"]
        else:
            declared["roles"] = ["evaluation" if arguments.partition == "evaluation" else "rl_tasks"]
        config.data["sources"].append(declared)
        _save_loaded(config)
        _emit(declared, as_json=arguments.json)
        return 0

    if arguments.source_command == "materialize":
        from .sources import materialize_repository

        result = materialize_repository(
            config.data,
            config.root,
            arguments.source_id,
            destination=arguments.destination,
        )
        if not arguments.no_update:
            for index, source in enumerate(config.data["sources"]):
                if source.get("id") == arguments.source_id:
                    config.data["sources"][index] = result["updated_source"]
                    break
            _save_loaded(config)
        result["config_updated"] = not arguments.no_update
        _emit(result, as_json=arguments.json)
        return 0

    from .sources import scan_sources

    scan = scan_sources(config.data, config.root, write=arguments.write)
    _emit(scan, as_json=arguments.json)
    return 0 if not scan.get("errors") else 3


def _run_validate(arguments: argparse.Namespace) -> int:
    config = load_config(arguments.config)
    report = validate_mapping(config.data, root=config.root)
    try:
        from .sources import scan_sources

        scan = scan_sources(config.data, config.root, write=False)
        source_errors = tuple(str(item) for item in scan.get("errors", []))
        source_warnings = tuple(str(item) for item in scan.get("warnings", []))
    except (ImportError, AttributeError):
        source_errors = ()
        source_warnings = ()
    payload = {
        "valid": not report.errors and not source_errors,
        "errors": [*report.errors, *source_errors],
        "warnings": [*report.warnings, *source_warnings],
    }
    if arguments.json:
        _emit(payload, as_json=True)
    else:
        print("valid" if payload["valid"] else "invalid")
        for warning in payload["warnings"]:
            print(f"warning: {warning}")
        for error in payload["errors"]:
            print(f"error: {error}", file=sys.stderr)
    return 0 if payload["valid"] else 2


def _scan_and_plan(config: Any, *, write: bool) -> dict[str, Any]:
    from .planner import build_plan
    from .sources import scan_sources

    scan = scan_sources(config.data, config.root, write=write)
    plan = build_plan(config.data, config.root, scan)
    if write:
        config.artifact_dir.mkdir(parents=True, exist_ok=True)
        destination = config.artifact_dir / "plan.json"
        destination.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        plan["artifact"] = str(destination)
    return plan


def _run_compile(arguments: argparse.Namespace) -> int:
    config = load_config(arguments.config)
    from .compiler import compile_data, invalidate_compile_provenance
    from .sources import scan_sources

    # Scanning writes source-lock artifacts and can fail before compile_data is
    # entered. Invalidate the earlier success first so that exceptional exits
    # cannot leave stale evaluation provenance executable.
    invalidate_compile_provenance(config.data, config.root)
    scan = scan_sources(config.data, config.root, write=True)
    compiled = compile_data(config.data, config.root, scan)
    plan = _scan_and_plan(config, write=True)
    payload = {"scan": scan, "compile": compiled, "plan": plan}
    _emit(payload, as_json=arguments.json)
    return 0 if not scan.get("errors") and not compiled.get("errors") else 3


def _run_lock(arguments: argparse.Namespace) -> int:
    from .locking import build_lock, write_lock
    from .sources import scan_sources

    config = load_config(arguments.config, check_paths=True)
    scan = scan_sources(config.data, config.root, write=True)
    if scan.get("errors"):
        raise ConfigError("source scan failed; fix it before locking: " + "; ".join(scan["errors"]))
    lock = build_lock(config.data, config.root, scan, resolve_model=not arguments.offline)
    destination = write_lock(config.artifact_dir / "autotrainer.lock.json", lock)
    lock["artifact"] = str(destination)
    _emit(lock, as_json=arguments.json)
    return 0


def _run_plan(arguments: argparse.Namespace) -> int:
    config = load_config(arguments.config)
    plan = _scan_and_plan(config, write=arguments.write)
    _emit(plan, as_json=arguments.json)
    return 0 if not plan.get("errors") else 3


def _run_doctor(arguments: argparse.Namespace) -> int:
    config = load_config(arguments.config)
    result = run_doctor(environment_backend=str(config.data["environment"]["backend"]))
    _emit(result, as_json=arguments.json)
    return 0 if result["sft_ready"] else 3


def _run_train(arguments: argparse.Namespace) -> int:
    config = load_config(arguments.config, check_paths=True)
    if arguments.train_command == "sft":
        from .training import run_sft

        output = config.resolve_path(config.data["sft"]["output_dir"])
        result = run_sft(
            config.data,
            project_root=config.root,
            output_dir=output,
            dry_run=arguments.dry_run,
        )
    else:
        from .training import run_grpo

        output = config.resolve_path(config.data["grpo"]["output_dir"])
        result = run_grpo(
            config.data,
            project_root=config.root,
            output_dir=output,
            dry_run=arguments.dry_run,
        )
    _emit(result, as_json=arguments.json)
    return 0


def _run_evaluate(arguments: argparse.Namespace) -> int:
    config = load_config(arguments.config, check_paths=True)
    from .evaluation import (
        build_evaluation_plan,
        build_evaluation_reports,
        export_blind_review,
        export_evaluation_suite,
        import_blind_reviews,
        ingest_evaluation_results,
        run_command_suite,
        write_evaluation_plan,
    )

    if arguments.evaluate_command == "plan":
        result = (
            write_evaluation_plan(config.data, config.root)
            if arguments.write
            else build_evaluation_plan(config.data, config.root)
        )
    elif arguments.evaluate_command == "run":
        result = run_command_suite(
            config.data,
            config.root,
            arguments.suite,
            resume=arguments.resume,
        )
    elif arguments.evaluate_command == "export":
        result = export_evaluation_suite(
            config.data,
            config.root,
            arguments.suite,
            arguments.output,
        )
    elif arguments.evaluate_command == "ingest":
        result = ingest_evaluation_results(
            config.data,
            config.root,
            arguments.suite,
            arguments.input,
        )
    elif arguments.evaluate_command == "report":
        result = build_evaluation_reports(config.data, config.root)
    elif arguments.evaluate_command == "review" and arguments.review_command == "export":
        result = export_blind_review(
            config.data,
            config.root,
            arguments.suite,
            arguments.output,
        )
    elif arguments.evaluate_command == "review" and arguments.review_command == "import":
        result = import_blind_reviews(
            config.data,
            config.root,
            arguments.suite,
            arguments.input,
        )
    else:
        raise ConfigError(f"unhandled evaluation command: {arguments.evaluate_command}")
    _emit(result, as_json=arguments.json)
    return 0


def _run_package(arguments: argparse.Namespace) -> int:
    config = load_config(arguments.config, check_paths=True)
    from .packaging import build_adapter_package

    result = build_adapter_package(
        config.data,
        config.root,
        config_path=config.path,
        output_dir=arguments.output,
        allow_unverified=arguments.allow_unverified,
    )
    _emit(result, as_json=arguments.json)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "init":
            return _run_init(arguments)
        if arguments.command == "models":
            return _run_models(arguments)
        if arguments.command == "model":
            return _run_model(arguments)
        if arguments.command == "source":
            return _run_source(arguments)
        if arguments.command == "validate":
            return _run_validate(arguments)
        if arguments.command == "compile":
            return _run_compile(arguments)
        if arguments.command == "lock":
            return _run_lock(arguments)
        if arguments.command == "plan":
            return _run_plan(arguments)
        if arguments.command == "doctor":
            return _run_doctor(arguments)
        if arguments.command == "train":
            return _run_train(arguments)
        if arguments.command in {"evaluate", "benchmark"}:
            return _run_evaluate(arguments)
        if arguments.command == "package":
            return _run_package(arguments)
        if arguments.command == "serve":
            from .local_api import serve_local_api

            serve_local_api(arguments.config, host=arguments.host, port=arguments.port)
            return 0
    except (ConfigError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    parser.error(f"unhandled command: {arguments.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
