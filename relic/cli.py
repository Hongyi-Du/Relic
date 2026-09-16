"""Command-line entry points for the Relic paper artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from relic.benchmark import load_benchmark_manifest, verify_benchmark
from relic.manifest import build_main_manifest, write_manifest
from relic.paper_results import write_results
from relic.paths import default_output_root


def _plan_main(args: argparse.Namespace) -> int:
    payload = build_main_manifest(
        model=args.model,
        output_root=args.output_root,
        max_parallel=args.max_parallel,
    )
    destination = args.manifest or Path(payload["output_root"]) / "run_manifest.json"
    write_manifest(payload, destination)
    advice = payload["resource_advice"]
    summary = {
        "model": payload["model"]["paper_label"],
        "workloads": len(payload["workloads"]),
        "seeds": payload["seeds"],
        "arms": list(payload["arms"]),
        "total_cells": payload["total_cells"],
        "max_parallel": payload["max_parallel"],
        "visible_memory_gib": advice["visible_memory_gib"],
        "recommended_max_parallel": advice["recommended_max_parallel"],
        "manifest": str(destination.resolve()),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if advice["warning"]:
        print(
            "WARNING: requested concurrency exceeds the conservative 16 GiB-per-cell "
            "memory policy. On WSL2, also check the WSL and Docker memory limits."
        )
    print("Dry plan only: no provider was contacted and no experiment was started.")
    return 0


def _verify_benchmark(_: argparse.Namespace) -> int:
    manifest = load_benchmark_manifest()
    failures = verify_benchmark()
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1
    print(f"OK: {manifest['benchmark']} ({len(manifest['workloads'])} frozen workloads)")
    return 0


def _build_paper_results(args: argparse.Namespace) -> int:
    json_path, markdown_path = write_results(args.source, args.output_directory)
    print(f"Wrote {json_path.resolve()}")
    print(f"Wrote {markdown_path.resolve()}")
    return 0


def _check_environment(args: argparse.Namespace) -> int:
    from relic.release_checks import check_environment, report_exit_code

    report = check_environment(
        scope=args.scope,
        model=args.model,
        requested_parallelism=args.max_parallel,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report_exit_code(report)


def _smoke(args: argparse.Namespace) -> int:
    from relic.release_checks import report_exit_code, smoke

    report = smoke(mode=args.mode, model=args.model)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report_exit_code(report)


def _run_cell(args: argparse.Namespace) -> int:
    from relic.cell_spec import compile_cell_spec
    from relic.cell_worker import CellWorkerError, run_cell

    try:
        spec = compile_cell_spec(
            model=args.model,
            workload=args.workload,
            arm=args.arm,
            seed=args.seed,
            output_root=args.output_root,
        )
        result = run_cell(spec, cell_dir=args.output_dir, resume=args.resume)
    except (CellWorkerError, ValueError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    return 0


def _run_main(args: argparse.Namespace) -> int:
    from relic.main_runner import MainRunnerError, run_main

    try:
        result = run_main(
            model=args.model,
            output_root=args.output_root,
            manifest_path=args.manifest,
            max_parallel=args.max_parallel,
            dry_run=args.dry_run,
            resume=args.resume,
            retry_failed=args.retry_failed,
            cell_id=args.cell_id,
        )
    except MainRunnerError as exc:
        print(json.dumps({"status": "failed", "error": exc.code}), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print(json.dumps({"status": "interrupted", "error": "scheduler_interrupted"}), file=sys.stderr)
        return 130
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    if result.max_parallel > result.recommended_max_parallel:
        print(
            "WARNING: requested concurrency exceeds the conservative 16 GiB-per-cell "
            "memory policy. On WSL2, also check the WSL and Docker memory limits."
        )
    if args.dry_run:
        print("Dry run only: no provider, evaluator, or cell subprocess was started.")
    return 0


def _evaluate_cell(args: argparse.Namespace) -> int:
    from relic.cell_worker import CellWorkerError, evaluate_cell
    from relic.evaluation.user_run_batch import UserRunBatchError, evaluate_user_run_batch

    try:
        if args.cell_dir is not None:
            if args.selection != "completed" or args.dry_run or args.receipt_directory:
                raise ValueError("batch_options_require_manifest_or_output_root")
            result = evaluate_cell(cell_dir=args.cell_dir)
        else:
            result = evaluate_user_run_batch(
                manifest_path=args.manifest,
                output_root=args.output_root,
                receipt_directory=args.receipt_directory,
                selection=args.selection,
                dry_run=args.dry_run,
            )
    except (CellWorkerError, UserRunBatchError, ValueError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    failed = getattr(result, "failed_cells", getattr(result, "failed", 0))
    return 0 if failed == 0 else 2


def _aggregate_user_runs(args: argparse.Namespace) -> int:
    from relic.evaluation.user_run_aggregate import (
        UserRunAggregateError,
        build_user_run_aggregate,
    )

    try:
        result = build_user_run_aggregate(
            args.evaluation_manifest,
            output_directory=args.output_directory,
            allow_partial=args.allow_partial,
        )
    except (UserRunAggregateError, ValueError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    return 0


def _status(args: argparse.Namespace) -> int:
    from relic.cell_worker import CellWorkerError, inspect_cell

    if args.cell_dir is not None:
        cell_directories = [args.cell_dir.expanduser().resolve()]
    else:
        root = (args.output_root or default_output_root()).expanduser().resolve()
        cell_directories = sorted(
            path.parent.parent
            for path in root.rglob("public/status.json")
            if path.is_file()
        )
    rows: list[dict] = []
    try:
        for cell_dir in cell_directories:
            row = inspect_cell(cell_dir)
            if args.cell_id and row.get("cell_id") != args.cell_id:
                continue
            rows.append(row)
    except (CellWorkerError, ValueError, OSError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}), file=sys.stderr)
        return 2
    payload: object = rows[0] if args.cell_dir is not None and len(rows) == 1 else rows
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="relic")
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan-main", help="write a 120-cell single-model dry-run manifest")
    plan.add_argument("--model", required=True, help="canonical model config name")
    plan.add_argument("--output-root", type=Path, default=None)
    plan.add_argument("--manifest", type=Path, default=None)
    plan.add_argument("--max-parallel", type=int, default=1)
    plan.set_defaults(func=_plan_main)

    verify = subparsers.add_parser(
        "verify-benchmark", help="verify all frozen relic-main-v1 pack digests"
    )
    verify.set_defaults(func=_verify_benchmark)

    results = subparsers.add_parser(
        "build-paper-results", help="build canonical JSON and Markdown from paper values"
    )
    results.add_argument("--source", type=Path, default=None)
    results.add_argument("--output-directory", type=Path, default=None)
    results.set_defaults(func=_build_paper_results)

    environment = subparsers.add_parser(
        "check-env", help="check core or formal release prerequisites without model calls"
    )
    environment.add_argument("--scope", default="core", help="core or formal")
    environment.add_argument("--model", default=None, help="limit formal checks to one model")
    environment.add_argument("--max-parallel", type=int, default=None)
    environment.set_defaults(func=_check_environment)

    smoke_parser = subparsers.add_parser(
        "smoke", help="run a bounded mock or formal no-provider smoke check"
    )
    smoke_parser.add_argument("--mode", default="mock", help="mock or formal")
    smoke_parser.add_argument(
        "--model", default=None, help="formal mode defaults to gpt-5.6-terra"
    )
    smoke_parser.set_defaults(func=_smoke)

    cell = subparsers.add_parser(
        "run-cell", help="run one canonical main-study cell with checkpointing"
    )
    cell.add_argument("--model", required=True)
    cell.add_argument("--workload", required=True, help="W01 through W10")
    cell.add_argument("--arm", required=True, help="B0 through B3")
    cell.add_argument("--seed", required=True, type=int)
    destination = cell.add_mutually_exclusive_group()
    destination.add_argument("--output-root", type=Path, default=None)
    destination.add_argument("--output-dir", type=Path, default=None)
    cell.add_argument("--resume", action="store_true")
    cell.set_defaults(func=_run_cell)

    main_run = subparsers.add_parser(
        "run-main",
        aliases=["run-main-120"],
        help="run or resume the canonical 120-cell single-model study",
    )
    main_run.add_argument("--model", default=None, help="required for a new run")
    main_run.add_argument("--output-root", type=Path, default=None)
    main_run.add_argument("--manifest", type=Path, default=None)
    main_run.add_argument("--max-parallel", type=int, default=1)
    main_run.add_argument("--dry-run", action="store_true")
    main_run.add_argument("--resume", action="store_true")
    main_run.add_argument("--retry-failed", action="store_true")
    main_run.add_argument("--cell-id", default=None)
    main_run.set_defaults(func=_run_main)

    status = subparsers.add_parser(
        "status", help="read public status and verified checkpoint sidecars"
    )
    status_location = status.add_mutually_exclusive_group(required=True)
    status_location.add_argument("--cell-dir", type=Path)
    status_location.add_argument("--output-root", type=Path)
    status.add_argument("--cell-id", default=None)
    status.set_defaults(func=_status)

    evaluate = subparsers.add_parser(
        "evaluate",
        aliases=["evaluate-manifest"],
        help="evaluate one local cell or a user-created v2 run manifest",
    )
    evaluation_source = evaluate.add_mutually_exclusive_group(required=True)
    evaluation_source.add_argument("--cell-dir", type=Path)
    evaluation_source.add_argument("--manifest", type=Path)
    evaluation_source.add_argument(
        "--output-root",
        type=Path,
        help="user run directory containing exactly run_manifest.json",
    )
    evaluate.add_argument("--receipt-directory", type=Path, default=None)
    evaluate.add_argument(
        "--selection",
        choices=("completed", "failed-evaluation", "all-eligible"),
        default="completed",
    )
    evaluate.add_argument("--dry-run", action="store_true")
    evaluate.set_defaults(func=_evaluate_cell)

    aggregate = subparsers.add_parser(
        "aggregate-user-runs",
        help="summarize only user-created evaluation receipts; never paper raw runs",
    )
    aggregate.add_argument(
        "--evaluation-manifest",
        type=Path,
        action="append",
        required=True,
        help="repeat once per single-model evaluation batch",
    )
    aggregate.add_argument("--output-directory", type=Path, required=True)
    aggregate.add_argument("--allow-partial", action="store_true")
    aggregate.set_defaults(func=_aggregate_user_runs)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
