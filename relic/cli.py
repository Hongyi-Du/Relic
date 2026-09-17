"""Command-line entry points for the Relic paper artifact."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from relic.benchmark import load_benchmark_manifest, verify_benchmark
from relic.paper_results import write_results
from relic.paths import default_output_root


def _plan_main(args: argparse.Namespace) -> int:
    from relic.source_runner import SourceMainRunnerError, run_source_main

    try:
        result = run_source_main(
            model=args.model,
            output_root=args.output_root,
            manifest_path=args.manifest,
            max_parallel=args.max_parallel,
            dry_run=True,
        )
    except SourceMainRunnerError as exc:
        print(json.dumps({"status": "failed", "error": exc.code}), file=sys.stderr)
        return 2
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    print(
        "Source-backed dry plan only: no provider or condition subprocess was started. "
        "Source case plans live under source-dry-run/."
    )
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


def _build_evaluator(args: argparse.Namespace) -> int:
    from relic.evaluator_release import EvaluatorReleaseError, build_local_evaluator

    try:
        result = build_local_evaluator(
            tag=args.tag,
            platform=args.platform,
            smoke=args.smoke,
        )
    except EvaluatorReleaseError as exc:
        print(json.dumps({"status": "failed", "error": exc.code}), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def _evaluator_hashes(args: argparse.Namespace) -> int:
    from relic.evaluator_release import (
        EvaluatorReleaseError,
        calculate_local_evaluator_hashes,
    )

    try:
        result = calculate_local_evaluator_hashes(
            dataset_id=args.dataset,
            backend=args.backend,
            container_image=args.container_image,
            container_platform=args.container_platform,
            timeout_seconds=args.timeout_seconds,
        )
    except EvaluatorReleaseError as exc:
        print(json.dumps({"status": "failed", "error": exc.code}), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["status"] == "passed" else 2


def _evaluator_preflight(args: argparse.Namespace) -> int:
    from relic.evaluator_release import preflight_local_evaluator

    payload, path = preflight_local_evaluator(
        repository_id=args.repository_id,
        dataset_id=args.dataset,
        backend=args.backend,
        container_image=args.container_image,
        container_platform=args.container_platform,
        expected_environment_hash=args.expected_environment_hash,
        expected_qualification_hash=args.expected_qualification_hash,
        timeout_seconds=args.timeout_seconds,
        output=args.output,
    )
    print(json.dumps({"receipt": str(path), **payload}, indent=2, ensure_ascii=False))
    return 0 if payload["status"] == "passed" else 2


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
    from relic.source_runner import SourceMainRunnerError, run_source_main

    try:
        result = run_source_main(
            model=args.model,
            output_root=args.output_root,
            manifest_path=args.manifest,
            max_parallel=args.max_parallel,
            dry_run=args.dry_run,
            resume=args.resume,
            retry_failed=args.retry_failed,
            evaluator_bindings_path=args.evaluator_bindings,
            batch_ids=args.batch,
            workloads=args.workload,
            seeds=args.seed,
        )
    except SourceMainRunnerError as exc:
        print(json.dumps({"status": "failed", "error": exc.code}), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print(json.dumps({"status": "interrupted", "error": "scheduler_interrupted"}), file=sys.stderr)
        return 130
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    if args.dry_run:
        print(
            "Dry run only: no provider or condition subprocess was started; "
            "source case plans were materialized under source-dry-run/."
        )
    return 0 if result.failed_batches == 0 else 2


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


def _replay_trace(args: argparse.Namespace) -> int:
    from relic.replay import load_trace, resolve_public_trace_path

    try:
        path = resolve_public_trace_path(
            trace_path=args.trace,
            run_directory=args.run_dir,
            cell_directory=args.cell_dir,
        )
        trace = load_trace(path)
    except (OSError, ValueError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}), file=sys.stderr)
        return 2
    final = trace["frames"][-1]
    organization = final["organization"]
    print(
        json.dumps(
            {
                "schema_version": "relic-replay-summary-v1",
                "status": "passed",
                "run_id": trace["run_id"],
                "trace_sha256": trace["trace_sha256"],
                "ticks": final["tick"],
                "frames": len(trace["frames"]),
                "agents": len(organization["agents"]),
                "tasks": len(organization["tasks"]),
                "proposals": len(organization["proposals"]),
                "protocols": len(organization["protocols"]),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


def _inspect_trace(args: argparse.Namespace) -> int:
    from relic.inspector import serve_inspector
    from relic.replay import resolve_public_trace_path

    try:
        path = resolve_public_trace_path(
            trace_path=args.trace,
            run_directory=args.run_dir,
            cell_directory=args.cell_dir,
        )
        serve_inspector(
            trace_path=path,
            host=args.host,
            port=args.port,
            mode=args.mode,
            open_browser=args.open_browser,
            verbose=args.verbose,
            allow_remote=args.allow_remote,
        )
    except (OSError, ValueError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}), file=sys.stderr)
        return 2
    return 0


def _export_trace(args: argparse.Namespace) -> int:
    """Validate and optionally copy an already exported public trace.

    This command does not reconstruct a trace from private checkpoints or a
    legacy replay.  New source runs export incrementally while their public
    state is available; historic records without that asset remain a gap.
    """

    from relic.replay import copy_public_trace_prefix, load_trace, resolve_public_trace_path

    try:
        path = resolve_public_trace_path(
            trace_path=args.trace,
            run_directory=args.run_dir,
            cell_directory=args.cell_dir,
        )
        if args.output is not None:
            path = copy_public_trace_prefix(path, args.output)
        trace = load_trace(path)
    except (OSError, ValueError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}), file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "schema_version": "relic-public-trace-export-v1",
                "status": "passed",
                "trace_path": str(path),
                "trace_sha256": trace["trace_sha256"],
                "frames": len(trace["frames"]),
                "run_id": trace["run_id"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _add_inspector_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--port",
        type=int,
        default=os.environ.get("RELIC_INSPECTOR_PORT", "8765"),
    )
    parser.add_argument("--mode", choices=("replay", "live"), default="replay")
    parser.add_argument("--open-browser", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--allow-remote", action="store_true")


def _add_trace_source_arguments(parser: argparse.ArgumentParser) -> None:
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--trace", type=Path)
    source.add_argument(
        "--run-dir",
        type=Path,
        help="source-run directory containing public/relic-trace-v1.json",
    )
    source.add_argument(
        "--cell-dir",
        type=Path,
        help="single-cell directory containing public/relic-trace-v1.json",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="relic")
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser(
        "plan-main", help="materialize the source-backed 30-batch / 120-cell dry plan"
    )
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

    evaluator_build = subparsers.add_parser(
        "evaluator-build",
        help=(
            "build the source-derived local evaluator image; never creates a paper binding"
        ),
    )
    evaluator_build.add_argument(
        "--tag",
        default="relic-oss-evaluator:local",
        help="local Docker tag only (default: relic-oss-evaluator:local)",
    )
    evaluator_build.add_argument(
        "--platform",
        default="linux/amd64",
        help="frozen evaluator platform (must be linux/amd64)",
    )
    evaluator_build.add_argument(
        "--smoke",
        action="store_true",
        help="run a network-isolated, read-only dependency import smoke after build",
    )
    evaluator_build.set_defaults(func=_build_evaluator)

    evaluator_hashes = subparsers.add_parser(
        "evaluator-hashes",
        help=(
            "qualify one released pack locally and print diagnostic hashes, not a paper binding"
        ),
    )
    evaluator_hashes.add_argument("--dataset", required=True, help="relic-main-v1 pack ID")
    evaluator_hashes.add_argument("--backend", required=True, choices=("docker", "apptainer"))
    evaluator_hashes.add_argument("--container-image", required=True)
    evaluator_hashes.add_argument("--container-platform", default="linux/amd64")
    evaluator_hashes.add_argument("--timeout-seconds", type=int, default=900)
    evaluator_hashes.set_defaults(func=_evaluator_hashes)

    evaluator_preflight = subparsers.add_parser(
        "evaluator-preflight",
        help=(
            "write a local evaluator qualification receipt; it is not paper evidence"
        ),
    )
    evaluator_preflight.add_argument("--repository-id", required=True)
    evaluator_preflight.add_argument("--dataset", required=True, help="relic-main-v1 pack ID")
    evaluator_preflight.add_argument(
        "--backend", required=True, choices=("docker", "apptainer")
    )
    evaluator_preflight.add_argument("--container-image", required=True)
    evaluator_preflight.add_argument("--container-platform", required=True)
    evaluator_preflight.add_argument("--expected-environment-hash", required=True)
    evaluator_preflight.add_argument("--expected-qualification-hash", required=True)
    evaluator_preflight.add_argument("--timeout-seconds", type=int, default=300)
    evaluator_preflight.add_argument("--output", type=Path, required=True)
    evaluator_preflight.set_defaults(func=_evaluator_preflight)

    cell = subparsers.add_parser(
        "run-cell", help="legacy single-cell compatibility runner; not the paired paper executor"
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
        help="run or resume the source-backed paired 120-cell single-model study",
    )
    main_run.add_argument("--model", default=None, help="required for a new run")
    main_run.add_argument("--output-root", type=Path, default=None)
    main_run.add_argument("--manifest", type=Path, default=None)
    main_run.add_argument("--max-parallel", type=int, default=1)
    main_run.add_argument("--dry-run", action="store_true")
    main_run.add_argument("--resume", action="store_true")
    main_run.add_argument("--retry-failed", action="store_true")
    main_run.add_argument(
        "--evaluator-bindings",
        type=Path,
        default=None,
        help=(
            "JSON mapping of each formal pack to its digest-pinned evaluator binding; "
            "required before any non-dry-run source batch starts"
        ),
    )
    main_run.add_argument(
        "--batch",
        action="append",
        default=[],
        help="narrow safely to one paired batch id, e.g. w01__seed1401 (repeatable)",
    )
    main_run.add_argument(
        "--workload",
        action="append",
        default=[],
        help="narrow safely to one canonical workload id, e.g. w01 (repeatable)",
    )
    main_run.add_argument(
        "--seed",
        action="append",
        type=int,
        default=[],
        help="narrow safely to one canonical seed (repeatable)",
    )
    main_run.set_defaults(func=_run_main)

    status = subparsers.add_parser(
        "status", help="read public status and verified checkpoint sidecars"
    )
    status_location = status.add_mutually_exclusive_group(required=True)
    status_location.add_argument("--cell-dir", type=Path)
    status_location.add_argument("--output-root", type=Path)
    status.add_argument("--cell-id", default=None)
    status.set_defaults(func=_status)

    replay = subparsers.add_parser(
        "replay", help="validate and summarize one public relic-trace-v1 file"
    )
    _add_trace_source_arguments(replay)
    replay.set_defaults(func=_replay_trace)

    inspect = subparsers.add_parser(
        "inspect", help="open the public Inspector for a relic-trace-v1 file"
    )
    _add_trace_source_arguments(inspect)
    _add_inspector_arguments(inspect)
    inspect.set_defaults(func=_inspect_trace)

    export_trace = subparsers.add_parser(
        "export-trace",
        help="validate or copy one already-exported public relic-trace-v1 file",
    )
    _add_trace_source_arguments(export_trace)
    export_trace.add_argument(
        "--output",
        type=Path,
        default=None,
        help="new destination for a verified immutable public trace copy",
    )
    export_trace.set_defaults(func=_export_trace)

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
