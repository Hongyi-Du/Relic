"""Command-line entry points for the Relic paper artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from relic.benchmark import load_benchmark_manifest, verify_benchmark
from relic.manifest import build_main_manifest, write_manifest


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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
