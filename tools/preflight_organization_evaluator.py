#!/usr/bin/env python3
"""Requalify one released pack in a local evaluator runtime.

Ported from the HCI source preflight utility.  The receipt is deliberately
local diagnostic evidence, never an author-published paper binding.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from relic.evaluator_release import preflight_local_evaluator  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-id", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--backend", required=True, choices=("docker", "apptainer"))
    parser.add_argument("--container-image", required=True)
    parser.add_argument("--container-platform", required=True)
    parser.add_argument("--expected-environment-hash", required=True)
    parser.add_argument("--expected-qualification-hash", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
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


if __name__ == "__main__":
    raise SystemExit(main())
