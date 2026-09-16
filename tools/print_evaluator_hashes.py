#!/usr/bin/env python3
"""Qualify one released pack and print local, non-paper evaluator hashes.

Ported from the HCI source's ``tools/print_evaluator_hashes.py``.  Unlike the
development script, this release wrapper requires a container runtime and
labels the resulting values as local diagnostics: it does not emit a binding
usable by the paper runner.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from relic.evaluator_release import (  # noqa: E402
    FORMAL_PLATFORM,
    EvaluatorReleaseError,
    calculate_local_evaluator_hashes,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_id")
    parser.add_argument("--backend", required=True, choices=("docker", "apptainer"))
    parser.add_argument("--image", required=True)
    parser.add_argument("--platform", default=FORMAL_PLATFORM)
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args(argv)
    try:
        result = calculate_local_evaluator_hashes(
            dataset_id=args.dataset_id,
            backend=args.backend,
            container_image=args.image,
            container_platform=args.platform,
            timeout_seconds=args.timeout,
        )
    except EvaluatorReleaseError as exc:
        print(json.dumps({"status": "failed", "error": exc.code}), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
