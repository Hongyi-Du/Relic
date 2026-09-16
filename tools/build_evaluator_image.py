#!/usr/bin/env python3
"""Build Relic's source-derived local evaluator image from a checkout.

Thin compatibility wrapper for ``relic evaluator-build``.  It never pushes an
image or writes a paper evaluator binding.
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
    DEFAULT_LOCAL_IMAGE_TAG,
    FORMAL_PLATFORM,
    EvaluatorReleaseError,
    build_local_evaluator,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default=DEFAULT_LOCAL_IMAGE_TAG)
    parser.add_argument("--platform", default=FORMAL_PLATFORM)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
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


if __name__ == "__main__":
    raise SystemExit(main())
