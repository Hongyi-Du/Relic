"""Withheld contract: importing the package must not drag the async event-loop machinery in.

A fresh interpreter is used because the test runner itself has already imported plenty of
modules. The probe only inspects ``sys.modules`` and the public package surface, so any
layout that defers the expensive import satisfies the contract.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import tenacity

_PROBE = """
import json, sys
import tenacity

json.dump(
    {
        "origin": tenacity.__file__,
        "eager_modules": sorted(
            name for name in ("asyncio", "concurrent.futures.thread")
            if name in sys.modules
        ),
        "public_surface": sorted(
            name
            for name in ("AsyncRetrying", "Retrying", "retry", "wait_fixed")
            if hasattr(tenacity, name)
        ),
    },
    sys.stdout,
)
"""


def _probe_fresh_interpreter():
    package_root = str(pathlib.Path(tenacity.__file__).resolve().parent.parent)
    env = dict(os.environ)
    env["PYTHONPATH"] = package_root
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    completed = subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)

    origin = os.path.normcase(str(pathlib.Path(payload["origin"]).resolve()))
    assert origin.startswith(os.path.normcase(package_root)), payload["origin"]
    return payload


def test_a_bare_import_does_not_start_the_asyncio_stack() -> None:
    assert _probe_fresh_interpreter()["eager_modules"] == []


def test_the_public_surface_is_still_reachable_after_a_bare_import() -> None:
    payload = _probe_fresh_interpreter()

    assert payload["public_surface"] == [
        "AsyncRetrying",
        "Retrying",
        "retry",
        "wait_fixed",
    ]
