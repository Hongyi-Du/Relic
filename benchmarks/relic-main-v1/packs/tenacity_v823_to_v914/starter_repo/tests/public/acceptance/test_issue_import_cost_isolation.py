"""Acceptance check for issue_import_cost_isolation.

Written from the report the engineers were given, not from the fix.
It states in code what the issue states in prose: In a fresh interpreter, importing the package leaves the asyncio machinery absent from `sys.modules`, while the asynchronous retrying entry point and the ordinary synchronous names are still reachable from the package root.
"""
import subprocess
import sys


def test_package_import_defers_asyncio_and_keeps_root_entries_reachable():
    code = """
import sys
import tenacity

assert "asyncio" not in sys.modules
assert tenacity.AsyncRetrying is not None
assert tenacity.Retrying is not None
assert tenacity.retry is not None
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
