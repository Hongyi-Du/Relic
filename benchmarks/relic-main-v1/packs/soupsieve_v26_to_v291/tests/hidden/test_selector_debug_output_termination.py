"""Withheld contract: dumping a compiled selector for inspection always terminates.

The dump runs in a child interpreter under a wall-clock budget because the observable defect
is non-termination: asking a compiled positional or attribute selector to describe itself
never comes back. The budget is orders of magnitude larger than rendering a few kilobytes of
text needs, and the assertions look at what the rendering says about the selector rather than
at how it is laid out, so nothing here pins down an implementation.
"""

from __future__ import annotations

import os
import subprocess
import sys

BUDGET_SECONDS = 8.0

_PREAMBLE = (
    "import os, sys\n"
    "for entry in reversed(sys.argv[1].split(os.pathsep)):\n"
    "    if entry and entry not in sys.path:\n"
    "        sys.path.insert(0, entry)\n"
)

_DUMP = """
import soupsieve
soupsieve.compile(%r).selectors.pretty()
print("DUMPED")
"""


def dump_outcome(pattern: str, budget: float = BUDGET_SECONDS) -> str:
    """Report the text a compiled ``pattern`` printed, or ``UNFINISHED`` past the budget."""
    try:
        completed = subprocess.run(
            [sys.executable, "-I", "-c", _PREAMBLE + (_DUMP % (pattern,)),
             os.pathsep.join(sys.path)],
            capture_output=True,
            text=True,
            timeout=budget,
        )
    except subprocess.TimeoutExpired:
        return "UNFINISHED"
    if completed.returncode != 0:
        return "CRASHED: " + (completed.stderr or "").strip()[-400:]
    return (completed.stdout or "").strip()


def rendering_of(pattern: str) -> str:
    """The dump a compiled ``pattern`` emitted, once it is known to have finished."""
    text = dump_outcome(pattern)
    assert text.endswith("DUMPED"), f"{pattern!r} produced {text[:200]!r}"
    body = text[: -len("DUMPED")].strip()
    assert body.count("\n") >= 4, f"{pattern!r} rendered {body[:200]!r}"
    return body


def test_positional_selectors_can_describe_themselves() -> None:
    body = rendering_of(":nth-of-type(3n-6)")
    assert "3" in body and "-6" in body


def test_attribute_selectors_can_describe_themselves() -> None:
    body = rendering_of("[data-x~=alpha]")
    assert "data-x" in body and "alpha" in body


def test_selector_kinds_that_already_described_themselves_still_do() -> None:
    assert "en" in rendering_of(":lang(en)")
    assert "beta" in rendering_of(".beta")
    assert "gamma" in rendering_of(':-soup-contains("gamma")')
    rendering_of("i > b")
