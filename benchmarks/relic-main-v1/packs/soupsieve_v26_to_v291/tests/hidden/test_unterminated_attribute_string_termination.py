"""Withheld contract: an attribute test whose quoted operand is never closed is refused promptly.

The compile runs in a child interpreter under a wall-clock budget because the observable
defect is non-termination: a caller must get a syntax error back, not a process that never
returns. The budget is orders of magnitude larger than a linear scan of this input needs, so
it does not measure implementation detail.
"""

from __future__ import annotations

import os
import subprocess
import sys

from bs4 import BeautifulSoup

import soupsieve

BUDGET_SECONDS = 8.0

_PREAMBLE = (
    "import os, sys\n"
    "for entry in reversed(sys.argv[1].split(os.pathsep)):\n"
    "    if entry and entry not in sys.path:\n"
    "        sys.path.insert(0, entry)\n"
)

_COMPILE = """
import soupsieve
try:
    soupsieve.compile(%r)
except soupsieve.SelectorSyntaxError:
    print("REFUSED")
else:
    print("ACCEPTED")
"""


def compile_outcome(pattern: str, budget: float = BUDGET_SECONDS) -> str:
    """Report what compiling ``pattern`` did, or ``UNFINISHED`` if it outlived the budget."""
    try:
        completed = subprocess.run(
            [sys.executable, "-I", "-c", _PREAMBLE + (_COMPILE % (pattern,)),
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


def selected_ids(selector: str, markup: str) -> list[str]:
    root = BeautifulSoup(markup, "html.parser").div
    assert root is not None
    return [element["id"] for element in soupsieve.select(selector, root)]


def test_unterminated_single_quoted_operand_is_refused() -> None:
    assert compile_outcome("[data-x='" + "q" * 400) == "REFUSED"


def test_unterminated_double_quoted_operand_is_refused() -> None:
    assert compile_outcome('[data-x="' + "q" * 400) == "REFUSED"
    assert compile_outcome('[data-x^="' + "q" * 900) == "REFUSED"


def test_properly_quoted_operands_still_match_the_same_elements() -> None:
    markup = """
    <div>
      <i id="spaced" data-x="q q">one</i>
      <i id="quoted" data-x="q'q">two</i>
      <i id="long" data-x="%s">three</i>
    </div>
    """ % ("q" * 400)
    assert selected_ids('[data-x="q q"]', markup) == ["spaced"]
    assert selected_ids("[data-x=\"q'q\"]", markup) == ["quoted"]
    assert selected_ids('[data-x="%s"]' % ("q" * 400), markup) == ["long"]
    assert selected_ids('[data-x^="q"]', markup) == ["spaced", "quoted", "long"]
