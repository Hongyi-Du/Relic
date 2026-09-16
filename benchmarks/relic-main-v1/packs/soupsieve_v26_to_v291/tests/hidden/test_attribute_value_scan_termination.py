"""Withheld contract: a malformed unquoted attribute test is rejected promptly, not chewed on.

The compile runs in a child interpreter under a wall-clock budget because the observable
defect is non-termination: a caller must get a syntax error back, not a process that never
returns. The budget is three orders of magnitude larger than a linear scan of this input
needs, so it does not measure implementation detail.
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


def test_unclosed_attribute_test_with_a_long_bare_value_is_refused() -> None:
    assert compile_outcome("[\\]!=Q7" + "Z" * 90) == "REFUSED"


def test_growing_the_bare_value_does_not_change_the_answer() -> None:
    assert compile_outcome("[\\]!=Q7" + "Z" * 140) == "REFUSED"
    assert compile_outcome("[data-x|=Q7" + "Z" * 140) == "REFUSED"


def test_well_formed_attribute_tests_still_match_the_same_elements() -> None:
    markup = """
    <div>
      <i id="escaped" a]b="Q7">one</i>
      <i id="dashed" data-x="Q7-tail">two</i>
      <i id="plain" data-x="other">three</i>
    </div>
    """
    assert selected_ids("[a\\]b]", markup) == ["escaped"]
    assert selected_ids("[a\\]b=Q7]", markup) == ["escaped"]
    assert selected_ids("[data-x|=Q7]", markup) == ["dashed"]
    assert selected_ids("[data-x!=Q7-tail]", markup) == ["escaped", "plain"]
