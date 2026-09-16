"""Withheld contract: padding inside a pattern costs time proportional to its length.

The select runs in a child interpreter under a wall-clock budget because the observable defect
is that a descendant combinator written as a long run of whitespace or comments stops coming
back. The budget is orders of magnitude larger than a linear pass over this input needs, so it
does not measure implementation detail.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Sequence, Tuple

from bs4 import BeautifulSoup

import soupsieve

BUDGET_SECONDS = 8.0

# The pattern is assembled inside the child from ``(text, repeat)`` chunks; a command line
# cannot carry tens of thousands of characters on every platform this suite runs on.
_SCRIPT = """
import json, os, sys
for entry in reversed(sys.argv[1].split(os.pathsep)):
    if entry and entry not in sys.path:
        sys.path.insert(0, entry)
import soupsieve
from bs4 import BeautifulSoup
pattern = "".join(text * repeat for text, repeat in json.loads(sys.argv[2]))
root = BeautifulSoup('<div><i id="target">x</i></div>', "html.parser").div
print("IDS:" + ",".join(element["id"] for element in soupsieve.select(pattern, root)))
print("SELECTED")
"""


def select_outcome(
    chunks: Sequence[Tuple[str, int]], budget: float = BUDGET_SECONDS
) -> str:
    """Report the ids the assembled pattern selected, or ``UNFINISHED`` past the budget."""
    try:
        completed = subprocess.run(
            [sys.executable, "-I", "-c", _SCRIPT,
             os.pathsep.join(sys.path), json.dumps([list(chunk) for chunk in chunks])],
            capture_output=True,
            text=True,
            timeout=budget,
        )
    except subprocess.TimeoutExpired:
        return "UNFINISHED"
    if completed.returncode != 0:
        return "CRASHED: " + (completed.stderr or "").strip()[-400:]
    lines = (completed.stdout or "").strip().splitlines()
    assert lines and lines[-1] == "SELECTED", completed.stdout
    return lines[0][len("IDS:"):]


def selected_ids(selector: str, markup: str) -> list[str]:
    root = BeautifulSoup(markup, "html.parser").div
    assert root is not None
    return [element["id"] for element in soupsieve.select(selector, root)]


def test_a_long_run_of_spaces_between_two_selectors_does_not_stall() -> None:
    assert select_outcome([("div", 1), (" ", 40000), ("i", 1)]) == "target"


def test_mixed_whitespace_between_two_selectors_does_not_stall() -> None:
    assert select_outcome([("div", 1), ("\t\n", 20000), ("i", 1)]) == "target"


def test_a_long_run_of_comments_between_two_selectors_does_not_stall() -> None:
    assert select_outcome([("div", 1), (" /*pad*/", 12000), (" i", 1)]) == "target"


def test_short_padding_still_trims_to_the_same_selector() -> None:
    markup = '<div><i id="target">x</i><b id="other">y</b></div>'
    assert selected_ids("  div i  ", markup) == ["target"]
    assert selected_ids("/*lead*/ div i /*trail*/", markup) == ["target"]
    assert selected_ids("\tdiv\t:is(i, b)\n", markup) == ["target", "other"]
    assert selected_ids("div /*mid*/ b", markup) == ["other"]
