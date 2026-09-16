"""Withheld contract: scanning a long element name costs time proportional to its length.

The compile runs in a child interpreter under a wall-clock budget because the observable
defect is that a long element name followed by an illegal character stops coming back instead
of reporting the illegal character. The budget is orders of magnitude larger than a linear
pass over this input needs, so it does not measure implementation detail.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Sequence, Tuple

from bs4 import BeautifulSoup

import soupsieve

BUDGET_SECONDS = 6.0

# The pattern is assembled inside the child from ``(text, repeat)`` chunks; a command line
# cannot carry tens of thousands of characters on every platform this suite runs on.
_SCRIPT = """
import json, os, sys
for entry in reversed(sys.argv[1].split(os.pathsep)):
    if entry and entry not in sys.path:
        sys.path.insert(0, entry)
import soupsieve
pattern = "".join(text * repeat for text, repeat in json.loads(sys.argv[2]))
try:
    soupsieve.compile(pattern)
except soupsieve.SelectorSyntaxError:
    print("REFUSED")
else:
    print("ACCEPTED")
"""


def compile_outcome(
    chunks: Sequence[Tuple[str, int]], budget: float = BUDGET_SECONDS
) -> str:
    """Report what compiling the assembled pattern did, or ``UNFINISHED`` past the budget."""
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
    return (completed.stdout or "").strip()


def selected_ids(selector: str, markup: str) -> list[str]:
    root = BeautifulSoup(markup, "html.parser").div
    assert root is not None
    return [element["id"] for element in soupsieve.select(selector, root)]


def test_a_long_element_name_followed_by_an_illegal_character_is_refused() -> None:
    assert compile_outcome([("b", 32000), ("@", 1)]) == "REFUSED"


def test_the_same_holds_when_the_long_name_sits_after_a_combinator() -> None:
    assert compile_outcome([("div > ", 1), ("b", 32000), ("@", 1)]) == "REFUSED"
    assert compile_outcome([("div ", 1), ("b", 24000), ("!", 1)]) == "REFUSED"


def test_long_class_and_id_names_stay_prompt() -> None:
    assert compile_outcome([(".", 1), ("b", 32000), ("@", 1)]) == "REFUSED"
    assert compile_outcome([("#", 1), ("b", 32000), ("@", 1)]) == "REFUSED"


def test_long_names_that_are_legal_still_compile_and_select() -> None:
    name = "b" * 3000
    markup = f'<div><{name} id="long">one</{name}><i id="short" class="{name}">two</i></div>'
    assert selected_ids(name, markup) == ["long"]
    assert selected_ids("." + name, markup) == ["short"]
    assert selected_ids("#long", markup) == ["long"]
