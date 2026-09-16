"""Withheld contract: pseudo-classes whose state only exists in a live viewer still parse.

A selector naming one of these states must compile and simply select nothing, instead of
aborting the whole pattern. Exercised through the top-level compile/select entry points.
"""

from __future__ import annotations

import pytest
from bs4 import BeautifulSoup

import soupsieve

MARKUP = """
<div>
  <video id="clip" controls></video>
  <audio id="track" controls></audio>
  <input id="field" type="text" value="x">
  <div id="panel" popover>panel</div>
</div>
"""

LIVE_VIEWER_STATES = (
    ":autofill",
    ":buffering",
    ":fullscreen",
    ":picture-in-picture",
    ":popover-open",
    ":seeking",
    ":stalled",
    ":volume-locked",
)


def selected_ids(selector: str, markup: str = MARKUP) -> list[str]:
    root = BeautifulSoup(markup, "html.parser").div
    assert root is not None
    return [element["id"] for element in soupsieve.select(selector, root)]


def test_live_viewer_states_compile_and_select_nothing() -> None:
    for state in LIVE_VIEWER_STATES:
        assert selected_ids(state) == [], state


def test_live_viewer_states_compose_inside_larger_patterns() -> None:
    assert selected_ids(":is(:seeking, audio)") == ["track"]
    assert selected_ids("video:not(:buffering)") == ["clip"]
    assert selected_ids("div :is(:volume-locked, :popover-open, input)") == ["field"]


def test_genuinely_unknown_pseudo_classes_are_still_refused() -> None:
    with pytest.raises(soupsieve.SelectorSyntaxError):
        soupsieve.compile(":no-such-state-exists")
    with pytest.raises(soupsieve.SelectorSyntaxError):
        soupsieve.compile("video:almost-buffering")


def test_previously_recognized_unobservable_states_keep_working() -> None:
    assert selected_ids(":hover") == []
    assert selected_ids(":focus") == []
    assert selected_ids("video:not(:playing)") == ["clip"]
