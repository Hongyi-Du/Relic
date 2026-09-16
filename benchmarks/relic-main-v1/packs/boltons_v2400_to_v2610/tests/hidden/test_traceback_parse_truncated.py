"""Withheld contract: a clipped traceback parses into whatever frames it does contain.

Only the public parsing entry point and the resulting frame data are inspected.
"""

from __future__ import annotations

from boltons.tbutils import ParsedException

CLIPPED = (
    "Traceback (most recent call last):\n"
    '  File "/srv/app/handler.py", line 41, in dispatch\n'
    "    return route(request)\n"
    '  File "/srv/app/routes.py", line 12, in route\n'
    "    return _lookup(request.path)\n"
)

COMPLETE = CLIPPED + "KeyError: '/missing'"


def test_a_traceback_missing_its_exception_line_still_parses() -> None:
    parsed = ParsedException.from_string(CLIPPED)

    assert [frame["funcname"] for frame in parsed.frames] == ["dispatch", "route"]
    assert parsed.frames[-1]["filepath"] == "/srv/app/routes.py"
    assert parsed.frames[-1]["lineno"] == "12"
    assert parsed.frames[-1]["source_line"] == "return _lookup(request.path)"


def test_a_traceback_clipped_mid_frame_keeps_the_header_it_has() -> None:
    parsed = ParsedException.from_string(
        "Traceback (most recent call last):\n"
        '  File "/srv/app/handler.py", line 41, in dispatch\n'
    )

    assert len(parsed.frames) == 1
    assert parsed.frames[0]["funcname"] == "dispatch"
    assert parsed.frames[0]["lineno"] == "41"


def test_a_complete_traceback_still_round_trips() -> None:
    parsed = ParsedException.from_string(COMPLETE)

    assert parsed.exc_type == "KeyError"
    assert parsed.exc_msg == "'/missing'"
    assert len(parsed.frames) == 2
    assert parsed.to_string() == COMPLETE
