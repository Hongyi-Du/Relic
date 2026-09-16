"""Acceptance check for issue_traceback_parse_truncated.

Written from the report the engineers were given, not from the fix.
It states in code what the issue states in prose: Parsing a traceback whose text ends before the exception line returns the frames it does contain instead of raising, and a complete traceback still round-trips to the identical text.
"""
from boltons.tbutils import ParsedException


def test_parsed_exception_accepts_tracebacks_truncated_after_frames():
    after_source = (
        'Traceback (most recent call last):\n'
        '  File "main.py", line 3, in <module>\n'
        '    print(add(1, 2))'
    )
    after_header = (
        'Traceback (most recent call last):\n'
        '  File "main.py", line 3, in <module>'
    )

    assert len(ParsedException.from_string(after_source).frames) == 1
    assert len(ParsedException.from_string(after_header).frames) == 1


def test_parsed_exception_complete_traceback_round_trips_exactly():
    traceback_text = (
        'Traceback (most recent call last):\n'
        '  File "main.py", line 3, in <module>\n'
        '    print(add(1, 2))\n'
        'TypeError: unsupported operand type(s) for +: \'int\' and \'str\''
    )

    assert ParsedException.from_string(traceback_text).to_string() == traceback_text
