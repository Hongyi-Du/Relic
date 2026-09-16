"""Acceptance check for issue_html_text_extraction.

Written from the report the engineers were given, not from the fix.
It states in code what the issue states in prose: html2text returns the readable text of an HTML fragment on the supported Python versions — tags removed, character references resolved, order preserved — and two successive calls share no state.
"""
from boltons.strutils import html2text


def test_html2text_extracts_text_resolves_references_and_has_no_shared_state():
    assert html2text('<p>First &amp; &#169;</p>') == 'First & ©'
    assert html2text('<p>Second</p>') == 'Second'
