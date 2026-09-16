"""Withheld contract: HTML markup is reduced to the text a reader would see.

Only the public text helpers are used, so any extractor that produces the right text
satisfies the contract.
"""

from __future__ import annotations

from boltons.strutils import camel2under, html2text, slugify, strip_ansi


def test_entities_and_numeric_references_become_characters() -> None:
    assert html2text("<p>Tom &amp; Jerry</p>") == "Tom & Jerry"
    assert html2text("&#72;&#x69;") == "Hi"
    assert html2text("<span>caf&eacute;</span>") == "caf\u00e9"


def test_tags_are_dropped_and_the_remaining_text_keeps_its_order() -> None:
    assert html2text("<ul><li>one</li><li>two</li></ul>") == "onetwo"
    assert html2text('<a href="https://example.test">link</a> tail') == "link tail"
    assert html2text("plain text") == "plain text"
    assert html2text("") == ""


def test_repeated_extractions_do_not_leak_state_between_calls() -> None:
    assert html2text("<b>first</b>") == "first"
    assert html2text("<b>second</b>") == "second"


def test_neighbouring_text_helpers_are_unaffected() -> None:
    assert strip_ansi("\x1b[31mred\x1b[0m") == "red"
    assert slugify("Hello World") == "hello_world"
    assert camel2under("HTMLParser") == "html_parser"
