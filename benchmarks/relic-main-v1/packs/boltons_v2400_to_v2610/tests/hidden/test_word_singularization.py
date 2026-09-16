"""Withheld contract: words that already end in a doubled 's' survive singularization.

Only the public word-form helpers are used.
"""

from __future__ import annotations

from boltons.strutils import cardinalize, pluralize, singularize

ALREADY_SINGULAR = ("glass", "boss", "kiss", "grass", "address", "process", "chess")


def test_words_ending_in_a_doubled_s_are_left_alone() -> None:
    for word in ALREADY_SINGULAR:
        assert singularize(word) == word


def test_case_is_preserved_and_singularizing_is_idempotent() -> None:
    assert singularize("Glass") == "Glass"
    assert singularize("GLASS") == "GLASS"
    assert singularize("Address") == "Address"

    for word in ("Glasses", "addresses", "processes", "classes"):
        once = singularize(word)
        assert singularize(once) == once


def test_regular_plurals_are_still_reduced() -> None:
    assert singularize("cats") == "cat"
    assert singularize("parties") == "party"
    assert singularize("classes") == "class"
    assert singularize("buses") == "bus"
    assert singularize("Glasses") == "Glass"

    assert pluralize("glass") == "glasses"
    assert pluralize("cat") == "cats"
    assert cardinalize("glass", 3) == "glasses"
    assert cardinalize("glass", 1) == "glass"
