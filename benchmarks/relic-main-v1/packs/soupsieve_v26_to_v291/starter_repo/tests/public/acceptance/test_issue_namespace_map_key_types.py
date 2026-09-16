"""Withheld contract: a namespace map is rejected unless every prefix is a string.

Exercised through the top-level ``compile`` entry point, so the check may live anywhere as
long as a caller passing a non-string prefix is told about it instead of getting a matcher
that can never resolve that prefix.
"""

from __future__ import annotations

import pytest
from bs4 import BeautifulSoup

import soupsieve

MARKUP = '<div><i id="alpha">a</i><b id="beta">b</b></div>'
URI = "http://example.invalid/ns"

BAD_PREFIXES = (
    {3: URI},
    {True: URI},
    {2.5: URI},
    {("ns",): URI},
    {frozenset({"ns"}): URI},
)


def selected_ids(selector: str, **kwargs: object) -> list[str]:
    root = BeautifulSoup(MARKUP, "html.parser").div
    assert root is not None
    return [element["id"] for element in soupsieve.select(selector, root, **kwargs)]


def test_non_string_prefixes_are_refused() -> None:
    for mapping in BAD_PREFIXES:
        with pytest.raises(TypeError):
            soupsieve.compile("i", namespaces=mapping)


def test_non_string_uris_are_still_refused() -> None:
    with pytest.raises(TypeError):
        soupsieve.compile("i", namespaces={"ns": 7})
    with pytest.raises(TypeError):
        soupsieve.compile("i", namespaces={"ns": None})


def test_well_formed_namespace_maps_still_compile_and_select() -> None:
    assert selected_ids("i", namespaces={"ns": URI}) == ["alpha"]
    assert selected_ids(":is(i, b)", namespaces={"ns": URI, "other": URI}) == ["alpha", "beta"]

    compiled = soupsieve.compile("ns|i", namespaces=[("ns", URI)])
    assert compiled.namespaces == {"ns": URI}
