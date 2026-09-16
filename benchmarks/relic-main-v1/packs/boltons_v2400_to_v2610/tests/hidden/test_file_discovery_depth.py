"""Withheld contract: a directory search can be capped at a number of levels.

Only the public ``iter_find_files`` entry point is exercised, so any traversal that
honours the requested depth satisfies the contract however it is structured.
"""

from __future__ import annotations

import os
import tempfile
from typing import List

from boltons.fileutils import iter_find_files


def build_tree(base: str) -> None:
    os.makedirs(os.path.join(base, "one", "two", "three"))
    for parts in (
        ("root.cfg",),
        ("root.txt",),
        ("one", "level1.cfg"),
        ("one", "two", "level2.cfg"),
        ("one", "two", "three", "level3.cfg"),
    ):
        with open(os.path.join(base, *parts), "w", encoding="utf-8") as handle:
            handle.write("")


def found(base: str, **options: object) -> List[str]:
    matches = iter_find_files(base, "*.cfg", **options)
    return sorted(os.path.relpath(path, base).replace(os.sep, "/") for path in matches)


def test_depth_cap_keeps_only_the_requested_levels() -> None:
    with tempfile.TemporaryDirectory() as base:
        build_tree(base)

        assert found(base, max_depth=0) == ["root.cfg"]
        assert found(base, max_depth=1) == ["one/level1.cfg", "root.cfg"]
        assert found(base, max_depth=2) == [
            "one/level1.cfg",
            "one/two/level2.cfg",
            "root.cfg",
        ]


def test_depth_cap_composes_with_ignore_patterns() -> None:
    with tempfile.TemporaryDirectory() as base:
        build_tree(base)

        assert found(base, ignored="root*", max_depth=1) == ["one/level1.cfg"]


def test_uncapped_search_and_pattern_filtering_are_unchanged() -> None:
    with tempfile.TemporaryDirectory() as base:
        build_tree(base)

        assert found(base) == [
            "one/level1.cfg",
            "one/two/level2.cfg",
            "one/two/three/level3.cfg",
            "root.cfg",
        ]
        assert found(base, ignored="level*") == ["root.cfg"]
