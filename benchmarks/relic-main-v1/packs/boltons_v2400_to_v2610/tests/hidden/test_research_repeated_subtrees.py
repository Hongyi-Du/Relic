"""Withheld contract: searching nested data reports every location a match occurs at.

Only the public search and path-lookup helpers are used, so any traversal that reaches
all of the tree satisfies the contract.
"""

from __future__ import annotations

from boltons.iterutils import get_path, research


def test_a_repeated_subtree_is_reported_at_each_location() -> None:
    shared = {"code": "E42", "retries": 3}
    tree = {"primary": shared, "backup": shared, "other": {"code": "E7"}}

    found = sorted(research(tree, query=lambda path, key, value: key == "code"))

    assert found == [
        (("backup", "code"), "E42"),
        (("other", "code"), "E7"),
        (("primary", "code"), "E42"),
    ]


def test_repeated_sequences_are_reported_too_and_their_paths_resolve() -> None:
    shared = ["hit"]
    tree = {"left": shared, "right": shared}

    found = sorted(research(tree, query=lambda path, key, value: value == "hit"))

    assert found == [(("left", 0), "hit"), (("right", 0), "hit")]
    for path, value in found:
        assert get_path(tree, path) == value


def test_searching_a_tree_without_repetition_is_unchanged() -> None:
    tree = {"a": {"code": 1}, "b": {"code": 2}, "c": [3, 4]}

    found = sorted(research(tree, query=lambda path, key, value: key == "code"))

    assert found == [(("a", "code"), 1), (("b", "code"), 2)]
    assert research(tree, query=lambda path, key, value: value == 4) == [(("c", 1), 4)]
