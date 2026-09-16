"""Withheld contract: duplicating a multidict keeps every value stored under every key.

Only the public container plus the standard :mod:`copy` and :mod:`pickle` modules are
used, so any duplication strategy that preserves the contents satisfies the contract.
"""

from __future__ import annotations

import copy
import pickle

from boltons.dictutils import OrderedMultiDict


def sample() -> OrderedMultiDict:
    return OrderedMultiDict([("p", "a"), ("p", "b"), ("q", "c")])


def test_shallow_and_deep_copies_keep_repeated_values() -> None:
    original = sample()

    shallow = copy.copy(original)
    deep = copy.deepcopy(original)

    assert shallow.getlist("p") == ["a", "b"]
    assert deep.getlist("p") == ["a", "b"]
    assert shallow == original
    assert deep == original
    assert list(deep.items(multi=True)) == [("p", "a"), ("p", "b"), ("q", "c")]


def test_a_deep_copy_does_not_share_mutable_values() -> None:
    nested = OrderedMultiDict([("n", ["i"]), ("n", ["j"])])

    duplicated = copy.deepcopy(nested)
    duplicated.getlist("n")[0].append("mutated")

    assert nested.getlist("n") == [["i"], ["j"]]
    assert duplicated.getlist("n") == [["i", "mutated"], ["j"]]


def test_pickling_and_the_built_in_copy_method_are_unchanged() -> None:
    original = sample()

    assert pickle.loads(pickle.dumps(original)).getlist("p") == ["a", "b"]
    assert original.copy().getlist("p") == ["a", "b"]
    assert original.todict(multi=True) == {"p": ["a", "b"], "q": ["c"]}
    assert original["p"] == "b"
