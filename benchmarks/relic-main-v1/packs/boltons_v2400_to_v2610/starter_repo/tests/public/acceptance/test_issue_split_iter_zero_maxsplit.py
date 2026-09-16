"""Acceptance check for issue_split_iter_zero_maxsplit.

Written from the report the engineers were given, not from the fix.
It states in code what the issue states in prose: split and split_iter with maxsplit=0 yield exactly one group holding the input's elements rather than the input object, including for one-shot iterators; larger budgets and the default behave as before.
"""
from boltons.iterutils import split, split_iter


def test_maxsplit_zero_returns_unsplit_values_for_sequences_and_iterators():
    values = [1, None, 2]

    assert split(values, maxsplit=0) == [values]
    assert list(split_iter(values, maxsplit=0)) == [values]
    assert split((value for value in values), maxsplit=0) == [values]
    assert list(split_iter((value for value in values), maxsplit=0)) == [values]
