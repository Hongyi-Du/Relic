"""Acceptance check for issue_partition_predicates.

Written from the report the engineers were given, not from the fix.
It states in code what the issue states in prose: partition(src, key, *more_keys) returns one list per predicate plus a final list for the unmatched remainder, placing each value in the first predicate that accepts it, while the single-predicate call keeps returning the (truthy, falsy) pair.
"""
from boltons.iterutils import partition


def test_partition_accepts_multiple_predicates_with_first_match_precedence():
    src = [-2, -1, 0, 1, 2, 3]

    buckets = partition(src, lambda value: value % 2 == 0, lambda value: value > 0)

    assert tuple(list(bucket) for bucket in buckets) == ([-2, 0, 2], [1, 3], [-1])
    assert tuple(list(bucket) for bucket in partition(src, lambda value: value > 0)) == (
        [1, 2, 3],
        [-2, -1, 0],
    )
