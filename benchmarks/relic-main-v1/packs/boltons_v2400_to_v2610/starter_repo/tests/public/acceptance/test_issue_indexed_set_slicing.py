"""Acceptance check for issue_indexed_set_slicing.

Written from the report the engineers were given, not from the fix.
It states in code what the issue states in prose: Slicing an IndexedSet returns the same window as slicing the list of its current contents no matter how many elements were removed beforehand, gives identical results to a freshly built set with the same contents, and still returns an IndexedSet.
"""
from boltons.setutils import IndexedSet


def test_indexed_set_slice_matches_current_contents_after_removals():
    indexed = IndexedSet(range(40))
    for item in range(4):
        indexed.remove(item)

    expected = list(indexed)[10:13]
    sliced = indexed[10:13]
    fresh_sliced = IndexedSet(list(indexed))[10:13]

    assert isinstance(sliced, IndexedSet)
    assert list(sliced) == expected
    assert list(sliced) == list(fresh_sliced)
