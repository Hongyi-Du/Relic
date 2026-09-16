"""Acceptance check for issue_barrel_list_sort.

Written from the report the engineers were given, not from the fix.
It states in code what the issue states in prose: BarrelList.sort() succeeds no matter how large the container has grown; afterwards it holds exactly the same elements in ascending order and still supports indexing and slicing.
"""
from boltons.listutils import BarrelList


def test_sort_grown_barrel_list_preserves_sorted_elements_and_access():
    values = list(range(50000, 0, -1))
    barrel = BarrelList(values)
    barrel.insert(0, 0)

    barrel.sort()

    expected = sorted([0] + values)
    assert list(barrel) == expected
    assert barrel[0] == 0
    assert barrel[-1] == 50000
    assert list(barrel[100:105]) == expected[100:105]
