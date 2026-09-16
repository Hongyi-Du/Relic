"""Withheld contract: sorting a chunk-backed list keeps every element once it has grown.

The container is driven only through the sequence API it advertises, so any internal
chunking scheme that sorts correctly satisfies the contract.
"""

from __future__ import annotations

from boltons.listutils import BarrelList

# Large enough that the container stops fitting in a single backing chunk.
SCRAMBLED = [(index * 37) % 60011 for index in range(60000)]
SENTINEL = -1


def grown() -> BarrelList:
    blist = BarrelList(SCRAMBLED)
    blist.insert(0, SENTINEL)  # the growth that pushes the container past one chunk
    return blist


def test_sorting_a_grown_container_keeps_every_element() -> None:
    blist = grown()
    expected = sorted(SCRAMBLED + [SENTINEL])

    blist.sort()

    assert len(blist) == len(expected)
    assert list(blist) == expected


def test_a_sorted_grown_container_still_indexes_and_slices() -> None:
    blist = grown()
    expected = sorted(SCRAMBLED + [SENTINEL])

    blist.sort()

    assert blist[0] == expected[0]
    assert blist[-1] == expected[-1]
    assert blist[30000] == expected[30000]
    assert list(blist[10:15]) == expected[10:15]


def test_small_containers_sort_exactly_as_before() -> None:
    blist = BarrelList([5, 3, 9, 1])

    blist.sort()

    assert list(blist) == [1, 3, 5, 9]
    assert BarrelList("cbad").pop() == "d"
