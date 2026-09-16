"""Withheld contract: a byte count rolls over to the next unit at each power of 1024.

Only the public formatting helper is exercised.
"""

from __future__ import annotations

from boltons.strutils import bytes2human


def test_exact_powers_of_1024_use_the_larger_unit() -> None:
    assert bytes2human(1024) == "1K"
    assert bytes2human(1024 ** 2) == "1M"
    assert bytes2human(1024 ** 3) == "1G"
    assert bytes2human(1024 ** 4) == "1T"


def test_rollover_also_applies_to_negatives_and_to_extra_digits() -> None:
    assert bytes2human(-1024) == "-1K"
    assert bytes2human(-1024 ** 3) == "-1G"
    assert bytes2human(1024, 2) == "1.00K"
    assert bytes2human(1024 ** 2, 1) == "1.0M"


def test_counts_below_a_boundary_keep_the_smaller_unit() -> None:
    assert bytes2human(1023) == "1023B"
    assert bytes2human(1024 ** 2 - 1) == "1024K"
    assert bytes2human(2048) == "2K"
    assert bytes2human(0, 2) == "0.00B"
    assert bytes2human(128991) == "126K"
