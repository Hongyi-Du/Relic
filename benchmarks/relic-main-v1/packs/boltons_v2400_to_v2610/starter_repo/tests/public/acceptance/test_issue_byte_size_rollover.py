"""Acceptance check for issue_byte_size_rollover.

Written from the report the engineers were given, not from the fix.
It states in code what the issue states in prose: bytes2human rolls over to the next unit at each exact power of 1024 (1024 renders as 1K, 1024**2 as 1M, and so on), including for negative counts and with a non-zero digit count, while sub-boundary values are unchanged.
"""
from boltons.strutils import bytes2human


def test_bytes2human_rolls_over_at_exact_unit_boundaries():
    assert bytes2human(1024) == '1K'
    assert bytes2human(1024 ** 2) == '1M'
    assert bytes2human(1024 ** 3) == '1G'


def test_bytes2human_boundary_rollover_preserves_negative_ndigits_and_subboundaries():
    assert bytes2human(-(1024 ** 2), ndigits=2) == '-1.00M'
    assert bytes2human(1023) == '1023B'
    assert bytes2human(2048) == '2K'
