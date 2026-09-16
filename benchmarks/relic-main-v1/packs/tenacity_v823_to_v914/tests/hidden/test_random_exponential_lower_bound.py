"""Withheld contract: the randomised backoff window is bounded below as well as above.

Delays are collected through an injected sleep callable driven by ``tenacity.Retrying``,
so the contract only observes the delays the strategy actually asks for.
"""

from __future__ import annotations

import random

import tenacity

FLOOR = 4.0
CEILING = 8.0
ATTEMPTS = 40


def _collect_delays(wait):
    requested_delays = []

    def always_fails():
        raise ValueError("saturated")

    retrying = tenacity.Retrying(
        wait=wait,
        stop=tenacity.stop_after_attempt(ATTEMPTS),
        sleep=lambda seconds: requested_delays.append(float(seconds)),
        reraise=True,
    )

    try:
        retrying(always_fails)
    except ValueError:
        pass
    else:
        raise AssertionError("expected the final failure to be re-raised")

    return requested_delays


def test_randomised_backoff_never_dips_below_the_floor() -> None:
    random.seed(20240729)

    delays = _collect_delays(
        tenacity.wait_random_exponential(multiplier=0.01, min=FLOOR, max=CEILING)
    )

    assert len(delays) == ATTEMPTS - 1
    assert min(delays) >= FLOOR
    assert max(delays) <= CEILING


def test_randomised_backoff_still_widens_towards_the_ceiling() -> None:
    random.seed(20240730)

    delays = _collect_delays(
        tenacity.wait_random_exponential(multiplier=0.01, min=FLOOR, max=CEILING)
    )

    assert max(delays) > FLOOR


def test_deterministic_backoff_bounds_are_unchanged() -> None:
    delays = _collect_delays(
        tenacity.wait_exponential(multiplier=0.01, min=FLOOR, max=CEILING)
    )

    assert min(delays) == FLOOR
    assert max(delays) == CEILING
    assert delays == sorted(delays)
