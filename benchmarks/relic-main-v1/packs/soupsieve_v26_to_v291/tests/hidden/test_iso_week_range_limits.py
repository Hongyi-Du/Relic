"""Withheld contract: week-numbered range controls agree with how many weeks a year has.

Exercised only through ``soupsieve.select``. A value naming a week the year does not have is
not a value at all, so the control carries no comparable value; a year that really does run
to a fifty-third week must keep comparing normally.
"""

from __future__ import annotations

from bs4 import BeautifulSoup

import soupsieve

# 2002 and 2013 both end inside the following year's first week and have 52 weeks;
# 2015 genuinely runs to a fifty-third week.
MARKUP = """
<div>
  <input id="phantom_week_a" type="week" min="2010-W02" max="2010-W40" value="2002-W53">
  <input id="phantom_week_b" type="week" min="2010-W02" max="2010-W40" value="2013-W53">
  <input id="real_week" type="week" min="2010-W02" max="2010-W40" value="2015-W53">
  <input id="inside" type="week" min="2010-W02" max="2010-W40" value="2010-W20">
  <input id="outside" type="week" min="2010-W02" max="2010-W40" value="2010-W45">
  <input id="valueless" type="week" min="2010-W02" max="2010-W40">
  <input id="date_inside" type="date" min="2010-01-05" max="2010-06-05" value="2010-03-01">
  <input id="date_outside" type="date" min="2010-01-05" max="2010-06-05" value="2011-03-01">
</div>
"""


def selected_ids(selector: str, markup: str = MARKUP) -> list[str]:
    root = BeautifulSoup(markup, "html.parser").div
    assert root is not None
    return [element["id"] for element in soupsieve.select(selector, root)]


def test_a_week_the_year_does_not_have_is_not_a_comparable_value() -> None:
    assert selected_ids(":in-range") == [
        "phantom_week_a",
        "phantom_week_b",
        "inside",
        "valueless",
        "date_inside",
    ]


def test_only_comparable_values_can_fall_outside_the_range() -> None:
    assert selected_ids(":out-of-range") == ["real_week", "outside", "date_outside"]


def test_the_two_range_states_stay_mutually_exclusive() -> None:
    both = set(selected_ids(":in-range")) & set(selected_ids(":out-of-range"))
    assert both == set()


def test_range_states_still_ignore_controls_that_carry_no_range() -> None:
    plain = '<div><input id="free" type="week" value="2010-W20"><i id="text">x</i></div>'
    assert selected_ids(":in-range", plain) == []
    assert selected_ids(":out-of-range", plain) == []
