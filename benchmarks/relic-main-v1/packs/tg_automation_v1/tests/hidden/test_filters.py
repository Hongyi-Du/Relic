"""Tests for tg_automation.filters."""

from __future__ import annotations

import pytest

from tg_automation.analysis import UserInteraction
from tg_automation.filters import filter_users, parse_custom_predicate


def _u(uid, msgs=0, reacts=0, days=0):
    return UserInteraction(user_id=uid, username=f"u{uid}",
                           messages_sent=msgs, reactions_given=reacts,
                           active_days=days)


def test_active_mode_keeps_engaged_users():
    users = [_u(1, msgs=5), _u(2, reacts=3), _u(3)]
    out = filter_users(users, mode="active")
    assert {u.user_id for u in out} == {1, 2}


def test_inactive_mode_is_complement():
    users = [_u(1, msgs=5), _u(2, reacts=3), _u(3)]
    out = filter_users(users, mode="inactive")
    assert {u.user_id for u in out} == {3}


def test_custom_predicate_requires_predicate():
    with pytest.raises(ValueError):
        filter_users([_u(1)], mode="custom")


def test_custom_predicate_gte():
    users = [_u(1, msgs=5), _u(2, msgs=4), _u(3, msgs=10)]
    out = filter_users(users, mode="custom", custom=["messages_sent>=5"])
    assert {u.user_id for u in out} == {1, 3}


def test_custom_multiple_predicates_anded():
    users = [_u(1, msgs=5, days=10), _u(2, msgs=6, days=2), _u(3, msgs=10, days=8)]
    out = filter_users(users, mode="custom",
                       custom=["messages_sent>=5", "active_days>=5"])
    assert {u.user_id for u in out} == {1, 3}


def test_custom_reactions_given_predicate():
    users = [_u(1, reacts=0), _u(2, reacts=4), _u(3, reacts=10)]
    out = filter_users(users, mode="custom", custom=["reactions_given>0"])
    assert {u.user_id for u in out} == {2, 3}


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        filter_users([], mode="bogus")


def test_unknown_field_raises():
    with pytest.raises(ValueError):
        parse_custom_predicate("nonsense>=1")


def test_invalid_predicate_syntax_raises():
    with pytest.raises(ValueError):
        parse_custom_predicate("messages_sent !! 5")
