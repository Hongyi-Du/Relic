"""Withheld contract: the "extra keys" error renders the same text every run and can
cross a process boundary.

Exercised through the publicly exported error class, a converter configured to
reject extra keys, and the standard library's ``pickle``.
"""

import pickle

from attrs import define

from cattrs import Converter
from cattrs.errors import ForbiddenExtraKeysError


@define
class Ticket:
    subject: str


def test_the_error_reports_the_values_it_was_built_from() -> None:
    error = ForbiddenExtraKeysError(None, dict, {"beta", "alpha"})

    assert error.args == (None, dict, {"beta", "alpha"})


def test_the_generated_message_is_ordered() -> None:
    error = ForbiddenExtraKeysError(None, dict, {"gamma", "alpha", "delta", "beta"})

    assert str(error) == "Extra fields in constructor for dict: alpha, beta, delta, gamma"


def test_the_error_survives_a_pickle_roundtrip() -> None:
    error = ForbiddenExtraKeysError(None, frozenset, {"yankee", "xray"})

    revived = pickle.loads(pickle.dumps(error))

    assert isinstance(revived, ForbiddenExtraKeysError)
    assert revived.cl is frozenset
    assert revived.extra_fields == {"xray", "yankee"}
    assert str(revived) == str(error)


def test_a_converter_rejection_renders_the_keys_in_order() -> None:
    converter = Converter(forbid_extra_keys=True, detailed_validation=False)

    try:
        converter.structure(
            {"subject": "hello", "zeta": 1, "alpha": 2, "mu": 3}, Ticket
        )
    except ForbiddenExtraKeysError as caught:
        assert str(caught).endswith("alpha, mu, zeta")
        assert caught.extra_fields == {"alpha", "mu", "zeta"}
    else:  # pragma: no cover - extra keys are always rejected here
        raise AssertionError("extra keys were accepted")


def test_an_explicit_message_and_the_attributes_are_unchanged() -> None:
    error = ForbiddenExtraKeysError("bespoke text", dict, {"solo"})

    assert str(error) == "bespoke text"
    assert error.cl is dict
    assert error.extra_fields == {"solo"}
