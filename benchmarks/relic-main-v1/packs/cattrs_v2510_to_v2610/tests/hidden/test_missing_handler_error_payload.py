"""Withheld contract: the "no handler registered" error can cross a process boundary.

Exercised only through the publicly exported error class and the standard library's
``pickle``, so any implementation that keeps the constructor arguments reachable
satisfies the contract.
"""

import pickle

from cattrs.errors import StructureHandlerNotFoundError


def test_the_error_reports_the_values_it_was_built_from() -> None:
    error = StructureHandlerNotFoundError("no hook for dict", dict)

    assert error.args == ("no hook for dict", dict)


def test_the_error_survives_a_pickle_roundtrip() -> None:
    error = StructureHandlerNotFoundError("no hook for frozenset", frozenset)

    revived = pickle.loads(pickle.dumps(error))

    assert isinstance(revived, StructureHandlerNotFoundError)
    assert revived.type_ is frozenset
    assert str(revived) == "no hook for frozenset"


def test_a_pickled_error_can_be_reraised_and_caught() -> None:
    revived = pickle.loads(
        pickle.dumps(StructureHandlerNotFoundError("no hook for complex", complex))
    )

    try:
        raise revived
    except StructureHandlerNotFoundError as caught:
        assert caught.type_ is complex
    else:  # pragma: no cover - the raise above always fires
        raise AssertionError("the revived error was not raisable")


def test_the_message_and_the_type_stay_addressable() -> None:
    error = StructureHandlerNotFoundError("boom", bytes)

    assert str(error) == "boom"
    assert error.type_ is bytes
