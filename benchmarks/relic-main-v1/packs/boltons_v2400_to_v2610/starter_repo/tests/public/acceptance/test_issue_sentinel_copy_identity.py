"""Acceptance check for issue_sentinel_copy_identity.

Written from the report the engineers were given, not from the fix.
It states in code what the issue states in prose: Shallow and deep copies of a sentinel — taken directly or nested inside a larger structure — are the identical object, while its falsy truth value, its repr and the fact that separate sentinels are distinct all stay as they are.
"""
import copy

from boltons.typeutils import make_sentinel


def test_sentinel_identity_survives_direct_and_nested_copies():
    sentinel = make_sentinel('NO_VALUE')

    assert copy.copy(sentinel) is sentinel
    assert copy.deepcopy(sentinel) is sentinel

    settings = {'default': sentinel}
    assert copy.copy(settings)['default'] is sentinel
    assert copy.deepcopy(settings)['default'] is sentinel

    assert not sentinel
    assert repr(sentinel) == "Sentinel('NO_VALUE')"
    assert make_sentinel('NO_VALUE') is not sentinel
