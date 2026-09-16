"""Acceptance check for issue_multidict_equality.

Written from the report the engineers were given, not from the fix.
It states in code what the issue states in prose: An OrderedMultiDict compared with a plain mapping is equal only when the values match as well as the keys, with != giving the opposite answer; comparisons against another OrderedMultiDict and against non-mappings are unchanged.
"""
from boltons.dictutils import OrderedMultiDict
from boltons.urlutils import QueryParamDict


def test_multidicts_compare_plain_mapping_values_and_ne():
    for multidict_type in (OrderedMultiDict, QueryParamDict):
        multidict = multidict_type([('a', 'one'), ('b', 'two')])
        matching = {'a': 'one', 'b': 'two'}
        changed = {'a': 'different-one', 'b': 'different-two'}

        assert multidict == matching
        assert not (multidict != matching)
        assert not (multidict == changed)
        assert multidict != changed
