"""Acceptance check for issue_word_singularization.

Written from the report the engineers were given, not from the fix.
It states in code what the issue states in prose: singularize leaves words ending in a doubled 's' unchanged with their casing preserved and is idempotent, while regular plurals such as 'cats', 'parties' and 'classes' still reduce as before.
"""
from boltons.strutils import singularize


def test_singularize_preserves_double_s_words_and_regular_plurals():
    assert singularize('glass') == 'glass'
    assert singularize('Glass') == 'Glass'
    assert singularize('BOSS') == 'BOSS'

    normalized = singularize('classes')
    assert normalized == 'class'
    assert singularize(normalized) == normalized

    assert singularize('cats') == 'cat'
    assert singularize('parties') == 'party'
