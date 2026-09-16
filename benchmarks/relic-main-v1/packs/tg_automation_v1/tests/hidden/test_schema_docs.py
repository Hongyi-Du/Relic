"""Verify documented output schema exists and field names are snake_case."""

from __future__ import annotations

import re
from pathlib import Path

import tg_automation


def _schema_path() -> Path:
    # tg_automation package lives at solution/tg_automation; the schema doc
    # is its sibling under solution/docs/.
    pkg_dir = Path(tg_automation.__file__).resolve().parent
    return pkg_dir.parent / "docs" / "output_schema.md"


def test_output_schema_doc_exists():
    p = _schema_path()
    assert p.is_file(), f"missing schema doc at {p}"


def test_output_schema_doc_lists_expected_fields():
    p = _schema_path()
    text = p.read_text(encoding="utf-8")
    for field in (
        "user_id", "messages_sent", "reactions_given",
        "last_active_at", "active_days",
        "post_id", "author_id", "reactions_by_emoji",
        "new_members", "active_members",
        "join_limit",
    ):
        assert field in text, f"schema doc missing field: {field}"


def test_all_documented_fields_are_snake_case():
    p = _schema_path()
    text = p.read_text(encoding="utf-8")
    fields = re.findall(r"^\|\s*`([a-zA-Z_][\w]*)`\s*\|", text, flags=re.MULTILINE)
    assert fields, "schema doc should document field names in the first column of markdown tables"
    for f in fields:
        assert f == f.strip(), f"field has surrounding whitespace: {f!r}"
        assert re.match(r"^[a-z][a-z0-9_]*$", f), f"non snake_case field: {f}"