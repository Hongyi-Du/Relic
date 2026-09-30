"""Splice anchored replacements into a text, or say exactly what missed.

Shared by the code editor and the document editor. Both previously showed the
model a truncated view of a file and asked for the whole file back, so anything
past the cut was invisible to the model and absent from its answer — an edit to
a long file silently deleted the rest of it. Both also had to fit an answer as
long as the file inside an output ceiling and a 60-second gateway timeout, which
a large file cannot meet. Anchored edits make the answer's size follow the
change rather than the file.
"""
from __future__ import annotations

from typing import Any, List

__all__ = ["apply_anchored_edits"]


def apply_anchored_edits(current: str, edits: Any) -> tuple[str, List[str]]:
    """Apply each `search`/`replace` pair in order. Returns (text, problems).

    An anchor must occur ONCE in the text it is applied to. Zero means the model
    quoted something that is not there — usually its own paraphrase rather than
    the original. More than one means the replacement has no single destination,
    and picking either is how a patcher lands a hunk in the wrong function while
    still producing something that compiles.

    Edits apply in order against the running text, so a later anchor may target
    text an earlier edit introduced. New content is added by anchoring on the
    neighbouring lines and repeating them in the replacement.

    An EMPTY file is the one case with nothing to quote, and create-class
    actions do produce one: create_eval_stub mints an artifact with no content
    and hands it straight to the editor. There an empty anchor means "the file
    is this". Everywhere else an empty anchor is the model omitting the field,
    and appending its answer to the end of a real file is not what it meant.
    """
    if not isinstance(edits, list) or not edits:
        return current, ["no edits were returned"]
    text = current
    problems: List[str] = []
    for index, edit in enumerate(edits, start=1):
        if not isinstance(edit, dict):
            problems.append(f"edit {index}: not an object with search/replace")
            continue
        search = str(edit.get("search") or "")
        replace = str(edit.get("replace") or "")
        if not search:
            if not text.strip():
                text = replace
            else:
                problems.append(
                    f"edit {index}: empty `search` only writes an empty file. This "
                    "file has content, so quote the lines your change goes next to "
                    "and include them in `replace`."
                )
            continue
        found = text.count(search)
        if found == 0:
            first_line = search.splitlines()[0][:90] if search.splitlines() else ""
            problems.append(
                f"edit {index}: `search` does not appear in the file. First line "
                f"of what you sent: {first_line!r}"
            )
        elif found > 1:
            problems.append(
                f"edit {index}: `search` appears {found} times, so there is no "
                "single place to put the replacement. Include more surrounding "
                "lines to make it unique."
            )
        else:
            text = text.replace(search, replace, 1)
    return text, problems
