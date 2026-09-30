"""The callable surface a product publishes, for whoever is writing the code.

A pack can hand the organization a contract document -- the exact class names,
method names and signatures a fixed test suite will import. mini_blobstore ships
one at knowledge/public_api.md, 3751 characters naming every symbol its five
steps are scored on.

Nothing ever showed it to the model writing the code. The code editor's prompt
carried the file's current text, the current build error, the symbols other
modules already import, and the organization's own adopted rules -- and not one
character of the contract. `review_doc` was offered to a single role and only for
the first document it could see, so no engineer had a way to read it either. Over
96 ticks the artifact was touched zero times.

So the names were guessed and then corrected by collision: UploadConflict for
UploadConflictError, ReplicaSet for ReplicaStatus, GCPlan never written at all,
and ReplicaStatus.__init__ taking *args where the contract names object_id and
states. Every arm missed, by 5 names to 17 -- which measures how many times each
happened to run the gate, not what any of them could do.
"""
from __future__ import annotations

import re
from typing import Any, Optional

from environments.org_env.cooperbench.source_reads import (
    actor_source_read_view, visible_source_artifacts,
)

# How a contract document opens a module's section: ### `blobstore.uploads`
_SECTION = re.compile(r"^#{2,4}\s*`([\w.]+)`\s*$", re.M)
# Enough of a document to be a contract rather than a README.
_LOOKS_PUBLISHED = re.compile(r"^-\s*Class\s+`\w+\s*\(", re.M)
_MAX_CHARS = 4000


def _documents(world: Any, view=None):
    for art in visible_source_artifacts(world, view):
        path = str(getattr(art, "linked_file_path", "") or "")
        if path.endswith(".md") and "knowledge/" in path:
            yield art


def module_of(path: str) -> str:
    """`blobstore/uploads.py` -> `blobstore.uploads`."""
    stem = str(path or "").rsplit(".py", 1)[0]
    return stem.replace("/", ".").replace("\\", ".").strip(".")


def section_for(contract: str, module: str) -> str:
    """The part of the contract that governs this module, or "" if it names none."""
    starts = [(m.start(), m.group(1), m.end()) for m in _SECTION.finditer(contract)]
    for i, (_start, named, end) in enumerate(starts):
        if named != module:
            continue
        stop = starts[i + 1][0] if i + 1 < len(starts) else len(contract)
        return contract[end:stop].strip()
    return ""


def published_surface(world: Any, artifact: Any, *, actor_id=None, source_snapshot=None) -> Optional[str]:
    """What the contract says about the file being edited, ready to paste in.

    The module's own section when the document has one, because that is what the
    author has to get exactly right; the whole document when it does not, since a
    contract short enough to read whole is better than none.
    """
    view = actor_source_read_view(world, actor_id, source_snapshot=source_snapshot)
    path = str(getattr(artifact, "linked_file_path", "") or "")
    if not path.endswith(".py"):
        return None
    module = module_of(path)
    for doc in _documents(world, view):
        text = (view.text(doc) if view is not None else
                (getattr(doc, "mainline_content", "") or getattr(doc, "content", "") or ""))
        if not _LOOKS_PUBLISHED.search(text):
            continue
        part = section_for(text, module)
        whole = not part
        body = part or text
        if len(body) > _MAX_CHARS:
            body = body[:_MAX_CHARS] + "\n… (the rest of this section is in "
            body += f"{getattr(doc, 'linked_file_path', '')})"
        which = (f"the whole of {getattr(doc, 'linked_file_path', '')}" if whole
                 else f"the `{module}` section of "
                      f"{getattr(doc, 'linked_file_path', '')}")
        return (f"\nPUBLISHED CONTRACT — {which}. The tests this product is "
                f"scored by import exactly these names with exactly these "
                f"parameters, so a name one letter or one suffix different is a "
                f"failure however well the code behaves:\n{body}\n")
    return None


__all__ = ["published_surface", "section_for", "module_of"]
