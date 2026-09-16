"""Bounded owner-local source context over an immutable actor desk.

Retrieval ranks an existing public/actor-visible inventory; it never changes the
physical baseline, publishes code, reads an attachment, or samples peer briefs.
The returned source-byte budget excludes JSON metadata and visibility receipts.
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import PurePosixPath
import re
from typing import Any, Mapping

from .source_reads import actor_source_read_view
from .source_views import SourceViewError
from .visibility import visible_brief_context


SCHEMA = "cooperbench_actor_source_context_v1"
_DEFAULT_LIMITS = {"max_files": 12, "max_bytes": 65_536, "max_file_bytes": 16_384}
_MAX_LIMITS = {"max_files": 384, "max_bytes": 16 * 1024 * 1024, "max_file_bytes": 2 * 1024 * 1024}
_WORDS = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
_STOP_WORDS = frozenset({"the", "and", "for", "with", "this", "that", "from", "into",
                         "must", "should", "implement", "feature", "request", "source",
                         "code", "return", "class", "def", "none", "true", "false"})


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _limits(value: Mapping[str, int] | None) -> dict[str, int]:
    if value is not None and not isinstance(value, Mapping):
        raise ValueError("actor_source_retrieval_limits_invalid")
    if value is not None and set(value).difference(_DEFAULT_LIMITS):
        raise ValueError("actor_source_retrieval_limits_unknown_key")
    result = {**_DEFAULT_LIMITS, **(value or {})}
    for key, number in result.items():
        if type(number) is not int or not 1 <= number <= _MAX_LIMITS[key]:
            raise ValueError("actor_source_retrieval_limit_invalid:" + key)
    return result


@lru_cache(maxsize=8_192)
def _terms(text: str) -> frozenset[str]:
    # This cache is used for bounded prompts, briefs and paths—not repository
    # bodies. Full files are scored by the small-intersection cache below.
    return frozenset(word.casefold() for word in _WORDS.findall(text)).difference(_STOP_WORDS)


@lru_cache(maxsize=65_536)
def _content_overlap_count(text: str, terms: tuple[str, ...]) -> int:
    """Count exact requested tokens without retaining every repo token.

    Cache entries retain only the existing immutable source string reference,
    a small query tuple and an integer. This avoids both re-tokenizing a large
    public baseline every action and materializing millions of token strings.
    """

    wanted = set(terms)
    if not wanted:
        return 0
    found: set[str] = set()
    for match in _WORDS.finditer(text):
        word = match.group(0).casefold()
        if word in wanted:
            found.add(word)
            if len(found) == len(wanted):
                break
    return len(found)


def _score(path: str, content: str, text: str, terms: frozenset[str]) -> int:
    if not text.strip():
        return 0
    folded = text.casefold()
    pure = PurePosixPath(path)
    return (10_000 * int(path.casefold() in folded)
            + 1_000 * int(pure.name.casefold() in folded)
            + 80 * len(_terms(path).intersection(terms))
            + 3 * min(64, _content_overlap_count(content, tuple(sorted(terms)))))


def _visible_briefs(world: Any, actor_id: str):
    context = visible_brief_context(world, actor_id)
    receipts = {feature: row["visibility_receipt"] for feature, row in context.items()}
    text = "\n".join(row["description"] for row in context.values())
    return receipts, text


def retrieve_actor_source_context(world: Any, actor_id: str, query: str = "",
                                  limits: Mapping[str, int] | None = None) -> dict:
    """Return deterministic excerpts using only this actor's visible inputs.

    Query relevance precedes visible-brief relevance, then canonical path/id.
    Zero-score public files remain addressable; this is not an authorization
    filter. ``content`` is a UTF-8-safe prefix and explicitly reports truncation.
    No cache or world state is written. Legacy worlds require a different caller
    path rather than falling back to omniscient mutable artifact contents.
    """
    if not isinstance(query, str) or len(query) > 8_192:
        raise ValueError("actor_source_retrieval_query_invalid")
    bounds = _limits(limits)
    view = actor_source_read_view(world, actor_id)
    if view is None:
        raise SourceViewError("actor_source_retrieval_private_desk_required")
    receipts, brief_text = _visible_briefs(world, actor_id)
    query_terms, brief_terms = _terms(query), _terms(brief_text)
    ranked = []
    # SourceSnapshot.files builds its mapping projection. Obtain it once, not
    # once per file, now that the fixed public inventory can be repository-wide.
    source_files = view.snapshot.files
    for artifact_id, path in view.catalog.items():
        text = source_files[path]
        query_score = _score(path, text, query, query_terms)
        brief_score = _score(path, text, brief_text, brief_terms)
        ranked.append((query_score, brief_score, path, artifact_id, text))
    ranked.sort(key=lambda row: (-row[0], -row[1], row[2], row[3]))
    files, returned_bytes = [], 0
    for query_score, brief_score, path, artifact_id, content in ranked[:bounds["max_files"]]:
        remaining = bounds["max_bytes"] - returned_bytes
        if remaining <= 0:
            break
        raw = content.encode("utf-8")
        excerpt = raw[:min(remaining, bounds["max_file_bytes"])].decode("utf-8", errors="ignore")
        excerpt_bytes = len(excerpt.encode("utf-8"))
        returned_bytes += excerpt_bytes
        files.append({"artifact_id": artifact_id, "path": path, "content": excerpt,
                      "source_sha256": _hash_text(content), "source_bytes": len(raw),
                      "returned_bytes": excerpt_bytes, "truncated": excerpt_bytes != len(raw),
                      "total_lines": len(content.splitlines()),
                      "score": {"query": query_score, "visible_briefs": brief_score}})
    # Callers can be instrumented or run concurrently with world changes. Never
    # publish a result that was selected across two source/permission revisions.
    view.assert_current(world)
    if _visible_briefs(world, actor_id)[0] != receipts:
        raise SourceViewError("actor_source_retrieval_brief_visibility_changed")
    result = {"schema_version": SCHEMA, "actor_id": actor_id, "query": query,
              "limits": bounds, "source_snapshot": view.snapshot.receipt(),
              "brief_visibility": receipts, "files": files,
              "returned_bytes": returned_bytes, "eligible_file_count": len(view.catalog)}
    result["context_digest"] = _hash_text(_canonical(result))
    return result


def assert_actor_source_context_current(world: Any, actor_id: str, context: Mapping[str, Any]) -> None:
    """Reject a saved context whose bytes, actor, source or brief grants changed.

    This is a stale-read check, not an approval or an unforgeable receipt. A
    consumer should call it immediately before using a retained retrieval.
    """
    if not isinstance(context, Mapping):
        raise SourceViewError("actor_source_retrieval_context_invalid")
    payload = dict(context)
    digest = payload.pop("context_digest", None)
    try:
        intact = digest == _hash_text(_canonical(payload))
    except (TypeError, ValueError):
        intact = False
    if not intact or context.get("schema_version") != SCHEMA or context.get("actor_id") != actor_id:
        raise SourceViewError("actor_source_retrieval_context_invalid")
    view = actor_source_read_view(world, actor_id)
    if view is None or view.snapshot.receipt() != context.get("source_snapshot"):
        raise SourceViewError("actor_source_retrieval_source_changed")
    if _visible_briefs(world, actor_id)[0] != context.get("brief_visibility"):
        raise SourceViewError("actor_source_retrieval_brief_visibility_changed")


def read_actor_source_range(world: Any, actor_id: str, request: Mapping[str, Any], *,
                            source_snapshot=None) -> dict:
    """Read a requested line range from the editor's same actor-visible source.

    Paths address the existing catalog, not the host filesystem or another
    member's desk. A too-large range is reported, never silently truncated.
    """
    if not isinstance(request, Mapping):
        raise ValueError("source_read_requires_path_start_line_end_line")
    path = request.get("path")
    start, end = request.get("start_line"), request.get("end_line")
    if (not isinstance(path, str) or type(start) is not int or type(end) is not int
            or start < 1 or end < start):
        raise ValueError("source_read_requires_path_and_1_based_inclusive_line_range")
    view = actor_source_read_view(world, actor_id, source_snapshot=source_snapshot)
    if view is None or path not in view.catalog.values():
        raise ValueError("source_read_path_not_actor_visible")
    lines = view.files[path].splitlines(keepends=True)
    if start > max(1, len(lines)):
        raise ValueError(f"source_read_start_beyond_file:total_lines={len(lines)}")
    text = "".join(lines[start - 1:end])
    if len(text.encode("utf-8")) > _DEFAULT_LIMITS["max_file_bytes"]:
        raise ValueError("source_read_range_too_large:request_fewer_lines;max_bytes="
                         + str(_DEFAULT_LIMITS["max_file_bytes"]))
    view.assert_current(world)
    return {"path": path, "start_line": start, "end_line": min(end, len(lines)),
            "total_lines": len(lines), "content": text,
            "source_snapshot": view.snapshot.receipt()}


def related_public_source_context(world: Any, actor_id: str, paths: list[str]) -> dict:
    """Expose existing callers, imports and public usage tests, not just a file list.

    Ranking uses only the reader's desk and already-visible briefs. Supporting
    excerpts are context, never extra edit targets or immutable test contracts.
    """
    view = actor_source_read_view(world, actor_id)
    if view is None:
        return {}
    anchors = set(paths)
    imports, symbols = [], []
    for path in paths:
        content = view.files.get(path, "")
        imports.extend(line.strip() for line in content.splitlines()
                       if re.match(r"\s*(?:from |import |use |export .*from )", line))
        symbols.extend(re.findall(r"\b(?:class|def|function|func|struct|trait)\s+(\w+)", content))
    # Module stems from import statements make a base class implementation
    # discoverable even when its name never occurs in the feature prose.
    import_paths = [token + ".py" for line in imports
                    for token in re.findall(r"\b[A-Za-z_]\w*\b", line)
                    if token not in {"from", "import", "as", "use"}]
    query = " ".join([*paths, *import_paths, *symbols, *imports])[:8192]
    context = retrieve_actor_source_context(
        world, actor_id, query,
        {"max_files": 18, "max_bytes": 196608, "max_file_bytes": 16384},
    )
    support = [row for row in context["files"] if row["path"] not in anchors
               and (row["score"]["query"] or row["score"]["visible_briefs"])]
    # Include usages as well as implementation neighbors; a helper's direct
    # call alone does not demonstrate that an existing consumer reaches it.
    tests = [row for row in support if re.search(r"(?:^|/)(?:tests?|__tests__)(?:/|$)|(?:test_|_test\.)", row["path"])]
    sources = [row for row in support if row not in tests]
    files = sources[:4] + tests[:2]
    return {"actor_id": actor_id, "source_snapshot": context["source_snapshot"],
            "brief_visibility": context["brief_visibility"], "files": files,
            "scope": "Read-only actor-visible source and public usage examples; not hidden tests. "
                     "Check new inputs through the existing entry point and consuming output, "
                     "not only by calling a newly added helper. Existing tests describe the old "
                     "behavior; an explicit new feature contract may intentionally change it."}


__all__ = ["SCHEMA", "retrieve_actor_source_context", "assert_actor_source_context_current",
           "read_actor_source_range", "related_public_source_context"]
