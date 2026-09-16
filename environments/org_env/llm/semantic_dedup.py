"""Semantic de-duplication helper (v6 P0.1/P0.2/P0.3).

Repeated behaviour in v5 moved from the action layer to the *semantic* layer:
the same README softening patch, three near-identical evidence protocols, and 13
copies of the same customer issue. This module gives the patch/proposal/issue
paths one shared "are these the same thing?" check.

Design (honours "可以用 LLM" without breaking mock/CI determinism):
  * deterministic token-overlap first — works with no client and keeps seeded
    runs reproducible; settles the clear-cut cases (identical text -> dup,
    unrelated text -> distinct);
  * an optional LLM equivalence judgement ONLY for the ambiguous gray zone, and
    only when a client is wired;
  * results cached on the world so a run stays cheap and self-consistent.
"""
from __future__ import annotations

import re
from typing import Any, Optional, Sequence

_WORD = re.compile(r"[a-z0-9_]+")
# generic verbs/articles carry no topical signal — drop them so overlap reflects
# the actual subject matter (fields, checks, claims) rather than boilerplate.
_STOP = {
    "the", "a", "an", "to", "of", "and", "or", "for", "with", "is", "are", "be",
    "that", "this", "it", "its", "on", "in", "by", "as", "at", "we", "our", "us",
    "you", "your", "they", "their", "add", "added", "adds", "adding", "change",
    "changed", "changes", "update", "updated", "updates", "make", "made", "makes",
    "so", "not", "no", "into", "from", "more", "less", "than", "then", "if",
}

# token-overlap thresholds: >= HI -> duplicate; < LO -> distinct; in-between -> LLM
DUP_HI = 0.80
DUP_LO = 0.45


def normalize(text: str) -> str:
    """Lowercased, punctuation-stripped, whitespace-collapsed form for cache keys."""
    return " ".join(_WORD.findall(_as_text(text).lower()))


def _as_text(t: Any) -> str:
    """Coerce any text field to a string. LLM-populated fields (e.g. proposed_solution,
    required_actions) are sometimes lists; flatten them instead of crashing on .lower()."""
    if t is None:
        return ""
    if isinstance(t, (list, tuple, set)):
        return " ".join(_as_text(x) for x in t)
    return t if isinstance(t, str) else str(t)


def tokens(texts: Sequence[Any]) -> set:
    out: set = set()
    for t in texts:
        for w in _WORD.findall(_as_text(t).lower()):
            if len(w) > 2 and w not in _STOP:
                out.add(w)
    return out


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _cache(world: Any) -> dict:
    if world is None:
        return {}
    return world.__dict__.setdefault("_sem_dedup_cache", {})


def equivalent(world: Any, a_texts: Sequence[str], b_texts: Sequence[str], *,
               kind: str, client: Optional[Any] = None,
               lo: float = DUP_LO, hi: float = DUP_HI,
               criterion: Optional[str] = None) -> bool:
    """True if the two text bundles describe the SAME underlying item (same intent
    / effect), not merely a related one. ``kind`` is a short noun used in the LLM
    prompt (e.g. "code change", "customer issue", "governance protocol").

    ``lo``/``hi`` tune the gray zone: overlap >= ``hi`` is a duplicate outright,
    < ``lo`` is distinct outright, and in-between defers to the LLM (when wired).
    Near-identical text (patches) wants a tight zone; varied wording for the same
    intent (protocols/issues) wants a low ``lo`` so the LLM is actually consulted.

    ``criterion`` overrides the default (strict) LLM rule — pass a looser one when a
    domain wants same-goal items merged even if scope/wording differ (e.g. protocols)."""
    ta, tb = tokens(a_texts), tokens(b_texts)
    j = jaccard(ta, tb)
    if j >= hi:
        return True
    if j < lo:
        return False
    client = client if client is not None else getattr(world, "llm_client", None)
    if client is None:
        return False                       # deterministic fallback: don't merge on a guess
    key = ("eq", kind, criterion or "", normalize(" ".join(_as_text(t) for t in a_texts))[:240],
           normalize(" ".join(_as_text(t) for t in b_texts))[:240])
    cache = _cache(world)
    if key in cache:
        return cache[key]
    res = _llm_same(client, a_texts, b_texts, kind, criterion)
    cache[key] = res
    return res


def _llm_same(client: Any, a_texts: Sequence[str], b_texts: Sequence[str], kind: str,
              criterion: Optional[str] = None) -> bool:
    rule = criterion or (
        f"two {kind} descriptions are the SAME underlying item — same intent and effect — "
        f"versus merely related or adjacent; be strict and only answer true when one is "
        f"redundant given the other")
    system = f"You decide whether {rule}. Return JSON only."
    user = ("A:\n" + " | ".join(s for s in (_as_text(t) for t in a_texts) if s)[:600]
            + "\n\nB:\n" + " | ".join(s for s in (_as_text(t) for t in b_texts) if s)[:600]
            + f"\n\nAre A and B the same {kind}? Return {{\"same\": true|false}}.")
    try:
        data = client.generate_json(system, user, {"same": "bool"})
        return bool(isinstance(data, dict) and data.get("same"))
    except Exception:
        return False


__all__ = ["equivalent", "jaccard", "tokens", "normalize", "DUP_HI", "DUP_LO"]
