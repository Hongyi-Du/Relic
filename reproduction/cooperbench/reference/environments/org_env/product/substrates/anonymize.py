"""Anonymization for OSS time-machine substrates (brief §10).

Minimal, structure-preserving anonymization so the agents work on a *real* runnable product without
being handed obvious identity cues (project name, URLs, commit hashes, release tags). The goal is to
reduce the "the model just recognized project X and recalled its future" attack surface — NOT to
obfuscate the code (runnable structure is preserved).

This is intentionally conservative: by default it only does identifier-level substitution declared in
the manifest (``anonymize`` block) and strips URLs / commit hashes / release tags from agent-visible
TEXT. A real paraphrase hook is left as an extension point.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

# URLs, 7–40 hex commit hashes, and vN.N.N(.N) release tags — identity cues that must not reach
# agent-visible text (issue bodies, README, docs). Code structure is preserved.
_URL_RE = re.compile(r"https?://\S+")
_HASH_RE = re.compile(r"\b[0-9a-f]{7,40}\b")
_TAG_RE = re.compile(r"\bv?\d+\.\d+\.\d+(?:\.\d+)?\b")


def _name_pairs(manifest: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Substitution pairs from the manifest: source project name -> anonymized name, plus any
    explicit ``anonymize.replacements`` mapping (e.g. package/module renames)."""
    pairs: List[Tuple[str, str]] = []
    src = str(manifest.get("source_project_name") or "").strip()
    dst = str(manifest.get("anonymized_product_name") or "").strip()
    if src and dst and src != dst:
        pairs.append((src, dst))
    repl = ((manifest.get("anonymize") or {}).get("replacements")) or {}
    if isinstance(repl, dict):
        for k, v in repl.items():
            if k and v:
                pairs.append((str(k), str(v)))
    # longest source first so substrings don't shadow longer matches
    pairs.sort(key=lambda p: len(p[0]), reverse=True)
    return pairs


def anonymize_text(text: str, manifest: Dict[str, Any], *, strip_identity: bool = True) -> str:
    """Anonymize a piece of agent-visible TEXT (issue body, doc, README)."""
    if not text:
        return text
    out = text
    for src, dst in _name_pairs(manifest):
        out = re.sub(re.escape(src), dst, out, flags=re.IGNORECASE)
    if strip_identity:
        out = _URL_RE.sub("[link removed]", out)
        out = _TAG_RE.sub("<version>", out)
        out = _HASH_RE.sub("<hash>", out)
    return out


def anonymize_code(text: str, manifest: Dict[str, Any]) -> str:
    """Anonymize CODE: only declared identifier substitutions (keep it runnable — no URL/tag/hash
    scrubbing, which could break string literals / imports)."""
    if not text:
        return text
    out = text
    for src, dst in _name_pairs(manifest):
        out = re.sub(r"\b" + re.escape(src) + r"\b", dst, out)
    return out


def anonymization_report(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Audit summary of what anonymization WILL do for this dataset (brief §10/§12)."""
    pairs = _name_pairs(manifest)
    return {
        "project_id": manifest.get("project_id"),
        "name_substitutions": [{"from": s, "to": d} for s, d in pairs],
        "strips": ["urls", "release_tags", "commit_hashes"],
        "paraphrase_hook": "not_enabled",
        "model_cutoff_policy": manifest.get("model_cutoff_policy", "unknown"),
    }


__all__ = ["anonymize_text", "anonymize_code", "anonymization_report"]
