"""Substrate-neutral behavior-surface predicates shared by planning and review.

Security review gating must key on the *evidence* a task spec carries
(its behavior_surface terms, theme tokens, issue type) rather than on the
identity of a curated theme name or the single literal surface word
"filesystem". Task specs derived from an arbitrary substrate express
security relevance with whatever vocabulary the substrate's issues use
("security", "auth", "traversal", ...), so the gate matches a closed set
of security-flavored tokens instead of one hardcoded spelling.
"""

from __future__ import annotations

import re
from typing import Iterable

SECURITY_SURFACE_TOKENS: frozenset[str] = frozenset(
    {
        "auth",
        "authentication",
        "authorization",
        "credential",
        "credentials",
        "csrf",
        "cve",
        "exploit",
        "filesystem",
        "injection",
        "oauth",
        "permission",
        "permissions",
        "privilege",
        "sandbox",
        "sandboxing",
        "secret",
        "secrets",
        "security",
        "traversal",
        "untrusted",
        "vulnerabilities",
        "vulnerability",
        "xss",
    }
)

_TOKEN_SPLIT = re.compile(r"[^a-z0-9]+")


def has_security_surface(*term_groups: Iterable[str]) -> bool:
    """True when any term in any group carries a security-flavored token.

    Terms are lowercased and split on non-alphanumerics before matching,
    so compound spellings ("path_security", "auth-flow", "Path Traversal")
    match while lookalike words ("author", "authoring") do not.
    """

    for group in term_groups:
        for raw_term in group:
            term = str(raw_term).strip().lower()
            if not term:
                continue
            if any(
                token in SECURITY_SURFACE_TOKENS
                for token in _TOKEN_SPLIT.split(term)
            ):
                return True
    return False
