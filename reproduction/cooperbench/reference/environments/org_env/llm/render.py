"""Salience-aware context rendering for LLM user prompts.

Replaces the crude ``json.dumps(context)[:6000]`` with an ordered, labeled section
layout so the most decision-relevant context comes first and isn't truncated away.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

DEFAULT_BUDGET = 1500
# The action menu defines what the agent may do, so it is never allowed to lose
# entries to a length budget the way descriptive context can.
UNBOUNDED = 0


def _bounded_lines(lines: Sequence[str], budget: int) -> str:
    """Drop whole entries rather than cutting one in half.

    Slicing the joined text at a character budget used to hand the model half an
    object - ``{"candidate_action": "edit_rep`` - which is both unusable and
    silent. A budget that has to bite must bite between entries and say so.
    """
    if budget <= 0:
        return "\n".join(lines)
    kept: list[str] = []
    used = 0
    for line in lines:
        if kept and used + len(line) + 1 > budget:
            kept.append(f"- ... {len(lines) - len(kept)} more omitted for length")
            break
        kept.append(line)
        used += len(line) + 1
    return "\n".join(kept)


def _val(v: Any, budget: int = DEFAULT_BUDGET) -> str:
    if isinstance(v, str):
        return v
    if isinstance(v, (list, tuple)):
        if not v:
            return "(none)"
        return _bounded_lines([f"- {_inline(x)}" for x in v], budget)
    if isinstance(v, dict):
        if not v:
            return "(none)"
        return _bounded_lines(
            [f"- {k}: {_inline(val)}" for k, val in v.items()], budget
        )
    return str(v)


def _inline(x: Any) -> str:
    """Render one entry whole.

    A flat 300-character cut here reintroduced exactly what ``_bounded_lines``
    exists to prevent: it landed inside a JSON object, and the entry that
    overflowed was usually a menu row carrying its edit goal. Sections that must
    stay short do so through their own budget, which drops whole entries.
    """
    if isinstance(x, (dict, list)):
        try:
            return json.dumps(x, ensure_ascii=False, default=str)
        except Exception:
            return str(x)
    return str(x)


def render_sections(
    sections: List[Tuple[str, Any]],
    budgets: Optional[Mapping[str, int]] = None,
) -> str:
    """sections: ordered list of (title, value). Renders labeled blocks."""
    limits = dict(budgets or {})
    out = []
    for title, value in sections:
        out.append(f"## {title}\n{_val(value, limits.get(title, DEFAULT_BUDGET))}")
    return "\n\n".join(out)


def render_context(
    ctx: Dict[str, Any],
    order: List[str],
    budgets: Optional[Mapping[str, int]] = None,
) -> str:
    secs = [(k.replace("_", " ").title(), ctx[k]) for k in order if k in ctx]
    # append any keys not explicitly ordered
    for k, v in ctx.items():
        if k not in order:
            secs.append((k.replace("_", " ").title(), v))
    return render_sections(secs, budgets)


__all__ = ["DEFAULT_BUDGET", "UNBOUNDED", "render_sections", "render_context"]
