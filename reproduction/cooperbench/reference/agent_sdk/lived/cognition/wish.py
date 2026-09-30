"""Grounded wish system (Stage C1, spec Part 2) + the §11 keyword wish parser.

A wish is NOT an action — it's a cognitive object emitted by Reflection /
PlanMonitor from a real, evidenced bottleneck, routed to a next module (material
problem-solving / craft). It must be GROUNDED (real evidence + real blocked
reason + a feasible, era-appropriate direction) and written in the FIRST person
(§28). LLM-free: v1 wishes are rule-based.

This module REUSES the existing :class:`~agent_sdk.lived.core.actions.WishCandidate`
(extended with the §7 grounded-wish fields) and keeps the original keyword
:class:`RuleBasedWishParser` scaffold (the ports.WishParserPort impl).
"""
from __future__ import annotations

import re
from typing import Any, List, Optional, Set, Tuple

from agent_sdk.lived.core.actions import WishCandidate
from agent_sdk.lived.core.contracts import Need, NeedType


class WishStatus:
    PROPOSED = "proposed"; GROUNDED = "grounded"; REJECTED = "rejected"
    IN_PROGRESS = "in_progress"; RESOLVED = "resolved"; OBSOLETE = "obsolete"


NEED_TYPES: Tuple[str, ...] = ("carrying_capacity", "storage_capacity", "material_knowledge")

# §8 first-person wish wording per need type (rule-based v1).
_FIRST_PERSON = {
    "carrying_capacity": "我手里已经拿满了，但这里还有谷物。冬天快到了，我需要一种办法一次带回更多谷物。",
    "storage_capacity": "我已经能把一些谷物带回家，但我没有足够好的地方保存它们。我需要更可靠的储藏方式。",
    "material_knowledge": "我有芦苇，但它们散开了。我需要知道什么东西能把它们绑牢。",
}
_PROBLEM = {
    "carrying_capacity": "carrying capacity full while food remains",
    "storage_capacity": "home storage insufficient for the food I keep bringing back",
    "material_knowledge": "I have reed but cannot bind it together",
}
_NEXT_MODULE = {
    "carrying_capacity": "material_problem_solving",
    "storage_capacity": "material_problem_solving",
    "material_knowledge": "craft_session",
}
_DIRECTION = {
    "carrying_capacity": "weave a container from reed bound with fiber",
    "storage_capacity": "build a sturdier storage spot",
    "material_knowledge": "find a binding material for reed",
}

# §9 forbidden (cross-era / over-modern / outcome-not-affordance) wish patterns.
_FORBIDDEN = [re.compile(p, re.I) for p in (
    r"农业|agriculture|farming\b", r"政府|government|\bstate\b", r"市场|market|economy|currency|货币",
    r"金属|metal|iron|bronze|steel", r"完美|perfect\b", r"机器|machine|engine|electric",
    r"枪|火药|\bgun\b|gunpowder")]


def make_wish(*, need_type: str, agent_id: str, agent_name: str, tick: int,
              evidence_ids: List[str], blocked_reasons: List[str],
              target_resource_type: Optional[str] = None, related_plan_id: Optional[str] = None,
              first_person: Optional[str] = None, confidence: float = 0.6) -> WishCandidate:
    """Build a first-person grounded WishCandidate (NOT an action)."""
    return WishCandidate(
        wish_id=f"wish:{agent_id}:{need_type}:{tick}", agent_id=agent_id, turn_id=tick,
        need_type=need_type, problem_summary=_PROBLEM.get(need_type, need_type),
        evidence_ids=list(evidence_ids), confidence=confidence,
        next_module=_NEXT_MODULE.get(need_type, "material_problem_solving"),
        status=WishStatus.PROPOSED, agent_name=agent_name, created_tick=tick,
        first_person_summary=first_person or _FIRST_PERSON.get(need_type, "我遇到了一个瓶颈，我需要想办法解决。"),
        related_plan_id=related_plan_id,
        target_resource_type=target_resource_type or ("reed" if need_type == "carrying_capacity" else None),
        candidate_direction=_DIRECTION.get(need_type, ""), blocked_reasons=list(blocked_reasons))


class WishGroundingValidator:
    """§9 — a wish must be grounded: real evidence, real blocked reason, relevant
    need, feasible next module, era-appropriate, and not an outcome."""

    def __init__(self, available_modules: Optional[Set[str]] = None) -> None:
        self.available_modules = available_modules or {"material_problem_solving", "craft_session"}

    def validate(self, wish: WishCandidate, *, visible_ids: Optional[Set[str]] = None,
                 memory_ids: Optional[Set[str]] = None,
                 real_blocked_reasons: Optional[Set[str]] = None) -> Tuple[bool, List[str]]:
        visible_ids = visible_ids or set()
        memory_ids = memory_ids or set()
        problems: List[str] = []
        if not wish.evidence_ids:
            problems.append("no evidence_ids")
        elif not any(e in visible_ids or e in memory_ids for e in wish.evidence_ids):
            problems.append("evidence not visible/in memory")
        if real_blocked_reasons is not None and wish.blocked_reasons:
            if not (set(wish.blocked_reasons) & real_blocked_reasons):
                problems.append("blocked reason not real")
        elif not wish.blocked_reasons:
            problems.append("no blocked reason")
        if wish.need_type not in NEED_TYPES:
            problems.append(f"unknown need_type {wish.need_type!r}")
        if wish.next_module not in self.available_modules:
            problems.append(f"next_module {wish.next_module!r} unavailable")
        for pat in _FORBIDDEN:
            if pat.search(wish.first_person_summary or "") or pat.search(wish.candidate_direction or ""):
                problems.append(f"forbidden/over-modern wish: {pat.pattern}")
                break
        grounded = not problems
        wish.status = WishStatus.GROUNDED if grounded else WishStatus.REJECTED
        wish.validator_result = "grounded" if grounded else "rejected"
        if not grounded:
            wish.rejected_reason = "; ".join(problems)
        return grounded, problems


# --------------------------------------------------------------------------- #
# §11 keyword wish parser scaffold (ports.WishParserPort) — preserved.
# --------------------------------------------------------------------------- #
_KEYWORDS = {
    NeedType.STORAGE: ("store", "storage", "granary", "stockpile", "储存", "粮仓"),
    NeedType.FAIRNESS: ("unfair", "hoard", "ration", "share equally", "公平", "配给"),
    NeedType.TEACHING: ("teach", "learn", "apprentice", "show how", "教", "学"),
    NeedType.TRADE: ("trade", "barter", "exchange", "deal", "交易", "交换"),
    NeedType.DEFENSE: ("defend", "threat", "attack", "protect", "防御", "威胁"),
    NeedType.LEADERSHIP: ("lead", "leader", "council", "in charge", "领导"),
    NeedType.RECORDING: ("record", "ledger", "write down", "archive", "记录"),
    NeedType.PUNISHMENT: ("punish", "penalty", "sanction", "惩罚"),
    NeedType.COORDINATION: ("coordinate", "together", "organize", "协作", "组织"),
}


class RuleBasedWishParser:
    """Keyword-matching wish parser (scaffold; satisfies WishParserPort)."""

    def parse(self, *, agent_id: str, thought: str, state: Any = None) -> List[Need]:
        if not thought:
            return []
        low = thought.lower()
        needs: List[Need] = []
        for ntype, kws in _KEYWORDS.items():
            if any(kw in low for kw in kws):
                needs.append(Need(need_type=ntype, motivation=thought[:200],
                                  target_object="", expected_effect="", urgency=0.5))
        return needs
