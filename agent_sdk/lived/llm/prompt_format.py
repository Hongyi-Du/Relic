"""First-person, named agent prompt formatting (survival upgrade Part 6).

Every LLM-facing surface (self-state, perception, memory, reflection, wish,
speech, episode summary) must describe the agent in the FIRST person ("我……")
and by NAME — never "the agent" / "Alice is…" / "agent_03". Internal systems keep
``agent_id``; the human-readable + LLM-facing layer uses ``agent_name``.

This module is env-agnostic and LLM-free: it formats a body-state snapshot
(:func:`agent_sdk.lived.world.survival.snapshot`) + identity into the first-person
blocks the LLM-assisted modules will consume once they are switched on (the
direct-takeover path is currently LLM-free, so these are exercised by tests and
ready for wiring — Part 6 of the spec).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# §24 — stable demo names by spawn group (no agent_0/agent_1 as identity).
DEMO_NAMES: Dict[str, List[str]] = {
    "Grain Meadow": ["Alice", "Arun"],
    "Reed / Wetland": ["Mira", "Bob"],
    "River": ["Finn", "Lio"],
    "Forest": ["Cara", "Niko"],
    "Stone / Ridge": ["Dave", "Sela"],
    "Herb Patch": ["Hana", "Ivar"],
    "Grassland / Hunting": ["Toma", "Rhea"],
    "Camp Edge": ["Kiran", "Yuna"],
}
DEMO_NAME_LIST: List[str] = [n for names in DEMO_NAMES.values() for n in names]


def assign_demo_names(agents: List[Any]) -> Dict[str, str]:
    """Give each agent a stable human name (§23/§24). Sets ``agent.agent_name``
    (+ ``agent.name`` if present). Returns the id→name map."""
    out: Dict[str, str] = {}
    for i, a in enumerate(agents):
        name = DEMO_NAME_LIST[i % len(DEMO_NAME_LIST)]
        if i >= len(DEMO_NAME_LIST):
            name = f"{name}-{i // len(DEMO_NAME_LIST) + 1}"
        setattr(a, "agent_name", name)
        if hasattr(a, "name"):
            a.name = name
        out[str(getattr(a, "id", i))] = name
    return out


@dataclass
class AgentPromptContext:
    """The named, first-person identity bundle passed to LLM-assisted modules (§23)."""
    agent_id: str
    agent_name: str
    home_group_id: str = ""
    group_name: str = ""
    home_location: Optional[Any] = None
    profile_summary: str = ""
    first_person_identity_prompt: str = ""
    body_state: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Risk summary (§25) — first person
# --------------------------------------------------------------------------- #
def risk_summary(body: Dict[str, Any]) -> str:
    """One-line first-person risk read from a survival snapshot."""
    if body.get("is_collapsed"):
        return "我已经崩溃，现在无法选择普通行动，只能强制休息。"
    parts: List[str] = []
    zone = body.get("fatigue_zone", "normal")
    if zone == "overexertion":
        parts.append("我已进入超负荷区，继续高强度工作可能导致强制休息或崩溃。")
    elif zone == "high":
        parts.append("我有些疲惫，但仍然可以行动。")
    if body.get("is_critically_starving"):
        parts.append("我已极度饥饿，再不进食会损伤健康。")
    elif body.get("is_starving"):
        parts.append("我处于饥饿状态，需要尽快进食。")
    elif body.get("is_hungry"):
        parts.append("我有点饿了。")
    if not parts:
        return "我状态尚可，可以正常行动。"
    return "".join(parts)


def risk_assessment(body: Dict[str, Any]) -> Dict[str, str]:
    """§27 structured risk levels (for the thinking wrapper / validator)."""
    def lvl(x: float, hi: float, mid: float) -> str:
        return "high" if x >= hi else ("medium" if x >= mid else "low")
    return {
        "hunger_risk": lvl(body.get("hunger_pressure", 0.0), 0.7, 0.4),
        "fatigue_risk": {"normal": "low", "high": "medium",
                         "overexertion": "high", "collapse": "high"}.get(
            body.get("fatigue_zone", "normal"), "low"),
        "overexertion_risk": "high" if body.get("fatigue_zone") in ("overexertion", "collapse") else "low",
        "starvation_risk": "high" if body.get("is_critically_starving") else (
            "medium" if body.get("is_starving") else "low"),
        "collapse_risk": "high" if body.get("is_collapsed") else (
            "medium" if body.get("fatigue_zone") == "overexertion" else "low"),
    }


# --------------------------------------------------------------------------- #
# First-person formatter
# --------------------------------------------------------------------------- #
class FirstPersonPromptFormatter:
    """Renders self-state / body-state / memory in the first person, by name."""

    def body_state_block(self, body: Dict[str, Any]) -> str:
        """§25 the BodyState block every LLM-assisted call must receive."""
        e, me = body.get("energy", 0), body.get("max_energy", 100)
        s, ms = body.get("satiety", 0), body.get("max_satiety", 100)
        f = body.get("fatigue", 0)
        hp, mhp = body.get("hp", 0), body.get("max_hp", 10)
        lines = [
            "【我的身体状态】",
            f"我的体力 energy {e:.0f}/{me:.0f}；我的饱腹 satiety {s:.0f}/{ms:.0f}"
            f"（{self._stage_cn(body.get('satiety_stage'))}）。",
            f"我的疲劳 fatigue {f:.0f}（{self._zone_cn(body.get('fatigue_zone'))}，"
            f"软上限 {body.get('fatigue_soft_cap', 100):.0f}/硬上限 {body.get('fatigue_hard_cap', 150):.0f}）；"
            f"我的健康 hp {hp:.0f}/{mhp:.0f}。",
            f"风险：{risk_summary(body)}",
        ]
        return "\n".join(lines)

    def self_state_first_person(self, body: Dict[str, Any]) -> str:
        bits: List[str] = []
        if body.get("is_starving"):
            bits.append("我很饿")
        elif body.get("is_hungry"):
            bits.append("我有点饿")
        if body.get("fatigue_zone") in ("high", "overexertion", "collapse"):
            bits.append("我很疲惫")
        if not bits:
            bits.append("我状态尚可")
        return "，".join(bits) + "。"

    def memory_first_person(self, text: str, agent_name: str) -> str:
        """Rewrite a memory line into the first person for the owning agent."""
        t = re.sub(rf"\b{re.escape(agent_name)}\b", "我", text)
        t = re.sub(r"\b[Tt]he agent\b", "我", t)
        return t

    @staticmethod
    def _stage_cn(stage: Optional[str]) -> str:
        return {"normal": "正常", "hungry": "饿", "starving": "饥饿", "critical": "极度饥饿"}.get(
            stage or "normal", "正常")

    @staticmethod
    def _zone_cn(zone: Optional[str]) -> str:
        return {"normal": "正常", "high": "高疲劳", "overexertion": "超负荷", "collapse": "崩溃"}.get(
            zone or "normal", "正常")


def build_first_person_agent_context(*, agent_id: str, agent_name: str, body: Dict[str, Any],
                                     home_group_id: str = "", group_name: str = "",
                                     home_location: Any = None, profile_summary: str = ""
                                     ) -> AgentPromptContext:
    """Assemble the named, first-person identity + body-state context (§23/§29)."""
    fmt = FirstPersonPromptFormatter()
    identity = (f"我是{agent_name}"
                + (f"，来自{group_name}" if group_name else "")
                + (f"，我的家在 {tuple(home_location)}" if home_location else "")
                + "。")
    return AgentPromptContext(
        agent_id=agent_id, agent_name=agent_name, home_group_id=home_group_id,
        group_name=group_name, home_location=home_location, profile_summary=profile_summary,
        first_person_identity_prompt=identity + "\n" + fmt.body_state_block(body),
        body_state=dict(body),
    )


# --------------------------------------------------------------------------- #
# Validation (§28 + §30)
# --------------------------------------------------------------------------- #
# Third-person self-description patterns forbidden in agent-facing prompts (§30).
_THIRD_PERSON_SELF = [
    re.compile(r"\bthe agent\b", re.I),
    re.compile(r"\bagent[_ ]?\d+\b", re.I),     # raw id as identity
    re.compile(r"\bshe should\b", re.I),
    re.compile(r"\bhe should\b", re.I),
]


def validate_first_person(text: str, agent_name: str = "") -> List[str]:
    """Return third-person self-description violations (§30). Empty == clean.
    References to OTHERS in third person are allowed; only self-as-third is flagged."""
    problems: List[str] = []
    for pat in _THIRD_PERSON_SELF:
        if pat.search(text):
            problems.append(f"third-person self pattern: {pat.pattern}")
    if agent_name:
        if re.search(rf"\b{re.escape(agent_name)}\s+(is|should|was|wants|sees|has)\b", text, re.I):
            problems.append(f"self named in third person: '{agent_name} is/should/...'")
    return problems


def validate_llm_output(output: Dict[str, Any], body: Dict[str, Any]) -> List[str]:
    """§28 LLM-output safety checks. ``output`` is the structured-thinking dict.
    Returns a list of validator errors (empty == safe)."""
    errs: List[str] = []
    proposed = [str(a) for a in (output.get("proposed_actions") or [])]
    high_cost = {"gather_resource", "move_to", "move_to_known_resource", "build", "prototype"}
    # 1. forced rest -> no ordinary high-cost proposals
    if body.get("is_collapsed") and any(a in high_cost for a in proposed):
        errs.append("forced_rest: proposed an ordinary high-cost action")
    # 2/3. high fatigue / low satiety require a risk assessment
    ra = output.get("risk_assessment") or {}
    if body.get("fatigue_zone") in ("high", "overexertion", "collapse") and "fatigue_risk" not in ra:
        errs.append("high fatigue but no fatigue_risk in risk_assessment")
    if (body.get("is_starving") or body.get("is_hungry")) and "starvation_risk" not in ra and "hunger_risk" not in ra:
        errs.append("low satiety but no hunger/starvation risk in risk_assessment")
    # 4. push-through must be marked
    for pc in (output.get("push_through_candidates") or []):
        if not (isinstance(pc, dict) and pc.get("push_through_candidate")
                and "overexertion_risk" in pc and pc.get("risk_acceptance_reason")):
            errs.append("push-through candidate missing overexertion_risk / risk_acceptance_reason")
    # 5. critical state must consider recovery
    if (body.get("is_critically_starving") or body.get("is_collapsed")) and not (
            output.get("recovery_candidates") or output.get("risk_policy")):
        errs.append("critical state ignores recovery options (unsafe/incomplete)")
    # 6/7. no fabricated/written body state
    if output.get("body_state_overrides"):
        errs.append("LLM must not fabricate / override body_state")
    return errs


def structured_thinking_schema() -> Dict[str, Any]:
    """§27 the extra thinking-wrapper fields LLM-assisted modules must emit."""
    return {
        "body_state_used": ["energy", "fatigue", "satiety", "hp", "risk_level"],
        "risk_assessment": ["hunger_risk", "fatigue_risk", "overexertion_risk",
                            "starvation_risk", "collapse_risk"],
        "risk_policy": ["avoid_risk", "accept_risk", "defer_action", "seek_help", "forced_recovery"],
        "push_through_candidate_fields": ["push_through_candidate", "overexertion_risk",
                                          "risk_acceptance_reason"],
    }
