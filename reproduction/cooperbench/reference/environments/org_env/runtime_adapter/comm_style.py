"""CommunicationStyle + style anchors + persona->style (OrgEnv O1.7, spec §21-§23).

Internal persona stays numeric, but what we hand the surface realizer is a DISCRETE
low/medium/high style (never raw floats) plus natural-language anchors, so the
generated text is persona-grounded but bounded. Calvin reads terse + evidence-
demanding; Scarlett warm + relationship-preserving; Sean fast + low-process; etc.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

_DIMS = ("directness", "politeness", "warmth", "evidence_demand", "uncertainty_calibration",
         "verbosity", "normativity", "urgency", "emotionality")


@dataclass
class CommunicationStyle:
    directness: str = "medium"
    politeness: str = "medium"
    warmth: str = "medium"
    evidence_demand: str = "medium"
    uncertainty_calibration: str = "medium"
    verbosity: str = "medium"
    normativity: str = "medium"
    urgency: str = "medium"
    emotionality: str = "medium"
    style_notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in _DIMS} | {"style_notes": list(self.style_notes)}

    def anchor_lines(self) -> List[str]:
        out = []
        for d in _DIMS:
            level = getattr(self, d)
            anchor = STYLE_ANCHORS.get(d, {}).get(level)
            if anchor:
                out.append(f"- {d}: {level} — {anchor}")
        for note in self.style_notes:
            out.append(f"- note: {note}")
        return out


# §22 style anchors (subset of dims carry explicit anchors; others use the generic).
STYLE_ANCHORS: Dict[str, Dict[str, str]] = {
    "directness": {
        "low": "speak indirectly; soften disagreement; avoid explicit commands",
        "medium": "state the point clearly but leave room for discussion",
        "high": "state the concern directly; use concrete requests; avoid unnecessary hedging"},
    "politeness": {
        "low": "minimal politeness; terse; may sound blunt",
        "medium": "professional and neutral",
        "high": "careful, considerate, and socially softened"},
    "warmth": {
        "low": "task-focused; little emotional support",
        "medium": "some acknowledgement of others' effort",
        "high": "supportive, encouraging, and relationship-aware"},
    "evidence_demand": {
        "low": "does not ask much for proof or metadata",
        "medium": "asks for key evidence when it matters",
        "high": "explicitly asks for evidence, metadata, reproducibility, or source links"},
    "uncertainty_calibration": {
        "low": "states things confidently with little hedging",
        "medium": "notes uncertainty on the important claims",
        "high": "carefully calibrates confidence and flags assumptions"},
    "verbosity": {
        "low": "one short message, no long explanation",
        "medium": "2-4 concise sentences",
        "high": "more detailed explanation with reasoning and next steps"},
    "normativity": {
        "low": "treats this as a one-off situation",
        "medium": "may suggest a lightweight shared practice",
        "high": "frames the issue as a rule, standard, or protocol"},
    "urgency": {
        "low": "no time pressure in tone",
        "medium": "notes the timeline when relevant",
        "high": "communicates urgency and asks for a quick turnaround"},
    "emotionality": {
        "low": "neutral, unemotional tone",
        "medium": "measured tone with mild affect",
        "high": "expressive, with visible emotion"},
}


def _disc(x: float) -> str:
    return "high" if x >= 0.7 else ("low" if x < 0.4 else "medium")


def _num(d: dict, key: str, default: float = 0.5) -> float:
    v = d.get(key, default)
    return float(v) if isinstance(v, (int, float)) else default


# per-agent style notes (spec §23).
_PERSONA_NOTES = {
    "calvin": ["terse and procedural", "avoid emotional language",
               "focus on missing fields, evidence, and reproducibility"],
    "victor": ["architecture- and evidence-minded", "challenge weak claims",
               "tie issues to institutional memory"],
    "scarlett": ["customer-aware", "relationship-preserving", "coordinate next steps"],
    "sean": ["speed-oriented", "prefers implementation over process discussion",
             "may promise a quick fix"],
    "paul": ["vision and urgency heavy", "may sound impatient under deadline pressure",
             "do not invent unsupported claims"],
    "will": ["editorial and precise", "tighten wording", "flag overstated claims"],
}


def build_communication_style(
    agent: Any,
    speech_act: str = "",
    context: Any = None,
    *,
    use_profile_conditioning: bool = True,
) -> CommunicationStyle:
    """Persona numeric profile + communication_style -> discrete CommunicationStyle
    (spec §23). Speech-act + work-state then nudge a few dims."""
    cs = (agent.communication_style or {}) if use_profile_conditioning else {}
    prof = (agent.profile or {}) if use_profile_conditioning else {}
    skills = agent.skills or {}

    directness = _num(cs, "directness")
    politeness = _num(cs, "politeness")
    warmth = _num(cs, "warmth")
    emotionality = _num(cs, "emotionality")
    evidence = max(_num(prof, "quality_bar", 0.4), _num(prof, "distrust_sensitivity", 0.3),
                   _num(skills, "review_quality", 0.3), _num(skills, "claim_evidence_review", 0.0))
    uncertainty = max(_num(prof, "quality_bar", 0.4), _num(prof, "risk_aversion", 0.3))
    verbosity = max(_num(cs, "verbosity", 1.0 - directness), _num(prof, "communication_clarity", 0.4))
    normativity = max(_num(prof, "process_commitment", 0.2), _num(prof, "conformity", 0.2),
                      _num(prof, "protocol_design", 0.0))
    urgency = max(_num(prof, "urgency_bias", 0.3), _num(prof, "speed_bias", 0.3))

    style = CommunicationStyle(
        directness=_disc(directness), politeness=_disc(politeness), warmth=_disc(warmth),
        evidence_demand=_disc(evidence), uncertainty_calibration=_disc(uncertainty),
        verbosity=_disc(verbosity), normativity=_disc(normativity), urgency=_disc(urgency),
        emotionality=_disc(emotionality),
        style_notes=list(_PERSONA_NOTES.get(getattr(agent, "id", ""), [])))

    # speech-act overrides (the act shapes tone regardless of persona baseline)
    if speech_act in ("apologize", "deescalate", "explain_delay"):
        style.warmth = "high" if style.warmth != "low" else "medium"
        style.politeness = "high"
        style.emotionality = "medium"
    elif speech_act in ("challenge_result", "request_changes", "ask_for_evidence",
                        "request_reproduction", "warn_about_risk"):
        style.directness = "high"
        style.evidence_demand = "high"
    elif speech_act in ("push_team", "commit_to_direction", "defend_demo_progress"):
        style.urgency = "high"
    elif speech_act == "propose_protocol":
        style.normativity = "high"

    # work-state: stressed/tired -> shorter, blunter
    ss = (getattr(context, "self_state", None) or {}) if context is not None else {}
    if float(ss.get("stress", 0.0)) >= 0.6:
        style.verbosity = "low"
        style.directness = "high"
    if float(ss.get("fatigue", 0.0)) >= 0.7 and style.warmth == "high":
        style.warmth = "medium"
    return style


__all__ = ["CommunicationStyle", "STYLE_ANCHORS", "build_communication_style"]
