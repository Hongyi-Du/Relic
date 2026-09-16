"""Policy-grounded text layer (OrgEnv O1.7, spec Part II §17-§26).

Text is a social action, not narration. The pipeline is:

  ObjectAppraisal -> FeedbackDecision (policy gate, §17) -> TextIntent (§19)
  -> SemanticPlan (§20) -> CommunicationStyle (§21-23) -> TextGenerationKernel
  (§24-25, template by default / LLM if injected) -> TextValidator (§26,
  grounding + semantic + style + 1 repair -> template fallback).

The realizer is deterministic-by-default (template) so the loop runs with NO LLM
and tests are reproducible; an injected core ``LLMEngine`` produces JSON-only
surface text instead. Either way the output is grounded + validated, then written
back to the world as messages + commitments/disputes/requests + graph edges
(write-back lives in execution.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from environments.org_env.growth.objects import effective_skill
from environments.org_env.runtime_adapter.comm_style import CommunicationStyle, build_communication_style

# --------------------------------------------------------------------------- #
# DTOs
# --------------------------------------------------------------------------- #
@dataclass
class FeedbackDecision:
    should_feedback: bool
    feedback_act: Optional[str] = None
    audience: Any = None
    target_object_ids: List[str] = field(default_factory=list)
    urgency: str = "normal"
    reason: str = ""
    score: float = 0.0


@dataclass
class TextIntent:
    author_id: str
    text_action_type: str               # message | reply | pr_review | doc_edit | meeting_speech | public_post | protocol_proposal
    speech_act: str
    audience: Any = "team"
    channel_id: Optional[str] = None
    target_object_ids: List[str] = field(default_factory=list)
    attachment_ids: List[str] = field(default_factory=list)
    intent_summary: str = ""
    urgency: str = "normal"
    expected_world_effects: List[str] = field(default_factory=list)


@dataclass
class SemanticPlan:
    required_points: List[str] = field(default_factory=list)
    allowed_claims: List[str] = field(default_factory=list)
    forbidden_claims: List[str] = field(default_factory=list)
    requested_actions: List[dict] = field(default_factory=list)
    commitments_to_create: List[dict] = field(default_factory=list)
    disputes_to_create: List[dict] = field(default_factory=list)
    protocol_events_to_create: List[dict] = field(default_factory=list)
    max_words: int = 60


@dataclass
class TextGenerationOutput:
    surface_text: str = ""
    referenced_objects: List[str] = field(default_factory=list)
    requested_actions: List[dict] = field(default_factory=list)
    commitments_created: List[dict] = field(default_factory=list)
    disputes_created: List[dict] = field(default_factory=list)
    protocol_events_created: List[dict] = field(default_factory=list)
    risk_flags: List[str] = field(default_factory=list)
    style_params_used: Optional[CommunicationStyle] = None
    validation_status: str = "unvalidated"


@dataclass
class ValidationResult:
    ok: bool = True
    problems: List[str] = field(default_factory=list)
    categories: Dict[str, bool] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# §17 feedback / speech gate (policy, NOT raw text or LLM)
# --------------------------------------------------------------------------- #
# persona-preferred feedback acts (spec §18).
PERSONA_FEEDBACK_ACTS = {
    "calvin": ["request_changes", "ask_for_evidence", "request_reproduction", "enforce_protocol"],
    "victor": ["challenge_result", "ask_for_evidence", "propose_protocol", "request_reproduction"],
    "will": ["suggest_rewrite", "warn_about_risk", "challenge_result"],
    "scarlett": ["deescalate", "share_external_signal", "coordinate_followup"],
    "sean": ["promise_work", "send_async_update", "defend_demo_progress"],
    "paul": ["push_team", "commit_to_direction"],
    "iris": ["suggest_rewrite", "warn_about_risk"],
    "skitty": ["share_result", "coordinate_followup"],
}


def feedback_score(agent: Any, appraisal: Any, *, is_assigned_reviewer: bool = False,
                   duplicate: bool = False, after_hours: bool = False,
                   use_profile_conditioning: bool = True) -> float:
    """§17 score: appraisal + persona + state decide whether to give feedback."""
    prof = (agent.profile or {}) if use_profile_conditioning else {}
    skills = agent.skills or {}
    ws = getattr(agent, "work_state", None)
    fatigue = ws.fatigue if ws else float(agent.vitals.get("fatigue", 0.0))
    return (1.2 * appraisal.risk_score
            + 0.9 * appraisal.evidence_gap_score
            + 0.8 * appraisal.protocol_violation_score
            + 0.7 * (1.0 if is_assigned_reviewer else 0.0)
            + 0.6 * effective_skill(skills, "review_quality", 0.3)
            + 0.5 * float(prof.get("distrust_sensitivity", 0.3))
            + 0.4 * float(prof.get("quality_bar", 0.3))
            - 0.7 * fatigue
            - 0.5 * (1.0 if duplicate else 0.0)
            - 0.3 * (1.0 if after_hours else 0.0))


def decide_feedback(agent: Any, appraisal: Any, *, audience: Any = "team",
                    is_assigned_reviewer: bool = False, duplicate: bool = False,
                    after_hours: bool = False, threshold: float = 0.8,
                    use_profile_conditioning: bool = True) -> FeedbackDecision:
    score = feedback_score(agent, appraisal, is_assigned_reviewer=is_assigned_reviewer,
                           duplicate=duplicate, after_hours=after_hours,
                           use_profile_conditioning=use_profile_conditioning)
    if score < threshold or not appraisal.suggested_feedback_acts:
        return FeedbackDecision(should_feedback=False, score=round(score, 3),
                                reason="below_threshold_or_no_issue")
    act = _pick_feedback_act(
        agent,
        appraisal,
        use_profile_conditioning=use_profile_conditioning,
    )
    urgency = "high" if appraisal.risk_level == "high" else "normal"
    return FeedbackDecision(should_feedback=True, feedback_act=act, audience=audience,
                            target_object_ids=[appraisal.target_object_id], urgency=urgency,
                            reason=f"risk={appraisal.risk_level};issues={appraisal.issue_tags}",
                            score=round(score, 3))


def _pick_feedback_act(
    agent,
    appraisal,
    *,
    use_profile_conditioning: bool = True,
) -> str:
    prefs = (
        PERSONA_FEEDBACK_ACTS.get(getattr(agent, "id", ""), [])
        if use_profile_conditioning
        else []
    )
    for a in prefs:
        if a in appraisal.suggested_feedback_acts:
            return a
    return appraisal.suggested_feedback_acts[0]


# --------------------------------------------------------------------------- #
# §19-§20 intent + semantic plan
# --------------------------------------------------------------------------- #
_TEXT_ACTION_FOR = {
    "challenge_result": "message", "request_changes": "pr_review", "ask_for_evidence": "message",
    "request_reproduction": "message", "ask_for_review": "message", "promise_work": "message",
    "warn_about_risk": "message", "suggest_rewrite": "message", "approve_with_note": "pr_review",
    "propose_protocol": "protocol_proposal", "post_company_update": "public_post",
    "send_async_update": "message", "apologize": "message", "deescalate": "message",
    "explain_delay": "message", "push_team": "message", "coordinate_followup": "message",
}


def build_text_intent(*, agent_id: str, speech_act: str, audience: Any = "team",
                      channel_id: Optional[str] = None, target_object_ids: Optional[List[str]] = None,
                      urgency: str = "normal", summary: str = "") -> TextIntent:
    return TextIntent(author_id=agent_id, text_action_type=_TEXT_ACTION_FOR.get(speech_act, "message"),
                      speech_act=speech_act, audience=audience, channel_id=channel_id,
                      target_object_ids=list(target_object_ids or []), urgency=urgency,
                      intent_summary=summary or speech_act.replace("_", " "),
                      expected_world_effects=_EFFECTS_FOR.get(speech_act, []))


_EFFECTS_FOR = {
    "challenge_result": ["create_claim_dispute"], "promise_work": ["create_commitment"],
    "ask_for_review": ["create_requested_action"], "request_reproduction": ["create_requested_action"],
    "ask_for_evidence": ["create_requested_action"], "request_changes": ["create_requested_action"],
    "propose_protocol": ["create_protocol_proposal"],
}


def build_semantic_plan(intent: TextIntent, appraisal: Any = None, *, max_words: int = 60,
                        due_tick: Optional[int] = None) -> SemanticPlan:
    sa = intent.speech_act
    plan = SemanticPlan(max_words=max_words, allowed_claims=list(intent.target_object_ids))
    plan.forbidden_claims = ["fabricated metrics", "unverified success", "promises beyond scope"]
    if appraisal is not None:
        plan.required_points.extend(appraisal.issue_tags[:3])
        if appraisal.evidence_gaps:
            plan.required_points.append("missing: " + ", ".join(appraisal.evidence_gaps[:3]))
    if sa == "challenge_result":
        plan.required_points.append("state the doubt about the result")
        plan.disputes_to_create.append({"target_object_id": _first(intent.target_object_ids),
                                        "issue_tags": (appraisal.issue_tags if appraisal else [])})
        plan.requested_actions.append({"action": "reproduction", "object": _first(intent.target_object_ids)})
    elif sa == "promise_work":
        plan.required_points.append("commit to specific work + a deadline")
        plan.commitments_to_create.append({"description": intent.intent_summary,
                                           "due_tick": due_tick, "object": _first(intent.target_object_ids)})
    elif sa in ("ask_for_review", "request_changes", "ask_for_evidence", "request_reproduction"):
        plan.required_points.append("ask for review/evidence on the object")
        plan.requested_actions.append({"action": sa.replace("ask_for_", "").replace("request_", ""),
                                       "object": _first(intent.target_object_ids)})
    elif sa == "warn_about_risk":
        plan.required_points.append("flag the risk clearly")
    elif sa == "propose_protocol":
        plan.required_points.append("propose a lightweight shared rule")
        plan.protocol_events_to_create.append({"type": "proposal", "rule": intent.intent_summary})
    elif sa in ("apologize", "deescalate", "explain_delay"):
        plan.required_points.append("acknowledge + propose a path forward")
    else:
        plan.required_points.append(intent.intent_summary)
    return plan


def _first(xs: List[str]) -> Optional[str]:
    return xs[0] if xs else None


# --------------------------------------------------------------------------- #
# §24-§25 generation kernel (template default / LLM optional)
# --------------------------------------------------------------------------- #
class TextGenerationKernel:
    def __init__(self, engine: Any = None):
        self.engine = engine

    def generate(self, intent: TextIntent, plan: SemanticPlan, style: CommunicationStyle,
                 context: Any = None) -> TextGenerationOutput:
        if self.engine is not None:
            out = self._llm_generate(intent, plan, style, context)
            if out is not None:
                return out
        return self._template_generate(intent, plan, style)

    # -- deterministic template realizer (no LLM) --------------------------
    def _template_generate(self, intent, plan, style) -> TextGenerationOutput:
        obj = _first(intent.target_object_ids) or "this"
        sa = intent.speech_act
        terse = style.verbosity == "low"
        ev_line = " Please share the config, seed and reproduction." if style.evidence_demand == "high" else ""
        req = "Could you" if style.directness != "high" else "Please"
        warm = "Thanks for the work so far. " if style.warmth == "high" else ""

        if sa == "challenge_result":
            text = f"{warm}I'm not convinced {obj} holds up ({_join(plan.required_points)}).{ev_line}"
        elif sa == "promise_work":
            text = f"I'll take {obj} and get it done" + (
                f" by tick {plan.commitments_to_create[0].get('due_tick')}." if plan.commitments_to_create
                and plan.commitments_to_create[0].get("due_tick") is not None else " soon.")
        elif sa in ("ask_for_review", "request_changes"):
            text = f"{req} review {obj}?{ev_line}"
        elif sa in ("ask_for_evidence", "request_reproduction"):
            text = f"{req} share evidence/reproduction for {obj}?{ev_line}"
        elif sa == "warn_about_risk":
            text = f"Heads up: there's a risk with {obj} ({_join(plan.required_points)})."
        elif sa == "suggest_rewrite":
            text = f"{warm}{req} tighten the wording in {obj}? A couple of claims read too strong."
        elif sa == "propose_protocol":
            text = f"Proposal: let's make this a shared rule — {intent.intent_summary}."
        elif sa in ("apologize", "explain_delay", "deescalate"):
            text = f"{warm}Sorry for the delay on {obj}; here's the plan to get back on track."
        elif sa == "post_company_update":
            text = f"Update: {intent.intent_summary}."
        else:
            text = f"{warm}{intent.intent_summary}."
        if terse:
            text = text.split(". ")[0].rstrip(".") + ("." if not text.endswith("?") else "")
        return TextGenerationOutput(
            surface_text=text.strip(), referenced_objects=list(intent.target_object_ids),
            requested_actions=list(plan.requested_actions), commitments_created=list(plan.commitments_to_create),
            disputes_created=list(plan.disputes_to_create),
            protocol_events_created=list(plan.protocol_events_to_create),
            style_params_used=style, validation_status="template")

    # -- LLM realizer (JSON-only, grounded) --------------------------------
    def _llm_generate(self, intent, plan, style, context) -> Optional[TextGenerationOutput]:
        schema = {"type": "object", "required": ["surface_text"],
                  "properties": {"surface_text": {"type": "string"},
                                 "referenced_objects": {"type": "array", "items": {"type": "string"}},
                                 "risk_flags": {"type": "array", "items": {"type": "string"}}}}
        visible = list(intent.target_object_ids)
        try:
            out = self.engine.call(
                module_name="text_kernel", agent_id=intent.author_id, turn_id=0,
                input_payload={"speech_act": intent.speech_act, "text_action_type": intent.text_action_type,
                               "required_points": plan.required_points, "allowed_claims": plan.allowed_claims,
                               "forbidden_claims": plan.forbidden_claims, "style": style.to_dict(),
                               "style_anchors": style.anchor_lines(),
                               "_evidence": {"public_records": visible}},
                output_schema=schema, prompt_template_id="text_kernel",
                visibility_context={"public_records": visible})
            res = out.get("result", {})
            return TextGenerationOutput(
                surface_text=str(res.get("surface_text", "")).strip(),
                referenced_objects=res.get("referenced_objects", visible),
                requested_actions=list(plan.requested_actions),
                commitments_created=list(plan.commitments_to_create),
                disputes_created=list(plan.disputes_to_create),
                protocol_events_created=list(plan.protocol_events_to_create),
                risk_flags=res.get("risk_flags", []), style_params_used=style, validation_status="llm")
        except Exception:
            return None


def _join(points: List[str]) -> str:
    return "; ".join(p for p in points[:2]) if points else "needs a closer look"


# --------------------------------------------------------------------------- #
# §26 validators (grounding + semantic preservation + style) + repair
# --------------------------------------------------------------------------- #
_EVIDENCE_WORDS = ("evidence", "repro", "reproduction", "config", "seed", "tracker", "metadata")


class TextValidator:
    def validate(self, output: TextGenerationOutput, intent: TextIntent, plan: SemanticPlan,
                 style: CommunicationStyle, context: Any = None) -> ValidationResult:
        vr = ValidationResult()
        visible = set(getattr(context, "visible_object_ids", None) or intent.target_object_ids)
        # 26.1 grounding
        for oid in output.referenced_objects:
            if visible and oid not in visible:
                vr.problems.append(f"references invisible object {oid}")
        low = output.surface_text.lower()
        for fc in plan.forbidden_claims:
            kw = fc.split()[0].lower()
            if kw and kw in low and kw not in ("promises",):
                vr.problems.append(f"forbidden claim: {fc}")
        vr.categories["grounding"] = not vr.problems
        # 26.2 semantic preservation — required actions expressed
        g0 = len(vr.problems)
        if plan.requested_actions and not (("?" in output.surface_text) or any(
                w in low for w in ("review", "share", "please", "could you", "reproduc"))):
            vr.problems.append("requested action not expressed")
        vr.categories["semantic"] = len(vr.problems) == g0
        # 26.3 style
        s0 = len(vr.problems)
        if style.directness == "high" and not any(w in low for w in ("please", "could you", "?", "let's", "i'll")):
            vr.problems.append("directness=high but no concrete request")
        if style.verbosity == "low" and len(output.surface_text.split()) > 30:
            vr.problems.append("verbosity=low but message too long")
        if style.evidence_demand == "high" and intent.speech_act in (
                "challenge_result", "ask_for_evidence", "request_reproduction", "request_changes") \
                and not any(w in low for w in _EVIDENCE_WORDS):
            vr.problems.append("evidence_demand=high but no evidence ask")
        vr.categories["style"] = len(vr.problems) == s0
        vr.ok = not vr.problems
        return vr

    def repair(self, output: TextGenerationOutput, vr: ValidationResult, intent: TextIntent,
               plan: SemanticPlan, style: CommunicationStyle) -> TextGenerationOutput:
        """Single repair pass: drop invisible refs + append missing markers."""
        visible = set(intent.target_object_ids)
        output.referenced_objects = [o for o in output.referenced_objects if not visible or o in visible]
        text = output.surface_text
        if any("evidence" in p for p in vr.problems):
            text += " Please share the config, seed and reproduction."
        if any("concrete request" in p for p in vr.problems):
            text = "Please review: " + text
        output.surface_text = text.strip()
        output.validation_status = "repaired"
        return output

    def template_fallback(self, intent: TextIntent, plan: SemanticPlan,
                          style: CommunicationStyle) -> TextGenerationOutput:
        out = TextGenerationKernel(engine=None)._template_generate(intent, plan, style)
        out.validation_status = "fallback"
        return out


def realize_text(intent: TextIntent, plan: SemanticPlan, style: CommunicationStyle, *,
                 kernel: Optional[TextGenerationKernel] = None,
                 validator: Optional[TextValidator] = None, context: Any = None) -> TextGenerationOutput:
    """Full §26.4 loop: generate -> validate -> 1 repair -> template fallback."""
    kernel = kernel or TextGenerationKernel()
    validator = validator or TextValidator()
    out = kernel.generate(intent, plan, style, context)
    vr = validator.validate(out, intent, plan, style, context)
    if not vr.ok:
        out = validator.repair(out, vr, intent, plan, style)
        vr = validator.validate(out, intent, plan, style, context)
    if not vr.ok:
        out = validator.template_fallback(intent, plan, style)
    else:
        out.validation_status = out.validation_status if out.validation_status in ("repaired",) else "valid"
    return out


__all__ = [
    "FeedbackDecision", "TextIntent", "SemanticPlan", "TextGenerationOutput", "ValidationResult",
    "feedback_score", "decide_feedback", "build_text_intent", "build_semantic_plan",
    "TextGenerationKernel", "TextValidator", "realize_text", "PERSONA_FEEDBACK_ACTS",
    "build_communication_style",
]
