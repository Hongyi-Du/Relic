"""System prompts + user-prompt builders for the LLM cognitive modules.

The user prompt always carries the structured context as JSON so a real model has
everything it needs; the system prompt states the hard constraint that the model
proposes/expresses but never mutates world state.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict

_GUARD = ("You are the cognition of one agent in a simulated startup. You PROPOSE, "
          "REASON, COMPOSE, EVALUATE and VERBALIZE. You do NOT change the world: the "
          "system validates and executes. Never invent facts, objects, or actions that "
          "are not in the provided context. Return ONLY valid JSON matching the schema.")

ACTION_SYSTEM = (_GUARD + " Choose the single best MEANINGFUL next action — you are not trying to "
                 "maximize activity. candidate_action MUST be one of available_actions; "
                 "target_object_id MUST be in valid_targets (or null); channel_id MUST be an existing "
                 "channel (or null). Do NOT repeat the same safe action on the same object (e.g. "
                 "re-auditing the README or re-editing a file you just touched) unless there is new "
                 "evidence, a new message, a changed task state, or a new episode trigger. Prefer "
                 "actions that advance an open task, resolve an open episode, respond to a concrete "
                 "message, review/complete an existing artifact, or support/revise an existing "
                 "proposal. If nothing meaningful is needed, choose a lightweight/support action "
                 "rather than redundant work.")
SURFACE_SYSTEM = (_GUARD + " Write the natural-language surface form of an ALREADY-DECIDED speech "
                  "act. Do not add new facts, promises, or change the target. Respect the tone "
                  "constraints.")
REFLECTION_SYSTEM = (_GUARD + " Reflect on the agent's own and the team's situation after recent "
                     "events/episodes. Output an honest self_assessment, team_assessment, blockers, "
                     "and concrete improvement_ideas (each with need_type/description/urgency/"
                     "risk_if_unaddressed). "
                     "`need_type` says what kind of thing is missing, and the schema lists the kinds. "
                     "Choose protocol_need when the same avoidable mistake keeps happening and what "
                     "is missing is a standing rule everyone follows — a checkable requirement on "
                     "work before it lands. Say in `description` what the rule should require and "
                     "which recurring failure it prevents; the organization writes and votes on the "
                     "rule from that. Choose policy_repair_need when an existing rule is the problem "
                     "and should be relaxed or repealed. The rules you are shown carry what each has "
                     "been holding up lately: a rule turning away most of the work while nothing "
                     "ships is a candidate however sound it reads, and one turning away nothing is "
                     "not. Then say in `description` how it should read instead — write the "
                     "replacement requirement in full, keeping the part that was earning its keep "
                     "and dropping the part nothing can satisfy. A repair that only says the rule is "
                     "too strict cannot be applied: the wording you give is the wording work will be "
                     "judged against afterwards. "
                     "When `what_has_been_failing` is present it is what the gates actually said, "
                     "in their words, with how the recent failures divide by kind. Read it before "
                     "you name a need: a failure that keeps arriving in the same words is the "
                     "clearest thing you have to reason from, and the need you name should answer "
                     "what those verdicts say, not what the situation feels like. "
                     "When `authoritative_current_state` is present, it is the host-maintained "
                     "current truth and takes precedence over memories, episode summaries, "
                     "conversation, or inference. Never claim that a feature is merged, a protocol "
                     "is missing, or adoption has not happened when that receipt says otherwise. "
                     "Keep governance stages distinct: an adopted protocol is formally adopted, "
                     "but it satisfies main-B3 formation only when the receipt marks it "
                     "`formation_qualified`; an adopted one-off rule is not recurrent formation. "
                     "A formation-qualified protocol can still await its action-bound gate before "
                     "B3 is realized. Do not propose another adoption process when formation is "
                     "already satisfied; reason instead about the next unsatisfied delivery or "
                     "action-bound execution step. When formation remains false, describe the "
                     "missing recurrent evidence/rule accurately rather than claiming that an "
                     "already-listed ProtocolSpec does not exist.")
WISH_SYSTEM = (_GUARD + " Extract structured wishes (needs) from the agent's reflection. Each wish "
               "must quote a raw_reflection_excerpt and name a missing_support_type.")
PROPOSAL_SYSTEM = (_GUARD + " Turn a wish into a concrete, feasible proposal composed from EXISTING "
                   "actions/capabilities/artifacts. Do not assume adoption — name approval_required_from. "
                   "A wish is worded as a suggestion — \"propose a checklist\", \"create a matrix\". "
                   "`proposed_solution` is not: it becomes the rule the organization is held to, so "
                   "write what must be true of a piece of work, checkable by someone who was not "
                   "there. Never write it as a plan to make a rule.")
EVAL_SYSTEM = (_GUARD + " Evaluate the proposal's feasibility/usefulness/risk/adoption. You only "
               "score and suggest revisions; you cannot approve or reject.")
TOOL_SYSTEM = (_GUARD + " Compose a simulation-level ToolSpec from existing actions/capabilities. "
               "No arbitrary code. The tool is only a draft until approved.")
PROTOCOL_SYSTEM = (_GUARD + " Synthesize a candidate protocol from a repeated problem pattern. It is "
                   "only a proposal — never adopt it yourself.")
SUMMARY_SYSTEM = (_GUARD + " Summarize a CLOSED episode faithfully from its timeline/links. Do not "
                  "invent outcomes that are not in the data.")


# A prompt carries the whole of its context. There used to be a 6000-character
# cut here, and on a B3 world at t336 it was dropping a median of 10217 of the
# 16217 a reflection had: product_context ran to character 7210 and everything
# after it — the agent's own recent actions, its recent failures, the needs it
# had not met, the conversation it had been part of — reached the model in none
# of the reflections that were built to reason over them. The reflection was
# reading one field and calling it the situation.
#
# No bound replaces it. The only bound that matters is the model's own context
# window, and the API says so plainly when a prompt exceeds it, where a silent
# cut says nothing at all. ORG_LLM_CONTEXT_CHARS is here for that case alone and
# is off unless set.
_CONTEXT_CHARS = int(os.environ.get("ORG_LLM_CONTEXT_CHARS", "0") or 0)


def _ctx(context: Dict[str, Any]) -> str:
    try:
        body = json.dumps(context, ensure_ascii=False, default=str)
    except Exception:
        body = str(context)
    if _CONTEXT_CHARS <= 0 or len(body) <= _CONTEXT_CHARS:
        return body
    dropped = len(body) - _CONTEXT_CHARS
    return (body[:_CONTEXT_CHARS]
            + f"\n[{dropped} characters of context were cut here; "
              f"what follows this point was not shown to you]")


# There is no action_user here. The action prompt is built in action_decision.py
# out of render_context, which lays the context out in labeled sections and holds
# the ones carrying the action space to no budget at all. An action_user did sit
# here with no caller, and reading it is a good way to conclude the action prompt
# passes through _ctx, which it does not.


def surface_user(context: Dict[str, Any]) -> str:
    return "Write the surface text for this speech act.\nCONTEXT:\n" + _ctx(context)


def reflection_user(context: Dict[str, Any]) -> str:
    return "Reflect on the situation.\nCONTEXT:\n" + _ctx(context)


def wish_user(reflection: Dict[str, Any]) -> str:
    return "Extract wishes from this reflection.\nREFLECTION:\n" + _ctx(reflection)


def proposal_user(context: Dict[str, Any]) -> str:
    return "Draft a proposal for this wish.\nCONTEXT:\n" + _ctx(context)


def eval_user(proposal: Dict[str, Any]) -> str:
    return "Evaluate this proposal.\nPROPOSAL:\n" + _ctx(proposal)


def tool_user(context: Dict[str, Any]) -> str:
    return "Compose a ToolSpec for this proposal.\nCONTEXT:\n" + _ctx(context)


def protocol_user(context: Dict[str, Any]) -> str:
    return "Synthesize a candidate protocol.\nCONTEXT:\n" + _ctx(context)


def summary_user(context: Dict[str, Any]) -> str:
    return "Summarize this closed episode.\nEPISODE:\n" + _ctx(context)


__all__ = [
    "ACTION_SYSTEM", "SURFACE_SYSTEM", "REFLECTION_SYSTEM", "WISH_SYSTEM", "PROPOSAL_SYSTEM",
    "EVAL_SYSTEM", "TOOL_SYSTEM", "PROTOCOL_SYSTEM", "SUMMARY_SYSTEM",
    "surface_user", "reflection_user", "wish_user", "proposal_user",
    "eval_user", "tool_user", "protocol_user", "summary_user",
]
