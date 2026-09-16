"""Profile-Conditioned Speech Policy (design doc §15).

Speech is a social action, not narration: promise/accuse/apologize/teach/
propose_rule all mutate the social or institution graph (§15.1). The pipeline
(§15.2): pick intent -> plan content -> pick style -> LLM renders utterance ->
graph update.

This scaffold implements the *structured* stages (intent selection bias from
profile/mood, style derivation) and leaves utterance rendering as a hook for
the LLM. Graph effects are declared so the controller can apply them after the
utterance is produced.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from agent_sdk.lived.core.contracts import (
    MoodState,
    ProfileVector,
    SpeechAct,
    SpeechIntent,
    SpeechStyle,
)


# Which graph edge a speech intent writes (§15.1). Used by the controller to
# turn a produced SpeechAct into a social/institution graph update.
INTENT_GRAPH_EFFECT: Dict[SpeechIntent, str] = {
    SpeechIntent.PROMISE: "promise",
    SpeechIntent.ACCUSE: "accusation",
    SpeechIntent.APOLOGIZE: "trust_repair",
    SpeechIntent.TEACH: "teaching",
    SpeechIntent.THREATEN: "rivalry",
    SpeechIntent.PROPOSE_RULE: "rule_proposal",
    SpeechIntent.RALLY_SUPPORT: "support",
    SpeechIntent.CHALLENGE_LEADER: "power_challenge",
    SpeechIntent.OFFER_HELP: "help_offer",
    SpeechIntent.ASK_FOR_HELP: "help_request",
}


def derive_style(profile: ProfileVector, mood: MoodState) -> SpeechStyle:
    """Profile + mood -> style variables (§15.5/§15.6).

    Long-term profile sets the center; mood applies bounded shifts (warmth up
    with positive valence, politeness down under stress, directness up with
    stress, emotionality up with anger, etc.).
    """
    def clamp(x: float) -> float:
        return max(0.0, min(1.0, x))

    return SpeechStyle(
        directness=clamp(0.4 + 0.4 * profile.dominance + 0.2 * mood.stress),
        warmth=clamp(0.4 + 0.4 * profile.altruism + 0.2 * (mood.valence - 0.5)),
        assertiveness=clamp(0.3 + 0.5 * profile.dominance + 0.2 * mood.confidence),
        politeness=clamp(0.6 + 0.3 * profile.conformity - 0.3 * mood.stress),
        verbosity=clamp(0.5 + 0.2 * (mood.valence - 0.5) - 0.3 * mood.fatigue),
        emotionality=clamp(0.3 + 0.4 * mood.anger + 0.3 * mood.gratitude),
        honesty=clamp(0.7 + 0.3 * profile.fairness - 0.3 * profile.opportunism),
        strategic_ambiguity=clamp(0.3 * profile.opportunism),
        collectivism_language=clamp(0.3 + 0.5 * profile.group_loyalty),
        self_disclosure=clamp(0.3 + 0.3 * mood.gratitude + 0.2 * (mood.valence - 0.5)),
    )


class SpeechPolicy:
    """Builds a structured :class:`SpeechAct` (utterance left for the LLM).

    ``render_fn`` is the optional LLM hook: ``render_fn(speech_act) -> str``.
    When absent the scaffold leaves ``utterance`` empty so the pipeline still
    runs deterministically in tests.
    """

    def __init__(self, render_fn: Optional[Any] = None):
        self.render_fn = render_fn

    def plan(
        self,
        *,
        profile: ProfileVector,
        mood: MoodState,
        intent: SpeechIntent,
        target_uid: Optional[str] = None,
        public: bool = True,
        content_refs: Optional[Dict[str, Any]] = None,
    ) -> SpeechAct:
        style = derive_style(profile, mood)
        act = SpeechAct(
            intent=intent,
            target_uid=target_uid,
            public=public,
            content_refs=dict(content_refs or {}),
            style=style,
            graph_effects={"edge": INTENT_GRAPH_EFFECT.get(intent, "")},
        )
        if target_uid:
            act.graph_effects["target"] = target_uid
        if self.render_fn is not None:
            try:
                act.utterance = str(self.render_fn(act) or "")
            except Exception:
                act.utterance = ""
        return act

    def available_intents(self, *, mood: MoodState) -> List[SpeechIntent]:
        """Mood-gated intent shortlist (§15.6). Returns all intents but ordered
        so the controller / LLM can bias selection — anger surfaces accuse/
        challenge, gratitude surfaces help/apologize."""
        intents = list(SpeechIntent)
        def key(i: SpeechIntent) -> float:
            score = 0.0
            if mood.anger > 0.5 and i in (SpeechIntent.ACCUSE, SpeechIntent.CHALLENGE_LEADER,
                                          SpeechIntent.THREATEN):
                score += mood.anger
            if mood.gratitude > 0.5 and i in (SpeechIntent.OFFER_HELP, SpeechIntent.APOLOGIZE):
                score += mood.gratitude
            if mood.confidence > 0.5 and i in (SpeechIntent.PROPOSE_RULE, SpeechIntent.TEACH,
                                               SpeechIntent.RALLY_SUPPORT):
                score += mood.confidence
            return -score
        intents.sort(key=key)
        return intents
