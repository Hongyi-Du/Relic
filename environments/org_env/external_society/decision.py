"""Decision providers for external users.

The society calls a provider to (a) decide whether/what a user posts after an event, and (b)
evaluate the product after experiencing it (-> purchase intent). Two implementations share this
interface:

  * HeuristicDecisionProvider — deterministic, offline, free (default + test + replay fallback).
  * LLMDecisionProvider (stage 4) — fixed model snapshots routed by user capability tier.

The product-evaluation output mirrors society_core's LLMIntentRecord fields so the LinkedIn forum
and the embodied society produce comparable purchase-intent signals.
"""
from __future__ import annotations

from random import Random
from typing import List

from environments.org_env.external_society.agent import (
    ExternalUser,
    PostDecision,
    ProductIntent,
    ProductOffering,
)
from environments.org_env.external_society.event_pool import EventRecord


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _topic_relevance(user: ExternalUser, topic: str) -> float:
    if not topic:
        return 0.35
    if topic in (user.profile.interests or ()):
        return 0.9
    # loose match against needs/working_on text
    blob = f"{user.profile.needs} {user.profile.working_on}".lower()
    return 0.6 if topic.replace("_", " ") in blob else 0.25


def _role_feedback(profile, product: ProductOffering, eff_quality: float) -> str:
    parts = []
    if profile.technical_role in {"professional_engineer", "technical_practitioner"}:
        if eff_quality >= 0.7:
            diagnosis = (
                f"the tested workflow worked for {profile.working_on}; broader reliability still "
                "needs edge-case evidence"
            )
        elif eff_quality >= 0.45:
            diagnosis = (
                f"the tested workflow only partly worked for {profile.working_on}; capture "
                "reproducible failures and traces"
            )
        else:
            diagnosis = (
                f"the tested workflow failed or produced unusable output for {profile.working_on}; "
                "start with a reproducer and execution traces"
            )
        parts.append(f"Technical diagnosis from hands-on evidence: {diagnosis}.")
    if profile.innovation_role in {"creative_originator", "early_builder"}:
        product_label = product.summary or product.product_id
        parts.append(
            f"Feature suggestion for {product_label}: add a focused workflow for "
            f"{profile.needs.rstrip('.')} without claiming support before it is tested."
        )
    return " ".join(parts)


class HeuristicDecisionProvider:
    source = "heuristic"

    # -- posting on an encountered event ---------------------------------------------------------
    def decide_post(
        self, user: ExternalUser, event: EventRecord, recent_posts: List, rng: Random,
        reply_fraction: float = 0.25,
    ) -> PostDecision:
        relevance = _topic_relevance(user, event.topic)
        post_prob = _clamp01(0.25 + 0.5 * user.profile.activity_level + 0.4 * relevance - 0.2)
        if rng.random() > post_prob:
            return PostDecision(will_post=False)
        topic = event.topic or (user.profile.interests[0] if user.profile.interests else "")
        # reply to a recent on-topic post, or post fresh
        on_topic = [p for p in recent_posts if getattr(p, "topic", "") == topic and p.author_id != user.user_id]
        if on_topic and rng.random() < reply_fraction:
            target = on_topic[rng.randrange(len(on_topic))]
            text = f"As a {user.profile.occupation}, re: {topic} — {event.text}."
            return PostDecision(will_post=True, text=text, topic=topic, reply_to=target.post_id)
        verb = {"insight": "Sharing an insight", "world_event": "Noticed", "life_event": "This week I"}.get(
            event.kind, "Thinking about")
        text = f"{verb}: {event.text}. (As a {user.profile.occupation}.)"
        return PostDecision(will_post=True, text=text, topic=topic)

    # -- evaluating the product (purchase intent) -----------------------------------------------
    def evaluate_product(
        self, user: ExternalUser, product: ProductOffering, recent_product_posts: List, rng: Random,
    ) -> ProductIntent:
        p = user.profile
        # peer sentiment from product-board posts the user has seen
        peer = recent_product_posts[-8:]
        peer_pos = sum(1 for q in peer if getattr(q, "likes", 0) >= getattr(q, "reposts", 0)) / max(1, len(peer))
        need_fit = _topic_relevance(user, product.topics[0] if product.topics else "")
        # real quality dominates; fallback-only grounding is penalised (anti spec-gaming, like §P2)
        eff_quality = product.quality * (0.55 if product.fallback_grounded else 1.0)
        satisfaction = _clamp01(
            0.15 + 0.5 * eff_quality + 0.2 * need_fit * p.domain_need_proxy()
            + 0.15 * peer_pos * p.peer_susceptibility_proxy()
            + 0.1 * p.tech_affinity + rng.uniform(-0.06, 0.06))
        intent_to_try = _clamp01(0.2 + 0.4 * need_fit + 0.3 * p.tech_affinity + rng.uniform(-0.05, 0.05))
        wtp = _clamp01(0.05 + 0.5 * satisfaction + 0.2 * p.tech_affinity - 0.25 * p.budget_sensitivity_proxy())
        intent_to_pay = _clamp01(0.1 + 0.7 * satisfaction + 0.2 * wtp - 0.18 * p.budget_sensitivity_proxy())
        recommend = _clamp01(0.05 + 0.5 * satisfaction + 0.2 * p.activity_level)
        converted = intent_to_pay >= 0.5 and wtp >= product.price
        if satisfaction >= 0.6:
            fb, reason = (f"Tried it for {p.working_on} — solid enough to {'pay for' if converted else 'keep using'}.",
                          "positive_value")
        elif satisfaction >= 0.4:
            fb, reason = (f"Promising but not yet worth paying for ({p.needs}).", "cautious_value")
        else:
            fb, reason = (f"Couldn't rely on it for {p.working_on} — churned.", "low_value")
        role_feedback = _role_feedback(p, product, eff_quality)
        if role_feedback:
            fb = f"{fb} {role_feedback}"
        return ProductIntent(
            user_id=user.user_id, product_id=product.product_id, day=0,
            satisfaction=round(satisfaction, 3), intent_to_try=round(intent_to_try, 3),
            intent_to_pay=round(intent_to_pay, 3), willingness_to_pay=round(wtp, 3),
            intent_to_recommend=round(recommend, 3), confidence=round(0.5 + 0.3 * eff_quality, 3),
            converted=converted, feedback=fb, reason_code=reason, source=self.source,
            profile_id=p.user_id, technical_role=p.technical_role,
            innovation_role=p.innovation_role)


__all__ = ["HeuristicDecisionProvider", "_topic_relevance"]
