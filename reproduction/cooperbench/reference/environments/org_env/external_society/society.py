"""ExternalSociety — a LinkedIn-like forum of profiled external users (OrgEnv story §6).

Each day, every user has a probability of *encountering a random event* (life event / world event /
insight); only then does the user decide (via the decision provider — heuristic offline, or an LLM
in stage 4) whether/what to post or reply. The whole forum is visible (any floor); the company's
product lives on a separate board. After a release, `run_product_experience()` has every user
experience the product (using their profile + memory + what they've seen) and report feedback +
purchase intent. A controlled event intervention (`activate_event`) skews discussion to a topic and
activates more users for a window. Every random draw is seeded by (society seed, day) for exact
reproducibility.
"""
from __future__ import annotations

from random import Random
from typing import Dict, List, Optional

from environments.org_env.external_society.agent import (
    ExternalUser,
    ForumPost,
    ProductIntent,
    ProductOffering,
)
from environments.org_env.external_society.event_pool import EventPool, build_event_pool
from environments.org_env.external_society.llm_user import make_provider
from environments.org_env.external_society.profiles_dataset import load_external_profiles


class ExternalSociety:
    EVENT_BASE_RATE = 0.12        # base per-user daily probability of encountering an event

    def __init__(self, n: int = 50, seed: int = 42, provider=None,
                 event_pool: Optional[EventPool] = None):
        self.seed = int(seed)
        self.users: List[ExternalUser] = [ExternalUser(p) for p in load_external_profiles(n, seed)]
        self.users_by_id: Dict[str, ExternalUser] = {u.user_id: u for u in self.users}
        self.event_pool = event_pool or build_event_pool()
        # provider: live tier-routed OpenAI snapshots when enabled + configured, else offline
        self.provider = provider or make_provider()
        self.posts: Dict[str, ForumPost] = {}
        self._seq = 0
        self.day = 0
        self._intervention: Optional[dict] = None   # {topic, until_day, boost}

    # -- reproducible per-(day, salt) RNG --------------------------------------------------------
    def _rng(self, salt: int) -> Random:
        return Random((self.seed * 1000003 + self.day * 9176 + salt) & 0xFFFFFFFF)

    def recent_posts(self, board: Optional[str] = None, limit: int = 30) -> List[ForumPost]:
        items = [p for p in self.posts.values() if board is None or p.board == board]
        items.sort(key=lambda p: (p.day, p.post_id))
        return items[-limit:]

    def _add_post(self, author_id: str, text: str, topic: str, *, board: str = "general",
                  reply_to: Optional[str] = None, event_id: Optional[str] = None,
                  about_product: Optional[str] = None) -> ForumPost:
        self._seq += 1
        pid = f"xpost_{self.day}_{self._seq}"
        post = ForumPost(post_id=pid, author_id=author_id, day=self.day, text=text, topic=topic,
                         board=board, parent_id=reply_to, about_event_id=event_id,
                         about_product=about_product)
        self.posts[pid] = post
        u = self.users_by_id.get(author_id)
        if u is not None:
            u.posted_ids.append(pid)
        if reply_to and reply_to in self.posts:
            self.posts[reply_to].reposts += 0   # thread floor recorded via parent_id
        return post

    # -- daily loop: event encounter -> (LLM/heuristic) decision -> post -------------------------
    def daily_tick(self, day: int) -> List[ForumPost]:
        self.day = int(day)
        recent = self.recent_posts(board="general", limit=30)
        intv = self._intervention
        topic = intv["topic"] if (intv and self.day <= intv["until_day"]) else None
        boost = intv["boost"] if topic else 0.0
        new: List[ForumPost] = []
        for i, user in enumerate(self.users):
            urng = self._rng(1000 + i)
            rate = self.EVENT_BASE_RATE * (0.5 + user.profile.activity_level) + boost
            if urng.random() > rate:
                continue
            event = (self.event_pool.sample_topic(urng, topic) if topic
                     else self.event_pool.sample(urng))
            user.remember(f"day{self.day}: {event.text}")
            decision = self.provider.decide_post(user, event, recent, urng)
            if decision.will_post:
                new.append(self._add_post(
                    user.user_id, decision.text, decision.topic, reply_to=decision.reply_to,
                    event_id=event.event_id))
        return new

    def activate_event(self, topic: str, *, day: Optional[int] = None, window: int = 3,
                       boost: float = 0.18) -> None:
        """Controlled intervention: for `window` days, events skew to `topic` and more users are
        activated (higher encounter rate) — short-term the forum is dominated by this event."""
        d = self.day if day is None else int(day)
        self._intervention = {"topic": topic, "until_day": d + window, "boost": boost}

    # -- product experience: everyone evaluates; publication follows activity --------------------
    def run_product_experience(self, product: ProductOffering, day: Optional[int] = None) -> dict:
        if day is not None:
            self.day = int(day)
        recent_prod = self.recent_posts(board="product", limit=20)
        public_posts_by_activity_tier = {
            "very_active": 0,
            "normal": 0,
            "inactive": 0,
        }
        for i, user in enumerate(self.users):
            urng = self._rng(5000 + i)
            intent = self.provider.evaluate_product(user, product, recent_prod, urng)
            intent.day = self.day
            user.product_intent = intent
            user.remember(f"day{self.day}: tried {product.product_id} -> {intent.reason_code}")
            publication_rng = self._rng(8000 + i)
            if publication_rng.random() <= _product_feedback_publication_probability(
                user,
                intent,
            ):
                self._add_post(user.user_id, intent.feedback,
                               product.topics[0] if product.topics else "",
                               board="product", about_product=product.product_id)
                tier = user.profile.activity_tier
                public_posts_by_activity_tier[tier] = (
                    public_posts_by_activity_tier.get(tier, 0) + 1
                )
        summary = self.market_summary(product.product_id)
        public_count = sum(public_posts_by_activity_tier.values())
        summary.update(
            {
                "public_feedback_posts": public_count,
                "silent_experiences": summary["evaluated"] - public_count,
                "public_posts_by_activity_tier": public_posts_by_activity_tier,
            }
        )
        return summary

    def market_summary(self, product_id: Optional[str] = None) -> dict:
        intents: List[ProductIntent] = [
            u.product_intent for u in self.users
            if u.product_intent and (product_id is None or u.product_intent.product_id == product_id)]
        n = len(intents)
        conv = sum(1 for it in intents if it.converted)
        wtp = sum(it.willingness_to_pay for it in intents)
        avg_sat = round(sum(it.satisfaction for it in intents) / n, 3) if n else 0.0
        return {
            "product_id": product_id, "evaluated": n, "conversions": conv,
            "conversion_rate": round(conv / n, 3) if n else 0.0,
            "total_wtp": round(wtp, 2), "avg_satisfaction": avg_sat,
            "recommenders": sum(1 for it in intents if it.intent_to_recommend >= 0.6),
        }


def _product_feedback_publication_probability(
    user: ExternalUser,
    intent: ProductIntent,
) -> float:
    tier_base = {
        "very_active": 0.72,
        "normal": 0.26,
        "inactive": 0.025,
    }.get(user.profile.activity_tier, 0.20)
    polarized_experience = min(1.0, abs(intent.satisfaction - 0.5) * 2.0)
    role_salience = (
        0.05
        if user.profile.technical_role == "professional_engineer"
        or user.profile.innovation_role in {"creative_originator", "early_builder"}
        else 0.0
    )
    return min(
        0.98,
        max(
            0.0,
            tier_base * (0.72 + 0.46 * user.profile.activity_level)
            + 0.10 * polarized_experience
            + role_salience,
        ),
    )


__all__ = ["ExternalSociety"]
