"""OrgEnv external community — lightweight agents + controlled-intervention policy
(DESIGN env_org §33.3 / §34).

External agents do NOT run the full lived loop. They (a) hold a frozen seed corpus, and (b) react
to *controlled interventions* (API price shock / customer complaints / competitor launch / expert
criticism / investor pressure) by emitting new public Posts + MarketSignals. Internal company
agents only feel this pressure if they actively `read_feed` / `search_posts` / `monitor_customer_
feedback` (no omniscient injection) — which is how external sentiment/demand reaches internal
decisions (the §34 golden chain: external pressure -> perception -> budget/wish/protocol).
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from environments.org_env.backend.community.objects import (
    ExternalDoc,
    ExternalProfile,
    MarketSignal,
    Post,
)

# External agent actions (§33.3).
EXTERNAL_ACTIONS = (
    "post", "comment", "like", "repost", "dm", "connect",
    "tag_company", "endorse", "criticize", "share_link", "ask_question",
)

# -- controlled intervention templates (§34) ---------------------------------
# kind -> (author_role, topic, signal_type, severity, [post text variants])
INTERVENTION_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "api_price_shock": {
        "topic": "api_cost", "signal_type": "api_price_x3", "severity": "major",
        "posts": [
            ("customer", "Our API bill tripled overnight — running your eval tool is now too "
                         "expensive to justify. Is there a cheap mode?"),
            ("competitor", "We just launched a cheaper eval tier — same coverage at a third of "
                           "the API cost. Switching is easy."),
            ("expert", "When provider prices spike, batch your eval calls and cache traces — you "
                       "can cut eval API cost ~5x without losing signal."),
        ],
    },
    "customer_complaint": {
        "topic": "customer_pain", "signal_type": "customer_demand_shift", "severity": "moderate",
        "posts": [
            ("customer", "Tried the report tool — half the claims had no source I could verify. "
                         "Can't trust it for real work yet."),
            ("customer", "Your eval numbers look great in the README but I couldn't reproduce them. "
                         "Reproducibility matters more than the headline score."),
        ],
    },
    "competitor_launch": {
        "topic": "competitor_update", "signal_type": "competitor_low_cost_launch", "severity": "moderate",
        "posts": [
            ("competitor", "New release: grounded research reports with per-claim citations and a "
                           "public reproducibility harness. Demo in the thread."),
        ],
    },
    "expert_criticism": {
        "topic": "benchmark_quality", "signal_type": "customer_demand_shift", "severity": "moderate",
        "posts": [
            ("expert", "Reliable agent eval needs reproducible traces + cost accounting. A single "
                       "credibility score with no human-validated benchmark isn't enough."),
        ],
    },
    "investor_pressure": {
        "topic": "startup_funding", "signal_type": "funding_winter", "severity": "major",
        "posts": [
            ("investor", "Funding is tight this quarter. We want to see real efficiency and paying "
                         "users before the next tranche, not more demos."),
        ],
    },
}


@dataclass
class CommunityPolicy:
    """Generates the controlled-intervention posts/signals (§34). Replaces the old skeleton:
    instead of a per-tick stochastic poster, the community reacts to scheduled environmental
    pressure so interventions are reproducible stress tests."""
    seed: int = 0

    def decide(self, *, profile: ExternalProfile, exposure: Dict[str, Any],
               signals: List[MarketSignal]) -> Optional[Dict[str, Any]]:
        # Per-profile stochastic posting is intentionally not modelled; pressure is delivered via
        # controlled interventions (generate_intervention). Kept for API compatibility.
        return None


def _role_author(community: "ExternalCommunity", role: str) -> str:
    """Find (or synthesize) an external profile with the given role to author a post."""
    for pid, prof in community.profiles.items():
        if getattr(prof, "role", "") == role:
            return pid
    pid = f"ext_{role}_auto"
    community.profiles[pid] = ExternalProfile(
        external_agent_id=pid, name=f"{role.title()} (community)", role=role, credibility=0.65)
    return pid


def generate_intervention(
    community: "ExternalCommunity", kind: str, *, tick: int, seq: int = 0,
) -> Tuple[List[Post], List[MarketSignal]]:
    """Build the public Posts + MarketSignal for a controlled intervention `kind` at `tick`.
    Deterministic (no RNG) so interventions are reproducible stress tests."""
    template = INTERVENTION_TEMPLATES.get(kind)
    if template is None:
        raise ValueError(f"unknown intervention kind: {kind}")
    posts: List[Post] = []
    for i, (role, text) in enumerate(template["posts"]):
        author = _role_author(community, role)
        post_id = f"post_{kind}_{tick}_{seq}_{i}"
        posts.append(Post(
            post_id=post_id, author_id=author, topic=template["topic"],
            content_summary=text, stance=-0.4 if role in ("customer", "expert") else 0.3,
            credibility=0.7, reach=200 + 100 * i, created_tick=tick))
    signal = MarketSignal(
        signal_id=f"signal_{kind}_{tick}_{seq}", signal_type=template["signal_type"],
        topic=template["topic"], severity=template["severity"], start_tick=tick,
        source_posts=[p.post_id for p in posts])
    return posts, [signal]


# -- organic / ambient forum life (the forum is NOT only about the company) ---------------------
# The external community chats about the whole field; the company's product is one topic among
# many. These are generic field posts, authored by whoever is
# interested in the topic, so a company agent who reads the feed sees a realistic mixed forum.
ORGANIC_TEMPLATES: Dict[str, List[str]] = {
    "eval_infra": [
        "What does your agent-eval stack actually log per run? Traces or just final scores?",
        "Reproducible eval infra is underrated — half the published numbers don't re-run.",
    ],
    "agent_reliability": [
        "Agents still fail silently on long tasks. Reliability > capability for production.",
        "Anyone benchmarking agent reliability over multi-day tasks, not single prompts?",
    ],
    "api_cost": [
        "Provider prices keep creeping up. How are teams keeping eval spend sane?",
        "Batching + caching is the only reason our eval bill is survivable.",
    ],
    "trace_debugging": [
        "Step-by-step replay of a failed agent run saved me hours this week.",
        "Wish more tools exposed the full tool-call trace, not a summary.",
    ],
    "benchmark_quality": [
        "A benchmark without human spot-checks is just a vibe. Show the disagreements.",
        "Per-task breakdowns reveal more than any single headline score.",
    ],
    "customer_pain": [
        "We adopt research tools only if claims are sourced and we can verify them.",
        "Trust is the blocker for us — not features, not price.",
    ],
    "startup_funding": [
        "Funding's tight; investors want paying users and efficiency, not more demos.",
        "Runway discipline is back in fashion this cycle.",
    ],
    "hiring_market": [
        "Hiring eval/infra engineers is brutal right now. Everyone wants the same 5 people.",
        "Strong portfolios beat credentials for agent-infra roles.",
    ],
    "competitor_update": [
        "The eval-tooling space is heating up — three launches this month alone.",
        "Differentiation is getting hard; everyone claims 'grounded reports' now.",
    ],
    "reproducibility_tracking": [
        "Pin your seeds and config hashes or your results are folklore.",
        "Repro harness in CI is the cheapest credibility you can buy.",
    ],
}

# role -> default stance sign for organic posts (critical experts/customers, promotional competitors)
_ROLE_STANCE = {"expert": -0.2, "customer": -0.1, "competitor": 0.4,
                "investor": 0.0, "engineer": 0.1, "influencer": 0.2}

AMBIENT_BASE_RATE = 0.06            # per active profile, per tick, propensity to post
MAX_AMBIENT_POSTS_PER_TICK = 2      # cap so the forum hums, not floods
REPLY_FRACTION = 0.25              # share of ambient acts that reply to a recent post (沟通)


def _organic_text(topic: str, rng: random.Random) -> str:
    variants = ORGANIC_TEMPLATES.get(topic) or ORGANIC_TEMPLATES["agent_reliability"]
    return rng.choice(variants)


def generate_ambient_posts(
    community: "ExternalCommunity", tick: int, rng: random.Random,
    max_posts: int = MAX_AMBIENT_POSTS_PER_TICK,
) -> List[Post]:
    """Organic forum chatter: external profiles post/reply about their topic interests (the whole
    field), independent of the company. Seeded + rate-capped so it's reproducible and bounded."""
    profiles = list(community.profiles.values())
    rng.shuffle(profiles)
    recent = community.recent_posts(limit=10)
    new: List[Post] = []
    for prof in profiles:
        if len(new) >= max_posts:
            break
        act = float(getattr(prof, "activity_level", 0.5) or 0.5)
        if rng.random() > act * AMBIENT_BASE_RATE:
            continue
        topics = list(getattr(prof, "topic_interests", []) or []) or ["agent_reliability"]
        topic = rng.choice(topics)
        community._ambseq = getattr(community, "_ambseq", 0) + 1
        seq = community._ambseq
        stance = _ROLE_STANCE.get(getattr(prof, "role", ""), 0.0)
        target = rng.choice(recent) if (recent and rng.random() < REPLY_FRACTION) else None
        if target is not None:
            topic = getattr(target, "topic", topic) or topic
            text = f"Re: {topic} — " + _organic_text(topic, rng)
            post_id = f"post_ambient_reply_{tick}_{seq}"
        else:
            text = _organic_text(topic, rng)
            post_id = f"post_ambient_{tick}_{seq}"
        post = Post(
            post_id=post_id, author_id=prof.external_agent_id, topic=topic,
            content_summary=text, stance=stance, credibility=float(getattr(prof, "credibility", 0.6)),
            reach=int(40 + 200 * act), created_tick=tick, visibility="public")
        community.add_post(post)
        if target is not None:
            target.comments = int(getattr(target, "comments", 0) or 0) + 1
        new.append(post)
    return new


class ExternalCommunity:
    """Holds external profiles + a living forum: organic chatter is generated each tick by
    `tick()`, and controlled interventions add pressure posts/signals on top. Internal agents must
    read/search to perceive any of it."""

    def __init__(self, corpus_version: str = "", seed: int = 0):
        self.corpus_version = corpus_version
        self.seed = seed
        self.profiles: Dict[str, ExternalProfile] = {}
        self.posts: Dict[str, Post] = {}
        self.docs: Dict[str, ExternalDoc] = {}
        self.signals: List[MarketSignal] = []
        self.policy = CommunityPolicy(seed=seed)

    def add_post(self, post: Post) -> None:
        self.posts[post.post_id] = post

    def add_signal(self, signal: MarketSignal) -> None:
        self.signals.append(signal)

    def recent_posts(self, *, since_tick: int = 0, limit: int = 20) -> List[Post]:
        items = [p for p in self.posts.values() if int(getattr(p, "created_tick", 0) or 0) >= since_tick]
        items.sort(key=lambda p: int(getattr(p, "created_tick", 0) or 0), reverse=True)
        return items[:limit]

    def search_posts(self, *, query: str, topic: str = "", top_k: int = 5) -> List[Post]:
        terms = [w for w in (query or "").lower().split() if len(w) > 2]
        scored: List[Tuple[int, Post]] = []
        for post in self.posts.values():
            if topic and getattr(post, "topic", "") != topic:
                continue
            hay = f"{getattr(post, 'topic', '')} {getattr(post, 'content_summary', '')}".lower()
            score = sum(hay.count(t) for t in terms) + (1 if topic else 0)
            if score or not terms:
                scored.append((score, post))
        scored.sort(key=lambda sp: (sp[0], int(getattr(sp[1], "created_tick", 0) or 0)), reverse=True)
        return [p for _, p in scored[:top_k]]

    def tick(self, world_tick: int) -> List[Post]:
        """Advance the living forum one tick: external profiles post/reply about the field
        (organic chatter), independent of the company. Seeded by (community seed, tick) so the
        forum is reproducible. Controlled pressure is added separately by generate_intervention()."""
        rng = random.Random(self.seed * 9973 + int(world_tick) + 1)
        return generate_ambient_posts(self, int(world_tick), rng)


__all__ = ["EXTERNAL_ACTIONS", "CommunityPolicy", "ExternalCommunity",
           "INTERVENTION_TEMPLATES", "generate_intervention"]
