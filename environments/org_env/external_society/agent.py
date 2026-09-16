"""External-society objects: a LinkedIn-like forum post, an external user (profile + memory),
and the bounded decision/intent records the decision provider returns.

This is the NON-embodied LinkedIn model (per the OrgEnv story §6): users have an occupation-based
profile + memory of what they've seen and experienced — NOT bodies/hunger. The purchase-intent
record mirrors society_core's `LLMIntentRecord` shape so results are comparable / swappable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from environments.org_env.external_society.profiles_dataset import ExternalUserProfile


@dataclass
class ForumPost:
    post_id: str
    author_id: str
    day: int
    text: str
    topic: str = ""
    board: str = "general"            # "general" (field chatter) | "product" (the company's board)
    parent_id: Optional[str] = None   # reply -> a thread floor
    likes: int = 0
    reposts: int = 0
    about_event_id: Optional[str] = None
    about_product: Optional[str] = None   # product/artifact id when this is product feedback

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class PostDecision:
    """What an external user decides to do after encountering an event (provider output)."""
    will_post: bool = False
    text: str = ""
    topic: str = ""
    reply_to: Optional[str] = None    # post_id being replied to (None = new top-level post)


@dataclass
class ProductOffering:
    """The company's released product as the external market sees it. `quality` is the REAL
    product quality (fed from OrgEnv readiness/grounding); `price` is on the same 0..1 scale as
    willingness_to_pay so conversion = WTP >= price."""
    product_id: str
    summary: str = ""
    quality: float = 0.5
    topics: tuple = ()
    price: float = 0.3
    fallback_grounded: bool = False   # true if claims are only structurally (not semantically) grounded
    quality_source: str = "unspecified"
    quality_evidence_ref: str = ""


@dataclass
class ProductIntent:
    """A user's evaluation after experiencing the product — mirrors society_core's LLMIntentRecord
    fields (intent_to_try / intent_to_pay / WTP / recommend / satisfaction) so the two are
    comparable. `converted` is the kernel-style decision (intent_to_pay over threshold + WTP)."""
    user_id: str
    product_id: str
    day: int
    satisfaction: float = 0.0
    intent_to_try: float = 0.0
    intent_to_pay: float = 0.0
    willingness_to_pay: float = 0.0
    intent_to_recommend: float = 0.0
    confidence: float = 0.0
    converted: bool = False
    feedback: str = ""
    reason_code: str = ""
    source: str = "heuristic"
    profile_id: str = ""
    technical_role: str = "general_user"
    innovation_role: str = "mainstream_evaluator"

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class ExternalUser:
    """An external forum user: occupation-based profile + memory. The whole forum is visible to
    them (they can read any floor); `memory` is what they've actually engaged with / experienced
    (events encountered, posts they reacted to, their own posts, the product experience)."""
    profile: ExternalUserProfile
    memory: List[str] = field(default_factory=list)       # short text memories (events/experiences)
    seen_post_ids: List[str] = field(default_factory=list)
    posted_ids: List[str] = field(default_factory=list)
    product_intent: Optional[ProductIntent] = None

    @property
    def user_id(self) -> str:
        return self.profile.user_id

    def remember(self, text: str, cap: int = 40) -> None:
        self.memory.append(text)
        if len(self.memory) > cap:
            del self.memory[:-cap]


__all__ = ["ForumPost", "PostDecision", "ProductOffering", "ProductIntent", "ExternalUser"]
