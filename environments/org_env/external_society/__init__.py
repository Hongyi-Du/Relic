"""LLM external society (OrgEnv) — a forum of ~200 profiled external users who chat about the
whole field, encounter random events, and (after a release) experience the company's product and
report purchase intent. Real datasets (Nemotron-Personas / world events) with deterministic
offline fallback; every random group is seeded for reproducibility.

Stage 1 (data layer) lives here; the society loop + LLM layer build on top.
"""
from environments.org_env.external_society.agent import (
    ExternalUser,
    ForumPost,
    PostDecision,
    ProductIntent,
    ProductOffering,
)
from environments.org_env.external_society.decision import HeuristicDecisionProvider
from environments.org_env.external_society.llm_user import (
    LLMDecisionProvider,
    external_llm_enabled,
    make_provider,
)
from environments.org_env.external_society.event_pool import (
    EventPool,
    EventRecord,
    build_event_pool,
)
from environments.org_env.external_society.profiles_dataset import (
    ExternalUserProfile,
    load_external_profiles,
)
from environments.org_env.external_society.bridge import (
    attach_external_society,
    drive_post_release_market,
    external_society_enabled,
    maybe_run_external_society,
    run_external_product_experience,
)
from environments.org_env.external_society.society import ExternalSociety

__all__ = [
    "ExternalUserProfile", "load_external_profiles",
    "EventRecord", "EventPool", "build_event_pool",
    "ExternalUser", "ForumPost", "PostDecision", "ProductOffering", "ProductIntent",
    "HeuristicDecisionProvider", "LLMDecisionProvider", "make_provider", "external_llm_enabled",
    "ExternalSociety",
    "attach_external_society", "maybe_run_external_society", "run_external_product_experience",
    "drive_post_release_market", "external_society_enabled",
]
