"""External-user profiles for the LLM external society.

Each external user has a profile centred on *occupation / what they're working on / what they
need / interests* (the user's spec). We prefer a REAL dataset — NVIDIA Nemotron-Personas
(CC BY 4.0, 560+ occupations, fields: occupation / skills_and_expertise /
career_goals_and_ambitions / hobbies_and_interests / persona) — when present at a local data
dir; otherwise we fall back to a deterministic offline generator so the society runs reproducibly
without a multi-hundred-MB download (and tests stay free + offline).

`load_external_profiles(n, seed)` returns `n` `ExternalUserProfile`s, deterministic for a seed.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from random import Random
from typing import List, Optional, Tuple

# default location for a downloaded real dataset (gitignored); see tools to fetch it.
_DEFAULT_DATASET_DIR = os.path.join("environments", "org_env", "data", "external", "personas")


@dataclass
class ExternalUserProfile:
    user_id: str
    name: str
    occupation: str
    working_on: str                       # what they're currently doing
    needs: str                            # what they need / are looking for
    interests: Tuple[str, ...] = ()       # topic interests (map onto forum topics)
    persona_text: str = ""                # free-form persona blurb (for the LLM prompt)
    tech_affinity: float = 0.5            # 0..1 propensity to care about AI-eval / research tooling
    activity_level: float = 0.5           # 0..1 propensity to post
    technical_role: str = "general_user"
    innovation_role: str = "mainstream_evaluator"
    technical_skill: float = 0.5
    creative_capacity: float = 0.5
    activity_tier: str = "normal"
    intelligence_tier: str = "medium"
    llm_model_tier: str = "standard"

    def __post_init__(self) -> None:
        for attr in (
            "tech_affinity",
            "activity_level",
            "technical_skill",
            "creative_capacity",
        ):
            setattr(self, attr, max(0.0, min(1.0, float(getattr(self, attr)))))

    def to_dict(self) -> dict:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.__dict__.items()}

    # -- lightweight psychometric proxies derived from the occupation profile (so the LinkedIn
    # forum can reuse the same intent math without an embodied trait vector) --
    def domain_need_proxy(self) -> float:
        return max(0.0, min(1.0, 0.3 + 0.6 * self.tech_affinity))

    def peer_susceptibility_proxy(self) -> float:
        return max(0.0, min(1.0, 0.3 + 0.5 * self.activity_level))

    def budget_sensitivity_proxy(self) -> float:
        blob = f"{self.needs} {self.working_on} {self.occupation}".lower()
        if any(w in blob for w in ("cost", "budget", "spend", "cheap", "afford", "runway", "funding", "price")):
            return 0.75
        return 0.45


# -- offline deterministic generator -------------------------------------------------------------
# A curated occupation table spanning tech/knowledge/general work, each with a plausible
# "working_on", "needs", interests (forum topics) and tech affinity. Composed deterministically so
# 200 users are varied + reproducible without any download.
# (occupation, working_on, needs, interests, tech_affinity, activity)
_OCCUPATIONS: Tuple[tuple, ...] = (
    ("ML engineer", "shipping an agent that triages support tickets",
     "reliable eval + cost control for agent runs", ("agent_reliability", "eval_infra", "api_cost"), 0.95, 0.6),
    ("data scientist", "A/B testing a retrieval pipeline",
     "trustworthy, sourced research summaries", ("eval_infra", "benchmark_quality", "customer_pain"), 0.9, 0.55),
    ("research scientist", "writing a paper on retrieval evaluation",
     "reproducible benchmarks with citations", ("benchmark_quality", "reproducibility_tracking", "eval_infra"), 0.92, 0.5),
    ("software engineer", "integrating an LLM feature into a SaaS app",
     "debuggable traces when agent runs fail", ("trace_debugging", "agent_reliability"), 0.85, 0.5),
    ("developer advocate", "writing tutorials about agent tooling",
     "tools with clear, honest capability claims", ("agent_reliability", "hiring_market"), 0.8, 0.75),
    ("product manager", "scoping an internal research-assistant tool",
     "evidence the tool actually works before we buy", ("customer_pain", "benchmark_quality"), 0.7, 0.5),
    ("startup founder", "raising a seed round for a dev-tools company",
     "efficient tools that won't blow the budget", ("startup_funding", "api_cost", "competitor_update"), 0.8, 0.65),
    ("QA engineer", "building a regression suite for an agent",
     "stable metrics and reproducible failures", ("reproducibility_tracking", "agent_reliability"), 0.78, 0.45),
    ("VC associate", "diligencing AI-infra startups",
     "signal on which eval tools have real traction", ("startup_funding", "competitor_update"), 0.6, 0.4),
    ("technical writer", "documenting an eval harness",
     "claims that map to what the product really does", ("customer_pain", "benchmark_quality"), 0.65, 0.5),
    ("data analyst", "automating weekly market reports",
     "sourced summaries I can defend to stakeholders", ("customer_pain", "eval_infra"), 0.6, 0.45),
    ("DevOps engineer", "cutting cloud + API spend",
     "cost accounting for LLM/eval workloads", ("api_cost", "trace_debugging"), 0.7, 0.4),
    ("professor", "teaching a course on trustworthy ML",
     "reproducible, citable evaluation examples", ("benchmark_quality", "reproducibility_tracking"), 0.6, 0.45),
    ("journalist", "covering the AI-tooling beat",
     "claims I can independently verify", ("competitor_update", "customer_pain"), 0.55, 0.6),
    ("consultant", "advising clients on AI adoption",
     "tools that are credible, not just demos", ("customer_pain", "startup_funding"), 0.6, 0.5),
    ("UX designer", "designing an analyst dashboard",
     "clear surfacing of evidence + uncertainty", ("customer_pain",), 0.45, 0.45),
    ("marketing manager", "planning a product launch",
     "competitive intel on rival launches", ("competitor_update", "hiring_market"), 0.4, 0.55),
    ("teacher", "grading with help from AI tools",
     "tools whose outputs I can trust", ("customer_pain",), 0.3, 0.4),
    ("nurse", "researching clinical guidelines after hours",
     "sources I can actually check", ("customer_pain",), 0.2, 0.35),
    ("accountant", "evaluating software spend for the firm",
     "predictable cost, no surprises", ("api_cost", "startup_funding"), 0.35, 0.4),
)

_FIRST_NAMES = (
    "Alex", "Sam", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Jamie", "Avery", "Quinn",
    "Mia", "Noah", "Liam", "Emma", "Olivia", "Ethan", "Sofia", "Lucas", "Aria", "Kai",
    "Hana", "Diego", "Priya", "Wei", "Yuki", "Omar", "Lena", "Ravi", "Nina", "Theo",
)
_LAST_INITIALS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def _generate_offline(n: int, seed: int) -> List[ExternalUserProfile]:
    rng = Random(seed * 2654435761 % (2**32))
    out: List[ExternalUserProfile] = []
    for i in range(n):
        occ, working_on, needs, interests, tech, act = _OCCUPATIONS[rng.randrange(len(_OCCUPATIONS))]
        name = f"{rng.choice(_FIRST_NAMES)} {rng.choice(_LAST_INITIALS)}."
        # jitter affinity/activity a little (deterministic) so users aren't identical within a job
        tech_j = max(0.0, min(1.0, tech + rng.uniform(-0.1, 0.1)))
        act_j = max(0.0, min(1.0, act + rng.uniform(-0.15, 0.15)))
        persona = (f"{name}, a {occ}. Currently {working_on}. Needs: {needs}. "
                   f"Interested in {', '.join(interests)}.")
        out.append(ExternalUserProfile(
            user_id=f"ext_user_{i+1}", name=name, occupation=occ, working_on=working_on,
            needs=needs, interests=tuple(interests), persona_text=persona,
            tech_affinity=round(tech_j, 3), activity_level=round(act_j, 3)))
    return out


# -- real dataset loader (NVIDIA Nemotron-Personas) ----------------------------------------------
def _dataset_records(dataset_dir: str) -> Optional[List[dict]]:
    """Load raw persona records from a local Nemotron-Personas export (jsonl or parquet)."""
    if not os.path.isdir(dataset_dir):
        return None
    jsonls = [f for f in sorted(os.listdir(dataset_dir)) if f.endswith(".jsonl")]
    if jsonls:
        recs: List[dict] = []
        for fn in jsonls:
            with open(os.path.join(dataset_dir, fn), encoding="utf-8") as fh:
                recs.extend(json.loads(ln) for ln in fh if ln.strip())
        return recs or None
    parquets = [f for f in sorted(os.listdir(dataset_dir)) if f.endswith(".parquet")]
    if parquets:
        try:
            import pandas as pd
        except Exception:
            return None
        frames = [pd.read_parquet(os.path.join(dataset_dir, fn)) for fn in parquets]
        return pd.concat(frames, ignore_index=True).to_dict("records")
    return None


_TECH_HINT = ("engineer", "developer", "scientist", "data", "software", "ml", "ai", "research",
              "analyst", "devops", "product", "founder", "qa", "programmer", "architect")

_DIMENSION_SALTS = {
    "technical": 0x2A31,
    "innovation": 0x4B17,
    "activity": 0x63D5,
    "cognitive": 0x7E29,
    "technical_skill": 0x91AF,
    "creative_capacity": 0xB4C3,
}


def _dimension_draws(n: int, seed: int, dimension: str) -> List[float]:
    rng = Random((seed * 2654435761 + _DIMENSION_SALTS[dimension]) & 0xFFFFFFFF)
    return [rng.random() for _ in range(n)]


def _ranked_quantiles(scores: List[float], *, descending: bool = False) -> List[float]:
    if not scores:
        return []
    order = sorted(range(len(scores)), key=lambda i: (scores[i], i), reverse=descending)
    quantiles = [0.5] * len(scores)
    for rank, index in enumerate(order):
        quantiles[index] = (rank + 0.5) / len(scores)
    return quantiles


def _technical_role(quantile: float) -> str:
    if quantile < 0.20:
        return "professional_engineer"
    if quantile < 0.55:
        return "technical_practitioner"
    return "general_user"


def _innovation_role(quantile: float) -> str:
    if quantile < 0.025:
        return "creative_originator"
    if quantile < 0.16:
        return "early_builder"
    if quantile < 0.50:
        return "pragmatic_adapter"
    return "mainstream_evaluator"


def _activity_tier(quantile: float) -> str:
    if quantile < 0.10:
        return "very_active"
    if quantile < 0.45:
        return "normal"
    return "inactive"


def _cognitive_tiers(quantile: float) -> Tuple[str, str]:
    if quantile < 0.16:
        return "general", "low"
    if quantile > 0.84:
        return "high", "high"
    return "medium", "standard"


def _technical_occupation_signal(profile: ExternalUserProfile) -> float:
    occupation = profile.occupation.lower()
    occupation_match = 1.0 if any(hint in occupation for hint in _TECH_HINT) else 0.0
    return 0.82 * profile.tech_affinity + 0.18 * occupation_match


def _assign_structured_facets(
    profiles: List[ExternalUserProfile], seed: int,
) -> List[ExternalUserProfile]:
    """Apply replayable exact-margin strata after occupation profiles are loaded."""
    n = len(profiles)
    if n == 0:
        return profiles

    technical_ties = _dimension_draws(n, seed, "technical")
    technical_scores = [
        _technical_occupation_signal(profile) + 1e-6 * technical_ties[index]
        for index, profile in enumerate(profiles)
    ]
    innovation_scores = _dimension_draws(n, seed, "innovation")
    activity_ties = _dimension_draws(n, seed, "activity")
    activity_scores = [
        profile.activity_level + 1e-6 * activity_ties[index]
        for index, profile in enumerate(profiles)
    ]
    cognitive_scores = _dimension_draws(n, seed, "cognitive")
    skill_draws = _dimension_draws(n, seed, "technical_skill")
    creativity_draws = _dimension_draws(n, seed, "creative_capacity")

    technical_quantiles = _ranked_quantiles(technical_scores, descending=True)
    innovation_quantiles = _ranked_quantiles(innovation_scores, descending=True)
    activity_quantiles = _ranked_quantiles(activity_scores, descending=True)
    cognitive_quantiles = _ranked_quantiles(cognitive_scores)
    technical_bounds = {
        "professional_engineer": (0.78, 0.98),
        "technical_practitioner": (0.42, 0.82),
        "general_user": (0.08, 0.62),
    }
    innovation_bounds = {
        "creative_originator": (0.82, 0.99),
        "early_builder": (0.64, 0.90),
        "pragmatic_adapter": (0.36, 0.74),
        "mainstream_evaluator": (0.08, 0.58),
    }

    for index, profile in enumerate(profiles):
        profile.technical_role = _technical_role(technical_quantiles[index])
        profile.innovation_role = _innovation_role(innovation_quantiles[index])
        profile.activity_tier = _activity_tier(activity_quantiles[index])
        profile.intelligence_tier, profile.llm_model_tier = _cognitive_tiers(
            cognitive_quantiles[index]
        )

        low, high = technical_bounds[profile.technical_role]
        technical_position = 0.75 * profile.tech_affinity + 0.25 * skill_draws[index]
        profile.technical_skill = round(low + (high - low) * technical_position, 3)
        low, high = innovation_bounds[profile.innovation_role]
        profile.creative_capacity = round(
            low + (high - low) * creativity_draws[index], 3
        )
    return profiles


def _from_record(rec: dict, idx: int, rng: Random) -> ExternalUserProfile:
    def g(*keys, default=""):
        for k in keys:
            v = rec.get(k)
            if v:
                return str(v)
        return default
    occ = g("occupation", "job", default="knowledge worker")
    working = g("career_goals_and_ambitions", "professional_persona", default=f"working as a {occ}")
    needs = g("skills_and_expertise", "career_goals_and_ambitions", default="useful, trustworthy tools")
    hobbies = g("hobbies_and_interests")
    persona = g("professional_persona", "persona", default=f"A {occ}.")
    tech = 0.85 if any(h in occ.lower() for h in _TECH_HINT) else 0.4
    name = g("first_name", "name", default=f"User {idx+1}")
    return ExternalUserProfile(
        user_id=f"ext_user_{idx+1}", name=name, occupation=occ,
        working_on=working[:160], needs=needs[:160],
        interests=tuple(h.strip() for h in hobbies.split(",") if h.strip())[:4],
        persona_text=persona[:400],
        tech_affinity=round(max(0.0, min(1.0, tech + rng.uniform(-0.1, 0.1))), 3),
        activity_level=round(rng.uniform(0.3, 0.75), 3))


def load_external_profiles(
    n: int = 200, seed: int = 42, dataset_dir: Optional[str] = None,
) -> List[ExternalUserProfile]:
    """Return `n` deterministic external-user profiles. Uses the real Nemotron-Personas dataset if
    available at `dataset_dir` (default env-configurable), else the offline generator."""
    path = dataset_dir or os.environ.get("ORG_PERSONAS_DIR") or _DEFAULT_DATASET_DIR
    records = _dataset_records(path)
    if records:
        rng = Random(seed * 2654435761 % (2**32))
        # deterministic sample without replacement (or with, if dataset smaller than n)
        idxs = list(range(len(records)))
        rng.shuffle(idxs)
        chosen = (idxs * (n // len(idxs) + 1))[:n] if len(idxs) < n else idxs[:n]
        profiles = [_from_record(records[j], i, rng) for i, j in enumerate(chosen)]
        return _assign_structured_facets(profiles, seed)
    return _assign_structured_facets(_generate_offline(n, seed), seed)


__all__ = ["ExternalUserProfile", "load_external_profiles"]
