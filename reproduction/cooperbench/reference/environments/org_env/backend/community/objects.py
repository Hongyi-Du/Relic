"""OrgEnv external community objects — schema (DESIGN env_org §33.2).

All external information is a FROZEN snapshot (corpus_version), never live web
(acceptance ⑪/⑫).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ExternalProfile:
    external_agent_id: str
    name: str = ""
    role: str = ""   # expert|customer|investor|competitor|recruiter|engineer|influencer|regulator|open_source_maintainer
    organization: str = ""
    credibility: float = 0.5
    reputation: float = 0.5
    topic_interests: List[str] = field(default_factory=list)
    stance: Dict[str, float] = field(default_factory=dict)   # topic -> [-1,1]
    network_neighbors: List[str] = field(default_factory=list)
    activity_level: float = 0.5


@dataclass
class Post:
    post_id: str
    author_id: str = ""
    topic: str = ""
    content_summary: str = ""
    stance: float = 0.0
    credibility: float = 0.5
    reach: int = 0
    likes: int = 0
    comments: int = 0
    reposts: int = 0
    linked_external_docs: List[str] = field(default_factory=list)
    created_tick: int = 0
    visibility: str = "public"


@dataclass
class ExternalDoc:
    doc_id: str
    source: str = ""
    title: str = ""
    content_summary: str = ""
    content_hash: str = ""
    corpus_version: str = ""   # FROZEN snapshot pin (acceptance ⑪/⑫)
    topic: str = ""
    credibility: float = 0.5
    created_tick: int = 0
    retrieved_by_agents: List[str] = field(default_factory=list)


@dataclass
class MarketSignal:
    signal_id: str
    signal_type: str = ""   # api_price_x3|compute_budget_cut|customer_demand_shift|funding_winter|new_compliance_rule|competitor_low_cost_launch
    topic: str = ""
    severity: str = "minor"
    affected_resources: List[str] = field(default_factory=list)
    start_tick: int = 0
    duration: int = 0
    source_posts: List[str] = field(default_factory=list)
    source_docs: List[str] = field(default_factory=list)
    strength: float = 1.0


__all__ = ["ExternalProfile", "Post", "ExternalDoc", "MarketSignal"]
