"""OrgEnv economy objects — ComputeBudget / CustomerTicket (DESIGN env_org §33.1).

The budget is the core of the API-price-shock experiment (§34): its
``cost_multiplier`` is bumped by the external signal, driving budget pressure ->
workflow wish -> governance protocol emergence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ComputeBudget:
    budget_id: str
    total_budget: float = 0.0
    remaining_budget: float = 0.0
    daily_quota: float = 0.0
    used_by_agent: Dict[str, float] = field(default_factory=dict)
    usage_events: List[Dict[str, Any]] = field(default_factory=list)
    approval_required: bool = False
    current_policy: Optional[str] = None
    cost_multiplier: float = 1.0   # bumped by API price shock (§34)


@dataclass
class CustomerTicket:
    ticket_id: str
    customer_type: str = ""
    complaint_or_request: str = ""
    severity: str = "minor"
    topic: str = ""
    linked_external_post_id: Optional[str] = None
    status: str = "open"
    assigned_agent_id: Optional[str] = None
    response_status: str = "pending"


@dataclass
class CustomerTrial:
    """v14 P5: a simulated external user who tries a *published* release and leaves a
    market signal. Unlike ``ExternalOffer`` (recruitment) this is the customer-side
    willingness-to-pay / conversion signal that drives the ``customers`` funding
    milestone. Satisfaction/WTP track the product's REAL runtime quality, so the market
    can only be won by actually shipping a usable, grounded product (not by gaming gates).
    """
    trial_id: str
    customer_type: str = ""           # smb_analyst / researcher / enterprise_eval / ...
    persona: str = ""                 # short human-readable description
    query: str = ""                   # the research question they tried
    release_version: str = ""         # which release they evaluated
    outcome: str = "rejected"         # "converted" | "interested" | "rejected"
    satisfaction: float = 0.0         # 0..1, tracks real product quality
    willingness_to_pay: float = 0.0   # tokens/month they'd pay (0 if not interested)
    converted: bool = False           # became a paying customer
    feedback: str = ""                # free-text praise / complaint / feature request
    linked_ticket_id: Optional[str] = None
    created_tick: int = 0
    decision_source: str = ""
    profile_id: str = ""
    product_evidence_source: str = ""


__all__ = ["ComputeBudget", "CustomerTicket", "CustomerTrial"]
