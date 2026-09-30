"""Budget / Funding / Cost / Payroll / Retention objects (DESIGN env_org §48-§56)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class CompanyBudget:
    budget_id: str = "main"
    total_committed: float = 3000.0
    cash_balance: float = 1000.0
    available_now: float = 1000.0
    remaining_budget: float = 1000.0
    daily_base_burn: float = 20.0
    compute_budget: float = 600.0
    api_budget: float = 600.0
    payroll_reserved: float = 0.0
    next_tranche_tick: int = 168                 # day 7 (24*7)
    runway_days: float = 0.0
    burn_rate: float = 0.0
    budget_pressure: float = 0.0                 # 0..1
    cost_multiplier: float = 1.0                 # API price shock lever (§34)
    spending_by_agent: Dict[str, float] = field(default_factory=dict)
    spending_by_task: Dict[str, float] = field(default_factory=dict)
    spending_by_experiment: Dict[str, float] = field(default_factory=dict)
    spending_by_category: Dict[str, float] = field(default_factory=dict)


@dataclass
class FundingTranche:
    tranche_id: str
    scheduled_tick: int
    amount: float
    condition: str = ""                          # "" = unconditional
    status: str = "scheduled"                    # scheduled|arrived|delayed|cancelled
    actual_arrival_tick: Optional[int] = None
    delay_reason: str = ""


@dataclass
class FundingSchedule:
    funding_schedule_id: str = "seed"
    tranches: List[FundingTranche] = field(default_factory=list)
    delayed_tranches: List[str] = field(default_factory=list)
    investor_confidence: float = 0.6
    milestone_status: Dict[str, bool] = field(default_factory=dict)
    # Internal Pipeline spec #5: each checkpoint records an explicit decision
    # (grant_full / grant_partial / delay / conditional_grant / reject) with its evidence.
    funding_history: List[Dict] = field(default_factory=list)


@dataclass
class CostEvent:
    cost_event_id: str
    tick: int
    agent_id: str
    action_type: str
    resource_type: str = "api"                   # api|compute|service|other
    base_cost: float = 0.0
    multiplier: float = 1.0
    final_cost: float = 0.0
    charged_to: str = "main"
    budget_before: float = 0.0
    budget_after: float = 0.0
    task_id: Optional[str] = None
    experiment_id: Optional[str] = None
    logged_to_ledger: bool = False
    approved: bool = True
    policy_violation: bool = False
    notes: str = ""


@dataclass
class PayrollState:
    payroll_state_id: str = "payroll"
    monthly_payroll: float = 700.0
    weekly_payroll: float = 175.0
    next_payroll_tick: int = 168                 # weekly (day 7)
    payroll_due: float = 0.0
    payroll_paid: float = 0.0
    missed_payroll_count: int = 0
    partial_payroll_count: int = 0
    founder_salary_deferred: float = 0.0
    employee_salary_debt: float = 0.0
    payroll_status: str = "normal"               # normal|at_risk|delayed|partial|missed


@dataclass
class CompensationProfile:
    agent_id: str
    base_salary: float = 25.0                    # per weekly cycle
    equity_share: float = 0.0
    is_founder: bool = False
    salary_paid_until_tick: int = 0
    unpaid_salary: float = 0.0
    cash_need_level: float = 0.5                 # 0..1 (founders lower)
    financial_tolerance: float = 0.5             # 0..1 (founders higher)
    commitment_level: float = 0.6
    trust_in_company: float = 0.8
    outside_options: float = 0.2                 # raised by external offers
    burnout_level: float = 0.0
    belief_in_mission: float = 0.6
    retention_risk: float = 0.0                  # 0..1 (computed)
    departure_threshold: float = 0.8


@dataclass
class ExternalOffer:
    offer_id: str
    candidate_agent_id: str
    source_external_agent_id: str
    role: str = ""
    compensation: float = 0.0
    reputation_gain: float = 0.0
    risk: float = 0.3
    deadline: int = 0
    status: str = "open"                         # open|considering|accepted|declined|expired


__all__ = [
    "CompanyBudget", "FundingTranche", "FundingSchedule", "CostEvent",
    "PayrollState", "CompensationProfile", "ExternalOffer",
]
