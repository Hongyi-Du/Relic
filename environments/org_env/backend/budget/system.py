"""BudgetSystem — cost / funding / payroll / retention (DESIGN env_org §48-§56).

Cohesive because they interact: charging spends the budget; funding tranches
refill it; payroll pays salaries (or misses them); missed payroll raises
``unpaid_salary`` -> trust down + retention_risk up -> consider_leaving. External
offers raise ``outside_options``. All deterministic given the same inputs.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from environments.org_env.backend.budget.objects import (
    CompanyBudget,
    CompensationProfile,
    CostEvent,
    ExternalOffer,
    FundingSchedule,
    FundingTranche,
    PayrollState,
)


def default_funding_schedule() -> FundingSchedule:
    """Day 0 / 7 / 14 tranches of 1000 (later two conditional) — §49."""
    return FundingSchedule(tranches=[
        FundingTranche(tranche_id="tr1", scheduled_tick=0, amount=1000.0),
        FundingTranche(tranche_id="tr2", scheduled_tick=168, amount=1000.0, condition="milestone_demo"),
        FundingTranche(tranche_id="tr3", scheduled_tick=336, amount=1000.0, condition="milestone_customers"),
    ])


class BudgetSystem:
    def __init__(self, *, budget: Optional[CompanyBudget] = None,
                 funding: Optional[FundingSchedule] = None,
                 payroll: Optional[PayrollState] = None):
        self.budget = budget or CompanyBudget()
        self.funding = funding or default_funding_schedule()
        self.payroll = payroll or PayrollState()
        self.comp: Dict[str, CompensationProfile] = {}
        self.cost_events: List[CostEvent] = []
        self.external_offers: Dict[str, ExternalOffer] = {}
        self.events: List[dict] = []
        # v8g P1: a single TOKEN ledger unifying every flow — action/tool/LLM/meeting/eval/CI
        # costs (debits), salary (debits), funding tranches (credits) — all in token units, so
        # treasury / daily-burn / runway / runway_pressure are one coherent currency.
        self.token_ledger: List[dict] = []
        self._seq = 0

    # token cost per action CATEGORY (registry categories) + an LLM-call surcharge.
    # Calibrated so ~8 agents acting through a day burn ~40-70 tokens/day against a ~1000-token
    # treasury (+1000/tranche): runway is a REAL constraint that rises under heavy/expensive
    # activity, but doesn't paralyse the company from day one.
    TOKEN_COST_BY_CAT = {
        "search": 1.2, "experiment": 1.2, "sandbox": 1.2, "repo": 0.8, "product": 0.8,
        "work": 0.8, "review": 0.6, "meeting": 0.4, "governance": 0.4, "comm": 0.3,
        "doc": 0.6, "artifact": 0.6, "protocol": 0.4, "bridge": 0.4, "release": 0.4,
        "social": 0.2, "time": 0.0, "payroll": 0.0, "other": 0.5,
    }
    LLM_CALL_TOKENS = 0.4

    def _record_ledger(self, kind: str, amount: float, reason: str, tick: int,
                       agent: str = "company") -> None:
        self.token_ledger.append({
            "tick": int(tick), "kind": kind, "amount": round(float(amount), 2),
            "reason": reason, "agent": agent, "balance_after": round(self.budget.cash_balance, 1)})

    def charge_tokens(self, *, agent_id: str, action_type: str, tick: int,
                      category: Optional[str] = None, llm: bool = False) -> Optional[CostEvent]:
        """v8g P1: charge an action's token cost (by category + LLM surcharge) against the
        treasury. Reuses charge() so it lands in cost_events AND the token ledger."""
        base = self.TOKEN_COST_BY_CAT.get(category or "other", 1.0) + (self.LLM_CALL_TOKENS if llm else 0.0)
        if base <= 0:
            return None
        return self.charge(agent_id=agent_id, action_type=action_type, base_cost=base,
                           resource_type="action", tick=tick)

    def daily_burn_tokens(self, tick: int, window: int = 24) -> float:
        """Actual rolling token burn over the last `window` ticks (not a fixed constant)."""
        lo = int(tick) - window
        spent = sum(e["amount"] for e in self.token_ledger if e["kind"] == "debit" and e["tick"] >= lo)
        return round(max(float(self.budget.daily_base_burn) * 0.25, spent), 2)

    def register_agent(self, profile: CompensationProfile) -> None:
        self.comp[profile.agent_id] = profile

    # -- cost ---------------------------------------------------------------
    def charge(self, *, agent_id: str, action_type: str, base_cost: float,
               resource_type: str = "api", tick: int = 0, task_id: Optional[str] = None,
               experiment_id: Optional[str] = None) -> CostEvent:
        self._seq += 1
        mult = self.budget.cost_multiplier
        final = base_cost * mult
        before = self.budget.remaining_budget
        violation = final > self.budget.remaining_budget
        self.budget.remaining_budget = max(0.0, self.budget.remaining_budget - final)
        self.budget.cash_balance = max(0.0, self.budget.cash_balance - final)
        self.budget.spending_by_agent[agent_id] = self.budget.spending_by_agent.get(agent_id, 0.0) + final
        self.budget.spending_by_category[resource_type] = \
            self.budget.spending_by_category.get(resource_type, 0.0) + final
        if task_id:
            self.budget.spending_by_task[task_id] = self.budget.spending_by_task.get(task_id, 0.0) + final
        if experiment_id:
            self.budget.spending_by_experiment[experiment_id] = \
                self.budget.spending_by_experiment.get(experiment_id, 0.0) + final
        self._refresh_pressure()
        ce = CostEvent(cost_event_id=f"cost_{self._seq}", tick=tick, agent_id=agent_id,
                       action_type=action_type, resource_type=resource_type, base_cost=base_cost,
                       multiplier=mult, final_cost=final, budget_before=before,
                       budget_after=self.budget.remaining_budget, task_id=task_id,
                       experiment_id=experiment_id, policy_violation=violation)
        ce.logged_to_ledger = True
        self.cost_events.append(ce)
        self._record_ledger("debit", final, action_type, tick, agent_id)
        return ce

    def _refresh_pressure(self) -> None:
        b = self.budget
        b.budget_pressure = round(1.0 - min(1.0, b.remaining_budget / max(1.0, b.total_committed)), 3)
        b.burn_rate = b.daily_base_burn * b.cost_multiplier
        b.runway_days = round(b.remaining_budget / max(1.0, b.burn_rate), 1)

    def apply_api_price_shock(self, multiplier: float) -> None:
        """API Price Shock lever (§34): raise cost_multiplier -> budget pressure."""
        self.budget.cost_multiplier = multiplier
        self._refresh_pressure()
        self.events.append({"type": "api_price_shock", "multiplier": multiplier})

    # -- funding ------------------------------------------------------------
    def advance_funding(self, tick: int, *, met_conditions: Optional[set] = None,
                        delay: Optional[set] = None) -> List[FundingTranche]:
        met = met_conditions or set()
        delay = delay or set()
        arrived = []
        for tr in self.funding.tranches:
            # a delayed tranche can still recover once its condition is met
            if tr.status not in ("scheduled", "delayed") or tr.scheduled_tick > tick:
                continue
            if tr.tranche_id in delay or (tr.condition and tr.condition not in met):
                tr.status = "delayed"
                tr.delay_reason = "condition_unmet" if tr.condition else "investor_delay"
                if tr.tranche_id not in self.funding.delayed_tranches:
                    self.funding.delayed_tranches.append(tr.tranche_id)
                continue
            tr.status = "arrived"
            tr.actual_arrival_tick = tick
            self.budget.cash_balance += tr.amount
            self.budget.remaining_budget += tr.amount
            self.budget.total_committed = max(self.budget.total_committed, self.budget.remaining_budget)
            self._record_ledger("credit", tr.amount, f"funding:{tr.tranche_id}", tick, "investor")
            arrived.append(tr)
        self._refresh_pressure()
        return arrived

    # -- payroll ------------------------------------------------------------
    def run_payroll(self, tick: int) -> PayrollState:
        due = sum(c.base_salary for c in self.comp.values() if not c.is_founder)
        founder_due = sum(c.base_salary for c in self.comp.values() if c.is_founder)
        self.payroll.payroll_due = due
        cash = self.budget.cash_balance
        if cash >= due:
            for c in self.comp.values():
                if c.is_founder:
                    continue
                c.salary_paid_until_tick = tick
            self.budget.cash_balance -= due
            self.budget.remaining_budget = max(0.0, self.budget.remaining_budget - due)  # v8h P2: one balance
            self.payroll.payroll_paid += due
            self.payroll.payroll_status = "normal"
            self._record_ledger("debit", due, "salary_payroll", tick, "company")
            # founders may defer when cash is tight afterwards
            self.payroll.founder_salary_deferred += founder_due
        elif cash > 0:
            ratio = cash / due if due else 1.0
            for c in self.comp.values():
                if c.is_founder:
                    continue
                paid = c.base_salary * ratio
                c.unpaid_salary += (c.base_salary - paid)
                c.trust_in_company = max(0.0, c.trust_in_company - 0.1)
            self.budget.cash_balance = 0.0
            self.budget.remaining_budget = max(0.0, self.budget.remaining_budget - cash)  # v8h P2: one balance
            self.payroll.partial_payroll_count += 1
            self.payroll.payroll_status = "partial"
            self.events.append({"type": "payroll_partial", "tick": tick})
        else:
            for c in self.comp.values():
                if c.is_founder:
                    continue
                c.unpaid_salary += c.base_salary
                c.trust_in_company = max(0.0, c.trust_in_company - 0.2)
            self.payroll.missed_payroll_count += 1
            self.payroll.payroll_status = "missed"
            self.events.append({"type": "payroll_missed", "tick": tick})
        self.payroll.next_payroll_tick = tick + 168
        for aid in self.comp:
            self.update_retention(aid, tick=tick)
        return self.payroll

    # -- retention ----------------------------------------------------------
    def add_external_offer(self, offer: ExternalOffer) -> None:
        self.external_offers[offer.offer_id] = offer
        c = self.comp.get(offer.candidate_agent_id)
        if c:
            c.outside_options = min(1.0, c.outside_options + 0.3)
            self.update_retention(offer.candidate_agent_id)

    def update_retention(self, agent_id: str, tick: int = 0) -> float:
        c = self.comp.get(agent_id)
        if not c:
            return 0.0
        unpaid_norm = min(1.0, c.unpaid_salary / max(1.0, c.base_salary * 3))
        risk = (0.30 * unpaid_norm
                + 0.20 * c.burnout_level
                + 0.20 * c.outside_options
                + 0.15 * (1.0 - c.trust_in_company)
                + 0.15 * (1.0 - c.belief_in_mission))
        risk *= (1.0 - 0.4 * c.financial_tolerance)   # tolerant agents resist
        c.retention_risk = round(max(0.0, min(1.0, risk)), 3)
        # consider_leaving fires on high continuous risk OR sustained unpaid
        # salary (>= ~3 cycles), so long-term non-payment alone can drive exits.
        sustained_unpaid = c.unpaid_salary >= c.base_salary * 3 and not c.is_founder
        if c.retention_risk >= c.departure_threshold or sustained_unpaid:
            self.events.append({"type": "consider_leaving", "agent_id": agent_id,
                                "tick": tick, "retention_risk": c.retention_risk,
                                "reason": "sustained_unpaid" if sustained_unpaid else "high_risk"})
        return c.retention_risk


__all__ = ["BudgetSystem", "default_funding_schedule"]
