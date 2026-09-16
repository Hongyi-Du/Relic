"""Relic organization world state.

Holds internal agents + their workspaces + entities + external community + public
records, and exposes a ``DomainState`` snapshot to the lived Core.

``build`` seeds a deterministic organization around the frozen OSS workload
selected by the scenario.
"""
from __future__ import annotations

import os
import random
import re
from typing import Any, Dict, List, Optional

from relic.core.domain import DomainScenarioConfig, DomainState
from environments.org_env.backend.agents import SEED_TEAM, OrgAgent
from environments.org_env.backend.budget import BudgetSystem, CompensationProfile
from environments.org_env.backend.clock import AgentAvailability, TimeSystem
from environments.org_env.backend.comm import CommunicationSystem
from environments.org_env.backend.community import ExternalCommunity, ExternalProfile, Post
from environments.org_env.backend.experiments import BenchmarkScenario, Dataset
from environments.org_env.backend.meetings import MeetingSystem
from environments.org_env.backend.repo import RepoLiteSystem
from environments.org_env.backend.protocol import ProtocolRegistry
from environments.org_env.backend.protocol.registry import (
    ADOPT_MIN_SUPPORTERS,
    effective_min_supporters,
)
from environments.org_env.backend.protocol.objects import (
    GovernedObjectSnapshot,
    IndependentOutcomeOracle,
)
from environments.org_env.backend.sandbox import SandboxSystem
from environments.org_env.backend.search import SearchSystem
from environments.org_env.backend.entities import (
    COMPLETED_TASK_STATUSES,
    ComputeBudget,
    CustomerTicket,
    CustomerTrial,
    Document,
    Experiment,
    Issue,
    Protocol,
    SharedBoard,
    Task,
    TaskStatus,
    WorkflowArtifact,
)
from environments.org_env.backend.workspace import (
    CompanyWorkspace,
    FileObject,
    PersonalWorkspace,
    Visibility,
)
from environments.org_env.config.baseline_conditions import (
    ACTION_SELECTION_FLAT_DETERMINISTIC,
    ACTION_SELECTION_LLM_DIRECT,
    ACTION_SELECTION_PROFILE_POLICY,
    B0_SINGLE_AGENT_FOUNDER,
    resolve_condition,
    roster_for_condition,
)

# Initial 8-member team is defined in backend/agents/seed_team.py (O1 §4).

# semi_auto governance: a high-score proposal left undecided this many ticks (~1 day)
# is auto-adopted so the simulation never deadlocks on a single missing approver.
APPROVAL_DEADLOCK_TICKS = 24

# customer-family enforcement moment: an inbound ticket still ownerless and
# unanswered after one full clock day (24 one-hour ticks, the same day length
# every other daily sweep keys off hour_in_day == 0) is a caught non-compliance
# with an adopted triage-family protocol ("every customer issue gets an owner").
TICKET_TRIAGE_DEADLINE_TICKS = 24

def _tick_value(v, fallback: int) -> int:
    """Tick with a missing-value fallback. 0 is a legitimate tick, NOT a missing
    value — a falsy-zero fallback (``v or tick``) here made tick-0 objects read
    as created-now forever, so every latency-gated loop was unreachable for them."""
    return int(v) if v is not None else int(fallback)


def _temporary_team_requested(params: dict, condition_default: bool) -> bool:
    """Whether member-local state resets at each sprint boundary.

    A scenario parameter wins over the environment, and the environment over
    the condition's own default, so the overlay is reproducible from a scenario
    file and still reachable for a one-off run.
    """
    requested = params.get("temporary_team")
    if requested is None:
        requested = os.environ.get("ORG_TEMPORARY_TEAM")
    if requested is None:
        return bool(condition_default)
    if isinstance(requested, str):
        return requested.strip().lower() in ("1", "true", "yes", "on")
    return bool(requested)


class OrgWorld:
    def __init__(self, scenario: DomainScenarioConfig):
        self.scenario = scenario
        params = scenario.params
        self.experiment_condition_explicit = bool(
            params.get(
                "experiment_condition_explicit",
                "experiment_condition" in params,
            )
        )
        self.condition_spec = resolve_condition(params.get("experiment_condition"))
        self.experiment_condition = self.condition_spec.condition_id
        self.experiment_mode = str(params.get("experiment_mode", "dev")).strip().lower()
        self.profile_conditioning_enabled = (
            self.condition_spec.profile_conditioning_enabled
        )
        self.institutionalization_enabled = (
            self.condition_spec.institutionalization_enabled
        )
        self.capability_learning_enabled = (
            self.condition_spec.capability_learning_enabled
        )
        # The sprint reset is an overlay rather than a rung: any condition can
        # carry it, so the persistence claim stays testable without a ladder
        # slot. The condition's own value is the default; a scenario parameter
        # or ORG_TEMPORARY_TEAM turns it on elsewhere.
        self.temporary_team_enabled = _temporary_team_requested(
            params, self.condition_spec.temporary_team
        )
        # P0's claim is "no stable role differences". Role LABELS must survive —
        # at least ten governance gates read them to decide who may approve a
        # release, publish, merge or speak externally, and collapsing them made
        # P0 an amputated organization rather than a homogeneity control. But the
        # per-role MANDATE text is a behavioural prior, not governance topology:
        # under homogeneous assignment all eight agents were still handed eight
        # different mandates ("you own fast implementation" vs "you own
        # reproducibility"), which is exactly the differentiation P0 removes
        # everywhere else. The two uses of role are separated here.
        self.role_mandates_enabled = self.condition_spec.role_mandates_enabled
        requested_sprint_ticks = int(
            params.get(
                "baseline_sprint_ticks",
                self.condition_spec.default_sprint_ticks,
            )
        )
        self.baseline_sprint_ticks = max(1, requested_sprint_ticks)
        self.baseline_epoch = 0
        self._baseline_epoch_start_tick = 0
        self.baseline_reset_ticks: List[int] = []
        self.baseline_epoch_archives: List[dict] = []
        self.baseline_archived_action_log: List[dict] = []
        self.baseline_archived_policy_trace: List[dict] = []
        self.baseline_archived_sandbox_jobs: List[Any] = []
        self.baseline_archived_sandbox_results: List[Any] = []
        condition_suffix = (
            f"_{self.condition_spec.short_name}"
            if self.experiment_condition_explicit
            else ""
        )
        self.run_id = f"org_{scenario.name}{condition_suffix}_{scenario.seed}"
        self.world_tick = 0
        self._rng = random.Random(scenario.seed)
        self.agents: Dict[str, OrgAgent] = {}
        self.personal: Dict[str, PersonalWorkspace] = {}
        self.company = CompanyWorkspace()
        self.comm = CommunicationSystem()
        self.meeting_system = MeetingSystem(self.comm)
        self.repo_system = RepoLiteSystem()
        self.sandbox_system = SandboxSystem()
        self.datasets: Dict[str, Dataset] = {}
        self.benchmarks: Dict[str, BenchmarkScenario] = {}
        self.search_system = SearchSystem(self, corpus_version=scenario.corpus_version)
        self.budget_system = BudgetSystem()
        self.protocol_registry = ProtocolRegistry()
        self.time = TimeSystem()
        # internal entities
        self.tasks: Dict[str, Task] = {}
        self.issues: Dict[str, Issue] = {}
        self.documents: Dict[str, Document] = {}
        self.experiments: Dict[str, Experiment] = {}
        self.budgets: Dict[str, ComputeBudget] = {}
        self.tickets: Dict[str, CustomerTicket] = {}
        self.trials: Dict[str, CustomerTrial] = {}   # v14 P5: external market-validation trials
        self.artifacts: Dict[str, WorkflowArtifact] = {}
        self.protocols: Dict[str, Protocol] = {}
        self.board: SharedBoard = SharedBoard(board_id="main")
        self.community = ExternalCommunity(
            corpus_version=scenario.corpus_version, seed=scenario.seed)
        self.events: List[dict] = []
        self.messages: List[dict] = []
        self.public_records: List[dict] = []
        # Frozen OSS workload selected by the scenario.
        self.product = None
        self.product_artifacts: Dict[str, "object"] = {}
        self.patches: Dict[str, Any] = {}              # v4 §2: concrete doc/code patches
        self.company_config: Dict[str, Any] = {}
        # Internal Pipeline spec #1: status-bearing gap registry + derived readiness,
        # kept consistent by the StateReconciler.
        self.known_gaps: Dict[str, Any] = {}
        self.product_readiness: Dict[str, Any] = {}
        self.institution_context: Dict[str, Any] = {}      # spec #4: live adopted protocols/tools
        self.pending_jobs: List[Dict[str, Any]] = []        # spec #6: split work units / async jobs
        self.company_skills: List[Dict[str, Any]] = []      # spec #8: institutionalised mechanisms
        self._rc_blocker_sig = None                          # #2: detect when release blockers change
        self._rc_blockers_changed_tick = 0
        self._last_readiness_tick: Dict[str, int] = {}
        self._reconciler = None
        self._reconcile_warnings: List[str] = []
        # O1 lived decision loop state
        self.memory: Dict[str, List[dict]] = {}
        self.action_log: List[dict] = []
        # v4 §3: policy attractor guard + explainability trace
        from environments.org_env.policy.attractor_guard import AttractorGuard
        self._attractor_guard = AttractorGuard()
        self._attractor_masked: Dict[str, List[dict]] = {}
        self.policy_trace: List[dict] = []
        # Internal Growth Module (§13): event-grounded skill/reputation/authority growth.
        from environments.org_env.growth import GrowthAppraiser, GrowthReconciler
        self._growth_appraiser = GrowthAppraiser()
        self._growth_reconciler = GrowthReconciler()
        self._growth_signals: List[Any] = []
        self.growth_events: List[Any] = []
        self._authority_t0: Dict[str, Dict[str, float]] = {}   # baseline for AuthorityShift
        self.appraised_log: List[dict] = []
        self.detector_summary: Dict[str, Any] = {}
        self.policy_mode: str = str(scenario.params.get("policy_mode", "mock"))
        from environments.org_env.experiments.ablations import (
            WORK_RHYTHM,
            resolve_mechanism_ablations,
        )
        from environments.org_env.experiments.resources import initialize_world_resource_control
        self.mechanism_ablations = resolve_mechanism_ablations(
            scenario.params.get("mechanism_ablations")
        )
        self.time.rhythm_enabled = not self.mechanism_ablations.is_disabled(WORK_RHYTHM)
        initialize_world_resource_control(
            self, scenario.params.get("frozen_resource_budget")
        )
        self._experiment_resource_actor = None
        # O1.7 policy-grounded text: commitments / disputes / requests created by text
        from environments.org_env.backend.commitments import CommitmentRegistry
        self.commitment_registry = CommitmentRegistry()
        # O1.7 analysis logs (appraisal / feedback decision / text generation)
        self.object_appraisal_log: List[dict] = []
        self.feedback_decision_log: List[dict] = []
        self.text_generation_log: List[dict] = []
        # OrgEpisode layer (causal organizational episodes from the event stream)
        from environments.org_env.episodes import OrgEpisodeManager
        self.episode_manager = OrgEpisodeManager()
        self._ep_mark = 0
        # Reflection / Wish layer (agents reflect on episodes -> wishes; durable memory)
        from environments.org_env.reflection import ReflectionManager
        self.reflection_manager = ReflectionManager()
        from environments.org_env.reflection.batch_manager import ReflectionBatchManager
        self.reflection_batch_manager = ReflectionBatchManager()
        self.agent_memories: Dict[str, "object"] = {}
        self.agent_log: List["object"] = []
        self._reflected_episode_ids: set = set()
        # The condition owns WHAT-action selection. B3 always uses profile_policy;
        # B0/B1/B2 use llm_direct whenever a client is attached. A client-free
        # non-formal baseline uses the explicitly-labelled flat deterministic test
        # fallback; a formal baseline without a client is rejected before stepping.
        self.llm_client = None
        self._legacy_llm_decides_actions_requested = False
        self.action_decisions: List["object"] = []
        self._llm_fallbacks = 0
        self._llm_policy = None
        self._action_validator = None
        # Proposal / Tool / Protocol layer (LLM drafts -> validate -> approve -> adopt)
        from environments.org_env.proposals import ProposalManager
        self.proposal_manager = ProposalManager()
        self.episode_summaries: Dict[str, dict] = {}
        # Baselines preserve evaluator logging but cannot create organization-level
        # capabilities through the proposal/adoption path.
        self.auto_propose = self.institutionalization_enabled
        self.auto_approve = self.institutionalization_enabled
        # approval_mode (governance friction, §13.x):
        #   "auto"      — debug: system adopts any proposal with score>=THRESH (auto_approve gate)
        #   "agent"     — realistic: designated approver agents must explicitly approve_proposal
        #                 / reject_proposal via their action pipeline (persona-driven; no auto)
        #   "semi_auto" — agent-driven, but a high-score proposal left undecided past
        #                 APPROVAL_DEADLOCK_TICKS is auto-adopted so the sim never stalls
        self.approval_mode = "auto"
        self._proposed_wish_ids: set = set()
        self._cog = None               # lazily-wired LLM cognitive generators
        self._loop = None          # lazily-wired O1 adapters (see _wire_loop)
        self.event_graph = None

    @property
    def action_selection_mode(self) -> str:
        """Return the effective, auditable WHAT-action selection mode."""
        configured = self.condition_spec.action_selection_mode
        if configured == ACTION_SELECTION_LLM_DIRECT and self.llm_client is None:
            return ACTION_SELECTION_FLAT_DETERMINISTIC
        return configured

    @property
    def llm_decides_actions(self) -> bool:
        """Compatibility surface; the condition, not an override flag, is authoritative."""
        return (
            self.condition_spec.action_selection_mode == ACTION_SELECTION_LLM_DIRECT
            and self.llm_client is not None
        )

    @llm_decides_actions.setter
    def llm_decides_actions(self, requested: bool) -> None:
        # Keep the legacy request for checkpoint/debug visibility, but deliberately
        # do not let ORG_LLM_ACTIONS or YAML change the condition assignment.
        self._legacy_llm_decides_actions_requested = bool(requested)

    def ensure_action_selection_ready(self) -> None:
        """Fail closed when an explicit formal B0--B3 arm has no usable LLM."""
        if (
            self.experiment_mode == "formal"
            and self.experiment_condition_explicit
            and self.llm_client is None
        ):
            mode = self.condition_spec.action_selection_mode
            raise RuntimeError(
                "formal B0-B3 experiment requires an LLM client; "
                f"action_selection_mode={mode!r} remains condition-owned"
            )

    # -- O-Infra-1 seeding --------------------------------------------------
    def build(self) -> "OrgWorld":
        n = int(self.scenario.params.get("num_internal_agents", len(SEED_TEAM)))
        team = roster_for_condition(
            self.condition_spec,
            SEED_TEAM,
            requested_size=n,
            condition_explicit=self.experiment_condition_explicit,
            seed=self.scenario.seed,
        )
        for member in team:
            aid = member.agent_id
            self.agents[aid] = OrgAgent.from_seed(member)
            self.personal[aid] = PersonalWorkspace(agent_id=aid, sandbox_id=f"sandbox_{aid}")
            self.company.add_member(aid)
            # founders defer salary more easily; early members are cash-sensitive (§53)
            self.budget_system.register_agent(CompensationProfile(
                agent_id=aid, is_founder=member.is_founder,
                base_salary=30.0 if member.is_founder else 25.0,
                financial_tolerance=0.85 if member.is_founder else 0.4,
                cash_need_level=0.2 if member.is_founder else 0.6,
                belief_in_mission=0.8 if member.is_founder else 0.55))
            self.time.register_agent(AgentAvailability(
                agent_id=aid,
                after_hours_responsiveness=float(member.work_rhythm.get("after_hours_responsiveness", 0.4)),
                weekend_work_tendency=float(member.work_rhythm.get("weekend_work_tendency", 0.3)),
                deep_work_preference=float(member.work_rhythm.get("deep_work_preference", 0.5)),
                meeting_tolerance=float(member.work_rhythm.get("meeting_tolerance", 0.5))))
        # Endorsement thresholds follow the roster the condition actually has,
        # so a one-person rung is not disqualified by arithmetic before it acts.
        self.protocol_registry.min_supporters = effective_min_supporters(
            len(self.agents)
        )
        # seed default channels with all members (§16)
        self.comm.seed_default_channels(set(self.agents.keys()))
        # Each member receives a sandbox. Experiments are workload-driven; the
        # runtime does not inject an unrelated synthetic benchmark.
        for aid in self.agents:
            self.sandbox_system.ensure_sandbox(aid)

        # Frozen OSS product substrate first (starter repo/docs/issues/eval),
        # so the backlog can link tasks to real artifacts/issues (preflight §1/§2).
        from environments.org_env.product import seed_product
        seed_product(self, self.scenario.params.get("company_config"))
        self._prewarm_product_smoke()           # v14 P0: beta is recognized as runnable at t0

        # Backlog tasks are generated from the workload's agent-visible historical issues.
        substrate_type = getattr(self.product, "substrate_type", "")
        if substrate_type != "oss_time_machine":
            raise RuntimeError("formal Relic world requires the OSS time-machine substrate")
        task_specs = list(getattr(self, "_oss_seed_tasks", []) or [])
        member_ids = list(self.agents.keys())
        for spec in task_specs:
            tid = spec["task_id"]
            owner = self._rng.choice(member_ids) if self._rng.random() < 0.25 else None
            t = Task(task_id=tid, title=spec["title"], description=spec["description"],
                     status=TaskStatus.OPEN, owner_id=owner, priority=int(spec.get("priority", 3)),
                     linked_issues=list(spec.get("linked_issues", [])),
                     linked_artifacts=list(spec.get("linked_artifacts", [])), visibility="team")
            self.tasks[tid] = t
            self.board.tasks.append(tid)
            if owner:
                self.board.owners[tid] = owner

        # Team-visible summaries never expose reference or hidden evaluator assets.
        _pname = getattr(self.product, "name", None) or "the product"
        seed_docs = [
            (
                "doc_readme",
                f"{_pname} README",
                "doc",
                f"{_pname} — OSS time-machine starter; product overview",
            ),
            (
                "doc_arch",
                f"{_pname} product overview",
                "design_doc",
                (getattr(self.product, "summary", "") or "early runnable OSS release")[:200],
            ),
        ]
        seed_doc_owner = "victor" if "victor" in self.agents else next(iter(self.agents))
        for did, title, ftype, summ in seed_docs:
            f = FileObject(object_id=did, file_type=ftype, title=title, creator_id=seed_doc_owner,
                           visibility=Visibility.TEAM, content_summary=summ, created_tick=0)
            f.provenance.origin_actor_id = seed_doc_owner
            self.company.register_file(f)
            self.documents[did] = Document(doc_id=did, title=title, doc_type=ftype,
                                           author_id=seed_doc_owner, owner_id=seed_doc_owner,
                                           content_summary=summ, visibility="team")

        # initial shared budget (funding tranche 1 arrived). Full CompanyBudget
        # (runway/api_budget/funding) lands in O-Infra-7; here a simple ComputeBudget.
        self.budgets["main"] = ComputeBudget(
            budget_id="main", total_budget=3000, remaining_budget=1000,
            daily_quota=120, approval_required=False)

        # a few external posts (frozen corpus) to seed the feed
        from environments.org_env.experiments.ablations import EXTERNAL_BRIDGE, mechanism_disabled
        if not mechanism_disabled(self, EXTERNAL_BRIDGE):
            self._seed_external_posts()
        self._wire_loop()
        for aid in self.agents:
            self.memory.setdefault(aid, [])
        # Internal Pipeline spec #1: build the status-bearing gap registry + reconciler,
        # then run one pass so initial gap/issue/readiness state is consistent.
        from environments.org_env.product.known_gap import build_known_gaps
        from environments.org_env.runtime_adapter.reconciler import StateReconciler
        self.known_gaps = build_known_gaps(self)
        self._reconciler = StateReconciler()
        self.reconcile(reason="seed")
        # Growth Module (§17.2): derive authority from seed skills + record the t0
        # baseline so AuthorityShift / specialization can be measured later.
        self._growth_reconciler.run(self, 0)
        self._authority_t0 = {aid: dict(a.authority) for aid, a in self.agents.items()}
        return self

    def reconcile(self, reason: str = "") -> dict:
        """Run the StateReconciler (gap/issue status + product readiness). Idempotent;
        called end-of-tick, after a PR merge, and before a release gate (spec #1)."""
        rec = getattr(self, "_reconciler", None)
        return rec.reconcile(self, reason=reason) if rec is not None else {}

    def note_protocol_use(self, keywords, tick: int, obj: str = None):
        """spec #4/#8: credit the adopted protocol that governed an action (release gate,
        readiness check) with a USE, so it accrues evidence and can register as a company
        skill — closes the gap where gate use only hit the live registry, not the spec."""
        pm = getattr(self, "proposal_manager", None)
        if pm is None:
            return None
        for s in pm.protocol_specs.values():
            if s.status != "adopted":
                continue
            blob = f"{s.name} {s.trigger_condition} {s.enforcement_rule}".lower()
            if any(k in blob for k in keywords):
                s.use_count = int(getattr(s, "use_count", 0) or 0) + 1
                s.last_used_tick = tick
                if obj and obj not in s.affected_artifacts:
                    s.affected_artifacts.append(obj)
                # v8g P2: emit the canonical use event so spec.use_count == registry usage ==
                # world events == company-skill evidence (one number everywhere).
                self.events.append({"type": "protocol_use_event", "protocol_id": s.protocol_id,
                                    "tick": tick, "object_id": obj, "auto": True})
                s.use_event_ids.append(f"protocol_use_event@t{tick}")   # v8h P1: same refs everywhere
                self._mirror_protocol_event(s, "use", tick, obj)   # v8f P1a: keep registry in sync
                return s.protocol_id
        return None

    def _mirror_protocol_event(
        self,
        spec,
        kind: str,
        tick: int,
        obj=None,
        agent=None,
        *,
        blocked: bool = False,
        state_impact_ref: str | None = None,
        state_before: dict[str, Any] | None = None,
        state_after: dict[str, Any] | None = None,
        governed_object_before: GovernedObjectSnapshot | None = None,
        governed_object_after: GovernedObjectSnapshot | None = None,
    ) -> None:
        """v8f P1a: keep the live protocol_registry mirror in step with the ProtocolSpec so
        use/enforcement counts are consistent across the spec, institution_context, the
        registry's event lists, AND the company-skill evidence (no 'enforced 1x' on one view
        while enforcement_events is [] on another)."""
        reg = getattr(self, "protocol_registry", None)
        if reg is None:
            return
        mirror = self._registry_mirror_id(spec.protocol_id)
        if mirror not in getattr(reg, "protocols", {}):
            return
        try:
            incident_refs = self.__dict__.setdefault(
                "_mirrored_protocol_violation_refs", {}
            )
            incident_key = (mirror, str(obj or ""), int(tick))
            if kind == "violate":
                event = reg.violate(
                    agent or "system",
                    mirror,
                    tick=tick,
                    context_id=str(obj) if obj else None,
                )
                incident_refs[incident_key] = event.event_id
            elif kind == "enforce":
                violation_event_id = incident_refs.get(incident_key)
                violation_event = next(
                    (
                        event
                        for event in reg.events
                        if event.event_id == violation_event_id
                    ),
                    None,
                )
                enforcer = agent or "organizational_gate"
                if (
                    violation_event is not None
                    and enforcer == violation_event.actor_id
                ):
                    enforcer = "organizational_gate"
                impact_ref = (
                    governed_object_before.object_id
                    if governed_object_before is not None
                    else state_impact_ref or (str(obj) if obj else None)
                )
                reg.enforce(
                    enforcer,
                    mirror,
                    tick=tick,
                    violation_event_id=violation_event_id,
                    state_impact_ref=impact_ref,
                    blocked=bool(blocked),
                    context_id=str(obj) if obj else None,
                    state_before=state_before,
                    state_after=state_after,
                    governed_object_before=governed_object_before,
                    governed_object_after=governed_object_after,
                )
            else:
                reg.use(
                    agent or "system",
                    mirror,
                    tick=tick,
                    context_id=str(obj) if obj else None,
                )
        except Exception:
            pass

    def _registry_mirror_id(self, spec_protocol_id: Any) -> str:
        """The registry id that stands for this spec.

        Normally derived from the spec id by string (``protospec_3`` mirrors as
        ``proto_spec_3``). A transferred protocol cannot be named that way -- it
        would land in the slot a locally formed protocol claims -- so injection
        states the pairing and this reads it first.
        """
        stated = (self.__dict__.get("_protocol_mirror_ids") or {}).get(
            str(spec_protocol_id)
        )
        if stated:
            return str(stated)
        return f"proto_spec_{str(spec_protocol_id).split('_')[-1]}"

    def _spec_behind(self, registry_protocol_id: str):
        """The ProtocolSpec whose registry mirror is this id, or None.

        The two id spaces differ by a underscore (``protospec_3`` mirrors as
        ``proto_spec_3``), so a caller holding the registry id cannot look the
        spec up directly.
        """
        pm = getattr(self, "proposal_manager", None)
        wanted = str(registry_protocol_id or "")
        if pm is None or not wanted:
            return None
        for s in getattr(pm, "protocol_specs", {}).values():
            if self._registry_mirror_id(s.protocol_id) == wanted:
                return s
        return None

    def note_protocol_enforcement(
        self,
        keywords,
        tick: int,
        obj: str = None,
        agent: str = None,
        actions: tuple = (),
        *,
        protocol_id: str | None = None,
        blocked: bool = False,
        state_impact_ref: str | None = None,
        state_before: dict[str, Any] | None = None,
        state_after: dict[str, Any] | None = None,
        governed_object_before: GovernedObjectSnapshot | None = None,
        governed_object_after: GovernedObjectSnapshot | None = None,
    ):
        """v8d P1a: credit the adopted protocol that an enforcement moment ENFORCED (a CI
        contract break, a blocked release candidate, an escalated evidence dispute).
        Gives the institution real enforcement events, not just uses.

        Spec selection runs two channels: the legacy keyword-in-blob match, OR the spec's
        own declared affected_actions intersecting the moment's instrument ``actions``
        (the validated action ids whose gate this moment is — run_ci for a CI break,
        run_launch_readiness_check for a blocked release). Keyword tuples only ever
        reached protocols WORDED like the evidence/CI/release families, so a protocol
        that governs the same gate under different wording could never accrue an
        enforcement at any run length.

        Every call site fires on a CAUGHT NON-COMPLIANCE, so the moment records the
        paired violation too (violation -> enforcement is one incident, exactly like the
        governed external-claim path). This establishes enforcement evidence only.
        Strong evidence additionally requires a typed governed-object transition and
        an evaluator-owned outcome oracle through ``note_protocol_outcome``."""
        from environments.org_env.experiments.ablations import (
            PROTOCOL_ENFORCEMENT,
            mechanism_disabled,
        )
        if mechanism_disabled(self, PROTOCOL_ENFORCEMENT):
            return None
        pm = getattr(self, "proposal_manager", None)
        if pm is None:
            return None
        instrument = {str(a).strip().lower().replace(" ", "_").replace("-", "_")
                      for a in (actions or ()) if str(a).strip()}
        # A caller that knows which rule bit says so, and is believed. Guessing by
        # keyword credits whichever adopted spec matches first, and a rule whose
        # text is long and general matches nearly any moment: over one run all 193
        # enforcements landed on the first-adopted rule, including the 26 refusals
        # a different rule had actually made. Whose rule does the work is the
        # measurement here, so it cannot rest on dict order.
        named = self._spec_behind(protocol_id) if protocol_id else None
        candidates = ([named] if named is not None
                      else list(pm.protocol_specs.values()))
        for s in candidates:
            if s.status != "adopted":
                continue
            blob = f"{s.name} {s.trigger_condition} {s.enforcement_rule}".lower()
            declared = s.declared_action_ids() if hasattr(s, "declared_action_ids") else set()
            if (named is not None
                    or any(k in blob for k in keywords)
                    or (instrument and declared & instrument)):
                s.violation_count = int(getattr(s, "violation_count", 0) or 0) + 1
                s.enforcement_count = int(getattr(s, "enforcement_count", 0) or 0) + 1
                if obj and obj not in s.affected_artifacts:
                    s.affected_artifacts.append(obj)
                self.events.append({"type": "protocol_violation_event", "protocol_id": s.protocol_id,
                                    "tick": tick, "object_id": obj, "agent_id": agent, "auto": True})
                self.events.append({"type": "protocol_enforcement_event", "protocol_id": s.protocol_id,
                                    "tick": tick, "object_id": obj, "agent_id": agent, "auto": True})
                s.violation_event_ids.append(f"protocol_violation_event@t{tick}")      # v8h P1
                s.enforcement_event_ids.append(f"protocol_enforcement_event@t{tick}")  # v8h P1
                self._mirror_protocol_event(s, "violate", tick, obj, agent)   # impact needs >=1 violation
                self._mirror_protocol_event(
                    s,
                    "enforce",
                    tick,
                    obj,
                    agent,
                    blocked=blocked,
                    state_impact_ref=state_impact_ref,
                    state_before=state_before,
                    state_after=state_after,
                    governed_object_before=governed_object_before,
                    governed_object_after=governed_object_after,
                )
                return s.protocol_id
        return None

    def note_protocol_outcome(
        self,
        protocol_id: str,
        observation: IndependentOutcomeOracle,
    ) -> None:
        """Attach evaluator-owned outcome evidence to a governed transition."""

        self.protocol_registry.record_independent_outcome(
            protocol_id,
            observation,
        )

    def _enforce_ticket_triage_backstop(self, tick: int) -> Optional[str]:
        """Customer-family enforcement moment. The CI-break / blocked-release /
        escalated-dispute moments give their families enforcement producers, but no
        sweep ever read ``tickets`` — so a triage-family protocol's enforcement_count
        could never leave zero at any run length. An open ticket that has gone a full clock day
        with no owner and no response IS the caught non-compliance the canonical
        triage rule names ("every customer issue gets an owner + a triage record").
        Oldest ticket first, at most one moment per tick, once per ticket; a ticket
        that goes overdue before any triage protocol is adopted stays eligible until
        one is."""
        tickets = self.__dict__.get("tickets") or {}
        if not tickets:
            return None
        seen = self.__dict__.setdefault("_ticket_first_seen", {})
        done = self.__dict__.setdefault("_ticket_protocol_enforced", set())
        overdue = []
        for tid, t in tickets.items():
            def _get(k, d=None, _t=t):
                return _t.get(k, d) if isinstance(_t, dict) else getattr(_t, k, d)
            if _get("status", "open") not in ("open", "in_progress", None):
                continue
            if _get("response_status", "pending") not in ("pending", None):
                continue
            if _get("assigned_agent_id"):
                continue
            first = _tick_value(_get("created_tick"), seen.setdefault(str(tid), int(tick)))
            if str(tid) in done:
                continue
            if int(tick) - first >= TICKET_TRIAGE_DEADLINE_TICKS:
                overdue.append((first, str(tid)))
        if not overdue:
            return None
        first, tid = min(overdue)
        pid = self.note_protocol_enforcement(
            ("customer", "triage", "ticket", "support", "inbound"),
            tick, obj=tid, agent="triage_backstop",
            actions=("create_customer_triage_sheet",),
            blocked=False)
        if pid is None:
            return None                       # no adopted triage-family protocol yet
        done.add(tid)
        self.events.append({"type": "governance_event", "subtype": "ticket_triage_backstop",
                            "ticket_id": tid, "protocol_id": pid, "tick": int(tick),
                            "overdue_ticks": int(tick) - first, "auto": True})
        return pid

    def _wire_loop(self) -> None:
        """Lazily wire the O1 lived-loop adapters (imported here to avoid a
        runtime_adapter <-> simulation import cycle)."""
        from environments.org_env.runtime_adapter.appraisal import OrgEventAppraisalImpl
        from environments.org_env.runtime_adapter.event_graph import OrgEventGraph
        from environments.org_env.runtime_adapter.execution import OrgActionMapper
        from environments.org_env.experiments.ablations import PROFILE_POLICY, mechanism_disabled
        from environments.org_env.experiments.controlled_execution import (
            ExperimentControlledExecutionAdapter,
        )
        from environments.org_env.runtime_adapter.feature_extractor import OrgFeatureExtractor
        from environments.org_env.runtime_adapter.perception import OrgPerceptionAdapter
        from environments.org_env.runtime_adapter.policy import OrgPolicy
        from environments.org_env.runtime_adapter.routine import RoutineScheduler
        self._loop = {
            "perception": OrgPerceptionAdapter(),
            "mapper": OrgActionMapper(),
            "features": OrgFeatureExtractor(),
            "policy": OrgPolicy(
                mode=(
                    "argmax"
                    if self.condition_spec.action_selection_mode
                    == ACTION_SELECTION_LLM_DIRECT
                    else self.policy_mode
                ),
                use_profile_conditioning=(
                    self.profile_conditioning_enabled
                    and not mechanism_disabled(self, PROFILE_POLICY)
                ),
            ),
            "execution": ExperimentControlledExecutionAdapter(),
            "appraisal": OrgEventAppraisalImpl(),
            "routine": RoutineScheduler(),
        }
        # O1.7 object appraiser + text generation kernel/validator (LLM-free by
        # default; inject an LLMEngine via self.text_engine for the LLM path).
        from environments.org_env.runtime_adapter.object_appraisal import ObjectAppraiser
        from environments.org_env.runtime_adapter.text_layer import TextGenerationKernel, TextValidator
        engine = getattr(self, "text_engine", None)
        self._loop["object_appraiser"] = ObjectAppraiser(engine=engine, use_llm=engine is not None)
        self._loop["text_kernel"] = TextGenerationKernel(engine=engine)
        self._loop["text_validator"] = TextValidator()
        self.event_graph = OrgEventGraph(
            agent_ids=set(self.agents.keys()),
            strict_edges=(
                str(os.environ.get("ORG_OSS_MODE", "")).strip().lower()
                == "formal"
            ),
        )
        # LLM action-decision policy + validator (used only when self.llm_client set)
        from environments.org_env.llm.action_decision import ActionValidator, LLMActionPolicy
        self._llm_policy = LLMActionPolicy()
        self._action_validator = ActionValidator()
        # LLM cognitive generators (proposal / tool / institution / episode summary)
        from environments.org_env.llm.episode_summarizer import EpisodeSummarizer
        from environments.org_env.llm.institution_synthesizer import InstitutionSynthesizer
        from environments.org_env.llm.proposal_evaluator import ProposalEvaluator
        from environments.org_env.llm.proposal_generator import ProposalGenerator
        from environments.org_env.llm.tool_composer import ToolComposer
        self._cog = {"proposal_gen": ProposalGenerator(), "proposal_eval": ProposalEvaluator(),
                     "tool_composer": ToolComposer(), "institution": InstitutionSynthesizer(),
                     "episode_summarizer": EpisodeSummarizer()}

    def _seed_external_posts(self) -> None:
        # A small but heterogeneous external community: experts / customers / competitors /
        # investor / engineer / influencer with varied topic interests + activity. They chat about
        # the whole field (eval infra, api cost, benchmarks, debugging, hiring, funding, rival
        # launches) — the company's product is only ONE topic in this forum. Ambient organic posts
        # are generated each tick by community.tick(); controlled interventions add pressure on top.
        #
        # None of it belongs on a substrate that brought its own work. These
        # personas discuss agent-eval infrastructure, which is the synthetic
        # product; on an OSS pack they are a forum for a company that does not
        # exist here, and their chatter becomes issues, and the issues become
        # tasks. Measured on mini_blobstore at t120: 11 of 19 tasks came from
        # this forum and they took 41% of all effort spent, against a score that
        # reads 35 hidden cases and nothing else. A pack that ships its own
        # external signals still gets them; they are loaded separately.
        from environments.org_env.product.substrates.eval_assets import is_oss_substrate
        if is_oss_substrate(self):
            return
        seeds = [
            # (id, name, role, [topic_interests], activity_level, first_post)
            ("ext_expert_1", "Dr. Hua", "expert", ["eval_infra", "reproducibility_tracking"], 0.6,
             "Reliable agent eval needs reproducible traces + cost accounting."),
            ("ext_expert_2", "Prof. Lin", "expert", ["benchmark_quality", "agent_reliability"], 0.5,
             "Single aggregate scores hide failure modes — publish per-task, human-checked benchmarks."),
            ("ext_customer_1", "Acme Corp", "customer", ["customer_pain", "agent_reliability"], 0.5,
             "We can't tell why our agent runs fail — need better failure modes."),
            ("ext_customer_2", "Brightleaf", "customer", ["api_cost", "customer_pain"], 0.45,
             "Eval tooling is useful but the API spend adds up fast for a small team."),
            ("ext_competitor_1", "EvalRivals", "competitor", ["competitor_update", "benchmark_quality"], 0.6,
             "Launching cheaper eval tier next month."),
            ("ext_competitor_2", "ProbeAI", "competitor", ["competitor_update", "trace_debugging"], 0.4,
             "Our trace debugger now replays any failed agent run step-by-step."),
            ("ext_investor_1", "Northstar VC", "investor", ["startup_funding"], 0.3,
             "Watching the agent-eval space — durable moats come from trust, not demos."),
            ("ext_engineer_1", "Kenji", "engineer", ["trace_debugging", "api_cost"], 0.5,
             "Caching embeddings + batching eval calls cut our cost ~4x with no quality loss."),
            ("ext_influencer_1", "DevPulse", "influencer", ["agent_reliability", "hiring_market"], 0.7,
             "Hot take: 'agent reliability' is the whole game in 2026. Everything else is demos."),
        ]
        for pid_a, name, role, topics, activity, text in seeds:
            self.community.profiles[pid_a] = ExternalProfile(
                external_agent_id=pid_a, name=name, role=role,
                topic_interests=list(topics), credibility=0.7, activity_level=activity)
            post_id = f"post_{pid_a}"
            self.community.posts[post_id] = Post(
                post_id=post_id, author_id=pid_a, topic=topics[0],
                content_summary=text, credibility=0.7, reach=self._rng.randint(50, 500),
                created_tick=0)

    # -- snapshot -----------------------------------------------------------
    def get_state(self) -> DomainState:
        from environments.org_env.experiments.resources import experiment_resource_snapshot
        resource_snapshot = experiment_resource_snapshot(self)
        return DomainState(
            run_id=self.run_id,
            world_tick=self.world_tick,
            agents={aid: a.domain_state() for aid, a in self.agents.items()},
            entities={},   # rich DomainEntity mapping in the adapter (later stages)
            events=list(self.events),
            messages=list(self.messages),
            public_records=list(self.public_records),
            domain_clock={"tick": self.world_tick},
            domain_metadata={"scenario": self.scenario.name,
                             "corpus_version": self.scenario.corpus_version,
                             "experiment_condition": self.experiment_condition,
                             "action_selection_mode": self.action_selection_mode,
                             "baseline_epoch": self.baseline_epoch,
                             "company": self.company.company_name,
                             "num_tasks": len(self.tasks),
                             "num_unowned_tasks": sum(1 for t in self.tasks.values() if not t.owner_id),
                             "mechanism_ablations": self.mechanism_ablations.to_dict(),
                             "mechanism_ablation_fingerprint": self.mechanism_ablations.fingerprint,
                             "experiment_resources": resource_snapshot},
        )

    # -- O-Infra-1 visibility helper (info asymmetry, §19/§70) -------------
    def visible_files_for(self, agent_id: str) -> List[FileObject]:
        """Company files this agent can see + their own private local files."""
        seen = list(self.company.visible_files_for(agent_id))
        pw = self.personal.get(agent_id)
        if pw:
            seen.extend(pw.local_files.values())
        return seen

    def _maybe_reset_temporary_team(self, tick: int) -> bool:
        """Re-instantiate B1 members exactly once per configured sprint."""
        if not self.temporary_team_enabled or tick <= 0:
            return False
        if tick % self.baseline_sprint_ticks != 0:
            return False
        ledger = getattr(self, "experiment_resource_ledger", None)
        max_ticks = getattr(getattr(ledger, "budget", None), "max_ticks", None)
        if max_ticks is not None and tick >= int(max_ticks):
            return False
        if tick in self.baseline_reset_ticks:
            return False
        from environments.org_env.runtime_adapter.baseline_reset import (
            reset_temporary_specialist_team,
        )

        reset_temporary_specialist_team(self, tick)
        return True

    def step(self) -> dict:
        """One-hour lived-agent tick (DESIGN env_org §13.1). Advances the clock,
        runs infra bookkeeping, then for each agent that *can act*:
        perception -> candidates -> condition-owned WHAT selector -> execute ->
        appraise -> log / memory / event-graph / protocol-detectors. No scripted
        work sessions — every event comes from the decision loop."""
        from environments.org_env.experiments.resources import (
            ExperimentResourceExhausted,
            TICKS,
            reserve_world_resources,
        )
        self.ensure_action_selection_ready()
        if not reserve_world_resources(self, {TICKS: 1}, detail="world_step"):
            raise ExperimentResourceExhausted("experiment_resource_exhausted:ticks")
        if self._loop is None:
            self._wire_loop()
        clk = self.time.clock
        clk.advance(1)
        self.world_tick = clk.current_tick
        tick = clk.current_tick
        self._maybe_reset_temporary_team(tick)
        self._ep_mark = len(self.events)        # episode layer: mark new-events window
        self._update_milestone()                # v13: persist work-mode stage (drives shipping mode)

        # -- scheduled / infra processing ----------------------------------
        self.process_funding_checkpoint(tick)          # spec #5: milestone-gated funding decisions
        if tick == self.budget_system.payroll.next_payroll_tick:
            self.budget_system.run_payroll(tick)
        if clk.hour_in_day == 0 and tick > 0:
            self.budget_system.charge(agent_id="company", action_type="daily_burn",
                                      base_cost=self.budget_system.budget.daily_base_burn,
                                      resource_type="service", tick=tick)
        from environments.org_env.experiments.ablations import (
            EVENT_GRAPH,
            EXTERNAL_BRIDGE,
            mechanism_disabled,
        )
        if not mechanism_disabled(self, EXTERNAL_BRIDGE):
            self.community.tick(tick)              # living forum: organic field chatter each tick
            self._apply_external_events(tick)      # controlled interventions (pressure) on top
            if getattr(self, "external_society", None) is not None or os.environ.get("ORG_EXTERNAL_SOCIETY"):
                from environments.org_env.external_society.bridge import maybe_run_external_society
                maybe_run_external_society(self, tick)   # opt-in LLM external society (story §6)
        if "_oss_issue_stream" in self.__dict__:    # OSS time-machine: release historical issues as
            from environments.org_env.product.substrates.issue_stream import release_due_issues
            release_due_issues(self, tick)          # external pressure mirroring the real timeline (§9)
            # close the issue->task/patch->hidden-test->ticket loop after a release (brief review §6).
            # Gated on ORG_OSS_HIDDEN_TESTS + de-duped per product-tree hash (off in unit tests).
            rels = getattr(getattr(self.repo_system, "repo", None), "releases", {}) or {}
            if rels:
                from environments.org_env.product.substrates.eval_assets import oss_post_release_evaluation
                oss_post_release_evaluation(self, tick)
            # universal backlog + renewal (self-iteration): make EVERY open issue (any type/source)
            # workable via the normal task loop, and file a renewal issue when otherwise idle so the
            # org keeps iterating instead of stalling once the seeded issue stream is exhausted.
            self._reconcile_issue_backlog(tick)
            self._reconcile_failing_test_backlog(tick)
            self._renew_backlog_if_idle(tick)
        # O1.6: complete due background jobs; begin-of-day partial reset; run the
        # meeting lifecycle (start due meetings -> block participants -> close).
        self._process_background_jobs(tick)
        if clk.hour_in_day == 0 and tick > 0:
            self._start_new_day_all(tick)
            if self.capability_learning_enabled:
                self._growth_reconciler.run(self, tick)    # §13: daily skill/reputation/authority
        self._process_meetings(tick)

        # -- per-agent lived decision loop ---------------------------------
        from environments.org_env.runtime_adapter.delivery_funnel import (
            note_choice as note_delivery_choice,
            record_availability as record_delivery_availability,
        )
        loop = self._loop
        for aid, agent in self.agents.items():
            ws = getattr(agent, "work_state", None)
            if ws is not None:                       # aux-speech slots reset each tick
                ws.aux_speech_slots_remaining = ws.default_aux_speech_slots
            # busy on a blocking/background activity: no new primary this tick. If
            # busy *because in an active meeting*, only meeting sub-actions are
            # allowed (record notes / decisions / action items) — §13.
            if ws is not None and ws.is_busy(tick):
                m = self._active_meeting_for(aid, ws)
                if m is not None:
                    self._run_meeting_subaction(aid, agent, m, tick)
                else:
                    self._maybe_reactive_aux_speech_while_busy(aid, agent, tick)
                continue
            if ws is not None and ws.current_activity_id is not None and not ws.is_busy(tick):
                ws.clear_activity()                  # activity finished -> free
            if not self.can_agent_act(agent, clk):
                if clk.phase_of_day == "night_sleep" or (self.time.availability.get(aid)
                                                         and self.time.availability[aid].current_availability_status == "resting"):
                    self.time.rest(agent)
                continue
            self._process_inbox(aid, tick)             # v14 P3: read/ack inbox -> memory, before deciding
            perception = loop["perception"].build_perception(aid, self, tick)
            candidates = loop["mapper"].to_core_candidates(perception, self)
            candidates = self._constrain_candidates(agent, candidates, clk)
            # Recorded before the choice and kept even when no action follows: a
            # decision point where commit_patch was on the menu and nothing was
            # executed is evidence about the chooser, and it is the point a
            # reader asks about when a condition delivers nothing.
            availability = record_delivery_availability(self, aid, tick, candidates)
            if not candidates:
                continue
            scored = []
            for c in candidates:
                feats = loop["features"].extract(c, perception, self)
                feats = loop["routine"].modify_features(agent, c, feats, clk, self)
                scored.append((c, feats))
            rng = random.Random(f"{self.scenario.seed}:{tick}:{aid}")
            selection_mode = self.action_selection_mode
            if selection_mode == ACTION_SELECTION_LLM_DIRECT:
                # Traditional LLM chooses only from the constrained candidate pool.
                # Invalid/error output is fail-closed for this action: never route it
                # through the profile policy and never execute unvalidated output.
                self._experiment_resource_actor = aid
                try:
                    selected = self._llm_decide(aid, agent, perception, candidates)
                finally:
                    self._experiment_resource_actor = None
                if selected is None:
                    continue
            elif selection_mode in (
                ACTION_SELECTION_PROFILE_POLICY,
                ACTION_SELECTION_FLAT_DETERMINISTIC,
            ):
                selected = loop["policy"].select(agent, perception, scored, self, rng=rng)
            else:
                raise RuntimeError(
                    f"unsupported action_selection_mode={selection_mode!r}"
                )
            if selection_mode == ACTION_SELECTION_LLM_DIRECT and self.action_decisions:
                # LLM-accepted: still trace + join (§4).
                d = self.action_decisions[-1]
                loop["policy"].trace_choice(agent, self, tick, scored, selected,
                                            decision_id=d.decision_id, source=d.decision_source)
            if selected is None:
                continue
            note_delivery_choice(self, availability, selected.action_type)
            result = loop["execution"].execute(aid, selected, self)
            if self.action_decisions and self.action_decisions[-1].agent_id == aid \
                    and self.action_decisions[-1].validation_status == "accepted" \
                    and self.action_decisions[-1].executed_event_id is None:
                self.action_decisions[-1].executed_event_id = result.action_id
            appraised = loop["appraisal"].appraise(result, self)
            self._log_events(result, appraised)
            self._update_memory(aid, appraised)
            if not mechanism_disabled(self, EVENT_GRAPH):
                self.event_graph.ingest(result)
            eps = self.episode_manager.observe_result(result, self)
            self._link_product_artifacts(result, eps, aid)
            self._record_review_evidence(result, aid)
            if self.capability_learning_enabled:
                self._growth_appraiser.collect(result, self, aid)   # buffer growth signals (§2)

        # -- overdue promises -> violation + trust hit (O1.7 §27) ----------
        self._check_commitments(tick)
        # -- overdue untriaged tickets -> customer-family enforcement moment
        self._enforce_ticket_triage_backstop(tick)

        # -- protocol detectors (end of day) -------------------------------
        if clk.hour_in_day == 0 and tick > 0:
            self._run_protocol_detectors(tick)

        # -- episode layer: observe world-level events + age open episodes -
        self._observe_world_episodes()
        self.episode_manager.update_open_episodes(self)
        self._summarize_closed_episodes()

        # -- reflection layer (preflight §3-§6): HALF-DAY BATCH, not per-action.
        #    Every 6 ticks at most a few salient agents reflect. (per-episode +
        #    periodic triggers retired.)
        #
        #    Every condition reflects, on the same cadence and the same budget.
        #    Reflecting on your own work is a capacity an agent has, not an
        #    institution an organization builds, and giving it only to the top
        #    rung would have let extra thinking time masquerade as the effect of
        #    institutionalization. What the top rung adds is downstream: only
        #    there do reflections become wishes, and wishes proposals, and
        #    proposals rules that outlive the moment. See
        #    ReflectionManager.reflect.
        self.reflection_batch_manager.maybe_run_batch(self, tick, self.llm_client)

        # -- cognitive layer: wishes -> proposals; repeated patterns -> institutions;
        #    consensus approval -> tool/protocol adoption (all manager/validator-gated).
        self._process_cognition(tick)

        # spec #1: end-of-tick reconciliation so gap/issue/readiness never lags behind
        # the merges/task-updates that happened this tick.
        self.reconcile(reason="end_of_tick")
        self.events.append({"type": "tick", **clk.snapshot()})
        return clk.snapshot()

    # proposal sparsity (preflight §10.3)
    MAX_NEW_PROPOSALS_PER_BATCH = 2
    MAX_OPEN_PROPOSALS_GLOBAL = 12

    def _wish_ready_for_proposal(self, w) -> bool:
        """A wish is promoted to a proposal only when it is STABLE (§10.2):
        very high urgency / supported by >=2 / spans >=2 episodes / founder-endorsed."""
        if w.urgency >= 0.85:
            return True
        if getattr(w, "support_count", 1) >= 2:
            return True
        if len(getattr(w, "related_episode_ids", [])) >= 2:
            return True
        founders = {a for a, ag in self.agents.items() if getattr(ag, "is_founder", False)}
        if (w.agent_id in founders) or (set(getattr(w, "supporting_agent_ids", [])) & founders):
            return True
        return False

    def _flag_harmful_protocols(self, tick: int) -> Optional[str]:
        """Self-correction trigger: when an ADOPTED protocol is doing more harm than good, file a
        policy-repair WISH so the org can RELAX (or deprecate) its own self-binding rule. The wish
        flows through the normal reflection -> proposal -> amendment path (repair proposals are
        exempt from the protocol cap). Deduped + per-protocol cooldown.

        What counts as harm lives in protocol/harm.py, because a rule that is
        obeyed and still stops all delivery is as harmful as one nobody can
        meet, and only the first of those used to be visible here."""
        pm = getattr(self, "proposal_manager", None)
        rm = getattr(self, "reflection_manager", None)
        if pm is None or rm is None:
            return None
        from environments.org_env.reflection.objects import (
            AgentReflection, Wish, make_wish_fingerprint)
        from environments.org_env.backend.protocol.harm import rule_is_doing_harm
        cd = self.__dict__.setdefault("_policy_repair_cd", {})
        for sid, s in list(getattr(pm, "protocol_specs", {}).items()):
            if getattr(s, "status", "") != "adopted":
                continue
            viol = int(getattr(s, "violation_count", 0) or 0)
            uses = int(getattr(s, "use_count", 0) or 0)
            harmful, why = rule_is_doing_harm(self, s)
            if not harmful:
                continue
            if tick - int(cd.get(sid, -999)) < 48:               # per-protocol cooldown (~2 days)
                continue
            if any(getattr(w, "wish_type", "") == "policy_repair_need" and w.status == "open"
                   and sid in (getattr(w, "related_object_ids", []) or []) for w in rm.wishes.values()):
                continue
            if any(getattr(p, "repair_target_protocol_id", None) == sid
                   and p.status in ("draft", "under_review") for p in pm.proposals.values()):
                continue
            cd[sid] = tick
            founder = next((a for a, ag in self.agents.items() if getattr(ag, "is_founder", False)), None) \
                or (next(iter(self.agents)) if self.agents else "system")
            prob = (f"Adopted protocol '{getattr(s, 'name', '')}' is costing more than it is "
                    f"worth: {why}. Relax it to what we can actually meet, or drop it.")
            wid = f"wish_repair_{sid}_{tick}"
            # trace the wish to a (system) reflection so the "every wish comes from a reflection"
            # invariant holds — the org REFLECTED that its own rule is doing more harm than good.
            rid = f"refl_repair_{sid}_{tick}"
            rm.reflections[rid] = AgentReflection(
                reflection_id=rid, agent_id=founder, tick=int(tick), source_object_ids=[sid],
                trigger_reason="policy_harm", team_assessment=prob,
                perceived_blockers=[f"{getattr(s, 'name', '')} is over-constraining"],
                created_wish_ids=[wid])
            rm.wishes[wid] = Wish(
                wish_id=wid, agent_id=founder, wish_type="policy_repair_need",
                source_reflection_id=rid, source_reflection_ids=[rid],
                missing_support_type="policy_repair", urgency=0.9,
                interpreted_need=f"relax the over-strict protocol {getattr(s, 'name', '')}",
                target_problem=prob, related_object_ids=[sid], status="open",
                created_at_tick=tick, updated_at_tick=tick,
                fingerprint=make_wish_fingerprint("policy_repair_need", prob, [sid]))
            self.events.append({"type": "governance_event", "subtype": "policy_repair_flagged",
                                "protocol_id": sid, "violation_count": viol,
                                "enforcement_count": int(getattr(s, "enforcement_count", 0) or 0),
                                "use_count": uses, "why": why,
                                "wish_id": wid, "tick": int(tick), "auto": True})
            return wid
        return None

    def _repackage_stuck_deliverables(self, tick: int) -> Optional[str]:
        """The delivery half of policy repair: land work a rule has walled off.

        Easing a merge-evidence rule cannot move work the desk already holds; it
        only lets the unfinished surfaces beside it through too. So when a rule
        is judged harmful for blocking delivery AND the desk carries a fix the
        mainline lacks, the repair opens a clean single-issue request from the
        desk and lands it if its own scoped gate is green. Bounded to one issue
        per invocation on a per-issue cooldown, and only while a rule is actually
        blocking, so it costs a scoped CI run only when there is stuck work and a
        rule to blame for it.
        """
        pm = getattr(self, "proposal_manager", None)
        if pm is None:
            return None
        from environments.org_env.backend.protocol.harm import (
            blocked_without_delivery, rule_is_doing_harm)
        adopted = [s for s in getattr(pm, "protocol_specs", {}).values()
                   if getattr(s, "status", "") == "adopted"]
        blocking = any(rule_is_doing_harm(self, s)[0]
                       and blocked_without_delivery(self, s)[0]
                       > blocked_without_delivery(self, s)[1]
                       for s in adopted)
        if not blocking:
            return None
        from environments.org_env.backend.protocol.delivery_repair import (
            landable_desk_deliverables, repackage_and_land, _clean_repackage_exists)
        cd = self.__dict__.setdefault("_delivery_repair_cd", {})
        for issue_id, artifact_ids in landable_desk_deliverables(self):
            if tick - int(cd.get(issue_id, -999)) < 48:
                continue
            if _clean_repackage_exists(self, issue_id):
                continue
            cd[issue_id] = tick
            result = repackage_and_land(self, issue_id, artifact_ids, tick)
            return result.get("pr_id")
        return None

    def _process_cognition(self, tick: int) -> None:
        rm, pm = self.reflection_manager, self.proposal_manager
        from environments.org_env.experiments.ablations import (
            INSTITUTIONALIZATION,
            mechanism_disabled,
        )
        institutionalization_enabled = (
            self.institutionalization_enabled
            and not mechanism_disabled(self, INSTITUTIONALIZATION)
        )
        if institutionalization_enabled:
            self._flag_harmful_protocols(tick)   # self-correction: over-strict rules -> policy-repair wish
            self._repackage_stuck_deliverables(tick)  # the delivery half: land work the gate walled off
        # wishes -> proposals: sparse + half-day cadence + capped (§10).
        if institutionalization_enabled and self.auto_propose and tick > 0 and tick % 6 == 0:
            open_props = sum(1 for p in pm.proposals.values()
                             if p.status in ("draft", "under_review"))
            made = 0
            for w in sorted(rm.wishes.values(), key=lambda w: -w.urgency):
                if made >= self.MAX_NEW_PROPOSALS_PER_BATCH or open_props >= self.MAX_OPEN_PROPOSALS_GLOBAL:
                    break
                if w.status != "open" or w.wish_id in self._proposed_wish_ids:
                    continue
                if not self._wish_ready_for_proposal(w):
                    continue
                self._proposed_wish_ids.add(w.wish_id)
                p = self.propose_from_wish(w.wish_id)
                if p is not None and p.status != "rejected":
                    made += 1
                    open_props += 1
        if institutionalization_enabled and tick > 0 and tick % 24 == 0 and self._cog is not None:
            synth = self._cog["institution"]
            # The catalogue first, then what the members have actually been asking
            # for. The catalogue is four entries deep and each fires once, so on
            # its own it stops speaking a third of the way into a run; the wishes
            # keep arriving. Both arrive as ordinary proposals and face the same
            # review, approval and semantic merge against what is already adopted.
            patterns = list(synth.detect(self))
            patterns.extend(synth.cluster_wishes(self, self.llm_client))
            for pat in patterns:
                p = self._cog["institution"].synthesize(pat, self, self.llm_client)
                self._submit_proposal(p)
                for wid in (pat.get("wish_ids") or []):
                    self._proposed_wish_ids.add(wid)
        # adoption path depends on governance mode (§13.x):
        mode = getattr(self, "approval_mode", "auto")
        if institutionalization_enabled and mode == "auto" and self.auto_approve:
            # debug: blanket consensus by score
            for p in list(pm.proposals.values()):
                if p.status == "under_review" and (p.adoption_score or 0) >= 0.6:
                    for aid in list(p.approval_required_from):
                        pm.approve_proposal(p.proposal_id, aid, self)
        elif institutionalization_enabled and mode == "semi_auto":
            # realistic, but break deadlocks
            for p in list(pm.proposals.values()):
                pending = tick - int(p.updated_at_tick or p.created_at_tick or 0)
                if (p.status == "under_review" and (p.adoption_score or 0) >= 0.6
                        and pending >= APPROVAL_DEADLOCK_TICKS):
                    for aid in list(p.approval_required_from):
                        if aid not in p.approved_by and aid not in p.rejected_by:
                            pm.approve_proposal(p.proposal_id, aid, self)
        # mode == "agent": nothing automatic — approver agents must act explicitly.
        # §7 sweep: adopt any proposal that now has enough approvers AND has cleared
        # the review-latency gate (so nothing adopts the same tick it was proposed).
        # A transfer arm's evaluation window is counted in closed episodes, so
        # it has to be re-checked as episodes close rather than set once at
        # build time. A no-op for every run that inherited nothing.
        from environments.org_env.experiments.capability_transfer import (
            refresh_capability_compilation_freeze,
        )

        refresh_capability_compilation_freeze(self)
        if institutionalization_enabled:
            pm.process_pending_adoptions(self)
        # v6 P0.4 sweep: close the PR review/merge chain that agents left waiting.
        self.process_repo_workflow()
        self._close_tasks_that_meet_their_gate(self.world_tick)
        # spec #3 sweep: advance the internal release-candidate pipeline.
        self.process_release_pipeline()
        # v8f P0a: ship the last-mile release blocker fix if an RC stays blocked too long.
        self.process_release_blockers(self.world_tick)
        # spec #6 sweep: work agents through their split work units.
        self.process_pending_jobs(self.world_tick)
        # #4 sweep: resolve disputes on provided evidence, else escalate to a release blocker.
        self.process_disputes(self.world_tick)
        # v8 Gap2: auto-convene a high-value meeting when a strong trigger persists.
        self.process_meetings(self.world_tick)
        # v8d P2a: close / escalate meeting action items so decisions reach completion.
        self.process_action_items(self.world_tick)

    def set_approval_mode(self, mode: str) -> None:
        """Switch governance mode. 'auto' keeps auto_approve; 'agent'/'semi_auto'
        require designated approvers to act (auto_approve is turned off)."""
        if mode not in ("auto", "semi_auto", "agent"):
            raise ValueError(f"unknown approval_mode '{mode}'")
        self.approval_mode = mode
        self.auto_approve = mode == "auto" and self.institutionalization_enabled

    def propose_from_wish(self, wish_id: str, *, evaluate: bool = True, route: bool = True):
        """Wish -> Proposal draft (LLM/template) -> validate -> evaluate -> route for
        approval. Generic helper; adoption still requires approval (never automatic)."""
        from environments.org_env.experiments.ablations import (
            INSTITUTIONALIZATION,
            mechanism_disabled,
        )
        if (
            not self.institutionalization_enabled
            or mechanism_disabled(self, INSTITUTIONALIZATION)
        ):
            return None
        if self._cog is None:
            self._wire_loop()
        w = self.reflection_manager.wishes.get(wish_id)
        if w is None or w.status != "open":
            return None
        p = self._cog["proposal_gen"].generate(w, self, self.llm_client)
        if p.proposal_type in ("tool_proposal", "workflow_proposal", "artifact_template_proposal"):
            self._cog["tool_composer"].compose(p, self, self.llm_client)
        p = self.proposal_manager.create_proposal(p, self)
        if p.status == "rejected":
            return p
        if evaluate:
            self.proposal_manager.evaluate_proposal(
                p, self, lambda pr, wd: self._cog["proposal_eval"].evaluate(pr, wd, self.llm_client))
        if route:
            self.proposal_manager.route_for_approval(p, self)
        return p

    def _submit_proposal(self, p):
        p = self.proposal_manager.create_proposal(p, self)
        if p.status != "rejected":
            self.proposal_manager.evaluate_proposal(
                p, self, lambda pr, wd: self._cog["proposal_eval"].evaluate(pr, wd, self.llm_client))
            self.proposal_manager.route_for_approval(p, self)
        return p

    def approve_proposal(self, proposal_id: str, agent_id: str):
        return self.proposal_manager.approve_proposal(proposal_id, agent_id, self)

    def _llm_decide(self, aid, agent, perception, candidates):
        """Run the baseline LLM-direct validator gate over the feasible pool.

        ``None`` means no action is executed. It never means "ask the profile
        policy instead", which would contaminate the B0/B1/B2 baselines.
        """
        if (
            self.action_selection_mode != ACTION_SELECTION_LLM_DIRECT
            or self.llm_client is None
            or self._llm_policy is None
        ):
            return None
        from environments.org_env.llm.action_decision import candidate_for_decision
        decision = self._llm_policy.decide(aid, self, perception, candidates, self.llm_client)
        sel = None
        if decision.decision_source == "llm":
            vr = self._action_validator.validate(decision, self, candidates)
            if vr.passed:
                sel = candidate_for_decision(decision, candidates, self)
                decision.validation_status = "accepted" if sel is not None else "rejected"
                if sel is None:
                    decision.rejection_reason = "no matching pool candidate"
            else:
                decision.validation_status = "rejected"
                decision.rejection_reason = vr.reason
        if sel is None:
            self._llm_fallbacks += 1
        self.action_decisions.append(decision)
        # v8 #4: the list is capped for memory, so keep UNCAPPED tallies for telemetry
        # (otherwise `total` and the accept/reject counts plateau at 500 mid-run).
        t = self.__dict__.setdefault("action_decision_tally",
                                     {"total": 0, "accepted": 0, "rejected": 0, "fallback": 0})
        t["total"] += 1
        vs = getattr(decision, "validation_status", "")
        if vs == "accepted":
            t["accepted"] += 1
        elif vs == "rejected":
            t["rejected"] += 1
        if vs in ("rejected", "fallback_used"):
            t["fallback"] += 1
        if len(self.action_decisions) > 500:
            del self.action_decisions[:-500]
        return sel

    def get_patch(self, patch_id: str):
        return self.patches.get(patch_id)

    def _materialize_release(self, version) -> dict:
        """Export the mainline tree as a real, persistent release snapshot + run smoke (best-effort)."""
        try:
            from environments.org_env.product.materialize import export_release_snapshot
            res = export_release_snapshot(self, version)
            return {"dir": (res.get("export") or {}).get("dir"),
                    "smoke_ok": (res.get("smoke") or {}).get("ok")}
        except Exception:
            return {"dir": None, "smoke_ok": None}

    def apply_product_patch(self, patch, res, aid: str) -> bool:
        """v4 §5.3: the ONLY place an existing product artifact's revision is bumped —
        a validated concrete patch (revision + patch history + change summary + causal
        links + awaiting_review). Returns True if applied."""
        art = self.product_artifacts.get(patch.target_object_id)
        if art is None:
            return False
        tick = self.world_tick
        patch.validation_status = "accepted"
        patch.applied_tick = tick                  # v8 #4: record when it entered the artifact
        self.patches[patch.patch_id] = patch
        art.revision += 1
        art.updated_at_tick = tick
        art.patch_history_ids.append(patch.patch_id)
        if patch.change_summary:
            art.change_summaries.append(patch.change_summary)
        # real file content grows here: apply the patch's full new text + record a real diff
        new_content = getattr(patch, "new_content", "") or ""
        if new_content and new_content != (art.content or ""):
            import difflib
            old = (art.content or "").splitlines()
            fp = art.linked_file_path or art.title
            patch.unified_diff = "\n".join(difflib.unified_diff(
                old, new_content.splitlines(),
                fromfile=f"{fp}@r{art.revision - 1}", tofile=f"{fp}@r{art.revision}", lineterm=""))
            art.content = new_content
        art.awaiting_review = True
        art.status = "needs_review"
        # v4 §4: only clear the gap(s) the patch actually resolved (validated against the
        # artifact's real known_gaps), instead of blindly popping the first one.
        resolved = [g for g in (getattr(patch, "resolved_gaps", None) or []) if g in art.known_gaps]
        for g in resolved:
            art.known_gaps.remove(g)
        for tid in getattr(patch, "related_task_ids", []) or []:
            if tid not in art.linked_task_ids:
                art.linked_task_ids.append(tid)
        # v8d P0c: the editor LLM rarely names the task, so the commit -> PR -> RC linkage
        # chain started empty. Infer the owning task(s) from each task's linked_artifacts
        # and reflect them onto BOTH the artifact and the patch (so the commit carries them).
        if not (getattr(patch, "related_task_ids", None) or []):
            for tid, t in self.tasks.items():
                if patch.target_object_id in (getattr(t, "linked_artifacts", []) or []):
                    if tid not in art.linked_task_ids:
                        art.linked_task_ids.append(tid)
                    if tid not in patch.related_task_ids:
                        patch.related_task_ids.append(tid)
        res.modified_objects.append(art.artifact_id)
        res.state_delta["patch_id"] = patch.patch_id
        res.events.append({"type": "product_event", "subtype": "patch_applied",
                           "artifact_id": art.artifact_id, "patch_id": patch.patch_id,
                           "agent_id": aid, "tick": tick,
                           "change_summary": patch.change_summary,
                           "patch_type": getattr(patch, "patch_type", "")})
        res.graph_edges.append((aid, "patched", art.artifact_id))
        # v5 §P0-5: remember the capabilities this patch added, so the policy can mask
        # re-adding an already-built capability (no marginal value).
        for cap in ((getattr(patch, "added_fields", []) or []) + (getattr(patch, "added_checks", []) or [])
                    + (getattr(patch, "checklist_items", []) or [])):
            key = str(cap).strip().lower()[:60]
            if key and key not in art.capabilities:
                art.capabilities.append(key)
        # An accepted patch is a candidate change awaiting a commit, and it belongs
        # to a branch from here: the work item it is for is known now, and working
        # it out later is what the routing layer kept getting wrong.
        from environments.org_env.backend.repo.workflow import record_patch
        record_patch(self, aid, patch, art, tick)
        return True

    def _link_product_artifacts(self, result, eps, aid: str) -> None:
        """Reverse-link product changes to their causal episode/wish/proposal/action
        + advance the linked task (preflight §13.1 + review #4/#5/#6). HARD RULE: any
        touched artifact gets at least linked_action_ids (never an orphan change)."""
        arts = getattr(self, "product_artifacts", {}) or {}
        touched = [o for o in (result.created_objects + result.modified_objects) if o in arts]
        if not touched:
            return
        target_eps = list(eps or [])
        if not target_eps:                       # fallback: an open episode the agent is in
            for ep in self.episode_manager.episodes.values():
                if ep.status == "open" and aid in getattr(ep, "participants", []):
                    target_eps = [ep]
                    break
        if not target_eps:                       # §6: never leave a product change orphaned
            target_eps = [self.episode_manager.ensure_product_episode(result, self)]
        rm = self.reflection_manager
        for oid in touched:
            art = arts[oid]
            if result.action_id not in art.linked_action_ids:
                art.linked_action_ids.append(result.action_id)   # always a causal link
            for ep in target_eps:
                if ep.episode_id not in art.linked_episode_ids:
                    art.linked_episode_ids.append(ep.episode_id)
                if oid not in ep.produced_artifacts:
                    ep.produced_artifacts.append(oid)
            for w in rm.wishes.values():
                if oid in w.related_object_ids:
                    if w.wish_id not in art.linked_wish_ids:
                        art.linked_wish_ids.append(w.wish_id)
                    for pid in w.generated_proposal_ids:
                        if pid not in art.linked_proposal_ids:
                            art.linked_proposal_ids.append(pid)
            self._progress_linked_tasks(oid, art, aid, result)

    # preflight v3 §4: actions that signal awareness but are NOT, by themselves, real progress.
    _WEAK_ACTIONS = {"audit_readme_claims", "monitor_customer_feedback", "read_feed",
                     "internal_search", "review_feed", "observe"}
    _REVIEW_ACTIONS = {"review_doc", "review_pr", "approve_proposal", "request_proposal_changes",
                       "request_changes", "ask_for_evidence", "challenge_result", "adopt_protocol"}

    def _progress_linked_tasks(self, artifact_id: str, art, aid: str, result) -> None:
        """A product change advances the task owning the artifact (review #5), but
        completion now requires real evidence (preflight v3 §4): a single weak action
        can never flip a task to done — it needs a revised artifact, cleared gaps, a
        second party (review / co-contributor), and >=2 substantive evidence events."""
        from environments.org_env.backend.entities import TaskStatus
        tick = self.world_tick
        act = result.action_type
        patch_id = (getattr(result, "state_delta", {}) or {}).get("patch_id")
        for t in self.tasks.values():
            rel = self._evidence_relevance(t, act, artifact_id, patch_id)   # v4 review §2
            if rel == "none":
                continue
            if t.task_id not in art.linked_task_ids:
                art.linked_task_ids.append(t.task_id)
            if rel == "weak":
                ev_type = "weak_signal"
            elif patch_id:
                ev_type = "substantive_patch"
            elif artifact_id in result.created_objects:
                ev_type = "artifact_created"
            else:
                ev_type = "artifact_revised"
            t.progress_evidence.append({"tick": tick, "actor": aid, "action": act, "patch_id": patch_id,
                                        "evidence_type": ev_type, "object": artifact_id})
            prev = getattr(t.status, "value", str(t.status))
            if prev == "open" and rel == "substantive":
                t.status = TaskStatus.IN_PROGRESS
                if not t.owner_id:
                    t.owner_id = aid
                    self.board.owners[t.task_id] = aid
            # spec #2: a product task is NOT done on an accepted patch — it reaches
            # implementation_done and must be merged (or released) to count as complete.
            if prev not in COMPLETED_TASK_STATUSES and self._task_requirements_met(t, art):
                t.status = (TaskStatus.IMPLEMENTATION_DONE if self._is_product_task(t)
                            else TaskStatus.DONE)
            self._emit_task_transition(t, prev, aid, act, artifact_id, tick)

    def _is_product_task(self, t) -> bool:
        """A task that ships a product deliverable (code/doc/eval/report artifact) — its
        completion must go through merge/release, not just an accepted patch (spec #2).
        Tasks with no product artifact (pure coordination) keep the plain 'done'."""
        arts = getattr(self, "product_artifacts", {}) or {}
        for aid in (getattr(t, "linked_artifacts", []) or []):
            a = arts.get(aid)
            if a is not None and getattr(a, "artifact_type", "") != "issue":
                return True
        return False

    def _evidence_relevance(self, t, act: str, artifact_id: str, patch_id) -> str:
        """Classify evidence against the artifacts named by the current workload task."""
        linked = artifact_id in getattr(t, "linked_artifacts", [])
        weak_act = act in self._WEAK_ACTIONS and not patch_id   # an accepted patch is never weak
        if not linked:
            return "none"
        return "weak" if weak_act else "substantive"

    def _emit_task_transition(self, t, prev, aid, act, artifact_id, tick) -> None:
        new = getattr(t.status, "value", str(t.status))
        if new != prev:
            t.history.append({"tick": tick, "agent_id": aid, "from": prev, "to": new,
                              "via": act, "artifact_id": artifact_id})
            self.events.append({"type": "task_progress_event", "subtype": new,
                                "task_id": t.task_id, "agent_id": aid, "tick": tick})

    def _task_requirements_met(self, t, art) -> bool:
        """Evaluate the task's completion_requirements against its evidence + the
        primary artifact. Updates progress_score; returns True only if ALL met."""
        ev = t.progress_evidence or []
        substantive = [e for e in ev if e.get("evidence_type") != "weak_signal"]
        actors = {e.get("actor") for e in substantive}
        actions = {(e.get("actor"), e.get("action")) for e in substantive}
        reviewed_ev = any(e.get("evidence_type") in ("review_completed", "proposal_approved",
                                                     "protocol_adopted") for e in ev)
        # What this gate is for is independent verification, and a co-contributor
        # is only one way to get it. A one-person condition can never produce a
        # second actor, so demanding one turns that rung into a hard zero that
        # cannot be told apart from a founder who tried and failed. The frozen
        # test suite verifies independently of whoever wrote the code, so on a
        # solo roster it stands in for the second party; larger rosters still owe
        # a real one.
        verified_by_execution = any(
            e.get("evidence_type") in ("substantive_patch", "merged_to_mainline")
            for e in substantive
        )
        checks = {
            "artifact_revised": int(getattr(art, "revision", 0) or 0) >= 1
                                and getattr(art, "artifact_type", "") != "issue",
            "gaps_cleared": not getattr(art, "known_gaps", []),
            # A release gate re-running and no longer listing its blocker is the
            # strongest verification here — an oracle outside the roster's reach,
            # stronger than a peer review. Blocker tasks used to complete on it
            # without recording it, which is why they showed a completed status
            # over zero evidence. Recording it lets one gate function stay the
            # only thing that can complete a task.
            "gate_cleared": any(e.get("evidence_type") == "gate_cleared" for e in ev),
            "reviewed": (
                reviewed_ev
                or len(actors) >= ADOPT_MIN_SUPPORTERS
                or bool(getattr(art, "linked_proposal_ids", []))
                or (len(self.agents) < ADOPT_MIN_SUPPORTERS and verified_by_execution)
            ),
            "multi_evidence": len(actions) >= 2,
        }
        req = list(getattr(t, "completion_requirements", []) or [])
        met = [k for k in req if checks.get(k, True)]
        t.progress_score = round(len(met) / max(1, len(req)), 3)
        return all(checks.get(k, True) for k in req)

    def _record_review_evidence(self, result, aid: str) -> None:
        """Record review/approval evidence (preflight v3 §4) against tasks whose
        linked artifact was reviewed, so peer review counts toward completion."""
        if result.action_type not in self._REVIEW_ACTIONS:
            return
        tick = self.world_tick
        objs = set(result.created_objects + result.modified_objects)
        for k in ("artifact_id", "target_object_id", "object_id", "doc_id", "pr_id"):
            v = (result.parameters or {}).get(k) if hasattr(result, "parameters") else None
            if isinstance(v, str):
                objs.add(v)
        ev_type = "proposal_approved" if "proposal" in result.action_type else (
            "protocol_adopted" if "protocol" in result.action_type else "review_completed")
        for t in self.tasks.values():
            la = getattr(t, "linked_artifacts", [])
            if objs & set(la):
                # pick the evidence object by linked_artifacts order (not set iteration
                # order, which varies with PYTHONHASHSEED)
                t.progress_evidence.append({"tick": tick, "actor": aid, "action": result.action_type,
                                            "evidence_type": ev_type,
                                            "object": next(o for o in la if o in objs)})

    # -- v6 P0.4: PR merge -> mainline + task evidence; deterministic workflow sweep ----
    def apply_merged_pr(self, pr, merger_id: str, tick: int, res=None) -> list:
        """Apply a merged PR's patches to the mainline product artifact(s) AND credit its
        linked task(s) with merge evidence. Shared by the merge_pr handler (pass ``res`` so
        the event lands on the action result) and the deterministic sweep (no ``res``)."""
        arts = self.product_artifacts
        sink = res.events if res is not None else self.events
        touched: list = []
        for patch_id, art_id in self.repo_system.merged_commit_patches(pr):
            art = arts.get(art_id) if art_id else None
            if art is None:
                continue
            art.mainline_revision += 1
            art.last_reviewed_tick = tick
            art.awaiting_review = False
            art.status = "active"
            # materialization: the merge promotes the carried patch's real text to mainline
            pp = self.patches.get(patch_id)
            promoted = getattr(pp, "new_content", "") if pp else ""
            if promoted:
                art.mainline_content = promoted
            elif art.content:
                art.mainline_content = art.content
            if pr.pr_id not in art.linked_pr_ids:
                art.linked_pr_ids.append(pr.pr_id)
            if art_id not in touched:
                touched.append(art_id)
            sink.append({"type": "product_event", "subtype": "merged_to_mainline",
                         "artifact_id": art_id, "pr_id": pr.pr_id, "patch_id": patch_id,
                         "agent_id": merger_id, "tick": tick})
        self.apply_merge_task_evidence(pr, touched, merger_id, tick)
        # v8d P0c: credit the PR's linked issues with the merge (resolved_by_pr_ids) so the
        # issue lifecycle reflects the work that closed it, not just the task.
        for iid in (list(getattr(pr, "linked_issue_ids", []) or [])
                    + ([pr.linked_issue] if getattr(pr, "linked_issue", None) else [])):
            art = arts.get(iid)
            if art is None:
                continue
            rb = art.__dict__.setdefault("resolved_by_pr_ids", [])
            if pr.pr_id not in rb:
                rb.append(pr.pr_id)
        if touched:
            self._retire_verdicts_made_against_the_old_mainline(pr, tick)
        self.reconcile(reason="pr_merge")          # spec #1: gaps/issues/readiness follow a merge
        return touched

    def _retire_verdicts_made_against_the_old_mainline(self, merged, tick: int) -> None:
        """A green verdict describes the mainline it was judged against.

        CI judges a request as the mainline plus that request's own changes, so
        every other open request's verdict was about a mainline that no longer
        exists. Left standing, requests that each passed alone merge one after the
        other into a combination none of them was ever checked against — the
        classic "all green, red once combined". Retiring the verdict is what makes
        the next CI run an integration check rather than a repeat.

        The tree marker goes too: the re-run gate skips a request whose tree has
        not moved, and without clearing it a retired verdict could never be
        renewed.
        """
        for pr in self.repo_system.repo.pull_requests.values():
            if pr.pr_id == getattr(merged, "pr_id", None):
                continue
            if str(getattr(getattr(pr, "status", None), "value",
                           getattr(pr, "status", ""))) in ("merged", "closed"):
                continue
            if not getattr(pr, "ci_passed", False):
                continue
            pr.ci_passed = False
            pr.__dict__.pop("ci_tree_hash", None)
            pr.ci_brief = (f"needs CI again: the mainline moved when "
                           f"{getattr(merged, 'pr_id', 'another request')} merged")
            self.events.append({"type": "repo_event", "subtype": "ci_verdict_retired",
                                "pr_id": pr.pr_id, "merged_pr_id": getattr(merged, "pr_id", None),
                                "tick": tick})

    def apply_merge_task_evidence(self, pr, artifact_ids: list, merger_id: str, tick: int) -> None:
        """A merge to mainline is the real completion signal: record merge + review
        evidence on the linked task(s) and re-evaluate completion (v6 P0.4)."""
        from environments.org_env.backend.entities import TaskStatus
        arts = self.product_artifacts
        task_ids = list(getattr(pr, "linked_task_ids", []) or [])
        if not task_ids and getattr(pr, "linked_task", None):
            task_ids = [pr.linked_task]
        for a in artifact_ids:
            art = arts.get(a)
            for tid in (getattr(art, "linked_task_ids", []) or []) if art else []:
                if tid not in task_ids:
                    task_ids.append(tid)
        reviewers = list(getattr(pr, "approved_by", []) or [])
        primary = artifact_ids[0] if artifact_ids else None
        for tid in task_ids:
            t = self.tasks.get(tid)
            if t is None:
                continue
            prev = getattr(t.status, "value", str(t.status))
            t.progress_evidence.append({"tick": tick, "actor": merger_id, "action": "merge_pr",
                                        "evidence_type": "merged_to_mainline", "pr_id": pr.pr_id,
                                        "object": primary})
            for r in reviewers:
                t.progress_evidence.append({"tick": tick, "actor": r, "action": "review_pr",
                                            "evidence_type": "review_completed", "pr_id": pr.pr_id,
                                            "object": primary})
            art = next((arts[a] for a in (getattr(t, "linked_artifacts", []) or []) if a in arts), None)
            if art is None and primary:
                art = arts.get(primary)
            # spec #2: merging the task's implemented work to mainline completes it ->
            # "merged" (gated on real implementation evidence, not on every unrelated gap).
            implemented = prev in ("in_progress", "implementation_done", "review_pending") or any(
                e.get("evidence_type") in ("substantive_patch", "artifact_revised", "artifact_created")
                for e in (t.progress_evidence or []))
            # Having implemented something is not the same as having finished the
            # task, and this path used to accept the first as the second. Measured
            # over 336 ticks: three tasks reached "merged" here carrying 47, 27 and
            # 17 evidence entries and a progress_score of 0.0 — not one completion
            # requirement satisfied. A merge that skips the gate makes tasks_done
            # report work the gate never accepted, in whichever direction the run
            # happens to lean.
            if prev not in COMPLETED_TASK_STATUSES and implemented \
                    and self._task_requirements_met(t, art):
                t.status = TaskStatus.MERGED
                self._emit_task_transition(t, prev, merger_id, "merge_pr",
                                           (art.artifact_id if art else primary), tick)

    def process_repo_workflow(self) -> list:
        """Deterministic backstop (mirrors process_pending_adoptions): after a review /
        merge latency, advance PRs that agents left waiting so the
        patch -> commit -> PR -> review -> merge -> mainline chain actually closes.
        Agents still act first within the latency window; events are tagged ``auto``."""
        rs = getattr(self, "repo_system", None)
        if rs is None:
            return []
        REVIEW_LATENCY, MERGE_LATENCY, PR_OPEN_LATENCY = 3, 2, 2
        tick = self.world_tick
        advanced: list = []
        # v8 #3 closure backstop: a branch that has commits but was never turned into a PR
        # would otherwise sit forever (artifact stays awaiting_review, mainline_revision 0).
        # After a short latency, auto-open its PR so the merge chain can complete.
        for b in list(rs.repo.branches.values()):
            if getattr(b.status, "value", str(b.status)) != "ready_for_pr" or not b.commit_ids:
                continue
            last_commit = max((int(getattr(rs.repo.commits.get(cid), "timestamp", 0) or 0)
                               for cid in b.commit_ids), default=0)
            if tick - last_commit < PR_OPEN_LATENCY:
                continue
            reviewer = self._a_lead_other_than(b.owner_id)
            if reviewer is None and len(self.agents or {}) <= 1:
                reviewer = b.owner_id      # solo roster: its own review, on the record
            pr = rs.open_pr(agent_id=b.owner_id, source_branch=b.branch_id,
                            reviewers=[reviewer] if reviewer else None)
            pr.opened_tick = tick
            tset, iset = [], []
            for cid in b.commit_ids:
                c = rs.repo.commits.get(cid)
                if c is None:
                    continue
                for tid in (getattr(c, "linked_task_ids", []) or
                            ([c.linked_task_id] if c.linked_task_id else [])):
                    if tid and tid not in tset:
                        tset.append(tid)
                for iid in getattr(c, "linked_issue_ids", []) or []:
                    if iid not in iset:
                        iset.append(iid)
            pr.linked_task_ids, pr.linked_issue_ids = tset, iset
            pr.linked_task = tset[0] if tset else None
            pr.linked_issue = iset[0] if iset else None
            self.events.append({"type": "repo_event", "subtype": "pr_opened", "pr_id": pr.pr_id,
                                "agent_id": b.owner_id, "tick": tick, "auto": True})
            advanced.append(pr.pr_id)
        # CI is NO LONGER a one-shot verdict — a not-yet-passing PR is re-checked
        # periodically, and a `changes_requested` PR whose CI turns green again is
        # REVIVED to review instead of being trapped forever (the
        # "changes_requested black hole").
        _CI_RECHECK_EVERY = 8
        _ci_tick_cache = {}

        def _integration_ci(pr=None):
            """This request's own verdict, or the desk's.

            One verdict used to be computed per tick from the shared working tree
            and stamped onto every open request. A request was then failed for
            everyone else's unfinished work, which is exactly what judging the
            merge candidate exists to stop: over 96 ticks five approved requests
            never merged while the verdict each was shown wandered across five
            different modules, and one of them carried a single file that appeared
            in none of them.

            Cached per request within the tick, and the export underneath is
            cached by content, so a candidate that has not changed costs nothing.
            """
            key = getattr(pr, "pr_id", "") or "__the_desk__"
            if key not in _ci_tick_cache:
                try:
                    from environments.org_env.product.contracts import run_integration_ci
                    _ci_tick_cache[key] = run_integration_ci(self, pr)
                except Exception as exc:  # noqa: BLE001
                    # A check that could not run has not passed. This used to
                    # answer ok, so any error inside CI cleared the request for
                    # merge: a stub of the wrong arity was enough to auto-merge a
                    # request whose tree was red. An outage is already a category
                    # that neither condemns the product nor clears it.
                    _ci_tick_cache[key] = {
                        "ok": False, "kind": "infrastructure_error", "boundary": "",
                        "brief": f"the integration check could not run: "
                                 f"{type(exc).__name__}: {exc}"[:300]}
            return _ci_tick_cache[key]

        def _run_pr_ci(pr):
            # same integration CI as the LLM run_ci action (end-to-end contract + eval metric
            # consistency), so the AUTO merge route can't ship a bad patch with "CI passed".
            ci = rs.run_ci(pr_id=pr.pr_id, tick=tick)
            ci_status = getattr(ci, "status", "passed") if ci is not None else "passed"
            cc = _integration_ci(pr)
            infra = cc.get("kind") == "infrastructure_error"
            from environments.org_env.product.contracts import record_working_tree_break
            record_working_tree_break(self, _integration_ci())
            # The same writer the run_ci action uses: the record and the flag are
            # set together, a refusal records its reason, and an outage does not
            # condemn a product the check could not judge.
            from environments.org_env.product.contracts import record_integration_verdict
            ci_status = record_integration_verdict(ci, pr, cc, world=self)
            if not cc["ok"]:
                self.events.append({
                    "type": "repo_event",
                    "subtype": ("ci_infrastructure_error" if infra
                                else "ci_metric_inconsistency" if cc["kind"] == "metric_inconsistency"
                                else "ci_contract_break"),
                    "pr_id": pr.pr_id, "agent_id": pr.author_id, "tick": tick,
                    "boundary": cc["boundary"], "brief": cc["brief"][:200], "auto": True})
            _ck = ("contract", "interface", "schema", "integration", "evidence chain")
            self.note_protocol_use(_ck, tick, obj=pr.pr_id)
            if not cc["ok"] and not infra:
                issue_ids = tuple(
                    sorted(
                        {
                            str(issue_id)
                            for issue_id in (
                                getattr(pr, "linked_issue_ids", ()) or ()
                            )
                            if str(issue_id)
                        }
                    )
                )
                workflow_id = f"ci_gate:{pr.pr_id}"
                before = GovernedObjectSnapshot(
                    object_type="workflow",
                    object_id=workflow_id,
                    lifecycle_state="candidate",
                    observed_tick=tick,
                    provenance_ref=(
                        f"runtime://ci/{pr.pr_id}/input/t{tick}"
                    ),
                    attributes={
                        "pull_request_id": pr.pr_id,
                        "issue_ids": issue_ids,
                        "ci_status": "pending",
                    },
                )
                after = GovernedObjectSnapshot(
                    object_type="workflow",
                    object_id=workflow_id,
                    lifecycle_state="blocked",
                    observed_tick=tick,
                    provenance_ref=(
                        f"runtime://ci/{pr.pr_id}/result/t{tick}"
                    ),
                    attributes={
                        "pull_request_id": pr.pr_id,
                        "issue_ids": issue_ids,
                        "ci_status": "failed",
                        "failure_kind": str(cc.get("kind") or "ci_failure"),
                    },
                )
                self.note_protocol_enforcement(
                    _ck,
                    tick,
                    obj=pr.pr_id,
                    agent=pr.author_id,
                    actions=("run_ci", "ci_test"),
                    blocked=True,
                    governed_object_before=before,
                    governed_object_after=after,
                )
            pr.__dict__["_last_ci_tick"] = tick
            self.events.append({"type": "repo_event", "subtype": "ci", "pr_id": pr.pr_id,
                                "agent_id": pr.author_id, "tick": tick, "status": ci_status, "auto": True})

        for pr in list(rs.repo.pull_requests.values()):
            st = getattr(pr.status, "value", str(pr.status))
            if st in ("merged", "closed"):
                continue
            # (re)run CI: first time, then re-check a not-yet-passing PR periodically so a later fix to
            # the shared working tree is re-evaluated (CI must not be a one-shot verdict).
            if (not pr.ci_run_ids) or (not pr.ci_passed
                    and tick - int(pr.__dict__.get("_last_ci_tick", -999)) >= _CI_RECHECK_EVERY):
                _run_pr_ci(pr)
            # revival: a changes_requested PR whose CI now passes returns to review (not a dead end).
            st = getattr(pr.status, "value", str(pr.status))
            if st == "changes_requested" and pr.ci_passed:
                from environments.org_env.backend.repo.repo import PRStatus
                pr.status = PRStatus.REVIEW_REQUESTED
                pr.reviewed = False
                self.events.append({"type": "repo_event", "subtype": "pr_resubmitted",
                                    "pr_id": pr.pr_id, "agent_id": pr.author_id, "tick": tick, "auto": True})
                st = "review_requested"
            if st in ("merged", "closed", "changes_requested"):
                continue
            if not pr.reviewed and st in ("open", "review_requested") \
                    and tick - int(getattr(pr, "opened_tick", 0) or 0) >= REVIEW_LATENCY:
                reviewer = (pr.reviewers[0] if pr.reviewers else None) or self._a_lead_other_than(pr.author_id)
                if reviewer is None and self.experiment_condition == B0_SINGLE_AGENT_FOUNDER:
                    # B0 has no independent reviewer; self-review is the explicit
                    # single-agent baseline rather than silently deadlocking PRs.
                    reviewer = pr.author_id
                if reviewer:
                    if pr.ci_passed:
                        rs.review_pr(reviewer_id=reviewer, pr_id=pr.pr_id, approve=True, tick=tick)
                        self.events.append({"type": "repo_event", "subtype": "pr_reviewed",
                                            "pr_id": pr.pr_id, "agent_id": reviewer, "tick": tick, "auto": True})
                    else:
                        rs.request_changes(reviewer_id=reviewer, pr_id=pr.pr_id,
                                           comment="CI not passing; please fix", tick=tick)
                        self.events.append({"type": "repo_event", "subtype": "changes_requested",
                                            "pr_id": pr.pr_id, "agent_id": reviewer, "tick": tick, "auto": True})
                    advanced.append(pr.pr_id)
            st = getattr(pr.status, "value", str(pr.status))
            if st == "approved" and pr.ci_passed:
                at = pr.approved_tick if pr.approved_tick is not None else getattr(pr, "opened_tick", 0)
                if tick - int(at or 0) >= MERGE_LATENCY:
                    merger = self._a_lead_other_than(None) or pr.author_id
                    if rs.merge_pr(pr_id=pr.pr_id, tick=tick, force=False):
                        self.events.append({"type": "repo_event", "subtype": "pr_merged",
                                            "pr_id": pr.pr_id, "agent_id": merger, "tick": tick, "auto": True})
                        self.apply_merged_pr(pr, merger, tick)
                        advanced.append(pr.pr_id)
        return advanced

    def _a_lead_other_than(self, author_id):
        for aid, a in (self.agents or {}).items():
            if aid == author_id:
                continue
            if getattr(a, "role", "") in ("cofounder", "founder", "reliability"):
                return aid
        return None

    # -- spec #3: internal release-candidate pipeline (deterministic backstop) ----------
    def process_release_pipeline(self) -> list:
        """Drive the internal release lifecycle so it actually fires by tick 200: once
        >=2 patch-carrying PRs have merged, create a candidate, run the 9-check gate
        (reconciling first), turn blockers into issues, approve a clean candidate, and
        publish internally. Agents act first within a tick; this is the backstop."""
        rs = getattr(self, "repo_system", None)
        if rs is None:
            return []
        from environments.org_env.backend.repo.release import release_gates_for, evaluate_release_gates
        tick = self.world_tick
        rcs = rs.repo.release_candidates
        rc = next((r for r in rcs.values()
                   if r.status in ("draft", "under_review", "approved", "blocked")), None)
        if rc is None:
            used = {p for r in rcs.values() if r.status == "released" for p in r.included_pr_ids}
            fresh = [pr for pr in rs.repo.pull_requests.values()
                     if getattr(pr.status, "value", str(pr.status)) == "merged"
                     and pr.patch_ids and pr.pr_id not in used]
            # v8g P0: release CADENCE — after a release, let later small PRs accumulate (they
            # update the next RC, not a release each) and only cut a new internal release
            # snapshot once the cadence has elapsed or enough new work piled up. Keeps a 14-day
            # run to ~1-3 internal releases instead of ~9.
            # release CADENCE batches small PRs (>=2) at most once per RELEASE_CADENCE. BUT never let a
            # merged fix sit UNSHIPPED indefinitely (the "fixed but never shipped" bug — a lone httpx
            # fix merged into mainline but no new release ever cut it): once there's >=1 fresh merged
            # patch-PR and it's been ~half a cadence since the last release, ship it anyway.
            since = tick - int(getattr(self, "_last_release_tick", -10**9))
            batch = len(fresh) >= 2 and (not rcs or since >= self.RELEASE_CADENCE)
            lone_stale = bool(rcs) and len(fresh) >= 1 and since >= (self.RELEASE_CADENCE // 2)
            if not (batch or lone_stale):
                return []
            lead = self._a_lead_other_than(None) or "system"
            rc = rs.create_release_candidate(created_by=lead, tick=tick,
                                             required_gates=list(release_gates_for(self)))
            self.events.append({"type": "release_event", "subtype": "rc_created", "auto": True,
                                "candidate_id": rc.candidate_id, "version": rc.version,
                                "agent_id": lead, "tick": tick})
            return [rc.candidate_id]
        self.reconcile(reason="before_release_gate")          # spec #3: reconcile before the gate
        if rc.status in ("draft", "blocked"):
            # v8d P0a: fold any later merged PRs into the candidate so a blocked RC keeps up
            # with the work instead of freezing at its creation tick (rc_13 stuck on pr_3/9).
            if rs.refresh_candidate(rc, tick=tick):
                self.events.append({"type": "release_event", "subtype": "rc_updated", "auto": True,
                                    "candidate_id": rc.candidate_id, "tick": tick,
                                    "included_pr_ids": list(rc.included_pr_ids),
                                    "included_task_ids": list(rc.included_task_ids)})
            was_draft = rc.status == "draft"
            rc.gate_results = evaluate_release_gates(self, rc)
            rc.blockers = [r["gate"] for r in rc.gate_results
                           if not r["passed"] and r["gate"] not in rc.waived_gates]
            self._note_rc_blockers(rc.blockers, tick)        # #2: track blocker changes
            self._close_cleared_blockers(rc, tick)           # v8e #3: gate re-run closes resolved blockers
            # v8d P1a: a BLOCKED release is the release-quality protocol doing its job —
            # credit one enforcement per candidate (the institution prevented an unsafe ship).
            #
            # An EMPTY candidate is not that. A candidate with no included PRs
            # fails gate_ci_passed_for_included_prs vacuously — there is no
            # unsafe ship to prevent — and crediting it spends the
            # once-per-candidate allowance at the candidate's emptiest moment.
            # Measured over 336 ticks: all eight enforcement events were
            # recorded before any PR was folded in, so every one snapshotted
            # issue_ids=() even though seven of those candidates went on to
            # carry work and ship. That empty tuple is the join key
            # _linked_outcomes uses to connect an enforcement to the evaluator's
            # per-issue outcomes, so nothing could ever link, no protocol could
            # earn an independent outcome oracle, and strong emergence was
            # unreachable by construction rather than by conduct — the run
            # reported strong_protocol_emergence_rate 0.0 with a protocol that
            # met the other two strong conditions.
            if rc.blockers and rc.included_pr_ids and rc.candidate_id not in \
                    self.__dict__.setdefault("_rc_protocol_enforced", set()):
                included_pull_requests = [
                    rs.repo.pull_requests.get(pr_id)
                    for pr_id in (rc.included_pr_ids or ())
                ]
                issue_ids = tuple(
                    sorted(
                        {
                            str(issue_id)
                            for pull_request in included_pull_requests
                            if pull_request is not None
                            for issue_id in (
                                getattr(
                                    pull_request,
                                    "linked_issue_ids",
                                    (),
                                )
                                or ()
                            )
                            if str(issue_id)
                        }
                    )
                )
                before = GovernedObjectSnapshot(
                    object_type="release_candidate",
                    object_id=rc.candidate_id,
                    lifecycle_state=str(rc.status),
                    observed_tick=tick,
                    provenance_ref=(
                        f"runtime://release/{rc.candidate_id}/pre-gate/t{tick}"
                    ),
                    attributes={
                        "included_pr_ids": tuple(rc.included_pr_ids or ()),
                        "issue_ids": issue_ids,
                        "blockers": (),
                    },
                )
                after = GovernedObjectSnapshot(
                    object_type="release_candidate",
                    object_id=rc.candidate_id,
                    lifecycle_state="blocked",
                    observed_tick=tick,
                    provenance_ref=(
                        f"runtime://release/{rc.candidate_id}/post-gate/t{tick}"
                    ),
                    attributes={
                        "included_pr_ids": tuple(rc.included_pr_ids or ()),
                        "issue_ids": issue_ids,
                        "blockers": tuple(sorted(rc.blockers)),
                    },
                )
                if self.note_protocol_enforcement(
                        ("evidence", "claim", "credib", "traceab", "quality", "gate", "release"),
                        tick, obj=rc.candidate_id, agent="release_gate",
                        actions=("run_launch_readiness_check", "create_release_candidate",
                                 "publish_product_release"),
                        blocked=True,
                        governed_object_before=before,
                        governed_object_after=after):
                    self._rc_protocol_enforced.add(rc.candidate_id)
            # spec #4/#8: the first gate run credits the evidence protocol with a use
            # (draft-only so a per-tick blocked re-gate doesn't inflate the count).
            if was_draft and any(r["gate"] == "gate_claim_evidence_protocol_active_or_pending"
                                 and r["passed"] for r in rc.gate_results):
                pid = self.note_protocol_use(("evidence", "claim", "credib", "traceab"), tick,
                                             obj=rc.candidate_id)
                if pid:
                    self.events.append({"type": "protocol_use_event", "protocol_id": pid, "auto": True,
                                        "tick": tick, "used_in": "release_gate", "object_id": rc.candidate_id})
            new = "blocked" if rc.blockers else "under_review"
            if new != rc.status or rc.status == "draft":
                rc.status = new
                self.events.append({"type": "release_event", "subtype": "readiness_check", "auto": True,
                                    "candidate_id": rc.candidate_id, "status": rc.status,
                                    "blockers": list(rc.blockers), "tick": tick})
                if rc.blockers:
                    self._blockers_to_issues(rc, tick)
            return [rc.candidate_id]
        if rc.status == "under_review" and not rc.blockers:
            for aid, a in self.agents.items():
                if getattr(a, "role", "") in ("founder", "cofounder", "reliability") \
                        and aid not in rc.approvals:
                    rs.approve_release_candidate(rc_id=rc.candidate_id, approver=aid)
            from environments.org_env.backend.repo.release import (
                release_approval_met,
            )

            if release_approval_met(self, rc):
                rc.status = "approved"
                self.events.append({"type": "release_event", "subtype": "rc_approved", "auto": True,
                                    "candidate_id": rc.candidate_id, "status": "approved", "tick": tick})
            return [rc.candidate_id]
        if rc.status == "approved":
            # v8g P0: release CADENCE at the PUBLISH step (catches agent-created RCs too) — an
            # approved candidate is held until the cadence elapses, so a 14-day run produces a
            # few internal release snapshots, not one every couple of days.
            if tick - int(getattr(self, "_last_release_tick", -10**9)) < self.RELEASE_CADENCE:
                return [rc.candidate_id]
            lims: list = []
            for cid in rc.included_commit_ids:
                c = rs.repo.commits.get(cid)
                for pid in (c.patch_ids if c else []):
                    p = self.patches.get(pid)
                    for l in (getattr(p, "added_limitations", []) or []) + (getattr(p, "known_limitations", []) or []):
                        if l not in lims:
                            lims.append(l)
            _pname = (self.company_config or {}).get("product_name") or "product"
            rel = rs.publish_release(rc_id=rc.candidate_id, released_by=self._a_lead_other_than(None) or "system",
                                     tick=tick, public_summary=f"{_pname} {rc.version} (internal)",
                                     known_limitations=lims)
            if rel is not None:
                if self.product is not None:
                    self.product.stage = "internal_release"
                self._last_release_tick = tick            # v8g P0: start the release-cadence clock
                snap = self._materialize_release(rel.version)
                self.events.append({"type": "release_event", "subtype": "published_internal", "auto": True,
                                    "release_id": rel.release_id, "version": rel.version, "tick": tick,
                                    "snapshot_dir": snap.get("dir"), "smoke_ok": snap.get("smoke_ok")})
                self.reconcile(reason="post_release")     # readiness.release_ready follows the publish
                # v16 §6: the internal cadence is how a run usually SHIPS — drive the external
                # market here too (else trials stay 0 / customers milestone never reachable, since
                # the agent rarely picks the explicit publish_product_release action).
                from environments.org_env.experiments.ablations import (
                    EXTERNAL_BRIDGE,
                    mechanism_disabled,
                )
                if not mechanism_disabled(self, EXTERNAL_BRIDGE):
                    try:
                        from environments.org_env.external_society.bridge import drive_post_release_market
                        drive_post_release_market(self, tick)
                    except Exception:
                        pass
                return [rel.release_id]
        return []

    RELEASE_CADENCE = 144          # v8g P0: min ticks between internal release snapshots (~6 days)

    # #2: a failing gate maps to the artifact whose work would clear it (so the blocker
    # task lands on the right deliverable and re-gates clean once merged).
    _GATE_PURPOSE = {
        "gate_source_credibility_supported": "source_tracker",
        "gate_report_quality_checklist_exists": "report_quality",
        "gate_readme_claims_audited": "readme",
        "gate_eval_metrics_defined_or_marked": "eval",
        "gate_claim_evidence_protocol_active_or_pending": "claim_tracker",
    }

    def _blockers_to_issues(self, rc, tick: int) -> None:
        """spec #3 + #2: a failing gate becomes an actionable issue AND an owned blocker
        task on the artifact that would clear it — so agents resolve blockers instead of
        re-running the gate."""
        from environments.org_env.product.objects import ProductArtifact, artifact_purpose
        from environments.org_env.backend.repo.release import _GATE_ACTION
        from environments.org_env.backend.entities import Task, TaskStatus
        from environments.org_env.experiments.ablations import CAPABILITY_MEMORY, mechanism_disabled
        capability_memory_off = mechanism_disabled(self, CAPABILITY_MEMORY)
        arts = self.product_artifacts
        detail_by_gate = {r["gate"]: (r.get("detail") or "") for r in (getattr(rc, "gate_results", []) or [])}
        for gate in rc.blockers:
            action = _GATE_ACTION.get(gate, gate)
            detail = detail_by_gate.get(gate, "")
            problem = action + (("  —  " + detail) if detail else "")
            # v11 coding memory: for a TECHNICAL gate, recall how a prior debugging episode
            # closed the same gate, so agents reuse the known fix instead of re-discovering it.
            if not capability_memory_off and any(k in gate.lower() for k in ("smoke", "ci", "eval", "test", "build")):
                recall = self.episode_manager.recall_brief(gate=gate)
                if recall:
                    problem = problem + "  —  " + recall
            iid = f"rel_blocker_{gate}"
            tid = f"task_rel_blocker_{gate}"
            # v8e #3: resolve owner + target + linked gaps ONCE so both the issue and the
            # task carry a complete, traceable link set (RC / gate / known_gap / owner).
            purpose = self._GATE_PURPOSE.get(gate)
            target = next((a.artifact_id for a in arts.values()
                           if a.artifact_type != "issue"
                           and artifact_purpose(getattr(a, "linked_file_path", "") or a.artifact_id) == purpose),
                          None) if purpose else None
            owner = self.board.owners.get(tid) or (self._domain_owner(arts[target]) if target in arts
                                                   else self._a_lead_other_than(None))
            gap_ids = [g.gap_id for g in (getattr(self, "known_gaps", {}) or {}).values()
                       if target and getattr(g, "artifact_id", None) == target
                       and getattr(g, "status", "active") in ("active", "regressed")]
            if iid in arts:
                # refresh the live blocker reason (e.g. the current smoke traceback) so agents
                # always see WHY it is blocked, not a stale generic action.
                arts[iid].problem = problem
                arts[iid].summary = problem
                arts[iid].updated_at_tick = tick
                # v14b: REOPEN a blocker that had been resolved but whose gate REGRESSED. Being in
                # rc.blockers means the gate is failing NOW, so a "resolved" status (and its done
                # task) is stale — leaving it closed made blocker_resolution_rate read 1.0 while the
                # RC was still blocked, and let a regressed gate vanish from the work surface.
                if getattr(arts[iid], "status", "open") not in ("open", "in_progress"):
                    from environments.org_env.backend.entities import TaskStatus
                    arts[iid].status = "open"
                    if self.product is not None and iid not in (self.product.open_issue_ids or []):
                        self.product.open_issue_ids.append(iid)
                    rt = self.tasks.get(tid)
                    if rt is not None:
                        rt.status = TaskStatus.OPEN
                    self.events.append({"type": "release_event", "subtype": "blocker_reopened",
                                        "candidate_id": rc.candidate_id, "gate": gate,
                                        "issue_id": iid, "tick": tick, "auto": True})
            if iid not in arts:
                art = ProductArtifact(
                    artifact_id=iid, artifact_type="issue", title=f"release blocker: {gate}",
                    status="open", problem=problem, summary=problem, priority="high",
                    created_at_tick=tick, updated_at_tick=tick)
                # complete, traceable blocker issue (the gate/RC/owner/gap it represents)
                art.__dict__.update({"issue_id": iid, "owner_id": owner, "linked_gate": gate,
                                     "linked_rc_id": rc.candidate_id, "linked_task_ids": [tid],
                                     "linked_gap_ids": gap_ids, "severity": "high"})
                arts[iid] = art
                if self.product is not None:
                    self.product.artifact_ids.append(iid)
                    if iid not in self.product.open_issue_ids:
                        self.product.open_issue_ids.append(iid)
                self.events.append({"type": "release_event", "subtype": "blocker_to_issue",
                                    "candidate_id": rc.candidate_id, "gate": gate, "issue_id": iid,
                                    "owner_id": owner, "tick": tick})
            if tid not in self.tasks:
                self.tasks[tid] = Task(
                    task_id=tid, title=f"resolve {gate}", description=problem, status=TaskStatus.OPEN,
                    owner_id=owner, priority=1, linked_issues=[iid],
                    linked_artifacts=[target] if target else [])
                self.board.tasks.append(tid)
                if owner:
                    self.board.owners[tid] = owner
                self.events.append({"type": "release_event", "subtype": "blocker_to_task",
                                    "candidate_id": rc.candidate_id, "gate": gate, "task_id": tid,
                                    "owner_id": owner, "tick": tick})

    def _close_cleared_blockers(self, rc, tick: int) -> None:
        """v8e #3: when the re-run gate no longer lists a blocker, close its blocker issue
        and complete its blocker task — so a resolved blocker doesn't linger as an open
        high-risk issue and the gate closure loop is visible."""
        from environments.org_env.backend.entities import COMPLETED_TASK_STATUSES, TaskStatus
        active = set(rc.blockers or [])
        arts = self.product_artifacts
        for iid, art in list(arts.items()):
            if not str(iid).startswith("rel_blocker_") or getattr(art, "artifact_type", "") != "issue":
                continue
            gate = getattr(art, "linked_gate", None) or str(iid).replace("rel_blocker_", "")
            if gate in active or getattr(art, "status", "") == "resolved":
                continue
            art.status = "resolved"
            art.updated_at_tick = tick
            if self.product is not None and iid in (self.product.open_issue_ids or []):
                self.product.open_issue_ids.remove(iid)
            t = self.tasks.get(f"task_rel_blocker_{gate}")
            if t is not None and getattr(t.status, "value", str(t.status)) not in COMPLETED_TASK_STATUSES:
                self._append_progress_evidence(t, actor="release_gate", tick=tick,
                                               evidence_type="gate_cleared", detail=gate)
                t.completion_requirements = ["gate_cleared"]
                if self._task_requirements_met(t, art):
                    t.status = TaskStatus.DONE
            # v8g P0: a cleared blocker also closes the meeting action items that TRACKED it,
            # so a passed gate doesn't leave overdue "assign owners for <gate>" items dangling.
            # Match the linked_gates tag set at creation — NOT the gate name appearing in the
            # description: agent-created work items ("write postmortem about <gate> flakiness")
            # legitimately mention gates and must survive the gate clearing. Untagged items
            # from pre-tag checkpoints fall back to the auto tracking item's exact wording.
            ms = getattr(self, "meeting_system", None)
            for ai in (ms.action_items.values() if ms is not None else []):
                if getattr(ai, "status", "") not in ("open", "overdue"):
                    continue
                tags = getattr(ai, "linked_gates", None) or []
                desc = ai.description or ""
                tracks = (gate in tags) if tags else (
                    desc.startswith("follow up: assign owners + patches for:") and gate in desc)
                if tracks:
                    ai.status = "done"
                    # Closing the item must also complete its linked follow-up task — the
                    # reverse of process_action_items' done branch (task done → item done).
                    # Once the item is done that loop skips it, so an unfinished linked
                    # task would otherwise stay open forever with a stale owner/deadline
                    # in the owner's assigned_tasks view.
                    lt = self.tasks.get(getattr(ai, "linked_task_id", None) or "")
                    task_completed = False
                    if lt is not None and getattr(lt.status, "value", str(lt.status)) not in COMPLETED_TASK_STATUSES:
                        # The follow-up exists to get this gate passing, and the
                        # gate is now passing: measured over one run, every one of
                        # the six clearings landed on the same tick as a PR merging
                        # with CI green. The outcome the item was created for was
                        # reached, so it completes — but through the same gate
                        # function as everything else, on recorded evidence, rather
                        # than by assignment. Its score reads 1.0 and says why.
                        #
                        # It does mean a gate that reopens is tracked, cleared and
                        # counted more than once; that recurrence is visible in the
                        # blocker_reopened events beside these.
                        #
                        # The evidence names the work that cleared the gate rather
                        # than crediting the follow-up on its own. Nobody picked
                        # this item up; the PRs in the candidate got CI green and
                        # the gate passed with them, and a reader of the trace
                        # should be able to see that without inferring it.
                        carried_by = list(getattr(rc, "included_pr_ids", []) or [])
                        self._append_progress_evidence(
                            lt, actor="release_gate", tick=tick,
                            evidence_type="gate_cleared",
                            detail=(f"{gate} cleared by the work in "
                                    f"{', '.join(carried_by) or 'this candidate'}; "
                                    "this follow-up was not worked separately"))
                        lt.completion_requirements = ["gate_cleared"]
                        lt_art = next(
                            (self.product_artifacts[a]
                             for a in (getattr(lt, "linked_artifacts", []) or [])
                             if a in self.product_artifacts),
                            None,
                        )
                        if self._task_requirements_met(lt, lt_art):
                            lt.status = TaskStatus.DONE
                            task_completed = True
                    self.events.append({"type": "meeting_event", "subtype": "action_item_done",
                                        "action_item_id": ai.action_item_id, "reason": "blocker_cleared",
                                        "task_id": getattr(ai, "linked_task_id", None),
                                        "task_completed": task_completed,
                                        # Completed alongside other work, not on its
                                        # own: these are the PRs that carried it.
                                        "completed_by_work": list(
                                            getattr(rc, "included_pr_ids", []) or []),
                                        "worked_separately": False,
                                        "tick": tick, "auto": True})
            self.events.append({"type": "release_event", "subtype": "blocker_cleared",
                                "candidate_id": rc.candidate_id, "gate": gate, "issue_id": iid, "tick": tick})

    BLOCKER_RESOLVE_LATENCY = 30      # v8f P0a: ticks a blocked RC waits before the owner ships the fix

    def _domain_owner(self, art):
        """v8f P1b: route a fix to a domain-appropriate NON-founder owner (engineering for
        code, documentation for docs) by domain authority — not the community role."""
        key = getattr(art, "linked_file_path", "") or getattr(art, "artifact_id", "")
        domain = "engineering_execution" if str(key).endswith(".py") else "documentation_quality"
        try:
            from environments.org_env.growth.authority import authority_in
            ranked = sorted(((authority_in(a, domain), aid) for aid, a in (self.agents or {}).items()),
                            reverse=True)
            if ranked and ranked[0][0] > 0:
                return ranked[0][1]
        except Exception:
            pass
        return self._a_lead_other_than(None) or next(iter(self.agents or {"system": 1}))

    def process_release_blockers(self, tick: int) -> list:
        """v8f P0a: deterministic last-mile backstop. When a release candidate has stayed
        blocked on a critical product gap past a latency, the domain owner ships the fix
        (a traceable resolving patch merged to mainline) so the gate can re-run and the RC
        can reach approved/published instead of looping forever blocked."""
        from environments.org_env.experiments.ablations import PRODUCT_WORKFLOW, mechanism_disabled
        if mechanism_disabled(self, PRODUCT_WORKFLOW):
            return []          # ablation: the auto repair backstop is part of the workflow
        rs = getattr(self, "repo_system", None)
        if rs is None:
            return []
        rc = next((r for r in rs.repo.release_candidates.values() if r.status == "blocked"), None)
        if rc is None or tick - _tick_value(getattr(rc, "created_at_tick", None), tick) < self.BLOCKER_RESOLVE_LATENCY:
            return []
        from types import SimpleNamespace
        from environments.org_env.product.patch_objects import CodePatch, DocPatch
        from environments.org_env.backend.entities import COMPLETED_TASK_STATUSES, TaskStatus
        arts = self.product_artifacts
        crit = [g for g in (self.known_gaps or {}).values()
                if getattr(g, "critical", False) and getattr(g, "status", "active") in ("active", "regressed")
                and g.linked_artifact_id and g.linked_artifact_id in arts]
        resolved_now: list = []
        for g in crit[:1]:                                   # gradual: one blocker fix per sweep
            art = arts[g.linked_artifact_id]
            owner = self._domain_owner(art)
            cap = (g.required_capability or "real_implementation").strip().lower()[:60]
            self._blkseq = getattr(self, "_blkseq", 0) + 1
            pid = f"patch_blockerfix_{self._blkseq}"
            is_doc = str(getattr(art, "linked_file_path", "")).endswith((".md", ".txt")) \
                or getattr(art, "artifact_type", "") == "doc"
            common = dict(patch_id=pid, target_object_id=art.artifact_id, actor_id=owner, tick=tick,
                          change_summary=f"ship blocker fix: {cap}", resolved_gaps=[g.gap_id],
                          edit_goal=f"resolve release blocker: {g.description[:60]}")
            patch = (DocPatch(patch_type="doc_patch", checklist_items=[cap], **common) if is_doc
                     else CodePatch(patch_type="code_patch", added_checks=[cap], **common))
            res = SimpleNamespace(modified_objects=[], created_objects=[], events=[], graph_edges=[],
                                  state_delta={}, success=True, action_id=pid, messages=[])
            if not self.apply_product_patch(patch, res, owner):
                continue
            for ev in res.events:
                self.events.append(ev)
            if g.description in art.known_gaps:
                art.known_gaps.remove(g.description)
            art.mainline_revision = int(getattr(art, "mainline_revision", 0) or 0) + 1
            art.awaiting_review = False
            art.status = "active"
            art.last_reviewed_tick = tick
            # This one went straight to the mainline artifact, so it is not
            # waiting on a branch for a commit.
            from environments.org_env.backend.repo.workflow import drop_patch
            drop_patch(self, pid)
            bt = self.tasks.get("task_rel_blocker_gate_no_critical_gaps_remaining")
            if bt is not None and getattr(bt.status, "value", str(bt.status)) not in COMPLETED_TASK_STATUSES:
                self._append_progress_evidence(
                    bt, actor=owner, tick=tick, evidence_type="gate_cleared",
                    detail=f"critical gap {g.gap_id} resolved and merged")
                bt.completion_requirements = ["gate_cleared"]
                if self._task_requirements_met(bt, art):
                    bt.status = TaskStatus.MERGED
            self.events.append({"type": "release_event", "subtype": "blocker_fix_merged", "auto": True,
                                "artifact_id": art.artifact_id, "gap_id": g.gap_id, "agent_id": owner, "tick": tick})
            resolved_now.append(g.gap_id)
        if resolved_now:
            self.reconcile(reason="blocker_resolution")
        return resolved_now

    def _note_rc_blockers(self, blockers, tick: int) -> None:
        """#2: record when the blocker set changes, so a repeated readiness check that sees
        the same blockers can be masked (resolve a blocker first)."""
        sig = tuple(sorted(blockers or []))
        if sig != self._rc_blocker_sig:
            self._rc_blocker_sig = sig
            self._rc_blockers_changed_tick = tick

    # -- #4: dispute lifecycle (resolve on evidence, else escalate -> release blocker) --
    DISPUTE_ESCALATE_LATENCY = 36

    def process_disputes(self, tick: int) -> list:
        cr = getattr(self, "commitment_registry", None)
        if cr is None:
            return []
        out = []
        for d in list(cr.disputes.values()):
            if d.status not in ("open", "escalated"):
                continue
            # resolved once the owner provides reproduction / re-export / a rerun on the target
            provided = any(
                (e.get("result_id") == d.target_object_id or e.get("object_id") == d.target_object_id)
                and int(e.get("tick", 0) or 0) >= int(d.created_tick or 0)
                and (e.get("subtype") in ("logged_to_tracker", "run_cheap_pilot", "json_result",
                                          "share_sandbox_output", "reproduced")
                     or e.get("type") == "requested_action_event")
                for e in self.events)
            if provided:
                d.status, d.resolution, d.resolved_tick = "resolved", "evidence_provided", tick
                d.resolved_by = d.resolved_by or self._result_owner(d.target_object_id) or "owner"
                self._close_dispute_issue(d, tick)     # v8d P1c: close the escalation issue
                self.events.append({"type": "claim_dispute_event", "subtype": "resolved",
                                    "dispute_id": d.dispute_id, "resolution": d.resolution,
                                    "resolved_by": d.resolved_by, "tick": tick})
                out.append(d.dispute_id)
            elif d.status == "open" and tick - int(d.created_tick or 0) >= self.DISPUTE_ESCALATE_LATENCY:
                d.status, d.resolution = "escalated", "escalated"
                self.events.append({"type": "claim_dispute_event", "subtype": "escalated",
                                    "dispute_id": d.dispute_id, "tick": tick})
                self._dispute_to_issue(d, tick)        # escalated dispute blocks the release gate
                out.append(d.dispute_id)
        return out

    def _result_owner(self, rid):
        ss = getattr(self, "sandbox_system", None)
        r = ss.results.get(rid) if (ss is not None and rid) else None
        if r is None:
            return None
        owner = getattr(r, "owner_id", None) or getattr(r, "agent_id", None)
        if owner:
            return owner
        run = getattr(ss, "runs", {}).get(getattr(r, "run_id", None))
        return getattr(run, "agent_id", None) if run is not None else None

    def _open_dispute_issue_for(self, target_object_id, tags):
        """v8d P1c: an existing OPEN escalation issue for the same (target, problem), so a
        second escalation supports it rather than spawning dispute_issue_dispute_7/_29 twins."""
        for a in self.product_artifacts.values():
            if getattr(a, "artifact_type", "") == "issue" and getattr(a, "status", "") != "resolved" \
                    and str(a.artifact_id).startswith("dispute_issue_") \
                    and getattr(a, "dispute_target", None) == target_object_id \
                    and (not tags or set(getattr(a, "dispute_tags", []) or []) & set(tags)):
                return a
        return None

    def _dispute_to_issue(self, d, tick: int) -> None:
        from environments.org_env.product.objects import ProductArtifact
        arts = self.product_artifacts
        # v8d P1c: one active escalation issue per (target, problem) — reuse it across disputes.
        existing = self._open_dispute_issue_for(d.target_object_id, d.issue_tags)
        if existing is not None:
            d.linked_issue_id = existing.artifact_id
            existing.support_count = int(getattr(existing, "support_count", 1) or 1) + 1
            return
        iid = f"dispute_issue_{d.dispute_id}"
        if iid in arts:
            d.linked_issue_id = iid
            return
        tags = ", ".join(d.issue_tags) or "evidence missing"
        art = ProductArtifact(
            artifact_id=iid, artifact_type="issue", status="open", priority="high",
            title=f"escalated dispute: {d.target_object_id}",
            problem=f"Unresolved dispute on {d.target_object_id} ({tags}); owner must reproduce/respond.",
            summary="escalated claim dispute", created_at_tick=tick, updated_at_tick=tick)
        art.__dict__["dispute_target"] = d.target_object_id
        art.__dict__["dispute_tags"] = list(d.issue_tags)
        arts[iid] = art
        d.linked_issue_id = iid
        # v8d P1a: an escalated evidence dispute ENFORCES the evidence/quality protocol
        # (it blocks the release gate until the owner responds).
        self.note_protocol_enforcement(("evidence", "claim", "credib", "traceab", "quality", "reproduc"),
                                       tick, obj=d.target_object_id, agent=d.challenger_id,
                                       actions=("request_reproduction", "export_result_to_tracker"),
                                       blocked=True,
                                       state_impact_ref=iid)
        if self.product is not None:
            self.product.artifact_ids.append(iid)
            if iid not in self.product.open_issue_ids:
                self.product.open_issue_ids.append(iid)

    def _close_dispute_issue(self, d, tick: int) -> None:
        """v8d P1c: when a dispute resolves, close its escalation issue so a resolved
        concern stops blocking the release gate / showing as an open high-risk issue."""
        iid = getattr(d, "linked_issue_id", None) or f"dispute_issue_{d.dispute_id}"
        art = self.product_artifacts.get(iid)
        if art is None:
            return
        art.status = "resolved"
        art.updated_at_tick = tick
        rb = art.__dict__.setdefault("resolved_by_dispute_ids", [])
        if d.dispute_id not in rb:
            rb.append(d.dispute_id)
        if self.product is not None and iid in (self.product.open_issue_ids or []):
            self.product.open_issue_ids.remove(iid)

    # -- universal backlog: make EVERY open issue workable, and keep iteration renewable -------
    _BACKLOG_TASK_EXCLUDE_PREFIXES = ("rel_blocker_", "proto_violation_")

    # How many feedback tasks may be OPEN at once. Twelve matches the seeded
    # coding backlog, so the two streams put comparable weight on the board
    # instead of one of them dominating it by however much happened to spawn.
    # Not a cap on the stream: an issue that finds no room queues and is
    # admitted when an open one closes.
    OPEN_FEEDBACK_TASK_CEILING = int(
        os.environ.get("ORG_OPEN_FEEDBACK_TASK_CEILING", "12") or 12
    )

    def _issue_link_artifacts(self, issue_id: str) -> list:
        try:
            from environments.org_env.product.substrates.issue_stream import _component_artifact_ids
            linked = _component_artifact_ids(self, issue_id)
        except Exception:
            linked = []
        return linked or self._artifacts_named_in_issue(issue_id)

    def _artifacts_named_in_issue(self, issue_id: str) -> list:
        """Repo files an issue's own text points at, when no component map does.

        The manifest maps historical OSS issues to components; customer and
        post-launch feedback has no such entry, so its tasks were created with
        no linked artifact at all. A task with nothing to change cannot show
        what it changed, so it can never satisfy completion: measured over 336
        ticks, all twelve code tasks carried an artifact and ten finished, while
        eleven feedback tasks carried none and not one did. That is not the
        organization declining to serve its customers — there was no move that
        would have counted.

        Matching is deliberately literal: a module stem has to appear as a word
        in the issue text. Guessing which file a vague complaint refers to would
        hand the organization a triage judgement that watching it make is the
        point.
        """
        artifact = (getattr(self, "product_artifacts", {}) or {}).get(issue_id)
        if artifact is None:
            return []
        text = " ".join(
            str(getattr(artifact, field, "") or "")
            for field in ("title", "problem", "summary", "description")
        ).lower()
        if not text.strip():
            return []
        import re

        words = set(re.findall(r"[a-z_][a-z0-9_]{2,}", text))
        hits = []
        for candidate in (getattr(self, "product_artifacts", {}) or {}).values():
            path = str(getattr(candidate, "linked_file_path", "") or "")
            if not path or getattr(candidate, "artifact_type", "") == "issue":
                continue
            stem = path.rsplit("/", 1)[-1].rsplit(".", 1)[0].lower()
            if len(stem) < 4:
                continue
            # "tableutils" in the text, or the plain word a module is named for
            # ("renderer" -> table_renderer.py). Substring matching on the whole
            # text would let "set" match setutils for any sentence about sets.
            if stem in words or stem in text:
                hits.append(str(getattr(candidate, "artifact_id", "")))
        return sorted(set(h for h in hits if h))[:3]

    def _close_tasks_that_meet_their_gate(self, tick: int) -> list:
        """Complete any task whose own completion requirements are satisfied.

        Every other place that evaluates requirements is reached through a
        patch landing on a linked artifact or a merge carrying implementation
        evidence. A task with no artifact reaches neither: a feedback task
        asked only for ``reviewed`` and ``multi_evidence`` can collect merges
        and reviews from four different people and never be looked at.

        Measured on b3_v10, two such tasks ended the run ``open`` at
        progress_score 0.0 while satisfying their gate outright — recomputing
        it scores them 1.0. The run reported 17 completions where the gate
        would have granted 19. The zero was "never asked", not "asked and
        refused", and the two are indistinguishable in the record.

        Asking every tick costs one pass over a few dozen tasks and removes
        the dependence on which call site happens to fire.
        """
        from environments.org_env.backend.entities import (
            COMPLETED_TASK_STATUSES,
            TaskStatus,
        )

        arts = self.product_artifacts
        closed = []
        for tid, t in self.tasks.items():
            prev = getattr(t.status, "value", str(t.status))
            if prev in COMPLETED_TASK_STATUSES:
                continue
            if not (getattr(t, "completion_requirements", None) or []):
                continue          # nothing declared: not this sweep's business
            if not (getattr(t, "progress_evidence", None) or []):
                continue          # no work recorded: nothing could have been met
            art = next((arts[a] for a in (getattr(t, "linked_artifacts", []) or [])
                        if a in arts), None)
            if not self._task_requirements_met(t, art):
                continue
            t.status = (TaskStatus.IMPLEMENTATION_DONE if self._is_product_task(t)
                        else TaskStatus.DONE)
            self._emit_task_transition(t, prev, "completion_sweep",
                                       "requirements_met",
                                       getattr(art, "artifact_id", None), tick)
            closed.append(tid)
        return closed

    def _reconcile_issue_backlog(self, tick: int, cap: int = 3) -> list:
        """Universal 'add a to-do' primitive: every OPEN issue — of ANY type or source (agent-filed
        via create_issue, market-derived, escalated disputes, product issues) — that lacks an open
        linked task gets one, so the org can ACT on it through the normal task loop, not only on coding
        issues. Deduped by a stable id + capped per tick; priority is floored so the task is actually
        surfaced for pickup (perception gates unowned tasks at priority>=4). Release-gate/protocol
        blocker issues are excluded (handled by their own drivers)."""
        from environments.org_env.backend.entities import (
            COMPLETED_TASK_STATUSES,
            Task,
            TaskStatus,
        )
        tasks = self.tasks
        # dedup canonical (#3): an issue that ALREADY has a task — ANY status, including the OSS-seeded
        # task_oss_<issue> or one that's since been done/merged — must NOT get a second task_iss_<issue>.
        # One historical issue -> one internal task (avoids task_oss_*/task_iss_* twins that polluted
        # tasks_done / ownership / backlog / capability metrics).
        covered = set()
        for t in tasks.values():
            for iid in (getattr(t, "linked_issues", []) or []):
                covered.add(iid)
        created: list = []

        def _prio(sev):
            return 5 if str(sev).lower() in ("critical", "high", "major") else 4

        # A board that grows without bound is not a harder board, it is a
        # different one. The seeded coding issues arrive on a fixed clock and
        # number twelve in every run; the feedback tasks generated beside them
        # ranged from 4 to 21 across four runs of the same pack and seed, so
        # the size of the backlog an arm faced was not held constant. Holding
        # the number of OPEN feedback tasks under a ceiling keeps the amount of
        # unaddressed work an organization is looking at comparable, while
        # leaving the stream itself endogenous: close one and the next is
        # admitted. Resolved issues still queue behind it rather than vanishing.
        open_feedback = sum(
            1 for tid, t in tasks.items()
            if str(tid).startswith("task_iss_")
            and str(getattr(getattr(t, "status", None), "value",
                            getattr(t, "status", ""))) not in COMPLETED_TASK_STATUSES
        )
        room = max(0, self.OPEN_FEEDBACK_TASK_CEILING - open_feedback)

        def _mk(iid, title, desc, sev, artifact_ids):
            if len(created) >= min(cap, room) or iid in covered \
                    or f"task_iss_{iid}" in tasks:
                return
            if any(str(iid).startswith(px) for px in self._BACKLOG_TASK_EXCLUDE_PREFIXES):
                return
            tid = f"task_iss_{iid}"
            task = Task(task_id=tid, title=(title or f"resolve {iid}")[:120],
                        description=(desc or ""), status=TaskStatus.OPEN, owner_id=None,
                        priority=_prio(sev), linked_issues=[iid],
                        linked_artifacts=list(artifact_ids or []), visibility="team")
            if not artifact_ids:
                # Nothing in the repository to change, so "artifact_revised" and
                # "gaps_cleared" can never be satisfied — the first also
                # explicitly excludes issue artifacts, which is all such a task
                # has. Over 336 ticks every one of the eleven feedback tasks sat
                # unfinished for exactly this reason while ten of twelve code
                # tasks completed, and the run reported a 0.53 completion rate
                # over a denominator that included work with no reachable
                # finish.
                #
                # The bar is not lowered, only made about the right thing: a
                # customer question is answered, not compiled. Review evidence
                # and two distinct acts are still required, so one agent
                # declaring it handled still does not close it.
                task.completion_requirements = ["reviewed", "multi_evidence"]
            tasks[tid] = task
            self.board.tasks.append(tid)
            covered.add(iid)
            created.append(tid)

        for a in list(self.product_artifacts.values()):
            if getattr(a, "artifact_type", "") != "issue":
                continue
            if getattr(a, "status", "open") not in ("open", "in_progress", "reopened"):
                continue
            _mk(a.artifact_id, getattr(a, "title", ""),
                getattr(a, "problem", "") or getattr(a, "summary", ""),
                getattr(a, "priority", "medium"), self._issue_link_artifacts(a.artifact_id))
        for iid, iss in list(getattr(self, "issues", {}).items()):
            if getattr(iss, "status", "open") not in ("open", "in_progress", "triaged", "reopened"):
                continue
            _mk(iid, getattr(iss, "title", ""), getattr(iss, "description", ""),
                getattr(iss, "severity", "medium"), [])
        return created

    def _reconcile_failing_test_backlog(self, tick: int, cap: int = 3) -> list:
        """Turn a test the org's own suite keeps failing into tracked, owned work.

        A repeat detection used to carry no more weight than the first, so the
        organization could observe the same failure indefinitely and never act:
        measured on a 336-tick run, one regression it introduced itself was
        detected at t225, t250 and t278 and was still failing at the end.

        Reopening matters as much as creating. When a task for this test was
        already marked done and the test is failing again, the organization's
        own evidence contradicts its own completion claim, and that claim is
        what must yield — otherwise "declared fixed" permanently outranks
        "demonstrably broken".

        Uses only agent-visible evidence (the public suite in the starter repo);
        hidden oracles never reach this path.
        """
        from environments.org_env.backend.entities import Task, TaskStatus
        from environments.org_env.product.test_history import persistently_failing_tests

        failing = persistently_failing_tests(self)
        if not failing:
            return []
        tasks = self.tasks
        touched: list = []
        done_states = (TaskStatus.DONE, TaskStatus.MERGED, TaskStatus.IMPLEMENTATION_DONE)
        for test_id in failing:
            if len(touched) >= cap:
                break
            tid = "task_test_" + re.sub(r"[^0-9a-zA-Z]+", "_", str(test_id)).strip("_")[:80]
            existing = tasks.get(tid)
            if existing is not None:
                if getattr(existing, "status", None) in done_states:
                    existing.status = TaskStatus.OPEN
                    existing.progress_score = 0.0
                    self._append_progress_evidence(
                        existing, actor="verification", tick=tick,
                        evidence_type="completion_contradicted_by_test",
                        detail=str(test_id)[:200])
                    touched.append(tid)
                continue
            tasks[tid] = Task(
                task_id=tid,
                title=f"repair failing test: {test_id}"[:120],
                description=(
                    f"{test_id} has failed in more than one public-test run. "
                    "Reproduce it, fix the cause, and re-run the suite to show it passing."
                )[:400],
                status=TaskStatus.OPEN, owner_id=None,
                priority=5, linked_issues=[], linked_artifacts=[], visibility="team")
            self.board.tasks.append(tid)
            touched.append(tid)
        return touched

    def _append_progress_evidence(self, task, *, actor: str, tick: int,
                                  evidence_type: str, detail: str = "") -> None:
        """Record one auditable progress fact on a task without scoring it."""
        evidence = getattr(task, "progress_evidence", None)
        if evidence is None:
            return
        evidence.append({"actor": actor, "tick": int(tick),
                         "evidence_type": evidence_type, "detail": detail})

    def _renew_backlog_if_idle(self, tick: int) -> Optional[str]:
        """Keep iteration renewable: when the coding backlog is drained AND a shipped release still has
        an unmet market signal (an open customer ticket with no issue behind it), file ONE improvement
        issue so the org keeps improving instead of idling. Self-limiting — only fires when otherwise
        idle, deduped, and capped at 2 concurrent auto-improvement issues (never busywork while real
        coding work remains)."""
        from environments.org_env.product.substrates.eval_assets import is_oss_substrate
        if not is_oss_substrate(self):
            return None
        from environments.org_env.product.substrates.issue_stream import unpatched_coding_issues
        if unpatched_coding_issues(self):
            return None                                   # real coding work remains -> don't invent more
        arts = self.product_artifacts
        autos = [a for a in arts.values()
                 if getattr(a, "artifact_type", "") == "issue"
                 and str(a.artifact_id).startswith("issue_improve_")
                 and getattr(a, "status", "") in ("open", "in_progress", "reopened")]
        if len(autos) >= 2:
            return None
        tickets = self.__dict__.get("tickets") or {}
        for tid, t in tickets.items():
            st = t.get("status") if isinstance(t, dict) else getattr(t, "status", None)
            rst = t.get("response_status") if isinstance(t, dict) else getattr(t, "response_status", None)
            if st in ("resolved", "closed") or rst == "resolved":
                continue
            topic = t.get("topic") if isinstance(t, dict) else getattr(t, "topic", "")
            text = t.get("complaint_or_request") if isinstance(t, dict) else getattr(t, "complaint_or_request", "")
            iid = f"issue_improve_{tid}"
            if iid in arts:
                continue
            from environments.org_env.product.objects import ProductArtifact
            arts[iid] = ProductArtifact(
                artifact_id=iid, artifact_type="issue", status="open", priority="medium",
                title=f"improve: {str(text or topic or tid)[:60]}",
                problem=f"Unmet customer signal from {tid} ({topic}): {str(text or '')[:200]}",
                summary="renewal: unmet market signal", created_at_tick=tick, updated_at_tick=tick)
            if self.product is not None:
                self.product.artifact_ids.append(iid)
                if iid not in self.product.open_issue_ids:
                    self.product.open_issue_ids.append(iid)
            self.events.append({"type": "product_event", "subtype": "backlog_renewal",
                                "artifact_id": iid, "source_ticket": tid, "tick": int(tick), "auto": True})
            return iid
        return None

    # -- spec #6: split oversized work into pending units worked through over ticks ------
    def enqueue_work_split(self, agent_id, action_type, extra_units, res, tick, workload, task_id=None):
        if extra_units <= 0:
            return None
        self._pending_seq = getattr(self, "_pending_seq", 0) + 1
        jid = f"job_{self._pending_seq}"
        job = {"job_id": jid, "agent_id": agent_id, "action_type": action_type,
               "units_total": extra_units + 1, "units_remaining": extra_units, "status": "pending",
               "created_tick": tick, "workload": round(float(workload), 2), "task_id": task_id,
               "parent_action_id": getattr(res, "action_id", None)}
        self.pending_jobs.append(job)
        self.events.append({"type": "work_split_event", "job_id": jid, "agent_id": agent_id,
                            "action_type": action_type, "units": extra_units + 1,
                            "workload": job["workload"], "tick": tick})
        return job

    def process_pending_jobs(self, tick: int) -> list:
        """Advance split work units: a free agent completes one unit per tick of their
        oldest pending job, so a big feature spans ticks (waiting while busy)."""
        advanced = []
        for job in self.pending_jobs:
            if job["status"] != "pending":
                continue
            ws = getattr(self.agents.get(job["agent_id"]), "work_state", None)
            if ws is not None and int(getattr(ws, "next_available_tick", 0) or 0) > tick:
                continue                               # agent still busy -> the unit waits
            job["units_remaining"] -= 1
            if ws is not None:
                ws.next_available_tick = max(int(getattr(ws, "next_available_tick", 0) or 0), tick + 1)
            done = job["units_remaining"] <= 0
            if done:
                job["status"] = "done"
            self.events.append({"type": "work_unit_completed", "job_id": job["job_id"],
                                "agent_id": job["agent_id"], "tick": tick,
                                "remaining": max(0, job["units_remaining"]), "final": done})
            advanced.append(job["job_id"])
        return advanced

    # -- v8 Gap2: auto-convene ONE high-value meeting per persistent trigger -------------
    def process_meetings(self, tick: int) -> list:
        """Convene a release-gate-review (persistent blocked RC) or dispute-resolution
        (escalated dispute) meeting — with agenda + decision + action item — so the
        high-value meeting types actually happen. Once per RC / dispute (anti-churn)."""
        ms = getattr(self, "meeting_system", None)
        if ms is None:
            return []
        convened = self.__dict__.setdefault("_auto_meeting_keys", set())
        plan = []
        for rc in self.repo_system.repo.release_candidates.values():
            if rc.status == "blocked" and rc.blockers \
                    and tick - _tick_value(getattr(rc, "created_at_tick", None), tick) >= 6:
                key = ("release_gate_review", rc.candidate_id)
                if key not in convened:
                    plan.append((key, f"clear release blockers: {', '.join(rc.blockers[:3])}",
                                 f"assign owners + patches for: {', '.join(rc.blockers[:2])}",
                                 list(rc.blockers[:2])))
                break
        cr = getattr(self, "commitment_registry", None)
        for d in (cr.disputes.values() if cr else []):
            if d.status == "escalated":
                key = ("dispute_resolution", d.dispute_id)
                if key not in convened:
                    plan.append((key, f"resolve escalated dispute on {d.target_object_id}",
                                 f"require owner reproduction/trace for {d.target_object_id}", []))
                break
        out = []
        for key, agenda_item, decision, gates in plan:
            mtype = key[0]
            leads = [a for a in self.agents if getattr(self.agents[a], "is_founder", False)]
            parts = list(dict.fromkeys((leads or list(self.agents)) + list(self.agents)))[:4]
            m = ms.schedule_meeting(created_by=parts[0], meeting_type=mtype,
                                    title=mtype.replace("_", " "), participants=parts,
                                    scheduled_tick=tick, agenda=[agenda_item])
            ms.start_meeting(m.meeting_id, tick=tick)
            ms.attend(parts[0], m.meeting_id)            # v14 P4: the convening lead attends (no 0-attendee notes)
            note = ms.record_meeting_notes(agent_id=parts[0], meeting_id=m.meeting_id,
                                           summary=agenda_item, decisions=[decision], tick=tick)
            # Tag the tracking items with the gates they exist to track, so the
            # blocker-cleared auto-close matches this tag instead of guessing from
            # the description text (which agent-created items may also mention).
            for ai_id in (note.action_items or []):
                ai = ms.action_items.get(ai_id)
                if ai is not None:
                    ai.linked_gates = list(gates)
            ms.close_meeting(m.meeting_id, tick=tick)
            convened.add(key)
            self.events.append({"type": "meeting_event", "subtype": "auto_convened", "auto": True,
                                "meeting_id": m.meeting_id, "meeting_type": mtype, "tick": tick})
            out.append(m.meeting_id)
        return out

    def process_action_items(self, tick: int) -> list:
        """v8d P2a: close the loop on meeting action items — a done linked task completes the
        item; an overdue item is reassigned to another attendee (not left open on one person)."""
        ms = getattr(self, "meeting_system", None)
        if ms is None:
            return []
        from environments.org_env.backend.entities import COMPLETED_TASK_STATUSES, Task, TaskStatus
        changed = []
        for ai in ms.action_items.values():
            if ai.status not in ("open", "overdue"):
                continue
            # v8e #5: a dangling action item becomes a real follow-up TASK (owned by the
            # assignee, linked to the meeting), so meeting decisions reach execution instead
            # of sitting as open items with follow_up_tasks == 0.
            if not ai.linked_task_id:
                tid = f"task_aitem_{ai.action_item_id}"
                if tid not in self.tasks:
                    m = ms.meetings.get(ai.meeting_id)
                    blockers = [b for b in (getattr(m, "agenda", []) or []) if "blocker" in str(b).lower()]
                    self.tasks[tid] = Task(
                        task_id=tid, title=(ai.description or "meeting follow-up")[:80],
                        description=ai.description or "", owner_id=ai.assignee_id,
                        status=TaskStatus.OPEN, priority=4, deadline_tick=ai.due_tick,
                        linked_issues=blockers)
                    if ai.assignee_id:
                        self.board.owners[tid] = ai.assignee_id
                    if m is not None and tid not in (m.follow_up_tasks or []):
                        m.follow_up_tasks.append(tid)
                    self.events.append({"type": "meeting_event", "subtype": "action_item_task_created",
                                        "action_item_id": ai.action_item_id, "task_id": tid,
                                        "agent_id": ai.assignee_id, "tick": tick, "auto": True})
                ai.linked_task_id = tid
                changed.append(ai.action_item_id)
            t = self.tasks.get(ai.linked_task_id) if ai.linked_task_id else None
            if t is not None and getattr(t.status, "value", str(t.status)) in COMPLETED_TASK_STATUSES:
                ai.status = "done"
                self.events.append({"type": "meeting_event", "subtype": "action_item_done",
                                    "action_item_id": ai.action_item_id, "tick": tick, "auto": True})
                changed.append(ai.action_item_id)
                continue
            if ai.due_tick is not None and tick > int(ai.due_tick):
                m = ms.meetings.get(ai.meeting_id)
                roster = list(getattr(m, "attendees", None) or getattr(m, "participants", None) or [])
                prev_assignee = ai.assignee_id
                if t is not None and t.owner_id not in (prev_assignee, None):
                    # An agent explicitly reassigned the task since the last rotation:
                    # adopt that choice as the new coupling point and grant one fresh
                    # window instead of immediately rotating the chosen owner off.
                    # Governance resumes next overdue — a permanent exemption would let
                    # a self-assignment rot forever, the very defect this loop repairs.
                    ai.assignee_id = t.owner_id
                elif roster:
                    nxt = roster[(roster.index(ai.assignee_id) + 1) % len(roster)] \
                        if ai.assignee_id in roster else roster[0]
                    if nxt != ai.assignee_id:
                        ai.assignee_id = nxt
                ai.status = "overdue"
                ai.reassigned_count += 1
                ai.due_tick = tick + 24
                # The rotation must move the WORK, not just the item: sync the linked
                # follow-up task's owner/deadline (and the board registry the protocol
                # detectors read) so the new assignee actually sees the task as theirs.
                task_synced = False
                if t is not None:
                    t.owner_id = ai.assignee_id
                    self.board.owners[t.task_id] = ai.assignee_id
                    t.deadline_tick = ai.due_tick
                    task_synced = True
                self.events.append({"type": "meeting_event", "subtype": "action_item_overdue",
                                    "action_item_id": ai.action_item_id, "assignee_id": ai.assignee_id,
                                    "task_id": ai.linked_task_id, "task_synced": task_synced,
                                    "tick": tick, "auto": True})
                changed.append(ai.action_item_id)
        return changed

    def validate_product_artifact_links(self) -> bool:
        """Preflight v3 §6.4 hard invariant: every revised artifact has a causal link."""
        for art in (getattr(self, "product_artifacts", {}) or {}).values():
            if int(getattr(art, "revision", 0) or 0) > 0:
                if not (art.linked_action_ids or art.linked_episode_ids):
                    raise AssertionError(f"orphan product change: {art.artifact_id} rev>0 with no link")
        return True

    def _summarize_closed_episodes(self) -> None:
        """Summarize newly-closed episodes (preflight §3: reflection is now a
        half-day batch, so episode close only triggers a summary, not reflection)."""
        for ep in self.episode_manager.episodes.values():
            if ep.status != "open" and ep.episode_id not in self._reflected_episode_ids:
                self._reflected_episode_ids.add(ep.episode_id)
                if self._cog is not None:           # Phase 6: summarize the closed episode
                    self.episode_summaries[ep.episode_id] = \
                        self._cog["episode_summarizer"].summarize(ep, self, self.llm_client)

    # world-level events (meeting lifecycle / commitment violation / background /
    # market shock) aren't single ExecutionResults — feed them to the episode layer
    # without double-counting action events (which observe_result already saw).
    _EPISODE_WORLD_EVENTS = {
        ("meeting_event", "attended"), ("meeting_event", "closed"),
        ("commitment_event", "violation"), ("background_job_event", "completed"),
        ("external_signal_event", "api_price_shock"),
        # v14 P5: market-validation signals open/extend a feedback episode so customer
        # trials & churn actually drive an internal product-change loop.
        ("external_signal_event", "customer_trial"),
        ("external_signal_event", "post_launch_feedback"),
        ("external_signal_event", "human_product_feedback"),
    }

    def _observe_world_episodes(self) -> None:
        for raw in self.events[self._ep_mark:]:
            if (raw.get("type"), raw.get("subtype")) in self._EPISODE_WORLD_EVENTS:
                self.episode_manager.observe_world_event(raw, self)

    def apply_action(self, agent_id: str, action_type: str, **params):
        """Execute ONE action through the full pipeline (execute → appraise → log →
        event graph → episode layer) and return the ExecutionResult. Generic helper
        for scripted/demo causal chains + tests (no per-agent logic). The caller
        manages the clock (advance + ``update_open_episodes``)."""
        if self._loop is None:
            self._wire_loop()
        from relic.decision.contracts import ActionCandidate
        cand = ActionCandidate(action_type=action_type, parameters=dict(params))
        result = self._loop["execution"].execute(agent_id, cand, self)
        appraised = self._loop["appraisal"].appraise(result, self)
        self._log_events(result, appraised)
        self._update_memory(agent_id, appraised)
        from environments.org_env.experiments.ablations import EVENT_GRAPH, mechanism_disabled
        if not mechanism_disabled(self, EVENT_GRAPH):
            self.event_graph.ingest(result)
        eps = self.episode_manager.observe_result(result, self)
        self._link_product_artifacts(result, eps, agent_id)
        self._record_review_evidence(result, agent_id)
        return result

    def reflect_agent(self, agent_id: str, *, episode=None, reason: str = "manual"):
        """Trigger one agent reflection through the full pipeline (reflection ->
        memory/log/event/event-graph -> wish extraction). Generic helper for
        scripted/demo chains + tests; the autonomous loop triggers on episode close."""
        ep = None
        if isinstance(episode, str):
            ep = self.episode_manager.episodes.get(episode)
        elif episode is not None:
            ep = episode
        return self.reflection_manager.reflect(agent_id, self, episode=ep, reason=reason)

    def _check_commitments(self, tick) -> None:
        for c in self.commitment_registry.overdue_commitments(tick):
            c.status = "violated"
            ev = {"type": "commitment_event", "subtype": "violation", "agent_id": c.agent_id,
                  "tick": tick, "commitment_id": c.commitment_id}
            self.events.append(ev)
            self.appraised_log.append({"type": "commitment_event", "agent_id": c.agent_id,
                                       "tick": tick, "salience": 0.7})
            agent = self.agents.get(c.agent_id)
            if agent is not None and getattr(agent, "org_state", None) is not None:
                agent.org_state.trust_in_company = max(0.0, agent.org_state.trust_in_company - 0.05)
            from environments.org_env.experiments.ablations import EVENT_GRAPH, mechanism_disabled
            if self.event_graph is not None and not mechanism_disabled(self, EVENT_GRAPH):
                self.event_graph.add_edge(c.agent_id, "violated", c.commitment_id)

    # -- §13.2 can the agent act this tick? --------------------------------
    def can_agent_act(self, agent, clk) -> bool:
        av = self.time.availability.get(agent.id)
        ws = getattr(agent, "work_state", None)
        if ws is not None and ws.is_busy(clk.current_tick):
            return False                              # mid blocking/background activity
        # Being mid-action is scheduling, so it still blocks above; everything
        # below is the lived-body layer and goes away with the ablation.
        if not self.time.rhythm_enabled:
            return True
        if agent.vitals.get("attention", 1.0) <= 0.04:
            return False
        if av and av.current_availability_status in ("asleep", "offline", "forced_rest", "resting"):
            return False
        if clk.phase_of_day == "night_sleep":
            return bool(av and av.after_hours_responsiveness > 0.7)   # rare night owls
        if clk.is_weekend:
            return bool(av and av.weekend_work_tendency >= 0.5)
        if clk.is_after_hours or clk.is_late_night:
            return bool(av and av.after_hours_responsiveness >= 0.4)
        if clk.phase_of_day == "lunch_low_activity":
            return False
        return True   # normal work hours

    # -- O1.6 scheduling passes -------------------------------------------
    def _constrain_candidates(self, agent, candidates, clk):
        """Drop candidates the agent can't take this tick (aux slot exhausted /
        after-hours/weekend-forbidden). Spec §29 apply_duration_constraints."""
        from environments.org_env.backend.actions import (
            CAT_GOVERNANCE,
            CAT_PROTOCOL,
            CAT_RELEASE,
            CAT_REPO,
            CAT_SANDBOX,
            action_category,
        )
        from environments.org_env.backend.actions.duration import DURATION_REGISTRY
        from environments.org_env.experiments.ablations import (
            INSTITUTIONALIZATION,
            PRODUCT_WORKFLOW,
            mechanism_disabled,
        )
        institutionalization_disabled = (
            not self.institutionalization_enabled
            or mechanism_disabled(self, INSTITUTIONALIZATION)
        )
        product_workflow_disabled = mechanism_disabled(self, PRODUCT_WORKFLOW)
        ws = getattr(agent, "work_state", None)
        out = []
        for c in candidates:
            if institutionalization_disabled and (
                action_category(c.action_type) in (CAT_PROTOCOL, CAT_GOVERNANCE)
                or c.action_type == "use_tool"
            ):
                continue
            if product_workflow_disabled and action_category(c.action_type) in (
                CAT_REPO, CAT_SANDBOX, CAT_RELEASE
            ):
                continue
            spec = DURATION_REGISTRY.get(c.action_type)
            if spec.is_auxiliary_speech and ws is not None and ws.aux_speech_slots_remaining <= 0:
                continue
            if not spec.can_execute_after_hours and (clk.is_after_hours or clk.is_late_night):
                continue
            if not spec.can_execute_weekend and clk.is_weekend:
                continue
            out.append(c)
        return out

    def _maybe_reactive_aux_speech_while_busy(self, aid, agent, tick) -> None:
        """A busy agent may still fire ONE auxiliary ack to an urgent unread
        mention (spec §29 maybe_reactive_aux_speech_while_busy)."""
        ws = getattr(agent, "work_state", None)
        if ws is None or ws.aux_speech_slots_remaining <= 0:
            return
        urgent = [m for m in self.comm.perceivable_messages(aid)
                  if aid not in m.read_by and m.urgency in ("urgent", "incident")]
        if not urgent:
            return
        m = urgent[0]
        self.comm.mark_read(aid, m.message_id)
        ws.aux_speech_slots_remaining -= 1
        self.events.append({"type": "communication_event", "subtype": "reactive_ack",
                            "agent_id": aid, "tick": tick, "message_id": m.message_id})
        self.action_log.append({"agent_id": aid, "action_type": "acknowledge_message",
                                "tick": tick, "success": True})

    def _active_meeting_for(self, aid, ws):
        mid = getattr(ws, "current_meeting_id", None)
        if not mid:
            return None
        m = self.meeting_system.meetings.get(mid)
        if m is not None and m.status.value == "active" and aid in m.participants:
            return m
        return None

    def _run_meeting_subaction(self, aid, agent, m, tick) -> None:
        """During an active meeting a participant takes ONE meeting sub-action
        (record notes if missing, else summarize/assign) — skill-weighted, loop-
        driven, so meetings actually produce notes (daily_sync_cadence detector)."""
        from relic.decision.contracts import ActionCandidate
        loop = self._loop
        cands = []
        if m.notes_doc_id is None:
            # Notes-first is a structural gate, not a score race: the
            # daily_sync_cadence detector requires notes to exist, so no other
            # sub-action competes until they do.
            cands.append(ActionCandidate(action_type="record_meeting_notes",
                                         parameters={"meeting_id": m.meeting_id}))
        else:
            cands.append(ActionCandidate(action_type="summarize_decision",
                                         parameters={"meeting_id": m.meeting_id}))
            cands.append(ActionCandidate(action_type="assign_action_item",
                                         parameters={"meeting_id": m.meeting_id, "description": "follow up"}))

        def _score(c):
            # Per-candidate skill channel: a common multiplier over all candidates
            # cancels out of the argmax, fixing the choice by base weight alone and
            # making assign_action_item unreachable. Summaries exercise the
            # documentation channel; assigning follow-ups exercises facilitation.
            base = {"record_meeting_notes": 1.0, "summarize_decision": 0.4,
                    "assign_action_item": 0.3}.get(c.action_type, 0.1)
            if c.action_type == "assign_action_item":
                sk = agent.skill("meeting_facilitation", 0.0)
            else:
                sk = agent.skill("documentation", 0.3)
            return base * (0.3 + sk)

        sel = max(cands, key=_score)
        result = loop["execution"].execute(aid, sel, self)
        appraised = loop["appraisal"].appraise(result, self)
        self._log_events(result, appraised)
        self._update_memory(aid, appraised)
        from environments.org_env.experiments.ablations import EVENT_GRAPH, mechanism_disabled
        if not mechanism_disabled(self, EVENT_GRAPH):
            self.event_graph.ingest(result)
        self.episode_manager.observe_result(result, self)

    def _process_background_jobs(self, tick) -> None:
        for job in self.time.due_background_jobs(tick):
            job.status = "completed"
            owner = self.agents.get(job.agent_id)
            if owner is not None and job.job_id in owner.work_state.background_jobs:
                owner.work_state.background_jobs.remove(job.job_id)
            ev = {"type": "background_job_event", "subtype": "completed", "agent_id": job.agent_id,
                  "tick": tick, "job_id": job.job_id, "result_id": job.result_id}
            self.events.append(ev)
            self.appraised_log.append({"type": "background_job_event", "agent_id": job.agent_id,
                                       "tick": tick, "salience": 0.4})

    def _start_new_day_all(self, tick) -> None:
        for agent in self.agents.values():
            slots = getattr(agent.work_state, "default_aux_speech_slots", 1) \
                if getattr(agent, "work_state", None) else 1
            self.time.start_new_day(agent, default_aux_slots=slots)

    def _process_meetings(self, tick) -> None:
        from environments.org_env.backend.meetings.system import meeting_duration
        ms = self.meeting_system
        for m in ms.due_to_start(tick):
            ms.start_meeting(m.meeting_id, tick)
            dur = meeting_duration(m.meeting_type)
            for pid in m.participants:
                agent = self.agents.get(pid)
                av = self.time.availability.get(pid)
                if agent is None:
                    continue
                if av and av.current_availability_status in ("offline", "asleep", "forced_rest"):
                    ms.skip_meeting(pid, m.meeting_id)
                    continue
                ms.attend(pid, m.meeting_id)
                ws = getattr(agent, "work_state", None)
                if ws is not None:
                    ws.next_available_tick = max(ws.next_available_tick, tick + dur)
                    ws.daily_meeting_count += 1
                    ws.current_meeting_id = m.meeting_id     # in-meeting -> sub-actions allowed
                self.time.log_work(agent, action_type="attend_meeting", is_deep_work=False, duration=dur)
                self.action_log.append({"agent_id": pid, "action_type": "attend_meeting",
                                        "tick": tick, "success": True})
                self.events.append({"type": "meeting_event", "subtype": "attended",
                                    "meeting_id": m.meeting_id, "agent_id": pid, "tick": tick})
                self.appraised_log.append({"type": "meeting_event", "agent_id": pid,
                                           "tick": tick, "salience": 0.6})
        for m in ms.due_to_close(tick):
            ms.close_meeting(m.meeting_id, tick)
            for pid in m.participants:
                a = self.agents.get(pid)
                if a is not None and getattr(a, "work_state", None) \
                        and a.work_state.current_meeting_id == m.meeting_id:
                    a.work_state.current_meeting_id = None
            self.events.append({"type": "meeting_event", "subtype": "closed",
                                "meeting_id": m.meeting_id, "tick": tick})

    def _process_inbox(self, aid: str, tick: int) -> None:
        """v14 P3: the acting agent READS its inbox (read receipts become real instead of
        read_by=1) and writes the decision-relevant messages into memory, so communication
        actually propagates: message -> read/ack -> memory entry -> influences reflection/action.

        The memory half of that sentence was never implemented. Only a counting
        summary reached agent_log, which nothing in any prompt reads, so over a
        336-tick run 256 messages were marked read and not one word of their
        content reached a single LLM call: not the decision prompt, not memory
        via context_for_decision, and not reflection, which filters message
        objects out by design. Marked-as-read was the only trace communication
        left, and it is set unconditionally here, so it was never evidence that
        anyone had taken anything in.
        """
        comm = getattr(self, "comm", None)
        if comm is None:
            return
        try:
            read = comm.triage_inbox(aid, tick)
        except Exception:
            return
        relevant = [m for m in read if getattr(m, "importance", "") == "decision_relevant"
                    or getattr(m, "urgency", "") in ("urgent", "incident") or aid in getattr(m, "mentions", [])]
        self._remember_the_conversation(aid)
        # This runs immediately before perception is built, so by prompt time
        # every message reads as already-read and "unread" carries no
        # information. What is actually new to the agent is what it took in on
        # this pass, so that is what perception marks.
        self.__dict__.setdefault("_newly_read_messages", {})[aid] = {
            str(getattr(m, "message_id", "")) for m in read
        }
        if relevant and self.agent_log is not None:
            from environments.org_env.reflection.objects import AgentLogEntry
            top = relevant[0]
            self.agent_log.append(AgentLogEntry(
                log_id=f"log_{len(self.agent_log)}", agent_id=aid, tick=tick, entry_type="inbox_read",
                summary=(f"read {len(read)} message(s); {len(relevant)} decision-relevant "
                         f"(e.g. {top.sender_id}: {top.text_summary[:80]})"),
                related_episode_ids=[],
                raw_payload={"read": len(read), "relevant_ids": [m.message_id for m in relevant[:8]]}))

    def _remember_the_conversation(self, aid: str) -> None:
        """Keep the transcript of every channel this agent is in.

        Memory is the one channel that reaches every condition. B0-B2 select
        actions with the LLM and see it through context_for_decision; B3 selects
        with the profile policy, never reads that prompt, and sees it only
        through reflection. Writing here therefore carries the conversation into
        all four rungs without touching how any of them chooses.

        Everything the agent can see is recorded, including what it said itself
        — a transcript missing your own turns cannot answer what you committed
        to — and including ordinary traffic, because which line mattered is a
        judgement to make when reading it back, not a filter to apply on the way
        in. Ordering is by tick, so the log reads as the conversation happened
        rather than as the order this agent happened to catch up.

        This sweeps rather than hooking each send handler: send_message,
        reply_thread, send_async_update and the rest all funnel into the same
        channels, so one pass over what is perceivable cannot miss a path that
        a per-handler hook would.
        """
        manager = getattr(self, "reflection_manager", None)
        comm = getattr(self, "comm", None)
        if manager is None or comm is None or getattr(self, "agent_memories", None) is None:
            return
        memory = manager._memory(self, aid)
        try:
            visible = comm.perceivable_messages(aid)
        except Exception:
            return
        for message in sorted(visible, key=lambda m: int(getattr(m, "created_tick", 0) or 0)):
            text = str(getattr(message, "text_summary", "") or "").strip()
            if not text:
                continue
            sender = str(getattr(message, "sender_id", "") or "someone")
            speaker = "you" if sender == aid else sender
            memory.record_turn(
                getattr(message, "message_id", ""),
                f"[t{int(getattr(message, 'created_tick', 0) or 0)}] {speaker}: {text}",
            )

    def _prewarm_product_smoke(self) -> None:
        """v14 P0: run the seed product's smoke once at build to populate the smoke cache, so a
        runnable beta reads as cli_runnable=true at t0 instead of 'unknown -> not ready'.

        Triggers on ORG_LLM (real runs), ORG_PRODUCT_PREWARM_SMOKE, OR the scenario
        ``evaluator_config.prewarm_smoke`` (brief review §3: OSS runtime readiness must NOT depend on
        ORG_LLM or a manual env var — a formal OSS scenario sets prewarm_smoke so the real starter is
        recognized as runnable independent of the cognition layer). Off by default in unit tests."""
        import os
        gate = ("ORG_LLM", "ORG_PRODUCT_PREWARM_SMOKE")
        on = any((os.environ.get(g, "") or "").lower() in ("1", "true", "yes", "on") for g in gate)
        if not on:
            try:
                from environments.org_env.product.substrates.eval_assets import oss_eval_enabled
                on = oss_eval_enabled(self, "prewarm_smoke", "ORG_PRODUCT_PREWARM_SMOKE")
            except Exception:
                on = False
        if not on:
            return
        try:
            from environments.org_env.product.materialize import release_smoke
            release_smoke(self, prefer_mainline=False)
        except Exception:
            pass

    def _update_milestone(self) -> None:
        """v13: persist the WORK-MODE milestone stage on the product (distinct from the legacy
        release-lifecycle `product.stage`): discovery -> stabilization -> integration ->
        market_validation. The integration stage = shipping mode (off-task busywork down-weighted)."""
        if getattr(self, "product", None) is None:
            return
        try:
            from environments.org_env.product.milestone import current_stage
            self.product.milestone_stage = current_stage(self)
        except Exception:
            pass

    def _evaluate_milestones(self):
        """spec #5: milestones grounded in real product state; the customers milestone is
        grounded in the v14 P5 external market-validation loop (paying conversions + WTP).
        Returns (met_conditions, status_dict, score_by_condition)."""
        from environments.org_env.product.objects import artifact_purpose
        arts = self.product_artifacts
        gaps = list(self.known_gaps.values())
        seed_crit = sum(1 for g in gaps if g.critical) or 1
        active_crit = sum(1 for g in gaps if g.critical and g.status in ("active", "regressed"))
        rcs = self.repo_system.repo.release_candidates
        gate_attempted = any(getattr(rc, "gate_results", None) for rc in rcs.values())
        merged = sum(1 for pr in self.repo_system.repo.pull_requests.values()
                     if getattr(pr.status, "value", str(pr.status)) == "merged")
        done_tasks = sum(1 for t in self.tasks.values()
                         if getattr(t.status, "value", str(t.status)) in ("merged", "released", "done"))

        def _cap(purpose, *kw):
            for a in arts.values():
                if artifact_purpose(getattr(a, "linked_file_path", "") or a.artifact_id) == purpose:
                    caps = [str(c).lower() for c in (getattr(a, "capabilities", []) or [])]
                    if (not kw) or any(k in c for k in kw for c in caps):
                        return True
            return False

        # v8d P0b: a blocked candidate only partially credits "demo" (a candidate exists but
        # the gate did not pass), so demo != release-ready.
        rc_ok = any(getattr(rc, "status", "") in ("approved", "released") for rc in rcs.values())
        rc_blocked = any(getattr(rc, "status", "") == "blocked" for rc in rcs.values())
        demo_rc = 1.0 if rc_ok else (0.5 if (rc_blocked or gate_attempted or rcs) else 0.0)
        demo = 0.5 * demo_rc + 0.5 * (1.0 - active_crit / seed_crit)
        quality = (int(_cap("claim_tracker", "evidence", "source"))
                   + int(_cap("report_quality")) + int(_cap("source_tracker", "credibility"))) / 3.0
        execution = 0.5 * min(1.0, merged / 3.0) + 0.5 * min(1.0, done_tasks / 3.0)
        # v14 P5: the customers milestone needs REAL market validation — PAYING conversions +
        # willingness-to-pay from post-release trials (market_summary), NOT raw ticket count, so
        # a flood of churn/complaint tickets from a broken product can't game it. With no released
        # product there are no trials, so it stays 0 (tr3 delays) instead of a false grant_full.
        from environments.org_env.backend.market import market_summary
        ms = market_summary(self)
        customers = ms["customers_score"]
        status = {
            "milestone_demo": demo >= 0.6, "milestone_quality": quality >= 0.6,
            "milestone_execution": execution >= 0.5, "milestone_customers": customers >= 0.6,
            "scores": {"demo": round(demo, 2), "quality": round(quality, 2),
                       "execution": round(execution, 2), "customers": round(customers, 2)},
            "market": ms,
        }
        met = {k for k, v in status.items() if k.startswith("milestone_") and v is True}
        return met, status, {"milestone_demo": demo, "milestone_quality": quality,
                             "milestone_execution": execution, "milestone_customers": customers}

    def process_funding_checkpoint(self, tick: int) -> list:
        """spec #5: at each tranche checkpoint, evaluate internal milestones and record an
        explicit funding decision (grant_full / grant_partial / delay), updating treasury +
        runway. Unconditional tranches grant in full; conditional ones depend on milestones."""
        bs = self.budget_system
        met, status, score_by = self._evaluate_milestones()
        bs.funding.milestone_status = status
        arrived = bs.advance_funding(tick, met_conditions=met)
        decided = []
        for tr in arrived:
            self._record_funding(tr, "grant_full", tick, status)
            decided.append(tr.tranche_id)
        # a conditional tranche due now but condition unmet: partial grant if there is real
        # progress, else a recorded delay (decided once per tranche).
        for tr in bs.funding.tranches:
            if tr.scheduled_tick > tick or not tr.condition:
                continue
            if any(h["tranche_id"] == tr.tranche_id for h in bs.funding.funding_history):
                continue
            if tr.status == "delayed":
                sc = score_by.get(tr.condition, 0.0)
                if sc >= 0.4:
                    amt = round(tr.amount * 0.5, 1)
                    bs.budget.cash_balance += amt
                    bs.budget.remaining_budget += amt
                    tr.status = "arrived"
                    tr.actual_arrival_tick = tick
                    tr.delay_reason = "partial_milestone"
                    bs._refresh_pressure()
                    self._record_funding(tr, "grant_partial", tick, status, amount=amt)
                else:
                    self._record_funding(tr, "delay", tick, status)
                decided.append(tr.tranche_id)
        return decided

    def _record_funding(self, tr, decision: str, tick: int, status: dict, amount: float = None) -> None:
        bs = self.budget_system
        entry = {"tranche_id": tr.tranche_id, "decision": decision, "tick": tick,
                 "amount": amount if amount is not None else tr.amount, "condition": tr.condition,
                 "milestone_status": {k: v for k, v in status.items() if k != "scores"},
                 "scores": status.get("scores", {}), "runway_days": bs.budget.runway_days,
                 "treasury": round(bs.budget.cash_balance, 1)}
        bs.funding.funding_history.append(entry)
        self.events.append({"type": "funding_event", "subtype": decision, "tranche_id": tr.tranche_id,
                            "amount": entry["amount"], "condition": tr.condition, "tick": tick})

    def _apply_external_events(self, tick: int) -> None:
        """Controlled environmental interventions (§34): at scheduled ticks the external community
        reacts with public posts + a MarketSignal, and an api_price_shock also raises the company's
        cost_multiplier. Internal agents only feel the pressure if they read_feed / search_posts —
        no omniscient injection. Driven by scenario.params:
          * back-compat single lever: {shock_tick, api_cost_multiplier}
          * general schedule: external_events=[{tick, kind, ...}] (kind in INTERVENTION_TEMPLATES)."""
        st = self.scenario.params.get("shock_tick")
        mult = self.scenario.params.get("api_cost_multiplier")
        if st is not None and mult and tick == int(st):
            self.inject_external_event("api_price_shock", tick=tick, multiplier=float(mult))
        for ev in (self.scenario.params.get("external_events") or []):
            if int(ev.get("tick", -1)) == tick:
                self.inject_external_event(
                    ev.get("kind", ""), tick=tick,
                    **{k: v for k, v in ev.items() if k not in ("tick", "kind")})

    def inject_external_event(self, kind: str, *, tick: int = None, multiplier: float = None,
                              **params) -> dict:
        """Inject a controlled external intervention NOW: the community publishes pressure posts +
        a MarketSignal (perceivable via read_feed/search_posts); an api_price_shock also raises the
        cost_multiplier (-> budget pressure). Returns a summary. Idempotent on cost_multiplier."""
        from environments.org_env.backend.community.policy import generate_intervention
        tick = self.world_tick if tick is None else int(tick)
        self._extseq = getattr(self, "_extseq", 0) + 1
        posts, signals = generate_intervention(self.community, kind, tick=tick, seq=self._extseq)
        for p in posts:
            self.community.add_post(p)
        for s in signals:
            self.community.add_signal(s)
        effects: dict = {}
        if kind == "api_price_shock":
            m = float(multiplier if multiplier is not None else params.get("multiplier", 3.0))
            if self.budget_system.budget.cost_multiplier < m:
                self.budget_system.apply_api_price_shock(m)
                effects["multiplier"] = m
        summary = {"kind": kind, "tick": tick,
                   "post_ids": [p.post_id for p in posts],
                   "signal_ids": [s.signal_id for s in signals], **effects}
        self.events.append({"type": "external_signal_event", "subtype": kind, **summary})
        return summary

    def _log_events(self, result, appraised) -> None:
        # record the touched product artifact so the attractor guard can cool down
        # per (agent, action, OBJECT) instead of per (agent, action) (v4 §2 fix).
        target = (getattr(result, "state_delta", {}) or {}).get("target_artifact")
        if not target:
            target = next((o for o in (result.modified_objects + result.created_objects)
                           if str(o).startswith("art_")), None)
        self.action_log.append({"agent_id": result.agent_id, "action_type": result.action_type,
                                "tick": self.world_tick, "success": result.success, "target": target})
        # v8g P1: every action draws TOKENS from the treasury (by category + an LLM-call
        # surcharge when the LLM drove the decision) — unifying action/tool/LLM/meeting/eval
        # cost into one currency that feeds daily-burn -> runway -> runway_pressure.
        bs = getattr(self, "budget_system", None)
        if bs is not None:
            try:
                from environments.org_env.backend.actions import action_category
                cat = action_category(result.action_type) or "other"
            except Exception:
                cat = "other"
            llm = bool(self.llm_client is not None and getattr(self, "llm_decides_actions", False))
            bs.charge_tokens(agent_id=result.agent_id, action_type=result.action_type,
                             tick=self.world_tick, category=cat, llm=llm)
        for ev in result.events:
            self.events.append(ev)
        for ae in appraised:
            self.appraised_log.append({"type": ae.event_type, "agent_id": ae.agent_id,
                                       "tick": ae.tick, "salience": ae.salience})

    def _update_memory(self, agent_id: str, appraised) -> None:
        mem = self.memory.setdefault(agent_id, [])
        for ae in appraised:
            if ae.salience >= 0.4:   # only salient events enter memory
                mem.append({"type": ae.event_type, "tick": ae.tick, "salience": ae.salience})
        if len(mem) > 100:
            del mem[:-100]

    def _run_protocol_detectors(self, tick: int) -> None:
        from environments.org_env.runtime_adapter.event_graph import OrgEventGraph  # noqa
        from environments.org_env.backend.protocol import run_all_detectors
        # Same-ledger diagnostic only. Strong evidence is gated separately on a typed
        # governed-object transition plus an evaluator-owned outcome attestation.
        self.protocol_registry.attribute_impact(tick)
        results = run_all_detectors(self, self.protocol_registry)
        from environments.org_env.experiments.ablations import (
            INSTITUTIONALIZATION,
            mechanism_disabled,
        )
        if (
            self.institutionalization_enabled
            and not mechanism_disabled(self, INSTITUTIONALIZATION)
        ):
            self.protocol_registry.tick_adoptions(tick)   # deferred, latency-gated adoption
        self.detector_summary = {r.protocol_type: {"detected": r.detected, "level": r.emergence_level,
                                                    "evidence": r.evidence} for r in results}

    def run(self, n_ticks: int) -> dict:
        """Run ``n_ticks`` hours and return a summary (for smoke / pilot runs)."""
        for _ in range(n_ticks):
            self.step()
        return self.summary()

    def summary(self) -> dict:
        from collections import Counter
        action_dist = Counter(a["action_type"] for a in self.action_log)
        event_dist = Counter(a["type"] for a in self.appraised_log)
        return {
            "ticks": self.world_tick, "days": self.time.clock.day_index,
            "experiment_condition": self.experiment_condition,
            "action_selection_mode": self.action_selection_mode,
            "baseline_epoch": self.baseline_epoch,
            "baseline_reset_ticks": list(self.baseline_reset_ticks),
            "agents": len(self.agents), "actions": len(self.action_log),
            "action_distribution": dict(action_dist), "event_distribution": dict(event_dist),
            "work_sessions": len(self.time.work_sessions),
            "overtime_logs": len(self.time.overtime_logs),
            "weekend_logs": len(self.time.weekend_logs),
            "recovery_logs": len(self.time.recovery_logs),
            "background_jobs": len(self.time.background_jobs),
            "background_jobs_completed": sum(1 for j in self.time.background_jobs
                                             if j.status == "completed"),
            "object_appraisals": len(self.object_appraisal_log),
            "feedback_decisions": len(self.feedback_decision_log),
            "text_generations": len(self.text_generation_log),
            "commitments": len(self.commitment_registry.commitments),
            "disputes": len(self.commitment_registry.disputes),
            "requests": len(self.commitment_registry.requests),
            "messages": len(self.comm.messages), "meetings": len(self.meeting_system.meetings),
            "prs": len(self.repo_system.repo.pull_requests),
            "sandbox_jobs": len(self.sandbox_system.jobs),
            "results": len(self.sandbox_system.results),
            "searches": len(self.search_system.logs),
            "protocols": len(self.protocol_registry.protocols),
            "cost_events": len(self.budget_system.cost_events),
            "funding_arrived": [t.tranche_id for t in self.budget_system.funding.tranches
                                if t.status == "arrived"],
            "payroll_status": self.budget_system.payroll.payroll_status,
            "owned_tasks": sum(1 for t in self.tasks.values() if t.owner_id),
            "event_graph": self.event_graph.summary() if self.event_graph else {},
            "detectors": self.detector_summary,
        }


__all__ = ["OrgWorld", "SEED_TEAM"]
