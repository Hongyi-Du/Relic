"""OrgEpisode data + episode ontology (triggers / compatibility / object typing).

An ``OrgEpisode`` is a causal organizational process — a trigger event, the agents
who notice/react, the messages/meetings/disputes it spawns, the artifacts/protocols/
product-changes it produces, and an outcome. ``EpEvent`` is a normalized view of a
single world event (built from an ``ExecutionResult`` or a world-level event dict)
that the manager observes.

All rules here are GENERIC (keyed by event type / action type / role / object-id
prefix) — never by a specific agent id.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# -- episode types (spec §3.2 + v11 coding layer) -------------------------- #
EPISODE_TYPES = (
    "feedback_ingestion_episode",
    "claim_dispute_episode",
    "experiment_episode",
    "protocol_formation_episode",
    "launch_crunch_episode",
    "customer_triage_episode",
    # v11 Coding Capability Layer §8 (Priority 2): a CI/smoke failure forms a
    # debugging episode (failure log -> suspected module -> patch -> test -> resolution)
    # so a later similar failure can recall "how did we fix this gate last time?".
    "debugging_episode",
)

# -- normalized event -------------------------------------------------------- #
@dataclass
class EpEvent:
    """A normalized world event the EpisodeManager observes."""
    event_id: str
    tick: int
    family: str                      # §11 event family (e.g. "experiment_event")
    subtype: str = ""
    action_type: str = ""            # underlying action (e.g. "run_cheap_pilot")
    actor_id: Optional[str] = None
    target_id: Optional[str] = None  # an agent the event is directed at
    object_ids: List[str] = field(default_factory=list)
    primary_object_id: Optional[str] = None
    channel_id: Optional[str] = None  # thread/channel context (a linkage signal)
    success: bool = True
    raw: Dict[str, Any] = field(default_factory=dict)

    def compact(self) -> Dict[str, Any]:
        return {"event_id": self.event_id, "tick": self.tick, "family": self.family,
                "subtype": self.subtype, "action_type": self.action_type,
                "actor_id": self.actor_id, "target_id": self.target_id,
                "channel_id": self.channel_id, "object_ids": list(self.object_ids)}


# -- object-id typing (prefix-based; speech objects BEFORE commit_) ---------- #
_OBJ_KINDS: List[Tuple[Tuple[str, ...], str]] = [
    (("dispute_", "commitment_", "request_"), "speech_object"),
    (("msg_", "att_"), "message"),
    (("task_", "issue_"), "task"),
    (("mnote_", "doc_"), "doc"),
    (("meeting_",), "meeting"),
    (("pr_", "branch_", "commit_"), "repo"),
    (("result_", "job_", "run_"), "sandbox_result"),
    (("proto_",), "protocol"),
    (("post_", "sig_", "ext_"), "external_signal"),
    (("ledger", "artifact_"), "artifact"),
]

# object kind -> the OrgEpisode.linked_* field it populates.
_KIND_TO_LINK = {
    "message": "linked_message_ids", "task": "linked_task_ids", "doc": "linked_doc_ids",
    "meeting": "linked_meeting_ids", "repo": "linked_repo_ids",
    "sandbox_result": "linked_sandbox_result_ids", "protocol": "linked_protocol_ids",
    "external_signal": "linked_external_signal_ids", "artifact": "linked_artifact_ids",
}


def classify_object(object_id: str) -> str:
    oid = str(object_id or "")
    for prefixes, kind in _OBJ_KINDS:
        if oid.startswith(prefixes):
            return kind
    return "other"


# -- triggers (deliberate, low-frequency signals only) ---------------------- #
# action_type -> episode_type
_TRIGGER_ACTIONS = {
    "challenge_result": "claim_dispute_episode",
    "ask_for_evidence": "claim_dispute_episode",
    "propose_protocol": "protocol_formation_episode",
    "enforce_protocol": "protocol_formation_episode",
    "run_cheap_pilot": "experiment_episode",
    "run_experiment": "experiment_episode",
    "run_paper_baseline": "experiment_episode",
    "push_team": "launch_crunch_episode",
    "work_overtime": "launch_crunch_episode",
    "weekend_work": "launch_crunch_episode",
    "create_customer_triage_sheet": "customer_triage_episode",
    "prepare_customer_feedback_summary": "customer_triage_episode",
}

# C-fix: sharing/replying to external chatter is only "feedback ingestion" when the signal is
# ACTIONABLE. Opening a feedback_ingestion_episode for EVERY share_external_post /
# respond_to_public_comment (incl. surfacing low-signal AMBIENT forum chatter) is what produced
# ~70% zero-action abandoned episodes. So: an actionable share/reply OPENS an ingestion loop; an
# ambient one may only ATTACH to an already-open loop, never OPEN a fresh episode.
_FEEDBACK_SHARE_ACTIONS = {"share_external_post", "respond_to_public_comment"}


def _is_ambient_signal(ev: "EpEvent") -> bool:
    """True if the shared/replied object is organic forum chatter (post_ambient_*), not a
    product-actionable signal (a market trial, bug report, or issue-linked post)."""
    raw = ev.raw or {}
    pid = str(raw.get("post_id") or ev.primary_object_id or "")
    return pid.startswith("post_ambient")


def attach_only_episode_type(ev: "EpEvent") -> Optional[str]:
    """Episode type this event may ATTACH to but must never OPEN — an ambient share/reply can
    join an open feedback loop, but cannot spawn a new (likely zero-action) episode."""
    if ev.action_type in _FEEDBACK_SHARE_ACTIONS and _is_ambient_signal(ev):
        return "feedback_ingestion_episode"
    return None

# "workflow" episodes span many agents/steps (an external→internal or launch
# process), so they attach type-compatible events more readily (across agents);
# "focused" episodes (experiment / dispute / protocol) stay tighter.
WORKFLOW_TYPES = {"feedback_ingestion_episode", "customer_triage_episode", "launch_crunch_episode"}

# when a TRIGGER fires, it may ABSORB into an already-open episode of a related
# type instead of opening a duplicate (e.g. a triage action joins the open feedback
# episode). A dispute, however, never absorbs into an experiment — it stays a
# distinct, cross-linked episode (Chain B wants both).
ABSORB_INTO = {
    "feedback_ingestion_episode": {"feedback_ingestion_episode", "customer_triage_episode"},
    "customer_triage_episode": {"customer_triage_episode", "feedback_ingestion_episode"},
    "launch_crunch_episode": {"launch_crunch_episode"},
    "experiment_episode": {"experiment_episode"},
    "claim_dispute_episode": {"claim_dispute_episode"},
    "protocol_formation_episode": {"protocol_formation_episode"},
    # repeated CI/smoke failures fold into one open debugging episode (not N dupes)
    "debugging_episode": {"debugging_episode"},
}
# event family -> episode_type (used when no triggering action matched)
_TRIGGER_FAMILIES = {
    "claim_dispute_event": "claim_dispute_episode",
    "protocol_proposal_event": "protocol_formation_episode",
    "protocol_violation_event": "protocol_formation_episode",
    "protocol_enforcement_event": "protocol_formation_episode",
    "experiment_event": "experiment_episode",
    "overtime_event": "launch_crunch_episode",
    "weekend_work_event": "launch_crunch_episode",
}
# priority when several could match (most specific first)
_TRIGGER_PRIORITY = (
    "debugging_episode",
    "claim_dispute_episode", "protocol_formation_episode", "experiment_episode",
    "customer_triage_episode", "launch_crunch_episode", "feedback_ingestion_episode",
)


# v11 coding layer: technical-failure signals that open a debugging episode.
_CI_FAIL_STATUS = {"failed", "failing", "error", "red", "broken", "fail"}
_DEBUG_GATE_KEYWORDS = ("smoke", "ci", "eval", "test", "build", "compile")


def is_debug_failure(ev: EpEvent) -> bool:
    """A CI run that failed, or a launch-readiness check blocked on a TECHNICAL gate
    (smoke / ci / eval / test / build) — i.e. a coding blocker, not a docs/evidence one."""
    raw = ev.raw or {}
    if ev.family == "repo_event" and ev.subtype == "ci":
        return str(raw.get("status", "")).lower() in _CI_FAIL_STATUS
    if ev.family == "release_event" and ev.subtype == "readiness_check" \
            and str(raw.get("status", "")).lower() == "blocked":
        return any(any(k in str(b).lower() for k in _DEBUG_GATE_KEYWORDS)
                   for b in (raw.get("blockers") or []))
    # v11 dogfooding: an agent ran the product and hit a crash / broken output
    if ev.family == "product_event" and ev.subtype == "dogfood":
        return str(raw.get("outcome", "")) in ("crash", "broken_output")
    return False


def trigger_episode_type(ev: EpEvent) -> Optional[str]:
    cands = set()
    if ev.action_type in _TRIGGER_ACTIONS:
        cands.add(_TRIGGER_ACTIONS[ev.action_type])
    # C-fix: a share/reply OPENS an ingestion loop only for an ACTIONABLE signal; an ambient one
    # is attach-only (see attach_only_episode_type) so it can't spawn a zero-action episode.
    if ev.action_type in _FEEDBACK_SHARE_ACTIONS and not _is_ambient_signal(ev):
        cands.add("feedback_ingestion_episode")
    if ev.family in _TRIGGER_FAMILIES:
        cands.add(_TRIGGER_FAMILIES[ev.family])
    # external shock is a feedback trigger; ordinary read_feed/monitor is NOT
    if ev.family == "external_signal_event" and ev.subtype == "api_price_shock":
        cands.add("feedback_ingestion_episode")
    # v14 P5: a market trial / post-launch feedback / human evaluation is the external
    # signal that should ingest into a product-change loop (not evaporate).
    # C-fix: only ACTIONABLE inbound feedback opens an ingestion loop. A happy, converted trial
    # needs no episode; an interested/churned trial (a fixable complaint) does.
    if ev.family == "external_signal_event" and ev.subtype in (
            "post_launch_feedback", "human_product_feedback"):
        cands.add("feedback_ingestion_episode")
    if ev.family == "external_signal_event" and ev.subtype == "customer_trial":
        raw = ev.raw or {}
        if str(raw.get("outcome", "")) != "converted" or not raw.get("converted", False):
            cands.add("feedback_ingestion_episode")
    # v11: a CI/smoke technical failure opens (or extends) a debugging episode
    if is_debug_failure(ev):
        cands.add("debugging_episode")
    if not cands:
        return None
    for t in _TRIGGER_PRIORITY:
        if t in cands:
            return t
    return None


# -- attachment compatibility (which event families belong to each episode) -- #
_BASE_COMPAT = {"communication_event", "speech_act_event", "meeting_event",
                "task_progress_event", "file_share_event"}
EPISODE_COMPATIBLE_EVENT_TYPES: Dict[str, set] = {
    "claim_dispute_episode": _BASE_COMPAT | {
        "claim_dispute_event", "requested_action_event", "experiment_event",
        "repo_event", "protocol_proposal_event", "protocol_use_event"},
    "experiment_episode": _BASE_COMPAT | {
        "experiment_event", "claim_dispute_event", "requested_action_event", "repo_event"},
    "protocol_formation_episode": _BASE_COMPAT | {
        "protocol_proposal_event", "protocol_support_event", "protocol_use_event",
        "protocol_violation_event", "protocol_enforcement_event"},
    "launch_crunch_episode": _BASE_COMPAT | {
        "overtime_event", "weekend_work_event", "repo_event", "external_signal_event"},
    # v14 P5: feedback/triage episodes absorb the resulting product change (repo edit /
    # patch / merge / release) so a customer complaint can CLOSE into a fix instead of
    # being abandoned with no product change.
    "customer_triage_episode": _BASE_COMPAT | {
        "external_signal_event", "requested_action_event", "repo_event", "product_event"},
    "feedback_ingestion_episode": _BASE_COMPAT | {
        "external_signal_event", "requested_action_event", "repo_event", "product_event", "release_event"},
    # a debugging episode absorbs the whole fix loop: edits, commits, PRs, CI, the
    # rerun readiness check, and any verification experiment.
    "debugging_episode": _BASE_COMPAT | {
        "repo_event", "release_event", "experiment_event", "product_event",
        "requested_action_event"},
}

# human-readable titles + opening problem templates
EPISODE_TITLE = {
    "feedback_ingestion_episode": "External Feedback Ingestion",
    "claim_dispute_episode": "Claim / Evidence Dispute",
    "experiment_episode": "Experiment Run",
    "protocol_formation_episode": "Protocol Formation",
    "launch_crunch_episode": "Launch Crunch",
    "customer_triage_episode": "Customer Triage",
    "debugging_episode": "Debugging / Blocker Closure",
}


@dataclass
class OrgEpisode:
    episode_id: str
    episode_type: str
    start_tick: int
    end_tick: Optional[int] = None
    status: str = "open"                       # open / resolved / abandoned / dormant

    trigger_event_id: Optional[str] = None
    trigger_object_id: Optional[str] = None

    participants: List[str] = field(default_factory=list)
    primary_agent_id: Optional[str] = None

    linked_event_ids: List[str] = field(default_factory=list)
    linked_message_ids: List[str] = field(default_factory=list)
    linked_meeting_ids: List[str] = field(default_factory=list)
    linked_task_ids: List[str] = field(default_factory=list)
    linked_doc_ids: List[str] = field(default_factory=list)
    linked_repo_ids: List[str] = field(default_factory=list)
    linked_sandbox_result_ids: List[str] = field(default_factory=list)
    linked_protocol_ids: List[str] = field(default_factory=list)
    linked_external_signal_ids: List[str] = field(default_factory=list)
    linked_artifact_ids: List[str] = field(default_factory=list)
    linked_object_ids: List[str] = field(default_factory=list)   # union (for matching)
    linked_channels: List[str] = field(default_factory=list)     # thread/channel context
    related_episode_ids: List[str] = field(default_factory=list)
    # reflection / wish / proposal chain produced from this episode
    linked_reflection_ids: List[str] = field(default_factory=list)
    linked_wish_ids: List[str] = field(default_factory=list)
    linked_proposal_ids: List[str] = field(default_factory=list)

    problem_statement: str = ""
    conflict_summary: Optional[str] = None
    decision_summary: Optional[str] = None
    outcome_summary: Optional[str] = None

    # v11 coding layer: debugging-episode specifics (failure log -> suspected module ->
    # attempted patches -> resolution), preserved for later recall.
    failure_log: str = ""
    suspected_module: str = ""
    failing_gates: List[str] = field(default_factory=list)
    attempted_patches: List[str] = field(default_factory=list)
    resolution: str = ""

    produced_artifacts: List[str] = field(default_factory=list)
    produced_protocols: List[str] = field(default_factory=list)
    produced_tasks: List[str] = field(default_factory=list)
    produced_product_changes: List[str] = field(default_factory=list)

    state_delta: Dict[str, Any] = field(default_factory=dict)
    graph_delta: Dict[str, Any] = field(default_factory=dict)

    timeline: List[Dict[str, Any]] = field(default_factory=list)
    title: str = ""

    created_at_tick: int = 0
    updated_at_tick: int = 0

    # -- helpers ----------------------------------------------------------- #
    def link_object(self, object_id: str) -> None:
        if not object_id:
            return
        if object_id not in self.linked_object_ids:
            self.linked_object_ids.append(object_id)
        kind = classify_object(object_id)
        fld = _KIND_TO_LINK.get(kind)
        if fld is not None:
            lst = getattr(self, fld)
            if object_id not in lst:
                lst.append(object_id)

    def add_participant(self, agent_id: Optional[str]) -> None:
        if agent_id and agent_id not in self.participants:
            self.participants.append(agent_id)


__all__ = [
    "EPISODE_TYPES", "EpEvent", "OrgEpisode", "classify_object", "trigger_episode_type",
    "EPISODE_COMPATIBLE_EVENT_TYPES", "EPISODE_TITLE", "WORKFLOW_TYPES", "ABSORB_INTO",
    "is_debug_failure",
]
