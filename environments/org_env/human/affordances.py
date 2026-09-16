"""What a seat may do, and to what.

Two things live here.

A declarative table of which actions apply to which kind of object, which is
what the UI's context menus are built from. It deliberately does NOT come from
``OrgActionMapper.to_core_candidates``: that pool has already been through
``AttractorGuard``, so it is what the policy thinks is *worth* doing right now,
not what the member is *allowed* to do. A human offered only that would silently
lose legal moves — replying twice in a row, re-reviewing during a cooldown —
and would be playing a smaller game than the agent it replaced.

And the role gates. The execution handlers check almost nothing: ``merge_pr``,
``review_pr``, ``approve_release_candidate`` and ``publish_product_release``
never look at who is calling. For autonomous agents that is harmless, because
the gates live upstream in candidate generation and nothing else constructs
actions. A human seat is a second construction site, so those same gates are
restated here. They mirror ``runtime_adapter/execution.py`` — when a gate moves
there it has to move here, and ``test_human_gateway.py`` pins the pairs.

Governance gates are not affected by the ``role_mandates_enabled`` ablation:
that switch removes the per-role *behavioural prior*, while role labels keep
driving governance (see where it is set in ``world.py``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from environments.org_env.backend.actions.registry import (
    CAT_ARTIFACT,
    CAT_BRIDGE,
    CAT_COMM,
    CAT_DOC,
    CAT_GOVERNANCE,
    CAT_MEETING,
    CAT_PAYROLL,
    CAT_PROTOCOL,
    CAT_RELEASE,
    CAT_REPO,
    CAT_SANDBOX,
    CAT_SEARCH,
    CAT_TIME,
    CAT_WORK,
    ORG_ACTION_CATEGORIES,
)


@dataclass(frozen=True)
class ActionSpec:
    """One offer: an action, what it needs, and how it reads to a human."""
    action_type: str
    label: str
    required: Tuple[str, ...] = ()
    optional: Tuple[str, ...] = ()
    #: Parameter that receives the object the menu was opened on.
    target_param: str = ""
    #: Actions that change what the organization sees or ships. A working agent
    #: may prepare these but a human has to confirm them (HCI V0 §7).
    confirm: bool = False

    def schema(self) -> Dict[str, Any]:
        return {"action_type": self.action_type, "label": self.label,
                "required": list(self.required), "optional": list(self.optional),
                "target_param": self.target_param, "confirm": self.confirm}


def _s(action_type, label, required=(), optional=(), target_param="", confirm=False):
    return ActionSpec(action_type, label, tuple(required), tuple(optional),
                      target_param, confirm)


# --------------------------------------------------------------------------- #
# Object context menus (HCI V0 §4.2)
# --------------------------------------------------------------------------- #
OBJECT_ACTIONS: Dict[str, Tuple[ActionSpec, ...]] = {
    "task": (
        _s("pick_task", "Claim this task", ("task_id",), target_param="task_id"),
        _s("assign_task_owner", "Assign an owner", ("task_id", "owner_id"),
           target_param="task_id"),
        _s("handoff_task", "Hand off", ("task_id", "to_agent"), target_param="task_id"),
        _s("update_task_status", "Change status", ("task_id", "status"),
           target_param="task_id"),
        _s("work_on_task", "Log effort", ("task_id",), target_param="task_id"),
        _s("open_issue", "Raise a blocker", ("title",), ("task_id", "severity"),
           target_param="task_id"),
        _s("link_doc_to_task", "Link a document", ("task_id", "doc_id"),
           target_param="task_id"),
    ),
    "pull_request": (
        _s("review_pr", "Review", ("pr_id",), ("comment",), target_param="pr_id"),
        _s("formal_pr_review", "Write a review verdict", ("pr_id",),
           ("comment", "channel_id"),
           target_param="pr_id"),
        _s("approve_pr", "Approve", ("pr_id",), target_param="pr_id", confirm=True),
        _s("request_changes", "Request changes", ("pr_id",), ("comment",),
           target_param="pr_id", confirm=True),
        _s("merge_pr", "Merge", ("pr_id",), target_param="pr_id", confirm=True),
        _s("run_ci", "Run CI", ("pr_id",), target_param="pr_id"),
    ),
    "branch": (
        _s("commit_patch", "Commit staged work", (), ("branch_id", "patch_id"),
           target_param="branch_id"),
        _s("open_pr", "Open a pull request", (),
           ("branch_id", "title", "source_branch", "reviewers", "linked_task_ids"),
           target_param="branch_id", confirm=True),
        _s("run_ci", "Run CI", (), ("branch_id",), target_param="branch_id"),
    ),
    "document": (
        _s("edit_doc", "Edit", ("doc_id",), ("edit_goal",), target_param="doc_id"),
        _s("comment_doc", "Comment", ("doc_id",), ("comment",), target_param="doc_id"),
        _s("review_doc", "Review", ("doc_id",), target_param="doc_id"),
        _s("approve_doc", "Approve", ("doc_id",), target_param="doc_id", confirm=True),
        _s("request_doc_changes", "Request changes", ("doc_id",), ("comment",),
           target_param="doc_id"),
        _s("share_doc", "Share to a channel", ("doc_id",), ("channel_id",),
           target_param="doc_id", confirm=True),
        _s("publish_doc", "Publish", ("doc_id",), target_param="doc_id", confirm=True),
        _s("archive_doc", "Archive", ("doc_id",), target_param="doc_id"),
    ),
    "experiment": (
        _s("run_experiment", "Run", ("experiment_id",), target_param="experiment_id"),
        _s("save_result", "Save the result", ("experiment_id",), ("result_id",),
           target_param="experiment_id"),
        _s("share_experiment_result", "Share the result", ("result_id",), ("channel_id",),
           target_param="experiment_id", confirm=True),
        _s("export_result_to_tracker", "Log to the tracker", ("result_id",),
           target_param="experiment_id"),
    ),
    "result": (
        _s("challenge_result", "Question this result", ("result_id",),
           ("comment", "channel_id", "object_id", "target_id"),
           target_param="result_id"),
        _s("request_reproduction", "Ask for a reproduction", ("result_id",),
           ("channel_id", "object_id"),
           target_param="result_id"),
        _s("share_json_result", "Share", ("result_id",), ("channel_id",),
           target_param="result_id", confirm=True),
    ),
    "meeting": (
        _s("attend_meeting", "Attend", ("meeting_id",), target_param="meeting_id",
           confirm=True),
        _s("skip_meeting", "Skip", ("meeting_id",), target_param="meeting_id",
           confirm=True),
        _s("record_meeting_notes", "Record notes", ("meeting_id",), ("notes",),
           target_param="meeting_id"),
        _s("propose_decision", "Propose a decision", ("meeting_id",), ("summary",),
           target_param="meeting_id", confirm=True),
        _s("summarize_decision", "Summarize the decision", ("meeting_id",),
           ("decision", "summary"),
           target_param="meeting_id", confirm=True),
        _s("assign_action_item", "Assign a follow-up", ("meeting_id", "description"),
           ("assignee_id", "due_tick"), target_param="meeting_id"),
        _s("share_object_in_meeting", "Share an object", ("meeting_id", "object_id"),
           target_param="meeting_id"),
        _s("close_meeting", "Close", ("meeting_id",), target_param="meeting_id"),
    ),
    "proposal": (
        _s("approve_proposal", "Approve", ("proposal_id",), target_param="proposal_id",
           confirm=True),
        _s("reject_proposal", "Reject", ("proposal_id",), ("reason",),
           target_param="proposal_id", confirm=True),
        _s("request_proposal_changes", "Request changes", ("proposal_id",), ("comment",),
           target_param="proposal_id", confirm=True),
    ),
    "protocol": (
        _s("support_protocol", "Support", ("protocol_id",), target_param="protocol_id",
           confirm=True),
        _s("oppose_protocol", "Oppose", ("protocol_id",), ("reason",),
           target_param="protocol_id", confirm=True),
        _s("follow_protocol", "Apply it to this work", ("protocol_id",), ("object_id",),
           target_param="protocol_id"),
        _s("enforce_protocol", "Hold someone to it", ("protocol_id",),
           ("object_id", "target_agent"), target_param="protocol_id", confirm=True),
        _s("violate_protocol", "Proceed without it", ("protocol_id",), ("reason",),
           target_param="protocol_id", confirm=True),
        _s("amend_protocol", "Propose an amendment", ("protocol_id",),
           ("rationale", "comment", "evidence", "target_protocol_id", "repair_kind",
            "programbench_friction_evidence_digest", "programbench_registry_only_repair"),
           target_param="protocol_id", confirm=True),
    ),
    "release_candidate": (
        _s("run_launch_readiness_check", "Run readiness check", ("candidate_id",),
           target_param="candidate_id"),
        _s("approve_release_candidate", "Approve", ("candidate_id",),
           target_param="candidate_id", confirm=True),
        _s("block_release_candidate", "Block", ("candidate_id",), ("reason",),
           target_param="candidate_id", confirm=True),
        _s("publish_product_release", "Publish", ("candidate_id",), ("summary",),
           target_param="candidate_id", confirm=True),
    ),
    "release": (
        _s("collect_post_launch_feedback", "Collect feedback", (), ("release_id",),
           target_param="release_id"),
    ),
    "message": (
        _s("reply_thread", "Reply", ("message_id",), ("text", "channel_id"),
           target_param="message_id", confirm=True),
        _s("acknowledge_message", "Acknowledge", ("message_id",),
           target_param="message_id"),
        _s("ask_for_clarification", "Ask for clarification", ("message_id",),
           ("text", "channel_id"), target_param="message_id", confirm=True),
    ),
    "issue": (
        _s("close_issue", "Close", ("issue_id",), ("resolution",),
           target_param="issue_id"),
    ),
}

# --------------------------------------------------------------------------- #
# Actions that need no object (the composer / global menu)
# --------------------------------------------------------------------------- #
GLOBAL_ACTIONS: Tuple[ActionSpec, ...] = (
    # communication
    _s("send_message", "Send a message", ("channel_id", "text"),
       ("mentions", "importance", "urgency"),
       confirm=True),
    _s("send_async_update", "Post a status update", ("text",), ("channel_id",),
       confirm=True),
    _s("ask_for_help", "Ask for help", ("text",), ("channel_id", "target_agent"),
       confirm=True),
    _s("ask_for_review", "Ask for a review", ("text",),
       ("channel_id", "object_id", "pr_id", "doc_id", "reviewer", "target_agent"),
       confirm=True),
    _s("ask_for_evidence", "Ask for evidence", ("text",), ("channel_id",), confirm=True),
    _s("promise_work", "Commit to something", ("text",),
       ("channel_id", "due_tick", "due_in", "object_id", "summary", "task_id"),
       confirm=True),
    _s("warn_about_risk", "Warn about a risk", ("text",),
       ("channel_id", "object_id", "pr_id", "result_id", "summary"), confirm=True),
    _s("escalate_incident", "Escalate an incident", ("text",), ("channel_id",),
       confirm=True),
    # work
    _s("create_issue", "File an issue", ("title",),
       ("severity", "topic", "post_id", "signal_id", "source_signal_id", "source")),
    _s("inspect_task_board", "Read the board", ()),
    # repo / product
    _s("edit_repo_file", "Change a repository file", ("file_path",),
       ("artifact_id", "edit_goal", "branch_id"), confirm=True),
    _s("run_public_tests", "Run the public test suite", (), ("probe_mode",)),
    _s("create_release_candidate", "Cut a release candidate", (), ("notes",),
       confirm=True),
    # documents and artifacts
    _s("create_doc", "Write a document", ("title",), ("doc_type", "purpose")),
    _s("write_design_note", "Record a design decision", ("title",), ("body",)),
    _s("create_experiment_tracker", "Create an experiment tracker", (), ("title",)),
    _s("create_review_checklist", "Create a review checklist", (), ("title",)),
    _s("create_claim_evidence_table", "Create a claim-evidence table", (), ("title",)),
    # governance
    _s("propose_protocol", "Propose a protocol", ("title",),
       ("problem_evidence", "required_steps", "enforcement_rule", "protocol_name",
        "protocol_type", "related_object_ids", "related_objects", "required_fields",
        "rule_summary", "scope", "source_problem", "trigger_condition"), confirm=True),
    # meetings
    _s("schedule_meeting", "Schedule a meeting", ("title",),
       ("meeting_type", "participants"), confirm=True),
    # search
    _s("internal_search", "Search internal history", ("query",)),
    _s("repo_search", "Search the repository", ("query",)),
    _s("read_feed", "Read the external feed", ()),
    _s("monitor_customer_feedback", "Read user reports", ()),
    # time
    _s("set_availability_status", "Set availability", ("status",)),
)


# --------------------------------------------------------------------------- #
# Complete organization action surface
# --------------------------------------------------------------------------- #
#
# ``OrgActionMapper.available_actions`` exposes *every* verb in
# ``ORG_ACTION_CATEGORIES`` to an ordinary organization member.  The per-tick
# mapper is deliberately a smaller, context-sensitive shortlist; it is a
# choice policy, not a second definition of what an actor is allowed to do.
# Treating that shortlist as the human contract used to make a human (and the
# P3 secretary that represents one) silently less capable than another agent.
#
# Keep the original verbs below.  In particular, do not turn ``edit_file``
# into ``edit_repo_file`` or ``run_script`` into ``run_experiment``: those are
# related runtime operations with distinct event/provenance semantics.  The
# category-shaped optional fields are intentionally a *bounded* vocabulary,
# rather than an unvalidated free-form parameter bag.  They give the model the
# names it needs to ground a visible target; gateway.py still checks every
# supplied object/channel/member against the seat's current view and the shared
# execution adapter remains the sole executor.
_CATEGORY_OPTIONAL_PARAMS: Dict[str, Tuple[str, ...]] = {
    CAT_WORK: (
        "task_id", "issue_id", "result_id", "owner_id", "assignee_id", "to_agent",
        "status", "title", "text", "comment", "resolution", "severity", "topic",
    ),
    CAT_COMM: (
        "message_id", "channel_id", "object_id", "result_id", "doc_id", "post_id",
        "file_id", "target_agent", "to_agent", "text", "comment", "importance", "urgency",
    ),
    CAT_MEETING: (
        "meeting_id", "issue_id", "object_id", "report_id", "title", "summary",
        "notes", "participants", "assignee_id", "description",
    ),
    CAT_REPO: (
        "branch_id", "task_id", "pr_id", "patch_id", "file_path", "message", "title",
        "linked_task", "comment", "resolution",
    ),
    CAT_SANDBOX: (
        "experiment_id", "result_id", "artifact_id", "script", "query", "file_path",
        "dataset_id", "package", "command", "channel_id", "object_id", "title",
    ),
    CAT_SEARCH: (
        "query", "result_id", "search_result_id", "post_id", "url", "title",
    ),
    CAT_DOC: (
        "doc_id", "artifact_id", "task_id", "experiment_id", "title", "body", "text",
        "comment", "edit_goal", "purpose", "doc_type",
    ),
    CAT_ARTIFACT: (
        "artifact_id", "doc_id", "experiment_id", "result_id", "title", "body", "text",
        "object_id", "tool_id", "checklist_id", "workflow_id",
    ),
    CAT_PROTOCOL: (
        "protocol_id", "proposal_id", "object_id", "target_agent", "title", "rationale",
        "reason", "comment", "problem_evidence", "required_steps", "enforcement_rule",
    ),
    CAT_TIME: (
        "task_id", "to_agent", "status", "text", "reason",
    ),
    CAT_PAYROLL: (
        "candidate_id", "member_id", "target_agent", "channel_id", "title", "text",
        "reason", "role", "amount",
    ),
    CAT_BRIDGE: (
        "artifact_id", "post_id", "channel_id", "external_id", "query", "url", "text",
        "object_id", "summary", "title", "target_agent",
    ),
    CAT_GOVERNANCE: (
        "proposal_id", "comment", "reason",
    ),
    CAT_RELEASE: (
        "candidate_id", "release_id", "notes", "reason", "comment", "summary",
    ),
}


def _display_action_label(action_type: str) -> str:
    """A compact, human-readable label while the registry remains canonical."""
    return action_type.replace("_", " ").capitalize()


def _registry_surface_specs() -> Tuple[ActionSpec, ...]:
    """Expose each registry verb not already represented by a P1/P2 schema.

    This is intentionally generated from the canonical action registry.  A new
    ordinary-agent verb cannot be registered without automatically entering the
    P1/P2/P3 contract and the parity test below.  The current static table is
    retained for object-menu placement and for the tighter required fields of
    actions whose UI has a dedicated form.
    """
    already_named = {spec.action_type for spec in GLOBAL_ACTIONS}
    for group in OBJECT_ACTIONS.values():
        already_named.update(spec.action_type for spec in group)
    return tuple(
        _s(action_type, _display_action_label(action_type), (),
           _CATEGORY_OPTIONAL_PARAMS.get(category, ()), confirm=True)
        for action_type, category in ORG_ACTION_CATEGORIES.items()
        if action_type not in already_named
    )


# These schemas are global because an agent may choose any registry action
# through ``available_actions``; object menus remain intentionally concise.
REGISTRY_SURFACE_ACTIONS: Tuple[ActionSpec, ...] = _registry_surface_specs()


def human_action_executable(action_type: str) -> bool:
    """Whether a human-confirmed action has action-specific runtime semantics."""
    from environments.org_env.runtime_adapter.execution import OrgExecutionAdapter

    return hasattr(OrgExecutionAdapter, f"_h_{action_type}")


def registry_action_parity_matrix() -> List[Dict[str, Any]]:
    """Return one auditable P1/P2/P3 route row for every registered action.

    ``runtime_route`` is resolved lazily so affordances stays a light declarative
    module.  A purpose-built handler is evidence of a distinct runtime action;
    ``generic_registry`` records a structural vocabulary entry only.  Human
    modes must not report that generic audit event as the requested effect.
    """
    specs_by_type: Dict[str, List[ActionSpec]] = {}
    for spec in all_offered_action_specs():
        specs_by_type.setdefault(spec.action_type, []).append(spec)
    return [
        {
            "action_type": action_type,
            "category": category,
            "classification": "exposed_shared_pipeline",
            "schemas": [spec.schema() for spec in specs_by_type.get(action_type, ())],
            "runtime_route": ("handler" if human_action_executable(action_type)
                              else "generic_registry"),
            "authority": (
                "model draft -> P1/P2 confirmation -> seat-visible gateway -> "
                "shared runtime adapter"
            ),
        }
        for action_type, category in ORG_ACTION_CATEGORIES.items()
    ]


# --------------------------------------------------------------------------- #
# Role gates — mirroring runtime_adapter/execution.py candidate generation
# --------------------------------------------------------------------------- #
#: action_type -> roles allowed to take it. Absent means every role may.
ROLE_GATES: Dict[str, Tuple[str, ...]] = {
    # execution.py: only institution-minded roles may propose or amend a protocol
    "propose_protocol": ("founder", "cofounder", "reliability", "editorial"),
    "amend_protocol": ("founder", "cofounder", "reliability", "editorial"),
    # execution.py _release_driven
    "create_release_candidate": ("founder", "cofounder"),
    "approve_release_candidate": ("founder", "cofounder", "reliability"),
    "publish_product_release": ("founder", "cofounder"),
    "collect_post_launch_feedback": ("community", "external_voice", "cofounder"),
}


def _merge_pr_allowed(world, agent, params) -> Optional[str]:
    """execution.py: the author, or a lead, may merge."""
    if agent.role in ("cofounder", "founder", "reliability"):
        return None
    pr = world.repo_system.repo.pull_requests.get(params.get("pr_id"))
    if pr is not None and pr.author_id == agent.id:
        return None
    return "merge_requires_author_or_lead"


def _proposal_verdict_allowed(world, agent, params) -> Optional[str]:
    """execution.py ``_h_approve_proposal``: only a designated approver."""
    pm = getattr(world, "proposal_manager", None)
    proposals = (getattr(pm, "proposals", {}) or {}) if pm else {}
    proposal = proposals.get(params.get("proposal_id"))
    if proposal is None:
        return "unknown_proposal"
    if agent.id not in (getattr(proposal, "approval_required_from", None) or []):
        return "not_a_designated_approver"
    return None


def _rc_approve_allowed(world, agent, params) -> Optional[str]:
    """A member approves a release candidate at most once."""
    from environments.org_env.runtime_adapter.execution import _open_rc

    rc = _open_rc(world, params.get("candidate_id"))
    if rc is not None and agent.id in getattr(rc, "approvals", ()):
        return "already_approved"
    return None


def _meeting_rsvp_allowed(world, agent, params, *, attending: bool) -> Optional[str]:
    """Mirror execution-layer RSVP checks for human-seat actions."""
    meeting = getattr(getattr(world, "meeting_system", None), "meetings", {}).get(
        params.get("meeting_id"))
    if meeting is None:
        return "unknown_meeting"
    if agent.id not in (getattr(meeting, "participants", None) or []):
        return "not_a_meeting_participant"
    status = getattr(getattr(meeting, "status", None), "value", getattr(meeting, "status", ""))
    if status != "scheduled":
        return "meeting_not_scheduled"
    if agent.id in (getattr(meeting, "attendees", None) or []):
        return "already_attending"
    if agent.id in (getattr(meeting, "skipped_by", None) or []):
        return "already_skipped"
    availability = getattr(getattr(world, "time", None), "availability", {}).get(agent.id)
    if attending and availability is not None and getattr(availability, "current_availability_status", "") in {
            "offline", "asleep", "forced_rest"}:
        return "unavailable_for_meeting"
    return None


def _attend_meeting_allowed(world, agent, params) -> Optional[str]:
    return _meeting_rsvp_allowed(world, agent, params, attending=True)


def _skip_meeting_allowed(world, agent, params) -> Optional[str]:
    return _meeting_rsvp_allowed(world, agent, params, attending=False)


#: Gates that need more than the role label.
CONTEXT_GATES: Dict[str, Callable[[Any, Any, dict], Optional[str]]] = {
    "merge_pr": _merge_pr_allowed,
    "approve_proposal": _proposal_verdict_allowed,
    "reject_proposal": _proposal_verdict_allowed,
    "request_proposal_changes": _proposal_verdict_allowed,
    "approve_release_candidate": _rc_approve_allowed,
    "attend_meeting": _attend_meeting_allowed,
    "skip_meeting": _skip_meeting_allowed,
}


def role_denial(world: Any, agent: Any, action_type: str, params: dict) -> Optional[str]:
    """Why this member may not take this action, or None if they may."""
    allowed = ROLE_GATES.get(action_type)
    if allowed is not None and agent.role not in allowed:
        return f"role_not_permitted:{agent.role}"
    gate = CONTEXT_GATES.get(action_type)
    if gate is not None:
        return gate(world, agent, params or {})
    return None


def specs_for_object(object_kind: str) -> Tuple[ActionSpec, ...]:
    return OBJECT_ACTIONS.get(object_kind, ())


def all_offered_action_specs() -> List[ActionSpec]:
    """Every P1/P2 action schema, including target-specific overloads.

    Most action types have one schema. ``run_ci`` intentionally has two: a PR
    action requires ``pr_id`` while the branch menu accepts ``branch_id``. A
    type-only set silently erases that distinction, so natural-language
    compilers and parameter validators must use this schema-level view.
    """
    specs = [*GLOBAL_ACTIONS, *REGISTRY_SURFACE_ACTIONS]
    for group in OBJECT_ACTIONS.values():
        specs.extend(group)
    return specs


def find_spec(action_type: str,
              params: Optional[Dict[str, Any]] = None) -> Optional[ActionSpec]:
    candidates = [spec for spec in all_offered_action_specs()
                  if spec.action_type == action_type]
    if not candidates or params is None or len(candidates) == 1:
        return candidates[0] if candidates else None

    supplied = {key for key, value in params.items()
                if value not in (None, "", [])}
    compatible = [spec for spec in candidates
                  if supplied <= (set(spec.required) | set(spec.optional))]
    if not compatible:
        return candidates[0]

    # Prefer a schema whose requirements are already satisfied, then the one
    # most specifically named by the supplied fields. This selects PR run_ci
    # for {pr_id}, branch run_ci for {branch_id}, and the valid target-free
    # branch form for {}.
    compatible.sort(
        key=lambda spec: (
            set(spec.required) <= supplied,
            bool(spec.target_param and spec.target_param in supplied),
            len(supplied & (set(spec.required) | set(spec.optional))),
            -len(spec.required),
        ),
        reverse=True,
    )
    return compatible[0]


def all_offered_action_types() -> List[str]:
    types = {s.action_type for s in (*GLOBAL_ACTIONS, *REGISTRY_SURFACE_ACTIONS)}
    for specs in OBJECT_ACTIONS.values():
        types.update(s.action_type for s in specs)
    return sorted(types)


__all__ = [
    "ActionSpec", "OBJECT_ACTIONS", "GLOBAL_ACTIONS", "REGISTRY_SURFACE_ACTIONS",
    "ROLE_GATES", "CONTEXT_GATES",
    "role_denial", "specs_for_object", "find_spec", "all_offered_action_specs",
    "all_offered_action_types", "registry_action_parity_matrix",
    "human_action_executable",
]
