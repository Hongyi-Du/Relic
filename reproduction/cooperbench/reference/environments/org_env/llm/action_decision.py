"""Traditional LLM-direct Action Decision layer (spec §4).

For B0/B1/B2, the LLM chooses AMONG the system-generated feasible candidate
pool, its choice is validated, and only then does the normal execution adapter
mutate the world. Illegal, empty, or failed LLM output is fail-closed for that
action; it never falls through to B3's profile-conditioned WHAT policy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from environments.org_env.llm.client import LLMError, OrgLLMClient
from environments.org_env.llm.prompts import ACTION_SYSTEM
from environments.org_env.llm.prompt_assets import agent_identity_for, render_product_context, system_for
from environments.org_env.llm.render import UNBOUNDED, render_context

# Section titles as render_context capitalizes them. These three carry the
# action space itself, so they are exempt from the descriptive-context budget.
ACTION_MENU_BUDGETS = {
    "Available Actions": UNBOUNDED,
    "Action Descriptions": UNBOUNDED,
    "Candidate Options": UNBOUNDED,
    "Valid Targets": UNBOUNDED,
    # What the work actually is. Open issues carry their reported text here, and
    # a feed post is only listed once the agent has read it — budgeting these
    # down would re-hide the very thing the agent went looking for.
    "Product Context": UNBOUNDED,
    "Cooperbench Visible Briefs": UNBOUNDED,
    "External Posts": UNBOUNDED,
    # A contract read and then budgeted down to its first entries is a contract
    # half-read, which is how a name one suffix short gets written.
    "Knowledge Read": UNBOUNDED,
    # The state of the work. A budget here decides which task the agent is
    # allowed to notice, which is the action space again wearing another name.
    "Task Board": UNBOUNDED,
    "Inbox": UNBOUNDED,
    "Pull Requests": UNBOUNDED,
    "Test Results": UNBOUNDED,
    # Bounded, but not at the 1500-character default: these entries now carry
    # the refusal reasons that are the organization's only account of why it is
    # blocked, and _bounded_lines drops from the end, so a tight budget would
    # spend itself on routine chatter and cut exactly that. _recent_events puts
    # refusals first for the same reason.
    "Recent Events": 3000,
    # The rules in force, with their text. At the 1500-character default a
    # fourth adopted protocol pushes the first one's rule out, and an agent
    # cannot comply with a rule it cannot read.
    "Active Protocols": 2500,
}
from environments.org_env.llm.schemas import ACTION_DECISION_SCHEMA

# Sections build_action_context derives from the world alone, so every agent
# deciding at the same tick receives these bytes identically. Leading with them
# is what makes the provider's prefix cache reachable: measured against the
# bound endpoint, a repeated prefix is served at 8514/8516 cached tokens, while
# the shipped run took 10.4% overall and 1.3% marginal across t120-t144 --
# roughly the system prompt and nothing more, because the user message opened
# with the agent's own identity and diverged at its first character.
#
# Membership is a claim about the source of each section, not a guess. Anything
# built from agent_id, perception or candidates belongs below, and putting it
# here would not corrupt the prompt, only quietly cost the cache.
SHARED_CONTEXT_ORDER = (
    "clock",                # world.time.clock, identical for every agent this tick
    "product_context",      # render_product_context(world)
    "active_protocols",     # world.protocol_registry
    "active_tools",         # _active_tools(world)
    "active_episodes",      # world.episode_manager, open episodes, unfiltered
    "test_results",         # _test_results(world)
    "channels",             # world.comm.channels
)

# This agent's own view, in the order it is useful to read: who it is, what it
# has been doing, what is in front of it, and last the menu it must choose from.
PERSONAL_CONTEXT_ORDER = (
    "agent", "persona_summary", "work_state", "memory",
    "cooperbench_visible_briefs",
    "recent_events", "external_posts", "knowledge_read",
    "task_board", "inbox", "pull_requests", "pending_approvals",
    "search_results", "meetings", "experiments",
    "available_actions", "action_descriptions",
    "candidate_options", "valid_targets",
)

# object-id param keys an action may carry (for valid-target extraction).
# artifact_id is last so the more specific ids keep priority, but it must be here:
# a repo edit names its target that way and nothing else, so without it every
# edit_repo_file option rendered with a null target.
_TARGET_KEYS = ("task_id", "pr_id", "result_id", "post_id", "doc_id", "message_id",
                "protocol_id", "branch_id", "meeting_id", "object_id", "experiment_id",
                "proposal_id", "tool_id", "artifact_id")


@dataclass
class ActionDecision:
    decision_id: str
    agent_id: str
    tick: int
    decision_source: str = "llm"        # llm / llm_error / scripted_test / system
    candidate_action: str = ""
    candidate_speech_act: Optional[str] = None
    target_agent_id: Optional[str] = None
    target_object_id: Optional[str] = None
    channel_id: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    expected_effect: Optional[str] = None
    risk_assessment: Optional[str] = None
    confidence: float = 0.5
    related_episode_ids: List[str] = field(default_factory=list)
    related_wish_ids: List[str] = field(default_factory=list)
    related_proposal_ids: List[str] = field(default_factory=list)
    related_memory_ids: List[str] = field(default_factory=list)
    validation_status: str = "pending"  # pending/accepted/rejected/revised/fallback_used
    rejection_reason: Optional[str] = None
    executed_event_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision_id": self.decision_id, "agent_id": self.agent_id, "tick": self.tick,
            "decision_source": self.decision_source, "candidate_action": self.candidate_action,
            "candidate_speech_act": self.candidate_speech_act, "target_agent_id": self.target_agent_id,
            "target_object_id": self.target_object_id, "channel_id": self.channel_id,
            "params": dict(self.params), "rationale": self.rationale,
            "expected_effect": self.expected_effect, "risk_assessment": self.risk_assessment,
            "confidence": round(float(self.confidence), 3),
            "related_episode_ids": list(self.related_episode_ids),
            "related_wish_ids": list(self.related_wish_ids),
            "validation_status": self.validation_status, "rejection_reason": self.rejection_reason,
            "executed_event_id": self.executed_event_id,
        }


@dataclass
class ValidationResult:
    passed: bool
    reason: str = ""


def _knowledge_read(agent_id: str, world: Any) -> List[Dict[str, Any]]:
    """The text of the knowledge files this agent has gone and read."""
    pw = (getattr(world, "personal", {}) or {}).get(agent_id)
    arts = getattr(world, "product_artifacts", {}) or {}
    from environments.org_env.cooperbench.source_reads import actor_source_read_view
    source_view = actor_source_read_view(world, agent_id)
    profile_state = None
    if "programbench_profile_state" in getattr(world, "__dict__", {}):
        try:
            from environments.org_env.programbench import (
                get_programbench_profile_state,
                programbench_profile_active,
            )

            if programbench_profile_active(world):
                profile_state = get_programbench_profile_state(world)
        except (ImportError, AttributeError, TypeError, ValueError):
            profile_state = None
    required_ids = {
        str(row.get("artifact_id") or "")
        for row in (profile_state or {}).get("required_public_documents") or []
        if isinstance(row, dict)
    }
    out = []
    adapted_chars = 0
    for oid in (getattr(pw, "downloaded_doc_ids", []) or []) if pw else ():
        art = arts.get(oid)
        if source_view is not None and oid not in source_view.catalog:
            continue
        path = str(getattr(art, "linked_file_path", "") or "") if art else ""
        required_programbench_surface = bool(
            profile_state is not None and str(oid) in required_ids
        )
        if (
            not path.endswith(".md")
            and not required_programbench_surface
        ):
            continue
        if profile_state is not None:
            # Preserve ordinary public markdown reading in the adapted run, but
            # never render a sealed namespace even if a malformed checkpoint
            # injects its artifact id into an agent's downloaded list.
            from environments.org_env.programbench import (
                programbench_public_path_is_agent_visible,
            )

            if not programbench_public_path_is_agent_visible(path):
                continue
        body = (source_view.text(art) if source_view is not None else
                (getattr(art, "mainline_content", "") or getattr(art, "content", "") or ""))
        if profile_state is not None:
            # ProgramBench also exposes required non-markdown contract surfaces
            # (notably compile.sh and the public probe schema), but keeps prompt
            # payloads bounded and uses only the frozen seed/mainline view.
            remaining = max(0, 128 * 1024 - adapted_chars)
            if remaining <= 0:
                break
            rendered = body[: min(32 * 1024, remaining)]
            adapted_chars += len(rendered)
            out.append(
                {
                    "file": path,
                    "text": rendered,
                    "truncated": len(rendered) < len(body),
                }
            )
        else:
            out.append({"file": path, "text": body})
    return out


def build_action_context(agent_id: str, world: Any, perception: Any,
                         candidates: List[Any]) -> Dict[str, Any]:
    agent = world.agents.get(agent_id)
    prof = dict(getattr(agent, "profile", {}) or {})
    profile_conditioning = bool(
        getattr(world, "profile_conditioning_enabled", True)
    )
    top_traits = (
        sorted(prof.items(), key=lambda kv: -abs(kv[1] - 0.5))[:5]
        if profile_conditioning
        else []
    )
    skills = dict(getattr(agent, "skills", {}) or {})
    top_skills = sorted(skills.items(), key=lambda kv: -kv[1])[:5]
    ws = agent.work_state.snapshot() if agent and hasattr(agent, "work_state") else {}
    rhythm = bool(getattr(getattr(world, "time", None), "rhythm_enabled", True))
    rm = getattr(world, "reflection_manager", None)
    mem = rm.context_for_decision(agent_id, world) if rm is not None else {}
    open_eps = [{"id": e.episode_id, "type": e.episode_type, "problem": e.problem_statement}
                for e in getattr(world, "episode_manager", None).episodes.values()
                if e.status == "open"][:6] if getattr(world, "episode_manager", None) else []
    # Every constrained candidate is shown. Truncating here silently decided the
    # agent's action space: the list arrives in generation order, task-management
    # actions are generated first, and a head slice of eight cut edit_repo_file
    # and open_pr off the end of every single-agent decision - so the founder
    # re-picked work_on_task 159 times over a 336-tick run because it was the
    # only listed way to make progress. _constrain_candidates is what bounds
    # this set (median 14, observed max 20); a second bound here just hides part
    # of it.
    visible_candidates = _source_visible_candidates(candidates, world, agent_id)
    available = sorted({c.action_type for c in visible_candidates})
    valid_targets = sorted(_valid_targets(visible_candidates, world, actor_id=agent_id))
    channels = sorted(getattr(world.comm, "channels", {}).keys())
    from environments.org_env.backend.actions import action_description
    context: Dict[str, Any] = {
        "agent": {"id": agent_id, "name": getattr(agent, "name", agent_id),
                  "role": getattr(agent, "role", "")},
        "persona_summary": {
            "representation": (
                "profile_conditioned"
                if profile_conditioning
                else "flat_role_and_skills"
            ),
            "top_traits": dict(top_traits),
            "top_skills": dict(top_skills),
        },
        "memory": mem,
        "active_episodes": open_eps,
        "recent_events": _recent_events(perception),
        "product_context": render_product_context(world, actor_id=agent_id),
        "available_actions": available,        # bare action ids (candidate_action must match one)
        "action_descriptions": {a: action_description(a) for a in available},
        "candidate_options": [
            _candidate_option(candidate) for candidate in visible_candidates
        ],
        # Only posts this agent has already read (perception gates on
        # saved_post_ids), so an issue's text is something it went and found
        # rather than something handed to it at t0.
        "external_posts": list(getattr(perception, "visible_external_posts", []) or []),
        # Same rule for the product's own knowledge: read one and its text is
        # here afterwards. Before there was an action for it, the contract naming
        # every symbol the tests import went a whole run unread.
        "knowledge_read": _knowledge_read(agent_id, world),
        # The state of the work itself. The perception layer has built all four
        # of these every tick since it was written, and none of them reached the
        # prompt: the agent chose among menu rows naming task ids and PR ids
        # whose contents, owners, and outcomes it could not see. A pilot B3 run
        # sent 44 messages and read none of them here, and reviewers approved
        # PRs whose test results were not in front of them.
        "task_board": _task_board(agent_id, perception, world),
        "inbox": _inbox(agent_id, perception),
        "pull_requests": _pull_requests(agent_id, perception),
        # The perception adapter fills these every tick for every agent and,
        # until now, nothing read any of them. Weighted by the actions that feed
        # them, that is 131 searches, 55 pilots and 28 meetings by t144 whose
        # output the organization produced and never saw.
        "search_results": _search_results(perception),
        "meetings": _meetings(perception),
        "experiments": _experiments(perception),
        "test_results": _test_results(world, actor_id=agent_id),
        # The clock is not decoration: _constrain_candidates drops the actions a
        # duration spec forbids after hours and at weekends, so the menu shrinks
        # overnight. Read only by that filter until now, which left the agent
        # with fewer options and no way to know it was the hour rather than the
        # work.
        "clock": {
            key: value
            for key, value in (getattr(perception, "clock_state", None) or {}).items()
            if key in ("hour_in_day", "day_index", "day_of_week", "is_weekend",
                       "is_after_hours", "is_late_night", "rhythm_phase", "phase")
        },
        "valid_targets": valid_targets,
        "channels": channels,
        "active_protocols": _active_protocols(world),
        "active_tools": _active_tools(world),
        "pending_approvals": _pending_approvals(agent_id, world),
    }
    from environments.org_env.cooperbench.visibility import is_strict_coop, visible_brief_context
    if is_strict_coop(world):
        context["cooperbench_visible_briefs"] = visible_brief_context(world, agent_id)
        from environments.org_env.cooperbench.work_schedule import (
            compressed_schedule_receipt,
            member_model_call_budget_context,
        )
        schedule = compressed_schedule_receipt(world)
        if schedule is not None:
            context["benchmark_work_window"] = schedule
        budget = member_model_call_budget_context(world, agent_id)
        if budget is not None:
            context["benchmark_model_query_budget"] = budget
    if rhythm:
        # Omitted rather than sent as frozen zeros when the lived-body layer is
        # ablated: a constant block would still read to the model as a signal.
        context["work_state"] = {
            key: ws.get(key)
            for key in (
                "stress",
                "fatigue",
                "burnout_risk",
                "morale",
                "attention_remaining_today",
            )
        }
    return context


def _coerce_confidence(v: Any) -> float:
    """LLMs return confidence as a number OR words ('High'/'medium'/...). Be robust."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return {"high": 0.8, "medium": 0.5, "med": 0.5, "moderate": 0.5,
                "low": 0.3, "very high": 0.9, "very low": 0.2}.get(str(v or "").strip().lower(), 0.5)


def _task_board(agent_id: str, perception: Any, world: Any = None) -> List[Dict[str, Any]]:
    """Open work, mine first — an agent cannot pick up what it cannot read."""
    rows = []
    from environments.org_env.cooperbench.visibility import is_strict_coop, visible_feature_brief
    for task in list(getattr(perception, "visible_tasks", []) or []):
        task_id = str(task.get("task_id") or "")
        description = task.get("description")
        if is_strict_coop(world) and task_id.startswith("task_oss_cooper_feature_"):
            description = visible_feature_brief(world, agent_id, task_id.removeprefix("task_oss_")) or ""
        rows.append(
            {
                "task_id": task.get("task_id"),
                "title": task.get("title"),
                "description": description,
                "status": task.get("status"),
                "owner": task.get("owner"),
                "priority": task.get("priority"),
                "mine": task.get("owner") == agent_id,
                "unowned": not task.get("owner"),
            }
        )
    rows.sort(key=lambda row: (not row["mine"], not row["unowned"],
                               -int(row.get("priority") or 0)))
    return rows


def _inbox(agent_id: str, perception: Any) -> List[Dict[str, Any]]:
    """Messages this agent can see, newest intake and mentions first.

    Without it a request to review, a question, and a handoff are all invisible,
    and the only cooperation a run can show is cooperation nobody asked for.

    `read` is deliberately absent. The inbox is triaged immediately before
    perception is built, so every message is already marked read by the time
    this renders and the flag would say "read" for all of them — including one
    that arrived seconds ago. `new` is the honest version of that question:
    whether the agent took this message in on this pass.
    """
    rows = []
    for message in list(getattr(perception, "visible_messages", []) or []):
        mentions_me = agent_id in (message.get("mentions") or [])
        rows.append(
            {
                "message_id": message.get("message_id"),
                "from": message.get("sender"),
                "channel": message.get("channel"),
                "text": message.get("summary"),
                "urgency": message.get("urgency"),
                "new": bool(message.get("newly_read")),
                "mentions_me": mentions_me,
                "attachments": message.get("attachments") or None,
            }
        )
    rows.sort(key=lambda row: (not row["new"], not row["mentions_me"]))
    return rows


def _pull_requests(agent_id: str, perception: Any) -> List[Dict[str, Any]]:
    """Open PRs this agent authored or is a reviewer on, and where each stands."""
    rows = []
    for pr in list(getattr(perception, "visible_prs", []) or []):
        rows.append(
            {
                "pr_id": pr.get("pr_id"),
                "author": pr.get("author"),
                "status": pr.get("status"),
                "reviewers": pr.get("reviewers"),
                "reviewed": pr.get("reviewed"),
                "source_branch": pr.get("source_branch"),
                # A request cannot merge without a green build, and its status
                # says nothing about that. Left out, a blocked request read as
                # approved-and-about-to-land, and the reason it was refused —
                # which names the module to fix — never reached anyone.
                "ci_passed": pr.get("ci_passed"),
                "blocked_by": pr.get("blocked_by"),
                "requested_changes": pr.get("requested_changes") or None,
                "review_comments": pr.get("review_comments") or None,
                "awaiting_my_review": agent_id in (pr.get("reviewers") or [])
                and not pr.get("reviewed"),
            }
        )
    # Blocked first, then what needs a review: both are asking for an action,
    # while a green request is merely waiting its turn.
    rows.sort(key=lambda row: (not row.get("blocked_by"),
                               not row["awaiting_my_review"]))
    return rows


# What a world event says, minus the parts a decision cannot use. A probe
# receipt carries its stdout, stderr, input and definition; those are megabytes
# across a run and say nothing the reason field does not.
_EVENT_DETAIL_KEYS = (
    "subtype", "action_type", "status", "outcome", "success",
    "reason", "failure_reason", "rejection_reason",
    "object_id", "artifact_id", "pr_id", "task_id", "issue_id",
    "result_id", "protocol_id", "probe_id", "agent_id",
)
_EVENT_REASON_KEYS = ("reason", "failure_reason", "rejection_reason")


def _project_event(event: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(event, dict):
        return None
    out: Dict[str, Any] = {"type": str(event.get("type") or "")}
    for key in _EVENT_DETAIL_KEYS:
        value = event.get(key)
        if value in (None, "", [], {}):
            continue
        out[key] = value if isinstance(value, (int, float, bool)) else str(value)[:160]
    return out


def _recent_events(perception: Any) -> List[Dict[str, Any]]:
    """What just happened, including why anything was refused.

    This was ``[e.get("type") for e in recent_events][-8:]``. Every event type
    in this world is one of a handful of coarse buckets, so the section rendered
    as eight near-identical words -- 'repo_event', 'proposal_event' -- while the
    event itself said which pull request, which proposal, and why the last
    attempt was turned down. Measured over 144 ticks: 1442 events carried a
    subtype and 571 carried a reason, and none of it reached the prompt.

    The organization therefore could not learn from its own refusals. It spent
    55 actions on a merge the profile refuses while the public differential has
    unresolved mismatches, and 24 patches reproducing a patch it had already
    made, having never once been told either thing. The same defect was fixed
    for the public suite in ``_test_results``: a pass count with no test names
    is enough to know something is wrong and not enough to know where to look.

    A refusal is rarer than routine chatter and is the only channel through
    which a blocked organization finds out what to fix, so refusals are carried
    even when the tail would have pushed them out.
    """
    events = list(getattr(perception, "recent_events", None) or [])
    projected = [p for p in (_project_event(e) for e in events) if p]
    tail_from = max(0, len(projected) - 12)
    tail = projected[tail_from:]
    refusals = [
        projected[i]
        for i in range(tail_from)
        if any(projected[i].get(key) for key in _EVENT_REASON_KEYS)
    ][-6:]
    return refusals + tail


def _active_protocols(world: Any) -> List[Dict[str, Any]]:
    """The rules in force, as rules rather than as identifiers.

    This section was ``list(world.protocol_registry.protocols.keys())``, so an
    agent read "proto_review_before_merge, protospec_1, protospec_2" and was
    told nothing about what any of them required. An organization cannot comply
    with a rule it cannot read, and the ledger shows what that looked like:
    proto_review_before_merge recorded 12 uses and 0 enforcements by t144, where
    a "use" is the system keyword-matching an action against the rule's text,
    not an agent having followed it.

    The spec carries the text; the registry carries the lifecycle. Both are
    needed: an adopted rule and a merely proposed one place different
    obligations, and only the registry knows which this is.
    """
    registry = getattr(world, "protocol_registry", None)
    live = getattr(registry, "protocols", {}) or {}
    manager = getattr(world, "proposal_manager", None)
    specs = getattr(manager, "protocol_specs", None) or {}
    by_id = {str(getattr(s, "protocol_id", "")): s for s in specs.values()}

    rows = []
    for protocol_id, protocol in live.items():
        spec = by_id.get(str(protocol_id))
        row: Dict[str, Any] = {
            "protocol_id": protocol_id,
            "status": getattr(protocol, "adoption_status", None),
        }
        if spec is not None:
            rule = str(getattr(spec, "enforcement_rule", "") or "").strip()
            steps = [str(s) for s in (getattr(spec, "required_steps", None) or [])]
            row.update({
                "name": str(getattr(spec, "name", "") or "")[:160],
                "applies_to": [
                    str(a) for a in (getattr(spec, "affected_actions", None) or [])
                ][:10],
                "rule": rule[:400],
                "required_steps": steps[:10],
            })
        rows.append(row)
    return rows


def _search_results(perception: Any) -> List[Dict[str, Any]]:
    """What this agent's own searches turned up.

    internal_search is the most frequent action in the run -- 131 of 749 by
    t144 -- and the perception adapter records every one of them, with the query
    and the object ids it retrieved. Nothing read the field. An agent could
    therefore search, be told the search happened, and never see a result, which
    makes searching again the only available follow-up.
    """
    rows = []
    for log in list(getattr(perception, "visible_search_results", None) or [])[-8:]:
        if not isinstance(log, dict):
            continue
        found = [str(x) for x in (log.get("results") or [])][:12]
        rows.append({
            "query": str(log.get("query") or "")[:160],
            "domain": log.get("domain"),
            "found": found or "(nothing matched)",
        })
    return rows


def _meetings(perception: Any) -> List[Dict[str, Any]]:
    """Meetings this agent is a participant in, and whether it turned up."""
    rows = []
    for meeting in list(getattr(perception, "visible_meetings", None) or [])[-8:]:
        if not isinstance(meeting, dict):
            continue
        rows.append({
            "meeting_id": meeting.get("meeting_id"),
            "title": str(meeting.get("title") or "")[:120],
            "type": meeting.get("type"),
            "status": meeting.get("status"),
            "scheduled_tick": meeting.get("scheduled_tick"),
            "attended": meeting.get("attended"),
        })
    return rows


def _experiments(perception: Any) -> List[Dict[str, Any]]:
    """Experiments this agent owns. run_cheap_pilot ran 55 times by t144."""
    rows = []
    for experiment in list(getattr(perception, "visible_experiments", None) or [])[-8:]:
        if not isinstance(experiment, dict):
            continue
        rows.append({
            "experiment_id": experiment.get("experiment_id"),
            "title": str(experiment.get("title") or "")[:120],
            "status": experiment.get("status"),
        })
    return rows


def _test_results(world: Any, *, actor_id=None) -> Dict[str, Any]:
    """What the public suite last reported, with the failing tests named.

    The suite is the only feedback the organization has on whether an edit
    worked, and test_history already derives every part of this — which tests
    keep failing, which stopped, whether the suite shrank. None of it reached
    the decision prompt, so an agent choosing what to fix next was working from
    a pass count with no test names: enough to know something is wrong, not
    enough to know where to look.
    """
    from environments.org_env.product import test_history
    from environments.org_env.cooperbench.source_reads import actor_source_read_view
    source_view = actor_source_read_view(world, actor_id)
    if source_view is not None:
        from environments.org_env.cooperbench.actor_workspace import actor_public_test_record
        record = actor_public_test_record(world, actor_id)
        if not record:
            return {"available": False, "source_snapshot": source_view.snapshot.receipt(),
                    "note": "No public test receipt matches your current private source."}
        return {"available": record.get("status") in {"passed", "failed"},
                "source_snapshot": record["source_snapshot"], "status": record.get("status"),
                "latest_tick": record.get("tick"), "latest_ok": record.get("passed"),
                "failing_now": list(record.get("failed_tests") or [])[:40],
                "summary": record.get("summary", ""),
                "failure_brief": record.get("failure_brief", "")}

    latest = test_history.latest_recorded_run(world)
    if not latest:
        return {"runs_recorded": 0, "note": "the public suite has not been run yet"}
    summary = test_history.summarize(world)
    return {
        "runs_recorded": summary["runs_recorded"],
        "latest_tick": latest.get("tick"),
        "latest_ok": summary["latest_ok"],
        "collected": latest.get("collected"),
        "failing_now": list(latest.get("failed") or [])[:40],
        "persistently_failing": summary["persistently_failing"][:20],
        "regressions_since_start": summary["regressions"][:20],
        "demonstrated_fixes": summary["demonstrated_fixes"][:20],
        "suite_size_regression": summary["suite_size_regression"],
    }


def _pending_approvals(agent_id: str, world) -> List[Dict[str, Any]]:
    """Proposals this agent is a designated approver of and hasn't decided yet, with
    enough detail (title/scores/risks) to vote on via approve/reject/request changes."""
    pm = getattr(world, "proposal_manager", None)
    if pm is None or getattr(world, "approval_mode", "auto") == "auto":
        return []
    out = []
    for p in pm.proposals.values():
        if p.status != "under_review" or agent_id not in (p.approval_required_from or []):
            continue
        if agent_id in p.approved_by or agent_id in p.rejected_by:
            continue
        out.append({"proposal_id": p.proposal_id, "type": p.proposal_type, "title": p.title,
                    "summary": p.summary[:160], "usefulness": p.usefulness_score,
                    "risk": p.risk_score, "adoption_score": p.adoption_score,
                    "risks": list(p.risks)[:3],
                    "suggested_revision": getattr(p, "suggested_revision", ""),
                    "required_actions": list(getattr(p, "required_actions", []) or [])})
        if getattr(world, "_cooperbench_delivery_focus", False):
            out[-1].update({
                "target_problem": p.target_problem,
                "proposed_solution": p.proposed_solution,
                "required_artifacts": list(p.required_artifacts or []),
            })
    return out


def _active_tools(world) -> List[str]:
    pm = getattr(world, "proposal_manager", None)
    if pm is None:
        return []
    return [f"{t.tool_id}: {t.name}" for t in pm.tools.values() if t.status == "active"]


def _valid_targets(candidates: List[Any], world: Any = None, *, actor_id=None) -> set:
    out = set()
    for c in candidates:
        for k in _TARGET_KEYS:
            v = (c.parameters or {}).get(k)
            if isinstance(v, str) and v:
                out.add(v)
        if getattr(c, "target_uid", None):
            out.add(c.target_uid)
    # product artifacts + adopted tools are always valid targets
    artifacts = getattr(world, "product_artifacts", {}) or {}
    from environments.org_env.cooperbench.source_reads import actor_source_read_view
    source_view = actor_source_read_view(world, actor_id)
    if source_view is None:
        out.update(artifacts)
    else:
        visible = set(source_view.catalog) | {
            oid for oid, art in artifacts.items() if getattr(art, "artifact_type", "") == "issue"
        }
        out.difference_update(set(artifacts) - visible)
        out.update(visible)
    pm = getattr(world, "proposal_manager", None)
    if pm is not None:
        out.update(t.tool_id for t in pm.tools.values() if t.status == "active")
    return out


def _source_visible_candidates(candidates, world, actor_id):
    """The prompt menu and later decision lookup use the same private targets."""
    from environments.org_env.cooperbench.source_reads import actor_source_read_view
    view = actor_source_read_view(world, actor_id)
    if view is None:
        return list(candidates)
    artifacts = getattr(world, "product_artifacts", {}) or {}
    hidden_ids = {oid for oid, art in artifacts.items()
                  if getattr(art, "artifact_type", "") != "issue" and oid not in view.catalog}
    visible_paths = set(view.catalog.values())
    out = []
    for candidate in candidates:
        params = getattr(candidate, "parameters", {}) or {}
        if any(params.get(key) in hidden_ids for key in _TARGET_KEYS
               if isinstance(params.get(key), str)) or getattr(candidate, "target_uid", None) in hidden_ids:
            continue
        path = params.get("file_path")
        explicit_create = (candidate.action_type == "edit_repo_file"
                           and isinstance(params.get("new_file_path"), str)
                           and bool(params["new_file_path"]))
        if (path and not explicit_create and candidate.action_type in {"edit_repo_file", "edit_doc", "review_doc"}
                and path not in visible_paths):
            continue
        out.append(candidate)
    return out


def _candidate_target(candidate: Any) -> Optional[str]:
    for key in _TARGET_KEYS:
        value = (candidate.parameters or {}).get(key)
        if isinstance(value, str) and value:
            return value
    target_uid = getattr(candidate, "target_uid", None)
    return str(target_uid) if target_uid else None


def _candidate_option(candidate: Any) -> Dict[str, Any]:
    """One menu row, carrying what the system already knows about the option.

    The action type and a target id do not distinguish three repo edits queued
    against three different files: they rendered as three identical rows, and
    the reason each was offered — including ``edit_goal``, which for an OSS
    backlog item is the issue's own acceptance text — was dropped on the floor.
    The chooser was picking blind and only learned what it was fixing one layer
    later, inside the code editor. Underscore-prefixed keys stay internal.
    """
    params = candidate.parameters or {}
    option: Dict[str, Any] = {
        "candidate_action": candidate.action_type,
        "target_object_id": _candidate_target(candidate),
        "channel_id": params.get("channel_id"),
    }
    for key, value in params.items():
        if key.startswith("_") or key == "channel_id" or value in (None, "", [], {}):
            continue
        option.setdefault(key, value)
    rationale = str(getattr(candidate, "rationale", "") or "").strip()
    if rationale:
        option["why"] = rationale
    return option


class LLMActionPolicy:
    """Asks the LLM to choose among the candidate pool; returns an ActionDecision."""

    def __init__(self) -> None:
        self._seq = 0

    def decide(self, agent_id: str, world: Any, perception: Any, candidates: List[Any],
               llm_client: OrgLLMClient) -> ActionDecision:
        self._seq += 1
        tick = int(getattr(world, "world_tick", 0))
        did = f"dec_{self._seq}"
        ctx = build_action_context(agent_id, world, perception, candidates)
        rm = getattr(world, "reflection_manager", None)
        wish_ids = (rm.context_for_decision(agent_id, world).get("open_wishes", []) if rm else [])
        ep_ids = [e["id"] for e in ctx["active_episodes"]]
        agent = world.agents.get(agent_id)
        system = system_for(agent, world, "action_decision", ACTION_SYSTEM)
        # Prefix Cache Rule: per-agent identity travels in the USER message so the
        # system prompt stays byte-identical across agents (see prompt_assets).
        _identity = agent_identity_for(agent, world)
        _identity = (_identity + "\n\n") if _identity else ""
        from environments.org_env.cooperbench.visibility import is_strict_coop
        # Strict Cooper product context contains actor-visible requirements and
        # belongs below the shared prefix, unlike the native public task board.
        shared = {key: ctx[key] for key in SHARED_CONTEXT_ORDER if key in ctx
                  and not (key == "product_context" and is_strict_coop(world))}
        personal = {key: value for key, value in ctx.items() if key not in shared}
        personal_order = (("product_context",) + PERSONAL_CONTEXT_ORDER
                          if is_strict_coop(world) else PERSONAL_CONTEXT_ORDER)
        # The menu and its targets are the action space, not context: a length
        # budget that drops entries there decides what the agent may do.
        user = (
            render_context(shared, SHARED_CONTEXT_ORDER, budgets=ACTION_MENU_BUDGETS)
            + "\n\n"
            + _identity
            + f"Choose the next intentional action for {agent_id} at tick {tick}.\n\n"
            + render_context(personal, personal_order,
                             budgets=ACTION_MENU_BUDGETS)
            + "\n\nReturn ONLY JSON matching the action schema "
              "(candidate_action and target_object_id MUST identify one exact "
              "entry from candidate_options)."
        )
        try:
            data = llm_client.generate_json(system, user, ACTION_DECISION_SCHEMA)
        except LLMError as e:
            return ActionDecision(decision_id=did, agent_id=agent_id, tick=tick,
                                  decision_source="llm_error", validation_status="rejected",
                                  rejection_reason=f"llm_error: {e}",
                                  related_episode_ids=ep_ids, related_wish_ids=wish_ids)
        return ActionDecision(
            decision_id=did, agent_id=agent_id, tick=tick, decision_source="llm",
            candidate_action=str(data.get("candidate_action", "")),
            candidate_speech_act=data.get("candidate_speech_act"),
            target_agent_id=data.get("target_agent_id"),
            target_object_id=data.get("target_object_id"),
            channel_id=data.get("channel_id"), params=dict(data.get("params") or {}),
            rationale=str(data.get("rationale", "")), expected_effect=data.get("expected_effect"),
            risk_assessment=data.get("risk_assessment"),
            confidence=_coerce_confidence(data.get("confidence")),
            related_episode_ids=ep_ids, related_wish_ids=wish_ids)


class ActionValidator:
    """Hard gate before any execution: the LLM's choice must be a legal action on a
    legal target in a legal channel. Anything else is rejected without execution."""

    def validate(self, decision: ActionDecision, world: Any, candidates: List[Any]) -> ValidationResult:
        if decision.decision_source != "llm":
            return ValidationResult(False, "not an llm decision")
        candidates = _source_visible_candidates(candidates, world, decision.agent_id)
        pool = {c.action_type for c in candidates}
        if not decision.candidate_action:
            return ValidationResult(False, "empty candidate_action")
        if decision.candidate_action not in pool:
            # An edit on a module some open issue points at is legal work even
            # on a tick that dealt no edit candidate at all; see
            # reachable_edit_candidate.
            if reachable_edit_candidate(decision, world) is None:
                return ValidationResult(
                    False, f"action '{decision.candidate_action}' not in available_actions")
        # propose_protocol: the new protocol id is created by the system (preflight
        # §12). target_object_id may be null; an LLM-invented proto_* "target" is not
        # a pre-existing object, so we drop it rather than reject the whole action.
        if decision.candidate_action == "propose_protocol":
            if decision.target_object_id and str(decision.target_object_id).startswith("proto"):
                decision.target_object_id = None
        elif decision.candidate_action == "schedule_meeting":
            # v14 P4: the meeting id is MINTED by the handler; an LLM-referenced meeting_* (the
            # topic it wants to discuss) is advisory, not a pre-existing object — drop an invalid
            # target instead of rejecting the whole action (fixes ~31 schedule_meeting rejections).
            if decision.target_object_id and decision.target_object_id not in _valid_targets(candidates, world, actor_id=decision.agent_id):
                decision.target_object_id = None
        elif decision.target_object_id and decision.target_object_id not in _valid_targets(candidates, world, actor_id=decision.agent_id):
            if reachable_edit_candidate(decision, world) is None:
                return ValidationResult(False, f"target '{decision.target_object_id}' not a valid target")
        if decision.channel_id and decision.channel_id not in getattr(world.comm, "channels", {}):
            return ValidationResult(False, f"channel '{decision.channel_id}' does not exist")
        if not isinstance(decision.params, dict):
            return ValidationResult(False, "params must be an object")
        if candidate_for_decision(decision, candidates, world) is None:
            return ValidationResult(
                False,
                "no concrete feasible candidate matches the selected action and target",
            )
        return ValidationResult(True, "")


def reachable_edit_candidate(decision: ActionDecision, world: Any) -> Optional[Any]:
    """Mint the edit candidate the affordance would deal on a luckier tick.

    The coding affordance deals a few modules per agent per tick and rotates
    which ones, so over a run every module with an open issue comes up. But an
    agent reads the issue list, not the rotation, and names the module the
    issue actually points at. When those disagree the decision used to be
    thrown away whole: on the pilot one engineer named typeutils.py correctly
    eight times across 238 ticks and was refused every time, while the file sat
    in the queue behind three others.

    So a named target is accepted when it is a real code module that a
    currently open coding issue points at — the same test the affordance
    applies — and refused otherwise. This admits no target the affordance would
    not eventually have dealt; it only stops making the agent guess the tick.
    """
    if decision.candidate_action != "edit_repo_file":
        return None
    target = decision.target_object_id or (decision.params or {}).get("artifact_id")
    if not target:
        return None
    artifact = (getattr(world, "product_artifacts", {}) or {}).get(str(target))
    from environments.org_env.cooperbench.source_reads import actor_source_read_view
    source_view = actor_source_read_view(world, decision.agent_id)
    if source_view is not None and str(target) not in source_view.catalog:
        return None
    if artifact is None or getattr(artifact, "artifact_type", "") == "issue":
        return None
    file_path = getattr(artifact, "linked_file_path", "")
    if not file_path:
        return None
    try:
        from environments.org_env.product.substrates.issue_stream import (
            unpatched_coding_issues,
        )
        open_issues = unpatched_coding_issues(world)
    except Exception:
        return None
    issue = next((it for it in open_issues
                  if str(target) in (it.get("artifact_ids") or ())), None)
    if issue is None:
        return None

    from agent_sdk.lived.core.contracts import ActionCandidate, CandidateSource
    goal = (issue.get("acceptance") or issue.get("title") or "").strip()
    return ActionCandidate(
        action_type="edit_repo_file",
        parameters={"file_path": file_path, "artifact_id": str(target),
                    "_blocker_fix": True, "_oss_issue": issue["issue_id"],
                    "edit_goal": goal},
        source=CandidateSource.NEED,
        rationale=f"Resolve issue {issue['issue_id']}: {issue.get('title', '')}".strip(),
    )


def candidate_for_decision(decision: ActionDecision, candidates: List[Any],
                           world: Any = None) -> Optional[Any]:
    """Map an accepted decision to the exact system-generated feasible candidate."""
    candidates = _source_visible_candidates(candidates, world, decision.agent_id)
    same = [c for c in candidates if c.action_type == decision.candidate_action]
    if decision.target_object_id:
        matching = [
            candidate
            for candidate in same
            if decision.target_object_id
            in (str(value) for value in (candidate.parameters or {}).values())
            or candidate.target_uid == decision.target_object_id
        ]
        if matching:
            return matching[0]
        if any(
            bool((candidate.parameters or {}).get("_cooperbench_repair_source"))
            for candidate in same
        ):
            return None
        return reachable_edit_candidate(decision, world) if world is not None else None
    if len(same) == 1:
        return same[0]
    if decision.params:
        matching = [
            candidate
            for candidate in same
            if all(
                (candidate.parameters or {}).get(key) == value
                for key, value in decision.params.items()
            )
        ]
        if len(matching) == 1:
            return matching[0]
    return None


__all__ = ["ActionDecision", "ValidationResult", "LLMActionPolicy", "ActionValidator",
           "build_action_context", "candidate_for_decision"]
