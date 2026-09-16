"""What a member seat is allowed to see.

The autonomous loop filters the world per agent inside
``OrgPerceptionAdapter.build_perception``. A human seat must be held to exactly
the same rules, or the two controllers would be deciding under different
information and nothing measured across them would compare. These predicates
restate those rules per object so the HCI layer can ask about one thing at a
time; ``tests/org_env/test_human_seat_view.py`` asserts object-for-object that
the two agree, which is what keeps them from drifting apart.

Some object kinds carry no visibility field at all in the domain model
(proposals, protocols, releases, meeting decisions, issues). Treating them as
organization-public is a decision, not an oversight: what an organization binds
itself to, ships, or files as a defect is not private to one member. It is
written down here so it can be argued with.
"""
from __future__ import annotations

from typing import Any, List

TEAM_VISIBLE = ("team", "public")


# --- per-object predicates, mirroring perception.py ------------------------ #
def task_visible_to(task: Any, agent_id: str) -> bool:
    return (getattr(task, "visibility", "team") in TEAM_VISIBLE
            or getattr(task, "owner_id", None) == agent_id)


def document_visible_to(doc: Any, agent_id: str) -> bool:
    return (getattr(doc, "visibility", "team") in TEAM_VISIBLE
            or getattr(doc, "owner_id", None) == agent_id)


def branch_visible_to(branch: Any, agent_id: str) -> bool:
    return getattr(branch, "owner_id", None) == agent_id


def pull_request_visible_to(pr: Any, agent_id: str) -> bool:
    return getattr(pr, "author_id", None) == agent_id or agent_id in getattr(pr, "reviewers", ())


def experiment_visible_to(experiment: Any, agent_id: str) -> bool:
    return getattr(experiment, "owner_id", None) == agent_id


def meeting_visible_to(meeting: Any, agent_id: str) -> bool:
    return agent_id in getattr(meeting, "participants", ())


def event_visible_to(event: dict, agent_id: str) -> bool:
    return (event.get("agent_id") in (agent_id, None)
            or agent_id in (event.get("participants") or []))


def ci_run_visible_to(world: Any, ci: Any, agent_id: str) -> bool:
    """CI results have no visibility of their own; they follow the pull request
    they ran for, so a reviewer sees the checks on what they are reviewing."""
    for pr in world.repo_system.repo.pull_requests.values():
        # ``CIResult`` is keyed and linked by ``ci_id``.  ``run_id`` was an
        # obsolete name here, which made every real CI record appear detached
        # from its visible PR.
        if getattr(ci, "ci_id", None) in (getattr(pr, "ci_run_ids", None) or []):
            return pull_request_visible_to(pr, agent_id)
    return False


# --- organization-public kinds (see module docstring) ---------------------- #
def proposal_visible_to(proposal: Any, agent_id: str) -> bool:
    return True


def protocol_visible_to(protocol: Any, agent_id: str) -> bool:
    return True


def release_visible_to(release: Any, agent_id: str) -> bool:
    return True


def issue_visible_to(issue: Any, agent_id: str) -> bool:
    return True


# --- collections that the domain already filters --------------------------- #
def visible_messages(world: Any, agent_id: str) -> List[Any]:
    """Channel membership decides perceivability; ``read_by`` decides knowledge.
    Both are the communication system's own rules — do not restate them here."""
    return world.comm.perceivable_messages(agent_id)


def visible_channels(world: Any, agent_id: str) -> List[Any]:
    return [ch for ch in world.comm.channels.values() if agent_id in ch.members]


def visible_files(world: Any, agent_id: str) -> List[Any]:
    return world.visible_files_for(agent_id)


def visible_results(world: Any, agent_id: str) -> List[Any]:
    return world.sandbox_system.visible_results_for(agent_id)


# --- what must never reach a seat ------------------------------------------ #
#: Attributes that would leak another member's interior, the evaluator's
#: answers, or which seats are human-driven. The seat view is built by naming
#: fields explicitly rather than dumping objects, so this list is a test
#: assertion (see test_seat_view_hides_*) rather than a runtime filter.
FORBIDDEN_IN_SEAT_VIEW = (
    "controller_type",          # HCI V0 §6: seats must look alike
    "controller_log",
    "policy_trace",             # agent interior
    "reflections",
    "reflection",
    "wishes",
    "memory",
    "action_decisions",
    "hidden_tests",             # evaluator-only
    "reference_repo",
    "held_out_issues",
    "personal_workspace",       # other members' private space
)


__all__ = [
    "TEAM_VISIBLE", "FORBIDDEN_IN_SEAT_VIEW",
    "task_visible_to", "document_visible_to", "branch_visible_to",
    "pull_request_visible_to", "experiment_visible_to", "meeting_visible_to",
    "event_visible_to", "ci_run_visible_to",
    "proposal_visible_to", "protocol_visible_to", "release_visible_to",
    "issue_visible_to",
    "visible_messages", "visible_channels", "visible_files", "visible_results",
]
