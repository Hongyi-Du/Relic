"""Runtime reset for the B1 temporary-specialist-team baseline.

The reset deliberately separates *agent-visible organizational state* from
*evaluator audit state*. Product/repository/task/budget/OSS evaluator state and
the append-only world event stream persist. Member-local memory, social
communication, commitments, local sandboxes, episodes, and governance state are
re-instantiated at each sprint boundary.
"""
from __future__ import annotations

import copy
from typing import Any

from environments.org_env.backend.agents import SEED_TEAM_BY_ID, OrgAgent
from environments.org_env.backend.clock import AgentAvailability
from environments.org_env.backend.comm import CommunicationSystem
from environments.org_env.backend.commitments import CommitmentRegistry
from environments.org_env.backend.meetings import MeetingSystem
from environments.org_env.backend.protocol import ProtocolRegistry
from environments.org_env.backend.protocol.registry import effective_min_supporters
from environments.org_env.backend.sandbox import SandboxSystem
from environments.org_env.backend.workspace import PersonalWorkspace
from environments.org_env.episodes import OrgEpisodeManager
from environments.org_env.growth import GrowthAppraiser, GrowthReconciler
from environments.org_env.policy.attractor_guard import AttractorGuard
from environments.org_env.proposals import ProposalManager
from environments.org_env.reflection import ReflectionManager
from environments.org_env.reflection.batch_manager import ReflectionBatchManager
from environments.org_env.runtime_adapter.execution import OrgActionMapper


def _availability_from_seed(agent_id: str) -> AgentAvailability:
    member = SEED_TEAM_BY_ID[agent_id]
    rhythm = member.work_rhythm
    return AgentAvailability(
        agent_id=agent_id,
        after_hours_responsiveness=float(rhythm.get("after_hours_responsiveness", 0.4)),
        weekend_work_tendency=float(rhythm.get("weekend_work_tendency", 0.3)),
        deep_work_preference=float(rhythm.get("deep_work_preference", 0.5)),
        meeting_tolerance=float(rhythm.get("meeting_tolerance", 0.5)),
    )


def reset_temporary_specialist_team(world: Any, tick: int) -> dict:
    """Re-instantiate B1 members without touching product/evaluator state."""
    agent_ids = tuple(world.agents)
    missing = [agent_id for agent_id in agent_ids if agent_id not in SEED_TEAM_BY_ID]
    if missing:
        raise ValueError(f"B1 reset cannot reconstruct non-seed agents: {missing}")

    epoch = int(getattr(world, "baseline_epoch", 0))
    archive = {
        "epoch": epoch,
        "start_tick": int(getattr(world, "_baseline_epoch_start_tick", 0)),
        "end_tick": int(tick),
        "action_count": len(getattr(world, "action_log", []) or []),
        "policy_trace_count": len(getattr(world, "policy_trace", []) or []),
        "message_count": len(getattr(world.comm, "messages", {}) or {}),
        "episode_count": len(getattr(world.episode_manager, "episodes", {}) or {}),
        "commitment_count": len(
            getattr(world.commitment_registry, "commitments", {}) or {}
        ),
        "sandbox_result_count": len(getattr(world.sandbox_system, "results", {}) or {}),
    }
    world.baseline_epoch_archives.append(archive)
    world.baseline_archived_action_log.extend(list(world.action_log))
    world.baseline_archived_policy_trace.extend(list(world.policy_trace))
    world.baseline_archived_sandbox_jobs.extend(
        copy.deepcopy(list(getattr(world.sandbox_system, "jobs", {}).values()))
    )
    world.baseline_archived_sandbox_results.extend(
        copy.deepcopy(list(getattr(world.sandbox_system, "results", {}).values()))
    )

    # Re-instantiate the same specialist identities. Product/task ownership uses
    # stable ids, so repository and backlog references remain valid.
    world.agents = {
        agent_id: OrgAgent.from_seed(SEED_TEAM_BY_ID[agent_id])
        for agent_id in agent_ids
    }
    world.personal = {
        agent_id: PersonalWorkspace(
            agent_id=agent_id,
            sandbox_id=f"sandbox_{agent_id}",
        )
        for agent_id in agent_ids
    }

    # Reset local/social organization surfaces.
    world.comm = CommunicationSystem()
    world.comm.seed_default_channels(set(agent_ids))
    world.meeting_system = MeetingSystem(world.comm)
    world.sandbox_system = SandboxSystem()
    for agent_id in agent_ids:
        world.sandbox_system.ensure_sandbox(agent_id)
    world.commitment_registry = CommitmentRegistry()
    world.episode_manager = OrgEpisodeManager()
    world.reflection_manager = ReflectionManager()
    world.reflection_batch_manager = ReflectionBatchManager()
    world.proposal_manager = ProposalManager()
    world.protocol_registry = ProtocolRegistry(
        min_supporters=effective_min_supporters(len(getattr(world, "agents", {}) or {}))
    )
    world.protocols = {}
    world.institution_context = {}
    world.company_skills = []
    world.detector_summary = {}
    world.episode_summaries = {}
    world.pending_jobs = []

    # Reset agent-visible memory and adaptive priors while retaining evaluator
    # events/action decisions on the world.
    world.memory = {agent_id: [] for agent_id in agent_ids}
    world.agent_memories = {}
    world.agent_log = []
    world._reflected_episode_ids = set()
    world._proposed_wish_ids = set()
    world._proposal_change_log = {}
    world._proto_cd = {"agent": {}, "global": -999, "day": {}}
    world._proto_fps = {}
    world._growth_signals = []
    world._growth_appraiser = GrowthAppraiser()
    world._growth_reconciler = GrowthReconciler()
    world.growth_events = []
    world._attractor_guard = AttractorGuard()
    world._attractor_masked = {}
    world.action_log = []
    world.policy_trace = []
    world.messages = []

    # Preserve the organizational clock and audit logs, but replace member
    # availability and cancel unfinished local background jobs.
    for job in world.time.background_jobs:
        if getattr(job, "status", "") == "running":
            job.status = "cancelled_at_sprint_reset"
    world.time.calendar.events = {
        event_id: event
        for event_id, event in world.time.calendar.events.items()
        if getattr(event, "event_type", "") != "meeting"
    }
    world.time.availability = {}
    for agent_id in agent_ids:
        world.time.register_agent(_availability_from_seed(agent_id))

    # Candidate cooldowns are member-local history. Keep the stateless policy,
    # feature extractor, and evaluator graph, but reset mapper emission state.
    if getattr(world, "_loop", None) is not None:
        world._loop["mapper"] = OrgActionMapper()

    world.baseline_epoch = epoch + 1
    world._baseline_epoch_start_tick = int(tick)
    world.baseline_reset_ticks.append(int(tick))
    world._ep_mark = len(world.events)
    event = {
        "type": "baseline_event",
        "subtype": "temporary_team_reinstantiated",
        "condition": getattr(world, "experiment_condition", ""),
        "epoch": world.baseline_epoch,
        "tick": int(tick),
        "preserved": [
            "repo_system",
            "product",
            "product_artifacts",
            "tasks",
            "issues",
            "budget_system",
            "oss_evaluator_state",
            "event_graph",
            "world_events",
        ],
    }
    world.events.append(event)
    return event


__all__ = ["reset_temporary_specialist_team"]
