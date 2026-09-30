"""Native two-person B3 communication loop; synthetic briefs, no provider calls.

These tests exercise the real perception, mapper, cap, constraints, feature and
routine scoring, profile policy, and executor. The source/feature objects are
public synthetic fixtures, not a Cooper task or evaluator asset.
"""
from __future__ import annotations

import copy
import json
import random
from types import SimpleNamespace

import pytest

from agent_sdk.lived.core.contracts import ActionCandidate
from agent_sdk.lived.domain.interfaces import DomainScenarioConfig
from environments.org_env.backend.repo.repo import PRStatus, PullRequest
from environments.org_env.backend.simulation.world import OrgWorld
from environments.org_env.cooperbench.visibility import (
    brief_visibility_receipt,
    pending_brief_shares,
    visible_feature_brief,
)
from environments.org_env.cooperbench.worker import _bind_feature_owners
from environments.org_env.llm.action_decision import (
    ActionDecision,
    ActionValidator,
    _candidate_option,
    candidate_for_decision,
)
from environments.org_env.llm.client import OrgLLMClient
from environments.org_env.product.objects import ProductArtifact
from environments.org_env.runtime_adapter.execution import MAX_CANDIDATES


F1, F2 = "cooper_feature_1", "cooper_feature_2"
OWNERS = {F1: "victor", F2: "calvin"}
CANARIES = {F1: "PROFILE_ALPHA_PRIVATE_CANARY", F2: "PROFILE_BETA_PRIVATE_CANARY"}
REVIEW_ACTIONS = {"review_pr", "formal_pr_review", "approve_pr"}


@pytest.fixture(autouse=True)
def _no_provider(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Communication visibility tests must not call a provider")

    monkeypatch.setattr(OrgLLMClient, "generate_json", forbidden)


def _world(*, peer_file_count=1):
    """Bootstrap the actual B3 roster and then bind public synthetic features."""
    scenario = DomainScenarioConfig(
        name="org_default", seed=42, corpus_version="v0",
        params={
            "num_internal_agents": 2,
            "internal_agent_ids": ["victor", "calvin"],
            "policy_mode": "mock",
            "experiment_condition": "b3_full_sociogenesis",
            "experiment_condition_explicit": True,
            "noncanonical_roster_variant": "cooperbench_two_person_b3",
        },
    )
    world = OrgWorld(scenario).build()
    template = next(iter(world.tasks.values()))
    world.tasks = {}
    world.product_artifacts = {}
    paths = {F1: ["alpha.py"], F2: [f"beta_{index}.py" for index in range(peer_file_count)]}
    briefs = {}
    for index, feature_id in enumerate((F1, F2), 1):
        for path in paths[feature_id]:
            artifact_id = "art_" + path.replace(".", "_")
            world.product_artifacts[artifact_id] = ProductArtifact(
                artifact_id=artifact_id, artifact_type="code", status="active",
                title=path, linked_file_path=path, content="VALUE = 0\n",
                mainline_content="VALUE = 0\n", known_gaps=[],
            )
        task = copy.deepcopy(template)
        task.task_id = "task_oss_" + feature_id
        task.title = f"CooperBench feature {index}"
        task.description = "Feature brief assigned privately to its owner."
        task.linked_issues = [feature_id]
        task.linked_artifacts = []
        world.tasks[task.task_id] = task
        world.product_artifacts[feature_id] = ProductArtifact(
            artifact_id=feature_id, artifact_type="issue", status="open",
            title=task.title, problem=task.description,
        )
        briefs[f"external{index}"] = (
            f"Implement {CANARIES[feature_id]}.\nFiles Modified: "
            + ", ".join(paths[feature_id])
            + f"\nPreserve {CANARIES[feature_id]}_TAIL."
        )
    world._oss_component_map = paths
    _bind_feature_owners(
        world, SimpleNamespace(agents=("external1", "external2"), tasks=briefs)
    )
    world.world_tick = 1
    assert tuple(world.agents) == ("victor", "calvin")
    assert world.action_selection_mode == "profile_policy"
    assert world.condition_spec.profile_conditioning_enabled
    assert world.condition_spec.institutionalization_enabled
    assert world._loop["policy"].use_profile_conditioning
    return world, briefs


def _native_pool(world, actor):
    packet = world._loop["perception"].build_perception(actor, world, world.world_tick)
    candidates = world._loop["mapper"].to_core_candidates(packet, world)
    candidates = world._constrain_candidates(world.agents[actor], candidates, world.time.clock)
    return packet, candidates


def _profile_select(world, actor, action_type):
    """Cover a deterministic stochastic branch without substituting a chooser.

    Trying fixed RNG seeds proves the action survives real profile scoring; it
    is not an estimate of the action's frequency in an experimental rollout.
    """
    packet, candidates = _native_pool(world, actor)
    assert any(candidate.action_type == action_type for candidate in candidates)
    agent = world.agents[actor]
    scored = []
    for candidate in candidates:
        features = world._loop["features"].extract(candidate, packet, world)
        features = world._loop["routine"].modify_features(
            agent, candidate, features, world.time.clock, world
        )
        scored.append((candidate, features))
    for seed in range(100):
        selected = world._loop["policy"].select(
            agent, packet, scored, world, rng=random.Random(seed)
        )
        if selected is not None and selected.action_type == action_type:
            assert any(selected is candidate for candidate in candidates)
            # The auxiliary LLM-direct mapping must also preserve the original
            # candidate, despite not exposing underscore flags in its menu.
            option = _candidate_option(selected)
            decision = ActionDecision(
                decision_id="visibility_regression", agent_id=actor,
                tick=world.world_tick, candidate_action=action_type,
                target_object_id=option["target_object_id"],
            )
            assert ActionValidator().validate(decision, world, candidates).passed
            assert candidate_for_decision(decision, candidates, world) is selected
            return selected
    pytest.fail(f"{action_type} never selectable through the native B3 profile policy")


def _execute_selected(world, actor, action_type):
    candidate = _profile_select(world, actor, action_type)
    original_parameters = copy.deepcopy(candidate.parameters)
    result = world._loop["execution"].execute(actor, candidate, world)
    assert result.success, result.failure_reason
    assert candidate.parameters == original_parameters
    world.world_tick += 1
    return candidate, result


def test_profile_policy_share_then_read_retains_full_parameters_and_private_canaries():
    world, briefs = _world()
    before, _ = _native_pool(world, "calvin")
    assert CANARIES[F1] not in json.dumps(before.visible_tasks)
    assert CANARIES[F2] in json.dumps(before.visible_tasks)
    share, sent = _execute_selected(world, "victor", "send_message")
    assert share.parameters["_cooperbench_brief_share"] is True
    assert share.parameters["share_feature_id"] == F1
    assert share.parameters["recipient_id"] == "calvin"
    assert CANARIES[F1] not in share.parameters["text"]
    pending = pending_brief_shares(world, "calvin")
    assert len(pending) == 1
    message_id = pending[0]["message_id"]
    assert message_id in sent.messages
    assert world.comm.messages[message_id].attachments
    assert visible_feature_brief(world, "calvin", F1) is None

    # The ordinary automatic inbox path is not an attachment-read action.
    world._process_inbox("calvin", world.world_tick)
    assert "calvin" in world.comm.messages[message_id].read_by
    unread, _ = _native_pool(world, "calvin")
    assert CANARIES[F1] not in json.dumps(unread.visible_tasks)
    assert visible_feature_brief(world, "calvin", F1) is None

    read, result = _execute_selected(world, "calvin", "read_knowledge")
    assert read.parameters["_cooperbench_brief_read"] is True
    assert read.parameters["artifact_id"] == F1
    assert read.parameters["message_id"] == message_id
    assert read.parameters["attachment_id"] == pending[0]["attachment_id"]
    receipt = brief_visibility_receipt(world, "calvin", F1)
    assert receipt == result.state_delta["cooperbench_brief_read"]
    assert receipt["source"] == "explicit_message_read"
    assert receipt["message_id"] == message_id
    assert visible_feature_brief(world, "calvin", F1) == briefs["external1"]
    after, _ = _native_pool(world, "calvin")
    assert CANARIES[F1] + "_TAIL" in json.dumps(after.visible_tasks)
    assert not pending_brief_shares(world, "calvin")
    assert visible_feature_brief(world, "victor", F2) is None


@pytest.mark.parametrize("share_state", ["unshared", "shared_unread", "inbox_triaged"])
def test_capped_profile_pool_cannot_review_or_approve_an_unread_peer_brief(
    monkeypatch, share_state
):
    world, _ = _world(peer_file_count=MAX_CANDIDATES + 1)
    if share_state != "unshared":
        _execute_selected(world, "victor", "send_message")
    if share_state == "inbox_triaged":
        world._process_inbox("calvin", world.world_tick)
    assert visible_feature_brief(world, "calvin", F1) is None

    pr = PullRequest(
        pr_id="pr_alpha", author_id="victor", source_branch="branch_alpha",
        linked_issue=F1, linked_issue_ids=[F1], reviewers=["calvin"],
        status=PRStatus.REVIEW_REQUESTED, ci_passed=True,
    )
    world.repo_system.repo.pull_requests[pr.pr_id] = pr
    before = copy.deepcopy(vars(pr))
    packet, candidates = _native_pool(world, "calvin")
    assert len(world._cooperbench_sdl_state["required_feature_paths"][F2]) > MAX_CANDIDATES
    assert len(candidates) <= MAX_CANDIDATES
    assert not (REVIEW_ACTIONS & {candidate.action_type for candidate in candidates})
    assert CANARIES[F1] not in json.dumps(packet.visible_tasks)

    def must_not_execute_review(*args, **kwargs):
        raise AssertionError("Unread peer brief must be rejected before its review handler")

    for action_type in sorted(REVIEW_ACTIONS):
        monkeypatch.setattr(
            world._loop["execution"], "_h_" + action_type, must_not_execute_review,
            raising=False,
        )
        decision = ActionDecision(
            decision_id="unread_review", agent_id="calvin", tick=world.world_tick,
            candidate_action=action_type, target_object_id=pr.pr_id,
        )
        assert not ActionValidator().validate(decision, world, candidates).passed
        result = world._loop["execution"].execute(
            "calvin", ActionCandidate(action_type=action_type, parameters={"pr_id": pr.pr_id}), world
        )
        assert not result.success
        assert result.failure_reason == (
            "cooperbench_action_blocked:semantic_review_brief_unshared_or_unread"
        )
        assert vars(pr) == before
    assert visible_feature_brief(world, "calvin", F1) is None
    assert not (getattr(world, "_cooperbench_semantic_reviews", {}) or {})
