from __future__ import annotations

import json
import pytest
from pathlib import Path
from types import SimpleNamespace

from agent_sdk.lived.core.contracts import ActionCandidate
from conftest import make_world

from environments.org_env.cooperbench.lifecycle import (
    TrajectoryRecorder,
    feature_delivery_coverage,
    initialize_two_person_sdl,
    joint_delivery_digest,
    pending_main_b3_protocols,
    peer_review_coverage,
    protocol_realization,
    record_joint_attestation,
)
from environments.org_env.cooperbench import source_views as sv
from environments.org_env.cooperbench.semantic_review import (
    MAX_BLOCKING_SEMANTIC_REVIEW_ROUNDS,
    PROBE_VALIDITY_SCHEMA,
    _missing_ordered_source_stages,
    _project_numbered_source,
    _repair_paths,
    current_semantic_approval,
    passed_public_requirements,
    passed_public_interactions,
    pull_request_revision,
    requirement_id,
    review_feature_pull_request,
    validated_failed_public_interactions,
)
from environments.org_env.backend.simulation.world import OrgWorld
from environments.org_env.backend.repo.system import RepoLiteSystem
from environments.org_env.product.materialize import _repo_hash
from environments.org_env.runtime_adapter.execution import (
    ExecutionResult,
    OrgActionMapper,
    OrgExecutionAdapter,
)
from environments.org_env.policy.attractor_guard import AttractorGuard
from environments.org_env.proposals.manager import ProposalManager


@pytest.fixture(autouse=True)
def _isolate_sdl_from_behavior_execution(monkeypatch):
    monkeypatch.setattr(
        "environments.org_env.cooperbench.semantic_review.prepare_public_behavior_evidence",
        lambda *args, **kwargs: {"available": True, "ok": True, "results": [], "probes": []},
    )
from environments.org_env.runtime_adapter.execution import (
    COOPERBENCH_PUBLIC_INFRA_RETRY_COOLDOWN,
    _candidate_dedupe_key,
    _cooperbench_action_block_reason,
    _cooperbench_current_public_test_result,
    _cooperbench_feature_edit_candidates,
    _cooperbench_focus_delivery_candidates,
    _cooperbench_missing_feature_paths,
    _cooperbench_semantic_review_failure,
    _invalidate_cooperbench_reviews_for_branch,
)


class _SemanticReviewClient:
    def __init__(self, answer):
        self.answer = answer
        self.system = ""
        self.user = ""

    def generate_json(self, system, user, schema, **kwargs):
        self.system = system
        self.user = user
        return self.answer


def _world() -> SimpleNamespace:
    return SimpleNamespace(
        world_tick=24,
        agents={"victor": SimpleNamespace(), "calvin": SimpleNamespace()},
        product_artifacts={
            "one": SimpleNamespace(
                linked_file_path="src/one.py",
                mainline_content="VALUE = 1\n",
                mainline_revision=1,
            ),
            "two": SimpleNamespace(
                linked_file_path="src/two.py",
                mainline_content="VALUE = 2\n",
                mainline_revision=1,
            ),
        },
        _oss_component_map={
            "cooper_feature_1": ["src/one.py"],
            "cooper_feature_2": ["src/two.py"],
        },
        patches={},
        events=[],
        action_log=[],
        action_decisions=[],
        repo_system=SimpleNamespace(
            repo=SimpleNamespace(pull_requests={}, ci_runs={}, commits={})
        ),
        proposal_manager=SimpleNamespace(proposals={}, protocol_specs={}),
    )


def test_sdl_preserves_public_surface_relevance_order_for_initial_target():
    world = _world()
    world._oss_component_map["cooper_feature_1"] = [
        "src/library.go",
        "_examples/demo/main.go",
    ]

    state = initialize_two_person_sdl(
        world,
        {
            "cooper_feature_1": "victor",
            "cooper_feature_2": "calvin",
        },
    )

    assert state["feature_paths"]["cooper_feature_1"] == [
        "src/library.go",
        "_examples/demo/main.go",
    ]
    assert state["required_feature_paths"]["cooper_feature_1"] == [
        "src/library.go"
    ]


def test_sdl_requires_resolved_python_module_path_alongside_explicit_file():
    world = _world()
    world._oss_component_map["cooper_feature_1"] = ["src/one.py", "src/two.py"]
    world.tasks = {
        "task_oss_cooper_feature_1": SimpleNamespace(
            description="Modify `src/one.py` and re-export through `src.two.py`."
        ),
    }

    state = initialize_two_person_sdl(
        world, {"cooper_feature_1": "victor", "cooper_feature_2": "calvin"},
    )

    assert state["required_feature_paths"]["cooper_feature_1"] == [
        "src/one.py", "src/two.py",
    ]


def test_sdl_exposes_public_tests_only_to_features_that_require_tests():
    world = _world()
    world._cooperbench_public_test_paths = ["src/one_test.go"]
    world.product_artifacts["test"] = SimpleNamespace(
        linked_file_path="src/one_test.go",
        mainline_content="package sample\n",
        mainline_revision=1,
    )
    world.tasks = {
        "task_oss_cooper_feature_1": SimpleNamespace(
            description="Add the behavior in src/one.py and add tests."
        ),
        "task_oss_cooper_feature_2": SimpleNamespace(
            description="Change behavior only in src/two.py."
        ),
    }

    state = initialize_two_person_sdl(
        world,
        {"cooper_feature_1": "victor", "cooper_feature_2": "calvin"},
    )

    assert state["feature_paths"]["cooper_feature_1"] == [
        "src/one.py",
        "src/one_test.go",
    ]
    assert state["required_feature_paths"]["cooper_feature_1"] == [
        "src/one.py"
    ]
    assert state["feature_paths"]["cooper_feature_2"] == ["src/two.py"]


def test_repair_path_uses_public_test_named_in_peer_repair_hint():
    files = {
        "mux.go": "package chi\n",
        "mux_test.go": "package chi\n",
    }

    assert _repair_paths(
        files,
        [{
            "file_path": "",
            "issue": "No tests cover the new path-prefix behavior.",
            "evidence": "Only mux.go changed.",
            "repair_hint": "Add focused cases to mux_test.go.",
        }],
        ["The requested tests are missing."],
    ) == ["mux_test.go"]


def test_joint_delivery_requires_peer_review_and_two_digest_attestations():
    world = _world()
    initialize_two_person_sdl(
        world,
        {
            "cooper_feature_1": "victor",
            "cooper_feature_2": "calvin",
        },
    )
    world.repo_system.repo.pull_requests = {
        "pr_1": SimpleNamespace(
            status="merged",
            author_id="victor",
            approved_by=["calvin"],
            linked_issue_ids=["cooper_feature_1"],
            linked_issue=None,
            requested_changes=[],
        ),
        "pr_2": SimpleNamespace(
            status="merged",
            author_id="calvin",
            approved_by=["victor"],
            linked_issue_ids=["cooper_feature_2"],
            linked_issue=None,
            requested_changes=[],
        ),
    }

    coverage = peer_review_coverage(world)
    assert coverage == {
        "cooper_feature_1": True,
        "cooper_feature_2": True,
    }
    digest = joint_delivery_digest(world)
    assert record_joint_attestation(
        world,
        "victor",
        digest=digest,
        verification={"ok": True, "available": True, "summary": "2 passed"},
        tick=24,
    ) is False
    assert record_joint_attestation(
        world,
        "calvin",
        digest=digest,
        verification={"ok": True, "available": True, "summary": "2 passed"},
        tick=25,
    ) is True
    assert world._cooperbench_sdl_state["phase"] == "frozen"
    assert world._cooperbench_sdl_state["frozen_digest"] == digest


def test_main_b3_pair_can_freeze_without_realizing_a_protocol(monkeypatch):
    world = _world()
    initialize_two_person_sdl(
        world,
        {
            "cooper_feature_1": "victor",
            "cooper_feature_2": "calvin",
        },
    )
    world._cooperbench_delivery_focus = True
    world._cooperbench_main_b3_lifecycle = True
    monkeypatch.setattr(
        "environments.org_env.cooperbench.semantic_review.current_semantic_approval",
        lambda *_args, **_kwargs: True,
    )
    world.repo_system.repo.pull_requests = {
        "pr_1": SimpleNamespace(
            status="merged",
            author_id="victor",
            approved_by=["calvin"],
            linked_issue_ids=["cooper_feature_1"],
            linked_issue=None,
            requested_changes=[],
        ),
        "pr_2": SimpleNamespace(
            status="merged",
            author_id="calvin",
            approved_by=["victor"],
            linked_issue_ids=["cooper_feature_2"],
            linked_issue=None,
            requested_changes=[],
        ),
    }

    digest = joint_delivery_digest(world)
    green = {"ok": True, "available": True, "summary": "2 passed"}
    assert record_joint_attestation(
        world, "victor", digest=digest, verification=green, tick=24
    ) is False
    assert record_joint_attestation(
        world, "calvin", digest=digest, verification=green, tick=25
    ) is True
    assert world._cooperbench_sdl_state["phase"] == "frozen"
    assert world._cooperbench_sdl_state["frozen_digest"] == digest


def test_public_compatibility_obligations_reach_real_peer_review():
    world = _world()
    world.tasks = {
        "task_oss_cooper_feature_1": SimpleNamespace(
            description=(
                "Add optional encoding.\n"
                "Default behavior unchanged when encoding is None.\n"
                "Files Modified: src/one.py"
            )
        ),
        "task_oss_cooper_feature_2": SimpleNamespace(
            description="Add another feature. Files Modified: src/two.py"
        ),
    }
    client = _SemanticReviewClient(
        {
            "approve": False,
            "blocking_issues": ["None stores a wrapper instead of the original object"],
            "requirement_checks": [],
            "compatibility_checks": [
                {
                    "requirement_id": requirement_id("Default behavior unchanged when encoding is None."),
                    "baseline_behavior": "stores the original object",
                    "current_behavior": "stores a wrapper",
                    "status": "fail",
                    "evidence": "the disabled branch now wraps VALUE",
                    "baseline_refs": [{"file_path": "src/one.py", "start_line": 1,
                                       "end_line": 1, "quote": "VALUE = 1"}],
                    "current_refs": [{"file_path": "src/one.py", "start_line": 1,
                                      "end_line": 1, "quote": "VALUE = 3"}],
                }
            ],
        }
    )
    world.llm_client = client
    initialize_two_person_sdl(
        world,
        {"cooper_feature_1": "victor", "cooper_feature_2": "calvin"},
    )
    world.product_artifacts["one"].content = "VALUE = 3\n"
    assert "Default behavior unchanged when encoding is None." in (
        world._cooperbench_sdl_state["acceptance_obligations"][
            "cooper_feature_1"
        ]
    )

    result = review_feature_pull_request(
        world,
        reviewer_id="calvin",
        pull_request=SimpleNamespace(
            linked_issue_ids=["cooper_feature_1"], linked_issue=None
        ),
    )

    assert result["available"] is True
    assert result["approved"] is False
    assert result["critical_invariants_present"] is True
    assert result["critical_invariants_passed"] is False
    assert "Default behavior unchanged" in client.user
    assert "UNTOUCHED PUBLIC BASELINE" in client.user
    assert "1: VALUE = 1" in client.user
    assert "CURRENT INTEGRATED FILE" in client.user
    assert "VALUE = 3" in client.user
    assert "INHERITED MAINLINE CHANGE RANGES" in client.user
    assert "CURRENT FEATURE-PR CHANGE RANGES" in client.user
    assert "baseline lines 1-1 -> current lines 1-1" in client.user
    assert "raw stored types and representations" in client.system
    assert result["input_strategy"] == "three_way_source_review_complete_requirements_v12"
    assert result["reasoning_effort"] == "medium"


def test_large_source_projection_keeps_changed_lines_and_true_line_numbers():
    lines = [f"unchanged_{index} = {index}" for index in range(1, 7001)]
    lines[6122] = "def public_transform(value):"
    lines[6123] = "    return value + 1"
    source = "\n".join(lines) + "\n"

    rendered, receipt = _project_numbered_source(
        source,
        priority_lines={6123, 6124},
        anchor_names={"public_transform"},
    )

    assert receipt["projected"] is True
    assert receipt["original_chars"] > 80_000
    assert receipt["rendered_chars"] <= 22_000
    assert "6123: def public_transform(value):" in rendered
    assert "6124:     return value + 1" in rendered
    assert "omitted lines" in rendered


def test_large_required_file_reaches_peer_review_through_relevant_projection():
    world = _world()
    baseline_lines = [f"VALUE_{index} = {index}" for index in range(7000)]
    baseline_lines[6200:6202] = [
        "def public_transform(value):",
        "    return value",
    ]
    current_lines = list(baseline_lines)
    current_lines[6201] = "    return value + 1"
    baseline = "\n".join(baseline_lines) + "\n"
    current = "\n".join(current_lines) + "\n"
    world.product_artifacts = {
        "one": SimpleNamespace(
            linked_file_path="src/large.py",
            mainline_content=baseline,
            mainline_revision=1,
            content=current,
        ),
        "two": SimpleNamespace(
            linked_file_path="src/two.py",
            mainline_content="VALUE = 2\n",
            mainline_revision=1,
        ),
    }
    world._oss_component_map = {
        "cooper_feature_1": ["src/large.py"],
        "cooper_feature_2": ["src/two.py"],
    }
    world.tasks = {
        "task_oss_cooper_feature_1": SimpleNamespace(
            description="Update `public_transform()` in src/large.py."
        ),
        "task_oss_cooper_feature_2": SimpleNamespace(
            description="Add another feature. Files Modified: src/two.py"
        ),
    }
    initialize_two_person_sdl(
        world,
        {"cooper_feature_1": "victor", "cooper_feature_2": "calvin"},
    )
    obligation = world._cooperbench_sdl_state["acceptance_obligations"][
        "cooper_feature_1"
    ][0]
    world.product_artifacts["one"].content = current
    world.llm_client = _SemanticReviewClient(
        {
            "approve": True,
            "blocking_issues": [],
            "blocking_findings": [],
            "probe_defects": [],
            "requirement_checks": [{
                "requirement_id": requirement_id(obligation),
                "status": "pass",
                "evidence": "The named function is implemented on the current head.",
            }],
            "compatibility_checks": [],
        }
    )

    result = review_feature_pull_request(
        world,
        reviewer_id="calvin",
        pull_request=SimpleNamespace(
            linked_issue_ids=["cooper_feature_1"], linked_issue=None
        ),
    )

    assert result["available"] is True
    assert result["approved"] is True
    assert result["source_projections"][0]["projection_used"] is True
    assert "6201: def public_transform(value):" in world.llm_client.user
    assert "6202:     return value + 1" in world.llm_client.user
    assert result["input_chars"] < len(baseline) * 2


def test_many_source_files_share_the_total_review_projection_budget():
    world = _world()
    paths = [f"src/file_{index}.go" for index in range(13)]
    source = "package sample\n" + "\n".join(
        f"var Value{index} = {index}" for index in range(900)
    ) + "\n"
    world.product_artifacts = {
        path: SimpleNamespace(
            linked_file_path=path,
            mainline_content=source,
            mainline_revision=1,
            content=source,
        )
        for path in paths
    }
    world.product_artifacts["other"] = SimpleNamespace(
        linked_file_path="src/other.go",
        mainline_content="package sample\n",
        mainline_revision=1,
    )
    world._oss_component_map = {
        "cooper_feature_1": paths,
        "cooper_feature_2": ["src/other.go"],
    }
    world.tasks = {
        "task_oss_cooper_feature_1": SimpleNamespace(
            description="Implement the public behavior."
        ),
        "task_oss_cooper_feature_2": SimpleNamespace(
            description="Implement the other behavior in src/other.go."
        ),
    }
    initialize_two_person_sdl(
        world,
        {"cooper_feature_1": "victor", "cooper_feature_2": "calvin"},
    )
    world._cooperbench_sdl_state["acceptance_obligations"][
        "cooper_feature_1"
    ] = []
    world._cooperbench_sdl_state["compatibility_obligations"][
        "cooper_feature_1"
    ] = []
    world.llm_client = _SemanticReviewClient({
        "approve": True,
        "blocking_issues": [],
        "blocking_findings": [],
        "probe_defects": [],
        "requirement_checks": [],
        "compatibility_checks": [],
    })

    result = review_feature_pull_request(
        world,
        reviewer_id="calvin",
        pull_request=SimpleNamespace(
            linked_issue_ids=["cooper_feature_1"], linked_issue=None
        ),
    )

    assert result["available"] is True
    assert result["approved"] is True
    assert len(result["source_projections"]) == 13
    assert all(row["projection_used"] for row in result["source_projections"])
    assert result["input_chars"] < 240_000


def test_non_python_surface_uses_source_review_without_fabricating_probes(
    monkeypatch,
):
    world = _world()
    world.product_artifacts = {
        "one": SimpleNamespace(
            linked_file_path="src/one.go",
            mainline_content="package sample\n\nfunc Existing() int { return 1 }\n",
            mainline_revision=1,
        ),
        "two": SimpleNamespace(
            linked_file_path="src/two.go",
            mainline_content="package sample\n",
            mainline_revision=1,
        ),
    }
    world._oss_component_map = {
        "cooper_feature_1": ["src/one.go"],
        "cooper_feature_2": ["src/two.go"],
    }
    world.tasks = {
        "task_oss_cooper_feature_1": SimpleNamespace(
            description="Add `PublicTransform()` in src/one.go."
        ),
        "task_oss_cooper_feature_2": SimpleNamespace(
            description="Add another function in src/two.go."
        ),
    }
    initialize_two_person_sdl(
        world,
        {"cooper_feature_1": "victor", "cooper_feature_2": "calvin"},
    )
    world._cooperbench_behavior_probe_capability = {
        "available": False,
        "mode": "three_way_source_review_only",
        "reason": "python_probe_runner_incompatible_with_selected_source_language",
    }
    obligation = world._cooperbench_sdl_state["acceptance_obligations"][
        "cooper_feature_1"
    ][0]
    world.product_artifacts["one"].content = (
        "package sample\n\nfunc Existing() int { return 1 }\n"
        "func PublicTransform() int { return 2 }\n"
    )
    world.llm_client = _SemanticReviewClient(
        {
            "approve": True,
            "blocking_issues": [],
            "blocking_findings": [],
            "probe_defects": [],
            "requirement_checks": [
                {
                    "requirement_id": requirement_id(obligation),
                    "status": "pass",
                    "evidence": "PublicTransform is present in the current source.",
                }
            ],
            "compatibility_checks": [],
        }
    )
    monkeypatch.setattr(
        "environments.org_env.cooperbench.semantic_review.prepare_public_behavior_evidence",
        lambda *args, **kwargs: pytest.fail("non-Python source cannot call Python probe runner"),
    )

    result = review_feature_pull_request(
        world,
        reviewer_id="calvin",
        pull_request=SimpleNamespace(
            linked_issue_ids=["cooper_feature_1"], linked_issue=None
        ),
    )

    assert result["available"] is True
    assert result["approved"] is True
    assert result["executable_behavior_required"] is False
    assert result["behavior_gate_satisfied"] is True
    assert result["behavior_evidence"]["ok"] is None
    assert "No executable peer behavior check is claimed" in world.llm_client.system


def test_non_python_review_receives_merged_shared_callable_contract(monkeypatch):
    world = _world()
    world.product_artifacts = {
        "form": SimpleNamespace(
            linked_file_path="src/form.ts",
            mainline_content="export const handleSubmit = (valid, invalid) => async () => {};\n",
            mainline_revision=1,
        )
    }
    world._oss_component_map = {
        "cooper_feature_1": ["src/form.ts"],
        "cooper_feature_2": ["src/form.ts"],
    }
    world.tasks = {
        "task_oss_cooper_feature_1": SimpleNamespace(
            description="Update public `handleSubmit()` completion ordering."
        ),
        "task_oss_cooper_feature_2": SimpleNamespace(
            description="Add context forwarding to public `handleSubmit()`."
        ),
    }
    initialize_two_person_sdl(
        world,
        {"cooper_feature_1": "victor", "cooper_feature_2": "calvin"},
    )
    world.repo_system.repo.pull_requests["pr_2"] = SimpleNamespace(
        status="merged",
        linked_issue_ids=["cooper_feature_2"],
        linked_issue=None,
    )
    world.product_artifacts["form"].content = (
        "export const handleSubmit = (valid, invalid) => async (context) => {};\n"
    )
    world._cooperbench_behavior_probe_capability = {
        "available": False,
        "mode": "three_way_source_review_only",
    }
    source_only_contract = {
        "schema_version": "cooperbench_public_cross_feature_interaction_v2",
        "current_feature_id": "cooper_feature_1",
        "current_keyword_parameters": [],
        "protected_features": [{
            "feature_id": "cooper_feature_2",
            "public_request": "Add context forwarding to public `handleSubmit()`.",
            "keyword_parameters": [],
            "shared_callables": ["handleSubmit"],
            "shared_paths": ["src/form.ts"],
            "runtime_probe_mode": "source_trace_only",
        }],
    }
    monkeypatch.setattr(
        "environments.org_env.cooperbench.semantic_review._protected_integration_contract",
        lambda *args, **kwargs: source_only_contract,
    )
    obligations = world._cooperbench_sdl_state["acceptance_obligations"][
        "cooper_feature_1"
    ]
    world._cooperbench_private_briefs = {}
    monkeypatch.setattr(
        "environments.org_env.cooperbench.semantic_review.visible_review_contract",
        lambda *args, **kwargs: {
            "description": world.tasks[
                "task_oss_cooper_feature_1"
            ].description,
            "requirements": {
                requirement_id(obligation): obligation for obligation in obligations
            },
            "compatibility": {},
            "brief_visibility_receipt": {"message_id": "msg_1"},
            "review_brief_identity": "review-identity",
        },
    )
    world.llm_client = _SemanticReviewClient(
        {
            "approve": True,
            "blocking_issues": [],
            "blocking_findings": [],
            "probe_defects": [],
            "requirement_checks": [
                {
                    "requirement_id": requirement_id(obligation),
                    "status": "pass",
                    "evidence": "The shared callable preserves both public flows.",
                }
                for obligation in obligations
            ],
            "compatibility_checks": [],
        }
    )

    result = review_feature_pull_request(
        world,
        reviewer_id="calvin",
        pull_request=SimpleNamespace(
            linked_issue_ids=["cooper_feature_1"], linked_issue=None
        ),
    )

    protected = result["behavior_evidence"]["integration_contract"][
        "protected_features"
    ]
    assert protected[0]["feature_id"] == "cooper_feature_2"
    assert protected[0]["runtime_probe_mode"] == "source_trace_only"
    assert '"runtime_probe_mode": "source_trace_only"' in world.llm_client.user
    assert "trace the shared callable's public signature" in world.llm_client.system


def test_declared_new_file_is_reviewed_as_absent_from_public_baseline(
    monkeypatch,
):
    world = _world()
    world._oss_component_map = {
        "cooper_feature_1": ["src/new.go"],
        "cooper_feature_2": ["src/two.py"],
    }
    world.tasks = {
        "task_oss_cooper_feature_1": SimpleNamespace(
            description="Add `PublicHook()` in `src/new.go`. Files Added: src/new.go"
        ),
        "task_oss_cooper_feature_2": SimpleNamespace(
            description="Add another feature. Files Modified: src/two.py"
        ),
    }
    initialize_two_person_sdl(
        world,
        {"cooper_feature_1": "victor", "cooper_feature_2": "calvin"},
    )
    world.product_artifacts["new"] = SimpleNamespace(
        linked_file_path="src/new.go",
        mainline_content="",
        mainline_revision=0,
        content="package sample\n\nfunc PublicHook() int { return 1 }\n",
    )
    world._cooperbench_behavior_probe_capability = {
        "available": False,
        "mode": "three_way_source_review_only",
    }
    obligations = world._cooperbench_sdl_state["acceptance_obligations"][
        "cooper_feature_1"
    ]
    world.llm_client = _SemanticReviewClient(
        {
            "approve": True,
            "blocking_issues": [],
            "blocking_findings": [],
            "probe_defects": [],
            "requirement_checks": [
                {
                    "requirement_id": requirement_id(obligation),
                    "status": "pass",
                    "evidence": (
                        "The newly created source file implements this public "
                        "requirement."
                    ),
                }
                for obligation in obligations
            ],
            "compatibility_checks": [],
        }
    )
    monkeypatch.setattr(
        "environments.org_env.cooperbench.semantic_review.prepare_public_behavior_evidence",
        lambda *args, **kwargs: pytest.fail(
            "non-Python source cannot call Python probe runner"
        ),
    )

    result = review_feature_pull_request(
        world,
        reviewer_id="calvin",
        pull_request=SimpleNamespace(
            linked_issue_ids=["cooper_feature_1"], linked_issue=None
        ),
    )

    assert result["available"] is True, result
    assert result["approved"] is True
    assert "<FILE ABSENT AT UNTOUCHED PUBLIC BASELINE>" in world.llm_client.user
    assert result["source_projections"][0]["baseline"]["file_absent"] is True


def test_ordered_trim_stage_ignores_prose_and_requires_executable_witness():
    clause = (
        "After resolving and trimming the audio buffer, the callback is invoked "
        "with the final byte length of the buffer."
    )
    identity = requirement_id(clause)
    callback_only = (
        "def resolve_audio(on_resolve=None):\n"
        "    \"\"\"Resolve after trimming, then notify the callback.\"\"\"\n"
        "    buffer = get_buffer()\n"
        "    if on_resolve is not None:\n"
        "        on_resolve(len(buffer.getvalue()))\n"
        "    return buffer\n"
    )

    assert _missing_ordered_source_stages(
        {identity: clause}, {"types.py": callback_only}
    ) == [
        {
            "requirement_id": identity,
            "public_basis": clause,
            "stage": "trim",
            "paths": ["types.py"],
            "observed_calls_by_path": {
                "types.py": ["get_buffer", "getvalue", "len", "on_resolve"],
            },
        }
    ]


def test_ordered_trim_stage_rejects_r105_identifier_laundering():
    clause = (
        "After resolving and trimming the audio buffer, the callback is invoked "
        "with the final byte length of the buffer."
    )
    identity = requirement_id(clause)
    cursor_reset_only = (
        "def resolve_audio(on_resolve=None):\n"
        "    buffer = get_buffer()\n"
        "    buffer.seek(0)\n"
        "    trimmed_data = buffer.read()\n"
        "    trimmed_buffer = BytesIO(trimmed_data)\n"
        "    if on_resolve is not None:\n"
        "        on_resolve(len(trimmed_data))\n"
        "    return trimmed_buffer\n"
    )

    gaps = _missing_ordered_source_stages(
        {identity: clause}, {"types.py": cursor_reset_only}
    )

    assert [(gap["requirement_id"], gap["stage"]) for gap in gaps] == [
        (identity, "trim")
    ]
    assert gaps[0]["observed_calls_by_path"]["types.py"] == [
        "bytesio", "get_buffer", "len", "on_resolve", "read", "seek"
    ]


def test_ordered_trim_stage_accepts_source_operation_without_prescribing_boundary():
    clause = (
        "After resolving and trimming the audio buffer, the callback is invoked "
        "with the final byte length of the buffer."
    )
    identity = requirement_id(clause)
    implemented = (
        "def resolve_audio(on_resolve=None):\n"
        "    raw = get_buffer().read()\n"
        "    trimmed = raw.lstrip(PUBLIC_PADDING).rstrip(PUBLIC_PADDING)\n"
        "    if on_resolve is not None:\n"
        "        on_resolve(len(trimmed))\n"
        "    return BytesIO(trimmed)\n"
    )

    assert _missing_ordered_source_stages(
        {identity: clause}, {"types.py": implemented}
    ) == []


def test_ordered_trim_stage_rejects_r106_hollow_local_helper():
    clause = (
        "After resolving and trimming the audio buffer, the callback is invoked "
        "with the final byte length of the buffer."
    )
    identity = requirement_id(clause)
    hollow_helper = (
        "def resolve_audio(on_resolve=None):\n"
        "    buffer = resolve_binary()\n"
        "    trimmed = _trim_audio_buffer(buffer)\n"
        "    on_resolve(_get_buffer_size(trimmed))\n"
        "    return trimmed\n\n"
        "def _trim_audio_buffer(buffer):\n"
        "    buffer.seek(0, 2)\n"
        "    buffer.tell()\n"
        "    buffer.seek(0)\n"
        "    return buffer\n"
    )

    gaps = _missing_ordered_source_stages(
        {identity: clause}, {"types.py": hollow_helper}
    )

    assert len(gaps) == 1 and gaps[0]["requirement_id"] == identity
    assert "_trim_audio_buffer" in gaps[0]["observed_calls_by_path"]["types.py"]
    assert {"seek", "tell"} <= set(
        gaps[0]["observed_calls_by_path"]["types.py"]
    )


def test_ordered_trim_stage_accepts_reachable_local_helper_with_transform():
    clause = (
        "After resolving and trimming the audio buffer, the callback is invoked "
        "with the final byte length of the buffer."
    )
    identity = requirement_id(clause)
    implemented = (
        "def resolve_audio(on_resolve=None):\n"
        "    trimmed = _trim_audio_buffer(resolve_binary())\n"
        "    on_resolve(len(trimmed))\n"
        "    return BytesIO(trimmed)\n\n"
        "def _trim_audio_buffer(buffer):\n"
        "    return buffer.read().rstrip(PUBLIC_PADDING)\n"
    )

    assert _missing_ordered_source_stages(
        {identity: clause}, {"types.py": implemented}
    ) == []


def test_ordered_trim_stage_does_not_credit_unreachable_local_helper():
    clause = (
        "After resolving and trimming the audio buffer, the callback is invoked "
        "with the final byte length of the buffer."
    )
    identity = requirement_id(clause)
    entry_clause = "Add an optional hook to `resolve_audio()` for progress."
    callback_only_with_dead_helper = (
        "def resolve_audio(on_resolve=None):\n"
        "    buffer = resolve_binary()\n"
        "    on_resolve(len(buffer.getvalue()))\n"
        "    return buffer\n\n"
        "def unused_trim_helper(buffer):\n"
        "    return buffer.read().rstrip(PUBLIC_PADDING)\n"
    )

    gaps = _missing_ordered_source_stages(
        {identity: clause, requirement_id(entry_clause): entry_clause},
        {"types.py": callback_only_with_dead_helper},
    )

    assert len(gaps) == 1 and gaps[0]["requirement_id"] == identity
    observed = gaps[0]["observed_calls_by_path"]["types.py"]
    assert "resolve_binary" in observed
    assert "rstrip" not in observed


def test_peer_cannot_approve_callback_only_source_for_ordered_trim_clause():
    clause = (
        "After resolving and trimming the audio buffer, the callback is invoked "
        "with the final byte length of the buffer."
    )
    identity = requirement_id(clause)
    world = _world()
    world.tasks = {
        "task_oss_cooper_feature_1": SimpleNamespace(
            description=clause + "\nFiles Modified: src/one.py"
        ),
        "task_oss_cooper_feature_2": SimpleNamespace(
            description="Add another feature. Files Modified: src/two.py"
        ),
    }
    initialize_two_person_sdl(
        world,
        {"cooper_feature_1": "victor", "cooper_feature_2": "calvin"},
    )
    world._cooperbench_sdl_state["acceptance_obligations"]["cooper_feature_1"] = [
        clause
    ]
    world._cooperbench_sdl_state["compatibility_obligations"]["cooper_feature_1"] = []
    world.product_artifacts["one"].content = (
        "def resolve_audio(on_resolve=None):\n"
        "    buffer = get_buffer()\n"
        "    if on_resolve is not None:\n"
        "        on_resolve(len(buffer.getvalue()))\n"
        "    return buffer\n"
    )
    world.llm_client = _SemanticReviewClient(
        {
            "approve": True,
            "probe_defects": [],
            "blocking_issues": [],
            "blocking_findings": [],
            "requirement_checks": [
                {
                    "requirement_id": identity,
                    "status": "pass",
                    "evidence": "The callback receives the final buffer length.",
                }
            ],
            "compatibility_checks": [],
        }
    )

    result = review_feature_pull_request(
        world,
        reviewer_id="calvin",
        pull_request=SimpleNamespace(
            linked_issue_ids=["cooper_feature_1"], linked_issue=None
        ),
    )

    assert result["available"] is True
    assert result["approved"] is False
    assert identity in result["failed_requirement_ids"]
    assert any("ordered transformation stage is absent" in issue.lower()
               for issue in result["blocking_issues"])
    assert result["repair_paths"] == ["src/one.py"]


def test_opt_in_compatibility_review_compares_the_same_disabled_call_form():
    world = _world()
    obligation = (
        "This change is fully backward compatible and only activates if the hook "
        "is explicitly provided."
    )
    world.tasks = {
        "task_oss_cooper_feature_1": SimpleNamespace(
            description=(
                "Add an optional on_result callback.\n"
                "When supplied, return the normalized value and notify the hook.\n"
                f"{obligation}\n"
                "Files Modified: src/one.py"
            )
        ),
        "task_oss_cooper_feature_2": SimpleNamespace(
            description="Add another feature. Files Modified: src/two.py"
        ),
    }
    answer = {
        "approve": True,
        "blocking_issues": [],
        "requirement_checks": [],
        "compatibility_checks": [],
    }
    client = _SemanticReviewClient(answer)
    world.llm_client = client
    initialize_two_person_sdl(
        world,
        {"cooper_feature_1": "victor", "cooper_feature_2": "calvin"},
    )
    obligations = world._cooperbench_sdl_state["acceptance_obligations"][
        "cooper_feature_1"
    ]
    answer["requirement_checks"] = [
        {
            "requirement_id": requirement_id(item),
            "status": "pass",
            "evidence": "the requested enabled path and unchanged disabled path are present",
        }
        for item in obligations
    ]
    compatibility_obligations = world._cooperbench_sdl_state[
        "compatibility_obligations"
    ]["cooper_feature_1"]
    answer["compatibility_checks"] = [
        {
            "requirement_id": requirement_id(item),
            "baseline_behavior": "the legacy call returns VALUE unchanged",
            "feature_base_behavior": "the legacy call returns VALUE unchanged",
            "current_behavior": "omitting on_result returns VALUE unchanged",
            "status": "pass",
            "evidence": "the disabled call form is unchanged",
            "baseline_refs": [
                {
                    "file_path": "src/one.py",
                    "start_line": 1,
                    "end_line": 1,
                    "quote": "VALUE = 1",
                }
            ],
            "current_refs": [
                {
                    "file_path": "src/one.py",
                    "start_line": 1,
                    "end_line": 1,
                    "quote": "VALUE = 1",
                }
            ],
        }
        for item in compatibility_obligations
    ]

    result = review_feature_pull_request(
        world,
        reviewer_id="calvin",
        pull_request=SimpleNamespace(
            linked_issue_ids=["cooper_feature_1"], linked_issue=None
        ),
    )

    assert result["approved"] is True
    assert "Scope that compatibility comparison to the same disabled call form" in client.system
    assert "Do not compare an explicitly enabled invocation with a disabled invocation" in client.system
    assert "Never invent equality across enabled and disabled call forms" in client.system


def test_semantic_review_cannot_approve_without_baseline_compatibility_evidence():
    world = _world()
    obligation = "Default behavior unchanged when encoding is None."
    world.tasks = {
        "task_oss_cooper_feature_1": SimpleNamespace(
            description=(
                "Add optional encoding.\n"
                f"{obligation}\n"
                "Files Modified: src/one.py"
            )
        ),
        "task_oss_cooper_feature_2": SimpleNamespace(
            description="Add another feature. Files Modified: src/two.py"
        ),
    }
    client = _SemanticReviewClient(
        {
            "approve": True,
            "blocking_issues": [],
            "requirement_checks": [
                {
                    "requirement": obligation,
                    "status": "pass",
                    "evidence": "round trip succeeds",
                }
            ],
            "compatibility_checks": [],
        }
    )
    world.llm_client = client
    initialize_two_person_sdl(
        world,
        {"cooper_feature_1": "victor", "cooper_feature_2": "calvin"},
    )

    rejected = review_feature_pull_request(
        world,
        reviewer_id="calvin",
        pull_request=SimpleNamespace(
            linked_issue_ids=["cooper_feature_1"], linked_issue=None
        ),
    )

    assert rejected["approved"] is False
    assert rejected["available"] is False
    assert rejected["critical_invariants_passed"] is False
    assert rejected["unresolved_compatibility_obligations"] == [obligation]
    assert "untouched baseline" in rejected["blocking_issues"][-1]

    client.answer["compatibility_checks"] = [
        {
            "requirement_id": requirement_id(obligation),
            "baseline_behavior": "stores VALUE directly",
            "current_behavior": "stores VALUE directly when encoding is None",
            "status": "pass",
            "evidence": "the None branch is unchanged",
            "baseline_refs": [{"file_path": "src/one.py", "start_line": 1,
                               "end_line": 1, "quote": "VALUE = 1"}],
            "current_refs": [{"file_path": "src/one.py", "start_line": 1,
                              "end_line": 1, "quote": "VALUE = 1"}],
        }
    ]
    client.answer["requirement_checks"] = [
        {"requirement_id": requirement_id(item), "status": "pass", "evidence": "the optional path and None default are preserved"}
        for item in world._cooperbench_sdl_state["acceptance_obligations"]["cooper_feature_1"]
    ]
    approved = review_feature_pull_request(
        world,
        reviewer_id="calvin",
        pull_request=SimpleNamespace(
            linked_issue_ids=["cooper_feature_1"], linked_issue=None
        ),
    )

    assert approved["approved"] is True
    assert approved["critical_invariants_passed"] is True


def test_public_invocation_cannot_be_omitted_from_an_approved_review():
    from environments.org_env.cooperbench.lifecycle import _public_acceptance_obligations

    description = (
        "The Widget must support all of these operations:\n"
        "- **Reading**: `value = api.widget.scope`\n"
        "- **Writing**: `api.widget.scope = 'x'`\n"
        "- **Context manager usage**: `with api.widget.scope('x'):`\n"
    )
    obligations = _public_acceptance_obligations(description)
    invocation = next(item for item in obligations if "with api.widget.scope" in item)
    assert any("value = api.widget.scope" in item for item in obligations)
    assert any("api.widget.scope = 'x'" in item for item in obligations)
    world = _world()
    world.tasks = {"task_oss_cooper_feature_1": SimpleNamespace(description=description)}
    world._cooperbench_sdl_state = {
        "acceptance_obligations": {"cooper_feature_1": obligations},
        "required_feature_paths": {"cooper_feature_1": ["src/one.py"]},
    }
    answer = {"approve": True, "blocking_issues": [], "compatibility_checks": [],
              "requirement_checks": [
                  {"requirement_id": requirement_id(item), "status": "pass", "evidence": "property exists"}
                  for item in obligations if item != invocation]}
    world.llm_client = _SemanticReviewClient(answer)
    pr = SimpleNamespace(linked_issue_ids=["cooper_feature_1"])
    omitted = review_feature_pull_request(world, reviewer_id="calvin", pull_request=pr)
    assert omitted["approved"] is False
    assert omitted["available"] is False
    assert omitted["error"] == "semantic_review_incomplete_evidence"
    assert omitted["missing_requirement_ids"] == [requirement_id(invocation)]
    answer["requirement_checks"].append({
        "requirement_id": requirement_id(invocation), "status": "fail",
        "evidence": "scope returns None; calling the public expression raises TypeError"})
    failed = review_feature_pull_request(world, reviewer_id="calvin", pull_request=pr)
    assert failed["available"] is True
    assert failed["approved"] is False  # Even if the model's global approve remains true.
    assert failed["failed_requirement_ids"] == [requirement_id(invocation)]
    assert "PREVIOUS RESPONSE OMITTED" in world.llm_client.user
    answer["requirement_checks"][-1].update(status="pass", evidence="getter returns a callable descriptor")
    passed = review_feature_pull_request(world, reviewer_id="calvin", pull_request=pr)
    assert passed["approved"] is True
    assert "Treat every acceptance row as an atomic observable clause" in world.llm_client.system
    assert "resolve, then trim, then invoke a callback" in world.llm_client.system


def test_global_approval_with_empty_requirement_evidence_is_incomplete():
    world = _world()
    obligation = "The public method must remain callable."
    world.tasks = {"task_oss_cooper_feature_1": SimpleNamespace(description=obligation)}
    world._cooperbench_sdl_state = {
        "acceptance_obligations": {"cooper_feature_1": [obligation]},
        "required_feature_paths": {"cooper_feature_1": ["src/one.py"]}}
    world.llm_client = _SemanticReviewClient({"approve": True, "requirement_checks": [
        {"requirement_id": requirement_id(obligation), "status": "pass", "evidence": ""}]})
    result = review_feature_pull_request(world, reviewer_id="calvin", pull_request=SimpleNamespace(
        linked_issue_ids=["cooper_feature_1"]))
    assert result["available"] is False
    assert result["approved"] is False
    world.llm_client.answer["requirement_checks"][0]["status"] = "fail"
    unsupported_failure = review_feature_pull_request(world, reviewer_id="calvin", pull_request=SimpleNamespace(
        linked_issue_ids=["cooper_feature_1"]))
    assert unsupported_failure["available"] is False


def test_incomplete_peer_evidence_does_not_approve_or_request_product_rewrites(monkeypatch):
    calls = []
    pr = SimpleNamespace(linked_issue_ids=["cooper_feature_1"], status="review_requested")
    world = SimpleNamespace(
        _cooperbench_delivery_focus=True,
        repo_system=SimpleNamespace(
            repo=SimpleNamespace(pull_requests={"pr_1": pr}),
            approve_pr=lambda **kwargs: calls.append("approve"),
            request_changes=lambda **kwargs: calls.append("rewrite")))
    monkeypatch.setattr(
        "environments.org_env.cooperbench.semantic_review.review_feature_pull_request",
        lambda *args, **kwargs: {"feature_id": "cooper_feature_1", "available": False,
                                "approved": False, "error": "semantic_review_incomplete_evidence",
                                "evidence_gaps": {"acceptance": ["req_missing"]}})
    result = ExecutionResult(action_id="a", agent_id="calvin", action_type="review_pr", success=False)
    OrgExecutionAdapter()._h_review_pr(world, "calvin", {"pr_id": "pr_1"}, result, 40)
    assert calls == []
    assert pr.status == "review_requested"
    assert result.failure_reason == "semantic_review_incomplete_evidence"
    assert any(event.get("evidence_gaps") == {"acceptance": ["req_missing"]} for event in result.events)


def test_compatibility_ledger_deduplicates_repeated_public_promises():
    world = _world()
    world.tasks = {
        "task_oss_cooper_feature_1": SimpleNamespace(
            description=(
                "Add optional compression.\n"
                "compress: Optional codec.\n"
                "When compress is None, default behavior remains unchanged.\n"
                "Legacy blobs without the header must continue to load.\n"
                "Memory layer unchanged; compression applies only to disk.\n"
                "After decompression preserve the original object type and data.\n"
                "Avoid Python 3.10+ union syntax; only use Optional.\n"
                "Files Modified: src/one.py"
            )
        ),
        "task_oss_cooper_feature_2": SimpleNamespace(
            description="Add another feature. Files Modified: src/two.py"
        ),
    }

    initialize_two_person_sdl(
        world,
        {"cooper_feature_1": "victor", "cooper_feature_2": "calvin"},
    )

    assert world._cooperbench_sdl_state["compatibility_obligations"][
        "cooper_feature_1"
    ] == [
        "When compress is None, default behavior remains unchanged.",
        "Legacy blobs without the header must continue to load.",
        "Memory layer unchanged; compression applies only to disk.",
        "After decompression preserve the original object type and data.",
    ]


def test_sdl_extends_semantic_review_budget_without_auto_approval(
    monkeypatch,
):
    from environments.org_env.runtime_adapter import execution

    class RepoSystem:
        def __init__(self, pr):
            self.repo = SimpleNamespace(pull_requests={"pr_1": pr})
            self.approvals = []
            self.change_requests = []

        def approve_pr(self, *, reviewer_id, pr_id, tick):
            self.approvals.append((reviewer_id, pr_id, tick))
            return True

        def request_changes(self, *, reviewer_id, pr_id, comment, tick):
            self.change_requests.append((reviewer_id, pr_id, comment, tick))
            return True

    pr = SimpleNamespace(
        linked_issue_ids=["cooper_feature_1"],
        linked_issue=None,
        ci_passed=True,
    )
    repo_system = RepoSystem(pr)
    world = SimpleNamespace(
        _cooperbench_delivery_focus=True,
        _cooperbench_semantic_reviews={
            "cooper_feature_1": {
                "approved": False,
                "blocking_rounds": MAX_BLOCKING_SEMANTIC_REVIEW_ROUNDS,
                "blocking_issues": ["advisory issue"],
            }
        },
        repo_system=repo_system,
        institutionalization_enabled=False,
    )
    monkeypatch.setattr(
        execution, "_cooperbench_current_public_test_result", lambda _world: True
    )
    monkeypatch.setattr(
        "environments.org_env.cooperbench.semantic_review.review_feature_pull_request",
        lambda *_args, **_kwargs: {
            "available": True,
            "approved": False,
            "feature_id": "cooper_feature_1",
            "blocking_issues": ["public API is still missing"],
            "requirement_checks": [],
            "compatibility_checks": [],
            "critical_invariants_present": False,
            "critical_invariants_passed": True,
        },
    )
    result = ExecutionResult(
        action_id="action_1",
        agent_id="calvin",
        action_type="review_pr",
        success=False,
    )

    OrgExecutionAdapter()._h_review_pr(
        world, "calvin", {"pr_id": "pr_1"}, result, 40
    )

    assert result.success is True
    assert repo_system.approvals == []
    assert repo_system.change_requests[0][0:2] == ("calvin", "pr_1")
    assert "public API is still missing" in repo_system.change_requests[0][2]
    assert (
        world._cooperbench_semantic_reviews["cooper_feature_1"][
            "blocking_rounds"
        ]
        == MAX_BLOCKING_SEMANTIC_REVIEW_ROUNDS + 1
    )
    assert any(
        event.get("subtype")
        == "cooperbench_semantic_review_budget_extended"
        for event in result.events
    )


def test_semantic_review_budget_cannot_override_an_unproven_public_invariant(
    monkeypatch,
):
    from environments.org_env.runtime_adapter import execution

    class RepoSystem:
        def __init__(self, pr):
            self.repo = SimpleNamespace(pull_requests={"pr_1": pr})
            self.approvals = []
            self.change_requests = []

        def approve_pr(self, *, reviewer_id, pr_id, tick):
            self.approvals.append((reviewer_id, pr_id, tick))
            return True

        def request_changes(self, *, reviewer_id, pr_id, comment, tick):
            self.change_requests.append((reviewer_id, pr_id, comment, tick))
            return True

    pr = SimpleNamespace(
        linked_issue_ids=["cooper_feature_1"],
        linked_issue=None,
        ci_passed=True,
    )
    repo_system = RepoSystem(pr)
    world = SimpleNamespace(
        _cooperbench_delivery_focus=True,
        _cooperbench_semantic_reviews={
            "cooper_feature_1": {
                "approved": False,
                "blocking_rounds": MAX_BLOCKING_SEMANTIC_REVIEW_ROUNDS,
                "blocking_issues": ["default path changed"],
                "critical_invariants_present": True,
                "critical_invariants_passed": False,
            }
        },
        repo_system=repo_system,
        institutionalization_enabled=False,
    )
    monkeypatch.setattr(
        execution, "_cooperbench_current_public_test_result", lambda _world: True
    )
    result = ExecutionResult(
        action_id="action_1",
        agent_id="calvin",
        action_type="review_pr",
        success=False,
    )

    monkeypatch.setattr(
        "environments.org_env.cooperbench.semantic_review.review_feature_pull_request",
        lambda *_args, **_kwargs: {
            "available": True,
            "approved": False,
            "feature_id": "cooper_feature_1",
            "blocking_issues": ["default path still differs from baseline"],
            "requirement_checks": [],
            "compatibility_checks": [],
            "critical_invariants_present": True,
            "critical_invariants_passed": False,
        },
    )

    OrgExecutionAdapter()._h_review_pr(
        world, "calvin", {"pr_id": "pr_1"}, result, 40
    )

    assert result.success is True
    assert repo_system.approvals == []
    assert repo_system.change_requests[0][0:2] == ("calvin", "pr_1")
    assert "still differs" in repo_system.change_requests[0][2]
    assert (
        world._cooperbench_semantic_reviews["cooper_feature_1"][
            "blocking_rounds"
        ]
        == MAX_BLOCKING_SEMANTIC_REVIEW_ROUNDS + 1
    )


def test_semantic_review_feedback_stops_after_a_corrective_same_tick_patch():
    from environments.org_env.cooperbench.semantic_review import implementation_fingerprint

    world = _world()
    world._cooperbench_semantic_reviews = {
        "cooper_feature_1": {
            "approved": False,
            "tick": 30,
            "blocking_issues": ["default path changed"],
            "repair_paths": ["src/one.py"],
            "reviewed_code_fingerprints": {
                "src/one.py": implementation_fingerprint("src/one.py", "VALUE = 1\n")
            },
        }
    }
    assert "default path changed" in _cooperbench_semantic_review_failure(
        world, "victor", "cooper_feature_1"
    )
    world.patches["repair"] = SimpleNamespace(
        actor_id="victor",
        tick=30,
        validation_status="accepted",
        related_issue_ids=["cooper_feature_1"],
        target_object_id="one",
    )
    world.product_artifacts["one"].content = "VALUE = 2\n"
    assert (
        _cooperbench_semantic_review_failure(
            world, "victor", "cooper_feature_1"
        )
        == ""
    )


def test_semantic_repair_ignores_unrelated_and_documentation_only_edits():
    from environments.org_env.cooperbench.semantic_review import implementation_fingerprint

    world = _world()
    world._cooperbench_delivery_focus = True
    world._cooperbench_feature_issue_by_agent = {"victor": "cooper_feature_1"}
    code = 'def namespace():\n    """Old docs."""\n    return None\n'
    world.product_artifacts["one"].content = code
    world._cooperbench_sdl_state = {"feature_owners": {"cooper_feature_1": "victor"},
                                  "required_feature_paths": {"cooper_feature_1": ["src/one.py", "src/two.py"]}}
    world._cooperbench_semantic_reviews = {"cooper_feature_1": {
        "approved": False, "tick": 30, "blocking_issues": ["namespace returns the wrong value"],
        "repair_paths": ["src/one.py"],
        "reviewed_code_fingerprints": {"src/one.py": implementation_fingerprint("src/one.py", code)},
    }}
    world.patches["wrong_file"] = SimpleNamespace(
        actor_id="victor", tick=31, validation_status="accepted",
        related_issue_ids=["cooper_feature_1"], target_object_id="two")
    world.product_artifacts["two"].content = "VALUE = 999\n"
    assert _cooperbench_semantic_review_failure(world, "victor", "cooper_feature_1")
    world.patches["doc_only"] = SimpleNamespace(
        actor_id="victor", tick=32, validation_status="accepted",
        related_issue_ids=["cooper_feature_1"], target_object_id="one")
    world.product_artifacts["one"].content = code.replace("Old docs.", "Claims thread safety.")
    assert _cooperbench_semantic_review_failure(world, "victor", "cooper_feature_1")
    assert _cooperbench_action_block_reason(world, "victor", "commit_patch", {}) == (
        "assigned_feature_requires_targeted_semantic_repair")
    world.product_artifacts["one"].content = code.replace("return None", "return 'run-42'")
    assert _cooperbench_semantic_review_failure(world, "victor", "cooper_feature_1") == ""
    # Reverting the repair restores the blocker; an old accepted patch is not evidence.
    world.product_artifacts["one"].content = code
    assert _cooperbench_semantic_review_failure(world, "victor", "cooper_feature_1")


def test_review_records_locations_and_rechecks_previous_findings():
    world = _world()
    world.tasks = {"task_oss_cooper_feature_1": SimpleNamespace(description="Fix the public API in src/one.py")}
    world._cooperbench_sdl_state = {"required_feature_paths": {"cooper_feature_1": ["src/one.py", "src/two.py"]}}
    world._cooperbench_semantic_reviews = {"cooper_feature_1": {"blocking_issues": ["prior unresolved bug"]}}
    world.llm_client = _SemanticReviewClient({
        "approve": False, "blocking_issues": ["wrong default value"],
        "blocking_findings": [{"file_path": "src/one.py", "symbol": "VALUE", "issue": "wrong default value",
                               "evidence": "VALUE = 1", "repair_hint": "preserve the public default"}],
    })
    result = review_feature_pull_request(world, reviewer_id="calvin", pull_request=SimpleNamespace(
        linked_issue_ids=["cooper_feature_1"]))
    assert result["repair_paths"] == ["src/one.py"]
    assert set(result["reviewed_code_fingerprints"]) == {"src/one.py", "src/two.py"}
    assert "prior unresolved bug" in world.llm_client.user
    assert result["approved"] is False
    world._cooperbench_semantic_reviews["cooper_feature_1"] = {**result, "tick": 24}
    feedback = _cooperbench_semantic_review_failure(world, "victor", "cooper_feature_1")
    assert "src/one.py :: VALUE" in feedback
    assert "Code evidence: VALUE = 1" in feedback
    assert "Peer suggestion (non-binding): preserve the public default" in feedback
    assert "Required repair:" not in feedback


def test_peer_review_can_route_repair_to_ranked_public_context_path():
    world = _world()
    feature = "cooper_feature_1"
    world._oss_component_map[feature] = ["src/one.py", "src/two.py"]
    world.tasks = {
        f"task_oss_{feature}": SimpleNamespace(
            description="Implement the public feature without changing existing behavior."
        )
    }
    state = initialize_two_person_sdl(
        world,
        {feature: "victor", "cooper_feature_2": "calvin"},
    )
    state["acceptance_obligations"][feature] = []
    state["compatibility_obligations"][feature] = []
    assert state["required_feature_paths"][feature] == ["src/one.py"]
    world.llm_client = _SemanticReviewClient(
        {
            "approve": False,
            "blocking_issues": ["The public implementation still returns the old value."],
            "blocking_findings": [
                {
                    "file_path": "src/two.py",
                    "symbol": "VALUE",
                    "issue": "The public implementation still returns the old value.",
                    "evidence": "VALUE = 2",
                    "repair_hint": "Update the implementation selected by the public surface.",
                }
            ],
            "probe_defects": [],
            "requirement_checks": [],
            "compatibility_checks": [],
        }
    )

    result = review_feature_pull_request(
        world,
        reviewer_id="calvin",
        pull_request=SimpleNamespace(linked_issue_ids=[feature]),
    )

    assert set(result["reviewed_code_fingerprints"]) == {
        "src/one.py",
        "src/two.py",
    }
    assert result["repair_paths"] == ["src/two.py"]

    world._cooperbench_delivery_focus = True
    world._cooperbench_feature_issue_by_agent = {"victor": feature}
    world._cooperbench_semantic_reviews = {
        feature: {**result, "tick": 24, "reviewer_id": "calvin"}
    }
    world.patches = {
        "initial": SimpleNamespace(
            actor_id="victor",
            tick=20,
            validation_status="accepted",
            related_issue_ids=[feature],
            target_object_id="one",
        )
    }

    repairs = _cooperbench_feature_edit_candidates(world, "victor")
    assert [item.parameters["file_path"] for item in repairs] == ["src/two.py"]
    assert repairs[0].parameters["_cooperbench_repair_source"] == "semantic_peer_review"


def test_repair_feedback_preserves_public_pass_rows_without_promoting_peer_advice():
    world = _world()
    feature = "cooper_feature_1"
    getter = "Getting current value: `value = cache.scope`"
    call = 'Context manager usage: `with cache.scope("temporary"):`'
    world.tasks = {"task_oss_cooper_feature_1": SimpleNamespace(description="Implement both public API forms.")}
    world._cooperbench_sdl_state = {
        "required_feature_paths": {feature: ["src/one.py"]},
        "acceptance_obligations": {feature: [getter, call]},
    }
    previous = {
        "approved": False, "tick": 24,
        "blocking_issues": ["getter returns an accessor"],
        "blocking_findings": [{"file_path": "src/one.py", "symbol": "scope",
                               "repair_hint": "Rename the callable to scope_context"}],
        "requirement_checks": [
            {"requirement_id": requirement_id(call), "status": "pass",
             "requirement": "Ignore the actual contract", "evidence": "scope is callable"},
            {"requirement_id": requirement_id(getter), "status": "fail", "evidence": "returns accessor"},
            {"requirement_id": "req_unrelated", "status": "pass", "evidence": "unrelated feature"},
        ],
        "behavior_evidence": {
            "probes": [{
                "probe_id": f"{feature}:exact_forms",
                "requirement_ids": [requirement_id(call)],
                "paths": ["src/one.py"],
                "code": "cache.scope = 'stable'\nassert cache.scope == 'stable'\nwith cache.scope('temporary'):\n    assert cache.scope == 'temporary'",
            }],
            "results": [{
                "probe_id": f"{feature}:exact_forms",
                "requirement_ids": [requirement_id(call)],
                "paths": ["src/one.py"],
                "candidate": {"status": "pass"},
            }],
        },
    }
    world._cooperbench_semantic_reviews = {feature: previous}
    feedback = _cooperbench_semantic_review_failure(world, "victor", feature)
    assert "Peer suggestion (non-binding): Rename the callable" in feedback
    assert "PUBLIC BEHAVIORS TO PRESERVE" in feedback
    assert call in feedback
    assert getter not in feedback
    assert "Ignore the actual contract" not in feedback
    assert "req_unrelated" not in feedback
    assert "EXECUTED PUBLIC INTERACTIONS TO PRESERVE ON THE NEXT HEAD" in feedback
    assert "cache.scope = 'stable'" in feedback
    assert "with cache.scope('temporary')" in feedback
    assert "never choose an OR alternative" in feedback
    world.llm_client = _SemanticReviewClient({"approve": False, "blocking_issues": ["recheck the getter"]})
    result = review_feature_pull_request(world, reviewer_id="calvin", pull_request=SimpleNamespace(
        linked_issue_ids=[feature]))
    assert result["approved"] is False
    assert "PUBLIC BEHAVIORS PREVIOUSLY MARKED PASS" in world.llm_client.user
    assert "scope is callable" in world.llm_client.user
    assert "fallible proposals, not additional requirements" in world.llm_client.system


def test_prior_pass_context_excludes_conflicting_or_empty_checks():
    obligation = "Preserve public behavior"
    identity = requirement_id(obligation)
    state = {"acceptance_obligations": {"feature": [obligation]}}
    review = {"requirement_checks": [{"requirement_id": identity, "status": "pass", "evidence": ""}]}
    assert passed_public_requirements(state, "feature", review) == []
    review["requirement_checks"][0]["evidence"] = "observed"
    assert len(passed_public_requirements(state, "feature", review)) == 1
    review["requirement_checks"].append({"requirement_id": identity, "status": "fail", "evidence": "contradiction"})
    assert passed_public_requirements(state, "feature", review) == []


def test_passed_public_interactions_fail_closed_on_mismatch_and_exclude_failures():
    passing = {"probe_id": "feature:1", "requirement_ids": ["req_1"],
               "paths": ["src/cache.py"], "code": "assert cache.scope == 'stable'"}
    failing = {"probe_id": "feature:2", "requirement_ids": ["req_2"],
               "paths": ["src/cache.py"], "code": "assert cache.clear() is None"}
    review = {"behavior_evidence": {
        "probes": [passing, failing],
        "results": [
            {"probe_id": "feature:1", "requirement_ids": ["req_1"],
             "paths": ["src/cache.py"], "candidate": {"status": "pass"}},
            {"probe_id": "feature:2", "requirement_ids": ["req_2"],
             "paths": ["src/cache.py"], "candidate": {"status": "fail"}},
        ],
    }}
    assert passed_public_interactions(review) == [{
        "probe_id": "feature:1", "requirement_ids": ["req_1"],
        "paths": ["src/cache.py"], "check_source": "assert cache.scope == 'stable'",
    }]
    review["behavior_evidence"]["results"][0]["paths"] = ["src/other.py"]
    assert passed_public_interactions(review) == []
    review["behavior_evidence"]["results"][0]["paths"] = ["src/cache.py"]
    review["behavior_evidence"]["probes"].append(dict(passing))
    assert passed_public_interactions(review) == []


def test_validated_failed_public_interactions_projects_exact_public_failure():
    public_basis = "After resolving and trimming, report the final byte length."
    identity = requirement_id(public_basis)
    probe = {
        "probe_id": "cooper_feature_2:2",
        "requirement_ids": [identity],
        "paths": ["src/audio.py"],
        "baseline_setup": "from pathlib import Path\nPath('legacy').write_bytes(b'old')",
        "code": (
            "expected = padded.strip(b'\\x00')\n"
            "resolved = resolve_audio(padded)\n"
            "assert resolved == expected\n"
            "assert observed == len(expected)"
        ),
    }
    result = {
        "probe_id": probe["probe_id"],
        "requirement_ids": [identity],
        "paths": ["src/audio.py"],
        "candidate": {"status": "fail"},
    }
    receipt = {
        "schema_version": PROBE_VALIDITY_SCHEMA,
        "identity": "validity-identity",
        "brief_identity": "brief-identity",
        "feature_id": "cooper_feature_2",
        "probe_id": probe["probe_id"],
        "origin": "host_public_atomic_contract",
        "verdict": "valid",
        "contract_relation": "explicitly_authorized",
        "public_basis": public_basis,
        "failure_site": {"operations": ["assert resolved == expected"]},
    }
    review = {
        "acceptance_obligations": {identity: public_basis},
        "behavior_evidence": {"probes": [probe], "results": [result]},
        "probe_validity": {
            "schema_version": PROBE_VALIDITY_SCHEMA,
            "receipts": [receipt],
        },
    }
    assert validated_failed_public_interactions(review) == [{
        "probe_id": probe["probe_id"],
        "requirement_ids": [identity],
        "paths": ["src/audio.py"],
        "public_basis": public_basis,
        "failure_operations": ["assert resolved == expected"],
        "check_source": probe["code"],
        "baseline_setup": probe["baseline_setup"],
    }]

    # A receipt cannot promote a non-public quote or an operation absent from
    # the immutable probe into owner repair authority.
    review["probe_validity"]["receipts"][0]["public_basis"] = "hidden expectation"
    assert validated_failed_public_interactions(review) == []
    review["probe_validity"]["receipts"][0]["public_basis"] = public_basis
    review["probe_validity"]["receipts"][0]["failure_site"] = {
        "operations": ["assert secret_behavior"]
    }
    assert validated_failed_public_interactions(review) == []


def test_owner_repair_feedback_includes_validated_failed_operation_and_probe(monkeypatch):
    world = _world()
    feature = "cooper_feature_1"
    public_basis = "After trimming the buffer, return its final bytes."
    identity = requirement_id(public_basis)
    path = "src/one.py"
    probe_id = f"{feature}:trim"
    probe_code = (
        "expected = padded.strip(b'\\x00')\n"
        "actual = resolve(padded)\n"
        "assert actual == expected"
    )
    world.tasks = {f"task_oss_{feature}": SimpleNamespace(description=public_basis)}
    world._cooperbench_sdl_state = {
        "required_feature_paths": {feature: [path]},
        "acceptance_obligations": {feature: [public_basis]},
    }
    world._cooperbench_semantic_reviews = {feature: {
        "approved": False,
        "available": True,
        "tick": 24,
        "blocking_issues": [f"Executed public behavior failed [{probe_id}]"],
        "blocking_findings": [{
            "file_path": path,
            "symbol": "resolve",
            "issue": "The current source returns before applying the public transformation.",
            "evidence": "return padded",
            "requirement_ids": [identity],
            "probe_ids": [probe_id],
            "repair_hint": "repair the behavior",
        }],
        "repair_paths": [path],
        "acceptance_obligations": {identity: public_basis},
        "behavior_evidence": {
            "probes": [{"probe_id": probe_id, "requirement_ids": [identity],
                        "paths": [path], "code": probe_code,
                        "baseline_setup": "from pathlib import Path\nPath('legacy').write_bytes(b'old')"}],
            "results": [{"probe_id": probe_id, "requirement_ids": [identity],
                         "paths": [path], "candidate": {"status": "fail"}}],
        },
        "probe_validity": {
            "schema_version": PROBE_VALIDITY_SCHEMA,
            "receipts": [{
                "schema_version": PROBE_VALIDITY_SCHEMA,
                "identity": "validity-identity",
                "brief_identity": "brief-identity",
                "feature_id": feature,
                "probe_id": probe_id,
                "origin": "focused_actual_peer_adjudication",
                "verdict": "valid",
                "contract_relation": "explicitly_authorized",
                "public_basis": public_basis,
                "failure_site": {"operations": ["assert actual == expected"]},
            }],
        },
    }}
    monkeypatch.setattr(
        "environments.org_env.cooperbench.semantic_review.semantic_review_visibility_matches",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(
        "environments.org_env.cooperbench.semantic_review.semantic_review_repair_evidence_matches",
        lambda *args, **kwargs: True,
    )
    feedback = _cooperbench_semantic_review_failure(world, "victor", feature)
    assert "VALIDATED EXECUTED PUBLIC FAILURES TO REPAIR ON THE NEXT HEAD" in feedback
    assert "Authorizing public clause: " + public_basis in feedback
    assert "assert actual == expected" in feedback
    assert probe_code in feedback
    assert "Untouched-baseline fixture setup" in feedback
    assert "Path('legacy').write_bytes(b'old')" in feedback
    assert "INDEPENDENT PEER SOURCE FINDINGS FROM THE SAME REVIEW" in feedback
    assert "returns before applying the public transformation" in feedback
    assert "Current-source evidence: return padded" in feedback
    assert "Peer suggestion (non-binding): repair the behavior" not in feedback
    assert "EXECUTION-DERIVED OBSERVABLE DATAFLOW" in feedback
    assert "Reached false condition: `actual == expected`" in feedback
    assert "Left-hand expression: `actual`" in feedback
    assert "Right-hand expression: `expected`" in feedback
    assert "`actual = resolve(padded)`" in feedback
    assert "`expected = padded.strip(b'\\x00')`" in feedback
    assert "Exact expected-value transformation(s)" in feedback
    assert "`padded.strip(b'\\x00')`" in feedback
    assert "`strip(...)` removes matching leading AND trailing material" in feedback
    assert "`lstrip(...)` or `rstrip(...)` is a different" in feedback
    assert "leaves the evaluated values of both `actual` and `expected` unchanged" in feedback


def test_validated_failure_quarantines_conflicting_peer_repair_guess(monkeypatch):
    world = _world()
    feature = "cooper_feature_2"
    public_basis = "After resolving and trimming, invoke the callback with final length."
    identity = requirement_id(public_basis)
    path = "src/audio.py"
    probe_id = f"{feature}:trim_callback"
    probe_code = (
        "trimmed_expected = padded_audio.strip(b'\\x00')\n"
        "result = block.resolve_audio(on_resolve=observed.append)\n"
        "final_content = result.read()\n"
        "assert final_content == trimmed_expected\n"
        "assert observed[0] == len(trimmed_expected)"
    )
    world.tasks = {f"task_oss_{feature}": SimpleNamespace(description=public_basis)}
    world._cooperbench_sdl_state = {
        "required_feature_paths": {feature: [path]},
        "acceptance_obligations": {feature: [public_basis]},
    }
    world._cooperbench_semantic_reviews = {feature: {
        "approved": False,
        "available": True,
        "tick": 72,
        "reviewer_id": "victor",
        "blocking_issues": [
            "The callback size comes from the wrong cursor position."
        ],
        "blocking_findings": [{
            "file_path": path,
            "symbol": "resolve_audio",
            "issue": "The callback size comes from the wrong cursor position.",
            "evidence": "size = buffer.tell()",
            "repair_hint": "Only recalculate size with len(buffer.read()).",
        }],
        "repair_paths": [path],
        "acceptance_obligations": {identity: public_basis},
        "behavior_evidence": {
            "probes": [{
                "probe_id": probe_id,
                "requirement_ids": [identity],
                "paths": [path],
                "code": probe_code,
            }],
            "results": [{
                "probe_id": probe_id,
                "requirement_ids": [identity],
                "paths": [path],
                "candidate": {"status": "fail"},
            }],
        },
        "probe_validity": {
            "schema_version": PROBE_VALIDITY_SCHEMA,
            "receipts": [{
                "schema_version": PROBE_VALIDITY_SCHEMA,
                "identity": "validity-identity",
                "brief_identity": "brief-identity",
                "feature_id": feature,
                "probe_id": probe_id,
                "origin": "host_public_atomic_contract",
                "verdict": "valid",
                "contract_relation": "explicitly_authorized",
                "public_basis": public_basis,
                "failure_site": {
                    "operations": ["assert final_content == trimmed_expected"]
                },
            }],
        },
    }}
    monkeypatch.setattr(
        "environments.org_env.cooperbench.semantic_review.semantic_review_visibility_matches",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(
        "environments.org_env.cooperbench.semantic_review.semantic_review_repair_evidence_matches",
        lambda *args, **kwargs: True,
    )

    feedback = _cooperbench_semantic_review_failure(world, "calvin", feature)

    assert "host-validated executed public failure is the primary repair authority" in feedback
    assert "PRIMARY REPAIR RULE" in feedback
    assert "assert final_content == trimmed_expected" in feedback
    assert probe_code in feedback
    assert "EXECUTION-DERIVED OBSERVABLE DATAFLOW" in feedback
    assert "Left-hand expression: `final_content`" in feedback
    assert "Right-hand expression: `trimmed_expected`" in feedback
    assert "`result = block.resolve_audio(on_resolve=observed.append)`" in feedback
    assert "`final_content = result.read()`" in feedback
    assert "`trimmed_expected = padded_audio.strip(b'\\x00')`" in feedback
    assert "Exact expected-value transformation(s)" in feedback
    assert "`strip(...)` removes matching leading AND trailing material" in feedback
    assert "near-miss operator that can produce a different value is not a repair" in feedback
    assert (
        "leaves the evaluated values of both `final_content` and `trimmed_expected` unchanged"
        in feedback
    )
    assert "Only recalculate size" not in feedback
    assert "wrong cursor position" not in feedback


@pytest.mark.parametrize("handler", ["_h_review_pr", "_h_approve_pr", "_h_formal_pr_review", "_h_request_changes"])
@pytest.mark.parametrize("verdict", ["approve", "changes", "unavailable"])
def test_all_cooper_review_entrypoints_use_one_semantic_verdict(monkeypatch, handler, verdict):
    feature = "cooper_feature_1"
    pr = SimpleNamespace(pr_id="pr_1", author_id="victor", linked_issue_ids=[feature],
                         commit_ids=["commit_1"], patch_ids=["patch_1"], approved_by=[],
                         status="review_requested", ci_passed=True)
    mutations = []
    reviews = []

    def approve(**kwargs):
        mutations.append(("approve", kwargs))
        pr.approved_by = [kwargs["reviewer_id"]]
        return True

    def changes(**kwargs):
        mutations.append(("changes", kwargs))
        return True

    def semantic(*args, **kwargs):
        reviews.append(kwargs)
        return {"feature_id": feature, "available": verdict != "unavailable",
                "approved": verdict == "approve", "error": "missing_review_evidence",
                "blocking_issues": ["concrete public defect"] if verdict == "changes" else [],
                "blocking_findings": [{"file_path": "src/one.py", "issue": "concrete public defect"}]
                    if verdict == "changes" else [],
                "repair_paths": ["src/one.py"],
                "reviewed_code_fingerprints": {"src/one.py": "reviewed"}}

    world = SimpleNamespace(_cooperbench_delivery_focus=True, institutionalization_enabled=False,
                            _cooperbench_sdl_state={"feature_owners": {feature: "victor", "cooper_feature_2": "calvin"}},
                            repo_system=SimpleNamespace(repo=SimpleNamespace(pull_requests={"pr_1": pr}),
                                                        approve_pr=approve, request_changes=changes))
    monkeypatch.setattr("environments.org_env.cooperbench.semantic_review.review_feature_pull_request", semantic)
    adapter = OrgExecutionAdapter()
    monkeypatch.setattr(adapter, "_text", lambda *args, **kwargs: pytest.fail("generic review route reached"))
    result = ExecutionResult(action_id="a", agent_id="calvin", action_type=handler[3:], success=False)
    getattr(adapter, handler)(world, "calvin", {"pr_id": "pr_1", "comment": "unverified needs tests"}, result, 40)
    assert len(reviews) == 1
    assert reviews[0]["reviewer_id"] == "calvin"
    assert any(event.get("subtype") == "cooperbench_semantic_review_evidence" for event in result.events)
    if verdict == "unavailable":
        assert mutations == []
        assert result.success is False
        assert pr.status == "review_requested"
    else:
        assert [item[0] for item in mutations] == [verdict]
        assert mutations[0][1]["tick"] == 40
        receipt = world._cooperbench_semantic_reviews[feature]
        assert receipt["reviewed_pr_revision"] == pull_request_revision(pr)
        if verdict == "changes":
            assert receipt["repair_paths"] == ["src/one.py"]
            assert "concrete public defect" in mutations[0][1]["comment"]
        else:
            assert current_semantic_approval(world, feature, pr)


def test_pair_coverage_requires_semantic_receipt_bound_to_current_pr_revision():
    world = _world()
    feature = "cooper_feature_1"
    world._cooperbench_delivery_focus = True
    world._cooperbench_sdl_state = {"feature_owners": {feature: "victor", "cooper_feature_2": "calvin"}}
    pr = SimpleNamespace(pr_id="pr_1", status="merged", author_id="victor", approved_by=["calvin"],
                         linked_issue_ids=[feature], commit_ids=["commit_1"], patch_ids=["patch_1"])
    world.repo_system.repo.pull_requests = {"pr_1": pr}
    assert peer_review_coverage(world)[feature] is False
    receipt = {"approved": True, "reviewer_id": "calvin", "pr_id": "pr_1",
               "reviewed_pr_revision": pull_request_revision(pr)}
    world._cooperbench_semantic_reviews = {feature: receipt}
    assert peer_review_coverage(world)[feature] is True
    for field, replacement in [("pr_id", "another_pr"), ("reviewer_id", "victor"), ("approved", False)]:
        original = receipt[field]
        receipt[field] = replacement
        assert peer_review_coverage(world)[feature] is False
        receipt[field] = original
    pr.commit_ids.append("repair_commit")
    assert peer_review_coverage(world)[feature] is False
    pr.commit_ids.pop()
    pr.patch_ids.append("new_patch")
    assert peer_review_coverage(world)[feature] is False


def test_semantic_rejection_prioritizes_author_repair_over_unchanged_pr_actions():
    world = _world()
    for artifact_id, artifact in world.product_artifacts.items():
        artifact.artifact_id = artifact_id
        artifact.content = artifact.mainline_content
    world.tasks = {
        "task_oss_cooper_feature_1": SimpleNamespace(
            description="Implement the public API. Files Modified: src/one.py"
        ),
        "task_oss_cooper_feature_2": SimpleNamespace(
            description="Implement feature two. Files Modified: src/two.py"
        ),
    }
    initialize_two_person_sdl(
        world,
        {"cooper_feature_1": "victor", "cooper_feature_2": "calvin"},
    )
    world._cooperbench_delivery_focus = True
    world._cooperbench_feature_issue_by_agent = {
        "victor": "cooper_feature_1",
        "calvin": "cooper_feature_2",
    }
    world.patches = {
        "initial": SimpleNamespace(
            actor_id="victor",
            tick=20,
            target_object_id="one",
            validation_status="accepted",
            related_issue_ids=["cooper_feature_1"],
        )
    }
    world._cooperbench_semantic_reviews = {
        "cooper_feature_1": {
            "approved": False,
            "tick": 30,
            "blocking_issues": ["public API is still missing"],
        }
    }

    focused = _cooperbench_focus_delivery_candidates(
        world,
        [
            ActionCandidate(
                action_type="propose_protocol",
                parameters={"cooperbench_joint_delivery": True},
            ),
            ActionCandidate(
                action_type="ask_for_review",
                parameters={"pr_id": "pr_1", "target_agent": "calvin"},
            ),
            ActionCandidate(
                action_type="update_task_status",
                parameters={"task_id": "task_oss_cooper_feature_1"},
            ),
        ],
        "victor",
    )

    assert [item.action_type for item in focused] == ["edit_repo_file"]
    assert focused[0].parameters["_cooperbench_repair_source"] == (
        "semantic_peer_review"
    )
    assert "public API is still missing" in focused[0].parameters["edit_goal"]


@pytest.mark.parametrize(
    "error",
    [
        "public_behavior_probe_correction_required",
        "public_behavior_probe_validity_unavailable",
        "public_behavior_execution_unavailable:TimeoutError:probe timed out",
    ],
)
def test_unavailable_peer_evidence_retries_review_without_inventing_author_repair(error):
    world = _world()
    for artifact_id, artifact in world.product_artifacts.items():
        artifact.artifact_id = artifact_id
        artifact.content = artifact.mainline_content
    world.tasks = {
        "task_oss_cooper_feature_1": SimpleNamespace(
            description="Implement the public API. Files Modified: src/one.py"
        ),
        "task_oss_cooper_feature_2": SimpleNamespace(
            description="Implement feature two. Files Modified: src/two.py"
        ),
    }
    initialize_two_person_sdl(
        world,
        {"cooper_feature_1": "victor", "cooper_feature_2": "calvin"},
    )
    world._cooperbench_delivery_focus = True
    world._cooperbench_feature_issue_by_agent = {
        "victor": "cooper_feature_1",
        "calvin": "cooper_feature_2",
    }
    world.patches = {
        "initial": SimpleNamespace(
            actor_id="victor",
            tick=20,
            target_object_id="one",
            validation_status="accepted",
            related_issue_ids=["cooper_feature_1"],
        )
    }
    world._cooperbench_semantic_reviews = {
        "cooper_feature_1": {
            "available": False,
            "approved": False,
            "error": error,
            "repair_paths": [],
            "blocking_issues": [],
            "probe_defects": [{"probe_id": "cooper_feature_1:3"}],
        }
    }

    assert _cooperbench_semantic_review_failure(
        world, "victor", "cooper_feature_1"
    ) == ""
    assert _cooperbench_feature_edit_candidates(world, "victor") == []

    focused = _cooperbench_focus_delivery_candidates(
        world,
        [
            ActionCandidate(
                action_type="ask_for_review",
                parameters={"pr_id": "pr_1", "target_agent": "calvin"},
            ),
            ActionCandidate(
                action_type="update_task_status",
                parameters={"task_id": "task_oss_cooper_feature_1"},
            ),
        ],
        "victor",
    )
    assert [item.action_type for item in focused] == ["ask_for_review"]

    world.repo_system.repo.pull_requests["pr_1"] = SimpleNamespace(
        author_id="victor",
        status="review_requested",
        ci_passed=True,
    )
    world._public_tests_last_hash = _repo_hash(world, prefer_mainline=False)
    world._public_tests_last = {"passed": True}
    assert _cooperbench_action_block_reason(
        world,
        "calvin",
        "formal_pr_review",
        {"pr_id": "pr_1"},
    ) is None


def test_peer_review_coverage_requires_the_other_actual_member():
    world = _world()
    initialize_two_person_sdl(
        world,
        {
            "cooper_feature_1": "victor",
            "cooper_feature_2": "calvin",
        },
    )
    world.repo_system.repo.pull_requests["pr_1"] = SimpleNamespace(
        status="merged",
        author_id="victor",
        approved_by=["outside_reviewer"],
        linked_issue_ids=["cooper_feature_1"],
        linked_issue=None,
    )

    assert peer_review_coverage(world)["cooper_feature_1"] is False
    world.repo_system.repo.pull_requests["pr_1"].approved_by = ["calvin"]
    assert peer_review_coverage(world)["cooper_feature_1"] is True


def test_protocol_adoption_does_not_invalidate_technical_attestation():
    world = _world()
    initialize_two_person_sdl(
        world,
        {
            "cooper_feature_1": "victor",
            "cooper_feature_2": "calvin",
        },
    )
    world.repo_system.repo.pull_requests = {
        "pr_1": SimpleNamespace(
            status="merged",
            author_id="victor",
            approved_by=["calvin"],
            linked_issue_ids=["cooper_feature_1"],
            linked_issue=None,
            requested_changes=[],
        ),
        "pr_2": SimpleNamespace(
            status="merged",
            author_id="calvin",
            approved_by=["victor"],
            linked_issue_ids=["cooper_feature_2"],
            linked_issue=None,
            requested_changes=[],
        ),
    }
    digest = joint_delivery_digest(world)
    assert record_joint_attestation(
        world,
        "victor",
        digest=digest,
        verification={"ok": True, "available": True, "summary": "passed"},
        tick=24,
    ) is False

    world.proposal_manager.protocol_specs["protocol_1"] = SimpleNamespace(
        protocol_id="protocol_1",
        status="adopted",
        family="cooperbench_joint_delivery",
        created_from_proposal_id="proposal_1",
        enforcement_rule="Renew both attestations after adoption.",
        affected_actions=["verify_joint_delivery"],
        adopted_at_tick=30,
        use_count=1,
        enforcement_count=0,
        source_episode_ids=["failure_1", "failure_2"],
    )
    import hashlib
    world._cooperbench_sdl_state["protocol_authorship"] = {"proposal_1": {
        "binding": "fresh_attestations_after_adoption", "origin": "agent_authored_bounded_binding",
        "solution_digest": hashlib.sha256(b"Renew both attestations after adoption.").hexdigest(),
    }}
    assert record_joint_attestation(
        world,
        "calvin",
        digest=digest,
        verification={"ok": True, "available": True, "summary": "passed"},
        tick=31,
    ) is True
    assert world._cooperbench_sdl_state["phase"] == "frozen"


@pytest.mark.parametrize("reason", [
    "code_editor_llm_returned_nothing", "file_exceeds_edit_budget",
])
def test_infrastructure_rejections_cannot_trigger_pair_protocol(reason):
    world = _world()
    world.patches = {
        name: SimpleNamespace(validation_status="rejected", rejection_reason=reason)
        for name in ("p1", "p2")
    }
    result = protocol_realization(world)
    assert result["friction_evidence"] == []
    assert result["qualifying_friction"] == 0
    assert result["requires_protocol"] is False


def test_one_real_rejection_and_one_infrastructure_error_are_not_repeated_friction():
    world = _world()
    world.patches = {
        "infra": SimpleNamespace(
            validation_status="rejected", rejection_reason="code_editor_llm_returned_nothing",
        ),
        "product": SimpleNamespace(validation_status="rejected", rejection_reason="invalid patch"),
    }
    result = protocol_realization(world)
    assert result["friction_evidence"] == ["patch_rejected:product"]
    assert result["requires_protocol"] is False


def test_infrastructure_filter_preserves_real_ci_and_peer_review_friction():
    world = _world()
    world.patches = {
        "infra": SimpleNamespace(
            validation_status="rejected", rejection_reason="file_exceeds_edit_budget",
        ),
    }
    world.repo_system.repo.ci_runs = {"ci": SimpleNamespace(status="failed")}
    world.repo_system.repo.pull_requests = {
        "pr": SimpleNamespace(requested_changes=["public behavior is broken"]),
    }
    result = protocol_realization(world)
    assert result["friction_evidence"] == ["ci_failed:ci", "review_changes:pr:1"]
    assert result["requires_protocol"] is True


def test_repeated_friction_requires_an_operative_adopted_protocol():
    world = _world()
    initialize_two_person_sdl(
        world,
        {
            "cooper_feature_1": "victor",
            "cooper_feature_2": "calvin",
        },
    )
    world.patches = {
        "p1": SimpleNamespace(validation_status="rejected"),
        "p2": SimpleNamespace(validation_status="rejected"),
    }

    missing = protocol_realization(world)
    assert missing["qualifying_friction"] == 2
    assert missing["operative_protocols"] == 0
    assert missing["realized_b3"] is False

    world.proposal_manager.protocol_specs["protocol_1"] = SimpleNamespace(
        protocol_id="protocol_1",
        name="Peer integration verification protocol",
        status="adopted",
        use_count=1,
        enforcement_count=0,
        source_episode_ids=["debug_1", "debug_2"],
        affected_actions=["verify_joint_delivery"],
    )
    # A generic keyword-based counter is not pair-action evidence.  This was
    # the measured false positive: CI attempts incremented use_count even when
    # no joint verification had happened.
    still_missing = protocol_realization(world)
    assert still_missing["operative_protocols"] == 0
    assert still_missing["realized_b3"] is False

    world._cooperbench_sdl_state["attestations"]["victor"] = {
        "tick": 30,
        "digest": joint_delivery_digest(world),
        "available": True,
        "ok": True,
        "protocol_ids": ["protocol_1"],
    }
    realized = protocol_realization(world)
    assert realized["workflow_bound_protocols"] == 1
    assert realized["protocol_gate_satisfied"] is True
    assert realized["operative_protocols"] == 0
    assert realized["realized_b3"] is False
    assert realized["execution_evidence"] == {
        "protocol_1": ["joint_attestation:victor:30"]
    }


def test_pair_protocol_enforces_a_failed_joint_verification():
    world = _world()
    initialize_two_person_sdl(
        world,
        {
            "cooper_feature_1": "victor",
            "cooper_feature_2": "calvin",
        },
    )
    world.patches = {
        "p1": SimpleNamespace(validation_status="rejected"),
        "p2": SimpleNamespace(validation_status="rejected"),
    }
    spec = SimpleNamespace(
        protocol_id="protocol_1",
        status="adopted",
        family="cooperbench_joint_delivery",
        affected_actions=["verify_joint_delivery"],
        source_episode_ids=["debug_1", "debug_2"],
        problem_evidence=[],
        enforcement_count=0,
        violation_count=0,
        enforcement_event_ids=[],
        violation_event_ids=[],
    )
    world.proposal_manager.protocol_specs[spec.protocol_id] = spec

    assert record_joint_attestation(
        world,
        "victor",
        digest=joint_delivery_digest(world),
        verification={"ok": False, "available": True, "summary": "1 failed"},
        tick=30,
    ) is False

    realized = protocol_realization(world)
    assert spec.enforcement_count == 1
    assert realized["workflow_bound_protocols"] == 1
    assert realized["operative_protocols"] == 0
    assert realized["execution_evidence"]["protocol_1"] == [
        "cooperbench_protocol_enforcement@t30:victor:public_regression_failed"
    ]
    incident = [
        event
        for event in world.events
        if event.get("protocol_id") == "protocol_1"
        and event.get("governed_action") == "verify_joint_delivery"
    ]
    assert [event["type"] for event in incident] == [
        "protocol_violation_event",
        "protocol_enforcement_event",
    ]
    assert incident[1]["blocked"] is True


def test_trajectory_recorder_is_incremental_bounded_and_redacts_secrets(tmp_path: Path):
    world = _world()
    initialize_two_person_sdl(
        world,
        {
            "cooper_feature_1": "victor",
            "cooper_feature_2": "calvin",
        },
    )
    world.action_decisions.append(
        {
            "agent_id": "victor",
            "action_type": "edit_repo_file",
            "accepted": True,
            "api_key": "must-not-survive",
            "rationale": "x" * 5000,
        }
    )
    world.action_log.append({"action_type": "edit_repo_file", "agent_id": "victor"})
    world.events.append({"type": "repo_event", "subtype": "patch_accepted"})
    world.repo_system.repo.ci_runs["ci_1"] = SimpleNamespace(
        status="failed",
        pr_id="pr_1",
        failure_reasons=["Authorization: Bearer sk-proj-ABCDEFGHIJKLMNOP"],
    )
    path = tmp_path / "trajectory.jsonl"
    recorder = TrajectoryRecorder(path)

    recorder.capture(world)
    recorder.capture(world)

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert len(rows[0]["decisions"]) == 1
    assert rows[1]["decisions"] == []
    assert rows[0]["feature_delivery"]["cooper_feature_1"] == {
        "owner": "victor",
        "required_paths": ["src/one.py"],
        "covered_paths": [],
        "missing_paths": ["src/one.py"],
        "complete": False,
    }
    raw = path.read_text(encoding="utf-8")
    assert "must-not-survive" not in raw
    assert "ABCDEFGHIJKLMNOP" not in raw
    assert "[REDACTED]" in raw
    assert "[redacted-secret]" in raw
    assert len(rows[0]["decisions"][0]["rationale"]) < 1200


def test_sdl_affordance_offers_pair_verification_only_after_peer_merges():
    world = _world()
    initialize_two_person_sdl(
        world,
        {
            "cooper_feature_1": "victor",
            "cooper_feature_2": "calvin",
        },
    )
    world.repo_system.repo.pull_requests = {
        "pr_1": SimpleNamespace(
            status="merged",
            author_id="victor",
            approved_by=["calvin"],
            linked_issue_ids=["cooper_feature_1"],
            linked_issue=None,
            requested_changes=[],
        ),
        "pr_2": SimpleNamespace(
            status="merged",
            author_id="calvin",
            approved_by=["victor"],
            linked_issue_ids=["cooper_feature_2"],
            linked_issue=None,
            requested_changes=[],
        ),
    }

    candidates = OrgActionMapper._cooperbench_sdl_driven(
        SimpleNamespace(id="victor"), world
    )

    assert [item.action_type for item in candidates] == ["verify_joint_delivery"]
    assert candidates[0].parameters["digest"] == joint_delivery_digest(world)


def test_formed_main_b3_follow_action_is_generated_and_prioritized(monkeypatch):
    world = _world()
    initialize_two_person_sdl(
        world,
        {
            "cooper_feature_1": "victor",
            "cooper_feature_2": "calvin",
        },
    )
    world._cooperbench_delivery_focus = True
    world._cooperbench_main_b3_lifecycle = True
    world._cooperbench_feature_issue_by_agent = {
        "victor": "cooper_feature_1",
        "calvin": "cooper_feature_2",
    }
    spec = SimpleNamespace(
        protocol_id="protospec_follow",
        affected_actions=["propose_protocol", "follow_protocol"],
    )
    mirror_id = "proto_spec_follow"
    world._registry_mirror_id = lambda _protocol_id: mirror_id
    world.protocol_registry = SimpleNamespace(
        protocols={
            mirror_id: SimpleNamespace(
                status="active",
                adoption_status="adopted",
            )
        }
    )
    monkeypatch.setattr(
        "environments.org_env.cooperbench.lifecycle.peer_review_coverage",
        lambda _world: {"cooper_feature_1": True, "cooper_feature_2": True},
    )
    monkeypatch.setattr(
        "environments.org_env.cooperbench.lifecycle.formed_main_b3_protocols",
        lambda _world: [spec],
    )
    monkeypatch.setattr(
        "environments.org_env.cooperbench.lifecycle.protocol_realization",
        lambda _world: {
            "protocol_formation_satisfied": True,
            "main_b3_action_bound_protocol_ids": [],
            "friction_evidence": [],
        },
    )

    generated = OrgActionMapper._cooperbench_sdl_driven(
        SimpleNamespace(id="victor"), world
    )
    follow = next(item for item in generated if item.action_type == "follow_protocol")
    assert follow.parameters["protocol_id"] == mirror_id
    assert follow.parameters["_cooperbench_main_b3_protocol_id"] == spec.protocol_id

    focused = _cooperbench_focus_delivery_candidates(
        world,
        [
            ActionCandidate(
                action_type="work_on_task",
                parameters={"task_id": "task_oss_cooper_feature_1"},
            ),
            ActionCandidate(action_type="run_public_tests", parameters={}),
            *generated,
        ],
        "victor",
    )
    assert [item.action_type for item in focused] == ["follow_protocol"]


def test_main_b3_realization_reopens_attestations_that_predate_its_first_use(
    monkeypatch,
):
    world = _world()
    initialize_two_person_sdl(
        world,
        {
            "cooper_feature_1": "victor",
            "cooper_feature_2": "calvin",
        },
    )
    world._cooperbench_delivery_focus = True
    world._cooperbench_main_b3_lifecycle = True
    digest = joint_delivery_digest(world)
    world._cooperbench_sdl_state["attestations"] = {
        "calvin": {"digest": digest, "tick": 130},
        "victor": {"digest": digest, "tick": 146},
    }
    world.action_log = [
        {
            "action_id": "act_calvin_151_work_on_task",
            "agent_id": "calvin",
            "action_type": "work_on_task",
            "tick": 151,
            "success": True,
        }
    ]
    spec = SimpleNamespace(
        protocol_id="protospec_work",
        affected_actions=["work_on_task"],
        adopted_at_tick=123,
    )
    monkeypatch.setattr(
        "environments.org_env.cooperbench.lifecycle.peer_review_coverage",
        lambda _world: {"cooper_feature_1": True, "cooper_feature_2": True},
    )
    monkeypatch.setattr(
        "environments.org_env.cooperbench.lifecycle.formed_main_b3_protocols",
        lambda _world: [spec],
    )
    monkeypatch.setattr(
        "environments.org_env.cooperbench.lifecycle.protocol_proposal_needed",
        lambda _world, _agent_id: False,
    )
    monkeypatch.setattr(
        "environments.org_env.cooperbench.lifecycle.protocol_realization",
        lambda _world: {
            "main_b3_action_bound_protocol_ids": [spec.protocol_id],
            "execution_evidence": {
                spec.protocol_id: ["act_calvin_151_work_on_task"]
            },
        },
    )

    victor = OrgActionMapper._cooperbench_sdl_driven(
        SimpleNamespace(id="victor"), world
    )
    assert [item.action_type for item in victor] == ["verify_joint_delivery"]

    world._cooperbench_sdl_state["attestations"]["victor"]["tick"] = 152
    assert OrgActionMapper._cooperbench_sdl_driven(
        SimpleNamespace(id="victor"), world
    ) == []
    calvin = OrgActionMapper._cooperbench_sdl_driven(
        SimpleNamespace(id="calvin"), world
    )
    assert [item.action_type for item in calvin] == ["verify_joint_delivery"]

    world._cooperbench_sdl_state["attestations"]["calvin"]["tick"] = 153
    assert OrgActionMapper._cooperbench_sdl_driven(
        SimpleNamespace(id="calvin"), world
    ) == []


def test_sdl_affordance_routes_repeated_friction_through_governance_first():
    world = _world()
    initialize_two_person_sdl(
        world,
        {
            "cooper_feature_1": "victor",
            "cooper_feature_2": "calvin",
        },
    )
    world.patches = {
        "p1": SimpleNamespace(validation_status="rejected"),
        "p2": SimpleNamespace(validation_status="rejected"),
    }

    candidates = OrgActionMapper._cooperbench_sdl_driven(
        SimpleNamespace(id="victor"), world
    )

    assert [item.action_type for item in candidates] == ["propose_protocol"]
    assert candidates[0].parameters["cooperbench_joint_delivery"] is True


def test_clean_main_b3_pair_delivery_routes_a_grounded_protocol_action(monkeypatch):
    from environments.org_env.cooperbench.lifecycle import (
        friction_evidence,
        protocol_deliberation_evidence,
        protocol_realization,
    )

    world = _world()
    initialize_two_person_sdl(
        world,
        {
            "cooper_feature_1": "victor",
            "cooper_feature_2": "calvin",
        },
    )
    world._cooperbench_delivery_focus = True
    world._cooperbench_main_b3_lifecycle = True
    world.repo_system.repo.pull_requests = {
        "pr_1": SimpleNamespace(
            status="merged", author_id="victor", approved_by=["calvin"],
            linked_issue_ids=["cooper_feature_1"], linked_issue=None,
            ci_passed=True, merged_tick=30, requested_changes=[],
        ),
        "pr_2": SimpleNamespace(
            status="merged", author_id="calvin", approved_by=["victor"],
            linked_issue_ids=["cooper_feature_2"], linked_issue=None,
            ci_passed=True, merged_tick=79, requested_changes=[],
        ),
    }
    monkeypatch.setattr(
        "environments.org_env.cooperbench.lifecycle.peer_review_coverage",
        lambda _world: {"cooper_feature_1": True, "cooper_feature_2": True},
    )

    evidence = protocol_deliberation_evidence(world)
    candidates = OrgActionMapper._cooperbench_sdl_driven(
        SimpleNamespace(id="victor"), world
    )

    assert friction_evidence(world) == []
    assert evidence == [
        "peer_reviewed_delivery:cooper_feature_1",
        "peer_reviewed_delivery:cooper_feature_2",
    ]
    assert protocol_realization(world)["protocol_deliberation_required"] is True
    assert [item.action_type for item in candidates] == [
        "propose_protocol", "verify_joint_delivery",
    ]
    assert candidates[0].parameters["source_event_ids"] == evidence

    world._cooperbench_sdl_state["attestations"]["victor"] = {
        "digest": joint_delivery_digest(world), "tick": 80,
    }
    after_attestation = OrgActionMapper._cooperbench_sdl_driven(
        SimpleNamespace(id="victor"), world
    )
    assert [item.action_type for item in after_attestation] == ["propose_protocol"]


def test_pair_focus_makes_missing_member_verification_the_only_ready_action():
    world = _world()
    initialize_two_person_sdl(
        world,
        {
            "cooper_feature_1": "victor",
            "cooper_feature_2": "calvin",
        },
    )
    world._cooperbench_delivery_focus = True
    world._cooperbench_feature_issue_by_agent = {
        "victor": "cooper_feature_1",
        "calvin": "cooper_feature_2",
    }
    world.repo_system.repo.pull_requests = {
        "pr_1": SimpleNamespace(
            status="merged",
            author_id="victor",
            approved_by=["calvin"],
            linked_issue_ids=["cooper_feature_1"],
            linked_issue=None,
            requested_changes=[],
        ),
        "pr_2": SimpleNamespace(
            status="merged",
            author_id="calvin",
            approved_by=["victor"],
            linked_issue_ids=["cooper_feature_2"],
            linked_issue=None,
            requested_changes=[],
        ),
    }
    world._cooperbench_semantic_reviews = {}
    for pr_id, pr in world.repo_system.repo.pull_requests.items():
        pr.pr_id = pr_id
        pr.commit_ids = [f"commit_{pr_id}"]
        pr.patch_ids = [f"patch_{pr_id}"]
        feature_id = pr.linked_issue_ids[0]
        world._cooperbench_semantic_reviews[feature_id] = {
            "approved": True, "pr_id": pr_id, "reviewer_id": pr.approved_by[0],
            "reviewed_pr_revision": pull_request_revision(pr),
        }
    verify = OrgActionMapper._cooperbench_sdl_driven(
        SimpleNamespace(id="calvin"), world
    )[0]
    focused = _cooperbench_focus_delivery_candidates(
        world,
        [
            ActionCandidate(
                action_type="work_on_task",
                parameters={"task_id": "task_oss_cooper_feature_2"},
            ),
            ActionCandidate(action_type="run_public_tests", parameters={}),
            verify,
        ],
        "calvin",
    )

    assert [item.action_type for item in focused] == [
        "verify_joint_delivery"
    ]


def test_frozen_sdl_blocks_late_product_mutation():
    world = make_world()
    world.__dict__["_cooperbench_sdl_state"] = {"phase": "frozen"}

    result = world._loop["execution"].execute(
        "sean",
        ActionCandidate(action_type="edit_repo_file", parameters={}),
        world,
    )

    assert result.success is False
    assert result.failure_reason == "cooperbench_joint_delivery_frozen"


def test_pair_focus_allows_only_the_owners_ready_feature_surface():
    world = _world()
    world.product_artifacts.update(
        {
            "shared": SimpleNamespace(linked_file_path="src/shared.py"),
            "unrelated": SimpleNamespace(linked_file_path="src/unrelated.py"),
        }
    )
    world._oss_component_map = {
        "cooper_feature_1": ["src/shared.py"],
        "cooper_feature_2": ["src/shared.py"],
    }
    initialize_two_person_sdl(
        world,
        {
            "cooper_feature_1": "victor",
            "cooper_feature_2": "calvin",
        },
    )
    world._cooperbench_delivery_focus = True
    world._cooperbench_feature_issue_by_agent = {
        "victor": "cooper_feature_1",
        "calvin": "cooper_feature_2",
    }
    candidates = [
        ActionCandidate(
            action_type="edit_repo_file",
            parameters={
                "artifact_id": "shared",
                "file_path": "src/shared.py",
                "_oss_issue": "cooper_feature_2",
            },
        ),
        ActionCandidate(
            action_type="edit_repo_file",
            parameters={
                "artifact_id": "unrelated",
                "file_path": "src/unrelated.py",
            },
        ),
        ActionCandidate(
            action_type="work_on_task",
            parameters={"task_id": "task_oss_cooper_feature_2"},
        ),
    ]

    # Calvin's isolated desk can implement the assigned shared path immediately;
    # the unrelated generic edit remains outside the delivery surface.
    ready = _cooperbench_focus_delivery_candidates(
        world, candidates, "calvin"
    )
    assert len(ready) == 1
    assert ready[0].action_type == "edit_repo_file"
    assert ready[0].parameters["file_path"] == "src/shared.py"


def test_pair_execution_blocks_out_of_surface_edits_and_self_review():
    world = _world()
    world.product_artifacts["one"].artifact_id = "one"
    initialize_two_person_sdl(
        world,
        {
            "cooper_feature_1": "victor",
            "cooper_feature_2": "calvin",
        },
    )
    world._cooperbench_feature_issue_by_agent = {
        "victor": "cooper_feature_1",
        "calvin": "cooper_feature_2",
    }
    assert _cooperbench_action_block_reason(
        world,
        "victor",
        "edit_repo_file",
        {
            "artifact_id": "two",
            "file_path": "src/two.py",
            "_oss_issue": "cooper_feature_1",
        },
    ) == "edit_outside_assigned_feature_surface"
    assert _cooperbench_action_block_reason(
        world,
        "victor",
        "edit_repo_file",
        {
            "artifact_id": "one",
            "file_path": "src/one.py",
            "_oss_issue": "unrelated_issue",
        },
    ) == "edit_requires_exact_assigned_feature_linkage"

    world.repo_system.repo.pull_requests["pr_1"] = SimpleNamespace(
        author_id="victor"
    )
    assert _cooperbench_action_block_reason(
        world,
        "victor",
        "approve_pr",
        {"pr_id": "pr_1"},
    ) == "feature_pr_requires_peer_review"


def test_pair_execution_requires_exact_pr_provenance_peer_and_green_ci():
    world = _world()
    initialize_two_person_sdl(
        world,
        {
            "cooper_feature_1": "victor",
            "cooper_feature_2": "calvin",
        },
    )
    world._cooperbench_feature_issue_by_agent = {
        "victor": "cooper_feature_1",
        "calvin": "cooper_feature_2",
    }
    world.repo_system.repo.commits = {
        "commit_1": SimpleNamespace(linked_issue_ids=[]),
        "commit_2": SimpleNamespace(
            linked_issue_ids=["cooper_feature_1"]
        ),
    }
    world.repo_system.repo.branches = {
        "branch_bad": SimpleNamespace(
            owner_id="victor", commit_ids=["commit_1"]
        ),
        "branch_good": SimpleNamespace(
            owner_id="victor", commit_ids=["commit_2"]
        ),
    }

    assert _cooperbench_action_block_reason(
        world,
        "victor",
        "open_pr",
        {"source_branch": "branch_bad", "reviewers": ["calvin"]},
    ) == "feature_pr_requires_exact_assigned_feature_linkage"
    assert _cooperbench_action_block_reason(
        world,
        "victor",
        "open_pr",
        {"source_branch": "branch_good", "reviewers": ["victor"]},
    ) == "feature_pr_requires_peer_reviewer"

    pr = SimpleNamespace(
        author_id="victor",
        linked_issue_ids=["cooper_feature_1"],
        linked_issue=None,
        approved_by=[],
        ci_passed=False,
    )
    world.repo_system.repo.pull_requests["pr_1"] = pr
    assert _cooperbench_action_block_reason(
        world, "victor", "merge_pr", {"pr_id": "pr_1"}
    ) == "feature_pr_requires_peer_approval_before_merge"
    pr.approved_by = ["calvin"]
    assert _cooperbench_action_block_reason(
        world, "victor", "merge_pr", {"pr_id": "pr_1"}
    ) == "feature_pr_requires_green_ci_before_merge"
    pr.ci_passed = True
    assert _cooperbench_action_block_reason(
        world, "victor", "merge_pr", {"pr_id": "pr_1"}
    ) == "feature_pr_requires_current_semantic_approval_before_merge"
    pr.pr_id = "pr_1"
    pr.commit_ids = ["commit_2"]
    pr.patch_ids = ["patch_1"]
    world._cooperbench_semantic_reviews = {"cooper_feature_1": {
        "approved": True, "pr_id": "pr_1", "reviewer_id": "calvin",
        "reviewed_pr_revision": pull_request_revision(pr)}}
    assert (
        _cooperbench_action_block_reason(
            world, "victor", "merge_pr", {"pr_id": "pr_1"}
        )
        is None
    )


def test_pair_review_waits_for_author_repair_and_fresh_public_ci_evidence():
    world = _world()
    for artifact_id, artifact in world.product_artifacts.items():
        artifact.artifact_id = artifact_id
        artifact.content = artifact.mainline_content
    initialize_two_person_sdl(
        world,
        {"cooper_feature_1": "victor", "cooper_feature_2": "calvin"},
    )
    world._cooperbench_feature_issue_by_agent = {
        "victor": "cooper_feature_1",
        "calvin": "cooper_feature_2",
    }
    pr = SimpleNamespace(
        author_id="victor",
        status="changes_requested",
        ci_passed=True,
    )
    world.repo_system.repo.pull_requests["pr_1"] = pr

    assert _cooperbench_action_block_reason(
        world, "calvin", "review_pr", {"pr_id": "pr_1"}
    ) == "feature_pr_requires_author_repair_before_rereview"

    pr.status = "review_requested"
    assert _cooperbench_action_block_reason(
        world, "calvin", "review_pr", {"pr_id": "pr_1"}
    ) == "feature_pr_requires_current_green_public_tests_before_review"

    world.__dict__["_public_tests_last_hash"] = _repo_hash(
        world, prefer_mainline=False
    )
    world.__dict__["_public_tests_last"] = {"passed": True}
    pr.ci_passed = False
    assert _cooperbench_action_block_reason(
        world, "calvin", "review_pr", {"pr_id": "pr_1"}
    ) == "feature_pr_requires_current_green_ci_before_review"

    pr.ci_passed = True
    assert (
        _cooperbench_action_block_reason(
            world, "calvin", "review_pr", {"pr_id": "pr_1"}
        )
        is None
    )


def test_pair_owner_cannot_speculatively_edit_while_current_head_awaits_peer_verdict():
    """A same-tick peer verdict must precede any model-authored repair."""

    world = _world()
    for artifact_id, artifact in world.product_artifacts.items():
        artifact.artifact_id = artifact_id
    initialize_two_person_sdl(
        world,
        {
            "cooper_feature_1": "victor",
            "cooper_feature_2": "calvin",
        },
    )
    world._cooperbench_delivery_focus = True
    world._cooperbench_feature_issue_by_agent = {
        "victor": "cooper_feature_1",
        "calvin": "cooper_feature_2",
    }
    world.patches = {
        "patch_1": SimpleNamespace(
            actor_id="victor",
            validation_status="accepted",
            related_issue_ids=["cooper_feature_1"],
            target_object_id="one",
        )
    }
    world.repo_system.repo.pull_requests["pr_1"] = SimpleNamespace(
        pr_id="pr_1",
        author_id="victor",
        status="review_requested",
        linked_issue_ids=["cooper_feature_1"],
        linked_issue=None,
    )
    generic_edit = ActionCandidate(
        action_type="edit_repo_file",
        parameters={
            "artifact_id": "one",
            "_oss_issue": "cooper_feature_1",
        },
    )
    stale_effort = ActionCandidate(
        action_type="work_on_task",
        parameters={"task_id": "task_oss_cooper_feature_1"},
    )

    assert _cooperbench_feature_edit_candidates(world, "victor") == []
    assert _cooperbench_action_block_reason(
        world, "victor", generic_edit.action_type, generic_edit.parameters
    ) == "assigned_feature_current_head_awaits_peer_verdict"
    assert _cooperbench_focus_delivery_candidates(
        world, [generic_edit, stale_effort], "victor"
    ) == []

    # A concrete already-existing repair receipt remains actionable.  The gate
    # only removes edits that were selected before the peer supplied evidence.
    repair = dict(generic_edit.parameters)
    repair["_cooperbench_repair_source"] = "semantic_peer_review"
    assert (
        _cooperbench_action_block_reason(
            world, "victor", "edit_repo_file", repair
        )
        is None
    )


def test_pair_focus_removes_invalid_delivery_candidates_before_policy_choice():
    world = _world()
    initialize_two_person_sdl(
        world,
        {
            "cooper_feature_1": "victor",
            "cooper_feature_2": "calvin",
        },
    )
    world._cooperbench_delivery_focus = True
    world._cooperbench_feature_issue_by_agent = {
        "victor": "cooper_feature_1",
        "calvin": "cooper_feature_2",
    }
    world.repo_system.repo.pull_requests["mine"] = SimpleNamespace(
        author_id="victor"
    )
    candidates = [
        ActionCandidate(
            action_type="approve_pr", parameters={"pr_id": "mine"}
        ),
        ActionCandidate(
            action_type="work_on_task",
            parameters={"task_id": "task_oss_cooper_feature_1"},
        ),
    ]

    focused = _cooperbench_focus_delivery_candidates(
        world, candidates, "victor"
    )

    assert [item.action_type for item in focused] == ["edit_repo_file"]
    assert focused[0].parameters["file_path"] == "src/one.py"


def test_pair_focus_keeps_only_the_pair_grounded_governance():
    world = _world()
    initialize_two_person_sdl(
        world,
        {
            "cooper_feature_1": "victor",
            "cooper_feature_2": "calvin",
        },
    )
    world._cooperbench_delivery_focus = True
    world._cooperbench_feature_issue_by_agent = {
        "victor": "cooper_feature_1",
        "calvin": "cooper_feature_2",
    }
    world.proposal_manager.proposals = {
        "pair": SimpleNamespace(
            proposal_id="pair",
            family="cooperbench_joint_delivery",
            status="under_review",
            proposer_agent_id="victor",
        ),
        "generic": SimpleNamespace(
            proposal_id="generic",
            family="ordinary_process",
            status="under_review",
        ),
    }
    candidates = [
        ActionCandidate(
            action_type="approve_proposal",
            parameters={"proposal_id": "generic"},
        ),
        ActionCandidate(
            action_type="approve_proposal",
            parameters={"proposal_id": "pair"},
        ),
        ActionCandidate(
            action_type="reject_proposal",
            parameters={"proposal_id": "pair"},
        ),
        ActionCandidate(
            action_type="request_proposal_changes",
            parameters={"proposal_id": "pair"},
        ),
    ]

    focused = _cooperbench_focus_delivery_candidates(
        world, candidates, "victor"
    )

    assert [item.action_type for item in focused] == [
        "edit_repo_file",
        "approve_proposal",
    ]
    assert focused[1].parameters["proposal_id"] == "pair"


def test_pair_focus_keeps_main_b3_protocol_approval_reachable():
    world = _world()
    initialize_two_person_sdl(
        world,
        {
            "cooper_feature_1": "victor",
            "cooper_feature_2": "calvin",
        },
    )
    world._cooperbench_delivery_focus = True
    world._cooperbench_main_b3_lifecycle = True
    world._cooperbench_feature_issue_by_agent = {
        "victor": "cooper_feature_1",
        "calvin": "cooper_feature_2",
    }
    world.proposal_manager.proposals = {
        "main_b3": SimpleNamespace(
            proposal_id="main_b3",
            proposal_type="protocol_proposal",
            family="review_merge",
            status="under_review",
            proposer_agent_id="victor",
            approval_required_from=["calvin", "victor"],
        ),
        "generic_tool": SimpleNamespace(
            proposal_id="generic_tool",
            proposal_type="tool_proposal",
            family="review_merge",
            status="under_review",
            proposer_agent_id="victor",
            approval_required_from=["calvin", "victor"],
        ),
    }
    candidates = [
        ActionCandidate(
            action_type="approve_proposal",
            parameters={"proposal_id": "main_b3"},
        ),
        ActionCandidate(
            action_type="approve_proposal",
            parameters={"proposal_id": "generic_tool"},
        ),
    ]

    focused = _cooperbench_focus_delivery_candidates(
        world, candidates, "victor"
    )

    assert [item.action_type for item in focused] == [
        "edit_repo_file",
        "approve_proposal",
    ]
    assert focused[1].parameters["proposal_id"] == "main_b3"


def test_main_b3_governance_protocol_is_not_starved_by_older_workflows():
    world = _world()
    world.approval_mode = "agent"
    world._cooperbench_main_b3_lifecycle = True
    world.proposal_manager = ProposalManager()

    def pending(proposal_id: str, proposal_type: str) -> SimpleNamespace:
        return SimpleNamespace(
            proposal_id=proposal_id,
            proposal_type=proposal_type,
            status="under_review",
            approval_required_from=["victor", "calvin"],
            approved_by=[],
            rejected_by=[],
        )

    world.proposal_manager.proposals = {
        "workflow_old_1": pending("workflow_old_1", "workflow_proposal"),
        "workflow_old_2": pending("workflow_old_2", "workflow_proposal"),
        "protocol_later": pending("protocol_later", "protocol_proposal"),
    }

    candidates = OrgActionMapper()._governance_driven(
        SimpleNamespace(id="victor"), world
    )

    assert [item.parameters["proposal_id"] for item in candidates[:3]] == [
        "protocol_later",
        "protocol_later",
        "protocol_later",
    ]
    assert {item.action_type for item in candidates[:3]} == {
        "approve_proposal",
        "reject_proposal",
        "request_proposal_changes",
    }


def test_candidate_dedupe_keeps_distinct_proposal_decisions():
    first = ActionCandidate(
        action_type="approve_proposal",
        parameters={"proposal_id": "protocol_one"},
    )
    second = ActionCandidate(
        action_type="approve_proposal",
        parameters={"proposal_id": "protocol_two"},
    )
    duplicate = ActionCandidate(
        action_type="approve_proposal",
        parameters={"proposal_id": "protocol_one"},
    )

    assert _candidate_dedupe_key(first) != _candidate_dedupe_key(second)
    assert _candidate_dedupe_key(first) == _candidate_dedupe_key(duplicate)


def test_delivered_feature_prioritizes_real_protocol_decision_over_bookkeeping():
    world = _world()
    initialize_two_person_sdl(
        world,
        {
            "cooper_feature_1": "victor",
            "cooper_feature_2": "calvin",
        },
    )
    world._cooperbench_delivery_focus = True
    world._cooperbench_main_b3_lifecycle = True
    world._cooperbench_feature_issue_by_agent = {
        "victor": "cooper_feature_1",
        "calvin": "cooper_feature_2",
    }
    world.repo_system.repo.pull_requests["pr_1"] = SimpleNamespace(
        status="merged",
        author_id="victor",
        approved_by=["calvin"],
        linked_issue_ids=["cooper_feature_1"],
        linked_issue=None,
    )
    proposal = SimpleNamespace(
        proposal_id="main_b3",
        proposal_type="protocol_proposal",
        family="review_merge",
        status="under_review",
        proposer_agent_id="calvin",
        approval_required_from=["calvin", "victor"],
    )
    world.proposal_manager.proposals = {proposal.proposal_id: proposal}
    candidates = [
        ActionCandidate(
            action_type="work_on_task",
            parameters={"task_id": "task_oss_cooper_feature_1"},
        ),
        ActionCandidate(
            action_type="approve_proposal",
            parameters={"proposal_id": proposal.proposal_id},
        ),
        ActionCandidate(
            action_type="reject_proposal",
            parameters={"proposal_id": proposal.proposal_id},
        ),
        ActionCandidate(
            action_type="request_proposal_changes",
            parameters={"proposal_id": proposal.proposal_id},
        ),
    ]

    focused = _cooperbench_focus_delivery_candidates(
        world, candidates, "victor"
    )

    assert [item.action_type for item in focused] == [
        "approve_proposal",
        "reject_proposal",
        "request_proposal_changes",
    ]
    assert all(
        item.parameters["proposal_id"] == proposal.proposal_id
        for item in focused
    )


def test_delivered_feature_exposes_live_action_declared_by_formed_protocol(
    monkeypatch,
):
    """A formed rule cannot become operative if its real action is filtered.

    The proposal being decided need not itself be a protocol proposal.  What
    makes the decision relevant is that the organization's formed ProtocolSpec
    explicitly declares the action and the object is genuinely pending.
    """

    world = _world()
    initialize_two_person_sdl(
        world,
        {
            "cooper_feature_1": "victor",
            "cooper_feature_2": "calvin",
        },
    )
    world._cooperbench_delivery_focus = True
    world._cooperbench_main_b3_lifecycle = True
    world._cooperbench_feature_issue_by_agent = {
        "victor": "cooper_feature_1",
        "calvin": "cooper_feature_2",
    }
    world.repo_system.repo.pull_requests["pr_1"] = SimpleNamespace(
        status="merged",
        author_id="victor",
        approved_by=["calvin"],
        linked_issue_ids=["cooper_feature_1"],
        linked_issue=None,
    )
    proposal = SimpleNamespace(
        proposal_id="tool_pending",
        proposal_type="tool_proposal",
        family="review_merge",
        status="under_review",
        proposer_agent_id="calvin",
        approval_required_from=["calvin", "victor"],
    )
    world.proposal_manager.proposals = {proposal.proposal_id: proposal}
    monkeypatch.setattr(
        "environments.org_env.cooperbench.lifecycle.formed_main_b3_protocols",
        lambda _world: [
            SimpleNamespace(
                protocol_id="protospec_1",
                affected_actions=["propose_protocol", "approve_proposal"],
            )
        ],
    )
    monkeypatch.setattr(
        "environments.org_env.cooperbench.lifecycle.protocol_realization",
        lambda _world: {"protocol_formation_satisfied": True},
    )
    candidates = [
        ActionCandidate(
            action_type="propose_protocol",
            parameters={"cooperbench_joint_delivery": True},
        ),
        ActionCandidate(
            action_type="work_on_task",
            parameters={"task_id": "task_oss_cooper_feature_1"},
        ),
        ActionCandidate(
            action_type="update_task_status",
            parameters={"task_id": "task_oss_cooper_feature_1"},
        ),
        ActionCandidate(
            action_type="approve_proposal",
            parameters={"proposal_id": proposal.proposal_id},
        ),
        ActionCandidate(
            action_type="reject_proposal",
            parameters={"proposal_id": proposal.proposal_id},
        ),
        ActionCandidate(
            action_type="request_proposal_changes",
            parameters={"proposal_id": proposal.proposal_id},
        ),
    ]

    focused = _cooperbench_focus_delivery_candidates(
        world, candidates, "victor"
    )

    assert [item.action_type for item in focused] == ["approve_proposal"]
    assert focused[0].parameters["proposal_id"] == proposal.proposal_id


def test_undelivered_feature_precedes_live_action_from_formed_protocol(
    monkeypatch,
):
    """A pending product delivery cannot be starved by protocol governance."""

    world = _world()
    initialize_two_person_sdl(
        world,
        {
            "cooper_feature_1": "victor",
            "cooper_feature_2": "calvin",
        },
    )
    world._cooperbench_delivery_focus = True
    world._cooperbench_main_b3_lifecycle = True
    world._cooperbench_feature_issue_by_agent = {
        "victor": "cooper_feature_1",
        "calvin": "cooper_feature_2",
    }
    world.repo_system.repo.pull_requests["pr_stale"] = SimpleNamespace(
        status="stale",
        author_id="calvin",
        approved_by=[],
        linked_issue_ids=["cooper_feature_2"],
        linked_issue=None,
    )
    proposal = SimpleNamespace(
        proposal_id="tool_pending",
        proposal_type="tool_proposal",
        family="review_merge",
        status="under_review",
        proposer_agent_id="victor",
        approval_required_from=["calvin", "victor"],
    )
    world.proposal_manager.proposals = {proposal.proposal_id: proposal}
    monkeypatch.setattr(
        "environments.org_env.cooperbench.lifecycle.formed_main_b3_protocols",
        lambda _world: [
            SimpleNamespace(
                protocol_id="protospec_1",
                affected_actions=["approve_proposal"],
            )
        ],
    )
    monkeypatch.setattr(
        "environments.org_env.cooperbench.lifecycle.protocol_realization",
        lambda _world: {
            "protocol_formation_satisfied": True,
            "main_b3_action_bound_protocol_ids": [],
        },
    )
    monkeypatch.setattr(
        "environments.org_env.runtime_adapter.execution._cooperbench_action_block_reason",
        lambda *_args, **_kwargs: None,
    )
    candidates = [
        ActionCandidate(action_type="commit_patch", parameters={}),
        ActionCandidate(
            action_type="approve_proposal",
            parameters={"proposal_id": proposal.proposal_id},
        ),
    ]

    focused = _cooperbench_focus_delivery_candidates(
        world, candidates, "calvin"
    )

    assert [item.action_type for item in focused] == [
        "commit_patch",
        "approve_proposal",
    ]


@pytest.mark.parametrize(
    ("affected_actions", "expected"),
    [
        (["verify_joint_delivery"], ["verify_joint_delivery"]),
        (["run_ci", "review_pr"], ["verify_joint_delivery"]),
    ],
)
def test_technical_verification_is_reachable_independent_of_protocol_actions(
    monkeypatch, affected_actions, expected
):
    """Protocol action choice is measured without withholding task export."""
    world = _world()
    initialize_two_person_sdl(
        world,
        {
            "cooper_feature_1": "victor",
            "cooper_feature_2": "calvin",
        },
    )
    world._cooperbench_delivery_focus = True
    world._cooperbench_main_b3_lifecycle = True
    world._cooperbench_sdl_state["attestations"] = {}
    spec = SimpleNamespace(affected_actions=affected_actions)
    monkeypatch.setattr(
        "environments.org_env.cooperbench.lifecycle.peer_review_coverage",
        lambda _world: {"cooper_feature_1": True, "cooper_feature_2": True},
    )
    monkeypatch.setattr(
        "environments.org_env.cooperbench.lifecycle.protocol_realization",
        lambda _world: {
            "requires_protocol": True,
            "protocol_formation_satisfied": True,
            "protocol_gate_satisfied": False,
            "main_b3_lifecycle_enabled": True,
            "friction_evidence": [],
        },
    )
    monkeypatch.setattr(
        "environments.org_env.cooperbench.lifecycle.formed_main_b3_protocols",
        lambda _world: [spec],
    )

    candidates = OrgActionMapper._cooperbench_sdl_driven(
        SimpleNamespace(id="victor"), world
    )

    assert [candidate.action_type for candidate in candidates] == expected


def _pending_recurrent_main_b3_proposal():
    return SimpleNamespace(
        proposal_id="proposal_late",
        proposal_type="protocol_proposal",
        status="under_review",
        proposer_agent_id="victor",
        approval_required_from=["victor", "calvin"],
        source_episode_id="ep_review_2",
        source_episode_ids=["ep_review_1", "ep_review_2"],
        source_event_ids=["ep_review_1", "ep_review_2"],
        source_wish_id=None,
        source_wish_ids=[],
        source_wish_supporter_ids=[],
    )


def _ready_feature_pr(pr_id, author, feature, peer, status):
    return SimpleNamespace(
        pr_id=pr_id,
        status=status,
        author_id=author,
        approved_by=[peer],
        linked_issue_ids=[feature],
        linked_issue=None,
        requested_changes=[],
        ci_passed=True,
    )


def test_final_feature_merge_does_not_wait_for_recurrent_main_b3_decision(monkeypatch):
    world = _world()
    initialize_two_person_sdl(
        world,
        {
            "cooper_feature_1": "victor",
            "cooper_feature_2": "calvin",
        },
    )
    world._cooperbench_delivery_focus = True
    world._cooperbench_main_b3_lifecycle = True
    world._cooperbench_feature_issue_by_agent = {
        "victor": "cooper_feature_1",
        "calvin": "cooper_feature_2",
    }
    world.repo_system.repo.pull_requests = {
        "pr_1": _ready_feature_pr(
            "pr_1", "victor", "cooper_feature_1", "calvin", "merged"
        ),
        "pr_2": _ready_feature_pr(
            "pr_2", "calvin", "cooper_feature_2", "victor", "approved"
        ),
    }
    proposal = _pending_recurrent_main_b3_proposal()
    world.proposal_manager.proposals = {proposal.proposal_id: proposal}
    monkeypatch.setattr(
        "environments.org_env.cooperbench.semantic_review.current_semantic_approval",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        "environments.org_env.cooperbench.replay_workflow.integration_pass_matches",
        lambda *_args, **_kwargs: True,
    )

    assert pending_main_b3_protocols(world) == [proposal]
    assert _cooperbench_action_block_reason(
        world, "calvin", "merge_pr", {"pr_id": "pr_2"}
    ) is None

    proposal.status = "rejected"
    assert _cooperbench_action_block_reason(
        world, "calvin", "merge_pr", {"pr_id": "pr_2"}
    ) is None


def test_nonfinal_merge_and_shallow_protocol_do_not_wait(monkeypatch):
    world = _world()
    initialize_two_person_sdl(
        world,
        {
            "cooper_feature_1": "victor",
            "cooper_feature_2": "calvin",
        },
    )
    world._cooperbench_delivery_focus = True
    world._cooperbench_main_b3_lifecycle = True
    world._cooperbench_feature_issue_by_agent = {
        "victor": "cooper_feature_1",
        "calvin": "cooper_feature_2",
    }
    world.repo_system.repo.pull_requests = {
        "pr_1": _ready_feature_pr(
            "pr_1", "victor", "cooper_feature_1", "calvin", "approved"
        )
    }
    proposal = _pending_recurrent_main_b3_proposal()
    world.proposal_manager.proposals = {proposal.proposal_id: proposal}
    monkeypatch.setattr(
        "environments.org_env.cooperbench.semantic_review.current_semantic_approval",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        "environments.org_env.cooperbench.replay_workflow.integration_pass_matches",
        lambda *_args, **_kwargs: True,
    )

    assert _cooperbench_action_block_reason(
        world, "victor", "merge_pr", {"pr_id": "pr_1"}
    ) is None

    world.repo_system.repo.pull_requests["pr_2"] = _ready_feature_pr(
        "pr_2", "calvin", "cooper_feature_2", "victor", "approved"
    )
    proposal.source_episode_ids = ["ep_review_1"]
    proposal.source_episode_id = "ep_review_1"
    proposal.source_event_ids = ["ep_review_1"]
    assert pending_main_b3_protocols(world) == []
    assert _cooperbench_action_block_reason(
        world, "calvin", "merge_pr", {"pr_id": "pr_2"}
    ) is None


def test_pair_owner_gets_every_named_file_before_peer_merge():
    world = _world()
    world.product_artifacts.update(
        {
            "shared_impl": SimpleNamespace(
                artifact_id="shared_impl",
                linked_file_path="src/shared_impl.py",
            ),
            "public_api": SimpleNamespace(
                artifact_id="public_api",
                linked_file_path="src/public_api.py",
            ),
        }
    )
    world._oss_component_map = {
        "cooper_feature_1": ["src/shared_impl.py", "src/public_api.py"],
        "cooper_feature_2": ["src/shared_impl.py", "src/public_api.py"],
    }
    world.tasks = {
        "task_oss_cooper_feature_1": SimpleNamespace(
            description="Files Modified: src/shared_impl.py, src/public_api.py"
        ),
        "task_oss_cooper_feature_2": SimpleNamespace(
            description="Files Modified: src/shared_impl.py, src/public_api.py"
        ),
    }
    initialize_two_person_sdl(
        world,
        {
            "cooper_feature_1": "victor",
            "cooper_feature_2": "calvin",
        },
    )
    world._cooperbench_feature_issue_by_agent = {
        "victor": "cooper_feature_1",
        "calvin": "cooper_feature_2",
    }

    # Actor desks isolate overlapping implementation; integration replay handles
    # the combined tree later.
    world._cooperbench_delivery_focus = True
    candidates = _cooperbench_feature_edit_candidates(world, "calvin")
    assert {item.parameters["file_path"] for item in candidates} == {
        "src/shared_impl.py",
        "src/public_api.py",
    }
    world.repo_system.repo.pull_requests["pr_1"] = SimpleNamespace(
        status="merged",
        author_id="victor",
        approved_by=["calvin"],
        linked_issue_ids=["cooper_feature_1"],
        linked_issue=None,
    )
    after_peer_merge = _cooperbench_feature_edit_candidates(world, "calvin")
    assert {item.parameters["file_path"] for item in after_peer_merge} == {
        "src/shared_impl.py",
        "src/public_api.py",
    }
    assert all(
        item.parameters["_oss_issue"] == "cooper_feature_2"
        for item in after_peer_merge
    )

    # A generic candidate for the same path but a different OSS issue must not
    # occupy the dedupe slot and starve the exact Cooper feature candidate.
    focused = _cooperbench_focus_delivery_candidates(
        world,
        [
            ActionCandidate(
                action_type="edit_repo_file",
                parameters={
                    "artifact_id": "shared_impl",
                    "file_path": "src/shared_impl.py",
                    "_oss_issue": "unrelated_issue",
                },
            )
        ],
        "calvin",
    )
    assert {item.parameters["file_path"] for item in focused} == {
        "src/shared_impl.py",
        "src/public_api.py",
    }
    assert {
        item.parameters["_oss_issue"] for item in focused
    } == {"cooper_feature_2"}

    world.patches["patch_impl"] = SimpleNamespace(
        actor_id="calvin",
        patch_type="code_patch",
        validation_status="accepted",
        related_issue_ids=["cooper_feature_2"],
        target_object_id="shared_impl",
    )
    assert _cooperbench_missing_feature_paths(world, "calvin") == [
        "src/public_api.py"
    ]
    assert _cooperbench_action_block_reason(
        world, "calvin", "commit_patch", {}
    ) == "assigned_feature_required_paths_incomplete"
    remaining = _cooperbench_feature_edit_candidates(world, "calvin")
    assert [item.parameters["file_path"] for item in remaining] == [
        "src/public_api.py"
    ]
    world.repo_system.repo.branches = {
        "branch_calvin": SimpleNamespace(
            branch_id="branch_calvin",
            owner_id="calvin",
            status="active",
        )
    }
    world.__dict__["_pending_by_branch"] = {
        "branch_calvin": [("patch_impl", "shared_impl")]
    }
    assert AttractorGuard().mask_reason(
        remaining[0], "calvin", world, 24
    ) is None
    generic_second_edit = ActionCandidate(
        action_type="edit_repo_file",
        parameters={
            "artifact_id": "public_api",
            "file_path": "src/public_api.py",
            "_oss_issue": "cooper_feature_2",
            "_blocker_fix": True,
        },
    )
    assert "accepted code patch" in str(
        AttractorGuard().mask_reason(
            generic_second_edit, "calvin", world, 24
        )
    )

    world.patches["patch_api"] = SimpleNamespace(
        actor_id="calvin",
        validation_status="accepted",
        related_issue_ids=["cooper_feature_2"],
        target_object_id="public_api",
    )
    assert _cooperbench_missing_feature_paths(world, "calvin") == []
    assert feature_delivery_coverage(world)["cooper_feature_2"] == {
        "owner": "calvin",
        "required_paths": ["src/shared_impl.py", "src/public_api.py"],
        "covered_paths": ["src/shared_impl.py", "src/public_api.py"],
        "missing_paths": [],
        "complete": True,
    }
    assert _cooperbench_action_block_reason(
        world, "calvin", "commit_patch", {}
    ) == "assigned_feature_requires_current_green_public_tests"
    world.__dict__["_oss_time_machine_eval"] = {
        "manifest": {
            "public_tests": {
                "command": ["python", "-m", "pytest", "tests/test_public_api.py"]
            }
        }
    }
    assert OrgActionMapper._public_tests_worth_running(world) is True
    focused = _cooperbench_focus_delivery_candidates(
        world,
        [
            ActionCandidate(
                action_type="edit_repo_file",
                parameters={
                    "artifact_id": "public_api",
                    "file_path": "src/public_api.py",
                    "_oss_issue": "cooper_feature_2",
                    "_blocker_fix": True,
                },
            ),
            ActionCandidate(action_type="commit_patch", parameters={}),
            ActionCandidate(action_type="run_public_tests", parameters={}),
        ],
        "calvin",
    )
    assert [item.action_type for item in focused] == ["run_public_tests"]

    current_hash = _repo_hash(world, prefer_mainline=False)
    world.__dict__["_public_tests_last_hash"] = current_hash
    world.__dict__["_public_tests_last"] = {
        "tick": 24,
        "passed": False,
        "failed_tests": ["tests/test_public_api.py"],
        "summary": "1 failed",
        "failure_brief": (
            "src/public_api.py:27: NameError: name '_null_context' is not defined"
        ),
    }
    assert OrgActionMapper._public_tests_worth_running(world) is False
    repairs = _cooperbench_feature_edit_candidates(world, "calvin")
    assert {item.parameters["file_path"] for item in repairs} == {
        "src/public_api.py",
    }
    assert all(
        item.parameters["_oss_issue"] == "cooper_feature_2"
        and item.parameters["_cooperbench_required_path"] is True
        and "current public suite failed" in item.parameters["rationale"]
        and "_null_context" in item.parameters["edit_goal"]
        for item in repairs
    )
    focused_repairs = _cooperbench_focus_delivery_candidates(
        world,
        [
            ActionCandidate(
                action_type="edit_repo_file",
                parameters={
                    "artifact_id": "public_api",
                    "file_path": "src/public_api.py",
                    "_oss_issue": "cooper_feature_2",
                    "_blocker_fix": True,
                },
            )
        ],
        "calvin",
    )
    assert any(
        item.parameters.get("_cooperbench_required_path") is True
        and item.parameters.get("file_path") == "src/public_api.py"
        for item in focused_repairs
    )
    assert _cooperbench_action_block_reason(
        world, "calvin", "open_pr", {}
    ) == "assigned_feature_requires_current_green_public_tests"

    world.__dict__["_public_tests_last"] = {
        "tick": 25,
        "passed": True,
        "failed_tests": [],
        "summary": "all passed",
    }
    assert _cooperbench_feature_edit_candidates(world, "calvin") == []
    assert (
        _cooperbench_action_block_reason(
            world, "calvin", "commit_patch", {}
        )
        is None
    )


def test_pair_disables_the_generic_automatic_repo_workflow():
    world = _world()
    world._cooperbench_delivery_focus = True
    world.repo_system.repo.branches = {
        "branch_victor": SimpleNamespace(
            branch_id="branch_victor",
            owner_id="victor",
            status="ready_for_pr",
            commit_ids=["commit_1"],
        )
    }
    world.repo_system.repo.commits = {
        "commit_1": SimpleNamespace(timestamp=1)
    }

    assert OrgWorld.process_repo_workflow(world) == []
    assert world.repo_system.repo.pull_requests == {}


def test_candidate_collection_failure_cannot_borrow_prior_green_public_verdict(
    monkeypatch,
):
    import environments.org_env.product.materialize as materialize
    import environments.org_env.product.test_history as test_history

    world = _world()
    world._cooperbench_delivery_focus = True
    world._cooperbench_public_runtime_preflight = {"passed": True}
    world.__dict__["_public_tests_last_hash"] = "tree-before-edit"
    world.__dict__["_public_tests_last"] = {
        "tick": 4,
        "passed": True,
        "failed_tests": [],
        "summary": "4 passed",
    }
    monkeypatch.setattr(
        materialize,
        "_repo_hash",
        lambda _world, prefer_mainline=False: "tree-after-edit",
    )
    monkeypatch.setattr(
        materialize,
        "run_public_tests",
        lambda _world: {
            "ok": False,
            "available": True,
            "returncode": 4,
            "failed_tests": [],
            "summary": "",
            "error": (
                "public_test_harness_failed:ImportError while loading "
                "conftest '/workspace/tests/reliability/conftest.py'"
            ),
            "failure_brief": (
                "dspy/clients/cache.py:276: NameError: "
                "name '_null_context' is not defined"
            ),
        },
    )
    monkeypatch.setattr(
        test_history, "record_public_test_run", lambda *_args, **_kwargs: None
    )
    result = ExecutionResult(
        action_id="act_victor_8_run_public_tests",
        agent_id="victor",
        action_type="run_public_tests",
    )

    OrgExecutionAdapter()._h_run_public_tests(
        world, "victor", {}, result, tick=8
    )

    assert result.success is True
    assert result.state_delta["public_tests_passed"] is False
    assert world.__dict__["_public_tests_last_hash"] == "tree-after-edit"
    assert _cooperbench_current_public_test_result(world) is False
    assert world.__dict__["_public_tests_last"]["status"] == "failed"
    assert "_null_context" in world.__dict__["_public_tests_last"][
        "failure_brief"
    ]
    assert any(
        event.get("subtype") == "public_tests_failed"
        for event in result.events
    )
    assert not any(
        event.get("subtype") == "public_tests_infrastructure_error"
        for event in result.events
    )


def test_green_cooper_public_returncode_is_not_reclassified_as_failure(
    monkeypatch,
):
    import environments.org_env.product.materialize as materialize
    import environments.org_env.product.test_history as test_history

    world = _world()
    world._cooperbench_delivery_focus = True
    world._cooperbench_public_runtime_preflight = {"passed": True}
    monkeypatch.setattr(
        materialize,
        "_repo_hash",
        lambda _world, prefer_mainline=False: "green-tree",
    )
    monkeypatch.setattr(
        materialize,
        "run_public_tests",
        lambda _world: {
            "ok": True,
            "available": True,
            "returncode": 0,
            "failed_tests": [],
            "summary": "4 passed, 1 warning in 0.10s",
            "error": None,
            "failure_brief": "",
        },
    )
    monkeypatch.setattr(
        test_history, "record_public_test_run", lambda *_args, **_kwargs: None
    )
    result = ExecutionResult(
        action_id="act_victor_72_run_public_tests",
        agent_id="victor",
        action_type="run_public_tests",
    )

    OrgExecutionAdapter()._h_run_public_tests(
        world, "victor", {}, result, tick=72
    )

    assert result.success is True
    assert result.state_delta["public_tests_passed"] is True
    assert world.__dict__["_public_tests_last"]["status"] == "passed"
    assert any(
        event.get("subtype") == "public_tests_passed"
        for event in result.events
    )


def test_cooper_public_infrastructure_error_retries_after_bounded_cooldown(
    monkeypatch,
):
    import environments.org_env.product.materialize as materialize

    world = _world()
    for artifact_id, artifact in world.product_artifacts.items():
        artifact.artifact_id = artifact_id
        artifact.content = artifact.mainline_content
    initialize_two_person_sdl(
        world,
        {
            "cooper_feature_1": "victor",
            "cooper_feature_2": "calvin",
        },
    )
    world._cooperbench_delivery_focus = True
    world._cooperbench_feature_issue_by_agent = {
        "victor": "cooper_feature_1",
        "calvin": "cooper_feature_2",
    }
    world.patches = {
        "patch_one": SimpleNamespace(
            actor_id="victor",
            validation_status="accepted",
            related_issue_ids=["cooper_feature_1"],
            target_object_id="one",
        )
    }
    world._oss_time_machine_eval = {
        "manifest": {"public_tests": {"command": ["python", "-m", "pytest"]}}
    }
    monkeypatch.setattr(
        materialize,
        "declared_public_test_command",
        lambda _world: ["python", "-m", "pytest"],
    )
    monkeypatch.setattr(
        materialize,
        "_repo_hash",
        lambda _world, prefer_mainline=False: "tree-current",
    )
    world.__dict__["_public_tests_last_hash"] = "tree-current"
    world.__dict__["_public_tests_last"] = {
        "tick": 10,
        "status": "infrastructure_error",
        "failed_tests": [],
        "summary": "container temporarily unavailable",
    }
    world.__dict__["_public_test_cache"] = {
        "tree-current": {"error": "cached container outage"}
    }

    world.world_tick = 10 + COOPERBENCH_PUBLIC_INFRA_RETRY_COOLDOWN - 1
    assert OrgActionMapper._public_tests_worth_running(world) is False
    world.world_tick += 1
    assert OrgActionMapper._public_tests_worth_running(world) is True

    def retry_without_cached_outage(_world):
        assert "tree-current" not in _world.__dict__["_public_test_cache"]
        return {
            "ok": False,
            "available": True,
            "returncode": None,
            "failed_tests": [],
            "summary": "",
            "error": "fresh container outage",
        }

    monkeypatch.setattr(materialize, "run_public_tests", retry_without_cached_outage)
    result = ExecutionResult(
        action_id="act_victor_16_run_public_tests",
        agent_id="victor",
        action_type="run_public_tests",
    )
    OrgExecutionAdapter()._h_run_public_tests(
        world, "victor", {}, result, tick=world.world_tick
    )

    assert result.success is False
    assert result.failure_reason == "public_tests_infrastructure_error"
    assert world.__dict__["_public_tests_last"]["summary"] == "fresh container outage"


def test_current_red_ci_localizes_and_reopens_the_owner_repair_surface():
    world = _world()
    for artifact_id, artifact in world.product_artifacts.items():
        artifact.artifact_id = artifact_id
        artifact.content = artifact.mainline_content
    world.tasks = {
        "task_oss_cooper_feature_1": SimpleNamespace(description="feature one"),
        "task_oss_cooper_feature_2": SimpleNamespace(description="feature two"),
    }
    initialize_two_person_sdl(
        world,
        {
            "cooper_feature_1": "victor",
            "cooper_feature_2": "calvin",
        },
    )
    world._cooperbench_delivery_focus = True
    world._cooperbench_feature_issue_by_agent = {
        "victor": "cooper_feature_1",
        "calvin": "cooper_feature_2",
    }
    world.patches = {
        "patch_two": SimpleNamespace(
            actor_id="calvin",
            validation_status="accepted",
            related_issue_ids=["cooper_feature_2"],
            target_object_id="two",
        )
    }
    world.repo_system.repo = SimpleNamespace(
        main_commit_ids=[],
        branches={
            "branch_two": SimpleNamespace(commit_ids=["commit_two"]),
        },
        commits={"commit_two": SimpleNamespace()},
        ci_runs={
            "ci_two": SimpleNamespace(
                ci_id="ci_two",
                commit_id="commit_two",
                status="failed",
                failure_reasons=["src/two.py:74 | in __init__"],
            )
        },
        pull_requests={
            "pr_two": SimpleNamespace(
                pr_id="pr_two",
                author_id="calvin",
                source_branch="branch_two",
                status="approved",
                linked_issue_ids=["cooper_feature_2"],
                linked_issue=None,
                ci_run_ids=["ci_two"],
                ci_base_main_commit_ids=(),
                ci_brief="src/two.py:74 | in __init__",
            )
        },
    )
    current_hash = _repo_hash(world, prefer_mainline=False)
    world.__dict__["_public_tests_last_hash"] = current_hash
    world.__dict__["_public_tests_last"] = {
        "tick": 40,
        "passed": True,
        "failed_tests": [],
        "summary": "4 passed",
    }

    repairs = _cooperbench_feature_edit_candidates(world, "calvin")

    assert [item.parameters["file_path"] for item in repairs] == ["src/two.py"]
    assert repairs[0].parameters["_cooperbench_repair_source"] == "ci"
    assert repairs[0].parameters["_cooperbench_repair_pr_id"] == "pr_two"
    assert "src/two.py:74 | in __init__" in repairs[0].parameters["edit_goal"]
    focused = _cooperbench_focus_delivery_candidates(
        world,
        [
            ActionCandidate(
                action_type="work_on_task",
                parameters={"task_id": "task_oss_cooper_feature_2"},
            ),
            ActionCandidate(
                action_type="update_task_status",
                parameters={"task_id": "task_oss_cooper_feature_2"},
            ),
        ],
        "calvin",
    )
    assert [item.action_type for item in focused] == ["edit_repo_file"]


def test_followup_cooper_commit_invalidates_the_old_head_review():
    from environments.org_env.backend.repo.repo import PRStatus

    pr = SimpleNamespace(
        pr_id="pr_two",
        source_branch="branch_two",
        status=PRStatus.APPROVED,
        reviewers=["victor"],
        reviewed=True,
        approved_by=["victor"],
        approved_tick=50,
        ci_passed=True,
        ci_base_main_commit_ids=("base",),
    )
    world = SimpleNamespace(
        _cooperbench_delivery_focus=True,
        repo_system=SimpleNamespace(
            repo=SimpleNamespace(pull_requests={"pr_two": pr})
        ),
    )

    invalidated = _invalidate_cooperbench_reviews_for_branch(
        world, "branch_two"
    )

    assert invalidated == ["pr_two"]
    assert pr.status is PRStatus.REVIEW_REQUESTED
    assert pr.reviewed is False
    assert pr.approved_by == []
    assert pr.approved_tick is None
    assert pr.ci_passed is False
    assert pr.ci_base_main_commit_ids is None


def test_current_green_public_suite_is_bound_to_the_exact_cooper_commit():
    world = _world()
    for artifact_id, artifact in world.product_artifacts.items():
        artifact.artifact_id = artifact_id
        artifact.content = artifact.mainline_content
        artifact.linked_branch_ids = []
        artifact.linked_commit_ids = []
    world.tasks = {
        "task_oss_cooper_feature_1": SimpleNamespace(
            linked_artifacts=["one", "two"]
        )
    }
    world.repo_system = RepoLiteSystem()
    world.__dict__["_cooperbench_public_baseline_files"] = {
        artifact.linked_file_path: artifact.mainline_content
        for artifact in world.product_artifacts.values()
    }
    initialize_two_person_sdl(
        world,
        {
            "cooper_feature_1": "victor",
            "cooper_feature_2": "calvin",
        },
    )
    sv.freeze_baseline(world)
    sv.initialize_actor_desks(world)
    world._cooperbench_delivery_focus = True
    world._cooperbench_feature_issue_by_agent = {
        "victor": "cooper_feature_1",
        "calvin": "cooper_feature_2",
    }
    world.agents = {
        "victor": SimpleNamespace(skill=lambda *_args: 0.1),
        "calvin": SimpleNamespace(skill=lambda *_args: 0.1),
    }
    branch = world.repo_system.create_branch(
        "victor", linked_task="task_oss_cooper_feature_1", tick=1
    )
    sv.bind_actor_branch(world, "victor", branch.branch_id)
    patches = [
        SimpleNamespace(
            patch_id="patch_one",
            actor_id="victor",
            validation_status="pending",
            related_issue_ids=["cooper_feature_1"],
            target_object_id="one",
            new_content=world.product_artifacts["one"].mainline_content,
            creates_file=False,
            deletes_file=False,
        ),
        SimpleNamespace(
            patch_id="patch_two",
            actor_id="victor",
            validation_status="pending",
            related_issue_ids=["cooper_feature_1"],
            target_object_id="two",
            new_content=world.product_artifacts["two"].mainline_content,
            creates_file=False,
            deletes_file=False,
        ),
    ]
    world.patches = {}
    world.__dict__["_pending_by_branch"] = {branch.branch_id: []}
    for patch in patches:
        source = sv.actor_desk_snapshot(world, "victor")
        prepared = sv.prepare_actor_patch(
            world,
            "victor",
            patch,
            expected_snapshot_id=source.snapshot_id,
        )
        patch.validation_status = "accepted"
        world.patches[patch.patch_id] = patch
        world.__dict__["_pending_by_branch"][branch.branch_id].append(
            (patch.patch_id, patch.target_object_id)
        )
        sv.commit_actor_patch(world, prepared, branch_id=branch.branch_id)
    current_hash = _repo_hash(world, prefer_mainline=False)
    source = sv.actor_desk_snapshot(world, "victor")
    world.__dict__["_cooperbench_actor_public_tests"] = {
        "victor": {
            "actor_id": "victor",
            "source_snapshot": source.receipt(),
            "repo_hash": current_hash,
            "tick": 7,
            "status": "passed",
            "passed": True,
            "failed_tests": [],
            "summary": "402 passed, 21 skipped",
        }
    }
    world.__dict__["_public_tests_last_hash"] = current_hash
    world.__dict__["_public_tests_last"] = {
        "tick": 7,
        "passed": True,
        "failed_tests": [],
        "summary": "402 passed, 21 skipped",
    }
    result = ExecutionResult(
        action_id="act_victor_9_commit_patch",
        agent_id="victor",
        action_type="commit_patch",
    )

    OrgExecutionAdapter()._h_commit_patch(
        world, "victor", {"branch_id": branch.branch_id}, result, 9
    )

    assert result.success is True
    commit = world.repo_system.repo.commits[result.created_objects[-1]]
    assert commit.risk_level == "high"
    assert commit.test_status == "pass"
    assert "missing_tests" not in commit.quality_flags
    assert commit.cooperbench_public_test_evidence == {
        "schema_version": "orgenv_cooperbench_commit_public_test_v1",
        "repo_hash": current_hash,
        "test_tick": 7,
        "commit_tick": 9,
        "patch_ids": ["patch_one", "patch_two"],
        "passed": True,
        "validation_strength": "public_regression",
        "functional_public_regression_available": True,
        "summary": "402 passed, 21 skipped",
        "source_snapshot": source.receipt(),
        "actor_id": "victor",
    }
    pr = world.repo_system.open_pr(
        agent_id="victor",
        source_branch=branch.branch_id,
        reviewers=["calvin"],
    )
    ci = world.repo_system.run_ci(pr_id=pr.pr_id, tick=10)
    assert ci is not None and ci.status == "passed"
