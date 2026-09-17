"""Release-focused checks for the final v2 Text/Exec transfer closure."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from environments.org_env.backend.simulation.world import OrgWorld
from environments.org_env.config.scenarios import oss_time_machine_formal
from environments.org_env.experiments.capability_transfer import (
    CANONICAL_V2_BUNDLE_SHA256,
    CAPABILITY_FORM_EXECUTABLE,
    CAPABILITY_FORM_TEXT_ONLY,
    ROSTER_ORIGIN_FRESH,
    capability_bundle_sha256,
    inherited_capability_texts,
    inject_capability_bundle,
    validate_capability_bundle,
)
from environments.org_env.experiments.records import (
    validate_capability_transfer_receipt,
)
from environments.org_env.policy.compiled_protocols import (
    GUARD_ACTIONS,
    GUARD_REASON_CODES,
    authorize_before_action,
)
from environments.org_env.proposals.objects import Proposal
from environments.org_env.runtime_adapter.execution import OrgExecutionAdapter
from relic.source_runner import SourceMainRunnerError
from relic.transfer_runner import run_transfer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_PATH = (
    PROJECT_ROOT
    / "environments/org_env/data/capability_bundles/canonical_v2.json"
)


def _bundle() -> dict:
    return json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))


def _world(seed: int) -> OrgWorld:
    return OrgWorld(
        oss_time_machine_formal(seed=seed, dataset_id="mini_blobstore_v1")
    ).build()


def test_canonical_v2_builder_and_closed_guard_routes() -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(PROJECT_ROOT)
    completed = subprocess.run(
        [sys.executable, "tools/build_canonical_v2_bundle.py", "--check"],
        cwd=PROJECT_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr

    bundle = _bundle()
    validate_capability_bundle(bundle)
    assert capability_bundle_sha256(bundle) == CANONICAL_V2_BUNDLE_SHA256
    assert len(bundle["protocols"]) == 6
    assert set(GUARD_ACTIONS) == set(GUARD_REASON_CODES) == {
        "issue_owner_assigned",
        "branch_owned_by_actor",
        "unchanged_failed_ci_not_retried",
        "current_ci_attested",
        "independent_review",
        "release_gate_covered",
    }


def test_exec_injection_is_compiled_fixed_and_fails_closed_when_tampered() -> None:
    world = _world(seed=31)
    receipt = inject_capability_bundle(
        world,
        _bundle(),
        roster_origin=ROSTER_ORIGIN_FRESH,
        capability_form=CAPABILITY_FORM_EXECUTABLE,
        fixed_protocol_landscape=True,
    )

    assert receipt["fixed_protocol_landscape"] is True
    assert len(receipt["protocols_injected"]) == 6
    assert len(receipt["compiled_protocols"]) == 6
    assert world.__dict__["_fixed_protocol_landscape"] is True
    validate_capability_transfer_receipt(
        {
            "experiment_phase": "capability_transfer",
            "pack": "mini_blobstore_v1",
            "capability_transfer": receipt,
        }
    )

    proposal = world.proposal_manager.create_proposal(
        Proposal("fixed-window-proposal", "protocol_proposal"), world
    )
    assert proposal.status == "rejected"
    assert proposal.rejection_reason == "endogenous_protocol_formation_disabled"
    with pytest.raises(ValueError, match="^endogenous_protocol_formation_disabled$"):
        world.protocol_registry.propose(
            proposer_id="paul",
            protocol_type="new_rule",
            rule_summary="new rule",
        )

    compiled = next(iter(world.proposal_manager.protocol_specs.values()))
    compiled.machine_bindings[0]["guard"] = "not_a_closed_guard"
    with pytest.raises(ValueError, match="^compiled_protocol_integrity_error:"):
        authorize_before_action(
            world,
            "paul",
            "open_pr",
            {"source_branch": "branch_missing"},
            tick=0,
        )


def test_text_injection_has_only_matched_direct_prose() -> None:
    bundle = _bundle()
    world = _world(seed=32)
    receipt = inject_capability_bundle(
        world,
        bundle,
        roster_origin=ROSTER_ORIGIN_FRESH,
        capability_form=CAPABILITY_FORM_TEXT_ONLY,
        fixed_protocol_landscape=True,
    )

    assert receipt["protocols_injected"] == []
    assert receipt["documents_injected"] == []
    assert receipt["compiled_protocols"] == []
    assert len(receipt["prose_entries_injected"]) == 6
    assert inherited_capability_texts(world) == [
        row["rule_summary"] for row in bundle["protocols"]
    ]


def test_direct_execution_cannot_bypass_the_exec_open_pr_guard() -> None:
    world = _world(seed=33)
    inject_capability_bundle(
        world,
        _bundle(),
        roster_origin=ROSTER_ORIGIN_FRESH,
        capability_form=CAPABILITY_FORM_EXECUTABLE,
        fixed_protocol_landscape=True,
    )
    owner, non_owner = list(world.agents)[:2]
    branch = world.repo_system.create_branch(owner_id=owner, base="main", tick=0)
    world.repo_system.commit_changes(
        agent_id=owner,
        branch_id=branch.branch_id,
        message="ready",
        changed_files=["module.py"],
        tick=0,
        test_status="passed",
        risk_level="low",
    )

    before = len(world.repo_system.repo.pull_requests)
    result = OrgExecutionAdapter().execute(
        non_owner,
        SimpleNamespace(
            action_type="open_pr", parameters={"source_branch": branch.branch_id}
        ),
        world,
    )

    assert result.success is False
    assert result.failure_reason == "compiled_protocol_blocked:branch_owner_mismatch"
    assert len(world.repo_system.repo.pull_requests) == before


def test_transfer_dry_plan_is_b2_only_and_uses_the_canonical_v2_bundle(
    tmp_path: Path,
) -> None:
    root = tmp_path / "transfer"
    result = run_transfer(
        output_root=root,
        dry_run=True,
        workloads=("w01",),
        seeds=(1401,),
    )

    assert result.status == "planned"
    assert result.total_target_runs == 60
    assert result.selected_target_runs == 2
    manifest = json.loads((root / "transfer_manifest.json").read_text(encoding="utf-8"))
    assert manifest["plan"]["canonical_bundle"]["bundle_sha256"] == (
        CANONICAL_V2_BUNDLE_SHA256
    )
    assert manifest["plan"]["target"]["fresh_reference"] == (
        "existing_main_study_b2_only"
    )
    for arm in ("text", "exec"):
        source_manifest = json.loads(
            (
                root
                / "source-dry-run"
                / arm
                / "w01__seed1401"
                / "manifest.json"
            ).read_text(encoding="utf-8")
        )
        assert [case["case"] for case in source_manifest["cases"]] == ["b2"]
        environment = source_manifest["cases"][0]["environment"]
        assert environment["ORG_TRANSFER_FIXED_PROTOCOL_LANDSCAPE"] == "1"
        assert environment["ORG_TRANSFER_ROSTER_ORIGIN"] == "fresh_roster"


def test_transfer_strict_run_requires_bindings_before_source_invocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> int:
        raise AssertionError("missing evaluator bindings must stop before invocation")

    monkeypatch.setattr("relic.transfer_runner._invoke_source_batch", forbidden)
    with pytest.raises(SourceMainRunnerError, match="^formal_evaluator_bindings_required$"):
        run_transfer(
            output_root=tmp_path / "formal",
            strict_reproducibility=True,
            workloads=("w01",),
            seeds=(1401,),
        )
