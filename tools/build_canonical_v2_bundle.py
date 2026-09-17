#!/usr/bin/env python3
"""Build the frozen, machine-bindable canonical v2 transfer bundle.

The source roster is copied only after the exact canonical-v1 bytes have been
verified.  Protocol prose is curated here, beside its closed binding, so the
generated JSON is deterministic and never relies on interpreting English at
runtime.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_V1_PATH = (
    REPOSITORY_ROOT
    / "environments"
    / "org_env"
    / "data"
    / "capability_bundles"
    / "canonical_v1.json"
)
CANONICAL_V2_PATH = CANONICAL_V1_PATH.with_name("canonical_v2.json")
CANONICAL_V1_SHA256 = (
    "ce3c96cd2263c79e2a53a4969167f52a61814f32ab14dfa83d0a9e7fda44cff5"
)
CANONICAL_V1_GIT_COMMIT = "d7353db891b2b66e66116cbce8db1dcba4faf3b5"


def _read_pinned_v1() -> dict[str, Any]:
    raw = CANONICAL_V1_PATH.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != CANONICAL_V1_SHA256:
        raise RuntimeError(
            "canonical_v1_sha256_mismatch:"
            f"expected={CANONICAL_V1_SHA256}:actual={actual}"
        )
    bundle = json.loads(raw)
    if bundle.get("schema_version") != "org_capability_bundle_v1":
        raise RuntimeError("canonical_v1_schema_mismatch")
    if not isinstance(bundle.get("roster"), list) or not bundle["roster"]:
        raise RuntimeError("canonical_v1_roster_missing")
    return bundle


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_rows(
    source: Mapping[str, Any], protocol_ids: Iterable[str]
) -> list[Mapping[str, Any]]:
    by_id = {
        str(row.get("protocol_id") or ""): row
        for row in source.get("protocols", [])
        if isinstance(row, Mapping)
    }
    rows: list[Mapping[str, Any]] = []
    for protocol_id in protocol_ids:
        if protocol_id not in by_id:
            raise RuntimeError(f"canonical_v1_protocol_missing:{protocol_id}")
        rows.append(by_id[protocol_id])
    return rows


def _supporters(rows: Iterable[Mapping[str, Any]]) -> list[str]:
    return list(
        dict.fromkeys(
            str(agent_id)
            for row in rows
            for agent_id in (row.get("supporters") or [])
            if str(agent_id)
        )
    )


def _emergence_level(rows: Iterable[Mapping[str, Any]]) -> str:
    levels = [str(row.get("emergence_level") or "none") for row in rows]
    for level in ("strong", "moderate", "weak"):
        if level in levels:
            return level
    return "none"


def _protocol(
    source: Mapping[str, Any],
    *,
    guard: str,
    source_protocol_ids: tuple[str, ...],
    rule_summary: str,
    scope: str,
    target_process: str,
    capability: str,
    actions: tuple[str, ...],
    reason_code: str,
    trigger_condition: str,
    required_steps: list[str],
    required_fields: list[str],
    enforcement_rule: str,
    violation_condition: str,
    exception_rule: str,
    responsible_roles: dict[str, list[str]],
    success_metric: str,
    affected_artifacts: list[str],
    benefits: list[str],
    costs: list[str],
    risks: list[str],
) -> dict[str, Any]:
    sources = _source_rows(source, source_protocol_ids)
    return {
        "protocol_id": f"proto_{guard}",
        "name": guard.replace("_", " ").title(),
        "protocol_type": guard,
        "rule_summary": rule_summary,
        "scope": scope,
        "target_process": target_process,
        "supporters": _supporters(sources),
        "emergence_level": _emergence_level(sources),
        "capability": capability,
        "spec": {
            "trigger_condition": trigger_condition,
            "required_steps": required_steps,
            "required_fields": required_fields,
            "enforcement_rule": enforcement_rule,
            "violation_condition": violation_condition,
            "exception_rule": exception_rule,
            "problem_evidence": [
                f"canonical_v1:{protocol_id}" for protocol_id in source_protocol_ids
            ],
            "scope": scope,
            "responsible_roles": responsible_roles,
            "success_metric": success_metric,
            "enforcement_action": (
                f"Block {', '.join(actions)} before business-state mutation and emit "
                f"reason_code={reason_code}."
            ),
            "sunset_rule": (
                "No automatic sunset; the guard remains active while the inherited "
                "ProtocolSpec status is adopted."
            ),
            "affected_agents": ["all_agents"],
            "affected_actions": list(actions),
            "affected_artifacts": affected_artifacts,
            "benefits": benefits,
            "costs": costs,
            "risks": risks,
        },
        "bindings": [
            {
                "binding_id": f"binding_{guard}",
                "hook": "before_action",
                "actions": list(actions),
                "guard": guard,
                "effect": "block_action",
                "reason_code": reason_code,
                "params": {},
            }
        ],
    }


def build_bundle(source: Mapping[str, Any]) -> dict[str, Any]:
    protocols = [
        _protocol(
            source,
            guard="issue_owner_assigned",
            source_protocol_ids=("proto_spec_6",),
            rule_summary=(
                "Before open_pr, resolve the source branch exactly as the action "
                "handler does. If its commits link work, every explicit linked task "
                "must exist and have a non-empty owner_id. Every native Issue must "
                "have owner_id set, and every Task associated with that issue must "
                "also exist and be owned. A ProductArtifact issue has no inferred "
                "owner: it must resolve to at least one associated Task, and every "
                "associated Task must exist and be owned. A branch with no issue or "
                "task linkage is outside this guard. Assign every missing owner and "
                "retry; editing and committing remain allowed."
            ),
            scope="pull_request_intake",
            target_process="Opening an issue- or task-linked pull request.",
            capability="customer_triage",
            actions=("open_pr",),
            reason_code="issue_owner_required",
            trigger_condition=(
                "The selected action is open_pr and its resolved source branch has a "
                "commit with linked_issue_ids, linked_task_ids, or linked_task_id."
            ),
            required_steps=[
                "Resolve parameters.source_branch, including the handler's omitted-parameter fallback.",
                "Collect unique issue and task identifiers from every commit on that branch.",
                "For a native Issue, require its non-empty owner_id and require every Task named by or reverse-linked to that Issue to exist and have owner_id.",
                "For a ProductArtifact issue, resolve Tasks named by linked_task_ids or whose linked_issues contains the issue id; require at least one and require every one to exist and have owner_id.",
                "If blocked, assign every missing task owner before retrying open_pr.",
            ],
            required_fields=[
                "action.action_type",
                "action.parameters.source_branch",
                "actor.id",
                "repo.branches[source_branch].commit_ids",
                "repo.commits[commit_id].linked_issue_ids",
                "repo.commits[commit_id].linked_task_id",
                "repo.commits[commit_id].linked_task_ids",
                "issues[issue_id].owner_id",
                "product_artifacts[issue_id].artifact_type",
                "tasks[task_id].linked_issues",
                "tasks[task_id].owner_id",
            ],
            enforcement_rule=(
                "For applicable work, allow open_pr only when every explicit linked "
                "Task exists with owner_id set, every native linked Issue has owner_id "
                "set and only owned associated Tasks, and every ProductArtifact issue "
                "has at least one associated Task with all associated Tasks owned."
            ),
            violation_condition=(
                "The branch carries linked work and an explicit linked Task is absent "
                "or unowned, a native linked Issue has no owner or an absent/unowned "
                "associated Task, or a ProductArtifact issue has no associated Task or "
                "has an absent/unowned associated Task."
            ),
            exception_rule=(
                "A branch with no issue and no task linkage is not governed; an "
                "unresolvable branch is left to the repository handler's native "
                "invalid-action refusal."
            ),
            responsible_roles={
                "executor": ["branch_owner"],
                "repairer": ["task_owner", "founder", "cofounder"],
            },
            success_metric=(
                "Every allowed applicable open_pr authorization has no missing or "
                "unowned explicit or issue-associated Task, and every native linked "
                "Issue has a non-empty owner_id."
            ),
            affected_artifacts=["branch", "commit", "issue", "task", "pull_request"],
            benefits=["Issue-linked requests enter review with an accountable task owner."],
            costs=["Unassigned issue work needs one ownership action before open_pr."],
            risks=["A stale or missing task link blocks open_pr until linkage is repaired."],
        ),
        _protocol(
            source,
            guard="branch_owned_by_actor",
            source_protocol_ids=("proto_spec_6",),
            rule_summary=(
                "Before open_pr, resolve the source branch exactly as the action "
                "handler does and require actor.id to equal branch.owner_id. A "
                "non-owner is blocked without changing the branch; the recorded owner "
                "can open the request, so editing, committing, review, and merge repair "
                "lanes remain available."
            ),
            scope="branch_ownership",
            target_process="Opening a pull request from an existing source branch.",
            capability="ownership_map",
            actions=("open_pr",),
            reason_code="branch_owner_mismatch",
            trigger_condition=(
                "The selected action is open_pr and its source branch resolves to an "
                "existing Branch."
            ),
            required_steps=[
                "Resolve parameters.source_branch, including the handler's omitted-parameter fallback.",
                "Read actor.id and the resolved Branch.owner_id.",
                "Allow only an exact identifier match; otherwise have the recorded owner retry open_pr.",
            ],
            required_fields=[
                "action.action_type",
                "action.parameters.source_branch",
                "actor.id",
                "repo.branches[source_branch].owner_id",
            ],
            enforcement_rule=(
                "Allow open_pr only when actor.id exactly equals the resolved branch.owner_id."
            ),
            violation_condition=(
                "The source branch exists and actor.id differs from branch.owner_id."
            ),
            exception_rule=(
                "An unresolvable branch is left to the repository handler's native "
                "invalid-action refusal."
            ),
            responsible_roles={
                "executor": ["branch_owner"],
                "repairer": ["branch_owner"],
            },
            success_metric=(
                "Every allowed open_pr authorization has actor.id equal to branch.owner_id."
            ),
            affected_artifacts=["branch", "pull_request"],
            benefits=["A pull request cannot be opened under another agent's branch identity."],
            costs=["A non-owner must route the open_pr action to the recorded owner."],
            risks=["An incorrect branch.owner_id blocks submission until ownership state is corrected."],
        ),
        _protocol(
            source,
            guard="unchanged_failed_ci_not_retried",
            source_protocol_ids=("proto_require_every_pr_to_be_rebased_onto_curr",),
            rule_summary=(
                "Before run_ci or ci_test, resolve the pull request exactly as the action handler "
                "does. Block only when its latest CI result is failed and already "
                "attests the current source-branch head against the current mainline "
                "commit list. A new branch head or changed mainline permits a retry; a "
                "not_run infrastructure result follows the native retry path."
            ),
            scope="ci_retry_control",
            target_process="Running CI for an open pull request through either CI action alias.",
            capability="evidence_workflow",
            actions=("run_ci", "ci_test"),
            reason_code="unchanged_failed_ci",
            trigger_condition=(
                "The selected action is run_ci or ci_test and its pull request resolves to an "
                "open request with a source branch and at least one CI result."
            ),
            required_steps=[
                "Resolve parameters.pr_id, including the shared CI handler's omitted-parameter fallback.",
                "Read the current source-branch head and current repo.main_commit_ids.",
                "Read the latest existing CIResult referenced by PullRequest.ci_run_ids.",
                "If the latest status is failed on that exact head and base, change the branch head before retrying run_ci or ci_test.",
            ],
            required_fields=[
                "action.action_type",
                "action.parameters.pr_id",
                "repo.pull_requests[pr_id].source_branch",
                "repo.pull_requests[pr_id].ci_run_ids",
                "repo.pull_requests[pr_id].ci_base_main_commit_ids",
                "repo.branches[source_branch].commit_ids",
                "repo.ci_runs[ci_id].commit_id",
                "repo.ci_runs[ci_id].status",
                "repo.main_commit_ids",
            ],
            enforcement_rule=(
                "Block run_ci and ci_test exactly when latest_ci.status is failed, "
                "latest_ci.commit_id equals the current branch head, and the PR's "
                "ci_base_main_commit_ids equals repo.main_commit_ids."
            ),
            violation_condition=(
                "run_ci or ci_test targets a failed CI attestation whose branch head and "
                "mainline base are both unchanged."
            ),
            exception_rule=(
                "Allow when there is no prior result, the head changed, the mainline "
                "base changed, or the latest status is not_run; malformed or terminal "
                "requests remain native repository refusals."
            ),
            responsible_roles={
                "executor": ["branch_owner", "reliability"],
                "repairer": ["branch_owner"],
            },
            success_metric=(
                "No allowed run_ci or ci_test authorization repeats a failed result on the same "
                "branch head and mainline commit list."
            ),
            affected_artifacts=["pull_request", "branch", "ci_result", "mainline"],
            benefits=["Failed CI consumes another run only after its evaluated code or base changes."],
            costs=["The author must commit a repair before repeating an unchanged product failure."],
            risks=["A misclassified product failure could delay a retry until the head or base changes."],
        ),
        _protocol(
            source,
            guard="current_ci_attested",
            source_protocol_ids=(
                "proto_spec_1",
                "proto_require_every_pr_to_be_rebased_onto_curr",
            ),
            rule_summary=(
                "Before merge_pr for a pull request carrying commits or patches, "
                "require approved status and no merge conflict; require pr.commit_ids "
                "to exactly equal the source branch commit list and pr.patch_ids to "
                "exactly equal the ordered flattening of those commits' patch_ids; "
                "require both pr.ci_passed and a latest passed CI result for the current "
                "branch head and require ci_base_main_commit_ids to equal the current "
                "repo.main_commit_ids. When the active profile exposes a deterministic "
                "candidate-tree digest, ci_tree_hash must exactly equal that digest; "
                "ordinary RepoLite uses the commit head and mainline base as its exact "
                "identity and does not require an unavailable digest. Missing, failed, "
                "conflicted, unsynchronized, or stale evidence blocks only merge; commit "
                "and run_ci remain available to repair it."
            ),
            scope="merge_attestation",
            target_process="Merging a delivery pull request into mainline.",
            capability="workflow_integration",
            actions=("merge_pr",),
            reason_code="current_ci_attestation_required",
            trigger_condition=(
                "The selected action is merge_pr and its resolved PullRequest carries "
                "at least one commit or patch."
            ),
            required_steps=[
                "Resolve parameters.pr_id, including the handler's omitted-parameter fallback.",
                "Resolve the pull request's source branch and current branch head.",
                "Require PullRequest.status approved and PullRequest.merge_conflict false.",
                "Require PullRequest.commit_ids to exactly equal Branch.commit_ids and PullRequest.patch_ids to exactly equal the ordered flattening of Commit.patch_ids for that branch.",
                "Read the latest existing CIResult referenced by PullRequest.ci_run_ids.",
                "Require PullRequest.ci_passed, latest-CI passed status, exact head equality, and exact mainline commit-list equality before merge.",
                "If the active profile can compute a candidate-tree digest, require PullRequest.ci_tree_hash to exactly equal it.",
                "If blocked, commit the repair if needed and run_ci against the current head and base before retrying merge_pr.",
            ],
            required_fields=[
                "action.action_type",
                "action.parameters.pr_id",
                "repo.pull_requests[pr_id].status",
                "repo.pull_requests[pr_id].merge_conflict",
                "repo.pull_requests[pr_id].commit_ids",
                "repo.pull_requests[pr_id].patch_ids",
                "repo.pull_requests[pr_id].source_branch",
                "repo.pull_requests[pr_id].ci_run_ids",
                "repo.pull_requests[pr_id].ci_passed",
                "repo.pull_requests[pr_id].ci_base_main_commit_ids",
                "repo.pull_requests[pr_id].ci_tree_hash when candidate digest is available",
                "repo.branches[source_branch].commit_ids",
                "repo.commits[commit_id].patch_ids",
                "repo.ci_runs[ci_id].commit_id",
                "repo.ci_runs[ci_id].status",
                "repo.main_commit_ids",
            ],
            enforcement_rule=(
                "Allow merge_pr for delivery work only when the PR is approved and not "
                "conflicted; its commit_ids equal Branch.commit_ids; its patch_ids equal "
                "the ordered flattening of those commits' patch_ids; pr.ci_passed is "
                "true; latest_ci.status is passed; latest_ci.commit_id equals the current "
                "source-branch head; and ci_base_main_commit_ids exactly equals "
                "repo.main_commit_ids. Additionally require exact ci_tree_hash equality "
                "whenever the active profile can compute a candidate-tree digest."
            ),
            violation_condition=(
                "A delivery merge is unapproved or conflicted; its PR commit or patch "
                "lists differ from the source branch; pr.ci_passed is false; its latest "
                "CI result is missing, non-passed, or for a different branch head; its "
                "CI base differs from the current mainline commit list; or an available "
                "computed candidate digest differs from ci_tree_hash."
            ),
            exception_rule=(
                "A metadata-only pull request with no commits and no patches is outside "
                "this CI guard; an unresolvable request remains a native repository refusal."
            ),
            responsible_roles={
                "executor": ["pull_request_author", "founder", "cofounder", "reliability"],
                "repairer": ["branch_owner", "reliability"],
            },
            success_metric=(
                "Every allowed delivery merge is approved, conflict-free, synchronized "
                "to the source branch's exact commit and patch lists, and has a passed "
                "latest CI result whose commit and base exactly equal the live branch "
                "head and mainline, plus an exact tree digest in profiles that expose one."
            ),
            affected_artifacts=["pull_request", "branch", "ci_result", "mainline"],
            benefits=["A delivery cannot merge on CI evidence for an older head or base."],
            costs=["A mainline or branch change requires one current CI run before merge."],
            risks=["Long-lived pull requests may require repeated CI after unrelated mainline merges."],
        ),
        _protocol(
            source,
            guard="independent_review",
            source_protocol_ids=("proto_spec_1",),
            rule_summary=(
                "When the roster contains an agent other than the pull-request author, "
                "review_pr, approve_pr, and formal_pr_review by that author are "
                "blocked, and merge_pr "
                "requires at least one approved_by identifier that is both a current "
                "roster member and different from author_id. Unknown actor identifiers "
                "cannot review or approve. "
                "A one-agent roster may self-review so the workflow does not deadlock; "
                "request_changes and all repair actions remain allowed."
            ),
            scope="independent_pull_request_review",
            target_process="Approving, reviewing, or merging a pull request.",
            capability="review_gate",
            actions=("review_pr", "approve_pr", "formal_pr_review", "merge_pr"),
            reason_code="independent_review_required",
            trigger_condition=(
                "The selected action is review_pr, approve_pr, formal_pr_review, or "
                "merge_pr and its pull request resolves to an existing PullRequest."
            ),
            required_steps=[
                "Resolve parameters.pr_id; merge_pr also uses the handler's omitted-parameter fallback.",
                "Read PullRequest.author_id and the identifiers in world.agents.",
                "For review_pr, approve_pr, or formal_pr_review, require actor.id to name a current roster member; in a staffed roster it must also differ from author_id.",
                "For merge_pr in a staffed roster, require PullRequest.approved_by to contain a current roster member different from author_id.",
                "If blocked, have another roster member approve before retrying merge_pr.",
            ],
            required_fields=[
                "action.action_type",
                "action.parameters.pr_id",
                "actor.id",
                "world.agents",
                "repo.pull_requests[pr_id].author_id",
                "repo.pull_requests[pr_id].approved_by",
            ],
            enforcement_rule=(
                "If any roster identifier differs from author_id, block self review or "
                "approval and allow merge only with at least one approved_by current "
                "roster member different from author_id; reject unknown review actors. "
                "Otherwise allow the solo-roster member to self-review."
            ),
            violation_condition=(
                "A review or approval actor is not a current roster member; or, in a "
                "roster with an available non-author, the author attempts review_pr, "
                "approve_pr, or formal_pr_review, or merge_pr lacks a non-author current "
                "roster member in approved_by."
            ),
            exception_rule=(
                "When no roster identifier differs from author_id, self review is "
                "allowed; request_changes is not governed and remains available."
            ),
            responsible_roles={
                "author": ["pull_request_author"],
                "reviewer": ["non_author_roster_member"],
                "repairer": ["non_author_roster_member"],
            },
            success_metric=(
                "Every allowed staffed-roster merge has at least one approved_by value "
                "that is a current roster member different from PullRequest.author_id."
            ),
            affected_artifacts=["pull_request", "review_record", "roster"],
            benefits=["Staffed organizations cannot treat an author's self-approval as independent review."],
            costs=["A staffed roster needs one non-author review before merge."],
            risks=["An unavailable non-author can delay merge until that reviewer acts."],
        ),
        _protocol(
            source,
            guard="release_gate_covered",
            source_protocol_ids=("proto_spec_4",),
            rule_summary=(
                "Before publish_product_release, resolve the release candidate exactly "
                "as the action handler does. Require approved status, the roster-aware "
                "release_approval_met check, no blockers, a non-empty included_pr_ids "
                "list whose requests all exist, are merged, and have ci_passed. Each "
                "candidate must carry the exact non-empty, duplicate-free gate profile "
                "selected by release_gates_for(world). Each profile gate must have exactly "
                "one recorded passed result or its exact identifier must already be listed "
                "in waived_gates; unknown gates, waivers, and results are rejected. Run "
                "readiness, collect the required "
                "approvals, and repair blockers before retrying publish."
            ),
            scope="release_publication",
            target_process="Publishing an approved release candidate to users.",
            capability="release_governance",
            actions=("publish_product_release",),
            reason_code="release_gate_coverage_required",
            trigger_condition=(
                "The selected action is publish_product_release and its candidate "
                "resolves to an existing ReleaseCandidate."
            ),
            required_steps=[
                "Resolve parameters.candidate_id using the same open-candidate selection as the action handler.",
                "Require candidate status approved, release_approval_met(world, candidate), no blockers, and at least one included pull request.",
                "Require every included pull request to exist, be merged, and have ci_passed true.",
                "Require required_gates to equal the current release_gates_for(world) profile, be non-empty, and contain no duplicate or unknown identifier.",
                "For every profile gate, require either exactly one recorded passed gate_result or an exact identifier in waived_gates; reject unknown waivers/results and duplicate results.",
                "If blocked, repair the reported blocker and rerun run_launch_readiness_check and approvals before retrying publish_product_release.",
            ],
            required_fields=[
                "action.action_type",
                "action.parameters.candidate_id",
                "repo.release_candidates[candidate_id].status",
                "repo.release_candidates[candidate_id].included_pr_ids",
                "repo.release_candidates[candidate_id].required_gates",
                "repo.release_candidates[candidate_id].gate_results",
                "repo.release_candidates[candidate_id].blockers",
                "repo.release_candidates[candidate_id].approvals",
                "repo.release_candidates[candidate_id].waived_gates",
                "world.agents[agent_id].role",
                "repo.pull_requests[pr_id].status",
                "repo.pull_requests[pr_id].ci_passed",
            ],
            enforcement_rule=(
                "Allow publish_product_release only for an approved candidate that "
                "satisfies release_approval_met(world, candidate), is non-empty and "
                "blocker-free, whose included PRs are present, merged, and CI passed, "
                "and whose non-empty, duplicate-free required gates equal the current "
                "release profile and each have exactly one recorded passed result or an "
                "explicit exact-id waiver, with no unknown waiver/result identifiers."
            ),
            violation_condition=(
                "The resolved candidate is not approved, fails the roster-aware release "
                "approval check, has blockers or no included PRs, names a missing, "
                "unmerged, or non-CI-passed PR, has a missing, duplicate, or unknown "
                "profile gate, has an unknown/duplicate result or waiver, or has a "
                "required gate without exactly one passed result or exact-id waiver."
            ),
            exception_rule=(
                "An unresolvable candidate is left to the release handler's native "
                "invalid-action refusal; an existing waived gate is accepted only when "
                "its identifier exactly matches required_gates."
            ),
            responsible_roles={
                "executor": ["founder", "cofounder"],
                "verifier": ["reliability"],
                "repairer": ["pull_request_author", "release_owner"],
            },
            success_metric=(
                "Every allowed publication satisfies the roster-aware approval check "
                "and has no blockers, no missing or nonpassing included PR, and no "
                "uncovered required gate."
            ),
            affected_artifacts=["release_candidate", "release_gate_result", "pull_request", "product_release"],
            benefits=["A release cannot publish from a partial or stale approval record."],
            costs=["Changed or incomplete readiness evidence must be repaired before publication."],
            risks=["A malformed required-gate identifier blocks publication until the candidate record is repaired."],
        ),
    ]

    return {
        "schema_version": "org_capability_bundle_v2",
        "source_repository_id": "canonical_v2",
        "source_seed": int(source.get("source_seed") or 0),
        "source_tick": int(source.get("source_tick") or 0),
        "provenance": {
            "roster_source": "environments/org_env/data/capability_bundles/canonical_v1.json",
            "roster_source_sha256": CANONICAL_V1_SHA256,
            "roster_snapshot_sha256": _json_sha256(source["roster"]),
            "roster_source_git_commit": CANONICAL_V1_GIT_COMMIT,
            "protocol_source_repository_id": str(
                source.get("source_repository_id") or "canonical_v1"
            ),
            "curation": (
                "Six closed guards derived from canonical-v1 protocol concerns and "
                "restricted to predicates over existing OrgEnv action and world fields."
            ),
        },
        "capabilities": sorted(
            {
                "customer_triage",
                "ownership_map",
                "evidence_workflow",
                "workflow_integration",
                "review_gate",
                "release_governance",
            }
        ),
        "protocols": protocols,
        "roster": source["roster"],
        "documents": [],
    }


def _render(bundle: Mapping[str, Any]) -> bytes:
    return (json.dumps(bundle, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if canonical_v2.json is missing or differs from generated bytes",
    )
    args = parser.parse_args()

    rendered = _render(build_bundle(_read_pinned_v1()))
    if args.check:
        if not CANONICAL_V2_PATH.exists() or CANONICAL_V2_PATH.read_bytes() != rendered:
            raise SystemExit("canonical_v2_bundle_out_of_date")
        return 0

    CANONICAL_V2_PATH.write_bytes(rendered)
    print(CANONICAL_V2_PATH.relative_to(REPOSITORY_ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
