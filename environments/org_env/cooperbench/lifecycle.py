"""Two-person SDL and auditable B3 lifecycle for CooperBench.

The benchmark supplies two public feature requests to one small organization.
This module keeps the adapter honest about what that organization actually did:
delivery is complete only after reciprocal review and two independent
attestations over the same integrated mainline.  The Cooper treatment also
runs the main experiment's episode -> reflection -> wish -> proposal ->
pair-approval -> ProtocolSpec/registry lifecycle; a delivered patch without a
formed institution is not reported as B3.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping


_FEATURE_IDS = ("cooper_feature_1", "cooper_feature_2")
_SECRET_KEYS = (
    "api_key",
    "authorization",
    "access_token",
    "refresh_token",
    "secret",
    "password",
    "credential",
)
_MAX_TEXT = 1000
_MAX_ITEMS = 256


def _public_acceptance_obligations(description: str) -> list[str]:
    """Supplement the full public request with a source-located prose index."""
    from .public_contract import indexed_obligations
    return indexed_obligations(description)


def _contains_cue(text: str, cue: str) -> bool:
    if cue == "preserv":
        return re.search(r"\bpreserv\w*", text) is not None
    if cue == "backward compat":
        return re.search(r"\bbackward[ -]compat\w*", text) is not None
    return re.search(
        rf"(?<!\w){re.escape(cue)}(?!\w)", text
    ) is not None


def _public_compatibility_obligations(obligations: list[str]) -> list[str]:
    """Select one informative clause per compatibility invariant category.

    These are not evaluator assertions.  They are the negative promises the
    user put in the public brief, and therefore need baseline-vs-current
    evidence before a peer can approve a change.  Public briefs often repeat
    the same promise in the summary, API docs, and notes; requiring a separate
    long JSON check for every repetition would turn evidence into an output-
    budget trap.  Keep the strongest representative for each category.
    """

    candidates: dict[str, list[tuple[int, str]]] = {
        "default_path": [],
        "legacy_read": [],
        "layer_boundary": [],
        "identity": [],
    }
    for item in obligations:
        lowered = item.casefold()
        if item.rstrip().endswith(":"):
            continue
        # Conditions and API values ("when", "None", "load") do not promise
        # equivalence with the baseline. Require an actual preservation or
        # compatibility statement before ranking its representative clause.
        invariant = re.search(
            r"\b(?:unchanged|preserv\w*|backward[ -]compat\w*)\b|"
            r"\b(?:same|identical|exactly)\b[^.!?]*\b(?:before|current|existing|original)\b|"
            r"\b(?:legacy|existing|old)\b[^.!?]*\b(?:continue\w*|still)\b",
            lowered,
        )
        if invariant is None:
            continue
        default_score = 0
        if _contains_cue(lowered, "default"):
            default_score += 4
        if _contains_cue(lowered, "unchanged"):
            default_score += 6
        if _contains_cue(lowered, "optional"):
            default_score += 2
        if "opt-in" in lowered:
            default_score += 3
        if re.search(r"\bnone\b", lowered):
            default_score += 7
        if _contains_cue(lowered, "when"):
            default_score += 8
        if "compress" in lowered or "namespace" in lowered:
            default_score += 3
        if default_score:
            candidates["default_path"].append((default_score, item))

        legacy_score = 0
        if _contains_cue(lowered, "legacy"):
            legacy_score += 6
        if _contains_cue(lowered, "backward compat"):
            legacy_score += 8
        if _contains_cue(lowered, "missing") or _contains_cue(lowered, "load"):
            legacy_score += 3
        if legacy_score:
            candidates["legacy_read"].append((legacy_score, item))

        if "memory layer" in lowered and (
            _contains_cue(lowered, "unchanged")
            or _contains_cue(lowered, "only")
        ):
            candidates["layer_boundary"].append((10, item))

        if re.search(r"\bpreserv\w*", lowered) and any(
            word in lowered for word in ("type", "data", "structure")
        ):
            candidates["identity"].append((8, item))

    selected: list[str] = []
    for category in (
        "default_path",
        "legacy_read",
        "layer_boundary",
        "identity",
    ):
        ranked = candidates[category]
        if not ranked:
            continue
        # Prefer the most explicit clause, then the more informative text.
        item = max(ranked, key=lambda pair: (pair[0], len(pair[1])))[1]
        if item not in selected:
            selected.append(item)
    return selected


def _status(value: Any) -> str:
    return str(getattr(value, "value", value) or "").casefold()


def _linked_feature_ids(pr: Any) -> set[str]:
    linked = list(getattr(pr, "linked_issue_ids", []) or [])
    fallback = getattr(pr, "linked_issue", None)
    if fallback:
        linked.append(fallback)
    return {str(item) for item in linked if str(item).startswith("cooper_feature_")}


def initialize_two_person_sdl(
    world: Any, feature_owners: Mapping[str, str]
) -> dict[str, Any]:
    """Install one pair-local SDL state and serialize only real file overlap."""

    owners = {str(key): str(value) for key, value in feature_owners.items()}
    if set(owners) != set(_FEATURE_IDS) or len(set(owners.values())) != 2:
        raise ValueError("cooperbench_sdl_requires_two_distinct_feature_owners")
    component_map = getattr(world, "_oss_component_map", {}) or {}
    # The public surface extractor already ranks these paths by task relevance.
    # Preserve that order: alphabetic sorting previously promoted an example or
    # auxiliary file above the implementation file and then made that first
    # entry the only initial delivery target.
    feature_paths = {
        feature_id: list(
            dict.fromkeys(
                str(path).replace("\\", "/")
                for path in (component_map.get(feature_id) or [])
                if str(path).strip()
            )
        )
        for feature_id in _FEATURE_IDS
    }
    tasks = getattr(world, "tasks", {}) or {}
    required_feature_paths: dict[str, list[str]] = {}
    acceptance_obligations: dict[str, list[str]] = {}
    compatibility_obligations: dict[str, list[str]] = {}
    for feature_id in _FEATURE_IDS:
        task = tasks.get(f"task_oss_{feature_id}")
        from .visibility import visible_feature_brief
        description = visible_feature_brief(world, owners[feature_id], feature_id) or ""
        acceptance_obligations[feature_id] = _public_acceptance_obligations(
            description
        )
        compatibility_obligations[feature_id] = (
            _public_compatibility_obligations(
                acceptance_obligations[feature_id]
            )
        )
        normalized_description = description.replace("\\", "/").casefold()
        if re.search(r"\btests?\b", normalized_description):
            feature_paths[feature_id].extend(
                path
                for path in (
                    getattr(world, "_cooperbench_public_test_paths", ()) or ()
                )
                if path not in feature_paths[feature_id]
            )
        explicit = [
            path
            for path in feature_paths[feature_id]
            if path.casefold() in normalized_description
            and path not in (
                getattr(world, "_cooperbench_public_test_paths", ()) or ()
            )
        ]
        # Cooper descriptions normally publish a "Files Modified" surface.
        # When they do, every named file is part of the feature delivery—not
        # optional context. A task without explicit paths keeps one ranked
        # implementation target rather than requiring its whole context set.
        required_feature_paths[feature_id] = list(
            explicit or feature_paths[feature_id][:1]
        )
    # Capture the whole public import surface before either member edits it.
    # Executable baseline controls must not import candidate helper modules.
    # Keep source out of serialized SDL/trajectory and limit review prompts to
    # the feature's required files. No evaluator-only assets are loaded here.
    artifacts_by_path = {
        str(getattr(artifact, "linked_file_path", "") or "").replace(
            "\\", "/"
        ): artifact
        for artifact in (getattr(world, "product_artifacts", {}) or {}).values()
        if str(getattr(artifact, "linked_file_path", "") or "")
        and str(getattr(artifact, "artifact_type", "")) != "issue"
    }
    baseline_files: dict[str, str] = {}
    for path in sorted(artifacts_by_path):
        artifact = artifacts_by_path.get(path)
        if artifact is None:
            continue
        content = getattr(artifact, "mainline_content", None)
        if content is None:
            content = getattr(artifact, "content", "")
        baseline_files[path] = str(content or "")
    world.__dict__["_cooperbench_public_baseline_files"] = baseline_files
    overlap = sorted(set(feature_paths[_FEATURE_IDS[0]]) & set(feature_paths[_FEATURE_IDS[1]]))
    # Actor desks isolate concurrent edits. Shared paths therefore remain an
    # integration concern; they must not turn the two-person unit into a
    # single-file serial queue before either owner can implement their feature.
    world.__dict__["_cooperbench_issue_predecessor"] = {}
    from .visibility import is_strict_coop
    strict = is_strict_coop(world)
    from .work_schedule import compressed_schedule_receipt, install_compressed_schedule
    schedule = compressed_schedule_receipt(world)
    if schedule is None:
        schedule = install_compressed_schedule(world)
    state = {
        "schema_version": "orgenv_cooperbench_two_person_sdl_v88",
        "information_policy": "owner_private_explicit_share_read" if strict else "shared_briefs_organizational_diagnostic",
        "initial_briefs_owner_private": strict,
        "workspace_policy": "shared_organization_working_tree",
        "standard_coop_information_boundary": False,
        "decision_schedule": schedule,
        "phase": "delivery",
        "feature_owners": owners,
        "feature_paths": feature_paths,
        "required_feature_paths": required_feature_paths,
        "acceptance_obligations": acceptance_obligations,
        "compatibility_obligations": compatibility_obligations,
        "overlapping_paths": overlap,
        "serialization": "parallel_actor_desks_with_integration_replay",
        "conflict_recovery": "abandon_stale_head_and_pending_sync_main_reimplement_and_redeliver",
        "semantic_review_policy": {
            "max_blocking_rounds": 3,
            "on_exhaustion": "continue_fail_closed_until_peer_approval",
            "execution": "peer_public_behavior_probes_v30_pair_integration_scope",
            "contract_source": "complete_public_request_with_source_located_index",
            "probe_repair": "immutable_unaffected_rows_and_targeted_interaction_replacements",
            "provider_failure_resolution": "route_only_same_plan_prior_peer_validated_failures",
            "probe_defect_resolution": "focused_actual_peer_adjudication_before_check_rewrite",
            "unclassified_failure_resolution": "adjudicate_before_owner_repair_routing",
            "preaccept_repair_execution": "replay_current_peer_validated_failures_and_prior_passes",
            "validated_failure_feedback": "exact_probe_result_validity_receipt_public_basis_join",
            "repair_feedback_authority": "host_validated_execution_first_peer_hypotheses_quarantined",
            "repair_observable_dataflow": "failed_assertion_ast_dependency_slice",
            "repair_expected_transform": "host_authorized_ast_operator_semantics",
            "probe_constraint_feedback": "rejected_row_exact_error_and_one_bounded_same_action_retry",
            "observer_defect_feedback": "same_callback_call_returned_output_measurement",
            "ordered_source_stage_feedback": "public_named_stage_requires_invoked_operation_witness",
            "rejected_repair_memory": "last_four_source_fingerprints_last_two_in_editor_prompt",
            "review_revival": "fresh_post_rejection_passing_ci_required",
            "local_stage_helper_witness": "reachable_body_requires_concrete_transform_call",
            "cross_feature_interaction_feedback": (
                "visible_merged_overlap_same_call_and_one_bounded_plan_retry"
            ),
            "opt_in_compatibility_scope": "same_disabled_call_form_unless_public_cross_mode_equality",
            "repair_path_resolution": "peer_named_path_within_ranked_public_component_surface",
            "editor_response_repair": "one_line_array_feedback_repair_after_confirmed_provider_response",
            "review_input_projection": "ast_compact_checks_v1",
            "large_source_projection": "changed_hunks_public_symbols_exact_line_numbers_v1",
            "delivery_realization_separation": "technical_dual_attestation_export_with_b3_outcome_recorded_separately",
            "source_reference_resolution": "exact_prefix_bounded_reference_coverage_v2",
        },
        "attestations": {},
        "protocol_authorship": {},
        "verification_failures": [],
        "frozen_digest": None,
        "frozen_tick": None,
    }
    world.__dict__["_cooperbench_sdl_state"] = state
    return state


def feature_delivery_coverage(world: Any) -> dict[str, dict[str, Any]]:
    """Expose per-owner required-path coverage for live SDL diagnostics."""

    state = getattr(world, "_cooperbench_sdl_state", {}) or {}
    owners = state.get("feature_owners") or {}
    required_by_feature = state.get("required_feature_paths") or {}
    artifacts = getattr(world, "product_artifacts", {}) or {}
    patches = getattr(world, "patches", {}) or {}
    coordination = world.__dict__.get("_cooperbench_coordination_required")
    coverage: dict[str, dict[str, Any]] = {}
    for feature_id in _FEATURE_IDS:
        owner = str(owners.get(feature_id) or "")
        required = [
            str(path).replace("\\", "/")
            for path in required_by_feature.get(feature_id, [])
            if str(path)
        ]
        scoped_patch_ids: set[str] | None = None
        if (
            isinstance(coordination, dict)
            and coordination.get("kind") == "coordination_conflict"
            and coordination.get("status") == "repairing"
            and str(coordination.get("feature_id") or "") == feature_id
            and str(coordination.get("owner_id") or "") == owner
        ):
            from .source_views import actor_desk_snapshot

            scoped_patch_ids = set(actor_desk_snapshot(world, owner).patch_ids)
        covered: set[str] = set()
        for patch_id, patch in patches.items():
            if scoped_patch_ids is not None and str(patch_id) not in scoped_patch_ids:
                continue
            if str(getattr(patch, "actor_id", "") or "") != owner:
                continue
            if _status(getattr(patch, "validation_status", "")) not in {
                "accepted",
                "applied",
                "merged",
            }:
                continue
            if feature_id not in {
                str(item)
                for item in (getattr(patch, "related_issue_ids", []) or [])
            }:
                continue
            artifact = artifacts.get(
                str(getattr(patch, "target_object_id", "") or "")
            )
            path = str(
                getattr(artifact, "linked_file_path", "") or ""
            ).replace("\\", "/")
            if path:
                covered.add(path)
        coverage[feature_id] = {
            "owner": owner,
            "required_paths": required,
            "covered_paths": [path for path in required if path in covered],
            "missing_paths": [path for path in required if path not in covered],
            "complete": bool(required)
            and all(path in covered for path in required),
        }
    return coverage


def joint_delivery_digest(world: Any) -> str:
    """Hash the integrated mainline files in the two public feature surfaces."""

    from .source_views import mainline_snapshot, source_views_enabled
    if source_views_enabled(world):
        return mainline_snapshot(world).snapshot_id
    rows: list[dict[str, Any]] = []
    for artifact_id, artifact in sorted(
        (getattr(world, "product_artifacts", {}) or {}).items()
    ):
        path = str(getattr(artifact, "linked_file_path", "") or "").replace("\\", "/")
        if not path:
            continue
        revision = int(getattr(artifact, "mainline_revision", 0) or 0)
        content = getattr(artifact, "mainline_content", None)
        if content is None:
            content = getattr(artifact, "content", "")
        rows.append(
            {
                "artifact_id": str(artifact_id),
                "path": path,
                "revision": revision,
                "content_sha256": hashlib.sha256(
                    str(content or "").encode("utf-8")
                ).hexdigest(),
            }
        )
    raw = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def peer_review_coverage(world: Any) -> dict[str, bool]:
    """Report whether every feature landed through its peer's approval."""

    state = getattr(world, "_cooperbench_sdl_state", {}) or {}
    owners = state.get("feature_owners") or {}
    members = {str(item) for item in owners.values() if str(item)}
    coverage = {feature_id: False for feature_id in _FEATURE_IDS}
    repo = getattr(getattr(world, "repo_system", None), "repo", None)
    for pr in (getattr(repo, "pull_requests", {}) or {}).values():
        if _status(getattr(pr, "status", "")) != "merged":
            continue
        author = str(getattr(pr, "author_id", "") or "")
        approvers = {str(item) for item in (getattr(pr, "approved_by", []) or [])}
        for feature_id in _linked_feature_ids(pr):
            owner = str(owners.get(feature_id) or "")
            peer_approvals = approvers & (members - {author})
            if owner and author == owner and peer_approvals:
                if getattr(world, "_cooperbench_delivery_focus", False):
                    from .semantic_review import current_semantic_approval

                    if not current_semantic_approval(world, feature_id, pr):
                        continue
                coverage[feature_id] = True
    return coverage


def peer_review_ready_coverage(world: Any) -> dict[str, bool]:
    """Report current owner heads that completed a real peer verdict.

    Unlike endpoint ``peer_review_coverage``, this includes an approved but not
    yet merged final PR. It is used only to protect the short decision window
    between the last review and the last merge.
    """

    state = getattr(world, "_cooperbench_sdl_state", {}) or {}
    owners = state.get("feature_owners") or {}
    members = {str(item) for item in owners.values() if str(item)}
    coverage = {feature_id: False for feature_id in _FEATURE_IDS}
    repo = getattr(getattr(world, "repo_system", None), "repo", None)
    for pr in (getattr(repo, "pull_requests", {}) or {}).values():
        if _status(getattr(pr, "status", "")) not in {"approved", "merged"}:
            continue
        author = str(getattr(pr, "author_id", "") or "")
        approvers = {str(item) for item in (getattr(pr, "approved_by", []) or [])}
        for feature_id in _linked_feature_ids(pr):
            owner = str(owners.get(feature_id) or "")
            peer_approvals = approvers & (members - {author})
            if owner and author == owner and peer_approvals:
                if getattr(world, "_cooperbench_delivery_focus", False):
                    from .semantic_review import current_semantic_approval

                    if not current_semantic_approval(world, feature_id, pr):
                        continue
                coverage[feature_id] = True
    return coverage


def friction_evidence(world: Any) -> list[str]:
    """Return distinct public workflow failures that can ground a protocol."""

    from environments.org_env.product.patch_validator import is_infrastructure_rejection

    refs: list[str] = []
    for patch_id, patch in (getattr(world, "patches", {}) or {}).items():
        if (_status(getattr(patch, "validation_status", "")) == "rejected"
                and not is_infrastructure_rejection(patch)):
            refs.append(f"patch_rejected:{patch_id}")
    repo = getattr(getattr(world, "repo_system", None), "repo", None)
    for ci_id, ci in (getattr(repo, "ci_runs", {}) or {}).items():
        if _status(getattr(ci, "status", "")) == "failed":
            refs.append(f"ci_failed:{ci_id}")
    for pr_id, pr in (getattr(repo, "pull_requests", {}) or {}).items():
        for index, _comment in enumerate(getattr(pr, "requested_changes", []) or [], start=1):
            refs.append(f"review_changes:{pr_id}:{index}")
    state = getattr(world, "_cooperbench_sdl_state", {}) or {}
    for index, item in enumerate(state.get("verification_failures") or [], start=1):
        digest = str((item or {}).get("digest") or "")[:12]
        refs.append(f"joint_verification_failed:{index}:{digest}")
    refs = list(dict.fromkeys(refs))
    from .visibility import is_strict_coop
    if is_strict_coop(world):
        from .pair_protocol import common_public_protocol_friction
        members = sorted(set((state.get("feature_owners") or {}).values()))
        return common_public_protocol_friction(world, refs, members)
    return refs


def adopted_pair_protocols(world: Any) -> list[Any]:
    manager = getattr(world, "proposal_manager", None)
    matches = []
    for spec in (getattr(manager, "protocol_specs", {}) or {}).values():
        declared_actions = {
            str(item).strip().casefold().replace("-", "_").replace(" ", "_")
            for item in (getattr(spec, "affected_actions", []) or [])
        }
        if _status(getattr(spec, "status", "")) != "adopted":
            continue
        if (
            str(getattr(spec, "family", "") or "") == "cooperbench_joint_delivery"
            or "verify_joint_delivery" in declared_actions
        ):
            matches.append(spec)
    return matches


def main_b3_lifecycle_enabled(world: Any) -> bool:
    """Whether this pair is running the main experiment's B3 mechanism."""

    return bool(
        getattr(world, "__dict__", {}).get("_cooperbench_delivery_focus", False)
        and getattr(world, "__dict__", {}).get(
            "_cooperbench_main_b3_lifecycle", False
        )
    )


def formed_main_b3_protocols(world: Any) -> list[Any]:
    """Return protocols that completed the main B3 lifecycle in this pair.

    A registry-only environment seed is not enough.  The rule must come from a
    real Proposal, be grounded in recurrence (episodes/events or clustered
    wishes), be proposed by one member, and be adopted by both members through
    ProposalManager.  This keeps ``proto_review_before_merge`` and
    ``proto_experiment_logging`` out of Cooper's B3 realization claim while
    retaining the main experiment's catalogue- and wish-driven formation paths.
    """

    manager = getattr(world, "proposal_manager", None)
    proposals = getattr(manager, "proposals", {}) or {}
    state = getattr(world, "_cooperbench_sdl_state", {}) or {}
    members = {
        str(agent_id)
        for agent_id in (state.get("feature_owners") or {}).values()
        if str(agent_id)
    }
    if len(members) != 2:
        return []

    from environments.org_env.proposals.manager import recurrence_behind

    formed = []
    registry = getattr(world, "protocol_registry", None)
    if registry is None:
        return []
    for spec in (getattr(manager, "protocol_specs", {}) or {}).values():
        if _status(getattr(spec, "status", "")) != "adopted":
            continue
        proposal = proposals.get(
            str(getattr(spec, "created_from_proposal_id", "") or "")
        )
        if proposal is None or getattr(proposal, "proposal_type", "") != "protocol_proposal":
            continue
        if str(getattr(proposal, "proposer_agent_id", "") or "") not in members:
            continue
        if not members <= {
            str(agent_id)
            for agent_id in (getattr(spec, "adopted_by", []) or [])
        }:
            continue
        if recurrence_behind(proposal) < 2:
            continue
        mirror_id = (
            world._registry_mirror_id(str(getattr(spec, "protocol_id", "") or ""))
            if callable(getattr(world, "_registry_mirror_id", None))
            else f"proto_spec_{str(getattr(spec, 'protocol_id', '') or '').split('_')[-1]}"
        )
        if mirror_id not in (getattr(registry, "protocols", {}) or {}):
            continue
        formed.append(spec)
    return formed


def pending_main_b3_protocols(world: Any) -> list[Any]:
    """Return recurrence-qualified pair-authored protocol proposals in review."""

    manager = getattr(world, "proposal_manager", None)
    state = getattr(world, "_cooperbench_sdl_state", {}) or {}
    members = {
        str(agent_id)
        for agent_id in (state.get("feature_owners") or {}).values()
        if str(agent_id)
    }
    if len(members) != 2:
        return []
    from environments.org_env.proposals.manager import recurrence_behind

    pending = []
    for proposal in (getattr(manager, "proposals", {}) or {}).values():
        if getattr(proposal, "proposal_type", "") != "protocol_proposal":
            continue
        if _status(getattr(proposal, "status", "")) not in {
            "draft",
            "under_review",
            "approved",
        }:
            continue
        if str(getattr(proposal, "proposer_agent_id", "") or "") not in members:
            continue
        required = {
            str(agent_id)
            for agent_id in (
                getattr(proposal, "approval_required_from", []) or []
            )
            if str(agent_id)
        }
        if not members <= required or recurrence_behind(proposal) < 2:
            continue
        pending.append(proposal)
    return pending


def _main_b3_emergence_levels(world: Any, specs: list[Any]) -> dict[str, str]:
    registry = getattr(world, "protocol_registry", None)
    levels: dict[str, str] = {}
    for spec in specs:
        spec_id = str(getattr(spec, "protocol_id", "") or "")
        mirror_id = (
            world._registry_mirror_id(spec_id)
            if callable(getattr(world, "_registry_mirror_id", None))
            else f"proto_spec_{spec_id.split('_')[-1]}"
        )
        if registry is None or mirror_id not in (getattr(registry, "protocols", {}) or {}):
            levels[spec_id] = "none"
            continue
        try:
            levels[spec_id] = str(registry.classify_emergence(mirror_id))
        except Exception:
            levels[spec_id] = "none"
    return levels


def protocol_friction_details(world: Any, evidence: list[str]) -> dict[str, Any]:
    """Resolve cited objects even when their events are outside the recent window."""
    repo = getattr(getattr(world, "repo_system", None), "repo", None)
    details: dict[str, Any] = {}
    for ref in evidence:
        kind, _, object_id = ref.partition(":")
        if kind == "review_changes":
            pr_id, _, number = object_id.rpartition(":")
            pr = (getattr(repo, "pull_requests", {}) or {}).get(pr_id)
            comments = list(getattr(pr, "requested_changes", []) or [])
            index = int(number) - 1 if number.isdigit() else -1
            details[ref] = comments[index] if 0 <= index < len(comments) else "unavailable"
        elif kind == "ci_failed":
            ci = (getattr(repo, "ci_runs", {}) or {}).get(object_id)
            details[ref] = list(getattr(ci, "failure_reasons", []) or [])
        elif kind == "patch_rejected":
            patch = (getattr(world, "patches", {}) or {}).get(object_id)
            details[ref] = {key: getattr(patch, key, None) for key in (
                "validation_reason", "rejection_reason", "validation_errors", "file_path")}
        elif kind == "joint_verification_failed":
            number = object_id.partition(":")[0]
            failures = (getattr(world, "_cooperbench_sdl_state", {}) or {}).get("verification_failures", [])
            index = int(number) - 1 if number.isdigit() else -1
            details[ref] = failures[index] if 0 <= index < len(failures) else "unavailable"
        elif kind == "peer_reviewed_delivery":
            state = getattr(world, "_cooperbench_sdl_state", {}) or {}
            owner = str((state.get("feature_owners") or {}).get(object_id) or "")
            members = {
                str(item)
                for item in (state.get("feature_owners") or {}).values()
                if str(item)
            }
            delivered = next((
                (pr_id, pr)
                for pr_id, pr in (getattr(repo, "pull_requests", {}) or {}).items()
                if _status(getattr(pr, "status", "")) == "merged"
                and str(getattr(pr, "author_id", "") or "") == owner
                and object_id in _linked_feature_ids(pr)
                and ({str(item) for item in (getattr(pr, "approved_by", []) or [])}
                     & (members - {owner}))
            ), None)
            details[ref] = (
                {
                    "feature_id": object_id,
                    "owner": owner,
                    "pr_id": delivered[0],
                    "peer_approvers": sorted(
                        {str(item) for item in (getattr(delivered[1], "approved_by", []) or [])}
                        & (members - {owner})
                    ),
                    "ci_passed": bool(getattr(delivered[1], "ci_passed", False)),
                    "merged_tick": getattr(delivered[1], "merged_tick", None),
                }
                if delivered is not None
                else "unavailable"
            )
    return _redact(details, max_chars=None)


def protocol_deliberation_evidence(world: Any) -> list[str]:
    """Ground main-B3 deliberation in failures or two completed peer cycles."""

    evidence = friction_evidence(world)
    if main_b3_lifecycle_enabled(world) and all(peer_review_coverage(world).values()):
        evidence.extend(
            f"peer_reviewed_delivery:{feature_id}" for feature_id in _FEATURE_IDS
        )
    return list(dict.fromkeys(evidence))


def protocol_deliberation_digest(world: Any, evidence: list[str]) -> str:
    from .pair_protocol import BINDING
    material = {"binding": BINDING, "evidence": protocol_friction_details(world, evidence)}
    return hashlib.sha256(json.dumps(material, sort_keys=True, default=str).encode()).hexdigest()


def protocol_proposal_needed(world: Any, agent_id: str) -> bool:
    state = getattr(world, "_cooperbench_sdl_state", {}) or {}
    if agent_id not in set((state.get("feature_owners") or {}).values()):
        return False
    evidence = protocol_deliberation_evidence(world)
    if (
        len(evidence) < 2
        or adopted_pair_protocols(world)
        or formed_main_b3_protocols(world)
        or pending_cooperbench_protocol(world)
    ):
        return False
    decision = (state.get("protocol_deliberations") or {}).get(agent_id) or {}
    return not (decision.get("applicable") is False
                and decision.get("evidence_digest") == protocol_deliberation_digest(world, evidence))


def _pair_protocol_execution_evidence(
    world: Any, protocol_id: str
) -> list[str]:
    """Return action-bound evidence for this pair protocol.

    Generic ``note_protocol_use`` performs a keyword match over protocol prose.
    A prior title contained the word "integration", so every unrelated CI
    attempt inflated ``use_count`` even though ``verify_joint_delivery`` never
    ran.  Pair-local realization therefore trusts only a successful attestation
    that names the adopted protocol, or an enforcement event that explicitly
    says it governed ``verify_joint_delivery``.
    """

    state = getattr(world, "_cooperbench_sdl_state", {}) or {}
    refs: list[str] = []
    for agent_id, attestation in (state.get("attestations") or {}).items():
        if protocol_id not in {
            str(item) for item in (attestation or {}).get("protocol_ids", [])
        }:
            continue
        if not bool((attestation or {}).get("ok", True)):
            continue
        refs.append(
            "joint_attestation:"
            f"{agent_id}:{int((attestation or {}).get('tick') or 0)}"
        )
    for index, event in enumerate(getattr(world, "events", []) or []):
        if not isinstance(event, Mapping):
            continue
        if str(event.get("type") or "") != "protocol_enforcement_event":
            continue
        if str(event.get("protocol_id") or "") != protocol_id:
            continue
        if str(event.get("governed_action") or "") != "verify_joint_delivery":
            continue
        refs.append(
            str(event.get("event_id") or f"pair_protocol_enforcement:{index}")
        )
    return list(dict.fromkeys(refs))


def _main_b3_action_bound_execution_evidence(world: Any, spec: Any) -> list[str]:
    """Return successful action receipts governed by a formed main-B3 rule.

    Main-experiment protocols can govern review/integration actions without
    being the pair adapter's special fresh-attestation binding.  The old Cooper
    receipt reported those rules as formed but ``operative=0`` even when the
    execution layer had emitted a protocol-use event tied to a successful
    action.  At the same time, generic keyword/PR counters are too weak to
    credit: they may be emitted without an agent action and caused the false
    positives that the pair-specific evidence filter was written to prevent.

    Require all three identities to join: the protocol's declared action, an
    action-bound use event, and the corresponding successful action result.
    """

    declared_actions = {
        str(item).strip().casefold().replace("-", "_").replace(" ", "_")
        for item in (getattr(spec, "affected_actions", []) or [])
        if str(item).strip()
    }
    successful: dict[str, str] = {}
    for result in (getattr(world, "action_log", []) or []):
        if isinstance(result, Mapping):
            action_id = str(result.get("action_id") or "")
            action_type = str(result.get("action_type") or "")
            success = result.get("success") is True
        else:
            action_id = str(getattr(result, "action_id", "") or "")
            action_type = str(getattr(result, "action_type", "") or "")
            success = getattr(result, "success", None) is True
        if action_id and success:
            successful[action_id] = action_type.casefold().replace("-", "_").replace(" ", "_")

    protocol_id = str(getattr(spec, "protocol_id", "") or "")
    refs: list[str] = []
    for event in (getattr(world, "events", []) or []):
        if not isinstance(event, Mapping):
            continue
        if event.get("type") != "protocol_use_event" or str(event.get("protocol_id") or "") != protocol_id:
            continue
        action_id = str(event.get("action_id") or "")
        governed = str(event.get("governed_action") or "").casefold().replace("-", "_").replace(" ", "_")
        if (
            action_id
            and governed in declared_actions
            and successful.get(action_id) == governed
        ):
            refs.append(action_id)
    return list(dict.fromkeys(refs))


def protocol_realization(world: Any) -> dict[str, Any]:
    """Separate delivery eligibility, workflow participation, and rule effect."""

    evidence = friction_evidence(world)
    operative: list[str] = []
    changed_admissibility: list[str] = []
    workflow_bound: list[str] = []
    adopted: list[str] = []
    execution_evidence: dict[str, list[str]] = {}
    from .pair_protocol import authored_bindings
    bound_specs = {spec.protocol_id for spec in authored_bindings(world, adopted_pair_protocols(world))}
    for spec in adopted_pair_protocols(world):
        spec_id = str(getattr(spec, "protocol_id", "") or "")
        adopted.append(str(spec_id))
        lineage = {
            str(item)
            for item in (
                list(getattr(spec, "source_episode_ids", []) or [])
                + list(getattr(spec, "problem_evidence", []) or [])
            )
            if str(item).strip()
        }
        bound_evidence = _pair_protocol_execution_evidence(world, spec_id)
        execution_evidence[spec_id] = bound_evidence
        if len(lineage) >= 2 and bound_evidence:
            workflow_bound.append(spec_id)
        # The unconditional SDL already checks digest, peer approval and
        # public green. Attaching a protocol id to those checks is evidence of
        # workflow participation, not evidence that the rule changed a choice.
        # No existing generic counter/attestation is upgraded to this receipt.
        if spec_id in bound_specs and len(lineage) >= 2 and any(
            isinstance(event, Mapping)
            and event.get("type") == "protocol_binding_effect"
            and event.get("protocol_id") == spec_id
            and event.get("binding_verified") is True
            and event.get("outcome_ref")
            and event.get("governed_action") == "joint_freeze"
            for event in (getattr(world, "events", []) or [])
        ):
            operative.append(spec_id)
            if any(isinstance(event, Mapping)
                   and event.get("type") == "protocol_binding_effect"
                   and event.get("protocol_id") == spec_id
                   and event.get("binding_verified") is True
                   and event.get("without_rule_allowed") is True
                   and event.get("with_rule_allowed") is False
                   and event.get("outcome_ref")
                   for event in (getattr(world, "events", []) or [])):
                changed_admissibility.append(spec_id)
    state = getattr(world, "_cooperbench_sdl_state", {}) or {}
    members = set((state.get("feature_owners") or {}).values())
    decisions = state.get("protocol_deliberations") or {}
    evidence_digest = protocol_deliberation_digest(world, evidence)
    non_applicable = sorted(member for member in members if
        (decisions.get(member) or {}).get("applicable") is False
        and (decisions.get(member) or {}).get("evidence_digest") == evidence_digest)
    # Friction requires deliberation, not fabrication of an irrelevant rule.
    # Both members must independently decline this exact evidence/binding set.
    jointly_declined = len(members) == 2 and len(non_applicable) == 2 and not adopted
    main_b3 = main_b3_lifecycle_enabled(world)
    formed_specs = formed_main_b3_protocols(world) if main_b3 else []
    formed_ids = [str(getattr(spec, "protocol_id", "") or "") for spec in formed_specs]
    emergence_levels = _main_b3_emergence_levels(world, formed_specs)
    main_b3_action_bound_ids: list[str] = []
    if main_b3:
        for spec in formed_specs:
            spec_id = str(getattr(spec, "protocol_id", "") or "")
            bound_evidence = _main_b3_action_bound_execution_evidence(world, spec)
            if not bound_evidence:
                continue
            main_b3_action_bound_ids.append(spec_id)
            execution_evidence[spec_id] = list(
                dict.fromkeys([*execution_evidence.get(spec_id, []), *bound_evidence])
            )
            if spec_id not in workflow_bound:
                workflow_bound.append(spec_id)
            if spec_id not in operative:
                operative.append(spec_id)
    wish_formed_ids = [
        str(getattr(spec, "protocol_id", "") or "")
        for spec in formed_specs
        if list(getattr(spec, "source_wish_ids", []) or [])
    ]
    # The main B3 treatment always owes an institution.  The former
    # friction-only rule allowed a clean task to submit with no protocol at all,
    # which is behaviorally B2 regardless of the condition label.
    requires_protocol = (
        True if main_b3 else len(evidence) >= 2 and not jointly_declined
    )
    protocol_formation_satisfied = bool(formed_specs)
    # A formed rule that never governs one of its own declared actions is a
    # durable proposal, not yet a behaviorally operative institution.  Main B3
    # therefore needs the exact three-way join produced above: formed spec,
    # action-bound protocol-use event, and successful action-log receipt.
    protocol_gate_satisfied = (
        bool(main_b3_action_bound_ids)
        if main_b3
        else bool(not requires_protocol or workflow_bound)
    )
    realized_b3 = (
        bool(main_b3_action_bound_ids) if main_b3 else bool(operative)
    )
    reported_adopted = list(dict.fromkeys([*adopted, *formed_ids]))
    return {
        "qualifying_friction": len(evidence),
        "friction_evidence": evidence,
        "requires_protocol": requires_protocol,
        "protocol_deliberation_required": bool(main_b3 or len(evidence) >= 2),
        "non_applicable_member_ids": non_applicable,
        "jointly_declined_supported_binding": jointly_declined,
        "protocol_formation_satisfied": protocol_formation_satisfied,
        "protocol_gate_satisfied": protocol_gate_satisfied,
        "workflow_bound_protocols": len(workflow_bound),
        "workflow_bound_protocol_ids": workflow_bound,
        "adopted_protocols": len(reported_adopted),
        "adopted_protocol_ids": reported_adopted,
        "operative_protocols": len(operative),
        "operative_protocol_ids": operative,
        "incremental_effect_protocol_ids": changed_admissibility,
        "execution_evidence": execution_evidence,
        "main_b3_lifecycle_enabled": main_b3,
        "formed_main_b3_protocols": len(formed_ids),
        "formed_main_b3_protocol_ids": formed_ids,
        "wish_formed_main_b3_protocol_ids": wish_formed_ids,
        "main_b3_emergence_levels": emergence_levels,
        "main_b3_action_bound_protocol_ids": main_b3_action_bound_ids,
        "realized_b3": realized_b3,
        "realization_basis": (
            "main_b3_recurrent_proposal_pair_approval_protocolspec_registry_and_action_bound_success_receipts"
            if main_b3
            else "verified_authored_rule_binding_not_baseline_sdl"
        ),
    }


def pending_cooperbench_protocol(world: Any) -> Any | None:
    manager = getattr(world, "proposal_manager", None)
    for proposal in (getattr(manager, "proposals", {}) or {}).values():
        if str(getattr(proposal, "family", "") or "") != "cooperbench_joint_delivery":
            continue
        if _status(getattr(proposal, "status", "")) in {
            "draft",
            "under_review",
            "approved",
            "adopted",
        }:
            return proposal
    return None


def submit_cooperbench_protocol_proposal(
    world: Any, agent_id: str, *, tick: int
) -> Any:
    """Create a grounded pair protocol only after an agent selects the action."""

    existing = pending_cooperbench_protocol(world)
    if existing is not None:
        return existing
    evidence = protocol_deliberation_evidence(world)
    if len(evidence) < 2:
        raise ValueError("cooperbench_protocol_requires_repeated_friction")
    from environments.org_env.proposals.objects import Proposal

    manager = getattr(world, "proposal_manager", None)
    if manager is None:
        raise ValueError("cooperbench_proposal_manager_missing")
    members = sorted(
        set(
            (getattr(world, "_cooperbench_sdl_state", {}) or {})
            .get("feature_owners", {})
            .values()
        )
    )
    from .pair_protocol import draft_pair_protocol
    authored = draft_pair_protocol(world, str(agent_id), evidence, members)
    world._cooperbench_sdl_state.setdefault("protocol_deliberations", {})[str(agent_id)] = {
        "applicable": authored["applicable"], "tick": int(tick),
        "evidence_digest": protocol_deliberation_digest(world, evidence),
        "source_event_ids": list(dict.fromkeys(authored["source_event_ids"])),
        "summary": authored["summary"], "target_problem": authored["target_problem"],
    }
    if not authored["applicable"]:
        return None
    proposal = Proposal(
        proposal_id=manager.next_id("proposal"),
        proposal_type="protocol_proposal",
        title=authored["title"],
        summary=authored["summary"],
        proposer_agent_id=str(agent_id),
        source_event_ids=list(dict.fromkeys(authored["source_event_ids"])),
        target_problem=authored["target_problem"],
        proposed_solution=authored["proposed_solution"],
        required_actions=["verify_joint_delivery"],
        required_participants=members,
        affected_agents=members,
        affected_objects=evidence,
        expected_benefits=["detect integration regressions before CooperBench submission"],
        expected_costs=["one public verification action per member"],
        risks=["public tests may not cover the private feature evaluator"],
        family="cooperbench_joint_delivery",
        created_at_tick=int(tick),
        updated_at_tick=int(tick),
    )
    submitted = world._submit_proposal(proposal)
    if submitted is None or _status(getattr(submitted, "status", "")) == "rejected":
        reason = str(getattr(submitted, "rejection_reason", "") or "rejected")
        raise ValueError(f"cooperbench_protocol_proposal_rejected:{reason}")
    # The standard role router has no natural founder in a two-engineer unit.
    # Pair governance therefore requires both actual members, preserving the
    # review-latency and two-distinct-approver gates in ProposalManager.
    submitted.approval_required_from = members
    submitted.approved_by = []
    submitted.status = "under_review"
    submitted.updated_at_tick = int(tick)
    world._cooperbench_sdl_state.setdefault("protocol_authorship", {})[submitted.proposal_id] = {
        "origin": "agent_authored_bounded_binding", "agent_id": str(agent_id),
        "binding": authored["binding"], "tick": int(tick),
        "solution_digest": hashlib.sha256(authored["proposed_solution"].encode()).hexdigest(),
    }
    return submitted


def _record_pair_protocol_enforcement(
    world: Any,
    *,
    agent_id: str,
    tick: int,
    digest: str,
    reason: str,
) -> None:
    """Credit an adopted pair rule only when it actually blocks an invalid freeze."""

    for spec in adopted_pair_protocols(world):
        protocol_id = str(getattr(spec, "protocol_id", "") or "")
        event_id = (
            "cooperbench_protocol_enforcement@"
            f"t{int(tick)}:{agent_id}:{reason}"
        )
        violation_event_id = event_id.replace(
            "enforcement", "violation", 1
        )
        spec.enforcement_count = int(
            getattr(spec, "enforcement_count", 0) or 0
        ) + 1
        spec.violation_count = int(
            getattr(spec, "violation_count", 0) or 0
        ) + 1
        enforcement_ids = getattr(spec, "enforcement_event_ids", None)
        if not isinstance(enforcement_ids, list):
            enforcement_ids = []
            spec.enforcement_event_ids = enforcement_ids
        violation_ids = getattr(spec, "violation_event_ids", None)
        if not isinstance(violation_ids, list):
            violation_ids = []
            spec.violation_event_ids = violation_ids
        enforcement_ids.append(event_id)
        violation_ids.append(violation_event_id)
        events = getattr(world, "events", None)
        if isinstance(events, list):
            events.append(
                {
                    "type": "protocol_violation_event",
                    "event_id": violation_event_id,
                    "protocol_id": protocol_id,
                    "agent_id": str(agent_id),
                    "tick": int(tick),
                    "governed_action": "verify_joint_delivery",
                    "digest": str(digest),
                    "reason": str(reason),
                    "auto": True,
                }
            )
            events.append(
                {
                    "type": "protocol_enforcement_event",
                    "event_id": event_id,
                    "protocol_id": protocol_id,
                    "agent_id": str(agent_id),
                    "tick": int(tick),
                    "governed_action": "verify_joint_delivery",
                    "digest": str(digest),
                    "reason": str(reason),
                    "blocked": True,
                }
            )
        mirror = getattr(world, "_mirror_protocol_event", None)
        if callable(mirror):
            # Registry enforcement must reference the matching violation;
            # mirror the two sides of this one blocked incident in order.
            mirror(
                spec,
                "violate",
                int(tick),
                str(digest),
                str(agent_id),
            )
            mirror(
                spec,
                "enforce",
                int(tick),
                str(digest),
                str(agent_id),
                blocked=True,
            )


def record_joint_attestation(
    world: Any,
    agent_id: str,
    *,
    digest: str,
    verification: Mapping[str, Any],
    tick: int,
) -> bool:
    """Record one member's public-test attestation and freeze after both agree."""

    state = getattr(world, "_cooperbench_sdl_state", None)
    if not isinstance(state, dict):
        raise ValueError("cooperbench_sdl_not_initialized")
    agent = str(agent_id)
    members = set((state.get("feature_owners") or {}).values())
    if agent not in members:
        raise ValueError("cooperbench_attestor_not_in_pair")
    current = joint_delivery_digest(world)
    ok = bool(verification.get("available") and verification.get("ok"))
    from .replay_workflow import workflow_enabled
    integrated_workflow = workflow_enabled(world)
    behavior_replay = verification.get("behavior_replay")
    if integrated_workflow:
        from .joint_probe_replay import current_accepted_probe_replay_matches
        ok = ok and isinstance(behavior_replay, dict) and current_accepted_probe_replay_matches(
            world, actor_id=agent, receipt=behavior_replay, require_all_features=True,
        )
    if digest != current or not ok:
        _record_pair_protocol_enforcement(
            world,
            agent_id=agent,
            tick=tick,
            digest=str(digest),
            reason=(
                "stale_digest"
                if digest != current
                else "public_regression_failed"
            ),
        )
        state.setdefault("verification_failures", []).append(
            {
                "agent_id": agent,
                "tick": int(tick),
                "digest": str(digest),
                "current_digest": current,
                "available": bool(verification.get("available")),
                "ok": bool(verification.get("ok")),
                "summary": str(verification.get("summary") or "")[:300],
            }
        )
        state["phase"] = "joint_verification"
        return False
    if not all(peer_review_coverage(world).values()):
        _record_pair_protocol_enforcement(
            world,
            agent_id=agent,
            tick=tick,
            digest=str(digest),
            reason="peer_review_incomplete",
        )
        raise ValueError("cooperbench_joint_attestation_requires_peer_review")
    state["phase"] = "joint_verification"
    from .pair_protocol import authored_bindings
    active_bindings = authored_bindings(world, adopted_pair_protocols(world))
    state.setdefault("attestations", {})[agent] = {
        "tick": int(tick),
        "digest": current,
        "available": True,
        "ok": True,
        "summary": str(verification.get("summary") or "")[:300],
        "protocol_ids": [
            str(getattr(spec, "protocol_id", "") or "")
            for spec in adopted_pair_protocols(world)
        ],
    }
    attestations = state["attestations"]
    if integrated_workflow:
        import copy
        attestations[agent]["behavior_replay"] = copy.deepcopy(behavior_replay)
        attestations[agent]["plan_set_digest"] = behavior_replay["plan_set_digest"]
    baseline_allowed = members <= set(attestations) and {
        str(attestations[item].get("digest") or "") for item in members
    } == {current}
    if integrated_workflow:
        baseline_allowed = baseline_allowed and all(
            isinstance(attestations[item].get("behavior_replay"), dict)
            and current_accepted_probe_replay_matches(
                world, actor_id=item, receipt=attestations[item]["behavior_replay"], require_all_features=True,
            )
            for item in members
        ) and {attestations[item].get("plan_set_digest") for item in members} == {behavior_replay["plan_set_digest"]}
    # Cooper task completion is technical: both owners reviewed and replayed
    # the current joint delivery.  B3 formation and realization remain observed
    # treatment outcomes, not an additional hidden benchmark requirement.
    jointly_attested = baseline_allowed
    for spec in active_bindings:
        floor = int(getattr(spec, "adopted_at_tick", 0) or 0)
        allowed = baseline_allowed and all(int(attestations[item].get("tick") or 0) >= floor
                                          for item in members)
        without_this_rule = baseline_allowed and all(
            int(attestations[item].get("tick") or 0) >= int(getattr(other, "adopted_at_tick", 0) or 0)
            for other in active_bindings if other.protocol_id != spec.protocol_id for item in members
        )
        event = {
            "type": "protocol_binding_effect", "protocol_id": spec.protocol_id,
            # The technical freeze is intentionally independent of protocol
            # realization.  Therefore a counterfactual showing that the rule
            # *would* have rejected stale attestations is not evidence that it
            # governed this successful freeze.  Credit the binding only when
            # both members actually satisfied its post-adoption requirement.
            "binding_verified": bool(allowed), "governed_action": "joint_freeze",
            "without_rule_allowed": without_this_rule,
            "with_rule_allowed": without_this_rule and allowed,
            "outcome_ref": f"joint_freeze:{current}:{int(tick)}:{agent}",
            "tick": int(tick), "agent_id": agent,
        }
        world.events.append(event)
    if jointly_attested:
        state["phase"] = "frozen"
        state["frozen_digest"] = current
        state["frozen_tick"] = int(tick)
        return True
    return False


def _redact(value: Any, *, key: str = "", max_chars: int | None = _MAX_TEXT) -> Any:
    folded = key.casefold()
    if any(secret in folded for secret in _SECRET_KEYS):
        return "[REDACTED]"
    if isinstance(value, Mapping):
        return {str(k): _redact(v, key=str(k), max_chars=max_chars) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_redact(item, max_chars=max_chars) for item in list(value)[:_MAX_ITEMS if max_chars is not None else None]]
    if isinstance(value, str):
        if max_chars is None:
            from relic.research.redaction import redact_sensitive_payload
            return redact_sensitive_payload(value, key=key)
        try:
            from relic.research.public_redaction import redact_public_text

            return redact_public_text(value, max_chars=max_chars)
        except Exception:
            return value[:max_chars]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _redact(value.to_dict(), max_chars=max_chars)
    if hasattr(value, "__dict__"):
        return _redact(value.__dict__, max_chars=max_chars)
    return str(value)[:max_chars]


class TrajectoryRecorder:
    """Append bounded incremental world evidence to one pair-local JSONL file."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._cursors = {"decisions": 0, "actions": 0, "events": 0, "policy": 0, "perceptions": 0}

    def _delta(self, world: Any, target: str, attribute: str) -> list[Any]:
        rows = list(getattr(world, attribute, []) or [])
        start = min(self._cursors[target], len(rows))
        self._cursors[target] = len(rows)
        return rows[start:]

    def capture(self, world: Any) -> dict[str, Any]:
        client = getattr(world, "llm_client", None)
        totals = getattr(client, "usage_totals", {}) or {}
        try:
            client_stats = dict(client.stats())
        except Exception:
            client_stats = {}
        repo = getattr(getattr(world, "repo_system", None), "repo", None)
        manager = getattr(world, "proposal_manager", None)
        row = {
            "schema_version": "orgenv_cooperbench_trajectory_v1",
            "model_call_session_id": getattr(world, "_cooperbench_model_call_session_id", None),
            "tick": int(getattr(world, "world_tick", 0) or 0),
            "decisions": self._delta(world, "decisions", "action_decisions"),
            "policy": self._delta(world, "policy", "policy_trace"),
            "perceptions": self._delta(world, "perceptions", "_cooperbench_perception_trace"),
            "actions": self._delta(world, "actions", "action_log"),
            "events": self._delta(world, "events", "events"),
            "sdl": getattr(world, "_cooperbench_sdl_state", {}) or {},
            "feature_delivery": feature_delivery_coverage(world),
            "protocol": protocol_realization(world),
            "peer_review": peer_review_coverage(world),
            "joint_digest": joint_delivery_digest(world),
            "repository": {
                "pull_requests": {
                    str(pr_id): {
                        "status": _status(getattr(pr, "status", "")),
                        "author_id": str(getattr(pr, "author_id", "") or ""),
                        "approved_by": list(getattr(pr, "approved_by", []) or [])[:8],
                        "linked_features": sorted(_linked_feature_ids(pr)),
                        "ci_passed": bool(getattr(pr, "ci_passed", False)),
                    }
                    for pr_id, pr in list(
                        (getattr(repo, "pull_requests", {}) or {}).items()
                    )[-32:]
                },
                "ci_runs": {
                    str(ci_id): {
                        "status": _status(getattr(ci, "status", "")),
                        "pr_id": str(getattr(ci, "pr_id", "") or ""),
                        "failure_reasons": list(
                            getattr(ci, "failure_reasons", []) or []
                        )[:8],
                    }
                    for ci_id, ci in list(
                        (getattr(repo, "ci_runs", {}) or {}).items()
                    )[-32:]
                },
            },
            "protocol_specs": {
                str(spec_id): {
                    "name": str(getattr(spec, "name", "") or "")[:200],
                    "status": _status(getattr(spec, "status", "")),
                    "family": str(getattr(spec, "family", "") or ""),
                    "problem_evidence": list(
                        getattr(spec, "problem_evidence", []) or []
                    )[:16],
                    "affected_actions": list(
                        getattr(spec, "affected_actions", []) or []
                    )[:16],
                    "use_count": int(getattr(spec, "use_count", 0) or 0),
                    "enforcement_count": int(
                        getattr(spec, "enforcement_count", 0) or 0
                    ),
                }
                for spec_id, spec in list(
                    (getattr(manager, "protocol_specs", {}) or {}).items()
                )[-16:]
            },
            "llm_usage": {
                "attempt_semantics": "admitted_send_stage_not_gateway_receipt",
                "calls": int(getattr(client, "calls", 0) or 0),
                "failures": int(getattr(client, "failures", 0) or 0),
                "retries": int(getattr(client, "retries", 0) or 0),
                "provider_attempts": int(
                    getattr(client, "provider_attempts", 0) or 0
                ),
                "provider_successes": int(
                    client_stats.get("provider_successes", 0) or 0
                ),
                "provider_failure_counts": dict(
                    client_stats.get("provider_failure_counts", {}) or {}
                ),
                "provider_exception_counts": dict(
                    client_stats.get("provider_exception_counts", {}) or {}
                ),
                "provider_status_counts": dict(
                    client_stats.get("provider_status_counts", {}) or {}
                ),
                "last_provider_failure": client_stats.get(
                    "last_provider_failure"
                ),
                "total_tokens": int(totals.get("total_tokens", 0) or 0),
            },
        }
        safe_row = _redact(row)
        full_row = _redact(row, max_chars=None)
        if full_row != safe_row:
            raw = json.dumps(full_row, sort_keys=True, ensure_ascii=False).encode("utf-8")
            digest = hashlib.sha256(raw).hexdigest()
            directory = self.path.parent / "trajectory_evidence"
            directory.mkdir(exist_ok=True)
            artifact = directory / f"{digest}.json"
            if not artifact.exists():
                artifact.write_bytes(raw)
            safe_row["full_evidence"] = {"path": artifact.relative_to(self.path.parent).as_posix(),
                                         "sha256": digest, "bytes": len(raw)}
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(
                json.dumps(safe_row, sort_keys=True, separators=(",", ":"))
                + "\n"
            )
        return safe_row


__all__ = [
    "adopted_pair_protocols",
    "TrajectoryRecorder",
    "feature_delivery_coverage",
    "friction_evidence",
    "initialize_two_person_sdl",
    "joint_delivery_digest",
    "peer_review_coverage",
    "peer_review_ready_coverage",
    "pending_cooperbench_protocol",
    "pending_main_b3_protocols",
    "protocol_realization",
    "record_joint_attestation",
    "submit_cooperbench_protocol_proposal",
]
