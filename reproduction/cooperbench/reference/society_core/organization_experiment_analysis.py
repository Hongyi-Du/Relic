"""Portfolio-level evidence assembly for the organization experiment."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

from environments.org_env.experiments.records import (
    assess_run_record_completeness,
    validate_experiment_run_record_schema,
)
from environments.org_env.experiments.statistics import (
    PairedContrastReport,
    analyze_arm_contrasts,
    analyze_primary_contrasts,
    holm_adjust_global,
    observations_from_payload,
)

from .hashing import stable_hash
from .organization_experiment_estimands import (
    GLOBAL_CONFIRMATORY_FAMILY_ID,
    PRIMARY_CONTRASTS_BY_PHASE,
    PRIMARY_METRIC_SPECS,
    SECONDARY_SENSITIVITY_METRIC_SPECS,
    TRANSFER_CONTRASTS,
    TRANSFER_REFERENCE_ARM,
    estimand_manifest,
)
from .organization_experiment_protocol import (
    CANONICAL_MODEL_SLOT,
    FULL_ORGANIZATION_CONDITION,
    MAIN_REPOSITORIES,
    MAIN_SEEDS_BY_REPOSITORY,
    PAPER_CONDITIONS,
    SEEDS_PER_CONDITION,
    TRANSFER_ARMS,
    TRANSFER_SEEDS_BY_MAPPING_INDEX,
    expected_phase_counts,
    transfer_mappings,
)


PORTFOLIO_EVIDENCE_SCHEMA_VERSION = "organization_portfolio_evidence_v4"
# Every preregistered cell is a complete agent run, so the expected record
# counts are the phase counts themselves.
EXPECTED_RUN_COUNTS = expected_phase_counts()
EXPECTED_AGENT_RUN_COUNT = sum(EXPECTED_RUN_COUNTS.values())
FORMAL_MIN_LLM_SUCCESS_RATE = 0.95
PREREGISTERED_PRIMARY_METRICS = tuple(PRIMARY_METRIC_SPECS)
TRANSFER_ARM_ORDER = (*TRANSFER_ARMS, TRANSFER_REFERENCE_ARM)


def _hash_ok(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _reference_seed_by_transfer_seed() -> dict[tuple[str, int], int]:
    """Map each (target repository, transfer seed) to its main-sweep B3 seed.

    Replicate k of a transfer mapping is paired with replicate k of the target
    repository's own main sweep, which is what makes the reference arm a paired
    comparison rather than a pooled one.
    """
    mapping: dict[tuple[str, int], int] = {}
    for index, (_source_id, target_id) in enumerate(transfer_mappings()):
        for replicate, seed in enumerate(TRANSFER_SEEDS_BY_MAPPING_INDEX[index]):
            mapping[(target_id, int(seed))] = MAIN_SEEDS_BY_REPOSITORY[target_id][
                replicate
            ]
    return mapping


def _reference_arm_records(
    run_records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Relabel each target's own B3 main run as the transfer reference arm.

    The reference is not a separate run: two of the five transfer contrasts
    compare an inherited-capability arm against the target repository starting
    from scratch, and that run already exists in the main sweep. Relabelling a
    copy into the transfer phase lets all five contrasts go through one paired
    analysis instead of a bespoke cross-phase path.
    """
    # Keyed by model slot too. Transfer runs are all on the canonical slot, but
    # the three slots share a seed schedule, so a (pack, seed) key collapses
    # three different main runs onto one another and the reference could be
    # drawn from a model the transfer arms never used.
    main_by_key: dict[tuple[str, str, str, int], Mapping[str, Any]] = {}
    for record in run_records:
        if record.get("experiment_phase") != "main_organization_sweep":
            continue
        if str(record.get("arm_id") or "") != FULL_ORGANIZATION_CONDITION:
            continue
        key = (
            str(record.get("pack") or ""),
            str(record.get("provider") or ""),
            str(record.get("model") or ""),
            int(record.get("seed") or 0),
        )
        main_by_key[key] = record
    reference_seeds = _reference_seed_by_transfer_seed()
    synthesized: list[dict[str, Any]] = []
    unresolved: list[str] = []
    seen: set[tuple[str, int]] = set()
    for record in run_records:
        if record.get("experiment_phase") != "capability_transfer":
            continue
        target = str(record.get("pack") or "")
        seed = int(record.get("seed") or 0)
        if (target, seed) in seen:
            continue
        seen.add((target, seed))
        reference_seed = reference_seeds.get((target, seed))
        source = (
            main_by_key.get(
                (
                    target,
                    str(record.get("provider") or ""),
                    str(record.get("model") or ""),
                    reference_seed,
                )
            )
            if reference_seed is not None
            else None
        )
        if source is None:
            unresolved.append(f"{target}:{seed}")
            continue
        # The relabelled copy joins the transfer paired block, so the three
        # fields that identify a block member have to be rewritten to match it.
        # Carrying the main sweep's values over put the reference in a
        # different lineage, gave it a randomization_order already taken by one
        # of the four arms, and left a randomization_block naming the main
        # seed — each of which a paired-cell guard rejects.
        arms_in_block = 1 + len(TRANSFER_ARMS)
        synthesized.append(
            {
                **source,
                "run_id": f"{source['run_id']}__as_transfer_reference",
                "experiment_phase": "capability_transfer",
                "arm_id": TRANSFER_REFERENCE_ARM,
                "seed": seed,
                "randomization_block": record.get("randomization_block"),
                "randomization_order": arms_in_block - 1,
                "replication_id": record.get("replication_id"),
            }
        )
    return synthesized, unresolved


def _report_rows(
    reports: Sequence[PairedContrastReport],
    *,
    confirmatory: bool,
) -> list[dict[str, Any]]:
    return [
        {
            **report.to_dict(),
            "confirmatory": confirmatory,
            "multiplicity_family": (
                GLOBAL_CONFIRMATORY_FAMILY_ID if confirmatory else None
            ),
        }
        for report in reports
    ]


def _bounded(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        value = float(value)
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"primary_component_missing:{label}") from exc
    if not 0.0 <= parsed <= 1.0:
        raise ValueError(f"primary_component_out_of_range:{label}:{parsed}")
    return parsed


def _record_with_primary_composites(
    record: Mapping[str, Any],
) -> dict[str, Any]:
    metrics = dict(record.get("metrics") or {})
    final = record.get("final_evaluation") or {}
    if not isinstance(final, Mapping):
        raise ValueError("final_evaluation_must_be_object")
    product_components = (
        _bounded(
            metrics.get("oss_hidden_pass_rate"),
            label="metrics.oss_hidden_pass_rate",
        ),
        _bounded(
            final.get("causal_fix_rate"),
            label="final_evaluation.causal_fix_rate",
        ),
    )
    release_readiness_sensitivity_components = (
        *product_components,
        _bounded(
            final.get("formal_release_ready"),
            label="final_evaluation.formal_release_ready",
        ),
    )
    capability_components = tuple(
        _bounded(metrics.get(name), label=f"metrics.{name}")
        for name in (
            "strong_protocol_emergence_rate",
            "cross_context_protocol_reuse_rate",
            "protocol_persistence_rate",
            "measurable_impact",
        )
    )
    metrics["product_outcome_composite"] = sum(product_components) / len(
        product_components
    )
    metrics["product_outcome_with_release_readiness_sensitivity"] = sum(
        release_readiness_sensitivity_components
    ) / len(release_readiness_sensitivity_components)
    metrics["organizational_capability_composite"] = sum(capability_components) / len(
        capability_components
    )
    families = dict(record.get("metric_families") or {})
    for metric, specification in PRIMARY_METRIC_SPECS.items():
        families[metric] = str(specification["family"])
    for metric, specification in SECONDARY_SENSITIVITY_METRIC_SPECS.items():
        families[metric] = str(specification["family"])
    return {
        **record,
        "metrics": metrics,
        "metric_families": families,
    }


def _statistical_reports(
    run_records: Sequence[Mapping[str, Any]],
    *,
    reference_records: Sequence[Mapping[str, Any]],
    bootstrap_samples: int,
    randomization_samples: int,
) -> dict[str, list[dict[str, Any]]]:
    analysis_records = [
        _record_with_primary_composites(record)
        for record in (*run_records, *reference_records)
    ]
    observations = observations_from_payload(analysis_records)
    common = {
        "metrics": PREREGISTERED_PRIMARY_METRICS,
        "bootstrap_samples": bootstrap_samples,
        "randomization_samples": randomization_samples,
        "require_complete_run_records": True,
        "require_contamination_clearance": False,
        "include_block_reports": False,
    }
    main = analyze_primary_contrasts(
        observations,
        experiment_phase="main_organization_sweep",
        expected_conditions=PAPER_CONDITIONS,
        contrasts=PRIMARY_CONTRASTS_BY_PHASE["main_organization_sweep"],
        expected_packs=MAIN_REPOSITORIES,
        expected_seeds_per_block=SEEDS_PER_CONDITION,
        **common,
    )
    main = holm_adjust_global(main)
    transfer = analyze_arm_contrasts(
        observations,
        experiment_phase="capability_transfer",
        expected_arms=TRANSFER_ARM_ORDER,
        contrasts=TRANSFER_CONTRASTS,
        expected_packs=MAIN_REPOSITORIES,
        expected_seeds_per_block=SEEDS_PER_CONDITION,
        **common,
    )
    return {
        "main_organization_sweep": _report_rows(main, confirmatory=True),
        "capability_transfer": _report_rows(transfer, confirmatory=False),
    }


def build_organization_portfolio_evidence(
    *,
    run_records: Sequence[Mapping[str, Any]],
    execution_manifest_hash: str | None = None,
    execution_state_hash: str | None = None,
    artifact_linkage_audit: Mapping[str, Any] | None = None,
    frozen_repository_evidence_kinds: Mapping[str, str] | None = None,
    bootstrap_samples: int = 10_000,
    randomization_samples: int = 100_000,
) -> dict[str, Any]:
    """Build one content-addressed portfolio artifact without gating on wins."""

    blocking: list[str] = []
    if not _hash_ok(execution_manifest_hash):
        blocking.append("execution_manifest_hash_missing_or_invalid")
    if not _hash_ok(execution_state_hash):
        blocking.append("execution_state_hash_missing_or_invalid")
    linkage = dict(artifact_linkage_audit or {})
    declared_linkage_hash = linkage.pop("audit_hash", None)
    if (
        linkage.get("valid") is not True
        or not _hash_ok(declared_linkage_hash)
        or stable_hash(linkage) != declared_linkage_hash
    ):
        blocking.append("artifact_linkage_audit_missing_or_invalid")
    expected_repository_ids = set(MAIN_REPOSITORIES)
    evidence_kinds = {
        str(repository_id): str(kind)
        for repository_id, kind in (frozen_repository_evidence_kinds or {}).items()
        if str(repository_id) and str(kind)
    }
    if set(evidence_kinds) != expected_repository_ids:
        blocking.append(
            "repository_evidence_kind_binding_mismatch:"
            f"expected={sorted(expected_repository_ids)}:"
            f"observed={sorted(evidence_kinds)}"
        )
    run_ids: list[str] = []
    identities: list[tuple[Any, ...]] = []
    phase_counts: Counter[str] = Counter()
    incomplete_records: list[str] = []
    ineligible_final_records: list[str] = []
    primary_metric_errors: list[str] = []
    degraded_llm_records: list[str] = []
    prompt_visibility_failures: list[str] = []
    contamination_counts: Counter[str] = Counter()
    contamination_clearance_counts: Counter[str] = Counter()
    run_source_pairs: list[tuple[str, str]] = []
    missing_execution_job_ids: list[str] = []
    evidence_kind_mismatches: list[str] = []
    observed_evidence_kinds: Counter[tuple[str, str]] = Counter()
    for record in run_records:
        try:
            validate_experiment_run_record_schema(record)
        except (TypeError, ValueError) as exc:
            blocking.append(f"invalid_run_record:{exc}")
            continue
        run_id = str(record["run_id"])
        run_ids.append(run_id)
        source_job_id = str(
            (record.get("provenance") or {}).get("execution_job_id") or ""
        )
        if not source_job_id:
            missing_execution_job_ids.append(run_id)
        else:
            run_source_pairs.append((source_job_id, run_id))
        phase = str(record.get("experiment_phase") or "")
        phase_counts[phase] += 1
        identities.append(
            (
                phase,
                record.get("pack"),
                record.get("provider"),
                record.get("model"),
                record.get("seed"),
                record.get("arm_id"),
            )
        )
        if not assess_run_record_completeness(record)["is_complete"]:
            incomplete_records.append(run_id)
        final = record.get("final_evaluation") or {}
        if not isinstance(final, Mapping) or final.get("status") != "passed":
            ineligible_final_records.append(run_id)
        observed_kind = (
            str(final.get("evidence_kind") or "") if isinstance(final, Mapping) else ""
        )
        repository_id = str(record.get("pack") or "")
        observed_evidence_kinds[(repository_id, observed_kind)] += 1
        if evidence_kinds.get(repository_id) != observed_kind:
            evidence_kind_mismatches.append(f"{run_id}:{repository_id}:{observed_kind}")
        contamination_counts[str(record.get("contamination_status") or "missing")] += 1
        contamination_clearance_counts[
            (
                "cleared"
                if record.get("contamination_clearance") is True
                else "not_cleared"
            )
        ] += 1
        try:
            _record_with_primary_composites(record)
        except (TypeError, ValueError) as exc:
            primary_metric_errors.append(f"{run_id}:{exc}")
        usage = record.get("llm_usage")
        resources = record.get("resource_usage") or {}
        denied = resources.get("denied") or {}
        if (
            not isinstance(usage, Mapping)
            or int(usage.get("calls", 0) or 0) <= 0
            or int(usage.get("resource_denials", 0) or 0) > 0
            or any(int(value or 0) > 0 for value in denied.values())
        ):
            degraded_llm_records.append(run_id)
        else:
            calls = int(usage.get("calls", 0) or 0)
            failures = int(usage.get("failures", 0) or 0)
            if (calls - failures) / calls < FORMAL_MIN_LLM_SUCCESS_RATE:
                degraded_llm_records.append(run_id)
        prompt_audit = record.get("prompt_visibility_audit")
        if (
            not isinstance(prompt_audit, Mapping)
            or prompt_audit.get("clear") is not True
            or int(prompt_audit.get("violation_count", -1) or 0) != 0
            or int(prompt_audit.get("calls_audited", -1) or 0)
            != int((usage or {}).get("calls", -2) or 0)
        ):
            prompt_visibility_failures.append(run_id)
    if len(run_records) != EXPECTED_AGENT_RUN_COUNT:
        blocking.append(
            f"agent_run_count_mismatch:expected={EXPECTED_AGENT_RUN_COUNT}:"
            f"observed={len(run_records)}"
        )
    if len(set(run_ids)) != len(run_ids):
        blocking.append("duplicate_run_id")
    if len(set(identities)) != len(identities):
        blocking.append("duplicate_run_identity")
    if missing_execution_job_ids:
        blocking.append(
            f"run_record_execution_job_id_missing:{len(missing_execution_job_ids)}"
        )
    if len(set(run_source_pairs)) != len(run_source_pairs):
        blocking.append("duplicate_run_source_pair")
    if dict(phase_counts) != EXPECTED_RUN_COUNTS:
        blocking.append(
            f"phase_run_count_mismatch:{dict(sorted(phase_counts.items()))}"
        )
    if incomplete_records:
        blocking.append(f"incomplete_run_records:{len(incomplete_records)}")
    if ineligible_final_records:
        blocking.append(f"final_evaluator_not_passed:{len(ineligible_final_records)}")
    if evidence_kind_mismatches:
        blocking.append(
            f"run_record_evidence_kind_mismatch:{len(evidence_kind_mismatches)}"
        )
    if primary_metric_errors:
        blocking.append(
            f"primary_metric_derivation_failed:{len(primary_metric_errors)}"
        )
    if degraded_llm_records:
        blocking.append(f"llm_treatment_integrity_failed:{len(degraded_llm_records)}")
    if prompt_visibility_failures:
        blocking.append(
            f"prompt_visibility_integrity_failed:{len(prompt_visibility_failures)}"
        )

    # Model is a matched control within a paired block, but the sweep spans
    # three frozen slots by design, so the identity count is reported rather
    # than forced to one.
    model_identities = {
        (record.get("provider"), record.get("model")) for record in run_records
    }
    transfer_model_identities = {
        (record.get("provider"), record.get("model"))
        for record in run_records
        if record.get("experiment_phase") == "capability_transfer"
    }
    if transfer_model_identities and len(transfer_model_identities) != 1:
        blocking.append(
            f"transfer_model_identity_not_matched:{len(transfer_model_identities)}"
        )

    reference_records, unresolved_references = _reference_arm_records(run_records)
    if unresolved_references:
        blocking.append(
            f"transfer_reference_run_unresolved:{len(unresolved_references)}"
        )

    statistical_reports: dict[str, list[dict[str, Any]]] = {}
    records_ready = not any(
        reason.startswith(
            (
                "invalid_run_record",
                "agent_run_count_mismatch",
                "duplicate_run",
                "phase_run_count_mismatch",
                "incomplete_run_records",
                "final_evaluator_not_passed",
                "repository_evidence_kind_binding_mismatch",
                "run_record_evidence_kind_mismatch",
                "primary_metric_derivation_failed",
                "llm_treatment_integrity_failed",
                "prompt_visibility_integrity_failed",
                "transfer_reference_run_unresolved",
            )
        )
        for reason in blocking
    )
    if records_ready:
        try:
            statistical_reports = _statistical_reports(
                run_records,
                reference_records=reference_records,
                bootstrap_samples=bootstrap_samples,
                randomization_samples=randomization_samples,
            )
        except ValueError as exc:
            blocking.append(f"statistical_contract_failed:{exc}")

    framework_ready = not blocking
    all_runs_contamination_cleared = (
        len(run_records) == EXPECTED_AGENT_RUN_COUNT
        and contamination_clearance_counts.get("cleared", 0) == EXPECTED_AGENT_RUN_COUNT
    )
    all_main_repositories_behavioral = all(
        evidence_kinds.get(repository_id) == "behavior"
        for repository_id in MAIN_REPOSITORIES
    )
    confirmatory_rows = [
        row
        for reports in statistical_reports.values()
        for row in reports
        if row.get("confirmatory") is True
    ]
    positive_confirmatory_rejections = sum(
        float(row.get("mean_difference", 0.0)) > 0.0
        and float(row.get("adjusted_p_value", 1.0)) < 0.05
        for row in confirmatory_rows
    )
    transfer_rows = statistical_reports.get("capability_transfer", [])
    transfer_positive_rows = [
        row
        for row in transfer_rows
        if float(row.get("mean_difference", 0.0)) > 0.0
        and float(row.get("p_value", 1.0)) < 0.05
    ]
    payload: dict[str, Any] = {
        "schema_version": PORTFOLIO_EVIDENCE_SCHEMA_VERSION,
        "execution_manifest_hash": execution_manifest_hash,
        "execution_state_hash": execution_state_hash,
        "artifact_linkage_audit": (
            dict(artifact_linkage_audit) if artifact_linkage_audit is not None else None
        ),
        "expected_agent_run_count": EXPECTED_AGENT_RUN_COUNT,
        "observed_agent_run_count": len(run_records),
        "expected_run_counts": dict(sorted(EXPECTED_RUN_COUNTS.items())),
        "phase_counts": dict(sorted(phase_counts.items())),
        "primary_metrics": list(PREREGISTERED_PRIMARY_METRICS),
        "estimands": estimand_manifest(),
        "repository_evidence_kinds": dict(sorted(evidence_kinds.items())),
        "observed_evidence_kind_counts": {
            f"{repository_id}:{kind}": count
            for (repository_id, kind), count in sorted(observed_evidence_kinds.items())
        },
        "contamination_status_counts": dict(sorted(contamination_counts.items())),
        "contamination_clearance_counts": dict(
            sorted(contamination_clearance_counts.items())
        ),
        "model_identity_count": len(model_identities),
        "statistical_reports": statistical_reports,
        "capability_transfer": {
            "arms": list(TRANSFER_ARM_ORDER),
            "reference_arm": TRANSFER_REFERENCE_ARM,
            "reference_model_slot": CANONICAL_MODEL_SLOT,
            "reference_run_count": len(reference_records),
            "contrast_count": len(transfer_rows),
        },
        "source_linkage": {
            "agent_source_pair_count": len(set(run_source_pairs)),
        },
        "framework_ready": framework_ready,
        "scientific_result_available": framework_ready,
        "scientific_result_summary": {
            "confirmatory_test_count": len(confirmatory_rows),
            "positive_confirmatory_rejection_count": (positive_confirmatory_rejections),
            "any_positive_confirmatory_effect_supported": (
                framework_ready and positive_confirmatory_rejections > 0
            ),
            "transfer_positive_contrast_count": len(transfer_positive_rows),
            "interpretation": (
                "result support is computed from corrected tests and is "
                "separate from evidence-package readiness"
            ),
        },
        "claim_readiness": {
            "paired_contrasts_analyzable": framework_ready,
            "paired_acceptance_oracle_effects": framework_ready,
            "capability_transfer_contrasts_analyzable": (
                framework_ready and len(transfer_rows) > 0
            ),
            "runtime_behavior_product_effects": (
                framework_ready and all_main_repositories_behavioral
            ),
            "absolute_product_performance": (
                framework_ready
                and all_runs_contamination_cleared
                and all_main_repositories_behavioral
            ),
            "all_runs_contamination_cleared": (all_runs_contamination_cleared),
            "all_main_repositories_behavioral": (all_main_repositories_behavioral),
        },
        "blocking_reasons": list(dict.fromkeys(blocking)),
        "claim_boundaries": {
            "readiness_not_success": (
                "framework_ready means the preregistered evidence is complete "
                "and valid; it does not mean any directional hypothesis won"
            ),
            "transfer_reference_arm": (
                "the reference arm is the target repository's own B3 run from "
                "the main sweep, relabelled for the paired comparison; it is "
                "not an extra run and is never double counted as one"
            ),
            "organization_ladder_step": (
                "B3-B2 moves the decision kernel, profile conditioning, and "
                "the capability lifecycle together and is reported as a "
                "bundled step"
            ),
            "prompt_visibility": (
                "every cognitive provider prompt is hashed and checked before "
                "dispatch against evaluator-private identifiers, paths, and "
                "high-specificity text; any hit blocks framework readiness"
            ),
            "product_outcome": (
                "the primary endpoint averages hidden pass rate and causal fix "
                "rate; thresholded release readiness is excluded to avoid "
                "duplicate evaluator weighting and appears only in the "
                "declared secondary sensitivity endpoint"
            ),
            "statistical_population": (
                "all intervals and tests are conditional on the frozen "
                "repositories, providers, models, and seeds; repositories were "
                "not probability sampled, so repository-population inference "
                "is not supported"
            ),
            "contamination": (
                "paired organization contrasts remain conditional on each "
                "frozen model-repository exposure state; absolute product "
                "claims require all operational clearances, which still cannot "
                "prove absence of latent training exposure"
            ),
        },
    }
    payload["evidence_hash"] = stable_hash(payload)
    return payload


__all__ = [
    "EXPECTED_AGENT_RUN_COUNT",
    "EXPECTED_RUN_COUNTS",
    "PORTFOLIO_EVIDENCE_SCHEMA_VERSION",
    "PREREGISTERED_PRIMARY_METRICS",
    "TRANSFER_ARM_ORDER",
    "build_organization_portfolio_evidence",
]
