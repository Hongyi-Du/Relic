"""Public CI on separately identified committed and integrated source trees."""
from __future__ import annotations

from typing import Any

from .source_views import (
    SourceViewError, assert_pr_head_current, conservative_merge_candidate,
    freeze_pr_head, pr_head_snapshot,
)


def current_source_ci_matches(world: Any, pr: Any) -> bool:
    """A green flag must name both the published head and current integration."""
    try:
        head = pr_head_snapshot(world, pr)
        merged = conservative_merge_candidate(world, pr)
    except SourceViewError:
        return False
    preflight = (
        getattr(world, "__dict__", {}).get(
            "_cooperbench_public_runtime_preflight"
        )
        or {}
    )
    current_strength = str(
        preflight.get("public_validation_strength") or "public_regression"
    )
    return bool(
        getattr(pr, "ci_passed", False)
        and getattr(pr, "ci_source_snapshot", None) == head.receipt()
        and getattr(pr, "ci_merge_snapshot", None) == merged.receipt()
        and getattr(pr, "ci_tree_hash", None) == merged.tree_digest
        and str(
            getattr(pr, "ci_validation_strength", "public_regression")
            or "public_regression"
        ) == current_strength
    )


def run_pr_source_ci(world: Any, pr: Any) -> dict[str, Any]:
    """Test the frozen PR and its conservative integration, never ambient desk.

    This checks only the public regression suite. Feature acceptance still
    requires the peer's public probes and the external official evaluator.
    """
    from environments.org_env.product.materialize import run_public_tests

    preflight = (
        getattr(world, "__dict__", {}).get(
            "_cooperbench_public_runtime_preflight"
        )
        or {}
    )
    evidence: dict[str, Any] = {
        "checks": {},
        "validation_strength": str(
            preflight.get("public_validation_strength")
            or "public_regression"
        ),
        "functional_public_regression_available": bool(
            preflight.get("functional_public_regression_available", True)
        ),
    }
    try:
        head = freeze_pr_head(world, pr)
        evidence["source_snapshot"] = head.receipt()
        evidence["main_commit_ids"] = list(world.repo_system.repo.main_commit_ids)
        merged = conservative_merge_candidate(world, pr)
        evidence.update(source_snapshot=head.receipt(), merge_snapshot=merged.receipt())
        for role, snapshot in (("pr_head", head), ("merge_candidate", merged)):
            outcome = run_public_tests(world, source_snapshot=snapshot)
            evidence["checks"][role] = {**outcome, "source_snapshot": snapshot.receipt()}
            if not outcome.get("ok"):
                from .actor_workspace import (
                    pr_public_contract_compatibility_resolution,
                )

                compatibility = pr_public_contract_compatibility_resolution(
                    world, pr, snapshot, outcome
                )
                if compatibility:
                    # Preserve the raw smoke result and attach the exact reason
                    # it is not authoritative for this feature-evolved tree.
                    # Both explicit run_ci and the world's periodic CI sweep
                    # call this function, so they cannot disagree.
                    evidence["checks"][role][
                        "public_contract_compatibility"
                    ] = compatibility
                    continue
                started = outcome.get("returncode") is not None
                proven_runtime = bool((getattr(world, "_cooperbench_public_runtime_preflight", {}) or {}).get("passed"))
                infrastructure = (not outcome.get("available") or
                                  (bool(outcome.get("error")) and not (started and proven_runtime)))
                return {"ok": False, "kind": "infrastructure_error" if infrastructure else "contract_break",
                        "boundary": f"cooperbench_{role}_public_tests",
                        "brief": str(outcome.get("error") or outcome.get("failure_brief") or outcome.get("summary") or "public tests unavailable")[:600],
                        "detail": str(outcome.get("failure_brief") or "")[:4000], **evidence}
        assert_pr_head_current(world, pr, head)
        if conservative_merge_candidate(world, pr).receipt() != merged.receipt():
            raise SourceViewError("integration_changed_during_ci")
    except SourceViewError as error:
        return {"ok": False, "kind": ("coordination_conflict" if error.code == "source_merge_conflict" else "infrastructure_error"),
                "boundary": "cooperbench_source_identity", "brief": str(error),
                "conflict_paths": list(getattr(error, "conflict_paths", ())),
                "native_append_error": str(getattr(error, "native_append_error", "") or ""), **evidence}
    except Exception as error:
        return {"ok": False, "kind": "infrastructure_error",
                "boundary": "cooperbench_source_ci", "brief": f"cooperbench_source_ci_failed:{type(error).__name__}",
                **evidence}
    return {"ok": True, "kind": "", "boundary": "", "brief": "", **evidence}
