"""PolicyActionAttractorGuard (v4 §3).

v3 showed that repeated "safe" actions (audit_readme_claims / monitor_customer_feedback /
create_doc / edit_repo_file) are not just an LLM-prompt problem — they are a *policy
attractor*: the LLM proposes a plausible action, the policy scores it high, the validator
passes it, task progress rewards it, the product gap remains, so the same action looks
attractive again. This module breaks the loop two ways:

  * hard masks   — remove a candidate from the choice set entirely (cooldown, awaiting
                   review, nothing new to monitor, task already done, doc already exists);
  * soft penalties — down-weight low-marginal-value actions in the utility so the policy
                   prefers actions with real marginal contribution.

It also produces a top-k policy trace so we can tell LLM-repetition from a scoring
attractor from a validator-fallback attractor.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

# How many ticks an agent must wait before repeating the SAME "safe" action.
ACTION_OBJECT_COOLDOWN: Dict[str, int] = {
    "audit_readme_claims": 12,
    "monitor_customer_feedback": 6,
    "create_doc": 12,
    "edit_doc": 4,
    "edit_repo_file": 4,
    "create_report_quality_checklist": 24,
    "schedule_meeting": 6,
    "propose_protocol": 12,
    "internal_search": 4,
    "review_doc": 4,
    "create_issue": 8,           # v4: stop re-filing the same issue every tick
    "read_feed": 6,              # v5: read_feed became the new safe-action attractor
    "use_tool": 6,               # v5: use_tool became the next attractor (55/100t)
    "run_launch_readiness_check": 6,   # #2: stop the readiness-check loop on a blocked RC
    "challenge_result": 8,       # v8 #1: stop re-challenging the same result every tick
    "send_async_update": 5,      # v8d P2b: stop the EOD broadcast loop (per-agent / per-object)
}
# v8 #1: once a dispute has this many supporters it is "established" — further challenges
# add no signal and are masked; the team should resolve it (reproduction / evidence) instead.
CHALLENGE_SUPPORT_CAP = 3
# v6 P0.1: an artifact at/above this revision that is still awaiting review is blocked
# from further edits — it must go through review/commit/PR/merge before more churn.
AWAITING_REVIEW_EDIT_BLOCK_REV = 3

# How many ticks before a heavily-touched artifact may be revised again by anyone.
GLOBAL_OBJECT_REVISION_COOLDOWN: Dict[str, int] = {
    "art_README_md": 6,
    "art_docs_product_design_md": 6,
    "art_report_quality_checklist_md": 8,
    "art_tools_claim_tracker_py": 4,
    "art_tools_report_writer_py": 4,
    "art_tools_source_tracker_py": 4,
    "art_eval_eval_stub_py": 4,
}

_DOC_ACTIONS = {"create_doc", "edit_doc", "audit_readme_claims", "create_report_quality_checklist"}
# spec #5: expensive, non-critical actions — down-weighted when runway is short so the
# policy prefers cheap pilots / prioritization (run_cheap_pilot is deliberately excluded).
_EXPENSIVE_ACTIONS = {"run_eval_stub", "run_eval", "run_experiment", "run_full_pilot",
                      "create_product_demo", "share_external_post"}
# #6: the heavy deep-work the review flagged as happening at night (task grind + expensive
# experiment runs + the release readiness-check loop). Lighter actions (search / messages /
# edits / rest) stay available so the agent is never stuck and the candidate pool is non-empty.
_NIGHT_WORK_MASKABLE = {
    "work_on_task", "run_eval_stub", "run_eval", "run_experiment", "run_cheap_pilot",
    "run_launch_readiness_check", "schedule_meeting",
}
# v5: both feed-reads are gated on NEW external signal (read_feed was the new attractor)
_MONITOR_ACTIONS = {"monitor_customer_feedback", "read_feed"}

# v13 P3: generic "looks busy" actions that are down-weighted in shipping mode (a release
# smoke/CI/contract blocker is open) so the org converges on closing the blocker. Closure
# actions (edit/commit/PR/CI/merge/readiness/dogfood/tracker fixes) are NOT here.
_SHIPPING_OFFTASK = {
    "internal_search", "read_feed", "create_doc", "create_onboarding_doc", "write_design_note",
    "schedule_meeting", "propose_protocol", "propose_product_direction", "share_external_post",
    "monitor_customer_feedback", "run_cheap_pilot", "audit_readme_claims",
}

# ship-it (self-iteration §): when a merged fix is sitting UNSHIPPED, a lead resting / deferring /
# doing off-task work is what leaves it never released — down-weight those so create_release_candidate
# / publish win at the (often nocturnal) decision point.
_SHIP_IT_DOWNWEIGHT = set(_SHIPPING_OFFTASK) | {"rest_offline", "defer_until_work_hours", "work_on_task"}

# v4 review §1: create-class doc actions dedup by PURPOSE, not by object id (each
# create makes a fresh path, so object cooldown can't stop them). If a doc of the
# same purpose is already active/awaiting-review/recent, revise/review it instead.
_CREATE_PURPOSE = {
    "create_onboarding_doc": "onboarding",
    "create_report_quality_checklist": "report_quality",
    "create_report_template": "report_template",
    "write_design_note": "design_note",
    "create_product_demo": "demo",
    "create_eval_stub": "eval",
}
DOC_PURPOSE_COOLDOWN = {"onboarding": 24, "report_quality": 48, "design_note": 12,
                        "report_template": 24, "demo": 24, "eval": 24}
# actions never re-admitted by the never-empty fallback (create-class + safe-action attractors)
_NO_FALLBACK = set(_CREATE_PURPOSE) | {"use_tool", "create_doc", "create_issue",
                                       "monitor_customer_feedback", "read_feed"}
# while an agent has uncommitted patches, the safe-action attractors are masked so the
# agent commits/PRs the work instead of looping on tools/feeds (create-class is handled
# by the purpose singleton; one extra cross-artifact edit stays allowed).
_PRE_COMMIT_BLOCKED = {"use_tool", "create_doc", "read_feed", "monitor_customer_feedback"}
# ESCALATION (growth-run finding): the LLM was offered commit_patch 22x but never picked
# it — it kept choosing work_on_task / edit_repo_file, so the patch->commit->PR->CI->merge
# chain never advanced (0 commits/PRs/releases in 240t). Once >=2 patches are uncommitted,
# also mask NEW product work (more edits + task work) so commit_patch / open_pr is the only
# way to make product progress. commit/PR/CI/merge/review + comm/meeting/governance stay
# available, so there is no deadlock.
_PATCH_PRODUCERS = {"edit_repo_file", "audit_readme_claims", "edit_doc", "update_claim_tracker",
                    "update_source_tracker", "write_design_note", "create_eval_stub",
                    "create_report_template", "create_report_quality_checklist",
                    "create_onboarding_doc", "propose_product_direction"}
_PRE_COMMIT_ESCALATED = _PATCH_PRODUCERS | {"work_on_task"}
_PRE_COMMIT_ESCALATE_AT = 2
# gpt-5 run finding: an agent with accepted-but-uncommitted patches escaped the repo chain via SOFT
# actions the pre-commit mask doesn't cover (messages / search / comments / defer / pilots), so
# commit_patch was never chosen and nothing reached mainline. Down-weight those escapes so
# commit_patch / open_pr wins (rest_offline stays free so night-time isn't a deadlock).
_UNCOMMITTED_SOFT_ESCAPE = {"send_async_update", "internal_search", "respond_to_public_comment",
                            "defer_until_work_hours", "run_cheap_pilot", "work_on_task",
                            "update_task_status"}

# Once a reviewed, green request can land, starting more product work merely
# widens the desk-to-mainline gap. Keep repository lifecycle actions free, but
# strongly down-weight ordinary escape hatches for an agent who can merge.
_MERGE_READY_SOFT_ESCAPE = (
    _UNCOMMITTED_SOFT_ESCAPE
    | _SHIPPING_OFFTASK
    | _EXPENSIVE_ACTIONS
    | {"run_public_tests", "rest_offline"}
)


def candidate_target_artifact(c) -> Optional[str]:
    """Best-effort resolve the product-artifact id a candidate would touch."""
    p = getattr(c, "parameters", None) or {}
    for k in ("artifact_id", "target_object_id", "object_id"):
        v = p.get(k)
        if isinstance(v, str) and v.startswith("art_"):
            return v
    fp = p.get("file_path") or p.get("path") or p.get("doc_path")
    if isinstance(fp, str) and fp:
        return "art_" + fp.replace("/", "_").replace(".", "_")
    if c.action_type == "audit_readme_claims":
        return "art_README_md"
    return None


def _cooperbench_required_path_edit(
    candidate: Any, agent_id: str, world: Any
) -> bool:
    """Whether this exact edit closes a registered pair-SDL path."""

    params = getattr(candidate, "parameters", None) or {}
    if (
        getattr(candidate, "action_type", "") != "edit_repo_file"
        or params.get("_cooperbench_required_path") is not True
        or not world.__dict__.get("_cooperbench_delivery_focus", False)
    ):
        return False
    assignments = world.__dict__.get("_cooperbench_feature_issue_by_agent")
    state = world.__dict__.get("_cooperbench_sdl_state")
    if not isinstance(assignments, dict) or not isinstance(state, dict):
        return False
    assigned_issue = str(assignments.get(str(agent_id)) or "")
    if not assigned_issue or str(params.get("_oss_issue") or "") != assigned_issue:
        return False
    path = str(params.get("file_path") or "").replace("\\", "/")
    required = {
        str(item).replace("\\", "/")
        for item in (state.get("required_feature_paths") or {}).get(
            assigned_issue, []
        )
        if str(item)
    }
    return bool(path and path in required)


_ANY = "__any__"


def _last_action_tick(world, agent_id: str, action_type: str, target: Any = _ANY) -> Optional[int]:
    """Most recent tick this agent did action_type. If ``target`` is given (and not
    ``_ANY``), only count actions that touched that same object — so cooldown is per
    (agent, action, OBJECT), and switching to a different object isn't blocked (v4 §2)."""
    last = None
    for a in getattr(world, "action_log", None) or []:
        if a.get("agent_id") != agent_id or a.get("action_type") != action_type:
            continue
        # per-object match only when the log entry actually records a target (real runtime
        # always does); legacy/synthetic entries without one match on action (back-compat).
        if target is not _ANY and "target" in a and a.get("target") != target:
            continue
        t = a.get("tick")
        if t is not None and (last is None or t > last):
            last = t
    return last


FORCED_EDIT_REJECT_CEILING = 4


def _forced_edit_dead_end(world: Any, target: str) -> bool:
    """True when forced edits on ``target`` keep being rejected with none accepted.

    Bounds the ``_blocker_fix`` exemption without weakening it: attempts are only
    blocked once the artifact has accumulated the ceiling in rejected patches AND
    has no accepted patch to show for them, so a file that is genuinely being
    fixed is never throttled.
    """
    from environments.org_env.product.patch_validator import is_infrastructure_rejection

    rejected = accepted = 0
    for patch in (getattr(world, "patches", {}) or {}).values():
        if target not in (
            getattr(patch, "target_object_id", None),
            getattr(patch, "artifact_id", None),
        ):
            continue
        status = getattr(patch, "validation_status", "")
        if status == "rejected":
            # An infrastructure rejection (the model never answered) is not an
            # attempt that failed; counting it toward the ceiling turns an
            # endpoint outage into a permanent mask on the file. Measured: the
            # issue-driver deals _blocker_fix candidates, so this ceiling - not
            # the churn guards - was the channel that shut down query_parsing.py
            # after repeated 5xx bursts.
            if not is_infrastructure_rejection(patch):
                rejected += 1
        elif status in ("accepted", "applied", "merged"):
            accepted += 1
    return accepted == 0 and rejected >= FORCED_EDIT_REJECT_CEILING


def _accepted_pending_code_count(
    world: Any, pending: List[Tuple[str, str]]
) -> int:
    """Count accepted code patches that are still waiting for a commit.

    The workflow ledger is authoritative for pending state, while the patch
    object is authoritative for acceptance and code-vs-document identity. A
    missing fixture or legacy patch is not guessed to be code, preserving the
    long-standing two-patch threshold for documents.
    """
    from environments.org_env.product.patch_objects import CODE_PATCH_TYPES

    patches = getattr(world, "patches", {}) or {}
    count = 0
    for patch_id, _artifact_id in pending:
        patch = patches.get(patch_id)
        if patch is None:
            continue
        status = str(getattr(patch, "validation_status", "") or "").lower()
        patch_type = str(getattr(patch, "patch_type", "") or "").lower()
        if status == "accepted" and patch_type in CODE_PATCH_TYPES:
            count += 1
    return count


def _merge_ready_for_agent(world: Any, agent_id: str) -> bool:
    """Whether this agent is offered a genuinely landable PR.

    This mirrors the generic repo workflow authority: an author may merge their
    own request, while founders/cofounders/reliability may land any request.
    Substrates with additional delivery gates must actually allow that merge;
    a stale approved/CI boolean must not suppress a required repair action.
    """
    repo = getattr(getattr(world, "repo_system", None), "repo", None)
    if repo is None:
        return False
    agent = (getattr(world, "agents", {}) or {}).get(agent_id)
    role = str(getattr(agent, "role", "") or "")
    may_land_others = role in {"founder", "cofounder", "reliability"}
    programbench_active = False
    if "programbench_profile_state" in getattr(world, "__dict__", {}):
        try:
            from environments.org_env.programbench import programbench_profile_active

            programbench_active = programbench_profile_active(world)
        except (ImportError, AttributeError, TypeError, ValueError):
            programbench_active = False
    for pr in (getattr(repo, "pull_requests", {}) or {}).values():
        raw_status = getattr(pr, "status", "")
        status = str(getattr(raw_status, "value", raw_status) or "").lower()
        if status != "approved" or not bool(getattr(pr, "ci_passed", False)):
            continue
        # A conflict needs repair rather than a hard merge attractor; masking
        # edits in that state would make the repair path unreachable.
        if bool(getattr(pr, "merge_conflict", False)):
            continue
        if isinstance(world.__dict__.get("_cooperbench_sdl_state"), dict):
            try:
                from environments.org_env.runtime_adapter.execution import _cooperbench_action_block_reason

                if _cooperbench_action_block_reason(
                    world, agent_id, "merge_pr", {"pr_id": pr.pr_id}
                ) is not None:
                    continue
            except (ImportError, AttributeError, TypeError, ValueError):
                continue
        if programbench_active:
            try:
                from environments.org_env.programbench import (
                    programbench_live_submission_block_reason,
                )

                if programbench_live_submission_block_reason(
                    world, merge_pr=pr
                ) is not None:
                    continue
            except (ImportError, AttributeError, TypeError, ValueError):
                continue
        if getattr(pr, "author_id", None) == agent_id or may_land_others:
            return True
    return False


class AttractorGuard:
    # -- hard masks ---------------------------------------------------------
    def mask_reason(self, c, agent_id: str, world: Any, tick: int) -> Optional[str]:
        at = c.action_type
        params = getattr(c, "parameters", None) or {}
        arts = getattr(world, "product_artifacts", {}) or {}
        tgt = candidate_target_artifact(c)
        programbench_state = None
        if "programbench_profile_state" in getattr(world, "__dict__", {}):
            try:
                from environments.org_env.programbench import (
                    get_programbench_profile_state,
                    programbench_profile_active,
                )

                if programbench_profile_active(world):
                    programbench_state = get_programbench_profile_state(world)
            except (ImportError, AttributeError, TypeError, ValueError):
                programbench_state = None
        if at == "write_design_note" and programbench_state is not None:
            phase = str(programbench_state.get("phase") or "")
            typed = bool(
                params.get("_programbench_contract") is True
                and params.get("programbench_artifact_kind")
                == "behavioral_contract"
            )
            integration_owner = next(
                (
                    str(row.get("agent_id") or "")
                    for row in programbench_state.get("role_assignments") or []
                    if isinstance(row, dict)
                    and row.get("work_role") == "integration_owner"
                ),
                "",
            )
            if typed and phase not in {"explore", "develop"}:
                return "ProgramBench contract requires an active workflow state"
            if typed and agent_id != integration_owner:
                return "only the ProgramBench integration owner authors the contract"
            if typed and params.get("artifact_id") != "programbench_reconstruction":
                return "ProgramBench contract must use its stable reconstruction target"
        if (
            at == "run_public_tests"
            and programbench_state is not None
            and params.get("probe_mode") == "reference_only"
        ):
            designated_runners = {
                str(row.get("agent_id") or "")
                for row in programbench_state.get("role_assignments") or []
                if isinstance(row, dict)
                and row.get("work_role") in {"probe_owner", "verifier"}
            }
            if agent_id not in designated_runners:
                return (
                    "ProgramBench reference probes are reserved for the "
                    "designated probe owner or verifier"
                )

        # Public probe/contract debt is profile-owned in both top-level states.
        # EXPLORE cannot commit its pending probe, while DEVELOP may acquire
        # the same debt after a probe revision even when source patches are
        # pending.  The native pre-commit/churn attractor must not mask the only
        # actions that can refresh the exact corpus and its typed contract.
        # Execution and handler validators still enforce the exact trusted
        # probe surface, owner, evidence binding, and singleton contract.
        programbench_debt_repair_action = False
        if (
            programbench_state is not None
            and str(programbench_state.get("phase") or "")
            in {"explore", "develop"}
        ):
            typed_contract = bool(
                at == "write_design_note"
                and params.get("_programbench_contract") is True
                and params.get("programbench_artifact_kind")
                == "behavioral_contract"
            )
            trusted_probe_refine = False
            if at == "edit_repo_file" and tgt and tgt in arts:
                try:
                    from environments.org_env.backend.repo.workflow import (
                        programbench_public_probe_artifact,
                    )

                    trusted_probe_refine = programbench_public_probe_artifact(
                        world, arts[tgt]
                    )
                except (ImportError, AttributeError, TypeError, ValueError):
                    trusted_probe_refine = False
            programbench_debt_repair_action = bool(
                typed_contract or trusted_probe_refine
            )
        if programbench_debt_repair_action:
            return None

        # Delivery gates precede the forced-edit exemption. A forced blocker
        # edit may bypass churn guards, but once accepted code is on a branch it
        # must be committed rather than repeatedly rewritten. Documents retain
        # the historical threshold of two pending patches.
        from environments.org_env.backend.repo.workflow import pending_for_agent

        uncommitted = pending_for_agent(world, agent_id)
        accepted_code = _accepted_pending_code_count(world, uncommitted)
        pair_required_edit = _cooperbench_required_path_edit(
            c, agent_id, world
        )
        pair_repair_edit = bool(
            at == "edit_repo_file"
            and params.get("_cooperbench_repair_source")
        )
        if (
            accepted_code
            and at in _PRE_COMMIT_ESCALATED
            and not (pair_required_edit or pair_repair_edit)
        ):
            return (
                f"{accepted_code} accepted code patch(es) pending; commit_patch / "
                "open_pr before more product work"
            )
        if _merge_ready_for_agent(world, agent_id) and at in _PRE_COMMIT_ESCALATED:
            return "an approved CI-passed PR is ready; merge_pr before more product work"
        # v11 coding layer: a forced release-blocker fix bypasses the CHURN guards —
        # fixing the blocker is exactly the work we want, even on an awaiting-review
        # file. It does NOT bypass the dead-end ceiling: an unconditional exemption is
        # unbounded, so a file whose every patch is rejected can absorb attempts
        # forever (observed: one file took 14 of ~20 patches while nine other issues
        # went untouched). Past the ceiling the org must diagnose instead of re-edit.
        if params.get("_blocker_fix"):
            if tgt and _forced_edit_dead_end(world, tgt):
                return (f"{tgt} has reached the forced-edit ceiling with no accepted "
                        "patch; diagnose (run tests / inspect) instead of re-editing")
            return None

        # 1. same agent repeated same safe action ON THE SAME OBJECT too recently
        #    (object actions cool down per-object; objectless actions per-action) — v4 §2.
        cd = ACTION_OBJECT_COOLDOWN.get(at, 0)
        if cd:
            last = _last_action_tick(world, agent_id, at, target=(tgt if tgt is not None else _ANY))
            if last is not None and tick - last < cd:
                return f"same agent repeated {at}{(' on ' + tgt) if tgt else ''} too recently"

        # 1b. v5 §P0-3: if you have accepted-but-uncommitted patches, FINISH the repo
        #     chain (commit_patch -> open_pr) before starting more edits/tools/docs —
        #     otherwise the LLM loops on use_tool/create_* and the workflow never advances.
        if uncommitted:
            if at in _PRE_COMMIT_BLOCKED:
                return "you have uncommitted patches; commit_patch / open_pr before more edits or tools"
            if (
                len(uncommitted) >= _PRE_COMMIT_ESCALATE_AT
                and at in _PRE_COMMIT_ESCALATED
                and not (pair_required_edit or pair_repair_edit)
            ):
                return (f"{len(uncommitted)} uncommitted patches; commit_patch / open_pr to advance "
                        "the repo chain before more product work")

        # 2. artifact revised too recently (still awaiting review) -> don't churn it
        if tgt and tgt in arts and at in ("edit_repo_file", "edit_doc", "audit_readme_claims"):
            art = arts[tgt]
            ocd = GLOBAL_OBJECT_REVISION_COOLDOWN.get(tgt, 0)
            upd = int(getattr(art, "updated_at_tick", 0) or 0)
            rev = int(getattr(art, "revision", 0) or 0)
            if ocd and rev > 0 and upd and tick - upd < ocd:
                return f"{tgt} revised too recently (awaiting review)"

        # 2b. all known gaps on this artifact are resolved -> no marginal value in another
        #     edit; commit/PR/review instead (v5 §P0-4/§P0-5).
        _EDIT_ACTIONS = ("edit_repo_file", "edit_doc", "audit_readme_claims",
                         "update_claim_tracker", "update_source_tracker")
        if tgt and tgt in arts and at in _EDIT_ACTIONS:
            art = arts[tgt]
            if int(getattr(art, "revision", 0) or 0) > 0 and not getattr(art, "known_gaps", []):
                return f"all known gaps on {tgt} are resolved; commit/PR/review instead of re-editing"
            rej = (getattr(world, "_patch_reject", {}) or {}).get((agent_id, tgt))
            if rej is not None and tick - rej < 6:
                return f"a patch on {tgt} was just rejected; cool down before re-editing"
            # v13 P4 no-op prevention: repeated duplicate/no-op rejects on this artifact (ANY
            # agent) mean re-editing isn't working — stop and diagnose the integration/contract.
            recent_rej = sum(1 for e in (getattr(world, "events", []) or [])[-150:]
                             if e.get("type") == "product_event" and e.get("subtype") == "patch_rejected"
                             and e.get("artifact_id") == tgt and tick - int(e.get("tick", 0) or 0) < 12)
            if recent_rej >= 2:
                return (f"{tgt} had {recent_rej} recent rejected patches; diagnose the integration "
                        "contract (run/inspect), don't re-edit")
            # cumulative dead-end (gpt-5 run finding: one self-made checklist doc absorbed 41
            # no-op patches over the run because the windowed check above kept resetting). If an
            # artifact has racked up many ALL-TIME rejected (mostly no-op) patches, stop editing it
            # entirely. The forced OSS code edit (_blocker_fix) is already exempt at the top.
            from environments.org_env.product.patch_validator import (
                is_infrastructure_rejection,
            )
            tot_rej = sum(1 for pp in (getattr(world, "patches", {}) or {}).values()
                          if getattr(pp, "validation_status", "") == "rejected"
                          and not is_infrastructure_rejection(pp)
                          and (getattr(pp, "target_object_id", None) == tgt
                               or getattr(pp, "artifact_id", None) == tgt))
            if tot_rej >= 3:
                return (f"{tgt} has {tot_rej} rejected (no-op) patches all-time; it is a dead end — "
                        "stop re-editing it")
            # v6 P0.1: once an artifact has had real revisions and is still awaiting
            #          review, stop piling on more edits — push it through the
            #          review/commit/PR/merge path first (this broke the README loop:
            #          12 revisions all "softened the overclaim", never reviewed/merged).
            if getattr(art, "awaiting_review", False) \
                    and int(getattr(art, "revision", 0) or 0) >= AWAITING_REVIEW_EDIT_BLOCK_REV:
                return (f"{tgt} is awaiting review (rev {art.revision}); "
                        "review/approve/commit/PR before editing it again")

        # 2d. #2: don't re-run the release readiness check if the blockers haven't changed
        #     since this agent last checked — resolve a blocker (work_on_task) instead.
        if at == "run_launch_readiness_check":
            changed = int(getattr(world, "_rc_blockers_changed_tick", 0) or 0)
            last = int((getattr(world, "_last_readiness_tick", {}) or {}).get(agent_id, -1))
            if last >= 0 and last >= changed:
                return "release blockers unchanged since your last readiness check; resolve a blocker first"

        # 2e. v8 #1: an established dispute (enough supporters) takes no more challenges —
        #     route to resolution (request_reproduction / resolve) instead of looping.
        if at == "challenge_result":
            d = self._dispute_for_result(world, params.get("result_id"))
            if d is not None and int(getattr(d, "support_count", 1) or 1) >= CHALLENGE_SUPPORT_CAP:
                return "dispute already established; request reproduction / resolve it instead of re-challenging"

        # 3. nothing NEW to monitor since this agent last monitored (v4 §3)
        if at in _MONITOR_ACTIONS and not self._has_external_signal(world, agent_id, tick):
            return "no new external signal to monitor"

        # 4. target task already done
        tid = params.get("task_id")
        if tid:
            t = (getattr(world, "tasks", {}) or {}).get(tid)
            if t is not None and getattr(getattr(t, "status", None), "value",
                                        str(getattr(t, "status", ""))) in ("done", "merged", "released"):
                return "target task already done"

        # 5. an equivalent doc already exists -> revise/review instead of re-create.
        #    (create_issue is NOT masked here: v6 P0.3 dedups it at the handler into a
        #    support bump on the existing issue, so the repeat still carries signal.)
        if at == "create_doc":
            title = (params.get("title") or "").strip().lower()
            if title and any(title == (getattr(a, "title", "") or "").strip().lower()
                             for a in arts.values()):
                return "document already exists; revise/review instead"

        # 6. purpose-level dedup for create-class doc actions (v4 review §1)
        purpose = _CREATE_PURPOSE.get(at)
        programbench_contract = bool(
            programbench_state is not None
            and at == "write_design_note"
            and params.get("_programbench_contract") is True
            and params.get("programbench_artifact_kind") == "behavioral_contract"
            and str(programbench_state.get("phase") or "") in {"explore", "develop"}
            and not any(
                str(
                    getattr(artifact, "programbench_artifact_kind", "") or ""
                )
                == "behavioral_contract"
                and int(getattr(artifact, "revision", 0) or 0) > 0
                and str(
                    getattr(
                        artifact,
                        "programbench_public_evidence_digest",
                        "",
                    )
                    or ""
                )
                == str(
                    programbench_state.get(
                        "exploration_reference_evidence_digest"
                    )
                    or ""
                )
                and str(getattr(artifact, "status", "") or "")
                not in {"deprecated", "closed"}
                for artifact in arts.values()
            )
        )
        if (
            purpose
            and self._active_purpose_artifact(arts, purpose, tick)
            and not programbench_contract
        ):
            return f"a {purpose} doc already exists; revise/review it instead of creating another"

        # 6b. meetings must be warranted + sparse (v8-run finding: 24 meetings, 0 decisions,
        #     Paul/Victor scheduling daily-sync at 3am). No meeting without a concrete
        #     trigger; at most one of a given type per day.
        if at == "schedule_meeting":
            mtype = params.get("meeting_type", "daily_sync")
            if self._meeting_type_today(world, mtype):
                return f"a {mtype} meeting was already scheduled today (one per type per day)"
            if not self._meeting_warranted(world):
                return "no concrete trigger (open blocker / dispute / pending proposal / crunch) for a meeting"

        # 7. #6 night rest: most agents sleep at night; only a founder may push heavy work
        #    during an urgent crunch (and pays the late-night fatigue cost in duration).
        #    Ablating work_rhythm is documented as making every agent continuously
        #    available, so this mask has to go with it. Left in, the ablation only
        #    removed the fatigue an agent would have paid while still refusing the
        #    turn, which cost the rhythm-off arms about a quarter of their decisions
        #    and made the ablation look worse than the mechanism it removed.
        time_system = getattr(world, "time", None)
        from environments.org_env.cooperbench.work_schedule import compressed_schedule_enabled
        if (at in _NIGHT_WORK_MASKABLE
                and getattr(time_system, "rhythm_enabled", True)
                and not compressed_schedule_enabled(world)):
            clk = getattr(time_system, "clock", None)
            if clk is not None and (getattr(clk, "phase_of_day", "") in ("night_sleep", "late_night")
                                    or getattr(clk, "is_late_night", False)):
                agent = (getattr(world, "agents", {}) or {}).get(agent_id)
                if not (bool(getattr(agent, "is_founder", False)) and self._urgent_now(world)):
                    return "night rest hours; resume in the morning (founders only on urgent crunch)"
        return None

    @staticmethod
    def _meeting_type_today(world, mtype: str) -> bool:
        ms = getattr(world, "meeting_system", None)
        if ms is None:
            return False
        day = int(getattr(world, "world_tick", 0)) // 24
        for m in ms.meetings.values():
            mt = int(getattr(m, "scheduled_tick", 0) or getattr(m, "created_tick", 0) or 0) // 24
            if getattr(m, "meeting_type", "") == mtype and mt == day:
                return True
        return False

    def _meeting_warranted(self, world) -> bool:
        # v11 #3: a meeting is NOT a default-safe action — warrant it only for real cross-role
        # coordination (proposal decision / open release blocker / crunch / escalated dispute),
        # NOT merely "a high-priority task exists" (that was used to avoid doing the work).
        pm = getattr(world, "proposal_manager", None)
        if pm is not None and any(p.status == "under_review" for p in pm.proposals.values()):
            return True
        for a in (getattr(world, "product_artifacts", {}) or {}).values():
            if (getattr(a, "artifact_type", "") == "issue" and getattr(a, "status", "") == "open"
                    and str(getattr(a, "artifact_id", "")).startswith("rel_blocker_")):
                return True
        return self._urgent_now(world)   # launch/incident crunch, escalated dispute, budget crisis

    @staticmethod
    def _urgent_now(world) -> bool:
        em = getattr(world, "episode_manager", None)
        for ep in (em.episodes.values() if em else []):
            if getattr(ep, "status", "") == "open" and getattr(ep, "episode_type", "") in (
                    "launch_crunch_episode", "incident_episode"):
                return True
        bp = float(getattr(getattr(getattr(world, "budget_system", None), "budget", None),
                           "budget_pressure", 0.0) or 0.0)
        if bp >= 0.7:
            return True
        cr = getattr(world, "commitment_registry", None)
        return bool(cr and any(getattr(d, "status", "") == "escalated" for d in cr.disputes.values()))

    @staticmethod
    def _active_purpose_artifact(arts, purpose: str, tick: int):
        """v5 §P0-1: HARD singleton — at most one active (non-deprecated/closed) artifact
        per purpose. Once one exists, creating another is masked (revise/review instead)."""
        from environments.org_env.product.objects import artifact_purpose
        for a in arts.values():
            if getattr(a, "status", "") in ("deprecated", "closed"):
                continue
            key = getattr(a, "linked_file_path", "") or getattr(a, "artifact_id", "")
            if artifact_purpose(key) == purpose:
                return a
        return None

    def _has_external_signal(self, world: Any, agent_id: str, tick: int) -> bool:
        """True only if there is a NEW external post since this agent last monitored —
        an existing frozen feed is not, by itself, a reason to monitor again (v4 §3)."""
        ext = getattr(world, "community", None) or getattr(world, "external", None)
        posts = getattr(ext, "posts", None) or {}
        if not posts:
            return False
        last = (getattr(world, "_last_monitor_tick", {}) or {}).get(agent_id)
        if last is None:
            return True                       # never monitored -> the feed is new to them
        try:
            return any(int(getattr(p, "created_tick", 0) or 0) > int(last) for p in posts.values())
        except Exception:
            return True   # fail-open: don't mask if we can't tell

    # -- soft penalties -----------------------------------------------------
    def penalties(self, c, agent_id: str, world: Any, tick: int) -> Dict[str, float]:
        at = c.action_type
        arts = getattr(world, "product_artifacts", {}) or {}
        tgt = candidate_target_artifact(c)
        out: Dict[str, float] = {}

        # weak, any-target: discourage (don't block) doing the same action repeatedly
        # even on a different object — the hard same-object cooldown lives in mask_reason.
        last = _last_action_tick(world, agent_id, at, target=_ANY)
        cd = ACTION_OBJECT_COOLDOWN.get(at, 0)
        if last is not None and cd and tick - last < 2 * cd:
            out["repetition_penalty"] = -0.18

        if tgt and tgt in arts:
            art = arts[tgt]
            recent_patches = sum(1 for _ in getattr(art, "patch_history_ids", []) or [])
            if recent_patches >= 2 and int(getattr(art, "updated_at_tick", 0) or 0) and \
                    tick - int(art.updated_at_tick) < 8:
                out["artifact_churn_penalty"] = -0.12
            if getattr(art, "awaiting_review", False):
                out["unresolved_review_penalty"] = -0.15

        if at in _MONITOR_ACTIONS and not self._has_external_signal(world, agent_id, tick):
            out["missing_new_information_penalty"] = -0.2

        if at in _DOC_ACTIONS and self._high_priority_impl_open(world):
            out["opportunity_cost"] = -0.12
        # spec #5: runway pressure raises cost-sensitivity — discourage expensive,
        # non-critical actions when the budget is tight (visible in the policy trace).
        bp = float(getattr(getattr(getattr(world, "budget_system", None), "budget", None),
                           "budget_pressure", 0.0) or 0.0)
        if bp >= 0.5 and at in _EXPENSIVE_ACTIONS:
            out["runway_pressure_penalty"] = round(-0.25 * bp, 3)
        # v8 #1: challenging a result has decreasing marginal value as its dispute
        # accumulates evidence requests — so critique doesn't become the safe high-score loop.
        if at == "challenge_result":
            d = self._dispute_for_result(world, (getattr(c, "parameters", None) or {}).get("result_id"))
            if d is not None:
                erq = int(getattr(d, "evidence_request_count", 1) or 1)
                out["challenge_saturation_penalty"] = round(-min(0.40, 0.05 * erq), 3)
        # v13 P3 shipping mode: a release smoke/CI/contract blocker is open -> converge on closing
        # it. Down-weight generic off-task busywork (broad search / new docs / generic meetings /
        # proposals / unrelated pilots). The forced blocker fix + closure actions are exempt.
        if at in _SHIPPING_OFFTASK and not (getattr(c, "parameters", None) or {}).get("_blocker_fix"):
            try:
                from environments.org_env.product.milestone import in_shipping_mode
                if in_shipping_mode(world):
                    out["shipping_mode_offtask"] = -0.5
            except Exception:
                pass
        # OSS time-machine (review fix §2/§3): an OPEN historical issue with a linked code module but
        # no patch yet -> converge on EDITING it. Down-weight generic soft busywork AND the no-op
        # `work_on_task` so the org doesn't loop on search/docs/proposals. The forced edit is exempt.
        if (at in _SHIPPING_OFFTASK or at == "work_on_task") \
                and not (getattr(c, "parameters", None) or {}).get("_blocker_fix") \
                and self._oss_unpatched_coding_open(world, tick):
            out["oss_unpatched_issue_offtask"] = -0.5
        # accepted-but-uncommitted patches -> finish the repo chain (commit/PR) instead of escaping
        # into soft actions (review fix §2: edits never reached mainline -> no release -> no market).
        if at in _UNCOMMITTED_SOFT_ESCAPE and not (getattr(c, "parameters", None) or {}).get("_blocker_fix"):
            from environments.org_env.backend.repo.workflow import pending_for_agent
            if pending_for_agent(world, agent_id):
                out["uncommitted_repo_chain_penalty"] = -0.5
        if at in _MERGE_READY_SOFT_ESCAPE and _merge_ready_for_agent(world, agent_id):
            out["merge_ready_delivery_penalty"] = -1.2
        # ship-it: a merged fix is sitting UNSHIPPED -> a LEAD should cut/publish a release, not idle.
        # Strong down-weight (rest_offline utility can dominate at night) so create_release_candidate /
        # publish wins the decision. The deterministic pipeline is the backstop; this makes leads ship
        # on their own too. Release-lifecycle actions themselves are never penalized.
        if at in _SHIP_IT_DOWNWEIGHT and not (getattr(c, "parameters", None) or {}).get("_blocker_fix"):
            role = getattr((getattr(world, "agents", {}) or {}).get(agent_id), "role", "")
            if role in ("founder", "cofounder") and self._has_shippable_work(world, tick):
                out["unshipped_release_idle"] = -1.2
        return out

    def _has_shippable_work(self, world: Any, tick: int) -> bool:
        """Whether a merged fix is sitting unshipped (cached per tick — penalties() runs per candidate)."""
        cache = getattr(world, "__dict__", {}).get("_shippable_cache")
        if cache and cache[0] == tick:
            return cache[1]
        val = False
        try:
            from environments.org_env.product.milestone import has_shippable_work
            val = bool(has_shippable_work(world))
        except Exception:
            val = False
        if hasattr(world, "__dict__"):
            world.__dict__["_shippable_cache"] = (tick, val)
        return val

    def _oss_unpatched_coding_open(self, world: Any, tick: int) -> bool:
        """Whether any OSS historical issue is open with a linked code module + no patch yet
        (cached per tick — penalties() is called per candidate)."""
        cache = getattr(world, "__dict__", {}).get("_oss_unpatched_cache")
        if cache and cache[0] == tick:
            return cache[1]
        val = False
        try:
            from environments.org_env.product.substrates.issue_stream import unpatched_coding_issues
            val = bool(unpatched_coding_issues(world))
        except Exception:
            val = False
        if hasattr(world, "__dict__"):
            world.__dict__["_oss_unpatched_cache"] = (tick, val)
        return val

    @staticmethod
    def _dispute_for_result(world, rid):
        if not rid:
            return None
        cr = getattr(world, "commitment_registry", None)
        for d in (cr.disputes.values() if cr else []):
            if getattr(d, "target_object_id", None) == rid \
                    and getattr(d, "status", "") in ("open", "escalated"):
                return d
        return None

    def _high_priority_impl_open(self, world: Any) -> bool:
        for t in (getattr(world, "tasks", {}) or {}).values():
            status = getattr(getattr(t, "status", None), "value", str(getattr(t, "status", "")))
            if status in ("open", "in_progress") and int(getattr(t, "priority", 3) or 3) <= 2:
                arts = getattr(t, "linked_artifacts", []) or []
                if any(str(a).endswith("_py") for a in arts):
                    return True
        return False

    # -- combined filter (used by the mapper) -------------------------------
    def filter(self, pool: List[Any], agent_id: str, world: Any, tick: int) -> Tuple[List[Any], List[Tuple[Any, str]]]:
        kept, masked = [], []
        for c in pool:
            r = self.mask_reason(c, agent_id, world, tick)
            (masked.append((c, r)) if r else kept.append(c))
        if not kept:                       # never-empty fallback, but NEVER re-admit a
            # create-class / safe-action attractor (v5 §P0-1/§P0-2) — better to idle than
            # to spam onboarding/use_tool/read_feed just to fill the pool.
            kept = [
                c
                for c, reason in masked
                if c.action_type not in _NO_FALLBACK
                and c.action_type not in ACTION_OBJECT_COOLDOWN
                # Adapted-profile authorization masks are physical workflow
                # constraints, not anti-attractor heuristics.  The generic
                # never-empty fallback must not give them back to an agent.
                and not str(reason or "").startswith("ProgramBench ")
                and not str(reason or "").startswith("only the ProgramBench ")
            ]
        return kept, masked


__all__ = ["AttractorGuard", "ACTION_OBJECT_COOLDOWN", "GLOBAL_OBJECT_REVISION_COOLDOWN",
           "candidate_target_artifact"]
