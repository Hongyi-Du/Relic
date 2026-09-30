"""Opt-in ProgramBench task-family profile primitives.

The native OrgEnv policy remains the default.  Nothing in this module is
activated by a task title or repository name: callers must request the exact
profile id and supply a public ProgramBench pack manifest.  This keeps the
benchmark adapter an explicit experimental treatment and makes accidental
attachment fail closed.

The profile may shape protocol work for leaderboard delivery, so its protocol
metrics are not evidence of unassisted institutional formation.  A phase
change first lets the rules inherited from the previous phase operate long
enough to produce public friction.  Only evidence-backed friction can then
open a repair window; ordinary agent deliberation is never blocked.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, fields, replace
from enum import Enum
import hashlib
import json
from pathlib import PurePosixPath
import re
from typing import Any, Iterable, Mapping, Sequence

from environments.org_env.product.repo_paths import (
    InvalidRepoPath,
    normalize_repo_relative_path,
)


PROFILE_ID = "programbench_leaderboard_v1"
PROFILE_STATE_KEY = "programbench_profile_state"
# v6 carries `public_behavior_coverage` and `public_probe_receipt_case_ceiling`.
# A v5 checkpoint has neither, and its exploration was gated on "one receipt
# fills the quota exactly", so resuming it under the cumulative rule would judge
# it by a rule it never ran under. The version moves so that refusal is legible.
PROFILE_STATE_SCHEMA_VERSION = "programbench_profile_state_v6"
WORKFLOW_OVERLAY_SCHEMA_VERSION = "programbench_workflow_overlay_v6"
COMPANY_BRIEF_OVERLAY_SCHEMA_VERSION = "programbench_company_brief_overlay_v1"
TASK_FAMILY = "programbench_cleanroom_reconstruction"
PROTOCOL_FRICTION_OBSERVATION_TICKS = 16
PROTOCOL_FRICTION_RECENCY_TICKS = 16
PROTOCOL_ADAPTATION_MIN_TICKS = 16
# Backwards-compatible import name.  It now names the *minimum* evidence-gated
# window, not a window opened unconditionally at each phase change.
PROTOCOL_ADAPTATION_TICKS = PROTOCOL_ADAPTATION_MIN_TICKS
PROTOCOL_FRICTION_SCHEMA_VERSION = "programbench_protocol_friction_v1"
EXPLORE_OBSERVATION_FULL_TICKS = 24
# The latter half of the fixed exploration floor is long enough for agents to
# notice recurring coordination needs, propose a rule in their own words, and
# still let the native review/adoption machinery operate before CONTRACT/PLAN.
# This is a policy preference only: it neither deals a written rule nor makes
# protocol formation a phase requirement.
EXPLORE_PROTOCOL_FORMATION_TICKS = 8
PROTOCOL_TRANSITION_COHORT_SCHEMA_VERSION = (
    "programbench_protocol_transition_cohort_v2"
)

_PUBLIC_DOCUMENT_PREFIXES = ("knowledge/", "tests/public/")
_PUBLIC_DOCUMENT_MAX_COUNT = 64
_PUBLIC_DOCUMENT_MAX_PATH_CHARS = 240
_PUBLIC_DOCUMENT_MAX_CONTENT_BYTES = 256 * 1024
_LEGAL_ONLY_BASENAMES = {"license", "copying", "notice"}
_SEALED_PATH_PARTS = {
    ".programbench",
    "hidden",
    "reference",
    "reference_repo",
    "evaluator",
    "evaluator_only",
    "oracle",
    "oracles",
    "private",
    ".private",
    "provenance",
    "contamination",
}

_BLOCKED_CONTEXTS_MIN = 3
_UNMEETABLE_VIOLATIONS_MIN = 4
_UNMEETABLE_VIOLATION_SHARE = 0.5
_EXHAUSTED_PATCH_ARTIFACTS_MIN = 2
_MAX_FRICTION_EVENT_REFS = 32

_PACK_SCHEMA = "org_env_oss_time_machine_pack_v2"
_ADAPTER_SCHEMA = "programbench_pack_eval_v1"
_OFFICIAL_RUNNER = "programbench_official_v1"
_ROOT_ISSUE_ID = "programbench_reconstruction"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
# A pack may bind exploration to 16, 32 or 64 public behaviours. The number is
# the organization's only measure of what "specified" means: every behaviour it
# does not observe is one it will never see itself get wrong. A 336-tick run at
# 16 satisfied all sixteen, reported zero open gaps and two open issues, ran out
# of work it could see, and finished 225/285 on the sealed suite -- sixty
# behaviours it had no instrument to notice.
#
# The number is a floor on cumulative distinct observations, not a cap on any
# one receipt. It used to be both, read from the same `limits.max_cases` field,
# which made iterative exploration impossible: a receipt had to fill the cap
# exactly, so the organization wrote the whole corpus blind in one shot, and a
# later edit to the definition threw away every observation already made.
#
# The set stays a whitelist rather than a range because an un-audited floor can
# still deadlock EXPLORE (too high to reach inside the horizon) or make the
# instrument meaningless (too low).
_SUPPORTED_PUBLIC_PROBE_CASE_QUOTAS = frozenset({16, 32, 64})
# Cumulative diversity minima. These are absolute, not proportional to the
# quota: they express the least variety that makes a corpus an instrument at
# all. Scaling them with the floor would reintroduce a hidden ceiling on how
# much the organization is allowed to learn per observation.
_MINIMUM_CUMULATIVE_PRIMARY_STIMULI = 12
_MAXIMUM_CUMULATIVE_ENV_ONLY_REPEATS = 4
_MINIMUM_CUMULATIVE_REFERENCE_OUTCOMES = 6
_AUDITED_PUBLIC_PROBE_EXACT_PATHS = ("eval/eval_stub.py",)
_AUDITED_PUBLIC_PROBE_PATH_PATTERNS = (r"^eval/eval_[0-9]+\.py$",)
_AUDITED_PUBLIC_PROBE_MAX_DEFINITIONS = 8
_BEHAVIOR_LEDGER_SCHEMA_VERSION = "programbench_public_behavior_ledger_v1"
_BEHAVIOR_COVERAGE_SCHEMA_VERSION = "programbench_public_behavior_coverage_v1"
_BEHAVIORAL_CONTRACT_ACCEPTANCE_SCHEMA_VERSION = (
    "programbench_behavioral_contract_acceptance_v1"
)
# v2 binds the cumulative coverage document instead of one receipt that filled
# the quota exactly. A v1 attestation cannot be re-checked under the cumulative
# rule -- it carries no `receipts` list and no `coverage_sha256` -- so it is
# refused rather than reinterpreted.
_DEVELOP_TRANSITION_ATTESTATION_SCHEMA_VERSION = (
    "programbench_develop_transition_attestation_v2"
)
_PUBLIC_PROBE_CONTRACT_STATE_SCHEMA_VERSION = (
    "programbench_frozen_public_probe_contract_v1"
)
_CANDIDATE_VIEW_MIGRATION_SCHEMA_VERSION = (
    "programbench_candidate_view_migration_v1"
)
_CANDIDATE_SURFACE_SCHEMA_VERSION = (
    "programbench_mainline_plus_integration_candidate_probe_isolated_v2"
)


class ProfileAttachmentError(ValueError):
    """The requested profile cannot be safely attached to this task."""


class ProgramBenchCoverageConflict(ValueError):
    """The same public input was observed twice with different reference results.

    Merging would have to pick a winner, and either choice is a lie: the
    accumulated ledger claims each row is a stable fact about the reference. A
    conflict means either the reference is non-deterministic on that input or
    the probe has a side effect, and both invalidate the case rather than the
    observation. It is raised so the disagreement is named at the moment it
    appears instead of surviving as whichever row happened to be merged last.
    """


class ProgramBenchPhase(str, Enum):
    """The only top-level states in the adapted workflow.

    Planning, implementation, verification and delivery remain useful work
    priorities inside ``DEVELOP``.  They are deliberately not lifecycle
    states: a successful merge is a baseline milestone and never terminates
    the 336-tick run.
    """

    EXPLORE = "explore"
    DEVELOP = "develop"


class ProgramBenchWorkRole(str, Enum):
    """Stable delivery roles; they are distinct from OrgEnv persona roles."""

    INTEGRATION_OWNER = "integration_owner"
    IMPLEMENTER = "implementer"
    PROBE_OWNER = "probe_owner"
    EXPLORER = "explorer"
    VERIFIER = "verifier"


_REQUIRED_PUBLIC_READER_ROLES = (
    ProgramBenchWorkRole.EXPLORER.value,
    ProgramBenchWorkRole.PROBE_OWNER.value,
    ProgramBenchWorkRole.INTEGRATION_OWNER.value,
)


@dataclass(frozen=True)
class TaskFamilyDetection:
    matched: bool
    reason: str
    task_family: str | None = None


@dataclass(frozen=True)
class ProgramBenchSignals:
    """Agent-visible evidence used to derive a phase.

    No evaluator or hidden-suite result belongs here.  A controller should
    populate these fields only from the public workspace and public probe
    receipts.
    """

    public_knowledge_reviewed: bool = False
    probe_inventory_nonempty: bool = False
    public_probe_execution_observed: bool = False
    exploration_case_quota_satisfied: bool = False
    behavior_ledger_complete: bool = False
    behavioral_contract_accepted: bool = False
    integration_owner_assigned: bool = False
    coherent_candidate_present: bool = False
    clean_root_compile_passed: bool = False
    executable_present: bool = False
    public_verification_complete: bool = False
    unresolved_public_mismatch_count: int = 0
    candidate_digest_frozen: bool = False

    def __post_init__(self) -> None:
        count = self.unresolved_public_mismatch_count
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("unresolved_public_mismatch_count_invalid")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProgramBenchPhaseState:
    phase: ProgramBenchPhase
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {"phase": self.phase.value, "phase_reason": self.reason}


@dataclass(frozen=True)
class ProgramBenchProtocolFriction:
    """Phase-scoped, public evidence that an inherited rule is now costly.

    The detector reads only the organization's protocol/repository event
    records.  It never traverses a ProgramBench evaluator, hidden suite, final
    score, or reference implementation.  ``target_metrics`` is deliberately
    bounded provenance: counts plus public event references, not event bodies.
    """

    phase: str
    phase_started_tick: int
    observed_at_tick: int
    eligible_protocol_ids: tuple[str, ...]
    friction_target_ids: tuple[str, ...]
    live_friction_target_ids: tuple[str, ...]
    merged_since_phase: int
    target_metrics: tuple[Mapping[str, Any], ...]
    evidence_digest: str | None

    @property
    def present(self) -> bool:
        return bool(self.live_friction_target_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PROTOCOL_FRICTION_SCHEMA_VERSION,
            "phase": self.phase,
            "phase_started_tick": self.phase_started_tick,
            "observed_at_tick": self.observed_at_tick,
            "eligible_protocol_ids": list(self.eligible_protocol_ids),
            "friction_target_ids": list(self.friction_target_ids),
            "live_friction_target_ids": list(self.live_friction_target_ids),
            "present": self.present,
            "merged_since_phase": self.merged_since_phase,
            "target_metrics": [dict(row) for row in self.target_metrics],
            "evidence_digest": self.evidence_digest,
        }


@dataclass(frozen=True)
class RoleAssignment:
    work_role: ProgramBenchWorkRole
    agent_id: str
    organization_role: str

    def to_dict(self) -> dict[str, str]:
        payload = asdict(self)
        payload["work_role"] = self.work_role.value
        return payload


@dataclass(frozen=True)
class PhaseActionDecision:
    phase: ProgramBenchPhase
    action_type: str
    allowed: bool
    bonus: float
    reason: str


@dataclass(frozen=True)
class ProfileAttachment:
    profile_id: str
    task_family: str
    initial_brief: str
    company_brief_overlay: Mapping[str, Any]
    product_context_labels: tuple[str, ...]
    workflow_overlay: Mapping[str, Any]
    phase_state: ProgramBenchPhaseState
    role_assignments: tuple[RoleAssignment, ...]


def detect_programbench_task_family(
    manifest: Mapping[str, Any] | None,
    root_issue: Mapping[str, Any] | None = None,
) -> TaskFamilyDetection:
    """Recognize only the frozen, public ProgramBench pack contract.

    Repository names, language labels, issue titles, and file extensions are
    deliberately not signals.  If a root issue is supplied it must also carry
    the pack builder's explicit benchmark provenance.
    """

    if not isinstance(manifest, Mapping):
        return TaskFamilyDetection(False, "public_manifest_required")
    if manifest.get("schema_version") != _PACK_SCHEMA:
        return TaskFamilyDetection(False, "pack_schema_mismatch")
    programbench = manifest.get("programbench")
    if not isinstance(programbench, Mapping):
        return TaskFamilyDetection(False, "programbench_contract_missing")
    if programbench.get("adapter_schema") != _ADAPTER_SCHEMA:
        return TaskFamilyDetection(False, "adapter_schema_mismatch")
    if programbench.get("runner") != _OFFICIAL_RUNNER:
        return TaskFamilyDetection(False, "programbench_runner_mismatch")
    task = programbench.get("task")
    if not isinstance(task, Mapping) or not all(
        isinstance(task.get(key), str) and bool(str(task.get(key)).strip())
        for key in ("instance_id", "repository", "commit")
    ):
        return TaskFamilyDetection(False, "programbench_task_identity_missing")
    strategy = manifest.get("test_strategy")
    if not isinstance(strategy, Mapping):
        return TaskFamilyDetection(False, "test_strategy_missing")
    if strategy.get("runner") != _OFFICIAL_RUNNER:
        return TaskFamilyDetection(False, "test_strategy_runner_mismatch")
    if strategy.get("allow_unbound_reconstruction_issue") is not True:
        return TaskFamilyDetection(False, "reconstruction_contract_missing")
    if root_issue is not None:
        if not isinstance(root_issue, Mapping):
            return TaskFamilyDetection(False, "root_issue_invalid")
        if root_issue.get("issue_id") != _ROOT_ISSUE_ID:
            return TaskFamilyDetection(False, "root_issue_id_mismatch")
        if root_issue.get("source") != "programbench_reconstruction":
            return TaskFamilyDetection(False, "root_issue_source_mismatch")
        if root_issue.get("source_type") != "benchmark_root_issue":
            return TaskFamilyDetection(False, "root_issue_type_mismatch")
    return TaskFamilyDetection(True, "programbench_contract_matched", TASK_FAMILY)


def programbench_root_task_brief(public_probe_case_quota: int = 16) -> str:
    """Return the task-family brief; it contains no instance-specific answer.

    The quota is a parameter because it is the pack's, not the family's, and it
    is the only number in this brief the organization cannot obtain anywhere
    else: public_probe_case_quota lives in the profile state and reaches no
    agent-facing surface. A brief that names 16 while the manifest demands 32
    does not under-specify the goal, it deadlocks EXPLORE, because the only
    instruction anyone ever read asked for a different number.

    The brief says "at least" and says observations accumulate, because the
    wording is what the organization optimises against. Under "exactly N in one
    receipt" a 32-case run produced a single 11186-character definition written
    before any reference output had been seen, and then never revised it: any
    edit would have discarded the whole observation. Telling it that batches add
    up is the difference between guessing a corpus and exploring one.
    """

    return f"""PROGRAMBENCH CLEAN-ROOM RECONSTRUCTION PROFILE

Objective: iteratively improve one coherent workspace whose externally
observable behavior matches the documented command-line program for the full
336-tick run.

This is behavioral reconstruction, not a conventional patch to a known
implementation. Work in this order:

1. EXPLORE — read the frozen, behaviorally relevant public contract surfaces
   (task documentation, public-probe authoring/schema material, and the compile
   entrypoint); inventory the documented CLI, input, output, exit, environment,
   filesystem, error, and boundary behavior; author non-empty declarative public
   probes and obtain actual execute-only public reference observations. The
   explorer, probe owner, and integration owner must each retrieve every required
   surface. Meet the manifest's {public_probe_case_quota}-case public quota,
   which is a floor on the number of DISTINCT inputs observed on the reference
   over the whole run, not a cap on any one probe run. Observations accumulate
   across probe runs and are deduplicated by input, so probe a small batch, read
   what the reference actually did, and let that decide what to probe next.
   Revising the probe definition between batches does not discard earlier
   observations. Probing more than the quota is allowed and useful; probing may
   continue after EXPLORE ends. EXPLORE lasts for all of ticks 1 through 24; the
   earliest transition is the tick-25 boundary. Candidate source, build, CI,
   pull-request and merge work is unavailable in EXPLORE.
2. DEVELOP/CONTRACT PRIORITY — record a versioned behavioral contract, select one
   architecture and source entry point, and assign exactly one integration
   owner. Source, build metadata, compile.sh, and probes form one system.
3. DEVELOP/IMPLEMENT PRIORITY — converge patches on the designated integration candidate. Do
   not create independent source-only and compile-only delivery branches.
4. DEVELOP/VERIFY PRIORITY — from a clean root, build ./executable and run public
   differential probes. Turn every public mismatch into a concrete repair tied
   to one artifact, owner, and public check.
5. Keep iterating probes, code, differential checks, CI, merges and repairs
   through tick 336. A verified merge is a baseline milestone, not completion.

Reference access is execute-only through the trusted public-probe runner. Never
read, copy, inspect, or mount reference files into the candidate workspace.
Evaluator-owned tests and results are sealed and cannot guide rollout choices.

There is no profile-derived COMPLETE state and no early stop. At final
evaluation, only a candidate with current-corpus, same-digest, zero-mismatch
public attestation may be primary; otherwise use the last strictly verified
mainline."""


def programbench_workflow_overlay() -> dict[str, Any]:
    """Return the frozen JSON-safe workflow contract consumed by controllers."""

    company_overlay = programbench_company_brief_overlay()
    return {
        "schema_version": WORKFLOW_OVERLAY_SCHEMA_VERSION,
        "profile_id": PROFILE_ID,
        "task_family": TASK_FAMILY,
        "initial_phase": ProgramBenchPhase.EXPLORE.value,
        "phase_order": [phase.value for phase in ProgramBenchPhase],
        "phase_labels": {
            ProgramBenchPhase.EXPLORE.value: "EXPLORE",
            ProgramBenchPhase.DEVELOP.value: "DEVELOP",
        },
        "required_public_artifacts": [
            "behavioral_contract",
            "public_probe_inventory",
            "public_probe_execution",
        ],
        "required_submission_artifacts": [
            "implementation_source",
            "compile_entrypoint",
            "executable",
        ],
        "single_integration_candidate": True,
        "integration_owner_required": True,
        "reference_access": "trusted_execute_only_public_probe_runner",
        "reference_probe_parameter": {
            "name": "probe_mode",
            "value": "reference_only",
        },
        "hidden_feedback_allowed": False,
        "protocol_transition_policy": {
            "scope": (
                "protocols_adopted_before_phase_transition_plus_"
                "evidence_linked_carryover_rules"
            ),
            "observation_ticks": PROTOCOL_FRICTION_OBSERVATION_TICKS,
            "friction_recency_ticks": PROTOCOL_FRICTION_RECENCY_TICKS,
            "minimum_adaptation_ticks": PROTOCOL_ADAPTATION_MIN_TICKS,
            "activation": "phase_scoped_public_friction_evidence",
            "late_entrant_observation_clock": "per_rule_full_ticks",
            "late_entrant_lineage": (
                "formed_in_previous_phase_or_frozen_public_evidence_reference"
            ),
            "repair_hard_blocked": False,
        },
        "exploration_transition_policy": {
            "minimum_full_action_ticks": EXPLORE_OBSERVATION_FULL_TICKS,
            "commit_boundary": "start_of_next_tick",
            "case_quota_source": (
                "public_probes.exploration_case_quota_"
                "defaulting_to_public_probes.limits.max_cases"
            ),
            "supported_case_quotas": sorted(_SUPPORTED_PUBLIC_PROBE_CASE_QUOTAS),
            "case_quota_rule": (
                "cumulative_distinct_reference_observations_at_least_the_quota"
            ),
            "case_quota_semantics": "floor_not_ceiling",
            "receipt_case_ceiling_source": "public_probes.limits.max_cases",
            "canonical_input_distinctness": (
                "argv_plus_stdin_plus_input_files_plus_env_sha256"
            ),
            "behavior_ledger_schema_version": _BEHAVIOR_LEDGER_SCHEMA_VERSION,
            "behavior_coverage_schema_version": (
                _BEHAVIOR_COVERAGE_SCHEMA_VERSION
            ),
            "cumulative_diversity_minima": {
                "distinct_primary_stimuli": _MINIMUM_CUMULATIVE_PRIMARY_STIMULI,
                "env_only_repeats_per_primary": (
                    _MAXIMUM_CUMULATIVE_ENV_ONLY_REPEATS
                ),
                "distinct_reference_outcomes": (
                    _MINIMUM_CUMULATIVE_REFERENCE_OUTCOMES
                ),
            },
            "public_document_readers": list(_REQUIRED_PUBLIC_READER_ROLES),
            "public_document_namespaces": list(_PUBLIC_DOCUMENT_PREFIXES),
            "probe_success_implies_document_review": False,
            "protocol_formation_preference_ticks": (
                EXPLORE_PROTOCOL_FORMATION_TICKS
            ),
            "protocol_formation_required": False,
        },
        "company_brief_overlay": company_overlay,
        "product_context_labels": list(PRODUCT_CONTEXT_LABELS),
        "run_horizon": {
            "ticks": 336,
            "early_stop": False,
            "merge_is_terminal": False,
        },
        "development_priorities": [
            "contract_and_owner",
            "candidate_implementation",
            "clean_build",
            "public_differential_and_repair",
            "ci_review_and_delivery",
            "postmerge_public_improvement",
        ],
    }


PRODUCT_CONTEXT_LABELS = (
    "clean_room_program_reconstruction",
    "empty_or_seedless_implementation",
    "public_documentation",
    "execute_only_public_probes",
    "single_coherent_executable_workspace",
)


def programbench_company_brief_overlay() -> dict[str, Any]:
    """Return the benchmark-family company narrative for prompt composition."""

    return {
        "schema_version": COMPANY_BRIEF_OVERLAY_SCHEMA_VERSION,
        "headline": "Clean-room program reconstruction organization",
        "mission": (
            "Explore documented public behavior, reconstruct the program from an "
            "empty or seedless implementation, and deliver one coherent executable "
            "workspace."
        ),
        "product_context": {
            "task_mode": "clean_room_program_reconstruction",
            "implementation_state": "empty_or_seedless",
            "evidence_surface": "public_documentation_and_execute_only_public_probes",
            "delivery_target": "single_coherent_executable_workspace",
        },
        "operating_principles": [
            "Explore public behavior before implementation.",
            "Use only public documentation and trusted execute-only public probes.",
            "Converge source, build contract, probes, and fixes on one candidate.",
            "Treat a clean-root executable build as delivery evidence, not file activity.",
        ],
    }


def _agent_field(agent: Any, name: str) -> str:
    if isinstance(agent, Mapping):
        value = agent.get(name)
    else:
        value = getattr(agent, name, None)
    return str(value or "").strip()


_ROLE_PREFERENCES: dict[ProgramBenchWorkRole, tuple[str, ...]] = {
    ProgramBenchWorkRole.INTEGRATION_OWNER: (
        "cofounder",
        "reliability",
        "founder",
        "fast_engineer",
        "artifact_design",
        "editorial",
    ),
    ProgramBenchWorkRole.IMPLEMENTER: (
        "fast_engineer",
        "cofounder",
        "reliability",
        "artifact_design",
    ),
    ProgramBenchWorkRole.PROBE_OWNER: (
        "reliability",
        "cofounder",
        "artifact_design",
        "editorial",
    ),
    ProgramBenchWorkRole.EXPLORER: (
        "artifact_design",
        "editorial",
        "cofounder",
        "reliability",
    ),
    ProgramBenchWorkRole.VERIFIER: (
        "founder",
        "editorial",
        "reliability",
        "cofounder",
    ),
}


def assign_fixed_roles(agents: Iterable[Any]) -> tuple[RoleAssignment, ...]:
    """Assign delivery roles deterministically, independent of mapping order.

    Agent ids are the final tie breaker.  Each agent receives at most one role
    while unused agents remain; small teams then reuse the best stable match.
    """

    if isinstance(agents, Mapping):
        roster: list[tuple[str | None, Any]] = [
            (str(key), agents[key])
            for key in sorted(agents, key=lambda item: str(item))
        ]
    else:
        roster = [(None, agent) for agent in agents]

    normalized: list[tuple[str, str]] = []
    seen: set[str] = set()
    for fallback_id, agent in roster:
        agent_id = (
            _agent_field(agent, "agent_id")
            or _agent_field(agent, "id")
            or str(fallback_id or "")
        )
        organization_role = _agent_field(agent, "role")
        if not agent_id:
            raise ValueError("programbench_role_agent_id_missing")
        if agent_id in seen:
            raise ValueError("programbench_role_agent_id_duplicate")
        seen.add(agent_id)
        normalized.append((agent_id, organization_role))
    if not normalized:
        raise ValueError("programbench_role_roster_empty")
    normalized.sort(key=lambda item: (item[0].casefold(), item[0]))

    assigned_ids: set[str] = set()
    result: list[RoleAssignment] = []
    for work_role, preferences in _ROLE_PREFERENCES.items():
        preference_rank = {role: index for index, role in enumerate(preferences)}

        def rank(item: tuple[str, str], *, allow_reuse: bool) -> tuple[int, int, str]:
            agent_id, organization_role = item
            used = int(agent_id in assigned_ids) if not allow_reuse else 0
            return (
                used,
                preference_rank.get(organization_role, len(preference_rank)),
                agent_id.casefold(),
            )

        unused = [item for item in normalized if item[0] not in assigned_ids]
        pool = unused or normalized
        chosen = min(pool, key=lambda item: rank(item, allow_reuse=not unused))
        assigned_ids.add(chosen[0])
        result.append(
            RoleAssignment(
                work_role=work_role,
                agent_id=chosen[0],
                organization_role=chosen[1],
            )
        )
    return tuple(result)


def _safe_agent_visible_path(raw_path: Any) -> str | None:
    """Normalize one public path without admitting sealed namespaces."""

    text = str(raw_path or "").strip().replace("\\", "/")
    if (
        not text
        or len(text) > _PUBLIC_DOCUMENT_MAX_PATH_CHARS
        or text.startswith("/")
    ):
        return None
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or ":" in (path.parts[0] if path.parts else "")
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        return None
    normalized = path.as_posix()
    lowered = normalized.casefold()
    for part in path.parts:
        tokens = {
            token
            for token in re.split(r"[^a-z0-9.]+", part.casefold())
            if token
        }
        if tokens & _SEALED_PATH_PARTS:
            return None
    return normalized


def programbench_public_path_is_agent_visible(raw_path: Any) -> bool:
    """Whether a path is safe to render in an adapted agent prompt."""

    return _safe_agent_visible_path(raw_path) is not None


def _heuristic_behavioral_document_path(raw_path: Any) -> str | None:
    """Return a generic public behavioral-doc path supplied by the seed pack."""

    normalized = _safe_agent_visible_path(raw_path)
    if normalized is None:
        return None
    lowered = normalized.casefold()
    if not lowered.endswith(".md") or not lowered.startswith(
        _PUBLIC_DOCUMENT_PREFIXES
    ):
        return None
    if PurePosixPath(lowered).stem in _LEGAL_ONLY_BASENAMES:
        return None
    return normalized


def _manifest_behavioral_public_paths(
    manifest: Mapping[str, Any],
    *,
    seed_paths: set[str],
) -> tuple[dict[str, str | None], set[str]]:
    """Extract only agent-visible behavioral contract surfaces from the pack.

    Provenance legal/attribution payloads remain readable in the workspace but
    are deliberately not workflow evidence.  The returned paths describe the
    command contract, public probe authoring contract, and task documentation;
    they contain no evaluator or reference path.
    """

    # The boolean marks a manifest-declared path whose spelling is itself part
    # of the public contract.  Compile command argv is different: dots and
    # slashes can occur in interpreter versions and flags, so argv tokens are
    # only a fallback when they exactly name a t0 product artifact.
    raw_paths: list[tuple[Any, Any, bool]] = []
    compile_tokens: list[Any] = []
    explicit_contract_entrypoint = False
    programbench = manifest.get("programbench")
    if isinstance(programbench, Mapping):
        provenance = programbench.get("provenance")
        if isinstance(provenance, Mapping):
            documents = provenance.get("cleanroom_public_documents")
            for row in documents if isinstance(documents, list) else []:
                if isinstance(row, Mapping):
                    raw_paths.append(
                        (row.get("starter_path"), row.get("sha256"), True)
                    )
        cleanroom = programbench.get("cleanroom")
        if isinstance(cleanroom, Mapping):
            entrypoint = cleanroom.get("contract_entrypoint")
            if str(entrypoint or "").strip():
                explicit_contract_entrypoint = True
                raw_paths.append((entrypoint, None, True))

    probes = manifest.get("public_probes")
    if isinstance(probes, Mapping):
        raw_paths.append((probes.get("schema_path"), None, True))
        compile_contract = probes.get("compile")
        if isinstance(compile_contract, Mapping):
            compile_tokens.extend(compile_contract.get("command") or [])

    entrypoints = manifest.get("entrypoints")
    if isinstance(entrypoints, Mapping):
        compile_entry = entrypoints.get("compile")
        if isinstance(compile_entry, Mapping):
            compile_tokens.extend(compile_entry.get("command") or [])

    if not explicit_contract_entrypoint:
        for token in compile_tokens:
            path = _safe_agent_visible_path(token)
            if path is not None and path in seed_paths:
                raw_paths.append((path, None, False))

    normalized: dict[str, str | None] = {}
    for raw_path, expected_digest, explicit in raw_paths:
        path = _safe_agent_visible_path(raw_path)
        if path is None:
            text = str(raw_path or "").strip()
            if explicit and text:
                raise ProfileAttachmentError(
                    "programbench_manifest_public_document_path_unsafe"
                )
            continue
        basename = PurePosixPath(path.casefold()).stem
        if basename in _LEGAL_ONLY_BASENAMES:
            continue
        digest = str(expected_digest or "") or None
        if digest is not None and not _SHA256.fullmatch(digest):
            raise ProfileAttachmentError(
                "programbench_manifest_public_document_digest_invalid"
            )
        prior = normalized.get(path)
        if prior is not None and digest is not None and prior != digest:
            raise ProfileAttachmentError(
                "programbench_manifest_public_document_digest_conflict"
            )
        normalized[path] = prior or digest
    inferred = (
        {"tests/public/README.md"}
        if isinstance(probes, Mapping)
        else set()
    )
    return normalized, inferred


def _artifact_public_text(artifact: Any) -> str:
    return str(
        getattr(artifact, "mainline_content", "")
        or getattr(artifact, "content", "")
        or ""
    )


def _freeze_required_public_documents(
    world: Any, manifest: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Freeze the seed public-doc allowlist at profile attachment.

    Agent-created notes never become exploration requirements.  The allowlist
    is derived only from already agent-visible product artifacts and admits at
    least the ProgramBench ``knowledge/*.md`` and ``tests/public/*.md``
    namespaces while rejecting reference/evaluator/oracle/hidden paths.
    """

    rows: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    artifacts = getattr(world, "product_artifacts", {}) or {}
    seed_paths = {
        path
        for artifact in artifacts.values()
        if isinstance(getattr(artifact, "created_at_tick", 0), int)
        and not isinstance(getattr(artifact, "created_at_tick", 0), bool)
        and int(getattr(artifact, "created_at_tick", 0)) == 0
        for path in [
            _safe_agent_visible_path(
                getattr(artifact, "linked_file_path", "")
            )
        ]
        if path is not None
    }
    manifest_paths, inferred_paths = _manifest_behavioral_public_paths(
        manifest,
        seed_paths=seed_paths,
    )
    for artifact_id, artifact in sorted(
        artifacts.items(), key=lambda item: str(item[0])
    ):
        created = getattr(artifact, "created_at_tick", 0)
        if isinstance(created, bool) or not isinstance(created, int) or created != 0:
            continue
        raw_path = getattr(artifact, "linked_file_path", "")
        path = _safe_agent_visible_path(raw_path)
        if path is None and len(str(raw_path or "")) > _PUBLIC_DOCUMENT_MAX_PATH_CHARS:
            normalized_prefix = str(raw_path or "").replace("\\", "/").casefold()
            if normalized_prefix.startswith(_PUBLIC_DOCUMENT_PREFIXES):
                raise ProfileAttachmentError(
                    "programbench_public_document_path_too_long"
                )
        if path is not None and (
            path not in manifest_paths
            and path not in inferred_paths
            and _heuristic_behavioral_document_path(path) is None
        ):
            path = None
        if path is None:
            continue
        if path in seen_paths:
            raise ProfileAttachmentError(
                "programbench_public_document_path_duplicated:" + path
            )
        seen_paths.add(path)
        body = _artifact_public_text(artifact)
        if len(body.encode()) > _PUBLIC_DOCUMENT_MAX_CONTENT_BYTES:
            raise ProfileAttachmentError(
                "programbench_public_document_content_too_large"
            )
        content_digest = hashlib.sha256(body.encode()).hexdigest()
        expected_digest = manifest_paths.get(path)
        if expected_digest is not None and content_digest != expected_digest:
            raise ProfileAttachmentError(
                "programbench_manifest_public_document_digest_mismatch:" + path
            )
        rows.append(
            {
                "artifact_id": str(artifact_id),
                "path": path,
                "seed_content_sha256": content_digest,
            }
        )
    if len(rows) > _PUBLIC_DOCUMENT_MAX_COUNT:
        raise ProfileAttachmentError("programbench_public_document_inventory_too_large")
    missing = sorted(set(manifest_paths) - seen_paths)
    if missing:
        raise ProfileAttachmentError(
            "programbench_manifest_public_document_missing:" + missing[0]
        )
    return sorted(rows, key=lambda row: (row["path"], row["artifact_id"]))


def _required_role_agents(state: Mapping[str, Any]) -> dict[str, str]:
    assignments = state.get("role_assignments")
    if not isinstance(assignments, list):
        return {}
    result: dict[str, str] = {}
    for row in assignments:
        if not isinstance(row, Mapping):
            continue
        role = str(row.get("work_role") or "")
        agent_id = str(row.get("agent_id") or "")
        if role in _REQUIRED_PUBLIC_READER_ROLES and agent_id:
            result[role] = agent_id
    return result


def _valid_public_document_receipt(
    row: Any,
    *,
    required: Mapping[str, Mapping[str, str]],
    role_agents: Mapping[str, str],
) -> bool:
    """Validate one checkpointed retrieval receipt against frozen provenance."""

    if not isinstance(row, Mapping):
        return False
    role = str(row.get("work_role") or "")
    artifact_id = str(row.get("artifact_id") or "")
    expected = required.get(artifact_id)
    tick = row.get("tick")
    return bool(
        role in _REQUIRED_PUBLIC_READER_ROLES
        and expected is not None
        and isinstance(tick, int)
        and not isinstance(tick, bool)
        and tick >= 0
        and str(row.get("agent_id") or "") == role_agents.get(role)
        and str(row.get("path") or "") == expected["path"]
        and str(row.get("content_sha256") or "")
        == expected["seed_content_sha256"]
    )


def _public_document_coverage(state: Mapping[str, Any]) -> dict[str, Any]:
    inventory = state.get("required_public_documents")
    receipts = state.get("public_document_read_receipts")
    required = {
        str(row.get("artifact_id") or ""): {
            "path": str(row.get("path") or ""),
            "seed_content_sha256": str(row.get("seed_content_sha256") or ""),
        }
        for row in inventory or []
        if isinstance(row, Mapping) and row.get("artifact_id")
    }
    required_ids = set(required)
    role_agents = _required_role_agents(state)
    observed: dict[str, set[str]] = {
        role: set() for role in _REQUIRED_PUBLIC_READER_ROLES
    }
    for row in receipts or []:
        if not _valid_public_document_receipt(
            row,
            required=required,
            role_agents=role_agents,
        ):
            continue
        role = str(row.get("work_role") or "")
        artifact_id = str(row.get("artifact_id") or "")
        observed[role].add(artifact_id)
    role_complete = {
        role: bool(required_ids) and required_ids.issubset(observed[role])
        for role in _REQUIRED_PUBLIC_READER_ROLES
    }
    return {
        "required_document_count": len(required_ids),
        "role_complete": role_complete,
        "complete": bool(required_ids) and all(role_complete.values()),
    }


def programbench_required_public_documents(
    world: Any, agent_id: str | None = None
) -> tuple[dict[str, Any], ...]:
    """Return frozen, safe documents still required from one designated reader."""

    if not programbench_profile_active(world):
        return ()
    state = world.__dict__[PROFILE_STATE_KEY]
    role_agents = _required_role_agents(state)
    aid = str(agent_id or "")
    roles = {role for role, owner in role_agents.items() if owner == aid}
    if agent_id is not None and not roles:
        return ()
    required = {
        str(row.get("artifact_id") or ""): {
            "path": str(row.get("path") or ""),
            "seed_content_sha256": str(row.get("seed_content_sha256") or ""),
        }
        for row in state.get("required_public_documents") or []
        if isinstance(row, Mapping) and row.get("artifact_id")
    }
    read_ids = {
        str(row.get("artifact_id") or "")
        for row in state.get("public_document_read_receipts") or []
        if _valid_public_document_receipt(
            row,
            required=required,
            role_agents=role_agents,
        )
        and (agent_id is None or str(row.get("work_role") or "") in roles)
    }
    return tuple(
        dict(row)
        for row in state.get("required_public_documents") or []
        if isinstance(row, Mapping)
        and (agent_id is None or str(row.get("artifact_id") or "") not in read_ids)
    )


def record_programbench_public_document_read(
    world: Any,
    *,
    agent_id: str,
    artifact_id: str,
    tick: int,
) -> dict[str, Any]:
    """Record one successful targeted read and stage audited coverage evidence."""

    if not programbench_profile_active(world):
        raise ProfileAttachmentError("programbench_profile_not_active")
    state = world.__dict__[PROFILE_STATE_KEY]
    target = next(
        (
            row
            for row in state.get("required_public_documents") or []
            if isinstance(row, Mapping)
            and str(row.get("artifact_id") or "") == str(artifact_id)
        ),
        None,
    )
    if target is None:
        raise ValueError("programbench_public_document_not_allowlisted")
    artifact = (getattr(world, "product_artifacts", {}) or {}).get(artifact_id)
    live_path = _safe_agent_visible_path(
        getattr(artifact, "linked_file_path", "") if artifact is not None else ""
    )
    if artifact is None or live_path != target.get("path"):
        raise ValueError("programbench_public_document_target_invalid")
    role_agents = _required_role_agents(state)
    roles = [
        role
        for role in _REQUIRED_PUBLIC_READER_ROLES
        if role_agents.get(role) == str(agent_id)
    ]
    if not roles:
        raise ValueError("programbench_public_document_reader_not_designated")
    receipts = state.setdefault("public_document_read_receipts", [])
    content_digest = hashlib.sha256(
        _artifact_public_text(artifact).encode()
    ).hexdigest()
    if content_digest != str(target.get("seed_content_sha256") or ""):
        raise ValueError("programbench_public_document_changed_since_attachment")
    for role in roles:
        if any(
            _valid_public_document_receipt(
                row,
                required={
                    str(artifact_id): {
                        "path": str(target.get("path") or ""),
                        "seed_content_sha256": str(
                            target.get("seed_content_sha256") or ""
                        ),
                    }
                },
                role_agents=role_agents,
            )
            and str(row.get("work_role") or "") == role
            for row in receipts
        ):
            continue
        receipts.append(
            {
                "work_role": role,
                "agent_id": str(agent_id),
                "artifact_id": str(artifact_id),
                "path": live_path,
                "content_sha256": content_digest,
                "tick": int(tick),
            }
        )
    coverage = _public_document_coverage(state)
    state["public_document_coverage"] = coverage
    update_programbench_signals(
        world, public_knowledge_reviewed=bool(coverage["complete"])
    )
    return coverage


def programbench_retrieved_public_surfaces(
    world: Any, agent_id: str
) -> tuple[dict[str, Any], ...]:
    """Render this agent's successful targeted retrievals for cognition prompts.

    The inventory carries hashes, not document bodies.  We therefore render a
    body only while the live seed/mainline bytes still match the frozen digest;
    working-tree edits, forged downloaded ids, and another agent's receipts do
    not enter cognition.  At most four surfaces, 8 KiB each and 16 KiB total,
    keep reflection/checkpoint prompt growth bounded while fitting the current
    public CLI README without truncation.
    """

    if not programbench_profile_active(world):
        return ()
    state = world.__dict__[PROFILE_STATE_KEY]
    inventory = {
        str(row.get("artifact_id") or ""): row
        for row in state.get("required_public_documents") or []
        if isinstance(row, Mapping) and row.get("artifact_id")
    }
    role_agents = _required_role_agents(state)
    receipts = state.get("public_document_read_receipts") or []
    retrieved: dict[str, int] = {}
    for row in receipts:
        if not isinstance(row, Mapping) or str(row.get("agent_id") or "") != str(
            agent_id
        ):
            continue
        artifact_id = str(row.get("artifact_id") or "")
        target = inventory.get(artifact_id)
        role = str(row.get("work_role") or "")
        tick = row.get("tick")
        if (
            target is not None
            and role_agents.get(role) == str(agent_id)
            and str(row.get("path") or "") == str(target.get("path") or "")
            and str(row.get("content_sha256") or "")
            == str(target.get("seed_content_sha256") or "")
            and isinstance(tick, int)
            and not isinstance(tick, bool)
        ):
            retrieved[artifact_id] = min(tick, retrieved.get(artifact_id, tick))
    if not retrieved:
        return ()

    artifacts = getattr(world, "product_artifacts", {}) or {}
    out: list[dict[str, Any]] = []
    total_chars = 0
    for target in sorted(
        (
            row
            for row in state.get("required_public_documents") or []
            if isinstance(row, Mapping)
            and str(row.get("artifact_id") or "") in retrieved
        ),
        key=lambda row: (str(row.get("path") or ""), str(row.get("artifact_id") or "")),
    ):
        if len(out) >= 4 or total_chars >= 16 * 1024:
            break
        artifact_id = str(target.get("artifact_id") or "")
        path = _safe_agent_visible_path(target.get("path"))
        artifact = artifacts.get(artifact_id)
        if artifact is None or path is None:
            continue
        live_path = _safe_agent_visible_path(
            getattr(artifact, "linked_file_path", "")
        )
        body = _artifact_public_text(artifact)
        digest = hashlib.sha256(body.encode()).hexdigest()
        if (
            live_path != path
            or digest != str(target.get("seed_content_sha256") or "")
        ):
            continue
        limit = min(8 * 1024, 16 * 1024 - total_chars)
        text = body[:limit]
        total_chars += len(text)
        out.append(
            {
                "artifact_id": artifact_id,
                "file": path,
                "text": text,
                "truncated": len(text) < len(body),
                "seed_content_sha256": digest,
                "retrieved_at_tick": retrieved[artifact_id],
            }
        )
    return tuple(out)


def _development_priority(signals: ProgramBenchSignals) -> str:
    """Return a public-evidence work priority without creating another phase."""

    if not (
        signals.behavioral_contract_accepted
        and signals.integration_owner_assigned
    ):
        return "contract_and_owner"
    if not signals.coherent_candidate_present:
        return "candidate_implementation"
    if not (signals.clean_root_compile_passed and signals.executable_present):
        return "clean_build"
    if (
        not signals.public_verification_complete
        or signals.unresolved_public_mismatch_count != 0
    ):
        return "public_differential_and_repair"
    if not signals.candidate_digest_frozen:
        return "ci_review_and_delivery"
    return "postmerge_public_improvement"


def derive_phase(signals: ProgramBenchSignals) -> ProgramBenchPhaseState:
    """Derive one of the two top-level states from public evidence."""

    if not (
        signals.public_knowledge_reviewed
        and signals.probe_inventory_nonempty
        and signals.public_probe_execution_observed
        and signals.exploration_case_quota_satisfied
        and signals.behavior_ledger_complete
        and signals.behavioral_contract_accepted
        and signals.integration_owner_assigned
    ):
        return ProgramBenchPhaseState(
            ProgramBenchPhase.EXPLORE,
            "exploration_floor_or_public_evidence_incomplete",
        )
    return ProgramBenchPhaseState(
        ProgramBenchPhase.DEVELOP,
        "development_priority:" + _development_priority(signals),
    )


def _adopted_protocol_ids(world: Any) -> tuple[str, ...]:
    manager = getattr(world, "proposal_manager", None)
    specs = getattr(manager, "protocol_specs", None) or {}
    return tuple(
        sorted(
            str(protocol_id)
            for protocol_id, spec in specs.items()
            if str(getattr(spec, "status", "") or "") == "adopted"
        )
    )


def _protocol_spec_id(world: Any, protocol_id: Any) -> str:
    """Map the registry's ``proto_spec_N`` id to ``protospec_N`` safely."""

    raw = str(protocol_id or "")
    manager = getattr(world, "proposal_manager", None)
    specs = getattr(manager, "protocol_specs", None) or {}
    if raw in specs:
        return raw
    mirror_id = getattr(world, "_registry_mirror_id", None)
    for candidate_id, spec in specs.items():
        mirrored = (
            str(mirror_id(getattr(spec, "protocol_id", candidate_id)))
            if callable(mirror_id)
            else f"proto_spec_{str(candidate_id).split('_')[-1]}"
        )
        if mirrored == raw:
            return str(candidate_id)
    return raw


def _canonical_digest(payload: Any) -> str:
    rendered = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _bounded_refs(values: Iterable[str]) -> list[str]:
    return sorted({str(value) for value in values if str(value)})[
        -_MAX_FRICTION_EVENT_REFS:
    ]


def _event_value(event: Any, name: str, default: Any = None) -> Any:
    if isinstance(event, Mapping):
        return event.get(name, default)
    return getattr(event, name, default)


def _registry_adoption_tick(registry: Any, protocol_id: str) -> int | None:
    ticks = [
        int(_event_value(event, "tick", 0) or 0)
        for event in (getattr(registry, "events", None) or [])
        if str(_event_value(event, "protocol_id", "") or "") == protocol_id
        and str(_event_value(event, "event_type", "") or "") == "adoption"
    ]
    return min(ticks) if ticks else None


def _protocol_lineage_refs(world: Any, spec: Any) -> set[str]:
    """Return exact public-object/event provenance carried by one rule.

    No protocol prose is interpreted here.  A late rule joins the cohort only
    through structured ids/digests that were frozen at the phase boundary, or
    because its proposal itself predates that boundary.
    """

    refs = {
        str(item)
        for field_name in (
            "problem_evidence",
            "source_episode_ids",
            "source_wish_ids",
        )
        for item in (getattr(spec, field_name, None) or [])
        if str(item)
    }
    for field_name in (
        "source_episode_id",
        "source_wish_id",
        "source_reflection_id",
    ):
        value = str(getattr(spec, field_name, "") or "")
        if value:
            refs.add(value)

    manager = getattr(world, "proposal_manager", None)
    proposal = (getattr(manager, "proposals", None) or {}).get(
        str(getattr(spec, "created_from_proposal_id", "") or "")
    )
    if proposal is not None:
        for field_name in (
            "source_event_ids",
            "source_episode_ids",
            "source_wish_ids",
            "affected_objects",
            "required_artifacts",
        ):
            refs.update(
                str(item)
                for item in (getattr(proposal, field_name, None) or [])
                if str(item)
            )
        for field_name in (
            "source_episode_id",
            "source_wish_id",
            "source_reflection_id",
        ):
            value = str(getattr(proposal, field_name, "") or "")
            if value:
                refs.add(value)

    reflection = getattr(world, "reflection_manager", None)
    wishes = getattr(reflection, "wishes", None) or {}
    for wish_id in tuple(refs):
        wish = wishes.get(wish_id)
        if wish is None:
            continue
        for field_name in (
            "source_event_ids",
            "source_episode_ids",
            "related_object_ids",
            "related_issue_ids",
        ):
            refs.update(
                str(item)
                for item in (getattr(wish, field_name, None) or [])
                if str(item)
            )
    return refs


def _adopted_protocol_records(world: Any) -> tuple[dict[str, Any], ...]:
    """Canonicalize ProtocolSpecs and their live-registry mirrors once.

    ProposalManager ids (``protospec_N``) are canonical when a spec exists;
    standalone registry rules retain their registry id.  This prevents a
    mirrored rule from receiving two observation clocks or double-counting its
    friction events.
    """

    from environments.org_env.backend.protocol.registry import protocol_is_live

    manager = getattr(world, "proposal_manager", None)
    specs = getattr(manager, "protocol_specs", None) or {}
    records: dict[str, dict[str, Any]] = {}
    for raw_id, spec in sorted(specs.items(), key=lambda item: str(item[0])):
        if str(getattr(spec, "status", "") or "") != "adopted":
            continue
        protocol_id = str(raw_id)
        created_raw = getattr(spec, "created_at_tick", None)
        created = (
            int(created_raw)
            if isinstance(created_raw, int) and not isinstance(created_raw, bool)
            else 0
        )
        adopted_raw = getattr(spec, "adopted_at_tick", None)
        adopted = (
            int(adopted_raw)
            if isinstance(adopted_raw, int) and not isinstance(adopted_raw, bool)
            else created
        )
        records[protocol_id] = {
            "canonical_protocol_id": protocol_id,
            "protocol_spec_id": protocol_id,
            "registry_protocol_id": None,
            "created_tick": created,
            "adopted_tick": adopted,
            "timing_attested": bool(
                isinstance(created_raw, int)
                and not isinstance(created_raw, bool)
                and isinstance(adopted_raw, int)
                and not isinstance(adopted_raw, bool)
            ),
            "lineage_refs": sorted(_protocol_lineage_refs(world, spec)),
        }

    registry = getattr(world, "protocol_registry", None)
    for raw_id, protocol in sorted(
        (getattr(registry, "protocols", None) or {}).items(),
        key=lambda item: str(item[0]),
    ):
        if (
            not protocol_is_live(protocol)
            or str(getattr(protocol, "adoption_status", "") or "")
            != "adopted"
        ):
            continue
        registry_id = str(raw_id)
        mapped = _protocol_spec_id(world, registry_id)
        canonical = mapped if mapped in records else registry_id
        created = int(getattr(protocol, "first_tick", 0) or 0)
        adopted = _registry_adoption_tick(registry, registry_id)
        row = records.setdefault(
            canonical,
            {
                "canonical_protocol_id": canonical,
                "protocol_spec_id": None,
                "registry_protocol_id": registry_id,
                "created_tick": created,
                "adopted_tick": adopted if adopted is not None else created,
                "timing_attested": adopted is not None,
                "lineage_refs": [],
            },
        )
        row["registry_protocol_id"] = registry_id
        timing_was_attested = row.get("timing_attested") is True
        row["created_tick"] = (
            min(int(row["created_tick"]), created)
            if timing_was_attested
            else created
        )
        if adopted is not None:
            row["adopted_tick"] = (
                min(int(row["adopted_tick"]), adopted)
                if timing_was_attested
                else adopted
            )
            row["timing_attested"] = True
    return tuple(records[key] for key in sorted(records))


def _transition_public_evidence_refs(
    world: Any,
    state: Mapping[str, Any],
    *,
    evidence_through_tick: int,
) -> list[str]:
    refs: set[str] = set()
    for key in (
        "public_evidence_digest",
        "exploration_reference_evidence_digest",
        "public_probe_evidence_corpus_digest",
        "latest_reference_evidence_digest",
        "latest_reference_probe_corpus_digest",
    ):
        value = str(state.get(key) or "")
        if value:
            refs.add(value)
    for row in state.get("required_public_documents") or []:
        if not isinstance(row, Mapping):
            continue
        refs.update(
            str(row.get(key) or "")
            for key in ("artifact_id", "path", "seed_content_sha256")
            if str(row.get(key) or "")
        )
    for artifact_id, artifact in (
        getattr(world, "product_artifacts", None) or {}
    ).items():
        created = getattr(artifact, "created_at_tick", None)
        path = str(getattr(artifact, "linked_file_path", "") or "").replace(
            "\\", "/"
        )
        if (
            isinstance(created, int)
            and not isinstance(created, bool)
            and created <= evidence_through_tick
            and path.startswith("eval/")
        ):
            refs.add(str(artifact_id))
            refs.add(path)
    return sorted(refs)


def _cohort_row(
    record: Mapping[str, Any],
    *,
    phase: str,
    observation_started_tick: int,
    registry_cursor: int,
    world_cursor: int,
    reason: str,
    evidence_digest: str | None,
) -> dict[str, Any]:
    return {
        "canonical_protocol_id": str(record["canonical_protocol_id"]),
        "protocol_spec_id": record.get("protocol_spec_id"),
        "registry_protocol_id": record.get("registry_protocol_id"),
        "origin_phase": str(phase),
        "created_tick": int(record.get("created_tick") or 0),
        "adopted_tick": int(record.get("adopted_tick") or 0),
        "cohort_reason": str(reason),
        "public_evidence_digest": evidence_digest,
        "observation_started_tick": int(observation_started_tick),
        "observation_until_tick": (
            int(observation_started_tick) + PROTOCOL_FRICTION_OBSERVATION_TICKS
        ),
        "registry_event_cursor": int(registry_cursor),
        "world_event_cursor": int(world_cursor),
    }


def _protocol_cohort_rows(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = [
        dict(row)
        for row in (state.get("protocol_transition_cohort") or [])
        if isinstance(row, Mapping) and row.get("canonical_protocol_id")
    ]
    if rows or not state.get("transition_protocol_ids"):
        return rows
    # Test/checkpoint compatibility for v4 states whose transition fields were
    # constructed manually. Production transitions always write full rows.
    started = int(state.get("phase_started_tick") or 0)
    until = int(
        state.get("protocol_observation_until_tick")
        or started + PROTOCOL_FRICTION_OBSERVATION_TICKS
    )
    return [
        {
            "canonical_protocol_id": str(protocol_id),
            "protocol_spec_id": str(protocol_id),
            "registry_protocol_id": None,
            "origin_phase": str(state.get("previous_phase") or ""),
            "created_tick": 0,
            "adopted_tick": started,
            "cohort_reason": "legacy_transition_snapshot",
            "public_evidence_digest": state.get(
                "protocol_transition_evidence_digest"
            ),
            "observation_started_tick": started,
            "observation_until_tick": until,
            "registry_event_cursor": int(
                state.get("protocol_registry_event_cursor") or 0
            ),
            "world_event_cursor": int(
                state.get("protocol_world_event_cursor") or 0
            ),
        }
        for protocol_id in state.get("transition_protocol_ids") or []
        if str(protocol_id)
    ]


def _extend_protocol_transition_cohort(
    world: Any, state: dict[str, Any]
) -> None:
    """Admit only genuine carryover rules, each with its own full clock."""

    if not state.get("protocol_observation_until_tick"):
        return
    rows = _protocol_cohort_rows(state)
    present = {str(row["canonical_protocol_id"]) for row in rows}
    evidence_refs = {
        str(item) for item in state.get("protocol_transition_evidence_refs") or []
    }
    evidence_through = int(
        state.get("protocol_transition_evidence_through_tick") or -1
    )
    formation_deadline = int(state.get("protocol_observation_until_tick") or 0)
    registry_cursor = len(
        list(
            getattr(
                getattr(world, "protocol_registry", None), "events", None
            )
            or []
        )
    )
    world_cursor = len(list(getattr(world, "events", None) or []))
    for record in _adopted_protocol_records(world):
        protocol_id = str(record["canonical_protocol_id"])
        if protocol_id in present:
            continue
        if record.get("timing_attested") is not True:
            # A rule appearing after transition without auditable proposal and
            # adoption ticks cannot prove either previous-phase origin or a
            # full post-adoption observation interval.
            continue
        created = int(record.get("created_tick") or 0)
        adopted = int(record.get("adopted_tick") or created)
        formed_before_transition = created <= evidence_through
        carries_public_evidence = bool(
            evidence_refs & set(record.get("lineage_refs") or [])
        )
        # The carryover cohort closes after the first full observation window.
        # An old, dormant proposal cannot join months later merely because its
        # creation timestamp predates the transition; both formation and
        # adoption must be auditable inside the bounded formation window.
        if (
            created > formation_deadline
            or adopted > formation_deadline
            or not (formed_before_transition or carries_public_evidence)
        ):
            continue
        # Adoption may happen midway through a tick. Begin at the next boundary
        # so every admitted late rule experiences sixteen *complete* action
        # ticks before it can become a repair target.
        observation_started = max(
            int(state.get("phase_started_tick") or 0), adopted + 1
        )
        reason = (
            "formed_in_previous_phase"
            if formed_before_transition
            else "transition_public_evidence_lineage"
        )
        rows.append(
            _cohort_row(
                record,
                phase=(
                    str(state.get("previous_phase") or "")
                    if formed_before_transition
                    else str(state.get("phase") or "")
                ),
                observation_started_tick=observation_started,
                registry_cursor=registry_cursor,
                world_cursor=world_cursor,
                reason=reason,
                evidence_digest=state.get(
                    "protocol_transition_evidence_digest"
                ),
            )
        )
        present.add(protocol_id)
    rows.sort(key=lambda row: str(row["canonical_protocol_id"]))
    state["protocol_transition_cohort"] = rows
    state["transition_protocol_ids"] = [
        str(row["canonical_protocol_id"]) for row in rows
    ]


def collect_programbench_protocol_friction(
    world: Any,
) -> ProgramBenchProtocolFriction:
    """Collect public friction for rules inherited by the current phase.

    Qualification is cumulative within the phase so friction may emerge
    gradually after the observation period.  Liveness is a separate rolling
    public-event check; it lets an adaptation window close once the friction
    stops recurring.  Repeated blocks of one request count once.
    """

    if not programbench_profile_active(world):
        raise ProfileAttachmentError("programbench_profile_not_active")
    state = world.__dict__[PROFILE_STATE_KEY]
    tick = int(getattr(world, "world_tick", 0) or 0)
    phase = str(state.get("phase") or "")
    phase_started = int(state.get("phase_started_tick") or 0)
    cohort = {
        str(row["canonical_protocol_id"]): row
        for row in _protocol_cohort_rows(state)
    }
    eligible = tuple(sorted(cohort))

    metrics: dict[str, dict[str, Any]] = {
        protocol_id: {
            "protocol_id": protocol_id,
            "blocked_contexts": set(),
            "recent_blocked_contexts": set(),
            "violation_contexts": {},
            "use_ticks": [],
            "exhausted_patch_artifacts": set(),
            "recent_friction_ticks": [],
            "event_refs": set(),
            "recent_event_refs": set(),
            "observation_started_tick": int(
                cohort[protocol_id].get("observation_started_tick")
                or phase_started
            ),
            "observation_until_tick": int(
                cohort[protocol_id].get("observation_until_tick")
                or phase_started + PROTOCOL_FRICTION_OBSERVATION_TICKS
            ),
        }
        for protocol_id in eligible
    }

    registry = getattr(world, "protocol_registry", None)
    registry_events = list(getattr(registry, "events", None) or [])
    registry_cursor = max(
        0,
        int(state.get("protocol_registry_event_cursor") or 0),
    )
    for event_index, event in enumerate(registry_events):
        event_tick = int(getattr(event, "tick", 0) or 0)
        protocol_id = _protocol_spec_id(world, getattr(event, "protocol_id", ""))
        row = metrics.get(protocol_id)
        if row is None:
            continue
        row_cursor = int(
            cohort[protocol_id].get("registry_event_cursor")
            or registry_cursor
        )
        if (
            event_index < row_cursor
            or event_tick < int(row["observation_started_tick"])
            or event_tick > tick
        ):
            continue
        recent_since = max(
            int(row["observation_started_tick"]),
            tick - PROTOCOL_FRICTION_RECENCY_TICKS + 1,
        )
        event_type = str(getattr(event, "event_type", "") or "")
        event_ref = "protocol_registry:" + str(
            getattr(event, "event_id", "") or _canonical_digest(
                [protocol_id, event_type, event_tick]
            )
        )
        if event_type == "violation":
            data = getattr(event, "data", None) or {}
            context_id = str(data.get("context_id") or "")
            # A repeated anonymous event has no stable governed object and
            # cannot prove distinct friction. Keep it out of the threshold
            # rather than manufacturing uniqueness from the event id.
            if not context_id:
                continue
            row["violation_contexts"][context_id] = max(
                int(row["violation_contexts"].get(context_id, -1)),
                event_tick,
            )
            row["event_refs"].add(event_ref)
            if event_tick >= recent_since:
                row["recent_friction_ticks"].append(event_tick)
                row["recent_event_refs"].add(event_ref)
        elif event_type == "use":
            row["use_ticks"].append(event_tick)
        elif event_type == "enforcement":
            data = getattr(event, "data", None) or {}
            if not data.get("blocked"):
                continue
            context_id = str(data.get("context_id") or "")
            if not context_id:
                # Anonymous blocked enforcements are retained nowhere as a
                # distinct request: otherwise three retries of the same opaque
                # gate would satisfy the three-context harm threshold.
                continue
            row["blocked_contexts"].add(context_id)
            row["event_refs"].add(event_ref)
            if event_tick >= recent_since:
                row["recent_blocked_contexts"].add(context_id)
                row["recent_friction_ticks"].append(event_tick)
                row["recent_event_refs"].add(event_ref)

    world_events = list(getattr(world, "events", None) or [])
    world_cursor = max(0, int(state.get("protocol_world_event_cursor") or 0))
    for index, event in enumerate(world_events):
        if not isinstance(event, Mapping):
            continue
        event_tick = int(event.get("tick") or 0)
        subtype = str(event.get("subtype") or "")
        if subtype != "patch_refused_by_protocol":
            continue
        protocol_id = _protocol_spec_id(world, event.get("protocol_id"))
        row = metrics.get(protocol_id)
        if row is None:
            continue
        row_cursor = int(
            cohort[protocol_id].get("world_event_cursor") or world_cursor
        )
        if (
            index < row_cursor
            or event_tick < int(row["observation_started_tick"])
            or event_tick > tick
        ):
            continue
        recent_since = max(
            int(row["observation_started_tick"]),
            tick - PROTOCOL_FRICTION_RECENCY_TICKS + 1,
        )
        artifact_id = str(event.get("artifact_id") or "")
        attempt = int(event.get("attempt") or 0)
        payload = {
            "source": "world_event",
            "subtype": subtype,
            "protocol_id": protocol_id,
            "artifact_id": artifact_id,
            "agent_id": str(event.get("agent_id") or ""),
            "tick": event_tick,
            "attempt": attempt,
            "ordinal": index,
        }
        event_ref = "world_event:" + _canonical_digest(payload)
        if artifact_id:
            # A refused patch is a blocked governed context even if a reduced
            # test double omitted the mirrored registry event.
            row["blocked_contexts"].add(artifact_id)
            if event_tick >= recent_since:
                row["recent_blocked_contexts"].add(artifact_id)
        if artifact_id and attempt >= 4:
            row["exhausted_patch_artifacts"].add(artifact_id)
        row["event_refs"].add(event_ref)
        if event_tick >= recent_since:
            row["recent_friction_ticks"].append(event_tick)
            row["recent_event_refs"].add(event_ref)

    repository = getattr(getattr(world, "repo_system", None), "repo", None)
    pull_requests = (getattr(repository, "pull_requests", None) or {}).values()
    merged_since_phase = sum(
        1
        for request in pull_requests
        if getattr(request, "merged_tick", None) is not None
        and int(getattr(request, "merged_tick")) >= phase_started
        and int(getattr(request, "merged_tick")) <= tick
    )

    target_rows: list[dict[str, Any]] = []
    friction_targets: list[str] = []
    live_targets: list[str] = []
    for protocol_id in eligible:
        raw = metrics[protocol_id]
        violations = len(raw["violation_contexts"])
        uses = len(raw["use_ticks"])
        blocked = len(raw["blocked_contexts"])
        exhausted = len(raw["exhausted_patch_artifacts"])
        merged_since_observation = sum(
            1
            for request in (
                getattr(repository, "pull_requests", None) or {}
            ).values()
            if getattr(request, "merged_tick", None) is not None
            and int(getattr(request, "merged_tick"))
            >= int(raw["observation_started_tick"])
            and int(getattr(request, "merged_tick")) <= tick
        )
        reasons: list[str] = []
        if (
            blocked >= _BLOCKED_CONTEXTS_MIN
            and blocked > merged_since_observation
        ):
            reasons.append("blocked_delivery_exceeds_landed_delivery")
        if (
            violations >= _UNMEETABLE_VIOLATIONS_MIN
            and violations / max(violations + uses, 1)
            >= _UNMEETABLE_VIOLATION_SHARE
        ):
            reasons.append("rule_unmeetable_in_current_phase")
        if exhausted >= _EXHAUSTED_PATCH_ARTIFACTS_MIN:
            reasons.append("patch_rewrites_exhausted_across_artifacts")
        recent = bool(raw["recent_friction_ticks"])
        observation_complete = tick >= int(raw["observation_until_tick"])
        if reasons:
            friction_targets.append(protocol_id)
            if recent and observation_complete:
                live_targets.append(protocol_id)
        target_rows.append(
            {
                "protocol_id": protocol_id,
                "qualified": bool(reasons) and observation_complete,
                "friction_detected": bool(reasons),
                "live": bool(reasons) and recent and observation_complete,
                "observation_started_tick": raw["observation_started_tick"],
                "observation_until_tick": raw["observation_until_tick"],
                "observation_complete": observation_complete,
                "reasons": reasons,
                "blocked_context_count": blocked,
                "recent_blocked_context_count": len(
                    raw["recent_blocked_contexts"]
                ),
                "violation_count": violations,
                "use_count": uses,
                "exhausted_patch_artifact_count": exhausted,
                "merged_since_observation": merged_since_observation,
                "latest_friction_tick": (
                    max(raw["recent_friction_ticks"])
                    if raw["recent_friction_ticks"]
                    else None
                ),
                "event_refs": _bounded_refs(raw["event_refs"]),
                "recent_event_refs": _bounded_refs(raw["recent_event_refs"]),
            }
        )

    digest_payload = {
        "schema_version": PROTOCOL_FRICTION_SCHEMA_VERSION,
        "phase": phase,
        "phase_started_tick": phase_started,
        "eligible_protocol_ids": list(eligible),
        "merged_since_phase": merged_since_phase,
        "target_metrics": target_rows,
    }
    evidence_digest = _canonical_digest(digest_payload) if eligible else None
    return ProgramBenchProtocolFriction(
        phase=phase,
        phase_started_tick=phase_started,
        observed_at_tick=tick,
        eligible_protocol_ids=eligible,
        friction_target_ids=tuple(friction_targets),
        live_friction_target_ids=tuple(live_targets),
        merged_since_phase=merged_since_phase,
        target_metrics=tuple(target_rows),
        evidence_digest=evidence_digest,
    )


_IMPLEMENTATION_ACTIONS = {
    "edit_file",
    "edit_repo_file",
    "commit_changes",
    "commit_patch",
    "push_commit",
    "open_pr",
}
_DELIVERY_ACTIONS = {
    "create_branch",
    "ci_test",
    "formal_pr_review",
    "review_pr",
    "request_changes",
    "request_pr_changes",
    "approve_pr",
    "revert_commit",
    "resolve_conflict",
    "sync_branch",
    "run_ci",
    "merge_pr",
    "dogfood_product",
    "run_launch_readiness_check",
    "create_release_candidate",
    "approve_release_candidate",
    "publish_product_release",
}
_SYNTHETIC_RELEASE_ACTIONS = {
    "run_launch_readiness_check",
    "create_release_candidate",
    "approve_release_candidate",
    "block_release_candidate",
    "publish_product_release",
    "collect_post_launch_feedback",
}
_PROTOCOL_FORMATION_ACTIONS = {
    "propose_protocol",
    "create_protocol_proposal",
    "create_workflow_proposal",
    "propose_workflow_change",
    "write_protocol",
}
_PROTOCOL_REPAIR_ACTIONS = {"amend_protocol"}
_PROTOCOL_ACTIONS = _PROTOCOL_FORMATION_ACTIONS | _PROTOCOL_REPAIR_ACTIONS


def _as_phase(
    value: ProgramBenchPhase | ProgramBenchPhaseState | str,
) -> ProgramBenchPhase:
    if isinstance(value, ProgramBenchPhaseState):
        return value.phase
    if isinstance(value, ProgramBenchPhase):
        return value
    try:
        return ProgramBenchPhase(str(value))
    except ValueError as error:
        raise ValueError("programbench_phase_invalid") from error


def candidate_decision(
    phase: ProgramBenchPhase | ProgramBenchPhaseState | str,
    action_type: str,
    *,
    parameters: Mapping[str, Any] | None = None,
    signals: ProgramBenchSignals | None = None,
) -> PhaseActionDecision:
    """Return one pure hard-guard/additive-bonus decision for a candidate.

    Unknown social actions are left untouched.  Controllers communicate
    contextual facts through explicit candidate parameters; no environment
    variable or hidden evaluator state is consulted.
    """

    current = _as_phase(phase)
    action = str(action_type or "").strip()
    params = parameters if isinstance(parameters, Mapping) else {}
    artifact_kind = str(params.get("programbench_artifact_kind") or "")
    probe_mode = str(params.get("probe_mode") or "")
    probe_inventory_ready = params.get("probe_inventory_ready") is True
    parallel_candidate = params.get("creates_parallel_candidate") is True
    integration_target = params.get("targets_integration_candidate") is True
    ci_green = params.get("ci_green") is True
    integration_pending = (
        params.get("programbench_integration_candidate_pending") is True
    )
    protocol_observation_active = (
        params.get("programbench_protocol_observation_active") is True
    )
    protocol_adaptation_active = (
        params.get("programbench_protocol_adaptation_active") is True
    )
    friction_target_ids = {
        str(item)
        for item in (
            params.get("programbench_protocol_friction_target_ids") or []
        )
    }
    repair_target = str(
        params.get("protocol_id") or params.get("target_protocol_id") or ""
    )
    protocol_repair = bool(
        action in _PROTOCOL_REPAIR_ACTIONS
        or (
            action in _PROTOCOL_ACTIONS
            and str(params.get("repair_kind") or "")
            in {"relax", "repair", "replace", "deprecate"}
        )
    )
    late_explore_formation = bool(
        current is ProgramBenchPhase.EXPLORE
        and action in _PROTOCOL_FORMATION_ACTIONS
        and params.get("programbench_exploration_protocol_formation_active")
        is True
    )
    # Native candidate generation and execution already require a recurring
    # pattern/protocol_need.  Require that provenance to travel on the
    # candidate before the profile offsets the generic OSS off-task penalty.
    formation_has_evidence = bool(
        str(params.get("source_problem") or "").strip()
        or params.get("related_objects")
        or params.get("related_object_ids")
        or params.get("related_episodes")
        or params.get("related_episode_ids")
    )

    def decision(allowed: bool, bonus: float, reason: str) -> PhaseActionDecision:
        return PhaseActionDecision(current, action, allowed, bonus, reason)

    if late_explore_formation and formation_has_evidence:
        return decision(
            True,
            0.65,
            "late_explore_evidence_grounded_protocol_formation_available",
        )
    if late_explore_formation:
        return decision(
            True,
            0.0,
            "late_explore_protocol_formation_requires_native_evidence",
        )

    # The first 16 ticks are deliberately neutral: agents may discuss, form,
    # or repair protocols through the native mechanisms, but the profile does
    # not decide for them before inherited rules have had time to meet the new
    # phase.  Once public evidence opens the window, a targeted repair outranks
    # adding another rule.  Repair is never hard-blocked, even when untargeted.
    if action in _PROTOCOL_ACTIONS and protocol_observation_active:
        return decision(True, 0.0, "protocol_transition_observation_is_neutral")
    if action in _PROTOCOL_ACTIONS and protocol_adaptation_active:
        if protocol_repair and (
            not repair_target or repair_target in friction_target_ids
        ):
            return decision(True, 3.0, "evidence_backed_protocol_repair_priority")
        if protocol_repair:
            return decision(True, 0.0, "unrelated_protocol_repair_remains_available")
        return decision(True, 0.5, "formation_available_during_repair_window")
    if action in _PROTOCOL_REPAIR_ACTIONS or (
        action in _PROTOCOL_ACTIONS and protocol_repair
    ):
        return decision(True, 0.0, "protocol_repair_remains_available")

    if parallel_candidate and current is ProgramBenchPhase.DEVELOP:
        return decision(False, 0.0, "single_integration_candidate_required")

    # ``run_eval_stub`` is a legacy sandbox affordance whose handler only emits
    # an experiment event.  It never invokes the trusted ProgramBench runner or
    # contributes public evidence, so treating it as equivalent to
    # ``run_public_tests`` creates a permanent high-bonus no-op attractor.
    if action == "run_eval_stub":
        return decision(
            False,
            0.0,
            "legacy_eval_stub_is_not_programbench_probe_runner",
        )

    if action == "run_public_tests":
        if current is ProgramBenchPhase.EXPLORE:
            if probe_mode != "reference_only":
                return decision(False, 0.0, "exploration_requires_reference_only_probe")
            if not probe_inventory_ready:
                return decision(False, 0.0, "public_probe_inventory_required")
            return decision(True, 3.0, "execute_only_reference_exploration_priority")
        if current is ProgramBenchPhase.DEVELOP:
            if (
                probe_mode == "reference_only"
                and params.get("programbench_reference_probe_current") is False
            ):
                return decision(
                    True, 3.0, "refresh_reference_before_candidate_comparison"
                )
            if probe_mode != "differential":
                return decision(False, 0.0, "current_reference_probe_already_available")
            if params.get("programbench_reference_probe_current") is False:
                return decision(
                    False,
                    0.0,
                    "current_reference_probe_required_before_candidate_comparison",
                )
            return decision(True, 3.0, "iterative_public_differential_priority")

    if action == "write_design_note":
        if (
            artifact_kind == "behavioral_contract"
            and params.get("_programbench_contract") is True
        ):
            return decision(True, 2.5, "typed_behavioral_contract_priority")
        return decision(True, 0.0, "public_document_authoring_available")

    # ProgramBench is delivered as the exact verified mainline executable and
    # workspace.  OrgEnv's generic product RC/release lifecycle is a synthetic
    # product-management mechanism, not part of the official evaluator, and it
    # cannot improve or validate this submission.  Keeping it out also avoids a
    # dead path where freeze commits COMPLETE before the multi-tick RC can run.
    if action in _SYNTHETIC_RELEASE_ACTIONS:
        return decision(
            False,
            0.0,
            "programbench_uses_verified_mainline_not_synthetic_release",
        )

    if current is ProgramBenchPhase.EXPLORE:
        if action in _IMPLEMENTATION_ACTIONS | _DELIVERY_ACTIONS:
            if (
                action == "edit_repo_file"
                and artifact_kind == "public_probe"
            ):
                return decision(True, 3.0, "public_probe_definition_revision_priority")
            return decision(
                False, 0.0, "exploration_evidence_required_before_implementation"
            )
        if action == "create_eval_stub":
            return decision(True, 3.0, "public_probe_inventory_priority")
        if action == "read_knowledge" and params.get(
            "programbench_required_public_doc"
        ) is True:
            return decision(True, 4.0, "required_public_document_read_priority")
        if action in {"internal_search", "read_repo_file"}:
            return decision(True, 2.0, "public_behavior_exploration_priority")

    if current is ProgramBenchPhase.DEVELOP:
        priority = _development_priority(signals or ProgramBenchSignals())
        if (
            action == "edit_repo_file"
            and artifact_kind == "public_probe"
            and params.get("_programbench_probe_definition_repair") is True
        ):
            return decision(True, 5.0, "public_probe_definition_repair_priority")
        if action == "edit_repo_file" and artifact_kind == "public_probe":
            return decision(
                False,
                0.0,
                "development_blocks_unproven_probe_revision",
            )
        if action in _PROTOCOL_FORMATION_ACTIONS:
            return decision(True, -0.1, "light_delivery_preference_over_new_protocol")
        if action in _IMPLEMENTATION_ACTIONS:
            if integration_target:
                return decision(
                    True, 3.0, priority + "_integration_candidate_priority"
                )
            return decision(True, -1.5, "prefer_designated_integration_candidate")
        if action == "run_ci":
            return decision(True, 3.0, priority + "_ci_priority")
        if action == "merge_pr" and integration_pending:
            return decision(
                False,
                0.0,
                "commit_integration_candidate_before_merge",
            )
        if action == "commit_patch" and integration_target and integration_pending:
            return decision(
                True, 4.0, "commit_integration_candidate_priority"
            )
        if action == "merge_pr" and ci_green:
            return decision(True, 5.0, "green_candidate_delivery_priority")
        if action in _DELIVERY_ACTIONS:
            return decision(True, 3.5, priority + "_delivery_priority")
        if action == "create_eval_stub":
            return decision(False, 0.0, "development_blocks_new_probe_definition")

    if signals is not None and derive_phase(signals).phase is not current:
        return decision(True, -0.25, "candidate_phase_is_stale_against_public_evidence")
    return decision(True, 0.0, "profile_has_no_adjustment_for_action")


def profile_policy_bonus(
    phase: ProgramBenchPhase | ProgramBenchPhaseState | str,
    action_type: str,
    *,
    parameters: Mapping[str, Any] | None = None,
    signals: ProgramBenchSignals | None = None,
) -> float:
    """Convenience wrapper for policy scorers; blocked candidates get no bonus."""

    result = candidate_decision(
        phase,
        action_type,
        parameters=parameters,
        signals=signals,
    )
    return result.bonus if result.allowed else 0.0


def _world_agents(world: Any) -> Sequence[Any]:
    agents = getattr(world, "agents", None)
    if isinstance(agents, Mapping):
        return tuple(agents[key] for key in sorted(agents, key=lambda item: str(item)))
    if isinstance(agents, Sequence) and not isinstance(agents, (str, bytes)):
        return agents
    return ()


def _manifest_public_probe_case_quota(manifest: Mapping[str, Any]) -> int:
    """Return the cumulative exploration floor, or reject the pack.

    The floor and the per-receipt technical ceiling used to be the same field,
    ``public_probes.limits.max_cases``, so raising one raised the other and the
    organization had to author the whole corpus in a single blind receipt. The
    floor now has its own sibling key.

    A manifest without ``exploration_case_quota`` falls back to
    ``limits.max_cases``. Every pack built before the split declared the floor
    there and nowhere else, so the fallback keeps those packs attaching
    unchanged and leaves their ``plan_hash`` where it is; without it, splitting
    the field would silently retire nine packs and every run recorded against
    them.
    """

    public = manifest.get("public_probes")
    if not isinstance(public, Mapping):
        raise ProfileAttachmentError(
            "programbench_public_probe_case_quota_unsupported"
        )
    quota = public.get("exploration_case_quota")
    if quota is None:
        limits = public.get("limits")
        quota = limits.get("max_cases") if isinstance(limits, Mapping) else None
    if (
        isinstance(quota, bool)
        or not isinstance(quota, int)
        or quota not in _SUPPORTED_PUBLIC_PROBE_CASE_QUOTAS
    ):
        raise ProfileAttachmentError(
            "programbench_public_probe_case_quota_unsupported"
        )
    return quota


def _manifest_public_probe_receipt_case_ceiling(
    manifest: Mapping[str, Any],
) -> int:
    """Return the per-receipt technical ceiling declared by the pack.

    This is the number the sandbox bridge and the JSON envelope can actually
    carry in one document, and it is deliberately unrelated to how many
    behaviours the organization must accumulate. Both numbers are validated
    because the authoring surface shown to a probe editor is derived from this
    one, and an editor told a different limit than the one enforced writes a
    definition that is refused after it is written.
    """

    public = manifest.get("public_probes")
    limits = public.get("limits") if isinstance(public, Mapping) else None
    ceiling = limits.get("max_cases") if isinstance(limits, Mapping) else None
    if (
        isinstance(ceiling, bool)
        or not isinstance(ceiling, int)
        or ceiling < 1
    ):
        raise ProfileAttachmentError(
            "programbench_public_probe_receipt_case_ceiling_invalid"
        )
    return ceiling


def _freeze_public_probe_contract(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Freeze the public definition surface used by pre-evaluator hashing.

    Final-primary selection must be complete before evaluator assets are even
    opened.  Consequently the active v5 profile cannot rediscover this public
    contract through ``oss_eval_assets``.  The formal manifest is public at
    attachment time, so retain only the small JSON-safe subset needed to
    enumerate and hash authored probe definitions later.
    """

    public = manifest.get("public_probes")
    surface = public.get("definition_surface") if isinstance(public, Mapping) else None
    if not isinstance(surface, Mapping) or set(surface) != {
        "exact_paths",
        "path_patterns",
        "max_definitions",
    }:
        raise ProfileAttachmentError("programbench_public_probe_surface_invalid")
    exact_paths = surface.get("exact_paths")
    path_patterns = surface.get("path_patterns")
    maximum = surface.get("max_definitions")
    if (
        not isinstance(exact_paths, list)
        or exact_paths != list(_AUDITED_PUBLIC_PROBE_EXACT_PATHS)
        or not isinstance(path_patterns, list)
        or path_patterns != list(_AUDITED_PUBLIC_PROBE_PATH_PATTERNS)
        or isinstance(maximum, bool)
        or not isinstance(maximum, int)
        or maximum != _AUDITED_PUBLIC_PROBE_MAX_DEFINITIONS
    ):
        raise ProfileAttachmentError("programbench_public_probe_surface_invalid")
    schema_version = str(public.get("schema_version") or "")
    schema_path = _safe_agent_visible_path(public.get("schema_path"))
    if not schema_version or schema_path is None:
        raise ProfileAttachmentError("programbench_public_probe_schema_invalid")
    build_paths: set[str] = set()
    probes = manifest.get("public_probes")
    compile_contract = probes.get("compile") if isinstance(probes, Mapping) else None
    if isinstance(compile_contract, Mapping):
        output = _safe_agent_visible_path(compile_contract.get("output_path"))
        if output:
            build_paths.add(output.casefold())
        for token in compile_contract.get("command") or []:
            path = _safe_agent_visible_path(token)
            if path:
                build_paths.add(path.casefold())
    if any(str(path).casefold() in build_paths for path in exact_paths):
        raise ProfileAttachmentError("programbench_public_probe_build_surface_overlap")
    contract = {
        "schema_version": _PUBLIC_PROBE_CONTRACT_STATE_SCHEMA_VERSION,
        "probe_schema_version": schema_version,
        "schema_path": schema_path,
        "definition_surface": {
            "exact_paths": list(exact_paths),
            "path_patterns": list(path_patterns),
            "max_definitions": maximum,
        },
    }
    contract["contract_sha256"] = _canonical_digest(contract)
    return contract


def _programbench_behavior_ledger_rows(
    evidence: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, int]] | None:
    """Validate that one reference receipt is internally self-consistent.

    This used to also assert that the receipt held exactly the pack's quota,
    which is what forced the organization to author the whole corpus in a
    single blind batch. A receipt is now judged only on its own coherence --
    every declared case present, well-formed, observed on the reference, and
    distinct from its siblings -- and the quota is decided over the cumulative
    coverage in :func:`_programbench_behavior_coverage_satisfied`.
    """

    counts = evidence.get("counts")
    cases = evidence.get("cases")
    if not isinstance(counts, Mapping) or not isinstance(cases, list) or not cases:
        return None
    # A case whose reference gave two different observations is not an
    # observation: no candidate can satisfy it, so counting it would put a
    # permanent merge blocker inside the evidence the contract is built on. It
    # is excluded here rather than invalidating the receipt, because the other
    # cases in the batch were observed and are real -- and under the cumulative
    # quota, discarding a whole batch over one bad stimulus is a live cost.
    if any(
        not isinstance(raw_case, Mapping)
        or raw_case.get("case_index") != expected_index
        for expected_index, raw_case in enumerate(cases)
    ):
        return None
    reproduced = [
        raw_case
        for raw_case in cases
        if raw_case.get("status") != "reference_nondeterministic"
    ]
    if (
        evidence.get("mode") != "reference_only"
        or evidence.get("status") != "completed"
        or (evidence.get("compile") or {}).get("reference") != "passed"
        or counts.get("case_count") != len(cases)
        or counts.get("reference_observed_case_count") != len(reproduced)
        or counts.get("infra_error_count") != 0
        or not reproduced
    ):
        return None

    rows: list[dict[str, Any]] = []
    full_inputs: set[str] = set()
    primary_counts: dict[str, int] = {}
    outcomes: set[str] = set()
    for raw_case in reproduced:
        public_input = raw_case.get("input")
        reference = raw_case.get("reference")
        if (
            raw_case.get("status") != "reference_observed"
            or raw_case.get("infra_side") is not None
            or not isinstance(public_input, Mapping)
            or not isinstance(reference, Mapping)
            or reference.get("status") != "completed"
        ):
            return None
        input_digest = str(public_input.get("sha256") or "")
        stdout = reference.get("stdout")
        stderr = reference.get("stderr")
        exit_value = reference.get("exit")
        effects = reference.get("filesystem_effects")
        if (
            not _SHA256.fullmatch(input_digest)
            or not isinstance(stdout, Mapping)
            or not _SHA256.fullmatch(str(stdout.get("sha256") or ""))
            or not isinstance(stderr, Mapping)
            or not _SHA256.fullmatch(str(stderr.get("sha256") or ""))
            or not isinstance(exit_value, Mapping)
            or not isinstance(effects, Mapping)
        ):
            return None
        primary_digest = _canonical_digest(
            {
                "argv": list(public_input.get("argv") or []),
                "stdin": dict(public_input.get("stdin") or {}),
                "input_files": list(public_input.get("input_files") or []),
            }
        )
        exit_digest = _canonical_digest(exit_value)
        effects_digest = _canonical_digest(effects)
        outcome_digest = _canonical_digest(
            {
                "status": reference.get("status"),
                "stdout_sha256": stdout.get("sha256"),
                "stderr_sha256": stderr.get("sha256"),
                "exit_sha256": exit_digest,
                "filesystem_effects_sha256": effects_digest,
            }
        )
        full_inputs.add(input_digest)
        primary_counts[primary_digest] = primary_counts.get(primary_digest, 0) + 1
        outcomes.add(outcome_digest)
        rows.append(
            {
                "case_index": int(raw_case["case_index"]),
                "input_sha256": input_digest,
                "primary_stimulus_sha256": primary_digest,
                "reference_status": "completed",
                "stdout_sha256": str(stdout["sha256"]),
                "stderr_sha256": str(stderr["sha256"]),
                "exit_sha256": exit_digest,
                "filesystem_effects_sha256": effects_digest,
                "outcome_sha256": outcome_digest,
            }
        )

    if len(full_inputs) != len(reproduced):
        # Two rows for the same canonical input inside one receipt make the
        # receipt's own case count a lie about how much was observed.
        return None
    return rows, {
        "case_count": len(reproduced),
        "reference_nondeterministic_case_count": len(cases) - len(reproduced),
        "unique_full_input_count": len(full_inputs),
        "unique_primary_stimulus_count": len(primary_counts),
        "unique_reference_outcome_count": len(outcomes),
        "maximum_env_only_repeat_count": max(primary_counts.values(), default=0),
    }


def _programbench_public_category_audit(
    world: Any,
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Record objective public-doc/input cues without making labels a gate."""

    state = world.__dict__.get(PROFILE_STATE_KEY)
    artifacts = getattr(world, "product_artifacts", {}) or {}
    documented_tokens: set[str] = set()
    for row in (state.get("required_public_documents") or []) if isinstance(state, Mapping) else []:
        if not isinstance(row, Mapping):
            continue
        artifact = artifacts.get(str(row.get("artifact_id") or ""))
        body = _artifact_public_text(artifact) if artifact is not None else ""
        documented_tokens.update(re.findall(r"(?<![\w-])--?[A-Za-z][\w-]*", body))
    observed_tokens: set[str] = set()
    baseline_count = 0
    exit_codes: set[int] = set()
    for row in evidence.get("cases") or []:
        if not isinstance(row, Mapping):
            continue
        public_input = row.get("input") or {}
        argv = list(public_input.get("argv") or []) if isinstance(public_input, Mapping) else []
        if not argv:
            baseline_count += 1
        observed_tokens.update(
            token for token in argv if isinstance(token, str) and token.startswith("-")
        )
        reference = row.get("reference") or {}
        exit_value = reference.get("exit") if isinstance(reference, Mapping) else None
        code = exit_value.get("code") if isinstance(exit_value, Mapping) else None
        if isinstance(code, int) and not isinstance(code, bool):
            exit_codes.add(code)
    return {
        "source": "frozen_public_documents_plus_public_reference_receipt",
        "documented_cli_tokens": sorted(documented_tokens)[:32],
        "observed_documented_cli_tokens": sorted(
            documented_tokens.intersection(observed_tokens)
        )[:32],
        "baseline_empty_argv_case_count": baseline_count,
        "observed_reference_exit_codes": sorted(exit_codes)[:16],
        "advisory_only": True,
    }


def programbench_reference_behavior_ledger(
    world: Any,
    evidence: Mapping[str, Any],
    *,
    evidence_digest: str,
    corpus_digest: str,
) -> dict[str, Any] | None:
    """Return exact, bounded public exploration evidence for the active pack."""

    state = world.__dict__.get(PROFILE_STATE_KEY)
    if not isinstance(state, Mapping):
        return None
    quota = state.get("public_probe_case_quota")
    if isinstance(quota, bool) or not isinstance(quota, int):
        return None
    if not _SHA256.fullmatch(evidence_digest) or not _SHA256.fullmatch(corpus_digest):
        return None
    try:
        from environments.org_env.product.materialize import (
            programbench_probe_corpus_digest,
        )
        from .public_evidence import public_evidence_digest

        if (
            public_evidence_digest(evidence) != evidence_digest
            or programbench_probe_corpus_digest(world) != corpus_digest
        ):
            return None
    except Exception:  # noqa: BLE001 - public binding fails closed
        return None
    if quota not in _SUPPORTED_PUBLIC_PROBE_CASE_QUOTAS:
        return None
    validated = _programbench_behavior_ledger_rows(evidence)
    if validated is None:
        return None
    rows, counts = validated
    return {
        "schema_version": _BEHAVIOR_LEDGER_SCHEMA_VERSION,
        "case_quota": quota,
        "probe_corpus_digest": corpus_digest,
        "public_evidence_digest": evidence_digest,
        "counts": counts,
        "rows": rows,
        "category_audit": _programbench_public_category_audit(world, evidence),
    }


# --------------------------------------------------------------------------- #
# cumulative public behaviour coverage
# --------------------------------------------------------------------------- #
def _programbench_coverage_counts(
    rows: Mapping[str, Any],
) -> dict[str, int]:
    """Derive every coverage count from the rows themselves.

    Nothing here is stored independently of the rows, so an auditor recomputes
    the whole summary from the content-addressed table rather than trusting a
    number somebody could have edited.
    """

    primary_counts: dict[str, int] = {}
    outcomes: set[str] = set()
    for row in rows.values():
        if not isinstance(row, Mapping):
            continue
        primary = str(row.get("primary_stimulus_sha256") or "")
        primary_counts[primary] = primary_counts.get(primary, 0) + 1
        outcomes.add(str(row.get("outcome_sha256") or ""))
    return {
        "distinct_input_count": len(rows),
        "distinct_primary_stimulus_count": len(primary_counts),
        "distinct_reference_outcome_count": len(outcomes),
        "maximum_env_only_repeat_count": max(primary_counts.values(), default=0),
    }


def programbench_empty_behavior_coverage(quota: int) -> dict[str, Any]:
    """Return the zero state of the cumulative coverage document."""

    coverage: dict[str, Any] = {
        "schema_version": _BEHAVIOR_COVERAGE_SCHEMA_VERSION,
        "exploration_case_quota": int(quota),
        "rows": {},
        "receipts": [],
        "counts": _programbench_coverage_counts({}),
    }
    coverage["coverage_sha256"] = _programbench_coverage_digest(coverage)
    return coverage


def _programbench_coverage_digest(coverage: Mapping[str, Any]) -> str:
    """Hash the parts of the coverage that an auditor would replay.

    The derived counts are excluded on purpose: they are a function of the rows
    and including them would let a wrong count change the identity of a correct
    accumulation.
    """

    return _canonical_digest(
        {
            "schema_version": coverage.get("schema_version"),
            "exploration_case_quota": coverage.get("exploration_case_quota"),
            "rows": coverage.get("rows"),
            "receipts": coverage.get("receipts"),
        }
    )


def programbench_merge_behavior_coverage(
    coverage: Mapping[str, Any] | None,
    ledger: Mapping[str, Any],
    *,
    quota: int,
    tick: int,
) -> dict[str, Any]:
    """Accumulate one validated reference receipt into the coverage document.

    Rows are keyed by ``input_sha256``, so re-observing an input the
    organization already probed costs nothing and merging the same receipt
    twice is a no-op -- which matters because the exploration gate re-attests
    the bound receipt at every tick boundary and a restored cache entry replays
    it again.

    Re-observing an input with a *different* reference outcome raises
    :class:`ProgramBenchCoverageConflict`. Silently keeping either row would
    turn a real fact about the reference -- that this stimulus is not
    reproducible -- into an arbitrary choice made by merge order.
    """

    if not isinstance(ledger, Mapping):
        raise ProgramBenchCoverageConflict(
            "programbench_public_behavior_coverage_ledger_invalid"
        )
    rows = ledger.get("rows")
    evidence_digest = str(ledger.get("public_evidence_digest") or "")
    corpus_digest = str(ledger.get("probe_corpus_digest") or "")
    if (
        ledger.get("schema_version") != _BEHAVIOR_LEDGER_SCHEMA_VERSION
        or not isinstance(rows, list)
        or not rows
        or not _SHA256.fullmatch(evidence_digest)
        or not _SHA256.fullmatch(corpus_digest)
    ):
        raise ProgramBenchCoverageConflict(
            "programbench_public_behavior_coverage_ledger_invalid"
        )
    merged = (
        copy.deepcopy(dict(coverage))
        if _programbench_behavior_coverage_valid(coverage, quota=quota)
        else programbench_empty_behavior_coverage(quota)
    )
    receipts: list[dict[str, Any]] = list(merged["receipts"])
    if any(
        str(item.get("evidence_digest") or "") == evidence_digest
        and str(item.get("corpus_digest") or "") == corpus_digest
        for item in receipts
        if isinstance(item, Mapping)
    ):
        return merged
    table: dict[str, Any] = dict(merged["rows"])
    new_inputs = 0
    for row in rows:
        if not isinstance(row, Mapping):
            raise ProgramBenchCoverageConflict(
                "programbench_public_behavior_coverage_row_invalid"
            )
        input_digest = str(row.get("input_sha256") or "")
        entry = {
            "primary_stimulus_sha256": str(row.get("primary_stimulus_sha256") or ""),
            "outcome_sha256": str(row.get("outcome_sha256") or ""),
            "stdout_sha256": str(row.get("stdout_sha256") or ""),
            "source_receipt_digest": evidence_digest,
            "source_corpus_digest": corpus_digest,
            "observed_at_tick": int(tick),
        }
        if not _SHA256.fullmatch(input_digest) or any(
            not _SHA256.fullmatch(entry[field])
            for field in (
                "primary_stimulus_sha256",
                "outcome_sha256",
                "stdout_sha256",
            )
        ):
            raise ProgramBenchCoverageConflict(
                "programbench_public_behavior_coverage_row_invalid"
            )
        existing = table.get(input_digest)
        if isinstance(existing, Mapping):
            divergent = sorted(
                field
                for field in (
                    "primary_stimulus_sha256",
                    "outcome_sha256",
                    "stdout_sha256",
                )
                if str(existing.get(field) or "") != entry[field]
            )
            if divergent:
                raise ProgramBenchCoverageConflict(
                    "programbench_public_behavior_coverage_conflict:"
                    + input_digest
                    + ":"
                    + ",".join(divergent)
                )
            continue
        table[input_digest] = entry
        new_inputs += 1
    receipts.append(
        {
            "evidence_digest": evidence_digest,
            "corpus_digest": corpus_digest,
            "tick": int(tick),
            "case_count": len(rows),
            "new_input_count": new_inputs,
        }
    )
    merged["rows"] = table
    merged["receipts"] = receipts
    merged["exploration_case_quota"] = int(quota)
    merged["counts"] = _programbench_coverage_counts(table)
    merged["coverage_sha256"] = _programbench_coverage_digest(merged)
    return merged


def programbench_replay_behavior_coverage(
    ledgers: Sequence[tuple[Mapping[str, Any], int]],
    *,
    quota: int,
) -> dict[str, Any]:
    """Fold receipts through the same merge an auditor would.

    The develop-transition attestation binds ``coverage_sha256`` and the
    ordered ``receipts`` list rather than one receipt's evidence digest, so the
    claim "these receipts, merged in this order, produce this ledger" is
    checkable by anyone holding the receipts. This is the function that makes
    that check the same code path the run itself took.
    """

    coverage = programbench_empty_behavior_coverage(quota)
    for ledger, tick in ledgers:
        coverage = programbench_merge_behavior_coverage(
            coverage, ledger, quota=quota, tick=tick
        )
    return coverage


def _programbench_behavior_coverage_valid(
    coverage: Any,
    *,
    quota: int,
) -> bool:
    """Re-derive the coverage summary and digest from its own rows."""

    if not isinstance(coverage, Mapping):
        return False
    rows = coverage.get("rows")
    receipts = coverage.get("receipts")
    if (
        coverage.get("schema_version") != _BEHAVIOR_COVERAGE_SCHEMA_VERSION
        or coverage.get("exploration_case_quota") != quota
        or not isinstance(rows, Mapping)
        or not isinstance(receipts, list)
        or coverage.get("counts") != _programbench_coverage_counts(rows)
        or _programbench_coverage_digest(coverage)
        != coverage.get("coverage_sha256")
    ):
        return False
    receipt_digests = set()
    for item in receipts:
        if not isinstance(item, Mapping) or set(item) != {
            "evidence_digest",
            "corpus_digest",
            "tick",
            "case_count",
            "new_input_count",
        }:
            return False
        if not _SHA256.fullmatch(str(item.get("evidence_digest") or "")) or not (
            _SHA256.fullmatch(str(item.get("corpus_digest") or ""))
        ):
            return False
        receipt_digests.add(str(item["evidence_digest"]))
    for input_digest, row in rows.items():
        if (
            not _SHA256.fullmatch(str(input_digest))
            or not isinstance(row, Mapping)
            or set(row) != {
                "primary_stimulus_sha256",
                "outcome_sha256",
                "stdout_sha256",
                "source_receipt_digest",
                "source_corpus_digest",
                "observed_at_tick",
            }
            or any(
                not _SHA256.fullmatch(str(row.get(field) or ""))
                for field in (
                    "primary_stimulus_sha256",
                    "outcome_sha256",
                    "stdout_sha256",
                    "source_receipt_digest",
                    "source_corpus_digest",
                )
            )
            or str(row["source_receipt_digest"]) not in receipt_digests
        ):
            return False
    return True


def _programbench_behavior_coverage_satisfied(
    coverage: Any,
    *,
    quota: int,
) -> bool:
    """Decide whether accumulated observation is enough to leave EXPLORE.

    The floor is cumulative and the diversity minima are absolute, so a
    hundred restatements of one stimulus never substitute for having looked at
    twelve different ones.
    """

    if not _programbench_behavior_coverage_valid(coverage, quota=quota):
        return False
    counts = coverage["counts"]
    return bool(
        counts["distinct_input_count"] >= quota
        and counts["distinct_primary_stimulus_count"]
        >= _MINIMUM_CUMULATIVE_PRIMARY_STIMULI
        and counts["maximum_env_only_repeat_count"]
        <= _MAXIMUM_CUMULATIVE_ENV_ONLY_REPEATS
        and counts["distinct_reference_outcome_count"]
        >= _MINIMUM_CUMULATIVE_REFERENCE_OUTCOMES
    )


def programbench_behavior_coverage_summary(world: Any) -> dict[str, Any] | None:
    """Return the agent-facing view of what has been accumulated so far."""

    state = get_programbench_profile_state(world)
    if state is None:
        return None
    quota = state.get("public_probe_case_quota")
    if isinstance(quota, bool) or not isinstance(quota, int):
        return None
    coverage = state.get("public_behavior_coverage")
    if not _programbench_behavior_coverage_valid(coverage, quota=quota):
        coverage = programbench_empty_behavior_coverage(quota)
    counts = coverage["counts"]
    return {
        "exploration_case_quota": quota,
        "distinct_input_count": counts["distinct_input_count"],
        "remaining_input_count": max(0, quota - counts["distinct_input_count"]),
        "distinct_primary_stimulus_count": counts[
            "distinct_primary_stimulus_count"
        ],
        "distinct_reference_outcome_count": counts[
            "distinct_reference_outcome_count"
        ],
        "maximum_env_only_repeat_count": counts["maximum_env_only_repeat_count"],
        "receipt_count": len(coverage["receipts"]),
        "coverage_sha256": str(coverage["coverage_sha256"]),
        "quota_satisfied": _programbench_behavior_coverage_satisfied(
            coverage, quota=quota
        ),
    }


def attach_programbench_profile(
    world: Any,
    profile_id: str | None,
    *,
    manifest: Mapping[str, Any] | None = None,
    root_issue: Mapping[str, Any] | None = None,
    agents: Iterable[Any] | None = None,
) -> ProfileAttachment | None:
    """Explicitly attach the adapted profile, or leave native mode untouched.

    ``None``, the empty string, and ``"native"`` are deliberate no-ops.  Any
    other unknown id or any task-family mismatch raises instead of silently
    applying a benchmark-specific policy to the wrong task.
    """

    requested = str(profile_id or "").strip()
    if requested in {"", "native"}:
        return None
    if requested != PROFILE_ID:
        raise ProfileAttachmentError("programbench_profile_id_unsupported")
    if manifest is None:
        candidate = getattr(world, "programbench_public_manifest", None)
        manifest = candidate if isinstance(candidate, Mapping) else None
    detection = detect_programbench_task_family(manifest, root_issue)
    if not detection.matched:
        raise ProfileAttachmentError(
            "programbench_profile_task_mismatch:" + detection.reason
        )
    if not isinstance(manifest, Mapping):
        raise ProfileAttachmentError("programbench_public_manifest_missing")
    public_probe_case_quota = _manifest_public_probe_case_quota(manifest)
    public_probe_receipt_case_ceiling = (
        _manifest_public_probe_receipt_case_ceiling(manifest)
    )
    frozen_public_probe_contract = _freeze_public_probe_contract(manifest)
    roster = agents if agents is not None else _world_agents(world)
    assignments = assign_fixed_roles(roster)
    initial_signals = ProgramBenchSignals(integration_owner_assigned=bool(assignments))
    phase_state = derive_phase(initial_signals)
    overlay = programbench_workflow_overlay()
    company_overlay = programbench_company_brief_overlay()
    brief = programbench_root_task_brief(public_probe_case_quota)
    attachment_tick = int(getattr(world, "world_tick", 0) or 0)
    if attachment_tick != 0:
        raise ProfileAttachmentError("programbench_profile_requires_fresh_t0")
    required_public_documents = _freeze_required_public_documents(world, manifest)
    if not isinstance(getattr(world, "events", None), list):
        world.__dict__["events"] = []
    initial_coverage = {
        "required_document_count": len(required_public_documents),
        "role_complete": {
            role: False for role in _REQUIRED_PUBLIC_READER_ROLES
        },
        "complete": False,
    }
    attachment = ProfileAttachment(
        profile_id=PROFILE_ID,
        task_family=TASK_FAMILY,
        initial_brief=brief,
        company_brief_overlay=company_overlay,
        product_context_labels=PRODUCT_CONTEXT_LABELS,
        workflow_overlay=overlay,
        phase_state=phase_state,
        role_assignments=assignments,
    )
    world.__dict__[PROFILE_STATE_KEY] = {
        "schema_version": PROFILE_STATE_SCHEMA_VERSION,
        "profile_id": PROFILE_ID,
        "task_family": TASK_FAMILY,
        "active": True,
        **phase_state.to_dict(),
        "role_assignments": [item.to_dict() for item in assignments],
        "workflow_overlay": overlay,
        "company_brief_overlay": company_overlay,
        "product_context_labels": list(PRODUCT_CONTEXT_LABELS),
        "initial_brief": brief,
        "public_evidence_digest": None,
        "exploration_reference_evidence_digest": None,
        "public_probe_evidence_corpus_digest": None,
        "latest_reference_evidence_digest": None,
        "latest_reference_probe_corpus_digest": None,
        "latest_reference_probe_required": False,
        "reference_probe_cache": [],
        "public_probe_case_quota": public_probe_case_quota,
        "public_probe_receipt_case_ceiling": public_probe_receipt_case_ceiling,
        "frozen_public_probe_contract": frozen_public_probe_contract,
        "public_behavior_ledger": None,
        # Content-addressed and cumulative. A probe edit invalidates the
        # current receipt but never these rows: they are observations that
        # actually happened, and what the definition file says today cannot
        # unmake them.
        "public_behavior_coverage": programbench_empty_behavior_coverage(
            public_probe_case_quota
        ),
        "accepted_behavioral_contract": None,
        "develop_transition_attestation": None,
        # Differential verification is a claim about both a concrete public
        # probe corpus and a concrete candidate tree.  Reference freshness is
        # deliberately tracked separately: refreshing the reference side must
        # never make an older candidate comparison current again.
        "public_verification_probe_corpus_digest": None,
        "public_verification_candidate_repo_digest": None,
        "public_verification_tested_candidate_repo_digest": None,
        "public_verification_candidate_view_schema": None,
        "candidate_surface_schema_version": _CANDIDATE_SURFACE_SCHEMA_VERSION,
        "candidate_view_migration_schema": (
            _CANDIDATE_VIEW_MIGRATION_SCHEMA_VERSION
        ),
        "candidate_view_migration_history": [],
        "strict_verified_mainline_digest": None,
        "delivery_milestones": [],
        "run_horizon_ticks": 336,
        "attachment_tick": attachment_tick,
        "signals": initial_signals.to_dict(),
        "decision_signals": initial_signals.to_dict(),
        "signals_staged_tick": None,
        "phase_boundary_tick": attachment_tick,
        "phase_evidence_through_tick": attachment_tick - 1,
        "pending_phase": phase_state.phase.value,
        "pending_phase_reason": phase_state.reason,
        "required_public_documents": required_public_documents,
        "required_public_reader_roles": list(_REQUIRED_PUBLIC_READER_ROLES),
        "public_document_read_receipts": [],
        "public_document_coverage": initial_coverage,
        # t1..t24 are twenty-four complete EXPLORE action ticks. Evidence
        # staged during them can first change the state at the t25 boundary.
        "exploration_full_ticks_required": EXPLORE_OBSERVATION_FULL_TICKS,
        "exploration_transition_eligible_tick": (
            attachment_tick + EXPLORE_OBSERVATION_FULL_TICKS + 1
        ),
        "previous_phase": None,
        "phase_started_tick": attachment_tick,
        "phase_transition_count": 0,
        # Initial EXPLORE is not a transition from an older working
        # environment.  Protocol formation therefore stays entirely native;
        # the observation clock starts only on the first real phase change.
        "transition_protocol_ids": [],
        "protocol_transition_cohort_schema_version": (
            PROTOCOL_TRANSITION_COHORT_SCHEMA_VERSION
        ),
        "protocol_transition_cohort": [],
        "protocol_transition_evidence_refs": [],
        "protocol_transition_evidence_digest": None,
        "protocol_transition_evidence_through_tick": None,
        "protocol_observing_ids": [],
        "protocol_observed_friction_target_ids": [],
        "protocol_repair_eligible_target_ids": [],
        "protocol_registry_event_cursor": len(
            list(
                getattr(
                    getattr(world, "protocol_registry", None),
                    "events",
                    None,
                )
                or []
            )
        ),
        "protocol_world_event_cursor": len(
            list(getattr(world, "events", None) or [])
        ),
        "protocol_transition_mode": "initial",
        "protocol_observation_until_tick": None,
        "protocol_adaptation_started_tick": None,
        "protocol_adaptation_min_until_tick": None,
        "protocol_adaptation_until_tick": None,
        "protocol_adaptation_active": False,
        "protocol_adaptation_trigger_evidence_digest": None,
        "protocol_adaptation_closed_tick": None,
        "protocol_adaptation_history": [],
        "protocol_friction_target_ids": [],
        "protocol_live_friction_target_ids": [],
        "protocol_friction_evidence": None,
    }
    # The starter mainline is the only non-pending public tree at t0. Freeze
    # its strict full-view digest as an explicit trusted baseline so t336 has a
    # valid fallback even if no integration candidate ever reaches public
    # attestation. Any later verified merge advances this digest.
    try:
        from environments.org_env.product.materialize import (
            programbench_mainline_digest,
        )

        starter_digest = programbench_mainline_digest(world)
    except Exception as error:  # noqa: BLE001 - attachment is atomic/fail closed
        world.__dict__.pop(PROFILE_STATE_KEY, None)
        raise ProfileAttachmentError(
            "programbench_trusted_starter_mainline_unavailable"
        ) from error
    state = world.__dict__[PROFILE_STATE_KEY]
    state["strict_verified_mainline_digest"] = starter_digest
    state["strict_mainline_attestation_kind"] = "trusted_starter_baseline"
    return attachment


def programbench_profile_active(world: Any) -> bool:
    state = world.__dict__.get(PROFILE_STATE_KEY)
    return bool(
        isinstance(state, Mapping)
        and state.get("schema_version") == PROFILE_STATE_SCHEMA_VERSION
        and state.get("profile_id") == PROFILE_ID
        and state.get("active") is True
    )


def get_programbench_profile_state(world: Any) -> dict[str, Any] | None:
    """Return a shallow JSON-safe state copy only when attachment is valid."""

    if not programbench_profile_active(world):
        return None
    return dict(world.__dict__[PROFILE_STATE_KEY])


def programbench_frozen_public_probe_contract(world: Any) -> dict[str, Any]:
    """Return the attachment-time public probe surface after strict re-hash."""

    if not programbench_profile_active(world):
        raise ProfileAttachmentError("programbench_profile_state_invalid")
    state = world.__dict__.get(PROFILE_STATE_KEY)
    contract = state.get("frozen_public_probe_contract") if isinstance(state, Mapping) else None
    if not isinstance(contract, Mapping) or set(contract) != {
        "schema_version",
        "probe_schema_version",
        "schema_path",
        "definition_surface",
        "contract_sha256",
    }:
        raise ProfileAttachmentError("programbench_public_probe_contract_state_invalid")
    digest = str(contract.get("contract_sha256") or "")
    unsigned = {key: copy.deepcopy(value) for key, value in contract.items() if key != "contract_sha256"}
    surface = contract.get("definition_surface")
    if (
        contract.get("schema_version")
        != _PUBLIC_PROBE_CONTRACT_STATE_SCHEMA_VERSION
        or not _SHA256.fullmatch(digest)
        or _canonical_digest(unsigned) != digest
        or not isinstance(surface, Mapping)
        or set(surface) != {"exact_paths", "path_patterns", "max_definitions"}
        or surface.get("max_definitions")
        != _AUDITED_PUBLIC_PROBE_MAX_DEFINITIONS
        or not isinstance(surface.get("exact_paths"), list)
        or not isinstance(surface.get("path_patterns"), list)
        or surface.get("exact_paths")
        != list(_AUDITED_PUBLIC_PROBE_EXACT_PATHS)
        or surface.get("path_patterns")
        != list(_AUDITED_PUBLIC_PROBE_PATH_PATTERNS)
        or contract.get("schema_path") != "tests/public/schema.json"
        or contract.get("probe_schema_version")
        != "programbench_public_probe_cases_v1"
    ):
        raise ProfileAttachmentError("programbench_public_probe_contract_state_invalid")
    return copy.deepcopy(dict(contract))


def programbench_frozen_public_probe_path_status(
    world: Any, raw_path: Any
) -> str:
    """Classify one path against the frozen probe surface without aliases.

    Repository artifact paths are serialized identities, not host filesystem
    conveniences.  The adapted profile therefore accepts only the exact
    canonical POSIX spelling admitted by the frozen manifest.  A case-folded,
    ``./`` or backslash spelling which would collide with that surface is
    explicitly invalid instead of becoming candidate code in one subsystem
    and a probe in another.
    """

    frozen = programbench_frozen_public_probe_contract(world)
    surface = frozen["definition_surface"]
    exact = tuple(str(path) for path in surface["exact_paths"])
    patterns = tuple(re.compile(str(value)) for value in surface["path_patterns"])
    raw = raw_path if isinstance(raw_path, str) else ""
    try:
        canonical = normalize_repo_relative_path(raw)
    except InvalidRepoPath:
        # Any non-canonical spelling which becomes a probe spelling after the
        # historical slash/dot cleanup is a dangerous surface alias.
        legacy = str(raw or "").strip().replace("\\", "/")
        while legacy.startswith("./"):
            legacy = legacy[2:]
        legacy = legacy.strip("/")
        folded = legacy.casefold()
        if any(folded == path.casefold() for path in exact) or any(
            pattern.fullmatch(folded) for pattern in patterns
        ):
            return "invalid_alias"
        return "other"
    exact_match = canonical in exact or any(
        pattern.fullmatch(canonical) for pattern in patterns
    )
    if raw == canonical and exact_match:
        return "trusted"
    folded = canonical.casefold()
    alias_match = any(folded == path.casefold() for path in exact) or any(
        pattern.fullmatch(folded) for pattern in patterns
    )
    if raw != canonical or alias_match:
        return "invalid_alias" if alias_match else "other"
    return "other"


def _programbench_bound_reference_evidence(
    world: Any,
) -> tuple[dict[str, Any], str, str] | None:
    """Return the qualified public reference receipt bound to current work.

    Before contract acceptance this is the immutable exploration receipt.  Once
    implementation starts, a newer current-corpus reference receipt may replace
    it for repair guidance.  Every cache row is re-hashed before prompt use so a
    malformed or partially restored checkpoint cannot become agent evidence.
    """

    state = get_programbench_profile_state(world)
    if state is None:
        return None
    signals = state.get("signals")
    if not isinstance(signals, Mapping):
        return None
    accepted = signals.get("behavioral_contract_accepted") is True
    digest = ""
    corpus = ""
    if accepted:
        digest = str(state.get("latest_reference_evidence_digest") or "")
        corpus = str(state.get("latest_reference_probe_corpus_digest") or "")
    if not digest or not corpus:
        digest = str(state.get("exploration_reference_evidence_digest") or "")
        corpus = str(state.get("public_probe_evidence_corpus_digest") or "")
    if not _SHA256.fullmatch(digest) or not _SHA256.fullmatch(corpus):
        return None
    try:
        from environments.org_env.product.materialize import (
            programbench_probe_corpus_digest,
        )

        if programbench_probe_corpus_digest(world) != corpus:
            return None
    except Exception:  # noqa: BLE001 - stale/unhashable public corpus fails closed
        return None
    cache = state.get("reference_probe_cache")
    if not isinstance(cache, list):
        return None
    from .public_evidence import (
        canonical_public_evidence_json,
        public_evidence_digest,
    )

    for row in reversed(cache):
        if not isinstance(row, Mapping):
            continue
        if (
            row.get("available") is not True
            or row.get("qualified") is not True
            or str(row.get("corpus_digest") or "") != corpus
            or str(row.get("evidence_digest") or "") != digest
        ):
            continue
        evidence = row.get("evidence")
        if not isinstance(evidence, Mapping):
            return None
        try:
            canonical = json.loads(canonical_public_evidence_json(evidence))
            if public_evidence_digest(canonical) != digest:
                return None
        except Exception:  # noqa: BLE001 - malformed prompt evidence fails closed
            return None
        ledger = programbench_reference_behavior_ledger(
            world,
            canonical,
            evidence_digest=digest,
            corpus_digest=corpus,
        )
        if ledger is None:
            return None
        return canonical, digest, corpus
    return None


def programbench_reference_observation_summary(
    world: Any,
) -> dict[str, Any] | None:
    """Return compact public reference metadata suitable for shared cognition."""

    bound = _programbench_bound_reference_evidence(world)
    if bound is None:
        return None
    evidence, digest, corpus = bound
    return {
        "evidence_digest": digest,
        "probe_corpus_digest": corpus,
        "case_count": int(evidence["counts"]["case_count"]),
        "probe_ids": [
            str(row.get("probe_id") or "")
            for row in evidence.get("cases", [])[:32]
            if isinstance(row, Mapping)
        ],
    }


def programbench_reference_observation_brief(
    world: Any,
    *,
    max_cases: int | None = None,
) -> str:
    """Return bounded case-level public reference evidence for an editor.

    The default follows the pack's own quota rather than a fixed 16. Both
    callers -- the editor prompt and the repo-work path -- take the default, so
    a hard 16 under a 32- or 64-case pack shows an editor half or a quarter of
    the observations the organization was required to author, and the behaviour
    it cannot see is the behaviour it cannot reproduce.
    """

    bound = _programbench_bound_reference_evidence(world)
    if bound is None:
        return ""
    evidence, _digest, _corpus = bound
    if max_cases is None:
        state = get_programbench_profile_state(world) or {}
        quota = state.get("public_probe_case_quota")
        max_cases = quota if isinstance(quota, int) and quota > 0 else 16
    from .public_evidence import reference_observation_brief

    try:
        return reference_observation_brief(evidence, max_cases=max_cases)
    except Exception:  # noqa: BLE001 - no prompt is safer than malformed evidence
        return ""


def programbench_public_repair_brief(world: Any) -> str:
    """Return stored or safely derived public repair feedback for current work."""

    state = get_programbench_profile_state(world)
    if state is None:
        return ""
    from .public_evidence import (
        public_failure_repair_brief,
        redact_public_text,
    )

    stored = redact_public_text(
        state.get("public_repair_brief", ""), max_chars=16_384
    ).strip()
    if stored:
        return stored
    evidence = state.get("public_evidence")
    if not isinstance(evidence, Mapping) or not evidence.get("failure"):
        return ""
    product = getattr(world, "product", None)
    substrate_meta = getattr(product, "substrate_meta", {}) or {}
    try:
        return public_failure_repair_brief(
            evidence,
            declared_output_path=str(
                substrate_meta.get("reconstruction_output_path") or ""
            ),
        )
    except Exception:  # noqa: BLE001 - malformed checkpoint evidence fails closed
        return ""


def programbench_current_public_candidate_attestation(
    world: Any,
) -> tuple[str | None, dict[str, Any] | None]:
    """Re-hash the complete public verification claim for the live candidate.

    The returned payload contains only public corpus/candidate/evidence hashes.
    Merge, freeze and final-primary selection share this predicate so a stale
    signal, reference-only receipt, or mismatched case set cannot be accepted by
    one boundary after another boundary rejected it.
    """

    if not programbench_profile_active(world):
        return "programbench_profile_state_invalid", None
    state = get_programbench_profile_state(world)
    if state is None:
        return "programbench_profile_state_invalid", None
    if str(state.get("phase") or "") != ProgramBenchPhase.DEVELOP.value:
        return "programbench_develop_phase_required", None
    try:
        from environments.org_env.runtime_adapter.execution import (
            programbench_candidate_implementation_coherence,
        )

        coherence_reason, coherence = (
            programbench_candidate_implementation_coherence(world)
        )
    except Exception:  # noqa: BLE001 - implementation provenance fails closed
        return "programbench_candidate_implementation_coherence_unavailable", None
    if coherence_reason is not None or not isinstance(coherence, Mapping):
        return coherence_reason or "programbench_candidate_implementation_incoherent", None
    try:
        signals = get_programbench_signals(world)
    except (ProfileAttachmentError, TypeError, ValueError):
        return "programbench_profile_signals_invalid", None
    ledger = state.get("public_behavior_ledger")
    if (
        not signals.behavioral_contract_accepted
        or not isinstance(ledger, Mapping)
        or not _programbench_current_behavioral_contract_valid(
            world, state, ledger
        )
    ):
        return "programbench_current_behavioral_contract_required", None
    if not signals.public_verification_complete:
        return "programbench_public_verification_required", None
    if signals.unresolved_public_mismatch_count != 0:
        return "programbench_public_mismatch_unresolved", None
    if state.get("latest_reference_probe_required") is not False:
        return "programbench_reference_probe_stale", None

    try:
        from environments.org_env.product.materialize import (
            PROGRAMBENCH_CANDIDATE_VIEW_SCHEMA_VERSION,
            programbench_integration_candidate_digest,
            programbench_probe_corpus_digest,
        )

        current_corpus = programbench_probe_corpus_digest(world)
        current_working = programbench_integration_candidate_digest(world)
    except Exception:  # noqa: BLE001 - live public hashing fails closed
        return "programbench_live_attestation_unavailable", None
    if PROGRAMBENCH_CANDIDATE_VIEW_SCHEMA_VERSION != (
        _CANDIDATE_SURFACE_SCHEMA_VERSION
    ):
        return "programbench_candidate_view_schema_unsupported", None
    if str(state.get("public_verification_candidate_view_schema") or "") != (
        PROGRAMBENCH_CANDIDATE_VIEW_SCHEMA_VERSION
    ):
        return "programbench_candidate_view_reverification_required", None
    verification_corpus = str(
        state.get("public_verification_probe_corpus_digest") or ""
    )
    if not verification_corpus or verification_corpus != current_corpus:
        return "programbench_verification_corpus_stale", None
    verification_candidate = str(
        state.get("public_verification_candidate_repo_digest") or ""
    )
    if not verification_candidate or verification_candidate != current_working:
        return "programbench_verified_candidate_stale", None
    if str(state.get("public_candidate_repo_digest") or "") != current_working:
        return "programbench_public_candidate_digest_stale", None
    if str(
        state.get("public_verification_tested_candidate_repo_digest") or ""
    ) != current_working:
        return "programbench_tested_candidate_digest_stale", None

    bound = _programbench_bound_reference_evidence(world)
    if bound is None:
        return "programbench_current_reference_receipt_invalid", None
    reference_evidence, reference_digest, reference_corpus = bound
    if reference_corpus != current_corpus:
        return "programbench_reference_probe_stale", None
    raw_differential = state.get("public_evidence")
    if not isinstance(raw_differential, Mapping):
        return "programbench_differential_receipt_missing", None
    try:
        from .public_evidence import (
            canonical_public_evidence_json,
            public_evidence_digest,
        )

        differential = json.loads(
            canonical_public_evidence_json(raw_differential)
        )
        differential_digest = public_evidence_digest(differential)
    except Exception:  # noqa: BLE001 - malformed public receipt fails closed
        return "programbench_differential_receipt_invalid", None
    if differential_digest != str(state.get("public_evidence_digest") or ""):
        return "programbench_differential_receipt_digest_mismatch", None

    quota = state.get("public_probe_case_quota")
    counts = differential.get("counts")
    cases = differential.get("cases")
    reference_cases = reference_evidence.get("cases")
    # The differential is a claim about the corpus that exists right now, not
    # about the exploration floor. Tying it to the quota would mean the current
    # definition file always had to hold exactly as many cases as the
    # organization once had to accumulate, which reimposes the ceiling the
    # cumulative quota exists to remove.
    case_count = len(cases) if isinstance(cases, list) else -1
    if (
        isinstance(quota, bool)
        or not isinstance(quota, int)
        or quota not in _SUPPORTED_PUBLIC_PROBE_CASE_QUOTAS
        or differential.get("mode") != "differential"
        or differential.get("status") != "completed"
        or (differential.get("compile") or {}).get("reference") != "passed"
        or (differential.get("compile") or {}).get("candidate") != "passed"
        or not isinstance(counts, Mapping)
        or case_count < 1
        or counts.get("case_count") != case_count
        or counts.get("compared_case_count") != case_count
        or counts.get("matched_case_count") != case_count
        or counts.get("mismatched_case_count") != 0
        or counts.get("infra_error_count") != 0
        or not isinstance(reference_cases, list)
        or len(reference_cases) != case_count
    ):
        return "programbench_differential_receipt_incomplete", None
    reference_inputs = [
        str((row.get("input") or {}).get("sha256") or "")
        for row in reference_cases
        if isinstance(row, Mapping)
    ]
    differential_inputs: list[str] = []
    for row in cases:
        if not isinstance(row, Mapping):
            return "programbench_differential_case_invalid", None
        input_digest = str((row.get("input") or {}).get("sha256") or "")
        reference_side = row.get("reference")
        candidate_side = row.get("candidate")
        if (
            not _SHA256.fullmatch(input_digest)
            or row.get("status") != "compared"
            or row.get("infra_side") is not None
            or row.get("matched") is not True
            or not isinstance(reference_side, Mapping)
            or reference_side.get("status") != "completed"
            or not isinstance(candidate_side, Mapping)
            or candidate_side.get("status") != "completed"
        ):
            return "programbench_differential_case_invalid", None
        differential_inputs.append(input_digest)
    if (
        differential_inputs != reference_inputs
        or len(set(differential_inputs)) != case_count
    ):
        return "programbench_differential_case_set_stale", None
    payload = {
        "schema_version": "programbench_current_public_candidate_attestation_v1",
        "candidate_surface_schema_version": (
            PROGRAMBENCH_CANDIDATE_VIEW_SCHEMA_VERSION
        ),
        "candidate_repo_digest": current_working,
        "tested_candidate_repo_digest": current_working,
        "probe_corpus_digest": current_corpus,
        "reference_evidence_digest": reference_digest,
        "differential_evidence_digest": differential_digest,
        "case_count": case_count,
        "exploration_case_quota": quota,
        "mismatched_case_count": 0,
        "infra_error_count": 0,
        "candidate_implementation_coherence": dict(coherence),
    }
    return None, payload


def _programbench_develop_transition_attestation_valid(
    world: Any,
    state: Mapping[str, Any],
) -> bool:
    """Re-attest the one-way EXPLORE->DEVELOP transition provenance.

    Current reference/contract signals may legitimately become false after a
    later probe revision.  The top-level phase nevertheless stays DEVELOP.
    This bounded snapshot proves that the original transition itself crossed
    the public quota with a validated typed contract; the original accepted
    patch and mirrored world event make a hand-edited phase string
    insufficient.

    v2 binds the cumulative coverage document -- its rows, its ordered
    ``receipts`` list and its ``coverage_sha256`` -- rather than the single
    receipt that used to have to fill the quota exactly. That is what keeps the
    transition replayable now that the evidence arrives in batches: anyone
    holding those receipts can re-fold them and land on the same digest.
    """

    attestation = state.get("develop_transition_attestation")
    required = {
        "schema_version",
        "transition_tick",
        "evidence_through_tick",
        "public_evidence_digest",
        "probe_corpus_digest",
        "behavior_ledger",
        "behavior_coverage",
        "contract_acceptance",
        "role_assignments_sha256",
        "public_document_coverage_sha256",
        "attestation_sha256",
    }
    if not isinstance(attestation, Mapping) or set(attestation) != required:
        return False
    unsigned = {
        key: copy.deepcopy(value)
        for key, value in attestation.items()
        if key != "attestation_sha256"
    }
    transition_tick = attestation.get("transition_tick")
    evidence_through = attestation.get("evidence_through_tick")
    evidence_digest = str(attestation.get("public_evidence_digest") or "")
    corpus_digest = str(attestation.get("probe_corpus_digest") or "")
    ledger = attestation.get("behavior_ledger")
    coverage = attestation.get("behavior_coverage")
    acceptance = attestation.get("contract_acceptance")
    quota = state.get("public_probe_case_quota")
    quota = quota if isinstance(quota, int) and not isinstance(quota, bool) else 0
    if (
        attestation.get("schema_version")
        != _DEVELOP_TRANSITION_ATTESTATION_SCHEMA_VERSION
        or not _SHA256.fullmatch(
            str(attestation.get("attestation_sha256") or "")
        )
        or _canonical_digest(unsigned) != attestation.get("attestation_sha256")
        or isinstance(transition_tick, bool)
        or not isinstance(transition_tick, int)
        or transition_tick < EXPLORE_OBSERVATION_FULL_TICKS + 1
        or evidence_through != transition_tick - 1
        or transition_tick != state.get("phase_started_tick")
        or not _SHA256.fullmatch(evidence_digest)
        or not _SHA256.fullmatch(corpus_digest)
        or not isinstance(ledger, Mapping)
        or ledger.get("schema_version") != _BEHAVIOR_LEDGER_SCHEMA_VERSION
        or ledger.get("case_quota") != quota
        or ledger.get("public_evidence_digest") != evidence_digest
        or ledger.get("probe_corpus_digest") != corpus_digest
        or not isinstance(ledger.get("rows"), list)
        or not ledger["rows"]
        or not _programbench_behavior_coverage_valid(coverage, quota=quota)
        or not _programbench_behavior_coverage_satisfied(coverage, quota=quota)
        or evidence_digest
        not in {
            str(row.get("evidence_digest") or "") for row in coverage["receipts"]
        }
        or not isinstance(acceptance, Mapping)
        or acceptance.get("schema_version")
        != _BEHAVIORAL_CONTRACT_ACCEPTANCE_SCHEMA_VERSION
        or acceptance.get("public_evidence_digest") != evidence_digest
        or acceptance.get("probe_corpus_digest") != corpus_digest
        or attestation.get("role_assignments_sha256")
        != _canonical_digest(state.get("role_assignments") or [])
        or attestation.get("public_document_coverage_sha256")
        != _canonical_digest(state.get("public_document_coverage") or {})
    ):
        return False
    rows = ledger["rows"]
    input_hashes = {
        str(row.get("input_sha256") or "")
        for row in rows
        if isinstance(row, Mapping)
    }
    outcome_hashes = {
        str(row.get("outcome_sha256") or "")
        for row in rows
        if isinstance(row, Mapping)
    }
    if (
        len(input_hashes) != len(rows)
        or any(not _SHA256.fullmatch(value) for value in input_hashes)
        or any(not _SHA256.fullmatch(value) for value in outcome_hashes)
    ):
        return False
    # Outcome diversity is now a property of the accumulation, checked above on
    # the coverage document. Demanding it of the final batch as well would
    # penalise the organization for closing a gap with a small, targeted
    # receipt, which is exactly the behaviour the cumulative quota is for.
    patch_id = str(acceptance.get("patch_id") or "")
    artifact_id = str(acceptance.get("artifact_id") or "")
    content_sha256 = str(acceptance.get("content_sha256") or "")
    patch = (getattr(world, "patches", {}) or {}).get(patch_id)
    patch_tick = getattr(patch, "applied_tick", None)
    patch_text = str(getattr(patch, "new_content", "") or "")
    if (
        patch is None
        or not patch_id
        or not artifact_id
        or not _SHA256.fullmatch(content_sha256)
        or str(getattr(patch, "target_object_id", "") or "") != artifact_id
        or str(getattr(patch, "validation_status", "") or "") != "accepted"
        or hashlib.sha256(patch_text.encode("utf-8")).hexdigest()
        != content_sha256
        or evidence_digest not in patch_text
        or corpus_digest not in patch_text
        or isinstance(patch_tick, bool)
        or not isinstance(patch_tick, int)
        or patch_tick > evidence_through
    ):
        return False
    events = getattr(world, "events", None)
    if not isinstance(events, list) or not any(
        isinstance(event, Mapping)
        and event.get("type") == "programbench_phase_transition"
        and event.get("previous_phase") == ProgramBenchPhase.EXPLORE.value
        and event.get("phase") == ProgramBenchPhase.DEVELOP.value
        and event.get("tick") == transition_tick
        and event.get("develop_transition_attestation_sha256")
        == attestation.get("attestation_sha256")
        for event in events
    ):
        return False
    return True


def validate_programbench_profile_state_for_step(
    world: Any,
    *,
    allow_stale_exploration_contract: bool = False,
) -> None:
    """Validate the claimed v5 control state without mutating the world."""

    if not programbench_profile_active(world):
        raise ProfileAttachmentError("programbench_profile_state_invalid")
    state = world.__dict__.get(PROFILE_STATE_KEY)
    if not isinstance(state, Mapping):
        raise ProfileAttachmentError("programbench_profile_state_invalid")
    tick = getattr(world, "world_tick", None)
    boundary = state.get("phase_boundary_tick")
    signal_names = {item.name for item in fields(ProgramBenchSignals)}
    boolean_signal_names = signal_names - {"unresolved_public_mismatch_count"}
    signals = state.get("signals")
    decision_signals = state.get("decision_signals")
    assignments = state.get("role_assignments")
    scenario = getattr(world, "scenario", None)
    scenario_params = getattr(scenario, "params", None)
    if isinstance(scenario_params, Mapping):
        configured_profile = str(
            scenario_params.get("execution_profile") or "native"
        )
    else:
        configured_profile = str(
            getattr(world, "__dict__", {}).get(
                "execution_profile", PROFILE_ID
            )
        )
    materialized_profile = str(
        getattr(world, "__dict__", {}).get(
            "execution_profile", configured_profile
        )
    )
    if (
        state.get("task_family") != TASK_FAMILY
        or configured_profile != PROFILE_ID
        or materialized_profile != PROFILE_ID
        or state.get("phase") not in {item.value for item in ProgramBenchPhase}
        or state.get("pending_phase") not in {
            item.value for item in ProgramBenchPhase
        }
        or state.get("run_horizon_ticks") != 336
        or state.get("attachment_tick") != 0
        or state.get("public_probe_case_quota")
        not in _SUPPORTED_PUBLIC_PROBE_CASE_QUOTAS
        or not _programbench_behavior_coverage_valid(
            state.get("public_behavior_coverage"),
            quota=int(state.get("public_probe_case_quota") or 0),
        )
        or not isinstance(state.get("frozen_public_probe_contract"), Mapping)
        or state.get("candidate_surface_schema_version")
        != _CANDIDATE_SURFACE_SCHEMA_VERSION
        or state.get("candidate_view_migration_schema")
        != _CANDIDATE_VIEW_MIGRATION_SCHEMA_VERSION
        or not isinstance(state.get("candidate_view_migration_history"), list)
        or isinstance(tick, bool)
        or not isinstance(tick, int)
        or not 0 <= tick <= 336
        or isinstance(boundary, bool)
        or not isinstance(boundary, int)
        or not 0 <= boundary <= tick
        or not isinstance(signals, Mapping)
        or set(signals) != signal_names
        or not isinstance(decision_signals, Mapping)
        or set(decision_signals) != signal_names
        or not isinstance(assignments, list)
        or any(type(signals.get(name)) is not bool for name in boolean_signal_names)
        or any(
            type(decision_signals.get(name)) is not bool
            for name in boolean_signal_names
        )
        or state.get("exploration_full_ticks_required")
        != EXPLORE_OBSERVATION_FULL_TICKS
        or state.get("exploration_transition_eligible_tick")
        != EXPLORE_OBSERVATION_FULL_TICKS + 1
        or state.get("phase_evidence_through_tick") != boundary - 1
    ):
        raise ProfileAttachmentError("programbench_profile_state_invalid")
    try:
        current_signals = ProgramBenchSignals(**dict(signals))
        ProgramBenchSignals(**dict(decision_signals))
        programbench_frozen_public_probe_contract(world)
        for artifact in (
            getattr(world, "product_artifacts", {}) or {}
        ).values():
            status = programbench_frozen_public_probe_path_status(
                world, getattr(artifact, "linked_file_path", None)
            )
            if status.startswith("invalid"):
                raise ValueError(
                    "programbench_public_probe_path_alias_invalid"
                )
        expected_assignments = [
            item.to_dict() for item in assign_fixed_roles(_world_agents(world))
        ]
    except (TypeError, ValueError) as error:
        raise ProfileAttachmentError(
            "programbench_profile_state_invalid"
        ) from error
    if (
        assignments != expected_assignments
        or len(assignments) != len(ProgramBenchWorkRole)
        or current_signals.integration_owner_assigned is not True
        or decision_signals.get("integration_owner_assigned") is not True
    ):
        raise ProfileAttachmentError("programbench_profile_state_invalid")
    current_phase = ProgramBenchPhase(str(state.get("phase") or ""))
    stale_contract_allowed = bool(
        allow_stale_exploration_contract
        and current_phase is ProgramBenchPhase.EXPLORE
    )
    if current_signals.behavioral_contract_accepted and not stale_contract_allowed:
        ledger = state.get("public_behavior_ledger")
        if not isinstance(ledger, Mapping) or not (
            _programbench_current_behavioral_contract_valid(
                world, state, ledger
            )
        ):
            raise ProfileAttachmentError(
                "programbench_behavioral_contract_state_invalid"
            )
    if current_phase is ProgramBenchPhase.DEVELOP:
        phase_started = state.get("phase_started_tick")
        integration_owners = [
            str(row.get("agent_id") or "")
            for row in assignments
            if isinstance(row, Mapping)
            and row.get("work_role")
            == ProgramBenchWorkRole.INTEGRATION_OWNER.value
            and str(row.get("agent_id") or "")
        ] if isinstance(assignments, list) else []
        if (
            tick < EXPLORE_OBSERVATION_FULL_TICKS + 1
            or isinstance(phase_started, bool)
            or not isinstance(phase_started, int)
            or not EXPLORE_OBSERVATION_FULL_TICKS + 1
            <= phase_started <= boundary
            or state.get("previous_phase") != ProgramBenchPhase.EXPLORE.value
            or state.get("phase_transition_count") != 1
            or len(integration_owners) != 1
            or not _programbench_develop_transition_attestation_valid(
                world, state
            )
        ):
            raise ProfileAttachmentError("programbench_profile_state_invalid")


def programbench_live_submission_block_reason(
    world: Any,
    *,
    require_frozen_mainline: bool = False,
    merge_pr: Any | None = None,
) -> str | None:
    """Fail closed when a ProgramBench delivery uses stale public evidence.

    The profile state is committed only at a tick boundary. Repository and
    probe edits made later in that same tick therefore cannot rely on stale
    signals: this live attestation re-hashes both public inputs immediately
    before an irreversible delivery. Native worlds are a strict no-op.

    Root-task completion and checkpoint resume use the stronger form.  A
    differential run verifies the complete working tree, whereas one PR may
    promote only a subset of it.  A completed delivery is consequently valid
    only when the frozen digest, verified working digest, current working tree,
    and current mainline are all identical.
    """

    world_dict = getattr(world, "__dict__", {})
    if PROFILE_STATE_KEY not in world_dict:
        return None
    raw_state = world_dict.get(PROFILE_STATE_KEY)
    if not (
        isinstance(raw_state, Mapping)
        and raw_state.get("profile_id") == PROFILE_ID
    ):
        return "programbench_profile_state_invalid"
    if not programbench_profile_active(world):
        return "programbench_profile_state_invalid"
    state = get_programbench_profile_state(world)
    if state is None:
        return "programbench_profile_state_invalid"

    attestation_reason, attestation = (
        programbench_current_public_candidate_attestation(world)
    )
    if attestation_reason is not None or attestation is None:
        return attestation_reason or "programbench_live_attestation_unavailable"
    current_working = str(attestation["candidate_repo_digest"])
    verification_candidate = current_working
    signals = state.get("signals") or {}
    try:
        from environments.org_env.product.materialize import (
            programbench_integration_candidate_has_pending,
            programbench_mainline_digest,
            programbench_pr_candidate_digest,
        )
    except Exception:  # noqa: BLE001 - irreversible delivery fails closed
        return "programbench_live_attestation_unavailable"

    if merge_pr is not None:
        try:
            if programbench_integration_candidate_has_pending(world):
                return "programbench_integration_candidate_commit_required"
        except Exception:  # noqa: BLE001 - delivery ledger fails closed
            return "programbench_live_attestation_unavailable"
        repo_system = getattr(world, "repo_system", None)
        branches = getattr(getattr(repo_system, "repo", None), "branches", {})
        source_branch = branches.get(
            str(getattr(merge_pr, "source_branch", "") or "")
        )
        integration_owners = [
            str(row.get("agent_id") or "")
            for row in (state.get("role_assignments") or [])
            if isinstance(row, Mapping)
            and row.get("work_role")
            == ProgramBenchWorkRole.INTEGRATION_OWNER.value
        ]
        if (
            source_branch is None
            or str(getattr(source_branch, "linked_task", "") or "")
            != "programbench_integration_candidate"
        ):
            return "programbench_merge_requires_integration_candidate"
        if (
            len(integration_owners) != 1
            or not integration_owners[0]
            or str(getattr(source_branch, "owner_id", "") or "")
            != integration_owners[0]
        ):
            return "programbench_merge_requires_integration_owner"
        pr_status = str(
            getattr(
                getattr(merge_pr, "status", None),
                "value",
                getattr(merge_pr, "status", ""),
            )
            or ""
        ).casefold()
        approved_by = [
            str(item)
            for item in (getattr(merge_pr, "approved_by", None) or [])
            if str(item)
        ]
        if (
            pr_status != "approved"
            or getattr(merge_pr, "reviewed", False) is not True
            or not approved_by
        ):
            return "programbench_merge_requires_approved_review"
        try:
            merge_candidate_digest = programbench_pr_candidate_digest(
                world, merge_pr
            )
        except Exception:  # noqa: BLE001 - merge candidate must be auditable
            return "programbench_merge_candidate_attestation_unavailable"
        if merge_candidate_digest != verification_candidate:
            return "programbench_merge_candidate_not_verified"
        branch_commit_ids = list(
            getattr(source_branch, "commit_ids", None) or []
        )
        if not branch_commit_ids or list(
            getattr(merge_pr, "commit_ids", None) or []
        ) != branch_commit_ids:
            return "programbench_merge_pr_head_not_synchronized"
        ci_run_ids = list(getattr(merge_pr, "ci_run_ids", None) or [])
        ci_runs = getattr(getattr(repo_system, "repo", None), "ci_runs", {})
        latest_ci = ci_runs.get(ci_run_ids[-1]) if ci_run_ids else None
        if (
            getattr(merge_pr, "ci_passed", False) is not True
            or latest_ci is None
            or str(getattr(latest_ci, "status", "") or "") != "passed"
            or str(getattr(latest_ci, "commit_id", "") or "")
            != branch_commit_ids[-1]
            or getattr(merge_pr, "ci_base_main_commit_ids", None) is None
            or tuple(getattr(merge_pr, "ci_base_main_commit_ids", ()))
            != tuple(getattr(getattr(repo_system, "repo", None), "main_commit_ids", ()))
            or str(getattr(merge_pr, "ci_tree_hash", "") or "")
            != merge_candidate_digest
        ):
            return "programbench_merge_requires_current_exact_ci"

    if not require_frozen_mainline:
        return None
    if signals.get("candidate_digest_frozen") is not True:
        return "programbench_candidate_freeze_required"
    frozen_candidate = str(state.get("frozen_candidate_digest") or "")
    try:
        current_mainline = programbench_mainline_digest(world)
    except Exception:  # noqa: BLE001 - irreversible delivery fails closed
        return "programbench_live_attestation_unavailable"
    if not (
        frozen_candidate
        and frozen_candidate == verification_candidate
        and frozen_candidate == current_working
        and frozen_candidate == current_mainline
    ):
        return "programbench_frozen_candidate_stale"
    return None


def refresh_programbench_candidate_freeze(world: Any) -> bool:
    """Bind the frozen candidate only when verified working and mainline agree.

    The explicit transactional merge calls this after mainline promotion and
    before task evidence.  A partial promotion cannot freeze a differential
    verdict for a larger working tree.
    """

    if not programbench_profile_active(world):
        raise ProfileAttachmentError("programbench_profile_not_active")
    state = world.__dict__[PROFILE_STATE_KEY]
    base_reason = programbench_live_submission_block_reason(world)
    freeze_is_verified = False
    candidate_digest = ""
    if base_reason is None:
        from environments.org_env.product.materialize import (
            programbench_integration_candidate_digest,
            programbench_mainline_digest,
        )

        candidate_digest = programbench_integration_candidate_digest(world)
        freeze_is_verified = bool(
            candidate_digest
            and candidate_digest == programbench_mainline_digest(world)
            and candidate_digest
            == str(state.get("public_verification_candidate_repo_digest") or "")
        )
    update_programbench_signals(
        world, candidate_digest_frozen=freeze_is_verified
    )
    state["frozen_candidate_digest"] = (
        candidate_digest if freeze_is_verified else None
    )
    if freeze_is_verified:
        # This monotone public attestation survives later integration edits and
        # gives final evaluation a safe fallback without inspecting hidden data.
        state["strict_verified_mainline_digest"] = candidate_digest
        state["strict_mainline_attestation_kind"] = "public_verified_merge"
    return freeze_is_verified


def _close_protocol_adaptation_window(
    state: dict[str, Any],
    *,
    tick: int,
    reason: str,
) -> None:
    started = state.get("protocol_adaptation_started_tick")
    if started is not None:
        history = state.setdefault("protocol_adaptation_history", [])
        history.append(
            {
                "phase": state.get("phase"),
                "started_tick": int(started),
                "minimum_until_tick": state.get(
                    "protocol_adaptation_min_until_tick"
                ),
                "ended_tick": int(tick),
                "reason": str(reason),
                "trigger_evidence_digest": state.get(
                    "protocol_adaptation_trigger_evidence_digest"
                ),
                "target_protocol_ids": list(
                    state.get("protocol_friction_target_ids") or []
                ),
            }
        )
        if len(history) > 16:
            del history[:-16]
    state["protocol_adaptation_started_tick"] = None
    state["protocol_adaptation_min_until_tick"] = None
    state["protocol_adaptation_until_tick"] = None
    state["protocol_adaptation_active"] = False
    state["protocol_adaptation_trigger_evidence_digest"] = None
    state["protocol_adaptation_closed_tick"] = int(tick)


def refresh_programbench_protocol_adaptation(
    world: Any,
) -> dict[str, Any]:
    """Refresh the deterministic transition/repair state from public events.

    A window starts no earlier than 16 ticks after a real phase transition,
    only for friction caused by rules adopted before that transition.  Once
    opened, a phase change closes it immediately; otherwise it lasts at least
    16 ticks and then remains open only while qualifying friction is live.
    """

    if not programbench_profile_active(world):
        raise ProfileAttachmentError("programbench_profile_not_active")
    state = world.__dict__[PROFILE_STATE_KEY]
    tick = int(getattr(world, "world_tick", 0) or 0)
    observation_until = state.get("protocol_observation_until_tick")
    if observation_until is None:
        state["protocol_transition_mode"] = "initial"
        state["protocol_adaptation_active"] = False
        return dict(state)

    _extend_protocol_transition_cohort(world, state)
    evidence = collect_programbench_protocol_friction(world)
    state["protocol_friction_evidence"] = evidence.to_dict()
    state["protocol_observed_friction_target_ids"] = list(
        evidence.friction_target_ids
    )
    evidence_rows = {
        str(row.get("protocol_id") or ""): row
        for row in evidence.target_metrics
        if isinstance(row, Mapping)
    }
    repair_eligible = sorted(
        protocol_id
        for protocol_id, row in evidence_rows.items()
        if protocol_id and row.get("qualified") is True
    )
    state["protocol_friction_target_ids"] = repair_eligible
    state["protocol_repair_eligible_target_ids"] = repair_eligible
    state["protocol_live_friction_target_ids"] = list(
        evidence.live_friction_target_ids
    )
    cohort_rows = _protocol_cohort_rows(state)
    observing_ids = sorted(
        str(row["canonical_protocol_id"])
        for row in cohort_rows
        if tick < int(row.get("observation_until_tick") or 0)
    )
    state["protocol_observing_ids"] = observing_ids
    if tick < int(observation_until):
        state["protocol_transition_mode"] = "observing"
        state["protocol_adaptation_active"] = False
        return dict(state)

    active = bool(state.get("protocol_adaptation_active"))
    if not active and evidence.present:
        state["protocol_adaptation_started_tick"] = tick
        state["protocol_adaptation_min_until_tick"] = (
            tick + PROTOCOL_ADAPTATION_MIN_TICKS
        )
        # Compatibility/audit alias: the guaranteed part of the dynamic
        # window, not its final end when friction persists.
        state["protocol_adaptation_until_tick"] = (
            tick + PROTOCOL_ADAPTATION_MIN_TICKS
        )
        state["protocol_adaptation_active"] = True
        state["protocol_adaptation_trigger_evidence_digest"] = (
            evidence.evidence_digest
        )
        state["protocol_transition_mode"] = "adaptation_minimum"
        return dict(state)

    if active:
        minimum_until = int(
            state.get("protocol_adaptation_min_until_tick") or tick
        )
        if tick < minimum_until:
            state["protocol_transition_mode"] = "adaptation_minimum"
            return dict(state)
        if evidence.present:
            state["protocol_transition_mode"] = "adaptation_friction_live"
            return dict(state)
        _close_protocol_adaptation_window(
            state,
            tick=tick,
            reason="public_friction_disappeared",
        )
        state["protocol_transition_mode"] = (
            "observing_late_carryover"
            if observing_ids
            else "waiting_for_friction"
        )
        return dict(state)

    state["protocol_transition_mode"] = (
        "observing_late_carryover"
        if observing_ids
        else "waiting_for_friction"
    )
    return dict(state)


def programbench_transition_repair_allowed(
    world: Any,
    protocol_id: str | None,
) -> bool:
    """Gate only system-dealt repair for rules inherited across a phase.

    Native worlds and protocols adopted during the current phase retain their
    original repair behavior.  Agent-authored formation/repair actions are not
    hard-blocked by this helper; callers use it only for automatic repair
    dealers and backstops.
    """

    claimed = PROFILE_STATE_KEY in getattr(world, "__dict__", {})
    if not programbench_profile_active(world):
        if claimed:
            raise ProfileAttachmentError("programbench_profile_state_invalid")
        return True
    state = refresh_programbench_protocol_adaptation(world)
    target = _protocol_spec_id(world, protocol_id)
    inherited = {
        str(item) for item in (state.get("transition_protocol_ids") or [])
    }
    if target not in inherited:
        return True
    return bool(
        state.get("protocol_adaptation_active")
        and target
        in set(state.get("protocol_repair_eligible_target_ids") or [])
    )


def programbench_agent_phase_context(world: Any) -> dict[str, Any] | None:
    """Return bounded public phase/window context suitable for agent prompts."""

    if not programbench_profile_active(world):
        return None
    state = refresh_programbench_protocol_adaptation(world)
    tick = int(getattr(world, "world_tick", 0) or 0)
    phase_started = int(state.get("phase_started_tick") or 0)
    exploration_eligible = int(
        state.get("exploration_transition_eligible_tick") or 0
    )
    explore_floor_active = bool(
        state.get("phase") == ProgramBenchPhase.EXPLORE.value
        and tick < exploration_eligible
    )
    explore_protocol_formation_active = bool(
        explore_floor_active
        and tick
        >= exploration_eligible - EXPLORE_PROTOCOL_FORMATION_TICKS
    )
    observation_until = state.get("protocol_observation_until_tick")
    cohort_rows = _protocol_cohort_rows(state)
    latest_observation_until = max(
        (
            int(row.get("observation_until_tick") or 0)
            for row in cohort_rows
        ),
        default=(int(observation_until) if observation_until is not None else 0),
    )
    evidence = state.get("protocol_friction_evidence")
    metrics = evidence.get("target_metrics", []) if isinstance(evidence, Mapping) else []
    reference_summary = programbench_reference_observation_summary(world)
    reference_preview = programbench_reference_observation_brief(
        world, max_cases=6
    )
    public_evidence = state.get("public_evidence")
    public_failure = (
        str(public_evidence.get("failure") or "")
        if isinstance(public_evidence, Mapping)
        else ""
    )
    public_repair = programbench_public_repair_brief(world)
    from .public_evidence import redact_public_text

    friction_summary = [
        {
            "protocol_id": row.get("protocol_id"),
            "reasons": list(row.get("reasons") or []),
            "blocked_context_count": row.get("blocked_context_count"),
            "violation_count": row.get("violation_count"),
            "use_count": row.get("use_count"),
            "latest_friction_tick": row.get("latest_friction_tick"),
        }
        for row in metrics
        if isinstance(row, Mapping) and row.get("qualified")
    ]
    return {
        "profile_id": PROFILE_ID,
        "current_phase": state.get("phase"),
        "previous_phase": state.get("previous_phase"),
        "entered_at_tick": phase_started,
        "phase_age_ticks": max(0, tick - phase_started),
        "phase_evidence_through_tick": state.get("phase_evidence_through_tick"),
        "pending_phase": state.get("pending_phase"),
        "pending_phase_reason": state.get("pending_phase_reason"),
        "transition_count": int(state.get("phase_transition_count") or 0),
        "exploration_observation_full_ticks_required": (
            EXPLORE_OBSERVATION_FULL_TICKS
        ),
        "exploration_transition_eligible_tick": exploration_eligible,
        "exploration_observation_active": explore_floor_active,
        "exploration_full_action_ticks_remaining": (
            max(0, exploration_eligible - tick) if explore_floor_active else 0
        ),
        "exploration_protocol_formation_preference_active": (
            explore_protocol_formation_active
        ),
        "exploration_protocol_formation_guidance": (
            "If repeated public work has exposed a coordination need, the team "
            "may discuss or propose its own evidence-grounded rule. Formation "
            "is optional and no rule text is prescribed."
            if explore_protocol_formation_active
            else "Observe public work before deciding whether a standing rule is useful."
        ),
        "required_public_documents": [
            {
                "artifact_id": row.get("artifact_id"),
                "path": row.get("path"),
            }
            for row in state.get("required_public_documents") or []
            if isinstance(row, Mapping)
        ],
        "public_document_coverage": dict(
            state.get("public_document_coverage") or {}
        ),
        "public_reference_observations": reference_summary,
        "public_reference_observation_preview": (
            reference_preview if reference_preview else None
        ),
        "public_probe_failure": (
            redact_public_text(public_failure, max_chars=512)
            if public_failure
            else None
        ),
        "public_repair_brief": (
            redact_public_text(public_repair, max_chars=4_096)
            if public_repair
            else None
        ),
        "protocol_transition_mode": state.get("protocol_transition_mode"),
        "observation_ticks_required": PROTOCOL_FRICTION_OBSERVATION_TICKS,
        "observation_until_tick": observation_until,
        "latest_rule_observation_until_tick": (
            latest_observation_until or None
        ),
        "observation_ticks_remaining": (
            max(0, latest_observation_until - tick)
        ),
        "inherited_protocol_ids": list(
            state.get("transition_protocol_ids") or []
        ),
        "protocol_observing_ids": list(
            state.get("protocol_observing_ids") or []
        ),
        "protocol_cohort": [
            {
                "protocol_id": row.get("canonical_protocol_id"),
                "cohort_reason": row.get("cohort_reason"),
                "observation_started_tick": row.get(
                    "observation_started_tick"
                ),
                "observation_until_tick": row.get("observation_until_tick"),
                "observation_ticks_remaining": max(
                    0, int(row.get("observation_until_tick") or 0) - tick
                ),
            }
            for row in cohort_rows
        ],
        "public_friction_targets": list(
            state.get("protocol_friction_target_ids") or []
        ),
        "live_public_friction_targets": list(
            state.get("protocol_live_friction_target_ids") or []
        ),
        "public_friction_summary": friction_summary,
        "public_friction_evidence_digest": (
            evidence.get("evidence_digest")
            if isinstance(evidence, Mapping)
            else None
        ),
        "repair_window_active": bool(
            state.get("protocol_adaptation_active")
        ),
        "repair_window_started_tick": state.get(
            "protocol_adaptation_started_tick"
        ),
        "repair_window_minimum_until_tick": state.get(
            "protocol_adaptation_min_until_tick"
        ),
        "repair_guidance": (
            "Prefer amending or relaxing the evidenced inherited rule before "
            "adding a new rule."
            if state.get("protocol_adaptation_active")
            else (
                "Observe inherited and evidence-linked carryover rules for each "
                "rule's full 16-tick clock. Discussing protocols remains "
                "available, but amend only after distinct public friction matures."
            )
        ),
    }


def get_programbench_signals(world: Any) -> ProgramBenchSignals:
    """Return the validated public workflow signals for an active profile."""

    if not programbench_profile_active(world):
        raise ProfileAttachmentError("programbench_profile_not_active")
    raw = world.__dict__[PROFILE_STATE_KEY].get("signals")
    if not isinstance(raw, Mapping):
        raise ProfileAttachmentError("programbench_profile_signals_missing")
    allowed = {item.name for item in fields(ProgramBenchSignals)}
    if set(raw) != allowed:
        raise ProfileAttachmentError("programbench_profile_signals_invalid")
    try:
        return ProgramBenchSignals(**dict(raw))
    except (TypeError, ValueError) as error:
        raise ProfileAttachmentError("programbench_profile_signals_invalid") from error


def update_programbench_signals(
    world: Any,
    *,
    public_evidence_digest: str | None = None,
    **changes: Any,
) -> ProgramBenchPhaseState:
    """Stage explicit public signals for the next tick boundary.

    Callers must name dataclass fields.  This prevents a materializer receipt
    from smuggling evaluator-only keys into checkpointed profile state.
    Evidence produced by agent N at tick T is intentionally unavailable as a
    decision-phase transition to later agents at tick T.
    """

    current = get_programbench_signals(world)
    allowed = {item.name for item in fields(ProgramBenchSignals)}
    unknown = sorted(set(changes) - allowed)
    if unknown:
        raise ValueError("programbench_signal_unknown:" + unknown[0])
    try:
        updated = replace(current, **changes)
    except (TypeError, ValueError) as error:
        raise ValueError("programbench_signal_update_invalid") from error
    return update_programbench_phase(
        world,
        updated,
        public_evidence_digest=public_evidence_digest,
    )


def update_programbench_phase(
    world: Any,
    signals: ProgramBenchSignals,
    *,
    public_evidence_digest: str | None = None,
) -> ProgramBenchPhaseState:
    """Stage a public-evidence snapshot; phase commit happens at tick start."""

    if not programbench_profile_active(world):
        raise ProfileAttachmentError("programbench_profile_not_active")
    if not isinstance(signals, ProgramBenchSignals):
        raise ValueError("programbench_signal_update_invalid")
    state = world.__dict__[PROFILE_STATE_KEY]
    coverage = _public_document_coverage(state)
    state["public_document_coverage"] = coverage
    if signals.public_knowledge_reviewed and not coverage["complete"]:
        raise ValueError("programbench_public_knowledge_requires_auditable_reads")
    current_phase_value = str(state.get("phase") or "")
    derived = derive_phase(signals)
    if current_phase_value == ProgramBenchPhase.DEVELOP.value:
        derived = ProgramBenchPhaseState(
            ProgramBenchPhase.DEVELOP,
            "development_priority:" + _development_priority(signals),
        )
    digest: str | None = None
    if public_evidence_digest is not None:
        digest = str(public_evidence_digest)
        if not _SHA256.fullmatch(digest):
            raise ValueError("programbench_public_evidence_digest_invalid")
    state["signals"] = signals.to_dict()
    state["signals_staged_tick"] = int(getattr(world, "world_tick", 0) or 0)
    state["pending_phase"] = derived.phase.value
    state["pending_phase_reason"] = derived.reason
    if digest is not None:
        state["public_evidence_digest"] = digest
    try:
        current_phase = ProgramBenchPhase(str(state.get("phase") or ""))
    except ValueError as error:
        raise ProfileAttachmentError("programbench_profile_phase_invalid") from error
    return ProgramBenchPhaseState(
        current_phase, str(state.get("phase_reason") or "")
    )


def _commit_programbench_phase_transition(
    world: Any,
    state: dict[str, Any],
    derived: ProgramBenchPhaseState,
    *,
    transition_tick: int,
) -> None:
    previous_phase = str(state.get("phase") or "")
    next_phase = derived.phase.value
    if previous_phase != next_phase:
        if (
            previous_phase == ProgramBenchPhase.EXPLORE.value
            and next_phase == ProgramBenchPhase.DEVELOP.value
        ):
            ledger = state.get("public_behavior_ledger")
            acceptance = state.get("accepted_behavioral_contract")
            coverage = state.get("public_behavior_coverage")
            quota = state.get("public_probe_case_quota")
            quota = (
                quota
                if isinstance(quota, int) and not isinstance(quota, bool)
                else 0
            )
            if not (
                isinstance(ledger, Mapping)
                and isinstance(acceptance, Mapping)
                and _programbench_behavior_coverage_satisfied(
                    coverage, quota=quota
                )
                and _programbench_current_behavioral_contract_valid(
                    world, state, ledger
                )
            ):
                raise ProfileAttachmentError(
                    "programbench_develop_transition_evidence_invalid"
                )
            transition_payload = {
                "schema_version": (
                    _DEVELOP_TRANSITION_ATTESTATION_SCHEMA_VERSION
                ),
                "transition_tick": int(transition_tick),
                "evidence_through_tick": int(transition_tick) - 1,
                "public_evidence_digest": str(
                    ledger.get("public_evidence_digest") or ""
                ),
                "probe_corpus_digest": str(
                    ledger.get("probe_corpus_digest") or ""
                ),
                "behavior_ledger": copy.deepcopy(dict(ledger)),
                "behavior_coverage": copy.deepcopy(dict(coverage)),
                "contract_acceptance": copy.deepcopy(dict(acceptance)),
                "role_assignments_sha256": _canonical_digest(
                    state.get("role_assignments") or []
                ),
                "public_document_coverage_sha256": _canonical_digest(
                    state.get("public_document_coverage") or {}
                ),
            }
            transition_payload["attestation_sha256"] = _canonical_digest(
                transition_payload
            )
            state["develop_transition_attestation"] = transition_payload
        if state.get("protocol_adaptation_active"):
            _close_protocol_adaptation_window(
                state,
                tick=transition_tick,
                reason="phase_transition",
            )
        registry_cursor = len(
            list(
                getattr(
                    getattr(world, "protocol_registry", None),
                    "events",
                    None,
                )
                or []
            )
        )
        world_cursor = len(list(getattr(world, "events", None) or []))
        evidence_through_tick = transition_tick - 1
        evidence_refs = _transition_public_evidence_refs(
            world,
            state,
            evidence_through_tick=evidence_through_tick,
        )
        evidence_snapshot = {
            "previous_phase": previous_phase or None,
            "next_phase": next_phase,
            "evidence_through_tick": evidence_through_tick,
            "refs": evidence_refs,
        }
        transition_evidence_digest = _canonical_digest(evidence_snapshot)
        cohort = [
            _cohort_row(
                record,
                phase=previous_phase,
                observation_started_tick=transition_tick,
                registry_cursor=registry_cursor,
                world_cursor=world_cursor,
                reason="adopted_at_transition",
                evidence_digest=transition_evidence_digest,
            )
            for record in _adopted_protocol_records(world)
        ]
        inherited_protocol_ids = [
            str(row["canonical_protocol_id"]) for row in cohort
        ]
        state["previous_phase"] = previous_phase or None
        state["phase_started_tick"] = transition_tick
        state["phase_transition_count"] = (
            int(state.get("phase_transition_count") or 0) + 1
        )
        state["transition_protocol_ids"] = inherited_protocol_ids
        state["protocol_transition_cohort"] = cohort
        state["protocol_transition_evidence_refs"] = evidence_refs
        state["protocol_transition_evidence_digest"] = (
            transition_evidence_digest
        )
        state["protocol_transition_evidence_through_tick"] = (
            evidence_through_tick
        )
        state["protocol_observing_ids"] = list(inherited_protocol_ids)
        state["protocol_observed_friction_target_ids"] = []
        state["protocol_repair_eligible_target_ids"] = []
        state["protocol_registry_event_cursor"] = registry_cursor
        state["protocol_world_event_cursor"] = world_cursor
        state["protocol_transition_mode"] = "observing"
        state["protocol_observation_until_tick"] = (
            transition_tick + PROTOCOL_FRICTION_OBSERVATION_TICKS
        )
        state["protocol_adaptation_started_tick"] = None
        state["protocol_adaptation_min_until_tick"] = None
        state["protocol_adaptation_until_tick"] = None
        state["protocol_adaptation_active"] = False
        state["protocol_adaptation_trigger_evidence_digest"] = None
        state["protocol_friction_target_ids"] = []
        state["protocol_live_friction_target_ids"] = []
        state["protocol_friction_evidence"] = None
        events = getattr(world, "events", None)
        if isinstance(events, list):
            events.append(
                {
                    "type": "programbench_phase_transition",
                    "subtype": "public_workflow_phase_changed",
                    "previous_phase": previous_phase or None,
                    "phase": next_phase,
                    "tick": transition_tick,
                    "evidence_through_tick": transition_tick - 1,
                    "protocol_observation_until_tick": (
                        transition_tick + PROTOCOL_FRICTION_OBSERVATION_TICKS
                    ),
                    "transition_protocol_ids": inherited_protocol_ids,
                    "protocol_transition_evidence_digest": (
                        transition_evidence_digest
                    ),
                    "develop_transition_attestation_sha256": (
                        (state.get("develop_transition_attestation") or {}).get(
                            "attestation_sha256"
                        )
                    ),
                    "auto": True,
                }
            )
    state.update(derived.to_dict())


def _migrate_programbench_candidate_view_state(
    world: Any,
    state: dict[str, Any],
    *,
    tick: int,
) -> bool:
    """One-shot migration from ambient workspace to the official candidate view.

    Old checkpoints may carry a valid public reference receipt and an ambient
    working-tree differential digest.  Reference receipts/cache, accepted
    contract state, protocol state, and the resource ledger remain valid.  The
    current differential/freeze claim and public-test suppression/cache are
    candidate-view-specific and are cleared before the next action.  A compact
    history row archives the old digest without retaining candidate-dependent
    output as live evidence.  The surface marker makes the operation idempotent.
    """

    from environments.org_env.product.materialize import (
        PROGRAMBENCH_CANDIDATE_VIEW_SCHEMA_VERSION,
    )

    if (
        PROGRAMBENCH_CANDIDATE_VIEW_SCHEMA_VERSION
        != _CANDIDATE_SURFACE_SCHEMA_VERSION
    ):
        raise ProfileAttachmentError(
            "programbench_candidate_surface_schema_mismatch"
        )
    if state.get("candidate_surface_schema_version") == (
        _CANDIDATE_SURFACE_SCHEMA_VERSION
    ):
        return False
    signals_raw = state.get("signals")
    if not isinstance(signals_raw, Mapping):
        raise ProfileAttachmentError("programbench_profile_signals_invalid")
    try:
        signals = ProgramBenchSignals(**dict(signals_raw))
    except (TypeError, ValueError) as error:
        raise ProfileAttachmentError("programbench_profile_signals_invalid") from error
    history_raw = state.get("candidate_view_migration_history", None)
    if "candidate_view_migration_history" not in state:
        # Checkpoints created before candidate-view migrations existed have no
        # history field at all.  That is a valid empty history, while an
        # explicitly present malformed value remains a fail-closed error.
        history: list[Any] = []
    elif not isinstance(history_raw, list):
        raise ProfileAttachmentError(
            "programbench_candidate_view_migration_history_invalid"
        )
    else:
        history = history_raw
    # Migration is transactional because checkpoint repair runs before any
    # agent action. A malformed PR/event/cache object must not leave a partially
    # cleared state with the new marker set; otherwise the next tick would skip
    # the only repair path. Snapshot only the bounded structures this helper
    # mutates and restore them in place on any exception.
    state_before = copy.deepcopy(state)
    world_fields = (
        "_public_test_history",
        "_public_tests_last_hash",
        "_public_tests_last",
        "_last_public_test_result",
        "_public_test_cache",
    )
    world_before = {
        field: copy.deepcopy(world.__dict__[field])
        for field in world_fields
        if field in world.__dict__
    }
    events = getattr(world, "events", None)
    events_before = copy.deepcopy(events) if isinstance(events, list) else None
    repo = getattr(getattr(world, "repo_system", None), "repo", None)
    pull_requests = getattr(repo, "pull_requests", {}) or {}
    pr_before = {
        str(pr_id): copy.deepcopy(pr)
        for pr_id, pr in pull_requests.items()
    }
    try:
        return _apply_programbench_candidate_view_migration(
            world,
            state,
            tick=tick,
            signals=signals,
            history=history,
        )
    except Exception:
        state.clear()
        state.update(state_before)
        for field in world_fields:
            if field in world_before:
                world.__dict__[field] = world_before[field]
            else:
                world.__dict__.pop(field, None)
        if isinstance(events, list) and events_before is not None:
            events[:] = events_before
        for pr_id, snapshot in pr_before.items():
            live = pull_requests.get(pr_id)
            if live is not None and hasattr(live, "__dict__"):
                live.__dict__.clear()
                live.__dict__.update(copy.deepcopy(snapshot.__dict__))
        raise


def _apply_programbench_candidate_view_migration(
    world: Any,
    state: dict[str, Any],
    *,
    tick: int,
    signals: ProgramBenchSignals,
    history: list[Any],
) -> bool:
    """Apply the already-preflighted candidate-view migration transaction."""

    prior_digest = str(
        state.get("public_verification_candidate_repo_digest") or ""
    )
    prior_public_evidence_digest = str(
        state.get("public_evidence_digest") or ""
    )
    prior_public_evidence = state.get("public_evidence")
    prior_public_evidence_mode = (
        str(prior_public_evidence.get("mode") or "")
        if isinstance(prior_public_evidence, Mapping)
        else ""
    )
    legacy_surface_marker = str(state.get("candidate_view_schema", "") or "")
    had_differential = bool(
        signals.public_verification_complete
        or prior_digest
        or state.get("public_verification_candidate_view_schema")
        or prior_public_evidence_mode == "differential"
    )
    state["signals"] = replace(
        signals,
        public_verification_complete=False,
        unresolved_public_mismatch_count=0,
        candidate_digest_frozen=False,
    ).to_dict()
    state["public_evidence"] = None
    state["public_evidence_digest"] = None
    state["public_candidate_repo_digest"] = None
    state["public_verification_probe_corpus_digest"] = None
    state["public_verification_candidate_repo_digest"] = None
    state["public_verification_tested_candidate_repo_digest"] = None
    state["public_verification_candidate_view_schema"] = None
    state["frozen_candidate_digest"] = None
    state["public_repair_brief"] = ""
    prior_public_test_history = world.__dict__.pop("_public_test_history", None)
    prior_public_test_history_count = (
        len(prior_public_test_history)
        if isinstance(prior_public_test_history, list)
        else 0
    )
    repo = getattr(getattr(world, "repo_system", None), "repo", None)
    retired_pr_ids: list[str] = []
    for pr in (getattr(repo, "pull_requests", {}) or {}).values():
        status = str(
            getattr(getattr(pr, "status", None), "value", getattr(pr, "status", ""))
            or ""
        ).casefold()
        if status in {"merged", "closed", "stale"}:
            continue
        if (
            getattr(pr, "ci_passed", False)
            or getattr(pr, "ci_tree_hash", None) is not None
            or getattr(pr, "ci_base_main_commit_ids", None) is not None
        ):
            retired_pr_ids.append(str(getattr(pr, "pr_id", "") or ""))
        pr.ci_passed = False
        pr.test_status = "unknown"
        pr.__dict__.pop("ci_tree_hash", None)
        pr.ci_base_main_commit_ids = None
    world.__dict__.pop("_public_tests_last_hash", None)
    world.__dict__.pop("_public_tests_last", None)
    world.__dict__.pop("_last_public_test_result", None)
    public_test_cache = world.__dict__.get("_public_test_cache")
    if isinstance(public_test_cache, dict):
        public_test_cache.clear()
    history.append(
        {
            "schema_version": _CANDIDATE_VIEW_MIGRATION_SCHEMA_VERSION,
            "tick": int(tick),
            "from_view": legacy_surface_marker or "ambient_working_tree_legacy",
            "to_view": _CANDIDATE_SURFACE_SCHEMA_VERSION,
            "cleared_differential": had_differential,
            "prior_candidate_digest": prior_digest or None,
            "prior_public_evidence_digest": (
                prior_public_evidence_digest or None
            ),
            "prior_public_evidence_mode": prior_public_evidence_mode or None,
            "prior_public_test_history_count": prior_public_test_history_count,
            "retired_live_pr_ids": sorted(item for item in retired_pr_ids if item),
            "reference_evidence_preserved": True,
            "protocol_state_preserved": True,
            "resource_ledger_preserved": True,
        }
    )
    if len(history) > 8:
        del history[:-8]
    events = getattr(world, "events", None)
    if isinstance(events, list):
        events.append(
            {
                "type": "repo_event",
                "subtype": "programbench_candidate_view_migrated",
                "tick": int(tick),
                "schema_version": _CANDIDATE_VIEW_MIGRATION_SCHEMA_VERSION,
                "candidate_view": _CANDIDATE_SURFACE_SCHEMA_VERSION,
                "cleared_differential": had_differential,
                "auto": True,
            }
        )
    state.pop("candidate_view_schema", None)
    state["candidate_view_migration_history"] = history
    state["candidate_view_migration_schema"] = (
        _CANDIDATE_VIEW_MIGRATION_SCHEMA_VERSION
    )
    # Commit marker last: only a completely applied and auditable migration is
    # allowed to make future begin-tick calls no-op.
    state["candidate_surface_schema_version"] = _CANDIDATE_SURFACE_SCHEMA_VERSION
    return True


def _refresh_programbench_exploration_gate(
    world: Any,
    state: dict[str, Any],
    signals: ProgramBenchSignals,
) -> ProgramBenchSignals:
    """Re-attest the current receipt and the accumulated coverage.

    Two independent facts are decided here and they used to be one. Whether the
    *current* definition bytes have a valid reference receipt controls what the
    contract may be bound to. Whether *enough distinct behaviour has ever been
    observed* controls the quota gate, and that is cumulative: it survives probe
    edits, because a past observation is not a claim about the current file.
    """

    bound = _programbench_bound_reference_evidence(world)
    ledger: dict[str, Any] | None = None
    if bound is not None:
        evidence, evidence_digest, corpus_digest = bound
        ledger = programbench_reference_behavior_ledger(
            world,
            evidence,
            evidence_digest=evidence_digest,
            corpus_digest=corpus_digest,
        )
    stored = state.get("public_behavior_ledger")
    ledger_valid = bool(
        ledger is not None
        and isinstance(stored, Mapping)
        and dict(stored) == ledger
        and str(state.get("exploration_reference_evidence_digest") or "")
        == str(ledger.get("public_evidence_digest") or "")
        and str(state.get("public_probe_evidence_corpus_digest") or "")
        == str(ledger.get("probe_corpus_digest") or "")
    )
    if not ledger_valid:
        state["public_behavior_ledger"] = None
    quota = state.get("public_probe_case_quota")
    quota = quota if isinstance(quota, int) and not isinstance(quota, bool) else 0
    coverage = state.get("public_behavior_coverage")
    if not _programbench_behavior_coverage_valid(coverage, quota=quota):
        coverage = programbench_empty_behavior_coverage(quota)
        state["public_behavior_coverage"] = coverage
    if ledger_valid and ledger is not None:
        try:
            coverage = programbench_merge_behavior_coverage(
                coverage,
                ledger,
                quota=quota,
                tick=int(getattr(world, "world_tick", 0) or 0),
            )
        except ProgramBenchCoverageConflict:
            # The conflict is the finding. Accumulation stops at the last
            # coherent state rather than absorbing a contradiction; the
            # non-reproducible case is named to the organization through the
            # public repair brief.
            pass
        else:
            state["public_behavior_coverage"] = coverage
    coverage_satisfied = _programbench_behavior_coverage_satisfied(
        coverage, quota=quota
    )
    contract_valid = bool(
        ledger_valid
        and _programbench_current_behavioral_contract_valid(
            world, state, ledger
        )
    )
    if not contract_valid:
        state["accepted_behavioral_contract"] = None
    return replace(
        signals,
        probe_inventory_nonempty=ledger_valid,
        public_probe_execution_observed=ledger_valid,
        exploration_case_quota_satisfied=coverage_satisfied,
        behavior_ledger_complete=bool(ledger_valid and coverage_satisfied),
        behavioral_contract_accepted=contract_valid,
    )


def _programbench_current_behavioral_contract_valid(
    world: Any,
    state: Mapping[str, Any],
    ledger: Mapping[str, Any],
) -> bool:
    """Re-attest the exact typed contract bytes bound to this ledger.

    Contract acceptance is a derived fact, never a sticky caller-controlled
    boolean.  The full-content patch was validated at acceptance; retaining
    its hash, revision and patch identity lets each boundary prove that the
    actual artifact is still those exact validated bytes without rerunning an
    editor or consulting evaluator data.
    """

    acceptance = state.get("accepted_behavioral_contract")
    if not isinstance(acceptance, Mapping) or set(acceptance) != {
        "schema_version",
        "artifact_id",
        "patch_id",
        "accepted_revision",
        "content_sha256",
        "public_evidence_digest",
        "probe_corpus_digest",
    }:
        return False
    evidence_digest = str(ledger.get("public_evidence_digest") or "")
    corpus_digest = str(ledger.get("probe_corpus_digest") or "")
    content_digest = str(acceptance.get("content_sha256") or "")
    revision = acceptance.get("accepted_revision")
    if (
        acceptance.get("schema_version")
        != _BEHAVIORAL_CONTRACT_ACCEPTANCE_SCHEMA_VERSION
        or not _SHA256.fullmatch(content_digest)
        or str(acceptance.get("public_evidence_digest") or "")
        != evidence_digest
        or str(acceptance.get("probe_corpus_digest") or "") != corpus_digest
        or isinstance(revision, bool)
        or not isinstance(revision, int)
        or revision <= 0
    ):
        return False
    artifact_id = str(acceptance.get("artifact_id") or "")
    artifacts = getattr(world, "product_artifacts", {}) or {}
    contracts = [
        artifact
        for artifact in artifacts.values()
        if str(getattr(artifact, "programbench_artifact_kind", "") or "")
        == "behavioral_contract"
    ]
    if len(contracts) != 1:
        return False
    artifact = contracts[0]
    body = str(getattr(artifact, "content", "") or "")
    patch_id = str(acceptance.get("patch_id") or "")
    patch = (getattr(world, "patches", {}) or {}).get(patch_id)
    return bool(
        str(getattr(artifact, "artifact_id", "") or "") == artifact_id
        and int(getattr(artifact, "revision", 0) or 0) == revision
        and hashlib.sha256(body.encode("utf-8")).hexdigest() == content_digest
        and evidence_digest in body
        and corpus_digest in body
        and str(
            getattr(artifact, "programbench_contract_content_sha256", "")
            or ""
        )
        == content_digest
        and getattr(artifact, "programbench_contract_accepted_revision", None)
        == revision
        and str(
            getattr(artifact, "programbench_contract_validation_schema", "")
            or ""
        )
        == _BEHAVIORAL_CONTRACT_ACCEPTANCE_SCHEMA_VERSION
        and str(
            getattr(artifact, "programbench_public_evidence_digest", "")
            or ""
        )
        == evidence_digest
        and str(
            getattr(artifact, "programbench_probe_corpus_digest", "") or ""
        )
        == corpus_digest
        and patch is not None
        and str(getattr(patch, "target_object_id", "") or "") == artifact_id
        and str(getattr(patch, "validation_status", "") or "") == "accepted"
        and str(getattr(patch, "new_content", "") or "") == body
        and patch_id in list(getattr(artifact, "patch_history_ids", None) or [])
    )


def begin_programbench_tick(world: Any) -> ProgramBenchPhaseState | None:
    """Atomically commit staged public evidence before any agent acts this tick.

    Native worlds are strict no-ops. Legacy checkpoint schemas fail closed;
    v5 experiments start from a fresh t0 attachment.
    """

    claimed = PROFILE_STATE_KEY in getattr(world, "__dict__", {})
    if not claimed:
        return None
    validate_programbench_profile_state_for_step(
        world, allow_stale_exploration_contract=True
    )
    state = world.__dict__[PROFILE_STATE_KEY]
    tick = int(getattr(world, "world_tick", 0) or 0)
    boundary_tick = state.get("phase_boundary_tick")
    if isinstance(boundary_tick, bool) or not isinstance(boundary_tick, int):
        raise ProfileAttachmentError("programbench_profile_boundary_missing")
    if tick <= boundary_tick:
        return ProgramBenchPhaseState(
            ProgramBenchPhase(str(state.get("phase") or "")),
            str(state.get("phase_reason") or ""),
        )

    if (
        state.get("candidate_surface_schema_version")
        != _CANDIDATE_SURFACE_SCHEMA_VERSION
        or state.get("candidate_view_migration_schema")
        != _CANDIDATE_VIEW_MIGRATION_SCHEMA_VERSION
        or not isinstance(state.get("candidate_view_migration_history"), list)
    ):
        raise ProfileAttachmentError(
            "programbench_candidate_surface_state_invalid"
        )

    signals = get_programbench_signals(world)
    try:
        current_phase = ProgramBenchPhase(str(state.get("phase") or ""))
    except ValueError as error:
        raise ProfileAttachmentError("programbench_profile_phase_invalid") from error
    coverage = _public_document_coverage(state)
    state["public_document_coverage"] = coverage
    if signals.public_knowledge_reviewed != bool(coverage["complete"]):
        signals = replace(
            signals, public_knowledge_reviewed=bool(coverage["complete"])
        )
        state["signals"] = signals.to_dict()
    candidate_surface_schema = _CANDIDATE_SURFACE_SCHEMA_VERSION
    try:
        from environments.org_env.product.materialize import (
            PROGRAMBENCH_CANDIDATE_VIEW_SCHEMA_VERSION,
            programbench_integration_candidate_digest,
            programbench_mainline_digest,
            programbench_probe_corpus_digest,
        )

        if PROGRAMBENCH_CANDIDATE_VIEW_SCHEMA_VERSION != (
            _CANDIDATE_SURFACE_SCHEMA_VERSION
        ):
            raise ValueError("candidate_surface_schema_mismatch")
        # The EXPLORE gate is intentionally independent of an implementation
        # candidate.  A fresh or reference-only world may not have a routable
        # integration branch yet, but its frozen public probe surface is still
        # sufficient to bind the exact reference corpus and typed contract.
        current_corpus = programbench_probe_corpus_digest(world)
    except Exception:  # noqa: BLE001 - public-corpus evidence fails closed
        current_corpus = ""
    try:
        current_candidate_digest = programbench_integration_candidate_digest(world)
        current_mainline_digest = programbench_mainline_digest(world)
    except Exception:  # noqa: BLE001 - downstream candidate claims fail closed
        current_candidate_digest = ""
        current_mainline_digest = ""

    verification_corpus = str(
        state.get("public_verification_probe_corpus_digest") or ""
    )
    verification_candidate = str(
        state.get("public_verification_candidate_repo_digest") or ""
    )
    verification_view_schema = str(
        state.get("public_verification_candidate_view_schema") or ""
    )
    frozen_candidate = str(state.get("frozen_candidate_digest") or "")
    verification_stale = bool(
        signals.public_verification_complete
        and (
            not verification_corpus
            or verification_corpus != current_corpus
            or verification_view_schema
            != candidate_surface_schema
            or not verification_candidate
            or verification_candidate != current_candidate_digest
        )
    )
    freeze_stale = bool(
        signals.candidate_digest_frozen
        and (
            verification_stale
            or not frozen_candidate
            or frozen_candidate != verification_candidate
            or frozen_candidate != current_candidate_digest
            or frozen_candidate != current_mainline_digest
        )
    )
    if verification_stale or freeze_stale:
        signals = replace(
            signals,
            public_verification_complete=(
                False if verification_stale else signals.public_verification_complete
            ),
            candidate_digest_frozen=False,
        )
        if verification_stale:
            state["public_verification_probe_corpus_digest"] = None
            state["public_verification_candidate_repo_digest"] = None
            state["public_verification_tested_candidate_repo_digest"] = None
            state["public_verification_candidate_view_schema"] = None
            state["public_candidate_repo_digest"] = None
        state["frozen_candidate_digest"] = None
        state["signals"] = signals.to_dict()

    if signals.public_probe_execution_observed:
        bound_corpus = str(
            state.get("public_probe_evidence_corpus_digest") or ""
        )
        latest_corpus = str(
            state.get("latest_reference_probe_corpus_digest") or bound_corpus
        )
        if not bound_corpus or bound_corpus != current_corpus:
            if current_phase is ProgramBenchPhase.EXPLORE:
                # A probe edit means the current definition bytes have not been
                # observed yet, so the contract has nothing current to bind to
                # and a fresh reference run is required. It does not mean the
                # organization un-learned anything: `public_behavior_coverage`
                # is deliberately untouched here. Wiping it is what made
                # iterative exploration impossible -- every revision of the
                # definition file discarded every observation already paid for,
                # so the only affordable strategy was to guess the whole corpus
                # in one shot and never look at it again.
                signals = replace(
                    signals,
                    public_probe_execution_observed=False,
                    behavior_ledger_complete=False,
                )
                state["public_behavior_ledger"] = None
                state["public_probe_evidence_corpus_digest"] = None
                state["exploration_reference_evidence_digest"] = None
                state["latest_reference_probe_corpus_digest"] = None
                state["latest_reference_evidence_digest"] = None
                state["latest_reference_probe_required"] = True
            elif not latest_corpus or latest_corpus != current_corpus:
                # After acceptance the original exploration receipt is
                # immutable provenance.  Probe edits require a fresh reference
                # observation and invalidate only downstream verification /
                # freeze claims; they do not blindly erase the contract or the
                # coherent implementation.
                signals = replace(
                    signals,
                    public_verification_complete=False,
                    unresolved_public_mismatch_count=0,
                    candidate_digest_frozen=False,
                )
                state["latest_reference_evidence_digest"] = None
                state["latest_reference_probe_corpus_digest"] = None
                state["latest_reference_probe_required"] = True
                state["public_candidate_repo_digest"] = None
                state["public_repair_brief"] = ""
                state["public_verification_probe_corpus_digest"] = None
                state["public_verification_candidate_repo_digest"] = None
                state["public_verification_tested_candidate_repo_digest"] = None
                state["public_verification_candidate_view_schema"] = None
                state["frozen_candidate_digest"] = None
            state["signals"] = signals.to_dict()
    if current_phase is ProgramBenchPhase.EXPLORE:
        signals = _refresh_programbench_exploration_gate(world, state, signals)
        state["signals"] = signals.to_dict()
    desired = derive_phase(signals)
    if current_phase is ProgramBenchPhase.DEVELOP:
        # DEVELOP is the sole post-exploration state. New probe/code work can
        # create reference, contract, build or verification debt, but those are
        # internal priorities rather than a return to EXPLORE.
        desired = ProgramBenchPhaseState(
            ProgramBenchPhase.DEVELOP,
            "development_priority:" + _development_priority(signals),
        )
    committed = desired
    eligible_tick = int(state.get("exploration_transition_eligible_tick") or 0)
    if current_phase is ProgramBenchPhase.EXPLORE and tick < eligible_tick:
        remaining = eligible_tick - tick
        committed = ProgramBenchPhaseState(
            ProgramBenchPhase.EXPLORE,
            f"exploration_observation_floor:{remaining}_full_ticks_remaining",
        )

    _commit_programbench_phase_transition(
        world, state, committed, transition_tick=tick
    )
    state["decision_signals"] = signals.to_dict()
    state["phase_boundary_tick"] = tick
    state["phase_evidence_through_tick"] = tick - 1
    state["pending_phase"] = desired.phase.value
    state["pending_phase_reason"] = desired.reason
    return committed


__all__ = [
    "PROFILE_ID",
    "PROFILE_STATE_KEY",
    "PROFILE_STATE_SCHEMA_VERSION",
    "COMPANY_BRIEF_OVERLAY_SCHEMA_VERSION",
    "EXPLORE_OBSERVATION_FULL_TICKS",
    "EXPLORE_PROTOCOL_FORMATION_TICKS",
    "TASK_FAMILY",
    "PROTOCOL_ADAPTATION_TICKS",
    "PROTOCOL_ADAPTATION_MIN_TICKS",
    "PROTOCOL_FRICTION_OBSERVATION_TICKS",
    "PROTOCOL_FRICTION_RECENCY_TICKS",
    "PROTOCOL_FRICTION_SCHEMA_VERSION",
    "PROTOCOL_TRANSITION_COHORT_SCHEMA_VERSION",
    "PRODUCT_CONTEXT_LABELS",
    "ProfileAttachmentError",
    "ProgramBenchPhase",
    "ProgramBenchWorkRole",
    "TaskFamilyDetection",
    "ProgramBenchSignals",
    "ProgramBenchPhaseState",
    "ProgramBenchProtocolFriction",
    "RoleAssignment",
    "PhaseActionDecision",
    "ProfileAttachment",
    "detect_programbench_task_family",
    "programbench_root_task_brief",
    "programbench_company_brief_overlay",
    "programbench_current_public_candidate_attestation",
    "programbench_workflow_overlay",
    "assign_fixed_roles",
    "derive_phase",
    "candidate_decision",
    "profile_policy_bonus",
    "attach_programbench_profile",
    "begin_programbench_tick",
    "programbench_profile_active",
    "validate_programbench_profile_state_for_step",
    "programbench_live_submission_block_reason",
    "refresh_programbench_candidate_freeze",
    "get_programbench_profile_state",
    "programbench_frozen_public_probe_contract",
    "programbench_reference_observation_summary",
    "programbench_reference_behavior_ledger",
    "programbench_reference_observation_brief",
    "programbench_public_repair_brief",
    "get_programbench_signals",
    "collect_programbench_protocol_friction",
    "refresh_programbench_protocol_adaptation",
    "programbench_transition_repair_allowed",
    "programbench_agent_phase_context",
    "programbench_public_path_is_agent_visible",
    "programbench_retrieved_public_surfaces",
    "programbench_required_public_documents",
    "record_programbench_public_document_read",
    "update_programbench_phase",
    "update_programbench_signals",
]
