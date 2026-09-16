"""Where a change lives from the moment it is made: task -> branch -> PR.

The simulator used to keep every accepted patch in one pool per agent and work
out afterwards which branch each belonged on. Nothing in Git does that, and the
guessing is where the delivery layer kept breaking:

  * an author had one long-lived branch, so opening a request took it out of
    reach and every later change joined the one request -- 17 commits and all
    five modules on one branch, green only when everything in it was right;
  * letting a repair join somebody else's red request rebuilt the same magnet
    from the other side, 11 of 11 commits into one request while four others
    went untouched for 56 to 100 ticks;
  * a request opened on one module was refused six times over for the same
    named symbol, and the fix, when it came, was committed together with an
    unrelated new module because commit took everything in hand at once. The
    verdict stopped naming a symbol and the clue was gone.

None of those are things an author did. They are what a routing layer does when
it has to infer what the author meant.

Here a patch belongs to a branch when it is made, because the work item it is
for is known then: the task it advances, or the module it changes when no task
claims it. A commit takes what is pending on its branch. A request is that
branch. A repair is a commit on the branch whose request is red -- which needs
no rule, since that is where the work already is.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from environments.org_env.backend.repo.repo import Branch, BranchStatus

PENDING = "_pending_by_branch"
PROGRAMBENCH_INTEGRATION_TASK = "programbench_integration_candidate"
PROGRAMBENCH_CONTRACT_TASK = "programbench_behavioral_contract"
# How many changes an organization may have in flight at once.
#
# This was two per agent, which quietly made it a headcount: a founder working
# alone got two, an eight-person organization got sixteen. The ladder varies how
# an organization coordinates, not how much of the repository it is allowed to
# touch, so a per-agent limit put a thumb on the scale of the very comparison it
# was under. b0 filled both of its branches by t24 and spent the remaining 312
# ticks unable to open another: 75 edits, 1 commit, 72 accepted patches stranded
# on branches whose one request had already gone red.
#
# Sixteen for everyone, which is what the largest arm had all along, so no arm
# loses room and the smallest stops being bounded by something that is not about
# coordination at all.
MAX_ACTIVE_BRANCHES = 16
_CLOSED = (BranchStatus.MERGED, BranchStatus.ABANDONED)
_PROGRAMBENCH_RESERVED_LANE_CAPABILITY = object()

_PROGRAMBENCH_CANDIDATE_TYPES = {"code", "repo_file", "tool_stub", "eval"}
_PROGRAMBENCH_NON_CANDIDATE_TYPES = {
    "issue", "doc", "docs", "documentation", "knowledge", "test", "tests",
    "private",
}
_PROGRAMBENCH_NON_CANDIDATE_PATH_PARTS = {
    "doc", "docs", "documentation", "knowledge", "private", ".private",
    "test", "tests", "testdata", "__tests__",
}


def work_items_of(world: Any, patch: Any, artifact: Any) -> List[str]:
    """Everything this change could be said to be for, best first.

    A file often belongs to more than one task, and taking whichever came first
    made the grouping arbitrary: two files of one task landed on two branches
    because each named a different task first.
    """
    keys: List[str] = []
    for source in ((getattr(patch, "related_task_ids", []) or []),
                   (getattr(artifact, "linked_task_ids", []) or [])):
        for tid in source:
            if tid and str(tid) not in keys:
                keys.append(str(tid))
    keys.append(str(getattr(artifact, "artifact_id", "") or "unassigned"))
    return keys


def work_item_of(world: Any, patch: Any, artifact: Any) -> str:
    """What this change is for: its task when it has one, else its module.

    A pack whose steps are modules gives one work item per module even before a
    task exists for it, which is what keeps step 1 landable on its own.
    """
    return work_items_of(world, patch, artifact)[0]


def _branches(world: Any) -> Dict[str, Branch]:
    repo = getattr(getattr(world, "repo_system", None), "repo", None)
    return getattr(repo, "branches", {}) or {}


def active_branches(world: Any, agent_id: str) -> List[Branch]:
    return [b for b in _branches(world).values()
            if getattr(b, "owner_id", None) == agent_id
            and getattr(b, "status", None) not in _CLOSED]


def active_branches_org(world: Any) -> List[Branch]:
    """Everything the organization has in flight, whoever is carrying it.

    A delivery-repair repackage does not count against this. The limit is the
    organization's coordination capacity -- how many pieces of work its members
    can carry at once -- and a repackage is not a member carrying work: it is the
    repair path lifting one already-written fix onto a clean request so the fat
    one blocking it can be worked down. Counting it would let the very congestion
    the repair exists to clear use up the room the repair needs to clear it.
    """
    return [b for b in _branches(world).values()
            if getattr(b, "status", None) not in _CLOSED
            and not getattr(b, "_delivery_repackage", False)]


def branch_creation_capacity_available(
    world: Any,
    *,
    linked_task: Any = None,
    _reserved_lane_capability: Any = None,
) -> bool:
    """Whether one more ordinary branch may be created right now.

    An active ProgramBench run reserves two of the sixteen organization slots:
    one for the single integration branch and one for the evidence-bound
    behavioral-contract lane.  Ordinary work may use only the unreserved
    remainder. Reusing an existing branch remains possible at the limit;
    native worlds retain the exact sixteen-slot rule.
    """

    branches = active_branches_org(world)
    claimed = "programbench_profile_state" in getattr(world, "__dict__", {})
    if not claimed:
        return len(branches) < MAX_ACTIVE_BRANCHES
    try:
        from environments.org_env.programbench import (
            validate_programbench_profile_state_for_step,
        )

        validate_programbench_profile_state_for_step(world)
    except (ImportError, AttributeError, TypeError, ValueError):
        return False
    integration_exists = any(
        str(getattr(branch, "linked_task", "") or "")
        == PROGRAMBENCH_INTEGRATION_TASK
        for branch in branches
    )
    contract_exists = any(
        str(getattr(branch, "linked_task", "") or "")
        == PROGRAMBENCH_CONTRACT_TASK
        for branch in branches
    )
    requested_integration = str(linked_task or "") == PROGRAMBENCH_INTEGRATION_TASK
    requested_contract = str(linked_task or "") == PROGRAMBENCH_CONTRACT_TASK
    if (requested_integration or requested_contract) and (
        _reserved_lane_capability is not _PROGRAMBENCH_RESERVED_LANE_CAPABILITY
    ):
        return False
    reserved = int(not integration_exists and not requested_integration) + int(
        not contract_exists
    )
    limit = MAX_ACTIVE_BRANCHES - reserved
    return len(branches) < limit


def branch_for(world: Any, agent_id: str, key: str, tick: int) -> Optional[Branch]:
    """The agent's branch for this work item, made if it has none.

    Returns None only when the agent is already carrying as much as it may and
    this would be one more thing -- and never for work it is already carrying,
    so a limit cannot strand a change that is already made.
    """
    for b in active_branches(world, agent_id):
        if str(getattr(b, "linked_task", "") or "") == key:
            return b
    if not branch_creation_capacity_available(world, linked_task=key):
        return None
    rs = getattr(world, "repo_system", None)
    if rs is None:
        return None
    b = rs.create_branch(agent_id, tick=tick)
    b.linked_task = key
    personal = (getattr(world, "personal", {}) or {}).get(agent_id)
    if personal is not None and b.branch_id not in personal.local_branch_ids:
        personal.local_branch_ids.append(b.branch_id)
    return b


def branch_already_touching(world: Any, agent_id: str,
                            artifact_id: str) -> Optional[Branch]:
    """The agent's open branch that already changes this file, if any.

    The work item a change is for can be named differently at different moments
    -- a module before a task claims it, the task afterwards -- and keying on the
    name alone then splits one piece of work across two branches the first time
    it is renamed. What the branch is actually about does not change: the file it
    is changing. So a second edit of a file goes where the first one went.
    """
    repo = getattr(getattr(world, "repo_system", None), "repo", None)
    book = world.__dict__.get(PENDING, {}) or {}
    for b in active_branches(world, agent_id):
        if any(a == artifact_id for _p, a in book.get(b.branch_id, [])):
            return b
        for cid in (getattr(b, "commit_ids", []) or []):
            commit = (getattr(repo, "commits", {}) or {}).get(cid)
            if artifact_id in (getattr(commit, "artifact_ids", []) or []):
                return b
    return None


def _normalized_repo_path(value: Any) -> str:
    path = str(value or "").strip().replace("\\", "/")
    while path.startswith("./"):
        path = path[2:]
    return path.strip("/")


def _programbench_non_candidate_path(path: str) -> bool:
    """Keep documentation, test, knowledge, and private assets off the candidate.

    Public probe definitions use ``eval/eval_N.py`` and are deliberately not
    classified as tests here.  Only conventional test locations/names are
    excluded; an ``eval`` artifact on the public probe surface remains part of
    the coherent ProgramBench candidate.
    """
    normalized = _normalized_repo_path(path).casefold()
    if not normalized:
        return False
    parts = tuple(part for part in normalized.split("/") if part)
    if any(part in _PROGRAMBENCH_NON_CANDIDATE_PATH_PARTS for part in parts):
        return True
    name = parts[-1]
    stem = name.rsplit(".", 1)[0]
    return (
        name.startswith("test_")
        or stem.endswith("_test")
        or ".test." in name
        or ".spec." in name
    )


def _programbench_public_build_paths(world: Any) -> set[str]:
    """Agent-visible build-contract paths, never evaluator/private metadata."""
    product = getattr(world, "product", None)
    meta = getattr(product, "substrate_meta", {}) or {}
    if not isinstance(meta, dict):
        return set()
    paths: set[str] = set()
    compile_path = _normalized_repo_path(meta.get("reconstruction_compile_path"))
    if compile_path:
        paths.add(compile_path.casefold())
    # Some packs identify a build metadata file only as a command operand.
    # Accept an operand only when it names an agent-visible repository artifact;
    # this avoids treating the shell/interpreter executable as candidate data.
    visible_paths = {
        _normalized_repo_path(getattr(item, "linked_file_path", None)).casefold()
        for item in (getattr(world, "product_artifacts", {}) or {}).values()
        if _normalized_repo_path(getattr(item, "linked_file_path", None))
    }
    command = meta.get("reconstruction_compile_command") or []
    if isinstance(command, (list, tuple)):
        for token in command:
            normalized = _normalized_repo_path(token).casefold()
            if normalized and normalized in visible_paths:
                paths.add(normalized)
    return paths


def programbench_public_probe_artifact(world: Any, artifact: Any) -> bool:
    """Whether the artifact is on the manifest-declared public probe surface."""

    raw_path = getattr(artifact, "linked_file_path", None)
    path = _normalized_repo_path(raw_path)
    claimed = "programbench_profile_state" in getattr(world, "__dict__", {})
    surface = None
    if claimed:
        try:
            from environments.org_env.programbench import (
                programbench_frozen_public_probe_path_status,
            )

            status = programbench_frozen_public_probe_path_status(world, raw_path)
        except (ImportError, AttributeError, TypeError, ValueError):
            # Invalid adapted claims have no trusted probe exception.
            return False
        if status.startswith("invalid"):
            raise ValueError("programbench_public_probe_path_alias_invalid")
        return status == "trusted"
    else:
        meta = getattr(getattr(world, "product", None), "substrate_meta", {}) or {}
        authoring = meta.get("public_probe_authoring") if isinstance(meta, dict) else None
        surface = authoring.get("definition_surface") if isinstance(authoring, dict) else None
    exact_paths = {
        _normalized_repo_path(item).casefold()
        for item in (surface.get("exact_paths") or [])
    } if isinstance(surface, dict) else set()
    patterns = list(surface.get("path_patterns") or []) if isinstance(surface, dict) else []
    return bool(
        path
        and (
            path.casefold() in exact_paths
            or any(
                isinstance(pattern, str)
                and re.fullmatch(pattern, path) is not None
                for pattern in patterns
            )
        )
    )


def programbench_public_probe_path(world: Any, raw_path: Any) -> bool:
    """Path-only form for contract/source validation before an artifact exists."""

    probe = type("_ProgramBenchProbePath", (), {
        "linked_file_path": raw_path,
    })()
    return programbench_public_probe_artifact(world, probe)


def programbench_candidate_bearing_artifact(world: Any, artifact: Any) -> bool:
    """Whether an artifact belongs on the one ProgramBench delivery branch.

    This is the canonical public classification used by both patch routing and
    the adapted candidate-view materializer.  Keeping one predicate prevents a
    file from being routed as collateral work but later (silently) admitted to
    the verified delivery candidate, or vice versa.
    """

    artifact_type = str(getattr(artifact, "artifact_type", "") or "").casefold()
    path = _normalized_repo_path(getattr(artifact, "linked_file_path", None))
    # Manifest-declared probe paths are candidate-bearing regardless of a
    # caller-supplied artifact type/kind.  Path membership comes from the
    # attachment-time frozen contract in adapted worlds.
    if programbench_public_probe_artifact(world, artifact):
        return True
    # An explicitly declared build dependency is candidate-bearing regardless
    # of its extension or directory.  A reconstructed executable may consume a
    # seed, document, test vector, or data file at runtime; path-name heuristics
    # must not route those declared bytes to a collateral branch.
    if path.casefold() in _programbench_public_build_paths(world):
        return True
    if artifact_type in _PROGRAMBENCH_NON_CANDIDATE_TYPES:
        return False
    if _programbench_non_candidate_path(path):
        return False
    return (
        artifact_type in _PROGRAMBENCH_CANDIDATE_TYPES
    )


def _invalidate_live_pr_attestations(world: Any, branch_id: str) -> None:
    """Retire every live verdict as soon as accepted branch bytes change."""

    repo = getattr(getattr(world, "repo_system", None), "repo", None)
    for pr in (getattr(repo, "pull_requests", {}) or {}).values():
        if str(getattr(pr, "source_branch", "") or "") != branch_id:
            continue
        status = str(
            getattr(getattr(pr, "status", None), "value", getattr(pr, "status", ""))
            or ""
        ).casefold()
        if status in {"merged", "closed", "stale"}:
            continue
        pr.ci_passed = False
        pr.test_status = "unknown"
        pr.__dict__.pop("ci_tree_hash", None)
        pr.ci_base_main_commit_ids = None


def programbench_integration_branches(world: Any) -> List[Branch]:
    """Return all non-terminal branches claiming the ProgramBench candidate."""

    return [
        branch
        for branch in _branches(world).values()
        if getattr(branch, "status", None) not in _CLOSED
        and str(getattr(branch, "linked_task", "") or "")
        == PROGRAMBENCH_INTEGRATION_TASK
    ]


def _programbench_integration_owner(world: Any) -> Tuple[bool, Optional[str]]:
    """Return ``(active, owner)``; active profiles with bad roles fail closed."""
    # Kept lazy so importing the native repository workflow does not activate
    # or otherwise couple it to the optional ProgramBench treatment.
    try:
        from environments.org_env.programbench import (
            get_programbench_profile_state,
            programbench_profile_active,
        )
    except ImportError:
        return False, None
    state_claimed = "programbench_profile_state" in getattr(world, "__dict__", {})
    if not state_claimed:
        return False, None
    try:
        if not programbench_profile_active(world):
            # A malformed or legacy claim is not a native world.  Treat it as
            # active-without-an-owner so direct/internal record_patch callers
            # cannot route around the profile validator.
            return True, None
        state = get_programbench_profile_state(world)
    except (AttributeError, TypeError, ValueError):
        # A state that claims to be active but cannot be read must never fall
        # through to ordinary per-author routing and create parallel candidates.
        return True, None
    assignments = state.get("role_assignments") if isinstance(state, dict) else None
    if not isinstance(assignments, list):
        return True, None
    owners = []
    for assignment in assignments:
        if not isinstance(assignment, dict):
            continue
        if str(assignment.get("work_role") or "") != "integration_owner":
            continue
        owner = str(assignment.get("agent_id") or "").strip()
        if owner:
            owners.append(owner)
    if len(owners) != 1:
        return True, None
    return True, owners[0]


def _programbench_integration_branch(
    world: Any, owner_id: str, tick: int,
) -> Optional[Branch]:
    """Return the sole active integration branch, creating at most one."""
    candidates = programbench_integration_branches(world)
    if candidates:
        if len(candidates) != 1:
            return None
        branch = candidates[0]
        if str(getattr(branch, "owner_id", "") or "") != owner_id:
            return None
        return branch
    if not branch_creation_capacity_available(
        world,
        linked_task=PROGRAMBENCH_INTEGRATION_TASK,
        _reserved_lane_capability=_PROGRAMBENCH_RESERVED_LANE_CAPABILITY,
    ):
        return None
    repo_system = getattr(world, "repo_system", None)
    if repo_system is None:
        return None
    branches = getattr(getattr(repo_system, "repo", None), "branches", None)
    if not isinstance(branches, dict):
        return None
    branches_before = dict(branches)
    sequence_before = getattr(repo_system, "_seq", None)

    def rollback_creation() -> None:
        branches.clear()
        branches.update(branches_before)
        if sequence_before is not None:
            repo_system._seq = sequence_before

    try:
        branch = repo_system.create_branch(owner_id, tick=tick)
    except Exception:  # noqa: BLE001 - fail-closed repository transaction
        rollback_creation()
        return None
    if branch is None:
        rollback_creation()
        return None
    branch_id = str(getattr(branch, "branch_id", "") or "")
    if (
        not branch_id
        or _branches(world).get(branch_id) is not branch
        or str(getattr(branch, "owner_id", "") or "") != owner_id
    ):
        rollback_creation()
        return None
    personal = (getattr(world, "personal", {}) or {}).get(owner_id)
    local_branches = getattr(personal, "local_branch_ids", None)
    if personal is not None and not isinstance(local_branches, list):
        rollback_creation()
        return None
    try:
        branch.linked_task = PROGRAMBENCH_INTEGRATION_TASK
        if local_branches is not None and branch_id not in local_branches:
            local_branches.append(branch_id)
    except Exception:  # noqa: BLE001 - fail-closed repository transaction
        rollback_creation()
        return None
    return branch


def _programbench_contract_branch(
    world: Any, owner_id: str, tick: int,
) -> Optional[Branch]:
    """Return the dedicated noncandidate contract lane.

    EXPLORE cannot drain ordinary branches.  The evidence-bound behavioral
    contract is nevertheless mandatory, so it gets one private lane in
    addition to the ordinary sixteen-slot coordination budget.  Only
    ``record_patch`` can reach this helper after validating the artifact kind;
    a caller-supplied ``linked_task`` cannot mint the exception.
    """

    candidates = [
        branch
        for branch in active_branches_org(world)
        if str(getattr(branch, "linked_task", "") or "")
        == PROGRAMBENCH_CONTRACT_TASK
    ]
    if candidates:
        if len(candidates) != 1:
            return None
        branch = candidates[0]
        return (
            branch
            if str(getattr(branch, "owner_id", "") or "") == owner_id
            else None
        )
    branches = active_branches_org(world)
    integration_exists = any(
        str(getattr(branch, "linked_task", "") or "")
        == PROGRAMBENCH_INTEGRATION_TASK
        for branch in branches
    )
    # Consume the contract's reserved slot while preserving the integration
    # slot if that mandatory lane has not yet been created. Total active work
    # never exceeds the native sixteen-branch organization bound.
    limit = MAX_ACTIVE_BRANCHES - int(not integration_exists)
    if len(branches) >= limit:
        return None
    repo_system = getattr(world, "repo_system", None)
    if repo_system is None:
        return None
    try:
        branch = repo_system.create_branch(owner_id, tick=tick)
    except Exception:  # noqa: BLE001 - fail closed before pending mutation
        return None
    branch.linked_task = PROGRAMBENCH_CONTRACT_TASK
    personal = (getattr(world, "personal", {}) or {}).get(owner_id)
    local = getattr(personal, "local_branch_ids", None)
    if isinstance(local, list) and branch.branch_id not in local:
        local.append(branch.branch_id)
    return branch


def record_patch(world: Any, agent_id: str, patch: Any, artifact: Any,
                 tick: int) -> Optional[str]:
    """Put an accepted patch on the branch its work item owns. Returns that id."""
    if "programbench_profile_state" in getattr(world, "__dict__", {}):
        try:
            from environments.org_env.programbench import (
                validate_programbench_profile_state_for_step,
            )

            validate_programbench_profile_state_for_step(world)
        except (ImportError, AttributeError, TypeError, ValueError):
            # A malformed/legacy claim is not a native routing context.  This
            # check precedes every branch or pending-ledger mutation and also
            # covers non-candidate docs which would otherwise fall through to
            # ordinary per-author routing.
            return None
    artifact_id = str(getattr(artifact, "artifact_id", "") or "")
    profile_active, integration_owner = _programbench_integration_owner(world)
    contract_routing = bool(
        profile_active
        and str(getattr(artifact, "programbench_artifact_kind", "") or "")
        == "behavioral_contract"
    )
    if profile_active:
        state = world.__dict__.get("programbench_profile_state") or {}
        probe_target = programbench_public_probe_artifact(world, artifact)
        if probe_target:
            try:
                from environments.org_env.backend.simulation.world import (
                    _consume_programbench_probe_patch_capability,
                )

                probe_capability = _consume_programbench_probe_patch_capability(
                    world, patch, artifact, agent_id
                )
            except (ImportError, AttributeError, TypeError, ValueError):
                world.__dict__.pop(
                    "_programbench_probe_patch_capability", None
                )
                probe_capability = False
            # Probe definitions used to be writable by the single designated
            # probe_owner. That made sense while a receipt had to contain
            # exactly the quota in one shot -- one author, one authoritative
            # corpus. Observations now accumulate across receipts and are
            # deduplicated by input digest, so several agents authoring
            # different batches is coherent, while the old rule concentrated
            # a 64-case floor on one agent and turned everyone else's capacity
            # into refusals: twenty of them in the first 36 ticks of a run,
            # with coverage frozen at 16.
            #
            # What the rule actually had to protect is attribution, and that is
            # kept: the patch must be authored by the agent presenting it, so
            # nobody writes the trusted surface on another's behalf.
            if str(getattr(patch, "actor_id", "") or "") != str(agent_id or ""):
                return None
            if not probe_capability:
                return None
        if (
            str(state.get("phase") or "") == "explore"
            and programbench_candidate_bearing_artifact(world, artifact)
            and not probe_target
        ):
            return None
    integration_routing = (
        profile_active and programbench_candidate_bearing_artifact(world, artifact)
    )
    if contract_routing:
        if integration_owner is None:
            return None
        branch = _programbench_contract_branch(
            world, integration_owner, tick
        )
    elif integration_routing:
        if integration_owner is None:
            return None
        branch = _programbench_integration_branch(world, integration_owner, tick)
    else:
        preferred_branch_id = str(
            getattr(patch, "delivery_branch_id", "") or ""
        )
        branch = _branches(world).get(preferred_branch_id) if preferred_branch_id else None
        if branch is not None and (
                str(getattr(branch, "owner_id", "") or "") != str(agent_id)
                or getattr(branch, "status", None) in _CLOSED):
            return None
        if preferred_branch_id and branch is None:
            return None
        if branch is None:
            branch = branch_already_touching(world, agent_id, artifact_id)
        if branch is None:
            keys = work_items_of(world, patch, artifact)
            open_keys = {str(getattr(b, "linked_task", "") or ""): b
                         for b in active_branches(world, agent_id)}
            # A file that belongs to several tasks joins the one already being worked
            # on, rather than whichever task its record happens to list first.
            chosen = next((k for k in keys if k in open_keys), keys[0])
            branch = branch_for(world, agent_id, chosen, tick)
    if branch is None:
        return None
    pending = world.__dict__.setdefault(PENDING, {}).setdefault(branch.branch_id, [])
    pending.append((getattr(patch, "patch_id", ""), getattr(artifact, "artifact_id", "")))
    branch.uncommitted_changes = len(pending)
    if integration_routing:
        # Pending accepted text is already part of the public candidate view.
        # Any green request for the older branch head is stale now, before a
        # later commit/sync can make the difference visible to RepoLiteSystem.
        _invalidate_live_pr_attestations(world, branch.branch_id)
    if getattr(branch, "status", None) == BranchStatus.CLEAN:
        branch.status = BranchStatus.DIRTY
    return branch.branch_id


def pending_on(world: Any, branch_id: str) -> List[Tuple[str, str]]:
    return list((world.__dict__.get(PENDING, {}) or {}).get(branch_id, []))


def clear_pending(world: Any, branch_id: str) -> None:
    (world.__dict__.setdefault(PENDING, {}))[branch_id] = []
    branch = _branches(world).get(branch_id)
    if branch is not None:
        branch.uncommitted_changes = 0


def drop_patch(world: Any, patch_id: str) -> None:
    """Forget a patch that was withdrawn before it was committed."""
    book = world.__dict__.get(PENDING, {}) or {}
    for bid, rows in book.items():
        kept = [row for row in rows if row[0] != patch_id]
        if len(kept) != len(rows):
            book[bid] = kept
            branch = _branches(world).get(bid)
            if branch is not None:
                branch.uncommitted_changes = len(kept)


def branches_with_pending(world: Any, agent_id: str) -> List[str]:
    """The agent's branches that have something to commit, readiest first.

    A branch whose request is red comes first: that is the repair, and it is
    already where the work is.
    """
    book = world.__dict__.get(PENDING, {}) or {}
    mine = [b.branch_id for b in active_branches(world, agent_id) if book.get(b.branch_id)]
    repo = getattr(getattr(world, "repo_system", None), "repo", None)
    red = {getattr(pr, "source_branch", "") for pr in
           (getattr(repo, "pull_requests", {}) or {}).values()
           if str(getattr(getattr(pr, "status", None), "value",
                          getattr(pr, "status", ""))) not in ("merged", "closed")
           and getattr(pr, "ci_run_ids", None) and not getattr(pr, "ci_passed", False)}
    return sorted(mine, key=lambda bid: (bid not in red, bid))


def pending_for_agent(world: Any, agent_id: str) -> List[Tuple[str, str]]:
    """Everything the agent has in hand, across its branches."""
    book = world.__dict__.get(PENDING, {}) or {}
    out: List[Tuple[str, str]] = []
    for b in active_branches(world, agent_id):
        out.extend(book.get(b.branch_id, []))
    return out


def branch_of_request(world: Any, pr: Any) -> Optional[Branch]:
    return _branches(world).get(str(getattr(pr, "source_branch", "") or ""))


__all__ = ["MAX_ACTIVE_BRANCHES", "PENDING", "PROGRAMBENCH_INTEGRATION_TASK",
           "PROGRAMBENCH_CONTRACT_TASK",
           "branch_creation_capacity_available",
           "active_branches",
           "active_branches_org",
           "branch_already_touching", "branch_for", "branch_of_request",
           "branches_with_pending", "clear_pending", "drop_patch",
           "pending_for_agent", "pending_on",
           "programbench_candidate_bearing_artifact",
           "programbench_integration_branches",
           "record_patch", "work_item_of",
           "work_items_of"]
