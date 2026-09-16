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

from typing import Any, Dict, List, Optional, Tuple

from environments.org_env.backend.repo.repo import Branch, BranchStatus

PENDING = "_pending_by_branch"
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


def branch_for(world: Any, agent_id: str, key: str, tick: int) -> Optional[Branch]:
    """The agent's branch for this work item, made if it has none.

    Returns None only when the agent is already carrying as much as it may and
    this would be one more thing -- and never for work it is already carrying,
    so a limit cannot strand a change that is already made.
    """
    for b in active_branches(world, agent_id):
        if str(getattr(b, "linked_task", "") or "") == key:
            return b
    if len(active_branches_org(world)) >= MAX_ACTIVE_BRANCHES:
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


def record_patch(world: Any, agent_id: str, patch: Any, artifact: Any,
                 tick: int) -> Optional[str]:
    """Put an accepted patch on the branch its work item owns. Returns that id."""
    artifact_id = str(getattr(artifact, "artifact_id", "") or "")
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


__all__ = ["MAX_ACTIVE_BRANCHES", "PENDING", "active_branches",
           "active_branches_org",
           "branch_already_touching", "branch_for", "branch_of_request",
           "branches_with_pending", "clear_pending", "drop_patch",
           "pending_for_agent", "pending_on", "record_patch", "work_item_of",
           "work_items_of"]
