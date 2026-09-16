"""RepoLiteSystem (DESIGN env_org §31/§41). State transitions only, no real diff.

Tracks whether a PR was reviewed before merge so the review_before_merge protocol
detector (O-Infra-8) can flag unreviewed merges.
"""
from __future__ import annotations

from typing import List, Optional

from environments.org_env.backend.repo.repo import (
    Branch,
    BranchStatus,
    CIResult,
    Commit,
    CompanyRepo,
    PRStatus,
    ProductRelease,
    PullRequest,
    ReleaseCandidate,
)


class RepoLiteSystem:
    def __init__(self, repo: Optional[CompanyRepo] = None):
        self.repo = repo or CompanyRepo()
        self._seq = 0

    def _id(self, p: str) -> str:
        self._seq += 1
        return f"{p}_{self._seq}"

    def create_branch(self, owner_id: str, *, base: str = "main",
                      linked_task: Optional[str] = None, tick: int = 0) -> Branch:
        bid = self._id("branch")
        b = Branch(branch_id=bid, owner_id=owner_id, base_branch=base,
                   status=BranchStatus.CLEAN, linked_task=linked_task, last_sync_tick=tick)
        self.repo.branches[bid] = b
        return b

    def edit_file(self, agent_id: str, branch_id: str) -> bool:
        b = self.repo.branches.get(branch_id)
        if not b:
            return False
        b.uncommitted_changes += 1
        if b.status != BranchStatus.UNDER_REVIEW:
            b.status = BranchStatus.DIRTY
        return True

    def commit_changes(self, *, agent_id: str, branch_id: str, message: str,
                       changed_files: List[str], tick: int = 0,
                       quality_flags: Optional[List[str]] = None,
                       test_status: str = "unknown", risk_level: str = "low",
                       linked_task_id: Optional[str] = None,
                       patch_ids: Optional[List[str]] = None,
                       artifact_ids: Optional[List[str]] = None) -> Optional[Commit]:
        b = self.repo.branches.get(branch_id)
        if not b:
            return None
        cid = self._id("commit")
        c = Commit(commit_id=cid, author_id=agent_id, branch_id=branch_id, message=message,
                   changed_files=list(changed_files), timestamp=tick,
                   quality_flags=list(quality_flags or []), test_status=test_status,
                   risk_level=risk_level, linked_task_id=linked_task_id or b.linked_task,
                   patch_ids=list(patch_ids or []), artifact_ids=list(artifact_ids or []),
                   status="local")
        self.repo.commits[cid] = c
        b.commit_ids.append(cid)
        b.uncommitted_changes = 0
        # a follow-up commit on a branch already under review stays attached to its
        # open PR (CI resyncs it in) instead of re-entering the ready-for-PR pool,
        # which would spawn a duplicate PR carrying the same commits.
        if b.status != BranchStatus.UNDER_REVIEW:
            b.status = BranchStatus.READY_FOR_PR
        # A CI verdict describes one branch head. A follow-up commit on an open
        # request invalidates that verdict immediately; otherwise merge_pr can
        # land the branch's new commit while the PR/apply layer still carries
        # only the old commit list and old green result.
        for pr in self.repo.pull_requests.values():
            if pr.source_branch != branch_id:
                continue
            if pr.status in {PRStatus.MERGED, PRStatus.CLOSED}:
                continue
            pr.ci_passed = False
            pr.test_status = "unknown"
            pr.__dict__.pop("ci_tree_hash", None)
            pr.ci_base_main_commit_ids = None
        return c

    def open_pr(self, *, agent_id: str, source_branch: str, target_branch: str = "main",
                reviewers: Optional[List[str]] = None, linked_task: Optional[str] = None) -> PullRequest:
        pid = self._id("pr")
        revs = list(reviewers or [])
        pr = PullRequest(pr_id=pid, author_id=agent_id, source_branch=source_branch,
                         target_branch=target_branch, reviewers=revs, linked_task=linked_task,
                         status=PRStatus.REVIEW_REQUESTED if revs else PRStatus.OPEN)
        self.repo.pull_requests[pid] = pr
        b = self.repo.branches.get(source_branch)
        if b:
            pr.commit_ids = list(b.commit_ids)          # carry the branch's commits + patches
            for cid in b.commit_ids:
                c = self.repo.commits.get(cid)
                if c:
                    pr.patch_ids.extend(c.patch_ids)
                    c.status = "included_in_pr"
            b.status = BranchStatus.UNDER_REVIEW
        return pr

    def _sync_pr_commits(self, pr: PullRequest) -> None:
        """Bring commits pushed to the source branch after ``open_pr`` into the PR —
        CI evaluates the branch's CURRENT state (like real CI running on the branch
        head), a follow-up commit can repair a gate failure, and the PR's carried
        patches and work-item links match what ``merge_pr`` actually lands on the
        mainline."""
        b = self.repo.branches.get(pr.source_branch)
        if not b:
            return
        for cid in b.commit_ids:
            if cid in pr.commit_ids:
                continue
            pr.commit_ids.append(cid)
            c = self.repo.commits.get(cid)
            if c:
                pr.patch_ids.extend(c.patch_ids)
                for task_id in (
                    c.linked_task_ids
                    or ([c.linked_task_id] if c.linked_task_id else [])
                ):
                    if task_id and task_id not in pr.linked_task_ids:
                        pr.linked_task_ids.append(task_id)
                for issue_id in c.linked_issue_ids:
                    if issue_id and issue_id not in pr.linked_issue_ids:
                        pr.linked_issue_ids.append(issue_id)
                if not pr.linked_task and pr.linked_task_ids:
                    pr.linked_task = pr.linked_task_ids[0]
                if not pr.linked_issue and pr.linked_issue_ids:
                    pr.linked_issue = pr.linked_issue_ids[0]
                c.status = "included_in_pr"

    def sync_pr_commits(self, pr_id: str) -> bool:
        """Synchronize one live request; explicit public seam for attestations."""

        pr = self.repo.pull_requests.get(pr_id)
        if pr is None or pr.status in {
            PRStatus.MERGED,
            PRStatus.CLOSED,
            PRStatus.STALE,
        }:
            return False
        self._sync_pr_commits(pr)
        return True

    def run_ci(self, *, pr_id: str, tick: int = 0) -> Optional[CIResult]:
        """Lightweight CI: build + tests pass unless the PR's HEAD commit is a
        high-risk change without passing tests. The gate judges the head only, so
        a later qualifying commit (author grew test_writing, wrote the tests)
        repairs a previously blocked PR instead of the flag pinning it forever.
        Lenient by design so a normal reviewed PR can merge; high-risk gets blocked."""
        pr = self.repo.pull_requests.get(pr_id)
        if not pr:
            return None
        if pr.status in {PRStatus.MERGED, PRStatus.CLOSED, PRStatus.STALE}:
            return None
        self._sync_pr_commits(pr)
        ci_id = self._id("ci")
        fails: List[str] = []
        head = self.repo.commits.get(pr.commit_ids[-1]) if pr.commit_ids else None
        if head is not None and head.risk_level == "high" and (
                "missing_tests" in head.quality_flags
                or head.test_status not in ("pass", "passed")):
            fails.append(f"{head.commit_id}: high-risk change without passing tests")
        status = "failed" if fails else "passed"
        ci = CIResult(ci_id=ci_id, pr_id=pr_id, commit_id=(pr.commit_ids[-1] if pr.commit_ids else None),
                      created_at_tick=tick, status=status,
                      checks=[{"name": "build", "ok": True}, {"name": "tests", "ok": not fails}],
                      failure_reasons=fails)
        self.repo.ci_runs[ci_id] = ci
        pr.ci_run_ids.append(ci_id)
        pr.ci_passed = status == "passed"
        pr.test_status = "passed" if pr.ci_passed else "failed"
        # The merge candidate was judged against this exact mainline. Another
        # PR landing invalidates the verdict even when this branch head did not
        # move; create/create conflicts are one concrete reason, and ordinary
        # integration drift is another.
        pr.ci_base_main_commit_ids = tuple(self.repo.main_commit_ids)
        return ci

    def review_pr(self, *, reviewer_id: str, pr_id: str, approve: bool, comment: str = "",
                  tick: int = 0) -> bool:
        pr = self.repo.pull_requests.get(pr_id)
        if not pr:
            return False
        pr.reviewed = True
        pr.review_comments.append({
            "review_id": f"review_{len(pr.review_comments) + 1}",
            "reviewer": reviewer_id,
            "approve": approve,
            "comment": comment,
            "tick": tick,
        })
        if approve:
            if reviewer_id not in pr.approved_by:
                pr.approved_by.append(reviewer_id)
            pr.status = PRStatus.APPROVED
            pr.approved_tick = tick                  # v6 P0.4: start the merge-latency clock
        else:
            pr.requested_changes.append(comment or "changes requested")
            pr.status = PRStatus.CHANGES_REQUESTED
        return True

    def request_changes(self, *, reviewer_id: str, pr_id: str, comment: str = "", tick: int = 0) -> bool:
        return self.review_pr(reviewer_id=reviewer_id, pr_id=pr_id, approve=False, comment=comment, tick=tick)

    def approve_pr(self, *, reviewer_id: str, pr_id: str, tick: int = 0,
                   comment: str = "") -> bool:
        return self.review_pr(reviewer_id=reviewer_id, pr_id=pr_id, approve=True,
                              comment=comment, tick=tick)

    def merge_pr(self, *, pr_id: str, tick: int = 0, force: bool = False) -> bool:
        """Merge a PR into its target. Requires an APPROVED review AND a passing CI run
        (v5 gate). ``force`` allows merging an unreviewed/un-CI'd PR — a protocol
        violation the detector can catch (§47/§59)."""
        pr = self.repo.pull_requests.get(pr_id)
        if not pr:
            return False
        if pr.status in {PRStatus.MERGED, PRStatus.CLOSED, PRStatus.STALE}:
            return False
        src = self.repo.branches.get(pr.source_branch)
        metadata_only = not pr.commit_ids and not pr.patch_ids
        commits = []
        if src is None or not src.commit_ids:
            # Preserve the long-standing metadata-only PR used by governance
            # experiments. A request that claims any delivery payload, however,
            # must have a real source branch and can never promote without it.
            if not metadata_only:
                return False
        else:
            # Synchronize before checking the gate so a late branch commit cannot
            # be marked merged yet disappear from pr.commit_ids/apply_merged_pr.
            self._sync_pr_commits(pr)
            commits = [self.repo.commits.get(cid) for cid in src.commit_ids]
            if any(
                commit is None or commit.branch_id != src.branch_id
                for commit in commits
            ):
                return False
            expected_patch_ids = [
                patch_id
                for commit in commits
                for patch_id in commit.patch_ids
            ]
            if pr.commit_ids != list(src.commit_ids) or pr.patch_ids != expected_patch_ids:
                return False
        if not force:
            if not (pr.status == PRStatus.APPROVED and pr.ci_passed):
                return False
            head = src.commit_ids[-1] if src and src.commit_ids else None
            if head is not None:
                latest_ci = next(
                    (
                        self.repo.ci_runs.get(ci_id)
                        for ci_id in reversed(pr.ci_run_ids)
                        if self.repo.ci_runs.get(ci_id) is not None
                    ),
                    None,
                )
                if (
                    latest_ci is None
                    or latest_ci.commit_id != head
                    or latest_ci.status != "passed"
                ):
                    return False
                if (
                    getattr(pr, "ci_base_main_commit_ids", None) is None
                    or tuple(pr.ci_base_main_commit_ids)
                    != tuple(self.repo.main_commit_ids)
                ):
                    pr.ci_passed = False
                    pr.test_status = "unknown"
                    return False
        if src and src.commit_ids:
            self.repo.main_commit_ids.extend(src.commit_ids)
            for commit in commits:
                commit.status = "merged"
            src.status = BranchStatus.MERGED
        pr.status = PRStatus.MERGED
        pr.merged_tick = tick
        return True

    def merged_commit_patches(self, pr) -> List[tuple]:
        """(patch_id, artifact_id) pairs carried by a merged PR's commits — so the
        caller can apply them to the mainline product artifact."""
        out: List[tuple] = []
        for cid in pr.commit_ids:
            c = self.repo.commits.get(cid)
            if not c:
                continue
            for i, pid in enumerate(c.patch_ids):
                aid = c.artifact_ids[i] if i < len(c.artifact_ids) else (
                    c.artifact_ids[0] if c.artifact_ids else None)
                out.append((pid, aid))
        return out

    # -- release lifecycle (v5 §3.7-§3.8/§4.2) -----------------------------
    def _bump_version(self) -> str:
        parts = (self.repo.current_version or "0.0.0").split(".")
        try:
            parts[-1] = str(int(parts[-1]) + 1)
        except Exception:
            parts = ["0", "0", "1"]
        return ".".join(parts)

    def create_release_candidate(self, *, created_by: str, tick: int = 0,
                                 required_gates: Optional[List[str]] = None) -> ReleaseCandidate:
        used = set()
        for rc in self.repo.release_candidates.values():
            if rc.status == "released":
                used.update(rc.included_pr_ids)
        merged = [pr for pr in self.repo.pull_requests.values()
                  if pr.status == PRStatus.MERGED and pr.pr_id not in used]
        commit_ids = [c for pr in merged for c in pr.commit_ids]
        artifact_ids = sorted({a for cid in commit_ids
                               for a in (self.repo.commits[cid].artifact_ids
                                         if cid in self.repo.commits else [])})
        cid = self._id("rc")
        task_ids = sorted({t for pr in merged for t in (getattr(pr, "linked_task_ids", []) or [])})
        rc = ReleaseCandidate(
            candidate_id=cid, version=self._bump_version(), created_by=created_by, created_at_tick=tick,
            included_pr_ids=[pr.pr_id for pr in merged], included_commit_ids=commit_ids,
            included_artifact_ids=artifact_ids, included_task_ids=task_ids,
            required_gates=list(required_gates or []), status="draft")
        self.repo.release_candidates[cid] = rc
        return rc

    def refresh_candidate(self, rc, *, tick: int = 0) -> bool:
        """v8d P0a: re-collect the merged, not-yet-released PR set into an existing draft/
        blocked candidate so it reflects later merges instead of freezing at creation tick.
        Returns True if the included set changed."""
        used = set()
        for other in self.repo.release_candidates.values():
            if other.status == "released":
                used.update(other.included_pr_ids)
        merged = [pr for pr in self.repo.pull_requests.values()
                  if pr.status == PRStatus.MERGED and pr.pr_id not in used]
        pr_ids = [pr.pr_id for pr in merged]
        if set(pr_ids) == set(rc.included_pr_ids):
            return False
        rc.included_pr_ids = pr_ids
        rc.included_commit_ids = [c for pr in merged for c in pr.commit_ids]
        rc.included_artifact_ids = sorted({a for cid in rc.included_commit_ids
                                           for a in (self.repo.commits[cid].artifact_ids
                                                     if cid in self.repo.commits else [])})
        rc.included_task_ids = sorted({t for pr in merged
                                       for t in (getattr(pr, "linked_task_ids", []) or [])})
        rc.last_updated_tick = tick
        return True

    def approve_release_candidate(self, *, rc_id: str, approver: str) -> bool:
        rc = self.repo.release_candidates.get(rc_id)
        if not rc or rc.status not in ("under_review", "draft"):
            return False
        if approver not in rc.approvals:
            rc.approvals.append(approver)
        return True

    def publish_release(self, *, rc_id: str, released_by: str, tick: int = 0,
                        public_summary: str = "", known_limitations: Optional[List[str]] = None) -> Optional[ProductRelease]:
        rc = self.repo.release_candidates.get(rc_id)
        if not rc or rc.status != "approved":
            return None
        rid = self._id("release")
        tag = f"v{rc.version}"
        rel = ProductRelease(
            release_id=rid, version=rc.version, candidate_id=rc_id, released_by=released_by,
            released_at_tick=tick, release_tag=tag, public_summary=public_summary,
            included_artifacts=list(rc.included_artifact_ids),
            included_pr_ids=list(rc.included_pr_ids),
            included_tasks=list(rc.included_task_ids),
            known_limitations=list(known_limitations or []))
        self.repo.releases[rid] = rel
        self.repo.release_tags.append(tag)
        self.repo.current_version = rc.version
        rc.status = "released"
        return rel

    # -- detector helpers (O-Infra-8) --------------------------------------
    def unreviewed_merged_prs(self) -> List[PullRequest]:
        return [pr for pr in self.repo.pull_requests.values()
                if pr.status == PRStatus.MERGED and not pr.reviewed]


__all__ = ["RepoLiteSystem"]
