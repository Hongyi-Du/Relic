"""PatchValidator (v4 §2.6) — decides whether an execution-layer patch may enter the
world. Rejects empty patches, patches that invent completed capabilities, patches that
don't address their edit goal, and near-duplicate re-patches of a just-touched artifact.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

from environments.org_env.product.patch_objects import CodePatch, DocPatch

# claims a simulated patch must never make (it can't have "fully built" anything)
_FAKE_CAPABILITY = (
    "fully implemented", "production ready", "production-ready", "fully working",
    "completely solved", "100%", "guaranteed", "all claims verified", "fully tested",
    "fully automated", "complete solution", "now fully",
)
# v6 P0.1: a new patch that re-states a recent ACCEPTED patch on the same artifact is a
# semantic duplicate (the v5 README softening loop). Look back over a wider window and
# several prior patches, and compare the whole patch (goal + summary + fields/checks),
# not just the single last change_summary string.
_DUP_WINDOW = 10
_DUP_LOOKBACK = 5


@dataclass
class ValidationResult:
    passed: bool
    reason: str = ""


_COMMENT_PREFIXES = ("#", "//", "<!--", "--", ";")

# Rejections that describe the INFRASTRUCTURE, not the edit. A patch refused for
# one of these reasons says nothing about whether editing its target is a good
# idea - the model simply never answered - so re-selection suppression, the
# recent-rejection window and the all-time dead-end counter must all skip them.
# Measured: two such rejections within 12 ticks masked every agent off
# query_parsing.py, the exact file two failing oracles needed edited.
INFRASTRUCTURE_REJECTION_REASONS = (
    "code_editor_llm_returned_nothing",
    # The file is larger than the editor will show in full, so no attempt was
    # made at all. Like an unanswered call, this says nothing about whether
    # editing the file is a good idea.
    "file_exceeds_edit_budget",
)


def is_infrastructure_rejection(patch: Any) -> bool:
    return str(getattr(patch, "rejection_reason", "") or "") in INFRASTRUCTURE_REJECTION_REASONS


def _adds_no_executable_line(patch: CodePatch, art: Any) -> Optional[str]:
    """Reason a code patch changes no behaviour, or None.

    Compares the patched content against the artifact's current content and asks
    whether anything but comments and blank lines differs. A patch is free to
    add comments; one that adds NOTHING ELSE has not edited the program, and
    accepting it spends one of the issue's limited attempts and leaves noise
    behind.

    Files without a comment convention we recognise are left alone: guessing
    wrong there would reject real edits, and a false rejection costs more than a
    missed no-op.
    """
    current = str(getattr(art, "content", "") or "")
    new = str(getattr(patch, "new_content", "") or "")
    if not new or not current:
        return None
    path = str(getattr(art, "linked_file_path", "") or "")
    if not path.endswith((".py", ".js", ".ts", ".go", ".rs", ".java", ".c", ".h", ".cpp")):
        return None

    def significant(text: str) -> list:
        out = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith(_COMMENT_PREFIXES):
                continue
            out.append(stripped)
        return out

    if significant(new) != significant(current):
        return None
    return "code patch adds no executable change"


def _has_fake_capability(*texts: str) -> Optional[str]:
    blob = " ".join(t for t in texts if t).lower()
    for phrase in _FAKE_CAPABILITY:
        if phrase in blob:
            return phrase
    return None


class PatchValidator:
    def _artifact(self, world: Any, oid: str):
        return (getattr(world, "product_artifacts", {}) or {}).get(oid)

    def _reconcile_resolved_gaps(self, patch, art) -> None:
        """v4 §4: a patch may only claim to resolve gaps the artifact actually has, so
        ``apply_product_patch`` removes exactly those (not a blind pop of gap #0)."""
        gaps = list(getattr(art, "known_gaps", []) or [])
        claimed = [g for g in (getattr(patch, "resolved_gaps", None) or []) if g in gaps]
        patch.resolved_gaps = claimed

    @staticmethod
    def _patch_texts(patch) -> List[str]:
        """The topical surface of a patch — what it changed and why."""
        return [
            getattr(patch, "edit_goal", "") or "",
            getattr(patch, "change_summary", "") or "",
            " ".join(getattr(patch, "added_fields", []) or []),
            " ".join(getattr(patch, "added_checks", []) or []),
            " ".join(getattr(patch, "removed_overclaims", []) or []),
            " ".join(getattr(patch, "added_limitations", []) or []),
            " ".join(s.get("after", "") for s in (getattr(patch, "changed_sections", []) or [])),
        ]

    def _dup_reason(self, world: Any, art, patch) -> Optional[str]:
        """Real-content era: a patch is a duplicate only if it produces NO change to the
        file, or reproduces a recent patch's exact resulting content. (The old text-similarity
        dedup rejected ~60% of edits as 'semantic duplicates' even when the result differed —
        that path is kept only for legacy symbolic patches with no new_content.)"""
        if art is None:
            return None
        new_content = (getattr(patch, "new_content", "") or "").strip()
        if new_content:
            if new_content == (getattr(art, "content", "") or "").strip():
                return "duplicate (no-op): patch produces no change to the file"
            patches = getattr(world, "patches", {}) or {}
            for pid in list(reversed(getattr(art, "patch_history_ids", []) or []))[:_DUP_LOOKBACK]:
                pp = patches.get(pid)
                if pp is None or getattr(pp, "patch_id", None) == patch.patch_id:
                    continue
                if (getattr(pp, "new_content", "") or "").strip() == new_content:
                    return "duplicate: reproduces a recent patch's exact result"
            return None
        if self._legacy_semantic_dup(world, art, patch):
            return "semantic duplicate of a recent patch on this artifact"
        return None

    def _legacy_semantic_dup(self, world: Any, art, patch) -> bool:
        """v6 P0.1 (legacy, symbolic-patch only): reject a patch that re-states a recent
        accepted patch on the same artifact via token-overlap / LLM equivalence."""
        if art is None:
            return False
        patches = getattr(world, "patches", {}) or {}
        tick = int(getattr(patch, "tick", 0) or 0)
        prior = []
        for pid in reversed(getattr(art, "patch_history_ids", []) or []):
            pp = patches.get(pid)
            if pp is None or getattr(pp, "patch_id", None) == patch.patch_id:
                continue
            if tick - int(getattr(pp, "tick", 0) or 0) > _DUP_WINDOW:
                break
            prior.append(pp)
            if len(prior) >= _DUP_LOOKBACK:
                break
        if not prior:
            return False
        from environments.org_env.llm.semantic_dedup import equivalent, normalize
        a_texts = self._patch_texts(patch)
        new_cs = normalize(getattr(patch, "change_summary", "") or "")
        client = getattr(world, "llm_client", None)
        kind = "code change" if isinstance(patch, CodePatch) else "document change"
        for pp in prior:
            # deterministic fast-path: an identical change_summary IS the same change
            # (this is the v5 README/claim_tracker softening loop — caught with no LLM).
            if new_cs and new_cs == normalize(getattr(pp, "change_summary", "") or ""):
                return True
            if equivalent(world, a_texts, self._patch_texts(pp), kind=kind, client=client):
                return True
        return False

    # -- doc ----------------------------------------------------------------
    def validate_doc_patch(self, patch: DocPatch, world: Any) -> ValidationResult:
        art = self._artifact(world, patch.target_object_id)
        if art is None:
            return ValidationResult(False, f"unknown artifact {patch.target_object_id}")
        if patch.is_empty():
            return ValidationResult(False, "empty doc patch")
        if not patch.change_summary:
            return ValidationResult(False, "missing change_summary")
        fake = _has_fake_capability(patch.change_summary, *(s.get("after", "")
                                                            for s in patch.changed_sections))
        if fake:
            return ValidationResult(False, f"invents completed capability: '{fake}'")
        dup = self._dup_reason(world, art, patch)
        if dup:
            return ValidationResult(False, dup)
        self._reconcile_resolved_gaps(patch, art)
        # README-style overclaim edits must actually soften/limit something. When the patch
        # carries real new_content the document IS the edit, so trust it instead of the
        # structured fields (the LLM may soften in prose without filling removed_overclaims).
        gl = (patch.edit_goal + " " + patch.change_summary).lower()
        if not (getattr(patch, "new_content", "") or "").strip():
            if ("readme" in patch.target_object_id.lower() or "overclaim" in gl or "overpromise" in gl):
                if not (patch.removed_overclaims or patch.added_limitations):
                    return ValidationResult(False, "overclaim edit must remove/soften a claim or add a limitation")
        return ValidationResult(True)

    # -- code ---------------------------------------------------------------
    def validate_code_patch(self, patch: CodePatch, world: Any) -> ValidationResult:
        art = self._artifact(world, patch.target_object_id)
        if art is None:
            return ValidationResult(False, f"unknown artifact {patch.target_object_id}")
        if patch.is_empty():
            return ValidationResult(False, "empty code patch")
        if getattr(patch, "llm_declined", False):
            # A model was attached and produced nothing usable, so the content
            # here is the no-LLM template - comment lines with a generic
            # summary. Accepting it spends one of the issue's limited attempts
            # on a patch that changed no behaviour.
            #
            # The reason separates two different facts. An unanswered call is
            # infrastructure and says nothing about the file. Anchors that
            # missed five times running is this model failing to quote a file
            # it was shown, which is a fact about the attempt.
            return ValidationResult(
                False,
                getattr(patch, "decline_reason", "")
                or "code_editor_llm_returned_nothing",
            )
        if getattr(world, "llm_client", None) is not None:
            # Backstop for LLM runs, independent of where the content came from:
            # a code patch that adds no executable line is not a code patch.
            # Measured: six such patches were ACCEPTED across one run, on six
            # different source files, each leaving comment noise and burning an
            # attempt.
            #
            # Gated on a client being attached because in no-LLM mode the
            # comment-appending template IS the mechanism, not a degradation -
            # rejecting it there removes the rule-based path's only way to make
            # an inspectable change.
            no_op = _adds_no_executable_line(patch, art)
            if no_op:
                return ValidationResult(False, no_op)
        if not patch.change_summary:
            return ValidationResult(False, "missing change_summary")
        fake = _has_fake_capability(patch.change_summary, patch.pseudo_diff,
                                    *patch.changed_behavior)
        if fake:
            return ValidationResult(False, f"invents completed capability: '{fake}'")
        dup = self._dup_reason(world, art, patch)
        if dup:
            return ValidationResult(False, dup)
        # materialization: a .py file must stay valid, importable Python AFTER the patch, so a
        # syntactically-broken edit is rejected at patch time (not just caught later by the
        # release smoke gate). Keeps the working tree + mainline always runnable.
        fp = getattr(art, "linked_file_path", "") or ""
        nc = getattr(patch, "new_content", "") or ""
        if fp.endswith(".py") and nc.strip():
            try:
                compile(nc, fp, "exec")
            except SyntaxError as e:
                return ValidationResult(False, f"patch produces invalid Python ({e.msg} @ line {e.lineno})")
            # NOTE (option A): we deliberately DO NOT hard-reject a patch that drops a cross-file public
            # symbol — a legitimate refactor sometimes must remove/rename it. Instead the removal is
            # ALLOWED and every importer is flagged for a coordinated "update references" edit
            # (execution._flag_interface_repairs + _interface_repair_driven), IDE-style.
        self._reconcile_resolved_gaps(patch, art)
        # claim_tracker edits must add evidence enforcement — check the real content too.
        if "claim_tracker" in patch.target_object_id:
            content = (getattr(patch, "new_content", "") or "").lower()
            ok = (any("source" in f.lower() for f in patch.added_fields)
                  or any("uncertainty" in f.lower() for f in patch.added_fields)
                  or any("evidence" in c.lower() or "source" in c.lower() for c in patch.added_checks)
                  or ("source" in content or "uncertainty" in content or "evidence" in content))
            if not ok:
                return ValidationResult(False, "claim_tracker patch must add source_ids/uncertainty/evidence check")
        return ValidationResult(True)


__all__ = ["PatchValidator", "ValidationResult"]
