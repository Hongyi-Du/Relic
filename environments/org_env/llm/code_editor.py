"""CodeEditorLLM (v4 §2.5) — the execution-layer LLM that turns an already-decided code
action into a structured CodePatch (pseudo-diff, added fields/checks, limitations). Falls
back to a deterministic, artifact-specific template patch when no client is available, so
code edits always produce an inspectable change rather than a bare revision bump.
"""
from __future__ import annotations

import os
from typing import Any, List, Optional

from environments.org_env.llm.anchored_edits import apply_anchored_edits
from environments.org_env.llm.client import UNCAPPED_OUTPUT, LLMError, OrgLLMClient
from environments.org_env.product.patch_objects import CodePatch
from environments.org_env.proposals.objects import ensure_list

CODE_EDITOR_SYSTEM = (
    "You are editing a real, runnable product code file in a simulated startup. You are shown "
    "the WHOLE file; you return only the PARTS YOU CHANGE, as a list of `edits`. Each edit is "
    "{\"search\": <text copied EXACTLY from the file>, \"replace\": <what it becomes>}. "
    "HARD REQUIREMENTS: (1) every `search` must be copied character-for-character from the file "
    "shown to you, including indentation, and must appear EXACTLY ONCE — include enough "
    "surrounding lines to make it unique; (2) the file MUST stay VALID, importable Python that "
    "still works with its existing callers and signatures; (3) actually IMPLEMENT a concrete step "
    "toward the edit goal — real logic (e.g. compute a credibility score, enforce that a claim "
    "links at least one source_id, add a metric) — do NOT merely add a comment, rename, or "
    "restate the gap; (4) do NOT use triple-quoted strings (\"\"\" or ''') ANYWHERE — use `#` line "
    "comments for any description, so the text round-trips safely through JSON. "
    "TO ADD NEW CODE: anchor on the lines your addition goes next to and repeat them in "
    "`replace` along with the new code — e.g. to add a function after an existing one, `search` "
    "that function's last line and `replace` it with that same line plus your new function. "
    "TO DELETE CODE: give an empty `replace`. "
    "IF THE FILE IS EMPTY there is nothing to quote: send one edit with an empty `search`, and "
    "`replace` is the whole new file. "
    "Keep it honest (do not claim it is fully done). "
    "Also fill change_summary + added_fields/added_checks. Return JSON only."
)

CODE_EDITOR_SCHEMA = {
    "patch_type": "code_patch",
    "edit_goal": "string",
    "edits": [{"search": "string", "replace": "string"}],
    "pseudo_diff": "string",
    "change_summary": "string",
    "added_fields": ["string"],
    "added_checks": ["string"],
    "changed_behavior": ["string"],
    "known_limitations": ["string"],
    "related_issue_ids": ["string"],
    "related_task_ids": ["string"],
}

# How many times the model may re-aim an edit whose anchor did not match. An
# anchor either resolves to exactly one place in the file or it does not, and
# when it does not the model is told which one failed and why, so a retry is a
# correction rather than a re-roll.
MAX_EDIT_ATTEMPTS = 5


def _render_code_change(current: str, data: dict) -> str:
    """Deterministically grow the real file text from a symbolic patch (no-LLM mode).
    IDEMPOTENT: only appends lines not already present, so re-applying the same template
    change yields no change (a no-op the validator rejects) — anti-churn without an LLM."""
    cur = current.rstrip("\n")
    # Append as COMMENTS so a no-LLM (template) edit never breaks the file's Python syntax —
    # the materialized repo must stay runnable (smoke rc0). A real LLM returns full valid
    # new_content instead and bypasses this path.
    cand: List[str] = []
    for ln in (data.get("pseudo_diff") or "").splitlines():
        s = ln.strip()
        if s.startswith("+"):
            cand.append("# + " + s[1:].strip())
        elif s.startswith("-"):
            cand.append("# - " + s[1:].strip())
        elif s:
            cand.append("# " + s)
    for f in data.get("added_fields") or []:
        cand.append(f"# field: {f}")
    for c in data.get("added_checks") or []:
        cand.append(f"# check: {c}")
    new_lines = [l for l in cand if l and l not in cur]
    if not new_lines:
        return (cur + "\n") if cur else ""
    summ = data.get("change_summary") or data.get("edit_goal") or "update"
    body = (cur + "\n\n" if cur else "") + "\n".join([f"# --- change: {summ} ---"] + new_lines)
    return body.rstrip("\n") + "\n"

# artifact-specific deterministic templates (grounded in each tool's real gap)
_TEMPLATES = {
    "art_tools_claim_tracker_py": {
        "patch_type": "schema_change",
        "pseudo_diff": "+ Claim.source_ids: list[str]\n+ Claim.uncertainty_note: str\n+ validate_claim_evidence(claim)",
        "change_summary": "Added source_ids + uncertainty_note fields and an evidence-validation check to Claim.",
        "added_fields": ["source_ids", "uncertainty_note"],
        "added_checks": ["validate_claim_evidence requires at least one source id"],
        "changed_behavior": ["claims without evidence are marked incomplete"],
        "known_limitations": ["does not yet score source credibility"],
    },
    "art_tools_source_tracker_py": {
        "patch_type": "schema_change",
        "pseudo_diff": "+ Source.credibility_score: float\n+ score_source_credibility(source)",
        "change_summary": "Added a credibility_score field and a scoring method to Source.",
        "added_fields": ["credibility_score"],
        "added_checks": ["score_source_credibility classifies sources into tiers"],
        "changed_behavior": ["low-credibility sources are flagged"],
        "known_limitations": ["scoring heuristic is coarse"],
    },
    "art_tools_report_writer_py": {
        "patch_type": "validator_change",
        "pseudo_diff": "+ require_supported_claims(report)\n+ block_unsupported_sentences()",
        "change_summary": "Report writer now requires each claim sentence to map to a tracked, sourced claim.",
        "added_fields": [],
        "added_checks": ["require_supported_claims rejects unsourced sentences"],
        "changed_behavior": ["unsupported sentences are flagged before export"],
        "known_limitations": ["does not detect subtle paraphrase overclaims"],
    },
    "art_eval_eval_stub_py": {
        "patch_type": "stub_update",
        "pseudo_diff": "+ METRICS = {grounding_rate, evidence_coverage, contradiction_rate}\n+ run_eval(report) -> MetricResult",
        "change_summary": "Defined concrete evaluation metrics (grounding_rate, evidence_coverage, contradiction_rate) beyond placeholders.",
        "added_fields": ["grounding_rate", "evidence_coverage", "contradiction_rate"],
        "added_checks": ["run_eval computes each metric on a report"],
        "changed_behavior": ["eval produces numeric metrics instead of placeholders"],
        "known_limitations": ["metrics are not yet validated against human judgement"],
    },
}


# The file is shown in full because anchors must be copied from text the model
# actually saw. It is no longer written back in full — edits are anchored
# search/replace — so this bounds INPUT only, and the old value bounded an
# output that no longer exists.
#
# The old 48k was calibrated on "the largest file in a real starter repo
# measured 15.6k characters". Measured across the twelve packs in the matrix,
# that is wrong by a factor of six: ten of them contain source above 48k, and
# it is core module code, not vendored blobs -- anyio's asyncio backend at 99k,
# celery's canvas at 97k, black's trans at 95k, aiohttp's connector at 69k. On
# boltons the ceiling made two of the twelve seeded issues unfixable by
# construction: both name iterutils.py at 56k, and one run opened 62 patches
# against them, every one declined before the model was asked anything.
#
# What is left is a physical bound, not a judgement about how much an
# organization should attempt: a file has to fit the model's context window.
# Unlike the output cap and the shrink floor, removing it does not hand the
# decision back to the organization — it hands back a provider error, retried
# five times, recorded as an LLM failure. black ships generated profiling
# fixtures from 160k to 1.4M; the largest is roughly 350k tokens of input.
#
# So it is an operator setting rather than a constant. 0 removes the ceiling
# for a model that can take it, matching UNCAPPED_OUTPUT in client.py.
UNCAPPED_EDIT_INPUT = 0
MAX_EDIT_CONTENT_CHARS = int(
    os.environ.get("ORG_LLM_MAX_EDIT_CONTENT_CHARS", "200000") or 200_000
)


def _edit_input_ceiling() -> int | None:
    """The ceiling in force, or None when the operator has removed it."""
    return None if MAX_EDIT_CONTENT_CHARS == UNCAPPED_EDIT_INPUT else MAX_EDIT_CONTENT_CHARS


def _record_skip(world: Any, target_object_id: str, reason: str, detail: str = "") -> None:
    """Record an edit the editor declined, so it is not read as an LLM failure.

    Declining leaves the caller falling back to the template path, which appends
    comment lines. That is a legitimate no-LLM mode but a poor explanation for a
    patch that did nothing, so the real reason is kept where diagnostics can
    find it.
    """
    try:
        skipped = world.__dict__.setdefault("_code_editor_skipped", [])
        skipped.append({"target_object_id": str(target_object_id),
                        "reason": str(reason), "detail": str(detail)[:200]})
        del skipped[:-200]
    except Exception:
        pass


def _last_skip_reason(world: Any, target_object_id: str) -> str:
    """The most recent reason the editor declined this artifact, if it recorded one."""
    try:
        for entry in reversed(world.__dict__.get("_code_editor_skipped") or []):
            if entry.get("target_object_id") == str(target_object_id):
                return str(entry.get("reason") or "")
    except Exception:
        pass
    return ""


def _compiles(code: str, fp: str) -> bool:
    try:
        compile(code, fp or "<patch>", "exec")
        return True
    except SyntaxError:
        return False


# Shared with the document editor, which had the identical truncate-then-ask-for-
# the-whole-file defect.
_apply_edits = apply_anchored_edits


def _unified_diff(before: str, after: str, path: str) -> str:
    import difflib

    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=path or "before",
            tofile=path or "after",
            n=2,
        )
    )


class CodeEditorLLM:
    def generate_patch(self, *, actor_id: str, target_object_id: str, edit_goal: str,
                       rationale: str, world: Any, tick: int, patch_id: str,
                       client: Optional[OrgLLMClient] = None,
                       revision: Optional[str] = None) -> CodePatch:
        """``revision`` carries what a previous attempt was refused for, so a
        rewrite answers the refusal instead of repeating the change that drew it."""
        art = (getattr(world, "product_artifacts", {}) or {}).get(target_object_id)
        rel_tasks = [t.task_id for t in getattr(world, "tasks", {}).values()
                     if target_object_id in getattr(t, "linked_artifacts", [])]
        data = None
        llm_declined = False
        if client is not None and art is not None:
            data = self._llm(client, art, target_object_id, edit_goal, rationale, world,
                             revision=revision)
            llm_declined = data is None
        if data is None:
            data = self._template(art, target_object_id, edit_goal)
        decline_reason = ""
        if llm_declined:
            # Why it declined matters downstream: an unanswered call says nothing
            # about whether editing this file is a good idea, while five failed
            # attempts to aim an anchor is a fact about this model on this file.
            decline_reason = _last_skip_reason(world, target_object_id)
            # The template appends COMMENT lines. That is the point in no-LLM
            # mode, where determinism is the whole design, but here a model was
            # attached and did not answer - and a comment block is not a fix.
            # Measured: six patches carrying the identical template summary
            # landed as ACCEPTED on cloning.py, download.py, query_parsing.py
            # and three more, each burning one of the issue's limited attempts
            # and leaving comment noise in the source. One of them was the
            # answer to "include_submodules is never passed on", so the
            # organization's own follow-up mechanism was consumed by a patch
            # that changed nothing.
            #
            # Marking it lets the caller record an infrastructure failure
            # instead of a product change - the same distinction that keeps an
            # empty container mount from reading as a broken product.
            _record_skip(world, target_object_id, "code_editor_llm_returned_nothing",
                         str(edit_goal)[:160])
        fp = getattr(art, "linked_file_path", None) if art else None
        cur = (getattr(art, "content", "") or "") if art else ""
        new_content = (data.get("new_content") or "").strip()
        if not new_content:
            new_content = _render_code_change(cur, data)
        # keep .py edits runnable: if the LLM produced invalid Python, repair it once via the
        # LLM, else fall back to the (valid) template-grown content, else to current (-> no-op).
        if fp and fp.endswith(".py") and new_content and not _compiles(new_content, fp):
            new_content = self._make_valid(client, fp, new_content, cur, data, art, target_object_id, edit_goal)
        patch = CodePatch(
            patch_id=patch_id, target_object_id=target_object_id, actor_id=actor_id, tick=tick,
            patch_type=data.get("patch_type", "code_patch"),
            edit_goal=edit_goal or data.get("edit_goal", ""),
            new_content=new_content,
            files_changed=[fp] if fp else [],
            pseudo_diff=data.get("pseudo_diff", ""),
            change_summary=data.get("change_summary", ""),
            added_fields=ensure_list(data.get("added_fields")),
            added_checks=ensure_list(data.get("added_checks")),
            changed_behavior=ensure_list(data.get("changed_behavior")),
            known_limitations=ensure_list(data.get("known_limitations")),
            resolved_gaps=ensure_list(data.get("resolved_gaps")),
            related_issue_ids=ensure_list(data.get("related_issue_ids")),
            related_task_ids=ensure_list(data.get("related_task_ids")) or rel_tasks)
        patch.unified_diff = str(data.get("unified_diff") or "")
        patch.llm_declined = llm_declined
        patch.decline_reason = decline_reason
        patch.edit_attempts = int(data.get("edit_attempts") or 0)
        return patch

    def _llm(self, client, art, oid, edit_goal, rationale, world, *, revision=None):
        gaps = "; ".join(getattr(art, "known_gaps", []) or []) or "(none recorded)"
        content = getattr(art, "content", "") or ""
        ceiling = _edit_input_ceiling()
        if ceiling is not None and len(content) > ceiling:
            # Refuse rather than truncate. Anchors must be copied from the file,
            # so a partial view invites the model to quote text it never saw.
            # Measured on a real run: five of the seven files the organization
            # patched were over the old 6000-char cap, including one it saw 56%
            # of, and the resulting patches referenced names never defined.
            _record_skip(world, oid, "file_exceeds_edit_budget",
                         f"{len(content)} chars > {ceiling}")
            return None
        build_err = getattr(world, "_build_error", "") or ""
        build_block = ""
        if build_err:
            build_block = (f"\nCURRENT BUILD ERROR (the product's smoke test fails end-to-end):\n{build_err}\n"
                           "If THIS file is the cause, fix the root cause. Either way, your edit MUST keep the "
                           "whole pipeline importable + runnable end-to-end (callers and callees must still "
                           "agree on signatures/contracts) — do not tighten a contract a caller still violates.\n")
        # IDE-like "find references": tell the editor which of this file's public symbols other modules
        # import, so it never deletes/renames them (the interface whack-a-mole). Adding new ones is fine.
        api_block = ""
        try:
            from environments.org_env.product.interface_guard import depended_upon_symbols
            dep = depended_upon_symbols(world, art)
            if dep:
                api_block = ("\nPUBLIC API — other modules IMPORT these symbols from this file: "
                             f"{', '.join(sorted(dep))}\n"
                             "Prefer keeping them backward-compatible. If a refactor really must remove/rename "
                             "one, that's allowed — the affected callers will be flagged to update — but do NOT "
                             "delete a symbol gratuitously (that only creates import breaks to chase).\n")
        except Exception:
            pass
        # The surface the pack publishes for this file. api_block above is about
        # symbols that already exist and other modules already import, so it says
        # nothing about a class nobody has written yet — which is exactly the kind
        # the tests were waiting for. Over one run the contract naming every one
        # of them sat in the world, was read zero times, and reached no prompt:
        # the names were guessed and then corrected by collision, UploadConflict
        # for UploadConflictError and ReplicaSet for ReplicaStatus, in every arm.
        contract_block = ""
        try:
            from environments.org_env.product.published_surface import published_surface
            contract_block = published_surface(world, art) or ""
        except Exception:  # noqa: BLE001
            pass
        # The rules the organization adopted for itself. Without them here, an arm
        # could carry a rule requiring a module's published callable surface to be
        # preserved exactly, and write code dropping a required class and renaming
        # a public method — because nothing writing code had ever been shown it.
        rules_block = ""
        try:
            from environments.org_env.llm.protocol_review import rules_block as _rules
            rules_block = _rules(world)
        except Exception:  # noqa: BLE001
            pass
        base = (f"Target file:\n{oid} — {art.title}\n{(art.summary or '')[:300]}\n\n"
                f"CURRENT FILE CONTENT:\n{content or '(empty file)'}\n"
                f"{build_block}{contract_block}{api_block}{rules_block}\n"
                f"Edit goal:\n{edit_goal}\n\nReason:\n{rationale}\n\nKnown gaps:\n{gaps}\n\n"
                "Return the patch JSON with `edits` = the list of anchored "
                "replacements that make this change. Quote each `search` exactly "
                "from the file above; do not return the whole file."
                + ("\n\nThis file is EMPTY: send one edit with an empty `search` "
                   "and the whole new file in `replace`." if not content.strip() else "")
                + (revision or ""))
        correction = ""
        for attempt in range(1, MAX_EDIT_ATTEMPTS + 1):
            try:
                # No output ceiling. The edits are short, but their length is not
                # known in advance, and a ceiling below the answer returns
                # nothing at all rather than a shorter answer.
                data = client.generate_json(CODE_EDITOR_SYSTEM, base + correction,
                                            CODE_EDITOR_SCHEMA,
                                            max_tokens=UNCAPPED_OUTPUT)
            except (LLMError, Exception):
                data = None
            if not isinstance(data, dict) or not data.get("change_summary"):
                # Nothing came back at all: a retry is a re-roll, not a
                # correction, so there is nothing to tell the model.
                continue
            patched, problems = _apply_edits(content, data.get("edits"))
            if not problems and patched != content:
                data["new_content"] = patched
                data["unified_diff"] = _unified_diff(
                    content, patched, getattr(art, "linked_file_path", "") or oid)
                data["edit_attempts"] = attempt
                return data
            if not problems and patched == content:
                problems = ["the edits left the file unchanged"]
            _record_skip(world, oid, "code_editor_edit_did_not_apply",
                         f"attempt {attempt}/{MAX_EDIT_ATTEMPTS}: {problems[0]}")
            correction = (
                "\n\nYour previous answer could not be applied:\n"
                + "\n".join(f"- {problem}" for problem in problems)
                + "\nRe-read the file above and copy each `search` from it exactly."
            )
        return None

    def _make_valid(self, client, fp, code, cur, data, art, oid, edit_goal):
        """Guarantee runnable Python: try an LLM repair, then the (valid) template-grown file,
        then the current content (-> no-op -> the validator rejects it). Never ship invalid code."""
        if client is not None:
            repaired = self._repair(client, fp, code)
            if repaired and _compiles(repaired, fp):
                return repaired
        seed = data if (data.get("pseudo_diff") or data.get("added_fields")) else self._template(art, oid, edit_goal)
        templ = _render_code_change(cur, seed)
        return templ if _compiles(templ, fp) else cur

    def _repair(self, client, fp, code):
        sysmsg = ("You are fixing a Python file that has a SYNTAX ERROR. Return JSON "
                  "{\"new_content\": <full corrected file>} ONLY. Do NOT use triple-quoted strings; "
                  "use # comments. Keep it valid, importable Python and preserve the intended change.")
        ceiling = _edit_input_ceiling()
        if ceiling is not None and len(code) > ceiling:
            # Same contradiction as the edit path, and worse here: this file is
            # already broken, so a repair written against a partial view cannot
            # be right. Returning empty lets the caller fall back to content it
            # knows compiles.
            #
            # Shares the edit ceiling deliberately: a file only reaches repair
            # after an edit broke it, so it already cleared that bar. Unlike the
            # edit path this one still asks for the whole file back, so an
            # operator who removes the ceiling is also removing the bound on
            # what a repair may be asked to emit.
            return ""
        user = f"File {fp} has a syntax error. Return the corrected FULL file:\n\n{code}"
        try:
            d = client.generate_json(sysmsg, user, {"new_content": "string"},
                                     max_tokens=UNCAPPED_OUTPUT)
            return (d.get("new_content") or "").strip() if isinstance(d, dict) else ""
        except (LLMError, Exception):
            return ""

    def _template(self, art, oid: str, edit_goal: str) -> dict:
        if oid in _TEMPLATES:
            return dict(_TEMPLATES[oid])
        gaps: List[str] = list(getattr(art, "known_gaps", []) or []) if art else []
        g = gaps[0] if gaps else "behavior is underspecified"
        return {
            "patch_type": "code_patch",
            "pseudo_diff": f"# address: {g}\n+ TODO marker replaced with explicit handling",
            "change_summary": f"Made a small, inspectable change addressing: {g}.",
            "added_checks": [f"explicit handling for: {g}"],
            "changed_behavior": [f"no longer silently ignores: {g}"],
            "known_limitations": ["partial fix; broader refactor still pending"],
        }


__all__ = ["CodeEditorLLM", "CODE_EDITOR_SYSTEM"]
