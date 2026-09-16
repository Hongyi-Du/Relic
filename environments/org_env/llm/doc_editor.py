"""DocEditorLLM (v4 §2.4) — the execution-layer LLM that turns an already-decided doc
action into a concrete, grounded DocPatch. Falls back to a deterministic template patch
(built from the artifact's known_gaps + edit goal) when no client is available or the LLM
output is unusable, so the patch layer works in mock / no-LLM mode too.
"""
from __future__ import annotations

from typing import Any, List, Optional

from environments.org_env.llm.anchored_edits import apply_anchored_edits
from environments.org_env.llm.client import UNCAPPED_OUTPUT, LLMError, OrgLLMClient
from environments.org_env.product.patch_objects import DocPatch
from environments.org_env.proposals.objects import ensure_list

DOC_EDITOR_SYSTEM = (
    "You are editing a real product document inside a simulated startup. The edit decision is "
    "already made. Return your change as a list of `edits`, each an exact `search` string copied "
    "from the document and the `replace` text to put in its place. Quote enough surrounding lines "
    "to make each `search` appear exactly once. To add a new section, anchor on the neighbouring "
    "lines and repeat them in `replace`. Do not invent completed capabilities. If the goal is to "
    "reduce overclaiming, soften claims and add an explicit limitation. If the goal is to clarify "
    "workflow, separate vision / workflow / evidence requirements / TODOs. JSON only."
)

MAX_EDIT_ATTEMPTS = 5

# A behavioral contract is a bounded planning artifact, not a whole repository
# file rewrite. Keeping this request explicit avoids the ``UNCAPPED_OUTPUT``
# accounting envelope (32k per provider attempt). OpenAI reasoning models may
# raise this to their 6k effective minimum, but the caller's requested budget
# remains explicit and auditable.
PROGRAMBENCH_CONTRACT_OUTPUT_TOKENS = 4096

_PROGRAMBENCH_CONTRACT_COMMON_SYSTEM = (
    "You are authoring the one evidence-bound behavioral contract for a "
    "ProgramBench clean-room reconstruction. Return a JSON document patch with "
    "a non-empty `change_summary` and `edits`. Do not write or audit a "
    "README, even if retrieved public material contains a README. The contract "
    "must preserve the supplied evidence and corpus digests verbatim; include "
    "Markdown sections DOCUMENTED, OBSERVED, INFERRED, and UNKNOWN; include "
    "case-level public reference findings in OBSERVED and copy the supplied "
    "`Reference observations accepted: N cases` attestation exactly; never say "
    "that reference cases or outputs were unobserved when observations are supplied; "
    "state one Architecture and one repo-relative Source entrypoint. Use these exact, "
    "standalone field forms (without Markdown heading or bold markers): "
    "`Architecture: <one concrete architecture>` and "
    "`Source entrypoint: <one repo-relative path>`; explain how compile.sh "
    "will produce ./executable; and include a Verification plan using public probe "
    "reference_only followed by differential. Do not claim hidden-test behavior "
    "or access to reference source. JSON only."
)

PROGRAMBENCH_CONTRACT_CREATE_SYSTEM = (
    _PROGRAMBENCH_CONTRACT_COMMON_SYSTEM
    + " The target document is EMPTY: use exactly one edit whose `search` is "
    "the empty string and whose `replace` is the complete new Markdown contract."
)

PROGRAMBENCH_CONTRACT_REVISION_SYSTEM = (
    _PROGRAMBENCH_CONTRACT_COMMON_SYSTEM
    + " The target document already contains the previous accepted contract. "
    "Revise it for the supplied current public evidence and corpus. Every "
    "`search` must be a non-empty exact string copied from CURRENT DOCUMENT "
    "CONTENT and must occur exactly once. You may replace the complete existing "
    "contract by using its entire current text as one `search`, or use bounded "
    "exact anchors that preserve all untouched content. Never use an empty "
    "`search` for a revision."
)

# Backward-compatible exported name for callers/tests that inspect the create
# prompt. Runtime selection below uses the explicit create/revision variants.
PROGRAMBENCH_CONTRACT_SYSTEM = PROGRAMBENCH_CONTRACT_CREATE_SYSTEM

DOC_EDITOR_SCHEMA = {
    "patch_type": "doc_patch",
    "edit_goal": "string",
    "edits": [{"search": "string", "replace": "string"}],
    "changed_sections": [{"section": "string", "before": "string", "after": "string"}],
    "change_summary": "string",
    "removed_overclaims": ["string"],
    "added_limitations": ["string"],
    "added_requirements": ["string"],
    "remaining_risks": ["string"],
    # All three are read when the patch is built, and the first two also compose
    # the rendered document in _render_doc_change, but none was declared — so a
    # checklist the document was supposed to grow, a workflow section it was
    # supposed to gain, and the gaps it was supposed to close could only ever
    # come back empty.
    "checklist_items": ["string"],
    "workflow_sections": ["string"],
    "resolved_gaps": ["string"],
    "related_issue_ids": ["string"],
    "related_task_ids": ["string"],
}


def _render_doc_change(current: str, data: dict) -> str:
    """Deterministically grow the real document text from a symbolic patch (no-LLM mode).
    IDEMPOTENT: only appends lines not already present (re-applying = no-op = rejected)."""
    cur = current.rstrip("\n")
    cand: List[str] = []
    for s in data.get("changed_sections") or []:
        if s.get("after"):
            cand.append(str(s["after"]))
    for r in data.get("removed_overclaims") or []:
        cand.append(f"- (softened overclaim) {r}")
    for l in data.get("added_limitations") or []:
        cand.append(f"- Limitation: {l}")
    for r in data.get("added_requirements") or []:
        cand.append(f"- Requirement: {r}")
    for c in data.get("checklist_items") or []:
        cand.append(f"- [ ] {c}")
    for w in data.get("workflow_sections") or []:
        cand.append(f"### {w}")
    new_lines = [l for l in cand if l and l not in cur]
    if not new_lines:
        return (cur + "\n") if cur else ""
    summ = data.get("change_summary") or data.get("edit_goal") or "update"
    body = (cur + "\n\n" if cur else "") + "\n".join([f"<!-- change: {summ} -->"] + new_lines)
    return body.rstrip("\n") + "\n"


class DocEditorLLM:
    def generate_patch(self, *, actor_id: str, target_object_id: str, edit_goal: str,
                       rationale: str, world: Any, tick: int, patch_id: str,
                       client: Optional[OrgLLMClient] = None,
                       programbench_contract: bool = False) -> DocPatch:
        art = (getattr(world, "product_artifacts", {}) or {}).get(target_object_id)
        rel_tasks = [t.task_id for t in getattr(world, "tasks", {}).values()
                     if target_object_id in getattr(t, "linked_artifacts", [])]
        data = None
        failure_reason = ""
        edit_attempts = 0
        if client is not None and art is not None:
            data, failure_reason, edit_attempts = self._llm_result(
                client,
                art,
                edit_goal,
                rationale,
                world,
                programbench_contract=programbench_contract,
            )
        elif programbench_contract:
            failure_reason = (
                "target_artifact_missing" if art is None else "llm_client_missing"
            )
        if data is None and not programbench_contract:
            data = self._template(art, edit_goal)
        elif data is None:
            # A failed ProgramBench generation must not become the generic
            # README fallback merely because a retrieved surface is README.md.
            data = {"patch_type": "doc_create", "change_summary": ""}
        cur = (getattr(art, "content", "") or "") if art else ""
        new_content = (data.get("new_content") or "").strip()
        if not new_content:
            new_content = _render_doc_change(cur, data)
        patch = DocPatch(
            patch_id=patch_id, target_object_id=target_object_id, actor_id=actor_id, tick=tick,
            patch_type=data.get("patch_type", "doc_patch"),
            edit_goal=edit_goal or data.get("edit_goal", ""),
            new_content=new_content,
            changed_sections=list(data.get("changed_sections") or []),
            change_summary=data.get("change_summary", ""),
            removed_overclaims=ensure_list(data.get("removed_overclaims")),
            added_limitations=ensure_list(data.get("added_limitations")),
            added_requirements=ensure_list(data.get("added_requirements")),
            remaining_risks=ensure_list(data.get("remaining_risks")),
            checklist_items=ensure_list(data.get("checklist_items")),
            workflow_sections=ensure_list(data.get("workflow_sections")),
            resolved_gaps=ensure_list(data.get("resolved_gaps")),
            related_issue_ids=ensure_list(data.get("related_issue_ids")),
            related_task_ids=ensure_list(data.get("related_task_ids")) or rel_tasks)

        if programbench_contract:
            # Patch objects serialize their __dict__, so these bounded
            # diagnostic codes remain checkpoint-visible without persisting
            # provider errors. Native patches retain their historical shape.
            patch.llm_declined = not bool(data.get("new_content"))
            patch.decline_reason = failure_reason if patch.llm_declined else ""
            patch.edit_attempts = int(
                data.get("edit_attempts") or edit_attempts or 0
            )
            patch.generation_mode = (
                "programbench_contract_revision"
                if cur.strip()
                else "programbench_contract_full_create"
            )
        return patch

    def _llm(self, client, art, edit_goal, rationale, world):
        data, _failure_reason, _attempts = self._llm_result(
            client, art, edit_goal, rationale, world
        )
        return data

    @staticmethod
    def _failure_code(error: BaseException) -> tuple[str, bool]:
        """Return a de-sensitive code and whether editor retries must stop."""

        text = str(error).casefold()
        if "experiment_resource_exhausted" in text:
            return "resource_exhausted", True
        if "prompt_visibility_violation" in text:
            return "prompt_visibility_denied", True
        if isinstance(error, LLMError):
            return "llm_error", False
        return "editor_exception", False

    def _llm_result(
        self,
        client,
        art,
        edit_goal,
        rationale,
        world,
        *,
        programbench_contract: bool = False,
    ):
        """Ask for anchored edits against the WHOLE document.

        This used to show 1800 characters and ask for "the FULL updated
        document" back. Everything past the cut was invisible to the model and
        absent from its answer, so an edit to a long document silently deleted
        the rest of it — and the answer had to be as long as the file, under an
        output ceiling and a 60-second gateway timeout that a long document
        cannot meet. Anchored edits make the answer's size follow the change
        instead of the file, which is what the code editor was changed to do
        after the same two failures.
        """
        gaps = "; ".join(getattr(art, "known_gaps", []) or []) or "(none recorded)"
        content = getattr(art, "content", "") or ""
        contract_system = (
            PROGRAMBENCH_CONTRACT_REVISION_SYSTEM
            if programbench_contract and content.strip()
            else PROGRAMBENCH_CONTRACT_CREATE_SYSTEM
        )
        base = (f"Target document:\n{art.artifact_id} — {art.title}\n{art.summary or ''}\n\n"
                f"CURRENT DOCUMENT CONTENT:\n{content or '(empty)'}\n\n"
                f"Edit goal:\n{edit_goal}\n\nReason:\n{rationale}\n\nKnown gaps:\n{gaps}\n\n"
                "Return the patch JSON with `edits`: exact `search` strings copied "
                "from the document above, each with its `replace` text."
                + (
                    "\n\nThis document is EMPTY. Return exactly one edit with "
                    "`search` set to the empty string and the complete new "
                    "Markdown document in `replace`."
                    if programbench_contract and not content.strip()
                    else (
                        "\n\nThis is a CONTRACT REVISION. Use non-empty exact "
                        "anchors copied from CURRENT DOCUMENT CONTENT. Replacing "
                        "the full existing document with one exact full-document "
                        "anchor is allowed."
                        if programbench_contract
                        else ""
                    )
                ))
        correction = ""
        last_failure = "invalid_response"
        attempts = 0
        for attempt in range(1, MAX_EDIT_ATTEMPTS + 1):
            attempts = attempt
            try:
                data = client.generate_json(
                    (
                        contract_system
                        if programbench_contract
                        else DOC_EDITOR_SYSTEM
                    ),
                    base + correction,
                    DOC_EDITOR_SCHEMA,
                    max_tokens=(
                        PROGRAMBENCH_CONTRACT_OUTPUT_TOKENS
                        if programbench_contract
                        else UNCAPPED_OUTPUT
                    ),
                )
            except Exception as error:  # noqa: BLE001 - classified, never exposed
                data = None
                last_failure, terminal = self._failure_code(error)
                if terminal:
                    break
            if not isinstance(data, dict) or not data.get("change_summary"):
                # Nothing came back: a retry is a re-roll, not a correction, so
                # there is nothing to tell the model.
                if data is not None:
                    last_failure = "invalid_response"
                continue
            patched, problems = apply_anchored_edits(content, data.get("edits"))
            if (
                programbench_contract
                and not problems
                and not patched.strip()
            ):
                problems = [
                    "the ProgramBench contract body is empty or whitespace-only"
                ]
            if not problems and patched != content:
                data["new_content"] = patched
                data["edit_attempts"] = attempt
                return data, "", attempt
            if not problems and patched == content:
                problems = ["the edits left the document unchanged"]
                last_failure = "unchanged_edit"
            else:
                last_failure = "anchor_apply_failed"
            correction = (
                "\n\nYour previous answer could not be applied:\n"
                + "\n".join(f"- {problem}" for problem in problems)
                + "\nRe-read the document above and copy each `search` from it exactly."
            )
        return None, last_failure, attempts

    def _template(self, art, edit_goal: str) -> dict:
        gaps: List[str] = list(getattr(art, "known_gaps", []) or []) if art else []
        goal = edit_goal or (gaps[0] if gaps else "clarify the document and align it with current capability")
        text = ((getattr(art, "summary", "") if art else "") + " " + goal + " " +
                (getattr(art, "artifact_id", "") if art else "")).lower()
        if "readme" in text or "overclaim" in text or "overpromise" in text or "evidence-grounded" in text:
            return {
                "patch_type": "doc_audit",
                "change_summary": "Softened the README's evidence-guarantee overclaim and recorded the current limitation.",
                "removed_overclaims": ["'evidence-grounded reports' implied a guarantee the claim tracker does not enforce yet"],
                "added_limitations": ["Reports may contain unsupported claims until evidence links are enforced."],
                "added_requirements": [],
                "remaining_risks": ["users may still over-trust generated reports"],
            }
        g = gaps[0] if gaps else "the workflow is underspecified"
        return {
            "patch_type": "doc_patch",
            "changed_sections": [{"section": "Scope",
                                  "before": (getattr(art, "summary", "") if art else "")[:120],
                                  "after": f"Clarified: {g}."}],
            "change_summary": f"Clarified the document around: {g}.",
            "added_requirements": [f"Document must explicitly address: {g}."],
            "remaining_risks": [],
        }


__all__ = [
    "DocEditorLLM",
    "DOC_EDITOR_SYSTEM",
    "PROGRAMBENCH_CONTRACT_OUTPUT_TOKENS",
    "PROGRAMBENCH_CONTRACT_CREATE_SYSTEM",
    "PROGRAMBENCH_CONTRACT_REVISION_SYSTEM",
    "PROGRAMBENCH_CONTRACT_SYSTEM",
]
