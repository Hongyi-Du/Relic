"""Seed the messy research-agent product substrate into an OrgWorld.

The product (LanternScout) is a REAL, runnable mini CLI research agent — `python smoke_check.py`
runs plan -> search(local corpus) -> track sources -> make claims -> write report -> eval and
prints a metrics JSON line. It is deliberately INCOMPLETE (claims not linked to sources, no
evidence gate, no credibility scoring) so the organization has concrete gaps to close; every
accepted patch grows the real file content (see ProductArtifact.content + materialize.py).
"""
from __future__ import annotations

from typing import Any, Dict

from environments.org_env.product.objects import ProductArtifact, ProductState

DEFAULT_COMPANY_CONFIG = {
    "company_name": "LanternForge",
    "product_name": "LanternScout",
    "product_stage": "messy pre-launch prototype",
    "product_purpose": (
        "an early-stage research agent prototype for evidence-grounded investigation, "
        "claim tracking, lightweight experiments, source organization, and credible report generation"
    ),
}

# starter repo files / docs / eval / example / corpus: (path, type, summary, [known gaps])
_STARTER_FILES = [
    ("README.md", "repo_file", "Honest README: capabilities + limitations, no overclaim.",
     ["search is a local stub", "claims not linked to sources", "no evidence gate", "no source credibility scoring"]),
    ("agent.py", "repo_file", "CLI entry: python agent.py '<question>'.",
     ["orchestration is minimal", "no error handling"]),
    ("research_loop.py", "repo_file", "plan -> search -> track -> claim -> report flow.",
     ["claims not linked to sources", "single-step plan only"]),
    ("llm.py", "repo_file", "LLM core: synthesize findings from sources (OpenAI + offline fallback).",
     ["no caching", "no token budget", "summary not grounded to source_ids"]),
    ("tools/search_stub.py", "tool_stub", "Local mock search over corpus/*.md.",
     ["local stub, not real retrieval"]),
    ("tools/source_tracker.py", "tool_stub", "Tracks sources by id/title/path.",
     ["no source credibility scoring"]),
    ("tools/claim_tracker.py", "tool_stub", "Records claims.",
     ["does not enforce source_ids/evidence", "no uncertainty validation"]),
    ("tools/report_writer.py", "tool_stub", "Drafts a markdown report.",
     ["no evidence gate", "shows unsupported claims"]),
    ("tools/notes_dump.py", "tool_stub", "Scattered tech-debt notes.", ["unstructured tech debt"]),
    ("eval/eval_stub.py", "eval", "Computes coverage metrics for a run.",
     ["metrics are basic", "no human-validated benchmark"]),
    ("smoke_check.py", "repo_file", "Runs the sample task + prints report + metrics JSON.",
     ["only one sample task"]),
    ("examples/sample_market_research_task.md", "doc", "Sample task: input + expected trace + report shape.",
     ["only one example"]),
    ("corpus/vector_dbs.md", "doc", "Mock corpus doc on vector databases.", []),
    ("corpus/rag.md", "doc", "Mock corpus doc on RAG.", []),
    ("docs/product_design.md", "doc", "Design notes (vision/workflow/TODOs).",
     ["not a coherent spec", "mixes vision/notes/TODOs"]),
    ("docs/random_notes.md", "doc", "Unstructured brainstorm.", ["unorganized brainstorm"]),
]

_README = """# LanternScout (mini research agent)

LanternScout is an early CLI research agent prototype. Given a question, it searches a local
corpus, tracks sources, makes claims, writes a markdown report, and runs a small evaluation.

## Run
    python smoke_check.py          # run the sample task + print metrics
    python agent.py "your question"

## Current status (honest)
- runnable beta: `python smoke_check.py` runs end-to-end (exit 0) on the sample task
- the report SUMMARY is synthesized by an LLM (llm.py); offline it falls back to an extractive summary
- claims ARE linked to their search-hit source (STRUCTURAL grounding), but the link is NOT
  semantically validated — the source may not actually support the claim
- claims can be near-duplicates; source titles / provenance are weak
- the LLM summary is NOT grounded to specific source_ids
- search is a LOCAL STUB over corpus/*.md (no real web retrieval)
- report writer has NO evidence gate yet (weakly-supported claims are shown, not gated)
- source credibility scoring is minimal; eval metrics are basic and not human-validated
- NO market validation yet (no real users, trials, or willingness-to-pay)

## Known limitations
Reports may contain unsupported claims until claim->source linking and an evidence gate are
enforced. Do not treat the output as verified.
"""

_DESIGN_DOC = """# Product Design Notes

Open questions: Is this a research assistant, report generator, or evidence tracker?
Should the user see sources first or report first? What counts as a credible claim?
Do we optimize for speed, quality, or traceability?

Intended workflow: user asks -> plan -> search -> track sources -> make claims (linked to
sources) -> write report (gated on evidence) -> run eval.

Problems today: claim tracker not connected to source tracker; source tracker does not score
credibility; report writer has no evidence gate; eval is basic; no one owns the release checklist.
"""

_RANDOM_NOTES = """# random notes

- maybe pivot to 'evidence tracker' not 'report generator'?
- victor wants tests, paul wants demo, calvin wants it cheap
- nobody owns the release checklist
- TODO: figure out what 'credible' means
"""

_SAMPLE_TASK = """# Sample task: market research

Input query:
"What are the leading open-source vector databases and their tradeoffs?"

Expected trace:
plan -> search(corpus) -> track sources -> make claims -> write report -> eval

Expected report shape:
# Research Report: <query>
## Sources (N)
- <title> (<source_id>)
## Findings
- <claim> [<source_ids> | unsupported]

Quality bar (target):
- unsupported_claim_rate should drop toward 0 as claim->source linking is enforced
- claim_evidence_coverage should rise toward 1.0
"""

_CORPUS_VDB = """# Vector databases (overview)

FAISS is a library for efficient similarity search; it is fast but not a managed service.
Milvus is an open-source vector database with horizontal scaling.
Qdrant is an open-source vector database written in Rust with strong payload filtering.
Weaviate supports hybrid search and a module ecosystem.
pgvector adds vector search to PostgreSQL, convenient when you already run Postgres.
"""

_CORPUS_RAG = """# Retrieval-augmented generation (overview)

RAG combines a retriever over a vector store with a generator model.
Chunking strategy and the embedding model strongly affect retrieval quality.
Citing retrieved sources is important for trust and verifiability.
"""

_AGENT_PY = '''"""LanternScout mini CLI. Usage: python agent.py "your research question"."""
import sys

from research_loop import run_research
from eval.eval_stub import run_eval


def main(argv):
    query = argv[1] if len(argv) > 1 else "What are the leading open-source vector databases?"
    res = run_research(query)
    print(res["report"])
    print("--- metrics ---")
    print(run_eval(res["claims"], res["sources"]))


if __name__ == "__main__":
    main(sys.argv)
'''

_RESEARCH_LOOP = '''"""LanternScout research loop: plan -> search -> track sources -> make claims -> write report.

Intentionally incomplete: claims are NOT linked to their sources yet (see GAP markers), so the
eval metrics start weak. Closing these gaps is the organization's job.
"""
from tools.search_stub import search
from tools.source_tracker import SourceTracker
from tools.claim_tracker import ClaimTracker
from tools.report_writer import write_report


def run_research(query, corpus_dir="corpus"):
    plan = [query]                      # single-step plan (GAP: no multi-step planning)
    st = SourceTracker()
    ct = ClaimTracker()
    for step in plan:
        for rec in search(step, corpus_dir=corpus_dir):
            st.add(rec)
            # claims ARE linked to the search-hit source (structural grounding). GAPs that remain:
            # the link is NOT semantically validated (the source may not actually support the
            # claim), claims can be near-duplicates, and source titles/provenance are weak.
            ct.add(rec["snippet"] or ("about " + step), source_ids=[rec["source_id"]])
    sources = st.all()
    claims = ct.all()
    report = write_report(query, claims, sources)
    return {"query": query, "sources": sources, "claims": claims, "report": report}
'''

_SEARCH_STUB = '''"""Local mock search over corpus/*.md -- returns real source records (no network)."""
import glob
import os


def search(query, k=5, corpus_dir="corpus"):
    terms = [w for w in query.lower().split() if len(w) > 2]
    out = []
    for i, path in enumerate(sorted(glob.glob(os.path.join(corpus_dir, "*.md")))):
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        score = sum(text.lower().count(t) for t in terms)
        snippet = ""
        for ln in text.splitlines():
            if ln.strip() and not ln.startswith("#"):
                snippet = ln.strip()[:200]
                break
        out.append({"source_id": "src_" + str(i), "title": os.path.basename(path),
                    "path": path, "snippet": snippet, "score": score})
    out.sort(key=lambda r: -r["score"])
    return out[:k]
'''

_SOURCE_TRACKER = '''"""Records sources by id/title/path. GAP: no credibility scoring yet."""


class Source:
    def __init__(self, source_id, title, path, url=None):
        self.source_id = source_id
        self.title = title
        self.path = path
        self.url = url
        # GAP: no credibility_score


class SourceTracker:
    def __init__(self):
        self.sources = {}

    def add(self, rec):
        s = Source(rec.get("source_id"), rec.get("title"), rec.get("path"), rec.get("url"))
        self.sources[s.source_id] = s
        return s

    def all(self):
        return list(self.sources.values())
'''

_CLAIM_TRACKER = '''"""Records claims. GAP: does not enforce source_ids / uncertainty / evidence."""


class Claim:
    def __init__(self, claim_id, text, source_ids=None, uncertainty_note=""):
        self.claim_id = claim_id
        self.text = text
        self.source_ids = list(source_ids or [])
        self.uncertainty_note = uncertainty_note


class ClaimTracker:
    def __init__(self):
        self.claims = []

    def add(self, text, source_ids=None, uncertainty_note=""):
        # GAP: accepts a claim with no source_ids; no validate_claim_evidence yet
        c = Claim("claim_" + str(len(self.claims)), text, source_ids, uncertainty_note)
        self.claims.append(c)
        return c

    def all(self):
        return list(self.claims)
'''

_LLM_CORE = '''"""LanternScout LLM core — the agent calls an LLM to synthesize findings from the
retrieved sources. Uses OpenAI when an API key is present (LANTERN_LLM_KEY or OPENAI_API_KEY);
otherwise returns "" so callers fall back to a deterministic offline synthesis (so the tool, and
the smoke test, always run). Self-contained (stdlib urllib only, no extra deps)."""
import json
import os
import urllib.request


def _key():
    return os.environ.get("LANTERN_LLM_KEY") or os.environ.get("OPENAI_API_KEY") or ""


def available():
    return bool(_key())


def call_llm(prompt, system="You are LanternScout, a careful research assistant.", max_tokens=500):
    """Return the model's text, or "" when offline / on any error (never raises)."""
    key = _key()
    if not key:
        return ""
    base = os.environ.get("LANTERN_LLM_BASE", "https://api.openai.com/v1").rstrip("/")
    model = os.environ.get("LANTERN_LLM_MODEL", "gpt-4o-mini")
    body = json.dumps({"model": model, "temperature": 0.2, "max_tokens": max_tokens,
                       "messages": [{"role": "system", "content": system},
                                    {"role": "user", "content": prompt}]}).encode("utf-8")
    req = urllib.request.Request(base + "/chat/completions", data=body,
                                 headers={"Authorization": "Bearer " + key,
                                          "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return (data["choices"][0]["message"]["content"] or "").strip()
    except Exception:
        return ""
'''

_REPORT_WRITER = '''"""Drafts a markdown report. The findings SUMMARY is synthesized by the LLM core
(falls back to an extractive summary offline). GAP: no evidence gate yet -- unsupported claims are
still shown (claim->source linking is not enforced)."""
from llm import call_llm


def write_report(query, claims, sources):
    nl = chr(10)
    lines = ["# Research Report: " + str(query), ""]
    lines.append("## Sources (" + str(len(sources)) + ")")
    for s in sources:
        lines.append("- " + str(s.title) + " (" + str(s.source_id) + ")")
    lines.append("")
    # LLM-backed synthesis (the product's core capability): summarize findings from the sources.
    src_text = nl.join("- " + str(getattr(s, "title", "")) + " (" + str(getattr(s, "source_id", "")) + ")"
                       for s in sources)
    claim_text = nl.join("- " + str(getattr(c, "text", "")) for c in claims)
    prompt = ("Question: " + str(query) + nl + "Sources:" + nl + src_text + nl
              + "Draft claims:" + nl + claim_text + nl
              + "Write a concise findings summary (3-5 bullets). Mark any claim you cannot tie to "
                "a listed source as (unsupported).")
    summary = call_llm(prompt)
    lines.append("## Summary")
    lines.append(summary if summary else "(offline: extractive summary) " + "; ".join(
        str(getattr(c, "text", "")) for c in claims[:3]))
    lines.append("")
    lines.append("## Findings")
    for c in claims:
        # GAP: a claim is included regardless of whether it has supporting source_ids
        cite = (" [" + ", ".join(c.source_ids) + "]") if c.source_ids else " [unsupported]"
        lines.append("- " + str(c.text) + cite)
    return nl.join(lines)
'''

_NOTES_DUMP = '''# scattered notes (tech debt)
# - claim tracker not wired to source tracker
# - no source credibility scoring
# - report writer has no evidence gate
# - what is "cheap mode"?
'''

_EVAL_STUB = '''"""Evaluation metrics for a research run (real, simple)."""


def run_eval(claims, sources):
    n = len(claims)
    supported = sum(1 for c in claims if getattr(c, "source_ids", None))
    used = set()
    for c in claims:
        for sid in (getattr(c, "source_ids", None) or []):
            used.add(sid)
    nsrc = max(1, len(sources))
    return {
        "n_claims": n,
        "n_sources": len(sources),
        "claim_evidence_coverage": round(supported / n, 3) if n else 0.0,
        "unsupported_claim_rate": round((n - supported) / n, 3) if n else 0.0,
        "source_coverage": round(len(used) / nsrc, 3),
    }
'''

_SMOKE_CHECK = '''"""Smoke test: run the sample task end-to-end; print the report + a metrics JSON line.
Exits non-zero only if the pipeline crashes. The in-world release gate runs this."""
import json
import sys

from research_loop import run_research
from eval.eval_stub import run_eval

QUERY = "What are the leading open-source vector databases and their tradeoffs?"


def main():
    res = run_research(QUERY)
    metrics = run_eval(res["claims"], res["sources"])
    print(res["report"])
    print(json.dumps({"ok": True, "query": QUERY, "metrics": metrics}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''

_STARTER_CONTENT = {
    "README.md": _README,
    "agent.py": _AGENT_PY,
    "research_loop.py": _RESEARCH_LOOP,
    "llm.py": _LLM_CORE,
    "tools/search_stub.py": _SEARCH_STUB,
    "tools/source_tracker.py": _SOURCE_TRACKER,
    "tools/claim_tracker.py": _CLAIM_TRACKER,
    "tools/report_writer.py": _REPORT_WRITER,
    "tools/notes_dump.py": _NOTES_DUMP,
    "eval/eval_stub.py": _EVAL_STUB,
    "smoke_check.py": _SMOKE_CHECK,
    "examples/sample_market_research_task.md": _SAMPLE_TASK,
    "corpus/vector_dbs.md": _CORPUS_VDB,
    "corpus/rag.md": _CORPUS_RAG,
    "docs/product_design.md": _DESIGN_DOC,
    "docs/random_notes.md": _RANDOM_NOTES,
}

_INITIAL_ISSUES = [
    ("issue_1", "Make research loop useful", "unclear",
     "The current loop can draft reports but the actual workflow is not clear."),
    ("issue_2", "Claim tracker maybe needed?", "medium",
     "Reports can include claims without explicit evidence links."),
    ("issue_3", "Search/source quality", "unclear",
     "Source tracker stores sources but does not judge credibility."),
    ("issue_4", "Eval stub is fake", "high",
     "eval/eval_stub.py has metric names but no validated evaluation."),
    ("issue_5", "Need demo soon", "high",
     "Paul wants a demo, but the product may overclaim credibility."),
    ("issue_6", "Docs do not match code", "medium",
     "README and design notes describe abilities that are only partially implemented."),
    ("issue_7", "Cheap mode?", "unclear",
     "There is pressure to make a cheap fast mode, but no one knows what it can safely skip."),
]

# v14: a RUNNABLE, structurally-grounded beta — the remaining gaps are about SEMANTIC quality
# and MARKET validation, not basic plumbing / integration.
_KNOWN_SYSTEMIC_ISSUES = [
    "claim->source links are structural (the search hit), NOT semantically validated",
    "claims can be near-duplicates and low-specificity",
    "the LLM summary is not grounded to specific source_ids",
    "source titles / provenance are weak (some sources lack a real title)",
    "report writer has no semantic evidence gate (weakly-supported claims are shown)",
    "source credibility scoring is minimal / coarse",
    "eval metrics are basic and not human-validated; report vs eval credibility may disagree",
    "search is a local stub, not real retrieval",
    "no market validation: no real users, trials, customer tickets, or willingness-to-pay",
    "docs may overpromise vs the actual (beta) capabilities",
]


def seed_product(world: Any, config: Dict[str, Any] = None) -> ProductState:
    """Route to the configured product substrate (OSS time-machine brief §6.1).

    ``company_config["product_substrate"]["type"]`` selects the substrate; missing or
    ``synthetic_lanternscout`` keeps the existing hand-written LanternScout behavior (default /
    debug / dev), ``oss_time_machine`` seeds a real OSS project's early release. The synthetic path
    is byte-for-byte unchanged."""
    cfg = dict(DEFAULT_COMPANY_CONFIG)
    cfg.update(config or {})
    substrate = cfg.get("product_substrate") or {}
    stype = substrate.get("type", "synthetic_lanternscout") if isinstance(substrate, dict) else "synthetic_lanternscout"
    if stype == "oss_time_machine":
        from environments.org_env.product.substrates.oss_time_machine import seed_oss_time_machine_product
        return seed_oss_time_machine_product(world, substrate)
    return seed_synthetic_lanternscout_product(world, config)


def seed_synthetic_lanternscout_product(world: Any, config: Dict[str, Any] = None) -> ProductState:
    cfg = dict(DEFAULT_COMPANY_CONFIG)
    cfg.update(config or {})
    world.company_config = cfg
    ps = ProductState(
        product_id="product_research_agent",
        name=f"{cfg['product_name']} Research Agent Prototype",
        stage="messy_prelaunch_prototype",
        summary=("A real but messy early prototype of a CLI research agent. It runs end-to-end on a "
                 "local corpus (search -> sources -> claims -> report -> eval) but claim->source "
                 "linking, an evidence gate, source credibility, and validated eval are incomplete."),
        known_systemic_issues=list(_KNOWN_SYSTEMIC_ISSUES),
        repo_id=getattr(getattr(world, "repo_system", None), "repo", None) and
        getattr(world.repo_system.repo, "repo_id", "repo") or "repo")
    arts: Dict[str, ProductArtifact] = {}
    for path, atype, summary, gaps in _STARTER_FILES:
        aid = f"art_{path.replace('/', '_').replace('.', '_')}"
        arts[aid] = ProductArtifact(
            artifact_id=aid, artifact_type=atype, title=path, status="active",
            linked_repo_id=ps.repo_id, linked_file_path=path,
            summary=summary, content=_STARTER_CONTENT.get(path, ""), known_gaps=list(gaps),
            created_at_tick=0, updated_at_tick=0)
        ps.artifact_ids.append(aid)
    for iid, title, priority, problem in _INITIAL_ISSUES:
        arts[iid] = ProductArtifact(
            artifact_id=iid, artifact_type="issue", title=title, status="open",
            summary=problem, problem=problem, priority=priority, created_at_tick=0, updated_at_tick=0)
        ps.artifact_ids.append(iid)
        ps.open_issue_ids.append(iid)
    world.product = ps
    world.product_artifacts = arts
    return ps


__all__ = ["seed_product", "seed_synthetic_lanternscout_product", "DEFAULT_COMPANY_CONFIG"]
