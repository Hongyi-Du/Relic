"""SearchSystem — 5-domain retrieval over frozen corpora (DESIGN env_org §41-§42/§43).

Domains: internal / repo / sandbox / external_community / frozen_web. Every search
writes a SearchLog with corpus_version + content_hashes. Reproducibility
(acceptance ⑭): in ``replay_mode`` a cache miss raises ``ReplaySearchMiss`` rather
than performing a live search. No live web — frozen_web is a seeded snapshot.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from environments.org_env.backend.workspace.provenance import content_hash


class ReplaySearchMiss(RuntimeError):
    """Raised when replay_mode is on and a search key is absent (acceptance ⑭)."""


SEARCH_DOMAINS = ("internal", "repo", "sandbox", "external_community", "frozen_web")


@dataclass
class SearchLog:
    search_id: str
    tick: int
    agent_id: str
    query: str
    search_domain: str
    corpus_version: str
    retrieved_object_ids: List[str] = field(default_factory=list)
    content_hashes: List[str] = field(default_factory=list)
    opened_results: List[str] = field(default_factory=list)
    used_in_action: bool = False
    cache_hit: bool = False


def _match(query: str, *texts: str) -> int:
    q = [t for t in query.lower().split() if t]
    blob = " ".join(x.lower() for x in texts if x)
    return sum(1 for t in q if t in blob)


class SearchSystem:
    def __init__(self, world: Any, *, corpus_version: str = "v0", replay_mode: bool = False):
        self.world = world
        self.corpus_version = corpus_version
        self.replay_mode = replay_mode
        self.logs: List[SearchLog] = []
        self._cache: Dict[str, List[Tuple[str, str]]] = {}   # key -> [(object_id, hash)]
        self._seq = 0
        self.frozen_web: Dict[str, dict] = self._seed_frozen_web()

    def _seed_frozen_web(self) -> Dict[str, dict]:
        docs = {
            "paper_eval_repro": {"title": "Reproducible agent evaluation", "topic": "eval_infra",
                                 "summary": "traces + seeds + config hashing for reproducible eval"},
            "blog_api_cost": {"title": "Cutting LLM API cost", "topic": "api_cost",
                              "summary": "batching, caching, cheaper models to reduce api cost"},
            "apidoc_pricing": {"title": "Provider pricing", "topic": "api_cost",
                               "summary": "api price tiers and rate limits"},
        }
        for d in docs.values():
            d["content_hash"] = content_hash(d)
        return docs

    def _key(self, domain: str, query: str, agent_id: str) -> str:
        # The sandbox domain is per-agent PRIVATE, so its cache must not cross
        # agents; shared domains (internal/repo/external/frozen_web) cache
        # globally per corpus_version.
        scope = agent_id if domain == "sandbox" else "*"
        return f"{domain}:{scope}:{self.corpus_version}:{' '.join(sorted(query.lower().split()))}"

    def search(self, *, agent_id: str, query: str, domain: str, tick: int = 0,
               top_k: int = 5) -> SearchLog:
        if domain not in SEARCH_DOMAINS:
            raise ValueError(f"unknown search domain {domain!r}")
        key = self._key(domain, query, agent_id)
        self._seq += 1
        sid = f"search_{self._seq}"

        if key in self._cache:
            ranked = self._cache[key]
            cache_hit = True
        else:
            if self.replay_mode:
                raise ReplaySearchMiss(
                    f"replay cache miss for {domain} query={query!r} "
                    f"corpus={self.corpus_version} (no live search allowed)")
            ranked = self._run(domain, query, agent_id)[:top_k]
            self._cache[key] = ranked
            cache_hit = False

        log = SearchLog(search_id=sid, tick=tick, agent_id=agent_id, query=query,
                        search_domain=domain, corpus_version=self.corpus_version,
                        retrieved_object_ids=[oid for oid, _ in ranked],
                        content_hashes=[h for _, h in ranked], cache_hit=cache_hit)
        self.logs.append(log)
        return log

    def _run(self, domain: str, query: str, agent_id: str) -> List[Tuple[str, str]]:
        w = self.world
        scored: List[Tuple[int, str, str]] = []
        if domain == "internal":
            for tid, t in w.tasks.items():
                s = _match(query, t.title, t.description)
                if s: scored.append((s, tid, content_hash(t.title)))
            for did, d in w.documents.items():
                s = _match(query, d.title, d.content_summary)
                if s: scored.append((s, did, content_hash(d.content_summary)))
        elif domain == "repo":
            for cid, c in w.repo_system.repo.commits.items():
                s = _match(query, c.message, " ".join(c.changed_files))
                if s: scored.append((s, cid, content_hash(c.message)))
            for mod in w.repo_system.repo.modules:
                s = _match(query, mod)
                if s: scored.append((s, mod, content_hash(mod)))
        elif domain == "sandbox":
            sb = w.sandbox_system.sandboxes.get(f"sandbox_{agent_id}")
            for rid in (sb.cached_result_ids if sb else []):
                res = w.sandbox_system.results.get(rid)
                if res and _match(query, res.experiment_id, " ".join(res.metrics.keys())):
                    scored.append((1, rid, res.config_hash))
        elif domain == "external_community":
            for pid, p in w.community.posts.items():
                s = _match(query, p.topic, p.content_summary)
                if s: scored.append((s, pid, content_hash(p.content_summary)))
            for eid, prof in w.community.profiles.items():
                s = _match(query, prof.role, prof.name, " ".join(prof.topic_interests))
                if s: scored.append((s, eid, content_hash(prof.name)))
        elif domain == "frozen_web":
            for did, d in self.frozen_web.items():
                s = _match(query, d["title"], d["topic"], d["summary"])
                if s: scored.append((s, did, d["content_hash"]))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [(oid, h) for _, oid, h in scored]


__all__ = ["SearchSystem", "SearchLog", "ReplaySearchMiss", "SEARCH_DOMAINS"]
