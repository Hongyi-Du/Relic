"""Random event pool for the external society.

Each day an external user has a chance to encounter an event from this pool — a personal life
event, a world/market event, or a professional insight worth sharing. The event is what the user's
LLM (or the offline fallback) reasons over to decide whether/what to post. We prefer a REAL world-
event dataset (e.g. HF `Reubencf/2024_events`) at a local data dir; otherwise a curated bundled
pool keeps the society reproducible offline.

Sampling is always driven by a caller-provided seeded RNG, so every "random group" is reproducible.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from random import Random
from typing import List, Optional

_DEFAULT_DATASET_DIR = os.path.join("environments", "org_env", "data", "external", "events")

# event kinds
LIFE, WORLD, INSIGHT = "life_event", "world_event", "insight"


@dataclass(frozen=True)
class EventRecord:
    event_id: str
    kind: str          # life_event | world_event | insight
    text: str
    topic: str = ""    # forum topic this event is about ("" = general / off-topic)


# -- curated bundled pool (reproducible offline) -------------------------------------------------
# (kind, topic, text). topics map onto the forum so events produce topical chatter.
_BUNDLED: tuple = (
    (LIFE, "hiring_market", "started a new job leading an applied-ML team"),
    (LIFE, "api_cost", "got an alarming cloud bill and is auditing every API call"),
    (LIFE, "agent_reliability", "had an agent fail silently in production overnight"),
    (LIFE, "trace_debugging", "spent the day bisecting a flaky multi-step agent run"),
    (LIFE, "benchmark_quality", "is preparing a talk on honest evaluation"),
    (LIFE, "startup_funding", "is in the middle of a tense fundraising week"),
    (LIFE, "", "is moving to a new city and feeling scattered"),
    (LIFE, "", "just got back from a long-overdue vacation, inbox on fire"),
    (LIFE, "customer_pain", "had a customer churn citing 'we couldn't trust the output'"),
    (LIFE, "reproducibility_tracking", "couldn't reproduce last quarter's eval numbers"),
    (WORLD, "api_cost", "a major LLM provider announced a price increase"),
    (WORLD, "competitor_update", "a well-funded competitor shipped a flashy eval demo"),
    (WORLD, "startup_funding", "reports say AI-infra funding is tightening this quarter"),
    (WORLD, "benchmark_quality", "a popular benchmark was shown to be gameable"),
    (WORLD, "agent_reliability", "a viral thread documented agents failing on long tasks"),
    (WORLD, "hiring_market", "another round of tech layoffs hit the news"),
    (WORLD, "", "a big conference announced its keynote lineup"),
    (INSIGHT, "api_cost", "realized batching + caching eval calls cuts cost ~4x"),
    (INSIGHT, "reproducibility_tracking", "learned that pinning seeds + config hashes saves hours"),
    (INSIGHT, "trace_debugging", "found that full tool-call traces beat summaries for debugging"),
    (INSIGHT, "benchmark_quality", "concluded per-task human spot-checks beat aggregate scores"),
    (INSIGHT, "customer_pain", "noticed buyers care about verifiable sources over features"),
    (INSIGHT, "eval_infra", "thinks reproducible eval infra is the real moat"),
)


class EventPool:
    def __init__(self, records: Optional[List[EventRecord]] = None):
        self.records: List[EventRecord] = records or [
            EventRecord(event_id=f"evt_seed_{i}", kind=k, text=t, topic=top)
            for i, (k, top, t) in enumerate(_BUNDLED)
        ]

    def sample(self, rng: Random) -> EventRecord:
        return self.records[rng.randrange(len(self.records))]

    def sample_topic(self, rng: Random, topic: str) -> EventRecord:
        """An event constrained to a topic (used by controlled interventions)."""
        pool = [r for r in self.records if r.topic == topic] or self.records
        return pool[rng.randrange(len(pool))]


def _load_world_events(dataset_dir: str, limit: int = 2000) -> Optional[List[EventRecord]]:
    if not os.path.isdir(dataset_dir):
        return None
    out: List[EventRecord] = []
    for fn in sorted(os.listdir(dataset_dir)):
        path = os.path.join(dataset_dir, fn)
        if fn.endswith(".jsonl"):
            with open(path, encoding="utf-8") as fh:
                for ln in fh:
                    if not ln.strip():
                        continue
                    rec = json.loads(ln)
                    text = str(rec.get("event") or rec.get("text") or rec.get("title") or "").strip()
                    if text:
                        out.append(EventRecord(event_id=f"evt_world_{len(out)}", kind=WORLD, text=text))
        elif fn.endswith(".parquet"):
            try:
                import pandas as pd
            except Exception:
                continue
            df = pd.read_parquet(path)
            col = next((c for c in ("event", "text", "title") if c in df.columns), None)
            if col:
                for v in df[col].astype(str).tolist():
                    if v.strip():
                        out.append(EventRecord(event_id=f"evt_world_{len(out)}", kind=WORLD, text=v.strip()))
        if len(out) >= limit:
            break
    return out[:limit] or None


def build_event_pool(dataset_dir: Optional[str] = None, include_bundled: bool = True) -> EventPool:
    """Build the event pool. Merges real world events (if a dataset is present) with the bundled
    professional life-events/insights so the pool always covers the company's domain."""
    path = dataset_dir or os.environ.get("ORG_EVENTS_DIR") or _DEFAULT_DATASET_DIR
    world = _load_world_events(path) or []
    pool = EventPool()                              # bundled
    if world:
        pool.records = (pool.records if include_bundled else []) + world
    return pool


__all__ = ["EventRecord", "EventPool", "build_event_pool", "LIFE", "WORLD", "INSIGHT"]
