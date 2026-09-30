"""Experience-grounded memory layer (infra task §15-§18).

Each agent has a PRIVATE memory store. Each turn a deterministic retriever
builds a :class:`MemoryContext` =
``RecentMemories ∪ RelevantMemories ∪ SalientMemories ∪ ProspectiveMemories``
(§16), scored deterministically (§18) with per-category budgets and de-dup that
keeps every retrieval reason. This is the read interface PlanMonitor /
DeepReflection / FeatureExtractor / PCBSP consume (§21); it neither selects
actions nor mutates persona.

Env-agnostic and fully deterministic (no LLM) so retrieval is reproducible and
auditable.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set

MEMORY_TYPES = ("episodic_memory", "semantic_memory", "procedural_memory",
                "social_memory", "prospective_memory")
MEMORY_STATUS = ("active", "stale", "contradicted", "reinforced", "archived")

# §16.3 autobiographical events that stay salient even when old.
SALIENT_TAGS = frozenset({
    "rescue", "betrayal", "public_blame", "near_starvation", "successful_prototype",
    "broken_promise", "successful_teaching", "important_rule_trial",
})

# §18 budgets
RECENT_CAP = 10
RELEVANT_CAP = 8
SALIENT_CAP = 4
PROSPECTIVE_CAP = 3
RECENT_WINDOW = 10            # turns
SALIENT_THRESHOLD = 0.7


@dataclass
class MemoryNode:
    memory_id: str
    agent_id: str
    memory_type: str = "episodic_memory"
    summary: str = ""
    source_event_id: Optional[str] = None
    source_episode_id: Optional[str] = None
    related_agents: List[str] = field(default_factory=list)
    related_resources: List[str] = field(default_factory=list)
    related_locations: List[str] = field(default_factory=list)
    related_items: List[str] = field(default_factory=list)
    related_mechanisms: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    emotional_valence: float = 0.0
    intensity: float = 0.0
    salience: float = 0.0
    confidence: float = 1.0
    created_turn: int = 0
    last_retrieved_turn: int = -1
    retrieval_count: int = 0
    decay_rate: float = 0.01
    visibility: str = "private"
    privacy: str = "private"
    reliability: float = 1.0
    status: str = "active"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MemoryRetrievalQuery:
    memory_retrieval_query_id: str
    run_id: str
    turn_id: int
    agent_id: str
    location: Optional[Sequence[float]] = None
    terrain_type: str = ""
    visible_agents: List[str] = field(default_factory=list)
    visible_resources: List[str] = field(default_factory=list)
    visible_objects: List[str] = field(default_factory=list)
    public_marks_seen: List[str] = field(default_factory=list)
    heard_messages: List[str] = field(default_factory=list)
    active_plan_ids: List[str] = field(default_factory=list)
    active_episode_ids: List[str] = field(default_factory=list)
    current_step_ids: List[str] = field(default_factory=list)
    blocked_reasons: List[str] = field(default_factory=list)
    active_mechanisms: List[str] = field(default_factory=list)
    candidate_action_types: List[str] = field(default_factory=list)
    self_state_signals: Dict[str, bool] = field(default_factory=dict)
    recent_event_tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MemoryContext:
    memory_context_id: str
    memory_retrieval_query_id: str
    run_id: str
    turn_id: int
    agent_id: str
    recent_memories: List[str] = field(default_factory=list)
    relevant_memories: List[str] = field(default_factory=list)
    salient_memories: List[str] = field(default_factory=list)
    prospective_memories: List[str] = field(default_factory=list)
    final_memory_context: List[str] = field(default_factory=list)
    dropped_due_to_budget: List[str] = field(default_factory=list)
    deduped_memory_ids: List[str] = field(default_factory=list)
    retrieval_reasons: Dict[str, List[str]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MemoryStore:
    """One agent's private memory store."""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self._nodes: Dict[str, MemoryNode] = {}
        self._seq = 0

    def add(self, node: MemoryNode) -> MemoryNode:
        if not node.memory_id:
            node.memory_id = f"mem:{self.agent_id}:{self._seq}"
        self._seq += 1
        self._nodes[node.memory_id] = node
        return node

    def new(self, **kwargs: Any) -> MemoryNode:
        return self.add(MemoryNode(memory_id=kwargs.pop("memory_id", ""),
                                   agent_id=self.agent_id, **kwargs))

    def get(self, memory_id: str) -> Optional[MemoryNode]:
        return self._nodes.get(memory_id)

    def all(self) -> List[MemoryNode]:
        return [n for n in self._nodes.values() if n.status != "archived"]

    def mark_retrieved(self, memory_id: str, turn: int) -> None:
        n = self._nodes.get(memory_id)
        if n is not None:
            n.last_retrieved_turn = turn
            n.retrieval_count += 1


# --------------------------------------------------------------------------- #
# §18 deterministic scoring
# --------------------------------------------------------------------------- #
def _overlap(a: Sequence[str], b: Sequence[str]) -> int:
    return len(set(a) & set(b))


def score_components(node: MemoryNode, q: MemoryRetrievalQuery, turn: int) -> Dict[str, float]:
    age = max(0, turn - node.created_turn)
    recency = max(0.0, 1.0 - age / float(RECENT_WINDOW))
    tag_overlap = _overlap(node.tags, q.recent_event_tags)
    agent_match = _overlap(node.related_agents, q.visible_agents)
    resource_match = _overlap(node.related_resources, q.visible_resources)
    object_match = _overlap(node.related_items, q.visible_objects)
    loc_match = 0.0
    if node.related_locations and q.terrain_type and q.terrain_type in node.related_locations:
        loc_match = 1.0
    plan_match = _overlap(node.tags, q.active_plan_ids)
    episode_match = 0.0
    if node.source_episode_id and node.source_episode_id in q.active_episode_ids:
        episode_match = 1.0
    episode_match += _overlap(node.tags, q.active_episode_ids)
    mechanism_match = _overlap(node.related_mechanisms, q.active_mechanisms)
    return {
        "recency_score": recency,
        "salience_score": node.salience,
        "tag_overlap_score": float(tag_overlap),
        "related_agent_match": float(agent_match),
        "related_resource_match": float(resource_match),
        "related_location_match": float(loc_match),
        "related_object_match": float(object_match),
        "active_plan_match": float(plan_match),
        "active_episode_match": float(episode_match),
        "mechanism_match": float(mechanism_match),
        "emotional_intensity_bonus": node.intensity * 0.5,
        "decay_penalty": node.decay_rate * age,
    }


def relevance_score(c: Dict[str, float]) -> float:
    return (1.0 * c["tag_overlap_score"] + 1.0 * c["related_agent_match"]
            + 1.0 * c["related_resource_match"] + 0.8 * c["related_location_match"]
            + 0.6 * c["related_object_match"] + 1.0 * c["active_plan_match"]
            + 1.2 * c["active_episode_match"] + 1.0 * c["mechanism_match"]
            + 0.5 * c["emotional_intensity_bonus"] - c["decay_penalty"])


def _relevance_reasons(c: Dict[str, float]) -> List[str]:
    dims = [("tag_overlap_score", "tag"), ("related_agent_match", "agent"),
            ("related_resource_match", "resource"), ("related_location_match", "location"),
            ("related_object_match", "object"), ("active_plan_match", "plan"),
            ("active_episode_match", "episode"), ("mechanism_match", "mechanism")]
    return [f"relevant:{label}" for key, label in dims if c[key] > 0]


# --------------------------------------------------------------------------- #
# Retriever (§16)
# --------------------------------------------------------------------------- #
class MemoryRetriever:
    def build_context(self, store: MemoryStore, query: MemoryRetrievalQuery, *,
                      turn: int, mark_retrieved: bool = True) -> MemoryContext:
        nodes = store.all()
        reasons: Dict[str, List[str]] = {}
        dropped: List[str] = []

        def add_reason(mid: str, reason: str) -> None:
            reasons.setdefault(mid, [])
            if reason not in reasons[mid]:
                reasons[mid].append(reason)

        # §16.1 Recent — recency window, capped (kept even w/o semantic match)
        recent_sorted = sorted([n for n in nodes if (turn - n.created_turn) <= RECENT_WINDOW],
                               key=lambda n: n.created_turn, reverse=True)
        recent = recent_sorted[:RECENT_CAP]
        dropped += [n.memory_id for n in recent_sorted[RECENT_CAP:]]
        for n in recent:
            add_reason(n.memory_id, "recent")

        # §16.2 Relevant — deterministic score vs current context
        scored = []
        for n in nodes:
            c = score_components(n, query, turn)
            rs = relevance_score(c)
            if rs > 0:
                scored.append((rs, n, c))
        scored.sort(key=lambda t: (t[0], -t[1].created_turn), reverse=True)
        relevant = [n for _, n, _ in scored[:RELEVANT_CAP]]
        dropped += [n.memory_id for _, n, _ in scored[RELEVANT_CAP:]]
        for _, n, c in scored[:RELEVANT_CAP]:
            for r in _relevance_reasons(c):
                add_reason(n.memory_id, r)

        # §16.3 Salient — important autobiographical (old ok)
        salient_sorted = sorted(
            [n for n in nodes if n.salience >= SALIENT_THRESHOLD or set(n.tags) & SALIENT_TAGS],
            key=lambda n: n.salience, reverse=True)
        salient = salient_sorted[:SALIENT_CAP]
        dropped += [n.memory_id for n in salient_sorted[SALIENT_CAP:]]
        for n in salient:
            add_reason(n.memory_id, "salient")

        # §16.4 Prospective — unfinished intents / commitments, relevant now
        prosp_candidates = [n for n in nodes
                            if n.memory_type == "prospective_memory" or n.status in ("active",) and
                            any(t in ("promise", "debt", "pending", "unfinished", "return")
                                for t in n.tags)]
        # prioritize prospective tied to current agents/plans/episodes
        def _prosp_key(n: MemoryNode) -> float:
            tie = _overlap(n.related_agents, query.visible_agents) + \
                  _overlap(n.tags, query.active_plan_ids) + \
                  (1 if n.source_episode_id in query.active_episode_ids else 0)
            return (tie, n.salience)
        prosp_sorted = sorted({n.memory_id: n for n in prosp_candidates}.values(),
                              key=_prosp_key, reverse=True)
        prospective = prosp_sorted[:PROSPECTIVE_CAP]
        dropped += [n.memory_id for n in prosp_sorted[PROSPECTIVE_CAP:]]
        for n in prospective:
            add_reason(n.memory_id, "prospective")

        # de-dup by memory_id (keep one, union reasons), priority order
        final: List[str] = []
        seen: Set[str] = set()
        for group in (recent, relevant, salient, prospective):
            for n in group:
                if n.memory_id not in seen:
                    seen.add(n.memory_id)
                    final.append(n.memory_id)
        deduped = [mid for mid in reasons if mid in seen and len(reasons[mid]) > 1]

        if mark_retrieved:
            for mid in final:
                store.mark_retrieved(mid, turn)

        return MemoryContext(
            memory_context_id=f"mc:{query.run_id}:{turn}:{query.agent_id}",
            memory_retrieval_query_id=query.memory_retrieval_query_id,
            run_id=query.run_id, turn_id=turn, agent_id=query.agent_id,
            recent_memories=[n.memory_id for n in recent],
            relevant_memories=[n.memory_id for n in relevant],
            salient_memories=[n.memory_id for n in salient],
            prospective_memories=[n.memory_id for n in prospective],
            final_memory_context=final,
            dropped_due_to_budget=sorted(set(dropped) - seen),
            deduped_memory_ids=sorted(deduped),
            retrieval_reasons={mid: reasons[mid] for mid in final},
        )


def query_from_perception(packet: Any, self_state: Any, *, run_id: str, turn: int,
                          agent_id: str, terrain_type: str = "",
                          candidate_action_types: Optional[List[str]] = None,
                          recent_event_tags: Optional[List[str]] = None) -> MemoryRetrievalQuery:
    """Convenience: build a :class:`MemoryRetrievalQuery` from a PerceptionPacket
    + SelfStatePercept (§17)."""
    p = packet.to_dict() if hasattr(packet, "to_dict") else dict(packet)
    s = self_state.to_dict() if hasattr(self_state, "to_dict") else dict(self_state)
    return MemoryRetrievalQuery(
        memory_retrieval_query_id=f"mq:{run_id}:{turn}:{agent_id}",
        run_id=run_id, turn_id=turn, agent_id=agent_id,
        location=p.get("location"), terrain_type=terrain_type,
        visible_agents=list(p.get("visible_agents", [])),
        visible_resources=[r.get("type") or r.get("id") for r in p.get("visible_resources", [])],
        visible_objects=[o.get("type") or o.get("id") for o in p.get("visible_objects", [])],
        public_marks_seen=[m.get("id") for m in p.get("public_marks_seen", []) if m.get("id")],
        heard_messages=[m.get("message_id") for m in p.get("heard_messages", []) if m.get("message_id")],
        active_plan_ids=list(s.get("active_plan_ids", [])),
        active_episode_ids=list(s.get("active_episode_ids", [])),
        blocked_reasons=list(s.get("blocked_plan_steps", [])),
        candidate_action_types=list(candidate_action_types or []),
        self_state_signals=dict(s.get("derived_signals", {})),
        recent_event_tags=list(recent_event_tags or []),
    )
