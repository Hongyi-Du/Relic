"""Structured logging layer (infra task §7) — a first-class research component.

Logs are how every downstream module (PlanMonitor, DeepReflection,
FeatureExtractor, PCBSP, detectors) and the replay/audit tooling read what each
agent perceived, remembered, and reasoned, **per agent per turn**. They strictly
separate ground-truth world state from per-agent perception (§20).

This module provides:
  * :class:`Journal` — the central writer/store. Every record is stamped with the
    base fields (run_id / turn_id / schema_version / logical_time / source_module
    / optional agent_id) and appended to a named stream; streams are JSON-
    serializable and dumpable to JSONL.
  * The 7 streams implemented this phase (§7.2): WorldStateLog, PerceptionLog,
    SelfStateLog, PerceptionAppraisalLog, MemoryRetrievalLog, LLMCallLog,
    ReplayTimelineLog — each with a typed convenience writer.
  * The 11 placeholder streams (PlanMonitor … DetectorLog) registered with a
    minimal generic writer so future modules can attach without schema churn.

Env-agnostic: plain data in, plain data out; no world/LLM imports.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = "1.0"

# §7.2 — streams implemented this phase.
IMPLEMENTED_STREAMS = (
    "WorldStateLog", "PerceptionLog", "SelfStateLog", "PerceptionAppraisalLog",
    "MemoryRetrievalLog", "LLMCallLog", "ReplayTimelineLog",
)
# §7.2 — placeholders: minimal writer reserved, filled by later modules.
PLACEHOLDER_STREAMS = (
    "PlanMonitorLog", "ReflectionLog", "CandidatePoolLog", "FeatureExtractionLog",
    "PolicyTraceLog", "ActionExecutionLog", "EventAppraisalLog", "GraphUpdateLog",
    "EpisodeLog", "InstitutionLog", "DetectorLog",
)
# async tick-loop streams (async infra §15, §16, §17, §21)
ASYNC_STREAMS = (
    "ClockLog", "DecisionOrderLog", "TieBreakLog", "LLMJobLog",
)
# Stage C0 direct-takeover streams (nature_env integration §11)
TAKEOVER_STREAMS = (
    "AdapterLog", "ActionMappingLog", "FallbackLog",
)
# Stage C1-C2 main-chain streams (plan / wish / material / craft, spec Part 8).
# PlanMonitorLog + ReflectionLog already live in PLACEHOLDER_STREAMS.
CHAIN_STREAMS = (
    "WishLog", "MaterialKnowledgeLog", "CraftSessionLog", "CraftAttemptLog",
    "PrototypeLog", "PrototypeTestLog", "SkillUpdateLog",
)
ALL_STREAMS = (IMPLEMENTED_STREAMS + PLACEHOLDER_STREAMS + ASYNC_STREAMS
               + TAKEOVER_STREAMS + CHAIN_STREAMS)

# base fields stamped by Journal.record — stripped from payload dicts to avoid
# duplicate-keyword collisions when a record object already carries them.
_BASE_KEYS = ("run_id", "turn_id", "agent_id", "schema_version", "timestamp",
              "logical_time", "source_module")


def _strip_base(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in d.items() if k not in _BASE_KEYS}


@dataclass
class Journal:
    """Central structured-log store for one run.

    ``record(stream, source_module, **fields)`` stamps base fields and appends.
    Use the typed ``log_*`` helpers for the 7 implemented streams; placeholders
    go through ``record`` directly (later modules add typed helpers)."""
    run_id: str = "run"
    schema_version: str = SCHEMA_VERSION
    streams: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    _logical_time: int = 0

    # -- core writer --------------------------------------------------------
    def record(self, stream: str, *, source_module: str, turn_id: Optional[int] = None,
               agent_id: Optional[str] = None, **fields: Any) -> Dict[str, Any]:
        if stream not in ALL_STREAMS:
            raise ValueError(f"unknown log stream {stream!r}; known: {ALL_STREAMS}")
        self._logical_time += 1
        rec: Dict[str, Any] = {
            "run_id": self.run_id,
            "turn_id": turn_id,
            "schema_version": self.schema_version,
            "timestamp": time.time(),
            "logical_time": self._logical_time,
            "source_module": source_module,
        }
        if agent_id is not None:
            rec["agent_id"] = agent_id
        rec.update(fields)
        self.streams.setdefault(stream, []).append(rec)
        return rec

    def stream(self, name: str) -> List[Dict[str, Any]]:
        return self.streams.get(name, [])

    def for_agent(self, stream: str, agent_id: str) -> List[Dict[str, Any]]:
        return [r for r in self.stream(stream) if r.get("agent_id") == agent_id]

    def for_turn(self, stream: str, turn_id: int) -> List[Dict[str, Any]]:
        return [r for r in self.stream(stream) if r.get("turn_id") == turn_id]

    # -- §7.2 typed writers (the 7 implemented streams) --------------------
    def log_world_state(self, *, turn_id: int, source_module: str = "world",
                        **fields: Any) -> Dict[str, Any]:
        """Ground truth (§20): the real world state, independent of who saw it."""
        return self.record("WorldStateLog", source_module=source_module, turn_id=turn_id, **fields)

    def log_perception(self, *, turn_id: int, agent_id: str, packet: Any,
                       source_module: str = "perception") -> Dict[str, Any]:
        d = packet.to_dict() if hasattr(packet, "to_dict") else dict(packet)
        keep = ("perception_packet_id", "visible_resources", "visible_objects",
                "visible_agents", "visible_agent_actions", "heard_messages",
                "overheard_messages", "public_marks_seen", "public_records_seen",
                "visible_mechanisms", "visible_episode_events", "self_state_percept_id",
                "salient_changes", "uncertainty_flags")
        return self.record("PerceptionLog", source_module=source_module, turn_id=turn_id,
                           agent_id=agent_id, **{k: d.get(k) for k in keep})

    def log_self_state(self, *, turn_id: int, agent_id: str, percept: Any,
                       source_module: str = "perception") -> Dict[str, Any]:
        d = percept.to_dict() if hasattr(percept, "to_dict") else dict(percept)
        keep = ("self_state_percept_id", "energy", "hp", "hunger_pressure", "fatigue",
                "stress", "carrying_capacity", "current_load", "remaining_capacity",
                "inventory_items", "home_storage_summary", "active_plan_ids",
                "active_episode_ids", "derived_signals")
        rec = {k: d.get(k) for k in keep}
        rec["inventory_summary"] = d.get("inventory_items")
        return self.record("SelfStateLog", source_module=source_module, turn_id=turn_id,
                           agent_id=agent_id, **rec)

    def log_perception_appraisal(self, *, turn_id: int, agent_id: str, appraisal: Any,
                                 source_module: str = "perception") -> Dict[str, Any]:
        d = appraisal.to_dict() if hasattr(appraisal, "to_dict") else dict(appraisal)
        return self.record("PerceptionAppraisalLog", source_module=source_module,
                           turn_id=turn_id, agent_id=agent_id, **_strip_base(d))

    def log_memory_retrieval(self, *, turn_id: int, agent_id: str, context: Any,
                             source_module: str = "memory") -> Dict[str, Any]:
        d = context.to_dict() if hasattr(context, "to_dict") else dict(context)
        return self.record("MemoryRetrievalLog", source_module=source_module,
                           turn_id=turn_id, agent_id=agent_id, **_strip_base(d))

    def log_llm_call(self, record: Dict[str, Any], *, source_module: str = "llm_engine") -> Dict[str, Any]:
        """Auto-written by the LLM Engine (§8). ``record`` already carries the
        full LLMCallLog field set; we stamp base fields around it."""
        turn_id = record.pop("turn_id", None)
        agent_id = record.pop("agent_id", None)
        return self.record("LLMCallLog", source_module=source_module, turn_id=turn_id,
                           agent_id=agent_id, **record)

    def log_replay_timeline(self, *, turn_id: int, text: str, agent_id: Optional[str] = None,
                            source_module: str = "replay", **fields: Any) -> Dict[str, Any]:
        """Human-readable important-event summary (§19.6)."""
        return self.record("ReplayTimelineLog", source_module=source_module, turn_id=turn_id,
                           agent_id=agent_id, text=text, **fields)

    # -- replay / audit -----------------------------------------------------
    def replay_timeline(self) -> List[str]:
        return [f"[t{r.get('turn_id')}] {r.get('text','')}" for r in self.stream("ReplayTimelineLog")]

    def to_dict(self) -> Dict[str, Any]:
        return {"run_id": self.run_id, "schema_version": self.schema_version,
                "streams": self.streams}

    def to_jsonl(self, path: str) -> str:
        with open(path, "w", encoding="utf-8") as fh:
            for stream, recs in self.streams.items():
                for r in recs:
                    fh.write(json.dumps({"stream": stream, **r}, ensure_ascii=False) + "\n")
        return path
