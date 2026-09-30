"""Structured context window for LLM agents."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ContextEntry:
    """Single entry in the agent's context window."""
    role: str               # "system" | "user" | "assistant"
    content: str            # Actual text sent to LLM
    tag: str = "unknown"    # Semantic tag for management
    source: str = "core"    # "core" | "mechanism" | "env_overlay" | "agent"
    compactable: bool = True
    turn: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class AgentContext:
    """Sequential, structured context window.

    Wraps the classic message_history list with metadata for debugging,
    compaction, and layer tracing. .render() produces the LLM-ready list.
    """

    def __init__(self):
        self.entries: List[ContextEntry] = []

    def append(
        self,
        role: str,
        content: str,
        tag: str = "unknown",
        source: str = "core",
        compactable: bool = True,
        turn: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.entries.append(ContextEntry(
            role=role, content=content, tag=tag,
            source=source, compactable=compactable, turn=turn,
            metadata=metadata if metadata is not None else {},
        ))

    def add(self, content: str, tag: str = "unknown",
            cross_gen: bool = False, turn: Optional[int] = None,
            **metadata) -> None:
        """Convenience: adds a system/env entry with optional cross_gen flag."""
        meta = dict(metadata)
        if cross_gen:
            meta["cross_gen"] = True
        self.append(role="system", content=content, tag=tag,
                    source="env", turn=turn, metadata=meta)

    def inherit(self) -> "AgentContext":
        """Extract cross_gen=True entries into a new AgentContext for child agent."""
        child_ctx = AgentContext()
        for entry in self.entries:
            if entry.metadata.get("cross_gen"):
                child_ctx.entries.append(ContextEntry(
                    role=entry.role, content=entry.content,
                    tag=entry.tag, source=entry.source,
                    compactable=entry.compactable, turn=entry.turn,
                    metadata=dict(entry.metadata),
                ))
        return child_ctx

    def render(self) -> List[dict]:
        """Return LLM-ready messages list — identical to legacy message_history format."""
        return [{"role": e.role, "content": e.content} for e in self.entries]

    def render_for_debug(self) -> str:
        """Return tagged, human-readable version for debugging."""
        lines = []
        for e in self.entries:
            lines.append(f"[{e.tag}] [{e.source}] ({e.role}) turn={e.turn}")
            lines.append(e.content)
            lines.append("")
        return "\n".join(lines)

    def compact(self, summary: str) -> None:
        """Replace all compactable entries with a single summary entry."""
        non_compactable = [e for e in self.entries if not e.compactable]
        non_compactable.append(ContextEntry(
            role="system", content=summary,
            tag="compacted_context", source="agent",
            compactable=False,
        ))
        self.entries = non_compactable

    def slice(self, tag: str = None, source: str = None) -> List[ContextEntry]:
        """Filter entries by tag and/or source."""
        result = self.entries
        if tag is not None:
            result = [e for e in result if e.tag == tag]
        if source is not None:
            result = [e for e in result if e.source == source]
        return result

    @classmethod
    def from_message_history(cls, history: List[dict]) -> "AgentContext":
        """Import a legacy message_history list into AgentContext."""
        ctx = cls()
        for msg in history:
            role = msg.get("role", "user")
            tag = "system_prompt" if role == "system" else "legacy"
            ctx.append(
                role=role,
                content=msg.get("content", ""),
                tag=tag,
                source="core",
                compactable=(role != "system"),
            )
        return ctx

    def clear_compactable(self) -> None:
        """Remove all compactable entries (used after compaction writes summary)."""
        self.entries = [e for e in self.entries if not e.compactable]

    def __len__(self) -> int:
        return len(self.entries)
