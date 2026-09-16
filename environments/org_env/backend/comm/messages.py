"""Messages + Attachments (DESIGN env_org §17/§18/§38).

Attachments reference objects (file/PR/result/external post) — they never copy
content (content_hash + object_id only). Messages carry read_by/acknowledged_by
so information asymmetry holds (§19): an agent only "knows" a message after
reading it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Set


@dataclass
class Attachment:
    attachment_id: str
    attachment_type: str   # doc|file|json_result|experiment_report|pr|issue|task|external_post|external_doc|search_result|cost_report|benchmark_config|sandbox_output|protocol_proposal|chart_summary
    object_id: str
    title: str = ""
    summary: str = ""
    content_hash: str = ""
    source: str = ""
    created_tick: int = 0


@dataclass
class Message:
    message_id: str
    sender_id: str
    channel_id: Optional[str] = None
    dm_id: Optional[str] = None
    recipients: List[str] = field(default_factory=list)
    created_tick: int = 0
    text_summary: str = ""
    full_text: Optional[str] = None
    attachments: List[Attachment] = field(default_factory=list)
    linked_objects: List[str] = field(default_factory=list)
    mentions: List[str] = field(default_factory=list)
    importance: str = "useful"   # trivial|useful|decision_relevant|blocker|policy_relevant
    urgency: str = "normal"      # low|normal|high|urgent|incident
    visibility: str = "channel"
    read_by: Set[str] = field(default_factory=set)
    acknowledged_by: Set[str] = field(default_factory=set)
    reply_to_message_id: Optional[str] = None
    thread_id: Optional[str] = None
    source_provenance: str = ""

    def carries_signal(self) -> bool:
        """True if this message brings an external/decision-relevant object that
        becomes a percept for readers (e.g. a shared external post)."""
        return bool(self.attachments) or self.importance in ("decision_relevant", "blocker", "policy_relevant")


__all__ = ["Attachment", "Message"]
