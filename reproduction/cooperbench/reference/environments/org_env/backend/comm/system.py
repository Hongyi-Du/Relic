"""CommunicationSystem (DESIGN env_org §16-§20/§38).

Real behaviour: channels + messages + attachments + read receipts. The system
ENFORCES information asymmetry (acceptance ㉗): a message is only *perceivable*
by channel members, and an agent only *knows* it after reading (``read_by``).
Sharing an object posts a Message with an Attachment (reference, not a copy).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set

from environments.org_env.backend.comm.channels import DEFAULT_CHANNELS, Channel
from environments.org_env.backend.comm.messages import Attachment, Message


class CommunicationSystem:
    def __init__(self):
        self.channels: Dict[str, Channel] = {}
        self.dms: Dict[str, List[str]] = {}        # dm_id -> [agent_a, agent_b]
        self.messages: Dict[str, Message] = {}
        self._seq = 0

    # -- setup --------------------------------------------------------------
    def create_channel(self, channel_id: str, channel_type: str = "team_general",
                       members: Optional[Set[str]] = None) -> Channel:
        ch = Channel(channel_id=channel_id, channel_type=channel_type,
                     members=set(members or ()))
        self.channels[channel_id] = ch
        return ch

    def seed_default_channels(self, members: Set[str]) -> None:
        for c in DEFAULT_CHANNELS:
            self.create_channel(c, channel_type=c, members=set(members))

    def join(self, channel_id: str, agent_id: str) -> None:
        ch = self.channels.get(channel_id)
        if ch:
            ch.add_member(agent_id)

    # -- messaging ----------------------------------------------------------
    def send_message(self, *, sender_id: str, channel_id: str, text: str,
                     tick: int = 0, attachments: Optional[List[Attachment]] = None,
                     mentions: Optional[List[str]] = None, importance: str = "useful",
                     urgency: str = "normal", linked_objects: Optional[List[str]] = None) -> Message:
        self._seq += 1
        mid = f"msg_{self._seq}"
        ch = self.channels.get(channel_id)
        recipients = sorted(ch.members) if ch else []
        m = Message(message_id=mid, sender_id=sender_id, channel_id=channel_id,
                    recipients=recipients, created_tick=tick, text_summary=text[:200],
                    full_text=text, attachments=list(attachments or []),
                    mentions=list(mentions or []), importance=importance, urgency=urgency,
                    linked_objects=list(linked_objects or []))
        m.read_by.add(sender_id)   # sender has seen their own message
        self.messages[mid] = m
        if ch:
            ch.message_ids.append(mid)
        return m

    def share_object(self, *, sender_id: str, channel_id: str, object_id: str,
                     attachment_type: str, title: str = "", content_hash: str = "",
                     source: str = "", tick: int = 0, importance: str = "decision_relevant",
                     text: str = "") -> Message:
        """Post a Message referencing an object (no content copy) — §18/§38."""
        att = Attachment(attachment_id=f"att_{self._seq + 1}", attachment_type=attachment_type,
                         object_id=object_id, title=title, content_hash=content_hash,
                         source=source, created_tick=tick)
        return self.send_message(sender_id=sender_id, channel_id=channel_id,
                                 text=text or f"shared {attachment_type}: {title}", tick=tick,
                                 attachments=[att], importance=importance,
                                 linked_objects=[object_id])

    # -- read receipts / asymmetry -----------------------------------------
    def mark_read(self, agent_id: str, message_id: str) -> bool:
        m = self.messages.get(message_id)
        if m is None:
            return False
        ch = self.channels.get(m.channel_id) if m.channel_id else None
        # can only read if a channel member (or direct recipient / mentioned)
        if ch is not None and agent_id not in ch.members and agent_id not in m.mentions:
            return False
        m.read_by.add(agent_id)
        return True

    def acknowledge(self, agent_id: str, message_id: str) -> bool:
        if self.mark_read(agent_id, message_id):
            self.messages[message_id].acknowledged_by.add(agent_id)
            return True
        return False

    def perceivable_messages(self, agent_id: str) -> List[Message]:
        """Messages an agent COULD see (channel member). Not the same as known
        — knowledge requires having read it (memory uses read messages)."""
        out: List[Message] = []
        for ch in self.channels.values():
            if agent_id in ch.members:
                out.extend(self.messages[mid] for mid in ch.message_ids if mid in self.messages)
        return out

    def triage_inbox(self, agent_id: str, tick: int = 0, cap: int = 25) -> List[Message]:
        """v14 P3: an agent processes its inbox — mark recent UNREAD perceivable messages as READ
        (so read receipts reflect reality, not read_by=1), and ACKNOWLEDGE the ones that matter
        (decision-relevant / urgent / @mentions). Returns the newly-read messages so the caller
        can write them into the agent's memory (message -> read -> memory -> influence)."""
        newly_read: List[Message] = []
        for m in sorted(self.perceivable_messages(agent_id), key=lambda x: -int(x.created_tick))[:cap]:
            if agent_id in m.read_by or m.sender_id == agent_id:
                continue
            m.read_by.add(agent_id)
            # acknowledge the ones DIRECTED at this agent (urgent / incident / @mention); ordinary
            # decision-relevant traffic is read (knowledge) but left for an explicit reply.
            if m.urgency in ("urgent", "incident") or agent_id in m.mentions:
                m.acknowledged_by.add(agent_id)
            newly_read.append(m)
        return newly_read

    def known_messages(self, agent_id: str) -> List[Message]:
        """Messages the agent has actually read -> eligible for memory (§19)."""
        return [m for m in self.messages.values() if agent_id in m.read_by]

    def agents_who_know(self, message_id: str) -> Set[str]:
        m = self.messages.get(message_id)
        return set(m.read_by) if m else set()


__all__ = ["CommunicationSystem"]
