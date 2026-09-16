"""Actor-scoped Cooper feature briefs and explicit message attachment receipts.

Private source text stays outside generic tasks, artifacts and public packs.
Publishing attaches a reference only after the owner selected a real message
action. Reading that exact attachment is a separate action, not inbox triage.
"""
from __future__ import annotations

import hashlib
from typing import Any, Mapping

from environments.org_env.backend.comm.messages import Attachment


_STORE = "_cooperbench_private_briefs"
_ATTACHMENT_TYPE = "cooperbench_feature_brief"


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def is_strict_coop(world: Any) -> bool:
    return _STORE in getattr(world, "__dict__", {})


def initialize_private_briefs(
    world: Any, briefs: Mapping[str, str], owners: Mapping[str, str],
) -> None:
    """Initialize immutable owner sources; never publish or read on their behalf."""
    if (len(briefs) != 2 or set(briefs) != set(owners)
            or len(set(owners.values())) != 2
            or any(not isinstance(key, str) or not key for key in briefs)
            or any(not isinstance(text, str) or not text.strip() for text in briefs.values())
            or any(not isinstance(owner, str) or not owner for owner in owners.values())):
        raise ValueError("cooperbench_private_briefs_invalid")
    sources = {
        feature_id: {"owner_id": owners[feature_id], "text": text, "digest": _digest(text)}
        for feature_id, text in briefs.items()
    }
    if is_strict_coop(world):
        existing = getattr(world, _STORE)
        if isinstance(existing, dict) and existing.get("briefs") == sources:
            return
        raise ValueError("cooperbench_private_briefs_already_initialized")
    world.__dict__[_STORE] = {
        "schema_version": "cooperbench_private_brief_visibility_v1",
        "briefs": sources, "shares": {}, "reads": {},
    }


def _source(world: Any, feature_id: str) -> Mapping[str, Any] | None:
    state = getattr(world, _STORE, None)
    if not isinstance(state, Mapping):
        return None
    source = (state.get("briefs") or {}).get(feature_id)
    if (not isinstance(source, Mapping) or not isinstance(source.get("text"), str)
            or source.get("digest") != _digest(source["text"])):
        return None
    return source


def _message_for_share(world: Any, sender_id: str, recipient_id: str, message_id: str):
    comm = getattr(world, "comm", None)
    message = (getattr(comm, "messages", {}) or {}).get(message_id)
    if (message is None or getattr(message, "sender_id", None) != sender_id
            or recipient_id not in (getattr(message, "recipients", []) or [])):
        raise ValueError("cooperbench_brief_share_message_mismatch")
    channel_id = getattr(message, "channel_id", None)
    channel = (getattr(comm, "channels", {}) or {}).get(channel_id)
    if channel is None or not {sender_id, recipient_id} <= set(channel.members):
        raise ValueError("cooperbench_brief_share_channel_denied")
    return message


def _validated_share(world: Any, reader_id: str, attachment_id: str, message_id: str):
    state = getattr(world, _STORE, {}) or {}
    share = (state.get("shares") or {}).get(attachment_id)
    if (not isinstance(share, Mapping) or share.get("recipient_id") != reader_id
            or share.get("message_id") != message_id):
        raise ValueError("cooperbench_brief_share_not_available")
    source = _source(world, str(share.get("feature_id") or ""))
    if (source is None or source.get("owner_id") != share.get("owner_id")
            or source.get("digest") != share.get("digest")):
        raise ValueError("cooperbench_brief_share_digest_mismatch")
    message = _message_for_share(world, share["owner_id"], reader_id, message_id)
    matches = [attachment for attachment in (getattr(message, "attachments", []) or [])
               if getattr(attachment, "attachment_id", None) == attachment_id]
    if (len(matches) != 1 or matches[0].attachment_type != _ATTACHMENT_TYPE
            or matches[0].object_id != share["feature_id"]
            or matches[0].content_hash != share["digest"]
            or matches[0].source != _ATTACHMENT_TYPE):
        raise ValueError("cooperbench_brief_share_attachment_mismatch")
    return share, source, message


def brief_visibility_receipt(world: Any, actor_id: str, feature_id: str) -> dict | None:
    if not is_strict_coop(world):
        text = visible_feature_brief(world, actor_id, feature_id)
        return ({"source": "legacy_public_task", "feature_id": feature_id,
                 "actor_id": actor_id, "digest": _digest(text)} if text else None)
    source = _source(world, feature_id)
    if source is None or not actor_id:
        return None
    if source["owner_id"] == actor_id:
        return {"source": "owner_private", "feature_id": feature_id,
                "owner_id": actor_id, "actor_id": actor_id, "digest": source["digest"]}
    receipt = ((getattr(world, _STORE).get("reads") or {}).get(actor_id) or {}).get(feature_id)
    if not isinstance(receipt, Mapping):
        return None
    try:
        share, _, _ = _validated_share(world, actor_id, receipt.get("attachment_id"), receipt.get("message_id"))
    except ValueError:
        return None
    if (receipt.get("actor_id") != actor_id or receipt.get("source") != "explicit_message_read"
            or receipt.get("digest") != source["digest"]
            or any(receipt.get(key) != value for key, value in share.items())
            or not isinstance(receipt.get("read_tick"), int)
            or receipt["read_tick"] < share["shared_tick"]):
        return None
    return dict(receipt)


def visible_feature_brief(world: Any, actor_id: str, feature_id: str) -> str | None:
    if not is_strict_coop(world):
        task = (getattr(world, "tasks", {}) or {}).get(f"task_oss_{feature_id}")
        text = str(getattr(task, "description", "") or "")
        return text if text.strip() else None
    if brief_visibility_receipt(world, actor_id, feature_id) is None:
        return None
    source = _source(world, feature_id)
    return source["text"] if source is not None else None


def visible_brief_context(world: Any, actor_id: str) -> dict:
    if is_strict_coop(world):
        features = (getattr(world, _STORE, {}) or {}).get("briefs", {})
    else:
        features = {str(task_id).removeprefix("task_oss_"): None
                    for task_id in (getattr(world, "tasks", {}) or {})
                    if str(task_id).startswith("task_oss_cooper_feature_")}
    result = {}
    for feature_id in sorted(features):
        text = visible_feature_brief(world, actor_id, feature_id)
        if text is not None:
            result[feature_id] = {"description": text,
                                  "visibility_receipt": brief_visibility_receipt(world, actor_id, feature_id)}
    return result


def share_feature_brief(
    world: Any, sender_id: str, feature_id: str, recipient_id: str,
    message_id: str, tick: int,
) -> dict:
    """Attach one owner-selected immutable brief to an already-sent message."""
    source = _source(world, feature_id)
    state = getattr(world, _STORE, {}) or {}
    members = {row["owner_id"] for row in (state.get("briefs") or {}).values()}
    if (source is None or source["owner_id"] != sender_id
            or recipient_id not in members or recipient_id == sender_id):
        raise ValueError("cooperbench_brief_share_owner_or_recipient_denied")
    message = _message_for_share(world, sender_id, recipient_id, message_id)
    if type(tick) is not int or tick < int(getattr(message, "created_tick", 0)):
        raise ValueError("cooperbench_brief_share_tick_invalid")
    identity = "\0".join((message_id, feature_id, sender_id, recipient_id, source["digest"]))
    attachment_id = "cooperbrief_" + _digest(identity)[:24]
    shares = state["shares"]
    if attachment_id in shares:
        share, _, _ = _validated_share(world, recipient_id, attachment_id, message_id)
        return dict(share)
    receipt = {"attachment_id": attachment_id, "feature_id": feature_id,
               "owner_id": sender_id, "recipient_id": recipient_id,
               "digest": source["digest"], "message_id": message_id, "shared_tick": tick}
    message.attachments.append(Attachment(
        attachment_id=attachment_id, attachment_type=_ATTACHMENT_TYPE,
        object_id=feature_id, title="Shared Cooper feature brief",
        content_hash=source["digest"], source=_ATTACHMENT_TYPE, created_tick=tick,
    ))
    if feature_id not in message.linked_objects:
        message.linked_objects.append(feature_id)
    shares[attachment_id] = receipt
    return dict(receipt)


def read_shared_feature_brief(
    world: Any, reader_id: str, attachment_id: str, message_id: str, tick: int,
) -> str:
    """Read only the exact attachment addressed to this member; return full text."""
    share, source, message = _validated_share(world, reader_id, attachment_id, message_id)
    if type(tick) is not int or tick < share["shared_tick"]:
        raise ValueError("cooperbench_brief_read_tick_invalid")
    reads = getattr(world, _STORE)["reads"].setdefault(reader_id, {})
    existing = brief_visibility_receipt(world, reader_id, share["feature_id"])
    if existing is None or existing.get("attachment_id") != attachment_id:
        reads[share["feature_id"]] = {**share, "source": "explicit_message_read",
                                      "actor_id": reader_id, "read_tick": tick}
    message.read_by.add(reader_id)
    return source["text"]


def pending_brief_shares(world: Any, actor_id: str) -> list[dict]:
    if not is_strict_coop(world):
        return []
    pending = []
    for attachment_id, share in (getattr(world, _STORE).get("shares") or {}).items():
        if share.get("recipient_id") != actor_id:
            continue
        try:
            _validated_share(world, actor_id, attachment_id, share["message_id"])
        except ValueError:
            continue
        if brief_visibility_receipt(world, actor_id, share["feature_id"]) is None:
            pending.append(dict(share))
    return sorted(pending, key=lambda row: (row["shared_tick"], row["attachment_id"]))


__all__ = [
    "is_strict_coop", "initialize_private_briefs", "visible_feature_brief",
    "brief_visibility_receipt", "visible_brief_context", "share_feature_brief",
    "read_shared_feature_brief", "pending_brief_shares",
]
