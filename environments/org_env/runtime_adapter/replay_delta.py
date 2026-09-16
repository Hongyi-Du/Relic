"""Incremental replay codec (frontend spec — replay size).

A full per-tick snapshot is large (a 336-tick LLM run produced a ~1.5 GB
``replay.json``). The frames are highly redundant tick-to-tick — most of an
8-agent world is unchanged each tick, and the big collections (messages, event
edges, action decisions) are append-only. This module stores a replay as a single
BASE frame plus a per-tick structural DELTA, and reconstructs the full frame list
on demand. Disk + network shrink by 1–2 orders of magnitude; the reconstructed
frames are byte-identical to the originals.

Delta node grammar (compact JSON arrays):
  ["=", value]                 -> replace with value (scalars, rewritten lists)
  ["a", [items]]               -> list append: cur = prev + items (append-only lists)
  ["d", {key: child}, [dels]]  -> dict patch: apply child per key, then delete dels
  None / absent key            -> unchanged
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

FORMAT = "org-delta-v1"

# Heavy DERIVED / DEBUG sections that the live snapshot caps (so they slide every tick
# and defeat append-detection). For replay they are keyframed: stored in full only every
# KEYFRAME_EVERY ticks and carried forward in between (graphs/persona barely move tick to
# tick; this is what shrinks the replay from ~1.5 GB to tens of MB). Per-tick org/agent
# state (agents, internal, product, episodes, ...) is still diffed exactly every tick.
KEYFRAME_KEYS = ("graphs", "logs", "timeline")
KEYFRAME_EVERY = 12

_MISSING = object()


def _diff(prev: Any, cur: Any) -> Optional[list]:
    """Structural delta turning ``prev`` into ``cur`` (None when identical)."""
    if prev is _MISSING:
        return ["=", cur]
    if prev == cur:
        return None
    if isinstance(prev, dict) and isinstance(cur, dict):
        patch: Dict[str, Any] = {}
        for k, v in cur.items():
            child = _diff(prev.get(k, _MISSING), v)
            if child is not None:
                patch[k] = child
        dels = [k for k in prev if k not in cur]
        if not patch and not dels:
            return None
        return ["d", patch, dels]
    if isinstance(prev, list) and isinstance(cur, list):
        # append-only fast path (messages / events / edges grow at the tail)
        if len(cur) > len(prev) and cur[:len(prev)] == prev:
            return ["a", cur[len(prev):]]
        return ["=", cur]
    return ["=", cur]


def apply_delta(prev: Any, delta: Optional[list]) -> Any:
    """Apply a delta node to ``prev`` and return the reconstructed value.

    Unchanged sub-trees are shared (not copied) with ``prev``; reconstruction is
    read-only so the aliasing is safe and keeps memory low across many frames.
    """
    if delta is None:
        return prev
    op = delta[0]
    if op == "=":
        return delta[1]
    if op == "a":
        return (prev or []) + delta[1]
    if op == "d":
        out = dict(prev) if isinstance(prev, dict) else {}
        for k, child in delta[1].items():
            out[k] = apply_delta(out.get(k), child)
        for k in delta[2]:
            out.pop(k, None)
        return out
    return delta[1] if len(delta) > 1 else prev


def build_delta_replay(meta: Dict[str, Any], frames: List[Dict[str, Any]], *,
                       keyframe_keys=KEYFRAME_KEYS, keyframe_every: int = KEYFRAME_EVERY) -> Dict[str, Any]:
    """Pack a frame list into a base + per-tick deltas envelope. Heavy derived/debug
    sections (``keyframe_keys``) are kept only on keyframe ticks and carried forward on
    expand, which is what makes the file small."""
    if not frames:
        return {"meta": meta, "format": FORMAT, "base": None, "deltas": [],
                "keyframe_keys": list(keyframe_keys), "keyframe_every": keyframe_every}
    kk = set(keyframe_keys or ())
    pruned: List[Dict[str, Any]] = []
    for i, fr in enumerate(frames):
        tick = int(fr.get("tick", i) or 0)
        if i == 0 or not kk or tick % keyframe_every == 0:
            pruned.append(fr)                                   # keyframe: keep everything
        else:
            pruned.append({k: v for k, v in fr.items() if k not in kk})
    base = pruned[0]
    deltas = [_diff(pruned[i - 1], pruned[i]) for i in range(1, len(pruned))]
    return {"meta": meta, "format": FORMAT, "base": base, "deltas": deltas,
            "keyframe_keys": list(keyframe_keys), "keyframe_every": keyframe_every}


def expand_delta_replay(replay: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Reconstruct the full frame list. Accepts the delta envelope OR a legacy
    ``{"frames": [...]}`` replay (returned as-is). Keyframed sections are carried
    forward so every frame still has graphs/logs/timeline (keyframe-fresh)."""
    if not isinstance(replay, dict):
        return []
    if replay.get("format") == FORMAT or ("base" in replay and "deltas" in replay):
        base = replay.get("base")
        if base is None:
            return []
        frames = [base]
        cur = base
        for d in (replay.get("deltas") or []):
            cur = apply_delta(cur, d)
            frames.append(cur)
        keys = replay.get("keyframe_keys") or []
        if keys:
            last: Dict[str, Any] = {}
            for fr in frames:
                for k in keys:
                    if k in fr:
                        last[k] = fr[k]
                    elif k in last:
                        fr[k] = last[k]
        return frames
    return replay.get("frames", []) or []


__all__ = ["FORMAT", "apply_delta", "build_delta_replay", "expand_delta_replay"]
