"""APC (RadixAttention) prefix-reuse probe for SGLang / vLLM-style backends.

Records per-LLM-call metadata (hash of system+tools+prior_msgs, token counts,
latency, per-agent ctx) to answer: is the serving stack's automatic prefix
cache reusing KV across agents (same bucket, same turn) and across turns
(same agent, growing history)?

Design reference:
  docs/superpowers/specs/2026-04-20/ (brainstorming conversation 2026-04-20)
  Brainstorm-scoped design; not a full spec doc by user preference.

Contract:
  - `record(system, tools, messages, response, latency_s)` is called once per
    LLM round-trip, after the provider returns.
  - Per-agent/turn/phase/iteration ctx is propagated via
    `extend_probe_ctx(**kwargs)` (contextvars-based → asyncio-task isolated).
  - Both hooks are no-ops unless `APC_PROBE_ENABLED=1`.
  - `APC_PROBE_DUMP=1` additionally writes raw prompt JSON per call to
    `$APC_PROBE_DIR/apc_probe_raw/call_<seq>.json`.
  - `flush()` writes `_probe_list` to a single JSON file and is safe to call
    multiple times; atexit-registered so runners never need to call it.

Failure isolation: any internal raise is caught, logged as warning, and
swallowed. The probe never breaks the LLM control path.
"""
from __future__ import annotations

import atexit
import contextlib
import contextvars
import datetime
import hashlib
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional, Sequence

_log = logging.getLogger(__name__)


# ---------- global state ---------------------------------------------------

# Per-asyncio-task metadata stack for probe records. Natively isolated per
# Task — two agents running concurrently see independent ctx.
_PROBE_CTX: contextvars.ContextVar[dict] = contextvars.ContextVar(
    "apc_probe_ctx", default={}
)

_probe_list: list[dict] = []
_list_lock = threading.Lock()
_seq_counter = 0
_flushed_paths: list[str] = []
# Stable output path for the current process — computed once on first flush
# and reused so incremental flushes overwrite the same file instead of
# generating a new timestamped file per call. Set to None here; assigned
# lazily in ``_resolve_session_path()``.
_session_path: Optional[str] = None
# Count of entries the LAST flush wrote. Used by ``_maybe_incremental_flush``
# to avoid rewriting the file when nothing new has been recorded since.
_last_flushed_count = 0


def reset_for_test() -> None:
    """Clear accumulator + seq + session path. Only used by the test suite."""
    global _probe_list, _seq_counter, _flushed_paths, _session_path, _last_flushed_count
    with _list_lock:
        _probe_list = []
        _seq_counter = 0
        _flushed_paths = []
        _session_path = None
        _last_flushed_count = 0
    _PROBE_CTX.set({})


# ---------- env-var gates --------------------------------------------------


def _enabled() -> bool:
    return os.environ.get("APC_PROBE_ENABLED") == "1"


def _dump_enabled() -> bool:
    return os.environ.get("APC_PROBE_DUMP") == "1"


def _out_dir() -> Path:
    d = os.environ.get("APC_PROBE_DIR", "logs")
    p = Path(d)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _flush_every() -> int:
    """Number of new ``record()`` calls after which to incrementally flush
    the JSON to disk. Default 20. Set ``APC_PROBE_FLUSH_EVERY=0`` to disable
    incremental flushing entirely (then only atexit flushes, as before)."""
    try:
        return max(0, int(os.environ.get("APC_PROBE_FLUSH_EVERY", "20")))
    except (TypeError, ValueError):
        return 20


def _resolve_session_path() -> str:
    """Return the stable output path for this process. Computed lazily on
    first call (ensures consistent timestamp even if APC_PROBE_DIR changes
    mid-run — we lock the path at first record)."""
    global _session_path
    if _session_path is None:
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        _session_path = str(_out_dir() / f"apc_probe_{stamp}.json")
    return _session_path


# ---------- ContextVar helpers ---------------------------------------------


def get_probe_ctx() -> dict:
    """Return the current probe ctx snapshot (empty dict if unset)."""
    return dict(_PROBE_CTX.get())


@contextlib.contextmanager
def extend_probe_ctx(**kwargs: Any) -> Iterator[None]:
    """Merge ``kwargs`` into the current probe ctx for the duration of the
    ``with`` block. Nested calls merge; inner scope pops on exit."""
    cur = _PROBE_CTX.get()
    new = {**cur, **kwargs}
    token = _PROBE_CTX.set(new)
    try:
        yield
    finally:
        _PROBE_CTX.reset(token)


def enter_task_scope(**kwargs: Any) -> None:
    """Merge ``kwargs`` into the current probe ctx WITHOUT keeping a reset
    token. Intended for callers whose execution is already scoped to an
    asyncio Task (e.g. ``Agent.step_async`` — each agent's step runs in
    its own Task created by ``asyncio.gather``). contextvars are per-Task
    so the mutation naturally dies when the Task finishes. This spares us
    reindenting the method body under a ``with`` block.

    Do NOT call this from synchronous code or from a task whose ctx leaks
    to its parent. Use ``extend_probe_ctx`` (the context manager) in
    those cases.
    """
    cur = _PROBE_CTX.get()
    new = {**cur, **kwargs}
    _PROBE_CTX.set(new)


# ---------- hashing --------------------------------------------------------


def _stable_bytes(obj: Any) -> bytes:
    """Serialise a Python object to deterministic bytes.

    - Preserves insertion order (does NOT sort keys) — tools catalog ordering
      is load-bearing for Hermes/Qwen template rendering; sorting would mask
      real ordering bugs in production.
    - Uses ``default=str`` so non-JSON-native values don't raise here — the
      caller's exception handler catches truly unserialisable inputs.
    """
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"),
                      default=str).encode("utf-8")


def _short_hash(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


def _hash_sys_tools_msgs(
    system: str,
    tools: Optional[Sequence[Mapping[str, Any]]],
    messages: Sequence[Mapping[str, Any]],
) -> dict:
    """Compute the 3 hashes that drive APC analysis.

    ``prior_msgs`` := ``messages[:-1]`` — the prefix that SHOULD match turn-
    over-turn (new content always appears as the last message). ``messages``
    being empty or single-element yields empty prior_msgs.
    """
    sys_h = _short_hash(_stable_bytes(system or ""))
    tools_h = _short_hash(_stable_bytes(list(tools or [])))
    prior = list(messages[:-1]) if messages else []
    prior_h = _short_hash(_stable_bytes(prior))
    return {
        "system_hash": sys_h,
        "tools_hash": tools_h,
        "prior_msgs_hash": prior_h,
    }


# ---------- usage extraction (dict OR attr object) -------------------------


def _get_usage_field(usage: Any, key: str) -> Optional[int]:
    if usage is None:
        return None
    if isinstance(usage, Mapping):
        val = usage.get(key)
    else:
        val = getattr(usage, key, None)
    if val is None:
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


# ---------- main entry point ----------------------------------------------


def record(
    *,
    system: str,
    tools: Optional[Sequence[Mapping[str, Any]]],
    messages: Sequence[Mapping[str, Any]],
    response: Mapping[str, Any],
    latency_s: float,
    model: str = "",
) -> None:
    """Append one probe entry. No-op if ``APC_PROBE_ENABLED`` is not set.

    All exceptions are caught and logged — probe must never break the LLM
    control path.
    """
    if not _enabled():
        return
    try:
        global _seq_counter
        with _list_lock:
            seq = _seq_counter
            _seq_counter += 1

        hashes = _hash_sys_tools_msgs(system, tools, messages)
        prior = list(messages[:-1]) if messages else []
        tail = messages[-1] if messages else {}
        tail_content = tail.get("content") if isinstance(tail, Mapping) else ""
        tail_content = tail_content or ""
        if not isinstance(tail_content, str):
            # Tool messages can have list content (Anthropic shape); just
            # serialise for preview.
            tail_content = str(tail_content)

        usage = response.get("usage") if isinstance(response, Mapping) else None
        prompt_tokens = _get_usage_field(usage, "prompt_tokens")
        prompt_tokens_cached = _get_usage_field(usage, "prompt_tokens_cached")
        completion_tokens = _get_usage_field(usage, "completion_tokens")

        prior_len = sum(
            len(m.get("content") or "") if isinstance(m.get("content"), str) else 0
            for m in prior if isinstance(m, Mapping)
        )

        ctx = get_probe_ctx()
        entry = {
            "seq": seq,
            "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "agent_id": ctx.get("agent_id"),
            "turn": ctx.get("turn"),
            "phase": ctx.get("phase", "unknown"),
            "iteration": ctx.get("iteration"),
            **hashes,
            "prior_msgs_count": len(prior),
            "prior_msgs_char_len": prior_len,
            "tail_role": tail.get("role") if isinstance(tail, Mapping) else None,
            "tail_preview": tail_content[:120],
            "tail_char_len": len(tail_content),
            "prompt_tokens": prompt_tokens,
            "prompt_tokens_cached": prompt_tokens_cached,
            "completion_tokens": completion_tokens,
            "latency_s": round(float(latency_s), 4),
            "model": model or "",
        }
        with _list_lock:
            _probe_list.append(entry)

        if _dump_enabled():
            _dump_raw(seq, system, tools, list(messages))

        _maybe_incremental_flush()

    except Exception as exc:  # noqa: BLE001 — deliberate: probe is advisory
        _log.warning("apc_probe record failed (non-fatal): %s", exc)


def _maybe_incremental_flush() -> None:
    """Flush to the stable session path if >= ``APC_PROBE_FLUSH_EVERY`` new
    entries have been recorded since last flush. No-op when disabled or
    threshold not reached. All exceptions are swallowed — incremental flush
    is best-effort; the atexit flush is the authoritative safety net."""
    every = _flush_every()
    if every <= 0:
        return
    global _last_flushed_count
    try:
        with _list_lock:
            pending = len(_probe_list) - _last_flushed_count
        if pending >= every:
            flush()  # writes to _session_path, updates _last_flushed_count
    except Exception as exc:  # noqa: BLE001
        _log.warning("apc_probe incremental flush failed (non-fatal): %s", exc)


def _dump_raw(
    seq: int,
    system: str,
    tools: Optional[Sequence[Mapping[str, Any]]],
    messages: list,
) -> None:
    try:
        raw_dir = _out_dir() / "apc_probe_raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        path = raw_dir / f"call_{seq:06d}.json"
        payload = {
            "seq": seq,
            "system": system,
            "tools": list(tools or []),
            "messages": messages,
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str)
        )
    except Exception as exc:  # noqa: BLE001
        _log.warning("apc_probe raw-dump failed for seq=%d: %s", seq, exc)


# ---------- flush ----------------------------------------------------------


def flush(path: Optional[str] = None) -> Optional[str]:
    """Write ``_probe_list`` to a JSON file (overwrites).

    When ``path`` is None (the normal case), writes to the stable session
    path so incremental + atexit flushes + manual flush() all update the
    same file. When ``path`` is explicit, writes there (useful for
    exporting a snapshot without disturbing the live session file).

    Returns the path written, or None if the probe is disabled or the
    buffer is empty.
    """
    if not _enabled():
        return None
    global _last_flushed_count
    with _list_lock:
        if not _probe_list:
            return None
        entries = list(_probe_list)

    if path is None:
        path = _resolve_session_path()

    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        # Atomic-ish write: stage to a .tmp sibling then os.replace. Avoids
        # readers seeing a half-written JSON during `tail -f` / periodic
        # analyzer runs.
        tmp_path = f"{path}.tmp"
        Path(tmp_path).write_text(
            json.dumps(entries, ensure_ascii=False, indent=2, default=str)
        )
        os.replace(tmp_path, path)
        if path not in _flushed_paths:
            _flushed_paths.append(path)
        _last_flushed_count = len(entries)
        _log.info("apc_probe flushed %d entries → %s", len(entries), path)
        return path
    except Exception as exc:  # noqa: BLE001
        _log.warning("apc_probe flush failed (%s): %s", path, exc)
        return None


def _atexit_flush() -> None:
    try:
        flush()
    except Exception:
        pass


atexit.register(_atexit_flush)
