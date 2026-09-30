"""
LLMCore — unified LLM interface with retry logic.

Supports quick init (model + api_key) and full config dict.
Default provider: Together AI.
"""
from __future__ import annotations

import os
import asyncio
import hashlib
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent_sdk.base import apc_probe

from .providers import LLMProvider, OpenAIProvider, TogetherProvider

logger = logging.getLogger("agent_sdk.llm")

# ── LLM call dumper (env-gated; zero overhead when LLM_DUMP_DIR is unset) ───
# Activate with: LLM_DUMP_DIR=logs/llm_dump python <run>
#   → Produces $LLM_DUMP_DIR/run_<TS>/<agent_id>/history.json compatible with
#     inspect_context.py.
# OR LLM_DUMP_COHORT_DIR=logs/<session>/cohort python <run>
#   → Produces $LLM_DUMP_COHORT_DIR/<agent_id>/history.json (no run_<TS> wrap)
# LLM_DUMP_COHORT_DIR takes precedence over LLM_DUMP_DIR when both set.
#
# Lookup is lazy on every call — engines (e.g. nature_env) set
# os.environ["LLM_DUMP_DIR"] during __init__ to route dumps into the
# session log dir, after this module is already imported.  A module-level
# cache here would freeze the value at import time and miss that.
_LLM_DUMP_RUN_ID: Optional[str] = None


def _get_llm_dump_dir() -> Optional[str]:
    """Return the active LLM dump dir (env var, looked up on every call)."""
    return os.environ.get("LLM_DUMP_DIR")


def _get_llm_dump_cohort_dir() -> Optional[str]:
    """Return the active cohort-flat dump dir (LLM_DUMP_COHORT_DIR env var).
    When set, takes precedence over LLM_DUMP_DIR and bypasses run_<TS>/ wrapper."""
    return os.environ.get("LLM_DUMP_COHORT_DIR")
_AGENT_ID_RE = re.compile(r"agent[-_](\d+(?:[-_]\d+)?)", re.IGNORECASE)
_TURN_RE = re.compile(r"(?:^|[^a-zA-Z])[Tt]urn[:\s]+(\d+)")
_PHASE_MARKERS = (
    ("Phase 1", "perceive"),
    ("Phase 2", "act"),
    ("Phase 3", "reflect"),
)
# Compaction calls have no Phase marker. Detect via stable headers that
# CompactionEngine emits on the trailing user message. Two variants exist:
#   - Legacy mode (_build_user_prompt): emits "## Message History This Window".
#   - APC mode (_build_apc_instruction, added for SGLang RadixAttention KV
#     reuse): emits "# Compaction Directive" at line 0 and does NOT include
#     the legacy section header (history is inlined as real chat turns, not
#     dumped into text). Also passes the act-phase `basic_tools`, which
#     would otherwise make tool-catalog inference mis-classify as "act".
# Either marker on a user-role message forces phase="compaction", which
# routes the dump into compaction_history.json.
_COMPACTION_USER_MARKERS = (
    "## Message History This Window",  # legacy CompactionEngine._build_user_prompt
    "# Compaction Directive",          # APC-mode CompactionEngine._build_apc_instruction
)
# Reflect-phase exclusive tools per environments/nature_env/prompts/reflect_terminals.yaml +
# agent_sdk.harness.agent.AgenticMiniLoop(phase="reflect") catalog. If any of
# these appear in the `tools` list, the call is a reflect-phase call (act
# phase never exposes these).
_REFLECT_TOOL_MARKERS = frozenset({
    "update_plan", "upload_gene", "vote_asset",
    "broadcast", "reply", "skip",
})  # `deliver` was removed 2026-04-19 — Act-phase `give` replaced it.


def _flatten_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict):
                parts.append(p.get("text") or p.get("content") or "")
            else:
                parts.append(str(p))
        return " ".join(parts)
    return ""


def _infer_llm_dump_metadata(messages, tool_names=None):
    """Best-effort extract (agent_id, turn, phase) for dump file routing.

    Precedence (high → low):
      1. Compaction user marker (overrides everything — compaction calls
         may carry stray tool hints from bad call sites and must still be
         routed to compaction_history.json).
      2. Tool catalog (authoritative in production — reflect-only tool
         names unambiguously signal reflect phase).
      3. Content "Phase N" string markers (fallback only; used by test
         fixtures that lack a tool catalog).

    Prior behaviour let the content marker override the tool catalog,
    which mis-classified every reflect call in nature_env as "act"
    because environments/nature_env/prompts/system_prompt.yaml literally
    contains the substring "within-turn Phase 2 validation" — see
    session_20260419_232114 (agent_1_9/history.json: 40/40 entries
    phase=act, 0 reflect).
    """
    agent_id = "unknown"
    turn: Optional[int] = None

    # Authoritative agent_id + turn from the per-Task apc_probe contextvar.
    # No env-var gate — the contextvar is independent of APC_PROBE_ENABLED;
    # only the probe's record() path is gated, not the ctx read.
    try:
        ctx = apc_probe.get_probe_ctx() or {}
    except Exception:
        ctx = {}
    ctx_agent_id = ctx.get("agent_id")
    ctx_turn = ctx.get("turn")
    ctx_has_agent_id = isinstance(ctx_agent_id, str) and bool(ctx_agent_id)
    ctx_has_turn = isinstance(ctx_turn, int)
    if ctx_has_agent_id:
        agent_id = ctx_agent_id
    if ctx_has_turn:
        turn = ctx_turn

    # Layer 1: tool-catalog signal (authoritative when present).
    tool_phase: Optional[str] = None
    if tool_names:
        if set(tool_names) & _REFLECT_TOOL_MARKERS:
            tool_phase = "reflect"
        else:
            tool_phase = "act"

    # Layer 2: scan messages for content markers + agent_id / turn hints.
    # Probe context remains authoritative. Without that context, keep the
    # historical fallback behavior: later messages may overwrite earlier
    # regex matches because the tail user message is usually the active call.
    content_phase: Optional[str] = None
    is_compaction = False
    for m in messages or []:
        if not isinstance(m, dict):
            continue
        content = _flatten_content(m.get("content"))
        if not content:
            continue
        # Content "Phase N" marker only used when tool catalog is silent.
        if tool_phase is None:
            for marker, ph in _PHASE_MARKERS:
                if marker in content:
                    content_phase = ph
        # Compaction marker (user-role only per test_compaction_marker_on_user_role_only).
        if m.get("role") == "user" and any(
            marker in content for marker in _COMPACTION_USER_MARKERS
        ):
            is_compaction = True
            if not ctx_has_agent_id:
                am = _AGENT_ID_RE.search(content)
                if am:
                    agent_id = f"agent_{am.group(1)}"
            if not ctx_has_turn:
                tm = _TURN_RE.search(content)
                if tm:
                    turn = int(tm.group(1))
        # Regex fallback for agent_id / turn — runs only when the probe ctx
        # didn't provide that field.
        if not ctx_has_agent_id:
            am = _AGENT_ID_RE.search(content)
            if am:
                agent_id = f"agent_{am.group(1)}"
        if not ctx_has_turn:
            tm = _TURN_RE.search(content)
            if tm:
                turn = int(tm.group(1))

    # Layer 3: resolve precedence.
    if is_compaction:
        phase = "compaction"
    elif tool_phase is not None:
        phase = tool_phase
    elif content_phase is not None:
        phase = content_phase
    else:
        phase = "unknown"

    return agent_id, turn, phase


def _get_run_id() -> str:
    global _LLM_DUMP_RUN_ID
    if _LLM_DUMP_RUN_ID is None:
        _LLM_DUMP_RUN_ID = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    return _LLM_DUMP_RUN_ID


def _extract_tool_name(tools) -> List[str]:
    names: List[str] = []
    for t in tools or []:
        tn = None
        if isinstance(t, dict):
            fn = t.get("function") or {}
            tn = fn.get("name") or t.get("name")
        else:
            tn = getattr(t, "name", None)
        if tn:
            names.append(tn)
    return names


def _parse_tool_call_for_dump(result) -> Dict[str, Any]:
    if not isinstance(result, dict):
        return {}
    tcs = result.get("tool_calls") or []
    if not tcs:
        return {}
    tc = tcs[0]
    fn = getattr(tc, "function", None)
    if fn is None and isinstance(tc, dict):
        fn = tc.get("function")
    if fn is None:
        return {}
    name = getattr(fn, "name", None) if not isinstance(fn, dict) else fn.get("name")
    args_raw = getattr(fn, "arguments", None) if not isinstance(fn, dict) else fn.get("arguments")
    try:
        args = json.loads(args_raw) if isinstance(args_raw, str) else dict(args_raw or {})
    except Exception:
        args = {"_raw": str(args_raw)}
    return {"tool_name": name, "tool_args": args}


def _llm_dump_write(messages, tools, result, attempt: int, latency_ms: int, usage: Optional[Dict[str, Any]]) -> None:
    """Write one LLM call entry to $LLM_DUMP_DIR/run_TS/agent_id/history.json."""
    cohort_dir = _get_llm_dump_cohort_dir()
    dump_dir = _get_llm_dump_dir()
    if not cohort_dir and not dump_dir:
        return
    try:
        tool_names = _extract_tool_name(tools)
        agent_id, turn, phase = _infer_llm_dump_metadata(messages, tool_names)
        if cohort_dir:
            base = Path(cohort_dir) / agent_id
        else:
            base = Path(dump_dir) / _get_run_id() / agent_id
        base.mkdir(parents=True, exist_ok=True)
        # Compaction calls go to a sibling file so history.json stays
        # purely act/reflect turn entries. Viewer can load both.
        filename = "compaction_history.json" if phase == "compaction" else "history.json"
        history_path = base / filename
        if history_path.exists():
            try:
                data = json.loads(history_path.read_text(encoding="utf-8"))
                if not isinstance(data, list):
                    data = []
            except Exception:
                data = []
        else:
            data = []

        step = sum(1 for e in data if e.get("turn") == (turn if turn is not None else -1))

        system_content = ""
        if messages and isinstance(messages[0], dict) and messages[0].get("role") == "system":
            system_content = _flatten_content(messages[0].get("content"))
        prompt_sha = hashlib.sha1(system_content.encode("utf-8")).hexdigest()[:16]

        raw_response = ""
        if isinstance(result, dict):
            raw_response = result.get("content") or ""

        entry = {
            "turn": turn if turn is not None else -1,
            "step": step,
            "agent_id": agent_id,
            "decision_type": f"{phase}_call",
            "timestamp": datetime.now().isoformat(),
            "context_window": [
                {"role": m.get("role", ""), "content": _flatten_content(m.get("content"))}
                for m in messages or []
                if isinstance(m, dict)
            ],
            "raw_response": raw_response,
            "decision": _parse_tool_call_for_dump(result),
            "_meta": {
                "phase": phase,
                "prompt_sha": prompt_sha,
                "latency_ms": latency_ms,
                "tokens_in": (usage or {}).get("prompt_tokens"),
                "tokens_out": (usage or {}).get("completion_tokens"),
                "attempt": attempt,
                "tool_count": len(tools or []),
                "tool_names": _extract_tool_name(tools),
            },
        }
        data.append(entry)
        history_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning(f"[llm_dump] exception: {e}")

# Env var names for API keys per provider
_ENV_KEY_MAP = {
    "together": "TOGETHER_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
}

# Default models per provider
_DEFAULT_MODEL = {
    "together": "Qwen/Qwen2.5-72B-Instruct-Turbo",
    "openai": "gpt-4o",
    "google": "gemini-3-flash-preview",
}

# Google Gemini uses OpenAI-compatible API
_GOOGLE_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


class LLMCore:
    """
    Unified LLM interface with retry logic and provider management.

    Supports two init modes:
    1. Quick: LLMCore(provider="together", model="...", api_key="...")
    2. Config: LLMCore(config={...})  or  LLMCore(config_path="llm.yaml")
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        config_path: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        provider: str = "together",
        config_section: str = "agent",
    ):
        # Mode 1: Load from YAML file
        if config_path is not None:
            import yaml
            from pathlib import Path as _Path
            # Per-subprocess override: if LLM_CONFIG_OVERRIDE is set, it
            # wins over the normal config_path and the *.local.yaml fallback.
            # Used by environments/paper_experiments/ launchers to route a
            # subset of runs to a different endpoint (e.g. gemini-3-flash
            # on lift-naval tunnel) without touching the global
            # config/llm.local.yaml that other running processes depend on.
            _env_override = os.environ.get("LLM_CONFIG_OVERRIDE")
            if _env_override and _Path(_env_override).exists():
                config_path = _env_override
            else:
                # Prefer sibling *.local.yaml (gitignored) for per-developer keys.
                # e.g. config/llm.yaml → config/llm.local.yaml if the latter exists.
                _cp = _Path(config_path)
                _local = _cp.with_name(_cp.stem + ".local" + _cp.suffix)
                if _local.exists():
                    config_path = str(_local)
            with open(config_path, "r", encoding="utf-8") as f:
                full = yaml.safe_load(f)
            self._config = full.get(config_section, full)
        # Mode 2: Full config dict
        elif config is not None:
            self._config = config
        # Mode 3: Quick params
        else:
            resolved_key = api_key or os.environ.get(_ENV_KEY_MAP.get(provider, ""), "")
            resolved_model = model or _DEFAULT_MODEL.get(provider, "gpt-4o")
            self._config = {
                "provider": provider,
                provider: {
                    "api_key": resolved_key,
                    "model": resolved_model,
                    "temperature": 0.7,
                    "max_tokens": 4096,
                    "timeout": 120,
                },
                "decision": {"max_retries": 3, "retry_delay": 1},
                "perception": {},
            }

        # Override api_key if explicitly passed with config/config_path
        if api_key and (config_path is not None or config is not None):
            pname = self._config.get("provider", provider)
            if pname in self._config:
                self._config[pname]["api_key"] = api_key
        if model and (config_path is not None or config is not None):
            pname = self._config.get("provider", provider)
            if pname in self._config:
                self._config[pname]["model"] = model

        self.provider_name = self._config.get("provider", provider)
        self.provider: LLMProvider = self._create_provider()

        self.decision_config = self._config.get("decision", {})
        self.perception_config = self._config.get("perception", {})

    def _create_provider(self) -> LLMProvider:
        provider_config = self._config.get(self.provider_name, {})
        if self.provider_name == "openai":
            return OpenAIProvider(provider_config)
        elif self.provider_name == "together":
            return TogetherProvider(provider_config)
        elif self.provider_name == "google":
            # Gemini uses OpenAI-compatible API with Google base_url
            google_cfg = dict(provider_config)
            if "base_url" not in google_cfg:
                google_cfg["base_url"] = _GOOGLE_BASE_URL
            if "model" not in google_cfg:
                google_cfg["model"] = _DEFAULT_MODEL["google"]
            return OpenAIProvider(google_cfg)
        else:
            raise ValueError(f"Unknown provider: {self.provider_name}")

    def create_completion_with_retry(self, messages, tools=None, tool_choice="auto", *, retry_overrides=None):
        """Sync wrapper — delegates to async version via asyncio.run()."""
        return asyncio.run(
            self.create_completion_with_retry_async(
                messages, tools, tool_choice, retry_overrides=retry_overrides
            )
        )

    async def create_completion_with_retry_async(
        self, messages, tools=None, tool_choice="auto", *, retry_overrides=None
    ):
        # ``retry_overrides`` (optional, default None): per-call dict that wins
        # over self.decision_config. Used by non-critical callers (e.g.
        # CompactionEngine) to fast-fail rather than block the whole turn
        # behind a 26-minute retry chain when an endpoint stalls. Recognised
        # keys: ``retry_delays`` (list[float]), ``block_on_failure`` (bool),
        # ``max_retries`` (int), ``retry_delay`` (float),
        # ``blocking_retry_delay`` (float). Unknown keys are ignored. When
        # None, behaviour is byte-identical to pre-2026-04-28.
        cfg_source = dict(self.decision_config)
        if retry_overrides:
            cfg_source.update({k: v for k, v in retry_overrides.items() if v is not None})

        # New schedule mode: retry_delays=[1,1,5,10] → 5 attempts total, sleeping
        # the i-th delay after the i-th failure, then give up (return None).
        # When retry_delays is absent, preserve the legacy behavior exactly:
        # max_retries fast-retries at retry_delay seconds, then blocking retries
        # at blocking_retry_delay (optionally infinite when block_on_failure=True).
        retry_delays_cfg = cfg_source.get("retry_delays")
        if retry_delays_cfg:
            retry_delays = [max(0.0, float(d)) for d in retry_delays_cfg]
            max_attempts = len(retry_delays) + 1
            max_retries = None
            retry_delay = None
            # block_on_failure honoured even with retry_delays — needed so
            # compaction can fast-fail (return None) instead of looping.
            block_on_failure = bool(cfg_source.get("block_on_failure", False))
            blocking_retry_delay = max(0.0, float(cfg_source.get("blocking_retry_delay", 30)))
        else:
            retry_delays = None
            max_attempts = None
            max_retries = max(1, int(cfg_source.get("max_retries", 3)))
            retry_delay = max(0.0, float(cfg_source.get("retry_delay", 1)))
            block_on_failure = bool(cfg_source.get("block_on_failure", True))
            blocking_retry_delay = max(0.0, float(cfg_source.get("blocking_retry_delay", 30)))

        # Lightweight token estimate + timing
        import time as _time
        _est_chars = sum(len(m.get("content", "")) for m in messages)
        _est_tokens = _est_chars // 4  # rough estimate
        _t0 = _time.perf_counter()

        attempt = 0
        while True:
            attempt += 1
            try:
                result = await self.provider.create_completion_async(messages, tools, tool_choice)
                _elapsed = _time.perf_counter() - _t0
                _out_tokens = 0
                if isinstance(result, dict):
                    _out_tokens = len(result.get("content", "") or "") // 4
                    _raw = result.get("raw_response")
                    _usage = getattr(_raw, "usage", None) if _raw is not None else None
                    if _usage:
                        _in = getattr(_usage, "prompt_tokens", 0)
                        _out = getattr(_usage, "completion_tokens", 0)
                        _usage_dict = {
                            "prompt_tokens": _in,
                            "completion_tokens": _out,
                            "latency_ms": round(_elapsed * 1000),
                        }
                        print(f"[LLM] {_elapsed:.2f}s | in={_in} out={_out} | msgs={len(messages)}")
                    else:
                        _usage_dict = None
                        print(f"[LLM] {_elapsed:.2f}s | ~{_est_tokens}tok in, ~{_out_tokens}tok out | msgs={len(messages)}")
                    result["usage"] = _usage_dict
                if _get_llm_dump_dir() or _get_llm_dump_cohort_dir():
                    _llm_dump_write(
                        messages=messages,
                        tools=tools,
                        result=result,
                        attempt=attempt,
                        latency_ms=round(_elapsed * 1000),
                        usage=_usage_dict,
                    )
                # APC prefix-cache probe — env-gated, records only when
                # APC_PROBE_ENABLED=1. Picks up per-agent ctx via
                # contextvars set by callers (agent.step_async, mini_loop).
                # System message is embedded as messages[0] in OpenAI-compat
                # shape; split it out so the probe's system_hash reflects the
                # bucket-shared prefix that SGLang RadixAttention indexes on.
                # Pass the raw `_usage` so the probe can read
                # ``prompt_tokens_cached`` when the provider exposes it
                # (SGLang with --enable-cache-report).
                _sys_text = ""
                _rest_msgs = messages
                if messages and isinstance(messages[0], dict) and messages[0].get("role") == "system":
                    _sys_text = messages[0].get("content", "") or ""
                    _rest_msgs = messages[1:]
                apc_probe.record(
                    system=_sys_text,
                    tools=tools,
                    messages=_rest_msgs,
                    response={"usage": _usage if _usage else _usage_dict},
                    latency_s=_elapsed,
                    model=getattr(self.provider, "model", ""),
                )
                return result
            except Exception as e:
                if retry_delays is not None:
                    logger.warning(
                        f"LLM async call attempt {attempt}/{max_attempts} failed: {e}"
                    )
                    if attempt >= max_attempts:
                        return None
                    await asyncio.sleep(retry_delays[attempt - 1])
                else:
                    bounded = min(attempt, max_retries)
                    logger.warning(f"LLM async call attempt {bounded}/{max_retries} failed: {e}")
                    if not block_on_failure and attempt >= max_retries:
                        return None
                    delay = retry_delay if attempt < max_retries else blocking_retry_delay
                    await asyncio.sleep(delay)

    def parse_tool_call(self, response):
        return self.provider.parse_tool_call(response)

    def get_perception_config(self):
        return self.perception_config
