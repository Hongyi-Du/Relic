"""Unified LLM Engine (infra task §2-§6, §8).

Every LLM-assisted module MUST call models through this engine — no module talks
to a provider directly. The engine adds, on top of the raw provider layer
(``agent_sdk.llm.providers`` / ``llm_core``):

  * provider abstraction + role-based routing (§5) + a Mock provider (§5/§22);
  * prompt-template management (§2.2);
  * structured ``{thinking, result, metadata}`` output + a structured-thinking
    validator (§3) that rejects fabricated/invisible evidence (§6);
  * minimal JSON-schema validation of ``result`` (no external deps);
  * content-addressed cache + deterministic replay (§4): a ``read_only`` replay
    that MISSES raises rather than silently calling the model;
  * validation + one feedback retry, then a safe conservative fallback;
  * automatic :class:`LLMCallLog` emission to a :class:`~agent_sdk.lived.record.logs.Journal` (§8);
  * per-module budget tracking (§2.2).

Deterministic by default (temperature 0, top_p 1). Env-agnostic.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

# --------------------------------------------------------------------------- #
# Decoding + roles (§4.3 / §5)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DecodingConfig:
    temperature: float = 0.0
    top_p: float = 1.0
    seed: Optional[int] = None

    def key(self) -> str:
        return f"t={self.temperature};p={self.top_p};s={self.seed}"


DEFAULT_MODEL_ROLES: Dict[str, Dict[str, str]] = {
    "default":               {"name": "mock_model", "version": "mock-1"},
    "routine_annotation":    {"name": "small_model", "version": "v1"},
    "deep_reflection":       {"name": "strong_model", "version": "v1"},
    "wish_parser":           {"name": "strong_model", "version": "v1"},
    "prototype_interpreter": {"name": "strong_model", "version": "v1"},
    "civic_composer":        {"name": "strong_model", "version": "v1"},
    "episode_summarizer":    {"name": "medium_model", "version": "v1"},
    "memory_summarizer":     {"name": "medium_model", "version": "v1"},
    "dispute_interpreter":   {"name": "strong_model", "version": "v1"},
    "debug_mock":            {"name": "mock_model", "version": "mock-1"},
}

# §3.1 structured thinking wrapper — the required evidence buckets.
EVIDENCE_KINDS = ("percepts", "memories", "plans", "episodes",
                  "public_records", "mechanisms", "events")
# evidence bucket -> visibility_context key (§6).
_EVIDENCE_TO_VIS = {
    "percepts": "percepts", "memories": "memories", "plans": "plans",
    "episodes": "episodes", "public_records": "public_records",
    "mechanisms": "mechanisms", "events": "events",
}


class ReplayCacheMiss(RuntimeError):
    """Raised when cache_policy='read_only' and the key is absent (§4.2)."""


# --------------------------------------------------------------------------- #
# Minimal JSON-schema validator (no external deps)
# --------------------------------------------------------------------------- #
_TYPE_MAP = {
    "object": dict, "array": (list, tuple), "string": str,
    "number": (int, float), "integer": int, "boolean": bool,
}


def validate_schema(obj: Any, schema: Optional[Dict[str, Any]], path: str = "result") -> List[str]:
    if not schema:
        return []
    errs: List[str] = []
    t = schema.get("type")
    if t and t in _TYPE_MAP and not isinstance(obj, _TYPE_MAP[t]):
        errs.append(f"{path}: expected {t}, got {type(obj).__name__}")
        return errs
    if t == "object" and isinstance(obj, dict):
        for req in schema.get("required", []):
            if req not in obj:
                errs.append(f"{path}.{req}: required field missing")
        for name, sub in (schema.get("properties") or {}).items():
            if name in obj:
                errs.extend(validate_schema(obj[name], sub, f"{path}.{name}"))
    if t == "array" and isinstance(obj, (list, tuple)) and schema.get("items"):
        for i, it in enumerate(obj):
            errs.extend(validate_schema(it, schema["items"], f"{path}[{i}]"))
    return errs


# --------------------------------------------------------------------------- #
# Providers (§2.2 / §5)
# --------------------------------------------------------------------------- #
@dataclass
class ProviderCall:
    module_name: str
    prompt: str
    output_schema: Dict[str, Any]
    decoding: DecodingConfig
    attempt: int = 0
    validator_feedback: str = ""
    input_payload: Dict[str, Any] = field(default_factory=dict)
    # Routed logical model identity (e.g. "strong_model"/"small_model") + role.
    # Lets a real provider map the engine's role routing to a concrete API model
    # without coupling the engine to any provider. Optional + defaulted so the
    # MockProvider and existing tests are unaffected.
    model_name: str = ""
    model_role: str = "default"
    require_thinking: bool = True


class MockProvider:
    """Deterministic provider for tests (§5/§22) — no external API.

    Register canned responses per key (module_name or template id); each key maps
    to a LIST consumed by attempt index (so attempt 0 can be invalid and attempt
    1 valid, exercising retry). A response may be a dict (used as-is) or a str
    (raw text the engine JSON-parses — use to test invalid-JSON retry). With no
    registered response, returns a schema-shaped valid output echoing the
    evidence ids passed in ``input_payload['_evidence']``."""

    name = "mock_model"
    version = "mock-1"

    def __init__(self, responses: Optional[Dict[str, List[Any]]] = None):
        self.responses: Dict[str, List[Any]] = dict(responses or {})
        self.calls: List[ProviderCall] = []

    def set_response(self, key: str, responses: List[Any]) -> None:
        self.responses[key] = list(responses)

    def generate(self, call: ProviderCall) -> Any:
        self.calls.append(call)
        seq = self.responses.get(call.module_name)
        if seq is not None and call.attempt < len(seq):
            return seq[call.attempt]
        if seq is not None and seq:
            return seq[-1]
        return self._default(call)

    def _default(self, call: ProviderCall) -> Dict[str, Any]:
        evidence = dict(call.input_payload.get("_evidence", {}))
        thinking = {k: "" for k in (
            "task_understanding", "rationale")}
        thinking["evidence"] = {k: list(evidence.get(k, [])) for k in EVIDENCE_KINDS}
        thinking.update({"constraints_checked": [], "alternatives_considered": [],
                         "rejected_alternatives": [], "uncertainties": [], "confidence": 0.5})
        thinking["task_understanding"] = f"mock handling for {call.module_name}"
        thinking["rationale"] = "mock deterministic response"
        return {"thinking": thinking, "result": _schema_defaults(call.output_schema)}


def _schema_defaults(schema: Optional[Dict[str, Any]]) -> Any:
    if not schema:
        return {}
    t = schema.get("type", "object")
    if t == "object":
        out = {}
        for name, sub in (schema.get("properties") or {}).items():
            out[name] = _schema_defaults(sub)
        for req in schema.get("required", []):
            out.setdefault(req, _schema_defaults((schema.get("properties") or {}).get(req, {})))
        return out
    return {"array": [], "string": "", "number": 0, "integer": 0, "boolean": False}.get(t, None)


# --------------------------------------------------------------------------- #
# The engine
# --------------------------------------------------------------------------- #
def _hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


def _summarize(d: Any, limit: int = 240) -> str:
    s = json.dumps(d, ensure_ascii=False, default=str)
    return s if len(s) <= limit else s[:limit] + "…"


@dataclass
class LLMEngine:
    provider: Any = field(default_factory=MockProvider)
    journal: Optional[Any] = None                         # agent_sdk.lived.record.logs.Journal
    run_id: str = "run"
    model_roles: Dict[str, Dict[str, str]] = field(default_factory=lambda: dict(DEFAULT_MODEL_ROLES))
    prompt_templates: Dict[str, str] = field(default_factory=dict)
    schema_versions: Dict[str, str] = field(default_factory=dict)  # template_id -> schema_version
    cache: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    budget: Dict[str, Dict[str, float]] = field(default_factory=dict)  # module -> {tokens,cost,calls}
    _seq: int = 0

    # -- templates ----------------------------------------------------------
    def register_template(self, template_id: str, template: str, *, schema_version: str = "1") -> None:
        self.prompt_templates[template_id] = template
        self.schema_versions[template_id] = schema_version

    def _render_prompt(self, template_id: str, input_payload: Dict[str, Any]) -> str:
        tmpl = self.prompt_templates.get(template_id)
        if tmpl is None:
            # no template registered -> deterministic canonical prompt
            return f"[{template_id}] " + json.dumps(input_payload, sort_keys=True, default=str)
        try:
            return tmpl.format(**input_payload)
        except Exception:
            return tmpl + "\n" + json.dumps(input_payload, sort_keys=True, default=str)

    # -- cache key (§4.1) ---------------------------------------------------
    def _cache_key(self, *, module_name, prompt_template_id, prompt, input_payload,
                   model_name, model_version, output_schema, decoding, visibility_context) -> str:
        parts = {
            "module": module_name,
            "template": prompt_template_id,
            "prompt_hash": _hash(prompt),
            "input_hash": _hash(input_payload),
            "model": model_name,
            "model_version": model_version,
            "schema_version": self.schema_versions.get(prompt_template_id, "1"),
            "schema_hash": _hash(output_schema),
            "decoding": decoding.key(),
            "visibility_hash": _hash(visibility_context or {}),
        }
        return _hash(parts)

    # -- visibility validation (§6) ----------------------------------------
    def _evidence_errors(self, output: Dict[str, Any],
                         visibility_context: Optional[Dict[str, Any]]) -> List[str]:
        errs: List[str] = []
        thinking = (output or {}).get("thinking") or {}
        evidence = thinking.get("evidence") or {}
        if visibility_context is None:
            return errs
        for kind in EVIDENCE_KINDS:
            allowed = set(visibility_context.get(_EVIDENCE_TO_VIS[kind], []) or [])
            for ref in (evidence.get(kind) or []):
                if ref not in allowed:
                    errs.append(f"invisible/fabricated evidence {kind}:{ref}")
        return errs

    def _thinking_errors(self, output: Dict[str, Any], require_thinking: bool) -> List[str]:
        if not require_thinking:
            return []
        if not isinstance(output, dict) or "thinking" not in output:
            return ["missing 'thinking' field"]
        th = output.get("thinking")
        if not isinstance(th, dict):
            return ["'thinking' must be an object"]
        errs = []
        if "rationale" not in th:
            errs.append("thinking.rationale missing")
        c = th.get("confidence")
        if c is not None and not (isinstance(c, (int, float)) and 0.0 <= c <= 1.0):
            errs.append("thinking.confidence must be in [0,1]")
        return errs

    # -- the call (§2.3) ----------------------------------------------------
    def call(
        self,
        *,
        module_name: str,
        agent_id: Optional[str] = None,
        turn_id: Optional[int] = None,
        input_payload: Dict[str, Any],
        output_schema: Dict[str, Any],
        prompt_template_id: str,
        model_role: str = "default",
        require_thinking: bool = True,
        visibility_context: Optional[Dict[str, Any]] = None,
        cache_policy: str = "read_write",
    ) -> Dict[str, Any]:
        self._seq += 1
        t0 = time.time()
        # Role routing decides the model IDENTITY (cache-key + log); the provider
        # is just the transport (a Mock during tests). So changing model_role
        # changes the routed model and thus the cache key (§4.1 / §5).
        model = self.model_roles.get(model_role) or self.model_roles["default"]
        model_name = model.get("name", "mock_model")
        model_version = model.get("version", "v1")
        decoding = DecodingConfig()
        prompt = self._render_prompt(prompt_template_id, input_payload)
        key = self._cache_key(module_name=module_name, prompt_template_id=prompt_template_id,
                              prompt=prompt, input_payload=input_payload, model_name=model_name,
                              model_version=model_version, output_schema=output_schema,
                              decoding=decoding, visibility_context=visibility_context)

        cache_hit = False
        output: Optional[Dict[str, Any]] = None
        validator_errors: List[str] = []
        retry_count = 0

        # -- cache read (§4.2) ---------------------------------------------
        if cache_policy in ("read_write", "read_only") and key in self.cache:
            output = self.cache[key]
            cache_hit = True
        elif cache_policy == "read_only":
            raise ReplayCacheMiss(
                f"replay cache miss for module={module_name} template={prompt_template_id} "
                f"key={key} (read_only: refusing to call the model)")

        # -- provider call + validate + 1 retry (§4 / §6) ------------------
        if output is None:
            feedback = ""
            for attempt in range(2):
                call = ProviderCall(module_name=module_name, prompt=prompt,
                                    output_schema=output_schema, decoding=decoding,
                                    attempt=attempt, validator_feedback=feedback,
                                    input_payload=input_payload,
                                    model_name=model_name, model_role=model_role,
                                    require_thinking=require_thinking)
                raw = self.provider.generate(call)
                parsed, perr = self._parse(raw)
                if perr:
                    validator_errors = [perr]; retry_count = attempt; feedback = perr; continue
                errs = (self._thinking_errors(parsed, require_thinking)
                        + validate_schema(parsed.get("result"), output_schema)
                        + self._evidence_errors(parsed, visibility_context))
                if not errs:
                    output = parsed; validator_errors = []; retry_count = attempt; break
                validator_errors = errs; retry_count = attempt; feedback = "; ".join(errs)
            if output is None:
                # safe conservative fallback (§6)
                output = {"thinking": {"rationale": "validation failed; conservative fallback",
                                        "evidence": {k: [] for k in EVIDENCE_KINDS}, "confidence": 0.0},
                          "result": _schema_defaults(output_schema),
                          "_validation": "failed_fallback"}
            elif cache_policy in ("read_write", "write_only"):
                self.cache[key] = output

        validation_result = ("cache_hit" if cache_hit else
                             ("failed_fallback" if output.get("_validation") == "failed_fallback"
                              else ("valid" if not validator_errors else "invalid")))

        latency_ms = int((time.time() - t0) * 1000)
        # cache hits cost nothing (no new tokens / budget)
        usage = {"prompt": 0, "completion": 0, "cost": 0.0} if cache_hit else self._usage(prompt, output)
        if not cache_hit:
            self._track_budget(module_name, usage)

        # -- LLMCallLog (§8) — auto-written ---------------------------------
        thinking = output.get("thinking") or {}
        evidence = thinking.get("evidence") or {}
        log_rec = {
            "llm_call_id": f"llm_{self._seq}", "agent_id": agent_id, "turn_id": turn_id,
            "module_name": module_name, "prompt_template_id": prompt_template_id,
            "prompt_hash": _hash(prompt), "input_payload_hash": _hash(input_payload),
            "input_state_hash": _hash(visibility_context or {}),
            "model_role": model_role, "model_name": model_name, "model_version": model_version,
            "temperature": decoding.temperature, "top_p": decoding.top_p, "seed": decoding.seed,
            "schema_version": self.schema_versions.get(prompt_template_id, "1"),
            "output_hash": _hash(output.get("result")), "cache_hit": cache_hit,
            "cache_key": key, "cache_policy": cache_policy,
            "validation_result": validation_result, "validator_errors": list(validator_errors),
            "retry_count": retry_count, "latency_ms": latency_ms,
            "token_usage_prompt": usage["prompt"], "token_usage_completion": usage["completion"],
            "cost_estimate": usage["cost"],
            "thinking_summary": _summarize(thinking), "result_summary": _summarize(output.get("result")),
            "referenced_percept_ids": list(evidence.get("percepts", [])),
            "referenced_memory_ids": list(evidence.get("memories", [])),
            "referenced_plan_ids": list(evidence.get("plans", [])),
            "referenced_episode_ids": list(evidence.get("episodes", [])),
            "referenced_mechanism_ids": list(evidence.get("mechanisms", [])),
            "referenced_event_ids": list(evidence.get("events", [])),
            "rejected_alternatives": list(thinking.get("rejected_alternatives", [])),
            "uncertainty_flags": list(thinking.get("uncertainties", [])),
        }
        if self.journal is not None:
            try:
                self.journal.log_llm_call(dict(log_rec))
            except Exception:
                pass

        return {
            "thinking": thinking,
            "result": output.get("result"),
            "metadata": {"llm_call_id": log_rec["llm_call_id"], "cache_hit": cache_hit,
                         "validation_result": validation_result, "retry_count": retry_count,
                         "model_name": model_name, "model_role": model_role,
                         "cache_key": key, "latency_ms": latency_ms,
                         "validator_errors": list(validator_errors)},
        }

    # -- helpers ------------------------------------------------------------
    def _parse(self, raw: Any) -> Tuple[Optional[Dict[str, Any]], str]:
        if isinstance(raw, dict):
            if "_provider_error" in raw:
                return None, f"provider error: {str(raw['_provider_error'])[:96]}"
            return raw, ""
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
                if not isinstance(parsed, dict):
                    return None, f"provider returned JSON {type(parsed).__name__}, not object"
                if "_provider_error" in parsed:
                    return None, f"provider error: {str(parsed['_provider_error'])[:96]}"
                return parsed, ""
            except json.JSONDecodeError as e:
                return None, f"invalid JSON: {e}"
        return None, f"provider returned unsupported type {type(raw).__name__}"

    def _usage(self, prompt: str, output: Dict[str, Any]) -> Dict[str, float]:
        pt = max(1, len(prompt) // 4)
        ct = max(1, len(json.dumps(output, default=str)) // 4)
        return {"prompt": pt, "completion": ct, "cost": round((pt + ct) * 1e-6, 8)}

    def _track_budget(self, module_name: str, usage: Dict[str, float]) -> None:
        b = self.budget.setdefault(module_name, {"tokens": 0.0, "cost": 0.0, "calls": 0.0})
        b["tokens"] += usage["prompt"] + usage["completion"]
        b["cost"] += usage["cost"]
        b["calls"] += 1

    def budget_for(self, module_name: str) -> Dict[str, float]:
        return dict(self.budget.get(module_name, {"tokens": 0.0, "cost": 0.0, "calls": 0.0}))
