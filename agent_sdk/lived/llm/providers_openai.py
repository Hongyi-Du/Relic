"""Real OpenAI provider for the lived ``LLMEngine`` (§6B.1).

The engine (``llm_engine.py``) talks to models through a tiny ``generate(call)``
provider interface and validates the returned ``{thinking, result}`` itself
(schema + structured-thinking + visibility checks, with one feedback retry and a
safe fallback). ``MockProvider`` is the deterministic test transport; this module
adds the **real OpenAI transport** so any LLM-assisted lived module can run on a
live model by swapping the provider — no other code changes.

Design:
  * ``generate(call)`` builds a Responses API request by default (with an
    explicit Chat Completions compatibility mode) that forces a single JSON
    object ``{"thinking": {...}, "result": {...}}``, embeds the output
    JSON-schema + the allowed evidence ids, and on a retry
    (``call.attempt > 0``) appends the engine's ``validator_feedback``.
  * The engine's role routing (``model_role`` -> logical name like
    ``strong_model``) is mapped here to a concrete API model via ``model_map``,
    so the strong/medium/small tiers are preserved.
  * Key from explicit arg or ``OPENAI_API_KEY`` env (loads ``.env`` if present).
    Never hard-code keys; ``.env`` / ``config/*.local.yaml`` are git-ignored.

Env-agnostic: imported only when a real run wires it in; ``openai`` is imported
lazily so the rest of ``agent_sdk.lived`` keeps working without the package.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional

from agent_sdk.lived.llm.llm_engine import (
    DEFAULT_MODEL_ROLES,
    EVIDENCE_KINDS,
    LLMEngine,
    ProviderCall,
)

# Logical engine model name -> concrete OpenAI model. Strong/medium/small tiers
# map to the gpt-4.1 family; mock/default fall back to a cheap model.
DEFAULT_MODEL_MAP: Dict[str, str] = {
    "strong_model": "gpt-4.1",
    "medium_model": "gpt-4.1-mini",
    "small_model": "gpt-4.1-nano",
    "mock_model": "gpt-4o-mini",
}
DEFAULT_FALLBACK_MODEL = "gpt-4o-mini"
_WIRE_APIS = frozenset({"responses", "chat_completions"})
_HEADER_NAME = re.compile(r"^[A-Za-z0-9!#$%&'*+.^_`|~-]+$")
_FORBIDDEN_HEADERS = frozenset(
    {"authorization", "content-length", "host", "proxy-authorization"}
)


def _provider_error_code(exc: Exception) -> str:
    """Return a stable, non-secret failure code for engine audit records."""

    status = getattr(exc, "status_code", None)
    if status in {401, 403}:
        return "authentication_failed"
    if status == 429:
        return "rate_limited"
    if status in {502, 503, 504}:
        return "upstream_unavailable"
    name = type(exc).__name__.casefold()
    if "timeout" in name:
        return "request_timeout"
    if "connection" in name:
        return "connection_failed"
    return f"provider_exception:{type(exc).__name__}"

_SYSTEM_TEMPLATE = (
    "You are the '{module}' module of a lived-agent simulation. "
    "Respond with EXACTLY ONE JSON object and nothing else, of the form: "
    '{{"thinking": {{...}}, "result": {{...}}}}. '
    "The 'result' object MUST conform to this JSON schema:\n{schema}\n"
    "The 'thinking' object MUST include: 'task_understanding' (string), "
    "'rationale' (string), 'confidence' (number in [0,1]), and 'evidence' "
    "(an object whose keys are a subset of {evidence_kinds}; each value is a "
    "list of ids). "
    "CRITICAL: every id you put under 'evidence' MUST come from the "
    "'allowed_evidence' provided in the user message — never invent ids. "
    "If you have no evidence for a bucket, use an empty list."
)


class OpenAILivedProvider:
    """Real OpenAI transport implementing the engine's ``generate(call)`` API."""

    name = "openai"
    version = "v1"

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        model_map: Optional[Dict[str, str]] = None,
        fallback_model: Optional[str] = None,
        base_url: Optional[str] = None,
        max_tokens: int = 1024,
        timeout: float = 60.0,
        wire_api: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
    ):
        key = api_key or _load_key()
        if not key:
            raise RuntimeError(
                "OpenAILivedProvider needs an API key: pass api_key= or set "
                "OPENAI_API_KEY (e.g. in a git-ignored .env)."
            )
        from openai import OpenAI  # lazy

        selected_base_url = base_url or os.environ.get("OPENAI_BASE_URL") or None
        headers = _configured_default_headers()
        client_kwargs: Dict[str, Any] = {
            "api_key": key,
            "max_retries": 0,
            "timeout": float(timeout),
        }
        if selected_base_url:
            client_kwargs["base_url"] = selected_base_url
        if headers:
            client_kwargs["default_headers"] = headers
        self._client = OpenAI(**client_kwargs)
        self.model_map = dict(model_map or _configured_model_map())
        self.fallback_model = fallback_model or _configured_low_model()
        self.max_tokens = int(max_tokens)
        self.timeout = float(timeout)
        selected_wire_api = (
            wire_api
            or os.environ.get("SOCIETY_CORE_OPENAI_WIRE_API")
            or os.environ.get("OPENAI_WIRE_API")
            or "responses"
        ).strip().lower().replace("-", "_")
        if selected_wire_api not in _WIRE_APIS:
            raise ValueError("unsupported_openai_lived_wire_api")
        self.wire_api = selected_wire_api
        self.reasoning_effort = (
            reasoning_effort
            or os.environ.get("SOCIETY_CORE_OPENAI_REASONING_EFFORT")
            or "high"
        ).strip().lower()
        if self.reasoning_effort not in {"low", "medium", "high", "xhigh"}:
            raise ValueError("unsupported_openai_lived_reasoning_effort")
        self.base_url = selected_base_url
        self.default_header_names = tuple(sorted(name.lower() for name in headers))

    def _model_for(self, call: ProviderCall) -> str:
        return self.model_map.get(call.model_name) or self.fallback_model

    @staticmethod
    def _is_reasoning_model(model: str) -> bool:
        return model.lower().startswith(("gpt-5", "o1", "o3", "o4"))

    @staticmethod
    def _response_text(response: Any) -> str:
        direct = getattr(response, "output_text", None)
        if isinstance(direct, str):
            return direct.strip()
        chunks: list[str] = []
        for item in getattr(response, "output", ()) or ():
            for content in getattr(item, "content", ()) or ():
                text = getattr(content, "text", None)
                if isinstance(text, str):
                    chunks.append(text)
        return "".join(chunks).strip()

    def generate(self, call: ProviderCall) -> Any:
        allowed = dict(call.input_payload.get("_evidence", {}))
        system = _SYSTEM_TEMPLATE.format(
            module=call.module_name,
            schema=json.dumps(call.output_schema, ensure_ascii=False),
            evidence_kinds=list(EVIDENCE_KINDS),
        )
        user_parts = [
            call.prompt,
            "\nallowed_evidence (only use ids from here):",
            json.dumps({k: list(allowed.get(k, [])) for k in EVIDENCE_KINDS}, ensure_ascii=False),
        ]
        if call.attempt > 0 and call.validator_feedback:
            user_parts.append(
                "\nYour previous answer was rejected for: "
                + call.validator_feedback
                + "\nFix these issues and return a corrected JSON object."
            )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": "\n".join(user_parts)},
        ]
        model = self._model_for(call)
        try:
            if self.wire_api == "responses":
                kwargs: Dict[str, Any] = {
                    "model": model,
                    "instructions": system + "\nReturn ONLY valid JSON.",
                    "input": messages[1]["content"],
                    "text": {"format": {"type": "json_object"}},
                    "max_output_tokens": (
                        max(self.max_tokens, 6000)
                        if self._is_reasoning_model(model)
                        else self.max_tokens
                    ),
                    "store": False,
                    "timeout": self.timeout,
                }
                if self._is_reasoning_model(model):
                    kwargs["reasoning"] = {"effort": self.reasoning_effort}
                else:
                    kwargs["temperature"] = call.decoding.temperature
                    kwargs["top_p"] = call.decoding.top_p
                resp = self._client.responses.create(**kwargs)
                content = self._response_text(resp) or "{}"
            else:
                kwargs = {
                    "model": model,
                    "messages": messages,
                    "response_format": {"type": "json_object"},
                    "store": False,
                    "timeout": self.timeout,
                }
                if self._is_reasoning_model(model):
                    kwargs["max_completion_tokens"] = max(self.max_tokens, 6000)
                    kwargs["reasoning_effort"] = self.reasoning_effort
                else:
                    kwargs["temperature"] = call.decoding.temperature
                    kwargs["top_p"] = call.decoding.top_p
                    kwargs["max_tokens"] = self.max_tokens
                    if call.decoding.seed is not None:
                        kwargs["seed"] = call.decoding.seed
                resp = self._client.chat.completions.create(**kwargs)
                content = resp.choices[0].message.content or "{}"
        except Exception as exc:
            # The engine recognizes this reserved envelope, retries once, then
            # records a conservative fallback without persisting raw errors.
            return json.dumps({"_provider_error": _provider_error_code(exc)})
        return content  # raw JSON string; engine parses + validates


def _configured_model_map() -> Dict[str, str]:
    high = os.environ.get("SOCIETY_CORE_OPENAI_HIGH_MODEL") or DEFAULT_MODEL_MAP[
        "strong_model"
    ]
    standard = os.environ.get(
        "SOCIETY_CORE_OPENAI_STANDARD_MODEL"
    ) or DEFAULT_MODEL_MAP["medium_model"]
    configured_low = os.environ.get("SOCIETY_CORE_OPENAI_LOW_MODEL")
    return {
        "strong_model": high,
        "medium_model": standard,
        "small_model": configured_low or DEFAULT_MODEL_MAP["small_model"],
        "mock_model": configured_low or DEFAULT_MODEL_MAP["mock_model"],
    }


def _configured_low_model() -> str:
    return (
        os.environ.get("SOCIETY_CORE_OPENAI_LOW_MODEL")
        or DEFAULT_FALLBACK_MODEL
    )


def _configured_default_headers() -> Dict[str, str]:
    raw = os.environ.get("SOCIETY_CORE_OPENAI_DEFAULT_HEADERS_JSON", "").strip()
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("invalid_openai_lived_default_headers_json") from exc
    if not isinstance(payload, dict):
        raise ValueError("openai_lived_default_headers_must_be_object")
    headers: Dict[str, str] = {}
    for raw_name, raw_value in payload.items():
        name = str(raw_name).strip()
        if not name or not _HEADER_NAME.fullmatch(name):
            raise ValueError("invalid_openai_lived_default_header_name")
        if name.lower() in _FORBIDDEN_HEADERS:
            raise ValueError("forbidden_openai_lived_default_header")
        if not isinstance(raw_value, str):
            raise ValueError("invalid_openai_lived_default_header_value")
        value = raw_value.strip()
        if (
            not value
            or len(value) > 4096
            or any(ord(character) < 0x20 or ord(character) == 0x7F for character in value)
        ):
            raise ValueError("invalid_openai_lived_default_header_value")
        headers[name] = value
    return dict(sorted(headers.items(), key=lambda item: item[0].lower()))


def _load_key() -> str:
    key = os.environ.get("OPENAI_API_KEY", "")
    if key:
        return key
    # best-effort .env load (repo root), without adding a hard dotenv dependency
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv()
        return os.environ.get("OPENAI_API_KEY", "")
    except Exception:
        return ""


def build_openai_lived_engine(
    *,
    api_key: Optional[str] = None,
    journal: Optional[Any] = None,
    run_id: str = "run",
    model_map: Optional[Dict[str, str]] = None,
    base_url: Optional[str] = None,
    max_tokens: int = 1024,
    wire_api: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
) -> LLMEngine:
    """Build an ``LLMEngine`` wired to the real OpenAI provider.

    Keeps the engine's logical ``model_roles`` (strong/medium/small) so caching,
    logging and role routing are unchanged; the provider maps those tiers to
    concrete OpenAI models. Falls back to ``MockProvider`` behaviour only if you
    don't call this factory.
    """
    provider = OpenAILivedProvider(
        api_key=api_key,
        model_map=model_map,
        base_url=base_url,
        max_tokens=max_tokens,
        wire_api=wire_api,
        reasoning_effort=reasoning_effort,
    )
    return LLMEngine(
        provider=provider,
        journal=journal,
        run_id=run_id,
        model_roles=dict(DEFAULT_MODEL_ROLES),
    )
