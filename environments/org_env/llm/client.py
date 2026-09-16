"""OrgLLMClient — provider-agnostic LLM wrapper (spec §2).

`generate_text` / `generate_json` are the only two surfaces the cognitive modules
use. `MockOrgLLMClient` is deterministic + schema-driven (default for tests);
`OpenAIOrgLLMClient` / `GenericHTTPOrgLLMClient` are thin runtime adapters (lazy
imports, never required for tests). On any failure (timeout / parse / schema /
empty) the client raises :class:`LLMError`; callers fall back to rule/template
paths — the LLM is never on the critical correctness path.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any, Callable, Dict, List, Mapping, Optional

USAGE_COUNTERS = ("prompt_tokens", "completion_tokens", "total_tokens", "cached_prompt_tokens")


def _openai_call_watchdog(seconds: float, *, label: str):
    """Import lazily to avoid a package-initialization cycle."""

    from relic.research.openai_runtime import openai_call_watchdog

    return openai_call_watchdog(seconds, label=label)


class LLMError(Exception):
    """Raised on any LLM failure (timeout / parse / schema / empty) so callers fall back."""


class OrgLLMClient:
    provider: str = "base"

    def __init__(self) -> None:
        self.calls = 0
        self.failures = 0
        self.retries = 0
        self.provider_attempts = 0
        self.response_model_counts: Dict[str, int] = {}
        self.response_id_digest = ""
        # actual API token usage (accumulated from response.usage across every
        # attempt, retries included — retried calls burn real tokens too).
        # Providers that expose no usage leave these at zero.
        self.usage_totals: Dict[str, int] = {name: 0 for name in USAGE_COUNTERS}

    def generate_text(self, system_prompt: str, user_prompt: str, *,
                      temperature: float = 0.3, max_tokens: int = 800) -> str:
        raise NotImplementedError

    def generate_json(self, system_prompt: str, user_prompt: str, schema: Dict[str, Any], *,
                      temperature: float = 0.2, max_tokens: int = 1200) -> Dict[str, Any]:
        raise NotImplementedError

    def stats(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "calls": self.calls,
            "failures": self.failures,
            "retries": self.retries,
            "provider_attempts": self.provider_attempts,
            "response_model_counts": dict(self.response_model_counts),
            "response_id_digest": self.response_id_digest or None,
            "usage_totals": dict(self.usage_totals),
        }

    def request_resource_envelope(self, max_tokens: int) -> Dict[str, int]:
        """Worst-case provider resources consumed by one logical request."""

        requested = max(0, int(max_tokens))
        return {
            "provider_attempts": 1,
            "output_tokens_per_attempt": (
                ACCOUNTING_OUTPUT_CEILING
                if requested == UNCAPPED_OUTPUT
                else requested
            ),
        }

    @staticmethod
    def _usage_field(usage: Any, *path: str) -> int:
        """Read a nested usage counter from an SDK object or a plain dict; 0 when absent."""
        cur = usage
        for key in path:
            if cur is None:
                return 0
            cur = cur.get(key) if isinstance(cur, dict) else getattr(cur, key, None)
        try:
            return max(0, int(cur))
        except (TypeError, ValueError):
            return 0

    def _record_usage(self, usage: Any) -> None:
        if usage is None:
            return
        prompt_tokens = self._usage_field(usage, "prompt_tokens")
        if not prompt_tokens:
            prompt_tokens = self._usage_field(usage, "input_tokens")
        completion_tokens = self._usage_field(usage, "completion_tokens")
        if not completion_tokens:
            completion_tokens = self._usage_field(usage, "output_tokens")
        cached_tokens = self._usage_field(
            usage, "prompt_tokens_details", "cached_tokens"
        )
        if not cached_tokens:
            cached_tokens = self._usage_field(
                usage, "input_tokens_details", "cached_tokens"
            )
        self.usage_totals["prompt_tokens"] += prompt_tokens
        self.usage_totals["completion_tokens"] += completion_tokens
        self.usage_totals["total_tokens"] += self._usage_field(usage, "total_tokens")
        self.usage_totals["cached_prompt_tokens"] += cached_tokens

    def _record_response_identity(self, response: Any) -> None:
        model = str(getattr(response, "model", None) or "unreported")
        self.response_model_counts[model] = (
            self.response_model_counts.get(model, 0) + 1
        )
        response_id = str(getattr(response, "id", None) or "")
        if response_id:
            self.response_id_digest = hashlib.sha256(
                (
                    self.response_id_digest
                    + "\x00"
                    + hashlib.sha256(response_id.encode("utf-8")).hexdigest()
                ).encode("ascii")
            ).hexdigest()

    @staticmethod
    def _schema_hint(schema: Dict[str, Any]) -> str:
        """Spell out the exact JSON keys so json_object mode returns the right field
        names (the model otherwise guesses, e.g. 'action' vs 'candidate_action')."""
        props = schema.get("properties") or {}
        if not props:
            return ""
        req = set(schema.get("required") or [])
        keys = ", ".join(f'"{k}"' + (" (required)" if k in req else "") for k in props)
        return ("\n\nReturn a JSON object using EXACTLY these keys (use these exact names, "
                f"no others): {keys}.")

    # shared schema validation (loose: required keys present, type object)
    @staticmethod
    def _validate(data: Any, schema: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(data, dict):
            raise LLMError("LLM output is not a JSON object")
        for req in schema.get("required", []):
            if req not in data:
                raise LLMError(f"LLM output missing required field '{req}'")
        return data


# --------------------------------------------------------------------------- #
# Mock (deterministic, schema-driven) — default for tests
# --------------------------------------------------------------------------- #
class MockOrgLLMClient(OrgLLMClient):
    """Deterministic LLM stand-in.

    - ``responder(system, user, schema) -> dict`` lets a test fully control output.
    - ``script`` is a list of dicts returned in order (then it repeats the last).
    - otherwise a schema-driven stub produces a plausible, *safe* object for any
      cognitive module (action decision defaults to the always-available
      ``internal_search`` so the validated path succeeds).
    - ``fail=True`` makes every call raise LLMError (to exercise fallbacks).
    """

    provider = "mock"

    def __init__(self, *, responder: Optional[Callable[..., Dict[str, Any]]] = None,
                 script: Optional[List[Dict[str, Any]]] = None, fail: bool = False) -> None:
        super().__init__()
        self.responder = responder
        self.script = list(script or [])
        self.fail = fail
        self._i = 0

    def generate_text(self, system_prompt, user_prompt, *, temperature=0.3, max_tokens=800) -> str:
        self.calls += 1
        if self.fail:
            self.failures += 1
            raise LLMError("mock failure")
        if self.responder is not None:
            r = self.responder(system_prompt, user_prompt, None)
            return r if isinstance(r, str) else json.dumps(r)
        return "(mock surface text)"

    def generate_json(self, system_prompt, user_prompt, schema, *, temperature=0.2, max_tokens=1200):
        self.calls += 1
        if self.fail:
            self.failures += 1
            raise LLMError("mock failure")
        if self.responder is not None:
            data = self.responder(system_prompt, user_prompt, schema)
        elif self.script:
            data = self.script[min(self._i, len(self.script) - 1)]
            self._i += 1
        else:
            data = self._stub(schema)
        return self._validate(data, schema)

    @staticmethod
    def _stub(schema: Dict[str, Any]) -> Dict[str, Any]:
        props = set((schema.get("properties") or {}).keys())
        if "candidate_action" in props:            # action decision -> safe no-op action
            return {"candidate_action": "internal_search", "candidate_speech_act": None,
                    "target_agent_id": None, "target_object_id": None, "channel_id": None,
                    "params": {"query": "reproducibility"}, "rationale": "(mock) gather context",
                    "expected_effect": "more information", "risk_assessment": "none", "confidence": 0.5}
        if "improvement_ideas" in props:           # reflection
            return {"self_assessment": "(mock) I notice recurring friction.",
                    "team_assessment": "(mock) the team hits the same problem repeatedly.",
                    "perceived_blockers": ["(mock) recurring blocker"],
                    "perceived_self_needs": ["(mock) a lighter way to do this"],
                    "perceived_team_needs": ["(mock) a shared standard"],
                    "perceived_repeated_failures": [],
                    "improvement_ideas": [{"need_type": "workflow_need", "missing_support_type": "workflow",
                                           "description": "a clearer shared workflow", "urgency": 0.6,
                                           "risk_if_unaddressed": "recurring friction", "team": True, "self": False}],
                    "raw_text": "(mock) reflection"}
        if "wishes" in props:                      # wish extraction
            return {"wishes": [{"raw_reflection_excerpt": "(mock) I need a clearer workflow.",
                                "wish_type": "workflow_need", "interpreted_need": "a clearer shared workflow",
                                "target_problem": "recurring friction", "self_related": False, "team_related": True,
                                "suggested_improvement": "standardize the workflow", "missing_support_type": "workflow",
                                "urgency": 0.6, "expected_benefit": "less friction",
                                "risk_if_unaddressed": "recurring friction"}]}
        if "surface_text" in props:                # surface realizer
            return {"surface_text": "(mock) grounded message.", "style_tags": ["direct"],
                    "contains_new_facts": False}
        if "proposal_type" in props:               # proposal generator
            return {"proposal_type": "workflow_proposal", "title": "(mock) workflow",
                    "summary": "(mock) standardize the workflow",
                    "target_problem": "recurring friction", "proposed_solution": "define + adopt a workflow",
                    "required_actions": [], "required_capabilities": [], "required_artifacts": [],
                    "required_participants": [], "expected_benefits": ["less friction"],
                    "expected_costs": [], "risks": [], "approval_required_from": []}
        if "feasibility_score" in props:           # proposal evaluator
            return {"feasibility_score": 0.7, "usefulness_score": 0.7, "risk_score": 0.3,
                    "adoption_score": 0.6, "blocking_issues": [], "suggested_revision": ""}
        if "trigger_summary" in props:             # episode summarizer
            return {"title": "(mock) episode", "trigger_summary": "(mock) trigger",
                    "participant_summary": "(mock) participants", "conflict_summary": "(mock) conflict",
                    "decision_summary": "(mock) decision", "outcome_summary": "(mock) outcome",
                    "open_questions": []}
        if "input_schema" in props:                # tool composer
            return {"name": "(mock) tool", "description": "(mock) composed tool",
                    "tool_type": "composed_action_tool", "input_schema": {}, "output_schema": {},
                    "required_actions": [], "required_capabilities": [], "risk_tags": [],
                    "validation_rules": [], "callable_by_roles": []}
        if "risk_level" in props:                  # object appraisal
            return {"risk_level": "medium", "issue_tags": ["mock_issue"], "evidence_gaps": []}
        if "success" in props and "intensity" in props:   # event appraisal
            return {"success": True, "reputation_delta": 0.0, "knowledge_gain": 0.1,
                    "rule_compliance_result": "none", "promise_kept_or_broken": "none",
                    "emotional_valence": 0.0, "intensity": "minor", "visibility": "team",
                    "rationale": "(mock) appraisal"}
        if "trigger_condition" in props:           # institution synthesizer
            return {"name": "(mock) protocol", "trigger_condition": "(mock) condition",
                    "required_steps": [], "required_fields": [], "enforcement_rule": "(mock)",
                    "violation_condition": "(mock)", "benefits": [], "costs": [], "risks": [],
                    "affected_actions": []}
        return {}


# --------------------------------------------------------------------------- #
# Runtime adapters (lazy imports; never required for tests)
# --------------------------------------------------------------------------- #
REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh")
OPENAI_WIRE_APIS = ("chat_completions", "responses")

# max_tokens=UNCAPPED_OUTPUT sends no output ceiling, leaving the model's own
# maximum in force. Accounting still needs a worst case, so the ledger reserves
# ACCOUNTING_OUTPUT_CEILING per attempt for such a request.
UNCAPPED_OUTPUT = 0
ACCOUNTING_OUTPUT_CEILING = int(
    os.environ.get("ORG_LLM_ACCOUNTING_OUTPUT_CEILING", "32000") or 32000
)


class OpenAIOrgLLMClient(OrgLLMClient):
    provider = "openai"

    def __init__(self, *, model: str = "gpt-4o-mini", api_key: Optional[str] = None,
                 base_url: Optional[str] = None, reasoning_effort: str = "low",
                 max_retries: int = 2, retry_backoff_seconds: float = 1.0,
                 wire_api: str = "chat_completions",
                 request_timeout_seconds: float = 120.0,
                 store_responses: bool = False,
                 default_headers: Mapping[str, str] | None = None) -> None:
        super().__init__()
        self.model = model
        self._api_key = api_key
        self._base_url = base_url
        wire_api = str(wire_api).strip().lower().replace("-", "_")
        if wire_api not in OPENAI_WIRE_APIS:
            raise ValueError(
                f"wire_api must be one of {OPENAI_WIRE_APIS}, got {wire_api!r}"
            )
        self.wire_api = wire_api
        effort = str(reasoning_effort).strip().lower()
        if effort not in REASONING_EFFORTS:
            raise ValueError(
                f"reasoning_effort must be one of {REASONING_EFFORTS}, got {reasoning_effort!r}")
        self.reasoning_effort = effort
        if (
            int(max_retries) < 0
            or float(retry_backoff_seconds) < 0
            or float(request_timeout_seconds) <= 0
        ):
            raise ValueError(
                "max_retries and retry_backoff_seconds must be non-negative; "
                "request_timeout_seconds must be positive"
            )
        self.max_retries = int(max_retries)
        self.retry_backoff_seconds = float(retry_backoff_seconds)
        self.request_timeout_seconds = float(request_timeout_seconds)
        self.store_responses = bool(store_responses)
        self.default_headers = {
            str(key): str(value)
            for key, value in dict(default_headers or {}).items()
            if str(key).strip() and str(value).strip()
        }
        self._client_instance: Any = None

    def _is_reasoning_model(self) -> bool:
        return self.model.lower().startswith(("gpt-5", "o1", "o3", "o4"))

    def effective_output_token_limit(self, max_tokens: int) -> Optional[int]:
        """Ceiling to send the provider, or None to send none at all.

        A caller that must receive a whole rewritten file cannot name a number
        here: it does not know how long the file will be after the edit, and
        naming one too low does not truncate the answer, it destroys it. On the
        Responses API reasoning tokens are drawn from the same allowance, so at
        high effort the reasoning alone can consume the budget and the call
        returns empty. Measured on a 336-tick b0 case: every one of 29 failed
        patches came back as an empty response, and each targeted a file whose
        rewrite could not fit — 43k and 37k characters against a 6000-token
        allowance — while the 11k-character file in the same run was revised
        six times.
        """
        requested = max(0, int(max_tokens))
        if requested == UNCAPPED_OUTPUT:
            return None
        return max(requested, 6000) if self._is_reasoning_model() else requested

    @property
    def routing_context_fingerprint(self) -> str:
        """Bind endpoint and credential routing without serializing secrets."""

        secret_payload = {
            "api_key": self._api_key or os.environ.get("OPENAI_API_KEY") or "",
            "default_headers": dict(sorted(self.default_headers.items())),
        }
        payload = {
            "base_url": str(self._base_url or "official_default"),
            "credential_digest": hashlib.sha256(
                json.dumps(
                    secret_payload,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
        }
        return hashlib.sha256(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

    def request_resource_envelope(self, max_tokens: int) -> Dict[str, int]:
        limit = self.effective_output_token_limit(max_tokens)
        return {
            "provider_attempts": 1 + self.max_retries,
            # An uncapped request still spends real tokens, so it reserves the
            # accounting ceiling rather than nothing.
            "output_tokens_per_attempt": (
                ACCOUNTING_OUTPUT_CEILING if limit is None else limit
            ),
        }

    @property
    def effective_reasoning_effort(self) -> Optional[str]:
        """The reasoning_effort actually sent to the API — None for non-reasoning
        models, which ignore the knob entirely. Run records read this attribute."""
        return self.reasoning_effort if self._is_reasoning_model() else None

    def _client(self):
        # Reuse one SDK client for the whole run: rebuilding per call discards
        # connection pooling and made the pre-fix code a natural place to also
        # drop response.usage on the floor.
        if self._client_instance is not None:
            return self._client_instance
        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:  # pragma: no cover - runtime only
            raise LLMError(f"openai sdk unavailable: {e}")
        # An explicit per-phase timeout, not a scalar. A scalar bounds the whole
        # call, but the SDK's default transport can still sit indefinitely in a
        # socket READ once the connection is established, and the SIGALRM
        # watchdog cannot fire on a worker thread - which is where the P5 arm's
        # calls run, because it is the only condition that executes agents in
        # parallel. That combination stalled a run at 0% CPU twice, and a resume
        # from checkpoint stalled at the same tick. A read timeout is what
        # actually unblocks the thread; the watchdog only reports.
        try:
            import httpx

            transport_timeout = httpx.Timeout(
                connect=min(30.0, self.request_timeout_seconds),
                read=self.request_timeout_seconds,
                write=min(30.0, self.request_timeout_seconds),
                pool=min(30.0, self.request_timeout_seconds),
            )
        except Exception:               # httpx absent: keep the scalar behaviour
            transport_timeout = self.request_timeout_seconds
        kw: Dict[str, Any] = {
            "api_key": self._api_key or os.environ.get("OPENAI_API_KEY"),
            "timeout": transport_timeout,
            # This class owns retry accounting and backoff. Disabling SDK
            # retries keeps the recorded retry count and deadline truthful.
            "max_retries": 0,
        }
        if self._base_url:
            kw["base_url"] = self._base_url
        if self.default_headers:
            kw["default_headers"] = dict(self.default_headers)
        self._client_instance = OpenAI(**kw)
        return self._client_instance

    def _invoke(self, method: Callable[..., Any], **request: Any) -> Any:
        """Call one SDK surface under one total wall-clock deadline."""
        client = self._client()
        attempt = 0
        deadline = time.monotonic() + self.request_timeout_seconds
        try:
            with _openai_call_watchdog(
                self.request_timeout_seconds,
                label=f"OrgEnv {self.wire_api} call",
            ):
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError(
                            "OrgEnv OpenAI call exhausted its wall-clock budget"
                        )
                    attempt_request = dict(request)
                    # Per-phase, not scalar. A scalar here SILENTLY REPLACES the
                    # httpx.Timeout configured on the client, so the read budget
                    # this class carefully sets is discarded on every request and
                    # an SSL read can block forever — observed as a main-thread
                    # stall in _ssl__SSLSocket_read -> poll, with the watchdog
                    # showing nothing because the call never returned to it.
                    try:
                        import httpx

                        attempt_request["timeout"] = httpx.Timeout(
                            connect=min(30.0, remaining),
                            read=remaining,
                            write=min(30.0, remaining),
                            pool=min(30.0, remaining),
                        )
                    except Exception:
                        attempt_request["timeout"] = remaining
                    try:
                        self.provider_attempts += 1
                        response = method(client, **attempt_request)
                    except Exception as exc:
                        if attempt >= self.max_retries:
                            raise LLMError(str(exc)) from exc
                        self.retries += 1
                        backoff = self.retry_backoff_seconds * (2 ** attempt)
                        if backoff >= deadline - time.monotonic():
                            raise TimeoutError(
                                "OrgEnv OpenAI retry would exceed its "
                                "wall-clock budget"
                            ) from exc
                        time.sleep(backoff)
                        attempt += 1
                        continue
                    self._record_usage(getattr(response, "usage", None))
                    self._record_response_identity(response)
                    return response
        except TimeoutError as exc:
            raise LLMError(str(exc)) from exc

    def _create_completion(self, **request: Any) -> Any:
        return self._invoke(
            lambda client, **payload: client.chat.completions.create(**payload),
            **request,
        )

    def _create_response(self, **request: Any) -> Any:
        return self._invoke(
            lambda client, **payload: client.responses.create(**payload),
            **request,
        )

    def _gen_kwargs(self, temperature: float, max_tokens: int) -> Dict[str, Any]:
        """GPT-5 / o-series reasoning models use max_completion_tokens + default
        temperature; older chat models use max_tokens + custom temperature.

        A None limit means the caller asked for no ceiling, so the key is left
        out and the model's own maximum applies.
        """
        limit = self.effective_output_token_limit(max_tokens)
        if self._is_reasoning_model():
            # reasoning tokens count against the budget — give ample headroom so
            # structured JSON output isn't starved by reasoning.
            kwargs: Dict[str, Any] = {"reasoning_effort": self.reasoning_effort}
            if limit is not None:
                kwargs["max_completion_tokens"] = limit
            return kwargs
        kwargs = {"temperature": temperature}
        if limit is not None:
            kwargs["max_tokens"] = limit
        return kwargs

    def _response_kwargs(
        self,
        temperature: float,
        max_tokens: int,
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {"store": self.store_responses}
        limit = self.effective_output_token_limit(max_tokens)
        if limit is not None:
            kwargs["max_output_tokens"] = limit
        if self._is_reasoning_model():
            kwargs["reasoning"] = {"effort": self.reasoning_effort}
        else:
            kwargs["temperature"] = temperature
        return kwargs

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

    def generate_text(self, system_prompt, user_prompt, *, temperature=0.3, max_tokens=800) -> str:
        self.calls += 1
        try:
            if self.wire_api == "responses":
                response = self._create_response(
                    model=self.model,
                    instructions=system_prompt,
                    input=user_prompt,
                    **self._response_kwargs(temperature, max_tokens),
                )
                text = self._response_text(response)
                if not text:
                    raise LLMError("empty Responses API output")
                return text
            r = self._create_completion(
                model=self.model, messages=[{"role": "system", "content": system_prompt},
                                            {"role": "user", "content": user_prompt}],
                **self._gen_kwargs(temperature, max_tokens))
            return (r.choices[0].message.content or "").strip()
        except LLMError:
            self.failures += 1
            raise
        except Exception as e:
            self.failures += 1
            raise LLMError(str(e))

    def generate_json(self, system_prompt, user_prompt, schema, *, temperature=0.2, max_tokens=1200):
        self.calls += 1
        try:
            if self.wire_api == "responses":
                response = self._create_response(
                    model=self.model,
                    instructions=system_prompt + "\nReturn ONLY valid JSON.",
                    input=user_prompt + self._schema_hint(schema),
                    text={"format": {"type": "json_object"}},
                    **self._response_kwargs(temperature, max_tokens),
                )
                return self._validate(
                    json.loads(self._response_text(response) or "{}"),
                    schema,
                )
            r = self._create_completion(
                model=self.model, response_format={"type": "json_object"},
                messages=[{"role": "system", "content": system_prompt + "\nReturn ONLY valid JSON."},
                          {"role": "user", "content": user_prompt + self._schema_hint(schema)}],
                **self._gen_kwargs(temperature, max_tokens))
            return self._validate(json.loads(r.choices[0].message.content or "{}"), schema)
        except LLMError:
            self.failures += 1
            raise
        except Exception as e:
            self.failures += 1
            raise LLMError(str(e))


class GenericHTTPOrgLLMClient(OrgLLMClient):
    provider = "http"

    def __init__(self, *, endpoint: str, model: str = "local") -> None:
        super().__init__()
        self.endpoint = endpoint
        self.model = model

    def _post(self, payload: Dict[str, Any]) -> str:  # pragma: no cover - runtime only
        import urllib.request
        req = urllib.request.Request(self.endpoint, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read().decode())
        return body.get("text") or body.get("content") or json.dumps(body)

    def generate_text(self, system_prompt, user_prompt, *, temperature=0.3, max_tokens=800) -> str:
        self.calls += 1
        try:  # pragma: no cover - runtime only
            return self._post({"model": self.model, "system": system_prompt, "prompt": user_prompt,
                               "temperature": temperature, "max_tokens": max_tokens})
        except Exception as e:
            self.failures += 1
            raise LLMError(str(e))

    def generate_json(self, system_prompt, user_prompt, schema, *, temperature=0.2, max_tokens=1200):
        self.calls += 1
        try:  # pragma: no cover - runtime only
            txt = self._post({"model": self.model, "system": system_prompt + " Return ONLY JSON.",
                              "prompt": user_prompt, "temperature": temperature, "max_tokens": max_tokens})
            return self._validate(json.loads(txt), schema)
        except LLMError:
            self.failures += 1
            raise
        except Exception as e:
            self.failures += 1
            raise LLMError(str(e))


def build_org_llm_client(provider: Optional[str] = None, **kw) -> OrgLLMClient:
    """Factory. Defaults to the deterministic Mock; real providers are opt-in."""
    p = (provider or "mock").lower()
    if p == "mock":
        return MockOrgLLMClient(**kw)
    if p == "openai":
        return OpenAIOrgLLMClient(**kw)
    if p in ("http", "generic"):
        return GenericHTTPOrgLLMClient(**kw)
    raise ValueError(f"unknown llm provider: {provider}")


__all__ = [
    "GenericHTTPOrgLLMClient",
    "LLMError",
    "MockOrgLLMClient",
    "OPENAI_WIRE_APIS",
    "OpenAIOrgLLMClient",
    "OrgLLMClient",
    "REASONING_EFFORTS",
    "build_org_llm_client",
]
