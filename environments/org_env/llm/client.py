"""OrgLLMClient — provider-agnostic LLM wrapper (spec §2).

`generate_text` / `generate_json` are the only two surfaces the cognitive modules
use. `MockOrgLLMClient` is deterministic + schema-driven (default for tests);
`OpenAIOrgLLMClient` / `GenericHTTPOrgLLMClient` are thin runtime adapters (lazy
imports, never required for tests). On any failure (timeout / parse / schema /
empty) the client raises :class:`LLMError`; callers fall back to rule/template
paths — the LLM is never on the critical correctness path.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass
import hashlib
import json
import math
import multiprocessing
import os
import re
import signal
import threading
import time
from typing import Any, Callable, Dict, List, Mapping, Optional

USAGE_COUNTERS = ("prompt_tokens", "completion_tokens", "total_tokens", "cached_prompt_tokens")


def _openai_call_watchdog(seconds: float, *, label: str):
    """Import lazily to avoid a package-initialization cycle through the runtime helper."""

    from relic.research.openai_runtime import openai_call_watchdog

    return openai_call_watchdog(seconds, label=label)


def _requires_provider_process_deadline() -> bool:
    """Whether this call needs a killable process for its hard deadline."""

    return (
        threading.current_thread() is not threading.main_thread()
        or not hasattr(signal, "SIGALRM")
        or not hasattr(signal, "setitimer")
    )


class LLMError(Exception):
    """Raised on any LLM failure (timeout / parse / schema / empty) so callers fall back."""


class _ProviderAttemptFailure(Exception):
    """A post-admission child/provider failure eligible for outer retry."""


_PROVIDER_ATTEMPT_RESERVER: ContextVar[
    Optional[Callable[[int], None]]
] = ContextVar("org_llm_provider_attempt_reserver", default=None)


@contextmanager
def provider_attempt_reservation(
    reserver: Optional[Callable[[int], None]],
):
    """Install a per-provider-attempt admission hook for one logical call.

    The metered wrapper uses this only for clients whose transport retry loop
    lives inside ``generate_*``.  A context variable keeps concurrent calls
    isolated: no callback is stored on the shared provider client.
    """

    token = _PROVIDER_ATTEMPT_RESERVER.set(reserver)
    try:
        yield
    finally:
        _PROVIDER_ATTEMPT_RESERVER.reset(token)


class OrgLLMClient:
    provider: str = "base"
    # Subclasses that set this must call the current attempt reserver exactly
    # once, immediately before every real provider request, including retries.
    meters_actual_provider_attempts: bool = False

    def __init__(self) -> None:
        self._stats_lock = threading.RLock()
        self._calls = 0
        self._failures = 0
        self._retries = 0
        self._provider_attempts = 0
        self._response_model_counts: Dict[str, int] = {}
        self._response_id_digest = ""
        # actual API token usage (accumulated from response.usage across every
        # attempt, retries included — retried calls burn real tokens too).
        # Providers that expose no usage leave these at zero.
        self._usage_totals: Dict[str, int] = {name: 0 for name in USAGE_COUNTERS}

    @property
    def calls(self) -> int:
        with self._stats_lock:
            return self._calls

    @calls.setter
    def calls(self, value: int) -> None:
        with self._stats_lock:
            self._calls = int(value)

    @property
    def failures(self) -> int:
        with self._stats_lock:
            return self._failures

    @failures.setter
    def failures(self, value: int) -> None:
        with self._stats_lock:
            self._failures = int(value)

    @property
    def retries(self) -> int:
        with self._stats_lock:
            return self._retries

    @retries.setter
    def retries(self, value: int) -> None:
        with self._stats_lock:
            self._retries = int(value)

    @property
    def provider_attempts(self) -> int:
        with self._stats_lock:
            return self._provider_attempts

    @provider_attempts.setter
    def provider_attempts(self, value: int) -> None:
        with self._stats_lock:
            self._provider_attempts = int(value)

    @property
    def usage_totals(self) -> Dict[str, int]:
        with self._stats_lock:
            return dict(self._usage_totals)

    @usage_totals.setter
    def usage_totals(self, value: Mapping[str, int]) -> None:
        with self._stats_lock:
            self._usage_totals = {
                str(key): max(0, int(item or 0))
                for key, item in dict(value).items()
            }

    @property
    def response_model_counts(self) -> Dict[str, int]:
        with self._stats_lock:
            return dict(self._response_model_counts)

    @response_model_counts.setter
    def response_model_counts(self, value: Mapping[str, int]) -> None:
        with self._stats_lock:
            self._response_model_counts = {
                str(key): max(0, int(item or 0))
                for key, item in dict(value).items()
            }

    @property
    def response_id_digest(self) -> str:
        with self._stats_lock:
            return self._response_id_digest

    @response_id_digest.setter
    def response_id_digest(self, value: str) -> None:
        with self._stats_lock:
            self._response_id_digest = str(value or "")

    def _increment_stat(self, name: str, amount: int = 1) -> None:
        private_name = f"_{name}"
        with self._stats_lock:
            setattr(self, private_name, int(getattr(self, private_name)) + int(amount))

    def __getstate__(self) -> Dict[str, Any]:
        """Keep deterministic/mock clients pickle-compatible despite the lock."""

        state = dict(self.__dict__)
        state.pop("_stats_lock", None)
        # A live SDK transport must never cross a checkpoint/process boundary.
        if "_client_instance" in state:
            state["_client_instance"] = None
        return state

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        self.__dict__.update(dict(state))
        self._stats_lock = threading.RLock()

    def generate_text(self, system_prompt: str, user_prompt: str, *,
                      temperature: float = 0.3, max_tokens: int = 800) -> str:
        raise NotImplementedError

    def generate_json(self, system_prompt: str, user_prompt: str, schema: Dict[str, Any], *,
                      temperature: float = 0.2, max_tokens: int = 1200) -> Dict[str, Any]:
        raise NotImplementedError

    def stats(self) -> Dict[str, Any]:
        with self._stats_lock:
            return {
                "provider": self.provider,
                "calls": self._calls,
                "failures": self._failures,
                "retries": self._retries,
                "provider_attempts": self._provider_attempts,
                "response_model_counts": dict(self._response_model_counts),
                "response_id_digest": self._response_id_digest or None,
                "usage_totals": dict(self._usage_totals),
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
        with self._stats_lock:
            self._usage_totals["prompt_tokens"] += prompt_tokens
            self._usage_totals["completion_tokens"] += completion_tokens
            self._usage_totals["total_tokens"] += self._usage_field(
                usage, "total_tokens"
            )
            self._usage_totals["cached_prompt_tokens"] += cached_tokens

    def _record_response_identity(self, response: Any) -> None:
        model = str(getattr(response, "model", None) or "unreported")
        response_id = str(getattr(response, "id", None) or "")
        with self._stats_lock:
            self._response_model_counts[model] = (
                self._response_model_counts.get(model, 0) + 1
            )
            if response_id:
                self._response_id_digest = hashlib.sha256(
                    (
                        self._response_id_digest
                        + "\x00"
                        + hashlib.sha256(response_id.encode("utf-8")).hexdigest()
                    ).encode("ascii")
                ).hexdigest()

    @staticmethod
    def _schema_hint(schema: Dict[str, Any]) -> str:
        """Spell out the exact JSON keys so json_object mode returns the right field
        names (the model otherwise guesses, e.g. 'action' vs 'candidate_action').

        Two schema shapes reach this method. Most callers pass a JSON Schema with
        a "properties" map. The two editors pass a literal example object --
        ``{"edits": [{"search": "string", ...}], "change_summary": "string"}`` --
        which has no "properties" key, so this returned "" and the model was told
        nothing at all. CodePatch is then built from eleven fields of that answer
        (code_editor.py:544-556) while the system prompt names only four in
        prose, so seven were asked of a model that had never heard of them, and
        ``resolved_gaps`` was read without appearing in the schema at all -- the
        prompt lists the artifact's known gaps, the validator reconciles claimed
        gaps against them, and nothing ever asked which ones the patch closed.

        An example object already states both the names and the nesting, so it is
        sent as-is rather than being flattened to a key list.
        """
        if "properties" in schema:
            # A JSON Schema, even one declaring no properties. Falling through
            # would describe the schema itself to the model as if it were the
            # answer's shape.
            props = schema.get("properties") or {}
            if not props:
                return ""
            req = set(schema.get("required") or [])
            keys = ", ".join(
                f'"{k}"' + (" (required)" if k in req else "") for k in props
            )
            return ("\n\nReturn a JSON object using EXACTLY these keys (use these exact names, "
                    f"no others): {keys}.")
        if not schema or not all(isinstance(key, str) for key in schema):
            return ""
        try:
            shape = json.dumps(schema, ensure_ascii=False, sort_keys=False)
        except (TypeError, ValueError):
            return ""
        return ("\n\nReturn a JSON object with EXACTLY these keys (use these exact "
                "names, no others). The values below show the expected type or "
                f"shape of each field, not its content:\n{shape}")

    # shared schema validation (loose: required keys present, type object)
    @staticmethod
    def _validate(data: Any, schema: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(data, dict):
            raise LLMError("LLM output is not a JSON object")
        for req in schema.get("required", []):
            if req not in data:
                raise LLMError(f"LLM output missing required field '{req}'")
        return data

    @staticmethod
    def _parse_json_object(text: Any) -> Dict[str, Any]:
        """Decode a JSON object, tolerating only one complete ``json`` fence.

        Some OpenAI-compatible providers ignore ``response_format`` and wrap an
        otherwise valid object in a Markdown JSON fence.  Accept that one
        mechanical wrapper, but never search free-form prose for an embedded
        object: the whole response must still be either JSON or one JSON fence.
        Schema validation remains a separate mandatory step.
        """

        raw = text if isinstance(text, str) else ""
        stripped = raw.strip()
        if not stripped:
            raise LLMError("empty JSON output")
        def _reject_constant(value: str) -> Any:
            raise LLMError(f"non-finite JSON number is not allowed: {value}")

        def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> Dict[str, Any]:
            result: Dict[str, Any] = {}
            for key, value in pairs:
                if key in result:
                    raise LLMError(f"duplicate JSON key is not allowed: {key}")
                result[key] = value
            return result

        def _finite_float(value: str) -> float:
            parsed = float(value)
            if not math.isfinite(parsed):
                raise LLMError(f"non-finite JSON number is not allowed: {value}")
            return parsed

        def _loads(value: str) -> Any:
            return json.loads(
                value,
                parse_constant=_reject_constant,
                parse_float=_finite_float,
                object_pairs_hook=_reject_duplicate_keys,
            )

        try:
            data = _loads(stripped)
        except json.JSONDecodeError:
            match = re.fullmatch(
                r"```json[ \t]*\r?\n(?P<body>[\s\S]*?)\r?\n```",
                stripped,
            )
            if match is None:
                raise
            data = _loads(match.group("body"))
        if not isinstance(data, dict):
            raise LLMError("LLM output is not a JSON object")
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
        self._increment_stat("calls")
        if self.fail:
            self._increment_stat("failures")
            raise LLMError("mock failure")
        if self.responder is not None:
            r = self.responder(system_prompt, user_prompt, None)
            return r if isinstance(r, str) else json.dumps(r)
        return "(mock surface text)"

    def generate_json(self, system_prompt, user_prompt, schema, *, temperature=0.2, max_tokens=1200):
        self._increment_stat("calls")
        if self.fail:
            self._increment_stat("failures")
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
OPENAI_JSON_TRANSPORTS = ("native", "prompt_only")

# max_tokens=UNCAPPED_OUTPUT sends no output ceiling, leaving the model's own
# maximum in force. Accounting still needs a worst case, so the ledger reserves
# ACCOUNTING_OUTPUT_CEILING per attempt for such a request.
UNCAPPED_OUTPUT = 0
ACCOUNTING_OUTPUT_CEILING = int(
    os.environ.get("ORG_LLM_ACCOUNTING_OUTPUT_CEILING", "32000") or 32000
)


@dataclass(frozen=True)
class ModelProfile:
    """What an endpoint honours for one model family.

    These were one boolean, ``_is_reasoning_model``, matched against the
    gpt-5/o1/o3/o4 name prefixes. That conflated four independent questions and
    answered all of them "no" for any model outside OpenAI's own naming, which
    is how a Claude run spent 144 ticks with its reasoning budget unset: the
    frozen ORG_LLM_REASONING_EFFORT=low was read, validated, recorded — and
    never sent, because the name did not start with "gpt-5".

    Measured against the bound endpoint on an identical prompt, 3 trials each:

        nothing sent                   843 completion tokens, 17.3s
        reasoning_effort=low           185 completion tokens,  5.9s
        reasoning_effort=none          943 completion tokens, 16.5s
        extra_body thinking disabled   863 completion tokens, 16.6s

    So the knob works, and not sending it costs 4.5x the output tokens and 3x
    the latency for a slightly *shorter* visible answer.
    """

    #: Provider honours an explicit reasoning/thinking budget request.
    reasoning_effort: bool
    #: Chat-completions field carrying the output ceiling.
    output_limit_field: str
    #: Provider accepts a caller-chosen temperature.
    accepts_temperature: bool
    #: Hidden reasoning is drawn from the same allowance as the answer, so a
    #: ceiling sized for the answer alone starves it.
    thinking_shares_output_budget: bool
    #: Sending no ceiling yields a provider-side default rather than the model's
    #: own maximum. Measured behind this OpenAI-compatible shim: a whole-file
    #: rewrite stopped at exactly 8192 tokens with finish_reason="length" and
    #: unparseable output, while the same call with an explicit ceiling
    #: finished normally. Anthropic's API requires max_tokens, so a shim in
    #: front of it must invent one; omitting the field concedes that choice.
    explicit_ceiling_when_uncapped: bool


_OPENAI_REASONING_PREFIXES = ("gpt-5", "o1", "o3", "o4")
_THINKING_CHAT_PREFIXES = ("claude-", "claude3", "claude-3")


def model_profile(model: str) -> ModelProfile:
    """Resolve one model name to what its endpoint actually accepts."""

    name = str(model or "").strip().lower()
    if name.startswith(_OPENAI_REASONING_PREFIXES):
        # Reasoning tokens are billed as completion tokens and the surface
        # renames the ceiling; a caller temperature is rejected outright.
        return ModelProfile(
            reasoning_effort=True,
            output_limit_field="max_completion_tokens",
            accepts_temperature=False,
            thinking_shares_output_budget=True,
            explicit_ceiling_when_uncapped=False,
        )
    if name.startswith(_THINKING_CHAT_PREFIXES):
        # A thinking model on the plain chat surface: it keeps max_tokens and
        # temperature, and — measured above — honours reasoning_effort.
        return ModelProfile(
            reasoning_effort=True,
            output_limit_field="max_tokens",
            accepts_temperature=True,
            thinking_shares_output_budget=True,
            explicit_ceiling_when_uncapped=True,
        )
    return ModelProfile(
        reasoning_effort=False,
        output_limit_field="max_tokens",
        accepts_temperature=True,
        thinking_shares_output_budget=False,
        explicit_ceiling_when_uncapped=False,
    )


class OpenAIOrgLLMClient(OrgLLMClient):
    provider = "openai"
    meters_actual_provider_attempts = True

    def __init__(self, *, model: str = "gpt-4o-mini", api_key: Optional[str] = None,
                 base_url: Optional[str] = None, reasoning_effort: str = "low",
                 max_retries: int = 2, retry_backoff_seconds: float = 1.0,
                 wire_api: str = "chat_completions",
                 json_transport: str = "native",
                 request_timeout_seconds: float = 120.0,
                 store_responses: bool = False,
                 default_headers: Mapping[str, str] | None = None,
                 azure_api_version: Optional[str] = None) -> None:
        super().__init__()
        self.model = model
        self._api_key = api_key
        self._base_url = base_url
        # Azure serves the same models behind a different SDK entry point, which
        # takes the endpoint and an API version rather than a base URL.
        self.azure_api_version = str(azure_api_version or "").strip() or None
        if self.azure_api_version and not self._base_url:
            raise ValueError("azure_api_version requires an Azure endpoint")
        wire_api = str(wire_api).strip().lower().replace("-", "_")
        if wire_api not in OPENAI_WIRE_APIS:
            raise ValueError(
                f"wire_api must be one of {OPENAI_WIRE_APIS}, got {wire_api!r}"
            )
        self.wire_api = wire_api
        json_transport = str(json_transport).strip().lower().replace("-", "_")
        if json_transport not in OPENAI_JSON_TRANSPORTS:
            raise ValueError(
                "json_transport must be one of "
                f"{OPENAI_JSON_TRANSPORTS}, got {json_transport!r}"
            )
        self.json_transport = json_transport
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

    @property
    def profile(self) -> ModelProfile:
        """What this model's endpoint accepts. Resolved per read so a client
        whose model is reassigned in a test does not answer for the old one."""
        return model_profile(self.model)

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
        profile = self.profile
        requested = max(0, int(max_tokens))
        if requested == UNCAPPED_OUTPUT:
            if profile.explicit_ceiling_when_uncapped:
                # Sending nothing does not reach the model's own maximum here,
                # it concedes the choice to the shim, which picked 8192 and
                # truncated a whole-file rewrite mid-answer. Name the number
                # accounting already reserves for this request, so the ledger
                # and the wire agree.
                return ACCOUNTING_OUTPUT_CEILING
            return None
        return (
            max(requested, 6000)
            if profile.thinking_shares_output_budget
            else requested
        )

    @property
    def routing_context_fingerprint(self) -> str:
        """Bind endpoint and credential routing without serializing secrets."""

        secret_payload = {
            "api_key": self._api_key or os.environ.get("OPENAI_API_KEY") or "",
            "default_headers": dict(sorted(self.default_headers.items())),
        }
        payload = {
            "base_url": str(self._base_url or "official_default"),
            "sdk_mode": "azure_openai" if self.azure_api_version else "openai",
            "azure_api_version": self.azure_api_version,
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
        """The reasoning_effort actually sent to the API — None for models whose
        endpoint ignores the knob. Run records read this attribute, so a run
        that could not spend its configured budget says so in its own record."""
        return self.reasoning_effort if self.profile.reasoning_effort else None

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
        if self.azure_api_version:
            # Imported here rather than beside OpenAI: a stand-in SDK that
            # supplies only OpenAI is enough for every non-Azure caller, and
            # naming AzureOpenAI in the shared import makes the whole client
            # unavailable to all of them.
            try:
                from openai import AzureOpenAI  # type: ignore
            except Exception as e:  # pragma: no cover - runtime only
                raise LLMError(f"azure openai sdk unavailable: {e}")
            kw["azure_endpoint"] = self._base_url
            kw["api_version"] = self.azure_api_version
            sdk_client = AzureOpenAI
        else:
            sdk_client = OpenAI
            if self._base_url:
                kw["base_url"] = self._base_url
        if self.default_headers:
            kw["default_headers"] = dict(self.default_headers)
        self._client_instance = sdk_client(**kw)
        return self._client_instance

    @staticmethod
    def _stop_provider_process(process: Any) -> None:
        """Bounded terminate -> join -> kill -> join, then close the handle."""

        try:
            try:
                process.join(timeout=0.02)
            except Exception:
                pass
            try:
                alive = process.is_alive()
            except Exception:
                alive = False
            if alive:
                try:
                    process.terminate()
                except Exception:
                    pass
                try:
                    process.join(timeout=0.20)
                except Exception:
                    pass
            try:
                alive = process.is_alive()
            except Exception:
                alive = False
            if alive:
                killer = getattr(process, "kill", None)
                if callable(killer):
                    try:
                        killer()
                    except Exception:
                        pass
                try:
                    process.join(timeout=0.50)
                except Exception:
                    pass
        finally:
            try:
                if not process.is_alive():
                    process.close()
            except Exception:
                pass

    def _provider_process_options(self) -> Dict[str, Any]:
        options: Dict[str, Any] = {
            "api_key": self._api_key or os.environ.get("OPENAI_API_KEY"),
        }
        if self._base_url:
            options["base_url"] = self._base_url
        if self.default_headers:
            options["default_headers"] = dict(self.default_headers)
        return options

    def _invoke_provider_process_attempt(
        self,
        surface: str,
        request: Mapping[str, Any],
        *,
        deadline: float,
        attempt_number: int,
        reserver: Optional[Callable[[int], None]],
    ) -> Any:
        """Run one SDK attempt in a spawn child that the parent can kill."""

        from environments.org_env.llm.provider_process import (
            PROVIDER_RESULT_BUFFER_BYTES,
            openai_provider_process_worker,
            restore_openai_response,
        )

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("OrgEnv OpenAI call exceeded its wall-clock budget")
        context = multiprocessing.get_context("spawn")
        ready_event = context.Event()
        go_event = context.Event()
        done_event = context.Event()
        result_buffer = context.RawArray("B", PROVIDER_RESULT_BUFFER_BYTES)
        result_length = context.RawValue("i", 0)
        # The request travels through shared memory, not through Process(args=).
        # Spawn hands a child its arguments down a default-sized pipe, and the
        # parent's flush of that pipe closes the one gap in this method's
        # deadline: everything after start() is polled against `deadline`, while
        # start() itself blocks for as long as the child takes to drain a
        # prompt-sized payload -- forever, if the child dies first. A b3m run sat
        # in Popen.__init__ for 40 minutes at t7 that way, with no timeout armed
        # and no child left to read. Results already came back this way.
        request_bytes = self._project_provider_request(request)
        request_buffer = context.RawArray("B", max(1, len(request_bytes)))
        request_buffer[: len(request_bytes)] = request_bytes
        request_length = context.RawValue("i", len(request_bytes))
        process = context.Process(
            target=openai_provider_process_worker,
            args=(
                ready_event,
                go_event,
                done_event,
                result_buffer,
                result_length,
                surface,
                self._provider_process_options(),
                request_buffer,
                request_length,
                remaining,
            ),
            name="orgenv-openai-attempt",
        )
        started = False
        admitted = False
        try:
            try:
                process.start()
                started = True
            except Exception as exc:
                raise LLMError(
                    "openai provider process setup failed "
                    f"({type(exc).__name__})"
                ) from None

            # Wait for either successful SDK/surface setup or a terminal setup
            # result.  Event polling and fixed shared memory avoid any recv()
            # that could block on a partial or oversized pipe frame.
            while not ready_event.is_set():
                if done_event.is_set():
                    break
                if process.exitcode is not None:
                    raise LLMError(
                        "openai provider child crashed before admission "
                        f"(exitcode={process.exitcode})"
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise LLMError("openai provider process setup timed out")
                ready_event.wait(timeout=min(0.01, remaining))

            if not ready_event.is_set():
                try:
                    envelope = self._read_provider_process_result(
                        result_buffer, result_length
                    )
                except Exception:
                    raise LLMError(
                        "openai provider process setup protocol failed"
                    ) from None
                exception_type = self._safe_child_exception_type(envelope)
                raise LLMError(
                    f"openai provider process setup failed ({exception_type})"
                )

            # No provider request can occur before these two parent-side steps.
            if process.exitcode is not None:
                raise LLMError(
                    "openai provider child crashed before admission "
                    f"(exitcode={process.exitcode})"
                )
            if reserver is not None:
                try:
                    reserver(attempt_number)
                except LLMError:
                    raise
                except Exception as exc:
                    raise LLMError(
                        "openai provider admission failed "
                        f"({type(exc).__name__})"
                    ) from None
            self._increment_stat("provider_attempts")
            admitted = True
            go_event.set()

            while not done_event.is_set():
                if process.exitcode is not None:
                    raise _ProviderAttemptFailure(
                        "openai provider child crashed after admission "
                        f"(exitcode={process.exitcode})"
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        "OrgEnv OpenAI call exceeded its wall-clock budget"
                    )
                done_event.wait(timeout=min(0.01, remaining))
            try:
                envelope = self._read_provider_process_result(
                    result_buffer, result_length
                )
            except Exception:
                raise _ProviderAttemptFailure(
                    "openai provider child protocol error after admission"
                ) from None
        finally:
            if started:
                self._stop_provider_process(process)

        if not isinstance(envelope, Mapping):
            raise _ProviderAttemptFailure("openai provider child protocol error")
        kind = envelope.get("kind")
        if kind == "success" and isinstance(envelope.get("response"), Mapping):
            return restore_openai_response(envelope["response"], surface)
        if kind == "provider_error":
            exception_type = self._safe_child_exception_type(envelope)
            raise _ProviderAttemptFailure(
                f"openai provider error ({exception_type})"
            )
        if kind == "setup_error" and not admitted:
            raise LLMError(
                "openai provider process setup failed "
                f"({self._safe_child_exception_type(envelope)})"
            )
        raise _ProviderAttemptFailure("openai provider child protocol error")

    @staticmethod
    def _project_provider_request(request: Mapping[str, Any]) -> bytes:
        """Encode the request for shared memory, refusing what will not survive.

        A request that cannot be projected has to fail here, in the parent, with
        a name: raised from inside the child it would arrive as a bare setup
        failure with no way to tell an unserializable request from a dead child.
        """

        try:
            return json.dumps(
                dict(request),
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise LLMError("openai provider request is not serializable") from exc

    @staticmethod
    def _read_provider_process_result(result_buffer: Any, result_length: Any) -> Any:
        length = int(result_length.value)
        if length <= 0 or length > len(result_buffer):
            raise RuntimeError("openai provider child protocol error")
        try:
            return json.loads(bytes(result_buffer[:length]).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("openai provider child protocol error") from exc

    @staticmethod
    def _safe_child_exception_type(envelope: Mapping[str, Any]) -> str:
        exception_type = str(envelope.get("exception_type") or "Exception")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]{0,127}", exception_type):
            return "Exception"
        return exception_type

    @staticmethod
    def _invoke_direct_attempt(client: Any, surface: str, request: Mapping[str, Any]) -> Any:
        if surface == "chat_completions":
            return client.chat.completions.create(**dict(request))
        if surface == "responses":
            return client.responses.create(**dict(request))
        raise LLMError("unsupported OpenAI wire API")

    def _invoke(self, surface: str, **request: Any) -> Any:
        """Call one SDK surface under one total wall-clock deadline."""

        provider_process = _requires_provider_process_deadline()
        if provider_process:
            client = None
        else:
            try:
                client = self._client()
            except LLMError:
                raise
            except Exception as exc:
                raise LLMError(
                    "openai provider client setup failed "
                    f"({type(exc).__name__})"
                ) from None
        attempt = 0
        deadline = time.monotonic() + self.request_timeout_seconds
        try:
            with (
                nullcontext()
                if provider_process
                else _openai_call_watchdog(
                    self.request_timeout_seconds,
                    label=f"OrgEnv {self.wire_api} call",
                )
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
                    if not provider_process:
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
                    # Admission is deliberately outside the provider exception
                    # handler.  A resource denial is terminal for this logical
                    # call; treating it as a transport failure would retry a
                    # request that the evaluator has explicitly refused.  A
                    # provider failure, by contrast, happens after this point
                    # and has therefore consumed one admitted attempt.
                    try:
                        if provider_process:
                            response = self._invoke_provider_process_attempt(
                                surface,
                                attempt_request,
                                deadline=deadline,
                                attempt_number=attempt + 1,
                                reserver=_PROVIDER_ATTEMPT_RESERVER.get(),
                            )
                        else:
                            reserver = _PROVIDER_ATTEMPT_RESERVER.get()
                            if reserver is not None:
                                try:
                                    reserver(attempt + 1)
                                except LLMError:
                                    raise
                                except Exception as exc:
                                    raise LLMError(
                                        "openai provider admission failed "
                                        f"({type(exc).__name__})"
                                    ) from None
                            self._increment_stat("provider_attempts")
                            response = self._invoke_direct_attempt(
                                client, surface, attempt_request
                            )
                    except TimeoutError:
                        # The total deadline is exhausted.  Do not count a
                        # retry that cannot actually be admitted or sent.
                        raise
                    except LLMError:
                        # Admission denials and child setup failures are
                        # terminal and must not consume an outer retry.
                        raise
                    except Exception as exc:
                        if attempt >= self.max_retries:
                            if isinstance(exc, _ProviderAttemptFailure):
                                message = str(exc)
                            else:
                                message = (
                                    "openai provider error "
                                    f"({type(exc).__name__})"
                                )
                            raise LLMError(message) from None
                        self._increment_stat("retries")
                        backoff = self.retry_backoff_seconds * (2 ** attempt)
                        if backoff >= deadline - time.monotonic():
                            raise TimeoutError(
                                "OrgEnv OpenAI retry would exceed its "
                                "wall-clock budget"
                            ) from None
                        time.sleep(backoff)
                        attempt += 1
                        continue
                    self._record_usage(getattr(response, "usage", None))
                    self._record_response_identity(response)
                    return response
        except TimeoutError as exc:
            raise LLMError(str(exc)) from None

    def _create_completion(self, **request: Any) -> Any:
        return self._invoke("chat_completions", **request)

    def _create_response(self, **request: Any) -> Any:
        return self._invoke("responses", **request)

    def _gen_kwargs(self, temperature: float, max_tokens: int) -> Dict[str, Any]:
        """The per-call knobs this model's endpoint accepts.

        The three axes are independent, which the previous single
        is-it-a-reasoning-model branch could not express: a thinking model on
        the plain chat surface keeps max_tokens and temperature *and* takes a
        reasoning budget. Answering that case with the non-reasoning branch is
        what left ORG_LLM_REASONING_EFFORT unsent for the whole Claude run.

        A None limit means the caller asked for no ceiling and this endpoint
        has its own maximum to fall back on, so the key is left out.
        """
        profile = self.profile
        limit = self.effective_output_token_limit(max_tokens)
        kwargs: Dict[str, Any] = {}
        if profile.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        if profile.accepts_temperature:
            kwargs["temperature"] = temperature
        if limit is not None:
            kwargs[profile.output_limit_field] = limit
        return kwargs

    def _response_kwargs(
        self,
        temperature: float,
        max_tokens: int,
    ) -> Dict[str, Any]:
        profile = self.profile
        kwargs: Dict[str, Any] = {"store": self.store_responses}
        limit = self.effective_output_token_limit(max_tokens)
        if limit is not None:
            kwargs["max_output_tokens"] = limit
        if profile.reasoning_effort:
            kwargs["reasoning"] = {"effort": self.reasoning_effort}
        if profile.accepts_temperature:
            kwargs["temperature"] = temperature
        return kwargs

    @staticmethod
    def _ceiling_note(response: Any) -> str:
        """A phrase naming the output ceiling as the cause, or "" if it was not.

        An answer cut off at the ceiling and an answer the model simply shaped
        wrongly arrive identically: text that will not parse. Only the provider
        distinguishes them, and it does, in finish_reason. Carrying that into
        the error means a run can tell "give it more room" apart from "it
        answered in prose", instead of counting both as one opaque failure --
        which is how a whole-file rewrite that stopped at exactly 8192 tokens
        was recorded as the model declining to answer.
        """
        choices = getattr(response, "choices", ()) or ()
        if choices and str(getattr(choices[0], "finish_reason", "")) == "length":
            return " (stopped at the output ceiling: finish_reason=length)"
        if str(getattr(response, "status", "")) == "incomplete":
            reason = getattr(
                getattr(response, "incomplete_details", None), "reason", ""
            )
            return f" (stopped at the output ceiling: {reason or 'incomplete'})"
        return ""

    def _decode(
        self, response: Any, body: Any, schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Parse and validate one answer, naming the ceiling when it was the cause."""
        try:
            return self._validate(self._parse_json_object(body), schema)
        except Exception as exc:
            note = self._ceiling_note(response)
            if not note:
                raise
            raise LLMError(f"{exc}{note}") from None

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
        self._increment_stat("calls")
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
                    raise LLMError(
                        "empty Responses API output"
                        + self._ceiling_note(response)
                    )
                return text
            r = self._create_completion(
                model=self.model, messages=[{"role": "system", "content": system_prompt},
                                            {"role": "user", "content": user_prompt}],
                **self._gen_kwargs(temperature, max_tokens))
            return (r.choices[0].message.content or "").strip()
        except LLMError:
            self._increment_stat("failures")
            raise
        except Exception as e:
            self._increment_stat("failures")
            raise LLMError(str(e))

    def generate_json(self, system_prompt, user_prompt, schema, *, temperature=0.2, max_tokens=1200):
        self._increment_stat("calls")
        try:
            if self.wire_api == "responses":
                request: Dict[str, Any] = {
                    "model": self.model,
                    "instructions": system_prompt + "\nReturn ONLY valid JSON.",
                    "input": user_prompt + self._schema_hint(schema),
                    **self._response_kwargs(temperature, max_tokens),
                }
                if self.json_transport == "native":
                    request["text"] = {"format": {"type": "json_object"}}
                response = self._create_response(**request)
                return self._decode(
                    response, self._response_text(response), schema
                )
            request = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt + "\nReturn ONLY valid JSON."},
                    {"role": "user", "content": user_prompt + self._schema_hint(schema)},
                ],
                **self._gen_kwargs(temperature, max_tokens),
            }
            if self.json_transport == "native":
                request["response_format"] = {"type": "json_object"}
            r = self._create_completion(**request)
            return self._decode(r, r.choices[0].message.content, schema)
        except LLMError:
            self._increment_stat("failures")
            raise
        except Exception as e:
            self._increment_stat("failures")
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
        self._increment_stat("calls")
        try:  # pragma: no cover - runtime only
            return self._post({"model": self.model, "system": system_prompt, "prompt": user_prompt,
                               "temperature": temperature, "max_tokens": max_tokens})
        except Exception as e:
            self._increment_stat("failures")
            raise LLMError(str(e))

    def generate_json(self, system_prompt, user_prompt, schema, *, temperature=0.2, max_tokens=1200):
        self._increment_stat("calls")
        try:  # pragma: no cover - runtime only
            txt = self._post({"model": self.model, "system": system_prompt + " Return ONLY JSON.",
                              "prompt": user_prompt, "temperature": temperature, "max_tokens": max_tokens})
            return self._validate(json.loads(txt), schema)
        except LLMError:
            self._increment_stat("failures")
            raise
        except Exception as e:
            self._increment_stat("failures")
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
    "ACCOUNTING_OUTPUT_CEILING",
    "GenericHTTPOrgLLMClient",
    "LLMError",
    "MockOrgLLMClient",
    "ModelProfile",
    "OPENAI_WIRE_APIS",
    "OPENAI_JSON_TRANSPORTS",
    "OpenAIOrgLLMClient",
    "OrgLLMClient",
    "REASONING_EFFORTS",
    "UNCAPPED_OUTPUT",
    "build_org_llm_client",
    "model_profile",
]
