"""Profile-aware subjective intent layer for v16.

The default provider is deterministic and local. It stands in for a frozen LLM
cache: the runtime can rely on its bounded JSON-like record, while future LLM
providers must preserve the same causal quarantine.
"""

from __future__ import annotations

import json
import math
import os
import re
import signal
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock, current_thread, main_thread
from types import FrameType
from typing import Any

from .hashing import canonicalize, stable_hash
from .openai_runtime import (
    build_openai_client,
    prepare_openai_response_request,
)
from .product_experience import artifact_price
from .schemas import (
    AgentState,
    Artifact,
    LLMIntentRecord,
    ProductExperiencePacket,
    clamp01,
)


DEFAULT_LLM_INTENT_LOW_MODEL = "gpt-5-nano-2025-08-07"
DEFAULT_LLM_INTENT_STANDARD_MODEL = "gpt-5-mini-2025-08-07"
DEFAULT_LLM_INTENT_HIGH_MODEL = "gpt-5.5-2026-04-23"
DEFAULT_LLM_INTENT_REQUEST_ATTEMPTS = 4
DEFAULT_LLM_INTENT_MAX_OUTPUT_TOKENS = 6_000
_LIVE_REVIEW_PROMPT_REVISION = "profile_conditioned_product_review_v19"
_LIVE_RESPONSE_CACHE_SCHEMA_VERSION = "live_intent_response_cache_v2"
_THEME_PATTERN = r"^[a-z][a-z0-9]*(?:_[a-z0-9]+){0,7}$"


_LIVE_REVIEW_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "intent_to_try": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "intent_to_pay": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "max_willingness_to_pay": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
        },
        "intent_to_recommend": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "subjective_satisfaction": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
        },
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "reason_code": {"type": "string", "minLength": 1, "maxLength": 64},
        "next_action_preference": {
            "type": "string",
            "minLength": 1,
            "maxLength": 64,
        },
        "experience_summary": {"type": "string", "minLength": 1, "maxLength": 800},
        "product_feedback": {"type": "string", "minLength": 1, "maxLength": 800},
        "suggested_improvements": {
            "type": "array",
            "minItems": 1,
            "maxItems": 5,
            "items": {"type": "string", "minLength": 1, "maxLength": 240},
        },
        "improvement_themes": {
            "type": "array",
            "minItems": 1,
            "maxItems": 5,
            "items": {
                "type": "string",
                "minLength": 1,
                "maxLength": 80,
                "pattern": _THEME_PATTERN,
            },
        },
        "feature_ideas": {
            "type": "array",
            "minItems": 0,
            "maxItems": 3,
            "items": {"type": "string", "minLength": 1, "maxLength": 240},
        },
        "feature_themes": {
            "type": "array",
            "minItems": 0,
            "maxItems": 3,
            "items": {
                "type": "string",
                "minLength": 1,
                "maxLength": 80,
                "pattern": _THEME_PATTERN,
            },
        },
    },
    "required": [
        "intent_to_try",
        "intent_to_pay",
        "max_willingness_to_pay",
        "intent_to_recommend",
        "subjective_satisfaction",
        "confidence",
        "reason_code",
        "next_action_preference",
        "experience_summary",
        "product_feedback",
        "suggested_improvements",
        "improvement_themes",
        "feature_ideas",
        "feature_themes",
    ],
}


class LiveLLMIntentRequestExhausted(RuntimeError):
    """Raised when strict live mode cannot obtain a valid provider response."""


class HeuristicLLMIntentProvider:
    source = "heuristic_frozen_llm_surrogate_v16"

    def generate(
        self, *, agent: AgentState, artifact: Artifact, tick: int
    ) -> LLMIntentRecord:
        profile = agent.profile
        experience = agent.latest_artifact_experience.get(artifact.id)
        peer_signal = agent.cognition.artifact_peer_signal_strength.get(
            artifact.id, 0.0
        )
        negative_signal = agent.cognition.artifact_negative_signal_strength.get(
            artifact.id, 0.0
        )
        social_proof = peer_signal * (
            0.7
            + 0.3
            * min(
                1.0,
                agent.cognition.artifact_peer_signal_count.get(artifact.id, 0) / 5.0,
            )
        )
        budget = agent.material_resources.food_or_budget_token
        price = artifact_price(artifact)
        budget_fit = clamp01((budget - price) / max(price, 0.001) * 0.5 + 0.5)

        if experience is None:
            objective_success = 0.5
            success_gain = 0.0
            time_saved = 0.0
            friction = 0.25
            observed_reliability = _mean(artifact.reliability_profile.values()) or 0.5
            experience_ref = None
        else:
            objective_success = experience.objective_success
            success_gain = experience.success_gain
            time_saved = experience.time_saved
            friction = experience.friction
            observed_reliability = experience.observed_reliability
            experience_ref = experience.experience_id

        satisfaction = clamp01(
            0.12
            + 0.34 * objective_success
            + 0.28 * success_gain
            + 0.20 * time_saved
            + 0.16 * observed_reliability * profile.reliability_preference
            + 0.14 * social_proof * profile.peer_susceptibility
            + 0.10 * profile.novelty_seeking
            - 0.26 * friction
            - 0.28 * negative_signal * profile.reliability_preference
            - 0.12 * agent.body.fatigue
            - 0.10 * agent.body.stress
        )
        intent_to_try = clamp01(
            0.16
            + 0.28 * profile.domain_need
            + 0.20 * profile.novelty_seeking
            + 0.18 * profile.risk_tolerance
            + 0.20 * peer_signal * profile.peer_susceptibility
            + 0.16 * agent.affect.curiosity
            - 0.22 * negative_signal
            - 0.14 * agent.body.fatigue
        )
        max_willingness_to_pay = clamp01(
            0.02
            + 0.22 * satisfaction
            + 0.18 * profile.domain_need
            + 0.18 * profile.deadline_pressure
            + 0.14 * budget_fit
            + 0.12 * social_proof
            - 0.24 * profile.budget_sensitivity
            - 0.16 * negative_signal
        )
        intent_to_pay = clamp01(
            0.08
            + 0.40 * satisfaction
            + 0.22 * max_willingness_to_pay
            + 0.18 * budget_fit
            + 0.16 * profile.risk_tolerance
            + 0.16 * social_proof
            - 0.25 * profile.budget_sensitivity
            - 0.20 * negative_signal
        )
        intent_to_recommend = clamp01(
            0.05
            + 0.36 * satisfaction
            + 0.20 * profile.social_activity
            + 0.18 * profile.opinion_leadership
            + 0.12 * observed_reliability
            - 0.24 * negative_signal
            - 0.16 * profile.privacy_sensitivity
        )
        confidence = clamp01(
            0.25
            + 0.26 * bool(experience)
            + 0.20 * observed_reliability
            + 0.18
            * min(
                1.0,
                agent.cognition.artifact_peer_signal_count.get(artifact.id, 0) / 4.0,
            )
            + 0.10 * profile.cognitive_capacity
            - 0.22 * agent.cognition.uncertainty
        )
        reason_code = _reason_code(
            satisfaction=satisfaction,
            intent_to_pay=intent_to_pay,
            negative_signal=negative_signal,
            budget_fit=budget_fit,
            reliability_preference=profile.reliability_preference,
            experience_seen=experience is not None,
        )
        next_action = _next_action_preference(
            intent_to_pay=intent_to_pay,
            intent_to_try=intent_to_try,
            intent_to_recommend=intent_to_recommend,
            experience_seen=experience is not None,
            negative_signal=negative_signal,
        )
        (
            experience_summary,
            product_feedback,
            suggested_improvements,
            feature_ideas,
        ) = _heuristic_review_text(
            experience=experience,
            reason_code=reason_code,
            next_action=next_action,
            profile=profile,
        )
        improvement_themes = _heuristic_improvement_themes(
            experience=experience,
        )
        feature_themes = _heuristic_feature_themes(
            profile=profile,
            experience=experience,
        )
        intent_id = stable_hash(
            {
                "agent_id": agent.id,
                "artifact_id": artifact.id,
                "tick": tick,
                "source": self.source,
                "profile": profile,
                "experience_ref": experience_ref,
                "satisfaction": satisfaction,
                "intent_to_pay": intent_to_pay,
            }
        )[:24]
        input_evidence_hash = stable_hash(
            _intent_input_payload(agent=agent, artifact=artifact, tick=tick)
        )
        output_evidence_hash = stable_hash(
            {
                "intent_to_try": intent_to_try,
                "intent_to_pay": intent_to_pay,
                "max_willingness_to_pay": max_willingness_to_pay,
                "intent_to_recommend": intent_to_recommend,
                "subjective_satisfaction": satisfaction,
                "confidence": confidence,
                "reason_code": reason_code,
                "next_action_preference": next_action,
                "experience_summary": experience_summary,
                "product_feedback": product_feedback,
                "suggested_improvements": suggested_improvements,
                "improvement_themes": improvement_themes,
                "feature_ideas": feature_ideas,
                "feature_themes": feature_themes,
            }
        )
        return LLMIntentRecord(
            intent_id=f"intent_{tick:06d}_{agent.id}_{artifact.id}_{intent_id}",
            artifact_id=artifact.id,
            tick=tick,
            source=self.source,
            profile_ref=profile.persona_label,
            experience_ref=experience_ref,
            intent_to_try=intent_to_try,
            intent_to_pay=intent_to_pay,
            max_willingness_to_pay=max_willingness_to_pay,
            intent_to_recommend=intent_to_recommend,
            subjective_satisfaction=satisfaction,
            confidence=confidence,
            reason_code=reason_code,
            next_action_preference=next_action,
            experience_summary=experience_summary,
            product_feedback=product_feedback,
            suggested_improvements=suggested_improvements,
            improvement_themes=improvement_themes,
            feature_ideas=feature_ideas,
            feature_themes=feature_themes,
            input_evidence_hash=input_evidence_hash,
            output_evidence_hash=output_evidence_hash,
        )


class OpenAILLMIntentProvider:
    """Live OpenAI-backed subjective intent provider.

    This provider is intentionally quarantined: it may only emit an
    `LLMIntentRecord`. The typed kernel still owns product experience, resource
    mutation, and final payment execution.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        timeout_seconds: float = 20.0,
        model_by_tier: dict[str, str] | None = None,
        request_attempts: int = DEFAULT_LLM_INTENT_REQUEST_ATTEMPTS,
        max_output_tokens: int = DEFAULT_LLM_INTENT_MAX_OUTPUT_TOKENS,
        strict_live: bool = False,
        response_cache_dir: str | Path | None = None,
        audit_path: str | Path | None = None,
        progress_path: str | Path | None = None,
        retry_backoff_seconds: float = 0.0,
        replay_reference_audit_path: str | Path | None = None,
    ) -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAILLMIntentProvider")
        if request_attempts <= 0:
            raise ValueError("request_attempts_must_be_positive")
        if max_output_tokens <= 0:
            raise ValueError("max_output_tokens_must_be_positive")
        if retry_backoff_seconds < 0:
            raise ValueError("retry_backoff_seconds_must_be_nonnegative")
        explicit_model = model is not None
        self.model = model or DEFAULT_LLM_INTENT_STANDARD_MODEL
        default_model_by_tier = {
            "low": os.environ.get(
                "SOCIETY_CORE_OPENAI_LOW_MODEL",
                self.model if explicit_model else DEFAULT_LLM_INTENT_LOW_MODEL,
            ),
            "standard": os.environ.get(
                "SOCIETY_CORE_OPENAI_STANDARD_MODEL",
                self.model,
            ),
            "high": os.environ.get(
                "SOCIETY_CORE_OPENAI_HIGH_MODEL",
                self.model if explicit_model else DEFAULT_LLM_INTENT_HIGH_MODEL,
            ),
        }
        self.model_by_tier = {
            tier: selected_model
            for tier, selected_model in {
                **default_model_by_tier,
                **(model_by_tier or {}),
            }.items()
            if selected_model
        }
        self.source = f"openai_live_v17:{self.model}"
        self.timeout_seconds = timeout_seconds
        self.request_attempts = request_attempts
        self.max_output_tokens = max_output_tokens
        self.strict_live = strict_live
        self.retry_backoff_seconds = retry_backoff_seconds
        self.response_cache_dir = (
            Path(response_cache_dir).expanduser().resolve()
            if response_cache_dir
            else None
        )
        self.audit_path = (
            Path(audit_path).expanduser().resolve() if audit_path else None
        )
        self.progress_path = (
            Path(progress_path).expanduser().resolve() if progress_path else None
        )
        self.replay_reference_audit_path = (
            Path(replay_reference_audit_path).expanduser().resolve()
            if replay_reference_audit_path
            else None
        )
        self._progress: dict[str, Any] = {
            "schema_version": "live_intent_progress_v1",
            "generate_call_count": 0,
            "fresh_success_count": 0,
            "cache_hit_count": 0,
            "fallback_count": 0,
            "exhausted_count": 0,
            "request_failure_attempt_count": 0,
            "model_counts": {},
        }
        if self.response_cache_dir:
            _ensure_private_directory(self.response_cache_dir)
        if self.audit_path:
            _ensure_private_directory(self.audit_path.parent)
        if self.progress_path:
            _ensure_private_directory(self.progress_path.parent)
        self._replay_reference = (
            _ReplayAuditReference(self.replay_reference_audit_path)
            if self.replay_reference_audit_path
            else None
        )
        self._fallback = HeuristicLLMIntentProvider()
        self._client = build_openai_client(
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            max_retries=0,
        )
        self._record_lock = Lock()

    def generate(
        self, *, agent: AgentState, artifact: Artifact, tick: int
    ) -> LLMIntentRecord:
        if self._replay_reference:
            self._replay_reference.assert_ready_for_tick(tick)
        payload = self._build_prompt_payload(agent=agent, artifact=artifact, tick=tick)
        model = self._model_for_agent(agent)
        source = f"openai_live_v17:{model}:tier_{agent.profile.llm_model_tier}"
        prompt = (
            "Act as exactly one user represented by the supplied profile inside a typed social simulation.\n"
            "You are not allowed to decide final purchase or mutate state. The kernel will validate all constraints.\n"
            "Judge only from this user's profile and the supplied objective product journey evidence. "
            "Do not use generic product knowledge or invent unobserved behavior.\n"
            "Report this user's subjective experience, constructive product feedback, and concrete improvements. "
            "Honor technical_role and innovation_role as separate attributes: professional engineers should "
            "give evidence-grounded technical diagnoses, while creative originators and early builders may "
            "propose feature ideas. Keep feature ideas separate from observed defects and return an empty "
            "feature_ideas list when the evidence and profile do not support one. For every suggested improvement, "
            "return one aligned short snake_case improvement theme at the same array index. For every feature idea, "
            "return one aligned short snake_case feature theme at the same array index. Theme labels must summarize "
            "the supplied text rather than import a roadmap or unobserved product knowledge. "
            "The numeric intents are advisory inputs; the typed kernel owns all actions and payment.\n"
            "All numeric values must be between 0 and 1. reason_code and next_action_preference must be "
            "short snake_case.\n\n"
            f"INPUT_JSON:\n{json.dumps(payload, sort_keys=True, separators=(',', ':'))}"
        )
        input_evidence_hash = stable_hash(payload)
        request_hash = stable_hash(
            {
                "schema_version": _LIVE_RESPONSE_CACHE_SCHEMA_VERSION,
                "prompt_revision": _LIVE_REVIEW_PROMPT_REVISION,
                "model": model,
                "prompt": prompt,
                "max_output_tokens": self.max_output_tokens,
                "json_schema": _LIVE_REVIEW_SCHEMA,
            }
        )
        cached, cache_error = self._load_cached_response(
            request_hash=request_hash,
            model=model,
            input_evidence_hash=input_evidence_hash,
        )
        if cached is not None:
            self._observe_replay_reference(
                agent=agent,
                tick=tick,
                model=model,
                request_hash=request_hash,
                input_evidence_hash=input_evidence_hash,
                output_evidence_hash=cached["output_evidence_hash"],
                response_id=str(cached.get("response_id", "")),
            )
            self._record_execution(
                status="cache_hit",
                agent=agent,
                tick=tick,
                model=model,
                request_hash=request_hash,
                input_evidence_hash=input_evidence_hash,
                output_evidence_hash=cached["output_evidence_hash"],
                attempts=0,
                errors=(),
                response_id=str(cached.get("response_id", "")),
            )
            return self._build_live_record(
                agent=agent,
                artifact=artifact,
                tick=tick,
                source=source,
                payload=payload,
                raw=cached["raw"],
                parsed=cached["parsed"],
            )

        raw = ""
        parsed: dict = {}
        last_error: Exception | None = None
        reason_prefix = "openai_live_error"
        errors: list[str] = []
        response_id = ""
        for attempt in range(1, self.request_attempts + 1):
            try:
                with _wall_clock_timeout(self.timeout_seconds):
                    request = _responses_request(
                        model=model,
                        prompt=prompt,
                        max_output_tokens=self.max_output_tokens,
                        timeout_seconds=self.timeout_seconds,
                        json_schema=_LIVE_REVIEW_SCHEMA,
                    )
                    response = self._client.responses.create(
                        **prepare_openai_response_request(request)
                    )
                raw = response.output_text.strip()
                parsed = _validate_live_review(_parse_json_object(raw))
                response_id = str(getattr(response, "id", "") or "")
                last_error = None
                break
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                last_error = exc
                reason_prefix = "openai_parse_error"
                errors.append(f"{reason_prefix}:{type(exc).__name__}")
            except Exception as exc:  # pragma: no cover - live network failures.
                last_error = exc
                reason_prefix = "openai_live_error"
                errors.append(f"{reason_prefix}:{type(exc).__name__}")
            if attempt < self.request_attempts and self.retry_backoff_seconds:
                time.sleep(
                    min(
                        self.retry_backoff_seconds * (2 ** (attempt - 1)),
                        8.0,
                    )
                )
        if last_error is not None:
            if cache_error:
                errors.insert(0, cache_error)
            self._record_execution(
                status="exhausted" if self.strict_live else "fallback",
                agent=agent,
                tick=tick,
                model=model,
                request_hash=request_hash,
                input_evidence_hash=input_evidence_hash,
                output_evidence_hash="",
                attempts=self.request_attempts,
                errors=tuple(errors),
                response_id="",
            )
            if self.strict_live:
                raise LiveLLMIntentRequestExhausted(
                    "live_llm_intent_request_exhausted:"
                    f"agent={agent.id}:tick={tick}:model={model}:"
                    f"reason={reason_prefix}:error={type(last_error).__name__}"
                ) from last_error
            return self._fallback_record(
                agent=agent,
                artifact=artifact,
                tick=tick,
                source=source,
                reason_prefix=reason_prefix,
                exc=last_error,
            )
        output_evidence_hash = stable_hash({"raw": raw, "parsed": parsed})
        self._store_cached_response(
            request_hash=request_hash,
            model=model,
            input_evidence_hash=input_evidence_hash,
            output_evidence_hash=output_evidence_hash,
            response_id=response_id,
            raw=raw,
            parsed=parsed,
        )
        combined_errors = tuple(([cache_error] if cache_error else []) + errors)
        self._observe_replay_reference(
            agent=agent,
            tick=tick,
            model=model,
            request_hash=request_hash,
            input_evidence_hash=input_evidence_hash,
            output_evidence_hash=output_evidence_hash,
            response_id=response_id,
        )
        self._record_execution(
            status="fresh_success",
            agent=agent,
            tick=tick,
            model=model,
            request_hash=request_hash,
            input_evidence_hash=input_evidence_hash,
            output_evidence_hash=output_evidence_hash,
            attempts=len(errors) + 1,
            errors=combined_errors,
            response_id=response_id,
        )
        return self._build_live_record(
            agent=agent,
            artifact=artifact,
            tick=tick,
            source=source,
            payload=payload,
            raw=raw,
            parsed=parsed,
        )

    def _build_live_record(
        self,
        *,
        agent: AgentState,
        artifact: Artifact,
        tick: int,
        source: str,
        payload: dict,
        raw: str,
        parsed: dict,
    ) -> LLMIntentRecord:
        experience = agent.latest_artifact_experience.get(artifact.id)
        experience_ref = experience.experience_id if experience else None
        reason_code = _safe_label(parsed["reason_code"], "llm_subjective_judgment")
        next_action = _safe_label(parsed["next_action_preference"], "wait")
        intent_to_try = float(parsed["intent_to_try"])
        intent_to_pay = float(parsed["intent_to_pay"])
        max_willingness_to_pay = float(parsed["max_willingness_to_pay"])
        intent_to_recommend = float(parsed["intent_to_recommend"])
        subjective_satisfaction = float(parsed["subjective_satisfaction"])
        confidence = float(parsed["confidence"])
        intent_hash = stable_hash(
            {
                "agent_id": agent.id,
                "artifact_id": artifact.id,
                "tick": tick,
                "source": source,
                "payload": payload,
                "raw": raw,
                "parsed": parsed,
            }
        )[:24]
        return LLMIntentRecord(
            intent_id=f"intent_{tick:06d}_{agent.id}_{artifact.id}_{intent_hash}",
            artifact_id=artifact.id,
            tick=tick,
            source=source,
            profile_ref=agent.profile.persona_label,
            experience_ref=experience_ref,
            intent_to_try=intent_to_try,
            intent_to_pay=intent_to_pay,
            max_willingness_to_pay=max_willingness_to_pay,
            intent_to_recommend=intent_to_recommend,
            subjective_satisfaction=subjective_satisfaction,
            confidence=confidence,
            reason_code=reason_code,
            next_action_preference=next_action,
            experience_summary=parsed["experience_summary"],
            product_feedback=parsed["product_feedback"],
            suggested_improvements=tuple(parsed["suggested_improvements"]),
            improvement_themes=tuple(parsed["improvement_themes"]),
            feature_ideas=tuple(parsed["feature_ideas"]),
            feature_themes=tuple(parsed["feature_themes"]),
            input_evidence_hash=stable_hash(payload),
            output_evidence_hash=stable_hash({"raw": raw, "parsed": parsed}),
        )

    def _model_for_agent(self, agent: AgentState) -> str:
        return self.model_by_tier.get(agent.profile.llm_model_tier, self.model)

    def _fallback_record(
        self,
        *,
        agent: AgentState,
        artifact: Artifact,
        tick: int,
        source: str,
        reason_prefix: str,
        exc: Exception,
    ) -> LLMIntentRecord:
        fallback = self._fallback.generate(agent=agent, artifact=artifact, tick=tick)
        fallback_hash = stable_hash(
            {
                "agent_id": agent.id,
                "artifact_id": artifact.id,
                "tick": tick,
                "source": source,
                "fallback_source": fallback.source,
                "reason_prefix": reason_prefix,
                "error_type": type(exc).__name__,
            }
        )[:24]
        return replace(
            fallback,
            intent_id=f"intent_{tick:06d}_{agent.id}_{artifact.id}_{fallback_hash}",
            source=f"{source}:fallback_heuristic",
            blocked_reason=f"{reason_prefix}:{type(exc).__name__}",
            input_evidence_hash=stable_hash(
                self._build_prompt_payload(agent=agent, artifact=artifact, tick=tick)
            ),
            output_evidence_hash=stable_hash(
                {
                    "reason_prefix": reason_prefix,
                    "error_type": type(exc).__name__,
                    "fallback_output_hash": fallback.output_evidence_hash,
                }
            ),
        )

    def _build_prompt_payload(
        self, *, agent: AgentState, artifact: Artifact, tick: int
    ) -> dict:
        return _intent_input_payload(agent=agent, artifact=artifact, tick=tick)

    def execution_summary(self) -> dict[str, Any]:
        if self._replay_reference:
            self._replay_reference.assert_complete()
        cache_entries: list[dict[str, str]] = []
        if self.response_cache_dir and self.response_cache_dir.exists():
            for path in sorted(self.response_cache_dir.glob("*/*.json")):
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if payload.get("schema_version") != _LIVE_RESPONSE_CACHE_SCHEMA_VERSION:
                    continue
                cache_entries.append(
                    {
                        "request_hash": str(payload.get("request_hash", "")),
                        "entry_hash": str(payload.get("entry_hash", "")),
                    }
                )
        with self._record_lock:
            progress = dict(self._progress)
            progress["model_counts"] = dict(self._progress.get("model_counts", {}))
        return canonicalize(
            {
                **progress,
                "request_attempts": self.request_attempts,
                "max_output_tokens": self.max_output_tokens,
                "strict_live": self.strict_live,
                "cache_enabled": self.response_cache_dir is not None,
                "cache_entry_count": len(cache_entries),
                "cache_manifest_hash": stable_hash(cache_entries),
                "audit_enabled": self.audit_path is not None,
                "replay_reference": (
                    self._replay_reference.summary()
                    if self._replay_reference
                    else {
                        "enabled": False,
                        "claim_ready": False,
                    }
                ),
            }
        )

    def _observe_replay_reference(
        self,
        *,
        agent: AgentState,
        tick: int,
        model: str,
        request_hash: str,
        input_evidence_hash: str,
        output_evidence_hash: str,
        response_id: str,
    ) -> None:
        if not self._replay_reference:
            return
        self._replay_reference.observe(
            agent_id=agent.id,
            tick=tick,
            model=model,
            request_hash=request_hash,
            input_evidence_hash=input_evidence_hash,
            output_evidence_hash=output_evidence_hash,
            response_id_hash=stable_hash(response_id) if response_id else "",
        )

    def _load_cached_response(
        self,
        *,
        request_hash: str,
        model: str,
        input_evidence_hash: str,
    ) -> tuple[dict[str, Any] | None, str | None]:
        if not self.response_cache_dir:
            return None, None
        path = _response_cache_path(self.response_cache_dir, request_hash)
        if not path.exists():
            return None, None
        try:
            if path.is_symlink() or not path.is_file():
                raise ValueError("unsafe_cache_entry")
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("schema_version") != _LIVE_RESPONSE_CACHE_SCHEMA_VERSION:
                raise ValueError("cache_schema_mismatch")
            if payload.get("request_hash") != request_hash:
                raise ValueError("cache_request_hash_mismatch")
            if payload.get("model") != model:
                raise ValueError("cache_model_mismatch")
            if payload.get("input_evidence_hash") != input_evidence_hash:
                raise ValueError("cache_input_hash_mismatch")
            raw = str(payload["raw"])
            parsed = _validate_live_review(dict(payload["parsed"]))
            output_hash = stable_hash({"raw": raw, "parsed": parsed})
            if payload.get("output_evidence_hash") != output_hash:
                raise ValueError("cache_output_hash_mismatch")
            entry_body = {
                key: value for key, value in payload.items() if key != "entry_hash"
            }
            if payload.get("entry_hash") != stable_hash(entry_body):
                raise ValueError("cache_entry_hash_mismatch")
            return payload, None
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            return None, f"cache_invalid:{type(exc).__name__}"

    def _store_cached_response(
        self,
        *,
        request_hash: str,
        model: str,
        input_evidence_hash: str,
        output_evidence_hash: str,
        response_id: str,
        raw: str,
        parsed: dict,
    ) -> None:
        if not self.response_cache_dir:
            return
        path = _response_cache_path(self.response_cache_dir, request_hash)
        _ensure_private_directory(path.parent)
        entry: dict[str, Any] = {
            "schema_version": _LIVE_RESPONSE_CACHE_SCHEMA_VERSION,
            "request_hash": request_hash,
            "model": model,
            "input_evidence_hash": input_evidence_hash,
            "output_evidence_hash": output_evidence_hash,
            "response_id": response_id,
            "raw": raw,
            "parsed": parsed,
        }
        entry["entry_hash"] = stable_hash(entry)
        _atomic_private_json_write(path, entry)

    def _record_execution(
        self,
        *,
        status: str,
        agent: AgentState,
        tick: int,
        model: str,
        request_hash: str,
        input_evidence_hash: str,
        output_evidence_hash: str,
        attempts: int,
        errors: tuple[str, ...],
        response_id: str,
    ) -> None:
        with self._record_lock:
            self._progress["generate_call_count"] += 1
            counter = {
                "fresh_success": "fresh_success_count",
                "cache_hit": "cache_hit_count",
                "fallback": "fallback_count",
                "exhausted": "exhausted_count",
            }[status]
            self._progress[counter] += 1
            self._progress["request_failure_attempt_count"] += len(
                tuple(
                    error for error in errors if not error.startswith("cache_invalid:")
                )
            )
            model_counts = self._progress["model_counts"]
            model_counts[model] = model_counts.get(model, 0) + 1
            now = datetime.now(timezone.utc).isoformat()
            event = {
                "schema_version": "live_intent_audit_event_v1",
                "recorded_at": now,
                "status": status,
                "agent_id": agent.id,
                "tick": tick,
                "model": model,
                "request_hash": request_hash,
                "input_evidence_hash": input_evidence_hash,
                "output_evidence_hash": output_evidence_hash,
                "attempts": attempts,
                "errors": errors,
                "response_id_hash": (stable_hash(response_id) if response_id else ""),
            }
            if self.audit_path:
                _append_private_json_line(self.audit_path, event)
            if self._replay_reference:
                self._progress["replay_reference"] = self._replay_reference.summary()
            self._progress.update(
                {
                    "last_status": status,
                    "last_agent_id": agent.id,
                    "last_tick": tick,
                    "last_model": model,
                    "last_request_hash": request_hash,
                    "updated_at": now,
                }
            )
            if self.progress_path:
                _atomic_private_json_write(
                    self.progress_path,
                    canonicalize(self._progress),
                )


class _ReplayAuditReference:
    def __init__(self, path: Path) -> None:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"unsafe_replay_reference_audit:{path}")
        success_events = []
        failure_boundary_events = []
        for line_number, raw_line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not raw_line.strip():
                continue
            try:
                event = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"replay_reference_invalid_json:line={line_number}"
                ) from exc
            if not isinstance(event, dict):
                raise ValueError(
                    f"replay_reference_event_not_object:line={line_number}"
                )
            if event.get("schema_version") != "live_intent_audit_event_v1":
                raise ValueError(f"replay_reference_schema_mismatch:line={line_number}")
            status = event.get("status")
            if status in {"fresh_success", "cache_hit"}:
                success_events.append(event)
                continue
            if status == "exhausted":
                if event.get("output_evidence_hash") or event.get("response_id_hash"):
                    raise ValueError(
                        f"replay_reference_exhausted_has_output:line={line_number}"
                    )
                failure_boundary_events.append(event)
                continue
            raise ValueError(
                "replay_reference_contains_non_success:"
                f"line={line_number}:status={status}"
            )
        all_events = success_events + failure_boundary_events
        if not all_events:
            raise ValueError("replay_reference_audit_empty")
        ticks = tuple(int(event["tick"]) for event in all_events)
        self.path = path
        self.partial_tick = max(ticks)
        self.complete_through_tick = self.partial_tick - 1
        invalid_boundary_ticks = tuple(
            sorted(
                {
                    int(event["tick"])
                    for event in failure_boundary_events
                    if int(event["tick"]) != self.partial_tick
                }
            )
        )
        if invalid_boundary_ticks:
            raise ValueError(
                "replay_reference_exhausted_before_partial_tick:"
                + ",".join(str(tick) for tick in invalid_boundary_ticks)
            )
        self._complete_expected: Counter[tuple[Any, ...]] = Counter()
        self._partial_expected: Counter[tuple[Any, ...]] = Counter()
        for event in success_events:
            key = self._event_key(event)
            target = (
                self._complete_expected
                if int(event["tick"]) <= self.complete_through_tick
                else self._partial_expected
            )
            target[key] += 1
        self._failure_boundary_expected: Counter[tuple[Any, ...]] = Counter(
            self._failure_boundary_key(event) for event in failure_boundary_events
        )
        self._complete_consumed: Counter[tuple[Any, ...]] = Counter()
        self._partial_consumed: Counter[tuple[Any, ...]] = Counter()
        self._failure_boundary_consumed: Counter[tuple[Any, ...]] = Counter()
        self._new_partial_count = 0
        self._lock = Lock()

    @staticmethod
    def _event_key(event: dict[str, Any]) -> tuple[Any, ...]:
        required = (
            "agent_id",
            "tick",
            "model",
            "request_hash",
            "input_evidence_hash",
            "output_evidence_hash",
            "response_id_hash",
        )
        missing = tuple(key for key in required if key not in event)
        if missing:
            raise ValueError("replay_reference_fields_missing:" + ",".join(missing))
        return (
            str(event["agent_id"]),
            int(event["tick"]),
            str(event["model"]),
            str(event["request_hash"]),
            str(event["input_evidence_hash"]),
            str(event["output_evidence_hash"]),
            str(event["response_id_hash"]),
        )

    @staticmethod
    def _failure_boundary_key(event: dict[str, Any]) -> tuple[Any, ...]:
        required = (
            "agent_id",
            "tick",
            "model",
            "request_hash",
            "input_evidence_hash",
        )
        missing = tuple(key for key in required if key not in event)
        if missing:
            raise ValueError(
                "replay_reference_boundary_fields_missing:" + ",".join(missing)
            )
        return (
            str(event["agent_id"]),
            int(event["tick"]),
            str(event["model"]),
            str(event["request_hash"]),
            str(event["input_evidence_hash"]),
        )

    def assert_ready_for_tick(self, tick: int) -> None:
        if tick <= self.partial_tick:
            return
        self.assert_complete()

    def observe(
        self,
        *,
        agent_id: str,
        tick: int,
        model: str,
        request_hash: str,
        input_evidence_hash: str,
        output_evidence_hash: str,
        response_id_hash: str,
    ) -> None:
        key = (
            agent_id,
            tick,
            model,
            request_hash,
            input_evidence_hash,
            output_evidence_hash,
            response_id_hash,
        )
        with self._lock:
            if tick <= self.complete_through_tick:
                if self._complete_consumed[key] >= self._complete_expected[key]:
                    raise RuntimeError(
                        "replay_reference_mismatch:"
                        f"agent={agent_id}:tick={tick}:model={model}:"
                        f"input={input_evidence_hash[:12]}:"
                        f"request={request_hash[:12]}:"
                        f"output={output_evidence_hash[:12]}"
                    )
                self._complete_consumed[key] += 1
                return
            if tick == self.partial_tick:
                if self._partial_consumed[key] < self._partial_expected[key]:
                    self._partial_consumed[key] += 1
                elif (
                    self._failure_boundary_consumed[key[:5]]
                    < self._failure_boundary_expected[key[:5]]
                ):
                    self._failure_boundary_consumed[key[:5]] += 1
                else:
                    self._new_partial_count += 1

    def assert_complete(self) -> None:
        with self._lock:
            complete_remaining = sum(
                (self._complete_expected - self._complete_consumed).values()
            )
            partial_remaining = sum(
                (self._partial_expected - self._partial_consumed).values()
            )
            failure_boundary_remaining = sum(
                (
                    self._failure_boundary_expected - self._failure_boundary_consumed
                ).values()
            )
            if complete_remaining or partial_remaining or failure_boundary_remaining:
                raise RuntimeError(
                    "replay_reference_unconsumed:"
                    f"complete={complete_remaining}:partial={partial_remaining}:"
                    f"failure_boundary={failure_boundary_remaining}"
                )

    def summary(self) -> dict[str, Any]:
        with self._lock:
            complete_expected = sum(self._complete_expected.values())
            complete_consumed = sum(self._complete_consumed.values())
            partial_expected = sum(self._partial_expected.values())
            partial_consumed = sum(self._partial_consumed.values())
            failure_boundary_expected = sum(self._failure_boundary_expected.values())
            failure_boundary_consumed = sum(self._failure_boundary_consumed.values())
            return {
                "enabled": True,
                "reference_path": str(self.path),
                "complete_through_tick": self.complete_through_tick,
                "partial_tick": self.partial_tick,
                "complete_expected_count": complete_expected,
                "complete_consumed_count": complete_consumed,
                "complete_remaining_count": (complete_expected - complete_consumed),
                "partial_expected_count": partial_expected,
                "partial_consumed_count": partial_consumed,
                "partial_remaining_count": partial_expected - partial_consumed,
                "failure_boundary_expected_count": (failure_boundary_expected),
                "failure_boundary_consumed_count": (failure_boundary_consumed),
                "failure_boundary_remaining_count": (
                    failure_boundary_expected - failure_boundary_consumed
                ),
                "new_partial_count": self._new_partial_count,
                "claim_ready": (
                    complete_expected == complete_consumed
                    and partial_expected == partial_consumed
                    and failure_boundary_expected == failure_boundary_consumed
                ),
            }


def _response_cache_path(cache_dir: Path, request_hash: str) -> Path:
    return cache_dir / request_hash[:2] / f"{request_hash}.json"


def _ensure_private_directory(path: Path) -> None:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    if not path.is_dir() or path.is_symlink():
        raise RuntimeError(f"unsafe_private_directory:{path}")
    path.chmod(0o700)


def _atomic_private_json_write(path: Path, payload: Any) -> None:
    _ensure_private_directory(path.parent)
    encoded = (
        json.dumps(
            canonicalize(payload),
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        path.chmod(0o600)
    finally:
        temporary.unlink(missing_ok=True)


def _append_private_json_line(path: Path, payload: Any) -> None:
    _ensure_private_directory(path.parent)
    encoded = (
        json.dumps(
            canonicalize(payload),
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_APPEND,
        0o600,
    )
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    path.chmod(0o600)


def _intent_input_payload(*, agent: AgentState, artifact: Artifact, tick: int) -> dict:
    return canonicalize(
        {
            "tick": tick,
            "agent_id": agent.id,
            "profile": agent.profile,
            "body": agent.body,
            "affect": agent.affect,
            "cognition": {
                "uncertainty": agent.cognition.uncertainty,
                "artifact_belief": agent.cognition.beliefs_about_artifacts.get(
                    artifact.id, 0.5
                ),
                "artifact_peer_signal_strength": (
                    agent.cognition.artifact_peer_signal_strength.get(artifact.id, 0.0)
                ),
                "artifact_peer_signal_count": (
                    agent.cognition.artifact_peer_signal_count.get(artifact.id, 0)
                ),
                "artifact_negative_signal_strength": (
                    agent.cognition.artifact_negative_signal_strength.get(
                        artifact.id, 0.0
                    )
                ),
            },
            "resources": agent.material_resources,
            "budgets": agent.action_budgets,
            "artifact": _user_model_artifact_payload(artifact),
            "artifact_price": artifact_price(artifact),
            "latest_product_experience": _user_model_experience_payload(
                agent.latest_artifact_experience.get(artifact.id)
            ),
            "payment_state": agent.artifact_payment_state.get(artifact.id),
        }
    )


def _user_model_artifact_payload(artifact: Artifact) -> dict:
    """Expose only base-release public facts, not evaluator-authored score labels."""

    return canonicalize(
        {
            "id": artifact.id,
            "provider_id": artifact.provider_id,
            "artifact_kind": artifact.artifact_kind,
            "version": artifact.version,
            "release_time": artifact.release_time,
            "public_claims": artifact.public_claims,
            "evidence_claims": artifact.evidence_claims,
            "cost_profile": artifact.cost_profile,
            "access_constraints": artifact.access_constraints,
        }
    )


def intent_input_evidence_hash(
    *,
    agent: AgentState,
    artifact: Artifact,
    tick: int,
) -> str:
    """Return the provider-neutral commitment to one subjective review input."""

    return stable_hash(_intent_input_payload(agent=agent, artifact=artifact, tick=tick))


def _user_model_experience_payload(
    experience: ProductExperiencePacket | None,
) -> dict | None:
    if experience is None:
        return None
    payload = canonicalize(experience)
    payload["experienced_task_success"] = payload.pop("objective_success")
    return payload


def payment_block_reason(agent: AgentState, artifact: Artifact | None) -> str | None:
    if artifact is None:
        return "missing_artifact"
    if artifact.id not in agent.artifact_access:
        return "no_artifact_access"
    intent = agent.latest_llm_intents.get(artifact.id)
    if intent is None:
        return "missing_subjective_intent"
    if artifact.id not in agent.latest_artifact_experience:
        return "no_prior_product_experience"
    price = artifact_price(artifact)
    if intent.max_willingness_to_pay < price:
        return "willingness_below_price"
    if intent.intent_to_pay < 0.5:
        return "intent_below_purchase_threshold"
    if agent.material_resources.food_or_budget_token < price:
        return "insufficient_budget"
    return None


def feedback_intent_record(
    *,
    agent: AgentState,
    artifact_id: str,
    tick: int,
    blocked_reason: str | None,
) -> LLMIntentRecord:
    base = agent.latest_llm_intents.get(artifact_id)
    if base is None:
        base = LLMIntentRecord(
            intent_id=f"intent_{tick:06d}_{agent.id}_{artifact_id}_missing",
            artifact_id=artifact_id,
            tick=tick,
            source="kernel_feedback_v17",
            profile_ref=agent.profile.persona_label,
            experience_ref=agent.latest_artifact_experience.get(
                artifact_id
            ).experience_id
            if artifact_id in agent.latest_artifact_experience
            else None,
            intent_to_try=0.0,
            intent_to_pay=0.0,
            max_willingness_to_pay=0.0,
            intent_to_recommend=0.0,
            subjective_satisfaction=0.0,
            confidence=0.0,
            reason_code="missing_intent",
            next_action_preference="wait",
        )
    return replace(
        base,
        tick=tick,
        source="kernel_feedback_v17",
        blocked_reason=blocked_reason or "payment_executed",
    )


def store_intent_feedback(agent: AgentState, record: LLMIntentRecord) -> None:
    agent.latest_llm_intents[record.artifact_id] = record
    agent.llm_intent_feedback_history.append(record)
    if len(agent.llm_intent_feedback_history) > 16:
        del agent.llm_intent_feedback_history[:-16]


def _parse_json_object(raw: str) -> dict:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            raise ValueError(
                f"OpenAI intent provider returned non-JSON text: {raw[:200]!r}"
            ) from None
        value = json.loads(raw[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("OpenAI intent provider returned JSON that is not an object")
    return value


def _bounded_number(value, default: float) -> float:
    try:
        return clamp01(float(value))
    except (TypeError, ValueError):
        return clamp01(default)


def _validate_live_review(value: dict) -> dict:
    missing = [key for key in _LIVE_REVIEW_SCHEMA["required"] if key not in value]
    if missing:
        raise ValueError(
            f"OpenAI intent review missing required fields: {','.join(missing)}"
        )
    numeric_fields = (
        "intent_to_try",
        "intent_to_pay",
        "max_willingness_to_pay",
        "intent_to_recommend",
        "subjective_satisfaction",
        "confidence",
    )
    normalized = dict(value)
    for key in numeric_fields:
        raw = value[key]
        if isinstance(raw, bool):
            raise ValueError(f"OpenAI intent review field {key} must be numeric")
        try:
            number = float(raw)
        except (TypeError, ValueError):
            raise ValueError(
                f"OpenAI intent review field {key} must be numeric"
            ) from None
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            raise ValueError(f"OpenAI intent review field {key} must be in [0,1]")
        normalized[key] = number
    normalized["reason_code"] = _required_text(value["reason_code"], 64, "reason_code")
    normalized["next_action_preference"] = _required_text(
        value["next_action_preference"], 64, "next_action_preference"
    )
    normalized["experience_summary"] = _required_text(
        value["experience_summary"], 800, "experience_summary"
    )
    normalized["product_feedback"] = _required_text(
        value["product_feedback"], 800, "product_feedback"
    )
    improvements = value["suggested_improvements"]
    if not isinstance(improvements, list) or not 1 <= len(improvements) <= 5:
        raise ValueError(
            "OpenAI intent review suggested_improvements must contain 1-5 items"
        )
    normalized["suggested_improvements"] = [
        _required_text(item, 240, "suggested_improvements") for item in improvements
    ]
    improvement_themes = _validated_theme_list(
        value["improvement_themes"],
        field_name="improvement_themes",
        maximum=5,
    )
    if len(improvement_themes) != len(improvements):
        raise ValueError(
            "OpenAI intent review improvement_themes must align with "
            "suggested_improvements"
        )
    normalized["improvement_themes"] = improvement_themes
    feature_ideas = value["feature_ideas"]
    if not isinstance(feature_ideas, list) or len(feature_ideas) > 3:
        raise ValueError("OpenAI intent review feature_ideas must contain 0-3 items")
    normalized["feature_ideas"] = [
        _required_text(item, 240, "feature_ideas") for item in feature_ideas
    ]
    feature_themes = _validated_theme_list(
        value["feature_themes"],
        field_name="feature_themes",
        maximum=3,
        allow_empty=True,
    )
    if len(feature_themes) != len(feature_ideas):
        raise ValueError(
            "OpenAI intent review feature_themes must align with feature_ideas"
        )
    normalized["feature_themes"] = feature_themes
    return normalized


def _validated_theme_list(
    value,
    *,
    field_name: str,
    maximum: int,
    allow_empty: bool = False,
) -> list[str]:
    minimum = 0 if allow_empty else 1
    if not isinstance(value, list) or not minimum <= len(value) <= maximum:
        raise ValueError(
            f"OpenAI intent review {field_name} must contain {minimum}-{maximum} items"
        )
    themes = [_required_text(item, 80, field_name) for item in value]
    if any(re.fullmatch(_THEME_PATTERN, theme) is None for theme in themes):
        raise ValueError(f"OpenAI intent review {field_name} must use short snake_case")
    return themes


def _required_text(value, max_length: int, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"OpenAI intent review field {field_name} must be text")
    normalized = " ".join(value.split()).strip()
    if not normalized:
        raise ValueError(f"OpenAI intent review field {field_name} must not be empty")
    return normalized[:max_length]


def _safe_label(value, default: str) -> str:
    if not isinstance(value, str):
        return default
    cleaned = "".join(char if char.isalnum() else "_" for char in value.strip().lower())
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return cleaned[:64] or default


def _reason_code(
    *,
    satisfaction: float,
    intent_to_pay: float,
    negative_signal: float,
    budget_fit: float,
    reliability_preference: float,
    experience_seen: bool,
) -> str:
    if not experience_seen:
        return "needs_firsthand_experience"
    if negative_signal > 0.55 and reliability_preference > 0.55:
        return "reliability_concern"
    if budget_fit < 0.45:
        return "budget_pressure"
    if satisfaction > 0.62 and intent_to_pay > 0.55:
        return "positive_value_judgment"
    if satisfaction > 0.45:
        return "cautious_value_judgment"
    return "low_perceived_value"


def _next_action_preference(
    *,
    intent_to_pay: float,
    intent_to_try: float,
    intent_to_recommend: float,
    experience_seen: bool,
    negative_signal: float,
) -> str:
    if intent_to_pay >= 0.62:
        return "pay_if_kernel_allows"
    if not experience_seen and intent_to_try >= 0.45:
        return "try_first"
    if negative_signal >= 0.55:
        return "warn_or_ask_community"
    if intent_to_recommend >= 0.52:
        return "share_success"
    if intent_to_try >= 0.45:
        return "try_or_reuse"
    return "wait"


def _mean(values) -> float:
    vals = [float(value) for value in values]
    return sum(vals) / len(vals) if vals else 0.0


def _heuristic_review_text(
    *,
    experience,
    reason_code: str,
    next_action: str,
    profile,
):
    if experience is None:
        return (
            "No firsthand product journey was available.",
            f"The current judgment is provisional ({reason_code}).",
            (
                "Provide a firsthand task-level product trial before purchase evaluation.",
            ),
            (),
        )
    outcome = "completed" if experience.objective_success >= 0.5 else "did not complete"
    summary = (
        f"The {experience.task_type} task {outcome}; friction was "
        f"{experience.friction:.2f} and observed reliability was "
        f"{experience.observed_reliability:.2f}."
    )
    issue = experience.failure_event or experience.blocked_stage
    is_technical_reviewer = profile.technical_role in {
        "professional_engineer",
        "technical_practitioner",
    }
    if issue and is_technical_reviewer:
        stage = experience.blocked_stage or "unresolved_stage"
        feedback = (
            f"Observed {issue} at {stage}; diagnostic clarity was "
            f"{experience.diagnostic_clarity:.2f} and workaround success was "
            f"{experience.workaround_success:.2f}."
        )
        improvements = (
            f"Add structured diagnostics and recovery at {stage} for {issue}, "
            "linked to reproducible failure evidence.",
        )
    elif issue:
        feedback = f"The main observed issue was {issue}."
        improvements = (f"Improve diagnostics and recovery for {issue}.",)
    elif experience.friction >= 0.45:
        feedback = (
            f"The evidence supports a {reason_code} response and {next_action} next."
        )
        improvements = (
            "Reduce setup and integration friction in the observed journey.",
        )
    else:
        feedback = (
            f"The evidence supports a {reason_code} response and {next_action} next."
        )
        improvements = (
            "Preserve the successful path and document the observed workflow.",
        )
    return (
        summary,
        feedback,
        improvements,
        _heuristic_feature_ideas(profile=profile, experience=experience),
    )


def _heuristic_feature_ideas(*, profile, experience) -> tuple[str, ...]:
    if profile.innovation_role not in {"creative_originator", "early_builder"}:
        return ()
    if profile.creative_capacity < 0.62:
        return ()
    if experience.integration_quality < 0.50:
        return (
            f"Add a programmable workflow composition surface for {experience.task_type}.",
        )
    if experience.first_value_time > 0.55:
        return (
            f"Add reusable guided templates for the {experience.task_type} journey.",
        )
    if experience.repeat_use_value < 0.45:
        return (
            f"Add collaborative reuse and sharing for {experience.task_type} workflows.",
        )
    return (
        f"Add optional automation around the successful {experience.task_type} path.",
    )


def _heuristic_improvement_themes(*, experience) -> tuple[str, ...]:
    if experience is None:
        return ("firsthand_product_trial",)
    if experience.failure_event or experience.blocked_stage:
        return ("diagnostics_and_recovery",)
    if experience.friction >= 0.45:
        return ("workflow_integration",)
    return ("workflow_documentation",)


def _heuristic_feature_themes(*, profile, experience) -> tuple[str, ...]:
    if experience is None:
        return ()
    if not _heuristic_feature_ideas(profile=profile, experience=experience):
        return ()
    if experience.integration_quality < 0.50:
        return ("programmable_workflows",)
    if experience.first_value_time > 0.55:
        return ("guided_workflow_templates",)
    if experience.repeat_use_value < 0.45:
        return ("collaborative_reuse",)
    return ("workflow_automation",)


def _responses_request(
    *,
    model: str,
    prompt: str,
    max_output_tokens: int,
    timeout_seconds: float,
    json_schema: dict | None = None,
) -> dict:
    request = {
        "model": model,
        "input": prompt,
        "max_output_tokens": max_output_tokens,
        "timeout": timeout_seconds,
    }
    if json_schema is not None:
        request["text"] = {
            "format": {
                "type": "json_schema",
                "name": "profile_conditioned_product_review",
                "strict": True,
                "schema": json_schema,
            }
        }
    if not model.startswith("gpt-5"):
        request["temperature"] = 0
    return request


@contextmanager
def _wall_clock_timeout(timeout_seconds: float):
    """Interrupt blocking provider calls even when the HTTP stack does not."""
    if (
        timeout_seconds <= 0
        or not hasattr(signal, "SIGALRM")
        or current_thread() is not main_thread()
    ):
        yield
        return

    def _raise_timeout(signum: int, frame: FrameType | None) -> None:
        raise TimeoutError(f"LLM intent call exceeded {timeout_seconds:.3f}s")

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, 0)
    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0 or previous_timer[1] > 0:
            signal.setitimer(signal.ITIMER_REAL, previous_timer[0], previous_timer[1])
