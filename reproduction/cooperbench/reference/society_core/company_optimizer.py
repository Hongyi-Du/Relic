"""Company-side product optimization agents.

The company optimizer is separated from external society agents. It can only
read public feedback reports and public traces, then propose a product direction.
It never receives private agent states or hidden society variables.
"""

from __future__ import annotations

import json
import os
import signal
from contextlib import contextmanager
from dataclasses import replace
from time import sleep
from types import FrameType

from .hashing import canonicalize, stable_hash
from .openai_runtime import (
    build_openai_client,
    prepare_openai_response_request,
)
from .org_adapter import (
    CompanyOptimizationProposal,
    CompanyPublicFeedbackReport,
    CompanyVisibleTrace,
    propose_company_optimization_from_public_feedback,
)

DEFAULT_COMPANY_OPTIMIZER_MODEL = "gpt-5.5-pro"


class HeuristicCompanyOptimizer:
    source = "heuristic_company_optimizer_v17"

    def propose(
        self,
        *,
        report: CompanyPublicFeedbackReport,
        artifact_id: str,
        target_improvement_themes: tuple[str, ...] = (),
        public_traces: tuple[CompanyVisibleTrace, ...] = (),
    ) -> CompanyOptimizationProposal:
        proposal = propose_company_optimization_from_public_feedback(
            report,
            artifact_id=artifact_id,
            target_improvement_themes=target_improvement_themes,
        )
        return replace(
            proposal,
            optimizer_source=self.source,
            optimizer_model=None,
            optimizer_reasoning="deterministic_public_feedback_mapping",
        )


class OpenAICompanyOptimizer:
    """High-capability company optimizer backed by OpenAI.

    The prompt is intentionally narrow: public report, bounded public trace
    summaries, and no hidden society state. Historical next-version targets are
    evaluator-only and are never included in optimizer inputs.
    """

    source = "openai_company_optimizer_v17"

    def __init__(
        self,
        *,
        model: str | None = None,
        timeout_seconds: float = 300.0,
        strict_live: bool = False,
        request_attempts: int = 2,
        retry_backoff_seconds: float = 0.0,
    ) -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAICompanyOptimizer")
        if request_attempts < 1:
            raise ValueError("company_optimizer_request_attempts_must_be_positive")
        if retry_backoff_seconds < 0:
            raise ValueError("company_optimizer_retry_backoff_must_be_nonnegative")
        self.model = (
            model
            or os.environ.get("SOCIETY_CORE_COMPANY_OPTIMIZER_MODEL")
            or os.environ.get("SOCIETY_CORE_OPENAI_HIGH_MODEL")
            or DEFAULT_COMPANY_OPTIMIZER_MODEL
        )
        self.timeout_seconds = timeout_seconds
        self.strict_live = strict_live
        self.request_attempts = request_attempts
        self.retry_backoff_seconds = retry_backoff_seconds
        self._client = build_openai_client(
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            max_retries=0,
        )
        self._fallback = HeuristicCompanyOptimizer()

    def propose(
        self,
        *,
        report: CompanyPublicFeedbackReport,
        artifact_id: str,
        target_improvement_themes: tuple[str, ...] = (),
        public_traces: tuple[CompanyVisibleTrace, ...] = (),
    ) -> CompanyOptimizationProposal:
        fallback = self._fallback.propose(
            report=report,
            artifact_id=artifact_id,
            public_traces=public_traces,
        )
        del target_improvement_themes
        if not fallback.priority_themes:
            return replace(
                fallback,
                optimizer_source=f"{self.source}:kernel_deferred",
                optimizer_model=self.model,
                optimizer_reasoning="no_kernel_adoptable_themes",
            )
        payload = _optimizer_payload(
            report=report,
            artifact_id=artifact_id,
            public_traces=public_traces,
            kernel_proposal=fallback,
        )
        prompt = (
            "You are the company's strongest internal product optimization agent.\n"
            "Use only the provided public society feedback and public trace summaries.\n"
            "Do not infer private user states, hidden preferences, or non-public facts.\n"
            "You may only reorder the kernel_adoptable_themes supplied in the input.\n"
            "Include every kernel-adoptable theme exactly once; do not promote deferred or rejected themes.\n"
            "Return only JSON with keys: requested_direction, priority_themes, reasoning.\n"
            "requested_direction must be a short snake_case string.\n"
            "priority_themes must be an ordered array of 1 to 24 short snake_case strings.\n"
            "reasoning must be one concise sentence grounded in public evidence.\n\n"
            f"INPUT_JSON:\n{json.dumps(payload, sort_keys=True, separators=(',', ':'))}"
        )
        request = {
            "model": self.model,
            "input": prompt,
            "max_output_tokens": 8000,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "company_optimization_proposal",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "requested_direction": {"type": "string"},
                            "priority_themes": {
                                "type": "array",
                                "minItems": 1,
                                "maxItems": 24,
                                "items": {"type": "string"},
                            },
                            "reasoning": {"type": "string"},
                        },
                        "required": [
                            "requested_direction",
                            "priority_themes",
                            "reasoning",
                        ],
                    },
                }
            },
            "timeout": self.timeout_seconds,
        }
        if _supports_temperature(self.model):
            request["temperature"] = 0
        if self.model.startswith("gpt-5.5"):
            request["reasoning"] = {"effort": "xhigh"}
        raw = ""
        parsed = None
        last_error: Exception | None = None
        last_failure_kind = "openai_live_error"
        for attempt in range(self.request_attempts):
            attempt_request = dict(request)
            if attempt:
                attempt_request["input"] = (
                    prompt + "\n\nA prior request failed or returned invalid output. "
                    "Regenerate one complete JSON object from the original input."
                )
            try:
                with _wall_clock_timeout(self.timeout_seconds):
                    response = self._client.responses.create(
                        **prepare_openai_response_request(attempt_request)
                    )
                raw = _response_text(response)
                parsed = _parse_json_object(raw)
                break
            except Exception as exc:  # pragma: no cover - live provider boundary.
                last_error = exc
                last_failure_kind = "openai_parse_error" if raw else "openai_live_error"
                if attempt + 1 < self.request_attempts:
                    sleep(self.retry_backoff_seconds * (2**attempt))
        if parsed is None:
            assert last_error is not None
            failure = (
                f"{last_failure_kind}:{type(last_error).__name__}:"
                f"attempts={self.request_attempts}"
            )
            if self.strict_live:
                raise RuntimeError(
                    f"live_company_optimizer_request_exhausted:{failure}"
                ) from last_error
            return replace(
                fallback,
                optimizer_source=f"{self.source}:fallback_heuristic",
                optimizer_model=self.model,
                optimizer_reasoning=failure,
            )
        requested_direction = _safe_label(
            parsed.get("requested_direction"),
            fallback.requested_direction,
        )
        model_themes = _safe_theme_tuple(parsed.get("priority_themes"))
        priority_themes = tuple(
            theme for theme in model_themes if theme in fallback.priority_themes
        )
        priority_themes = tuple(
            dict.fromkeys((*priority_themes, *fallback.priority_themes))
        )
        reasoning = _safe_reasoning(parsed.get("reasoning"))
        proposal_hash = stable_hash(
            {
                "report_id": report.report_id,
                "artifact_id": artifact_id,
                "requested_direction": requested_direction,
                "priority_themes": priority_themes,
                "theme_decisions": fallback.theme_decisions,
                "support_refs": fallback.support_refs,
                "optimizer_source": self.source,
                "optimizer_model": self.model,
                "raw": raw,
            }
        )[:24]
        return CompanyOptimizationProposal(
            proposal_id=f"proposal_{artifact_id}_{proposal_hash}",
            artifact_id=artifact_id,
            requested_direction=requested_direction,
            priority_themes=priority_themes,
            support_refs=fallback.support_refs,
            public_trace_hash=stable_hash(report),
            historical_target_overlap=None,
            optimizer_source=self.source,
            optimizer_model=self.model,
            optimizer_reasoning=reasoning,
            theme_decisions=fallback.theme_decisions,
            safety_obligation_themes=fallback.safety_obligation_themes,
        )


def build_company_optimizer(
    *,
    provider: str = "heuristic",
    model: str | None = None,
    timeout_seconds: float = 60.0,
    strict_live: bool = False,
    request_attempts: int = 2,
    retry_backoff_seconds: float = 0.0,
):
    if provider == "heuristic":
        return HeuristicCompanyOptimizer()
    if provider == "openai":
        return OpenAICompanyOptimizer(
            model=model,
            timeout_seconds=timeout_seconds,
            strict_live=strict_live,
            request_attempts=request_attempts,
            retry_backoff_seconds=retry_backoff_seconds,
        )
    raise ValueError(f"Unsupported company optimizer provider: {provider}")


def _optimizer_payload(
    *,
    report: CompanyPublicFeedbackReport,
    artifact_id: str,
    public_traces: tuple[CompanyVisibleTrace, ...],
    kernel_proposal: CompanyOptimizationProposal,
) -> dict:
    return canonicalize(
        {
            "artifact_id": artifact_id,
            "kernel_adoptable_themes": kernel_proposal.priority_themes,
            "theme_decisions": kernel_proposal.theme_decisions,
            "company_public_feedback_report": report,
            "public_trace_summaries": [
                {
                    "event_id": trace.event_id,
                    "tick": trace.tick,
                    "action_type": trace.action_type,
                    "artifact_id": trace.artifact_id,
                    "content_id": trace.content_id,
                    "channel_id": trace.channel_id,
                    "public_summary": trace.public_summary,
                    "support_refs": trace.support_refs,
                }
                for trace in public_traces[:80]
            ],
            "privacy_boundary": "public_traces_only",
        }
    )


def _parse_json_object(raw: str) -> dict:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            raise ValueError(
                f"Company optimizer returned non-JSON text: {raw[:200]!r}"
            ) from None
        value = json.loads(raw[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("Company optimizer returned JSON that is not an object")
    return value


def _response_text(response) -> str:
    raw = getattr(response, "output_text", "") or ""
    if raw.strip():
        return raw.strip()
    fragments: list[str] = []
    for item in getattr(response, "output", []) or []:
        item_content = (
            item.get("content", [])
            if isinstance(item, dict)
            else getattr(item, "content", [])
        )
        for content in item_content or []:
            if isinstance(content, dict):
                text = content.get("text") or content.get("value")
            else:
                text = getattr(content, "text", None) or getattr(content, "value", None)
            if text:
                fragments.append(text)
    return "\n".join(fragments).strip()


def _safe_label(value, default: str) -> str:
    if not isinstance(value, str):
        value = default
    cleaned = "".join(char if char.isalnum() else "_" for char in value.strip().lower())
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return cleaned[:80] or default


def _safe_theme_tuple(value) -> tuple[str, ...]:
    if not isinstance(value, list | tuple):
        return ()
    themes: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        label = _safe_label(item, "")
        if label and label not in themes:
            themes.append(label)
        if len(themes) >= 24:
            break
    return tuple(themes)


def _safe_reasoning(value) -> str:
    if not isinstance(value, str):
        return "openai_optimizer_returned_no_reasoning"
    return " ".join(value.split())[:500] or "openai_optimizer_returned_empty_reasoning"


def _supports_temperature(model: str) -> bool:
    return not model.startswith("gpt-5")


@contextmanager
def _wall_clock_timeout(timeout_seconds: float):
    if timeout_seconds <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return

    def _raise_timeout(signum: int, frame: FrameType | None) -> None:
        raise TimeoutError(f"Company optimizer call exceeded {timeout_seconds:.3f}s")

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
