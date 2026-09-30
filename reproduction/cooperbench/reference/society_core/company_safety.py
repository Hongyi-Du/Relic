"""Company-side safety and robustness review for product proposals."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from time import sleep

from .company_optimizer import (
    _parse_json_object,
    _response_text,
    _safe_label,
    _safe_reasoning,
    _supports_temperature,
    _wall_clock_timeout,
)
from .hashing import canonicalize, stable_hash
from .openai_runtime import (
    build_openai_client,
    prepare_openai_response_request,
)
from .org_adapter import (
    CompanyOptimizationProposal,
    CompanyPublicFeedbackReport,
    CompanyVisibleTrace,
)
from .product_experience import (
    _looks_like_repo_digest_artifact,
    derive_substrate_theme_lexicon,
)
from .schemas import Artifact, clamp01

DEFAULT_COMPANY_SAFETY_MODEL = "gpt-5.5-pro"
DEFAULT_COMPANY_SAFETY_MAX_OUTPUT_TOKENS = 20000
MAX_EXPANDED_EVIDENCE_THEMES = 4
MAX_EXPANDED_EVIDENCE_TRACES_PER_THEME = 18
MAX_AUGMENTED_THEMES = 24


@dataclass(frozen=True)
class SafetyObligation:
    theme: str
    risk_score: float
    severity: float
    exploitability: float
    blast_radius: float
    evidence_strength: float
    uncertainty_bonus: float
    underrepresentation_bonus: float
    support_refs: tuple[str, ...]
    expected_invariant: str
    regression_test_idea: str
    rationale: str
    enter_current_sprint: bool
    needs_more_evidence: bool = False
    evidence_gap: str = ""


@dataclass(frozen=True)
class CompanySafetyReview:
    review_id: str
    artifact_id: str
    source: str
    model: str | None
    obligations: tuple[SafetyObligation, ...]
    added_themes: tuple[str, ...]
    preserved_themes: tuple[str, ...]
    augmented_proposal_id: str
    public_trace_hash: str
    reasoning: str
    fallback_reason: str | None = None
    evidence_rounds: int = 1
    expanded_evidence_themes: tuple[str, ...] = ()
    evidence_expansion_error: str | None = None


@dataclass(frozen=True)
class SafetyThemeProfile:
    theme: str
    severity: float
    exploitability: float
    blast_radius: float
    expected_invariant: str
    regression_test_idea: str
    keywords: tuple[str, ...]


SAFETY_THEME_PROFILES: tuple[SafetyThemeProfile, ...] = (
    SafetyThemeProfile(
        theme="path_security",
        severity=0.96,
        exploitability=0.86,
        blast_radius=0.88,
        expected_invariant="Repository ingestion must not read files outside the requested root or follow unsafe traversal paths.",
        regression_test_idea="Create traversal and symlink-escape fixtures and assert out-of-root content never appears in the digest.",
        keywords=(
            "path",
            "traversal",
            "symlink",
            "escape",
            "outside",
            "root",
            "secret",
            "unsafe",
        ),
    ),
    SafetyThemeProfile(
        theme="diagnostics_and_recovery",
        severity=0.78,
        exploitability=0.35,
        blast_radius=0.70,
        expected_invariant="Failures must return actionable cause and recovery guidance without crashing the product path.",
        regression_test_idea="Force each known failure mode and assert the user-visible result includes cause, next step, and graceful fallback.",
        keywords=(
            "diagnostic",
            "error",
            "recover",
            "workaround",
            "unclear",
            "dead_end",
            "failure",
            "crash",
        ),
    ),
    SafetyThemeProfile(
        theme="http_client_portability",
        severity=0.74,
        exploitability=0.48,
        blast_radius=0.64,
        expected_invariant="Network reachability checks must work without shell-only binaries and degrade predictably on network errors.",
        regression_test_idea="Disable curl and assert HTTP status handling still distinguishes reachable, missing, and unauthorized repositories.",
        keywords=(
            "curl",
            "http",
            "network",
            "reachability",
            "binary",
            "portability",
            "offline",
        ),
    ),
    SafetyThemeProfile(
        theme="token_count_resilience",
        severity=0.72,
        exploitability=0.42,
        blast_radius=0.62,
        expected_invariant="Token counting must never prevent repository digest generation when tokenizer dependencies fail.",
        regression_test_idea="Patch tokenizer loading to raise and assert digest generation completes without a token estimate.",
        keywords=("token", "tiktoken", "encoding", "network", "offline", "estimate"),
    ),
    SafetyThemeProfile(
        theme="max_file_size_enforcement",
        severity=0.82,
        exploitability=0.62,
        blast_radius=0.76,
        expected_invariant="Files above configured limits must be excluded before content is emitted or counted.",
        regression_test_idea="Add oversized files and assert content, tree counts, and total-size accounting exclude them.",
        keywords=(
            "max_file_size",
            "large",
            "oversized",
            "resource",
            "memory",
            "token",
            "size",
        ),
    ),
    SafetyThemeProfile(
        theme="privacy_boundary",
        severity=0.94,
        exploitability=0.78,
        blast_radius=0.90,
        expected_invariant="Public feedback processing must not expose private user state, tokens, hidden files, or evaluator-only data.",
        regression_test_idea="Inject private-only markers and assert public traces, prompts, and reports never include them.",
        keywords=(
            "privacy",
            "private",
            "token",
            "secret",
            "hidden",
            "boundary",
            "credential",
        ),
    ),
)

# Substrate-neutral profiles that apply to any product; the rest of the curated
# set above encodes one substrate's (repo-digest) failure vocabulary and must
# not be applied to arbitrary products.
_GENERIC_SAFETY_THEMES: frozenset[str] = frozenset(
    {"diagnostics_and_recovery", "privacy_boundary"}
)

# Severity band observed across the curated profiles (0.72..0.96); derived
# severities map declared unreliability linearly into the same band so curated
# and derived risk scores stay commensurable.
_DERIVED_SEVERITY_FLOOR = 0.55
_DERIVED_SEVERITY_SPAN = 0.40
_DERIVED_SEVERITY_DEFAULT = 0.62
# No substrate channel declares exploitability or blast radius, so derived
# profiles carry an uninformative prior and let evidence drive the risk score.
_DERIVED_NEUTRAL_PRIOR = 0.5


def derive_substrate_safety_profiles(
    artifact: Artifact,
) -> tuple[SafetyThemeProfile, ...]:
    """Safety-theme profiles derived from the artifact's declared semantics.

    Reuses the substrate theme lexicon (reliability_profile keys + failure
    modes): each derived theme becomes a profile whose severity reflects the
    declared unreliability of that area and whose keywords are the lexicon
    tokens, so evidence matching works in the substrate's own vocabulary.
    """

    profiles: list[SafetyThemeProfile] = []
    for theme, keywords in derive_substrate_theme_lexicon(artifact):
        reliability = artifact.reliability_profile.get(theme)
        if reliability is None:
            severity = _DERIVED_SEVERITY_DEFAULT
        else:
            severity = clamp01(
                _DERIVED_SEVERITY_FLOOR
                + _DERIVED_SEVERITY_SPAN * (1.0 - clamp01(float(reliability)))
            )
        human = theme.replace("_", " ")
        profiles.append(
            SafetyThemeProfile(
                theme=theme,
                severity=round(severity, 4),
                exploitability=_DERIVED_NEUTRAL_PRIOR,
                blast_radius=_DERIVED_NEUTRAL_PRIOR,
                expected_invariant=(
                    f"Behavior covered by {human} must hold under the "
                    "failure conditions users reported."
                ),
                regression_test_idea=(
                    f"Reproduce the reported {human} failure and assert the "
                    "user-visible outcome stays correct."
                ),
                keywords=tuple(sorted(keywords)),
            )
        )
    return tuple(profiles)


def active_safety_theme_profiles(
    artifact: Artifact | None,
) -> tuple[SafetyThemeProfile, ...]:
    """Profiles to review against for this artifact.

    No artifact (legacy callers) or the curated repo-digest substrate keeps the
    full curated set; any other substrate gets the substrate-neutral profiles
    plus profiles derived from its own declared semantics.
    """

    if artifact is None or _looks_like_repo_digest_artifact(artifact):
        return SAFETY_THEME_PROFILES
    generic = tuple(
        profile
        for profile in SAFETY_THEME_PROFILES
        if profile.theme in _GENERIC_SAFETY_THEMES
    )
    return generic + derive_substrate_safety_profiles(artifact)


class HeuristicCompanySafetyReviewAgent:
    source = "heuristic_company_safety_review_v17"

    def review(
        self,
        *,
        proposal: CompanyOptimizationProposal,
        report: CompanyPublicFeedbackReport,
        target_improvement_themes: tuple[str, ...] = (),
        public_traces: tuple[CompanyVisibleTrace, ...] = (),
        artifact: Artifact | None = None,
    ) -> tuple[CompanyOptimizationProposal, CompanySafetyReview]:
        del target_improvement_themes
        obligations = _heuristic_obligations(
            proposal=proposal,
            report=report,
            public_traces=public_traces,
            profiles=active_safety_theme_profiles(artifact),
        )
        return _augment_proposal_with_review(
            proposal=proposal,
            review_source=self.source,
            review_model=None,
            obligations=obligations,
            public_evidence_themes=_public_evidence_themes(
                report=report,
                artifact_id=proposal.artifact_id,
            ),
            public_trace_hash=stable_hash((report, public_traces[:120])),
            reasoning="deterministic_risk_weighted_safety_review",
        )


class OpenAICompanySafetyReviewAgent:
    source = "openai_company_safety_review_v17"

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
            raise RuntimeError(
                "OPENAI_API_KEY is required for OpenAICompanySafetyReviewAgent"
            )
        if request_attempts < 1:
            raise ValueError("company_safety_request_attempts_must_be_positive")
        if retry_backoff_seconds < 0:
            raise ValueError("company_safety_retry_backoff_must_be_nonnegative")
        self.model = (
            model
            or os.environ.get("SOCIETY_CORE_COMPANY_SAFETY_MODEL")
            or os.environ.get("SOCIETY_CORE_OPENAI_HIGH_MODEL")
            or DEFAULT_COMPANY_SAFETY_MODEL
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
        self._fallback = HeuristicCompanySafetyReviewAgent()

    def review(
        self,
        *,
        proposal: CompanyOptimizationProposal,
        report: CompanyPublicFeedbackReport,
        target_improvement_themes: tuple[str, ...] = (),
        public_traces: tuple[CompanyVisibleTrace, ...] = (),
        artifact: Artifact | None = None,
    ) -> tuple[CompanyOptimizationProposal, CompanySafetyReview]:
        fallback_proposal, fallback_review = self._fallback.review(
            proposal=proposal,
            report=report,
            public_traces=public_traces,
            artifact=artifact,
        )
        del target_improvement_themes
        profiles = active_safety_theme_profiles(artifact)
        payload = _safety_payload(
            proposal=proposal,
            report=report,
            public_traces=public_traces,
            profiles=profiles,
        )
        try:
            obligations, reasoning = self._call_review_model_with_retries(
                payload=payload, review_stage="screening"
            )
        except (
            Exception
        ) as exc:  # pragma: no cover - live network or malformed output fallback.
            failure_kind = (
                "openai_live_error"
                if _is_openai_live_error(exc)
                else "openai_parse_error"
            )
            if self.strict_live:
                raise RuntimeError(
                    "live_company_safety_request_exhausted:"
                    f"{failure_kind}:{type(exc).__name__}:"
                    f"attempts={self.request_attempts}"
                ) from exc
            return fallback_proposal, replace(
                fallback_review,
                source=f"{self.source}:fallback_heuristic",
                model=self.model,
                fallback_reason=f"{failure_kind}:{type(exc).__name__}:{str(exc)[:160]}",
            )
        if not obligations:
            obligations = fallback_review.obligations

        expanded_themes = _themes_requiring_more_evidence(obligations)
        evidence_rounds = 1
        expansion_error = None
        if expanded_themes:
            expanded_payload = _expanded_safety_payload(
                proposal=proposal,
                report=report,
                public_traces=public_traces,
                prior_obligations=obligations,
                prior_reasoning=reasoning,
                expansion_themes=expanded_themes,
                profiles=profiles,
            )
            try:
                expanded_obligations, expanded_reasoning = (
                    self._call_review_model_with_retries(
                        payload=expanded_payload,
                        review_stage="expanded_evidence",
                    )
                )
            except (
                Exception
            ) as exc:  # pragma: no cover - live secondary review fallback.
                if self.strict_live:
                    raise RuntimeError(
                        "live_company_safety_expansion_request_exhausted:"
                        f"{type(exc).__name__}:attempts={self.request_attempts}"
                    ) from exc
                expansion_error = f"openai_expanded_evidence_error:{type(exc).__name__}:{str(exc)[:160]}"
            else:
                if expanded_obligations:
                    obligations = _merge_obligations(obligations, expanded_obligations)
                    reasoning = _join_second_pass_reasoning(
                        reasoning, expanded_reasoning
                    )
                    evidence_rounds = 2

        return _augment_proposal_with_review(
            proposal=proposal,
            review_source=self.source,
            review_model=self.model,
            obligations=obligations,
            public_evidence_themes=_public_evidence_themes(
                report=report,
                artifact_id=proposal.artifact_id,
            ),
            public_trace_hash=stable_hash((report, public_traces[:120])),
            reasoning=reasoning,
            evidence_rounds=evidence_rounds,
            expanded_evidence_themes=expanded_themes,
            evidence_expansion_error=expansion_error,
        )

    def _call_review_model_with_retries(
        self, *, payload: dict, review_stage: str
    ) -> tuple[tuple[SafetyObligation, ...], str]:
        last_error: Exception | None = None
        for attempt in range(self.request_attempts):
            try:
                return self._call_review_model(
                    payload=payload,
                    review_stage=review_stage,
                )
            except Exception as exc:
                last_error = exc
                if attempt + 1 < self.request_attempts:
                    sleep(self.retry_backoff_seconds * (2**attempt))
        assert last_error is not None
        raise last_error

    def _call_review_model(
        self, *, payload: dict, review_stage: str
    ) -> tuple[tuple[SafetyObligation, ...], str]:
        request = {
            "model": self.model,
            "input": (
                "You are the company's safety and robustness review agent.\n"
                "Use only the provided public feedback report, current product proposal, and public trace summaries.\n"
                "Focus on low-frequency but high-impact risks that product ranking may underweight.\n"
                "Return only JSON with key obligations. Each obligation must include: theme, severity, "
                "exploitability, blast_radius, evidence_strength, uncertainty_bonus, underrepresentation_bonus, "
                "support_refs, expected_invariant, regression_test_idea, rationale, enter_current_sprint, "
                "needs_more_evidence, evidence_gap.\n"
                "Set needs_more_evidence=true only when more public traces could materially change the sprint decision.\n"
                "Themes must be short snake_case strings. Do not invent private facts.\n\n"
                f"REVIEW_STAGE: {review_stage}\n"
                f"INPUT_JSON:\n{json.dumps(payload, sort_keys=True, separators=(',', ':'))}"
            ),
            "max_output_tokens": DEFAULT_COMPANY_SAFETY_MAX_OUTPUT_TOKENS,
            "text": {"format": _safety_review_text_format()},
            "timeout": self.timeout_seconds,
        }
        if _supports_temperature(self.model):
            request["temperature"] = 0
        if self.model.startswith("gpt-5.5"):
            request["reasoning"] = {"effort": "xhigh"}
        try:
            with _wall_clock_timeout(self.timeout_seconds):
                response = self._client.responses.create(
                    **prepare_openai_response_request(request)
                )
        except Exception as exc:  # pragma: no cover - live network fallback.
            raise RuntimeError(f"live:{type(exc).__name__}") from exc

        raw = _response_text(response)
        try:
            parsed = _parse_json_object(raw)
            obligations = _safe_obligation_tuple(parsed.get("obligations"))
            reasoning = _safe_reasoning(parsed.get("reasoning"))
        except Exception as exc:  # pragma: no cover - live malformed output fallback.
            diagnostic = _response_parse_diagnostic(response=response, raw=raw)
            raise ValueError(f"parse:{type(exc).__name__}:{diagnostic}") from exc
        return obligations, reasoning


class DisabledCompanySafetyReviewAgent:
    source = "disabled_company_safety_review_v17"

    def review(
        self,
        *,
        proposal: CompanyOptimizationProposal,
        report: CompanyPublicFeedbackReport,
        target_improvement_themes: tuple[str, ...] = (),
        public_traces: tuple[CompanyVisibleTrace, ...] = (),
        artifact: Artifact | None = None,
    ) -> tuple[CompanyOptimizationProposal, CompanySafetyReview]:
        del artifact
        review = CompanySafetyReview(
            review_id=f"safety_review_disabled_{stable_hash(proposal)[:16]}",
            artifact_id=proposal.artifact_id,
            source=self.source,
            model=None,
            obligations=(),
            added_themes=(),
            preserved_themes=proposal.priority_themes,
            augmented_proposal_id=proposal.proposal_id,
            public_trace_hash=stable_hash((report, public_traces[:120])),
            reasoning="safety_review_disabled",
        )
        return proposal, review


def build_company_safety_review_agent(
    *,
    provider: str = "heuristic",
    model: str | None = None,
    timeout_seconds: float = 60.0,
    strict_live: bool = False,
    request_attempts: int = 2,
    retry_backoff_seconds: float = 0.0,
):
    if provider == "off":
        return DisabledCompanySafetyReviewAgent()
    if provider == "heuristic":
        return HeuristicCompanySafetyReviewAgent()
    if provider == "openai":
        return OpenAICompanySafetyReviewAgent(
            model=model,
            timeout_seconds=timeout_seconds,
            strict_live=strict_live,
            request_attempts=request_attempts,
            retry_backoff_seconds=retry_backoff_seconds,
        )
    raise ValueError(f"Unsupported company safety provider: {provider}")


def _heuristic_obligations(
    *,
    proposal: CompanyOptimizationProposal,
    report: CompanyPublicFeedbackReport,
    public_traces: tuple[CompanyVisibleTrace, ...],
    profiles: tuple[SafetyThemeProfile, ...] = SAFETY_THEME_PROFILES,
) -> tuple[SafetyObligation, ...]:
    feedback = report.artifact_feedback.get(proposal.artifact_id)
    theme_support = (
        feedback.theme_support if feedback and feedback.theme_support else {}
    )
    obligations: list[SafetyObligation] = []
    for profile in profiles:
        evidence_strength, support_refs = _evidence_for_profile(
            profile=profile,
            theme_support=theme_support,
            public_traces=public_traces,
            artifact_id=proposal.artifact_id,
        )
        underrepresentation_bonus = _underrepresentation_bonus(
            theme=profile.theme,
            priority_themes=proposal.priority_themes,
        )
        uncertainty_bonus = clamp01(
            0.35 * (1.0 - evidence_strength)
            + 0.35 * underrepresentation_bonus
            + 0.30 * float(profile.severity >= 0.85)
        )
        risk_score = clamp01(
            profile.severity
            * (0.34 + 0.22 * profile.exploitability + 0.20 * profile.blast_radius)
            * (0.55 + 0.35 * evidence_strength + 0.10 * uncertainty_bonus)
            * (1.0 + 0.15 * underrepresentation_bonus)
        )
        sprint_threshold = 0.38 if profile.theme == "diagnostics_and_recovery" else 0.42
        enter_current_sprint = (
            risk_score >= sprint_threshold and evidence_strength >= 0.18
        )
        if risk_score < 0.30:
            continue
        obligations.append(
            SafetyObligation(
                theme=profile.theme,
                risk_score=round(risk_score, 12),
                severity=profile.severity,
                exploitability=profile.exploitability,
                blast_radius=profile.blast_radius,
                evidence_strength=round(evidence_strength, 12),
                uncertainty_bonus=round(uncertainty_bonus, 12),
                underrepresentation_bonus=underrepresentation_bonus,
                support_refs=support_refs,
                expected_invariant=profile.expected_invariant,
                regression_test_idea=profile.regression_test_idea,
                rationale=(
                    f"Risk score combines severity, exploitability, blast radius, public evidence, "
                    f"uncertainty, and proposal underrepresentation for {profile.theme}."
                ),
                enter_current_sprint=enter_current_sprint,
            )
        )
    return tuple(
        sorted(
            obligations,
            key=lambda item: (-item.enter_current_sprint, -item.risk_score, item.theme),
        )
    )


def _augment_proposal_with_review(
    *,
    proposal: CompanyOptimizationProposal,
    review_source: str,
    review_model: str | None,
    obligations: tuple[SafetyObligation, ...],
    public_evidence_themes: tuple[str, ...],
    public_trace_hash: str,
    reasoning: str,
    evidence_rounds: int = 1,
    expanded_evidence_themes: tuple[str, ...] = (),
    evidence_expansion_error: str | None = None,
) -> tuple[CompanyOptimizationProposal, CompanySafetyReview]:
    del public_evidence_themes
    sprint_obligation_themes = tuple(
        dict.fromkeys(
            obligation.theme
            for obligation in obligations
            if obligation.enter_current_sprint
        )
    )
    added_themes = tuple(
        theme
        for theme in sprint_obligation_themes
        if theme not in proposal.safety_obligation_themes
    )
    safety_obligation_themes = tuple(
        dict.fromkeys(
            (
                *proposal.safety_obligation_themes,
                *sprint_obligation_themes,
            )
        )
    )[:MAX_AUGMENTED_THEMES]
    support_refs = tuple(
        dict.fromkeys(
            (
                *proposal.support_refs,
                *(ref for obligation in obligations for ref in obligation.support_refs),
            )
        )
    )[:80]
    proposal_hash = stable_hash(
        {
            "base_proposal_id": proposal.proposal_id,
            "review_source": review_source,
            "review_model": review_model,
            "priority_themes": proposal.priority_themes,
            "safety_obligation_themes": safety_obligation_themes,
            "obligations": obligations,
        }
    )[:24]
    augmented = replace(
        proposal,
        proposal_id=f"proposal_{proposal.artifact_id}_safety_{proposal_hash}",
        requested_direction=f"{proposal.requested_direction}_with_safety_review",
        priority_themes=proposal.priority_themes,
        safety_obligation_themes=safety_obligation_themes,
        support_refs=support_refs,
        public_trace_hash=public_trace_hash,
        historical_target_overlap=None,
        optimizer_source=f"{proposal.optimizer_source}+{review_source}",
        optimizer_model=proposal.optimizer_model,
        optimizer_reasoning=_join_reasoning(proposal.optimizer_reasoning, reasoning),
    )
    review_id = f"safety_review_{stable_hash((augmented, obligations, review_source, review_model))[:24]}"
    review = CompanySafetyReview(
        review_id=review_id,
        artifact_id=proposal.artifact_id,
        source=review_source,
        model=review_model,
        obligations=obligations,
        added_themes=added_themes,
        preserved_themes=proposal.priority_themes,
        augmented_proposal_id=augmented.proposal_id,
        public_trace_hash=public_trace_hash,
        reasoning=reasoning,
        evidence_rounds=evidence_rounds,
        expanded_evidence_themes=expanded_evidence_themes,
        evidence_expansion_error=evidence_expansion_error,
    )
    return augmented, review


def _public_evidence_themes(
    *,
    report: CompanyPublicFeedbackReport,
    artifact_id: str,
) -> tuple[str, ...]:
    feedback = report.artifact_feedback.get(artifact_id)
    if feedback is None:
        return ()
    return tuple(dict.fromkeys(feedback.top_upgrade_themes))[:8]


def _safety_review_text_format() -> dict:
    obligation_properties = {
        "theme": {"type": "string"},
        "severity": {"type": "number"},
        "exploitability": {"type": "number"},
        "blast_radius": {"type": "number"},
        "evidence_strength": {"type": "number"},
        "uncertainty_bonus": {"type": "number"},
        "underrepresentation_bonus": {"type": "number"},
        "support_refs": {"type": "array", "items": {"type": "string"}},
        "expected_invariant": {"type": "string"},
        "regression_test_idea": {"type": "string"},
        "rationale": {"type": "string"},
        "enter_current_sprint": {"type": "boolean"},
        "needs_more_evidence": {"type": "boolean"},
        "evidence_gap": {"type": "string"},
    }
    return {
        "type": "json_schema",
        "name": "company_safety_review",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "obligations": {
                    "type": "array",
                    "minItems": 0,
                    "maxItems": 8,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": obligation_properties,
                        "required": list(obligation_properties),
                    },
                },
                "reasoning": {"type": "string"},
            },
            "required": ["obligations", "reasoning"],
        },
    }


def _evidence_for_profile(
    *,
    profile: SafetyThemeProfile,
    theme_support: dict[str, dict[str, float]],
    public_traces: tuple[CompanyVisibleTrace, ...],
    artifact_id: str,
) -> tuple[float, tuple[str, ...]]:
    support = theme_support.get(profile.theme, {})
    count = float(support.get("count", 0.0) or 0.0)
    mean_priority = clamp01(float(support.get("mean_priority", 0.0) or 0.0))
    mean_confidence = clamp01(float(support.get("mean_confidence", 0.0) or 0.0))
    support_score = clamp01(
        0.38 * min(1.0, count / 50.0) + 0.36 * mean_priority + 0.26 * mean_confidence
    )
    keyword_refs: list[str] = []
    for trace in public_traces:
        if trace.artifact_id != artifact_id:
            continue
        haystack = f"{trace.public_summary} {trace.upgrade_theme or ''}".lower()
        if any(keyword in haystack for keyword in profile.keywords):
            keyword_refs.append(trace.event_id)
        if len(keyword_refs) >= 12:
            break
    keyword_score = clamp01(len(keyword_refs) / 12.0)
    evidence_strength = max(support_score, keyword_score)
    support_refs = tuple(keyword_refs[:8])
    if not support_refs:
        support_refs = tuple(
            trace.event_id
            for trace in public_traces
            if trace.artifact_id == artifact_id and trace.upgrade_theme == profile.theme
        )[:8]
    return evidence_strength, support_refs


def _underrepresentation_bonus(
    *, theme: str, priority_themes: tuple[str, ...]
) -> float:
    if theme not in priority_themes:
        return 1.0
    position = priority_themes.index(theme)
    if position >= 6:
        return 0.55
    return 0.0


def _safe_obligation_tuple(value) -> tuple[SafetyObligation, ...]:
    if not isinstance(value, list | tuple):
        return ()
    obligations: list[SafetyObligation] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        theme = _safe_label(item.get("theme"), "")
        if not theme:
            continue
        obligation = SafetyObligation(
            theme=theme,
            risk_score=_risk_score_from_item(item),
            severity=_safe_float01(item.get("severity")),
            exploitability=_safe_float01(item.get("exploitability")),
            blast_radius=_safe_float01(item.get("blast_radius")),
            evidence_strength=_safe_float01(item.get("evidence_strength")),
            uncertainty_bonus=_safe_float01(item.get("uncertainty_bonus")),
            underrepresentation_bonus=_safe_float01(
                item.get("underrepresentation_bonus")
            ),
            support_refs=_safe_string_tuple(item.get("support_refs"))[:12],
            expected_invariant=_safe_text(
                item.get("expected_invariant"), "safety invariant required"
            ),
            regression_test_idea=_safe_text(
                item.get("regression_test_idea"), "add a regression test for this risk"
            ),
            rationale=_safe_text(
                item.get("rationale"), "risk identified from public evidence"
            ),
            enter_current_sprint=bool(item.get("enter_current_sprint")),
            needs_more_evidence=bool(item.get("needs_more_evidence")),
            evidence_gap=_safe_text(item.get("evidence_gap"), ""),
        )
        obligations.append(obligation)
    return tuple(
        sorted(
            obligations,
            key=lambda item: (-item.enter_current_sprint, -item.risk_score, item.theme),
        )
    )[:8]


def _themes_requiring_more_evidence(
    obligations: tuple[SafetyObligation, ...],
) -> tuple[str, ...]:
    requested = [
        obligation
        for obligation in obligations
        if obligation.needs_more_evidence and obligation.theme
    ]
    requested.sort(key=lambda item: (-item.risk_score, -item.severity, item.theme))
    return tuple(dict.fromkeys(obligation.theme for obligation in requested))[
        :MAX_EXPANDED_EVIDENCE_THEMES
    ]


def _merge_obligations(
    first_pass: tuple[SafetyObligation, ...],
    expanded_pass: tuple[SafetyObligation, ...],
) -> tuple[SafetyObligation, ...]:
    merged = {obligation.theme: obligation for obligation in first_pass}
    for obligation in expanded_pass:
        merged[obligation.theme] = obligation
    return tuple(
        sorted(
            merged.values(),
            key=lambda item: (-item.enter_current_sprint, -item.risk_score, item.theme),
        )
    )[:8]


def _risk_score_from_item(item: dict) -> float:
    severity = _safe_float01(item.get("severity"))
    exploitability = _safe_float01(item.get("exploitability"))
    blast_radius = _safe_float01(item.get("blast_radius"))
    evidence_strength = _safe_float01(item.get("evidence_strength"))
    uncertainty_bonus = _safe_float01(item.get("uncertainty_bonus"))
    underrepresentation_bonus = _safe_float01(item.get("underrepresentation_bonus"))
    return round(
        clamp01(
            severity
            * (0.34 + 0.22 * exploitability + 0.20 * blast_radius)
            * (0.55 + 0.35 * evidence_strength + 0.10 * uncertainty_bonus)
            * (1.0 + 0.15 * underrepresentation_bonus)
        ),
        12,
    )


def _safety_payload(
    *,
    proposal: CompanyOptimizationProposal,
    report: CompanyPublicFeedbackReport,
    public_traces: tuple[CompanyVisibleTrace, ...],
    profiles: tuple[SafetyThemeProfile, ...] = SAFETY_THEME_PROFILES,
) -> dict:
    feedback = report.artifact_feedback.get(proposal.artifact_id)
    return canonicalize(
        {
            "proposal": {
                "artifact_id": proposal.artifact_id,
                "requested_direction": proposal.requested_direction,
                "priority_themes": proposal.priority_themes,
                "support_ref_count": len(proposal.support_refs),
            },
            "company_public_feedback_summary": {
                "report_id": report.report_id,
                "since_tick": report.since_tick,
                "trace_count": report.trace_count,
                "artifact_feedback": _compact_artifact_feedback(feedback),
            },
            "safety_theme_profiles": [
                {
                    "theme": profile.theme,
                    "severity": profile.severity,
                    "exploitability": profile.exploitability,
                    "blast_radius": profile.blast_radius,
                    "expected_invariant": profile.expected_invariant,
                    "regression_test_idea": profile.regression_test_idea,
                    "keywords": profile.keywords,
                }
                for profile in profiles
            ],
            "public_trace_evidence": _compact_safety_trace_evidence(
                public_traces=public_traces,
                artifact_id=proposal.artifact_id,
                profiles=profiles,
            ),
            "risk_formula": (
                "risk_score=severity*(0.34+0.22*exploitability+0.20*blast_radius)"
                "*(0.55+0.35*evidence_strength+0.10*uncertainty_bonus)"
                "*(1+0.15*underrepresentation_bonus)"
            ),
            "privacy_boundary": "public_traces_and_public_feedback_only",
        }
    )


def _expanded_safety_payload(
    *,
    proposal: CompanyOptimizationProposal,
    report: CompanyPublicFeedbackReport,
    public_traces: tuple[CompanyVisibleTrace, ...],
    prior_obligations: tuple[SafetyObligation, ...],
    prior_reasoning: str,
    expansion_themes: tuple[str, ...],
    profiles: tuple[SafetyThemeProfile, ...] = SAFETY_THEME_PROFILES,
) -> dict:
    feedback = report.artifact_feedback.get(proposal.artifact_id)
    return canonicalize(
        {
            "review_stage": "expanded_evidence",
            "instruction": (
                "Reconsider only the themes listed in expansion_themes using the larger public trace window. "
                "Return final obligations for those themes plus any prior obligations that remain release gates."
            ),
            "proposal": {
                "artifact_id": proposal.artifact_id,
                "requested_direction": proposal.requested_direction,
                "priority_themes": proposal.priority_themes,
                "support_ref_count": len(proposal.support_refs),
            },
            "company_public_feedback_summary": {
                "report_id": report.report_id,
                "since_tick": report.since_tick,
                "trace_count": report.trace_count,
                "artifact_feedback": _compact_artifact_feedback(feedback),
            },
            "expansion_themes": expansion_themes,
            "prior_obligations": [
                _compact_obligation(obligation) for obligation in prior_obligations
            ],
            "prior_reasoning": prior_reasoning[:700],
            "expanded_public_trace_evidence": _expanded_safety_trace_evidence(
                public_traces=public_traces,
                artifact_id=proposal.artifact_id,
                themes=expansion_themes,
                profiles=profiles,
            ),
            "risk_formula": (
                "risk_score=severity*(0.34+0.22*exploitability+0.20*blast_radius)"
                "*(0.55+0.35*evidence_strength+0.10*uncertainty_bonus)"
                "*(1+0.15*underrepresentation_bonus)"
            ),
            "privacy_boundary": "public_traces_and_public_feedback_only",
        }
    )


def _compact_obligation(obligation: SafetyObligation) -> dict:
    return {
        "theme": obligation.theme,
        "risk_score": obligation.risk_score,
        "severity": obligation.severity,
        "evidence_strength": obligation.evidence_strength,
        "support_refs": obligation.support_refs[:8],
        "expected_invariant": obligation.expected_invariant,
        "regression_test_idea": obligation.regression_test_idea,
        "enter_current_sprint": obligation.enter_current_sprint,
        "needs_more_evidence": obligation.needs_more_evidence,
        "evidence_gap": obligation.evidence_gap,
    }


def _expanded_safety_trace_evidence(
    *,
    public_traces: tuple[CompanyVisibleTrace, ...],
    artifact_id: str,
    themes: tuple[str, ...],
    profiles: tuple[SafetyThemeProfile, ...] = SAFETY_THEME_PROFILES,
) -> tuple[dict, ...]:
    evidence: list[dict] = []
    for theme in themes:
        profile = _profile_for_theme(theme, profiles=profiles)
        matched_traces = [
            trace
            for trace in public_traces
            if trace.artifact_id == artifact_id
            and _trace_matches_theme(trace=trace, theme=theme, profile=profile)
        ]
        sampled = matched_traces[:MAX_EXPANDED_EVIDENCE_TRACES_PER_THEME]
        evidence.append(
            {
                "theme": theme,
                "total_matching_public_traces": len(matched_traces),
                "sampled_public_traces": [
                    {
                        "event_id": trace.event_id,
                        "tick": trace.tick,
                        "action_type": trace.action_type,
                        "channel_id": trace.channel_id,
                        "public_summary": trace.public_summary[:320],
                        "upgrade_theme": trace.upgrade_theme,
                        "upgrade_priority": trace.upgrade_priority,
                        "upgrade_confidence": trace.upgrade_confidence,
                    }
                    for trace in sampled
                ],
            }
        )
    return tuple(evidence)


def _compact_artifact_feedback(feedback) -> dict:
    if feedback is None:
        return {}
    return {
        "artifact_id": feedback.artifact_id,
        "public_mentions": feedback.public_mentions,
        "trial_events": feedback.trial_events,
        "paid_events": feedback.paid_events,
        "abandonment_events": feedback.abandonment_events,
        "failure_reports": feedback.failure_reports,
        "tutorial_events": feedback.tutorial_events,
        "requested_direction": feedback.requested_direction,
        "upgrade_suggestion_events": feedback.upgrade_suggestion_events,
        "top_upgrade_themes": feedback.top_upgrade_themes[:16],
        "theme_support": feedback.theme_support,
        "feedback_uncertainty": feedback.feedback_uncertainty,
        "support_ref_count": len(feedback.support_refs),
        "support_refs_sample": feedback.support_refs[:20],
    }


def _compact_safety_trace_evidence(
    *,
    public_traces: tuple[CompanyVisibleTrace, ...],
    artifact_id: str,
    profiles: tuple[SafetyThemeProfile, ...] = SAFETY_THEME_PROFILES,
) -> tuple[dict, ...]:
    evidence: list[dict] = []
    seen: set[str] = set()
    for profile in profiles:
        matched = 0
        for trace in public_traces:
            if trace.artifact_id != artifact_id or trace.event_id in seen:
                continue
            if not _trace_matches_theme(
                trace=trace, theme=profile.theme, profile=profile
            ):
                continue
            evidence.append(
                {
                    "event_id": trace.event_id,
                    "tick": trace.tick,
                    "action_type": trace.action_type,
                    "channel_id": trace.channel_id,
                    "public_summary": trace.public_summary[:260],
                    "upgrade_theme": trace.upgrade_theme,
                    "upgrade_priority": trace.upgrade_priority,
                    "upgrade_confidence": trace.upgrade_confidence,
                    "matched_safety_theme": profile.theme,
                }
            )
            seen.add(trace.event_id)
            matched += 1
            if matched >= 4:
                break
    if len(evidence) < 12:
        for trace in public_traces:
            if trace.artifact_id != artifact_id or trace.event_id in seen:
                continue
            evidence.append(
                {
                    "event_id": trace.event_id,
                    "tick": trace.tick,
                    "action_type": trace.action_type,
                    "channel_id": trace.channel_id,
                    "public_summary": trace.public_summary[:220],
                    "upgrade_theme": trace.upgrade_theme,
                    "upgrade_priority": trace.upgrade_priority,
                    "upgrade_confidence": trace.upgrade_confidence,
                    "matched_safety_theme": None,
                }
            )
            seen.add(trace.event_id)
            if len(evidence) >= 12:
                break
    return tuple(evidence[:36])


def _profile_for_theme(
    theme: str,
    *,
    profiles: tuple[SafetyThemeProfile, ...] = SAFETY_THEME_PROFILES,
) -> SafetyThemeProfile | None:
    for profile in profiles:
        if profile.theme == theme:
            return profile
    return None


def _trace_matches_theme(
    *,
    trace: CompanyVisibleTrace,
    theme: str,
    profile: SafetyThemeProfile | None,
) -> bool:
    if trace.upgrade_theme == theme:
        return True
    if profile is None:
        return False
    haystack = f"{trace.public_summary} {trace.upgrade_theme or ''}".lower()
    return any(keyword in haystack for keyword in profile.keywords)


def _safe_string_tuple(value) -> tuple[str, ...]:
    if not isinstance(value, list | tuple):
        return ()
    return tuple(str(item)[:160] for item in value if isinstance(item, str) and item)


def _safe_float01(value) -> float:
    try:
        return clamp01(float(value or 0.0))
    except (TypeError, ValueError):
        return 0.0


def _safe_text(value, default: str) -> str:
    if not isinstance(value, str):
        return default
    return " ".join(value.split())[:500] or default


def _join_reasoning(left: str | None, right: str) -> str:
    if left:
        return f"{left}; safety_review:{right}"[:800]
    return f"safety_review:{right}"[:800]


def _join_second_pass_reasoning(first: str, second: str) -> str:
    return f"{first}; expanded_evidence_review:{second}"[:900]


def _is_openai_live_error(exc: Exception) -> bool:
    return str(exc).startswith("live:")


def _response_parse_diagnostic(*, response, raw: str) -> str:
    incomplete = getattr(response, "incomplete_details", None)
    incomplete_reason = (
        getattr(incomplete, "reason", None) if incomplete is not None else None
    )
    response_error = getattr(response, "error", None)
    error_type = (
        getattr(response_error, "type", None) if response_error is not None else None
    )
    output = getattr(response, "output", []) or []
    return (
        f"raw_len={len(raw)};"
        f"output_items={len(output)};"
        f"incomplete_reason={incomplete_reason or 'none'};"
        f"error_type={error_type or 'none'}"
    )[:240]
