"""Frozen, company-wide resource controls for matched OrgEnv experiments.

This ledger is deliberately separate from ``BudgetSystem``.  The latter is part
of the simulated organization and can affect agent decisions; this module is an
evaluator-owned control that gives B0--B3 the same total resource ceiling,
independent of team size.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from contextlib import nullcontext
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Dict, Mapping, Optional

from environments.org_env.llm.client import LLMError, OrgLLMClient

LLM_CALLS = "llm_calls"
LLM_REQUESTED_TOKENS = "llm_requested_tokens"
LLM_PROMPT_CHARACTERS = "llm_prompt_characters"
PRIMARY_ACTIONS = "primary_actions"
TICKS = "ticks"

RESOURCE_NAMES = (
    LLM_CALLS,
    LLM_REQUESTED_TOKENS,
    LLM_PROMPT_CHARACTERS,
    PRIMARY_ACTIONS,
    TICKS,
)

_FIELD_BY_RESOURCE = {
    LLM_CALLS: "max_llm_calls",
    LLM_REQUESTED_TOKENS: "max_llm_requested_tokens",
    LLM_PROMPT_CHARACTERS: "max_llm_prompt_characters",
    PRIMARY_ACTIONS: "max_primary_actions",
    TICKS: "max_ticks",
}

_ENV_BY_FIELD = {
    "max_llm_calls": "ORG_EXPERIMENT_MAX_LLM_CALLS",
    "max_llm_requested_tokens": "ORG_EXPERIMENT_MAX_LLM_REQUESTED_TOKENS",
    "max_llm_prompt_characters": "ORG_EXPERIMENT_MAX_LLM_PROMPT_CHARACTERS",
    "max_primary_actions": "ORG_EXPERIMENT_MAX_PRIMARY_ACTIONS",
    "max_ticks": "ORG_EXPERIMENT_MAX_TICKS",
}


class ExperimentResourceExhausted(RuntimeError):
    """Raised when a hard evaluator-owned resource ceiling is exhausted."""


def _optional_non_negative_int(value: Any, *, field_name: str) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer or null") from exc
    if result < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return result


@dataclass(frozen=True)
class FrozenResourceBudget:
    """Immutable total resource ceiling shared by every condition in a block."""

    SCHEMA_VERSION: ClassVar[str] = "orgenv_frozen_resources_v1"

    max_llm_calls: Optional[int] = None
    max_llm_requested_tokens: Optional[int] = None
    max_llm_prompt_characters: Optional[int] = None
    max_primary_actions: Optional[int] = None
    max_ticks: Optional[int] = None

    def __post_init__(self) -> None:
        for field_name in _ENV_BY_FIELD:
            _optional_non_negative_int(getattr(self, field_name), field_name=field_name)

    @property
    def enabled(self) -> bool:
        return any(getattr(self, field_name) is not None for field_name in _ENV_BY_FIELD)

    def limit_for(self, resource: str) -> Optional[int]:
        if resource not in _FIELD_BY_RESOURCE:
            return None
        return getattr(self, _FIELD_BY_RESOURCE[resource])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            **{field_name: getattr(self, field_name) for field_name in _ENV_BY_FIELD},
        }

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            self.to_dict(),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "FrozenResourceBudget":
        return cls(
            **{
                field_name: _optional_non_negative_int(
                    values.get(field_name), field_name=field_name
                )
                for field_name in _ENV_BY_FIELD
            }
        )


@dataclass(frozen=True)
class ResourceUsageEvent:
    sequence: int
    resource: str
    amount: int
    accepted: bool
    cumulative_before: int
    cumulative_after: int
    limit: Optional[int]
    tick: Optional[int] = None
    agent_id: Optional[str] = None
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sequence": self.sequence,
            "resource": self.resource,
            "amount": self.amount,
            "accepted": self.accepted,
            "cumulative_before": self.cumulative_before,
            "cumulative_after": self.cumulative_after,
            "limit": self.limit,
            "tick": self.tick,
            "agent_id": self.agent_id,
            "detail": self.detail,
        }


class ExperimentResourceLedger:
    """Append-only usage ledger with atomic multi-resource reservations."""

    def __init__(self, budget: FrozenResourceBudget) -> None:
        if not budget.enabled:
            raise ValueError("resource ledger requires at least one frozen limit")
        self.budget = budget
        self.consumed: Dict[str, int] = {name: 0 for name in RESOURCE_NAMES}
        self.denied: Dict[str, int] = {name: 0 for name in RESOURCE_NAMES}
        self.events: list[ResourceUsageEvent] = []

    def reserve(
        self,
        requests: Mapping[str, int],
        *,
        tick: Optional[int] = None,
        agent_id: Optional[str] = None,
        detail: str = "",
    ) -> bool:
        normalized: Dict[str, int] = {}
        for resource, amount in requests.items():
            if resource not in RESOURCE_NAMES:
                raise ValueError(f"unknown experiment resource: {resource}")
            value = int(amount)
            if value < 0:
                raise ValueError("resource reservation amount must be non-negative")
            if value:
                normalized[resource] = value
        if not normalized:
            return True

        accepted = all(
            self.budget.limit_for(resource) is None
            or self.consumed[resource] + amount <= int(self.budget.limit_for(resource))
            for resource, amount in normalized.items()
        )
        for resource, amount in normalized.items():
            before = self.consumed[resource]
            limit = self.budget.limit_for(resource)
            after = before + amount if accepted else before
            if accepted:
                self.consumed[resource] = after
            else:
                self.denied[resource] += 1
            self.events.append(
                ResourceUsageEvent(
                    sequence=len(self.events) + 1,
                    resource=resource,
                    amount=amount,
                    accepted=accepted,
                    cumulative_before=before,
                    cumulative_after=after,
                    limit=limit,
                    tick=tick,
                    agent_id=agent_id,
                    detail=detail,
                )
            )
        return accepted

    def remaining(self, resource: str) -> Optional[int]:
        limit = self.budget.limit_for(resource)
        if limit is None:
            return None
        return max(0, limit - self.consumed.get(resource, 0))

    def snapshot(self, *, include_events: bool = False) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "budget": self.budget.to_dict(),
            "budget_fingerprint": self.budget.fingerprint,
            "consumed": dict(self.consumed),
            "denied": dict(self.denied),
            "remaining": {name: self.remaining(name) for name in RESOURCE_NAMES},
        }
        if include_events:
            result["events"] = [event.to_dict() for event in self.events]
        return result


class PromptVisibilityAuditor:
    """Fail closed when evaluator-private literals reach a provider prompt."""

    SCHEMA_VERSION = "orgenv_prompt_visibility_audit_v1"

    def __init__(
        self,
        *,
        always_forbidden: tuple[tuple[str, str], ...],
        pre_reveal_forbidden: tuple[tuple[str, str], ...],
        heldout_revealed: Callable[[], bool],
    ) -> None:
        self._always_forbidden = always_forbidden
        self._pre_reveal_forbidden = pre_reveal_forbidden
        self._heldout_revealed = heldout_revealed
        self.calls_audited = 0
        self.characters_audited = 0
        self.violation_count = 0
        self.violation_codes: set[str] = set()
        self.audit_chain_hash = "0" * 64

    def audit(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        surface: str,
    ) -> None:
        combined = f"{system_prompt}\n{user_prompt}"
        folded = combined.casefold()
        candidates = list(self._always_forbidden)
        if not self._heldout_revealed():
            candidates.extend(self._pre_reveal_forbidden)
        hits = tuple(
            sorted(
                code
                for code, literal in candidates
                if literal.casefold() in folded
            )
        )
        self.calls_audited += 1
        self.characters_audited += len(system_prompt) + len(user_prompt)
        if hits:
            self.violation_count += 1
            self.violation_codes.update(hits)
        event = {
            "sequence": self.calls_audited,
            "surface": surface,
            "prompt_sha256": hashlib.sha256(
                combined.encode("utf-8")
            ).hexdigest(),
            "prompt_characters": len(system_prompt) + len(user_prompt),
            "heldout_revealed": bool(self._heldout_revealed()),
            "violation_codes": hits,
        }
        self.audit_chain_hash = hashlib.sha256(
            (
                self.audit_chain_hash
                + json.dumps(
                    event,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            ).encode("utf-8")
        ).hexdigest()
        if hits:
            raise LLMError(
                "prompt_visibility_violation:" + ",".join(hits)
            )

    def snapshot(self) -> Dict[str, Any]:
        payload = {
            "schema_version": self.SCHEMA_VERSION,
            "calls_audited": self.calls_audited,
            "characters_audited": self.characters_audited,
            "violation_count": self.violation_count,
            "violation_codes": sorted(self.violation_codes),
            "clear": self.violation_count == 0,
            "audit_chain_hash": self.audit_chain_hash,
        }
        return {
            **payload,
            "audit_hash": hashlib.sha256(
                json.dumps(
                    payload,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
        }

    def restore(self, snapshot: Mapping[str, Any]) -> None:
        payload = dict(snapshot)
        declared = str(payload.pop("audit_hash", "") or "")
        observed = hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        if (
            payload.get("schema_version") != self.SCHEMA_VERSION
            or declared != observed
            or bool(payload.get("clear"))
            != (int(payload.get("violation_count", 0) or 0) == 0)
        ):
            raise ValueError("prompt_visibility_audit_state_invalid")
        self.calls_audited = max(
            0, int(payload.get("calls_audited", 0) or 0)
        )
        self.characters_audited = max(
            0, int(payload.get("characters_audited", 0) or 0)
        )
        self.violation_count = max(
            0, int(payload.get("violation_count", 0) or 0)
        )
        self.violation_codes = {
            str(value)
            for value in (payload.get("violation_codes") or ())
            if str(value)
        }
        chain_hash = str(payload.get("audit_chain_hash") or "")
        if len(chain_hash) != 64:
            raise ValueError("prompt_visibility_audit_chain_invalid")
        self.audit_chain_hash = chain_hash


def resolve_frozen_resource_budget(
    config: Optional[Mapping[str, Any]] = None,
    *,
    environ: Optional[Mapping[str, str]] = None,
) -> Optional[FrozenResourceBudget]:
    """Resolve an explicit scenario config, otherwise the standard env surface."""

    if config is not None:
        budget = FrozenResourceBudget.from_mapping(config)
        return budget if budget.enabled else None
    env = environ if environ is not None else os.environ
    values = {
        field_name: env.get(env_name)
        for field_name, env_name in _ENV_BY_FIELD.items()
    }
    budget = FrozenResourceBudget.from_mapping(values)
    return budget if budget.enabled else None


def initialize_world_resource_control(
    world: Any,
    config: Optional[Mapping[str, Any]] = None,
) -> None:
    budget = resolve_frozen_resource_budget(config)
    world.experiment_resource_budget = budget
    world.experiment_resource_ledger = (
        ExperimentResourceLedger(budget) if budget is not None else None
    )


def reserve_world_resources(
    world: Any,
    requests: Mapping[str, int],
    *,
    agent_id: Optional[str] = None,
    detail: str = "",
) -> bool:
    ledger = getattr(world, "experiment_resource_ledger", None)
    if ledger is None:
        return True
    return ledger.reserve(
        requests,
        tick=int(getattr(world, "world_tick", 0)),
        agent_id=agent_id,
        detail=detail,
    )


def experiment_resource_snapshot(
    world: Any, *, include_events: bool = False
) -> Optional[Dict[str, Any]]:
    ledger = getattr(world, "experiment_resource_ledger", None)
    return ledger.snapshot(include_events=include_events) if ledger is not None else None


class MeteredOrgLLMClient(OrgLLMClient):
    """Transparent LLM wrapper enforcing the company-wide frozen call budget.

    Requested-token usage is the effective output ceiling admitted for each
    provider attempt that actually starts.  It is intentionally neither actual
    completion usage (which cannot fail closed before a request) nor a prepaid
    envelope for retries that may never happen.
    """

    def __init__(
        self,
        inner: OrgLLMClient,
        ledger: ExperimentResourceLedger,
        *,
        context_provider: Optional[Callable[[], Mapping[str, Any]]] = None,
        prompt_auditor: PromptVisibilityAuditor | None = None,
    ) -> None:
        super().__init__()
        self.inner = inner
        self.ledger = ledger
        self.context_provider = context_provider
        self.prompt_auditor = prompt_auditor
        self.provider = inner.provider
        self.resource_denials = 0
        self.prompt_visibility_denials = 0
        self._denial_lock = threading.Lock()

    def _record_resource_denial(self) -> None:
        with self._denial_lock:
            self.resource_denials += 1

    def _record_prompt_visibility_denial(self) -> None:
        with self._denial_lock:
            self.prompt_visibility_denials += 1

    @property
    def calls(self) -> int:
        # Attempt-aware clients enter ``generate_*`` before admission, so their
        # own logical-call counter already includes a resource denial.  Legacy
        # clients are admitted before entry and still need the wrapper count.
        outside_inner_denials = (
            0
            if bool(
                getattr(
                    getattr(self, "inner", None),
                    "meters_actual_provider_attempts",
                    False,
                )
            )
            else self.resource_denials
        )
        return (
            int(getattr(self.inner, "calls", 0))
            + outside_inner_denials
            + self.prompt_visibility_denials
        )

    @calls.setter
    def calls(self, value: int) -> None:
        # OrgLLMClient.__init__ writes this before ``inner`` exists.
        self.__dict__["_initial_calls"] = int(value)

    @property
    def failures(self) -> int:
        outside_inner_denials = (
            0
            if bool(
                getattr(
                    getattr(self, "inner", None),
                    "meters_actual_provider_attempts",
                    False,
                )
            )
            else self.resource_denials
        )
        return (
            int(getattr(self.inner, "failures", 0))
            + outside_inner_denials
            + self.prompt_visibility_denials
        )

    @failures.setter
    def failures(self, value: int) -> None:
        self.__dict__["_initial_failures"] = int(value)

    @property
    def retries(self) -> int:
        return int(getattr(self.inner, "retries", 0))

    @retries.setter
    def retries(self, value: int) -> None:
        self.__dict__["_initial_retries"] = int(value)

    @property
    def provider_attempts(self) -> int:
        return int(getattr(self.inner, "provider_attempts", 0))

    @provider_attempts.setter
    def provider_attempts(self, value: int) -> None:
        self.__dict__["_initial_provider_attempts"] = int(value)

    @property
    def usage_totals(self) -> Dict[str, int]:
        return dict(getattr(self.inner, "usage_totals", {}))

    @usage_totals.setter
    def usage_totals(self, value: Mapping[str, int]) -> None:
        self.__dict__["_initial_usage_totals"] = dict(value)

    @property
    def response_model_counts(self) -> Dict[str, int]:
        return dict(getattr(self.inner, "response_model_counts", {}))

    @response_model_counts.setter
    def response_model_counts(self, value: Mapping[str, int]) -> None:
        self.__dict__["_initial_response_model_counts"] = dict(value)

    @property
    def response_id_digest(self) -> str:
        return str(getattr(self.inner, "response_id_digest", "") or "")

    @response_id_digest.setter
    def response_id_digest(self, value: str) -> None:
        self.__dict__["_initial_response_id_digest"] = str(value or "")

    def _reservation(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        *,
        surface: str,
    ) -> Any:
        if self.prompt_auditor is not None:
            try:
                self.prompt_auditor.audit(
                    system_prompt,
                    user_prompt,
                    surface=surface,
                )
            except LLMError:
                self._record_prompt_visibility_denial()
                raise
        context = dict(self.context_provider() if self.context_provider else {})
        envelope = self.inner.request_resource_envelope(max_tokens)
        attempts = max(1, int(envelope["provider_attempts"]))
        per_attempt_tokens = max(
            0,
            int(envelope["output_tokens_per_attempt"]),
        )
        prompt_characters = len(system_prompt) + len(user_prompt)

        if bool(getattr(self.inner, "meters_actual_provider_attempts", False)):
            from environments.org_env.llm.client import (
                provider_attempt_reservation,
            )

            def reserve_attempt(attempt_number: int) -> None:
                requests = {
                    LLM_CALLS: 1,
                    LLM_REQUESTED_TOKENS: per_attempt_tokens,
                }
                if int(attempt_number) == 1:
                    requests[LLM_PROMPT_CHARACTERS] = prompt_characters
                accepted = self.ledger.reserve(
                    requests,
                    tick=context.get("tick"),
                    agent_id=context.get("agent_id"),
                    detail=(
                        f"{surface}:actual_provider_attempt="
                        f"{int(attempt_number)}:"
                        f"output_tokens_per_attempt={per_attempt_tokens}"
                    ),
                )
                if not accepted:
                    self._record_resource_denial()
                    raise LLMError("experiment_resource_exhausted:llm")

            return provider_attempt_reservation(reserve_attempt)

        # Mock, generic HTTP, and legacy/custom clients have no internal retry
        # loop exposed to the meter.  Preserve their prior fail-closed envelope
        # admission; the built-in mock and generic clients both declare one
        # provider attempt.
        accepted = self.ledger.reserve(
            {
                LLM_CALLS: attempts,
                LLM_REQUESTED_TOKENS: attempts * per_attempt_tokens,
                LLM_PROMPT_CHARACTERS: prompt_characters,
            },
            tick=context.get("tick"),
            agent_id=context.get("agent_id"),
            detail=(
                f"{surface}:reserved_attempt_envelope={attempts}:"
                f"output_tokens_per_attempt={per_attempt_tokens}"
            ),
        )
        if not accepted:
            self._record_resource_denial()
            raise LLMError("experiment_resource_exhausted:llm")
        return nullcontext()

    def generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.3,
        max_tokens: int = 800,
    ) -> str:
        with self._reservation(
            system_prompt,
            user_prompt,
            max_tokens,
            surface="generate_text",
        ):
            return self.inner.generate_text(
                system_prompt,
                user_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: Dict[str, Any],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1200,
    ) -> Dict[str, Any]:
        with self._reservation(
            system_prompt,
            user_prompt,
            max_tokens,
            surface="generate_json",
        ):
            return self.inner.generate_json(
                system_prompt,
                user_prompt,
                schema,
                temperature=temperature,
                max_tokens=max_tokens,
            )

    def stats(self) -> Dict[str, Any]:
        result = {
            **self.inner.stats(),
            "calls": self.calls,
            "failures": self.failures,
            "resource_denials": self.resource_denials,
            "prompt_visibility_denials": self.prompt_visibility_denials,
            "experiment_resources": self.ledger.snapshot(include_events=False),
        }
        if self.prompt_auditor is not None:
            result["prompt_visibility_audit"] = (
                self.prompt_auditor.snapshot()
            )
        return result

    def prompt_visibility_audit(self) -> Dict[str, Any] | None:
        return (
            self.prompt_auditor.snapshot()
            if self.prompt_auditor is not None
            else None
        )

    def restore_prompt_visibility_audit(
        self,
        snapshot: Mapping[str, Any],
    ) -> None:
        if self.prompt_auditor is None:
            raise ValueError("prompt_visibility_auditor_missing")
        self.prompt_auditor.restore(snapshot)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


def attach_metered_llm_client(world: Any, client: Any) -> Any:
    ledger = getattr(world, "experiment_resource_ledger", None)
    if client is None or ledger is None:
        return client
    if isinstance(client, MeteredOrgLLMClient):
        return client

    def context() -> Mapping[str, Any]:
        return {
            "tick": int(getattr(world, "world_tick", 0)),
            "agent_id": getattr(world, "_experiment_resource_actor", None),
        }

    return MeteredOrgLLMClient(
        client,
        ledger,
        context_provider=context,
        prompt_auditor=_prompt_visibility_auditor(world),
    )


def _prompt_visibility_auditor(world: Any) -> PromptVisibilityAuditor | None:
    from environments.org_env.product.substrates.eval_assets import (
        oss_eval_assets,
    )

    assets = oss_eval_assets(world)
    if not assets:
        return None
    always: dict[str, tuple[str, str]] = {}
    pre_reveal: dict[str, tuple[str, str]] = {}

    def add(
        target: dict[str, tuple[str, str]],
        category: str,
        value: Any,
        *,
        minimum_length: int = 8,
    ) -> None:
        literal = str(value or "").strip()
        if len(literal) < minimum_length:
            return
        normalized = literal.casefold()
        code = (
            f"{category}:"
            + hashlib.sha256(literal.encode("utf-8")).hexdigest()[:16]
        )
        target.setdefault(normalized, (code, literal))

    for key in ("reference_repo_dir", "hidden_tests_dir"):
        add(always, f"private_path.{key}", assets.get(key))
    for spec in assets.get("hidden_test_specs") or ():
        add(always, "hidden_test.id", getattr(spec, "test_id", ""))
        add(always, "hidden_test.name", getattr(spec, "name", ""), minimum_length=12)
        add(
            always,
            "hidden_test.behavior",
            getattr(spec, "expected_behavior", ""),
            minimum_length=24,
        )
        add(always, "hidden_test.path", getattr(spec, "rel_path", ""))
        add(always, "hidden_test.source", getattr(spec, "source_url", ""))
    for issue in assets.get("heldout_issues") or ():
        add(pre_reveal, "heldout_issue.id", getattr(issue, "issue_id", ""))
        add(
            pre_reveal,
            "heldout_issue.title",
            getattr(issue, "title", ""),
            minimum_length=24,
        )
        add(
            pre_reveal,
            "heldout_issue.body",
            getattr(issue, "body", ""),
            minimum_length=40,
        )
        for value in getattr(issue, "linked_hidden_test_ids", ()) or ():
            add(always, "heldout_issue.hidden_test", value)
        for field in (
            "acceptance_hint",
            "source_url",
            "source_id",
            "title_original",
            "text_original",
        ):
            add(
                always,
                f"heldout_issue.private.{field}",
                getattr(issue, field, ""),
                minimum_length=12,
            )
    # A string the pack itself hands to the members cannot be a leak, whatever
    # else it is also written in. traffic_watch's builder wrote one sentence
    # into each hidden contract and into the public issue beside it -- "6
    # executable checks over the dashboard surface" -- so the sentence was
    # forbidden and published at once. Every prompt carrying product context
    # then failed the audit: 1385 of 1522 calls in one transfer arm were
    # refused before reaching the provider, the arm finished 336 ticks in seven
    # minutes having barely thought, and the only symptom was a treatment
    # success rate below the threshold.
    published = _agent_visible_corpus(assets)
    for table in (always, pre_reveal):
        for normalized in [k for k in table if k in published]:
            del table[normalized]

    return PromptVisibilityAuditor(
        always_forbidden=tuple(always.values()),
        pre_reveal_forbidden=tuple(pre_reveal.values()),
        heldout_revealed=lambda: bool(
            getattr(world, "_heldout_transfer_revealed", False)
        ),
    )


# What a member reads off a public issue. The rest of the file is provenance
# and evaluator bookkeeping that never reaches a prompt.
_PUBLISHED_ISSUE_FIELDS = (
    "title",
    "body",
    "text_rewritten",
    "acceptance_hint",
    "component",
)


def _agent_visible_corpus(assets: Mapping[str, Any]) -> str:
    """Everything the pack publishes to its members, case-folded.

    The public issues are what an organization is given to work from, so their
    text is the definition of what is not private. Read from the pack rather
    than from the world so the answer does not depend on what has been seeded
    yet.
    """
    root = Path(str(assets.get("hidden_tests_dir") or "")).parent.parent
    public = root / "issues" / "public"
    chunks: list[str] = []
    if public.is_dir():
        for path in sorted(public.glob("*.json")):
            try:
                issue = json.loads(path.read_text(encoding="utf-8", errors="replace"))
            except (OSError, ValueError):
                continue
            if not isinstance(issue, dict):
                continue
            # The prose a member is shown, and only that. A public issue file
            # also carries bookkeeping the member never sees -- above all
            # `linked_hidden_test_ids`, which names the evaluator's own tests.
            # Reading the file whole would publish those ids and unban the one
            # thing this audit most needs to hold.
            for field in _PUBLISHED_ISSUE_FIELDS:
                value = issue.get(field)
                if isinstance(value, str) and value.strip():
                    chunks.append(value)
    summary = (assets.get("manifest") or {}).get("product_summary")
    if summary:
        chunks.append(str(summary))
    return "\n".join(chunks).casefold()


__all__ = [
    "ExperimentResourceExhausted",
    "ExperimentResourceLedger",
    "FrozenResourceBudget",
    "LLM_CALLS",
    "LLM_PROMPT_CHARACTERS",
    "LLM_REQUESTED_TOKENS",
    "MeteredOrgLLMClient",
    "PRIMARY_ACTIONS",
    "PromptVisibilityAuditor",
    "RESOURCE_NAMES",
    "ResourceUsageEvent",
    "TICKS",
    "attach_metered_llm_client",
    "experiment_resource_snapshot",
    "initialize_world_resource_control",
    "reserve_world_resources",
    "resolve_frozen_resource_budget",
]
