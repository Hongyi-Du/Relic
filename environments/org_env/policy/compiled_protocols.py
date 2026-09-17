"""Closed, deterministic bindings for transferred protocol specifications.

The ordinary :class:`ProtocolSpec` fields are intentionally human-readable.  A
trigger or violation written in English is useful in a prompt, but it is not an
executable predicate.  This module provides the deliberately small exception:
capability-bundle v2 rows may select one of a fixed set of guards and bind it to
an exact action.  No expression evaluator, import path, or callback is accepted
from bundle data.

Compilation produces JSON-safe dictionaries only.  Checkpoints therefore store
data rather than callables, and a fresh process dispatches the stored guard id
through the code-owned registry below.  Runtime activation is per protocol
(``compiler_status == "compiled"``); it does not depend on the B3m candidate-menu
ablation.  Candidate masking is useful feedback, but :func:`authorize_before_action`
is the authoritative pre-mutation check.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from hashlib import sha256
import json
import re
from typing import Any, Callable, Mapping, Optional, Sequence


BINDING_SCHEMA_VERSION = "org_protocol_bindings_v2"

_BINDING_KEYS = frozenset(
    {
        "binding_id",
        "hook",
        "actions",
        "guard",
        "effect",
        "reason_code",
        "params",
    }
)
_ROW_REQUIRED_KEYS = frozenset(
    {
        "protocol_id",
        "name",
        "protocol_type",
        "rule_summary",
        "scope",
        "target_process",
        "supporters",
        "emergence_level",
        "capability",
        "spec",
        "bindings",
    }
)
_ROW_OPTIONAL_KEYS: frozenset[str] = frozenset()
_SPEC_KEYS = frozenset(
    {
        "trigger_condition",
        "required_steps",
        "required_fields",
        "enforcement_rule",
        "violation_condition",
        "exception_rule",
        "problem_evidence",
        "scope",
        "responsible_roles",
        "success_metric",
        "enforcement_action",
        "sunset_rule",
        "affected_agents",
        "affected_actions",
        "affected_artifacts",
        "benefits",
        "costs",
        "risks",
    }
)
_SPEC_REQUIRED_TEXT_FIELDS = frozenset(
    {
        "trigger_condition",
        "enforcement_rule",
        "violation_condition",
        "scope",
        "success_metric",
        "enforcement_action",
        "sunset_rule",
    }
)
_SPEC_STRING_LIST_FIELDS = frozenset(
    {
        "required_steps",
        "required_fields",
        "problem_evidence",
        "affected_agents",
        "affected_actions",
        "affected_artifacts",
        "benefits",
        "costs",
        "risks",
    }
)
_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_ACTION_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


GUARD_ACTIONS: Mapping[str, frozenset[str]] = {
    "issue_owner_assigned": frozenset({"open_pr"}),
    "branch_owned_by_actor": frozenset({"open_pr"}),
    "unchanged_failed_ci_not_retried": frozenset({"run_ci", "ci_test"}),
    "current_ci_attested": frozenset({"merge_pr"}),
    "independent_review": frozenset(
        {"approve_pr", "review_pr", "formal_pr_review", "merge_pr"}
    ),
    "release_gate_covered": frozenset({"publish_product_release"}),
}

GUARD_REASON_CODES: Mapping[str, str] = {
    "issue_owner_assigned": "issue_owner_required",
    "branch_owned_by_actor": "branch_owner_mismatch",
    "unchanged_failed_ci_not_retried": "unchanged_failed_ci",
    "current_ci_attested": "current_ci_attestation_required",
    "independent_review": "independent_review_required",
    "release_gate_covered": "release_gate_coverage_required",
}


@dataclass(frozen=True)
class ResolvedActionTarget:
    """A native repository object selected exactly as the action will select it."""

    target_type: str
    target_id: str
    native: Any = field(repr=False, compare=False)


@dataclass(frozen=True)
class CompiledProtocolDecision:
    """One applicable compiled binding's decision for one action attempt."""

    protocol_id: str
    protocol_spec_id: str
    binding_id: str
    action_type: str
    guard: str
    allowed: bool
    reason_code: str
    reason: str
    target_type: str
    target_id: str
    evidence: tuple[tuple[str, str], ...] = ()
    telemetry_recorded: bool = False


@dataclass(frozen=True)
class CompiledProtocolAuthorization:
    """Aggregate decision returned by the authoritative pre-action check."""

    action_type: str
    allowed: bool
    decisions: tuple[CompiledProtocolDecision, ...] = ()

    @property
    def blocking_decisions(self) -> tuple[CompiledProtocolDecision, ...]:
        return tuple(decision for decision in self.decisions if not decision.allowed)


@dataclass(frozen=True)
class _GuardResult:
    blocked: bool
    reason: str = ""
    evidence: tuple[tuple[str, str], ...] = ()


def _exact_keys(
    value: Mapping[str, Any],
    required: frozenset[str],
    *,
    optional: frozenset[str] = frozenset(),
    label: str,
) -> None:
    keys = frozenset(value.keys())
    missing = sorted(required - keys)
    unknown = sorted(keys - required - optional)
    if missing:
        raise ValueError(f"{label}_missing_keys:{','.join(missing)}")
    if unknown:
        raise ValueError(f"{label}_unknown_keys:{','.join(unknown)}")


def _nonempty_text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}_must_be_nonempty_text")
    return value


def _token(value: Any, *, label: str) -> str:
    text = _nonempty_text(value, label=label)
    if _TOKEN_RE.fullmatch(text) is None:
        raise ValueError(f"{label}_invalid")
    return text


def _action(value: Any, *, label: str) -> str:
    text = _nonempty_text(value, label=label)
    if _ACTION_RE.fullmatch(text) is None:
        raise ValueError(f"{label}_invalid")
    return text


def _string_list(
    value: Any,
    *,
    label: str,
    nonempty: bool = False,
    unique: bool = False,
) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{label}_must_be_a_list")
    if nonempty and not value:
        raise ValueError(f"{label}_must_not_be_empty")
    out: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{label}_{index}_must_be_nonempty_text")
        out.append(item)
    if unique and len(set(out)) != len(out):
        raise ValueError(f"{label}_must_be_unique")
    return out


def _validate_spec(spec: Any) -> None:
    if not isinstance(spec, Mapping):
        raise ValueError("compiled_protocol_v2_spec_must_be_an_object")
    _exact_keys(spec, _SPEC_KEYS, label="compiled_protocol_v2_spec")
    for key in _SPEC_REQUIRED_TEXT_FIELDS:
        _nonempty_text(spec.get(key), label=f"compiled_protocol_v2_spec_{key}")
    exception_rule = spec.get("exception_rule")
    if exception_rule is not None and not isinstance(exception_rule, str):
        raise ValueError("compiled_protocol_v2_spec_exception_rule_must_be_text_or_null")
    for key in _SPEC_STRING_LIST_FIELDS:
        _string_list(
            spec.get(key),
            label=f"compiled_protocol_v2_spec_{key}",
            unique=key in {"affected_agents", "affected_actions", "affected_artifacts"},
        )
    responsible_roles = spec.get("responsible_roles")
    if not isinstance(responsible_roles, Mapping):
        raise ValueError("compiled_protocol_v2_spec_responsible_roles_must_be_an_object")
    for role, actors in responsible_roles.items():
        _token(role, label="compiled_protocol_v2_spec_responsible_role")
        _string_list(
            actors,
            label=f"compiled_protocol_v2_spec_responsible_roles_{role}",
            unique=True,
        )


def _validate_binding(binding: Any, *, index: int = 0) -> None:
    label = f"compiled_protocol_v2_binding_{index}"
    if not isinstance(binding, Mapping):
        raise ValueError(f"{label}_must_be_an_object")
    _exact_keys(binding, _BINDING_KEYS, label=label)
    binding_id = _token(binding.get("binding_id"), label=f"{label}_binding_id")
    if binding.get("hook") != "before_action":
        raise ValueError(f"{label}_hook_unsupported:{binding.get('hook')}")
    if binding.get("effect") != "block_action":
        raise ValueError(f"{label}_effect_unsupported:{binding.get('effect')}")
    guard = _token(binding.get("guard"), label=f"{label}_guard")
    if guard not in GUARD_ACTIONS:
        raise ValueError(f"{label}_guard_unsupported:{guard}")
    if binding_id != f"binding_{guard}":
        raise ValueError(f"{label}_binding_id_guard_mismatch")
    actions = _string_list(
        binding.get("actions"),
        label=f"{label}_actions",
        nonempty=True,
        unique=True,
    )
    normalized_actions = {
        _action(action, label=f"{label}_action") for action in actions
    }
    expected_actions = GUARD_ACTIONS[guard]
    if normalized_actions != expected_actions:
        raise ValueError(
            f"{label}_guard_actions_mismatch:{guard}:"
            f"expected={','.join(sorted(expected_actions))}:"
            f"actual={','.join(sorted(normalized_actions))}"
        )
    reason_code = _token(binding.get("reason_code"), label=f"{label}_reason_code")
    if reason_code != GUARD_REASON_CODES[guard]:
        raise ValueError(
            f"{label}_guard_reason_code_mismatch:{guard}:"
            f"expected={GUARD_REASON_CODES[guard]}:actual={reason_code}"
        )
    params = binding.get("params")
    if not isinstance(params, Mapping) or dict(params):
        raise ValueError(f"{label}_params_must_be_empty")


def validate_protocol_row_v2(row: Mapping[str, Any]) -> None:
    """Validate one closed capability-bundle v2 protocol row.

    Validation is intentionally exact.  Unknown fields are not silently ignored,
    and a guard cannot be rebound to an action the code-owned registry did not
    declare.  Successful validation returns ``None``.
    """

    if not isinstance(row, Mapping):
        raise ValueError("compiled_protocol_v2_row_must_be_an_object")
    _exact_keys(
        row,
        _ROW_REQUIRED_KEYS,
        optional=_ROW_OPTIONAL_KEYS,
        label="compiled_protocol_v2_row",
    )
    _token(row.get("protocol_id"), label="compiled_protocol_v2_protocol_id")
    _nonempty_text(row.get("protocol_type"), label="compiled_protocol_v2_protocol_type")
    _nonempty_text(row.get("rule_summary"), label="compiled_protocol_v2_rule_summary")
    _nonempty_text(row.get("scope"), label="compiled_protocol_v2_scope")
    _nonempty_text(row.get("target_process"), label="compiled_protocol_v2_target_process")
    _string_list(
        row.get("supporters"),
        label="compiled_protocol_v2_supporters",
        unique=True,
    )
    if row.get("emergence_level") not in {"none", "weak", "strong"}:
        raise ValueError("compiled_protocol_v2_emergence_level_invalid")
    _nonempty_text(row.get("capability"), label="compiled_protocol_v2_capability")
    _nonempty_text(row.get("name"), label="compiled_protocol_v2_name")

    spec = row.get("spec")
    _validate_spec(spec)
    if str(spec.get("scope")) != str(row.get("scope")):
        raise ValueError("compiled_protocol_v2_scope_mismatch")

    bindings = row.get("bindings")
    if not isinstance(bindings, list) or len(bindings) != 1:
        raise ValueError("compiled_protocol_v2_bindings_must_contain_exactly_one_binding")
    binding_ids: set[str] = set()
    routes: set[tuple[str, str, str]] = set()
    bound_actions: set[str] = set()
    for index, binding in enumerate(bindings):
        _validate_binding(binding, index=index)
        guard = str(binding["guard"])
        if str(row.get("protocol_id")) != f"proto_{guard}":
            raise ValueError("compiled_protocol_v2_protocol_id_guard_mismatch")
        if str(row.get("protocol_type")) != guard:
            raise ValueError("compiled_protocol_v2_protocol_type_guard_mismatch")
        if str(binding.get("binding_id")) != f"binding_{guard}":
            raise ValueError("compiled_protocol_v2_binding_id_guard_mismatch")
        binding_id = str(binding["binding_id"])
        if binding_id in binding_ids:
            raise ValueError(f"compiled_protocol_v2_duplicate_binding_id:{binding_id}")
        binding_ids.add(binding_id)
        for action_type in binding["actions"]:
            route = (str(binding["hook"]), str(action_type), str(binding["guard"]))
            if route in routes:
                raise ValueError(
                    "compiled_protocol_v2_duplicate_binding_route:"
                    + ":".join(route)
                )
            routes.add(route)
            bound_actions.add(str(action_type))

    declared_actions = set(spec.get("affected_actions") or [])
    if declared_actions != bound_actions:
        raise ValueError(
            "compiled_protocol_v2_affected_actions_mismatch:"
            f"declared={','.join(sorted(declared_actions))}:"
            f"bound={','.join(sorted(bound_actions))}"
        )


def _plain_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _plain_json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, list):
        return [_plain_json_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"compiled_protocol_non_json_value:{type(value).__name__}")


def _normalized_bindings(bindings: Sequence[Mapping[str, Any]]) -> list[dict]:
    """Canonical JSON projection shared by compile-time and checkpoint load."""

    compiled: list[dict] = []
    for source in sorted(bindings, key=lambda item: str(item["binding_id"])):
        compiled.append(
            {
                "binding_id": str(source["binding_id"]),
                "hook": "before_action",
                "actions": sorted(str(action) for action in source["actions"]),
                "guard": str(source["guard"]),
                "effect": "block_action",
                "reason_code": str(source["reason_code"]),
                "params": {},
            }
        )
    return compiled


def binding_semantic_hash(
    protocol_id: str,
    spec: Mapping[str, Any],
    bindings: Sequence[Mapping[str, Any]],
    *,
    row_snapshot: Optional[Mapping[str, Any]] = None,
) -> str:
    """Hash the immutable compiler inputs, not live counters or object ids."""

    compiled = _normalized_bindings(bindings)
    hash_domain = {
        "binding_schema_version": BINDING_SCHEMA_VERSION,
        "protocol_id": str(protocol_id),
        "spec": _plain_json_value(spec),
        "bindings": compiled,
        "row_snapshot": (
            _plain_json_value(row_snapshot) if row_snapshot is not None else None
        ),
    }
    encoded = json.dumps(
        hash_domain,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def compile_protocol_row_v2(row: Mapping[str, Any]) -> tuple[list[dict], str]:
    """Compile a validated v2 row to data-only bindings and a semantic hash."""

    validate_protocol_row_v2(row)
    compiled = _normalized_bindings(row["bindings"])
    return compiled, binding_semantic_hash(
        str(row["protocol_id"]),
        row["spec"],
        compiled,
        row_snapshot={**dict(row), "bindings": compiled},
    )


def _status(value: Any) -> str:
    native = getattr(value, "value", value)
    return str(native or "").strip().casefold()


def _actor_id(actor_id: Any) -> str:
    if isinstance(actor_id, str):
        return actor_id
    for field_name in ("agent_id", "uid", "id"):
        value = getattr(actor_id, field_name, None)
        if value:
            return str(value)
    return str(actor_id or "")


def _parameters(parameters: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    if parameters is None:
        return {}
    if not isinstance(parameters, Mapping):
        raise TypeError("compiled_protocol_action_parameters_must_be_a_mapping")
    return parameters


def _repo(world: Any) -> Any:
    return getattr(getattr(world, "repo_system", None), "repo", None)


def _latest_ci(repo: Any, pull_request: Any) -> Any:
    ci_runs = getattr(repo, "ci_runs", {}) or {}
    for ci_id in reversed(list(getattr(pull_request, "ci_run_ids", []) or [])):
        result = ci_runs.get(ci_id)
        if result is not None:
            return result
    return None


def _branch_head(repo: Any, pull_request: Any) -> tuple[Any, Optional[str]]:
    source_branch = str(getattr(pull_request, "source_branch", "") or "")
    branch = (getattr(repo, "branches", {}) or {}).get(source_branch)
    commits = list(getattr(branch, "commit_ids", []) or []) if branch is not None else []
    return branch, (str(commits[-1]) if commits else None)


def _open_branch_for_actor(repo: Any, actor_id: str) -> Any:
    terminal = {"merged", "closed", "stale", "abandoned"}
    for branch in (getattr(repo, "branches", {}) or {}).values():
        if str(getattr(branch, "owner_id", "") or "") != actor_id:
            continue
        if not (getattr(branch, "commit_ids", []) or []):
            continue
        if _status(getattr(branch, "status", "")) in terminal:
            continue
        return branch
    return None


def _branch_can_open_pr(repo: Any, branch: Any) -> bool:
    if branch is None or not (getattr(branch, "commit_ids", []) or []):
        return False
    if _status(getattr(branch, "status", "")) in {
        "merged",
        "closed",
        "stale",
        "abandoned",
    }:
        return False
    for pull_request in (getattr(repo, "pull_requests", {}) or {}).values():
        if str(getattr(pull_request, "source_branch", "") or "") != str(
            getattr(branch, "branch_id", "") or ""
        ):
            continue
        if _status(getattr(pull_request, "status", "")) not in {"merged", "closed"}:
            return False
    return True


def _pull_request_is_live(pull_request: Any) -> bool:
    return _status(getattr(pull_request, "status", "")) not in {
        "merged",
        "closed",
        "stale",
    }


def _pr_needing_ci(repo: Any, world: Any) -> Any:
    active = {"open", "review_requested", "approved", "changes_requested"}
    for pull_request in (getattr(repo, "pull_requests", {}) or {}).values():
        if _status(getattr(pull_request, "status", "")) not in active:
            continue
        if bool(getattr(pull_request, "merge_conflict", False)):
            continue
        if bool(getattr(pull_request, "ci_passed", False)):
            continue
        branch, head = _branch_head(repo, pull_request)
        latest = _latest_ci(repo, pull_request)
        same_head = latest is not None and str(getattr(latest, "commit_id", "") or "") == str(head or "")
        ci_base = getattr(pull_request, "ci_base_main_commit_ids", None)
        same_base = ci_base is not None and tuple(ci_base) == tuple(
            getattr(repo, "main_commit_ids", []) or []
        )
        # The ordinary selector does not offer an unchanged failed verdict again.
        # An explicit action naming the PR still resolves and is blocked by the
        # compiled guard below.
        if latest is not None and same_head and same_base:
            continue
        if branch is not None:
            return pull_request
    return None


def _mergeable_pr(repo: Any) -> Any:
    fallback = None
    for pull_request in (getattr(repo, "pull_requests", {}) or {}).values():
        if bool(getattr(pull_request, "merge_conflict", False)):
            continue
        if _status(getattr(pull_request, "status", "")) != "approved":
            continue
        if not bool(getattr(pull_request, "ci_passed", False)):
            continue
        if getattr(pull_request, "patch_ids", []) or []:
            return pull_request
        fallback = fallback or pull_request
    return fallback


def _open_release_candidate(repo: Any, candidate_id: str = "") -> Any:
    candidates = getattr(repo, "release_candidates", {}) or {}
    if candidate_id and candidate_id in candidates:
        return candidates[candidate_id]
    for candidate in candidates.values():
        if _status(getattr(candidate, "status", "")) in {
            "draft",
            "under_review",
            "approved",
            "blocked",
        }:
            return candidate
    return None


def resolve_action_target(
    world: Any,
    actor_id: Any,
    action_type: str,
    parameters: Optional[Mapping[str, Any]] = None,
) -> Optional[ResolvedActionTarget]:
    """Resolve the native object governed by ``action_type``.

    A missing branch, PR, or release candidate is intentionally non-applicable:
    the ordinary action handler remains responsible for reporting malformed or
    stale targets.  This avoids a process protocol claiming enforcement over an
    action which could not have mutated the repository in the first place.
    """

    action_type = str(action_type or "")
    params = _parameters(parameters)
    repo = _repo(world)
    if repo is None:
        return None
    actor = _actor_id(actor_id)

    if action_type == "open_pr":
        branch_id = str(params.get("source_branch") or params.get("branch_id") or "")
        branch = (getattr(repo, "branches", {}) or {}).get(branch_id) if branch_id else None
        if branch is None and not branch_id:
            try:
                # This is the handler's own fallback.  Keep the import local so
                # execution.py can import this module without an import cycle.
                from environments.org_env.backend.repo.workflow import active_branches

                ready = [
                    candidate
                    for candidate in active_branches(world, actor)
                    if getattr(candidate, "commit_ids", []) or []
                ]
                branch = ready[0] if ready else None
            except (ImportError, AttributeError, TypeError, ValueError):
                branch = _open_branch_for_actor(repo, actor)
        if not _branch_can_open_pr(repo, branch):
            return None
        return ResolvedActionTarget(
            "branch", str(getattr(branch, "branch_id", "") or ""), branch
        )

    if action_type in {
        "run_ci",
        "ci_test",
        "approve_pr",
        "review_pr",
        "formal_pr_review",
        "merge_pr",
    }:
        pr_id = str(params.get("pr_id") or "")
        pull_request = (
            (getattr(repo, "pull_requests", {}) or {}).get(pr_id) if pr_id else None
        )
        if pull_request is None and not pr_id:
            if action_type in {"run_ci", "ci_test"}:
                try:
                    from environments.org_env.runtime_adapter.execution import (
                        _pr_needing_ci as handler_pr_needing_ci,
                    )

                    fallback_id = str(handler_pr_needing_ci(world) or "")
                    pull_request = (
                        (getattr(repo, "pull_requests", {}) or {}).get(fallback_id)
                        if fallback_id
                        else None
                    )
                except (ImportError, AttributeError, TypeError, ValueError):
                    pull_request = _pr_needing_ci(repo, world)
            elif action_type == "merge_pr":
                try:
                    from environments.org_env.runtime_adapter.execution import (
                        _mergeable_pr as handler_mergeable_pr,
                    )

                    fallback_id = str(handler_mergeable_pr(world) or "")
                    pull_request = (
                        (getattr(repo, "pull_requests", {}) or {}).get(fallback_id)
                        if fallback_id
                        else None
                    )
                except (ImportError, AttributeError, TypeError, ValueError):
                    pull_request = _mergeable_pr(repo)
        if pull_request is None:
            return None
        if not _pull_request_is_live(pull_request):
            return None
        return ResolvedActionTarget(
            "pull_request",
            str(getattr(pull_request, "pr_id", "") or ""),
            pull_request,
        )

    if action_type == "publish_product_release":
        candidate_id = str(params.get("candidate_id") or params.get("rc_id") or "")
        try:
            from environments.org_env.runtime_adapter.execution import (
                _open_rc as handler_open_release_candidate,
            )

            candidate = handler_open_release_candidate(world, candidate_id or None)
        except (ImportError, AttributeError, TypeError, ValueError):
            candidate = _open_release_candidate(repo, candidate_id)
        if candidate is None:
            return None
        if _status(getattr(candidate, "status", "")) in {
            "released",
            "closed",
            "cancelled",
            "abandoned",
        }:
            return None
        return ResolvedActionTarget(
            "release_candidate",
            str(getattr(candidate, "candidate_id", "") or ""),
            candidate,
        )
    return None


def _evidence(**values: Any) -> tuple[tuple[str, str], ...]:
    rows: list[tuple[str, str]] = []
    for key, value in sorted(values.items()):
        if isinstance(value, str):
            rendered = value
        else:
            rendered = json.dumps(
                _plain_json_value(value),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
        rows.append((str(key), rendered))
    return tuple(rows)


def _branch_linkage(world: Any, branch: Any) -> tuple[list[str], list[str]]:
    repo = _repo(world)
    if repo is None:
        return [], []
    issue_ids: list[str] = []
    task_ids: list[str] = []
    tasks = getattr(world, "tasks", {}) or {}
    artifacts = getattr(world, "product_artifacts", {}) or {}

    def add_task_namespace_link(value: Any) -> None:
        """Classify fields whose legacy name says task but whose value may not.

        Branch.linked_task and Commit.linked_task_id(s) predate the product
        artifact namespace and are also used to carry artifact identifiers.
        An artifact is not a missing Task: issue artifacts enter the issue
        ownership path, while other known artifacts are outside this guard.
        Unknown identifiers remain Tasks so corrupt or stale linkage fails
        closed instead of silently leaving the rule's scope.
        """
        text = str(value or "")
        if not text:
            return
        if tasks.get(text) is not None:
            if text not in task_ids:
                task_ids.append(text)
            return
        artifact = artifacts.get(text)
        if artifact is not None:
            if (
                str(getattr(artifact, "artifact_type", "") or "") == "issue"
                and text not in issue_ids
            ):
                issue_ids.append(text)
            return
        if text not in task_ids:
            task_ids.append(text)

    branch_task = str(getattr(branch, "linked_task", "") or "")
    if branch_task:
        add_task_namespace_link(branch_task)
    commits = getattr(repo, "commits", {}) or {}
    for commit_id in list(getattr(branch, "commit_ids", []) or []):
        commit = commits.get(commit_id)
        if commit is None:
            continue
        for issue_id in list(getattr(commit, "linked_issue_ids", []) or []):
            text = str(issue_id or "")
            if text and text not in issue_ids:
                issue_ids.append(text)
        linked_tasks = list(getattr(commit, "linked_task_ids", []) or [])
        legacy_task = str(getattr(commit, "linked_task_id", "") or "")
        if legacy_task:
            linked_tasks.append(legacy_task)
        for task_id in linked_tasks:
            add_task_namespace_link(task_id)
    # Directly linked tasks may be the only source of the issue id.
    for task_id in list(task_ids):
        task = tasks.get(task_id)
        for issue_id in list(getattr(task, "linked_issues", []) or []) if task is not None else []:
            text = str(issue_id or "")
            if text and text not in issue_ids:
                issue_ids.append(text)
    return issue_ids, task_ids


def _guard_issue_owner_assigned(
    world: Any, actor_id: str, action_type: str, target: ResolvedActionTarget
) -> _GuardResult:
    issue_ids, directly_linked_task_ids = _branch_linkage(world, target.native)
    tasks = getattr(world, "tasks", {}) or {}
    native_issues = getattr(world, "issues", {}) or {}
    artifacts = getattr(world, "product_artifacts", {}) or {}

    issue_owners: dict[str, str] = {}
    issue_task_ids: dict[str, list[str]] = {}
    unknown_issues: list[str] = []
    unowned_issues: list[str] = []
    for issue_id in issue_ids:
        reverse_associated = [
            str(task_id)
            for task_id, task in tasks.items()
            if issue_id
            in {
                str(value)
                for value in (getattr(task, "linked_issues", []) or [])
            }
        ]
        native_issue = native_issues.get(issue_id)
        if native_issue is not None:
            owner = str(getattr(native_issue, "owner_id", "") or "")
            issue_owners[issue_id] = owner
            associated = [
                str(value)
                for value in (getattr(native_issue, "linked_task_ids", []) or [])
                if str(value)
            ]
            related_task = str(getattr(native_issue, "related_task_id", "") or "")
            if related_task and related_task not in associated:
                associated.append(related_task)
            for task_id in reverse_associated:
                if task_id and task_id not in associated:
                    associated.append(task_id)
            issue_task_ids[issue_id] = associated
            if not owner:
                unowned_issues.append(issue_id)
            continue

        artifact = artifacts.get(issue_id)
        if artifact is None or str(getattr(artifact, "artifact_type", "") or "") != "issue":
            unknown_issues.append(issue_id)
            issue_task_ids[issue_id] = []
            continue
        associated: list[str] = []
        for task_id in list(getattr(artifact, "linked_task_ids", []) or []):
            text = str(task_id or "")
            if text and text not in associated:
                associated.append(text)
        for task_id in reverse_associated:
            if task_id and task_id not in associated:
                associated.append(task_id)
        issue_task_ids[issue_id] = associated
        if not associated:
            unowned_issues.append(issue_id)

    all_task_ids: list[str] = list(directly_linked_task_ids)
    for associated in issue_task_ids.values():
        for task_id in associated:
            if task_id not in all_task_ids:
                all_task_ids.append(task_id)
    task_owners = {
        task_id: str(getattr(tasks.get(task_id), "owner_id", "") or "")
        for task_id in all_task_ids
    }
    missing_tasks = [task_id for task_id in all_task_ids if tasks.get(task_id) is None]
    unowned_tasks = [
        task_id
        for task_id in all_task_ids
        if tasks.get(task_id) is not None and not task_owners[task_id]
    ]
    evidence = _evidence(
        issue_ids=issue_ids,
        issue_owners=issue_owners,
        issue_task_ids=issue_task_ids,
        directly_linked_task_ids=directly_linked_task_ids,
        task_ids=all_task_ids,
        task_owners=task_owners,
        missing_tasks=missing_tasks,
        unowned_tasks=unowned_tasks,
        unowned_issues=unowned_issues,
        unknown_issues=unknown_issues,
    )
    if not issue_ids and not directly_linked_task_ids:
        # The curated rule explicitly declares unlinked work outside its scope.
        return _GuardResult(False, evidence=evidence)
    if missing_tasks or unowned_tasks or unowned_issues or unknown_issues:
        details: list[str] = []
        if missing_tasks:
            details.append("missing tasks " + ", ".join(sorted(missing_tasks)))
        if unowned_tasks:
            details.append("unowned tasks " + ", ".join(sorted(unowned_tasks)))
        if unowned_issues:
            details.append("unowned issues " + ", ".join(sorted(unowned_issues)))
        if unknown_issues:
            details.append("unknown issues " + ", ".join(sorted(unknown_issues)))
        return _GuardResult(
            True,
            "linked work ownership is incomplete: " + "; ".join(details),
            evidence,
        )
    return _GuardResult(False, evidence=evidence)


def _guard_branch_owned_by_actor(
    world: Any, actor_id: str, action_type: str, target: ResolvedActionTarget
) -> _GuardResult:
    owner = str(getattr(target.native, "owner_id", "") or "")
    evidence = _evidence(actor_id=actor_id, branch_owner_id=owner)
    if not owner or owner != actor_id:
        return _GuardResult(
            True,
            f"{target.target_id} is owned by {owner or 'nobody'}, not {actor_id or 'the actor'}",
            evidence,
        )
    return _GuardResult(False, evidence=evidence)


def _guard_unchanged_failed_ci_not_retried(
    world: Any, actor_id: str, action_type: str, target: ResolvedActionTarget
) -> _GuardResult:
    repo = _repo(world)
    pull_request = target.native
    branch, head = _branch_head(repo, pull_request)
    latest = _latest_ci(repo, pull_request)
    latest_status = _status(getattr(latest, "status", "")) if latest is not None else ""
    latest_head = str(getattr(latest, "commit_id", "") or "") if latest is not None else ""
    ci_base = getattr(pull_request, "ci_base_main_commit_ids", None)
    current_base = tuple(getattr(repo, "main_commit_ids", []) or [])
    same_base = ci_base is not None and tuple(ci_base) == current_base
    same_head = bool(head) and latest_head == str(head)
    evidence = _evidence(
        branch_id=str(getattr(branch, "branch_id", "") or ""),
        candidate_head=str(head or ""),
        latest_ci_id=str(getattr(latest, "ci_id", "") or "") if latest is not None else "",
        latest_ci_head=latest_head,
        latest_ci_status=latest_status,
        ci_base=list(ci_base) if ci_base is not None else None,
        current_mainline=list(current_base),
    )
    if latest_status == "failed" and same_head and same_base:
        return _GuardResult(
            True,
            "the latest failed CI already tested this exact head and mainline base; change the candidate before retrying",
            evidence,
        )
    return _GuardResult(False, evidence=evidence)


def _guard_current_ci_attested(
    world: Any, actor_id: str, action_type: str, target: ResolvedActionTarget
) -> _GuardResult:
    repo = _repo(world)
    pull_request = target.native
    # The curated protocol deliberately leaves governance-only metadata PRs to
    # RepoLite's existing rules.  It binds delivery PRs, not an empty record.
    if not (getattr(pull_request, "commit_ids", []) or []) and not (
        getattr(pull_request, "patch_ids", []) or []
    ):
        return _GuardResult(
            False,
            evidence=_evidence(metadata_only=True, pull_request_id=target.target_id),
        )
    branch, head = _branch_head(repo, pull_request)
    branch_commits = list(getattr(branch, "commit_ids", []) or []) if branch is not None else []
    pr_commits = list(getattr(pull_request, "commit_ids", []) or [])
    commits = getattr(repo, "commits", {}) or {}
    missing_commit_ids = [
        str(commit_id) for commit_id in branch_commits if commits.get(commit_id) is None
    ]
    expected_patch_ids = [
        str(patch_id)
        for commit_id in branch_commits
        for patch_id in (
            getattr(commits.get(commit_id), "patch_ids", []) or []
            if commits.get(commit_id) is not None
            else []
        )
    ]
    pr_patch_ids = [str(value) for value in (getattr(pull_request, "patch_ids", []) or [])]
    latest = _latest_ci(repo, pull_request)
    latest_status = _status(getattr(latest, "status", "")) if latest is not None else ""
    latest_head = str(getattr(latest, "commit_id", "") or "") if latest is not None else ""
    ci_base = getattr(pull_request, "ci_base_main_commit_ids", None)
    current_base = tuple(getattr(repo, "main_commit_ids", []) or [])
    tree_hash_field_present = hasattr(pull_request, "ci_tree_hash")
    tree_hash = str(getattr(pull_request, "ci_tree_hash", "") or "")
    failures: list[str] = []
    if _status(getattr(pull_request, "status", "")) != "approved":
        failures.append("pull_request_not_approved")
    if bool(getattr(pull_request, "merge_conflict", False)):
        failures.append("merge_conflict_present")
    if branch is None:
        failures.append("source_branch_missing")
    if not head:
        failures.append("branch_head_missing")
    if pr_commits != branch_commits:
        failures.append("pull_request_head_not_synchronized")
    if missing_commit_ids:
        failures.append("source_branch_commit_missing")
    if pr_patch_ids != expected_patch_ids:
        failures.append("pull_request_patches_not_synchronized")
    if latest is None:
        failures.append("ci_result_missing")
    else:
        if latest_status != "passed":
            failures.append("ci_not_passed")
        if not head or latest_head != str(head):
            failures.append("ci_head_stale")
    if not bool(getattr(pull_request, "ci_passed", False)):
        failures.append("pull_request_ci_not_passed")
    if ci_base is None or tuple(ci_base) != current_base:
        failures.append("ci_mainline_base_stale")
    expected_tree_hash = ""
    if tree_hash_field_present and not tree_hash:
        # Native RepoLite historically had no tree-digest field, so absence of
        # the attribute is not a new gate.  Once a CI path elects to expose the
        # attestation, however, clearing it is explicit stale evidence.
        failures.append("ci_candidate_tree_attestation_cleared")
    evidence = _evidence(
        branch_commits=branch_commits,
        pull_request_commits=pr_commits,
        expected_patch_ids=expected_patch_ids,
        pull_request_patch_ids=pr_patch_ids,
        missing_source_commit_ids=missing_commit_ids,
        pull_request_status=_status(getattr(pull_request, "status", "")),
        merge_conflict=bool(getattr(pull_request, "merge_conflict", False)),
        candidate_head=str(head or ""),
        latest_ci_head=latest_head,
        latest_ci_status=latest_status,
        ci_base=list(ci_base) if ci_base is not None else None,
        current_mainline=list(current_base),
        ci_tree_hash=tree_hash,
        ci_tree_hash_field_present=tree_hash_field_present,
        expected_ci_tree_hash=expected_tree_hash,
        failures=failures,
    )
    if failures:
        return _GuardResult(
            True,
            "current CI attestation is incomplete: " + ", ".join(failures),
            evidence,
        )
    return _GuardResult(False, evidence=evidence)


def _guard_independent_review(
    world: Any, actor_id: str, action_type: str, target: ResolvedActionTarget
) -> _GuardResult:
    pull_request = target.native
    author = str(getattr(pull_request, "author_id", "") or "")
    approved_by = [str(value) for value in (getattr(pull_request, "approved_by", []) or []) if str(value)]
    reviewed = bool(getattr(pull_request, "reviewed", False))
    roster_ids = {str(value) for value in (getattr(world, "agents", {}) or {}).keys()}
    valid_approvers = sorted({reviewer for reviewer in approved_by if reviewer in roster_ids})
    independent = sorted({reviewer for reviewer in valid_approvers if reviewer != author})
    independent_reviewer_exists = any(agent_id != author for agent_id in roster_ids)
    evidence = _evidence(
        actor_id=actor_id,
        author_id=author,
        approved_by=approved_by,
        valid_roster_approvers=valid_approvers,
        independent_approvers=independent,
        roster_ids=sorted(roster_ids),
        independent_reviewer_exists=independent_reviewer_exists,
        reviewed=reviewed,
    )
    if action_type in {"approve_pr", "review_pr", "formal_pr_review"}:
        if not author:
            return _GuardResult(
                True,
                "the pull request has no author identity against which independence can be checked",
                evidence,
            )
        if actor_id not in roster_ids:
            return _GuardResult(
                True,
                "review or approval must be supplied by a current roster member",
                evidence,
            )
        if independent_reviewer_exists and actor_id == author:
            return _GuardResult(
                True,
                "the pull-request author cannot supply the independent review",
                evidence,
            )
        return _GuardResult(False, evidence=evidence)
    acceptable_approvers = independent if independent_reviewer_exists else valid_approvers
    if not reviewed or not acceptable_approvers:
        return _GuardResult(
            True,
            (
                "merge requires a completed approval from someone other than the author"
                if independent_reviewer_exists
                else "merge requires a completed recorded approval"
            ),
            evidence,
        )
    return _GuardResult(False, evidence=evidence)


def _guard_release_gate_covered(
    world: Any, actor_id: str, action_type: str, target: ResolvedActionTarget
) -> _GuardResult:
    repo = _repo(world)
    candidate = target.native
    required = [str(value) for value in (getattr(candidate, "required_gates", []) or []) if str(value)]
    waived = {str(value) for value in (getattr(candidate, "waived_gates", []) or []) if str(value)}
    try:
        from environments.org_env.backend.repo.release import release_gates_for

        profile_gates = [str(value) for value in release_gates_for(world)]
    except (ImportError, AttributeError, TypeError, ValueError):
        profile_gates = []
    profile_gate_set = set(profile_gates)
    results_by_gate: dict[str, list[bool]] = {}
    for row in list(getattr(candidate, "gate_results", []) or []):
        if not isinstance(row, Mapping):
            continue
        gate = str(row.get("gate") or "")
        if gate:
            results_by_gate.setdefault(gate, []).append(row.get("passed") is True)
    uncovered: list[str] = []
    for gate in required:
        if gate in waived:
            continue
        results = results_by_gate.get(gate, [])
        if len(results) != 1 or results != [True]:
            uncovered.append(gate)
    duplicate_required = sorted(
        {gate for gate in required if required.count(gate) > 1}
    )
    unknown_required = sorted(set(required) - profile_gate_set)
    missing_profile_gates = sorted(profile_gate_set - set(required))
    unknown_waivers = sorted(waived - profile_gate_set)
    unknown_result_gates = sorted(set(results_by_gate) - profile_gate_set)
    duplicate_result_gates = sorted(
        gate for gate, results in results_by_gate.items() if len(results) > 1
    )
    invalid_prs: list[str] = []
    nonpassing_prs: list[str] = []
    pull_requests = getattr(repo, "pull_requests", {}) or {}
    included_pr_ids = [
        str(value)
        for value in (getattr(candidate, "included_pr_ids", []) or [])
        if str(value)
    ]
    for pr_id in included_pr_ids:
        pull_request = pull_requests.get(pr_id)
        if pull_request is None or _status(getattr(pull_request, "status", "")) != "merged":
            invalid_prs.append(str(pr_id))
        elif not bool(getattr(pull_request, "ci_passed", False)):
            nonpassing_prs.append(str(pr_id))
    blockers = [str(value) for value in (getattr(candidate, "blockers", []) or []) if str(value)]
    status = _status(getattr(candidate, "status", ""))
    try:
        from environments.org_env.backend.repo.release import release_approval_met

        approval_met = bool(release_approval_met(world, candidate))
    except (ImportError, AttributeError, TypeError, ValueError):
        approval_met = False
    failures: list[str] = []
    if status != "approved":
        failures.append("release_candidate_not_approved")
    if not approval_met:
        failures.append("release_approval_requirements_not_met")
    if not required:
        failures.append("required_release_gates_missing")
    if not profile_gates:
        failures.append("release_gate_profile_unresolvable")
    if duplicate_required:
        failures.append("required_release_gates_duplicated")
    if unknown_required or missing_profile_gates:
        failures.append("required_release_gate_profile_mismatch")
    if unknown_waivers:
        failures.append("release_gate_waiver_unknown")
    if unknown_result_gates or duplicate_result_gates:
        failures.append("release_gate_results_malformed")
    if uncovered:
        failures.append("required_release_gates_uncovered")
    if blockers:
        failures.append("release_blockers_present")
    if not included_pr_ids:
        failures.append("included_pull_requests_missing")
    if invalid_prs:
        failures.append("included_pull_requests_not_merged")
    if nonpassing_prs:
        failures.append("included_pull_requests_ci_not_passed")
    evidence = _evidence(
        candidate_status=status,
        release_approval_met=approval_met,
        required_gates=required,
        profile_gates=profile_gates,
        duplicate_required_gates=duplicate_required,
        unknown_required_gates=unknown_required,
        missing_profile_gates=missing_profile_gates,
        waived_gates=sorted(waived),
        unknown_waived_gates=unknown_waivers,
        gate_results=results_by_gate,
        unknown_result_gates=unknown_result_gates,
        duplicate_result_gates=duplicate_result_gates,
        uncovered_gates=uncovered,
        blockers=blockers,
        included_pr_ids=included_pr_ids,
        invalid_included_pr_ids=invalid_prs,
        nonpassing_included_pr_ids=nonpassing_prs,
        failures=failures,
    )
    if failures:
        return _GuardResult(
            True,
            "release gate coverage is incomplete: " + ", ".join(failures),
            evidence,
        )
    return _GuardResult(False, evidence=evidence)


_GUARDS: Mapping[
    str,
    Callable[[Any, str, str, ResolvedActionTarget], _GuardResult],
] = {
    "issue_owner_assigned": _guard_issue_owner_assigned,
    "branch_owned_by_actor": _guard_branch_owned_by_actor,
    "unchanged_failed_ci_not_retried": _guard_unchanged_failed_ci_not_retried,
    "current_ci_attested": _guard_current_ci_attested,
    "independent_review": _guard_independent_review,
    "release_gate_covered": _guard_release_gate_covered,
}


def _registry_protocol_id(world: Any, spec: Any) -> str:
    spec_id = str(getattr(spec, "protocol_id", "") or "")
    mappings = getattr(world, "__dict__", {}).get("_protocol_mirror_ids") or {}
    mapped = str(mappings.get(spec_id) or "") if isinstance(mappings, Mapping) else ""
    if mapped:
        return mapped
    resolver = getattr(world, "_registry_mirror_id", None)
    if callable(resolver):
        try:
            resolved = str(resolver(spec_id) or "")
            if resolved:
                return resolved
        except Exception:  # noqa: BLE001 - a telemetry helper cannot change a decision
            pass
    return spec_id


def _spec_is_live(world: Any, spec: Any) -> bool:
    checker = getattr(world, "_protocol_mirror_is_live_or_absent", None)
    if callable(checker):
        try:
            return bool(checker(spec))
        except Exception:  # noqa: BLE001 - malformed liveness state is not executable
            return False
    registry = getattr(world, "protocol_registry", None)
    protocols = getattr(registry, "protocols", {}) or {}
    mirror = protocols.get(_registry_protocol_id(world, spec))
    if mirror is None:
        return True
    try:
        from environments.org_env.backend.protocol.registry import protocol_is_live

        return bool(protocol_is_live(mirror))
    except (ImportError, AttributeError, TypeError, ValueError):
        return False


def _expected_compiled_specs_from_receipt(
    world: Any,
    specs: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    """Validate the checkpoint-stable manifest of a v2 executable transfer.

    Looking only for non-empty compiler fields is insufficient: erasing every
    marker would make a formerly executable treatment disappear from the scan.
    The injection receipt is written after compilation and independently lists
    every expected spec, mirror, binding id, and semantic hash.  Treat it as the
    manifest and fail-stop when any expected compiled object or seal vanishes.
    """

    receipt = getattr(world, "__dict__", {}).get("_capability_transfer_receipt")
    if receipt is None:
        return {}
    if not isinstance(receipt, Mapping):
        raise ValueError("compiled_protocol_integrity_error:receipt_not_mapping")
    if (
        str(receipt.get("schema_version") or "") != "org_capability_bundle_v2"
        or str(receipt.get("capability_form") or "") != "executable"
    ):
        return {}
    if re.fullmatch(r"[0-9a-f]{64}", str(receipt.get("bundle_sha256") or "")) is None:
        raise ValueError("compiled_protocol_integrity_error:receipt_bundle_hash_invalid")
    bundle_snapshot = receipt.get("bundle_snapshot")
    if not isinstance(bundle_snapshot, Mapping):
        raise ValueError("compiled_protocol_integrity_error:receipt_bundle_snapshot_missing")
    try:
        from environments.org_env.experiments.capability_transfer import (
            capability_bundle_sha256,
            validate_capability_bundle,
        )

        validate_capability_bundle(bundle_snapshot)
        snapshot_hash = capability_bundle_sha256(bundle_snapshot)
    except (ImportError, TypeError, ValueError) as exc:
        raise ValueError(
            f"compiled_protocol_integrity_error:receipt_bundle_snapshot_invalid:{exc}"
        ) from exc
    if snapshot_hash != str(receipt.get("bundle_sha256") or ""):
        raise ValueError("compiled_protocol_integrity_error:receipt_bundle_snapshot_hash_mismatch")
    if str(bundle_snapshot.get("source_repository_id") or "") != str(
        receipt.get("source_repository_id") or ""
    ):
        raise ValueError("compiled_protocol_integrity_error:receipt_bundle_source_mismatch")
    source_rows = {
        str(row.get("protocol_id") or ""): row
        for row in (bundle_snapshot.get("protocols") or [])
        if isinstance(row, Mapping)
    }
    rows = receipt.get("compiled_protocols")
    injected = receipt.get("protocols_injected")
    if not isinstance(rows, list) or not rows:
        raise ValueError(
            "compiled_protocol_integrity_error:receipt_compiled_manifest_missing"
        )
    if not isinstance(injected, list) or not injected:
        raise ValueError(
            "compiled_protocol_integrity_error:receipt_injected_manifest_missing"
        )

    expected: dict[str, Mapping[str, Any]] = {}
    expected_mirrors: set[str] = set()
    expected_source_ids: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(
                f"compiled_protocol_integrity_error:receipt_row_{index}_invalid"
            )
        spec_id = str(row.get("protocol_spec_id") or "")
        mirror_id = str(row.get("protocol_id") or "")
        stored_hash = str(row.get("binding_hash") or "")
        binding_ids = row.get("binding_ids")
        if (
            not spec_id
            or not mirror_id
            or not re.fullmatch(r"[0-9a-f]{64}", stored_hash)
            or not isinstance(binding_ids, list)
            or not binding_ids
            or any(not str(value or "") for value in binding_ids)
            or len({str(value) for value in binding_ids}) != len(binding_ids)
        ):
            raise ValueError(
                f"compiled_protocol_integrity_error:receipt_row_{index}_invalid"
            )
        if spec_id in expected or mirror_id in expected_mirrors:
            raise ValueError(
                f"compiled_protocol_integrity_error:receipt_row_{index}_duplicate"
            )
        if spec_id not in specs:
            raise ValueError(
                f"compiled_protocol_integrity_error:{spec_id}:expected_spec_missing"
            )
        spec = specs[spec_id]
        source_id = str(getattr(spec, "binding_source_protocol_id", "") or "")
        if source_id in expected_source_ids:
            raise ValueError(
                f"compiled_protocol_integrity_error:{spec_id}:duplicate_source_row_binding"
            )
        source_row = source_rows.get(source_id)
        if source_row is None:
            raise ValueError(
                f"compiled_protocol_integrity_error:{spec_id}:source_row_missing"
            )
        compiled_bindings, source_hash = compile_protocol_row_v2(source_row)
        normalized_source_row = {**dict(source_row), "bindings": compiled_bindings}
        if _plain_json_value(getattr(spec, "binding_row_snapshot", None)) != _plain_json_value(
            normalized_source_row
        ):
            raise ValueError(
                f"compiled_protocol_integrity_error:{spec_id}:source_row_snapshot_mismatch"
            )
        if source_hash != stored_hash:
            raise ValueError(
                f"compiled_protocol_integrity_error:{spec_id}:source_row_hash_mismatch"
            )
        source_base = (
            source_id[len("inherited_") :]
            if source_id.startswith("inherited_")
            else source_id
        )
        if mirror_id != f"inherited_{source_base}":
            raise ValueError(
                f"compiled_protocol_integrity_error:{spec_id}:source_mirror_identity_mismatch"
            )
        expected[spec_id] = row
        expected_mirrors.add(mirror_id)
        expected_source_ids.add(source_id)

    injected_ids = {str(value or "") for value in injected if str(value or "")}
    if injected_ids != expected_mirrors:
        raise ValueError(
            "compiled_protocol_integrity_error:receipt_protocol_manifest_mismatch"
        )
    if expected_source_ids != set(source_rows):
        raise ValueError(
            "compiled_protocol_integrity_error:receipt_source_protocol_manifest_mismatch"
        )
    return expected


def _active_compiled_specs(world: Any) -> list[Any]:
    manager = getattr(world, "proposal_manager", None)
    specs = getattr(manager, "protocol_specs", {}) or {}
    expected = _expected_compiled_specs_from_receipt(world, specs)
    mappings = getattr(world, "__dict__", {}).get("_protocol_mirror_ids")
    registry = getattr(world, "protocol_registry", None)
    protocols = getattr(registry, "protocols", {}) or {}
    for spec_id, manifest in expected.items():
        spec = specs[spec_id]
        if str(getattr(spec, "compiler_status", "") or "") != "compiled":
            raise ValueError(
                f"compiled_protocol_integrity_error:{spec_id}:compiler_status_invalid"
            )
        if str(getattr(spec, "binding_schema_version", "") or "") != BINDING_SCHEMA_VERSION:
            raise ValueError(
                f"compiled_protocol_integrity_error:{spec_id}:binding_schema_invalid"
            )
        if not isinstance(mappings, Mapping) or not str(mappings.get(spec_id) or ""):
            raise ValueError(
                f"compiled_protocol_integrity_error:{spec_id}:registry_mapping_missing"
            )
        mirror_id = str(mappings[spec_id])
        if mirror_id != str(manifest.get("protocol_id") or ""):
            raise ValueError(
                f"compiled_protocol_integrity_error:{spec_id}:receipt_mirror_mismatch"
            )
        if protocols.get(mirror_id) is None:
            raise ValueError(
                f"compiled_protocol_integrity_error:{spec_id}:registry_mirror_missing"
            )
        mirror = protocols[mirror_id]
        row_snapshot = getattr(spec, "binding_row_snapshot", None)
        if not isinstance(row_snapshot, Mapping):
            raise ValueError(
                f"compiled_protocol_integrity_error:{spec_id}:row_snapshot_missing"
            )
        mirror_projection = {
            "protocol_type": str(getattr(mirror, "protocol_type", "") or ""),
            "rule_summary": str(getattr(mirror, "rule_summary", "") or ""),
            "scope": str(getattr(mirror, "scope", "") or ""),
            "target_process": str(getattr(mirror, "target_process", "") or ""),
        }
        expected_mirror_projection = {
            key: str(row_snapshot.get(key) or "") for key in mirror_projection
        }
        if mirror_projection != expected_mirror_projection:
            raise ValueError(
                f"compiled_protocol_integrity_error:{spec_id}:registry_mirror_semantics_mismatch"
            )
        if str(getattr(spec, "binding_hash", "") or "") != str(
            manifest.get("binding_hash") or ""
        ):
            raise ValueError(
                f"compiled_protocol_integrity_error:{spec_id}:receipt_hash_mismatch"
            )
        actual_binding_ids = [
            str(binding.get("binding_id") or "")
            for binding in (getattr(spec, "machine_bindings", None) or [])
            if isinstance(binding, Mapping)
        ]
        if actual_binding_ids != [
            str(value) for value in (manifest.get("binding_ids") or [])
        ]:
            raise ValueError(
                f"compiled_protocol_integrity_error:{spec_id}:receipt_bindings_mismatch"
            )
        _sealed_runtime_bindings(spec)

    active: list[Any] = []
    for spec in specs.values():
        compiled_markers_present = bool(
            str(getattr(spec, "compiler_status", "") or "") == "compiled"
            or str(getattr(spec, "binding_schema_version", "") or "")
            or (getattr(spec, "machine_bindings", None) or [])
            or str(getattr(spec, "binding_source_protocol_id", "") or "")
            or (getattr(spec, "binding_spec_snapshot", None) or {})
            or (getattr(spec, "binding_row_snapshot", None) or {})
            or str(getattr(spec, "binding_hash", "") or "")
        )
        if not compiled_markers_present:
            continue
        lifecycle = _status(getattr(spec, "status", ""))
        if lifecycle in {"deprecated", "obsolete", "rejected"}:
            continue
        spec_id = str(getattr(spec, "protocol_id", "") or "")
        if lifecycle != "adopted":
            raise ValueError(
                f"compiled_protocol_integrity_error:{spec_id}:lifecycle_not_adopted"
            )
        if str(getattr(spec, "compiler_status", "") or "") != "compiled":
            raise ValueError(
                f"compiled_protocol_integrity_error:{spec_id}:compiler_status_invalid"
            )
        if str(getattr(spec, "binding_schema_version", "") or "") != BINDING_SCHEMA_VERSION:
            raise ValueError(
                f"compiled_protocol_integrity_error:{spec_id}:binding_schema_invalid"
            )
        if not isinstance(mappings, Mapping) or not str(mappings.get(spec_id) or ""):
            raise ValueError(
                f"compiled_protocol_integrity_error:{spec_id}:registry_mapping_missing"
            )
        mirror_id = str(mappings[spec_id])
        mirror = protocols.get(mirror_id)
        if mirror is None:
            raise ValueError(
                f"compiled_protocol_integrity_error:{spec_id}:registry_mirror_missing"
            )
        try:
            from environments.org_env.backend.protocol.registry import protocol_is_live

            mirror_is_live = bool(protocol_is_live(mirror))
        except (ImportError, AttributeError, TypeError, ValueError):
            mirror_is_live = False
        if not mirror_is_live:
            raise ValueError(
                f"compiled_protocol_integrity_error:{spec_id}:registry_mirror_not_live"
            )
        # A compiled checkpoint is executable data, so its seal is part of the
        # runtime contract.  Silently ignoring a corrupt binding would turn an
        # Exec arm into Text/Removed mid-run; executing it anyway would make the
        # receipt hash meaningless.  Stop the run instead of producing an
        # unlabelled treatment.
        _sealed_runtime_bindings(spec)
        active.append(spec)
    return sorted(
        active,
        key=lambda spec: (
            _registry_protocol_id(world, spec),
            str(getattr(spec, "protocol_id", "") or ""),
        ),
    )


def _runtime_binding(binding: Any) -> Optional[dict]:
    try:
        _validate_binding(binding)
    except (TypeError, ValueError):
        return None
    return {
        "binding_id": str(binding["binding_id"]),
        "hook": str(binding["hook"]),
        "actions": tuple(str(value) for value in binding["actions"]),
        "guard": str(binding["guard"]),
        "effect": str(binding["effect"]),
        "reason_code": str(binding["reason_code"]),
        "params": {},
    }


def _sealed_runtime_bindings(spec: Any) -> list[dict]:
    """Validate and unseal one checkpointed compiled ProtocolSpec.

    The source row's spec snapshot is immutable compiler input.  Live counters,
    evidence object ids, and other runtime telemetry are intentionally outside
    the seal; none of them can select a guard or change its transition logic.
    """

    spec_id = str(getattr(spec, "protocol_id", "") or "")
    source_protocol_id = str(
        getattr(spec, "binding_source_protocol_id", "") or ""
    )
    snapshot = getattr(spec, "binding_spec_snapshot", None)
    row_snapshot = getattr(spec, "binding_row_snapshot", None)
    raw_bindings = getattr(spec, "machine_bindings", None)
    stored_hash = str(getattr(spec, "binding_hash", "") or "")
    try:
        if (
            not source_protocol_id
            or not isinstance(snapshot, Mapping)
            or not isinstance(row_snapshot, Mapping)
        ):
            raise ValueError("sealed_compiler_inputs_missing")
        _validate_spec(snapshot)
        validate_protocol_row_v2(row_snapshot)
        if _plain_json_value(row_snapshot.get("spec")) != _plain_json_value(snapshot):
            raise ValueError("sealed_row_spec_snapshot_mismatch")
        if not isinstance(raw_bindings, list) or len(raw_bindings) != 1:
            raise ValueError("sealed_binding_count_invalid")
        normalized: list[dict] = []
        for raw in raw_bindings:
            binding = _runtime_binding(raw)
            if binding is None:
                raise ValueError("sealed_binding_invalid")
            normalized.append(
                {
                    **binding,
                    "actions": list(binding["actions"]),
                }
            )
        guard = normalized[0]["guard"]
        if source_protocol_id != f"proto_{guard}":
            raise ValueError("sealed_protocol_guard_mismatch")
        if str(row_snapshot.get("protocol_id") or "") != source_protocol_id:
            raise ValueError("sealed_row_protocol_id_mismatch")
        if _normalized_bindings(row_snapshot.get("bindings") or []) != normalized:
            raise ValueError("sealed_row_bindings_mismatch")
        if str(getattr(spec, "name", "") or "") != str(
            row_snapshot.get("name") or ""
        ):
            raise ValueError("sealed_name_mismatch")
        if str(getattr(spec, "family", "") or "") != guard:
            raise ValueError("sealed_family_guard_mismatch")
        if str(row_snapshot.get("protocol_type") or "") != guard:
            raise ValueError("sealed_row_family_guard_mismatch")
        live_spec_projection = {
            "trigger_condition": str(getattr(spec, "trigger_condition", "") or ""),
            "required_steps": list(getattr(spec, "required_steps", []) or []),
            "required_fields": list(getattr(spec, "required_fields", []) or []),
            "enforcement_rule": str(getattr(spec, "enforcement_rule", "") or ""),
            "violation_condition": str(getattr(spec, "violation_condition", "") or ""),
            "exception_rule": getattr(spec, "exception_rule", None),
            "problem_evidence": list(getattr(spec, "problem_evidence", []) or []),
            "scope": str(getattr(spec, "scope", "") or ""),
            "responsible_roles": dict(getattr(spec, "responsible_roles", {}) or {}),
            "success_metric": str(getattr(spec, "success_metric", "") or ""),
            "enforcement_action": str(getattr(spec, "enforcement_action", "") or ""),
            "sunset_rule": str(getattr(spec, "sunset_rule", "") or ""),
            "affected_agents": list(getattr(spec, "affected_agents", []) or []),
            "affected_actions": list(getattr(spec, "affected_actions", []) or []),
            "benefits": list(getattr(spec, "benefits", []) or []),
            "costs": list(getattr(spec, "costs", []) or []),
            "risks": list(getattr(spec, "risks", []) or []),
        }
        sealed_live_snapshot = {
            key: value
            for key, value in snapshot.items()
            # Runtime telemetry appends governed object ids here.  The declared
            # source list remains sealed in row_snapshot/hash, but its live copy
            # is deliberately not an immutable compiler input.
            if key != "affected_artifacts"
        }
        if _plain_json_value(live_spec_projection) != _plain_json_value(
            sealed_live_snapshot
        ):
            raise ValueError("sealed_live_spec_snapshot_mismatch")
        if set(snapshot.get("affected_actions") or []) != set(
            normalized[0]["actions"]
        ):
            raise ValueError("sealed_affected_actions_mismatch")
        actual_hash = binding_semantic_hash(
            source_protocol_id,
            snapshot,
            normalized,
            row_snapshot=row_snapshot,
        )
        if not stored_hash or stored_hash != actual_hash:
            raise ValueError("sealed_binding_hash_mismatch")
        return normalized
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"compiled_protocol_integrity_error:{spec_id}:{exc}"
        ) from exc


def _evaluate_with_specs(
    world: Any,
    actor_id: Any,
    action_type: str,
    parameters: Optional[Mapping[str, Any]] = None,
) -> list[tuple[Any, CompiledProtocolDecision]]:
    action_type = str(action_type or "")
    actor = _actor_id(actor_id)
    # Validate the executable treatment before target resolution.  Otherwise a
    # corrupt binding could remain silently dormant whenever a handler-native
    # target happened to be absent, then reappear later in the same run.
    active_specs = _active_compiled_specs(world)
    target = resolve_action_target(world, actor, action_type, parameters)
    if target is None:
        return []
    evaluated: list[tuple[Any, CompiledProtocolDecision]] = []
    for spec in active_specs:
        bindings = _sealed_runtime_bindings(spec)
        for binding in sorted(bindings, key=lambda row: row["binding_id"]):
            if action_type not in binding["actions"]:
                continue
            guard = _GUARDS[binding["guard"]]
            outcome = guard(world, actor, action_type, target)
            evaluated.append(
                (
                    spec,
                    CompiledProtocolDecision(
                        protocol_id=_registry_protocol_id(world, spec),
                        protocol_spec_id=str(getattr(spec, "protocol_id", "") or ""),
                        binding_id=binding["binding_id"],
                        action_type=action_type,
                        guard=binding["guard"],
                        allowed=not outcome.blocked,
                        reason_code=binding["reason_code"],
                        reason=outcome.reason,
                        target_type=target.target_type,
                        target_id=target.target_id,
                        evidence=outcome.evidence,
                    ),
                )
            )
    return evaluated


def evaluate_before_action(
    world: Any,
    actor_id: Any,
    action_type: str,
    parameters: Optional[Mapping[str, Any]] = None,
) -> tuple[CompiledProtocolDecision, ...]:
    """Evaluate every live compiled binding without recording telemetry."""

    return tuple(
        decision
        for _spec, decision in _evaluate_with_specs(
            world, actor_id, action_type, parameters
        )
    )


def _record_exact_use(
    world: Any,
    spec: Any,
    decision: CompiledProtocolDecision,
    actor_id: str,
    tick: int,
) -> bool:
    note = getattr(world, "note_protocol_use", None)
    if callable(note):
        try:
            recorded = note(
                (),
                tick,
                obj=decision.target_id or None,
                protocol_id=decision.protocol_id,
                agent=actor_id or None,
                compiled_metadata={
                    "binding_id": decision.binding_id,
                    "guard": decision.guard,
                    "binding_hash": str(getattr(spec, "binding_hash", "") or ""),
                    "binding_schema_version": str(
                        getattr(spec, "binding_schema_version", "") or ""
                    ),
                },
            )
            return recorded is not None
        except (AttributeError, TypeError, ValueError):
            return False
    # Minimal-world fallback used by isolated runtimes/tests.  This is governance
    # telemetry only; it does not touch repo, task, product, or release state.
    spec.use_count = int(getattr(spec, "use_count", 0) or 0) + 1
    spec.last_used_tick = int(tick)
    if decision.target_id:
        artifacts = getattr(spec, "affected_artifacts", None)
        if isinstance(artifacts, list) and decision.target_id not in artifacts:
            artifacts.append(decision.target_id)
    events = getattr(world, "events", None)
    if isinstance(events, list):
        events.append(
            {
                "type": "protocol_use_event",
                "protocol_id": decision.protocol_id,
                "protocol_spec_id": decision.protocol_spec_id,
                "binding_id": decision.binding_id,
                "guard": decision.guard,
                "binding_hash": str(getattr(spec, "binding_hash", "") or ""),
                "binding_schema_version": str(
                    getattr(spec, "binding_schema_version", "") or ""
                ),
                "object_id": decision.target_id or None,
                "agent_id": actor_id or None,
                "tick": int(tick),
                "auto": True,
                "compiled": True,
            }
        )
    return True


def _record_exact_enforcement(
    world: Any,
    spec: Any,
    decision: CompiledProtocolDecision,
    actor_id: str,
    tick: int,
) -> bool:
    note = getattr(world, "note_protocol_enforcement", None)
    if callable(note):
        try:
            recorded = note(
                (),
                tick,
                obj=decision.target_id or None,
                agent=actor_id or None,
                actions=(decision.action_type,),
                protocol_id=decision.protocol_id,
                blocked=True,
                compiled_metadata={
                    "binding_id": decision.binding_id,
                    "guard": decision.guard,
                    "binding_hash": str(getattr(spec, "binding_hash", "") or ""),
                    "binding_schema_version": str(
                        getattr(spec, "binding_schema_version", "") or ""
                    ),
                },
            )
            return recorded is not None
        except (AttributeError, TypeError, ValueError):
            return False
    spec.violation_count = int(getattr(spec, "violation_count", 0) or 0) + 1
    spec.enforcement_count = int(getattr(spec, "enforcement_count", 0) or 0) + 1
    events = getattr(world, "events", None)
    if isinstance(events, list):
        common = {
            "protocol_id": decision.protocol_id,
            "protocol_spec_id": decision.protocol_spec_id,
            "binding_id": decision.binding_id,
            "guard": decision.guard,
            "binding_hash": str(getattr(spec, "binding_hash", "") or ""),
            "binding_schema_version": str(
                getattr(spec, "binding_schema_version", "") or ""
            ),
            "object_id": decision.target_id or None,
            "agent_id": actor_id or None,
            "tick": int(tick),
            "auto": True,
            "compiled": True,
        }
        events.append({"type": "protocol_violation_event", **common})
        events.append({"type": "protocol_enforcement_event", **common})
    return True


def _block_fingerprint(decision: CompiledProtocolDecision) -> str:
    payload = {
        "protocol_id": decision.protocol_id,
        "protocol_spec_id": decision.protocol_spec_id,
        "binding_id": decision.binding_id,
        "action_type": decision.action_type,
        "target_type": decision.target_type,
        "target_id": decision.target_id,
        "reason_code": decision.reason_code,
        "reason": decision.reason,
        "evidence": list(decision.evidence),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _decision_route_key(
    decision: CompiledProtocolDecision,
    actor_id: str,
) -> str:
    """Identity of one governed transition lane, separate from its state."""

    payload = {
        "protocol_id": decision.protocol_id,
        "protocol_spec_id": decision.protocol_spec_id,
        "binding_id": decision.binding_id,
        "action_type": decision.action_type,
        "target_type": decision.target_type,
        "target_id": decision.target_id,
        "actor_id": actor_id,
    }
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _block_state_by_route(world: Any) -> dict[str, str]:
    """Last continuously blocked state for each route (checkpoint-safe)."""

    state = getattr(world, "__dict__", {})
    value = state.get("_compiled_protocol_block_state_by_route")
    if isinstance(value, Mapping):
        restored = {
            str(route): str(fingerprint)
            for route, fingerprint in value.items()
            if str(route) and str(fingerprint)
        }
    else:
        restored = {}
    state["_compiled_protocol_block_state_by_route"] = restored
    return restored


def _emit_decision_event(
    world: Any,
    decision: CompiledProtocolDecision,
    actor_id: str,
    tick: int,
) -> None:
    events = getattr(world, "events", None)
    if not isinstance(events, list):
        return
    events.append(
        {
            "type": "protocol_prevented_action_event",
            "subtype": "compiled_guard_blocked",
            "protocol_id": decision.protocol_id,
            "protocol_spec_id": decision.protocol_spec_id,
            "binding_id": decision.binding_id,
            "guard": decision.guard,
            "action": decision.action_type,
            "reason_code": decision.reason_code,
            "reason": decision.reason,
            "object_type": decision.target_type,
            "object_id": decision.target_id,
            "resolved_target_type": decision.target_type,
            "resolved_target_id": decision.target_id,
            "agent_id": actor_id or None,
            "tick": int(tick),
            "allowed": decision.allowed,
            "blocked": not decision.allowed,
            "telemetry_recorded": bool(decision.telemetry_recorded),
            "evidence": dict(decision.evidence),
        }
    )


def authorize_before_action(
    world: Any,
    actor_id: Any,
    action_type: str,
    parameters: Optional[Mapping[str, Any]] = None,
    *,
    tick: Optional[int] = None,
) -> CompiledProtocolAuthorization:
    """Authorize an action and record exact-protocol governance telemetry.

    The function mutates only governance telemetry (protocol counters, events,
    and the block-deduplication set).  It never invokes an action handler and
    never mutates repository, task, product, or release business state.
    """

    actor = _actor_id(actor_id)
    observed_tick = int(
        getattr(world, "world_tick", 0) if tick is None else tick
    )
    evaluated = _evaluate_with_specs(world, actor, action_type, parameters)
    if not evaluated:
        # Preserve byte-for-byte v1/Text/Removed checkpoint shape: an action
        # outside every compiled route must not allocate compiled telemetry state.
        return CompiledProtocolAuthorization(
            action_type=str(action_type or ""),
            allowed=True,
            decisions=(),
        )
    block_state = _block_state_by_route(world)
    aggregate_allowed = not any(not decision.allowed for _, decision in evaluated)

    # Observing compliance clears the continuous-block marker even when another
    # protocol blocks the aggregate action.  If this predicate later regresses
    # to the same bad evidence it is a new incident, not a forever-suppressed
    # duplicate.
    for _spec, decision in evaluated:
        if decision.allowed:
            block_state.pop(_decision_route_key(decision, actor), None)

    use_recorded_specs: set[str] = set()
    if aggregate_allowed:
        # An action counts as successful protocol use only if every applicable
        # admission rule allows the transition.  A sibling rule's refusal means
        # the business action never happened and no passing rule gets credit.
        for spec, decision in evaluated:
            if decision.protocol_spec_id in use_recorded_specs:
                continue
            if _record_exact_use(world, spec, decision, actor, observed_tick):
                use_recorded_specs.add(decision.protocol_spec_id)

    returned: list[CompiledProtocolDecision] = []
    blocked_use_recorded_specs: set[str] = set()
    for spec, decision in evaluated:
        if decision.allowed:
            returned.append(
                replace(
                    decision,
                    telemetry_recorded=(
                        aggregate_allowed
                        and decision.protocol_spec_id in use_recorded_specs
                    ),
                )
            )
            continue

        route = _decision_route_key(decision, actor)
        fingerprint = _block_fingerprint(decision)
        is_new = block_state.get(route) != fingerprint
        telemetry_recorded = False
        if is_new:
            enforcement_recorded = _record_exact_enforcement(
                world, spec, decision, actor, observed_tick
            )
            use_recorded = decision.protocol_spec_id in blocked_use_recorded_specs
            if enforcement_recorded and not use_recorded:
                use_recorded = _record_exact_use(
                    world, spec, decision, actor, observed_tick
                )
                if use_recorded:
                    blocked_use_recorded_specs.add(decision.protocol_spec_id)
            telemetry_recorded = bool(enforcement_recorded and use_recorded)
            if telemetry_recorded:
                block_state[route] = fingerprint
        reported = replace(decision, telemetry_recorded=telemetry_recorded)
        returned.append(reported)
        if telemetry_recorded:
            _emit_decision_event(world, reported, actor, observed_tick)

    decisions = tuple(returned)
    return CompiledProtocolAuthorization(
        action_type=str(action_type or ""),
        allowed=not any(not decision.allowed for decision in decisions),
        decisions=decisions,
    )


def compiled_mask_reason(
    world: Any,
    actor_id: Any,
    action_type: str,
    parameters: Optional[Mapping[str, Any]] = None,
) -> Optional[str]:
    """Return a candidate-mask explanation without recording telemetry."""

    for decision in evaluate_before_action(world, actor_id, action_type, parameters):
        if not decision.allowed:
            detail = f": {decision.reason}" if decision.reason else ""
            return (
                f"{decision.protocol_id}/{decision.binding_id} forbids "
                f"{decision.action_type} ({decision.reason_code}){detail}"
            )
    return None


__all__ = [
    "BINDING_SCHEMA_VERSION",
    "GUARD_ACTIONS",
    "CompiledProtocolAuthorization",
    "CompiledProtocolDecision",
    "ResolvedActionTarget",
    "authorize_before_action",
    "binding_semantic_hash",
    "compile_protocol_row_v2",
    "compiled_mask_reason",
    "evaluate_before_action",
    "resolve_action_target",
    "validate_protocol_row_v2",
]
