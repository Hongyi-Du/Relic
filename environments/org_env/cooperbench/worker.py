"""Isolated B3-2 organization worker used by the CooperBench adapter."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, MutableMapping

from environments.org_env.cooperbench.contract import (
    CONTRACT_SCHEMA_VERSION,
    DELIVERY_MODE,
    TREATMENT_ID,
    PairOutcome,
    PairRequest,
)
from environments.org_env.cooperbench.lifecycle import (
    TrajectoryRecorder,
    feature_delivery_coverage,
    initialize_two_person_sdl,
    joint_delivery_digest,
    peer_review_coverage,
    protocol_realization,
)
from environments.org_env.cooperbench.surface import (
    FEATURE_INDEPENDENT_SOURCE_POLICY,
    SurfacePlan,
    materialize_public_pack,
    public_validation_capability,
    select_public_surface,
)


_B3_CONDITION = "b3_full_sociogenesis"
# A two-person coding unit needs strong implementation plus reciprocal review
# and delivery coverage.  Victor supplies systems implementation and technical
# review; Calvin supplies implementation, tests and reproducible CI.  The
# worker records this non-canonical roster explicitly and never uses it for the
# B0-B3 matrix.
_INTERNAL_AGENT_IDS = ("victor", "calvin")
_ROSTER_VARIANT = "cooperbench_two_person_b3"
_IMAGE_REPOSITORY_WORKSPACE = "/workspace/repo"
_HEALTH_TICK = 24
_MAX_FAILURE_RATE = 0.05


class CooperWorkerError(RuntimeError):
    """A treatment or delivery gate failed."""


def _terminal_coordination_requirement(world: Any) -> dict[str, Any] | None:
    """Return only coordination states that cannot continue inside the pair."""

    value = world.__dict__.get("_cooperbench_coordination_required")
    if not isinstance(value, dict):
        return None
    if value.get("kind") == "coordination_conflict":
        return None
    return value


def _run(
    command: list[str],
    *,
    cwd: Path | None = None,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        input=input_text,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )


def _git(repo_root: Path, *args: str, input_text: str | None = None) -> str:
    completed = _run(
        [
            "git",
            "-c",
            f"safe.directory={repo_root}",
            "-C",
            str(repo_root),
            *args,
        ],
        input_text=input_text,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()[:600]
        raise CooperWorkerError(f"git_{args[0]}_failed:{detail}")
    return completed.stdout


def _extract_image_repository(image: str, destination: Path) -> None:
    """Copy the public task repository out of CooperBench's Docker image."""

    destination.mkdir(parents=True, exist_ok=False)
    created = _run(["docker", "create", image])
    if created.returncode != 0:
        detail = (created.stderr or created.stdout).strip()[:600]
        raise CooperWorkerError(f"docker_create_failed:{detail}")
    container_id = created.stdout.strip().splitlines()[-1].strip()
    if not container_id or not container_id.replace("-", "").isalnum():
        raise CooperWorkerError("docker_create_container_id_invalid")
    try:
        copied = _run(
            ["docker", "cp", f"{container_id}:/workspace/repo/.", str(destination)]
        )
        if copied.returncode != 0:
            detail = (copied.stderr or copied.stdout).strip()[:600]
            raise CooperWorkerError(f"docker_copy_repository_failed:{detail}")
    finally:
        _run(["docker", "rm", "-f", container_id])
    if not (destination / ".git").exists():
        raise CooperWorkerError("cooper_image_workspace_repo_git_metadata_missing")
    _git(destination, "rev-parse", "--verify", "HEAD")


def _resolved_image_runtime(image: str) -> tuple[str, str]:
    inspected = _run(
        [
            "docker",
            "image",
            "inspect",
            image,
            "--format",
            "{{.Id}}|{{.Os}}/{{.Architecture}}",
        ]
    )
    if inspected.returncode != 0:
        detail = (inspected.stderr or inspected.stdout).strip()[:600]
        raise CooperWorkerError(f"docker_image_inspect_failed:{detail}")
    line = inspected.stdout.strip().splitlines()[-1]
    image_id, separator, platform = line.partition("|")
    if not separator or not image_id.startswith("sha256:") or "/" not in platform:
        raise CooperWorkerError("docker_image_runtime_identity_invalid")
    return image_id, platform


def _configure_worker_environment(
    request: PairRequest,
    dataset_dir: Path,
    project_id: str,
    *,
    environ: MutableMapping[str, str] | None = None,
    evaluator_image: str | None = None,
    evaluator_platform: str | None = None,
) -> None:
    # This worker exists partly to make these mutations process-local.  Remove
    # every condition/transfer control which could have leaked from a parent
    # experiment, then install the single requested treatment.
    target = os.environ if environ is None else environ
    for key in list(target):
        if key.startswith("ORG_TRANSFER_"):
            target.pop(key, None)
    for key in (
        "ORG_EXPERIMENT_CONDITION",
        "ORG_MECHANISM_ABLATIONS",
        "ORG_TEMPORARY_TEAM",
        "ORG_EVALUATOR_BACKEND",
        "ORG_EVALUATOR_CONTAINER_IMAGE",
        "ORG_EVALUATOR_CONTAINER_PLATFORM",
        "ORG_EVALUATOR_CLEAR_ENTRYPOINT",
        "RELIC_EVALUATOR_BACKEND",
        "RELIC_EVALUATOR_CONTAINER_IMAGE",
        "RELIC_EVALUATOR_CONTAINER_PLATFORM",
        "RELIC_EVALUATOR_CLEAR_ENTRYPOINT",
    ):
        target.pop(key, None)
    target.update(
        {
            "ORG_EXECUTION_PROFILE": "native",
            "ORG_PRODUCT_SUBSTRATE": "oss_time_machine",
            "ORG_OSS_DATASET": str(dataset_dir.resolve()),
            "ORG_OSS_REPOSITORY_ID": project_id,
            # Each Cooper pair runs in its own process, but formal smoke
            # exports otherwise fall back to one source-tree directory. Two
            # pairs with the same untouched repo hash can then overwrite the
            # same files concurrently. Keep every pair's export/cache tree
            # beside its private public pack so parallel workers never share
            # mutable materialization state.
            "ORG_PRODUCT_SMOKE_ROOT": str(
                (dataset_dir.resolve().parent / "product_smoke").resolve()
            ),
            # Relic's public command boundary recognizes the task image only
            # under its formal, digest-bound evaluator contract.  The source
            # tree used the older ``pilot`` name for the same Cooper path.
            "ORG_OSS_MODE": "formal" if evaluator_image else "dev",
            "ORG_OSS_CONTROL": "none",
            "ORG_OSS_ANONYMIZE": "0",
            "ORG_LLM": "1",
            "ORG_LLM_ENABLED": "1",
            "ORG_LLM_PROVIDER": request.provider,
            "ORG_LLM_MODEL": request.model_name,
            "ORG_LLM_REASONING_EFFORT": request.reasoning_effort,
            "ORG_LLM_STORE_RESPONSES": "0",
            # A streaming/drip-feed HTTPS peer can keep resetting httpx's read
            # timeout, and SIGALRM did not interrupt an observed Linux SSL
            # poll. The provider subprocess gives every Cooper attempt a
            # terminate/kill wall-clock boundary on every platform.
            "ORG_LLM_FORCE_PROVIDER_PROCESS_DEADLINE": "1",
            # Large Cooper code-editor requests need a logical-call budget
            # which remains after one slow provider attempt. The previous
            # 120-second client default made the 150-second attempt cap
            # ineffective and left no wall-clock budget for a retry.
            "ORG_LLM_REQUEST_TIMEOUT_SECONDS": "3600",
            "ORG_LLM_PROVIDER_ATTEMPT_TIMEOUT_SECONDS": "900",
            # Four concurrent Cooper workers still observed multi-minute 403/5xx
            # bursts. Retry the same logical decision instead of advancing ticks
            # without model output: 2, 4, 8, 16, 32, 64, then 128 seconds.
            "ORG_LLM_MAX_RETRIES": "7",
            "ORG_LLM_RETRY_BACKOFF_SECONDS": "2",
            "ORG_EXPERIMENT_PHASE": "cooperbench_external",
            "ORG_EXPERIMENT_ARM_ID": TREATMENT_ID,
            "ORG_BASELINE_SPRINT_TICKS": str(min(168, request.ticks)),
            "ORG_REPLAY_MAX_FRAMES": "1",
        }
    )
    if evaluator_image:
        if not evaluator_platform:
            raise CooperWorkerError("cooperbench_evaluator_platform_missing")
        target.update(
            {
                "RELIC_EVALUATOR_BACKEND": request.backend,
                "RELIC_EVALUATOR_CONTAINER_IMAGE": evaluator_image,
                "RELIC_EVALUATOR_CONTAINER_PLATFORM": evaluator_platform,
                # Cooper task images own an ENTRYPOINT that launches the
                # benchmark runner.  Public regression commands must execute
                # directly in the pinned image rather than becoming arguments
                # to that runner.
                "RELIC_EVALUATOR_CLEAR_ENTRYPOINT": "1",
            }
        )


def _require_evaluator_runtime_binding(
    request: PairRequest,
    *,
    evaluator_image: str,
    evaluator_platform: str,
) -> dict[str, Any]:
    """Resolve and attest the exact task-image executor before any live work."""

    from environments.org_env.product.materialize import (
        _formal_product_executor,
    )
    from .runtime_binding import require_evaluator_runtime

    executor = _formal_product_executor()
    if executor is None:
        raise CooperWorkerError("cooperbench_evaluator_runtime_missing")
    try:
        return require_evaluator_runtime(
            executor,
            expected_backend=request.backend,
            expected_image=evaluator_image,
            expected_platform=evaluator_platform,
        )
    except RuntimeError as error:
        raise CooperWorkerError(str(error)) from None


def _bind_feature_owners(world: Any, request: PairRequest, *, private_features: bool = True) -> dict[str, str]:
    internal_ids = tuple(str(item) for item in getattr(world, "agents", {}))
    if internal_ids != _INTERNAL_AGENT_IDS:
        raise CooperWorkerError(
            f"unexpected_two_member_roster:{list(internal_ids)}"
        )
    mapping = dict(zip(request.agents, internal_ids))
    tasks = getattr(world, "tasks", {}) or {}
    board = getattr(world, "board", None)
    for agent in (getattr(world, "agents", {}) or {}).values():
        active_tasks = getattr(agent, "active_tasks", None)
        if active_tasks is not None:
            active_tasks.clear()
    for task in tasks.values():
        task.owner_id = None
    if board is not None:
        board.owners.clear()
    artifacts = getattr(world, "product_artifacts", {}) or {}
    issue_by_internal_agent: dict[str, str] = {}
    private_briefs: dict[str, str] = {}
    artifact_by_path = {str(getattr(artifact, "linked_file_path", "") or ""): artifact_id
                        for artifact_id, artifact in artifacts.items()
                        if getattr(artifact, "linked_file_path", None)}
    for index, external_agent in enumerate(request.agents, start=1):
        issue_id = f"cooper_feature_{index}"
        task_id = f"task_oss_{issue_id}"
        task = tasks.get(task_id)
        if task is None:
            raise CooperWorkerError(f"cooper_feature_task_missing:{task_id}")
        internal_agent = mapping[external_agent]
        task.owner_id = internal_agent
        task.description = "Feature brief assigned privately to its owner." if private_features else request.tasks[external_agent]
        private_briefs[issue_id] = request.tasks[external_agent]
        # Keep the external contract one-to-one even if generic OSS evidence
        # bookkeeping has previously widened a shared artifact's task union.
        # Commit attribution uses this public task/issue assignment, not the
        # shared artifact union.
        task.linked_issues = [issue_id]
        active_tasks = getattr(world.agents[internal_agent], "active_tasks", None)
        if active_tasks is not None and task_id not in active_tasks:
            active_tasks.append(task_id)
        if board is not None:
            board.owners[task_id] = internal_agent
        issue_artifact = artifacts.get(issue_id)
        if issue_artifact is None:
            raise CooperWorkerError(f"cooper_feature_issue_missing:{issue_id}")
        issue_artifact.owner_agent_id = internal_agent
        if private_features:
            issue_artifact.problem = task.description
            issue_artifact.summary = f"CooperBench feature {index}"
            paths = (getattr(world, "_oss_component_map", {}) or {}).get(issue_id, [])
            linked_artifacts = [artifact_by_path[path] for path in paths if path in artifact_by_path]
            task.linked_artifacts = linked_artifacts
            issue_artifact.linked_artifact_ids = list(linked_artifacts)
        issue_by_internal_agent[internal_agent] = issue_id
    # The generic OSS target chooser deliberately lets any maker choose any
    # open issue.  CooperBench has a stronger public contract: each external
    # agent is assigned one feature.  Carry that already-visible assignment to
    # target selection so two features which name the same source file do not
    # collapse onto whichever issue appears first in the shared menu.
    world.__dict__["_cooperbench_feature_issue_by_agent"] = (
        issue_by_internal_agent
    )
    if private_features:
        from .visibility import initialize_private_briefs
        initialize_private_briefs(world, private_briefs, {
            issue_id: owner for owner, issue_id in issue_by_internal_agent.items()
        })
        # A feature assignment is not an automatically published external
        # report. Keep generic OSS stream/seed copies free of private text.
        for entry in world.__dict__.get("_oss_issue_stream", []) or []:
            if entry.get("issue_id") in private_briefs:
                entry["body"] = "Feature brief assigned privately to its owner."
        for entry in world.__dict__.get("_oss_seed_tasks", []) or []:
            if entry.get("task_id") in {f"task_oss_{key}" for key in private_briefs}:
                entry["description"] = "Feature brief assigned privately to its owner."
    from .work_schedule import install_compressed_schedule, install_model_call_budget
    install_compressed_schedule(world)
    install_model_call_budget(world, _INTERNAL_AGENT_IDS)
    initialize_two_person_sdl(
        world,
        {
            issue_id: internal_agent
            for internal_agent, issue_id in issue_by_internal_agent.items()
        },
    )
    # This external unit has exactly two user-supplied feature requests.  The
    # generic OSS simulation normally invents renewal issues once its coding
    # backlog appears drained; doing that here creates out-of-scope eval/docs
    # work and can displace the second Cooper feature before delivery.
    world.__dict__["_cooperbench_disable_backlog_renewal"] = True
    world.__dict__["_cooperbench_delivery_focus"] = True
    # This is the treatment-defining difference from the previous pair-local
    # implementation: Cooper now runs the main experiment's complete B3
    # episode/reflection/wish/proposal/adoption lifecycle.  Keep the flag
    # explicit so no other delivery-focus adapter changes semantics silently.
    world.__dict__["_cooperbench_main_b3_lifecycle"] = True
    # Four open wishes is a useful clustering backlog for the canonical
    # eight-person organization.  In this explicit two-person treatment it
    # requires each member to generate multiple unresolved wishes before the
    # same recurrence can even be considered.  Two remains a real recurrence
    # and the downstream model, review latency, pair approval and adoption
    # gates are unchanged.
    world.__dict__["_institution_wish_cluster_minimum"] = 2
    world.__dict__["_cooperbench_sdl_state"]["institutionalization_policy"] = (
        "main_b3_episode_reflection_wish_proposal_pair_approval_adoption"
    )
    world.__dict__["_cooperbench_sdl_state"]["late_protocol_action_policy"] = (
        "model_authored_future_action_with_one_bounded_repair_and_final_merge_hold"
    )
    world.__dict__["_cooperbench_sdl_state"]["wish_cluster_minimum"] = 2
    return mapping


def _usage_payload(client: Any) -> dict[str, int]:
    totals = getattr(client, "usage_totals", {}) or {}

    def count(value: Any) -> int:
        try:
            return max(0, int(value or 0))
        except (TypeError, ValueError):
            return 0

    return {
        "calls": count(getattr(client, "calls", 0)),
        "failures": count(getattr(client, "failures", 0)),
        "retries": count(getattr(client, "retries", 0)),
        "provider_attempts": count(getattr(client, "provider_attempts", 0)),
        "prompt_tokens": count(totals.get("prompt_tokens")),
        "completion_tokens": count(totals.get("completion_tokens")),
        "total_tokens": count(totals.get("total_tokens")),
        "cached_prompt_tokens": count(totals.get("cached_prompt_tokens")),
    }


def _usage_delta(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> dict[str, int]:
    """Return non-negative solver usage after a cumulative counter baseline."""

    keys = (
        "calls",
        "failures",
        "retries",
        "provider_attempts",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cached_prompt_tokens",
    )

    def count(value: Any) -> int:
        try:
            return max(0, int(value or 0))
        except (TypeError, ValueError):
            return 0

    return {
        key: max(0, count(after.get(key)) - count(before.get(key)))
        for key in keys
    }


def _transport_payload(client: Any) -> dict[str, Any]:
    """Return bounded provider-attempt evidence, never provider messages."""

    try:
        raw = dict(client.stats())
    except Exception:
        raw = {}

    def count(value: Any) -> int:
        try:
            return max(0, int(value or 0))
        except (TypeError, ValueError):
            return 0

    def counts(name: str) -> dict[str, int]:
        value = raw.get(name)
        if not isinstance(value, Mapping):
            return {}
        return {
            str(key)[:128]: count(item)
            for key, item in value.items()
            if count(item) > 0
        }

    attempts = count(raw.get("provider_attempts", getattr(client, "provider_attempts", 0)))
    successes = count(raw.get("provider_successes"))
    failure_counts = counts("provider_failure_counts")
    classified = successes + sum(failure_counts.values())
    last = raw.get("last_provider_failure")
    safe_last = None
    if isinstance(last, Mapping):
        category = str(last.get("category") or "provider_error")
        if category not in {
            "connection_error",
            "http_error",
            "process_error",
            "provider_error",
            "timeout",
        }:
            category = "provider_error"
        exception_type = str(last.get("exception_type") or "Exception")
        if not exception_type.replace("_", "a").replace(".", "a").isalnum():
            exception_type = "Exception"
        safe_last = {
            "attempt_number": count(last.get("attempt_number")) or 1,
            "category": category,
            "exception_type": exception_type[:128],
        }
        try:
            status = int(last.get("status_code"))
        except (TypeError, ValueError):
            status = None
        if status is not None and 100 <= status <= 599:
            safe_last["status_code"] = status
        try:
            retry_after = float(last.get("retry_after_seconds"))
        except (TypeError, ValueError):
            retry_after = None
        if retry_after is not None and 0 <= retry_after <= 60:
            safe_last["retry_after_seconds"] = retry_after
    return {
        "attempt_semantics": "admitted_send_stage_not_gateway_receipt",
        "provider_attempts": attempts,
        "provider_successes": successes,
        "provider_failure_counts": failure_counts,
        "provider_exception_counts": counts("provider_exception_counts"),
        "provider_status_counts": counts("provider_status_counts"),
        "unclassified_attempts": max(0, attempts - classified),
        "last_provider_failure": safe_last,
    }


def _transport_delta(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> dict[str, Any]:
    """Subtract cumulative transport snapshots for one bounded operation."""

    def count(value: Any) -> int:
        try:
            return max(0, int(value or 0))
        except (TypeError, ValueError):
            return 0

    def count_delta(name: str) -> dict[str, int]:
        previous = before.get(name)
        current = after.get(name)
        previous = previous if isinstance(previous, Mapping) else {}
        current = current if isinstance(current, Mapping) else {}
        keys = set(previous) | set(current)
        return {
            str(key): max(0, count(current.get(key)) - count(previous.get(key)))
            for key in keys
            if count(current.get(key)) - count(previous.get(key)) > 0
        }

    attempts = max(
        0,
        count(after.get("provider_attempts"))
        - count(before.get("provider_attempts")),
    )
    successes = max(
        0,
        count(after.get("provider_successes"))
        - count(before.get("provider_successes")),
    )
    failure_counts = count_delta("provider_failure_counts")
    classified = successes + sum(failure_counts.values())
    return {
        "attempt_semantics": "admitted_send_stage_not_gateway_receipt",
        "provider_attempts": attempts,
        "provider_successes": successes,
        "provider_failure_counts": failure_counts,
        "provider_exception_counts": count_delta("provider_exception_counts"),
        "provider_status_counts": count_delta("provider_status_counts"),
        "unclassified_attempts": max(0, attempts - classified),
        "last_provider_failure": (
            after.get("last_provider_failure")
            if sum(failure_counts.values()) > 0
            else None
        ),
    }


def _require_live_llm_preflight(client: Any) -> dict[str, Any]:
    """Prove the configured provider can serve this exact model before tick one.

    A Cooper worker is intentionally fail-closed: a live client object is not
    evidence that any provider request can succeed.  The structured challenge
    mirrors the JSON surface used by the organization's cognitive modules and
    has to return the exact value plus metered token usage.  Error details are
    already normalized by ``OpenAIOrgLLMClient`` and are bounded here so this
    receipt never becomes a credential-bearing transport dump.
    """

    before = _usage_payload(client)
    transport_before = _transport_payload(client)
    try:
        reply = client.generate_json(
            "You are a transport-integrity probe.",
            "Return transport_ready as probe and 1 as version.",
            {
                "type": "object",
                "properties": {
                    "probe": {"type": "string"},
                    "version": {"type": "integer"},
                },
                "required": ["probe", "version"],
                "additionalProperties": False,
            },
            temperature=0.0,
            max_tokens=128,
        )
    except Exception as error:
        detail = f"{type(error).__name__}:{error}"[:300]
        raise CooperWorkerError(f"llm_preflight_failed:{detail}") from None
    after = _usage_payload(client)
    transport_after = _transport_payload(client)
    call_delta = after["calls"] - before["calls"]
    failure_delta = after["failures"] - before["failures"]
    token_delta = after["total_tokens"] - before["total_tokens"]
    if not isinstance(reply, Mapping) or reply != {
        "probe": "transport_ready",
        "version": 1,
    }:
        raise CooperWorkerError("llm_preflight_failed:challenge_mismatch")
    if call_delta != 1 or failure_delta != 0 or token_delta <= 0:
        raise CooperWorkerError(
            "llm_preflight_failed:unmetered_or_failed_response:"
            f"calls={call_delta}:failures={failure_delta}:tokens={token_delta}"
        )
    return {
        "passed": True,
        "calls": call_delta,
        "failures": failure_delta,
        "total_tokens": token_delta,
        "provider_attempts": (
            after["provider_attempts"] - before["provider_attempts"]
        ),
        "transport": _transport_delta(transport_before, transport_after),
    }


def _observed_live_llm_preflight(client: Any) -> dict[str, Any]:
    """Attempt the live probe, retaining failure evidence without aborting."""

    before = _usage_payload(client)
    transport_before = _transport_payload(client)
    try:
        receipt = _require_live_llm_preflight(client)
    except CooperWorkerError as error:
        after = _usage_payload(client)
        transport_after = _transport_payload(client)
        return {
            "passed": False,
            "enforced": False,
            "policy": "observe",
            "calls": after["calls"] - before["calls"],
            "failures": after["failures"] - before["failures"],
            "total_tokens": after["total_tokens"] - before["total_tokens"],
            "provider_attempts": (
                after["provider_attempts"] - before["provider_attempts"]
            ),
            "transport": _transport_delta(
                transport_before, transport_after
            ),
            "error": str(error)[:300],
        }
    receipt["enforced"] = False
    receipt["policy"] = "observe"
    return receipt


def _public_runtime_preflight(
    world: Any, command: tuple[str, ...] | list[str]
) -> dict[str, Any]:
    """Run the untouched task image's public suite before spending model calls.

    Cooper images can define their own Docker ENTRYPOINT.  Without this gate an
    entrypoint collision looked like 168 ticks of ordinary CI activity while
    every check was actually ``not_run``.  The baseline tree supplied by the
    benchmark must execute and pass under the exact pinned runtime that later
    judges both members' PRs.
    """

    from environments.org_env.product.materialize import release_smoke

    outcome = release_smoke(
        world,
        prefer_mainline=True,
        timeout=120,
        command=list(command),
    )
    text = "\n".join(
        str(outcome.get(key) or "")
        for key in ("stdout_tail", "stderr_tail", "error")
    )
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    summary = (lines[-1] if lines else "public baseline completed")[:500]
    receipt = {
        "passed": bool(outcome.get("ok") and not outcome.get("error")),
        "command": [str(item) for item in command],
        "returncode": outcome.get("returncode"),
        "summary": summary,
        "infrastructure_error": str(outcome.get("error") or "")[:500],
    }
    runtime = getattr(world, "_cooperbench_evaluator_runtime", None)
    if isinstance(runtime, Mapping):
        receipt["evaluator_runtime"] = dict(runtime)
    return receipt


_PUBLIC_MODULE_PROBE_MARKER = "COOPERBENCH_PUBLIC_MODULE_PROBE="

# This fallback is intentionally weaker than a repository's own public test
# suite.  It proves only that the pinned task runtime can materialize and read
# the exact candidate source tree.  That is still enough to keep source
# identity, peer review, integration and the official Cooper evaluator
# reachable when an upstream repository has no runnable public suite.  Every
# receipt labels the weaker evidence; it must never be reported as a public
# regression pass.
_SOURCE_SNAPSHOT_INTEGRITY_COMMAND = [
    "python",
    "-c",
    (
        "from pathlib import Path;"
        "files=[p for p in Path('.').rglob('*') if p.is_file() and '.git' not in p.parts];"
        "assert files,'materialized source tree is empty';"
        "total=sum(p.stat().st_size for p in files);"
        "print(f'COOPERBENCH_SOURCE_SNAPSHOT_INTEGRITY files={len(files)} bytes={total}')"
    ),
]


def _source_snapshot_integrity_preflight(
    world: Any,
    *,
    candidate_failure: Mapping[str, Any] | None,
    reason: str,
) -> dict[str, Any]:
    """Prove a portable, explicitly source-only validation capability."""

    verified = _public_runtime_preflight(
        world, _SOURCE_SNAPSHOT_INTEGRITY_COMMAND
    )
    return {
        **verified,
        "selection_policy": "portable_source_snapshot_integrity_fallback",
        "candidate_command": list(
            (candidate_failure or {}).get("candidate_command")
            or (candidate_failure or {}).get("command")
            or []
        ),
        "candidate_failure": dict(candidate_failure or {}),
        "module_probe_used": bool(
            (candidate_failure or {}).get("module_probe_used")
        ),
        "functional_public_regression_available": False,
        "public_validation_strength": "source_snapshot_integrity",
        "degraded_reason": str(reason)[:500],
    }


def _verified_public_runtime_preflight(
    world: Any, command: tuple[str, ...] | list[str]
) -> dict[str, Any]:
    """Select a task-independent subset that is green on the untouched image.

    Some upstream repositories ship public test modules that assume optional
    developer dependencies or a full Git checkout. The feature-independent
    selector cannot know that from filenames, and treating their collection
    error as a solver failure prevents the pair from starting. We first run the
    fixed candidate suite unchanged. Only when it fails do we execute each
    already-selected public module in one bounded container, choose the
    strongest same-package-root group (capped at four modules), and prove that
    exact combined subset green. No feature text participates in this fallback.
    """

    if not command:
        return _source_snapshot_integrity_preflight(
            world,
            candidate_failure={
                "passed": False,
                "command": [],
                "candidate_command": [],
                "selection_policy": "no_public_regression_declared",
                "module_probe_used": False,
                "summary": "repository declares no runnable public regression suite",
                "infrastructure_error": "",
            },
            reason="public_regression_surface_unavailable",
        )

    initial = _public_runtime_preflight(world, command)
    if initial.get("passed"):
        return {
            **initial,
            "selection_policy": "fixed_feature_independent_candidate_suite",
            "candidate_command": [str(item) for item in command],
            "module_probe_used": False,
            **public_validation_capability(command),
        }
    raw = [str(item) for item in command]
    # A lockfile or packageManager field identifies the repository's preferred
    # client, not necessarily a binary installed in the benchmark image.  The
    # script itself is the public regression contract, so when that preferred
    # runner cannot start, verify the same package.json script with npm before
    # degrading the whole JavaScript/TypeScript surface to source-only checks.
    runner_failure_text = " ".join(
        str(initial.get(key) or "")
        for key in ("summary", "infrastructure_error")
    ).casefold()
    runner = raw[0].casefold() if raw else ""
    unavailable_runner = any(
        marker in runner_failure_text
        for marker in (
            f"{runner}: not found",
            f"{runner}: command not found",
            f"'{runner}' is not recognized",
        )
    )
    if (
        len(raw) >= 3
        and raw[0] in {"pnpm", "yarn"}
        and raw[1] == "run"
        and unavailable_runner
    ):
        npm_command = ["npm", "run", raw[2]]
        if raw[3:]:
            npm_command.extend(["--", *raw[3:]])
        npm_verified = _public_runtime_preflight(world, npm_command)
        if npm_verified.get("passed"):
            return {
                **npm_verified,
                "selection_policy": "verified_equivalent_package_script_runner",
                "candidate_command": raw,
                "candidate_failure": initial,
                "module_probe_used": False,
                **public_validation_capability(npm_command),
            }
    paths = [item for item in raw if item.endswith(".py") and not item.startswith("-")]
    if raw[:4] != ["python", "-m", "pytest", "-q"] or len(paths) < 2:
        failed = {
            **initial,
            "selection_policy": "fixed_candidate_suite_not_recoverable",
            "candidate_command": raw,
            "module_probe_used": False,
        }
        return _source_snapshot_integrity_preflight(
            world,
            candidate_failure=failed,
            reason=str(
                initial.get("infrastructure_error")
                or initial.get("summary")
                or "fixed public suite failed on untouched source"
            ),
        )

    from environments.org_env.product.materialize import release_smoke

    probe_script = (
        "import json,re,subprocess,sys;"
        "paths=json.loads(sys.argv[1]);rows=[];chosen=[];"
        "\nfor path in paths:"
        "\n try:"
        "\n  p=subprocess.run([sys.executable,'-m','pytest','-q',path,'--tb=no'],capture_output=True,text=True,timeout=15);"
        "\n  text=(p.stdout or '')+'\\n'+(p.stderr or '');"
        "\n  passed=sum(map(int,re.findall(r'(\\d+) passed',text)));"
        "\n  row={'path':path,'returncode':p.returncode,'passed_tests':passed,'timed_out':False};"
        "\n except subprocess.TimeoutExpired:"
        "\n  row={'path':path,'returncode':124,'passed_tests':0,'timed_out':True};"
        "\n rows.append(row);"
        "\n if row['returncode']==0 and row['passed_tests']>0: chosen.append(path);"
        f"\nprint('{_PUBLIC_MODULE_PROBE_MARKER}'+json.dumps({{'rows':rows,'chosen':chosen}},sort_keys=True))"
    )
    probe_outcome = release_smoke(
        world,
        prefer_mainline=True,
        timeout=150,
        command=["python", "-c", probe_script, json.dumps(paths)],
    )
    stdout = str(probe_outcome.get("stdout_tail") or "")
    payload: dict[str, Any] = {}
    for line in reversed(stdout.splitlines()):
        if line.startswith(_PUBLIC_MODULE_PROBE_MARKER):
            try:
                parsed = json.loads(line[len(_PUBLIC_MODULE_PROBE_MARKER):])
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict):
                payload = parsed
            break
    green = [
        str(path)
        for path in (payload.get("chosen") or [])
        if str(path) in paths
    ]
    # Monorepos commonly contain several top-level ``tests`` packages. Two
    # modules can each pass alone yet collide as ``tests.conftest`` when pytest
    # imports them in one process. Keep the strongest deterministic group that
    # shares the same package root before the first tests segment.
    grouped: dict[str, list[str]] = {}
    for path in green:
        parts = path.replace("\\", "/").split("/")
        try:
            tests_index = parts.index("tests")
        except ValueError:
            tests_index = len(parts) - 1
        root = "/".join(parts[:tests_index])
        grouped.setdefault(root, []).append(path)
    chosen = (
        sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0]))[0][1][:4]
        if grouped
        else []
    )
    recovery = {
        "attempted": True,
        "probe_infrastructure_error": str(probe_outcome.get("error") or "")[:500],
        "module_results": list(payload.get("rows") or [])[:8],
        "green_paths_before_package_grouping": green,
        "selected_paths": chosen,
    }
    if not chosen:
        failed = {
            **initial,
            "selection_policy": "fixed_candidate_suite_no_green_public_module",
            "candidate_command": raw,
            "module_probe_used": True,
            "candidate_failure": initial,
            "module_probe": recovery,
        }
        fallback = _source_snapshot_integrity_preflight(
            world,
            candidate_failure=failed,
            reason=str(
                initial.get("infrastructure_error")
                or initial.get("summary")
                or "no untouched public module was runnable"
            ),
        )
        fallback["module_probe"] = recovery
        fallback["module_probe_used"] = True
        return fallback
    selected_command = ["python", "-m", "pytest", "-q", *chosen]
    verified = _public_runtime_preflight(world, selected_command)
    selected = {
        **verified,
        "selection_policy": "feature_independent_untouched_green_module_subset",
        "candidate_command": raw,
        "module_probe_used": True,
        "candidate_failure": initial,
        "module_probe": recovery,
        "functional_public_regression_available": True,
        "public_validation_strength": "public_regression",
    }
    if selected.get("passed"):
        return selected
    return _source_snapshot_integrity_preflight(
        world,
        candidate_failure=selected,
        reason=str(
            selected.get("infrastructure_error")
            or selected.get("summary")
            or "selected untouched public module group did not remain runnable"
        ),
    )


def _bind_verified_public_test_command(
    world: Any, receipt: Mapping[str, Any]
) -> None:
    command = receipt.get("command")
    if not receipt.get("passed") or not isinstance(command, list) or not command:
        raise CooperWorkerError("verified_public_test_command_missing")
    world.__dict__["_cooperbench_verified_public_test_command"] = [
        str(item) for item in command
    ]


_NON_PYTHON_EXECUTABLE_SOURCE_SUFFIXES = {
    ".c", ".cc", ".cpp", ".cs", ".go", ".java", ".js", ".jsx",
    ".kt", ".kts", ".m", ".mm", ".php", ".rb", ".rs", ".swift",
    ".ts", ".tsx", ".vue",
}


def _behavior_probe_capability(plan: SurfacePlan) -> dict[str, Any]:
    """Describe whether the Python peer-probe runner can call this product.

    CooperBench spans several implementation languages.  The peer-probe
    runner executes Python snippets, so requiring those snippets for a Go,
    Rust or JavaScript task is a harness category error.  Those tasks retain
    exact-source CI, three-way peer review and the official evaluator, while
    the missing executable-probe layer is recorded rather than fabricated.
    """

    paths = sorted(
        {
            str(path).replace("\\", "/")
            for items in plan.component_map.values()
            for path in items
            if str(path).strip()
        }
    )
    suffixes = {
        PurePosixPath(path).suffix.casefold()
        for path in paths
        if PurePosixPath(path).suffix
    }
    non_python = sorted(suffixes & _NON_PYTHON_EXECUTABLE_SOURCE_SUFFIXES)
    python_paths = [
        path for path in paths
        if PurePosixPath(path).suffix.casefold() in {".py", ".pyi"}
    ]
    available = bool(python_paths) and not non_python
    return {
        "schema_version": "cooperbench_peer_probe_capability_v1",
        "available": available,
        "mode": (
            "python_executable_public_behavior"
            if available
            else "three_way_source_review_only"
        ),
        "reason": (
            "python_public_entrypoint_surface"
            if available
            else "python_probe_runner_incompatible_with_selected_source_language"
        ),
        "selected_feature_paths": paths,
        "selected_suffixes": sorted(suffixes),
        "non_python_executable_suffixes": non_python,
    }


def _require_public_runtime_preflight(receipt: Mapping[str, Any]) -> None:
    if receipt.get("passed"):
        return
    category = (
        "infrastructure"
        if str(receipt.get("infrastructure_error") or "")
        else "baseline"
    )
    detail = str(
        receipt.get("infrastructure_error") or receipt.get("summary") or "failed"
    )[:500]
    raise CooperWorkerError(
        f"cooperbench_public_runtime_preflight_{category}_failed:{detail}"
    )


def _treatment_health_snapshot(
    usage: Mapping[str, int], *, tick: int
) -> dict[str, Any]:
    """Build a health receipt without preventing diagnostic persistence."""

    calls = int(usage.get("calls", 0) or 0)
    failures = int(usage.get("failures", 0) or 0)
    total_tokens = int(usage.get("total_tokens", 0) or 0)
    success_rate = (calls - failures) / max(calls, 1)
    receipt = {
        "tick": int(tick),
        "calls": calls,
        "failures": failures,
        "success_rate": round(success_rate, 12),
        "total_tokens": total_tokens,
        "passed": bool(
            calls > 0
            and total_tokens > 0
            and failures / max(calls, 1) <= _MAX_FAILURE_RATE
        ),
        "threshold": {
            "max_failure_rate": _MAX_FAILURE_RATE,
            "requires_nonzero_calls": True,
            "requires_nonzero_total_tokens": True,
        },
    }
    return receipt


def _treatment_health(usage: Mapping[str, int], *, tick: int) -> dict[str, Any]:
    receipt = _treatment_health_snapshot(usage, tick=tick)
    if not receipt["passed"]:
        raise CooperWorkerError(
            "llm_treatment_health_gate_failed:"
            f"tick={tick}:calls={receipt['calls']}:"
            f"failures={receipt['failures']}:"
            f"tokens={receipt['total_tokens']}"
        )
    return receipt


def _observed_treatment_health(
    usage: Mapping[str, int], *, tick: int
) -> dict[str, Any]:
    """Record the strict health verdict without making it a delivery gate.

    The Cooper treatment keeps the provider/model/token evidence and the
    historical five-percent threshold visible, but a degraded endpoint does
    not abort an otherwise progressing two-person task. The separate t24
    minimum-liveness gate rejects only post-preflight solver trajectories with
    no usable metered response.
    """

    receipt = _treatment_health_snapshot(usage, tick=tick)
    receipt["enforced"] = False
    receipt["policy"] = "observe"
    return receipt


def _minimum_solver_liveness_snapshot(
    usage: Mapping[str, int], *, tick: int
) -> dict[str, Any]:
    """Reject only template-only solver trajectories, not degraded live ones.

    The historical five-percent treatment threshold remains observational.
    This narrower gate only asks whether at least one post-preflight solver
    query produced a usable, metered response by t24. A full benchmark can
    then re-run a provider-outage case instead of spending the remaining
    horizon on deterministic fallbacks, while a noisy but progressing case is
    allowed to continue as requested.
    """

    calls = max(0, int(usage.get("calls", 0) or 0))
    failures = max(0, int(usage.get("failures", 0) or 0))
    total_tokens = max(0, int(usage.get("total_tokens", 0) or 0))
    usable_calls = max(0, calls - failures)
    return {
        "tick": int(tick),
        "calls": calls,
        "failures": failures,
        "usable_calls": usable_calls,
        "total_tokens": total_tokens,
        "passed": bool(calls > 0 and usable_calls > 0 and total_tokens > 0),
        "enforced": True,
        "policy": "require_usable_metered_solver_response_by_t24",
        "usage_scope": "solver_queries_after_worker_transport_preflight",
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _changed_mainline_paths(world: Any) -> list[str]:
    paths = []
    for artifact in (getattr(world, "product_artifacts", {}) or {}).values():
        path = str(getattr(artifact, "linked_file_path", "") or "")
        if not path or int(getattr(artifact, "mainline_revision", 0) or 0) <= 0:
            continue
        pure = PurePosixPath(path)
        if pure.is_absolute() or ".." in pure.parts:
            raise CooperWorkerError(f"unsafe_mainline_artifact_path:{path}")
        paths.append(pure.as_posix())
    return sorted(set(paths))


def _export_joint_mainline_patch(world: Any, base_repo: Path, overlay: Path) -> tuple[str, list[str]]:
    from environments.org_env.product.materialize import export_product_repo
    from .source_views import (
        freeze_baseline, mainline_snapshot, snapshot_projection, source_views_enabled,
    )

    projection = {}
    deleted_paths: set[str] = set()
    if source_views_enabled(world):
        baseline = freeze_baseline(world)
        mainline = mainline_snapshot(world)
        baseline_files = baseline.files
        mainline_files = mainline.files
        changed_paths = sorted(
            path for path in set(baseline_files) | set(mainline_files)
            if baseline_files.get(path) != mainline_files.get(path)
        )
        deleted_paths = set(baseline_files) - set(mainline_files)
        projection = snapshot_projection(world, mainline)
    else:
        changed_paths = _changed_mainline_paths(world)
    if not changed_paths:
        raise CooperWorkerError("b3_two_agent_mainline_has_no_merged_changes")
    exported = export_product_repo(world, str(overlay), prefer_mainline=True, **projection)
    written = set(exported.get("files") or [])
    for path in changed_paths:
        destination = base_repo.joinpath(*PurePosixPath(path).parts)
        if destination.is_symlink() or not destination.resolve().is_relative_to(base_repo.resolve()):
            raise CooperWorkerError(f"unsafe_mainline_destination:{path}")
        if path in deleted_paths:
            destination.unlink(missing_ok=True)
            continue
        if path not in written:
            raise CooperWorkerError(f"changed_mainline_path_not_exported:{path}")
        source = overlay.joinpath(*PurePosixPath(path).parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination, follow_symlinks=False)

    _git(base_repo, "add", "-N", "--", ".")
    patch = _git(
        base_repo,
        "diff",
        "--binary",
        "--full-index",
        "--no-ext-diff",
        "HEAD",
        "--",
    )
    if not patch.strip():
        raise CooperWorkerError("b3_two_agent_joint_patch_empty")
    # Restore the disposable extraction to the exact image base, then prove the
    # emitted patch applies there using the same recount tolerance CooperBench
    # uses for malformed hunk line counts.
    _git(base_repo, "reset", "--hard", "HEAD")
    _git(base_repo, "clean", "-fd")
    _git(base_repo, "apply", "--check", "--recount", "-", input_text=patch)
    return patch, changed_paths


def _persist_candidate_joint_delivery(
    output_dir: Path,
    world: Any,
    base_repo: Path,
    overlay: Path,
    merged_feature_issues: set[str],
) -> tuple[str, list[str]]:
    """Persist the merged candidate before the final SDL freeze gate.

    This artifact is diagnostic/recoverable only.  The adapter still returns
    an empty joint patch unless reciprocal review and same-digest member
    attestations pass the final gate.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    patch, changed_paths = _export_joint_mainline_patch(
        world, base_repo, overlay
    )
    patch_path = output_dir / "candidate_joint.patch"
    with patch_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(patch)
    _write_json(
        output_dir / "candidate_delivery_receipt.json",
        {
            "schema_version": "orgenv_cooperbench_candidate_delivery_v1",
            "status": "unsubmitted_candidate",
            "submission_gate": "two_person_sdl_freeze_pending",
            "merged_feature_issues": sorted(merged_feature_issues),
            "changed_paths": changed_paths,
            "patch_sha256": hashlib.sha256(
                patch.encode("utf-8")
            ).hexdigest(),
            "patch_bytes": len(patch.encode("utf-8")),
            "patch_apply_check": "passed_against_image_head",
        },
    )
    return patch, changed_paths


def _communication_payload(
    world: Any, external_to_internal: Mapping[str, str]
) -> dict[str, list[dict[str, Any]]]:
    internal_to_external = {
        internal: external for external, internal in external_to_internal.items()
    }
    result = {external: [] for external in external_to_internal}
    messages = getattr(getattr(world, "comm", None), "messages", {}) or {}
    for message in sorted(
        messages.values(), key=lambda item: int(getattr(item, "created_tick", 0) or 0)
    ):
        external_sender = internal_to_external.get(
            str(getattr(message, "sender_id", "") or "")
        )
        if external_sender is None:
            continue
        recipients = [
            internal_to_external[item]
            for item in (getattr(message, "recipients", []) or [])
            if item in internal_to_external and item != getattr(message, "sender_id", "")
        ]
        result[external_sender].append(
            {
                "from": external_sender,
                "to": recipients,
                "timestamp": int(getattr(message, "created_tick", 0) or 0),
                "channel": str(getattr(message, "channel_id", "") or ""),
                "content": str(
                    getattr(message, "full_text", "")
                    or getattr(message, "text", "")
                    or getattr(message, "text_summary", "")
                    or ""
                )[:1000],
                "summary": str(getattr(message, "text_summary", "") or "")[:200],
                "linked_objects": [
                    str(item)
                    for item in (getattr(message, "linked_object_ids", []) or [])
                ][:16],
                "read_by": [
                    internal_to_external.get(str(item), str(item))
                    for item in (getattr(message, "read_by", []) or [])
                ][:8],
                "acknowledged_by": [
                    internal_to_external.get(str(item), str(item))
                    for item in (getattr(message, "acknowledged_by", []) or [])
                ][:8],
            }
        )
    return result


def _delivery_stats(world: Any) -> dict[str, Any]:
    repo = getattr(getattr(world, "repo_system", None), "repo", None)
    prs = list((getattr(repo, "pull_requests", {}) or {}).values())
    commits = list((getattr(repo, "commits", {}) or {}).values())
    patches = list((getattr(world, "patches", {}) or {}).values())
    return {
        "accepted_patches": sum(
            str(getattr(item, "validation_status", "")) == "accepted"
            for item in patches
        ),
        "merged_pull_requests": sum(
            str(getattr(getattr(item, "status", ""), "value", getattr(item, "status", "")))
            == "merged"
            for item in prs
        ),
        "merged_commits": sum(
            str(getattr(item, "status", "")) == "merged" for item in commits
        ),
        "adopted_protocols": protocol_realization(world)["adopted_protocols"],
        "operative_protocols": protocol_realization(world)["operative_protocols"],
    }


def _require_two_person_sdl_complete(world: Any) -> dict[str, Any]:
    state = getattr(world, "_cooperbench_sdl_state", {}) or {}
    delivery_coverage = feature_delivery_coverage(world)
    incomplete = sorted(
        feature_id
        for feature_id, coverage in delivery_coverage.items()
        if not coverage["complete"]
    )
    if incomplete:
        raise CooperWorkerError(
            "b3_two_agent_required_path_coverage_incomplete:"
            + ",".join(incomplete)
        )
    from .work_schedule import (
        compressed_schedule_receipt,
        model_call_budget_receipt,
    )
    schedule = compressed_schedule_receipt(world)
    call_budget = model_call_budget_receipt(world)
    if schedule is None or state.get("decision_schedule") != schedule:
        raise CooperWorkerError("b3_two_agent_decision_schedule_missing_or_stale")
    if call_budget is None:
        raise CooperWorkerError("b3_two_agent_model_call_budget_missing_or_invalid")
    if state.get("phase") != "frozen":
        raise CooperWorkerError("b3_two_agent_joint_delivery_not_frozen")
    current_digest = joint_delivery_digest(world)
    if str(state.get("frozen_digest") or "") != current_digest:
        raise CooperWorkerError("b3_two_agent_frozen_digest_is_stale")
    reviews = peer_review_coverage(world)
    if not all(reviews.values()):
        missing = ",".join(sorted(key for key, value in reviews.items() if not value))
        raise CooperWorkerError(f"b3_two_agent_peer_review_incomplete:{missing}")
    protocol = protocol_realization(world)
    return {
        "state": state,
        "feature_delivery": delivery_coverage,
        "peer_review": reviews,
        "protocol": protocol,
        "technical_delivery_satisfied": True,
        "b3_protocol_formation_satisfied": bool(
            protocol["protocol_formation_satisfied"]
        ),
        "b3_realization_satisfied": bool(protocol["realized_b3"]),
        "submission_gate_policy": "technical_delivery_independent_of_b3_realization",
        "decision_schedule": schedule,
        "model_call_budget": call_budget,
        "joint_digest": current_digest,
    }


def _merged_feature_issue_ids(world: Any) -> set[str]:
    """Cooper feature issues with provenance on an actually merged PR."""

    repo = getattr(getattr(world, "repo_system", None), "repo", None)
    merged: set[str] = set()
    for pr in (getattr(repo, "pull_requests", {}) or {}).values():
        status = str(
            getattr(
                getattr(pr, "status", ""),
                "value",
                getattr(pr, "status", ""),
            )
        )
        if status != "merged":
            continue
        issue_ids = list(getattr(pr, "linked_issue_ids", []) or [])
        linked_issue = str(getattr(pr, "linked_issue", "") or "")
        if linked_issue:
            issue_ids.append(linked_issue)
        merged.update(
            str(issue_id)
            for issue_id in issue_ids
            if str(issue_id).startswith("cooper_feature_")
        )
    return merged


def _require_all_feature_deliveries_merged(world: Any) -> set[str]:
    expected = {"cooper_feature_1", "cooper_feature_2"}
    merged = _merged_feature_issue_ids(world)
    missing = sorted(expected - merged)
    if missing:
        raise CooperWorkerError(
            "b3_two_agent_feature_delivery_incomplete:" + ",".join(missing)
        )
    return merged


def _delivery_diagnostics(world: Any) -> dict[str, Any]:
    """Persist bounded delivery state even when the final patch gate fails."""

    def action_name(row: Any) -> str:
        if isinstance(row, Mapping):
            return str(
                row.get("action_type") or row.get("action") or row.get("type") or ""
            )
        return str(
            getattr(row, "action_type", "")
            or getattr(row, "action", "")
            or type(row).__name__
        )

    patches = list((getattr(world, "patches", {}) or {}).values())
    repo = getattr(getattr(world, "repo_system", None), "repo", None)
    branches = getattr(repo, "branches", {}) or {}
    pull_requests = getattr(repo, "pull_requests", {}) or {}
    ci_runs = getattr(repo, "ci_runs", {}) or {}
    feature_tasks = {
        task_id: {
            "owner_id": str(getattr(task, "owner_id", "") or ""),
            "status": str(
                getattr(
                    getattr(task, "status", ""),
                    "value",
                    getattr(task, "status", ""),
                )
            ),
            "actual_effort": float(getattr(task, "actual_effort", 0.0) or 0.0),
        }
        for task_id, task in (getattr(world, "tasks", {}) or {}).items()
        if str(task_id).startswith("task_oss_cooper_feature_")
    }
    from .work_schedule import model_call_budget_receipt

    return {
        "tick": int(getattr(world, "world_tick", 0) or 0),
        "delivery": _delivery_stats(world),
        "agents": {
            agent_id: {
                "role": str(getattr(agent, "role", "") or ""),
                "active_tasks": list(getattr(agent, "active_tasks", []) or []),
            }
            for agent_id, agent in (getattr(world, "agents", {}) or {}).items()
        },
        "feature_tasks": feature_tasks,
        "feature_delivery": feature_delivery_coverage(world),
        "action_counts": dict(
            sorted(
                Counter(
                    action_name(row)
                    for row in (getattr(world, "action_log", []) or [])
                    if action_name(row)
                ).items()
            )
        ),
        "model_call_budget": model_call_budget_receipt(world),
        "patch_validation_counts": dict(
            sorted(
                Counter(
                    str(getattr(patch, "validation_status", "") or "unknown")
                    for patch in patches
                ).items()
            )
        ),
        "patch_rejection_reasons": dict(
            sorted(
                Counter(
                    str(getattr(patch, "rejection_reason", "") or "unspecified")
                    for patch in patches
                    if str(getattr(patch, "validation_status", "") or "")
                    == "rejected"
                ).items()
            )
        ),
        "branches": {
            branch_id: {
                "owner_id": str(getattr(branch, "owner_id", "") or ""),
                "status": str(
                    getattr(
                        getattr(branch, "status", ""),
                        "value",
                        getattr(branch, "status", ""),
                    )
                ),
                "commit_count": len(getattr(branch, "commit_ids", []) or []),
            }
            for branch_id, branch in branches.items()
        },
        "pull_requests": {
            pr_id: {
                "author_id": str(getattr(pr, "author_id", "") or ""),
                "status": str(
                    getattr(
                        getattr(pr, "status", ""),
                        "value",
                        getattr(pr, "status", ""),
                    )
                ),
                "ci_passed": bool(getattr(pr, "ci_passed", False)),
                "reviewed": bool(getattr(pr, "reviewed", False)),
                "test_status": str(getattr(pr, "test_status", "") or ""),
                "ci_run_ids": list(getattr(pr, "ci_run_ids", []) or []),
                "linked_issue_ids": sorted(
                    {
                        str(item)
                        for item in (
                            list(getattr(pr, "linked_issue_ids", []) or [])
                            + ([getattr(pr, "linked_issue", None)]
                               if getattr(pr, "linked_issue", None)
                               else [])
                        )
                        if str(item or "")
                    }
                ),
                "linked_task_ids": sorted(
                    str(item)
                    for item in (getattr(pr, "linked_task_ids", []) or [])
                    if str(item or "")
                ),
            }
            for pr_id, pr in pull_requests.items()
        },
        "ci_runs": {
            ci_id: {
                "pr_id": str(getattr(ci, "pr_id", "") or ""),
                "status": str(getattr(ci, "status", "") or ""),
                "failure_reasons": [
                    str(item)[:300]
                    for item in (getattr(ci, "failure_reasons", []) or [])
                ],
            }
            for ci_id, ci in ci_runs.items()
        },
    }


def run_pair(request: PairRequest) -> PairOutcome:
    request = request.validated()
    output_dir = Path(request.log_dir).resolve() / "orgenv_b3_two_agent"
    output_dir.mkdir(parents=True, exist_ok=True)
    ordered_tasks = {agent: request.tasks[agent] for agent in request.agents}
    project_digest = hashlib.sha256(
        f"{request.run_id}\0{request.image}".encode("utf-8")
    ).hexdigest()[:16]
    project_id = f"cooperbench_{project_digest}"
    started = dt.datetime.now(dt.timezone.utc)

    with tempfile.TemporaryDirectory(prefix="orgenv-cooper-") as temporary:
        temp_root = Path(temporary).resolve()
        base_repo = temp_root / "image_repo"
        dataset_dir = temp_root / "public_pack"
        overlay = temp_root / "mainline_overlay"
        _extract_image_repository(request.image, base_repo)
        plan: SurfacePlan = select_public_surface(
            base_repo,
            ordered_tasks,
            max_files=request.max_surface_files,
            max_total_bytes=request.max_surface_bytes,
            source_policy=FEATURE_INDEPENDENT_SOURCE_POLICY,
        )
        materialize_public_pack(
            base_repo,
            dataset_dir,
            ordered_tasks,
            plan,
            project_id=project_id,
            private_features=True,
        )
        _write_json(output_dir / "public_surface_receipt.json", plan.to_dict())
        evaluator_image, evaluator_platform = _resolved_image_runtime(request.image)
        _configure_worker_environment(
            request,
            dataset_dir,
            project_id,
            evaluator_image=evaluator_image,
            evaluator_platform=evaluator_platform,
        )

        from environments.org_env.config.baseline_conditions import (
            B3_FULL_SOCIOGENESIS,
        )
        from environments.org_env.runtime_adapter.live import OrgInspectorSession

        session = OrgInspectorSession(
            seed=request.seed,
            n_agents=2,
            member_ids=_INTERNAL_AGENT_IDS,
            load_llm=True,
            approval_mode="agent",
            experiment_condition=_B3_CONDITION,
            noncanonical_roster_variant=_ROSTER_VARIANT,
            max_frames=1,
        )
        world = session.world
        if world is None:
            raise CooperWorkerError("orgenv_world_missing")
        if (
            B3_FULL_SOCIOGENESIS != _B3_CONDITION
            or str(getattr(world, "experiment_condition", "")) != _B3_CONDITION
            or not bool(getattr(world, "experiment_condition_explicit", False))
        ):
            raise CooperWorkerError(
                "b3_two_agent_condition_contract_failed:"
                f"condition={getattr(world, 'experiment_condition', None)}:"
                f"explicit={getattr(world, 'experiment_condition_explicit', None)}"
            )
        condition = getattr(world, "condition_spec", None)
        if not (
            bool(getattr(condition, "profile_conditioning_enabled", False))
            and bool(getattr(condition, "institutionalization_enabled", False))
            and bool(getattr(condition, "capability_learning_enabled", False))
        ):
            raise CooperWorkerError("b3_two_agent_mechanisms_not_enabled")
        if len(getattr(world, "agents", {}) or {}) != 2:
            raise CooperWorkerError("b3_two_agent_roster_size_mismatch")
        evaluator_runtime = _require_evaluator_runtime_binding(
            request,
            evaluator_image=evaluator_image,
            evaluator_platform=evaluator_platform,
        )
        world.__dict__["_cooperbench_evaluator_runtime"] = dict(
            evaluator_runtime
        )
        world.__dict__["_cooperbench_image_workspace"] = (
            _IMAGE_REPOSITORY_WORKSPACE
        )
        _write_json(
            output_dir / "evaluator_runtime_receipt.json",
            evaluator_runtime,
        )
        public_runtime_preflight = _verified_public_runtime_preflight(
            world, plan.public_test_command
        )
        _write_json(
            output_dir / "public_runtime_preflight.json",
            public_runtime_preflight,
        )
        _require_public_runtime_preflight(public_runtime_preflight)
        _bind_verified_public_test_command(world, public_runtime_preflight)
        # Bind the untouched-suite verdict into the live world.  The public
        # test handler uses it to distinguish a candidate-induced collection
        # failure from an image/launcher outage: this exact command and runtime
        # were already proven green before any organization edit.
        world.__dict__["_cooperbench_public_runtime_preflight"] = dict(
            public_runtime_preflight
        )
        behavior_probe_capability = _behavior_probe_capability(plan)
        world.__dict__["_cooperbench_behavior_probe_capability"] = dict(
            behavior_probe_capability
        )
        client = getattr(world, "llm_client", None)
        if client is None:
            detail = str(getattr(world, "llm_client_load_error", "") or "")
            raise CooperWorkerError(f"llm_client_not_attached:{detail}")
        llm_preflight = _observed_live_llm_preflight(client)
        _write_json(output_dir / "llm_preflight.json", llm_preflight)
        # The transport challenge is part of provider cost, but it is not a
        # solver query and must not make a template-only trajectory look live.
        solver_usage_baseline = _usage_payload(client)
        # Task-to-path routing is host metadata, never a public pack hint about
        # the other member's feature. Restore it only inside the live adapter.
        world.__dict__["_oss_component_map"] = {key: list(paths) for key, paths in plan.component_map.items()}
        world.__dict__["_cooperbench_public_test_paths"] = list(
            plan.public_test_paths
        )
        ownership = _bind_feature_owners(world, request)
        from .call_trace import RecordedCooperClient
        client = RecordedCooperClient(world, client, output_dir / "model_calls.jsonl")
        world.llm_client = client
        from .source_views import freeze_baseline, initialize_actor_desks
        freeze_baseline(world)
        initialize_actor_desks(world)
        if behavior_probe_capability["available"]:
            from .replay_workflow import SCHEMA as REPLAY_WORKFLOW_SCHEMA
            world.__dict__["_cooperbench_integrated_probe_workflow"] = {
                "schema_version": REPLAY_WORKFLOW_SCHEMA
            }
            integrated_probe_policy = (
                "actual_peer_candidate_and_two_member_final_mainline_replay"
            )
        else:
            integrated_probe_policy = (
                "unavailable_for_source_language_three_way_source_review_and_official_eval"
            )
        world.__dict__["_cooperbench_sdl_state"]["integrated_probe_policy"] = (
            integrated_probe_policy
        )
        world.__dict__["_cooperbench_sdl_state"]["behavior_probe_capability"] = (
            dict(behavior_probe_capability)
        )
        world.__dict__["_cooperbench_sdl_state"]["workspace_policy"] = "actor_private_pinned_desk_explicit_mainline_sync"
        world.__dict__["_cooperbench_sdl_state"]["public_test_policy"] = "actor_and_exact_tree_scoped"
        world.__dict__["_cooperbench_sdl_state"]["committed_source_policy"] = "frozen_pr_head_and_explicit_integration"
        trajectory_path = output_dir / "trajectory.jsonl"
        if trajectory_path.exists():
            trajectory_path.unlink()
        trajectory = TrajectoryRecorder(trajectory_path)
        trajectory.capture(world)

        health_t24: dict[str, Any] | None = None
        liveness_failure: str | None = None
        while int(getattr(world, "world_tick", 0) or 0) < request.ticks:
            remaining = request.ticks - int(getattr(world, "world_tick", 0) or 0)
            step_count = min(_HEALTH_TICK, remaining)
            for _ in range(step_count):
                world.step()
                trajectory.capture(world)
                if _terminal_coordination_requirement(world):
                    break
                if (
                    int(getattr(world, "world_tick", 0) or 0) >= _HEALTH_TICK
                    and (
                        getattr(world, "_cooperbench_sdl_state", {}) or {}
                    ).get("phase") == "frozen"
                ):
                    break
            current_tick = int(getattr(world, "world_tick", 0) or 0)
            if _terminal_coordination_requirement(world):
                break
            if health_t24 is None and current_tick >= _HEALTH_TICK:
                solver_usage_t24 = _usage_delta(
                    solver_usage_baseline, _usage_payload(client)
                )
                health_t24 = _observed_treatment_health(
                    solver_usage_t24, tick=current_tick
                )
                health_t24["usage_scope"] = (
                    "solver_queries_after_worker_transport_preflight"
                )
                minimum_liveness = _minimum_solver_liveness_snapshot(
                    solver_usage_t24, tick=current_tick
                )
                health_t24["minimum_liveness"] = minimum_liveness
                health_t24["transport"] = _transport_payload(client)
                health_t24["transport_scope"] = (
                    "cumulative_including_worker_transport_preflight"
                )
                _write_json(output_dir / "health_t24.json", health_t24)
                if not minimum_liveness["passed"]:
                    liveness_failure = (
                        "llm_treatment_liveness_gate_failed:"
                        f"tick={current_tick}:calls={minimum_liveness['calls']}:"
                        f"failures={minimum_liveness['failures']}:"
                        f"tokens={minimum_liveness['total_tokens']}"
                    )
                    break
            if (
                current_tick >= _HEALTH_TICK
                and (getattr(world, "_cooperbench_sdl_state", {}) or {}).get(
                    "phase"
                )
                == "frozen"
            ):
                break

        executed_ticks = int(getattr(world, "world_tick", 0) or 0)
        final_usage = _usage_payload(client)
        final_solver_usage = _usage_delta(solver_usage_baseline, final_usage)
        final_health = _treatment_health_snapshot(
            final_solver_usage, tick=executed_ticks
        )
        final_health["enforced"] = False
        final_health["policy"] = "observe"
        final_health["usage_scope"] = (
            "solver_queries_after_worker_transport_preflight"
        )
        final_health["minimum_liveness"] = _minimum_solver_liveness_snapshot(
            final_solver_usage, tick=executed_ticks
        )
        final_transport = _transport_payload(client)
        final_health["transport"] = final_transport
        delivery_diagnostics = _delivery_diagnostics(world)
        delivery_diagnostics["llm_usage"] = final_usage
        delivery_diagnostics["solver_llm_usage"] = final_solver_usage
        delivery_diagnostics["llm_transport"] = final_transport
        delivery_diagnostics["llm_health_final"] = final_health
        delivery_diagnostics["sdl"] = getattr(
            world, "_cooperbench_sdl_state", {}
        )
        delivery_diagnostics["peer_review"] = peer_review_coverage(world)
        delivery_diagnostics["protocol"] = protocol_realization(world)
        delivery_diagnostics["trajectory_path"] = str(trajectory_path)
        delivery_diagnostics["coordination_required"] = world.__dict__.get("_cooperbench_coordination_required")
        _write_json(
            output_dir / "delivery_diagnostics.json",
            delivery_diagnostics,
        )
        # Always leave the trajectory and delivery diagnosis behind before a
        # terminal health rejection.  The prior ordering erased the most useful
        # failure-mode artifact on exactly the runs that needed analysis.
        if liveness_failure is not None:
            raise CooperWorkerError(liveness_failure)
        terminal_coordination = _terminal_coordination_requirement(world)
        if terminal_coordination:
            reason = terminal_coordination.get("kind") or "coordination_required"
            raise CooperWorkerError("cooperbench_coordination_required:" + str(reason))
        merged_feature_issues = _require_all_feature_deliveries_merged(world)
        joint_patch, changed_paths = _persist_candidate_joint_delivery(
            output_dir,
            world,
            base_repo,
            overlay,
            merged_feature_issues,
        )
        sdl = _require_two_person_sdl_complete(world)
        communications = _communication_payload(world, ownership)
        patch_digest = hashlib.sha256(joint_patch.encode("utf-8")).hexdigest()
        completed = dt.datetime.now(dt.timezone.utc)
        from .work_schedule import model_call_budget_receipt

        receipt = {
            "schema_version": CONTRACT_SCHEMA_VERSION,
            "treatment_id": TREATMENT_ID,
            "delivery_mode": DELIVERY_MODE,
            "run_id": request.run_id,
            "organization": {
                "condition": _B3_CONDITION,
                "condition_explicit": True,
                "roster_size": 2,
                "noncanonical_roster_variant": True,
                "information_policy": "owner_private_explicit_share_read",
                "initial_briefs_owner_private": True,
                "workspace_policy": "actor_private_pinned_desk_explicit_mainline_sync",
                "public_test_policy": "actor_and_exact_tree_scoped",
                "committed_source_policy": "frozen_pr_head_and_explicit_integration",
                "integrated_probe_policy": integrated_probe_policy,
                "behavior_probe_capability": behavior_probe_capability,
                "initial_source_policy": FEATURE_INDEPENDENT_SOURCE_POLICY,
                "source_context_policy": "actor_local_brief_and_query_retrieval",
                "standard_coop_information_boundary": False,
                "roster_variant_id": _ROSTER_VARIANT,
                "internal_members": list(_INTERNAL_AGENT_IDS),
                "external_to_internal": ownership,
                "b0_b1_b2_executed": False,
                "execution_profile": "native",
                "transfer_applied": False,
            },
            "runtime": {
                "backend": request.backend,
                "image": request.image,
                "model": request.model_name,
                "provider": request.provider,
                "reasoning_effort": request.reasoning_effort,
                "provider_attempt_timeout_seconds": getattr(
                    client, "provider_attempt_timeout_seconds", None
                ),
                "seed": request.seed,
                "requested_tick_budget": request.ticks,
                "executed_ticks": executed_ticks,
                "stopped_on_frozen_delivery": executed_ticks < request.ticks,
                "model_call_budget": model_call_budget_receipt(world),
            },
            "public_surface": {
                "surface_digest": plan.baseline_digest,
                "host_selection_digest": plan.digest,
                "source_policy": plan.source_policy,
                "runtime_assets": plan.runtime_assets_manifest,
                "selected_file_count": len(plan.files),
                "selected_bytes": plan.total_bytes,
                "feature_count": 2,
                "public_test_paths": list(plan.public_test_paths),
                "public_test_command": list(public_runtime_preflight["command"]),
                "public_test_candidate_command": list(plan.public_test_command),
                "public_validation_strength": str(
                    public_runtime_preflight.get("public_validation_strength")
                    or "public_regression"
                ),
                "functional_public_regression_available": bool(
                    public_runtime_preflight.get(
                        "functional_public_regression_available", True
                    )
                ),
                "cooper_evaluator_assets_copied": False,
            },
            "public_runtime_preflight": public_runtime_preflight,
            "llm_health_t24": health_t24,
            "llm_health_final": final_health,
            "llm_preflight": llm_preflight,
            "llm_usage": final_usage,
            "solver_llm_usage": final_solver_usage,
            "llm_transport": final_transport,
            "cost_accounting": {
                "cost_usd": None,
                "status": "unavailable_from_orgenv_client",
            },
            "delivery": {
                **_delivery_stats(world),
                "merged_feature_issues": sorted(merged_feature_issues),
                "changed_paths": changed_paths,
                "patch_sha256": patch_digest,
                "patch_bytes": len(joint_patch.encode("utf-8")),
                "patch_apply_check": "passed_against_image_head",
                "sdl": sdl,
                "trajectory": {
                    "schema_version": "orgenv_cooperbench_trajectory_v1",
                    "path": str(trajectory_path),
                },
                "cooperbench_evaluation_run": False,
            },
            "timing": {
                "started_at": started.isoformat(),
                "completed_at": completed.isoformat(),
                "elapsed_seconds": round((completed - started).total_seconds(), 6),
            },
        }
        _write_json(output_dir / "organization_receipt.json", receipt)
        return PairOutcome(
            schema_version=CONTRACT_SCHEMA_VERSION,
            treatment_id=TREATMENT_ID,
            delivery_mode=DELIVERY_MODE,
            run_id=request.run_id,
            agents=request.agents,
            status="Submitted",
            joint_patch=joint_patch,
            usage_totals=final_usage,
            communications_by_agent=communications,
            receipt=receipt,
        ).validated()


def _safe_error(error: BaseException) -> str:
    try:
        from relic.research.public_redaction import redact_public_evidence

        return str(
            redact_public_evidence(
                f"{type(error).__name__}:{error}"
            )
        )[:1000]
    except Exception:
        return type(error).__name__


def _load_request(path: Path) -> PairRequest:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CooperWorkerError("pair_request_file_invalid") from error
    if not isinstance(payload, dict):
        raise CooperWorkerError("pair_request_payload_must_be_object")
    return PairRequest.from_dict(payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args(argv)
    result_path = Path(args.result).resolve()
    request = _load_request(Path(args.request).resolve())
    try:
        outcome = run_pair(request)
    except Exception as error:
        safe = _safe_error(error)
        outcome = PairOutcome(
            schema_version=CONTRACT_SCHEMA_VERSION,
            treatment_id=TREATMENT_ID,
            delivery_mode=DELIVERY_MODE,
            run_id=request.run_id,
            agents=request.agents,
            status="Error",
            joint_patch="",
            receipt={
                "schema_version": CONTRACT_SCHEMA_VERSION,
                "treatment_id": TREATMENT_ID,
                "run_id": request.run_id,
                "failure_reason": safe,
                "cooperbench_evaluation_run": False,
            },
            error=safe,
        ).validated()
        print(safe, flush=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(result_path, outcome.to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
