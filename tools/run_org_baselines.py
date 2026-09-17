"""Run the B0/B1/B2/B3 OrgEnv conditions in contamination-isolated processes.

The shuffled-profile (``b4``) and random-profile (``b5``) control arms are
opt-in via ``--cases`` (e.g. ``--cases b3,b4,b5``); the default case set
remains the primary B0-B3 ladder so existing batch costs and the strict
paired-cell analysis contract are unchanged.

Each condition receives the same dataset, seed, OSS control, model surface, and
optional frozen evaluator budget. Cases never share a world, checkpoint, output
directory, or inherited ``ORG_*`` variable. A timed-out case may continue from
its own latest committed checkpoint only when the frozen case, source, model,
route, resource, seed, condition, and target-tick identities all match. Within
each pack/provider/model/seed block, case execution order is deterministically
randomized from the paired seed.

``--resume`` re-enters an existing ``--output-root`` batch: cases whose
``case_result.json`` reports ``"passed"`` are skipped, while failed or
interrupted cases use an identity-checked same-case checkpoint when available.
Once a same-plan checkpoint or checkpoint-backed resume has been observed, a
missing or invalid checkpoint fails closed instead of silently starting
another paid run from tick zero.
``--max-parallel N`` runs up to N case subprocesses
concurrently; cases remain fresh-process isolated, and the manifest records the
setting because concurrency changes execution-order semantics relative to the
paired-seed serial randomization.

Because every org_env cognitive module falls back to rule/template behaviour
on ``LLMError``, a provider outage degrades an ``--llm`` case instead of
aborting it. The runner therefore reads the produced record's ``llm_usage``:
a case whose every cognitive call failed (``failures >= calls > 0`` — the
LLM treatment was entirely absent, a derived boundary, not a tuned threshold)
fails with ``failure_reason: "llm_blackout"`` so ``--resume`` re-runs it, and
trips a batch-level provider circuit — cases not yet started are recorded
``"not_run"`` / ``"provider_circuit_open"`` instead of launching into the
outage (cases already running finish; at most ``--max-parallel`` of them).
Partial degradation (``0 < failures < calls``) stays "passed" but the usage
counters are surfaced in ``case_result.json`` for analysis-side judgment.
Every record aggregated into ``experiment_runs.json{,l}`` carries a
``batch_case`` annotation (case, condition_id, status, failure_reason when
present) so a degraded run can never be ingested as a healthy one; records
are annotated rather than filtered because paired-cell analysis needs the
whole block.

Examples:
    python tools/run_org_baselines.py --ticks 24 --dry-run
    python tools/run_org_baselines.py --ticks 336 --seed 42 --llm
    python tools/run_org_baselines.py --ticks 336 --seed 42 --llm \
        --output-root log/baselines/seed42 --resume --max-parallel 4
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import random
import re
import signal
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from environments.org_env.config.baseline_conditions import (  # noqa: E402
    ACTION_SELECTION_LLM_DIRECT,
    resolve_condition,
)
from environments.org_env.llm.client import REASONING_EFFORTS  # noqa: E402
from environments.org_env.llm.config import load_org_llm_config  # noqa: E402
from relic.research.hashing import stable_hash  # noqa: E402
from relic.research.process_compat import (  # noqa: E402
    new_process_group_kwargs,
    terminate_process_tree,
)
from relic.replay import load_trace  # noqa: E402
from relic.replay.source_export import (  # noqa: E402
    SourceTraceExportError,
    append_run_record_evidence,
    public_trace_path,
)

DEFAULT_CASES = ("b0", "b1", "b2", "b3")
# A direct source-runner invocation is one paired four-arm batch.  The
# release's thin 120-cell CLI supplies each of the ten canonical pack IDs; a
# small public pack makes the standalone dry-run command useful as well.
DEFAULT_DATASET = "mini_blobstore_v1"

# --evaluation-perturbation is DECLARATIVE here, and that is not an oversight.
#
# ORG_EXPERIMENT_EVALUATION_PERTURBATION has no consumer on the live path: the
# only readers are runtime_adapter/live.py (keeps it as world params metadata)
# and experiments/records.py (writes it to the record). Reading that as "the
# perturbation is never applied" is wrong, and the mistake is easy to make --
# it looks exactly like a treatment that was declared and forgotten.
#
# The two perturbations are EVALUATOR-side, applied in a second stage against
# the checkpoint this runner produces:
#
#   shuffled_capability_event_graph (NC3) -- the agent run executes normally;
#       tools/replay_capability_event_order.py then replays the capability
#       detector over a shuffled event order and writes a
#       capability_event_order_replay_v1 artifact. That artifact carries the
#       NC3 evidence, and organization_experiment_analysis._validate_event_replay
#       refuses it unless the shuffle is a genuine permutation of the observed
#       order (identity permutations are rejected outright).
#
#   shuffled_future_history (NC5) -- an evaluator_replay job, not an agent run
#       at all; see _future_shuffle_job in organization_experiment_execution.
#
# So the flag records which preregistered cell this run belongs to, while the
# perturbation itself lands on the paired evaluator job that the execution DAG
# schedules alongside it. Running this script standalone with a perturbation
# therefore yields only half of a negative control -- the unperturbed half.
CASE_IDS = {
    "b0": "b0_single_agent_founder",
    "b1": "b1_persistent_role_org",
    "b2": "b2_policy_conditioned_org",
    "b3": "b3_full_sociogenesis",
}


def _freeze_contamination_assessment(
    *,
    source: Path,
    expected_sha256: str,
    batch_root: Path,
) -> Path:
    lexical = Path(os.path.abspath(os.fspath(source.expanduser())))
    if lexical.is_symlink():
        raise ValueError("contamination_assessment_symlink_rejected")
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as exc:
        raise ValueError("contamination_assessment_missing") from exc
    if resolved != lexical or not resolved.is_file():
        raise ValueError("contamination_assessment_path_invalid")
    content = resolved.read_bytes()
    observed = hashlib.sha256(content).hexdigest()
    if observed != expected_sha256:
        raise ValueError("contamination_assessment_sha256_mismatch")
    try:
        payload = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("contamination_assessment_invalid_json") from exc
    if not isinstance(payload, dict):
        raise ValueError("contamination_assessment_must_be_object")
    destination = (
        batch_root
        / "frozen_inputs"
        / "contamination"
        / f"{observed}.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.is_symlink() or destination.read_bytes() != content:
            raise ValueError("contamination_frozen_input_collision")
    else:
        temporary = destination.with_name(
            f".{destination.name}.{os.getpid()}.tmp"
        )
        with temporary.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        os.chmod(destination, 0o444)
        # Windows does not allow opening a directory with ``os.open``.  The
        # file itself has already been flushed before the atomic replace; keep
        # the stronger parent-directory durability barrier on POSIX, where it
        # is supported, without making every Windows formal run fail before a
        # case can start.
        if os.name != "nt":
            directory = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    return destination.resolve()
_RUN_FOLDER_RE = re.compile(r"^run folder:\s*(.+?)\s*$", re.MULTILINE)
# Applies to EVERY LLM-requested case (formal and exploratory alike): a case
# whose cognitive success rate falls below this floor did not reliably receive
# the LLM treatment. Named without the FORMAL_ prefix it once carried, because
# its scope was never formal-only.
MIN_LLM_SUCCESS_RATE = 0.95

# These are the non-secret OpenAI runtime controls that affect a case's wire
# behaviour.  They belong in the frozen case identity.  Credentials, endpoint
# locators, and headers deliberately do not: the runner may consume those to
# launch the child, but must never serialize them into a plan or manifest.
_FROZEN_LLM_RUNTIME_ENV_KEYS = frozenset(
    {
        "ORG_LLM_JSON_TRANSPORT",
        "ORG_LLM_MAX_RETRIES",
        "ORG_LLM_REASONING_EFFORT",
        "ORG_LLM_REQUEST_TIMEOUT_SECONDS",
        "ORG_LLM_RETRY_BACKOFF_SECONDS",
        "ORG_LLM_STORE_RESPONSES",
        "ORG_LLM_WIRE_API",
    }
)


@dataclass(frozen=True)
class CasePlan:
    short_name: str
    condition_id: str
    action_selection_mode: str
    randomization_block: str
    randomization_order: int
    replication_id: str
    seed: int
    output_dir: Path
    command: tuple[str, ...]
    environment: Mapping[str, str]
    wall_clock_seconds: float = 0.0        # 0 = run to completion

    def public_environment(self) -> dict[str, str]:
        allowed = {
            "ORG_ACTION_SELECTION_MODE",
            "ORG_BASELINE_SPRINT_TICKS",
            "ORG_CASE_PLAN_FINGERPRINT",
            "ORG_CONTAMINATION_CLEARANCE",
            "ORG_CONTAMINATION_PROBES_PATH",
            "ORG_CONTAMINATION_STATUS",
            "ORG_EXPERIMENT_CONDITION",
            "ORG_EXECUTION_JOB_ID",
            "ORG_EXPERIMENT_MAX_LLM_CALLS",
            "ORG_EXPERIMENT_MAX_LLM_PROMPT_CHARACTERS",
            "ORG_EXPERIMENT_MAX_LLM_REQUESTED_TOKENS",
            "ORG_EXPERIMENT_MAX_PRIMARY_ACTIONS",
            "ORG_EXPERIMENT_MAX_TICKS",
            "ORG_EXPERIMENT_TARGET_TICK",
            "ORG_EXPERIMENT_PAIRED_SEED",
            "ORG_EXPERIMENT_PHASE",
            "ORG_EXPERIMENT_ARM_ID",
            "ORG_EXPERIMENT_EVALUATION_PERTURBATION",
            "ORG_EXPERIMENT_RANDOMIZATION_BLOCK",
            "ORG_EXPERIMENT_RANDOMIZATION_ORDER",
            "ORG_EXPERIMENT_REPLICATION_ID",
            "ORG_EXECUTION_PROFILE",
            "ORG_LLM",
            "ORG_LLM_ACTIONS",
            "ORG_LLM_MODEL",
            "ORG_LLM_MAX_RETRIES",
            "ORG_LLM_JSON_TRANSPORT",
            "ORG_LLM_PROVIDER",
            "ORG_LLM_REASONING_EFFORT",
            "ORG_LLM_REQUEST_TIMEOUT_SECONDS",
            "ORG_LLM_RETRY_BACKOFF_SECONDS",
            "ORG_LLM_STORE_RESPONSES",
            "ORG_LLM_WIRE_API",
            "ORG_LLM_EXPECTED_ENDPOINT_HASH",
            "ORG_LLM_EXPECTED_RESPONSE_MODEL",
            "ORG_LLM_EXPECTED_ROUTING_FINGERPRINT",
            "ORG_MODEL_BINDING_FINGERPRINT",
            "ORG_EXECUTION_RESOURCE_BUDGET_FINGERPRINT",
            "ORG_EVALUATOR_BACKEND",
            "ORG_EVALUATOR_CONTAINER_IMAGE",
            "ORG_EVALUATOR_CONTAINER_PLATFORM",
            "ORG_EVALUATOR_EXPECTED_ENVIRONMENT_HASH",
            "ORG_EVALUATOR_EXPECTED_QUALIFICATION_HASH",
            "ORG_EVALUATOR_VENV",
            "ORG_SOURCE_PROVENANCE_FINGERPRINT",
            "ORG_MECHANISM_ABLATIONS",
            "ORG_MODEL_CUTOFF_POLICY",
            "ORG_OSS_CONTROL",
            "ORG_OSS_DATASET",
            "ORG_OSS_REPOSITORY_ID",
            "ORG_TRANSFER_ARM",
            "ORG_TRANSFER_CAPABILITY_FORM",
            "ORG_TRANSFER_FIXED_PROTOCOL_LANDSCAPE",
            "ORG_TRANSFER_FROZEN_EPISODES",
            "ORG_TRANSFER_ROSTER_ORIGIN",
            "ORG_TRANSFER_SOURCE_REPOSITORY",
            "ORG_OSS_HIDDEN_TESTS",
            "ORG_OSS_MODE",
            "ORG_OSS_QUALIFICATION_TIMEOUT",
            "ORG_PRODUCT_PREWARM_SMOKE",
            "ORG_PRODUCT_SUBSTRATE",
            "ORG_RUN_TAG",
        }
        return {
            key: value
            for key, value in self.environment.items()
            if key in allowed
        }


def _parse_cases(raw: str) -> tuple[str, ...]:
    values = tuple(part.strip().lower() for part in raw.split(",") if part.strip())
    unknown = set(values) - set(CASE_IDS)
    if unknown:
        raise ValueError(f"unknown baseline cases: {sorted(unknown)}")
    if not values:
        raise ValueError("at least one baseline case is required")
    return values


def deterministic_case_order(
    cases: Sequence[str],
    *,
    seed: int,
    randomize: bool = True,
) -> tuple[str, ...]:
    """Return a seed-reproducible condition order without mutating ``cases``."""
    ordered = list(cases)
    if randomize:
        random.Random(int(seed)).shuffle(ordered)
    return tuple(ordered)


def build_randomization_block(
    *,
    pack: str,
    provider: str,
    model: str,
    seed: int,
) -> str:
    """Build the stable blocking identifier shared by all paired conditions."""
    return (
        f"pack={pack}|provider={provider}|model={model}|seed={int(seed)}"
    )


def _resolve_llm_identity(
    provider_override: str | None,
    model_override: str | None,
) -> tuple[str, str]:
    config = load_org_llm_config(str(REPO_ROOT))
    provider = str(
        provider_override or config.get("provider") or "unknown"
    ).lower()
    if provider == "generic":
        provider = "http"
    model = str(model_override or config.get("model") or "unknown")
    return provider, model


def _effective_llm_runtime_environment(
    parent: Mapping[str, str],
    *,
    config: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Freeze the effective, non-secret OpenAI runtime controls.

    ``load_org_llm_client`` accepts each of these controls from either an
    ``ORG_LLM_*`` override or ``config/llm{,.local}.yaml``.  Previously the
    baseline runner copied only overrides into the child/case plan.  The same
    effective treatment therefore acquired two different plan fingerprints
    depending on whether an operator spelled it in YAML or in the parent
    environment, and a paid checkpoint could not be resumed.

    Keep normalization aligned with ``environments.org_env.llm.config``.  The
    return value is an explicit allow-list, so ``api_key``, ``base_url``, and
    provider headers can never escape through this path.
    """

    cfg = dict(config if config is not None else load_org_llm_config(str(REPO_ROOT)))

    def boolean(value: Any, default: bool) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        normalized = str(value).strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
        return default

    def non_negative(value: Any, default: float) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return default
        return parsed if parsed >= 0 else default

    def number_text(value: float) -> str:
        # Canonicalize YAML numeric scalars and equivalent environment strings
        # to one identity (3, 3.0, and "3" all become "3").
        return str(int(value)) if value.is_integer() else str(value)

    effort = str(
        parent.get("ORG_LLM_REASONING_EFFORT")
        or cfg.get("reasoning_effort", "low")
    ).strip().lower()
    if effort not in REASONING_EFFORTS:
        effort = "low"

    max_retries = int(
        non_negative(
            parent.get("ORG_LLM_MAX_RETRIES") or cfg.get("max_retries"),
            5,
        )
    )
    backoff = non_negative(
        parent.get("ORG_LLM_RETRY_BACKOFF_SECONDS")
        or cfg.get("retry_backoff_seconds"),
        1.0,
    )
    request_timeout = non_negative(
        parent.get("ORG_LLM_REQUEST_TIMEOUT_SECONDS")
        or cfg.get("request_timeout_seconds"),
        120.0,
    )
    if request_timeout <= 0:
        request_timeout = 120.0

    wire_api = str(
        parent.get("ORG_LLM_WIRE_API")
        or cfg.get("wire_api")
        or "responses"
    ).strip().lower().replace("-", "_")
    json_transport = str(
        parent.get("ORG_LLM_JSON_TRANSPORT")
        or cfg.get("json_transport")
        or "native"
    ).strip().lower().replace("-", "_")

    store_raw = parent.get("ORG_LLM_STORE_RESPONSES")
    if store_raw is None:
        # The child environment intentionally does not inherit the global
        # OPENAI_DISABLE_RESPONSE_STORAGE knob.  With no OrgEnv setting the
        # client therefore takes its privacy-preserving default (store=False).
        store_responses = boolean(cfg.get("store_responses"), False)
    else:
        store_responses = boolean(store_raw, False)

    identity = {
        "ORG_LLM_JSON_TRANSPORT": json_transport,
        "ORG_LLM_MAX_RETRIES": str(max_retries),
        "ORG_LLM_REASONING_EFFORT": effort,
        "ORG_LLM_REQUEST_TIMEOUT_SECONDS": number_text(request_timeout),
        "ORG_LLM_RETRY_BACKOFF_SECONDS": number_text(backoff),
        "ORG_LLM_STORE_RESPONSES": "1" if store_responses else "0",
        "ORG_LLM_WIRE_API": wire_api,
    }
    assert identity.keys() == _FROZEN_LLM_RUNTIME_ENV_KEYS
    return identity


def build_case_environment(
    parent: Mapping[str, str],
    *,
    condition_id: str,
    dataset: str,
    seed: int,
    sprint_ticks: int,
    repository_id: str | None = None,
    llm: bool,
    llm_actions: bool,
    run_tag: str,
    llm_provider: str | None = None,
    llm_model: str | None = None,
    llm_runtime_environment: Mapping[str, str] | None = None,
    randomization_block: str | None = None,
    randomization_order: int | None = None,
    replication_id: str | None = None,
    max_llm_calls: int | None = None,
    max_llm_requested_tokens: int | None = None,
    max_llm_prompt_characters: int | None = None,
    max_primary_actions: int | None = None,
    max_ticks: int | None = None,
    target_tick: int | None = None,
    contamination_status: str | None = None,
    contamination_probes_path: str | None = None,
    contamination_clearance: bool | None = None,
    model_cutoff_policy: str | None = None,
    mechanism_ablations: str | None = None,
    oss_control: str = "none",
    experiment_phase: str | None = None,
    arm_id: str | None = None,
    evaluation_perturbation: str | None = None,
    experiment_mode: str = "formal",
    evaluator_backend: str | None = None,
    evaluator_container_image: str | None = None,
    evaluator_container_platform: str | None = None,
    evaluator_virtualenv: str | None = None,
    expected_evaluator_environment_hash: str | None = None,
    expected_qualification_plan_hash: str | None = None,
    qualification_timeout_seconds: int | None = None,
    execution_profile: str = "native",
) -> dict[str, str]:
    """Return a clean child environment with no inherited OrgEnv state."""
    from environments.org_env.experiments.ablations import MechanismAblations
    from environments.org_env.product.substrates.controls import CONTROLS

    condition = resolve_condition(condition_id)
    normalized_execution_profile = str(execution_profile or "native").strip()
    # ProgramBench is intentionally not in the Relic release closure.  Do not
    # silently reinterpret a request for it as a native B0--B3 run.
    if normalized_execution_profile != "native":
        raise ValueError(
            "programbench_execution_profile_not_available_in_relic_release"
        )
    if oss_control not in CONTROLS:
        raise ValueError(f"unknown OSS control: {oss_control!r}")
    if (
        qualification_timeout_seconds is not None
        and qualification_timeout_seconds <= 0
    ):
        raise ValueError("qualification timeout must be positive")
    normalized_ablations = MechanismAblations.from_values(
        (mechanism_ablations or "").split(",")
    )
    # Kept in the API for old callers. The condition now owns WHAT selection:
    # B0/B1/B2 force llm_direct when a client exists, independently of this flag.
    operating_system_keys = {
        "PATH",
        "HOME",
        # ``pathlib.Path.home()`` uses USERPROFILE on Windows.  HOME is the
        # POSIX spelling and is commonly absent from a native Windows parent;
        # dropping USERPROFILE therefore makes checkpoint metrics fail before
        # they can resolve their per-user cache paths.
        "USERPROFILE",
        "TMPDIR",
        "TEMP",
        "TMP",
        "LANG",
        "LC_ALL",
        "TERM",
        "SYSTEMROOT",
        "WINDIR",
        "SSL_CERT_FILE",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
        # Both spellings of every proxy variable. The list used to carry the
        # upper-case four and, of the lower-case four, only no_proxy — so an
        # operator who set https_proxy, which is the spelling Linux convention
        # and both curl and httpx use, had it dropped on the way into the case
        # while no_proxy survived. The case then dialled the API directly, and
        # on a machine that needs the proxy it sat in SYN-SENT until the run
        # timed out with nothing in the log to say why.
        "HTTP_PROXY",
        "http_proxy",
        "HTTPS_PROXY",
        "https_proxy",
        "ALL_PROXY",
        "all_proxy",
        "NO_PROXY",
        "no_proxy",
    }
    env = {
        key: value
        for key, value in parent.items()
        if key in operating_system_keys
    }
    provider_passthrough = {
        "ORG_EXECUTION_JOB_ID",
        "ORG_LLM_API_KEY",
        "ORG_LLM_BASE_URL",
        "ORG_LLM_DEFAULT_HEADERS_JSON",
        "ORG_LLM_MAX_RETRIES",
        "ORG_LLM_JSON_TRANSPORT",
        "ORG_LLM_REASONING_EFFORT",
        "ORG_LLM_REQUEST_TIMEOUT_SECONDS",
        "ORG_LLM_RETRY_BACKOFF_SECONDS",
        "ORG_LLM_STORE_RESPONSES",
        "ORG_LLM_WIRE_API",
        "ORG_LLM_EXPECTED_ENDPOINT_HASH",
        "ORG_LLM_EXPECTED_RESPONSE_MODEL",
        "ORG_LLM_EXPECTED_ROUTING_FINGERPRINT",
        "ORG_MODEL_BINDING_FINGERPRINT",
        "ORG_EXECUTION_RESOURCE_BUDGET_FINGERPRINT",
        "ORG_EVALUATOR_BACKEND",
        "ORG_EVALUATOR_CONTAINER_IMAGE",
        "ORG_EVALUATOR_CONTAINER_PLATFORM",
        "ORG_EVALUATOR_VENV",
        "ORG_EVALUATOR_EXPECTED_ENVIRONMENT_HASH",
        "ORG_EVALUATOR_EXPECTED_QUALIFICATION_HASH",
        "ORG_SOURCE_PROVENANCE_FINGERPRINT",
    }
    env.update(
        {
            key: value
            for key, value in parent.items()
            if key in provider_passthrough
        }
    )
    env.update(
        {
            "PYTHONPATH": str(REPO_ROOT),
            "ORG_ACTION_SELECTION_MODE": condition.action_selection_mode,
            "ORG_BASELINE_SPRINT_TICKS": str(max(1, int(sprint_ticks))),
            "ORG_EXPERIMENT_CONDITION": condition.condition_id,
            "ORG_EXPERIMENT_PAIRED_SEED": str(int(seed)),
            "ORG_LLM": "1" if llm else "0",
            "ORG_LLM_ACTIONS": (
                "1"
                if llm
                and condition.action_selection_mode == ACTION_SELECTION_LLM_DIRECT
                else "0"
            ),
            "ORG_LOG_ZIP": "0",
            "ORG_OSS_CONTROL": oss_control,
            "ORG_OSS_DATASET": dataset,
            # Identity, not locator. The dataset may be given as a path; the
            # record's `pack` is checked against the manifest project_id.
            "ORG_OSS_REPOSITORY_ID": repository_id or dataset,
            "ORG_OSS_HIDDEN_TESTS": "1",
            "ORG_OSS_MODE": experiment_mode,
            "ORG_PRODUCT_PREWARM_SMOKE": "1",
            "ORG_PRODUCT_SUBSTRATE": "oss_time_machine",
            "ORG_RUN_TAG": run_tag,
        }
    )
    if llm_model:
        env["ORG_LLM_MODEL"] = llm_model
    if llm_provider:
        env["ORG_LLM_PROVIDER"] = llm_provider
    if llm and str(llm_provider or "").lower() == "openai":
        runtime_environment = (
            dict(llm_runtime_environment)
            if llm_runtime_environment is not None
            else _effective_llm_runtime_environment(parent)
        )
        unexpected = set(runtime_environment) - _FROZEN_LLM_RUNTIME_ENV_KEYS
        if unexpected:
            raise ValueError(
                "unexpected frozen LLM runtime fields: "
                + ", ".join(sorted(unexpected))
            )
        env.update(runtime_environment)
    if contamination_status:
        env["ORG_CONTAMINATION_STATUS"] = contamination_status
    if contamination_probes_path:
        env["ORG_CONTAMINATION_PROBES_PATH"] = contamination_probes_path
    if contamination_clearance is not None:
        env["ORG_CONTAMINATION_CLEARANCE"] = json.dumps(
            contamination_clearance
        )
    if model_cutoff_policy:
        env["ORG_MODEL_CUTOFF_POLICY"] = model_cutoff_policy
    if normalized_ablations.disabled:
        env["ORG_MECHANISM_ABLATIONS"] = ",".join(
            sorted(normalized_ablations.disabled)
        )
    experiment_metadata = {
        "ORG_EXPERIMENT_PHASE": experiment_phase,
        "ORG_EXPERIMENT_ARM_ID": arm_id,
        "ORG_EXPERIMENT_EVALUATION_PERTURBATION": evaluation_perturbation,
        "ORG_EXPERIMENT_RANDOMIZATION_BLOCK": randomization_block,
        "ORG_EXPERIMENT_RANDOMIZATION_ORDER": randomization_order,
        "ORG_EXPERIMENT_REPLICATION_ID": replication_id,
        "ORG_EVALUATOR_BACKEND": evaluator_backend,
        "ORG_EVALUATOR_CONTAINER_IMAGE": evaluator_container_image,
        "ORG_EVALUATOR_CONTAINER_PLATFORM": evaluator_container_platform,
        "ORG_EVALUATOR_VENV": evaluator_virtualenv,
        "ORG_EVALUATOR_EXPECTED_ENVIRONMENT_HASH": (
            expected_evaluator_environment_hash
        ),
        "ORG_EVALUATOR_EXPECTED_QUALIFICATION_HASH": (
            expected_qualification_plan_hash
        ),
        "ORG_OSS_QUALIFICATION_TIMEOUT": qualification_timeout_seconds,
    }
    env.update(
        {
            key: str(value)
            for key, value in experiment_metadata.items()
            if value is not None
        }
    )
    # The source runner records the frozen ORG_* binding in its case plan and
    # formal-record gate.  Relic's curated evaluator consumes the matching
    # RELIC_* names.  Carry both values from one argument set rather than let
    # an ambient host binding select a different evaluator after planning.
    evaluator_bridge = {
        "RELIC_EVALUATOR_BACKEND": evaluator_backend,
        "RELIC_EVALUATOR_CONTAINER_IMAGE": evaluator_container_image,
        "RELIC_EVALUATOR_CONTAINER_PLATFORM": evaluator_container_platform,
        "RELIC_EVALUATOR_EXPECTED_ENVIRONMENT_HASH": (
            expected_evaluator_environment_hash
        ),
        "RELIC_EVALUATOR_EXPECTED_QUALIFICATION_HASH": (
            expected_qualification_plan_hash
        ),
    }
    env.update(
        {
            key: str(value)
            for key, value in evaluator_bridge.items()
            if value is not None
        }
    )
    frozen_limits = {
        "ORG_EXPERIMENT_MAX_LLM_CALLS": max_llm_calls,
        "ORG_EXPERIMENT_MAX_LLM_REQUESTED_TOKENS": max_llm_requested_tokens,
        "ORG_EXPERIMENT_MAX_LLM_PROMPT_CHARACTERS": max_llm_prompt_characters,
        "ORG_EXPERIMENT_MAX_PRIMARY_ACTIONS": max_primary_actions,
        "ORG_EXPERIMENT_MAX_TICKS": max_ticks,
        "ORG_EXPERIMENT_TARGET_TICK": target_tick,
    }
    env.update(
        {
            key: str(value)
            for key, value in frozen_limits.items()
            if value is not None
        }
    )
    return env


def build_case_plans(args: argparse.Namespace, batch_root: Path) -> tuple[CasePlan, ...]:
    provider = getattr(args, "provider", None)
    model = getattr(args, "model", None)
    if not provider or not model:
        configured_provider, configured_model = _resolve_llm_identity(
            provider,
            model,
        )
        provider = provider or configured_provider
        model = model or configured_model
    repository_id = getattr(args, "repository_id", None) or args.dataset
    # Read the YAML overlay once for the whole batch and freeze only its
    # effective non-secret wire controls.  All paired cases must see the same
    # treatment even if the local config changes while plans are being built.
    llm_runtime_environment: dict[str, str] = {}
    if args.llm and str(provider).lower() == "openai":
        llm_runtime_environment = _effective_llm_runtime_environment(
            os.environ,
            config=load_org_llm_config(str(REPO_ROOT)),
        )
    transfer_env = _transfer_environment(args)
    randomization_block = build_randomization_block(
        # The block names a repository, not a filesystem location: two jobs on
        # the same pack must land in the same block whether the DAG addressed
        # it by id or by path.
        pack=repository_id,
        provider=provider,
        model=model,
        seed=args.seed,
    )
    replication_id = (
        getattr(args, "replication_id", None) or f"seed-{int(args.seed)}"
    )
    case_order = deterministic_case_order(
        _parse_cases(args.cases),
        seed=args.seed,
        randomize=getattr(args, "randomize_order", True),
    )
    arm_map = dict(getattr(args, "arm_map", {}) or {})
    order_override = dict(
        getattr(args, "randomization_order_override", {}) or {}
    )
    plans = []
    for default_order, short_name in enumerate(case_order):
        randomization_order = int(
            order_override.get(short_name, default_order)
        )
        condition_id = CASE_IDS[short_name]
        action_selection_mode = resolve_condition(condition_id).action_selection_mode
        output_dir = batch_root / f"{short_name}_seed{args.seed}"
        run_name = f"baseline_{batch_root.name}_{short_name}"
        command = (
            sys.executable,
            str(REPO_ROOT / "tools" / "org_inspector_replay.py"),
            run_name,
            str(args.ticks),
            str(args.seed),
            "0",
            f"--checkpoint-every={args.checkpoint_every}",
            f"--target-tick={args.ticks}",
        )
        env = build_case_environment(
            os.environ,
            condition_id=condition_id,
            dataset=args.dataset,
            repository_id=repository_id,
            seed=args.seed,
            sprint_ticks=args.sprint_ticks,
            llm=args.llm,
            llm_actions=args.llm_actions,
            llm_provider=provider,
            llm_model=model,
            llm_runtime_environment=llm_runtime_environment,
            run_tag=f"{batch_root.name}_{short_name}_seed{args.seed}",
            randomization_block=randomization_block,
            randomization_order=randomization_order,
            replication_id=replication_id,
            max_llm_calls=args.max_llm_calls,
            max_llm_requested_tokens=args.max_llm_requested_tokens,
            max_llm_prompt_characters=args.max_llm_prompt_characters,
            max_primary_actions=args.max_primary_actions,
            max_ticks=args.max_ticks,
            target_tick=args.ticks,
            contamination_status=args.contamination_status,
            contamination_probes_path=(
                str(args.contamination_probes_path)
                if args.contamination_probes_path
                else None
            ),
            contamination_clearance=args.contamination_clearance,
            model_cutoff_policy=args.model_cutoff_policy,
            mechanism_ablations=args.mechanism_ablations,
            oss_control=args.oss_control,
            experiment_phase=getattr(args, "experiment_phase", None),
            arm_id=arm_map.get(short_name, short_name),
            evaluation_perturbation=getattr(
                args,
                "evaluation_perturbation",
                None,
            ),
            experiment_mode=(
                "pilot" if getattr(args, "pilot", False) else "formal"
            ),
            evaluator_backend=args.evaluator_backend,
            evaluator_container_image=args.evaluator_container_image,
            evaluator_container_platform=args.evaluator_container_platform,
            evaluator_virtualenv=(
                str(args.evaluator_venv.expanduser().resolve())
                if getattr(args, "evaluator_venv", None)
                else None
            ),
            expected_evaluator_environment_hash=(
                args.expected_evaluator_environment_hash
            ),
            expected_qualification_plan_hash=(
                args.expected_qualification_plan_hash
            ),
            qualification_timeout_seconds=getattr(
                args,
                "qualification_timeout_seconds",
                None,
            ),
            execution_profile=getattr(args, "execution_profile", "native"),
        )
        # A transfer arm's inheritance travels with the case, not the batch: the
        # world reads it while building, before tick one.
        env.update(transfer_env)
        plans.append(
            CasePlan(
                short_name=short_name,
                condition_id=condition_id,
                action_selection_mode=action_selection_mode,
                randomization_block=randomization_block,
                randomization_order=randomization_order,
                replication_id=replication_id,
                seed=args.seed,
                output_dir=output_dir,
                command=command,
                environment=env,
                wall_clock_seconds=float(
                    getattr(args, "case_wall_clock_seconds", 0.0)
                ),
            )
        )
    return tuple(plans)


def _case_resume_contract_payload(plan: CasePlan) -> dict[str, object]:
    """Immutable state identity required for a same-case continuation."""
    return {
        "schema_version": "orgenv_case_resume_contract_v1",
        "case": plan.short_name,
        "condition_id": plan.condition_id,
        "action_selection_mode": plan.action_selection_mode,
        "randomization_block": plan.randomization_block,
        "randomization_order": plan.randomization_order,
        "replication_id": plan.replication_id,
        "seed": plan.seed,
        "command": list(plan.command),
        "environment": plan.public_environment(),
        "target_tick": int(
            plan.environment.get("ORG_EXPERIMENT_TARGET_TICK", 0) or 0
        ),
    }


def _case_resume_contract_fingerprint(plan: CasePlan) -> str:
    return stable_hash(_case_resume_contract_payload(plan))


def _load_completed_case_result(plan: CasePlan) -> dict | None:
    """Return the prior result if this case already completed successfully.

    Only a readable ``case_result.json`` with ``status == "passed"`` counts;
    a failed, interrupted, or unreadable case is eligible for a strictly
    identity-bound same-case continuation.
    """
    result_path = plan.output_dir / "case_result.json"
    if not result_path.is_file():
        return None
    try:
        previous = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(previous, dict) or previous.get("status") != "passed":
        return None
    expected_plan = _case_manifest_payload(plan)
    if previous.get("case_plan_fingerprint") != stable_hash(expected_plan):
        return None
    if plan.environment.get("ORG_OSS_MODE") == "formal":
        record_path = _case_record_path(previous)
        if record_path is None:
            return None
        if _formal_case_record_errors(plan, record_path):
            return None
    if plan.environment.get("ORG_LLM") == "1":
        # Retroactive treatment-integrity check: a result graded "passed" before
        # the blackout/attachment gates existed must not be reused as completed
        # if today's gates would fail it. llm_usage is stored in the result
        # payload; fall back to the run record for pre-gate results without it.
        prior_usage = previous.get("llm_usage")
        if not isinstance(prior_usage, dict):
            prior_usage = None
            record_path = _case_record_path(previous)
            if record_path is not None:
                prior_usage = _load_llm_usage(record_path)
        if _llm_treatment_failure_reason(plan, prior_usage) is not None:
            return None
    return {**previous, "skipped_as_completed": True}


def _load_llm_usage(record_path: Path) -> dict | None:
    """Return the record's ``llm_usage`` mapping, or None when absent/unreadable.

    An unreadable or shapeless record is treated as carrying no usage — the
    blackout gate only ever acts on affirmative evidence of total LLM failure,
    never on missing data.
    """
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    usage = record.get("llm_usage") if isinstance(record, dict) else None
    return usage if isinstance(usage, dict) else None


def _is_llm_blackout(plan: CasePlan, llm_usage: dict | None) -> bool:
    """True when an LLM-requested case saw every cognitive call fail.

    ``failures >= calls > 0`` means not one LLM call succeeded across the whole
    run — the cognitive layer fell back to rule/template behaviour throughout,
    so the case factually did not receive the LLM treatment. This is the
    complete-absence boundary, not a tuned threshold; partial failure rates are
    surfaced in the result for analysis-side judgment instead of being gated.
    """
    if plan.environment.get("ORG_LLM") != "1" or llm_usage is None:
        return False
    try:
        calls = int(llm_usage.get("calls", 0) or 0)
        failures = int(llm_usage.get("failures", 0) or 0)
    except (TypeError, ValueError):
        return False
    return calls > 0 and failures >= calls


def _llm_treatment_failure_reason(
    plan: CasePlan,
    llm_usage: dict | None,
) -> str | None:
    """Return the preregistered treatment-integrity failure, if any."""

    if plan.environment.get("ORG_LLM") != "1":
        return None
    if llm_usage is None:
        # An LLM-requested case whose record carries no usage at all means the
        # cognitive client never attached (e.g. a config/load error swallowed at
        # session init) and the world ran rule/template-only. That is the same
        # treatment-not-delivered defect the blackout gate exists for, one layer
        # earlier — it must never grade as a healthy LLM arm.
        return "llm_client_not_attached"
    try:
        calls = int(llm_usage.get("calls", 0) or 0)
        failures = int(llm_usage.get("failures", 0) or 0)
        denials = int(llm_usage.get("resource_denials", 0) or 0)
    except (TypeError, ValueError):
        return "llm_usage_invalid"
    if denials:
        return "llm_resource_budget_denied"
    if calls <= 0:
        return "llm_treatment_has_no_calls"
    if failures >= calls:
        return "llm_blackout"
    success_rate = (calls - failures) / calls
    if success_rate < MIN_LLM_SUCCESS_RATE:
        return "llm_treatment_success_rate_below_0_95"
    return None


def _case_record_path(result: Mapping[str, object]) -> Path | None:
    raw = result.get("experiment_run_record")
    if raw:
        candidate = Path(str(raw))
        if not candidate.is_absolute():
            candidate = (REPO_ROOT / candidate).resolve()
        if candidate.is_file():
            return candidate
    run_folder = result.get("run_folder")
    if run_folder:
        candidate = (
            Path(str(run_folder)).expanduser().resolve()
            / "experiment_run_record.json"
        )
        if candidate.is_file():
            return candidate
    return None


def _case_public_trace_path(run_folder: str | None) -> tuple[str | None, str | None]:
    """Return a verified public trace locator, never a private replay locator."""

    if not run_folder:
        return None, None
    candidate = public_trace_path(Path(run_folder))
    if candidate.is_symlink() or not candidate.is_file():
        return None, "public_trace_missing"
    try:
        load_trace(candidate)
    except (OSError, ValueError):
        return None, "public_trace_invalid"
    try:
        return os.path.relpath(candidate.resolve(), REPO_ROOT), None
    except ValueError:
        return None, "public_trace_outside_repository"


def _formal_case_record_errors(
    plan: CasePlan,
    record_path: Path,
) -> tuple[str, ...]:
    """Validate that a formal case received the treatment in its frozen plan."""

    from environments.org_env.experiments.records import (
        assess_run_record_completeness,
        validate_experiment_run_record_schema,
    )

    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
        validate_experiment_run_record_schema(record)
    except (OSError, TypeError, ValueError) as exc:
        return (f"invalid_experiment_run_record:{exc}",)
    reasons: list[str] = []
    completeness = assess_run_record_completeness(record)
    if completeness["is_complete"] is not True:
        reasons.append(
            "incomplete_experiment_run_record:"
            + ",".join(completeness["missing_fields"])
        )
    expected_scalars = {
        "condition": plan.condition_id,
        "pack": plan.environment.get("ORG_OSS_REPOSITORY_ID"),
        "provider": plan.environment.get("ORG_LLM_PROVIDER"),
        "model": plan.environment.get("ORG_LLM_MODEL"),
        "seed": plan.seed,
        "experiment_phase": plan.environment.get("ORG_EXPERIMENT_PHASE"),
        "arm_id": plan.environment.get("ORG_EXPERIMENT_ARM_ID"),
        "evaluation_perturbation": plan.environment.get(
            "ORG_EXPERIMENT_EVALUATION_PERTURBATION"
        ),
        "replication_id": plan.replication_id,
        "randomization_block": plan.randomization_block,
        "randomization_order": plan.randomization_order,
        "status": "completed",
    }
    for field, expected in expected_scalars.items():
        if expected is not None and record.get(field) != expected:
            reasons.append(
                f"formal_record_mismatch:{field}:"
                f"expected={expected!r}:observed={record.get(field)!r}"
            )
    runtime = record.get("llm_runtime_identity") or {}
    expected_wire = plan.environment.get("ORG_LLM_WIRE_API")
    if expected_wire and runtime.get("wire_api") != expected_wire:
        reasons.append(
            "formal_record_mismatch:llm_runtime_identity.wire_api"
        )
    expected_json_transport = plan.environment.get("ORG_LLM_JSON_TRANSPORT")
    if (
        expected_json_transport
        and runtime.get("json_transport") != expected_json_transport
    ):
        reasons.append(
            "formal_record_mismatch:llm_runtime_identity.json_transport"
        )
    expected_runtime = {
        "endpoint_hash": plan.environment.get(
            "ORG_LLM_EXPECTED_ENDPOINT_HASH"
        ),
        "routing_context_fingerprint": plan.environment.get(
            "ORG_LLM_EXPECTED_ROUTING_FINGERPRINT"
        ),
    }
    for field, expected in expected_runtime.items():
        if expected and runtime.get(field) != expected:
            reasons.append(
                f"formal_record_mismatch:llm_runtime_identity.{field}"
            )
    expected_evaluator_environment = plan.environment.get(
        "ORG_EVALUATOR_EXPECTED_ENVIRONMENT_HASH"
    )
    if (
        expected_evaluator_environment
        and record.get("evaluator_environment_hash")
        != expected_evaluator_environment
    ):
        reasons.append(
            "formal_record_mismatch:evaluator_environment_hash"
        )
    expected_qualification = plan.environment.get(
        "ORG_EVALUATOR_EXPECTED_QUALIFICATION_HASH"
    )
    final_evaluation = record.get("final_evaluation") or {}
    if (
        expected_qualification
        and final_evaluation.get("plan_hash") != expected_qualification
    ):
        reasons.append("formal_record_mismatch:final_evaluation.plan_hash")
    expected_response_model = plan.environment.get(
        "ORG_LLM_EXPECTED_RESPONSE_MODEL"
    )
    if expected_response_model:
        observed = runtime.get("observed_response_models") or {}
        if set(observed) != {expected_response_model}:
            reasons.append(
                "formal_record_mismatch:"
                "llm_runtime_identity.observed_response_models"
            )
    provenance = record.get("provenance") or {}
    for field, environment_key in (
        ("execution_job_id", "ORG_EXECUTION_JOB_ID"),
        (
            "source_provenance_fingerprint",
            "ORG_SOURCE_PROVENANCE_FINGERPRINT",
        ),
        ("model_binding_fingerprint", "ORG_MODEL_BINDING_FINGERPRINT"),
        (
            "execution_resource_budget_fingerprint",
            "ORG_EXECUTION_RESOURCE_BUDGET_FINGERPRINT",
        ),
    ):
        expected = plan.environment.get(environment_key)
        if expected and provenance.get(field) != expected:
            reasons.append(f"formal_record_mismatch:provenance.{field}")
    if provenance.get("case_plan_fingerprint") != (
        _expected_case_plan_fingerprint(plan)
    ):
        reasons.append(
            "formal_record_mismatch:provenance.case_plan_fingerprint"
        )
    expected_target_tick = int(
        plan.environment.get("ORG_EXPERIMENT_TARGET_TICK", 0) or 0
    )
    if (
        expected_target_tick
        and provenance.get("target_tick") != expected_target_tick
    ):
        reasons.append("formal_record_mismatch:provenance.target_tick")
    usage = record.get("llm_usage")
    if plan.environment.get("ORG_LLM") == "1":
        if not isinstance(usage, Mapping):
            reasons.append("formal_llm_usage_missing")
        else:
            calls = int(usage.get("calls", 0) or 0)
            failures = int(usage.get("failures", 0) or 0)
            if calls <= 0:
                reasons.append("formal_llm_treatment_has_no_calls")
            elif failures >= calls:
                reasons.append("formal_llm_treatment_blackout")
            elif (
                (calls - failures) / calls
                < MIN_LLM_SUCCESS_RATE
            ):
                reasons.append(
                    "formal_llm_treatment_success_rate_below_0_95"
                )
            if int(usage.get("resource_denials", 0) or 0):
                reasons.append("formal_llm_resource_budget_denied")
    return tuple(reasons)


def _case_manifest_payload(plan: CasePlan) -> dict[str, object]:
    return {
        "case": plan.short_name,
        "condition_id": plan.condition_id,
        "action_selection_mode": plan.action_selection_mode,
        "randomization_block": plan.randomization_block,
        "randomization_order": plan.randomization_order,
        "replication_id": plan.replication_id,
        "seed": plan.seed,
        "command": list(plan.command),
        "environment": plan.public_environment(),
        "resume_allowed": True,
        "resume_scope": "same_case_identity_checked_checkpoint",
        "resume_contract_fingerprint": _case_resume_contract_fingerprint(plan),
        "state_isolation": "fresh_process",
        "wall_clock_seconds": plan.wall_clock_seconds,
    }


def _expected_case_plan_fingerprint(plan: CasePlan) -> str:
    return stable_hash(_case_manifest_payload(plan))


def _read_json_mapping(path: Path) -> Mapping[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, Mapping) else None


def _root_manifest_case_records(plan: CasePlan) -> tuple[Mapping[str, Any], ...]:
    """Return prior root-manifest entries for this case identity.

    The root manifest is written only after all case results have been
    collected.  It is therefore a durable secondary index when an interrupted
    resume truncates ``case_result.json`` or overwrites ``stdout.log``.  A case
    name alone is not enough: bind recovery to the same condition and seed.
    Full plan fingerprints are checked separately before a record is trusted.
    """

    manifest = _read_json_mapping(plan.output_dir.parent / "manifest.json")
    if manifest is None:
        return ()
    try:
        paired_seed = int(manifest.get("paired_seed", -1))
    except (TypeError, ValueError):
        return ()
    if paired_seed != plan.seed:
        return ()
    cases = manifest.get("cases")
    if not isinstance(cases, Sequence) or isinstance(cases, (str, bytes)):
        return ()
    records = []
    for candidate in cases:
        if not isinstance(candidate, Mapping):
            continue
        try:
            candidate_seed = int(candidate.get("seed", -1))
        except (TypeError, ValueError):
            continue
        if (
            candidate.get("case") == plan.short_name
            and candidate.get("condition_id") == plan.condition_id
            and candidate_seed == plan.seed
        ):
            records.append(candidate)
    return tuple(records)


def _prior_plan_records(plan: CasePlan) -> tuple[Mapping[str, Any], ...]:
    records: list[Mapping[str, Any]] = []
    case_plan = _read_json_mapping(plan.output_dir / "case_plan.json")
    if case_plan is not None:
        records.append(case_plan)
    records.extend(_root_manifest_case_records(plan))
    return tuple(records)


def _prior_record_matches_plan(
    plan: CasePlan,
    record: Mapping[str, Any],
) -> bool:
    return (
        record.get("case_plan_fingerprint")
        == _expected_case_plan_fingerprint(plan)
        and record.get("resume_contract_fingerprint")
        == _case_resume_contract_fingerprint(plan)
    )


def _prior_plan_matches(plan: CasePlan) -> bool:
    return any(
        _prior_record_matches_plan(plan, record)
        for record in _prior_plan_records(plan)
    )


def _has_prior_case_evidence(plan: CasePlan) -> bool:
    """Whether resume is re-entering observable state for this case."""

    if plan.output_dir.is_dir():
        try:
            next(plan.output_dir.iterdir())
        except (OSError, StopIteration):
            pass
        else:
            return True
    return bool(_root_manifest_case_records(plan))


def _resume_plan_mismatch(plan: CasePlan) -> bool:
    return _has_prior_case_evidence(plan) and not _prior_plan_matches(plan)


def _checkpoint_run_folder(checkpoint_path: object) -> Path | None:
    if not checkpoint_path:
        return None
    checkpoint = Path(str(checkpoint_path)).expanduser().resolve()
    if checkpoint.parent.name == "checkpoints":
        return checkpoint.parent.parent
    return checkpoint.parent


def _prior_run_folders(plan: CasePlan) -> tuple[Path, ...]:
    """Return every preserved run-folder reference for this case.

    A failed resume can print a new run folder and overwrite both
    ``case_result.json.run_folder`` and ``stdout.log``.  The checkpoint source
    from that resume is therefore carried forward explicitly and consulted
    before the latest child folder on the next attempt.

    Missing directories remain in the result: their presence is evidence that
    a checkpoint was expected, and must fail closed rather than authorize a
    fresh run.
    """

    candidates: list[Path] = []

    def add(raw: object) -> None:
        if raw:
            candidates.append(Path(str(raw)).expanduser().resolve())

    def add_from_record(record: Mapping[str, Any]) -> None:
        resume = record.get("resume")
        if isinstance(resume, Mapping):
            sources = resume.get("source_run_folders")
            if isinstance(sources, Sequence) and not isinstance(
                sources, (str, bytes)
            ):
                for source in sources:
                    add(source)
            checkpoint = resume.get("checkpoint")
            if isinstance(checkpoint, Mapping):
                source = _checkpoint_run_folder(checkpoint.get("path"))
                if source is not None:
                    candidates.append(source)
        add(record.get("run_folder"))

    # Prefer the completed root manifest.  A killed resume may have already
    # truncated the case-local result/index before it can replace this file.
    for record in _root_manifest_case_records(plan):
        if _prior_record_matches_plan(plan, record):
            add_from_record(record)

    result_path = plan.output_dir / "case_result.json"
    result = _read_json_mapping(result_path)
    if (
        result is not None
        and result.get("case_plan_fingerprint")
        == _expected_case_plan_fingerprint(plan)
    ):
        add_from_record(result)
    stdout_path = plan.output_dir / "stdout.log"
    if stdout_path.is_file():
        try:
            match = _RUN_FOLDER_RE.search(
                stdout_path.read_text(encoding="utf-8")
            )
        except OSError:
            match = None
        if match:
            add(match.group(1))
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        identity = os.path.normcase(str(candidate))
        if identity not in seen:
            seen.add(identity)
            unique.append(candidate)
    return tuple(unique)


def _prior_run_folder(plan: CasePlan) -> Path | None:
    """Compatibility wrapper returning the first extant prior run folder."""

    return next(
        (candidate for candidate in _prior_run_folders(plan) if candidate.is_dir()),
        None,
    )


def _resume_checkpoint_expected(plan: CasePlan) -> bool:
    """Whether prior evidence requires resume to find a valid checkpoint."""

    result_path = plan.output_dir / "case_result.json"
    if result_path.is_file():
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            result = None
        if isinstance(result, Mapping):
            resume = result.get("resume")
            if isinstance(resume, Mapping) and (
                resume.get("used_checkpoint") is True
                or resume.get("checkpoint_required") is True
                or isinstance(resume.get("checkpoint"), Mapping)
            ):
                return True
    for run_folder in _prior_run_folders(plan):
        if any((run_folder / "checkpoints").glob("*.pkl")):
            return True
        if any(run_folder.glob("checkpoint_t*.pkl")):
            return True
    return False


def _resume_checkpoint_errors(
    plan: CasePlan,
    info: Mapping[str, object],
) -> tuple[str, ...]:
    expected_target = int(
        plan.environment.get("ORG_EXPERIMENT_TARGET_TICK", 0) or 0
    )
    expected = {
        "case_plan_fingerprint": _expected_case_plan_fingerprint(plan),
        "seed": plan.seed,
        "experiment_condition": plan.condition_id,
        "action_selection_mode": plan.action_selection_mode,
        "source_provenance_fingerprint": plan.environment.get(
            "ORG_SOURCE_PROVENANCE_FINGERPRINT"
        ),
        "model_binding_fingerprint": plan.environment.get(
            "ORG_MODEL_BINDING_FINGERPRINT"
        ),
        "resource_budget_fingerprint": plan.environment.get(
            "ORG_EXECUTION_RESOURCE_BUDGET_FINGERPRINT"
        ),
        "target_tick": expected_target or None,
    }
    errors = [
        f"{field}_mismatch"
        for field, value in expected.items()
        if value is not None and info.get(field) != value
    ]
    try:
        tick = int(info.get("tick", -1))
    except (TypeError, ValueError):
        tick = -1
    if tick <= 0:
        errors.append("checkpoint_tick_invalid")
    if expected_target and tick > expected_target:
        errors.append("checkpoint_tick_after_target")
    return tuple(errors)


def _latest_resume_checkpoint(plan: CasePlan) -> dict[str, object] | None:
    """Return the latest valid committed checkpoint for this exact case plan."""
    if not _prior_plan_matches(plan):
        return None
    run_folders = _prior_run_folders(plan)
    if not run_folders:
        return None
    from environments.org_env.runtime_adapter.checkpoint import checkpoint_info

    eligible: list[dict[str, object]] = []
    for source_priority, run_folder in enumerate(run_folders):
        candidates = list((run_folder / "checkpoints").glob("*.pkl"))
        candidates.extend(run_folder.glob("checkpoint_t*.pkl"))
        for path in candidates:
            try:
                info = checkpoint_info(str(path))
            except (OSError, TypeError, ValueError):
                continue
            if not isinstance(info, Mapping):
                continue
            errors = _resume_checkpoint_errors(plan, info)
            if errors:
                continue
            eligible.append(
                {
                    "path": str(path.resolve()),
                    "tick": int(info["tick"]),
                    "created": info.get("created"),
                    "run_folder": str(run_folder),
                    "source_priority": source_priority,
                    "case_plan_fingerprint": info.get(
                        "case_plan_fingerprint"
                    ),
                }
            )
    return (
        max(
            eligible,
            key=lambda item: (
                int(item["tick"]),
                -int(item["source_priority"]),
                str(item["path"]),
            ),
        )
        if eligible
        else None
    )


def _invocation_plan(
    plan: CasePlan,
    checkpoint: Mapping[str, object] | None,
) -> CasePlan:
    environment = dict(plan.environment)
    environment["ORG_CASE_PLAN_FINGERPRINT"] = (
        _expected_case_plan_fingerprint(plan)
    )
    if checkpoint is None:
        return replace(plan, environment=environment)
    command = list(plan.command)
    if (
        len(command) < 6
        or Path(command[1]).name != "org_inspector_replay.py"
    ):
        raise ValueError("checkpoint resume requires org_inspector_replay")
    target = int(
        plan.environment.get("ORG_EXPERIMENT_TARGET_TICK", 0) or 0
    )
    tick = int(checkpoint["tick"])
    if not target or tick > target:
        raise ValueError("checkpoint resume target is invalid")
    command[3] = str(target - tick)
    command.append(f"--from-checkpoint={checkpoint['path']}")
    return replace(
        plan,
        command=tuple(command),
        environment=environment,
    )


def _timeout_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _child_failure_evidence(
    *,
    returncode: int | None,
    stderr: str,
    experiment_run_record: str | None,
) -> dict[str, object] | None:
    """Return a redacted receipt for a child that died before its record.

    ``stderr.log`` remains the private diagnostic artifact.  The public case
    result carries only an allow-listed category and hashes, so an exception
    message containing a token, source payload, or other secret is never
    copied into ``case_result.json``.
    """

    if returncode in (None, 0) or experiment_run_record is not None:
        return None
    category = "nonzero_exit"
    if "unknown_formal_event_graph_edge:" in stderr:
        category = "unknown_formal_event_graph_edge"
    elif "Traceback (most recent call last):" in stderr:
        category = "python_exception"
    stderr_bytes = stderr.encode("utf-8")
    evidence: dict[str, object] = {
        "category": category,
        "stderr_log": "stderr.log",
        "stderr_sha256": hashlib.sha256(stderr_bytes).hexdigest(),
    }
    last_nonempty_line = next(
        (line for line in reversed(stderr.splitlines()) if line.strip()),
        None,
    )
    if last_nonempty_line is not None:
        evidence["stderr_last_nonempty_line_sha256"] = hashlib.sha256(
            last_nonempty_line.encode("utf-8")
        ).hexdigest()
    return evidence


def _run_case_process(plan: CasePlan) -> subprocess.CompletedProcess[str]:
    process = subprocess.Popen(
        plan.command,
        cwd=REPO_ROOT,
        env=dict(plan.environment),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        **new_process_group_kwargs(),
    )
    previous_handlers = {}

    def terminate_group() -> None:
        terminate_process_tree(process)

    def forward_signal(signum, _frame) -> None:
        terminate_group()
        raise SystemExit(128 + int(signum))

    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, forward_signal)
    deadline = plan.wall_clock_seconds if plan.wall_clock_seconds > 0 else None
    try:
        stdout, stderr = process.communicate(timeout=deadline)
    except subprocess.TimeoutExpired as exc:
        terminate_group()
        stdout, stderr = process.communicate()
        raise subprocess.TimeoutExpired(
            cmd=plan.command,
            timeout=plan.wall_clock_seconds,
            output=stdout or exc.output,
            stderr=stderr or exc.stderr,
        ) from exc
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    return subprocess.CompletedProcess(
        args=plan.command,
        returncode=process.returncode,
        stdout=stdout,
        stderr=stderr,
    )


def _run_case(
    plan: CasePlan,
    *,
    dry_run: bool,
    resume: bool = False,
    circuit: threading.Event | None = None,
) -> dict:
    resume_checkpoint = None
    resume_sources: tuple[Path, ...] = ()
    prior_plan_matches = False
    resume_checkpoint_expected = False
    if resume:
        # Never turn ``--resume`` into an implicit fresh paid run.  In
        # particular, do this before mkdir/write_text so a mismatched attempt
        # cannot destroy the plan evidence needed to diagnose or recover it.
        if _resume_plan_mismatch(plan):
            return {
                "case": plan.short_name,
                "condition_id": plan.condition_id,
                "status": "failed",
                "failure_reason": "resume_plan_mismatch",
                "returncode": None,
                "run_folder": None,
                "experiment_run_record": None,
                "llm_usage": None,
                "resume": {
                    "used_checkpoint": False,
                    "checkpoint": None,
                    "remaining_ticks": None,
                    "source_run_folders": [],
                },
            }
        previous = _load_completed_case_result(plan)
        if previous is not None:
            return previous
        prior_plan_matches = _prior_plan_matches(plan)
        if prior_plan_matches:
            resume_sources = _prior_run_folders(plan)
            resume_checkpoint_expected = _resume_checkpoint_expected(plan)
        resume_checkpoint = _latest_resume_checkpoint(plan)
    plan.output_dir.mkdir(parents=True, exist_ok=resume)
    case_payload = _case_manifest_payload(plan)
    case_manifest = {
        **case_payload,
        "case_plan_fingerprint": _expected_case_plan_fingerprint(plan),
    }
    if circuit is not None and circuit.is_set():
        # A circuit-skipped case still needs a frozen plan.  Otherwise a later
        # --resume cannot distinguish the intended pending cell from unrelated
        # artifacts and must (correctly) reject it as a mismatch.
        (plan.output_dir / "case_plan.json").write_text(
            json.dumps(case_manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        result = {
            **case_manifest,
            "status": "not_run",
            "failure_reason": "provider_circuit_open",
            "returncode": None,
            "run_folder": (
                str(resume_sources[0]) if resume_sources else None
            ),
            "resume": {
                "used_checkpoint": False,
                "checkpoint": None,
                "remaining_ticks": None,
                "source_run_folders": [
                    str(source) for source in resume_sources
                ],
            },
        }
        (plan.output_dir / "case_result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return result
    (plan.output_dir / "case_plan.json").write_text(
        json.dumps(case_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if dry_run:
        return {**case_manifest, "status": "dry_run", "returncode": None}

    if (
        resume
        and prior_plan_matches
        and resume_checkpoint_expected
        and resume_checkpoint is None
    ):
        result = {
            **case_manifest,
            "status": "failed",
            "failure_reason": "resume_checkpoint_unavailable",
            "returncode": None,
            "run_folder": (
                str(resume_sources[0]) if resume_sources else None
            ),
            "experiment_run_record": None,
            "llm_usage": None,
            "resume": {
                "used_checkpoint": False,
                "checkpoint": None,
                "remaining_ticks": None,
                "checkpoint_required": True,
                "source_run_folders": [
                    str(source) for source in resume_sources
                ],
            },
        }
        (plan.output_dir / "case_result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return result

    invocation = _invocation_plan(plan, resume_checkpoint)
    timed_out = False
    try:
        completed = _run_case_process(invocation)
        stdout = completed.stdout
        stderr = completed.stderr
        returncode = completed.returncode
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        stdout = _timeout_output(exc.stdout)
        stderr = _timeout_output(exc.stderr)
        returncode = None
    (plan.output_dir / "stdout.log").write_text(stdout, encoding="utf-8")
    (plan.output_dir / "stderr.log").write_text(stderr, encoding="utf-8")
    match = _RUN_FOLDER_RE.search(stdout)
    run_folder = match.group(1) if match else None
    source_run_folders = [str(source) for source in resume_sources]
    if resume_checkpoint is not None:
        checkpoint_source = str(resume_checkpoint["run_folder"])
        if checkpoint_source not in source_run_folders:
            source_run_folders.append(checkpoint_source)
    if run_folder:
        observed_source = str(Path(run_folder).expanduser().resolve())
        if observed_source not in source_run_folders:
            source_run_folders.append(observed_source)
    experiment_run_record = None
    llm_usage = None
    if run_folder:
        candidate_record = Path(run_folder) / "experiment_run_record.json"
        if candidate_record.is_file():
            experiment_run_record = os.path.relpath(
                candidate_record,
                REPO_ROOT,
            )
            llm_usage = _load_llm_usage(candidate_record)
    public_trace, public_trace_error = _case_public_trace_path(run_folder)
    child_failure = _child_failure_evidence(
        returncode=returncode,
        stderr=stderr,
        experiment_run_record=experiment_run_record,
    )
    treatment_failure = _llm_treatment_failure_reason(plan, llm_usage)
    formal_record_errors = (
        _formal_case_record_errors(plan, candidate_record)
        if plan.environment.get("ORG_OSS_MODE") == "formal"
        and experiment_run_record is not None
        else ()
    )
    # Only a provider-outage-shaped failure (complete blackout) should halt the
    # whole batch: other treatment failures (budget denial, low success rate,
    # client not attached) are case-local and must not record healthy pending
    # cases as not_run/provider_circuit_open.
    if treatment_failure == "llm_blackout" and circuit is not None:
        circuit.set()
    result = {
        **case_manifest,
        "status": (
            "passed"
            if returncode == 0
            and experiment_run_record is not None
            and public_trace is not None
            and treatment_failure is None
            and not formal_record_errors
            else ("timeout" if timed_out else "failed")
        ),
        "returncode": returncode,
        "run_folder": run_folder,
        "experiment_run_record": experiment_run_record,
        "public_trace": public_trace,
        "llm_usage": llm_usage,
        "resume": {
            "used_checkpoint": resume_checkpoint is not None,
            "checkpoint": (
                {
                    "path": resume_checkpoint["path"],
                    "tick": resume_checkpoint["tick"],
                    "created": resume_checkpoint.get("created"),
                    "case_plan_fingerprint": resume_checkpoint.get(
                        "case_plan_fingerprint"
                    ),
                }
                if resume_checkpoint is not None
                else None
            ),
            "remaining_ticks": (
                int(invocation.command[3])
                if resume_checkpoint is not None
                else None
            ),
            "source_run_folders": source_run_folders,
        },
    }
    if child_failure is not None:
        # A non-zero child exit before a record exists is the primary failure.
        # Missing usage is only a consequence in this shape, not evidence that
        # the LLM client failed to attach.
        result["failure_reason"] = "child_process_failed_before_record"
        result["child_failure"] = child_failure
    elif treatment_failure:
        result["failure_reason"] = treatment_failure
    elif formal_record_errors:
        result["failure_reason"] = (
            "formal_record_validation_failed:"
            + "|".join(formal_record_errors)
        )
    elif public_trace_error and returncode == 0 and experiment_run_record is not None:
        result["failure_reason"] = public_trace_error
    elif timed_out:
        result["failure_reason"] = "case_wall_clock_timeout"
    (plan.output_dir / "case_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    trace_update_error = _stamp_verdict_on_run_record(result, run_folder)
    if trace_update_error is not None:
        result["status"] = "failed"
        result["failure_reason"] = trace_update_error
        (plan.output_dir / "case_result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return result


def _stamp_verdict_on_run_record(
    result: Mapping[str, Any], run_folder: str | None
) -> str | None:
    """Write the batch's verdict back into the run's own record.

    The record is written by the child process before this verdict is computed,
    so a case judged failed still described itself as completed. Two pilot runs
    rejected for `llm_treatment_success_rate_below_0_95` and
    `llm_resource_budget_denied` both carried `status: "completed"` in their run
    folders, and anything that collects records by walking the filesystem —
    rglob over experiment_run_record.json is one such path — would have taken
    them for healthy data. The portfolio layer recomputes LLM completeness and
    would catch those two, but only those two: a formal_record_validation_failed
    case has no second line of defence.
    """
    if not run_folder or str(result.get("status")) == "passed":
        return None
    record_path = Path(run_folder) / "experiment_run_record.json"
    if not record_path.is_file():
        return None
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "public_trace_verdict_record_unreadable"
    record["status"] = str(result.get("status") or "failed")
    reason = result.get("failure_reason")
    if reason:
        # Keep the child's own reason when it had one; the batch verdict is an
        # additional judgement, not a replacement for what the run observed.
        existing = record.get("failure_reason")
        record["failure_reason"] = (
            f"{existing}|{reason}" if existing and existing != reason else reason
        )
    try:
        record_path.write_text(
            json.dumps(record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError:
        return "public_trace_verdict_record_write_failed"
    trace_path = public_trace_path(Path(run_folder))
    try:
        append_run_record_evidence(trace_path, record)
    except SourceTraceExportError:
        return "public_trace_verdict_append_failed"
    return None


def _transfer_source_bundle(args: argparse.Namespace) -> str:
    """Locate the source formation run's capability bundle, or explain why not.

    Every arm inherits from one source run, so the bundle is a hard requirement:
    an arm that started with nothing inherited would be a fifth copy of the
    reference wearing another arm's label, and its contrast would be noise
    carrying the name of an effect.
    """
    explicit = getattr(args, "transfer_source_bundle", None)
    if explicit:
        if not Path(explicit).is_file():
            raise SystemExit(f"transfer_source_bundle_missing:{explicit}")
        return str(Path(explicit).resolve())
    job = getattr(args, "transfer_source_bundle_from_job", None)
    if not job:
        raise SystemExit(
            "a transfer arm needs --transfer-source-bundle or "
            "--transfer-source-bundle-from-job"
        )
    root = getattr(args, "output_root", None)
    candidates = (
        sorted(Path(root).parent.glob(f"jobs/{job}/**/capability_bundle.json"))
        if root
        else []
    )
    candidates += sorted(REPO_ROOT.glob("log/*/capability_bundle.json"))
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())
    raise SystemExit(f"transfer_source_bundle_not_found_for_job:{job}")


def _transfer_environment(args: argparse.Namespace) -> dict[str, str]:
    """Env the target world reads at build time to inherit its arm's state."""
    if not getattr(args, "transfer_arm", None):
        return {}
    return {
        "ORG_TRANSFER_ARM": str(args.transfer_arm),
        "ORG_TRANSFER_SOURCE_REPOSITORY": str(
            getattr(args, "transfer_source_repository", "") or ""
        ),
        "ORG_TRANSFER_ROSTER_ORIGIN": str(
            getattr(args, "transfer_roster_origin", "") or ""
        ),
        "ORG_TRANSFER_CAPABILITY_FORM": str(
            getattr(args, "transfer_capability_form", "") or ""
        ),
        "ORG_TRANSFER_SOURCE_BUNDLE": _transfer_source_bundle(args),
        "ORG_TRANSFER_FROZEN_EPISODES": str(
            getattr(args, "freeze_capability_compilation_episodes", 0) or 0
        ),
        "ORG_TRANSFER_FIXED_PROTOCOL_LANDSCAPE": (
            "1" if getattr(args, "fixed_protocol_landscape", False) else "0"
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default=",".join(DEFAULT_CASES))
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument(
        "--repository-id",
        help=(
            "the repository's declared id, when --dataset addresses it by path. "
            "The record's `pack` is cross-checked against the manifest's "
            "project_id, so a path there fails every record; defaults to "
            "--dataset, which is already the id for pack-tree repositories."
        ),
    )
    # A capability_transfer job carries the arm's inheritance: which source run
    # it draws from, whether the roster is retained, and in what form the
    # capabilities arrive. The target world reads these at build time; see
    # experiments/capability_transfer.py.
    parser.add_argument("--transfer-arm")
    parser.add_argument("--transfer-source-repository")
    parser.add_argument("--transfer-roster-origin")
    parser.add_argument("--transfer-capability-form")
    parser.add_argument("--transfer-source-bundle-from-job")
    parser.add_argument(
        "--transfer-source-bundle",
        help="path to the source formation run's capability_bundle.json",
    )
    parser.add_argument(
        "--freeze-capability-compilation-episodes", type=int, default=0
    )
    parser.add_argument(
        "--fixed-protocol-landscape",
        action="store_true",
        help=(
            "after transfer injection, disable endogenous protocol creation, "
            "adoption, amendment, and retirement for the full run"
        ),
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help=(
            "run a non-formal single-pack pilot; required for the Bubblewrap "
            "venv backend, whose results cannot satisfy the container gate"
        ),
    )
    parser.add_argument(
        "--evaluator-backend",
        choices=("docker", "apptainer", "bubblewrap"),
    )
    parser.add_argument("--evaluator-container-image")
    parser.add_argument("--evaluator-container-platform")
    parser.add_argument("--evaluator-venv", type=Path)
    parser.add_argument("--expected-evaluator-environment-hash")
    parser.add_argument("--expected-qualification-plan-hash")
    parser.add_argument(
        "--qualification-timeout-seconds",
        type=int,
        default=900,
        help=(
            "frozen timeout for each formal evaluator qualification pass; "
            "defaults to 900 seconds, matching print_evaluator_hashes.py"
        ),
    )
    parser.add_argument("--ticks", type=int, default=336)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sprint-ticks", type=int, default=168)
    parser.add_argument("--checkpoint-every", type=int, default=24)
    parser.add_argument("--llm", action="store_true")
    parser.add_argument(
        "--llm-actions",
        action="store_true",
        help="deprecated compatibility flag; baseline conditions force LLM-direct actions",
    )
    parser.add_argument("--model", default=os.environ.get("ORG_LLM_MODEL"))
    parser.add_argument("--provider", default=os.environ.get("ORG_LLM_PROVIDER"))
    parser.add_argument("--replication-id")
    parser.add_argument(
        "--no-randomize-order",
        dest="randomize_order",
        action="store_false",
        help="preserve --cases order instead of applying paired-seed randomization",
    )
    parser.set_defaults(randomize_order=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "re-enter an existing --output-root batch: skip cases whose "
            "case_result.json is 'passed', continue failed or interrupted "
            "cases from an identity-checked same-case checkpoint when "
            "available, otherwise restart"
        ),
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=1,
        help=(
            "run up to N case subprocesses concurrently (default 1, serial); "
            "cases are fresh-process isolated either way"
        ),
    )
    parser.add_argument("--max-llm-calls", type=int)
    parser.add_argument("--max-llm-requested-tokens", type=int)
    parser.add_argument("--max-llm-prompt-characters", type=int)
    parser.add_argument("--max-primary-actions", type=int)
    parser.add_argument("--max-ticks", type=int)
    parser.add_argument(
        "--case-wall-clock-seconds",
        type=float,
        default=0.0,
        help=(
            "wall-clock ceiling for each isolated condition process; 0 (the "
            "default) runs to completion. Only the tick/action/call/token "
            "ceilings bound the experiment — wall clock varies with provider "
            "latency, so a ceiling here truncates slow evenings and keeps fast "
            "ones. Set a positive value only to guard against a wedged process."
        ),
    )
    parser.add_argument("--model-cutoff-policy")
    parser.add_argument("--contamination-status")
    parser.add_argument(
        "--mechanism-ablations",
        default="",
        help="comma-separated registered mechanisms disabled for this arm",
    )
    parser.add_argument("--oss-control", default="none")
    parser.add_argument("--experiment-phase")
    parser.add_argument(
        "--execution-profile",
        default="native",
        help="must remain native; ProgramBench is not part of this release",
    )
    parser.add_argument(
        "--arm-map",
        default="{}",
        help="JSON object mapping each short condition name to its arm id",
    )
    parser.add_argument(
        "--randomization-order-override",
        default="{}",
        help=(
            "JSON object mapping short condition names to preregistered "
            "cross-job order ranks"
        ),
    )
    parser.add_argument("--evaluation-perturbation", default="none")
    parser.add_argument("--contamination-probes-path", type=Path)
    parser.add_argument(
        "--expected-contamination-assessment-sha256",
    )
    parser.add_argument(
        "--contamination-clearance",
        choices=("true", "false"),
        help="curator result; omit until contamination probes have been reviewed",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    args.contamination_clearance = (
        None
        if args.contamination_clearance is None
        else args.contamination_clearance == "true"
    )
    try:
        args.arm_map = json.loads(args.arm_map)
    except json.JSONDecodeError as exc:
        raise SystemExit("--arm-map must be valid JSON") from exc
    if not isinstance(args.arm_map, dict) or not all(
        isinstance(key, str) and isinstance(value, str) and value
        for key, value in args.arm_map.items()
    ):
        raise SystemExit("--arm-map must be a JSON object of non-empty strings")
    try:
        args.randomization_order_override = json.loads(
            args.randomization_order_override
        )
    except json.JSONDecodeError as exc:
        raise SystemExit(
            "--randomization-order-override must be valid JSON"
        ) from exc
    if not isinstance(args.randomization_order_override, dict) or not all(
        isinstance(key, str)
        and isinstance(value, int)
        and not isinstance(value, bool)
        and value >= 0
        for key, value in args.randomization_order_override.items()
    ):
        raise SystemExit(
            "--randomization-order-override must map strings to "
            "non-negative integers"
        )
    if args.llm_actions and not args.llm:
        raise SystemExit("--llm-actions requires --llm")
    if not args.dry_run and not args.llm:
        raise SystemExit(
            "formal B0-B3 runs require --llm; B3 retains profile_policy action "
            "selection but still uses LLM member cognition/content"
        )
    if args.ticks <= 0 or args.checkpoint_every <= 0 or args.sprint_ticks <= 0:
        raise SystemExit("ticks, checkpoint-every, and sprint-ticks must be positive")
    if (
        args.qualification_timeout_seconds is not None
        and args.qualification_timeout_seconds <= 0
    ):
        raise SystemExit("--qualification-timeout-seconds must be positive")
    if str(args.execution_profile or "native").strip() != "native":
        raise SystemExit(
            "programbench_execution_profile_not_available_in_relic_release"
        )
    if args.max_parallel < 1:
        raise SystemExit("--max-parallel must be a positive integer")
    if args.case_wall_clock_seconds < 0:
        raise SystemExit("--case-wall-clock-seconds must not be negative")
    if args.resume and args.output_root is None:
        raise SystemExit(
            "--resume requires --output-root pointing at the batch to re-enter "
            "(default batch roots are timestamped and never collide)"
        )
    if args.max_ticks is None:
        args.max_ticks = args.ticks
    if args.pilot and args.evaluator_backend != "bubblewrap":
        raise SystemExit("--pilot requires --evaluator-backend bubblewrap")
    if not args.pilot and args.evaluator_backend == "bubblewrap":
        raise SystemExit("bubblewrap evaluator requires --pilot")
    if not args.dry_run:
        required_limits = {
            "--max-llm-calls": args.max_llm_calls,
            "--max-llm-requested-tokens": args.max_llm_requested_tokens,
            "--max-llm-prompt-characters": args.max_llm_prompt_characters,
            "--max-primary-actions": args.max_primary_actions,
        }
        missing_limits = [
            flag for flag, value in required_limits.items() if value is None
        ]
        if missing_limits:
            raise SystemExit(
                "formal B0-B3 runs require frozen resource ceilings: "
                + ", ".join(missing_limits)
            )
        required_evaluator = (
            {
                "--evaluator-backend": args.evaluator_backend,
                "--evaluator-venv": args.evaluator_venv,
            }
            if args.pilot
            else {
                "--evaluator-backend": args.evaluator_backend,
                "--evaluator-container-image": args.evaluator_container_image,
                "--evaluator-container-platform": args.evaluator_container_platform,
                "--expected-evaluator-environment-hash": (
                    args.expected_evaluator_environment_hash
                ),
                "--expected-qualification-plan-hash": (
                    args.expected_qualification_plan_hash
                ),
            }
        )
        missing_evaluator = [
            flag for flag, value in required_evaluator.items() if not value
        ]
        if missing_evaluator:
            raise SystemExit(
                (
                    "pilot runs require a Bubblewrap virtualenv: "
                    if args.pilot
                    else "formal runs require a frozen evaluator runtime: "
                )
                + ", ".join(missing_evaluator)
            )

    args.provider, args.model = _resolve_llm_identity(
        args.provider,
        args.model,
    )
    batch_id = dt.datetime.now().strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}"
    batch_root = (
        args.output_root.expanduser().resolve()
        if args.output_root
        else REPO_ROOT / "log" / "baselines" / batch_id
    )
    batch_root.mkdir(parents=True, exist_ok=args.resume)
    contamination_source_path = args.contamination_probes_path
    if args.contamination_probes_path is not None:
        if not args.expected_contamination_assessment_sha256:
            raise SystemExit(
                "contamination assessment requires a frozen SHA-256"
            )
        try:
            args.contamination_probes_path = (
                _freeze_contamination_assessment(
                    source=args.contamination_probes_path,
                    expected_sha256=(
                        args.expected_contamination_assessment_sha256
                    ),
                    batch_root=batch_root,
                )
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    elif args.expected_contamination_assessment_sha256:
        raise SystemExit(
            "contamination assessment SHA-256 requires a source path"
        )
    plans = build_case_plans(args, batch_root)
    if args.resume:
        mismatched = [
            plan.short_name for plan in plans if _resume_plan_mismatch(plan)
        ]
        if mismatched:
            # Batch-level preflight happens before the executor is created: one
            # mismatched cell cannot race healthy-looking cells into paid child
            # calls, and the original root manifest remains untouched.
            print(
                "resume_plan_mismatch:" + ",".join(mismatched),
                file=sys.stderr,
            )
            return 1
    circuit = threading.Event()
    if args.max_parallel > 1:
        with ThreadPoolExecutor(
            max_workers=min(args.max_parallel, len(plans))
        ) as pool:
            futures = [
                pool.submit(
                    _run_case,
                    plan,
                    dry_run=args.dry_run,
                    resume=args.resume,
                    circuit=circuit,
                )
                for plan in plans
            ]
            results = [future.result() for future in futures]
    else:
        results = [
            _run_case(
                plan,
                dry_run=args.dry_run,
                resume=args.resume,
                circuit=circuit,
            )
            for plan in plans
        ]
    run_records = []
    if not args.dry_run:
        for result in results:
            relative = result.get("experiment_run_record")
            if not relative:
                continue
            record_path = REPO_ROOT / relative
            record = json.loads(record_path.read_text(encoding="utf-8"))
            # Annotate rather than filter: paired-cell analysis needs every
            # record of the block, so degraded runs stay in the aggregate but
            # carry the batch-level verdict. The runner is the authority for
            # this key; additive so existing record fields are untouched.
            annotation = {
                "case": result.get("case"),
                "condition_id": result.get("condition_id"),
                "status": result.get("status"),
            }
            if result.get("failure_reason"):
                annotation["failure_reason"] = result["failure_reason"]
            record["batch_case"] = annotation
            run_records.append(record)
        (batch_root / "experiment_runs.json").write_text(
            json.dumps({"runs": run_records}, indent=2, ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )
        (batch_root / "experiment_runs.jsonl").write_text(
            "".join(
                json.dumps(record, ensure_ascii=False) + "\n"
                for record in run_records
            ),
            encoding="utf-8",
        )
    manifest = {
        "batch_id": batch_id,
        "paired_seed": args.seed,
        # Identity, not locator — see --repository-id.
        "pack": getattr(args, "repository_id", None) or args.dataset,
        "dataset": args.dataset,
        "provider": args.provider,
        "model": args.model,
        "randomization_block": plans[0].randomization_block,
        "randomization_order": [plan.short_name for plan in plans],
        "replication_id": plans[0].replication_id,
        "experiment_phase": args.experiment_phase,
        "arm_map": args.arm_map,
        "randomization_order_override": args.randomization_order_override,
        "evaluation_perturbation": args.evaluation_perturbation,
        "action_selection_mode": {
            plan.short_name: plan.action_selection_mode for plan in plans
        },
        "resource_ceilings": {
            "max_llm_calls": args.max_llm_calls,
            "max_llm_requested_tokens": args.max_llm_requested_tokens,
            "max_llm_prompt_characters": args.max_llm_prompt_characters,
            "max_primary_actions": args.max_primary_actions,
            "max_ticks": args.max_ticks,
            "case_wall_clock_seconds": args.case_wall_clock_seconds,
        },
        "contamination": {
            "model_cutoff_policy": args.model_cutoff_policy,
            "status": args.contamination_status,
            "source_path": (
                str(contamination_source_path.resolve())
                if contamination_source_path
                else None
            ),
            "source_sha256": (
                args.expected_contamination_assessment_sha256
            ),
            "probes_path": (
                str(args.contamination_probes_path)
                if args.contamination_probes_path
                else None
            ),
            "clearance": args.contamination_clearance,
        },
        "record_files": (
            {
                "json": "experiment_runs.json",
                "jsonl": "experiment_runs.jsonl",
                "count": len(run_records),
            }
            if not args.dry_run
            else None
        ),
        "cases": results,
        "execution": {
            "max_parallel": args.max_parallel,
            "resume": args.resume,
            "completed_cases_skipped": [
                result["case"]
                for result in results
                if result.get("skipped_as_completed")
            ],
            "provider_circuit": {
                "tripped": circuit.is_set(),
                "tripped_by": [
                    result["case"]
                    for result in results
                    if result.get("failure_reason") == "llm_blackout"
                ],
                "cases_not_run": [
                    result["case"]
                    for result in results
                    if result.get("status") == "not_run"
                ],
            },
        },
        "isolation": {
            "process_per_case": True,
            "shared_world_state": False,
            "shared_checkpoint": False,
            "resume_allowed": True,
            "resume_scope": "same_case_identity_checked_checkpoint",
            "inherited_org_environment": False,
        },
    }
    (batch_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(batch_root)
    return 0 if all(case["status"] in ("passed", "dry_run") for case in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CASE_IDS",
    "DEFAULT_CASES",
    "DEFAULT_DATASET",
    "CasePlan",
    "build_case_environment",
    "build_case_plans",
    "build_randomization_block",
    "deterministic_case_order",
    "main",
]
