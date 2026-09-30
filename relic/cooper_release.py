"""Boundary for the final frozen CooperBench B3-2 full-652 implementation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ROOT = REPOSITORY_ROOT / "reproduction/cooperbench/reference"
COOPERBENCH_COMMIT = "4913c4ebb84d2606cdb5628936b88529f3e181df"
COOPERBENCH_DATASET_REVISION = "b612b1a35af722751454813d9e5a7888f065fc9e"
PAPER_SUBSET = "relic_full652"
REPORTED_MODEL = "Claude Opus 4.6"
EXTERNAL_ADAPTER = "environments.org_env.cooperbench.adapter"
EXTERNAL_AGENT = "orgenv_b3_two_agent"
EXPECTED_PAIR_COUNT = 652
EXPECTED_TASK_COUNT = 30
EXPECTED_REPOSITORY_COUNT = 12
_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_SAFE_RUN_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_SAFE_IMAGE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/:@+-]*\Z")


class CooperReleaseError(ValueError):
    """A source selection or required external input is unavailable."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class CooperPair:
    repo: str
    task_id: int
    features: tuple[int, int]

    @property
    def key(self) -> str:
        return f"{self.repo}:{self.task_id}:{self.features[0]},{self.features[1]}"


@dataclass(frozen=True)
class SourceBatch:
    path: Path
    source_path: str
    sha256: str


SOURCE_BATCHES = (
    SourceBatch(
        REPOSITORY_ROOT / "configs/cooperbench/full652.json",
        "full652 pair manifest",
        "655ca1f61a6d3e1ba83ffc4f5b8c96381f57daef0e35bb83be068ce3ce8a8e92",
    ),
)


@dataclass(frozen=True)
class CooperSelection:
    batches: tuple[SourceBatch, ...]
    pairs: tuple[CooperPair, ...]

    @property
    def keys(self) -> frozenset[str]:
        return frozenset(pair.key for pair in self.pairs)


def _pairs(document: Any, code: str) -> tuple[CooperPair, ...]:
    """Read exactly the upstream subset grammar: tasks -> repo/task_id/pairs."""

    try:
        tasks = document["tasks"]
        if not isinstance(tasks, list):
            raise TypeError
        result: list[CooperPair] = []
        for task in tasks:
            repo, task_id, pairs = task["repo"], task["task_id"], task["pairs"]
            if (
                not isinstance(repo, str)
                or _SAFE_NAME.fullmatch(repo) is None
                or isinstance(task_id, bool)
                or int(task_id) < 0
                or not isinstance(pairs, list)
            ):
                raise ValueError
            for pair in pairs:
                if not isinstance(pair, list) or len(pair) != 2:
                    raise ValueError
                first, second = int(pair[0]), int(pair[1])
                if isinstance(pair[0], bool) or isinstance(pair[1], bool) or first < 1 or first >= second:
                    raise ValueError
                result.append(CooperPair(repo, int(task_id), (first, second)))
        return tuple(result)
    except (KeyError, TypeError, ValueError, IndexError) as error:
        raise CooperReleaseError(code) from error


def source_selection() -> CooperSelection:
    """Return the complete manifest, without filtering by observed results."""

    pairs: list[CooperPair] = []
    for batch in SOURCE_BATCHES:
        try:
            raw = batch.path.read_bytes()
        except OSError as error:
            raise CooperReleaseError("cooper_source_selection_file_missing") from error
        if hashlib.sha256(raw).hexdigest() != batch.sha256:
            raise CooperReleaseError("cooper_source_selection_digest_mismatch")
        try:
            pairs.extend(_pairs(json.loads(raw), "cooper_source_selection_invalid_json"))
        except json.JSONDecodeError as error:
            raise CooperReleaseError("cooper_source_selection_invalid_json") from error
    selection = CooperSelection(SOURCE_BATCHES, tuple(pairs))
    if (
        len(selection.pairs) != EXPECTED_PAIR_COUNT
        or len(selection.keys) != EXPECTED_PAIR_COUNT
        or len({(pair.repo, pair.task_id) for pair in selection.pairs}) != EXPECTED_TASK_COUNT
        or len({pair.repo for pair in selection.pairs}) != EXPECTED_REPOSITORY_COUNT
    ):
        raise CooperReleaseError("cooper_source_selection_shape_mismatch")
    return selection


def get_pair(selection: CooperSelection, key: str) -> CooperPair:
    matches = [pair for pair in selection.pairs if pair.key == key]
    if len(matches) != 1:
        raise CooperReleaseError("cooper_source_pair_key_not_found")
    return matches[0]


def verify_upstream_subset(selection: CooperSelection, dataset_dir: Path) -> Path:
    """Fail closed unless upstream's named combined subset equals the source union."""

    path = dataset_dir.resolve() / "subsets" / f"{PAPER_SUBSET}.json"
    try:
        observed = _pairs(json.loads(path.read_bytes()), "cooperbench_paper_subset_invalid_json")
    except OSError as error:
        raise CooperReleaseError("cooperbench_paper_subset_missing") from error
    except json.JSONDecodeError as error:
        raise CooperReleaseError("cooperbench_paper_subset_invalid_json") from error
    keys = {pair.key for pair in observed}
    if len(observed) != EXPECTED_PAIR_COUNT or len(keys) != EXPECTED_PAIR_COUNT:
        raise CooperReleaseError("cooperbench_paper_subset_pair_count_mismatch")
    if keys != selection.keys:
        raise CooperReleaseError("cooperbench_paper_subset_source_union_mismatch")
    return path


def verify_cooperbench_checkout(cooperbench_root: Path) -> Path:
    root = cooperbench_root.resolve()
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError as error:
        raise CooperReleaseError("cooperbench_checkout_revision_unavailable") from error
    if not root.is_dir():
        raise CooperReleaseError("cooperbench_checkout_missing")
    if result.returncode != 0:
        raise CooperReleaseError("cooperbench_checkout_revision_unavailable")
    if result.stdout.strip() != COOPERBENCH_COMMIT:
        raise CooperReleaseError("cooperbench_checkout_revision_mismatch")
    return root


def resolve_cooperbench_binary(value: str) -> str:
    path = Path(value)
    if path.parent != Path("."):
        if path.is_file() and os.access(path, os.X_OK):
            return str(path.resolve())
        raise CooperReleaseError("cooperbench_cli_missing")
    resolved = shutil.which(value)
    if resolved is None:
        raise CooperReleaseError("cooperbench_cli_missing")
    return resolved


def provider_environment_issues(model_name: str) -> list[str]:
    """Check source-required gateway settings without reading/printing secrets."""

    environment = os.environ
    checks = (
        (bool(model_name.strip()), "cooperbench_gateway_model_required"),
        (environment.get("ORG_LLM_ENABLED", "").lower() in {"1", "true"}, "org_llm_enabled_required"),
        (environment.get("ORG_LLM_PROVIDER", "").lower() == "openai", "org_llm_openai_gateway_required"),
        (bool(environment.get("ORG_LLM_BASE_URL", "").strip()), "org_llm_base_url_required"),
        (bool(environment.get("ORG_LLM_API_KEY", "").strip() or environment.get("OPENAI_API_KEY", "").strip()), "openai_compatible_gateway_credential_required"),
        (environment.get("ORG_LLM_WIRE_API", "") == "chat_completions", "org_llm_wire_api_mismatch"),
        (environment.get("ORG_LLM_JSON_TRANSPORT", "") == "prompt_only", "org_llm_json_transport_mismatch"),
        (environment.get("ORG_LLM_REASONING_EFFORT", "").lower() == "high", "org_llm_reasoning_effort_mismatch"),
    )
    return [error for passed, error in checks if not passed]


def external_environment(cooperbench_root: Path) -> dict[str, str]:
    environment = dict(os.environ)
    paths = [str(REFERENCE_ROOT), str(cooperbench_root / "src"), str(REPOSITORY_ROOT)]
    if existing := environment.get("PYTHONPATH"):
        paths.append(existing)
    environment["PYTHONPATH"] = os.pathsep.join(paths)
    environment["COOPERBENCH_EXTERNAL_AGENTS"] = EXTERNAL_ADAPTER
    return environment


def _run_name(value: str) -> str:
    if _SAFE_RUN_NAME.fullmatch(value) is None:
        raise CooperReleaseError("cooperbench_run_name_invalid")
    return value


def upstream_run_command(
    *,
    cooperbench_binary: str,
    dataset_dir: Path,
    log_dir: Path,
    run_name: str,
    model_name: str,
    concurrency: int,
    eval_concurrency: int,
    redis_url: str,
    agent_config: Path,
) -> list[str]:
    if concurrency < 1 or eval_concurrency < 1:
        raise CooperReleaseError("cooperbench_concurrency_invalid")
    if not model_name.strip() or "\0" in model_name or not redis_url.strip():
        raise CooperReleaseError("cooperbench_gateway_model_required")
    return [
        cooperbench_binary, "run", "-n", _run_name(run_name), "-s", PAPER_SUBSET,
        "-m", model_name.strip(), "-a", EXTERNAL_AGENT, "--setting", "coop",
        "--backend", "docker", "--redis", redis_url, "--dataset-dir", str(dataset_dir.resolve()),
        "--log-dir", str(log_dir.resolve()), "--agent-config", str(agent_config.resolve()),
        "--eval-concurrency", str(eval_concurrency), "-c", str(concurrency),
    ]


def upstream_eval_command(
    *,
    cooperbench_binary: str,
    dataset_dir: Path,
    log_dir: Path,
    run_name: str,
    concurrency: int,
) -> list[str]:
    if concurrency < 1:
        raise CooperReleaseError("cooperbench_concurrency_invalid")
    return [
        cooperbench_binary, "eval", "-n", _run_name(run_name), "-s", PAPER_SUBSET,
        "--backend", "docker", "--dataset-dir", str(dataset_dir.resolve()),
        "--log-dir", str(log_dir.resolve()), "-c", str(concurrency),
    ]


def public_preflight_command(
    *,
    selection: CooperSelection,
    pair_key: str,
    image: str,
    dataset_dir: Path,
    output: Path,
    config: Path,
) -> list[str]:
    pair = get_pair(selection, pair_key)
    if pair.task_id == 0:
        raise CooperReleaseError("cooperbench_public_preflight_task_id_zero_unsupported")
    if _SAFE_IMAGE.fullmatch(image.strip()) is None:
        raise CooperReleaseError("cooperbench_task_image_reference_invalid")
    return [
        sys.executable, str(REFERENCE_ROOT / "tools/cooperbench_public_preflight.py"),
        "--dataset-dir", str(dataset_dir.resolve()), "--repo", pair.repo,
        "--task-id", str(pair.task_id), "--features", f"{pair.features[0]},{pair.features[1]}",
        "--image", image.strip(), "--config", str(config.resolve()),
        "--output", str(output.resolve()),
    ]


def read_upstream_summary(log_dir: Path, run_name: str) -> bytes:
    try:
        raw = (log_dir.resolve() / _run_name(run_name) / "summary.json").read_bytes()
        if not isinstance(json.loads(raw), Mapping):
            raise ValueError
        return raw
    except OSError as error:
        raise CooperReleaseError("cooperbench_upstream_summary_missing") from error
    except (json.JSONDecodeError, ValueError) as error:
        raise CooperReleaseError("cooperbench_upstream_summary_invalid_json") from error


def missing_input_report(
    *,
    cooperbench_root: Path | None = None,
    dataset_dir: Path | None = None,
    cooperbench_binary: str | None = None,
    check_provider: bool = False,
    model_name: str = "",
) -> dict[str, Any]:
    missing: list[str] = []
    try:
        selection = source_selection()
    except CooperReleaseError as error:
        selection = None
        missing.append(error.code)
    if dataset_dir is None:
        missing.append("cooperbench_dataset_dir_required")
    elif selection is not None:
        try:
            verify_upstream_subset(selection, dataset_dir)
        except CooperReleaseError as error:
            missing.append(error.code)
    if cooperbench_root is None:
        missing.append("cooperbench_checkout_required")
    else:
        try:
            verify_cooperbench_checkout(cooperbench_root)
        except CooperReleaseError as error:
            missing.append(error.code)
    if cooperbench_binary is None:
        missing.append("cooperbench_cli_required")
    else:
        try:
            resolve_cooperbench_binary(cooperbench_binary)
        except CooperReleaseError as error:
            missing.append(error.code)
    if check_provider:
        missing.extend(provider_environment_issues(model_name))
    if not (REFERENCE_ROOT / "environments/org_env/cooperbench/worker.py").is_file():
        missing.append("cooperbench_frozen_reference_checkout_missing")
    gaps = []
    if not (REPOSITORY_ROOT / "LICENSE").is_file():
        gaps.insert(0, "relic_root_license_missing")
    return {
        "schema_version": "relic-cooperbench-missing-input-report-v2",
        "paper_scope": {"subset": PAPER_SUBSET, "pairs": EXPECTED_PAIR_COUNT, "task_instances": 30,
                        "repositories": 12,
                        "reported_model": REPORTED_MODEL},
        "historical_artifacts": "https://huggingface.co/datasets/Horseback-Eridute/CooperBench-B3-2-Full-652",
        "implementation": "final frozen reference; historical runs used earlier fixes too",
        "source_selection": None if selection is None else {"pairs": len(selection.pairs), "batches": [
            {"path": str(batch.path.relative_to(REPOSITORY_ROOT)), "source_path": batch.source_path,
             "sha256": batch.sha256} for batch in selection.batches]},
        "runtime_ready": not missing,
        "runtime_missing": sorted(set(missing)),
        "release_complete": not missing and not gaps,
        "release_gaps": gaps,
        "not_a_benchmark_result": True,
    }


def run_command(command: Sequence[str], *, cwd: Path, environment: Mapping[str, str]) -> int:
    return subprocess.run(list(command), cwd=str(cwd), env=dict(environment), check=False).returncode
