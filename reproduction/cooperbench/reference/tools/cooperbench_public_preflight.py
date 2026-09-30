"""Public-only, zero-provider CooperBench bootstrap and runtime diagnostic.

Run this from the source checkout that will execute the experiment. The output
must be a new directory outside source, dataset and existing benchmark runs.
Passing this command is not an evaluator result or a benchmark score.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any
from unittest.mock import patch


SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

SCHEMA = "cooperbench_zero_provider_public_preflight_v3"
_PRIVATE_PARTS = {"hidden", "private", "gold", "solution", "solutions"}
_CONFIG_FIELDS = {
    "orgenv_ticks", "orgenv_seed", "orgenv_provider", "orgenv_reasoning_effort",
    "orgenv_worker_timeout_seconds", "orgenv_max_surface_files", "orgenv_max_surface_bytes",
}


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _public_path(path: Path) -> Path:
    if _PRIVATE_PARTS.intersection(part.casefold() for part in path.parts):
        raise ValueError("private_or_gold_path_prohibited")
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("symlink_input_or_output_prohibited")
    return path.resolve()


def _new_output(raw: str, dataset: Path, config: Path) -> Path:
    output = _public_path(Path(raw).absolute())
    if (output == SOURCE_ROOT or SOURCE_ROOT in output.parents
            or output == dataset or dataset in output.parents
            or output == config or output in config.parents):
        raise ValueError("output_must_be_independent_of_source_and_inputs")
    if any(part.casefold() in {"cooperbench-runs", "orgenv_b3_two_agent"} for part in output.parts):
        raise ValueError("output_must_not_be_a_benchmark_run")
    for parent in output.parents:
        if any((parent / marker).exists() for marker in (
                "trajectory.jsonl", "pair_request.json", "pair_outcome.json", "experiment_run_record.json")):
            raise ValueError("output_must_not_be_inside_an_existing_run")
    output.mkdir(parents=True, exist_ok=False)
    return output


@contextmanager
def _zero_provider_environment():
    before, bytecode = dict(os.environ), sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        for key in list(os.environ):
            upper = key.upper()
            if (any(word in upper for word in ("API_KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "AUTHORIZATION", "ACCESS_KEY"))
                    or (upper.endswith("PROXY") and "://" in os.environ[key] and "@" in os.environ[key])):
                os.environ.pop(key, None)
        os.environ.update(ORG_LLM="0", ORG_LLM_ENABLED="0")
        yield
    finally:
        os.environ.clear()
        os.environ.update(before)
        sys.dont_write_bytecode = bytecode


def _safe(value: Any) -> Any:
    from environments.org_env.programbench.public_evidence import redact_public_evidence
    return redact_public_evidence(value)


def _write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_safe(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _features(value: str) -> tuple[int, int]:
    if not re.fullmatch(r"[1-9][0-9]*,[1-9][0-9]*", value):
        raise argparse.ArgumentTypeError("features must be two positive IDs, e.g. 4,6")
    parsed = tuple(map(int, value.split(",")))
    if parsed[0] == parsed[1]:
        raise argparse.ArgumentTypeError("features must be distinct")
    return parsed


def _diagnostic_command(value: str | None) -> list[str] | None:
    if value is None:
        return None
    command = json.loads(value)
    if (not isinstance(command, list) or not command
            or any(not isinstance(item, str) or not item or "\0" in item for item in command)):
        raise ValueError("public_test_command_json_must_be_nonempty_argv")
    for item in command:
        parts = set(re.split(r"[/\\\s=]+", item.casefold()))
        if parts & _PRIVATE_PARTS or ".patch" in item.casefold() or ".." in parts:
            raise ValueError("diagnostic_command_must_be_public_only")
    return command


def _code_receipts() -> dict[str, str]:
    paths = (
        "tools/cooperbench_public_preflight.py", "environments/org_env/cooperbench/worker.py",
        "environments/org_env/cooperbench/surface.py", "environments/org_env/cooperbench/contract.py",
        "environments/org_env/cooperbench/runtime_assets.py", "environments/org_env/cooperbench/baseline_assets.py",
        "environments/org_env/cooperbench/source_views.py", "environments/org_env/product/materialize.py",
        "environments/org_env/product/substrates/oss_time_machine.py", "environments/org_env/runtime_adapter/live.py",
    )
    return {name: _sha((SOURCE_ROOT / name).read_bytes()) for name in paths}


def run_preflight(args: argparse.Namespace) -> tuple[int, dict]:
    """Use actual worker bootstrap seams, without running an actor or provider."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.repo) or args.repo.casefold() in _PRIVATE_PARTS:
        raise ValueError("public_repository_name_invalid")
    if args.task_id < 1:
        raise ValueError("positive_task_id_required")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/:@+-]*", args.image):
        raise ValueError("docker_image_reference_invalid")
    dataset = _public_path(Path(args.dataset_dir).absolute())
    config_path = _public_path(Path(args.config).absolute())
    if not dataset.is_dir() or not config_path.is_file():
        raise ValueError("dataset_directory_and_config_file_required")
    diagnostic = _diagnostic_command(args.public_test_command_json)
    output = _new_output(args.output, dataset, config_path)
    report = {
        "schema_version": SCHEMA, "purpose": "Untouched public bootstrap/runtime only; NOT a benchmark score",
        "source_root": str(SOURCE_ROOT), "repo": args.repo, "task_id": args.task_id,
        "features": list(args.features), "image": args.image, "provider_calls": 0,
        "provider_call_attempts": 0, "official_evaluator_invoked": False,
        "run_pair_invoked": False, "actor_steps": 0, "hidden_or_gold_content_read": False,
        "started_at": dt.datetime.now(dt.timezone.utc).isoformat(), "passed": False,
    }
    counts = {"provider_calls": 0}

    def emit(stage: str):
        report["stage"] = stage
        report["provider_call_attempts"] = counts["provider_calls"]
        _write(output / "preflight_report.json", report)
        print(json.dumps({"stage": stage, "output": str(output)}), flush=True)

    def forbidden_provider(*_args, **_kwargs):
        counts["provider_calls"] += 1
        raise RuntimeError("provider_calls_prohibited_in_public_preflight")

    try:
        with _zero_provider_environment():
            import yaml
            from environments.org_env.cooperbench import worker
            from environments.org_env.cooperbench.contract import FeatureCall, PairRequest
            from environments.org_env.cooperbench.surface import (
                FEATURE_INDEPENDENT_SOURCE_POLICY, materialize_public_pack, select_public_surface,
            )
            from environments.org_env.cooperbench.runtime_assets import runtime_asset_receipt
            from environments.org_env.llm.client import OrgLLMClient
            from environments.org_env.runtime_adapter.live import OrgInspectorSession
            config_bytes = config_path.read_bytes()
            raw_config = yaml.safe_load(config_bytes)
            if not isinstance(raw_config, dict):
                raise ValueError("worker_config_mapping_required")
            config = {key: raw_config[key] for key in _CONFIG_FIELDS if key in raw_config}
            config.update(run_id=output.name, backend="docker")
            model_metadata = str(raw_config.get("orgenv_model") or raw_config.get("model") or "unspecified_zero_provider")
            report.update(worker_config_sha256=_sha(config_bytes), source_code_sha256=_code_receipts(),
                          model_metadata_only=model_metadata, public_brief_files=[])
            calls = {}
            agents = ("agent1", "agent2")
            for actor, feature in zip(agents, args.features):
                path = _public_path(dataset / args.repo / f"task{args.task_id}" / f"feature{feature}" / "feature.md")
                if dataset not in path.parents:
                    raise ValueError("feature_brief_outside_dataset")
                raw = path.read_bytes()
                report["public_brief_files"].append({"path": str(path), "sha256": _sha(raw)})
                calls[actor] = FeatureCall.from_adapter_call(
                    task=raw.decode("utf-8"), image=args.image, agent_id=actor, agents=agents,
                    model_name=model_metadata, config=config, log_dir=str(output))
            request = PairRequest.from_calls(calls)
            report.update(seed=request.seed, treatment_id=request.treatment_id,
                          configured_ticks_metadata_only=request.ticks)
            with patch.object(OrgLLMClient, "generate_json", forbidden_provider):
                emit("extract_public_image_repository")
                base_repo = output / "image_repo"
                worker._extract_image_repository(request.image, base_repo)
                report["image_repository_head"] = worker._git(base_repo, "rev-parse", "HEAD").strip()
                report["image_repository_clean"] = not bool(worker._git(base_repo, "status", "--porcelain").strip())
                emit("select_and_materialize_public_surface")
                plan = select_public_surface(
                    base_repo, request.tasks, max_files=request.max_surface_files,
                    max_total_bytes=request.max_surface_bytes, source_policy=FEATURE_INDEPENDENT_SOURCE_POLICY)
                _write(output / "public_surface_receipt.json", plan.to_dict())
                report.update(source_policy=plan.source_policy, surface_digest=plan.baseline_digest,
                              public_test_paths=list(plan.public_test_paths), public_test_command=list(plan.public_test_command))
                pack_dir = output / "public_pack"
                project_id = "cooperbench_" + _sha(f"{request.run_id}\0{request.image}".encode())[:16]
                materialize_public_pack(base_repo, pack_dir, request.tasks, plan, project_id=project_id, private_features=True)
                image_id, platform = worker._resolved_image_runtime(request.image)
                worker._configure_worker_environment(request, pack_dir, project_id,
                                                     evaluator_image=image_id, evaluator_platform=platform)
                os.environ.update(ORG_LLM="0", ORG_LLM_ENABLED="0", ORG_PRODUCT_SMOKE_ROOT=str(output / "public_runtime"))
                report.update(image_id=image_id, platform=platform,
                              zero_provider_overrides={"load_llm": False, "defer_initial_readiness_check": True,
                                                       "ORG_LLM": "0", "ORG_LLM_ENABLED": "0"})
                emit("build_actual_two_person_B3_without_provider")
                session = OrgInspectorSession(
                    seed=request.seed, n_agents=2, member_ids=worker._INTERNAL_AGENT_IDS,
                    load_llm=False, approval_mode="agent", experiment_condition=worker._B3_CONDITION,
                    noncanonical_roster_variant=worker._ROSTER_VARIANT, max_frames=1,
                    defer_initial_readiness_check=True)
                world = session.world
                condition = world.condition_spec
                if (world.llm_client is not None or tuple(world.agents) != worker._INTERNAL_AGENT_IDS
                        or world.experiment_condition != worker._B3_CONDITION or not world.experiment_condition_explicit
                        or not all(getattr(condition, key, False) for key in (
                            "profile_conditioning_enabled", "institutionalization_enabled", "capability_learning_enabled"))):
                    raise ValueError("zero_provider_two_person_B3_contract_failed")
                evaluator_runtime = worker._require_evaluator_runtime_binding(
                    request,
                    evaluator_image=image_id,
                    evaluator_platform=platform,
                )
                world.__dict__["_cooperbench_evaluator_runtime"] = dict(
                    evaluator_runtime
                )
                world.__dict__["_cooperbench_image_workspace"] = (
                    worker._IMAGE_REPOSITORY_WORKSPACE
                )
                report["evaluator_runtime"] = evaluator_runtime
                _write(output / "evaluator_runtime_receipt.json", evaluator_runtime)
                if runtime_asset_receipt(world) != plan.runtime_assets_manifest:
                    raise ValueError("bootstrap_runtime_asset_manifest_mismatch")
                report["runtime_assets"] = runtime_asset_receipt(world)
                initial_tick = world.world_tick
                report["world"] = {"members": list(world.agents), "condition": world.experiment_condition,
                                   "mode": world.experiment_mode, "action_selection": world.action_selection_mode,
                                   "tick": initial_tick, "llm_client_attached": False}
                emit("execute_untouched_public_runtime_preflight")
                receipt = worker._verified_public_runtime_preflight(world, plan.public_test_command)
                report["public_runtime_preflight"] = receipt
                report["public_test_candidate_command"] = list(plan.public_test_command)
                report["public_test_command"] = list(receipt.get("command") or [])
                _write(output / "public_runtime_preflight.json", receipt)
                # A supplied command is a separate fixed diagnostic, NEVER a
                # substitute for the official plan-selected public gate.
                if diagnostic is not None:
                    emit("execute_separate_public_diagnostic")
                    extra = worker._public_runtime_preflight(world, tuple(diagnostic))
                    report["separate_public_diagnostic"] = {"official_gate": False, "result": extra}
                    _write(output / "separate_public_diagnostic.json", report["separate_public_diagnostic"])
                _write(output / "public_runtime_observation.json", {
                    "observations": list(world.__dict__.get("_smoke_cache", {}).values())})
                worker._require_public_runtime_preflight(receipt)
                worker._bind_verified_public_test_command(world, receipt)
                world._cooperbench_public_runtime_preflight = dict(receipt)
                world._oss_component_map = {key: list(paths) for key, paths in plan.component_map.items()}
                behavior_probe_capability = worker._behavior_probe_capability(plan)
                world._cooperbench_behavior_probe_capability = dict(
                    behavior_probe_capability
                )
                report["behavior_probe_capability"] = behavior_probe_capability
                report["ownership"] = worker._bind_feature_owners(world, request)
                from environments.org_env.cooperbench.work_schedule import (
                    compressed_schedule_receipt,
                    model_call_budget_receipt,
                )
                report["decision_schedule"] = compressed_schedule_receipt(world)
                report["model_call_budget"] = model_call_budget_receipt(world)
                from environments.org_env.cooperbench.source_views import freeze_baseline, initialize_actor_desks, actor_desk_snapshot
                from environments.org_env.cooperbench.visibility import is_strict_coop, visible_brief_context
                baseline = freeze_baseline(world)
                initialize_actor_desks(world)
                if behavior_probe_capability["available"]:
                    from environments.org_env.cooperbench.replay_workflow import SCHEMA as REPLAY_SCHEMA
                    world._cooperbench_integrated_probe_workflow = {
                        "schema_version": REPLAY_SCHEMA
                    }
                report["baseline_snapshot"] = baseline.receipt()
                report["actor_snapshots"] = {actor: actor_desk_snapshot(world, actor).receipt() for actor in world.agents}
                report["visible_feature_ids"] = {actor: sorted(visible_brief_context(world, actor)) for actor in world.agents}
                report["strict_coop"] = is_strict_coop(world)
                if (not report["strict_coop"] or any(len(ids) != 1 for ids in report["visible_feature_ids"].values())
                        or any(actor_desk_snapshot(world, actor).tree_digest != baseline.tree_digest for actor in world.agents)
                        or baseline.runtime_assets_digest != plan.runtime_assets_manifest["asset_set_digest"]
                        or world.world_tick != initial_tick or counts["provider_calls"]):
                    raise ValueError("zero_provider_source_or_visibility_invariant_failed")
                report["passed"] = True
        report["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        emit("complete")
        return 0, _safe(report)
    except Exception as error:
        report.update(passed=False, error=str(_safe(f"{type(error).__name__}:{error}"))[:1200],
                      provider_call_attempts=counts["provider_calls"], finished_at=dt.datetime.now(dt.timezone.utc).isoformat())
        emit("failed")
        return 1, _safe(report)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--task-id", required=True, type=int)
    parser.add_argument("--features", required=True, type=_features)
    parser.add_argument("--image", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--public-test-command-json", help="Separate public diagnostic argv; never replaces the selected suite")
    args = parser.parse_args(argv)
    try:
        code, report = run_preflight(args)
    except (ValueError, OSError) as error:
        print(json.dumps({"passed": False, "error": str(_safe(f"{type(error).__name__}:{error}"))[:1200]}), file=sys.stderr)
        return 1
    print(json.dumps({"passed": report["passed"], "provider_calls": report["provider_calls"],
                      "provider_call_attempts": report["provider_call_attempts"], "not_a_benchmark_score": True}), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
