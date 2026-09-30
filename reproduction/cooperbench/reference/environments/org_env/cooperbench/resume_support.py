"""Verified checkpoint continuation for interrupted CooperBench pair workers.

The existing checkpoint API saves the full OrgWorld but the pair worker did
not call its loader.  Resume evidence stays separate from normal fresh runs.
"""
from __future__ import annotations

import contextlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator
from uuid import uuid4

from environments.org_env.runtime_adapter.checkpoint import checkpoint_info, load_world_checkpoint


@dataclass(frozen=True)
class ResumeCandidate:
    checkpoint: Path
    tick: int
    temp_root: Path
    dataset_dir: Path
    world: Any
    skipped_corrupt: tuple[str, ...]
    resources_prepared: bool = False


def _tick(path: Path) -> int:
    name = path.stem
    return int(name[1:]) if name.startswith("t") and name[1:].isdigit() else -1


def discover(request: Any, output_dir: Path, *, materializer=None) -> ResumeCandidate | None:
    """Select the newest verified same-pair checkpoint; never silently restart."""
    paths = sorted((output_dir / "checkpoints").glob("t*.pkl"), key=_tick, reverse=True)
    if not paths:
        return None
    skipped: list[str] = []
    for path in paths:
        try:
            info = checkpoint_info(str(path))
        except (OSError, ValueError) as error:
            skipped.append(f"{path.name}:{type(error).__name__}:{error}")
            continue
        meta = info.get("meta") or {}
        expected = {
            "schema_version": "orgenv_cooperbench_pair_checkpoint_v1",
            "run_id": request.run_id,
            "agents": list(request.agents),
            "tasks": dict(request.tasks),
            "image": request.image,
            "model_name": request.model_name,
            "target_ticks": request.ticks,
        }
        if any(meta.get(key) != value for key, value in expected.items()):
            raise ValueError(f"cooperbench_resume_pair_identity_mismatch:{path.name}")
        tick = int(info.get("tick", -1))
        if tick != _tick(path) or not 0 < tick < request.ticks:
            raise ValueError(f"cooperbench_resume_tick_invalid:{path.name}")
        rebuilt = set()
        def restore_missing_assets(asset_root):
            temp = asset_root.parent.parent
            if (asset_root.name != "cooperbench_runtime_assets"
                    or asset_root.parent.name != "public_pack"
                    or temp.parent != Path(tempfile.gettempdir())
                    or not temp.name.startswith("orgenv-cooper-")
                    or temp.is_symlink() or materializer is None):
                raise ValueError("cooperbench_resume_asset_path_invalid")
            if temp.exists():
                raise ValueError("cooperbench_resume_temp_root_already_exists")
            temp.mkdir(mode=0o700)
            materializer(temp)
            rebuilt.add(temp)
        from .runtime_assets import restoring_runtime_directories
        with restoring_runtime_directories(restore_missing_assets):
            world, _ = load_world_checkpoint(str(path), load_llm=False)
        if int(getattr(world, "world_tick", -1)) != tick:
            raise ValueError(f"cooperbench_resume_world_tick_mismatch:{path.name}")
        binding = world.__dict__.get("_oss_time_machine_eval_binding") or {}
        dataset_dir = Path(str(binding.get("dataset_dir") or ""))
        temp_root = dataset_dir.parent
        if (
            not dataset_dir.is_absolute()
            or dataset_dir.name != "public_pack"
            or temp_root.parent != Path(tempfile.gettempdir())
            or not temp_root.name.startswith("orgenv-cooper-")
        ):
            raise ValueError(f"cooperbench_resume_dataset_path_invalid:{path.name}")
        if rebuilt and rebuilt != {temp_root}:
            raise ValueError("cooperbench_resume_asset_dataset_mismatch")
        return ResumeCandidate(path, tick, temp_root, dataset_dir, world, tuple(skipped), bool(rebuilt))
    raise ValueError("cooperbench_resume_no_verified_checkpoint:" + ";".join(skipped))


@contextlib.contextmanager
def workspace(candidate: ResumeCandidate | None) -> Iterator[str]:
    if candidate is None:
        with tempfile.TemporaryDirectory(prefix="orgenv-cooper-") as temporary:
            yield temporary
        return
    root = candidate.temp_root
    if candidate.resources_prepared:
        yield str(root)
        return
    if root.exists():
        raise ValueError("cooperbench_resume_temp_root_already_exists")
    root.mkdir(mode=0o700)
    # Keep the recreated pack for diagnosis until official result/eval are durable.
    yield str(root)


def verify_materialized_pack(candidate: ResumeCandidate) -> None:
    """Reopen the checkpoint's sealed evaluator binding against the new pack."""
    from environments.org_env.product.substrates.eval_assets import oss_eval_assets
    assets = oss_eval_assets(candidate.world)
    if not isinstance(assets, dict):
        raise ValueError("cooperbench_resume_evaluator_assets_missing")
    if Path(str(assets.get("dataset_dir") or "")) != candidate.dataset_dir:
        raise ValueError("cooperbench_resume_evaluator_dataset_mismatch")


def _atomic_jsonl(path: Path, rows: list[str]) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n", dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp", delete=False,
    ) as stream:
        temporary = Path(stream.name)
        stream.writelines(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _trim(
    path: Path, *, boundary: int, include_boundary: bool,
) -> tuple[str, int, dict[str, Any] | None]:
    if not path.is_file():
        raise ValueError(f"cooperbench_resume_evidence_missing:{path.name}")
    backup = path.with_name(path.name + f".pre-resume-{uuid4().hex[:12]}")
    shutil.copy2(path, backup)
    kept: list[str] = []
    dropped = 0
    first: dict[str, Any] | None = None
    with backup.open("r", encoding="utf-8") as stream:
        for line in stream:
            try:
                record = json.loads(line)
                tick = record["tick"]
                if isinstance(tick, bool) or not isinstance(tick, int):
                    raise ValueError("tick_invalid")
            except (KeyError, ValueError, TypeError, json.JSONDecodeError):
                dropped += 1
                continue
            if first is None:
                first = record
            if tick < boundary or (include_boundary and tick == boundary):
                kept.append(line)
            else:
                dropped += 1
    _atomic_jsonl(path, kept)
    return str(backup), dropped, first


def prepare_evidence(output_dir: Path, candidate: ResumeCandidate) -> dict[str, int]:
    """Back up and roll logs back to the last committed world tick."""
    trajectory_backup, trajectory_dropped, first = _trim(
        output_dir / "trajectory.jsonl", boundary=candidate.tick, include_boundary=True,
    )
    calls_backup, calls_dropped, _ = _trim(
        output_dir / "model_calls.jsonl", boundary=candidate.tick, include_boundary=False,
    )
    if not first or first.get("tick") != 0 or not isinstance(first.get("llm_usage"), dict):
        raise ValueError("cooperbench_resume_original_usage_baseline_missing")
    usage = first["llm_usage"]
    keys = (
        "calls", "failures", "retries", "provider_attempts", "prompt_tokens",
        "completion_tokens", "total_tokens", "cached_prompt_tokens",
    )
    baseline = {key: max(0, int(usage.get(key, 0) or 0)) for key in keys}
    receipt = {
        "schema_version": "cooperbench_checkpoint_resume_v1",
        "checkpoint": str(candidate.checkpoint),
        "checkpoint_tick": candidate.tick,
        "trajectory_backup": trajectory_backup,
        "model_calls_backup": calls_backup,
        "discarded_trajectory_rows": trajectory_dropped,
        "discarded_model_call_rows": calls_dropped,
        "skipped_corrupt_checkpoints": list(candidate.skipped_corrupt),
        "preflight_token_split_unavailable": True,
    }
    path = output_dir / "checkpoint_resume_receipt.json"
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return baseline


def prime_trajectory(recorder: Any, world: Any) -> None:
    for target, attribute in (
        ("decisions", "action_decisions"),
        ("actions", "action_log"),
        ("events", "events"),
        ("policy", "policy_trace"),
        ("perceptions", "_cooperbench_perception_trace"),
    ):
        recorder._cursors[target] = len(getattr(world, attribute, []) or [])
