"""World checkpoint / resume (faithful, pickle-based).

A checkpoint is the FULL ``OrgWorld`` object pickled at a given tick — agents, comm threads,
repo (branches/commits/PRs), episodes, proposals/tools/protocols, product substrate, growth,
budget, AND the external market/community. So a checkpoint is a complete, continuable state:
load it and keep stepping, from the end of a run OR from any mid-run tick.

The only non-picklable piece is the live LLM client (it holds sockets/threads), so it is
detached on save and re-attached on load (either an explicitly-passed client, or reloaded
from ``config/llm(.local).yaml``). Everything else is plain dataclasses / dicts and pickles
cleanly (verified: a built+stepped world has zero non-picklable top-level attrs).
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import pickle
import re
import stat
import tempfile
from pathlib import Path
from typing import Any, BinaryIO, Dict, Mapping, Optional, Tuple

CHECKPOINT_VERSION = 1
CHECKPOINT_SIDECAR_SCHEMA_VERSION = "orgenv_checkpoint_sidecar_v1"
CHECKPOINT_SIDECAR_SUFFIX = ".metadata.json"
_MAX_SIDECAR_BYTES = 16 * 1024 * 1024
_SHA256_RE = re.compile(r"[0-9a-f]{64}")

# live, non-picklable handles stripped before pickling and restored after / on load
_DETACH_ATTRS = ("llm_client",)


def _world_seed(world: Any) -> Optional[int]:
    sc = getattr(world, "scenario", None) or getattr(world, "scenario_config", None)
    return getattr(sc, "seed", None) if sc is not None else getattr(world, "seed", None)


def checkpoint_sidecar_path(path: str | os.PathLike[str]) -> Path:
    """Return the independent metadata path without opening the checkpoint."""

    checkpoint = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    return checkpoint.with_name(checkpoint.name + CHECKPOINT_SIDECAR_SUFFIX)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_handle(handle: BinaryIO) -> str:
    digest = hashlib.sha256()
    handle.seek(0)
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
    handle.seek(0)
    return digest.hexdigest()


def _open_regular_nofollow(path: Path) -> BinaryIO:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("checkpoint_file_invalid")
        return os.fdopen(descriptor, "rb")
    except BaseException:
        os.close(descriptor)
        raise


def _canonical_json_bytes(payload: Any) -> bytes:
    try:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("checkpoint_metadata_not_json_serializable") from exc


def _fsync_directory(directory: Path) -> None:
    if os.name != "posix":  # pragma: no cover - Windows compatibility.
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _new_temporary_path(target: Path) -> Path:
    descriptor, raw = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    os.close(descriptor)
    return Path(raw)


def _write_bytes_fsynced(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_TRUNC)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


def _replace_durable(temporary: Path, target: Path) -> None:
    os.replace(temporary, target)
    _fsync_directory(target.parent)


def _formal_checkpoint_context() -> bool:
    if str(os.environ.get("ORG_OSS_MODE") or "").strip().lower() == "formal":
        return True
    return any(
        os.environ.get(name)
        for name in (
            "ORG_SOURCE_PROVENANCE_FINGERPRINT",
            "ORG_MODEL_BINDING_FINGERPRINT",
            "ORG_EXECUTION_RESOURCE_BUDGET_FINGERPRINT",
            "ORG_CASE_PLAN_FINGERPRINT",
        )
    )


def _checkpoint_expected_values(
    *,
    expected_source_provenance_fingerprint: Optional[str],
    expected_model_binding_fingerprint: Optional[str],
    expected_resource_budget_fingerprint: Optional[str],
    expected_case_plan_fingerprint: Optional[str],
    expected_target_tick: Optional[int],
    expected_seed: Optional[int],
) -> dict[str, Any]:
    return {
        "source_provenance_fingerprint": (
            expected_source_provenance_fingerprint
            or os.environ.get("ORG_SOURCE_PROVENANCE_FINGERPRINT")
        ),
        "model_binding_fingerprint": (
            expected_model_binding_fingerprint
            or os.environ.get("ORG_MODEL_BINDING_FINGERPRINT")
        ),
        "resource_budget_fingerprint": (
            expected_resource_budget_fingerprint
            or os.environ.get("ORG_EXECUTION_RESOURCE_BUDGET_FINGERPRINT")
        ),
        "case_plan_fingerprint": (
            expected_case_plan_fingerprint
            or os.environ.get("ORG_CASE_PLAN_FINGERPRINT")
        ),
        "target_tick": (
            expected_target_tick
            if expected_target_tick is not None
            else (
                int(os.environ["ORG_EXPERIMENT_TARGET_TICK"])
                if os.environ.get("ORG_EXPERIMENT_TARGET_TICK")
                else None
            )
        ),
        "experiment_condition": os.environ.get("ORG_EXPERIMENT_CONDITION"),
        "action_selection_mode": os.environ.get("ORG_ACTION_SELECTION_MODE"),
        "seed": (
            expected_seed
            if expected_seed is not None
            else (
                int(os.environ["ORG_EXPERIMENT_PAIRED_SEED"])
                if os.environ.get("ORG_EXPERIMENT_PAIRED_SEED")
                else None
            )
        ),
    }


def _validate_checkpoint_identity(
    metadata: Mapping[str, Any],
    expected_values: Mapping[str, Any],
) -> None:
    for field, expected in expected_values.items():
        if expected not in (None, "") and metadata.get(field) != expected:
            raise ValueError(f"checkpoint_{field}_mismatch")


def _read_verified_sidecar(
    path: Path,
    checkpoint_handle: BinaryIO,
) -> tuple[dict[str, Any], dict[str, Any]]:
    sidecar = checkpoint_sidecar_path(path)
    try:
        sidecar_handle = _open_regular_nofollow(sidecar)
    except FileNotFoundError as exc:
        raise ValueError("checkpoint_sidecar_missing") from exc
    except OSError as exc:
        raise ValueError("checkpoint_sidecar_invalid") from exc
    with sidecar_handle:
        sidecar_size = os.fstat(sidecar_handle.fileno()).st_size
        if sidecar_size > _MAX_SIDECAR_BYTES:
            raise ValueError("checkpoint_sidecar_too_large")
        sidecar_bytes = sidecar_handle.read(_MAX_SIDECAR_BYTES + 1)
    if len(sidecar_bytes) > _MAX_SIDECAR_BYTES:
        raise ValueError("checkpoint_sidecar_too_large")
    if not sidecar_bytes:
        raise ValueError("checkpoint_sidecar_malformed")
    try:
        envelope = json.loads(sidecar_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("checkpoint_sidecar_malformed") from exc
    if not isinstance(envelope, dict):
        raise ValueError("checkpoint_sidecar_malformed")
    metadata = envelope.get("metadata")
    declared_hash = envelope.get("checkpoint_sha256")
    declared_metadata_hash = envelope.get("metadata_sha256")
    if (
        envelope.get("schema_version") != CHECKPOINT_SIDECAR_SCHEMA_VERSION
        or envelope.get("checkpoint_version") != CHECKPOINT_VERSION
        or envelope.get("checkpoint_filename") != path.name
        or not isinstance(metadata, dict)
        or not isinstance(declared_hash, str)
        or _SHA256_RE.fullmatch(declared_hash) is None
        or not isinstance(declared_metadata_hash, str)
        or _SHA256_RE.fullmatch(declared_metadata_hash) is None
    ):
        raise ValueError("checkpoint_sidecar_malformed")
    if metadata.get("version") != CHECKPOINT_VERSION:
        raise ValueError("checkpoint_sidecar_version_mismatch")
    metadata_hash = hashlib.sha256(_canonical_json_bytes(metadata)).hexdigest()
    if metadata_hash != declared_metadata_hash:
        raise ValueError("checkpoint_sidecar_metadata_hash_mismatch")
    try:
        declared_size = int(envelope.get("checkpoint_size_bytes", -1))
    except (TypeError, ValueError) as exc:
        raise ValueError("checkpoint_sidecar_malformed") from exc
    if os.fstat(checkpoint_handle.fileno()).st_size != declared_size:
        raise ValueError("checkpoint_size_mismatch")
    if _sha256_handle(checkpoint_handle) != declared_hash:
        raise ValueError("checkpoint_sha256_mismatch")
    return envelope, metadata


def _llm_checkpoint_state(client: Any) -> Optional[Dict[str, Any]]:
    """Capture cumulative, non-secret provider evidence needed after a resume."""
    if client is None:
        return None
    prompt_audit = (
        client.prompt_visibility_audit()
        if callable(getattr(client, "prompt_visibility_audit", None))
        else None
    )
    inner = getattr(client, "inner", client)
    if not all(
        hasattr(inner, name)
        for name in ("calls", "failures", "usage_totals")
    ):
        return None

    def _count(name: str) -> int:
        try:
            return max(0, int(getattr(inner, name, 0) or 0))
        except (TypeError, ValueError):
            return 0

    usage = getattr(inner, "usage_totals", {})
    response_models = getattr(inner, "response_model_counts", {})
    try:
        inner_stats = dict(inner.stats())
    except Exception:
        inner_stats = {}
    return {
        "schema_version": "orgenv_llm_checkpoint_state_v1",
        "provider": str(getattr(inner, "provider", "") or ""),
        "calls": _count("calls"),
        "failures": _count("failures"),
        "retries": _count("retries"),
        "provider_attempts": _count("provider_attempts"),
        "provider_transport": {
            "provider_successes": max(
                0, int(inner_stats.get("provider_successes", 0) or 0)
            ),
            "provider_failure_counts": dict(
                inner_stats.get("provider_failure_counts", {}) or {}
            ),
            "provider_exception_counts": dict(
                inner_stats.get("provider_exception_counts", {}) or {}
            ),
            "provider_status_counts": dict(
                inner_stats.get("provider_status_counts", {}) or {}
            ),
            "last_provider_failure": inner_stats.get("last_provider_failure"),
        },
        "resource_denials": max(
            0,
            int(getattr(client, "resource_denials", 0) or 0),
        ),
        "prompt_visibility_denials": max(
            0,
            int(getattr(client, "prompt_visibility_denials", 0) or 0),
        ),
        "usage_totals": {
            str(key): max(0, int(value or 0))
            for key, value in (
                usage.items() if isinstance(usage, Mapping) else ()
            )
        },
        "response_model_counts": {
            str(key): max(0, int(value or 0))
            for key, value in (
                response_models.items()
                if isinstance(response_models, Mapping)
                else ()
            )
        },
        "response_id_digest": str(
            getattr(inner, "response_id_digest", "") or ""
        ),
        "prompt_visibility_audit": prompt_audit,
    }


def _restore_llm_checkpoint_state(client: Any, state: Any) -> None:
    """Restore cumulative counters without restoring sockets, auth, or prompts."""
    if client is None or not isinstance(state, Mapping):
        return
    if state.get("schema_version") != "orgenv_llm_checkpoint_state_v1":
        raise ValueError("checkpoint_llm_state_malformed")
    observed_provider = str(getattr(client, "provider", "") or "")
    saved_provider = str(state.get("provider", "") or "")
    if saved_provider and observed_provider and saved_provider != observed_provider:
        raise ValueError("checkpoint_llm_provider_mismatch")
    for name in ("calls", "failures", "retries", "provider_attempts"):
        setattr(client, name, max(0, int(state.get(name, 0) or 0)))
    usage = state.get("usage_totals")
    models = state.get("response_model_counts")
    client.usage_totals = {
        str(key): max(0, int(value or 0))
        for key, value in (usage.items() if isinstance(usage, Mapping) else ())
    }
    client.response_model_counts = {
        str(key): max(0, int(value or 0))
        for key, value in (models.items() if isinstance(models, Mapping) else ())
    }
    client.response_id_digest = str(state.get("response_id_digest", "") or "")
    inner = getattr(client, "inner", client)
    restore_transport = getattr(inner, "restore_provider_transport_state", None)
    if callable(restore_transport):
        transport = state.get("provider_transport")
        restore_transport(transport if isinstance(transport, Mapping) else {})


def save_world_checkpoint(
    world: Any,
    path: str,
    *,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Durably save a checkpoint and independently verifiable metadata."""

    target = Path(os.path.abspath(os.path.expanduser(path)))
    target.parent.mkdir(parents=True, exist_ok=True)
    sidecar = checkpoint_sidecar_path(target)
    detached: Dict[str, Any] = {}
    for a in _DETACH_ATTRS:
        if hasattr(world, a):
            detached[a] = getattr(world, a)
            setattr(world, a, None)
    llm_state = _llm_checkpoint_state(detached.get("llm_client"))
    tick = int(getattr(world, "world_tick", 0))
    checkpoint_temporary: Path | None = None
    sidecar_temporary: Path | None = None
    try:
        target_tick_raw = os.environ.get("ORG_EXPERIMENT_TARGET_TICK")
        metadata = {
            "version": CHECKPOINT_VERSION,
            "tick": tick,
            "seed": _world_seed(world),
            "llm_decides_actions": bool(getattr(world, "llm_decides_actions", False)),
            "approval_mode": getattr(world, "approval_mode", None),
            "created": datetime.datetime.now().isoformat(timespec="seconds"),
            "experiment_condition": str(
                os.environ.get("ORG_EXPERIMENT_CONDITION")
                or getattr(world, "experiment_condition", "")
                or ""
            ),
            "action_selection_mode": str(
                os.environ.get("ORG_ACTION_SELECTION_MODE")
                or getattr(world, "action_selection_mode", "")
                or ""
            ),
            "case_plan_fingerprint": os.environ.get(
                "ORG_CASE_PLAN_FINGERPRINT"
            ),
            "target_tick": (
                int(target_tick_raw) if target_tick_raw not in (None, "") else None
            ),
            "source_provenance_fingerprint": os.environ.get(
                "ORG_SOURCE_PROVENANCE_FINGERPRINT"
            ),
            "model_binding_fingerprint": os.environ.get(
                "ORG_MODEL_BINDING_FINGERPRINT"
            ),
            "resource_budget_fingerprint": str(
                os.environ.get(
                    "ORG_EXECUTION_RESOURCE_BUDGET_FINGERPRINT"
                )
                or getattr(
                    getattr(world, "experiment_resource_budget", None),
                    "fingerprint",
                    "",
                )
                or ""
            ),
            "llm_checkpoint_state": llm_state,
            "meta": dict(meta or {}),
        }
        metadata = json.loads(_canonical_json_bytes(metadata))
        payload = {**metadata, "world": world}
        checkpoint_temporary = _new_temporary_path(target)
        with checkpoint_temporary.open("wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
            fh.flush()
            os.fsync(fh.fileno())
        checkpoint_hash = _sha256_file(checkpoint_temporary)
        checkpoint_size_bytes = checkpoint_temporary.stat().st_size
        envelope = {
            "schema_version": CHECKPOINT_SIDECAR_SCHEMA_VERSION,
            "checkpoint_version": CHECKPOINT_VERSION,
            "checkpoint_filename": target.name,
            "checkpoint_sha256": checkpoint_hash,
            "checkpoint_size_bytes": checkpoint_size_bytes,
            "metadata_sha256": hashlib.sha256(
                _canonical_json_bytes(metadata)
            ).hexdigest(),
            "metadata": metadata,
        }
        sidecar_temporary = _new_temporary_path(sidecar)
        _write_bytes_fsynced(
            sidecar_temporary,
            _canonical_json_bytes(envelope) + b"\n",
        )
        _replace_durable(checkpoint_temporary, target)
        checkpoint_temporary = None
        _replace_durable(sidecar_temporary, sidecar)
        sidecar_temporary = None
    finally:
        for a, v in detached.items():
            setattr(world, a, v)
        for temporary in (checkpoint_temporary, sidecar_temporary):
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    size_mb = round(checkpoint_size_bytes / 1e6, 2)
    return {
        "path": str(target),
        "sidecar_path": str(sidecar),
        "sha256": checkpoint_hash,
        "tick": tick,
        "version": CHECKPOINT_VERSION,
        "size_mb": size_mb,
    }


def load_world_checkpoint(
    path: str,
    *,
    llm_client: Any = None,
    load_llm: bool = False,
    decides: Optional[bool] = None,
    expected_source_provenance_fingerprint: Optional[str] = None,
    expected_model_binding_fingerprint: Optional[str] = None,
    expected_resource_budget_fingerprint: Optional[str] = None,
    expected_case_plan_fingerprint: Optional[str] = None,
    expected_target_tick: Optional[int] = None,
    expected_seed: Optional[int] = None,
    world_pre_attach_validator: Any = None,
) -> Tuple[Any, Dict[str, Any]]:
    """Unpickle a world checkpoint and re-attach a live LLM client.

    - ``llm_client`` (if given) is attached directly.
    - else if ``load_llm`` is True, a client is reloaded from config/llm(.local).yaml.
    - ``decides`` overrides ``llm_decides_actions``; otherwise the checkpoint's value is kept
      (only when a client is actually attached).
    Returns ``(world, payload_meta_without_world)``.
    """
    checkpoint = Path(os.path.abspath(os.path.expanduser(path)))
    expected_values = _checkpoint_expected_values(
        expected_source_provenance_fingerprint=(
            expected_source_provenance_fingerprint
        ),
        expected_model_binding_fingerprint=expected_model_binding_fingerprint,
        expected_resource_budget_fingerprint=(
            expected_resource_budget_fingerprint
        ),
        expected_case_plan_fingerprint=expected_case_plan_fingerprint,
        expected_target_tick=expected_target_tick,
        expected_seed=expected_seed,
    )
    try:
        checkpoint_handle = _open_regular_nofollow(checkpoint)
    except OSError as exc:
        if isinstance(exc, FileNotFoundError):
            raise
        raise ValueError("checkpoint_file_invalid") from exc
    with checkpoint_handle:
        sidecar_metadata: dict[str, Any] | None
        try:
            _, sidecar_metadata = _read_verified_sidecar(
                checkpoint,
                checkpoint_handle,
            )
        except ValueError as exc:
            if str(exc) != "checkpoint_sidecar_missing":
                raise
            if _formal_checkpoint_context():
                raise ValueError("legacy_checkpoint_sidecar_required") from exc
            sidecar_metadata = None
        if sidecar_metadata is not None:
            _validate_checkpoint_identity(sidecar_metadata, expected_values)
        checkpoint_handle.seek(0)
        payload = pickle.load(checkpoint_handle)
    if not isinstance(payload, dict) or payload.get("version") != CHECKPOINT_VERSION:
        raise ValueError("unsupported_or_malformed_checkpoint")
    if "world" not in payload:
        raise ValueError("checkpoint_world_missing")
    payload_metadata = {key: value for key, value in payload.items() if key != "world"}
    if (
        sidecar_metadata is not None
        and _canonical_json_bytes(payload_metadata)
        != _canonical_json_bytes(sidecar_metadata)
    ):
        raise ValueError("checkpoint_sidecar_payload_metadata_mismatch")
    _validate_checkpoint_identity(payload_metadata, expected_values)
    world = payload["world"]
    if world_pre_attach_validator is not None:
        if not callable(world_pre_attach_validator):
            raise TypeError("checkpoint_world_pre_attach_validator_not_callable")
        # This hook runs after the authenticated payload/identity boundary but
        # before config loading, usage restoration, metering attachment, loop
        # rebinding, or any mutation of the detached world.
        world_pre_attach_validator(world, payload_metadata)

    client = llm_client
    cfg_decides = None
    if client is None and load_llm:
        try:
            from environments.org_env.llm.config import load_org_llm_client
            client, cfg_decides = load_org_llm_client()
        except Exception:
            client = None
    _restore_llm_checkpoint_state(client, payload.get("llm_checkpoint_state"))
    from environments.org_env.experiments.resources import attach_metered_llm_client
    world.llm_client = attach_metered_llm_client(world, client)
    if (
        world.llm_client is not None
        and isinstance(payload.get("llm_checkpoint_state"), Mapping)
    ):
        prompt_audit = (payload.get("llm_checkpoint_state") or {}).get(
            "prompt_visibility_audit"
        )
        if prompt_audit is not None:
            restore = getattr(
                world.llm_client,
                "restore_prompt_visibility_audit",
                None,
            )
            if not callable(restore):
                raise ValueError("checkpoint_prompt_visibility_auditor_missing")
            restore(prompt_audit)
        world.llm_client.prompt_visibility_denials = max(
            0,
            int(
                (payload.get("llm_checkpoint_state") or {}).get(
                    "prompt_visibility_denials",
                    0,
                )
                or 0
            ),
        )
        world.llm_client.resource_denials = max(
            0,
            int(
                (payload.get("llm_checkpoint_state") or {}).get(
                    "resource_denials",
                    0,
                )
                or 0
            ),
        )
    if getattr(world, "_loop", None) is not None:
        from environments.org_env.experiments.controlled_execution import (
            ExperimentControlledExecutionAdapter,
        )
        world._loop["execution"] = ExperimentControlledExecutionAdapter()

    if decides is not None:
        world.llm_decides_actions = bool(decides)
    elif client is not None:
        world.llm_decides_actions = bool(cfg_decides if cfg_decides is not None
                                         else payload.get("llm_decides_actions", False))
    else:
        world.llm_decides_actions = False

    info = {k: v for k, v in payload.items() if k != "world"}
    return world, info


def checkpoint_info(path: str) -> Dict[str, Any]:
    """Read verified independent metadata without deserializing the checkpoint."""

    checkpoint = Path(os.path.abspath(os.path.expanduser(path)))
    try:
        checkpoint_handle = _open_regular_nofollow(checkpoint)
    except OSError as exc:
        if isinstance(exc, FileNotFoundError):
            raise
        raise ValueError("checkpoint_file_invalid") from exc
    with checkpoint_handle:
        _, metadata = _read_verified_sidecar(checkpoint, checkpoint_handle)
    return dict(metadata)


__all__ = [
    "CHECKPOINT_SIDECAR_SCHEMA_VERSION",
    "CHECKPOINT_SIDECAR_SUFFIX",
    "CHECKPOINT_VERSION",
    "checkpoint_info",
    "checkpoint_sidecar_path",
    "load_world_checkpoint",
    "save_world_checkpoint",
]
