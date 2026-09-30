"""Resolve a verified continuation identity before the pair rendezvous executes."""
from dataclasses import replace
from pathlib import Path

from environments.org_env.runtime_adapter.checkpoint import checkpoint_info


def resolve_request(request):
    request = request.validated()
    root = Path(request.log_dir).resolve() / "orgenv_b3_two_agent"
    paths = sorted(
        (p for p in (root / "checkpoints").glob("t*.pkl") if p.stem[1:].isdigit()),
        key=lambda p: int(p.stem[1:]), reverse=True,
    )
    if not paths:
        return request
    errors = []
    for path in paths:
        try:
            info = checkpoint_info(str(path))
        except (OSError, ValueError) as error:
            errors.append(f"{path.name}:{type(error).__name__}")
            continue
        meta = info.get("meta") or {}
        expected = {
            "schema_version": "orgenv_cooperbench_pair_checkpoint_v1",
            "agents": list(request.agents), "tasks": dict(request.tasks),
            "image": request.image, "model_name": request.model_name,
            "target_ticks": request.ticks,
        }
        if any(meta.get(k) != v for k, v in expected.items()):
            raise ValueError("cooperbench_resume_pair_identity_mismatch:" + path.name)
        identity = meta.get("run_id")
        if not isinstance(identity, str) or not identity:
            raise ValueError("cooperbench_resume_original_run_id_missing")
        return replace(request, run_id=identity).validated()
    raise ValueError("cooperbench_resume_no_verified_checkpoint:" + ";".join(errors))
