from __future__ import annotations
import hashlib, json, dataclasses
from enum import Enum
from typing import Any, Mapping, Literal

Namespace = Literal["shard", "comp", "clus"]

def _canonicalize(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        if obj != obj or obj == float("inf") or obj == float("-inf"):
            raise ValueError("NaN/Infinity not allowed in canonical payload")
        return obj
    if isinstance(obj, Enum):
        return obj.value
    if dataclasses.is_dataclass(obj):
        return _canonicalize(dataclasses.asdict(obj))
    if isinstance(obj, Mapping):
        return {str(k): _canonicalize(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [_canonicalize(x) for x in obj]
    raise TypeError(f"cannot canonicalize {type(obj).__name__}")

def canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(_canonicalize(payload), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)

def mint_id(payload: Mapping[str, Any], *, namespace: Namespace) -> str:
    digest = hashlib.sha256(canonical_json(payload).encode("ascii")).hexdigest()[:6]
    return f"{namespace}:sha256:{digest}"
