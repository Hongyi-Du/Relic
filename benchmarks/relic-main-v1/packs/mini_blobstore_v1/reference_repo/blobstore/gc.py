from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .manifests import ManifestStore
from .objects import ObjectStore
from .uploads import UploadStore


@dataclass(frozen=True)
class GCPlan:
    candidates: tuple[str, ...]
    state_fingerprint: str


@dataclass(frozen=True)
class GCAudit:
    reachable: tuple[str, ...]
    orphaned: tuple[str, ...]
    corrupt: tuple[str, ...]


class GarbageCollector:
    def __init__(
        self,
        root: str | os.PathLike[str],
        objects: ObjectStore,
        manifests: ManifestStore,
        uploads: UploadStore,
    ):
        self.root = Path(root)
        self.objects = objects
        self.manifests = manifests
        self.uploads = uploads

    def _fingerprint(self) -> str:
        digest = hashlib.sha256()
        roots = [self.root / "manifests", self.root / "uploads"]
        for base in roots:
            if not base.exists():
                continue
            for path in sorted(item for item in base.rglob("*") if item.is_file()):
                relative = path.relative_to(self.root).as_posix().encode()
                payload = path.read_bytes()
                digest.update(len(relative).to_bytes(4, "big"))
                digest.update(relative)
                digest.update(hashlib.sha256(payload).digest())
        return digest.hexdigest()

    def reachable(self) -> set[str]:
        return self.manifests.referenced_ids() | self.uploads.incomplete_object_ids()

    def plan(self) -> GCPlan:
        reachable = self.reachable()
        candidates = tuple(
            object_id for object_id in self.objects.iter_ids() if object_id not in reachable
        )
        return GCPlan(candidates, self._fingerprint())

    def apply(self, plan: GCPlan) -> int:
        if plan.state_fingerprint != self._fingerprint():
            raise RuntimeError("stale garbage-collection plan")
        reachable = self.reachable()
        if any(object_id in reachable for object_id in plan.candidates):
            raise RuntimeError("plan contains reachable object")
        quarantine = self.root / "gc-quarantine"
        quarantine.mkdir(parents=True, exist_ok=True)
        moved = 0
        for object_id in plan.candidates:
            path = self.objects.path_for(object_id)
            if not path.exists():
                continue
            os.replace(path, quarantine / f"{object_id}.{time.time_ns()}")
            moved += 1
        directory_fd = os.open(quarantine, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return moved

    def audit(self) -> GCAudit:
        reachable = self.reachable()
        present = set(self.objects.iter_ids())
        corrupt = tuple(
            sorted(object_id for object_id in present if not self.objects.verify(object_id))
        )
        return GCAudit(
            tuple(sorted(reachable)),
            tuple(sorted(present - reachable)),
            corrupt,
        )
