from __future__ import annotations

import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from .objects import ObjectStore


@dataclass(frozen=True)
class ReplicaStatus:
    object_id: str
    states: tuple[str, ...]


class ReplicaSet:
    def __init__(self, primary: ObjectStore, replicas: list[ObjectStore]):
        if not replicas:
            raise ValueError("at least one replica is required")
        self.primary = primary
        self.replicas = list(replicas)

    @staticmethod
    def _state(store: ObjectStore, object_id: str) -> str:
        if not store.exists(object_id):
            return "missing"
        return "healthy" if store.verify(object_id) else "corrupt"

    def audit(self, object_id: str) -> ReplicaStatus:
        self.primary.validate_id(object_id)
        stores = [self.primary, *self.replicas]
        return ReplicaStatus(
            object_id,
            tuple(self._state(store, object_id) for store in stores),
        )

    @staticmethod
    def _quarantine(store: ObjectStore, object_id: str) -> None:
        path = store.path_for(object_id)
        if not path.exists():
            return
        quarantine = store.root / "quarantine"
        quarantine.mkdir(parents=True, exist_ok=True)
        target = quarantine / f"{object_id}.{time.time_ns()}"
        os.replace(path, target)

    def repair(self, object_id: str, replica_index: int) -> str:
        if replica_index < 0 or replica_index >= len(self.replicas):
            raise IndexError("replica index out of range")
        sources = [self.primary, *self.replicas]
        source = next(
            (store for store in sources if self._state(store, object_id) == "healthy"),
            None,
        )
        if source is None:
            raise RuntimeError("no healthy source")
        destination = self.replicas[replica_index]
        state = self._state(destination, object_id)
        if state == "healthy":
            return object_id
        if state == "corrupt":
            self._quarantine(destination, object_id)
        published = destination.put_bytes(source.get(object_id))
        if published != object_id or not destination.verify(object_id):
            raise RuntimeError("replica verification failed")
        return object_id

    def replicate_all(self) -> int:
        repaired = 0
        for object_id in self.primary.iter_ids():
            for index, replica in enumerate(self.replicas):
                if self._state(replica, object_id) != "healthy":
                    self.repair(object_id, index)
                    repaired += 1
        return repaired
