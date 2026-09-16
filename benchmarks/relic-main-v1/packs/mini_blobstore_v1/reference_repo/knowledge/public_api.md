# Public API contract — mini-blobstore

This file defines the callable surface that is part of the task contract.
Hidden tests may exercise additional semantic boundaries, but they must not
require undisclosed public names or signatures.

## Step 1 — content-addressing

### `blobstore.objects`

- Class `InvalidObjectId()`
  - Bases: `ValueError`
- Class `CorruptObjectError()`
  - Bases: `RuntimeError`
- Class `ObjectStore(root: str | os.PathLike[str])`
  - Observable attributes: `objects`, `root`
  - `validate_id(object_id: str) -> None`
  - `path_for(object_id: str) -> Path`
  - `put_bytes(data: bytes) -> str`
  - `put_stream(source: BinaryIO) -> str`
  - `get(object_id: str) -> bytes`
  - `exists(object_id: str) -> bool`
  - `verify(object_id: str) -> bool`
  - `iter_ids() -> Iterator[str]`

Published behavior cases: `digest_and_roundtrip`, `sharded_path`

Public smoke: `python public_contract_tests/step_01_smoke.py`

## Step 2 — manifest-store

### `blobstore.manifests`

- Class `ManifestConflictError()`
  - Bases: `RuntimeError`
- Class `Manifest(name: str, objects: tuple[str, ...], metadata: dict)`
  - Observable attributes: `metadata`, `name`, `objects`
- Class `ManifestStore(root: str | os.PathLike[str], objects: ObjectStore)`
  - Observable attributes: `directory`, `objects`, `root`
  - `validate_name(name: str) -> None`
  - `path_for(name: str) -> Path`
  - `create(name: str, object_ids: list[str], metadata: dict | None = ...) -> Manifest`
  - `get(name: str) -> Manifest`
  - `delete(name: str) -> bool`
  - `iter_names() -> Iterator[str]`
  - `referenced_ids() -> set[str]`

Published behavior cases: `create_get_canonical`, `unsafe_names_rejected`

Public smoke: `python public_contract_tests/step_02_smoke.py`

## Step 3 — multipart-upload

### `blobstore.uploads`

- Class `UploadConflictError()`
  - Bases: `RuntimeError`
- Class `UploadStore(root: str | os.PathLike[str], objects: ObjectStore, manifests: ManifestStore)`
  - Observable attributes: `directory`, `manifests`, `objects`, `root`
  - `begin(target: str, metadata: dict | None = ...) -> str`
  - `upload_part(upload_id: str, number: int, data: bytes) -> str`
  - `complete(upload_id: str, part_count: int) -> dict`
  - `incomplete_object_ids() -> set[str]`

Published behavior cases: `multipart_complete_order`, `part_retry_and_conflict`

Public smoke: `python public_contract_tests/step_03_smoke.py`

## Step 4 — replication-repair

### `blobstore.replicas`

- Class `ReplicaStatus(object_id: str, states: tuple[str, ...])`
  - Observable attributes: `object_id`, `states`
- Class `ReplicaSet(primary: ObjectStore, replicas: list[ObjectStore])`
  - Observable attributes: `primary`, `replicas`
  - `audit(object_id: str) -> ReplicaStatus`
  - `repair(object_id: str, replica_index: int) -> str`
  - `replicate_all() -> int`

Published behavior cases: `audit_states`, `replicate_missing`

Public smoke: `python public_contract_tests/step_04_smoke.py`

## Step 5 — gc-integrity

### `blobstore.gc`

- Class `GCPlan(candidates: tuple[str, ...], state_fingerprint: str)`
  - Observable attributes: `candidates`, `state_fingerprint`
- Class `GCAudit(reachable: tuple[str, ...], orphaned: tuple[str, ...], corrupt: tuple[str, ...])`
  - Observable attributes: `corrupt`, `orphaned`, `reachable`
- Class `GarbageCollector(root: str | os.PathLike[str], objects: ObjectStore, manifests: ManifestStore, uploads: UploadStore)`
  - Observable attributes: `manifests`, `objects`, `root`, `uploads`
  - `reachable() -> set[str]`
  - `plan() -> GCPlan`
  - `apply(plan: GCPlan) -> int`
  - `audit() -> GCAudit`

Published behavior cases: `reachability_manifest_and_upload`, `plan_is_sorted_read_only`

Public smoke: `python public_contract_tests/step_05_smoke.py`
