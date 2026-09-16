# Content-addressed blob-store contract

## Immutable objects

An object ID is the lowercase SHA-256 digest of its exact bytes. Objects live
under `objects/<first two hex>/<remaining hex>`. A write first streams into a
temporary file while hashing, fsyncs it, and atomically publishes it. Repeating
the same write is idempotent. Existing bytes with a mismatching digest indicate
corruption and must never be overwritten silently.

## Manifests

A manifest gives a validated name to an ordered list of object IDs plus a
metadata object. Names contain ASCII letters, digits, dot, underscore, slash,
and hyphen, but no empty, dot, or parent segments. Manifests are canonical JSON
with sorted keys and compact separators. Publication is an fsynced temporary
file followed by rename and parent-directory fsync. Creating a manifest checks
that all referenced objects exist and verify.

## Multipart uploads

An upload has a random hexadecimal ID, immutable target name, and numbered
parts starting at one. Each part is content-addressed and its metadata is
published atomically. Uploading the same part bytes twice is idempotent;
different bytes for the same part are rejected. Completion requires exactly
parts `1..N`, concatenates them in order into the object store, creates the
target manifest, and records a durable completed result. Retrying completion
returns the same result.

## Replication and repair

A replica set has one primary store and one or more replica stores. Replication
copies only objects whose digest is missing, using the destination store's
normal verified publication path. Audit classifies each object per replica as
healthy, missing, or corrupt. Repair may copy only from a healthy source and
must verify the destination afterward. Corrupt files are quarantined before
replacement.

## Garbage collection

Reachable objects are those referenced by valid manifests and incomplete
multipart parts. Collection has a plan phase and an apply phase. A plan records
the sorted candidate IDs plus a fingerprint of the current manifest/upload
roots. Apply refuses a stale plan. Candidates move to a quarantine directory
before deletion so interruption cannot make a reachable object disappear.
Dry-run and audit operations are read-only and deterministic.
