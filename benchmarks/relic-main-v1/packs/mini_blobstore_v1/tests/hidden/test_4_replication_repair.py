"""Step 4 — Replication audit and repair

Generated from the task's own cases.py; the bodies are verbatim.
"""
from __future__ import annotations

import pathlib
import subprocess
import sys
import tempfile

import pytest

PACKAGE = 'blobstore'


def _workspace() -> pathlib.Path:
    """Find the tree under test, which does not always sit beside this file.

    Copied into a candidate repo, the tree is two levels up. Run by the
    evaluator, this suite is mounted read-only at /evaluator/hidden while the
    candidate is mounted at /workspace, so two levels up is the filesystem root
    and every case dies importing the package. The cases run under `-I`, which
    ignores PYTHONPATH, so the path named here is the only one they get.

    Naming the package rather than a depth resolves both layouts.
    """
    here = pathlib.Path(__file__).resolve()
    for candidate in (pathlib.Path('/workspace'), *here.parents):
        if (candidate / PACKAGE).is_dir():
            return candidate
    return here.parents[2]


WORKSPACE = _workspace()
CASES = [('prior_regression', "\nfrom blobstore.objects import ObjectStore\nfrom blobstore.manifests import ManifestStore\nfrom blobstore.uploads import UploadStore\no=ObjectStore(ROOT/'p'); m=ManifestStore(ROOT/'p',o); u=UploadStore(ROOT/'p',o,m); uid=u.begin('x'); u.upload_part(uid,1,b'a'); assert u.complete(uid,1)\n"), ('audit_states', "\nfrom blobstore.objects import ObjectStore\nfrom blobstore.replicas import ReplicaSet\np=ObjectStore(ROOT/'p'); a=ObjectStore(ROOT/'a'); b=ObjectStore(ROOT/'b'); oid=p.put_bytes(b'x'); a.put_bytes(b'x'); b.put_bytes(b'x'); b.path_for(oid).write_bytes(b'bad')\nr=ReplicaSet(p,[a,b]).audit(oid); assert r.states==('healthy','healthy','corrupt')\n"), ('replicate_missing', "\nfrom blobstore.objects import ObjectStore\nfrom blobstore.replicas import ReplicaSet\np=ObjectStore(ROOT/'p'); r=ObjectStore(ROOT/'r'); ids=[p.put_bytes(x) for x in (b'a',b'b')]\nassert ReplicaSet(p,[r]).replicate_all()==2 and all(r.verify(x) for x in ids)\n"), ('repair_quarantines_corrupt', "\nfrom blobstore.objects import ObjectStore\nfrom blobstore.replicas import ReplicaSet\np=ObjectStore(ROOT/'p'); r=ObjectStore(ROOT/'r'); oid=p.put_bytes(b'good'); r.put_bytes(b'good'); r.path_for(oid).write_bytes(b'bad')\nReplicaSet(p,[r]).repair(oid,0)\nassert r.verify(oid) and len(list((r.root/'quarantine').iterdir()))==1\n"), ('healthy_replica_not_mutated', "\nfrom blobstore.objects import ObjectStore\nfrom blobstore.replicas import ReplicaSet\np=ObjectStore(ROOT/'p'); r=ObjectStore(ROOT/'r'); oid=p.put_bytes(b'x'); r.put_bytes(b'x'); before=r.path_for(oid).stat().st_mtime_ns\nReplicaSet(p,[r]).repair(oid,0); assert r.path_for(oid).stat().st_mtime_ns==before\n"), ('no_healthy_source_fails', "\nfrom blobstore.objects import ObjectStore\nfrom blobstore.replicas import ReplicaSet\np=ObjectStore(ROOT/'p'); r=ObjectStore(ROOT/'r'); oid=p.put_bytes(b'x'); p.path_for(oid).write_bytes(b'bad')\ntry: ReplicaSet(p,[r]).repair(oid,0)\nexcept RuntimeError: pass\nelse: raise AssertionError('repaired without source')\n"), ('bounds_and_determinism', "\nfrom blobstore.objects import ObjectStore\nfrom blobstore.replicas import ReplicaSet\np=ObjectStore(ROOT/'p'); r=ObjectStore(ROOT/'r'); oid=p.put_bytes(b'x'); rs=ReplicaSet(p,[r])\nassert rs.audit(oid)==rs.audit(oid)\ntry: rs.repair(oid,1)\nexcept IndexError: pass\nelse: raise AssertionError('bad replica accepted')\n")]


def _run(body: str) -> None:
    with tempfile.TemporaryDirectory(prefix='case-') as raw:
        root = pathlib.Path(raw)
        prefix = (
            'import pathlib,sys\n'
            f'sys.path.insert(0,{str(WORKSPACE)!r})\n'
            f'ROOT=pathlib.Path({str(root)!r})\n'
        )
        done = subprocess.run(
            [sys.executable, '-I', '-B', '-c', prefix + body],
            cwd=root, text=True, capture_output=True, timeout=20)
    if done.returncode != 0:
        pytest.fail(done.stderr.strip()[-1500:] or 'case failed')


@pytest.mark.parametrize('name,body', CASES, ids=[c[0] for c in CASES])
def test_case(name, body):
    _run(body)
