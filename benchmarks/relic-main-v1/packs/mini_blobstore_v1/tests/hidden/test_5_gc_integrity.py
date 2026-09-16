"""Step 5 — Safe garbage collection

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
CASES = [('full_stack_regression', "\nfrom blobstore.objects import ObjectStore\nfrom blobstore.manifests import ManifestStore\nfrom blobstore.uploads import UploadStore\nfrom blobstore.replicas import ReplicaSet\no=ObjectStore(ROOT/'p'); m=ManifestStore(ROOT/'p',o); u=UploadStore(ROOT/'p',o,m); uid=u.begin('x'); u.upload_part(uid,1,b'a'); oid=u.complete(uid,1)['object_id']\nr=ObjectStore(ROOT/'r'); ReplicaSet(o,[r]).replicate_all(); assert r.verify(oid)\n"), ('reachability_manifest_and_upload', "\nfrom blobstore.objects import ObjectStore\nfrom blobstore.manifests import ManifestStore\nfrom blobstore.uploads import UploadStore\nfrom blobstore.gc import GarbageCollector\no=ObjectStore(ROOT); m=ManifestStore(ROOT,o); u=UploadStore(ROOT,o,m)\na=o.put_bytes(b'a'); b=o.put_bytes(b'b'); c=o.put_bytes(b'c'); m.create('keep',[a]); uid=u.begin('pending'); u.upload_part(uid,1,b'b')\ngc=GarbageCollector(ROOT,o,m,u); assert gc.reachable()=={a,b} and gc.plan().candidates==(c,)\n"), ('plan_is_sorted_read_only', "\nimport hashlib\nfrom blobstore.objects import ObjectStore\nfrom blobstore.manifests import ManifestStore\nfrom blobstore.uploads import UploadStore\nfrom blobstore.gc import GarbageCollector\no=ObjectStore(ROOT); [o.put_bytes(x) for x in (b'z',b'a',b'm')]; gc=GarbageCollector(ROOT,o,ManifestStore(ROOT,o),UploadStore(ROOT,o,ManifestStore(ROOT,o)))\nbefore={x:hashlib.sha256(o.path_for(x).read_bytes()).hexdigest() for x in o.iter_ids()}; p=gc.plan()\nassert p.candidates==tuple(sorted(p.candidates)) and before=={x:hashlib.sha256(o.path_for(x).read_bytes()).hexdigest() for x in o.iter_ids()}\n"), ('stale_plan_rejected', "\nfrom blobstore.objects import ObjectStore\nfrom blobstore.manifests import ManifestStore\nfrom blobstore.uploads import UploadStore\nfrom blobstore.gc import GarbageCollector\no=ObjectStore(ROOT); orphan=o.put_bytes(b'x'); m=ManifestStore(ROOT,o); u=UploadStore(ROOT,o,m); gc=GarbageCollector(ROOT,o,m,u); plan=gc.plan(); m.create('new',[orphan])\ntry: gc.apply(plan)\nexcept RuntimeError: pass\nelse: raise AssertionError('stale plan applied')\nassert o.exists(orphan)\n"), ('apply_quarantines_only_orphans', "\nfrom blobstore.objects import ObjectStore\nfrom blobstore.manifests import ManifestStore\nfrom blobstore.uploads import UploadStore\nfrom blobstore.gc import GarbageCollector\no=ObjectStore(ROOT); keep=o.put_bytes(b'k'); drop=o.put_bytes(b'd'); m=ManifestStore(ROOT,o); m.create('keep',[keep]); u=UploadStore(ROOT,o,m); gc=GarbageCollector(ROOT,o,m,u)\nassert gc.apply(gc.plan())==1 and o.exists(keep) and not o.exists(drop)\nassert any(p.name.startswith(drop) for p in (ROOT/'gc-quarantine').iterdir())\n"), ('audit_classifies_corruption', "\nfrom blobstore.objects import ObjectStore\nfrom blobstore.manifests import ManifestStore\nfrom blobstore.uploads import UploadStore\nfrom blobstore.gc import GarbageCollector\no=ObjectStore(ROOT); keep=o.put_bytes(b'k'); orphan=o.put_bytes(b'o'); m=ManifestStore(ROOT,o); m.create('keep',[keep]); o.path_for(keep).write_bytes(b'bad')\na=GarbageCollector(ROOT,o,m,UploadStore(ROOT,o,m)).audit()\nassert a.reachable==(keep,) and a.orphaned==(orphan,) and a.corrupt==(keep,)\n"), ('deterministic_repeat_plan', "\nfrom blobstore.objects import ObjectStore\nfrom blobstore.manifests import ManifestStore\nfrom blobstore.uploads import UploadStore\nfrom blobstore.gc import GarbageCollector\no=ObjectStore(ROOT); o.put_bytes(b'x'); m=ManifestStore(ROOT,o); u=UploadStore(ROOT,o,m); gc=GarbageCollector(ROOT,o,m,u)\nassert gc.plan()==gc.plan() and gc.audit()==gc.audit()\n")]


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
