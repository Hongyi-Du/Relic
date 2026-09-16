"""Step 3 — Multipart uploads

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
CASES = [('prior_regression', "\nfrom blobstore.objects import ObjectStore\nfrom blobstore.manifests import ManifestStore\no=ObjectStore(ROOT); oid=o.put_bytes(b'x'); assert ManifestStore(ROOT,o).create('x',[oid]).objects==(oid,)\n"), ('multipart_complete_order', "\nfrom blobstore.objects import ObjectStore\nfrom blobstore.manifests import ManifestStore\nfrom blobstore.uploads import UploadStore\no=ObjectStore(ROOT); m=ManifestStore(ROOT,o); u=UploadStore(ROOT,o,m); uid=u.begin('release/v1',{'k':'v'})\nu.upload_part(uid,2,b'world'); u.upload_part(uid,1,b'hello ')\nr=u.complete(uid,2); assert o.get(r['object_id'])==b'hello world' and m.get('release/v1').metadata=={'k':'v'}\n"), ('part_retry_and_conflict', "\nfrom blobstore.objects import ObjectStore\nfrom blobstore.manifests import ManifestStore\nfrom blobstore.uploads import UploadStore,UploadConflictError\no=ObjectStore(ROOT); u=UploadStore(ROOT,o,ManifestStore(ROOT,o)); uid=u.begin('x')\na=u.upload_part(uid,1,b'a'); assert u.upload_part(uid,1,b'a')==a\ntry: u.upload_part(uid,1,b'b')\nexcept UploadConflictError: pass\nelse: raise AssertionError('changed part accepted')\n"), ('contiguous_parts_required', "\nfrom blobstore.objects import ObjectStore\nfrom blobstore.manifests import ManifestStore\nfrom blobstore.uploads import UploadStore\no=ObjectStore(ROOT); u=UploadStore(ROOT,o,ManifestStore(ROOT,o)); uid=u.begin('x'); u.upload_part(uid,2,b'b')\ntry: u.complete(uid,2)\nexcept ValueError: pass\nelse: raise AssertionError('gap accepted')\n"), ('completion_retry_durable', "\nfrom blobstore.objects import ObjectStore\nfrom blobstore.manifests import ManifestStore\nfrom blobstore.uploads import UploadStore\no=ObjectStore(ROOT); m=ManifestStore(ROOT,o); u=UploadStore(ROOT,o,m); uid=u.begin('x'); u.upload_part(uid,1,b'a')\na=u.complete(uid,1); b=UploadStore(ROOT,o,m).complete(uid,1); assert a==b\n"), ('completed_upload_is_immutable', "\nfrom blobstore.objects import ObjectStore\nfrom blobstore.manifests import ManifestStore\nfrom blobstore.uploads import UploadStore,UploadConflictError\no=ObjectStore(ROOT); m=ManifestStore(ROOT,o); u=UploadStore(ROOT,o,m); uid=u.begin('x'); u.upload_part(uid,1,b'a'); u.complete(uid,1)\nfor action in [lambda:u.upload_part(uid,2,b'b'),lambda:u.complete(uid,2)]:\n    try: action()\n    except UploadConflictError: pass\n    else: raise AssertionError('completed upload changed')\n"), ('incomplete_parts_are_reachable', "\nfrom blobstore.objects import ObjectStore\nfrom blobstore.manifests import ManifestStore\nfrom blobstore.uploads import UploadStore\no=ObjectStore(ROOT); u=UploadStore(ROOT,o,ManifestStore(ROOT,o)); a=u.begin('a'); b=u.begin('b')\nx=u.upload_part(a,1,b'x'); y=u.upload_part(b,1,b'y'); u.complete(a,1)\nassert u.incomplete_object_ids()=={y} and x not in u.incomplete_object_ids()\n")]


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
