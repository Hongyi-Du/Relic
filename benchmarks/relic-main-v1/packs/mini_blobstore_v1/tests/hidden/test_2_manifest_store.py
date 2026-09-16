"""Step 2 — Named manifests

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
CASES = [('object_regression', "\nfrom blobstore.objects import ObjectStore\ns=ObjectStore(ROOT); oid=s.put_bytes(b'x'); assert s.verify(oid)\n"), ('create_get_canonical', "\nimport json\nfrom blobstore.objects import ObjectStore\nfrom blobstore.manifests import ManifestStore\no=ObjectStore(ROOT); oid=o.put_bytes(b'x'); m=ManifestStore(ROOT,o)\nr=m.create('team/data',[oid],{'z':1,'a':2}); assert r.objects==(oid,)\np=m.path_for('team/data'); assert p.read_bytes()==(json.dumps(json.loads(p.read_text()),sort_keys=True,separators=(',',':'))+'\\n').encode()\n"), ('unsafe_names_rejected', "\nfrom blobstore.objects import ObjectStore\nfrom blobstore.manifests import ManifestStore\nm=ManifestStore(ROOT,ObjectStore(ROOT))\nfor name in ['', '/abs', '../x', 'a/../b', 'a//b']:\n    try: m.create(name,[])\n    except ValueError: pass\n    else: raise AssertionError(name)\nassert not list((ROOT/'manifests').rglob('*.json'))\n"), ('missing_or_corrupt_reference_rejected', "\nfrom blobstore.objects import ObjectStore\nfrom blobstore.manifests import ManifestStore\no=ObjectStore(ROOT); oid=o.put_bytes(b'x'); m=ManifestStore(ROOT,o); o.path_for(oid).write_bytes(b'bad')\ntry: m.create('x',[oid])\nexcept ValueError: pass\nelse: raise AssertionError('corrupt ref accepted')\n"), ('idempotent_conflict', "\nfrom blobstore.objects import ObjectStore\nfrom blobstore.manifests import ManifestStore,ManifestConflictError\no=ObjectStore(ROOT); a=o.put_bytes(b'a'); b=o.put_bytes(b'b'); m=ManifestStore(ROOT,o)\nm.create('x',[a]); before=m.path_for('x').read_bytes(); m.create('x',[a])\ntry: m.create('x',[b])\nexcept ManifestConflictError: pass\nelse: raise AssertionError('conflict accepted')\nassert m.path_for('x').read_bytes()==before\n"), ('ordered_names_and_references', "\nfrom blobstore.objects import ObjectStore\nfrom blobstore.manifests import ManifestStore\no=ObjectStore(ROOT); a=o.put_bytes(b'a'); b=o.put_bytes(b'b'); m=ManifestStore(ROOT,o)\nm.create('z',[a]); m.create('a/nested',[a,b])\nassert list(m.iter_names())==['a/nested','z'] and m.referenced_ids()=={a,b}\n"), ('delete_semantics', "\nfrom blobstore.objects import ObjectStore\nfrom blobstore.manifests import ManifestStore\nm=ManifestStore(ROOT,ObjectStore(ROOT)); m.create('x',[])\nassert m.delete('x') is True and m.delete('x') is False\n")]


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
