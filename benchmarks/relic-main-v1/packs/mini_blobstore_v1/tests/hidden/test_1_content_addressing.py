"""Step 1 — Content-addressed objects

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
CASES = [('digest_and_roundtrip', "\nimport hashlib\nfrom blobstore.objects import ObjectStore\ns=ObjectStore(ROOT); data=b'hello\\x00world'; oid=s.put_bytes(data)\nassert oid==hashlib.sha256(data).hexdigest() and s.get(oid)==data and s.verify(oid)\n"), ('sharded_path', "\nfrom blobstore.objects import ObjectStore\ns=ObjectStore(ROOT); oid=s.put_bytes(b'x'); p=s.path_for(oid)\nassert p.parent.name==oid[:2] and p.name==oid[2:] and p.is_file()\n"), ('idempotent_write', "\nfrom blobstore.objects import ObjectStore\ns=ObjectStore(ROOT); a=s.put_bytes(b'x'); before=s.path_for(a).stat().st_mtime_ns\nb=s.put_bytes(b'x'); assert a==b and s.path_for(a).stat().st_mtime_ns==before\n"), ('streaming_large_input', "\nimport io\nfrom blobstore.objects import ObjectStore\ns=ObjectStore(ROOT); data=b'abcd'*400000; oid=s.put_stream(io.BytesIO(data))\nassert s.get(oid)==data\n"), ('invalid_ids_rejected', "\nfrom blobstore.objects import ObjectStore,InvalidObjectId\ns=ObjectStore(ROOT)\nfor oid in ['', '../x', 'A'*64, '0'*63]:\n    try: s.path_for(oid)\n    except InvalidObjectId: pass\n    else: raise AssertionError(oid)\n"), ('corruption_detected_not_overwritten', "\nfrom blobstore.objects import ObjectStore,CorruptObjectError\ns=ObjectStore(ROOT); oid=s.put_bytes(b'good'); s.path_for(oid).write_bytes(b'bad')\nassert not s.verify(oid)\ntry: s.put_bytes(b'good')\nexcept CorruptObjectError: pass\nelse: raise AssertionError('corruption hidden')\nassert s.path_for(oid).read_bytes()==b'bad'\n"), ('sorted_discovery_ignores_noise', "\nfrom blobstore.objects import ObjectStore\ns=ObjectStore(ROOT); ids=[s.put_bytes(x) for x in (b'z',b'a',b'm')]\n(s.objects/'noise').write_text('x'); assert list(s.iter_ids())==sorted(ids)\n")]


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
