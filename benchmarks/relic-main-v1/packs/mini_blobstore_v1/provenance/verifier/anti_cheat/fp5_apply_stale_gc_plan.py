from pathlib import Path
p=Path("blobstore/gc.py");t=p.read_text()
for n in (
    "if plan.state_fingerprint != self._fingerprint():",
    "if any(object_id in reachable for object_id in plan.candidates):",
):
    assert n in t
    t=t.replace(n,"if False:",1)
p.write_text(t)
