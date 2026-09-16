#!/usr/bin/env python3
"""The organization's own build gate: does what has been written still work?

PR-time CI runs this. It cannot demand the finished product, or nothing could
ever merge until everything was done at once; and it cannot wave everything
through, or a broken module would reach the mainline. So the gate is every step
whose module is no longer a stub: work that has been attempted must satisfy its
published contract, and work nobody has started is not held against the build.

Checking a contiguous prefix instead let the wrong thing merge. A candidate that
implemented steps 2-5 while step 1 was still a stub had an empty prefix, so the
gate found nothing to check and passed it — four modules reached the mainline
unexamined, importing a step 1 that raises. Every later request was then failed,
because with all five modules non-stub the gate suddenly demanded all of them.
Checking each started step instead refuses that first merge, which is the whole
point: the foundation has to land before what stands on it.

Run with --all to check every step regardless, which is what the organization
sees when it asks for the public test verdict: unwritten steps report as
failures there because that is an honest account of what is left.
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
STEPS = [{'name': 'step 1', 'module': 'blobstore/objects.py', 'smoke': 'public_contract_tests/step_01_smoke.py'}, {'name': 'step 2', 'module': 'blobstore/manifests.py', 'smoke': 'public_contract_tests/step_02_smoke.py'}, {'name': 'step 3', 'module': 'blobstore/uploads.py', 'smoke': 'public_contract_tests/step_03_smoke.py'}, {'name': 'step 4', 'module': 'blobstore/replicas.py', 'smoke': 'public_contract_tests/step_04_smoke.py'}, {'name': 'step 5', 'module': 'blobstore/gc.py', 'smoke': 'public_contract_tests/step_05_smoke.py'}]


def _is_stub(module_path: str) -> bool:
    """A module nobody has touched: it only announces that it is unwritten."""
    source = HERE / module_path
    if not source.is_file():
        return True
    text = source.read_text(encoding="utf-8", errors="replace").strip()
    return text.startswith("raise NotImplementedError") and len(text.splitlines()) == 1


def main() -> int:
    check_all = "--all" in sys.argv[1:]
    started = [step for step in STEPS if not _is_stub(step["module"])]
    selected = list(STEPS) if check_all else list(started)

    failures = []
    for step in selected:
        done = subprocess.run([sys.executable, str(HERE / step["smoke"])],
                              cwd=str(HERE), capture_output=True, text=True,
                              timeout=120)
        if done.returncode != 0:
            tail = (done.stderr or done.stdout or "").strip().splitlines()
            reason = tail[-1].strip() if tail else "no output"
            failures.append((step, reason))
            print("%s (%s) failed: %s" % (step["name"], step["module"], reason))
        else:
            print("%s passed" % step["name"])

    skipped = len(STEPS) - len(selected)
    print("%d passed, %d failed, %d not started (%d of %d modules written)"
          % (len(selected) - len(failures), len(failures), skipped,
             len(started), len(STEPS)))
    if failures:
        # The build gate's caller keeps only the LAST line of stderr as the
        # message it shows and files against the break, so the most useful
        # sentence has to be last. The steps stack, so the earliest failure is
        # the one to act on: reporting a later step's collapse instead sent
        # people to a module that was only failing because of this one. Writing
        # to stderr at all is what makes the message survive — a gate that
        # printed everything to stdout was reported as "smoke exited rc=1",
        # which says nothing a reader could act on.
        #
        # The line has to LEAD with the module path. What turns a build error
        # into a repair is the reader that localizes it to a file, and it reads a
        # filename off the front of the message; a line beginning "step 2 failed"
        # localizes to nothing, so no fix is ever aimed at the broken module.
        for step, reason in failures[1:]:
            sys.stderr.write("%s: %s also failed: %s\n"
                             % (step["module"], step["name"], reason))
        step, reason = failures[0]
        sys.stderr.write("%s: %s failed: %s\n"
                         % (step["module"], step["name"], reason))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
