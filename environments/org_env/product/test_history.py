"""Per-test outcome history, so a fix can be DEMONSTRATED rather than declared.

The organization cannot see the hidden oracles, so its only honest evidence that
a patch worked is its own test suite. A single latest-result snapshot is not
enough for that: it answers "is the tree green right now", not "did this work
change anything". Two failure modes measured on a real 336-tick run need the
difference:

* ten issue-linked tasks reached MERGED with progress 1.0 while one of eleven
  reachable oracles actually passed — merging was treated as fixing;
* a regression the organization itself introduced was detected three times
  (t225, t250, t278) and never repaired, because a repeat detection carried no
  more weight than the first.

Both become answerable once outcomes are kept over time: a fix is a test that
was failing and later is not, and a stuck defect is a test that keeps failing
across separate runs.

Only runs that actually produced a verdict are recorded. An infrastructure
error yields an empty failure list, and admitting one would read as "every
failure disappeared" — the same misattribution that once turned an empty
container mount into 172 phantom contract breaks.

Nothing here knows what substrate it is running on; it works for any repo whose
manifest declares a public test command.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

__all__ = [
    "HISTORY_LIMIT",
    "record_public_test_run",
    "public_test_history",
    "latest_recorded_run",
    "persistently_failing_tests",
    "demonstrated_fixes_since",
    "regressions_since",
    "suite_size_regression",
    "undemonstrated_test_additions",
]

HISTORY_LIMIT = 200

_HISTORY_KEY = "_public_test_history"


def public_test_history(world: Any) -> List[Dict[str, Any]]:
    """The recorded runs, oldest first. Never None."""
    history = world.__dict__.get(_HISTORY_KEY)
    return history if isinstance(history, list) else []


def record_public_test_run(
    world: Any,
    outcome: Any,
    tick: int,
    *,
    repo_hash: str = "",
) -> bool:
    """Append one suite verdict to the history. Returns whether it was recorded.

    A run is only admitted when the suite was available, launched, and produced
    a verdict. Declining the rest is what keeps "no signal" from being read as
    "no failures".
    """
    if not isinstance(outcome, dict):
        return False
    if not outcome.get("available") or outcome.get("error"):
        return False
    entry = {
        "tick": int(tick),
        "repo_hash": str(repo_hash or ""),
        "ok": bool(outcome.get("ok")),
        "failed": sorted(str(t) for t in (outcome.get("failed_tests") or [])),
        "collected": int(outcome.get("collected") or 0),
    }
    history = world.__dict__.setdefault(_HISTORY_KEY, [])
    if not isinstance(history, list):
        history = []
        world.__dict__[_HISTORY_KEY] = history
    history.append(entry)
    if len(history) > HISTORY_LIMIT:
        del history[: len(history) - HISTORY_LIMIT]
    return True


def latest_recorded_run(world: Any) -> Dict[str, Any] | None:
    history = public_test_history(world)
    return history[-1] if history else None


def _runs_since(world: Any, since_tick: int | None) -> List[Dict[str, Any]]:
    history = public_test_history(world)
    if since_tick is None:
        return list(history)
    return [r for r in history if int(r.get("tick", 0)) >= int(since_tick)]


def persistently_failing_tests(world: Any, *, min_detections: int = 2) -> List[str]:
    """Tests failing in the latest run that have also failed in earlier runs.

    A defect seen once may already be under repair; one seen again after the
    organization had a chance to act is the one worth escalating.
    """
    latest = latest_recorded_run(world)
    if not latest:
        return []
    still_failing = set(latest.get("failed") or [])
    if not still_failing:
        return []
    counts = {t: 0 for t in still_failing}
    for run in public_test_history(world):
        for test_id in run.get("failed") or []:
            if test_id in counts:
                counts[test_id] += 1
    return sorted(t for t, n in counts.items() if n >= int(min_detections))


def demonstrated_fixes_since(world: Any, since_tick: int | None = None) -> List[str]:
    """Tests that were observed failing and are not failing in a later run.

    This is the red-then-green evidence: it credits work only when the
    organization's own suite changed its verdict, which a patch that merely
    claims to fix something cannot produce.

    The suite is agent-editable, so a failing test can also stop failing by
    being deleted or weakened. A disappearance is therefore only credited when
    the later run reported at least as many outcomes as the run that saw the
    failure — a shrunk suite proves nothing about the product.
    """
    runs = _runs_since(world, since_tick)
    if len(runs) < 2:
        return []
    fixed: set[str] = set()
    # test id -> how many outcomes the suite reported when it was last failing
    failing_at_size: Dict[str, int] = {}
    for run in runs:
        failing = set(run.get("failed") or [])
        size = int(run.get("collected") or 0)
        for test_id, seen_size in list(failing_at_size.items()):
            if test_id in failing:
                continue
            if size and seen_size and size < seen_size:
                continue                 # the suite shrank; this is not evidence
            fixed.add(test_id)
        for test_id in failing:
            failing_at_size[test_id] = size
    latest_failing = set(runs[-1].get("failed") or [])
    return sorted(fixed - latest_failing)


def suite_size_regression(world: Any, since_tick: int | None = None) -> Dict[str, int] | None:
    """Whether the suite reports fewer outcomes than it once did.

    Deleting or weakening tests is the cheapest way to turn a red suite green,
    and no pass rate can see it. Returns ``{peak, latest}`` when the latest
    measured run is smaller than the largest one seen, else None. Runs that
    reported no counts are ignored — absence of measurement is not shrinkage.
    """
    sizes = [int(r.get("collected") or 0) for r in _runs_since(world, since_tick)]
    measured = [s for s in sizes if s > 0]
    if len(measured) < 2:
        return None
    peak, latest = max(measured), measured[-1]
    return {"peak": peak, "latest": latest} if latest < peak else None


def undemonstrated_test_additions(world: Any, since_tick: int | None = None) -> int:
    """Tests added to the suite that were never once observed failing.

    Writing the test is only half the discipline. A test authored after the fix
    already landed passes the moment it exists, and a test that has never been
    red cannot distinguish a working fix from a test that does not exercise the
    behaviour at all — both look green.

    Measured on a 336-tick run: the organization added exactly one test, aimed
    it correctly at the issue, and the suite was green in all fourteen recorded
    runs, so nothing about that test constitutes evidence.

    Returns how many additions lack a red observation, so the gap is visible
    rather than being read as one more passing test.
    """
    runs = _runs_since(world, since_tick)
    measured = [r for r in runs if int(r.get("collected") or 0) > 0]
    if len(measured) < 2:
        return 0
    growth = int(measured[-1].get("collected") or 0) - int(measured[0].get("collected") or 0)
    if growth <= 0:
        return 0
    return max(0, growth - len(demonstrated_fixes_since(world, since_tick)))


def regressions_since(world: Any, since_tick: int | None = None) -> List[str]:
    """Tests failing in the latest run that were not failing in an earlier one.

    Distinguishing these from pre-existing failures matters because a
    regression is caused by the organization's own recent work, so it is
    actionable in a way an inherited failure is not.
    """
    runs = _runs_since(world, since_tick)
    if len(runs) < 2:
        return []
    latest_failing = set(runs[-1].get("failed") or [])
    if not latest_failing:
        return []
    previously_clean: set[str] = set()
    for run in runs[:-1]:
        previously_clean |= latest_failing - set(run.get("failed") or [])
    return sorted(previously_clean)


def summarize(world: Any) -> Dict[str, Any]:
    """Compact view for logging and run records."""
    latest = latest_recorded_run(world)
    return {
        "runs_recorded": len(public_test_history(world)),
        "latest_ok": bool(latest.get("ok")) if latest else None,
        "latest_failed_count": len(latest.get("failed") or []) if latest else 0,
        "persistently_failing": persistently_failing_tests(world),
        "demonstrated_fixes": demonstrated_fixes_since(world),
        "regressions": regressions_since(world),
        "suite_size_regression": suite_size_regression(world),
        "undemonstrated_test_additions": undemonstrated_test_additions(world),
    }
