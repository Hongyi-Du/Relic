"""Real-time human mode (HCI P1).

The autonomous members run on a background clock with no tick ceiling; the
human is not on that clock and acts the moment they choose to. These tests
cover the three things that makes possible: a clock that really starts and
really stops, a lock that lets a human write while the loop is stepping, and
lock-free reads so a browser is never blocked behind a slow tick.

Run:  PYTHONPATH="." python tests/org_env/test_human_runtime.py
"""
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent_sdk.lived.core.contracts import ActionCandidate
from environments.org_env.human.runtime import HumanModeRuntime
from environments.org_env.human.seat import SeatUnavailable
from environments.org_env.runtime_adapter.live import OrgInspectorSession

SEAT = "sean"
FAST = 0.05          # seconds per organizational hour, for tests


def _runtime(ticks=8, **kw):
    s = OrgInspectorSession(seed=42)
    if ticks:
        s.step(ticks)
    kw.setdefault("seconds_per_tick", FAST)
    return HumanModeRuntime(s.world, **kw)


def _message(text="status update"):
    return ActionCandidate(action_type="send_message",
                           parameters={"channel_id": "ch_general", "text": text})


def _wait_until(predicate, timeout=5.0, interval=0.02):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


# ---- the clock ------------------------------------------------------------ #
def test_the_background_clock_advances_the_world_on_its_own():
    rt = _runtime()
    start = rt.world.world_tick
    try:
        rt.start()
        assert _wait_until(lambda: rt.world.world_tick >= start + 3), "clock never advanced"
    finally:
        rt.shutdown()


def test_an_overdue_tick_still_yields_the_world_lock_before_the_next_tick():
    class StopAfterFirstWait:
        def __init__(self):
            self.stopped = False
            self.waits = []

        def is_set(self):
            return self.stopped

        def set(self):
            self.stopped = True

        def wait(self, timeout):
            self.waits.append(timeout)
            self.stopped = True
            return True

    rt = _runtime(ticks=0, seconds_per_tick=0.001)
    stop = StopAfterFirstWait()
    rt._stop = stop
    rt._running.set()
    rt.world.step = lambda: time.sleep(0.01)

    rt._run()

    assert len(stop.waits) == 1
    assert stop.waits[0] >= 0.05


def test_pause_actually_stops_the_clock():
    """The pre-existing OrgInspectorSession.pause only set a flag nothing read.
    A human pressing pause has to mean the organization stops moving."""
    rt = _runtime()
    try:
        rt.start()
        assert _wait_until(lambda: rt.world.world_tick >= 2)
        rt.pause()
        time.sleep(FAST * 4)
        settled = rt.world.world_tick
        time.sleep(FAST * 6)
        assert rt.world.world_tick == settled, "world kept moving after pause"
        assert rt.status()["running"] is False

        rt.start()
        assert _wait_until(lambda: rt.world.world_tick > settled), "resume did not restart"
    finally:
        rt.shutdown()


def test_human_mode_lifts_the_tick_ceiling():
    """This mode runs until the work is done; an evaluator tick budget would
    stop it mid-session with a resource error."""
    from environments.org_env.experiments.resources import (
        ExperimentResourceLedger,
        FrozenResourceBudget,
    )

    s = OrgInspectorSession(seed=42)
    budget = FrozenResourceBudget(max_ticks=2)
    s.world.experiment_resource_budget = budget
    s.world.experiment_resource_ledger = ExperimentResourceLedger(budget)

    rt = HumanModeRuntime(s.world, seconds_per_tick=FAST)
    try:
        assert s.world.experiment_resource_ledger is None
        rt.start()
        assert _wait_until(lambda: rt.world.world_tick >= 5), "stopped at the old ceiling"
        assert rt.status()["last_error"] is None
    finally:
        rt.shutdown()


def test_shutdown_stops_the_thread():
    rt = _runtime()
    rt.start()
    assert _wait_until(lambda: rt.world.world_tick >= 1)
    rt.shutdown()
    assert not any(t.name == "org-human-clock" and t.is_alive()
                   for t in threading.enumerate())


# ---- human writes --------------------------------------------------------- #
def test_a_human_action_lands_immediately_not_on_a_tick_boundary():
    rt = _runtime(seconds_per_tick=30.0)        # next tick is far away
    rt.claim_seat(SEAT)
    try:
        before = len(rt.world.action_log)
        started = time.monotonic()
        rt.submit_action(SEAT, _message())
        assert time.monotonic() - started < 5.0
        assert len(rt.world.action_log) == before + 1
        assert rt.world.action_log[-1]["agent_id"] == SEAT
        assert rt.world.controller_log[-1]["controller_type"] == "human"
    finally:
        rt.shutdown()


def test_human_writes_and_the_clock_do_not_corrupt_each_other():
    """The world is not thread-safe; the runtime lock is the only thing making
    a live clock plus a typing human safe."""
    rt = _runtime()
    rt.claim_seat(SEAT)
    rt.claim_seat("calvin")
    errors = []
    submitted = 24

    def hammer(agent_id):
        for i in range(submitted // 2):
            try:
                rt.submit_action(agent_id, _message(f"{agent_id} {i}"))
            except Exception as exc:                      # noqa: BLE001
                errors.append(repr(exc))

    try:
        rt.start()
        threads = [threading.Thread(target=hammer, args=(a,))
                   for a in (SEAT, "calvin")]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        rt.pause()

        assert not errors, errors
        human_actions = [c for c in rt.world.controller_log if c["controller_type"] == "human"]
        assert len(human_actions) == submitted, f"{len(human_actions)} of {submitted} landed"
        # The world survived: its own loop kept logging alongside the humans.
        assert rt.world.world_tick > 8
    finally:
        rt.shutdown()


# ---- reads ---------------------------------------------------------------- #
def test_reads_do_not_block_on_a_slow_tick():
    """A browser polls every second or so. If reads took the world lock, a tick
    waiting on an LLM would freeze every seat's UI."""
    rt = _runtime()
    rt.claim_seat(SEAT)
    try:
        held = threading.Event()
        release = threading.Event()

        def hog():
            with rt.lock:
                held.set()
                release.wait(timeout=5)

        threading.Thread(target=hog, daemon=True).start()
        assert held.wait(timeout=5)

        started = time.monotonic()
        view = rt.seat_view(SEAT)              # must not wait for the lock
        assert time.monotonic() - started < 1.0
        assert view["seat"]["agent_id"] == SEAT
        release.set()
    finally:
        rt.shutdown()


def test_views_refresh_after_a_tick_and_after_a_human_action():
    rt = _runtime()
    rt.claim_seat(SEAT)
    try:
        v0 = rt.seat_view(SEAT)["version"]
        rt.submit_action(SEAT, _message("first"))
        v1 = rt.seat_view(SEAT)["version"]
        assert v1 > v0

        rt.start()
        assert _wait_until(lambda: rt.seat_view(SEAT)["version"] > v1)
    finally:
        rt.shutdown()


# ---- seats ---------------------------------------------------------------- #
def test_claiming_a_seat_takes_it_off_the_autonomous_loop_and_back():
    rt = _runtime()
    try:
        seat = rt.claim_seat(SEAT)
        assert seat.token and rt.world.is_human_controlled(SEAT)
        assert rt.seats.authenticate(seat.token).agent_id == SEAT

        rt.release_seat(SEAT)
        assert not rt.world.is_human_controlled(SEAT)
        try:
            rt.seats.authenticate(seat.token)
        except SeatUnavailable:
            pass
        else:
            raise AssertionError("a released token still authenticates")
    finally:
        rt.shutdown()


def test_a_seat_cannot_be_claimed_twice_and_unknown_members_are_rejected():
    rt = _runtime()
    try:
        rt.claim_seat(SEAT)
        for bad in (SEAT, "nobody"):
            try:
                rt.claim_seat(bad)
            except SeatUnavailable:
                continue
            raise AssertionError(f"claiming {bad!r} should have failed")
    finally:
        rt.shutdown()


def test_seats_are_isolated_from_each_other():
    rt = _runtime()
    try:
        a = rt.claim_seat(SEAT)
        b = rt.claim_seat("calvin")
        assert a.token != b.token
        assert rt.seat_view(SEAT)["seat"]["agent_id"] == SEAT
        assert rt.seat_view("calvin")["seat"]["agent_id"] == "calvin"
        # Presence is public; the token never is.
        assert "token" not in a.public_state()
    finally:
        rt.shutdown()


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print("  ok ", fn.__name__)
    print(f"All {len(fns)} human runtime tests passed!")
