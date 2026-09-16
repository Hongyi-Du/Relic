"""Running the organization in real time next to a human.

The autonomous members keep their own clock: a background thread advances the
world one tick every ``seconds_per_tick`` of real time, indefinitely — there is
no tick ceiling in this mode and no end condition other than the work being
done. The human is not on that clock at all. A submitted action executes the
moment it arrives, between ticks, through the same pipeline the loop uses.

Two things make that safe. One lock serialises every mutation of the world,
which is otherwise entirely thread-unsafe. And reads never take that lock:
after each tick (and after each human action) the runtime rebuilds an immutable
per-seat view, so a browser polling every second is not blocked behind a step
that is waiting on an LLM.
"""
from __future__ import annotations

import copy
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from environments.org_env.human.seat import HumanSeat, HumanSeatRegistry
from environments.org_env.human.resource_inspector import (
    RESOURCE_SECTIONS_BY_KIND,
    build_resource,
)
from environments.org_env.human.seat_view import build_seat_view


OVERDUE_TICK_HANDOFF_SECONDS = 0.05

#: Real seconds per organizational hour. At 20s a working day passes in about
#: eight minutes — fast enough that a human sees the org move within a sitting,
#: slow enough to read what happened.
DEFAULT_SECONDS_PER_TICK = 20.0


class HumanModeRuntime:
    """Owns the background clock, the world lock, and the per-seat views."""

    def __init__(self, world: Any, *,
                 seconds_per_tick: float = DEFAULT_SECONDS_PER_TICK,
                 on_tick: Optional[Callable[[Any], None]] = None) -> None:
        self.world = world
        self.seconds_per_tick = max(0.05, float(seconds_per_tick))
        self.on_tick = on_tick
        self.lock = threading.RLock()
        self.seats = HumanSeatRegistry(world)

        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._running = threading.Event()      # a real pause signal, checked by the loop
        self._views: Dict[str, Dict[str, Any]] = {}
        self._resource_views: Dict[str, Dict[tuple, Dict[str, Any]]] = {}
        self._view_version = 0
        self.ticks_run = 0
        self.last_error: Optional[str] = None
        self._epoch_wall = time.time()
        self._epoch_tick = int(getattr(world, "world_tick", 0))

        self._lift_tick_ceiling()
        self.refresh_views()

    # -- time ---------------------------------------------------------------
    def _lift_tick_ceiling(self) -> None:
        """Human mode runs until the work is done, so an evaluator tick ceiling
        would be a wrong answer rather than a safety net. Frozen experiment runs
        never come through here."""
        budget = getattr(self.world, "experiment_resource_budget", None)
        if budget is not None and getattr(budget, "max_ticks", None) is not None:
            self.world.experiment_resource_ledger = None

    def tick_to_wall(self, tick: int) -> float:
        """POSIX timestamp for an organizational tick, so the UI can show real
        clock times and never a tick number."""
        return self._epoch_wall + (int(tick) - self._epoch_tick) * self.seconds_per_tick

    # -- lifecycle ----------------------------------------------------------
    def start(self) -> Dict[str, Any]:
        self._running.set()
        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="org-human-clock",
                                            daemon=True)
            self._thread.start()
        return self.status()

    def pause(self) -> Dict[str, Any]:
        self._running.clear()
        return self.status()

    def shutdown(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self._running.set()                    # release the loop so it can exit
        thread, self._thread = self._thread, None
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)

    def _run(self) -> None:
        while not self._stop.is_set():
            if not self._running.wait(timeout=0.2):
                continue
            if self._stop.is_set():
                break
            started = time.monotonic()
            try:
                with self.lock:
                    self.world.step()
            except Exception as exc:                       # noqa: BLE001 - surface, don't spin
                self.last_error = repr(exc)
                self._running.clear()
                continue
            self.ticks_run += 1
            self.refresh_views()
            if self.on_tick is not None:
                try:
                    self.on_tick(self.world)
                except Exception as exc:                   # noqa: BLE001
                    self.last_error = f"on_tick: {exc!r}"
            # A model-backed tick can take longer than its nominal interval.
            # Yield after releasing the world lock even when already overdue;
            # otherwise the clock can reacquire it continuously and starve a
            # waiting secretary read before that read ever reaches its model.
            self._stop.wait(max(
                OVERDUE_TICK_HANDOFF_SECONDS,
                self.seconds_per_tick - (time.monotonic() - started),
            ))

    def status(self) -> Dict[str, Any]:
        return {"running": self._running.is_set() and not self._stop.is_set(),
                "seconds_per_tick": self.seconds_per_tick,
                "ticks_run": self.ticks_run,
                "world_tick": int(getattr(self.world, "world_tick", 0)),
                "now": time.time(),
                "started_at": self._epoch_wall,
                "view_version": self._view_version,
                "last_error": self.last_error}

    # -- seats --------------------------------------------------------------
    def claim_seat(self, agent_id: str, *, display_name: str = "") -> HumanSeat:
        with self.lock:
            seat = self.seats.claim(agent_id, display_name=display_name)
        self.refresh_views()
        return seat

    def release_seat(self, agent_id: str) -> None:
        with self.lock:
            self.seats.release(agent_id)
        self.refresh_views()

    def presence(self) -> Dict[str, Dict[str, Any]]:
        now = time.time()
        return {aid: s.public_state(now) for aid, s in self.seats.seats.items()}

    # -- reads (never take the world lock) ----------------------------------
    def refresh_views(self) -> int:
        """Publish one coherent seat view, including rich read-only resources."""
        presence = self.presence()
        views: Dict[str, Dict[str, Any]] = {}
        resource_views: Dict[str, Dict[tuple, Dict[str, Any]]] = {}
        with self.lock:
            snapshot_world_tick = int(getattr(self.world, "world_tick", 0))
            for agent_id in list(self.seats.seats):
                views[agent_id] = build_seat_view(self.world, agent_id,
                                                  wall_clock=self.tick_to_wall,
                                                  seat_presence=presence)
            self._view_version += 1
            for view in views.values():
                view["version"] = self._view_version
                # This is the tick at which this already-published P2 projection
                # was built.  Lock-free generic Inspector reads must never consult
                # the mutable world merely to stamp their response.
                view["snapshot_world_tick"] = snapshot_world_tick
            for agent_id, view in views.items():
                resources: Dict[tuple, Dict[str, Any]] = {}
                for rows in (view.get("objects") or {}).values():
                    for row in rows or []:
                        kind = str(row.get("kind") or "")
                        resource_id = str(row.get("id") or "")
                        for resource_section in RESOURCE_SECTIONS_BY_KIND.get(kind, ()):
                            payload = build_resource(
                                self.world, agent_id, kind, resource_id,
                                section=resource_section, view=view,
                            )
                            payload["observed_at"] = {
                                "view_version": self._view_version,
                                "world_tick": snapshot_world_tick,
                            }
                            resources[(kind, resource_id, resource_section)] = payload
                resource_views[agent_id] = resources
            self._views = views
            self._resource_views = resource_views
        return self._view_version

    def seat_view(self, agent_id: str) -> Dict[str, Any]:
        """The latest cached view. Reads are lock-free so a browser polling
        during a slow LLM-driven tick still gets an answer immediately."""
        view = self._views.get(agent_id)
        if view is None:
            raise KeyError(f"no_view_for_seat:{agent_id}")
        return view

    def seat_resource(self, agent_id: str, resource_kind: str, resource_id: str,
                      section: str = "overview") -> Dict[str, Any]:
        """Resolve a resource from the latest coherent, published seat snapshot.

        This read never waits for a model-backed organization tick. Rich child
        lineage was validated while the snapshot was published under the world
        lock, so Files/Diff/Tests stay responsive without weakening visibility.
        """
        if agent_id not in self.seats.seats:
            raise KeyError(f"no_view_for_seat:{agent_id}")
        view = self.seat_view(agent_id)
        if resource_kind in RESOURCE_SECTIONS_BY_KIND:
            payload = self._resource_views.get(agent_id, {}).get(
                (resource_kind, resource_id, section or "overview"))
            if payload is None:
                return {
                    "error": "resource_not_visible",
                    "resource": {"kind": resource_kind, "id": resource_id},
                }
            return copy.deepcopy(payload)
        payload = build_resource(
            None, agent_id, resource_kind, resource_id,
            section=section, view=view)
        payload["observed_at"] = {
            "view_version": view.get("version"),
            "world_tick": view.get("snapshot_world_tick"),
        }
        return payload

    # -- writes -------------------------------------------------------------
    def submit_action(self, agent_id: str, action: Any, *,
                      execution_mode: str = "direct") -> Any:
        """Execute a human's action now, without waiting for a tick boundary."""
        with self.lock:
            result = self.world.apply_action_candidate(
                agent_id, action, controller_type="human", execution_mode=execution_mode)
        self.refresh_views()
        return result

    def with_world(self, fn: Callable[[Any], Any]) -> Any:
        """Run ``fn(world)`` under the world lock, for callers that need to
        mutate or read consistently outside the action pipeline."""
        with self.lock:
            return fn(self.world)


__all__ = ["HumanModeRuntime", "DEFAULT_SECONDS_PER_TICK"]
