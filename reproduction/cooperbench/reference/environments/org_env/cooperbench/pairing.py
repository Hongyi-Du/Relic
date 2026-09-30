"""Thread-safe rendezvous for CooperBench's two per-feature adapter calls."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from environments.org_env.cooperbench.contract import (
    CooperContractError,
    FeatureCall,
    PairOutcome,
    PairRequest,
)


PairExecutor = Callable[[PairRequest, int], PairOutcome]


@dataclass
class _PairState:
    signature: tuple
    calls: dict[str, FeatureCall] = field(default_factory=dict)
    paired: threading.Event = field(default_factory=threading.Event)
    done: threading.Event = field(default_factory=threading.Event)
    executing: bool = False
    outcome: PairOutcome | None = None
    error: BaseException | None = None
    consumers: int = 0
    created_monotonic: float = field(default_factory=time.monotonic)


class PairCoordinator:
    """Run one organization after both feature calls have arrived.

    CooperBench creates one adapter instance per agent but invokes the pair in
    threads in the same Python process.  A module-level coordinator can thus
    rendezvous the calls without Redis, while the actual OrgEnv execution is a
    subprocess so concurrent benchmark pairs do not share environment state.
    """

    def __init__(self, executor: PairExecutor):
        self._executor = executor
        self._states: dict[str, _PairState] = {}
        self._lock = threading.Lock()

    def submit(
        self, call: FeatureCall, *, rendezvous_timeout_seconds: float = 120.0
    ) -> PairOutcome:
        call = call.validated()
        execute_here = False
        with self._lock:
            state = self._states.get(call.run_id)
            if state is None:
                state = _PairState(signature=call.pair_signature())
                self._states[call.run_id] = state
            elif state.signature != call.pair_signature():
                raise CooperContractError("pair_execution_config_mismatch")
            if call.agent_id in state.calls:
                raise CooperContractError(f"duplicate_pair_call:{call.agent_id}")
            if state.executing or state.done.is_set():
                raise CooperContractError("pair_call_arrived_after_execution_started")
            state.calls[call.agent_id] = call
            if len(state.calls) == 2:
                state.executing = True
                execute_here = True
                state.paired.set()

        if not execute_here and not state.paired.wait(
            timeout=max(0.01, float(rendezvous_timeout_seconds))
        ):
            timeout_error = TimeoutError(
                f"cooper_pair_rendezvous_timeout:{call.run_id}:"
                f"received={sorted(state.calls)}"
            )
            with self._lock:
                if not state.executing and not state.done.is_set():
                    state.error = timeout_error
                    state.done.set()
                    self._states.pop(call.run_id, None)
            raise timeout_error

        if execute_here:
            try:
                request = PairRequest.from_calls(state.calls)
                from .resume_identity import resolve_request
                request = resolve_request(request)
                outcome = self._executor(request, call.worker_timeout_seconds)
                outcome = outcome.validated()
                if outcome.run_id != request.run_id or outcome.agents != request.agents:
                    raise CooperContractError("pair_outcome_identity_mismatch")
                state.outcome = outcome
            except BaseException as error:  # wake the peer for every failure
                state.error = error
            finally:
                state.done.set()

        # Once paired, the peer waits for the worker budget rather than the
        # short arrival budget.  A real B3 run takes hours; using the rendezvous
        # timeout here made the first Cooper thread fail after two minutes even
        # though its partner had arrived and the one valid worker was running.
        if not state.done.wait(timeout=max(1.0, call.worker_timeout_seconds + 60.0)):
            execution_timeout = TimeoutError(
                f"cooper_pair_execution_wait_timeout:{call.run_id}:"
                f"budget={call.worker_timeout_seconds}"
            )
            with self._lock:
                if not state.done.is_set():
                    state.error = execution_timeout
                    state.done.set()
                    self._states.pop(call.run_id, None)
            raise execution_timeout

        try:
            if state.error is not None:
                raise state.error
            if state.outcome is None:
                raise RuntimeError("pair_execution_completed_without_outcome")
            return state.outcome
        finally:
            with self._lock:
                state.consumers += 1
                if state.consumers >= 2:
                    self._states.pop(call.run_id, None)

    def pending_run_ids(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._states))


__all__ = ["PairCoordinator", "PairExecutor"]
