"""Route logic for human seats.

Plain functions over a session, in the same style as the rest of
``backend/main.py``: no FastAPI here, so every endpoint can be tested directly.
Errors come back as ``{"error": ...}`` rather than exceptions for the same
reason.

Everything that acts as a seat requires that seat's token. A caller holding no
token can list members and read runtime status; it cannot see any seat's view
or act as anyone.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Callable, Dict, List, Optional

from environments.org_env.human import gateway
from environments.org_env.human.runtime import DEFAULT_SECONDS_PER_TICK, HumanModeRuntime
from environments.org_env.human.seat import SeatUnavailable
from environments.org_env.human.working_agent import WorkingAgentSession
from environments.org_env.human.execution_team import ExecutionTeam
from environments.org_env.human.organization_brief import build_organization_brief


class HumanApi:
    """Owns the live human-mode runtime for whichever world the session holds.

    The inspector session can be reset or restored from a checkpoint under us,
    which swaps the world object; the runtime is rebound when that happens so a
    stale clock never keeps stepping a discarded world.
    """

    def __init__(self, session_provider: Callable[[], Any], *,
                 seconds_per_tick: float = DEFAULT_SECONDS_PER_TICK) -> None:
        self._session_provider = session_provider
        self._seconds_per_tick = seconds_per_tick
        self._runtime: Optional[HumanModeRuntime] = None
        # Every bound world gets a new epoch. Long-running compiler/worker
        # calls capture the epoch they started in, so a late result from a
        # discarded world cannot create work in the replacement organization.
        self._runtime_epoch = 0
        self._agents: Dict[str, WorkingAgentSession] = {}   # agent_id -> assistant
        self._execution_team: Optional[ExecutionTeam] = None
        self._hci_logs: List[Dict[str, Any]] = []  # HCI interaction logs
        # A bounded delivery journal keeps seat-filtered facts available to P3
        # even when a browser is closed long enough for the P2 working feed to
        # roll over. It is an HCI cache, never a world event store.
        self._visible_event_journal: Dict[str, List[Dict[str, Any]]] = {}
        self._visible_event_seen: Dict[str, Dict[str, None]] = {}
        self._visible_message_journal: Dict[str, List[Dict[str, Any]]] = {}
        self._visible_message_seen: Dict[str, Dict[str, None]] = {}

    # -- runtime binding ----------------------------------------------------
    def runtime(self) -> HumanModeRuntime:
        session = self._session_provider()

        # If SESSION not initialized yet, raise error
        if session is None or not hasattr(session, 'world') or session.world is None:
            raise RuntimeError("World not initialized. Visit /org/setup to select a pack.")

        if self._runtime is None or self._runtime.world is not session.world:
            self._runtime_epoch += 1
            runtime_epoch = self._runtime_epoch
            old_runtime = self._runtime
            old_execution_team = self._execution_team
            # Invalidate callback targets before shutdown waits for worker
            # threads. A provider response may return after that bounded wait.
            self._runtime = None
            self._execution_team = None
            self._agents.clear()
            if old_execution_team is not None:
                old_execution_team.shutdown()
            if old_runtime is not None:
                old_runtime.shutdown()
            self._visible_event_journal.clear()
            self._visible_event_seen.clear()
            self._visible_message_journal.clear()
            self._visible_message_seen.clear()
            self._runtime = HumanModeRuntime(
                session.world,
                seconds_per_tick=self._seconds_per_tick,
                on_tick=self._capture_frame,
            )
            self._execution_team = ExecutionTeam(
                self._runtime, llm_provider=self._llm_client,
                on_complete=lambda job, expected_epoch=runtime_epoch: (
                    self._execution_complete(job, expected_epoch=expected_epoch)),
                on_progress=lambda job, expected_epoch=runtime_epoch: (
                    self._execution_progress(job, expected_epoch=expected_epoch)),
            )
        return self._runtime

    def _execution_complete(self, job: Dict[str, Any], *,
                            expected_epoch: Optional[int] = None) -> None:
        """Durable secretary result for a completed/cancelled worker job."""
        if expected_epoch is not None and expected_epoch != self._runtime_epoch:
            return
        seat_id = str(job.get("seat_id") or "")
        if self._runtime is None or self._runtime.seats.get(seat_id) is None:
            return
        agent = self._agents.get(seat_id)
        if agent is not None:
            agent._append("agent", str(job.get("final_report") or "Execution job settled."),
                          kind="execution_summary",
                          thread_id=str(job.get("thread_id") or ""),
                          references=list(job.get("references") or []),
                          evidence_records=list(job.get("evidence_records") or []))
        self._log_hci_event("liaison_execution_job_settled", {
            "job_id": job.get("job_id"), "status": job.get("status"),
            "timeline": job.get("timeline") or [],
        })

    def _execution_progress(self, job: Dict[str, Any], *,
                            expected_epoch: Optional[int] = None) -> None:
        """Deliver a scheduled snapshot to the same secretary conversation."""
        if expected_epoch is not None and expected_epoch != self._runtime_epoch:
            return
        seat_id = str(job.get("seat_id") or "")
        if self._runtime is None or self._runtime.seats.get(seat_id) is None:
            return
        agent = self._agents.get(seat_id)
        if agent is not None:
            agent._append("agent", str(job["progress_summary"]),
                          kind="execution_progress",
                          thread_id=str(job.get("thread_id") or ""),
                          references=list(job.get("references") or []),
                          evidence_records=list(job.get("evidence_records") or []))

    def _prepare_execution_plan(self, agent_id: str, plan: Dict[str, Any], request: str,
                                thread_id: str = "", *,
                                expected_epoch: Optional[int] = None) -> Dict[str, Any]:
        if expected_epoch is not None and expected_epoch != self._runtime_epoch:
            return {"error": "stale_runtime_discarded"}
        team = self._execution_team
        if team is None:
            return {"error": "execution_team_unavailable"}
        result = team.prepare(agent_id, plan, request=request, thread_id=thread_id)
        if result.get("ok"):
            self._log_hci_event("liaison_execution_job_prepared", {"agent_id": agent_id,
                                "job_id": (result.get("job") or {}).get("job_id")})
        return result

    def _capture_frame(self, _world: Any) -> None:
        """Keep the inspector's frame buffer following the live clock.

        The two views share one world. Without this the human clock advances it
        while /api/org/lived/* stays frozen at whatever tick the last manual
        step left behind — the same organization, apparently stopped.
        """
        runtime, session = self._runtime, self._session_provider()
        if runtime is None or session.world is not runtime.world:
            return
        with runtime.lock:
            session._capture()
        self._capture_visible_delivery()

    @staticmethod
    def _delivery_identity(row: Dict[str, Any]) -> str:
        encoded = json.dumps(row, sort_keys=True, default=str,
                             separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:32]

    def _capture_visible_delivery(self) -> None:
        """Journal already-filtered seat events/messages for secretary delivery."""
        runtime = self._runtime
        if runtime is None:
            return
        for agent_id in list(runtime.seats.seats):
            try:
                view = runtime.seat_view(agent_id)
            except KeyError:
                continue
            event_history = self._visible_event_journal.setdefault(agent_id, [])
            event_seen = self._visible_event_seen.setdefault(agent_id, {})
            for event in ((view.get("feed") or {}).get("events") or []):
                identity = self._delivery_identity(event)
                if identity not in event_seen:
                    event_history.append(dict(event))
                    event_seen[identity] = None
            message_history = self._visible_message_journal.setdefault(agent_id, [])
            message_seen = self._visible_message_seen.setdefault(agent_id, {})
            for thread in ((view.get("feed") or {}).get("threads") or []):
                for message in thread.get("messages") or []:
                    identity = str(message.get("id") or self._delivery_identity(message))
                    if identity not in message_seen:
                        message_history.append(dict(message))
                        message_seen[identity] = None
            del event_history[:-2048]
            del message_history[:-1024]
            while len(event_seen) > 4096:
                del event_seen[next(iter(event_seen))]
            while len(message_seen) > 2048:
                del message_seen[next(iter(message_seen))]

    def liaison_visible_events(self, agent_id: str) -> List[Dict[str, Any]]:
        """Internal P3 input: copies of events already filtered for this seat."""
        return [dict(row) for row in self._visible_event_journal.get(agent_id, [])]

    def liaison_visible_messages(self, agent_id: str) -> List[Dict[str, Any]]:
        """Internal P3 input: copies of messages already filtered for this seat."""
        return [dict(row) for row in self._visible_message_journal.get(agent_id, [])]

    def _log_hci_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Log HCI interaction event with timestamp."""
        import time
        self._hci_logs.append({
            "timestamp": time.time(),
            "type": event_type,
            "data": data,
        })

    def get_hci_logs(self) -> List[Dict[str, Any]]:
        """Get all HCI logs for trajectory export."""
        return self._hci_logs.copy()

    def _seat(self, token: str):
        return self.runtime().seats.authenticate(token)

    # -- members and seats --------------------------------------------------
    def members(self) -> Dict[str, Any]:
        """Who could be taken over. Says nothing about who already is a human
        beyond the seat being claimed, which the claimer needs to know."""
        try:
            return {"members": self.runtime().seats.claimable()}
        except RuntimeError as e:
            return {"error": str(e), "members": []}

    def claim(self, agent_id: str, display_name: str = "") -> Dict[str, Any]:
        try:
            seat = self.runtime().claim_seat(agent_id, display_name=display_name)
        except SeatUnavailable as exc:
            return {"error": str(exc)}
        self._capture_visible_delivery()
        return {"agent_id": seat.agent_id, "token": seat.token,
                "claimed_at": seat.claimed_at}

    def release(self, token: str) -> Dict[str, Any]:
        try:
            seat = self._seat(token)
        except SeatUnavailable as exc:
            return {"error": str(exc)}
        if self._execution_team is not None:
            self._execution_team.cancel_for(seat.agent_id, reason="human_seat_released")
        self.runtime().release_seat(seat.agent_id)
        self._visible_event_journal.pop(seat.agent_id, None)
        self._visible_event_seen.pop(seat.agent_id, None)
        self._visible_message_journal.pop(seat.agent_id, None)
        self._visible_message_seen.pop(seat.agent_id, None)
        return {"released": seat.agent_id}

    # -- reads --------------------------------------------------------------
    def view(self, token: str, since_version: int = -1) -> Dict[str, Any]:
        """The seat's view of the organization, or a short 'nothing changed'
        when the caller already has the current version."""
        try:
            seat = self._seat(token)
        except SeatUnavailable as exc:
            return {"error": str(exc)}
        except RuntimeError as exc:
            return {"error": str(exc)}

        try:
            rt = self.runtime()
        except RuntimeError as exc:
            return {"error": str(exc)}

        try:
            view = rt.seat_view(seat.agent_id)
        except KeyError:
            rt.refresh_views()
            view = rt.seat_view(seat.agent_id)
        if since_version >= 0 and view.get("version") == since_version:
            return {"unchanged": True, "version": view.get("version"),
                    "runtime": rt.status()}
        return {"unchanged": False, "view": view, "runtime": rt.status()}

    def brief(self, token: str) -> Dict[str, Any]:
        """Return an auditable liaison brief for the authenticated seat only."""
        try:
            seat = self._seat(token)
            rt = self.runtime()
            try:
                view = rt.seat_view(seat.agent_id)
            except KeyError:
                rt.refresh_views()
                view = rt.seat_view(seat.agent_id)
        except (SeatUnavailable, RuntimeError) as exc:
            return {"error": str(exc)}
        return {"brief": build_organization_brief(view), "version": view.get("version")}

    # -- what this seat may do ----------------------------------------------
    def offers(self, token: str, object_id: str = "") -> Dict[str, Any]:
        """The action menu: for one object, or the global composer menu."""
        try:
            seat = self._seat(token)
        except SeatUnavailable as exc:
            return {"error": str(exc)}
        except RuntimeError as exc:
            return {"error": str(exc)}

        try:
            rt = self.runtime()
        except RuntimeError as exc:
            return {"error": str(exc)}

        try:
            with rt.lock:
                if object_id:
                    return gateway.offers_for_object(rt.world, seat.agent_id, object_id)
                return {"object_id": None, "kind": "global",
                        "actions": gateway.global_offers(rt.world, seat.agent_id)}
        except gateway.ActionRefused as exc:
            return {"error": str(exc)}

    def act(self, token: str, action_type: str,
            params: Optional[Dict[str, Any]] = None,
            execution_mode: str = "direct") -> Dict[str, Any]:
        """Take an action as this seat, right now."""
        try:
            seat = self._seat(token)
        except SeatUnavailable as exc:
            return {"error": str(exc)}
        except RuntimeError as exc:
            return {"error": str(exc)}

        try:
            rt = self.runtime()
        except RuntimeError as exc:
            return {"error": str(exc)}

        try:
            result = gateway.submit(rt, seat.agent_id, action_type, params,
                                    execution_mode=execution_mode)
        except gateway.ActionRefused as exc:
            return {"error": str(exc)}

        # Log the action
        self._log_hci_event("user_action", {
            "agent_id": seat.agent_id,
            "action_type": action_type,
            "params": params,
            "success": result.success,
            "created": list(result.created_objects),
            "modified": list(result.modified_objects),
        })
        self._capture_visible_delivery()

        return {"ok": bool(result.success), "action_type": result.action_type,
                "action_id": result.action_id,
                "failure_reason": result.failure_reason,
                "created": list(result.created_objects),
                "modified": list(result.modified_objects),
                "version": rt.seat_view(seat.agent_id).get("version")}

    # -- the seat's private working agent -----------------------------------
    def _agent(self, agent_id: str) -> WorkingAgentSession:
        runtime = self.runtime()
        session = self._agents.get(agent_id)
        if session is None or session.runtime is not runtime:
            agent_epoch = self._runtime_epoch

            def log_liaison(event_type: str, data: Dict[str, Any]) -> None:
                if agent_epoch != self._runtime_epoch:
                    return
                self._log_hci_event(event_type, {"agent_id": agent_id, **data})

            def prepare_execution(aid: str, plan: Dict[str, Any], request: str,
                                  thread_id: str = "") -> Dict[str, Any]:
                return self._prepare_execution_plan(
                    aid, plan, request, thread_id, expected_epoch=agent_epoch)

            def execution_agents() -> List[Dict[str, Any]]:
                if agent_epoch != self._runtime_epoch or self._execution_team is None:
                    return []
                return self._execution_team.agents_for(agent_id)

            session = WorkingAgentSession(runtime, agent_id,
                                          llm_client=self._llm_client(),
                                          event_sink=log_liaison,
                                          execution_plan_sink=prepare_execution,
                                          execution_agents_provider=execution_agents)
            self._agents[agent_id] = session
        return session

    def _llm_client(self) -> Any:
        """Reuse whatever the world is already talking to, so the assistant and
        the organization run on the same model and the same budget."""
        client = getattr(self._session_provider().world, "llm_client", None)
        if client is not None:
            return client
        try:
            from environments.org_env.llm.config import load_org_llm_client
            client, _ = load_org_llm_client()
            return client
        except Exception:                                  # noqa: BLE001
            return None

    def agent_send(self, token: str, text: str, *, reply_to: str = "",
                   thread_id: str = "", require_model: bool = False) -> Dict[str, Any]:
        try:
            seat = self._seat(token)
        except SeatUnavailable as exc:
            return {"error": str(exc)}
        if not text.strip():
            return {"error": "empty_message"}

        return self._agent(seat.agent_id).send(
            text, reply_to=reply_to, thread_id=thread_id,
            require_model=require_model)

    def agent_state(self, token: str, since: int = 0) -> Dict[str, Any]:
        try:
            seat = self._seat(token)
        except SeatUnavailable as exc:
            return {"error": str(exc)}
        state = self._agent(seat.agent_id).state(since)
        if self._execution_team is not None:
            state["execution_jobs"] = self._execution_team.jobs_for(seat.agent_id)
            state["execution_agents"] = self._execution_team.agents_for(seat.agent_id)
        return state

    def agent_cancel_request(self, token: str) -> Dict[str, Any]:
        try:
            seat = self._seat(token)
        except SeatUnavailable as exc:
            return {"error": str(exc)}
        return self._agent(seat.agent_id).cancel_current()

    def agent_retry_request(self, token: str, *, thread_id: str = "") -> Dict[str, Any]:
        try:
            seat = self._seat(token)
        except SeatUnavailable as exc:
            return {"error": str(exc)}
        return self._agent(seat.agent_id).retry_failed(thread_id=thread_id)

    def execution_start(self, token: str, job_id: str) -> Dict[str, Any]:
        try:
            seat = self._seat(token)
        except (SeatUnavailable, RuntimeError) as exc:
            return {"error": str(exc)}
        if self._execution_team is None:
            return {"error": "execution_team_unavailable"}
        return self._execution_team.start(seat.agent_id, str(job_id or ""))

    def execution_cancel(self, token: str, job_id: str) -> Dict[str, Any]:
        try:
            seat = self._seat(token)
        except (SeatUnavailable, RuntimeError) as exc:
            return {"error": str(exc)}
        if self._execution_team is None:
            return {"error": "execution_team_unavailable"}
        return self._execution_team.cancel(seat.agent_id, str(job_id or ""))

    def agent_confirm(self, token: str, draft_id: str) -> Dict[str, Any]:
        try:
            seat = self._seat(token)
        except SeatUnavailable as exc:
            return {"error": str(exc)}
        return self._agent(seat.agent_id).confirm(draft_id)

    def agent_discard(self, token: str, draft_id: str) -> Dict[str, Any]:
        try:
            seat = self._seat(token)
        except SeatUnavailable as exc:
            return {"error": str(exc)}
        return self._agent(seat.agent_id).discard(draft_id)

    # -- clock --------------------------------------------------------------
    def runtime_status(self) -> Dict[str, Any]:
        try:
            return self.runtime().status()
        except RuntimeError as exc:
            return {"error": str(exc), "running": False, "ticks_run": 0}

    def runtime_context(self) -> Dict[str, Any]:
        """Return non-sensitive live identity and clock facts for the liaison.

        ``runtime_status`` belongs to the real-time human clock, while pack,
        engine, event-source, and model attachment live on the inspector
        session.  The P3 UI already displays both; the secretary model must see
        the same verified facts or it will deny information that is visibly on
        screen.  ``runtime_metadata`` is an explicit public allowlist and never
        contains credentials or private agent state.
        """
        payload = dict(self.runtime_status())
        session = self._session_provider()
        metadata = getattr(session, "runtime_metadata", None)
        if callable(metadata):
            payload.update(dict(metadata() or {}))
        return payload

    def runtime_start(self, seconds_per_tick: Optional[float] = None) -> Dict[str, Any]:
        try:
            rt = self.runtime()
        except RuntimeError as exc:
            return {"error": str(exc)}

        if seconds_per_tick:
            rt.seconds_per_tick = max(0.05, float(seconds_per_tick))
            self._seconds_per_tick = rt.seconds_per_tick
        return rt.start()

    def runtime_pause(self) -> Dict[str, Any]:
        try:
            return self.runtime().pause()
        except RuntimeError as exc:
            return {"error": str(exc)}

    def shutdown(self) -> None:
        # Fence callbacks before the bounded worker shutdown begins.
        self._runtime_epoch += 1
        self._agents.clear()
        if self._execution_team is not None:
            self._execution_team.shutdown()
        self._execution_team = None
        self._visible_event_journal.clear()
        self._visible_event_seen.clear()
        self._visible_message_journal.clear()
        self._visible_message_seen.clear()
        if self._runtime is not None:
            self._runtime.shutdown()
            self._runtime = None


__all__ = ["HumanApi"]
