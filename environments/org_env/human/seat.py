"""Human seats and the registry that tracks them.

A seat is a claim on an existing member: the human takes over who decides, and
nothing else about that member changes. Several people can hold different seats
in the same world at once, each with their own token and their own view.
"""
from __future__ import annotations

import secrets
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

#: A seat with no traffic for this long is reported as offline. It is not
#: released — the member keeps its work; only the presence indicator changes.
OFFLINE_AFTER_SECONDS = 90.0


@dataclass
class HumanSeat:
    agent_id: str
    token: str
    display_name: str = ""
    claimed_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)

    def touch(self) -> None:
        self.last_seen = time.time()

    def is_online(self, now: Optional[float] = None) -> bool:
        return ((now or time.time()) - self.last_seen) < OFFLINE_AFTER_SECONDS

    def public_state(self, now: Optional[float] = None) -> Dict[str, Any]:
        """What may be shown about this seat. Never the token, and never that a
        human is behind it — other members must not be able to tell."""
        return {"agent_id": self.agent_id, "online": self.is_online(now),
                "last_seen": self.last_seen}


class SeatUnavailable(Exception):
    """Raised when a seat cannot be claimed (unknown member, or already held)."""


class HumanSeatRegistry:
    """Owns the human seats of one world.

    Claiming marks the member as human-controlled so ``OrgWorld.step`` stops
    deciding for it; releasing hands it back to the autonomous loop, so a run
    can move a seat between controllers without rebuilding the world.
    """

    def __init__(self, world: Any) -> None:
        self.world = world
        self.seats: Dict[str, HumanSeat] = {}          # agent_id -> seat
        self._by_token: Dict[str, str] = {}            # token -> agent_id

    # -- lifecycle ---------------------------------------------------------
    def claim(self, agent_id: str, *, display_name: str = "") -> HumanSeat:
        if agent_id not in self.world.agents:
            raise SeatUnavailable(f"unknown_member:{agent_id}")
        if agent_id in self.seats:
            raise SeatUnavailable(f"seat_already_claimed:{agent_id}")
        seat = HumanSeat(agent_id=agent_id, token=secrets.token_urlsafe(24),
                         display_name=display_name)
        self.seats[agent_id] = seat
        self._by_token[seat.token] = agent_id
        self.world.assign_human_seat(agent_id)
        return seat

    def release(self, agent_id: str) -> None:
        seat = self.seats.pop(agent_id, None)
        if seat is not None:
            self._by_token.pop(seat.token, None)
        self.world.release_human_seat(agent_id)

    def release_all(self) -> None:
        for agent_id in list(self.seats):
            self.release(agent_id)

    # -- lookup ------------------------------------------------------------
    def authenticate(self, token: str) -> HumanSeat:
        """Resolve a token to its seat, refreshing presence. Raises on unknown
        tokens so a caller can never act as a seat it does not hold."""
        agent_id = self._by_token.get(token or "")
        seat = self.seats.get(agent_id) if agent_id else None
        if seat is None:
            raise SeatUnavailable("invalid_seat_token")
        seat.touch()
        return seat

    def get(self, agent_id: str) -> Optional[HumanSeat]:
        return self.seats.get(agent_id)

    def claimable(self) -> List[Dict[str, Any]]:
        """Members a human could take over, with enough detail to choose one."""
        out = []
        for agent_id, agent in self.world.agents.items():
            out.append({"agent_id": agent_id, "name": agent.name,
                        "role": agent.role, "codename": agent.codename,
                        "is_founder": agent.is_founder,
                        "identity": agent.initial_identity,
                        "claimed": agent_id in self.seats})
        return out


__all__ = ["HumanSeat", "HumanSeatRegistry", "SeatUnavailable", "OFFLINE_AFTER_SECONDS"]
