"""Human-controlled member seats — the HCI layer over OrgEnv.

A human drives one real member of the organization: same role, same
permissions, same action space, same information limits as the autonomous seat
it took over. Nothing in-world distinguishes the two (HCI V0 §6/§7) — only the
research-side ``OrgWorld.controller_log`` records who was behind an action.
"""
from environments.org_env.human.seat import HumanSeat, HumanSeatRegistry
from environments.org_env.human.seat_view import build_seat_view

__all__ = ["HumanSeat", "HumanSeatRegistry", "build_seat_view"]
