"""Policy guardrails shared by the P2 and P3 HCI conditions."""

import inspect

from environments.org_env.backend.comm.messages import Message
from environments.org_env.backend.simulation.world import OrgWorld
from environments.org_env.runtime_adapter.perception import OrgPerceptionPacket


def test_the_organization_cannot_detect_or_prioritize_a_human_controller():
    """Controller provenance belongs in controller_log, not the social world.

    Equal standing means the organization receives a normal member action.  It
    must not get an in-world `from_human` signal or a hidden scoring multiplier.
    """
    assert "from_human" not in Message.__dataclass_fields__
    assert "unread_human_messages" not in OrgPerceptionPacket.__dataclass_fields__

    source = inspect.getsource(OrgWorld)
    assert "_current_action_from_human" not in source
    assert "unread_human_messages" not in source
    assert "5x multiplier on communication actions" not in source
