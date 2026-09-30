"""OrgEnv communication system (DESIGN env_org §16-§20/§38):
  channels.py — Channel + DEFAULT_CHANNELS
  messages.py — Message + Attachment (object references, read receipts)
  system.py   — CommunicationSystem (send/share/read, information asymmetry)
"""
from environments.org_env.backend.comm.channels import DEFAULT_CHANNELS, Channel
from environments.org_env.backend.comm.messages import Attachment, Message
from environments.org_env.backend.comm.system import CommunicationSystem

__all__ = ["DEFAULT_CHANNELS", "Channel", "Attachment", "Message", "CommunicationSystem"]
