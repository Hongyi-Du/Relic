"""OrgEnv episode layer — organizes scattered events into causal organizational
episodes (feedback → discussion → artifact/protocol → outcome).

See ``episode.py`` (data + ontology) and ``episode_manager.py`` (open / attach /
close / summarize). No per-agent hardcoding: episodes form from generic trigger /
attachment / closure rules over the event stream + the agent's own role.
"""
from environments.org_env.episodes.episode import (
    EPISODE_TYPES,
    EpEvent,
    OrgEpisode,
    classify_object,
    trigger_episode_type,
)
from environments.org_env.episodes.episode_manager import OrgEpisodeManager

__all__ = [
    "EPISODE_TYPES", "EpEvent", "OrgEpisode", "classify_object",
    "trigger_episode_type", "OrgEpisodeManager",
]
