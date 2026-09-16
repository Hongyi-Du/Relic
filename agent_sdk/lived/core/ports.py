"""Pluggable ports — the SDK<->Env seam for the lived decision system.

The boundary rule (CLAUDE.md): ``agent_sdk`` never imports ``environments``.
Anything that needs world/grid/energy state is expressed as a Protocol here;
the environment provides the concrete implementation and injects it at wiring
time (mirroring ``agent_sdk.mechanisms`` port pattern).

Ports:
  * ``FeatureExtractorPort``   — (state, candidate) -> ActionFeatures (§14.3)
  * ``CandidateSourcePort``    — observation/graphs -> [ActionCandidate] (§14.2)
  * ``GraphStorePort``         — minimal node/edge query surface (§4-7)
  * ``WishParserPort``         — thought -> [Need] (§11)
  * ``AffordanceSynthPort``    — Need -> [AffordanceProposal] (§12.2)

All are ``@runtime_checkable`` so env components can be duck-typed and the
scaffold's NoOp fallbacks can be asserted in tests.
"""
from __future__ import annotations

from typing import Any, List, Protocol, runtime_checkable

from agent_sdk.lived.core.contracts import (
    ActionCandidate,
    ActionFeatures,
    AffordanceProposal,
    Need,
)


@runtime_checkable
class FeatureExtractorPort(Protocol):
    """Maps a candidate action in the current decision state to a feature
    vector (§14.3). Implementations should be MOSTLY deterministic functions of
    env state to keep per-turn cost bounded."""

    def extract(self, *, agent_id: str, candidate: ActionCandidate, state: Any) -> ActionFeatures:
        ...


@runtime_checkable
class CandidateSourcePort(Protocol):
    """Generates candidate actions from one of the §14.2 sources. The
    controller fans out across multiple sources and concatenates results."""

    def generate(self, *, agent_id: str, state: Any) -> List[ActionCandidate]:
        ...


@runtime_checkable
class GraphStorePort(Protocol):
    """Minimal query surface the policy / detector need from any graph (§4-7).

    Concrete stores live in ``agent_sdk.lived.graphs``; envs may also supply
    richer query objects, but the controller only relies on this surface."""

    def add_node(self, node_id: str, ntype: str, **attrs: Any) -> None: ...
    def add_edge(self, src: str, dst: str, etype: str, **attrs: Any) -> None: ...
    def neighbors(self, node_id: str, etype: str | None = None) -> List[str]: ...
    def edge_attr(self, src: str, dst: str, etype: str, key: str, default: Any = None) -> Any: ...


@runtime_checkable
class WishParserPort(Protocol):
    """Extracts structured Needs from a ReAct thought (§11)."""

    def parse(self, *, agent_id: str, thought: str, state: Any) -> List[Need]:
        ...


@runtime_checkable
class AffordanceSynthPort(Protocol):
    """Synthesizes candidate affordances from a Need (§12.2)."""

    def synthesize(self, *, need: Need, state: Any) -> List[AffordanceProposal]:
        ...


# --------------------------------------------------------------------------- #
# NoOp fallbacks — let the controller run end-to-end before env wiring exists.
# --------------------------------------------------------------------------- #
class NoOpFeatureExtractor:
    """Returns an all-zero feature vector. Replace with an env extractor."""

    def extract(self, *, agent_id: str, candidate: ActionCandidate, state: Any) -> ActionFeatures:
        return ActionFeatures()


class NoOpCandidateSource:
    """Yields no candidates. Replace with env/persona/social sources."""

    def generate(self, *, agent_id: str, state: Any) -> List[ActionCandidate]:
        return []
