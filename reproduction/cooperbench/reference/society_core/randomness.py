"""Seeded random streams with replayable draw references."""

from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass
class RandomDraw:
    stream: str
    index: int
    kind: str
    value: float


class SeededRandom:
    """Named deterministic random streams.

    Draw refs are stored as small records so event logs can cite the randomness
    that produced an action without relying on process-global RNG state.
    """

    def __init__(self, seed: int, streams: tuple[str, ...] = ("action", "body", "feed", "event")) -> None:
        self.seed = seed
        self._streams = {
            name: random.Random(f"{seed}:{name}") for name in streams
        }
        self._counts = {name: 0 for name in streams}
        self.draws: list[RandomDraw] = []

    def random(self, stream: str, kind: str = "random") -> float:
        if stream not in self._streams:
            self._streams[stream] = random.Random(f"{self.seed}:{stream}")
            self._counts[stream] = 0
        value = self._streams[stream].random()
        index = self._counts[stream]
        self._counts[stream] += 1
        self.draws.append(RandomDraw(stream=stream, index=index, kind=kind, value=value))
        return value

    def uniform(self, stream: str, low: float, high: float, kind: str = "uniform") -> float:
        return low + (high - low) * self.random(stream, kind=kind)

    def choice_index(self, stream: str, weights: list[float], kind: str = "choice") -> int:
        total = sum(max(0.0, weight) for weight in weights)
        if total <= 0:
            return 0
        draw = self.random(stream, kind=kind) * total
        cumulative = 0.0
        for index, weight in enumerate(weights):
            cumulative += max(0.0, weight)
            if draw <= cumulative:
                return index
        return len(weights) - 1

    def refs_since(self, start_index: int) -> list[str]:
        refs = []
        for draw in self.draws[start_index:]:
            refs.append(f"{draw.stream}:{draw.index}:{draw.kind}:{draw.value:.12f}")
        return refs

    def snapshot(self) -> dict[str, int]:
        return dict(self._counts)
