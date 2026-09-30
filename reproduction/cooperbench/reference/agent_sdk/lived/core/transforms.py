"""Feature transforms (PCBSP §6) — prospect-style nonlinearities.

Raw feature intensities (the discrete {0,.25,.5,.75,1.0} the annotator emits, §5)
do NOT enter utility linearly. Each feature is mapped through one of four fixed
transforms (declared per-feature in ``schema.FEATURE_SPECS``):

  * bounded_linear   (§6.1) — identity, clamped to [0,1].
  * diminishing_gain (§6.2) — log marginal-decreasing gain; x=1 still →1.
  * loss_aversion    (§6.3) — diminishing curve × λ_loss (>1), for costs/risks.
  * survival         (§6.4) — same diminishing curve for the three survival
    gains; the emergency *pressure* (hunger / hp sigmoid) is computed
    separately and multiplies these in :mod:`agent_sdk.lived.persona.pcbsp`.

These are standard behavioral-decision intuitions (diminishing returns, loss
aversion, threshold-like survival urgency) — see §17.2. All constants are
module-level so the loss-aversion / β sensitivity sweeps (§17.5) can vary them.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

from agent_sdk.lived.core.contracts import ActionFeatures
from agent_sdk.lived.core.schema import FEATURE_SPECS, TransformKind

# §6.2/§6.3 constants (tunable for sensitivity analysis, §17.5).
BETA = 4.0
LAMBDA_LOSS = 2.0

# §6.4 emergency thresholds.
ENERGY_THRESHOLD = 30.0
ENERGY_SCALE = 8.0
HP_THRESHOLD = 40.0
HP_SCALE = 10.0


def _diminishing(x: float, beta: float = BETA) -> float:
    """§6.2 log marginal-decreasing gain. f(0)=0, f(1)=1, concave between."""
    if x <= 0.0:
        return 0.0
    x = min(1.0, x)
    return math.log(1.0 + beta * x) / math.log(1.0 + beta)


def transform_value(feature: str, x: float) -> float:
    """Apply the §6 transform declared for ``feature`` to a raw intensity ``x``.

    Costs are returned as POSITIVE magnitudes (loss aversion amplifies them); the
    policy applies the negative sign. ``bounded_linear`` simply clamps."""
    spec = FEATURE_SPECS.get(feature)
    kind = spec.transform if spec else TransformKind.BOUNDED_LINEAR
    if kind == TransformKind.BOUNDED_LINEAR:
        return max(0.0, min(1.0, x))
    if kind in (TransformKind.DIMINISHING_GAIN, TransformKind.SURVIVAL):
        return _diminishing(x)
    if kind == TransformKind.LOSS_AVERSION:
        return LAMBDA_LOSS * _diminishing(x)
    return max(0.0, min(1.0, x))


def transform_features(raw: ActionFeatures) -> Dict[str, float]:
    """Transform every field of an :class:`ActionFeatures` to its decision value
    (§6). Returns a plain name→value dict the scorer consumes."""
    out: Dict[str, float] = {}
    for name in ActionFeatures.feature_names():
        out[name] = transform_value(name, float(getattr(raw, name)))
    return out


# --------------------------------------------------------------------------- #
# §6.4 emergency / hunger sigmoids
# --------------------------------------------------------------------------- #
def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def hunger_pressure(energy: float,
                    threshold: float = ENERGY_THRESHOLD,
                    scale: float = ENERGY_SCALE) -> float:
    """[0,1] urgency that rises sharply as energy falls below threshold (§6.4)."""
    return _sigmoid((threshold - energy) / scale)


def hp_pressure(hp: float,
                threshold: float = HP_THRESHOLD,
                scale: float = HP_SCALE) -> float:
    """[0,1] urgency that rises sharply as hp falls below threshold (§6.4)."""
    return _sigmoid((threshold - hp) / scale)


@dataclass
class SurvivalPressure:
    """Computed once per decision from energy/hp (§6.4). Feeds SurvivalScore and
    the Survival Guard (§13)."""
    hunger: float = 0.0
    hp: float = 0.0

    @property
    def max(self) -> float:
        return max(self.hunger, self.hp)

    @classmethod
    def from_vitals(cls, energy: float | None, hp: float | None) -> "SurvivalPressure":
        return cls(
            hunger=hunger_pressure(energy) if energy is not None else 0.0,
            hp=hp_pressure(hp) if hp is not None else 0.0,
        )
