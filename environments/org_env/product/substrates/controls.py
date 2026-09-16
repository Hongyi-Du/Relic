"""Anti-scripting control conditions for OSS time-machine substrates (brief §12).

For the paper's "the environment did not script the emergence" defense, the substrate supports a set
of CONTROL conditions that perturb the friction / affordance / event-ordering so a capability that
forms in the main condition should NOT form (or should not transfer) under the controls:

* ``none``                                  — main condition (real friction + real affordance + real order).
* ``friction_present_affordance_absent``    — issues remain, but the *means to act* is removed (no
  backlog tasks generated, acceptance hints stripped). Capability should not form without affordance.
* ``affordance_present_friction_absent``    — the runnable product + tools remain, but there is no
  friction (no issues / no stream). Capability should not form without a problem to solve.
* ``shuffled_event_history_detector_control`` — the historical-issue release order is shuffled
  (deterministically by seed), breaking the real timeline. A detector that depends on real ordering
  should lose signal — a control for "the detector just read the script".
* ``heldout_issue_transfer``                — main condition + held-out issues reserved for a transfer
  test (their existence is the control; they are never released during formation).

These are dataset-/seed-level transformations applied at seeding time (brief §12: phase-1 implements
the flags + tests, not the full experiment matrix).
"""
from __future__ import annotations

import random
from dataclasses import replace
from typing import Any, Dict, List

NONE = "none"
FRICTION_PRESENT_AFFORDANCE_ABSENT = "friction_present_affordance_absent"
AFFORDANCE_PRESENT_FRICTION_ABSENT = "affordance_present_friction_absent"
SHUFFLED_EVENT_HISTORY = "shuffled_event_history_detector_control"
HELDOUT_ISSUE_TRANSFER = "heldout_issue_transfer"

CONTROLS = (NONE, FRICTION_PRESENT_AFFORDANCE_ABSENT, AFFORDANCE_PRESENT_FRICTION_ABSENT,
            SHUFFLED_EVENT_HISTORY, HELDOUT_ISSUE_TRANSFER)


def resolve(substrate_config: Dict[str, Any], manifest: Dict[str, Any]) -> str:
    """Pick the control from substrate_config (scenario), else the dataset manifest default."""
    c = (substrate_config or {}).get("control")
    if not c:
        c = ((manifest or {}).get("controls") or {}).get("default")
    c = c or NONE
    return c if c in CONTROLS else NONE


def generates_backlog(control: str) -> bool:
    """Whether to create the issue→task backlog (the affordance to act on friction)."""
    return control != FRICTION_PRESENT_AFFORDANCE_ABSENT


def apply_to_issues(control: str, public_issues: List[Any]) -> List[Any]:
    """Transform the public issues for a control condition.

    * affordance_present_friction_absent -> no issues (friction removed).
    * friction_present_affordance_absent -> issues kept, but acceptance hints stripped (no how-to).
    """
    if control == AFFORDANCE_PRESENT_FRICTION_ABSENT:
        return []
    if control == FRICTION_PRESENT_AFFORDANCE_ABSENT:
        return [replace(i, acceptance_hint="", linked_hidden_test_ids=list(i.linked_hidden_test_ids))
                for i in public_issues]
    return list(public_issues)


def apply_to_stream(control: str, stream: List[Dict[str, Any]], seed: int) -> List[Dict[str, Any]]:
    """Shuffle the release schedule for the detector control (deterministic by seed)."""
    if control == SHUFFLED_EVENT_HISTORY and len(stream) > 1:
        ticks = [int(e.get("release_tick", 0) or 0) for e in stream]
        rng = random.Random(int(seed) * 131 + 7)
        rng.shuffle(ticks)
        for e, t in zip(stream, ticks):
            e["release_tick"] = t
            e["shuffled"] = True
    return stream


def control_meta(control: str) -> Dict[str, Any]:
    return {
        "control": control,
        "friction_present": control != AFFORDANCE_PRESENT_FRICTION_ABSENT,
        "affordance_present": control != FRICTION_PRESENT_AFFORDANCE_ABSENT,
        "event_order": "shuffled" if control == SHUFFLED_EVENT_HISTORY else "historical",
        "heldout_transfer": control == HELDOUT_ISSUE_TRANSFER,
    }


__all__ = [
    "CONTROLS", "NONE", "FRICTION_PRESENT_AFFORDANCE_ABSENT", "AFFORDANCE_PRESENT_FRICTION_ABSENT",
    "SHUFFLED_EVENT_HISTORY", "HELDOUT_ISSUE_TRANSFER",
    "resolve", "generates_backlog", "apply_to_issues", "apply_to_stream", "control_meta",
]
