"""Candidate pool + env-aware feature extraction (Stage B §6, §7).

:class:`CandidatePoolGenerator` turns perception + self-state + inventory + the
visible world into a list of :class:`ActionCandidate` s — **profile-independent**
(§6.1: it reads state/skill/affordance, NEVER personality traits), drawn only
from the real :data:`~agent_sdk.lived.core.actions.ACTION_REGISTRY`, with speech
actions included and **wish excluded** (§6.6). Hard-infeasible actions (capacity
full → gather, no food → eat, too exhausted → high-energy actions) are masked
out (§6 / energy addendum §5).

:class:`EnvAwareFeatureExtractor` annotates each candidate from REAL state
(self-state vitals, inventory, world, distance) + the action template's energy
fields (§7) — not just the action name. It exposes ``extract`` (the
FeatureExtractorPort PCBSP calls) and ``extract_with_reasons`` (for the
FeatureExtractionLog).
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from agent_sdk.lived.core.actions import ACTION_REGISTRY, get_spec
from agent_sdk.lived.core.contracts import ActionCandidate, ActionFeatures, CandidateSource
from agent_sdk.lived.perceive.perception import AgentGT, GroundTruthScene, PerceptionPacket, SelfStatePercept


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(x)))


def _dist(a, b) -> float:
    try:
        return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))
    except (TypeError, ValueError, IndexError):
        return float("inf")


def _is_food(name: str) -> bool:
    name = str(name).lower()
    return any(t in name for t in ("food", "grain", "berry", "meat", "fish",
                                   "tomato", "beef", "cooked", "cow", "fruit", "veg"))


# --------------------------------------------------------------------------- #
# §6 Candidate pool (profile-independent)
# --------------------------------------------------------------------------- #
class CandidatePoolGenerator:
    def generate(self, *, agent: AgentGT, packet: PerceptionPacket,
                 self_state: SelfStatePercept, scene: GroundTruthScene) -> List[ActionCandidate]:
        out: List[ActionCandidate] = []
        energy = float(agent.energy)
        remaining_cap = agent.carrying_capacity - agent.current_load
        ds = self_state.derived_signals or {}

        def add(action_type: str, params=None, source=CandidateSource.ENVIRONMENT, target=None):
            spec = get_spec(action_type)
            if spec is None:
                return
            # hard energy mask (§ energy addendum §5) unless exhausted-ok/emergency
            if energy < spec.min_energy_required and not (spec.can_execute_when_exhausted
                                                          or spec.is_emergency_survival):
                return
            out.append(ActionCandidate(action_type=action_type, parameters=dict(params or {}),
                                       source=source, target_uid=target))

        # always-available
        add("inspect_area", {"radius": agent.vision_radius})
        add("wait_or_continue", {"reason": "no_better_option"})
        add("rest", source=CandidateSource.NEED)
        # SocioGenesis day/night + fatigue (§21): sleep is an option at night OR when
        # exhausted / in a high-fatigue zone. PCBSP (+ hunger압低 in §20.4) then
        # decides whether to take it. Profile never changes the pool, only utility.
        _clock = getattr(scene, "clock", None)
        _night = _clock is not None and getattr(_clock, "is_nighttime", False)
        _tired = bool(getattr(self_state, "is_exhausted", False)) or str(
            getattr(self_state, "fatigue_zone", "normal")) in ("high", "overexertion", "collapse")
        if _night or _tired:
            add("sleep", source=CandidateSource.NEED)

        # survival
        has_food = ds.get("has_food") or any(_is_food(k) and v > 0 for k, v in agent.inventory.items())
        if has_food:
            add("eat_food", source=CandidateSource.NEED)
        if (ds.get("is_injured") or self_state.stress > 0.6) and agent.home_location is not None:
            add("seek_safety", {"target_safe_location": agent.home_location}, source=CandidateSource.NEED)

        # resources (gather masked if capacity full)
        for r in packet.visible_resources:
            rid = r.get("id")
            if rid is None:
                continue
            add("inspect_resource", {"resource_id": rid, "resource_type": r.get("type")})
            if remaining_cap > 0 and float(r.get("amount", 0)) > 0:
                add("gather_resource", {"resource_id": rid, "resource_type": r.get("type"),
                                        "amount_requested": 1}, source=CandidateSource.ENVIRONMENT)
            # generic move toward a visible resource
            add("move_to", {"target_location": [r.get("x"), r.get("y")], "max_steps": 2,
                            "reason_tag": "approach_resource"})

        for o in packet.visible_objects:
            if o.get("id"):
                add("inspect_object", {"object_id": o.get("id")})
        for rec in (packet.public_marks_seen + packet.public_records_seen):
            if rec.get("id"):
                add("inspect_public_record", {"record_id": rec.get("id")})

        # home / storage
        if agent.home_location is not None:
            at_home = _dist((agent.x, agent.y), agent.home_location) <= 2.0
            if not at_home:
                # returning home is only a real option when not already there —
                # offering it at distance 0 creates a high-value no-op loop
                # (full load -> storage_value 1.0) that shadows store_item_home.
                add("return_home", {}, source=CandidateSource.ENVIRONMENT)
            if at_home:
                for item, qty in agent.inventory.items():
                    if qty > 0:
                        add("store_item_home", {"item_id": item, "amount": 1})
                for item, qty in agent.home_storage.items():
                    if qty > 0 and remaining_cap > 0:
                        add("retrieve_item_home", {"item_id": item, "amount": 1})

        # inventory
        for item, qty in agent.inventory.items():
            if qty > 0:
                add("drop_item", {"item_id": item, "amount": 1})

        # craft: offered when the materials at hand cover a known function
        # recipe. Judged purely via MATERIAL_REGISTRY properties (env-agnostic,
        # §6.1: state/affordance, never personality). "At hand" = carried
        # inventory, plus home storage when standing at home (you craft at home
        # with your stores; no need to first carry them). One candidate — the
        # first coverable function in FUNCTION_SPECS order — keeps the pool
        # small and the params deterministic.
        from agent_sdk.lived.world.craft import FUNCTION_SPECS, propose_materials
        craft_pool = dict(agent.inventory)
        if agent.home_location is not None and _dist((agent.x, agent.y), agent.home_location) <= 2.0:
            for item, qty in agent.home_storage.items():
                craft_pool[item] = craft_pool.get(item, 0) + int(qty)
        for fn in FUNCTION_SPECS:
            mats = propose_materials(craft_pool, fn)
            if mats:
                add("attempt_craft", {"intended_function": fn, "materials": mats},
                    source=CandidateSource.NEED)
                break

        # known resource zones (move toward) — explicit zones via agent.extra
        for z in getattr(agent, "extra", {}).get("known_resource_zones", []) or []:
            add("move_to_known_resource", {"resource_zone_id": z})

        # social / speech — only if a visible agent is in comm range
        nearby = [g for g in scene.agents if g.id != agent.id
                  and _dist((agent.x, agent.y), (g.x, g.y)) <= agent.comm_radius]
        if nearby:
            add("comm_local", {"content_summary": "local message", "speech_intent": "tell_info"},
                source=CandidateSource.SOCIAL)
            for g in nearby:
                add("ask_help", {"target_agent_id": g.id, "help_type": "general",
                                 "problem_summary": "need assistance"}, source=CandidateSource.SOCIAL, target=g.id)
                add("tell_info", {"target_agent_id": g.id, "content_summary": "info"},
                    source=CandidateSource.SOCIAL, target=g.id)
                add("promise_action", {"target_agent_id": g.id, "promised_action_type": "help"},
                    source=CandidateSource.SOCIAL, target=g.id)
                add("inspect_agent_visible_state", {"target_agent_id": g.id})
                for item, qty in agent.inventory.items():
                    if qty > 0:
                        add("transfer_item_to_agent", {"target_agent_id": g.id, "item_id": item,
                                                       "amount": 1}, source=CandidateSource.SOCIAL, target=g.id)
                        break  # one transfer candidate per nearby agent is enough

        # camp announcement if in camp
        camp = scene.camp_zone
        if camp and _dist((agent.x, agent.y), (camp.get("x"), camp.get("y"))) <= float(camp.get("radius", 0)):
            add("camp_announce", {"content_summary": "announcement"}, source=CandidateSource.SOCIAL)

        return out


# --------------------------------------------------------------------------- #
# §7 Env-aware feature extractor
# --------------------------------------------------------------------------- #
class EnvAwareFeatureExtractor:
    """Reads real state (vitals/inventory/world/distance) + the action energy
    template to fill :class:`ActionFeatures` (§7). ``state`` is a dict with
    ``agent`` / ``self_state`` / ``scene`` keys."""

    def extract(self, *, agent_id: str, candidate: ActionCandidate, state: Any) -> ActionFeatures:
        feats, _, _ = self.extract_with_reasons(agent_id=agent_id, candidate=candidate, state=state)
        return feats

    def extract_with_reasons(self, *, agent_id: str, candidate: ActionCandidate, state: Any
                             ) -> Tuple[ActionFeatures, Dict[str, str], List[str]]:
        agent: AgentGT = (state or {}).get("agent")
        ss: SelfStatePercept = (state or {}).get("self_state")
        scene: GroundTruthScene = (state or {}).get("scene")
        spec = get_spec(candidate.action_type)
        f: Dict[str, float] = {}
        sources: Dict[str, str] = {}
        reasons: List[str] = []

        hunger = float(getattr(ss, "hunger_pressure", 0.0)) if ss else 0.0
        fatigue = float(getattr(ss, "fatigue", 0.0)) if ss else 0.0
        exhaustion = float(getattr(ss, "exhaustion_pressure", 0.0)) if ss else 0.0
        fatigue_zone = str(getattr(ss, "fatigue_zone", "normal")) if ss else "normal"
        winter = bool(scene and scene.season == "winter")
        cap = float(getattr(agent, "carrying_capacity", 1) or 1) if agent else 1.0
        load = float(getattr(agent, "current_load", 0)) if agent else 0.0
        load_ratio = _clamp(load / cap) if cap else 0.0
        a = candidate.action_type

        # energy/time cost from the action template (energy addendum §4).
        # Gentle normalization: routine costs stay modest so loss-aversion (λ=2
        # in the transform) doesn't drown out the action's gains.
        if spec is not None:
            f["energy_cost"] = _clamp(spec.base_energy_cost / 20.0)
            f["time_cost"] = _clamp((spec.base_energy_cost + 1) / 24.0)
            sources["energy_cost"] = "action_template.base_energy_cost"
            if spec.energy_recovery > 0:
                f["energy_gain"] = max(f.get("energy_gain", 0.0), _clamp(spec.energy_recovery / 25.0))
                f["energy_cost"] = 0.0
                reasons.append("restorative action")

        if a == "eat_food":
            # SocioGenesis: eat restores SATIETY (hunger) — big survival when hungry,
            # only a small energy bump (§12). hunger comes from satiety, not energy.
            f["survival_gain"] = _clamp(0.3 + 0.7 * hunger)
            f["energy_gain"] = _clamp(0.15 + 0.15 * hunger)   # small energy from food
            reasons.append(f"hunger={hunger:.2f} -> satiety/survival scaled")
        elif a == "rest":
            # rest restores energy + lowers fatigue; it does NOT satisfy hunger, so
            # no survival_gain from hunger (§13/§20.4).
            f["energy_gain"] = _clamp(0.4 + 0.5 * exhaustion)
            reasons.append(f"exhaustion={exhaustion:.2f}")
        elif a == "sleep":
            f["energy_gain"] = _clamp(0.5 + 0.5 * exhaustion)
            f["survival_gain"] = _clamp(0.2 * exhaustion)
            # §20.4: a starving agent should eat/gather/return_home, not sleep —
            # push sleep utility down hard when very hungry (unless it can't act).
            if hunger > 0.6:
                f["energy_gain"] = _clamp(f["energy_gain"] * (1.0 - 0.6 * hunger))
                f["survival_gain"] = 0.0
            reasons.append(f"exhaustion={exhaustion:.2f}, hunger={hunger:.2f}")
        elif a == "gather_resource":
            rtype = str(candidate.parameters.get("resource_type") or "")
            f["resource_gain"] = 0.75
            if _is_food(rtype):
                f["survival_gain"] = _clamp(0.3 + 0.5 * hunger)   # food gathering vs hunger
            f["future_security_gain"] = _clamp(0.3 + (0.4 if winter else 0.0))
            if winter:
                reasons.append("winter -> future_security_gain higher")
        elif a in ("return_home",):
            f["storage_value"] = _clamp(load_ratio)
            f["future_security_gain"] = _clamp(0.3 * load_ratio + (0.3 if winter else 0.0))
            if hunger > 0.5:
                f["survival_gain"] = _clamp(0.3 * hunger)
            if agent and agent.home_location is not None:
                d = _dist((agent.x, agent.y), agent.home_location)
                f["time_cost"] = _clamp(d / 60.0)
                f["energy_cost"] = _clamp(0.2 + d / 80.0)
                reasons.append(f"distance_to_home={d:.1f}")
        elif a in ("move_to", "move_to_known_resource", "seek_safety"):
            tgt = candidate.parameters.get("target_location") or candidate.parameters.get("target_safe_location")
            if agent and tgt:
                d = _dist((agent.x, agent.y), tgt)
                f["energy_cost"] = _clamp(0.2 + d / 80.0)
                f["time_cost"] = _clamp(d / 60.0)
            if a == "seek_safety":
                f["survival_gain"] = _clamp(0.4 + 0.3 * float(getattr(ss, "stress", 0.0) if ss else 0))
                f["risk_cost"] = 0.1
            else:
                f["information_gain"] = 0.3
        elif a == "attempt_craft":
            # crafting converts carried materials into a lasting affordance:
            # worth more the tighter carrying capacity is (real state), and it
            # is genuinely experimental + skill-building for an untested maker.
            f["future_security_gain"] = _clamp(0.4 + 0.4 * load_ratio + (0.2 if winter else 0.0))
            f["experiment_value"] = 0.5
            f["skill_gain"] = 0.5
            f["novelty_value"] = 0.25
            # the outcome (quality band) is uncertain; material downside is NOT
            # also charged — a scrap result returns most inputs (craft.py), so
            # charging material_cost on top would double-count the same risk.
            f["uncertainty_cost"] = 0.25
            reasons.append(f"load_ratio={load_ratio:.2f} -> capacity relief value")
        elif a in ("inspect_resource", "inspect_object", "inspect_public_record",
                   "inspect_area", "inspect_agent_visible_state"):
            # information diminishes with familiarity: re-inspecting the same
            # target teaches less each time (real state via inspect counts kept
            # by the scene handler / takeover controller). Without this a
            # curious profile inspect-loops forever on one resource.
            tgt = str(candidate.parameters.get("resource_id")
                      or candidate.parameters.get("object_id")
                      or candidate.parameters.get("record_id")
                      or candidate.parameters.get("target_agent_id") or "_area")
            counts = ((getattr(agent, "extra", {}) or {}).get("inspect_counts") or {}) if agent else {}
            scounts = ((getattr(scene, "inspected", {}) or {}).get("_counts") or {}) if scene else {}
            seen = max(int(counts.get(tgt, 0) or 0), int(scounts.get(tgt, 0) or 0))
            f["information_gain"] = (0.6 if a == "inspect_resource" else 0.4) / (1.0 + seen)
            if seen:
                reasons.append(f"inspected {tgt} x{seen} -> information diminishes")
            if a == "inspect_resource":
                f["experiment_value"] = 0.3 / (1.0 + seen)
            # §20.5: looking around has rising opportunity cost when hungry, unless
            # it's a food source we're inspecting.
            if hunger > 0.4 and not _is_food(str(candidate.parameters.get("resource_type") or "")):
                f["time_cost"] = max(f.get("time_cost", 0.0), _clamp(0.2 + 0.5 * hunger))
        elif a == "ask_help":
            f["reciprocity_gain"] = 0.3
            f["reputation_risk"] = 0.25
            reasons.append("dependency/ask -> reputation_risk")
        elif a == "tell_info":
            f["altruistic_gain"] = 0.4
            f["reciprocity_gain"] = 0.3
            f["reputation_gain"] = 0.3
        elif a == "comm_local":
            f["information_gain"] = 0.2
        elif a == "camp_announce":
            f["coordination_gain"] = 0.6
            f["dominance_gain"] = 0.4
            f["public_good_gain"] = 0.4
            f["reputation_gain"] = 0.3
            f["conflict_risk"] = 0.3
            reasons.append("public broadcast -> reputation/conflict")
        elif a == "transfer_item_to_agent":
            f["altruistic_gain"] = 0.6
            f["reciprocity_gain"] = 0.4
            f["trust_gain"] = 0.4
            f["material_cost"] = 0.3
        elif a == "promise_action":
            f["reciprocity_gain"] = 0.4
            f["reputation_gain"] = 0.3
        elif a in ("store_item_home",):
            f["future_security_gain"] = 0.5
            f["storage_value"] = 0.5
        elif a in ("retrieve_item_home",):
            f["resource_gain"] = 0.4
            f["private_gain"] = 0.4
        elif a == "apologize":
            f["reputation_gain"] = 0.3
            f["trust_gain"] = 0.4
        elif a == "explain_action":
            f["reputation_gain"] = 0.3

        # §20.6 fatigue effects: in the high/overexertion zone, high-energy actions
        # cost more energy + carry more risk (push-through is costlier). Recovery
        # actions are exempt. Important plan/social value still competes in PCBSP.
        if fatigue_zone in ("high", "overexertion", "collapse") and a not in (
                "rest", "sleep", "eat_food", "wait_or_continue"):
            bump = 1.3 if fatigue_zone == "high" else 1.6
            if f.get("energy_cost", 0.0) > 0:
                f["energy_cost"] = _clamp(f["energy_cost"] * bump)
            if spec is not None and spec.base_energy_cost >= 4.0:
                f["risk_cost"] = max(f.get("risk_cost", 0.0), 0.25 if fatigue_zone == "high" else 0.4)
            reasons.append(f"fatigue_zone={fatigue_zone} -> energy/risk up")

        feats = ActionFeatures(**{k: v for k, v in f.items() if k in ActionFeatures.feature_names()})
        for k in f:
            sources.setdefault(k, "env_state")
        return feats, sources, reasons


def candidate_key(candidate: ActionCandidate) -> str:
    """Stable per-candidate identity: action_type + params. Distinguishes e.g.
    gather_resource(tomato) from gather_resource(reed) so per-target features
    (a food gather's survival_gain) never collapse onto the wrong target."""
    import json
    try:
        p = json.dumps(candidate.parameters or {}, sort_keys=True, default=str)
    except (TypeError, ValueError):
        p = str(candidate.parameters)
    return f"{candidate.action_type}|{p}"


class PrecomputedFeatureExtractor:
    """Wraps a {candidate_key: ActionFeatures} map so the control loop can
    compute + log features once (with reasons) then hand PCBSP the same
    vectors. Falls back to a plain action_type key for older tables."""

    def __init__(self, table: Dict[str, ActionFeatures]):
        self.table = table

    def extract(self, *, agent_id: str, candidate: ActionCandidate, state: Any) -> ActionFeatures:
        feats = self.table.get(candidate_key(candidate))
        if feats is not None:
            return feats
        return self.table.get(candidate.action_type, ActionFeatures())
