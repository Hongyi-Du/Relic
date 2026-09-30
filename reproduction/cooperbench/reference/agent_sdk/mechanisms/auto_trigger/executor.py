"""
auto_trigger/executor.py — ActionProgramExecutor.

Interprets the ActionProgram DSL (node tree) and produces AgentAction dicts.
Supports multi-turn stateful programs (sequence, while, move_then_act).
Also handles emit_signal nodes (free, no adapter.execute).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from .types import (
    AgentReflexEntry,
    ReflexContext,
    ReflexSignal,
    ReflexSignalType,
    TriggerResult,
)

logger = logging.getLogger(__name__)

# Dedup key cache for unknown action_program node warnings (same rationale
# as evaluator._LOGGED_UNKNOWN_KEYS).
_LOGGED_UNKNOWN_NODES: Set[tuple] = set()


class ActionProgramExecutor:
    """
    Execute one step of an ActionProgram node tree.

    Each invocation advances the program by one turn. Multi-turn programs
    (sequence, while, move_then_act) store progress in entry.program_state.

    Returns:
        (action_dict | None, signals_to_emit: List[ReflexSignal], mode_activations: List[str])

    * ``action_dict`` is ``{"type": ..., "parameters": {...}}`` or ``None``
      for emit_signal-only / activate_mode-only fires.
    * ``signals_to_emit`` are ReflexSignal objects to push onto the signal
      bus. Pair emitters stamp ``pair_id`` + ``emit_id`` on the signal so
      PairClosureTracker can match consumption.
    * ``mode_activations`` is a list of ``mode_id`` strings. The agent core
      forwards them to the ``ModeManager`` to register TTL windows.
    """

    def step(
        self,
        program: dict,
        ctx: ReflexContext,
        entry: AgentReflexEntry,
        matched_signal: Optional[ReflexSignal] = None,
    ) -> Tuple[Optional[Dict[str, Any]], List[ReflexSignal], List[str]]:
        signals: List[ReflexSignal] = []
        mode_activations: List[str] = []
        action = self._exec_node(program, ctx, entry, matched_signal, signals, mode_activations)
        return action, signals, mode_activations

    def _exec_node(
        self,
        node: dict,
        ctx: ReflexContext,
        entry: AgentReflexEntry,
        matched_signal: Optional[ReflexSignal],
        signals: List[ReflexSignal],
        mode_activations: List[str],
    ) -> Optional[Dict[str, Any]]:
        ntype = node.get("node", "")
        if ntype == "single":
            return self._exec_single(node, ctx, matched_signal)
        elif ntype == "move_then_act":
            return self._exec_move_then_act(node, ctx, entry, matched_signal, signals, mode_activations)
        elif ntype == "conditional":
            return self._exec_conditional(node, ctx, entry, matched_signal, signals, mode_activations)
        elif ntype == "sequence":
            return self._exec_sequence(node, ctx, entry, matched_signal, signals, mode_activations)
        elif ntype == "repeat":
            return self._exec_repeat(node, ctx, entry, matched_signal, signals, mode_activations)
        elif ntype == "while":
            return self._exec_while(node, ctx, entry, matched_signal, signals, mode_activations)
        elif ntype == "emit_signal":
            self._exec_emit_signal(node, ctx, entry, matched_signal, signals)
            return None
        elif ntype == "activate_mode":
            self._exec_activate_mode(node, ctx, entry, mode_activations)
            return None
        else:
            key = (id(node), str(ntype))
            if key not in _LOGGED_UNKNOWN_NODES:
                _LOGGED_UNKNOWN_NODES.add(key)
                logger.warning(
                    "[Executor] Unknown action_program node %r — "
                    "this reflex will never produce an action. Use "
                    "publish-time schema validation "
                    "(auto_trigger/validation.py) to catch this before "
                    "it lands in the store.", ntype
                )
            return None

    # -- single --

    def _exec_single(
        self, node: dict, ctx: ReflexContext,
        matched_signal: Optional[ReflexSignal],
    ) -> Dict[str, Any]:
        action_type = node.get("action_type", "idle")
        params = dict(node.get("params", {}))
        params = self._resolve_params(params, ctx, matched_signal)
        return {"type": action_type, "parameters": params}

    # -- move_then_act --

    def _exec_move_then_act(
        self, node: dict, ctx: ReflexContext,
        entry: AgentReflexEntry,
        matched_signal: Optional[ReflexSignal],
        signals: List[ReflexSignal],
        mode_activations: List[str],
    ) -> Optional[Dict[str, Any]]:
        """Two-turn execution: turn 1 always moves, turn 2 always acts.

        Even if the agent is already within arrival_distance, the move phase
        still emits a (possibly zero-distance) move on turn 1 so that the
        action is always deferred to turn 2.  This prevents a single reflex
        from consuming two conceptual actions in one turn.
        """
        state = entry.program_state
        phase = state.get("mta_phase", "MOVING")
        arrival_distance = node.get("arrival_distance", 1)

        if phase == "MOVING":
            target = self._resolve_value(node.get("target", ""), ctx, matched_signal)
            tx, ty = self._parse_position(target)
            if tx is None or ty is None:
                state["mta_phase"] = "DONE"
                return {"type": "idle", "parameters": {}}

            dist = max(abs(ctx.x - tx), abs(ctx.y - ty))
            if dist <= arrival_distance:
                # Already close enough — skip straight to ACTING next turn
                state["mta_phase"] = "ACTING"
                state["mta_target"] = (tx, ty)
                return {"type": "move", "parameters": {"x": tx, "y": ty}}
            # Still far — move toward target, stay in MOVING for re-evaluation
            state["mta_phase"] = "ACTING"
            state["mta_target"] = (tx, ty)
            return {"type": "move", "parameters": {"x": tx, "y": ty}}

        elif phase == "ACTING":
            state["mta_phase"] = "DONE"
            return self._exec_node(
                node.get("action", {}), ctx, entry, matched_signal, signals, mode_activations
            )
        else:
            state["mta_phase"] = "MOVING"
            return None

    # -- conditional --

    def _exec_conditional(
        self, node: dict, ctx: ReflexContext,
        entry: AgentReflexEntry,
        matched_signal: Optional[ReflexSignal],
        signals: List[ReflexSignal],
        mode_activations: List[str],
    ) -> Optional[Dict[str, Any]]:
        from .evaluator import ConditionEvaluator
        evaluator = ConditionEvaluator()
        cond = node.get("condition", {})
        result, sig = evaluator.evaluate(cond, ctx)
        if sig is not None and matched_signal is None:
            matched_signal = sig
        if result:
            return self._exec_node(
                node.get("then", {}), ctx, entry, matched_signal, signals, mode_activations
            )
        elif "else" in node:
            return self._exec_node(
                node["else"], ctx, entry, matched_signal, signals, mode_activations
            )
        return None

    # -- sequence --

    def _exec_sequence(
        self, node: dict, ctx: ReflexContext,
        entry: AgentReflexEntry,
        matched_signal: Optional[ReflexSignal],
        signals: List[ReflexSignal],
        mode_activations: List[str],
    ) -> Optional[Dict[str, Any]]:
        steps = node.get("steps", [])
        if not steps:
            return None
        idx = entry.program_state.get("step_index", 0)
        if idx >= len(steps):
            entry.program_state["step_index"] = 0
            return None
        action = self._exec_node(steps[idx], ctx, entry, matched_signal, signals, mode_activations)
        entry.program_state["step_index"] = idx + 1
        if entry.program_state["step_index"] >= len(steps):
            entry.program_state["step_index"] = 0
        return action

    # -- repeat --

    def _exec_repeat(
        self, node: dict, ctx: ReflexContext,
        entry: AgentReflexEntry,
        matched_signal: Optional[ReflexSignal],
        signals: List[ReflexSignal],
        mode_activations: List[str],
    ) -> Optional[Dict[str, Any]]:
        times = node.get("times", 1)
        count = entry.program_state.get("repeat_count", 0)
        if times != -1 and count >= times:
            entry.program_state["repeat_count"] = 0
            return None
        action = self._exec_node(
            node.get("body", {}), ctx, entry, matched_signal, signals, mode_activations
        )
        entry.program_state["repeat_count"] = count + 1
        return action

    # -- while --

    def _exec_while(
        self, node: dict, ctx: ReflexContext,
        entry: AgentReflexEntry,
        matched_signal: Optional[ReflexSignal],
        signals: List[ReflexSignal],
        mode_activations: List[str],
    ) -> Optional[Dict[str, Any]]:
        from .evaluator import ConditionEvaluator
        evaluator = ConditionEvaluator()
        cond = node.get("condition", {})
        result, _ = evaluator.evaluate(cond, ctx)
        if not result:
            return None
        return self._exec_node(
            node.get("body", {}), ctx, entry, matched_signal, signals, mode_activations
        )

    # -- emit_signal --

    def _exec_emit_signal(
        self, node: dict, ctx: ReflexContext,
        entry: AgentReflexEntry,
        matched_signal: Optional[ReflexSignal],
        signals: List[ReflexSignal],
    ) -> None:
        sig_type = node.get("signal_type", "")
        radius = node.get("radius", 10)
        payload = dict(node.get("payload", {}))
        payload = self._resolve_params(payload, ctx, matched_signal)

        # L4 pair stamping — emitter templates carry their pair_id so that
        # responder consumption can be credited to the pair.
        template = entry.template if entry is not None else None
        pair_id = getattr(template, "pair_id", "") or node.get("pair_id", "")
        emit_id = ReflexSignal.make_emit_id()

        sig = ReflexSignal(
            signal_id=ReflexSignal.make_id(),
            signal_type=sig_type,
            sender_id=ctx.agent_id,
            sender_name="",
            sender_position=(ctx.x, ctx.y),
            turn_sent=ctx.turn,
            radius=radius,
            payload=payload,
            pair_id=pair_id,
            emit_id=emit_id,
        )
        signals.append(sig)

        if pair_id and entry is not None:
            entry.pending_emits[emit_id] = ctx.turn

    # -- activate_mode --

    def _exec_activate_mode(
        self, node: dict, ctx: ReflexContext,
        entry: AgentReflexEntry,
        mode_activations: List[str],
    ) -> None:
        """Register a mode activation request for the agent core to dispatch.

        The template's declared ``mode_id`` wins, but a node-level override
        is allowed so a single template can parameterise which mode to
        activate based on runtime data.
        """
        template = entry.template if entry is not None else None
        mode_id = node.get("mode_id", "") or (getattr(template, "mode_id", "") if template else "")
        if mode_id:
            mode_activations.append(mode_id)

    # -- Variable resolution --

    def _resolve_params(
        self, params: dict, ctx: ReflexContext,
        matched_signal: Optional[ReflexSignal],
    ) -> dict:
        resolved = {}
        for k, v in params.items():
            if isinstance(v, str) and "{" in v:
                resolved[k] = self._resolve_value(v, ctx, matched_signal)
            elif isinstance(v, dict):
                resolved[k] = self._resolve_params(v, ctx, matched_signal)
            elif isinstance(v, list):
                resolved[k] = [
                    self._resolve_value(item, ctx, matched_signal)
                    if isinstance(item, str) and "{" in item else item
                    for item in v
                ]
            else:
                resolved[k] = v
        return resolved

    def _resolve_value(
        self, value: str, ctx: ReflexContext,
        matched_signal: Optional[ReflexSignal],
    ) -> Any:
        if not isinstance(value, str) or "{" not in value:
            return value

        raw = value.strip()
        if raw.startswith("{") and raw.endswith("}"):
            inner = raw[1:-1]
            result = self._lookup_var(inner, ctx, matched_signal)
            if result is not None:
                return result
        return value

    def _lookup_var(
        self, var_path: str, ctx: ReflexContext,
        matched_signal: Optional[ReflexSignal],
    ) -> Any:
        # Agent fields
        if var_path == "agent.id":
            return ctx.agent_id
        if var_path == "agent.x":
            return ctx.x
        if var_path == "agent.y":
            return ctx.y
        if var_path == "agent.energy_pct":
            return ctx.energy_pct
        if var_path == "agent.hp_pct":
            return ctx.hp_pct
        if var_path.startswith("agent.inventory."):
            item = var_path.split(".", 2)[2]
            return ctx.inventory.get(item, 0)
        if var_path == "turn":
            return ctx.turn

        # Nearest entities
        if var_path.startswith("nearest."):
            parts = var_path.split(".")
            if len(parts) >= 3:
                ent_key = parts[1]
                attr = parts[2]
                ent = ctx.nearest_entities.get(ent_key)
                # Fallback aliases: tomato_plant → tomato_plant_ripe (when ripe),
                # cow → cow_alive/cow_dead, etc.
                if ent is None:
                    _FALLBACKS = {
                        "tomato_plant": ["tomato_plant_ripe", "tomato_plant"],
                        "cow":          ["cow_alive", "cow_dead", "cow_harvestable"],
                        "cow_dead":     ["cow_harvestable", "cow_dead"],
                    }
                    for alias in _FALLBACKS.get(ent_key, []):
                        ent = ctx.nearest_entities.get(alias)
                        if ent is not None:
                            break
                if ent is not None:
                    if attr == "id":
                        return ent.entity_id
                    elif attr == "position":
                        return f"{ent.x},{ent.y}"
                    elif attr == "distance":
                        return ent.distance

        # Nearest agent
        if var_path.startswith("nearest_agent."):
            attr = var_path.split(".", 1)[1]
            if ctx.nearby_agents:
                ag = min(ctx.nearby_agents, key=lambda a: a.distance)
                if attr == "id":
                    return ag.agent_id
                elif attr == "position":
                    return f"{ag.x},{ag.y}"
                elif attr == "distance":
                    return ag.distance

        # Most hungry agent
        if var_path.startswith("most_hungry_agent."):
            attr = var_path.split(".", 1)[1]
            hungry = [a for a in ctx.nearby_agents if a.is_hungry]
            if hungry:
                ag = min(hungry, key=lambda a: a.energy_pct)
                if attr == "id":
                    return ag.agent_id
                elif attr == "position":
                    return f"{ag.x},{ag.y}"

        # Nearest empty tile
        if var_path.startswith("nearest_empty_tile"):
            ent = ctx.nearest_entities.get("empty_tile")
            if ent is not None:
                return f"{ent.x},{ent.y}"

        # Signals
        if var_path.startswith("signals."):
            attr = var_path.split(".", 1)[1]
            return getattr(ctx, attr, None)

        # Safest direction
        if var_path == "safest_direction":
            return ctx.observation_metadata.get("_safest_direction", f"{ctx.x},{ctx.y}")

        # Observation-metadata pass-through. Env adapters can stash arbitrary
        # per-turn fields here (e.g. best_food_item, nearest_lake_x). Templates
        # reference them as bare `{key}` — keep this generic so new env-side
        # helpers don't require executor changes.
        if var_path in ctx.observation_metadata:
            return ctx.observation_metadata.get(var_path)

        # Received signal fields
        if var_path.startswith("received_signal.") and matched_signal is not None:
            rest = var_path[len("received_signal."):]
            if rest == "sender_id":
                return matched_signal.sender_id
            elif rest == "sender_name":
                return matched_signal.sender_name
            elif rest == "sender_position":
                return f"{matched_signal.sender_position[0]},{matched_signal.sender_position[1]}"
            elif rest == "signal_type":
                return matched_signal.signal_type
            elif rest.startswith("payload."):
                key = rest[len("payload."):]
                return matched_signal.payload.get(key)

        # Safest direction from signal — computed inline from matched_signal
        if var_path == "safest_direction_from_signal":
            if matched_signal is not None:
                danger_pos = matched_signal.payload.get("danger_position")
                if danger_pos is not None:
                    dpx, dpy = self._parse_position(danger_pos)
                    if dpx is not None and dpy is not None:
                        dx = ctx.x - dpx
                        dy = ctx.y - dpy
                        if dx == 0 and dy == 0:
                            dx, dy = 1, 0
                        norm = max(abs(dx), abs(dy), 1)
                        tx = ctx.x + int(dx / norm * 5)
                        ty = ctx.y + int(dy / norm * 5)
                        return f"{max(0, tx)},{max(0, ty)}"
            return ctx.observation_metadata.get(
                "_safest_direction_from_signal",
                ctx.observation_metadata.get(
                    "_safest_direction",
                    f"{ctx.x},{ctx.y}",
                ),
            )

        # Surplus items for storage
        if var_path == "surplus_items":
            surplus = {}
            thresholds = {"wood": 10, "stone": 10, "raw_beef": 5, "tomato": 5}
            for item, threshold in thresholds.items():
                count = ctx.inventory.get(item, 0)
                if count > threshold:
                    surplus[item] = count - threshold
            return surplus if surplus else {"wood": 0}

        return None

    def _parse_position(self, value: Any) -> Tuple[Optional[int], Optional[int]]:
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            try:
                return int(value[0]), int(value[1])
            except (ValueError, TypeError):
                return None, None
        if isinstance(value, str) and "," in value:
            parts = value.split(",")
            try:
                return int(parts[0].strip()), int(parts[1].strip())
            except (ValueError, TypeError):
                return None, None
        return None, None
