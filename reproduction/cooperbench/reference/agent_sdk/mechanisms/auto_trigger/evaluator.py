"""
auto_trigger/evaluator.py — ConditionEvaluator.

Evaluates JSON-based TriggerCondition DSL against a ReflexContext snapshot.
Safe, sandboxed — no Python code execution, only structured dict lookups.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set

from .types import ReflexContext, ReflexSignal

logger = logging.getLogger(__name__)

# Dedup key cache for unknown-type / malformed condition warnings. The
# evaluator runs every reflex every turn, so a single bad template would
# spam the log file 10000+ times in a 240-turn run. We log each unique
# (id(condition_dict), ctype) tuple only once.
_LOGGED_UNKNOWN_KEYS: Set[tuple] = set()

_SIGNAL_ORDER = {"none": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
_OPS = {
    "lt": lambda a, b: a < b,
    "lte": lambda a, b: a <= b,
    "gt": lambda a, b: a > b,
    "gte": lambda a, b: a >= b,
    "eq": lambda a, b: a == b,
    "ne": lambda a, b: a != b,
}


class ConditionEvaluator:
    """
    Evaluate a JSON condition tree against a ReflexContext.

    Returns (matched: bool, matched_signal: Optional[ReflexSignal]).
    The matched_signal is set when a signal_received condition matches,
    so ActionProgramExecutor can reference {received_signal.*} variables.
    """

    def evaluate(
        self, condition: dict, ctx: ReflexContext
    ) -> tuple:
        """Returns (bool, Optional[ReflexSignal])."""
        if not condition:
            return False, None
        ctype = condition.get("type", "")
        try:
            return self._dispatch(ctype, condition, ctx)
        except Exception as e:
            logger.warning("[ConditionEvaluator] Error evaluating %s: %s", ctype, e)
            return False, None

    def _dispatch(
        self, ctype: str, cond: dict, ctx: ReflexContext
    ) -> tuple:
        if ctype == "and":
            return self._eval_and(cond, ctx)
        elif ctype == "or":
            return self._eval_or(cond, ctx)
        elif ctype == "not":
            return self._eval_not(cond, ctx)
        elif ctype == "threshold":
            return self._eval_threshold(cond, ctx), None
        elif ctype == "signal":
            return self._eval_signal(cond, ctx), None
        elif ctype == "inventory":
            return self._eval_inventory(cond, ctx), None
        elif ctype == "cooldown":
            return self._eval_cooldown(cond, ctx), None
        elif ctype == "entity_nearby":
            return self._eval_entity_nearby(cond, ctx), None
        elif ctype == "agent_nearby":
            return self._eval_agent_nearby(cond, ctx), None
        elif ctype == "no_threat":
            return self._eval_no_threat(cond, ctx), None
        elif ctype == "turn_range":
            return self._eval_turn_range(cond, ctx), None
        elif ctype == "bool_flag":
            return self._eval_bool_flag(cond, ctx), None
        elif ctype == "action_available":
            return self._eval_action_available(cond, ctx), None
        elif ctype == "signal_received":
            return self._eval_signal_received(cond, ctx)
        elif ctype == "entity_visible":
            return self._eval_entity_visible(cond, ctx), None
        elif ctype == "mode_active":
            return self._eval_mode_active(cond, ctx), None
        elif ctype == "mode_inactive":
            return (not self._eval_mode_active(cond, ctx)), None
        elif ctype == "group_pressure":
            return self._eval_group_pressure(cond, ctx), None
        elif ctype == "trend":
            return self._eval_trend(cond, ctx), None
        else:
            # 2026-06-02 audit fix: unknown type used to log a warning
            # every single tick. Dedupe so a malformed legacy template
            # only logs once per process. Publish-time validation
            # (auto_trigger/validation.py) now blocks this from landing
            # in the store going forward, so this branch is mostly a
            # safety net for old shards / direct-injected fixtures.
            key = (id(cond), str(ctype))
            if key not in _LOGGED_UNKNOWN_KEYS:
                _LOGGED_UNKNOWN_KEYS.add(key)
                logger.warning(
                    "[ConditionEvaluator] Unknown condition type %r — "
                    "this template will never fire. Use publish-time "
                    "schema validation (auto_trigger/validation.py) to "
                    "catch this before it lands in the store.", ctype
                )
            return False, None

    # -- Composite --

    def _eval_and(self, cond: dict, ctx: ReflexContext) -> tuple:
        matched_signal = None
        for sub in cond.get("conditions", []):
            result, sig = self.evaluate(sub, ctx)
            if not result:
                return False, None
            if sig is not None:
                matched_signal = sig
        return True, matched_signal

    def _eval_or(self, cond: dict, ctx: ReflexContext) -> tuple:
        for sub in cond.get("conditions", []):
            result, sig = self.evaluate(sub, ctx)
            if result:
                return True, sig
        return False, None

    def _eval_not(self, cond: dict, ctx: ReflexContext) -> tuple:
        inner = cond.get("condition", {})
        result, sig = self.evaluate(inner, ctx)
        return not result, None

    # -- Atomic --

    def _eval_threshold(self, cond: dict, ctx: ReflexContext) -> bool:
        source = cond.get("source", "derived")
        field_name = cond.get("field", "")
        op = cond.get("op", "lt")
        value = cond.get("value", 0)

        actual = self._resolve_field(field_name, ctx)
        if actual is None:
            return False
        return _OPS.get(op, lambda a, b: False)(float(actual), float(value))

    def _eval_signal(self, cond: dict, ctx: ReflexContext) -> bool:
        field_name = cond.get("field", "")
        op = cond.get("op", "gte")
        value = str(cond.get("value", "none")).lower()

        actual = getattr(ctx, field_name, "none")
        actual_ord = _SIGNAL_ORDER.get(str(actual).lower(), 0)
        value_ord = _SIGNAL_ORDER.get(value, 0)
        return _OPS.get(op, lambda a, b: False)(actual_ord, value_ord)

    def _eval_inventory(self, cond: dict, ctx: ReflexContext) -> bool:
        item = cond.get("item", "")
        op = cond.get("op", "gt")
        value = cond.get("value", 0)

        # Handle food (aggregate) item
        if item == "food":
            count = sum(
                v for k, v in ctx.inventory.items()
                if k in (
                    "food", "cooked_beef", "cooked_tomatoe",
                    "raw_beef", "tomatoe", "tomato",
                )
            )
        else:
            count = ctx.inventory.get(item, 0)
        return _OPS.get(op, lambda a, b: False)(count, value)

    def _eval_cooldown(self, cond: dict, ctx: ReflexContext) -> bool:
        min_turns = cond.get("min_turns", 1)
        template_id = cond.get("_template_id", "")
        last_fired = ctx.last_reflex_fired_turn.get(template_id, -1)
        result = True if last_fired < 0 else (ctx.turn - last_fired) >= min_turns
        if not result and ctx.agent_id in ("agent_1_0", "agent_1_6"):
            print(f"[CD] {ctx.agent_id} T{ctx.turn}: tid={template_id}, last_fired={last_fired}, min={min_turns}, delta={ctx.turn - last_fired}")
        return result

    def _eval_entity_nearby(self, cond: dict, ctx: ReflexContext) -> bool:
        entity_type = cond.get("entity_type", "")
        max_distance = cond.get("max_distance", 10)
        count_op = cond.get("count_op", "gte")
        count_val = cond.get("count", 1)
        entity_state = cond.get("entity_state", "")

        # Build lookup key from entity_type + state
        key = entity_type
        if entity_state:
            key = f"{entity_type}_{entity_state}"

        ent = ctx.nearest_entities.get(key)
        if ent is None:
            ent = ctx.nearest_entities.get(entity_type)
        # Fallback aliases for entities stored under variant keys
        if ent is None:
            _FALLBACKS = {
                "tomato_plant": ["tomato_plant_ripe", "tomato_plant"],
                "cow":          ["cow_alive", "cow_dead", "cow_harvestable"],
                "cow_dead":     ["cow_harvestable", "cow_dead"],
            }
            for alias in _FALLBACKS.get(entity_type, []):
                ent = ctx.nearest_entities.get(alias)
                if ent is not None:
                    break
        if ent is None:
            actual_count = 0
        elif ent.distance <= max_distance:
            actual_count = 1
        else:
            actual_count = 0

        return _OPS.get(count_op, lambda a, b: False)(actual_count, count_val)

    def _eval_agent_nearby(self, cond: dict, ctx: ReflexContext) -> bool:
        state = cond.get("state", "any")
        max_distance = cond.get("max_distance", 10)
        count_op = cond.get("count_op", "gte")
        count_val = cond.get("count", 1)

        actual = 0
        for a in ctx.nearby_agents:
            if a.distance > max_distance:
                continue
            if state == "any" or state == "alive":
                actual += 1
            elif state == "hungry" and a.is_hungry:
                actual += 1
        return _OPS.get(count_op, lambda a, b: False)(actual, count_val)

    def _eval_no_threat(self, cond: dict, ctx: ReflexContext) -> bool:
        max_distance = cond.get("max_distance", 5)
        # No wolf/threat entities within distance
        for key, ent in ctx.nearest_entities.items():
            if "wolf" in key or "threat" in key:
                if ent.distance <= max_distance:
                    return False
        if _SIGNAL_ORDER.get(ctx.danger_level, 0) >= _SIGNAL_ORDER.get("high", 3):
            return False
        return True

    def _eval_turn_range(self, cond: dict, ctx: ReflexContext) -> bool:
        min_turn = cond.get("min", 0)
        max_turn = cond.get("max", 999999)
        return min_turn <= ctx.turn <= max_turn

    def _eval_bool_flag(self, cond: dict, ctx: ReflexContext) -> bool:
        key = cond.get("key", "")
        expected = cond.get("expected", True)
        actual = ctx.observation_metadata.get(key, False)
        return bool(actual) == expected

    def _eval_action_available(self, cond: dict, ctx: ReflexContext) -> bool:
        action_type = cond.get("action_type", "")
        return action_type in ctx.available_action_types

    def _eval_signal_received(
        self, cond: dict, ctx: ReflexContext
    ) -> tuple:
        signal_type = cond.get("signal_type", "")
        from_filter = cond.get("from", "any")
        max_age = cond.get("max_age_turns", 2)
        max_distance = cond.get("max_distance", 9999)
        consume = cond.get("consume", True)

        for sig in ctx.received_signals:
            if sig.signal_type != signal_type:
                continue
            if from_filter != "any" and sig.sender_id != from_filter:
                continue
            if (ctx.turn - sig.turn_sent) > max_age:
                continue
            # Distance check (chebyshev)
            dx = abs(ctx.x - sig.sender_position[0])
            dy = abs(ctx.y - sig.sender_position[1])
            if max(dx, dy) > max_distance:
                continue
            return True, sig

        return False, None

    def _eval_entity_visible(self, cond: dict, ctx: ReflexContext) -> bool:
        keywords = cond.get("keywords", [])
        view_text = ctx.observation_metadata.get("_view_text", "").lower()
        return any(kw.lower() in view_text for kw in keywords)

    # -- Level 5 / Level 4 extensions --

    def _eval_mode_active(self, cond: dict, ctx: ReflexContext) -> bool:
        """True when ``cond['mode_id']`` (or any of ``cond['mode_ids']``) is active.

        Composite semantics: if both are provided, either match is sufficient.
        Combine with an explicit 'not' wrapper to negate.
        """
        single = cond.get("mode_id", "")
        multi = cond.get("mode_ids", []) or []
        candidates = []
        if single:
            candidates.append(single)
        candidates.extend(multi)
        if not candidates:
            return False
        active = ctx.active_modes or set()
        return any(m in active for m in candidates)

    def _eval_group_pressure(self, cond: dict, ctx: ReflexContext) -> bool:
        """Check a group-level metric against a threshold.

        Fields live in ``ReflexContext.group_metrics``, populated by the
        env adapter (e.g. ``food_stock_low``, ``group_hp_trend``).
        """
        field = cond.get("field", "")
        op = cond.get("op", "gte")
        value = float(cond.get("value", 0.0) or 0.0)
        actual = float(ctx.group_metrics.get(field, 0.0) or 0.0)
        return _OPS.get(op, lambda a, b: False)(actual, value)

    def _eval_trend(self, cond: dict, ctx: ReflexContext) -> bool:
        """Check a trend scalar (positive=rising, negative=falling).

        Fields live in ``ReflexContext.trends``. Typical usage:
            {"type": "trend", "field": "hp_trend", "op": "lt", "value": 0.0}
        """
        field = cond.get("field", "")
        op = cond.get("op", "lt")
        value = float(cond.get("value", 0.0) or 0.0)
        actual = float(ctx.trends.get(field, 0.0) or 0.0)
        return _OPS.get(op, lambda a, b: False)(actual, value)

    # -- Field resolution --

    def _resolve_field(self, field_name: str, ctx: ReflexContext) -> Optional[float]:
        if field_name == "energy_pct":
            return ctx.energy_pct
        elif field_name == "hp_pct":
            return ctx.hp_pct
        elif field_name == "energy":
            return ctx.energy
        elif field_name == "hp":
            return ctx.hp
        elif field_name == "age":
            return float(ctx.age)
        elif field_name == "turn":
            return float(ctx.turn)
        elif field_name.startswith("inventory."):
            item = field_name.split(".", 1)[1]
            return float(ctx.inventory.get(item, 0))
        # Fall back to observation_metadata for env-specific derived fields
        val = ctx.observation_metadata.get(field_name)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                return None
        return None
