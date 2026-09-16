"""Stage 4 — tier-routed LLM decision provider for external users.

When `ORG_EXTERNAL_LLM` is on, an external user who ENCOUNTERS an event calls the LLM once to decide
whether/what to post or reply; after a release, each user calls the LLM to evaluate the product →
purchase intent. The LLM is gated (only event-triggered / activated users post), uses fixed model
snapshots by profile tier, temperature 0 + a prompt cache for reproducibility/cost, and ALWAYS falls
back to the deterministic HeuristicDecisionProvider on any error / when no API key is configured
(so tests + offline runs stay free and reproducible).

Mirrors society_core's provider pattern (Heuristic surrogate ↔ live OpenAI) + JSON-record output.
"""
from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from random import Random
from typing import List, Optional

from environments.org_env.external_society.agent import (
    ExternalUser,
    PostDecision,
    ProductIntent,
    ProductOffering,
)
from environments.org_env.external_society.decision import HeuristicDecisionProvider
from environments.org_env.external_society.event_pool import EventRecord


MODEL_BY_TIER = {
    "low": "gpt-5-nano-2025-08-07",
    "standard": "gpt-5-mini-2025-08-07",
    "high": "gpt-5.5-2026-04-23",
}


def _truthy(v: str) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def external_llm_enabled() -> bool:
    return _truthy(os.environ.get("ORG_EXTERNAL_LLM", ""))


def _llm_config() -> Optional[dict]:
    """Read OpenAI credentials, preferring the process environment."""
    env_key = os.environ.get("OPENAI_API_KEY")
    if env_key:
        return {
            "key": env_key,
            "base": os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
        }
    try:
        import yaml
    except Exception:
        return None
    for path in ("config/llm.local.yaml", "config/llm.yaml"):
        try:
            with open(path, encoding="utf-8") as fh:
                cfg = yaml.safe_load(fh) or {}
        except Exception:
            continue
        org = cfg.get("org_env", {}) or {}
        key = org.get("api_key")
        base = org.get("base_url") or "https://api.openai.com/v1"
        if not key:
            ag = ((cfg.get("agent", {}) or {}).get("openai", {}) or {})
            keys = ag.get("api_keys") or []
            key = keys[0] if keys else None
            base = ag.get("base_url") or base
        if key:
            return {"key": str(key), "base": str(base).rstrip("/")}
    return None


def _extract_json(raw: str) -> Optional[dict]:
    s = (raw or "").strip()
    if not s:
        return None
    start, end = s.find("{"), s.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        v = json.loads(s[start:end + 1])
        return v if isinstance(v, dict) else None
    except Exception:
        return None


def _num(v, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(v)))
    except (TypeError, ValueError):
        return default


def _experience_cue(product: ProductOffering) -> str:
    # #5: product-neutral experience language (not research-agent "sourced output"), so an OSS tool
    # (e.g. a repo-ingestion CLI) reads as a developer-tool trial, not a report-writing agent.
    q = product.quality * (0.55 if product.fallback_grounded else 1.0)
    if q >= 0.7:
        return "It ran cleanly end-to-end and produced correct, usable output."
    if q >= 0.45:
        return "It mostly ran, but some cases were broken or the output was off."
    return "It was unreliable: it didn't run cleanly / produced wrong output."


def _profile_facets(user: ExternalUser) -> str:
    profile = user.profile
    return json.dumps(
        {
            "technical_role": profile.technical_role,
            "innovation_role": profile.innovation_role,
            "technical_skill": profile.technical_skill,
            "creative_capacity": profile.creative_capacity,
            "activity_tier": profile.activity_tier,
            "intelligence_tier": profile.intelligence_tier,
            "llm_model_tier": profile.llm_model_tier,
        },
        sort_keys=True,
    )


def _model_for_user(user: ExternalUser) -> str:
    return MODEL_BY_TIER.get(user.profile.llm_model_tier, MODEL_BY_TIER["standard"])


class LLMDecisionProvider:
    source = "llm:tiered"

    def __init__(self, model: str = MODEL_BY_TIER["standard"], timeout: int = 30):
        self.fallback = HeuristicDecisionProvider()
        self.cfg = _llm_config()
        self.model = model
        self.timeout = timeout
        self._cache: dict = {}
        self.calls = 0
        self.failures = 0

    def _call(
        self, system: str, prompt: str, max_tokens: int = 220, *, model: Optional[str] = None,
    ) -> str:
        if not self.cfg:
            return ""
        request_model = model or self.model
        ck = hashlib.sha1(
            f"{request_model}\x00{system}\x00{prompt}".encode("utf-8")
        ).hexdigest()
        if ck in self._cache:
            return self._cache[ck]
        body = json.dumps({
            "model": request_model, "temperature": 0, "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        }).encode("utf-8")
        req = urllib.request.Request(
            self.cfg["base"] + "/chat/completions", data=body,
            headers={"Authorization": "Bearer " + self.cfg["key"], "Content-Type": "application/json"})
        self.calls += 1
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            out = (data["choices"][0]["message"]["content"] or "").strip()
        except Exception:
            self.failures += 1
            out = ""
        self._cache[ck] = out
        return out

    # -- posting on an encountered event ---------------------------------------------------------
    def decide_post(self, user: ExternalUser, event: EventRecord, recent_posts: List, rng: Random,
                    reply_fraction: float = 0.25) -> PostDecision:
        ctx = "\n".join(f"[{i}] ({getattr(p,'topic','')}) {getattr(p,'text','')[:120]}"
                        for i, p in enumerate(recent_posts[-8:])) or "(quiet feed)"
        system = ("You simulate ONE professional on a LinkedIn-like forum. Decide if this person "
                  "would post about what they just encountered. Return ONLY JSON: "
                  '{"will_post": bool, "text": str (<=240 chars, first person), '
                  '"reply_to_index": int (-1 for a new post, else the [index] you reply to), '
                  '"topic": str (snake_case)}.')
        prompt = (f"PERSON: {user.profile.persona_text}\n"
                  f"STRUCTURED FACETS: {_profile_facets(user)}\n"
                  f"JUST ENCOUNTERED ({event.kind}): {event.text}\n"
                  f"RECENT FEED:\n{ctx}\n"
                  f"Would this person post or reply? Stay in character (their occupation/needs).")
        data = _extract_json(self._call(system, prompt, model=_model_for_user(user)))
        if data is None:
            return self.fallback.decide_post(user, event, recent_posts, rng, reply_fraction)
        if not bool(data.get("will_post")):
            return PostDecision(will_post=False)
        reply_to = None
        idx = data.get("reply_to_index", -1)
        recent8 = recent_posts[-8:]
        try:
            if isinstance(idx, (int, float)) and 0 <= int(idx) < len(recent8):
                reply_to = recent8[int(idx)].post_id
        except Exception:
            reply_to = None
        text = str(data.get("text") or "").strip()[:240] or f"On {event.topic or 'this'}: {event.text}"
        return PostDecision(will_post=True, text=text,
                            topic=str(data.get("topic") or event.topic or ""), reply_to=reply_to)

    # -- evaluating the product (purchase intent) -----------------------------------------------
    def evaluate_product(self, user: ExternalUser, product: ProductOffering,
                         recent_product_posts: List, rng: Random) -> ProductIntent:
        mem = "; ".join(user.memory[-6:]) or "(none)"
        peer = "\n".join(f"- {getattr(p,'text','')[:120]}" for p in recent_product_posts[-6:]) or "(no buzz yet)"
        system = ("You simulate ONE professional trying a product, then reporting an HONEST reaction. "
                  "You cannot decide final purchase; just report intent. Return ONLY JSON with keys: "
                  "intent_to_try, intent_to_pay, willingness_to_pay, intent_to_recommend, "
                  "subjective_satisfaction, confidence (all 0..1), feedback (<=200 chars), "
                  "reason_code (snake_case).")
        prompt = (f"PERSON: {user.profile.persona_text}\n"
                  f"STRUCTURED FACETS: {_profile_facets(user)}\n"
                  f"THEIR RECENT MEMORY: {mem}\n"
                  f"PRODUCT: {product.summary or product.product_id} "
                  f"(serves: {', '.join(product.topics) or 'general research'})\n"
                  f"YOUR HANDS-ON EXPERIENCE: {_experience_cue(product)}\n"
                  f"WHAT OTHERS SAY:\n{peer}\n"
                  f"Report your honest intent as this person.")
        model = _model_for_user(user)
        data = _extract_json(self._call(system, prompt, max_tokens=240, model=model))
        if data is None:
            return self.fallback.evaluate_product(user, product, recent_product_posts, rng)
        sat = _num(data.get("subjective_satisfaction"), 0.3)
        wtp = _num(data.get("willingness_to_pay"), 0.0)
        itp = _num(data.get("intent_to_pay"), 0.0)
        return ProductIntent(
            user_id=user.user_id, product_id=product.product_id, day=0,
            satisfaction=round(sat, 3), intent_to_try=round(_num(data.get("intent_to_try"), 0.0), 3),
            intent_to_pay=round(itp, 3), willingness_to_pay=round(wtp, 3),
            intent_to_recommend=round(_num(data.get("intent_to_recommend"), 0.0), 3),
            confidence=round(_num(data.get("confidence"), 0.5), 3),
            converted=(itp >= 0.5 and wtp >= product.price),
            feedback=str(data.get("feedback") or "").strip()[:200] or "(no comment)",
            reason_code=str(data.get("reason_code") or "llm_judgment")[:48],
            source=f"llm:{model}", profile_id=user.profile.user_id,
            technical_role=user.profile.technical_role,
            innovation_role=user.profile.innovation_role)


def make_provider():
    """Provider factory: live LLM when ORG_EXTERNAL_LLM is on AND a key is configured, else the
    deterministic offline heuristic."""
    if external_llm_enabled():
        prov = LLMDecisionProvider()
        if prov.cfg:
            return prov
    return HeuristicDecisionProvider()


__all__ = [
    "LLMDecisionProvider", "MODEL_BY_TIER", "make_provider", "external_llm_enabled",
]
