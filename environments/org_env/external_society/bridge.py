"""Stage 5 — bridge the LLM external society into OrgWorld (opt-in, gated by ORG_EXTERNAL_SOCIETY).

When enabled, the rich external society IS the company's external world:
  * once per day the society advances (event-triggered posts) and its forum posts are
    mirrored into `world.community.posts`, so the company's existing read_feed / perception surface
    them (external sentiment -> internal perception);
  * on a published release, every external user experiences the product (quality from the company's
    real readiness) and their purchase intent is written as `CustomerTrial`s, so the EXISTING
    `customers` funding milestone (market_summary) is driven by real profiled users instead of the
    P5 synthetic personas.

Default (flag off) leaves OrgEnv unchanged. The whole society is seeded for reproducibility.
"""
from __future__ import annotations

import os
from typing import List, Optional

from relic.research.hashing import stable_hash

TICKS_PER_DAY = 24


class ExternalMarketInfrastructureError(RuntimeError):
    """The release could not be evaluated; no product judgment was produced."""


def _truthy(v: str) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def external_society_enabled() -> bool:
    return _truthy(os.environ.get("ORG_EXTERNAL_SOCIETY", ""))


def attach_external_society(world, *, n: Optional[int] = None, seed: Optional[int] = None):
    """Create + attach an ExternalSociety to the world (idempotent)."""
    if getattr(world, "external_society", None) is not None:
        return world.external_society
    from environments.org_env.external_society.society import ExternalSociety
    n = int(n if n is not None else os.environ.get("ORG_EXTERNAL_N", "50"))
    seed = int(seed if seed is not None else getattr(getattr(world, "scenario", None), "seed", 42))
    soc = ExternalSociety(n=n, seed=seed)
    world.__dict__["external_society"] = soc
    world.__dict__["_external_day"] = -1
    return soc


def _sync_forum(world, soc, posts: List, tick: int) -> None:
    """Mirror company-readable society boards into the OrgEnv community feed."""
    from environments.org_env.backend.community import ExternalProfile, Post
    comm = getattr(world, "community", None)
    if comm is None:
        return
    for p in posts:
        if p.board not in {"general", "product"} or p.post_id in comm.posts:
            continue
        if p.author_id not in comm.profiles:
            u = soc.users_by_id.get(p.author_id)
            occ = u.profile.occupation if u else "community member"
            role = occ
            if u is not None:
                role = (
                    f"{occ} | technical_role={u.profile.technical_role} | "
                    f"innovation_role={u.profile.innovation_role}"
                )
            comm.profiles[p.author_id] = ExternalProfile(
                external_agent_id=p.author_id, name=(u.profile.name if u else p.author_id),
                role=role, topic_interests=list(u.profile.interests) if u else [], credibility=0.65)
        comm.posts[p.post_id] = Post(
            post_id=p.post_id, author_id=p.author_id, topic=p.topic,
            content_summary=p.text, credibility=0.65, reach=150, created_tick=tick)


def maybe_run_external_society(world, tick: int) -> None:
    """Called each OrgEnv tick. If the society is enabled, attach it once and advance it one day per
    day (mirroring its fresh posts into the company feed). No-op when the flag is off."""
    if getattr(world, "external_society", None) is None:
        if not external_society_enabled():
            return
        attach_external_society(world)
    soc = world.external_society
    day = int(tick) // TICKS_PER_DAY
    if day == world.__dict__.get("_external_day"):
        return
    world.__dict__["_external_day"] = day
    new = soc.daily_tick(day)
    _sync_forum(world, soc, new, tick)


def product_offering_from_world(world):
    """Build the market-facing ProductOffering from the company's REAL product state."""
    from environments.org_env.external_society.agent import ProductOffering
    try:
        from environments.org_env.backend.market.validation import product_quality
        pq = product_quality(world)
        quality = float(pq["quality"])
        cli = bool(pq["cli_runnable"])
        runtime = bool(pq["runtime_ready"])
        top_gap = str(pq.get("top_gap", ""))
    except Exception as exc:
        raise ExternalMarketInfrastructureError(
            f"product_quality_unavailable:{type(exc).__name__}"
        ) from None
    if not 0.0 <= quality <= 1.0:
        raise ExternalMarketInfrastructureError("product_quality_out_of_range")
    quality_source = str(
        pq.get("quality_basis")
        or "org_env.market.validation.product_quality"
    )
    quality_evidence_ref = f"product_quality:{stable_hash(pq)[:24]}"
    rs = getattr(world, "repo_system", None)
    rels = getattr(getattr(rs, "repo", None), "releases", {}) or {} if rs else {}
    rel = next(reversed(list(rels.values())), None) if rels else None
    version = getattr(rel, "version", "") if rel else ""
    prod = getattr(world, "product", None)
    name = getattr(prod, "name", None) or "the product"
    summary = getattr(prod, "summary", "") or "an early-stage developer tool"
    # #5: market topics follow the actual product substrate — an OSS tool (e.g. gitingest, a repo->text
    # ingestion CLI) is evaluated according to its actual developer-tool surface.
    topics = ("eval_infra", "agent_reliability")
    try:
        from environments.org_env.product.substrates.eval_assets import is_oss_substrate
        if is_oss_substrate(world):
            topics = ("developer_tools", "repo_ingestion")
    except Exception:
        pass
    return ProductOffering(
        product_id=str(name).lower().replace(" ", "_"),
        summary=(summary + (f" — gap: {top_gap}" if top_gap else ""))[:200],
        quality=float(quality), topics=topics,
        price=0.3, fallback_grounded=bool(cli and not runtime),
        quality_source=quality_source,
        quality_evidence_ref=quality_evidence_ref), version


def drive_post_release_market(world, tick: int) -> List[str]:
    """Drive the external market on ANY published release — the agent action
    `publish_product_release` OR the internal auto-publish cadence (world.py). If the rich
    LLM society is attached, all profiled users experience the release (purchase intent ->
    CustomerTrials); otherwise the P5 synthetic-persona market runs. De-duped per release
    VERSION so repeated internal publishes of the same version don't re-run the market (and
    don't re-bill the LLM). Returns the new trial ids.

    NOTE: this MUST be reachable from both publish paths — a 14-day run typically ships via
    the internal cadence (`published_internal`), and the market was previously wired only to
    the rarely-chosen agent action, so trials stayed 0 / customers milestone unreachable.
    """
    from environments.org_env.experiments.ablations import EXTERNAL_BRIDGE, mechanism_disabled
    if mechanism_disabled(world, EXTERNAL_BRIDGE):
        return []
    rs = getattr(world, "repo_system", None)
    rels = getattr(getattr(rs, "repo", None), "releases", {}) or {} if rs else {}
    rel = next(reversed(list(rels.values())), None) if rels else None
    version = getattr(rel, "version", "") if rel else ""
    seen = world.__dict__.setdefault("_market_experienced_versions", set())
    if version and version in seen:
        return []
    before = set((getattr(world, "trials", {}) or {}).keys())
    try:
        if getattr(world, "external_society", None) is not None:
            run_external_product_experience(world, tick)
        else:
            from environments.org_env.backend.market import run_market_trials
            run_market_trials(world, tick, n=3, trigger="release")
    except Exception as exc:
        _record_external_market_failure(world, tick, version, exc)
        return []
    if version:
        seen.add(version)
    after = getattr(world, "trials", {}) or {}
    new_ids = [tid for tid in after if tid not in before]
    _post_oss_release_reactions(world, tick, version, new_ids)
    return new_ids


def _record_external_market_failure(world, tick: int, version: str, exc: Exception) -> None:
    events = getattr(world, "events", None)
    if events is None:
        events = []
        world.events = events
    events.append(
        {
            "type": "external_signal_event",
            "subtype": "external_market_infrastructure_failure",
            "tick": int(tick),
            "release_version": version,
            "error_type": type(exc).__name__,
            "retryable": True,
        }
    )


def _post_oss_release_reactions(world, tick: int, version: str, trial_ids: List[str]) -> None:
    """OSS time-machine: mirror the release EXPERIENCE back into the external community as a forum
    reaction whose sentiment tracks real satisfaction (which now follows real behavior quality) — so
    the org's feed shows users reacting to whether the shipped product actually works, not only the
    pre-release issue complaints. De-duped per release version; no-op for non-OSS worlds."""
    try:
        from environments.org_env.product.substrates.eval_assets import is_oss_substrate
    except Exception:
        return
    community = getattr(world, "community", None)
    if not is_oss_substrate(world) or community is None or not hasattr(community, "add_post"):
        return
    seen = world.__dict__.setdefault("_oss_reaction_versions", set())
    if version and version in seen:
        return
    if version:
        seen.add(version)
    trials = getattr(world, "trials", {}) or {}
    sats = [float(getattr(trials.get(t), "satisfaction", 0.0) or 0.0) for t in trial_ids if trials.get(t)]
    if not sats:
        return
    avg = sum(sats) / len(sats)
    conv = sum(1 for t in trial_ids if getattr(trials.get(t), "converted", False))
    if avg >= 0.6:
        text = f"Tried the v{version} release — works well for my repos now, would pay for it."
    elif avg >= 0.4:
        text = f"v{version} is promising but still hit rough edges before I'd rely on it."
    else:
        text = f"v{version} still didn't work for my workflow — churned."
    from environments.org_env.backend.community.objects import Post
    community.add_post(Post(
        post_id=f"post_oss_release_{version}_{tick}", author_id="ext_user_market", topic="product",
        content_summary=text, stance=round((avg - 0.5) * 2, 2), credibility=0.6,
        reach=30 + conv * 5, created_tick=int(tick), visibility="public"))
    if getattr(world, "events", None) is None:
        world.events = []
    world.events.append({"type": "external_signal_event", "subtype": "oss_release_reaction",
                         "version": version, "avg_satisfaction": round(avg, 3),
                         "conversions": conv, "tick": int(tick)})


def run_external_product_experience(world, tick: int) -> Optional[dict]:
    """On a published release: every external user experiences the product; write their purchase
    intent as CustomerTrials (drives the existing `customers` milestone) + churn tickets."""
    soc = getattr(world, "external_society", None)
    if soc is None:
        return None
    from environments.org_env.backend.entities.economy import CustomerTicket, CustomerTrial
    offering, version = product_offering_from_world(world)
    summary = soc.run_product_experience(offering, day=int(tick) // TICKS_PER_DAY)
    product_posts = [
        post
        for post in soc.posts.values()
        if post.board == "product" and post.about_product == offering.product_id
    ]
    _sync_forum(world, soc, product_posts, tick)
    trials = world.__dict__.setdefault("trials", {})
    tickets = world.__dict__.setdefault("tickets", {})
    for u in soc.users:
        it = u.product_intent
        if it is None:
            continue
        tid = f"trial_ext_{u.user_id}_{tick}"
        outcome = "converted" if it.converted else ("interested" if it.intent_to_pay >= 0.4 else "rejected")
        persona = (
            f"external_user_id={u.user_id}; technical_role={u.profile.technical_role}; "
            f"innovation_role={u.profile.innovation_role}; {u.profile.persona_text}"
        )[:240]
        trials[tid] = CustomerTrial(
            trial_id=tid, customer_type=u.profile.occupation, persona=persona,
            query=u.profile.needs, release_version=version, outcome=outcome,
            satisfaction=it.satisfaction, willingness_to_pay=it.willingness_to_pay,
            converted=it.converted, feedback=it.feedback, created_tick=tick,
            decision_source=it.source, profile_id=it.profile_id,
            product_evidence_source=(
                f"{offering.quality_source}:{offering.quality_evidence_ref}"
            ))
        if outcome == "rejected":
            ktid = f"ticket_ext_{u.user_id}_{tick}"
            tickets[ktid] = CustomerTicket(
                ticket_id=ktid, customer_type=u.profile.occupation, complaint_or_request=it.feedback,
                severity="major", topic="market_validation", status="open", response_status="pending")
    world.__dict__["_external_market"] = summary
    world.events.append({"type": "external_signal_event", "subtype": "external_society_experience",
                         "tick": tick, "release_version": version, **summary})
    return summary


__all__ = ["ExternalMarketInfrastructureError", "external_society_enabled", "attach_external_society", "maybe_run_external_society",
           "run_external_product_experience", "drive_post_release_market",
           "product_offering_from_world", "TICKS_PER_DAY"]
