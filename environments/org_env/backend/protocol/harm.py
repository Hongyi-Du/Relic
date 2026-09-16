"""When is an adopted rule doing more harm than good?

The question matters because it is the only thing that puts "relax this rule"
or "repeal it" in front of anybody. Nothing else in the organization proposes
undoing a rule; a member who is being refused can revise the work, and can
propose a further amendment, but the option to loosen the rule itself is dealt
by this predicate or not at all.

It used to ask one thing: is the rule broken at least as often as it is
followed. That misses the shape that actually costs a run. A B3 arm adopted a
contract-evidence gate at tick 18, unanimously, and kept it for the remaining
318 ticks. The gate blocked 290 merges. It was also complied with 349 times,
so the violation share came to 0.45 -- below the 0.5 bar -- and the repair
option was never dealt to anyone. Across the whole run there were 33
amendments and every one of them extended the rule; there was no repeal, and
not one policy-repair wish. The arm finished at 7 of 35 behavioural cases with
five patches on the mainline, against 14 and forty-three for the arm with no
institutions at all.

The old test has a perverse property: compliance is the denominator, so the
more seriously a rule is taken, the harder it becomes to call it harmful. A
rule that everyone obeys and that stops all delivery is invisible to it.

So harm is judged two ways now, and either is enough:

  * the rule is broken at least as often as it is followed -- nobody can meet
    it, which is the shape the old test was written for; or
  * the rule has been refusing requests while nothing reaches the mainline --
    it is being obeyed, and it is the reason the work is not landing.

The second reads the enforcement events, which already record whether the
refusal blocked the request and which request it was.
"""

from __future__ import annotations

from typing import Any

# How far back to look, and how many distinct requests must have been refused
# in that window. Two days of blocked work with nothing shipped is enough to
# ask whether the rule is worth what it costs; less than that is ordinary
# friction and the rule should be given the chance to do its job.
BLOCK_WINDOW_TICKS = 48
BLOCKED_REQUESTS_MIN = 3


def _registry_id(spec: Any) -> str:
    """The registry's id for a proposal manager's spec (they number separately)."""
    return f"proto_spec_{str(getattr(spec, 'protocol_id', '')).split('_')[-1]}"


def blocked_without_delivery(world: Any, spec: Any) -> tuple[int, int]:
    """How many distinct requests this rule refused lately, and how many merged.

    The first is the rule's own; the second is the organization's. Only the
    first tells one rule from another -- when the pipeline is stalled every rule
    reads the same on the second -- so the count is what a reader should weigh
    and the second is what to weigh it against.
    """
    reg = getattr(world, "protocol_registry", None)
    if reg is None:
        return 0, True
    tick = int(getattr(world, "world_tick", 0) or 0)
    since = tick - BLOCK_WINDOW_TICKS
    pid = _registry_id(spec)

    blocked: set[str] = set()
    for event in (getattr(reg, "events", None) or []):
        if getattr(event, "protocol_id", None) != pid:
            continue
        if str(getattr(event, "event_type", "")) != "enforcement":
            continue
        if int(getattr(event, "tick", 0) or 0) < since:
            continue
        data = getattr(event, "data", None) or {}
        if data.get("blocked"):
            blocked.add(str(data.get("context_id") or getattr(event, "event_id", "")))

    merged = 0
    repo = getattr(getattr(world, "repo_system", None), "repo", None)
    for pr in (getattr(repo, "pull_requests", None) or {}).values():
        merged_tick = getattr(pr, "merged_tick", None)
        if merged_tick is not None and int(merged_tick) >= since:
            merged += 1
    return len(blocked), merged


def recent_refusals(world: Any, spec: Any, limit: int = 8) -> tuple[list[dict], int]:
    """What this rule turned away in the window, and how many there were in all.

    A count says a rule is expensive; it does not say whether the price is
    worth paying. What was being attempted, and what the rule demanded instead,
    is what tells an over-strict rule from one that is catching real mistakes,
    and it is the difference a member has to judge before asking to loosen it.

    Same window as the count beside it. Reading every refusal ever recorded and
    showing the newest few put three refusals from t187 and t221 under a count
    of what had been turned away since t288: two different questions answered
    as though they were one.
    """
    pid = _registry_id(spec)
    tick = int(getattr(world, "world_tick", 0) or 0)
    since = tick - BLOCK_WINDOW_TICKS
    out: list[dict] = []
    total = 0
    for event in reversed(list(getattr(world, "events", None) or [])):
        if not isinstance(event, dict):
            continue
        if event.get("subtype") != "patch_refused_by_protocol":
            continue
        if str(event.get("protocol_id") or "") not in (pid, str(getattr(spec, "protocol_id", ""))):
            continue
        if int(event.get("tick") or 0) < since:
            continue
        total += 1
        if len(out) >= limit:
            continue
        # The recorded reason opens with the rule's own text, which is already
        # above and would otherwise crowd out the part that differs between one
        # refusal and the next.
        said = str(event.get("reason") or "")
        said = said.split(" — ", 1)[-1] if " — " in said else said
        out.append({
            "tick": int(event.get("tick") or 0),
            "who": str(event.get("agent_id") or ""),
            "what_they_were_changing": str(event.get("artifact_id") or ""),
            "rewrite_number": int(event.get("attempt") or 1),
            "what_it_demanded": said[:300],
        })
    return out, total


def held_at_the_gate(world: Any, spec: Any, limit: int = 8) -> list[dict]:
    """Which requests this rule is holding, and what each was trying to land.

    The gate is the channel that matters on a stalled pipeline -- a rule can
    hold every open request there while sending nothing back to any author --
    and the enforcement event names only a request id. A member weighing
    whether the rule is worth its cost needs to know whose work is behind it
    and how long it has been waiting.
    """
    reg = getattr(world, "protocol_registry", None)
    if reg is None:
        return []
    tick = int(getattr(world, "world_tick", 0) or 0)
    since = tick - BLOCK_WINDOW_TICKS
    pid = _registry_id(spec)
    repo = getattr(getattr(world, "repo_system", None), "repo", None)
    requests = getattr(repo, "pull_requests", None) or {}

    latest: dict[str, int] = {}
    for event in (getattr(reg, "events", None) or []):
        if getattr(event, "protocol_id", None) != pid:
            continue
        if str(getattr(event, "event_type", "")) != "enforcement":
            continue
        at = int(getattr(event, "tick", 0) or 0)
        if at < since:
            continue
        data = getattr(event, "data", None) or {}
        if not data.get("blocked"):
            continue
        rid = str(data.get("context_id") or "")
        if rid:
            latest[rid] = max(latest.get(rid, 0), at)

    out: list[dict] = []
    for rid, at in sorted(latest.items(), key=lambda kv: -kv[1])[:limit]:
        pr = requests.get(rid)
        entry: dict = {"request": rid, "last_held_at_tick": at}
        if pr is not None:
            entry["opened_by"] = str(getattr(pr, "author_id", "")
                                     or getattr(pr, "opened_by", "") or "")
            opened = getattr(pr, "opened_tick", None)
            if opened is not None:
                entry["waiting_since_tick"] = int(opened)
            title = str(getattr(pr, "title", "") or "")
            if title:
                entry["what_it_would_land"] = title[:160]
            brief = str(getattr(pr, "ci_brief", "") or "")
            if brief:
                entry["what_the_check_said"] = brief[:200]
        out.append(entry)
    return out


def gate_holds(world: Any, spec: Any, limit: int = 8) -> tuple[list[dict], int]:
    """The requests this rule is holding at the gate, and who is waiting on them.

    A count of eight held requests tells a member that something is stuck; it
    does not say whose work, on which issues, or whether the check ever ran.
    All three are already on the enforcement event -- the request, the issues it
    carries, the state of its check -- and the author is on the violation event
    it is paired with.
    """
    reg = getattr(world, "protocol_registry", None)
    if reg is None:
        return [], 0
    tick = int(getattr(world, "world_tick", 0) or 0)
    since = tick - BLOCK_WINDOW_TICKS
    pid = _registry_id(spec)
    events = list(getattr(reg, "events", None) or [])
    authors = {getattr(e, "event_id", ""): getattr(e, "actor_id", "")
               for e in events if str(getattr(e, "event_type", "")) == "violation"}

    seen: set[str] = set()
    out: list[dict] = []
    total = 0
    for event in reversed(events):
        if getattr(event, "protocol_id", None) != pid:
            continue
        if str(getattr(event, "event_type", "")) != "enforcement":
            continue
        if int(getattr(event, "tick", 0) or 0) < since:
            continue
        data = getattr(event, "data", None) or {}
        if not data.get("blocked"):
            continue
        request = str(data.get("context_id") or "")
        if request in seen:
            continue
        seen.add(request)
        total += 1
        if len(out) >= limit:
            continue
        attributes = ((data.get("state_before") or {}).get("attributes") or {})
        out.append({
            "tick": int(getattr(event, "tick", 0) or 0),
            "request": request,
            "whose": authors.get(str(data.get("violation_event_id") or ""), ""),
            "issues_it_carries": list(attributes.get("issue_ids") or [])[:4],
            "its_check": str(attributes.get("ci_status") or ""),
        })
    return out, total


def rules_and_their_refusals(world: Any, *, limit: int = 8) -> list[dict]:
    """Every adopted rule, what it has turned away lately, and in whose words.

    Assembled once here because two readers need the same picture: a member
    reflecting on whether a rule has become the problem, and a member choosing
    among protocol actions. Neither could see it before -- no per-rule count
    reached any prompt anywhere -- so the only rules anybody could judge were
    the ones that had personally refused them.
    """
    pm = getattr(world, "proposal_manager", None)
    reg = getattr(world, "protocol_registry", None)
    if pm is None or reg is None:
        return []
    protocols = getattr(reg, "protocols", None) or {}
    out: list[dict] = []
    for spec in (getattr(pm, "protocol_specs", None) or {}).values():
        if str(getattr(spec, "status", "")) != "adopted":
            continue
        protocol = protocols.get(_registry_id(spec))
        rule = str((getattr(protocol, "rule_summary", "") if protocol is not None else "")
                   or getattr(spec, "name", "") or "")[:400]
        entry: dict = {"rule": rule or str(getattr(spec, "protocol_id", ""))}
        # Two ways this rule turns work away, and they are not the same event:
        # a request held at the gate, and a patch sent back while it was being
        # written. Reporting one count over the other's list read as a list
        # that had lost most of its entries.
        held, held_total = gate_holds(world, spec, limit=limit)
        refusals, refused_total = recent_refusals(world, spec, limit=limit)
        if held_total:
            entry["requests_it_is_holding"] = held_total
            entry["which_requests"] = held
            if held_total > len(held):
                entry["which_requests_shown"] = f"{len(held)} of {held_total}"
        if refused_total:
            entry["patches_it_sent_back"] = refused_total
            entry["what_it_sent_back"] = refusals
            if refused_total > len(refusals):
                entry["what_it_sent_back_shown"] = f"{len(refusals)} of {refused_total}"
        out.append(entry)
    out.sort(key=lambda e: -(int(e.get("requests_it_is_holding") or 0)
                             + int(e.get("patches_it_sent_back") or 0)))
    return out


def rule_is_doing_harm(world: Any, spec: Any) -> tuple[bool, str]:
    """Whether to offer the organization the chance to loosen this rule, and why."""
    violations = int(getattr(spec, "violation_count", 0) or 0)
    uses = int(getattr(spec, "use_count", 0) or 0)
    if violations >= 4 and violations / max(violations + uses, 1) >= 0.5:
        return True, (f"it is broken {violations} times against {uses} compliant "
                      f"uses — nobody can meet it as written")

    blocked, merged = blocked_without_delivery(world, spec)
    # Turning away more than the organization managed to land. Requiring that
    # nothing at all merged never fired: a stalled pipeline still dribbles the
    # occasional merge, and one merge in a window was enough to excuse a rule
    # that had refused fourteen requests in it. What matters is the exchange
    # rate -- a gate that refuses three while twenty ship is doing its job.
    if blocked >= BLOCKED_REQUESTS_MIN and blocked > merged:
        landed = (f"only {merged} reached the mainline" if merged
                  else "nothing reached the mainline")
        return True, (f"it refused {blocked} requests in the last "
                      f"{BLOCK_WINDOW_TICKS} ticks while {landed} in that time "
                      f"— it is being followed, and it is what the work is "
                      f"waiting on")
    return False, ""
