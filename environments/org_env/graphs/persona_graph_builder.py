"""PersonaGraphBuilder — generic persona graph generation (frontend redesign).

Turns ``AgentProfile + Skills + State + Routine + ontology rules + event evidence``
into a persona graph that EXPLAINS how a persona drives action/speech/risk — not a
raw dump of attributes. Four views:

* **summary** (default, human-readable): agent → 5 clusters (Core Traits / Skills /
  Communication Style / Failure Risks / Work State) → top-salient nodes → top
  action/speech tendencies. NO policy-feature nodes.
* **policy_debug**: trait/skill/state → policy feature → action/speech tendency.
* **evidence**: recent events → support → trait/skill/failure/tendency.
* **raw**: everything.

HARD RULE: no agent-specific hardcoding (no ``if agent_id == ...``). All structure
comes from :mod:`persona_mapping` rules + the agent's own profile/skills/state +
world events. New agents work automatically.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from environments.org_env.graphs import persona_mapping as M

MODES = ("summary", "policy_debug", "evidence", "raw")


def _node(nid, ntype, cluster, *, label=None, value=None, salience=0.0, **attrs):
    return {"id": nid, "type": ntype, "cluster": cluster,
            "label": label if label is not None else (nid.split(":", 1)[-1] if ":" in nid else nid),
            "value": value, "salience": round(max(0.0, min(1.0, float(salience))), 3),  # normalized [0,1]
            "attrs": attrs}


def _edge(src, dst, etype, *, weight=1.0, explanation="", evidence_ids=None):
    return {"src": src, "dst": dst, "type": etype, "weight": round(float(weight), 3),
            "explanation": explanation, "evidence_ids": list(evidence_ids or [])}


def _num(d, k, default=0.0):
    v = (d or {}).get(k, default)
    return float(v) if isinstance(v, (int, float)) else default


def _clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def _trait_label(name: str, value) -> str:
    """Direction-aware label so a low-value-but-salient trait reads correctly
    (fix #1: conformity=0.15 -> 'low_conformity', not 'conformity')."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return name
    if v < 0.4:
        return f"low_{name}"
    if v > 0.6:
        return f"high_{name}"
    return name


def _state_salience(field: str, value: float, current_tick: int) -> float:
    """Normalize work-state fields to [0,1] *signal strength* (NOT raw value) so a
    timestamp like next_available_tick never dominates the summary (fix #1)."""
    if field == "next_available_tick":
        return _clamp(max(0, value - current_tick) / 8.0)        # busy-until delta
    if field == "daily_message_count":
        return _clamp(value / 10.0)
    if field in ("daily_meeting_count", "deep_work_blocks_used_today"):
        return _clamp(value / 5.0)
    if field == "aux_speech_slots_remaining":
        return _clamp(value / 2.0) * 0.3
    if field == "morale":
        return _clamp(abs(value - 0.6))
    if field == "attention_remaining_today":
        return _clamp(abs(value - 1.0))                          # deviation from full
    return _clamp(abs(value))                                    # fatigue/stress/burnout/... (0-1)


def _banned_state(nid):
    return nid.startswith("state:") and nid.split(":", 1)[1] in M.SUMMARY_STATE_BAN


def _salient(ns, k):
    """Top-k salient ids, EXCLUDING layout-only hub/cluster nodes AND scheduling/
    bookkeeping state nodes (next_available_tick, aux_speech_slots_remaining, ...)."""
    real = [n for n in ns if n["type"] != "cluster" and not _banned_state(n["id"])]
    return [n["id"] for n in sorted(real, key=lambda x: -x["salience"])[:k]]


def _merge_edges(edges):
    """De-duplicate by (src, dst, type): sum weights, union evidence, merge text (fix #3)."""
    seen = {}
    order = []
    for e in edges:
        key = (e["src"], e["dst"], e["type"])
        if key in seen:
            s = seen[key]
            if s["weight"] and e["weight"]:
                s["weight"] = round(s["weight"] + e["weight"], 3)
            else:
                s["weight"] = s["weight"] or e["weight"]
            s["evidence_ids"] = sorted(set(s["evidence_ids"]) | set(e["evidence_ids"]))
            if e["explanation"] and e["explanation"] not in (s["explanation"] or ""):
                s["explanation"] = (s["explanation"] + "; " + e["explanation"]).strip("; ")
        else:
            seen[key] = dict(e)
            order.append(key)
    return [seen[k] for k in order]


class PersonaGraphBuilder:
    def build(self, agent: Any, world: Any = None, context: Any = None,
              mode: str = "summary") -> Dict[str, Any]:
        aid = agent.id
        prof, skills = dict(agent.profile or {}), dict(agent.skills or {})
        ws = agent.work_state.snapshot() if hasattr(agent, "work_state") else {}
        current_tick = int(getattr(world, "world_tick", 0) or 0) if world is not None else 0

        nodes: Dict[str, Dict[str, Any]] = {}
        edges: List[Dict[str, Any]] = []

        def add(n):
            if n["id"] not in nodes:
                nodes[n["id"]] = n
            return n["id"]

        agent_id = add(_node(aid, "agent", "agent", label=getattr(agent, "name", aid), salience=1.0))

        # -- Layer A: static persona --------------------------------------
        for t, v in prof.items():
            sal = abs(float(v) - 0.5) * 2 * (1 + min(1.0, sum(abs(c) for c in [c for (k, f, c) in M.TRAIT_TO_FEATURE_RULES if k == t])))
            add(_node(f"trait:{t}", "trait", "core_traits", value=round(float(v), 3), salience=sal,
                      label=_trait_label(t, v)))
            edges.append(_edge(agent_id, f"trait:{t}", "exhibits", weight=float(v)))
        for s, v in skills.items():
            add(_node(f"skill:{s}", "skill", "skills", value=round(float(v), 3), salience=float(v)))
            edges.append(_edge(agent_id, f"skill:{s}", "has_skill", weight=float(v)))
        for k, v in (agent.communication_style or {}).items():
            is_str = isinstance(v, str)
            sv = _num(agent.communication_style, k)
            # fix: string-valued styles (e.g. tone="warm") must NOT get salience 1.0
            # from _num()->0.0; give them a modest fixed salience instead.
            sal = 0.3 if is_str else abs(sv - 0.5) * 2
            add(_node(f"style:{k}", "communication_style", "communication_style",
                      value=v if is_str else round(sv, 3), salience=sal))
            edges.append(_edge(agent_id, f"style:{k}", "speaks_with"))
        for k, v in ws.items():
            if isinstance(v, (int, float)) and abs(float(v)) > 0.001:
                sal = _state_salience(k, float(v), current_tick)
                if sal <= 0.001:
                    continue
                add(_node(f"state:{k}", "state", "work_state", value=round(float(v), 3), salience=sal))
                edges.append(_edge(agent_id, f"state:{k}", "currently"))

        # -- Layer B: policy mechanism (trait/skill -> feature) -----------
        feat_ids = set()
        for key, feat, coeff in M.TRAIT_TO_FEATURE_RULES:
            if key not in prof and key not in skills:
                continue
            val = prof.get(key, skills.get(key, 0.0))
            src = f"trait:{key}" if key in prof else f"skill:{key}"
            fid = add(_node(f"feat:{feat}", "feature", "policy_feature", label=feat,
                            salience=abs(coeff * float(val))))
            feat_ids.add(fid)
            edges.append(_edge(src, fid, "boosts" if coeff >= 0 else "suppresses",
                               weight=coeff * float(val),
                               explanation=f"{key}={val:.2f} {'raises' if coeff>=0 else 'lowers'} {feat}"))

        # -- Layer C: behavior tendencies (action / speech / risk) --------
        role_priors = M.role_priors_for(getattr(agent, "role", ""))
        act_strengths = self._tendencies(agent, M.ACTION_METADATA, prof, skills, nodes, edges,
                                         add, ntype="action", cluster="action_tendency", kind="act",
                                         role_priors=role_priors)
        sp_strengths = self._tendencies(agent, M.SPEECH_METADATA, prof, skills, nodes, edges,
                                        add, ntype="speech", cluster="speech_tendency", kind="speech",
                                        role_priors=role_priors)
        # state -> tendency
        for st, rules in M.STATE_TO_TENDENCY.items():
            sv = _num(ws, st)
            if abs(sv) < 0.05:
                continue
            for action, sign in rules:
                tid = f"act:{action}"
                add(_node(tid, "action", "action_tendency", label=action, salience=abs(sign * sv)))
                add(_node(f"state:{st}", "state", "work_state", value=round(sv, 3),
                          salience=_state_salience(st, sv, current_tick)))
                edges.append(_edge(f"state:{st}", tid, "raises" if sign >= 0 else "lowers",
                                   weight=sign * sv, explanation=f"{st}={sv:.2f}"))
        # failure-mode -> risk
        for fm in (agent.failure_modes or []):
            add(_node(f"fail:{fm}", "failure_mode", "failure_risks", salience=0.55))
            edges.append(_edge(agent_id, f"fail:{fm}", "prone_to"))
            risk_feat, risk_action = self._failure_risk(fm)
            if risk_action:
                rid = add(_node(f"risk:{risk_action}", "risk", "failure_risks", label=risk_action, salience=0.6))
                edges.append(_edge(f"fail:{fm}", rid, "increases_risk",
                                   explanation=f"{fm} → {risk_action}"))

        # -- Layer D: evidence (recent events support tendencies/traits) --
        ev_nodes, ev_edges = self._evidence(agent, world, nodes)
        for n in ev_nodes:
            add(n)
        edges.extend(ev_edges)
        # evidence boosts salience of supported nodes (clamped to [0,1])
        for e in ev_edges:
            tgt = nodes.get(e["dst"])
            if tgt:
                tgt["salience"] = round(min(1.0, tgt["salience"] + 0.15), 3)

        edges = _merge_edges(edges)                       # fix #3: de-dup edges
        raw_nodes = list(nodes.values())
        raw_count = (len(raw_nodes), len(edges))
        sub = self._view(mode, aid, nodes, edges)
        sub["agent_id"] = aid
        sub["mode"] = mode
        sub["raw_node_count"] = raw_count[0]
        sub["raw_edge_count"] = raw_count[1]
        sub["available_modes"] = list(MODES)
        return sub

    # -- tendency derivation (generic) ------------------------------------
    def _tendencies(self, agent, registry, prof, skills, nodes, edges, add, *, ntype, cluster, kind,
                    role_priors=None):
        role_priors = role_priors or {"families": set(), "speech": set()}
        strengths: Dict[str, float] = {}
        for action, meta in registry.items():
            s = 0.0
            contrib: List[Tuple[str, float]] = []
            for feat in meta.get("feature_tags", []) or []:
                if feat in M.RISK_FEATURES:        # fix #4: don't flip sign on risk features
                    continue
                for k, f, c in M.TRAIT_TO_FEATURE_RULES:
                    if f == feat and (k in prof or k in skills):
                        v = prof.get(k, skills.get(k, 0.0))
                        s += c * v
                        if abs(c * v) > 0.05:
                            contrib.append((f"trait:{k}" if k in prof else f"skill:{k}", c * v))
            # explicit driver traits (correctly-signed ontology where the feature path is ambiguous)
            for tr, coeff in meta.get("driver_traits", []) or []:
                if tr in prof:
                    v = float(prof[tr])
                    s += coeff * v
                    if abs(coeff * v) > 0.05:
                        contrib.append((f"trait:{tr}", coeff * v))
            req = meta.get("required_skills", []) or []
            for sk in req:
                v = float(skills.get(sk, 0.0))
                s += 0.5 * v
                if v > 0.1:
                    contrib.append((f"skill:{sk}", 0.5 * v))
            if abs(s) < 0.25:
                continue
            # fix #5: skill-fit = the BEST relevant skill (so adding alternative
            # skills to an action's metadata lets a differently-skilled agent qualify).
            skill_fit = max((float(skills.get(sk, 0.0)) for sk in req), default=0.5) if req else 0.5
            salience = abs(s) * (0.35 + 0.65 * skill_fit)
            # role/context gating: boost on-role tendencies, damp off-role low-skill ones
            # (so a founder's urgency doesn't pull run_cheap_pilot above their real role).
            fam = meta.get("family")
            on_role = (kind == "act" and fam in role_priors["families"]) or \
                      (kind == "speech" and action in role_priors["speech"])
            if on_role:
                salience *= 1.5
            elif skill_fit < 0.3:
                salience *= 0.6
            suppressed = s < 0                    # fix #3: negative = suppressed tendency
            strengths[action] = s
            tid = f"{'act' if kind=='act' else 'speech'}:{action}"
            add(_node(tid, ntype, cluster, label=(f"suppressed: {action}" if suppressed else action),
                      value=round(s, 3), salience=salience, skill_fit=round(skill_fit, 3),
                      suppressed=suppressed, on_role=on_role, family=fam))
            contrib.sort(key=lambda x: -abs(x[1]))
            for src, w in contrib[:3]:
                nm = src.split(":")[1]
                is_tr = src.startswith("trait")
                val = prof.get(nm) if is_tr else skills.get(nm)
                add(_node(src, "trait" if is_tr else "skill",
                          "core_traits" if is_tr else "skills", value=val, salience=abs(w),
                          label=_trait_label(nm, val) if is_tr else nm))
                edges.append(_edge(src, tid, "inclines_toward" if w >= 0 else "disinclines",
                                   weight=w, explanation=f"{nm} -> {action}"))
        return strengths

    def _failure_risk(self, fm: str) -> Tuple[str, str]:
        low = fm.lower()
        for kw, (feat, action) in M.FAILURE_KEYWORD_RISK.items():
            if kw in low:
                return feat, action
        return "", ""

    def _top_trait_driver(self, meta, prof, skills):
        """The single trait/skill that most drives an action (for evidence edges)."""
        best, best_w = None, 0.0
        for feat in meta.get("feature_tags", []) or []:
            for k, f, c in M.TRAIT_TO_FEATURE_RULES:
                if f == feat and (k in prof or k in skills):
                    w = c * prof.get(k, skills.get(k, 0.0))
                    if abs(w) > abs(best_w):
                        best, best_w = (f"trait:{k}" if k in prof else f"skill:{k}"), w
        return best

    # -- evidence from real events (semantic: event -> tendency + trait/skill) --
    def _evidence(self, agent, world, nodes, limit: int = 14):
        ev_nodes: List[Dict[str, Any]] = []
        ev_edges: List[Dict[str, Any]] = []
        if world is None:
            return ev_nodes, ev_edges
        aid = agent.id
        prof, skills = dict(agent.profile or {}), dict(agent.skills or {})
        seq = [0]
        # repetition counts -> stronger evidence (10x promise_work > 1x schedule_meeting)
        counts: Dict[str, int] = {}
        for a in getattr(world, "action_log", []):
            if a.get("agent_id") == aid:
                counts[a.get("action_type", "")] = counts.get(a.get("action_type", ""), 0) + 1
        for t in getattr(world, "text_generation_log", []):
            if t.get("agent") == aid:
                counts[t.get("speech_act", "")] = counts.get(t.get("speech_act", ""), 0) + 1

        def _strength(key):
            return round(_clamp(0.2 + 0.15 * counts.get(key, 1)), 3)

        def ev_node(tick, label, event_type, target):
            seq[0] += 1
            eid = f"ev:{tick}:{seq[0]}"
            ev_nodes.append(_node(eid, "evidence", "evidence", label=label, value=tick, salience=0.5,
                                  event_type=event_type, target_object=target))
            return eid

        def support(eid, target_id, reason, strength):
            if target_id and target_id in nodes:
                e = _edge(eid, target_id, "supports", weight=strength, explanation=reason, evidence_ids=[eid])
                e["strength"] = strength
                e["reason"] = reason
                ev_edges.append(e)

        # recent actions -> support act tendency + required skills + top trait driver
        acts = [a for a in getattr(world, "action_log", []) if a.get("agent_id") == aid][-limit:]
        for a in reversed(acts):
            at = a.get("action_type", "")
            meta = M.ACTION_METADATA.get(at)
            if meta is None:
                continue
            st = _strength(at)
            eid = ev_node(a.get("tick"), f"did {at} (x{counts.get(at,1)})", "action", at)
            support(eid, f"act:{at}", f"performed {at} {counts.get(at,1)}x", st)
            for sk in (meta.get("required_skills", []) or []):
                support(eid, f"skill:{sk}", f"{at} exercises {sk}", st)
            support(eid, self._top_trait_driver(meta, prof, skills), f"{at} reflects disposition", st * 0.7)
        # recent speech acts -> support speech tendency + style/skill drivers
        for t in [t for t in getattr(world, "text_generation_log", []) if t.get("agent") == aid][-8:]:
            sa = t.get("speech_act", "")
            meta = M.SPEECH_METADATA.get(sa, {})
            st = _strength(sa)
            eid = ev_node(t.get("tick"), f"said {sa} (x{counts.get(sa,1)})", "speech", t.get("target"))
            support(eid, f"speech:{sa}", f"uttered {sa} {counts.get(sa,1)}x", st)
            for sk in (meta.get("required_skills", []) or []):
                support(eid, f"skill:{sk}", f"{sa} exercises {sk}", st)
            for stl in (meta.get("style_tags", []) or []):
                support(eid, f"style:{stl}", f"{sa} in {stl} style", st * 0.6)
        # commitments / disputes by this agent
        cr = getattr(world, "commitment_registry", None)
        if cr is not None:
            for c in [c for c in cr.commitments.values() if c.agent_id == aid][-4:]:
                eid = ev_node(c.created_tick, "promised work", "commitment", c.target_object_id)
                support(eid, "speech:promise_work", "made a commitment", _strength("promise_work"))
            for d in [d for d in cr.disputes.values() if d.challenger_id == aid][-4:]:
                eid = ev_node(d.created_tick, f"challenged {d.target_object_id}", "dispute", d.target_object_id)
                support(eid, "speech:challenge_result", "disputed a result", _strength("challenge_result"))
        return ev_nodes[:limit + 8], ev_edges

    # -- per-mode subgraph selection --------------------------------------
    def _view(self, mode, aid, nodes, edges):
        alln = list(nodes.values())
        if mode == "raw":
            return {"nodes": alln, "edges": edges, "clusters": self._clusters(alln),
                    "salient_node_ids": _salient(alln, 12)}
        if mode == "policy_debug":
            keep = {n["id"] for n in alln if n["cluster"] in
                    ("agent", "core_traits", "skills", "work_state", "policy_feature",
                     "action_tendency", "speech_tendency")}
            return self._assemble(aid, nodes, edges, keep)
        if mode == "evidence":
            keep = {n["id"] for n in alln if n["cluster"] in
                    ("agent", "core_traits", "skills", "failure_risks", "evidence",
                     "action_tendency", "speech_tendency")}
            return self._assemble(aid, nodes, edges, keep)
        # summary (default): clusters + top-salient nodes + tendencies, NO features
        return self._summary(aid, nodes, edges)

    # tightened, paper-ready caps (~23 nodes incl. agent, NO hub nodes)
    _SUMMARY_CAPS = {"core_traits": 4, "skills": 4, "communication_style": 2, "work_state": 2,
                     "failure_risks": 3, "action_tendency": 4, "speech_tendency": 3}
    _CLUSTER_LABEL = {"core_traits": "Core Traits", "skills": "Skills",
                      "communication_style": "Communication Style", "work_state": "Current Work State",
                      "failure_risks": "Failure Risks", "action_tendency": "Action Tendencies",
                      "speech_tendency": "Speech Tendencies"}

    def _summary(self, aid, nodes, edges):
        alln = list(nodes.values())
        keep = {aid}
        cluster_members: Dict[str, List[str]] = {}
        for cl, cap in self._SUMMARY_CAPS.items():
            members = [n for n in alln if n["cluster"] == cl and n["type"] not in ("agent", "cluster")]
            if cl == "work_state":      # drop scheduling/bookkeeping states from summary
                members = [n for n in members if not _banned_state(n["id"])]
            if cl in ("action_tendency", "speech_tendency"):   # no suppressed in summary
                members = [n for n in members if not n["attrs"].get("suppressed")]
            if cl == "action_tendency":
                members = [
                    n for n in members
                    if _summary_action_allowed(n)
                ]
            members.sort(key=lambda x: -x["salience"])
            if cl == "action_tendency":     # family diversity: avoid 4 near-identical review actions
                top, fam_count = [], {}
                for n in members:
                    fam = (M.ACTION_METADATA.get(n["id"].split(":", 1)[1], {}) or {}).get("family", "_")
                    if fam_count.get(fam, 0) >= 3:
                        continue
                    top.append(n)
                    fam_count[fam] = fam_count.get(fam, 0) + 1
                    if len(top) >= cap:
                        break
            else:
                top = members[:cap]
            if top:
                cluster_members[cl] = [n["id"] for n in top]
                keep.update(n["id"] for n in top)
        # NO hub nodes: grouping is returned as `clusters` metadata; the frontend uses it
        # for section titles and draws nodes grouped by each node's own `cluster` field.
        out_nodes = [nodes[aid]] + [nodes[i] for i in keep if i != aid]
        # keep agent->node structural edges + persona mechanism edges among kept nodes
        struct = ("exhibits", "has_skill", "speaks_with", "currently", "prone_to")
        mech = ("inclines_toward", "disinclines", "increases_risk", "raises", "lowers")
        es = [e for e in edges if e["src"] in keep and e["dst"] in keep and e["type"] in struct + mech]
        # connectors so tendency/risk nodes (whose driver trait may be capped out) stay
        # attached to the agent (agent-centered star + persona mechanism overlay)
        linked = {e["dst"] for e in es} | {e["src"] for e in es}
        for nid in keep:
            if nid == aid or nid in linked:
                continue
            cl = nodes[nid]["cluster"]
            etype = ("tends_to" if cl in ("action_tendency", "speech_tendency")
                     else "at_risk" if cl == "failure_risks" else "has")
            es.append(_edge(aid, nid, etype))
        clusters_meta = [{"cluster": cl, "label": self._CLUSTER_LABEL.get(cl, cl),
                          "node_ids": cluster_members[cl]}
                         for cl in self._SUMMARY_CAPS if cl in cluster_members]
        return {"nodes": out_nodes, "edges": es, "clusters": clusters_meta,
                "salient_node_ids": _salient(out_nodes, 10)}

    def _assemble(self, aid, nodes, edges, keep):
        ns = [n for n in nodes.values() if n["id"] in keep]
        es = [e for e in edges if e["src"] in keep and e["dst"] in keep]
        return {"nodes": ns, "edges": es, "clusters": self._clusters(ns),
                "salient_node_ids": _salient(ns, 12)}

    def _clusters(self, ns):
        out: Dict[str, int] = {}
        for n in ns:
            out[n["cluster"]] = out.get(n["cluster"], 0) + 1
        return out


def _summary_action_allowed(node: Dict[str, Any]) -> bool:
    attrs = node.get("attrs", {})
    if attrs.get("on_role"):
        return True
    skill_fit = float(attrs.get("skill_fit", 0.5))
    if attrs.get("family") == "tracking":
        return skill_fit >= 0.5
    return skill_fit >= 0.3


_BUILDER = PersonaGraphBuilder()


def build_persona_graph(agent, world=None, context=None, mode="summary") -> Dict[str, Any]:
    return _BUILDER.build(agent, world, context, mode)


def build_all_modes(agent, world=None, context=None) -> Dict[str, Any]:
    return {m: _BUILDER.build(agent, world, context, m) for m in MODES}


__all__ = ["PersonaGraphBuilder", "build_persona_graph", "build_all_modes", "MODES"]
