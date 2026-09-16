import { useMemo, useState } from "react";
import { useInspector } from "../ctx";
import GraphView from "../GraphView";
import { Cards, DataTable, Section, StatCard, StatusTag, Tag, num } from "../ui";
import { EP_COLORS, WISH_COLORS } from "../theme";
import type { PanelProps } from "./common";

const arrOf = (x: any): any[] => (Array.isArray(x) ? x : x?.items || []);

export function Episodes({ f }: PanelProps) {
  const { openEpisode } = useInspector();
  const [filt, setFilt] = useState("all");
  const eb = f.episodes || { items: [], by_type: {}, open_count: 0, closed_count: 0, total: 0 };
  const types = Object.keys(eb.by_type || {});
  const items = (eb.items || []).slice().reverse();
  const epName = (a: string) => (f.agents[a] || {}).name || a;
  const shown = items.filter((e: any) => filt === "all" || e.episode_type === filt);
  return (
    <>
      <Cards>
        <StatCard label="episodes" value={eb.total} />
        <StatCard label="open" value={eb.open_count} />
        <StatCard label="closed" value={eb.closed_count} />
      </Cards>
      <div className="row" style={{ margin: "10px 0" }}>
        <span className="muted">filter:</span>
        <span className={"chip" + (filt === "all" ? " active" : "")} onClick={() => setFilt("all")}>all</span>
        {types.map((t) => (
          <span key={t} className={"chip" + (filt === t ? " active" : "")} onClick={() => setFilt(t)}>{t.replace("_episode", "")} ({eb.by_type[t]})</span>
        ))}
      </div>
      {shown.length ? shown.map((e: any) => {
        const col = EP_COLORS[e.episode_type] || "#8b949e";
        return (
          <div className="section" key={e.episode_id} style={{ borderLeft: `3px solid ${col}`, cursor: "pointer" }} onClick={() => openEpisode(e.episode_id)}>
            <div className="row" style={{ justifyContent: "space-between" }}>
              <b>{e.title || e.episode_type}</b>
              <StatusTag value={e.status} />
            </div>
            <div className="muted" style={{ fontSize: 11 }}>{e.episode_type} · t{e.start_tick}{e.end_tick != null ? `→${e.end_tick}` : "→…"} · {(e.linked_object_ids || []).length} objects · {(e.participants || []).length} ppl</div>
            <div className="kv" style={{ marginTop: 4 }}>{e.problem_statement || ""}</div>
            <div className="pillrow" style={{ marginTop: 4 }}>{(e.participants || []).map((a: string) => <Tag key={a}>{epName(a)}</Tag>)}</div>
            {e.outcome_summary ? <div className="muted" style={{ fontSize: 11, marginTop: 4 }}>{e.outcome_summary}</div> : null}
          </div>
        );
      }) : <div className="empty">no episodes yet — run more ticks</div>}
    </>
  );
}

export function Reflections({ f }: PanelProps) {
  const { openWish, openReflection } = useInspector();
  const refl = arrOf(f.reflections);
  const wishes = arrOf(f.wishes);
  const reflectors = new Set(refl.map((r: any) => r.agent_id)).size;
  return (
    <>
      <Cards>
        <StatCard label="reflections" value={refl.length} />
        <StatCard label="wishes" value={wishes.length} />
        <StatCard label="agents reflecting" value={reflectors} />
      </Cards>
      <Section title="Wishes" sub="extracted from reflections">
        <DataTable
          rows={wishes.slice().reverse()}
          onRowClick={(w: any) => openWish(w.wish_id)}
          empty="no wishes yet"
          columns={[
            { header: "type", cell: (w: any) => <Tag color={WISH_COLORS[w.wish_type]}>{w.wish_type}</Tag> },
            { header: "agent", cell: (w: any) => w.agent_id },
            { header: "urgency", cell: (w: any) => num(w.urgency, 2) },
            { header: "need", cell: (w: any) => String(w.interpreted_need || "").slice(0, 70) },
          ]}
        />
      </Section>
      <Section title="Reflections">
        <DataTable
          rows={refl.slice().reverse()}
          onRowClick={(r: any) => openReflection(r.reflection_id)}
          empty="no reflections yet"
          columns={[
            { header: "t", cell: (r: any) => r.tick },
            { header: "agent", cell: (r: any) => r.agent_id },
            { header: "ideas", cell: (r: any) => (r.ideas || []).length },
            { header: "wishes", cell: (r: any) => (r.created_wish_ids || []).length },
            { header: "assessment", cell: (r: any) => String(r.team_assessment || r.self_assessment || "").slice(0, 70) },
          ]}
        />
      </Section>
    </>
  );
}

export function LLM({ f }: PanelProps) {
  const llm = f.llm || {};
  const decisions = arrOf(f.action_decisions).length ? arrOf(f.action_decisions) : arrOf(llm.decisions);
  const accepted = decisions.filter((d: any) => (d.validation_status || "").includes("accept")).length;
  const rejected = decisions.filter((d: any) => (d.validation_status || "").includes("reject")).length;
  return (
    <>
      <Cards>
        <StatCard label="LLM" value={llm.provider || (llm.enabled ? "on" : "off")} />
        <StatCard label="calls" value={llm.calls ?? llm.call_count ?? 0} />
        <StatCard label="failures" value={llm.failures ?? llm.failure_count ?? 0} />
        <StatCard label="fallbacks" value={llm.fallbacks ?? llm.fallback_count ?? 0} />
        <StatCard label="decisions" value={decisions.length} />
        <StatCard label="accepted" value={accepted} />
        <StatCard label="rejected" value={rejected} />
      </Cards>
      <Section title={`Action decisions (${decisions.length})`}>
        {decisions.length ? decisions.slice().reverse().slice(0, 60).map((d: any, i: number) => (
          <div className="kv" key={i} style={{ paddingBottom: 5 }}>
            <b>t{d.tick}</b> {d.agent || d.agent_id} → <Tag tone={(d.validation_status || "").includes("accept") ? "good" : "bad"}>{d.candidate_action || d.action}</Tag>
            <span className="muted" style={{ fontSize: 11 }}> {d.decision_source ? `[${d.decision_source}]` : ""} {d.confidence != null ? `conf ${num(d.confidence, 2)}` : ""} {d.target ? `→ ${d.target}` : ""}</span>
            {(d.rationale || d.expected_effect) ? <div className="muted" style={{ fontSize: 11 }}>{d.rationale || d.expected_effect}</div> : null}
            {d.rejection_reason ? <div style={{ fontSize: 11, color: "var(--bad)" }}>rejected: {d.rejection_reason}</div> : null}
          </div>
        )) : <div className="empty">no LLM decisions yet</div>}
      </Section>
    </>
  );
}

export function Proposals({ f }: PanelProps) {
  const { openProposal } = useInspector();
  const props = arrOf(f.proposals);
  const tools = arrOf(f.tools);
  const protos = arrOf(f.protocol_specs);
  return (
    <>
      <Cards>
        <StatCard label="proposals" value={props.length} />
        <StatCard label="adopted tools" value={tools.length} />
        <StatCard label="adopted protocols" value={protos.length} />
      </Cards>
      <Section title="Proposals">
        {props.length ? props.slice().reverse().map((p: any) => (
          <div className="section" key={p.proposal_id} style={{ cursor: "pointer" }} onClick={() => openProposal(p.proposal_id)}>
            <div className="row" style={{ justifyContent: "space-between" }}>
              <b>{p.title || p.proposal_type || p.proposal_id}</b>
              <StatusTag value={p.status} />
            </div>
            <div className="muted" style={{ fontSize: 11 }}>{p.proposal_type} · by {p.proposer_id || p.author_id || "—"}</div>
            {p.rationale ? <div className="kv" style={{ marginTop: 4 }}>{String(p.rationale).slice(0, 140)}</div> : null}
          </div>
        )) : <div className="empty">no proposals yet</div>}
      </Section>
      <Section title="Adopted Tools">
        <div className="pillrow">{tools.length ? tools.map((t: any, i: number) => <Tag key={i} tone="good">{t.name || t.tool_id || JSON.stringify(t)}</Tag>) : <span className="muted">none</span>}</div>
      </Section>
      <Section title="Adopted Protocols">
        <div className="pillrow">{protos.length ? protos.map((t: any, i: number) => <Tag key={i} color="#d2a8ff">{t.name || t.protocol_type || t.protocol_id || JSON.stringify(t)}</Tag>) : <span className="muted">none</span>}</div>
      </Section>
    </>
  );
}

export function ProductSub({ f }: PanelProps) {
  const { openArtifact } = useInspector();
  const prod = f.product || { repo_files: [], open_issues: [], known_gaps: [], recent_changes: [] };
  const files = prod.repo_files || prod.artifacts || [];
  return (
    <>
      <Cards>
        <StatCard label="stage" value={String(prod.stage || "—").replace(/_/g, " ")} />
        <StatCard label="files" value={files.length} />
        <StatCard label="open issues" value={(prod.open_issues || []).length} />
        <StatCard label="known gaps" value={(prod.known_gaps || []).length} />
      </Cards>
      {prod.summary ? <Section title={prod.name || "Product"}><div className="kv">{prod.summary}</div></Section> : null}
      <Section title="Repo / docs / eval">
        <DataTable
          rows={files}
          onRowClick={(a: any) => openArtifact(a.artifact_id)}
          empty="no product files yet"
          columns={[
            { header: "path", cell: (a: any) => <code>{a.linked_file_path || a.title}</code> },
            { header: "type", cell: (a: any) => a.artifact_type },
            { header: "status", cell: (a: any) => <StatusTag value={a.status} /> },
            { header: "rev", cell: (a: any) => a.revision },
            { header: "known gaps", cell: (a: any) => (a.known_gaps || []).length },
          ]}
        />
      </Section>
      {(prod.open_issues || []).length ? (
        <Section title="Open issues">
          {(prod.open_issues || []).map((g: any, i: number) => (
            <div className="kv" key={i}><b>{g.artifact_id || g.issue_id || ""}</b> {g.title || ""} {g.priority ? <Tag tone="warn">{g.priority}</Tag> : null}<div className="muted" style={{ fontSize: 11 }}>{g.problem || g.summary || ""}</div></div>
          ))}
        </Section>
      ) : null}
      {(prod.recent_changes || []).length ? (
        <Section title="Recent product changes">
          {(prod.recent_changes || []).map((c: any, i: number) => (
            <div className="kv" key={i}><b>t{c.tick}</b> {c.path || ""} rev {c.rev} <StatusTag value={c.status} /></div>
          ))}
        </Section>
      ) : null}
      {(prod.known_gaps || []).length ? (
        <Section title="Known systemic gaps">{(prod.known_gaps || []).map((g: any, i: number) => <div className="kv muted" key={i}>• {typeof g === "string" ? g : g.description || JSON.stringify(g)}</div>)}</Section>
      ) : null}
    </>
  );
}

export function EventGraph({ f }: PanelProps) {
  const [agent, setAgent] = useState("");
  const agents = Object.keys(f.agents || {});
  const g = f.graphs?.event || { nodes: [], edges: [] };
  const filtered = useMemo(() => {
    if (!agent) return g;
    const keep = new Set<string>([agent]);
    (g.edges || []).forEach((e: any) => { if (e.src === agent) keep.add(e.dst); if (e.dst === agent) keep.add(e.src); });
    return { nodes: (g.nodes || []).filter((n: any) => keep.has(n.id)), edges: (g.edges || []).filter((e: any) => keep.has(e.src) && keep.has(e.dst)) };
  }, [g, agent]);
  return (
    <Section title="Event Graph" right={
      <select value={agent} onChange={(e) => setAgent(e.target.value)}>
        <option value="">all agents</option>
        {agents.map((a) => <option key={a} value={a}>{a}</option>)}
      </select>
    }>
      <GraphView graph={filtered} height={560} focus={agent || null} />
    </Section>
  );
}
