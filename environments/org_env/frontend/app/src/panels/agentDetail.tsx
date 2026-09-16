import { useState, type ReactNode } from "react";
import { useInspector } from "../ctx";
import GraphView from "../GraphView";
import { DataTable, KV, Section, Tag, num } from "../ui";
import { MeetingTable, MsgTable, TaskTable, TextTable } from "./common";
import type { PanelProps } from "./common";

const MODES = ["summary", "policy_debug", "evidence", "raw"] as const;
const ML: Record<string, string> = { summary: "Summary", policy_debug: "Policy Debug", evidence: "Evidence", raw: "Raw" };

function MemList({ title, items }: { title: string; items: any[] }) {
  if (!items || !items.length) return null;
  return <div className="kv"><b>{title}</b><div>{items.map((x, i) => <div className="muted" key={i} style={{ fontSize: 11 }}>• {String(x)}</div>)}</div></div>;
}

export function AgentDetail({ f }: PanelProps) {
  const { agent, selectAgent, openObj, openFile, openWish, openReflection } = useInspector();
  const [mode, setMode] = useState<string>("summary");
  const a = f.agents?.[agent!];
  if (!a) return <div className="empty">agent not found in this frame</div>;
  const ws = a.work_state || {}, os = a.org_state || {};
  const acts = (f.logs?.actions || []).filter((x: any) => x.agent_id === agent).slice(-25).reverse();
  const myMsgs = (f.internal?.messages || []).filter((m: any) => m.sender_id === agent).slice(-15).reverse();
  const myTasks = (f.internal?.tasks || []).filter((t: any) => (a.assigned_tasks || []).includes(t.task_id) || t.owner_id === agent);
  const myMeet = (f.internal?.meetings || []).filter((m: any) => (m.participants || []).includes(agent));
  const myCommit = (f.internal?.commitments || []).filter((c: any) => c.agent_id === agent);
  const myDisp = (f.internal?.disputes || []).filter((c: any) => c.challenger_id === agent);
  const myReq = (f.internal?.requests || []).filter((c: any) => c.requester_id === agent);
  const text = (f.logs?.text_generation || []).filter((t: any) => t.agent === agent).slice(-15).reverse();
  const mem = a.memory || {};
  const myRefl = ((f.reflections || {}).items || []).filter((r: any) => r.agent_id === agent).slice(-10).reverse();
  const wsx = a.workspace || {};
  const lf = wsx.local_files || [];
  const pgall = (f.graphs?.persona || {})[agent!] || {};
  const pg = pgall[mode] || pgall.summary || { nodes: [], edges: [] };
  const rc = pgall.summary || {};
  const trace = (f.logs?.policy_trace || []).filter((pt: any) => pt.agent_id === agent).slice(-12).reverse();

  return (
    <>
      <div className="row" style={{ marginBottom: 10 }}>
        <button onClick={() => selectAgent("")}>← all agents</button>
        <h2>{a.name} <span className="muted">· {a.codename} · {a.role}</span></h2>
        <Tag tone="viz">{a.current_status}</Tag>
      </div>

      <Section title="Persona Graph" right={
        <span className="pillrow">{MODES.map((m) => <span key={m} className={"chip" + (mode === m ? " active" : "")} style={{ padding: "1px 8px" }} onClick={() => setMode(m)}>{ML[m]}</span>)}</span>
      }>
        <div className="muted" style={{ marginBottom: 4 }}>{mode} view · {pg.nodes.length} nodes / {pg.edges.length} edges (raw: {rc.raw_node_count ?? "?"}n/{rc.raw_edge_count ?? "?"}e) · click a node for affected actions + evidence</div>
        <GraphView graph={pg} height={420} persona={pg} />
      </Section>

      <Section title="Social Graph (this agent)"><GraphView graph={f.graphs?.social} height={360} focus={agent} /></Section>

      <Section title={`Assigned Tasks (${myTasks.length})`}><TaskTable tasks={myTasks} /></Section>

      <Section title="Workspace" sub={`private to ${a.name} · ${lf.length} local file${lf.length === 1 ? "" : "s"}`}>
        <div className="kv"><b>current focus</b>: {wsx.current_focus || "—"}</div>
        <MemList title="Personal notes" items={(wsx.personal_notes || []).slice(-6)} />
        <MemList title="Private todos" items={(wsx.private_todos || []).slice(-6)} />
        <MemList title="Open questions" items={(wsx.open_questions || []).slice(-6)} />
        {(wsx.draft_docs || []).length ? <div className="kv"><b>drafts</b> <div className="pillrow">{wsx.draft_docs.map((d: string) => <span key={d} className="link-chip" onClick={() => openFile(d)}>{d}</span>)}</div></div> : null}
        {(wsx.saved_post_ids || []).length ? <div className="kv"><b>saved posts</b> <div className="pillrow">{wsx.saved_post_ids.map((d: string) => <span key={d} className="link-chip" onClick={() => openObj(d)}>{d}</span>)}</div></div> : null}
        <div style={{ marginTop: 6 }}>
          <DataTable rows={lf} onRowClick={(fo: any) => openFile(fo.object_id)} empty="no local files yet"
            columns={[
              { header: "id", cell: (fo: any) => <code>{fo.object_id}</code> },
              { header: "title", cell: (fo: any) => fo.title || "" },
              { header: "type", cell: (fo: any) => fo.file_type || "" },
              { header: "rev", cell: (fo: any) => fo.version || 1 },
              { header: "review", cell: (fo: any) => fo.review_status || "" },
            ]} />
        </div>
      </Section>

      <Section title={`Messages sent (${myMsgs.length})`}><MsgTable msgs={myMsgs} /></Section>
      <Section title={`Meetings (${myMeet.length})`}><MeetingTable meetings={myMeet} /></Section>
      <Section title={`Speech / Text events (${text.length})`}><TextTable rows={text} /></Section>

      <Section title="Memory" sub={`last reflection t${mem.last_reflection_tick ?? "—"} · ${mem.reflection_count || 0} reflections`}>
        <MemList title="Lessons learned" items={(mem.lessons_learned || []).slice(-4)} />
        <MemList title="Unresolved needs" items={(mem.unresolved_needs || []).slice(-5)} />
        <MemList title="Repeated blockers" items={(mem.repeated_blockers || []).slice(-4)} />
        <MemList title="Concerns" items={(mem.concerns || []).slice(-4)} />
        {(mem.open_wishes || []).length ? <div className="kv"><b>Open wishes</b> <div className="pillrow">{mem.open_wishes.map((w: string) => <span key={w} className="link-chip" onClick={() => openWish(w)}>{w}</span>)}</div></div> : null}
      </Section>

      <Section title={`Reflection log (${myRefl.length})`}>
        {myRefl.length ? myRefl.map((r: any) => (
          <div className="kv" key={r.reflection_id} style={{ cursor: "pointer" }} onClick={() => openReflection(r.reflection_id)}><b>t{r.tick}</b> {String(r.team_assessment || r.self_assessment || "").slice(0, 80)} <span className="muted">({(r.created_wish_ids || []).length} wishes)</span></div>
        )) : <div className="muted">none</div>}
      </Section>

      <div className="cards" style={{ gridTemplateColumns: "1fr 1fr", marginTop: 14 }}>
        <Section title="Work State"><KV obj={ws} /></Section>
        <Section title="Org State"><KV obj={os} /></Section>
        <Section title="Compensation"><KV obj={a.compensation || {}} /></Section>
        <Section title="Routine"><KV obj={a.routine_profile || {}} /></Section>
      </div>

      <Section title="Commitments / Disputes / Requests">
        {myCommit.map((c: any) => <div className="kv" key={c.commitment_id} style={{ cursor: "pointer" }} onClick={() => openObj(c.commitment_id)}><b>commit</b> {String(c.description || "").slice(0, 40)} <Tag tone={c.status === "violated" ? "bad" : c.status === "fulfilled" ? "good" : ""}>{c.status}</Tag></div>)}
        {myDisp.map((c: any) => <div className="kv" key={c.dispute_id} style={{ cursor: "pointer" }} onClick={() => openObj(c.dispute_id)}><b>dispute</b> → {c.target_object_id} <Tag tone="bad">{c.status}</Tag></div>)}
        {myReq.map((c: any) => <div className="kv" key={c.request_id} style={{ cursor: "pointer" }} onClick={() => openObj(c.request_id)}><b>request</b> {c.action_requested} → {c.target_object_id || ""}</div>)}
        {!myCommit.length && !myDisp.length && !myReq.length ? <div className="muted">none</div> : null}
      </Section>

      <Section title={`Activity Log (${acts.length})`}>
        {acts.length ? acts.map((x: any, i: number) => <div className="kv" key={i}><b>t{x.tick}</b> {x.action_type}{x.target ? <span className="muted"> [{x.target}]</span> : null} <Tag tone={x.success ? "good" : "bad"}>{x.success ? "ok" : "fail"}</Tag></div>) : <div className="muted">none</div>}
      </Section>

      <Section title="Policy Trace" sub="top-k utility + masks (how the policy chose, not the LLM)">
        {trace.length ? trace.map((pt: any, i: number) => (
          <div className="kv" key={i} style={{ paddingBottom: 4 }}>
            <b>t{pt.tick}</b> → chose <Tag tone="good">{pt.chosen_action}</Tag>
            {(pt.candidate_actions || []).map((c: any, j: number) => (
              <div className="muted" key={j} style={{ fontSize: 11 }}>{c.action}{c.target ? ` [${c.target}]` : ""}: <b>{c.final_score}</b>{c.raw_score != null && c.raw_score !== c.final_score ? <span style={{ color: "#c66" }}> (raw {c.raw_score})</span> : null}</div>
            ))}
            {(pt.masked_actions || []).length ? <div style={{ fontSize: 11, color: "#c66" }}>masked: {(pt.masked_actions || []).map((m: any) => m.action + (m.reason ? ` (${m.reason})` : "")).join(", ")}</div> : null}
          </div>
        )) as ReactNode : <div className="muted">no policy trace yet</div>}
      </Section>
    </>
  );
}
