import { useEffect, useRef } from "react";
import hljs from "highlight.js";
import type { Frame, DrawerDesc, GNode, Graph, ReaderDoc } from "./types";
import { EXT_LANG } from "./theme";
import { KV, Section, Tag, StatusTag } from "./ui";
import { useInspector } from "./ctx";

// ----------------------------- object bank lookup --------------------------
function findObject(f: Frame, id: string): { type: string; obj: any } | null {
  const I = f.internal || {}, X = f.external || {}, R = I.repo || {};
  const banks: [string, any[], string][] = [
    ["task", I.tasks, "task_id"], ["doc", I.docs, "doc_id"], ["file", I.files, "object_id"],
    ["message", I.messages, "message_id"], ["meeting", I.meetings, "meeting_id"],
    ["pr", R.pull_requests, "pr_id"], ["commit", R.commits, "commit_id"], ["branch", R.branches, "branch_id"],
    ["result", I.results, "result_id"], ["experiment", I.experiments, "experiment_id"], ["protocol", I.protocols, "protocol_id"],
    ["issue", I.issues, "issue_id"], ["commitment", I.commitments, "commitment_id"], ["dispute", I.disputes, "dispute_id"],
    ["request", I.requests, "request_id"], ["sandbox_job", (I.sandbox || {}).jobs, "job_id"],
    ["cost_event", I.cost_events, "cost_event_id"], ["ticket", I.tickets, "ticket_id"],
    ["external_profile", X.profiles, "external_agent_id"], ["external_post", X.posts, "post_id"],
  ];
  for (const [type, arr, key] of banks) {
    const o = (arr || []).find((x: any) => x[key] === id);
    if (o) return { type, obj: o };
  }
  return null;
}

function findFile(f: Frame, id: string): any | null {
  const I = f.internal || {};
  const fileMatch = (fo: any) => fo.object_id === id || fo.id === id || fo.file_id === id;
  for (const fo of I.files || []) if (fileMatch(fo)) return fo;
  for (const ag of Object.values(f.agents || {})) for (const fo of ((ag as any).workspace || {}).local_files || []) if (fileMatch(fo)) return fo;
  for (const fo of Object.values((f.company || {}).files || {})) if (fileMatch(fo)) return fo;
  for (const d of I.docs || []) if (d.doc_id === id || d.id === id) return d;
  return null;
}

function findArtifact(f: Frame, id: string): any | null {
  const prod = f.product || {};
  return (prod.artifacts || []).find((x: any) => x.artifact_id === id)
    || (prod.repo_files || []).find((x: any) => x.artifact_id === id)
    || null;
}

function LinkChip({ id, onClick }: { id: string; onClick: () => void }) {
  return <span className="link-chip" onClick={onClick}>{id}</span>;
}

const SPECIAL = new Set(["visible_to_agents", "read_by", "lifecycle", "internal_exposure", "provenance", "event_chain", "debug_visible"]);

// ------------------------------- per-kind content --------------------------
function ObjectBody({ id }: { id: string }) {
  const { frame, openObj } = useInspector();
  const r = findObject(frame, id);
  if (!r) return <><h3>Object <code>{id}</code></h3><div className="muted">not found in this frame</div></>;
  const o = r.obj;
  const edges = ((frame.graphs?.event?.edges) || []).filter((e: any) => e.src === id || e.dst === id);
  const fields = Object.fromEntries(Object.entries(o).filter(([k]) => !SPECIAL.has(k)));
  const agents = Object.keys(frame.agents || {});
  return (
    <>
      <h3>{r.type} · <code>{id}</code></h3>
      {o.visible_to_agents && (
        <Section title="Visibility">
          {agents.map((a) => (
            <div className="kv" key={a}><b>visible to {a}</b>: {(o.visible_to_agents || []).includes(a) ? <Tag tone="good">yes</Tag> : <Tag tone="bad">no</Tag>}</div>
          ))}
          {o.read_by && <div className="kv"><b>read_by</b>: {(o.read_by || []).join(", ") || "—"}</div>}
          {o.private_owner && <div className="kv"><b>private_owner</b>: {o.private_owner}</div>}
        </Section>
      )}
      {o.lifecycle && <Section title="Lifecycle"><KV obj={o.lifecycle} /></Section>}
      {o.internal_exposure && <Section title="Internal exposure"><KV obj={o.internal_exposure} /></Section>}
      {o.provenance && <Section title="Provenance"><KV obj={o.provenance} /></Section>}
      {o.event_chain && (
        <Section title="Event chain">
          {(o.event_chain || []).map((e: any, i: number) => <div className="kv" key={i}><b>t{e.tick}</b> {e.event_type} by {e.actor_id || ""}</div>)}
        </Section>
      )}
      <Section title="Fields"><KV obj={fields} /></Section>
      <Section title={`Linked event-graph edges (${edges.length})`}>
        {edges.length ? edges.map((e: any, i: number) => (
          <div className="kv" key={i}>{e.src} <i className="muted">{e.type}</i> → {e.dst}</div>
        )) : <div className="muted">none</div>}
      </Section>
    </>
  );
}

function fileContent(fo: any): { text: string } {
  const rp = fo.raw_payload != null ? fo.raw_payload : (fo.body != null ? fo.body : fo.content);
  const text = typeof rp === "string" ? rp : (rp != null ? JSON.stringify(rp, null, 2) : (fo.content_summary || fo.summary || ""));
  return { text };
}

function FileView({ id, fo }: { id: string; fo: any }) {
  const { openReader } = useInspector();
  const meta = { owner: fo.owner_id || fo.creator_id, type: fo.file_type || fo.doc_type, visibility: fo.visibility, version: fo.version, review_status: fo.review_status || fo.status, trust_level: fo.trust_level, created_tick: fo.created_tick, last_modified_tick: fo.last_modified_tick };
  const { text } = fileContent(fo);
  const doc: ReaderDoc = { title: fo.title || id, meta: `${fo.file_type || fo.doc_type || "file"} · v${fo.version || 1}`, text };
  return (
    <>
      <h3>file · <code>{id}</code></h3>
      <h2>{fo.title || ""}</h2>
      {text ? <button className="btn-accent" style={{ marginBottom: 8 }} onClick={() => openReader(doc)}>▶ View file</button> : null}
      <Section title="Meta"><KV obj={meta} /></Section>
      {(fo.content_summary || fo.summary || fo.body_summary) && <Section title="Summary"><div className="kv">{fo.content_summary || fo.summary || fo.body_summary}</div></Section>}
      {text ? <Section title="Content"><pre style={{ whiteSpace: "pre-wrap", fontSize: 11, maxHeight: 440, overflow: "auto", background: "var(--panel2)", padding: 8, borderRadius: 8 }}>{text}</pre></Section> : <div className="muted">(no stored content for this file)</div>}
      {fo.visible_to_agents && <Section title="Visible to"><div className="pillrow">{(fo.visible_to_agents || []).map((x: string) => <Tag key={x}>{x}</Tag>) }</div></Section>}
    </>
  );
}

function FileBody({ id }: { id: string }) {
  const { frame } = useInspector();
  const fo = findFile(frame, id);
  if (fo) return <FileView id={id} fo={fo} />;
  const art = findArtifact(frame, id);
  if (art) return <ArtifactView id={id} a={art} />;
  const obj = findObject(frame, id);
  if (obj) return <><h3>{obj.type} · <code>{id}</code></h3><Section title="Fields"><KV obj={obj.obj} /></Section></>;
  return <><h3>file · <code>{id}</code></h3><div className="muted">not found in this frame (may be a draft id not yet materialised)</div></>;
}

function artifactText(a: any, patches: any[]): string {
  // real file content (grows as patches apply) takes precedence over the synthesized view
  if ((a.content || "").trim()) return a.content;
  const L: string[] = [];
  L.push(`# ${a.linked_file_path || a.title || a.artifact_id}`);
  L.push(`# status=${a.status} · revision=${a.revision} · mainline=${a.mainline_revision == null ? "-" : a.mainline_revision}`);
  L.push("");
  if ((a.capabilities || []).length) { L.push('"""'); L.push("Capabilities (what this file can do):"); a.capabilities.forEach((c: string) => L.push("  - " + c)); L.push('"""'); L.push(""); }
  const applied = (patches || []).filter((p) => (p.status || p.validation_status) === "accepted");
  if (applied.length) {
    L.push("# ===== built from " + applied.length + " applied change(s), oldest first ====="); L.push("");
    applied.forEach((p) => {
      L.push(`# --- t${p.created_tick != null ? p.created_tick : p.tick} · ${p.patch_type || "patch"} · ${p.change_summary || p.edit_goal || ""}`);
      if (p.pseudo_diff) L.push(p.pseudo_diff);
      (p.changed_sections || []).forEach((s: any) => { L.push("[" + (s.section || "section") + "]"); if (s.after) L.push(s.after); });
      (p.added_fields || []).forEach((x: string) => L.push("+ field: " + x));
      (p.added_checks || []).forEach((x: string) => L.push("+ check: " + x));
      (p.changed_behavior || []).forEach((x: string) => L.push("~ " + x));
      L.push("");
    });
  } else { L.push("# (no applied patches yet — file is a stub / symbolic placeholder)"); L.push(""); }
  if ((a.known_gaps || []).length) { L.push("# TODO — known gaps:"); a.known_gaps.forEach((g: string) => L.push("#   - " + g)); }
  return L.join("\n");
}

function ArtifactView({ id, a }: { id: string; a: any }) {
  const { frame, openReader } = useInspector();
  const prod = frame.product || {};
  const patches = (prod.patches || []).filter((p: any) => (p.artifact_id || p.target_object_id) === id);
  const meta = { path: a.linked_file_path || a.title, type: a.artifact_type, status: a.status, revision: a.revision, mainline_revision: a.mainline_revision, awaiting_review: a.awaiting_review, last_reviewed_tick: a.last_reviewed_tick, linked_tasks: (a.linked_task_ids || []).join(", ") || "—" };
  const doc: ReaderDoc = { title: a.linked_file_path || a.title || id, meta: `${a.artifact_type || ""} · rev ${a.revision} · status ${a.status}`, text: artifactText(a, patches) };
  return (
    <>
      <h3>artifact · <code>{id}</code></h3>
      <h2>{a.title || a.linked_file_path || ""}</h2>
      <button className="btn-accent" style={{ marginBottom: 8 }} onClick={() => openReader(doc)}>▶ View file</button>
      <Section title="Meta"><KV obj={meta} /></Section>
      {(a.capabilities || []).length ? <Section title={`Capabilities (${a.capabilities.length})`}><div className="pillrow">{a.capabilities.map((c: string) => <Tag key={c} tone="good">{c}</Tag>)}</div></Section> : null}
      {(a.known_gaps || []).length ? <Section title={`Known gaps (${a.known_gaps.length})`}>{a.known_gaps.map((g: string, i: number) => <div className="kv muted" key={i}>• {g}</div>)}</Section> : null}
      {(a.change_summaries || []).length ? <Section title={`Change history (${a.change_summaries.length})`}>{a.change_summaries.map((s: string, i: number) => <div className="kv" key={i}><b>v{i + 1}</b> {s}</div>)}</Section> : null}
      <Section title={`Patches (${patches.length}) — what changed inside`}>
        {patches.length ? patches.map((p: any, i: number) => {
          const body = p.unified_diff || p.pseudo_diff || ((p.changed_sections || []).map((s: any) => `[${s.section}] ${s.before || ""} → ${s.after || ""}`).join("\n")) || "";
          const st = p.status || p.validation_status || "";
          return (
            <div className="kv" key={i} style={{ paddingBottom: 5 }}>
              <b>t{p.created_tick != null ? p.created_tick : p.tick} · {p.patch_type || ""}</b> <Tag tone={st === "accepted" ? "good" : "bad"}>{st}</Tag>
              <div className="muted" style={{ fontSize: 11 }}>{p.change_summary || p.edit_goal || ""}</div>
              {body ? <pre style={{ whiteSpace: "pre-wrap", fontSize: 11, background: "var(--panel2)", padding: 6, borderRadius: 6, maxHeight: 220, overflow: "auto" }}>{body}</pre> : null}
            </div>
          );
        }) : <div className="muted">no patches recorded for this artifact</div>}
      </Section>
    </>
  );
}

function ArtifactBody({ id }: { id: string }) {
  const { frame } = useInspector();
  const art = findArtifact(frame, id);
  if (art) return <ArtifactView id={id} a={art} />;
  const fo = findFile(frame, id);
  if (fo) return <FileView id={id} fo={fo} />;
  return <><h3>artifact · <code>{id}</code></h3><div className="muted">not found in this frame</div></>;
}

function PersonaNodeBody({ node, persona }: { node: GNode; persona: Graph }) {
  const g = persona || { nodes: [], edges: [] };
  const outs = g.edges.filter((e) => e.src === node.id);
  const ins = g.edges.filter((e) => e.dst === node.id);
  const pick = (arr: typeof outs, re: RegExp) => arr.filter((e) => re.test(e.dst)).map((e) => e.dst.split(":")[1]);
  const acts = pick(outs, /^act:/), speech = pick(outs, /^speech:/), risks = pick(outs, /^risk:/), feats = pick(outs, /^feat:/);
  const drivenBy = ins.filter((e) => /^(trait:|skill:|state:|ev:)/.test(e.src));
  const att = node.attrs || {};
  const edgeRow = (e: any, i: number) => (
    <div className="kv" key={i}>{e.src.split(":").pop()} <i className="muted">{e.type}{e.weight != null && e.weight !== 1 ? ` (${e.weight > 0 ? "+" : ""}${e.weight})` : ""}{e.strength != null ? ` · str ${e.strength}` : ""}</i> → {e.dst.split(":").pop()}{(e.reason || e.explanation) ? <div className="muted" style={{ fontSize: 11 }}>{e.reason || e.explanation}</div> : null}</div>
  );
  return (
    <>
      <h3>{node.type} · <code>{node.label || node.id}</code>{att.suppressed ? <Tag tone="bad">suppressed</Tag> : null}</h3>
      <Section title="Node">
        {node.cluster != null && <div className="kv"><b>cluster</b>: {String(node.cluster)}</div>}
        {node.value != null && <div className="kv"><b>value</b>: {String(node.value)}</div>}
        {node.salience != null && <div className="kv"><b>salience</b>: {String(node.salience)}</div>}
        {att.skill_fit != null && <div className="kv"><b>skill_fit</b>: {String(att.skill_fit)}</div>}
        {att.event_type && <div className="kv"><b>event</b>: {att.event_type}{att.target_object ? ` → ${att.target_object}` : ""}</div>}
      </Section>
      {acts.length ? <Section title="Affected actions"><div className="pillrow">{acts.map((a, i) => <Tag key={i} tone="good">{a}</Tag>)}</div></Section> : null}
      {speech.length ? <Section title="Affected speech acts"><div className="pillrow">{speech.map((a, i) => <Tag key={i} color="#d2a8ff">{a}</Tag>)}</div></Section> : null}
      {risks.length ? <Section title="Risks"><div className="pillrow">{risks.map((a, i) => <Tag key={i} tone="bad">{a}</Tag>)}</div></Section> : null}
      {feats.length ? <Section title="Policy features"><div className="pillrow">{feats.map((a, i) => <Tag key={i} tone="viz">{a}</Tag>)}</div></Section> : null}
      {drivenBy.length ? <Section title={`Driven by (${drivenBy.length})`}>{drivenBy.slice(0, 12).map(edgeRow)}</Section> : null}
      {outs.length ? <Section title={`Outgoing (${outs.length})`}>{outs.slice(0, 14).map(edgeRow)}</Section> : null}
    </>
  );
}

function EpisodeBody({ id }: { id: string }) {
  const { frame, openObj } = useInspector();
  const e = ((frame.episodes || {}).items || []).find((x: any) => x.episode_id === id);
  if (!e) return <><h3>Episode <code>{id}</code></h3><div className="muted">not found</div></>;
  const epName = (a: string) => (frame.agents[a] || {}).name || a;
  const objSec = (t: string, arr: string[]) => (arr && arr.length) ? <Section title={`${t} (${arr.length})`}><div className="pillrow">{arr.map((o) => <LinkChip key={o} id={o} onClick={() => openObj(o)} />)}</div></Section> : null;
  return (
    <>
      <h3>{e.title || e.episode_type} <StatusTag value={e.status} /></h3>
      <Section title="Episode">
        <div className="kv"><b>type</b>: {e.episode_type}</div>
        <div className="kv"><b>ticks</b>: {e.start_tick} → {e.end_tick != null ? e.end_tick : "(open)"}</div>
        <div className="kv"><b>primary</b>: {epName(e.primary_agent_id)}</div>
        {e.problem_statement && <div className="kv"><b>problem</b>: {e.problem_statement}</div>}
        {e.outcome_summary && <div className="kv"><b>outcome</b>: {e.outcome_summary}</div>}
        {(e.participants || []).length ? <div className="kv"><b>participants</b>: <div className="pillrow">{(e.participants || []).map((a: string) => <Tag key={a}>{epName(a)}</Tag>)}</div></div> : null}
      </Section>
      {objSec("Linked objects", e.linked_object_ids)}
      {(e.event_log || []).length ? <Section title={`Event log (${e.event_log.length})`}>{(e.event_log || []).slice(-20).map((ev: any, i: number) => <div className="kv" key={i}><b>t{ev.tick}</b> {ev.event_type} {ev.actor_id ? <span className="muted">by {ev.actor_id}</span> : null}</div>)}</Section> : null}
    </>
  );
}

function ReflectionBody({ id }: { id: string }) {
  const { frame, openWish } = useInspector();
  const r = ((frame.reflections || {}).items || []).find((x: any) => x.reflection_id === id);
  if (!r) return <><h3>Reflection <code>{id}</code></h3><div className="muted">not found</div></>;
  return (
    <>
      <h3>reflection · <code>{id}</code></h3>
      <Section title="Reflection">
        <div className="kv"><b>tick</b>: {r.tick}</div>
        <div className="kv"><b>agent</b>: {r.agent_id}</div>
        {r.team_assessment && <div className="kv"><b>team</b>: {r.team_assessment}</div>}
        {r.self_assessment && <div className="kv"><b>self</b>: {r.self_assessment}</div>}
      </Section>
      {(r.ideas || []).length ? <Section title={`Ideas (${r.ideas.length})`}>{r.ideas.map((x: any, i: number) => <div className="kv muted" key={i}>• {typeof x === "string" ? x : JSON.stringify(x)}</div>)}</Section> : null}
      {(r.created_wish_ids || []).length ? <Section title="Wishes"><div className="pillrow">{r.created_wish_ids.map((w: string) => <LinkChip key={w} id={w} onClick={() => openWish(w)} />)}</div></Section> : null}
      {(r.source_episode_ids || []).length ? <Section title="Source episodes"><div className="pillrow">{r.source_episode_ids.map((x: string) => <Tag key={x}>{x}</Tag>)}</div></Section> : null}
    </>
  );
}

function WishBody({ id }: { id: string }) {
  const { frame } = useInspector();
  const w = ((frame.wishes || {}).items || []).find((x: any) => x.wish_id === id);
  if (!w) return <><h3>Wish <code>{id}</code></h3><div className="muted">not found</div></>;
  return (
    <>
      <h3>wish · <code>{id}</code></h3>
      <Section title="Wish">
        <div className="kv"><b>type</b>: {w.wish_type}</div>
        <div className="kv"><b>agent</b>: {w.agent_id}</div>
        <div className="kv"><b>urgency</b>: {String(w.urgency)}</div>
        {w.interpreted_need && <div className="kv"><b>need</b>: {w.interpreted_need}</div>}
        {w.raw_reflection_excerpt && <div className="kv"><b>excerpt</b>: {w.raw_reflection_excerpt}</div>}
        {w.source_reflection_id && <div className="kv"><b>source</b>: {w.source_reflection_id}</div>}
      </Section>
      {(w.grounded_object_ids || []).length ? <Section title="Grounded objects"><div className="pillrow">{w.grounded_object_ids.map((x: string) => <Tag key={x}>{x}</Tag>)}</div></Section> : null}
    </>
  );
}

function ProposalBody({ id }: { id: string }) {
  const { frame } = useInspector();
  const pr = ((frame.proposals || {}).items || frame.proposals || []).find?.((x: any) => x.proposal_id === id)
    || (((frame.proposals || {}).items) || []).find((x: any) => x.proposal_id === id);
  if (!pr) return <><h3>Proposal <code>{id}</code></h3><div className="muted">not found</div></>;
  return (
    <>
      <h3>proposal · <code>{id}</code> <StatusTag value={pr.status} /></h3>
      <Section title="Proposal"><KV obj={Object.fromEntries(Object.entries(pr).filter(([, v]) => typeof v !== "object"))} /></Section>
      {(pr.rationale || pr.description) && <Section title="Rationale"><div className="kv">{pr.rationale || pr.description}</div></Section>}
    </>
  );
}

function KVBody({ title, obj }: { title: string; obj: Record<string, any> }) {
  return <><h3>{title}</h3><Section title="Fields"><KV obj={obj} /></Section></>;
}

// ------------------------------- Drawer shell ------------------------------
export function Drawer({ desc, onClose }: { desc: DrawerDesc; onClose: () => void }) {
  let body: React.ReactNode = null;
  if (desc) {
    switch (desc.kind) {
      case "object": body = <ObjectBody id={desc.id} />; break;
      case "file": body = <FileBody id={desc.id} />; break;
      case "artifact": body = <ArtifactBody id={desc.id} />; break;
      case "episode": body = <EpisodeBody id={desc.id} />; break;
      case "reflection": body = <ReflectionBody id={desc.id} />; break;
      case "wish": body = <WishBody id={desc.id} />; break;
      case "proposal": body = <ProposalBody id={desc.id} />; break;
      case "personaNode": body = <PersonaNodeBody node={desc.node} persona={desc.persona} />; break;
      case "kv": body = <KVBody title={desc.title} obj={desc.obj} />; break;
    }
  }
  return (
    <div className={"drawer" + (desc ? " open" : "")}>
      <span className="x" onClick={onClose}>✕</span>
      <div>{body}</div>
    </div>
  );
}

// ------------------------------- Reader shell ------------------------------
export function Reader({ doc, onClose }: { doc: ReaderDoc; onClose: () => void }) {
  const codeRef = useRef<HTMLDivElement>(null);
  const text = doc?.text ?? "";
  const title = doc?.title ?? "file";
  let lang: string | null = null;
  const m = /\.([a-z0-9]+)$/i.exec(title);
  if (m && EXT_LANG[m[1].toLowerCase()] && hljs.getLanguage(EXT_LANG[m[1].toLowerCase()])) lang = EXT_LANG[m[1].toLowerCase()];
  let html: string | null = null;
  let shownLang = "text";
  if (text) {
    try {
      const res = lang ? hljs.highlight(text, { language: lang }) : hljs.highlightAuto(text);
      html = res.value; shownLang = (res as any).language || lang || "text";
    } catch { html = null; }
  }
  const lineCount = text.split("\n").length;
  useEffect(() => {
    if (codeRef.current && html != null) codeRef.current.innerHTML = `<code class="hljs">${html}</code>`;
  }, [html]);
  return (
    <div className={"reader" + (doc ? " open" : "")}>
      <div className="reader-bar">
        <span className="fname">{title}</span>
        {doc ? <span className="lang">{shownLang}</span> : null}
        <span className="muted">{doc?.meta}</span>
        <span className="spacer" />
        <button onClick={onClose}>✕ close</button>
      </div>
      <div className="reader-body">
        <div className="reader-gutter">{Array.from({ length: lineCount }, (_, i) => i + 1).join("\n")}</div>
        {html != null ? <div className="reader-code" ref={codeRef} /> : <div className="reader-code">{text || "(empty)"}</div>}
      </div>
    </div>
  );
}
