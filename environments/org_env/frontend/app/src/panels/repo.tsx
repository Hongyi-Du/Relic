import { useState } from "react";
import { useInspector } from "../ctx";
import { api } from "../api";
import { Cards, DataTable, KV, Section, StatCard, StatusTag, Tag, num } from "../ui";
import { useVisFilter, type PanelProps } from "./common";

function MaterializeBar() {
  const [exp, setExp] = useState<any | null>(null);
  const [busy, setBusy] = useState(false);
  const run = async () => { setBusy(true); try { setExp(await api.exportRepo()); } finally { setBusy(false); } };
  const s = exp?.smoke, e = exp?.export;
  return (
    <Section title="Materialized repo" sub="export the in-world files to a real directory + run smoke_check.py">
      <button className="btn-accent" disabled={busy} onClick={run}>{busy ? "running…" : "⤓ Export & smoke-test repo"}</button>
      {exp && (
        <div style={{ marginTop: 10 }}>
          <div className="kv"><b>dir</b>: <code>{e?.dir}</code> <span className="muted">({e?.file_count} files{e?.empty_files?.length ? `, ${e.empty_files.length} empty` : ""})</span></div>
          <div className="kv"><b>smoke</b>: {s?.ok ? <Tag tone="good">passed (rc {s?.returncode})</Tag> : <Tag tone="bad">failed</Tag>}{s?.error ? <span style={{ color: "var(--bad)" }}> {s.error}</span> : null}</div>
          {s?.metrics ? <div className="kv"><b>metrics</b>: <code>{JSON.stringify(s.metrics)}</code></div> : null}
          {s?.stderr_tail ? <pre style={{ whiteSpace: "pre-wrap", fontSize: 11, color: "var(--bad)", maxHeight: 160, overflow: "auto", background: "var(--panel2)", padding: 8, borderRadius: 8 }}>{s.stderr_tail}</pre> : null}
        </div>
      )}
    </Section>
  );
}

export function Repo({ f }: PanelProps) {
  const { openObj, openArtifact } = useInspector();
  const r = f.internal?.repo || {};
  const prod = f.product || {};
  const files = prod.repo_files || prod.artifacts || [];
  return (
    <>
      <MaterializeBar />
      <Cards>
        <StatCard label="files" value={files.length} />
        <StatCard label="modules" value={(r.modules || []).length} />
        <StatCard label="branches" value={(r.branches || []).length} />
        <StatCard label="commits" value={(r.commits || []).length} />
        <StatCard label="PRs" value={(r.pull_requests || []).length} />
        <StatCard label="build" value={r.build_status} />
        <StatCard label="tech debt" value={num(r.technical_debt, 2)} />
      </Cards>
      <Section title={`Repo files (${files.length})`} sub="click a file to read its contents (capabilities + applied patches)">
        <DataTable
          rows={files}
          onRowClick={(a: any) => openArtifact(a.artifact_id)}
          empty="no repo files yet"
          columns={[
            { header: "path", cell: (a: any) => <code>{a.linked_file_path || a.title}</code> },
            { header: "type", cell: (a: any) => a.artifact_type },
            { header: "status", cell: (a: any) => <StatusTag value={a.status} /> },
            { header: "rev", cell: (a: any) => a.revision },
            { header: "capabilities", cell: (a: any) => (a.capabilities || []).length },
            { header: "gaps", cell: (a: any) => (a.known_gaps || []).length },
          ]}
        />
        {(r.modules || []).length ? <div className="pillrow" style={{ marginTop: 8 }}>{(r.modules || []).map((m: string) => <Tag key={m}>{m}</Tag>)}</div> : null}
      </Section>
      <Section title="Pull Requests">
        <DataTable
          rows={r.pull_requests || []}
          onRowClick={(pr: any) => openObj(pr.pr_id)}
          empty="none"
          columns={[
            { header: "id", cell: (pr: any) => <code>{pr.pr_id}</code> },
            { header: "author", cell: (pr: any) => pr.author_id },
            { header: "status", cell: (pr: any) => <StatusTag value={String(pr.status)} /> },
            { header: "reviewed", cell: (pr: any) => (pr.reviewed ? <Tag tone="good">yes</Tag> : <Tag tone="bad">no</Tag>) },
            { header: "linked tasks", cell: (pr: any) => { const lt = pr.linked_task_ids || (pr.linked_task ? [pr.linked_task] : []); return lt.length ? lt.map((t: string) => <Tag key={t}>{t}</Tag>) : <Tag tone="bad">none</Tag>; } },
            { header: "linked issues", cell: (pr: any) => { const li = pr.linked_issue_ids || (pr.linked_issue ? [pr.linked_issue] : []); return li.length ? li.map((t: string) => <Tag key={t}>{t}</Tag>) : "—"; } },
          ]}
        />
      </Section>
      <Section title={`Commits (${(r.commits || []).length})`}>
        <DataTable
          rows={(r.commits || []).slice(-30).reverse()}
          empty="none"
          columns={[
            { header: "id", cell: (c: any) => <code>{c.commit_id}</code> },
            { header: "author", cell: (c: any) => c.author_id },
            { header: "branch", cell: (c: any) => c.branch_id },
            { header: "msg", cell: (c: any) => c.message },
            { header: "flags", cell: (c: any) => (c.quality_flags || []).map((x: string) => <Tag key={x} tone="warn">{x}</Tag>) },
          ]}
        />
      </Section>
    </>
  );
}

export function Sandbox({ f }: PanelProps) {
  const { openObj } = useInspector();
  const vis = useVisFilter();
  const jobs = f.internal?.sandbox?.jobs || [];
  const res = f.internal?.results || [];
  return (
    <>
      <Section title={`Sandbox Jobs (${jobs.length})`}>
        <DataTable
          rows={jobs.slice(-30).reverse()}
          empty="none"
          columns={[
            { header: "id", cell: (j: any) => <code>{j.job_id}</code> },
            { header: "agent", cell: (j: any) => j.agent_id },
            { header: "type", cell: (j: any) => j.job_type },
            { header: "status", cell: (j: any) => <StatusTag value={j.status} /> },
            { header: "result", cell: (j: any) => j.result_id || "—" },
          ]}
        />
      </Section>
      <Section title={`Results (${res.length}) — lifecycle`}>
        <DataTable
          rows={res.filter(vis)}
          onRowClick={(r: any) => openObj(r.result_id)}
          empty="none"
          columns={[
            { header: "id", cell: (r: any) => <code>{r.result_id}</code> },
            { header: "exp", cell: (r: any) => r.experiment_id },
            { header: "tracker", cell: (r: any) => (r.logged_to_tracker ? <Tag tone="good">logged</Tag> : <Tag tone="warn">private</Tag>) },
            { header: "repro", cell: (r: any) => r.reproducibility_status },
            { header: "config", cell: (r: any) => (r.config_hash ? <Tag tone="good">y</Tag> : <Tag tone="bad">n</Tag>) },
            { header: "disputed", cell: (r: any) => ((r.disputed_by || []).length ? <Tag tone="bad">yes</Tag> : "—") },
            { header: "seen by", cell: (r: any) => (r.visible_to_agents || []).length },
          ]}
        />
      </Section>
    </>
  );
}

export function Budget({ f }: PanelProps) {
  const { selectAgent } = useInspector();
  const b = f.internal?.budget || {}, pay = f.internal?.payroll || {}, c = f.company || {};
  const led = (c.token_ledger_tail || []).slice().reverse();
  return (
    <>
      <Cards>
        <StatCard label="treasury (tok)" value={num(c.treasury_tokens)} />
        <StatCard label="daily burn (tok)" value={num(c.daily_burn_tokens, 1)} />
        <StatCard label="runway (days)" value={num(c.runway_days_tokens)} />
        <StatCard label="runway pressure" value={num(c.runway_pressure, 2)} />
        <StatCard label="cost x" value={num(b.cost_multiplier || 1, 1)} />
        <StatCard label="payroll" value={pay.payroll_status} />
        <StatCard label="debits" value={num(c.token_debits_total)} />
        <StatCard label="credits" value={num(c.token_credits_total)} />
      </Cards>
      <Section title="Token ledger" sub={`${c.token_ledger_count || 0} entries (latest first)`}>
        <DataTable
          rows={led}
          empty="no token flows yet"
          columns={[
            { header: "tick", cell: (e: any) => e.tick },
            { header: "kind", cell: (e: any) => <Tag tone={e.kind === "credit" ? "good" : "bad"}>{e.kind}</Tag> },
            { header: "amount", cell: (e: any) => e.amount },
            { header: "reason", cell: (e: any) => e.reason },
            { header: "by", cell: (e: any) => e.agent },
            { header: "balance", cell: (e: any) => e.balance_after },
          ]}
        />
      </Section>
      <Section title="Compensation / Retention by agent">
        <DataTable
          rows={Object.values(f.agents || {}) as any[]}
          onRowClick={(a: any) => selectAgent(a.id)}
          empty="none"
          columns={[
            { header: "agent", cell: (a: any) => a.name },
            { header: "salary", cell: (a: any) => num((a.compensation || {}).base_salary) },
            { header: "unpaid", cell: (a: any) => num((a.compensation || {}).unpaid_salary) },
            { header: "trust", cell: (a: any) => num((a.compensation || {}).trust_in_company, 2) },
            { header: "outside", cell: (a: any) => num((a.compensation || {}).outside_options, 2) },
            { header: "retention risk", cell: (a: any) => { const rr = (a.compensation || {}).retention_risk || 0; return <Tag tone={rr > 0.6 ? "bad" : rr > 0.3 ? "warn" : "good"}>{num(rr, 2)}</Tag>; } },
          ]}
        />
      </Section>
      <Section title={`Cost events (${(f.internal?.cost_events || []).length})`}>
        <DataTable
          rows={(f.internal?.cost_events || []).slice(-25).reverse()}
          empty="none"
          columns={[
            { header: "t", cell: (c: any) => c.tick },
            { header: "agent", cell: (c: any) => c.agent_id },
            { header: "action", cell: (c: any) => c.action_type },
            { header: "cost", cell: (c: any) => num(c.final_cost, 1) },
          ]}
        />
      </Section>
    </>
  );
}

export function Protocols({ f }: PanelProps) {
  const { openObj } = useInspector();
  const ps = f.internal?.protocols || [];
  const det = f.internal?.detectors || {};
  return (
    <>
      <Section title={`Protocols (${ps.length})`}>
        <DataTable
          rows={ps}
          onRowClick={(pr: any) => openObj(pr.protocol_id)}
          empty="none"
          columns={[
            { header: "type", cell: (pr: any) => pr.protocol_type },
            { header: "status", cell: (pr: any) => <StatusTag value={pr.adoption_status} /> },
            { header: "emergence", cell: (pr: any) => <Tag tone={pr.emergence_level === "strong" ? "good" : pr.emergence_level === "weak" ? "warn" : ""}>{pr.emergence_level}</Tag> },
            { header: "supporters", cell: (pr: any) => (pr.supporters || []).length },
            { header: "uses", cell: (pr: any) => (pr.usage_events || []).length },
            { header: "violations", cell: (pr: any) => (pr.violation_events || []).length },
            { header: "persist", cell: (pr: any) => pr.persistence_ticks },
          ]}
        />
      </Section>
      <Section title="Detector Summary">
        {Object.entries(det).length ? Object.entries(det).map(([k, v]: [string, any]) => (
          <div className="kv" key={k}>
            <b>{k}</b>: {v.detected ? <Tag tone="good">detected</Tag> : <Tag>no</Tag>} level=
            <Tag tone={v.level === "strong" ? "good" : v.level === "weak" ? "warn" : ""}>{v.level}</Tag>{" "}
            <span className="muted">{JSON.stringify(v.evidence || {}).slice(0, 80)}</span>
          </div>
        )) : <div className="empty">no detector output yet (runs at day boundary)</div>}
      </Section>
    </>
  );
}
