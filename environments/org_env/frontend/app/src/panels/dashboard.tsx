import { useInspector } from "../ctx";
import { Bar, Cards, Section, StatCard, Tag, num } from "../ui";
import type { PanelProps } from "./common";

export function AgentGrid({ f }: PanelProps) {
  const { selectAgent } = useInspector();
  const ags = Object.values(f.agents || {}) as any[];
  return (
    <div className="agrid">
      {ags.map((a) => {
        const ws = a.work_state || {}, os = a.org_state || {};
        const busy = ws.next_available_tick > f.tick;
        return (
          <div className="acard" key={a.id} onClick={() => selectAgent(a.id)}>
            <div className="nm">{a.name} <span className="cn">· {a.codename}</span></div>
            <div className="cn">{a.role} {a.is_founder ? "· founder" : ""} · {a.current_status}</div>
            <div className="mini"><span>activity</span><span>{ws.current_activity_type || "idle"}</span></div>
            <div className="mini"><span>attention</span><span>{num(ws.attention_remaining_today, 2)}</span></div>
            <Bar value={ws.attention_remaining_today} color="#34d399" />
            <div className="mini"><span>fatigue</span><span>{num(ws.fatigue, 2)}</span></div>
            <Bar value={ws.fatigue} color="#fbbf24" />
            <div className="mini"><span>stress / burnout</span><span>{num(ws.stress, 2)} / {num(ws.burnout_risk, 2)}</span></div>
            <div className="mini"><span>trust / comp-stress</span><span>{num(os.trust_in_company, 2)} / {num(os.compensation_stress, 2)}</span></div>
            <div className="mini"><span>retention risk</span><span>{num(os.retention_risk, 2)}</span></div>
            <div className="pillrow" style={{ marginTop: 4 }}>
              <Tag>tasks {a.assigned_tasks_count}</Tag>
              <Tag>unread {a.unread_messages_count}</Tag>
              <Tag>commit {a.open_commitments_count}</Tag>
              <Tag>jobs {a.sandbox_jobs_count}</Tag>
              <Tag tone={busy ? "warn" : "good"}>{busy ? "busy→" + ws.next_available_tick : "free"}</Tag>
            </div>
          </div>
        );
      })}
    </div>
  );
}

export function Dashboard({ f }: PanelProps) {
  const { setSub } = useInspector();
  const c = f.company || {};
  const m = c.org_metrics || {};
  return (
    <>
      <Cards>
        <StatCard label="tasks done" value={`${c.tasks_done}/${c.task_count}`} onClick={() => setSub("tasks")} />
        <StatCard label="cash" value={num(c.cash_balance)} onClick={() => setSub("budget")} />
        <StatCard label="runway days" value={num(c.runway_days)} onClick={() => setSub("budget")} />
        <StatCard label="payroll" value={c.payroll_status} onClick={() => setSub("budget")} />
        <StatCard label="demo ready" value={c.demo_ready ? "yes" : "no"} />
        <StatCard label="critical blockers" value={c.critical_blockers} onClick={() => setSub("tasks")} />
        <StatCard label="open PRs" value={c.open_prs} onClick={() => setSub("repo")} />
        <StatCard label="running jobs" value={c.running_sandbox_jobs} onClick={() => setSub("sandbox")} />
        <StatCard label="untracked results" value={c.untracked_results} onClick={() => setSub("sandbox")} />
        <StatCard label="protocols" value={c.protocol_count} onClick={() => setSub("protocols")} />
        <StatCard label="meetings today" value={c.meetings_today} onClick={() => setSub("meetings")} />
        <StatCard label="overtime" value={c.overtime_count} />
        <StatCard label="burnout max" value={num(c.burnout_risk_max, 2)} />
        <StatCard label="budget pressure" value={num(c.budget_pressure, 2)} />
        <StatCard label="weak protocols" value={(c.weak_protocols || []).length} onClick={() => setSub("protocols")} />
      </Cards>
      <Section title="Org Metrics" sub="token economy + org health">
        <Cards>
          <StatCard label="treasury (tok)" value={num(c.treasury_tokens)} />
          <StatCard label="daily burn (tok)" value={num(c.daily_burn_tokens, 1)} />
          <StatCard label="runway (days)" value={num(c.runway_days_tokens)} />
          <StatCard label="runway pressure" value={num(c.runway_pressure, 2)} />
          <StatCard label="token burn total" value={num(c.token_debits_total)} />
          <StatCard label="burn / merged PR" value={num(m.budget_burn_per_merged_pr, 1)} />
          <StatCard label="task cycle (t)" value={num(m.task_cycle_time_avg, 1)} />
          <StatCard label="PR review lat" value={num(m.pr_review_latency_avg, 1)} />
          <StatCard label="blocker dur" value={num(m.release_blocker_duration_avg, 1)} />
          <StatCard label="gap resolve %" value={num((m.gap_resolution_rate || 0) * 100)} />
          <StatCard label="rework rate" value={num(m.rework_rate, 2)} />
          <StatCard label="msg linked %" value={num((m.message_linked_ratio || 0) * 100)} />
          <StatCard label="proto use/enf" value={`${m.protocol_use_count || 0}/${m.protocol_enforcement_count || 0}`} />
          <StatCard label="go-to count" value={m.go_to_count || 0} />
          <StatCard label="releases" value={m.release_count || 0} />
        </Cards>
      </Section>
      <Section title="Agents"><AgentGrid f={f} /></Section>
    </>
  );
}
