// The three panes: what is on my plate, what the organization is saying, and my
// own working agent.

import { SeatView, clockTime } from "./api";
import { CollapsibleGroup } from "./CollapsibleGroup";
import { OrganizationOverview } from "./OrganizationOverview";

type Open = (objectId: string, object?: any) => void;

const hours = (h: number | null | undefined) =>
  h == null ? "" : h < 1 ? "just now" : h < 24 ? `${h}h ago` : `${Math.round(h / 24)}d ago`;

// Get urgency badge for an item
const getUrgencyBadge = (item: any): string => {
  if (item.age_hours > 24 && item.unread) return "🔥";
  if (item.mentions_me || item.urgency === "high") return "⚠️";
  if (item.status === "blocked") return "🚫";
  return "";
};

// Get urgency tone class
const getUrgencyTone = (item: any): "normal" | "warn" | "critical" => {
  if (item.age_hours > 24 && item.unread) return "critical";
  if (item.mentions_me || item.urgency === "high" || item.status === "blocked") return "warn";
  return "normal";
};

// --------------------------------------------------------------------------
// Left: this member's standing in the organization
// --------------------------------------------------------------------------
export function MemberPanel({ view, open }: { view: SeatView; open: Open }) {
  const { seat, member } = view;
  const waiting = member.awaiting_me;

  // Categorize tasks by status
  const blockedTasks = member.my_tasks.filter((t: any) => t.status === "blocked");
  const activeTasks = member.my_tasks.filter((t: any) => t.status === "in_progress");
  const openTasks = member.my_tasks.filter((t: any) => t.status === "open");
  const otherTasks = member.my_tasks.filter(
    (t: any) => !["blocked", "in_progress", "open"].includes(t.status)
  );

  // Count urgent items
  const urgentCount =
    waiting.mentions.length +
    waiting.reviews.length +
    waiting.proposals.filter((p: any) => p.age_hours > 12).length +
    blockedTasks.length;

  return (
    <div className="pane member-pane">
      <OrganizationOverview view={view} open={open} />
      <section className="who">
        <div className="name">{seat.name}</div>
        <div className="role">{seat.role.replace(/_/g, " ")}{seat.is_founder ? " · founder" : ""}</div>
        <div className="identity">{seat.identity}</div>
      </section>

      {urgentCount > 0 && (
        <CollapsibleGroup title="🔥 Needs attention" count={urgentCount} defaultOpen={true} urgency="critical">
          {waiting.mentions.map((m: any) => (
            <Row key={m.id} onClick={() => open(m.id, m)}
                 title={`${getUrgencyBadge(m)} @ ${m.sender}`} sub={m.text}
                 when={hours(m.age_hours)} tone={getUrgencyTone(m)} />
          ))}
          {waiting.reviews.map((p: any) => (
            <Row key={p.id} onClick={() => open(p.id, p)}
                 title={`${getUrgencyBadge(p)} Review ${p.id}`} sub={`by ${p.author}`}
                 when={hours(p.age_hours)} tone={getUrgencyTone(p)} />
          ))}
          {waiting.proposals.filter((p: any) => p.age_hours > 12).map((p: any) => (
            <Row key={p.id} onClick={() => open(p.id, p)}
                 title={`${getUrgencyBadge(p)} Decide: ${p.title || p.id}`}
                 sub={p.proposal_type} when={hours(p.age_hours)} tone={getUrgencyTone(p)} />
          ))}
          {blockedTasks.map((t: any) => (
            <Row key={t.id} onClick={() => open(t.id, t)}
                 title={`🚫 ${t.title}`} sub="blocked"
                 when={hours(t.age_hours)} tone="warn" />
          ))}
        </CollapsibleGroup>
      )}

      {(waiting.proposals.filter((p: any) => p.age_hours <= 12).length > 0 ||
        waiting.meetings.length > 0) && (
        <CollapsibleGroup title="⏰ Pending"
                         count={waiting.proposals.filter((p: any) => p.age_hours <= 12).length + waiting.meetings.length}
                         defaultOpen={false} urgency="warn">
          {waiting.proposals.filter((p: any) => p.age_hours <= 12).map((p: any) => (
            <Row key={p.id} onClick={() => open(p.id, p)}
                 title={`Decide: ${p.title || p.id}`} sub={p.proposal_type}
                 when={hours(p.age_hours)} tone="normal" />
          ))}
          {waiting.meetings.map((m: any) => (
            <Row key={m.id} onClick={() => open(m.id, m)}
                 title={m.title || m.meeting_type} sub="meeting" when={m.status} tone="normal" />
          ))}
        </CollapsibleGroup>
      )}

      {activeTasks.length > 0 && (
        <CollapsibleGroup title="🏃 In progress" count={activeTasks.length} defaultOpen={true}>
          {activeTasks.map((t: any) => (
            <Row key={t.id} onClick={() => open(t.id, t)} title={t.title}
                 sub={t.status.replace(/_/g, " ")} when={hours(t.age_hours)} tone="normal" />
          ))}
        </CollapsibleGroup>
      )}

      {openTasks.length > 0 && (
        <CollapsibleGroup title="📋 Open tasks" count={openTasks.length} defaultOpen={false}>
          {openTasks.map((t: any) => (
            <Row key={t.id} onClick={() => open(t.id, t)} title={t.title}
                 sub={t.status.replace(/_/g, " ")} when={hours(t.age_hours)} tone="normal" />
          ))}
        </CollapsibleGroup>
      )}

      {otherTasks.length > 0 && (
        <CollapsibleGroup title="📦 Other tasks" count={otherTasks.length} defaultOpen={false}>
          {otherTasks.map((t: any) => (
            <Row key={t.id} onClick={() => open(t.id, t)} title={t.title}
                 sub={t.status.replace(/_/g, " ")} when={hours(t.age_hours)} tone="normal" />
          ))}
        </CollapsibleGroup>
      )}

      {member.my_work.branches.length > 0 && (
        <CollapsibleGroup title="🌿 My branches" count={member.my_work.branches.length} defaultOpen={false}>
          {member.my_work.branches.map((b: any) => (
            <Row key={b.id} onClick={() => open(b.id, b)} title={b.id}
                 sub={`${b.commits} commits${b.uncommitted ? " · uncommitted" : ""}`}
                 when={b.status} tone="normal" />
          ))}
        </CollapsibleGroup>
      )}

      {member.my_work.experiments.length > 0 && (
        <CollapsibleGroup title="🔬 My experiments" count={member.my_work.experiments.length} defaultOpen={false}>
          {member.my_work.experiments.map((e: any) => (
            <Row key={e.id} onClick={() => open(e.id, e)} title={e.title || e.id}
                 sub={e.status} when={hours(e.age_hours)} tone="normal" />
          ))}
        </CollapsibleGroup>
      )}

      <CollapsibleGroup title="🏢 Company" defaultOpen={true}>
        <div className="kv"><span className="k">runway</span>
          <span className="v">{member.company.runway_days} days</span></div>
        <div className="kv"><span className="k">pressure</span>
          <span className="v">{member.company.budget_pressure}</span></div>
        {member.company.product_stage && (
          <div className="kv"><span className="k">stage</span>
            <span className="v">{member.company.product_stage}</span></div>
        )}
      </CollapsibleGroup>

      <CollapsibleGroup title="👥 Team" count={member.members.length} defaultOpen={false}>
        {member.members.map((m: any) => (
          <div className="teammate" key={m.agent_id}>
            <span className={`dot ${m.online ? "on" : "off"}`} />
            <span className="tm-name">{m.name}</span>
            <span className="tm-role">{m.role.replace(/_/g, " ")}</span>
          </div>
        ))}
      </CollapsibleGroup>
    </div>
  );
}

function Row({ title, sub, when, tone, onClick }: any) {
  return (
    <button className={`row ${tone || ""}`} onClick={onClick}>
      <span className="row-title">{title}</span>
      <span className="row-sub">{sub}</span>
      <span className="row-when">{when}</span>
    </button>
  );
}

export { clockTime };
