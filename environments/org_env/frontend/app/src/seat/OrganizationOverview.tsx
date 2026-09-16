import { OrganizationBrief, SeatView } from "./api";

type Open = (objectId: string, object?: any) => void;

const asText = (value: any, fallback = "") => {
  if (value == null) return fallback;
  if (typeof value === "string") return value;
  return String(value);
};

const labelFor = (item: any, fallback: string) =>
  asText(item?.title || item?.name || item?.label || item?.what || item?.id, fallback);

const objectIdFor = (item: any) => item?.id || item?.refs?.[0]?.id || "";

const statusFor = (item: any) =>
  asText(item?.status || item?.state || item?.phase, "active").replace(/_/g, " ");

/**
 * Normalize the brief without making the UI depend on one backend snapshot
 * spelling.  The brief is public/permission-filtered state; the fallbacks are
 * also public seat state and never reach into an agent's private workspace.
 */
export function organizationData(view: SeatView) {
  const brief: OrganizationBrief = view.organization_brief || {};
  const visibleTasks = (view.objects?.tasks || []).filter((task: any) =>
    ["open", "in_progress", "blocked"].includes(task?.status),
  );
  const happening = brief.what_is_happening || [];
  const candidateWorkstreams = [brief.workstreams, brief.active_workstreams, happening, visibleTasks]
    .find((items: any) => Array.isArray(items) && items.length > 0) || [];
  const workstreams = candidateWorkstreams.slice(0, 6);
  const agents = (brief.agents || view.member?.members || []).slice(0, 10);
  const candidateBlockers = [
    brief.blockers,
    happening.filter((item: any) => ["blocked", "changes_requested", "failed"].includes(item?.status)),
    visibleTasks.filter((task: any) => task?.status === "blocked"),
  ].find((items: any) => Array.isArray(items) && items.length > 0) || [];
  const blockers = candidateBlockers.slice(0, 5);
  const pending = [brief.pending_decisions, brief.decisions, brief.needs_your_decision]
    .find((items: any) => Array.isArray(items) && items.length > 0)?.slice(0, 4) || [];
  return { brief, workstreams, agents, blockers, pending };
}

/**
 * The first thing a member sees after joining.  Keep this as a standalone
 * component so P3 can reuse the same summary while hiding the transparent
 * organization surface behind progressive disclosure.
 */
export function OrganizationBriefCard({ view, open }: { view: SeatView; open?: Open }) {
  const { brief, workstreams, blockers, pending } = organizationData(view);
  const hasBrief = Boolean(view.organization_brief);
  const summary = asText(
    brief.summary || brief.goal || brief.milestone,
    "The organization is running from its shared public state. Explore a workstream or ask the liaison for an explanation.",
  );

  return (
    <section className="organization-brief" aria-label="Organization brief">
      <div className="brief-kicker">
        <span>ORGANIZATION BRIEF</span>
        <span className="brief-new">{hasBrief ? "visible since you joined" : "public view"}</span>
      </div>
      <h2>{asText(brief.title, "What is happening")}</h2>
      <p className="brief-summary">{summary}</p>
      {brief.milestone && brief.goal && (
        <div className="brief-goal"><span>Current milestone</span>{asText(brief.milestone)}</div>
      )}
      <div className="brief-stats">
        <span><strong>{workstreams.length}</strong> workstreams</span>
        <span><strong>{blockers.length}</strong> blockers</span>
        <span><strong>{pending.length}</strong> decisions</span>
        {Array.isArray(brief.source_refs) && <span><strong>{brief.source_refs.length}</strong> sources</span>}
      </div>
      {blockers.length > 0 && (
        <div className="brief-callout warn">
          <span className="callout-icon">!</span>
          <div><strong>Needs attention</strong><div>{labelFor(blockers[0], "A visible workstream is blocked")}</div></div>
        </div>
      )}
      {brief.updated_at && <div className="brief-updated">Updated {asText(brief.updated_at)}</div>}
    </section>
  );
}

export function OrganizationOverview({ view, open }: { view: SeatView; open: Open }) {
  const { workstreams, agents, blockers } = organizationData(view);

  return (
    <>
      <OrganizationBriefCard view={view} open={open} />
      <section className="organization-surface" aria-label="Visible organization">
        <div className="surface-title">Visible organization</div>
        <div className="surface-note">Shared state only · click an item to inspect it</div>
        <div className="surface-group">
          <div className="surface-label">Workstreams</div>
          {workstreams.length === 0 && <div className="muted compact">No active workstreams are visible.</div>}
          {workstreams.map((item: any, index: number) => (
            <button className={`surface-item ${["blocked", "changes requested", "failed"].includes(statusFor(item)) ? "blocked" : ""}`} key={objectIdFor(item) || index}
                    onClick={() => objectIdFor(item) && open(objectIdFor(item), item)}>
              <span className="surface-item-main">
                <span className="surface-item-title">{labelFor(item, `Workstream ${index + 1}`)}</span>
                <span className="surface-item-meta">{asText(item?.summary || item?.description, statusFor(item))}</span>
              </span>
              <span className="surface-item-status">{statusFor(item)}</span>
            </button>
          ))}
        </div>
        <div className="surface-group">
          <div className="surface-label">Agents</div>
          {agents.length === 0 && <div className="muted compact">No agent activity is visible yet.</div>}
          <div className="agent-roster">
            {agents.map((agent: any, index: number) => (
              <button className="agent-chip" key={agent?.agent_id || agent?.id || index}
                      onClick={() => agent?.id && open(agent.id, agent)}>
                <span className={`agent-presence ${agent?.online === false ? "offline" : ""}`} />
                <span>{labelFor(agent, `Agent ${index + 1}`)}</span>
                <span className="agent-role">{asText(agent?.role || agent?.responsibility, "member").replace(/_/g, " ")}</span>
              </button>
            ))}
          </div>
        </div>
      </section>
    </>
  );
}
