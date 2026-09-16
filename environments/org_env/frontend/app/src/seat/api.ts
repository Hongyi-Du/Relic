// Everything the seat UI can ask the server. The seat token goes on every call
// that reads or writes as this member; there is no other identity.

export type Offer = {
  action_type: string;
  label: string;
  required: string[];
  optional: string[];
  target_param: string;
  confirm: boolean;
  allowed: boolean;
  denied_because?: string;
  target?: string;
};

export type OfferMenu = { object_id: string | null; kind: string; actions: Offer[] };

export type SeatView = {
  version: number;
  seat: {
    agent_id: string; name: string; codename: string; role: string;
    is_founder: boolean; identity: string; permissions: string[]; status: string;
  };
  clock: { now?: number };
  member: any;
  feed: { channels: any[]; threads: any[]; events: any[] };
  objects: Record<string, any[]>;
  /**
   * A human-readable, permission-filtered organization summary.  This is
   * intentionally optional while older seat snapshots are still in use; the
   * transparent liaison falls back to the public roster/feed when it is not
   * present.
   */
  organization_brief?: OrganizationBrief;
};

/** The public shape consumed by the P2 transparent liaison UI. */
export type OrganizationBrief = {
  title?: string;
  summary?: string;
  goal?: string;
  milestone?: string;
  updated_at?: string;
  workstreams?: any[];
  agents?: any[];
  blockers?: any[];
  recent_changes?: any[];
  protocols?: any[];
  pending_decisions?: any[];
  [key: string]: any;
};

export type RuntimeStatus = {
  running: boolean; seconds_per_tick: number; ticks_run: number;
  now: number; started_at: number; view_version: number; last_error: string | null;
  pack?: { id?: string; product_name?: string };
};

const j = (u: string) => fetch(u).then((r) => r.json());
const p = (u: string, b?: any) =>
  fetch(u, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(b || {}),
  }).then((r) => r.json());

export const api = {
  members: () => j(`/api/org/human/members`),
  claim: (agent_id: string, display_name = "") =>
    p(`/api/org/human/seats/claim`, { agent_id, display_name }),
  release: (token: string) => p(`/api/org/human/seats/release`, { token }),

  view: (token: string, since_version = -1) =>
    j(`/api/org/human/view?token=${encodeURIComponent(token)}&since_version=${since_version}`),
  brief: (token: string) =>
    j(`/api/org/human/brief?token=${encodeURIComponent(token)}`),

  offers: (token: string, object_id = "") =>
    j(`/api/org/human/offers?token=${encodeURIComponent(token)}` +
      (object_id ? `&object_id=${encodeURIComponent(object_id)}` : "")),

  act: (token: string, action_type: string, params: Record<string, any>,
        execution_mode = "direct") =>
    p(`/api/org/human/act`, { token, action_type, params, execution_mode }),

  agentState: (token: string, since = 0) =>
    j(`/api/org/human/agent?token=${encodeURIComponent(token)}&since=${since}`),
  agentSend: (token: string, text: string) =>
    p(`/api/org/human/agent/send`, { token, text }),
  agentConfirm: (token: string, draft_id: string) =>
    p(`/api/org/human/agent/confirm`, { token, draft_id }),
  agentDiscard: (token: string, draft_id: string) =>
    p(`/api/org/human/agent/discard`, { token, draft_id }),

  runtime: () => j(`/api/org/human/runtime`) as Promise<RuntimeStatus>,
  runtimeStart: (seconds_per_tick?: number) =>
    p(`/api/org/human/runtime/start`, { seconds_per_tick }),
  runtimePause: () => p(`/api/org/human/runtime/pause`, {}),
};

/** "3 minutes ago" — the seat never sees a tick, only elapsed real time. */
export function ago(seconds: number | null | undefined): string {
  if (seconds == null) return "";
  const s = Math.max(0, Math.round(seconds));
  if (s < 45) return "just now";
  const m = Math.round(s / 60);
  if (m < 60) return `${m}m ago`;
  const h = Math.round(m / 60);
  if (h < 24) return `${h}h ago`;
  return `${Math.round(h / 24)}d ago`;
}

export function clockTime(epoch?: number): string {
  if (!epoch) return "";
  return new Date(epoch * 1000).toLocaleTimeString([], {
    hour: "2-digit", minute: "2-digit",
  });
}
