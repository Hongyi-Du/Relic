/**
 * P3's only network boundary.
 *
 * This façade deliberately does not know about seats, agents, channels, or
 * object offers.  The server projects a permission-scoped organization view
 * and owns interpretation, pending drafts, and confirmation revalidation.
 */

export type P3Ref = {
  id?: string;
  object_id?: string;
  index?: number;
  kind?: string;
  label?: string;
  title?: string;
  status?: string;
  owner?: string;
  author?: string;
  [key: string]: any;
};

export type P3Workstream = {
  id?: string;
  title?: string;
  status?: string;
  summary?: string;
  owner?: string;
  why?: string;
  refs?: P3Ref[];
  evidence_refs?: string[];
  [key: string]: any;
};

export type P3Decision = {
  id?: string;
  draft_id?: string;
  title?: string;
  what?: string;
  why?: string;
  impact?: string;
  status?: string;
  refs?: P3Ref[];
  evidence_refs?: string[];
  decision_type?: string;
  preferred_resource_tab?: "summary" | "review";
  requires_human_decision?: boolean;
  decision_options?: P3DecisionOption[];
  [key: string]: any;
};

export type P3DecisionOption = {
  id?: string;
  label: string;
  instruction: string;
};

export type P3DecisionContext = {
  resource_ref: string;
  title: string;
  kind?: string;
  decision_type?: string;
  option_id?: string;
  option_label?: string;
};

export type P3Draft = {
  draft_id: string;
  thread_id?: string;
  label?: string;
  action_type?: string;
  rationale?: string;
  params?: Record<string, any>;
  display_params?: Record<string, any>;
  [key: string]: any;
};

export type P3Meeting = {
  title?: string;
  agenda?: string | string[];
  participants?: Array<string | { name?: string; role?: string; [key: string]: any }>;
  status?: string;
  decision?: string;
  viewpoint?: string;
  return_focus?: string;
  attendance_confirmed?: boolean;
  evidence_refs?: string[];
  [key: string]: any;
};

/** One human-facing meeting decision, potentially compiled from several P2 actions. */
export type P3MeetingPlan = {
  plan_id: string;
  title?: string;
  status?: string;
  decision?: string;
  viewpoint?: string;
  return_focus?: string;
  attendance_confirmed?: boolean;
  confirmation_required?: boolean;
  evidence_refs?: string[];
  [key: string]: any;
};

export type P3ExecutionWorker = {
  run_id: string;
  worker_id: string;
  name: string;
  role: string;
  assignment: string;
  status: string;
  model_calls: number;
  lifetime_model_calls: number;
  model_call_phase?: string;
  model_call_started_at?: number | null;
  activation_number: number;
  reused: boolean;
  report: string;
  [key: string]: any;
};

export type P3ExecutionAgent = {
  worker_id: string;
  name: string;
  role: string;
  status: "inactive" | "active" | string;
  created_at: number;
  last_active_at?: number | null;
  activation_count: number;
  model_calls: number;
  last_assignment: string;
  recent_assignments: string[];
  recent_reports: string[];
  active_job_id?: string;
  source: string;
  [key: string]: any;
};

export type P3ExecutionTimelineItem = {
  kind?: string;
  type?: string;
  summary?: string;
  text?: string;
  status?: string;
  at?: string | number;
  tick?: number;
  world_tick?: number;
  evidence_text?: string;
  evidence_refs?: string[];
  [key: string]: any;
};

export type P3ExecutionEvidence = {
  evidence_id: string;
  worker_id: string;
  worker_name?: string;
  tool: string;
  args?: Record<string, any>;
  source_text?: string;
  world_tick?: number;
};

/** Human-facing projection of a real execution-team job, when the liaison API returns one. */
export type P3ExecutionJob = {
  job_id: string;
  task_id?: string;
  claim_task?: boolean;
  progress_interval_seconds?: number;
  thread_id: string;
  goal: string;
  status: string;
  startable: boolean;
  start_block_reason?: string;
  cancelable: boolean;
  source: string;
  created_at: number;
  title: string;
  completion_criteria: string;
  secretary_model_calls: number;
  workers: P3ExecutionWorker[];
  timeline: P3ExecutionTimelineItem[];
  final_report: string;
  references?: P3Ref[];
  evidence_records?: P3ExecutionEvidence[];
  [key: string]: any;
};

export type P3FailedRequest = {
  thread_id?: string;
  original_request?: string;
  error?: string;
  failure_kind?: "model_interrupted" | "invalid_model_plan" | string;
  retry_current_step?: boolean;
  step?: number;
  max_steps?: number;
  next_step?: number;
  source_count?: number;
  sources?: Array<{ tool?: string; args?: Record<string, any>; observation_id?: string }>;
};

export type P3AttentionPreferences = {
  message_delivery?: string;
  description?: string;
  [key: string]: any;
};

/** Safe progress projection from the liaison work loop; never chain of thought. */
export type P3WorkingState = {
  status?: string;
  phase?: "understanding" | "grounding" | "investigating" | "re_grounding" | "synthesizing" | string;
  summary?: string;
  step?: number;
  max_steps?: number;
  remaining_steps?: number;
  tools_completed?: number;
  unique_sources?: number;
  duplicate_calls_blocked?: number;
  queued_requests?: number;
  started_at?: number;
  elapsed_seconds?: number;
  thread_id?: string;
  activity_log?: Array<{ step?: number; label?: string; status?: string }>;
};

export type P3Message = {
  id?: string;
  thread_id?: string;
  parent_id?: string;
  threadable?: boolean;
  role?: "human" | "liaison" | "system" | string;
  text: string;
  at?: string | number;
  tick?: number;
  evidence?: P3Ref[];
  /** Seat-validated objects behind names, pronouns, or numbered follow-ups. */
  references?: P3Ref[];
  /** Human-visible label plus the opaque, seat-bound decision reference sent with this turn. */
  decision_context?: P3DecisionContext;
  evidence_ref?: string;
  evidence_refs?: string[];
  trace_ref?: string;
  /** A token-bound resource-inspector handle, never a world/object id. */
  resource_ref?: string;
  resource_refs?: Array<string | { ref?: string; title?: string; kind?: string; section?: string; [key: string]: any }>;
  resource_focus?: { ref?: string; title?: string; kind?: string; section?: string; [key: string]: any };
  kind?: "message" | "clarification" | "status" | "event" | "reflection" | "meeting" |
         "meeting_invitation" | "meeting_report" | "incoming_message" |
         "attention_preference" | "meeting_instruction" | string;
  event_type?: string;
  status?: "requested" | "assigned" | "active" | "verified" |
           "delegated" | "executing" | "completed" | string;
  task_status?: string;
  details?: string;
  summary?: string;
  clarification?: string;
  decision_type?: string;
  preferred_resource_tab?: "summary" | "review";
  decision_options?: P3DecisionOption[];
  requires_human_decision?: boolean;
  meeting?: P3Meeting;
  importance?: string;
  urgency?: string;
  delivery_mode?: string;
  [key: string]: any;
};

export type P3ConversationThread = {
  thread_id: string;
  root_message_id: string;
  root_text?: string;
  reply_count?: number;
  latest_at?: string | number;
  latest_message_id?: string;
  latest_text?: string;
};

export type P3State = {
  mode?: string;
  organization?: any;
  conversation?: P3Message[];
  conversation_threads?: P3ConversationThread[];
  pending_actions?: P3Draft[];
  busy?: boolean;
  working?: P3WorkingState;
  session_id?: string;
  project_summary?: any;
  summary?: any;
  workstreams?: P3Workstream[];
  blocker?: any;
  max_blocker?: any;
  risk?: any;
  decision_inbox?: P3Decision[];
  decisions?: P3Decision[];
  messages?: P3Message[];
  pending_drafts?: P3Draft[];
  drafts?: P3Draft[];
  meeting_invitations?: P3Meeting[];
  meeting_plans?: P3MeetingPlan[];
  execution_jobs?: P3ExecutionJob[];
  execution_agents?: P3ExecutionAgent[];
  failed_request?: P3FailedRequest | null;
  failed_requests?: P3FailedRequest[];
  last_model_error?: string | null;
  attention_preferences?: P3AttentionPreferences;
  /** The last task that Victor explicitly confirmed, keyed by conversation thread. */
  active_task_focus?: Record<string, P3Ref & {
    claimed_by?: string;
    source_draft_id?: string;
    confirmed_at?: number;
  }>;
  resource_focus?: { ref?: string; title?: string; kind?: string; section?: string; [key: string]: any } | null;
  [key: string]: any;
};

/** The real human-mode clock, exposed by the liaison runtime routes. */
export type P3RuntimeStatus = {
  running: boolean;
  seconds_per_tick: number;
  ticks_run: number;
  world_tick?: number;
  pack?: {
    id?: string;
    dataset_id?: string;
    product_name?: string;
    source?: string;
    [key: string]: any;
  };
  engine?: string;
  event_source?: string;
  llm?: {
    requested?: boolean;
    attached?: boolean;
    mode?: "model_backed" | "rule_template_only" | string;
    load_error?: boolean;
  };
  policy_mode?: string;
  action_selection_mode?: string;
  session_last_error?: string | null;
  now?: number;
  started_at?: number;
  view_version?: number;
  last_error?: string | null;
  [key: string]: any;
};

export type Disclosure = {
  summary?: any;
  evidence?: any;
  trace?: any;
  [key: string]: any;
};

export type P3ResourceSection = {
  availability?: "available" | "unavailable" | string;
  available?: boolean;
  reason?: string;
  [key: string]: any;
};

/** Seat-visible, read-only resource projection returned by the liaison API. */
export type P3Resource = {
  ref?: string;
  kind?: string;
  title?: string;
  status?: string;
  summary?: string;
  overview?: string;
  review?: P3ResourceSection | null;
  files?: P3ResourceSection | Array<any> | null;
  diff?: P3ResourceSection | string | null;
  tests?: P3ResourceSection | Array<any> | null;
  evidence?: P3ResourceSection | null;
  provenance?: P3ResourceSection | null;
  sections?: Record<string, P3ResourceSection | any>;
  [key: string]: any;
};

export type P3ResourceResponse = {
  resource?: P3Resource;
  resource_ref?: string;
  resource_kind?: string;
  resource_section?: string;
  execution_context?: { jobs?: P3ExecutionJob[] };
  secretary_context?: {
    reports?: Array<{ message_id?: string; kind?: string; text?: string; evidence_records?: P3ExecutionEvidence[] }>;
    evidence_records?: P3ExecutionEvidence[];
  };
  section?: string;
  availability?: Record<string, { available?: boolean; reason?: string; [key: string]: any }>;
  content?: any;
  links?: any[];
  provenance?: Record<string, any>;
  observed_at?: Record<string, any>;
  ref?: string;
  kind?: string;
  title?: string;
  status?: string;
  [key: string]: any;
};

const request = async (path: string, init?: RequestInit): Promise<any> => {
  let response: Response;
  try {
    response = await fetch(`/api/org/liaison${path}`, init);
  } catch {
    throw new Error("Connection lost. The last loaded state is still visible; reconnecting automatically.");
  }
  let payload: any = {};
  try { payload = await response.json(); }
  catch {
    if (!response.ok) throw new Error(`liaison request failed (${response.status})`);
    throw new Error("The liaison service returned an unreadable response.");
  }
  if (!response.ok || payload?.error)
    throw new Error(payload?.error || `liaison request failed (${response.status})`);
  return payload;
};

const get = async (path: string): Promise<any> => request(path);

const post = async (path: string, body: Record<string, any>): Promise<any> => {
  return request(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
};

export const liaisonApi = {
  session: (token = "") => post("/session", { token }),
  release: (token: string) => post("/release", { token }),
  state: (token: string, since = 0) => get(`/state?token=${encodeURIComponent(token)}&since=${since}`) as Promise<P3State>,
  ask: (token: string, text: string, options: { reply_to?: string; thread_id?: string; decision_context?: P3DecisionContext } = {}) =>
    post("/ask", { token, text, ...options }),
  cancelRequest: (token: string) => post("/request/cancel", { token }),
  retryRequest: (token: string, threadId = "") =>
    post("/request/retry", { token, thread_id: threadId }),
  evidence: (token: string, ref: string) =>
    get(`/evidence?token=${encodeURIComponent(token)}&ref=${encodeURIComponent(ref)}`) as Promise<Disclosure>,
  trace: (token: string, ref: string) =>
    get(`/trace?token=${encodeURIComponent(token)}&ref=${encodeURIComponent(ref)}`) as Promise<Disclosure>,
  resource: (token: string, ref: string, section?: string) =>
    get(`/resource?token=${encodeURIComponent(token)}&ref=${encodeURIComponent(ref)}${section ? `&section=${encodeURIComponent(section)}` : ""}`) as Promise<P3ResourceResponse>,
  organization: (token: string) =>
    get(`/organization?token=${encodeURIComponent(token)}`) as Promise<any>,
  runtime: () => get("/runtime") as Promise<P3RuntimeStatus>,
  runtimeStart: (seconds_per_tick?: number) =>
    post("/runtime/start", seconds_per_tick == null ? {} : { seconds_per_tick }) as Promise<P3RuntimeStatus>,
  runtimePause: () => post("/runtime/pause", {}) as Promise<P3RuntimeStatus>,
  confirmMeetingPlan: (token: string, planId: string) =>
    post("/confirm-meeting-plan", { token, plan_id: planId }),
  discardMeetingPlan: (token: string, planId: string) =>
    post("/discard-meeting-plan", { token, plan_id: planId }),
  startExecution: (token: string, jobId?: string) =>
    post("/execution/start", { token, ...(jobId ? { job_id: jobId } : {}) }),
  cancelExecution: (token: string, jobId?: string) =>
    post("/execution/cancel", { token, ...(jobId ? { job_id: jobId } : {}) }),
  confirm: (token: string, draftId: string) => post("/confirm", { token, draft_id: draftId }),
  discard: (token: string, draftId: string) => post("/discard", { token, draft_id: draftId }),
};
