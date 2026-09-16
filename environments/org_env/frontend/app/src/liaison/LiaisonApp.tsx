import { useCallback, useEffect, useRef, useState } from "react";
import type { KeyboardEvent } from "react";
import { liaisonApi, P3ConversationThread, P3DecisionContext, P3DecisionOption, P3Draft, P3ExecutionAgent, P3ExecutionJob, P3ExecutionTimelineItem, P3ExecutionWorker, P3FailedRequest, P3MeetingPlan, P3Message, P3Resource, P3ResourceResponse, P3RuntimeStatus, P3State, P3WorkingState } from "./api";
import { MarkdownMessage } from "./MarkdownMessage";

// P3 has one deliberate perspective: Victor.  The shared P2 key lets a
// browser resume Victor when switching views, but the server rejects every
// other seat token instead of trusting this client-side convention.
const TOKEN_KEY = "socio.seat.token";
const textOf = (value: any, fallback = "") => {
  if (value == null || value === "") return fallback;
  if (typeof value === "string" || typeof value === "number") return String(value);
  return String(value.text || value.summary || value.what || value.title || value.name || fallback);
};
const objectRows = <T extends object>(value: unknown): T[] =>
  Array.isArray(value)
    ? value.filter((item): item is T => Boolean(item) && typeof item === "object" && !Array.isArray(item))
    : [];

const normalizeP3State = (value: unknown): P3State => {
  const raw = value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, any> : {};
  return {
    ...raw,
    conversation: objectRows<P3Message>(raw.conversation),
    conversation_threads: objectRows<P3ConversationThread>(raw.conversation_threads),
    pending_actions: objectRows<P3Draft>(raw.pending_actions),
    decision_inbox: objectRows(raw.decision_inbox),
    meeting_invitations: objectRows(raw.meeting_invitations),
    meeting_plans: objectRows<P3MeetingPlan>(raw.meeting_plans),
    execution_jobs: objectRows<P3ExecutionJob>(raw.execution_jobs),
    execution_agents: objectRows<P3ExecutionAgent>(raw.execution_agents),
    failed_requests: objectRows<P3FailedRequest>(raw.failed_requests),
  };
};

const evidenceRefOf = (message: P3Message | null | undefined) => {
  if (!message || typeof message !== "object") return "";
  if (message.evidence_ref) return message.evidence_ref;
  if (message.evidence_refs?.length) return message.evidence_refs[0];
  const first = message.evidence?.[0] as any;
  return typeof first === "string" ? first : first?.id || "";
};
const resourceRefOf = (message: P3Message | P3ResourceResponse | null | undefined) => {
  if (!message || typeof message !== "object") return "";
  const candidates: any[] = [
    message.resource_ref,
    ...(Array.isArray(message.resource_refs) ? message.resource_refs : []),
    message.resource_focus,
  ];
  for (const candidate of candidates) {
    const ref = typeof candidate === "string" ? candidate : candidate?.ref;
    if (typeof ref === "string" && ref.startsWith("ri_")) return ref;
  }
  return "";
};
type ResourceTab = "summary" | "review" | "files" | "diff" | "tests" | "evidence";
const RESOURCE_TABS: Array<{ id: ResourceTab; label: string }> = [
  { id: "summary", label: "Summary" },
  { id: "review", label: "Review" },
  { id: "files", label: "Files" },
  { id: "diff", label: "Diff" },
  { id: "tests", label: "Tests" },
  { id: "evidence", label: "Evidence" },
];
const RESOURCE_SECTION_BY_TAB: Record<ResourceTab, string> = {
  summary: "overview",
  review: "review",
  files: "files",
  diff: "diff",
  tests: "ci",
  evidence: "provenance",
};
const resourceTabOfSection = (section: unknown): ResourceTab | undefined => {
  const normalized = String(section || "").toLowerCase();
  if (RESOURCE_TABS.some((tab) => tab.id === normalized)) return normalized as ResourceTab;
  if (normalized === "overview" || normalized === "details") return "summary";
  if (normalized === "ci") return "tests";
  if (normalized === "provenance" || normalized === "commits" || normalized === "patches") return "evidence";
  if (normalized === "content") return "files";
  return undefined;
};
type ResourceTarget = { ref: string; kind?: string; title?: string; section?: ResourceTab };
const resourceFocusOf = (value: any): ResourceTarget | null => {
  const candidate = value?.resource_focus || value?.resourceFocus || value?.resource_ref || value?.resource_refs?.[0] || value;
  const ref = typeof candidate === "string" ? candidate : candidate?.ref;
  if (typeof ref !== "string" || !ref.startsWith("ri_")) return null;
  return { ref, kind: typeof candidate === "object" ? candidate.kind : undefined, title: typeof candidate === "object" ? candidate.title : undefined, section: typeof candidate === "object" ? resourceTabOfSection(candidate.section) : undefined };
};
const decisionContextOf = (
  message: P3Message,
  option?: Pick<P3DecisionOption, "id" | "label">,
): P3DecisionContext | undefined => {
  const resourceRef = resourceRefOf(message);
  if (!resourceRef) return undefined;
  return {
    resource_ref: resourceRef,
    title: textOf(message.title || message.what || message.summary, "Decision"),
    kind: message.kind,
    decision_type: message.decision_type,
    option_id: option?.id,
    option_label: option?.label,
  };
};
// A resource_ref makes a card clickable; it is not an instruction to move the
// inspector.  Only the liaison's recorded resource_opened reply carries this
// explicit focus command.
const secretaryResourceFocusCommandOf = (message: P3Message) => {
  if (message.role !== "liaison" || !message.resource_focus) return null;
  const target = resourceFocusOf({ resource_focus: message.resource_focus });
  const messageId = String(message.id || "");
  if (!target || !messageId) return null;
  const tab = target.section || "summary";
  return { target, tab, key: `${messageId}\u0000${target.ref}\u0000${tab}` };
};

const isOrganizationEvent = (message: P3Message) => {
  // These are conversation boundaries even when the backend attaches a
  // status/event field for bookkeeping.
  if (message.role === "human" || message.clarification || message.kind === "clarification") return false;
  if (["meeting_invitation", "meeting_report", "incoming_message", "attention_preference", "meeting_instruction"].includes(message.kind || "")) return false;
  const kind = message.kind || (message.event_type ? "event" : message.status ? "status" : "message");
  return kind === "event" || kind === "reflection" || kind === "meeting" || kind === "status";
};

const messageStatus = (message: P3Message) => (message.status || message.task_status || "").toLowerCase();
const messageFamily = (message: P3Message) => `${message.kind || ""} ${message.event_type || ""}`.toLowerCase();
const isDecisionEvent = (message: P3Message) => {
  const status = messageStatus(message);
  return message.requires_human_decision === true || message.kind === "decision_request" ||
    message.kind === "meeting_invitation" ||
    /^(decision[_ -]?required|human[_ -]?judgment)$/.test(status);
};
const isExecutionEvent = (message: P3Message) => {
  const status = messageStatus(message);
  if (!/(requested|assigned|active|delegated|executing|completed|verified|merged|released|done)/.test(status)) return false;
  if (/meeting|reflection|proposal/.test(messageFamily(message))) return false;
  return /task|execution/.test(`${messageFamily(message)} ${message.task_status ? "task" : ""}`);
};
const isReadyMessage = (message: P3Message) => {
  // "Ready to try" is a product claim, not a synonym for a completed event.
  // Render it only when the backend explicitly supplies that contract.
  return message.ready_to_try === true;
};
type EventTone = "decision" | "execution" | "fyi";
const eventTone = (message: P3Message): EventTone => isDecisionEvent(message) ? "decision" : isExecutionEvent(message) ? "execution" : "fyi";
const listText = (value: any, fallback: string) => Array.isArray(value) ? value.map((item) => textOf(item, "")).filter(Boolean).join(" · ") || fallback : textOf(value, fallback);
const participantText = (value: any, fallback = "Participants not returned") => Array.isArray(value) ? value.map((item) => typeof item === "string" ? item : textOf(item?.name || item?.role, "Participant")).join(" · ") || fallback : textOf(value, fallback);

const executionTimeOf = (value: any, fallback = Number.POSITIVE_INFINITY) => {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string" && value.trim()) {
    const numeric = Number(value);
    if (Number.isFinite(numeric)) return numeric;
    const parsed = Date.parse(value);
    if (Number.isFinite(parsed)) return parsed;
  }
  return fallback;
};
const executionJobsOf = (state: P3State | null): P3ExecutionJob[] =>
  objectRows<P3ExecutionJob>(state?.execution_jobs)
    .map((job, index) => ({ job, index }))
    .filter(({ job }) => job && typeof job === "object" && Object.keys(job).length)
    .sort((left, right) => {
      // Use the runtime's creation timestamp; the timeline is a defensive
      // fallback for older state snapshots that predate that field.
      const leftAt = executionTimeOf(left.job.created_at, executionTimeOf(left.job.timeline?.[0]?.at));
      const rightAt = executionTimeOf(right.job.created_at, executionTimeOf(right.job.timeline?.[0]?.at));
      const bothTimestamped = Number.isFinite(leftAt) && Number.isFinite(rightAt);
      if (bothTimestamped && leftAt !== rightAt) return leftAt - rightAt;
      if (Number.isFinite(leftAt) !== Number.isFinite(rightAt)) return Number.isFinite(leftAt) ? -1 : 1;
      return left.index - right.index;
    })
    .map(({ job }) => job);
const executionAgentsOf = (state: P3State | null): P3ExecutionAgent[] =>
  objectRows<P3ExecutionAgent>(state?.execution_agents)
    .filter((agent) => agent && typeof agent === "object" && Boolean(agent.worker_id))
    .sort((left, right) => executionTimeOf(left.created_at, 0) - executionTimeOf(right.created_at, 0));
const executionFocusOf = (jobs: P3ExecutionJob[]): P3ExecutionJob | null =>
  [...jobs].reverse().find((job) => /^(running|starting|pending_confirmation)$/i.test(String(job.status || ""))) || jobs[jobs.length - 1] || null;
const executionWorkersOf = (job: P3ExecutionJob): P3ExecutionWorker[] =>
  objectRows<P3ExecutionWorker>(job?.workers);
const executionTimelineOf = (job: P3ExecutionJob): P3ExecutionTimelineItem[] =>
  objectRows<P3ExecutionTimelineItem>(job?.timeline);
const executionStatus = (job: P3ExecutionJob) => String(job.status || "status not returned").replace(/_/g, " ");
const executionCanStart = (job: P3ExecutionJob) => job.startable === true;
const executionCanCancel = (job: P3ExecutionJob) => job.cancelable === true;
const executionLabel = (value: any, fallback: string) => textOf(value?.name || value?.display_name || value?.title || value, fallback);
const executionStartBlockText = (job: P3ExecutionJob) => {
  const reason = String(job.start_block_reason || "");
  if (reason.startsWith("execution_agent_busy:")) {
    const workerId = reason.slice("execution_agent_busy:".length);
    const worker = executionWorkersOf(job).find((candidate) => candidate.worker_id === workerId);
    return `${executionLabel(worker, "This persistent agent")} is active in another job. This plan stays queued until that activation settles.`;
  }
  if (reason.startsWith("execution_agent_not_found:")) return "The selected persistent agent is no longer available in this organization runtime.";
  if (reason.startsWith("execution_agent_wrong_runtime:")) return "The selected agent belongs to an earlier organization runtime and cannot be activated here.";
  return reason.replace(/_/g, " ");
};
const executionItemSummary = (item: P3ExecutionTimelineItem) => {
  if (item.summary || item.text) return textOf(item.summary || item.text, "Action update not returned");
  if (item.action_type) return String(item.action_type).replace(/_/g, " ");
  return "Action update not returned";
};
const executionEvidenceText = (item: P3ExecutionTimelineItem) => {
  if (item.evidence_text || item.evidence_summary) return textOf(item.evidence_text || item.evidence_summary, "Evidence returned by liaison");
  if (typeof item.ok === "boolean") return item.ok ? "Action completed successfully; evidence was recorded." : `Action failed: ${textOf(item.failure_reason, "failure reason not returned")}`;
  return item.evidence ? "Evidence returned by liaison." : "";
};
const executionWorldTick = (item: P3ExecutionTimelineItem) => item.world_tick ?? item.tick;

type ConversationBlock =
  | { kind: "events"; messages: P3Message[]; tone: EventTone }
  | { kind: "turn"; message: P3Message };

type OptimisticHumanMessage = P3Message & {
  client_id: string;
  delivery_status: "sending" | "failed";
};

const newOptimisticHumanMessage = (text: string, threadId = "", decisionContext?: P3DecisionContext): OptimisticHumanMessage => {
  const clientId = `client_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`;
  return {
    id: clientId,
    client_id: clientId,
    role: "human",
    text,
    decision_context: decisionContext,
    thread_id: threadId,
    threadable: false,
    at: Date.now() / 1000,
    delivery_status: "sending",
  };
};

/**
 * Keep a locally accepted human turn visible while the server's intent model
 * is still routing it. Server echoes replace optimistic turns one-for-one, so
 * repeated identical messages do not either disappear or render twice.
 */
const mergeOptimisticMessages = (
  serverMessages: P3Message[], optimisticMessages: OptimisticHumanMessage[],
): P3Message[] => {
  // State is normalized at the API boundary, but keep this merge defensive:
  // an older gateway or a partially written snapshot must not make the whole
  // liaison page disappear while a human send is in flight.
  const safeServerMessages = objectRows<P3Message>(serverMessages);
  const matchedServerIndexes = new Set<number>();
  const unacknowledged = optimisticMessages.filter((optimistic) => {
    const optimisticAt = Number(optimistic.at || 0);
    const match = safeServerMessages.findIndex((server, index) => {
      if (matchedServerIndexes.has(index) || server.role !== "human") return false;
      if (String(server.text || "") !== optimistic.text) return false;
      if (String(server.thread_id || "") !== String(optimistic.thread_id || "")) return false;
      const serverAt = Number(server.at || 0);
      // An older identical message is not an acknowledgement of this send.
      return !serverAt || !optimisticAt || (
        serverAt >= optimisticAt - 1 && serverAt - optimisticAt < 600
      );
    });
    if (match < 0) return true;
    matchedServerIndexes.add(match);
    return false;
  });
  return [...safeServerMessages, ...objectRows<OptimisticHumanMessage>(unacknowledged)]
    .sort((left, right) => Number(left.at || 0) - Number(right.at || 0));
};

/**
 * Consecutive organization activity is one readable update, while human and
 * clarification turns deliberately break the digest so the conversation's
 * causal order remains visible.
 */
const groupConversation = (messages: P3Message[]): ConversationBlock[] => {
  const blocks: ConversationBlock[] = [];
  for (const message of objectRows<P3Message>(messages)) {
    if (isOrganizationEvent(message)) {
      const previous = blocks[blocks.length - 1];
      const tone = eventTone(message);
      if (previous?.kind === "events" && previous.tone === tone) previous.messages.push(message);
      else blocks.push({ kind: "events", messages: [message], tone });
    } else {
      blocks.push({ kind: "turn", message });
    }
  }
  return blocks;
};

export function LiaisonApp() {
  const [token, setToken] = useState<string | null>(null);
  const [booting, setBooting] = useState(true);
  const [state, setState] = useState<P3State | null>(null);
  const [text, setText] = useState("");
  const [threadText, setThreadText] = useState("");
  const [decisionContext, setDecisionContext] = useState<P3DecisionContext | undefined>();
  const [threadDecisionContext, setThreadDecisionContext] = useState<P3DecisionContext | undefined>();
  const [optimisticHumanMessages, setOptimisticHumanMessages] = useState<OptimisticHumanMessage[]>([]);
  const [threadRootId, setThreadRootId] = useState("");
  const [requestScope, setRequestScope] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [stateError, setStateError] = useState("");
  const [runtimeError, setRuntimeError] = useState("");
  const [organizationOpen, setOrganizationOpen] = useState(false);
  const [organization, setOrganization] = useState<any>(null);
  const [runtime, setRuntime] = useState<P3RuntimeStatus | null>(null);
  const [runtimeBusy, setRuntimeBusy] = useState(false);
  const [view, setView] = useState<"conversation" | "decisions" | "activity">("conversation");
  const [resourceTarget, setResourceTarget] = useState<ResourceTarget | null>(null);
  const [resourceTab, setResourceTab] = useState<ResourceTab>("summary");
  const [resourcePayload, setResourcePayload] = useState<P3ResourceResponse | null>(null);
  const [resourceBusy, setResourceBusy] = useState(false);
  const [resourceError, setResourceError] = useState("");
  const end = useRef<HTMLDivElement | null>(null);
  const threadEnd = useRef<HTMLDivElement | null>(null);
  const lastConsumedResourceFocusCommand = useRef("");
  const resourceRequestId = useRef(0);
  const resourceSelection = useRef<{ ref: string; tab: ResourceTab } | null>(null);
  const executionJobs = executionJobsOf(state);
  const executionJob = executionFocusOf(executionJobs);

  const connectVictor = useCallback(async (candidate = localStorage.getItem(TOKEN_KEY) || "") => {
    setBooting(true); setError("");
    try {
      let result: any;
      try { result = await liaisonApi.session(candidate); }
      catch (err: any) {
        const message = err.message || String(err);
        if (message.includes("World not initialized")) {
          window.location.assign("/org/setup?next=liaison");
          return;
        }
        if (!candidate || (!message.includes("invalid_seat_token") && !message.includes("p3_fixed_seat_required"))) throw err;
        localStorage.removeItem(TOKEN_KEY);
        result = await liaisonApi.session();
      }
      localStorage.setItem(TOKEN_KEY, result.token);
      setToken(result.token);
    } catch (err: any) { setError(err.message || String(err)); }
    finally { setBooting(false); }
  }, []);

  const load = useCallback(async () => {
    if (!token) return;
    try { setState(normalizeP3State(await liaisonApi.state(token))); setStateError(""); }
    catch (err: any) {
      const message = err.message || String(err);
      if (message.includes("invalid_seat_token")) {
        localStorage.removeItem(TOKEN_KEY); setToken(null); setState(null); setStateError("");
        void connectVictor("");
      }
      else setStateError(message);
    }
  }, [connectVictor, token]);

  const loadRuntime = useCallback(async () => {
    try { setRuntime(await liaisonApi.runtime()); setRuntimeError(""); }
    catch (err: any) { setRuntimeError(err.message || String(err)); }
  }, []);

  useEffect(() => {
    if (!token) return;
    let cancelled = false;
    let stateTimer: number | undefined;
    let runtimeTimer: number | undefined;
    const pollState = async () => {
      await load();
      if (!cancelled) stateTimer = window.setTimeout(pollState, 1200);
    };
    const pollRuntime = async () => {
      await loadRuntime();
      if (!cancelled) runtimeTimer = window.setTimeout(pollRuntime, 1200);
    };
    void pollState();
    void pollRuntime();
    return () => {
      cancelled = true;
      if (stateTimer != null) window.clearTimeout(stateTimer);
      if (runtimeTimer != null) window.clearTimeout(runtimeTimer);
    };
  }, [load, loadRuntime, token]);

  useEffect(() => { void connectVictor(); }, [connectVictor]);

  useEffect(() => {
    const serverMessages = state?.conversation || [];
    setOptimisticHumanMessages((current) => current.filter((optimistic) =>
      !serverMessages.some((server) => server.role === "human"
        && String(server.text || "") === optimistic.text
        && String(server.thread_id || "") === String(optimistic.thread_id || ""))));
  }, [state?.conversation]);

  const selectResource = useCallback((target: ResourceTarget, tab: ResourceTab) => {
    resourceRequestId.current += 1;
    resourceSelection.current = { ref: target.ref, tab };
    setResourceBusy(false);
    setResourceError("");
    setResourcePayload(null);
    setResourceTarget(target);
    setResourceTab(tab);
  }, []);

  const closeSelectedResource = useCallback(() => {
    resourceRequestId.current += 1;
    resourceSelection.current = null;
    setResourceBusy(false);
    setResourceTarget(null);
    setResourcePayload(null);
    setResourceError("");
  }, []);

  const loadResource = useCallback(async (target: ResourceTarget, section: ResourceTab) => {
    if (!token || !target.ref.startsWith("ri_")) return;
    const requestId = ++resourceRequestId.current;
    setResourceBusy(true); setResourceError("");
    try {
      const payload = await liaisonApi.resource(token, target.ref, RESOURCE_SECTION_BY_TAB[section]);
      if (requestId === resourceRequestId.current && resourceSelection.current?.ref === target.ref && resourceSelection.current.tab === section) setResourcePayload(payload);
    }
    catch (err: any) {
      if (requestId === resourceRequestId.current && resourceSelection.current?.ref === target.ref && resourceSelection.current.tab === section) setResourceError(err.message || String(err));
    }
    finally {
      if (requestId === resourceRequestId.current && resourceSelection.current?.ref === target.ref && resourceSelection.current.tab === section) setResourceBusy(false);
    }
  }, [token]);

  useEffect(() => {
    const command = [...(state?.conversation || [])].reverse()
      .map(secretaryResourceFocusCommandOf)
      .find(Boolean);
    if (!command || command.key === lastConsumedResourceFocusCommand.current) return;
    lastConsumedResourceFocusCommand.current = command.key;
    selectResource(command.target, command.tab);
  }, [selectResource, state?.conversation]);

  useEffect(() => {
    if (resourceTarget) void loadResource(resourceTarget, resourceTab);
  }, [loadResource, resourceTarget, resourceTab]);

  useEffect(() => { end.current?.scrollIntoView({ behavior: "smooth" }); }, [state?.conversation?.length, state?.pending_actions?.length]);
  useEffect(() => { if (threadRootId) threadEnd.current?.scrollIntoView({ behavior: "smooth" }); }, [state?.conversation?.length, state?.pending_actions?.length, threadRootId]);

  const ask = async () => {
    const request = text.trim();
    if (!token || !request || busy) return;
    // The composer stays available in the Decisions and Activity filters, but
    // a newly sent human turn belongs to the conversation.  Switch before the
    // optimistic insert so the request never appears to vanish behind the
    // active filter while the model is working.
    setView("conversation");
    const submittedContext = decisionContext;
    const optimistic = newOptimisticHumanMessage(request, "", submittedContext);
    setOptimisticHumanMessages((current) => [...current, optimistic]);
    setText(""); setBusy(true); setRequestScope(""); setError("");
    try {
      await liaisonApi.ask(token, request, submittedContext ? { decision_context: submittedContext } : {});
      setDecisionContext(undefined);
      await load();
    }
    catch (err: any) {
      setOptimisticHumanMessages((current) => current.map((message) => message.client_id === optimistic.client_id
        ? { ...message, delivery_status: "failed" }
        : message));
      setError(`Message was not sent: ${err.message || String(err)}. Your text is preserved above.`);
    }
    finally { setBusy(false); setRequestScope(""); }
  };
  const askThread = async () => {
    const request = threadText.trim();
    if (!token || !threadRootId || !request || busy) return;
    const hasReplies = (state?.conversation || []).some((message) => message.thread_id === threadRootId);
    const submittedContext = threadDecisionContext;
    const optimistic = newOptimisticHumanMessage(request, threadRootId, submittedContext);
    setOptimisticHumanMessages((current) => [...current, optimistic]);
    setThreadText(""); setBusy(true); setRequestScope(threadRootId); setError("");
    try {
      await liaisonApi.ask(token, request, {
        ...(hasReplies ? { thread_id: threadRootId } : { reply_to: threadRootId }),
        ...(submittedContext ? { decision_context: submittedContext } : {}),
      });
      setThreadDecisionContext(undefined);
      await load();
    } catch (err: any) {
      setOptimisticHumanMessages((current) => current.map((message) => message.client_id === optimistic.client_id
        ? { ...message, delivery_status: "failed" }
        : message));
      setError(`Message was not sent: ${err.message || String(err)}. Your text is preserved in this thread.`);
    }
    finally { setBusy(false); setRequestScope(""); }
  };
  const stopCurrentRequest = async () => {
    if (!token) return;
    setError("");
    try { await liaisonApi.cancelRequest(token); await load(); }
    catch (err: any) { setError(err.message || String(err)); }
  };
  const retryFailedRequest = async (threadId = "") => {
    if (!token || busy) return;
    setBusy(true); setRequestScope(threadId); setError("");
    try { await liaisonApi.retryRequest(token, threadId); await load(); }
    catch (err: any) { setError(err.message || String(err)); }
    finally { setBusy(false); setRequestScope(""); }
  };
  const settle = async (draftId: string, choice: "confirm" | "discard") => {
    if (!token || busy) return;
    setBusy(true); setError("");
    try { if (choice === "confirm") await liaisonApi.confirm(token, draftId); else await liaisonApi.discard(token, draftId); await load(); }
    catch (err: any) { setError(err.message || String(err)); }
    finally { setBusy(false); }
  };
  const settleMeetingPlan = async (planId: string, choice: "confirm" | "discard") => {
    if (!token || busy) return;
    setBusy(true); setError("");
    try {
      if (choice === "confirm") await liaisonApi.confirmMeetingPlan(token, planId);
      else await liaisonApi.discardMeetingPlan(token, planId);
      await load();
    } catch (err: any) { setError(err.message || String(err)); }
    finally { setBusy(false); }
  };
  const settleExecution = async (jobId: string, choice: "start" | "cancel") => {
    if (!token || busy || !jobId) return;
    setBusy(true); setError("");
    try {
      if (choice === "start") await liaisonApi.startExecution(token, jobId);
      else await liaisonApi.cancelExecution(token, jobId);
      await load();
    } catch (err: any) { setError(err.message || String(err)); }
    finally { setBusy(false); }
  };
  const toggleOrganization = async () => {
    if (!token) return;
    if (resourceTarget) { closeResource(); return; }
    if (organizationOpen) { setOrganizationOpen(false); return; }
    setThreadRootId("");
    setOrganizationOpen(true);
    try { const payload = await liaisonApi.organization(token); setOrganization(payload.organization || payload); }
    catch (err: any) { setError(err.message || String(err)); }
  };

  const openResource = (messageOrFocus: P3Message | ResourceTarget | any, initialTab: ResourceTab = "summary") => {
    const focus = resourceFocusOf(messageOrFocus) || resourceFocusOf(messageOrFocus?.resource_focus);
    if (!focus) return;
    setThreadRootId("");
    setOrganizationOpen(false);
    selectResource(focus, focus.section || initialTab);
  };
  const closeResource = () => { closeSelectedResource(); };
  const selectResourceTab = (tab: ResourceTab) => {
    if (tab === resourceTab) return;
    if (!resourceTarget) return;
    resourceRequestId.current += 1;
    resourceSelection.current = { ref: resourceTarget.ref, tab };
    setResourceError("");
    setResourceTab(tab);
  };

  const setNavigation = (next: "conversation" | "decisions" | "activity") => {
    setView(next);
    setThreadRootId("");
    setOrganizationOpen(false);
    if (resourceTarget) closeResource();
    // Navigation is local and read-only. Keep the one composer immediately
    // reachable even when the user is inspecting a filtered view.
    if (next === "conversation") window.setTimeout(() => document.querySelector<HTMLTextAreaElement>(".composer-card textarea")?.focus(), 0);
  };
  const toggleRuntime = async () => {
    if (runtimeBusy) return;
    setRuntimeBusy(true); setError("");
    try { setRuntime(await (runtime?.running ? liaisonApi.runtimePause() : liaisonApi.runtimeStart())); }
    catch (err: any) { setError(err.message || String(err)); }
    finally { setRuntimeBusy(false); }
  };
  const openThread = (message: P3Message) => {
    const rootId = message.thread_id || message.id || "";
    if (!rootId) return;
    setView("conversation");
    setOrganizationOpen(false);
    setThreadRootId(rootId);
    window.setTimeout(() => document.querySelector<HTMLTextAreaElement>(".thread-panel textarea")?.focus(), 0);
  };
  const composeDecision = (instruction: string, context?: P3DecisionContext) => {
    setText(instruction);
    setDecisionContext(context);
    setError("");
    window.setTimeout(() => {
      const composer = document.querySelector<HTMLTextAreaElement>(".composer-zone .composer-card textarea");
      composer?.focus();
      if (composer) composer.selectionStart = composer.selectionEnd = composer.value.length;
    }, 0);
  };
  const composeThreadDecision = (instruction: string, context?: P3DecisionContext) => {
    setThreadText(instruction);
    setThreadDecisionContext(context);
    setError("");
    window.setTimeout(() => {
      const composer = document.querySelector<HTMLTextAreaElement>(".thread-composer-zone .composer-card textarea");
      composer?.focus();
      if (composer) composer.selectionStart = composer.selectionEnd = composer.value.length;
    }, 0);
  };

  if (!token) return <VictorBootstrap booting={booting} error={error} onRetry={() => connectVictor("")} />;
  const serverMessages = state?.conversation || [];
  const messages = mergeOptimisticMessages(serverMessages, optimisticHumanMessages);
  const mainMessages = messages.filter((message) => !message.thread_id);
  const threadSummaries = state?.conversation_threads || [];
  const threadByRoot = new Map(threadSummaries.map((thread) => [thread.root_message_id, thread]));
  const meetingPlans = state?.meeting_plans || [];
  const technicalMeetingActions = new Set(["send_message", "skip_meeting"]);
  // Older servers may still return component drafts while a plan is being
  // refreshed. Never make those implementation details a second confirmation
  // card when a semantic meeting plan is available.
  const drafts = (state?.pending_actions || []).filter((draft) =>
    !(meetingPlans.length > 0 && technicalMeetingActions.has(String(draft.action_type || ""))));
  const mainDrafts = drafts.filter((draft) => !draft.thread_id);
  const threadDrafts = drafts.filter((draft) => draft.thread_id === threadRootId);
  const decisionInboxMessages: P3Message[] = (state?.decision_inbox || []).map((item, index) => ({
    id: item.id || `decision-${index}`,
    role: "liaison",
    threadable: true,
    kind: "decision_request",
    status: "decision_required",
    text: textOf(item.summary || item.what || item.title, "Victor's decision is required."),
    summary: textOf(item.summary || item.what || item.title, "Victor's decision is required."),
    evidence_refs: item.evidence_refs,
    requires_human_decision: true,
    title: item.title,
    what: item.what,
    why: item.why,
    impact: item.impact,
    context: item.context,
    decision_type: item.decision_type,
    decision_options: item.decision_options,
    resource_ref: (item as any).resource_ref,
    resource_refs: (item as any).resource_refs,
    resource_focus: (item as any).resource_focus,
  }));
  const threadRoot = mainMessages.find((message) => message.id === threadRootId)
    || decisionInboxMessages.find((message) => message.id === threadRootId)
    || null;
  const threadMessages = messages.filter((message) => message.thread_id === threadRootId);
  const decisionMessages = [...decisionInboxMessages, ...mainMessages.filter((message) => isDecisionEvent(message))];
  const activityMessages = mainMessages.filter(isOrganizationEvent);
  const failedRequests = state?.failed_requests || [];
  const mainFailedRequest = failedRequests.find((request) => !request.thread_id);
  const queuedRequests = Number(state?.working?.queued_requests || 0);
  const executionDecisionCount = executionJobs.filter((job) => job.startable === true).length;
  const visibleMessages = view === "decisions" ? decisionMessages : view === "activity" ? activityMessages : mainMessages;
  const conversationBlocks = groupConversation(visibleMessages);
  const visibleExecutionJobs = view === "decisions"
    ? executionJobs.filter((job) => job.startable === true)
    : executionJobs;
  const conversationRows = [
    ...conversationBlocks.map((block, index) => ({ kind: "conversation" as const, block, at: Number(block.kind === "events" ? block.messages[0]?.at : block.message.at) || 0, tie: `m-${index}` })),
    ...visibleExecutionJobs.map((job, index) => ({ kind: "execution" as const, plan: job, at: executionTimeOf(job.created_at, executionTimeOf(job.timeline?.[0]?.at)), tie: `e-${job.job_id || index}` })),
  ].sort((left, right) => left.at - right.at || left.tie.localeCompare(right.tie));
  const activityRows = [
    ...conversationBlocks.map((block, index) => ({ kind: "conversation" as const, block, at: Number(block.kind === "events" ? block.messages[0]?.at : block.message.at) || 0, tie: `m-${index}` })),
    ...executionJobs.map((job, index) => ({ kind: "execution" as const, plan: job, at: executionTimeOf(job.created_at, executionTimeOf(job.timeline?.[0]?.at)), tie: `e-${job.job_id || index}` })),
  ].sort((left, right) => left.at - right.at || left.tie.localeCompare(right.tie));
  const projectSummary = state?.project_summary?.summary || state?.organization?.summary || state?.summary || "";

  const projectTitle = textOf(state?.project_summary?.title || state?.project_summary?.name || runtime?.pack?.product_name || runtime?.pack?.dataset_id, "Organization");
  const readyMessage = [...mainMessages].reverse().find(isReadyMessage);
  const conversationMeetingTitles = new Set(mainMessages.filter((message) => message.kind === "meeting_invitation").map((message) => message.meeting?.title).filter(Boolean));
  const pendingMeetingInvitations = (state?.meeting_invitations || []).filter((meeting) => !conversationMeetingTitles.has(meeting.title));

  const latestHuman = [...messages].reverse().find((message) => message.role === "human");
  const optimisticInFlight = [...optimisticHumanMessages].reverse().find((message) =>
    message.delivery_status === "sending" && String(message.thread_id || "") === String(requestScope || ""));
  const routeWorkingState: P3WorkingState | undefined = busy && !state?.busy && optimisticInFlight
    ? {
        phase: "awaiting_model",
        summary: "Waiting for the liaison model to interpret your request.",
        elapsed_seconds: Math.max(0, Date.now() / 1000 - Number(optimisticInFlight.at || Date.now() / 1000)),
        thread_id: requestScope,
      }
    : undefined;
  const visibleWorkingState = state?.busy ? state.working : routeWorkingState;
  const modelBusyThread = (state?.busy ? latestHuman?.thread_id : requestScope) || "";
  const serverConnected = Boolean(state && runtime && !stateError && !runtimeError);
  const visibleError = error || stateError || runtimeError;

  return <div className={`liaison-app ${organizationOpen || threadRoot || resourceTarget ? "details-visible" : ""}`}>
    <aside className="left-rail" aria-label="Secretary navigation">
      <div className="rail-brand"><span className="brand-mark">S</span><div><strong>Secretary</strong><span>{projectTitle}</span></div><button className="rail-search" aria-label="Focus secretary composer" onClick={() => document.querySelector<HTMLTextAreaElement>(".composer-card textarea")?.focus()}>⌕</button></div>
      <nav className="experience-switch" aria-label="Experience version">
        <span>Experience</span>
        <a href="/org/seat">P2 Transparent</a>
        <a className="active" aria-current="page" href="/org/liaison">P3 Secretary</a>
      </nav>
      <nav className="rail-nav">
        <button className={`rail-nav-item ${view === "conversation" ? "active" : ""}`} aria-current={view === "conversation" ? "page" : undefined} onClick={() => setNavigation("conversation")}><span aria-hidden="true">◌</span><span className="rail-nav-copy">Conversation</span><span className="rail-count">{mainMessages.length + executionJobs.length}</span></button>
        <button className={`rail-nav-item ${view === "decisions" ? "active" : ""}`} aria-current={view === "decisions" ? "page" : undefined} onClick={() => setNavigation("decisions")}><span aria-hidden="true">♧</span><span className="rail-nav-copy">Decisions</span><span className="rail-count">{decisionMessages.length + mainDrafts.length + meetingPlans.length + executionDecisionCount}</span></button>
        <button className={`rail-nav-item ${view === "activity" ? "active" : ""}`} aria-current={view === "activity" ? "page" : undefined} onClick={() => setNavigation("activity")}><span aria-hidden="true">⌁</span><span className="rail-nav-copy">Activity</span><span className="rail-count">{activityMessages.length + executionJobs.length}</span></button>
      </nav>
      <div className="rail-section-label">Projects</div>
      <div className="sidebar-projects">
        <button className="project-item active" aria-current="page" onClick={() => setNavigation("conversation")}><span className="project-icon" aria-hidden="true">□</span><span>{projectTitle}</span></button>
      </div>
      <div className="rail-footer"><span className={`human-presence ${!serverConnected ? "offline" : runtime?.running ? "live" : "paused"}`} /> <span>{!runtime && !runtimeError ? "Connecting…" : !serverConnected ? "Service disconnected" : runtime?.running ? "Simulation live" : "Simulation paused"}</span><strong>Victor</strong></div>
    </aside>
    <div className="app-main">
      <header className="project-header">
        <div className="project-heading"><span className="folder-icon" aria-hidden="true">□</span><div><strong>{projectTitle}</strong><span>P3 Secretary liaison · Victor's seat</span></div></div>
        <div className="project-header-actions"><RuntimeControl runtime={runtime} connected={serverConnected} busy={runtimeBusy} onToggle={toggleRuntime} /><button className={`details-toggle ${organizationOpen ? "active" : ""}`} aria-label={organizationOpen ? "Hide organization details" : "Show organization details"} aria-expanded={organizationOpen} onClick={toggleOrganization}><SlidersIcon /><span className="sr-only">{organizationOpen ? "Hide organization" : "Show organization"}</span></button></div>
      </header>
      <main className="conversation-shell" aria-label="Secretary liaison thread">
        <div className="conversation-scroll"><div className="conversation-column">
          <ProjectContext summary={projectSummary} />
          {view !== "activity" && pendingMeetingInvitations.map((meeting, index) => <MeetingInvitationTurn key={`pending-meeting-${meeting.title || index}`} message={{ ...meeting, kind: "meeting_invitation", text: "", status: meeting.status, meeting }} token={token} onCompose={composeDecision} onOpenResource={openResource} />)}
          {(view === "activity" ? activityRows : conversationRows).map((row, index) => row.kind === "execution"
            ? <ExecutionJobTurn job={row.plan} busy={busy} onStart={() => settleExecution(row.plan.job_id || "", "start")} onCancel={() => settleExecution(row.plan.job_id || "", "cancel")} onOpenResource={openResource} key={`execution-${row.plan.job_id || index}`} />
            : row.block.kind === "events"
              ? <OrganizationDigest messages={row.block.messages} tone={row.block.tone} key={`digest-${row.block.messages[0]?.at || index}`} token={token} onOpenResource={openResource} />
              : <ThreadTurn message={row.block.message} thread={row.block.message.id ? threadByRoot.get(row.block.message.id) : undefined} onOpenThread={openThread} onOpenResource={openResource} onComposeDecision={composeDecision} key={row.block.message.id || `${row.block.message.at || "turn"}-${index}`} token={token} />)}
          {view !== "activity" && meetingPlans.map((plan) => <MeetingPlanTurn key={plan.plan_id} plan={plan} busy={busy} onConfirm={() => settleMeetingPlan(plan.plan_id, "confirm")} onDiscard={() => settleMeetingPlan(plan.plan_id, "discard")} token={token} onOpenResource={openResource} />)}
          {view !== "activity" && mainDrafts.map((draft: P3Draft) => <DraftTurn key={draft.draft_id} draft={draft} busy={busy} onConfirm={() => settle(draft.draft_id, "confirm")} onDiscard={() => settle(draft.draft_id, "discard")} />)}
          {view === "conversation" && readyMessage && <ReadyTurn message={readyMessage} />}
          {view === "conversation" && mainFailedRequest && <FailedRequestTurn request={mainFailedRequest} busy={busy} onRetry={() => retryFailedRequest("")} />}
          {!visibleMessages.length && !(mainDrafts.length || pendingMeetingInvitations.length || meetingPlans.length || visibleExecutionJobs.length) && <EmptyView view={view} />}
          {(state?.busy || optimisticInFlight) && !modelBusyThread && <WorkingTurn working={visibleWorkingState} fallback="Interpreting your request…" onStop={state?.busy ? stopCurrentRequest : undefined} />}
          <div ref={end} />
        </div></div>
        <div className="composer-zone">{visibleError && !threadRoot && <div className="inline-error">{visibleError}</div>}<Composer value={text} onChange={setText} onSend={ask} busy={busy} queuedRequests={queuedRequests} context={decisionContext} onClearContext={() => setDecisionContext(undefined)} ariaLabel="Message the organization secretary" placeholder="Ask a question, share a goal, or give feedback…" hint="Clarifications are welcome · consequential requests require confirmation" onError={setError} /></div>
      </main>
    </div>
    {threadRoot && <ThreadPanel root={threadRoot} messages={threadMessages} drafts={threadDrafts} executionJobs={executionJobs} failedRequest={failedRequests.find((request) => request.thread_id === threadRootId)} busy={busy} working={Boolean((state?.busy || optimisticInFlight) && modelBusyThread === threadRootId)} workingState={visibleWorkingState} text={threadText} onTextChange={setThreadText} onSend={askThread} onStop={stopCurrentRequest} onRetry={() => retryFailedRequest(threadRootId)} onConfirm={(draftId) => settle(draftId, "confirm")} onDiscard={(draftId) => settle(draftId, "discard")} onStartExecution={(jobId) => settleExecution(jobId, "start")} onCancelExecution={(jobId) => settleExecution(jobId, "cancel")} onComposeDecision={composeThreadDecision} decisionContext={threadDecisionContext} onClearDecisionContext={() => setThreadDecisionContext(undefined)} onError={setError} error={visibleError} token={token} onOpenResource={openResource} onClose={() => setThreadRootId("")} endRef={threadEnd} />}
    {resourceTarget
      ? <ResourceInspector target={resourceTarget} tab={resourceTab} payload={resourcePayload} busy={resourceBusy} error={resourceError} onTabChange={selectResourceTab} onRefresh={() => void loadResource(resourceTarget, resourceTab)} onClose={closeResource} />
      : threadRoot
        ? null
        : organizationOpen && <DetailsPanel state={state} organization={organization} messages={mainMessages} runtime={runtime} executionJob={executionJob} token={token} onOpenResource={openResource} onClose={() => setOrganizationOpen(false)} />}
  </div>;
}

function ProjectContext({ summary }: { summary: any }) {
  return <section className="fyi-turn"><div className="turn-heading fyi-heading"><span className="heading-icon" aria-hidden="true">✓</span><span>FYI</span><span className="heading-slash">/</span><strong>Current project context</strong></div><p>{textOf(summary, "The secretary is reading the current project state.")}</p><span className="turn-hint">Visible through the secretary; no action is assumed.</span></section>;
}

function EmptyView({ view }: { view: "conversation" | "decisions" | "activity" }) {
  const copy = view === "decisions"
    ? ["No decisions need your attention", "The secretary will surface a decision here when the visible organization asks for your judgment."]
    : ["No activity to show yet", "Live organization events will appear here as the simulation advances."];
  return <section className="empty-view" aria-live="polite"><span className="empty-icon" aria-hidden="true">{view === "decisions" ? "♧" : "⌁"}</span><strong>{copy[0]}</strong><p>{copy[1]}</p></section>;
}

function ExecutionJobTurn({ job, busy, onStart, onCancel, onOpenResource }: { job: P3ExecutionJob; busy: boolean; onStart: () => void; onCancel: () => void; onOpenResource?: (message: P3Message | ResourceTarget | any, tab?: ResourceTab) => void }) {
  const [threadOpen, setThreadOpen] = useState(false);
  const workers = executionWorkersOf(job);
  const timeline = executionTimelineOf(job);
  const goal = textOf(job.goal, "Execution goal not returned");
  const status = executionStatus(job);
  const resourceRef = resourceRefOf(job as any);
  return <AssistantTurn kind="execution-job-turn"><div className="execution-job-card">
    <div className="execution-job-heading"><div><span className="turn-label">Execution team</span><h2>{textOf(job.title, "Human-office execution")}</h2></div><span className={`execution-job-status ${String(job.status || "unknown").toLowerCase().replace(/[^a-z0-9_-]+/g, "-")}`}>{status}</span></div>
    <div className="execution-job-goal"><span>Goal</span><p>{goal}</p></div>
    {job.completion_criteria && <div className="execution-job-criteria"><span>Completion criteria</span><p>{job.completion_criteria}</p></div>}
    {job.claim_task && job.task_id && <p>启动后先由 Victor 认领任务，再由执行团队开始工作。</p>}
    {job.progress_interval_seconds && <p>执行期间每 {job.progress_interval_seconds % 60 === 0 ? `${job.progress_interval_seconds / 60} 分钟` : `${job.progress_interval_seconds} 秒`}在本对话汇报一次进度；结束后发送结果。</p>}
    {workers.length > 0 && <div className="execution-worker-grid">{workers.map((worker, index) => <div className="execution-worker" key={worker.run_id || `${executionLabel(worker, "worker")}-${index}`}><span className="worker-avatar">{executionLabel(worker, "W").slice(0, 1).toUpperCase()}</span><div><strong>{executionLabel(worker, "Worker")}</strong><span>{textOf(worker.role, "Role not returned")} · {worker.reused ? `persistent agent · activation ${worker.activation_number}` : "new persistent agent"}</span><p>{textOf(worker.assignment, "Assignment not returned")}</p>{worker.model_call_started_at && <span className="worker-model-call">Model call {worker.model_calls} · {textOf(worker.model_call_phase, "working with current evidence")}…</span>}{worker.report && !job.final_report && <div className="execution-worker-report"><MarkdownMessage text={worker.report} /></div>}</div><em>{textOf(worker.status, "status not returned").replace(/_/g, " ")}</em></div>)}</div>}
    {timeline.length > 0 && <div className="execution-job-latest"><span className="details-label">Latest action</span><strong>{executionItemSummary(timeline[timeline.length - 1])}</strong><span>{textOf(timeline[timeline.length - 1].status || timeline[timeline.length - 1].kind || timeline[timeline.length - 1].type || (typeof timeline[timeline.length - 1].ok === "boolean" ? (timeline[timeline.length - 1].ok ? "succeeded" : "failed") : ""), "status not returned").replace(/_/g, " ")}</span></div>}
    {job.final_report && <div className="execution-final-report"><span className="details-label">Final report</span><MarkdownMessage text={job.final_report} /></div>}
    <div className="execution-job-actions">
      {executionCanStart(job) && <button className="primary-button" disabled={busy} onClick={onStart}>Start execution team</button>}
      {executionCanCancel(job) && <button className="danger-button" disabled={busy} onClick={onCancel}>Cancel execution</button>}
      {timeline.length > 0 && <button className="thread-open-button" aria-expanded={threadOpen} onClick={() => setThreadOpen((open) => !open)}>{threadOpen ? "Hide execution thread" : `Open execution thread · ${timeline.length} updates`}</button>}
      {resourceRef && onOpenResource && <button className="resource-context-button" onClick={() => onOpenResource({ resource_ref: resourceRef }, "summary")}>Open task context</button>}
    </div>
    {job.status === "pending_confirmation" && !job.startable && job.start_block_reason && <span className="turn-hint">Waiting: {executionStartBlockText(job)}</span>}
    {threadOpen && <ExecutionThread job={job} />}
    <span className="turn-hint">This card is a projection of the active liaison execution job; no worker or action was inferred by the browser.</span>
  </div></AssistantTurn>;
}

function ExecutionThread({ job }: { job: P3ExecutionJob }) {
  const timeline = executionTimelineOf(job);
  return <div className="execution-thread" aria-label="Execution team thread"><div className="execution-thread-header"><strong>Execution thread</strong><span>{timeline.length} updates from the liaison state</span></div>{timeline.map((item, index) => <ExecutionTimelineItemView item={item} key={`${item.at || item.tick || "event"}-${index}`} />)}</div>;
}

function ExecutionTimelineItemView({ item }: { item: P3ExecutionTimelineItem }) {
  const worldTick = executionWorldTick(item);
  const evidence = executionEvidenceText(item);
  return <div className="execution-thread-event"><span className="execution-event-dot" /><div><strong>{executionItemSummary(item)}</strong><span>{textOf(item.status || item.kind || item.type || (typeof item.ok === "boolean" ? (item.ok ? "succeeded" : "failed") : ""), "status not returned").replace(/_/g, " ")}{typeof worldTick === "number" ? ` · world t${worldTick}` : ""}</span>{evidence && <small>{evidence}{item.evidence_truncated ? ` … (${item.evidence_length} characters; full source is available in the resource inspector)` : ""}</small>}</div></div>;
}

function RuntimeControl({ runtime, connected, busy, onToggle }: { runtime: P3RuntimeStatus | null; connected: boolean; busy: boolean; onToggle: () => void }) {
  const running = Boolean(runtime?.running);
  const speed = runtime?.seconds_per_tick == null ? "—" : `${runtime.seconds_per_tick}s / tick`;
  const worldTick = runtime?.world_tick ?? runtime?.ticks_run ?? "—";
  const pack = runtime?.pack?.id || runtime?.pack?.dataset_id || "unbound";
  const secretary = runtime?.llm?.mode === "model_backed" ? "Secretary LLM connected" : "Secretary rule/template";
  const actionMode = runtime?.action_selection_mode || "unknown action selection";
  const status = !runtime ? "Connecting…" : !connected ? "Service disconnected" : running ? "Simulation live" : "Simulation paused";
  const tickLabel = connected ? "world t" : "last loaded world t";
  return <div className={`runtime-control ${!connected ? "is-disconnected" : running ? "is-running" : "is-paused"}`} aria-label="Simulation runtime status">
    <div className="runtime-summary"><span className="runtime-dot" aria-hidden="true" /><span><strong>{status}</strong><small>{pack} · {tickLabel}{worldTick} · {speed}</small>{(runtime?.last_error || runtime?.session_last_error) && <small className="runtime-error" role="status" title={runtime.last_error || runtime.session_last_error || ""}>Error: {runtime.last_error || runtime.session_last_error}</small>}</span></div>
    <button className="runtime-toggle" disabled={!runtime || !connected || busy} onClick={onToggle}>{busy ? "…" : running ? "Pause" : "Resume"}</button>
    <span className="runtime-source">{connected ? `live OrgWorld events · ${secretary} · organization actions: ${actionMode}` : "Last loaded runtime metadata · waiting for service heartbeat"}</span>
  </div>;
}

function ThreadTurn({ message, token, thread, onOpenThread, onOpenResource, onComposeDecision }: { message: P3Message; token: string; thread?: P3ConversationThread; onOpenThread?: (message: P3Message) => void; onOpenResource?: (message: P3Message, tab?: ResourceTab) => void; onComposeDecision?: (instruction: string, context?: P3DecisionContext) => void }) {
  let content: React.ReactNode;
  if (message.role === "human") content = <div className="human-turn"><div><span className="turn-label">You</span>{message.decision_context && <span className="human-reference">Replying to · {message.decision_context.title}{message.decision_context.option_label ? ` · ${message.decision_context.option_label}` : ""}</span>}<p>{message.text}</p>{message.delivery_status && <span className={`human-delivery ${message.delivery_status}`}>{message.delivery_status === "sending" ? "Sending…" : "Not sent · text preserved"}</span>}</div></div>;
  else {
  const kind = message.kind || (message.event_type ? "event" : message.status ? "status" : "message");
    if (kind === "meeting_invitation") content = <MeetingInvitationTurn message={message} token={token} onCompose={onComposeDecision} onOpenResource={onOpenResource} />;
    else if (kind === "decision_request") content = <DecisionRequestTurn message={message} token={token} onCompose={onComposeDecision} onOpenResource={onOpenResource} />;
    else if (kind === "incoming_message") content = <IncomingMessageTurn message={message} token={token} onOpenResource={onOpenResource} />;
    else if (kind === "meeting_report") content = <MeetingReportTurn message={message} token={token} onOpenResource={onOpenResource} />;
    else if (kind === "attention_preference" || kind === "meeting_instruction") content = <AttentionPolicyTurn message={message} token={token} onOpenResource={onOpenResource} />;
    else if (kind === "clarification" || message.clarification) content = <AssistantTurn kind="clarification"><div className="turn-heading clarification-heading"><span className="heading-icon" aria-hidden="true">?</span><span>Secretary</span><span className="heading-slash">/</span><strong>More context would help</strong></div><MarkdownMessage text={message.clarification || message.text} /><span className="turn-hint">Continue naturally or ask another question; this is not a decision and nothing has been routed.</span><TurnDisclosure message={message} token={token} onOpenResource={onOpenResource} /></AssistantTurn>;
    else content = <AssistantTurn kind="secretary"><span className="turn-label secretary-label">Secretary</span><MarkdownMessage text={message.text} /><TurnDisclosure message={message} token={token} onOpenResource={onOpenResource} /></AssistantTurn>;
  }
  const canThread = Boolean(message.threadable && message.id && onOpenThread);
  return <div className={`threadable-turn ${canThread ? "can-thread" : ""}`}>
    {content}
    {canThread && <button className="thread-reply-button" aria-label="Reply in thread" title="Reply in thread" onClick={() => onOpenThread?.(message)}><span aria-hidden="true">◌</span> Reply</button>}
    {thread && Number(thread.reply_count || 0) > 0 && <button className="thread-summary" onClick={() => onOpenThread?.(message)}><span className="thread-avatar" aria-hidden="true">C</span><strong>{thread.reply_count} {thread.reply_count === 1 ? "reply" : "replies"}</strong><span>{textOf(thread.latest_text, "Open thread")}</span><b aria-hidden="true">›</b></button>}
  </div>;
}

function DecisionRequestTurn({ message, token, onCompose, onOpenResource }: { message: P3Message; token: string; onCompose?: (instruction: string, context?: P3DecisionContext) => void; onOpenResource?: (message: P3Message, tab?: ResourceTab) => void }) {
  const returnedOptions = objectRows<P3DecisionOption>(message.decision_options)
    .filter((option) => typeof option.label === "string" && typeof option.instruction === "string");
  const matter = textOf(message.what || message.title || message.summary, "this pending item");
  const contextInstruction = `Explain “${matter}”, show me the recorded evidence and exact decision options, and do not take action yet.`;
  const responseInstruction = `I need to decide “${matter}”. Help me prepare a response through the secretary for my confirmation: `;
  const resourceRef = resourceRefOf(message);
  const decisionResourceTab: ResourceTab = message.preferred_resource_tab === "review" ? "review" : "summary";
  return <AssistantTurn kind="decision-needed"><div className="decision-request-card">
    <div className="turn-heading decision-heading"><span className="heading-icon" aria-hidden="true">♧</span><span>Decision needed</span><span className="heading-slash">/</span><strong>Human judgment required</strong></div>
    <dl className="decision-facts">
      <div><dt>Matter</dt><dd>{matter}</dd></div>
      <div><dt>Why this is surfaced</dt><dd>{textOf(message.why, "The organization is waiting for your judgment.")}</dd></div>
      <div><dt>Impact</dt><dd>{textOf(message.impact || message.impact_summary || message.consequence, "Impact was not returned by the liaison.")}</dd></div>
    </dl>
    {message.context && typeof message.context === "object" && <div className="decision-current-context"><strong>Current recorded context</strong><ObjectFacts value={message.context} /></div>}
    {returnedOptions.length > 0 && <div className="decision-request-actions" aria-label="Decision options">{returnedOptions.map((option) =>
      <div className="decision-option" key={option.id || option.label}><div><strong>{option.label}</strong><span>{textOf(option.instruction, "No option instruction was returned.")}</span></div><button type="button" className={option.id === "context" || option.id === "review" ? "decision-context-button" : "decision-compose-button"} onClick={() => onCompose?.(option.instruction, decisionContextOf(message, option))}>Use option</button></div>)}</div>}
    <div className="decision-starter-actions" aria-label="Decision composer actions"><button type="button" className="decision-context-button" onClick={() => onCompose?.(contextInstruction, decisionContextOf(message))}>Ask secretary for context</button><button type="button" className="decision-compose-button" onClick={() => onCompose?.(responseInstruction, decisionContextOf(message))}>Prepare a response</button></div>
    {resourceRef && onOpenResource && <button type="button" className="resource-context-button decision-resource-button" onClick={() => onOpenResource({ ...message, resource_ref: resourceRef }, decisionResourceTab)}>Open decision context</button>}
    <span className="turn-hint">Choose a starter or write naturally below. Nothing is routed until you send it; consequential actions still require confirmation.</span>
    <TurnDisclosure message={message} token={token} onOpenResource={onOpenResource} contextLabel="Open decision context" resourceTab={decisionResourceTab} />
  </div></AssistantTurn>;
}

type ComposerProps = {
  value: string;
  onChange: (value: string) => void;
  onSend: () => void;
  onError: (error: string) => void;
  busy: boolean;
  queuedRequests?: number;
  ariaLabel: string;
  placeholder: string;
  hint: string;
  context?: P3DecisionContext;
  onClearContext?: () => void;
};

const appendVoiceText = (left: string, right: string) => {
  const prefix = String(left || "").trimEnd();
  const suffix = String(right || "").trimStart();
  if (!prefix) return suffix;
  if (!suffix) return prefix;
  // English recognition chunks need a word separator. CJK transcript chunks
  // already carry their natural boundaries and should not gain spaces merely
  // because the browser restarted recognition after a pause.
  const separator = /[A-Za-z0-9]$/.test(prefix) && /^[A-Za-z0-9]/.test(suffix) ? " " : "";
  return `${prefix}${separator}${suffix}`;
};

function Composer({ value, onChange, onSend, onError, busy, queuedRequests = 0, ariaLabel, placeholder, hint, context, onClearContext }: ComposerProps) {
  const recognitionRef = useRef<any>(null);
  const restartTimerRef = useRef<number | null>(null);
  const restartAttemptRef = useRef(0);
  const keepListeningRef = useRef(false);
  const voicePrefixRef = useRef("");
  const committedTranscriptRef = useRef("");
  const cycleTranscriptRef = useRef("");
  const startCycleRef = useRef<() => void>(() => undefined);
  const [listening, setListening] = useState(false);
  const SpeechRecognitionCtor = typeof window === "undefined" ? undefined
    : (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
  const voiceSupported = Boolean(SpeechRecognitionCtor);

  const stopVoice = useCallback(() => {
    keepListeningRef.current = false;
    restartAttemptRef.current = 0;
    if (restartTimerRef.current != null) {
      window.clearTimeout(restartTimerRef.current);
      restartTimerRef.current = null;
    }
    const recognition = recognitionRef.current;
    if (recognition) {
      try { recognition.stop?.(); }
      catch {
        recognitionRef.current = null;
        setListening(false);
      }
    } else setListening(false);
  }, []);

  const scheduleRecognitionRestart = useCallback(() => {
    if (!keepListeningRef.current || restartTimerRef.current != null) return;
    const attempt = restartAttemptRef.current;
    restartAttemptRef.current += 1;
    const delay = Math.min(160 * (2 ** Math.min(attempt, 4)), 2600);
    restartTimerRef.current = window.setTimeout(() => {
      restartTimerRef.current = null;
      startCycleRef.current();
    }, delay);
  }, []);

  const startRecognitionCycle = useCallback(() => {
    if (!keepListeningRef.current || !SpeechRecognitionCtor || recognitionRef.current) return;
    const recognition = new SpeechRecognitionCtor();
    cycleTranscriptRef.current = "";
    recognitionRef.current = recognition;
    recognition.lang = navigator.language || "zh-CN";
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.onstart = () => { setListening(true); onError(""); };
    recognition.onresult = (event: any) => {
      restartAttemptRef.current = 0;
      let cycle = "";
      for (let index = 0; index < event.results.length; index += 1)
        cycle += String(event.results[index]?.[0]?.transcript || "");
      cycleTranscriptRef.current = cycle;
      const spoken = appendVoiceText(committedTranscriptRef.current, cycle);
      onChange(appendVoiceText(voicePrefixRef.current, spoken));
    };
    recognition.onerror = (event: any) => {
      const reason = String(event?.error || "voice_input_failed");
      // Chrome commonly emits no-speech before ending a recognition cycle.
      // Silence is not user intent to stop, so onend will restart this cycle.
      if (reason === "no-speech" || reason === "speech-not-recognized" || reason === "aborted") return;
      if (reason === "not-allowed" || reason === "service-not-allowed" || reason === "audio-capture" || reason === "language-not-supported") {
        keepListeningRef.current = false;
        setListening(false);
        onError(reason === "not-allowed" || reason === "service-not-allowed"
          ? "Microphone permission was denied. You can enable it in the browser site settings."
          : reason === "audio-capture"
            ? "No working microphone was found. Keyboard input is still available."
            : "The browser does not support the selected speech-recognition language.");
        return;
      }
      // Network/service interruptions are not human intent to stop. Keep the
      // explicit session active and let onend reconnect with bounded backoff.
      onError(`Voice recognition interrupted (${reason.replace(/-/g, " ")}); reconnecting…`);
    };
    recognition.onend = () => {
      if (recognitionRef.current === recognition) recognitionRef.current = null;
      const completedCycle = cycleTranscriptRef.current.trim();
      if (completedCycle) {
        committedTranscriptRef.current = appendVoiceText(
          committedTranscriptRef.current, completedCycle);
        cycleTranscriptRef.current = "";
        onChange(appendVoiceText(
          voicePrefixRef.current, committedTranscriptRef.current));
      }
      if (keepListeningRef.current) {
        scheduleRecognitionRestart();
      } else {
        setListening(false);
      }
    };
    try { recognition.start(); }
    catch (err: any) {
      recognitionRef.current = null;
      const name = String(err?.name || "");
      if (keepListeningRef.current && !["NotAllowedError", "SecurityError"].includes(name)) {
        setListening(true);
        onError("Voice recognition is reconnecting…");
        scheduleRecognitionRestart();
      } else {
        keepListeningRef.current = false;
        setListening(false);
        onError(err.message || String(err));
      }
    }
  }, [SpeechRecognitionCtor, onChange, onError, scheduleRecognitionRestart]);

  useEffect(() => { startCycleRef.current = startRecognitionCycle; }, [startRecognitionCycle]);
  useEffect(() => () => {
    keepListeningRef.current = false;
    if (restartTimerRef.current != null) window.clearTimeout(restartTimerRef.current);
    recognitionRef.current?.abort?.();
  }, []);

  const toggleVoice = () => {
    if (!voiceSupported) {
      onError("Voice input is not available in this browser. Chrome or Edge on localhost can provide it.");
      return;
    }
    if (listening || keepListeningRef.current) {
      stopVoice();
      return;
    }
    voicePrefixRef.current = value.trimEnd();
    committedTranscriptRef.current = "";
    cycleTranscriptRef.current = "";
    restartAttemptRef.current = 0;
    keepListeningRef.current = true;
    setListening(true);
    onError("");
    startRecognitionCycle();
  };

  const handleComposerKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key !== "Enter" || event.shiftKey) return;
    event.preventDefault();
    // One physical Enter press has exactly one meaning.  While recording it
    // requests the browser's final transcript; while onend is still settling
    // that transcript, repeated/extra keydown events must not send stale text.
    if (event.repeat) return;
    if (keepListeningRef.current) {
      stopVoice();
      return;
    }
    if (listening) return;
    onSend();
  };

  return <div className="composer-card">
    {context && <div className="composer-reference"><span><b>Replying to</b>{context.title}{context.option_label ? ` · ${context.option_label}` : ""}</span><button type="button" aria-label="Remove decision reference" onClick={onClearContext}>×</button></div>}
    <textarea rows={2} aria-label={ariaLabel} placeholder={placeholder} value={value} onChange={(event) => onChange(event.target.value)} onKeyDown={handleComposerKeyDown} />
    <div className="composer-footer"><span>{listening ? "Listening continuously · pauses will not stop recording · press Enter once to stop and finish the transcript; press Enter again after recording stops to send" : queuedRequests > 0 ? `${queuedRequests} request${queuedRequests === 1 ? "" : "s"} queued · the secretary will process them in order` : hint}</span><button className={`voice-button ${listening ? "listening" : ""}`} type="button" aria-label={listening ? "Stop voice input" : "Start voice input"} aria-pressed={listening} aria-disabled={!voiceSupported} title={voiceSupported ? (listening ? "Stop voice input" : "Dictate a message") : "Voice input unavailable in this browser"} disabled={busy} onClick={toggleVoice}><MicIcon listening={listening} /></button><button className="send-button" aria-label="Send" disabled={!value.trim() || busy || listening} onClick={onSend}>↑</button></div>
  </div>;
}

function MicIcon({ listening }: { listening: boolean }) {
  return <svg className="mic-icon" aria-hidden="true" viewBox="0 0 24 24">{listening
    ? <path d="M7 12h10" />
    : <><rect x="9" y="3.5" width="6" height="11" rx="3" /><path d="M6.5 11.5a5.5 5.5 0 0 0 11 0M12 17v3M9 20h6" /></>}</svg>;
}

function WorkingTurn({ working, fallback, onStop }: { working?: P3WorkingState; fallback: string; onStop?: () => void }) {
  const step = Number(working?.step || 0);
  const max = Number(working?.max_steps || 0);
  const sources = Number(working?.unique_sources || 0);
  const tools = Number(working?.tools_completed || 0);
  const duplicateCalls = Number(working?.duplicate_calls_blocked || 0);
  const queuedRequests = Number(working?.queued_requests || 0);
  const elapsedSeconds = Math.max(0, Math.floor(Number(working?.elapsed_seconds || 0)));
  const activity = working?.activity_log || [];
  const phase = textOf(working?.phase, "interpreting").replace(/_/g, " ");
  return <AssistantTurn kind="working"><div className="thinking working-progress" aria-live="polite">
    <strong>{textOf(working?.summary, fallback)}</strong>
    <span>{phase}{elapsedSeconds > 0 ? ` · ${elapsedSeconds}s` : ""}{step > 0 && max > 0 ? ` · step ${step}/${max}` : ""}{tools > 0 ? ` · ${tools} tools` : ""}{sources > 0 ? ` · ${sources} sources` : ""}{queuedRequests > 0 ? ` · ${queuedRequests} queued` : ""}</span>
    {duplicateCalls > 0 && <span className="working-recovery">Recovered from {duplicateCalls} repeated read{duplicateCalls === 1 ? "" : "s"}; synthesizing from stored evidence.</span>}
    {activity.length > 0 && <ol className="working-activity-list">{activity.map((item, index) => <li key={`${item.step || index}-${item.label || "activity"}`}><span>{textOf(item.label, "Secretary tool")}</span><small>{textOf(item.status, "completed").replace(/_/g, " ")}</small></li>)}</ol>}
    {onStop && <div className="working-actions"><button type="button" onClick={onStop} disabled={working?.phase === "stopping"}>{working?.phase === "stopping" ? "Stopping…" : "Stop current request"}</button></div>}
  </div></AssistantTurn>;
}

function FailedRequestTurn({ request, busy, onRetry }: { request: P3FailedRequest; busy: boolean; onRetry: () => void }) {
  const sources = request.sources || [];
  const invalidPlan = request.failure_kind === "invalid_model_plan";
  const hasSources = Number(request.source_count || 0) > 0;
  return <AssistantTurn kind="failed-request"><div className="failed-request-card" role="alert">
    <span className="turn-label">{invalidPlan ? "Plan rejected · no action ran" : hasSources ? "Model interrupted · evidence preserved" : "Request interrupted · input preserved"}</span>
    <h3>{invalidPlan ? "Retry the allocation against current organization state" : hasSources ? "Resume this request without starting over" : "Retry this request"}</h3>
    {invalidPlan
      ? <p>The generated allocation did not pass structural validation. Your original instruction is preserved, and retrying will plan again from the current eligible tasks and members.</p>
      : hasSources
        ? <p>The model stopped at step {request.step || 0}/{request.max_steps || 0}. {request.source_count || 0} source{request.source_count === 1 ? " was" : "s were"} kept in the request context.</p>
        : <p>Your original instruction is preserved. No source was read and no action was prepared.</p>}
    {sources.length > 0 && <ul>{sources.map((source, index) => <li key={source.observation_id || index}><code>{textOf(source.tool, "source")}</code> · {textOf(Object.values(source.args || {})[0], "current state")}</li>)}</ul>}
    <small>{textOf(request.error, "The model request failed.")}</small>
    <button className="primary-button" type="button" disabled={busy} onClick={onRetry}>{invalidPlan ? "Retry allocation" : hasSources ? `Resume from step ${request.next_step || 1}` : "Retry request"}</button>
  </div></AssistantTurn>;
}

function ThreadPanel({ root, messages, drafts, executionJobs, failedRequest, busy, working, workingState, text, onTextChange, onSend, onStop, onRetry, onConfirm, onDiscard, onStartExecution, onCancelExecution, onComposeDecision, decisionContext, onClearDecisionContext, onError, error, token, onOpenResource, onClose, endRef }: { root: P3Message; messages: P3Message[]; drafts: P3Draft[]; executionJobs: P3ExecutionJob[]; failedRequest?: P3FailedRequest; busy: boolean; working: boolean; workingState?: P3WorkingState; text: string; onTextChange: (value: string) => void; onSend: () => void; onStop: () => void; onRetry: () => void; onConfirm: (draftId: string) => void; onDiscard: (draftId: string) => void; onStartExecution: (jobId: string) => void; onCancelExecution: (jobId: string) => void; onComposeDecision: (instruction: string, context?: P3DecisionContext) => void; decisionContext?: P3DecisionContext; onClearDecisionContext: () => void; onError: (error: string) => void; error: string; token: string; onOpenResource: (message: P3Message | ResourceTarget | any, tab?: ResourceTab) => void; onClose: () => void; endRef: { current: HTMLDivElement | null } }) {
  // A job belongs in a Slack-style thread only when the runtime explicitly
  // associates it with that root. Jobs with no thread_id stay in the main
  // conversation and are never guessed into an arbitrary thread.
  const matchingExecutionJobs = executionJobs.filter((job) => Boolean(job.thread_id) && job.thread_id === root.id);
  const showExecutionJob = matchingExecutionJobs.length > 0;
  return <aside className="thread-panel" aria-label="Message thread">
    <div className="thread-panel-header"><div><h2>Thread</h2><span>{messages.length} {messages.length === 1 ? "reply" : "replies"}</span></div><button className="icon-button" aria-label="Close thread" onClick={onClose}>×</button></div>
    <div className="thread-panel-scroll">
      <div className={`thread-root ${root.role === "human" ? "from-human" : "from-secretary"}`}><span className="thread-root-author">{root.role === "human" ? "You" : "Secretary"}</span>{root.role === "human" ? <p>{root.text}</p> : <MarkdownMessage text={root.text} />}</div>
      {showExecutionJob && matchingExecutionJobs.map((job) => <ExecutionJobTurn key={`thread-execution-${job.job_id}`} job={job} busy={busy} onStart={() => onStartExecution(job.job_id)} onCancel={() => onCancelExecution(job.job_id)} onOpenResource={onOpenResource} />)}
      <div className="thread-divider"><span>{messages.length ? `${messages.length} ${messages.length === 1 ? "reply" : "replies"}` : "No replies yet"}</span></div>
      {messages.map((message) => <ThreadTurn message={message} token={token} onOpenResource={onOpenResource} onComposeDecision={onComposeDecision} key={message.id || `${message.at}-${message.text}`} />)}
      {drafts.map((draft) => <DraftTurn key={draft.draft_id} draft={draft} busy={busy} onConfirm={() => onConfirm(draft.draft_id)} onDiscard={() => onDiscard(draft.draft_id)} />)}
      {failedRequest && <FailedRequestTurn request={failedRequest} busy={busy} onRetry={onRetry} />}
      {working && <WorkingTurn working={workingState} fallback="Secretary is interpreting this thread…" onStop={onStop} />}
      <div ref={endRef} />
    </div>
    <div className="thread-composer-zone">{error && <div className="inline-error">{error}</div>}<Composer value={text} onChange={onTextChange} onSend={onSend} onError={onError} busy={busy} queuedRequests={Number(workingState?.queued_requests || 0)} context={decisionContext} onClearContext={onClearDecisionContext} ariaLabel="Reply in thread" placeholder="Reply…" hint="Reply in this thread · the main conversation stays uncluttered" /></div>
  </aside>;
}

function MeetingPlanTurn({ plan, busy, onConfirm, onDiscard, token, onOpenResource }: { plan: P3MeetingPlan; busy: boolean; onConfirm: () => void; onDiscard: () => void; token: string; onOpenResource?: (message: P3Message, tab?: ResourceTab) => void }) {
  const decision = String(plan.decision || "decision required").replace(/_/g, " ");
  const planMessage: P3Message = {
    ...plan,
    kind: "meeting_instruction",
    text: plan.title || "Meeting plan",
    summary: plan.title || "Meeting plan",
    evidence_refs: plan.evidence_refs,
  };
  return <AssistantTurn kind="meeting-plan"><div className="confirmation-card meeting-plan-card">
    <span className="confirmation-label">Meeting plan · one confirmation</span>
    <h2>{textOf(plan.title, "Meeting plan")}</h2>
    <div className="object-facts meeting-plan-facts">
      <div><dt>Decision</dt><dd>{decision}</dd></div>
      {plan.viewpoint && <div><dt>Exact viewpoint</dt><dd>{plan.viewpoint}</dd></div>}
      {plan.return_focus && <div><dt>Return focus</dt><dd>{plan.return_focus}</dd></div>}
    </div>
    <span className="turn-hint">This plan may contain multiple organization actions, but it is confirmed once and revalidated by the server.</span>
    <TurnDisclosure message={planMessage} token={token} onOpenResource={onOpenResource} contextLabel="Open meeting context" />
    <div className="confirmation-actions"><button className="primary-button" disabled={busy} onClick={onConfirm}>Confirm and route</button><button disabled={busy} onClick={onDiscard}>Discard</button></div>
  </div></AssistantTurn>;
}

function MeetingInvitationTurn({ message, token, onCompose, onOpenResource }: { message: P3Message; token: string; onCompose?: (instruction: string, context?: P3DecisionContext) => void; onOpenResource?: (message: P3Message, tab?: ResourceTab) => void }) {
  const meeting = message.meeting || {};
  const status = meeting.status || message.status || "decision_required";
  const title = textOf(meeting.title, "this meeting");
  return <AssistantTurn kind="meeting-invitation"><div className="meeting-card decision-card">
    <div className="turn-heading decision-heading"><span className="heading-icon" aria-hidden="true">♧</span><span>Decision needed</span><span className="heading-slash">/</span><strong>Meeting invitation</strong></div>
    <h3>{textOf(meeting.title, "Meeting invitation")}</h3>
    <div className="meeting-facts"><div><span>Agenda</span><strong>{listText(meeting.agenda, "Agenda not returned")}</strong></div><div><span>Participants</span><strong>{participantText(meeting.participants)}</strong></div><div><span>Waiting for</span><strong>{textOf(message.details || message.summary, status === "decision_required" ? "Your decision on whether to attend or delegate" : status.replace(/_/g, " "))}</strong></div></div>
    <div className="decision-starter-actions" aria-label="Meeting decision actions">
      <button type="button" className="decision-context-button" onClick={() => onCompose?.(`Before I decide whether to attend “${title}”, show me the agenda, participants, current proposal, relevant evidence, and the decision the meeting needs. Do not RSVP yet.`, decisionContextOf(message, { id: "context", label: "Ask for meeting context" }))}>Ask for meeting context</button>
      <button type="button" className="decision-compose-button" onClick={() => onCompose?.(`I will attend “${title}”. Prepare the attendance action for my confirmation.`, decisionContextOf(message, { id: "attend", label: "Attend" }))}>Attend</button>
      <button type="button" className="decision-compose-button" onClick={() => onCompose?.(`I will not attend “${title}”. First ask me what exact viewpoint the secretary should relay and what information it should bring back. Do not RSVP yet.`, decisionContextOf(message, { id: "delegate", label: "Delegate to secretary" }))}>Delegate to secretary</button>
    </div>
    <span className="turn-hint">You can ask the secretary for context first; you do not need to answer immediately.</span>
    <TurnDisclosure message={message} token={token} onOpenResource={onOpenResource} />
  </div></AssistantTurn>;
}

function IncomingMessageTurn({ message, token, onOpenResource }: { message: P3Message; token: string; onOpenResource?: (message: P3Message, tab?: ResourceTab) => void }) {
  const delivery = message.delivery_mode || message.attention_preference?.message_delivery || message.attention_preference || "not returned";
  return <AssistantTurn kind="incoming-message"><div className="incoming-card"><div className="turn-heading incoming-heading"><span className="heading-icon" aria-hidden="true">◌</span><span>Incoming message</span><span className="heading-slash">/</span><strong>Secretary screened</strong></div><p>{textOf(message.summary || message.text, "A screened message is available.")}</p><div className="message-facts"><span><b>Importance</b>{textOf(message.importance, "not returned")}</span><span><b>Urgency</b>{textOf(message.urgency, "not returned")}</span><span><b>Delivery</b>{delivery}</span></div><TurnDisclosure message={message} token={token} onOpenResource={onOpenResource} /></div></AssistantTurn>;
}

function MeetingReportTurn({ message, token, onOpenResource }: { message: P3Message; token: string; onOpenResource?: (message: P3Message, tab?: ResourceTab) => void }) {
  return <AssistantTurn kind="meeting-report"><div className="report-card"><div className="turn-heading report-heading"><span className="heading-icon" aria-hidden="true">▤</span><span>Meeting report</span><span className="heading-slash">/</span><strong>Post-meeting summary</strong></div><p>{textOf(message.summary || message.text, "The secretary returned a meeting summary.")}</p><TurnDisclosure message={message} token={token} onOpenResource={onOpenResource} /></div></AssistantTurn>;
}

function AttentionPolicyTurn({ message, token, onOpenResource }: { message: P3Message; token: string; onOpenResource?: (message: P3Message, tab?: ResourceTab) => void }) {
  return <AssistantTurn kind="attention-policy"><span className="turn-label secretary-label">Secretary</span><p>{textOf(message.summary || message.text, "The secretary updated the attention policy.")}</p><div className="policy-inline">Current delivery: {message.delivery_mode || "secretary triage"}</div><TurnDisclosure message={message} token={token} onOpenResource={onOpenResource} /></AssistantTurn>;
}

function OrganizationDigest({ messages, tone, token, onOpenResource }: { messages: P3Message[]; tone: EventTone; token: string; onOpenResource?: (message: P3Message, tab?: ResourceTab) => void }) {
  const hasDecision = tone === "decision";
  const hasExecution = tone === "execution";
  const digestClass = hasDecision ? "decision-digest" : hasExecution ? "execution-digest" : "fyi-digest";
  return <AssistantTurn kind="organization-digest">
    <div className={`digest-card ${digestClass}`}>
      <div className="digest-heading"><div><span className="heading-icon" aria-hidden="true">{hasDecision ? "♧" : hasExecution ? "⌘" : "✓"}</span><span className="turn-label">{hasDecision ? "Decision needed" : hasExecution ? "Execution team progress" : "FYI"}</span><span className="heading-slash">/</span><strong>{hasDecision ? "Human judgment required" : hasExecution ? "Visible work in progress" : "Organization handled it"}</strong><span className="digest-count">{messages.length} {messages.length === 1 ? "update" : "updates"}</span></div><span className="digest-scope">visible activity</span></div>
      <ol className="digest-list">
        {messages.map((message, index) => <DigestItem message={message} token={token} onOpenResource={onOpenResource} key={`${message.at || "update"}-${index}`} />)}
      </ol>
    </div>
  </AssistantTurn>;
}

function DigestItem({ message, token, onOpenResource }: { message: P3Message; token: string; onOpenResource?: (message: P3Message, tab?: ResourceTab) => void }) {
  const label = textOf(message.event_type || message.kind, "organization update").replace(/_/g, " ");
  const status = message.status || message.task_status;
  const statusClass = status?.toLowerCase().replace(/[^a-z0-9_-]+/g, "-");
  const summary = textOf(message.summary || message.text || message.details, "Organization activity recorded");
  return <li className="digest-item">
    <div className="digest-item-heading"><span className="digest-dot" aria-hidden="true">•</span><span className="digest-type">{label}</span>{typeof message.tick === "number" && <span className="digest-tick">t{message.tick}</span>}{status && <span className={`execution-status ${statusClass}`}>{status.replace(/_/g, " ")}</span>}</div>
    <p>{summary}</p>
    <TurnDisclosure message={message} token={token} onOpenResource={onOpenResource} />
  </li>;
}

function TurnDisclosure({ message, token, onOpenResource, contextLabel = "Open full context", resourceTab = "summary" }: { message: P3Message; token: string; onOpenResource?: (message: P3Message | ResourceTarget | any, tab?: ResourceTab) => void; contextLabel?: string; resourceTab?: ResourceTab }) {
  const [level, setLevel] = useState<"closed" | "evidence" | "trace">("closed");
  const [payload, setPayload] = useState<any>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const ref = evidenceRefOf(message);
  const traceRef = message.trace_ref || ref;
  const inlineDetails = message.details || (message.summary && message.summary !== message.text ? message.summary : "");
  const hasDetails = Boolean(ref || inlineDetails);
  const hasTrace = Boolean(traceRef);
  const reveal = async () => {
    if ((!ref && !hasDetails) || busy) return;
    setBusy(true); setError("");
    try { if (level === "closed") { setPayload(ref ? await liaisonApi.evidence(token, ref) : { evidence: { title: "Details", why: inlineDetails } }); setLevel("evidence"); } else if (level === "evidence" && hasTrace) { setPayload(await liaisonApi.trace(token, traceRef)); setLevel("trace"); } else { setLevel("closed"); setPayload(null); } }
    catch (err: any) { setError(err.message || String(err)); }
    finally { setBusy(false); }
  };
  const body = payload?.evidence || payload?.trace;
  const messageHeadline = textOf(message.summary || message.text);
  const evidenceHeadline = textOf(body?.title);
  const eventRecord = body?.event && typeof body.event === "object" && !Array.isArray(body.event) ? body.event : null;
  const eventSummary = body?.event_summary && typeof body.event_summary === "object" ? body.event_summary : null;
  const evidenceFacts = eventRecord ? {
    actor: eventSummary?.actor || eventRecord.actor || eventRecord.agent_id,
    action: eventSummary?.action || eventRecord.action || eventRecord.subtype || eventRecord.type,
    organization_tick: eventSummary?.organization_tick ?? eventRecord.organization_tick ?? eventRecord.tick,
    status: body.status,
    source: body.source?.kind,
  } : null;
  const relatedObjects = objectRows<Record<string, any>>(body?.related_current_visible_objects);
  const sourceRef = resourceRefOf(message) || resourceRefOf(payload as any);
  return <div className="turn-disclosure">
    <button className="text-action" disabled={(!ref && !hasDetails) || busy} onClick={reveal}>{(!ref && !hasDetails) ? "No details" : busy ? "Loading…" : level === "closed" ? "Show details" : level === "evidence" ? (hasTrace ? "Show trace" : "Hide details") : "Hide trace"}</button>
    {sourceRef && onOpenResource && <button className="resource-context-button" type="button" disabled={busy} onClick={() => onOpenResource({ resource_ref: sourceRef }, resourceTab)}>{contextLabel}</button>}
    {level === "closed" && message.details && <span className="detail-hint">Details available</span>}
    {body && level === "evidence" && <div className="details-card evidence-context-card">
      <span className="details-label">Evidence</span>
      {evidenceHeadline && evidenceHeadline !== messageHeadline && <strong>{evidenceHeadline}</strong>}
      <p>{textOf(body.why || body.summary || body.status, "No further explanation was returned.")}</p>
      {evidenceFacts && <><h4>Event at the time</h4><ObjectFacts value={evidenceFacts} /></>}
      {eventRecord && <><h4>Complete seat-visible event record</h4><ObjectFacts value={eventRecord} /></>}
      {relatedObjects.length > 0 && <div className="related-visible-objects"><h4>Related objects visible now</h4>{relatedObjects.map((item, index) => <section key={`${item.kind || "object"}-${item.id || index}`}><strong>{textOf(item.title || item.id, "Visible object")}</strong><ObjectFacts value={item} /></section>)}</div>}
      {sourceRef && onOpenResource && <button className="resource-context-button resource-context-button-full" type="button" onClick={() => onOpenResource({ resource_ref: sourceRef }, resourceTab)}>{contextLabel} in inspector</button>}
    </div>}
    {body && level === "trace" && <div className="details-card trace">
      <span className="details-label">Organizational trace</span>
      <ol>{Array.isArray(body.steps) ? body.steps.map((step: any, index: number) => { const facts = step && typeof step === "object" ? Object.fromEntries(Object.entries(step).filter(([key]) => key !== "summary" && key !== "text")) : {}; return <li key={index}><strong>{textOf(step?.summary || step?.text || step, "Visible organizational event")}</strong>{Object.keys(facts).length > 0 && <ObjectFacts value={facts} />}</li>; }) : <li>No trace steps were returned.</li>}</ol>
      {sourceRef && onOpenResource && <button className="resource-context-button resource-context-button-full" type="button" onClick={() => onOpenResource({ resource_ref: sourceRef }, resourceTab)}>{contextLabel} in inspector</button>}
    </div>}
    {error && <span className="disclosure-error">{error}</span>}
  </div>;
}

const resourceValue = (payload: P3ResourceResponse, tab: ResourceTab) => {
  const section = RESOURCE_SECTION_BY_TAB[tab];
  const availability = payload.availability?.[section];
  if (availability?.available === false) return availability;
  if (tab === "evidence") return {
    domain_chain: payload.content,
    projection_provenance: payload.provenance,
    linked_records: payload.links,
    observed_at: payload.observed_at,
  };
  return payload.content;
};
const resourceData = (value: any) => value && typeof value === "object" && !Array.isArray(value) && Object.prototype.hasOwnProperty.call(value, "data") ? value.data : value;
const resourceUnavailableReason = (value: any, fallback: string) => {
  if (value && typeof value === "object" && (value.available === false || value.availability === "unavailable")) {
    const reason = String(value.reason || "");
    const explanations: Record<string, string> = {
      no_visible_task_linked_file_content: "This task has no seat-visible linked file content. Open Summary to inspect its related PRs, issues, and organization events.",
      no_domain_recorded_task_unified_diff: "No exact unified diff has been recorded for this task yet. The inspector does not synthesize a diff.",
      no_visible_task_linked_pull_request: "No seat-visible pull request is linked to this task yet.",
      no_visible_task_linked_ci_run: "No seat-visible CI run is linked to this task yet.",
    };
    return explanations[reason] || reason || fallback;
  }
  return fallback;
};

function ResourceInspector({ target, tab, payload, busy, error, onTabChange, onRefresh, onClose }: { target: ResourceTarget; tab: ResourceTab; payload: P3ResourceResponse | null; busy: boolean; error: string; onTabChange: (tab: ResourceTab) => void; onRefresh: () => void; onClose: () => void }) {
  const [copied, setCopied] = useState(false);
  const resource = payload?.resource || payload || {};
  const value = resourceValue(payload || {}, tab);
  const data = resourceData(value);
  const linkedJobs = objectRows<P3ExecutionJob>(payload?.execution_context?.jobs);
  const secretaryReports = objectRows<any>(payload?.secretary_context?.reports);
  const secretaryEvidence = objectRows<any>(payload?.secretary_context?.evidence_records);
  const workerEvidence = linkedJobs.flatMap((job) => objectRows<any>(job.evidence_records));
  const sourceEvidence = [...secretaryEvidence, ...workerEvidence];
  const evidenceFileGroups = new Map<string, { path: string; content: string; evidence_id?: string; workers: string[] }>();
  sourceEvidence.filter((row) => row.tool === "read_repo" && row.args?.path).forEach((row) => {
    const path = String(row.args.path);
    const content = textOf(row.source_text);
    const key = `${path}\u0000${content}`;
    const worker = textOf(row.worker_name, "Secretary");
    const existing = evidenceFileGroups.get(key);
    if (existing) {
      if (!existing.workers.includes(worker)) existing.workers.push(worker);
    } else {
      evidenceFileGroups.set(key, { path, content, evidence_id: row.evidence_id, workers: [worker] });
    }
  });
  const evidenceFiles = [...evidenceFileGroups.values()].map(({ workers, ...file }) => ({
    ...file,
    source: `${workers.join(" + ")} · source evidence`,
  }));
  const evidenceTests = sourceEvidence.filter((row) => row.tool === "run_tests" || (row.tool === "read_repo" && /(^|[\\/])tests?([\\/]|$)|test_/i.test(String(row.args?.path || ""))));
  const copy = async (text: string) => {
    if (!text || !navigator.clipboard) return;
    try { await navigator.clipboard.writeText(text); setCopied(true); window.setTimeout(() => setCopied(false), 1200); }
    catch { setCopied(false); }
  };
  const title = textOf(resource.title || target.title, "Resource");
  const unavailable = (reason: string) => <div className="resource-empty" role="status"><strong>Unavailable</strong><span>{reason}</span></div>;
  const renderSummary = () => {
    if (!value || (typeof value === "object" && value.available === false)) return unavailable(resourceUnavailableReason(value, "The source overview is not currently available to this seat."));
    const overview = typeof data === "object" && data ? data : { summary: data };
    const narrative = textOf(overview.summary || overview.description || overview.body || overview.text, "");
    return <div className="resource-pane-content"><div className="resource-summary-title"><span className="turn-label">{textOf(resource.kind || target.kind, "visible resource")}</span><h3>{title}</h3>{resource.status && <span className="execution-status">{String(resource.status).replace(/_/g, " ")}</span>}</div>{narrative ? <MarkdownMessage text={narrative} /> : <p className="detail-muted">Current structured source record.</p>}<ObjectFacts value={overview} />{secretaryReports.map((report) => <section className="resource-execution-summary" key={report.message_id}><span className="turn-label">Secretary investigation · {objectRows(report.evidence_records).length} sources</span><MarkdownMessage text={textOf(report.text, "No report text returned.")} /></section>)}{linkedJobs.map((job) => <section className="resource-execution-summary" key={job.job_id}><span className="turn-label">Execution team · {executionStatus(job)}</span><h4>{textOf(job.title, "Linked execution")}</h4>{job.final_report ? <MarkdownMessage text={job.final_report} /> : <p>{textOf(job.goal, "Execution report pending")}</p>}</section>)}</div>;
  };
  const renderReview = () => {
    if (!value || (typeof value === "object" && (value.available === false || value.availability === "unavailable"))) return unavailable(resourceUnavailableReason(value, "Review details were not returned by the liaison resource projection."));
    if (typeof data === "string") return <div className="resource-pane-content"><p>{data}</p></div>;
    const comments = Array.isArray(data?.comments) ? data.comments : [];
    if (!Object.prototype.hasOwnProperty.call(data || {}, "comments") &&
        !Object.prototype.hasOwnProperty.call(data || {}, "reviewed")) {
      return <div className="resource-pane-content"><ObjectFacts value={data || {}} /></div>;
    }
    const reviewFacts = { reviewed: data?.reviewed, approved_by: data?.approved_by, requested_changes: data?.requested_changes, recording_note: data?.recording_note };
    return <div className="resource-pane-content"><ObjectFacts value={reviewFacts} />{comments.length ? <div className="resource-list">{comments.map((comment: any, index: number) => <div className="resource-list-row" key={comment.review_id || index}><strong>{textOf(comment.reviewer, "Reviewer")} · {comment.approve ? "approved" : "requested changes"}{comment.tick != null ? ` · t${comment.tick}` : ""}</strong><MarkdownMessage text={textOf(comment.comment, "No review comment was recorded.")} className="resource-review-comment" /></div>)}</div> : <div className="resource-empty" role="status"><strong>No recorded comments</strong><span>{textOf(data?.recording_note, "No review comments were returned.")}</span></div>}</div>;
  };
  const renderFiles = () => {
    if ((!value || (typeof value === "object" && (value.available === false || value.availability === "unavailable"))) && !evidenceFiles.length) return unavailable(resourceUnavailableReason(value, "File content is not exposed by the liaison resource projection."));
    if (typeof data === "string") return <div className="resource-code-wrap"><button className="resource-copy" type="button" onClick={() => void copy(data)}>{copied ? "Copied" : "Copy"}</button><pre className="resource-code">{data}</pre></div>;
    const returnedFiles = Array.isArray(data?.files) ? data.files : Array.isArray(data) ? data : [];
    const files = [...new Map([...returnedFiles, ...evidenceFiles].map((file: any) => {
      const path = String(file.path || file.name || "");
      const content = textOf(file.content || file.new_content || "");
      return [`${path}\u0000${content}`, file];
    })).values()];
    if (!files.length) return unavailable("No files were returned for this resource.");
    return <div className="resource-file-list">{files.map((file: any, index: number) => { const content = textOf(file.content || file.new_content || "", ""); return <div className="resource-file-row" key={file.path || file.name || index}><div><strong>{textOf(file.path || file.name, "File path not returned")}</strong><span>{textOf(file.source || file.status || file.language, "source not returned")}{file.content_hash ? ` · sha256 ${file.content_hash}` : ""}</span></div>{content && <div className="resource-code-wrap"><button className="resource-copy" type="button" onClick={() => void copy(content)}>{copied ? "Copied" : "Copy"}</button><pre className="resource-code">{content}</pre></div>}</div>; })}</div>;
  };
  const renderDiff = () => {
    if (!value || (typeof value === "object" && (value.available === false || value.availability === "unavailable"))) return unavailable(resourceUnavailableReason(value, "A unified diff is not exposed by the liaison resource projection."));
    const diffs = Array.isArray(data?.diffs) ? data.diffs : [];
    if (!diffs.length) return unavailable("No exact unified diff was returned for this resource.");
    return <div className="resource-file-list">{diffs.map((item: any, index: number) => { const diff = textOf(item.unified_diff, ""); return <div className="resource-file-row" key={item.patch_id || index}><div><strong>{textOf(item.path, "File path not returned")}</strong><span>{textOf(item.patch_id, "patch id not returned")}{item.content_hash ? ` · sha256 ${item.content_hash}` : ""}</span></div><div className="resource-code-wrap"><button className="resource-copy" type="button" onClick={() => void copy(diff)}>{copied ? "Copied" : "Copy"}</button><pre className="resource-code resource-diff">{diff}</pre></div></div>; })}</div>;
  };
  const renderTests = () => {
    if ((!value || (typeof value === "object" && (value.available === false || value.availability === "unavailable"))) && !evidenceTests.length) return unavailable(resourceUnavailableReason(value, "Test and CI details are not exposed by the liaison resource projection."));
    const runs = Array.isArray(data?.runs) ? data.runs : [];
    return <div className="resource-pane-content">{runs.length > 0 && <div className="resource-list">{runs.map((run: any, index: number) => <div className="resource-list-row" key={run.id || index}><strong>{textOf(run.id, "CI run")} · {textOf(run.status, "status not returned")}{run.created_at_tick != null ? ` · t${run.created_at_tick}` : ""}</strong><ObjectFacts value={{ commit_id: run.commit_id, checks: run.checks, failure_reasons: run.failure_reasons, tree_hash: run.tree_hash, brief: run.brief }} /></div>)}</div>}{evidenceTests.length > 0 && <section className="resource-execution-tests"><span className="turn-label">Source evidence</span>{evidenceTests.map((row, index) => <div className="resource-list-row" key={`${row.message_id || row.worker_id || "source"}-${row.evidence_id || index}`}><strong>{row.tool === "run_tests" ? "Test command executed" : "Test source inspected"} · {textOf(row.args?.path || Object.values(row.args || {})[0], "current worktree")}</strong><pre className="resource-code">{textOf(row.source_text, "No output returned")}</pre></div>)}</section>}{!runs.length && !evidenceTests.some((row) => row.tool === "run_tests") && <p className="detail-muted">No CI or test execution was recorded. Inspected test source is evidence of review, not a passing test.</p>}</div>;
  };
  const renderEvidence = () => {
    if (!value && !linkedJobs.length && !secretaryEvidence.length) return unavailable("Evidence payload was not returned by the liaison resource projection.");
    const safeEvidence = typeof data === "object" ? data : { summary: data };
    const serialized = JSON.stringify({ resource: safeEvidence, secretary_reports: secretaryReports, secretary_evidence: secretaryEvidence, execution_jobs: linkedJobs }, null, 2);
    return <div className="resource-code-wrap"><p className="detail-muted">Seat-visible evidence projection; private and unrestricted backend payloads are not exposed.</p><button className="resource-copy" type="button" onClick={() => void copy(serialized)}>{copied ? "Copied" : "Copy"}</button><pre className="resource-code">{serialized}</pre></div>;
  };
  const content = tab === "summary" ? renderSummary() : tab === "review" ? renderReview() : tab === "files" ? renderFiles() : tab === "diff" ? renderDiff() : tab === "tests" ? renderTests() : renderEvidence();
  return <aside className="details-panel resource-inspector" aria-label="Resource inspector">
    <div className="details-header resource-inspector-header"><div><span className="turn-label">Source</span><h2>{title}</h2></div><div className="resource-header-actions"><button className="icon-button" aria-label="Refresh resource" onClick={onRefresh} disabled={busy}>↻</button><button className="icon-button" aria-label="Close resource inspector" onClick={onClose}>×</button></div></div>
    <div className="resource-inspector-tabs" role="tablist" aria-label="Resource sections">{RESOURCE_TABS.map((item) => <button key={item.id} role="tab" aria-selected={tab === item.id} aria-controls={`resource-pane-${item.id}`} className={`resource-inspector-tab ${tab === item.id ? "active" : ""}`} onClick={() => onTabChange(item.id)}>{item.label}</button>)}</div>
    <div className="details-scroll resource-inspector-scroll"><section id={`resource-pane-${tab}`} role="tabpanel" aria-label={RESOURCE_TABS.find((item) => item.id === tab)?.label}>{busy ? <div className="resource-loading" role="status">Loading current seat-visible resource…</div> : error ? <div className="inline-error" role="alert">{error}</div> : !payload ? <div className="resource-loading" role="status">Loading current seat-visible resource…</div> : content}</section></div>
  </aside>;
}

function AssistantTurn({ children, kind = "assistant" }: { children: React.ReactNode; kind?: string }) { return <section className={`assistant-turn ${kind}`}><div className="assistant-avatar">C</div><div className="assistant-content">{children}</div></section>; }

function DraftTurn({ draft, busy, onConfirm, onDiscard }: { draft: P3Draft; busy: boolean; onConfirm: () => void; onDiscard: () => void }) { return <AssistantTurn kind="draft"><div className="confirmation-card"><span className="confirmation-label">Structured action · confirmation required</span><h2>{textOf(draft.label, "Proposed organizational action")}</h2>{draft.rationale && <p>{draft.rationale}</p>}{draft.action_type && <div className="action-type">Action · {draft.action_type}</div>}{(draft.display_params || draft.params) && <ObjectFacts value={draft.display_params || draft.params || {}} />}<div className="confirmation-actions"><button className="primary-button" disabled={busy} onClick={onConfirm}>Confirm and execute</button><button disabled={busy} onClick={onDiscard}>Discard</button></div></div></AssistantTurn>; }

function StructuredValue({ value, depth = 0 }: { value: any; depth?: number }) {
  if (value == null || value === "") return <span className="structured-empty">not recorded</span>;
  if (typeof value !== "object") return <span>{String(value)}</span>;
  if (depth >= 4) return <code>{JSON.stringify(value)}</code>;
  if (Array.isArray(value)) {
    if (!value.length) return <span className="structured-empty">none</span>;
    return <ul className="structured-list">{value.map((item, index) => <li key={index}><StructuredValue value={item} depth={depth + 1} /></li>)}</ul>;
  }
  const entries = Object.entries(value);
  if (!entries.length) return <span className="structured-empty">none</span>;
  return <dl className="structured-object">{entries.map(([key, item]) => <div key={key}><dt>{key.replace(/_/g, " ")}</dt><dd><StructuredValue value={item} depth={depth + 1} /></dd></div>)}</dl>;
}

function ObjectFacts({ value }: { value: Record<string, any> }) { return <dl className="object-facts">{Object.entries(value).map(([key, item]) => <div key={key}><dt>{key.replace(/_/g, " ")}</dt><dd><StructuredValue value={item} /></dd></div>)}</dl>; }

function SlidersIcon() {
  return <svg className="sliders-icon" aria-hidden="true" viewBox="0 0 24 24"><path d="M4 6h10M18 6h2M4 12h3M11 12h9M4 18h8M16 18h4" /><circle cx="16" cy="6" r="2" /><circle cx="9" cy="12" r="2" /><circle cx="14" cy="18" r="2" /></svg>;
}

function ReadyTurn({ message }: { message: P3Message }) {
  return <section className="ready-turn"><div className="ready-heading"><span className="heading-icon" aria-hidden="true">▣</span><strong>Ready to try</strong><span className="execution-status verified">{message.status || message.task_status}</span></div><p>{textOf(message.summary || message.text, "The organization returned a verified result.")}</p><span className="turn-hint">This state comes from the organization update above.</span></section>;
}

function DetailsPanel({ state, organization, messages, runtime, executionJob, token, onOpenResource, onClose }: { state: P3State | null; organization: any; messages: P3Message[]; runtime: P3RuntimeStatus | null; executionJob: P3ExecutionJob | null; token: string; onOpenResource: (message: P3Message | ResourceTarget | any, tab?: ResourceTab) => void; onClose: () => void }) {
  const [organizationDetailsOpen, setOrganizationDetailsOpen] = useState(false);
  const organizationWorkstreams = objectRows<Record<string, any>>(organization?.workstreams);
  const organizationAgents = objectRows<Record<string, any>>(organization?.agents);
  const recentActivity = objectRows<Record<string, any>>(organization?.recent_activity);
  const currentWork = organizationWorkstreams[0];
  const projectTitle = textOf(currentWork?.title || state?.project_summary?.title || state?.project_summary?.name, "Current project");
  const currentStatus = textOf(currentWork?.status || currentWork?.state, "No current work status has been returned.");
  const decisionItems = objectRows<Record<string, any>>(organization?.decision_inbox || organization?.needs_your_decision);
  const currentOwner = currentWork?.owner;
  const involved = currentOwner ? [currentOwner] : organizationAgents.slice(0, 3);
  const evidenceItems = messages.filter((message) =>
    evidenceRefOf(message) || resourceRefOf(message)).slice(-4);
  const attentionPreferences = state?.attention_preferences;
  const deliveryMode = attentionPreferences?.message_delivery || "No delivery policy returned";
  const deliveryDescription = attentionPreferences?.description || "The secretary will only use a delivery mode returned by the organization.";
  const executionAgents = executionAgentsOf(state);
  const meetingChoicesByTitle = new Map<string, { meeting: any; status?: string }>();
  for (const message of messages.filter((row) => row.kind === "meeting_invitation" && row.meeting)) {
    const meeting = message.meeting!;
    meetingChoicesByTitle.set(textOf(meeting.title, "Meeting invitation"), {
      meeting, status: meeting.status || message.status,
    });
  }
  for (const meeting of state?.meeting_invitations || []) {
    const title = textOf(meeting.title, "Meeting invitation");
    const earlier = meetingChoicesByTitle.get(title);
    const status = meeting.attendance_confirmed ? meeting.decision || meeting.status
      : meeting.decision && meeting.decision !== "pending" ? `${meeting.decision} · confirmation pending`
      : "decision required";
    meetingChoicesByTitle.set(title, { meeting: { ...(earlier?.meeting || {}), ...meeting }, status });
  }
  const pendingMeetingChoices = [...meetingChoicesByTitle.values()];
  return <aside className="details-panel" aria-label="In this conversation">
    <div className="details-header"><div><span className="details-eyebrow">Shown on request</span><h2>In this conversation</h2></div><button className="icon-button" aria-label="Close conversation details" onClick={onClose}>×</button></div>
    <div className="details-scroll">
      {!organization ? <div className="loading-line panel-loading">Loading visible organization…</div> : <>
        <div className="details-note"><span aria-hidden="true">◉</span> Details are read-only and scoped to this liaison.</div>
        <PanelSection title="Simulation"><div className="detail-value"><strong>{runtime?.running ? "Running" : "Paused"} · world t{runtime?.world_tick ?? "—"}</strong><span>Pack: {runtime?.pack?.id || "not returned"} · {runtime?.pack?.source || "source not returned"}</span><span>Engine: {runtime?.engine || "not returned"}</span><span>Events: {runtime?.event_source || "not returned"}</span><span>Secretary LLM: {runtime?.llm?.mode === "model_backed" ? "connected" : "rule/template only"}</span><span>Organization action selection: {runtime?.action_selection_mode || "not returned"}</span></div></PanelSection>
        {executionJob && <ExecutionDetails job={executionJob} token={token} />}
        {executionAgents.length > 0 && <PersistentExecutionAgents agents={executionAgents} />}
        <PanelSection title="Message delivery"><div className="detail-value"><strong>{deliveryMode}</strong><span>{deliveryDescription}</span><span>You can say “all messages” or “secretary triage” in the composer to request a change.</span></div></PanelSection>
        {pendingMeetingChoices.length > 0 && <PanelSection title="Pending meeting choice"><div className="meeting-choice-list">{pendingMeetingChoices.map(({ meeting, status }, index) => <div className="meeting-choice" key={`${meeting.title || "meeting"}-${index}`}><strong>{textOf(meeting.title, "Meeting invitation")}</strong><span>{textOf(status, "decision required").replace(/_/g, " ")}</span><small>Ask for context first; no immediate answer is required.</small></div>)}</div></PanelSection>}
        <PanelSection title="Current work"><div className="detail-value"><strong>{projectTitle}</strong><span>{currentStatus}</span></div></PanelSection>
        <PanelSection title="Status"><div className="detail-status"><span className={`status-pip ${currentWork?.status || currentWork?.state ? "active" : ""}`} />{currentWork?.status || currentWork?.state || "No status signal"}</div></PanelSection>
        <PanelSection title="Who is involved"><div className="detail-people"><span><b>V</b> Victor · human member</span>{involved.map((agent: any, index: number) => <span key={index}><b>{textOf(agent.name, "M").slice(0, 1).toUpperCase()}</b>{textOf(agent.name, "Member")} · {textOf(agent.role, "member").replace(/_/g, " ")}</span>)}</div></PanelSection>
        <PanelSection title="Why this work"><p className="detail-copy">{textOf(currentWork?.why || currentWork?.summary || organization.summary || state?.project_summary?.summary || state?.summary, "The secretary is mediating visible organizational work for Victor.")}</p></PanelSection>
        {decisionItems.length > 0 && <PanelSection title="Needs your decision"><div className="panel-decision-list">{decisionItems.map((item: any, index: number) => { const tab: ResourceTab = item.preferred_resource_tab === "review" ? "review" : "summary"; return <div className="panel-item panel-decision-item" key={index}><strong>{textOf(item.what || item.title, "Decision")}</strong><span>{textOf(item.why || item.summary, "Awaiting your judgment")}</span>{item.impact && <small>{textOf(item.impact)}</small>}{resourceRefOf(item) && <button type="button" className="resource-context-button" onClick={() => onOpenResource(item, tab)}>Open decision context</button>}</div>; })}</div></PanelSection>}
        <PanelSection title="Evidence">{evidenceItems.length ? <div className="detail-evidence-list">{evidenceItems.map((message, index) => <div className="detail-evidence-item" key={`${message.at || "evidence"}-${index}`}><strong>{textOf(message.summary || message.text, "Visible evidence")}</strong><span>{textOf(message.event_type || message.kind, "activity").replace(/_/g, " ")}{typeof message.tick === "number" ? ` · t${message.tick}` : ""}</span>{resourceRefOf(message) && <button type="button" className="resource-context-button" onClick={() => onOpenResource(message)}>Open full context</button>}</div>)}</div> : <div className="detail-value"><strong>No evidence links returned yet.</strong><span>Evidence appears here only after the organization exposes it.</span></div>}</PanelSection>
        <button className="organization-details-toggle" onClick={() => setOrganizationDetailsOpen((open) => !open)}>{organizationDetailsOpen ? "Hide organization details" : "View organization details"}<span aria-hidden="true">{organizationDetailsOpen ? "↑" : "→"}</span></button>
        {organizationDetailsOpen && <div className="organization-details"><PanelSection title="Workstreams">{organizationWorkstreams.map((item: any, index: number) => <div className="panel-item" key={index}><strong>{textOf(item.title, "Workstream")}</strong><span>{textOf(item.why || item.summary, "Visible organizational state")}</span></div>)}</PanelSection><PanelSection title="Visible agents"><div className="agent-list">{organizationAgents.map((agent: any, index: number) => <div className="agent-row" key={index}><span className={`presence ${agent.online ? "online" : ""}`} /><strong>{textOf(agent.name, "Member")}</strong><span>{textOf(agent.role, "member").replace(/_/g, " ")}</span></div>)}</div></PanelSection><PanelSection title="Recent activity">{recentActivity.map((event: any, index: number) => <div className="activity-row" key={index}><strong>{textOf(event.summary, "Visible organization activity")}</strong><span>{textOf(event.type, "activity").replace(/_/g, " ")} · {textOf(event.actor, "organization")}{typeof event.tick === "number" ? ` · t${event.tick}` : ""}</span><TurnDisclosure message={event as P3Message} token={token} onOpenResource={onOpenResource} contextLabel="Open activity context" /></div>)}</PanelSection></div>}
        <div className="panel-boundary">Read-only. Show organization is available on demand and does not reveal private interiors.</div>
      </>}
    </div>
  </aside>;
}

function ExecutionDetails({ job, token }: { job: P3ExecutionJob; token: string }) {
  const workers = executionWorkersOf(job);
  const timeline = executionTimelineOf(job);
  const reportText = textOf(job.final_report, "Final report not returned");
  return <section className="execution-details" aria-label="Execution team details">
    <PanelSection title="Execution team"><div className="detail-value"><strong>{textOf(job.title, "Human-office execution")}</strong><span>Goal: {textOf(job.goal, "Goal not returned")}</span><span>Status: {executionStatus(job)}</span><span>Source: {textOf(job.source, "liaison state")}</span>{job.completion_criteria && <span>Completion: {job.completion_criteria}</span>}{typeof job.secretary_model_calls === "number" && <span>Secretary model calls: {job.secretary_model_calls}</span>}</div></PanelSection>
    {workers.length > 0 && <PanelSection title="Workers"><div className="execution-detail-workers">{workers.map((worker, index) => <div className="execution-detail-worker" key={worker.run_id || `${executionLabel(worker, "worker")}-${index}`}><strong>{executionLabel(worker, "Worker")}</strong><span>{textOf(worker.role, "Role not returned")} · {textOf(worker.status, "status not returned")} · activation {worker.activation_number} · {worker.model_calls} calls this run / {worker.lifetime_model_calls} lifetime</span><p>{textOf(worker.assignment, "Assignment not returned")}</p>{worker.report && <MarkdownMessage text={worker.report} />}</div>)}</div></PanelSection>}
    {timeline.length > 0 && <PanelSection title="Action and evidence timeline"><div className="execution-detail-timeline">{timeline.map((item, index) => { const evidence = executionEvidenceText(item); return <div className="execution-detail-event" key={`${item.at || item.tick || "event"}-${index}`}><strong>{executionItemSummary(item)}</strong><span>{textOf(item.status || item.kind || item.type || (typeof item.ok === "boolean" ? (item.ok ? "succeeded" : "failed") : ""), "status not returned").replace(/_/g, " ")}{typeof executionWorldTick(item) === "number" ? ` · world t${executionWorldTick(item)}` : ""}</span>{evidence && <p>{evidence}{item.evidence_truncated ? ` … (${item.evidence_length} characters; full source is available under Files or Evidence)` : ""}</p>}{item.evidence_refs?.[0] && <TurnDisclosure token={token} message={{ text: executionItemSummary(item), summary: executionItemSummary(item), evidence_refs: item.evidence_refs }} />}</div>; })}</div></PanelSection>}
    <PanelSection title="Final report"><div className="execution-final-report"><MarkdownMessage text={reportText} /></div></PanelSection>
  </section>;
}

function PersistentExecutionAgents({ agents }: { agents: P3ExecutionAgent[] }) {
  return <PanelSection title="Your execution department"><div className="persistent-agent-list">{agents.map((agent) => {
    const status = textOf(agent.status, "inactive").replace(/_/g, " ");
    const report = agent.recent_reports?.[agent.recent_reports.length - 1];
    return <div className="persistent-agent-card" key={agent.worker_id}>
      <span className={`persistent-agent-presence ${status === "active" ? "active" : ""}`} />
      <div><strong>{executionLabel(agent, "Working agent")}</strong><span>{textOf(agent.role, "generalist")} · {status === "inactive" ? "dormant" : status}</span><p>{textOf(agent.last_assignment, "Created and ready for its first assignment.")}</p>{report && <div className="persistent-agent-report"><small>Last report</small><MarkdownMessage text={report} /></div>}</div>
      <em>{agent.activation_count} activation{agent.activation_count === 1 ? "" : "s"}<br />{agent.model_calls} model calls</em>
    </div>;
  })}</div><p className="persistent-agent-note">Inactive agents remain available with provenance-marked prior-run context. The Secretary reuses one by stable identity for revisions and related work, then rechecks live state before acting.</p></PanelSection>;
}

function PanelSection({ title, children }: { title: string; children: React.ReactNode }) { return <section className="panel-section"><h3>{title}</h3>{children}</section>; }
function VictorBootstrap({ booting, error, onRetry }: { booting: boolean; error: string; onRetry: () => void }) { return <main className="bootstrap-shell"><div className="codex-mark">S</div><h1>Opening the organization secretary</h1><p>Connecting the liaison to Victor's seat…</p>{error && <><div className="inline-error">{error}</div><button disabled={booting} onClick={onRetry}>Retry Victor session</button></>}</main>; }
