import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, SeatView } from "./api";
import { organizationData } from "./OrganizationOverview";

type Open = (objectId: string, object?: any) => void;
type Msg = { role: string; text: string; at: number; tool: string };
type Draft = {
  draft_id: string;
  action_type: string;
  label: string;
  params: Record<string, any>;
  rationale: string;
};

const POLL_MS = 900;
const textOf = (value: any, fallback = "") =>
  value == null || value === "" ? fallback : typeof value === "string" ? value : String(value);

/**
 * P2's only default natural-language input. Answers and drafts come from the
 * server-side liaison session, so the browser cannot invent an action or skip
 * draft -> confirm -> current-state gateway validation.
 */
export function LiaisonPanel({ token, view, onActed, open }: {
  token: string;
  view: SeatView;
  onActed: () => void;
  open: Open;
}) {
  const [messages, setMessages] = useState<Msg[]>([]);
  const [drafts, setDrafts] = useState<Draft[]>([]);
  const [busy, setBusy] = useState(false);
  const [text, setText] = useState("");
  const [error, setError] = useState("");
  const [openTool, setOpenTool] = useState<number | null>(null);
  const [showActivity, setShowActivity] = useState(false);
  const bottom = useRef<HTMLDivElement | null>(null);
  const { brief, workstreams, blockers } = organizationData(view);

  const poll = useCallback(async () => {
    const state = await api.agentState(token, 0);
    if (state.error) {
      setError(state.error);
      return;
    }
    setMessages(state.messages || []);
    setDrafts(state.drafts || []);
    setBusy(Boolean(state.busy));
    setError(state.error || "");
  }, [token]);

  useEffect(() => {
    poll();
    const timer = window.setInterval(poll, POLL_MS);
    return () => window.clearInterval(timer);
  }, [poll]);

  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages.length, drafts.length]);

  const recentMessages = useMemo(() => view.feed.threads.flatMap((thread: any) =>
    (thread.messages || []).slice(-2).map((message: any) => ({ ...message, channel: thread.channel })),
  ).slice(-8), [view.feed.threads]);

  const send = async () => {
    const request = text.trim();
    if (!request || busy) return;
    setText("");
    setError("");
    setBusy(true);
    const result = await api.agentSend(token, request);
    if (result.error) setError(result.error);
    await poll();
  };

  const confirm = async (draftId: string) => {
    setError("");
    const result = await api.agentConfirm(token, draftId);
    if (result.error) setError(result.error);
    await poll();
    await onActed();
  };

  const discard = async (draftId: string) => {
    setError("");
    const result = await api.agentDiscard(token, draftId);
    if (result.error) setError(result.error);
    await poll();
  };

  const initialSummary = textOf(
    brief.summary,
    workstreams.length
      ? `I can see ${workstreams.length} active workstream${workstreams.length === 1 ? "" : "s"}${blockers.length ? ` and ${blockers.length} blocker${blockers.length === 1 ? "" : "s"}` : ""}.`
      : "I can explain the current seat-visible organization state and cite its sources.",
  );

  return (
    <div className="pane liaison-pane">
      <div className="liaison-head">
        <div>
          <div className="liaison-title">Human–Organization Liaison</div>
          <div className="liaison-subtitle">Transparent mode · public organization state</div>
        </div>
        <span className="liaison-badge">P2</span>
      </div>
      <div className="liaison-boundary">
        <strong>Interpret · summarize · route — never govern</strong>
        <span>Consequential actions always need your confirmation.</span>
      </div>
      <div className="liaison-stream">
        <div className="liaison-context">
          <span className="context-mark">◎</span>
          <span>{initialSummary} Private agent interiors stay private.</span>
        </div>

        {messages.map((message, index) => message.role === "tool" ? (
          <div className="tool-call" key={`${message.at}-${index}`}>
            <button onClick={() => setOpenTool(openTool === index ? null : index)}>
              {openTool === index ? "▾" : "▸"} Evidence · {message.tool || "seat-visible state"}
            </button>
            {openTool === index && <pre>{message.text}</pre>}
          </div>
        ) : (
          <div className={`liaison-message ${message.role === "human" ? "human" : "liaison"}`}
               key={`${message.at}-${index}`}>
            <div className="liaison-message-label">{message.role === "human" ? "You" : "Liaison"}</div>
            <div>{message.text}</div>
          </div>
        ))}
        {busy && <div className="liaison-message liaison thinking">Interpreting the current organization state…</div>}

        {drafts.map((draft) => (
          <div className="liaison-confirm" role="alert" key={draft.draft_id}>
            <div className="confirm-kicker">Confirmation required</div>
            <div className="confirm-intent">{draft.label}</div>
            {draft.rationale && <div className="confirm-consequence">{draft.rationale}</div>}
            <div className="draft-params">
              {Object.entries(draft.params).map(([key, value]) => (
                <div className="kv" key={key}>
                  <span className="k">{key.replace(/_/g, " ")}</span>
                  <span className="v">{String(value)}</span>
                </div>
              ))}
            </div>
            <div className="confirm-actions">
              <button className="primary" onClick={() => confirm(draft.draft_id)}>Confirm and route</button>
              <button onClick={() => discard(draft.draft_id)}>Discard draft</button>
            </div>
          </div>
        ))}

        {error && <div className="error">{error}</div>}
        <div className="liaison-activity-toggle">
          <button className="link" onClick={() => setShowActivity(!showActivity)}>
            {showActivity ? "▾ Hide" : "▸ Show"} recent organization activity
          </button>
        </div>
        {showActivity && (
          <div className="liaison-activity" aria-label="Recent organization activity">
            {recentMessages.length === 0 && <div className="muted pad">No recent public messages.</div>}
            {recentMessages.map((message: any, index: number) => (
              <button className="activity-row" key={message.id || index}
                      onClick={() => message.id && open(message.id, message)}>
                <span className="activity-who">{textOf(message.sender, "org")}</span>
                <span className="activity-text">{textOf(message.text, "(empty message)")}</span>
                <span className="activity-channel">#{textOf(message.channel, "org")}</span>
              </button>
            ))}
            {(view.feed.events || []).slice(-5).reverse().map((event: any, index: number) => (
              <div className="activity-event" key={`event-${index}`}>
                <span>{textOf(event.subtype || event.type, "event")}</span>
                <span>{textOf(event.agent_id, "org")}</span>
              </div>
            ))}
          </div>
        )}
        <div ref={bottom} />
      </div>
      <div className="liaison-composer">
        <textarea
          rows={2}
          aria-label="Ask the human organization liaison"
          placeholder="Ask what is happening, why a workstream is blocked, or route a request…"
          value={text}
          onChange={(event) => setText(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              send();
            }
          }}
        />
        <div className="liaison-composer-foot">
          <span className="muted">Questions are private; confirmed drafts enter the organization as this seat.</span>
          <button className="primary" disabled={!text.trim() || busy} onClick={send}>
            {busy ? "Interpreting…" : "Ask liaison"}
          </button>
        </div>
      </div>
    </div>
  );
}
