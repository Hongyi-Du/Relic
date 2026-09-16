// The member's private assistant. Nothing here is visible to the organization
// until the human confirms a prepared action.

import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "./api";
import { parseCommand, getCommandSuggestions, COMMANDS } from "./commands";

type Msg = { role: string; text: string; at: number; tool: string };
type Draft = {
  draft_id: string; action_type: string; label: string;
  params: Record<string, any>; rationale: string;
};

const POLL_MS = 900;

export function WorkingAgent({ token, onActed }: { token: string; onActed: () => void }) {
  const [messages, setMessages] = useState<Msg[]>([]);
  const [drafts, setDrafts] = useState<Draft[]>([]);
  const [busy, setBusy] = useState(false);
  const [text, setText] = useState("");
  const [openTool, setOpenTool] = useState<number | null>(null);
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [suggestionFocus, setSuggestionFocus] = useState(0);
  const [showGuide, setShowGuide] = useState(true);  // Can be toggled
  const bottom = useRef<HTMLDivElement | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  const poll = useCallback(async () => {
    const d = await api.agentState(token, 0);
    if (d.error) return;
    setMessages(d.messages || []);
    setDrafts(d.drafts || []);
    setBusy(!!d.busy);
  }, [token]);

  useEffect(() => {
    poll();
    const t = window.setInterval(poll, POLL_MS);
    return () => window.clearInterval(t);
  }, [poll]);

  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages.length, drafts.length]);

  // Update command suggestions when text changes
  useEffect(() => {
    if (text.startsWith("/")) {
      setSuggestions(getCommandSuggestions(text));
      setSuggestionFocus(0);
    } else {
      setSuggestions([]);
    }
  }, [text]);

  const send = async () => {
    if (!text.trim() || busy) return;
    let body = text.trim();

    // Parse command if it starts with /
    const cmd = parseCommand(body);
    if (cmd) {
      if (cmd.type === "help") {
        // Show help inline
        setMessages([...messages, { role: "system", text: cmd.naturalLanguage, at: Date.now(), tool: "" }]);
        setText("");
        return;
      }
      // Convert command to natural language
      body = cmd.naturalLanguage;
    }

    setText("");
    setBusy(true);
    await api.agentSend(token, body);
    poll();
  };

  const confirm = async (id: string) => {
    await api.agentConfirm(token, id);
    await poll();
    onActed();
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (suggestions.length > 0) {
      if (e.key === "ArrowDown") {
        e.preventDefault();
        setSuggestionFocus((f) => Math.min(f + 1, suggestions.length - 1));
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        setSuggestionFocus((f) => Math.max(f - 1, 0));
      } else if (e.key === "Tab" || e.key === "Enter") {
        e.preventDefault();
        setText(suggestions[suggestionFocus] + " ");
        setSuggestions([]);
      }
    } else if (e.key === "Enter" && !e.shiftKey) {
      // Enter = send, Shift+Enter = newline
      e.preventDefault();
      send();
    }
  };

  return (
    <div className="pane agent-pane">
      <div className="pane-head">
        Working agent
        <span className="agent-private">private to you</span>
        {messages.length > 0 && !showGuide && (
          <button className="link guide-toggle" onClick={() => setShowGuide(true)}>
            📖 Show Guide
          </button>
        )}
      </div>

      <div className="agent-stream">
        {showGuide && (
          <div className="onboarding-guide">
            <div className="guide-header">
              <div className="guide-title">👋 Welcome! I'm your private assistant</div>
              {messages.length > 0 && (
                <button className="guide-close" onClick={() => setShowGuide(false)}>✕</button>
              )}
            </div>
            <div className="guide-section">
              <div className="guide-label">Try asking me:</div>
              <div className="guide-examples">
                <button className="guide-example" onClick={() => setText("现在是什么情况")}>
                  💼 "现在是什么情况" - Your tasks and mentions
                </button>
                <button className="guide-example" onClick={() => setText("帮我搜索最近的消息")}>
                  🔍 "帮我搜索最近的消息" - Recent discussions
                </button>
                <button className="guide-example" onClick={() => setText("产品代码有哪些文件")}>
                  📁 "产品代码有哪些文件" - List product files
                </button>
              </div>
            </div>
            <div className="guide-section">
              <div className="guide-label">Quick commands:</div>
              <div className="guide-tips">
                Type <code>/</code> to see all commands (help, status, search, read, tests)
              </div>
            </div>
            <div className="guide-footer">
              💡 <strong>Tip:</strong> Press <kbd>Enter</kbd> to send, <kbd>Shift+Enter</kbd> for new line
            </div>
          </div>
        )}
        {messages.map((m, i) =>
          m.role === "tool" ? (
            <div className="tool-call" key={i}>
              <button onClick={() => setOpenTool(openTool === i ? null : i)}>
                {openTool === i ? "▾" : "▸"} {m.tool}
              </button>
              {openTool === i && <pre>{m.text}</pre>}
            </div>
          ) : (
            <div className={`agent-msg ${m.role}`} key={i}>
              {m.text}
            </div>
          ),
        )}
        {busy && <div className="agent-msg agent thinking">working…</div>}
        <div ref={bottom} />
      </div>

      {drafts.length > 0 && (
        <div className="drafts">
          {drafts.map((d) => (
            <div className="draft" key={d.draft_id}>
              <div className="draft-head">Ready to send · {d.label}</div>
              {d.rationale && <div className="draft-why">{d.rationale}</div>}
              <div className="draft-params">
                {Object.entries(d.params).map(([k, v]) => (
                  <div className="kv" key={k}>
                    <span className="k">{k.replace(/_/g, " ")}</span>
                    <span className="v">{String(v)}</span>
                  </div>
                ))}
              </div>
              <div className="draft-actions">
                <button className="primary" onClick={() => confirm(d.draft_id)}>
                  Send as me
                </button>
                <button onClick={() => api.agentDiscard(token, d.draft_id).then(poll)}>
                  Discard
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      <div className="agent-composer">
        <div className="composer-wrapper">
          <textarea
            ref={textareaRef}
            rows={2}
            placeholder="Ask your working agent… (Enter to send, Shift+Enter for new line)"
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={handleKeyDown}
          />
          {suggestions.length > 0 && (
            <div className="command-suggestions">
              {suggestions.map((sug, i) => (
                <button
                  key={sug}
                  className={`suggestion ${i === suggestionFocus ? "focused" : ""}`}
                  onClick={() => {
                    setText(sug + " ");
                    setSuggestions([]);
                    textareaRef.current?.focus();
                  }}
                  onMouseEnter={() => setSuggestionFocus(i)}
                >
                  <span className="sug-cmd">{sug}</span>
                  <span className="sug-desc">{COMMANDS[sug.slice(1) as keyof typeof COMMANDS]?.description}</span>
                </button>
              ))}
            </div>
          )}
        </div>
        <button className="primary" disabled={busy || !text.trim()} onClick={send}>
          {busy ? "…" : "Ask"}
        </button>
      </div>
    </div>
  );
}
