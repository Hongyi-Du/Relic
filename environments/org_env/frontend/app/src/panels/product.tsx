import { useState } from "react";
import { api } from "../api";
import { Section, Tag, cls } from "../ui";
import type { PanelProps } from "./common";

export function ProductTerminal({ f }: PanelProps) {
  const [query, setQuery] = useState("");
  const [out, setOut] = useState<any | null>(null);
  const [busy, setBusy] = useState(false);
  const [rating, setRating] = useState(0);
  const [comment, setComment] = useState("");
  const [fbMsg, setFbMsg] = useState("");

  const run = async () => {
    setBusy(true);
    setOut({ report: "… running the product …", profile: {}, caveats: [], confidence: null });
    const d = await api.productTry(query);
    setOut({ report: d.report || d.error || "(no output)", profile: d.profile || {}, caveats: d.caveats || [], confidence: d.confidence, llm: d.llm });
    setBusy(false);
  };
  const submitFeedback = async () => {
    if (!rating && !comment) { setFbMsg("pick a rating or add a comment first"); return; }
    const d = await api.productFeedback(rating || 0, comment, query);
    setFbMsg(`✓ recorded ${d.ticket_id} (rating ${d.rating}) at t${d.tick} · human tickets: ${d.total_human_tickets}`);
  };

  const prof = out?.profile || {};
  return (
    <div className="pt-shell">
      <Section title="🧪 Product Terminal — try & evaluate the company's product"
        sub={`Runs the LIVE product at tick ${f.tick} with EXACTLY its current capabilities + limitations. Your rating is recorded as a customer signal. Use Step / Run on the top bar to evolve the product, then re-run.`}>
        <div className="pt-input">
          <input type="text" value={query} placeholder="Ask the product to do something — e.g. 'Research the best open-source vector DB for RAG and cite sources'"
            onChange={(e) => setQuery(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter" && !busy) run(); }} />
          <button className="btn-accent" disabled={busy} onClick={run}>▶ Run product</button>
        </div>
      </Section>

      {out && (
        <>
          <Section title="Report" right={<span className="muted">{out.llm === false ? <Tag tone="warn">no-LLM stub</Tag> : null} · confidence {out.confidence ?? "—"}</span>}>
            <pre className="pt-out" style={{ background: "var(--panel2)", padding: 10, borderRadius: 8, maxHeight: 420, overflow: "auto" }}>{out.report || ""}</pre>
            {(out.caveats || []).length ? <div className="kv"><b>caveats</b>{(out.caveats || []).map((c: string, i: number) => <div className="muted" key={i} style={{ fontSize: 11 }}>• {c}</div>)}</div> : null}
          </Section>
          <Section title={`Product profile @ t${prof.tick ?? "—"}`} sub={`${prof.name || ""} · ${prof.stage || ""}`}>
            <div className="kv"><b>capabilities</b> <div className="pillrow">{(prof.capabilities || []).length ? (prof.capabilities || []).map((c: string) => <Tag key={c} tone="good">{c}</Tag>) : <span className="muted">minimal</span>}</div></div>
            <div className="kv"><b>known limitations</b>{(prof.known_limitations || []).length ? (prof.known_limitations || []).map((c: string, i: number) => <div className="muted" key={i} style={{ fontSize: 11 }}>• {c}</div>) : <span className="muted">none recorded</span>}</div>
          </Section>
          <Section title="Evaluate this product">
            <div className="row stars">
              <span className="muted">rating:</span>
              {[1, 2, 3, 4, 5].map((n) => <button key={n} className={cls(rating >= n && "on")} onClick={() => setRating(n)}>★</button>)}
            </div>
            <input type="text" style={{ width: "100%", marginTop: 8 }} value={comment} placeholder="optional feedback — what worked, what didn't, would you pay?" onChange={(e) => setComment(e.target.value)} />
            <button className="btn-accent" style={{ marginTop: 8 }} onClick={submitFeedback}>Submit feedback → customer signal</button>
            {fbMsg ? <div style={{ marginTop: 6 }}><Tag tone="good">{fbMsg}</Tag></div> : null}
          </Section>
        </>
      )}
    </div>
  );
}
