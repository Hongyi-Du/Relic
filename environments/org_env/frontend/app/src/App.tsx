import { useMemo, useState } from "react";
import { useSession } from "./session";
import { Ctx, type InspectorCtx } from "./ctx";
import { SUBS_EXTERNAL, SUBS_INTERNAL, OBJECT_PREFIX, PERSONA_PREFIX } from "./theme";
import { num } from "./ui";
import { Drawer, Reader } from "./drawer";
import { SourcePicker } from "./SourcePicker";
import { InternalPanel, ExternalPanel } from "./panels";
import { ProductTerminal } from "./panels/product";
import type { DrawerDesc, GNode, Graph, ReaderDoc, Tab, ViewMode } from "./types";

export default function App() {
  const s = useSession();
  const [tab, setTab] = useState<Tab>("internal");
  const [sub, setSub] = useState<string>("dashboard");
  const [agent, setAgent] = useState<string | null>(null);
  const [view, setView] = useState<ViewMode>("omniscient");
  const [drawer, setDrawer] = useState<DrawerDesc>(null);
  const [reader, setReader] = useState<ReaderDoc>(null);
  const [seed, setSeed] = useState(42);

  const f = s.frame;

  // default agent on first frame (mirrors vanilla)
  const firstAgent = f ? Object.keys(f.agents || {})[0] || null : null;
  const activeAgent = agent ?? firstAgent;

  const goSub = (x: string) => { setSub(x); };
  const selectAgent = (a: string) => {
    setTab("internal"); setSub("agents"); setAgent(a || null);
  };

  const ctx: InspectorCtx = useMemo(() => {
    const openNode = (node: GNode, persona?: Graph) => {
      const t = node.type || "";
      if (t === "agent" || t === "internal_agent") {
        if (t === "agent" && persona) { setDrawer({ kind: "personaNode", node, persona }); return; }
        selectAgent(node.id); return;
      }
      if (persona && PERSONA_PREFIX.test(node.id)) { setDrawer({ kind: "personaNode", node, persona }); return; }
      if (OBJECT_PREFIX.test(node.id)) { setDrawer({ kind: "object", id: node.id }); return; }
      setDrawer({ kind: "kv", title: "Node " + node.id, obj: { type: node.type, ...(node.attrs || {}) } });
    };
    return {
      frame: f || {},
      view,
      agent: activeAgent,
      setSub: goSub,
      selectAgent,
      openObj: (id) => setDrawer({ kind: "object", id }),
      openFile: (id) => setDrawer({ kind: "file", id }),
      openArtifact: (id) => setDrawer({ kind: "artifact", id }),
      openEpisode: (id) => setDrawer({ kind: "episode", id }),
      openReflection: (id) => setDrawer({ kind: "reflection", id }),
      openWish: (id) => setDrawer({ kind: "wish", id }),
      openProposal: (id) => setDrawer({ kind: "proposal", id }),
      openNode,
      openKV: (title, obj) => setDrawer({ kind: "kv", title, obj }),
      openReader: (doc) => setReader(doc),
    };
  }, [f, view, activeAgent]);

  const c = (f && f.company) || {};
  const aliveAgents = f ? Object.keys(f.agents || {}).length : 0;
  const subs = tab === "internal" ? SUBS_INTERNAL : SUBS_EXTERNAL;

  return (
    <Ctx.Provider value={ctx}>
      <div className="app">
        {/* ---------- top bar ---------- */}
        <div className="topbar">
          <div className="brand"><span className="mark">🔬</span><span className="name">OrgEnv Inspector</span></div>
          <span className="clock">
            {f ? `tick ${f.tick} · day ${f.day} · ${String(f.hour).padStart(2, "0")}:00 · ${f.phase}${f.day_of_week ? " · " + f.day_of_week : ""}` : "tick — · day —"}
          </span>
          <span className="runstate"><span className={"dot" + (s.running ? " run" : "")} />{s.running ? "running" : s.mode === "replay" ? "replay" : "idle"}</span>

          <span className="row">
            <SourcePicker value={s.source} sources={s.sources} running={s.running} onSelect={s.selectSource} />
            <button onClick={() => s.simStep(1)}>Step</button>
            <button onClick={() => s.simRun(10)}>Run 10</button>
            <button onClick={() => s.simRun(24)}>Run 24</button>
            <button onClick={() => s.simRun(72)}>Run 3d</button>
            <button onClick={() => s.simPause()}>Pause</button>
            <button onClick={() => s.simReset(seed)}>Reset</button>
            seed <input type="number" value={seed} onChange={(e) => setSeed(+e.target.value || 42)} />
            <button onClick={async () => { const d = await s.exportReplay(); alert("Saved replay: " + (d.name || d.error)); }}>Export</button>
          </span>

          <span className="scrubrow">
            <button onClick={s.togglePlay}>{s.playing ? "⏸" : "▶"}</button>
            <input type="range" min={0} max={Math.max(0, s.frames.length - 1)} value={s.fi} onChange={(e) => s.setFrame(+e.target.value)} />
            <span className="clock">{s.frames.length ? s.fi + 1 : 0}/{s.frames.length}</span>
          </span>

          <span className="row">
            view
            <select value={view} onChange={(e) => setView(e.target.value as ViewMode)}>
              <option value="omniscient">Omniscient (debug)</option>
              <option value="agent_visible">Agent-visible</option>
            </select>
          </span>

          <span className="statstrip">
            <Stat label="tasks" value={`${c.tasks_done ?? 0}/${c.task_count ?? 0}`} />
            <Stat label="cash" value={num(c.cash_balance)} />
            <Stat label="runway" value={num(c.runway_days)} />
            <Stat label="open PRs" value={c.open_prs ?? 0} />
            <Stat label="protocols" value={c.protocol_count ?? 0} />
            <Stat label="agents" value={aliveAgents} />
          </span>
        </div>

        {/* ---------- tabs ---------- */}
        <div className="maintabs">
          <div className={"maintab" + (tab === "internal" ? " active" : "")} onClick={() => { setTab("internal"); setSub("dashboard"); }}>Company Internal</div>
          <div className={"maintab" + (tab === "external" ? " active" : "")} onClick={() => { setTab("external"); setSub("dashboard"); }}>External Network</div>
          <div className={"maintab" + (tab === "product" ? " active" : "")} onClick={() => setTab("product")}>🧪 Product Terminal</div>
        </div>

        {tab !== "product" && (
          <div className="subnav">
            {subs.map((x) => <span key={x} className={"chip" + (sub === x ? " active" : "")} onClick={() => goSub(x)}>{x}</span>)}
          </div>
        )}

        {/* ---------- content ---------- */}
        <div className="scroll-area">
          <div className="wrap">
            {!f ? (
              <div className="fallback"><div className="spinner" /><p>waiting for a frame… click Step / Run.</p></div>
            ) : tab === "product" ? (
              <ProductTerminal f={f} />
            ) : tab === "internal" ? (
              <InternalPanel sub={sub} f={f} />
            ) : (
              <ExternalPanel sub={sub} f={f} />
            )}
          </div>
        </div>

        <Drawer desc={drawer} onClose={() => setDrawer(null)} />
        <Reader doc={reader} onClose={() => setReader(null)} />
      </div>
    </Ctx.Provider>
  );
}

function Stat({ label, value }: { label: string; value: React.ReactNode }) {
  return <div className="s"><span className="v">{value}</span><span className="l">{label}</span></div>;
}
