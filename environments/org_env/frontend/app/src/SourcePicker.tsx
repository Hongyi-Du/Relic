import { useEffect, useRef, useState } from "react";
import { LIVE } from "./session";
import type { ReplaySource } from "./types";

/** Custom dropdown for the data source. A native <select> closes whenever the
 * app re-renders (live polling), so we use a React-controlled menu instead. */
export function SourcePicker({
  value, sources, running, onSelect,
}: {
  value: string;
  sources: ReplaySource[];
  running: boolean;
  onSelect: (v: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const h = (e: MouseEvent) => { if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false); };
    document.addEventListener("mousedown", h);
    return () => document.removeEventListener("mousedown", h);
  }, [open]);

  const label = value === LIVE
    ? "LIVE (running sim)"
    : (() => { const s = sources.find((x) => x.name === value); return s ? `${s.name} (${s.ticks}t)` : value; })();

  const pick = (v: string) => { setOpen(false); onSelect(v); };

  return (
    <div className="dd" ref={ref}>
      <button className="dd-btn" onClick={() => setOpen((o) => !o)} title="data source">
        {value === LIVE ? <span className={"dot" + (running ? " run" : "")} /> : null}
        <span className="dd-label">{label}</span>
        <span className="dd-caret">▾</span>
      </button>
      {open && (
        <div className="dd-menu">
          <div className={"dd-item" + (value === LIVE ? " active" : "")} onClick={() => pick(LIVE)}>
            <span className={"dot" + (running ? " run" : "")} /> LIVE (running sim)
          </div>
          {sources.length === 0 ? (
            <div className="dd-empty">no saved replays</div>
          ) : (
            sources.map((s) => (
              <div key={s.name} className={"dd-item" + (value === s.name ? " active" : "")} onClick={() => pick(s.name)}>
                <span className="dd-name">{s.name}</span>
                <span className="dd-sub">{s.ticks}t</span>
              </div>
            ))
          )}
        </div>
      )}
    </div>
  );
}
