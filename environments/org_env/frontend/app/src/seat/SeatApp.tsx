// A human holding one seat in the organization.
//
// Deliberately not the inspector: no ticks, no other members' interiors, no
// omniscient view. What is on screen is what this member is allowed to know.

import { useCallback, useMemo, useState } from "react";
import { ObjectDrawer } from "./ObjectDrawer";
import { MemberPanel } from "./panels";
import { LiaisonPanel } from "./LiaisonPanel";
import { clockTime } from "./api";
import { useSeat } from "./useSeat";

export function SeatApp() {
  const seat = useSeat();
  const [drawer, setDrawer] = useState<{ id: string; obj?: any } | null>(null);

  // Objects arrive in the view by kind; the drawer takes an id from anywhere
  // (a message attachment, a linked task) so index them all once.
  const byId = useMemo(() => {
    const index: Record<string, any> = {};
    for (const rows of Object.values(seat.view?.objects || {}))
      for (const o of rows as any[]) index[o.id] = o;
    for (const t of seat.view?.feed.threads || [])
      for (const m of t.messages) index[m.id] = m;
    return index;
  }, [seat.view]);

  const open = useCallback((id: string, obj?: any) => setDrawer({ id, obj }), []);

  if (!seat.token || !seat.view) {
    return <SeatPicker seat={seat} />;
  }

  return (
    <div className="seat-app">
      <TopBar seat={seat} />
      <div className="columns p2-columns">
        <MemberPanel view={seat.view} open={open} />
        <LiaisonPanel token={seat.token} view={seat.view}
                      onActed={seat.refresh} open={open} />
      </div>
      {drawer && (
        <ObjectDrawer
          token={seat.token}
          objectId={drawer.id}
          object={drawer.obj || byId[drawer.id]}
          onClose={() => setDrawer(null)}
          onAct={seat.act}
          view={seat.view}
        />
      )}
      {seat.error && (
        <div className="toast" onClick={() => window.location.reload()}>
          {seat.error}
        </div>
      )}
    </div>
  );
}

function TopBar({ seat }: { seat: ReturnType<typeof useSeat> }) {
  const rt = seat.runtime;
  const now = seat.view?.clock.now;
  const openP3 = async () => {
    if (seat.view?.seat.agent_id !== "victor") await seat.leave();
    window.location.assign("/org/liaison");
  };
  return (
    <header className="topbar">
      <div className="brand">{rt?.pack?.product_name || "Relic"}</div>
      <div className="clock">
        <span className="time">{clockTime(now)}</span>
        <span className={`status ${rt?.running ? "live" : "paused"}`}>
          {rt?.running ? "team working" : "team paused"}
        </span>
      </div>
      <ModeSwitch active="p2" onOpenP3={openP3} />
      <div className="spacer" />
      <button onClick={rt?.running ? seat.pauseClock : seat.startClock}>
        {rt?.running ? "Pause the team" : "Resume the team"}
      </button>
      <div className="me">
        {seat.view?.seat.name}
        <button className="link" onClick={seat.leave}>leave seat</button>
      </div>
    </header>
  );
}

function SeatPicker({ seat }: { seat: ReturnType<typeof useSeat> }) {
  return (
    <div className="picker-shell">
      <header className="picker-modebar">
        <strong>Relic</strong>
        <ModeSwitch active="p2" />
      </header>
      <div className="picker">
        <h1>Take a seat</h1>
        <p className="muted">
          You will hold this member's place in the organization: their role, their
          permissions, and only what they are allowed to see.
        </p>
        {seat.error && <div className="error">{seat.error}</div>}
        <div className="member-grid">
          {seat.members.map((m: any) => (
            <button
              key={m.agent_id}
              className={`member-card ${m.claimed ? "taken" : ""}`}
              disabled={m.claimed || seat.busy}
              onClick={() => seat.claim(m.agent_id)}
            >
              <div className="mc-name">{m.name}</div>
              <div className="mc-role">{m.role.replace(/_/g, " ")}</div>
              <div className="mc-identity">{m.identity}</div>
              {m.claimed && <div className="mc-taken">already taken</div>}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

function ModeSwitch({ active, onOpenP3 }: { active: "p2" | "p3"; onOpenP3?: () => void }) {
  return (
    <nav className="experience-switch" aria-label="Experience version">
      <span className="experience-switch-label">Experience</span>
      <a className={active === "p2" ? "active" : ""} aria-current={active === "p2" ? "page" : undefined} href="/org/seat">P2 Transparent</a>
      <a className={active === "p3" ? "active" : ""} aria-current={active === "p3" ? "page" : undefined} href="/org/liaison" onClick={onOpenP3 ? (event) => { event.preventDefault(); void onOpenP3(); } : undefined}>P3 Secretary</a>
    </nav>
  );
}
