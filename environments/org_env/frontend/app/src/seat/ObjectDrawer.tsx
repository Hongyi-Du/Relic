// Clicking any organizational object opens it here, with the actions this seat
// may take on it. Actions the member's role forbids are shown greyed with the
// reason rather than hidden, so the interface teaches the rules.

import { useEffect, useState } from "react";
import { ActionForm } from "./ActionForm";
import { Offer, OfferMenu, ago, api } from "./api";

type Props = {
  token: string;
  objectId: string;
  object?: any;
  onClose: () => void;
  onAct: (actionType: string, params: Record<string, any>) => Promise<any>;
  view?: any;  // SeatView for context
};

const HIDE_FIELDS = new Set(["id", "kind", "age_hours", "at"]);

export function ObjectDrawer({ token, objectId, object, onClose, onAct, view }: Props) {
  const [menu, setMenu] = useState<OfferMenu | null>(null);
  const [error, setError] = useState("");
  const [open, setOpen] = useState<Offer | null>(null);

  useEffect(() => {
    let live = true;
    setMenu(null);
    setOpen(null);
    setError("");
    api.offers(token, objectId).then((d: any) => {
      if (!live) return;
      if (d.error) setError(d.error);
      else setMenu(d);
    });
    return () => { live = false; };
  }, [token, objectId]);

  // Build context for smart form fields
  const context = view ? {
    tasks: view.objects?.tasks || [],
    issues: view.objects?.issues || [],
    members: view.member?.members || [],
    channels: view.feed?.channels || [],
  } : undefined;

  return (
    <aside className="drawer">
      <header>
        <div>
          <div className="drawer-kind">{menu?.kind || object?.kind || "object"}</div>
          <div className="drawer-title">{object?.title || objectId}</div>
        </div>
        <button onClick={onClose}>✕</button>
      </header>

      {error && <div className="error">{error}</div>}

      {object && (
        <section className="drawer-fields">
          {Object.entries(object)
            .filter(([k, v]) => !HIDE_FIELDS.has(k) && v !== null && v !== "")
            .map(([k, v]) => (
              <div className="kv" key={k}>
                <span className="k">{k.replace(/_/g, " ")}</span>
                <span className="v">
                  {typeof v === "object" ? JSON.stringify(v) : String(v)}
                </span>
              </div>
            ))}
          {object.age_hours != null && (
            <div className="kv">
              <span className="k">last activity</span>
              <span className="v">{object.age_hours}h of org time ago</span>
            </div>
          )}
        </section>
      )}

      <section className="drawer-actions">
        <h4>Actions</h4>
        {!menu && !error && <div className="muted">…</div>}
        {menu?.actions.filter((offer) => offer.allowed).map((offer) =>
          open?.action_type === offer.action_type ? (
            <ActionForm
              key={offer.action_type}
              offer={offer}
              onSubmit={onAct}
              onDone={() => setOpen(null)}
              context={context}
            />
          ) : (
            <button
              key={offer.action_type}
              className="offer"
              onClick={() => setOpen(offer)}
            >
              {offer.label}
            </button>
          ),
        )}
        {menu && menu.actions.filter((o) => o.allowed).length === 0 && (
          <div className="muted">No actions available for your role.</div>
        )}
      </section>
    </aside>
  );
}

export { ago };
