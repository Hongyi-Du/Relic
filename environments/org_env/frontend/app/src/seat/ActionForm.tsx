// One action, its parameters, and a confirm step for the ones that change what
// the organization sees or ships (HCI V0 §7).

import { useMemo, useState } from "react";
import { Offer } from "./api";
import { SmartFormField } from "./SmartFormField";

type Props = {
  offer: Offer;
  seed?: Record<string, any>;
  onSubmit: (actionType: string, params: Record<string, any>) => Promise<any>;
  onDone?: () => void;
  context?: {
    tasks?: any[];
    issues?: any[];
    members?: any[];
    channels?: any[];
  };
};

const LONG_FIELDS = new Set([
  "text", "comment", "notes", "body", "reason", "rationale", "summary",
  "edit_goal", "problem_evidence", "description", "resolution",
]);

export function ActionForm({ offer, seed, onSubmit, onDone, context }: Props) {
  const fields = useMemo(
    () => [...offer.required, ...offer.optional].filter(
      (f) => !(offer.target && f === offer.target_param),
    ),
    [offer],
  );
  const [values, setValues] = useState<Record<string, string>>(() => {
    const init: Record<string, string> = {};
    for (const f of fields) init[f] = String(seed?.[f] ?? "");
    return init;
  });
  const [confirming, setConfirming] = useState(false);
  const [busy, setBusy] = useState(false);

  const missing = offer.required.filter(
    (f) => !(offer.target && f === offer.target_param) && !values[f]?.trim(),
  );

  const run = async () => {
    setBusy(true);
    try {
      const params: Record<string, any> = {};
      for (const [k, v] of Object.entries(values)) if (v.trim()) params[k] = v.trim();
      if (offer.target && offer.target_param) params[offer.target_param] = offer.target;
      await onSubmit(offer.action_type, params);
      setConfirming(false);
      onDone?.();
    } finally {
      setBusy(false);
    }
  };

  if (!offer.allowed) {
    return (
      <div className="act-form denied">
        <div className="act-title">{offer.label}</div>
        <div className="denied-why">Not available to your role — {offer.denied_because}</div>
      </div>
    );
  }

  return (
    <div className="act-form">
      <div className="act-title">{offer.label}</div>
      {fields.map((f) => (
        <label key={f} className="act-field">
          <span>{f.replace(/_/g, " ")}{offer.required.includes(f) ? " *" : ""}</span>
          {context ? (
            <SmartFormField
              name={f}
              value={values[f] || ""}
              onChange={(v) => setValues({ ...values, [f]: v })}
              required={offer.required.includes(f)}
              context={context}
            />
          ) : LONG_FIELDS.has(f) ? (
            <textarea
              rows={3}
              value={values[f] || ""}
              onChange={(e) => setValues({ ...values, [f]: e.target.value })}
            />
          ) : (
            <input
              value={values[f] || ""}
              onChange={(e) => setValues({ ...values, [f]: e.target.value })}
            />
          )}
        </label>
      ))}

      {confirming ? (
        <div className="act-confirm">
          <span>This is visible to the organization. Send it?</span>
          <button className="primary" disabled={busy} onClick={run}>
            {busy ? "…" : "Confirm"}
          </button>
          <button onClick={() => setConfirming(false)}>Cancel</button>
        </div>
      ) : (
        <button
          className="primary"
          disabled={busy || missing.length > 0}
          title={missing.length ? `needs ${missing.join(", ")}` : ""}
          onClick={() => (offer.confirm ? setConfirming(true) : run())}
        >
          {busy ? "…" : offer.label}
        </button>
      )}
    </div>
  );
}
