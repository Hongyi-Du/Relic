// Collapsible group component for organizing information by priority
import { useState, ReactNode } from "react";

type Props = {
  title: string;
  count?: number;
  defaultOpen?: boolean;
  urgency?: "normal" | "warn" | "critical";
  children: ReactNode;
};

export function CollapsibleGroup({ title, count, defaultOpen = true, urgency = "normal", children }: Props) {
  const [open, setOpen] = useState(defaultOpen);

  const displayTitle = count !== undefined ? `${title} (${count})` : title;

  return (
    <section className={`group collapsible ${urgency}`}>
      <button className="group-header" onClick={() => setOpen(!open)}>
        <span className="expand-icon">{open ? "▾" : "▸"}</span>
        <h4>{displayTitle}</h4>
      </button>
      {open && <div className="group-content">{children}</div>}
    </section>
  );
}
