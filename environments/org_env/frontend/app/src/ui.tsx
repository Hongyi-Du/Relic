import type { ReactNode } from "react";
import { statusTone } from "./theme";

export const cls = (...xs: (string | false | null | undefined)[]) => xs.filter(Boolean).join(" ");

export const num = (v: any, d = 0): string => {
  const n = Number(v);
  return Number.isFinite(n) ? n.toFixed(d) : String(v ?? "—");
};
export const pct = (v: any): string => (Number.isFinite(Number(v)) ? `${Math.round(Number(v) * 100)}%` : "—");

export function Cards({ children }: { children: ReactNode }) {
  return <div className="cards">{children}</div>;
}

export function StatCard({ label, value, onClick }: { label: string; value: ReactNode; onClick?: () => void }) {
  return (
    <div className={cls("card", onClick && "clk")} onClick={onClick}>
      <div className="v">{value}</div>
      <div className="l">{label}</div>
    </div>
  );
}

export function Section({ title, sub, children, right }: { title: string; sub?: string; children: ReactNode; right?: ReactNode }) {
  return (
    <div className="section">
      <h3>
        {title}
        {right && <span style={{ marginLeft: "auto", textTransform: "none", letterSpacing: 0 }}>{right}</span>}
      </h3>
      {sub && <div className="sub">{sub}</div>}
      {children}
    </div>
  );
}

export function Tag({ children, tone, color }: { children: ReactNode; tone?: "good" | "warn" | "bad" | "viz" | ""; color?: string }) {
  return (
    <span className={cls("tag", tone)} style={color ? { color, borderColor: color + "55", background: color + "1f" } : undefined}>
      {children}
    </span>
  );
}

/** auto-toned status tag */
export function StatusTag({ value }: { value: any }) {
  const v = value == null || value === "" ? "—" : String(value);
  return <Tag tone={statusTone(v)}>{v}</Tag>;
}

export function Bar({ value, max = 1, color = "var(--accent)" }: { value: number; max?: number; color?: string }) {
  const w = Math.max(0, Math.min(100, (Number(value) / (max || 1)) * 100));
  return (
    <div className="bar">
      <i style={{ width: `${w}%`, background: color }} />
    </div>
  );
}

export function Pills({ items, color }: { items: any[]; color?: string }) {
  if (!items || !items.length) return <span className="faint">—</span>;
  return (
    <div className="pillrow">
      {items.map((x, i) => (
        <Tag key={i} color={color}>{String(x)}</Tag>
      ))}
    </div>
  );
}

export interface Column<T> {
  header: string;
  cell: (row: T) => ReactNode;
  width?: number | string;
}

export function DataTable<T>({
  columns, rows, onRowClick, empty = "no rows in this frame",
}: {
  columns: Column<T>[];
  rows: T[];
  onRowClick?: (row: T) => void;
  empty?: string;
}) {
  if (!rows || !rows.length) return <div className="empty">{empty}</div>;
  return (
    <div className="tablewrap">
      <table>
        <thead>
          <tr>{columns.map((c, i) => <th key={i} style={c.width ? { width: c.width } : undefined}>{c.header}</th>)}</tr>
        </thead>
        <tbody>
          {rows.map((r, ri) => (
            <tr key={ri} className={onRowClick ? "clk" : undefined} onClick={onRowClick ? () => onRowClick(r) : undefined}>
              {columns.map((c, ci) => <td key={ci}>{c.cell(r)}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** kv block — object entries to labelled rows (kvBlock equivalent) */
export function KV({ obj, cols = 1 }: { obj: Record<string, any>; cols?: number }) {
  const entries = Object.entries(obj || {});
  if (!entries.length) return <span className="faint">—</span>;
  return (
    <div className={cols > 1 ? "kvgrid" : undefined}>
      {entries.map(([k, v]) => (
        <div className="kv" key={k}>
          <b>{k}</b>: {v == null || v === "" ? "—" : typeof v === "object" ? JSON.stringify(v) : String(v)}
        </div>
      ))}
    </div>
  );
}
