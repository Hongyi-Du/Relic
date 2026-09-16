// The backend frame is a deep, evolving dict (built by runtime_adapter/snapshot.py).
// We keep it intentionally loose — panels read documented paths defensively.

export interface GNode {
  id: string;
  label?: string;
  type?: string;
  attrs?: Record<string, any>;
  cluster?: string;
  value?: any;
  salience?: any;
  [k: string]: any;
}

export interface GEdge {
  src: string;
  dst: string;
  type?: string;
  weight?: number;
  strength?: number;
  reason?: string;
  explanation?: string;
  [k: string]: any;
}

export interface Graph {
  nodes: GNode[];
  edges: GEdge[];
}

export type Frame = Record<string, any>;

export type Tab = "internal" | "external" | "product";
export type ViewMode = "omniscient" | "agent_visible";

export type ReplaySource = { name: string; ticks: number };

// drawer descriptor — what the right-hand inspector panel is currently showing
export type DrawerDesc =
  | { kind: "object"; id: string }
  | { kind: "file"; id: string }
  | { kind: "artifact"; id: string }
  | { kind: "episode"; id: string }
  | { kind: "reflection"; id: string }
  | { kind: "wish"; id: string }
  | { kind: "proposal"; id: string }
  | { kind: "personaNode"; node: GNode; persona: Graph }
  | { kind: "kv"; title: string; obj: Record<string, any> }
  | null;

export type ReaderDoc = { title: string; meta: string; text: string; lang?: string } | null;
