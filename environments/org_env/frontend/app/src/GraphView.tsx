import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Background, BackgroundVariant, Controls, Handle, MarkerType, Position, ReactFlow,
  useEdgesState, useNodesState, type Edge, type Node, type NodeProps,
} from "@xyflow/react";
import dagre from "@dagrejs/dagre";
import { NODE_COLORS, OBJECT_PREFIX, PERSONA_PREFIX } from "./theme";
import { cls } from "./ui";
import { useInspector } from "./ctx";
import type { GNode, Graph } from "./types";

const NODE_W = 172;
const NODE_H = 46;
const MAX_NODES = 300;

function CardNode({ data }: NodeProps) {
  const d = data as any;
  const color = NODE_COLORS[d.ntype] || "#8b949e";
  return (
    <div className={cls("rf-node", d.dim && "dim", d.hl && "hl", d.sel && "sel")} style={{ borderColor: color }}>
      <Handle type="target" position={Position.Left} />
      {d.ntype ? <div className="nt" style={{ color }}>{d.ntype}</div> : null}
      <div className="lbl">{d.label}</div>
      <Handle type="source" position={Position.Right} />
    </div>
  );
}
const nodeTypes = { card: CardNode };

function dagreLayout(ids: string[], links: [string, string][], dir: "LR" | "TB"): Record<string, { x: number; y: number }> {
  const g = new dagre.graphlib.Graph();
  g.setDefaultEdgeLabel(() => ({}));
  g.setGraph({ rankdir: dir, nodesep: 34, ranksep: 130, edgesep: 16, marginx: 26, marginy: 26 });
  ids.forEach((id) => g.setNode(id, { width: NODE_W, height: NODE_H }));
  links.forEach(([s, t]) => { if (s !== t) g.setEdge(s, t); });
  dagre.layout(g);
  const out: Record<string, { x: number; y: number }> = {};
  ids.forEach((id) => {
    const p = g.node(id) as any;
    out[id] = { x: (p?.x ?? 0) - NODE_W / 2, y: (p?.y ?? 0) - NODE_H / 2 };
  });
  return out;
}

export default function GraphView({
  graph, height = 460, focus = null, direction = "LR", persona,
}: {
  graph: Graph | undefined;
  height?: number;
  focus?: string | null;
  direction?: "LR" | "TB";
  persona?: Graph;
}) {
  const ctx = useInspector();
  const g: Graph = graph || { nodes: [], edges: [] };
  const baseNodes = useMemo(() => (g.nodes || []).slice(0, MAX_NODES), [g.nodes]);
  const idset = useMemo(() => new Set(baseNodes.map((n) => n.id)), [baseNodes]);
  const baseEdges = useMemo(() => (g.edges || []).filter((e) => idset.has(e.src) && idset.has(e.dst)), [g.edges, idset]);

  const idsKey = useMemo(() => baseNodes.map((n) => n.id).join("|"), [baseNodes]);
  const layout = useMemo(
    () => dagreLayout(baseNodes.map((n) => n.id), baseEdges.map((e) => [e.src, e.dst] as [string, string]), direction),
    [idsKey, direction], // eslint-disable-line react-hooks/exhaustive-deps
  );

  const [sel, setSel] = useState<string | null>(focus);
  useEffect(() => { setSel(focus); }, [focus, idsKey]);

  // neighborhood for highlight
  const { hlNodes, hlEdges } = useMemo(() => {
    const hn = new Set<string>();
    const he = new Set<string>();
    if (sel) {
      hn.add(sel);
      baseEdges.forEach((e) => {
        if (e.src === sel) { hn.add(e.dst); he.add(e.src + "->" + e.dst); }
        if (e.dst === sel) { hn.add(e.src); he.add(e.src + "->" + e.dst); }
      });
    }
    return { hlNodes: hn, hlEdges: he };
  }, [sel, baseEdges]);

  const [nodes, setNodes, onNodesChange] = useNodesState<Node>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);
  const rawById = useRef<Record<string, GNode>>({});

  useEffect(() => {
    rawById.current = Object.fromEntries(baseNodes.map((n) => [n.id, n]));
    const any = hlNodes.size > 0;
    const nextNodes: Node[] = baseNodes.map((n) => ({
      id: n.id,
      type: "card",
      position: layout[n.id] || { x: 0, y: 0 },
      data: {
        label: String(n.label ?? n.id).replace(/^[a-z_]+:/, "").slice(0, 30),
        ntype: n.type || "",
        dim: any && !hlNodes.has(n.id),
        hl: hlNodes.has(n.id) && n.id !== sel,
        sel: n.id === sel,
      },
    }));
    // preserve dragged positions across rebuilds
    setNodes((prev) => {
      const pp = Object.fromEntries(prev.map((p) => [p.id, p.position]));
      return nextNodes.map((n) => (pp[n.id] ? { ...n, position: pp[n.id] } : n));
    });
    const nextEdges: Edge[] = baseEdges.map((e) => {
      const key = e.src + "->" + e.dst;
      const on = hlEdges.has(key);
      return {
        id: key,
        source: e.src,
        target: e.dst,
        type: "smoothstep", // orthogonal (horizontal/vertical) routing
        label: e.type || "",
        labelShowBg: true,
        labelBgPadding: [5, 2] as [number, number],
        labelBgBorderRadius: 6,
        labelBgStyle: { fill: "#0b1222", fillOpacity: 0.92, stroke: on ? "#2dd4bf" : "#1e293b", strokeWidth: 0.75 },
        labelStyle: { fill: on ? "#2dd4bf" : "#94a3b8", fontSize: 9, fontWeight: 600 },
        className: cls(on && "hl", any && !on && "dim"),
        markerEnd: { type: MarkerType.ArrowClosed, color: on ? "#2dd4bf" : "#334155", width: 14, height: 14 },
        style: { stroke: on ? "#2dd4bf" : "#334155", strokeWidth: on ? 2.2 : 1.1 },
      } as Edge;
    });
    setEdges(nextEdges);
  }, [baseNodes, baseEdges, layout, hlNodes, hlEdges, sel, setNodes, setEdges]);

  const onNodeClick = useCallback((_e: any, node: Node) => {
    setSel(node.id);
    const raw = rawById.current[node.id];
    if (raw) ctx.openNode(raw, persona);
  }, [ctx, persona]);

  const onPaneClick = useCallback(() => setSel(null), []);

  const types = useMemo(() => [...new Set(baseNodes.map((n) => n.type || "").filter(Boolean))], [baseNodes]);

  if (!baseNodes.length) {
    return <div className="graphwrap" style={{ height }}><div className="empty">no graph data in this frame</div></div>;
  }

  return (
    <div className="graphwrap" style={{ height }}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onNodeClick={onNodeClick}
        onPaneClick={onPaneClick}
        fitView
        minZoom={0.06}
        maxZoom={4}
        proOptions={{ hideAttribution: true }}
        nodesConnectable={false}
        elementsSelectable
        zoomOnScroll={false}
        preventScrolling={false}
        zoomOnPinch
        zoomOnDoubleClick
        panOnDrag
      >
        <Background variant={BackgroundVariant.Dots} gap={26} size={1} color="#1b2740" />
        <Controls showInteractive={false} />
      </ReactFlow>
      <div className="legend">
        {types.map((t) => (
          <span className="li" key={t}>
            <span className="sw" style={{ background: NODE_COLORS[t] || "#8b949e" }} /> {t}
          </span>
        ))}
        <div className="hint">drag node · click = focus + neighbors · drag bg = pan · ⊕/⊖ or pinch = zoom · scroll = page</div>
      </div>
    </div>
  );
}

// expose prefix routing for openNode (used by App)
export { OBJECT_PREFIX, PERSONA_PREFIX };
