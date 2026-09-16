import { useInspector } from "../ctx";
import { DataTable, StatusTag, Tag } from "../ui";
import type { Frame } from "../types";

export function useVisFilter() {
  const { view, agent } = useInspector();
  return (o: any) => (view !== "agent_visible" || !agent ? true : !o.visible_to_agents || o.visible_to_agents.includes(agent));
}

export function VizTag({ o }: { o: any }) {
  const { frame } = useInspector();
  if (!o.visible_to_agents) return null;
  const all = Object.keys(frame.agents || {}).length;
  return <Tag tone="viz">vis {o.visible_to_agents.length}/{all}</Tag>;
}

export function TaskTable({ tasks }: { tasks: any[] }) {
  const { openObj } = useInspector();
  const vis = useVisFilter();
  return (
    <DataTable
      rows={tasks.filter(vis)}
      onRowClick={(t: any) => openObj(t.task_id)}
      empty="none"
      columns={[
        { header: "id", cell: (t: any) => <code>{t.task_id}</code> },
        { header: "title", cell: (t: any) => t.title },
        { header: "status", cell: (t: any) => <StatusTag value={t.status} /> },
        { header: "pri", cell: (t: any) => t.priority },
        { header: "owner", cell: (t: any) => t.owner_id || "—" },
      ]}
    />
  );
}

export function MsgTable({ msgs }: { msgs: any[] }) {
  const { openObj } = useInspector();
  const vis = useVisFilter();
  return (
    <DataTable
      rows={msgs.filter(vis)}
      onRowClick={(m: any) => openObj(m.message_id)}
      empty="none"
      columns={[
        { header: "t", cell: (m: any) => m.created_tick },
        { header: "from", cell: (m: any) => m.sender_id },
        { header: "ch", cell: (m: any) => m.channel_id || "" },
        { header: "text", cell: (m: any) => String(m.full_text || m.text_summary || "").slice(0, 60) },
        { header: "read", cell: (m: any) => (m.read_by || []).length },
      ]}
    />
  );
}

export function MeetingTable({ meetings }: { meetings: any[] }) {
  const { openObj } = useInspector();
  return (
    <DataTable
      rows={meetings}
      onRowClick={(m: any) => openObj(m.meeting_id)}
      empty="none"
      columns={[
        { header: "id", cell: (m: any) => <code>{m.meeting_id}</code> },
        { header: "type", cell: (m: any) => m.meeting_type },
        { header: "status", cell: (m: any) => <StatusTag value={String(m.status)} /> },
        { header: "parts", cell: (m: any) => (m.participants || []).length },
        { header: "notes", cell: (m: any) => (m.notes ? <Tag tone="good">yes</Tag> : m.notes_missing ? <Tag tone="bad">missing</Tag> : "—") },
      ]}
    />
  );
}

export function TextTable({ rows }: { rows: any[] }) {
  return (
    <DataTable
      rows={rows}
      empty="none"
      columns={[
        { header: "t", cell: (x: any) => x.tick },
        { header: "type", cell: (x: any) => x.speech_act || x.type || x.kind || "—" },
        { header: "text", cell: (x: any) => String(x.text || x.full_text || x.summary || "").slice(0, 80) },
      ]}
    />
  );
}

export type PanelProps = { f: Frame };
