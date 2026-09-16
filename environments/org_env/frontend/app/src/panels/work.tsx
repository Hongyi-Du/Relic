import { useInspector } from "../ctx";
import { DataTable, Section, Tag } from "../ui";
import { MeetingTable, MsgTable, TaskTable, VizTag, useVisFilter, type PanelProps } from "./common";

export function Tasks({ f }: PanelProps) {
  const ts = f.internal?.tasks || [];
  return <Section title={`Tasks (${ts.length})`}><TaskTable tasks={ts} /></Section>;
}

export function Messages({ f }: PanelProps) {
  const chs = f.internal?.channels || [];
  const ms = (f.internal?.messages || []).slice().reverse();
  return (
    <>
      <Section title="Channels">
        <div className="pillrow">{chs.map((c: any) => <Tag key={c.channel_id}>#{c.channel_id} ({c.message_count})</Tag>)}</div>
      </Section>
      <Section title={`Messages (${(f.internal?.messages || []).length})`}><MsgTable msgs={ms} /></Section>
    </>
  );
}

export function Meetings({ f }: PanelProps) {
  const ms = f.internal?.meetings || [];
  return <Section title={`Meetings (${ms.length})`}><MeetingTable meetings={ms} /></Section>;
}

export function Docs({ f }: PanelProps) {
  const { openObj, openFile } = useInspector();
  const vis = useVisFilter();
  const ds = (f.internal?.docs || []).filter(vis);
  const fs = (f.internal?.files || []).filter(vis);
  return (
    <>
      <Section title={`Documents (${ds.length})`}>
        <DataTable
          rows={ds}
          onRowClick={(d: any) => openObj(d.doc_id)}
          empty="none"
          columns={[
            { header: "id", cell: (d: any) => <code>{d.doc_id}</code> },
            { header: "title", cell: (d: any) => d.title },
            { header: "type", cell: (d: any) => d.doc_type },
            { header: "owner", cell: (d: any) => d.owner_id || "" },
            { header: "vis", cell: (d: any) => <VizTag o={d} /> },
          ]}
        />
      </Section>
      <Section title={`Files (${fs.length})`}>
        <DataTable
          rows={fs}
          onRowClick={(d: any) => openFile(d.object_id)}
          empty="none"
          columns={[
            { header: "id", cell: (d: any) => <code>{d.object_id}</code> },
            { header: "title", cell: (d: any) => d.title },
            { header: "type", cell: (d: any) => d.file_type },
            { header: "owner", cell: (d: any) => d.owner_id || "" },
            { header: "vis", cell: (d: any) => <VizTag o={d} /> },
          ]}
        />
      </Section>
    </>
  );
}
