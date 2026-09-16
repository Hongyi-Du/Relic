import { useInspector } from "../ctx";
import GraphView from "../GraphView";
import { Cards, DataTable, Section, StatCard, Tag, num } from "../ui";
import type { PanelProps } from "./common";

export function ExtDashboard({ f }: PanelProps) {
  const { setSub } = useInspector();
  const X = f.external || {};
  const profiles = X.profiles || [], posts = X.posts || [];
  const sharedInt = posts.filter((p: any) => ((p.internal_exposure || {}).shared_to_internal_channels || []).length).length;
  const readInt = posts.filter((p: any) => ((p.read_by_internal_agents) || []).length).length;
  const roles: Record<string, number> = {};
  profiles.forEach((p: any) => { roles[p.role || "—"] = (roles[p.role || "—"] || 0) + 1; });
  return (
    <>
      <Cards>
        <StatCard label="profiles" value={profiles.length} onClick={() => setSub("profiles")} />
        <StatCard label="posts" value={posts.length} onClick={() => setSub("posts")} />
        <StatCard label="signals" value={(X.signals || []).length} onClick={() => setSub("signals")} />
        <StatCard label="offers" value={(X.offers || []).length} onClick={() => setSub("offers")} />
        <StatCard label="shared internally" value={sharedInt} onClick={() => setSub("flow")} />
        <StatCard label="read internally" value={readInt} onClick={() => setSub("flow")} />
      </Cards>
      <Section title="Roles">
        <div className="pillrow">{Object.entries(roles).map(([r, n]) => <Tag key={r}>{r} ({n})</Tag>)}</div>
      </Section>
    </>
  );
}

export function Profiles({ f }: PanelProps) {
  const { openObj } = useInspector();
  return (
    <Section title={`External profiles (${(f.external?.profiles || []).length})`}>
      <DataTable rows={f.external?.profiles || []} onRowClick={(p: any) => openObj(p.external_agent_id)} empty="none"
        columns={[
          { header: "id", cell: (p: any) => <code>{p.external_agent_id}</code> },
          { header: "name", cell: (p: any) => p.name },
          { header: "role", cell: (p: any) => p.role },
          { header: "org", cell: (p: any) => p.organization || p.org || "—" },
          { header: "cred", cell: (p: any) => num(p.credibility, 2) },
          { header: "topics", cell: (p: any) => (p.topics || []).join(", ") },
        ]} />
    </Section>
  );
}

export function Posts({ f }: PanelProps) {
  const { openObj } = useInspector();
  return (
    <Section title={`Posts (${(f.external?.posts || []).length})`}>
      <DataTable rows={f.external?.posts || []} onRowClick={(p: any) => openObj(p.post_id)} empty="none"
        columns={[
          { header: "id", cell: (p: any) => <code>{p.post_id}</code> },
          { header: "author", cell: (p: any) => p.author_id || p.author },
          { header: "topic", cell: (p: any) => p.topic },
          { header: "summary", cell: (p: any) => String(p.summary || p.text_summary || "").slice(0, 70) },
          { header: "reach", cell: (p: any) => p.reach ?? "—" },
          { header: "read int.", cell: (p: any) => (p.read_by_internal_agents || []).length },
          { header: "shared int.", cell: (p: any) => ((p.internal_exposure || {}).shared_to_internal_channels || []).length },
        ]} />
    </Section>
  );
}

export function Signals({ f }: PanelProps) {
  return (
    <Section title={`Signals (${(f.external?.signals || []).length})`}>
      <DataTable rows={f.external?.signals || []} empty="no signals yet (market shocks + external_signal events appear here)"
        columns={[
          { header: "id", cell: (s: any) => <code>{s.signal_id || s.id}</code> },
          { header: "type", cell: (s: any) => s.signal_type || s.type },
          { header: "tick", cell: (s: any) => s.tick },
          { header: "topic", cell: (s: any) => s.topic },
          { header: "source", cell: (s: any) => s.source || s.source_id || "—" },
        ]} />
    </Section>
  );
}

export function Offers({ f }: PanelProps) {
  return (
    <Section title={`External offers (${(f.external?.offers || []).length})`}>
      <DataTable rows={f.external?.offers || []} empty="no external offers yet"
        columns={[
          { header: "id", cell: (o: any) => <code>{o.offer_id || o.id}</code> },
          { header: "candidate", cell: (o: any) => o.candidate_id || o.candidate || "—" },
          { header: "source", cell: (o: any) => o.source || "—" },
          { header: "role", cell: (o: any) => o.role || "—" },
          { header: "status", cell: (o: any) => o.status || "—" },
        ]} />
    </Section>
  );
}

export function Network({ f }: PanelProps) {
  return <Section title="External Network Graph"><GraphView graph={f.graphs?.external_network} height={560} /></Section>;
}

export function Flow({ f }: PanelProps) {
  const posts = (f.external?.posts || []).filter((p: any) => ((p.read_by_internal_agents) || []).length || ((p.internal_exposure || {}).shared_to_internal_channels || []).length);
  return (
    <Section title="External → Internal information flow">
      {posts.length ? posts.map((p: any) => (
        <div className="kv" key={p.post_id}>
          <b>post by {p.author_id || p.author}</b>
          <span className="arrow" style={{ color: "var(--accent)", margin: "0 6px" }}>→</span>
          read by {(p.read_by_internal_agents || []).join(", ") || "—"}
          <span className="arrow" style={{ color: "var(--accent)", margin: "0 6px" }}>→</span>
          shared to {((p.internal_exposure || {}).shared_to_internal_channels || []).join(", ") || "—"}
        </div>
      )) : <div className="empty">no external→internal flow yet</div>}
    </Section>
  );
}
