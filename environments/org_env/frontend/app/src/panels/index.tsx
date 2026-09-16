import { useInspector } from "../ctx";
import type { Frame } from "../types";
import { AgentGrid, Dashboard } from "./dashboard";
import { AgentDetail } from "./agentDetail";
import { Docs, Meetings, Messages, Tasks } from "./work";
import { Budget, Protocols, Repo, Sandbox } from "./repo";
import { Episodes, EventGraph, LLM, ProductSub, Proposals, Reflections } from "./cognition";
import { ExtDashboard, Flow, Network, Offers, Posts, Profiles, Signals } from "./external";

export function InternalPanel({ sub, f }: { sub: string; f: Frame }) {
  const { agent } = useInspector();
  switch (sub) {
    case "dashboard": return <Dashboard f={f} />;
    case "product": return <ProductSub f={f} />;
    case "episodes": return <Episodes f={f} />;
    case "reflections": return <Reflections f={f} />;
    case "llm": return <LLM f={f} />;
    case "proposals": return <Proposals f={f} />;
    case "agents": return agent ? <AgentDetail f={f} /> : <AgentGrid f={f} />;
    case "tasks": return <Tasks f={f} />;
    case "docs": return <Docs f={f} />;
    case "messages": return <Messages f={f} />;
    case "meetings": return <Meetings f={f} />;
    case "repo": return <Repo f={f} />;
    case "sandbox": return <Sandbox f={f} />;
    case "budget": return <Budget f={f} />;
    case "protocols": return <Protocols f={f} />;
    case "event graph": return <EventGraph f={f} />;
    default: return <Dashboard f={f} />;
  }
}

export function ExternalPanel({ sub, f }: { sub: string; f: Frame }) {
  switch (sub) {
    case "dashboard": return <ExtDashboard f={f} />;
    case "profiles": return <Profiles f={f} />;
    case "posts": return <Posts f={f} />;
    case "signals": return <Signals f={f} />;
    case "network": return <Network f={f} />;
    case "offers": return <Offers f={f} />;
    case "flow": return <Flow f={f} />;
    default: return <ExtDashboard f={f} />;
  }
}
