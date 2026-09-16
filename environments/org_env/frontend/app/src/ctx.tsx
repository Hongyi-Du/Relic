import { createContext, useContext } from "react";
import type { Frame, GNode, Graph, ReaderDoc, ViewMode } from "./types";

export interface InspectorCtx {
  frame: Frame;
  view: ViewMode;
  agent: string | null;
  setSub: (s: string) => void;
  selectAgent: (a: string) => void;
  openObj: (id: string) => void;
  openFile: (id: string) => void;
  openArtifact: (id: string) => void;
  openEpisode: (id: string) => void;
  openReflection: (id: string) => void;
  openWish: (id: string) => void;
  openProposal: (id: string) => void;
  openNode: (node: GNode, persona?: Graph) => void;
  openKV: (title: string, obj: Record<string, any>) => void;
  openReader: (doc: ReaderDoc) => void;
}

export const Ctx = createContext<InspectorCtx | null>(null);

export function useInspector(): InspectorCtx {
  const c = useContext(Ctx);
  if (!c) throw new Error("useInspector must be used within <Ctx.Provider>");
  return c;
}
