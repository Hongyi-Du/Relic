import { useCallback, useEffect, useRef, useState } from "react";
import { api, expandReplay } from "./api";
import type { Frame, ReplaySource } from "./types";

export const LIVE = "__live__";

/** Owns the frame buffer + live tailing + replay scrubbing + sim controls.
 * Mirrors the vanilla inspector's data layer (pollLive / expandReplay / sim/*). */
export function useSession() {
  const [frames, setFrames] = useState<Frame[]>([]);
  const [fi, setFi] = useState(0);
  const [mode, setMode] = useState<"live" | "replay">("live");
  const [running, setRunning] = useState(false);
  const [sources, setSources] = useState<ReplaySource[]>([]);
  const [source, setSource] = useState<string>(LIVE);
  const [playing, setPlaying] = useState(false);

  const framesRef = useRef<Frame[]>([]);
  const fiRef = useRef(0);
  const liveOn = useRef(false);
  const liveLast = useRef(-1);
  const liveTimer = useRef<number | null>(null);
  const playTimer = useRef<number | null>(null);

  const commitFrames = (fr: Frame[]) => { framesRef.current = fr; setFrames(fr); };
  const commitFi = (i: number) => { fiRef.current = i; setFi(i); };

  const setFrame = useCallback((i: number) => {
    const n = framesRef.current.length;
    commitFi(Math.max(0, Math.min(n - 1, i)));
  }, []);

  const stopPlay = useCallback(() => {
    setPlaying(false);
    if (playTimer.current) { clearInterval(playTimer.current); playTimer.current = null; }
  }, []);

  const stopLive = useCallback(() => {
    liveOn.current = false;
    if (liveTimer.current) { clearInterval(liveTimer.current); liveTimer.current = null; }
  }, []);

  const pollLive = useCallback(async () => {
    if (!liveOn.current) return;
    try {
      const d = await api.framesSince(liveLast.current);
      const fresh: Frame[] = d.frames || [];
      if (fresh.length) {
        const cur = framesRef.current;
        const wasEnd = fiRef.current >= cur.length - 1;
        const next = cur.concat(fresh);
        liveLast.current = next[next.length - 1].tick;
        commitFrames(next);
        if (wasEnd) commitFi(next.length - 1);
      }
      setRunning(!!d.running);
    } catch { /* transient fetch error — keep polling */ }
  }, []);

  const startLive = useCallback(() => {
    stopLive(); stopPlay();
    setMode("live"); setSource(LIVE);
    liveOn.current = true; liveLast.current = -1;
    commitFrames([]); commitFi(0);
    pollLive();
    liveTimer.current = window.setInterval(pollLive, 1500);
  }, [pollLive, stopLive, stopPlay]);

  const loadReplay = useCallback(async (name: string) => {
    stopLive(); stopPlay();
    setMode("replay"); setSource(name);
    const d = await api.getReplay(name);
    const fr = expandReplay(d);
    commitFrames(fr);
    commitFi(Math.max(0, fr.length - 1));
    setRunning(false);
  }, [stopLive, stopPlay]);

  const loadSources = useCallback(async () => {
    try {
      const d = await api.listReplays();
      setSources((d.replays || []).map((r: any) => ({ name: r.name, ticks: (r.meta && r.meta.ticks) || 0 })));
    } catch { /* ignore */ }
  }, []);

  const refreshLive = useCallback(async () => {
    const d = await api.framesSince(-1);
    const fr: Frame[] = d.frames || [];
    liveLast.current = fr.length ? fr[fr.length - 1].tick : -1;
    commitFrames(fr); commitFi(Math.max(0, fr.length - 1));
    setRunning(!!d.running);
  }, []);

  const selectSource = useCallback((value: string) => {
    if (value === LIVE) startLive(); else loadReplay(value);
  }, [startLive, loadReplay]);

  const togglePlay = useCallback(() => {
    if (playTimer.current) { stopPlay(); return; }
    setPlaying(true);
    playTimer.current = window.setInterval(() => {
      const n = framesRef.current.length;
      if (fiRef.current >= n - 1) { stopPlay(); return; }
      commitFi(fiRef.current + 1);
    }, 700);
  }, [stopPlay]);

  const simStep = useCallback(async (n: number) => {
    if (!liveOn.current) startLive();
    await api.step(n); liveLast.current = -1; await refreshLive();
  }, [startLive, refreshLive]);
  const simRun = useCallback(async (n: number) => {
    if (!liveOn.current) startLive();
    setRunning(true);
    await api.run(n); liveLast.current = -1; await refreshLive();
  }, [startLive, refreshLive]);
  const simPause = useCallback(async () => { await api.pause(); setRunning(false); }, []);
  const simReset = useCallback(async (seed: number) => { await api.reset(seed); startLive(); }, [startLive]);
  const exportReplay = useCallback(async () => {
    const d = await api.exportReplay("org_run_" + Date.now());
    await loadSources();
    return d;
  }, [loadSources]);

  useEffect(() => {
    loadSources();
    startLive();
    return () => { stopLive(); stopPlay(); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const frame: Frame | null = frames[fi] ?? null;

  return {
    frames, fi, frame, mode, running, sources, source, playing,
    setFrame, togglePlay, selectSource,
    simStep, simRun, simPause, simReset, exportReplay,
  };
}
