// Seat state: claim a member, then poll their view.
//
// Polling rather than a socket, matching the inspector. The server already
// serves a cached per-seat snapshot, so a poll is a dictionary read and never
// waits on a tick that is talking to an LLM.

import { useCallback, useEffect, useRef, useState } from "react";
import { RuntimeStatus, SeatView, api } from "./api";

const POLL_MS = 1200;
const TOKEN_KEY = "socio.seat.token";

export type SeatSession = {
  token: string | null;
  view: SeatView | null;
  runtime: RuntimeStatus | null;
  members: any[];
  error: string;
  busy: boolean;
  claim: (agentId: string) => Promise<void>;
  leave: () => Promise<void>;
  act: (actionType: string, params: Record<string, any>, mode?: string) => Promise<any>;
  refresh: () => Promise<void>;
  startClock: () => Promise<void>;
  pauseClock: () => Promise<void>;
};

export function useSeat(): SeatSession {
  const [token, setToken] = useState<string | null>(
    () => localStorage.getItem(TOKEN_KEY),
  );
  const [view, setView] = useState<SeatView | null>(null);
  const [runtime, setRuntime] = useState<RuntimeStatus | null>(null);
  const [members, setMembers] = useState<any[]>([]);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const version = useRef(-1);
  const timer = useRef<number | null>(null);

  const loadMembers = useCallback(async () => {
    const d = await api.members();
    setMembers(d.members || []);
  }, []);

  const poll = useCallback(async (tok: string) => {
    try {
      const d = await api.view(tok, version.current);
      if (d.error) {
        // A token stops working when the seat is released or the world reset.
        setError("");
        setToken(null);
        localStorage.removeItem(TOKEN_KEY);
        return;
      }
      setRuntime(d.runtime || null);
      if (!d.unchanged && d.view) {
        version.current = d.view.version;
        // Newer snapshots carry the brief inline.  During the P2 rollout the
        // same permission-filtered projection is exposed by the small brief
        // endpoint, so older snapshots still get the onboarding surface.
        let nextView = d.view;
        if (!nextView.organization_brief) {
          try {
            const brief = await api.brief(tok);
            if (brief?.brief) nextView = { ...nextView, organization_brief: brief.brief };
          } catch (_) {
            // The brief is an enhancement; the seat remains usable from its
            // existing public view if the endpoint is not deployed yet.
          }
        }
        setView(nextView);
      }
    } catch (e: any) {
      setError(String(e));
    }
  }, []);

  useEffect(() => {
    loadMembers();
  }, [loadMembers]);

  useEffect(() => {
    if (!token) {
      if (timer.current) window.clearInterval(timer.current);
      timer.current = null;
      return;
    }
    version.current = -1;
    poll(token);
    timer.current = window.setInterval(() => poll(token), POLL_MS);
    return () => {
      if (timer.current) window.clearInterval(timer.current);
    };
  }, [token, poll]);

  const claim = useCallback(async (agentId: string) => {
    setBusy(true);
    setError("");
    try {
      const d = await api.claim(agentId);
      if (d.error) {
        setError(d.error);
        await loadMembers();
        return;
      }
      localStorage.setItem(TOKEN_KEY, d.token);
      setToken(d.token);
    } finally {
      setBusy(false);
    }
  }, [loadMembers]);

  const leave = useCallback(async () => {
    if (token) await api.release(token);
    localStorage.removeItem(TOKEN_KEY);
    setToken(null);
    setView(null);
    await loadMembers();
  }, [token, loadMembers]);

  const refresh = useCallback(async () => {
    if (token) {
      version.current = -1;
      await poll(token);
    }
  }, [token, poll]);

  const act = useCallback(
    async (actionType: string, params: Record<string, any>, mode = "direct") => {
      if (!token) return { error: "no_seat" };
      setBusy(true);
      setError("");
      try {
        const out = await api.act(token, actionType, params, mode);
        if (out.error) setError(out.error);
        else if (out.ok === false && out.failure_reason)
          setError(`${actionType}: ${out.failure_reason}`);
        await refresh();
        return out;
      } finally {
        setBusy(false);
      }
    },
    [token, refresh],
  );

  const startClock = useCallback(async () => {
    setRuntime(await api.runtimeStart());
  }, []);
  const pauseClock = useCallback(async () => {
    setRuntime(await api.runtimePause());
  }, []);

  return { token, view, runtime, members, error, busy,
           claim, leave, act, refresh, startClock, pauseClock };
}
