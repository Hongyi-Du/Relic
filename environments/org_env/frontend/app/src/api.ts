import type { Frame } from "./types";

const j = (u: string) => fetch(u).then((r) => r.json());
const p = (u: string, b?: any) =>
  fetch(u, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(b || {}),
  }).then((r) => r.json());

export const api = {
  framesSince: (since: number) => j(`/api/org/lived/frames?since=${since}`),
  listReplays: () => j(`/api/org/lived/replays`),
  getReplay: (name: string) => j(`/api/org/lived/replay/${encodeURIComponent(name)}`),
  step: (n: number) => p(`/api/org/sim/step`, { n }),
  run: (n: number) => p(`/api/org/sim/run_ticks`, { n }),
  pause: () => p(`/api/org/sim/pause`, {}),
  reset: (seed: number) => p(`/api/org/sim/reset`, { seed }),
  exportReplay: (name: string) => p(`/api/org/sim/export`, { name }),
  productTry: (query: string) => p(`/api/org/product/try`, { query }),
  productFeedback: (rating: number, comment: string, query: string) =>
    p(`/api/org/product/feedback`, { rating, comment, query }),
  exportRepo: () => p(`/api/org/product/export`, {}),
};

// ---- incremental replay reconstruction (mirrors runtime_adapter/replay_delta.py) ----
function applyDelta(prev: any, delta: any): any {
  if (delta == null) return prev;
  const op = delta[0];
  if (op === "=") return delta[1];
  if (op === "a") return (prev || []).concat(delta[1]);
  if (op === "d") {
    const out: any = Object.assign({}, prev || {});
    const patch = delta[1] || {};
    for (const k in patch) out[k] = applyDelta(out[k], patch[k]);
    for (const k of delta[2] || []) delete out[k];
    return out;
  }
  return delta.length > 1 ? delta[1] : prev;
}

export function expandReplay(d: any): Frame[] {
  if (d && (d.format === "org-delta-v1" || (d.base !== undefined && d.deltas !== undefined))) {
    if (d.base == null) return [];
    const frames: Frame[] = [d.base];
    let cur = d.base;
    for (const dl of d.deltas || []) {
      cur = applyDelta(cur, dl);
      frames.push(cur);
    }
    // keyframed heavy sections (graphs/logs/timeline) are carried forward so every
    // frame still renders them (keyframe-fresh).
    const keys: string[] = d.keyframe_keys || [];
    if (keys.length) {
      const last: Record<string, any> = {};
      for (const fr of frames) {
        for (const k of keys) {
          if (k in fr) last[k] = fr[k];
          else if (k in last) fr[k] = last[k];
        }
      }
    }
    return frames;
  }
  return (d && d.frames) || [];
}
