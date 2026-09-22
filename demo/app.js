import { drawPerson, drawNameTag, drawBubble } from "./sprites.js";
import { loadWorkbench, renderPanels, workbenchSummary } from "./workbench.js";
import { initActs, startActs, stopActs, actsRunning } from "./acts.js";
import { t, getLang, toggleLang, onLangChange } from "./i18n.js";
import {
  W,
  H,
  DESK_SLOTS,
  ZONE_SPOTS,
  drawBackdrop,
  drawProps,
  drawLighting,
  drawZoneLabels,
} from "./scene.js";

const DATA_URL = "data/cattrs_b3_4013.json";
const GAP = 14; // gap between the two offices in compare mode

const VERB_LABEL = {
  edit_repo_file: "⌨️ coding",
  commit_patch: "📦 commit",
  merge_pr: "🎉 merge!",
  open_pr: "🔀 open PR",
  review_pr: "👀 reviewing",
  formal_pr_review: "🧐 formal review",
  approve_pr: "✅ LGTM",
  request_changes: "✋ changes",
  run_ci: "⚙️ CI run",
  run_public_tests: "🧪 tests",
  run_eval_stub: "📐 eval",
  approve_proposal: "📜 aye!",
  reject_proposal: "🚫 nay",
  propose_protocol: "💡 new rule!",
  amend_protocol: "✏️ amend rule",
  publish_product_release: "🚀 release",
  create_release_candidate: "🏗️ build RC",
  approve_release_candidate: "🛂 RC ok",
  work_on_task: "💼 working",
  update_task_status: "📋 status",
  internal_search: "🔎 digging",
  review_doc: "📄 doc review",
  pick_task: "🎯 pick task",
  assign_task_owner: "👉 assign",
  share_external_post: "📣 post",
  attend_meeting: "🗣️ meeting",
};

const TOASTS = {
  merge_pr: { cls: "merge", text: (a) => `${a} merged a PR into main` },
  propose_protocol: { cls: "protocol", text: (a) => `${a} proposed a new protocol` },
  amend_protocol: { cls: "protocol", text: (a) => `${a} amended a protocol` },
  publish_product_release: { cls: "ship", text: (a) => `${a} shipped a release` },
};

/** Solo-mode gauges read from the per-tick action counters. */
const GAUGES = [
  { key: "merges", label: "merged PRs", max: 40 },
  { key: "commits", label: "commits", max: 95 },
  { key: "edits", label: "file edits", max: 135 },
  { key: "ci_runs", label: "CI runs", max: 150 },
  { key: "protocol_votes", label: "protocol votes", max: 135, tag: "protocol" },
  { key: "releases", label: "releases", max: 45 },
];

/**
 * Compare-mode gauges read the official checkpoint metrics so the numbers match
 * the evaluator record rather than the raw action log.
 */
const PAIRS = [
  { key: "merged_prs", label: "merged PRs", max: 80 },
  { key: "releases", label: "releases", max: 50 },
  { key: "protocol_uses", label: "protocol uses", max: 1200, tag: "protocol" },
  { key: "protocol_enforcements", label: "enforcements", max: 700, tag: "protocol" },
];

const el = (id) => document.getElementById(id);

const stage = el("stage");
const ctx = stage.getContext("2d");
const trackCanvas = el("track");
const trackCtx = trackCanvas.getContext("2d");

let data;
let tick = 1;
let playing = false;
let speed = 450;
let timer = null;
let compare = false;
let lastAppliedTick = -1;

/* ------------------------------------------------------------------ world */

/** Each arm gets its own office: agent positions, particles, machine heat. */
function makeWorld(label, accent) {
  const agents = new Map();
  data.agents.forEach((name, i) => {
    const home = DESK_SLOTS[i];
    agents.set(name, {
      name,
      home,
      x: home.x,
      y: home.y + 26,
      tx: home.x,
      ty: home.y + 26,
      facing: 1,
      phase: Math.random() * 6,
      action: null,
      bubbleAge: 99,
    });
  });
  return { label, accent, agents, notes: [], ciHeat: 0, rocketLift: 0 };
}

function targetFor(zone, agent) {
  const spot = ZONE_SPOTS[zone];
  if (!spot) return { x: agent.home.x, y: agent.home.y + 26 };
  const jitter = (agent.name.charCodeAt(0) % 5) * 11 - 22;
  return { x: spot.x + jitter, y: spot.y + (agent.name.charCodeAt(1) % 3) * 9 };
}

/**
 * What each room announces when something happens in it, in priority order.
 * One line per room per hour, counted rather than repeated, so an hour with
 * four CI runs says "CI ×4" instead of stacking four notes on top of itself.
 */
// Each room is tinted, so the note that rises off it is a pale version of the
// room's own hue rather than the saturated one, which would read as green on
// green.
const ANNOUNCEMENTS = [
  { at: (a) => a.verb === "merge_pr", key: "fx.merged", x: 300, y: 372, color: "#b6ffd9" },
  { at: (a) => a.verb === "publish_product_release", key: "fx.shipped", x: 838, y: 430, color: "#ffd2a6" },
  { at: (a) => a.verb === "propose_protocol", key: "fx.proposed", x: 872, y: 196, color: "#ffb3cc" },
  { at: (a) => a.verb === "amend_protocol", key: "fx.amended", x: 872, y: 196, color: "#ffb3cc" },
  { at: (a) => a.verb === "approve_proposal" || a.verb === "reject_proposal", key: "fx.vote", x: 872, y: 196, color: "#ffb3cc" },
  { at: (a) => a.zone === "ci", key: "fx.ci", x: 640, y: 196, color: "#c9e9ff" },
];

/** A note that rises off the room where the work happened, then fades. */
function announce(world, text, x, y, color) {
  // Rooms that report several things in one hour stack upwards.
  const above = world.notes.filter((p) => Math.abs(p.x - x) < 60).length;
  world.notes.push({ text, x, y: y - above * 15, life: 1, color });
}

function toast(cls, text) {
  const layer = el("toasts");
  const node = document.createElement("div");
  node.className = `toast ${cls}`;
  node.textContent = text;
  layer.appendChild(node);
  setTimeout(() => node.remove(), 2600);
  if (layer.children.length > 4) layer.firstChild.remove();
}

function applyTick(world, row, animate, withToasts) {
  const byAgent = new Map();
  for (const a of row.actions) byAgent.set(a.agent, a);

  for (const agent of world.agents.values()) {
    const act = byAgent.get(agent.name) || null;
    agent.action = act;
    const tgt = targetFor(act?.zone, agent);
    agent.facing = tgt.x >= agent.x ? 1 : -1;
    agent.tx = tgt.x;
    agent.ty = tgt.y;
    if (act) agent.bubbleAge = 0;
  }

  if (!animate) return;

  for (const rule of ANNOUNCEMENTS) {
    const n = row.actions.filter(rule.at).length;
    if (!n) continue;
    const text = t(rule.key);
    announce(world, n > 1 ? `${text} ×${n}` : text, rule.x, rule.y, rule.color);
  }

  for (const a of row.actions) {
    if (a.verb === "publish_product_release") world.rocketLift = 1;
    if (a.zone === "ci") world.ciHeat = Math.min(1, world.ciHeat + 0.4);
    const toasted = withToasts && TOASTS[a.verb];
    if (toasted) toast(toasted.cls, toasted.text(a.agent));
  }
}

function drawNotes(world, dt) {
  world.notes = world.notes.filter((p) => {
    p.life -= dt * 0.5;
    p.y -= dt * 9; // a short lift, not enough to drift into the room above
    if (p.life <= 0) return false;
    ctx.save();
    // Hold at full strength, then fade over the last second: these are meant to
    // be read, not glimpsed.
    ctx.globalAlpha = Math.max(0, Math.min(1, p.life * 2));
    ctx.translate(p.x, p.y);
    ctx.font = "bold 15px 'Courier New', monospace";
    ctx.textAlign = "center";
    ctx.lineWidth = 3;
    ctx.strokeStyle = "rgba(8,10,16,0.9)";
    ctx.strokeText(p.text, 0, 0);
    ctx.fillStyle = p.color;
    ctx.fillText(p.text, 0, 0);
    ctx.restore();
    return true;
  });
}

/** Banner naming the arm, drawn inside the office in compare mode. */
function drawArmPlate(world, row, hud) {
  ctx.save();
  ctx.fillStyle = "rgba(8,10,16,0.82)";
  ctx.beginPath();
  ctx.roundRect(12, 86, 250, 26, 8);
  ctx.fill();
  ctx.strokeStyle = world.accent;
  ctx.lineWidth = 2;
  ctx.stroke();
  ctx.fillStyle = world.accent;
  ctx.font = "bold 13px 'Courier New', monospace";
  ctx.textAlign = "left";
  ctx.fillText(world.label, 22, 104);
  ctx.fillStyle = "rgba(230,237,243,0.75)";
  ctx.font = "11px 'Courier New', monospace";
  ctx.fillText(`${hud.merged_prs ?? 0} merged`, 168, 104);
  ctx.restore();
}

/** "No protocols here" stamp for the arm that has no institutional layer. */
function drawClosedStamp() {
  ctx.save();
  ctx.translate(876, 200);
  ctx.rotate(-0.18);
  ctx.strokeStyle = "rgba(248,81,73,0.75)";
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.roundRect(-74, -22, 148, 44, 6);
  ctx.stroke();
  ctx.fillStyle = "rgba(248,81,73,0.85)";
  ctx.font = "bold 15px 'Courier New', monospace";
  ctx.textAlign = "center";
  ctx.fillText("NO PROTOCOLS", 0, -1);
  ctx.font = "10px 'Courier New', monospace";
  ctx.fillText("roles only", 0, 14);
  ctx.restore();
}

/** Draw one complete office into the current transform (1000x500 space). */
function drawWorld(world, row, hud, now, dt, opts = {}) {
  const { plate = false, closedProtocol = false } = opts;

  world.ciHeat = Math.max(0, world.ciHeat - dt * 0.5);
  world.rocketLift = world.rocketLift > 0.01 ? world.rocketLift * 0.94 : 0;

  ctx.save();
  drawBackdrop(ctx, row.hour, now / 1000);

  const busyDesks = new Set();
  data.agents.forEach((name, i) => {
    const a = world.agents.get(name);
    if (a.action && (a.action.zone === "code" || a.action.zone === "desk")) {
      busyDesks.add(i);
    }
  });

  drawProps(ctx, {
    t: now / 1000,
    hour: row.hour,
    busyDesks,
    ciHeat: world.ciHeat,
    notes:
      row.totals.protocols_proposed * 4 + Math.floor(row.totals.protocol_votes / 8),
    merges: row.totals.merges,
    rocketLift: world.rocketLift * 54,
  });
  drawZoneLabels(ctx);
  if (closedProtocol) drawClosedStamp();

  const list = [...world.agents.values()].sort((a, b) => a.y - b.y);
  const lit = [];
  for (const a of list) {
    const dx = a.tx - a.x;
    const dy = a.ty - a.y;
    const dist = Math.hypot(dx, dy);
    const moving = dist > 2;
    if (moving) {
      const step = Math.min(dist, 150 * dt);
      a.x += (dx / dist) * step;
      a.y += (dy / dist) * step;
      a.phase += dt * 11;
    } else {
      a.phase += dt * 2;
    }
    a.bubbleAge += dt;
    drawPerson(ctx, a.name, a.x, a.y, {
      facing: a.facing,
      walkPhase: a.phase,
      moving,
      glow: a.action?.highlight ? "rgba(255,209,102,0.5)" : null,
    });
    drawNameTag(ctx, a.name, a.x, a.y, Boolean(a.action));
    if (a.action) lit.push({ x: a.x, y: a.y - 20, r: 70 });
  }

  for (const a of list) {
    if (!a.action || a.bubbleAge > 2.4) continue;
    const label = VERB_LABEL[a.action.verb] || a.action.verb.replace(/_/g, " ");
    const alpha = Math.min(1, (2.4 - a.bubbleAge) * 1.6);
    drawBubble(ctx, label, a.x, a.y - 46, alpha, a.action.highlight ? "#ffd166" : "#93a4bd");
  }

  drawLighting(ctx, row.hour, lit);

  // Last, above the night dimming and above the speech bubbles: a merge is
  // announced exactly where someone is standing and talking, and these notes
  // are now the only thing saying what happened.
  drawNotes(world, dt);
  if (plate) drawArmPlate(world, row, hud);
  ctx.restore();
}

/* ------------------------------------------------------------------ state */

let worldB3;
let worldB2;

/**
 * Checkpoints only land every 24 ticks. Before the first one the run genuinely
 * has no recorded totals yet, so report zeros rather than leaking t24's values
 * backwards into the opening hours.
 */
function hudAt(checkpoints, t) {
  const first = checkpoints[0];
  if (!first) return {};
  if (t < first.tick) {
    return Object.fromEntries(
      Object.entries(first).map(([k, v]) => [k, typeof v === "number" ? 0 : v])
    );
  }
  let hud = first;
  for (const row of checkpoints) {
    if (row.tick <= t) hud = row;
    else break;
  }
  return hud;
}

const cmp = () => data.compare_track;

/* ------------------------------------------------------------------ frame */

let lastFrame = performance.now();

function frame(now) {
  const dt = Math.min(0.05, (now - lastFrame) / 1000);
  lastFrame = now;
  const row = data.ticks[tick - 1];

  ctx.clearRect(0, 0, stage.width, stage.height);

  if (!compare) {
    drawWorld(worldB3, row, hudAt(data.checkpoints, tick), now, dt);
  } else {
    // B2 on the left, B3 on the right, both driven by the same hour.
    ctx.save();
    drawWorld(worldB2, cmp().ticks[tick - 1], hudAt(cmp().checkpoints, tick), now, dt, {
      plate: true,
      closedProtocol: true,
    });
    ctx.restore();

    ctx.save();
    ctx.translate(W + GAP, 0);
    drawWorld(worldB3, row, hudAt(data.checkpoints, tick), now, dt, { plate: true });
    ctx.restore();

    ctx.fillStyle = "#0a0c12";
    ctx.fillRect(W, 0, GAP, H);
    ctx.fillStyle = "rgba(255,209,102,0.5)";
    ctx.fillRect(W + GAP / 2 - 1, 0, 2, H);
  }

  requestAnimationFrame(frame);
}

/* --------------------------------------------------------------------- UI */

const lastGauge = {};

function buildGauges() {
  lastGauge.__mode = compare ? "cmp" : "solo";
  if (!compare) {
    el("gauges").className = "gauges";
    el("gauges").innerHTML = GAUGES.map(
      (g) => `
      <div class="gauge" data-k="${g.tag || g.key}">
        <div class="v" id="g-${g.key}">0</div>
        <div class="k">${g.label}</div>
        <div class="bar" id="b-${g.key}" style="width:0%"></div>
      </div>`
    ).join("");
    return;
  }

  el("gauges").className = "gauges paired";
  el("gauges").innerHTML = PAIRS.map(
    (g) => `
      <div class="gauge pair" data-k="${g.tag || g.key}">
        <div class="k">${g.label}</div>
        <div class="pair-row b2">
          <span class="tagname">B2</span>
          <span class="v" id="p2-${g.key}">0</span>
          <div class="minibar"><i id="m2-${g.key}"></i></div>
        </div>
        <div class="pair-row b3">
          <span class="tagname">B3</span>
          <span class="v" id="p3-${g.key}">0</span>
          <div class="minibar"><i id="m3-${g.key}"></i></div>
        </div>
      </div>`
  ).join("");
}

function bump(node, value, key) {
  if (lastGauge[key] === value) return;
  node.textContent = value;
  node.classList.remove("bump");
  void node.offsetWidth;
  node.classList.add("bump");
  lastGauge[key] = value;
}

function updateGauges(row) {
  if (!compare) {
    for (const g of GAUGES) {
      const v = row.totals[g.key] ?? 0;
      bump(el(`g-${g.key}`), v, g.key);
      el(`b-${g.key}`).style.width = `${Math.min(100, (v / g.max) * 100)}%`;
    }
    return;
  }
  const h3 = hudAt(data.checkpoints, tick);
  const h2 = hudAt(cmp().checkpoints, tick);
  for (const g of PAIRS) {
    const v2 = h2[g.key] ?? 0;
    const v3 = h3[g.key] ?? 0;
    bump(el(`p2-${g.key}`), v2, `2${g.key}`);
    bump(el(`p3-${g.key}`), v3, `3${g.key}`);
    el(`m2-${g.key}`).style.width = `${Math.min(100, (v2 / g.max) * 100)}%`;
    el(`m3-${g.key}`).style.width = `${Math.min(100, (v3 / g.max) * 100)}%`;
  }
}

function drawTrack() {
  const c = trackCanvas;
  const w = c.width;
  const h = c.height;
  trackCtx.clearRect(0, 0, w, h);

  const maxAct = Math.max(...data.ticks.map((t) => t.actions.length));
  data.ticks.forEach((t, i) => {
    const x = (i / data.ticks.length) * w;
    const bw = w / data.ticks.length + 0.6;
    const bh = (t.actions.length / maxAct) * (h - 16);
    const hot = t.actions.some((a) => a.verb === "merge_pr");
    const proto = t.actions.some((a) =>
      ["propose_protocol", "amend_protocol"].includes(a.verb)
    );
    trackCtx.fillStyle = proto
      ? "#ff6b9d"
      : hot
        ? "#3ec07a"
        : t.night
          ? "#2f3b5c"
          : "#4a5b86";
    trackCtx.fillRect(x, h - 8 - bh, bw, bh);
  });

  trackCtx.fillStyle = "rgba(255,255,255,0.07)";
  for (let d = 24; d < data.max_tick; d += 24) {
    trackCtx.fillRect((d / data.max_tick) * w, 0, 1, h);
  }

  if (compare && cmp().split_tick) {
    const x = (cmp().split_tick / data.max_tick) * w;
    trackCtx.fillStyle = "#f85149";
    trackCtx.fillRect(x - 1, 0, 2, h);
  }

  for (const m of data.milestones) {
    const x = (m.tick / data.max_tick) * w;
    trackCtx.fillStyle = "#ffd166";
    trackCtx.fillRect(x - 1, 0, 2, h);
    trackCtx.beginPath();
    trackCtx.moveTo(x, 2);
    trackCtx.lineTo(x + 14, 7);
    trackCtx.lineTo(x, 12);
    trackCtx.fill();
  }

  const px = (tick / data.max_tick) * w;
  trackCtx.fillStyle = "rgba(255,255,255,0.9)";
  trackCtx.fillRect(px - 1, 0, 2, h);
}

function updateChrome(row) {
  el("clock").textContent = row.clock;
  el("day").textContent = `DAY ${row.day} · ${row.phase.replace(/_/g, " ")}`;
  el("tickOut").textContent = tick;
  el("tick").value = tick;
  updateGauges(row);
  renderPanels(tick);
  drawTrack();
}

function setTick(n, animate = true) {
  const next = Math.max(1, Math.min(data.max_tick, n));
  const jumped = Math.abs(next - tick) > 1;
  tick = next;
  const row = data.ticks[tick - 1];
  if (tick !== lastAppliedTick) {
    applyTick(worldB3, row, animate && !jumped, true);
    applyTick(worldB2, cmp().ticks[tick - 1], compare && animate && !jumped, false);
    lastAppliedTick = tick;
  }
  updateChrome(row);
}

function togglePlay() {
  playing = !playing;
  el("play").textContent = playing ? "⏸" : "▶";
  clearInterval(timer);
  if (playing) {
    timer = setInterval(() => {
      if (tick >= data.max_tick) return togglePlay();
      setTick(tick + 1);
    }, speed);
  }
}

function toggleCompare() {
  compare = !compare;
  stage.width = compare ? W * 2 + GAP : W;
  stage.height = H;
  document.body.classList.toggle("compare", compare);
  el("vs").classList.toggle("on", compare);
  el("vs").textContent = compare ? "SOLO VIEW" : "VS B2";
  buildGauges();
  for (const k of Object.keys(lastGauge)) delete lastGauge[k];
  // Re-seat both offices at the current hour so neither lags behind.
  lastAppliedTick = -1;
  setTick(tick, false);
  renderVerdict();
}

/** Solo view is about this one company, not the cross-arm horse race. */
function renderSoloOutcome() {
  const wb = workbenchSummary();
  const oss = wb.tasks.filter((t) => t.oss);
  const merged = oss.filter((t) => ["merged", "done"].includes(t.final_status)).length;
  const adopted = wb.protocols.filter((p) => p.status === "adopted").length;
  const vetoed = wb.proposals.filter((p) => p.status === "rejected").length;
  const blocked = wb.protocols.reduce((n, p) => n + (p.totals.enforcement || 0), 0);
  const ev = wb.evaluation;

  const cards = [
    { nm: "cattrs issues closed", v: `${merged} / ${oss.length}` },
    { nm: "protocols adopted", v: `${adopted} / ${wb.protocols.length}` },
    { nm: "proposals vetoed", v: `${vetoed} / ${wb.proposals.length}` },
    { nm: "merges blocked by rules", v: blocked.toLocaleString() },
    { nm: "hidden contracts passed", v: `${ev.passed.length} / ${wb.product.hidden_tests}`, win: true },
  ];
  el("arms").innerHTML = cards
    .map(
      (c) => `<div class="arm ${c.win ? "win" : ""}">
        <div class="nm">${c.nm}</div>
        <div class="pct">${c.v}</div>
      </div>`
    )
    .join("");

  el("note").innerHTML =
    `The team shipped <b>${merged} of ${oss.length}</b> cattrs issues and wrote <b>${adopted} rules</b> ` +
    `that went on to block <b>${blocked.toLocaleString()} merge attempts</b>. The hidden evaluator ` +
    `then passed <b>${ev.passed.map((p) => `<code>${p}</code>`).join(", ")}</b> — ` +
    `${ev.passed.length} of ${wb.product.hidden_tests}. Shipping and being correct are not the same thing, ` +
    `which is the whole point of the run.`;

  el("sub").textContent =
    `${wb.product.label} · ${wb.product.gaps.length} open issues · ` +
    `${wb.product.hidden_tests} hidden tests · seed 4013`;
}

function renderVerdict() {
  if (!compare) return renderSoloOutcome();
  const arms = ["b0", "b1", "b2", "b3"];
  const names = {
    b0: "B0 solo founder",
    b1: "B1 chat team",
    b2: "B2 roles",
    b3: "B3 institutions",
  };
  // For b2/b3 the cards read the same checkpoint series the gauges animate,
  // so the live counters land exactly on the final numbers.
  const finals = {
    b2: data.compare_track.checkpoints.at(-1),
    b3: data.checkpoints.at(-1),
  };
  el("arms").innerHTML = arms
    .map((a) => {
      const c = data.compare_arms[a];
      const fin = finals[a];
      const pct = c.pass_rate != null ? `${(c.pass_rate * 100).toFixed(1)}%` : "—";
      const merged = fin?.merged_prs ?? c.merged_prs ?? 0;
      const uses = fin?.protocol_uses ?? c.protocol_uses ?? 0;
      const lit = compare ? a === "b2" || a === "b3" : a === "b3";
      return `<div class="arm ${a === "b3" ? "win" : ""} ${lit ? "" : "faded"}">
        <div class="nm">${names[a]}</div>
        <div class="pct">${pct}</div>
        <div class="sub">${merged} merged · ${uses} protocol uses</div>
      </div>`;
    })
    .join("");

  const f2 = finals.b2;
  const f3 = finals.b3;
  const n2 = data.final_eval_b2.passed_contracts.length;
  const n3 = data.final_eval_b3.passed_contracts.length;
  el("note").innerHTML = compare
    ? `Same seed, same model, same 12 issues — only the organization differs. ` +
      `B2 actually <b>ships slightly more</b> (${f2.merged_prs} merged PRs, ${f2.releases} releases ` +
      `vs B3's ${f3.merged_prs} and ${f3.releases}) with <b>zero</b> protocol machinery. ` +
      `B3 spends its extra effort on <b>${f3.protocol_uses.toLocaleString()} protocol uses and ` +
      `${f3.protocol_enforcements.toLocaleString()} enforcements</b>, and ends up passing ` +
      `<b>${n3} hidden contracts to B2's ${n2}</b>. The two offices are identical until ` +
      `<b>t${cmp().split_tick}</b> — watch them drift apart from there.`
    : `Same task, same seed, same model. B3 ends with <b>${data.final_eval_b3.passed_contracts.join(", ")}</b> ` +
      `passing the hidden evaluator; B2 only gets <b>${data.final_eval_b2.passed_contracts.join(", ")}</b>. ` +
      `Parity run on DeepSeek V4 Flash — not the paper's headline models.`;

  el("sub").textContent = "cattrs migration · B2 vs B3 · seed 4013 · same 336 ticks";
}

async function main() {
  [data] = await Promise.all([fetch(DATA_URL).then((r) => r.json()), loadWorkbench()]);
  worldB3 = makeWorld(cmp().b3_label, "#3ec07a");
  worldB2 = makeWorld(cmp().label, "#7dd3fc");
  buildGauges();
  renderVerdict();
  el("tick").max = data.max_tick;
  setTick(1, false);

  initActs({
    detail: workbenchSummary(),
    setTick,
    maxTick: data.max_tick,
    exitCompare: () => compare && toggleCompare(),
  });

  el("tick").addEventListener("input", (e) => {
    if (actsRunning()) stopActs();
    setTick(Number(e.target.value));
  });
  el("play").addEventListener("click", () => {
    if (actsRunning()) stopActs();
    togglePlay();
  });
  el("vs").addEventListener("click", toggleCompare);
  el("lang").addEventListener("click", toggleLang);
  onLangChange((lang) => {
    el("lang").textContent = lang === "en" ? "中文" : "EN";
    document.documentElement.lang = lang === "en" ? "en" : "zh";
  });
  el("lang").textContent = getLang() === "en" ? "中文" : "EN";
  el("story").addEventListener("click", () => (actsRunning() ? stopActs() : startActs()));
  document.querySelectorAll(".spd").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".spd").forEach((b) => b.classList.remove("on"));
      btn.classList.add("on");
      speed = Number(btn.dataset.speed);
      if (playing) {
        togglePlay();
        togglePlay();
      }
    });
  });
  window.addEventListener("keydown", (e) => {
    if (e.key === " ") {
      e.preventDefault();
      if (actsRunning()) stopActs();
      togglePlay();
    }
    if (e.key === "ArrowRight") setTick(tick + 1);
    if (e.key === "ArrowLeft") setTick(tick - 1);
    if (e.key.toLowerCase() === "v") toggleCompare();
    if (e.key.toLowerCase() === "s") (actsRunning() ? stopActs : startActs)();
  });

  requestAnimationFrame(frame);
}

main().catch((err) => {
  document.body.innerHTML = `<pre style="color:#f88;padding:2rem">${err}\n\nServe from demo/:\n  python3 -m http.server</pre>`;
});
