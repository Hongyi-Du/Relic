/**
 * The three content panels around the office: what the team is working on,
 * what rules they argued into existence, and what they actually said.
 *
 * Everything here is a pure function of the current tick, so scrubbing
 * backwards is as correct as playing forwards.
 */

import { t, onLangChange } from "./i18n.js";

const DETAIL_URL = "data/b3_detail_4013.json";

const STATUS_RANK = {
  open: 0,
  in_progress: 1,
  implementation_done: 2,
  done: 3,
  merged: 4,
};

const CHANNEL_CLASS = {
  engineering: "ch-eng",
  team_general: "ch-gen",
  customer_feedback: "ch-cust",
  experiments: "ch-exp",
};

/** Panel headings, which are chrome rather than run data, so they translate. */
const HEADINGS = {
  workHead: "wb.work",
  govHead: "wb.gov",
  feedHead: "wb.channels",
};

const el = (id) => document.getElementById(id);
const esc = (s) =>
  String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

let detail = null;
/** protocolId -> { use: Int32Array, violation: ..., enforcement: ... } prefix sums */
let protoPrefix = new Map();
let lastRendered = -1;

export async function loadWorkbench() {
  detail = await fetch(DETAIL_URL).then((r) => r.json());
  buildPrefixes();
  wireExpanders();
  onLangChange(() => {
    for (const [id, key] of Object.entries(HEADINGS)) el(id).textContent = t(key);
    const tick = lastRendered;
    lastRendered = -1;
    if (tick >= 0) renderPanels(tick);
  });
  for (const [id, key] of Object.entries(HEADINGS)) el(id).textContent = t(key);
  return detail;
}

function buildPrefixes() {
  const n = detail.max_tick + 1;
  for (const p of detail.protocols) {
    const acc = {
      use: new Int32Array(n),
      violation: new Int32Array(n),
      enforcement: new Int32Array(n),
    };
    for (const [tickStr, counts] of Object.entries(p.per_tick)) {
      const at = Number(tickStr);
      if (at >= n) continue;
      for (const key of Object.keys(acc)) {
        if (counts[key]) acc[key][at] += counts[key];
      }
    }
    for (const key of Object.keys(acc)) {
      const arr = acc[key];
      for (let i = 1; i < n; i++) arr[i] += arr[i - 1];
    }
    protoPrefix.set(p.id, acc);
  }
}

/** Cards expand in place rather than opening a modal. */
function wireExpanders() {
  for (const id of ["workList", "govList", "feedList"]) {
    el(id).addEventListener("click", (e) => {
      const card = e.target.closest("[data-more]");
      if (card) card.classList.toggle("open");
    });
  }
}

/* --------------------------------------------------------------- the work */

function taskStateAt(task, tick) {
  let status = "open";
  let last = null;
  for (const h of task.history) {
    if (h.tick > tick) break;
    status = h.to || status;
    last = h;
  }
  return { status, last, justNow: last?.tick === tick };
}

function renderWork(tick) {
  const tasks = detail.tasks.filter((x) => x.oss);
  const rows = [];
  const tally = { merged: 0, active: 0, open: 0 };

  for (const task of tasks) {
    const { status, last, justNow } = taskStateAt(task, tick);
    if (status === "merged" || status === "done") tally.merged++;
    else if (status === "open") tally.open++;
    else tally.active++;

    const trail = last
      ? `t${last.tick} · ${esc(last.agent)} · ${esc(last.via.replace(/_/g, " "))}`
      : t("wb.notStarted");
    // Anything just touched jumps to the top, then live work, then the done pile.
    rows.push({
      key: [justNow ? 0 : 1, STATUS_RANK[status] ?? 0, -(last?.tick ?? 0)],
      html: `
      <div class="card task s-${status} ${justNow ? "just" : ""}" data-more>
        <div class="card-top">
          <i class="dot"></i>
          <span class="card-title">${esc(task.title)}</span>
          <span class="pill">${t(`wb.status.${status}`)}</span>
        </div>
        <div class="card-sub">${trail}</div>
        <div class="more">
          <p>${esc(task.detail) || t("wb.noDetail")}</p>
          <div class="trail">${task.history
            .filter((h) => h.tick <= tick)
            .map(
              (h) =>
`<span><b>t${h.tick}</b> ${esc(h.agent)} → ${esc(t(`wb.status.${h.to}`))}</span>`
            )
            .join("")}</div>
        </div>
      </div>`,
    });
  }

  rows.sort((a, b) => {
    for (let i = 0; i < a.key.length; i++) {
      if (a.key[i] !== b.key[i]) return a.key[i] - b.key[i];
    }
    return 0;
  });
  el("workCount").textContent = t("wb.workCount", { m: tally.merged, a: tally.active, o: tally.open });
  el("workList").innerHTML = rows.map((r) => r.html).join("");
}

/* ------------------------------------------------------------- governance */

function renderGov(tick) {
  const live = detail.protocols.filter((p) => (p.first_tick ?? 1e9) <= tick);
  const adopted = live.filter((p) => p.status === "adopted");

  const ruleCards = live
    .map((p) => {
      const acc = protoPrefix.get(p.id);
      const use = acc.use[tick];
      const viol = acc.violation[tick];
      const enf = acc.enforcement[tick];
      const firing = p.per_tick[String(tick)]?.enforcement > 0;
      const pending = p.status !== "adopted";
      return `
      <div class="card rule ${firing ? "firing" : ""} ${pending ? "pending" : ""}" data-more>
        <div class="card-top">
          <i class="dot"></i>
          <span class="card-title">${esc(p.name)}</span>
          <span class="pill">${pending ? t("wb.proposed") : `t${p.first_tick}`}</span>
        </div>
        <div class="card-sub">${esc(p.proposer)} · ${
          pending
            ? t("wb.neverAdopted")
            : t("wb.usesBlocked", { u: use.toLocaleString(), b: enf.toLocaleString() })
        }</div>
        <div class="more">
          <p class="rule-text">“${esc(p.rule)}”</p>
          ${p.problem ? `<p class="problem"><b>${t("wb.writtenAgainst")}</b> ${esc(p.problem)}</p>` : ""}
          <p class="meta">${t("wb.backedBy", {
            who: esc(p.supporters.join(", ")) || t("wb.nobody"),
            n: viol.toLocaleString(),
          })}</p>
        </div>
      </div>`;
    })
    .join("");

  const recent = detail.proposals.filter((p) => p.tick <= tick).slice(-6).reverse();
  const decisionCards = recent
    .map((p) => {
      const vetoed = p.status === "rejected";
      const veto = p.vetoed_by.join(", ");
      return `
      <div class="card vote ${vetoed ? "no" : p.status === "adopted" ? "yes" : ""} ${
        p.tick === tick ? "just" : ""
      }" data-more>
        <div class="card-top">
          <span class="card-title">${esc(p.title)}</span>
          <span class="pill">${vetoed ? t("wb.vetoed") : p.status.toUpperCase()}</span>
        </div>
        <div class="card-sub">t${p.tick} · ${esc(p.proposer)}${
          vetoed && veto ? ` · ${t("wb.vetoedBy", { who: esc(veto) })}` : ""
        }</div>
        <div class="more">
          ${p.problem ? `<p class="problem"><b>${t("wb.problem")}</b> ${esc(p.problem)}</p>` : ""}
          ${p.solution ? `<p>${esc(p.solution)}</p>` : ""}
        </div>
      </div>`;
    })
    .join("");

  const decided = detail.proposals.filter((p) => p.tick <= tick);
  const vetoCount = decided.filter((p) => p.status === "rejected").length;
  el("govCount").textContent = t("wb.govCount", { r: adopted.length, v: vetoCount });
  el("govList").innerHTML =
    `<div class="sec">${t("wb.liveRules")}</div>${ruleCards || `<div class="empty">${t("wb.noRules")}</div>`}` +
    `<div class="sec">${t("wb.decisions")}</div>${decisionCards || `<div class="empty">${t("wb.noProposals")}</div>`}`;
}

/* ------------------------------------------------------------------ feed */

function renderFeed(tick) {
  const upto = detail.messages.filter((m) => m.tick <= tick);
  const shown = upto.slice(-5);
  el("feedCount").textContent = t("wb.feedCount", { a: upto.length, b: detail.messages.length });
  el("feedList").innerHTML = shown
    .map(
      (m) => `
      <div class="card msg ${m.tick === tick ? "just" : ""}" data-more>
        <div class="card-top">
          <span class="chan ${CHANNEL_CLASS[m.channel] || ""}">#${esc(m.channel)}</span>
          <span class="who">${esc(m.from)}</span>
          <span class="pill">t${m.tick}</span>
        </div>
        <div class="msg-text">${esc(m.text)}</div>
      </div>`
    )
    .reverse()
    .join("");
}

export function renderPanels(tick) {
  if (!detail || tick === lastRendered) return;
  lastRendered = tick;
  renderWork(tick);
  renderGov(tick);
  renderFeed(tick);
}

export function workbenchSummary() {
  return detail;
}
