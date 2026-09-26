/**
 * The run told as five acts, in order:
 *
 *   1 THE BRIEF      what this company was asked to do
 *   2 THE CAST       who the eight of them are
 *   3 THE FIRST RULE one complete governance cycle, slowly
 *   4 THE MACHINE    the same cycle repeating, fast
 *   5 THE LEDGER     what got shipped and what rules survived
 *
 * Acts 1, 2 and 5 take over the screen. Acts 3 and 4 hand the screen back to
 * the office and drive the existing workbench panels instead.
 *
 * Acts never auto-advance: each one ends on a prompt and waits for the reader.
 */

import { drawPerson } from "./sprites.js";
import { t, getLang, onLangChange } from "./i18n.js";

const el = (id) => document.getElementById(id);
const esc = (s) =>
  String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

let ctx = null; // { detail, setTick, maxTick, exitCompare }
let token = 0;
let running = false;
let actIdx = 0;
let waiting = false; // act finished, holding for the reader

export const actsRunning = () => running;

export function initActs(hooks) {
  ctx = hooks;
  el("actPrev").addEventListener("click", () => jump(-1));
  el("actNext").addEventListener("click", () => jump(1));
  el("actSkip").addEventListener("click", stopActs);
  el("capPrev").addEventListener("click", () => jump(-1));
  el("capNext").addEventListener("click", () => jump(1));
  onLangChange(() => {
    relabel();
    if (!running) return;
    // A finished act only needs its text swapped; one still playing has to be
    // replayed, since its narration is emitted beat by beat.
    if (waiting) redraw();
    else play(actIdx);
  });
  relabel();
}

/** Button and act-name text that lives outside any single act. */
function relabel() {
  el("actPrev").textContent = t("ui.back");
  el("actNext").textContent = t("ui.next");
  el("actSkip").textContent = t("ui.exit");
  el("story").textContent = running ? t("ui.storyStop") : t("ui.story");
  if (running) {
    el("actChap").textContent = t("ui.act", { i: actIdx + 1, n: ACTS.length });
    el("actName").textContent = t(`act.${actIdx + 1}`);
  }
}

/* ------------------------------------------------------------- act shells */

function setAct(i) {
  actIdx = i;
  const act = ACTS[i];
  document.body.className = `acts ${act.full ? "act-full" : "act-live"}`;
  el("actChap").textContent = t("ui.act", { i: i + 1, n: ACTS.length });
  el("actName").textContent = t(`act.${i + 1}`);
  el("acts").hidden = !act.full;
}

function beat(title, text) {
  el("caption").hidden = false;
  el("capChap").textContent = t("ui.actShort", { i: actIdx + 1 });
  el("capTitle").textContent = title;
  el("capText").innerHTML = text;
}

/** Every act ends here rather than rolling into the next one on a timer. */
function prompt() {
  waiting = true;
  const isLast = actIdx === ACTS.length - 1;
  const key = isLast ? "ui.promptLast" : "ui.prompt";
  if (ACTS[actIdx].full) {
    el("acts").insertAdjacentHTML("beforeend", `<div class="act-prompt">${t(key)}</div>`);
  } else {
    beat(t("ui.promptTitle"), t(key));
  }
  el("actNext").classList.toggle("pulse", !isLast);
  el("actSkip").classList.toggle("pulse", isLast);
}

/** Re-render a finished act after a language switch, skipping its animation. */
async function redraw() {
  if (!ACTS[actIdx].full) {
    prompt(); // the office acts only have a caption to swap
    return;
  }
  const mine = ++token;
  await ACTS[actIdx].run(mine);
  if (token === mine) prompt();
}

/** Ring the card the narration is talking about. */
function spotlight(needle) {
  for (const node of document.querySelectorAll(".spot")) node.classList.remove("spot");
  if (!needle) return;
  const cards = [...document.querySelectorAll("#govList .card, #workList .card")];
  const hit = cards.find((c) => c.textContent.includes(needle));
  if (!hit) return;
  hit.classList.add("spot", "open");
  // Scroll the panel, not the document — scrollIntoView would shove the whole
  // cabinet off the top of the window.
  const pane = hit.closest(".panel-body");
  if (pane) {
    pane.scrollTop = hit.offsetTop - pane.offsetTop - (pane.clientHeight - hit.offsetHeight) / 2;
  }
}

/**
 * Every act is handed the id of the run that started it and bails the moment a
 * newer run takes over, so skipping between acts cannot leave two playing.
 */
const live = (id) => running && id === token;

/** Step the clock one hour at a time so the office animation reads. */
async function walk(id, from, to, step) {
  for (let tick = from; tick <= to; tick++) {
    if (!live(id)) return;
    ctx.setTick(tick);
    await sleep(step);
  }
}

/**
 * Sweep the clock at a rate in hours/second.
 *
 * The position is derived from elapsed wall-clock time rather than from a tick
 * count, and the wait races a frame against a timer. A visible tab advances
 * smoothly on frames; a hidden one gets no frames and clamps short timers, but
 * still lands on the right hour after the right number of seconds.
 */
async function ramp(id, from, to, hoursPerSecond) {
  const t0 = performance.now();
  let pos = from;
  ctx.setTick(Math.round(pos), false);
  while (pos < to) {
    if (!live(id)) return;
    await Promise.race([new Promise(requestAnimationFrame), sleep(60)]);
    pos = Math.min(to, from + ((performance.now() - t0) / 1000) * hoursPerSecond);
    ctx.setTick(Math.round(pos), false);
  }
}

/* ---------------------------------------------------------------- act one */

async function runBrief() {
  const { product } = ctx.detail;
  el("caption").hidden = true;
  el("acts").innerHTML = `
    <div class="act-brief">
      <div class="act-kicker">${t("act.1")}</div>
      <h2>${esc(product.label)}</h2>
      <p class="act-lede">${t("a1.lede")}</p>
      <div class="act-grid">
        ${product.gaps
          .map(
            (g, i) => `<div class="brief-item" style="--i:${i}">
              <i>${String(i + 1).padStart(2, "0")}</i><span>${esc(g)}</span>
            </div>`
          )
          .join("")}
      </div>
      <p class="act-foot">${t("a1.foot", { n: product.hidden_tests })}</p>
    </div>`;
  await sleep(product.gaps.length * 160 + 600); // let the list finish arriving
}

/* ---------------------------------------------------------------- act two */

async function runCast() {
  el("caption").hidden = true;
  el("acts").innerHTML = `
    <div class="act-cast">
      <div class="act-kicker">${t("act.2")}</div>
      <h2>${t("a2.head")}</h2>
      <p class="act-lede">${t("a2.note")}</p>
      <div class="cast-grid">
        ${ctx.detail.cast
          .map(
            (c, i) => `
          <div class="cast-card" style="--i:${i}">
            <canvas class="portrait" width="60" height="76" data-agent="${esc(c.id)}"></canvas>
            <div class="cast-body">
              <div class="cast-name">${esc(c.name)} <span>${esc(c.role)}</span></div>
              <div class="cast-title">${esc(c.title)}</div>
              <div class="cast-traits">${esc(c.traits)}</div>
              <div class="chips">
                ${c.skills.map((s) => `<b class="chip good">${esc(s.name)} ${s.score}</b>`).join("")}
              </div>
              <div class="chips">
                ${c.flaws.map((f) => `<b class="chip bad">⚠ ${esc(f)}</b>`).join("")}
              </div>
            </div>
          </div>`
          )
          .join("")}
      </div>
    </div>`;
  for (const cv of document.querySelectorAll(".portrait")) {
    const c = cv.getContext("2d");
    c.imageSmoothingEnabled = false;
    c.save();
    c.translate(30, 70);
    c.scale(0.86, 0.86);
    drawPerson(c, cv.dataset.agent, 0, 0, { facing: 1, walkPhase: 0, moving: false });
    c.restore();
  }
  await sleep(ctx.detail.cast.length * 400 + 600);
}

/* -------------------------------------------------------------- act three */

async function runFirstRule(id) {
  const fc = ctx.detail.first_cycle;
  const proto = ctx.detail.protocols.find((p) => p.id === fc.protocol_id);
  const veto = fc.veto;

  ctx.setTick(1);
  beat(t("a3.quiet.title"), t("a3.quiet.text"));
  await walk(id, 1, 5, 900);
  if (!live(id)) return;

  ctx.setTick(veto.tick);
  spotlight(veto.title.slice(0, 40));
  beat(
    t("a3.veto.title", { t: veto.tick }),
    t("a3.veto.text", {
      who: esc(veto.proposer),
      problem: esc(veto.problem),
      vetoer: esc(veto.vetoed_by.join(", ")),
    })
  );
  await sleep(11000);
  if (!live(id)) return;

  await walk(id, veto.tick + 1, fc.proposal_tick - 1, 260);
  if (!live(id)) return;

  ctx.setTick(fc.proposal_tick);
  spotlight(proto.name.slice(0, 30));
  beat(
    t("a3.retry.title", { t: fc.proposal_tick }),
    t("a3.retry.text", {
      who: esc(fc.proposer),
      problem: esc(proto.problem),
      rule: esc(proto.rule.slice(0, 220)),
    })
  );
  await sleep(12000);
  if (!live(id)) return;

  await walk(id, fc.proposal_tick + 1, fc.adoption.tick, 700);
  if (!live(id)) return;
  beat(t("a3.adopt.title", { t: fc.adoption.tick }), t("a3.adopt.text", { who: esc(fc.support.actor) }));
  await sleep(8000);
  if (!live(id)) return;

  await walk(id, fc.adoption.tick + 1, fc.first_use_tick, 800);
  if (!live(id)) return;
  beat(t("a3.bind.title", { t: fc.first_use_tick }), t("a3.bind.text"));
  await sleep(6500);
  if (!live(id)) return;

  const blk = fc.first_block;
  await walk(id, fc.first_use_tick + 1, blk.tick - 1, 700);
  if (!live(id)) return;

  ctx.setTick(blk.tick);
  document.body.classList.add("alarm");
  beat(
    t("a3.block.title", { t: blk.tick }),
    t("a3.block.text", { who: esc(veto.proposer), pr: esc(blk.context || "a PR") })
  );
  await sleep(11000);
  document.body.classList.remove("alarm");
  if (!live(id)) return;

  await walk(id, blk.tick + 1, fc.amendment.tick, 700);
  if (!live(id)) return;
  beat(t("a3.amend.title", { t: fc.amendment.tick }), t("a3.amend.text", { who: esc(fc.amendment.actor) }));
  await sleep(8000);
  if (!live(id)) return;

  if (fc.impact) {
    await walk(id, fc.amendment.tick + 1, fc.impact.tick, 600);
    if (!live(id)) return;
    beat(t("a3.done.title", { t: fc.impact.tick }), t("a3.done.text"));
    await sleep(9500);
  }
}

/* --------------------------------------------------------------- act four */

async function runMachine(id) {
  const d = ctx.detail;
  const fc = d.first_cycle;
  const start = (fc.impact?.tick || fc.amendment.tick) + 1;
  const later = d.protocols.filter((p) => (p.first_tick ?? 0) > start);
  const RATE = 45; // hours per second

  beat(t("a4.intro.title"), t("a4.intro.text"));

  let cursor = start;
  for (const p of later) {
    await ramp(id, cursor, p.first_tick, RATE);
    if (!live(id)) return;
    spotlight(p.name.slice(0, 30));
    const adopted = p.status === "adopted";
    beat(
      t(adopted ? "a4.lands.title" : "a4.fails.title", { t: p.first_tick }),
      adopted
        ? t("a4.lands.text", {
            who: esc(p.proposer),
            name: esc(p.name),
            n: (p.totals.enforcement || 0).toLocaleString(),
          })
        : t("a4.fails.text", { who: esc(p.proposer), rule: esc(p.rule.slice(0, 120)) })
    );
    await sleep(6500);
    if (!live(id)) return;
    spotlight(null);
    beat(t("a4.intro.title"), t("a4.resume"));
    cursor = p.first_tick + 1;
  }
  await ramp(id, cursor, ctx.maxTick, RATE);
}

/* --------------------------------------------------------------- act five */

async function runLedger() {
  const d = ctx.detail;
  const oss = d.tasks.filter((x) => x.oss);
  const shipped = oss.filter((x) => ["merged", "done"].includes(x.final_status));
  const adopted = d.protocols.filter((p) => p.status === "adopted");
  const blocked = d.protocols.reduce((n, p) => n + (p.totals.enforcement || 0), 0);
  const vetoed = d.proposals.filter((p) => p.status === "rejected").length;
  const ev = d.evaluation;

  // Land the clock on the last hour, so exiting here drops you at the end.
  ctx.setTick(ctx.maxTick, false);
  el("caption").hidden = true;
  el("acts").innerHTML = `
    <div class="act-ledger">
      <div class="act-kicker">${t("a5.kicker")}</div>
      <div class="ledger-cols">
        <div>
          <h3>${t("a5.shipped")}</h3>
          <div class="ledger-list">
            ${oss
              .map(
                (x) => `<div class="ledger-row ${
                  ["merged", "done"].includes(x.final_status) ? "ok" : "no"
                }"><i></i><span>${esc(x.title)}</span></div>`
              )
              .join("")}
          </div>
          <p class="act-foot">${t("a5.shippedFoot", { a: shipped.length, b: oss.length })}</p>
        </div>
        <div>
          <h3>${t("a5.rules")}</h3>
          <div class="ledger-list">
            ${adopted
              .map(
                (p) => `<div class="ledger-rule">
                  <div class="lr-top"><b>${esc(p.name)}</b><span>t${p.first_tick} · ${esc(p.proposer)}</span></div>
                  <div class="lr-rule">“${esc(p.rule.slice(0, 200))}”</div>
                  <div class="lr-stat">${t("a5.uses", { n: (p.totals.use || 0).toLocaleString() })} ·
                    <b>${t("a5.blockedBy", { n: (p.totals.enforcement || 0).toLocaleString() })}</b></div>
                </div>`
              )
              .join("")}
          </div>
          <p class="act-foot">${t("a5.rulesFoot", { a: adopted.length, b: vetoed })}</p>
        </div>
      </div>
      <div class="ledger-verdict">
        <div><b>${blocked.toLocaleString()}</b><span>${t("a5.blockedTotal")}</span></div>
        <div class="win"><b>${ev.passed.length} / ${d.product.hidden_tests}</b><span>${t("a5.passed")}</span></div>
      </div>
      <p class="act-lede">${t("a5.close", {
        names: ev.passed.map((p) => `<code>${esc(p)}</code>`).join(", "),
      })}</p>
    </div>`;
  await sleep(600);
}

/* ------------------------------------------------------------------ drive */

const ACTS = [
  { full: true, run: runBrief },
  { full: true, run: runCast },
  { full: false, run: runFirstRule },
  { full: false, run: runMachine },
  { full: true, run: runLedger },
];

async function play(i) {
  const mine = ++token;
  running = true;
  waiting = false;
  setAct(i);
  el("actNext").classList.remove("pulse");
  el("actSkip").classList.remove("pulse");
  spotlight(null);
  await ACTS[i].run(mine);
  if (token !== mine) return;
  prompt();
}

function jump(delta) {
  const next = actIdx + delta;
  if (next < 0 || next >= ACTS.length) return;
  play(next);
}

export function startActs() {
  // The narrative is about this one company, so the split screen steps aside.
  ctx.exitCompare();
  el("actBar").hidden = false;
  running = true;
  relabel();
  el("story").classList.add("on");
  ctx.setTick(1, false);
  play(0);
}

export function stopActs() {
  token++;
  running = false;
  waiting = false;
  document.body.className = "";
  el("acts").hidden = true;
  el("actBar").hidden = true;
  el("caption").hidden = true;
  el("story").classList.remove("on");
  el("actNext").classList.remove("pulse");
  el("actSkip").classList.remove("pulse");
  relabel();
  spotlight(null);
}
