/**
 * Bilingual narration.
 *
 * Only the narration is translated. Everything that came out of the run —
 * issue titles, protocol rule text, proposal summaries, channel messages — stays
 * verbatim in English, because it is evidence rather than copy.
 */

const STORE_KEY = "lanternforge-lang";

let lang = localStorage.getItem(STORE_KEY) === "zh" ? "zh" : "en";
const listeners = [];

export const getLang = () => lang;

export function setLang(next) {
  if (next === lang) return;
  lang = next;
  localStorage.setItem(STORE_KEY, lang);
  for (const fn of listeners) fn(lang);
}

export const toggleLang = () => setLang(lang === "en" ? "zh" : "en");
export const onLangChange = (fn) => listeners.push(fn);

/** `t("key")` or `t("key", { ...params })` for the interpolated ones. */
export function t(key, params = {}) {
  const entry = DICT[key];
  // Statuses come from the run, so an unseen one falls back to its raw name
  // rather than leaking a dictionary key into the UI.
  if (!entry) return key.split(".").pop().replace(/_/g, " ");
  const value = entry[lang] ?? entry.en;
  return typeof value === "function" ? value(params) : value;
}

const DICT = {
  /* ---------------------------------------------------------- act chrome */
  "ui.back": { en: "◀ back", zh: "◀ 上一幕" },
  "ui.next": { en: "next ▶", zh: "下一幕 ▶" },
  "ui.exit": { en: "exit", zh: "退出" },
  "ui.act": { en: ({ i, n }) => `ACT ${i} / ${n}`, zh: ({ i, n }) => `第 ${i} 幕 / 共 ${n}` },
  "ui.actShort": { en: ({ i }) => `ACT ${i}`, zh: ({ i }) => `第 ${i} 幕` },
  "ui.story": { en: "▶ STORY", zh: "▶ 剧情" },
  "ui.storyStop": { en: "■ EXIT", zh: "■ 退出" },
  "ui.prompt": {
    en: "End of this act — press <b>next ▶</b> when you are ready.",
    zh: "本幕结束 —— 看完后按 <b>下一幕 ▶</b> 继续。",
  },
  "ui.promptLast": {
    en: "That is the whole run. Press <b>exit</b> to explore the timeline yourself.",
    zh: "整个过程就是这样。按 <b>退出</b> 可以自己拖时间轴探索。",
  },
  "ui.promptTitle": { en: "READY WHEN YOU ARE", zh: "等你准备好" },

  "act.1": { en: "THE BRIEF", zh: "任务简报" },
  "act.2": { en: "THE CAST", zh: "团队成员" },
  "act.3": { en: "THE FIRST RULE", zh: "第一条规则" },
  "act.4": { en: "THE MACHINE", zh: "制度运转" },
  "act.5": { en: "THE LEDGER", zh: "最终账本" },

  /* --------------------------------------------------------------- act 1 */
  "a1.lede": {
    en: "Eight agents at a company called <b>LanternForge</b> get 336 simulated hours — fourteen days — to carry a real Python library across one version boundary.",
    zh: "一家叫 <b>LanternForge</b> 的公司，八个 agent，336 个模拟小时（十四天），要把一个真实的 Python 库跨过一个版本边界。",
  },
  "a1.foot": {
    en: ({ n }) =>
      `These twelve issues are what the team can see. They are graded on <b>${n} hidden tests</b> they never get to read.`,
    zh: ({ n }) =>
      `这十二个 issue 是团队能看见的全部。而真正评判他们的，是另外 <b>${n} 个他们永远读不到的隐藏测试</b>。`,
  },

  /* --------------------------------------------------------------- act 2 */
  "a2.head": {
    en: "Eight people, eight different ways to be wrong",
    zh: "八个人，八种不同的犯错方式",
  },
  "a2.note": {
    en: "Each card is the agent's own configuration: what they are good at, and the failure modes the simulation gave them. The fight in the next act is already written here.",
    zh: "每张卡片都来自 agent 自己的配置：他擅长什么，以及模拟给他设定的失败模式。下一幕的那场冲突，其实已经写在这里了。",
  },

  /* --------------------------------------------------------------- act 3 */
  "a3.quiet.title": { en: "HOURS 1–5 · NO RULES YET", zh: "第 1–5 小时 · 还没有任何规则" },
  "a3.quiet.text": {
    en: "Eight people start work on the backlog. The only rule on the books asks them to log experiment results — nothing yet defines what <i>done</i> means, so anyone can call anything finished and nobody has to prove it.",
    zh: "八个人开始处理待办。此时账上唯一一条规则只要求记录实验结果 —— 还没有任何规则定义什么算<i>「做完」</i>，所以谁都可以宣称任何事情完成了，而且不用拿出任何证据。",
  },
  "a3.veto.title": { en: ({ t }) => `HOUR ${t} · THE FIRST ATTEMPT FAILS`, zh: ({ t }) => `第 ${t} 小时 · 第一次尝试失败` },
  "a3.veto.text": {
    en: ({ who, problem, vetoer }) =>
      `<b>${who}</b> — the reliability operator, whose listed weakness is <i>conflict with visionary push</i> — proposes the first rule of the company.<br>` +
      `<b>The problem he names:</b> ${problem}<br>` +
      `<b>${vetoer} vetoes it.</b> No rule is created. Work continues exactly as before.`,
    zh: ({ who, problem, vetoer }) =>
      `<b>${who}</b> —— 可靠性负责人，他的角色卡上写明的弱点是<i>「与愿景派冲突」</i>—— 提出了这家公司的第一条规则。<br>` +
      `<b>他指出的问题：</b>${problem}<br>` +
      `<b>${vetoer} 否决了它。</b>没有任何规则诞生，一切照旧。`,
  },
  "a3.retry.title": { en: ({ t }) => `HOUR ${t} · SOMEONE TRIES AGAIN`, zh: ({ t }) => `第 ${t} 小时 · 有人再试一次` },
  "a3.retry.text": {
    en: ({ who, problem, rule }) =>
      `<b>${who}</b> proposes a narrower rule, aimed at something nobody can argue with: <i>“${problem}”</i><br>The rule itself: <b>${rule}</b>`,
    zh: ({ who, problem, rule }) =>
      `<b>${who}</b> 换了个更窄的切入点重提 —— 这次针对的是谁都无法反驳的事实：<i>「${problem}」</i><br>规则本身：<b>${rule}</b>`,
  },
  "a3.adopt.title": { en: ({ t }) => `HOUR ${t} · IT BECOMES LAW`, zh: ({ t }) => `第 ${t} 小时 · 它成为了法律` },
  "a3.adopt.text": {
    en: ({ who }) =>
      `<b>${who}</b> backs it and the proposal is <b>adopted</b>. From this hour on the rule is not advice — it is wired into the merge path, and it applies to everyone, including the two people who wrote it.`,
    zh: ({ who }) =>
      `<b>${who}</b> 表示支持，提案<b>通过</b>。从这一小时起，这条规则不再是建议 —— 它被接进了合并流程，对所有人生效，包括写下它的那两个人。`,
  },
  "a3.bind.title": { en: ({ t }) => `HOUR ${t} · THE FIRST TIME IT BINDS`, zh: ({ t }) => `第 ${t} 小时 · 规则第一次生效` },
  "a3.bind.text": {
    en: "The rule is consulted for the first time. Nothing dramatic happens yet — it just starts sitting in the path of every piece of work that claims to be done.",
    zh: "规则第一次被调用。此刻还没有任何戏剧性的事发生 —— 它只是开始横在每一件「声称已完成」的工作前面。",
  },
  "a3.block.title": { en: ({ t }) => `HOUR ${t} · THE RULE BITES`, zh: ({ t }) => `第 ${t} 小时 · 规则咬人了` },
  "a3.block.text": {
    en: ({ who, pr }) =>
      `<b>${who}</b> submits <code>${pr}</code> without the required evidence. The <b>organizational gate blocks the merge</b>. ` +
      `This is the moment the company stops being eight people with opinions and starts being an institution: a rule just overruled a person.`,
    zh: ({ who, pr }) =>
      `<b>${who}</b> 提交了 <code>${pr}</code>，但没有附上规则要求的证据。<b>组织闸门直接拦下了这次合并。</b>` +
      `这一刻，这家公司不再只是八个有意见的人，而开始成为一个制度：规则压过了人。`,
  },
  "a3.amend.title": { en: ({ t }) => `HOUR ${t} · AND THEN IT CHANGES`, zh: ({ t }) => `第 ${t} 小时 · 然后它被修订` },
  "a3.amend.text": {
    en: ({ who }) =>
      `<b>${who}</b> amends the rule rather than abandoning it. This is the last piece of the cycle — the rule is now something the team maintains.`,
    zh: ({ who }) =>
      `<b>${who}</b> 选择修订这条规则，而不是废弃它。这是闭环的最后一块 —— 规则从此成了团队需要持续维护的东西。`,
  },
  "a3.done.title": { en: ({ t }) => `HOUR ${t} · ONE FULL CYCLE, DONE`, zh: ({ t }) => `第 ${t} 小时 · 一个完整闭环` },
  "a3.done.text": {
    en: "<b>propose → veto → re-propose → adopt → bind → violate → enforce → amend.</b> The system now starts measuring whether the rule actually reduces violations. Everything after this hour is this same loop, over and over.",
    zh: "<b>提出 → 否决 → 重提 → 通过 → 生效 → 违规 → 强制 → 修订。</b>系统开始测量这条规则到底有没有减少违规。这一小时之后的全部内容，都是同一个循环的重复。",
  },

  /* --------------------------------------------------------------- act 4 */
  "a4.intro.title": { en: "THE SAME LOOP, ×336", zh: "同一个循环，重复 336 小时" },
  "a4.intro.text": {
    en: "Now at speed. Every flash on the timeline is that same cycle running again: a proposal, a vote, a rule, a violation, a blocked merge. Watch the blocked counter climb.",
    zh: "现在加速。时间轴上每一次闪烁都是同一个循环在重跑：提案、投票、规则、违规、拦截。注意看拦截计数一路往上爬。",
  },
  "a4.resume": { en: "Back to speed.", zh: "继续加速。" },
  "a4.lands.title": { en: ({ t }) => `HOUR ${t} · ANOTHER RULE LANDS`, zh: ({ t }) => `第 ${t} 小时 · 又一条规则落地` },
  "a4.lands.text": {
    en: ({ who, name, n }) =>
      `<b>${who}</b> gets <i>“${name}”</i> adopted. It will go on to block ${n} more merges.`,
    zh: ({ who, name, n }) =>
      `<b>${who}</b> 让 <i>「${name}」</i> 获得通过。这条规则之后还会再拦下 ${n} 次合并。`,
  },
  "a4.fails.title": { en: ({ t }) => `HOUR ${t} · A RULE THAT NEVER LANDS`, zh: ({ t }) => `第 ${t} 小时 · 一条始终没落地的规则` },
  "a4.fails.text": {
    en: ({ who, rule }) =>
      `<b>${who}</b> proposes <i>“${rule}”</i> — and it is never adopted. Not every attempt becomes an institution.`,
    zh: ({ who, rule }) =>
      `<b>${who}</b> 提出了 <i>「${rule}」</i> —— 但它始终没有被采纳。不是每一次尝试都能变成制度。`,
  },

  /* --------------------------------------------------------------- act 5 */
  "a5.kicker": { en: "THE LEDGER · HOUR 336", zh: "最终账本 · 第 336 小时" },
  "a5.shipped": { en: "What shipped", zh: "交付了什么" },
  "a5.shippedFoot": { en: ({ a, b }) => `${a} of ${b} closed.`, zh: ({ a, b }) => `${b} 个里关掉了 ${a} 个。` },
  "a5.rules": { en: "What the team institutionalised", zh: "沉淀成了哪些制度" },
  "a5.rulesFoot": {
    en: ({ a, b }) => `${a} adopted, ${b} proposals vetoed along the way.`,
    zh: ({ a, b }) => `${a} 条规则被采纳，过程中还有 ${b} 个提案被否决。`,
  },
  "a5.uses": { en: ({ n }) => `${n} uses`, zh: ({ n }) => `调用 ${n} 次` },
  "a5.blockedBy": { en: ({ n }) => `${n} merges blocked`, zh: ({ n }) => `拦下 ${n} 次合并` },
  "a5.blockedTotal": { en: "merges blocked by the team's own rules", zh: "次合并被团队自己订的规则拦下" },
  "a5.passed": { en: "hidden contracts passed", zh: "个隐藏契约通过" },
  "a5.close": {
    en: ({ names }) =>
      `The two the evaluator accepted: ${names}. Shipping ten issues and passing two hidden tests are very different numbers — which is exactly what the run is built to measure.`,
    zh: ({ names }) =>
      `评估器最终认可的两个是：${names}。交付十个 issue，和通过两个隐藏测试，是完全不同的两个数字 —— 而这正是这次 run 要测量的东西。`,
  },

  /* ---------------------- notes that rise off the room where the work landed */
  "fx.merged": { en: "merged", zh: "合入主干" },
  "fx.shipped": { en: "shipped", zh: "发版" },
  "fx.proposed": { en: "rule proposed", zh: "提出规则" },
  "fx.amended": { en: "rule amended", zh: "修订规则" },
  "fx.vote": { en: "vote", zh: "投票" },
  "fx.ci": { en: "CI", zh: "CI" },

  /* ------------------------------------------------------------ workbench */
  "wb.work": { en: "THE WORK", zh: "工作内容" },
  "wb.gov": { en: "GOVERNANCE", zh: "治理" },
  "wb.channels": { en: "CHANNELS", zh: "频道" },
  "wb.liveRules": { en: "LIVE RULES", zh: "生效中的规则" },
  "wb.decisions": { en: "DECISIONS", zh: "决策记录" },
  "wb.noRules": { en: "no rules yet", zh: "还没有任何规则" },
  "wb.noProposals": { en: "nothing proposed yet", zh: "还没有任何提案" },
  "wb.notStarted": { en: "not started", zh: "未开始" },
  "wb.noDetail": { en: "No description recorded.", zh: "没有记录描述。" },
  "wb.workCount": {
    en: ({ m, a, o }) => `${m} merged · ${a} active · ${o} open`,
    zh: ({ m, a, o }) => `已合入 ${m} · 进行中 ${a} · 未开始 ${o}`,
  },
  "wb.govCount": {
    en: ({ r, v }) => `${r} rules live · ${v} vetoed`,
    zh: ({ r, v }) => `${r} 条规则生效 · ${v} 个提案被否`,
  },
  "wb.feedCount": {
    en: ({ a, b }) => `${a} / ${b} messages`,
    zh: ({ a, b }) => `${a} / ${b} 条消息`,
  },
  "wb.proposed": { en: "proposed", zh: "已提出" },
  "wb.neverAdopted": { en: "never adopted", zh: "从未通过" },
  "wb.usesBlocked": {
    en: ({ u, b }) => `${u} uses · <b class="blocked">${b} blocked</b>`,
    zh: ({ u, b }) => `调用 ${u} 次 · <b class="blocked">拦下 ${b} 次</b>`,
  },
  "wb.backedBy": {
    en: ({ who, n }) => `backed by ${who} · ${n} violations caught`,
    zh: ({ who, n }) => `支持者：${who} · 捕获 ${n} 次违规`,
  },
  "wb.nobody": { en: "nobody else", zh: "无其他人" },
  "wb.writtenAgainst": { en: "Written against:", zh: "针对的问题：" },
  "wb.problem": { en: "Problem:", zh: "问题：" },
  "wb.vetoedBy": { en: ({ who }) => `blocked by <b>${who}</b>`, zh: ({ who }) => `被 <b>${who}</b> 否决` },
  "wb.vetoed": { en: "VETOED", zh: "已否决" },
  "wb.status.open": { en: "open", zh: "未开始" },
  "wb.status.in_progress": { en: "in progress", zh: "进行中" },
  "wb.status.implementation_done": { en: "built", zh: "已实现" },
  "wb.status.done": { en: "done", zh: "已完成" },
  "wb.status.merged": { en: "merged", zh: "已合入" },
};
