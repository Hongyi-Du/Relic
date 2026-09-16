// Shared maps + ordering, ported verbatim from the vanilla inspector so colors
// and sub-view ordering stay identical.

export const SUBS_INTERNAL = [
  "dashboard", "product", "episodes", "reflections", "llm", "proposals", "agents",
  "tasks", "docs", "messages", "meetings", "repo", "sandbox", "budget", "protocols", "event graph",
] as const;

export const SUBS_EXTERNAL = [
  "dashboard", "profiles", "posts", "signals", "network", "offers", "flow",
] as const;

export const NODE_COLORS: Record<string, string> = {
  agent: "#58a6ff", trait: "#bc8cff", skill: "#3fb950", failure_mode: "#f85149", routine: "#d29922",
  communication_style: "#79c0ff", state: "#ff9bce", policy_bias: "#ffa657", action_feature: "#56d4dd",
  task: "#3fb950", doc: "#79c0ff", file: "#56d4dd", message: "#d2a8ff", meeting: "#ffa657", pr: "#f0883e",
  commit: "#8b949e", result: "#3fb950", protocol: "#f778ba", commitment: "#a5d6ff", dispute: "#f85149",
  request: "#d29922", external_profile: "#ff7b72", external_post: "#7ee787", internal_agent: "#58a6ff",
  sandbox_job: "#8b949e", experiment: "#56d4dd", cost_event: "#d29922", action: "#3fb950",
  feature: "#56d4dd", speech: "#d2a8ff", risk: "#f85149", evidence: "#7ee787", cluster: "#ffa657",
};

export const EP_COLORS: Record<string, string> = {
  feedback_ingestion_episode: "#58a6ff", claim_dispute_episode: "#f85149", experiment_episode: "#3fb950",
  protocol_formation_episode: "#d2a8ff", launch_crunch_episode: "#ffa657", customer_triage_episode: "#79c0ff",
};

export const WISH_COLORS: Record<string, string> = {
  tool_need: "#3fb950", protocol_need: "#d2a8ff", workflow_need: "#ffa657", artifact_need: "#56d4dd",
  role_clarity_need: "#79c0ff", resource_need: "#f85149", coordination_need: "#58a6ff", information_need: "#8b949e",
};

export const EXT_LANG: Record<string, string> = {
  py: "python", pyw: "python", js: "javascript", mjs: "javascript", ts: "typescript", tsx: "typescript",
  jsx: "javascript", md: "markdown", markdown: "markdown", json: "json", yaml: "yaml", yml: "yaml",
  toml: "ini", ini: "ini", html: "xml", xml: "xml", css: "css", scss: "scss", sh: "bash", bash: "bash",
  zsh: "bash", sql: "sql", java: "java", kt: "kotlin", go: "go", rs: "rust", c: "c", h: "c", cpp: "cpp",
  cc: "cpp", hpp: "cpp", rb: "ruby", php: "php", cs: "csharp", swift: "swift", txt: "plaintext", log: "plaintext",
};

// node "type prefix" routing for graph clicks (id like "task_...", "trait:...")
export const OBJECT_PREFIX = /^(task_|doc_|msg_|pr_|result|proto_|meeting_|commitment_|dispute_|request_|post_|ext_)/;
export const PERSONA_PREFIX = /^(trait:|skill:|state:|style:|fail:|feat:|act:|speech:|risk:|ev:|hub:)/;

/** map a status-ish string to a tag tone (good / warn / bad / "") */
export function statusTone(s: unknown): "good" | "warn" | "bad" | "" {
  const v = String(s ?? "").toLowerCase();
  if (!v) return "";
  if (/(done|merged|approved|adopted|success|active|paid|resolved|accepted|ready|passed|green|ok|complete)/.test(v)) return "good";
  if (/(fail|reject|blocked|critical|dispute|violation|red|error|overdue|unpaid|regress|missing|at_risk|breach)/.test(v)) return "bad";
  if (/(pending|review|draft|open|awaiting|proposed|queued|in_progress|wip|partial|yellow|stalled|warn)/.test(v)) return "warn";
  return "";
}
