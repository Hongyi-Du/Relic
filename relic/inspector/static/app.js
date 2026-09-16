"use strict";

const state = {
  trace: null,
  health: null,
  frameIndex: 0,
  panel: "overview",
  selectedId: null,
  selectedFrameIndex: null,
  playing: false,
  playTimer: null,
  refreshTimer: null,
};

const panelCopy = {
  overview: ["Overview", "A public snapshot of the organization at the selected tick."],
  members: ["Members", "Roles, active work, recent decisions, and relevant protocols."],
  tasks: ["Tasks", "Ownership, progress, dependencies, blockers, and evidence."],
  timeline: ["Timeline", "Public organization events up to the selected tick."],
  episodes: ["Episodes", "Bounded stretches of work, friction, decisions, and outcomes."],
  reflections: ["Reflections", "Public lineage from private reflection outcomes to organization wishes."],
  proposals: ["Proposals", "Suggested institutional changes and their source lineage."],
  governance: ["Governance", "Review participation, approvals, and adoption latency."],
  protocols: ["Protocols", "Adopted rules, use, enforcement, amendment, and retirement."],
  artifacts: ["Artifacts", "Public documents, files, and output objects emitted by this trace."],
  repo: ["Repo / PR / CI", "Public repository state associated with this organization frame."],
  evidence: ["Evaluation", "Tests, evaluator outputs, and public evidence annotations."],
  decisions: ["Decisions", "Structured, profile-conditioned choices made at this tick."],
};

const idKeys = [
  ["organization_id", "organization"],
  ["agent_id", "member"],
  ["task_id", "task"],
  ["event_id", "event"],
  ["episode_id", "episode"],
  ["proposal_id", "proposal"],
  ["protocol_id", "protocol"],
  ["artifact_id", "artifact"],
  ["evidence_id", "evidence"],
  ["branch_id", "branch"],
  ["commit_id", "commit"],
  ["pr_id", "pull request"],
  ["review_id", "review"],
  ["ci_id", "ci"],
  ["result_id", "result"],
  ["reflection_id", "reflection"],
  ["wish_id", "wish"],
];

const $ = (id) => document.getElementById(id);

function element(tag, className = "", text = null) {
  const item = document.createElement(tag);
  if (className) item.className = className;
  if (text !== null) item.textContent = String(text);
  return item;
}

function append(parent, ...children) {
  children.flat().filter(Boolean).forEach((child) => parent.append(child));
  return parent;
}

function asArray(value) {
  if (Array.isArray(value)) return value;
  if (value && typeof value === "object") return Object.values(value);
  return [];
}

function hasOwn(value, key) {
  return Boolean(value && Object.prototype.hasOwnProperty.call(value, key));
}

function textValue(value, fallback = "—") {
  if (value === null || value === undefined || value === "") return fallback;
  if (Array.isArray(value)) {
    if (!value.length) return fallback;
    return value.some((item) => item && typeof item === "object")
      ? JSON.stringify(value)
      : value.join(", ");
  }
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

function titleCase(value) {
  return textValue(value).replaceAll("_", " ").replace(/\b\w/g, (char) => char.toUpperCase());
}

function statusClass(value) {
  const status = String(value || "").toLowerCase();
  if (["done", "completed", "adopted", "active", "passed", "available"].includes(status)) return "good";
  if (["blocked", "failed", "rejected", "retired", "violated"].includes(status)) return "bad";
  return "warn";
}

function statusPill(value) {
  return element("span", `status-pill ${statusClass(value)}`, titleCase(value));
}

function objectIdentity(value) {
  if (!value || typeof value !== "object") return null;
  for (const [key, kind] of idKeys) {
    if (typeof value[key] === "string" && value[key]) return { id: value[key], kind };
  }
  return null;
}

function objectLink(id, label = null, frameIndex = null) {
  if (!id) return element("span", "muted", "—");
  const button = element("button", "object-link", label || id);
  button.type = "button";
  button.addEventListener("click", (event) => {
    event.stopPropagation();
    selectObject(id, frameIndex);
  });
  return button;
}

function objectLinks(ids, fallback = "None", frameIndex = null) {
  const values = [...new Set(asArray(ids).filter(Boolean))];
  if (!values.length) return element("span", "muted", fallback);
  const links = element("span");
  values.forEach((id, index) => {
    if (index) links.append(document.createTextNode(", "));
    links.append(objectLink(id, null, frameIndex));
  });
  return links;
}

function tag(value, className = "") {
  return element("span", `tag ${className}`.trim(), textValue(value));
}

function currentFrame() {
  return state.trace?.frames?.[state.frameIndex] || null;
}

function organization(frame = currentFrame()) {
  return frame?.organization || {};
}

function allEvents() {
  if (!state.trace) return [];
  return state.trace.frames.flatMap((frame, frameIndex) =>
    asArray(frame.events).map((event) => ({ event, frameIndex })),
  );
}

function allGovernanceEvents() {
  if (!state.trace) return [];
  return state.trace.frames.flatMap((frame, frameIndex) =>
    asArray(frame.governance_events).map((event) => ({ event, frameIndex })),
  );
}

function panelHeader(key) {
  const [title, copy] = panelCopy[key];
  const header = element("header", "panel-header");
  append(header, element("h1", "", title), element("p", "", copy));
  return header;
}

function sectionTitle(text) {
  return element("h2", "section-title", text);
}

function emptyState(text) {
  return element("div", "empty-state", text);
}

function metric(label, value) {
  const card = element("div", "metric");
  append(card, element("strong", "", value), element("span", "", label));
  return card;
}

function keyValueCard(title, identity, rows, labels = []) {
  const card = element("article", "card");
  const heading = element("div", "card-heading");
  append(heading, identity ? objectLink(identity.id, title) : element("strong", "", title));
  if (identity) append(heading, element("span", "kind-pill", identity.kind));
  card.append(heading);
  const details = element("dl");
  rows.forEach(([label, value]) => {
    details.append(element("dt", "", label));
    if (value instanceof Node) {
      const dd = element("dd");
      dd.append(value);
      details.append(dd);
    } else {
      details.append(element("dd", "", textValue(value)));
    }
  });
  card.append(details);
  if (labels.length) {
    const tags = element("div", "tag-row");
    labels.forEach((label) => tags.append(tag(label)));
    card.append(tags);
  }
  return card;
}

function renderOverview(container) {
  const frame = currentFrame();
  const org = organization(frame);
  const agents = asArray(org.agents);
  const tasks = asArray(org.tasks);
  const protocols = asArray(org.protocols);
  const episodes = asArray(frame.episodes);
  const activeTasks = tasks.filter((item) => !["done", "merged", "released", "abandoned"].includes(item.status));
  const openEpisodes = episodes.filter((item) => item.status === "open");
  const activeProtocols = protocols.filter((item) => item.status === "active" || item.adoption_status === "adopted");

  const metrics = element("div", "metric-grid");
  append(
    metrics,
    metric("Active members", agents.filter((item) => item.status !== "inactive").length),
    metric("Active tasks", activeTasks.length),
    metric("Active protocols", activeProtocols.length),
    metric("Open episodes", openEpisodes.length),
  );
  container.append(metrics);

  const privacy = state.trace.privacy || {};
  const notice = element("div", "notice");
  const privacyText = [
    privacy.private_reflections_included ? "private reflections present" : "private reflections excluded",
    privacy.private_memories_included ? "private memories present" : "private memories excluded",
    privacy.provider_messages_included ? "provider messages present" : "provider messages excluded",
  ].join(" · ");
  append(notice, element("strong", "", "Public trace boundary. "), document.createTextNode(privacyText));
  const causalNotice = element("div", "notice");
  append(
    causalNotice,
    element("strong", "", "Descriptive evidence only. "),
    document.createTextNode(
      "Observed lineage and state differences record sequence and provenance; they do not establish causal attribution to a protocol, member, or mechanism.",
    ),
  );
  causalNotice.append(
    element(
      "p",
      "",
      "HCI-facing use is an interface demonstration or formative artifact unless separately supported by reviewed participant-study evidence; it is not a powered participant evaluation.",
    ),
  );
  container.append(sectionTitle("Trace contract"), notice, causalNotice);

  const events = allEvents().filter((item) => item.frameIndex <= state.frameIndex).slice(-6).reverse();
  container.append(sectionTitle("Recent public events"));
  if (!events.length) container.append(emptyState("No public events have been emitted yet."));
  else container.append(renderEventList(events));
}

function renderSnapshotSummary(container) {
  const frame = currentFrame();
  const org = organization(frame);
  const tasks = asArray(org.tasks);
  const protocols = asArray(org.protocols);
  const summary = element("div", "metric-grid snapshot-summary");
  append(
    summary,
    metric("Snapshot tick", frame.tick),
    metric("Members", asArray(org.agents).length),
    metric(
      "Open tasks",
      tasks.filter((item) => !["done", "merged", "released", "abandoned"].includes(item.status)).length,
    ),
    metric(
      "Active protocols",
      protocols.filter((item) => item.status === "active" || item.adoption_status === "adopted").length,
    ),
  );
  container.append(summary);
}

function renderMembers(container) {
  const frame = currentFrame();
  const org = organization(frame);
  const protocols = asArray(org.protocols);
  const decisions = asArray(frame.decisions);
  const grid = element("div", "card-grid");
  asArray(org.agents).forEach((agent) => {
    const recent = decisions.find((decision) => decision.agent_id === agent.agent_id);
    const relevant = protocols.filter((protocol) =>
      JSON.stringify(protocol).includes(agent.agent_id),
    );
    const taskLinks = element("span");
    asArray(agent.active_task_ids).forEach((id, index) => {
      if (index) taskLinks.append(document.createTextNode(", "));
      taskLinks.append(objectLink(id));
    });
    const card = keyValueCard(
      agent.display_name || agent.agent_id,
      { id: agent.agent_id, kind: "member" },
      [
        ["Role", agent.role],
        ["Status", agent.status],
        ["Current tasks", taskLinks.childNodes.length ? taskLinks : "None"],
        ["Recent decision", recent?.chosen_action_id || "No action"],
        ["Relevant protocols", relevant.map((item) => item.protocol_id)],
      ],
      asArray(agent.tools),
    );
    grid.append(card);
  });
  container.append(grid.childNodes.length ? grid : emptyState("No public members in this frame."));
}

function renderTasks(container) {
  const tasks = asArray(organization().tasks);
  if (!tasks.length) {
    container.append(emptyState("No public tasks in this frame."));
    return;
  }
  const table = element("table", "data-table");
  const head = element("tr");
  ["Task", "Owner", "Status", "Progress", "Dependencies", "Protocols", "Evidence"].forEach((name) =>
    head.append(element("th", "", name)),
  );
  const thead = element("thead");
  thead.append(head);
  const tbody = element("tbody");
  tasks.forEach((task) => {
    const row = element("tr");
    const taskCell = element("td");
    append(taskCell, objectLink(task.task_id, task.title || task.task_id), element("p", "muted", task.description || ""));
    const owner = element("td");
    owner.append(task.owner_id ? objectLink(task.owner_id) : document.createTextNode("Unowned"));
    const status = element("td");
    status.append(statusPill(task.status));
    const protocolIds = [
      ...asArray(task.related_protocol_ids),
      ...asArray(task.history).map((item) => item?.protocol_id).filter(Boolean),
    ];
    const protocols = element("td");
    protocols.append(
      hasOwn(task, "related_protocol_ids") || hasOwn(task, "history")
        ? objectLinks(protocolIds)
        : document.createTextNode("Not published"),
    );
    append(
      row,
      taskCell,
      owner,
      status,
      element(
        "td",
        "mono",
        hasOwn(task, "progress_score")
          ? `${Math.round(Number(task.progress_score) * 100)}%`
          : "Not published",
      ),
      element(
        "td",
        "",
        hasOwn(task, "dependencies") ? textValue(task.dependencies, "None") : "Not published",
      ),
      protocols,
      element(
        "td",
        "",
        hasOwn(task, "progress_evidence")
          ? textValue(task.progress_evidence, "None")
          : "Not published",
      ),
    );
    tbody.append(row);
  });
  append(table, thead, tbody);
  const scroller = element("div", "table-scroller");
  scroller.append(table);
  container.append(scroller);
}

function renderEventList(items) {
  const timeline = element("div", "timeline");
  items.forEach(({ event, frameIndex }) => {
    const button = element("button", "event");
    button.type = "button";
    const summary = element("span", "event-summary");
    append(
      summary,
      element("strong", "", titleCase(event.event_type)),
      element("span", "", event.payload?.summary || textValue(event.object_ids, "No public summary")),
    );
    append(
      button,
      element("span", "event-tick", `t${event.tick}`),
      summary,
      element("span", "kind-pill", event.actor_id || "system"),
    );
    button.addEventListener("click", () => {
      setFrame(frameIndex);
      state.selectedId = event.event_id;
      state.selectedFrameIndex = frameIndex;
      render();
    });
    timeline.append(button);
  });
  return timeline;
}

function renderTimeline(container) {
  const events = allEvents().filter((item) => item.frameIndex <= state.frameIndex).reverse();
  renderSnapshotSummary(container);
  container.append(sectionTitle("Public events"));
  container.append(events.length ? renderEventList(events) : emptyState("No public events by this tick."));
}

function renderEpisodes(container) {
  const episodes = asArray(currentFrame().episodes);
  const grid = element("div", "card-grid");
  episodes.forEach((episode) => {
    grid.append(
      keyValueCard(
        episode.title || episode.episode_id,
        { id: episode.episode_id, kind: "episode" },
        [
          ["Trigger / problem", episode.problem_statement],
          ["Participants", episode.participants],
          ["Timeline events", asArray(episode.timeline).length],
          ["Decision", episode.decision_summary],
          ["Outcome", episode.outcome_summary],
        ],
        [episode.status, episode.episode_type],
      ),
    );
  });
  container.append(grid.childNodes.length ? grid : emptyState("No public episodes in this frame."));
}

function renderReflections(container) {
  const notice = element("div", "notice");
  append(
    notice,
    element("strong", "", "Privacy-preserving view. "),
    document.createTextNode(
      "Private reflection text and memory are never exported. This panel shows only public proposal lineage and generated organization-wish identifiers.",
    ),
  );
  container.append(notice, sectionTitle("Public reflection outcomes"));
  const proposals = asArray(organization().proposals).filter(
    (item) => item.source_reflection_id || item.source_wish_id || asArray(item.source_wish_ids).length,
  );
  const grid = element("div", "card-grid");
  proposals.forEach((proposal) => {
    grid.append(
      keyValueCard(
        proposal.title || proposal.proposal_id,
        { id: proposal.proposal_id, kind: "proposal" },
        [
          ["Source episode", proposal.source_episode_id ? objectLink(proposal.source_episode_id) : null],
          ["Private reflection record", proposal.source_reflection_id || "Withheld"],
          ["Generated wish", proposal.source_wish_id || proposal.source_wish_ids],
          ["Perceived blocker", proposal.target_problem],
          ["Public improvement idea", proposal.proposed_solution],
        ],
      ),
    );
  });
  container.append(grid.childNodes.length ? grid : emptyState("No public reflection outcome exists by this tick."));
}

function renderProposals(container) {
  const proposals = asArray(organization().proposals);
  const grid = element("div", "card-grid");
  proposals.forEach((proposal) => {
    grid.append(
      keyValueCard(
        proposal.title || proposal.proposal_id,
        { id: proposal.proposal_id, kind: "proposal" },
        [
          ["Proposer", proposal.proposer_agent_id ? objectLink(proposal.proposer_agent_id) : null],
          ["Rationale", proposal.summary],
          ["Affected process", proposal.family || proposal.target_problem],
          ["Status", proposal.status],
          [
            "Adopted protocol",
            proposal.object_created_id ? objectLink(proposal.object_created_id) : "Not adopted",
          ],
          ["Created → updated", `${proposal.created_at_tick ?? "—"} → ${proposal.updated_at_tick ?? "—"}`],
        ],
        [proposal.proposal_type, ...(proposal.required_actions || [])],
      ),
    );
  });
  container.append(grid.childNodes.length ? grid : emptyState("No public proposals by this tick."));
}

function renderGovernance(container) {
  const proposals = asArray(organization().proposals);
  const grid = element("div", "card-grid");
  proposals.forEach((proposal) => {
    const latency = proposal.adopted_tick == null || proposal.created_at_tick == null
      ? "Pending"
      : `${proposal.adopted_tick - proposal.created_at_tick} ticks`;
    grid.append(
      keyValueCard(
        proposal.title || proposal.proposal_id,
        { id: proposal.proposal_id, kind: "proposal" },
        [
          ["Required approvers", proposal.approval_required_from],
          ["Support / approved", proposal.approved_by || proposal.supporters],
          ["Oppose / rejected", proposal.rejected_by || proposal.opposers],
          ["Changes requested", proposal.suggested_revision],
          ["Adoption", proposal.status],
          ["Review latency", latency],
        ],
      ),
    );
  });
  container.append(grid.childNodes.length ? grid : emptyState("No governance review exists by this tick."));
  const events = allGovernanceEvents().filter((item) => item.frameIndex <= state.frameIndex).reverse();
  container.append(sectionTitle("Protocol lifecycle ledger"));
  if (!events.length) {
    container.append(emptyState("No public protocol lifecycle event exists by this tick."));
    return;
  }
  const ledger = element("div", "card-grid");
  events.forEach(({ event, frameIndex }) => {
    ledger.append(
      keyValueCard(
        titleCase(event.event_type),
        { id: event.event_id, kind: "governance event" },
        [
          ["Protocol", objectLink(event.protocol_id, null, frameIndex)],
          ["Actor", event.actor_id ? objectLink(event.actor_id, null, frameIndex) : "System"],
          ["Tick", event.tick],
          ["Public data", event.data],
        ],
      ),
    );
  });
  container.append(ledger);
}

function renderProtocols(container) {
  const protocols = asArray(organization().protocols);
  const grid = element("div", "card-grid");
  protocols.forEach((protocol) => {
    grid.append(
      keyValueCard(
        protocol.rule_summary || protocol.protocol_id,
        { id: protocol.protocol_id, kind: "protocol" },
        [
          ["Type / version", `${textValue(protocol.protocol_type)} / ${textValue(protocol.version ?? protocol.revision, "v1")}`],
          [
            "Source proposal",
            protocol.created_from_proposal_id
              ? objectLink(protocol.created_from_proposal_id)
              : "Not published",
          ],
          ["Trigger", protocol.trigger_condition || protocol.proposal_event_id],
          ["Scope", protocol.scope],
          ["Adoption", protocol.adoption_status],
          ["Activation", protocol.first_tick],
          [
            "Usage records",
            hasOwn(protocol, "usage_events")
              ? objectLinks(protocol.usage_events)
              : "Not published",
          ],
          [
            "Violation records",
            hasOwn(protocol, "violation_events")
              ? objectLinks(protocol.violation_events)
              : "Not published",
          ],
          [
            "Enforcement records",
            hasOwn(protocol, "enforcement_events")
              ? objectLinks(protocol.enforcement_events)
              : "Not published",
          ],
          [
            "Amendments",
            hasOwn(protocol, "revisions") ? asArray(protocol.revisions).length : "Not published",
          ],
          ["Retirement", protocol.retired_tick],
        ],
        [protocol.status, protocol.target_process],
      ),
    );
  });
  container.append(grid.childNodes.length ? grid : emptyState("No active or historical protocols by this tick."));
}

function renderGenericCollection(container, values, emptyText) {
  const items = asArray(values);
  if (!items.length) {
    container.append(emptyState(emptyText));
    return;
  }
  const grid = element("div", "card-grid");
  items.forEach((item, index) => {
    const identity = objectIdentity(item);
    const title = item.title || item.name || identity?.id || `Record ${index + 1}`;
    const card = keyValueCard(title, identity, [["Public record", JSON.stringify(item, null, 2)]]);
    grid.append(card);
  });
  container.append(grid);
}

function renderArtifacts(container) {
  const frame = currentFrame();
  renderGenericCollection(
    container,
    organization(frame).artifacts || frame.artifacts,
    "This trace does not publish artifact records. Absence is not interpreted as zero artifacts.",
  );
}

function renderRepo(container) {
  const frame = currentFrame();
  const repoState = frame.repo_state || organization(frame).repo_state;
  renderGenericCollection(
    container,
    repoState ? [repoState] : [],
    "This trace does not publish branch, commit, PR, review, CI, or merge state.",
  );
}

function renderEvidence(container) {
  const frame = currentFrame();
  renderGenericCollection(
    container,
    frame.evaluation_annotations || organization(frame).evidence || state.trace.evaluation_annotations,
    "This trace does not publish tests, evaluator outputs, or evidence annotations.",
  );
}

function renderDecisions(container) {
  const decisions = asArray(currentFrame().decisions);
  const grid = element("div", "card-grid");
  decisions.forEach((decision) => {
    grid.append(
      keyValueCard(
        `${decision.agent_id} · ${decision.chosen_action_id || "no action"}`,
        { id: decision.decision_id, kind: "decision" },
        [
          ["Member", objectLink(decision.agent_id)],
          ["Chosen object", decision.chosen_object_id ? objectLink(decision.chosen_object_id) : null],
          ["Tick", decision.tick],
        ],
      ),
    );
  });
  container.append(grid.childNodes.length ? grid : emptyState("No structured decisions at this tick."));
}

const panelRenderers = {
  overview: renderOverview,
  members: renderMembers,
  tasks: renderTasks,
  timeline: renderTimeline,
  episodes: renderEpisodes,
  reflections: renderReflections,
  proposals: renderProposals,
  governance: renderGovernance,
  protocols: renderProtocols,
  artifacts: renderArtifacts,
  repo: renderRepo,
  evidence: renderEvidence,
  decisions: renderDecisions,
};

function addCollectionObjects(objects, values, fallbackKind) {
  asArray(values).forEach((value) => {
    if (!value || typeof value !== "object") return;
    const identity = objectIdentity(value);
    if (identity) objects.push({ ...identity, value });
    else if (value.id) objects.push({ id: String(value.id), kind: fallbackKind, value });
  });
}

function frameObjects(frame, trace = state.trace) {
  if (!frame) return [];
  const objects = [];
  const org = organization(frame);
  const organizationId = org.organization_id || trace?.organization_id;
  if (organizationId) {
    objects.push({ id: organizationId, kind: "organization", value: org });
  }
  [
    [org.agents, "member"],
    [org.tasks, "task"],
    [org.proposals, "proposal"],
    [org.protocols, "protocol"],
    [org.artifacts || frame.artifacts, "artifact"],
    [frame.events, "event"],
    [frame.episodes, "episode"],
    [frame.evaluation_annotations || org.evidence, "evidence"],
  ].forEach(([values, fallbackKind]) => addCollectionObjects(objects, values, fallbackKind));

  asArray(frame.governance_events).forEach((event) => {
    if (event?.event_id) {
      objects.push({ id: event.event_id, kind: "governance event", value: event });
    }
  });
  asArray(frame.decisions).forEach((decision) => {
    if (decision?.decision_id) {
      objects.push({ id: decision.decision_id, kind: "decision", value: decision });
    }
  });

  const repoState = frame.repo_state || org.repo_state;
  if (repoState && typeof repoState === "object") {
    const repoId = repoState.repository_id || repoState.repo_id || "repo_state";
    objects.push({ id: String(repoId), kind: "repository", value: repoState });
    ["branches", "commits", "pull_requests", "prs", "reviews", "ci", "ci_runs", "merges"].forEach(
      (key) => addCollectionObjects(objects, repoState[key], key),
    );
  }

  return [...new Map(objects.filter((item) => item.id).map((item) => [item.id, item])).values()];
}

function findObject(id, preferredFrame = state.frameIndex) {
  if (!id || !state.trace) return null;
  for (let index = preferredFrame; index >= 0; index -= 1) {
    const found = frameObjects(state.trace.frames[index]).find((item) => item.id === id);
    if (found) return { ...found, frameIndex: index };
  }
  return null;
}

function visibleObjectsThrough(frameIndex) {
  const objects = new Map();
  for (let index = 0; index <= frameIndex; index += 1) {
    frameObjects(state.trace.frames[index]).forEach((item) => {
      objects.set(item.id, { ...item, frameIndex: index });
    });
  }
  return objects;
}

function selectObject(id, frameIndex = null) {
  if (frameIndex !== null) setFrame(frameIndex);
  state.selectedId = id;
  state.selectedFrameIndex = frameIndex ?? state.frameIndex;
  renderObjectInspector();
}

function containsReference(value, id) {
  if (value === id) return true;
  if (Array.isArray(value)) return value.some((item) => containsReference(item, id));
  if (value && typeof value === "object") return Object.values(value).some((item) => containsReference(item, id));
  return false;
}

function renderObjectInspector() {
  const container = $("object-inspector");
  container.replaceChildren();
  const selected = findObject(state.selectedId, state.selectedFrameIndex ?? state.frameIndex)
    || frameObjects(currentFrame())[0];
  if (!selected) {
    append(container, element("h2", "", "Object Inspector"), element("p", "", "Select an event or object to inspect its public record."));
    return;
  }
  state.selectedId = selected.id;
  state.selectedFrameIndex = selected.frameIndex ?? state.frameIndex;
  const title = element("div", "object-title");
  append(title, element("h2", "", selected.id), element("span", "kind-pill", selected.kind));
  container.append(title);
  container.append(element("p", "mono", `Visible at tick ${state.trace.frames[state.selectedFrameIndex].tick}`));

  const related = [...visibleObjectsThrough(state.selectedFrameIndex).values()].filter(
    (candidate) =>
      candidate.id !== selected.id &&
      (containsReference(candidate.value, selected.id) ||
        containsReference(selected.value, candidate.id)),
  );
  if (related.length) {
    container.append(sectionTitle("Related objects"));
    const links = element("div", "related-list");
    related.slice(0, 20).forEach((item) =>
      links.append(objectLink(item.id, item.id, item.frameIndex)),
    );
    container.append(links);
  }
  const pre = element("pre");
  pre.textContent = JSON.stringify(selected.value, null, 2);
  container.append(sectionTitle("Public fields"), pre);
}

function objectMap(frame, trace = state.trace) {
  const transientKinds = new Set(["event", "decision", "governance event"]);
  return new Map(
    frameObjects(frame, trace)
      .filter((item) => !transientKinds.has(item.kind))
      .map((item) => [item.id, item]),
  );
}

function changedFields(before, after) {
  const keys = new Set([...Object.keys(before || {}), ...Object.keys(after || {})]);
  return [...keys].filter((key) => JSON.stringify(before?.[key]) !== JSON.stringify(after?.[key]));
}

function computeFrameDiff(beforeFrame, afterFrame, trace = state.trace) {
  if (!beforeFrame) return [];
  const before = objectMap(beforeFrame, trace);
  const after = objectMap(afterFrame, trace);
  const ids = new Set([...before.keys(), ...after.keys()]);
  const diffs = [];
  ids.forEach((id) => {
    if (!before.has(id)) diffs.push({ id, kind: after.get(id).kind, type: "added", fields: [] });
    else if (!after.has(id)) diffs.push({ id, kind: before.get(id).kind, type: "removed", fields: [] });
    else {
      const fields = changedFields(before.get(id).value, after.get(id).value);
      if (fields.length) diffs.push({ id, kind: after.get(id).kind, type: "changed", fields });
    }
  });
  return diffs;
}

function computeDiff() {
  if (state.frameIndex === 0) return [];
  return computeFrameDiff(state.trace.frames[state.frameIndex - 1], currentFrame());
}

function renderStateDiff() {
  const container = $("state-diff");
  container.replaceChildren();
  const currentTick = currentFrame().tick;
  const previousTick = state.frameIndex ? state.trace.frames[state.frameIndex - 1].tick : null;
  const heading = element("div", "diff-heading");
  append(
    heading,
    element("h2", "", "State Diff"),
    element("span", "mono", previousTick === null ? `Initial state · tick ${currentTick}` : `tick ${previousTick} → ${currentTick}`),
  );
  container.append(heading);
  const diffs = computeDiff();
  if (!diffs.length) {
    container.append(emptyState(previousTick === null ? "The first frame establishes the baseline." : "No public organization object changed."));
    return;
  }
  const grid = element("div", "diff-grid");
  diffs.slice(0, 36).forEach((diff) => {
    const item = element("div", `diff-item ${diff.type}`);
    append(
      item,
      objectLink(diff.id),
      element("span", "", diff.fields.length ? `${diff.type}: ${diff.fields.join(", ")}` : `${diff.type} ${diff.kind}`),
    );
    grid.append(item);
  });
  container.append(grid);
}

function navCounts() {
  const frame = currentFrame();
  const org = organization(frame);
  const artifactsPublished = hasOwn(org, "artifacts") || hasOwn(frame, "artifacts");
  const repoPublished = hasOwn(frame, "repo_state") || hasOwn(org, "repo_state");
  const evidencePublished =
    hasOwn(frame, "evaluation_annotations") ||
    hasOwn(org, "evidence") ||
    hasOwn(state.trace, "evaluation_annotations");
  return {
    members: asArray(org.agents).length,
    tasks: asArray(org.tasks).length,
    timeline: allEvents().filter((item) => item.frameIndex <= state.frameIndex).length,
    episodes: asArray(frame.episodes).length,
    reflections: asArray(org.proposals).filter((item) => item.source_reflection_id).length,
    proposals: asArray(org.proposals).length,
    governance: asArray(org.proposals).length,
    protocols: asArray(org.protocols).length,
    artifacts: artifactsPublished ? asArray(org.artifacts || frame.artifacts).length : null,
    repo: repoPublished ? 1 : null,
    evidence: evidencePublished
      ? asArray(frame.evaluation_annotations || org.evidence || state.trace.evaluation_annotations).length
      : null,
    decisions: asArray(frame.decisions).length,
  };
}

function renderNav() {
  const counts = navCounts();
  document.querySelectorAll("#primary-nav button[data-panel]").forEach((button) => {
    button.classList.toggle("active", button.dataset.panel === state.panel);
    button.querySelector(".nav-count")?.remove();
    const count = counts[button.dataset.panel];
    if (count !== undefined) {
      button.append(element("span", "nav-count", count === null ? "—" : count));
    }
  });
}

function renderTransport() {
  const frames = state.trace.frames;
  $("frame-slider").max = String(Math.max(0, frames.length - 1));
  $("frame-slider").value = String(state.frameIndex);
  $("tick-label").textContent = `Tick ${currentFrame().tick}`;
  $("frame-label").textContent = `Frame ${state.frameIndex + 1} / ${frames.length}`;
  $("play-pause").textContent = state.playing ? "Pause" : "Play";
  $("previous-frame").disabled = state.frameIndex === 0;
  $("first-frame").disabled = state.frameIndex === 0;
  $("next-frame").disabled = state.frameIndex === frames.length - 1;
  $("last-frame").disabled = state.frameIndex === frames.length - 1;
}

function renderPanel() {
  const container = $("panel");
  container.replaceChildren(panelHeader(state.panel));
  panelRenderers[state.panel](container);
}

function render() {
  if (!state.trace) return;
  renderTransport();
  renderNav();
  renderPanel();
  renderObjectInspector();
  renderStateDiff();
}

function setFrame(index) {
  if (!state.trace) return;
  state.frameIndex = Math.max(0, Math.min(Number(index), state.trace.frames.length - 1));
  state.selectedFrameIndex = state.frameIndex;
  render();
}

function stopPlayback() {
  state.playing = false;
  if (state.playTimer) window.clearInterval(state.playTimer);
  state.playTimer = null;
}

function togglePlayback() {
  if (state.playing) {
    stopPlayback();
    renderTransport();
    return;
  }
  if (state.frameIndex === state.trace.frames.length - 1) state.frameIndex = 0;
  state.playing = true;
  state.playTimer = window.setInterval(() => {
    if (state.frameIndex >= state.trace.frames.length - 1) {
      stopPlayback();
      render();
      return;
    }
    setFrame(state.frameIndex + 1);
  }, 900);
  renderTransport();
}

async function getJson(path) {
  const response = await fetch(path, { cache: "no-store" });
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || `${response.status} ${response.statusText}`);
  return body;
}

async function refreshTrace() {
  try {
    if (state.health?.mode === "live") state.health = await getJson("/api/health");
    const next = await getJson("/api/trace");
    const oldLength = state.trace?.frames?.length || 0;
    state.trace = next;
    state.frameIndex = nextFrameIndexAfterRefresh(
      state.frameIndex,
      oldLength,
      next.frames.length,
    );
    state.selectedFrameIndex = Math.min(
      state.selectedFrameIndex ?? state.frameIndex,
      next.frames.length - 1,
    );
    const degraded = state.health?.status === "degraded";
    $("trace-status").className = degraded ? "status-dot error" : "status-dot ready";
    $("trace-status").textContent = degraded ? "Last verified trace · source updating" : "Trace verified";
    $("run-id").textContent = `run ${next.run_id}`;
    render();
  } catch (error) {
    $("trace-status").className = "status-dot error";
    $("trace-status").textContent = "Trace unavailable";
    if (!state.trace) throw error;
  }
}

function installControls() {
  $("first-frame").addEventListener("click", () => setFrame(0));
  $("previous-frame").addEventListener("click", () => setFrame(state.frameIndex - 1));
  $("play-pause").addEventListener("click", togglePlayback);
  $("next-frame").addEventListener("click", () => setFrame(state.frameIndex + 1));
  $("last-frame").addEventListener("click", () => setFrame(state.trace.frames.length - 1));
  $("frame-slider").addEventListener("input", (event) => setFrame(event.target.value));
  document.querySelectorAll("#primary-nav button[data-panel]").forEach((button) => {
    button.addEventListener("click", () => {
      state.panel = button.dataset.panel;
      render();
    });
  });
  window.addEventListener("keydown", (event) => {
    const target = event.target;
    if (
      event.altKey ||
      event.ctrlKey ||
      event.metaKey ||
      target instanceof HTMLInputElement ||
      target instanceof HTMLButtonElement ||
      target instanceof HTMLSelectElement ||
      target instanceof HTMLTextAreaElement ||
      target?.isContentEditable
    ) return;
    if (event.key === "ArrowLeft") setFrame(state.frameIndex - 1);
    if (event.key === "ArrowRight") setFrame(state.frameIndex + 1);
    if (event.key === " ") {
      event.preventDefault();
      togglePlayback();
    }
  });
}

async function boot() {
  installControls();
  try {
    state.health = await getJson("/api/health");
    const badge = $("mode-badge");
    badge.textContent = state.health.mode;
    badge.classList.toggle("live", state.health.mode === "live");
    await refreshTrace();
    if (state.health.mode === "live") {
      state.refreshTimer = window.setInterval(refreshTrace, 1500);
    }
  } catch (error) {
    const fatal = $("fatal-error");
    fatal.hidden = false;
    fatal.textContent = `Relic Inspector could not load the public trace.\n\n${error.message}`;
  }
}

function nextFrameIndexAfterRefresh(currentIndex, oldLength, nextLength) {
  if (oldLength === 0) return nextLength - 1;
  if (oldLength > 0 && currentIndex === oldLength - 1) return nextLength - 1;
  return Math.min(currentIndex, nextLength - 1);
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = { computeFrameDiff, nextFrameIndexAfterRefresh };
}

if (typeof document !== "undefined") boot();
