"""Static source guardrails for the independent P3 façade bundle."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2] / "environments" / "org_env" / "frontend" / "app"
LIAISON = ROOT / "src" / "liaison"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_vite_has_an_independent_liaison_entry():
    config = _read(ROOT / "vite.config.ts")
    html = _read(ROOT / "liaison.html")
    main = _read(LIAISON / "main.tsx")
    assert 'liaison: "liaison.html"' in config
    assert "/src/liaison/main.tsx" in html
    assert "LiaisonApp" in main


def test_p3_does_not_import_or_call_the_p2_seat_surface():
    source = "\n".join(_read(path) for path in LIAISON.glob("*.ts"))
    assert "useSeat" not in source
    assert "SeatApp" not in source
    assert "/api/org/human/view" not in source
    assert "/api/org/human/offers" not in source
    assert "/api/org/human/act" not in source
    assert "/api/org/liaison/" not in source  # wrapper owns the prefix
    assert "/api/org/liaison" in _read(LIAISON / "api.ts")


def test_p3_defaults_to_summary_and_has_progressive_disclosure():
    app = _read(LIAISON / "LiaisonApp.tsx")
    assert "Current project context" in app
    assert "Secretary liaison thread" in app
    assert "Show details" in app
    assert "Show trace" in app
    assert "Show organization" in app
    assert "read-only" in app
    assert "Clarifications are welcome" in app
    assert "body?.event_summary" in app
    assert "eventSummary?.actor" in app
    assert "eventSummary?.action" in app
    assert "eventSummary?.organization_tick" in app
    assert "evidenceHeadline !== messageHeadline" in app


def test_p3_matches_light_liaison_desktop_information_architecture():
    app = _read(LIAISON / "LiaisonApp.tsx")
    css = _read(LIAISON / "liaison.css")
    for marker in ("left-rail", "Secretary", "sidebar-projects", "project-header",
                   "details-toggle", "SlidersIcon", "aria-expanded", "In this conversation", "Current work",
                   "Who is involved", "Why this work", "View organization details",
                   "decision-needed", "fyi-digest", "execution-digest", "ready-turn",
                   "isDecisionEvent", "isExecutionEvent", "isReadyMessage", "eventTone", "tone"):
        assert marker in app or marker in css, marker
    assert "--canvas: #f7f9fc" in css
    assert "--surface: #ffffff" in css
    assert "details-visible" in css
    assert 'aria-label="Experience version"' in app
    assert 'href="/org/seat"' in app
    assert "P3 Secretary liaison" in app


def test_p3_attention_mediation_cards_keep_decisions_in_the_composer():
    app = _read(LIAISON / "LiaisonApp.tsx")
    api = _read(LIAISON / "api.ts")
    css = _read(LIAISON / "liaison.css")
    for marker in ("meeting_invitation", "MeetingInvitationTurn", "meeting.agenda", "meeting.participants",
                   "decision_required", "do not need to answer immediately", "incoming_message",
                   "Incoming message", "Secretary screened", "importance", "urgency", "delivery_mode",
                   "meeting_report", "Meeting report", "Post-meeting summary", "attention_preference",
                   "meeting_instruction", "attention_preferences", "message_delivery", "Pending meeting choice",
                   "all messages", "secretary triage", "meeting_invitations"):
        assert marker in app or marker in api, marker
    assert "MeetingInvitationTurn" in app
    assert "TurnDisclosure message={message}" in app
    assert "textarea" in app
    invitation_component = app.split("function MeetingInvitationTurn", 1)[1].split("function IncomingMessageTurn", 1)[0].lower()
    assert "liaisonapi.confirm" not in invitation_component
    assert "Ask for meeting context" in app
    assert "Attend" in app
    assert "Delegate to secretary" in app
    assert "Do not RSVP yet" in app
    assert "oncompose?." in invitation_component
    for marker in (".meeting-card", ".incoming-card", ".report-card", ".meeting-choice", ".message-facts"):
        assert marker in css
    assert "detail-evidence-list" in css


def test_liaison_locking_reads_do_not_block_the_asgi_event_loop():
    backend = _read(ROOT.parent.parent / "backend" / "main.py")
    session_route = backend.split('@app.post("/api/org/liaison/session")', 1)[1].split(
        '@app.post("/api/org/liaison/release")', 1)[0]
    resource_route = backend.split('@app.get("/api/org/liaison/resource")', 1)[1].split(
        '@app.post("/api/org/liaison/ask")', 1)[0]
    assert "await run_in_threadpool" in session_route
    assert "await run_in_threadpool" in resource_route


def test_p3_uses_one_bottom_composer_and_hides_dashboard_console_surfaces():
    app = _read(LIAISON / "LiaisonApp.tsx")
    css = _read(LIAISON / "liaison.css")
    assert app.count("<textarea") == 1
    assert "composer-zone" in app
    assert "task-list" not in app.lower()
    assert "file-tree" not in app.lower()
    assert "terminal" not in app.lower()
    assert "model picker" not in app.lower()
    assert "agent manager" not in app.lower()
    assert ".task-list" not in css
    assert ".file-tree" not in css
    assert ".terminal" not in css


def test_p3_opens_a_seat_scoped_resource_inspector_instead_of_an_always_on_ide():
    app = _read(LIAISON / "LiaisonApp.tsx")
    api = _read(LIAISON / "api.ts")
    css = _read(LIAISON / "liaison.css")
    for marker in (
        "ResourceInspector", "Open full context", "RESOURCE_SECTION_BY_TAB",
        'summary: "overview"', 'tests: "ci"', 'evidence: "provenance"',
        'role="tablist"', 'role="tab"', 'role="tabpanel"',
        "resourcePayload", "payload.content", "unified_diff", "failure_reasons",
    ):
        assert marker in app, marker
    assert 'get(`/resource?token=${encodeURIComponent(token)}&ref=${encodeURIComponent(ref)}' in api
    assert "ri_" in app
    assert "pr_id=" not in api
    assert "patch_id=" not in api
    assert "file_path=" not in api
    assert ".resource-inspector-scroll" in css
    assert "overflow: auto" in css


def test_p3_null_resource_payload_and_bad_collection_rows_cannot_blank_the_app():
    app = _read(LIAISON / "LiaisonApp.tsx")
    main = _read(LIAISON / "main.tsx")
    css = _read(LIAISON / "liaison.css")
    resource_helper = app.split("const resourceRefOf", 1)[1].split("};", 1)[0]
    assert 'if (!message || typeof message !== "object") return ""' in resource_helper
    assert "Array.isArray(message.resource_refs)" in resource_helper
    assert "normalizeP3State(await liaisonApi.state(token))" in app
    assert "objectRows<P3ExecutionJob>(state?.execution_jobs)" in app
    assert "objectRows<P3ExecutionWorker>(job?.workers)" in app
    assert "objectRows<P3ExecutionTimelineItem>(job?.timeline)" in app
    assert "class LiaisonErrorBoundary" in main
    assert "P3 liaison render failure" in main
    assert "Reload secretary view" in main
    assert "<LiaisonErrorBoundary>" in main
    assert ".liaison-crash" in css


def test_p3_failure_card_distinguishes_invalid_plans_from_interrupted_reads():
    app = _read(LIAISON / "LiaisonApp.tsx")
    api = _read(LIAISON / "api.ts")
    card = app.split("function FailedRequestTurn", 1)[1].split(
        "function ThreadPanel", 1)[0]

    assert 'request.failure_kind === "invalid_model_plan"' in card
    assert "Plan rejected · no action ran" in card
    assert "Request interrupted · input preserved" in card
    assert "Retry allocation" in card
    assert "Retry request" in card
    assert "Resume from step" in card
    assert "Retry from step" not in card
    assert "failure_kind?" in api
    assert "retry_current_step?" in api


def test_p3_navigation_controls_are_real_local_views_not_codex_placeholders():
    app = _read(LIAISON / "LiaisonApp.tsx")
    css = _read(LIAISON / "liaison.css")
    for marker in ("Conversation", "Decisions", "Activity", "setNavigation", "aria-current",
                   "decisionMessages", "activityMessages", "EmptyView"):
        assert marker in app, marker
    assert "New conversation" not in app
    assert "Pull requests" not in app
    assert ".rail-nav-item.active" in css
    navigation = app.split("const setNavigation", 1)[1].split("};", 1)[0]
    assert 'setThreadRootId("")' in navigation
    assert "setOrganizationOpen(false)" in navigation
    assert "closeResource()" in navigation


def test_p3_decision_inbox_has_model_routed_composer_actions():
    app = _read(LIAISON / "LiaisonApp.tsx")
    api = _read(LIAISON / "api.ts")
    css = _read(LIAISON / "liaison.css")
    for marker in ("DecisionRequestTurn", "Decision needed", "Human judgment required",
                   "decision_options", "composeDecision", "Ask secretary for context",
                   "Nothing is routed until you send it", "consequential actions still require confirmation"):
        assert marker in app, marker
    assert "onClick={() => onCompose?.(option.instruction, decisionContextOf(message, option))}" in app
    assert "setText(instruction)" in app
    assert "setDecisionContext(context)" in app
    assert "decisionInboxMessages.find((message) => message.id === threadRootId)" in app
    assert "threadable: true" in app
    assert "Replying to" in app
    assert "Remove decision reference" in app
    assert "decision_context: submittedContext" in app
    assert "decision_context?: P3DecisionContext" in api
    assert "Matter" in app
    assert "Why this is surfaced" in app
    assert "Impact" in app
    assert "Current recorded context" in app
    assert ".decision-request-actions" in css
    decision = app.split("function DecisionRequestTurn", 1)[1].split("function Composer", 1)[0]
    assert "liaisonApi.confirm" not in decision
    assert "liaisonApi.ask" not in decision


def test_p3_decision_card_can_open_its_direct_resource_as_well_as_show_evidence():
    """Decision evidence must not require a second NL request before source review."""
    app = _read(LIAISON / "LiaisonApp.tsx")
    thread_turn = app.split("function ThreadTurn", 1)[1].split("function DecisionRequestTurn", 1)[0]
    decision = app.split("function DecisionRequestTurn", 1)[1].split("function Composer", 1)[0]

    # The parent turn carries the same inspector callback used for FYI cards.
    assert "<DecisionRequestTurn message={message} token={token} onCompose={onComposeDecision} onOpenResource={onOpenResource} />" in thread_turn
    # The decision's ordinary disclosure then exposes a direct `ri_` source
    # button after the server provides its resource_ref; this does not route an
    # approval or otherwise widen decision authority.
    assert "onOpenResource?: (message: P3Message, tab?: ResourceTab) => void" in decision
    assert "const resourceRef = resourceRefOf(message);" in decision
    assert 'message.preferred_resource_tab === "review"' in decision
    assert "onClick={() => onOpenResource({ ...message, resource_ref: resourceRef }, decisionResourceTab)}" in decision
    assert ">Open decision context</button>" in decision
    assert "<TurnDisclosure message={message} token={token} onOpenResource={onOpenResource} contextLabel=\"Open decision context\" resourceTab={decisionResourceTab} />" in decision


def test_p3_details_render_complete_visible_event_context_and_related_objects():
    app = _read(LIAISON / "LiaisonApp.tsx")
    disclosure = app.split("function TurnDisclosure", 1)[1].split("const resourceValue", 1)[0]

    assert "Complete seat-visible event record" in disclosure
    assert "Related objects visible now" in disclosure
    assert "body?.event_summary" in disclosure
    assert "body?.related_current_visible_objects" in disclosure
    assert "<ObjectFacts value={eventRecord}" in disclosure
    assert "<ObjectFacts value={facts}" in disclosure
    assert "Open full context" in disclosure


def test_p3_top_right_details_panel_exposes_decision_sources():
    app = _read(LIAISON / "LiaisonApp.tsx")
    details = app.split("function DetailsPanel", 1)[1].split("function ExecutionDetails", 1)[0]

    assert "onOpenResource" in details
    assert "Open decision context" in details
    assert "Open full context" in details
    assert "resourceRefOf(item)" in details
    assert 'item.preferred_resource_tab === "review"' in details


def test_p3_open_resource_is_not_starved_by_live_worker_updates():
    """Timeline polling must not cancel an in-flight section request forever."""
    app = _read(LIAISON / "LiaisonApp.tsx")

    resource_effect = app.split(
        "useEffect(() => {\n    if (resourceTarget) void loadResource", 1
    )[1].split("useEffect(() => { end.current", 1)[0]
    assert "[loadResource, resourceTarget, resourceTab]" in resource_effect
    assert "resourceStateRevision" not in app

    details = app.split("function DetailsPanel", 1)[1].split(
        "function ExecutionDetails", 1)[0]
    assert "evidenceRefOf(message) || resourceRefOf(message)" in details


def test_p3_only_a_new_explicit_secretary_focus_can_move_the_inspector():
    """Only the latest explicit focus command is applied once per revision."""
    app = _read(LIAISON / "LiaisonApp.tsx")
    command = app.split("const secretaryResourceFocusCommandOf", 1)[1].split(
        "const isOrganizationEvent", 1)[0]
    focus_effect = app.split("useEffect(() => {\n    const command =", 1)[1].split(
        "useEffect(() => {\n    if (resourceTarget)", 1)[0]

    assert 'message.role !== "liaison" || !message.resource_focus' in command
    assert "message.resource_ref" not in command
    assert "key: `${messageId}\\u0000${target.ref}\\u0000${tab}`" in command
    assert "state?.resource_focus" not in focus_effect
    assert "resourceFocusOf(state)" not in focus_effect
    assert ".find(Boolean)" in focus_effect
    assert "candidate && !" not in focus_effect
    assert "lastConsumedResourceFocusCommand.current" in focus_effect
    assert "command.key === lastConsumedResourceFocusCommand.current" in focus_effect
    assert "lastConsumedResourceFocusCommand.current = command.key" in focus_effect
    assert "selectResource(command.target, command.tab)" in focus_effect
    assert "dismissedResourceRef" not in app


def test_p3_ignores_historical_focuses_and_an_already_active_resource_tab():
    app = _read(LIAISON / "LiaisonApp.tsx")
    focus_effect = app.split("useEffect(() => {\n    const command =", 1)[1].split(
        "useEffect(() => {\n    if (resourceTarget)", 1)[0]
    tab_selection = app.split("const selectResourceTab", 1)[1].split(
        "const setNavigation", 1)[0]

    # reverse() followed by find(Boolean) selects only the newest command;
    # its key remains recorded, so the next poll cannot replay older history.
    assert "[...(state?.conversation || [])].reverse()" in focus_effect
    assert ".map(secretaryResourceFocusCommandOf)" in focus_effect
    assert ".find(Boolean)" in focus_effect
    assert "if (tab === resourceTab) return;" in tab_selection
    assert "resourceSelection.current = { ref: resourceTarget.ref, tab }" in tab_selection
    assert "setResourceTab(tab)" in tab_selection
    assert "setResourcePayload(null)" not in tab_selection


def test_p3_resource_loads_cannot_overwrite_a_newer_resource_or_tab():
    app = _read(LIAISON / "LiaisonApp.tsx")
    loader = app.split("const loadResource = useCallback", 1)[1].split(
        "useEffect(() => {\n    const command =", 1)[0]

    assert "const resourceRequestId = useRef(0)" in app
    assert "const resourceSelection = useRef" in app
    assert "const requestId = ++resourceRequestId.current" in loader
    assert "requestId === resourceRequestId.current" in loader
    assert "resourceSelection.current?.ref === target.ref" in loader
    assert "resourceSelection.current.tab === section" in loader
    assert "const closeSelectedResource" in app
    assert "resourceSelection.current = null" in app


def test_p3_resource_inspector_shows_one_of_loading_error_or_content():
    app = _read(LIAISON / "LiaisonApp.tsx")
    inspector = app.split("function ResourceInspector", 1)[1].split(
        "function AssistantTurn", 1)[0]

    assert "{busy ? <div className=\"resource-loading\"" in inspector
    assert ": error ? <div className=\"inline-error\"" in inspector
    assert ": !payload ? <div className=\"resource-loading\"" in inspector
    assert ": content}" in inspector
    assert "{busy && <div className=\"resource-loading\"" not in inspector


def test_p3_shows_elapsed_progress_during_the_initial_model_route():
    """The first provider call must not look like a frozen, context-free page."""
    app = _read(LIAISON / "LiaisonApp.tsx")

    assert "const optimisticInFlight" in app
    assert 'phase: "awaiting_model"' in app
    assert 'summary: "Waiting for the liaison model to interpret your request."' in app
    assert "elapsed_seconds: Math.max" in app
    assert "const visibleWorkingState" in app
    assert 'working={visibleWorkingState}' in app
    assert "workingState={visibleWorkingState}" in app


def test_p3_keeps_optimistic_human_text_until_the_server_echo_exists():
    app = _read(LIAISON / "LiaisonApp.tsx")
    assert "const serverMessages = state?.conversation || []" in app
    assert "serverMessages.some((server) => server.role === \"human\"" in app
    assert "message.role === \"human\"" in app
    ask = app.split("const ask = async", 1)[1].split("const askThread", 1)[0]
    thread_ask = app.split("const askThread = async", 1)[1].split("const retryFailedRequest", 1)[0]
    assert "setOptimisticHumanMessages((current) => current.filter" not in ask
    assert "setOptimisticHumanMessages((current) => current.filter" not in thread_ask


def test_p3_thread_decision_actions_fill_the_thread_composer():
    app = _read(LIAISON / "LiaisonApp.tsx")
    assert "const composeThreadDecision" in app
    assert 'document.querySelector<HTMLTextAreaElement>(".thread-composer-zone .composer-card textarea")' in app
    root_call = app.split("{threadRoot && <ThreadPanel", 1)[1].split("/>}", 1)[0]
    assert "onComposeDecision={composeThreadDecision}" in root_call
    panel = app.split("function ThreadPanel", 1)[1].split("function ExecutionJobTurn", 1)[0]
    assert "onComposeDecision={onComposeDecision}" in panel


def test_p3_uses_the_runtime_cancelable_flag_without_guessing_statuses():
    app = _read(LIAISON / "LiaisonApp.tsx")
    assert "const executionCanCancel = (job: P3ExecutionJob) => job.cancelable === true" in app


def test_p3_recent_activity_can_open_trace_and_current_resource_context():
    app = _read(LIAISON / "LiaisonApp.tsx")
    details = app.split("function DetailsPanel", 1)[1].split("function ExecutionDetails", 1)[0]
    assert '<TurnDisclosure message={event as P3Message}' in details
    assert 'contextLabel="Open activity context"' in details


def test_p3_resource_files_dedupe_exact_secretary_and_worker_reads():
    """The same exact source read once by each actor appears once in Files."""
    app = _read(LIAISON / "LiaisonApp.tsx")
    inspector = app.split("function ResourceInspector", 1)[1].split(
        "function AssistantTurn", 1)[0]

    assert "const evidenceFileGroups = new Map" in inspector
    assert 'const key = `${path}\\u0000${content}`' in inspector
    assert 'workers.join(" + ")' in inspector
    assert "[...new Map([...returnedFiles, ...evidenceFiles].map" in inspector


def test_p3_surfaces_real_runtime_status_and_pause_resume_routes():
    app = _read(LIAISON / "LiaisonApp.tsx")
    api = _read(LIAISON / "api.ts")
    css = _read(LIAISON / "liaison.css")
    for marker in ("P3RuntimeStatus", "liaisonApi.runtime", "runtimeStart", "runtimePause",
                   "world_tick", "seconds_per_tick", "last_error", "Simulation live",
                   "Simulation paused", "Pause", "Resume", "live OrgWorld events",
                   "event_source", "rule_template_only", "action_selection_mode"):
        assert marker in app or marker in api, marker
    assert 'get("/runtime")' in api
    assert 'post("/runtime/start"' in api
    assert 'post("/runtime/pause"' in api
    assert "runtime-control" in css


def test_p3_does_not_present_stale_runtime_or_confirmation_as_model_work():
    app = _read(LIAISON / "LiaisonApp.tsx")
    api = _read(LIAISON / "api.ts")
    css = _read(LIAISON / "liaison.css")

    assert "Service disconnected" in app
    assert "connected={serverConnected}" in app
    assert "!connected || busy" in app
    assert "Connection lost. The last loaded state is still visible" in api
    assert ".runtime-control.is-disconnected" in css
    assert "state?.busy || optimisticInFlight" in app
    assert "state?.busy || busy" not in app


def test_p3_first_run_uses_fixed_victor_setup_without_seat_selection():
    app = _read(LIAISON / "LiaisonApp.tsx")
    setup = _read(ROOT / "src" / "setup" / "SetupApp.tsx")

    assert 'window.location.assign("/org/setup?next=liaison")' in app
    assert 'get("next") === "liaison"' in setup
    assert "liaisonSetup ? 0 : 24" in setup
    assert 'liaisonSetup ? "/org/liaison" : "/org/seat"' in setup


def test_p3_does_not_hardcode_a_demo_project_or_infer_product_readiness():
    app = _read(LIAISON / "LiaisonApp.tsx")
    assert "LanternForge" not in app
    assert "LanternScout" not in app
    assert "runtime?.pack?.product_name" in app
    ready = app.split("const isReadyMessage", 1)[1].split("};", 1)[0]
    assert "ready_to_try === true" in ready
    assert "completed|verified" not in ready


def test_p3_locks_the_page_and_gives_each_column_its_own_scroll_boundary():
    css = _read(LIAISON / "liaison.css")
    assert "height: 100dvh" in css
    assert "body { margin: 0; overflow: hidden" in css
    assert ".left-rail {" in css and "overflow-y: auto" in css
    assert ".conversation-scroll {" in css and "overflow-y: auto" in css
    assert ".details-scroll {" in css and "overflow-y: auto" in css
    assert ".app-main {" in css and "overflow: hidden" in css


def test_p3_thread_supports_clarification_execution_and_event_turns():
    app = _read(LIAISON / "LiaisonApp.tsx")
    api = _read(LIAISON / "api.ts")
    for marker in ("clarification", "requested", "assigned", "active", "verified",
                   "delegated", "executing", "completed", "reflection", "meeting", "event"):
        assert marker in app.lower() or marker in api.lower(), marker
    assert "Continue naturally or ask another question" in app
    assert "this is not a decision" in app
    assert "Confirm and execute" in app


def test_p3_has_slack_style_message_threads_with_a_scoped_side_panel():
    app = _read(LIAISON / "LiaisonApp.tsx")
    api = _read(LIAISON / "api.ts")
    css = _read(LIAISON / "liaison.css")
    for marker in ("Reply in thread", "ThreadPanel", "threadRootId", "thread_id",
                   "reply_to", "conversation_threads", "thread-summary",
                   "thread-panel-scroll", "Reply…"):
        assert marker in app or marker in api or marker in css, marker
    assert "mainMessages = messages.filter((message) => !message.thread_id)" in app
    assert "threadMessages = messages.filter" in app
    assert "threadDrafts = drafts.filter" in app
    assert "the main conversation stays uncluttered" in app


def test_p3_voice_input_persists_across_silence_until_explicit_stop_and_never_auto_sends():
    app = _read(LIAISON / "LiaisonApp.tsx")
    css = _read(LIAISON / "liaison.css")
    composer = app.split("function Composer", 1)[1].split("function ThreadPanel", 1)[0]
    assert "SpeechRecognition" in composer
    assert "webkitSpeechRecognition" in composer
    assert "recognition.continuous = true" in composer
    assert "recognition.interimResults = true" in composer
    assert "keepListeningRef.current" in composer
    assert 'reason === "no-speech"' in composer
    assert 'reason === "no-speech" || reason === "speech-not-recognized" || reason === "aborted"' in composer
    assert "startCycleRef.current();" in composer
    assert "scheduleRecognitionRestart" in composer
    assert "Voice recognition interrupted" in composer
    assert "reconnecting" in composer
    assert "pauses will not stop recording" in composer
    assert "if (event.repeat) return;" in composer
    assert "if (keepListeningRef.current)" in composer
    assert "stopVoice();\n      return;" in composer
    assert "if (listening) return;\n    onSend();" in composer
    assert "disabled={!value.trim() || busy || listening}" in composer
    assert "Microphone permission was denied" in composer
    assert "press Enter once to stop and finish the transcript" in composer
    assert "press Enter again after recording stops to send" in composer
    assert "onChange(" in composer
    assert "onSend();" not in composer.split("recognition.onresult", 1)[1].split(
        "recognition.onerror", 1)[0]
    assert ".voice-button.listening" in css


def test_every_p3_human_instruction_is_model_routed_before_special_handling():
    source = _read(Path(__file__).resolve().parents[2] /
                   "environments" / "org_env" / "human" / "liaison.py")
    router = source.split("def _route_human_instruction", 1)[1].split(
        "def _record_local_exchange", 1)[0]
    ask = source.split("def ask(self, token: str, text: str", 1)[1].split(
        "def prepare", 1)[0]
    assert "worker.llm.generate_json" in router
    assert "_LIAISON_ROUTE_SCHEMA" in router
    assert "_route_human_instruction" in ask
    assert "_handle_model_routed_instruction" in ask
    assert "_looks_like_meeting_response" not in source
    assert "_meeting_context_question" not in source
    assert "_message_preference_intent" not in source
    assert "re.search" not in router

    compiler_source = _read(Path(__file__).resolve().parents[2] /
                            "environments" / "org_env" / "human" /
                            "working_agent.py")
    compiler = compiler_source.split("def _compiler_context", 1)[1].split(
        "def _run_tool", 1)[0]
    assert "re." not in compiler
    assert "_ground_compiled_params" not in compiler_source
    assert "no downstream text parser will infer an ID" in compiler
    assert "The word 'report' does NOT authorize create_doc" in compiler


def test_clarification_is_not_rendered_or_counted_as_a_human_decision():
    app = _read(LIAISON / "LiaisonApp.tsx")
    decision_classifier = app.split("const isDecisionEvent", 1)[1].split("};", 1)[0]
    clarification_turn = app.split('if (kind === "clarification"', 1)[1].split(";\n", 1)[0]
    assert 'message.kind === "clarification"' not in decision_classifier
    assert 'kind="clarification"' in clarification_turn
    assert "Human judgment required" not in clarification_turn
    assert "More context would help" in clarification_turn


def test_p3_collapses_adjacent_organization_activity_into_a_digest():
    app = _read(LIAISON / "LiaisonApp.tsx")
    css = _read(LIAISON / "liaison.css")
    # Event-like turns share one assistant frame, but each item keeps its own
    # summary and disclosure controls instead of becoming an opaque summary.
    assert "groupConversation" in app
    assert "isOrganizationEvent" in app
    assert 'kind: "events"' in app
    assert "OrganizationDigest" in app
    assert "DigestItem" in app
    assert "EventTurn" not in app
    assert "digest-item" in app
    assert "message.summary || message.text" in app
    assert "execution-status" in app
    assert "message.tick" in app
    assert "t{message.tick}" in app
    assert "tick?: number" in _read(LIAISON / "api.ts")
    assert ".digest-tick" in css
    assert "Show details" in app and "Show trace" in app
    # Human and clarification turns remain ordinary blocks and therefore act
    # as explicit grouping boundaries.
    assert 'kind: "turn"' in app
    assert 'message.role === "human"' in app
    assert 'kind === "clarification"' in app
    assert "DraftTurn" in app
    assert "pending_actions" in app
    assert ".digest-card" in css
    assert ".digest-list" in css
    assert ".digest-item" in css
    assert ".assistant-turn.organization-digest" in css
    # Explicit organization disclosure must also show the useful summary and
    # organizational time, not fall back to raw event family + actor labels.
    assert "event.summary" in app
    assert "event.tick" in app


def test_p3_consequential_requests_use_backend_pending_draft_lifecycle():
    app = _read(LIAISON / "LiaisonApp.tsx")
    api = _read(LIAISON / "api.ts")
    assert "pending_actions" in app
    assert "Confirm and execute" in app
    assert ">Discard<" in app
    assert "draft.action_type" in app
    assert "draft.params" in app
    assert "draft.display_params || draft.params" in app
    assert "ObjectFacts" in app
    assert 'post("/confirm"' in api
    assert 'post("/discard"' in api
    assert "draft_id" in api


def test_p3_keeps_the_human_turn_visible_while_the_model_is_routing_it():
    app = _read(LIAISON / "LiaisonApp.tsx")
    css = _read(LIAISON / "liaison.css")
    for marker in (
        "OptimisticHumanMessage", "newOptimisticHumanMessage",
        "mergeOptimisticMessages", "optimisticHumanMessages",
        'delivery_status: "sending"', 'delivery_status: "failed"',
        "Not sent · text preserved", "Your text is preserved above",
    ):
        assert marker in app, marker
    assert app.index("setOptimisticHumanMessages((current) => [...current, optimistic])") < app.index(
        "await liaisonApi.ask(token, request, submittedContext")
    ask_block = app.split("const ask = async () => {", 1)[1].split(
        "const askThread = async () => {", 1)[0]
    assert ask_block.index('setView("conversation")') < ask_block.index(
        "setOptimisticHumanMessages((current) => [...current, optimistic])")
    assert "matchedServerIndexes" in app
    assert "serverAt >= optimisticAt - 1" in app
    assert ".human-delivery" in css
    assert ".human-delivery.failed" in css


def test_p3_projects_real_context_progress_without_model_reasoning():
    app = _read(LIAISON / "LiaisonApp.tsx")
    api = _read(LIAISON / "api.ts")
    css = _read(LIAISON / "liaison.css")
    for marker in (
        "P3WorkingState", "state?.working", "WorkingTurn", "elapsed_seconds",
        "working?.summary", "working?.phase", "working?.step",
        "working?.max_steps", "working?.tools_completed",
        "working?.unique_sources", "working?.duplicate_calls_blocked",
    ):
        assert marker in app or marker in api, marker
    assert "working?: P3WorkingState" in api
    assert ".working-progress" in css
    assert "model thought" not in app.lower()


def test_p3_meeting_plan_is_one_semantic_confirmation_not_component_actions():
    app = _read(LIAISON / "LiaisonApp.tsx")
    api = _read(LIAISON / "api.ts")
    css = _read(LIAISON / "liaison.css")
    for marker in ("P3MeetingPlan", "meeting_plans", "MeetingPlanTurn", "one confirmation",
                   "confirmMeetingPlan", "discardMeetingPlan", "technicalMeetingActions",
                   "send_message", "skip_meeting"):
        assert marker in app or marker in api, marker
    assert 'post("/confirm-meeting-plan"' in api
    assert 'post("/discard-meeting-plan"' in api
    assert ".meeting-plan-card" in css


def test_p3_human_office_execution_job_is_state_backed_and_controllable():
    app = _read(LIAISON / "LiaisonApp.tsx")
    api = _read(LIAISON / "api.ts")
    css = _read(LIAISON / "liaison.css")
    for marker in ("P3ExecutionJob", "execution_jobs", "executionJobsOf",
                   "ExecutionJobTurn", "ExecutionThread", "ExecutionDetails", "goal",
                   "workers", "assignment", "timeline", "final_report", "Start execution team",
                   "model_call_started_at", "model_call_phase", "Model call",
                   "Cancel execution", "startExecution", "cancelExecution",
                   "P3ExecutionAgent", "execution_agents", "executionAgentsOf",
                   "PersistentExecutionAgents", "Your execution department"):
        assert marker in app or marker in api, marker
    assert 'post("/execution/start", { token, ...(jobId ? { job_id: jobId } : {}) })' in api
    assert 'post("/execution/cancel", { token, ...(jobId ? { job_id: jobId } : {}) })' in api
    assert "pending_confirmation" in app
    assert "const executionCanStart = (job: P3ExecutionJob) => job.startable === true" in app
    assert "start_block_reason" in app
    assert "job_id" in app
    assert "executionJob &&" in app
    assert "executionJob={executionJob}" in app
    assert "showExecutionJob" in app
    assert "executionWorkersOf" in app
    assert "executionTimelineOf" in app
    assert "execution-job-card" in css
    assert "execution-thread" in css
    assert "execution-final-report" in css
    assert "persistent-agent-card" in css
    # The browser must not manufacture a demo job when the state has no job.
    assert 'job && typeof job === "object"' in app
    assert "execution_plans" not in app
    assert "execution_plans" not in api


def test_p3_renders_all_execution_jobs_and_focuses_latest_active_or_recent():
    app = _read(LIAISON / "LiaisonApp.tsx")
    api = _read(LIAISON / "api.ts")
    assert "execution_jobs?: P3ExecutionJob[]" in api
    assert "const executionJobs = executionJobsOf(state)" in app
    assert "const executionJob = executionFocusOf(executionJobs)" in app
    assert "...executionJobs.map" in app
    assert "executionJobs.map((job" in app
    assert "job.job_id" in app
    # Job order comes from the runtime creation order, with timestamped
    # timeline entries used when the runtime exposes them.
    assert "executionTimeOf(left.job.created_at" in app
    assert "return left.index - right.index" in app


def test_p3_shows_persistent_workers_as_active_or_dormant_without_rewriting_jobs():
    app = _read(LIAISON / "LiaisonApp.tsx")
    api = _read(LIAISON / "api.ts")
    assert "execution_agents?: P3ExecutionAgent[]" in api
    assert 'status: "inactive" | "active" | string' in api
    assert "activation_count" in api
    assert "lifetime_model_calls" in api
    assert 'status === "inactive" ? "dormant" : status' in app
    assert "Inactive agents remain available with provenance-marked prior-run context" in app
    assert "rechecks live state before acting" in app
    assert 'worker.reused ? `persistent agent · activation ${worker.activation_number}`' in app


def test_p3_only_projects_a_job_into_its_matching_slack_thread():
    app = _read(LIAISON / "LiaisonApp.tsx")
    assert "const matchingExecutionJobs = executionJobs.filter((job) => Boolean(job.thread_id) && job.thread_id === root.id);" in app
    assert "matchingExecutionJobs.map((job)" in app
    assert "executionJob && (!executionThreadId" not in app


def test_p3_surfaces_the_real_backend_request_queue_without_faking_progress():
    app = _read(LIAISON / "LiaisonApp.tsx")
    api = _read(LIAISON / "api.ts")
    assert "queued_requests" in api
    assert "const queuedRequests = Number(state?.working?.queued_requests || 0)" in app
    assert "queuedRequests={queuedRequests}" in app
    assert "queuedRequests={Number(workingState?.queued_requests || 0)}" in app
    assert "request${queuedRequests === 1 ? \"\" : \"s\"} queued" in app
    assert "process them in order" in app
    # A local browser queue must not invent completion/status; the server
    # remains the source of truth for queue depth and working state.
    assert "state?.working?.queued_requests" in app
    assert "window.setTimeout(pollState, 1200)" in app
    assert "window.setTimeout(pollRuntime, 1200)" in app
    assert "setInterval(load, 1200)" not in app


def test_p3_activity_and_navigation_counts_include_real_execution_jobs():
    app = _read(LIAISON / "LiaisonApp.tsx")
    assert "const activityRows = [" in app
    assert "(view === \"activity\" ? activityRows : conversationRows)" in app
    assert "mainMessages.length + executionJobs.length" in app
    assert "activityMessages.length + executionJobs.length" in app
    assert "executionDecisionCount" in app
    assert "decisionMessages.length + mainDrafts.length + meetingPlans.length + executionDecisionCount" in app


def test_p3_consequential_ui_entrypoints_map_to_liaison_contracts():
    app = _read(LIAISON / "LiaisonApp.tsx")
    api = _read(LIAISON / "api.ts")
    for marker in (
        "liaisonApi.ask(token, request, submittedContext", "liaisonApi.confirm(token, draftId)",
        "liaisonApi.discard(token, draftId)", "liaisonApi.confirmMeetingPlan(token, planId)",
        "liaisonApi.discardMeetingPlan(token, planId)", "liaisonApi.startExecution(token, jobId)",
        "liaisonApi.cancelExecution(token, jobId)", "liaisonApi.runtimePause()",
        "liaisonApi.runtimeStart()", "liaisonApi.evidence(token, ref)",
        "liaisonApi.trace(token, traceRef)", "reply_to", "thread_id",
    ):
        assert marker in app, marker
    for endpoint in (
        'post("/confirm"', 'post("/discard"',
        'post("/confirm-meeting-plan"', 'post("/discard-meeting-plan"',
        'post("/execution/start"', 'post("/execution/cancel"',
        'post("/runtime/start"', 'post("/runtime/pause"',
    ):
        assert endpoint in api, endpoint
    # Work requests such as task/PR/proposal/review/test/merge stay on the
    # model-routed composer and become typed pending_actions; there are no
    # browser-only operation buttons or fabricated action results.
    assert "pending_actions" in app
    assert "draft.action_type" in app
    assert "Confirm and execute" in app
    assert "ObjectFacts" in app


def test_p3_meeting_rsvp_relay_and_return_focus_are_state_backed():
    app = _read(LIAISON / "LiaisonApp.tsx")
    api = _read(LIAISON / "api.ts")
    for marker in ("decision", "viewpoint", "return_focus", "Confirm and route",
                   "MeetingInvitationTurn", "meeting_plans", "meeting_invitations"):
        assert marker in app or marker in api, marker
    assert "settleMeetingPlan(plan.plan_id, \"confirm\")" in app
    assert "settleMeetingPlan(plan.plan_id, \"discard\")" in app
    assert 'post("/confirm-meeting-plan", { token, plan_id: planId })' in api
    assert 'post("/discard-meeting-plan", { token, plan_id: planId })' in api


def test_p3_secretary_messages_render_safe_markdown_and_code_blocks():
    app = _read(LIAISON / "LiaisonApp.tsx")
    markdown = _read(LIAISON / "MarkdownMessage.tsx")
    css = _read(LIAISON / "liaison.css")
    assert 'import { MarkdownMessage } from "./MarkdownMessage"' in app
    assert '<MarkdownMessage text={message.text}' in app
    assert '<MarkdownMessage text={message.clarification || message.text}' in app
    assert 'root.role === "human" ? <p>{root.text}</p> : <MarkdownMessage' in app
    for marker in ("<pre", "<code", "<h1", "<h2", "<h3", "<ul", "<ol",
                   "<blockquote", 'target="_blank"', 'rel="noreferrer"'):
        assert marker in markdown, marker
    assert "dangerouslySetInnerHTML" not in markdown
    assert "dangerouslySetInnerHTML" not in app
    assert ".markdown-message pre" in css
    assert ".markdown-message code" in css


def test_p3_skips_seat_selection_and_opens_only_the_fixed_victor_session():
    p2 = _read(ROOT / "src" / "seat" / "SeatApp.tsx")
    p3 = _read(LIAISON / "LiaisonApp.tsx")
    api = _read(LIAISON / "api.ts")
    assert 'const TOKEN_KEY = "socio.seat.token"' in p3
    assert "Victor's seat" in p3
    assert "connectVictor" in p3
    assert "PerspectivePicker" not in p3
    assert "Choose the member perspective" not in p3
    assert 'post("/session"' in api
    assert 'post("/claim"' not in api
    assert 'get("/members"' not in api
    assert 'href="/org/liaison"' in p2
    assert 'aria-label="Experience version"' in p3
    assert 'href="/org/seat"' in p3


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
