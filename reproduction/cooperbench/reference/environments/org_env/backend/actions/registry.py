"""OrgEnv action registry — DomainAction descriptors (DESIGN env_org §33.4/§61-§69,
O1 §5).

Declares the full action vocabulary (type + category + cost hints). Execution
handlers live in runtime_adapter/execution.py (the OrgExecutionAdapter). Costs
here are *hints* the FeatureExtractor / ExecutionAdapter refine per-context.
"""
from __future__ import annotations

from typing import Dict, List

from agent_sdk.lived.domain.interfaces import DomainAction

CAT_WORK = "work"
CAT_COMM = "comm"
CAT_MEETING = "meeting"
CAT_REPO = "repo"
CAT_SANDBOX = "sandbox"
CAT_SEARCH = "search"
CAT_DOC = "doc"
CAT_ARTIFACT = "artifact"
CAT_PROTOCOL = "protocol"
CAT_TIME = "time"
CAT_PAYROLL = "payroll"
CAT_BRIDGE = "bridge"   # internal <-> external community
CAT_GOVERNANCE = "governance"   # approve / reject / request-changes on proposals
CAT_RELEASE = "release"   # release candidate / gate / publish lifecycle

# action_type -> category (O1 §5.1-§5.10). Legacy aliases kept for back-compat.
ORG_ACTION_CATEGORIES: Dict[str, str] = {
    # §5.1 work
    "pick_task": CAT_WORK, "work_on_task": CAT_WORK, "create_issue": CAT_WORK,
    "update_task_status": CAT_WORK, "assign_task_owner": CAT_WORK, "handoff_task": CAT_WORK,
    "inspect_task_board": CAT_WORK, "debug_code": CAT_WORK, "review_result": CAT_WORK,
    "update_tracker": CAT_WORK, "reply_customer": CAT_WORK, "inspect_repo_module": CAT_WORK,
    # §5.2 communication
    "send_message": CAT_COMM, "reply_thread": CAT_COMM, "mention_agent": CAT_COMM,
    "share_doc": CAT_COMM, "share_file": CAT_COMM, "share_json_result": CAT_COMM,
    "share_experiment_result": CAT_COMM, "share_external_post": CAT_COMM,
    "share_search_result": CAT_COMM, "acknowledge_message": CAT_COMM,
    "ask_for_clarification": CAT_COMM, "ask_for_help": CAT_COMM, "send_async_update": CAT_COMM,
    "defer_until_work_hours": CAT_COMM, "request_after_hours_help": CAT_COMM,
    "escalate_incident": CAT_COMM,
    # §5.3 meeting
    "schedule_meeting": CAT_MEETING, "attend_meeting": CAT_MEETING, "skip_meeting": CAT_MEETING,
    "share_object_in_meeting": CAT_MEETING, "present_report": CAT_MEETING,
    "discuss_issue": CAT_MEETING, "propose_decision": CAT_MEETING,
    "record_meeting_notes": CAT_MEETING, "assign_action_item": CAT_MEETING,
    "close_meeting": CAT_MEETING, "summarize_decision": CAT_MEETING,
    # §5.4 repo
    "create_branch": CAT_REPO, "edit_file": CAT_REPO, "commit_changes": CAT_REPO,
    "open_pr": CAT_REPO, "review_pr": CAT_REPO, "request_changes": CAT_REPO,
    "approve_pr": CAT_REPO, "merge_pr": CAT_REPO, "revert_commit": CAT_REPO,
    "resolve_conflict": CAT_REPO, "sync_branch": CAT_REPO,
    # v5 repo workflow (patch -> commit -> push -> PR -> CI -> merge)
    "commit_patch": CAT_REPO, "push_commit": CAT_REPO, "run_ci": CAT_REPO,
    # Verification affordance: run the substrate's OWN public test suite (ships in the
    # agent-visible starter repo; hidden oracles stay evaluator-only). Without this the
    # org patches blindly and can never tell a fix worked.
    "run_public_tests": CAT_REPO, "verify_joint_delivery": CAT_REPO,
    "sync_actor_workspace": CAT_REPO,
    "verify_pr_integration": CAT_REPO,
    "review_integration_failure": CAT_REPO,
    "repair_accepted_probe_plan": CAT_REPO,
    # v5 release lifecycle
    "create_release_candidate": CAT_RELEASE, "run_launch_readiness_check": CAT_RELEASE,
    "approve_release_candidate": CAT_RELEASE, "block_release_candidate": CAT_RELEASE,
    "publish_product_release": CAT_RELEASE, "collect_post_launch_feedback": CAT_RELEASE,
    # §5.5 sandbox / experiment
    "run_script": CAT_SANDBOX, "run_experiment": CAT_SANDBOX, "run_cheap_pilot": CAT_SANDBOX,
    "run_paper_baseline": CAT_SANDBOX, "install_package": CAT_SANDBOX, "load_dataset": CAT_SANDBOX,
    "debug_failure": CAT_SANDBOX, "save_result": CAT_SANDBOX, "export_result_to_tracker": CAT_SANDBOX,
    "share_sandbox_output": CAT_SANDBOX, "clean_sandbox": CAT_SANDBOX,
    # §5.6 search
    "internal_search": CAT_SEARCH, "repo_search": CAT_SEARCH, "sandbox_search": CAT_SEARCH,
    "external_community_search": CAT_SEARCH, "frozen_web_search": CAT_SEARCH,
    "open_search_result": CAT_SEARCH, "save_search_result": CAT_SEARCH,
    # §5.7 document / artifact
    "create_doc": CAT_DOC, "edit_doc": CAT_DOC, "comment_doc": CAT_DOC, "review_doc": CAT_DOC,
    "approve_doc": CAT_DOC, "request_doc_changes": CAT_DOC, "link_doc_to_task": CAT_DOC,
    "link_doc_to_experiment": CAT_DOC, "publish_doc": CAT_DOC, "archive_doc": CAT_DOC,
    "create_experiment_tracker": CAT_ARTIFACT, "update_experiment_tracker": CAT_ARTIFACT,
    "create_cost_ledger": CAT_ARTIFACT, "add_cost_ledger_entry": CAT_ARTIFACT,
    "create_review_checklist": CAT_ARTIFACT, "use_review_checklist": CAT_ARTIFACT,
    "create_customer_triage_sheet": CAT_ARTIFACT, "create_claim_evidence_table": CAT_ARTIFACT,
    "add_claim_evidence_row": CAT_ARTIFACT,
    "create_workflow_artifact": CAT_ARTIFACT, "use_workflow_artifact": CAT_ARTIFACT,
    "revise_workflow_artifact": CAT_ARTIFACT, "abandon_artifact": CAT_ARTIFACT,
    # §5.8 protocol
    "propose_protocol": CAT_PROTOCOL, "support_protocol": CAT_PROTOCOL,
    "oppose_protocol": CAT_PROTOCOL, "follow_protocol": CAT_PROTOCOL,
    "violate_protocol": CAT_PROTOCOL, "enforce_protocol": CAT_PROTOCOL,
    "amend_protocol": CAT_PROTOCOL,
    # §5.9 time / availability
    "rest_offline": CAT_TIME, "sleep": CAT_TIME, "check_messages": CAT_TIME,
    "work_overtime": CAT_TIME, "weekend_work": CAT_TIME, "set_availability_status": CAT_TIME,
    "handoff_before_offline": CAT_TIME,
    # §5.10 payroll / hiring / retention
    "run_payroll": CAT_PAYROLL, "announce_payroll_delay": CAT_PAYROLL,
    "ask_about_payroll": CAT_PAYROLL, "create_runway_update": CAT_PAYROLL,
    "read_job_post": CAT_PAYROLL, "consider_external_offer": CAT_PAYROLL,
    "dm_recruiter": CAT_PAYROLL, "post_job": CAT_PAYROLL, "search_candidates": CAT_PAYROLL,
    "ask_network_for_referral": CAT_PAYROLL, "dm_candidate": CAT_PAYROLL,
    "evaluate_candidate": CAT_PAYROLL, "interview_candidate": CAT_PAYROLL,
    "make_offer": CAT_PAYROLL, "onboard_new_member": CAT_PAYROLL, "resign": CAT_PAYROLL,
    # bridge to external community (§44 / §66)
    "read_knowledge": CAT_BRIDGE,
    "read_feed": CAT_BRIDGE, "search_posts": CAT_BRIDGE, "read_external_doc": CAT_BRIDGE,
    "ask_external_expert": CAT_BRIDGE, "share_signal_to_team": CAT_BRIDGE,
    "post_company_update": CAT_BRIDGE, "monitor_customer_feedback": CAT_BRIDGE,
    "dm_external_contact": CAT_BRIDGE, "respond_to_public_comment": CAT_BRIDGE,
    "read_external_feed": CAT_BRIDGE,
    # --- O1.6 communication split: micro/auxiliary speech (0-block) --------
    "quick_check_messages": CAT_COMM, "read_mention": CAT_COMM, "reply_thread_short": CAT_COMM,
    # --- O1.6 communication: formal/blocking information processing --------
    "inbox_triage": CAT_COMM, "read_long_thread": CAT_COMM, "review_thread_history": CAT_COMM,
    "process_customer_feedback_queue": CAT_COMM, "prepare_customer_feedback_summary": CAT_COMM,
    "summarize_external_discussion": CAT_COMM,
    # --- O1.7 feedback / speech acts (§18) --------------------------------
    "ask_for_review": CAT_COMM, "ask_for_evidence": CAT_COMM, "request_reproduction": CAT_COMM,
    "promise_work": CAT_COMM, "challenge_result": CAT_COMM, "warn_about_risk": CAT_COMM,
    "approve_with_note": CAT_COMM, "suggest_rewrite": CAT_COMM, "explain_delay": CAT_COMM,
    "apologize": CAT_COMM, "deescalate": CAT_COMM, "push_team": CAT_COMM,
    "share_result": CAT_COMM, "coordinate_followup": CAT_COMM, "defend_demo_progress": CAT_COMM,
    "commit_to_direction": CAT_COMM, "share_external_signal": CAT_COMM,
    # --- O1.7 primary text / review ---------------------------------------
    "formal_pr_review": CAT_REPO,
    # background helpers
    "ci_test": CAT_SANDBOX,
    # --- product substrate actions (messy research-agent prototype) -------
    "edit_repo_file": CAT_REPO, "open_issue": CAT_WORK, "close_issue": CAT_WORK,
    "create_eval_stub": CAT_SANDBOX, "run_eval_stub": CAT_SANDBOX,
    "create_report_template": CAT_ARTIFACT, "update_claim_tracker": CAT_ARTIFACT,
    "update_source_tracker": CAT_ARTIFACT, "create_product_demo": CAT_DOC,
    "create_onboarding_doc": CAT_DOC, "write_design_note": CAT_DOC,
    "propose_product_direction": CAT_DOC, "audit_readme_claims": CAT_DOC,
    "create_report_quality_checklist": CAT_ARTIFACT,
    # v11 dogfooding: an agent actually RUNS/experiences the product they built
    "dogfood_product": CAT_SANDBOX,
    # invoke an adopted composed tool (runs its required actions)
    "use_tool": CAT_ARTIFACT,
    # governance: designated approvers act explicitly on proposals (§13.x)
    "approve_proposal": CAT_GOVERNANCE, "reject_proposal": CAT_GOVERNANCE,
    "request_proposal_changes": CAT_GOVERNANCE,
}

# action_type -> what it actually does, for an agent that has to choose from a
# menu. A category alone is not semantics: a founder told to fix a sorting crash
# read "work_on_task (work)" as the way to fix it and picked it 159 times in a
# 336-tick run, because nothing said that action only records effort. Say what
# each one changes, and say plainly when it changes nothing in the product.
ORG_ACTION_DESCRIPTIONS: Dict[str, str] = {
    # --- work: task bookkeeping, NOT product change ---------------------
    "pick_task": "Take a task off the board and make it yours. Bookkeeping only.",
    "work_on_task": (
        "Log effort against a task. Records that you spent time and nothing "
        "else: it does NOT edit code, produce an artifact, or move the task "
        "towards done. Repeating it cannot complete anything. To change the "
        "product use edit_repo_file."
    ),
    "work_overtime": "Log effort outside work hours. Same effect as work_on_task.",
    "update_task_status": "Set a task's status field. Bookkeeping only.",
    "assign_task_owner": "Give a task an owner. Bookkeeping only.",
    "handoff_task": "Transfer a task to another member.",
    "inspect_task_board": "Read the board. No change.",
    "open_issue": "File a new issue describing a defect.",
    "close_issue": "Close an issue you believe is resolved.",
    "debug_code": "Investigate a failure. Logs effort; does not edit code.",
    "create_issue": "File a new issue describing a defect you have found.",
    "inspect_repo_module": "Read a repository module. No change.",
    "review_result": "Read someone's recorded result and judge it.",
    "update_tracker": "Write a row into a shared tracker.",
    "reply_customer": "Answer a customer ticket.",
    # --- repo: the actions that actually change the product -------------
    "edit_repo_file": (
        "Write a change into a repository file. This is the only way to modify "
        "the product; the frozen test suite scores the code, not the effort."
    ),
    "commit_patch": "Commit staged edits so they can be reviewed and merged.",
    "open_pr": "Open a pull request so a change can be reviewed and merged.",
    "review_pr": "Review someone's pull request; counts as third-party review.",
    "formal_pr_review": "Write a substantive review verdict on a pull request.",
    "request_changes": "Ask a pull request's author for changes before merge.",
    "approve_pr": "Approve a pull request for merge.",
    "merge_pr": "Merge an approved pull request into mainline.",
    "revert_commit": "Undo a merged change.",
    "run_ci": "Run continuous integration on a branch.",
    "create_branch": "Start a new branch off mainline.",
    "edit_file": "Write a change into a file on your branch.",
    "commit_changes": "Commit the edits sitting on your branch.",
    "push_commit": "Push your commits to the shared remote.",
    "sync_branch": "Pull mainline into your branch.",
    "resolve_conflict": "Reconcile a branch that mainline has moved under.",
    "run_public_tests": (
        "Run the repository's own visible test suite and see pass/fail. The "
        "only way to find out whether an edit worked before it is graded."
    ),
    "verify_joint_delivery": (
        "Independently run the declared public checks on the integrated mainline "
        "and attest its exact digest before the two-person unit freezes delivery."
    ),
    "sync_actor_workspace": (
        "Explicitly import the current committed mainline into your own clean "
        "workspace; peer uncommitted edits are not included."
    ),
    "verify_pr_integration": "Re-run the accepted feature checks on the exact proposed integrated tree before merging.",
    "review_integration_failure": "Judge a reached integration failure against the public request before requesting a product repair.",
    "repair_accepted_probe_plan": "Correct the specific peer-authored checks judged defective, preserving all unaffected definitions and merged code.",
    # --- sandbox / evaluation -------------------------------------------
    "run_experiment": "Run an experiment in your sandbox.",
    "run_cheap_pilot": "Run a small, cheap probe before committing to a direction.",
    "run_eval_stub": "Execute an evaluation stub and record its result.",
    "create_eval_stub": "Create a reusable evaluation stub.",
    "debug_failure": "Reproduce and diagnose a concrete failure.",
    "dogfood_product": "Use the product yourself and record what you hit.",
    "run_script": "Run a script in your sandbox and see its output.",
    "run_paper_baseline": "Reproduce a published baseline for comparison.",
    "install_package": "Add a dependency to your sandbox.",
    "load_dataset": "Load a dataset into your sandbox.",
    "save_result": "Persist a sandbox result so it can be referenced later.",
    "export_result_to_tracker": "Copy a sandbox result into the shared tracker.",
    "share_sandbox_output": "Make a sandbox result visible to the team; until "
                            "you do, it exists only for you.",
    "clean_sandbox": "Discard your sandbox state.",
    "ci_test": "Run the automated checks over a change.",
    # --- documents and artifacts ----------------------------------------
    "create_doc": "Write a shared document.",
    "review_doc": "Review a shared document; counts as third-party review.",
    "write_design_note": "Record a design decision and its reasoning.",
    "create_onboarding_doc": "Write onboarding material for the team.",
    "audit_readme_claims": "Check the README's claims against what the code does.",
    "create_experiment_tracker": "Create the shared tracker experiments log into.",
    "create_report_quality_checklist": "Create a checklist reports are held to.",
    "use_workflow_artifact": "Apply an existing shared artifact to current work.",
    "update_source_tracker": "Record where a claim's supporting source came from.",
    "update_claim_tracker": "Record a claim and the evidence standing behind it.",
    "create_report_template": "Create the template reports are written against.",
    "create_product_demo": "Build a demo of what the product currently does.",
    "propose_product_direction": "Propose where the product should go next.",
    "edit_doc": "Revise a shared document.",
    "comment_doc": "Leave a comment on a shared document.",
    "approve_doc": "Sign off on a shared document.",
    "request_doc_changes": "Send a document back to its author with changes.",
    "publish_doc": "Make a document visible outside the team.",
    "archive_doc": "Retire a document from active use.",
    "link_doc_to_task": "Attach a document to a task.",
    "link_doc_to_experiment": "Attach a document to an experiment.",
    "update_experiment_tracker": "Log an experiment's result in the tracker.",
    "create_cost_ledger": "Create the ledger spend is recorded in.",
    "add_cost_ledger_entry": "Record one spend against the ledger.",
    "create_review_checklist": "Create a checklist reviews are held to.",
    "use_review_checklist": "Review something against the adopted checklist.",
    "create_customer_triage_sheet": "Create the sheet customer reports are sorted in.",
    "create_claim_evidence_table": "Create the table claims and their evidence live in.",
    "add_claim_evidence_row": "Record one claim and the evidence for it.",
    "create_workflow_artifact": "Create a shared artifact others can reuse.",
    "revise_workflow_artifact": "Change a shared artifact others already use.",
    "abandon_artifact": "Retire a shared artifact.",
    "use_tool": "Invoke an adopted composed tool; it runs the actions it is made of.",
    # --- protocol: how the organization binds itself --------------------
    "propose_protocol": "Propose a rule the organization should follow.",
    "support_protocol": "Endorse a proposed rule; adoption needs enough endorsers.",
    "oppose_protocol": "Object to a proposed rule.",
    "follow_protocol": "Apply an adopted rule to the work in front of you.",
    "enforce_protocol": "Hold someone else's work to an adopted rule.",
    "amend_protocol": "Change an adopted rule.",
    "violate_protocol": "Act against an adopted rule; the breach is recorded.",
    "approve_proposal": "Approve a proposal you are a designated approver for.",
    "reject_proposal": "Reject a proposal you are a designated approver for.",
    "request_proposal_changes": "Send a proposal back for revision.",
    # --- release ---------------------------------------------------------
    "create_release_candidate": "Cut a release candidate from mainline.",
    "run_launch_readiness_check": "Check a candidate against release criteria.",
    "approve_release_candidate": "Approve a candidate for release.",
    "block_release_candidate": "Block a candidate you judge unready.",
    "publish_product_release": "Publish a release to users.",
    "collect_post_launch_feedback": "Gather what users hit after a release.",
    # --- meetings ---------------------------------------------------------
    "schedule_meeting": "Put a meeting on the calendar; it needs an agenda.",
    "attend_meeting": "Attend a scheduled meeting.",
    "skip_meeting": "Decline a meeting you were expected at.",
    "discuss_issue": "Raise an issue with the people in the room.",
    "present_report": "Present a report to the people in the room.",
    "propose_decision": "Put a decision to the room.",
    "summarize_decision": "State what the room decided.",
    "record_meeting_notes": "Write down what was said, so it survives the meeting.",
    "assign_action_item": "Give a meeting outcome an owner; it becomes a task.",
    "close_meeting": "End the meeting.",
    "share_object_in_meeting": "Show the room an object only you could see.",
    # --- communication ----------------------------------------------------
    "send_message": "Message a channel or a person.",
    "send_async_update": "Post a status update others read later.",
    "reply_thread": "Reply in an existing thread.",
    "ask_for_help": "Ask a specific person for help with a specific thing.",
    "ask_for_review": "Ask someone to review your work.",
    "promise_work": "Commit publicly to doing something; it is tracked.",
    "quick_check_messages": "Skim the inbox. No change.",
    "mention_agent": "Address a specific person so they are notified.",
    "acknowledge_message": "Confirm you read a message.",
    "read_mention": "Read a message that named you.",
    "reply_thread_short": "Post a brief reply in a thread.",
    "read_long_thread": "Read a whole thread rather than its last message.",
    "review_thread_history": "Read back through a thread's history.",
    "inbox_triage": "Sort the inbox into what needs answering.",
    "ask_for_clarification": "Ask what someone actually meant.",
    "ask_for_evidence": "Ask what a claim rests on.",
    "request_reproduction": "Ask someone to reproduce a result you doubt.",
    "challenge_result": "Say publicly that you do not believe a result.",
    "suggest_rewrite": "Propose that someone redo their work differently.",
    "approve_with_note": "Agree, with a reservation on record.",
    "warn_about_risk": "Put a risk you see on record.",
    "explain_delay": "Say why something is late.",
    "apologize": "Take responsibility for something that went wrong.",
    "deescalate": "Take heat out of a conflict.",
    "push_team": "Press the team to move faster.",
    "defend_demo_progress": "Argue that progress is real when it is doubted.",
    "commit_to_direction": "State publicly which direction you are taking.",
    "coordinate_followup": "Agree with someone who does what next.",
    "escalate_incident": "Raise an incident to whoever can act on it.",
    "request_after_hours_help": "Ask for help outside working hours.",
    "share_result": "Make a result you hold visible to others.",
    "share_doc": "Make a document you hold visible to others.",
    "share_file": "Make a file you hold visible to others.",
    "share_json_result": "Make a structured result visible to others.",
    "share_experiment_result": "Make an experiment's result visible to others.",
    "share_search_result": "Make something you found visible to others.",
    "share_external_post": "Bring an outside post to the team's attention.",
    "share_external_signal": "Bring an outside signal to the team's attention.",
    "process_customer_feedback_queue": "Work through the queue of customer reports.",
    "prepare_customer_feedback_summary": "Condense customer reports into a summary.",
    "summarize_external_discussion": "Condense an outside discussion for the team.",
    # --- search: retrieval, and it returns references, not content --------
    "internal_search": "Search internal documents and history.",
    "repo_search": "Search the repository's commits and modules.",
    "sandbox_search": "Search your own sandbox results.",
    "external_community_search": "Search the external community.",
    "frozen_web_search": "Search the frozen web snapshot.",
    "open_search_result": "Open one result a search returned.",
    "save_search_result": "Keep a result so you can return to it.",
    # --- bridge to the outside --------------------------------------------
    "read_knowledge": ("Read one of the product's own knowledge files, the "
                       "published API contract among them. What it says becomes "
                       "visible to you."),
    "read_feed": "Read the external community feed.",
    "read_external_feed": "Read the external community feed.",
    "read_external_doc": "Read a document from outside the company.",
    "search_posts": "Search external posts.",
    "monitor_customer_feedback": "Read what users are reporting.",
    "ask_external_expert": "Ask someone outside the company.",
    "dm_external_contact": "Message someone outside the company privately.",
    "respond_to_public_comment": "Answer a public comment in the company's name.",
    "post_company_update": "Publish an update in the company's name.",
    "share_signal_to_team": "Bring an outside signal inside.",
    # --- time -------------------------------------------------------------
    "rest_offline": "Go offline and recover.",
    "sleep": "Sleep; recovers more than rest.",
    "defer_until_work_hours": "Postpone this until working hours.",
    "check_messages": "Look at the inbox. No change.",
    "weekend_work": "Work on a non-working day.",
    "set_availability_status": "Tell others whether you are reachable.",
    "handoff_before_offline": "Pass your open work on before going offline.",
    # --- payroll, hiring, retention ----------------------------------------
    "run_payroll": "Pay the team from the treasury.",
    "announce_payroll_delay": "Tell the team pay is late.",
    "ask_about_payroll": "Ask when you are getting paid.",
    "create_runway_update": "Publish how long the money lasts.",
    "read_job_post": "Read a job posting from another company.",
    "consider_external_offer": "Weigh leaving for an outside offer.",
    "dm_recruiter": "Reply to a recruiter privately.",
    "resign": "Leave the company.",
    "post_job": "Advertise an open role.",
    "search_candidates": "Look for people who could fill a role.",
    "ask_network_for_referral": "Ask your contacts for a referral.",
    "dm_candidate": "Reach out to a candidate privately.",
    "evaluate_candidate": "Judge a candidate against the role.",
    "interview_candidate": "Interview a candidate.",
    "make_offer": "Offer a candidate the role.",
    "onboard_new_member": "Bring a new hire into the team.",
}


def action_description(action_type: str) -> str:
    """What the action does, falling back to its category when undescribed."""

    described = ORG_ACTION_DESCRIPTIONS.get(action_type)
    if described:
        return described
    category = ORG_ACTION_CATEGORIES.get(action_type, "")
    return f"({category} action)" if category else ""


# Rough per-category cost hints (attention / fatigue / stress). Refined per-context.
CATEGORY_COST_HINT: Dict[str, Dict[str, float]] = {
    CAT_WORK: {"attention": 0.12, "fatigue": 0.05, "stress": 0.02},
    CAT_COMM: {"attention": 0.05, "fatigue": 0.01, "stress": 0.01},
    CAT_MEETING: {"attention": 0.10, "fatigue": 0.06, "stress": 0.03},
    CAT_REPO: {"attention": 0.12, "fatigue": 0.05, "stress": 0.03},
    CAT_SANDBOX: {"attention": 0.14, "fatigue": 0.06, "stress": 0.04},
    CAT_SEARCH: {"attention": 0.06, "fatigue": 0.02, "stress": 0.01},
    CAT_DOC: {"attention": 0.10, "fatigue": 0.04, "stress": 0.02},
    CAT_ARTIFACT: {"attention": 0.12, "fatigue": 0.05, "stress": 0.02},
    CAT_PROTOCOL: {"attention": 0.08, "fatigue": 0.03, "stress": 0.04},
    CAT_TIME: {"attention": -0.10, "fatigue": -0.08, "stress": -0.05},
    CAT_PAYROLL: {"attention": 0.08, "fatigue": 0.03, "stress": 0.05},
    CAT_BRIDGE: {"attention": 0.06, "fatigue": 0.02, "stress": 0.02},
    CAT_GOVERNANCE: {"attention": 0.06, "fatigue": 0.02, "stress": 0.03},
    CAT_RELEASE: {"attention": 0.10, "fatigue": 0.04, "stress": 0.05},
}


def make_action(action_type: str, **params) -> DomainAction:
    """Build a DomainAction descriptor with a category + cost hint."""
    cat = ORG_ACTION_CATEGORIES.get(action_type, "")
    return DomainAction(action_type=action_type, action_category=cat,
                        cost=dict(CATEGORY_COST_HINT.get(cat, {})), parameters=dict(params))


def registered_action_types() -> List[str]:
    return list(ORG_ACTION_CATEGORIES.keys())


def action_category(action_type: str) -> str:
    return ORG_ACTION_CATEGORIES.get(action_type, "")


__all__ = [
    "CAT_WORK", "CAT_COMM", "CAT_MEETING", "CAT_REPO", "CAT_SANDBOX", "CAT_SEARCH",
    "CAT_DOC", "CAT_ARTIFACT", "CAT_PROTOCOL", "CAT_TIME", "CAT_PAYROLL", "CAT_BRIDGE",
    "CAT_GOVERNANCE", "CAT_RELEASE",
    "ORG_ACTION_CATEGORIES", "ORG_ACTION_DESCRIPTIONS", "CATEGORY_COST_HINT",
    "make_action", "registered_action_types", "action_category",
    "action_description",
]
