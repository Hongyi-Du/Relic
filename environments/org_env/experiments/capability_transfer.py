"""Cross-repository capability transfer (protocol §54-§59, TRANSFER_ARMS).

A capability formed in a source repository is carried into a target repository,
and each arm is a COMPLETE target run from the frozen initial snapshot at
episode one. The source contributes three things and nothing else: the
capability bundle, the original roster's member state, and the provenance that
binds one to the other. The target's own B3 main run is the fresh-from-scratch
reference, so there is no blank arm.

The four arms are a 2x2 over what is inherited:

                      executable            removed / text-only
    retained roster   R_Exec                R_Removed
    fresh roster      F_Exec                F_Text

which isolates: what the capability objects contribute with the people held
constant (R_Exec vs R_Removed), what they contribute without the people
(F_Exec vs the reference), what the people contribute with the capabilities
held constant (R_Exec vs F_Exec), and whether the capability has to be
executable or whether matched prose is enough (F_Exec vs F_Text).

F_Text is the arm the design exists to make possible, and it is also the one
that can defeat itself: prose describing a rule is exactly the input an
organization needs to re-derive the rule. Capability compilation is therefore
frozen for the first TRANSFER_EVALUATION_WINDOW_EPISODES episodes, in the
reference too, so the window measures USING an inherited capability rather than
forming a new one.
"""
from __future__ import annotations

import copy
from hashlib import sha256
import json
import re
from typing import Any, Mapping, Sequence

from environments.org_env.experiments.capability_carriers import (
    CARRIER_KIND_DOCUMENT,
    CARRIER_KIND_PROTOCOL,
    canonical_capability,
)

BUNDLE_SCHEMA_VERSION_V1 = "org_capability_bundle_v1"
BUNDLE_SCHEMA_VERSION_V2 = "org_capability_bundle_v2"
# Keep the default export format pinned to v1.  V2 is an opt-in, curated
# transfer artifact with closed machine bindings; accepting it must not alter
# historic v1 bundle semantics.
BUNDLE_SCHEMA_VERSION = BUNDLE_SCHEMA_VERSION_V1
SUPPORTED_BUNDLE_SCHEMA_VERSIONS = (
    BUNDLE_SCHEMA_VERSION_V1,
    BUNDLE_SCHEMA_VERSION_V2,
)

CANONICAL_V1_SHA256 = "ce3c96cd2263c79e2a53a4969167f52a61814f32ab14dfa83d0a9e7fda44cff5"
CANONICAL_V1_GIT_COMMIT = "d7353db891b2b66e66116cbce8db1dcba4faf3b5"
CANONICAL_V2_BUNDLE_SHA256 = "a627adfcf6c299d0fabb185345bbc01106ea6901380aa0fbfee37c830ad8cdb6"

_V2_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "source_repository_id",
        "source_seed",
        "source_tick",
        "provenance",
        "protocols",
        "documents",
        "roster",
        "capabilities",
    }
)
_V2_PROVENANCE_KEYS = frozenset(
    {
        "roster_source",
        "roster_source_sha256",
        "roster_snapshot_sha256",
        "roster_source_git_commit",
        "protocol_source_repository_id",
        "curation",
    }
)

ROSTER_ORIGIN_RETAINED = "retained_source_roster"
ROSTER_ORIGIN_FRESH = "fresh_roster"
CAPABILITY_FORM_EXECUTABLE = "executable"
CAPABILITY_FORM_REMOVED = "removed"
CAPABILITY_FORM_TEXT_ONLY = "content_matched_text_only"

ROSTER_ORIGINS = (ROSTER_ORIGIN_RETAINED, ROSTER_ORIGIN_FRESH)
CAPABILITY_FORMS = (
    CAPABILITY_FORM_EXECUTABLE,
    CAPABILITY_FORM_REMOVED,
    CAPABILITY_FORM_TEXT_ONLY,
)

# What an inherited member brings that a fresh hire does not. Physical and
# scheduling state (vitals, availability, current task) is deliberately absent:
# the arm transfers who the people ARE, not where they left off, and a target
# run starts at episode one on its own snapshot.
_MEMBER_STATE_FIELDS = (
    "profile",
    "skills",
    "failure_modes",
    "communication_style",
    "reputation",
    "go_to_tags",
)

# The doc_type a text-only capability is written under. It must NOT be one of
# DOCUMENT_TYPE_TO_CAPABILITY's keys, or the prose would be counted as a
# capability carrier and the arm would measure the thing it removes.
TEXT_ONLY_DOC_TYPE = "inherited_capability_description"


def _adopted_protocols(world: Any) -> list[Any]:
    registry = getattr(world, "protocol_registry", None)
    return [
        protocol
        for protocol in (getattr(registry, "protocols", {}) or {}).values()
        if str(getattr(protocol, "adoption_status", "")) == "adopted"
    ]


def _carrier_documents(world: Any) -> list[Any]:
    return [
        document
        for document in (getattr(world, "documents", {}) or {}).values()
        if canonical_capability(
            str(getattr(document, "doc_type", "")), CARRIER_KIND_DOCUMENT
        )
    ]


def export_capability_bundle(
    world: Any,
    *,
    source_repository_id: str,
    source_seed: int,
) -> dict[str, Any]:
    """Serialise what a formation run can hand to a target run.

    Only ADOPTED protocols travel. A proposal the source never adopted is not a
    capability the organization had, and carrying it would let an arm inherit
    something the source itself was still arguing about.
    """
    protocols: list[dict[str, Any]] = []
    for protocol in _adopted_protocols(world):
        protocols.append(
            {
                "protocol_id": str(getattr(protocol, "protocol_id", "")),
                "protocol_type": str(getattr(protocol, "protocol_type", "")),
                "rule_summary": str(getattr(protocol, "rule_summary", "")),
                "scope": str(getattr(protocol, "scope", "review")),
                "target_process": str(getattr(protocol, "target_process", "")),
                "supporters": list(getattr(protocol, "supporters", []) or []),
                "emergence_level": str(getattr(protocol, "emergence_level", "none")),
                "capability": canonical_capability(
                    str(getattr(protocol, "protocol_type", "")), CARRIER_KIND_PROTOCOL
                ),
            }
        )

    documents: list[dict[str, Any]] = []
    for document in _carrier_documents(world):
        documents.append(
            {
                "doc_id": str(getattr(document, "doc_id", "")),
                "doc_type": str(getattr(document, "doc_type", "")),
                "title": str(getattr(document, "title", "")),
                "content_summary": str(getattr(document, "content_summary", "")),
                "capability": canonical_capability(
                    str(getattr(document, "doc_type", "")), CARRIER_KIND_DOCUMENT
                ),
            }
        )

    roster: list[dict[str, Any]] = []
    for agent_id, agent in sorted((getattr(world, "agents", {}) or {}).items()):
        state = {"agent_id": str(agent_id), "role": str(getattr(agent, "role", ""))}
        for field in _MEMBER_STATE_FIELDS:
            value = getattr(agent, field, None)
            if isinstance(value, dict):
                state[field] = dict(value)
            elif isinstance(value, (list, tuple)):
                state[field] = list(value)
            elif value is not None:
                state[field] = value
        roster.append(state)

    return {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "source_repository_id": str(source_repository_id),
        "source_seed": int(source_seed),
        "source_tick": int(getattr(world, "world_tick", 0) or 0),
        "protocols": protocols,
        "documents": documents,
        "roster": roster,
        "capabilities": sorted(
            {row["capability"] for row in protocols + documents if row["capability"]}
        ),
    }


def validate_capability_bundle(bundle: Mapping[str, Any]) -> None:
    """Validate a legacy v1 or sealed canonical v2 transfer bundle."""

    schema_version = bundle.get("schema_version")
    if schema_version not in SUPPORTED_BUNDLE_SCHEMA_VERSIONS:
        raise ValueError(
            f"capability_bundle_schema_unsupported:{bundle.get('schema_version')}"
        )
    source_repository_id = str(bundle.get("source_repository_id") or "")
    if not source_repository_id.strip():
        raise ValueError("capability_bundle_source_repository_required")
    for key in ("protocols", "documents", "roster"):
        if not isinstance(bundle.get(key), list):
            raise ValueError(f"capability_bundle_{key}_must_be_a_list")
    if schema_version != BUNDLE_SCHEMA_VERSION_V2:
        return

    from environments.org_env.policy.compiled_protocols import (
        validate_protocol_row_v2,
    )

    keys = frozenset(bundle.keys())
    if keys != _V2_TOP_LEVEL_KEYS:
        missing = sorted(_V2_TOP_LEVEL_KEYS - keys)
        unknown = sorted(keys - _V2_TOP_LEVEL_KEYS)
        raise ValueError(
            "capability_bundle_v2_top_level_schema_mismatch:"
            f"missing={','.join(missing)}:unknown={','.join(unknown)}"
        )
    provenance = bundle.get("provenance")
    if (
        not isinstance(provenance, Mapping)
        or frozenset(provenance.keys()) != _V2_PROVENANCE_KEYS
    ):
        raise ValueError("capability_bundle_v2_provenance_schema_mismatch")
    for key in _V2_PROVENANCE_KEYS:
        if not str(provenance.get(key) or "").strip():
            raise ValueError(f"capability_bundle_v2_provenance_{key}_required")
    for field in ("source_seed", "source_tick"):
        value = bundle.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"capability_bundle_v2_{field}_must_be_nonnegative_int")
    if re.fullmatch(
        r"[0-9a-f]{64}", str(provenance.get("roster_source_sha256") or "")
    ) is None:
        raise ValueError("capability_bundle_v2_roster_source_sha256_invalid")
    if re.fullmatch(
        r"[0-9a-f]{40}", str(provenance.get("roster_source_git_commit") or "")
    ) is None:
        raise ValueError("capability_bundle_v2_roster_source_git_commit_invalid")
    roster = bundle.get("roster") or []
    roster_ids: list[str] = []
    for index, member in enumerate(roster):
        if not isinstance(member, Mapping):
            raise ValueError(f"capability_bundle_v2_roster_{index}_must_be_an_object")
        agent_id = str(member.get("agent_id") or "")
        if not agent_id:
            raise ValueError(f"capability_bundle_v2_roster_{index}_agent_id_required")
        roster_ids.append(agent_id)
    if len(set(roster_ids)) != len(roster_ids):
        raise ValueError("capability_bundle_v2_roster_agent_ids_must_be_unique")
    roster_projection = json.dumps(
        roster,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    if str(provenance.get("roster_snapshot_sha256") or "") != sha256(
        roster_projection
    ).hexdigest():
        raise ValueError("capability_bundle_v2_roster_snapshot_hash_mismatch")
    if bundle.get("documents") != []:
        raise ValueError("capability_bundle_v2_documents_must_be_empty")
    protocols = bundle.get("protocols") or []
    if not protocols:
        raise ValueError("capability_bundle_v2_protocols_required")
    seen: set[str] = set()
    for row in protocols:
        if not isinstance(row, Mapping):
            raise ValueError("capability_bundle_v2_protocol_row_must_be_an_object")
        validate_protocol_row_v2(row)
        protocol_id = str(row.get("protocol_id") or "")
        if protocol_id in seen:
            raise ValueError(f"capability_bundle_v2_duplicate_protocol:{protocol_id}")
        seen.add(protocol_id)
    expected_capabilities = sorted(
        {str(row.get("capability") or "") for row in protocols}
    )
    if bundle.get("capabilities") != expected_capabilities:
        raise ValueError("capability_bundle_v2_capabilities_mismatch")
    if source_repository_id == "canonical_v2":
        if (
            str(provenance.get("roster_source_sha256") or "")
            != CANONICAL_V1_SHA256
            or str(provenance.get("roster_source_git_commit") or "")
            != CANONICAL_V1_GIT_COMMIT
            or str(provenance.get("protocol_source_repository_id") or "")
            != "canonical_v1"
            or capability_bundle_sha256(bundle) != CANONICAL_V2_BUNDLE_SHA256
        ):
            raise ValueError("capability_bundle_v2_canonical_identity_mismatch")


def capability_bundle_sha256(bundle: Mapping[str, Any]) -> str:
    """Content identity for the exact validated transfer input."""

    encoded = json.dumps(
        bundle,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _inject_roster(world: Any, roster: Sequence[Mapping[str, Any]]) -> list[str]:
    """Give the target's members the source members' learned state.

    Both repositories are staffed from the same seed team, so a retained roster
    is the same people carrying what they became — matched by id. A source
    member with no counterpart in the target is skipped rather than created: the
    arm varies what the roster KNOWS, not how large it is.
    """
    applied: list[str] = []
    agents = getattr(world, "agents", {}) or {}
    for member in roster:
        agent = agents.get(str(member.get("agent_id") or ""))
        if agent is None:
            continue
        for field in _MEMBER_STATE_FIELDS:
            if field not in member:
                continue
            value = member[field]
            setattr(agent, field, copy.deepcopy(value))
        applied.append(str(member.get("agent_id")))
    return applied


INHERITED_PREFIX = "inherited_"


def inherited_base_id(protocol_id: Any) -> str:
    """The id a transferred protocol carried in the run it came from."""
    text = str(protocol_id or "")
    return text[len(INHERITED_PREFIX):] if text.startswith(INHERITED_PREFIX) else text


def _inherited_id(protocol_id: str) -> str:
    """A registry id a locally formed protocol cannot land on.

    A ProtocolSpec formed here mirrors into the registry as
    ``proto_spec_{n}`` (`OrgWorld._mirror_protocol_event`), and a source run's
    protocols carry exactly those ids because they were mirrors there too. So
    an injected ``proto_spec_3`` sat in the slot this run's own third protocol
    would claim, and by t336 seven of eight injected rules had been overwritten
    by rules written during the run -- leaving the arm that was supposed to
    inherit a capability holding one. Namespacing the injected ids keeps the
    two populations apart and makes "did the inherited rule do the work"
    answerable without matching on rule text.
    """
    base = inherited_base_id(protocol_id)
    return f"{INHERITED_PREFIX}{base}" if base else ""


def _inherited_spec(
    world: Any,
    registry_id: str,
    row: Mapping[str, Any],
    tick: int,
    index: int,
    *,
    bundle_schema_version: str = BUNDLE_SCHEMA_VERSION_V1,
) -> str:
    """Give a transferred rule the spec that makes it countable and repairable.

    Injection wrote only to ``protocol_registry``, but every counter and every
    repair path reads ``proposal_manager.protocol_specs``: ``note_protocol_use``
    and ``note_protocol_enforcement`` iterate specs, and so does
    ``_flag_harmful_protocols``. So an inherited rule could refuse work all run
    and record none of it, and an inherited rule that strangled the
    organization could never be amended or repealed -- a handicap on the arm
    holding the capability, produced by the instrument rather than by the
    design.
    """
    manager = getattr(world, "proposal_manager", None)
    if manager is None or not hasattr(manager, "protocol_specs"):
        return ""
    from environments.org_env.proposals.objects import ProtocolSpec

    spec_id = f"protospec_inherited{index}"
    rule = str(row.get("rule_summary") or "")
    supporters = [str(value) for value in (row.get("supporters") or []) if str(value)]
    common = {
        "protocol_id": spec_id,
        "name": str(row.get("name") or rule[:80]),
        "family": str(row.get("protocol_type") or ""),
        "status": "adopted",
        "proposed_by": supporters[0] if supporters else "inherited",
        "adopted_by": list(supporters),
        "created_at_tick": tick,
        "adopted_at_tick": tick,
    }
    if bundle_schema_version == BUNDLE_SCHEMA_VERSION_V2:
        from environments.org_env.policy.compiled_protocols import (
            BINDING_SCHEMA_VERSION,
            compile_protocol_row_v2,
        )

        structured = row.get("spec") or {}
        bindings, binding_hash = compile_protocol_row_v2(row)
        spec = ProtocolSpec(
            **common,
            trigger_condition=str(structured.get("trigger_condition") or ""),
            required_steps=copy.deepcopy(structured.get("required_steps") or []),
            required_fields=copy.deepcopy(structured.get("required_fields") or []),
            enforcement_rule=str(structured.get("enforcement_rule") or ""),
            violation_condition=str(structured.get("violation_condition") or ""),
            exception_rule=(
                str(structured.get("exception_rule"))
                if structured.get("exception_rule") is not None
                else None
            ),
            problem_evidence=copy.deepcopy(structured.get("problem_evidence") or []),
            scope=str(structured.get("scope") or row.get("scope") or ""),
            responsible_roles=copy.deepcopy(
                structured.get("responsible_roles") or {}
            ),
            success_metric=str(structured.get("success_metric") or ""),
            enforcement_action=str(structured.get("enforcement_action") or ""),
            sunset_rule=str(structured.get("sunset_rule") or ""),
            affected_agents=copy.deepcopy(structured.get("affected_agents") or []),
            affected_actions=copy.deepcopy(structured.get("affected_actions") or []),
            affected_artifacts=copy.deepcopy(
                structured.get("affected_artifacts") or []
            ),
            binding_schema_version=BINDING_SCHEMA_VERSION,
            machine_bindings=bindings,
            binding_source_protocol_id=str(row.get("protocol_id") or ""),
            binding_spec_snapshot=copy.deepcopy(dict(structured)),
            binding_row_snapshot=copy.deepcopy({**dict(row), "bindings": bindings}),
            binding_hash=binding_hash,
            compiler_status="compiled",
            benefits=copy.deepcopy(structured.get("benefits") or []),
            costs=copy.deepcopy(structured.get("costs") or []),
            risks=copy.deepcopy(structured.get("risks") or []),
        )
    else:
        spec = ProtocolSpec(
            **common,
            # Legacy rows remain advisory and retain the historic readable
            # keyword path; only curated v2 rows receive machine bindings.
            trigger_condition=rule,
            enforcement_rule=rule,
            scope=str(row.get("scope") or ""),
        )
    manager.protocol_specs[spec_id] = spec
    # The mirror id is derived from the spec id by string elsewhere, which
    # cannot produce a namespaced id; state the pairing instead of deriving it.
    world.__dict__.setdefault("_protocol_mirror_ids", {})[spec_id] = registry_id
    return spec_id


def _inject_executable(
    world: Any, bundle: Mapping[str, Any], tick: int
) -> tuple[list[str], list[str]]:
    """Recreate the capability objects so they are in force from tick one.

    Adoption is forced. The support chain that earned adoption happened in the
    source run, and re-deriving it here would be the organization forming the
    capability again — the thing this arm holds constant.
    """
    from environments.org_env.backend.entities import Document

    registry = getattr(world, "protocol_registry", None)
    is_v2 = str(bundle.get("schema_version") or "") == BUNDLE_SCHEMA_VERSION_V2
    if is_v2:
        manager = getattr(world, "proposal_manager", None)
        if registry is None:
            raise ValueError("capability_bundle_v2_protocol_registry_required")
        if manager is None or not hasattr(manager, "protocol_specs"):
            raise ValueError("capability_bundle_v2_proposal_manager_required")
        proposed_ids = [
            _inherited_id(str(row.get("protocol_id") or ""))
            for row in (bundle.get("protocols") or [])
        ]
        if any(not protocol_id for protocol_id in proposed_ids):
            raise ValueError("capability_bundle_v2_inherited_protocol_id_missing")
        collisions = [
            protocol_id
            for protocol_id in proposed_ids
            if protocol_id in registry.protocols
        ]
        if collisions:
            raise ValueError(
                "capability_bundle_v2_protocol_collision:"
                + ",".join(sorted(collisions))
            )
    protocols: list[str] = []
    if registry is not None:
        for index, row in enumerate(bundle.get("protocols") or [], start=1):
            protocol_id = _inherited_id(str(row.get("protocol_id") or ""))
            if not protocol_id:
                if is_v2:
                    raise ValueError("capability_bundle_v2_inherited_protocol_id_missing")
                continue
            if protocol_id in registry.protocols:
                if is_v2:
                    raise ValueError(
                        f"capability_bundle_v2_protocol_collision:{protocol_id}"
                    )
                continue
            supporters = [str(value) for value in (row.get("supporters") or [])]
            proposer = supporters[0] if supporters else ""
            protocol = registry.propose(
                proposer_id=proposer,
                protocol_type=str(row.get("protocol_type") or ""),
                rule_summary=str(row.get("rule_summary") or ""),
                scope=str(row.get("scope") or "review"),
                target_process=str(row.get("target_process") or ""),
                tick=tick,
                protocol_id=protocol_id,
            )
            for supporter in supporters:
                if supporter and supporter not in protocol.supporters:
                    protocol.supporters.append(supporter)
            # Where it came from, so analysis does not have to infer it from the
            # source pack's vocabulary appearing in the rule text.
            setattr(protocol, "inherited_from",
                    str(bundle.get("source_repository_id") or "") or "transfer")
            setattr(protocol, "inherited_as", inherited_base_id(row.get("protocol_id")))
            registry.adopt(protocol_id, tick=tick, force=True)
            spec_id = _inherited_spec(
                world,
                protocol_id,
                row,
                tick,
                index,
                bundle_schema_version=str(
                    bundle.get("schema_version") or BUNDLE_SCHEMA_VERSION_V1
                ),
            )
            if is_v2 and not spec_id:
                raise ValueError(
                    "capability_bundle_v2_protocol_spec_injection_failed:"
                    f"{protocol_id}"
                )
            protocols.append(protocol_id)

    documents: list[str] = []
    store = getattr(world, "documents", None)
    if store is not None:
        for row in bundle.get("documents") or []:
            doc_id = str(row.get("doc_id") or "")
            if not doc_id or doc_id in store:
                continue
            store[doc_id] = Document(
                doc_id=doc_id,
                title=str(row.get("title") or ""),
                doc_type=str(row.get("doc_type") or ""),
                author_id="inherited",
                owner_id="inherited",
                visibility="team",
                content_summary=str(row.get("content_summary") or ""),
            )
            documents.append(doc_id)
    return protocols, documents


def capability_as_prose(row: Mapping[str, Any]) -> str:
    """Return exactly the rule text the executable arm exposes to members.

    Executable metadata remains on the protocol object but is not part of the
    prompt.  Rendering it only for Text would change information as well as the
    executable binding, so the content-matched treatment uses this summary.
    """

    return str(row.get("rule_summary") or "").strip()


def _inject_text_only(
    world: Any, bundle: Mapping[str, Any]
) -> list[str]:
    """Put matched prose in the dedicated always-visible prompt state only."""

    state = getattr(world, "__dict__", {})
    prose = state.setdefault("_inherited_capability_prose", [])
    injected: list[str] = []
    for row in bundle.get("protocols") or []:
        text = capability_as_prose(row)
        if not text or text in prose:
            continue
        prose.append(text)
        injected.append(
            f"prose_inherited_{row.get('protocol_id') or len(injected)}"
        )
    return injected


def inject_capability_bundle(
    world: Any,
    bundle: Mapping[str, Any],
    *,
    roster_origin: str,
    capability_form: str,
    frozen_episodes: int = 0,
    fixed_protocol_landscape: bool = False,
) -> dict[str, Any]:
    """Apply one arm's inheritance to a freshly built target world."""
    validate_capability_bundle(bundle)
    if roster_origin not in ROSTER_ORIGINS:
        raise ValueError(f"unknown_roster_origin:{roster_origin}")
    if capability_form not in CAPABILITY_FORMS:
        raise ValueError(f"unknown_capability_form:{capability_form}")

    tick = int(getattr(world, "world_tick", 0) or 0)
    receipt: dict[str, Any] = {
        "schema_version": str(bundle.get("schema_version") or BUNDLE_SCHEMA_VERSION),
        "source_repository_id": str(bundle.get("source_repository_id") or ""),
        "source_seed": int(bundle.get("source_seed") or 0),
        "source_tick": int(bundle.get("source_tick") or 0),
        "roster_origin": roster_origin,
        "capability_form": capability_form,
        "inherited_capabilities": list(bundle.get("capabilities") or []),
        "roster_applied": [],
        "protocols_injected": [],
        "documents_injected": [],
        "text_documents_written": [],
        "prose_entries_injected": [],
        "compiled_protocols": [],
    }
    if str(bundle.get("schema_version") or "") == BUNDLE_SCHEMA_VERSION_V2:
        receipt["bundle_sha256"] = capability_bundle_sha256(bundle)
        receipt["bundle_snapshot"] = copy.deepcopy(dict(bundle))

    if roster_origin == ROSTER_ORIGIN_RETAINED:
        receipt["roster_applied"] = _inject_roster(world, bundle.get("roster") or [])

    if capability_form == CAPABILITY_FORM_EXECUTABLE:
        protocols, documents = _inject_executable(world, bundle, tick)
        if (
            str(bundle.get("schema_version") or "") == BUNDLE_SCHEMA_VERSION_V2
            and len(protocols) != len(bundle.get("protocols") or [])
        ):
            raise ValueError("capability_bundle_v2_partial_protocol_injection")
        receipt["protocols_injected"] = protocols
        receipt["documents_injected"] = documents
        manager = getattr(world, "proposal_manager", None)
        for spec in (getattr(manager, "protocol_specs", {}) or {}).values():
            if getattr(spec, "compiler_status", "") != "compiled":
                continue
            spec_id = str(getattr(spec, "protocol_id", ""))
            mirror_id = str(
                (getattr(world, "__dict__", {}).get("_protocol_mirror_ids") or {}).get(
                    spec_id, ""
                )
            )
            if not mirror_id:
                mirror_id = str(
                    getattr(world, "_registry_mirror_id", lambda _id: "")(spec_id)
                )
            if mirror_id not in protocols:
                continue
            receipt["compiled_protocols"].append(
                {
                    "protocol_id": mirror_id,
                    "protocol_spec_id": str(getattr(spec, "protocol_id", "")),
                    "binding_hash": str(getattr(spec, "binding_hash", "")),
                    "binding_ids": [
                        str(binding.get("binding_id") or "")
                        for binding in (getattr(spec, "machine_bindings", []) or [])
                    ],
                }
            )
        if str(bundle.get("schema_version") or "") == BUNDLE_SCHEMA_VERSION_V2:
            if len(receipt["compiled_protocols"]) != len(
                bundle.get("protocols") or []
            ):
                raise ValueError("capability_bundle_v2_partial_compiled_injection")
            if {
                str(row.get("protocol_id") or "")
                for row in receipt["compiled_protocols"]
            } != set(protocols):
                raise ValueError("capability_bundle_v2_compiled_manifest_mismatch")
    elif capability_form == CAPABILITY_FORM_TEXT_ONLY:
        receipt["prose_entries_injected"] = _inject_text_only(world, bundle)
    # CAPABILITY_FORM_REMOVED injects nothing by construction.

    # The freeze is set AFTER injection: the injected objects are inherited, not
    # compiled here, and forcing them in is exactly what the window protects.
    freeze_capability_compilation(world, frozen_episodes)
    receipt["frozen_episodes"] = int(frozen_episodes)
    if fixed_protocol_landscape:
        fix_protocol_landscape(world)
    receipt["fixed_protocol_landscape"] = bool(fixed_protocol_landscape)
    world.__dict__["_capability_transfer_receipt"] = receipt
    return receipt


def inherited_capability_texts(world: Any) -> list[str]:
    """The prose an arm inherited instead of executable rules.

    Read straight from the documents so it reaches a prompt without anyone
    having to remember it exists. An executable rule is in front of every
    member on every decision whether or not they thought to look it up, and
    prose behind a retrieval step is not the same treatment: the arms would
    then differ in what it costs to attend to a rule, and "always in view beats
    fetched on demand" is a different and far smaller claim than the one this
    contrast is for.
    """
    state = getattr(world, "__dict__", {})
    direct = [
        str(text).strip()
        for text in (state.get("_inherited_capability_prose") or [])
        if str(text).strip()
    ]
    if direct:
        return direct

    # Compatibility reader for historic v1 checkpoints.  Fresh v2 Text arms
    # use the non-searchable prompt state above rather than documents.
    store = getattr(world, "documents", None) or {}
    out: list[str] = []
    for document in store.values():
        if str(getattr(document, "doc_type", "")) != TEXT_ONLY_DOC_TYPE:
            continue
        text = str(getattr(document, "content_summary", "") or "").strip()
        if text:
            out.append(text)
    return out


def note_inherited_prose_shown(world: Any) -> None:
    """Count a prompt that carried the inherited prose.

    The manipulation check for this arm: the claim is that both arms had the
    rules in front of them, and a number is what makes that checkable rather
    than asserted.
    """
    key = "_inherited_prose_prompt_count"
    world.__dict__[key] = int(world.__dict__.get(key, 0) or 0) + 1


def inherited_prose_exposure(world: Any) -> dict[str, int]:
    """How often the prose was put in front of members, and how often read."""
    reads = 0
    for row in (list(getattr(world, "baseline_archived_action_log", []) or [])
                + list(getattr(world, "action_log", []) or [])):
        if not isinstance(row, dict):
            continue
        if str(row.get("action_type")) in ("read_knowledge", "internal_search"):
            if str(row.get("target") or "").startswith("doc_inherited_"):
                reads += 1
    return {
        "prompts_carrying_the_prose": int(
            world.__dict__.get("_inherited_prose_prompt_count", 0) or 0
        ),
        "documents_read": reads,
        "documents_available": sum(
            1
            for document in (getattr(world, "documents", None) or {}).values()
            if str(getattr(document, "doc_type", "")) == TEXT_ONLY_DOC_TYPE
        ),
        "prose_entries_available": len(inherited_capability_texts(world)),
    }


def freeze_capability_compilation(world: Any, episodes: int) -> None:
    """Block NEW capability formation for the first ``episodes`` episodes.

    Applied to the reference arm as well as the four inheriting arms. Without
    it the window stops being a comparison: F_Text can recompile its prose into
    rules and the reference can independently learn an equivalent capability,
    and both would read as the inherited capability having no effect.
    """
    episodes = max(0, int(episodes))
    world.__dict__["_capability_compilation_frozen_episodes"] = episodes
    registry = getattr(world, "protocol_registry", None)
    if registry is not None:
        registry.compilation_frozen = episodes > 0


def fix_protocol_landscape(world: Any) -> None:
    """Seal formation and lifecycle mutations for the complete target window."""

    world.__dict__["_fixed_protocol_landscape"] = True
    registry = getattr(world, "protocol_registry", None)
    if registry is not None:
        registry.compilation_frozen = True
        registry.formation_locked = True


def refresh_capability_compilation_freeze(world: Any) -> bool:
    """Lift the freeze once the window has passed. Returns whether it is on."""
    if bool(world.__dict__.get("_fixed_protocol_landscape", False)):
        registry = getattr(world, "protocol_registry", None)
        if registry is not None:
            registry.compilation_frozen = True
            registry.formation_locked = True
        return True
    episodes = int(
        world.__dict__.get("_capability_compilation_frozen_episodes", 0) or 0
    )
    registry = getattr(world, "protocol_registry", None)
    if episodes <= 0:
        if registry is not None:
            registry.compilation_frozen = False
        return False
    manager = getattr(world, "episode_manager", None)
    closed = sum(
        1
        for episode in (getattr(manager, "episodes", {}) or {}).values()
        if str(getattr(episode, "status", "")) != "open"
    )
    frozen = closed < episodes
    if registry is not None:
        registry.compilation_frozen = frozen
    return frozen


__all__ = [
    "BUNDLE_SCHEMA_VERSION",
    "BUNDLE_SCHEMA_VERSION_V1",
    "BUNDLE_SCHEMA_VERSION_V2",
    "SUPPORTED_BUNDLE_SCHEMA_VERSIONS",
    "CAPABILITY_FORMS",
    "CAPABILITY_FORM_EXECUTABLE",
    "CAPABILITY_FORM_REMOVED",
    "CAPABILITY_FORM_TEXT_ONLY",
    "ROSTER_ORIGINS",
    "ROSTER_ORIGIN_FRESH",
    "ROSTER_ORIGIN_RETAINED",
    "TEXT_ONLY_DOC_TYPE",
    "export_capability_bundle",
    "freeze_capability_compilation",
    "fix_protocol_landscape",
    "inject_capability_bundle",
    "refresh_capability_compilation_freeze",
    "validate_capability_bundle",
]
