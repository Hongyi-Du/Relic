"""Episode Manager (Priority 5) — multi-turn activity objects.

A single action updates persona at most once (§5). But teaching, prototyping,
building, civic trials and disputes span *many* turns; if each turn updated the
stable persona we would double-count (§6.1: "Bob teaches Alice for 5 turns —
trust must not go +0.08 every turn"). So multi-turn activity is abstracted into
**Episodes**: stateful task objects with participants, a goal, progress,
start/end conditions, and a single :class:`TerminalAppraisal` that is the *only*
place a long-running activity is allowed to move stable traits (§6.8).

What lives here:
  * :class:`EpisodeType` / :class:`EpisodeStatus` — the v1 episode taxonomy (§6.3)
    and lifecycle states (§6.7).
  * :class:`EpisodeRule` + :data:`EPISODE_RULES` — per-type start/progress/end
    parameters as DATA (§6.7), so the manager stays generic.
  * :class:`Episode` — the live task object (§6.2).
  * :class:`TerminalAppraisal` — the end-of-episode appraisal (§6.8).
  * :class:`EpisodeManager` — start / match / route / advance / end, plus the
    per-agent 1-foreground + ≤3-background attention budget (§6.6) and the
    multi-signal matching score (§6.5 / §8.8).

It is env-agnostic: episodes reference agents/resources/mechanisms by opaque ids
only, and the manager consumes the same :class:`~agent_sdk.lived.cognition.appraisal.\
EventAppraisal` the update pipeline produces.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from agent_sdk.lived.core.schema import Intensity


# --------------------------------------------------------------------------- #
# §6.3 Episode taxonomy  /  §6.7 lifecycle states
# --------------------------------------------------------------------------- #
class EpisodeType(str, Enum):
    # v1 minimum set (§6.3)
    MATERIAL_PROTOTYPE = "material_prototype"
    TEACHING = "teaching"
    BUILD_PROJECT = "build_project"
    CIVIC_TRIAL = "civic_trial"
    DISPUTE = "dispute"
    # optional later (schema reserved, no rules wired yet)
    MEDICAL_HELP = "medical_help"
    RESOURCE_EXPEDITION = "resource_expedition"
    TRADE_DEBT = "trade_debt"
    GROUP_MEETING = "group_meeting"


class EpisodeStatus(str, Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    ABANDONED = "abandoned"
    INTERRUPTED = "interrupted"
    EXPIRED = "expired"


TERMINAL_STATUSES = frozenset({
    EpisodeStatus.COMPLETED, EpisodeStatus.FAILED, EpisodeStatus.ABANDONED,
    EpisodeStatus.INTERRUPTED, EpisodeStatus.EXPIRED,
})


# --------------------------------------------------------------------------- #
# §6.7 Per-type rules (as data)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class EpisodeRule:
    """Start/progress/end parameters for one episode type (§6.7).

    The manager applies these generically: progress ≥ ``progress_target`` →
    COMPLETED; ``failure_streak`` consecutive failed micro-events → FAILED; no
    activity for ``max_idle_turns`` → ABANDONED; total span > ``max_duration``
    → EXPIRED. ``match_action_types`` feeds the §6.5 matching score
    (event_type_matches_episode_type). ``needs_group_adoption`` marks civic
    episodes that cannot be owned by a single agent (§17A.3).
    """
    progress_target: float = 1.0
    failure_streak: int = 3
    max_idle_turns: int = 10
    max_duration: int = 60
    needs_group_adoption: bool = False
    terminal_trait_update_allowed: bool = True
    match_action_types: Tuple[str, ...] = ()


EPISODE_RULES: Dict[EpisodeType, EpisodeRule] = {
    EpisodeType.MATERIAL_PROTOTYPE: EpisodeRule(
        progress_target=1.0, failure_streak=4, max_idle_turns=8, max_duration=40,
        match_action_types=("experiment", "prototype", "craft", "test_prototype", "wish"),
    ),
    EpisodeType.TEACHING: EpisodeRule(
        progress_target=1.0, failure_streak=5, max_idle_turns=6, max_duration=30,
        match_action_types=("teach", "learn", "demonstrate", "observe"),
    ),
    EpisodeType.BUILD_PROJECT: EpisodeRule(
        progress_target=1.0, failure_streak=6, max_idle_turns=12, max_duration=80,
        match_action_types=("build", "gather", "deposit", "construct", "haul"),
    ),
    EpisodeType.CIVIC_TRIAL: EpisodeRule(
        progress_target=1.0, failure_streak=99, max_idle_turns=15, max_duration=30,
        needs_group_adoption=True,
        match_action_types=("propose_rule", "support", "oppose", "enforce",
                            "deposit", "withdraw", "tally", "vote"),
    ),
    EpisodeType.DISPUTE: EpisodeRule(
        progress_target=1.0, failure_streak=99, max_idle_turns=5, max_duration=20,
        match_action_types=("accuse", "deny", "blame", "witness", "mediate",
                            "appeal", "compensate", "over_withdraw"),
    ),
}


# --------------------------------------------------------------------------- #
# §6.2 Episode object
# --------------------------------------------------------------------------- #
@dataclass
class Episode:
    episode_id: str
    episode_type: EpisodeType
    participants: List[str] = field(default_factory=list)
    goal: str = ""
    status: EpisodeStatus = EpisodeStatus.ACTIVE
    start_turn: int = 0
    last_active_turn: int = 0
    progress: float = 0.0
    failure_count: int = 0
    success_count: int = 0
    micro_event_ids: List[str] = field(default_factory=list)
    # §6.5 matching keys (all optional)
    session_id: Optional[str] = None
    prototype_id: Optional[str] = None
    mechanism_id: Optional[str] = None
    target_resource: Optional[str] = None
    location: Optional[Any] = None
    # filled at end (§6.8)
    terminal: Optional["TerminalAppraisal"] = None

    @property
    def active(self) -> bool:
        return self.status == EpisodeStatus.ACTIVE

    @property
    def duration(self) -> int:
        return self.last_active_turn - self.start_turn

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "episode_type": self.episode_type.value,
            "participants": list(self.participants),
            "goal": self.goal,
            "status": self.status.value,
            "start_turn": self.start_turn,
            "last_active_turn": self.last_active_turn,
            "progress": round(self.progress, 4),
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "micro_event_count": len(self.micro_event_ids),
            "session_id": self.session_id,
            "prototype_id": self.prototype_id,
            "mechanism_id": self.mechanism_id,
            "target_resource": self.target_resource,
            "terminal": self.terminal.to_dict() if self.terminal else None,
        }


# --------------------------------------------------------------------------- #
# §6.8 Terminal appraisal
# --------------------------------------------------------------------------- #
@dataclass
class TerminalAppraisal:
    """Produced once when an episode ends (§6.8). Only this (or a repeated
    high-intensity pattern) may update long-term traits."""
    episode_id: str
    episode_type: EpisodeType
    success_level: float = 0.0          # [0,1]
    completion_reason: str = ""
    cost_to_each_agent: Dict[str, float] = field(default_factory=dict)
    benefit_to_each_agent: Dict[str, float] = field(default_factory=dict)
    knowledge_gain: float = 0.0
    resource_gain: float = 0.0
    survival_impact: float = 0.0
    cooperation_quality: float = 0.0
    conflict_level: float = 0.0
    fairness_signal: float = 0.0
    trust_signal: float = 0.0
    reputation_signal: float = 0.0
    emotional_valence: float = 0.0
    intensity: Intensity = Intensity.MODERATE
    visibility: str = "witnessed"
    repeat_pattern: bool = False
    stable_trait_update_allowed: bool = True

    def to_dict(self) -> Dict[str, Any]:
        from dataclasses import asdict
        d = asdict(self)
        d["episode_type"] = self.episode_type.value
        d["intensity"] = self.intensity.value
        return d


# --------------------------------------------------------------------------- #
# §6.6 per-agent attention budget (1 foreground + ≤3 background)
# --------------------------------------------------------------------------- #
MAX_BACKGROUND = 3
MAX_RELATED_EPISODES = 3            # §8.8 cap on related_episode_ids
MATCH_PRIMARY_THRESHOLD = 0.5
MATCH_RELATED_THRESHOLD = 0.3


@dataclass
class AgentAttention:
    foreground: Optional[str] = None
    background: List[str] = field(default_factory=list)  # most-recent first


@dataclass
class RouteResult:
    """Outcome of routing one event through the manager (§6.6)."""
    primary_episode_id: Optional[str] = None
    related_episode_ids: List[str] = field(default_factory=list)
    started: bool = False
    transitioned: List[str] = field(default_factory=list)  # ep_ids that ended this step

    def to_dict(self) -> Dict[str, Any]:
        return {
            "primary_episode_id": self.primary_episode_id,
            "related_episode_ids": list(self.related_episode_ids),
            "started": self.started,
            "transitioned": list(self.transitioned),
        }


# --------------------------------------------------------------------------- #
# The manager
# --------------------------------------------------------------------------- #
class EpisodeManager:
    """Owns all episodes + the per-agent attention budget (§6)."""

    def __init__(self) -> None:
        self.episodes: Dict[str, Episode] = {}
        self.attention: Dict[str, AgentAttention] = {}
        self._seq = 0

    # -- attention helpers (§6.6) ------------------------------------------
    def _att(self, agent: str) -> AgentAttention:
        a = self.attention.get(agent)
        if a is None:
            a = AgentAttention()
            self.attention[agent] = a
        return a

    def _focus(self, agent: str, ep_id: str) -> Optional[str]:
        """Make ``ep_id`` the agent's foreground; demote the prior foreground to
        background; evict the least-recent background past the cap. Returns an
        evicted (interrupted) episode id, if any."""
        att = self._att(agent)
        if att.foreground == ep_id:
            return None
        # remove from background if it was there
        if ep_id in att.background:
            att.background.remove(ep_id)
        # demote current foreground
        if att.foreground is not None:
            att.background.insert(0, att.foreground)
        att.foreground = ep_id
        evicted: Optional[str] = None
        while len(att.background) > MAX_BACKGROUND:
            evicted = att.background.pop()       # least-recent
            ep = self.episodes.get(evicted)
            # only interrupt if this agent was its last attentive participant
            if ep is not None and ep.active and not self._anyone_attending(evicted, exclude=agent):
                self._terminate(ep, EpisodeStatus.INTERRUPTED, "attention_evicted", ep.last_active_turn)
        return evicted

    def _anyone_attending(self, ep_id: str, *, exclude: str = "") -> bool:
        for agent, att in self.attention.items():
            if agent == exclude:
                continue
            if att.foreground == ep_id or ep_id in att.background:
                return True
        return False

    def _detach_all(self, ep_id: str) -> None:
        for att in self.attention.values():
            if att.foreground == ep_id:
                att.foreground = None
            if ep_id in att.background:
                att.background.remove(ep_id)

    # -- §6.4 start --------------------------------------------------------
    def start_episode(
        self,
        episode_type: EpisodeType,
        participants: List[str],
        *,
        turn: int,
        goal: str = "",
        foreground: bool = True,
        episode_id: Optional[str] = None,
        **match_keys: Any,
    ) -> Episode:
        """Create an episode and (optionally) make it foreground for its
        participants (§6.4). ``match_keys`` may include session_id /
        prototype_id / mechanism_id / target_resource / location (§6.5)."""
        if episode_id is None:
            episode_id = f"ep_{episode_type.value}_{self._seq}"
        self._seq += 1
        ep = Episode(
            episode_id=episode_id, episode_type=episode_type,
            participants=list(participants), goal=goal,
            start_turn=turn, last_active_turn=turn,
            session_id=match_keys.get("session_id"),
            prototype_id=match_keys.get("prototype_id"),
            mechanism_id=match_keys.get("mechanism_id"),
            target_resource=match_keys.get("target_resource"),
            location=match_keys.get("location"),
        )
        self.episodes[episode_id] = ep
        for agent in participants:
            if foreground:
                self._focus(agent, episode_id)
            else:
                att = self._att(agent)
                if episode_id not in att.background and att.foreground != episode_id:
                    att.background.insert(0, episode_id)
                    while len(att.background) > MAX_BACKGROUND:
                        att.background.pop()
        return ep

    # -- §6.5 matching score -----------------------------------------------
    def match_score(self, ep: Episode, *, actor: str, target: Optional[str],
                    action_type: str, **hints: Any) -> float:
        """Weighted multi-signal match between an event and an episode (§6.5)."""
        if not ep.active:
            return 0.0
        score = 0.0
        if hints.get("session_id") and hints["session_id"] == ep.session_id:
            score += 0.5
        if hints.get("prototype_id") and hints["prototype_id"] == ep.prototype_id:
            score += 0.5
        if hints.get("mechanism_id") and hints["mechanism_id"] == ep.mechanism_id:
            score += 0.5
        if hints.get("target_resource") and hints["target_resource"] == ep.target_resource:
            score += 0.3
        # participant overlap
        evt_parts = {p for p in (actor, target) if p}
        if evt_parts & set(ep.participants):
            score += 0.3
        # location proximity (exact tile / id match as a cheap proxy)
        if hints.get("location") is not None and hints["location"] == ep.location:
            score += 0.1
        # temporal proximity (recent activity)
        turn = int(hints.get("turn", ep.last_active_turn))
        rule = EPISODE_RULES.get(ep.episode_type)
        if rule and (turn - ep.last_active_turn) <= rule.max_idle_turns:
            score += 0.1
        # event type matches episode type
        if rule and action_type in rule.match_action_types:
            score += 0.3
        # explicit reference
        if hints.get("episode_ref") == ep.episode_id:
            score += 1.0
        return score

    # -- §6.5/§6.6 route an event ------------------------------------------
    def route_event(self, appraisal: Any, **hints: Any) -> Dict[str, Any]:
        """Attach an appraised event to the best-matching episode(s), advance
        progress, run lifecycle transitions, and (for disputes) auto-start a
        cluster episode. Returns a serializable :class:`RouteResult`.

        ``appraisal`` is an :class:`~agent_sdk.lived.cognition.appraisal.EventAppraisal`
        (duck-typed). ``hints`` supply the §6.5 traceability ids."""
        actor = getattr(appraisal, "actor", "")
        target = getattr(appraisal, "target", None)
        action_type = getattr(appraisal, "action_type", "")
        turn = int(getattr(appraisal, "turn", 0))
        hints.setdefault("turn", turn)
        explicit = getattr(appraisal, "primary_episode_id", None)
        if explicit:
            hints.setdefault("episode_ref", explicit)

        result = RouteResult()

        # score every active episode
        scored: List[Tuple[float, Episode]] = []
        for ep in self.episodes.values():
            s = self.match_score(ep, actor=actor, target=target,
                                 action_type=action_type, **hints)
            if s > 0:
                scored.append((s, ep))
        scored.sort(key=lambda x: x[0], reverse=True)

        primary: Optional[Episode] = None
        if scored and scored[0][0] >= MATCH_PRIMARY_THRESHOLD:
            primary = scored[0][1]
        elif self._is_dispute_trigger(action_type) and not scored:
            # §6.4 event-cluster start: over-withdraw/accuse with no home episode
            parts = [p for p in (actor, target) if p]
            primary = self.start_episode(
                EpisodeType.DISPUTE, parts, turn=turn,
                goal=f"dispute around {action_type}",
                target_resource=hints.get("target_resource"),
            )
            result.started = True

        if primary is not None:
            result.primary_episode_id = primary.episode_id
            self._attach(primary, appraisal, turn)
            self._focus(actor, primary.episode_id)
            # related episodes (capped)
            for s, ep in scored:
                if ep is primary:
                    continue
                if s >= MATCH_RELATED_THRESHOLD and len(result.related_episode_ids) < MAX_RELATED_EPISODES:
                    result.related_episode_ids.append(ep.episode_id)
                    self._attach(ep, appraisal, turn, related=True)

        # lifecycle transitions for everything touched + idle/expiry sweep
        ended = self._sweep_transitions(turn)
        result.transitioned = ended
        return result.to_dict()

    def _is_dispute_trigger(self, action_type: str) -> bool:
        return action_type in ("accuse", "over_withdraw", "blame", "deny")

    def _attach(self, ep: Episode, appraisal: Any, turn: int, *, related: bool = False) -> None:
        """Record a micro-event on an episode + advance progress (§6.6)."""
        eid = getattr(appraisal, "actor", "") + ":" + getattr(appraisal, "action_type", "") + ":" + str(turn)
        ep.micro_event_ids.append(eid)
        ep.last_active_turn = max(ep.last_active_turn, turn)
        for p in (getattr(appraisal, "actor", ""), getattr(appraisal, "target", "")):
            if p and p not in ep.participants:
                ep.participants.append(p)
        if related:
            return  # related episodes log the event but don't advance progress
        success = bool(getattr(appraisal, "success", True))
        if success:
            ep.success_count += 1
            ep.failure_count = 0
            ep.progress = min(1.0, ep.progress + self._progress_step(ep))
        else:
            ep.failure_count += 1

    def _progress_step(self, ep: Episode) -> float:
        """Default progress increment per successful micro-event. Civic trials
        advance by adoption/use rather than a fixed step, so they move slower."""
        if ep.episode_type == EpisodeType.CIVIC_TRIAL:
            return 0.2
        return 0.34   # ~3 successful steps complete a material/teaching episode

    # -- §6.7 lifecycle transitions ----------------------------------------
    def _sweep_transitions(self, turn: int) -> List[str]:
        ended: List[str] = []
        for ep in list(self.episodes.values()):
            if not ep.active:
                continue
            rule = EPISODE_RULES.get(ep.episode_type)
            if rule is None:
                continue
            if ep.progress >= rule.progress_target:
                self._terminate(ep, EpisodeStatus.COMPLETED, "progress_target_met", turn)
                ended.append(ep.episode_id)
            elif ep.failure_count >= rule.failure_streak:
                self._terminate(ep, EpisodeStatus.FAILED, "failure_streak", turn)
                ended.append(ep.episode_id)
            elif (turn - ep.last_active_turn) > rule.max_idle_turns:
                self._terminate(ep, EpisodeStatus.ABANDONED, "idle_timeout", turn)
                ended.append(ep.episode_id)
            elif (turn - ep.start_turn) > rule.max_duration:
                self._terminate(ep, EpisodeStatus.EXPIRED, "max_duration", turn)
                ended.append(ep.episode_id)
        return ended

    def tick(self, turn: int) -> List[str]:
        """Per-turn idle/expiry sweep (call once per turn from the engine)."""
        return self._sweep_transitions(turn)

    # -- §6.7/§6.8 end + terminal appraisal --------------------------------
    def end_episode(self, episode_id: str, status: EpisodeStatus, *,
                    turn: int, reason: str = "", **signals: Any) -> Optional[TerminalAppraisal]:
        """Explicitly end an episode (e.g. agent abandons, civic trial adopted)
        and produce its terminal appraisal (§6.8)."""
        ep = self.episodes.get(episode_id)
        if ep is None or not ep.active:
            return ep.terminal if ep else None
        return self._terminate(ep, status, reason or status.value, turn, **signals)

    def _terminate(self, ep: Episode, status: EpisodeStatus, reason: str,
                   turn: int, **signals: Any) -> TerminalAppraisal:
        ep.status = status
        ep.last_active_turn = max(ep.last_active_turn, turn)
        self._detach_all(ep.episode_id)
        ep.terminal = self._build_terminal(ep, reason, **signals)
        return ep.terminal

    def _build_terminal(self, ep: Episode, reason: str, **signals: Any) -> TerminalAppraisal:
        rule = EPISODE_RULES.get(ep.episode_type, EpisodeRule())
        success_level = {
            EpisodeStatus.COMPLETED: 1.0,
            EpisodeStatus.FAILED: 0.0,
            EpisodeStatus.ABANDONED: 0.2,
            EpisodeStatus.INTERRUPTED: 0.3,
            EpisodeStatus.EXPIRED: 0.25,
        }.get(ep.status, ep.progress)
        # repeated pattern raises intensity → allowed to move stable traits more
        repeat = bool(signals.get("repeat_pattern", False))
        if ep.status == EpisodeStatus.COMPLETED:
            intensity = Intensity.REPEATED if repeat else Intensity.MAJOR
        elif ep.status == EpisodeStatus.FAILED:
            intensity = Intensity.MODERATE
        else:
            intensity = Intensity.MINOR
        ta = TerminalAppraisal(
            episode_id=ep.episode_id, episode_type=ep.episode_type,
            success_level=success_level, completion_reason=reason,
            knowledge_gain=float(signals.get("knowledge_gain", 0.0)),
            resource_gain=float(signals.get("resource_gain", 0.0)),
            survival_impact=float(signals.get("survival_impact", 0.0)),
            cooperation_quality=float(signals.get("cooperation_quality", 0.0)),
            conflict_level=float(signals.get("conflict_level",
                                             1.0 if ep.episode_type == EpisodeType.DISPUTE else 0.0)),
            fairness_signal=float(signals.get("fairness_signal", 0.0)),
            trust_signal=float(signals.get("trust_signal", 0.0)),
            reputation_signal=float(signals.get("reputation_signal", 0.0)),
            emotional_valence=float(signals.get("emotional_valence", success_level * 2 - 1)),
            intensity=intensity,
            visibility=str(signals.get("visibility", "witnessed")),
            repeat_pattern=repeat,
            stable_trait_update_allowed=rule.terminal_trait_update_allowed
            and ep.status in (EpisodeStatus.COMPLETED, EpisodeStatus.FAILED),
        )
        ta.cost_to_each_agent = {a: float(signals.get("cost", 0.0)) for a in ep.participants}
        ta.benefit_to_each_agent = {a: success_level for a in ep.participants}
        return ta

    # -- §6.8 bridge: terminal appraisal -> per-agent EventAppraisal --------
    def terminal_to_appraisals(self, terminal: TerminalAppraisal) -> List[Any]:
        """Turn a terminal appraisal into one
        :class:`~agent_sdk.lived.cognition.appraisal.EventAppraisal` per participant, so
        the §5.2 update path applies the *single* allowed stable-trait update for
        the whole episode (§6.8). Returns [] if updates aren't allowed."""
        from agent_sdk.lived.cognition.appraisal import EventAppraisal  # lazy: avoid cycle

        if not terminal.stable_trait_update_allowed:
            return []
        out: List[EventAppraisal] = []
        kind = self._episode_trigger_kind(terminal)
        for agent in terminal.benefit_to_each_agent:
            ap = EventAppraisal(
                actor=agent, action_type=f"episode::{terminal.episode_type.value}",
                success=terminal.success_level >= 0.5,
                knowledge_gain=terminal.knowledge_gain,
                emotional_valence=terminal.emotional_valence,
                intensity=terminal.intensity,
                related_episode_id=terminal.episode_id,
                trigger_kinds=[kind] if kind else [],
                visibility=terminal.visibility,
            )
            out.append(ap)
        return out

    def _episode_trigger_kind(self, terminal: TerminalAppraisal) -> str:
        """Map a completed episode type to the canonical trait-trigger kind."""
        if terminal.success_level < 0.5:
            if terminal.episode_type == EpisodeType.MATERIAL_PROTOTYPE:
                return "repeated_experiment_failure"
            return ""
        return {
            EpisodeType.TEACHING: "successful_teaching",
            EpisodeType.MATERIAL_PROTOTYPE: "prototype_success",
            EpisodeType.BUILD_PROJECT: "group_success",
            EpisodeType.CIVIC_TRIAL: "rule_compliance_rewarded",
        }.get(terminal.episode_type, "")

    # -- queries -----------------------------------------------------------
    def active_episodes(self) -> List[Episode]:
        return [e for e in self.episodes.values() if e.active]

    def foreground_of(self, agent: str) -> Optional[Episode]:
        att = self.attention.get(agent)
        if att is None or att.foreground is None:
            return None
        return self.episodes.get(att.foreground)

    def background_of(self, agent: str) -> List[Episode]:
        att = self.attention.get(agent)
        if att is None:
            return []
        return [self.episodes[e] for e in att.background if e in self.episodes]

    def snapshot(self) -> List[Dict[str, Any]]:
        return [e.to_dict() for e in self.episodes.values()]
