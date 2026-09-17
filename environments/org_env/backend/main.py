"""OrgEnv Live Inspector server (frontend spec §9).

A small FastAPI app that serves the two-tab live debugger at ``/org/inspector``
and feeds it JSON from ``/api/org/lived/*`` + sim controls at ``/api/org/sim/*``
+ object endpoints at ``/api/org/*``. All route LOGIC lives in plain functions
(``api_*``) operating on a module-global :class:`OrgInspectorSession`, so they're
unit-testable without FastAPI installed; ``build_app()`` imports FastAPI lazily
and wires thin route wrappers.

Run:  PYTHONPATH="." python -m environments.org_env.backend.main
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from environments.org_env.human.api import HumanApi
from environments.org_env.human.liaison import LiaisonFacade
from environments.org_env.runtime_adapter.live import OrgInspectorSession
from relic.paths import benchmark_root

BASE_DIR = Path(__file__).resolve().parents[3]
TEMPLATE = BASE_DIR / "environments" / "org_env" / "frontend" / "templates" / "org_inspector.html"
# React + Vite + React Flow build output (served at /org/inspector; assets under /org/app)
APP_DIST = BASE_DIR / "environments" / "org_env" / "frontend" / "app" / "dist"
REPLAY_DIR = BASE_DIR / "docs" / "replays"
CHECKPOINT_DIR = BASE_DIR / "log" / "checkpoints"   # gitignored; faithful world snapshots
PACK_DIR = benchmark_root() / "relic-main-v1" / "packs"

# module-global running session (the inspector drives ONE world)
# Now lazily initialized after pack selection
SESSION: Optional[OrgInspectorSession] = None
_SESSION_LOCK = __import__("threading").Lock()

# Human seats over the same world (HCI V0). Lazily binds a real-time runtime the
# first time anyone asks, so plain inspector runs pay nothing for it.
HUMAN = HumanApi(lambda: SESSION)
LIAISON = LiaisonFacade(HUMAN)


# --------------------------------------------------------------------------- #
# Route logic (plain, testable) — every function returns a JSON-able dict/list.
# --------------------------------------------------------------------------- #

def api_packs() -> List[Dict[str, Any]]:
    """List all available OSS packs with metadata."""
    packs = []

    if not PACK_DIR.exists():
        return packs

    for pack_path in sorted(PACK_DIR.iterdir()):
        if not pack_path.is_dir():
            continue

        pack_id = pack_path.name

        # Parse pack name and version from directory name
        # e.g., "black_v2610_to_v2651" -> name="Black", from_version="2.6.10", to_version="2.6.51"
        parts = pack_id.split("_")

        # Extract name (everything before version indicators)
        name_parts = []
        for part in parts:
            if part.startswith("v") and part[1:2].isdigit():
                break
            name_parts.append(part)

        display_name = " ".join(name_parts).title()

        # Extract version info if present
        version_info = ""
        if "to" in parts:
            try:
                to_idx = parts.index("to")
                if to_idx > 0 and to_idx + 1 < len(parts):
                    from_v = parts[to_idx - 1].lstrip("v")
                    to_v = parts[to_idx + 1].lstrip("v")
                    # Format version numbers (e.g., "2610" -> "2.6.10")
                    version_info = f"v{from_v[:1]}.{from_v[1:2]}.{from_v[2:]} → v{to_v[:1]}.{to_v[1:2]}.{to_v[2:]}"
            except (ValueError, IndexError):
                pass
        elif len(parts) > 1 and parts[-1].startswith("v"):
            version_info = parts[-1]

        # Count issues and PRs in starter repo
        starter_repo = pack_path / "starter_repo"
        issues_count = 0
        prs_count = 0

        if starter_repo.exists():
            git_fixtures = starter_repo / ".git_fixtures"
            if git_fixtures.exists():
                issues_dir = git_fixtures / "issues"
                prs_dir = git_fixtures / "pull_requests"

                if issues_dir.exists():
                    issues_count = len(list(issues_dir.glob("*.md")))
                if prs_dir.exists():
                    prs_count = len(list(prs_dir.glob("*.md")))

        # Read description if manifest exists
        description = f"Work on {display_name} project"
        manifest_path = pack_path / "manifest.json"
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text())
                description = manifest.get("description", description)
            except (json.JSONDecodeError, OSError):
                pass

        packs.append({
            "id": pack_id,
            "name": display_name,
            "version": version_info,
            "description": description,
            "issues": issues_count,
            "prs": prs_count,
        })

    return packs


def api_init_world(pack_id: str, warmup: int = 36, clock_speed: float = 15.0,
                   user_id: Optional[str] = None,
                   start_runtime: Optional[bool] = None) -> Dict[str, Any]:
    """Initialize the world with a selected pack."""
    global SESSION

    with _SESSION_LOCK:
        if SESSION is not None:
            return {"error": "World already initialized"}

        pack_path = PACK_DIR / pack_id
        if not pack_path.exists():
            return {"error": f"Pack not found: {pack_id}"}

        # Create session
        SESSION = OrgInspectorSession(
            seed=42,
            interaction_profile="human_project_workspace",
        )

        # Store user_id in session metadata
        if user_id:
            SESSION.user_id = user_id

        # Load pack (we'll implement this next)
        try:
            _load_pack_into_session(pack_id, pack_path)
        except Exception as e:
            SESSION = None
            return {"error": f"Failed to load pack: {e}"}

        # Warmup
        if warmup > 0:
            SESSION.step(warmup)

        # Configure clock
        runtime = HUMAN.runtime()
        runtime.seconds_per_tick = max(0.05, clock_speed)
        # ``tools/run_hci.py --paused`` reaches this lazy initialization point
        # only now.  An API caller can override the launcher default explicitly.
        if start_runtime is None:
            start_runtime = os.environ.get("ORG_HCI_START_PAUSED", "0").lower() not in {
                "1", "true", "yes",
            }
        if start_runtime:
            runtime.start()
        else:
            runtime.pause()

        return {
            "ok": True,
            "pack": dict(SESSION.selected_pack),
            "agents": len(SESSION.world.agents),
            "warmup_hours": warmup,
            "user_id": user_id,
            "runtime": _runtime_status(),
        }


def _load_pack_into_session(pack_id: str, pack_path: Path) -> None:
    """Bind a live session to the selected frozen OSS pack and rebuild it.

    This is intentionally a real substrate load, not a display label.  The
    loader resolves the manifest and the session passes its public starter
    repository / issue stream into ``OrgWorld`` through ``company_config``.
    """
    if SESSION is None:
        raise RuntimeError("session_required_before_pack_load")
    from environments.org_env.product.substrates.loader import load_oss_substrate_spec

    spec = load_oss_substrate_spec(pack_id)
    if Path(spec.dataset_dir).resolve() != pack_path.resolve():
        raise RuntimeError("selected_pack_path_does_not_match_resolved_dataset")
    manifest = spec.manifest
    substrate = {
        "type": "oss_time_machine",
        "dataset_id": pack_id,
        "repository_id": spec.project_id,
        "anonymize": bool(manifest.get("anonymized_product_name")),
        "mode": "dev",
    }
    selected_pack = {
        "id": pack_id,
        "source": "oss_time_machine/real",
        "dataset_id": spec.project_id,
        "product_name": spec.product_name,
    }
    import os
    SESSION.load_llm = os.environ.get("ORG_LLM", "1").lower() in ("1", "true", "yes")
    SESSION.approval_mode = "semi_auto"
    SESSION.reset(seed=42, product_substrate=substrate, selected_pack=selected_pack)


def _runtime_status(status: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return the live clock state plus the session's verified pack binding."""
    payload = dict(status if status is not None else HUMAN.runtime_status())
    if SESSION is not None:
        payload.update(SESSION.runtime_metadata())
        payload["session_last_error"] = SESSION.last_error
    return payload


def _optional_bool(value: Any) -> Optional[bool]:
    """Parse an optional JSON-ish boolean without treating ``"false"`` as true."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError("start_runtime_must_be_boolean")


def _ensure_session() -> Dict[str, Any]:
    """Return error dict if SESSION not initialized, None if OK."""
    if SESSION is None:
        return {"error": "World not initialized. Visit /org/setup first."}
    return None


def api_full() -> Dict[str, Any]:
    err = _ensure_session()
    if err:
        return err
    return SESSION.full()


def api_frames(since: int = -1) -> Dict[str, Any]:
    err = _ensure_session()
    if err:
        return err
    return SESSION.frames_since(since)


def api_state() -> Dict[str, Any]:
    err = _ensure_session()
    if err:
        return err
    state = SESSION.state()
    state["runtime"] = _runtime_status()
    return state


def api_sim_step(n: int = 1) -> Dict[str, Any]:
    err = _ensure_session()
    if err:
        return err
    return SESSION.step(n)


def api_sim_run_ticks(n: int = 24) -> Dict[str, Any]:
    err = _ensure_session()
    if err:
        return err
    return SESSION.run_ticks(n)


def api_sim_pause() -> Dict[str, Any]:
    err = _ensure_session()
    if err:
        return err
    return SESSION.pause()


def api_sim_reset(seed: Optional[int] = None) -> Dict[str, Any]:
    err = _ensure_session()
    if err:
        return err
    return SESSION.reset(seed=seed)


def api_agents() -> Dict[str, Any]:
    err = _ensure_session()
    if err:
        return err
    f = SESSION.full()
    return {"agents": f.get("agents", {})}


def api_agent(agent_id: str) -> Dict[str, Any]:
    err = _ensure_session()
    if err:
        return err
    f = SESSION.full()
    a = (f.get("agents") or {}).get(agent_id)
    if a is None:
        return {"error": f"agent '{agent_id}' not found", "available": list(f.get("agents", {}).keys())}
    return {"agent": a, "persona_graph": (f["graphs"]["persona"] or {}).get(agent_id, {}),
            "tick": f["tick"]}


def _internal_collection(f: Dict[str, Any], type_: str) -> List[Any]:
    internal, external = f.get("internal", {}), f.get("external", {})
    repo = internal.get("repo", {})
    table = {
        "task": internal.get("tasks", []), "doc": internal.get("docs", []),
        "file": internal.get("files", []), "message": internal.get("messages", []),
        "channel": internal.get("channels", []), "meeting": internal.get("meetings", []),
        "branch": repo.get("branches", []), "commit": repo.get("commits", []),
        "pr": repo.get("pull_requests", []), "experiment": internal.get("experiments", []),
        "result": internal.get("results", []), "sandbox_job": internal.get("sandbox", {}).get("jobs", []),
        "protocol": internal.get("protocols", []), "commitment": internal.get("commitments", []),
        "dispute": internal.get("disputes", []), "request": internal.get("requests", []),
        "cost_event": internal.get("cost_events", []), "ticket": internal.get("tickets", []),
        "artifact": internal.get("artifacts", []), "search": internal.get("searches", []),
        "external_profile": external.get("profiles", []), "external_post": external.get("posts", []),
        "signal": external.get("signals", []), "offer": external.get("offers", []),
    }
    return table.get(type_, [])


def api_objects(type_: str) -> Dict[str, Any]:
    err = _ensure_session()
    if err:
        return err
    return {"type": type_, "objects": _internal_collection(SESSION.full(), type_)}


def api_object(object_id: str) -> Dict[str, Any]:
    err = _ensure_session()
    if err:
        return err
    f = SESSION.full()
    for type_ in ("task", "doc", "file", "message", "meeting", "branch", "commit", "pr",
                  "experiment", "result", "protocol", "commitment", "dispute", "request",
                  "cost_event", "ticket", "artifact", "external_profile", "external_post"):
        for obj in _internal_collection(f, type_):
            oid = (obj.get("post_id") or obj.get("external_agent_id") or obj.get("message_id")
                   or obj.get("task_id") or obj.get("doc_id") or obj.get("object_id")
                   or obj.get("pr_id") or obj.get("commit_id") or obj.get("branch_id")
                   or obj.get("result_id") or obj.get("protocol_id") or obj.get("meeting_id")
                   or obj.get("commitment_id") or obj.get("dispute_id") or obj.get("request_id")
                   or obj.get("experiment_id") or obj.get("cost_event_id") or obj.get("ticket_id")
                   or obj.get("artifact_id"))
            if oid == object_id:
                edges = [e for e in f["graphs"]["event"]["edges"]
                         if e["src"] == object_id or e["dst"] == object_id]
                return {"type": type_, "object": obj, "linked_edges": edges}
    return {"error": f"object '{object_id}' not found"}


def api_section(section: str) -> Dict[str, Any]:
    err = _ensure_session()
    if err:
        return err
    f = SESSION.full()
    if section in ("messages", "channels", "meetings", "experiments", "protocols"):
        return {section: f["internal"].get(section, [])}
    if section == "repo":
        return {"repo": f["internal"].get("repo", {})}
    if section == "sandbox":
        return {"sandbox": f["internal"].get("sandbox", {}), "results": f["internal"].get("results", [])}
    if section == "external":
        return f["external"]
    if section == "event_graph":
        return f["graphs"]["event"]
    return {"error": f"unknown section '{section}'"}


def api_list_replays() -> Dict[str, Any]:
    out = []
    if REPLAY_DIR.is_dir():
        for p in sorted(REPLAY_DIR.glob("org_*.json")):
            meta = {}
            try:
                with open(p, "r", encoding="utf-8") as fh:
                    meta = (json.load(fh) or {}).get("meta", {})
            except Exception:
                meta = {}
            out.append({"name": p.stem, "file": p.name, "meta": meta})
    return {"replays": out}


def api_get_replay(name: str) -> Dict[str, Any]:
    safe = Path(name).name
    p = REPLAY_DIR / f"{safe}.json"
    if not p.is_file():
        return {"error": f"replay '{safe}' not found"}
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


def api_export_replay(name: str = "org_run") -> Dict[str, Any]:
    err = _ensure_session()
    if err:
        return err
    REPLAY_DIR.mkdir(parents=True, exist_ok=True)
    path = REPLAY_DIR / f"{name}.json"

    # Collect extra metadata
    extra_meta = {}

    # Add user_id if set
    user_id = getattr(SESSION, 'user_id', None)
    if user_id:
        extra_meta['user_id'] = user_id

    # Add HCI logs
    hci_logs = HUMAN.get_hci_logs()
    if hci_logs:
        extra_meta['hci_logs'] = hci_logs

    SESSION.save_replay(str(path), name=name, extra_meta=extra_meta)
    return {"saved": str(path), "name": name, "ticks": len(SESSION.buffer.frames),
            "user_id": user_id, "hci_events": len(hci_logs)}


def api_checkpoint_save(name: str = "") -> Dict[str, Any]:
    """Save the full world at the CURRENT tick (resumable). Default name embeds the tick."""
    err = _ensure_session()
    if err:
        return err
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    safe = Path(name or f"ckpt_t{int(SESSION.world.world_tick)}").name
    if not safe.endswith(".pkl"):
        safe += ".pkl"
    info = SESSION.save_checkpoint(str(CHECKPOINT_DIR / safe), meta={"source": "inspector"})
    return {"saved": info["path"], "name": safe, "tick": info["tick"], "size_mb": info["size_mb"]}


def api_checkpoint_list() -> Dict[str, Any]:
    """Cheap listing (no unpickle): name + tick (from filename) + size + mtime."""
    import datetime as _dt
    import re
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for p in sorted(CHECKPOINT_DIR.glob("*.pkl")):
        m = re.search(r"t(\d+)", p.stem)
        out.append({"name": p.name, "tick": int(m.group(1)) if m else None,
                    "size_mb": round(p.stat().st_size / 1e6, 2),
                    "modified": _dt.datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds")})
    return {"checkpoints": out}


def api_checkpoint_load(name: str) -> Dict[str, Any]:
    """Resume the live session from a saved checkpoint (by name in CHECKPOINT_DIR or a path)."""
    err = _ensure_session()
    if err:
        return err
    base = Path(name).name
    candidates = [CHECKPOINT_DIR / base]
    if not base.endswith(".pkl"):
        candidates.append(CHECKPOINT_DIR / (base + ".pkl"))
    candidates.append(Path(name))
    for p in candidates:
        if p.is_file():
            return SESSION.load_checkpoint(str(p))
    return {"error": f"checkpoint '{name}' not found"}


def _product_profile(world) -> Dict[str, Any]:
    """The product's CURRENT capabilities + active limitations, derived from the live
    artifacts + known gaps — so a human trial reflects the actual emergent product."""
    arts = getattr(world, "product_artifacts", {}) or {}
    caps = sorted({c for a in arts.values() for c in (getattr(a, "capabilities", []) or [])})
    gaps = [getattr(g, "description", "") for g in (getattr(world, "known_gaps", {}) or {}).values()
            if getattr(g, "status", "active") in ("active", "regressed")]
    prod = getattr(world, "product", None)
    return {"name": getattr(prod, "name", "the product") if prod else "the product",
            "stage": getattr(prod, "stage", "prototype") if prod else "prototype",
            "capabilities": caps, "known_limitations": [g for g in gaps if g][:12],
            "tick": int(getattr(world, "world_tick", 0))}


def api_product_try(query: str) -> Dict[str, Any]:
    """Human product terminal: run the company's product on a human query. The output is
    generated to reflect EXACTLY the product's current capabilities + limitations, so a
    human can experience/evaluate the emergent product honestly (frontend spec — product
    trial)."""
    err = _ensure_session()
    if err:
        return err
    w = SESSION.world
    profile = _product_profile(w)
    if not (query or "").strip():
        return {"error": "empty query", "profile": profile}
    client = getattr(w, "llm_client", None)
    if client is None:
        rep = (f"[{profile['name']} · {profile['stage']} · t{profile['tick']}] (no LLM client)\n\n"
               f"Query: {query}\n\nCapabilities: {', '.join(profile['capabilities']) or 'minimal'}\n"
               f"Known limitations: {'; '.join(profile['known_limitations']) or 'none recorded'}")
        return {"report": rep, "profile": profile, "caveats": profile["known_limitations"][:6],
                "confidence": 0.3, "llm": False}
    system = (f"You ARE the company's product — a research agent named {profile['name']} at stage "
              f"'{profile['stage']}'. Produce ONLY what a product with EXACTLY the listed capabilities "
              f"and limitations could produce, and honestly reflect the limitations (e.g. if claim-"
              f"evidence is not enforced, some claims may be unsupported; if eval metrics are stubs, say "
              f"results are not validated). Do not invent capabilities you don't have. Return JSON only.")
    user = ("CAPABILITIES:\n- " + ("\n- ".join(profile["capabilities"]) or "(barely functional)")
            + "\n\nKNOWN LIMITATIONS:\n- " + ("\n- ".join(profile["known_limitations"]) or "(none recorded)")
            + f"\n\nUSER REQUEST:\n{query}\n\n"
            + "Return {\"report\": str, \"caveats\": [str], \"confidence\": number}.")
    try:
        data = client.generate_json(system, user,
                                    {"report": "str", "caveats": "list", "confidence": "number"})
    except Exception as e:  # pragma: no cover
        data = {"report": f"(product run failed: {e})", "caveats": [], "confidence": 0.0}
    data = data if isinstance(data, dict) else {"report": str(data), "caveats": [], "confidence": 0.0}
    data["profile"] = profile
    data["llm"] = True
    return data


def api_product_feedback(rating: float = 0.0, comment: str = "", query: str = "") -> Dict[str, Any]:
    """Record a human's product evaluation as an external customer signal/ticket on the live
    world — so human trials feed the market loop (customer tickets advance the customers
    milestone / funding)."""
    err = _ensure_session()
    if err:
        return err
    from environments.org_env.backend.entities.economy import CustomerTicket
    w = SESSION.world
    tick = int(getattr(w, "world_tick", 0))
    tickets = w.__dict__.setdefault("tickets", {})
    tid = f"ticket_human_{len(tickets) + 1}"
    sev = "major" if rating and float(rating) <= 2 else ("minor" if rating and float(rating) >= 4 else "moderate")
    t = CustomerTicket(ticket_id=tid, customer_type="human_evaluator",
                       complaint_or_request=(comment or query or "human product evaluation"),
                       severity=sev, topic="product_evaluation", status="open")
    t.__dict__.update({"rating": float(rating or 0), "query": query, "comment": comment, "created_tick": tick})
    tickets[tid] = t
    if hasattr(w, "events"):
        w.events.append({"type": "external_signal_event", "subtype": "human_product_feedback",
                         "ticket_id": tid, "rating": float(rating or 0), "tick": tick})
    return {"ok": True, "ticket_id": tid, "rating": float(rating or 0), "tick": tick,
            "total_human_tickets": sum(1 for x in tickets.values()
                                       if getattr(x, "customer_type", "") == "human_evaluator")}


def api_product_export() -> Dict[str, Any]:
    """Materialize the in-world product repo to a real directory + run its smoke test
    (Product Materialization Layer). Returns the export manifest + smoke metrics."""
    err = _ensure_session()
    if err:
        return err
    from environments.org_env.product.materialize import export_and_smoke
    w = SESSION.world
    dest = str(BASE_DIR / "docs" / "product_exports" / "lanternscout")
    res = export_and_smoke(w, dest)
    res["tick"] = int(getattr(w, "world_tick", 0))
    return res


def seat_html() -> str:
    """The human member workspace. Built alongside the inspector as a second
    Vite entry; unlike the inspector it has no vanilla fallback, because a
    partial workspace would show a member the wrong organization."""
    page = APP_DIST / "seat.html"
    if page.is_file():
        return page.read_text(encoding="utf-8")
    return ("<h1>Seat UI not built</h1><p>Run <code>npm install &amp;&amp; npm run build</code>"
            " in environments/org_env/frontend/app</p>")


def liaison_html() -> str:
    """The integrated P3 liaison workspace, built as its own Vite entry.

    It intentionally has no inspector or P2 fallback: serving a different UI
    would silently change the HCI condition and could expose a richer surface.
    """
    page = APP_DIST / "liaison.html"
    if page.is_file():
        return page.read_text(encoding="utf-8")
    return ("<h1>Liaison UI not built</h1><p>Run <code>npm install &amp;&amp; npm run build</code>"
            " in environments/org_env/frontend/app</p>")


def setup_html() -> str:
    """The pack selection and configuration page. Shown before world initialization."""
    page = APP_DIST / "setup.html"
    if page.is_file():
        return page.read_text(encoding="utf-8")
    return ("<h1>Setup UI not built</h1><p>Run <code>npm install &amp;&amp; npm run build</code>"
            " in environments/org_env/frontend/app</p>")


def inspector_html() -> str:
    """Serve the React/Vite/React Flow build if present; else the legacy vanilla template."""
    idx = APP_DIST / "index.html"
    if idx.is_file():
        return idx.read_text(encoding="utf-8")
    if TEMPLATE.is_file():
        return TEMPLATE.read_text(encoding="utf-8")
    return "<h1>OrgEnv inspector not built — run `npm install && npm run build` in environments/org_env/frontend/app</h1>"


def legacy_inspector_html() -> str:
    if TEMPLATE.is_file():
        return TEMPLATE.read_text(encoding="utf-8")
    return "<h1>org_inspector.html not found</h1>"


# --------------------------------------------------------------------------- #
# FastAPI app (lazy import so this module loads without fastapi for tests)
# --------------------------------------------------------------------------- #
def build_app():
    from fastapi import FastAPI, Body
    from fastapi.responses import HTMLResponse
    from starlette.concurrency import run_in_threadpool

    app = FastAPI(title="Relic OrgEnv Live Inspector")

    # serve the built React app's hashed assets (index.html uses base "/org/app/")
    if APP_DIST.is_dir():
        from fastapi.staticfiles import StaticFiles
        app.mount("/org/app", StaticFiles(directory=str(APP_DIST), html=True), name="org_app")

    @app.get("/org/inspector", response_class=HTMLResponse)
    async def inspector():
        return inspector_html()

    @app.get("/org/legacy", response_class=HTMLResponse)
    async def inspector_legacy():
        return legacy_inspector_html()

    @app.get("/org/seat", response_class=HTMLResponse)
    async def seat():
        return seat_html()

    @app.get("/org/liaison", response_class=HTMLResponse)
    async def liaison():
        return liaison_html()

    @app.get("/org/setup", response_class=HTMLResponse)
    async def setup():
        # If world already initialized, redirect to seat
        if SESSION is not None:
            from fastapi.responses import RedirectResponse
            return RedirectResponse("/org/seat")
        return setup_html()

    @app.get("/", response_class=HTMLResponse)
    async def root():
        # Redirect to setup if world not initialized, else to seat
        from fastapi.responses import RedirectResponse
        if SESSION is None:
            return RedirectResponse("/org/setup")
        return RedirectResponse("/org/seat")

    @app.get("/api/org/packs")
    async def packs():
        return api_packs()

    @app.post("/api/org/init")
    async def init_world(payload: dict = Body(default={})):
        return api_init_world(
            pack_id=str(payload.get("pack", "")),
            warmup=int(payload.get("warmup", 36)),
            clock_speed=float(payload.get("clock_speed", 15.0)),
            user_id=payload.get("user_id"),  # Pass user_id
            start_runtime=_optional_bool(payload.get("start_runtime")),
        )

    @app.get("/api/org/lived/full")
    async def lived_full():
        return api_full()

    @app.get("/api/org/lived/frames")
    async def lived_frames(since: int = -1):
        return api_frames(since)

    @app.get("/api/org/lived/state")
    async def lived_state():
        return api_state()

    @app.get("/api/org/lived/replays")
    async def lived_replays():
        return api_list_replays()

    @app.get("/api/org/lived/replay/{name}")
    async def lived_replay(name: str):
        return api_get_replay(name)

    @app.post("/api/org/sim/step")
    async def sim_step(payload: dict = Body(default={})):
        return api_sim_step(int(payload.get("n", 1)))

    @app.post("/api/org/sim/run_ticks")
    async def sim_run(payload: dict = Body(default={})):
        return api_sim_run_ticks(int(payload.get("n", 24)))

    @app.post("/api/org/sim/pause")
    async def sim_pause():
        return api_sim_pause()

    @app.post("/api/org/sim/reset")
    async def sim_reset(payload: dict = Body(default={})):
        return api_sim_reset(payload.get("seed"))

    @app.post("/api/org/sim/export")
    async def sim_export(payload: dict = Body(default={})):
        return api_export_replay(str(payload.get("name", "org_run")))

    @app.post("/api/org/sim/checkpoint/save")
    async def ckpt_save(payload: dict = Body(default={})):
        return api_checkpoint_save(str(payload.get("name", "")))

    @app.get("/api/org/sim/checkpoints")
    async def ckpt_list():
        return api_checkpoint_list()

    @app.post("/api/org/sim/checkpoint/load")
    async def ckpt_load(payload: dict = Body(default={})):
        return api_checkpoint_load(str(payload.get("name", "")))

    @app.post("/api/org/product/try")
    async def product_try(payload: dict = Body(default={})):
        return api_product_try(str(payload.get("query", "")))

    @app.post("/api/org/product/feedback")
    async def product_feedback(payload: dict = Body(default={})):
        return api_product_feedback(float(payload.get("rating", 0) or 0),
                                    str(payload.get("comment", "")), str(payload.get("query", "")))

    @app.post("/api/org/product/export")
    async def product_export():
        return api_product_export()

    @app.get("/api/org/agents")
    async def agents():
        return api_agents()

    @app.get("/api/org/agents/{agent_id}")
    async def agent(agent_id: str):
        return api_agent(agent_id)

    @app.get("/api/org/objects")
    async def objects(type: str = "task"):
        return api_objects(type)

    @app.get("/api/org/objects/{object_id}")
    async def obj(object_id: str):
        return api_object(object_id)

    # -- human seats (HCI). Must be registered before the /api/org/{section}
    # catch-all below, which would otherwise swallow every one of these.
    @app.get("/api/org/human/members")
    async def human_members():
        return HUMAN.members()

    # -- P3 liaison façade.  Keep the browser in this namespace: it is a
    # filtered organization-as-a-service contract, not the rich P2 seat view.
    @app.post("/api/org/liaison/session")
    async def liaison_session(payload: dict = Body(default={})):
        # P3 has one deliberate human perspective: Victor.  The browser may
        # resume that token but cannot request an arbitrary organization seat.
        # Claiming or resuming a seat takes the same world lock as a live tick.
        # Keep that wait off the ASGI loop so another tab can still read runtime
        # status or pause the simulation while the current tick finishes.
        return await run_in_threadpool(
            LIAISON.session, str(payload.get("token", "")))

    @app.post("/api/org/liaison/release")
    async def liaison_release(payload: dict = Body(default={})):
        return await run_in_threadpool(
            LIAISON.release, str(payload.get("token", "")))

    @app.get("/api/org/liaison/state")
    async def liaison_state(token: str = "", since: int = 0):
        # Building the projection can resolve source evidence against the
        # current world and briefly wait for a live tick's lock.  Never let
        # that wait block pause/resume or the lightweight runtime heartbeat.
        return await run_in_threadpool(LIAISON.state, token, since)

    @app.get("/api/org/liaison/organization")
    async def liaison_organization(token: str = ""):
        return LIAISON.organization(token)

    @app.get("/api/org/liaison/evidence")
    async def liaison_evidence(token: str = "", ref: str = ""):
        return LIAISON.evidence(token, ref)

    @app.get("/api/org/liaison/trace")
    async def liaison_trace(token: str = "", ref: str = ""):
        return LIAISON.trace(token, ref)

    @app.get("/api/org/liaison/resource")
    async def liaison_resource(token: str = "", ref: str = "", section: str = "overview"):
        # The browser supplies only a token-bound ri_ handle.  Liaison resolves
        # it against the current P2 seat view before asking the runtime adapter
        # for any raw, read-only resource material.
        return await run_in_threadpool(
            LIAISON.resource, token, ref, section)

    @app.post("/api/org/liaison/ask")
    async def liaison_ask(payload: dict = Body(default={})):
        # Semantic routing performs a real provider call before an ordinary
        # request is handed to the background working-agent queue.  Running it
        # on the ASGI event loop would freeze state polling, pause/resume, and
        # every other browser tab for the entire model call.
        return await run_in_threadpool(
            LIAISON.ask,
            str(payload.get("token", "")),
            str(payload.get("text", "")),
            reply_to=str(payload.get("reply_to") or ""),
            thread_id=str(payload.get("thread_id") or ""),
            decision_context=payload.get("decision_context"),
        )

    @app.post("/api/org/liaison/execution/start")
    async def liaison_execution_start(payload: dict = Body(default={})):
        return LIAISON.execution_start(str(payload.get("token", "")),
                                       str(payload.get("job_id", "")))

    @app.post("/api/org/liaison/request/cancel")
    async def liaison_request_cancel(payload: dict = Body(default={})):
        return LIAISON.cancel_request(str(payload.get("token", "")))

    @app.post("/api/org/liaison/request/retry")
    async def liaison_request_retry(payload: dict = Body(default={})):
        return LIAISON.retry_request(
            str(payload.get("token", "")),
            thread_id=str(payload.get("thread_id") or ""),
        )

    @app.post("/api/org/liaison/execution/cancel")
    async def liaison_execution_cancel(payload: dict = Body(default={})):
        return LIAISON.execution_cancel(str(payload.get("token", "")),
                                        str(payload.get("job_id", "")))

    @app.post("/api/org/liaison/prepare")
    async def liaison_prepare(payload: dict = Body(default={})):
        return LIAISON.prepare(str(payload.get("token", "")),
                               str(payload.get("action_type", "")),
                               dict(payload.get("params") or {}),
                               str(payload.get("rationale", "")))

    @app.post("/api/org/liaison/confirm")
    async def liaison_confirm(payload: dict = Body(default={})):
        # A confirmed action may run the real public test suite or another
        # blocking OrgWorld handler.  Keep the live UI responsive while the
        # gateway performs that work.
        return await run_in_threadpool(
            LIAISON.confirm,
            str(payload.get("token", "")),
            str(payload.get("draft_id", "")),
        )

    @app.post("/api/org/liaison/confirm-meeting-plan")
    async def liaison_confirm_meeting_plan(payload: dict = Body(default={})):
        return await run_in_threadpool(
            LIAISON.confirm_meeting_plan,
            str(payload.get("token", "")),
            str(payload.get("plan_id", "")),
        )

    @app.post("/api/org/liaison/discard")
    async def liaison_discard(payload: dict = Body(default={})):
        return LIAISON.discard(str(payload.get("token", "")), str(payload.get("draft_id", "")))

    @app.post("/api/org/liaison/discard-meeting-plan")
    async def liaison_discard_meeting_plan(payload: dict = Body(default={})):
        return LIAISON.discard_meeting_plan(
            str(payload.get("token", "")), str(payload.get("plan_id", "")))

    @app.get("/api/org/liaison/runtime")
    async def liaison_runtime():
        return _runtime_status(LIAISON.runtime_status())

    @app.post("/api/org/liaison/runtime/start")
    async def liaison_runtime_start(payload: dict = Body(default={})):
        return _runtime_status(LIAISON.runtime_start(payload.get("seconds_per_tick")))

    @app.post("/api/org/liaison/runtime/pause")
    async def liaison_runtime_pause():
        return _runtime_status(LIAISON.runtime_pause())

    @app.post("/api/org/human/seats/claim")
    async def human_claim(payload: dict = Body(default={})):
        return HUMAN.claim(str(payload.get("agent_id", "")),
                           str(payload.get("display_name", "")))

    @app.post("/api/org/human/seats/release")
    async def human_release(payload: dict = Body(default={})):
        return HUMAN.release(str(payload.get("token", "")))

    @app.get("/api/org/human/view")
    async def human_view(token: str = "", since_version: int = -1):
        payload = HUMAN.view(token, since_version)
        if isinstance(payload.get("runtime"), dict):
            payload["runtime"] = _runtime_status(payload["runtime"])
        return payload

    @app.get("/api/org/human/brief")
    async def human_brief(token: str = ""):
        return HUMAN.brief(token)

    @app.get("/api/org/human/offers")
    async def human_offers(token: str = "", object_id: str = ""):
        return HUMAN.offers(token, object_id)

    @app.post("/api/org/human/act")
    async def human_act(payload: dict = Body(default={})):
        return HUMAN.act(str(payload.get("token", "")),
                         str(payload.get("action_type", "")),
                         payload.get("params") or {},
                         str(payload.get("execution_mode", "direct")))

    @app.post("/api/org/human/agent/send")
    async def human_agent_send(payload: dict = Body(default={})):
        return HUMAN.agent_send(str(payload.get("token", "")),
                                str(payload.get("text", "")))

    @app.get("/api/org/human/agent")
    async def human_agent_state(token: str = "", since: int = 0):
        return HUMAN.agent_state(token, since)

    @app.post("/api/org/human/agent/confirm")
    async def human_agent_confirm(payload: dict = Body(default={})):
        return HUMAN.agent_confirm(str(payload.get("token", "")),
                                   str(payload.get("draft_id", "")))

    @app.post("/api/org/human/agent/discard")
    async def human_agent_discard(payload: dict = Body(default={})):
        return HUMAN.agent_discard(str(payload.get("token", "")),
                                   str(payload.get("draft_id", "")))

    @app.post("/api/org/human/agent/execution/start")
    async def human_execution_start(payload: dict = Body(default={})):
        return HUMAN.execution_start(str(payload.get("token", "")),
                                     str(payload.get("job_id", "")))

    @app.post("/api/org/human/agent/execution/cancel")
    async def human_execution_cancel(payload: dict = Body(default={})):
        return HUMAN.execution_cancel(str(payload.get("token", "")),
                                      str(payload.get("job_id", "")))

    @app.get("/api/org/human/runtime")
    async def human_runtime():
        return _runtime_status(HUMAN.runtime_status())

    @app.post("/api/org/human/runtime/start")
    async def human_runtime_start(payload: dict = Body(default={})):
        return _runtime_status(HUMAN.runtime_start(payload.get("seconds_per_tick")))

    @app.post("/api/org/human/runtime/pause")
    async def human_runtime_pause():
        return _runtime_status(HUMAN.runtime_pause())

    @app.get("/api/org/{section}")
    async def section(section: str):
        return api_section(section)

    return app


def main() -> None:
    import socket
    import threading
    import webbrowser

    import uvicorn

    def free_port(start=8100):
        for p in range(start, start + 200):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(("127.0.0.1", p)) != 0:
                    return p
        return start

    import os
    # SESSION is now lazily initialized after pack selection in /org/setup
    # No longer initialized here at startup
    print("Waiting for pack selection at /org/setup...")

    # Loopback by default. Human seats are the reason to bind wider: several
    # people hold different seats in one world from their own machines. There is
    # no auth beyond the per-seat token, so only open this on a trusted network.
    host = os.environ.get("ORG_HOST", "127.0.0.1")
    requested = os.environ.get("ORG_PORT", "")
    port = int(requested) if requested.isdigit() else free_port()
    reachable = "127.0.0.1" if host in ("127.0.0.1", "localhost") else host
    if host == "0.0.0.0":
        reachable = _lan_address() or host

    if os.environ.get("ORG_OPEN_BROWSER", "0").lower() in ("1", "true", "yes"):
        threading.Timer(1.5, lambda: webbrowser.open(
            f"http://{reachable}:{port}/org/setup")).start()
    print(f"Pack selection        → http://{reachable}:{port}/org/setup")
    print(f"OrgEnv Live Inspector → http://{reachable}:{port}/org/inspector")
    print(f"Member workspace      → http://{reachable}:{port}/org/seat")
    print(f"Integrated liaison    → http://{reachable}:{port}/org/liaison")
    if host == "0.0.0.0":
        print("Bound to all interfaces: anyone who can reach this port can claim a seat.")
    uvicorn.run(build_app(), host=host, port=port)


def _lan_address() -> str:
    """Best-effort address others can reach us on, for the printed URL."""
    import socket

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))       # no packet is sent; just picks a route
            return s.getsockname()[0]
    except OSError:
        return ""


if __name__ == "__main__":
    main()
