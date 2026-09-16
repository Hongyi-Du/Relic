"""The P2/P3 workspaces are served, and neither is the inspector.

Two builds come out of one Vite project and both are HTML served by FastAPI, so
it is easy for them to silently swap: a wrong `base`, a wrong entry name, and
/org/seat would quietly hand a member the omniscient debugger.

Run:  PYTHONPATH="." python tests/org_env/test_human_seat_ui.py
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from environments.org_env.backend import main

ASSET = re.compile(r'/org/app/assets/([\w.-]+\.(?:js|css))')


def _built() -> bool:
    if (main.APP_DIST / "seat.html").is_file():
        return True
    print("  (skip: frontend not built — run npm run build in frontend/app)")
    return False


def test_the_seat_page_loads_its_own_bundle_not_the_inspectors():
    if not _built():
        return
    html = main.seat_html()
    assets = ASSET.findall(html)

    assert assets, f"no built assets referenced: {html[:200]}"
    assert any(a.startswith("seat") for a in assets), assets
    assert not any(a.startswith("inspector") for a in assets), (
        f"the member workspace is loading the omniscient inspector: {assets}")
    assert "<title>Relic · P2 Transparent" in html


def test_the_inspector_still_loads_its_own_bundle():
    if not _built():
        return
    assets = ASSET.findall(main.inspector_html())
    assert any(a.startswith("inspector") for a in assets), assets
    assert not any(a.startswith("seat") for a in assets), assets


def test_the_liaison_page_loads_only_its_p3_bundle():
    if not _built():
        return
    html = main.liaison_html()
    assets = ASSET.findall(html)
    assert any(a.startswith("liaison") for a in assets), assets
    assert not any(a.startswith("seat") or a.startswith("inspector") for a in assets), assets
    assert "Organization" in html


def test_the_p3_api_exposes_a_fixed_session_not_a_seat_picker():
    source = Path(main.__file__).read_text(encoding="utf-8")
    assert '"/api/org/liaison/session"' in source
    assert '"/api/org/liaison/claim"' not in source
    assert '"/api/org/liaison/members"' not in source


def test_an_unbuilt_frontend_says_so_instead_of_serving_half_a_workspace():
    """A partial workspace would show a member the wrong organization, so there
    is deliberately no vanilla fallback for this page."""
    real, main.APP_DIST = main.APP_DIST, Path("/nonexistent-build")
    try:
        html = main.seat_html()
        assert "not built" in html.lower()
        assert "npm run build" in html
        liaison = main.liaison_html()
        assert "liaison ui not built" in liaison.lower()
        assert "npm run build" in liaison
    finally:
        main.APP_DIST = real


def test_routes_serve_the_two_pages_and_the_assets():
    try:
        from fastapi.testclient import TestClient
    except Exception:
        print("  (skip HTTP test: fastapi not installed)")
        return
    if not _built():
        return

    client = TestClient(main.build_app())
    try:
        page = client.get("/org/seat")
        assert page.status_code == 200
        bundle = next(a for a in ASSET.findall(page.text) if a.endswith(".js"))
        assert client.get(f"/org/app/assets/{bundle}").status_code == 200

        assert client.get("/org/inspector").status_code == 200
        liaison = client.get("/org/liaison")
        assert liaison.status_code == 200
        assert any(a.startswith("liaison") for a in ASSET.findall(liaison.text))
    finally:
        main.HUMAN.shutdown()


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print("  ok ", fn.__name__)
    print(f"All {len(fns)} seat UI tests passed!")
