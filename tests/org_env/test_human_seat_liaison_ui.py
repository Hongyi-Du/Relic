"""Static source guardrails for the P2 transparent liaison surface.

These checks intentionally do not build or serve the frontend.  They protect
the interaction boundary while the backend brief shape is still evolving:
joining lands on a liaison, organization state stays visible, and consequential
requests cannot be routed without an explicit confirmation affordance.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2] / "environments" / "org_env" / "frontend" / "app" / "src" / "seat"


def _source(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def test_p2_seat_defaults_to_transparent_liaison_without_a_second_composer():
    seat = _source("SeatApp.tsx")
    assert "<LiaisonPanel" in seat
    assert 'aria-label="Experience version"' in seat
    assert "P2 Transparent" in seat
    assert "P3 Secretary" in seat
    assert 'window.location.assign("/org/liaison")' in seat
    assert "<OrgChat" not in seat
    assert "<WorkingAgent" not in seat


def test_liaison_explains_its_boundary_and_confirms_consequential_routes():
    liaison = _source("LiaisonPanel.tsx")
    assert "Interpret · summarize · route" in liaison
    assert "Consequential actions always need your confirmation" in liaison
    assert "Confirm and route" in liaison
    assert "api.agentSend" in liaison
    assert "api.agentConfirm" in liaison
    assert "api.act" not in liaison
    assert "Private agent interiors stay private" in liaison


def test_organization_surface_keeps_brief_agents_workstreams_and_drilldown():
    overview = _source("OrganizationOverview.tsx")
    assert "organization_brief" in overview
    assert "Workstreams" in overview
    assert "Agents" in overview
    assert "open(objectIdFor(item), item)" in overview


def test_seat_view_accepts_a_top_level_organization_brief():
    api = _source("api.ts")
    assert "organization_brief?: OrganizationBrief" in api
    assert "brief: (token: string)" in api


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
