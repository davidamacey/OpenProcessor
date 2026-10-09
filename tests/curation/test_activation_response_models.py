"""Every activation-state route declares a typed response, so the committed
OpenAPI contract describes it (the get/rollback/deactivate routes already did;
``/activate`` and ``/active/impact`` returned bare dicts)."""

from __future__ import annotations

from typing import Any

from _route_helpers import api_routes

from src.main import create_app


def test_activate_and_impact_routes_declare_a_response_model() -> None:
    wanted = (
        '/prompt_packs/{name}/activate',
        '/region_profiles/{name}/activate',
        '/region_profiles/active/impact',
    )
    routes = api_routes(create_app())
    untyped = [
        r.path
        for r in routes
        if r.path.removeprefix('/v1').endswith(wanted) and r.response_model in (None, Any)
    ]
    covered = {w for w in wanted for r in routes if r.path.endswith(w)}
    assert untyped == []
    assert covered == set(wanted)
