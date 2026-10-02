"""``GET /active`` for packs and profiles publish a typed response schema."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


SPEC = json.loads(
    (Path(__file__).resolve().parents[2] / 'contracts/openapi/curation.json').read_text()
)
PREFIX = '/curation/projects/{project}'


@pytest.mark.parametrize(
    ('path', 'method'),
    [
        ('/prompt_packs/active', 'get'),
        ('/prompt_packs/active/rollback', 'post'),
        ('/region_profiles/active', 'get'),
        ('/region_profiles/active/rollback', 'post'),
        ('/region_profiles/deactivate', 'post'),
    ],
)
def test_active_routes_have_a_typed_ok_schema(path: str, method: str) -> None:
    schema = SPEC['paths'][PREFIX + path][method]['responses']['200']['content'][
        'application/json'
    ]['schema']
    assert schema == {'$ref': '#/components/schemas/ActiveConfigResponse'}
