"""The status route serves each level's ``stale_reasons`` in the committed contract Cropwright reads."""

from __future__ import annotations

import json
from pathlib import Path


CONTRACT = Path(__file__).resolve().parents[2] / 'contracts' / 'openapi' / 'curation.json'


def _spec() -> dict:
    return json.loads(CONTRACT.read_text(encoding='utf-8'))


def test_the_status_route_answers_with_the_status_model() -> None:
    op = _spec()['paths']['/curation/projects/{project}/near_duplicates/status']['get']
    ref = op['responses']['200']['content']['application/json']['schema']['$ref']
    assert ref.endswith('/NearDupStatus')


def test_each_level_names_why_it_is_stale() -> None:
    schemas = _spec()['components']['schemas']
    level = schemas['NearDupLevelStatus']['properties']
    assert set(level['stale_reasons']['items']['enum']) == {'pending', 'rebuild_required', 'no_run'}
    assert 'pending_count' in level
    assert {'image', 'item'} == set(level['level']['enum'])
