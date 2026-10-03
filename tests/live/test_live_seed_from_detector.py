"""Live: the ingest config's detector block and seeding the registry from it by name."""

from __future__ import annotations

from typing import Any

import pytest


pytestmark = pytest.mark.live


def _detector(api_client: Any) -> dict[str, Any] | None:
    resp = api_client.get('/ingest/config')
    resp.raise_for_status()
    return resp.json()['detector']


def test_seed_matches_the_detector_block(api_client: Any) -> None:
    detector = _detector(api_client)
    if detector is None:
        assert api_client.post('/classes/seed_from_detector', json={}).status_code == 503
        return
    assert detector['n_labels'] == len(detector['labels'])

    dry = api_client.post('/classes/seed_from_detector', json={})
    dry.raise_for_status()
    assert dry.json()['dry_run'] is True
    before = api_client.get('/classes').json()['classes']

    applied = api_client.post('/classes/seed_from_detector', json={'dry_run': False})
    applied.raise_for_status()
    body = applied.json()
    slugs = {label['slug'] for label in detector['labels'] if label['slug']}
    names = {c['class_name'] for c in api_client.get('/classes').json()['classes']}
    assert slugs <= names
    assert len(names) == len(before) + len(body['created'])

    again = api_client.post('/classes/seed_from_detector', json={'dry_run': False}).json()
    assert again['created'] == []
