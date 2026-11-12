"""A label write against a deprecated or merged-away class is refused with 400, at the registry's
one validation point (``validate_id``), on every human label route."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from _class_registry_fake import use_registry

from curation.query_fakes import QueryFakeOpenSearch
from curation.test_crops_undo_exclude import ITEMS, _proposal_doc
from src.core.db_session import get_db


if TYPE_CHECKING:
    from _class_registry_fake import InMemoryRegistry


BASE = '/curation/projects/default'


@pytest.fixture
def deprecated_ids(registry: InMemoryRegistry, ids: dict[str, int]) -> dict[str, int]:
    use_registry(registry)
    registry.deprecate(ids['gadget'])
    registry.merge(ids['gizmo'], ids['widget'])
    return {'deprecated': ids['gadget'], 'merged_away': ids['gizmo'], 'active': ids['widget']}


@pytest.mark.parametrize('which', ['deprecated', 'merged_away'])
def test_single_label_is_refused(client_for, deprecated_ids, which: str) -> None:
    fake = QueryFakeOpenSearch({ITEMS: {'c1': _proposal_doc('c1')}})
    before = dict(fake.docs(ITEMS)['c1'])
    r = client_for(fake).put(f'{BASE}/crops/c1/label', json={'class_id': deprecated_ids[which]})
    assert r.status_code == 400, r.text
    assert 'unknown class_id' in r.text
    assert fake.docs(ITEMS)['c1'] == before


@pytest.mark.parametrize('which', ['deprecated', 'merged_away'])
def test_batch_label_is_refused(client_for, deprecated_ids, which: str) -> None:
    fake = QueryFakeOpenSearch({ITEMS: {'c1': _proposal_doc('c1')}})
    client = client_for(fake)
    client.app.dependency_overrides[get_db] = lambda: None  # refused before any session use
    r = client.put(
        f'{BASE}/crops/batch_label',
        json={'crop_ids': ['c1'], 'class_id': deprecated_ids[which]},
    )
    assert r.status_code == 400, r.text
    assert 'unknown class_id' in r.text


def test_an_active_class_is_still_accepted(client_for, deprecated_ids) -> None:
    fake = QueryFakeOpenSearch({ITEMS: {'c1': _proposal_doc('c1')}})
    r = client_for(fake).put(f'{BASE}/crops/c1/label', json={'class_id': deprecated_ids['active']})
    assert r.status_code == 200, r.text
