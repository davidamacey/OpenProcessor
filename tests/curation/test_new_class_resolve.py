"""``POST /review/new_class_proposals/resolve`` — bulk-resolve every
``vlm_new_class_pending`` item proposing a given term, in one call.

The summary endpoint (``GET /review/new_class_proposals/summary``) only
ever returns up to 20 sample crop ids per term; Cropwright resolves a term
by relabeling those samples via ``PUT /crops/batch_label``, silently
leaving the rest of a large cohort still pending. This route selects and
writes *every* matching item server-side.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from curation.query_fakes import QueryFakeOpenSearch
from src.clients.curation_opensearch import ClassRegistry
from src.config import get_curation_config


ITEMS = get_curation_config().items_index

CLASS_STATE_FIELDS = (
    'class_id',
    'class_name',
    'class_source',
    'label_source',
    'confidence',
    'class_validated',
    'cluster_id',
    'cluster_subid',
)


@pytest.fixture
def registry(tmp_path: Any) -> ClassRegistry:
    reg = ClassRegistry(path=tmp_path / 'class_registry.json')
    reg.add_class('sedan', group='vehicle')
    return reg


def _client(fake: Any, registry: ClassRegistry, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    monkeypatch.setattr('src.routers.curation._ensure_indexes', AsyncMock(return_value=None))
    monkeypatch.setattr('src.routers.curation.get_class_registry', lambda: registry)
    app = FastAPI()
    app.include_router(curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake
    return TestClient(app)


def _pending_doc(crop_id: str, label: str, **extra: Any) -> dict[str, Any]:
    return {
        'crop_id': crop_id,
        'class_source': 'vlm_new_class_pending',
        'vlm_proposed_class': label,
        'needs_new_class': True,
        **extra,
    }


def _class_state(doc: dict[str, Any]) -> dict[str, Any]:
    state = {f: doc.get(f) for f in CLASS_STATE_FIELDS}
    state['class_validated'] = bool(state['class_validated'])
    return state


# =============================================================================
# class_id path
# =============================================================================


def test_resolve_with_class_id_writes_every_matching_pending_item(
    registry: ClassRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    class_id = registry.load().classes[0].class_id  # 'sedan'
    n = 25
    docs = {f'p{i}': _pending_doc(f'p{i}', 'sidecar') for i in range(n)}
    # A different term — must not be touched.
    docs['other'] = _pending_doc('other', 'kayak')
    # Already validated for the same term — must not be touched.
    docs['done'] = _pending_doc('done', 'sidecar', class_validated=True, class_id=99)
    fake = QueryFakeOpenSearch({ITEMS: docs})
    client = _client(fake, registry, monkeypatch)

    r = client.post(
        '/curation/review/new_class_proposals/resolve',
        json={'label': 'sidecar', 'class_id': class_id},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body['matched'] == n
    assert body['updated'] == n
    assert body['created'] is False
    assert body['class_id'] == class_id
    assert body['class_name'] == 'sedan'
    assert sorted(body['updated_ids']) == sorted(f'p{i}' for i in range(n))
    assert body['conflicts'] == []
    assert body['skipped'] == []

    for i in range(n):
        doc = fake.docs(ITEMS)[f'p{i}']
        assert doc['class_source'] == 'human'
        assert doc['class_validated'] is True
        assert doc['class_id'] == class_id
        assert doc['cluster_id'] == class_id
        assert doc['needs_new_class'] is False

    # Untouched.
    assert fake.docs(ITEMS)['other']['class_source'] == 'vlm_new_class_pending'
    assert fake.docs(ITEMS)['done']['class_id'] == 99


def test_resolve_unknown_class_id_is_400(
    registry: ClassRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = QueryFakeOpenSearch({ITEMS: {'p1': _pending_doc('p1', 'sidecar')}})
    client = _client(fake, registry, monkeypatch)
    r = client.post(
        '/curation/review/new_class_proposals/resolve',
        json={'label': 'sidecar', 'class_id': 999999},
    )
    assert r.status_code == 400, r.text
    assert fake.docs(ITEMS)['p1']['class_source'] == 'vlm_new_class_pending'


# =============================================================================
# create path
# =============================================================================


def test_resolve_with_create_registers_class_and_writes_items(
    registry: ClassRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    docs = {f'p{i}': _pending_doc(f'p{i}', 'sidecar') for i in range(3)}
    fake = QueryFakeOpenSearch({ITEMS: docs})
    client = _client(fake, registry, monkeypatch)

    r = client.post(
        '/curation/review/new_class_proposals/resolve',
        json={
            'label': 'sidecar',
            'create': {'class_name': 'sidecar', 'group': 'motorcycle', 'notes': None},
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body['created'] is True
    assert body['class_name'] == 'sidecar'
    new_id = body['class_id']
    assert new_id is not None

    entry = registry.get(new_id)
    assert entry is not None
    assert entry.class_name == 'sidecar'
    assert entry.group == 'motorcycle'

    assert body['matched'] == 3
    assert body['updated'] == 3
    for i in range(3):
        assert fake.docs(ITEMS)[f'p{i}']['class_id'] == new_id


def test_resolve_create_with_zero_matches_still_creates_the_class(
    registry: ClassRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = QueryFakeOpenSearch({ITEMS: {}})
    client = _client(fake, registry, monkeypatch)
    r = client.post(
        '/curation/review/new_class_proposals/resolve',
        json={'label': 'sidecar', 'create': {'class_name': 'sidecar'}},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body['created'] is True
    assert body['matched'] == 0
    assert body['updated'] == 0
    assert registry.get(body['class_id']) is not None


def test_resolve_create_duplicate_name_is_409_and_writes_nothing(
    registry: ClassRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    docs = {'p1': _pending_doc('p1', 'sedan')}
    fake = QueryFakeOpenSearch({ITEMS: docs})
    client = _client(fake, registry, monkeypatch)
    r = client.post(
        '/curation/review/new_class_proposals/resolve',
        json={'label': 'sedan', 'create': {'class_name': 'sedan'}},
    )
    assert r.status_code == 409, r.text
    assert fake.docs(ITEMS)['p1']['class_source'] == 'vlm_new_class_pending'
    # Registry unchanged — still just the seeded 'sedan'.
    assert len(registry.load().classes) == 1


def test_resolve_create_bad_slug_is_422(
    registry: ClassRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = QueryFakeOpenSearch({ITEMS: {}})
    client = _client(fake, registry, monkeypatch)
    r = client.post(
        '/curation/review/new_class_proposals/resolve',
        json={'label': 'sidecar', 'create': {'class_name': 'Not A Slug!'}},
    )
    assert r.status_code == 422, r.text
    assert len(registry.load().classes) == 1


# =============================================================================
# class_id / create mutual exclusivity
# =============================================================================


def test_resolve_neither_class_id_nor_create_is_422(
    registry: ClassRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = QueryFakeOpenSearch({ITEMS: {}})
    client = _client(fake, registry, monkeypatch)
    r = client.post('/curation/review/new_class_proposals/resolve', json={'label': 'sidecar'})
    assert r.status_code == 422, r.text


def test_resolve_both_class_id_and_create_is_422(
    registry: ClassRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    class_id = registry.load().classes[0].class_id
    fake = QueryFakeOpenSearch({ITEMS: {}})
    client = _client(fake, registry, monkeypatch)
    r = client.post(
        '/curation/review/new_class_proposals/resolve',
        json={'label': 'sidecar', 'class_id': class_id, 'create': {'class_name': 'sidecar'}},
    )
    assert r.status_code == 422, r.text


# =============================================================================
# dry_run
# =============================================================================


def test_resolve_dry_run_writes_and_creates_nothing(
    registry: ClassRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    docs = {f'p{i}': _pending_doc(f'p{i}', 'sidecar') for i in range(5)}
    fake = QueryFakeOpenSearch({ITEMS: docs})
    client = _client(fake, registry, monkeypatch)

    r = client.post(
        '/curation/review/new_class_proposals/resolve',
        params={'dry_run': 'true'},
        json={'label': 'sidecar', 'create': {'class_name': 'sidecar'}},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body['matched'] == 5
    assert body['updated'] == 0
    assert body['updated_ids'] == []
    assert body['created'] is False
    assert body['class_id'] is None
    # Registry unchanged.
    assert len(registry.load().classes) == 1
    for i in range(5):
        assert fake.docs(ITEMS)[f'p{i}']['class_source'] == 'vlm_new_class_pending'


def test_resolve_dry_run_with_class_id_writes_nothing(
    registry: ClassRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    class_id = registry.load().classes[0].class_id
    docs = {'p1': _pending_doc('p1', 'sidecar')}
    fake = QueryFakeOpenSearch({ITEMS: docs})
    client = _client(fake, registry, monkeypatch)
    r = client.post(
        '/curation/review/new_class_proposals/resolve',
        params={'dry_run': 'true'},
        json={'label': 'sidecar', 'class_id': class_id},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body['matched'] == 1
    assert body['updated'] == 0
    assert fake.docs(ITEMS)['p1']['class_source'] == 'vlm_new_class_pending'


# =============================================================================
# State-changed-between-select-and-write -> skipped
# =============================================================================


def test_resolve_skips_item_whose_state_changed_before_write(
    registry: ClassRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    class_id = registry.load().classes[0].class_id
    docs = {
        'p1': _pending_doc('p1', 'sidecar'),
        'p2': _pending_doc('p2', 'sidecar'),
    }
    fake = QueryFakeOpenSearch({ITEMS: docs})

    real_get = fake.get

    async def _get_and_flip(*, index: str, id: str, **kw: Any) -> dict[str, Any]:  # noqa: A002
        if id == 'p2':
            # Simulate a concurrent write that resolved p2 to something
            # else between selection and this item's write — flip before
            # the OCC merger's own read so it observes the changed state.
            fake.docs(ITEMS)['p2']['class_validated'] = True
        return await real_get(index=index, id=id, **kw)

    fake.get = _get_and_flip  # type: ignore[method-assign]
    client = _client(fake, registry, monkeypatch)

    r = client.post(
        '/curation/review/new_class_proposals/resolve',
        json={'label': 'sidecar', 'class_id': class_id},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body['matched'] == 2
    assert sorted(body['updated_ids'] + body['skipped']) == ['p1', 'p2']
    assert 'p2' in body['skipped']
    assert fake.docs(ITEMS)['p2']['class_source'] == 'vlm_new_class_pending'


# =============================================================================
# Undo round-trip
# =============================================================================


def test_resolve_undo_batch_restores_pending_state(
    registry: ClassRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    class_id = registry.load().classes[0].class_id
    docs = {f'p{i}': _pending_doc(f'p{i}', 'sidecar') for i in range(3)}
    fake = QueryFakeOpenSearch({ITEMS: docs})
    before = {k: _class_state(v) for k, v in fake.docs(ITEMS).items()}
    client = _client(fake, registry, monkeypatch)

    r = client.post(
        '/curation/review/new_class_proposals/resolve',
        json={'label': 'sidecar', 'class_id': class_id},
    )
    assert r.status_code == 200, r.text
    updated_ids = r.json()['updated_ids']
    assert sorted(updated_ids) == ['p0', 'p1', 'p2']
    for cid in updated_ids:
        assert fake.docs(ITEMS)[cid]['class_validated'] is True

    r = client.post('/curation/crops/label/undo_batch', json={'crop_ids': updated_ids})
    assert r.status_code == 200, r.text
    undo_body = r.json()
    assert undo_body['undone'] == 3

    for cid in updated_ids:
        doc = fake.docs(ITEMS)[cid]
        assert doc['class_source'] == 'vlm_new_class_pending'
        assert _class_state(doc) == before[cid]
