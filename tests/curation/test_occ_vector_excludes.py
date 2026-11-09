"""Every OCC fetch leaves out the vectors of every live space, not the 0.6.0 constants.

After a flip the primary space's field is ``vec_<key>``; a fetch that still excluded only
``pe_embedding`` would ship the new vectors on every read-merge-write (embeddings plan 5.4).
"""

from __future__ import annotations

from typing import Any

import pytest
from _embedding_registry_fake import new_space

from src.clients.occ_upsert import occ_upsert_bulk
from src.domain.embedding_spaces import Target
from src.domain.embedding_state import keep_stored_vector_state
from src.services.curation.assist.slot_write import write_slots
from src.services.curation.crop_bulk import occ_bulk_human_relabel
from src.services.curation.item_delete import delete_items


NEW_ITEM_FIELD = 'vec_mobileclip2_s2_v1'
NEW_BOX_PATH = 'boxvec_mobileclip2_s2_v1'


class RecordingClient:
    """An items index that holds ``c1`` and records the ``_source`` excludes of every read."""

    def __init__(self) -> None:
        self.excludes: list[list[str]] = []

    async def mget(self, *, body: dict[str, Any], **kw: Any) -> dict[str, Any]:
        for doc in body.get('docs', [])[:1]:
            if '_source' in doc:
                self.excludes.append(doc['_source'].get('excludes', []))
        if '_source_excludes' in kw:
            self.excludes.append(kw['_source_excludes'])
        ids = [d['_id'] for d in body['docs']] if 'docs' in body else body['ids']
        return {
            'docs': [
                {
                    '_id': i,
                    '_seq_no': 1,
                    '_primary_term': 1,
                    'found': True,
                    '_source': {'crop_id': i, 'embedding_state': 'embedded'},
                }
                for i in ids
            ]
        }

    async def bulk(self, *, body: list[dict[str, Any]], **_: Any) -> dict[str, Any]:
        items = [
            {'update': {'_id': a['update']['_id'], 'status': 200}} for a in body[0::2]
        ]  # one per action
        return {'errors': False, 'items': items}


@pytest.fixture
def flipped(embedding_registry: Any) -> None:
    embedding_registry.add(new_space(Target.ITEM, 'mobileclip2_s2'), primary=True)
    embedding_registry.add(new_space(Target.BOX, 'mobileclip2_s2'), primary=True)


def _assert_all_vectors_excluded(client: RecordingClient) -> None:
    assert client.excludes, 'the fetch carried no excludes'
    for excludes in client.excludes:
        assert {NEW_ITEM_FIELD, NEW_BOX_PATH, 'pe_embedding'} <= set(excludes), excludes


@pytest.mark.asyncio
@pytest.mark.usefixtures('flipped')
async def test_the_human_bulk_relabel_excludes_the_flipped_vectors() -> None:
    client = RecordingClient()
    await occ_bulk_human_relabel(client, ['c1'], lambda _s: {'class_id': 2}, writer_id='t')
    _assert_all_vectors_excluded(client)


@pytest.mark.asyncio
@pytest.mark.usefixtures('flipped')
async def test_the_worker_bulk_write_excludes_the_flipped_vectors() -> None:
    client = RecordingClient()
    await write_slots(client, {'c1': {'run_created_at': '2026-01-01T00:00:00+00:00'}})  # type: ignore[arg-type]
    _assert_all_vectors_excluded(client)


@pytest.mark.asyncio
@pytest.mark.usefixtures('flipped')
async def test_the_guarded_delete_excludes_the_flipped_vectors() -> None:
    client = RecordingClient()

    async def _delete(*_a: Any, **_k: Any) -> dict[str, Any]:
        return {'items': [{'delete': {'status': 200}}]}

    async def _deletable(_cid: str, _source: dict[str, Any]) -> bool:
        return True

    client.bulk = _delete  # type: ignore[method-assign]
    await delete_items(
        client,  # type: ignore[arg-type]
        ['c1'],
        items_index='op_prj_default__items',
        crop_cache_dir=None,
        deletable=_deletable,
    )
    _assert_all_vectors_excluded(client)


@pytest.mark.asyncio
async def test_the_upsert_excludes_what_the_caller_names() -> None:
    client = RecordingClient()
    await occ_upsert_bulk(
        client,  # type: ignore[arg-type]
        [{'crop_id': 'c1'}],
        human_field_guards=[],
        vector_excludes=['vec_x'],
        vector_field='vec_x',
        state_field='vec_x_state',
    )
    assert client.excludes
    assert all(e == ['vec_x'] for e in client.excludes)


def test_a_failed_rerun_keeps_the_state_of_the_flipped_space() -> None:
    merged = {'vec_x_state': 'failed'}
    keep_stored_vector_state(merged, {'vec_x_state': 'embedded'}, 'vec_x', 'vec_x_state')
    assert merged == {}
    merged = {'vec_x': [0.1], 'vec_x_state': 'embedded'}
    keep_stored_vector_state(merged, {'vec_x_state': 'failed'}, 'vec_x', 'vec_x_state')
    assert merged['vec_x_state'] == 'embedded'
