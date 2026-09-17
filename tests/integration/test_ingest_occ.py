"""Ingest OCC upsert + human-label preservation.

The reference this was ported from ran these scenarios against a real
dev OpenSearch instance (skipped when unreachable). Per plan §6.0's
house rule ("do not add the repo's first live-stack dependency" — fake
the I/O boundary instead), this port exercises
:func:`src.clients.occ.occ_upsert_bulk` against a small in-memory fake
OpenSearch that implements just enough of ``mget``/``bulk`` (create,
with a 409 on an existing id)/``update`` (with ``if_seq_no``/
``if_primary_term`` conflict detection) to reproduce the same three
scenarios:

1. Preserves human-set guard fields when re-ingesting a crop.
2. Resolves concurrent ingest of the same deterministic ``crop_id`` to
   N docs, not 2N (one create wins, the other falls back to OCC update).
3. Performs a normal field overwrite when no human-guard field is set
   on the existing doc — i.e. the OCC path is invisible in the happy
   case.

A fourth reference scenario (`test_occ_skip_on_conflict_bulk_human_wins_on_real_conflict`)
deliberately provokes a genuine OpenSearch version-conflict by mutating
the document via a raw HTTP side-channel between the internal mget and
bulk calls of a real cluster — that is exactly the "live smoke" case
plan §6.0 says not to force into a fake (reproducing real engine-level
OCC semantics precisely enough to be meaningful would mean
reimplementing OpenSearch's version-conflict detection); it's covered
generically instead by ``tests/curation/test_occ.py``'s conflict-branch
tests against ``occ_skip_on_conflict_bulk``.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest


pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

_HUMAN_GUARDS = ['region_label_source', 'class_source', 'region_text_source']


class _ConflictError(Exception):
    """Stands in for opensearchpy's ConflictError (409)."""


class FakeUpsertOpenSearch:
    """Minimal in-memory OpenSearch double for ``occ_upsert_bulk``.

    Supports exactly the operations that function calls: ``mget``,
    ``bulk`` (only the ``create`` action), ``update`` (with
    ``if_seq_no``/``if_primary_term`` optimistic-concurrency checks),
    plus ``index``/``get``/``count`` for test setup/assertions.
    """

    def __init__(self) -> None:
        self._docs: dict[str, dict[str, Any]] = {}
        self._seq: dict[str, int] = {}

    async def index(
        self,
        *,
        index: str,  # noqa: ARG002
        id: str,  # noqa: A002
        body: dict[str, Any],
        refresh: bool = False,  # noqa: ARG002
    ) -> None:
        self._docs[id] = dict(body)
        self._seq[id] = self._seq.get(id, -1) + 1

    async def get(self, *, index: str, id: str) -> dict[str, Any]:  # noqa: A002, ARG002
        return {'_id': id, '_source': dict(self._docs[id]), 'found': True}

    async def count(self, *, index: str) -> dict[str, Any]:  # noqa: ARG002
        return {'count': len(self._docs)}

    async def mget(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:  # noqa: ARG002
        docs = []
        for doc_id in body['ids']:
            if doc_id in self._docs:
                docs.append(
                    {
                        '_id': doc_id,
                        'found': True,
                        '_source': dict(self._docs[doc_id]),
                        '_seq_no': self._seq[doc_id],
                        '_primary_term': 1,
                    }
                )
            else:
                docs.append({'_id': doc_id, 'found': False})
        return {'docs': docs}

    async def bulk(
        self,
        *,
        body: list[dict[str, Any]],
        refresh: bool | str = False,  # noqa: ARG002
    ) -> dict[str, Any]:
        items = []
        for action, doc in zip(body[0::2], body[1::2], strict=True):
            if 'create' not in action:
                continue  # pragma: no cover - occ_upsert_bulk only bulk-creates
            doc_id = action['create']['_id']
            if doc_id in self._docs:
                items.append({'create': {'_id': doc_id, 'status': 409}})
                continue
            self._docs[doc_id] = dict(doc)
            self._seq[doc_id] = 0
            items.append({'create': {'_id': doc_id, 'status': 201}})
        return {'items': items}

    async def update(
        self,
        *,
        index: str,  # noqa: ARG002
        id: str,  # noqa: A002
        body: dict[str, Any],
        if_seq_no: int,
        if_primary_term: int,  # noqa: ARG002
        refresh: bool | str = False,  # noqa: ARG002
    ) -> dict[str, Any]:
        current_seq = self._seq.get(id, -1)
        if current_seq != if_seq_no:
            msg = f'version conflict on {id}'
            raise _ConflictError(msg)
        self._docs[id].update(body['doc'])
        self._seq[id] = current_seq + 1
        return {'result': 'updated'}


def _counter_value(counter, **labels) -> float:
    if labels:
        sample = counter.labels(**labels)
    else:
        sample = counter
    return sample._value.get()


async def test_existing_crop_with_human_label_preserved_on_reingest() -> None:
    """Scenario #1 — human labels a crop, then the source image is
    re-ingested. The deterministic ``crop_id`` already exists, so the
    blind bulk would clobber. The OCC upsert must preserve every
    human-guard field on the existing doc and increment the per-field
    preserved-label counter.
    """
    from src.clients.occ import occ_upsert_bulk
    from src.services.curation.metrics import LEGACY_INGEST_PRESERVED_HUMAN_LABEL

    client = FakeUpsertOpenSearch()
    index = 'op_items_test'

    crop_id = 'crop-preserve-1'
    await client.index(
        index=index,
        id=crop_id,
        body={
            'crop_id': crop_id,
            'image_id': 'img-1',
            'class_id': 5,
            'class_name': 'sedan',
            'class_source': 'ingest',
        },
    )
    # Operator labels the crop — sets human-guard fields.
    await client.update(
        index=index,
        id=crop_id,
        body={
            'doc': {
                'region_label_source': 'human',
                'region_text': 'ABC123',
                'region_text_source': 'human',
            }
        },
        if_seq_no=0,
        if_primary_term=1,
    )

    before = _counter_value(LEGACY_INGEST_PRESERVED_HUMAN_LABEL, field='region_label_source')
    before_text = _counter_value(LEGACY_INGEST_PRESERVED_HUMAN_LABEL, field='region_text_source')

    reingest_doc = {
        'crop_id': crop_id,
        'image_id': 'img-1',
        'class_id': 7,
        'class_name': 'suv',
        'class_source': 'ingest',
        'region_label_source': 'ingest',
        'region_text': '',
        'region_text_source': 'ingest',
    }
    result = await occ_upsert_bulk(
        client,
        [reingest_doc],
        index=index,
        human_field_guards=_HUMAN_GUARDS,
        writer_id='ingest',
        refresh='wait_for',
    )

    assert result['created'] == 0
    assert result['updated'] == 1
    assert result['preserved_human'] >= 2
    assert result['final_conflicts'] == 0

    after = _counter_value(LEGACY_INGEST_PRESERVED_HUMAN_LABEL, field='region_label_source')
    after_text = _counter_value(LEGACY_INGEST_PRESERVED_HUMAN_LABEL, field='region_text_source')
    assert after == before + 1
    assert after_text == before_text + 1

    got = await client.get(index=index, id=crop_id)
    src = got['_source']
    assert src['region_label_source'] == 'human'
    assert src['region_text'] == 'ABC123'
    assert src['region_text_source'] == 'human'
    # Non-guarded fields are overwritten normally.
    assert src['class_id'] == 7
    assert src['class_name'] == 'suv'


async def test_parallel_ingest_of_same_image_yields_n_crops_not_2n() -> None:
    """Scenario #2 — two ingest calls for the same image race past
    dedup. They build identical deterministic ``crop_id`` values. One
    create wins; the other gets 409, falls back to OCC update. Final
    state has exactly the new doc count, not 2x it.
    """
    from src.clients.occ import occ_upsert_bulk

    client = FakeUpsertOpenSearch()
    index = 'op_items_test'

    crop_ids = [f'crop-parallel-{i}' for i in range(3)]
    docs_a = [
        {
            'crop_id': cid,
            'image_id': 'img-par',
            'class_id': 1,
            'class_name': 'sedan',
            'class_source': 'ingest',
        }
        for cid in crop_ids
    ]
    docs_b = [dict(d) for d in docs_a]

    results = await asyncio.gather(
        occ_upsert_bulk(
            client, docs_a, index=index, human_field_guards=_HUMAN_GUARDS, writer_id='ingest-a'
        ),
        occ_upsert_bulk(
            client, docs_b, index=index, human_field_guards=_HUMAN_GUARDS, writer_id='ingest-b'
        ),
    )
    total_created = sum(r['created'] for r in results)
    total_handled = sum(r['created'] + r['updated'] for r in results)
    assert total_created == len(crop_ids)
    # Each call must account for every requested id (create or update),
    # so no doc is dropped silently on the conflict path.
    assert total_handled == 2 * len(crop_ids)

    count_resp = await client.count(index=index)
    assert count_resp['count'] == len(crop_ids)


async def test_non_human_doc_overwritten_normally() -> None:
    """Scenario #3 — when no human-guard field is set on the existing
    doc, OCC upsert still applies the new value (normal overwrite). No
    preserved-label counter should increment.
    """
    from src.clients.occ import occ_upsert_bulk
    from src.services.curation.metrics import LEGACY_INGEST_PRESERVED_HUMAN_LABEL

    client = FakeUpsertOpenSearch()
    index = 'op_items_test'

    crop_id = 'crop-overwrite-1'
    await client.index(
        index=index,
        id=crop_id,
        body={
            'crop_id': crop_id,
            'image_id': 'img-3',
            'class_id': 3,
            'class_name': 'pickup',
            'class_source': 'ingest',
        },
    )

    before = _counter_value(LEGACY_INGEST_PRESERVED_HUMAN_LABEL, field='region_label_source')

    reingest_doc = {
        'crop_id': crop_id,
        'image_id': 'img-3',
        'class_id': 9,
        'class_name': 'truck',
        'class_source': 'ingest',
    }
    result = await occ_upsert_bulk(
        client,
        [reingest_doc],
        index=index,
        human_field_guards=_HUMAN_GUARDS,
        writer_id='ingest',
    )
    assert result['updated'] == 1
    assert result['preserved_human'] == 0

    after = _counter_value(LEGACY_INGEST_PRESERVED_HUMAN_LABEL, field='region_label_source')
    assert after == before

    got = await client.get(index=index, id=crop_id)
    assert got['_source']['class_id'] == 9
    assert got['_source']['class_name'] == 'truck'
