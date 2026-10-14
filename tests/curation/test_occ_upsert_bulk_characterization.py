"""Pins ``occ_upsert_bulk`` branch by branch: result counters, outcomes,
``created_ids``, client calls, log events and the Grafana counters.

The update phase is driven through a scripted ``occ_update_bulk`` that runs the
real merge function on given stored sources, so the human-label preservation
and ``fill_if_absent`` rules are exercised exactly as production calls them.
"""

from __future__ import annotations

from typing import Any

import pytest
import structlog
from prometheus_client import REGISTRY

from src.clients import occ_bulk, occ_upsert
from src.clients.occ import OCC_BULK_MGET_SOURCE_EXCLUDES
from src.clients.occ_upsert import DocOutcome, occ_upsert_bulk


pytestmark = pytest.mark.asyncio

INDEX = 'op_prj_t__items'


class ScriptedClient:
    """mget returns ``stored`` (found) for requested ids; bulk returns scripted create statuses."""

    def __init__(
        self,
        stored: dict[str, dict[str, Any]] | None = None,
        *,
        create_items: list[dict[str, Any]] | None = None,
        refetch: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.stored = stored or {}
        self.create_items = create_items
        self.refetch = refetch if refetch is not None else {}
        self.mget_calls: list[dict[str, Any]] = []
        self.bulk_calls: list[dict[str, Any]] = []

    async def mget(self, *, body: dict[str, Any], index: str, _source_excludes: Any) -> Any:
        self.mget_calls.append(
            {'ids': body['ids'], 'index': index, 'excludes': _source_excludes},
        )
        pool = self.stored if len(self.mget_calls) == 1 else self.refetch
        return {
            'docs': [
                {'_id': i, 'found': i in pool, '_source': pool.get(i, {})} for i in body['ids']
            ]
        }

    async def bulk(self, *, body: list[dict[str, Any]], refresh: Any) -> dict[str, Any]:
        self.bulk_calls.append({'body': body, 'refresh': refresh})
        if self.create_items is not None:
            return {'items': self.create_items}
        return {
            'items': [{'create': {'_id': a['create']['_id'], 'status': 201}} for a in body[::2]]
        }


class UpdateScript:
    """Stand-in for occ_update_bulk: runs merge_fn on ``sources`` and returns ``statuses``."""

    def __init__(
        self,
        sources: dict[str, dict[str, Any]],
        statuses: dict[str, str],
        *,
        raises: Exception | None = None,
    ) -> None:
        self.sources = sources
        self.statuses = statuses
        self.raises = raises
        self.calls: list[dict[str, Any]] = []
        self.merged: dict[str, dict[str, Any]] = {}

    async def __call__(self, _client: Any, **kw: Any) -> dict[str, str]:
        self.calls.append({k: v for k, v in kw.items() if k != 'merge_fn'})
        if self.raises:
            raise self.raises
        for doc_id in kw['ids']:
            self.merged[doc_id] = kw['merge_fn'](doc_id, self.sources.get(doc_id, {}))
        return self.statuses


def _sample(name: str, labels: dict[str, str] | None = None) -> float:
    return REGISTRY.get_sample_value(name, labels or {}) or 0.0


@pytest.fixture
def update_script(monkeypatch: pytest.MonkeyPatch):
    def install(script: UpdateScript) -> UpdateScript:
        monkeypatch.setattr(occ_bulk, 'occ_update_bulk', script)
        return script

    return install


async def _upsert(client: Any, docs: list[dict[str, Any]], **kw: Any) -> dict[str, int]:
    kw.setdefault('index', INDEX)
    kw.setdefault('human_field_guards', ['class_source'])
    return await occ_upsert_bulk(client, docs, **kw)


async def test_empty_docs_touch_nothing() -> None:
    client = ScriptedClient()
    assert await _upsert(client, []) == {
        'created': 0,
        'updated': 0,
        'preserved_human': 0,
        'final_conflicts': 0,
        'filled_absent': 0,
    }
    assert client.mget_calls == []
    assert client.bulk_calls == []


async def test_index_defaults_to_items_index(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(occ_upsert, 'items_index', lambda: 'bound_items')
    client = ScriptedClient()
    await occ_upsert_bulk(client, [{'crop_id': 'a'}], human_field_guards=[])
    assert client.mget_calls[0]['index'] == 'bound_items'
    assert client.bulk_calls[0]['body'][0] == {'create': {'_index': 'bound_items', '_id': 'a'}}


async def test_all_new_docs_are_created_in_one_bulk() -> None:
    client = ScriptedClient()
    created: list[str] = []
    outcomes: dict[str, DocOutcome] = {}
    docs = [{'crop_id': 'a', 'x': 1}, {'crop_id': 'b', 'x': 2}]
    result = await _upsert(client, docs, refresh='wait_for', created_ids=created, outcomes=outcomes)
    assert result == {
        'created': 2,
        'updated': 0,
        'preserved_human': 0,
        'final_conflicts': 0,
        'filled_absent': 0,
    }
    assert client.mget_calls == [
        {'ids': ['a', 'b'], 'index': INDEX, 'excludes': OCC_BULK_MGET_SOURCE_EXCLUDES}
    ]
    assert client.bulk_calls == [
        {
            'body': [
                {'create': {'_index': INDEX, '_id': 'a'}},
                docs[0],
                {'create': {'_index': INDEX, '_id': 'b'}},
                docs[1],
            ],
            'refresh': 'wait_for',
        }
    ]
    assert created == ['a', 'b']
    assert outcomes == {'a': DocOutcome('created'), 'b': DocOutcome('created')}


async def test_custom_id_field_and_status_200_counts_as_created() -> None:
    client = ScriptedClient(create_items=[{'create': {'_id': 'k', 'status': 200}}])
    result = await _upsert(client, [{'image_id': 'k'}], id_field='image_id')
    assert result['created'] == 1
    assert client.mget_calls[0]['ids'] == ['k']


async def test_create_error_is_logged_and_reported_not_counted() -> None:
    client = ScriptedClient(
        create_items=[
            {'create': {'_id': 'a', 'status': 400, 'error': {'type': 'mapper', 'reason': 'bad'}}},
            {'create': {'_id': 'b', 'status': 503, 'error': 'down'}},
            {'create': {'_id': 'c', 'status': 201}},
        ]
    )
    outcomes: dict[str, DocOutcome] = {}
    created: list[str] = []
    with structlog.testing.capture_logs() as logs:
        result = await _upsert(
            client,
            [{'crop_id': 'a'}, {'crop_id': 'b'}, {'crop_id': 'c'}],
            writer_id='w1',
            outcomes=outcomes,
            created_ids=created,
        )
    assert result['created'] == 1
    assert created == ['c']
    assert outcomes == {
        'a': DocOutcome('error', error='400: mapper: bad'),
        'b': DocOutcome('error', error='503: down'),
        'c': DocOutcome('created'),
    }
    errs = [e for e in logs if e['event'] == 'curation_ingest_upsert_create_error']
    assert [(e['doc_id'], e['status'], e['writer_id']) for e in errs] == [
        ('a', 400, 'w1'),
        ('b', 503, 'w1'),
    ]
    assert errs[0]['error'] == {'type': 'mapper', 'reason': 'bad'}
    assert errs[0]['log_level'] == 'warning'


async def test_bulk_items_may_be_missing_or_null() -> None:
    client = ScriptedClient()
    client.create_items = None
    result = await _upsert(client, [{'crop_id': 'a'}], outcomes=None)
    assert result['created'] == 1


async def test_create_conflict_refetches_and_joins_update_path(update_script) -> None:
    client = ScriptedClient(
        create_items=[
            {'create': {'_id': 'b', 'status': 409}},
            {'create': {'_id': 'a', 'status': 409}},
            {'create': {'_id': 'c', 'status': 409}},
        ],
        refetch={'a': {'v': 'A'}, 'b': {'v': 'B'}},
    )
    script = update_script(
        UpdateScript({'a': {'v': 'A'}, 'b': {'v': 'B'}}, {'a': 'updated', 'b': 'updated'})
    )
    outcomes: dict[str, DocOutcome] = {}
    created: list[str] = []
    result = await _upsert(
        client,
        [{'crop_id': 'a', 'n': 1}, {'crop_id': 'b', 'n': 2}, {'crop_id': 'c', 'n': 3}],
        outcomes=outcomes,
        created_ids=created,
        refresh=True,
    )
    # refetch asks for the conflicted ids sorted; 'c' vanished (found False) and is dropped
    assert client.mget_calls[1]['ids'] == ['a', 'b', 'c']
    assert client.mget_calls[1]['index'] == INDEX
    assert client.mget_calls[1]['excludes'] == OCC_BULK_MGET_SOURCE_EXCLUDES
    assert script.calls == [{'index': INDEX, 'ids': ['a', 'b'], 'max_retries': 1, 'refresh': True}]
    assert result['created'] == 0
    assert result['updated'] == 2
    assert created == []
    assert outcomes == {'a': DocOutcome('updated', 0, False), 'b': DocOutcome('updated', 0, False)}


async def test_existing_docs_skip_the_create_bulk(update_script) -> None:
    client = ScriptedClient({'a': {'v': 1}})
    update_script(UpdateScript({'a': {'v': 1}}, {'a': 'updated'}))
    result = await _upsert(client, [{'crop_id': 'a', 'n': 1}])
    assert client.bulk_calls == []
    assert result['updated'] == 1
    assert result['created'] == 0


async def test_mixed_new_and_existing(update_script) -> None:
    client = ScriptedClient({'old': {'v': 1}})
    update_script(UpdateScript({'old': {'v': 1}}, {'old': 'updated'}))
    created: list[str] = []
    result = await _upsert(client, [{'crop_id': 'new'}, {'crop_id': 'old'}], created_ids=created)
    assert result['created'] == 1
    assert result['updated'] == 1
    assert created == ['new']
    assert [a['create']['_id'] for a in client.bulk_calls[0]['body'][::2]] == ['new']


async def test_human_class_is_preserved_with_owned_fields_and_counted(update_script) -> None:
    before = _sample('op_ingest_preserved_human_label_total', {'field': 'class_source'})
    stored = {
        'class_source': 'human',
        'class_name': 'car',
        'class_validated': True,
        'test_holdout': True,
    }
    client = ScriptedClient({'a': stored})
    script = update_script(UpdateScript({'a': stored}, {'a': 'updated'}))
    outcomes: dict[str, DocOutcome] = {}
    new_doc = {
        'crop_id': 'a',
        'class_source': 'item_proposal',
        'class_name': 'truck',
        'class_validated': False,
        'history': ['x'],
        'confidence': 0.4,
        'crop_extra': 5,
    }
    result = await _upsert(client, [new_doc], outcomes=outcomes)
    merged = script.merged['a']
    assert merged['class_source'] == 'human'
    assert merged['class_validated'] is True
    assert merged['test_holdout'] is True
    assert merged['class_name'] == 'car'
    assert 'confidence' not in merged  # a class-state field the stored doc never had
    assert merged['crop_extra'] == 5
    assert result['preserved_human'] == 1
    assert outcomes['a'] == DocOutcome('updated', 1, False)
    after = _sample('op_ingest_preserved_human_label_total', {'field': 'class_source'})
    assert after - before == 1


async def test_owned_fields_absent_on_stored_doc_are_dropped(update_script) -> None:
    stored = {'class_source': 'human'}
    client = ScriptedClient({'a': stored})
    script = update_script(UpdateScript({'a': stored}, {'a': 'updated'}))
    await _upsert(
        client, [{'crop_id': 'a', 'class_source': 'x', 'class_name': 'truck', 'test_holdout': True}]
    )
    merged = script.merged['a']
    assert 'class_name' not in merged
    assert 'test_holdout' not in merged


async def test_unvalidated_import_class_is_not_locked(update_script) -> None:
    stored = {'class_source': 'import', 'class_validated': False}
    client = ScriptedClient({'a': stored})
    script = update_script(UpdateScript({'a': stored}, {'a': 'updated'}))
    result = await _upsert(client, [{'crop_id': 'a', 'class_source': 'item_proposal'}])
    assert script.merged['a']['class_source'] == 'item_proposal'
    assert result['preserved_human'] == 0


async def test_region_guard_uses_marker_check(update_script) -> None:
    stored = {'region_text_source': 'human_edit', 'class_source': 'item_proposal'}
    client = ScriptedClient({'a': stored})
    script = update_script(UpdateScript({'a': stored}, {'a': 'updated'}))
    result = await _upsert(
        client,
        [{'crop_id': 'a', 'region_text_source': 'ocr', 'class_source': 'x'}],
        human_field_guards=['region_text_source', 'class_source'],
    )
    assert script.merged['a']['region_text_source'] == 'human_edit'
    assert script.merged['a']['class_source'] == 'x'
    assert result['preserved_human'] == 1


async def test_fill_if_absent_applies_only_when_missing(update_script) -> None:
    stored_a = {'region_status': 'validated'}
    stored_b: dict[str, Any] = {'region_status': ''}
    stored_c: dict[str, Any] = {}
    client = ScriptedClient({'a': stored_a, 'b': stored_b, 'c': stored_c})
    script = update_script(
        UpdateScript(
            {'a': stored_a, 'b': stored_b, 'c': stored_c},
            {'a': 'updated', 'b': 'updated', 'c': 'updated'},
        )
    )
    outcomes: dict[str, DocOutcome] = {}
    result = await _upsert(
        client,
        [{'crop_id': i, 'region_status': 'pending', 'k': 1} for i in 'abc'] + [],
        fill_if_absent=['region_status', 'not_in_doc'],
        outcomes=outcomes,
    )
    assert 'region_status' not in script.merged['a']
    assert script.merged['b']['region_status'] == 'pending'
    assert script.merged['c']['region_status'] == 'pending'
    assert result['filled_absent'] == 2
    assert outcomes['a'].filled is False
    assert outcomes['b'].filled is True


async def test_stored_vector_state_is_kept(update_script) -> None:
    stored = {'embedding_state': 'embedded'}
    client = ScriptedClient({'a': stored})
    script = update_script(UpdateScript({'a': stored}, {'a': 'updated'}))
    await _upsert(client, [{'crop_id': 'a', 'embedding_state': 'failed'}])
    assert 'embedding_state' not in script.merged['a']


async def test_final_conflict_counts_and_logs(update_script) -> None:
    before = _sample('op_ingest_occ_final_conflict_total')
    client = ScriptedClient({'a': {}, 'b': {}})
    update_script(UpdateScript({}, {'a': 'conflict-exhausted', 'b': 'updated'}))
    outcomes: dict[str, DocOutcome] = {}
    with structlog.testing.capture_logs() as logs:
        result = await _upsert(
            client, [{'crop_id': 'a'}, {'crop_id': 'b'}], outcomes=outcomes, writer_id='w9'
        )
    assert result['final_conflicts'] == 1
    assert result['updated'] == 1
    assert outcomes['a'] == DocOutcome('conflict')
    assert _sample('op_ingest_occ_final_conflict_total') - before == 1
    [event] = [e for e in logs if e['event'] == 'curation_ingest_upsert_final_conflict']
    assert (event['doc_id'], event['writer_id'], event['log_level']) == ('a', 'w9', 'info')


async def test_unknown_status_is_a_refetch_failure(update_script) -> None:
    client = ScriptedClient({'a': {}, 'b': {}})
    update_script(UpdateScript({}, {'a': 'weird'}))
    outcomes: dict[str, DocOutcome] = {}
    with structlog.testing.capture_logs() as logs:
        result = await _upsert(client, [{'crop_id': 'a'}, {'crop_id': 'b'}], outcomes=outcomes)
    assert result['updated'] == 0
    assert result['final_conflicts'] == 0
    assert outcomes == {
        'a': DocOutcome('error', error='weird'),
        'b': DocOutcome('error', error='update failed'),
    }
    failed = [e for e in logs if e['event'] == 'curation_ingest_upsert_refetch_failed']
    assert [(e['doc_id'], e['error']) for e in failed] == [('a', 'weird'), ('b', None)]


async def test_update_exception_is_swallowed_and_logged(update_script) -> None:
    client = ScriptedClient({'a': {}, 'b': {}})
    update_script(UpdateScript({}, {}, raises=RuntimeError('boom')))
    outcomes: dict[str, DocOutcome] = {}
    with structlog.testing.capture_logs() as logs:
        result = await _upsert(
            client, [{'crop_id': 'a'}, {'crop_id': 'b'}], outcomes=outcomes, writer_id='w2'
        )
    assert result['updated'] == 0
    [event] = [e for e in logs if e['event'] == 'curation_ingest_upsert_bulk_update_failed']
    assert (event['writer_id'], event['n'], event['error']) == ('w2', 2, 'boom')
    assert outcomes == {i: DocOutcome('error', error='update failed') for i in 'ab'}


async def test_outcomes_and_created_ids_are_optional(update_script) -> None:
    client = ScriptedClient({'a': {}})
    update_script(UpdateScript({}, {'a': 'conflict-exhausted'}))
    result = await _upsert(client, [{'crop_id': 'a'}, {'crop_id': 'n'}])
    assert result['created'] == 1
    assert result['final_conflicts'] == 1


async def test_refetch_ids_are_sorted() -> None:
    ids = [f'crop-{n}' for n in (7, 3, 9, 1, 5, 8, 2, 6, 4)]
    client = ScriptedClient(create_items=[{'create': {'_id': i, 'status': 409}} for i in ids])
    result = await _upsert(client, [{'crop_id': i} for i in ids])
    assert client.mget_calls[1]['ids'] == sorted(ids)
    assert result['updated'] == 0
