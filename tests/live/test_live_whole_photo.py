"""A photo is stored whole or not at all, on REAL OpenSearch (plan 13a, lane L0).

The unit tests run the writer, the lookup and the repair over a fake store. This one checks what a
fake cannot: the real mapping accepts the list and the failure mark (``keyword`` and ``boolean``,
``index: false``, ``doc_values: false``, stored in ``_source`` only), a real-time ``mget`` with the
source switched off finds an item that was never refreshed, a partial update REPLACES the list
(arrays are not merged), and the OCC writes of the list, the mark and the delete hold on a real
cluster. OpenSearch semantics: https://docs.opensearch.org/latest/api-reference/document-apis/
(multi-get, bulk, update-document).

It runs in the ``api`` container, over the product's own writer, with a client whose ``bulk`` fails
the first items-index write (the fault the bug needed: the images doc stored, the items step
failed), then repairs the photo through the same writer and deletes one item through
``delete_items``. The consistency audit (``scripts/bench/ingest_consistency.py``) runs on the
harness project at the end: 0 inconsistencies.
"""

from __future__ import annotations

import json
import subprocess  # nosec B404 - the harness's own compose project
import uuid
from typing import Any

import pytest

from tests.live.conftest import (
    COMPOSE_FILE_ARGS,
    COMPOSE_PROJECT,
    INDEXES,
    REPO_ROOT,
    get_doc,
    refresh,
)


pytestmark = pytest.mark.live

_SCRIPT = r"""
import asyncio, json, sys
from datetime import UTC, datetime
from opensearchpy.exceptions import TransportError
from src.config.settings import get_settings
from src.config.curation_accessors import images_index, items_index
from src.services.projects.script_binding import abind_script_project
from src.services.projects.guard import make_script_opensearch
from src.services.curation.ingest_writer import IngestWriter, RepairTarget
from src.services.curation.image_completeness import missing_items, set_item_ids
from src.services.curation.item_delete import delete_items

req = json.loads(sys.stdin.read())

async def main():
    await abind_script_project('default')
    client = make_script_opensearch([get_settings().opensearch_url])
    real_bulk = client.bulk
    fail = {'left': 1}

    async def bulk(*args, **kwargs):
        body = kwargs.get('body') or args[0]
        if fail['left'] and body[0].get('create', {}).get('_index') == items_index():
            fail['left'] -= 1
            raise TransportError(500, 'injected items bulk failure')
        return await real_bulk(*args, **kwargs)

    client.bulk = bulk
    image_id = req['image_id']
    crop_docs = [{**req['template'], 'crop_id': c, 'image_id': image_id} for c in req['crops']]
    image_doc = {'image_id': image_id, 'image_path': '/live/whole.jpg',
                 'indexed_at': datetime.now(UTC).isoformat()}
    kw = dict(images_index=images_index(), items_index=items_index(), guards=('class_source',))
    out = {}
    try:
        writer = IngestWriter(client)
        first = await writer.write(image_doc, crop_docs, **kw)
        out['first'] = {'failure': first.failure, 'images_indexed': first.images_indexed}
        out['missing_after_failure'] = await missing_items(
            client, items_index(), {image_id: req['crops']})
        repaired = await writer.write(None, crop_docs, repair=RepairTarget(image_id), **kw)
        out['repaired'] = {'failure': repaired.failure, 'repaired': repaired.repaired,
                           'created': repaired.created}
        out['missing_after_repair'] = await missing_items(
            client, items_index(), {image_id: req['crops']})
        gone = req['crops'][0]
        result = await delete_items(client, [gone], items_index=items_index(),
                                    crop_cache_dir=None, deletable=lambda _i, _d: _true())
        out['deleted'] = result['deleted']
        await writer.close()
    finally:
        await client.close()
    print(json.dumps(out))

async def _true():
    return True

from src.db.engine import run_with_engine
asyncio.run(run_with_engine(main(), 'op-live-whole-photo'))
"""


def _run_in_api(request: dict[str, Any]) -> dict[str, Any]:
    done = subprocess.run(  # nosec B603 - fixed argv, no shell
        ['docker', 'compose', '-p', COMPOSE_PROJECT, *COMPOSE_FILE_ARGS, 'exec', '-T',
         'api', 'python', '-c', _SCRIPT],
        input=json.dumps(request), capture_output=True, text=True, timeout=240, cwd=str(REPO_ROOT),
        check=False,
    )  # fmt: skip
    assert done.returncode == 0, done.stderr[-2500:]
    return json.loads(done.stdout.strip().splitlines()[-1])


@pytest.fixture(autouse=True)
def _drop_probe(opensearch: Any) -> Any:
    yield
    for role, field in (('items', 'crop_id'), ('images', 'image_id')):
        opensearch.post(
            f'/{INDEXES[role]}/_delete_by_query',
            params={'refresh': 'true'},
            json={'query': {'prefix': {field: 'whole_probe_'}}},
        ).raise_for_status()


def test_a_failed_items_step_is_found_repaired_and_a_delete_keeps_the_list_true(
    opensearch: Any, seeded: dict[str, Any]
) -> None:
    template = get_doc(opensearch, INDEXES['items'], 'cls0_0000')['_source']
    image_id = f'whole_probe_{uuid.uuid4().hex[:8]}'
    crops = sorted(f'{image_id}_c{i}' for i in range(3))

    out = _run_in_api({'image_id': image_id, 'crops': crops, 'template': template})

    assert 'injected items bulk failure' in out['first']['failure']
    assert out['first']['images_indexed'] == 1
    assert out['missing_after_failure'] == {image_id: crops}
    assert out['repaired']['failure'] is None
    assert out['repaired']['repaired'] is True
    assert out['missing_after_repair'] == {}
    assert out['deleted'] == 1
    refresh(opensearch, INDEXES['images'])
    refresh(opensearch, INDEXES['items'])
    image = get_doc(opensearch, INDEXES['images'], image_id)['_source']
    assert image['item_ids'] == crops[1:], 'the deleted item left the list; the rest are listed'
    assert image['ingest_incomplete'] is False, 'the repair cleared the failure mark'
    # the list is stored, not indexed
    mapping = opensearch.get(f'/{INDEXES["images"]}/_mapping').json()
    props = next(iter(mapping.values()))['mappings']['properties']
    assert props['item_ids'] == {'type': 'keyword', 'index': False, 'doc_values': False}
    assert props['ingest_incomplete'] == {'type': 'boolean', 'index': False, 'doc_values': False}


# --- what a repair may touch, on real OpenSearch ------------------------------------------------

_REPAIR_SCRIPT = r"""
import asyncio, json, sys
from datetime import UTC, datetime
from src.config.settings import get_settings
from src.config.curation_accessors import images_index, items_index
from src.services.projects.script_binding import abind_script_project
from src.services.projects.guard import make_script_opensearch
from src.services.curation.ingest_writer import IngestWriter, RepairTarget
from src.services.curation.image_completeness import holders_completeness
from src.services.curation.item_delete import delete_items

req = json.loads(sys.stdin.read())

def docs(crops, boxes, shift):
    return [{**req['template'], 'crop_id': c, 'image_id': req['image_id'],
             'bbox_norm': [b[0] + shift, b[1] + shift, b[2] + shift, b[3] + shift]}
            for c, b in zip(crops, boxes)]

async def always(_i, _d):
    return True

async def main():
    await abind_script_project('default')
    client = make_script_opensearch([get_settings().opensearch_url])
    kw = dict(images_index=images_index(), items_index=items_index(), guards=('class_source',))
    image_id, crops, boxes = req['image_id'], req['crops'], req['boxes']
    out = {}
    try:
        if req['op'] == 'seed':
            image_doc = {'image_id': image_id, 'image_path': '/live/whole.jpg',
                         'indexed_at': datetime.now(UTC).isoformat()}
            unit = await IngestWriter(client).write(image_doc, docs(crops, boxes, 0.0), **kw)
            out['failure'] = unit.failure
        elif req['op'] == 'repair':
            # the same objects as another detector run sees them: new crop ids, boxes a hair apart
            target = RepairTarget(image_id)
            writers = [IngestWriter(client) for _ in range(req['repairers'])]
            units = await asyncio.gather(*[
                w.write(None, docs([f'{c}-r{n}' for c in crops], boxes, 2e-5 * (n + 1)),
                        repair=target, **kw)
                for n, w in enumerate(writers)])
            out['units'] = [{'repaired': u.repaired, 'incomplete': u.incomplete,
                             'duplicate': u.duplicate, 'failure': u.failure,
                             'created': u.created} for u in units]
            for w in writers:
                await w.close()
        elif req['op'] == 'race':
            real_bulk = client.bulk
            raced = []

            async def bulk(*args, **kwargs):
                resp = await real_bulk(*args, **kwargs)
                body = kwargs.get('body') or args[0]
                made = [a['create']['_id'] for a in body[0::2]
                        if a.get('create', {}).get('_index') == items_index()]
                if made and not raced:
                    raced.append(made[0])
                    client.bulk = real_bulk
                    await delete_items(client, [made[0]], items_index=items_index(),
                                       crop_cache_dir=None, deletable=always)
                    client.bulk = bulk
                return resp

            client.bulk = bulk
            w = IngestWriter(client)
            unit = await w.write(None, docs([f'{c}-r' for c in crops], boxes, 2e-5),
                                 repair=RepairTarget(image_id), **kw)
            await w.close()
            client.bulk = real_bulk
            out['failure'] = unit.failure
            out['raced'] = raced
            out['state'] = (await holders_completeness(
                client, images_index(), items_index(), [image_id], repair_after_s=0))[image_id]
    finally:
        await client.close()
    print(json.dumps(out))

from src.db.engine import run_with_engine
asyncio.run(run_with_engine(main(), 'op-live-repair-rules'))
"""


def _run_repair_script(request: dict[str, Any]) -> dict[str, Any]:
    done = subprocess.run(  # nosec B603 - fixed argv, no shell
        ['docker', 'compose', '-p', COMPOSE_PROJECT, *COMPOSE_FILE_ARGS, 'exec', '-T',
         'api', 'python', '-c', _REPAIR_SCRIPT],
        input=json.dumps(request), capture_output=True, text=True, timeout=240, cwd=str(REPO_ROOT),
        check=False,
    )  # fmt: skip
    assert done.returncode == 0, done.stderr[-2500:]
    return json.loads(done.stdout.strip().splitlines()[-1])


def _seed_three(
    opensearch: Any, op: str = 'seed'
) -> tuple[str, list[str], list[list[float]], dict]:
    """A whole photo of three objects, then two of its items lost and its doc made old, as a kill
    between the two bulks leaves it. Returns ``(image id, crop ids, boxes, template)``."""
    template = get_doc(opensearch, INDEXES['items'], 'cls0_0000')['_source']
    image_id = f'whole_probe_{uuid.uuid4().hex[:8]}'
    crops = sorted(f'{image_id}_c{i}' for i in range(3))
    boxes = [[0.05 + 0.3 * i, 0.1, 0.25 + 0.3 * i, 0.4] for i in range(3)]
    req = {'op': op, 'image_id': image_id, 'crops': crops, 'boxes': boxes, 'template': template}
    assert _run_repair_script(req)['failure'] is None
    refresh(opensearch, INDEXES['items'])
    return image_id, crops, boxes, template


def _lose_two_and_age(opensearch: Any, image_id: str, crops: list[str]) -> None:
    for crop_id in crops[1:]:
        opensearch.delete(f'/{INDEXES["items"]}/_doc/{crop_id}').raise_for_status()
    opensearch.post(
        f'/{INDEXES["images"]}/_update/{image_id}',
        json={'doc': {'indexed_at': '2020-01-01T00:00:00+00:00'}},
    ).raise_for_status()
    refresh(opensearch, INDEXES['items'])


def _repair(image_id: str, crops: list[str], boxes: list[Any], template: dict, repairers: int = 1):
    return _run_repair_script(
        {'op': 'repair', 'image_id': image_id, 'crops': crops, 'boxes': boxes,
         'template': template, 'repairers': repairers}
    )  # fmt: skip


def _items_of(opensearch: Any, image_id: str) -> dict[str, Any]:
    refresh(opensearch, INDEXES['items'])
    body = {'size': 100, 'query': {'term': {'image_id': image_id}}}
    hits = opensearch.post(f'/{INDEXES["items"]}/_search', json=body).json()['hits']['hits']
    return {h['_id']: h['_source'] for h in hits}


def test_live_repair_keeps_a_human_moved_box_and_vector(
    opensearch: Any, seeded: dict[str, Any]
) -> None:
    image_id, crops, boxes, template = _seed_three(opensearch)
    moved = {'geometry_source': 'human', 'bbox_norm': [0.06, 0.11, 0.26, 0.41]}
    opensearch.post(
        f'/{INDEXES["items"]}/_update/{crops[0]}', json={'doc': moved}
    ).raise_for_status()
    _lose_two_and_age(opensearch, image_id, crops)
    before = get_doc(opensearch, INDEXES['items'], crops[0])['_source']

    out = _repair(image_id, crops, boxes, template)

    assert out['units'][0]['repaired'] is True
    assert get_doc(opensearch, INDEXES['items'], crops[0])['_source'] == before
    assert len(_items_of(opensearch, image_id)) == 3


def test_live_repair_keeps_a_holdout_item_without_a_class(
    opensearch: Any, seeded: dict[str, Any]
) -> None:
    image_id, crops, boxes, template = _seed_three(opensearch)
    holdout = {'test_holdout': True, 'bbox_norm': [0.06, 0.11, 0.26, 0.41], 'blur_lap_var': 123.0}
    opensearch.post(
        f'/{INDEXES["items"]}/_update/{crops[0]}', json={'doc': holdout}
    ).raise_for_status()
    _lose_two_and_age(opensearch, image_id, crops)
    before = get_doc(opensearch, INDEXES['items'], crops[0])['_source']

    _repair(image_id, crops, boxes, template)

    assert get_doc(opensearch, INDEXES['items'], crops[0])['_source'] == before


def test_live_two_repairs_of_one_image_leave_one_item_per_object(
    opensearch: Any, seeded: dict[str, Any]
) -> None:
    image_id, crops, boxes, template = _seed_three(opensearch)
    _lose_two_and_age(opensearch, image_id, crops)

    out = _repair(image_id, crops, boxes, template, repairers=2)

    created = [u for u in out['units'] if u['created']]
    assert len(created) == 1, f'exactly one repair creates items: {out}'
    (other,) = [u for u in out['units'] if u is not created[0]]
    assert other['incomplete'] or other['duplicate'], f'the other one only answers: {out}'
    items = _items_of(opensearch, image_id)
    assert len(items) == 3, 'one item per object, not one set per repairer'
    image = get_doc(opensearch, INDEXES['images'], image_id)['_source']
    assert sorted(items) == image['item_ids']
    assert not image.get('repair_started_at'), 'the lease went back'


def test_live_a_delete_racing_a_repair_does_not_relist(
    opensearch: Any, seeded: dict[str, Any]
) -> None:
    image_id, crops, boxes, template = _seed_three(opensearch)
    _lose_two_and_age(opensearch, image_id, crops)

    out = _run_repair_script(
        {'op': 'race', 'image_id': image_id, 'crops': crops, 'boxes': boxes, 'template': template}
    )

    assert out['failure'] is None
    (gone,) = out['raced']
    image = get_doc(opensearch, INDEXES['images'], image_id)['_source']
    assert gone not in image['item_ids']
    assert out['state'] == 'complete', 'the photo is whole: a re-upload is a duplicate'
