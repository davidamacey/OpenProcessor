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
        repaired = await writer.write(None, crop_docs, repair=RepairTarget(image_id, tuple(req['crops'])), **kw)
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
