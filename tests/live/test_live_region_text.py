"""Live: the region cascade reads text through OCR and the text filters find it.

A region profile naming the fake Triton's detector and ``text_reader: ocr`` is activated on a fresh
project; pictures are uploaded; a detection worker bound to that project (``detection-worker`` in
``docker/test/compose.yml``, behind the ``region-worker`` profile so it never works the seeded
``default`` project) claims the items, runs the detector leg and the OCR leg against the fake
Triton, and writes the boxes and their text. The assertions read the stored boxes and then ask the
API for the items by that text (``/crops?region_text=``, exact, fuzzy and fragment), the path a
user follows with a plate number.
"""

from __future__ import annotations

import subprocess  # nosec B404 - removes the one container the fixture started
from typing import TYPE_CHECKING, Any

import pytest

from ._ingest_support import IngestProject, ingest_project, picture
from .conftest import compose, wait_until


if TYPE_CHECKING:
    from collections.abc import Iterator


pytestmark = pytest.mark.live

OCR_TEXT = 'FAKE-PLATE-0042'
PROFILE = 'text_reading'


@pytest.fixture(scope='module')
def project(api: Any, opensearch: Any) -> Iterator[IngestProject]:
    with ingest_project(api, opensearch, 'regtext') as handle:
        yield handle


@pytest.fixture(scope='module')
def worker(project: IngestProject) -> Iterator[str]:
    """One detection worker bound to the project; stopped and removed by its container id."""
    started = compose(
        'run',
        '-d',
        '--no-deps',
        'detection-worker',
        'python',
        '-m',
        'scripts.curation.worker',
        '--continuous',
        '--project',
        project.slug,
        timeout=300,
    )
    container = started.stdout.strip().splitlines()[-1]
    try:
        yield container
    finally:
        subprocess.run(  # nosec B603 B607 - fixed argv, the container this fixture started
            ['docker', 'rm', '-f', container], check=False, capture_output=True, timeout=120
        )


@pytest.fixture(scope='module')
def read_items(
    api: Any, opensearch: Any, project: IngestProject, worker: str
) -> list[dict[str, Any]]:
    created = api.post(
        f'{project.prefix}/region_profiles',
        json={
            'name': PROFILE,
            'body': {
                'detector_model': 'yolov11_small_trt_end2end',
                'region_class_name': 'plate',
                'display_name': 'Plates',
                'display_name_singular': 'Plate',
                'text_reader': 'ocr',
            },
        },
    )
    assert created.status_code == 201, created.text
    activated = api.post(f'{project.prefix}/region_profiles/{PROFILE}/activate', json={})
    assert activated.status_code == 200, activated.text

    uploaded = api.post(
        f'{project.prefix}/ingest/upload',
        files=[
            ('images', (f'p{i}.jpg', picture((40 * i, 90, 200 - 30 * i)), 'image/jpeg'))
            for i in range(1, 4)
        ],
    )
    assert uploaded.status_code == 200, uploaded.text
    assert uploaded.json()['summary']['successful'] == 3, uploaded.text

    def boxes() -> list[dict[str, Any]]:
        opensearch.post(f'/{project.items_index}/_refresh')
        hits = opensearch.post(
            f'/{project.items_index}/_search',
            json={'size': 20, 'query': {'match_all': {}}},
        ).json()['hits']['hits']
        docs = [{'crop_id': h['_id'], **h['_source']} for h in hits]
        read = [d for d in docs if any(b.get('text') for b in d.get('region_boxes') or [])]
        return read if len(read) == len(docs) and docs else []

    items = wait_until(boxes, timeout=180, interval=2.0)
    assert items, 'the detection worker never wrote text on every item'
    return list(items)


def test_the_worker_stores_ocr_text_with_its_source(read_items: list[dict[str, Any]]) -> None:
    assert len(read_items) == 3
    for item in read_items:
        (box,) = item['region_boxes']
        assert box['text'] == OCR_TEXT, box
        assert box['text_source'] == 'ocr', box
        assert box['text_confidence'] == pytest.approx(0.97, abs=0.01), box
        assert item['region_status'] in ('detected', 'pending_verification'), item['region_status']


def test_region_text_filters_find_the_items_the_worker_read(
    api: Any, project: IngestProject, read_items: list[dict[str, Any]]
) -> None:
    every = {i['crop_id'] for i in read_items}

    def found(**params: Any) -> set[str]:
        resp = api.get(f'{project.prefix}/crops', params={'limit': 100, **params})
        assert resp.status_code == 200, resp.text
        return {c['crop_id'] for c in resp.json()['crops']}

    assert found(region_text='FAKEPLATE0042', region_text_match='exact') == every
    assert found(region_text='FAKEPLATE0043', region_text_match='fuzzy') == every
    assert found(region_text='PLATE00', region_text_match='fragment') == every
    assert found(region_text='ZZZZ9999', region_text_match='exact') == set()
