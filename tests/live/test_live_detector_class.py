"""Live: the detector's own class on ingested items.

Ingest gives an item the class named like its detector label as a machine label
(``class_source`` ``detector``, never validated) and adds the registry class when it is missing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from ._ingest_support import IngestProject, count, ingest_project, picture


if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = pytest.mark.live

COLOURS = [(200, 30, 30)]


@pytest.fixture(scope='module')
def project(api: Any, opensearch: Any) -> Iterator[IngestProject]:
    with ingest_project(api, opensearch, 'detcls') as handle:
        yield handle


def _items(opensearch: Any, project: IngestProject) -> list[dict[str, Any]]:
    count(opensearch, project.items_index)
    hits = opensearch.post(
        f'/{project.items_index}/_search', json={'size': 100, 'query': {'match_all': {}}}
    ).json()['hits']['hits']
    return [{'_id': h['_id'], **h['_source']} for h in hits]


def _upload(api: Any, project: IngestProject, colour: tuple[int, int, int], name: str) -> None:
    resp = api.post(
        f'{project.prefix}/ingest/upload',
        files=[('images', (f'{name}.jpg', picture(colour), 'image/jpeg'))],
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()['summary']['successful'] == 1, resp.text


def _class_names(api: Any, project: IngestProject) -> set[str]:
    resp = api.get(f'{project.prefix}/classes')
    assert resp.status_code == 200, resp.text
    return {c['class_name'] for c in resp.json()['classes']}


def test_ingest_assigns_the_detector_class_and_creates_the_registry_class(
    api: Any, opensearch: Any, project: IngestProject
) -> None:
    assert api.get(f'{project.prefix}/ingest/policy').json()['detect']['class_resolution'] == (
        'by_name'
    )
    _upload(api, project, COLOURS[0], 'first')
    items = [i for i in _items(opensearch, project) if i.get('detector_class_name')]
    assert items
    names = _class_names(api, project)
    for item in items:
        assert item['class_name'] == item['detector_class_name']
        assert item['class_source'] == 'detector'
        assert item['class_validated'] is False
        assert item['class_name'] in names
    # A machine label is reviewable ...
    queue = api.get(f'{project.prefix}/review/check_machine')
    assert queue.status_code == 200, queue.text
    assert {i['crop_id'] for i in queue.json()['items']} >= {i['crop_id'] for i in items}
    # ... and the class source is in the served catalog as a machine one.
    catalog = api.get(f'{project.prefix}/class_sources').json()['class_sources']
    assert {'id': 'detector', 'role': 'detector'}.items() <= next(
        c for c in catalog if c['id'] == 'detector'
    ).items()
