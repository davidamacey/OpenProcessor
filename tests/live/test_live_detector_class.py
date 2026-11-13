"""Live: the detector's own class on ingested items, and adopting it for items stored without one.

Ingest gives an item the class named like its detector label as a machine label
(``class_source`` ``detector``, never validated) and adds the registry class when it is missing.
``POST /label_ops/adopt_detector`` does the same for items stored before (here: stored while the
policy kept detections as proposals), as a recorded operation that Label history lists and that
``POST /label_ops/{op_id}/undo`` takes back. A person's class is never overwritten.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from ._ingest_support import IngestProject, count, ingest_project, picture
from .conftest import wait_job, wait_until


if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = pytest.mark.live

COLOURS = [(200, 30, 30), (30, 200, 30), (30, 30, 200), (220, 220, 30)]


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


def _set_resolution(api: Any, project: IngestProject, resolution: str) -> None:
    current = api.get(f'{project.prefix}/ingest/policy').json()
    resp = api.put(
        f'{project.prefix}/ingest/policy',
        json={
            'expected_revision': current['revision'],
            'detect': {**current['detect'], 'class_resolution': resolution},
        },
    )
    assert resp.status_code == 200, resp.text


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


def test_adopt_gives_stored_items_their_detector_class_and_can_be_undone(
    api: Any, opensearch: Any, project: IngestProject
) -> None:
    _set_resolution(api, project, 'proposal')
    _upload(api, project, COLOURS[1], 'second')
    _upload(api, project, COLOURS[2], 'third')
    undecided = [
        i
        for i in _items(opensearch, project)
        if i.get('detector_class_name') and 'class_id' not in i
    ]
    assert len(undecided) == 2, undecided

    # A person's class on one of them is never overwritten.
    human = undecided[0]
    labeled = api.put(
        f'{project.prefix}/crops/{human["crop_id"]}/label',
        json={'class_id': _any_class_id(api, project), 'label_source': 'human'},
    )
    assert labeled.status_code == 200, labeled.text
    human_class = labeled.json()['class_name']

    dry = api.post(f'{project.prefix}/label_ops/adopt_detector', json={})
    assert dry.status_code == 200, dry.text
    assert dry.json()['dry_run'] is True
    assert dry.json()['items'] == 1

    started = api.post(f'{project.prefix}/label_ops/adopt_detector', json={'dry_run': False})
    assert started.status_code == 202, started.text
    op_id = started.json()['op_id']
    job = wait_job(api, started.json()['job_id'], prefix=project.prefix)
    assert job['status'] == 'completed', job

    count(opensearch, project.items_index)
    after = {i['crop_id']: i for i in _items(opensearch, project)}
    adopted = after[undecided[1]['crop_id']]
    assert adopted['class_name'] == adopted['detector_class_name']
    assert adopted['class_source'] == 'detector'
    assert adopted['class_validated'] is False
    assert after[human['crop_id']]['class_name'] == human_class
    assert after[human['crop_id']]['class_source'] == 'human'

    ops = api.get(f'{project.prefix}/label_ops').json()['items']
    mine = next(o for o in ops if o['op_id'] == op_id)
    assert mine['kind'] == 'adopt_detector'
    assert mine['undoable'] is True

    undone = api.post(f'{project.prefix}/label_ops/{op_id}/undo', json={'dry_run': False})
    assert undone.status_code == 202, undone.text
    assert wait_job(api, undone.json()['job_id'], prefix=project.prefix)['status'] == 'completed'
    assert wait_until(
        lambda: (
            'class_id'
            not in {i['crop_id']: i for i in _items(opensearch, project)}[undecided[1]['crop_id']]
        )
    )
    restored = {i['crop_id']: i for i in _items(opensearch, project)}
    assert restored[human['crop_id']]['class_source'] == 'human'

    # Adopt again (the undo put the item back): the next steps stop offering the action.
    again = api.post(f'{project.prefix}/label_ops/adopt_detector', json={'dry_run': False})
    assert again.status_code == 202, again.text
    assert wait_job(api, again.json()['job_id'], prefix=project.prefix)['status'] == 'completed'
    steps = api.get(f'{project.prefix}/next_steps').json()['steps']
    classes = next(s for s in steps if s['id'] == 'classes')
    assert classes['done'] is True
    assert classes['action'] is None


def _any_class_id(api: Any, project: IngestProject) -> int:
    classes = api.get(f'{project.prefix}/classes').json()['classes']
    return int(classes[0]['class_id'])
