"""``dataset_import`` and ``dataset_import_undo`` as queued jobs on ``dataset_imports`` and its
ledger (4B-3).

The routes insert the import rows and the job in one transaction and answer 202; nothing is
scanned, resumed or undone in the request. The worker scans the dataset again at the start of
every attempt, claims pending ledger rows in chunks and imports them.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import TYPE_CHECKING, Any

import pytest
from curation.dataset_import.harness import write_yolo
from curation.query_fakes import QueryFakeOpenSearch
from families_b.support import bind_worker_to, project
from fixtures.job_rows import sql
from integration.ingest_fakes import FakePEEncoder, FakeTritonPool

from src.config import DetectionProfile
from src.services.curation.dataset_import import job as import_job, runner


if TYPE_CHECKING:
    from pathlib import Path


pytestmark = pytest.mark.db


@pytest.fixture(autouse=True)
def _environment(pg, monkeypatch, tmp_path):
    from src.services.curation import image_serving

    monkeypatch.setenv('OP_PROJECTS_DATA_ROOT', str(tmp_path / 'projects_data'))
    monkeypatch.setenv('OP_STATE_DIR', str(tmp_path / 'state'))
    monkeypatch.setattr(
        image_serving,
        '_configured_roots',
        lambda config=None: (tmp_path.resolve(),),  # noqa: ARG005
    )


@pytest.fixture
def worker(app, pg, worker_app, monkeypatch, tmp_path):
    """The task side: a fake OpenSearch, Triton and encoder instead of the real clients."""
    fake = QueryFakeOpenSearch()
    app.state.opensearch = fake
    bind_worker_to(monkeypatch, app, fake)

    class Runtime:
        pool = FakeTritonPool()
        pe_encoder = FakePEEncoder()

        async def close(self) -> None:
            return None

    async def open_runtime() -> Any:
        return Runtime()

    async def service(opensearch, runtime, *, need_detector):
        from src.services.classes.registry import get_class_registry
        from src.services.curation.ingest import CurationIngestService

        return CurationIngestService(
            opensearch=opensearch,
            triton_pool=runtime.pool,
            registry=await get_class_registry().snapshot(),
            profile=DetectionProfile(
                name='primary', detector_model='fake_item_detector', assigns_class=False
            ),
            pe_encoder=runtime.pe_encoder,
        )

    monkeypatch.setattr(import_job, 'open_ingest_runtime', open_runtime)
    monkeypatch.setattr(import_job, '_import_service', service)
    return fake


def _dataset(root: Path, stems: tuple[str, ...] = ('a', 'b', 'c'), cls: str = 'car') -> None:
    write_yolo(root, names=[cls], images={stem: ['0 0.3 0.3 0.2 0.2'] for stem in stems})


def _body(root: Path, cls: str = 'car') -> dict[str, Any]:
    return {
        'source': {'path': str(root)},
        'mapping': [{'dataset_class': cls, 'action': 'create', 'new_class_name': cls}],
        'options': {},
    }


async def _start(client, slug, root, cls: str = 'car') -> Any:
    return await client.post(f'/curation/projects/{slug}/datasets/imports', json=_body(root, cls))


async def _run(worker_app, name: str, job_id: str) -> Any:
    return await asyncio.to_thread(
        worker_app.tasks[f'op.families_b.{name}'].apply, args=[job_id], task_id=job_id
    )


def _setup(app, pg, tmp_path):
    slug, pid = project(app, pg)
    root = tmp_path / 'ds'
    _dataset(root)
    return slug, pid, root


@pytest.mark.asyncio
async def test_the_start_route_inserts_the_rows_and_one_queued_job_and_scans_nothing_more(
    app, client, pg, producer, tmp_path
) -> None:
    slug, pid, root = _setup(app, pg, tmp_path)

    reply = await _start(client, slug, root)

    assert reply.status_code == 202, reply.text
    body = reply.json()
    assert body['status'] == 'queued'
    assert producer.sent() == [body['job_id']]
    [(kind, status)] = sql(pg, 'SELECT kind, status FROM jobs WHERE project_id = %s', pid)
    assert (kind, status) == ('dataset_import', 'queued')
    [(job, scan, mapping, source)] = sql(
        pg,
        'SELECT job_id, scan, mapping, source_key FROM dataset_imports WHERE id = %s',
        uuid.UUID(body['import_id']),
    )
    assert str(job) == body['job_id']
    assert scan['images_total'] == 3
    assert scan['source_sha']
    assert source == str(root)
    assert mapping['created_classes']['car'] is not None  # the class ids are pinned at the start
    assert sql(pg, 'SELECT count(*) FROM dataset_import_images') == [(0,)]  # the task seeds them


@pytest.mark.asyncio
async def test_a_second_start_while_an_import_is_active_is_409_import_busy(
    app, client, pg, producer, tmp_path
) -> None:
    slug, _pid, root = _setup(app, pg, tmp_path)
    first = (await _start(client, slug, root)).json()
    other = tmp_path / 'ds2'
    _dataset(other, ('x',), 'truck')

    reply = await _start(client, slug, other, 'truck')

    assert reply.status_code == 409
    assert reply.json()['detail']['error'] == 'import_busy'
    assert reply.json()['detail']['import_id'] == first['import_id']


@pytest.mark.asyncio
async def test_broker_down_is_503_and_the_job_fails(app, client, pg, producer, tmp_path) -> None:
    slug, pid, root = _setup(app, pg, tmp_path)
    producer.fail_send = True

    reply = await _start(client, slug, root)

    assert reply.status_code == 503
    assert reply.json()['detail']['error'] == 'queue_unavailable'
    assert sql(pg, 'SELECT status FROM jobs WHERE project_id = %s', pid) == [('failed',)]


@pytest.mark.asyncio
async def test_the_task_imports_through_the_wrapper_and_records_the_ledger(
    app, client, pg, producer, worker_app, worker, tmp_path
) -> None:
    slug, _pid, root = _setup(app, pg, tmp_path)
    started = (await _start(client, slug, root)).json()

    outcome = await _run(worker_app, 'dataset_import', started['job_id'])

    assert outcome.successful(), outcome.traceback
    [(status, result)] = sql(
        pg, 'SELECT status, result FROM jobs WHERE id = %s', uuid.UUID(started['job_id'])
    )
    assert status == 'completed'
    assert result['images_total'] == 3
    ledger = sql(
        pg,
        "SELECT status, image_id, cardinality(item_ids), detail->>'split' "
        'FROM dataset_import_images WHERE import_id = %s ORDER BY image_key',
        uuid.UUID(started['import_id']),
    )
    assert [r[0] for r in ledger] == ['written'] * 3
    assert all(r[1] and r[2] == 1 for r in ledger)
    assert len(worker.docs(_items_index(app, slug))) == 3
    pending = sql(pg, "SELECT level, member_id, reason FROM near_dup_pending WHERE level = 'image'")
    assert sorted(member for _level, member, _reason in pending) == sorted(r[1] for r in ledger)
    assert {(level, reason) for level, _member, reason in pending} == {('image', 'added')}
    done = (
        await client.get(f'/curation/projects/{slug}/datasets/imports/{started["import_id"]}')
    ).json()
    assert done['status'] == 'completed'
    assert done['report']['images_created'] == 3
    assert done['progress']['images_done'] == 3
    entries = (
        await client.get(
            f'/curation/projects/{slug}/datasets/imports/{started["import_id"]}/entries'
        )
    ).json()
    assert entries['total'] == 3
    assert {e['status'] for e in entries['items']} == {'ok'}


def _items_index(app, slug) -> str:
    from src.config.curation_roles import IndexRole

    return app.state.records[slug].resources.indexes[IndexRole.ITEMS]


@pytest.mark.asyncio
async def test_cancel_while_queued_cancels_the_job_and_the_import(
    app, client, pg, producer, tmp_path
) -> None:
    slug, _pid, root = _setup(app, pg, tmp_path)
    started = (await _start(client, slug, root)).json()

    reply = await client.post(
        f'/curation/projects/{slug}/datasets/imports/{started["import_id"]}/cancel'
    )

    assert reply.status_code == 200
    assert reply.json()['status'] == 'cancelled'
    assert sql(pg, 'SELECT status FROM jobs WHERE id = %s', uuid.UUID(started['job_id'])) == [
        ('cancelled',)
    ]


@pytest.mark.asyncio
async def test_a_cancelled_import_resumes_as_a_new_job_without_a_rescan_in_the_request(
    app, client, pg, producer, worker_app, worker, monkeypatch, tmp_path
) -> None:
    slug, _pid, root = _setup(app, pg, tmp_path)
    started = (await _start(client, slug, root)).json()
    await client.post(f'/curation/projects/{slug}/datasets/imports/{started["import_id"]}/cancel')
    scans: list[Path] = []
    real = runner.scan_dataset
    monkeypatch.setattr(
        runner, 'scan_dataset', lambda source, **kw: scans.append(source) or real(source, **kw)
    )

    reply = await client.post(
        f'/curation/projects/{slug}/datasets/imports/{started["import_id"]}/resume'
    )

    assert reply.status_code == 202, reply.text
    assert scans == []  # the route scanned nothing: the task does
    resumed = reply.json()
    assert resumed['status'] == 'queued'
    assert resumed['job_id'] != started['job_id']
    assert producer.sent()[-1] == resumed['job_id']
    assert sql(
        pg, 'SELECT job_id FROM dataset_imports WHERE id = %s', uuid.UUID(started['import_id'])
    ) == [(uuid.UUID(resumed['job_id']),)]

    outcome = await _run(worker_app, 'dataset_import', resumed['job_id'])

    assert outcome.successful(), outcome.traceback
    assert len(scans) == 1  # once, in the task
    assert (
        await client.get(f'/curation/projects/{slug}/datasets/imports/{started["import_id"]}')
    ).json()['status'] == 'completed'


@pytest.mark.asyncio
async def test_an_interrupted_import_resumes_under_its_own_job_and_continues_the_ledger(
    app, client, pg, producer, worker_app, worker, monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv('OP_DATASET_IMPORT_CHUNK', '1')
    slug, pid, root = _setup(app, pg, tmp_path)
    started = (await _start(client, slug, root)).json()
    job = uuid.UUID(started['job_id'])
    real = import_job.run_import_job

    async def crash_after_one_chunk(control, ctx, entries):
        async def die(_idx: int) -> None:
            raise SystemExit('worker killed')  # the wrapper interrupts the job on an exit

        ctx.after_chunk = die
        return await real(control, ctx, entries)

    monkeypatch.setattr(import_job, 'run_import_job', crash_after_one_chunk)
    with pytest.raises(SystemExit):  # Celery requeues the message of an exited worker
        await _run(worker_app, 'dataset_import', started['job_id'])
    assert sql(pg, 'SELECT status FROM jobs WHERE id = %s', job) == [('interrupted',)]
    ledger = lambda: [  # noqa: E731
        r[0]
        for r in sql(
            pg,
            'SELECT status FROM dataset_import_images WHERE import_id = %s',
            uuid.UUID(started['import_id']),
        )
    ]
    assert sorted(ledger()) == ['pending', 'pending', 'written']
    monkeypatch.setattr(import_job, 'run_import_job', real)

    reply = await client.post(
        f'/curation/projects/{slug}/datasets/imports/{started["import_id"]}/resume'
    )

    assert reply.status_code == 202, reply.text
    assert reply.json()['job_id'] == started['job_id']  # the same job, resumed
    assert sql(pg, 'SELECT count(*) FROM jobs WHERE project_id = %s', pid) == [(1,)]
    outcome = await _run(worker_app, 'dataset_import', started['job_id'])
    assert outcome.successful(), outcome.traceback
    assert ledger() == ['written'] * 3
    assert len(worker.docs(_items_index(app, slug))) == 3  # no duplicate writes


@pytest.mark.asyncio
async def test_a_dataset_that_changed_fails_the_job_and_a_restored_one_resumes(
    app, client, pg, producer, worker_app, worker, tmp_path
) -> None:
    slug, _pid, root = _setup(app, pg, tmp_path)
    started = (await _start(client, slug, root)).json()
    label = root / 'labels/train/b.txt'
    original = label.read_text()
    label.write_text('0 0.5 0.5 0.3 0.3\n')

    await _run(worker_app, 'dataset_import', started['job_id'])

    [(status, message)] = sql(
        pg,
        'SELECT status, error_message FROM jobs WHERE id = %s',
        uuid.UUID(started['job_id']),
    )
    assert status == 'failed'
    assert 'DatasetChangedError' in message
    assert not worker.docs(_items_index(app, slug))  # nothing was written
    label.write_text(original)
    resumed = (
        await client.post(
            f'/curation/projects/{slug}/datasets/imports/{started["import_id"]}/resume'
        )
    ).json()
    assert (await _run(worker_app, 'dataset_import', resumed['job_id'])).successful()
    assert len(worker.docs(_items_index(app, slug))) == 3


@pytest.mark.asyncio
async def test_undo_is_a_queued_job_that_removes_what_the_import_wrote(
    app, client, pg, producer, worker_app, worker, tmp_path
) -> None:
    slug, _pid, root = _setup(app, pg, tmp_path)
    started = (await _start(client, slug, root)).json()
    assert (await _run(worker_app, 'dataset_import', started['job_id'])).successful()
    base = f'/curation/projects/{slug}/datasets/imports/{started["import_id"]}'

    dry = await client.post(f'{base}/undo', json={'dry_run': True})
    assert dry.status_code == 200, dry.text
    assert dry.json()['items_deleted'] == 3
    assert len(worker.docs(_items_index(app, slug))) == 3  # a dry run changes nothing

    reply = await client.post(f'{base}/undo', json={'dry_run': False})

    assert reply.status_code == 202, reply.text
    body = reply.json()
    assert body['status'] == 'undoing'
    assert producer.sent()[-1] == body['job_id']
    [(kind,)] = sql(pg, 'SELECT kind FROM jobs WHERE id = %s', uuid.UUID(body['job_id']))
    assert kind == 'dataset_import_undo'
    again = await client.post(f'{base}/undo', json={'dry_run': False})
    assert again.status_code == 409  # an undo is running

    assert (await _run(worker_app, 'dataset_import_undo', body['job_id'])).successful()

    assert not worker.docs(_items_index(app, slug))
    after = (await client.get(base)).json()
    assert after['status'] == 'undone'
    assert after['undo']['items_deleted'] == 3
    ledger = sql(
        pg,
        'SELECT DISTINCT status FROM dataset_import_images WHERE import_id = %s',
        uuid.UUID(started['import_id']),
    )
    assert ledger == [('undone',)]
    assert sql(
        pg,
        'SELECT undone_at IS NOT NULL FROM dataset_imports WHERE id = %s',
        uuid.UUID(started['import_id']),
    ) == [(True,)]


@pytest.mark.asyncio
async def test_a_repeat_of_a_completed_import_is_200_reused_and_queues_nothing(
    app, client, pg, producer, worker_app, worker, tmp_path
) -> None:
    slug, pid, root = _setup(app, pg, tmp_path)
    started = (await _start(client, slug, root)).json()
    assert (await _run(worker_app, 'dataset_import', started['job_id'])).successful()
    sent = list(producer.sent())

    again = await _start(client, slug, root)  # its 'create' now names a class that exists

    assert again.status_code == 200, again.text
    assert again.json()['reused'] is True
    assert again.json()['import_id'] == started['import_id']
    assert producer.sent() == sent
    assert sql(pg, 'SELECT count(*) FROM jobs WHERE project_id = %s', pid) == [(1,)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('mutate', 'status', 'code'),
    [
        (lambda b: b.update(mapping=[]), 422, 'class_mapping_incomplete'),
        (
            lambda b: b['mapping'].append({'dataset_class': 'nope', 'action': 'skip'}),
            422,
            'class_mapping_invalid',
        ),
        (lambda b: b.update(expected_import_key='0' * 64), 409, 'dataset_changed'),
        (lambda b: b['source'].update(path='/etc'), 422, 'dataset_path_not_allowed'),
        (lambda b: b['source'].update(path='/nonexistent/x'), 422, 'dataset_path_not_allowed'),
        (lambda b: b.update(surprise=True), 422, None),
    ],
)
async def test_start_rejects_every_bad_request_before_any_job(
    app, client, pg, producer, tmp_path, mutate, status: int, code: str | None
) -> None:
    slug, pid, root = _setup(app, pg, tmp_path)
    body = _body(root)
    mutate(body)

    resp = await client.post(f'/curation/projects/{slug}/datasets/imports', json=body)

    assert resp.status_code == status, resp.text
    if code is not None:
        assert resp.json()['detail']['error'] == code
    assert sql(pg, 'SELECT count(*) FROM jobs WHERE project_id = %s', pid) == [(0,)]
    assert producer.sent() == []


@pytest.mark.asyncio
async def test_a_bad_mapping_row_says_what_is_wrong(app, client, pg, producer, tmp_path) -> None:
    slug, _pid, root = _setup(app, pg, tmp_path)
    body = _body(root)
    body['mapping'].append({'dataset_class': 'nope', 'action': 'skip'})

    detail = (await client.post(f'/curation/projects/{slug}/datasets/imports', json=body)).json()[
        'detail'
    ]

    assert detail['error'] == 'class_mapping_invalid'
    assert (
        'nope: class_mapping_invalid (this dataset has no class with that name' in detail['message']
    )


@pytest.mark.asyncio
async def test_a_blocking_issue_refuses_the_import(app, client, pg, producer, tmp_path) -> None:
    slug, _pid = project(app, pg)
    root = tmp_path / 'broken'
    root.mkdir()
    (root / 'data.yaml').write_text('train: images/train\nnames: [unterminated\n')

    resp = await client.post(
        f'/curation/projects/{slug}/datasets/imports', json={'source': {'path': str(root)}}
    )

    assert resp.status_code == 422, resp.text
    detail = resp.json()['detail']
    assert detail['error'] == 'import_blocked'
    assert 'data_yaml_invalid' in {i['code'] for i in detail['issues']}


@pytest.mark.asyncio
async def test_an_id_that_is_not_an_import_id_is_404_on_every_route(
    app, client, pg, producer, tmp_path
) -> None:
    slug, _pid, _root = _setup(app, pg, tmp_path)
    base = f'/curation/projects/{slug}/datasets/imports'
    for bad in ('..', '%2e%2e', 'imp_x', 'imp_20260101T000000_zzzzzzzz', 'nope'):
        assert (await client.get(f'{base}/{bad}')).status_code == 404
        assert (await client.post(f'{base}/{bad}/cancel')).status_code == 404
        assert (await client.post(f'{base}/{bad}/resume')).status_code == 404
        assert (await client.post(f'{base}/{bad}/undo', json={'dry_run': True})).status_code == 404
    assert (await client.get(f'{base}/nope')).json()['detail']['error'] == 'import_not_found'
