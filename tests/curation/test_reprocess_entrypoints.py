"""Reprocess entry points: the routes, the file-backed job, the CLI, the
startup repair and the busy inventory. One test walks every entry point
with a locked input that must be skipped and counted."""

from __future__ import annotations

import asyncio
import copy
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from curation.reprocess_fixtures import (
    F,
    FakePE,
    FakeTriton,
    RegionStatus,
    box,
    docs,
    images_index,
    item,
    jpeg_bytes,
    make_fake,
    make_service,
    servable_root,
)
from src.services.curation import reprocess_job
from src.services.curation.reprocess import apply_reprocess
from src.services.curation.reprocess_models import ReprocessRequest, ReprocessTargets


FAILED = RegionStatus.DETECTION_FAILED.value
LOCKED_SET = {'validated': True, 'verifier': 'human'}


@pytest.fixture(autouse=True)
def _jobs_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('OP_REPROCESS_JOBS_DIR', str(tmp_path / 'jobs'))


def _factory(service: Any) -> Any:
    async def build() -> Any:
        return service

    return build


def _client(fake: Any, monkeypatch: pytest.MonkeyPatch, service: Any = None) -> TestClient:
    from _curation_app import mount_curation_routers

    from src.routers.curation import _raw_opensearch_dep, reprocess as route_mod, router

    app = FastAPI()
    mount_curation_routers(app, router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake

    async def factory(_opensearch: Any, _registry: Any) -> Any:
        return service

    monkeypatch.setattr(route_mod, '_service_factory', factory)
    return TestClient(app).__enter__()


BASE = '/curation/projects/default'


def _region_items() -> list[dict[str, Any]]:
    return [
        item('m1', FAILED, boxes=(box('b1'),)),
        item('locked', FAILED, boxes=(box('b2', state='accepted', source='import'),),
             validated=True, verifier='import'),
    ]  # fmt: skip


# ------------------------------------------------------------------- routes


def test_batch_route_is_a_dry_run_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = make_fake(_region_items())
    before = copy.deepcopy(docs(fake))
    client = _client(fake, monkeypatch)
    body = {'targets': {'filter': {'region_status': [FAILED]}}, 'scopes': ['region']}
    r = client.post(f'{BASE}/reprocess', json=body)
    assert r.status_code == 200, r.text
    out = r.json()
    assert out['dry_run'] is True
    assert (out['scopes'][0]['selected'], out['scopes'][0]['locked_skipped']) == (2, 1)
    assert docs(fake) == before

    applied = client.post(f'{BASE}/reprocess', json={**body, 'dry_run': False})
    assert applied.json()['scopes'][0]['queued'] == 1
    assert docs(fake)['m1'][F.status] == RegionStatus.PENDING_DETECTION


@pytest.mark.parametrize(
    'targets',
    [{}, {'crop_ids': ['a'], 'image_ids': ['b']}, {'filter': {}}, {'crop_ids': []}],
)
def test_malformed_targets_are_a_structured_422(
    monkeypatch: pytest.MonkeyPatch, targets: dict[str, Any]
) -> None:
    client = _client(make_fake([]), monkeypatch)
    r = client.post(f'{BASE}/reprocess', json={'targets': targets, 'scopes': ['region']})
    assert r.status_code == 422
    assert r.json()['detail']['error'] == 'reprocess_targets_invalid'


def test_unknown_body_key_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(make_fake([]), monkeypatch)
    r = client.post(
        f'{BASE}/reprocess',
        json={'targets': {'crop_ids': ['a']}, 'scopes': ['region'], 'clear_detection': True},
    )
    assert r.status_code == 422


def test_single_crop_route_applies_by_default_and_returns_the_item(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = make_fake(_region_items())
    client = _client(fake, monkeypatch)
    r = client.post(f'{BASE}/crops/m1/reprocess', json={'scopes': ['region']})
    assert r.status_code == 200, r.text
    out = r.json()
    assert out['dry_run'] is False
    assert out['scopes'][0]['queued'] == 1
    assert [i['crop_id'] for i in out['items']] == ['m1']
    assert docs(fake)['m1'][F.status] == RegionStatus.PENDING_DETECTION

    dry = client.post(
        f'{BASE}/crops/locked/reprocess', json={'scopes': ['region'], 'dry_run': True}
    )
    assert dry.json()['scopes'][0]['locked_skipped'] == 1


def test_single_routes_404_for_unknown_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(make_fake(_region_items()), monkeypatch)
    crop = client.post(f'{BASE}/crops/nope/reprocess', json={'scopes': ['region']})
    assert (crop.status_code, crop.json()['detail']['error']) == (404, 'not_found')
    image = client.post(f'{BASE}/images/nope/reprocess', json={'scopes': ['region']})
    assert (image.status_code, image.json()['detail']['error']) == (404, 'image_not_found')


def test_single_image_route_applies_region_to_the_images_items(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = make_fake(_region_items(), [{'image_id': 'img-1', 'image_path': '/x.jpg'}])
    client = _client(fake, monkeypatch)
    r = client.post(f'{BASE}/images/img-1/reprocess', json={'scopes': ['region']})
    assert r.status_code == 200, r.text
    assert r.json()['scopes'][0]['queued'] == 1
    assert {i['crop_id'] for i in r.json()['items']} == {'m1', 'locked'}


# --------------------------------------------------------------- job + cold


def _embed_world(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, n: int = 3
) -> tuple[Any, list[str], Any]:
    root = servable_root(tmp_path, monkeypatch)
    items, images, ids = [], [], []
    for i in range(n):
        path = root / f'{i}.jpg'
        path.write_bytes(jpeg_bytes(seed=i))
        items.append(item(f'c{i}', image_id=f'img-{i}', image_path=str(path)))
        images.append({'image_id': f'img-{i}', 'image_path': str(path)})
        ids.append(f'img-{i}')
    fake = make_fake(items, images)
    return fake, ids, make_service(fake, FakeTriton([]), FakePE())


@pytest.mark.asyncio
async def test_many_images_run_as_a_job_and_a_second_start_is_busy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('OP_REPROCESS_SYNC_MAX', '1')
    fake, ids, service = _embed_world(tmp_path, monkeypatch)
    request = ReprocessRequest(
        targets=ReprocessTargets(image_ids=ids), scopes=['embed'], dry_run=False
    )
    resp = await apply_reprocess(fake, request, service_factory=_factory(service))
    assert resp.job is not None
    assert resp.scopes[0].queued == 0  # the job, not this call, does the work
    job_id = resp.job.job_id
    # the job is live until its task finishes: a second start is refused
    with pytest.raises(reprocess_job.ReprocessBusyError):
        await apply_reprocess(fake, request, service_factory=_factory(service))
    await reprocess_job._tasks[job_id]
    done = reprocess_job.read_job(job_id)
    assert done is not None
    assert (done.status, done.images_done, done.images_total) == ('completed', 3, 3)
    assert done.results[0].queued == 3
    assert all(docs(fake)[f'c{i}']['pe_embedding'] == [0.0, 0.0, 1.0] for i in range(3))
    assert done.poll_after_s is None


@pytest.mark.asyncio
async def test_a_cold_process_runs_the_request_an_api_call_persisted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The API persisted ``request.json``; a process that never saw the call
    (fresh OpenSearch client, fresh service) consumes it from the files."""
    fake, ids, _api_service = _embed_world(tmp_path, monkeypatch)
    job = reprocess_job.create_job(request={'scopes': ['embed']}, scopes=['embed'], image_ids=ids)
    cold_fake = make_fake(list(docs(fake).values()), list(fake.docs(images_index()).values()))
    cold_service = make_service(cold_fake, FakeTriton([]), FakePE())

    await reprocess_job.run_job(job, cold_fake, cold_service)

    state = reprocess_job.read_job(job.directory.name)
    assert state is not None
    assert state.status == 'completed'
    assert docs(cold_fake)['c0']['pe_embedding'] == [0.0, 0.0, 1.0]


@pytest.mark.asyncio
async def test_a_job_planned_for_other_indexes_refuses_to_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake, ids, service = _embed_world(tmp_path, monkeypatch)
    job = reprocess_job.create_job(request={}, scopes=['embed'], image_ids=ids)
    path = job.directory / 'request.json'
    payload = json.loads(path.read_text())
    payload['items_index'] = 'op_prj_other__items'
    path.write_text(json.dumps(payload))
    await reprocess_job.run_job(job, fake, service)
    state = reprocess_job.read_job(job.directory.name)
    assert state is not None
    assert state.status == 'failed'
    assert 'indexes' in (state.error or '')
    assert 'pe_embedding' not in docs(fake)['c0']


@pytest.mark.asyncio
async def test_cancel_stops_a_job_before_its_next_chunk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake, ids, service = _embed_world(tmp_path, monkeypatch)
    job = reprocess_job.create_job(request={}, scopes=['embed'], image_ids=ids)
    assert reprocess_job.cancel_job(job.directory.name) is True
    await reprocess_job.run_job(job, fake, service)
    state = reprocess_job.read_job(job.directory.name)
    assert state is not None
    assert state.status == 'cancelled'
    assert 'pe_embedding' not in docs(fake)['c0']
    assert reprocess_job.cancel_job(job.directory.name) is False  # no longer live
    assert reprocess_job.cancel_job('../escape') is False


def test_job_status_and_cancel_routes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client(make_fake([]), monkeypatch)
    assert client.get(f'{BASE}/reprocess/jobs/rp_nope').status_code == 404
    job = reprocess_job.create_job(request={}, scopes=['embed'], image_ids=['a'])
    got = client.get(f'{BASE}/reprocess/jobs/{job.directory.name}')
    assert got.status_code == 200
    assert got.json()['status'] == 'queued'
    cancelled = client.post(f'{BASE}/reprocess/jobs/{job.directory.name}/cancel')
    assert cancelled.status_code == 200
    assert job.cancel_requested()


def test_busy_route_error_is_a_409(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('OP_REPROCESS_SYNC_MAX', '1')
    fake, ids, service = _embed_world(tmp_path, monkeypatch)
    reprocess_job.create_job(request={}, scopes=['embed'], image_ids=ids)  # a live job
    client = _client(fake, monkeypatch, service)
    r = client.post(
        f'{BASE}/reprocess',
        json={'targets': {'image_ids': ids}, 'scopes': ['embed'], 'dry_run': False},
    )
    assert (r.status_code, r.json()['detail']['error']) == (409, 'reprocess_busy')


def test_orphaned_job_is_marked_interrupted_at_startup_and_busy_sees_live_ones() -> None:
    from datetime import UTC, datetime

    from src.config.curation import base_curation_config
    from src.config.projects import ProjectRecord, resources_for_new
    from src.services.projects.busy import _reprocess_jobs

    job = reprocess_job.create_job(request={}, scopes=['embed'], image_ids=['a'])
    now = datetime.now(UTC).isoformat()
    record = ProjectRecord(
        slug='default', display_name='d', description='', status='active', revision=1,
        created_at=now, updated_at=now, origin=None,
        resources=resources_for_new('default', base_curation_config()),
    )  # fmt: skip
    assert [j.job_id for j in _reprocess_jobs(record)] == [job.directory.name]

    job.heartbeat_file.unlink()  # the owning process died before ticking
    assert reprocess_job.reconcile_orphaned_jobs() is True
    state = reprocess_job.read_job(job.directory.name)
    assert state is not None
    assert state.status == 'interrupted'
    assert _reprocess_jobs(record) == []


# ------------------------------------------------- every entry point, locked


def _load_cli() -> Any:
    path = Path(__file__).resolve().parents[2] / 'scripts' / 'curation' / 'requeue_regions.py'
    spec = importlib.util.spec_from_file_location('requeue_cli_locked_test', path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Closable:
    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    async def close(self) -> None:
        return None


def _locked_world() -> tuple[Any, dict[str, dict[str, Any]]]:
    fake = make_fake(
        [
            item('locked', FAILED, boxes=(box('b2', state='accepted', source='import'),),
                 validated=True, verifier='import'),
        ],
        [{'image_id': 'img-1', 'image_path': '/x.jpg'}],
    )  # fmt: skip
    return fake, copy.deepcopy(docs(fake))


def test_every_entry_point_skips_a_locked_set_and_counts_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """One locked region set (an imported, validated box): the batch route
    (filter), the image route, the crop route, the service call and the CLI
    each leave it byte-identical and report ``locked_skipped == 1``."""
    reached: dict[str, int] = {}

    fake, before = _locked_world()
    client = _client(fake, monkeypatch)
    batch = client.post(
        f'{BASE}/reprocess',
        json={
            'targets': {'filter': {'region_status': [FAILED]}},
            'scopes': ['region'],
            'dry_run': False,
        },
    )
    reached['batch route'] = batch.json()['scopes'][0]['locked_skipped']
    assert docs(fake) == before

    image = client.post(f'{BASE}/images/img-1/reprocess', json={'scopes': ['region']})
    reached['image route'] = image.json()['scopes'][0]['locked_skipped']
    assert docs(fake) == before

    crop = client.post(f'{BASE}/crops/locked/reprocess', json={'scopes': ['region']})
    reached['crop route'] = crop.json()['scopes'][0]['locked_skipped']
    assert docs(fake) == before

    from src.services.curation.reprocess_models import ReprocessFilter

    service_call = asyncio.run(
        apply_reprocess(
            fake,
            ReprocessRequest(
                targets=ReprocessTargets(filter=ReprocessFilter(region_status=[FAILED])),
                scopes=['region'],
                dry_run=False,
            ),
        )
    )
    reached['service call'] = service_call.scopes[0].locked_skipped
    assert docs(fake) == before

    cli = _load_cli()
    monkeypatch.setattr(cli, 'make_script_opensearch', lambda *_a, **_kw: _Closable(fake))
    monkeypatch.setattr(sys, 'argv', ['requeue_regions.py', '--status', FAILED, '--apply'])
    assert cli.main() == 0
    out = capsys.readouterr().out
    reached['cli'] = 1 if '(locked, skipped)' in out else 0
    assert docs(fake) == before

    assert reached == {
        'batch route': 1,
        'image route': 1,
        'crop route': 1,
        'service call': 1,
        'cli': 1,
    }


@pytest.mark.asyncio
async def test_the_job_entry_point_never_writes_a_locked_item_either(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from curation.reprocess_fixtures import make_service as make

    root = servable_root(tmp_path, monkeypatch)
    path = root / 'a.jpg'
    path.write_bytes(jpeg_bytes())
    fake = make_fake([])
    triton = FakeTriton([(0.05, 0.05, 0.6, 0.6, 0.9, 1)])
    service = make(fake, triton)
    res = await service.ingest_one(path.read_bytes(), str(path))
    crop_id = next(iter(docs(fake)))
    docs(fake)[crop_id].update(
        {'class_source': 'human', 'label_source': 'human', 'class_id': 0, 'class_name': 'gadget',
         'class_validated': True}
    )  # fmt: skip
    before = copy.deepcopy(docs(fake)[crop_id])
    triton.detections = []  # nothing found: a stale-removal pass

    job = reprocess_job.create_job(request={}, scopes=['detect'], image_ids=[res.image_id])
    await reprocess_job.run_job(job, fake, service)

    state = reprocess_job.read_job(job.directory.name)
    assert state is not None
    assert state.status == 'completed'
    assert state.results[0].locked_skipped == 1
    after = docs(fake)[crop_id]
    assert {k for k in after if after[k] != before.get(k)} == set()
