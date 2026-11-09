"""Reprocess entry points: the routes, the queued job's run, and the CLI. One test walks every entry point
with a locked input that must be skipped and counted."""

from __future__ import annotations

import asyncio
import copy
import importlib.util
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
    item,
    jpeg_bytes,
    make_fake,
    make_service,
    servable_root,
)
from src.services.curation.reprocess import apply_reprocess
from src.services.curation.reprocess_models import ReprocessRequest, ReprocessTargets


FAILED = RegionStatus.DETECTION_FAILED.value
LOCKED_SET = {'validated': True, 'verifier': 'human'}


def _factory(service: Any) -> Any:
    async def build() -> Any:
        return service

    return build


def _client(fake: Any, monkeypatch: pytest.MonkeyPatch, service: Any = None) -> TestClient:
    from _curation_app import mount_curation_routers

    from src.core.db_session import get_db
    from src.routers.curation import _raw_opensearch_dep, reprocess as route_mod, router

    app = FastAPI()
    mount_curation_routers(app, router)
    app.dependency_overrides[_raw_opensearch_dep] = lambda: fake

    async def no_database() -> Any:  # these requests stay under the sync limit: no job, no session
        yield None

    app.dependency_overrides[get_db] = no_database

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
