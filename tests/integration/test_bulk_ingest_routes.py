"""``/ingest/bulk*`` through the real app and the real ingest service. The
OpenSearch, Triton and PE-encoder boundary is faked (``ingest_fakes``); the
routes, the runner, the ledger and the pipeline are production code."""

from __future__ import annotations

import io
import time
from typing import TYPE_CHECKING, Any

import pytest
from PIL import Image

from integration.ingest_fakes import (
    FakeOpenSearch,
    FakeTritonPool,
    curation_app,
    route_bulk_ingest_through_the_app,
)


if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from fastapi.testclient import TestClient

pytestmark = pytest.mark.integration

BASE = '/curation/projects/default/ingest/bulk'


def _write_images(root: Path, names: list[str], *, shade_base: int = 10) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(names):
        buf = io.BytesIO()
        Image.new('RGB', (64, 48), (shade_base + 9 * i, 30 + i, 30)).save(buf, format='JPEG')
        (root / name).write_bytes(buf.getvalue())


@pytest.fixture
def fake_os() -> FakeOpenSearch:
    return FakeOpenSearch()


@pytest.fixture
def served_root(tmp_path: Path) -> Path:
    root = tmp_path / 'served'
    root.mkdir()
    return root


@pytest.fixture
def client(
    fake_os: FakeOpenSearch, served_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    monkeypatch.setenv('OP_BULK_INGEST_DIR', str(tmp_path / 'bulk'))
    with curation_app(fake_os, FakeTritonPool(), monkeypatch) as c:
        route_bulk_ingest_through_the_app(monkeypatch)
        from src.services.curation import image_serving

        monkeypatch.setattr(
            image_serving,
            '_configured_roots',
            lambda config=None: (served_root.resolve(),),  # noqa: ARG005
        )
        yield c


def _wait(client: TestClient, run_id: str, *, timeout: float = 20.0) -> dict[str, Any]:
    deadline = time.time() + timeout
    run: dict[str, Any] = {}
    while time.time() < deadline:
        run = client.get(f'{BASE}/{run_id}').json()
        if run['status'] not in {'queued', 'running'}:
            return run
        time.sleep(0.05)
    msg = f'run stuck: {run}'
    raise AssertionError(msg)


def test_config_serves_the_roots_and_run_bounds(client: TestClient, served_root: Path) -> None:
    body = client.get(f'{BASE}/config').json()
    assert body['source_roots'] == [str(served_root.resolve())]
    assert body['default_extensions'] == ['.jpg', '.jpeg', '.png']
    assert 1 <= body['default_batch_size'] <= body['max_batch_size']
    assert 1 <= body['default_concurrency'] <= body['max_concurrency']


def test_folder_run_ingests_every_image_and_reports_the_run(
    client: TestClient, fake_os: FakeOpenSearch, served_root: Path
) -> None:
    _write_images(served_root / 'a', [f'{i}.jpg' for i in range(7)])
    _write_images(served_root / 'a' / 'sub', ['x.jpg'], shade_base=200)
    (served_root / 'a' / 'notes.txt').write_text('not an image')

    started = client.post(BASE, json={'folder': str(served_root / 'a'), 'source': 'census'})
    assert started.status_code == 202, started.text
    assert started.json()['resumed'] is False
    assert started.json()['poll_after_s'] is not None
    run = _wait(client, started.json()['run_id'])

    assert run['status'] == 'completed', run
    assert (run['can_cancel'], run['can_resume']) == (False, False)
    progress = run['progress']
    assert (progress['discovered'], progress['ingested'], progress['failed']) == (8, 8, 0)
    assert progress['discovery_complete'] is True
    assert progress['images_per_s'] > 0
    assert run['poll_after_s'] is None
    assert run['options']['source_tag'] == 'census'
    assert run['source'] == {
        'kind': 'folder',
        'path': str(served_root / 'a'),
        'extensions': ['.jpg', '.jpeg', '.png'],
        'label': None,
    }
    assert len(fake_os.images) == 8
    assert {d['source'] for d in fake_os.images.values()} == {'census'}
    assert {d['ingest_run_id'] for d in fake_os.images.values()} == {run['run_id']}


def test_running_the_same_folder_again_ingests_nothing_twice(
    client: TestClient, fake_os: FakeOpenSearch, served_root: Path
) -> None:
    _write_images(served_root / 'a', [f'{i}.jpg' for i in range(5)])
    first = _wait(
        client, client.post(BASE, json={'folder': str(served_root / 'a')}).json()['run_id']
    )
    docs_before = {k: dict(v) for k, v in fake_os.items.items()}

    second = _wait(
        client, client.post(BASE, json={'folder': str(served_root / 'a')}).json()['run_id']
    )

    assert second['run_id'] != first['run_id']
    assert second['progress']['ingested'] == 0
    assert second['progress']['duplicates'] == 5
    assert fake_os.items == docs_before
    assert len(fake_os.images) == 5


def test_a_manifest_run_lists_failures_with_their_reason(
    client: TestClient, served_root: Path, tmp_path: Path
) -> None:
    _write_images(served_root / 'm', ['one.jpg', 'two.jpg'])
    outside = tmp_path / 'outside.jpg'
    outside.write_bytes(b'x')
    manifest = served_root / 'list.txt'
    manifest.write_text(
        f'{served_root / "m" / "one.jpg"}\n{outside}\n{served_root / "m" / "two.jpg"}\n'
        f'{served_root / "m" / "gone.jpg"}\n'
    )

    run = _wait(client, client.post(BASE, json={'manifest': str(manifest)}).json()['run_id'])

    assert run['status'] == 'completed_with_errors'
    assert (run['can_cancel'], run['can_resume']) == (False, True)
    assert (run['progress']['ingested'], run['progress']['failed']) == (2, 2)
    page = client.get(f'{BASE}/{run["run_id"]}/failures').json()
    assert page['total'] == 2
    kinds = {row['path']: row['error_kind'] for row in page['items']}
    assert kinds == {
        str(outside): 'unservable_path',
        str(served_root / 'm' / 'gone.jpg'): 'decode_failed',
    }
    paged = client.get(f'{BASE}/{run["run_id"]}/failures', params={'offset': 1, 'limit': 1}).json()
    assert (paged['total'], len(paged['items'])) == (2, 1)


def test_the_run_list_is_newest_first(client: TestClient, served_root: Path) -> None:
    _write_images(served_root / 'a', ['1.jpg'])
    _write_images(served_root / 'b', ['2.jpg'], shade_base=100)
    one = _wait(client, client.post(BASE, json={'folder': str(served_root / 'a')}).json()['run_id'])
    time.sleep(1.05)  # ids carry a one-second timestamp
    two = _wait(client, client.post(BASE, json={'folder': str(served_root / 'b')}).json()['run_id'])
    listed = client.get(BASE).json()
    assert listed['total'] == 2
    assert [r['run_id'] for r in listed['items']] == [two['run_id'], one['run_id']]


@pytest.mark.parametrize(
    ('body', 'code'),
    [
        ({}, 422),
        ({'folder': '/x', 'manifest': '/y'}, 422),
        ({'folder': '/x', 'batch_size': 0}, 422),
        ({'folder': '/x', 'concurrency': 999}, 422),
        ({'folder': '/x', 'surprise': 1}, 422),
    ],
)
def test_malformed_requests_are_rejected(client: TestClient, body: dict, code: int) -> None:
    assert client.post(BASE, json=body).status_code == code


def test_a_source_outside_the_roots_or_missing_is_refused_with_a_code(
    client: TestClient, served_root: Path, tmp_path: Path
) -> None:
    for body in (
        {'folder': str(tmp_path)},
        {'folder': str(served_root / 'nope')},
        {'manifest': str(served_root)},
    ):
        resp = client.post(BASE, json=body)
        assert resp.status_code == 422, resp.text
        assert resp.json()['detail']['error'] == 'bulk_ingest_invalid_source'


def test_a_second_start_while_one_is_live_is_a_409_naming_the_run(
    client: TestClient, served_root: Path
) -> None:
    from src.services.curation.bulk_ingest import runner
    from src.services.curation.bulk_ingest.source import BulkSource

    _write_images(served_root / 'a', ['1.jpg'])
    live, _ = runner.claim_run(
        BulkSource('folder', str(served_root / 'a')), runner.BulkOptions('t')
    )

    resp = client.post(BASE, json={'folder': str(served_root / 'a')})

    assert resp.status_code == 409
    detail = resp.json()['detail']
    assert (detail['error'], detail['run_id']) == ('bulk_ingest_busy', live.run_id)


def test_cancel_marks_a_live_run_and_unknown_runs_are_404(
    client: TestClient, served_root: Path
) -> None:
    from src.services.curation.bulk_ingest import runner
    from src.services.curation.bulk_ingest.source import BulkSource

    _write_images(served_root / 'a', ['1.jpg'])
    live, _ = runner.claim_run(
        BulkSource('folder', str(served_root / 'a')), runner.BulkOptions('t')
    )
    cancelled = client.post(f'{BASE}/{live.run_id}/cancel').json()
    assert (cancelled['can_cancel'], cancelled['cancel_requested']) == (True, True)
    assert cancelled['can_resume'] is False
    assert live.job.cancel_requested()

    for url in (
        f'{BASE}/bulk_20260101T000000_deadbeef',
        f'{BASE}/not-a-run-id',
        f'{BASE}/bulk_20260101T000000_deadbeef/failures',
    ):
        resp = client.get(url)
        assert resp.status_code == 404, url
        assert resp.json()['detail']['error'] == 'bulk_ingest_not_found'
    assert client.post(f'{BASE}/bulk_20260101T000000_deadbeef/cancel').status_code == 404


def test_an_unknown_project_is_refused_before_anything_runs(
    client: TestClient, served_root: Path
) -> None:
    resp = client.post(
        '/curation/projects/nonexistent/ingest/bulk', json={'folder': str(served_root)}
    )
    assert resp.status_code == 404
    assert resp.json()['detail']['error'] == 'project_not_found'
