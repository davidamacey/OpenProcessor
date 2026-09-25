"""S8: byte-upload ingest — ``POST /curation/ingest/upload`` and the
``scripts/curation/ingest_upload.py`` driver, run against the real app
with the OpenSearch/Triton/PE boundary faked."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
import pytest

from integration.ingest_fakes import (
    FakeOpenSearch,
    FakePEEncoder,
    FakeTritonPool,
    curation_app,
    jpeg_bytes,
)


if TYPE_CHECKING:
    from fastapi.testclient import TestClient


pytestmark = pytest.mark.integration


@pytest.fixture
def fake_opensearch() -> FakeOpenSearch:
    return FakeOpenSearch()


@pytest.fixture
def fake_triton() -> FakeTritonPool:
    return FakeTritonPool()


@pytest.fixture
def pe() -> FakePEEncoder:
    return FakePEEncoder()


@pytest.fixture
def client(
    fake_opensearch: FakeOpenSearch,
    fake_triton: FakeTritonPool,
    pe: FakePEEncoder,
    monkeypatch: pytest.MonkeyPatch,
) -> TestClient:
    with curation_app(fake_opensearch, fake_triton, monkeypatch, pe_encoder=pe) as c:
        yield c


def _files(*blobs: bytes) -> list[tuple[str, tuple[str, bytes, str]]]:
    return [('images', (f'img{i}.jpg', b, 'image/jpeg')) for i, b in enumerate(blobs)]


# =============================================================================
# Endpoint
# =============================================================================


def test_upload_ingests_bytes_under_client_identifiers(
    client: TestClient,
    fake_opensearch: FakeOpenSearch,
    fake_triton: FakeTritonPool,
    pe: FakePEEncoder,
) -> None:
    a, b = jpeg_bytes(11), jpeg_bytes(12)
    ids = ['remote://shoot1/a.jpg', 'remote://shoot1/b.jpg']
    resp = client.post(
        '/curation/ingest/upload',
        files=_files(a, b),
        data={'image_paths': json.dumps(ids), 'source': 'upload_test'},
    )
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert payload['status'] == 'success'
    assert payload['summary']['successful'] == 2
    # BA-1: image_path is now the server-persisted path, not the client
    # identifier -- the identifier is served/stored separately.
    assert [r['source_identifier'] for r in payload['results']] == ids
    assert all(r['image_path'] not in ids for r in payload['results'])

    docs = {d['source_identifier']: d for d in fake_opensearch.images.values()}
    assert set(docs) == set(ids)
    assert {d['source'] for d in docs.values()} == {'upload_test'}
    # The persisted image_path is real and distinct per image, under the
    # configured upload root, content-addressed.
    persisted_paths = {d['image_path'] for d in docs.values()}
    assert len(persisted_paths) == 2
    assert all('op_test_uploads' in p for p in persisted_paths)
    # The identifiers do not exist on the server: the whole-frame
    # embedding must come from the uploaded bytes, never a path re-read.
    assert pe.whole_frame_paths == []
    assert sorted(pe.whole_frame_bytes) == sorted([a, b])
    assert all(d.get('pe_embedding') is not None for d in docs.values())
    # One batched detector call for the whole upload.
    assert fake_triton.batch_sizes == [2]


def test_reupload_is_content_deduplicated(
    client: TestClient, fake_opensearch: FakeOpenSearch, fake_triton: FakeTritonPool
) -> None:
    blob = jpeg_bytes(21)
    first = client.post(
        '/curation/ingest/upload',
        files=_files(blob),
        data={'image_paths': json.dumps(['a/one.jpg'])},
    )
    assert first.json()['summary']['successful'] == 1
    # Same bytes under a different identifier: still a duplicate, no inference.
    second = client.post(
        '/curation/ingest/upload',
        files=_files(blob),
        data={'image_paths': json.dumps(['b/renamed.jpg'])},
    )
    assert second.status_code == 200, second.text
    assert second.json()['summary']['duplicates'] == 1
    assert second.json()['results'][0]['status'] == 'duplicate'
    assert len(fake_opensearch.images) == 1
    assert fake_triton.batch_sizes == [1]


def test_identifiers_default_to_filenames(
    client: TestClient, fake_opensearch: FakeOpenSearch
) -> None:
    resp = client.post('/curation/ingest/upload', files=_files(jpeg_bytes(31)))
    assert resp.status_code == 200, resp.text
    assert [d['source_identifier'] for d in fake_opensearch.images.values()] == ['img0.jpg']


def test_path_count_mismatch_rejected(client: TestClient) -> None:
    resp = client.post(
        '/curation/ingest/upload',
        files=_files(jpeg_bytes(41), jpeg_bytes(42)),
        data={'image_paths': json.dumps(['only_one.jpg'])},
    )
    assert resp.status_code == 422


def test_bad_paths_json_rejected(client: TestClient) -> None:
    resp = client.post(
        '/curation/ingest/upload', files=_files(jpeg_bytes(43)), data={'image_paths': '[oops'}
    )
    assert resp.status_code == 422


def test_oversized_upload_rejected(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    import src.config.curation as curation_config_mod

    monkeypatch.setenv('OP_UPLOAD_MAX_IMAGES_PER_REQUEST', '1')
    monkeypatch.setattr(curation_config_mod, '_default_curation_config', None)
    resp = client.post('/curation/ingest/upload', files=_files(jpeg_bytes(44), jpeg_bytes(45)))
    assert resp.status_code == 413


def test_undecodable_upload_fails_per_item(
    client: TestClient, fake_opensearch: FakeOpenSearch
) -> None:
    resp = client.post(
        '/curation/ingest/upload',
        files=_files(jpeg_bytes(46), b'not an image'),
        data={'image_paths': json.dumps(['good.jpg', 'bad.jpg'])},
    )
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert payload['status'] == 'partial'
    assert payload['summary']['successful'] == 1
    assert payload['summary']['failed'] == 1
    assert [d['source_identifier'] for d in fake_opensearch.images.values()] == ['good.jpg']


# =============================================================================
# Driver
# =============================================================================


def _tree(root: Path, n: int) -> list[Path]:
    paths = []
    for i in range(n):
        p = root / f'sub{i % 2}' / f'frame_{i}.jpg'
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(jpeg_bytes(100 + i))
        paths.append(p)
    (root / 'notes.txt').write_text('not an image')
    return sorted(paths)


def _run_driver(client: TestClient, paths: list[Path], **overrides: object):
    from scripts.curation.ingest_upload import UploadConfig, UploadRunner

    cfg = UploadConfig(
        api_base='http://test/curation',
        batch_size=2,
        reader_threads=3,
        submit_concurrency=2,
        queue_size=1,
        retry_backoff_s=0.0,
        **overrides,  # type: ignore[arg-type]
    )

    async def _go():
        transport = httpx.ASGITransport(app=client.app)
        async with httpx.AsyncClient(transport=transport) as http:
            return await UploadRunner(cfg, http).run(paths)

    return asyncio.run(_go())


def test_driver_uploads_tree_and_resumes(
    client: TestClient, fake_opensearch: FakeOpenSearch, tmp_path: Path
) -> None:
    from scripts.curation._fast_walk import iter_image_paths

    root = tmp_path / 'corpus'
    expected = _tree(root, 5)
    walked = sorted(iter_image_paths(root))
    assert walked == expected  # the .txt is not picked up

    progress = tmp_path / 'progress.json'
    counts = _run_driver(
        client,
        walked,
        path_map=(str(root), 'corpus://'),
        progress_file=progress,
    )
    assert counts.successful == 5
    assert counts.failed == 0
    stored = sorted(d['source_identifier'] for d in fake_opensearch.images.values())
    assert stored == sorted(f'corpus://{p.relative_to(root)}' for p in expected)
    snapshot = json.loads(progress.read_text())
    assert snapshot['final'] is True
    assert snapshot['counts']['successful'] == 5

    # Resume #1: identifiers already indexed are skipped before any read.
    again = _run_driver(client, walked, path_map=(str(root), 'corpus://'))
    assert again.skipped_known == 5
    assert again.uploaded == 0

    # Resume #2: with the pre-filter off, server content dedup catches all.
    forced = _run_driver(client, walked, path_map=(str(root), 'corpus://'), path_lookup=False)
    assert forced.uploaded == 5
    assert forced.duplicates == 5
    assert forced.successful == 0
    assert len(fake_opensearch.images) == 5


def test_driver_records_read_failures(client: TestClient, tmp_path: Path) -> None:
    root = tmp_path / 'corpus'
    paths = _tree(root, 2)
    missing = root / 'gone.jpg'
    failed_log = tmp_path / 'failed.jsonl'
    counts = _run_driver(client, [*paths, missing], failed_log=failed_log)
    assert counts.successful == 2
    assert counts.read_failed == 1
    rows = [json.loads(line) for line in failed_log.read_text().splitlines()]
    assert rows == [{'image_path': str(missing), 'error': rows[0]['error']}]


def test_driver_counts_server_rejections_as_failed(tmp_path: Path) -> None:
    """A 4xx is not retried and every image in the batch is counted failed."""
    from scripts.curation.ingest_upload import UploadConfig, UploadRunner

    calls = {'upload': 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith('/ingest/path_lookup'):
            return httpx.Response(200, json={'known_paths': {}})
        calls['upload'] += 1
        return httpx.Response(422, json={'detail': 'nope'})

    paths = _tree(tmp_path / 'c', 3)
    cfg = UploadConfig(api_base='http://x/curation', batch_size=2, retry_backoff_s=0.0)

    async def _go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            return await UploadRunner(cfg, http).run(paths)

    counts = asyncio.run(_go())
    assert counts.failed == 3
    assert calls['upload'] == 2  # two batches, no retries


def test_driver_retries_transient_5xx(tmp_path: Path) -> None:
    from scripts.curation.ingest_upload import UploadConfig, UploadRunner

    attempts = {'n': 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith('/ingest/path_lookup'):
            return httpx.Response(200, json={'known_paths': {}})
        attempts['n'] += 1
        if attempts['n'] == 1:
            return httpx.Response(503)
        return httpx.Response(
            200, json={'status': 'success', 'summary': {'successful': 1}, 'results': []}
        )

    paths = _tree(tmp_path / 'c', 1)
    cfg = UploadConfig(api_base='http://x/curation', retry_backoff_s=0.0)

    async def _go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            return await UploadRunner(cfg, http).run(paths)

    counts = asyncio.run(_go())
    assert attempts['n'] == 2
    assert counts.successful == 1
    assert counts.failed == 0


def test_path_map_rewrites_prefix_only() -> None:
    from scripts.curation.ingest_upload import map_identifier, parse_path_map

    pm = parse_path_map('/media/usb0/archive=archive://')
    assert map_identifier(Path('/media/usb0/archive/a/b.jpg'), pm) == 'archive://a/b.jpg'
    assert map_identifier(Path('/media/usb0/archive2/x.jpg'), pm) == '/media/usb0/archive2/x.jpg'
    pm2 = parse_path_map('/data=/mnt/shared')
    assert map_identifier(Path('/data/x.jpg'), pm2) == '/mnt/shared/x.jpg'


# =============================================================================
# BA-2 / BA-5: served limits enforced with typed errors
# =============================================================================


def test_unsupported_extension_fails_per_item_with_a_stable_code(
    client: TestClient, fake_opensearch: FakeOpenSearch
) -> None:
    resp = client.post(
        '/curation/ingest/upload',
        files=_files(jpeg_bytes(51)),
        data={'image_paths': json.dumps(['weird.gif'])},
    )
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert payload['summary']['failed'] == 1
    result = payload['results'][0]
    assert result['status'] == 'failed'
    assert result['error_kind'] == 'unsupported_type'
    assert result['source_identifier'] == 'weird.gif'
    assert len(fake_opensearch.images) == 0


def test_total_bytes_over_the_configured_limit_is_413(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.config.curation as curation_config_mod

    monkeypatch.setenv('OP_UPLOAD_MAX_BYTES_PER_REQUEST', '10')
    monkeypatch.setattr(curation_config_mod, '_default_curation_config', None)
    resp = client.post('/curation/ingest/upload', files=_files(jpeg_bytes(52)))
    assert resp.status_code == 413
