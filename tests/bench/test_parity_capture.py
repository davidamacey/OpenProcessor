"""Capture against a fake stack (httpx.MockTransport): routes, paging, vectors, the whole run."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import httpx
import numpy as np
import pytest

from scripts.bench import parity, parity_capture as cap, parity_store as ps
from scripts.datasets.bench_set import build_pin


if TYPE_CHECKING:
    from pathlib import Path

DIM = 4
OS_URL = 'http://os.test'
API = 'http://api.test'


class FakeStack:
    """Two items per image on ``n_images`` images, vectors derived from the item index."""

    def __init__(
        self, n_images: int = 5, policy: str = 'all', fail_image: str | None = None
    ) -> None:
        self.names = [f'{i:012d}.jpg' for i in range(n_images)]
        self.policy = policy
        self.fail_image = fail_image
        self.items: dict[str, dict[str, Any]] = {}
        self.vectors: dict[str, list[float]] = {}
        self.frames: dict[str, list[float]] = {}
        self.calls: list[tuple[str, str]] = []
        self.deleted = False
        self.polls = 0

    def _ingest(self, filename: str) -> dict[str, Any]:
        n = self.names.index(filename)
        image_id = f'img{n}'
        self.frames[image_id] = [float(n), 1.0, 0.0, 0.0]
        for k in range(2):
            item_id = f'{image_id}_{k}'
            self.items[item_id] = {
                'id': item_id,
                'image_id': image_id,
                'bbox_norm': [0.1 + 0.4 * k, 0.1, 0.4 + 0.4 * k, 0.6],
                'confidence': 0.5 + 0.1 * k,
                'detector_confidence': 0.5 + 0.1 * k,
                'detector_class_name': 'cat' if k == 0 else 'dog',
                'class_name': 'cat' if k == 0 else 'dog',
                'class_source': 'detector',
                'cluster_id': 10000 + k,
                'cluster_distance': 0.1,
                'cluster_similarity': 0.9,
                'cluster_is_core': True,
            }
            self.vectors[item_id] = [float(n), float(k), 1.0, 0.5]
        return {
            'status': 'success',
            'image_id': image_id,
            'image_path': f'/data/{n}',
            'source_identifier': filename,
        }

    def handler(self, request: httpx.Request) -> httpx.Response:  # noqa: PLR0911
        path = request.url.path
        self.calls.append((request.method, path))
        if request.method == 'DELETE':
            self.deleted = True
            return httpx.Response(200, json={})
        if path.endswith('/projects'):
            return httpx.Response(200, json={})
        if path.endswith('/ingest/policy'):
            return httpx.Response(200, json={'embedding': {'mode': self.policy}})
        if path.endswith('/ingest/upload'):
            names = request.content.decode('latin-1').split('filename="')[1:]
            results = []
            for chunk in names:
                filename = chunk.split('"', 1)[0]
                if filename == self.fail_image:
                    results.append({'status': 'failed', 'image_path': filename, 'error': 'decode'})
                else:
                    results.append(self._ingest(filename))
            return httpx.Response(200, json={'results': results})
        if path.endswith('/pipeline/auto_label/start'):
            return httpx.Response(200, json={'job_id': 'j1'})
        if '/pipeline/auto_label/status/' in path:
            return httpx.Response(
                200, json={'status': 'completed', 'stage_durations': {'train': 1.0}}
            )
        if path.endswith('/crops'):
            return self._crops(request)
        if path.endswith('/_mget'):
            ids = json.loads(request.content)['ids']
            pool = self.vectors if '__items' in path else self.frames
            docs = [
                {
                    '_id': i,
                    'found': i in pool,
                    '_source': {'pe_embedding': pool[i]} if i in pool else {},
                }
                for i in ids
            ]
            return httpx.Response(200, json={'docs': docs})
        if path.endswith('/embed/image'):
            return httpx.Response(200, json={'embedding': [0.0, 1.0]})
        if path.endswith('/faces/recognize'):
            return httpx.Response(
                200,
                json={
                    'faces': [{'box': [0.1, 0.1, 0.2, 0.3], 'score': 0.9}],
                    'embeddings': [[0.5, 0.5]],
                },
            )
        if path.endswith('/ocr/predict'):
            return httpx.Response(422, json={'detail': 'no text'})
        if path.endswith(('/classes', '/vlm/label_batch')):
            return httpx.Response(200, json={})
        # GET project: index names
        return httpx.Response(
            200,
            json={'resources': {'indexes': {'items': 'op_p__items', 'images': 'op_p__images'}}},
        )

    def _crops(self, request: httpx.Request) -> httpx.Response:
        q = request.url.params
        rows = list(self.items.values())
        if q.get('ids'):
            wanted = q['ids'].split(',')
            rows = [dict(r, vlm_proposed_class_name='cat') for r in rows if r['id'] in wanted]
            return httpx.Response(200, json={'total': len(rows), 'crops': rows})
        if 'conf_min' in q:
            low, high = float(q['conf_min']), float(q['conf_max'])
            rows = [r for r in rows if low <= r['confidence'] <= high]
        size, page = int(q.get('page_size', 50)), int(q.get('page', 1))
        return httpx.Response(
            200, json={'total': len(rows), 'crops': rows[(page - 1) * size : page * size]}
        )


def _client(stack: FakeStack) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(stack.handler))


def test_fetch_items_pages_through_a_single_band() -> None:
    stack = FakeStack()
    for n in range(5):
        stack._ingest(stack.names[n])
    with _client(stack) as client:
        items = cap.fetch_items(client, f'{API}/p', page_size=3)
    assert len(items) == 10


def test_fetch_items_splits_bands_when_the_window_is_exceeded() -> None:
    stack = FakeStack()
    for n in range(5):
        stack._ingest(stack.names[n])
    with _client(stack) as client:
        items = cap.fetch_items(client, f'{API}/p', window=6, page_size=4)
    assert sorted(i['id'] for i in items) == sorted(stack.items)


def test_fetch_items_aborts_when_the_union_misses_rows() -> None:
    stack = FakeStack()
    for n in range(5):
        stack._ingest(stack.names[n])
    stack.items['ghost'] = {**stack.items['img0_0'], 'id': 'ghost', 'confidence': 1.7}
    with _client(stack) as client, pytest.raises(ps.CaptureError, match='reports'):
        cap.fetch_items(client, f'{API}/p')


def test_mget_vectors_skips_documents_without_one() -> None:
    stack = FakeStack()
    stack._ingest(stack.names[0])
    with _client(stack) as client:
        got = cap.mget_vectors(client, OS_URL, 'op_p__items', ['img0_0', 'missing'])
    assert list(got) == ['img0_0']


def test_ingest_images_maps_ids_to_file_names(tmp_path: Path) -> None:
    stack = FakeStack(n_images=3)
    files = []
    for name in stack.names:
        (tmp_path / name).write_bytes(b'x')
        files.append(tmp_path / name)
    with _client(stack) as client:
        names = cap.ingest_images(client, f'{API}/p', files, batch_size=2, concurrency=2)
    assert names == {f'img{i}': stack.names[i] for i in range(3)}


def test_ingest_images_aborts_on_a_failed_image(tmp_path: Path) -> None:
    stack = FakeStack(n_images=2, fail_image='000000000001.jpg')
    files = []
    for name in stack.names:
        (tmp_path / name).write_bytes(b'x')
        files.append(tmp_path / name)
    with _client(stack) as client, pytest.raises(ps.CaptureError, match='failed to ingest'):
        cap.ingest_images(client, f'{API}/p', files, batch_size=8, concurrency=1)


def test_policy_other_than_all_is_refused() -> None:
    with _client(FakeStack(policy='lazy')) as client, pytest.raises(ps.CaptureError, match='all'):
        cap.require_embed_all(client, f'{API}/p')


def test_probe_core_treats_422_as_no_rows(tmp_path: Path) -> None:
    path = tmp_path / 'a.jpg'
    path.write_bytes(b'x')
    with _client(FakeStack()) as client:
        out = cap.probe_core(client, API, [path], concurrency=1)
    assert out['clip'] == [('a.jpg', [0.0, 1.0])]
    assert out['faces'] == [('a.jpg', [0.1, 0.1, 0.2, 0.3], 0.9, [0.5, 0.5])]
    assert out['ocr'] == []


def test_vlm_sample_depends_only_on_file_names() -> None:
    items = [{'id': f'{n}_{k}', 'image_id': f'i{n}'} for n in range(20) for k in range(2)]
    a = cap.vlm_sample(items, {f'i{n}': f'{n}.jpg' for n in range(20)}, 10)
    shuffled = list(reversed(items))
    b = cap.vlm_sample(shuffled, {f'i{n}': f'{n}.jpg' for n in range(20)}, 10)
    assert a == b
    assert len(a) == 10


def _write_pin(tmp_path: Path, stack: FakeStack) -> tuple[Path, Path]:
    images = tmp_path / 'images'
    images.mkdir()
    rows = []
    for i, name in enumerate(stack.names):
        (images / name).write_bytes(b'jpeg')
        rows.append(
            {
                'image_id': i,
                'file_name': name,
                'split': 'val2017',
                'license_name': 'x',
                'flickr_url': '',
                'width': 100,
                'height': 80,
                'bytes': 4,
                'sha256': 'x',
            }
        )
    manifest = tmp_path / 'pin.json'
    manifest.write_text(
        json.dumps(build_pin(rows, seed=1, licenses=['x'], annotations_sha256='abc'))
    )
    return manifest, images


def _args(manifest: Path, images: Path, out: Path, **extra: Any) -> Any:
    ns = parity.build_parser().parse_args(
        [
            'capture',
            '--tag',
            'unit',
            '--manifest',
            str(manifest),
            '--images',
            str(images),
            '--api-url',
            API,
            '--triton-url',
            'http://triton.test',
            '--opensearch-url',
            OS_URL,
            '--compose-project',
            'w0par',
            '--engine-label',
            'fp16',
            '--engine-record',
            'precision=fp16',
            '--out',
            str(out),
            '--probe-images',
            '2',
        ]
    )
    for key, value in extra.items():
        setattr(ns, key, value)
    return ns


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr('scripts.bench.suite_ingest.time.sleep', lambda _s: None)
    monkeypatch.setattr(
        parity, 'environment_record', lambda *_a, **_k: {'git_sha': 'abc', 'hardware': {}}
    )


def test_run_capture_end_to_end(tmp_path: Path, no_sleep: None) -> None:
    stack = FakeStack()
    manifest, images = _write_pin(tmp_path, stack)
    capture = parity.run_capture(
        _args(manifest, images, tmp_path / 'out'), transport=httpx.MockTransport(stack.handler)
    )
    a = capture.arrays
    assert len(a['item_id']) == 10
    assert a['item_has_vec'].all()
    assert a['item_vec'].shape == (10, DIM)
    assert a['item_vec'].dtype == np.float16
    assert a['frame_vec'].shape == (5, DIM)
    assert a['item_image'][0] == '000000000000.jpg'
    assert a['item_crop_rect'][0].tolist() == [10, 8, 40, 48]
    assert capture.meta['sections'] == ['items', 'frames', 'clusters', 'clip', 'faces', 'ocr']
    assert capture.meta['manifest']['count'] == 5
    assert capture.meta['engine'] == {'label': 'fp16', 'precision': 'fp16'}
    assert stack.deleted is True
    ps.validate(a)


def test_run_capture_keeps_the_project_on_request(tmp_path: Path, no_sleep: None) -> None:
    stack = FakeStack()
    manifest, images = _write_pin(tmp_path, stack)
    args = _args(manifest, images, tmp_path / 'out', keep_project=True)
    parity.run_capture(args, transport=httpx.MockTransport(stack.handler))
    assert stack.deleted is False


def test_run_capture_removes_the_project_when_ingest_fails(tmp_path: Path, no_sleep: None) -> None:
    stack = FakeStack(fail_image='000000000002.jpg')
    manifest, images = _write_pin(tmp_path, stack)
    with pytest.raises(ps.CaptureError):
        parity.run_capture(
            _args(manifest, images, tmp_path / 'out'), transport=httpx.MockTransport(stack.handler)
        )
    assert stack.deleted is True


def test_run_capture_refuses_missing_images(tmp_path: Path, no_sleep: None) -> None:
    stack = FakeStack()
    manifest, images = _write_pin(tmp_path, stack)
    (images / stack.names[0]).unlink()
    with pytest.raises(ps.CaptureError, match='missing'):
        parity.run_capture(
            _args(manifest, images, tmp_path / 'out'), transport=httpx.MockTransport(stack.handler)
        )


def test_urls_use_the_configured_api_prefix(tmp_path: Path, no_sleep: None) -> None:
    from scripts.curation._project_worker_utils import curation_api_prefix

    stack = FakeStack()
    manifest, images = _write_pin(tmp_path, stack)
    parity.run_capture(
        _args(manifest, images, tmp_path / 'out'), transport=httpx.MockTransport(stack.handler)
    )
    prefix = curation_api_prefix()
    curation_calls = [p for _, p in stack.calls if '/projects' in p]
    assert curation_calls
    assert all(p.startswith(prefix + '/projects') for p in curation_calls)
