"""S9: bulk import of an existing YOLO-labeled dataset through the real app.

Runs ``scripts/curation/import_labeled_dataset.py`` against the real
``/curation/ingest/batch`` + label importer, with a *near-real-time* fake
OpenSearch (writes are not searchable until a refresh) so the test sees
the same visibility the live index has. The fixture dataset is shaped like
a whole-frame single-class export: class 0 only, positives plus
background images with empty label files.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
import pytest

from integration.ingest_fakes import FakeOpenSearch, FakeTritonPool, curation_app, jpeg_bytes


if TYPE_CHECKING:
    from fastapi.testclient import TestClient


pytestmark = pytest.mark.integration

# FakeTritonPool proposes one class-0 box per image at letterbox-normalized
# [0.1, 0.1, 0.5, 0.5]; for the 300x200 fixture frames this label overlaps it.
MATCHING_LABEL = '0 0.3 0.3 0.4 0.4\n'
FAR_LABEL = '0 0.9 0.9 0.1 0.1\n'


@pytest.fixture
def fake_opensearch() -> FakeOpenSearch:
    return FakeOpenSearch(near_real_time=True)


@pytest.fixture
def fake_triton() -> FakeTritonPool:
    return FakeTritonPool()


@pytest.fixture
def client(
    fake_opensearch: FakeOpenSearch,
    fake_triton: FakeTritonPool,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> TestClient:
    from src.clients.curation_opensearch import ClassRegistry
    from src.routers.curation import classes

    # One real single-class registry backs both GET /classes (the driver's
    # preflight) and the label importer.
    registry = ClassRegistry(tmp_path / 'registry' / 'class_registry.json')
    registry.add_class('widget')
    monkeypatch.setattr(classes, 'get_class_registry', lambda: registry)
    with curation_app(fake_opensearch, fake_triton, monkeypatch, registry=registry) as c:
        yield c


def _write(root: Path, split: str, name: str, label: str | None, seed: int) -> None:
    img = root / 'images' / split / f'{name}.jpg'
    img.parent.mkdir(parents=True, exist_ok=True)
    img.write_bytes(jpeg_bytes(seed))
    if label is not None:
        lbl = root / 'labels' / split / f'{name}.txt'
        lbl.parent.mkdir(parents=True, exist_ok=True)
        lbl.write_text(label)


def _dataset(root: Path, *, names: str = '{0: widget}') -> Path:
    seed = 500
    for i in range(3):
        _write(root, 'train', f'pos{i}', MATCHING_LABEL, seed := seed + 1)
    _write(root, 'train', 'far', FAR_LABEL, seed := seed + 1)
    _write(root, 'train', 'bg0', '', seed := seed + 1)
    _write(root, 'train', 'bg1', '', seed := seed + 1)
    _write(root, 'val', 'pos0', MATCHING_LABEL, seed := seed + 1)
    _write(root, 'val', 'bg_nolabel', None, seed := seed + 1)
    # Ultralytics' label caches sit next to the split dirs; must be ignored.
    (root / 'labels' / 'train.cache').write_bytes(b'\x00cache')
    data = root / 'data.yaml'
    # A copied dataset keeps its original, now-stale absolute ``path:``.
    data.write_text(
        f'path: /nonexistent/original/location\ntrain: images/train\nval: images/val\n'
        f'nc: 1\nnames: {names}\n'
    )
    return data


def _mod():
    from scripts.curation import import_labeled_dataset

    return import_labeled_dataset


def _run(client: TestClient, data: Path, state: Path, **cfg_overrides: Any) -> dict[str, Any]:
    mod = _mod()
    found, names = mod.discover(data)
    splits = {k: mod.load_samples(v) for k, v in found.items()}
    cfg = mod.ImportConfig(
        api_base='http://test/curation',
        state_dir=state,
        source_prefix='ds',
        batch_size=2,
        concurrency=2,
        retry_backoff_s=0.0,
        **cfg_overrides,
    )

    async def _go() -> dict[str, Any]:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=client.app)) as http:
            return await mod.run(cfg, http, splits, names)

    return asyncio.run(_go())


# =============================================================================
# Server-side: labels imported in the same call must see the fresh docs
# =============================================================================


def test_ingest_batch_labels_visible_despite_refresh_interval(
    client: TestClient, fake_opensearch: FakeOpenSearch, tmp_path: Path
) -> None:
    """Regression: ingest bulk-writes with refresh=False and the importer
    then *searches* for the image. Without a refresh in between, the label
    was silently dropped as "image not ingested"."""
    img = tmp_path / 'a.jpg'
    img.write_bytes(jpeg_bytes(1))
    lbl = tmp_path / 'a.txt'
    lbl.write_text(MATCHING_LABEL)
    resp = client.post(
        '/curation/ingest/batch',
        json={
            'items': [{'path': str(img), 'source': 'x', 'label_txt_path': str(lbl)}],
            'detect_mismatches': True,
        },
    )
    assert resp.status_code == 200, resp.text
    summary = resp.json()['summary']
    assert summary['labels_imported'] == 1
    assert summary['missed_labels'] == 0
    assert fake_opensearch.refresh_calls


def test_ingest_batch_returns_disagreement_records(client: TestClient, tmp_path: Path) -> None:
    far = tmp_path / 'far.jpg'
    far.write_bytes(jpeg_bytes(2))
    far_lbl = tmp_path / 'far.txt'
    far_lbl.write_text(FAR_LABEL)
    bg = tmp_path / 'bg.jpg'
    bg.write_bytes(jpeg_bytes(3))
    bg_lbl = tmp_path / 'bg.txt'
    bg_lbl.write_text('')
    resp = client.post(
        '/curation/ingest/batch',
        json={
            'items': [
                {'path': str(far), 'label_txt_path': str(far_lbl)},
                {'path': str(bg), 'label_txt_path': str(bg_lbl)},
            ],
            'detect_mismatches': True,
        },
    )
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert payload['summary']['missed_labels'] == 1
    assert payload['summary']['unmatched_detections'] == 2
    kinds = sorted((r['kind'], Path(r['image_path']).name) for r in payload['disagreements'])
    assert kinds == [
        ('missed_label', 'far.jpg'),
        ('unmatched_detection', 'bg.jpg'),
        ('unmatched_detection', 'far.jpg'),
    ]


# =============================================================================
# Driver
# =============================================================================


def test_single_class_dataset_with_backgrounds(
    client: TestClient,
    fake_opensearch: FakeOpenSearch,
    tmp_path: Path,
) -> None:
    data = _dataset(tmp_path / 'ds')
    state = tmp_path / 'state'
    summary = _run(client, data, state, label_source='dataset_v1')

    train, val, total = summary['splits']['train'], summary['splits']['val'], summary['total']
    assert (train['images'], train['positives'], train['backgrounds']) == (6, 4, 2)
    assert (val['images'], val['positives'], val['backgrounds']) == (2, 1, 1)
    assert val['label_files_missing'] == 1
    assert total['successful'] == 8
    assert total['failed'] == 0
    assert total['label_rows'] == 5
    assert total['labels_imported'] == 5
    assert total['mismatches'] == 0
    assert total['missed_labels'] == 1
    # far.jpg's own detection + one per background image (3 of them).
    assert total['unmatched_detections'] == 4
    assert total['backgrounds_with_detections'] == 3
    assert total['label_match_rate'] == pytest.approx(0.8)

    labels = list(fake_opensearch.labels.values())
    assert len(labels) == 5
    assert {doc['class_id'] for doc in labels} == {0}
    assert {doc['label_source'] for doc in labels} == {'dataset_v1'}
    sources = {d['hdd_source'] for d in fake_opensearch.images.values()}
    assert sources == {'ds:train', 'ds:val'}

    report = [json.loads(line) for line in (state / 'disagreements.jsonl').read_text().splitlines()]
    assert len(report) == 5
    bg_rows = [r for r in report if r['background']]
    assert {Path(r['local_image_path']).name for r in bg_rows} == {
        'bg0.jpg',
        'bg1.jpg',
        'bg_nolabel.jpg',
    }
    assert {r['split'] for r in report} == {'train', 'val'}
    assert (state / 'checkpoints' / 'train.json').exists()
    assert json.loads((state / 'summary.json').read_text())['total']['successful'] == 8


def test_rerun_skips_checkpointed_and_resumes_partial_split(
    client: TestClient, fake_triton: FakeTritonPool, tmp_path: Path
) -> None:
    data = _dataset(tmp_path / 'ds')
    state = tmp_path / 'state'
    first = _run(client, data, state)
    calls = len(fake_triton.batch_sizes)

    # Whole splits checkpointed: nothing is re-sent.
    again = _run(client, data, state)
    assert len(fake_triton.batch_sizes) == calls
    assert again['total'] == first['total']

    # A split interrupted after its batches were recorded: the progress
    # file alone restores both the done-set and the counts.
    (state / 'checkpoints' / 'train.json').unlink()
    resumed = _run(client, data, state)
    assert len(fake_triton.batch_sizes) == calls
    assert resumed['splits']['train'] == first['splits']['train']
    assert (state / 'checkpoints' / 'train.json').exists()


def test_class_registry_mismatch_aborts_before_ingest(
    client: TestClient, fake_opensearch: FakeOpenSearch, tmp_path: Path
) -> None:
    mod = _mod()
    data = _dataset(tmp_path / 'ds', names='{0: something_else}')
    with pytest.raises(mod.PreflightError, match='something_else'):
        _run(client, data, tmp_path / 'state')
    assert fake_opensearch.images == {}


def test_unreadable_labels_abort_after_first_batch(
    client: TestClient, fake_triton: FakeTritonPool, tmp_path: Path
) -> None:
    """If the server sees the images but not the label files, the run must
    stop instead of recording every image as a background."""
    mod = _mod()
    local = tmp_path / 'ds'
    data = _dataset(local)
    server_copy = tmp_path / 'server_view'
    for img in (local / 'images').rglob('*.jpg'):
        dest = server_copy / img.relative_to(local)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(img.read_bytes())
    with pytest.raises(mod.PreflightError, match='cannot read the label files'):
        _run(client, data, tmp_path / 'state', path_map=(str(local), str(server_copy)))
    # Sorted train order is bg0, bg1 | far, pos0 | pos1, pos2: the first
    # batch is all backgrounds (nothing to verify), the second carries
    # labels and trips the check — the third is never sent.
    assert len(fake_triton.batch_sizes) == 2


# =============================================================================
# Discovery
# =============================================================================


class TestDiscovery:
    def test_label_path_follows_yolo_rule(self) -> None:
        mod = _mod()
        assert mod.label_path_for(Path('/d/images/train/a.jpg')) == Path('/d/labels/train/a.txt')
        assert mod.label_path_for(Path('/d/train/images/x/a.png')) == Path(
            '/d/train/labels/x/a.txt'
        )
        assert mod.label_path_for(Path('/d/flat/a.jpg')) == Path('/d/flat/a.txt')

    def test_yaml_with_stale_path_resolves_relative_to_yaml(self, tmp_path: Path) -> None:
        mod = _mod()
        data = _dataset(tmp_path / 'ds')
        splits, names = mod.discover(data)
        assert names == ['widget']
        assert sorted(splits) == ['train', 'val']
        assert len(splits['train']) == 6
        assert mod.discover(data.parent)[0].keys() == splits.keys()

    def test_split_first_layout_without_yaml(self, tmp_path: Path) -> None:
        mod = _mod()
        for split in ('train', 'valid'):
            img = tmp_path / split / 'images' / 'a.jpg'
            img.parent.mkdir(parents=True)
            img.write_bytes(jpeg_bytes(9))
        splits, names = mod.discover(tmp_path)
        assert sorted(splits) == ['train', 'valid']
        assert names is None

    def test_missing_split_entry_is_an_error(self, tmp_path: Path) -> None:
        mod = _mod()
        (tmp_path / 'data.yaml').write_text('train: images/nope\nnames: [a]\n')
        with pytest.raises(mod.DatasetError, match='images/nope'):
            mod.discover(tmp_path)

    def test_stratified_sample_keeps_both_strata(self) -> None:
        mod = _mod()
        samples = [
            mod.Sample(
                image=Path(f'/p{i}.jpg'), label=Path(f'/p{i}.txt'), n_labels=1, label_exists=True
            )
            for i in range(80)
        ] + [
            mod.Sample(
                image=Path(f'/b{i}.jpg'), label=Path(f'/b{i}.txt'), n_labels=0, label_exists=True
            )
            for i in range(20)
        ]
        picked = mod.stratified_sample(samples, 10, seed=1)
        assert len(picked) == 10
        assert sum(1 for s in picked if s.positive) == 8
        assert picked == mod.stratified_sample(samples, 10, seed=1)


# =============================================================================
# --images-only (region ground truth: ingest images, never their labels)
# =============================================================================


def _ingested(state: Path, split: str) -> list[dict[str, Any]]:
    text = (state / 'ingested' / f'{split}.jsonl').read_text()
    return [json.loads(line) for line in text.splitlines()]


def test_images_only_ingests_without_labels_or_class_check(
    client: TestClient, fake_opensearch: FakeOpenSearch, tmp_path: Path
) -> None:
    # The labels are a region taxonomy the item registry has never heard of.
    data = _dataset(tmp_path / 'ds', names='{0: license_plate}')
    state = tmp_path / 'state'
    summary = _run(client, data, state, images_only=True)

    total = summary['total']
    assert (total['successful'], total['failed']) == (8, 0)
    assert (total['positives'], total['backgrounds']) == (5, 3)
    assert total['labels_imported'] == 0
    assert total['label_rows_on_ingested'] == 0
    assert total['label_match_rate'] is None
    assert fake_opensearch.labels == {}
    assert not (state / 'disagreements.jsonl').exists()

    train = _ingested(state, 'train')
    assert len(train) == 6
    assert {r['status'] for r in train} == {'success'}
    assert sum(r['positive'] for r in train) == 4
    image_ids = {d['image_id'] for d in fake_opensearch.images.values()}
    assert {r['image_id'] for r in train} <= image_ids
    assert all(r['server_path'] == r['image'] for r in train)
    val = _ingested(state, 'val')
    assert {Path(r['image']).name for r in val} == {'pos0.jpg', 'bg_nolabel.jpg'}


def test_images_only_cohort_survives_resume_and_records_duplicate_ids(
    client: TestClient, fake_opensearch: FakeOpenSearch, tmp_path: Path
) -> None:
    data = _dataset(tmp_path / 'ds')
    state = tmp_path / 'state'
    _run(client, data, state, images_only=True)
    first = _ingested(state, 'train')

    # Checkpointed split: the list is rebuilt from progress, not lost.
    (state / 'ingested' / 'train.jsonl').unlink()
    _run(client, data, state, images_only=True)
    assert _ingested(state, 'train') == first

    # A second import of the same bytes lands as duplicates; each keeps the
    # original's image_id, which is what the region evaluator joins on.
    asyncio.run(fake_opensearch.indices.refresh('all'))
    again = tmp_path / 'state2'
    summary = _run(client, data, again, images_only=True)
    assert summary['total']['duplicates'] == 8
    dup = _ingested(again, 'train')
    assert {r['status'] for r in dup} == {'duplicate'}
    by_name = {Path(r['image']).name: r['image_id'] for r in first}
    assert {Path(r['image']).name: r['image_id'] for r in dup} == by_name


def test_images_only_cli_guards(tmp_path: Path) -> None:
    mod = _mod()
    data = _dataset(tmp_path / 'ds', names='{0: license_plate}')
    base = ['--dataset', str(data), '--state-dir', str(tmp_path / 'state'), '--images-only']
    # Dry run skips the registry check entirely (no server is reachable here).
    assert mod.main([*base, '--dry-run', '--api-base', 'http://127.0.0.1:9']) == 0
    assert mod.main([*base, '--relabel-duplicates']) == 1
