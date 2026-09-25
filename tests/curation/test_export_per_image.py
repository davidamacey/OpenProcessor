"""The multi-class YOLO export writes ONE image + ONE label file per source
image, with one line per validated object on it.

The regression this pins: the exporter used to write one full-frame image
copy and one single-line label file per validated ITEM. A source image
with three validated objects came out as three copies of the same pixels,
each labeled with only one of the three objects, so the detector learned
the other two as background.

Also pinned here: the partial-frame policy (a frame whose other objects
are unreviewed is still exported by default, and the unlabeled objects on
it are counted; ``require_fully_labeled_images`` drops such frames), the
per-image dedup, and that ``GET /export/status`` counts agree with what is
on disk.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from src.clients.curation_opensearch import ClassRegistry
from src.config import CurationConfig
from src.services.curation.export import GenericYoloExportService, resolve_current_export_dir


CLASS_NAMES = ['unused', 'alpha', 'beta']
SPLITS = ('train', 'val', 'test')


# ------------------------------------------------------------------ fakes


def _matches(doc: dict[str, Any], clause: dict[str, Any]) -> bool:
    if 'term' in clause:
        ((field, value),) = clause['term'].items()
        return doc.get(field) == value
    if 'terms' in clause:
        ((field, values),) = clause['terms'].items()
        return doc.get(field) in values
    if 'exists' in clause:
        return doc.get(clause['exists']['field']) is not None
    msg = f'unsupported clause in fake: {clause}'
    raise AssertionError(msg)


class _FilteringOpenSearch:
    """Scroll + mget fake that honors the bool/term/terms/exists subset the
    exporter uses, so unvalidated / excluded / dismissed items are actually
    filtered rather than every query returning every doc."""

    def __init__(
        self, docs: list[dict[str, Any]], embeddings: dict[str, list[float]] | None = None
    ) -> None:
        self._docs = docs
        self._embeddings = embeddings or {}
        self.queries: list[dict[str, Any]] = []

    async def search(self, index: str, body: dict[str, Any], scroll: str | None = None) -> dict:  # noqa: ARG002
        query = body['query']
        self.queries.append(query)
        bool_q = query.get('bool') or {}
        hits = [
            {'_id': d['crop_id'], '_source': d}
            for d in self._docs
            if all(_matches(d, c) for c in bool_q.get('filter') or [])
            and not any(_matches(d, c) for c in bool_q.get('must_not') or [])
        ]
        return {'_scroll_id': 'scroll-1', 'hits': {'hits': hits}}

    async def scroll(self, scroll_id: str, scroll: str) -> dict:  # noqa: ARG002
        return {'_scroll_id': scroll_id, 'hits': {'hits': []}}

    async def clear_scroll(self, scroll_id: str) -> None:  # noqa: ARG002
        return None

    async def mget(self, index: str, body: dict[str, Any], _source: list[str]) -> dict:  # noqa: ARG002
        docs = [
            {'_id': i, '_source': {'image_id': i, _source[1]: self._embeddings[i]}}
            for i in body['ids']
            if i in self._embeddings
        ]
        return {'docs': docs}


def _registry(base_dir: Path) -> ClassRegistry:
    """Registry ids 0/1/2 with id 0 deprecated, so dense export ids (alpha=0,
    beta=1) differ from registry ids (alpha=1, beta=2)."""
    reg = ClassRegistry(path=base_dir / 'class_registry.json')
    for name in CLASS_NAMES:
        reg.add_class(name)
    loaded = reg.load()
    for c in loaded.classes:
        if c.class_id == 0:
            c.deprecated = True
    reg._atomic_write(loaded)
    return reg


def _service(
    tmp_path: Path,
    docs: list[dict[str, Any]],
    *,
    embeddings: dict[str, list[float]] | None = None,
) -> GenericYoloExportService:
    cfg = CurationConfig(export_root=tmp_path / 'exports', source_root=tmp_path / 'src')
    return GenericYoloExportService(
        _FilteringOpenSearch(docs, embeddings), config=cfg, registry=_registry(tmp_path / 'reg')
    )


def _doc(
    item_id: str,
    image_id: str,
    class_id: int,
    bbox: list[float] | None = None,
    *,
    validated: bool = True,
    holdout: bool = False,
    **extra: Any,
) -> dict[str, Any]:
    return {
        'crop_id': item_id,
        'image_id': image_id,
        'image_path': f'{image_id}.jpg',
        'bbox_norm': bbox or [0.1, 0.1, 0.5, 0.5],
        'class_id': class_id,
        'class_name': CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else str(class_id),
        'class_validated': validated,
        'test_holdout': holdout,
        **extra,
    }


def _write_jpeg(tmp_path: Path, image_id: str) -> None:
    src = tmp_path / 'src'
    src.mkdir(parents=True, exist_ok=True)
    Image.new('RGB', (40, 20), (10, 20, 30)).save(src / f'{image_id}.jpg', format='JPEG')


def _label_files(export_dir: Path) -> dict[str, Path]:
    return {
        f'{split}/{p.stem}': p
        for split in SPLITS
        for p in (export_dir / 'labels' / split).glob('*')
    }


def _image_files(export_dir: Path) -> dict[str, Path]:
    return {
        f'{split}/{p.stem}': p
        for split in SPLITS
        for p in (export_dir / 'images' / split).glob('*')
    }


def _parse(label: Path) -> list[tuple[int, float, float, float, float]]:
    out = []
    for line in label.read_text().splitlines():
        cid, *coords = line.split()
        cx, cy, w, h = (float(v) for v in coords)
        out.append((int(cid), cx, cy, w, h))
    return sorted(out)


def _manifest(result: Any) -> dict[str, Any]:
    return json.loads(Path(result.manifest_path).read_text())


# ------------------------------------------------------ one file per image


@pytest.mark.asyncio
async def test_image_with_three_objects_is_one_image_file_and_one_label_file(
    tmp_path: Path,
) -> None:
    docs = [
        _doc('obj-1', 'img-a', 1, [0.1, 0.2, 0.3, 0.6]),
        _doc('obj-2', 'img-a', 2, [0.5, 0.5, 0.9, 0.7]),
        _doc('obj-3', 'img-a', 1, [0.0, 0.0, 1.0, 1.0]),
    ]
    _write_jpeg(tmp_path, 'img-a')
    result = await _service(tmp_path, docs).export_dataset(seed=1)

    export_dir = Path(result.export_dir)
    labels = _label_files(export_dir)
    images = _image_files(export_dir)
    assert len(labels) == 1
    assert len(images) == 1
    ((key, label),) = labels.items()
    assert key.endswith('/img-a')
    assert set(images) == {key}  # image and label share the image-level stem

    lines = _parse(label)
    expected = sorted(
        [
            (0, 0.2, 0.4, 0.2, 0.4),  # alpha: registry id 1 -> dense id 0
            (1, 0.7, 0.6, 0.4, 0.2),  # beta: registry id 2 -> dense id 1
            (0, 0.5, 0.5, 1.0, 1.0),
        ]
    )
    assert len(lines) == 3
    for got, want in zip(lines, expected, strict=True):
        assert got[0] == want[0]
        assert got[1:] == pytest.approx(want[1:], abs=1e-6)

    assert result.image_count == 1
    assert result.object_count == 3
    manifest = _manifest(result)
    assert manifest['image_count'] == 1
    assert manifest['object_count'] == 3
    assert manifest['image_copy']['attempted'] == 1
    assert manifest['image_copy']['copied'] == 1
    stats = json.loads((export_dir / 'label_stats.json').read_text())
    assert stats == {'alpha': 2, 'beta': 1}


@pytest.mark.asyncio
async def test_two_images_with_one_object_each_are_two_files(tmp_path: Path) -> None:
    docs = [_doc('obj-1', 'img-a', 1), _doc('obj-2', 'img-b', 2)]
    for image_id in ('img-a', 'img-b'):
        _write_jpeg(tmp_path, image_id)
    result = await _service(tmp_path, docs).export_dataset(seed=1)

    export_dir = Path(result.export_dir)
    labels = _label_files(export_dir)
    assert sorted(k.split('/')[1] for k in labels) == ['img-a', 'img-b']
    assert set(_image_files(export_dir)) == set(labels)
    assert all(len(_parse(p)) == 1 for p in labels.values())
    assert (result.image_count, result.object_count) == (2, 2)


@pytest.mark.asyncio
async def test_holdout_object_forces_its_whole_image_to_test(tmp_path: Path) -> None:
    docs = [
        _doc('held', 'img-held', 1, holdout=True),
        _doc('mate-1', 'img-held', 2),
        _doc('mate-2', 'img-held', 1),
    ]
    docs += [_doc(f'o{i}', f'img-{i:02d}', 1 + i % 2) for i in range(20)]
    result = await _service(tmp_path, docs).export_dataset(seed=3, copy_images=False)

    labels = _label_files(Path(result.export_dir))
    held_keys = [k for k in labels if k.endswith('/img-held')]
    assert held_keys == ['test/img-held']
    assert len(_parse(labels['test/img-held'])) == 3

    manifest = _manifest(result)
    assert manifest['split_counts']['test'] >= 1
    assert manifest['split_object_counts']['test'] >= 3
    assert manifest['frozen_holdout_sha'] is not None
    # Every image is in exactly one split, and every object in its image's split.
    stems = [k.split('/')[1] for k in labels]
    assert len(stems) == len(set(stems)) == 21


# ------------------------------------------------------ partial frames


def _partial_frame_docs() -> list[dict[str, Any]]:
    return [
        # img-a: 2 validated + 1 unreviewed + 1 excluded + 1 dismissed
        _doc('a-1', 'img-a', 1),
        _doc('a-2', 'img-a', 2),
        _doc('a-unreviewed', 'img-a', 1, validated=False),
        _doc('a-excluded', 'img-a', 1, validated=False, class_excluded=True),
        _doc('a-dismissed', 'img-a', 2, review_dismissed_at='2026-01-01T00:00:00Z'),
        # img-b: fully labeled
        _doc('b-1', 'img-b', 1),
        # img-c: a validated object on a class the export doesn't include
        # (deprecated registry id 0) next to an exported one
        _doc('c-1', 'img-c', 2),
        _doc('c-deprecated', 'img-c', 0),
        # unreviewed item on an image that has no validated item at all:
        # that image is not exported, so it is not counted
        _doc('d-unreviewed', 'img-d', 1, validated=False),
    ]


@pytest.mark.asyncio
async def test_partial_frames_are_exported_and_their_unlabeled_objects_counted(
    tmp_path: Path,
) -> None:
    result = await _service(tmp_path, _partial_frame_docs()).export_dataset(
        seed=1, copy_images=False
    )

    labels = _label_files(Path(result.export_dir))
    by_stem = {k.split('/')[1]: p for k, p in labels.items()}
    assert sorted(by_stem) == ['img-a', 'img-b', 'img-c']
    assert len(_parse(by_stem['img-a'])) == 2
    assert len(_parse(by_stem['img-c'])) == 1

    manifest = _manifest(result)
    assert manifest['require_fully_labeled_images'] is False
    assert manifest['unlabeled_items_on_exported_images'] == 2  # a-unreviewed + c-deprecated
    assert manifest['images_with_unlabeled_items'] == 2
    assert manifest['images_dropped_not_fully_labeled'] == 0
    assert manifest['image_count'] == 3
    assert manifest['object_count'] == 4


@pytest.mark.asyncio
async def test_require_fully_labeled_images_drops_partial_frames(tmp_path: Path) -> None:
    result = await _service(tmp_path, _partial_frame_docs()).export_dataset(
        seed=1, copy_images=False, require_fully_labeled_images=True
    )

    labels = _label_files(Path(result.export_dir))
    assert [k.split('/')[1] for k in labels] == ['img-b']

    manifest = _manifest(result)
    assert manifest['require_fully_labeled_images'] is True
    assert manifest['images_dropped_not_fully_labeled'] == 2
    assert manifest['unlabeled_items_on_exported_images'] == 0
    assert manifest['images_with_unlabeled_items'] == 0
    assert (manifest['image_count'], manifest['object_count']) == (1, 1)
    assert result.images_dropped_not_fully_labeled == 2


@pytest.mark.asyncio
async def test_excluded_and_dismissed_items_are_not_unlabeled(tmp_path: Path) -> None:
    docs = [
        _doc('a-1', 'img-a', 1),
        _doc('a-excluded', 'img-a', 2, validated=False, class_excluded=True),
        _doc('a-dismissed', 'img-a', 2, validated=False, review_dismissed_at='2026-01-01'),
        _doc('a-dismissed-validated', 'img-a', 1, review_dismissed_at='2026-01-01'),
    ]
    default = await _service(tmp_path / 'default', docs).export_dataset(copy_images=False)
    strict = await _service(tmp_path / 'strict', docs).export_dataset(
        copy_images=False, require_fully_labeled_images=True
    )

    for result in (default, strict):
        manifest = _manifest(result)
        assert manifest['unlabeled_items_on_exported_images'] == 0
        assert manifest['images_with_unlabeled_items'] == 0
        assert manifest['images_dropped_not_fully_labeled'] == 0
        assert (manifest['image_count'], manifest['object_count']) == (1, 1)


@pytest.mark.asyncio
async def test_require_fully_labeled_images_with_no_full_frame_is_nothing_to_export(
    tmp_path: Path,
) -> None:
    from src.services.curation.export_readiness import NothingToExportError

    docs = [_doc('a-1', 'img-a', 1), _doc('a-2', 'img-a', 1, validated=False)]
    with pytest.raises(NothingToExportError, match='fully labeled'):
        await _service(tmp_path, docs).export_dataset(
            copy_images=False, require_fully_labeled_images=True
        )
    assert not (tmp_path / 'exports' / 'current').exists()


def test_unlabeled_objects_preflight_check_warns_with_the_counts() -> None:
    from src.services.curation.export_readiness import export_unlabeled_objects_check

    sev, msg, detail = export_unlabeled_objects_check(
        {
            'image_count': 3,
            'unlabeled_items_on_exported_images': 2,
            'images_with_unlabeled_items': 2,
            'require_fully_labeled_images': False,
        }
    )
    assert sev == 'warn'
    assert 'background' in msg
    assert '2' in msg
    assert detail == {
        'image_count': 3,
        'unlabeled_items_on_exported_images': 2,
        'images_with_unlabeled_items': 2,
        'require_fully_labeled_images': False,
        'images_dropped_not_fully_labeled': None,
    }

    ok = export_unlabeled_objects_check(
        {
            'image_count': 3,
            'unlabeled_items_on_exported_images': 0,
            'images_with_unlabeled_items': 0,
        }
    )
    assert ok[0] == 'ok'
    # An export made before these counts were recorded (per-item layout).
    assert export_unlabeled_objects_check({'image_count': 3})[0] == 'unknown'


# ------------------------------------------------------ dedup + cap


@pytest.mark.asyncio
async def test_dedup_collapses_near_duplicate_images_with_all_their_objects(
    tmp_path: Path,
) -> None:
    docs = [
        _doc('a-1', 'img-a', 1),
        _doc('a-2', 'img-a', 2),
        _doc('b-1', 'img-b', 1),
        _doc('b-2', 'img-b', 2),
        _doc('c-1', 'img-c', 1),
    ]
    embeddings = {
        'img-a': [1.0, 0.0, 0.0],
        'img-b': [0.999, 0.01, 0.0],  # near-duplicate of img-a
        'img-c': [0.0, 1.0, 0.0],
    }
    service = _service(tmp_path, docs, embeddings=embeddings)
    result = await service.export_dataset(seed=1, copy_images=False, dedup_threshold=0.98)

    labels = _label_files(Path(result.export_dir))
    stems = sorted(k.split('/')[1] for k in labels)
    assert len(stems) == 2
    assert 'img-c' in stems
    kept_dup = next(s for s in stems if s != 'img-c')
    assert len(_parse(next(p for k, p in labels.items() if k.endswith(kept_dup)))) == 2

    dedup = _manifest(result)['dedup']
    assert dedup['n_input_rows'] == 3  # images, not objects
    assert dedup['frames_dropped'] == 1
    assert dedup['n_output_rows'] == 2
    assert (result.image_count, result.object_count) == (2, 3)


@pytest.mark.asyncio
async def test_max_images_caps_images_and_keeps_every_object_on_them(tmp_path: Path) -> None:
    docs = [_doc(f'{i}-{k}', f'img-{i}', 1 + k % 2) for i in range(6) for k in range(3)]
    result = await _service(tmp_path, docs).export_dataset(seed=2, max_images=4, copy_images=False)
    labels = _label_files(Path(result.export_dir))
    assert len(labels) == 4
    assert all(len(_parse(p)) == 3 for p in labels.values())
    assert (result.image_count, result.object_count) == (4, 12)


# ------------------------------------------------------ counts + status


@pytest.fixture
def status_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    from src.routers.curation import _raw_opensearch_dep, router as curation_router

    monkeypatch.setattr('src.routers.curation._ensure_indexes', AsyncMock(return_value=None))
    app = FastAPI()
    app.include_router(curation_router)
    app.dependency_overrides[_raw_opensearch_dep] = AsyncMock
    return TestClient(app)


@pytest.mark.asyncio
async def test_status_counts_match_the_files_on_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status_client: TestClient
) -> None:
    docs = [
        _doc(f'{i}-{k}', f'img-{i:02d}', 1 + (i + k) % 2)
        for i in range(30)
        for k in range(i % 3 + 1)
    ]
    docs.append(_doc('held', 'img-00', 1, holdout=True))
    docs.append(_doc('pending', 'img-01', 2, validated=False))
    service = _service(tmp_path, docs)
    result = await service.export_dataset(seed=4, copy_images=False)
    monkeypatch.setattr(
        'src.routers.curation.export._resolve_current_export_dir',
        lambda: resolve_current_export_dir(service.config),
    )
    body = status_client.get('/curation/export/status').json()

    export_dir = Path(result.export_dir)
    image_files = {s: len(list((export_dir / 'labels' / s).glob('*.txt'))) for s in SPLITS}
    object_lines = {
        s: sum(len(_parse(p)) for p in (export_dir / 'labels' / s).glob('*.txt')) for s in SPLITS
    }
    validated = [d for d in docs if d['class_validated']]
    assert body['image_count'] == 30 == sum(image_files.values())
    assert body['object_count'] == len(validated) == sum(object_lines.values())
    assert body['split_counts'] == image_files
    assert body['split_object_counts'] == object_lines
    for split in SPLITS:
        assert sum(r[split] for r in body['class_split_counts']) == object_lines[split]
    assert body['unlabeled_items_on_exported_images'] == 1
    assert body['images_with_unlabeled_items'] == 1
    assert body['require_fully_labeled_images'] is False
    assert body['images_dropped_not_fully_labeled'] == 0

    manifest = _manifest(result)
    for key in (
        'image_count',
        'object_count',
        'split_counts',
        'split_object_counts',
        'class_split_counts',
        'unlabeled_items_on_exported_images',
        'images_with_unlabeled_items',
    ):
        assert body[key] == manifest[key], key
