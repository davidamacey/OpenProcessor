"""Unit tests for :mod:`src.services.curation.export`.

Fake OpenSearch (scroll + clear_scroll) and a tmp_path export root — no
real cluster. Image copy is exercised against real tiny on-disk JPEGs
under a tmp ``source_root`` so the resize stage runs for real; most tests
disable it (``copy_images=False``) when only the label/manifest side is
under test.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pytest

from src.clients.curation_opensearch import ClassRegistry
from src.config import CurationConfig
from src.services.curation.export import (
    GenericYoloExportService,
    _ExportRow,
    even_stratified_sample,
    frozen_test_sha_of,
    hash_split,
    label_content_sha,
    resolve_current_export_dir,
    stratified_split,
)


class _FakeOpenSearch:
    def __init__(self, docs: list[dict[str, Any]]) -> None:
        self._docs = docs

    async def search(self, index: str, body: dict[str, Any], scroll: str | None = None) -> dict:  # noqa: ARG002
        hits = [{'_id': d['crop_id'], '_source': d} for d in self._docs]
        return {'_scroll_id': 'scroll-1', 'hits': {'hits': hits}}

    async def scroll(self, scroll_id: str, scroll: str) -> dict:  # noqa: ARG002
        return {'_scroll_id': scroll_id, 'hits': {'hits': []}}

    async def clear_scroll(self, scroll_id: str) -> None:  # noqa: ARG002
        return None


def _make_registry(base_dir: Path, names: list[str]) -> ClassRegistry:
    reg = ClassRegistry(path=base_dir / 'class_registry.json')
    for name in names:
        reg.add_class(name)
    return reg


def _service(
    tmp_path: Path, docs: list[dict[str, Any]], names: list[str]
) -> GenericYoloExportService:
    cfg = CurationConfig(export_root=tmp_path / 'exports')
    registry = _make_registry(tmp_path / 'registry', names)
    return GenericYoloExportService(_FakeOpenSearch(docs), config=cfg, registry=registry)


def test_hash_split_is_deterministic():
    a = hash_split('crop-1', seed=42, train_ratio=0.8, val_ratio=0.1)
    b = hash_split('crop-1', seed=42, train_ratio=0.8, val_ratio=0.1)
    assert a == b
    assert a in {'train', 'val', 'test'}


def test_hash_split_changes_with_seed_sometimes():
    # Not a strict guarantee for every key, but across many keys at least
    # one seed change should move at least one key to a different bucket.
    keys = [f'crop-{i}' for i in range(200)]
    a = {k: hash_split(k, seed=1, train_ratio=0.8, val_ratio=0.1) for k in keys}
    b = {k: hash_split(k, seed=2, train_ratio=0.8, val_ratio=0.1) for k in keys}
    assert a != b


def test_label_content_sha_tracks_content_and_class_names(tmp_path: Path):
    """Base sanity check on the shared primitive before exercising it
    through the full exporter below: same label bytes -> same digest,
    different ``class_names`` -> different digest."""
    (tmp_path / 'labels' / 'train').mkdir(parents=True)
    label = tmp_path / 'labels' / 'train' / 'a.txt'
    label.write_text('0 0.5 0.5 0.1 0.1\n')

    assert label_content_sha(tmp_path) == label_content_sha(tmp_path)
    assert label_content_sha(tmp_path, ['car']) != label_content_sha(tmp_path, ['truck'])
    assert label_content_sha(tmp_path) != label_content_sha(tmp_path, ['car'])
    # truncate=None yields the full 64-char sha256 hex digest the
    # multi-class exporter records (a 64-hex dataset_sha is what every
    # existing consumer assumes).
    assert len(label_content_sha(tmp_path, truncate=None)) == 64
    assert len(label_content_sha(tmp_path)) == 16


def test_resolve_current_export_dir_missing_raises(tmp_path):
    cfg = CurationConfig(export_root=tmp_path / 'exports')
    with pytest.raises(FileNotFoundError):
        resolve_current_export_dir(cfg)


@pytest.mark.asyncio
async def test_export_dataset_writes_manifest_and_labels(tmp_path):
    docs = [
        {
            'crop_id': 'crop-1',
            'image_id': 'img-1',
            'image_path': 'a.jpg',
            'bbox_norm': [0.1, 0.1, 0.5, 0.5],
            'class_id': 0,
            'class_name': 'car',
            'test_holdout': True,
        },
        {
            'crop_id': 'crop-2',
            'image_id': 'img-2',
            'image_path': 'b.jpg',
            'bbox_norm': [0.2, 0.2, 0.6, 0.6],
            'class_id': 1,
            'class_name': 'truck',
            'test_holdout': False,
        },
    ]
    service = _service(tmp_path, docs, ['car', 'truck'])

    result = await service.export_dataset(version_tag='t1', seed=7, copy_images=False)

    assert result.image_count == 2
    assert result.class_count == 2
    assert result.classes_with_objects == 2
    assert result.split_counts.test == 1  # crop-1's test_holdout=True is honored

    current_link = tmp_path / 'exports' / 'current'
    assert current_link.resolve() == Path(result.export_dir).resolve()

    manifest = json.loads(Path(result.manifest_path).read_text())
    assert manifest['dataset_sha'] == result.dataset_sha
    assert manifest['image_count'] == 2
    assert manifest['classes_with_objects'] == 2
    assert manifest['frozen_holdout_sha'] is not None
    assert manifest['code_sha']


@pytest.mark.asyncio
async def test_classes_with_objects_excludes_registry_classes_with_no_labeled_crops(tmp_path):
    """E2: class_count is the registry size (data.yaml's nc); a class
    with zero labeled objects in this export must not count as one that
    "has data" -- classes_with_objects is the honest denominator."""
    docs = [
        {
            'crop_id': 'crop-1',
            'image_id': 'img-1',
            'image_path': 'a.jpg',
            'bbox_norm': [0.1, 0.1, 0.5, 0.5],
            'class_id': 0,
            'class_name': 'car',
        },
    ]
    # 'truck' and 'bus' are registered classes with no crops at all.
    service = _service(tmp_path, docs, ['car', 'truck', 'bus'])
    result = await service.export_dataset(version_tag='t1', seed=7, copy_images=False)

    assert result.class_count == 3
    assert result.classes_with_objects == 1
    manifest = json.loads(Path(result.manifest_path).read_text())
    assert manifest['class_count'] == 3
    assert manifest['classes_with_objects'] == 1


@pytest.mark.asyncio
async def test_export_dataset_respects_max_images(tmp_path):
    docs = [
        {
            'crop_id': f'crop-{i}',
            'image_id': f'img-{i}',
            'image_path': f'{i}.jpg',
            'bbox_norm': [0.0, 0.0, 1.0, 1.0],
            'class_id': 0,
            'class_name': 'car',
        }
        for i in range(5)
    ]
    service = _service(tmp_path, docs, ['car'])

    result = await service.export_dataset(max_images=2, copy_images=False)

    assert result.image_count == 2


# ---------------------------------------------------------------------------
# Class-balanced sampling under a max_images cap (plan §4.2 G10)
# ---------------------------------------------------------------------------


def test_even_stratified_sample_keeps_every_stratum():
    """A cap >= the stratum count must leave no stratum empty, even when
    one stratum dominates the input and sorts first."""
    rows = [('a', i) for i in range(100)] + [('b', 0), ('b', 1)] + [('c', 0)]
    picked = even_stratified_sample(rows, 9, lambda r: r[0], random.Random(1))

    assert len(picked) == 9
    assert {r[0] for r in picked} == {'a', 'b', 'c'}
    # Even, not proportional: 'b' and 'c' are exhausted rather than sampled
    # in proportion to their 2/103 and 1/103 share.
    assert sum(1 for r in picked if r[0] == 'b') == 2
    assert sum(1 for r in picked if r[0] == 'c') == 1


def test_even_stratified_sample_passthrough_and_edges():
    rows = list(range(5))
    assert even_stratified_sample(rows, None, str, random.Random(0)) == rows
    assert even_stratified_sample(rows, 10, str, random.Random(0)) == rows
    assert even_stratified_sample(rows, 0, str, random.Random(0)) == []


def test_even_stratified_sample_is_seed_reproducible():
    rows = [(f'cls-{i % 4}', i) for i in range(40)]
    a = even_stratified_sample(rows, 11, lambda r: r[0], random.Random(7))
    b = even_stratified_sample(rows, 11, lambda r: r[0], random.Random(7))
    assert a == b


def _imbalanced_docs() -> list[dict[str, Any]]:
    """40 cars, 3 trucks, 1 bus -- cars sort first, as a scroll would return
    them, so plain ``hits[:n]`` truncation drops truck and bus entirely."""
    spec = [('car', 0, 40), ('truck', 1, 3), ('bus', 2, 1)]
    return [
        {
            'crop_id': f'{name}-{i}',
            'image_id': f'{name}-img-{i}',
            'image_path': f'{name}-{i}.jpg',
            'bbox_norm': [0.0, 0.0, 1.0, 1.0],
            'class_id': class_id,
            'class_name': name,
        }
        for name, class_id, count in spec
        for i in range(count)
    ]


@pytest.mark.asyncio
async def test_max_images_cap_keeps_rare_classes_and_hits_the_cap(tmp_path):
    """G10: the cap is an even per-class sample applied after dedup, not a
    truncation of the raw hit list."""
    service = _service(tmp_path, _imbalanced_docs(), ['car', 'truck', 'bus'])

    result = await service.export_dataset(seed=42, max_images=12, copy_images=False)

    assert result.image_count == 12  # exactly the cap, not some count below it

    label_stats = json.loads((Path(result.export_dir) / 'label_stats.json').read_text())
    # (a) no class silently vanishes...
    assert label_stats['car'] > 0
    assert label_stats['truck'] == 3  # small class taken whole
    assert label_stats['bus'] == 1
    # ...and (b) the dominant class is the one trimmed to fit the budget.
    assert label_stats['car'] == 8
    assert sum(label_stats.values()) == 12

    manifest = json.loads(Path(result.manifest_path).read_text())
    assert manifest['sampling_mode'] == 'stratified_even'
    assert manifest['max_images'] == 12
    assert manifest['image_count'] == 12


@pytest.mark.asyncio
async def test_uncapped_export_records_sampling_mode_all(tmp_path):
    service = _service(tmp_path, _imbalanced_docs(), ['car', 'truck', 'bus'])

    result = await service.export_dataset(seed=42, copy_images=False)

    manifest = json.loads(Path(result.manifest_path).read_text())
    assert manifest['sampling_mode'] == 'all'
    assert manifest['max_images'] is None
    assert result.image_count == 44


@pytest.mark.asyncio
async def test_capped_export_is_reproducible_from_the_recorded_seed(tmp_path):
    """The sample is RNG-driven, so pin that the recorded seed reproduces it."""
    docs = _imbalanced_docs()
    service_a = _service(tmp_path / 'a', docs, ['car', 'truck', 'bus'])
    service_b = _service(tmp_path / 'b', docs, ['car', 'truck', 'bus'])

    result_a = await service_a.export_dataset(seed=5, max_images=12, copy_images=False)
    result_b = await service_b.export_dataset(seed=5, max_images=12, copy_images=False)

    assert result_a.dataset_sha == result_b.dataset_sha
    assert result_a.split_counts.to_dict() == result_b.split_counts.to_dict()


# ---------------------------------------------------------------------------
# W3.a restored cases (plan §4 Wave 3)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_data_yaml_nc_matches_class_count(tmp_path):
    docs = [
        {
            'crop_id': 'crop-1',
            'image_id': 'img-1',
            'image_path': 'a.jpg',
            'bbox_norm': [0.1, 0.1, 0.5, 0.5],
            'class_id': 0,
            'class_name': 'car',
        },
    ]
    service = _service(tmp_path, docs, ['car', 'truck', 'bus'])

    result = await service.export_dataset(copy_images=False)

    data_yaml = (Path(result.export_dir) / 'data.yaml').read_text()
    assert 'nc: 3' in data_yaml
    assert result.class_count == 3


@pytest.mark.asyncio
async def test_frozen_holdout_rows_land_in_test_split(tmp_path):
    """``export.py``'s own module docstring advertises this; pin it."""
    docs = [
        {
            'crop_id': f'crop-{i}',
            'image_id': f'img-{i}',
            'image_path': f'{i}.jpg',
            'bbox_norm': [0.0, 0.0, 1.0, 1.0],
            'class_id': 0,
            'class_name': 'car',
            # Every row hashes wherever it hashes EXCEPT this one, which is
            # frozen -- it must land in test regardless of the hash bucket.
            'test_holdout': i == 0,
        }
        for i in range(20)
    ]
    service = _service(tmp_path, docs, ['car'])

    result = await service.export_dataset(seed=42, copy_images=False)

    # Files are per source image: crop-0 sits on img-0.
    label_path = Path(result.export_dir) / 'labels' / 'test' / 'img-0.txt'
    assert label_path.is_file()
    assert not list((Path(result.export_dir) / 'labels').glob('*/crop-*.txt'))


@pytest.mark.asyncio
async def test_stratification_distributes_classes_across_splits(tmp_path):
    """Each class's actual per-split ratio should track the target ratio
    closely -- not just "some" items in each split by luck."""
    docs = [
        {
            'crop_id': f'{name}-{i}',
            'image_id': f'{name}-img-{i}',
            'image_path': f'{name}-{i}.jpg',
            'bbox_norm': [0.0, 0.0, 1.0, 1.0],
            'class_id': cls_idx,
            'class_name': name,
        }
        for cls_idx, name in enumerate(['car', 'truck'])
        for i in range(50)
    ]
    service = _service(tmp_path, docs, ['car', 'truck'])

    result = await service.export_dataset(seed=1, copy_images=False)

    for name in ('car', 'truck'):
        train_files = list((Path(result.export_dir) / 'labels' / 'train').glob(f'{name}-*.txt'))
        val_files = list((Path(result.export_dir) / 'labels' / 'val').glob(f'{name}-*.txt'))
        test_files = list((Path(result.export_dir) / 'labels' / 'test').glob(f'{name}-*.txt'))
        total = len(train_files) + len(val_files) + len(test_files)
        assert total == 50
        # Target is 0.8/0.1/0.1 -- allow a couple of rows of slack from rounding.
        assert abs(len(train_files) - 40) <= 2
        assert abs(len(val_files) - 5) <= 2
        assert abs(len(test_files) - 5) <= 2


def test_stratified_split_keeps_group_together():
    """Rows sharing a group_key value never straddle a split."""
    rows = [
        _ExportRow(
            item_id=f'crop-{i}',
            image_id='img-shared',  # all ten items were cut from one image
            image_path='shared.jpg',
            bbox_norm=[0.0, 0.0, 1.0, 1.0],
            class_id=0,
            class_name='car',
        )
        for i in range(10)
    ]
    splits = stratified_split(rows, seed=3, train_ratio=0.8, val_ratio=0.1)
    assert len(set(splits.values())) == 1  # every row in the shared group got the same split


def test_region_bbox_round_trip():
    """normalized bbox -> YOLO cx/cy/w/h text -> back to a normalized bbox."""
    from src.services.curation.export_images import normalized_box, yolo_line

    bbox = [0.1, 0.2, 0.5, 0.6]

    def _yolo_to_norm(cx, cy, w, h):
        return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]

    box = normalized_box(bbox)
    assert box is not None
    parts = yolo_line(3, box).split()
    cid = int(parts[0])
    cx, cy, w, h = (float(x) for x in parts[1:])
    recovered = _yolo_to_norm(cx, cy, w, h)
    assert cid == 3
    for a, b in zip(bbox, recovered, strict=True):
        assert a == pytest.approx(b, abs=1e-5)


def test_normalized_box_clamps_to_the_frame_and_rejects_empty_boxes():
    from src.services.curation.export_images import normalized_box

    assert normalized_box([-0.1, 0.2, 0.5, 1.2]) == (0.0, 0.2, 0.5, 1.0)
    assert normalized_box([0.5, 0.2, 0.5, 0.6]) is None  # zero width
    assert normalized_box([0.6, 0.2, 0.5, 0.6]) is None  # inverted
    assert normalized_box([1.1, 0.2, 1.5, 0.6]) is None  # wholly outside the frame
    assert normalized_box([0.1, 0.2, 0.5]) is None
    assert normalized_box(None) is None


@pytest.mark.asyncio
async def test_letterbox_resize_is_refused(tmp_path):
    """Labels are normalized to the source frame; a letterbox pad would
    shift every box against them."""
    service = _service(tmp_path, [], ['alpha'])
    with pytest.raises(ValueError, match='letterbox'):
        await service.export_dataset(resize_mode='letterbox')  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_manifest_structure_and_deterministic_sha(tmp_path, monkeypatch):
    monkeypatch.setenv('OP_BUILD_SHA', 'deadbeef')
    docs = [
        {
            'crop_id': 'crop-1',
            'image_id': 'img-1',
            'image_path': 'a.jpg',
            'bbox_norm': [0.1, 0.1, 0.5, 0.5],
            'class_id': 0,
            'class_name': 'car',
        },
    ]
    service = _service(tmp_path, docs, ['car'])
    result = await service.export_dataset(seed=99, copy_images=False)
    manifest = json.loads(Path(result.manifest_path).read_text())
    for key in (
        'version_tag',
        'seed',
        'group_key',
        'dataset_sha',
        'image_count',
        'object_count',
        'split_counts',
        'split_object_counts',
        'class_count',
        'classes_with_objects',
        'require_fully_labeled_images',
        'unlabeled_items_on_exported_images',
        'images_with_unlabeled_items',
        'images_dropped_not_fully_labeled',
        'started_at',
        'finished_at',
        'code_sha',
        'frozen_holdout_sha',
        'dedup',
        'max_images',
        'sampling_mode',
        'image_copy',
        'skipped_items',
    ):
        assert key in manifest
    assert manifest['code_sha'] == 'deadbeef'
    assert len(manifest['dataset_sha']) == 64  # full sha256 hex, not the 16-char truncation
    assert manifest['dataset_sha'] == label_content_sha(
        Path(result.export_dir), ['car'], truncate=None
    )


@pytest.mark.asyncio
async def test_manifest_has_frozen_test_sha_and_test_label_sha(tmp_path):
    """W1 lineage: the multi-class manifest must record both test-split
    identity hashes, independently recomputable from the written files by
    the same functions the exporter used (parity with the single-class
    exporter and with ``scripts.curation.bakeoff.freeze.test_sha``)."""
    docs = [
        {
            'crop_id': 'crop-1',
            'image_id': 'img-1',
            'image_path': 'a.jpg',
            'bbox_norm': [0.1, 0.1, 0.5, 0.5],
            'class_id': 0,
            'class_name': 'car',
            'test_holdout': True,
        },
        {
            'crop_id': 'crop-2',
            'image_id': 'img-2',
            'image_path': 'b.jpg',
            'bbox_norm': [0.2, 0.2, 0.6, 0.6],
            'class_id': 1,
            'class_name': 'truck',
            'test_holdout': False,
        },
    ]
    service = _service(tmp_path, docs, ['car', 'truck'])
    result = await service.export_dataset(version_tag='t1', seed=7, copy_images=False)
    manifest = json.loads(Path(result.manifest_path).read_text())
    export_dir = Path(result.export_dir)

    assert manifest['frozen_test_sha'] == frozen_test_sha_of(export_dir)
    assert manifest['test_label_sha'] == label_content_sha(
        export_dir, None, truncate=16, split='test'
    )
    assert manifest['frozen_test_sha']
    assert manifest['test_label_sha']


@pytest.mark.asyncio
async def test_current_symlink_resolves_to_export_dir(tmp_path):
    docs = [
        {
            'crop_id': 'crop-1',
            'image_id': 'img-1',
            'image_path': 'a.jpg',
            'bbox_norm': [0.1, 0.1, 0.5, 0.5],
            'class_id': 0,
            'class_name': 'car',
        },
    ]
    service = _service(tmp_path, docs, ['car'])
    result = await service.export_dataset(copy_images=False)
    cfg = service.config
    resolved = resolve_current_export_dir(cfg)
    assert resolved == Path(result.export_dir).resolve()


@pytest.mark.asyncio
async def test_export_is_rederivable_from_recorded_seed(tmp_path):
    """Re-running the export with the manifest's recorded seed against the
    same item pool reproduces byte-identical labels (modulo the
    timestamp/code-sha fields the manifest itself carries)."""
    docs = [
        {
            'crop_id': f'crop-{i}',
            'image_id': f'img-{i}',
            'image_path': f'{i}.jpg',
            'bbox_norm': [0.0, 0.0, 1.0, 1.0],
            'class_id': i % 2,
            'class_name': 'car' if i % 2 == 0 else 'truck',
            'test_holdout': i == 0,
        }
        for i in range(30)
    ]

    def _label_bytes(export_dir: Path) -> dict[str, bytes]:
        out = {}
        for split in ('train', 'val', 'test'):
            for f in sorted((export_dir / 'labels' / split).glob('*.txt')):
                out[f'{split}/{f.name}'] = f.read_bytes()
        return out

    cfg1 = CurationConfig(export_root=tmp_path / 'run1')
    registry1 = _make_registry(tmp_path / 'reg1', ['car', 'truck'])
    service1 = GenericYoloExportService(_FakeOpenSearch(docs), config=cfg1, registry=registry1)
    result1 = await service1.export_dataset(seed=123, copy_images=False)
    manifest1 = json.loads(Path(result1.manifest_path).read_text())

    cfg2 = CurationConfig(export_root=tmp_path / 'run2')
    registry2 = _make_registry(tmp_path / 'reg2', ['car', 'truck'])
    service2 = GenericYoloExportService(_FakeOpenSearch(docs), config=cfg2, registry=registry2)
    result2 = await service2.export_dataset(seed=manifest1['seed'], copy_images=False)

    assert _label_bytes(Path(result1.export_dir)) == _label_bytes(Path(result2.export_dir))
    assert result1.dataset_sha == result2.dataset_sha
    assert result1.split_counts.to_dict() == result2.split_counts.to_dict()


# ---------------------------------------------------------------------------
# dataset_sha semantics: content-derived, not item-id-derived (export-sha cutover)
# ---------------------------------------------------------------------------


def _split_sensitive_docs() -> list[dict[str, Any]]:
    """Many independent (single-item) image groups so a seed change is
    overwhelmingly likely to move at least one group to a different split,
    changing at least one label file's relative path."""
    return [
        {
            'crop_id': f'{name}-{i}',
            'image_id': f'{name}-img-{i}',
            'image_path': f'{name}-{i}.jpg',
            'bbox_norm': [0.0, 0.0, 1.0, 1.0],
            'class_id': cls_idx,
            'class_name': name,
        }
        for cls_idx, name in enumerate(['car', 'truck'])
        for i in range(40)
    ]


@pytest.mark.asyncio
async def test_dataset_sha_differs_for_the_same_items_with_different_splits(tmp_path):
    """Requirement: same item set, different split assignment -> different
    dataset_sha. The old ``dataset_checksum(item_ids)`` collided here."""
    docs = _split_sensitive_docs()
    service_a = _service(tmp_path / 'a', docs, ['car', 'truck'])
    service_b = _service(tmp_path / 'b', docs, ['car', 'truck'])

    result_a = await service_a.export_dataset(seed=1, copy_images=False)
    result_b = await service_b.export_dataset(seed=2, copy_images=False)

    def _split_of(export_dir: Path) -> dict[str, str]:
        return {
            f.stem: split
            for split in ('train', 'val', 'test')
            for f in (export_dir / 'labels' / split).glob('*.txt')
        }

    splits_a = _split_of(Path(result_a.export_dir))
    splits_b = _split_of(Path(result_b.export_dir))
    assert set(splits_a) == set(splits_b)  # exact same item/image set in both exports
    # ...but at least one image landed in a different split between the two
    # seeds (overwhelmingly likely with 80 independent groups).
    assert splits_a != splits_b
    assert result_a.dataset_sha != result_b.dataset_sha


@pytest.mark.asyncio
async def test_dataset_sha_differs_when_a_box_moves(tmp_path):
    """Requirement: identical item set, one box moved -> different
    dataset_sha (the split is pinned by using the same seed and a single
    item, so only the label content differs)."""

    def _docs(bbox: list[float]) -> list[dict[str, Any]]:
        return [
            {
                'crop_id': 'crop-1',
                'image_id': 'img-1',
                'image_path': 'a.jpg',
                'bbox_norm': bbox,
                'class_id': 0,
                'class_name': 'car',
            }
        ]

    service_a = _service(tmp_path / 'a', _docs([0.1, 0.1, 0.5, 0.5]), ['car'])
    service_b = _service(tmp_path / 'b', _docs([0.1, 0.1, 0.6, 0.6]), ['car'])

    result_a = await service_a.export_dataset(seed=1, copy_images=False)
    result_b = await service_b.export_dataset(seed=1, copy_images=False)

    assert result_a.split_counts.to_dict() == result_b.split_counts.to_dict()
    assert result_a.dataset_sha != result_b.dataset_sha


@pytest.mark.asyncio
async def test_dataset_sha_is_deterministic_for_an_identical_rerun(tmp_path):
    """Requirement: same items, same splits, same label content, run
    twice -> the SAME dataset_sha."""
    docs = [
        {
            'crop_id': 'crop-1',
            'image_id': 'img-1',
            'image_path': 'a.jpg',
            'bbox_norm': [0.1, 0.1, 0.5, 0.5],
            'class_id': 0,
            'class_name': 'car',
        }
    ]
    service_a = _service(tmp_path / 'a', docs, ['car'])
    service_b = _service(tmp_path / 'b', docs, ['car'])

    result_a = await service_a.export_dataset(seed=1, copy_images=False)
    result_b = await service_b.export_dataset(seed=1, copy_images=False)

    assert result_a.dataset_sha == result_b.dataset_sha


@pytest.mark.asyncio
async def test_dataset_sha_differs_for_a_class_rename_with_byte_identical_labels(tmp_path):
    """Requirement: identical label file bytes, but a different class map
    (here, a pure rename — same registry class_id, same ascending
    class_id order, so the dense export id written into the label file is
    unchanged) -> different dataset_sha, because the ordered class-names
    list is folded into the digest."""
    docs = [
        {
            'crop_id': 'crop-1',
            'image_id': 'img-1',
            'image_path': 'a.jpg',
            'bbox_norm': [0.1, 0.1, 0.5, 0.5],
            'class_id': 0,
            'class_name': 'car',
        }
    ]
    registry = _make_registry(tmp_path / 'registry', ['car'])

    cfg_a = CurationConfig(export_root=tmp_path / 'a')
    service_a = GenericYoloExportService(_FakeOpenSearch(docs), config=cfg_a, registry=registry)
    result_a = await service_a.export_dataset(seed=1, copy_images=False)

    registry.rename_class(0, 'automobile')

    cfg_b = CurationConfig(export_root=tmp_path / 'b')
    service_b = GenericYoloExportService(_FakeOpenSearch(docs), config=cfg_b, registry=registry)
    result_b = await service_b.export_dataset(seed=1, copy_images=False)

    label_a = (Path(result_a.export_dir) / 'labels' / 'train' / 'img-1.txt').read_bytes()
    label_b = (Path(result_b.export_dir) / 'labels' / 'train' / 'img-1.txt').read_bytes()
    assert label_a == label_b  # same dense id, same box -- byte-identical label file

    data_yaml_a = (Path(result_a.export_dir) / 'data.yaml').read_text()
    data_yaml_b = (Path(result_b.export_dir) / 'data.yaml').read_text()
    assert data_yaml_a != data_yaml_b  # the rename IS visible in the class map

    assert result_a.dataset_sha != result_b.dataset_sha


@pytest.mark.asyncio
async def test_manifest_dataset_sha_equals_shared_label_content_sha(tmp_path):
    """Requirement: the manifest's dataset_sha is exactly what a third
    party gets from calling the shared hash function directly against the
    export directory -- not some other computation that happens to agree."""
    docs = [
        {
            'crop_id': 'crop-1',
            'image_id': 'img-1',
            'image_path': 'a.jpg',
            'bbox_norm': [0.1, 0.1, 0.5, 0.5],
            'class_id': 0,
            'class_name': 'car',
        },
        {
            'crop_id': 'crop-2',
            'image_id': 'img-2',
            'image_path': 'b.jpg',
            'bbox_norm': [0.2, 0.2, 0.6, 0.6],
            'class_id': 1,
            'class_name': 'truck',
        },
    ]
    service = _service(tmp_path, docs, ['car', 'truck'])

    result = await service.export_dataset(seed=1, copy_images=False)
    manifest = json.loads(Path(result.manifest_path).read_text())
    data_yaml = (Path(result.export_dir) / 'data.yaml').read_text()
    names = json.loads(data_yaml.splitlines()[-1].split('names: ', 1)[1])

    assert manifest['dataset_sha'] == label_content_sha(
        Path(result.export_dir), names, truncate=None
    )
