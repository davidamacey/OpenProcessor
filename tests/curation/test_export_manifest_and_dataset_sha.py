"""Generic export: manifest contents, the current symlink and the dataset content sha."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from curation.export_fakes import FakeOpenSearch, generic_service, make_registry
from src.config import CurationConfig
from src.services.curation.export import (
    GenericYoloExportService,
    frozen_test_sha_of,
    label_content_sha,
    resolve_current_export_dir,
)


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
    service = generic_service(tmp_path, docs, ['car', 'truck'])

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
    service = generic_service(tmp_path, docs, ['car'])
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
        'item_filter',
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
    """Lineage: the multi-class manifest must record both test-split
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
    service = generic_service(tmp_path, docs, ['car', 'truck'])
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
    service = generic_service(tmp_path, docs, ['car'])
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
    registry1 = make_registry(tmp_path / 'reg1', ['car', 'truck'])
    service1 = GenericYoloExportService(FakeOpenSearch(docs), config=cfg1, registry=registry1)
    result1 = await service1.export_dataset(seed=123, copy_images=False)
    manifest1 = json.loads(Path(result1.manifest_path).read_text())

    cfg2 = CurationConfig(export_root=tmp_path / 'run2')
    registry2 = make_registry(tmp_path / 'reg2', ['car', 'truck'])
    service2 = GenericYoloExportService(FakeOpenSearch(docs), config=cfg2, registry=registry2)
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
    service_a = generic_service(tmp_path / 'a', docs, ['car', 'truck'])
    service_b = generic_service(tmp_path / 'b', docs, ['car', 'truck'])

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

    service_a = generic_service(tmp_path / 'a', _docs([0.1, 0.1, 0.5, 0.5]), ['car'])
    service_b = generic_service(tmp_path / 'b', _docs([0.1, 0.1, 0.6, 0.6]), ['car'])

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
    service_a = generic_service(tmp_path / 'a', docs, ['car'])
    service_b = generic_service(tmp_path / 'b', docs, ['car'])

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
    registry = make_registry(tmp_path / 'registry', ['car'])

    cfg_a = CurationConfig(export_root=tmp_path / 'a')
    service_a = GenericYoloExportService(FakeOpenSearch(docs), config=cfg_a, registry=registry)
    result_a = await service_a.export_dataset(seed=1, copy_images=False)

    registry.rename_class(0, 'automobile')

    cfg_b = CurationConfig(export_root=tmp_path / 'b')
    service_b = GenericYoloExportService(FakeOpenSearch(docs), config=cfg_b, registry=registry)
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
    service = generic_service(tmp_path, docs, ['car', 'truck'])

    result = await service.export_dataset(seed=1, copy_images=False)
    manifest = json.loads(Path(result.manifest_path).read_text())
    data_yaml = (Path(result.export_dir) / 'data.yaml').read_text()
    names = json.loads(data_yaml.splitlines()[-1].split('names: ', 1)[1])

    assert manifest['dataset_sha'] == label_content_sha(
        Path(result.export_dir), names, truncate=None
    )


@pytest.mark.asyncio
async def test_export_item_filter_narrows_the_cohort_query_and_is_recorded(tmp_path):
    from src.services.curation.item_filter import ItemFilter

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
    service = generic_service(tmp_path, docs, ['car'])
    queries: list[dict[str, Any]] = []
    scroll = service._scroll_items

    async def spy(query: dict[str, Any]) -> list[dict[str, Any]]:
        queries.append(query)
        return await scroll(query)

    service._scroll_items = spy  # type: ignore[method-assign]
    flt = ItemFilter(class_names=['car'], conf_min=0.5, min_area=0.01)

    result = await service.export_dataset(version_tag='f1', copy_images=False, item_filter=flt)

    clauses = queries[0]['bool']['filter']
    assert {'term': {'class_validated': True}} in clauses
    assert any('case_insensitive' in str(c) for c in clauses)  # the class-name clause
    assert any('crop_area_norm' in str(c) for c in clauses)
    manifest = json.loads((Path(result.export_dir) / 'manifest.json').read_text())
    assert manifest['item_filter'] == {'class_names': ['car'], 'conf_min': 0.5, 'min_area': 0.01}

    plain = await generic_service(tmp_path / 'p', docs, ['car']).export_dataset(
        version_tag='f2', copy_images=False
    )
    assert json.loads((Path(plain.export_dir) / 'manifest.json').read_text())['item_filter'] is None
