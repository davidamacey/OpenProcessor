"""W10.2.2: an OpenProcessor export is read back faithfully.

Every export here is produced by the REAL exporters; the reader must take
class names from ``class_registry.json`` (never from a position in any
registry), keep the frozen test split, turn backgrounds into reviewed
negatives and keep strata -- and must survive a hostile directory.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from curation.dataset_import.op_export_fixtures import (
    IMAGES,
    ITEMS,
    export_multi_class,
    export_region_whole_frame,
    item,
    multi_registry,
    region_item,
)
from curation.query_fakes import QueryFakeOpenSearch
from src.config.region_state import RegionStatus
from src.services.curation.dataset_import.op_export import read_op_export
from src.services.curation.dataset_import.op_export_stems import (
    StemResolution,
    project_item_crop_boxes,
    resolve_stems,
)
from src.services.curation.dataset_import.paths import dataset_path_guard
from src.services.curation.dataset_import.scan import detect_format


if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.asyncio


def _codes(scan) -> dict[str, int]:
    return {i.code: i.count for i in scan.issues.issues()}


def _multi_docs(n: int = 6) -> dict:
    docs = {}
    for i in range(n):
        cls, name = [(0, 'car'), (2, 'bus')][i % 2]
        docs[f'c{i}'] = item(f'c{i}', f'img{i}', cls, name, [0.1, 0.1, 0.1 * (i + 2), 0.5])
    return docs


async def _multi_export(tmp_path: Path) -> Path:
    # truck is merged away: dense ids are {car: 0, bus: 1} while registry ids
    # are {car: 0, bus: 2}, so a raw index would name registry id 1 'truck'.
    reg = multi_registry(tmp_path / 'reg', ['car', 'truck', 'bus'], deprecate=('truck',))
    export_dir, _ = await export_multi_class(tmp_path, _multi_docs(), reg)
    return export_dir


# ------------------------------------------------------------------ multi-class


async def test_multi_class_export_is_detected_and_classes_come_from_the_registry_file(
    tmp_path: Path,
) -> None:
    export_dir = await _multi_export(tmp_path)
    assert detect_format(export_dir) == 'openprocessor_export'
    scan = read_op_export(export_dir)
    info = scan.op_export
    assert info is not None
    assert scan.class_ids == {'car': 0, 'bus': 1}
    # The source project's registry ids ride along as a hint only.
    assert info.source_classes == {'car': 0, 'bus': 2}
    assert info.dataset_kind == 'multi_class'
    assert info.manifest_dataset_sha == info.recomputed_dataset_sha
    assert len(info.manifest_dataset_sha or '') == 64
    assert _codes(scan)['op_export_detected'] == 1
    names = {e.source_stem: {b.dataset_class for b in e.boxes} for e in scan.entries}
    assert len(names) == 6
    # image i was labeled car for even i, bus for odd i -- never 'truck'.
    expected = {f'img{i}': {['car', 'bus'][i % 2]} for i in range(6)}
    assert names == expected
    assert all(e.stem_kind == 'frame' for e in scan.entries)


async def test_detect_format_requires_all_three_files(tmp_path: Path) -> None:
    export_dir = await _multi_export(tmp_path)
    (export_dir / 'data.yaml').unlink()
    assert detect_format(export_dir) != 'openprocessor_export'


async def test_names_mismatch_is_blocking_and_reads_nothing(tmp_path: Path) -> None:
    export_dir = await _multi_export(tmp_path)
    yaml_path = export_dir / 'data.yaml'
    yaml_path.write_text(yaml_path.read_text().replace('["car", "bus"]', '["bus", "car"]'))
    scan = read_op_export(export_dir)
    assert 'names_mismatch' in _codes(scan)
    assert scan.entries == []


async def test_dataset_sha_mismatch_after_a_label_edit(tmp_path: Path) -> None:
    export_dir = await _multi_export(tmp_path)
    assert 'dataset_sha_mismatch' not in _codes(read_op_export(export_dir))
    label = next(iter(sorted((export_dir / 'labels' / 'train').glob('*.txt'))), None) or next(
        iter(sorted(export_dir.glob('labels/*/*.txt')))
    )
    label.write_text(label.read_text() + '0 0.500000 0.500000 0.100000 0.100000\n')
    scan = read_op_export(export_dir)
    assert _codes(scan)['dataset_sha_mismatch'] == 1
    info = scan.op_export
    assert info is not None
    assert info.manifest_dataset_sha != info.recomputed_dataset_sha


async def test_test_split_changed_after_editing_a_frozen_test_label(tmp_path: Path) -> None:
    from scripts.curation.bakeoff import freeze

    export_dir = await _multi_export(tmp_path)
    test_labels = sorted((export_dir / 'labels' / 'test').glob('*.txt'))
    assert test_labels, 'fixture must put at least one frame in test'
    freeze.freeze(export_dir)
    scan = read_op_export(export_dir)
    assert 'test_split_changed' not in _codes(scan)
    info = scan.op_export
    assert info is not None
    assert info.test_frozen.verified
    assert info.freeze_test_split_default is True

    freeze.freeze  # noqa: B018 - lock written above stays; now break a label
    target = test_labels[0]
    target.chmod(0o644)
    target.write_text(target.read_text() + '1 0.500000 0.500000 0.100000 0.100000\n')
    broken = read_op_export(export_dir)
    assert _codes(broken)['test_split_changed'] == 1
    assert broken.op_export is not None
    assert broken.op_export.freeze_test_split_default is False


async def test_freeze_defaults_without_a_lock_file(tmp_path: Path) -> None:
    export_dir = await _multi_export(tmp_path)
    scan = read_op_export(export_dir)
    assert 'test_frozen_missing' in _codes(scan)
    assert scan.op_export is not None
    assert scan.op_export.freeze_test_split_default is False

    manifest_path = export_dir / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['frozen_holdout_sha'] = 'a' * 64
    manifest_path.write_text(json.dumps(manifest))
    scan = read_op_export(export_dir)
    assert 'test_frozen_from_manifest' in _codes(scan)
    assert scan.op_export is not None
    assert scan.op_export.freeze_test_split_default is True


# ----------------------------------------------------------------- single-class


def _region_docs() -> dict:
    return {
        'r0': region_item('r0', 'frameA', RegionStatus.DETECTED, box=[0.2, 0.2, 0.4, 0.4]),
        'r1': region_item('r1', 'frameB', RegionStatus.DETECTED, box=[0.3, 0.3, 0.6, 0.6]),
        'r2': region_item('r2', 'frameC', RegionStatus.DETECTED, box=[0.1, 0.5, 0.3, 0.9]),
        'fp': region_item('fp', 'frameFP', RegionStatus.FALSE_POSITIVE),
    }


def _negative_image(image_id: str, **extra) -> dict:
    return {
        'image_id': image_id,
        'image_path': f'/data/{image_id}.jpg',
        'import_label_state': 'negative',
        'negative_for': ['plate'],
        **extra,
    }


async def test_region_export_backgrounds_strata_and_names(tmp_path: Path) -> None:
    images = {
        'bgImport': _negative_image('bgImport', import_stratum='bg:imported', dataset_split='val'),
    }
    export_dir, _ = await export_region_whole_frame(tmp_path, _region_docs(), images=images)
    scan = read_op_export(export_dir)
    info = scan.op_export
    assert info is not None
    assert info.is_region_export
    assert info.box_source == 'region'
    assert scan.class_ids == {'plate': 0}
    assert info.parent_classes == ['car']
    assert info.stratum_map.present
    by_stem = {e.source_stem: e for e in scan.entries}
    assert by_stem['bgImport'].label_state == 'negative'
    assert by_stem['bgImport'].split == 'val'
    assert by_stem['bgImport'].stratum == 'bg:imported'
    assert by_stem['frameFP'].label_state == 'negative'
    assert by_stem['frameFP'].hard_negative is True
    assert by_stem['frameFP'].stratum is not None
    assert by_stem['frameFP'].stratum.startswith('neg:')
    assert by_stem['frameA'].label_state == 'labeled'
    assert by_stem['frameA'].hard_negative is False
    assert [b.dataset_class for b in by_stem['frameA'].boxes] == ['plate']
    assert all(e.stem_kind == 'frame' for e in scan.entries)
    assert 'dataset_sha_mismatch' not in _codes(scan)


async def test_item_crop_export_is_flagged_and_stems_are_crop_ids(tmp_path: Path) -> None:
    docs = {
        'r0': region_item('r0', 'frameA', RegionStatus.DETECTED, box=[0.2, 0.2, 0.4, 0.4]),
        'r1': region_item('r1', 'frameB', RegionStatus.DETECTED, box=[0.3, 0.3, 0.6, 0.6]),
        'r2': region_item('r2', 'frameC', RegionStatus.DETECTED, box=[0.3, 0.3, 0.7, 0.7]),
    }
    export_dir, _ = await export_region_whole_frame(tmp_path, docs, image_mode='item_crop')
    scan = read_op_export(export_dir)
    assert {e.source_stem for e in scan.entries} == {'r0', 'r1', 'r2'}
    assert all(e.stem_kind == 'item_crop' for e in scan.entries)
    # Nothing in the project resolves them: crop-sized frames, warned.
    fake = QueryFakeOpenSearch({ITEMS: {}, IMAGES: {}})
    issues = scan.issues
    resolved = await resolve_stems(
        fake, scan.entries, images_index=IMAGES, items_index=ITEMS, issues=issues
    )
    assert resolved == {}
    assert _codes(scan)['item_crop_export_imported_as_frames'] == 3


# --------------------------------------------------------------- resolve_stems


async def test_resolve_stems_frames_and_crops_and_projection(tmp_path: Path) -> None:
    export_dir = await _multi_export(tmp_path)
    scan = read_op_export(export_dir)
    fake = QueryFakeOpenSearch(
        {
            IMAGES: {'img0': {'image_id': 'img0', 'image_path': '/data/img0.jpg'}},
            ITEMS: {},
        }
    )
    resolved = await resolve_stems(
        fake, scan.entries, images_index=IMAGES, items_index=ITEMS, issues=scan.issues
    )
    assert resolved == {'img0': StemResolution('frame', 'img0', '/data/img0.jpg')}
    assert _codes(scan)['images_reused_by_stem'] == 1

    crop = StemResolution('item_crop', 'frameA', '/data/a.jpg', 'c1', (0.2, 0.2, 0.6, 0.8))
    from src.services.curation.dataset_import.scan import LabelBox

    projected = project_item_crop_boxes([LabelBox('plate', (0.0, 0.0, 0.5, 0.5))], crop)
    assert projected[0].bbox_norm == pytest.approx((0.2, 0.2, 0.4, 0.5))
    frame = StemResolution('frame', 'x', '/d/x.jpg')
    same = [LabelBox('plate', (0.0, 0.0, 0.5, 0.5))]
    assert project_item_crop_boxes(same, frame) == same


async def test_resolve_stems_item_crop_found(tmp_path: Path) -> None:
    from src.services.curation.dataset_import.scan import ScanEntry

    fake = QueryFakeOpenSearch(
        {
            ITEMS: {
                'cropX': {
                    'crop_id': 'cropX',
                    'image_id': 'frameX',
                    'image_path': '/data/x.jpg',
                    'bbox_norm': [0.1, 0.1, 0.5, 0.5],
                }
            },
            IMAGES: {},
        }
    )
    entry = ScanEntry(
        rel_path='images/train/cropX.jpg',
        source_stem='cropX',
        abs_image_path=tmp_path / 'cropX.jpg',
        split='train',
        label_state='labeled',
        stem_kind='item_crop',
    )
    resolved = await resolve_stems(fake, [entry], images_index=IMAGES, items_index=ITEMS)
    assert resolved['cropX'] == StemResolution(
        'item_crop', 'frameX', '/data/x.jpg', 'cropX', (0.1, 0.1, 0.5, 0.5)
    )


# ------------------------------------------------------------------ hostile input


@pytest.mark.parametrize('victim', ['manifest.json', 'class_registry.json'])
@pytest.mark.parametrize('payload', ['{not json', '[1, 2]', '"text"', 'null'])
async def test_malformed_manifest_or_registry_is_a_blocking_issue(
    tmp_path: Path, victim: str, payload: str
) -> None:
    export_dir = await _multi_export(tmp_path)
    (export_dir / victim).write_text(payload)
    scan = read_op_export(export_dir)
    assert 'manifest_unreadable' in _codes(scan)
    assert scan.entries == []


@pytest.mark.parametrize(
    'mutate',
    [
        lambda r: r.update(classes='x'),
        lambda r: r.update(export_id_map=[1]),
        lambda r: r.update(export_id_map={'999': 0}),
        lambda r: r.update(export_id_map={'0': 0, '2': 0}),
        lambda r: r.update(export_id_map={'0': -1}),
        lambda r: r.update(export_id_map={'zero': 'one'}),
        lambda r: r.update(classes=[{'class_id': 'a', 'class_name': 'car'}]),
        lambda r: r.update(classes=[{'class_id': 0, 'class_name': 5}]),
        lambda r: r.update(names=['car', 'wrong']),
    ],
)
async def test_inconsistent_class_registry_never_raises(tmp_path: Path, mutate) -> None:
    export_dir = await _multi_export(tmp_path)
    path = export_dir / 'class_registry.json'
    registry = json.loads(path.read_text())
    mutate(registry)
    path.write_text(json.dumps(registry))
    scan = read_op_export(export_dir)
    assert 'manifest_unreadable' in _codes(scan)
    assert scan.entries == []


async def test_oversize_manifest_is_refused_unread(tmp_path: Path, monkeypatch) -> None:
    import src.services.curation.dataset_import.op_export as mod

    export_dir = await _multi_export(tmp_path)
    monkeypatch.setattr(mod, 'MAX_MANIFEST_BYTES', 10)
    assert 'manifest_unreadable' in _codes(read_op_export(export_dir))


async def test_malformed_data_yaml_is_an_issue(tmp_path: Path) -> None:
    export_dir = await _multi_export(tmp_path)
    (export_dir / 'data.yaml').write_text('names: [unterminated\n')
    scan = read_op_export(export_dir)
    assert 'data_yaml_invalid' in _codes(scan)
    assert scan.entries == []


@pytest.mark.parametrize(
    'payload',
    ['{broken', '[1]', '{"x": 5, "y": ["a"]}'],
)
async def test_malformed_stratum_map_is_a_warning_not_an_error(
    tmp_path: Path, payload: str
) -> None:
    export_dir, _ = await export_region_whole_frame(tmp_path, _region_docs())
    (export_dir / 'stratum_map.json').write_text(payload)
    scan = read_op_export(export_dir)
    assert _codes(scan).get('manifest_unreadable') is None
    assert scan.entries
    assert all(e.stratum is None for e in scan.entries)


async def test_traversal_stems_in_stratum_map_are_only_dictionary_keys(tmp_path: Path) -> None:
    export_dir, _ = await export_region_whole_frame(tmp_path, _region_docs())
    path = export_dir / 'stratum_map.json'
    strata = json.loads(path.read_text())
    strata.update({'../../etc/passwd': 'pos:1', '/abs/path': 'neg:2', 'a\x00b': 'bg:3'})
    path.write_text(json.dumps(strata))
    scan = read_op_export(export_dir)
    assert _codes(scan)['stratum_map_partial'] == 3
    assert {e.source_stem for e in scan.entries} <= {
        p.stem for p in export_dir.glob('labels/*/*.txt')
    }


async def test_symlinked_image_and_label_are_never_followed(tmp_path: Path) -> None:
    export_dir = await _multi_export(tmp_path)
    secret = tmp_path / 'secret.txt'
    secret.write_text('0 0.5 0.5 0.2 0.2\n')
    outside_image = tmp_path / 'outside.jpg'
    from PIL import Image

    Image.new('RGB', (8, 8)).save(outside_image, format='JPEG')
    label = sorted(export_dir.glob('labels/*/*.txt'))[0]
    label.unlink()
    label.symlink_to(secret)
    image = sorted((export_dir / 'images' / label.parent.name).glob('*.jpg'))[-1]
    victim_stem = image.stem
    image.unlink()
    image.symlink_to(outside_image)
    scan = read_op_export(export_dir, path_guard=dataset_path_guard(tmp_path))
    codes = _codes(scan)
    assert 'dataset_path_not_allowed' in codes or 'label_file_orphan' in codes
    assert 'image_path_not_servable' in codes
    stems = {e.source_stem for e in scan.entries}
    assert label.stem not in {e.source_stem for e in scan.entries if e.boxes}
    assert victim_stem not in stems
    assert 'dataset_sha_mismatch' in codes  # recompute refused over symlinks


@pytest.mark.parametrize('lock', ['{broken', '[1, 2]', '"str"', '{"test_label_sha": 5}'])
async def test_malformed_lock_file_is_test_split_changed_not_a_crash(
    tmp_path: Path, lock: str
) -> None:
    export_dir = await _multi_export(tmp_path)
    (export_dir / 'TEST_FROZEN.json').write_text(lock)
    scan = read_op_export(export_dir)
    assert 'test_split_changed' in _codes(scan)
    assert scan.op_export is not None
    assert not scan.op_export.test_frozen.verified


async def test_symlinked_lock_file_is_not_read(tmp_path: Path) -> None:
    export_dir = await _multi_export(tmp_path)
    real = tmp_path / 'lock.json'
    real.write_text('{"test_label_sha": "abc"}')
    (export_dir / 'TEST_FROZEN.json').symlink_to(real)
    scan = read_op_export(export_dir)
    assert 'test_split_changed' in _codes(scan)


async def test_image_files_outside_the_guard_are_skipped(tmp_path: Path) -> None:
    export_dir = await _multi_export(tmp_path)
    scan = read_op_export(export_dir, path_guard=lambda _p: False)
    assert scan.entries == []
    assert _codes(scan)['image_path_not_servable'] >= 6


async def test_a_symlink_inside_the_frozen_test_labels_is_never_verified(tmp_path: Path) -> None:
    from scripts.curation.bakeoff import freeze

    export_dir = await _multi_export(tmp_path)
    freeze.freeze(export_dir)
    outside = tmp_path / 'outside.txt'
    outside.write_text('0 0.5 0.5 0.2 0.2\n')
    victim = sorted((export_dir / 'labels' / 'test').glob('*.txt'))[0]
    victim.chmod(0o644)
    victim.unlink()
    victim.symlink_to(outside)
    info = read_op_export(export_dir).op_export
    assert info is not None
    assert info.test_frozen.present
    assert not info.test_frozen.verified
    assert 'symlinks' in info.test_frozen.message


def test_a_deeply_nested_manifest_reads_as_absent_not_as_a_crash(tmp_path: Path) -> None:
    from src.services.curation.dataset_import.op_export import read_json_object

    (tmp_path / 'manifest.json').write_text('[' * 100_000 + ']' * 100_000)
    assert read_json_object(tmp_path, 'manifest.json') is None
    (tmp_path / 'ok.json').write_text('{"a": 1}')
    assert read_json_object(tmp_path, 'ok.json') == {'a': 1}
