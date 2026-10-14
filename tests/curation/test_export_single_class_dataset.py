"""Single-class and class-subset dataset export (G2): primitives, layout, sampling, determinism and the manifest shape."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from curation.export_fakes import (
    FIELDS,
    FakeOpenSearch,
    make_item,
    make_region_item,
    make_registry,
    single_class_service,
)
from src.config import CurationConfig
from src.config.region_state import RegionStatus
from src.services.curation.export_single_class import (
    SingleClassExportProfile,
    SingleClassExportService,
    resolve_current_single_class_dir,
)
from src.services.curation.export_single_class_rows import (
    dominant,
    reproject_into_crop,
    xyxy_to_yolo,
)
from src.services.curation.export_support import frozen_test_sha_of, label_content_sha


# ---------------------------------------------------------------------------
# Geometry + integrity primitives
# ---------------------------------------------------------------------------


def test_xyxy_to_yolo_converts_and_rejects_degenerate():
    assert xyxy_to_yolo([0.0, 0.0, 1.0, 1.0]) == (0.5, 0.5, 1.0, 1.0)
    assert xyxy_to_yolo([0.2, 0.4, 0.2, 0.8]) is None  # zero width
    assert xyxy_to_yolo(None) is None
    assert xyxy_to_yolo([0.1, 0.2]) is None


def test_reproject_into_crop_maps_and_drops_outside_boxes():
    crop = (0.25, 0.25, 0.75, 0.75)
    # A box filling the right half of the crop.
    mapped = reproject_into_crop([0.5, 0.25, 0.75, 0.75], crop)
    assert mapped == pytest.approx((0.75, 0.5, 0.5, 1.0))
    # Entirely left of the crop -> clamps to zero area -> dropped.
    assert reproject_into_crop([0.0, 0.0, 0.2, 0.2], crop) is None


def test_dominant_breaks_ties_deterministically():
    assert dominant(['b', 'a', 'b', 'a']) == 'a'
    assert dominant([]) == ''


def test_label_content_sha_tracks_content_not_just_filenames(tmp_path):
    """The whole point of hashing content: a corrected box must change the
    dataset sha, which a checksum over item ids alone would miss."""
    for split in ('train', 'val', 'test'):
        (tmp_path / 'labels' / split).mkdir(parents=True)
    label = tmp_path / 'labels' / 'train' / 'a.txt'
    label.write_text('0 0.500000 0.500000 0.100000 0.100000\n')
    before = label_content_sha(tmp_path)

    label.write_text('0 0.500000 0.500000 0.200000 0.200000\n')
    after = label_content_sha(tmp_path)

    assert before
    assert after
    assert before != after


@pytest.mark.asyncio
async def test_single_class_dataset_sha_differs_for_a_class_rename_with_byte_identical_labels(
    tmp_path,
):
    """This exporter's dense id is the row's position in
    ``profile.class_ids`` (fixed by the caller, not derived from the live
    registry's id ordering), so the same investigation the multi-class
    exporter needed applies here too: a pure registry rename (same
    class_id, same class_ids tuple) leaves the label bytes unchanged but
    must still change dataset_sha, because the ordered class_names are
    folded into the digest."""
    docs = [make_item(1, 0)]
    registry = make_registry(tmp_path / 'registry', ['car'])
    profile = SingleClassExportProfile(class_ids=(0,))

    service_a = SingleClassExportService(
        FakeOpenSearch(docs),
        profile=profile,
        config=CurationConfig(export_root=tmp_path / 'a', source_root=tmp_path / 'images'),
        registry=registry,
        region_fields=FIELDS,
    )
    result_a = await service_a.export(seed=1, copy_images=False)

    registry.rename_class(0, 'automobile')

    service_b = SingleClassExportService(
        FakeOpenSearch(docs),
        profile=profile,
        config=CurationConfig(export_root=tmp_path / 'b', source_root=tmp_path / 'images'),
        registry=registry,
        region_fields=FIELDS,
    )
    result_b = await service_b.export(seed=1, copy_images=False)

    label_a = next((Path(result_a.export_dir) / 'labels').rglob('img-1.txt')).read_bytes()
    label_b = next((Path(result_b.export_dir) / 'labels').rglob('img-1.txt')).read_bytes()
    assert label_a == label_b

    assert result_a.dataset_sha != result_b.dataset_sha


def test_frozen_test_sha_ignores_content_changes_inside_the_test_split(tmp_path):
    """The guarantee is 'the same frames are held out', which must keep
    holding after a label correction inside the test set."""
    (tmp_path / 'labels' / 'test').mkdir(parents=True)
    label = tmp_path / 'labels' / 'test' / 'a.txt'
    label.write_text('0 0.5 0.5 0.1 0.1\n')
    before = frozen_test_sha_of(tmp_path)

    label.write_text('0 0.5 0.5 0.9 0.9\n')
    assert frozen_test_sha_of(tmp_path) == before

    (tmp_path / 'labels' / 'test' / 'b.txt').write_text('')
    assert frozen_test_sha_of(tmp_path) != before


def test_frozen_test_sha_empty_without_a_test_split(tmp_path):
    assert frozen_test_sha_of(tmp_path) == ''


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_item_mode_requires_class_ids(tmp_path):
    service = single_class_service(tmp_path, FakeOpenSearch([]), SingleClassExportProfile())
    with pytest.raises(ValueError, match='class_ids'):
        await service.export()


@pytest.mark.asyncio
async def test_item_crop_mode_rejected_for_item_box_source(tmp_path):
    profile = SingleClassExportProfile(class_ids=(0,))
    service = single_class_service(tmp_path, FakeOpenSearch([]), profile)
    with pytest.raises(ValueError, match='item_crop'):
        await service.export(image_mode='item_crop')


@pytest.mark.asyncio
async def test_unknown_image_mode_rejected(tmp_path):
    profile = SingleClassExportProfile(class_ids=(0,))
    service = single_class_service(tmp_path, FakeOpenSearch([]), profile)
    with pytest.raises(ValueError, match='image_mode'):
        await service.export(image_mode='sideways')  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Item box source: single class + class subset
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_single_class_export_writes_labels_yaml_and_symlink(tmp_path):
    docs = [make_item(i, 0) for i in range(6)] + [make_item(100 + i, 1) for i in range(4)]
    profile = SingleClassExportProfile(class_ids=(0,), name='regions')
    service = single_class_service(tmp_path, FakeOpenSearch(docs), profile)

    result = await service.export(version_tag='v1', seed=11, copy_images=False)

    assert result.class_count == 1
    assert result.positive_images == 6
    export_dir = Path(result.export_dir)

    # Every positive label carries dense id 0 — the single target class.
    written = sorted((export_dir / 'labels').rglob('*.txt'))
    non_empty = [p for p in written if p.read_text().strip()]
    assert len(non_empty) == 6
    assert all(p.read_text().startswith('0 ') for p in non_empty)

    data_yaml = (export_dir / 'data.yaml').read_text()
    assert 'nc: 1\n' in data_yaml
    assert '  0: car\n' in data_yaml
    assert f'path: {export_dir}\n' in data_yaml

    # The profile gets its OWN current symlink, under its own root — the
    # multi-class export root is untouched.
    link = tmp_path / 'exports' / 'regions' / 'current'
    assert link.resolve() == export_dir.resolve()
    assert not (tmp_path / 'exports' / 'current').exists()
    assert resolve_current_single_class_dir(profile, service.config) == export_dir.resolve()


@pytest.mark.asyncio
async def test_class_subset_export_uses_dense_ids_in_profile_order(tmp_path):
    """Dense label ids are the index into ``class_ids``, not the registry id,
    so a subset export's label files are self-consistent with its data.yaml."""
    docs = [make_item(0, 2), make_item(1, 0), make_item(2, 1)]
    # Deliberately non-ascending: the profile's order IS the dataset order.
    profile = SingleClassExportProfile(class_ids=(2, 0))
    service = single_class_service(tmp_path, FakeOpenSearch(docs), profile)

    result = await service.export(copy_images=False)
    export_dir = Path(result.export_dir)

    assert result.class_count == 2
    data_yaml = (export_dir / 'data.yaml').read_text()
    assert '  0: bus\n' in data_yaml  # registry id 2
    assert '  1: car\n' in data_yaml  # registry id 0

    dense_by_frame = {
        p.stem: p.read_text().split()[0]
        for p in (export_dir / 'labels').rglob('*.txt')
        if p.read_text().strip()
    }
    assert dense_by_frame == {'img-0': '0', 'img-1': '1'}

    registry = json.loads((export_dir / 'class_registry.json').read_text())
    assert registry['export_id_map'] == {'2': 0, '0': 1}
    assert registry['names'] == ['bus', 'car']

    stats = json.loads((export_dir / 'label_stats.json').read_text())
    assert stats == {'bus': 1, 'car': 1}


@pytest.mark.asyncio
async def test_non_target_frames_become_background_images(tmp_path):
    """A narrowed detector must learn to NOT fire on the classes it dropped,
    so frames holding only non-target items are emitted as empty labels."""
    docs = [make_item(i, 0) for i in range(10)] + [make_item(100 + i, 1) for i in range(10)]
    profile = SingleClassExportProfile(class_ids=(0,))
    service = single_class_service(tmp_path, FakeOpenSearch(docs), profile)

    result = await service.export(empty_bg_ratio=0.5, seed=3, copy_images=False)

    assert result.positive_images == 10
    assert result.background_images == 5  # round(10 * 0.5)
    export_dir = Path(result.export_dir)
    empty = [p for p in (export_dir / 'labels').rglob('*.txt') if not p.read_text().strip()]
    assert len(empty) == 5


@pytest.mark.asyncio
async def test_max_positive_images_caps_evenly_across_strata(tmp_path):
    """The cap runs through the shared ``even_stratified_sample``, so the
    rare class survives instead of being crowded out by the dominant one."""
    docs = [make_item(i, 0) for i in range(50)] + [make_item(100 + i, 1) for i in range(2)]
    profile = SingleClassExportProfile(class_ids=(0, 1))
    service = single_class_service(tmp_path, FakeOpenSearch(docs), profile)

    result = await service.export(max_positive_images=6, seed=5, copy_images=False)

    assert result.positive_images == 6
    manifest = json.loads(Path(result.manifest_path).read_text())
    assert manifest['sampling_mode'] == 'stratified_even'
    # Both strata present: an even round-robin, not a flat truncation.
    assert set(manifest['stratum_distribution']) == {'pos:0', 'pos:1'}


@pytest.mark.asyncio
async def test_test_holdout_frames_are_pinned_to_the_test_split(tmp_path):
    docs = [make_item(i, 0) for i in range(9)]
    docs.append(make_item(99, 0, test_holdout=True))
    profile = SingleClassExportProfile(class_ids=(0,))
    service = single_class_service(tmp_path, FakeOpenSearch(docs), profile)

    result = await service.export(seed=1, copy_images=False)

    assert (Path(result.export_dir) / 'labels' / 'test' / 'img-99.txt').is_file()


@pytest.mark.asyncio
async def test_skip_test_split_emits_only_train_and_val(tmp_path):
    docs = [make_item(i, 0) for i in range(20)]
    docs.append(make_item(99, 0, test_holdout=True))
    profile = SingleClassExportProfile(class_ids=(0,))
    service = single_class_service(tmp_path, FakeOpenSearch(docs), profile)

    result = await service.export(seed=1, skip_test_split=True, copy_images=False)

    assert result.split_counts.test == 0
    # No test split means no frozen-test guarantee to make.
    assert result.frozen_test_sha == ''


@pytest.mark.asyncio
async def test_export_is_reproducible_for_the_same_seed(tmp_path):
    docs = [make_item(i, 0) for i in range(12)] + [make_item(100 + i, 1) for i in range(6)]
    profile = SingleClassExportProfile(class_ids=(0,))

    shas = []
    for run in range(2):
        service = single_class_service(tmp_path / f'run{run}', FakeOpenSearch(docs), profile)
        result = await service.export(seed=42, empty_bg_ratio=0.5, copy_images=False)
        shas.append((result.dataset_sha, result.frozen_test_sha))

    assert shas[0] == shas[1]
    assert all(sha for pair in shas for sha in pair)


@pytest.mark.asyncio
async def test_zero_positive_export_is_flagged_not_silent(tmp_path):
    """A background-only dataset can't train a detector; the manifest has to
    say so rather than looking like an ordinary small export."""
    docs = [make_item(i, 1) for i in range(4)]
    profile = SingleClassExportProfile(class_ids=(0,))
    service = single_class_service(tmp_path, FakeOpenSearch(docs), profile)

    result = await service.export(copy_images=False)

    assert result.positive_images == 0
    manifest = json.loads(Path(result.manifest_path).read_text())
    assert manifest['positives_zero_warning'] is True


@pytest.mark.asyncio
async def test_symlink_flip_retargets_a_previous_export(tmp_path):
    docs = [make_item(i, 0) for i in range(4)]
    profile = SingleClassExportProfile(class_ids=(0,))
    service = single_class_service(tmp_path, FakeOpenSearch(docs), profile)

    first = await service.export(export_dir=tmp_path / 'e1', copy_images=False)
    second = await service.export(export_dir=tmp_path / 'e2', copy_images=False)

    link = tmp_path / 'exports' / 'single_class' / 'current'
    assert link.resolve() == Path(second.export_dir).resolve()
    assert link.resolve() != Path(first.export_dir).resolve()


# ---------------------------------------------------------------------------
# Acceptance: manifest parity with the reference single-class export
# ---------------------------------------------------------------------------

# Exact manifest key set written by an earlier internal version's
# single-class export service. Pinned here as literal
# data so this test fails if the generic exporter ever drops one of the
# fields a downstream training lineage reads.
_REFERENCE_MANIFEST_KEYS: frozenset[str] = frozenset(
    {
        'dataset_kind',
        'version_tag',
        'exported_at',
        'started_at',
        'class_count',
        'class_name',
        'image_mode',
        'img_max_side',
        'empty_bg_ratio',
        'max_positive_images',
        'sampling',
        'dedup',
        'positive_images',
        'background_images',
        'false_positive_background_images',
        'positives_zero_warning',
        'dataset_sha',
        'frozen_test_sha',
        'git_sha',
        'split_counts',
        'stratum_distribution',
        'stratum_count',
        'image_count',
    }
)

# The only two reference keys this exporter renames, both onto names the
# public branch had already standardized on BEFORE this exporter existed:
#   - 'sampling'  -> 'sampling_mode', the name the multi-class exporter's
#     manifest already uses for the same 'all' / 'stratified_even' value.
#   - 'git_sha'   -> 'code_sha', ditto, and additionally honors the
#     OP_BUILD_SHA baked into a container image built outside a checkout.
# Emitting both spellings would mean two manifest keys for one fact, so
# these are renames, documented here rather than silently divergent.
_REFERENCE_KEY_RENAMES: dict[str, str] = {'sampling': 'sampling_mode', 'git_sha': 'code_sha'}


@pytest.mark.asyncio
async def test_manifest_shape_matches_reference_single_class_export(tmp_path):
    """An earlier internal version's single-class export must be
    reproducible through this generic exporter with a byte-comparable
    manifest.

    Byte-for-byte *content* necessarily differs (different dataset), so
    what is asserted is the claim that actually matters: the same manifest
    keys, carrying the same semantics, produced by the same integrity
    mechanism — the shas recomputed independently from the written files
    agree with the ones the manifest reports.
    """
    docs = [
        *[
            make_region_item(
                i,
                RegionStatus.DETECTED,
                region_bbox=[0.4, 0.4, 0.5, 0.45],
                region_cluster_id=i % 3,
            )
            for i in range(30)
        ],
        *[make_region_item(100 + i, RegionStatus.FALSE_POSITIVE) for i in range(4)],
        *[make_region_item(200 + i, RegionStatus.NO_REGION_VISIBLE) for i in range(10)],
    ]
    profile = SingleClassExportProfile(
        name='regions',
        box_source='region',
        region_class_name='region',
        dataset_kind='region_single_class',
    )
    service = single_class_service(tmp_path, FakeOpenSearch(docs, by_status=True), profile)

    result = await service.export(
        version_tag='region-v1',
        seed=42,
        empty_bg_ratio=0.1,
        max_positive_images=20,
        image_mode='whole_frame',
        img_max_side=1280,
        copy_images=False,
    )
    manifest = json.loads(Path(result.manifest_path).read_text())

    expected = {_REFERENCE_KEY_RENAMES.get(k, k) for k in _REFERENCE_MANIFEST_KEYS}
    missing = expected - set(manifest)
    assert not missing, f'manifest is missing reference keys: {sorted(missing)}'

    # Same semantics, not just the same key names.
    assert manifest['dataset_kind'] == 'region_single_class'
    assert manifest['version_tag'] == 'region-v1'
    assert manifest['class_count'] == 1
    assert manifest['class_name'] == 'region'
    assert manifest['image_mode'] == 'whole_frame'
    assert manifest['img_max_side'] == 1280
    assert manifest['empty_bg_ratio'] == 0.1
    assert manifest['max_positive_images'] == 20
    assert manifest['sampling_mode'] == 'stratified_even'
    assert manifest['dedup'] == {'enabled': False}
    assert manifest['positive_images'] == 20
    assert manifest['false_positive_background_images'] == 4
    assert manifest['background_images'] == 4 + 2  # all FPs + round(20 * 0.1)
    assert manifest['positives_zero_warning'] is False
    assert manifest['stratum_count'] == len(manifest['stratum_distribution'])
    assert manifest['image_count'] == sum(manifest['split_counts'].values())
    assert manifest['image_count'] == manifest['positive_images'] + manifest['background_images']
    assert set(manifest['split_counts']) == {'train', 'val', 'test'}
    assert manifest['code_sha']

    # The integrity mechanism itself: both shas are derivable from the
    # written files alone, which is what makes them checkable by a third
    # party who only has the export directory.
    export_dir = Path(result.export_dir)
    assert manifest['dataset_sha'] == label_content_sha(
        export_dir, class_names=manifest['class_names']
    )
    assert manifest['frozen_test_sha'] == frozen_test_sha_of(export_dir)
    assert manifest['test_label_sha'] == label_content_sha(
        export_dir, None, truncate=16, split='test'
    )
    assert manifest['dataset_sha']
    assert manifest['frozen_test_sha']
    assert manifest['test_label_sha']

    # ...and the atomically-flipped current symlink the reference export
    # also guaranteed.
    link = tmp_path / 'exports' / 'regions' / 'current'
    assert link.is_symlink()
    assert link.resolve() == export_dir.resolve()
