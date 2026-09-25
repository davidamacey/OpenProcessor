"""Tests for the generic single-class / class-subset dataset export (G2).

Fake OpenSearch (scroll + clear_scroll) over a tmp export root — no real
cluster, and ``copy_images=False`` everywhere except the one test that
deliberately exercises the crop+resize stage against real tiny JPEGs.

The acceptance test for this capability is
``test_manifest_shape_matches_reference_single_class_export`` at the
bottom: the cutover plan's §10 checklist requires the reference
implementation's single-class export to be reproducible through this
generic exporter with a byte-comparable manifest, so that test pins the
reference manifest's exact key set and asserts this exporter's manifest
carries every one of them with the same semantics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from src.clients.curation_opensearch import ClassRegistry
from src.config import CurationConfig
from src.config.region_fields import RegionFields
from src.config.region_state import RegionStatus
from src.services.curation.export_single_class import (
    SingleClassExportProfile,
    SingleClassExportService,
    frozen_test_sha_of,
    label_content_sha,
    resolve_current_single_class_dir,
)
from src.services.curation.export_single_class_rows import (
    dominant,
    reproject_into_crop,
    xyxy_to_yolo,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


class _FakeOpenSearch:
    """Returns one page of hits per distinct query, then exhausts the scroll.

    ``queries`` records every query body it was asked for, so a test can
    assert the exporter pushed the right filter into OpenSearch rather
    than scrolling everything and filtering in Python.
    """

    def __init__(self, docs: list[dict[str, Any]], *, by_status: bool = False) -> None:
        self._docs = docs
        self._by_status = by_status
        self.queries: list[dict[str, Any]] = []
        self._served: set[str] = set()

    async def search(self, index: str, body: dict[str, Any], scroll: str | None = None) -> dict:  # noqa: ARG002
        self.queries.append(body['query'])
        docs = self._docs if not self._by_status else self._match(body['query'])
        scroll_id = f'scroll-{len(self.queries)}'
        self._served.add(scroll_id)
        hits = [{'_id': d.get('crop_id', ''), '_source': d} for d in docs]
        return {'_scroll_id': scroll_id, 'hits': {'hits': hits}}

    def _match(self, query: dict[str, Any]) -> list[dict[str, Any]]:
        """Crude status-terms matcher, enough to split the three region pools.

        F-29: the empty-frame sample query is wrapped in ``function_score``
        (random_score) rather than a bare bool — unwrap it first.
        """
        if 'function_score' in query:
            query = query['function_score']['query']
        # F-19: status/class_id predicates now live in filter context.
        clauses = query.get('bool', {}).get('filter', [])
        wanted: set[str] = set()
        for clause in clauses:
            for field_name, values in clause.get('terms', {}).items():
                if field_name == _F.status:
                    wanted.update(values)
        if not wanted:
            return self._docs
        return [d for d in self._docs if d.get(_F.status) in wanted]

    async def scroll(self, scroll_id: str, scroll: str) -> dict:  # noqa: ARG002
        return {'_scroll_id': scroll_id, 'hits': {'hits': []}}

    async def clear_scroll(self, scroll_id: str) -> None:  # noqa: ARG002
        return None


_F = RegionFields()


def _registry(base: Path, names: list[str]) -> ClassRegistry:
    reg = ClassRegistry(path=base / 'class_registry.json')
    for name in names:
        reg.add_class(name)
    return reg


def _service(
    tmp_path: Path,
    opensearch: _FakeOpenSearch,
    profile: SingleClassExportProfile,
    *,
    names: list[str] | None = None,
    source_root: Path | None = None,
) -> SingleClassExportService:
    cfg = CurationConfig(
        export_root=tmp_path / 'exports',
        source_root=source_root or (tmp_path / 'images'),
    )
    return SingleClassExportService(
        opensearch,
        profile=profile,
        config=cfg,
        registry=_registry(tmp_path / 'registry', names or ['car', 'truck', 'bus']),
        region_fields=_F,
    )


def _item(
    idx: int,
    class_id: int,
    *,
    bbox: list[float] | None = None,
    test_holdout: bool = False,
    cluster_id: int | None = None,
) -> dict[str, Any]:
    return {
        'crop_id': f'crop-{idx}',
        'image_id': f'img-{idx}',
        'image_path': f'{idx}.jpg',
        'bbox_norm': bbox or [0.1, 0.1, 0.5, 0.5],
        'class_id': class_id,
        'test_holdout': test_holdout,
        'cluster_id': cluster_id,
    }


def _region_item(
    idx: int,
    status: RegionStatus,
    *,
    region_bbox: list[float] | None = None,
    class_id: int = 0,
    test_holdout: bool = False,
    region_cluster_id: int | None = None,
) -> dict[str, Any]:
    doc = _item(idx, class_id, test_holdout=test_holdout)
    doc[_F.status] = status.value
    doc[_F.bbox_norm] = region_bbox
    doc[_F.cluster_id] = region_cluster_id
    return doc


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
    service = _service(tmp_path, _FakeOpenSearch([]), SingleClassExportProfile())
    with pytest.raises(ValueError, match='class_ids'):
        await service.export()


@pytest.mark.asyncio
async def test_item_crop_mode_rejected_for_item_box_source(tmp_path):
    profile = SingleClassExportProfile(class_ids=(0,))
    service = _service(tmp_path, _FakeOpenSearch([]), profile)
    with pytest.raises(ValueError, match='item_crop'):
        await service.export(image_mode='item_crop')


@pytest.mark.asyncio
async def test_unknown_image_mode_rejected(tmp_path):
    profile = SingleClassExportProfile(class_ids=(0,))
    service = _service(tmp_path, _FakeOpenSearch([]), profile)
    with pytest.raises(ValueError, match='image_mode'):
        await service.export(image_mode='sideways')  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Item box source: single class + class subset
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_single_class_export_writes_labels_yaml_and_symlink(tmp_path):
    docs = [_item(i, 0) for i in range(6)] + [_item(100 + i, 1) for i in range(4)]
    profile = SingleClassExportProfile(class_ids=(0,), name='regions')
    service = _service(tmp_path, _FakeOpenSearch(docs), profile)

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
    docs = [_item(0, 2), _item(1, 0), _item(2, 1)]
    # Deliberately non-ascending: the profile's order IS the dataset order.
    profile = SingleClassExportProfile(class_ids=(2, 0))
    service = _service(tmp_path, _FakeOpenSearch(docs), profile)

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
    docs = [_item(i, 0) for i in range(10)] + [_item(100 + i, 1) for i in range(10)]
    profile = SingleClassExportProfile(class_ids=(0,))
    service = _service(tmp_path, _FakeOpenSearch(docs), profile)

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
    docs = [_item(i, 0) for i in range(50)] + [_item(100 + i, 1) for i in range(2)]
    profile = SingleClassExportProfile(class_ids=(0, 1))
    service = _service(tmp_path, _FakeOpenSearch(docs), profile)

    result = await service.export(max_positive_images=6, seed=5, copy_images=False)

    assert result.positive_images == 6
    manifest = json.loads(Path(result.manifest_path).read_text())
    assert manifest['sampling_mode'] == 'stratified_even'
    # Both strata present: an even round-robin, not a flat truncation.
    assert set(manifest['stratum_distribution']) == {'pos:0', 'pos:1'}


@pytest.mark.asyncio
async def test_test_holdout_frames_are_pinned_to_the_test_split(tmp_path):
    docs = [_item(i, 0) for i in range(9)]
    docs.append(_item(99, 0, test_holdout=True))
    profile = SingleClassExportProfile(class_ids=(0,))
    service = _service(tmp_path, _FakeOpenSearch(docs), profile)

    result = await service.export(seed=1, copy_images=False)

    assert (Path(result.export_dir) / 'labels' / 'test' / 'img-99.txt').is_file()


@pytest.mark.asyncio
async def test_skip_test_split_emits_only_train_and_val(tmp_path):
    docs = [_item(i, 0) for i in range(20)]
    docs.append(_item(99, 0, test_holdout=True))
    profile = SingleClassExportProfile(class_ids=(0,))
    service = _service(tmp_path, _FakeOpenSearch(docs), profile)

    result = await service.export(seed=1, skip_test_split=True, copy_images=False)

    assert result.split_counts.test == 0
    # No test split means no frozen-test guarantee to make.
    assert result.frozen_test_sha == ''


@pytest.mark.asyncio
async def test_export_is_reproducible_for_the_same_seed(tmp_path):
    docs = [_item(i, 0) for i in range(12)] + [_item(100 + i, 1) for i in range(6)]
    profile = SingleClassExportProfile(class_ids=(0,))

    shas = []
    for run in range(2):
        service = _service(tmp_path / f'run{run}', _FakeOpenSearch(docs), profile)
        result = await service.export(seed=42, empty_bg_ratio=0.5, copy_images=False)
        shas.append((result.dataset_sha, result.frozen_test_sha))

    assert shas[0] == shas[1]
    assert all(sha for pair in shas for sha in pair)


@pytest.mark.asyncio
async def test_zero_positive_export_is_flagged_not_silent(tmp_path):
    """A background-only dataset can't train a detector; the manifest has to
    say so rather than looking like an ordinary small export."""
    docs = [_item(i, 1) for i in range(4)]
    profile = SingleClassExportProfile(class_ids=(0,))
    service = _service(tmp_path, _FakeOpenSearch(docs), profile)

    result = await service.export(copy_images=False)

    assert result.positive_images == 0
    manifest = json.loads(Path(result.manifest_path).read_text())
    assert manifest['positives_zero_warning'] is True


@pytest.mark.asyncio
async def test_symlink_flip_retargets_a_previous_export(tmp_path):
    docs = [_item(i, 0) for i in range(4)]
    profile = SingleClassExportProfile(class_ids=(0,))
    service = _service(tmp_path, _FakeOpenSearch(docs), profile)

    first = await service.export(export_dir=tmp_path / 'e1', copy_images=False)
    second = await service.export(export_dir=tmp_path / 'e2', copy_images=False)

    link = tmp_path / 'exports' / 'single_class' / 'current'
    assert link.resolve() == Path(second.export_dir).resolve()
    assert link.resolve() != Path(first.export_dir).resolve()


# ---------------------------------------------------------------------------
# Region box source (the reference implementation's export shape)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_region_mode_splits_positives_hard_negatives_and_empties(tmp_path):
    docs = [
        *[
            _region_item(i, RegionStatus.DETECTED, region_bbox=[0.4, 0.4, 0.5, 0.45])
            for i in range(10)
        ],
        *[_region_item(100 + i, RegionStatus.FALSE_POSITIVE) for i in range(3)],
        *[_region_item(200 + i, RegionStatus.NO_REGION_VISIBLE) for i in range(20)],
    ]
    profile = SingleClassExportProfile(box_source='region', region_class_name='region')
    service = _service(tmp_path, _FakeOpenSearch(docs, by_status=True), profile)

    result = await service.export(empty_bg_ratio=0.2, seed=2, copy_images=False)

    assert result.positive_images == 10
    # Every human false-positive is kept in full; the empty pool is sampled
    # to empty_bg_ratio * positives on top of that.
    assert result.background_images == 3 + 2
    manifest = json.loads(Path(result.manifest_path).read_text())
    assert manifest['false_positive_background_images'] == 3
    assert manifest['class_name'] == 'region'
    assert '  0: region\n' in (Path(result.export_dir) / 'data.yaml').read_text()


@pytest.mark.asyncio
async def test_region_mode_empty_frame_sample_uses_random_score_with_fixed_seed(tmp_path):
    """F-29: the empty-frame sample must not be index-order-biased (the
    same leading docs every export) — it's a function_score/random_score
    query with a fixed seed instead."""
    from src.services.curation.export_single_class_rows import EMPTY_FRAME_SAMPLE_SEED

    docs = [
        _region_item(i, RegionStatus.DETECTED, region_bbox=[0.4, 0.4, 0.5, 0.45]) for i in range(4)
    ]
    profile = SingleClassExportProfile(box_source='region', region_class_name='region')
    fake_os = _FakeOpenSearch(docs, by_status=True)
    service = _service(tmp_path, fake_os, profile)

    await service.export(empty_bg_ratio=0.5, seed=1, copy_images=False)

    empty_queries = [
        q
        for q in fake_os.queries
        if 'function_score' in q and 'random_score' in q['function_score']
    ]
    assert empty_queries, 'expected at least one random_score-wrapped empty-frame query'
    rs = empty_queries[0]['function_score']['random_score']
    assert rs['seed'] == EMPTY_FRAME_SAMPLE_SEED


@pytest.mark.asyncio
async def test_region_mode_ignores_verify_rejected(tmp_path):
    """A model-rejected region is an unverified non-detection — training on
    it teaches the next detector the current one's mistakes."""
    docs = [
        _region_item(0, RegionStatus.DETECTED, region_bbox=[0.4, 0.4, 0.5, 0.45]),
        _region_item(1, RegionStatus.VERIFY_REJECTED),
    ]
    profile = SingleClassExportProfile(box_source='region')
    service = _service(tmp_path, _FakeOpenSearch(docs, by_status=True), profile)

    result = await service.export(empty_bg_ratio=1.0, copy_images=False)

    assert result.image_count == 1
    assert result.positive_images == 1


@pytest.mark.asyncio
async def test_region_mode_filters_by_parent_class_ids(tmp_path):
    docs = [_region_item(0, RegionStatus.DETECTED, region_bbox=[0.4, 0.4, 0.5, 0.45])]
    fake = _FakeOpenSearch(docs, by_status=True)
    profile = SingleClassExportProfile(box_source='region', class_ids=(0, 2))
    service = _service(tmp_path, fake, profile)

    await service.export(empty_bg_ratio=0.0, copy_images=False)

    # The parent-class narrowing is pushed into OpenSearch, not applied
    # after scrolling the whole index.
    filt = fake.queries[0]['bool']['filter']
    assert {'terms': {'class_id': [0, 2]}} in filt


@pytest.mark.asyncio
async def test_item_crop_mode_reprojects_the_region_into_the_parent_crop(tmp_path):
    parent = [0.25, 0.25, 0.75, 0.75]
    docs = [
        {
            **_region_item(0, RegionStatus.DETECTED, region_bbox=[0.5, 0.25, 0.75, 0.75]),
            'bbox_norm': parent,
        }
    ]
    profile = SingleClassExportProfile(box_source='region')
    service = _service(tmp_path, _FakeOpenSearch(docs, by_status=True), profile)

    result = await service.export(image_mode='item_crop', empty_bg_ratio=0.0, copy_images=False)

    # One row per ITEM in crop mode, keyed on crop_id rather than image_id.
    label = next((Path(result.export_dir) / 'labels').rglob('crop-0.txt'))
    cid, cx, cy, w, h = label.read_text().split()
    assert cid == '0'
    assert (float(cx), float(cy), float(w), float(h)) == pytest.approx((0.75, 0.5, 0.5, 1.0))


@pytest.mark.asyncio
async def test_item_crop_mode_crops_the_written_image(tmp_path):
    source_root = tmp_path / 'images'
    source_root.mkdir()
    Image.new('RGB', (400, 400), (10, 20, 30)).save(source_root / '0.jpg')

    parent = [0.0, 0.0, 0.5, 0.25]  # 200x100 px of the 400x400 source
    docs = [
        {
            **_region_item(0, RegionStatus.DETECTED, region_bbox=[0.1, 0.05, 0.2, 0.1]),
            'bbox_norm': parent,
        }
    ]
    profile = SingleClassExportProfile(box_source='region')
    service = _service(
        tmp_path, _FakeOpenSearch(docs, by_status=True), profile, source_root=source_root
    )

    result = await service.export(
        image_mode='item_crop', empty_bg_ratio=0.0, img_max_side=640, copy_images=True
    )

    manifest = json.loads(Path(result.manifest_path).read_text())
    assert manifest['image_copy'] == {'attempted': 1, 'copied': 1, 'failed': 0, 'errors': []}
    written = next((Path(result.export_dir) / 'images').rglob('crop-0.jpg'))
    # Cropped to the parent region, and not upscaled past the source size.
    assert Image.open(written).size == (200, 100)


# ---------------------------------------------------------------------------
# Acceptance: manifest parity with the reference single-class export
# ---------------------------------------------------------------------------

# Exact manifest key set written by the reference implementation's
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
    """Cutover plan §10: the reference single-class export must be
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
            _region_item(
                i,
                RegionStatus.DETECTED,
                region_bbox=[0.4, 0.4, 0.5, 0.45],
                region_cluster_id=i % 3,
            )
            for i in range(30)
        ],
        *[_region_item(100 + i, RegionStatus.FALSE_POSITIVE) for i in range(4)],
        *[_region_item(200 + i, RegionStatus.NO_REGION_VISIBLE) for i in range(10)],
    ]
    profile = SingleClassExportProfile(
        name='regions',
        box_source='region',
        region_class_name='region',
        dataset_kind='region_single_class',
    )
    service = _service(tmp_path, _FakeOpenSearch(docs, by_status=True), profile)

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
    assert manifest['dataset_sha'] == label_content_sha(export_dir)
    assert manifest['frozen_test_sha'] == frozen_test_sha_of(export_dir)
    assert manifest['dataset_sha']
    assert manifest['frozen_test_sha']

    # ...and the atomically-flipped current symlink the reference export
    # also guaranteed.
    link = tmp_path / 'exports' / 'regions' / 'current'
    assert link.is_symlink()
    assert link.resolve() == export_dir.resolve()
