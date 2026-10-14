"""Single-class export in region mode and item-crop mode."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from curation.export_fakes import FakeOpenSearch, make_region_item, single_class_service
from src.config.region_state import RegionStatus
from src.services.curation.export_single_class import SingleClassExportProfile


# ---------------------------------------------------------------------------
# Region box source (an earlier internal version's export shape)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_region_mode_splits_positives_hard_negatives_and_empties(tmp_path):
    docs = [
        *[
            make_region_item(i, RegionStatus.DETECTED, region_bbox=[0.4, 0.4, 0.5, 0.45])
            for i in range(10)
        ],
        *[make_region_item(100 + i, RegionStatus.FALSE_POSITIVE) for i in range(3)],
        *[make_region_item(200 + i, RegionStatus.NO_REGION_VISIBLE) for i in range(20)],
    ]
    profile = SingleClassExportProfile(box_source='region', region_class_name='region')
    service = single_class_service(tmp_path, FakeOpenSearch(docs, by_status=True), profile)

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
    """The empty-frame sample must not be index-order-biased (the
    same leading docs every export) — it's a function_score/random_score
    query with a fixed seed instead."""
    from src.services.curation.export_single_class_rows import EMPTY_FRAME_SAMPLE_SEED

    docs = [
        make_region_item(i, RegionStatus.DETECTED, region_bbox=[0.4, 0.4, 0.5, 0.45])
        for i in range(4)
    ]
    profile = SingleClassExportProfile(box_source='region', region_class_name='region')
    fake_os = FakeOpenSearch(docs, by_status=True)
    service = single_class_service(tmp_path, fake_os, profile)

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
        make_region_item(0, RegionStatus.DETECTED, region_bbox=[0.4, 0.4, 0.5, 0.45]),
        make_region_item(1, RegionStatus.VERIFY_REJECTED),
    ]
    profile = SingleClassExportProfile(box_source='region')
    service = single_class_service(tmp_path, FakeOpenSearch(docs, by_status=True), profile)

    result = await service.export(empty_bg_ratio=1.0, copy_images=False)

    assert result.image_count == 1
    assert result.positive_images == 1


@pytest.mark.asyncio
async def test_region_mode_filters_by_parent_class_ids(tmp_path):
    docs = [make_region_item(0, RegionStatus.DETECTED, region_bbox=[0.4, 0.4, 0.5, 0.45])]
    fake = FakeOpenSearch(docs, by_status=True)
    profile = SingleClassExportProfile(box_source='region', class_ids=(0, 2))
    service = single_class_service(tmp_path, fake, profile)

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
            **make_region_item(0, RegionStatus.DETECTED, region_bbox=[0.5, 0.25, 0.75, 0.75]),
            'bbox_norm': parent,
        }
    ]
    profile = SingleClassExportProfile(box_source='region')
    service = single_class_service(tmp_path, FakeOpenSearch(docs, by_status=True), profile)

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
            **make_region_item(0, RegionStatus.DETECTED, region_bbox=[0.1, 0.05, 0.2, 0.1]),
            'bbox_norm': parent,
        }
    ]
    profile = SingleClassExportProfile(box_source='region')
    service = single_class_service(
        tmp_path, FakeOpenSearch(docs, by_status=True), profile, source_root=source_root
    )

    result = await service.export(
        image_mode='item_crop', empty_bg_ratio=0.0, img_max_side=640, copy_images=True
    )

    manifest = json.loads(Path(result.manifest_path).read_text())
    assert manifest['image_copy'] == {'attempted': 1, 'copied': 1, 'failed': 0, 'errors': []}
    written = next((Path(result.export_dir) / 'images').rglob('crop-0.jpg'))
    # Cropped to the parent region, and not upscaled past the source size.
    assert Image.open(written).size == (200, 100)
