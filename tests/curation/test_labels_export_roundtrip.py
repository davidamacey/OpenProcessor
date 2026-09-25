"""Label paths -> ``/export/yolo`` round trip (no labels-confirmed write-through hole).

A reference deployment this stack descends from had an exporter that read
only the confirmed-labels index while the labeler and auto-promote wrote only
``class_validated`` onto items — so an export after hours of labeling came
out with zero rows. On this codebase the contract is the other way round:

- every labeling path (single / batch human label, cluster move, cluster
  auto-promote, YOLO label import) sets ``class_validated=true`` on the item;
- :class:`~src.services.curation.export.GenericYoloExportService` selects
  ``class_validated=true`` items straight from the items index and never
  reads the confirmed-labels index, which only ``label_import`` writes (as a
  provenance ledger of imported ground truth).

These tests run each real write path against one query-evaluating in-memory
OpenSearch and then run the real exporter over the result, so they fail if
any label path stops marking items validated, or if the exporter is ever
switched to a source that only one of those paths populates.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from curation.query_fakes import QueryFakeOpenSearch
from src.clients.curation_opensearch import ClassRegistry
from src.config import get_curation_config
from src.services.curation.cluster_ids import RESIDUAL_CLUSTER_ID_OFFSET


CFG = get_curation_config()
ITEMS = CFG.items_index
IMAGES = CFG.images_index
CONFIRMED = CFG.labels_confirmed_index


def _item(doc_id: str, *, image: str, cluster_id: int = 7, **extra: Any) -> dict[str, Any]:
    return {
        'crop_id': doc_id,
        'image_id': image,
        'image_path': f'{image}.jpg',
        'bbox_norm': [0.1, 0.1, 0.5, 0.5],
        'class_id': 0,
        'class_name': 'widget',
        'class_source': 'detector',
        'class_validated': False,
        'cluster_id': cluster_id,
        **extra,
    }


@pytest.fixture
def registry(tmp_path: Path) -> ClassRegistry:
    reg = ClassRegistry(path=tmp_path / 'class_registry.json')
    reg.add_class('widget')
    reg.add_class('gadget')
    return reg


@pytest.fixture
def fake() -> QueryFakeOpenSearch:
    items = {
        'h1': _item('h1', image='img1'),
        'b1': _item('b1', image='img2'),
        'b2': _item('b2', image='img3'),
        'm1': _item('m1', image='img4'),
        'never': _item('never', image='img5'),
    }
    # One high-purity CANDIDATE cluster (cluster_id >= the residual
    # offset -- auto-promote excludes class-range clusters, cluster_id ==
    # class_id, from auto-promote entirely since their purity is 1.0 by
    # construction): 4 members all predicted 'gadget' by the detector ->
    # auto-promote validates them.
    for i in range(4):
        items[f'ap{i}'] = _item(
            f'ap{i}',
            image=f'ap_img{i}',
            cluster_id=RESIDUAL_CLUSTER_ID_OFFSET + 42,
            class_id=1,
            class_name='gadget',
            class_source='classifier_model',
        )
    images = {'imp': {'image_id': 'imp', 'image_path': '/data/imp.jpg'}}
    return QueryFakeOpenSearch({ITEMS: items, IMAGES: images})


async def _export(fake: QueryFakeOpenSearch, registry: ClassRegistry, tmp_path: Path):
    from src.config import CurationConfig
    from src.services.curation.export import GenericYoloExportService

    cfg = CurationConfig(
        items_index=ITEMS,
        labels_confirmed_index=CONFIRMED,
        export_root=tmp_path / 'exports',
    )
    fake.searched_indexes.clear()
    service = GenericYoloExportService(fake, config=cfg, registry=registry)
    return await service.export_dataset(version_tag='roundtrip', copy_images=False)


async def _human_label_paths(fake, registry, monkeypatch) -> None:
    from src.routers.curation import crops
    from src.routers.curation._common import (
        CropBatchLabelRequest,
        CropLabelRequest,
        CropMoveRequest,
    )

    monkeypatch.setattr(crops, 'get_class_registry', lambda: registry)
    await crops.label_crop('h1', CropLabelRequest(class_id=1), fake, registry)
    await crops.batch_label_crops(CropBatchLabelRequest(crop_ids=['b1', 'b2'], class_id=1), fake)
    await crops.move_crops(CropMoveRequest(crop_ids=['m1'], cluster_id=0), fake)


@pytest.mark.asyncio
async def test_human_labels_alone_produce_a_non_empty_export(fake, registry, tmp_path, monkeypatch):
    """The exact scenario of the reference hole: humans label, nothing is
    imported, the confirmed-labels index stays empty — export must not."""
    await _human_label_paths(fake, registry, monkeypatch)

    assert fake.docs(CONFIRMED) == {}
    result = await _export(fake, registry, tmp_path)

    assert result.image_count == 4  # h1, b1, b2, m1
    assert CONFIRMED not in fake.searched_indexes
    labels = sorted(Path(result.export_dir).glob('labels/*/*.txt'))
    assert len(labels) == 4


@pytest.mark.asyncio
@pytest.mark.usefixtures('reference_ingest_profiles')
async def test_every_label_path_reaches_the_export(fake, registry, tmp_path, monkeypatch):
    from src.services.curation.clustering.orchestrator import auto_promote_clusters
    from src.services.curation.label_import import import_yolo_labels

    await _human_label_paths(fake, registry, monkeypatch)

    promoted = await auto_promote_clusters(fake, min_purity=0.85, min_members=4)
    assert promoted['promoted'] == 4

    txt = tmp_path / 'imp.txt'
    txt.write_text('1 0.7 0.7 0.2 0.2\n')
    assert await import_yolo_labels(Path('/data/imp.jpg'), txt, registry, fake) == 1

    items = fake.docs(ITEMS)
    validated = {i for i, d in items.items() if d.get('class_validated') is True}
    assert validated == {'h1', 'b1', 'b2', 'm1', 'ap0', 'ap1', 'ap2', 'ap3'} | {
        i for i, d in items.items() if d.get('image_id') == 'imp'
    }
    assert 'never' not in validated
    # the confirmed-labels index is a ledger of imported labels only
    assert len(fake.docs(CONFIRMED)) == 1

    result = await _export(fake, registry, tmp_path)
    assert result.image_count == len(validated) == 9
    assert CONFIRMED not in fake.searched_indexes
    assert set(fake.searched_indexes) == {ITEMS}


@pytest.mark.asyncio
async def test_unvalidated_items_are_not_exported(fake, registry, tmp_path):
    # DQ-M9: with no validated item there is nothing to export — refused,
    # rather than an empty dataset flipped to `current`.
    from src.services.curation.export_readiness import NothingToExportError

    with pytest.raises(NothingToExportError):
        await _export(fake, registry, tmp_path)
