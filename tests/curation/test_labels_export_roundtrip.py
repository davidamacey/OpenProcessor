"""Label paths -> ``/export/yolo`` round trip (no labels-confirmed write-through hole).

A reference deployment this stack descends from had an exporter that read
only the confirmed-labels index while the labeler and auto-promote wrote only
``class_validated`` onto items — so an export after hours of labeling came
out with zero rows. On this codebase the contract is the other way round:

- every labeling path (single / batch human label, cluster move, cluster
  auto-promote, dataset import) sets ``class_validated=true`` on the item;
- :class:`~src.services.curation.export.GenericYoloExportService` selects
  ``class_validated=true`` items straight from the items index and never
  reads the confirmed-labels index, which nothing writes anymore (W10:
  dataset import's own ledger is the provenance record now).

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
from src.config.curation import base_curation_config
from src.services.curation.cluster_ids import RESIDUAL_CLUSTER_ID_OFFSET


CFG = base_curation_config()
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
    """The fifth "every label path" writer, post-W10: dataset import.

    Ported from the pre-W10 version of this test, which used the since-
    deleted ``label_import.import_yolo_labels`` (a parallel write path
    with no OCC, no restorable snapshot — any_domain_plan.md W10.1 I2).
    Dataset import now goes through the SAME ``class_label_update()`` /
    ``occ_upsert_bulk()`` primitive every other writer exercised by this
    test uses.
    """
    from curation.dataset_import.harness import Harness, write_yolo
    from src.services.curation.clustering.auto_promote import auto_promote_clusters
    from src.services.curation.dataset_import.mapping import ClassMappingEntry
    from src.services.curation.ingest import CurationIngestService

    await _human_label_paths(fake, registry, monkeypatch)

    promoted = await auto_promote_clusters(fake, min_purity=0.85, min_members=4)
    assert promoted['promoted'] == 4

    # The production importer, pointed at this test's index and registry.
    h = Harness(tmp_path, monkeypatch)
    h.os = fake
    h.registry = registry
    h.service = CurationIngestService(
        opensearch=fake,
        triton_pool=h.triton,
        registry=registry,
        profile=h.profile,
        pe_encoder=h.pe,
        config=h.cfg,
    )
    ds_root = tmp_path / 'one_image_ds'
    write_yolo(ds_root, names=['widget', 'gadget'], images={'imp': ['1 0.7 0.7 0.2 0.2']})
    store, _ = await h.run(
        h.request(ds_root, [ClassMappingEntry(dataset_class='gadget', action='map', class_id=1)])
    )
    assert store.job.read()['status'] == 'completed'
    import_id = store.import_id

    items = fake.docs(ITEMS)
    imported_ids = {i for i, d in items.items() if d.get('import_ids') == [import_id]}
    assert len(imported_ids) == 1
    validated = {i for i, d in items.items() if d.get('class_validated') is True}
    assert validated == {'h1', 'b1', 'b2', 'm1', 'ap0', 'ap1', 'ap2', 'ap3'} | imported_ids
    assert 'never' not in validated
    # W10 (any_domain_plan.md I13): dataset import no longer writes the
    # labels_confirmed ledger at all -- the import's own ledger (not
    # built this pass; see the wave report) is the record now.
    assert fake.docs(CONFIRMED) == {}

    result = await _export(fake, registry, tmp_path)
    assert result.image_count == len(validated) == 9
    assert CONFIRMED not in fake.searched_indexes
    # Items, plus the images index the imported-negative-frame scan reads.
    assert set(fake.searched_indexes) == {ITEMS, IMAGES}


@pytest.mark.asyncio
async def test_unvalidated_items_are_not_exported(fake, registry, tmp_path):
    # DQ-M9: with no validated item there is nothing to export — refused,
    # rather than an empty dataset flipped to `current`.
    from src.services.curation.export_readiness import NothingToExportError

    with pytest.raises(NothingToExportError):
        await _export(fake, registry, tmp_path)
