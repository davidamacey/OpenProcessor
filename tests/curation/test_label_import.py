"""Tests for src.services.curation.label_import."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.clients.curation_opensearch import ClassRegistry
from src.config import get_curation_config
from src.services.curation.ingest import _crop_id as ingest_crop_id
from src.services.curation.label_import import (
    _crop_id_for,
    _parse_yolo_txt,
    import_labels_batch,
    import_yolo_labels,
)


class FakeLabelOpenSearch:
    """AsyncOpenSearch double for the label-import lookup + bulk surface.

    ``images``: ``{image_id: doc}`` (doc must carry ``image_path``).
    ``items``: ``{crop_id: doc}`` (doc must carry ``image_id``, ``bbox_norm``).
    """

    def __init__(
        self,
        *,
        images: dict[str, dict[str, Any]] | None = None,
        items: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.images = dict(images or {})
        self.items = dict(items or {})
        self.bulk_calls: list[list[dict[str, Any]]] = []

    async def search(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:
        cfg = get_curation_config()
        query = body.get('query') or {}
        if index == cfg.images_index:
            path = (query.get('term') or {}).get('image_path')
            hits = [
                {'_id': doc_id, '_source': doc}
                for doc_id, doc in self.images.items()
                if doc.get('image_path') == path
            ]
            return {'hits': {'hits': hits[:1]}}
        # items index: bool must image_id term, must_not test_holdout.
        must = (query.get('bool') or {}).get('must') or []
        image_id = None
        for clause in must:
            term = clause.get('term') or {}
            if 'image_id' in term:
                image_id = term['image_id']
        hits = [
            {'_id': doc_id, '_source': doc}
            for doc_id, doc in self.items.items()
            if doc.get('image_id') == image_id and not doc.get('test_holdout')
        ]
        return {'hits': {'hits': hits}}

    async def bulk(
        self,
        *,
        body: list[dict[str, Any]],
        refresh: bool | str = False,  # noqa: ARG002
    ) -> dict[str, Any]:
        self.bulk_calls.append(body)
        cfg = get_curation_config()
        items: list[dict[str, Any]] = []
        for action, doc in zip(body[0::2], body[1::2], strict=True):
            if 'index' in action:
                meta = action['index']
                store = self.items if meta['_index'] == cfg.items_index else {}
                store[meta['_id']] = doc
                items.append({'index': {'_id': meta['_id'], 'status': 201}})
            elif 'update' in action:
                meta = action['update']
                target = self.items.setdefault(meta['_id'], {})
                target.update(doc['doc'])
                items.append({'update': {'_id': meta['_id'], 'status': 200}})
        return {'errors': False, 'items': items}


@pytest.fixture
def registry(tmp_path: Path) -> ClassRegistry:
    reg = ClassRegistry(path=tmp_path / 'class_registry.json')
    reg.add_class('widget')  # class_id 0
    reg.add_class('gadget')  # class_id 1
    return reg


class TestParseYoloTxt:
    def test_parses_valid_rows(self, tmp_path: Path, registry: ClassRegistry) -> None:
        txt = tmp_path / 'a.txt'
        txt.write_text('0 0.5 0.5 0.2 0.2\n1 0.1 0.1 0.05 0.05\n')
        parsed = _parse_yolo_txt(txt, registry)
        assert len(parsed) == 2
        cls_id, bbox = parsed[0]
        assert cls_id == 0
        assert bbox == pytest.approx([0.4, 0.4, 0.6, 0.6])

    def test_rejects_out_of_range_class(self, tmp_path: Path, registry: ClassRegistry) -> None:
        txt = tmp_path / 'a.txt'
        txt.write_text('99 0.5 0.5 0.2 0.2\n')
        assert _parse_yolo_txt(txt, registry) == []

    def test_rejects_deprecated_class(self, tmp_path: Path, registry: ClassRegistry) -> None:
        registry.merge_class(source_id=1, target_id=0)
        txt = tmp_path / 'a.txt'
        txt.write_text('1 0.5 0.5 0.2 0.2\n')
        assert _parse_yolo_txt(txt, registry) == []

    def test_skips_malformed_lines(self, tmp_path: Path, registry: ClassRegistry) -> None:
        txt = tmp_path / 'a.txt'
        txt.write_text('not a valid row\n0 0.5 0.5 0.2 0.2\n# a comment\n\n')
        parsed = _parse_yolo_txt(txt, registry)
        assert len(parsed) == 1

    def test_missing_file_returns_empty(self, tmp_path: Path, registry: ClassRegistry) -> None:
        assert _parse_yolo_txt(tmp_path / 'missing.txt', registry) == []


class TestCropIdParity:
    def test_crop_id_for_matches_ingest_crop_id(self) -> None:
        bbox = [0.1, 0.2, 0.3, 0.4]
        assert _crop_id_for('image123', bbox) == ingest_crop_id('image123', bbox)


class TestImportYoloLabels:
    @pytest.mark.asyncio
    async def test_no_image_doc_is_noop(self, tmp_path: Path, registry: ClassRegistry) -> None:
        os_fake = FakeLabelOpenSearch()
        txt = tmp_path / 'a.txt'
        txt.write_text('0 0.5 0.5 0.2 0.2\n')
        n = await import_yolo_labels(Path('/tmp/missing.jpg'), txt, registry, os_fake)
        assert n == 0

    @pytest.mark.asyncio
    async def test_no_iou_match_creates_new_item(
        self, tmp_path: Path, registry: ClassRegistry
    ) -> None:
        os_fake = FakeLabelOpenSearch(
            images={'img1': {'image_id': 'img1', 'image_path': '/tmp/a.jpg'}}
        )
        txt = tmp_path / 'a.txt'
        txt.write_text('0 0.5 0.5 0.2 0.2\n')
        n = await import_yolo_labels(Path('/tmp/a.jpg'), txt, registry, os_fake)
        assert n == 1
        [item] = list(os_fake.items.values())
        assert item['class_id'] == 0
        assert item['class_name'] == 'widget'
        assert item['class_validated'] is True
        assert item['label_source'] == 'external_label'
        assert item['class_detector'] == 'external_label'
        assert item['class_labeler'] == 'label_import'

    @pytest.mark.asyncio
    async def test_iou_match_flips_existing_item_validated(
        self, tmp_path: Path, registry: ClassRegistry
    ) -> None:
        existing_crop_id = _crop_id_for('img1', [0.41, 0.41, 0.59, 0.59])
        os_fake = FakeLabelOpenSearch(
            images={'img1': {'image_id': 'img1', 'image_path': '/tmp/a.jpg'}},
            items={
                existing_crop_id: {
                    'crop_id': existing_crop_id,
                    'image_id': 'img1',
                    'bbox_norm': [0.41, 0.41, 0.59, 0.59],
                    'class_validated': False,
                    'class_detector': 'some_detector',
                    'class_labeler': 'ingest',
                }
            },
        )
        txt = tmp_path / 'a.txt'
        txt.write_text('1 0.5 0.5 0.2 0.2\n')  # bbox_norm ~ [0.4, 0.4, 0.6, 0.6], IoU high
        n = await import_yolo_labels(Path('/tmp/a.jpg'), txt, registry, os_fake)
        assert n == 1
        assert os_fake.items[existing_crop_id]['class_validated'] is True
        assert os_fake.items[existing_crop_id]['class_id'] == 1
        assert os_fake.items[existing_crop_id]['class_name'] == 'gadget'
        # The label replaced the detector's class; provenance must follow.
        assert os_fake.items[existing_crop_id]['class_detector'] == 'external_label'
        assert os_fake.items[existing_crop_id]['class_labeler'] == 'label_import'

    @pytest.mark.asyncio
    async def test_label_source_is_stamped(self, tmp_path: Path, registry: ClassRegistry) -> None:
        os_fake = FakeLabelOpenSearch(
            images={'img1': {'image_id': 'img1', 'image_path': '/tmp/a.jpg'}}
        )
        txt = tmp_path / 'a.txt'
        txt.write_text('0 0.5 0.5 0.2 0.2\n')
        await import_yolo_labels(
            Path('/tmp/a.jpg'), txt, registry, os_fake, label_source='human_review'
        )
        [item] = list(os_fake.items.values())
        assert item['class_source'] == 'human_review'
        assert item['label_source'] == 'human_review'


class TestImportLabelsBatch:
    @pytest.mark.asyncio
    async def test_batch_summary_counts(self, tmp_path: Path, registry: ClassRegistry) -> None:
        os_fake = FakeLabelOpenSearch(
            images={
                'img1': {'image_id': 'img1', 'image_path': '/tmp/a.jpg'},
                'img2': {'image_id': 'img2', 'image_path': '/tmp/b.jpg'},
            }
        )
        txt_a = tmp_path / 'a.txt'
        txt_a.write_text('0 0.5 0.5 0.2 0.2\n')
        txt_b = tmp_path / 'b.txt'
        txt_b.write_text('1 0.2 0.2 0.1 0.1\n1 0.7 0.7 0.1 0.1\n')
        # A path with no matching image doc -> files_processed still counts,
        # 0 labels imported for it.
        txt_missing = tmp_path / 'missing.txt'
        txt_missing.write_text('0 0.5 0.5 0.2 0.2\n')

        summary = await import_labels_batch(
            [
                (Path('/tmp/a.jpg'), txt_a),
                (Path('/tmp/b.jpg'), txt_b),
                (Path('/tmp/missing.jpg'), txt_missing),
            ],
            registry,
            os_fake,
        )
        assert summary['labels_imported'] == 3
        assert summary['files_processed'] == 3
        assert summary['files_failed'] == 0
