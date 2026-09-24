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
        self.search_calls = 0
        self.image_search_calls = 0
        self.msearch_calls: list[list[dict[str, Any]]] = []

    async def msearch(self, *, body: list[dict[str, Any]]) -> dict[str, Any]:
        self.msearch_calls.append(body)
        responses: list[dict[str, Any]] = []
        for line in body[1::2]:
            path = ((line.get('query') or {}).get('term') or {}).get('image_path')
            hits = [
                {'_id': doc_id, '_source': {'image_id': doc.get('image_id', doc_id)}}
                for doc_id, doc in self.images.items()
                if doc.get('image_path') == path
            ]
            responses.append({'hits': {'hits': hits[:1]}})
        return {'responses': responses}

    async def search(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:
        self.search_calls += 1
        cfg = get_curation_config()
        query = body.get('query') or {}
        if index == cfg.images_index:
            self.image_search_calls += 1
            path = (query.get('term') or {}).get('image_path')
            hits = [
                {'_id': doc_id, '_source': doc}
                for doc_id, doc in self.images.items()
                if doc.get('image_path') == path
            ]
            return {'hits': {'hits': hits[:1]}}
        # items index: bool filter image_id term, must_not test_holdout.
        must = (query.get('bool') or {}).get('filter') or []
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
    @pytest.mark.usefixtures('reference_region_profile')
    async def test_created_item_is_seeded_pending_region_detection(
        self, tmp_path: Path, registry: ClassRegistry
    ) -> None:
        from src.config import RegionStatus
        from src.config.region_fields import get_region_fields

        os_fake = FakeLabelOpenSearch(
            images={'img1': {'image_id': 'img1', 'image_path': '/tmp/a.jpg'}}
        )
        txt = tmp_path / 'a.txt'
        txt.write_text('0 0.5 0.5 0.2 0.2\n')
        await import_yolo_labels(Path('/tmp/a.jpg'), txt, registry, os_fake)
        [item] = list(os_fake.items.values())
        assert item[get_region_fields().status] == RegionStatus.PENDING_DETECTION.value

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('reference_region_profile')
    async def test_iou_match_never_touches_region_status(
        self, tmp_path: Path, registry: ClassRegistry
    ) -> None:
        from src.config import RegionStatus
        from src.config.region_fields import get_region_fields

        status = get_region_fields().status
        existing_crop_id = _crop_id_for('img1', [0.41, 0.41, 0.59, 0.59])
        os_fake = FakeLabelOpenSearch(
            images={'img1': {'image_id': 'img1', 'image_path': '/tmp/a.jpg'}},
            items={
                existing_crop_id: {
                    'crop_id': existing_crop_id,
                    'image_id': 'img1',
                    'bbox_norm': [0.41, 0.41, 0.59, 0.59],
                    status: RegionStatus.DETECTED.value,
                }
            },
        )
        txt = tmp_path / 'a.txt'
        txt.write_text('1 0.5 0.5 0.2 0.2\n')
        await import_yolo_labels(Path('/tmp/a.jpg'), txt, registry, os_fake)
        assert os_fake.items[existing_crop_id][status] == RegionStatus.DETECTED.value

    @pytest.mark.asyncio
    async def test_created_item_has_no_region_status_without_a_profile(
        self, tmp_path: Path, registry: ClassRegistry, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.config.region_fields import get_region_fields
        from src.services.detection import profile_registry

        monkeypatch.delenv('OP_REGION_PROFILE', raising=False)
        profile_registry._reset_registry_for_tests()
        os_fake = FakeLabelOpenSearch(
            images={'img1': {'image_id': 'img1', 'image_path': '/tmp/a.jpg'}}
        )
        txt = tmp_path / 'a.txt'
        txt.write_text('0 0.5 0.5 0.2 0.2\n')
        await import_yolo_labels(Path('/tmp/a.jpg'), txt, registry, os_fake)
        [item] = list(os_fake.items.values())
        assert get_region_fields().status not in item

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
        # F-26: the image docs are resolved via one batched _msearch
        # instead of one `search` per file.
        assert len(os_fake.msearch_calls) == 1
        assert len(os_fake.msearch_calls[0]) == 6  # 3 files * (index line + query line)

    @pytest.mark.asyncio
    async def test_batch_over_100_files_chunks_msearch_calls(
        self, tmp_path: Path, registry: ClassRegistry
    ) -> None:
        n = 105
        images = {
            f'img{i}': {'image_id': f'img{i}', 'image_path': f'/tmp/{i}.jpg'} for i in range(n)
        }
        os_fake = FakeLabelOpenSearch(images=images)
        pairs = []
        for i in range(n):
            txt = tmp_path / f'{i}.txt'
            txt.write_text('0 0.5 0.5 0.2 0.2\n')
            pairs.append((Path(f'/tmp/{i}.jpg'), txt))

        summary = await import_labels_batch(pairs, registry, os_fake)

        assert summary['labels_imported'] == n
        # F-26: chunked at 100 -> 2 _msearch calls for 105 files, not 105
        # individual `search` calls.
        assert len(os_fake.msearch_calls) == 2
        assert os_fake.image_search_calls == 0


def _detector_item(image_id: str, bbox: list[float], class_id: int | None = 0) -> dict[str, Any]:
    crop_id = _crop_id_for(image_id, bbox)
    return {
        'crop_id': crop_id,
        'image_id': image_id,
        'bbox_norm': bbox,
        'class_id': class_id,
        'class_name': 'widget' if class_id == 0 else None,
        'class_source': 'detector',
        'confidence': 0.8,
        'class_validated': False,
    }


class TestDisagreementReport:
    """``detect_mismatches`` reports every way the detector and the labels
    disagree — not just class flips — so a re-ingest of a labeled dataset
    yields misses and false positives too."""

    @pytest.mark.asyncio
    async def test_missed_label_and_unmatched_detection(
        self, tmp_path: Path, registry: ClassRegistry
    ) -> None:
        hit = _detector_item('img1', [0.41, 0.41, 0.59, 0.59])
        fp = _detector_item('img1', [0.0, 0.0, 0.1, 0.1])
        os_fake = FakeLabelOpenSearch(
            images={'img1': {'image_id': 'img1', 'image_path': '/tmp/a.jpg'}},
            items={hit['crop_id']: hit, fp['crop_id']: fp},
        )
        txt = tmp_path / 'a.txt'
        # One label matching `hit`, one label nothing overlaps.
        txt.write_text('0 0.5 0.5 0.2 0.2\n0 0.85 0.85 0.1 0.1\n')
        sink: list[dict[str, Any]] = []
        n = await import_yolo_labels(
            Path('/tmp/a.jpg'), txt, registry, os_fake, detect_mismatches=True, mismatch_sink=sink
        )
        assert n == 2
        kinds = sorted(r['kind'] for r in sink)
        assert kinds == ['missed_label', 'unmatched_detection']
        missed = next(r for r in sink if r['kind'] == 'missed_label')
        assert missed['bbox_norm'] == pytest.approx([0.8, 0.8, 0.9, 0.9])
        assert missed['label_class_id'] == 0
        unmatched = next(r for r in sink if r['kind'] == 'unmatched_detection')
        assert unmatched['crop_id'] == fp['crop_id']
        assert unmatched['detector_confidence'] == 0.8

    @pytest.mark.asyncio
    async def test_empty_label_file_reports_every_detection(
        self, tmp_path: Path, registry: ClassRegistry
    ) -> None:
        """A background image (empty .txt) with detections: all are FPs."""
        fp1 = _detector_item('img1', [0.1, 0.1, 0.2, 0.2])
        fp2 = _detector_item('img1', [0.5, 0.5, 0.7, 0.7], class_id=None)
        os_fake = FakeLabelOpenSearch(
            images={'img1': {'image_id': 'img1', 'image_path': '/tmp/bg.jpg'}},
            items={fp1['crop_id']: fp1, fp2['crop_id']: fp2},
        )
        txt = tmp_path / 'bg.txt'
        txt.write_text('')
        sink: list[dict[str, Any]] = []
        n = await import_yolo_labels(
            Path('/tmp/bg.jpg'), txt, registry, os_fake, detect_mismatches=True, mismatch_sink=sink
        )
        assert n == 0
        assert os_fake.bulk_calls == []  # nothing to write for a background
        assert sorted(r['crop_id'] for r in sink) == sorted([fp1['crop_id'], fp2['crop_id']])
        assert {r['kind'] for r in sink} == {'unmatched_detection'}

    @pytest.mark.asyncio
    async def test_clean_background_reports_nothing(
        self, tmp_path: Path, registry: ClassRegistry
    ) -> None:
        os_fake = FakeLabelOpenSearch(
            images={'img1': {'image_id': 'img1', 'image_path': '/tmp/bg.jpg'}}
        )
        sink: list[dict[str, Any]] = []
        n = await import_yolo_labels(
            Path('/tmp/bg.jpg'),
            tmp_path / 'absent.txt',
            registry,
            os_fake,
            detect_mismatches=True,
            mismatch_sink=sink,
        )
        assert n == 0
        assert sink == []

    @pytest.mark.asyncio
    async def test_validated_items_are_not_detections(
        self, tmp_path: Path, registry: ClassRegistry
    ) -> None:
        prior = _detector_item('img1', [0.1, 0.1, 0.2, 0.2])
        prior['class_validated'] = True
        os_fake = FakeLabelOpenSearch(
            images={'img1': {'image_id': 'img1', 'image_path': '/tmp/a.jpg'}},
            items={prior['crop_id']: prior},
        )
        txt = tmp_path / 'a.txt'
        txt.write_text('')
        sink: list[dict[str, Any]] = []
        await import_yolo_labels(
            Path('/tmp/a.jpg'), txt, registry, os_fake, detect_mismatches=True, mismatch_sink=sink
        )
        assert sink == []

    @pytest.mark.asyncio
    async def test_batch_counts_by_kind(self, tmp_path: Path, registry: ClassRegistry) -> None:
        flip = _detector_item('img1', [0.41, 0.41, 0.59, 0.59], class_id=0)
        fp = _detector_item('img2', [0.1, 0.1, 0.2, 0.2])
        os_fake = FakeLabelOpenSearch(
            images={
                'img1': {'image_id': 'img1', 'image_path': '/tmp/a.jpg'},
                'img2': {'image_id': 'img2', 'image_path': '/tmp/b.jpg'},
            },
            items={flip['crop_id']: flip, fp['crop_id']: fp},
        )
        a = tmp_path / 'a.txt'
        a.write_text('1 0.5 0.5 0.2 0.2\n0 0.85 0.85 0.1 0.1\n')  # class flip + miss
        b = tmp_path / 'b.txt'
        b.write_text('')  # background with one detection
        records: list[dict[str, Any]] = []
        summary = await import_labels_batch(
            [(Path('/tmp/a.jpg'), a), (Path('/tmp/b.jpg'), b)],
            registry,
            os_fake,
            detect_mismatches=True,
            disagreement_sink=records,
        )
        assert summary['mismatches'] == 1
        assert summary['missed_labels'] == 1
        assert summary['unmatched_detections'] == 1
        assert len(records) == 3


class TestImportLabelsBatchImageDocsPassthrough:
    @pytest.mark.asyncio
    async def test_precomputed_image_docs_skip_msearch_entirely(
        self, tmp_path: Path, registry: ClassRegistry
    ) -> None:
        """F-26: a caller that already knows image_id (e.g. an ingest
        batch's own results) can pass image_docs= and skip the _msearch
        image-lookup round-trip completely."""
        os_fake = FakeLabelOpenSearch(
            images={'img1': {'image_id': 'img1', 'image_path': '/tmp/a.jpg'}}
        )
        txt_a = tmp_path / 'a.txt'
        txt_a.write_text('0 0.5 0.5 0.2 0.2\n')

        summary = await import_labels_batch(
            [(Path('/tmp/a.jpg'), txt_a)],
            registry,
            os_fake,
            image_docs={'/tmp/a.jpg': {'image_id': 'img1', '_id': 'img1'}},
        )

        assert summary['labels_imported'] == 1
        assert os_fake.msearch_calls == []
        assert os_fake.image_search_calls == 0
