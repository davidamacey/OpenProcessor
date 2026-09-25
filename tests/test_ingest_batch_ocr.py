"""
DF2: generic `/ingest/batch` never actually ran OCR — `run_single_unified`
inside `VisualSearchService.ingest_batch` had no OCR call at all, so
`num_texts` stayed 0 for every image and the OCR-indexing step downstream
never had anything to index.

This drives `ingest_batch` end-to-end (detection/CLIP/faces disabled to
keep the test narrow) with a fake OCR service standing in for the real
Triton-backed one, and asserts `num_texts > 0` and that OCR indexing ran.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from src.services.visual_search import VisualSearchService


class _FakeOcrService:
    """Stands in for get_ocr_service(); always finds one text region."""

    def extract_text(self, image_bytes: bytes, filter_by_score: bool = True) -> dict[str, Any]:  # noqa: ARG002
        return {
            'status': 'success',
            'texts': ['HELLO123'],
            'boxes': [[10, 10, 50, 10, 50, 20, 10, 20]],
            'boxes_normalized': [[0.1, 0.1, 0.5, 0.2]],
            'det_scores': [0.95],
            'rec_scores': [0.9],
            'num_texts': 1,
        }


class _FakeOpenSearch:
    def __init__(self):
        self.indexed_ocr_calls: list[dict] = []
        self.check_duplicates_by_hash_batch = AsyncMock(return_value={})
        self.bulk_ingest = AsyncMock(return_value={'global': 1, 'vehicles': 0, 'people': 0})
        self.bulk_index_faces = AsyncMock(return_value={'indexed': 0})

    async def index_ocr_results(self, **kwargs):
        self.indexed_ocr_calls.append(kwargs)


@pytest.fixture
def service(monkeypatch: pytest.MonkeyPatch) -> tuple[VisualSearchService, _FakeOpenSearch]:
    fake_opensearch = _FakeOpenSearch()
    svc = VisualSearchService.__new__(VisualSearchService)
    svc.opensearch = fake_opensearch  # type: ignore[assignment]
    svc.inference = None  # type: ignore[assignment]

    monkeypatch.setattr(
        'src.clients.triton_client.get_triton_client',
        lambda url: object(),  # noqa: ARG005
    )
    monkeypatch.setattr(
        'src.clients.fast_face_client.get_fast_face_client',
        lambda url: object(),  # noqa: ARG005
    )
    monkeypatch.setattr('src.services.ocr_service.get_ocr_service', lambda: _FakeOcrService())
    return svc, fake_opensearch


@pytest.mark.asyncio
async def test_ingest_batch_runs_ocr_and_reports_num_texts(
    service: tuple[VisualSearchService, _FakeOpenSearch],
) -> None:
    svc, fake_opensearch = service

    images_data: list[tuple[bytes, str, str | None]] = [(b'fake-jpeg-bytes', 'img-1', None)]

    result = await svc.ingest_batch(
        images_data,
        skip_duplicates=False,
        detect_near_duplicates=False,
        enable_ocr=True,
        enable_detection=False,
        enable_faces=False,
        enable_clip=False,
    )

    assert result['status'] == 'success'
    # DF2: OCR must actually have run and been indexed, not stayed a no-op.
    assert result['indexed']['ocr'] == 1
    assert len(fake_opensearch.indexed_ocr_calls) == 1
    ocr_call = fake_opensearch.indexed_ocr_calls[0]
    assert ocr_call['texts'] == ['HELLO123']
    assert ocr_call['image_id'] == 'img-1'
