"""Live: the core ``POST /ingest`` path and the OCR search, through the harness's fake Triton.

``docker/test/fake_triton.py`` serves SCRFD (one face with five landmarks), ArcFace, MobileCLIP and
the OCR pipeline over the KServe v2 gRPC wire, with the tensor names and shapes of
``models/<name>/config.pbtxt``. The core route therefore runs its whole path: decode, detect,
embed, detect and align the face, embed it, read the text, and write the ``visual_search_*``
documents. The picture is generated (the fakes read only its coarse colour layout); the real-photo
COCO ingest stays in the opt-in bench checks.

Assertions read the response and the stores (the core indexes), not only the envelope.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest

from ._ingest_support import picture
from .conftest import refresh, wait_until


pytestmark = pytest.mark.live

FAKE_OCR_TEXT = 'FAKE-PLATE-0042'
FACE_DIM = 512
CLIP_DIM = 512


def _docs(opensearch: Any, index: str, image_id: str) -> list[dict[str, Any]]:
    refresh(opensearch, index)
    resp = opensearch.post(
        f'/{index}/_search',
        json={'size': 10, 'query': {'term': {'image_id': image_id}}},
    )
    assert resp.status_code == 200, resp.text
    return [hit['_source'] for hit in resp.json()['hits']['hits']]


def _vector_dims(docs: list[dict[str, Any]]) -> set[int]:
    return {
        len(v) for doc in docs for k, v in doc.items() if 'embedding' in k and isinstance(v, list)
    }


def test_core_ingest_writes_faces_embeddings_and_ocr(api: Any, opensearch: Any) -> None:
    image_id = f'core-{uuid.uuid4().hex[:8]}'
    resp = api.post(
        '/ingest',
        files={'image': (f'{image_id}.jpg', picture((200, 30, 30), size=(640, 480)), 'image/jpeg')},
        data={'image_id': image_id, 'enable_ocr': 'true', 'enable_faces': 'true'},
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body['status'] == 'success', body
    assert body['num_faces'] == 1, body
    assert body['indexed']['faces'] == 1, body
    assert body['indexed']['global'] is True, body
    assert body['embedding_norm'] == pytest.approx(1.0, abs=1e-3), body
    assert body['ocr']['num_texts'] == 1, body
    assert FAKE_OCR_TEXT in body['ocr']['full_text'], body

    faces = _docs(opensearch, 'visual_search_faces', image_id)
    assert len(faces) == 1, faces
    assert _vector_dims(faces) == {FACE_DIM}, faces
    global_docs = _docs(opensearch, 'visual_search_global', image_id)
    assert len(global_docs) == 1, global_docs
    assert _vector_dims(global_docs) == {CLIP_DIM}, global_docs
    ocr_docs = _docs(opensearch, 'visual_search_ocr', image_id)
    assert ocr_docs, 'no OCR document was indexed'
    assert FAKE_OCR_TEXT in str(ocr_docs), ocr_docs

    # The same bytes again are a duplicate by content hash, not a second set of documents.
    again = api.post(
        '/ingest',
        files={'image': ('again.jpg', picture((200, 30, 30), size=(640, 480)), 'image/jpeg')},
        data={'image_id': f'{image_id}-copy'},
    )
    assert again.status_code == 200, again.text
    assert again.json()['status'] == 'duplicate', again.text

    def found() -> list[dict[str, Any]]:
        refresh(opensearch, 'visual_search_ocr')
        r = api.post('/search/ocr', params={'text': FAKE_OCR_TEXT, 'top_k': 50})
        assert r.status_code == 200, r.text
        return [x for x in r.json()['results'] if x['image_id'] == image_id]

    hits = wait_until(found, timeout=30)
    assert FAKE_OCR_TEXT in hits[0]['matched_text'], hits[0]
