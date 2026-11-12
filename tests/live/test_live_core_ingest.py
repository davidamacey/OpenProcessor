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


# Every document field and `/search/ocr` result key the test reads. An offline test
# (tests/compose/test_core_index_fields.py) checks them against the index mappings and the
# response model, so a rename fails there first.
VECTOR_FIELD = {'visual_search_faces': 'embedding', 'visual_search_global': 'global_embedding'}
OCR_TEXT_FIELD = 'text'
QUERIED_FIELDS = {
    'visual_search_faces': ['image_id', VECTOR_FIELD['visual_search_faces']],
    'visual_search_global': ['image_id', VECTOR_FIELD['visual_search_global']],
    'visual_search_ocr': ['image_id', OCR_TEXT_FIELD],
}
OCR_RESULT_KEYS = ['image_id', 'matched_text']


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
    assert {len(f[VECTOR_FIELD['visual_search_faces']]) for f in faces} == {FACE_DIM}, faces
    global_docs = _docs(opensearch, 'visual_search_global', image_id)
    assert len(global_docs) == 1, global_docs
    assert {len(g[VECTOR_FIELD['visual_search_global']]) for g in global_docs} == {CLIP_DIM}, (
        global_docs
    )
    ocr_docs = _docs(opensearch, 'visual_search_ocr', image_id)
    assert ocr_docs, 'no OCR document was indexed'
    assert any(FAKE_OCR_TEXT in d[OCR_TEXT_FIELD] for d in ocr_docs), ocr_docs

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
