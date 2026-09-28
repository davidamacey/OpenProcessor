"""Ingest -> crops/status round trip through the real app.

Ingests two images through ``POST /curation/ingest/batch`` against a
fake OpenSearch + a scripted detector/PE-encoder, then asserts
``GET /curation/crops`` lists both items with the quality fields
(``crop_area_norm``, ``blur_lap_var``, ``blur_lap_ratio``) populated,
and ``GET /curation/ingest/status`` reflects the new images.

Per this repo's house rule (don't add the repo's first live-stack
dependency), this fakes the OpenSearch/Triton I/O boundary
rather than standing up a live stack — see
``tests/integration/test_ingest_occ.py`` for the same convention.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from integration.ingest_fakes import (
    FakeOpenSearch,
    FakeTritonPool,
    curation_app,
    jpeg_bytes as _jpeg_bytes,
)


if TYPE_CHECKING:
    from fastapi.testclient import TestClient


pytestmark = pytest.mark.integration


@pytest.fixture
def fake_opensearch() -> FakeOpenSearch:
    return FakeOpenSearch()


@pytest.fixture
def fake_triton() -> FakeTritonPool:
    return FakeTritonPool()


@pytest.fixture
def client(
    fake_opensearch: FakeOpenSearch,
    fake_triton: FakeTritonPool,
    monkeypatch: pytest.MonkeyPatch,
) -> TestClient:
    with curation_app(fake_opensearch, fake_triton, monkeypatch) as c:
        yield c


def test_ingest_batch_then_crops_and_status(
    client: TestClient, fake_opensearch: FakeOpenSearch, fake_triton: FakeTritonPool
) -> None:
    body = {
        'items': [
            {'path': '/tmp/roundtrip_a.jpg', 'source': 'roundtrip_test'},
            {'path': '/tmp/roundtrip_b.jpg', 'source': 'roundtrip_test'},
        ]
    }
    # Write real files so the router's Path(...).read_bytes() succeeds.
    from pathlib import Path

    Path('/tmp/roundtrip_a.jpg').write_bytes(_jpeg_bytes(1))
    Path('/tmp/roundtrip_b.jpg').write_bytes(_jpeg_bytes(2))

    resp = client.post('/curation/projects/default/ingest/batch', json=body)
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    assert payload['summary']['successful'] == 2
    assert payload['summary']['crops_indexed'] == 2

    crops_resp = client.get('/curation/projects/default/crops')
    assert crops_resp.status_code == 200, crops_resp.text
    crops_payload = crops_resp.json()
    items = crops_payload.get('items') or crops_payload.get('crops') or crops_payload
    assert isinstance(items, list)
    assert len(items) == 2
    for item in items:
        # crops.py's wire model surfaces crop_area_norm/crop_rank_in_image/
        # blur_lap_ratio (not blur_lap_var, which is diagnostic-only and
        # excluded from that response by design) -- checked against the
        # response here, and against the raw OpenSearch doc below.
        assert item.get('crop_area_norm') is not None
        assert item.get('blur_lap_ratio') is not None
        assert item.get('crop_rank_in_image') is not None

    assert len(fake_opensearch.items) == 2
    for doc in fake_opensearch.items.values():
        assert doc.get('blur_lap_var') is not None
        assert doc.get('blur_lap_ratio') is not None
        assert doc.get('crop_area_norm') is not None
        assert doc.get('pe_embedding') is not None

    status_resp = client.get('/curation/projects/default/ingest/status')
    assert status_resp.status_code == 200, status_resp.text
    status_payload = status_resp.json()
    assert status_payload['total'] == 2

    # G11 guard, end to end through the router: two images cost exactly
    # one batched Triton round-trip, not two single-image ones.
    assert fake_triton.batch_sizes == [2]


def test_ingest_batch_rejects_the_removed_label_fields(client: TestClient) -> None:
    """W10: ``label_txt_path``/``label_source``/``detect_mismatches`` are
    removed from ``/ingest/batch`` outright (no back-compat) -- ingesting
    an already-labeled dataset is ``POST /datasets/imports`` now. The old
    G12 companion-label roundtrip this replaces
    (``test_ingest_batch_imports_companion_labels``) lives on as
    ``tests/curation/dataset_import/test_job_import.py`` and
    ``tests/integration/test_class_identity_e2e.py``."""
    from pathlib import Path

    image_path = Path('/tmp/roundtrip_labeled.jpg')
    image_path.write_bytes(_jpeg_bytes(7))

    resp = client.post(
        '/curation/projects/default/ingest/batch',
        json={
            'items': [{'path': str(image_path), 'source': 'roundtrip_test', 'label_txt_path': 'x'}],
        },
    )
    assert resp.status_code == 422, resp.text
