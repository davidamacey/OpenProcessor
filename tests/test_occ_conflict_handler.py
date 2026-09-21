"""Verify the app-level 409 mapping for :class:`OCCFinalConflictError`.

``src.clients.occ.OCCFinalConflictError``'s docstring claims human-write
endpoints surface it as HTTP 409. Several curation router call sites
(``src/routers/curation/regions.py``, ``crops.py``) do a bare
``except OCCFinalConflictError: raise`` specifically so the exception
propagates unconverted to a dedicated FastAPI handler rather than the
generic 404/500 fallbacks in the same ``try`` block — that only works if
such a handler is actually registered on the app. This test builds a
fresh app via ``src.main.create_app()`` (no lifespan triggered, so no
real Triton/OpenSearch connection is attempted), mounts a throwaway
route that raises the exception, and asserts the response the handler
produces.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.clients.occ import OCCFinalConflictError
from src.main import create_app


def test_occ_final_conflict_error_maps_to_409() -> None:
    app = create_app()

    @app.get('/__test_occ_conflict')
    async def _raise_conflict() -> None:
        raise OCCFinalConflictError(doc_id='crop-123', retries=3)

    client = TestClient(app)  # no `with`: lifespan startup is not exercised
    resp = client.get('/__test_occ_conflict')

    assert resp.status_code == 409, resp.text
    body = resp.json()
    assert body['doc_id'] == 'crop-123'
    assert body['retries'] == 3
    assert 'request_id' in body
