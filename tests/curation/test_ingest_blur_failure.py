"""A blur computation that fails fails the image; it is not stored as a flat (blur 0.0) frame."""

from __future__ import annotations

import cv2
import pytest

from curation.ingest_fakes import _jpeg_bytes, _make_service
from src.services.curation import ingest_index


@pytest.mark.asyncio
async def test_a_failed_blur_fails_the_image_and_stores_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _broken(*_a: object, **_k: object) -> float:
        raise cv2.error('laplacian failed')

    monkeypatch.setattr(ingest_index, 'image_lap_var', _broken)
    svc, os_fake, _ = _make_service()
    result = await svc.ingest_one(_jpeg_bytes(), '/tmp/photo.jpg')
    assert result.status == 'failed'
    assert result.error_kind == 'blur_failed'
    assert 'laplacian failed' in (result.error or '')
    assert not os_fake.items
    assert not os_fake.images
