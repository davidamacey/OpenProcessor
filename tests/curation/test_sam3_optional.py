"""D5 — the segmenter leg of the detection cascade is optional.

A deployment with no segmentation service of its own leaves
``SAM3_URL``/``--sam3-url`` empty. :class:`SegmenterClient` then constructs
in a *disabled* state: ``segment`` always returns ``None`` (the
same "no candidate" result an unhealthy or empty-response segmenter
already produces) without ever attempting an HTTP call, so the cascade
degrades cleanly instead of crashing or hanging on an unreachable host.

These tests exercise both the client in isolation and the real
cascade routing path (``scripts.curation.region_worker_main._process_crop``,
the same entry point ``tests/curation/test_sam_worker.py`` covers) with
a genuinely-disabled ``SegmenterClient`` — not a mock standing in for it.
"""

from __future__ import annotations

import io
import logging

import httpx
import pytest
from PIL import Image

import scripts.curation.region_worker_main as worker
from scripts.curation.worker.client import SegmenterClient
from src.config import get_region_fields


pytestmark = [
    pytest.mark.asyncio,
    # The cascade needs an active region profile; the default is none.
    pytest.mark.usefixtures('reference_region_profile'),
]


def _make_jpeg(width: int = 320, height: int = 240) -> bytes:
    buf = io.BytesIO()
    Image.new('RGB', (width, height), (50, 80, 120)).save(buf, format='JPEG', quality=85)
    return buf.getvalue()


def _make_task(
    *,
    region_status: str | None = 'pending',
    class_name: str = 'audi',
    group: str = 'cars',
) -> worker._ItemTask:
    return worker._ItemTask(
        crop_id='crop-1',
        image_path='/dev/null/never-read',
        vehicle_bbox_norm=(0.0, 0.0, 1.0, 1.0),
        region_status=region_status,
        class_name=class_name,
        group=group,
        detector_region_in_source=None,
        detector_score=0.0,
        crop_jpeg=_make_jpeg(),
    )


def _lpr_mock(candidates):
    from unittest.mock import AsyncMock, MagicMock

    detector = MagicMock()
    detector.detect_batch = AsyncMock(return_value=candidates)
    return detector


def _vlm_mock(*, is_region: bool):
    from unittest.mock import AsyncMock, MagicMock

    from src.services.labeling.vlm_labeler import VlmRegionVerdict

    g = MagicMock()
    g.verify_region = AsyncMock(
        return_value=VlmRegionVerdict(
            crop_id='ignored', is_region=is_region, confidence='high', reason='test'
        )
    )
    g.aclose = AsyncMock()
    return g


def _ocr_recognizer_mock():
    from unittest.mock import AsyncMock, MagicMock

    r = MagicMock()
    r.detect_regions = AsyncMock(return_value=[])
    r.pick_best_text_region = MagicMock(return_value=None)
    return r


class TestSam3ClientDisabled:
    """Unit-level: the client itself never touches the network when disabled."""

    @pytest.mark.parametrize('base_url', [None, '', '   ', ',,'])
    async def test_disabled_client_construction_does_not_raise(self, base_url) -> None:
        client = SegmenterClient(base_url=base_url)
        assert client.enabled is False
        assert client.base_urls == []
        assert client.base_url == ''
        await client.aclose()

    async def test_disabled_client_returns_none_without_http_call(self, caplog) -> None:
        def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
            raise AssertionError('disabled SegmenterClient must never issue an HTTP request')

        transport = httpx.MockTransport(handler)
        httpx_client = httpx.AsyncClient(transport=transport, timeout=5.0)
        client = SegmenterClient(base_url='', client=httpx_client)

        with caplog.at_level(logging.INFO):
            result = await client.segment(b'\xff\xd8fake')

        assert result is None
        await client.aclose()

    async def test_enabled_client_is_unaffected(self) -> None:
        client = SegmenterClient(base_url='http://sam3-fake:8000')
        assert client.enabled is True
        assert client.base_urls == ['http://sam3-fake:8000']
        assert client.base_url == 'http://sam3-fake:8000'
        await client.aclose()


class TestCascadeWithoutSegmenter:
    """Integration-level: the real cascade path with a genuinely-disabled client."""

    async def test_pending_car_cascade_completes_without_segmenter(self) -> None:
        """No segmenter configured, primary detector also misses → the
        crop lands in a terminal review status instead of crashing, and
        the trace records the segmenter as skipped/missed rather than
        silently omitting it.
        """
        F = get_region_fields()
        sam3 = SegmenterClient(base_url=None)
        assert sam3.enabled is False

        task = _make_task(region_status='pending', group='cars')
        await worker._process_crop(
            task,
            detector=_lpr_mock([None]),
            sam3=sam3,
            ocr_recognizer=_ocr_recognizer_mock(),
            vlm=_vlm_mock(is_region=False),
        )

        # Cascade completed cleanly (no exception) and reached a terminal
        # write rather than hanging or crashing on the missing segmenter.
        assert task.update_doc[F.status] == 'no_region_box'
        chain = task.update_doc.get(F.detector_chain) or []
        assert any('sam3:miss' in s for s in chain)

    async def test_secondary_shape_cascade_completes_without_segmenter(self) -> None:
        """Secondary-shape crops route straight to the segmenter first;
        with none configured the cascade must still fall through cleanly
        (segment returns None) instead of raising.
        """
        F = get_region_fields()
        sam3 = SegmenterClient(base_url='')

        task = _make_task(region_status='pending', class_name='sportbike', group='sportbikes')
        await worker._process_crop(
            task,
            detector=_lpr_mock([]),
            sam3=sam3,
            ocr_recognizer=_ocr_recognizer_mock(),
            vlm=_vlm_mock(is_region=False),
        )

        assert task.update_doc[F.status] == 'no_region_box'
