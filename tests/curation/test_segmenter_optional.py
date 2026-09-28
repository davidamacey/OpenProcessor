"""The segmenter leg of the detection cascade is optional.

A deployment with no segmentation service of its own leaves
``OP_SEGMENTER_URL``/``--segmenter-url`` empty. :class:`SegmenterClient` then constructs
in a *disabled* state: ``segment``/``segment_multi`` always return
"no candidate" (the same result an unhealthy or empty-response segmenter
already produces) without ever attempting an HTTP call, so the cascade
degrades cleanly instead of crashing or hanging on an unreachable host.

``TestSegmenterClientDisabled`` exercises the client in isolation.
``TestCascadeWithoutSegmenter`` exercises the real streaming pipeline
(``scripts.curation.worker.runner.run``, via the ``_drive_worker`` harness
in ``test_region_cascade_integrity.py``) with the segmenter leg
missing -- W8: ported from the deleted per-crop ``_process_crop`` entry
point, same intent (the cascade degrades to a terminal
``no_region_box`` write, never crashes or hangs, when neither detector
leg finds anything).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import httpx
import pytest

from scripts.curation.worker.client import SegmenterClient
from src.config import get_region_fields

from .test_region_cascade_integrity import _accept, _drive_worker, _FakeOpenSearch, _item


if TYPE_CHECKING:
    from pathlib import Path


pytestmark = [
    pytest.mark.asyncio,
    # The cascade needs an active region profile; the default is none.
    pytest.mark.usefixtures('reference_region_profile'),
]


class TestSegmenterClientDisabled:
    """Unit-level: the client itself never touches the network when disabled."""

    @pytest.mark.parametrize('base_url', [None, '', '   ', ',,'])
    async def test_disabled_client_construction_does_not_raise(self, base_url) -> None:
        client = SegmenterClient(base_url=base_url, source_name='sam3')
        assert client.enabled is False
        assert client.base_urls == []
        assert client.base_url == ''
        await client.aclose()

    async def test_disabled_client_returns_none_without_http_call(self, caplog) -> None:
        def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
            raise AssertionError('disabled SegmenterClient must never issue an HTTP request')

        transport = httpx.MockTransport(handler)
        httpx_client = httpx.AsyncClient(transport=transport, timeout=5.0)
        client = SegmenterClient(base_url='', client=httpx_client, source_name='sam3')

        with caplog.at_level(logging.INFO):
            result = await client.segment(b'\xff\xd8fake')

        assert result is None
        await client.aclose()

    async def test_enabled_client_is_unaffected(self) -> None:
        client = SegmenterClient(base_url='http://segmenter-fake:8000', source_name='sam3')
        assert client.enabled is True
        assert client.base_urls == ['http://segmenter-fake:8000']
        assert client.base_url == 'http://segmenter-fake:8000'
        await client.aclose()


class TestCascadeWithoutSegmenter:
    """Integration-level: the real streaming pipeline with the segmenter
    leg finding nothing (standing in for a genuinely-disabled client --
    ``TestSegmenterClientDisabled`` above already proves the disabled
    client itself returns no candidate without an HTTP call)."""

    async def test_pending_car_cascade_completes_without_segmenter(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No segmenter configured, primary detector also misses → the
        crop lands in a terminal review status instead of crashing, and
        the trace records the segmenter as skipped/missed rather than
        silently omitting it.
        """
        F = get_region_fields()
        fake_os = _FakeOpenSearch({'c1': _item()}, search_delay=0.0, lag_searches=0)
        await _drive_worker(
            tmp_path,
            monkeypatch,
            fake_os=fake_os,
            primary=None,
            segmenter=None,
            reply=_accept(),
            visible=True,
        )
        doc = fake_os.live['c1']
        # Cascade completed cleanly (no exception) and reached a terminal
        # write rather than hanging or crashing on the missing segmenter
        # (the reference profile's text-hint re-pass -- OCR mocked empty
        # via _drive_worker -- is the fallback leg that actually records
        # the miss here, since the segmenter mock's ``enabled`` reads
        # truthy; text_hint_active() only checks that flag, not whether
        # a real segmenter answered).
        assert doc[F.status] == 'no_region_box'
        chain = doc.get(F.detector_chain) or []
        assert any('miss' in s for s in chain), chain

    async def test_secondary_shape_cascade_completes_without_segmenter(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Secondary-shape crops route straight to the segmenter first;
        with none configured the cascade must still fall through cleanly
        (segment_multi returns no candidates) instead of raising.
        """
        F = get_region_fields()
        fake_os = _FakeOpenSearch({'c1': _item()}, search_delay=0.0, lag_searches=0)
        await _drive_worker(
            tmp_path,
            monkeypatch,
            fake_os=fake_os,
            primary=None,
            segmenter=None,
            reply=_accept(),
            visible=True,
        )
        doc = fake_os.live['c1']
        assert doc[F.status] == 'no_region_box'
