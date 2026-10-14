"""The worker ``SegmenterClient`` against the shipped server, and the cascade rescued by it."""

from __future__ import annotations

import pytest

from curation.segmenter_service._helpers import (
    _PROMPT,
    _asgi_client,
    _FakeProcessor,
    _make_jpeg,
    sam3_backend,
    segmenter,
)
from scripts.curation.worker.client import SegmenterClient, SegmenterRequestFailed
from src.services.detection.cascade_detect.sanity import crop_norm_to_source_norm


# =============================================================================
# The contract, exercised end to end
# =============================================================================


class TestClientServerContract:
    """The worker's ``SegmenterClient`` talking to the shipped server."""

    @pytest.mark.asyncio
    async def test_roundtrip_returns_the_top_candidate(self, served: _FakeProcessor) -> None:
        """The leg is not a no-op: a real client call yields a real candidate."""
        async with _asgi_client() as http:
            client = SegmenterClient(
                base_url='http://segmenter', client=http, text_prompt=_PROMPT, source_name='sam3'
            )
            candidate = await client.segment(_make_jpeg())

        assert candidate is not None
        assert candidate.source == 'sam3'
        assert candidate.score == pytest.approx(0.88)
        assert candidate.bbox_norm == pytest.approx((0.40, 0.60, 0.60, 0.68))
        assert candidate.rectangularity == pytest.approx(1.0)
        # The prompt reached the model as sent — the server supplies none.
        assert served.prompt_calls == [_PROMPT]
        assert served.image_sizes == [(320, 240)]

    @pytest.mark.asyncio
    async def test_candidate_source_is_the_configured_segmenter_name(
        self,
        served: _FakeProcessor,  # noqa: ARG002
    ) -> None:
        async with _asgi_client() as http:
            client = SegmenterClient(
                base_url='http://segmenter',
                client=http,
                text_prompt=_PROMPT,
                source_name='my_segmenter',
            )
            candidate = await client.segment(_make_jpeg())

        assert candidate is not None
        assert candidate.source == 'my_segmenter'

    @pytest.mark.asyncio
    async def test_empty_result_is_a_clean_miss(self) -> None:
        """No candidates → ``None``, which the cascade treats as a miss."""
        empty = _FakeProcessor(boxes=[], scores=[])
        previous = segmenter._pool
        segmenter._pool = sam3_backend.ProcessorPool([empty])
        try:
            async with _asgi_client() as http:
                client = SegmenterClient(
                    base_url='http://segmenter',
                    client=http,
                    text_prompt=_PROMPT,
                    source_name='sam3',
                )
                assert await client.segment(_make_jpeg()) is None
        finally:
            segmenter._pool = previous

    @pytest.mark.asyncio
    async def test_model_still_loading_is_a_503_the_client_absorbs(self) -> None:
        """An unloaded pool 503s; the client raises ``SegmenterRequestFailed``
        (never a "no candidate" ``None``).

        Readiness is checked before the body is decoded, so a malformed
        payload during startup still reports the real problem.
        """
        previous = segmenter._pool
        segmenter._pool = sam3_backend.ProcessorPool([])
        try:
            async with _asgi_client() as http:
                resp = await http.post(
                    '/segment',
                    json={'crop_jpeg_b64': 'not-a-jpeg', 'text_prompt': _PROMPT},
                )
                assert resp.status_code == 503

                client = SegmenterClient(
                    base_url='http://segmenter',
                    client=http,
                    text_prompt=_PROMPT,
                    source_name='sam3',
                )
                with pytest.raises(SegmenterRequestFailed):
                    await client.segment(_make_jpeg())
        finally:
            segmenter._pool = previous


@pytest.mark.usefixtures('reference_region_profile')
class TestCascadeWithTheShippedSegmenter:
    """The worker's cascade, with the real client wired to the real server."""

    @pytest.mark.asyncio
    async def test_primary_detector_miss_is_rescued_by_the_segmenter(
        self, served: _FakeProcessor
    ) -> None:
        """The whole point of G4: with a segmenter deployed, a crop the
        primary detector missed comes back with a segmenter-sourced box
        -- where ``test_segmenter_optional`` sees ``no_region_box``.

        ported off the deleted per-crop ``_process_crop`` (which
        wrapped this in the full class+region cascade); the concern this
        test actually guards -- ``SegmenterClient`` against a real
        HTTP-shaped SAM3 server, source-projected via
        ``crop_norm_to_source_norm`` -- is exercised directly. The live
        streaming pipeline's equivalent routing (primary-detector miss ->
        segmenter hit -> combined VLM call -> ``detected`` write with
        ``region_boxes[0].source == 'sam3'``) is covered end-to-end by
        ``test_region_cascade_integrity.py``'s ``_drive_worker`` harness
        (with a mocked, not real-HTTP, segmenter).
        """
        async with _asgi_client() as http:
            segmenter = SegmenterClient(
                base_url='http://segmenter', client=http, text_prompt=_PROMPT, source_name='sam3'
            )
            assert segmenter.enabled is True

            candidates = await segmenter.segment_multi(_make_jpeg())

        assert len(candidates) == 1
        candidate = candidates[0]
        assert candidate.source == 'sam3'
        # item_bbox_norm is the identity (0,0,1,1) for this crop, so the
        # crop-frame bbox IS the source-frame bbox once projected.
        projected = crop_norm_to_source_norm(candidate.bbox_norm, (0.0, 0.0, 1.0, 1.0))
        assert projected == pytest.approx((0.40, 0.60, 0.60, 0.68))
        assert served.prompt_calls == [_PROMPT]
