"""G4 — the shipped segmenter server, against the shipped segmenter client.

``docker/segmenter/`` exists to give the detection cascade's segmenter leg
something to point at. These tests run the real FastAPI app from that
image in-process (ASGI transport, no container) and drive it with the
real :class:`SegmenterClient` the detection worker uses — so the wire contract
is verified from both ends at once rather than asserted twice.

The only thing faked is SAM 3 itself: a ``_FakeProcessor`` implementing
the three-call protocol ``sam3_backend.segment_images`` relies on
(``set_image`` → text encode → ``_forward_grounding``). Everything else —
route paths, request/response schemas, base64 decoding, candidate
sorting, box normalization, the pool lock, the client's parsing and
failure handling — is the code that ships.

Complements ``test_segmenter_optional.py``, which proves the *absence* of a
segmenter degrades cleanly. This proves the presence of one actually
works.
"""

from __future__ import annotations

import asyncio
import base64
import importlib.util
import io
import sys
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
from PIL import Image

from scripts.curation.worker.client import SegmenterClient, SegmenterRequestFailed
from src.services.detection.cascade_detect import crop_norm_to_source_norm


_SEGMENTER_DIR = Path(__file__).resolve().parents[2] / 'docker' / 'segmenter'


def _load_segmenter_module(file_stem: str, module_name: str) -> Any:
    """Import a module out of ``docker/segmenter/`` by path.

    That directory is a container build context, not an importable
    package, and ``main.py`` is a name we do not want to leak into
    ``sys.modules``. ``sam3_backend`` must keep its real name because
    ``main.py`` imports it by that name at container runtime.
    """
    if module_name in sys.modules:
        return sys.modules[module_name]
    path = _SEGMENTER_DIR / f'{file_stem}.py'
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None, f'cannot load {path}'
    assert spec.loader is not None, f'no loader for {path}'
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


sam3_backend = _load_segmenter_module('sam3_backend', 'sam3_backend')
segmenter = _load_segmenter_module('main', 'openprocessor_segmenter_main')

# The prompt is a deployment value, never a server default — these tests
# pick an arbitrary one to prove the server carries it through untouched.
_PROMPT = 'widget, gadget'


# =============================================================================
# Fake SAM 3
# =============================================================================


class _FakeBackbone:
    def __init__(self, prompt_calls: list[str]) -> None:
        self._prompt_calls = prompt_calls

    def forward_text(self, prompts: list[str], device: str | None = None) -> dict:
        self._prompt_calls.append(prompts[0])
        return {'text_embed': prompts[0], 'device': device}


class _FakeModel:
    def __init__(self, prompt_calls: list[str]) -> None:
        self.backbone = _FakeBackbone(prompt_calls)

    def _get_dummy_prompt(self) -> dict:
        return {'dummy': True}


class _FakeProcessor:
    """Implements the slice of ``Sam3Processor`` the backend actually uses."""

    def __init__(
        self,
        boxes: list[list[float]],
        scores: list[float],
        masks: np.ndarray | None = None,
    ) -> None:
        self._boxes = boxes
        self._scores = scores
        self._masks = masks
        self.prompt_calls: list[str] = []
        self.image_sizes: list[tuple[int, int]] = []
        self.device = 'cpu'
        self.model = _FakeModel(self.prompt_calls)
        # W8c: mirrors the real Sam3Processor's plain instance attribute
        # that ProcessorPool.acquire(min_score=...) sets/restores.
        self.confidence_threshold: float | None = None

    def set_image(self, image: Image.Image) -> dict:
        self.image_sizes.append(image.size)
        return {
            'backbone_out': {},
            'original_width': image.size[0],
            'original_height': image.size[1],
        }

    def _forward_grounding(self, state: dict) -> dict:
        state['boxes'] = self._boxes
        state['scores'] = self._scores
        state['masks'] = self._masks
        return state


def _full_mask(h: int = 8, w: int = 8) -> np.ndarray:
    """A mask that fills its own bbox — rectangularity 1.0."""
    return np.ones((1, h, w), dtype=np.float32)


def _make_jpeg(width: int = 320, height: int = 240) -> bytes:
    buf = io.BytesIO()
    Image.new('RGB', (width, height), (50, 80, 120)).save(buf, format='JPEG', quality=85)
    return buf.getvalue()


@pytest.fixture
def processor() -> _FakeProcessor:
    """One candidate, a plausible box, a mask that fills it."""
    return _FakeProcessor(
        boxes=[[0.40, 0.60, 0.60, 0.68]],
        scores=[0.88],
        masks=_full_mask(),
    )


@pytest.fixture
def served(processor: _FakeProcessor):
    """Install a pool on the real app; restore whatever was there after."""
    previous = segmenter._pool
    segmenter._pool = sam3_backend.ProcessorPool([processor])
    try:
        yield processor
    finally:
        segmenter._pool = previous


def _asgi_client() -> httpx.AsyncClient:
    """An httpx client wired straight into the segmenter's ASGI app."""
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=segmenter.app),
        base_url='http://segmenter',
        timeout=10.0,
    )


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


class TestWireSurface:
    """Direct HTTP assertions on the shape the client depends on."""

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('served')
    async def test_generic_path_serves_the_wire_contract(self) -> None:
        """``/segment`` is the shipped client's only path (the
        ``/segmenter/segment`` alias was removed -- no deployed worker
        posts to it anymore)."""
        payload = {
            'crop_jpeg_b64': base64.b64encode(_make_jpeg()).decode('ascii'),
            'text_prompt': _PROMPT,
        }
        async with _asgi_client() as http:
            generic = await http.post('/segment', json=payload)

        assert generic.status_code == 200
        assert generic.json()['prompt'] == _PROMPT
        assert generic.json()['crop_size'] == [320, 240]

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('served')
    async def test_retired_alias_path_is_gone(self) -> None:
        payload = {
            'crop_jpeg_b64': base64.b64encode(_make_jpeg()).decode('ascii'),
            'text_prompt': _PROMPT,
        }
        async with _asgi_client() as http:
            legacy = await http.post('/segmenter/segment', json=payload)
        assert legacy.status_code == 404

    @pytest.mark.asyncio
    async def test_text_prompt_has_no_server_side_default(self, served: _FakeProcessor) -> None:
        """Omitting the prompt is a 422, not a silently-applied domain default.

        This is the genericization guarantee: the server cannot quietly
        segment for someone else's concept.
        """
        async with _asgi_client() as http:
            resp = await http.post(
                '/segment',
                json={'crop_jpeg_b64': base64.b64encode(_make_jpeg()).decode('ascii')},
            )
            blank = await http.post(
                '/segment',
                json={
                    'crop_jpeg_b64': base64.b64encode(_make_jpeg()).decode('ascii'),
                    'text_prompt': '',
                },
            )

        assert resp.status_code == 422
        assert blank.status_code == 422
        assert served.prompt_calls == []

    @pytest.mark.asyncio
    async def test_candidates_are_sorted_and_capped(self) -> None:
        many = _FakeProcessor(
            boxes=[[0.1, 0.1, 0.2, 0.2], [0.3, 0.3, 0.5, 0.4], [0.5, 0.5, 0.7, 0.6]],
            scores=[0.20, 0.91, 0.55],
        )
        previous = segmenter._pool
        segmenter._pool = sam3_backend.ProcessorPool([many])
        try:
            async with _asgi_client() as http:
                resp = await http.post(
                    '/segment',
                    json={
                        'crop_jpeg_b64': base64.b64encode(_make_jpeg()).decode('ascii'),
                        'text_prompt': _PROMPT,
                        'max_candidates': 2,
                    },
                )
        finally:
            segmenter._pool = previous

        cands = resp.json()['candidates']
        assert [c['score'] for c in cands] == [pytest.approx(0.91), pytest.approx(0.55)]
        # No mask head output → mask_iou is null, not fabricated.
        assert all(c['mask_iou'] is None for c in cands)

    @pytest.mark.asyncio
    async def test_batch_results_align_with_request_order(self, served: _FakeProcessor) -> None:
        small = base64.b64encode(_make_jpeg(64, 48)).decode('ascii')
        large = base64.b64encode(_make_jpeg(320, 240)).decode('ascii')
        async with _asgi_client() as http:
            resp = await http.post(
                '/segment/batch',
                json={'crops_jpeg_b64': [small, large], 'text_prompt': _PROMPT},
            )

        body = resp.json()
        assert resp.status_code == 200
        assert [r['crop_size'] for r in body['results']] == [[64, 48], [320, 240]]
        # One text-encode for the whole batch — the prompt cache is per
        # processor, not per call.
        assert served.prompt_calls == [_PROMPT]

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('served')
    async def test_bad_base64_is_a_400_with_the_offending_index(self) -> None:
        good = base64.b64encode(_make_jpeg()).decode('ascii')
        async with _asgi_client() as http:
            resp = await http.post(
                '/segment/batch',
                json={'crops_jpeg_b64': [good, 'not-a-jpeg'], 'text_prompt': _PROMPT},
            )

        assert resp.status_code == 400
        assert 'idx 1' in resp.json()['detail']

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('served')
    async def test_health_reports_pool_state(self) -> None:
        async with _asgi_client() as http:
            loaded = (await http.get('/health')).json()
        assert loaded == {
            'status': 'healthy',
            'model': 'sam3',
            'device': sam3_backend.device_name(),
            'loaded': True,
            'instances': 1,
            'max_candidates': sam3_backend.MAX_CANDIDATES_CAP,
            'default_min_score': sam3_backend.DEFAULT_CONFIDENCE_THRESHOLD,
        }

        previous = segmenter._pool
        segmenter._pool = sam3_backend.ProcessorPool([])
        try:
            async with _asgi_client() as http:
                loading = (await http.get('/health')).json()
        finally:
            segmenter._pool = previous
        assert loading['status'] == 'loading'
        assert loading['loaded'] is False


# =============================================================================
# The cascade leg itself
# =============================================================================


def _detector_mock(candidates):
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
            crop_id='ignored', is_region=is_region, confidence='medium', reason='test'
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

        W8: ported off the deleted per-crop ``_process_crop`` (which
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


# =============================================================================
# W8c: min_score + the raised max_candidates ceiling
# =============================================================================


class TestMinScoreAndCandidateCap:
    @pytest.mark.asyncio
    async def test_min_score_filters_on_the_leased_processor_and_is_restored(self) -> None:
        """The real ``Sam3Processor`` filters inside ``_forward_grounding``
        via ``self.confidence_threshold``; ``_FakeProcessor`` doesn't, so
        this wires that filter in to prove ``min_score`` actually reaches
        the leased processor (``ProcessorPool.acquire``) rather than just
        being accepted and ignored -- and that it's gone afterward, so it
        never leaks onto the next caller's request."""
        many = _FakeProcessor(
            boxes=[[0.1, 0.1, 0.2, 0.2], [0.3, 0.3, 0.5, 0.4]],
            scores=[0.3, 0.9],
        )
        base_forward = many._forward_grounding

        def _thresholded_forward(state: dict) -> dict:
            out = base_forward(state)
            threshold = getattr(many, 'confidence_threshold', None)
            if threshold is not None:
                keep = [i for i, s in enumerate(out['scores']) if s >= threshold]
                out['boxes'] = [out['boxes'][i] for i in keep]
                out['scores'] = [out['scores'][i] for i in keep]
            return out

        many._forward_grounding = _thresholded_forward  # type: ignore[method-assign]
        assert many.confidence_threshold is None

        previous = segmenter._pool
        segmenter._pool = sam3_backend.ProcessorPool([many])
        try:
            async with _asgi_client() as http:
                resp = await http.post(
                    '/segment',
                    json={
                        'crop_jpeg_b64': base64.b64encode(_make_jpeg()).decode('ascii'),
                        'text_prompt': _PROMPT,
                        'min_score': 0.5,
                    },
                )
        finally:
            segmenter._pool = previous

        assert resp.status_code == 200
        assert [c['score'] for c in resp.json()['candidates']] == [pytest.approx(0.9)]
        # Restored to what it was before this call (never set) -- no
        # per-instance state leaks onto the next caller.
        assert many.confidence_threshold is None

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('served')
    async def test_max_candidates_at_the_cap_is_accepted_over_the_cap_is_422(self) -> None:
        async with _asgi_client() as http:
            ok = await http.post(
                '/segment',
                json={
                    'crop_jpeg_b64': base64.b64encode(_make_jpeg()).decode('ascii'),
                    'text_prompt': _PROMPT,
                    'max_candidates': sam3_backend.MAX_CANDIDATES_CAP,
                },
            )
            over = await http.post(
                '/segment',
                json={
                    'crop_jpeg_b64': base64.b64encode(_make_jpeg()).decode('ascii'),
                    'text_prompt': _PROMPT,
                    'max_candidates': sam3_backend.MAX_CANDIDATES_CAP + 1,
                },
            )
        assert ok.status_code == 200
        assert over.status_code == 422


# =============================================================================
# Backend units that the HTTP path cannot reach
# =============================================================================


class TestBoxNormalization:
    def test_pixel_space_boxes_are_divided_by_the_image_size(self) -> None:
        assert sam3_backend._bbox_to_norm([64.0, 48.0, 128.0, 96.0], 256, 192) == pytest.approx(
            (0.25, 0.25, 0.5, 0.5)
        )

    def test_normalized_boxes_pass_through(self) -> None:
        assert sam3_backend._bbox_to_norm([0.25, 0.25, 0.5, 0.5], 256, 192) == pytest.approx(
            (0.25, 0.25, 0.5, 0.5)
        )

    def test_inverted_and_out_of_range_boxes_are_canonicalized(self) -> None:
        assert sam3_backend._bbox_to_norm([0.9, 0.9, -0.2, 0.4], 100, 100) == pytest.approx(
            (0.0, 0.4, 0.9, 0.9)
        )

    def test_short_box_is_rejected_as_degenerate(self) -> None:
        assert sam3_backend._bbox_to_norm([0.1, 0.2], 100, 100) == (0.0, 0.0, 0.0, 0.0)


class TestRectangularity:
    def test_a_filled_bbox_scores_one(self) -> None:
        assert sam3_backend._rectangularity(_full_mask(4, 4)) == pytest.approx(1.0)

    def test_a_half_filled_bbox_scores_half(self) -> None:
        mask = np.zeros((4, 4), dtype=np.float32)
        mask[:2, :] = 1.0
        mask[2:, :2] = 1.0  # bbox is the full 4x4; 8+4 of 16 pixels set
        assert sam3_backend._rectangularity(mask) == pytest.approx(0.75)

    def test_an_empty_mask_scores_zero(self) -> None:
        assert sam3_backend._rectangularity(np.zeros((4, 4), dtype=np.float32)) == 0.0


class TestProcessorPool:
    @pytest.mark.asyncio
    async def test_concurrent_callers_get_different_processors(self) -> None:
        pool = sam3_backend.ProcessorPool(['a', 'b'])
        held: list[str] = []
        release = asyncio.Event()

        async def hold() -> None:
            async with pool.acquire() as p:
                held.append(p)
                await release.wait()

        tasks = [asyncio.create_task(hold()) for _ in range(2)]
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        release.set()
        await asyncio.gather(*tasks)

        assert sorted(held) == ['a', 'b']

    @pytest.mark.asyncio
    async def test_a_busy_pool_serializes_rather_than_dropping(self) -> None:
        pool = sam3_backend.ProcessorPool(['only'])
        order: list[str] = []

        async def use(tag: str) -> None:
            async with pool.acquire():
                order.append(f'{tag}:in')
                await asyncio.sleep(0.01)
                order.append(f'{tag}:out')

        await asyncio.gather(use('a'), use('b'))

        # Never interleaved: one holder at a time.
        assert order in (
            ['a:in', 'a:out', 'b:in', 'b:out'],
            ['b:in', 'b:out', 'a:in', 'a:out'],
        )

    @pytest.mark.asyncio
    async def test_an_empty_pool_refuses_rather_than_dividing_by_zero(self) -> None:
        pool = sam3_backend.ProcessorPool([])
        assert not pool
        with pytest.raises(RuntimeError, match='empty'):
            async with pool.acquire():
                pass  # pragma: no cover


# =============================================================================
# W5: opt-in mask polygons
# =============================================================================


def _disc_mask(h: int = 64, w: int = 96) -> np.ndarray:
    """A filled circle in the left half of an h x w mask, plus a small blob
    elsewhere (the polygon must be the LARGEST external contour)."""
    yy, xx = np.mgrid[0:h, 0:w]
    mask = ((yy - 32) ** 2 + (xx - 24) ** 2 <= 20**2).astype(np.float32)
    mask[58:62, 80:90] = 1.0
    return mask[None]


async def _segment(http: httpx.AsyncClient, **extra: Any) -> dict[str, Any]:
    resp = await http.post(
        '/segment',
        json={
            'crop_jpeg_b64': base64.b64encode(_make_jpeg()).decode('ascii'),
            'text_prompt': _PROMPT,
            **extra,
        },
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


class TestMaskPolygons:
    @pytest.mark.asyncio
    async def test_return_masks_serves_a_normalized_bounded_polygon(self) -> None:
        proc = _FakeProcessor(boxes=[[0.1, 0.2, 0.5, 0.8]], scores=[0.9], masks=_disc_mask())
        previous = segmenter._pool
        segmenter._pool = sam3_backend.ProcessorPool([proc])
        try:
            async with _asgi_client() as http:
                body = await _segment(http, return_masks=True)
        finally:
            segmenter._pool = previous

        polygon = body['candidates'][0]['mask_polygon']
        assert 3 <= len(polygon) <= 256
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
        assert all(0.0 <= v <= 1.0 for v in xs + ys)
        # The disc (centre 24/96, 32/64; radius 20px) and not the small blob
        # at x ~ 0.9: the largest external contour only.
        assert min(xs) == pytest.approx(4 / 96, abs=0.03)
        assert max(xs) == pytest.approx(44 / 96, abs=0.03)
        assert max(xs) < 0.6
        assert min(ys) == pytest.approx(12 / 64, abs=0.03)
        assert max(ys) == pytest.approx(52 / 64, abs=0.03)

    @pytest.mark.asyncio
    async def test_the_polygon_is_not_computed_unless_asked_for(self) -> None:
        proc = _FakeProcessor(boxes=[[0.1, 0.2, 0.5, 0.8]], scores=[0.9], masks=_disc_mask())
        previous = segmenter._pool
        segmenter._pool = sam3_backend.ProcessorPool([proc])
        try:
            async with _asgi_client() as http:
                body = await _segment(http)
        finally:
            segmenter._pool = previous

        cand = body['candidates'][0]
        assert cand['mask_polygon'] is None
        assert cand['mask_iou'] is not None  # the rectangularity is unaffected

    @pytest.mark.asyncio
    async def test_return_masks_with_the_mask_head_off_is_null_not_fabricated(self) -> None:
        proc = _FakeProcessor(boxes=[[0.1, 0.2, 0.5, 0.8]], scores=[0.9], masks=None)
        previous = segmenter._pool
        segmenter._pool = sam3_backend.ProcessorPool([proc])
        try:
            async with _asgi_client() as http:
                body = await _segment(http, return_masks=True)
        finally:
            segmenter._pool = previous

        assert body['candidates'][0]['mask_polygon'] is None

    def test_a_ragged_mask_is_coarsened_to_the_point_ceiling_never_truncated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        h, w = 400, 400
        yy, xx = np.mgrid[0:h, 0:w]
        theta = np.arctan2(yy - 200, xx - 200)
        radius = 120 + 60 * np.sin(37 * theta)
        mask = (np.hypot(yy - 200, xx - 200) <= radius).astype(np.float32)[None]

        detailed = sam3_backend.mask_polygon(mask)
        assert detailed is not None
        assert len(detailed) > 40, 'the fixture must need more points than the ceiling below'
        monkeypatch.setattr(sam3_backend, '_POLYGON_MAX_POINTS', 24)

        coarse = sam3_backend.mask_polygon(mask)

        assert coarse is not None
        assert 3 <= len(coarse) <= 24
        # Coarsened, not cut off: it still spans the whole star.
        xs = [p[0] for p in coarse]
        assert max(xs) - min(xs) > 0.5

    def test_the_shipped_ceiling_is_256(self) -> None:
        assert sam3_backend._POLYGON_MAX_POINTS == 256

    def test_an_empty_mask_has_no_polygon(self) -> None:
        assert sam3_backend.mask_polygon(np.zeros((1, 8, 8), dtype=np.float32)) is None
