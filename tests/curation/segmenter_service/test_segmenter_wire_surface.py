"""The shipped segmenter wire surface: routes, prompts, sorting, base64 errors and score filtering."""

from __future__ import annotations

import base64

import pytest

from curation.segmenter_service._helpers import (
    _PROMPT,
    _asgi_client,
    _FakeProcessor,
    _make_jpeg,
    sam3_backend,
    segmenter,
)


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
# min_score + the raised max_candidates ceiling
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
