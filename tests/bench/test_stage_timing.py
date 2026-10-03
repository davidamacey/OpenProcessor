"""The shared per-stage timer: fake clock, metric output, no behaviour change."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image
from prometheus_client import REGISTRY

from src.utils import stage_timing


def _hist_sum(stage: str) -> float:
    return REGISTRY.get_sample_value('op_pipeline_stage_seconds_sum', {'stage': stage}) or 0.0


def _hist_count(stage: str) -> float:
    return REGISTRY.get_sample_value('op_pipeline_stage_seconds_count', {'stage': stage}) or 0.0


def _bytes(stage: str) -> float:
    return REGISTRY.get_sample_value('op_pipeline_stage_bytes_total', {'stage': stage}) or 0.0


def test_timer_observes_elapsed_from_the_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    ticks = iter([10.0, 10.25])
    monkeypatch.setattr(stage_timing, '_clock', lambda: next(ticks))
    before_sum, before_count = _hist_sum('decode'), _hist_count('decode')
    with stage_timing.stage_timer('decode'):
        pass
    assert _hist_sum('decode') - before_sum == pytest.approx(0.25)
    assert _hist_count('decode') - before_count == 1


def test_timer_counts_bytes_only_when_given(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(stage_timing, '_clock', lambda: 0.0)
    before = _bytes('embed')
    with stage_timing.stage_timer('embed'):
        pass
    assert _bytes('embed') == before
    with stage_timing.stage_timer('embed', nbytes=4096):
        pass
    assert _bytes('embed') - before == 4096


def test_timer_records_on_exception_and_does_not_swallow_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ticks = iter([1.0, 3.0])
    monkeypatch.setattr(stage_timing, '_clock', lambda: next(ticks))
    before = _hist_sum('crop')
    with pytest.raises(ValueError, match='boom'), stage_timing.stage_timer('crop'):
        raise ValueError('boom')
    assert _hist_sum('crop') - before == pytest.approx(2.0)


def test_unknown_stage_is_rejected() -> None:
    with pytest.raises(ValueError, match='unknown stage'):
        stage_timing.stage_timer('not_a_stage')


def test_decode_output_is_unchanged_by_the_timer() -> None:
    from src.services.curation.ingest import _decode_image

    buf = io.BytesIO()
    Image.fromarray(np.arange(48 * 64 * 3, dtype=np.uint8).reshape(48, 64, 3)).save(buf, 'PNG')
    before = _hist_count('decode')
    img, w, h = _decode_image(buf.getvalue())
    assert (w, h, img.mode) == (64, 48, 'RGB')
    assert np.asarray(img)[0, 1, 0] == 3
    assert _hist_count('decode') - before == 1


def test_jpeg_crop_cache_write_is_timed_and_bytes_identical(tmp_path) -> None:
    from src.services.curation.source_image_cache import write_crop_cache

    crop = Image.fromarray(np.full((32, 32, 3), 120, dtype=np.uint8))
    expected = io.BytesIO()
    crop.save(expected, format='JPEG', quality=90)
    before = _hist_count('jpeg_encode')
    write_crop_cache('abc', crop, tmp_path, quality=90)
    assert (tmp_path / 'abc.jpg').read_bytes() == expected.getvalue()
    assert _hist_count('jpeg_encode') - before == 1
    assert _bytes('jpeg_encode') >= len(expected.getvalue())


def test_crop_and_resize_helpers_are_timed() -> None:
    from src.services.curation.ingest_index import crop_pil

    img = Image.new('RGB', (100, 80))
    before = _hist_count('crop')
    out = crop_pil(img, (10.0, 10.0, 50.0, 40.0))
    assert out.size == (40, 30)
    assert _hist_count('crop') - before == 1


def test_embed_crops_times_resize_and_embed_and_keeps_output() -> None:
    import asyncio

    from src.clients.pe_encoder import PEEncoder

    class _Result:
        def __init__(self, n: int) -> None:
            self._n = n

        def as_numpy(self, _name: str) -> np.ndarray:
            return np.tile(np.array([3.0, 4.0] + [0.0] * 1022, dtype=np.float32), (self._n, 1))

    seen_shapes: list[list[int]] = []

    class _Pool:
        async def infer(self, _model, inputs, **_kw):
            seen_shapes.append(list(inputs[0].shape()))
            return _Result(inputs[0].shape()[0])

    encoder = PEEncoder(triton_pool=_Pool())  # type: ignore[arg-type]
    crops = [np.zeros((40, 30, 3), dtype=np.uint8) for _ in range(2)]
    resize_before, embed_before = _hist_count('resize'), _hist_count('embed')
    bytes_before = _bytes('embed')
    out = asyncio.run(encoder.embed_crops(crops))
    assert out.shape == (2, 1024)
    assert out[0, 0] == pytest.approx(0.6)
    assert _hist_count('resize') - resize_before == 1
    assert _hist_count('embed') - embed_before == 1
    assert _bytes('embed') - bytes_before == int(np.prod(seen_shapes[0])) * 4


def test_bulk_index_is_timed() -> None:
    import asyncio
    from types import SimpleNamespace

    from src.services.curation.ingest import CurationIngestService

    class _Client:
        async def bulk(self, **_kw):
            return {'errors': False, 'items': []}

    service = SimpleNamespace(
        opensearch=_Client(), config=SimpleNamespace(images_index='images', items_index='items')
    )
    before = _hist_count('opensearch_write')
    result = asyncio.run(
        CurationIngestService._bulk_index(service, {'image_id': 'i1'}, [])  # type: ignore[arg-type]
    )
    assert result['images_indexed'] == 1
    assert _hist_count('opensearch_write') - before == 1
