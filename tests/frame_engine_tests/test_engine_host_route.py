"""The engine's host decode route: images nvJPEG does not take (PNG, progressive and CMYK JPEG, a
JPEG below the size gate) are decoded on the host and uploaded, then every branch runs on that
frame (``frame_engine/engine.py``, ``frame_engine/host_decode.py``). Against the stand-ins of
``test_engine.py`` for CuPy, nvJPEG, the kernels and Triton."""

from __future__ import annotations

import io
from typing import Any

import numpy as np
import pytest

from frame_engine import host_decode

from . import test_engine as te
from .test_engine import EVENTS, MIB, H, W, _colour_counts, _Dev, _Lanes, _run, _seen


eng = te.eng  # the engine fixture of test_engine.py, collected here by name


def _encoded(fmt: str, w: int = W, h: int = H, **kw: Any) -> bytes:
    from PIL import Image

    rng = np.random.default_rng(w * h)
    img = Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, fmt, **kw)
    return buf.getvalue()


def _cmyk_jpeg() -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new('CMYK', (W, H), (10, 20, 30, 40)).save(buf, 'JPEG')
    return buf.getvalue()


HOST_CASES = {
    'not_jpeg': lambda: _encoded('PNG'),
    'progressive': lambda: _encoded('JPEG', progressive=True),
    'components': _cmyk_jpeg,
}


@pytest.mark.parametrize('reason', sorted(HOST_CASES))
def test_an_image_nvjpeg_does_not_take_runs_every_branch_on_the_host_route(
    eng: Any, reason: str
) -> None:
    engine = eng()
    seen_in_use: list[int] = []
    real_submit = engine.transport.submit

    def submit(*a: Any, **k: Any) -> Any:
        seen_in_use.append(engine.budget.in_use)
        return real_submit(*a, **k)

    engine.transport.submit = submit
    r = _run(engine, data=HOST_CASES[reason]())
    assert (r.status, r.decode_route) == ('ok', f'cpu_{reason}')
    assert r.branches == dict.fromkeys(('detect', 'embed_frame', 'embed_crops'), ('ok', ''))
    assert r.outputs['size'].tolist() == [W, H]
    assert r.outputs['crop_embeddings'].shape == (2, 8)
    assert len(r.outputs['crop_jpegs']) == 2
    assert _Lanes.last is None or _Lanes.seen_streams == []  # nvJPEG was never asked
    # the frame is charged like a GPU frame: width x height x 3, no oriented copy next to it
    assert set(seen_in_use) == {W * H * 3}
    assert (engine.budget.in_use, engine.transport.staged) == (0, 0)
    assert engine.transport.models()[0] == 'det'


def test_a_small_jpeg_takes_the_host_route_with_reason_small(eng: Any) -> None:
    engine = eng(min_megapixels=1.0)
    r = _run(engine)  # 96 x 64 is far below one megapixel
    assert (r.status, r.decode_route) == ('ok', 'cpu_small')
    assert _Lanes.seen_streams == []


def test_a_photo_nvjpeg_takes_is_still_decoded_on_the_gpu(eng: Any) -> None:
    engine = eng()
    assert _run(engine).decode_route == 'gpu'
    assert len(_Lanes.seen_streams) == 1
    assert not [e for e in EVENTS if e[0] == 'set']


def test_the_host_frame_is_the_decode_images_pixels_upright(eng: Any) -> None:
    from PIL import Image, ImageOps

    exif = Image.Exif()
    exif[0x0112] = 6
    data = _encoded('PNG', exif=exif.tobytes())
    engine = eng()
    uploaded: list[np.ndarray] = []
    real_set = _Dev.set

    def spy(self: Any, arr: Any, stream: Any = None) -> None:
        uploaded.append(np.array(arr))
        real_set(self, arr, stream)

    _Dev.set = spy  # type: ignore[method-assign]
    try:
        r = _run(engine, data=data)
    finally:
        _Dev.set = real_set  # type: ignore[method-assign]
    assert r.outputs['size'].tolist() == [H, W]  # turned by orientation 6
    (frame,) = uploaded
    assert np.array_equal(frame, np.asarray(ImageOps.exif_transpose(Image.open(io.BytesIO(data)))))


def test_a_cpu_decoded_frame_is_not_converted_to_srgb_a_second_time(eng: Any) -> None:
    """The host decode converts an RGB profile itself (``decode_image``); the engine's colour step
    is for nvJPEG frames only. Applied again, a Display P3 frame would move twice."""
    from pathlib import Path

    from PIL import Image, ImageCms

    p3 = (
        Path(__file__).resolve().parents[1] / 'fixtures' / 'icc' / 'DisplayP3-v4.icc'
    ).read_bytes()
    rng = np.random.default_rng(5)
    raw = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(raw).save(buf, 'PNG', icc_profile=p3)
    once = np.asarray(ImageCms.profileToProfile(
        Image.fromarray(raw), ImageCms.ImageCmsProfile(io.BytesIO(p3)),
        ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')), outputMode='RGB',
    ))  # fmt: skip
    assert np.abs(once.astype(int) - raw).max() > 8  # the profile does move these pixels
    engine = eng()
    uploaded: list[np.ndarray] = []
    real_set = _Dev.set

    def spy(self: Any, arr: Any, stream: Any = None) -> None:
        uploaded.append(np.array(arr))
        real_set(self, arr, stream)

    _Dev.set = spy  # type: ignore[method-assign]
    try:
        r = _run(engine, data=buf.getvalue())
    finally:
        _Dev.set = real_set  # type: ignore[method-assign]
    assert r.decode_route == 'cpu_not_jpeg'
    (frame,) = uploaded
    assert np.array_equal(frame, once)  # converted once, by the host decode
    assert eng.cp.gathers == 0
    assert engine.colour.table_bytes == 0
    assert _colour_counts(engine) == {}


def test_a_host_frame_is_allocated_on_the_arena_stream_and_uploaded_on_the_workers_own(
    eng: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The crash this guards came from a device array written on one worker's stream, read on
    another and freed while a queued kernel still read it. The nvJPEG route allocates on the arena
    stream and every branch reads on the worker stream; the host route does the same: allocate on
    ``alloc_stream``, copy in on the worker stream, wait for that stream before the host array is
    dropped, before the first model call, and before the frame returns to the arena."""
    engine = eng()
    real_release = engine.budget.release
    real_submit = engine.transport.submit

    def release(n: int) -> None:
        EVENTS.append(('release',))
        real_release(n)

    def submit(*a: Any, **k: Any) -> Any:
        EVENTS.append(('call',))
        return real_submit(*a, **k)

    engine.budget.release = release
    engine.transport.submit = submit
    real_decode = host_decode.decode

    class _Host(np.ndarray):
        def __del__(self) -> None:
            EVENTS.append(('host_dropped',))

    monkeypatch.setattr(host_decode, 'decode', lambda d: real_decode(d).view(_Host))
    _run(engine, data=HOST_CASES['not_jpeg']())
    (alloc,) = [e for e in EVENTS if e[0] == 'alloc' and e[2] == W * H * 3]
    (copy,) = [e for e in EVENTS if e[0] == 'set']
    assert alloc[1] is engine.alloc_stream  # the arena every worker's frames come from
    worker = copy[1]
    assert worker is not None
    assert worker is not engine.alloc_stream  # copied in on the worker's own stream
    wait = ('sync', worker)
    at = {name: [k for k, e in enumerate(EVENTS) if e[:1] == (name,)] for name in
          ('set', 'host_dropped', 'call', 'release')}  # fmt: skip
    syncs = [k for k, e in enumerate(EVENTS) if e == wait]
    # waited for on the worker stream after the copy and before the host array is dropped
    assert any(at['set'][0] < k < at['host_dropped'][0] for k in syncs)
    # and again after the last model call, immediately before the budget (the frame) is released
    assert at['call'][-1] < syncs[-1] < at['release'][-1]
    assert EVENTS[at['release'][-1] - 1] == wait


def test_a_huge_png_is_refused_by_its_header_before_it_is_decoded(
    eng: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = eng(max_megapixels=0.1)  # 400 x 300 is 0.12 MP
    monkeypatch.setattr(
        host_decode, 'decode', lambda _d: pytest.fail('decoded an image that is too large')
    )
    r = _run(engine, data=_encoded('PNG', 400, 300))
    assert (r.status, r.decode_route) == ('huge', 'none')
    assert engine.budget.in_use == 0
    assert 'budget' not in {s for s, n in _seen(engine).items() if n}


def test_a_header_that_lies_about_the_pixel_count_answers_decoder_and_frees_the_budget(
    eng: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = eng()
    monkeypatch.setattr(host_decode, 'decode', lambda _d: np.zeros((H, W + 1, 3), np.uint8))
    r = _run(engine, data=HOST_CASES['not_jpeg']())
    assert (r.status, r.decode_route) == ('decoder', 'none')
    assert engine.budget.in_use == 0
    assert not [e for e in EVENTS if e[0] == 'set' or (e[:1] == ('alloc',) and e[2] < MIB * 64)]


def test_a_host_frame_larger_than_the_whole_budget_answers_over_budget(eng: Any) -> None:
    engine = eng(budget_mib=1)
    big = _encoded('PNG', 800, 600)  # 1.44 MB of frame, over the 1 MiB budget
    r = _run(engine, data=big)
    assert (r.status, r.decode_route) == ('over_budget', 'none')
    assert engine.budget.in_use == 0


def test_a_failed_upload_allocation_answers_gpu_memory_and_frees_the_budget(eng: Any) -> None:
    engine = eng()
    eng.cp.alloc_limit = 1000  # the frame (96 x 64 x 3) does not fit
    r = _run(engine, data=HOST_CASES['not_jpeg']())
    assert (r.status, r.decode_route) == ('gpu_memory', 'none')
    assert engine.budget.in_use == 0


def test_an_undecodable_image_that_passes_the_header_answers_decoder(eng: Any) -> None:
    data = HOST_CASES['not_jpeg']()
    truncated = data[: len(data) // 2]  # the header reads, the pixel data does not
    engine = eng()
    r = _run(engine, data=truncated)
    assert (r.status, r.decode_route) == ('decoder', 'none')
    assert engine.budget.in_use == 0
