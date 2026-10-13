"""Ingest decodes a photo once: the whole-frame square, the letterbox, the crops and the blur
sums all read the one decoded frame (#217, owner decisions D1-D3)."""

from __future__ import annotations

import asyncio
import io
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image, ImageOps

from src.services.detection.pe_preprocess import PE_SIZE


def _jpeg(width: int, height: int, orientation: int | None = None) -> bytes:
    rng = np.random.default_rng(width * 7 + height)
    pixels = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(pixels)
    buf = io.BytesIO()
    if orientation is None:
        img.save(buf, 'JPEG', quality=90)
    else:
        exif = Image.Exif()
        exif[0x0112] = orientation
        img.save(buf, 'JPEG', quality=90, exif=exif.tobytes())
    return buf.getvalue()


def _reference_square(img: Image.Image, target: int = PE_SIZE) -> np.ndarray:
    """Shorter edge to ``target`` (Pillow BILINEAR, antialiased) on the whole frame, centre crop."""
    w, h = img.size
    scale = target / min(w, h)
    nw, nh = max(target, round(w * scale)), max(target, round(h * scale))
    full = np.asarray(img.resize((nw, nh), Image.BILINEAR))
    top, left = (nh - target) // 2, (nw - target) // 2
    return full[top : top + target, left : left + target]


@pytest.mark.parametrize('size', [(1600, 1200), (900, 1400), (640, 480), (200, 150), (336, 336)])
def test_whole_frame_square_is_the_full_frame_resized_then_centre_cropped(size) -> None:
    from src.services.detection.pe_preprocess import whole_frame_rgb_from_image

    img = Image.open(io.BytesIO(_jpeg(*size))).convert('RGB')
    out = whole_frame_rgb_from_image(img)
    assert out.shape == (PE_SIZE, PE_SIZE, 3)
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out, _reference_square(img))


def test_whole_frame_from_a_file_applies_exif_and_matches_the_decoded_frame(tmp_path) -> None:
    from src.services.detection.pe_preprocess import whole_frame_chw, whole_frame_rgb_from_image

    data = _jpeg(800, 500, orientation=6)
    path = tmp_path / 'rotated.jpg'
    path.write_bytes(data)
    upright = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert('RGB')
    assert upright.size == (500, 800)
    chw = whole_frame_chw(str(path))
    assert chw is not None
    np.testing.assert_array_equal(chw.transpose(1, 2, 0), whole_frame_rgb_from_image(upright))


def _pillow_reference(data: bytes) -> np.ndarray:
    return np.asarray(ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert('RGB'))


@pytest.mark.parametrize('orientation', [None, *range(1, 9)])
def test_jpeg_decode_matches_pillow_and_returns_the_array(orientation) -> None:
    from src.services.curation import ingest

    data = _jpeg(320, 200, orientation=orientation)
    decoded = ingest._decode_image(data)
    expected = _pillow_reference(data)
    assert decoded.rgb is not None
    np.testing.assert_array_equal(decoded.rgb, expected)
    np.testing.assert_array_equal(np.asarray(decoded.pil), expected)
    assert decoded.pil.mode == 'RGB'


@pytest.mark.parametrize('kind', ['progressive', 'grey', 'cmyk', 'png'])
def test_other_encodings_decode_as_pillow_does(kind) -> None:
    from src.services.curation import ingest

    rng = np.random.default_rng(5)
    img = Image.fromarray(rng.integers(0, 256, (90, 120, 3), dtype=np.uint8))
    buf = io.BytesIO()
    if kind == 'progressive':
        img.save(buf, 'JPEG', quality=90, progressive=True)
    elif kind == 'grey':
        img.convert('L').save(buf, 'JPEG', quality=90)
    elif kind == 'cmyk':
        img.convert('CMYK').save(buf, 'JPEG', quality=90)
    else:
        img.save(buf, 'PNG')
    data = buf.getvalue()
    decoded = ingest._decode_image(data)
    np.testing.assert_array_equal(np.asarray(decoded.pil), _pillow_reference(data))
    if decoded.rgb is not None:
        np.testing.assert_array_equal(decoded.rgb, _pillow_reference(data))


def test_a_truncated_jpeg_is_refused() -> None:
    from src.services.curation import ingest

    data = _jpeg(320, 200)
    with pytest.raises(ValueError, match='Premature end of JPEG file'):
        ingest._decode_image(data[: len(data) // 2])


def test_index_items_embeds_the_whole_frame_from_the_decoded_frame(tmp_path) -> None:
    from src.services.curation.ingest_index import ImageContext, index_items
    from src.services.curation.ingest_policy import IngestPolicy
    from src.services.curation.item_doc import DetectedItem

    seen: list[Image.Image] = []

    class _Encoder:
        async def embed_crops(self, crops, **_kw):
            return np.ones((len(crops), 1024), dtype=np.float32)

        async def embed_whole_frame_image(self, img):
            seen.append(img)
            return np.ones(1024, dtype=np.float32)

    async def _bulk_index(image_doc, crop_docs, created_ids, *, seed_region=True):
        return {'crops_created': len(crop_docs)}

    service = SimpleNamespace(
        policy=IngestPolicy(),
        profile=SimpleNamespace(batch_limit=16),
        pe_encoder=_Encoder(),
        config=SimpleNamespace(crop_cache_dir=str(tmp_path / 'crops'), crop_cache_max_bytes=0),
        region_seed_status=None,
        _bulk_index=_bulk_index,
        _publish_created=lambda *_a, **_k: None,
    )
    frame = Image.open(io.BytesIO(_jpeg(128, 96))).convert('RGB')
    ctx = ImageContext(
        image_id='img',
        image_path='/client/side/name.jpg',
        created=True,
        pil=frame,
        width=128,
        height=96,
        imohash='h',
        source='t',
    )
    items = [DetectedItem(bbox_pixel=(10.0, 10.0, 60.0, 50.0), score=0.9)]
    outcome = asyncio.run(index_items(service, ctx, items))  # type: ignore[arg-type]
    assert outcome.result.status == 'success'
    assert seen == [frame]
    assert seen[0] is frame


def test_embed_whole_frame_image_sends_the_square_of_the_frame() -> None:
    from src.clients.pe_encoder import PEEncoder
    from src.services.detection.pe_preprocess import whole_frame_rgb_from_image

    sent: list[np.ndarray] = []

    async def _encode(batch):
        sent.append(batch)
        return np.ones((len(batch), 1024), dtype=np.float32)

    frame = Image.open(io.BytesIO(_jpeg(700, 400))).convert('RGB')
    encoder = PEEncoder(triton_pool=SimpleNamespace())  # type: ignore[arg-type]
    encoder.encode_images = _encode  # type: ignore[method-assign]
    vector = asyncio.run(encoder.embed_whole_frame_image(frame))
    assert vector is not None
    assert vector.shape == (1024,)
    assert len(sent) == 1
    assert sent[0].shape == (1, 3, PE_SIZE, PE_SIZE)
    np.testing.assert_array_equal(sent[0][0], whole_frame_rgb_from_image(frame).transpose(2, 0, 1))
