"""The embed pass works on several images at once so their crops share engine batches (#216)."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from curation.reprocess_fixtures import docs, item, jpeg_bytes, make_fake, servable_root
from src.services.curation import reprocess_embed
from src.services.curation.reprocess_embed import EmbedTarget, reembed_items


if TYPE_CHECKING:
    from pathlib import Path

BOX = (0.1, 0.1, 0.6, 0.6)


def _image(path: str) -> Any:
    """The frame the CPU route of the pass decodes for ``path``."""
    return reprocess_embed._decode_image(reprocess_embed._read_image(path) or b'', path)


class TaggingPE:
    """An encoder that answers a crop with a vector derived from its pixels and counts overlap."""

    def __init__(self, fail_on_path_tag: float | None = None) -> None:
        self.inflight = 0
        self.peak = 0
        self.frame_inflight = 0
        self.fail_tag = fail_on_path_tag

    @staticmethod
    def vector(crop: np.ndarray) -> np.ndarray:
        return np.array([float(crop.mean()) / 255.0, 1.0, 0.0], dtype=np.float32)

    async def embed_crops(self, crops: list[np.ndarray], max_batch: int = 32) -> np.ndarray:  # noqa: ARG002
        self.inflight += 1
        self.peak = max(self.peak, self.inflight)
        try:
            await asyncio.sleep(0.02)
            if self.fail_tag is not None and any(
                abs(float(c.mean()) / 255.0 - self.fail_tag) < 1e-6 for c in crops
            ):
                raise RuntimeError('encoder failed')
            return np.stack([self.vector(c) for c in crops])
        finally:
            self.inflight -= 1

    async def embed_whole_frame_image(self, img: Any) -> np.ndarray:  # noqa: ARG002
        await asyncio.sleep(0.01)
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)


def _world(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, n: int = 8
) -> tuple[Any, list[EmbedTarget]]:
    root = servable_root(tmp_path, monkeypatch)
    items = []
    targets = []
    for i in range(n):
        path = root / f'im{i}.jpg'
        path.write_bytes(jpeg_bytes(seed=i))
        doc = item(
            f'c{i}',
            image_id=f'img-{i}',
            image_path=str(path),
            bbox_norm=BOX,
            class_id=1,
            class_name='x',
        )
        items.append(doc)
        targets.append(
            EmbedTarget(image_id=f'img-{i}', image_path=str(path), items=[(f'c{i}', doc)])
        )
    return make_fake(
        items, [{'image_id': f'img-{i}', 'image_path': t.image_path} for i, t in enumerate(targets)]
    ), targets


@pytest.mark.asyncio
async def test_images_are_embedded_concurrently_and_each_item_gets_its_own_vector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake, targets = _world(tmp_path, monkeypatch)
    pe = TaggingPE()
    counts = await reembed_items(fake, pe, targets, parts=frozenset({'crop'}))
    assert counts['crop_written'] == 8
    assert pe.peak > 1  # one image at a time would leave the engine batch at one image's crops
    stored = docs(fake)
    for i, target in enumerate(targets):
        img = _image(target.image_path)
        crop = np.asarray(reprocess_embed.crop_pil(img, reprocess_embed._pixel_box(BOX, *img.size)))
        np.testing.assert_allclose(stored[f'c{i}']['pe_embedding'], pe.vector(crop), atol=1e-6)


@pytest.mark.asyncio
async def test_one_images_encoder_failure_stops_the_pass_like_before(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake, targets = _world(tmp_path, monkeypatch, n=3)
    img = _image(targets[1].image_path)
    tag = TaggingPE.vector(
        np.asarray(reprocess_embed.crop_pil(img, reprocess_embed._pixel_box(BOX, *img.size)))
    )[0]
    with pytest.raises(RuntimeError, match='encoder failed'):
        await reembed_items(
            fake, TaggingPE(fail_on_path_tag=float(tag)), targets, parts=frozenset({'crop'})
        )


def test_a_reprocess_chunk_holds_several_embed_windows() -> None:
    from src.services.curation import reprocess_images

    assert reprocess_images.CHUNK >= 2 * reprocess_embed._IMAGES_IN_FLIGHT
