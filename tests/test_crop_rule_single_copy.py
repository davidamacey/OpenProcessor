"""The crop encoder's input is cut by ONE rule, the letterbox, and every path uses it.

A crop is the whole box scaled to fit the encoder's square and padded with the plan's fill. The
centre-crop rule it replaced (shorter edge to the size, centre square) lost about half of a wide
box and is gone from every path that prepares a crop: the engine's kernels
(``frame_engine/branches/embed.py``), the API's host copy (``src/utils/pe_preprocess.py``) and the
callers of both. These tests fail if it comes back.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from PIL import Image

from src.clients.pe_encoder import PE_CROP_MODEL, PE_CROP_SIZE, PE_IMAGE_MODEL, PEEncoder
from src.domain import frame_engine_wire as wire
from src.domain.image_geometry import CROP_FILL, letterbox_geometry
from src.utils import pe_preprocess
from src.utils.pe_preprocess import letterbox_crop_rgb


ROOT = Path(__file__).resolve().parent.parent
#: The trees that ship: nothing here may cut a crop square the old way.
PRODUCT = ('src', 'frame_engine', 'docker', 'scripts/curation')


def _python_files() -> list[Path]:
    return [p for d in PRODUCT for p in sorted((ROOT / d).rglob('*.py'))]


def _crop(h: int, w: int, seed: int = 1) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


def test_the_centre_crop_helper_and_recipe_are_gone() -> None:
    assert not hasattr(pe_preprocess, 'resize_crop_rgb')
    assert wire.CROP_RECIPES == ('letterbox',)
    with pytest.raises(wire.EnginePlanError):
        wire.parse_plan(
            '{"version":1,"branches":["embed_frame","embed_crops"],"embed":{"model":"a","size":336},'
            '"crop_embed":{"model":"b","size":224,"batch":8,"crop_recipe":"resize_crop"}}',
            has_boxes=True,
        )


def test_no_product_file_names_the_retired_helper_or_recipe() -> None:
    offenders = [
        str(p.relative_to(ROOT))
        for p in _python_files()
        if re.search(r'\bresize_crop(_rgb)?\b', p.read_text())
    ]
    assert offenders == []


def test_the_centre_square_geometry_serves_the_frame_only() -> None:
    """``whole_square_geometry`` is the whole-frame square; a crop never calls it."""
    callers = set()
    for p in _python_files():
        tree = ast.parse(p.read_text())
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and getattr(node.func, 'id', None) == 'whole_square_geometry'
            ):
                callers.add((str(p.relative_to(ROOT)), node.lineno))
    files = {f for f, _ in callers}
    assert files == {'frame_engine/branches/embed.py'}, files
    source = (ROOT / 'frame_engine/branches/embed.py').read_text()
    frame_part, _, crop_part = source.partition('class EmbedCrops')
    assert 'whole_square_geometry(' in frame_part
    assert 'whole_square_geometry(' not in crop_part
    assert 'letterbox_geometry(' in crop_part


@pytest.mark.parametrize('size', [(200, 20), (20, 200), (3, 2), (1, 1), (64, 64), (333, 97)])
def test_the_host_square_is_the_letterbox_of_the_whole_crop(size: tuple[int, int]) -> None:
    w, h = size
    crop = _crop(h, w)
    sq = letterbox_crop_rgb(crop, 224)
    g = letterbox_geometry(w, h, 224)
    assert sq.shape == (224, 224, 3)
    inside = sq[g.top : g.top + g.new_h, g.left : g.left + g.new_w]
    outside = np.ones(sq.shape[:2], bool)
    outside[g.top : g.top + g.new_h, g.left : g.left + g.new_w] = False
    assert (sq[outside] == CROP_FILL).all()
    fitted = Image.fromarray(crop).resize((g.new_w, g.new_h), Image.Resampling.BILINEAR)
    np.testing.assert_array_equal(
        inside, np.asarray(fitted)
    )  # the whole box, aspect kept, nothing cut
    assert max(g.new_w, g.new_h) == 224 or max(w, h) == 1


def test_a_wide_box_keeps_both_ends_and_a_box_without_pixels_is_the_fill() -> None:
    crop = np.zeros((10, 400, 3), np.uint8)
    crop[:, :20] = (255, 0, 0)
    crop[:, -20:] = (0, 255, 0)
    sq = letterbox_crop_rgb(crop, 224)
    row = sq[112]
    assert row[:, 0].max() == 255  # the left end, red
    assert row[:, 1].max() == 255  # the right end, green
    assert (letterbox_crop_rgb(np.zeros((0, 5, 3), np.uint8), 32) == CROP_FILL).all()


@pytest.mark.asyncio
async def test_embed_crops_letterboxes_and_calls_the_crop_model() -> None:
    """The API's crop path (drawn objects, geometry edits, reprocess, re-index, search by
    picture) sends letterboxed squares of the crop encoder's size to the crop model."""
    pool = MagicMock()
    result = MagicMock()
    result.as_numpy = MagicMock(return_value=np.ones((2, 1024), np.float32))
    pool.infer = AsyncMock(return_value=result)
    crops = [_crop(10, 400, 1), _crop(300, 30, 2)]
    await PEEncoder(triton_pool=pool).embed_crops(crops)
    model, (inp,) = pool.infer.call_args.args[0], pool.infer.call_args.args[1]
    assert model == PE_CROP_MODEL != PE_IMAGE_MODEL
    assert inp.shape() == [2, 3, PE_CROP_SIZE, PE_CROP_SIZE]
    sent = np.frombuffer(inp._raw_content, np.uint8).reshape(inp.shape())
    for k, crop in enumerate(crops):
        np.testing.assert_array_equal(
            sent[k], letterbox_crop_rgb(crop, PE_CROP_SIZE).transpose(2, 0, 1)
        )


@pytest.mark.asyncio
async def test_the_frame_path_still_uses_the_frame_encoder() -> None:
    pool = MagicMock()
    result = MagicMock()
    result.as_numpy = MagicMock(return_value=np.ones((1, 1024), np.float32))
    pool.infer = AsyncMock(return_value=result)
    await PEEncoder(triton_pool=pool).encode_images(np.zeros((1, 3, 336, 336), np.uint8))
    assert pool.infer.call_args.args[0] == PE_IMAGE_MODEL
