"""Loader for the ``docker/segmenter`` modules, the fake SAM 3 and the in-process ASGI client."""

from __future__ import annotations

import base64
import importlib.util
import io
import sys
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from PIL import Image


_SEGMENTER_DIR = Path(__file__).resolve().parents[3] / 'docker' / 'segmenter'


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
        # mirrors the real Sam3Processor's plain instance attribute
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


def _asgi_client() -> httpx.AsyncClient:
    """An httpx client wired straight into the segmenter's ASGI app."""
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=segmenter.app),
        base_url='http://segmenter',
        timeout=10.0,
    )


# =============================================================================
# The cascade leg itself
# =============================================================================


def _vlm_mock(*, is_region: bool):
    from unittest.mock import AsyncMock, MagicMock

    from src.services.labeling.vlm_models import VlmRegionVerdict

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


# =============================================================================
# opt-in mask polygons
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
