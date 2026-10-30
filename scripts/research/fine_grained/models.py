"""Frozen image encoders for the benchmark, loaded in the research venv (torch, timm, open_clip).

Each :class:`Encoder` takes a batch of HWC uint8 squares of its own ``size`` and returns L2
normalised float32 vectors. Weights come from public Hugging Face repositories; nothing is
installed into the product. fp16 autocast on the GPU.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from scripts.research.fine_grained.data import env_path


if TYPE_CHECKING:
    from collections.abc import Callable

REPO = Path(__file__).resolve().parents[3]
HALF = (0.5, 0.5, 0.5)
IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


@dataclass(frozen=True)
class Spec:
    name: str
    size: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    source: str


SPECS = {
    's_pe_core': Spec('pe_core_l14_336', 336, HALF, HALF, 'timm/vit_pe_core_large_patch14_336.fb'),
    's_mobileclip2': Spec(
        'mobileclip2_s2',
        256,
        (0, 0, 0),
        (1, 1, 1),
        'FG_MOBILECLIP_CKPT (the MobileCLIP2-S2 checkpoint the product export uses)',
    ),
    's_dinov2': Spec('dinov2_l14', 224, *IMAGENET, 'timm/vit_large_patch14_dinov2.lvd142m'),
    's_siglip2': Spec(
        'siglip2_so400m_p16_256', 256, HALF, HALF, 'timm/vit_so400m_patch16_siglip_256.v2_webli'
    ),
    's_pe_core_g': Spec(
        'pe_core_g14_448', 448, HALF, HALF, 'timm/vit_pe_core_gigantic_patch14_448.fb'
    ),
}


class Encoder:
    def __init__(self, spec: Spec, forward: Callable[[Any], Any], device: str) -> None:
        import torch

        self.spec, self._forward, self.device = spec, forward, device
        self._mean = torch.tensor(spec.mean, device=device).view(1, 3, 1, 1)
        self._std = torch.tensor(spec.std, device=device).view(1, 3, 1, 1)

    def encode(self, batch_hwc_uint8: np.ndarray) -> np.ndarray:
        import torch

        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
            x = torch.from_numpy(batch_hwc_uint8).to(self.device).permute(0, 3, 1, 2).float() / 255
            out = self._forward((x - self._mean) / self._std).float()
            out = torch.nn.functional.normalize(out, dim=-1)
        return np.asarray(out.cpu().numpy(), dtype=np.float32)


def _timm(tag: str, spec: Spec, device: str) -> Encoder:
    import timm

    model = timm.create_model(tag, pretrained=True, img_size=spec.size).to(device).train(False)
    return Encoder(spec, model, device)


def _mobileclip(spec: Spec, device: str) -> Encoder:
    import open_clip

    sys.path.insert(0, str(REPO / 'export'))
    from utils import reparameterize_model

    ckpt = env_path('FG_MOBILECLIP_CKPT')
    model, _, _ = open_clip.create_model_and_transforms(
        'MobileCLIP2-S2', pretrained=str(ckpt), image_mean=(0, 0, 0), image_std=(1, 1, 1)
    )
    model = reparameterize_model(model.train(False)).to(device).train(False)
    return Encoder(spec, cast('Callable[[Any], Any]', model.encode_image), device)


def load(key: str, device: str = 'cuda') -> Encoder:
    spec = SPECS[key]
    if key == 's_mobileclip2':
        return _mobileclip(spec, device)
    return _timm(spec.source.removeprefix('timm/'), spec, device)
