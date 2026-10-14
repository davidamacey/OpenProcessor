"""Triton ``config.pbtxt`` rendering for the PE image encoder (pure, unit-tested without a GPU)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pe_image_constants import (
    DEFAULT_MAX_BATCH,
    EMBEDDING_DIM,
    IMAGE_SIZE,
    INPUT_TENSOR,
    OUTPUT_TENSOR,
    PE_GPU_MIB_PER_INSTANCE,
    PE_VARIANT,
    PLATFORMS,
    TRITON_MODEL_NAME,
    TRT_PLATFORM,
)


if TYPE_CHECKING:
    from pathlib import Path


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PETritonConfig:
    """Everything that varies in the rendered ``config.pbtxt``.

    Defaults describe the model ``src/clients/pe_encoder.py`` expects:
    ``pe_image_encoder``, UINT8 ``images`` in, FP32 ``image_embeddings``
    out, dynamic batch up to 32.
    """

    model_name: str = TRITON_MODEL_NAME
    platform: str = TRT_PLATFORM
    max_batch_size: int = DEFAULT_MAX_BATCH
    image_size: int = IMAGE_SIZE
    embedding_dim: int = EMBEDDING_DIM
    input_name: str = INPUT_TENSOR
    output_name: str = OUTPUT_TENSOR
    instance_count: int = 1
    # None: no `gpus` list, so Triton runs an instance on every GPU the container sees
    # (TRITON_GPU_ID in .env decides how many). A tuple pins the instances to those cards.
    gpu_ids: tuple[int, ...] | None = None
    max_queue_delay_us: int = 15000

    def __post_init__(self) -> None:
        if not self.model_name:
            raise ValueError('model_name must be a non-empty string')
        if self.platform not in PLATFORMS:
            raise ValueError(f'platform must be one of {PLATFORMS}, got {self.platform!r}')
        for attr in ('max_batch_size', 'image_size', 'embedding_dim', 'instance_count'):
            value = getattr(self, attr)
            if not isinstance(value, int) or value < 1:
                raise ValueError(f'{attr} must be a positive int, got {value!r}')
        if self.max_queue_delay_us < 0:
            raise ValueError(f'max_queue_delay_us must be >= 0, got {self.max_queue_delay_us}')
        if self.gpu_ids is not None and not self.gpu_ids:
            raise ValueError('gpu_ids must contain at least one device id')
        if any(not isinstance(g, int) or g < 0 for g in self.gpu_ids or ()):
            raise ValueError(f'gpu_ids must be non-negative ints, got {self.gpu_ids!r}')
        if not self.input_name or not self.output_name:
            raise ValueError('input_name and output_name must be non-empty strings')


def preferred_batch_sizes(max_batch: int) -> list[int]:
    """Triton ``preferred_batch_size`` ladder bounded by ``max_batch``."""
    ladder = [size for size in (4, 8, 16, 32, 64) if size <= max_batch]
    return ladder or [max_batch]


def render_config(cfg: PETritonConfig) -> str:
    """Render the Triton ``config.pbtxt`` body for the PE image encoder."""
    preferred = ', '.join(str(b) for b in preferred_batch_sizes(cfg.max_batch_size))
    gpu_pin = (
        f'    gpus: [ {", ".join(str(g) for g in cfg.gpu_ids)} ]\n'
        if cfg.gpu_ids is not None
        else ''
    )
    if cfg.platform == TRT_PLATFORM:
        provenance = (
            '# TensorRT plan built by export/build_pe_trt.sh from the ONNX\n'
            '# produced by export/export_pe_image_encoder.py.\n'
            f'# Engine profile: min=1 / opt=8 / max={cfg.max_batch_size} '
            f'at {cfg.image_size}x{cfg.image_size}.'
        )
    else:
        provenance = (
            '# ONNX served directly by Triton (Path 2 fallback, installed by\n'
            '# export/build_pe_ort_fallback.sh). Use this when the TensorRT\n'
            "# build fails on PE's attention-pool ops. Same numerics, lower\n"
            '# throughput. The ONNX MUST have a dynamic leading dimension or\n'
            '# dynamic_batching below is invalid.'
        )

    return f"""# {PE_VARIANT} image encoder — whole-frame and per-crop embeddings.
#
# Consumed by src/clients/pe_encoder.py (PEEncoder.encode_images); the
# resulting vector is stored as the curation `pe_embedding` field and backs
# semantic search, near-duplicate detection, clustering and the embedding
# visualization. The tensor names below are a contract with that client.
#
# Input:  {cfg.input_name} [B, 3, {cfg.image_size}, {cfg.image_size}] UINT8 RGB pixels,
#         resized and center-cropped (see src/services/detection/pe_preprocess.py);
#         the engine scales them and applies PE-Core's 0.5/0.5 mean/std itself
#         (export/uint8_input.py)
# Output: {cfg.output_name} [B, {cfg.embedding_dim}] FP32, L2-normalized
#
{provenance}

name: "{cfg.model_name}"
platform: "{cfg.platform}"
max_batch_size: {cfg.max_batch_size}

input [
  {{
    name: "{cfg.input_name}"
    data_type: TYPE_UINT8
    dims: [ 3, {cfg.image_size}, {cfg.image_size} ]
  }}
]

output [
  {{
    name: "{cfg.output_name}"
    data_type: TYPE_FP32
    dims: [ {cfg.embedding_dim} ]
  }}
]

dynamic_batching {{
  preferred_batch_size: [ {preferred} ]
  max_queue_delay_microseconds: {cfg.max_queue_delay_us}
}}

# No `gpus` list: Triton places `count` instances on every GPU the container
# sees, so TRITON_GPU_ID in .env decides how many cards encode in parallel
# (one card by default; "0,2" runs two). One dynamic-batcher queue feeds all
# instances. Every other model stays pinned to `gpus: [ 0 ]`, the lowest
# listed card. See INSTALLATION.md, "Spreading the PE encoder over GPUs".
instance_group [
  {{
    count: {cfg.instance_count}
    kind: KIND_GPU
{gpu_pin}  }}
]

# Budget accounting for `openprocessor gpu plan` and the installer: GPU memory
# of one warm instance, measured at max batch 32 (1312 MiB resident on an RTX
# 3080 Ti, the same on an RTX A6000), recorded with headroom.
parameters: {{
  key: "GPU_MIB_PER_INSTANCE"
  value: {{ string_value: "{PE_GPU_MIB_PER_INSTANCE}" }}
}}
"""


def write_triton_config(models_dir: Path, cfg: PETritonConfig) -> Path:
    """Write ``<models_dir>/<model_name>/config.pbtxt`` and return its path."""
    model_dir = models_dir / cfg.model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    config_path = model_dir / 'config.pbtxt'
    config_path.write_text(render_config(cfg))
    logger.info(f'Generated Triton config: {config_path}')
    return config_path
