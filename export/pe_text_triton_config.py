"""Triton ``config.pbtxt`` rendering and install for the PE text encoder (pure, unit-tested without torch)."""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pe_text_constants import (
    CONTEXT_LENGTH,
    DEFAULT_MAX_BATCH,
    EMBEDDING_DIM,
    INPUT_TENSOR,
    INSTANCE_KINDS,
    OUTPUT_TENSOR,
    PE_VARIANT,
    PLATFORMS,
    TRITON_MODEL_NAME,
    TRT_PLATFORM,
)


if TYPE_CHECKING:
    from pathlib import Path


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PETextTritonConfig:
    """Everything that varies in the rendered ``config.pbtxt``."""

    model_name: str = TRITON_MODEL_NAME
    max_batch_size: int = DEFAULT_MAX_BATCH
    context_length: int = CONTEXT_LENGTH
    embedding_dim: int = EMBEDDING_DIM
    input_name: str = INPUT_TENSOR
    output_name: str = OUTPUT_TENSOR
    instance_count: int = 1
    platform: str = TRT_PLATFORM
    kind: str = 'KIND_GPU'
    gpu_ids: tuple[int, ...] = (0,)
    max_queue_delay_us: int = 2000

    def __post_init__(self) -> None:
        if not self.model_name:
            raise ValueError('model_name must be a non-empty string')
        for attr in ('max_batch_size', 'context_length', 'embedding_dim', 'instance_count'):
            value = getattr(self, attr)
            if not isinstance(value, int) or value < 1:
                raise ValueError(f'{attr} must be a positive int, got {value!r}')
        if self.max_queue_delay_us < 0:
            raise ValueError(f'max_queue_delay_us must be >= 0, got {self.max_queue_delay_us}')
        if self.platform not in PLATFORMS:
            raise ValueError(f'platform must be one of {PLATFORMS}, got {self.platform!r}')
        if self.platform == TRT_PLATFORM and self.kind != 'KIND_GPU':
            raise ValueError(
                f'{TRT_PLATFORM} runs on the GPU: kind must be KIND_GPU, got {self.kind!r}'
            )
        if self.kind not in INSTANCE_KINDS:
            raise ValueError(f'kind must be one of {INSTANCE_KINDS}, got {self.kind!r}')
        if self.kind == 'KIND_GPU':
            if not self.gpu_ids:
                raise ValueError('gpu_ids must contain at least one device id for KIND_GPU')
            if any(not isinstance(g, int) or g < 0 for g in self.gpu_ids):
                raise ValueError(f'gpu_ids must be non-negative ints, got {self.gpu_ids!r}')
        if not self.input_name or not self.output_name:
            raise ValueError('input_name and output_name must be non-empty strings')


def model_file_name(platform: str) -> str:
    """File Triton loads from ``<model>/1/`` for ``platform``."""
    if platform not in PLATFORMS:
        raise ValueError(f'platform must be one of {PLATFORMS}, got {platform!r}')
    return 'model.plan' if platform == TRT_PLATFORM else 'model.onnx'


def preferred_batch_sizes(max_batch: int) -> list[int]:
    """Triton ``preferred_batch_size`` ladder bounded by ``max_batch``."""
    ladder = [size for size in (4, 8, 16, 32) if size <= max_batch]
    return ladder or [max_batch]


def render_config(cfg: PETextTritonConfig) -> str:
    """Render the Triton ``config.pbtxt`` body for the PE text encoder."""
    preferred = ', '.join(str(b) for b in preferred_batch_sizes(cfg.max_batch_size))
    if cfg.kind == 'KIND_GPU':
        gpus = ', '.join(str(g) for g in cfg.gpu_ids)
        instance = f'    count: {cfg.instance_count}\n    kind: KIND_GPU\n    gpus: [ {gpus} ]'
    else:
        instance = f'    count: {cfg.instance_count}\n    kind: KIND_CPU'

    return f"""# {PE_VARIANT} text encoder — semantic-search query embeddings.
#
# src/clients/pe_encoder.py encodes every query through this Triton model
# (OP_PE_TEXT_TRITON_MODEL names it). The tensor names below are a contract
# with that client.
#
# Input:  {cfg.input_name} [B, T] INT64, T <= {cfg.context_length} — PE SimpleTokenizer ids
#         (SOT + BPE + EOT, zero-padded; tokenized in Python by the client,
#         which trims the padding after the batch's last EOT, hence dims -1)
# Output: {cfg.output_name} [B, {cfg.embedding_dim}] FP32, L2-normalized
#
# Model file: 1/{model_file_name(cfg.platform)} from export/export_pe_text_encoder.py
# (--install-triton writes both; the TensorRT FP16 plan is the default, the
# ONNX Runtime graph is the explicit --platform onnxruntime_onnx fallback). Load it explicitly:
#   --load-model={cfg.model_name}

name: "{cfg.model_name}"
platform: "{cfg.platform}"
max_batch_size: {cfg.max_batch_size}

input [
  {{
    name: "{cfg.input_name}"
    data_type: TYPE_INT64
    dims: [ -1 ]
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

instance_group [
  {{
{instance}
  }}
]
"""


def write_triton_config(models_dir: Path, cfg: PETextTritonConfig) -> Path:
    """Write ``<models_dir>/<model_name>/config.pbtxt`` and return its path."""
    model_dir = models_dir / cfg.model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    config_path = model_dir / 'config.pbtxt'
    config_path.write_text(render_config(cfg))
    logger.info(f'Generated Triton config: {config_path}')
    return config_path


def install_triton_model(model_path: Path, models_dir: Path, cfg: PETextTritonConfig) -> Path:
    """Copy the plan or graph to ``<models_dir>/<name>/1/<model file>`` + write config.

    The other platform's file is removed: Triton loads whichever the config names and would
    silently ignore a stale one.
    """
    version_dir = models_dir / cfg.model_name / '1'
    version_dir.mkdir(parents=True, exist_ok=True)
    for platform in PLATFORMS:
        if platform != cfg.platform:
            (version_dir / model_file_name(platform)).unlink(missing_ok=True)
    target = version_dir / model_file_name(cfg.platform)
    shutil.copy2(model_path, target)
    target.chmod(0o644)
    logger.info(f'Installed Triton model file: {target}')
    write_triton_config(models_dir, cfg)
    return target
