"""Triton ``config.pbtxt`` of a fused detector engine: one ``output0 [300, 6]`` tensor."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from config_write import write_generated_config
from yolo_export_settings import IMG_SIZE


if TYPE_CHECKING:
    from pathlib import Path


logger = logging.getLogger(__name__)


def render_fused_config(
    triton_name: str, max_batch: int, output_name: str, output_dims: list[int]
) -> str:
    """The ``config.pbtxt`` of an engine that takes uint8 pixels and returns the fused tensor.

    The output is FP32: its last column is the class index, which must stay integral
    (``tests/export/test_family_matrix.py``).
    """
    preferred = [size for size in (8, 16, 32, 64) if size <= max_batch] or [1]
    preferred_str = ', '.join(str(b) for b in preferred)
    dims_str = ', '.join(str(d) for d in output_dims)
    return f"""name: "{triton_name}"
platform: "tensorrt_plan"
max_batch_size: {max_batch}

input [
  {{
    name: "images"
    data_type: TYPE_UINT8
    dims: [ 3, {IMG_SIZE}, {IMG_SIZE} ]
  }}
]

# One row per detection, [x1, y1, x2, y2, score, class], in letterboxed input pixels
output [
  {{
    name: "{output_name}"
    data_type: TYPE_FP32
    dims: [ {dims_str} ]
  }}
]

dynamic_batching {{
  preferred_batch_size: [ {preferred_str} ]
  max_queue_delay_microseconds: 25000
}}

instance_group [
  {{
    count: 2
    kind: KIND_GPU
    gpus: [ 0 ]
  }}
]
"""


def save_fused_config(
    model_dir: Path,
    triton_name: str,
    max_batch: int,
    output_name: str,
    output_dims: list[int],
    *,
    overwrite: bool = False,
) -> Path:
    """Write ``model_dir/config.pbtxt`` (see ``config_write``: a differing tracked config is kept
    unless ``overwrite``). Returns the file written."""
    model_dir.mkdir(parents=True, exist_ok=True)
    return write_generated_config(
        model_dir / 'config.pbtxt',
        render_fused_config(triton_name, max_batch, output_name, output_dims),
        overwrite=overwrite,
    )
