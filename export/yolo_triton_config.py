"""Triton ``config.pbtxt`` of a fused detector engine: one ``output0 [300, 6]`` tensor."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from config_write import write_generated_config
from yolo_export_settings import IMG_SIZE


if TYPE_CHECKING:
    from pathlib import Path


logger = logging.getLogger(__name__)


class BatchMismatchError(ValueError):
    """The batch an ONNX was exported for and the batch Triton would send disagree."""


def onnx_batch(onnx_path: Path) -> int:
    """The ``batch`` Ultralytics recorded in the ONNX metadata when it exported the graph."""
    import onnx

    meta = {
        p.key: p.value for p in onnx.load(str(onnx_path), load_external_data=False).metadata_props
    }
    try:
        return int(meta['batch'])
    except (KeyError, ValueError) as e:
        raise BatchMismatchError(
            f'{onnx_path} does not record the batch it was exported for: export it again'
        ) from e


def check_served_batch(
    onnx_path: Path, model_dir: Path, max_batch: int, *, overwrite: bool
) -> None:
    """Refuse an engine whose Triton ``max_batch_size`` could exceed the batch it was exported for.

    With NMS in the graph the export ``batch`` is a hard cap: images past it come back with no
    detections and no error (``tests/export/test_family_matrix.py``). The writer's own config is
    ``max_batch_size == batch``. A tracked config that is kept (``overwrite`` false) may be
    smaller, because a GPU profile tunes it down (``scripts/lib/config.sh``), and is never larger.
    """
    exported = onnx_batch(onnx_path)
    if exported != max_batch:
        raise BatchMismatchError(
            f'{onnx_path} was exported for batch {exported}, the engine is built for {max_batch}'
        )
    kept = model_dir / 'config.pbtxt'
    if overwrite or not kept.is_file():
        return
    found = re.search(r'^max_batch_size:\s*(\d+)', kept.read_text(), re.MULTILINE)
    served = int(found.group(1)) if found else None
    if served is None or served > exported:
        raise BatchMismatchError(
            f'{kept} serves max_batch_size {served}, past the batch {exported} the ONNX was '
            'exported for (images beyond it would come back empty): pass --overwrite-config'
        )
    if served < exported:
        logger.warning(f'{kept} serves max_batch_size {served}, below the engine batch {exported}')


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
