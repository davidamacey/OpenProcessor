"""Triton ``config.pbtxt`` rendering and saving for the YOLO11 exporter."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from config_write import write_generated_config
from yolo_export_settings import IMG_SIZE


if TYPE_CHECKING:
    from pathlib import Path


logger = logging.getLogger(__name__)


def generate_triton_config(
    triton_name: str,
    model_format: str,
    max_batch: int,
    num_classes: int = 80,
    has_nms: bool = False,
    output_dtypes: dict[str, str] | None = None,
) -> str:
    """
    Generate Triton config.pbtxt content for a model.

    Args:
        triton_name: Triton model name
        model_format: Model format ('onnx', 'trt', 'trt_end2end')
        max_batch: Maximum batch size
        num_classes: Number of classes (default 80 for COCO)
        has_nms: Whether model has built-in NMS (end2end models)

    Returns:
        config.pbtxt content as string
    """
    # Determine backend
    if model_format in ('trt', 'trt_end2end'):
        backend = 'tensorrt'
    else:
        backend = 'onnxruntime'

    # A TensorRT engine scales its own uint8 pixels (parse_onnx_model); the ORT graphs keep float32.
    input_dtype = 'TYPE_UINT8' if backend == 'tensorrt' else 'TYPE_FP32'

    # Calculate preferred batch sizes
    preferred_batches = [size for size in [8, 16, 32, 64] if size <= max_batch]
    if not preferred_batches:
        preferred_batches = [1]

    preferred_batch_str = ', '.join(str(b) for b in preferred_batches)

    # Determine platform string
    platform = f'{backend}_plan' if backend == 'tensorrt' else 'onnxruntime_onnx'

    if has_nms:
        # End2End model with NMS outputs. Box/score precision follows the
        # BUILT engine (differs across TRT releases) — see engine_output_dtypes.
        dtypes = output_dtypes or {}
        box_dtype = dtypes.get('det_boxes', 'TYPE_FP32')
        score_dtype = dtypes.get('det_scores', 'TYPE_FP32')
        config = f"""name: "{triton_name}"
platform: "{platform}"
max_batch_size: {max_batch}

input [
  {{
    name: "images"
    data_type: {input_dtype}
    dims: [ 3, {IMG_SIZE}, {IMG_SIZE} ]
  }}
]

output [
  {{
    name: "num_dets"
    data_type: TYPE_INT32
    dims: [ 1 ]
  }},
  {{
    name: "det_boxes"
    data_type: {box_dtype}
    dims: [ 300, 4 ]
  }},
  {{
    name: "det_scores"
    data_type: {score_dtype}
    dims: [ 300 ]
  }},
  {{
    name: "det_classes"
    data_type: TYPE_INT32
    dims: [ 300 ]
  }}
]

dynamic_batching {{
  preferred_batch_size: [ {preferred_batch_str} ]
  max_queue_delay_microseconds: 5000
}}

instance_group [
  {{
    count: 2
    kind: KIND_GPU
    gpus: [ 0 ]
  }}
]
"""
    else:
        # Standard model without NMS
        # Output shape: [batch, 84, 8400] for YOLO11 with 80 classes
        output_dim_1 = 4 + num_classes  # 4 bbox coords + num_classes
        config = f"""name: "{triton_name}"
platform: "{platform}"
max_batch_size: {max_batch}

input [
  {{
    name: "images"
    data_type: {input_dtype}
    dims: [ 3, {IMG_SIZE}, {IMG_SIZE} ]
  }}
]

output [
  {{
    name: "output0"
    data_type: TYPE_FP32
    dims: [ {output_dim_1}, 8400 ]
  }}
]

dynamic_batching {{
  preferred_batch_size: [ {preferred_batch_str} ]
  max_queue_delay_microseconds: 5000
}}

instance_group [
  {{
    count: 2
    kind: KIND_GPU
    gpus: [ 0 ]
  }}
]
"""

    return config


def save_triton_config(
    model_dir: Path,
    triton_name: str,
    model_format: str,
    max_batch: int,
    num_classes: int = 80,
    has_nms: bool = False,
    output_dtypes: dict[str, str] | None = None,
    overwrite: bool = False,
) -> Path:
    """
    Save Triton config.pbtxt file for a model (see ``config_write``: an
    existing config that differs is kept unless ``overwrite``).

    Args:
        model_dir: Model directory path
        triton_name: Triton model name
        model_format: Model format
        max_batch: Maximum batch size
        num_classes: Number of classes
        has_nms: Whether model has built-in NMS

    Returns:
        Path to saved config file
    """
    config_content = generate_triton_config(
        triton_name=triton_name,
        model_format=model_format,
        max_batch=max_batch,
        num_classes=num_classes,
        has_nms=has_nms,
        output_dtypes=output_dtypes,
    )

    return write_generated_config(
        model_dir / 'config.pbtxt',
        config_content,
        overwrite=overwrite,
        output_dtypes=output_dtypes if has_nms else None,
    )
