"""Constants of the PE image encoder export: names, shapes, platforms, batch and trace defaults.

The tensor and model names are a hard contract with ``src/clients/pe_encoder.py``.
"""

import os
from pathlib import Path

from trt_build_settings import max_batch


# Triton model name + tensor names are a hard contract with
# src/clients/pe_encoder.py — do not rename one side without the other.
TRITON_MODEL_NAME = 'pe_image_encoder'
INPUT_TENSOR = 'images'
OUTPUT_TENSOR = 'image_embeddings'

PE_VARIANT = 'PE-Core-L14-336'
HF_MODEL_ID = 'facebook/PE-Core-L14-336-hf'
IMAGE_SIZE = 336

# The crop encoder is a model of its own: the same weights built at a 224 input (PE's own code
# interpolates the absolute position embedding and rebuilds the RoPE grid for another input size:
# core/vision_encoder/pe.py ``_sample_abs_posemb``, ``rope.update_grid``), 2.1 times the images per
# second of the 336 build at equal crop quality, fed each box letterboxed whole. Its name and size
# are the API's ``PE_CROP_MODEL`` / ``PE_CROP_SIZE`` (a test keeps them equal). The input can also be
# 336 (the owner's choice for classes that differ in fine detail); the export tree cannot import
# ``src`` (importing ``src.config`` loads the whole settings module, and the host build runs
# without it), so the sizes are listed here and a test keeps them equal to ``CROP_ENCODERS``.
CROP_TRITON_MODEL_NAME = 'pe_crop_encoder'
CROP_IMAGE_SIZES = (224, 336)
DEFAULT_CROP_IMAGE_SIZE = 224


def crop_image_size(environ: dict[str, str] | None = None) -> int:
    """The crop encoder's input size: ``PE_CROP_SIZE`` (224 or 336), 224 when unset or blank.

    Anything else raises, as the API does at start-up: a build must never pick a size by guessing.
    """
    raw = (os.environ if environ is None else environ).get('PE_CROP_SIZE', '').strip()
    if not raw:
        return DEFAULT_CROP_IMAGE_SIZE
    if raw not in {str(s) for s in CROP_IMAGE_SIZES}:
        raise ValueError(
            f'PE_CROP_SIZE must be one of {", ".join(map(str, CROP_IMAGE_SIZES))}, got {raw!r}'
        )
    return int(raw)


EMBEDDING_DIM = 1024

# opset 17 is the floor for clean LayerNorm + dynamic-batch handling in the
# TensorRT ONNX parser (same rationale as export_mobileclip_image_encoder.py).
ONNX_OPSET_VERSION = 17

# Batch 32 at 336x336 is the profile the reference deployment ships:
# min=1 / opt=8 / max=32. Larger batches mostly buy queueing, not
# throughput, for a ViT-L at this resolution.
DEFAULT_MAX_BATCH = max_batch('pe_image_encoder', 32)
# GPU memory of one warm instance at max batch, with headroom (gpu plan and the installer read it).
PE_GPU_MIB_PER_INSTANCE = 1536
# The crop encoder (224 input), measured: 750 MiB of model-server device memory, rounded up to 256.
PE_CROP_GPU_MIB_PER_INSTANCE = 768

# Triton platform identifiers for the two serving paths.
TRT_PLATFORM = 'tensorrt_plan'
ORT_PLATFORM = 'onnxruntime_onnx'
PLATFORMS = (TRT_PLATFORM, ORT_PLATFORM)

# Container paths, matching the sibling export scripts. Override on the CLI
# when running on the host.
MODELS_DIR = Path('/app/models')
EXPORT_DIR = Path('/app/pytorch_models')

# The ONNX exporter is traced with a batch-2 dummy on purpose. With a
# batch-1 dummy the PE attention-pool traces a Reshape that captures
# batch=1 as a *constant* volume; TensorRT then rejects every batch > 1
# with "reshape would change volume". A batch-2 dummy forces the tracer to
# treat the leading dimension as symbolic. Verified against ONNX Runtime at
# batch 1/2/4/8/16/32.
TRACE_BATCH = 2
