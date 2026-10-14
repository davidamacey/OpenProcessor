"""Constants of the PE image encoder export: names, shapes, platforms, batch and trace defaults.

The tensor and model names are a hard contract with ``src/clients/pe_encoder.py``.
"""

from pathlib import Path


# Triton model name + tensor names are a hard contract with
# src/clients/pe_encoder.py — do not rename one side without the other.
TRITON_MODEL_NAME = 'pe_image_encoder'
INPUT_TENSOR = 'images'
OUTPUT_TENSOR = 'image_embeddings'

PE_VARIANT = 'PE-Core-L14-336'
HF_MODEL_ID = 'facebook/PE-Core-L14-336-hf'
IMAGE_SIZE = 336
EMBEDDING_DIM = 1024

# opset 17 is the floor for clean LayerNorm + dynamic-batch handling in the
# TensorRT ONNX parser (same rationale as export_mobileclip_image_encoder.py).
ONNX_OPSET_VERSION = 17

# Batch 32 at 336x336 is the profile the reference deployment ships:
# min=1 / opt=8 / max=32. Larger batches mostly buy queueing, not
# throughput, for a ViT-L at this resolution.
DEFAULT_MAX_BATCH = 32
# GPU memory of one warm instance at max batch, with headroom (gpu plan and the installer read it).
PE_GPU_MIB_PER_INSTANCE = 1536

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
