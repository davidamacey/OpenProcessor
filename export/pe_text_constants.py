"""Constants of the PE text encoder export: names, shapes, token ids, defaults and parity prompts.

The tensor and model names are a hard contract with ``src/clients/pe_text_backends.py``.
"""

from pathlib import Path


# Model name + tensor names are a hard contract with src/clients/pe_text_backends.py
# (PE_TEXT_TRITON_MODEL / PE_TEXT_INPUT / PE_TEXT_OUTPUT) — rename both sides
# together.
TRITON_MODEL_NAME = 'pe_text_encoder'
INPUT_TENSOR = 'text_tokens'
OUTPUT_TENSOR = 'text_embeddings'

PE_VARIANT = 'PE-Core-L14-336'
# PE_TEXT_CONFIG['PE-Core-L14-336'].context_length in perception_models.
CONTEXT_LENGTH = 32
EMBEDDING_DIM = 1024
# CLIP BPE start/end-of-text ids; EOT is the largest id in the vocabulary,
# which is what PE's argmax pooling relies on.
SOT_TOKEN_ID = 49406
EOT_TOKEN_ID = 49407

# The torch.export-based exporter's floor; ORT >= 1.14 and Triton's
# onnxruntime backend both run it.
ONNX_OPSET_VERSION = 18
DEFAULT_MAX_BATCH = 32

# The served form is a TensorRT FP16 plan on the GPU (query embedding is 75% of a
# semantic-search request on one CPU instance). The ONNX Runtime model stays as the explicit
# fallback build (``--platform onnxruntime_onnx``), as for the image tower.
TRT_PLATFORM = 'tensorrt_plan'
ORT_PLATFORM = 'onnxruntime_onnx'
PLATFORMS = (TRT_PLATFORM, ORT_PLATFORM)
INSTANCE_KINDS = ('KIND_CPU', 'KIND_GPU')

# Engine optimization profile: batch 1 is the hot path (one query), tokens are dynamic
# (the client trims padding after the batch's last EOT; SOT + EOT is the 2-token floor).
ENGINE_MIN_TOKENS = 2
ENGINE_OPT_TOKENS = 12
ENGINE_OPT_BATCH = 1
# Gate on the built engine against the FP32 graph (measured 2026-11-13 on 2,400 COCO captions:
# median 0.99998, p01 0.99995, min 0.99978; the gate leaves headroom and refuses a real regression).
ENGINE_PARITY_MEDIAN_MIN = 0.999
ENGINE_PARITY_P01_MIN = 0.995

MODELS_DIR = Path('/app/models')
EXPORT_DIR = Path('/app/pytorch_models')

# See export_pe_image_encoder.TRACE_BATCH: a batch-1 trace can bake the
# leading dimension into a Reshape; batch 2 keeps it symbolic.
TRACE_BATCH = 2

PARITY_MIN_COSINE = 0.9999
PARITY_BATCH_SIZES = (1, 8)
# Mix of short, long (> 32 tokens, exercises truncation), punctuation and
# non-ASCII prompts, typical of operator curation queries.
PARITY_PROMPTS: tuple[str, ...] = (
    'a photo of a white pickup truck',
    'red sedan',
    'a person riding a bicycle at night',
    'close-up of a license plate',
    'blurry image',
    (
        'two dogs playing in the snow on a sunny winter afternoon near a frozen lake '
        'with pine trees and mountains in the background under a clear blue sky while '
        'children build a snowman and skate across the ice at dusk'
    ),
    'Stop sign!',
    'motorcycle, side view, black',
    'a crowded street market with colorful umbrellas',
    'x',
    'café storefront with a neon sign',
    'an empty parking lot',
)
# Typical operator queries (all well under the 32-token context) for the
# latency benchmark; the long truncation prompt above is parity-only.
BENCH_PROMPTS: tuple[str, ...] = (
    'a photo of a white pickup truck',
    'red sedan',
    'a person riding a bicycle at night',
    'close-up of a license plate',
    'blurry image',
    'motorcycle, side view, black',
    'a crowded street market with colorful umbrellas',
    'an empty parking lot',
)
