"""Constants of the SCRFD export: input size, batch cap, paths, download sources and output names."""

from pathlib import Path

from trt_build_settings import max_batch


TRITON_MODEL_NAME = 'scrfd_10g_bnkps'
INPUT_SIZE = 640
MAX_BATCH_SIZE = max_batch(TRITON_MODEL_NAME, 64)  # Must match the max profile's max_batch_size

# Paths
ONNX_DIR = Path('/app/pytorch_models') if Path('/app').exists() else Path('pytorch_models')
ONNX_FILENAME = 'scrfd_10g_bnkps.onnx'
OUTPUT_DIR = Path('/app/models') if Path('/app').exists() else Path('models')

# Download URLs (in priority order)
DOWNLOAD_URLS = [
    'https://huggingface.co/LPDoctor/insightface/resolve/main/scrfd_10g_bnkps.onnx',
    'https://huggingface.co/Aitrepreneur/insightface/resolve/main/models/antelopev2/scrfd_10g_bnkps.onnx',
]

# FPN configuration for SCRFD-10G
FPN_STRIDES = [8, 16, 32]
NUM_ANCHORS = 2

# Expected output tensor names
SCORE_NAMES = [f'score_{s}' for s in FPN_STRIDES]
BBOX_NAMES = [f'bbox_{s}' for s in FPN_STRIDES]
KPS_NAMES = [f'kps_{s}' for s in FPN_STRIDES]
ALL_OUTPUT_NAMES = SCORE_NAMES + BBOX_NAMES + KPS_NAMES
