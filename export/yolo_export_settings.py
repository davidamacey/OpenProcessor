"""Export settings shared by the YOLO11 export modules (input size, device, precision, NMS thresholds)."""

import torch


# Export settings
IMG_SIZE = 640
DEVICE = 0 if torch.cuda.is_available() else 'cpu'  # GPU 0 or CPU fallback
HALF = bool(torch.cuda.is_available())  # FP16 on GPU, FP32 on CPU
WORKSPACE_GB = 4  # TensorRT workspace size in GB

# NMS settings (for end2end exports)
IOU_THRESHOLD = 0.7
CONF_THRESHOLD = 0.25
