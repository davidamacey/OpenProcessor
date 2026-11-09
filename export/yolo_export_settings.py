"""Export settings of the Ultralytics detector export (input size, device)."""

IMG_SIZE = 640
# The ONNX trace runs on the CPU in FP32: ``dynamic=True`` only traces on the CPU, and the FP16
# engine is built from that FP32 graph later (ModelOpt AutoCast, yolo_trt_build). The CPU-only
# torch of the former second environment made this implicit; the main environment's CUDA torch
# would otherwise pick the GPU and half precision (tests/test_export_yolo_settings.py).
DEVICE = 'cpu'
