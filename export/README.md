# Model Export Scripts

This folder contains scripts for exporting models to TensorRT format for NVIDIA Triton Inference Server deployment.

## Overview

The export process transforms PyTorch models into optimized TensorRT engines for high-performance GPU inference.

## Export Scripts

| Script | Purpose | Output |
|--------|---------|--------|
| `export_models.py` | YOLO11 object detection with end2end NMS | TensorRT engine |
| `export_detector_dual_head.py` | Any YOLO-family detector, re-exported with a backbone feature-map output | ONNX + TensorRT engine |
| `export_detector_dual_head.sh` | `trtexec` engine build + model-repo install for the above | TensorRT engine |
| `export_scrfd.py` | SCRFD-10G face detection + landmarks | TensorRT engine |
| `export_face_recognition.py` | ArcFace face embeddings | TensorRT engine |
| `export_mobileclip_image_encoder.py` | MobileCLIP image encoder | TensorRT engine |
| `export_mobileclip_text_encoder.py` | MobileCLIP text encoder | TensorRT engine |
| `export_paddleocr_det.py` | PP-OCRv5 text detection | TensorRT engine |
| `export_paddleocr_rec.py` | PP-OCRv5 text recognition | TensorRT engine |
| `download_face_models.py` | Download pre-trained face models | PyTorch weights |
| `download_paddleocr.py` | Download PP-OCRv5 models | ONNX models |
| `download_pytorch_models.py` | Download YOLO11 PyTorch models | PyTorch weights |

## Model Directory Structure

```
pytorch_models/
├── yolo11s.pt                          # YOLO11 PyTorch model
├── arcface_w600k_r50.onnx              # ArcFace ONNX model
├── mobileclip2_s2/                     # MobileCLIP checkpoint
├── mobileclip2_s2_image_encoder.onnx   # MobileCLIP image encoder ONNX
└── mobileclip2_s2_text_encoder.onnx    # MobileCLIP text encoder ONNX

models/
├── yolov11_small_trt/                  # YOLO11 TensorRT (standard)
│   ├── 1/model.plan
│   └── config.pbtxt
├── yolov11_small_trt_end2end/          # YOLO11 TensorRT with GPU NMS
│   ├── 1/model.plan
│   └── config.pbtxt
├── scrfd_10g_bnkps/                    # SCRFD-10G face detection TensorRT
│   ├── 1/model.plan
│   └── config.pbtxt
├── arcface_w600k_r50/                  # ArcFace TensorRT
│   ├── 1/model.plan
│   └── config.pbtxt
├── mobileclip2_s2_image_encoder/       # MobileCLIP image encoder
│   ├── 1/model.plan
│   └── config.pbtxt
├── mobileclip2_s2_text_encoder/        # MobileCLIP text encoder
│   ├── 1/model.plan
│   └── config.pbtxt
├── ppocr_det_v5/                       # PP-OCRv5 detection
│   ├── 1/model.plan
│   └── config.pbtxt
└── ppocr_rec_v5/                       # PP-OCRv5 recognition
    ├── 1/model.plan
    └── config.pbtxt
```

## Usage

### YOLO11 Object Detection

```bash
# Export TensorRT with GPU NMS (recommended)
make export-models

# Or directly:
docker compose exec yolo-api python /app/export/export_models.py \
    --models small \
    --formats trt trt_end2end \
    --normalize-boxes
```

### Dual-Head Detector (detections + backbone embedding source)

The curation subsystem stores a per-item **backbone embedding**
(`v6_embedding`, dimension `CurationConfig.backbone_embedding_dim`) and
consumes it in residual clustering, the embedding visualization, item
scores and the OCC conflict handler. It is produced by RoI-pooling a
detector's backbone feature map over each detection box
(`src.services.detection.geometry.roi_pool_sppf`) — which requires the
feature map to be on the wire. A stock detector export emits only the
detection tensor, so the detector has to be re-exported with a second
output:

| Output | Shape | Meaning |
|--------|-------|---------|
| `output0` | family-specific (YOLOv5 `[B, N, 5 + nc]`, Ultralytics v8+ `[B, 4 + nc, N]`) | Detection tensor, unchanged from a single-head export |
| `sppf_feat` | `[B, C, H, W]` where `H = W = imgsz / 32` | Backbone bottleneck (SPPF) feature map |

```bash
# ONNX only (CPU-friendly; validates via onnxruntime round-trip)
docker compose exec yolo-api python /app/export/export_detector_dual_head.py \
    --weights /app/pytorch_models/my_detector.pt \
    --triton-name my_detector_dual_head --imgsz 640

# ONNX + TensorRT engine + config.pbtxt + labels.txt into the model repo
docker compose exec yolo-api python /app/export/export_detector_dual_head.py \
    --weights /app/pytorch_models/my_detector.pt \
    --triton-name my_detector_dual_head \
    --imgsz 1280 --max-batch 16 --formats onnx trt

# Legacy YOLOv5-fork checkpoint (fork path defaults to $DETECTION_YOLOV5_FORK)
docker compose exec yolo-api python /app/export/export_detector_dual_head.py \
    --weights /app/pytorch_models/legacy_v5.pt --loader yolov5 \
    --imgsz 1280 --triton-name legacy_v5_dual_head
```

Nothing is hardcoded to one model: checkpoint, Triton name, input size,
tapped module (`--feature-module`/`--feature-index`) and both output
names are CLI arguments. The tapped module is found by **class name**
(`SPPF` by default), not by a fixed layer index, so architecture drift
cannot silently tap the wrong layer.

Where `trtexec` is available but the TensorRT Python bindings are not
(e.g. inside the triton-server container), build the engine with the
shell companion instead of `--formats trt`:

```bash
export/export_detector_dual_head.sh \
    --onnx pytorch_models/my_detector_dual_head.onnx \
    --name my_detector_dual_head --input-size 1280 --max-batch 16
```

Whichever path builds the engine, the model's `config.pbtxt` must declare
**both** outputs — Triton serves only the tensors its config names, so
omitting `sppf_feat` silently drops the feature map even though the
engine produces it. `--formats trt` writes that file for you.

### SCRFD Face Detection

```bash
docker compose exec yolo-api python /app/export/export_scrfd.py
```

### Face Recognition (ArcFace)

```bash
# Download pre-trained model
docker compose exec yolo-api python /app/export/download_face_models.py

# Export to TensorRT
docker compose exec yolo-api python /app/export/export_face_recognition.py
```

### MobileCLIP (Visual Search)

```bash
# Export both image and text encoders
make export-mobileclip

# Or individually:
docker compose exec yolo-api python /app/export/export_mobileclip_image_encoder.py
docker compose exec yolo-api python /app/export/export_mobileclip_text_encoder.py
```

### PP-OCRv5 (Text Recognition)

```bash
# Download PP-OCRv5 models
docker compose exec yolo-api python /app/export/download_paddleocr.py

# Export detection and recognition
docker compose exec yolo-api python /app/export/export_paddleocr_det.py
docker compose exec yolo-api python /app/export/export_paddleocr_rec.py
```

## Model Specifications

### YOLO11 Object Detection
- Input: `[B, 3, 640, 640]` FP16, normalized [0, 1]
- Output (end2end): `num_dets`, `det_boxes`, `det_scores`, `det_classes`
- Dynamic batching: 1-64 (configurable)

### Dual-Head Detector
- Input: `[B, 3, S, S]` FP32, letterboxed, normalized [0, 1] (`S` = `--imgsz`)
- Output: `output0` (detection tensor) + `sppf_feat` `[B, C, S/32, S/32]`
- Dynamic batching: 1-`--max-batch`

### SCRFD-10G Face Detection
- Input: `[B, 3, 640, 640]` FP32, RGB, (x-127.5)/128.0 normalized
- Output: 9 tensors (3 FPN strides x score/bbox/kps), CPU post-processed
- Dynamic batching: 1-32

### ArcFace Embeddings
- Input: `[B, 3, 112, 112]` FP16, aligned face crops
- Output: `[B, 512]` L2-normalized embeddings
- Dynamic batching: 1-128

### MobileCLIP Image Encoder
- Input: `[B, 3, 256, 256]` FP32, normalized [0, 1]
- Output: `[B, 512]` L2-normalized embeddings
- Dynamic batching: 1-128

### MobileCLIP Text Encoder
- Input: `[B, 77]` INT64 token IDs
- Output: `[B, 512]` L2-normalized embeddings
- Dynamic batching: 1-64

### PP-OCRv5 Detection
- Input: `[B, 3, H, W]` FP32, dynamic size
- Output: Text region polygons

### PP-OCRv5 Recognition
- Input: `[B, 3, 48, W]` FP32, dynamic width
- Output: Character sequence probabilities

## TensorRT Build Settings

All exports use these common settings:
- Precision: FP16 (configurable)
- Workspace: 4GB
- Optimization profiles for dynamic batching

## Troubleshooting

### "Model file not found"
Download PyTorch models first:
```bash
make download-pytorch
```

### "Failed to build TensorRT engine"
Check GPU memory and reduce batch size if needed:
```bash
docker compose exec yolo-api nvidia-smi
```

### Triton fails to load model
Verify file structure and restart Triton:
```bash
ls -lh models/{model_name}/1/
docker compose restart triton-server
```
