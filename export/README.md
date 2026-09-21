# Model Export Scripts

This folder contains scripts for exporting models to TensorRT format for NVIDIA Triton Inference Server deployment.

## Overview

The export process transforms PyTorch models into optimized TensorRT engines for high-performance GPU inference.

## Export Scripts

| Script | Purpose | Output |
|--------|---------|--------|
| `export_models.py` | YOLO11 object detection with end2end NMS | TensorRT engine |
| `export_scrfd.py` | SCRFD-10G face detection + landmarks | TensorRT engine |
| `export_face_recognition.py` | ArcFace face embeddings | TensorRT engine |
| `export_mobileclip_image_encoder.py` | MobileCLIP image encoder | TensorRT engine |
| `export_mobileclip_text_encoder.py` | MobileCLIP text encoder | TensorRT engine |
| `export_pe_image_encoder.py` | PE-Core-L14-336 image encoder (curation `pe_embedding`) | ONNX + `config.pbtxt` |
| `build_pe_trt.sh` | PE-Core ONNX → TensorRT engine (Path 1) | TensorRT engine |
| `build_pe_ort_fallback.sh` | PE-Core ONNX served by Triton's ORT backend (Path 2) | ONNX model dir |
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
├── mobileclip2_s2_text_encoder.onnx    # MobileCLIP text encoder ONNX
└── pe_image_encoder.onnx               # PE-Core-L14-336 image encoder ONNX

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
├── pe_image_encoder/                   # PE-Core-L14-336 image encoder
│   ├── 1/model.plan                    #   Path 1 (TensorRT), OR
│   ├── 1/model.onnx                    #   Path 2 (ONNX Runtime) - never both
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

### PE-Core Image Encoder (Curation Embeddings)

**Required by the curation subsystem.** `src/clients/pe_encoder.py` calls the
Triton model `pe_image_encoder` and stores the result as the `pe_embedding`
field, which backs semantic search, near-duplicate detection, residual
clustering and the embedding visualization (see
[`docs/CURATION.md`](../docs/CURATION.md#models-you-must-supply)). The model
name and both tensor names are a hardcoded contract with that client, not
configuration.

This export runs in two stages — the ONNX export in the API container, the
TensorRT build in the Triton container (that's where `trtexec` lives):

```bash
# Stage 1: PE-Core-L14-336 vision tower -> ONNX (+ config.pbtxt)
docker compose exec yolo-api python /app/export/export_pe_image_encoder.py

# Stage 2, Path 1 (preferred): ONNX -> TensorRT engine
ONNX_PATH=./pytorch_models/pe_image_encoder.onnx bash export/build_pe_trt.sh

# Stage 2, Path 2 (fallback): serve the ONNX directly via Triton's ORT backend
ONNX_PATH=./pytorch_models/pe_image_encoder.onnx bash export/build_pe_ort_fallback.sh
```

Path 2 exists because PE's attention-pooling head has, on some TensorRT
releases, used ops the ONNX parser rejects. `build_pe_trt.sh` exits `3` with
an explicit pointer to the fallback when that happens. Both paths install
into `models/pe_image_encoder/` and render the matching `config.pbtxt`
(`tensorrt_plan` vs. `onnxruntime_onnx`); each removes the other's artifact
so Triton never sees both.

Useful flags on the exporter:

| Flag | Purpose |
|------|---------|
| `--method optimum` | Export `facebook/PE-Core-L14-336-hf` via Optimum instead of `perception_models`. Needs no local PE install, but names its tensors `pixel_values`/`image_embeds` — the exporter reports the mismatch rather than shipping a model the client can't call. |
| `--config-only --platform ...` | Re-render just `config.pbtxt` (what the two build scripts call). |
| `--models-dir`, `--max-batch`, `--gpus`, `--instance-count` | Target repository + Triton tuning. `--max-batch` must match the engine's profile. |
| `--skip-validate` | Skip the ONNX Runtime probe (which also catches a static leading axis — see below). |

> **Trace batch size matters.** The exporter traces with a batch-**2** dummy
> on purpose. With batch 1, PE's attention pool bakes the batch dimension
> into a Reshape as a constant volume; Triton then loads the model fine and
> rejects every request with batch > 1. The validation step detects and
> reports this — re-export rather than editing `config.pbtxt`.

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

### PE-Core-L14-336 Image Encoder
- Input: `images` `[B, 3, 336, 336]` FP32, ImageNet mean/std normalized
  (resize shorter edge to 336, center-crop — see
  `src/services/detection/pe_preprocess.py`)
- Output: `image_embeddings` `[B, 1024]` FP32, L2-normalized
- Dynamic batching: 1-32 (engine profile min=1 / opt=8 / max=32)

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
