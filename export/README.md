# Model Export Scripts

This folder contains scripts for exporting models to TensorRT format for NVIDIA Triton Inference Server deployment.

## Overview

The export process transforms PyTorch models into optimized TensorRT engines for high-performance GPU inference.

## Export Scripts

| Script | Purpose | Output |
|--------|---------|--------|
| `export_models.py` | YOLO11 object detection with end2end NMS | TensorRT engine |
| `export_yolo11_face.py` | YOLO11-face detection | TensorRT engine |
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
├── yolo11_face.pt                      # YOLO11-face PyTorch model
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
├── yolo11_face_trt/                    # YOLO11-face TensorRT
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

### YOLO11-face Detection

```bash
docker compose exec yolo-api python /app/export/export_yolo11_face.py
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

### YOLO11-face Detection
- Input: `[B, 3, 640, 640]` FP16, normalized [0, 1]
- Output: Face boxes with 5-point landmarks
- Dynamic batching: 1-64

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
