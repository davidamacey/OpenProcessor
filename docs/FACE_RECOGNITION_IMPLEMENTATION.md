# Face Detection & Recognition Implementation

Production face pipeline using **SCRFD-10G** for detection with 5-point landmarks, **Umeyama affine alignment**, and **ArcFace w600k_r50** for 512-dim face embeddings.

## Pipeline Overview

```
FastAPI Client (fast_face_client.py)
├── Step 1: Preprocess image (letterbox 640x640, normalize)
├── Step 2: SCRFD-10G face detection via Triton gRPC
│   └── Outputs: face boxes, 5-point landmarks, confidence scores
├── Step 3: CPU post-processing
│   ├── Anchor-based decode (3 strides: 8, 16, 32)
│   ├── NMS (IoU 0.4 threshold)
│   └── Umeyama affine alignment → 112x112 face crops
└── Step 4: ArcFace embedding via Triton gRPC
    └── Outputs: 512-dim L2-normalized face embeddings
```

---

## Models

### SCRFD-10G (Face Detection)

| Property | Value |
|----------|-------|
| Model | `scrfd_10g_bnkps` (Batch Normalization, with Keypoints) |
| Input | `[B, 3, 640, 640]` FP32, RGB, normalized |
| Outputs | 9 tensors: 3 strides × (scores, boxes, landmarks) |
| Max Batch | 32 |
| TensorRT | FP16, GPU 0, 2 instances |
| Dynamic Batching | preferred_batch_size: [1, 4, 8, 16, 32] |
| WiderFace | 95.2% Easy / 93.9% Medium / 83.1% Hard |

**Landmark Order (5 points):**
1. Left eye center
2. Right eye center
3. Nose tip
4. Left mouth corner
5. Right mouth corner

### ArcFace w600k_r50 (Face Recognition)

| Property | Value |
|----------|-------|
| Model | `arcface_w600k_r50` (WebFace600K trained ResNet-50) |
| Input | `[B, 3, 112, 112]` FP32, RGB, `(x - 127.5) / 128` |
| Output | `[B, 512]` FP32, L2-normalized |
| Max Batch | 64 |
| TensorRT | FP16, GPU 0, 2 instances |
| Dynamic Batching | preferred_batch_size: [1, 4, 8, 16, 32, 64] |
| LFW Accuracy | 99.8% |

**ArcFace Reference Landmarks (112x112 aligned face):**
```python
ARCFACE_REF_LANDMARKS = [
    [38.2946, 51.6963],   # Left eye
    [73.5318, 51.5014],   # Right eye
    [56.0252, 71.7366],   # Nose
    [41.5493, 92.3655],   # Left mouth
    [70.7299, 92.2041],   # Right mouth
]
```

---

## Key Source Files

| File | Purpose |
|------|---------|
| `src/clients/fast_face_client.py` | Primary face client: SCRFD detect → align → ArcFace embed |
| `src/clients/triton_client.py` | Triton gRPC inference for SCRFD and ArcFace |
| `src/utils/scrfd_decode.py` | Anchor generation, bbox decode, NMS |
| `src/utils/face_align.py` | Umeyama similarity transform, ArcFace alignment |
| `src/routers/faces.py` | API endpoints: /faces/detect, /recognize, /verify, /search, /identify |
| `src/services/inference.py` | Orchestrates detection + recognition pipeline |
| `export/export_scrfd.py` | SCRFD ONNX download + TensorRT export |
| `export/export_face_recognition.py` | ArcFace ONNX → TensorRT export |

---

## OpenSearch Faces Index

**Index:** `visual_search_faces`

| Field | Type | Description |
|-------|------|-------------|
| face_id | keyword | Unique face detection ID |
| person_id | keyword | Identity cluster (same person) |
| image_id | keyword | Source image ID |
| image_path | keyword | Source image path |
| embedding | knn_vector[512] | ArcFace embedding |
| cluster_id | integer | FAISS IVF cluster |
| cluster_distance | float | Distance to centroid |
| box | float[4] | [x1,y1,x2,y2] normalized |
| landmarks | object | 5-point facial landmarks |
| confidence | float | Detection confidence |
| quality_score | float | Face quality metric |
| person_name | keyword | Optional identity label |
| is_reference | boolean | Reference face for clustering |

**HNSW Parameters:**
- ef_construction: 1024
- m: 32
- ef_search: 512

---

## API Endpoints

### Face Detection
```
POST /faces/detect
POST /v1/faces/detect
```
Detect faces using SCRFD-10G. Returns face boxes, 5-point landmarks, and confidence scores.

### Face Recognition
```
POST /faces/recognize
POST /v1/faces/recognize
```
Detect faces + extract ArcFace 512-dim embeddings.

### Face Verification (1:1)
```
POST /faces/verify
POST /v1/faces/verify
```
Compare two face images and return match/no-match with similarity score.

### Face Search
```
POST /faces/search
POST /v1/faces/search
```
Find matching identities for a face in OpenSearch.

### Face Identification (1:N)
```
POST /faces/identify
POST /v1/faces/identify
```
Identify all people in image against the face database.

---

## Setup

```bash
# Export SCRFD and ArcFace to TensorRT
make setup-face-pipeline

# Or individually:
make export-scrfd              # SCRFD → TensorRT
make export-face-recognition   # ArcFace → TensorRT
make load-face-models          # Load into Triton
```

---

## Testing

```bash
# Download test images
make download-test-images

# Quick face test
make test-faces-quick

# Full face endpoint tests (detect, recognize, verify, landmarks)
make test-faces

# Full system test
source .venv/bin/activate && python tests/test_full_system.py
```

---

## References

- [InsightFace GitHub](https://github.com/deepinsight/insightface)
- [SCRFD Paper (ICLR 2022)](https://github.com/deepinsight/insightface/tree/master/detection/scrfd)
- [InsightFace Model Zoo](https://github.com/deepinsight/insightface/blob/master/model_zoo/README.md)
- [InsightFace-REST (TensorRT deployment)](https://github.com/SthPhoenix/InsightFace-REST)
- [LFW Dataset](https://vis-www.cs.umass.edu/lfw/)

---

**Last Updated:** January 2026
