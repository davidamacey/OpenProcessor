# PP-OCRv5 OCR Documentation

Complete guide for deploying and using PP-OCRv5 text detection and recognition on NVIDIA Triton Inference Server.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Model Specifications](#model-specifications)
3. [Prerequisites](#prerequisites)
4. [Setup and Deployment](#setup-and-deployment)
5. [Configuration](#configuration)
6. [API Usage](#api-usage)
7. [Performance Tuning](#performance-tuning)
8. [Troubleshooting](#troubleshooting)
9. [File Reference](#file-reference)

---

## Architecture Overview

The OCR pipeline consists of three Triton models working together:

```
                              +-------------------+
                              |   ocr_pipeline    |
                              |   (Python BLS)    |
                              +--------+----------+
                                       |
              +------------------------+------------------------+
              |                                                 |
    +---------v----------+                          +-----------v---------+
    | paddleocr_det_trt  |                          | paddleocr_rec_trt   |
    |    (TensorRT)      |                          |    (TensorRT)       |
    |   DB++ Detection   |                          | SVTR-LCNet Recog    |
    +--------------------+                          +---------------------+
```

### Components

| Model | Architecture | Purpose |
|-------|-------------|---------|
| `paddleocr_det_trt` | DB++ (Differentiable Binarization) | Detect text regions in images |
| `paddleocr_rec_trt` | SVTR-LCNet | Recognize text within cropped regions |
| `ocr_pipeline` | Python BLS (Backend Language Support) | Orchestrate detection and recognition |

### Data Flow

1. **Input**: Raw image bytes (JPEG/PNG), decoded once by the API; the same oriented frame is sent to Triton and reported as `image_size`
2. **Preprocessing**: Decode with EXIF orientation applied (transparency composited onto white), scale so the long side is 640 to 960 px, then stretch each axis to its own multiple of 32 (no padding, as PaddleOCR's `DetResizeForTest`); the BLS normalizes (x/127.5 - 1)
3. **Detection**: Probability map identifying text regions
4. **Post-processing**: Threshold, contour detection, box expansion (unclip) with `pyclipper`, which is required: the `ocr_pipeline` model fails to load without it (`Dockerfile.triton` installs it)
5. **Cropping**: Perspective transform to extract text crops
6. **Recognition**: CTC decoder produces text strings
7. **Output**: Text strings with bounding boxes and confidence scores

---

## Model Specifications

### Detection Model (DB++)

| Property | Value |
|----------|-------|
| **Model** | PP-OCRv5 Mobile Detection |
| **Architecture** | DB++ (Differentiable Binarization) |
| **Input Shape** | `[B, 3, H, W]` where H,W are multiples of 32, max 960 |
| **Input Format** | FP32, BGR, normalized (x/127.5 - 1); the `ocr_pipeline` BLS builds this from the uint8 pixels it receives |
| **Output Shape** | `[B, 1, H, W]` probability map |
| **Output Range** | [0, 1] sigmoid probabilities |
| **Max Batch Size** | 4 |

### Recognition Model (SVTR-LCNet)

| Property | Value |
|----------|-------|
| **Model** | PP-OCRv5 Mobile Recognition (Multilingual) |
| **Architecture** | SVTR-LCNet |
| **Input Shape** | `[B, 3, 48, W]` where W is 48-2048 (dynamic) |
| **Input Format** | FP32, BGR, normalized (x/127.5 - 1) |
| **Output Shape** | `[B, T, 18385]` character probabilities |
| **Output Format** | Softmax probabilities for CTC decoding |
| **Character Set** | 18385 characters (18383 from dict + blank + space) |
| **Max Batch Size** | 64 |

### Dictionary Configuration

The recognition model uses `ppocrv5_dict.txt` with 18383 characters:

- Chinese, Japanese, Korean characters
- English uppercase/lowercase letters
- Digits 0-9
- Common punctuation and symbols (multilingual)
- Special tokens: blank (index 0), space (index 18384)

---

## Prerequisites

### Hardware Requirements

- NVIDIA GPU with at least 4GB VRAM
- Recommended: RTX 3000/4000 series or A-series datacenter GPUs
- 16GB+ system RAM for TensorRT engine building

### Software Requirements

- Docker with NVIDIA Container Toolkit
- NVIDIA GPU with TensorRT 11 support (Triton 26.06)
- Python 3.10+ with the following packages:
  - `onnx`, `onnxruntime-gpu`

---

## Setup and Deployment

### Step 1: Download and export

Engines are built inside the `model-export` container. `./openprocessor models install`
does this for a normal install. To build the OCR engines by hand:

```bash
./scripts/export_paddleocr.sh all        # download, build both engines, place the dictionary
./scripts/export_paddleocr.sh download   # ONNX files only (also: trt, det, rec, status, clean)
```

or run the Python exporters directly, as in [export/README.md](../export/README.md#pp-ocrv5-text-recognition).
The builds use TensorRT 11 strongly-typed mode and the `--memPoolSize`-style workspace
(`export/trt_build_settings.py`: detection 4 GB, recognition 8 GB). The dynamic shapes are
detection `1x3x32x32` to `4x3x960x960` (opt `736x736`) and recognition `1x3x48x48` to
`64x3x48x2048` (opt `32x3x48x320`). The old `--workspace=<MB>` flag causes
`Cudnn Error: CUDNN_STATUS_NOT_SUPPORTED`.

### Step 3: Setup Dictionary

The dictionary file must be in the recognition model directory:

```bash
# Verify dictionary exists (should have 18383 lines, plus blank+space = 18385 total)
wc -l models/paddleocr_rec_trt/ppocrv5_dict.txt

# Expected output: 18383 lines
# Total classes: 18385 (18383 dict + blank token + space token)
```

### Step 4: Deploy and Test

#### Restart Triton

```bash
# Reload all models
docker compose restart triton-server

# Or load specific models via API
curl -X POST localhost:4600/v2/repository/models/paddleocr_det_trt/load
curl -X POST localhost:4600/v2/repository/models/paddleocr_rec_trt/load
curl -X POST localhost:4600/v2/repository/models/ocr_pipeline/load
```

#### Verify Models Loaded

```bash
curl -s localhost:4600/v2/models | jq '.models[] | select(.name | startswith("paddle") or startswith("ocr"))'
```

Expected output:
```json
{"name": "paddleocr_det_trt", "state": "READY"}
{"name": "paddleocr_rec_trt", "state": "READY"}
{"name": "ocr_pipeline", "state": "READY"}
```

#### Run Tests

```bash
make test-ocr

# Or use curl with any image of text
curl -X POST http://localhost:4603/ocr/predict -F "image=@your_image.jpg"
```

---

## Configuration

The Triton configs are the source of truth; read them rather than a copy here.

The batch size, instance count, GPU and queue delay of each model are in its file:
`models/paddleocr_det_trt/config.pbtxt` and `models/paddleocr_rec_trt/config.pbtxt` (`tensorrt_plan`, dynamic
batching) and `models/ocr_pipeline/config.pbtxt` (`python` BLS, `max_batch_size: 0` because the BLS batches
itself, no dynamic batching). The recognizer ships one instance as a baseline for 12 GB cards.

The pipeline takes `ocr_images` (the resized UINT8 BGR frame), `original_image` (the full-size UINT8
frame the crops are cut from) and `orig_shape`. It returns up to 128 texts per image:
`num_texts`, `text_boxes` (`[128, 8]` quadrilaterals), `text_boxes_normalized` (`[128, 4]`),
`texts`, `text_scores` (detection) and `rec_scores` (recognition). An FP32 pipeline config is
refused at request time with the `./openprocessor models repair` hint
(`src/clients/triton_ocr.py`). The installer GPU profiles set the instance counts.

---

## API Usage

### Extract Text (Single Image)

```bash
curl -X POST http://localhost:4603/ocr/predict \
    -F "image=@your_image.jpg" \
    -F "min_det_score=0.5" \
    -F "min_rec_score=0.8"
```

Response:
```json
{
  "status": "success",
  "texts": ["Hello", "World"],
  "boxes": [[x1,y1,x2,y2,x3,y3,x4,y4], ...],
  "boxes_normalized": [[0.1, 0.2, 0.5, 0.3], ...],
  "det_scores": [0.95, 0.88],
  "rec_scores": [0.99, 0.97],
  "num_texts": 2,
  "image_size": [480, 640]
}
```

### Batch Processing

```bash
curl -X POST http://localhost:4603/ocr/batch \
    -F "images=@image1.jpg" \
    -F "images=@image2.jpg" \
    -F "images=@image3.jpg"
```

### Search Images by Text

```bash
curl -X POST "http://localhost:4603/search/ocr?text=invoice&top_k=10"
```

### Python Client

```python
import requests

def extract_text(image_path: str, api_url: str = "http://localhost:4603") -> dict:
    """Extract text from an image using OCR API."""
    with open(image_path, 'rb') as f:
        response = requests.post(
            f"{api_url}/ocr/predict",
            files={"image": f},
            data={"min_det_score": 0.5, "min_rec_score": 0.8}
        )
    return response.json()

# Usage
result = extract_text("your_image.jpg")
print(f"Found {result['num_texts']} text regions:")
for text, score in zip(result['texts'], result['rec_scores']):
    print(f"  '{text}' (confidence: {score:.2f})")
```

---

## Performance Tuning

### Optimizing Detection

| Parameter | Tuning |
|-----------|--------|
| `max_batch_size` | Increase for batch processing |
| `instance_count` | Match to GPU utilization |
| `max_queue_delay` | Lower for latency, higher for throughput |

### Optimizing Recognition

| Parameter | Tuning |
|-----------|--------|
| `max_batch_size` | Increase for batch processing |
| `instance_count` | Increase on 24 GB+ cards for parallel crops |
| Dynamic width | Narrow the range (set in the engine build) if text length is known |

### Memory Optimization

1. **Reduce max shapes** if you know image sizes:
   ```bash
   --maxShapes=x:4x3x640x640  # Instead of 960x960 (the client scales up to 960 px)
   ```

2. **Precision** follows the ONNX dtypes (TensorRT 11 builds are strongly typed; there is no `--fp16` switch)

3. **Enable CUDA graphs** for repeated inference:
   ```protobuf
   optimization {
     cuda {
       graphs: true
     }
   }
   ```

### Throughput

Measured `/ocr/predict` throughput and latency are in [PERFORMANCE.md](PERFORMANCE.md); this page holds no numbers.

--------------|------------|---------------|
| Single image | 15-20 RPS | 50-70ms |
| Batch (4 images) | 40-50 RPS | 100-150ms |

---

## Troubleshooting

### Common Errors

#### HTTP status of an OCR failure

`POST /ocr/predict` answers `422` when the image cannot be decoded or is too small, and `502`
when Triton or the OCR pipeline fails; Triton being unreachable after retries is `503`. An image
with no text is a `200` with `num_texts: 0`. `POST /analyze` and `/analyze/batch` keep the rest of
the analysis and set `ocr_error` when OCR failed. `POST /ocr/batch` lists a failed image under
`failures`.

#### "Cudnn Error: CUDNN_STATUS_NOT_SUPPORTED"

**Cause**: Insufficient TensorRT workspace memory.

**Solution**: Use correct syntax for workspace allocation:
```bash
# TensorRT 10 and 11 syntax:
--memPoolSize=workspace:4G

# Not:
--workspace=4096  # Deprecated
```

#### "Engine deserialization failed"

**Cause**: TensorRT engine was built with different GPU or TensorRT version.

**Solution**: Rebuild the engine on the target GPU:
```bash
rm models/paddleocr_det_trt/1/model.plan
rm models/paddleocr_rec_trt/1/model.plan
./scripts/export_paddleocr.sh trt
```

#### "Model not found: paddleocr_det_trt"

**Cause**: Model not loaded or config error.

**Solution**:
1. Check model files exist:
   ```bash
   ls -la models/paddleocr_det_trt/1/model.plan
   ls -la models/paddleocr_rec_trt/1/model.plan
   ```
2. Check Triton logs:
   ```bash
   docker compose logs triton-server | grep -i error
   ```
3. Reload models:
   ```bash
   curl -X POST localhost:4600/v2/repository/models/paddleocr_det_trt/load
   ```

#### Empty OCR Results

**Cause**: Detection threshold too high or preprocessing mismatch.

**Solution**:
1. Lower detection threshold:
   ```bash
   curl -X POST ... -F "min_det_score=0.3"
   ```
2. Check preprocessing:
   - Input should be BGR, not RGB
   - Normalization: (x / 127.5) - 1 = range [-1, 1]
   - Each axis stretched to a multiple of 32 (boxes are mapped back as original size over map size, so the map must cover the whole image)

#### GPU Out of Memory

**Cause**: Multiple models competing for GPU memory.

**Solution**:
1. Unload unused models:
   ```bash
   curl -X POST localhost:4600/v2/repository/models/yolov11_small_trt/unload
   ```
2. Reduce instance count in config.pbtxt

---

## File Reference

### Model Files

```
models/
├── paddleocr_det_trt/
│   ├── 1/model.plan              # TensorRT detection engine
│   └── config.pbtxt              # Triton config
├── paddleocr_rec_trt/
│   ├── 1/model.plan              # TensorRT recognition engine
│   ├── config.pbtxt              # Triton config
│   └── ppocrv5_dict.txt          # Character dictionary (18383 chars, multilingual)
└── ocr_pipeline/
    ├── 1/model.py                # Python BLS orchestrator
    └── config.pbtxt              # Triton config
```

### Source Files

```
export/
├── download_paddleocr.py         # Download ONNX models
├── export_paddleocr_det.py       # Export detection to TRT
└── export_paddleocr_rec.py       # Export recognition to TRT

scripts/
└── export_paddleocr.sh           # Automated export script

src/
├── services/ocr_service.py       # OCR service wrapper
└── clients/triton_client.py      # Triton gRPC client (infer_ocr delegates to clients/triton_ocr.py)
```

---

## References

- [PaddleOCR GitHub](https://github.com/PaddlePaddle/PaddleOCR)
- [Triton Inference Server Documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/)
- [TensorRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/)

---

**Version History**

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-01-02 | Initial PP-OCRv5 implementation |
| 1.1 | 2025-01-10 | Added workspace memory fix documentation |
| 2.0 | 2026-01-26 | Consolidated documentation |
| 3.0 | 2026-01-27 | Switched to multilingual model (18385 classes), removed PaddleX dependency |
