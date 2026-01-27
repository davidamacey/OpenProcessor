# Ingest Benchmark Methodology

## Test Configuration

### Hardware
- GPU 0: NVIDIA A6000 (49GB) - Primary inference GPU
- CPU: Available for preprocessing comparison tests

### Software
- Triton Inference Server with unified_complete_pipeline
- 8 pipeline instances (configurable in models/unified_complete_pipeline/config.pbtxt)

### Test Datasets
| Dataset | Path | Images | Description |
|---------|------|--------|-------------|
| KILLBOY | `/mnt/nvm/KILLBOY_SAMPLE_PICTURES` | 1,137 | Real-world DSLR photos |
| FACE_TEST | `/mnt/nvm/FACE_TEST_IMAGES` | 200 | Face detection test images |
| OCR Synthetic | `test_images/ocr-synthetic` | 18 | Synthetic OCR test images |
| OCR Real | `test_images/ocr-real` | 3 | Real-world OCR images |
| **Total** | - | **1,358** | Combined test set |

## Baseline Test Command

```bash
# Clear indexes
curl -s -X DELETE "http://localhost:4603/query/index"
curl -s -X POST "http://localhost:4603/query/index/create"

# Run ingestion benchmark
python scripts/ingest_all_indexes.py \
    --images-dirs "/mnt/nvm/KILLBOY_SAMPLE_PICTURES,/mnt/nvm/FACE_TEST_IMAGES,/mnt/nvm/repos/OpenProcessor/test_images/ocr-synthetic,/mnt/nvm/repos/OpenProcessor/test_images/ocr-real" \
    --max-images 5000 \
    --batch-size 16 \
    --workers 16 \
    --skip-clustering
```

## Metrics Collection

### Triton Metrics (after test)
```bash
# Queue duration (bottleneck indicator)
curl -s "http://localhost:4602/metrics" | grep "queue_duration" | grep -v "^#"

# Compute duration
curl -s "http://localhost:4602/metrics" | grep "compute_infer_duration" | grep -v "^#"

# Request counts
curl -s "http://localhost:4602/metrics" | grep "request_success" | grep -v "^#"
```

### Key Metrics to Record
1. **Total time** (seconds)
2. **Throughput** (images/sec)
3. **Queue time per image** (ms) - from Triton metrics
4. **Compute time per image** (ms) - from Triton metrics

## Baseline Results (2024-01-14)

### Configuration: DALI GPU Preprocessing
- Pipeline instances: 8
- Batch size: 16
- Workers: 16

| Metric | Value |
|--------|-------|
| Total Images | 1,360 |
| Total Time | 81.2 sec |
| **Throughput** | **16.7 img/sec** |
| Queue Time/Image | 11.6 sec (unified_complete_pipeline) |
| Compute Time/Image | 0.49 sec |

### ML Models Processed
- Global embeddings: 1,360
- Vehicles: 1,044
- People: 1,123
- Faces: 566
- OCR text regions: 1,133

## Preprocessing Comparison Test

### Current: DALI GPU Preprocessing
```
Image bytes → DALI (GPU nvJPEG decode + GPU resize) → GPU inference
```
- Uses: dual_preprocess_dali (1 instance)
- Bottleneck: Single DALI instance serving 8 pipeline instances

### Alternative: CPU Preprocessing (Ultralytics-style)
```
Image bytes → CPU ThreadPool (cv2 decode + cv2 resize) → GPU inference
```
- Uses: ThreadPoolExecutor with cv2/PIL
- Advantage: Unlimited CPU parallelism

## 100k Image Projection

| Scenario | Throughput | Time for 100k |
|----------|------------|---------------|
| Current (8 instances) | 16.7 img/sec | ~100 min |
| 16 instances | ~33 img/sec | ~50 min |
| CPU preprocess (TBD) | TBD | TBD |
