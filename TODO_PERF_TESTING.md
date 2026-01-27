# Performance Testing TODO

Track performance benchmarking and optimization tasks for all Triton models.

## Completed

- [x] SCRFD-10G: Batch-patched ONNX for dynamic batching (Transpose perm fix + Reshape fix)
- [x] SCRFD-10G: TRT FP16 engine built with dynamic batch 1-32
- [x] SCRFD-10G: Triton config with dynamic_batching (max_batch_size=32, preferred=[4,8,16,32])
- [x] SCRFD-10G: Baseline Python benchmark (~100 RPS concurrent, GPU compute 1.2ms)

## In Progress

### SCRFD-10G Face Detection
- [ ] Optimize queue delay (currently 5ms, try 1-2ms for ad-hoc requests)
- [ ] Test with varied batch sizes to find throughput sweet spot
- [ ] Measure full pipeline throughput (SCRFD detect + Umeyama align + ArcFace embed)
- [ ] Wire into face API endpoints and benchmark via HTTP
- [ ] Compare with InsightFace-REST reference (~820 FPS detection on RTX 4090)

## Pending - All Models

### Queue Delay Optimization
Each model needs optimal `max_queue_delay_microseconds`:
- [ ] `yolov11_small_trt_end2end` - Currently 5000us, test 1000-10000us
- [ ] `scrfd_10g_bnkps` - Currently 5000us (face detection)
- [ ] `arcface_w600k_r50` - Currently 5000us, may benefit from lower for face pipeline latency
- [ ] `mobileclip2_s2_image_encoder` - Currently 5000us
- [ ] `mobileclip2_s2_text_encoder` - Currently 5000us
- [ ] `paddleocr_det_trt` - Currently 5000us
- [ ] `paddleocr_rec_trt` - Currently 5000us
- [ ] `scrfd_10g_bnkps` - Currently 5000us

### Batch Size Optimization
- [ ] Test preferred_batch_size combinations for each model
- [ ] Measure throughput vs latency tradeoff at different batch sizes
- [ ] Profile GPU utilization at each batch size to find saturation point

### Instance Group Tuning
- [ ] Test instance_group count (1,2,4) for each model
- [ ] Measure if multiple instances improve throughput or just add memory overhead
- [ ] Check GPU memory usage with different instance counts

### Concurrency Testing
For each model, measure:
- [ ] Serial (1 client) latency and RPS
- [ ] Concurrent (4, 8, 16, 32, 64 clients) throughput
- [ ] Latency distribution (p50, p95, p99) under load
- [ ] Error rates at high concurrency

### Pipeline Benchmarks
- [ ] Full face pipeline: SCRFD detect + align + ArcFace embed (target: 200-400 RPS)
- [ ] Full detection pipeline: YOLO detect + CLIP embed (target: 200+ RPS)
- [ ] Full ingest pipeline: detect + faces + embed + OCR (target: 100+ RPS)
- [ ] Full analyze pipeline: all capabilities combined

### Benchmark Tooling
- [ ] Create `tests/bench_triton_models.py` - per-model Triton benchmark (async gRPC)
- [ ] Create `tests/bench_pipelines.py` - end-to-end pipeline benchmark via HTTP
- [ ] Add Makefile targets: `bench-scrfd`, `bench-arcface`, `bench-clip`, `bench-ocr`
- [ ] Add Grafana dashboard for real-time throughput monitoring

## Notes

### perf_analyzer Warning
DO NOT use `perf_analyzer` with SCRFD (9 output tensors, 640x640 input) without
memory limits. It consumed 437 GB RAM and caused two system-wide OOM events.
Use bounded Python benchmarks instead, or run perf_analyzer with:
- `--shared-memory=cuda` to keep buffers in GPU VRAM
- Container memory limit: `deploy.resources.limits.memory: 32g`
- Low concurrency: `--concurrency-range 1:8:2`

### Key Metrics from Triton
Query via `curl http://localhost:4602/metrics | grep model_name`:
- `nv_inference_count` - Total inferences (with batch expansion)
- `nv_inference_exec_count` - Actual GPU executions (inference_count/exec_count = avg batch)
- `nv_inference_queue_duration_us` - Time spent waiting for batch
- `nv_inference_compute_infer_duration_us` - Pure GPU compute time
- `nv_inference_compute_input_duration_us` - Data transfer to GPU
- `nv_inference_compute_output_duration_us` - Data transfer from GPU

### Current SCRFD Breakdown (from Triton metrics)
- GPU compute: 1.2ms per execution
- Queue delay: 5.3ms (configured max)
- Input transfer: 1.4ms
- Output transfer: 1.1ms
- Total server: 10.2ms
- Client overhead: 7ms (gRPC marshal + network)
- Throughput: ~100 RPS at 4 clients (Python GIL limited)

### Hardware
- GPU: NVIDIA RTX A6000 (48 GB VRAM, Ampere)
- RAM: 503 GB
- Reference: RTX 4090 is ~1.5-2x faster in TRT FP16 for face detection models
