# Ingest concurrency tuning: small batches, high concurrency

Status: research snapshot (2026-05-17); measured on RTX 3080 Ti 12 GB running Triton detectors and encoders, a 32-worker API process, network-attached bulk storage; numbers will drift.

The source material was a sweep over client batch size and submit concurrency for a bulk-ingest path. The result is generic to any client that feeds a server running Triton dynamic batching, so only that part is kept. Storage-specific details (the disk array, reader thread counts tuned to it) are omitted.

## Headline

| Setting | Rate (img/s) | GPU mean utilization |
|---|---:|---:|
| Batch 64, concurrency 16 | 3.1 | 9.0 % |
| Batch 8, concurrency 8 | 4.7 | 4.6 % |
| Batch 1, concurrency 128 | **16.1** (3.4x over batch 64) | 41 % |

## Sweep

500 images per run, indexes emptied before each run, no auto-labeling (it adds 1-3 minutes of unrelated work), 32 API workers.

| Batch | Concurrency | Rate (img/s) | Elapsed (s) | GPU mean | GPU p95 | Failures |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 8 | 4.72 | 106 | 4.6 % | 22 % | 0 |
| 8 | 16 | 6.17 | 81 | 15.4 % | 56 % | 0 |
| 8 | 32 | 9.09 | 55 | 22.8 % | 69 % | 0 |
| 8 | 48 | 8.77 | 57 | 21.4 % | 82 % | 0 |
| 8 | 64 | 8.33 | 60 | 24.4 % | 71 % | 0 |
| 16 | 32 | 6.25 | 80 | 14.0 % | 57 % | 0 |
| 32 | 32 (more readers, larger queue) | 4.34 | 115 | n/a | n/a | **16** (bulk-write timeout) |
| 64 | 16 | 3.11 | 161 | 9.0 % | 46 % | 0 |
| 4 | 32 | 13.16 | 38 | 25.8 % | 70 % | 0 |
| 4 | 64 | 11.90 | 42 | 22.0 % | 60 % | 0 |
| 2 | 64 | 14.29 | 35 | 28.5 % | 83 % | 0 |
| 2 | 128 | 15.15 | 33 | 33.9 % | 87 % | 0 |
| **1** | **128** | **16.67 / 15.62 / 16.13** | **30-32** | **41 %** | **66-77 %** | 0 |
| 1 | 256 | 15.62 | 32 | 40.9 % | 66 % | 0 |

Batch 1 / concurrency 128 reproduced across 4 runs at 15.6-16.7 img/s.

## Why small batches win

Per request, the ingest path does JPEG decode, object detection, a custom detector, image-encoder embedding, then a bulk index write. Segmentation and VLM work run asynchronously afterwards and are not part of this measurement.

Triton does dynamic batching server-side with a max-queue-delay window. Small client batches with high concurrency keep many requests in flight, so Triton coalesces them across instances and keeps the GPU busy. Large client batches (32-64) did the opposite: they serialized the per-request bulk index writes (one large write per response), and at batch 32 the bulk-write path hit connection timeouts on 16 of 500 images.

## Bottleneck at the best setting

Triton metrics for one ingest cycle at batch 1 / concurrency 128:

| Model | exec_count | queue (s) | compute (s) | queue / compute |
|---|---:|---:|---:|---:|
| YOLO11 small detector | 882 | 2.3 | 8.5 | 0.27 |
| custom vehicle detector | 882 | **112** | 32 | **3.5** |
| PE image encoder | 2,638 | 89 | 282 | 0.32 |

The custom detector was queue-bound with a single instance on a 12 GB card. CPU was not the limit (API container about 1566 % of 32+ cores, Triton about 140 %), while GPU mean was 41 % with 100 % peaks, so there was about 2-3x compute headroom waiting behind that single-instance queue. The next lever is raising that model's `instance_group` count from 1 to 2, which needs a VRAM check first (Triton sat at about 8.8 GB of 12 GB) and possibly an engine rebuild with a smaller workspace.

## Generic lessons

- Prefer small client batches (1-4) with high concurrency (about 128) when the server does dynamic batching. Re-run the sweep if preferred batch sizes or instance counts in the model configs change; the knee moves.
- Look at per-model queue/compute ratios in Triton metrics to find the real bottleneck. A ratio well above 1 means more instances (or a faster engine) for that model, not more client threads.
- Pushing concurrency past the saturation point (256 versus 128) did not help and slightly regressed.
- Large batches can fail through the downstream index's bulk timeout before the GPU is a factor.
- File-reader threads and queue depth were not on the critical path at these batch sizes, because read latency was hidden behind GPU wait.
- Empty the indexes and verify before each run; use the same input set; reproduce the winner several times.

## How to repeat

The public repo has a directory ingest script, `scripts/curation/ingest_walker.py`, with `--batch-size` (default 32), `--concurrency` (default 4, in-flight batch POSTs), and `--walk-workers`. To repeat the sweep:

```bash
python scripts/curation/ingest_walker.py --root <image-dir> --project <project> \
  --batch-size 1 --concurrency 128
```

Vary `--batch-size` and `--concurrency` over the grid above, time each run, and sample GPU utilization with `nvidia-smi --query-gpu=utilization.gpu --format=csv -l 1`. Triton queue and compute times come from its Prometheus metrics (`nv_inference_queue_duration_us`, `nv_inference_compute_infer_duration_us`, `nv_inference_exec_count`). The measurements above used a different bulk-ingest endpoint and driver, so absolute rates for this script will differ; the batch/concurrency trend is what to compare.
