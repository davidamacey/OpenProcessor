# PE-Core-L14-336 batched TensorRT: parity and throughput

Status: research snapshot (2026-05-15); measured on RTX 3080 Ti 12 GB, single Triton instance; numbers will drift.

## Setup

- Model: PE-Core-L14-336 image encoder, served by Triton as `pe_image_encoder`.
- Engine: FP16 TensorRT plan, 612 MiB.
- Optimization profile: `--minShapes=images:1x3x336x336 --optShapes=images:8x3x336x336 --maxShapes=images:32x3x336x336`.
- Triton: `max_batch_size=32`, dynamic batching with `preferred_batch_size=[4,8,16,32]`, one instance.

## 1. Parity versus single-image reference

Eight real crops were each served twice: once alone (b=1, reference) and once as part of a b=8 batched call. Cosine similarity per crop between the two outputs:

| Metric | Value |
|---|---|
| min cosine | **1.000000** |
| mean cosine | 1.000000 |
| max drift (1 - min_cos) | **0.000000** |

Batched output is bit-equivalent to the reference at FP16 on this input set.

## 2. Throughput

30 iterations per batch size after 3 warm-up iterations, one Triton instance, no client-side concurrency.

| Batch | p50 crops/s | p95 crops/s | p50 latency (ms) | Speedup vs b=1 |
|---:|---:|---:|---:|---:|
| 1 | 29.8 | 30.4 | 33.6 | 1.00x |
| 4 | **90.2** | 95.3 | 44.4 | 3.03x |
| 8 | 76.6 | 81.7 | 104.7 | 2.57x |
| 16 | 75.2 | 81.0 | 212.7 | 2.52x |
| 32 | 77.1 | 81.6 | 415.1 | 2.59x |

## 3. Interpretation

- The sweet spot is b=4 at about 90 crops/s. Above b=4 the forward pass saturates near 80 crops/s: extra batching stops amortizing per-call overhead because the model's compute becomes the bottleneck.
- Unbatched serving gives about 30 crops/s; batched about 80 crops/s, a sustained speedup of about 2.6x.
- Typical ingest chunks of 1-4 crops per image land in the sweet spot.
- Above b=4, latency scales near-linearly with batch size (104.7, 212.7, 415.1 ms for b=8/16/32, roughly doubling per step). The engine effectively processes the batch serially on the GPU; the gain is call-overhead amortization only.

## 4. Comparison to the unbatched ONNX Runtime path

The earlier ONNX Runtime serving used `max_batch_size=0` with a fixed `[1,3,336,336]` input, so single-image inference with no Triton batching. It reached about 20-25 crops/s (gRPC overhead and per-call Python preprocessing dominate). The batched TensorRT path is 2.6-3x faster with bit-equivalent output.

## 5. GPU memory

On a 12 GB card already hosting the other detectors, Triton went from about 8.8 GiB to about 9.4 GiB total after adding the PE engine (612 MiB plan plus roughly 3 GiB of activations is the upper estimate). That fits. Scaling PE to more instances needs a larger card.

## 6. Regression criteria

After changing the PE export (model variant, opset, TensorRT version), the batched output should still show max drift of 1e-3 or less versus the b=1 reference, and b=4 should reach at least 80 crops/s on a 3080 Ti-class card; otherwise treat the change as a regression.

## How to repeat

The public repo ships the engine config at `models/pe_image_encoder/config.pbtxt` (`max_batch_size: 32`, `preferred_batch_size: [4, 8, 16, 32]`, one instance) and the client in `src/clients/pe_encoder.py`, which chunks crops at `PE_CROP_MAX_BATCH = 32`. The original standalone benchmark script is not included in the repo. To reproduce:

1. Send N crops one at a time (batch 1) through Triton for the model `pe_image_encoder`, then the same crops in batches of 4, 8, 16, and 32.
2. Use 3 warm-up iterations and 30 timed iterations per batch size; report p50/p95 crops/s and p50 latency.
3. Compute cosine similarity between each crop's b=1 embedding and its embedding from the b=8 call.

Triton's `perf_analyzer` (from the `triton-sdk` benchmark profile in `docker-compose.yml`) can drive step 1-2 with `--shape images:3,336,336` and `-b <batch>`.
