# Triton performance cross-validation: batch size vs output payload

Status: research snapshot (2026-09); measured on one RTX A6000 serving a live Triton deployment, driven by `perf_analyzer`; numbers will drift.

Implemented in the public repo: not applicable (measurement record). The end-to-end small detector used as the control, `yolov11_small_trt_end2end`, is in `models/`. The large-output detector measured below is a legacy YOLOv5-fork export that is not part of the public model set; the finding applies to any model with a similarly large raw output.

## Origin

An independent benchmark harness (a separate Rust inference-server project) needed a real-world Triton throughput reference. Rather than stand up a throwaway server, it queried a live deployment with `perf_analyzer`, read-only, against two already-deployed models: a large 1280 px vehicle detector with a raw output, and the small NMS-baked `yolov11_small_trt_end2end`.

## Existing axis: concurrency

A prior concurrency sweep of the large-output detector, at whatever batch the dynamic batcher formed, gave about 4.6 (c=1), 9.0 (c=8), 8.5 (c=16) and 8.2 (c=32) crops per second. That sweep did not isolate batch size.

## New axis: batch size at fixed concurrency 1

| batch | img/s | latency |
|------:|------:|--------:|
| 1 | 3.60 | 163 ms |
| 4 | 3.23 | 556 ms |
| 8 | 2.07 | 1083 ms |

Throughput falls as batch grows, the opposite of the usual benefit. Cause: the model's `output0` is `[batch, 100800, 85]` FP32, so a batch of 8 is about 274 MB in one gRPC/HTTP response, and moving that payload dominates wall time when concurrency is too low for Triton to amortize it across callers.

Control: the small end-to-end detector (NMS baked in, outputs such as `num_dets` and a `[300, 4]` box tensor) scaled cleanly.

| batch | img/s |
|------:|------:|
| 1 | 18.67 |
| 4 | 59.62 |
| 8 | 63.44 |

So the scaling failure comes from payload size, not from Triton or TensorRT.

## Implications

- If the large-output model's `config.pbtxt` sets `preferred_batch_size: [4, 8, 16]`, a low-concurrency caller (an ad-hoc script, debugging, an off-peak trickle) can trigger large batches that this measurement shows are worse than batch 1. When production keeps concurrency high (the c=8 sweet spot suggests it does), the effect is amortized away. Verify rather than assume; if low-concurrency callers exist, lower the preferred batch sizes for this model.
- Shrinking the output is the structural fix. An NMS-free end-to-end export (one conflict-free box per object, fixed `[N, 300, 6]` output) cuts the payload by orders of magnitude and should show the clean batch scaling of the control. Ultralytics reports about +43% CPU inference speed over YOLO11n from NMS removal alone. An NMS cannot be baked into a YOLOv5-fork export graph the way the YOLO11 end-to-end export allows; an NMS-free architecture removes that constraint.

## Suggested method for future models

Benchmark candidates both concurrency-swept and batch-swept (fixed concurrency 1, batch 1/4/8) with `perf_analyzer`. The batch sweep catches output-payload regressions before deployment.
