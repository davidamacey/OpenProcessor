# Detector quantization: precision, runtime and hardware matrix

Status: research snapshot (2026-05); measured on an RTX A6000 with a Xeon E5-2680 v3 host (no VNNI) and an Apple Silicon Mac Studio (macOS 15.6); numbers will drift.

Implemented in the public repo: no. This is a measurement record, not a feature; no quantization or throughput harness is shipped. Commands below are intentionally omitted.

The subject is a single curated YOLO-nano-class detector for a full-frame small-region example task (an NMS-free end-to-end export), evaluated at several precisions and runtimes. Weights are frozen (post-training quantization, no QAT). FP16 ONNX comes from the Ultralytics export; INT8 ONNX uses `onnxruntime.quantize_static` (QDQ, Conv and MatMul only) calibrated on 1,000 training images. The frozen test split has 5,837 frames (4,704 with plates, 1,133 background).

## Accuracy and size (COCOeval)

| Variant | Runtime | mAP@.5:.95 | mAP@.5 | AP_small | mean IoU | Size (MB) | delta mAP vs FP32 |
|---|---|--:|--:|--:|--:|--:|--:|
| `.pt` reference | ultralytics | 0.8077 | 0.9615 | 0.7720 | 0.9059 | 5.14 | - |
| FP32 ONNX | ort-cuda | 0.8211 | 0.9681 | 0.7876 | 0.9107 | 9.35 | baseline |
| FP16 ONNX | ort-cuda | 0.8213 | 0.9681 | 0.7874 | 0.9102 | 4.74 | +0.000 |
| INT8 QDQ ONNX | ort-cpu | 0.8118 | 0.9658 | 0.7767 | 0.9086 | 2.88 | -0.009 |

FP16 is accuracy-identical to FP32 at half the size. INT8 loses under one mAP point with no small-object cliff at 3.2x smaller. Accuracy is execution-provider independent (CoreML may differ in the last decimal from floating-point ordering).

## Throughput: steady state, full `detect()` (letterbox, model, decode, NMS)

200 frames preloaded in memory (no disk I/O), 30-frame warmup, at least 8 s steady state.

### NVIDIA RTX A6000 and Xeon E5-2680 v3

| Variant | Provider | img/s | mean ms | vs `.pt` |
|---|---|--:|--:|--:|
| `.pt` ultralytics | CUDA | 53.5 | 18.7 | 1.0x |
| FP32 ONNX | ORT-CUDA | 101.8 | 9.8 | 1.9x |
| FP16 ONNX | ORT-CUDA | 56.6 | 17.7 | 1.06x |
| INT8 ONNX | ORT-CUDA | 67.8 | 14.8 | 1.27x |
| FP32 ONNX | ORT-CPU | 38.5 | 26.0 | 0.72x |
| FP16 ONNX | ORT-CPU | 22.0 | 45.4 | 0.41x |
| INT8 ONNX | ORT-CPU | 10.9 | 91.9 | 0.20x |

- Leaving the Ultralytics Python path for ONNX Runtime is the largest single win (about 1.9x).
- FP16's benefit is size, not CUDA-EP speed: on the CUDA execution provider FP16 is slower than FP32 (cast overhead; the CUDA EP does not engage FP16 tensor cores automatically).
- INT8 is slow on this CPU because Haswell lacks VNNI INT8 acceleration and QDQ overhead dominates. Expect the opposite on VNNI/AVX-512 CPUs and on ARM with dot-product instructions.

### Apple Silicon (Mac Studio)

| Variant | Runtime | img/s | mean ms | Notes |
|---|---|--:|--:|---|
| FP32 ONNX | ort-coreml | - | - | crash: `GatherElements out of range` in the decode head |
| FP32 ONNX | ort-cpu | 38.9 | 25.7 | |
| FP16 ONNX | ort-coreml | - | - | crash (same bug) |
| FP16 ONNX | ort-cpu | 36.8 | 27.2 | |
| INT8 ONNX | ort-coreml | 12.1 | 82.7 | heavy CPU fallback |
| INT8 ONNX | ort-cpu | 54.1 | 18.5 | fastest portable path on Mac |
| INT8 CoreML (native) | ANE | 348.3 | 2.87 | fastest overall |
| FP16 CoreML (native) | ANE | 334.5 | 2.99 | |
| INT8 CoreML (native) | CoreML-CPU | 69.5 | 14.4 | |
| FP16 CoreML (native) | CoreML-CPU | 68.3 | 14.7 | |

- Native CoreML on the Neural Engine is the clear winner: 348 img/s (2.87 ms), about 5x the same package on the Apple CPU, 6.4x the ORT-CPU path and 3.4x the best Linux GPU path above.
- INT8 on ORT-CPU (54.1 img/s) is about 1.4x FP32, the reverse of the no-VNNI Xeon.
- ORT's CoreML execution provider does not work for this NMS-free export: its decode tail uses `GatherElements`, which the provider mis-indexes (crash for FP32/FP16, slow CPU fallback for INT8). Apple Neural Engine acceleration must come from a native CoreML package.

## TensorRT execution provider on the GPU (negative result)

| Variant | Cold engine build | Cached load | Warm throughput |
|---|--:|--:|--:|
| FP16 TRT-EP | 330.7 s (~5.5 min) | 0.19 s | 93.2 img/s |
| INT8 TRT-EP | - | - | fails to build |

- The cold build is about 5.5 minutes and the engine is locked to the GPU architecture and TensorRT version, a poor step to run on a user machine.
- Even built, FP16 TRT (93.2) is slower than FP32 ONNX on ORT-CUDA (101.8): for a nano model the convolution compute is tiny and fixed per-image Python pre/post-processing dominates.
- INT8 fails because the QDQ model uses asymmetric activations (MinMax) and TensorRT needs symmetric INT8 (`Non-zero zero point is not supported`); `ActivationSymmetric=True` would fix it but is pointless given the above.
- The TRT EP needs the TensorRT libraries on `LD_LIBRARY_PATH`, otherwise it silently falls back to CUDA.

Conclusion: for a nano detector, a portable ONNX run as FP32 on ORT-CUDA gives 1.9x with zero build. This conclusion is specific to small models where pre/post-processing dominates; a large model behaves differently (see [triton_perf_crossvalidation.md](triton_perf_crossvalidation.md)).

## Recommendations

- NVIDIA or server: ship portable ONNX run through ORT-CUDA; FP16 ONNX as the default storage format (half size, identical accuracy). For maximum GPU speed on larger models, build TensorRT engines locally.
- Cross-platform desktop: one portable ONNX through a runtime such as the Rust `ort` crate (CoreML on macOS, DirectML or CPU on Windows, CUDA or CPU on Linux); add a native CoreML package only if the Neural Engine win justifies a second pipeline.
- Linux and Windows: INT8 or FP16 ONNX via ONNX Runtime; macOS: native CoreML on the Neural Engine. Ship INT8 where size or power matter (negligible accuracy cost).
- Never ship a serialized TensorRT `.plan`; it is locked to GPU, driver and TensorRT version.
