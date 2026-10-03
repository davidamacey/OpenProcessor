# DALI, GPU decode and server-side preprocessing: lessons

Status: research snapshot (2026-09); measured on one RTX A6000 under a host load average of about 20, with 20 MP (5472x3648) JPEGs and YOLO26n / YOLO11n at 640 px; numbers are preliminary and will drift.

Implemented in the public repo: no. There is no DALI pipeline or DALI-based Triton ensemble in the repo today (preprocessing is CPU-side, for example `letterbox_cpu` in `src/services/cpu_preprocess.py`). Everything below is guidance for anyone building GPU decode into an ensemble or BLS; the DALI snippet and the config are proposals. Measurements came from an independent benchmark harness (a separate Rust YOLO server) compared against Triton.

## 1. What Triton 26.08 ships

- `tritonserver:26.08-py3` bundles the DALI backend with DALI 2.2.0, TensorRT 11.2.1 and ONNX Runtime 1.28.0; no custom build is needed. Serialize pipelines with the image's own Python so the graph matches the backend's DALI version. Serializing a `mixed` (GPU) pipeline needs a GPU (`--gpus`).
- DALI 2.3 has no breaking changes for this use; the container's 2.2.0 is fine.
- TensorRT 11 is strongly typed: FP16 plans come from an FP16-weight ONNX (FP32 I/O kept), not a `--fp16` builder flag on an FP32 graph.

## 2. Benchmark Triton with host networking

With `-p 8000:8000` every byte goes through Docker's userland proxy; a raw 640 px FP32 request is about 4.9 MB, a real extra copy. Switching Triton to `--network host` raised TRT FP16 raw-tensor throughput from 112 to 145 img/s at 8 clients and from 219 to 260 at 32 clients (YOLO11n). A Triton benchmark that uses port mapping understates Triton.

## 3. Making DALI letterboxing match Ultralytics (three traps)

A letterbox that looked right changed detections on about 18 of 32 test images versus the Ultralytics CPU reference.

1. **`fn.resize` defaults differ from cv2.** DALI defaults to `antialias=True` and `subpixel_scale=True`; Ultralytics uses `cv2.resize(..., INTER_LINEAR)` (plain bilinear, no antialiasing, onto the rounded output size). Set `antialias=False, subpixel_scale=False, interp_type=types.INTER_LINEAR`.
2. **Pad split.** Ultralytics uses `top = round(dh - 0.1)` and `left = round(dw - 0.1)`, so an odd pad puts the extra row at the bottom/right. DALI's centred crop (`crop_pos 0.5`, `out_of_bounds_policy="pad"`) puts it at the top (107 rows above instead of 106 for a 427-row image), which alone moved confidences by up to about 0.3. Compute the pads and set `crop_pos = pad / (edge - new)`.
3. **A GPU image's shape cannot be read inside Triton.** `images.shape()` after a `mixed` decode fails in the Triton DALI backend ("GPU->CPU transitions are not allowed in legacy execution model"). Use `fn.peek_image_shape` on the encoded bytes on the CPU.

Working pipeline (DALI 2.2, serialized in the 26.08 image; a proposal for this repo):

```python
from nvidia.dali import fn, pipeline_def, types
from nvidia.dali import math as dmath

@pipeline_def(batch_size=MAX_BATCH, num_threads=8, device_id=0)
def letterbox():
    jpegs = fn.external_source(device="cpu", name="IMAGE", dtype=types.UINT8)
    images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)  # "cpu" for CPU decode
    images = fn.resize(images, resize_longer=EDGE, interp_type=types.INTER_LINEAR,
                       antialias=False, subpixel_scale=False)
    shape = fn.cast(fn.peek_image_shape(jpegs), dtype=types.FLOAT)   # CPU, from the JPEG header
    h0, w0 = shape[0], shape[1]
    r = EDGE / dmath.max(h0, w0)
    h = dmath.floor(h0 * r + 0.5)
    w = dmath.floor(w0 * r + 0.5)
    pad_top = dmath.floor((EDGE - h) * 0.5 - 0.1 + 0.5)    # Ultralytics round(dh - 0.1)
    pad_left = dmath.floor((EDGE - w) * 0.5 - 0.1 + 0.5)
    images = fn.crop(images, crop=(EDGE, EDGE),
                     crop_pos_y=pad_top / dmath.max(EDGE - h, 1.0),
                     crop_pos_x=pad_left / dmath.max(EDGE - w, 1.0),
                     out_of_bounds_policy="pad", fill_values=114)
    images = fn.crop_mirror_normalize(images, dtype=types.FLOAT, output_layout="CHW",
                                      mean=[0.0, 0.0, 0.0], std=[255.0, 255.0, 255.0])
    return images   # for device="cpu" decode: images.gpu()
```

After the fix the DALI tensor is pixel-aligned with the reference letterbox (best shift (0,0) on every image, mean absolute difference 0.08 grey levels; the GPU path's larger maximum difference comes from nvJPEG's IDCT, not layout). YOLO26n detections matched the CPU reference on all four DALI ensembles (GPU/CPU decode x ONNX/TRT FP16). For YOLO11n every box matched (IoU >= 0.995) with one borderline confidence off by 0.02-0.04 on one image, the same FP16 rounding plain TRT FP16 shows.

### Reference choice for a CPU letterbox

`PIL.Image.resize(..., Image.BILINEAR)` widens its filter support when downscaling (effectively antialiased), unlike `cv2.INTER_LINEAR`. On a 20 MP to 640 px downscale (about 8.5x) the model input visibly differs from what Ultralytics `predict()` feeds the model. Neither is wrong, but moving to DALI or any cv2-style GPU resize shifts detections slightly against a PIL path. Choose the reference deliberately (Ultralytics parity or the current PIL outputs) and validate against it before switching. A half-pad split of `int((target - new) / 2)` already matches Ultralytics' `round(d - 0.1)`, because the half-pad is always k or k + 0.5.

## 4. DALI in an ensemble: configuration that worked

- DALI model: `backend: "dali"`, `max_batch_size: 8`, dynamic batching, `instance_group [{ kind: KIND_GPU, gpus: [0] }]`; input `IMAGE TYPE_UINT8 dims [-1]` with `allow_ragged_batch: true` (JPEG lengths differ); output FP32 `[3, E, E]`.
- Ensemble: input `IMAGE UINT8 [-1]`, then DALI, then the model's `images` input, then `output0`. The client sends raw JPEG bytes as a `UINT8 [1, n]` tensor.
- DALI dynamic-batches like the models: Triton statistics showed average batches of 3-8 under load.
- Validate with a tensor-level diff first: run the DALI model alone, compare against a reference letterbox, and search plus or minus 1 px shifts. An off-by-one pad appears immediately as a consistent `dy = 1`, while detection-level diffs only say that some images changed.

## 5. Measured throughput (YOLO26n 640, TRT FP16; img/s with p99 ms in parentheses)

Same GPU, images and closed-loop client; max batch 8, 2 ms queue delay, one instance per model; JPEG in, 20 MP photos.

| Clients | Triton: DALI GPU decode, TRT | Triton: DALI CPU decode, TRT | Triton: client decodes, TRT raw | Independent harness: nvJPEG + GPU letterbox, TRT |
|---|---|---|---|---|
| 1 | 14.6 (93) | 3.6 (322) | 3.8 (316) | 18.6 (78) |
| 8 | 44.5 (264) | 10.7 (986) | 31.3 (319) | 135.3 (89) |
| 32 | 106.2 (396) | 27.6 (1330) | 89.5 (459) | 155.5 (261) |
| 128 | 119.2 (1537) | 27.6 (5214) | 97.6 (2361) | 178.0 (1394) |

- GPU decode is the lever for 20 MP input. CPU decode caps a single GPU at about 90-100 img/s even with many client threads (GPU utilisation 7-29%). DALI GPU decode raised Triton's ceiling about 20% over client-side decode and cut p99 substantially at low concurrency.
- DALI CPU decode inside Triton is the worst option (about 28 img/s cap with `num_threads=8`): decode runs in the DALI instance's thread pool, serialized per batch. If GPU decode is unavailable, decode in many client processes instead.
- DALI GPU at 8 clients (44 img/s) is well below what the GPU can do. Untested tuning ideas: two or more DALI instances (`instance_group count`), `preallocate_width_hint`/`preallocate_height_hint` and `device_memory_padding`/`host_memory_padding` sized for 20 MP frames, `hw_decoder_load` (hardware JPEG decoder on A100/H100; absent on the A6000) and larger `num_threads`.
- The independent harness's own nvJPEG path (decode and letterbox on the GPU, tensor never leaves the device, pipelined with inference) was 1.5-3x the DALI ensemble. The difference is pipelining and per-request latency, not the codec; the same nvJPEG library is under both.

## 6. Other findings for Triton model serving

- **Triton's `onnxruntime` backend collapses under mixed batch sizes.** YOLO26n ONNX on the CUDA EP: 11-15 img/s with p99 about 0.8-0.9 s at 8 clients, versus about 300 for the TRT plan. The CUDA EP rebuilds cuDNN convolution plans whenever the batch dimension changes (1.5-2.7 s per new shape, measured). Use TensorRT plans, or pad every batch to one shape.
- **Raw-tensor serving (client sends a letterboxed FP32 tensor).** Triton's TRT FP16 path is excellent at high concurrency: 567 img/s at 32 clients and 626 at 128 (YOLO26n), about 300 at 8 clients. When decode happens elsewhere, Triton is not the bottleneck; keep TRT models on raw tensors and move decode and letterbox to the GPU (DALI) or to many client processes.
- **End-to-end export output size.** A YOLO26 end-to-end export (`[batch, 300, 6]`, NMS-free) returns about 7 KB per image versus about 2.8 MB for YOLO11's raw `[84, 8400]`. Over HTTP that alone changes high-concurrency throughput (YOLO11 raw TRT: 475 img/s at 128 clients versus 626 for YOLO26).

## 7. YOLO26: which export is actually NMS-free

With Ultralytics 8.4.156:

- `export(format="onnx", nms=True)` appends a classic ONNX NMS block (`NonMaxSuppression` plus `NonZero`) to the one-to-many head. This is not YOLO26's NMS-free model. CoreML cannot run it (data-dependent `NonZero` output shape); TensorRT and CUDA can.
- `export(format="onnx", nms=False)` exports the real end-to-end one-to-one head: `TopK` only, metadata `end2end: True`, output `[batch, 300, 6]`. `nms=None` (default) gives the raw one-to-many head `[batch, 84, anchors]`.
- The two heads score differently. `YOLO("yolo26n.pt").predict()` defaults to the one-to-many head plus NMS for this checkpoint (`end2end=False`). Validate an end-to-end export against `predict(..., end2end=True)`; on one box the heads gave 0.557 versus 0.331. Against the wrong reference a correct end-to-end deployment looks like a bug.

## 8. Parity-check method (worth copying)

Every configuration was checked before it was timed. The same 32 images went through each server path and the detections were compared with a CPU-execution-provider reference: matched at IoU >= 0.5, with box IoU >= 0.9 and confidence within 0.02 counted as the same. Each image gets a verdict of identical, rounding-only or changed. A preprocessing-only mode diffs the preprocessing model's output tensor against the reference letterbox and reports the best-aligned (dx, dy) shift; that mode found the DALI pad bug in one run.
