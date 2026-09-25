# Model Export Scripts

This folder contains scripts for exporting models to TensorRT format for NVIDIA Triton Inference Server deployment.

## Overview

The export process transforms PyTorch models into optimized TensorRT engines for high-performance GPU inference.

## Export Scripts

| Script | Purpose | Output |
|--------|---------|--------|
| `export_models.py` | YOLO11 object detection with end2end NMS | TensorRT engine |
| `export_detector_dual_head.py` | Any YOLO-family detector, re-exported with a backbone feature-map output | ONNX + TensorRT engine |
| `export_detector_dual_head.sh` | `trtexec` engine build + model-repo install for the above | TensorRT engine |
| `export_scrfd.py` | SCRFD-10G face detection + landmarks | TensorRT engine |
| `export_face_recognition.py` | ArcFace face embeddings | TensorRT engine |
| `export_mobileclip_image_encoder.py` | MobileCLIP image encoder | TensorRT engine |
| `export_mobileclip_text_encoder.py` | MobileCLIP text encoder | TensorRT engine |
| `download_pe_weights.py` | Pinned, SHA-256-verified PE-Core-L14-336 checkpoint (both PE exporters) | HF cache |
| `export_pe_image_encoder.py` | PE-Core-L14-336 image encoder (curation `pe_embedding`) | ONNX + `config.pbtxt` |
| `build_pe_trt.sh` | PE-Core ONNX → TensorRT engine (Path 1) | TensorRT engine |
| `build_pe_ort_fallback.sh` | PE-Core ONNX served by Triton's ORT backend (Path 2) | ONNX model dir |
| `export_pe_text_encoder.py` | PE-Core-L14-336 text encoder (semantic-search queries), parity-gated | ONNX (+ optional Triton ORT model dir) |
| `export_paddleocr_det.py` | PP-OCRv5 text detection | TensorRT engine |
| `export_paddleocr_rec.py` | PP-OCRv5 text recognition | TensorRT engine |
| `download_face_models.py` | Download pre-trained face models | PyTorch weights |
| `download_paddleocr.py` | Download PP-OCRv5 models | ONNX models |
| `download_pytorch_models.py` | Download YOLO11 PyTorch models | PyTorch weights |

## Model Directory Structure

```
pytorch_models/
├── yolo11s.pt                          # YOLO11 PyTorch model
├── arcface_w600k_r50.onnx              # ArcFace ONNX model
├── mobileclip2_s2/                     # MobileCLIP checkpoint
├── mobileclip2_s2_image_encoder.onnx   # MobileCLIP image encoder ONNX
├── mobileclip2_s2_text_encoder.onnx    # MobileCLIP text encoder ONNX
├── pe_image_encoder.onnx               # PE-Core-L14-336 image encoder ONNX
└── pe_text_encoder.onnx                # PE-Core-L14-336 text encoder ONNX (API loads in-process)

models/
├── yolov11_small_trt/                  # YOLO11 TensorRT (standard)
│   ├── 1/model.plan
│   └── config.pbtxt
├── yolov11_small_trt_end2end/          # YOLO11 TensorRT with GPU NMS
│   ├── 1/model.plan
│   └── config.pbtxt
├── scrfd_10g_bnkps/                    # SCRFD-10G face detection TensorRT
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
├── pe_image_encoder/                   # PE-Core-L14-336 image encoder
│   ├── 1/model.plan                    #   Path 1 (TensorRT), OR
│   ├── 1/model.onnx                    #   Path 2 (ONNX Runtime) - never both
│   └── config.pbtxt
├── pe_text_encoder/                    # PE-Core-L14-336 text encoder (optional)
│   ├── 1/model.onnx                    #   --install-triton
│   └── config.pbtxt                    #   onnxruntime_onnx, KIND_CPU
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

### Dual-Head Detector (detections + backbone embedding source)

The curation subsystem stores a per-item **backbone embedding**
(`backbone_embedding`, dimension `CurationConfig.backbone_embedding_dim`) and
consumes it in residual clustering, the embedding visualization, item
scores and the OCC conflict handler. It is produced by RoI-pooling a
detector's backbone feature map over each detection box
(`src.services.detection.geometry.roi_pool_sppf`) — which requires the
feature map to be on the wire. A stock detector export emits only the
detection tensor, so the detector has to be re-exported with a second
output:

| Output | Shape | Meaning |
|--------|-------|---------|
| `output0` | family-specific (YOLOv5 `[B, N, 5 + nc]`, Ultralytics v8+ `[B, 4 + nc, N]`) | Detection tensor, unchanged from a single-head export |
| `sppf_feat` | `[B, C, H, W]` where `H = W = imgsz / 32` | Backbone bottleneck (SPPF) feature map |

```bash
# ONNX only (CPU-friendly; validates via onnxruntime round-trip)
docker compose exec yolo-api python /app/export/export_detector_dual_head.py \
    --weights /app/pytorch_models/my_detector.pt \
    --triton-name my_detector_dual_head --imgsz 640

# ONNX + TensorRT engine + config.pbtxt + labels.txt into the model repo
docker compose exec yolo-api python /app/export/export_detector_dual_head.py \
    --weights /app/pytorch_models/my_detector.pt \
    --triton-name my_detector_dual_head \
    --imgsz 1280 --max-batch 16 --formats onnx trt

# Legacy YOLOv5-fork checkpoint (the fork is only needed to load its checkpoints)
docker compose exec yolo-api python /app/export/export_detector_dual_head.py \
    --weights /app/pytorch_models/legacy_v5.pt --loader yolov5 \
    --yolov5-fork /app/external/yolov5 \
    --imgsz 1280 --triton-name legacy_v5_dual_head
```

Nothing is hardcoded to one model: checkpoint, Triton name, input size,
tapped module (`--feature-module`/`--feature-index`) and both output
names are CLI arguments. The tapped module is found by **class name**
(`SPPF` by default), not by a fixed layer index, so architecture drift
cannot silently tap the wrong layer.

Where `trtexec` is available but the TensorRT Python bindings are not
(e.g. inside the triton-server container), build the engine with the
shell companion instead of `--formats trt`:

```bash
export/export_detector_dual_head.sh \
    --onnx pytorch_models/my_detector_dual_head.onnx \
    --name my_detector_dual_head --input-size 1280 --max-batch 16
```

Whichever path builds the engine, the model's `config.pbtxt` must declare
**both** outputs — Triton serves only the tensors its config names, so
omitting `sppf_feat` silently drops the feature map even though the
engine produces it. `--formats trt` writes that file for you.

### SCRFD Face Detection

```bash
docker compose exec yolo-api python /app/export/export_scrfd.py
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

### PE-Core Encoders (Curation Embeddings)

**Required by the curation subsystem.** PE-Core-L14-336 (Meta's Perception
Encoder) provides both towers of one shared 1024-d embedding space:

| Tower | Consumer | Serving | Contract (hardcoded in `src/clients/pe_encoder.py`) |
|-------|----------|---------|-----------------------------------------------------|
| Image | `PEEncoder.encode_images` → `pe_embedding` field (semantic search, near-dup, clustering, embedding viz) | Triton `pe_image_encoder`, **required** | `images` FP32 `[B, 3, 336, 336]` → `image_embeddings` FP32 `[B, 1024]` |
| Text  | `PEEncoder.encode_text` → `GET /curation/search/text` queries | **Triton `pe_text_encoder`** (preferred: one shared CPU instance for every uvicorn worker), lazily-loaded in-process PyTorch fallback; in-process ONNX Runtime is an explicit opt-in only | `text_tokens` INT64 `[B, T≤32]` → `text_embeddings` FP32 `[B, 1024]` |

Both embeddings come out L2-normalized. See
[`docs/CURATION.md`](../docs/CURATION.md#models-you-must-supply) for what
depends on them.

**Fresh deployment, end to end** (API container = `yolo-api`; it has torch,
`perception_models` and the HF cache mounted from `./cache/huggingface`):

```bash
make pe-download        # 0. weights: pinned commit + SHA-256 into the HF cache
make pe-export-image    # 1. image tower -> pytorch_models/pe_image_encoder.onnx
make pe-build-trt       # 2. -> models/pe_image_encoder/1/model.plan (Path 1)
#   make pe-build-ort   #    ...or serve the ONNX via Triton ORT (Path 2 fallback)
make pe-export-text-triton  # 3. text tower -> pytorch_models/pe_text_encoder.onnx
                             #    + installs models/pe_text_encoder/1/model.onnx + config.pbtxt
# 4. load both models in Triton (explicit model control): add
#    --load-model=pe_image_encoder and --load-model=pe_text_encoder to
#    the triton-server command (both are in docker-compose.yml's default
#    load list already), then
make restart-triton
make pe-text-status     # expect "backend": "triton"
```

`make export-pe` runs steps 0–3 (with the text tower installed for
Triton) plus the Triton restart.

#### 0. Weights — `download_pe_weights.py`

Fetches `facebook/PE-Core-L14-336` : `PE-Core-L14-336.pt` (~2.7 GB, both
towers) with `huggingface_hub`, **pinned** to repo commit
`bafb0f76541d399057e980a25947f67acec76575` and verified against SHA-256
`0cdab5b338cbaa1e7a5dcd1b2fb4c9f4d5df1abd289564658edbab64a650e7e8` (the
Hub's LFS object id). The repo is **not gated** (Apache-2.0) — no
`huggingface-cli login` needed. Both exporters call the same resolver, so
they reuse the cached file; pass `--checkpoint-path` to either exporter to
use a hand-copied checkpoint (air-gapped hosts) — it is still checksum
verified. `download_pe_weights.py --verify-file <path>` checks a file
offline. The API's PyTorch fallback loads through `perception_models`' own
`hf_hub_download` into the same cache.

#### 1–2. Image tower — `export_pe_image_encoder.py` + `build_pe_trt.sh` / `build_pe_ort_fallback.sh`

The ONNX export runs in the API container, the TensorRT build in the Triton
image (that's where `trtexec` lives):

```bash
docker compose exec yolo-api python /app/export/export_pe_image_encoder.py
ONNX_PATH=./pytorch_models/pe_image_encoder.onnx bash export/build_pe_trt.sh          # Path 1
ONNX_PATH=./pytorch_models/pe_image_encoder.onnx bash export/build_pe_ort_fallback.sh # Path 2
```

Path 2 exists because PE's attention-pooling head has, on some TensorRT
releases, used ops the ONNX parser rejects. `build_pe_trt.sh` exits `3` with
an explicit pointer to the fallback when that happens. Both paths install
into `models/pe_image_encoder/` and render the matching `config.pbtxt`
(`tensorrt_plan` vs. `onnxruntime_onnx`); each removes the other's artifact
so Triton never sees both. The committed `models/pe_image_encoder/config.pbtxt`
is the rendered default (TensorRT, max batch 32).

| Flag | Purpose |
|------|---------|
| `--checkpoint-path`, `--no-verify-checkpoint` | Use a local checkpoint / skip the SHA-256 check. |
| `--method optimum` | Export `facebook/PE-Core-L14-336-hf` via Optimum instead of `perception_models`. Needs no local PE install, but names its tensors `pixel_values`/`image_embeds` — the exporter reports the mismatch rather than shipping a model the client can't call. That HF repo **is gated**: set `HF_TOKEN`. |
| `--config-only --platform ...` | Re-render just `config.pbtxt` (what the two build scripts call). |
| `--models-dir`, `--max-batch`, `--gpus`, `--instance-count` | Target repository + Triton tuning. `--max-batch` must match the engine's profile. |
| `--skip-validate` | Skip the ONNX Runtime probe (which also catches a static leading axis — see below). |

> **Trace batch size matters.** The image exporter traces with a batch-**2**
> dummy on purpose. With batch 1, PE's attention pool bakes the batch
> dimension into a Reshape as a constant volume; Triton then loads the model
> fine and rejects every request with batch > 1. The validation step detects
> and reports this — re-export rather than editing `config.pbtxt`.

#### 3. Text tower — `export_pe_text_encoder.py`

```bash
docker compose exec yolo-api python /app/export/export_pe_text_encoder.py [--benchmark]
```

Exports `clip.encode_text(tokens, normalize=True)` — causal transformer,
final LayerNorm, EOT (argmax) pooling, projection, L2 norm — to
`pytorch_models/pe_text_encoder.onnx` (~1.4 GB, text weights only), then
gates it:

- **Contract check** (ORT CPU probe): tensor names/dtypes, 1024-d output,
  dynamic batch axis **and** dynamic token axis.
- **Parity gate**: ORT vs PyTorch eager on a fixed prompt set (short, long,
  punctuation, non-ASCII, one longer than the 32-token context) at batch 1
  and 8, both with full 32-token input and with the client's EOT-trimmed
  input; fails below cosine **0.9999** (`--parity-threshold`). Reference
  run: min cosine 1.0000000, max |diff| 4.8e-7.
- `--benchmark` times PyTorch eager vs ORT, full vs trimmed, batch 1 and 8.

Why a dynamic token axis: the text tower's mask is strictly causal and the
pooled vector is read at the EOT position, so padding after a row's EOT can
never affect it. The client drops that tail (`T = max EOT index + 1`), so a
typical 3–7 word query runs ~9 positions instead of 32 — the largest single
CPU latency win. The legacy TorchScript tracer bakes the traced length into
the `nn.MultiheadAttention` reshapes, so this exporter uses the
`torch.export`-based exporter (`dynamo=True`, opset 18) with symbolic
`Dim`s.

The API picks the text backend at startup (`OP_PE_TEXT_BACKEND`, default
`auto`): **Triton** `pe_text_encoder` if it reports ready, else a
**lazily-loaded PyTorch eager** fallback (loaded once, on the first query
a given worker actually serves — never at warm time). **ONNX Runtime**
(`CPUExecutionProvider`, also lazy) only runs when explicitly pinned
(`OP_PE_TEXT_BACKEND=onnx`); `auto` no longer considers it at all. This
matters because it's an in-process backend: with N uvicorn workers, N
independent ~1.4 GB ONNX Runtime sessions (or worse, N PyTorch eager
loads) is exactly the failure this ordering avoids — see F-07 in
`docs/design/openprocessor_internal/fresh_start_e2e_findings_2026-09-25.md`.
Pin with `onnx` / `triton` / `torch`. A Triton call that fails switches
the process to in-process encoding for good, so Triton is never a hard
dependency. `GET /health/pe_text` (`make pe-text-status`) reports the
active backend. Tokenization always stays in Python with PE's own
`SimpleTokenizer`, so `perception_models` stays a runtime dependency (for
the tokenizer only, when the PyTorch or ONNX in-process backend is
used).

Reference CPU numbers (Xeon E5-2680 v3, 8 intra-op threads on a shared,
loaded host — treat as relative; median ms per call, typical queries):

| Backend | Batch 1, full (T=32) | Batch 1, trimmed (T=9) | Batch 8, full | Batch 8, trimmed (T=9) | Warm-up | RSS |
|---|---:|---:|---:|---:|---:|---:|
| PyTorch eager (previous path: full) | 113.6 | 69.3 | 603.4 | 186.6 | 12.0 s | 6.1 GB |
| ONNX Runtime CPU (new default: trimmed) | 94.2 | 48.0 | 536.5 | 205.4 | 4.3 s | 2.8 GB |

Net effect for a single query: ~114 ms → ~48 ms, half the memory, a third
of the warm-up. At batch 8 both backends are within noise once trimmed —
the tower is weight-bandwidth bound in FP32 on this CPU.

Optional Triton serving (onnxruntime backend, CPU instances by default so it
takes no VRAM; `--kind gpu --gpus N` for GPU):

```bash
docker compose exec yolo-api python /app/export/export_pe_text_encoder.py \
    --install-triton --models-dir /app/models      # = make pe-export-text-triton
# then add --load-model=pe_text_encoder to the triton-server command
```

`models/pe_text_encoder/config.pbtxt` (committed) is the rendered default
(`--config-only` re-renders it): `text_tokens` `dims: [ -1 ]` so trimmed
batches are accepted. The API only routes to it when no local ONNX file is
configured or `OP_PE_TEXT_BACKEND=triton`.

| Flag | Purpose |
|------|---------|
| `--checkpoint-path`, `--no-verify-checkpoint` | As for the image exporter. |
| `--onnx-out` | Destination (default `/app/pytorch_models/pe_text_encoder.onnx` = the API default). |
| `--install-triton`, `--models-dir`, `--kind`, `--gpus`, `--instance-count`, `--max-batch` | Triton model-repo entry. |
| `--config-only` | Re-render only `config.pbtxt` (no torch needed). |
| `--skip-validate`, `--skip-parity`, `--parity-threshold` | QA gates. A failed gate exits 1 and never installs into Triton. |
| `--benchmark`, `--bench-threads`, `--bench-iterations` | Latency table. |

#### Sources relied on

- facebookresearch/perception_models — `core/vision_encoder/pe.py`
  (`CLIP.encode_text`, `TextTransformer` causal mask + argmax pooling),
  `config.py` (`PE-Core-L14-336`: text context 32, width 1024, output 1024;
  `fetch_pe_checkpoint` → `hf://facebook/PE-Core-L14-336:PE-Core-L14-336.pt`),
  `tokenizer.py` (`SimpleTokenizer`), commit `3e352cc`:
  <https://github.com/facebookresearch/perception_models>
- Hugging Face model card + API for `facebook/PE-Core-L14-336` (license,
  gating, commit, LFS SHA-256): <https://huggingface.co/facebook/PE-Core-L14-336>
- PyTorch `torch.onnx.export(..., dynamo=True, dynamic_shapes=...)`:
  <https://docs.pytorch.org/docs/stable/onnx_export.html>
- ONNX Runtime Python API / threading (`SessionOptions.intra_op_num_threads`,
  `CPUExecutionProvider`): <https://onnxruntime.ai/docs/performance/tune-performance/threading.html>
- Triton ONNX Runtime backend (`platform: "onnxruntime_onnx"`, variable dims
  `-1`, `KIND_CPU` instances): <https://github.com/triton-inference-server/onnxruntime_backend>
  and model configuration: <https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html>
- TensorRT `trtexec` optimization profiles (image Path 1):
  <https://docs.nvidia.com/deeplearning/tensorrt/latest/reference/command-line-programs.html>

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

### Dual-Head Detector
- Input: `[B, 3, S, S]` FP32, letterboxed, normalized [0, 1] (`S` = `--imgsz`)
- Output: `output0` (detection tensor) + `sppf_feat` `[B, C, S/32, S/32]`
- Dynamic batching: 1-`--max-batch`

### SCRFD-10G Face Detection
- Input: `[B, 3, 640, 640]` FP32, RGB, (x-127.5)/128.0 normalized
- Output: 9 tensors (3 FPN strides x score/bbox/kps), CPU post-processed
- Dynamic batching: 1-32

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

### PE-Core-L14-336 Image Encoder
- Input: `images` `[B, 3, 336, 336]` FP32, ImageNet mean/std normalized
  (resize shorter edge to 336, center-crop — see
  `src/services/detection/pe_preprocess.py`)
- Output: `image_embeddings` `[B, 1024]` FP32, L2-normalized
- Dynamic batching: 1-32 (engine profile min=1 / opt=8 / max=32)

### PE-Core-L14-336 Text Encoder
- Input: `text_tokens` `[B, T]` INT64, `T <= 32` — PE `SimpleTokenizer` ids,
  trimmed after the batch's last EOT by the client
- Output: `text_embeddings` `[B, 1024]` FP32, L2-normalized
- Served in-process by ONNX Runtime (default) or Triton `onnxruntime_onnx`
  (optional, max batch 32)

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
