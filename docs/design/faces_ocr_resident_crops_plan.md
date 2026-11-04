# uint8 OCR input, width-bucketed recognition, face alignment on resident frames (#220)

Status: open. Step 1 (uint8 OCR inputs) is done (`27ebeb45`); step 2 was measured and rejected (#220 comment, 2026-11-02); steps 3 and 4 are `v070_rebuild/06c_b5_crops_ocr_faces.md` and `06d_b5_processing_plans.md` (v0.7.0 B5c and B5d; remove after B5c). Line numbers drift:
re-find by symbol. Read first: `docs/design/triton_pipeline_optimization_plan.md` (Wave 3 item 3,
Wave 5, section 7 parity gates), `docs/OCR.md`.

## 0. Split decision

#220 is three unrelated changes with different risk. **Ship it in steps; this plan makes Step 1
fully executable, Step 2 executable, and records what stays open.**

| Step | Content | Engine rebuild | Parity risk | Ships as |
|---|---|---|---|---|
| 1 | uint8 `ocr_images` and `original_image` into the `ocr_pipeline` BLS, both callers, datatype read from Triton metadata (#428 pattern), `MODEL_VERSION` so installs pick it up | no | none: proven bit-identical (the BLS expression on the uint8 tensor equals the old client-side float expression, tested in `tests/test_ocr_uint8_inputs.py`) | PR 1 |
| 2 | recognition batched by width bucket inside the BLS | no (engine already takes batch 64, width 48-2048) | low-medium (padding changes CTC input) | PR 2, gated on P-ocr |
| 3 | recognition argmax in the graph | yes (re-export `paddleocr_rec_trt`) | medium | stays open on #220 |
| 4 | face alignment on resident GPU frames | yes / new BLS work | blocked (C4) | stays open; move under #67 P5 |

## 1. Corrections to the issue premises (verified)

| # | Premise | Reality | Evidence |
|---|---|---|---|
| C1 | "OCR receives a full-resolution FP32 crop" | Two FP32 inputs: `ocr_images` (detector input, long side 640-960, padded to 32, `x/127.5-1`) and `original_image` (full resolution, `x/255`). Only the second is full resolution. Both go to uint8. | `models/ocr_pipeline/config.pbtxt:30-46`; `src/clients/triton_ocr.py:55-96`; `src/services/detection/cascade_detect/ocr_recognizer.py:398-445` |
| C2 | Model input change is like #429's uint8 prefix | No ONNX prefix applies: `ocr_pipeline` is a **Python BLS**; it already converts `original_image` back to uint8 (`(original_image.transpose(1,2,0)*255).astype(np.uint8)`) and normalizes `ocr_images` before the det engine. The change is in `model.py` plus the two clients. `export/uint8_input.py` is not used. | `models/ocr_pipeline/1/model.py:142-174,195-202` |
| C3 | "recognition is not batched" | True in the BLS (one `exec()` per crop, `model.py:218-292`), but the engine is **not** batch-1: `max_batch_size: 64`, profile batch 1/32/64, width 48-2048, plus Triton dynamic batching. The BLS docstring "TensorRT engine supports batch=1 only" and "4 model instances" are stale (config count is 2). | `models/paddleocr_rec_trt/config.pbtxt` (max_batch_size 64, dynamic_batching), `export/export_paddleocr_rec.py:39-49,217-219`, `models/ocr_pipeline/config.pbtxt` instance count 2 |
| C4 | "face alignment on resident GPU frames"; "#217 shipped, so target that transport" | The only GPU-resident frame is inside `gpu_image_prep` during **curation ingest**, which never runs faces or OCR. `/faces/*`, `/analyze` and core `/ingest` decode on the CPU (`fast_face_client.py:155`) and align on the host (`fast_face_client.py:203`, `src/utils/face_align.py:91-145`). There is no resident frame on any face route to align from. | `docs/design/gpu_decode_plan.md:108-110`, `models/gpu_image_prep/config.pbtxt` inputs |
| C5 | Upgrade safety | `ocr_pipeline` has **no `MODEL_VERSION` file**, so `export/sync_model_seed.py` never replaces an installed `model.py`/`config.pbtxt`; an upgraded API would talk to the old FP32 BLS. Only `models/gpu_image_prep/MODEL_VERSION` exists. | `export/sync_model_seed.py:1-49`, `git ls-files models \| rg MODEL_VERSION` |
| C6 | #428 helper reuse | `AsyncTritonPool.get_model_input_datatype` and `TritonClient._read_model_info` read only the input named `images`; they need an input-name parameter for OCR. `images_input` is detector-specific (FP32 path divides by 255) and is not reused for OCR. | `src/clients/triton_pool_core.py:58-80,382-400`; `src/clients/triton_client.py:103-136` |

## 5. What stays open on #220 after PR 1 and PR 2

- Step 3: argmax in the recognition graph (re-export with an ArgMax/ReduceMax tail, new version dir,
  CTC on indices; Wave 5 tests). Output shrinks from `[-1, 18385]` FP32 per step.
- Step 4: face alignment on resident frames. Needs a face route that holds a GPU frame, i.e. faces
  as a `gpu_image_prep` branch (#67 phase P5). Recommendation: move this item to #67/P5 and keep
  #220's acceptance "ArcFace cosine p01 >= 0.995, P-face 1.0" there.
- Acceptance line "OCR/face throughput recorded" is met for OCR by PR 1/2; faces stays open.

## 6. Judgment calls (recommended option first; pre-approved)

| Q | Recommendation |
|---|---|
| J1 Keep an FP32 input for one release (Wave 3 item 3 said so)? | No. No shims; `MODEL_VERSION` replaces the BLS with the API; clients fail closed with a repair hint. |
| J2 CHW vs HWC uint8 | Keep CHW in PR 1 (dtype only); HWC saves one host transpose and can follow later. |
| J3 Read datatype from metadata at all, if only UINT8 is accepted? | Yes: one cached call turns an opaque Triton shape/dtype 400 into an actionable error, same pattern as #428. |
| J4 Cap `original_image` long side (Wave 3 suggested 2x det size) | Not in #220: it changes crops and thus text (a parity change); belongs to #219. |
| J5 Step 4 location | Move to #67 P5 (C4). |
