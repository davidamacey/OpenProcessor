# B5c: work crops at model size (#219) and OCR on the GPU (#220, GPU-7, GPU-19)

| Field | Value |
|---|---|
| Status | plan, not started |
| Release | v0.7.0 (inside the release; R-final runs after B5; see `06_b5_overview.md` section 2) |
| Depends on | 5b merged (`06b_b5_decode_once.md`); `ref-b5a` with OCR probes and (for P-vlm) VLM labels; a GPU plan for the VLM and the segmenter during the gate |
| Blocks | 5d P5 (faces and OCR as `gpu_image_prep` branches) uses the OCR contract this batch leaves |
| Decisions applied | #40 Q2/Q3 (2026-10-03: work-crop cap recommended 1024, shipped off first; the VLM cap set from the VLM's real input size); `faces_ocr_resident_crops_plan.md` J1-J5 (pre-approved; J5 moves face alignment to #67 P5); public data only |
| Open decisions | none new |
| Source IDs | former master plan (rev 4, in git history) 7.5 row 5c, 9.1 (N14: GPU-19), 9.2 (#220), 11 (#17/#18 leftover: #220 step 3), 14 (the #220 step-2 evidence); #219, #220; GPU-7, GPU-19 (`codebase_audit_plan.md` 4.3); `triton_pipeline_optimization_plan.md` Wave 3 and 5, section 7 |
| Verified | Verified against `fix/411-r2-round` at `af62c947` on 2026-11-03 (read-only `rg`, `gh issue view`; nothing run) |
| Consolidated | Consolidated 2026-11-03 |

## 1. Goal

Crops are carried at the size their consumer needs, not at source resolution, and OCR recognition
gets faster without changing its text. Every change passes the task gates (P-vlm, P-sam,
P-region, P-ocr) against `ref-b5a` and beats 2x the spread.

## 2. What is done, what remains

| Item | State | Evidence |
|---|---|---|
| #219 `crop_max_side` | not started; `rg crop_max_side src` finds nothing | #219 triage 2026-10-31 |
| #220 step 1: uint8 `ocr_images` and `original_image` into `ocr_pipeline` | **done** (`27ebeb45`; `models/ocr_pipeline/config.pbtxt` inputs `TYPE_UINT8`; `MODEL_VERSION` present; `tests/test_ocr_uint8_inputs.py`) | `faces_ocr_resident_crops_plan.md` 0 |
| #220 step 2: width-bucketed recognition | **measured, rejected**: buckets with zero padding P-ocr 86.0 % (char edit rate 1.94 %), BLS 43.2 ms vs 48.0 ms; exact-width grouping P-ocr 100 % but 49.5 ms (no gain) | #220 comment 2026-11-02; the #220 comment of 2026-11-02 |
| GPU-7: one blocking `exec()` per crop in `models/ocr_pipeline/1/model.py::_call_recognition` (up to `MAX_TEXTS = 128`) | open | audit 4.3 |
| #220 step 3: argmax in the recognition graph | open (research leftover #17/#18) | `faces_ocr_resident_crops_plan.md` 5 |
| #220 step 4: face alignment on resident frames | **moved to 5d** (#67 P5, J5) | |
| GPU-19: OCR detector normalisation (`x/127.5-1`, BGR to RGB) versus upstream ImageNet mean/std | evidence task | audit 4.3 |

## 3. Branch, worktrees, tracks

Integration branch `int/b5c-crops-ocr-faces` from main after 5b merged. Tracks `b5c/<track>` in
`<repo>-wt-b5c-<track>`:

| Order | Track | Scope | Size |
|---|---|---|---|
| 1 | `crops` | 5c-K1..K3 #219 | L |
| 2 | `ocr` | 5c-O1 GPU-7, 5c-O2 padding-aware batched decode, 5c-O3 step 3 research | M-L |
| 3 | `ocrnorm` | 5c-N1 GPU-19 evidence | S |
| - | gate run | one GPU window for P-vlm, P-sam, P-region and P-ocr together (and 5b-C1(a) if deferred) | GPU day |

## 4. Measurement protocol and gates for this batch

- `06_b5_overview.md` M-1..M-10. Set B is the gate set for #219 (the cap never triggers on COCO
  crops, which are near model size already; the public COCO gate is the unit tests plus byte
  accounting). Set A and the pinned OCR set for P-ocr.
- **Pinned OCR set:** the public text-bearing set used for B0's OCR before/after
  (`b0_close_out_done.md`; COCO-Text v2 rule with its fallback), so P-ocr is comparable across
  batches. Re-check its download location and licence at fetch time (the former master plan, rev 4, section 15 notes it was
  not re-checked); record both in the manifest.
- Baselines: 5a endpoint rows (`/ocr` at 1 and 16 clients; the OCR BLS ms per request from Triton
  `compute_infer` of `ocr_pipeline`), 5a set B ingest rows, crop-cache bytes per image from
  `run_baseline.py --crop-cache-dir`.
- Gates (thresholds in `06_b5_overview.md` section 6): P-vlm >= 97 % label agreement with the
  disagreement list reviewed by a human; P-sam hit/miss >= 97 %, IoU mean >= 0.97, p05 >= 0.90;
  P-region hit/miss >= 99 %, IoU >= 0.9; P-ocr >= 98 % exact, CER not worse; P-pe unchanged (PE
  reads pixels, not the cached JPEG).
- GPUs for the gate: Triton and API on GPU 0; VLM (`VLM_GPU_ID`) and segmenter (`SEGMENTER_GPU_ID`)
  on GPU 2 only with the peer sessions' agreement, else sequentially on GPU 0. The segmenter needs
  the HF token already in `.env`: pass the env file to compose, never read or print it.

## 5. Tasks

### Track `crops` (#219)

**5c-K1 `crop_sizing`: one function for the work crop.** Size M.
- New `src/services/detection/crop_sizing.py`: ROI -> work crop, resized with OpenCV
  `INTER_AREA` (antialiased) so the long side is at most `crop_max_side`, never upscaled, aspect
  kept; JPEG encode from the small array.
- Callers: `src/services/curation/ingest_index.py::crop_pil`, `source_image_cache.write_crop_cache`,
  `src/services/curation/crop_bytes.py` (`crop_jpeg_from_disk`, `load_item_crop_jpeg`,
  `load_vlm_item_jpeg`) so a cache hit and a miss give the same size and bytes; the GPU route's crop
  JPEGs in `models/gpu_image_prep/1/model.py` (the same cap, applied before encode).
- Settings: `crop_max_side` (0 = today's behaviour, the default until the gate passes), env name
  chosen at implementation, `env.template` and `tests/test_env_surface.py` in the same commit.
- Red: long side <= cap; never upscaled; aspect kept; hit and miss give identical dimensions; with
  cap 0 the bytes equal today's (golden); a synthetic 4000x3000 crop's payload to the segmenter and
  VLM is smaller (byte-count assertion); `crop_norm_to_source_norm` round trip unchanged. Mutation
  checks: ignore the cap, upscale, drop aspect.

**5c-K2 VLM cap from the VLM's real input size.** Size S.
- Read the active VLM's processor input size (the catalog `examples/vlm/catalog.tsv` row and the
  model's processor config), not a guess (#40 Q3). Apply `vlm_crop_max_side` where the VLM payload
  is built (`src/services/labeling/vlm_labeler_combined.py`, `crop_bytes.load_vlm_item_jpeg`).
- Red: the payload long side equals the cap for a large crop; cap 0 is byte-identical.

**5c-K3 Gate run and default.** With the caps on (1024 and the VLM size): set B crop-cache bytes per
image at least **50 % lower** than cap off (#219 acceptance; Wave 3 aims at 5x for crops above the
cap); bytes to the segmenter and VLM per large crop lower; cache hit rate not lower; worker crops/s
not lower; P-vlm, P-sam, P-region, P-ocr pass. Then the default becomes 1024 in its own commit.
With the cap off, outputs byte-identical to the 5a reference.

**5c-K4 Reduced decode once 5c-K3 sets the cap (measurement first, M).**
- Origin: `docs/research/v070_measurement_log.md` ("measure a second decode route").
- Idea: with work crops capped at `crop_max_side`, a DCT-scaled decode (scale chosen so the largest
  work crop still has at least the cap in pixels) can feed both the detector input and the crops of
  a large photo, so one smaller decode replaces the full-frame decode.
- Libraries (check before writing code): OpenCV `IMREAD_REDUCED_COLOR_2/4/8` (imread flags docs),
  Pillow `Image.draft` (Pillow docs, JPEG draft mode); on the GPU route read the nvImageCodec
  decode parameters (https://docs.nvidia.com/cuda/nvimagecodec/) for a scaled decode before
  writing anything.
- Files: `src/utils/image_decode.py` (the 5b-D1 helper) behind a setting, default off
  (`env.template`, `tests/test_env_surface.py`); tests in `tests/`.
- Measure: decode ms per image on set B; crop pixels per work crop. Gate: P-box (boxes within 1 px
  at the original resolution), P-vlm, P-sam, P-region, P-pe against `ref-b5a`. Adopt only with a
  gain beyond 2x the spread; otherwise record the numbers and close (delete the setting).
- Red: a fixture 12 MP JPEG decoded reduced yields a work crop whose long side is at least the
  cap, and unscaled boxes map back within 1 px (fails until the path exists); control: setting off
  is byte-identical. Done when: adopted or recorded in `docs/PERFORMANCE.md`.

### Track `ocr` (#220, GPU-7)

**5c-O1 GPU-7: concurrent recognition calls with `async_exec`.** Size M.
- Pattern: Triton python backend BLS `inference_request.async_exec` "allows ... multiple
  inflight inference requests"; it needs `async def execute`; "Async BLS is not supported when
  running a Python model in decoupled mode" (https://github.com/triton-inference-server/python_backend).
  `ocr_pipeline` is **not** decoupled (`max_batch_size: 0`, no `model_transaction_policy`), so it
  qualifies.
- Fact that bounds the gain: the dynamic batcher merges only requests whose inputs have the same
  shape (ragged batching is off; Triton ragged batching docs), so crops of different widths are
  **not** batched together. `async_exec` gives in-flight overlap; `paddleocr_rec_trt` has
  `instance_group count: 1`, so test `count: 2` as a second factor.
- Steps: `async def execute`; gather every crop's `async_exec` (one gather per request); keep each
  crop's exact width (no padding) and the per-crop `-1.0` failure sentinel.
- Red (audit): for 40 crops, today 40 sequential `exec` calls; after, one gather of 40 concurrent
  calls (stand-in `pb_utils` records concurrency). Control: a single crop gives byte-identical text
  and scores.
- Pass: P-ocr 100 % identical text (no padding, so the math is unchanged); OCR BLS ms per request
  lower than the 5a row by > 2x spread; else record and do not merge.

**5c-O2 Padding-aware batched decode (only if 5c-O1 leaves recognition binding).** Size M.
- The rejected bucket run padded crops and decoded the padded logits (TOYOTA -> TOYOTR). Try:
  pad to a bucket width, run the batch, then cut each crop's logits to the time steps its true
  width produces before CTC decode (compute the step count from the recognizer's horizontal stride,
  read from the model, not assumed).
- Gate: P-ocr >= 98 % exact and CER not worse on the pinned OCR set and set A probes. SVTR-style
  recognisers mix context globally, so padding may change valid steps too; if P-ocr fails, record
  and stop (the #220 comment of 2026-11-02 plus this run decide).

**5c-O3 Step 3 research: argmax in the recognition graph.** Size M (research; code only on a pass).
- Re-export `paddleocr_rec_trt` with an ArgMax/ReduceMax tail in a new version directory selected
  by `version_policy`; CTC on indices; output shrinks from `[-1, 18385]` FP32 per step to indices
  and max scores.
- Red/offline: CTC on indices equals CTC on the softmax argmax for fixtures; ONNX-level parity.
- Gate: P-ocr >= 98 % (expected identical: argmax is exact), OCR BLS ms per request and wire bytes
  lower by > 2x spread; Triton GPU memory not higher.
- Upgrade: new engine version -> the OCR group's stamp must rebuild (check `model_stamps.sh`
  flags for the ocr group); CHANGELOG.

### Track `ocrnorm` (GPU-19, evidence first)

**5c-N1 Detector normalisation evidence.** No code unless a gap is measured.
- 50 public images of the pinned OCR set through (a) the upstream PP-OCRv5 reference
  preprocessing (ImageNet mean/std) in onnxruntime with the same detector ONNX, (b) the BLS's
  `x/127.5-1` with the BGR to RGB flip, against the set's ground truth: box recall and P-ocr.
- Act only on a measured gap beyond the noise floor; then the fix is a BLS change with a
  `MODEL_VERSION` bump and its own P-ocr gate. Record the result either way (closes N14's GPU-19
  checkbox).

## 6. Gate (5c PR)

- T0 (targeted red/green plus static checks on the touched files) per task; one offline sweep on
  the merged integration head (pre-commit `--all-files`, suite `-n 8`, contracts-check, docs check,
  CHANGELOG rules; frontend only if `frontend/` changed).
- The gate runs inside the B5 GPU window (`decision_register.md` section 8.1), after 5b's: P-vlm, P-sam,
  P-region and P-ocr together; one confirmation row on set A
  and set B against 5a.
- #219 acceptance (public set B), #220 acceptance as re-scoped (OCR bytes already 4x down by step 1;
  P-ocr; OCR throughput before and after recorded; faces part moved to 5d).

## 7. Done when

- caps exist, default 1024 only after the gate; cap off byte-identical;
- GPU-7 done (one async gather and identical text) or recorded as no gain;
- 5c-O2 and 5c-O3 adopted or recorded; GPU-19 recorded;
- cite the #220 step-2 numbers (the #220 comment of 2026-11-02) in `docs/research/`; the evidence
  branch was deleted on 2026-11-04 with the owner's OK;
- `docs/PERFORMANCE.md`, `docs/OCR.md` updated; CI green; PR merged with a merge commit; #219 and
  #220 closed (faces part points at #67);
- 5c-K4 adopted or recorded;
- the R-3 docs removal for 5c is done in the closing PR (`faces_ocr_resident_crops_plan.md`; table
  in `06_b5_overview.md` section 10.1), and this file is deleted after the keep-list below is
  moved (`decision_register.md` "Plan lifecycle"), with its row removed from `decision_register.md` and
  `docs/design/README.md`.

### What to keep after merge

| Fact | Target |
|---|---|
| `crop_sizing` contract (cap, never upscale, `INTER_AREA`), VLM cap rule, cache-hit equality | `docs/ARCHITECTURE.md`, `docs/CURATION.md` (crop cache size) |
| OCR BLS design (`async_exec`, no padding, rejected buckets with the P-ocr 86 % evidence, step 3 outcome) | `docs/OCR.md`, numbers in `docs/PERFORMANCE.md`, method in `docs/research/` |
| GPU-19 evidence | `docs/research/` note |
| new settings and behaviour changes | `env.template`, docs-site config pages, CHANGELOG |

## 8. Docs, CHANGELOG, upgrade impact, rollback

- Docs: `docs/PERFORMANCE.md`, `docs/OCR.md`, `docs/CURATION.md` (crop cache size), docs-site
  configuration pages for the caps, `env.template`.
- CHANGELOG: "Added" caps; "Changed" when the default turns on (crop bytes smaller); OCR engine
  version if step 3 lands.
- Upgrade: `ocr_pipeline` `MODEL_VERSION` bump for BLS changes (the sync replaces the installed
  model); a recognition engine version rebuild via the stamp if step 3 lands; existing crop cache
  entries stay valid (a miss re-cuts at the cap; mixed sizes are fine because consumers take JPEG
  bytes).
- Rollback: caps to 0; revert the BLS commit (`MODEL_VERSION` back); `version_policy` back to the
  old recognition version.

## 9. Risks

| Risk | Mitigation |
|---|---|
| smaller VLM crops change labels | P-vlm >= 97 % with a human-reviewed disagreement list; cap from the VLM's real input |
| text reading needs native resolution | OCR of a located region cuts from the source at native resolution (Wave 3 item 3); P-ocr |
| async BLS behaviour differs in the pinned Triton image's python backend | read that release's python_backend README before coding; the red test runs against the stand-ins, plus one live check on the pinned image |
| GPU contention during the gate (VLM, segmenter, Triton) | one gate window, peers informed, GPU 2 only with agreement |

## 10. Lane briefs (ready to paste)

Common block: as in `04_b3_ultralytics_one_version.md` section 17, with worktree
`<repo>-wt-b5c-<track>`, branch `b5c/<track>`, file `docs/design/v070_rebuild/06c_b5_crops_ocr_faces.md`,
plus: "Parity and benchmarks per `06_b5_overview.md` against `ref-b5a`. Stack `docker compose -p
op-b5c-bench`. GPU 0; GPU 2 only if the coordinator says the peer agreed. Never read or print
`.env`; pass it to compose. Project slugs start with `b5c-`."

- **`crops`**: "Do 5c-K1, 5c-K2 (red tests and mutation checks first), with the caps off by
  default. 5c-K3's gate run is scheduled by the coordinator; flip the default only after it passes."
- **`ocr`**: "Do 5c-O1 (red call-count test first). Then 5c-O2 only if the coordinator says
  recognition still binds; then 5c-O3 as research with an offline parity test first."
- **`ocrnorm`**: "Do 5c-N1 as evidence only; propose a fix only if the gap is measured."

## Research amendments (2026-11-04, the research integration pass)

| Where | Change |
|---|---|
| 5c-K1 | `crop_sizing` uses the storage seam's size-variant crop key (GPU S-10, lane 1E-4), so a source-size crop is never served as a capped one |
| 5c-K4 | reduced decode on the GPU route is impossible on the A6000 (nvJPEG scaled decode is hardware-backend only); the task is the CPU-route measurement only |
| 5c-O2, 5c-O3 | read `gpu:PR-6`'s record first (argmax in the recognition graph, padding-aware CTC on bucketed widths) |
| track `ocr`, model choice (owner Q-B, 2026-11-04; review RR-5) | PP-OCRv6 (Apache-2.0, released June 2026) is compared with PP-OCRv5 once, in the short `pt:P1` run of PW-2 ([00f](08_research_prototype_track.md) section 3), on the pinned OCR set's lines as well as the plate crops. PP-OCRv5 stays the model unless that run shows a clear win on both at no more than today's latency; on a win, the swap is one task of this track (export, `MODEL_VERSION` bump, the P-ocr gate, `ATTRIBUTION.md`), done before `rec_char_min` so the new output is added once |
| track `ocr` | **research plate 5.7: a new output tensor `rec_char_min`** (FP32, `[128]`) in `ocr_pipeline`: the minimum per-character max probability over the decoded characters of each line, computed in `CTCLabelDecode` beside the mean; padded slots 0.0; threaded through `src/clients/triton_ocr.py` and the cascade OCR recognizer (`OcrLine.char_min`); `MODEL_VERSION` bumped; one more output in the decode loop 5c-O2 already rewrites. Red: `tests/ocr/test_rec_char_min.py`: the output exists and equals the per-line minimum on a fixture. Control (parity): `rec_scores` and `texts` byte-identical on the fixed fixture. Its first reader is the W5e confident tier ([06e](06e_w5e_plate_text.md)) |
