# B5b: decode once (GPU-3b, the rest of #40, #407, GPU-12)

| Field | Value |
|---|---|
| Status | plan, not started |
| Release | v0.7.0 (inside the release; R-final runs after B5; see `06_b5_overview.md` section 2) |
| Depends on | 5a merged (`06a_b5_baseline.md`): `ref-b5a` captures, the 5a-R0 rows, D-35 decided |
| Blocks | 5c and 5d build on the one decode helper and the GPU route this batch leaves |
| Decisions applied | #40 owner decisions (2026-10-03: baseline first; every wave behind a rollback switch; compiled code only if the profile says Python dominates after the data-volume work); `gpu_decode_plan.md` D1-D3 as decided (13); public data only |
| Open decisions | none new; D-35's outcome may add the `/detect` `scaleup` switch to 5b-D1 |
| Source IDs | former master plan (rev 4, in git history) 7.5 row 5b, 9.2 (#40, #407), 10; GPU-3b, GPU-12 (`codebase_audit_plan.md` 4.3); #40, #407; `triton_pipeline_optimization_plan.md` Waves 6-8; `gpu_decode_plan.md` 13; `sp_gpu_prep_concurrency.md` C2-C4 |
| Verified | Verified against `fix/411-r2-round` at `af62c947` on 2026-11-03 (read-only `rg`, `gh issue view`; nothing run) |
| Consolidated | Consolidated 2026-11-03 |

## 1. Goal

Every image is decoded once, in one orientation, and every model input comes from that decode.
Then the GPU image-prep path gets faster where a profile says the time goes. Outputs stay
byte-identical where the math is unchanged; numerics changes pass the 1 px and cosine gates.

## 2. What is done, what remains

| Item | State | Evidence |
|---|---|---|
| CPU single decode on ingest (whole frame cut from the detector decode) | done | `gpu_decode_plan.md` 13 |
| Triton GPU decode `gpu_image_prep` (nvJPEG GPU-Huffman via ctypes, fused CuPy kernels, decoupled, size gate 3 MP) | done, opt-in | `gpu_decode_plan.md` 13; photos 27.0 -> 35.0 img/s |
| memory budget, stage counters, 2 instances on the ingest profile | done (0.7.0) | PERFORMANCE "GPU memory and the ingest load profile" |
| **GPU-3b**: `/embed*`, `/analyze`, legacy ingest decode with PIL and no `exif_transpose`; faces use the oriented `cv2.IMREAD_COLOR`, so `/analyze` can mix frames | open | audit 4.3 |
| **#407**: C2 batched kernels, DALI front end, native backend, in that order, profile first | not started; C2's launch-overhead premise unsupported | #407 comment 2026-10-31 |
| **GPU-12**: letterbox undo subtracts a half-pixel pad the paste never applied | open | audit 4.3 |
| **#40 Wave 6**: GPU decode for the stateless routes (`/detect`, `/embed/image`) | not started | `triton_pipeline_optimization_plan.md` Wave 6 |
| GPU encode of crop JPEGs; GPU decode of progressive JPEGs (11 % of set B goes to the CPU) | follow-ups | `gpu_decode_plan.md` 13 "Not reached" |
| faces and OCR on the resident frame | moved to 5d (#67 P5) | `processing_plans_plan.md` 10 |
| work crops at model size | 5c (#219) | |
| Wave 8 region stage on resident crops (option a) and the segmenter on TensorRT (option b) | not in B5; option b needs its own plan file first (owner decision Q8, 2026-10-03) | Wave 8 |

## 3. Branch, worktrees, tracks

Integration branch `int/b5b-decode-once` from main after 5a merged. Tracks `b5b/<track>` in
`<repo>-wt-b5b-<track>`:

| Order | Track | Scope | Size |
|---|---|---|---|
| 1 | `decode` | 5b-D1 GPU-3b (and the D-35 switch if decided) | M |
| 2 | `profile` | 5b-P1 profile of `gpu_image_prep` (runs, no code) | S (GPU hours) |
| 3 | `prep` | 5b-G1 GPU-12; 5b-G2 #407 stage 1 only if 5b-P1 supports it; 5b-G3 stage 2 only if stage 1 leaves the decode/letterbox stage binding; stage 3 only on the rule | L |
| 4 | `routes` | 5b-W1 Wave 6 bake-off for the stateless routes | M |
| 5 | `codec` | 5b-C1 crop JPEG encode and progressive decode evaluation | M |

`decode` merges first (every later track uses the helper). `prep`, `routes` and `codec` are
independent in files; `prep` waits for `profile`'s report.

## 4. Measurement protocol for this batch

- `06_b5_overview.md` M-1..M-10 for every number.
- Baseline: the 5a-R0 rows (A-upload, A-batch, B-4, B-8, endpoints, perf_analyzer) and the 5a set B
  ingest-profile row with 2 instances. Spread: the 5a spread of the same row, or the candidate's,
  whichever is larger.
- Pass to adopt: median gain > 2x spread at the target concurrency (B-4 and B-8 for `prep`;
  32 clients for `routes`), p99 not worse than +10 %, zero CPU-fallback lines
  (`op_ingest_decode_total` routes `cpu_decoder`, `cpu_call_failed`, `cpu_gpu_memory` at most
  0.5 % of eligible photos), peak card memory within the profile budget.
- Parity (against `ref-b5a`): byte-identical where math is unchanged (`gpu_prep_parity.py compare`);
  otherwise every matched box within 1 px at the original resolution, P-box >= 0.98, P-item >= 0.99,
  PE crop and frame cosine median >= 0.999 and p01 >= 0.995, P-knn >= 0.95.
- Small-card checks (owner #407 acceptance "24 GB and 12 GB cases still work"): the ballast
  emulation in `sp_gpu_prep_concurrency.md` section 7 (24 GiB and 12 GiB rows): loads or refuses at
  load, never fails at runtime.

## 5. Tasks

### Track `decode`

**5b-D1 GPU-3b: one oriented decode helper.** Size M.
- Pattern: move `src/services/curation/image_decode.decode_image` (proven equal to Pillow
  `ImageOps.exif_transpose` for all 8 EXIF tags; returns `DecodedImage`, raises `ImageDecodeError`)
  to `src/utils/image_decode.py`; update its 6 importers (`rg -n "image_decode import" src scripts tests`);
  no re-export shim at the old path.
- Callers to adapt (PIL `Image.open(...).convert('RGB')` without orientation today):
  `src/services/analyze_pipeline.py`, `src/clients/triton_client.py::infer_yolo_clip_cpu`,
  `src/clients/triton_preprocess.py::preprocess_for_mobileclip`, `src/services/embedding.py`,
  `src/routers/embed.py`; and decide `src/utils/image_processing.decode_image` (cv2
  `IMREAD_COLOR`, oriented) and the RGBA path beside it: either route both through the helper or
  document why they stay (they already orient).
- `src/utils/affine.py::get_jpeg_dimensions_fast`: return the oriented size, or document that it
  does not and make every caller that un-maps boxes use the oriented size.
- If D-35 chose `scaleup=True`: change `LetterBox(..., scaleup=False)` in
  `src/clients/triton_client.py` in its own commit with a CHANGELOG behaviour-change line.
- Check before relying on either behaviour: DALI's image decoder now applies EXIF orientation by
  default (`adjust_orientation=True`), contrary to the older note that DALI ignores EXIF. On the
  pinned DALI version, decode the eight EXIF-tag fixtures and compare against
  `ImageOps.exif_transpose` before the GPU path assumes the decoder orients, or sets
  `adjust_orientation` explicitly. Source: `docs/design/v070_rebuild/09g_research_gpu_pipeline.md` risk table.
- Red (audit): for EXIF tags 1-8, `/embed` preprocessing, `/analyze` objects and `/analyze` faces
  all agree on (w, h); red today for tags 5-8. Control: tag 1 output byte-identical to before.
- Gate: P-box, P-clip and P-face on set A must stay identical for tag-1 images (COCO is mostly
  tag 1); the rotated cases are a deliberate behaviour change listed in CHANGELOG.
- Done when: one decode helper serves these paths; `rg -n "Image.open\(.*\)\.convert\('RGB'\)" src`
  lists only documented exceptions.

**5b-D2 CPU letterbox on OpenCV (M).** After 5b-D1; track `decode`.
- Origin: `triton_pipeline_optimization_plan.md` Wave 2 (`rg ingest_decode src` finds nothing, so it
  never shipped).
- Files: `src/services/detection/geometry.py` (`letterbox_to_square`), `env.template` and
  `tests/test_env_surface.py` (the setting), tests in `tests/`.
- Pattern: an ndarray path in `letterbox_to_square` with `cv2.resize`; `INTER_AREA` for downscales
  (OpenCV docs, "Image resizing": `INTER_AREA` is preferred for shrinking; PIL bilinear
  antialiases, `INTER_LINEAR` does not), and the 640 letterbox from the 1280 intermediate. Keep
  the documented PIL-versus-cv2 split between ingest and `/detect` unless the gate says otherwise.
  A setting, default off, until the gate passes.
- Red: a test that the cv2 path produces boxes within 1 px of the PIL path on a fixed fixture set
  (fails until the path exists); control: the default (setting off) output is byte-identical to
  before.
- Measure: API CPU-seconds per image on the CPU route, set A and set B below the 3 MP GPU gate.
  Gate: P-box against `ref-b5a`; adopt only with a gain beyond 2x the spread.
- Done when: adopted (default on) with numbers in `docs/PERFORMANCE.md`, or recorded and the setting
  deleted (no dead code).

### Track `profile`

**5b-P1 Profile before any #407 code.** No code.
- Tools: Nsight Systems on the Triton container's `gpu_image_prep` stub (CUDA API trace: launches,
  `cudaStreamSynchronize`, `cudaMalloc`, memcpy kinds) and py-spy `record` on the stub PID
  (Python versus native time, GIL waits), during set B at 4 and 8 batches on the ingest profile with
  2 instances.
- Report: per image, GPU time per kernel phase, sync count, allocation count, host-to-device and
  device-to-host copies, Python self time per stage of `model.py` (`_detect`, `_embed`,
  `_crop_pixels`, `_blur_sums`, JPEG encode), GIL wait share; and which stage binds
  (`op_gpu_prep_stage_seconds`).
- Decision rule (from #407 and the 2026-10-31 triage): start with the stage the profile points to.
  Launch/sync overhead dominant -> 5b-G2. Decode/letterbox dominant -> 5b-G3. Python/GIL dominant
  after G2 -> stage 3 evaluation. Crop JPEG encode dominant -> 5b-C1.

### Track `prep` (#407, GPU-12)

**5b-G1 GPU-12: consistent integer pad.** Size S.
- Files: `src/services/detection/geometry.py` (`canvas.paste(resized, (int(pad_w), int(pad_h)))`),
  `models/gpu_image_prep/1/prep_geometry.py` (returns `int(pad_w), int(pad_h)` and the float pads),
  the client un-map in `src/services/curation/ingest_gpu_decode.py` and `src/utils/affine.py`
  (`(boxes - pad_x) / scale`), `tests/models/test_gpu_image_prep_geometry.py`.
- Fix: use `float(int(pad))` in every undo so it matches the paste.
- Red: round-trip error on an odd margin is 0.5 px today. Control: an even margin unchanged.
- Done when: round-trip error < 0.01 px; set A boxes within 1 px of `ref-b5a`, P-box >= 0.98.

**5b-G2 #407 stage 1: fewer syncs and allocations (only if 5b-P1 supports it).** Size L.
- Design source: `sp_gpu_prep_concurrency.md` C2 (one packed pinned upload per phase; batched
  resample launches over a job table; one device-to-host copy for all crops; about 4-5 syncs per
  image from about 25).
- Library first: evaluate CV-CUDA's batched variable-shape `resize` / `pillowresize`,
  `copymakeborder` and `convertto` (https://cvcuda.github.io/CV-CUDA/) against the existing CuPy
  kernels on the same inputs. Adopt CV-CUDA if its output is byte-identical to today's
  `prep_kernels.py` (then the gate stays byte-identity), or if it is within the tensor gate
  (best shift (0,0), mean abs diff < 0.5 grey, max <= 8) **and** P-box/P-pe pass against `ref-b5a`.
  Write custom batched kernels (C2 as designed) only if CV-CUDA fails both, and say why in the PR.
- Constraint: `gpu_image_prep` is a decoupled Python model; the python backend docs say async BLS
  is not supported in decoupled mode, so keep the per-image worker threads and their BLS calls.
- Red: `scripts/bench/gpu_prep_parity.py compare` byte-identical (or the tensor gate, per the
  adoption path); offline stand-in tests as `sp_gpu_prep_concurrency.md` 6.1; a sync-count test
  with the stand-ins (about 25 -> at most 5 per image).
- Pass: B-4 and B-8 above the 5a ingest-profile row by > 2x spread; small-card checks pass.
- Rollback: the model's `MODEL_VERSION` and a config parameter selecting the old kernels until the
  gate passes; then the old path is deleted (no shim).

**5b-G3 #407 stage 2: DALI for decode and letterbox only (only if decode/letterbox binds).** Size M.
- Library: NVIDIA DALI with the Triton DALI backend (https://docs.nvidia.com/deeplearning/dali/user-guide/docs/),
  `fn.decoders.image(device="mixed")`, `fn.resize`, `fn.crop`; the three parity traps and the
  working pipeline are in `docs/research/dali_gpu_decode_lessons.md` (resize defaults
  `antialias=False, subpixel_scale=False`; Ultralytics pad split `round(d - 0.1)`; read shapes with
  `fn.peek_image_shape` on the CPU). The box-dependent crops, detector and PE calls and crop JPEG
  encode stay in `gpu_image_prep`.
- Check: chroma upsampling and EXIF must match the current decoder (gpu_decode_plan 6.1: default
  GPU chroma differs from libjpeg; DALI's default ignores EXIF orientation).
- Gate: tensor gate on the letterbox, then boxes within 1 px of `ref-b5a`, P-box >= 0.98, PE cosine
  gates; pass bar as 5b-G2.

**5b-G4 #407 stage 3: native backend (rule only).** A C++ Triton backend is evaluated only if the
5b-P1 profile, repeated after G2, still shows Python/GIL overhead as the binding stage. That
evaluation is its own plan file (`docs/design/`) before any code. Rust adds no support advantage
(Triton's backend API is C++), per #407.

### Track `routes` (#40 Wave 6)

**5b-W1 Stateless routes on the GPU decode: bake-off before adopting.** Size M.
- Arms on `/detect` and `/embed/image` with set B at 8, 32 and 128 clients, 3 rounds:
  A = CPU decode (today, after 5b-D1); B = DALI ensemble (`fn.decoders.image` mixed + letterbox,
  1/2/4 instances, Triton ensemble to the detector); C = reuse `gpu_image_prep` with detect-only
  flags (no new model).
- Choose by images/s per GB of GPU memory and p99. Adopt only if the best arm is >= 1.5x arm A at
  32 clients with p99 not worse (Wave 6 acceptance) and the gates pass (tensor gate, boxes within
  1 px, P-clip p01 >= 0.995). Otherwise record and ship nothing.
- Box un-mapping on the client from the JPEG header (size plus EXIF orientation, no decode) with
  the same arithmetic as the CPU path; equality test.
- Switch: a setting (default off) until the gate passes; `env.template` and
  `tests/test_env_surface.py` in the same commit.

### Track `codec` (follow-ups from #217)

**5b-C1 Crop JPEG encode and progressive decode.** Size M.
- Library: nvImageCodec (`Encoder`, batch decode of heterogeneous images;
  https://docs.nvidia.com/cuda/nvimagecodec/) for (a) crop JPEG encode on the GPU inside
  `gpu_image_prep` (today: host threads, a stage of 12 % at one batch in the 0.7.0 profile) and
  (b) progressive JPEGs (11 % of set B falls back to the CPU decode).
- Gate for (a): the crop bytes change, so P-vlm, P-sam and P-region apply (they are owned by 5c's
  gate run; run (a) together with 5c's gate if the VLM and segmenter are not up in 5b), plus P-pe
  unchanged (PE reads pixels, not the JPEG). Gate for (b): the decoder's pixels for progressive
  files against the CPU decoder (tensor gate) and boxes within 1 px.
- Adopt only with a > 2x spread gain on B-4/B-8; else record.

## 6. Gate (5b PR)

- T0 (targeted red/green plus static checks on the touched files) per task; one offline sweep on
  the merged integration head (pre-commit `--all-files`, suite `-n 8`, contracts-check, docs check,
  CHANGELOG rules; frontend only if `frontend/` changed). No long or GPU run per branch.
- The batch's benchmark rows and parity compares on GPU 0 (GPU 2 when agreed), in the B5 GPU window
  after 5a's baseline (`decision_register.md` section 8.1); one confirmation run of the final head against
  5a on set A and set B.
- #40 checks (public): set A one card not below the 5a A-batch row by more than the spread; set B
  above it; decode once per image (the `decode` stage counter equals the image count on every
  route that 5b touched).

## 7. Done when

- GPU-3b and GPU-12 done-whens hold;
- each #407 stage either adopted with its pass line or recorded as not adopted with the binding
  stage named; #407 closes with that record;
- Wave 6 adopted or recorded;
- numbers in `docs/PERFORMANCE.md`; CI green; PR merged with a merge commit;
- the R-3 docs removal for 5b is done in the closing PR (`gpu_decode_plan.md` and
  `sp_gpu_prep_concurrency.md`; table in `06_b5_overview.md` section 10.1), and this file is deleted
  after the keep-list below is moved (`decision_register.md` "Plan lifecycle"), with its rows removed from
  `decision_register.md` and `docs/design/README.md`.
- #40 closes when its children (#45, #268, #219, #220, #67, #231 decision) are closed and its
  whole-epic acceptance is recorded on public data.

### What to keep after merge

| Fact | Target |
|---|---|
| the one-decode helper contract (oriented, location, error type), the GPU route and its switches | `docs/ARCHITECTURE.md` (GPU pipeline) |
| profile table, adopted/not-adopted record per #407 stage, Wave 6 bake-off, 5b-D2 and 5b-C1 results | `docs/PERFORMANCE.md`, method note in `docs/research/` |
| rotated-photo behaviour change, new settings | CHANGELOG, `env.template`, docs-site config pages |

## 8. Docs, CHANGELOG, upgrade impact, rollback

- Docs: `docs/PERFORMANCE.md`, `docs/ARCHITECTURE.md` (GPU pipeline), `docs/research/` (profile),
  `env.template` and docs-site config pages for any new setting.
- CHANGELOG: "Changed" (rotated photos now decode upright on `/embed`, `/analyze` and legacy
  ingest; sub-pixel box change from GPU-12), "Added" for new settings.
- Upgrade: `gpu_image_prep` changes ship with a `MODEL_VERSION` bump so `export/sync_model_seed.py`
  replaces the installed model; a DALI model (if adopted) is a new model directory; no engine
  rebuild.
- Rollback: each track reverts alone; new paths stay behind their switch until their gate passes.

## 9. Risks

| Risk | Mitigation |
|---|---|
| GPU-3b changes vectors of rotated photos already stored by `/embed` users | it is a correctness fix (vectors now match ingest); CHANGELOG; no stored-data migration (core `/ingest` documents keep their vectors) |
| new kernels or CV-CUDA move pixels | byte-identity or tensor gate first, then task gates |
| DALI defaults silently differ from the CPU path | the three documented traps; tensor gate |
| optimizing a stage that does not bind | 5b-P1 first; the decision rule |
| GPU memory regressions on small cards | the ballast checks |

## 10. Lane briefs (ready to paste)

Common block: as in `04_b3_ultralytics_one_version.md` section 17, with worktree
`<repo>-wt-b5b-<track>`, branch `b5b/<track>`, file `docs/design/v070_rebuild/06b_b5_decode_once.md`,
plus: "Benchmarks and parity follow `06_b5_overview.md` M-1..M-10 and section 6, against
`ref-b5a`. Stack `docker compose -p op-b5b-bench`, GPU 0 only unless the coordinator says GPU 2 is
agreed. Project slugs start with `b5b-`. Evaluate the library named in your task before writing
custom code and put the comparison in the PR."

- **`decode`**: "Do 5b-D1 (red EXIF 1-8 test first), then 5b-D2. No shim at the old import path. If the
  coordinator says D-35 chose `scaleup=True`, add that change as its own commit."
- **closing PR (coordinator)**: "Move the keep-list, remove the R-3 docs, delete this sub-batch's
  plan files: `decision_register.md` Plan lifecycle."
- **`profile`**: "Run 5b-P1. No code. Report the table and the decision-rule outcome."
- **`prep`**: "Do 5b-G1. Then 5b-G2 only if the coordinator forwards a 5b-P1 report that supports
  it; evaluate CV-CUDA first. 5b-G3 only on the decision rule. Do not start 5b-G4."
- **`routes`**: "Do 5b-W1 as a bake-off; ship code only if the 1.5x bar and the gates pass."
- **`codec`**: "Do 5b-C1 with nvImageCodec; coordinate (a)'s P-vlm/P-sam/P-region gate with 5c."

## Research amendments (2026-11-04, the research integration pass)

| Where | Change |
|---|---|
| 5b-D1 | **research spatial finding L-X1: exports ignore EXIF orientation 2-8.** Add `src/services/curation/export_support.py::_copy_or_resize_one` to "Callers to adapt": the resize and crop path decodes through the oriented helper instead of a raw decode without `exif_transpose`. Red: `tests/curation/test_export_orientation.py`: an orientation-6 fixture exported with `resize_mode` set lands its boxes on the object (red today). Control: a tag-1 fixture exports byte-identical. The move of `decode_image` to `src/utils/image_decode.py` carries the W4 capture header read and `DecodedImage.capture` with it |
| 5b-D1 metrics | GPU S-18: `op_ingest_decode_total` and `op_reprocess_decode_total` are renamed to one `op_decode_route_total{route, path}` in the same commit, no alias |
| 5b-P1 | Nsight Systems is not in the Triton image: mount the host `nsys` read-only into the bench stack (`gpu:PR-3` and `gpu:PR-4` are cut before the tag, so there are no prototype traces to reuse); add the DCGM fields to the report |
| 5b-G1 | GPU S-17: one geometry module, `src/domain/image_geometry.py` (letterbox, square, crop rectangle, un-map), copied by `export/sync_model_seed.py` into the image-prep model directory at seed time; replaces the pair `src/services/detection/geometry.py` and `models/gpu_image_prep/1/prep_geometry.py`. Red: `tests/test_geometry_copy_identical.py`: the copy in the model directory is byte-identical to the domain module |
| 5b-G2 | the CV-CUDA evaluation is `gpu:PR-2`; the lane reads its record before writing kernels (a library that passes replaces the custom code) |
| 5b-G3 | set `jpeg_fancy_upsampling=True` and `adjust_orientation=True` explicitly; verify ragged JPEG batches on the pinned DALI backend before relying on them |
| 5b-G4 | **not opened in v0.7.0** (owner 2026-11-04: short runs only, about 10 GPU hours before the tag; `gpu:PR-3` and `gpu:PR-4` are cut, so W5b keeps today's instance shape and the Python backend). After the tag, on the owner's OK, the rule is the `gpu:PR-4` bar: a C++ backend is evaluated only if the through-Triton gap is at least 25 % against the direct pipeline at SM active below 85 % (RD-15), with its own plan file (RW-1) |
| 5b-C1 | nvImageCodec's `Encoder` refuses chroma-subsampled image objects: encode from the RGB crop array and record the encoder id used |
| 5b metrics | GPU S-19, S-20: `op_gpu_prep_stage_seconds` and `op_gpu_prep_stage_observations` gain the label `server`; new gauges `op_gpu_prep_budget_bytes_in_use{server}` and `op_gpu_prep_frames_in_flight{server}` |
