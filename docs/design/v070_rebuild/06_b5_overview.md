# B5: speed on the final stack (overview, shared protocol, #42 side track)

| Field | Value |
|---|---|
| Status | plan, not started |
| Release | v0.7.0 (inside the release; R-final runs after B5). D-37 is decided: yes (section 2) |
| Depends on | B4 merged (`05_b4_stateless_api.md`) |
| Sub-batches | `06a_b5_baseline.md` (5a), `06b_b5_decode_once.md` (5b), `06c_b5_crops_ocr_faces.md` (5c), `06d_b5_processing_plans.md` (5d); order 5a, then 5b, then 5c, then 5d |
| Decisions applied | D-5 (#42 side track, never blocks), D-13 (B5 last), owner decisions of 2026-11-03 items 2, 5, 6, 7 |
| Open decisions | D-35 (GPU-6, owned by `06a_b5_baseline.md`; register in [decision_register.md](reference/decision_register.md) section 6); OQ-B5-1..3 (section 9) |
| Source IDs | former master plan (rev 4, in git history) 7.5, 7.6, 7.7 (speed items), 8 (D-5, D-35), 9.1 (N13, N14 parts), 9.2 (#40, #42, #220, #407), 10, 11, 12 (T3, T5), 15, 16; issues #40, #42, #45, #67, #219, #220, #231, #268, #403, #407 |
| Verified | Verified against `fix/411-r2-round` at `af62c947` on 2026-11-03 (read-only `rg`, `gh issue view`; nothing run) |
| Consolidated | Consolidated 2026-11-03 |

## 1. Goal

The mission's "fast" half: process images fast for many users at once, measured once on a stack
that has stopped changing. B5 takes one public baseline (5a), then removes repeated decodes and
copies (5b), shrinks crops and OCR/face work to model size (5c), and lets a request run only the
branches it asked for (5d). Every change passes a parity gate against the 5a reference and beats
the run-to-run spread before a default moves.

Rule from the owner: **#45 baseline before any optimization code.** No 5b, 5c or 5d code merges
before 5a's reference numbers and parity captures exist.

## 2. Release placement (D-37, decided yes)

B5 is inside the release: v0.7.0 = B0 to B5, and R-final runs last, after 5d (decision D-37 in
[decision_register.md](reference/decision_register.md) section 6). There is no separate speed release and no separate
acceptance run. 5a is taken once, on the final B4 stack, and every later gate compares to it. If
the stack under 5a changes before 5d merges, 5a is retaken only for the changed part (diff the
image digests and model configs recorded in 5a's environment JSON).

**Compute.** B5 is the second of the three GPU windows (budget: [decision_register.md](reference/decision_register.md)
section 8.1 "Test and GPU budget"). Sub-batches code in their worktrees with targeted tests only
(T0); the baseline is measured once (5a), then the measurement and parity gate of 5b, 5c and 5d
run in sequence inside that one window, each on the integration head of its sub-batch. No long or
GPU run per branch or per commit.

## 3. What is already done (do not redo)

Every figure is from `docs/PERFORMANCE.md` (section "Summary of the optimization milestone" unless
named) or the issue comments.

| Area | Done | Source |
|---|---|---|
| PE image encoder | FP16 engine, 13.7 -> 5.5 ms per embedding; INT8 rejected (fails accuracy gates) | PERFORMANCE "FP16 PE engine", "INT8 evaluation" |
| Inputs | uint8 detector and PE inputs, 14.94 -> 3.73 MB per image to Triton | "uint8 inputs" (#214) |
| Decode | CPU single decode; Triton GPU decode `gpu_image_prep` (nvJPEG GPU-Huffman, fused CuPy kernels), opt-in, 3 MP size gate; photos 27.0 -> 35.0 img/s (set B) | `gpu_decode_plan.md` 13, "Decode once and the GPU decode" |
| Multi-GPU | data-parallel PE, 2 x A6000 44.60 -> 55.08-56.37 img/s tuned | "Multi-GPU PE encoding", "Stage 2 tuning" |
| Stage 2 | one-card detector engine and concurrency sweep, 23.39 -> 28.81-29.07 img/s set A | "Stage 2 tuning" (#226, #259) |
| 0.7.0 ingest speed | `gpu_image_prep` memory budget, stage counters, ingest load profile; 2 instances +51 % on the ingest profile | "0.7.0 stage profile", "GPU memory and the ingest load profile" (#390, #392, #393) |
| Ladders | 2k/4k COCO pins, 10k COCO and 100k Open Images v7 ladders | "Scale ladder: 10k and 100k" |
| OCR | #220 step 1: uint8 `ocr_images` and `original_image` (commit `27ebeb45`, `tests/test_ocr_uint8_inputs.py`) | `faces_ocr_resident_crops_plan.md` |
| OCR buckets | #220 step 2 measured and **rejected**: bucket padding P-ocr 86 %, exact-width grouping no speedup; evidence: the #220 comment of 2026-11-02 | #220 comment 2026-11-02 |
| Plans | #67 P1: `src/domain/processing_plan.py`, `src/schemas/processing_plan.py`, `tests/test_processing_plan.py`; design doc `processing_plans_plan.md` | PR #426 |
| Not adopted | system shared memory (under 5 % of API CPU); gRPC ingest channel | "shared memory", "Ingest transports" |

## 4. Sub-batch map

| Sub | File | Branch (PR) | Scope | Gate summary | Issues closed |
|---|---|---|---|---|---|
| 5a | `06a_b5_baseline.md` | `int/b5a-baseline` | #45 public ladders; #268 sweeps (set B, profiles, two-card, CUDA graphs, `priority_levels`, 2-instance parity); GPU-6 recall for D-35 | numbers recorded with host load; a default changes only if the gain beats 2x the spread and parity passes | #45, #268, N13 (after D-35) |
| 5b | `06b_b5_decode_once.md` | `int/b5b-decode-once` | GPU-3b one oriented decode; #40 remaining waves; #407 profile-first speed ideas; GPU-12 | boxes within 1 px at original resolution; embedding cosine median >= 0.999, p01 >= 0.995; byte-identical where the math is unchanged | #40 (epic, when all children close), #407 |
| 5c | `06c_b5_crops_ocr_faces.md` | `int/b5c-crops-ocr-faces` | #219 work crops at model size; #220 OCR (GPU-7 `async_exec`, padding-aware decode or argmax in the graph, step 3 research); GPU-19 evidence | P-vlm, P-sam, P-region, P-ocr; crop-cache bytes | #219, #220 |
| 5d | `06d_b5_processing_plans.md` | `int/b5d-processing-plans` | #67 P0, P2-P6, including face alignment on resident frames (P5) | per-stage "not run" tests; M1-M4; P-face, ArcFace p01 >= 0.995 | #67 |

Branch model (owner decision 6): one integration branch per sub-batch, cut from main after the
previous sub-batch merged; one PR and one T1 sweep per sub-batch, because each is gated on its own
measurement. Tracks: one Sonnet agent per worktree (effort high for the W5b decode-once GPU tracks
`decode`, `prep` and `codec`, medium otherwise; D-85) `<repo>-wt-b5<x>-<track>`, branch
`b5<x>/<track>`, merged `--no-ff` one merge commit per track.

**Test cycle per sub-batch (once):** each task's red test and control in its lane (T0: targeted
red/green plus static checks on the touched files); on the merged integration head one offline
sweep (pre-commit `--all-files`, full suite `-n 8`, contracts-check, docs check, CHANGELOG rules;
the frontend set only if `frontend/` changed); the sub-batch's benchmark and parity gate on GPU 0
(GPU 2 when agreed) inside the B5 GPU window; one confirmation run; numbers into
`docs/PERFORMANCE.md`. Local CI parity before every push.

**Size:** XL over four PRs (5a M-L in GPU time, 5b L, 5c L, 5d L).

## 5. Shared measurement protocol (every B5 number)

This protocol is self-contained (the retired ship-track plan's measurement rules are inlined below)
and extends the baseline protocol in `docs/PERFORMANCE.md` ("Baseline protocol", "Wave 0 parity and
FP16 gates"). Two rules carried from the ship track: the crop cache cap is off when counting files
(`OP_CROP_CACHE_MAX_BYTES=0`), and stage seconds are sums over concurrent work, so they are compared
across runs, never added up to wall time.

| # | Rule |
|---|---|
| M-1 | **Public data only** (owner rule; changed from the ship track, which also used the private set P). Sets: A = `coco_bench_2000` (2,000 COCO val images, about 333 MB, seed 20261009); B = `commons_hires_500` (500 Wikimedia Commons photos, 12-24 MP, about 5.2 GB, seed 20261010); COCO val2017 (5,000, about 1 GB); COCO train2017 + val2017 (123,287 images, about 19 GB; with test2017 about 164k and 25 GB); Open Images v7 pins (`openimages_bench_100000`). Manifests in `scripts/datasets/manifests/`, each with seed, count and sha256 |
| M-2 | **Isolated stack**: own compose project (`-p op-b5<x>-bench`), own port base, Triton and API on GPU 0 (`TRITON_GPU_ID=0`, `API_GPU_ID=0` in the stack's env); GPU 2 only when agreed with peer sessions; never the default project |
| M-3 | **Host load recorded** for every run: `/proc/loadavg` at start and end, `/proc/pressure/io` through `scripts/bench/host_pressure.py`, foreign GPU processes (`nvidia-smi --query-compute-apps=pid,used_memory --format=csv`) before the run, GPU sampled every 500 ms (`nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv -lms 500`), Triton CPU from `docker stats --no-stream`. A run whose 1-minute load exceeds 16 is **noisy**: repeat that point only |
| M-4 | **Per run**: fresh project with a `b5<x>-` slug prefix (never reuse or touch existing `baseline-*` projects); input page cache dropped (`posix_fadvise(DONTNEED)`); first 100 images are warm-up, excluded |
| M-5 | **Repetitions and spread**: 1 warm-up + 3 counted repetitions per headline row; report median and min-max; **spread = (max - min) / median**. A change is adopted only if its median gain exceeds **2x the larger spread** of the two arms and the parity gate passes |
| M-6 | **Tiers** (short tests when the gain is known): tier 1 = one repetition on a 500-image slice to screen; tier 2 = 3 repetitions on the candidate and its control only; one full confirmation per bundle. No full-factorial sweeps; one factor at a time |
| M-7 | **Record**: commit, image digests, driver, CUDA, Triton, TensorRT, torch, Ultralytics versions, model configs (instance counts, batch, delays), env profile; per-stage report (API stage shares from `op_pipeline_stage_seconds`, Triton `nv_inference_queue_duration_us` / `compute_infer` deltas, `op_gpu_prep_stage_seconds`, GPU util mean and p50, API CPU-s per image) |
| M-8 | **Raw data**: `artifacts_local/bench/v080/<sub>/<row>/` (gitignored, in the main checkout). Public result files that back a published number: `docs/benchmarks/v080/<sub>/` with `MANIFEST.sha256` updated |
| M-9 | **Where numbers go**: `docs/PERFORMANCE.md` (public rows), `docs/research/` for the notes; never a private-set number in public text |
| M-10 | **Honesty**: never invent, round up or carry forward a number; a bar not met is reported as not met with the binding stage named |

Tools that already exist (use them, do not rewrite): `scripts/bench/run_baseline.py`,
`ingest_profile.py`, `ingest_sweep_suite.py`, `scale_run.py`, `parity.py capture|compare`
(profiles in `parity_gates.json`), `gpu_prep_parity.py`, `detector_quality.py`, `host_pressure.py`,
`scripts/datasets/fetch_coco_subset.py`, `fetch_hires_set.py`, `fetch_openimages_bench.py`.

## 6. Shared parity gates

The 5a capture `ref-b5a` (`parity.py capture --tag ref-b5a` on set A and set B) is the reference for
every later gate. Gates by kind of change:

| Change kind | Gate |
|---|---|
| math unchanged (copies, syncs, batching, launch order) | **byte-identical** outputs (`gpu_prep_parity.py compare`, P-json for routes) |
| geometry or decode numerics | boxes identical within **1 px at the original resolution** on every matched box; P-box >= 0.98, P-item >= 0.99 (`parity_gates.json` profile `precision`) |
| embeddings | PE crop and frame cosine **median >= 0.999, p01 >= 0.995**; P-clip, P-face p01 >= 0.995; P-knn overlap@10 >= 0.95; P-cluster-assign >= 0.98 |
| OCR | **P-ocr**: exact text on >= 98 % of lines after matching boxes at IoU >= 0.5; character error rate not worse |
| region stage | **P-region**: hit/miss agreement >= 99 %, box IoU >= 0.9 |
| segmenter | **P-sam**: top-1 hit/miss >= 97 %, box IoU mean >= 0.97, p05 >= 0.90 |
| VLM | **P-vlm**: label agreement >= 97 % on the same crops; every disagreement listed for a human |
| benchmark | at the target concurrency, throughput gain beats 2x spread (M-5), p99 not worse than +10 %, no CPU fallback lines, peak GPU memory within budget |

Source of the thresholds: `docs/design/triton_pipeline_optimization_plan.md` section 7 and
`scripts/bench/parity_gates.json`.

## 7. Libraries and documented patterns to evaluate before custom code

Owner rule: docs and maintained libraries before hand-rolling. Each sub-batch file names where it
applies. Licences must be read from each project's LICENSE at evaluation time and an `ATTRIBUTION.md`
row added in the adopting commit.

| Library / feature | What it offers here | Doc (read 2026-11-03) | Where |
|---|---|---|---|
| NVIDIA DALI + Triton DALI backend | GPU decode and letterbox pipelines inside Triton ("a GPU-accelerated library for data loading and pre-processing"); known traps for Ultralytics parity in `docs/research/dali_gpu_decode_lessons.md` | https://docs.nvidia.com/deeplearning/dali/user-guide/docs/ ; https://github.com/triton-inference-server/dali_backend | 5b (#407 stage 2, stateless routes) |
| nvImageCodec | "a library of accelerated codecs with unified interface", batch decode of heterogeneous images, an `Encoder` (crop JPEG encode candidate) | https://docs.nvidia.com/cuda/nvimagecodec/ | 5b (GPU crop encode, progressive JPEG), 5c |
| CV-CUDA | batched GPU operators incl. `resize`, `pillowresize`, `copymakeborder`, `warp_affine`, `normalize`, `convertto`; variable-shape batches (`ImageBatchVarShape`); interop with PyTorch, CuPy, nvImageCodec | https://cvcuda.github.io/CV-CUDA/ | 5b (#407 C2 batched resample), 5d (face alignment `warp_affine`) |
| Triton python backend `async_exec` | "multiple inflight inference requests" from a BLS; needs `async def execute`; **"Async BLS is not supported when running a Python model in decoupled mode"** | https://github.com/triton-inference-server/python_backend | 5c (OCR BLS, not decoupled); not usable in `gpu_image_prep` (decoupled) |
| Triton dynamic batcher | "requests can be dynamically batched only if each input has the same shape across the requests"; `max_queue_delay_microseconds`; `preferred_batch_size` "should not be used for most models"; `priority_levels` | https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html ; .../ragged_batching.html | 5a (priority levels), 5c (OCR widths) |
| Triton CUDA graphs | `optimization { cuda { graphs: true graph_spec ... } }`, "Currently only recognized by TensorRT backend" | https://github.com/triton-inference-server/common/blob/main/protobuf/model_config.proto | 5a (#268 detector) |
| Triton rate limiter, warmup, instance groups | warmup blocks READY until done; rate-limiter priorities | https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html | 5a |
| perf_analyzer / Model Analyzer | the documented tuning tools | https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/optimization.html | 5a, 5b |
| Nsight Systems, py-spy | timeline of CUDA launches and syncs; Python stack sampling (already used in the repo) | https://docs.nvidia.com/nsight-systems/UserGuide/index.html ; https://github.com/benfred/py-spy | 5b (profile before #407 code) |

## 8. B5 gates feed R-final (procedure owned by `07_release_acceptance.md`)

After 5d merges, R-final in `07_release_acceptance.md` runs on main and carries these B5 checks
(there is no separate acceptance run for B5):

- `acceptance_run.sh --only` ingest, cluster, export, train, promote, and the faces/OCR/analyze
  endpoints the sub-batches changed;
- `tests/live`, `tests/test_full_system.py`, `tests/validate_visual_results.py`;
- a final set A and set B confirmation row (3 repetitions) against 5a, recorded in
  `docs/PERFORMANCE.md`;
- `installer_upgrade` from 0.6.0 (engine rebuilds only if 5b or 5c changed a model graph; the
  stamp decides);
- image-vs-git audit, Trivy, screenshots at 1440 and 430 px of any Cropwright page touched.

The tag v0.7.0 is cut only on the owner's word (see `07_release_acceptance.md`).

## 9. Open questions (present with a recommendation)

| ID | Question | Recommendation | Consequence | Needed by |
|---|---|---|---|---|
| OQ-B5-1 | #40, #45 and #407 acceptance bars name the private set P and a private 100k 20 MP collection; the owner rule is public-only | replace them with public set B (pinned 500; extend to 2,000 with `fetch_hires_set.py --n 2000`, a new pinned manifest, at a rate the Commons API accepts) and drop the private 100k row from public claims; private runs may still inform aggregates in `docs/research/` only if the owner says so | the "20 MP 100k" row of #45 has no public equivalent at 100k; #40's photo bar is judged on set B | before 5a |
| OQ-B5-2 | the 1M Open Images run | extrapolate from measured per-image cost first (#45); run 1M only on the owner's word (storage about 1 TB of images plus the index) | no surprise multi-day run on a shared host | 5a |
| OQ-B5-3 | GPU 2 for two-card rows | ask the peer sessions each time; two-card rows are optional for every gate except #268's two-card sweep | the gate never waits on GPU 2 | 5a |
| D-35 | GPU-6 `/detect` upscaling | see `06a_b5_baseline.md` and [decision_register.md](reference/decision_register.md) section 6 | | end of 5a |

## 10. Deferred inside the milestone (not in a sub-batch)

| Issue | Rule |
|---|---|
| #231 ortloom-serve vs Triton | owner gate (2026-10-10): only after the Triton work lands. Runs after 5b against the post-#407 Triton baseline with its own decision rule (extend only if PE embeddings/s >= 1.10x at equal p99, or set B >= 1.3x with P-box/P-item passing). Otherwise record and close |
| #403 CUDA MPS overlay | parked: the 2-instance slowdown was device-memory exhaustion (#392). Revisit only if #407 stage 1 leaves a context time-slicing gap; the one-off check is E8 in `sp_gpu_prep_concurrency.md`; adopt only with >= 10 % gain; opt-in overlay only (needs host root) |
| #415 engine build stamps (yolo, mobileclip, faces, ocr) | moved out of #268; not B5 |
| Cluster-pass geometry stage (#337 leftover: 0.8 ms per crop, the largest stage since the fit change) | act only if the 2.5 ms/crop bar tightens; open an issue then |
| `triton_pipeline_optimization_plan.md` Wave 8 (region stage on resident crops; segmenter on TensorRT) | v0.8.0 (platform breadth); needs its own plan file before work (owner answer to its Q8) |
| `triton_pipeline_optimization_plan.md` Wave 10 embedding-space migration | v0.8.0 (platform breadth) with #365 (one re-embed with a version field; no in-place swap) |

### 10.1 Docs removed after each sub-batch (task text for the sub-batch's docs commit)

Removal wave R-3 (rules in `prior_plans_reconciliation.md` section 6). Move the content first,
fix every reference (`rg -l '<file name>'` outside `CHANGELOG.md` entries of past releases), then
`git rm` in the same PR. Plan lifecycle: [decision_register.md](reference/decision_register.md) "Plan lifecycle".

| After | Doc removed | Content to move first (target) | References to fix at that time |
|---|---|---|---|
| 5b | `gpu_decode_plan.md` | section 13 build record and 6.1 decoder notes -> `docs/PERFORMANCE.md` "Decode once and the GPU decode" and `docs/ARCHITECTURE.md` (GPU pipeline) | `scripts/bench/gpu_decode_bench.py` line 2 (cited-paths guard), `docs/PERFORMANCE.md` lines 2976, 3215, 3316, `07b`, `07`, `decision_register.md`, `10`, `processing_plans_plan.md`, `faces_ocr_resident_crops_plan.md`; issue comments #40, #220 |
| 5b | `sp_gpu_prep_concurrency.md` | sections 2-4 diagnosis -> `docs/research/v070_measurement_log.md` (or a new `docs/research/` note) | `docs/benchmarks/v070/sp0/gpu_prep_concurrency_matrix.md`, `docs/reviews/sp_fixes_review.md`, `docs/WHITEPAPER.md`, `07`, `07b`, `decision_register.md`, `processing_plans_plan.md`; issue comments #403, #407 |
| 5c | `faces_ocr_resident_crops_plan.md` | none (`07c` and `07d` hold the steps) | `07`, `07c`, `07d`, `processing_plans_plan.md`; issue comment #220 |
| 5d | `processing_plans_plan.md` | branch vocabulary and per-route defaults -> `docs/ARCHITECTURE.md` and the API docs | `CHANGELOG.md` (`[Unreleased]` mention: re-point), `decision_register.md`, `07`, `07b`, `07d` |
| 5d | `triton_pipeline_optimization_plan.md` | section 7 accuracy gates -> `docs/PERFORMANCE.md` parity section (section 11 verification log is already there) | `docs/PERFORMANCE.md` lines 7, 421, 516, 748, 1007, `docs/VISION_AND_GOALS.md` line 115, `docs/research/triton_deep_research_2026-09.md` line 1, `scripts/bench/parity_gates.json`, the `07` files, `decision_register.md`; issue comments #40, #67, #84, #219, #220 |

Also at each sub-batch's closing PR: delete that sub-batch's `v070_rebuild/06*` file(s) after moving
the keep-list in its "What to keep after merge" section, and remove its row from `decision_register.md`
and `docs/design/README.md`. The overview `06_b5_overview.md` goes with 5d (after the #42 side
track facts move to `docs/PERFORMANCE.md`).

### 10.2 What to keep after merge (this file)

| Fact or section | Target |
|---|---|
| section 3 "done" figures are already in `docs/PERFORMANCE.md`; keep nothing | none |
| section 5 measurement rules M-1..M-10 and section 6 parity gates | `docs/PERFORMANCE.md` "Benchmark protocol" (merge with the existing baseline protocol), `scripts/bench/parity_gates.json` stays |
| section 7 library table (only the rows adopted) | `docs/ARCHITECTURE.md` (GPU pipeline) and `ATTRIBUTION.md` rows |
| section 10 deferred rows (#231, #403, #415, Wave 8, Wave 10, geometry stage) | the GitHub issues named, each with the rule as a comment |
| section 11 #42 result | `docs/PERFORMANCE.md` and the #42 comment |

Done-when for 5d includes: this file and the 07a-07d files are deleted per the lifecycle rule.

## 11. Side track: #42 small-GPU acceptance (never blocks a batch)

- **What is done:** the `small` profile shipped in #441 (auto-picked under 16 GB; old stamps read
  as default). Only the 12 GB-card acceptance remains.
- **When:** whenever the peer session that owns GPU 1 (the transcribe app) says the RTX 3080 Ti
  (12 GB) is free. Ask it first and wait for its reply; never stop another session's stack.
  Owner decision 2026-10-29: test on the real 12 GB card; CUDA MPS capping GPU 0 stays a fallback.
- **Worktree:** `<repo>-wt-fix42` from main, branch `perf/42-small-gpu-acceptance` (recreate; the
  old empty one was removed). Rebase on main first.
- **Stack:** `-p op-42-small`, `TRITON_GPU_ID=1`, `API_GPU_ID=1`, profile `small`, GPU decode off.
- **Pass:** `small` profile engines built with the small target; Triton READY for every `standard`
  model with peak memory < 11 GB (500 ms samples); the live suite passes; set A ingest throughput
  recorded (3 repetitions, M-3 host record); embedding cosine p01 >= 0.995 against the 5a reference
  engines (or the B3 gate references if 5a has not run).
- **Results:** `docs/PERFORMANCE.md` and a comment on #42. If it has not run by the v0.7.0 tag, the
  release notes say the `small` profile is not yet verified on a 12 GB card (D-5). Milestone v0.8.0.
- **Brief:** "Sonnet agent for #42. Read `docs/design/v070_rebuild/06_b5_overview.md` section 11. Work
  only in `<repo>-wt-fix42`. Before any GPU use, confirm in writing that the GPU 1 owner session
  released the card; stop if `nvidia-smi` shows a foreign process on GPU 1. `docker compose -p
  op-42-small` only. No pgrep -f, pkill -f, sleep or until loops. Record host load per run. Report
  numbers against each pass line; never round or carry forward. Deadline: <set by coordinator>."

## 12. Docs, CHANGELOG, rollback, upgrade impact

- Docs per sub-batch: `docs/PERFORMANCE.md` (every gate), `docs/research/` (method notes),
  `docs/ARCHITECTURE.md` (GPU pipeline, 5b and 5d), `docs/OCR.md` (5c), contracts per sub-batch
  when a wire changes (5d), root `CHANGELOG.md` per PR, `frontend/CHANGELOG.md` only if Cropwright
  changes.
- Rollback: each sub-PR reverts alone; defaults move only behind a passed gate, and every new path
  keeps its switch until the gate passes.
- Upgrade impact: engine rebuilds only if 5b or 5c change a model graph (new version directory or
  `MODEL_VERSION`; the stamp handles it); otherwise none.

## 13. Risks

| Risk | Mitigation |
|---|---|
| peer load skews numbers (shared 48-core host, other projects on GPU 0) | M-3 record; noisy runs repeated; one heavy run at a time from this repo; announce to peers |
| parity regressions hidden by a better mean | every gate compares to `ref-b5a`; byte-identity where the math is unchanged |
| a gain that is noise | M-5: gain must beat 2x the spread |
| public set B is small (500) | extend to 2,000 (OQ-B5-1); report intervals |
| optimization on an old stack | 5a is taken on the final B4 stack (section 2); it records the digests |

## 14. Corrections (stale references found while verifying)

1. The former master plan (rev 4, in git history) 7.5 and 9.3 put B5 in v0.10.0; obsolete. B5 is
   inside v0.7.0 (D-37). GitHub issues #40, #42, #45, #67, #219, #220, #231, #268, #403, #407
   carry milestone `v0.9.0` today (#42 moves to v0.8.0 as a side track); the move is
   `github_actions.md`'s job.
2. The former master plan 7.5 says the OCR BLS gets batching from "Triton's dynamic batcher" with per-crop
   widths kept. The batcher batches only same-shape requests unless ragged batching is set up, so
   `async_exec` buys in-flight overlap, not cross-width batches (`06c_b5_crops_ocr_faces.md`).
3. The former master plan 7.5 puts face alignment on resident frames in 5c; the owner-approved #220 plan
   (J5) and `processing_plans_plan.md` P5 put it in #67, so it is in 5d.
4. `docs/PERFORMANCE.md` "Baseline protocol" still calls set B a private archive; the summary table
   and the 0.6.0 rows use the public Commons set B. Fix the protocol text in 5a.
5. `processing_plans_plan.md` 6.3 says `faces_ocr_resident_crops_plan.md` is "not written yet"; it
   exists and its step 1 is merged.
6. #40's acceptance says "P-clip/P-face/P-ocr = 1.0" while `parity_gates.json` uses p01 >= 0.995
   for cosine gates and 98 % for P-ocr; B5 uses the gates file.

## Research amendments (2026-11-04, the research integration pass)

Each line names its source; the GPU pipeline research is [09g_research_gpu_pipeline.md](09g_research_gpu_pipeline.md) (its prototypes are `gpu:PR-0..PR-9`
in [00f](08_research_prototype_track.md)). Pending owner question RD-2 (prototypes during W2-W4) for every
"read the record first" line: without it, the sub-batch runs as written above.

| Where | Change |
|---|---|
| M-1 | B5 gates and published rows stay on public data. Set P (the owner's private high-resolution photos) is allowed for the research prototypes and private gates only, under the one set-P rule of [00d](reference/owner_questions.md) RD-8: public text says only "met / not met on set P" |
| M-3, M-7 | record the DCGM profiling fields `SM_ACTIVE` and `PIPE_TENSOR_ACTIVE`, clocks, power and the GPU UUIDs (`gpu_devices`) for every row; `JobPerfSummary` in `jobs.result` is the per-job record ([02b](02b_jobs_locking_events.md) section 2.2) |
| section 7 library table | add: nvJPEG batched GPU Huffman is used only above a batch of about 50, and scaled and batched ROI decode are hardware-backend only (not on the A6000); DALI's `jpeg_fancy_upsampling` default is off (set it explicitly); CV-CUDA 0.19.0 is evaluated in `gpu:PR-2` and `gpu:PR-5` with its licence read at adoption |
| decisions fed by the prototype window | `gpu:PR-0` and `gpu:PR-1` -> 5a-R0 and 5a-S4; `PR-1`, `PR-2` -> 5b-G2, 5b-G3 (`PR-3`, `PR-4` cut before the tag, so 5b-G4 is not opened; [00f](08_research_prototype_track.md) section 2.2); `PR-6` -> 5c-O2, 5c-O3; `PR-5` -> 5d-P5 (each lane reads the record before writing code) |
| side track #42 | the `small` build profile is a separate `model_engines` row (`build_profile = 'small'`, GPU S-3) |
