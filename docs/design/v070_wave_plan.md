# v0.7.0 wave plan: production-ready for the winter labeling and training run

Status: plan (nothing implemented). Milestone v0.7.0. Issues: #337 #335 #286 #268 #259 #231 #220
#219 #84 #67 #45 #42 #41 #40, plus #341 (cluster evaluation harness). Accounted for but not built
here: the clustering epic #346 (#342 to #345, v0.8.0 and Backlog) and #347 (the embedding-model study,
Backlog).

The job this plan prepares for, before winter: ingest a very large set of public (COCO, Open
Images) and high-resolution owner-local photos, cluster them, label them with a VLM and the
auto-label loop, review, export a dataset, train a better vehicle detector, promote it. Every wave
below is judged by whether it makes that run faster, safer or measurable. Work that does not is
recommended for re-milestoning (section 1.3).

A fresh agent can execute any wave from this file alone. Paths were checked with `rg` against
`origin/main` at `fd64d086` (v0.6.0); line numbers are avoided on purpose, re-find by symbol.
New settings are written in lowercase (`crop_max_side`) as proposals; the implementing wave picks
the `OP_*` name and adds it to `env.template` and `tests/test_env_surface.py`. New routes are named
without a method prefix here until they exist in `contracts/openapi/curation.json`.

Read first: [`docs/VISION_AND_GOALS.md`](../VISION_AND_GOALS.md) (standards),
[`v060_wave_plan.md`](v060_wave_plan.md) (benchmark protocol, parity chain, gate table),
[`docs/PERFORMANCE.md`](../PERFORMANCE.md) ("Scale ladder: 10k and 100k", "Embed pass",
"Cluster pass plumbing"), [`ingest_write_path_investigation.md`](ingest_write_path_investigation.md),
[`triton_deep_research_2026-09.md`](../research/triton_deep_research_2026-09.md) (section 1:
keep Triton, move the image dataplane into it).

## 0. Ground rules (binding for every wave)

1. **Isolation.** The compose project `openprocessor` and any stack another session runs are live.
   Never run `docker compose` without `-p <name>` (`-p op070-w1` and so on), own ports, and pass
   `--container-prefix <name>` to `scripts/bench/scale_run.py` (its default is another stack's
   prefix). Preflight before every GPU run: `nvidia-smi`, check other agents' containers, time
   `curl /health` through the API port. Never stop or touch another project's containers.
2. **GPUs.** GPU 0 and GPU 2 (RTX A6000) are usable for tests. GPU 1 and part of GPU 0 are shared
   with other workloads: every GPU test is short (tiered, section 2.0), announced, and run in a
   window the owner confirmed. Two-card runs need both A6000s free at once (owner window).
3. **Data.** Committed material uses public data only: COCO (CC BY 4.0 annotations, Flickr images),
   Open Images v7 (CC BY 2.0 images), the pinned Commons photos (set B,
   `scripts/datasets/manifests/commons_hires_500.json`). The owner-local high-resolution photos are
   called **set P** in this plan and in every commit; their paths, counts per folder, file names,
   thumbnails and per-image results stay under `artifacts_local/` (gitignored). Public docs carry
   aggregates and pass/fail only. `scripts/codegen/check_naming_leaks.py` runs in pre-commit; do
   not introduce any other name for set P.
4. **Process.** One branch per wave, cut from `main` after the previous wave merged (or as stated).
   Targeted tests while working, red-first for every bug fix, then **one** offline full suite per
   wave: `.venv/bin/python -m pytest tests/ -q --no-cov -m 'not live'`. Every commit:
   `.venv/bin/pre-commit run --all-files` (never `--no-verify`). A route or wire change runs
   `make contracts` and commits `contracts/` in the same commit. No dead code, no shims: a
   superseded path is deleted in the same change. Conventional commits, merge with `--no-ff`. No
   push, PR, tag or release without the owner's yes.
5. **Worktree venv.** A new worktree has no `.venv`. Create it as CONTRIBUTING.md shows
   (`python3 -m venv .venv`, CPU torch from the PyTorch index, `requirements-test.txt`,
   `pre-commit`, `pytest-xdist`); never install `requirements.txt` into it.
6. **Benchmarks.** Protocol of `v060_wave_plan.md` section 0 rule 3 (fresh project per repetition,
   warm-up, load-average guard, `nvidia-smi dmon -s pucv` alongside). Short once a gain is known:
   one confirmation per bundle (3 repetitions on the headline row only), one-factor-at-a-time
   sweeps, never full factorial. Raw JSON to `artifacts_local/bench/v070/<wave>/`; public summaries
   to `docs/PERFORMANCE.md`.
7. **Parity.** A change that alters numerics of one stage passes that stage's gate and every
   downstream gate (`v060_wave_plan.md` section 2, thresholds in `scripts/bench/parity_gates.json`,
   tool `scripts/bench/parity.py capture|compare`). Capture the reference twice first (the detector
   is about 1 % non-reproducible run to run).

## 1. Issues: dependency graph, order, re-milestoning

### 1.1 Graph

```text
#341 eval harness ──┬──> #337 cluster pass ──┬──> #286 auto-label loop ──> rehearsal ──> #84 train/promote (public twin)
                    │                        │                                   │
#45a OI category ───┤    #259 embed pass ────┘ (vector-clamp decision before     └──> #45 rows (full COCO, 20 MP 100k, 1M?)
     pins + driver  │                          any big ingest)
                    └──> #342..#345, #347 (v0.8.0+, consume the harness, not built here)
set B re-pin ──> #268 (stamps, two-card, nvJPEG) ──> #219 work crops ──> #40 hi-res acceptance
#40 epic resident frame ──> #220 faces/OCR ──> #67 request plans ──> #231 ortloom head-to-head   (all v0.8.0)
everything above settles the dependency set ──> #41 slim images;  #335 screenshots (independent)
#268 standard-profile sweeps ──> #42 12 GB engines (v0.8.0)
```

### 1.2 Recommended order and rationale

1. **#341 cluster evaluation harness (W0).** Nothing else about clustering or label propagation can
   be trusted without a fixed public evaluation set that reports false merges separately from false
   splits. The loop (#286) propagates labels through clusters, so a false merge becomes a wrong
   label on hundreds of items; the cluster-pass fix (#337) must prove it did not change partitions
   for the worse; the v0.8.0 refinement phases (#342 to #345) and the embedding-model study (#347)
   score candidate partitions with it. Design it to take any partition as a JSONL file
   (`item_id`, `cluster_id`) so those later issues need no change to it. Keep.
2. **#45 part a: Open Images category subsets and the rehearsal driver (W0).** The winter run needs
   a large public vehicle set, and every later wave needs a cheap end-to-end check. The category
   fetch (vehicles, people, animals) is small code; the full-COCO and 20 MP rows are measurements
   that belong at the end (W4). **Split #45**: category pins plus full COCO plus the 20 MP 100k
   repeat stay in v0.7.0; the 1M row runs only if the owner grants the window and disk (O7),
   otherwise it moves to a new v0.8.0 issue with the extrapolation recorded.
3. **#337 cluster pass and storage attribution (W0 sampling, W1 fix, W2 admission item).** At the
   winter scale (about 1.2M crops, section 3) the cluster pass is the longest post-ingest stage:
   4.2 ms/crop (public) and 7.6 ms/crop (set P) means 85 to 155 minutes on the CPU. Finding: the
   issue's linearity acceptance is already met by the 0.6.0 data (269 s at 64,415 crops to
   2,042 s at 490,804 is 7.59x time for 7.62x crops, on different sets), so **re-scope the
   acceptance** to the absolute rate and the 1.8x high-resolution gap (W1 numbers). The IO-pressure
   sampling goes in W0 with the driver; the admission-gate per-image charge goes in W2 with the
   other high-resolution levers. Keep.
4. **#259 embed reprocess pass (W1).** A re-embed of a large project must not take twice the
   ingest time, and the crop-vector mismatch decision (ingest crops the unclamped box, the stored
   `bbox_norm` is clamped) changes PE inputs. If the owner chooses to clamp at ingest (O1), that
   numerics change must land **before** the winter ingest so the run is embedded once in its final
   space. Pairs with #337 in one wave: both are post-ingest passes over the same indexes and share
   the measurement window. Keep.
5. **#268 stage 2 follow-ups (W2).** Engine build stamps for the yolo, mobileclip, faces and ocr
   groups are a production-safety item (an engine that does not match its stamp must not reach
   READY). Set B numbers and the two-card sweep size the winter ingest. **Split #268**: CUDA graphs
   and `priority_levels` go to #40 (v0.8.0) unless W2 has time; the `minimal`/`standard` profile
   and RTX 3080 Ti sweeps move with #42 (GPU 1 is shared). Keep the rest.
6. **#219 work crops at model size (W2).** The single biggest saving for set P: about 7 crops per
   20 MP photo are cut, JPEG-encoded and cached at source size although the VLM and segmenter
   consume 336 to 1008 px; the crop cache (`crop_cache_max_bytes`, 3 GiB tmpfs by default) churns.
   Needs set B for its gates. Keep.
7. **#40 epic (W2 for the hi-res acceptance, rest v0.8.0).** The epic's 20 MP target (29 img/s,
   GPU above 75 %) is what the winter run needs; the measured limit is upstream of the GPU (reads,
   decode, the in-flight byte budget `OP_INGEST_INFLIGHT_MB`, storage), which W2 attacks. Keeping a
   decoded frame resident across detect, embed, faces and OCR (plan waves 6 to 8) does not help a
   vehicle run (ingest does not call faces or OCR). **Split #40**: the hi-res acceptance and the
   remaining v0.7.0 children stay; the resident-frame epic task moves to a new v0.8.0 issue.
8. **#286 cluster-sampled auto-labeling loop (W3).** The core of "label with a VLM and the
   auto-label loop" at scale: without it a person runs assist cluster by cluster. Highest
   data-integrity risk of the milestone (it writes labels to many items concurrently with ingest
   and the refresh worker), so Opus writes its plan doc and reviews the lock and concurrency code.
   It reads clusters through a small partition-reader interface so the versioned partitions of
   #343 can replace the current `cluster_id` fields later without touching the driver. Keep.
9. **#84 car-wheel teacher-student example (W4).** The public twin of the winter run: export
   validated boxes, train, bake off, promote, then use the promoted detector. Running it first on
   public data shakes out the train and promote path at a few thousand images before set P depends
   on it. Keep.
10. **#335 self-seeding screenshot capture (W5).** Independent, cheap, no GPU beyond a stub VLM.
    Keep; good Haiku/Sonnet work.
11. **#41 slim container images (W5).** Depends on a settled dependency set, which is true once W4
    lands (no resident-frame work in v0.7.0). Pulls of a 21.4 GB image slow installs and upgrades.
    Not on the winter critical path: may slip to v0.8.0 without harm. Keep, last.
12. **#42 12 GB engine builds.** Not needed for the winter run (A6000s) and its live verification
    needs GPU 1, which is shared. **Re-milestone to v0.8.0** with the `standard`/3080 Ti sweeps
    from #268.
13. **#220 faces and OCR on resident crops.** Not on the winter path; depends on the resident-frame
    work. **Re-milestone to v0.8.0.**
14. **#67 request-driven processing plans.** Depends on #40 and #220; a large API design. The
    shipped embedding modes and reprocess scopes cover the winter run. **Re-milestone to v0.8.0.**
15. **#231 ortloom-serve head-to-head.** Owner rule: after the Triton hardening; arm (c) needs a
    generic tensor passthrough built in the ortloom repository. **Re-milestone to v0.8.0 or
    Backlog.**

Not built here, accounted for: #342 (Leiden prototype) and #347 consume #341's JSONL partition
input and JSON output; #343 needs the loop's partition-reader seam and the lock rule tests W3 adds;
#344 needs the 100k and rehearsal numbers W1/W4 record; #345 needs `docker-compose.gpu-clustering.yml`
untouched. No clustering method is rewritten in v0.7.0 (owner decision): W1 changes plumbing, and
any numerics change in a method (for example FAISS GPU IVF fit) passes the agreement gate.

### 1.3 Re-milestone summary for the owner

| Issue | Recommendation |
|---|---|
| #220, #67, #42 | move to v0.8.0 |
| #231 | move to v0.8.0 or Backlog |
| #40 | split: resident-frame task to a new v0.8.0 issue; hi-res acceptance stays |
| #45 | split: 1M row to a new v0.8.0 issue unless O7 grants the run |
| #268 | split: CUDA graphs/priority_levels to #40 successor; standard-profile and 3080 Ti sweeps to #42 |
| #337 | re-scope acceptance to ms/crop (linearity already met) |

## 2. Waves

### 2.0 Test tiers (every wave)

| Tier | What | Cost |
|---|---|---|
| T0 | unit tests of the touched modules, `.venv/bin/python -c "import src.main"`, pre-commit | minutes, no GPU |
| T1 | live write-path harness: `docker compose -p op-live-verify -f docker/test/compose.yml up -d --wait`, `.venv/bin/python -m pytest tests/live -q --no-cov -m live`, then `down -v --remove-orphans` | 15 min, no GPU (fake VLM, fake trainer) |
| T2 | rehearsal-lite (section 3.2) on an isolated stack, one A6000 | 40 min GPU |
| T3 | the wave's headline benchmark, 3 repetitions, plus one parity capture if numerics moved | 1 to 3 h GPU, owner window |
| Full | offline suite once, at the end of the wave | 25 to 35 min CPU |

### W0: evaluation harness, category pins, rehearsal driver, IO sampling

Branch `feat/v070-eval-harness-and-pins`. Issues #341, #45 (part a), #337 (sampling items),
set B re-pin from #268. Tier: Opus defines the evaluation set composition (step 1 only); Sonnet
implements; Haiku runs the checks.

1. **Evaluation set** (`scripts/bench/cluster_eval_set.py`, manifest
   `scripts/datasets/manifests/cluster_eval_3000.json`): 3,000 public items, seed 20261028, built
   from COCO val/train annotations and Open Images labels as independent ground truth (the labels
   are not produced by our detector or VLM). Strata, each at least 10 %: visually similar distinct
   classes (car/truck/bus/train, bicycle/motorcycle, cat/dog), rare classes (fewest COCO
   instances), low resolution (box area under 32x32 px), near-duplicates (seeded re-encode and
   crop jitter of 100 sources, flagged as such), mixed scenes. It builds a throw-away project
   through the product's ingest (policy `all`) so vectors are the production ones.
2. **Metrics** (`scripts/bench/cluster_eval_metrics.py`, pure numpy): pairwise precision and recall,
   false-merge rate and false-split rate reported separately, correct-split recovery, unassigned
   fraction, stability across seeds (ARI; reuse `scripts/bench/parity_metrics.py` helpers, do not
   copy them), runtime and peak memory, and the count of human-validated, import-validated, locked
   or holdout items a candidate partition would move.
3. **CLI** (`scripts/bench/cluster_eval.py score --eval <manifest> --partition <jsonl|project>
   --out <json>`): scores the current project partition or any JSONL partition. Baseline: run the
   current IVF + AHC + residual pipeline twice with the same seed; record in a new
   `docs/research/cluster_eval_baseline.md` (IVF must be identical across the two runs; AHC is not
   run-to-run deterministic on real vectors, record its spread).
4. **Category pins** (`scripts/datasets/fetch_openimages_bench.py`): add `--category` (Open Images
   display names resolved to label ids; repeatable) and record per-image license and the matched
   classes in the manifest. Pin `vehicles` at 200,000 images (seed 20261009), committing only the
   `.sha256` like `openimages_bench_100000.sha256`. Extend `tests/test_fetch_openimages_bench.py`.
5. **Set B re-pin** (`scripts/datasets/fetch_hires_set.py`): add a polite rate limit and resume so
   the 429s of 0.6.0 do not recur; target 2,000 photos (12 to 24 MP); keep `commons_hires_500.json`
   as the gate subset.
6. **Scale driver sampling** (`scripts/bench/scale_run.py`): sample `/proc/pressure/io` next to the
   merge counters (move `pressure_io` out of `scripts/bench/os_write_ceiling.py` into a shared
   helper both import; no copy); optional `--forcemerge-between-legs`. Extend `tests/test_scale_run.py`.
7. **Rehearsal driver** (`scripts/bench/rehearsal_run.py`, pure stage logic in
   `scripts/bench/rehearsal_stages.py`, tests `tests/bench/test_rehearsal_stages.py`): runs the
   stages of section 3 through existing routes only (`POST /curation/projects/{project}/ingest/bulk`,
   `POST /curation/projects/{project}/clusters/auto_promote`,
   `POST /curation/projects/{project}/assist/runs`,
   `POST /curation/projects/{project}/pipeline/auto_label/start`,
   `POST /curation/projects/{project}/test_holdout/freeze`,
   `POST /curation/projects/{project}/export/yolo`, `POST /curation/projects/{project}/train/start`,
   `POST /curation/projects/{project}/bakeoff/run`,
   `POST /curation/projects/{project}/train/promote/{job_id}`), timing each, sampling GPU, host,
   OpenSearch and IO, and writing one JSON plus a markdown table. A stage that cannot run fails the
   run with a named reason; it is never skipped silently. W3 adds the loop stage.

Acceptance: `cluster_eval.py` emits JSON for the current partition in one command; two seeded
baseline runs give identical IVF metrics; the vehicles manifest verifies (`--verify-only`);
`scale_run.py` JSON carries IO pressure and merges per sample; rehearsal-lite (T2) completes end to
end on the 10k COCO pin with every stage timed.
Verify: T0 for `tests/bench/`, `tests/test_scale_run.py`, `tests/test_fetch_openimages_bench.py`;
T2 once; Full once. No contracts change expected (`make contracts-check` must stay clean).

### W1: cluster pass and embed pass

Branch `perf/v070-cluster-and-embed-pass`. Issues #337 (investigation and fix), #259. Tier: Sonnet
investigates and implements; Opus reviews any change to write-back, OCC or the cluster write
guard, and decides the gate if a method's numerics move.

Files: `src/services/curation/clustering/orchestrator.py` (`cluster_residuals`,
`assign_only_residuals`), `residual_fetch.py`, `residual_gate.py`, `cluster_geometry.py`,
`pit_pages.py`, `assignment_writer.py`, `cluster_write_guard.py`, `methods/ivf.py`, `methods/ivf_store.py`,
`pool_size.py`; `src/services/curation/reprocess_embed.py`, `reprocess_embed_spec.py`,
`src/utils/cpu_pool.py` (`OFFLOAD_MIN_BYTES`); if O1 says clamp: `src/services/curation/ingest_index.py`
(`crop_pil`) and the GPU-decode crop path in `src/services/curation/ingest_gpu_decode.py`.

1. **Profile before changing.** Export a 100k public crop dump
   (`scripts/bench/vectordb/export_embeddings.py` from a public Open Images project) and a set B
   project dump. Run `scripts/bench/cluster_plumbing_bench.py --geometry` and
   `scripts/bench/cluster_bench.py --source vecdb --sizes 10000,100000 --methods ivf --reps 1`
   with py-spy on the worker. Hypotheses to confirm or refute, in order: IVF `nlist` grows with the
   pool so fit is O(n·sqrt n); nearest-cluster distances in the geometry stage are O(n·k); residual
   gating (`_park_gated_residuals`) re-reads; the refresh daemon's assign-only competes during the
   pass; set P items carry larger documents (more region boxes, text) so fetch bytes per crop are
   higher (the 7.6 vs 4.2 ms gap).
2. **Fix plumbing only** (owner decision: methods unchanged). Output-identical changes keep
   `tests/curation/test_cluster_geometry_golden.py` and the stored IVF hash identical. A change
   that alters a method's numerics (FAISS GPU IVF fit through the existing `backend.py` path, for
   example) passes the agreement gate: nearest-centroid unchanged at least 0.98, purity not lower
   by more than 0.5 point, and #341 false-merge rate not higher by more than 0.5 point.
3. **Embed pass.** Move crop and whole-frame preprocess off the event loop for the embed pass
   regardless of `OFFLOAD_MIN_BYTES`, and overlap read of chunk N+1 with process of chunk N. New
   tests: `tests/curation/test_reprocess_embed_overlap.py` (order and content identical to the
   serial pass; a failure in chunk N does not drop chunk N+1); keep
   `tests/curation/test_reprocess_embed_missing.py` green.
4. **Vector mismatch (O1).** If clamp: ingest crops the clamped box on both CPU and GPU decode
   paths; one P-pe parity capture against the 0.6.0 reference (crop cosine p01 at least 0.995,
   kNN overlap@10 at least 0.95, cluster agreement). Either way record the decision in
   `docs/PERFORMANCE.md` ("Embed pass").

Acceptance: cluster pass at most 2.5 ms per crop at 100k public vectors on a quiet host (from 4.2
at scale), the set B per-crop rate within 1.3x of the public one, or the remaining step named with
its measured complexity; embed pass at least 20 img/s on set A (500 and 800 images, from 14.4 to
15.9), encoder busy above 80 %, API CPU at most 0.047 CPU-s per image, re-embedded vectors
bit-identical to the pre-change pass.
Verify: T0 (`tests/curation/test_cluster_geometry_golden.py`, `tests/test_cluster_bench.py`,
`tests/curation/test_cluster_backend_nonblocking.py`, `tests/curation/test_reprocess_embed_*.py`,
`tests/curation/test_lock_rule.py`), T1, T3 (cluster bench 3 reps at 100k; embed pass 3 reps), one
parity capture if O1 = clamp, T2, Full.

### W2: high-resolution ingest path

Branch `perf/v070-hires-path`. Issues #219, #268 (set B, two-card sweep, nvJPEG and
`gpu_image_prep` instances, build stamps), #337 (admission item, NVMe measurement), #40 (hi-res
acceptance). Tier: Sonnet implements; Opus reviews the stamp enforcement (it can refuse to load
models on existing installs) and the crop-cap gate results.

1. **Build stamps for all five groups.** Extend `scripts/lib/model_setup.sh`
   (`model_setup_stamp_stale`, `_ms_group_stamp_stale`, which today says "only the PE engine
   carries a stamp") to yolo, mobileclip, faces and ocr; record from the exporters
   (`export/yolo_trt_build.py`, `export/export_mobileclip_image_encoder.py`, `export/export_scrfd.py`,
   `export/export_face_recognition.py`, `export/export_paddleocr_det.py`,
   `export/export_paddleocr_rec.py`, via `export/trt_utils.py`). A mismatching engine is rebuilt or
   refused, never served. Tests next to `tests/installer/test_pe_uint8_build.py`.
2. **Work-crop cap.** `crop_max_side` (off by default, recommended 1024) applied where crops are cut
   for the cache and region consumers (`crop_pil` in `src/services/curation/ingest_index.py`, the
   GPU-decode crop path, `src/services/curation/crop_bytes.py`), and a VLM crop cap set from the
   active VLM's input size (VLM crop sites: `rg -n 'crop_jpeg' src/services/labeling src/services/curation`).
   Embedding inputs are untouched (PE preprocesses from the frame). With the cap off, outputs are
   byte-identical to 0.6.0 (test). Gates on set B: P-vlm agreement at least 0.97, P-sam and
   P-region box IoU within the parent plan thresholds, crop-cache bytes per image down at least 50 %.
3. **Admission and storage.** Charge the admission gate per image size when a request mixes GPU and
   CPU decode routes if `op_ingest_decode_total{route="cpu_call_failed"}` is frequent; one-factor
   sweep of `OP_INGEST_INFLIGHT_MB` on set B; one set B run with OpenSearch on local NVMe
   (`OP_OPENSEARCH_DATA`) against the NAS figure; `gpu_image_prep` instance count and nvJPEG check
   with the `gpu_decode` stage timer.
4. **Two-card sweep** (owner window, both A6000s): instance counts, queue delay, worker count, one
   factor at a time, then one confirmation; change a default only if it beats 55.08 img/s by more
   than the run-to-run spread and passes parity.

Acceptance: set B one-card ingest at least 35.0 img/s (0.6.0 figure) with the cap on, mean GPU
above 75 %; set P 20k slice at least 29 img/s with GPU above 75 % (#40 target; aggregates only,
recorded locally, published as pass/fail and ratio) or the binding stage named; every stamped
group refuses a mismatched engine; crop-cap gates pass.
Verify: T0 (`tests/installer/`, crop and ingest tests found by `rg -l 'crop_pil|crop_cache' tests`),
T1, T3 (set B 3 reps cap off and on; set P slice by the owner or with his yes), parity capture with
the cap on, T2, Full.

### W3: cluster-sampled auto-labeling loop

Branch `feat/v070-assist-loop`. Issue #286. Can start in parallel with W1/W2 (separate worktree);
merges after W1 because it reads the cluster fields W1 may re-plumb. Tier: **Opus** writes the plan
doc and reviews concurrency, lock rule and budget code; Sonnet implements backend and Cropwright;
Haiku runs the stub-VLM checks.

1. **Plan doc first** (`docs/design/assist_loop_plan.md`): stop rules (agreement passes and
   `min_agreement` with at least 3 answers, depth limit, minimum cluster size, budget exhausted),
   call and depth budgets, resume after restart, cancel, what is written and by whom, interaction
   with ingest and the refresh worker running at the same time, the partition-reader seam for #343.
2. **Driver** (`src/services/curation/assist_loop/`: `driver.py`, `budget.py`, `partition.py`,
   `record_store.py`), run as a background job like the assist job
   (`src/services/curation/assist/job.py`). Per cluster or sub-cluster: a seeded sampled assist
   (`src/services/curation/assist/runner.py`, `src/domain/assist_sampling.py`); on pass, propagate
   by calling the same code as `POST /curation/projects/{project}/assist/runs/{run_id}/apply`
   (`src/services/curation/assist/apply.py`, dry-run first, `409 scope_changed` stops that branch);
   on fail, refine (`src/services/curation/clustering/refine.py`, as
   `POST /curation/projects/{project}/clusters/refine/{cluster_id}`) and recurse. All writes go
   through the OCC helpers and `src/clients/occ_locks.py`.
3. **Routes and wire** (thin router `src/routers/curation/assist_loop.py`, models in
   `src/schemas/curation/assist_loop.py`, serialized through `src/services/curation/wire.py`): start,
   status, cancel, run record with per-level progress (calls used, items propagated, clusters left
   for review). `make contracts`.
4. **Cropwright** (`frontend/src/lib/api_assist.ts` or a new `api_assistLoop.ts`, a loop panel on
   the clusters page under `frontend/src/routes/p/[project]/clusters/`): start with a budget, watch
   progress, cancel. `cd frontend && npm run check && npm run build`, then open the page with
   `DISPLAY=:11` and view full-page screenshots at desktop and 430 px width.
5. **Rehearsal stage**: add `loop` to `scripts/bench/rehearsal_stages.py`.

Tests (`tests/curation/test_assist_loop_*.py`): lock rule per item class (locked, human-validated,
import-validated, holdout: never overwritten); budget never exceeded (randomized fake VLM answers,
many seeds); same seed gives same samples; propagated labels equal an explicit apply of the same
proposal (document diff, minus ids and timestamps); cancel and resume mid-level lose and duplicate
nothing; a concurrent ingest and refresh during the loop cause no lost OCC write; live harness test
with the fake VLM in `docker/test/compose.yml`.

Acceptance: issue criteria plus, on the #341 evaluation project with the local VLM: propagated
label precision at least 0.95 against ground truth, false propagation on the similar-class stratum
at most 3 %, at least 60 % of embedded items labeled at no more than one VLM call per 20 items
labeled (owner to confirm the numbers, O5).
Verify: T0, T1, T2 with the stub VLM then once with the real VLM on 10k, Full.

### W4: public rehearsal, wheel example, ladders

Branch `docs/v070-rehearsal-and-ladders`. Issues #84, #45 (rows), #40 (public table). Tier: Sonnet
implements fixes and docs; Opus reviews any change to export, holdout, training class mapping or
promote (class identity by name, never index).

1. **#84**: pin about 3,000 COCO car images (`scripts/datasets/fetch_coco_subset.py`, manifest
   `scripts/datasets/manifests/coco_car_3000.json`, CC BY 2.0 images, CC BY 4.0 annotations);
   extend `scripts/examples/wheel_example_live.py` through
   `POST /curation/projects/{project}/export/single_class`, train, bake off, promote; add
   `examples/region_profiles/vehicle_wheel_trained.json` (segmenter-free detector leg); compare with
   SAM 3 on a held-out subset; tutorial in `docs-site/docs/guides/wheel-example.mdx`, table in
   `docs/PERFORMANCE.md`.
2. **#45 rows**: full COCO (about 164k images, detect + embed + store + cluster, stop/resume
   check), Open Images vehicles 200k, set P 20 MP 100k repeat (aggregates only, GPU utilisation over
   the whole run), all with `scripts/bench/scale_run.py`.
3. **Public rehearsal** (section 3.1, public part) with `scripts/bench/rehearsal_run.py`; every bug
   found in export, train or promote at scale is fixed in this branch with a red-first test.
4. **1M row** only if O7 grants it (about 7.7 h ingest at 36 img/s, about 150 GB sources and
   95 GB index); otherwise record the extrapolation from measured per-image and per-crop cost.

Acceptance: wheel loop runs from a clean project with repository scripts and public data only;
`wheel` stays a name through export, train, promote and region profile; full-COCO throughput within
10 % of 29.0 img/s with zero failed images; rehearsal stage budgets of section 3 met or the overrun
named with its cause.
Verify: T0 for touched modules, T1, the runs themselves (T3-sized, owner windows), Full,
`.venv/bin/python scripts/docs/check_docs_vs_code.py`, `make docs-build`.

### W5: images and screenshots

Branch `build/v070-slim-images-and-docs`. Issues #41, #335. Tier: Sonnet; Haiku for size and
naming-scan checks.

1. **#41**: split the runtime API/worker image from the export environment (`Dockerfile`,
   `Dockerfile.triton` untouched unless needed, `docker/evaluator/Dockerfile`,
   `docker/trainer/Dockerfile`, `docker/segmenter/Dockerfile`, `scripts/release/build_and_publish.sh`),
   shared layers, digest pins, non-root user; before/after sizes in `docs/image_size_audit.md`.
2. **#335**: `scripts/docs/capture_backend_screens.py` seeds its own public project (classes, region
   profile, stored assist run, bulk run, stub VLM endpoint) on an isolated stack (own `-p`), then
   regenerates the 17 listed states.

Acceptance: runtime image at most 50 % of 21.4 GB, no image grows, `make release-dry-run` with no
un-allowlisted CRITICAL; offline and live suites pass on the slim images, export works from the
export image; all screenshot states regenerate in one run from an empty stack, re-run gives the
same file set, naming-leak scan clean.
Verify: T0, T1 on the slim images, `make release-dry-run`, `make docs-build`, Full.

## 3. The winter rehearsal

### 3.1 Full rehearsal (run after W4 merges, before the real run; repeated as the real run)

Sets: **P1** Open Images v7 vehicles, 200,000 images at most 1,024 px (about 0.7 MP), public pin
from W0; **B** 2,000 Commons photos, 12 to 24 MP, public; **P** 100,000 set P photos, median about
20 MP, owner-local. Expected crops (0.6.0 rates, policy `all`): 200k x 4.9 + 2k x 2.4 + 100k x 2.4,
about 1.23M. Stack: one A6000 for Triton (two if O2 grants), OpenSearch data on local NVMe,
refresh worker on, VLM stopped during ingest and started after.

| Stage | Budget (one A6000; two in brackets) | Measured |
|---|---|---|
| 1. Ingest P1 (bulk runner, forced stop at 50 %, resume) | at most 100 min, at least 34 img/s [60 min, 55 img/s] | img/s per quarter, crops/s, mean GPU, `stage_s`, admission wait, segments and merges, IO pressure, heap, index bytes per image, failures 0, stored + skipped + duplicates + failed = input |
| 2. Ingest B then P (cap on) | P at most 62 min at 27 img/s floor, target 57 min at 29 img/s with GPU above 75 % | same as 1, plus crop-cache bytes per image, decode route counts, host CPU per photo |
| 3. Cluster pass (after the last ingest; force-merge between) | at most 52 min (2.5 ms per crop) | wall per sub-stage (fetch, fit, write-back, geometry), ms per crop per set, peak memory, cluster count, #341 metrics on the evaluation project in the same stack |
| 4. Assist loop with the local VLM | call budget 30,000 crops, at most 60 min at about 10 crops/s | calls used, items propagated, clusters left for review, precision on P1 against Open Images labels (sampled), budget never exceeded |
| 5. Auto-label and agreement-gated auto-promote | at most 20 min | classes promoted, items changed, lock-rule violations 0 (audit query on locked/validated/holdout items before and after) |
| 6. Review (owner, Cropwright) | owner time; queue sized to at most 4 h | review-queue size per tab, items per minute, share of loop labels corrected |
| 7. Freeze test holdout | at most 5 min | holdout size per class name; holdout disjoint from train including near-duplicates |
| 8. Export (`resize_mode` aspect, `image_size` 1280) | at most 30 min | images/s, bytes, `data.yaml` class names, holdout excluded, manifest checksums |
| 9. Train YOLO26 (trainer, arbiter-assigned A6000) | rehearsal: 20 epochs on a 20k subset, at most 75 min; real run: at most 24 h | epoch time, GPU util, projected full time, mAP50-95 on the holdout per class name |
| 10. Bake-off and promote | at most 30 min | candidate vs incumbent mAP50-95 per vehicle class by name, promote gate result, Triton READY with a matching stamp, `/detect` latency |
| 11. Use the promoted detector | at most 15 min on 2,000 images | `POST /curation/projects/{project}/reprocess` scope detect on a seeded slice; box agreement with the review labels |

Machine wall time excluding training and review: about 6 h on one card, split across owner windows
(ingest one day, cluster to export the next). "Better vehicle detector" (O4, owner to confirm):
candidate mAP50-95 on the frozen holdout at least 3 points above the incumbent (the shipped default
detector restricted to the vehicle class names), no vehicle class more than 1 point worse.

### 3.2 Rehearsal-lite (T2, every wave)

10k COCO pin plus set B 500 on one A6000: ingest at most 8 min, cluster at most 3 min, assist (or
loop with the stub VLM from W3) at most 5 min, auto-label at most 3 min, holdout plus export at most
3 min, train smoke (3 epochs on 2k images) at most 10 min, bake-off plus promote at most 8 min;
total at most 40 min. Command (W0 builds it):

```bash
.venv/bin/python scripts/bench/rehearsal_run.py --api http://127.0.0.1:<isolated api port> \
  --project rlite-<date> --manifest <10k manifest> --extra-manifest scripts/datasets/manifests/commons_hires_500.json \
  --stages ingest,cluster,assist,autolabel,holdout,export,train,promote \
  --container-prefix <-p name> --out-dir artifacts_local/rehearsal/<label>
```

## 4. Model tiers per task

| Task | Haiku (lookups, checks) | Sonnet (implementation, review) | Opus (design, risky changes) |
|---|---|---|---|
| W0 | manifest verify, `rg` path checks, run T0/Full and report | metrics, CLI, fetch flags, driver, sampling | evaluation-set strata (step 1) |
| W1 | run benches and collect JSON | profiling, plumbing fixes, embed overlap, review | write-back/OCC review; gate call if a method's numerics move; O1 parity verdict |
| W2 | size and byte accounting, stamp report | stamps, crop cap, admission, sweeps | stamp-enforcement review (can block READY); crop-cap gate verdict |
| W3 | stub-VLM runs, contract check | driver, routes, Cropwright panel, tests | plan doc, lock/budget/concurrency design and review |
| W4 | data pins, doc link and route checks | wheel loop, ladder runs, bug fixes, docs | review of export/holdout/train class-mapping/promote fixes |
| W5 | image sizes, naming scan, Trivy summary | Dockerfile split, capture seeding | none |
| Rehearsal | stage timing extraction | run and triage | go/no-go call on budget overruns |

## 5. Risks

1. **Cluster pass dominates at 1.2M crops.** If W1 misses 2.5 ms/crop, stage 3 takes 1.5 to 2.6 h
   and blocks the loop. Mitigation: agreement-gated GPU fit (O6); assign-only for set P after a
   full pass on P1.
2. **Wrong labels at scale.** A false merge propagates one wrong class to a whole cluster. Mitigation:
   #341 gates, `min_agreement`, dry-run apply, refine-and-recurse, loop labels never overwrite
   locked or validated items, review sample before export (O3).
3. **Numerics change after the big ingest.** The clamp decision (O1) or a crop-cap change after the
   run forces a re-embed of 1.2M crops. Mitigation: O1 decided before W1; crop cap does not touch
   embeddings.
4. **Shared host contamination.** GPU 0 and the host are shared; numbers drift. Mitigation: load
   guard, `nvidia-smi dmon`, foreign-SM sampling, repeats only on the headline row.
5. **Set P leak.** Mitigation: `artifacts_local/` only, aggregates in docs, naming scan, no
   screenshots of set P projects (W5 seeds its own public project).
6. **Stamp enforcement refuses engines on existing installs.** Mitigation: rebuild path in the
   installer and a release-note line, as for the PE stamp in 0.6.0.
7. **Holdout leakage.** Near-duplicates across holdout and train inflate the promote gate. Mitigation:
   group by perceptual-hash duplicate cluster when freezing (verify the current freeze does; fix in
   W4 if not).
8. **Storage.** the data volume had 471 GB free on 2026-10-28; the full rehearsal needs about 30 GB of
   index plus the 20 MP export at 1280 px (about 50 GB); the 1M row needs about 250 GB more.
9. **Training competes with Triton and the VLM.** Mitigation: GPU arbiter overlay
   (`docker-compose.gpu-arbiter.yml`), training in its own window.

## 6. Open questions for the owner

| ID | Question | Recommendation |
|---|---|---|
| O1 | #259: make ingest embed the clamped box (one-time numerics change, parity-gated) or document the mismatch as accepted? | clamp, before W1 merges, so the winter run embeds once in the final space |
| O2 | GPU windows: when are GPU 0 and GPU 2 free together for the two-card sweep (W2) and the rehearsal ingest (about 3 h)? | two windows of 3 h each |
| O3 | Which label states may enter the training export: human-validated only, or also agreement-gated auto-promoted and loop-propagated labels? | validated plus agreement-gated, with a reviewed sample per class |
| O4 | "Better vehicle detector": which class names (COCO vehicle names only, or finer VLM classes), margin (3 points mAP50-95?), image size (640 or 1280 for 20 MP photos), holdout mix (public plus set P)? | COCO vehicle names first, 1280, +3 points, mixed holdout |
| O5 | Loop acceptance numbers in W3 (precision 0.95, 60 % coverage, 1 call per 20 items): right targets? Which VLM model from the catalog, local or remote? | confirm, local catalog model on GPU 0 |
| O6 | Still allowed: FAISS GPU IVF fit behind the agreement gate if W1 shows fit dominates? | yes (0.6.0 decision O2 stands) |
| O7 | Run the 1M Open Images row in v0.7.0 (about 8 h ingest plus about 3.5 h cluster, about 250 GB)? | no; record the extrapolation, move the run to v0.8.0 |
| O8 | Approve the re-milestoning and splits of section 1.3? | yes |
| O9 | Should #41 stay in v0.7.0 even if it delays the release, or slip? | slip if W4 runs late |
