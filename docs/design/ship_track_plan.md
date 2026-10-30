# Ship track: ingest speed, OpenSearch, cluster outputs and Cropwright to v0.7.0

Status: plan (2026-10-29, written against `main` at `0e2d3fe7`, after PR #386). Nothing here is
implemented. Owner: program coordinator. Audience: a fresh agent with no conversation memory that
takes one wave below and executes it standalone.

This plan narrows Phase A of [`program_plan.md`](program_plan.md) to what ships 0.7.0 and makes
measurable progress now on five areas: ingest speed, processing passes, the OpenSearch data path,
cluster outputs, and Cropwright showing those outputs with filtering and labeling so clusters get
better. It reuses the program plan's binding rules (section 0), definition of done (0.1), model
tiers (0.2) and audit gate (2.2), and does not restate them. Where the two disagree, this file wins
for v0.7.0 work (section 2 lists what changed).

Clustering research is **paused**. The only research wave this plan schedules is the registered
follow-up **Z5** (section 7.3) and, if it passes, its integration into the auto-label loop.

Read first, in order: `CLAUDE.md`, `frontend/CLAUDE.md`, [`../VISION_AND_GOALS.md`](../VISION_AND_GOALS.md),
[`program_plan.md`](program_plan.md) sections 0 to 2, then the section of this file for your wave.

---

## 0. How to run a wave from this file

1. Find your wave in section 4 (the issue map) and its detail section (5 to 9).
2. Create or enter the lane worktree named in section 10 (one agent per worktree at a time).
3. Run the **preflight** (10.3), write the failing tests named for the wave, watch them fail, then
   implement.
4. Verify with the wave's commands; record numbers with the protocol of section 3.
5. Write the owner review package (section 12) and commit it with the wave's last commit.
6. Do not push, open a PR, close an issue, tag or touch GPU 2 without the owner's yes (section 11).

Shell names used below (defined here, used everywhere):

```bash
MAIN=$(git worktree list | head -n1 | cut -d" " -f1)   # the main checkout; the venv lives here only
VPY=$MAIN/.venv/bin/python
VPC=$MAIN/.venv/bin/pre-commit
SCRATCH=$(dirname "$MAIN")/../scratch            # host scratch; locks in $SCRATCH/locks/gpu0.lock (confirm the path in the session handoff)
# every command below runs from the lane worktree root, never from $MAIN
$VPY -m pytest <touched tests> -q --no-cov && $VPY -c 'import src.main' && $VPC run --all-files
```

Frontend checks (from `frontend/` of the worktree): `npm ci`, `npm run check`, `npm run lint`,
`npx vitest run <files>`, `npx knip`, `npm run build`, `npm run test:e2e`.

---

## 1. Current state per area (evidence on `main` at `0e2d3fe7` unless a branch is named)

### 1.1 Ingest

| Fact | Evidence |
|---|---|
| Photos of at least 3 MP take the GPU route: `gpu_image_prep` decodes once (nvJPEG), runs the detector, embeds frame and crops with PE, computes blur sums and encodes crop JPEGs in one Triton call | `src/services/curation/ingest_gpu_decode.py` docstring; `models/gpu_image_prep/1/model.py`, `prep_geometry.py`, `prep_kernels.py`, `nvjpeg_lanes.py` |
| Smaller photos (all of COCO) take the CPU route: decode in the API, batched detector, PE crops | `src/services/curation/ingest_batch.py` `_prefill_detections` |
| Two `gpu_image_prep` instances on the `full` profile: 31.8 to 48.0 img/s on 3,000 set P photos, GPU 78 to 87 percent; a third adds 0.6 percent | `v070_measurement_log.md` E4; `models/gpu_image_prep/config.pbtxt` (`count: 2`) |
| Throughput saturates at 37 to 39 img/s (one instance) from concurrency 8; batch and concurrency are not levers | log section B |
| The API stage timers (`op_pipeline_stage_seconds`, 17 stages) see the whole GPU route as one `gpu_decode` stage; **nothing times the stages inside `gpu_image_prep`** (decode, detect, embed, crop JPEG, blur) | `src/utils/stage_timing.py` `STAGES`; `rg perf_counter models/gpu_image_prep` finds nothing |
| Set P with 2.35 crops per photo ingests at 28 img/s against 37 at 1.5 crops per photo (one instance): crop work scales the cost | log section C |
| Crops are cut and JPEG-encoded at source resolution (#219 not started) | `src/services/curation/ingest_index.py` (`crop` stage), the crop-JPEG path in `gpu_image_prep` |
| `scripts/bench/run_baseline.py` already records `op_*` deltas, Triton metrics deltas and GPU utilisation (`--gpu-interval`, default 1.0 s); it has no repetition loop, no page-cache drop and no load-average record | `scripts/bench/run_baseline.py` argparse |
| `scripts/bench/scale_run.py` samples every 30 s and keeps peaks only; no query-latency probe | `scripts/bench/scale_run.py` |
| 5 s GPU samples under-read a pass by about 14 points; 500 ms is the rule | log G6 finding 2 |

### 1.2 Processing passes (embed pass, worker region stage)

| Fact | Evidence |
|---|---|
| Embed pass on the GPU route with two chunks in flight: 51 img/s on set P, GPU 85 to 92 percent, API 0.025 CPU-s per image (#259 closed) | log G3, G5; `src/services/curation/reprocess_images.py` `EMBED_CHUNKS_IN_FLIGHT = 2` |
| COCO (CPU route) embed pass: 16.2 to 16.7 img/s, GPU 42 to 54 percent; the chunk window applies but was **not measured** | log E1, F3, G5 last paragraph |
| Worker region stage decodes each source once per flush (#362 closed); the post-fix cost against v0.6.0 (86 ms, 0.077 CPU-s per item) is **not in the log** | CHANGELOG `[Unreleased]` Fixed; log G2 is the pre-fix 220 ms |

### 1.3 OpenSearch (databasing)

| Fact | Evidence |
|---|---|
| k-NN p99 6.8 to 12.2 ms from 10k to 1M (Lucene HNSW, filtered and unfiltered) | log "k-NN query latency" |
| Text search: per-worker index bootstrap moved to startup (#356, `6eb0b54c`), FastAPI lazy route state warmed at startup (`b77da6e3`); the p99 under 300 ms at 100k bar is **not yet re-measured** after both | log E3 (measured before the route warm-up); `git log --grep '#356'` |
| During ingest the text-search tail grows to p99 350 to 460 ms | log D |
| Filtered curation search is 4 to 5 times slower than unfiltered (#384, milestone v0.8.0) | `docs/research/retrieval_baseline.md` "Curation filter and pagination behaviour" |
| `GET /clusters` (curation) answers 134 to 238 ms at 91k items (0.6.0 install); its response is **untyped** (`dict[str, Any]`, OpenAPI `additionalProperties: true`) and sits on the `LEGACY_DICT` allowlist of the typed-response ratchet with `POST /crops/label/undo_batch` | #356 body; `src/routers/curation/clusters.py` `list_clusters`; `contracts/openapi/curation.json`; `tests/test_contract_typed_responses.py` |
| `POST /crops/label/undo_batch` accepts at most 5,000 ids (`CropUndoBatchRequest`) while `PUT /crops/batch_label` takes a selection of up to 20,000 (`MAX_BULK_SELECTION`), and it runs one unbounded `asyncio.gather` over all ids | `src/routers/curation/_common_models.py`, `src/routers/curation/_selection.py`, `src/routers/curation/label_undo.py` `undo_crop_labels` |

### 1.4 Clustering outputs

| Fact | Evidence |
|---|---|
| Cluster pass is linear except the IVF k-means fit below the 50,000 training cap; 91,184 set P crops: 231 s (fetch 17, fit 182, write-back 30), 2.54 ms per crop; 1M synthetic: 596 s | log "Cluster pass wall time" |
| The fit runs `niter=50, nredo=5` (five restarts) on CPU FAISS | `src/services/curation/clustering/methods/ivf.py` `IVF_DEFAULT_NREDO = 5` |
| The comment "50k >> 256*512" in `ivf.py` is arithmetically wrong (256 x 512 = 131,072) | `methods/ivf.py` above `IVF_MAX_TRAIN_SAMPLE` |
| Geometry stage and the refresh daemon's assign-only pass are **not timed** at 100k; the #337 bar (2.5 ms per crop) includes them | #337 acceptance; `cluster_geometry.py` `cluster_geometry_stage`, `orchestrator.py` `assign_only_residuals` |
| A partial pass reports `partial` (#357 part 1 merged); a daemon pass without an expected pool still cannot see a pool that shrank before it started, so #357 stays open | CHANGELOG Fixed; `src/services/curation/clustering/pool_coverage.py` |
| Cluster cards serve size, validated count, dominant class, geometry purity, label agreement, sub-cluster ids, promotable; they do **not** serve detector agreement, detector confidence, review counts or quality flags, and they cannot be sorted or filtered by the server | `list_clusters` aggregation body |
| `label_purity` and `label_agreement` are the same value under two wire names (a duplicate the no-dead-code rule forbids) | `list_clusters` card dict |
| Item fields available for those outputs: `detector_class_name`, `detector_class_id`, `detector_confidence` (written once at ingest), `class_validated`, `class_source`, `cluster_nearest_id`, `cluster_distance_cluster_id`, `mistakenness_score` | `src/clients/curation_opensearch/items_extra.py` `DETECTOR_MAPPING`, `bodies_core.py` |
| Review tab queries live in one place, `SIMPLE_TABS` in `src/services/curation/review_tab_clauses.py` (`all`, `mismatches`, `vlm_low_conf`, `detector_disagreements`, `outliers`, ...) | that module |
| Class-level clustering research concluded: one gate per detector key roughly matches clustering; Z4 (detector-key roots plus the loop's recursion, with and without a hold-back guard) is INCONCLUSIVE on the open-set bar because the set had 395 held-out open-set crops; key roots cost 0.293 and key + guard 0.377 decisions per item against IVF's 0.834 (about 2.2 times fewer with the guard) at U2 0.982 and 0.988 | `docs/research/cluster_config_decision.md` section 0A and `docs/research/cluster_config_z4.md` on branch `feat/cluster-z4-experiment` (`117cb99f`) |

### 1.5 Cropwright

| Fact | Evidence |
|---|---|
| Health pass merged (axe contrast and names fixed, 9 defects fixed); left: #380 nested interactive card, #381 four small inconsistencies, #382 `/projects` combine probe 404 | [`../research/cropwright_health_2026-10-29.md`](../research/cropwright_health_2026-10-29.md) |
| #382 work is **in progress, uncommitted** in worktree `OpenProcessor-wt-ui` (branch `feat/v070-cluster-ui`): deletes `frontend/src/lib/combine/combineAvailability.svelte.ts` and edits `/projects` and combine pages | `git -C $MAIN/../OpenProcessor-wt-ui status` |
| The cluster grid **sorts on the client** (`sortClusters` in `frontend/src/lib/clusters/clusterListController.svelte.ts`) and then fetches representatives **one request per card** in display order (`displayOrderRepresentatives.ts`, `loadMoreRepresentatives`), because the server pages representatives in its own size order. Both break the "backend serves decisions" rule | those files |
| Sort options today: `purity_asc` (default), `purity_desc`, `size_desc`, `size_asc`, `dominant_class`; filter: "unlabeled only" (client side) | `frontend/src/lib/types.ts` `ClusterFilter`; controller |
| In-cluster filters: label source, primary subject, blur, semantic search, StrategyBar | `frontend/src/lib/components/clusters/ClusterDetailFilters.svelte` |
| Bulk assign exists (`bulkLabel`, `bulkLabelSelection` with served dry run) and per-session undo through `undo_batch` with client-held ids; undo does not survive a reload | `frontend/src/lib/api_cropWrites.ts`, `api_select.ts`, `clusterController.svelte.ts` |
| Keyboard: cluster actions exist (`cluster.confirm`, `cluster.undo`, `cluster.move`, ...); there is no next/previous **cluster** action | `frontend/src/lib/keymapFallback.ts` |
| No axe check is committed; the health pass injected axe-core by hand | `rg -l axe frontend/e2e frontend/package.json` |
| Playwright `full_page` captures only the viewport because the shell scrolls inside `<main class="min-h-0 flex-1 overflow-auto">` | health note "Environment notes"; `frontend/src/routes/p/[project]/+layout.svelte` |

### 1.6 Auto-label loop (#286)

| Fact | Evidence |
|---|---|
| S1 to S8 committed on `feat/v070-assist-loop` (worktree `OpenProcessor-wt-loop`, head `0259a440`, main merged in): stats, tree, propagated kind, shared writer, `ask_sample`, partition and splitter, job store and budgets, evaluate, propagate, driver, audit, rollback, worker, routes, docs | `git log origin/main..feat/v070-assist-loop` |
| One uncommitted edit in that worktree (`tests/curation/test_propagated_label_kind.py`, 2 lines) | `git -C $MAIN/../OpenProcessor-wt-loop status` |
| Open: S0 (`scripts/bench/vlm_rate.py` absent), S9 (no frontend files), S10 (`docker/test/fake_vlm.py` has no `class_cycle`; no `tests/live/test_live_assist_loop.py`), S11 (no T2 numbers) | loop branch tree |
| The loop reads roots through `PartitionReader` (`ClusterFieldPartition` only): the seam Z5 integrates into | `src/services/curation/assist_loop/partition.py` on the loop branch |

### 1.7 Installer, docs, tests

Fixed on `main` but the issue is still open (verify and close with the owner's yes): #335
(`d3bac95d`, 17 seeded states), #373 (`9de79579`), #377 (`e76c3574`), #378 (`b443a66e`). Not
started: #363, #371 (`OP_SEMANTIC_SEARCH_ENABLED` still default off in `src/routers/curation/search.py`),
#355, #383 (`tests/test_ingest_file_read.py::test_slow_reads_overlap_and_leave_the_loop_free`,
`tests/test_ingest_offload.py::test_batch_decode_runs_off_the_loop_thread`), #154
(`docker/trainer/mlflow_callbacks.py` calls `transition_model_version_stage`), #84, #41.

---

## 2. Where `program_plan.md` is out of date

| Program plan says | Now |
|---|---|
| Section 1: work lives on `release/0.7.0`; `fix/v070-362-work`, `fix/v070-357-work`, `chore/v070-followups` in progress | All of it reached `main` through PR #386 and earlier merges. #362 closed; #357 part 1 merged; #341, #372, #374, #375, #376 closed |
| Section 1: loop at S6a | S1 to S8 done; S0, S9, S10, S11 open |
| Matrix row #349 (unattended bind default 127.0.0.1) and O-A2 | Owner decided: Cropwright keeps the LAN bind, #349 closed; basic auth is #351 (v0.8.0) |
| O-A1 open | Decided: semantic search on by default (#371) |
| A2.2: reproduce the 250k `update_by_query` failure | The bench reset was fixed (task API, fails unless none left). What remains of #357 is the daemon pass without an expected pool (5.2) |
| A4.3: #40 hi-res row still to reach 29 img/s | E4 already measured 48 img/s with GPU 78 to 87 percent on 3,000 set P photos; only the formal 20k row and the close-out remain (5.5) |
| A5.1 and A5.4 assume the W0 Open Images category pins and `rehearsal_run.py` | Neither exists (`fetch_openimages_bench.py` has no `--category`; no `scripts/bench/rehearsal_run.py`); this plan builds the pins in SP5 and leaves the rehearsal driver to an owner decision (O-S4) |
| Phase C (Leiden, GPU clustering, partitions) runs in parallel from week 3 | Class-level research concluded no new method; research paused except Z5; C-phase issues stay in v0.8.0 untouched by this plan |
| #154 in F1 (v0.8.0) | Milestone says v0.7.0; it is small and goes in wave D1 |
| No rows for #380 to #383 | Added here (UI0, D1) |
| `docs/design/README.md` rows: `auto_label_loop_plan.md` "nothing implemented", `v070_wave_plan.md` "waves not started" | Both stale; updated when the loop branch merges (L1) and at release (R1) |

---

## 3. Measurement protocol (every number in this plan)

Applies to ingest, pass, query and UI-latency numbers. `docs/research/CLAUDE.md` rules hold.

1. **Data.** Public: set A (COCO 2,000 pin `coco_bench_2000`), set B gate subset
   (`scripts/datasets/manifests/commons_hires_500.json`), COCO full, Open Images pins. Private:
   **set P**, copied (never moved, never hard-linked from the NAS) to NVMe as seeded samples of
   3,000 (about 28 GB) and 20,000 (about 190 GB, owner yes, one at a time with any other big
   sample). Set P is reported as **numbers only**: no paths, folder or file names, thumbnails.
2. **Stack.** Isolated: own compose project name and port base, Triton and API on GPU 0 only
   (`TRITON_GPU_ID=0`, `API_GPU_ID=0` in the stack's own env), every GPU job under the GPU 0 lock
   (section 10.2). OpenSearch data on local NVMe. Note the commit, image tags, driver, Triton and
   TensorRT versions in the JSON.
3. **Per run.** Fresh project; input page cache dropped first (`posix_fadvise(DONTNEED)` over the
   input files, as the log's campaign did); crop cache cap off when counting files
   (`OP_CROP_CACHE_MAX_BYTES=0`); GPU sampled every **500 ms** (`nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv -lms 500`);
   Triton CPU from `docker stats`; `/proc/loadavg` at start and end; `op_pipeline_stage_seconds`
   and Triton per-model `nv_inference_queue_duration_us` / `nv_inference_compute_infer_duration_us`
   deltas; the new `gpu_image_prep` stage deltas (SP0); img/s = images / (finished_at - started_at).
4. **Repetitions.** 3 per headline row, median and range; one factor changed at a time. A run whose
   1-minute load average exceeds 16 is marked noisy and repeated before it decides anything.
5. **Per-stage report.** One table per row: wall share of each API stage, each Triton model's queue
   and compute time, each `gpu_image_prep` internal stage, GPU utilisation mean and p50, Triton
   cores, API CPU-s per image, crops per image. Stage seconds are sums over concurrent work, so they
   are compared across runs, never added up to wall time.
6. **Raw data.** Raw JSON per repetition under `$MAIN/artifacts_local/bench/v070/<wave>/<row>/`
   (gitignored; in the main checkout so it survives worktree removal). Small public result files
   that back a published number go to `docs/benchmarks/v070/<wave>/` with `MANIFEST.sha256` updated.
   Every note section carries its reproduce command and raw-data path.
7. **Where numbers go.** Public rows: `docs/PERFORMANCE.md`. Set P: a section of
   [`../research/v070_measurement_log.md`](../research/v070_measurement_log.md) named after the
   wave (aggregates, ratios and pass/fail only).
8. **Never** invent, round up or carry forward a number; a bar not met is reported as not met with
   the binding stage named.

---

## 4. Issue to wave map (acceptance bars are the merge gate)

| Issue | Wave | Files (main ones) | Tests (red first) | Measurable acceptance |
|---|---|---|---|---|
| #40 epic (v0.7.0 part) | SP0, SP4 | `models/gpu_image_prep/1/model.py`, `scripts/bench/run_baseline.py` | `tests/models/test_gpu_image_prep_stage_metrics.py` (new); `tests/bench/test_run_baseline_reps.py` (new) | Per-stage table for set A, set B, set P 3k (3 reps); set P 20k one card at least 29 img/s, GPU above 75 percent (500 ms); then close #40 at v0.7.0 and file R1 for v0.8.0 (owner yes) |
| #337 | SP1 | `clustering/orchestrator.py`, `cluster_geometry.py`, `methods/ivf.py`, `scripts/bench/cluster_ladder.py` | `tests/curation/test_cluster_geometry_golden.py` unchanged; new `tests/test_cluster_ladder_stages.py` | Geometry and assign-only timed at 100k; 0.6.0 2,042 s explained (stage named); whole pass at most 2.5 ms per crop at 100k high-res including geometry and assign-only |
| #357 | SP1 | `clustering/pool_coverage.py`, `orchestrator.py`, `scripts/curation/cluster_refresh_daemon.py` | `tests/curation/test_cluster_pass_partial_daemon.py` (new) | A daemon pass over a pool that shrank under it cannot report `success`; issue closable |
| #362/#259 follow-ups | SP2 | `src/services/curation/reprocess_images.py`, worker region stage | none new unless a fix lands | Worker region stage at most 1.2x v0.6.0 per item (86 ms, 0.077 CPU-s) recorded; COCO embed pass with the chunk window measured (3 reps) |
| #219 | SP3 | `ingest_index.py`, `crop_bytes.py`, `models/gpu_image_prep/1/prep_geometry.py`, `ingest_policy.py`, VLM crop sites | `tests/curation/test_crop_max_side.py` (new): off is byte-identical, on caps the long side | Set B crop-cache bytes per image at least 50 percent lower with the cap on; VLM agreement at least 0.97; set P img/s change recorded |
| #45 | SP5 | `scripts/datasets/fetch_openimages_bench.py` (`--category`), `scripts/bench/scale_run.py` | `tests/test_fetch_openimages_bench.py`, `tests/test_scale_run.py` | Full COCO within 10 percent of 29.0 img/s, 0 failures; 20 MP 100k row with mean GPU; R3 limits table (100k, 500k, 1M synthetic); 1M real row stays v0.8.0 |
| #356 | SP6 | `scripts/bench/scale_run.py` latency probe | `tests/test_scale_run.py` probe test | Text search p99 under 300 ms at 100k idle after an API restart (300 distinct phrases, new connection per request); p99 during ingest recorded |
| (none: cluster outputs) | CQ1, CQ2, CQ3 | section 6 | section 6 | Section 6 bars; `make contracts` committed |
| #380, #381, #382 | UI0 | section 8.1 | vitest per item, axe e2e | axe clean on touched pages; no console error on `/projects`; screenshots viewed |
| (cluster list, filters, bulk, keys) | UI1 to UI4 | section 8 | section 8 | Section 8 bars |
| #286 | L1, UI5, L2, Z5-I | section 7 | section 7 | Loop acceptance of the loop plan section 12; Z5 rule |
| #363, #371, #355, #383, #154 | D1 | section 9 | section 9 | Section 9 bars |
| #335, #373, #377, #378 | D1 (verify only) | none | re-run the fix's test | Owner closes with the merge commit linked |
| #84 | EX | `scripts/examples/wheel_example_live.py`, `examples/region_profiles/` | per program plan A5.2 | Program plan A5.2 bar, or moved to v0.8.0 by the owner (O-S5) |
| #41 | R1 (last) | Dockerfiles, `scripts/release/build_and_publish.sh` | T1 on slim images | Runtime image at most 50 percent of 21.4 GB; no image grows; Trivy clean |

---

## 5. Ingest speed lane (SP), branch `perf/v070-ingest-speed`

Worktree `$MAIN/../OpenProcessor-wt-sp` (exists, at `main`). One agent at a time. Every GPU
step runs under the GPU 0 lock. Order: SP0, SP1, SP2, SP3, SP4, SP5 with SP6 inside SP5's runs.
Expected gains below are estimates from the cited numbers, not results.

### SP0: baseline protocol and stage profile (first; nothing is optimized before it reads)

Tier: Sonnet implements and runs; Haiku collects JSON; Opus reviews nothing here (metrics only).

Code:

1. `models/gpu_image_prep/1/model.py`: per-stage wall timing inside the model through the Python
   backend's custom metrics (`pb_utils.MetricFamily`, histogram or counter pair named
   `op_gpu_prep_stage_seconds` with a `stage` label: `decode`, `letterbox`, `detect`,
   `embed_frame`, `embed_crops`, `crop_jpeg`, `blur`), exported on Triton's metrics port. Outputs
   must not change: next to `tests/models/test_gpu_image_prep_geometry.py`, add
   `tests/models/test_gpu_image_prep_stage_metrics.py` with a test that the model's outputs are
   identical with metrics collected, and one that every named stage is observed once per image.
2. `scripts/bench/run_baseline.py`: `--reps N` (fresh project per rep), `--drop-cache` (fadvise
   over the manifest), load average start and end in the JSON, a `gpu_prep_stages` block from the
   Triton metrics delta; default `--gpu-interval` stays, the protocol passes `0.5`. Test:
   `tests/bench/test_run_baseline_reps.py` (rep loop writes N files with distinct slugs; cache drop
   called once per rep; loadavg recorded).
3. `scripts/bench/scale_run.py`: a query-latency probe used by SP6 (`--latency-probe text,knn,clusters`,
   every N seconds and once after the ingest: p50/p95/p99 of `GET /search/text`, a stored-vector
   k-NN and `GET /clusters`), and 500 ms GPU sampling into the time series. Extend
   `tests/test_scale_run.py`.

Runs (3 reps each, protocol of section 3): set A COCO 2,000 (CPU route), set B 500 (GPU route,
public), set P 3,000 NVMe sample (GPU route). Output: the per-stage table per row in a new log
section "SP0: stage profile", public rows in `docs/PERFORMANCE.md` under a new heading
"0.7.0 stage profile".

Acceptance: three tables exist with every stage named and its share; the stage that bounds each
route is named (expected from the log: on the GPU route the work inside `gpu_image_prep`, with
`crop_jpeg` growing with crops per image; on the CPU route the PE crop embed). This ranks SP1 to
SP3 by measured share; if the ranking contradicts the order below, the owner review package says so
and the order follows the data.

### SP1: #337 cluster pass and #357 close-out

Tier: Sonnet; **Opus reviews** any change that alters the fit's numerics and the #357 change
(DR-A2 of the program plan, abbreviated: a written verdict appended to this section's review file).

1. **Measure the missing stages** (`scripts/bench/cluster_ladder.py`: `--with-geometry` runs
   `cluster_geometry_stage` after the pass; `--assign-only` times `assign_only_residuals` on a 5,000
   item top-up). Rows: 91k set P export and 100k public crops (Open Images 100k pin), 3 reps.
2. **Explain 0.6.0's 2,042 s** at 491k: rerun the pass during a sustained ingest with merge
   pressure (`scale_run.py` with IO pressure and merge counters, `_forcemerge` between legs), name
   the stage that grows. Measured so far: a concurrent 5k ingest adds about 8 percent.
3. **Fix, one factor at a time**, each behind the agreement gate of `v070_wave_plan.md` W1 step 2
   (pair agreement with the incumbent partition, purity not lower by more than 0.5 point, #341
   false-merge rate not higher by more than 0.5 point, scored with `scripts/bench/cluster_eval.py`):
   - (a) `nredo` 5 to 1 in `methods/ivf.py`. Expected: fit about 5 times cheaper (182 s to about
     40 s at 91k), wall about 90 s, about 1.0 ms per crop. Risk: medium (numerics move; the gate
     decides).
   - (b) only if (a) fails the gate: FAISS GPU fit through the existing `backend.py` detection,
     GPU 0 under the lock. Expected: fit under 10 s. Risk: medium (VRAM next to Triton; fall back
     to CPU when VRAM is short is already the rule).
   - Correct the `IVF_MAX_TRAIN_SAMPLE` comment arithmetic in the same commit (it belongs to #355
     but lives in this lane's file).
4. **#357 remainder**: a pass started by the refresh daemon records the expected pool size at
   start (count query) and reports `partial` when it read under 99 percent of it; red-first test
   `tests/curation/test_cluster_pass_partial_daemon.py` with the query-capable fake.

Acceptance: whole pass at most 2.5 ms per crop at 100k high-res crops including geometry and
assign-only, 3 reps; or the remaining stage named with its measured complexity. Log section K
(program plan name). Verify: T0 with `tests/curation/test_cluster_geometry_golden.py`,
`tests/test_cluster_bench.py`, `tests/curation/test_cluster_backend_nonblocking.py`,
`tests/curation/test_lock_rule.py`; T1.

### SP2: #362 and #259 follow-ups (measurement first, fix only if a bar is missed)

1. Worker region stage after #362: the G2 harness (500 set P photos, 2 boxes per item, flushes of
   32, 3 runs). Bar: at most 1.2x v0.6.0 per item (86 ms, 0.077 CPU-s per item).
2. COCO embed pass with the two-chunk window (1,500 COCO images, 3 runs). Record img/s, GPU, API
   CPU per image against log F3 (16.2 img/s). No bar; a gain under the spread is reported as none.

Expected: region stage near v0.6.0 (each photo decoded once per flush instead of once per item:
about 2.4 times fewer decodes); COCO pass +0 to +30 percent. Risk: low (measurement).

### SP3: #219 work crops at model size

Tier: Sonnet; Opus reviews the gate results.

- A per-project ingest policy field `crop_max_side` (int or null; null = off, recommended 1024) in
  `src/services/curation/ingest_policy.py`, served and edited through the existing ingest-policy
  routes (`make contracts`). Applied where work crops are cut: `crop` in
  `src/services/curation/ingest_index.py`, the crop-JPEG path in
  `models/gpu_image_prep/1/prep_geometry.py` (input from the policy), `src/services/curation/crop_bytes.py`.
  A VLM crop cap from the active VLM's input size at the VLM crop sites
  (`rg -n crop_jpeg src/services/labeling src/services/curation`). Embedding inputs are untouched.
- Tests first (`tests/curation/test_crop_max_side.py`): off produces byte-identical crops to the
  current code; on caps the long side and keeps aspect; the GPU route and the CPU route agree on
  the capped size.
- Gates on set B: crop-cache bytes per image at least 50 percent lower; VLM label agreement at
  least 0.97 (parity capture twice, `scripts/bench/parity.py capture|compare`); P-region box IoU
  within the parent plan's thresholds.

Expected: crop bytes per 20 MP crop down several times; GPU-route img/s up if SP0 shows
`crop_jpeg` above about 10 percent of the model's time. Risk: medium (label agreement; the cap
is off until the owner turns it on, O-S2).

### SP4: #40 high-resolution acceptance row and close-out

Set P 20k NVMe sample (owner yes for the 190 GB copy), one card, two `gpu_image_prep` instances,
3 reps, crop cap as decided in SP3. Bar: at least 29 img/s, GPU above 75 percent (500 ms samples).
Expected from E4: about 45 to 48 img/s. Then the owner closes #40 for v0.7.0 and R1 (resident
frame across every model) is filed in v0.8.0 (program plan O-A4).

### SP5: #45 ladders (public rows) and the limits table

1. `scripts/datasets/fetch_openimages_bench.py --category` (repeatable, display names resolved to
   label ids, per-image licence in the manifest); pin `vehicles` 200,000 (seed 20261009, commit the
   `.sha256` only). Test in `tests/test_fetch_openimages_bench.py`.
2. Rows with `scale_run.py` (latency probe on): full COCO (about 164k; bar within 10 percent of
   29.0 img/s, 0 failures), 20 MP 100k from the NAS read-only (0.6.0: 21.0 img/s at 51 percent;
   expected about 45), R3 synthetic limits table (k-NN, text, filtered and `GET /clusters` latency
   p50/p95/p99, cluster pass time at 100k, 500k, 1M; storage per 100k).
3. The 1M real row stays v0.8.0 (B6).

### SP6: #356 re-check (inside SP5's 100k runs)

After an API restart and after a worker recycle: 300 distinct phrases, new connection per request,
`page_size` 30. Bar: p99 under 300 ms idle at 100k; during ingest p99 recorded (target under
500 ms). If the bar is missed, the per-stage timing (encode, k-NN, hydration) names the stage, and
the fix lands in this lane. Log section J; `docs/PERFORMANCE.md` row.

---

## 6. Backend cluster outputs (CQ), branch `feat/v070-cluster-outputs`

New worktree `$MAIN/../OpenProcessor-wt-cq` from `main`. Merged into `feat/v070-cluster-ui`
(`--no-ff`) before UI1 starts. Rule: the backend serves data and decisions (ranks, flags, counts,
options); Cropwright renders. Every route or wire change runs `make contracts` in the same commit.

Layout (layering: routers > services > clients; `domain` and `schemas` import neither; every file
under 640 lines):

| Path | Content |
|---|---|
| `src/schemas/curation/clusters.py` (new) | `ClusterCard`, `ClusterListResponse`, `ClusterSortOption`, `ClusterFlag`, `ClusterFacets`, `LabelOp`, `LabelOpList`, `LabelOpUndoResponse` |
| `src/domain/cluster_flags.py` (new, pure; tests in `tests/curation/` like the other domain modules) | Thresholds and the rule that turns card numbers into flags; `validation_state(size, validated)` |
| `src/services/curation/cluster_cards.py` (new) | Aggregation body, bucket to card, server sort and filter, truncation flag (moved out of the router) |
| `src/routers/curation/clusters.py` | Thin: parameters, call the service, return the typed model |
| `src/services/curation/item_filter.py`, `src/routers/curation/crops_list.py` | New in-cluster filters (CQ2) |
| `src/domain/history.py`, `src/routers/curation/label_undo.py`, `crops_write.py`, `_common_models.py` | Label operations (CQ3) |

### CQ1: typed cluster list with server sort, filter and quality fields

`GET /clusters` keeps its path and becomes typed (`response_model=ClusterListResponse`).

Card fields kept: `cluster_id`, `cluster_kind`, `size`, `validated_count`, `labelled_count`,
`dominant_class_id`, `dominant_class_name`, `dominant_count`, `purity`, `purity_n`,
`purity_basis`, `purity_tier`, `label_agreement`, `labelled_share`, `promotable`, `is_unlabeled`,
`n_subclusters`, `subcluster_ids`, `updated_at`, `representatives`.
Removed: `label_purity` (duplicate of `label_agreement`; the auto-promote gate keeps its internal
name). Frontend readers move to `label_agreement` in UI1 (same merge).

Card fields added (one aggregation; no per-card query):

| Field | Source | Meaning |
|---|---|---|
| `detector_top_class` | terms agg on `detector_class_name` (size 1) | The detector's most common class among members |
| `detector_top_share` | that bucket / members with a detector class | Detector agreement inside the cluster |
| `detector_conf_mean` | avg agg on `detector_confidence` | Mean detector confidence |
| `validation_state` | `domain.cluster_flags.validation_state` | `none`, `partial` or `complete` |
| `review_counts` | one filter agg per tab in `CLUSTER_REVIEW_TABS` built from `SIMPLE_TABS` | Members a review tab would show, per tab id (`all`, `detector_disagreements`, `outliers`, `vlm_low_conf`, `mismatches`) |
| `needs_review_count` | `review_counts['all']` | Sort key for the review queue |
| `flags` | `domain.cluster_flags.card_flags` | Served codes: `mixed_detector_classes` (detector_top_share under 0.8 with at least 10 members), `low_geometry_purity` (purity tier low), `label_disagreement` (label agreement under 0.9 with at least 5 labelled), `promotable`, `needs_review` (needs_review_count above 0), `unvalidated` (validation_state none) |

Response fields added: `sort` (the applied one), `sort_options` (`[{id, label}]`, served
vocabulary), `flag_vocabulary` (`[{id, label, description}]`), `total` (after filters),
`truncated` (true when the aggregation returned `max_clusters` buckets: the list is incomplete and
the UI says so; fail closed, never silently partial).

New query parameters: `sort` (`needs_review_desc`, `size_desc`, `size_asc`, `label_agreement_asc`,
`purity_asc`, `purity_desc`, `detector_conf_asc`, `unvalidated_desc`, `updated_desc`,
`dominant_class`; default `size_desc`), `dominant_class` (repeatable name), `detector_class`
(repeatable name), `min_size`, `max_size`, `validation_state`, `flag` (repeatable),
`needs_review` (bool). Representatives are paged (`offset`, `limit`) over the **served** order,
so the frontend's per-card fetch becomes unnecessary (deleted in UI1).

Tests (red first, `tests/curation/test_cluster_cards.py`, query-capable fake OpenSearch):
each sort orders a fixed fixture as expected and is stable on ties (by `cluster_id`); each filter
removes exactly the expected cards; `truncated` is true at `max_clusters`; flags at each threshold
boundary (`tests/curation/test_cluster_flags.py`); `review_counts['all']` equals the count
`GET /review/{tab}` serves for that cluster with tab `all` (same clause builder from `SIMPLE_TABS`);
`GET /clusters` leaves the `LEGACY_DICT` allowlist of `tests/test_contract_typed_responses.py`
(the ratchet fails on a stale entry, so remove it in the same commit);
holdout and excluded members never counted; the response validates against `ClusterListResponse`;
cross-project leak row added in `tests/curation/cross_project_leak/`.

Bars: contracts regenerated; `GET /clusters` p95 at 100k items (512 candidate clusters plus class
clusters, `limit=50`, `per_cluster=4`) at most 400 ms on the SP stack, measured 3 times with the
SP6 probe, and within 1.5x of the same call on `main` in the same session; at 1M synthetic
recorded.

### CQ2: in-cluster filters and facets

- `ItemFilter` gains `detector_class_names` (list), `detector_conf_min`, `detector_conf_max`,
  `review_tab` (a `SIMPLE_TABS` id: applies that tab's clauses, so "outliers", "detector
  disagreements" and "needs review" inside a cluster are the same queries as `/review`).
  `GET /crops` `sort` gains `detector_confidence`.
- New route `{P}/clusters/{cluster_id}/facets` (GET), response `ClusterFacets`: counts per
  `detector_class_name` (top 20), per `class_source`, validated and unvalidated, per review tab of
  `CLUSTER_REVIEW_TABS`, and a detector-confidence histogram (10 fixed bins). The UI renders chips
  with these counts and sends the matching `GET /crops` filter.
- Tests: `tests/curation/test_item_filter_detector.py`, `tests/curation/test_cluster_facets.py`
  (counts equal what the filtered `/crops` returns, per facet value; holdout excluded), leak row.
- Bar: facets p95 at most 150 ms for a 2,000-member cluster at 100k items.

### CQ3: label operations with server-side undo (Opus review: data integrity)

- `PUT /crops/batch_label` (crop ids or selection) returns `op_id` and records it on every class
  history entry it writes (`src/domain/history.py` `record_class_snapshot` gains `op_id`).
- New route `{P}/label_ops` (GET, newest first, `limit` up to 50): `op_id`, `at`, `class_name`,
  `count`, `undone`, `undoable` (false once any item's newest entry is no longer this op).
- New route `{P}/label_ops/{op_id}/undo` (POST, body `{dry_run: bool}`): restores each item whose
  newest class history entry is this op; items changed since are `kept_changed`; locked or holdout
  items were never written and are not touched; bounded concurrency (32); a second undo is a no-op
  (409 `already_undone`).
- `POST /crops/label/undo_batch`: bound its concurrency the same way (today unbounded) and give
  it a response model, removing it from the `LEGACY_DICT` allowlist.
- Tests (red first, `tests/curation/test_label_op_undo.py`): exact restore of a 12,000-item op
  (fake), a human edit after the op is kept, second undo 409, OCC conflict reported not swallowed,
  concurrency never above the bound, dry run writes nothing, leak row. The undo of a selection
  op larger than 5,000 items works (today it cannot).
- Bar: undo of a 10,000-item op on the SP stack completes in under 60 s with 0 lost or extra
  writes (before/after audit query).

---

## 7. Auto-label loop: S9 to S11 and Z5

### 7.1 L1 (branch `feat/v070-assist-loop`, worktree `OpenProcessor-wt-loop`)

First commit or drop the one uncommitted test edit (read the diff; it renames the expected
source). Then S10 of the loop plan:

- `docker/test/fake_vlm.py`: `class_cycle` setting on its test-only control endpoint (one root
  fails and splits) and an `answers` mode that answers from a posted `crop_id -> class` map (used
  by Z5-I to replay oracle labels through the product).
- `tests/live/test_live_assist_loop.py` exactly as loop plan section 12 "T1 live" lists.
- Skip the `rehearsal_stages.py` sub-item (that driver does not exist; O-S4).
- Update the `auto_label_loop_plan.md` status line and its README row (S1 to S8, S10 done).

Verify: T0 for `tests/curation/test_assist_loop_*.py`, T1, `make contracts-check`, docs check.
Then the owner's yes for the PR (gate G1). After merge to `main`, UI5 can start.

### 7.2 L2: S0 and S11 (GPU 0 under the lock, local VLM; after L1)

- S0 (`scripts/bench/vlm_rate.py`) only as far as the loop's time estimate needs it: concurrency
  4, 8, 16 at images per call 1 and 4 on 500 public crops, one row repeated 3 times.
- S11 T2 runs of loop plan section 12: the #341 evaluation project (precision), the 10k COCO
  project (coverage and cost), stub VLM then the tested local catalog model.
- Bars (loop plan, owner-confirmed Q7): lock-rule violations 0 (before/after audit query);
  precision at least 0.95; similar-class false propagation at most 3 percent; at least 60 percent of
  eligible items labelled at no more than 1 VLM crop per 20 labelled; budget never exceeded.
- Records: log section H, `docs/PERFORMANCE.md` loop row, raw JSON under
  `$MAIN/artifacts_local/bench/v070/L2/`.

### 7.3 Z5: the registered follow-up of Z4 (the only research wave)

Branch `feat/cluster-z5-experiment` from `feat/cluster-z4-experiment` (new worktree
`OpenProcessor-wt-z5`). Tier: Sonnet; Opus reviews the registration before any score.

1. **Registration commit first** (`docs/research/cluster_config_z5.md`, same structure as Z4): set
   from the Open Images v7 **train** split, disjoint from every earlier pin (Z4's included, checked
   by name and SHA-256); quota so the labelled set holds at least 1,200 open-set crops (at least 600
   in the held-out half), other groups as Z4; arms `key`, `key_guard`, `ivf_loop`, `ivf_gate`
   (`within_key` dropped: it failed B-b in Z4 and the registration says so); bars B-a, B-b, B-c and
   the adoption rule **unchanged**; 20 seeds; fold 1.
2. Fetch (`scripts/datasets/fetch_vehicle_eval.py --split train`, on the Z4 branch), ingest into a throwaway project
   on an isolated stack on GPU 0 under the lock (or on the `optest1` test stack only with the
   owner's yes, gate G3), export vectors read-only, build labels, score on CPU with
   `scripts/research/cluster_diag/z4_run.py` (generalised to read the Z5 pin; tests
   `tests/research_scripts/test_loop_sim.py`, `test_z4_bars.py` stay green).
3. Results commit. **Rule:** if `key` or `key_guard` passes, Z5-I follows; otherwise IVF roots stay,
   the note says so, and no further clustering wave is planned in v0.7.0.

### 7.4 Z5-I: detector-key roots and the hold-back guard in the loop (only after Z5 passes)

Branch: the loop branch if it has not merged yet, else `feat/v070-loop-key-roots` from `main`.
Tier: **Opus design review first** (lock rule, drift, rollback with held-back members), Sonnet
implements.

- `src/services/curation/assist_loop/partition.py`: `DetectorKeyPartition` implementing
  `PartitionReader` (roots = eligible members grouped by `detector_class_name`, a `none` root for
  missing, `root_of` by the stored field). `ClusterFieldPartition` stays for scoped runs over
  chosen clusters.
- `src/domain/loop_guard.py` (pure): `held_back(sampled_vecs, member_vecs, percentile)`; called in
  `node_step._propagate` before the write; held-back members are never written, are recorded as
  node decision `held_back` and appear in the review list.
- Wire (`src/schemas/curation/assist_loop.py`): request `scope.partition` (`detector_key` or
  `cluster`; the default is the one Z5 adopted), `guard {enabled, percentile}`; `LoopNode.held_back`,
  `totals.skipped.held_back`. `make contracts`.
- Tests (red first): guard boundary at the percentile; held-back never written; detector-key roots
  include `none`; same seed same roots; lock rule per item class unchanged; rollback restores
  exactly with held-back members untouched; a replay test: the product driver with the fake VLM in
  `answers` mode on a fixed public fixture reproduces the simulator's U1 within 0.02.
- Bar: on the Z5 held-out set replayed through the product (fake VLM `answers` mode, 3 seeds),
  decisions per item at most 0.5x the IVF-root run with propagated precision at least 0.95; then
  one S11 rerun on the 10k COCO project with the local VLM meets the L2 bars.

---

## 8. Cropwright lane (UI), branch `feat/v070-cluster-ui`

Worktree `$MAIN/../OpenProcessor-wt-ui` (exists; holds uncommitted #382 work: finish and
commit it as UI0's first commit after reading the diff). Rules of `frontend/CLAUDE.md` apply in
full: served facts only, no new global keydown listener, keys through `keymapStore`, formatters in
`src/lib/format/`, `Modal.svelte` for dialogs, `frontend/CHANGELOG.md` entry per push,
`frontend/CLAUDE.md` updated for an architecture change, domain-neutral strings.

### 8.0 Bars for every UI wave

1. Vitest tests written first and seen failing (controllers in `*.svelte.ts` hold the logic).
2. A stubbed e2e test per new flow in `frontend/e2e/stubbed/` (fail-closed stub, no fixed sleeps).
3. `npm run check`, `npm run lint`, `npx knip`, `npm run build`, `npm run test:e2e` green.
4. **axe clean**: UI0 commits `frontend/e2e/stubbed/test_a11y_axe.py` (axe-core added as a
   devDependency and injected from `node_modules`); it runs on `/projects`, `/clusters`,
   `/clusters/[id]`, `/review`, `/classes`, `/settings` against the stub and fails on any violation.
   Each later wave adds its new page states to it.
5. **Real browser**: a build served by `vite preview` pointed at an isolated stack holding a public
   project (COCO), `DISPLAY=:11`, Chromium flags `--no-sandbox --disable-dev-shm-usage`. Full-height
   screenshots at 1600 and 430 px of every touched page and state, saved under
   `frontend/artifacts_local/<wave>/{d,m}_<state>.png`, **each opened and looked at** before the
   wave is reported done (overlap, clipping, empty states, wrong labels, private or domain words).
   Full height, because the shell scrolls inside `<main>`:

   ```python
   page.set_viewport_size({'width': w, 'height': 900})
   page.goto(url); page.wait_for_selector('<a selector that proves the data loaded>')
   h = page.evaluate("(() => { const m = document.querySelector('main');"
                     " return Math.ceil(m.getBoundingClientRect().top + m.scrollHeight); })()")
   page.set_viewport_size({'width': w, 'height': max(900, h)})
   page.screenshot(path=out)
   ```
6. No console error and no unexpected 4xx/5xx on touched routes (live run).

### 8.1 UI0: #380, #381, #382 and the axe gate

- #382: finish the in-flight change: `/projects` learns combine availability from a served fact
  (the global health or projects payload; if none exists, add a boolean to the global
  `GET /curation/health` response in a tiny backend commit with `make contracts`), never a probe
  that 404s. Test: no request to `projects/combine/__probe__`; e2e asserts zero console errors on
  `/projects`.
- #380: the crop card in `ClusterDetailGrid.svelte` becomes a non-button container with one real
  select button; inner actions stay buttons; remove `role` from `<button>`; give every empty `<th>`
  a visually hidden label (ScoresCard, prompt-pack, region-profile, open-vocab, VLM and config
  tables). axe test covers it.
- #381: footer count from the served cluster size after a bulk move (`60 / 3 listed` gone);
  placeholder not clipped at 430 px; the `Proposed` row hidden when no proposal; one timestamp
  formatter from `src/lib/format/` everywhere (pick the one `/settings` uses; test both sites).

### 8.2 UI1: cluster list filter and sort (after CQ1 is merged into this branch)

- Toolbar renders the served `sort_options` and `flag_vocabulary`; filters: dominant class and
  detector class (pickers fed by the registry and served values), size range, validation state,
  flags, needs review. Persist the choice per tab in `sessionStorage` (existing
  `persistedFilter.ts`), send it as query parameters.
- Cards show the served `detector_top_class` with `detector_top_share`, `detector_conf_mean`,
  `label_agreement`, `purity` tier, `needs_review_count`, `validation_state`, flags as chips with
  the served description as tooltip. A `truncated` response shows a banner with the served count.
- **Delete** the client sort (`sortClusters`), the client "unlabeled only" filter, and the per-card
  representative fetch (`displayOrderRepresentatives.ts`, `idsNeedingRepresentatives`,
  `loadMoreRepresentatives`'s per-id path) with their tests; representatives come from the paged
  list. Update readers of `label_purity` to `label_agreement`.
- Tests: controller sends each served sort and filter unchanged; renders served order verbatim
  (a shuffled fixture stays in served order); no `Array.prototype.sort` on cluster lists
  (scan test like `format.scan.test.ts`); `$contracts` pin of `ClusterListResponse`.
- Bar: with 512 clusters loaded, first screen of cards with thumbnails in at most 2 requests
  (was 1 + one per visible card); screenshots of each sort and of a filtered empty state.

### 8.3 UI2: in-cluster filters (after CQ2)

`ClusterDetailFilters.svelte` adds facet chips from `{P}/clusters/{cluster_id}/facets` (detector
class with counts, label source, validated or not, review-tab chips "needs review", "outliers",
"detector disagreements", detector-confidence band) and a sort by detector confidence or distance.
Selecting a chip sends the matching `GET /crops` parameters; counts are the served ones. Tests:
chip click sends the served value; counts rendered verbatim; empty facet shows the served zero.

### 8.4 UI3: bulk label with undo (after CQ3)

- "Label all shown" on the cluster detail: builds an `ItemSelection` from the active filters, runs
  the served dry run, shows the served count and the lock note in `Modal.svelte`, then writes.
- The toast and an "Undo" entry come from the served op (`op_id`); an "Operations" popover lists
  `{P}/label_ops` so undo survives a reload; undo shows `kept_changed` verbatim.
- Tests: dry run before write; confirm shows the served count; undo calls the op route with the
  served id; reload then undo works (stub); refusal text verbatim.

### 8.5 UI4: keyboard flow

New actions in `src/lib/keymapFallback.ts` (defaults chosen to not collide; the scan tests
enforce it): `cluster.next_cluster`, `cluster.prev_cluster` (walk the served list order with the
persisted sort and filters; the detail page fetches the list with `per_cluster=0`),
`cluster.label_shown` (opens UI3's dialog), `cluster.undo_op` (undo the newest op). Registered
with `keyboardStore.registerAction`; printed keys from `keymapStore.glyph`. Tests: no new
`window`/`document` keydown listener; next on the last cluster shows the served end state; a
rebind applies without reload.

Bar for the whole flow: a stubbed e2e walks "open the first needs-review cluster, filter to
detector disagreements, label shown, undo, next cluster" using keys only.

### 8.6 UI5: loop panel (S9 of the loop plan; after L1 is on `main` and merged here)

Exactly loop plan section 11: `frontend/src/lib/api_assistLoop.ts`, `types_assistLoop.ts`,
`assistLoop/assistLoopController.svelte.ts`, `components/assistLoop/AssistLoopPanel.svelte`
(+ `LoopLevelTable.svelte`, `LoopReviewList.svelte`), `frontend/src/lib/contract/assistLoopContract.test.ts`,
`frontend/e2e/stubbed/test_assist_loop_panel.py`. Plan (dry run), run (poll, budget bar, levels,
audit with bound, cancel, resume), review list (opens a node's members as a selection with UI2
filters), rollback (dry-run count, confirm). If Z5-I has landed, the panel renders the served
`held_back` decision and the partition kind; nothing is computed client-side.

---

## 9. Installer and docs wave (D1), branch `fix/v070-installer-docs`

New worktree `OpenProcessor-wt-d1` from `main`. Tier: Sonnet; Haiku runs checks. Small commits.

| Issue | Change | Test / bar |
|---|---|---|
| #371 | Default `OP_SEMANTIC_SEARCH_ENABLED` on in code (`src/routers/curation/search.py`), `env.template` and the installer-written env; explicit off still works; check startup cost and memory of the text encoder warm-up on a 24 GB card and the CPU text backend | `tests/test_env_surface.py`, a search default test; fresh unattended install answers text search with 200 |
| #363 | Each listed item fixed or documented (Ctrl-C trap with a resume message, `status` probes the stack's own ports and tiers, dry-run disk check on the target filesystem, control-plane summary wording, short SHA, branch-mode tag note, upgrade note for control-plane-only installs) | `tests/installer/` per testable item; `docs/research/installer_drive_log.md` updated |
| #355 | Every listed comment and doc line corrected (except `methods/ivf.py`, done in SP1); `backbone_embedding` removed from the mapping body if no reader (`rg -n backbone_embedding src`) with the mapping rule for existing indexes, or kept with the reason written | docs check clean; `rg` shows no wrong route or field |
| #383 | Both tests assert event ordering or loop-thread identity instead of elapsed time | each passes 20 times under `stress-ng --cpu 48` or equivalent load; seen failing first under load |
| #154 | MLflow model version aliases instead of `transition_model_version_stage` in `docker/trainer/mlflow_callbacks.py` | trainer test asserts the alias call and no stage call |
| #335, #373, #377, #378 | Verify only: run each fix's test on `main`, list the merge commit | owner closes (gate G5) |

Verify: T0, `$VPY scripts/docs/check_docs_vs_code.py`, `make docs-build`, installer tests.

---

## 10. Order, conflicts, GPUs

### 10.1 Lanes, worktrees and merge order

| Lane | Branch | Worktree | Waves | Merges to `main` |
|---|---|---|---|---|
| D1 | `fix/v070-installer-docs` | `OpenProcessor-wt-d1` (new) | D1 | 1st (small, owner yes) |
| L | `feat/v070-assist-loop` | `OpenProcessor-wt-loop` | L1 then L2 runs | 2nd (after L1; L2 numbers land in a follow-up docs commit) |
| CQ | `feat/v070-cluster-outputs` | `OpenProcessor-wt-cq` (new) | CQ1, CQ2, CQ3 | into the UI branch, not `main` |
| UI | `feat/v070-cluster-ui` | `OpenProcessor-wt-ui` | UI0, then UI1 to UI4 after CQ, UI5 after L1 is on `main` | 3rd, one PR with CQ |
| SP | `perf/v070-ingest-speed` | `OpenProcessor-wt-sp` | SP0 to SP6 | 4th (code parts); measurement docs may follow |
| Z5 | `feat/cluster-z5-experiment` then Z5-I branch | `OpenProcessor-wt-z5` (new) | Z5, Z5-I | research PR any time; Z5-I after L |
| EX | `docs/v070-84-wheel-loop` | new | #84 | before R1, or moved (O-S5) |
| R | `release/0.7.0` | new | R1 | tag |

Parallel from day 1 (CPU work): D1, L1, CQ1, UI0, SP0 code, Z5 registration. GPU work queues on
GPU 0 (10.2). Each lane merges `main` into itself (`--no-ff`) before its PR.

### 10.2 Files each lane touches (conflict matrix)

| Shared file or area | D1 | L | CQ | UI | SP | Z5-I | Rule |
|---|---|---|---|---|---|---|---|
| `contracts/` | maybe (#371 none expected) | yes | yes | UI0 maybe | SP3 | yes | Never hand-merge: on conflict take either side, run `make contracts`, commit |
| `CHANGELOG.md` `[Unreleased]` | yes | yes | yes | yes | yes | yes | Append in the right subsection; resolve by keeping both |
| `frontend/CHANGELOG.md` | | | | yes | | | UI lane only |
| `src/services/curation/clustering/` | comments only (#355, not `methods/ivf.py`, `orchestrator.py`) | | | | `methods/ivf.py`, `orchestrator.py`, `cluster_geometry.py`, `pool_coverage.py` | | D1 must not edit SP's files |
| `src/routers/curation/clusters.py`, `crops_list.py`, `label_undo.py`, `item_filter.py`, `src/domain/history.py` | | | yes | | | | CQ only |
| `src/services/curation/assist_loop/`, `src/schemas/curation/assist_loop.py` | | yes | | | | yes | Z5-I after L merges, or on the L branch after L1 |
| `frontend/src/routes/p/[project]/clusters/` | | | | yes | | | UI only (UI5 too) |
| `models/gpu_image_prep/`, `ingest_*.py`, `crop_bytes.py`, `ingest_policy.py` | | | | | yes | | SP only |
| `docker/test/fake_vlm.py`, `tests/live/` | | yes | | | | yes | L first |
| `docs/design/README.md` | | yes | | | | | One row per merge |
| `docs/research/v070_measurement_log.md`, `docs/PERFORMANCE.md` | | L2 | | | yes | | New sections only; never edit another lane's section |

### 10.3 GPUs, locks, stacks and preflight

- **GPU 0** (A6000, PCI `03:00`): ours, one job at a time, every job under
  `flock -w 3600 $SCRATCH/locks/gpu0.lock <cmd>` per job. **GPU 2** (A6000, `84:00`) holds the
  owner's `optest1` test stack: use only in an owner window under `$SCRATCH/locks/gpu2.lock`.
  **GPU 1** (3080 Ti) belongs to another session: never.
- Always `CUDA_DEVICE_ORDER=PCI_BUS_ID` so indices match `nvidia-smi`.
- GPU 0 queue (estimates): SP0 runs (about 2 h), SP1 (CPU fit; GPU only if (b)), L2 S0 and S11
  (about 4 h with the VLM), Z5 ingest (about 30 min), SP3 parity (about 1 h), SP4 20k row (about
  1.5 h), SP5 full COCO and 20 MP 100k (about 6 h, NAS read-only), EX training if kept.
- Stacks: one isolated stack per lane that needs one (`setup-openprocessor.sh --dir $SCRATCH/op-<lane> --port-base <unused, e.g. 5700 SP, 5800 UI, 5900 L>`,
  then the branch's `src/` and `scripts/` mounted read-only over the API and workers as in the
  measurement log's "Live verification" setup), always `docker compose -p <lane name>`. Never act
  on the default project `openprocessor` or on `optest1`, and never delete its projects.
- **Preflight** before any live step: `nvidia-smi`, `docker ps --filter name=<project>`, timed
  `curl` of the stack's `/health` (and through the Cropwright origin for UI checks); a stack that is
  down or restarting stops the step and is reported, not worked around.

### 10.4 Pitfalls learned (binding)

1. Never `pgrep -f` or `pkill -f` with a pattern that also appears in your own command line: it
   matches your own shell and kills it. Use PIDs you recorded, `pgrep -x`, or `docker compose -p <name> ps`.
2. Pre-commit runs whole-tree hooks (naming scan, contract drift, docs check, ratchets): commit
   self-contained changes, one agent per worktree, nothing half-done left unstaged.
3. The venv lives in the main checkout only: `$VPY`, `$VPC` from the worktree root; never
   `source`, never install `requirements.txt`.
4. The naming-leak scan (`scripts/codegen/check_naming_leaks.py`) bans the private company and its
   initials, vendor model family names, vehicle-registration wording and the related field-literal
   prefixes; Cropwright's `domainNeutral.scan.test.ts` bans domain nouns under `frontend/src/`.
   Name the local VLM "the tested local catalog model". Allowlist only with a reason.
5. Route or wire change: `make contracts` in the same commit; frontend contract tests read
   `contracts/`.
6. Docs checker: a design doc names a planned route as `{P}/...` with no HTTP method in front,
   names only `OP_*` variables that exist, and links only files that exist on the branch.
7. Playwright full-page shots show only the viewport here; use the recipe in 8.0.
8. 5 s GPU samples under-read; use 500 ms. A load average above 16 marks a run noisy.
9. The 3 GiB crop-cache cap prunes during long runs; set it to 0 for count checks.
10. Watchtower re-pulls public `:latest` over local builds: release candidates use local-only tags.
11. File size cap 700 lines (warn 640), complexity and layering ratchets only shrink.
12. Wall-clock assertions flake on this host (#383): assert ordering or thread identity.
13. Bake-off outputs and the private projects on `optest1` are never bulk-deleted.

---

## 11. Release path for 0.7.0 (R1) and owner gates

### 11.1 Must be true to tag

1. Every v0.7.0 issue is closed with the DoD of program plan 0.1, or moved by the owner:
   #40 (SP4), #41 (R1), #45 (SP5 public rows; 1M row moves to v0.8.0), #84 (EX or moved), #154,
   #219, #286 (L2 and, if Z5 passes, Z5-I), #335, #337, #355, #356, #357, #363, #371, #373, #377,
   #378, #380, #381, #382, #383.
2. On `release/0.7.0` (cut from `main` after the lanes merge): the offline suite green once
   (`$VPY -m pytest tests/ -q --no-cov -m 'not live'`), T1 green, frontend `npm run check`, `lint`,
   `knip`, `vitest`, `build`, `test:e2e` green, `make contracts-check`, docs check, `make docs-build`.
3. Phase A audit (`[O-audit]`, program plan 2.2) written to `docs/research/audits/phase_a_audit.md`
   with zero open high findings; release audit `release_v070_audit.md`.
4. **#41 last**, on the final dependency set: runtime image at most 50 percent of 21.4 GB, no image
   grows, `docs/image_size_audit.md` before and after.
5. `make release-dry-run` (Trivy, no un-allowlisted CRITICAL) and `make release-verify`.
6. Installer release candidate: `make acceptance` (scripted clean install to uninstall) on an
   isolated directory with local-only RC tags, plus an upgrade from a 0.6.0 install (#377 path)
   and a 24 GB-card plan check (#373).
7. `CHANGELOG.md` `[Unreleased]` becomes `[0.7.0] - <date>` with upgrade notes (re-embed, crop
   cap, semantic search default, propagated label kind, label operations); `frontend/CHANGELOG.md`
   likewise; `docs/design/README.md` statuses current.
8. `docs/PERFORMANCE.md` carries the 0.7.0 public rows (stage profile, cluster pass, COCO full,
   loop); set P figures only as ratios and pass/fail.

### 11.2 Owner approval gates

| Gate | What | When |
|---|---|---|
| G1 | Push, PR and merge of each lane | end of each lane |
| G2 | GPU 2 or two-card windows | only if a row needs it (none planned) |
| G3 | Z5 ingest on `optest1` instead of GPU 0 | Z5 step 2 |
| G4 | A change to the IVF fit numerics (SP1 (a) or (b)) after the gate result | SP1 |
| G5 | Closing issues on GitHub (#335, #373, #377, #378, then each wave's issues) | per merge |
| G6 | Close #40 at v0.7.0 and file R1 in v0.8.0 | after SP4 |
| G7 | Label operation semantics (CQ3): undo restores per item, keeps later edits | before CQ3 code |
| G8 | 190 GB NVMe copy of the set P 20k sample | before SP4 |
| G9 | Tag, GitHub release (published, not draft), image push to Docker Hub | R1 |

### 11.3 Open questions for the owner (recommendations)

| ID | Question | Recommendation |
|---|---|---|
| O-S1 | Cluster list default sort | `needs_review_desc` on `/clusters` once CQ1 lands (the server default stays `size_desc` for API callers) |
| O-S2 | Turn the crop cap (`crop_max_side` 1024) on by default if the SP3 gates pass? | yes for new projects; existing projects keep off until set |
| O-S3 | Detector-key roots as the loop default if Z5 passes | yes, with the guard, IVF roots kept for scoped runs |
| O-S4 | Build `rehearsal_run.py` (program plan W0 step 7) before the tag or before the winter run | before the winter run, not a tag blocker |
| O-S5 | #84 wheel loop in 0.7.0 or 0.8.0 | keep in 0.7.0 only if GPU 0 has a free day after SP5; else move |

---

## 12. Owner review package per wave

Each wave's last commit adds `docs/reviews/<wave>_review.md` (for example `docs/reviews/SP1_review.md`).
Links only to tracked files; private figures as ratios or pass/fail. Template:

```markdown
# <wave id>: <title>

Branch, head commit, merge base. Issues: #...

## What changed
One line per commit (subject and short hash).

## Acceptance
| Bar | Measured | Met | Raw data (path under artifacts_local or docs/benchmarks) |

## How it was tested
Red-first tests (file::name, seen failing at <commit>), suites run, live tier, screenshots
(paths under frontend/artifacts_local/<wave>/, listed; each was opened and viewed).

## Risks and what was not done
## Decisions needed from the owner (gate ids)
## Five-minute check for the owner
Commands or pages to open, with the expected result.
```

---

## 13. Model split per wave

| Wave | Opus | Sonnet | Haiku |
|---|---|---|---|
| SP0 | | instrument, run, write tables | collect JSON, verify sums |
| SP1 | gate verdict on numerics; #357 review | ladder flags, fix, runs | run fixed commands |
| SP2, SP4, SP5, SP6 | | runs and notes | JSON collection, PERFORMANCE rows check |
| SP3 | gate results review | policy field, cap, parity | contract check |
| CQ1, CQ2 | | implement, tests | contract and leak-row checks |
| CQ3 | design review before code, code review | implement, tests | |
| UI0 to UI5 | | implement, tests, screenshots viewed | knip, scan tests, screenshot listing |
| L1 | | fake VLM, live test | run T1 |
| L2 | go/no-go on the numbers | runs | |
| Z5 | registration review | fetch, ingest, score, note | disjointness and manifest checks |
| Z5-I | design review, code review (lock rule, rollback) | implement | contracts |
| D1 | | implement | docs check, installer test runs |
| R1 | phase and release audits | images, Trivy, acceptance run | size and naming scans |
