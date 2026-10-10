# v0.6.0 wave plan: precision, host CPU, zero-copy data path, clustering, vector store

Status: plan (nothing implemented). Milestone v0.6.0. Umbrella issues: #40 (pipeline), #56
(vector quantization), #67 (request-driven plans), #42 (12 GB engines), #41 (image slimming).
Work-package issues: #208 to #221, plus #207 and #153 (table in section 10), and #226 to #232 from
the serving backend trade study (section 13,
[v060_serving_backend_trade_study.md](v060_serving_backend_trade_study.md)).

This is the **sequencing** plan for v0.6.0. It does not replace
[triton_pipeline_optimization_plan.md](triton_pipeline_optimization_plan.md) (the "parent plan":
design detail, accuracy-gate table in its section 7, Waves 1 to 10); it orders that work against
the measured v0.5.0 baseline, adds what the parent plan does not cover (engine precision, API CPU
attribution, idle CPU, cluster scheduling, INT8, vector store), and fixes the downstream order in
which numerics may change. The vector store comparison has its own protocol:
[v060_vector_db_evaluation_plan.md](v060_vector_db_evaluation_plan.md).

A fresh agent can execute any work package (WP) from this file plus the parent plan. File:line
references were checked against `origin/main` at `877f62c9` and will drift; re-find by symbol.
New settings are written in lowercase (`ingest_transport`) as proposals; the implementing WP picks
the `OP_*` name and adds it to `env.template` and `tests/test_env_surface.py`.

## 0. Ground rules (binding for every WP)

1. **Isolation.** The compose project `openprocessor` is the owner's live stack. Never run
   `docker compose` without `-p <bench-name>` (for example `-p op060-w1`), never against the default
   name, own ports, GPU 0 only (`TRITON_GPU_ID=0`, `API_GPU_ID=0`, ...). GPU 2 belongs to another
   project. GPU 1 (the 12 GB card) only with the owner's explicit yes (open question O3).
2. **Public data only.** The pinned set is `scripts/datasets/manifests/coco_bench_2000.json`
   (2,000 COCO images, seed 20261009, manifest sha256 `edde8ae3...`), plus larger pins from the
   same command. Set B (private high-resolution archive) is run by the owner only; public docs
   carry pass/fail and ratios for it, never paths or counts.
3. **Benchmark protocol (every WP that claims speed).** Tooling from PR #206
   (`scripts/bench/baseline_suite.py`, `suite_report.py`; must be merged first, see Wave 0):
   fresh project per repetition, 100-image warm-up project first, `/ingest/upload` and
   `/ingest/batch` interleaved, **3 repetitions, report median (min-max)**, 32 images per request,
   4 client threads, ingest policy `all` as the headline row (vehicles-style "embed everything" is
   the case that must be fast) plus `selected` and `lazy` as extra rows when the WP touches
   embedding. Record host load average before each repetition and discard runs above 8. VLM worker
   stopped during ingest runs unless the WP measures it. Raw JSON to
   `artifacts_local/bench/v060/<wp>/<arm>/`; the summary table goes to `docs/PERFORMANCE.md`.
4. **Parity before speed.** A WP may change the numerics of **one stage only**. Its gates are
   checked at that stage and at every downstream consumer in this fixed order (section 2). A
   transport-only change must be bit-identical (gate P-json).
5. Red-first tests, a mutation check, `.venv/bin/pre-commit run --all-files`, the offline suite
   (`.venv/bin/python -m pytest tests/ -q --no-cov -m 'not live'`), `make contracts` when a route or
   wire model changes, conventional commits, merge with `--no-ff`. Opus writes any sub-plan, Sonnet
   implements and reviews (suggested tiers per WP below).

## 1. Measured baseline this plan answers to (v0.5.0, PR #206)

Source: `docs/PERFORMANCE.md` "v0.5.0 baseline" and parent plan section 11.1 on branch
`perf/baseline-045` (PR #206, not yet merged); raw JSON `docs/benchmarks/v050_baseline.json`,
`v050_storage.json`, `v050_idle_background.json` on that branch. One RTX A6000 (GPU 0),
Xeon E5-2680 v3 48 threads, Triton 2.70.0, TensorRT `libnvinfer.so.11`, OpenSearch 3.6.0.

| ID | Fact | Number |
|---|---|---|
| B1 | Ingest throughput, policy `all` | 8.93 (8.89-9.13) img/s upload, 9.19 (8.94-9.45) batch; 6.39 items/image |
| B2 | PE GPU time per image | 7.39 inferences x 13.7 ms = 101 ms, **90 %** of wall; perf_analyzer ceiling 74-78 inf/s flat over batch 8-32 |
| B3 | PE engine | 1.27 GB plan = size of the FP32 ONNX, built 2026-09-26 and reused; a fresh install bakes a 636 MB FP16 ONNX. Diagnosis "stale FP32" is **inferred from size**, not inspected |
| B4 | Detector GPU time | 4.9 ms per image (4 %), mean batch 16 |
| B5 | API CPU per image | 0.490 (0.299-0.954) s upload, 0.358 (0.270-0.360) s batch; timed stages sum to 33 ms |
| B6 | Timed stages (ms/image) | decode 4.6, `resize` 24.1, jpeg_encode 3.3, crop 0.6; opensearch_write 404 ms summed latency (2 calls, concurrent) |
| B7 | Wire bytes to Triton | 14.9 MB/image FP32 (PE 10.0 MB, detector 4.9 MB) vs 166 KB JPEG |
| B8 | Cluster training after ingest | 88 (68-125) s upload, 78 (68-80) s batch over 12.8k items, CPU IVF; 30-55 % of ingest wall |
| B9 | Idle CPU, 18 projects with data, no VLM configured | VLM worker running: api 1.56, vlm-worker 0.66, opensearch 0.97 cores; stopped: 0.34 / 0.0 / 0.13 |
| B10 | Single-image `/detect` | 94 ms at the API for about 5 ms of GPU (95 % host) |
| B11 | Storage | 65.4 KB/image, 8.85 KB/vector (1.06x the 8.35 KB reference) |
| B12 | Triton log | `--log-verbose=1`: one pinned-memory alloc/free line pair per tensor per request (12,038 lines in the window) |

Corrections found while planning (verified in code, fold into the PR #206 text or WP-1.2):

- The `resize` stage (24.1 ms/image) is **PE crop preprocessing** (`PEEncoder.embed_crops`,
  `src/clients/pe_encoder.py:387`, cv2 resize plus FP32 normalize for 6.4 crops), not the detector
  letterbox. The detector letterbox (`letterbox_to_square`, `src/services/detection/geometry.py:113`)
  and the FP32 tensor build (`src/services/curation/ingest_detect.py:132,182`) are untimed.
- `embed_crops` preprocessing runs synchronously on the event loop (no `to_thread`).
- Vectors reach OpenSearch as `list(ndarray)` (`src/services/curation/item_doc.py:138,222`): a
  list of numpy float32 scalars that the opensearch-py `JSONSerializer.default` converts one by one
  (an `isinstance` chain plus an `import numpy` per float, about 7.6k calls per image).
- IVF clustering trains on the CPU although `faiss-gpu-cu12` is installed and the API container has a
  GPU: `detect_cluster_backend` (`src/services/curation/clustering/backend.py:98`) returns `cpu` when
  cuML/CuPy do not import (the stock image has neither), and `ivf.py` only tries FAISS GPU when the
  backend is `gpu`.
- The cluster-refresh daemon (`scripts/curation/cluster_refresh_daemon.py`, `--growth-threshold`
  default 200, `--auto-label` default on) posts the **synchronous** `/pipeline/auto_label` route
  (`train_clusters` defaults to true) every 200 new crops, so full IVF retrains run inside the API
  process while ingest is running. The module docstring still says 1000.
- The installer reuses an existing plan if it loads and the Triton image digest is unchanged
  (`group_should_skip`, `scripts/lib/model_setup.sh:205-221`); there is no precision, TensorRT, GPU
  or ONNX stamp, and `_ms_pe_build_engine` (`:283-299`) silently retries FP32 when FP16 fails.

Ranking that drives the waves: PE precision first (B2, B3), then host CPU (B5, B9, B12) because it
binds once the GPU doubles, then the wire and transport (B7), then cluster scheduling (B8), then
the consumers. GPU decode is not worth it on COCO (B6: 4.6 ms decode); its case rests on set B.

## 2. The downstream order and its parity chain

Data flows `decode -> detection (boxes, classes) -> crops -> embeddings (PE crop, PE whole frame;
MobileCLIP on core routes) -> faces / OCR -> clustering -> VLM labeling and region stage ->
review -> training / export`. A change upstream silently changes everything below it, so:

- **Rule D1.** Waves change stages strictly in this order: precision and host work that do not
  change numerics first (Wave 1), then detection inputs, then embedding inputs, then the GPU-resident
  merge of both (Wave 2), then clustering (Wave 3), then the VLM/region consumers (Wave 4), then
  packaging (Wave 5). Faces/OCR are planned in Wave 2 and built after it.
- **Rule D2.** A WP that changes stage S must show, on the pinned set, that every consumer below S
  either is unchanged or passes its gate using the new S outputs. The chain of gates:

| Stage changed | Gate at the stage | Downstream gates that must also pass |
|---|---|---|
| Detection input (decode, letterbox, uint8) | P-box, P-item (parent plan section 7), class **names** identical per item | crop boxes identical (integer pixel rect per item) => crops and everything below unchanged; if any crop box differs, run all gates below |
| PE precision or input (FP16, INT8, uint8, GPU resize) | P-pe (crop), P-pew (whole frame): cosine against the reference capture | kNN neighbour overlap@10 >= 0.95 on 1,000 query items; IVF agreement (section 2.1); near-duplicate decisions at 0.98 identical on >= 99.5 % of pairs; VLM label agreement only if the VLM consumes vectors (it does not today: skip) |
| Clustering implementation | IVF agreement and objective (section 2.1) | auto-promote outcome per item (class, `class_validated`) identical on >= 99 %; review-queue membership per tab identical on >= 99 % |
| Work crop size (VLM, segmenter, region) | P-vlm, P-sam, P-region, P-ocr (parent plan) | export manifest equality (class names, boxes within 1 px) |
| Transport only (shared memory, batching, serializer) | P-json: outputs and stored documents identical, minus ids and timestamps | none (identity) |

### 2.1 Cluster-agreement metric (new, defined once here)

On the residual pool of the pinned set with a fixed seed: (a) adjusted Rand index between the
reference and candidate assignment, (b) fraction of items whose nearest centroid is unchanged when
the candidate vectors are assigned to the **reference** centroids, (c) k-means objective relative
difference, (d) purity against COCO ground truth: for each item, the class of the best-IoU COCO
annotation (IoU >= 0.5, else "none"); cluster purity = share of the modal class, weighted mean.
Pass for a numerics change: (b) >= 0.98, (d) not lower by more than 0.5 points; (a) and (c) are
recorded (k-means is chaotic: ARI alone is too strict a gate). Implemented in WP-0.1.

## 3. Wave 0: prerequisites

**WP-0.0 Merge PR #206** (baseline suite, pinned manifest, PERFORMANCE section). Owner action. All
later WPs need `scripts/bench/baseline_suite.py` and the manifest. Fold in the corrections of
section 1 ("Corrections") in the same PR or in WP-1.2.

**WP-0.1 Parity capture and compare (#208).** Tier: Sonnet implements from parent plan sections
5.3 and 7 plus section 2 here.

- New `scripts/bench/parity_capture.py`: runs ingest of the pinned set (or a 200-image parity
  subset, seeded) into an isolated project and dumps, keyed by `(sha256, crop box)`: detector
  class name, box, confidence; integer crop rect; PE crop vector; PE whole-frame vector;
  `cluster_id` after one `train_clusters`; auto-promote outcome; optionally VLM label on a 200-crop
  subset (only when a WP touches the VLM path). Output `artifacts_local/bench/parity/<tag>/` as
  npz + jsonl.
- New `scripts/bench/parity_compare.py <ref> <cand> --gates P-box,P-pe,...`: implements the gate
  table of parent plan section 7 plus section 2.1 here; exits non-zero on failure; writes a markdown
  summary.
- Pure logic in `scripts/bench/parity_lib.py`, unit tests `tests/test_bench_parity.py` (synthetic
  boxes with shifted IoU, vectors with injected noise, a permuted cluster labelling that must pass
  ARI, strings). Mutation: flip the IoU comparison, drop the class-name check.
- Capture the **v0.5.0 reference** on the pinned set with the engines as measured (FP32-sized PE)
  and keep it as `ref-v050`; after WP-1.1 capture `ref-fp16`. Both are the references for later
  waves (most later WPs compare against `ref-fp16`).

## 4. Wave 1: precision, host CPU, idle CPU, cluster scheduling (low risk)

Goal: double the GPU-bound rate by fixing engine precision, then make sure the host does not
become the new limit, and remove idle and scheduling waste. No stored embedding space changes
except the FP16 rebuild, which is gated to be the same space.

### WP-1.1 PE FP16 engine, build stamp, re-baseline (#209). Tier: Sonnet.

Evidence: B2, B3. Touch points: `scripts/lib/model_setup.sh` (`group_should_skip`,
`_ms_pe_trtexec` `:269-279`, `_ms_pe_build_engine` `:283-299`, `_ms_record`), `export/build_pe_trt.sh`
(`BAKE_FP16` `:80`, FP32 retry `:162-181`, install `:198-212`), `export/trt_utils.py`
(`bake_fp16_onnx` `:96-133`, CLI `:215-230`), `openprocessor` (`cmd_models_install` `:334-339`, model
status), the other exporters' install steps (`export/export_*.py`, `export/build_*.sh`).

Steps:
1. Diagnose before changing: on an isolated stack, `trtexec --loadEngine=<plan> --dumpLayerInfo
   --profilingVerbosity=detailed` (or the TensorRT engine inspector) on the measured plan and on a
   fresh FP16 build; record layer precisions, plan size, build log. This confirms or refutes B3.
2. Build stamp: every installer/exporter writes `models/<name>/1/build.json` next to the plan:
   `{precision_requested, precision_built, trt_version, gpu_name, compute_capability,
   onnx_sha256, builder_args, built_at, exporter}`. `precision_built` comes from the bake result
   (`trt_utils.py` stderr `fp16|fp32`), never assumed.
3. `group_should_skip` also compares the stamp with the current target (TensorRT version from the
   Triton image, compute capability of the target GPU, requested precision, ONNX hash); a missing
   stamp means rebuild. `openprocessor models status` prints precision and stamp age per model.
4. FP32 fallback stays (a working stack beats none) but is recorded as `precision_built: fp32`, logged
   as a warning, and shown by `models status`.
5. Rebuild PE FP16 through the installer path on the isolated stack; capture `ref-fp16`; run the
   parity compare against `ref-v050`; run the full baseline protocol.

Parity gates: P-pe and P-pew FP16 vs `ref-v050`: p01 >= 0.995, median >= 0.999 (if they fail, the
stored space differs: stop and escalate, existing deployments would need a re-embed decision); kNN
overlap@10 >= 0.95; cluster agreement (section 2.1); near-duplicate decisions >= 99.5 % identical.

Expected win (**estimate**, uncertain): the A6000 runs FP16 tensor math with FP32 accumulate at
about 2x the dense TF32 rate on spec; the 12 GB card measured an FP16 PE plan at 75-90 crops/s
(`docs/research/pe_encoder_batching_benchmark.md`) and the parent plan quotes an older private
170 img/s for FP16 on the A6000. Expect PE ceiling 140-180 inf/s, ingest 14-18 img/s at policy
`all`. If the plan turns out to be FP16 already, the win is zero and B2 is a model-size fact: then
the levers are INT8 (WP-1.6) and the embedding policy, and the owner is told.

Tests: extend `tests/installer/test_model_setup.py` (it already drives `model_setup.sh`): run
`group_should_skip` in a temp dir with fake stamps (missing, matching, wrong precision, wrong
compute capability, wrong ONNX hash); mutation: ignore the precision field.
Rollback: revert; old plans still load (a missing stamp only forces a rebuild).
Acceptance: plan precision FP16 verified by layer dump; gates green; PE `infer ms/image` and
img/s recorded in `docs/PERFORMANCE.md` as "v0.6.0 Wave 1"; a stale plan is rebuilt on `models
install` in a test.

### WP-1.2 API CPU attribution (#210). Tier: Sonnet.

Evidence: B5 (0.27-0.95 s/image vs 33 ms timed), B10. Touch points: `src/utils/stage_timing.py`
(`STAGES` `:25`), `src/services/curation/ingest_detect.py` (`run_primary_batch` `:146`, tensor build
`:182`), `src/services/detection/geometry.py` (`letterbox_to_square`), `src/clients/pe_encoder.py`
(`encode_images` `:331`, `embed_whole_frame_bytes` `:418`), `src/services/curation/item_doc.py`,
`src/services/curation/ingest.py` (`_bulk_index` `:288`), `src/clients/occ.py` (`occ_upsert_bulk`
`:394`).

Steps:
1. Add stages: `letterbox` (detector letterbox plus FP32 tensor), `pe_preprocess` (rename the
   misnamed `resize`), `whole_frame_preprocess`, `triton_request` (InferInput build plus
   `set_data_from_numpy`; the gRPC send is measured as the difference to `embed`), `doc_build`,
   `bulk_serialize` (time the serializer by wrapping the client's serializer object), `occ_mget`.
   Update the Grafana panel and `docs/PERFORMANCE.md` stage list. Contracts are not affected.
2. Profile: `py-spy record --native --subprocesses -d 120` against the API container of the isolated
   stack during one pinned upload repetition; save the flamegraph under `artifacts_local/`. Also
   record per-worker CPU (32 uvicorn workers, `docker-compose.yml:247`).
3. Publish the attribution table (stage, CPU ms/image, share).

Acceptance: >= 80 % of API CPU seconds per image attributed; top three costs named with evidence.
No behaviour change (P-json identical). Rollback: revert.

### WP-1.3 API CPU fixes (#211). Tier: Sonnet; review by Sonnet with the WP-1.2 table.

Candidate fixes, each kept only if WP-1.2 shows it costs >= 5 % of API CPU, each its own commit:

1. `ndarray.tolist()` instead of `list(ndarray)` in `item_doc.py` (`:138`, `:222`, backbone `:227`)
   and `reprocess_embed.py:186`: same Python floats, identical JSON, no per-float `default()`.
2. A numpy-aware fast serializer for the OpenSearch clients built by the factory
   (`src/clients/opensearch/client.py:121`, `make_script_opensearch` in
   `src/services/projects/guard.py`): `orjson` with `OPT_SERIALIZE_NUMPY`. Gate: byte-level JSON
   equality is not required, value equality of every stored field is (P-json on `_source` read
   back); the guard keeps inspecting the same request paths (it does not parse bodies; verify).
3. Off-event-loop pixel work: `embed_crops` preprocessing and `letterbox_to_square` via
   `asyncio.to_thread` (cv2 releases the GIL); parent plan Wave 1 item 4.
4. Triton `--log-verbose=0` in `docker-compose.yml:116` (B12); update `tests/test_compose_contract.py`.
5. Parent plan Wave 1 items 1 and 2 (crop view instead of `crop_pil` copy, one gray plane for blur)
   only if WP-1.2 shows them on COCO (B6 says crop is 0.6 ms: likely not; they matter on set B).

Gates: P-json, P-pe bit-identical. Acceptance: API CPU per image >= 40 % lower than the WP-1.1
re-baseline at the same img/s; img/s not lower; p99 per batch not worse by > 10 %; event-loop
blocking test (a slow fake preprocess does not delay a concurrent `/health`). Rollback: revert per
commit.

### WP-1.4 VLM worker idle polling (#207). Tier: Sonnet.

Evidence: B9 (3.19 cores idle with 18 projects vs 0.47 with the worker stopped). Touch points:
`scripts/curation/vlm_worker.py` (producer `:412-463`, consumer `:465-510`, `label_batch` `:284-304`),
`scripts/curation/_project_worker_utils.py` (`unpaused_projects` `:42`), route
`/vlm/endpoints/active` (`src/routers/curation/vlm_activation.py:83`, already in
`contracts/openapi/curation.json`), `src/services/labeling/vlm_endpoints.py` (`active_vlm_endpoint`
`:436`, `vlm_configured` `:460`).

Steps:
1. Per cycle, before any pending fetch, resolve each candidate project's activation with the
   existing GET route (`.../vlm/endpoints/active`; `active.name` null means off), cached per project
   for `poll_interval` x 6. Skip projects with no active VLM; never fetch pending items for them.
2. When no project has a VLM, back off exponentially from `poll_interval` to 60 s; reset on any
   activation change.
3. Consumer: treat 409 `vlm_not_configured` and `vlm_endpoint_unavailable` like `no_classes`: drop
   the chunk, mark the project inactive in the cache, log once per state change (no per-chunk
   "HTTP error" lines).
4. If the GET route is too heavy per project, add a deployment-level "any project active" check to
   the existing projects listing instead of a new route (prefer no new route; if one is needed, run
   `make contracts`).

Tests: `tests/curation/test_vlm_worker_idle.py` with a fake API (httpx MockTransport) counting
calls: zero `label_batch` calls and zero pending fetches when no project is active; calls resume
within one cycle after activation; backoff caps at 60 s. Mutation: remove the activation check.
Live check (isolated stack, 18 projects of the pinned set, no VLM): idle cores api <= 0.4,
opensearch <= 0.2, vlm-worker <= 0.05 (the "stopped" row of B9). Rollback: revert.

### WP-1.5 Cluster training off the ingest critical path, full work kept (#212). Tier: Opus writes a 1-page sub-plan in the issue, Sonnet implements.

Evidence: B8 and the daemon finding in section 1. Owner constraint: no cut in clustering quality
or work; method-pluggable; measurable. Touch points: `scripts/curation/cluster_refresh_daemon.py`
(`_trigger_auto_label` `:139`, `_iteration` `:151`, args `:76`), `src/routers/curation/pipeline_start.py`
(`/pipeline/auto_label/start`), `src/services/curation/autolabel/job.py` (`start_job` `:422`),
`scripts/curation/auto_label_worker.py` (`_main_loop` `:414`), `src/services/curation/ingest_index.py`
(ingest-time assignment `:220`), `src/services/curation/clustering/ivf_ingest.py`.

Design (scheduling only):
1. The daemon starts the **background job** (`.../pipeline/auto_label/start`) instead of the
   synchronous route, so training runs in the auto-label worker, not in an API worker. One job per
   project at a time (the job module already serializes; verify and test).
2. Debounce: while a project's item count grew in the last `interval` (ingest active), the daemon
   skips the full retrain; ingest keeps assigning new residuals to the persisted centroids in O(1).
   When growth stops for one interval (quiescence) and the growth since the last train is >= the
   threshold, it starts one full retrain.
3. Daemon growth state persists across restarts (today in memory: a restart retriggers over the
   whole pool) in the project's settings doc via the existing revisioned store.
4. Cluster stage durations (fetch, train, assign, write-back) are recorded by the job (they exist in
   the auto-label status `stage_durations`) and scraped by the baseline suite.

Equivalence gate: same pool, same method, same parameters, same seed => identical `cluster_id`
per item to the sequential run (the IVF sample uses `default_rng(42)`; verify determinism on CPU
first, 3 runs). Acceptance: identical final assignment; API CPU during ingest lower (B5 upload
range 0.30-0.95 suggests interference); "ingest start to clustered" wall time not longer than
ingest + B8; method selection unchanged. Rollback: daemon flag back to the synchronous route
(keep one code path: the flag is removed after acceptance, no shim).

### WP-1.6 INT8 PE evaluation on Ampere, measurement only (#213). Tier: Opus designs the quantization recipe in the issue, Sonnet runs it.

Context: RTX A6000 and the 12 GB RTX 3080 Ti are both SM86 (Ampere): INT8 tensor cores exist (spec
peak about 2x FP16), **FP8 does not** (needs SM89 or later). TensorRT 11 removed the FP16 builder
flag (`trt_utils.py` `enable_fp16` is a no-op there); assume implicit INT8 calibration is gone too
and use an **explicit Q/DQ ONNX** (verify against the TensorRT version in the Triton image). The
detector research (`docs/research/detector_quantization_results.md`) shows TensorRT rejects
asymmetric zero points: use symmetric activations. ViT-style encoders lose accuracy in INT8 when
softmax, LayerNorm and GELU are quantized: keep them FP16, quantize MatMul/Conv only (variants
below).

Steps:
1. Calibration set: 1,000 crops from 300 COCO train2017 images **not** in the pinned 2,000 (draw
   with `fetch_coco_subset.py --bench-set 300` and a different seed, exclude pinned ids; manifest
   in `artifacts_local/`, never committed), cut with the production crop function.
2. Variants: (a) FP16 reference (`ref-fp16`), (b) INT8 Q/DQ on all MatMul with per-channel weights,
   entropy calibration, (c) as (b) but the first and last blocks and the projection head in FP16,
   (d) weight-only INT8 if the TensorRT version supports it for MatMul.
3. Build each as a separate model version directory on the isolated stack (never in `models/` of
   the live tree).
4. Accuracy on the pinned set: P-pe cosine vs FP16 (p01, median), kNN overlap@10, cluster agreement
   (section 2.1), near-duplicate decisions at 0.98, and task-level: kNN class precision@10 using
   COCO ground-truth classes.
5. Speed: perf_analyzer batch 1/8/16/32, concurrency 1:16, plus the full ingest protocol for the best
   variant; engine size and GPU memory. On the 12 GB card only with the owner's yes (O3).

Decision table (owner decides; nothing ships in v0.6.0 without it):

| Result | Action |
|---|---|
| p01 cosine >= 0.995 and all downstream gates pass and >= 1.3x PE throughput | Offer INT8 as a per-deployment option (stamped precision), default stays FP16 |
| Gates pass only for the 12 GB card's memory need, < 1.3x speed | Option for small GPUs only (#42) |
| p01 < 0.995 or cluster/kNN gates fail | Record the numbers in `docs/PERFORMANCE.md`, do not ship; a changed space would need the parent plan's Wave 10 migration |

Expected (**estimate**, low confidence): 1.2-1.6x PE throughput on SM86 for variant (c); accuracy
risk is the deciding factor.

### WP-1.7 Response serialization (#153). Tier: Sonnet.

Replace `ORJSONResponse` defaults with response-model serialization (about 15 routers). Measured
with the single-image endpoint phase of the baseline suite (B10) and ingest; must be P-json
identical. Low priority inside Wave 1; can run in parallel with WP-1.3.

### Wave 1 exit criteria

All Wave 1 WPs merged, re-baseline recorded ("v0.6.0 Wave 1" table: img/s, PE ms/image, API CPU
s/image, idle cores, cluster wall), reference capture `ref-fp16` stored. Expected (estimate):
ingest 14-18 img/s at policy `all`, API CPU <= 0.2 s/image, idle cores with no VLM <= 0.5.

## 5. Wave 2: the data path (wire, transport, decode once, resident tensors)

Goal: after Wave 1 the GPU is faster and the host share grows; remove host copies and bytes in
the downstream order of section 2. Ordering decision and why:

1. **Detection input first (WP-2.1 part A)** because crops derive from boxes: once crop boxes are
   proven identical, every later embedding gate measures only the embedding change.
2. **Embedding input next (WP-2.1 part B, WP-2.3)** with boxes frozen.
3. **Transport (WP-2.2)** after uint8, because uint8 already cuts bytes 4x and shared memory then
   only removes copies; measuring in this order shows what each step is worth.
4. **GPU decode-once and resident crops (WP-2.4)** last and conditional: it changes decode, resize
   and crop numerics at once (nvJPEG IDCT, GPU interpolation), so it needs every earlier gate and
   reference in place, and on COCO decode is only 4.6 ms (B6).
5. **Faces/OCR (WP-2.5)** are planned now and built after the transport that ships is known; ingest
   does not call them (B: "Face and OCR models are measured only through their endpoints").

### WP-2.1 uint8 inputs with in-graph normalization (#214). Tier: Opus reviews the export diff, Sonnet implements.

Evidence: B7 (14.9 MB/image FP32). Touch points: `export/export_models.py` (detector), the YOLO26
exporter `export/export_yolo26.py`, `export/export_pe_image_encoder.py`, `export/build_pe_trt.sh`,
`models/yolov11_small_trt_end2end/config.pbtxt` (input FP32 [3,640,640]),
`models/pe_image_encoder/config.pbtxt` (FP32 [3,336,336]), new version directories `2/` selected by
`version_policy`, `src/services/curation/ingest_detect.py` (`:132`, `:182`, `:333`, `:374`),
`src/clients/pe_encoder.py` (`encode_images`), `src/services/detection/pe_preprocess.py`
(`normalize_chw` becomes in-graph), `scripts/lib/model_setup.sh` (stamp includes input dtype).

Steps: Part A detector: uint8 NCHW input (letterbox stays on CPU), `/255` in graph; gates P-box,
P-item, class names identical, crop rects identical on 100 % of items (if not, run every
downstream gate). Part B PE: uint8 input after the CPU resize/center-crop, `(x/255-0.5)/0.5` in
graph; gates P-pe/P-pew vs `ref-fp16` (expect cosine >= 0.9999: same math, FP16 rounding moves),
kNN overlap, cluster agreement. Core routes (`/detect`, `/embed`, MobileCLIP) follow the same
pattern only if they share the model; MobileCLIP is a separate WP if B10 shows it matters.

Acceptance: request KiB/image (baseline suite) >= 70 % lower; API CPU per image lower; img/s not
lower. Rollback: `version_policy` back to version 1, reload model.

### WP-2.2 Triton system shared memory for the API hop (#215). Tier: Opus writes the isolation sub-plan, Sonnet implements.

No shared memory is used today (tensors inline in gRPC, `src/clients/triton_pool.py`). Design: a
per-API-worker pool of system shared memory regions (`tritonclient.utils.shared_memory`),
registered with Triton once per worker, sized for the largest batch (uint8 after WP-2.1: PE 32 x
336 x 336 x 3 = 10.8 MB; detector 16 x 640 x 640 x 3 = 19.7 MB), outputs also in regions. Requires
a shared `/dev/shm` IPC namespace between `api` and `triton-server` (compose `ipc:` or a shared
tmpfs volume): add to `docker-compose.yml` only, and region names prefixed with
`COMPOSE_PROJECT_NAME` so isolated projects on one host never collide. Setting `ingest_transport`
(`grpc | shm`, default `grpc` until gated). CUDA shared memory is out: the API and Triton may sit on
different GPUs (parent plan section 9).

Gates: P-json identical (transport only). Acceptance: API CPU per image lower by a measured amount
(record it; expected small after uint8, **estimate** 10-30 ms/image); no leaked regions after 3 runs
and an API restart (`/v2/systemsharedmemory/status` before and after); `tests/test_compose_contract.py`
updated. Rollback: setting to `grpc`. If the win is < 5 % of API CPU, close the WP as measured and
not adopted.

### WP-2.3 Cross-image PE batching (#216). Tier: Sonnet.

Touch points: `src/services/curation/ingest_batch.py` (`run_ingest_batch` `:129`), `ingest_index.py`
(`embed_items` `:118`, `index_items` `:152`), `pe_encoder.py` (`embed_crops`, `embed_whole_frame_bytes`),
`src/services/curation/reprocess_embed.py`. Gather the crop tensors (after the embedding policy
selects them) and whole-frame tensors of all images in a batch request, run chunks of the engine
max (32), scatter back in order; the same function serves embed-missing so "embed all" now and
"embed later" produce the same vectors at the same speed (owner point 3). Gates: P-pe, P-pew
bit-identical (batched TensorRT output was bit-equal in the research note; re-verify on FP16).
Acceptance: PE requests per image < 0.3 (from 2.07); img/s not lower; p99 not worse by > 10 %.

### WP-2.4 GPU decode once and a GPU-resident ingest pipeline (#217). Tier: Opus writes the sub-plan (parent plan Waves 6-7 plus C1-C3, C6), Sonnet implements.

Go/no-go measured after WP-2.1 to 2.3 on the pinned set and on set B (owner run): proceed only if
host work (decode, letterbox, preprocess, serialize) is >= 25 % of ingest wall time, or set B shows
the full-resolution cost of parent plan section 2.1. If go: parent plan Wave 6 (DALI decode and
static ensembles for stateless routes, bake-off A/B/C) then Wave 7 (`ingest_pipeline` BLS: decode
once, crops cut and resized on the GPU, one PE call per frame plus crops, return kilobytes),
composed with #67 (request-driven plans decide which branches run; answer C1 first). Backend
setting `ingest_backend` (`cpu | gpu`, default `cpu`). Gates: full chain of section 2 against
`ref-fp16` (P-box, P-item, P-pe and P-pew with the parent plan's GPU tolerances, cluster agreement,
P-blur). Stop-and-escalate if the BLS arm loses by > 15 % to the CPU-batched arm. Rollback: setting.

### WP-2.5 Faces and OCR plan (#220). Tier: Opus, plan file only.

Write `docs/design/faces_ocr_resident_plan.md`: uint8 OCR input instead of the full-resolution FP32
`original_image` (parent plan section 2.1 row 11), recognition batched by width bucket and argmax
in graph (parent Wave 5), face alignment on resident frames, gates P-ocr and ArcFace cosine. After
the WP-2.4 go/no-go.

### Wave 2 exit criteria

Request bytes per image >= 70 % lower than Wave 1; API CPU per image at its Wave 2 target (set
from the WP-1.2 table, not guessed here); img/s at least the Wave 1 number (the GPU is the limit;
the win of Wave 2 is headroom and latency, stated as such); all gates green against `ref-fp16`.

## 6. Wave 3: clustering, full work, faster and measurable

Owner constraint: full clustering work after ingest stays (several methods under evaluation for
semi-supervised self-clustering); no subsampling beyond today's `IVF_MAX_TRAIN_SAMPLE` (50k), no
fewer iterations (niter 50, nredo 5, K 512 in `methods/ivf.py:59-76`).

### WP-3.1 Clustering benchmark, GPU IVF, incremental (#218). Tier: Opus writes the method-harness interface, Sonnet implements.

1. `scripts/bench/cluster_bench.py` (pure logic in `scripts/bench/cluster_bench_lib.py`, tests
   `tests/test_cluster_bench.py`): input a fixed embedding dump (pinned set residual pool, exported
   once via the existing PIT fetch `fetch_residual_embeddings_parallel`, plus the 10k pin); runs any
   method in the registry (`src/services/curation/clustering/methods/__init__.py`) with fixed seeds;
   reports per stage (fetch, train, assign, write-back measured separately from the live job's
   `stage_durations`), objective, seed stability (3 seeds), CPU-vs-GPU agreement and purity (section
   2.1), peak RAM/VRAM. New methods plug in by registering; the harness needs no change.
2. Decouple FAISS GPU from the cuML probe: `ivf.py` uses `faiss.get_num_gpus()` independently of
   `detect_cluster_backend` (cuML still gates UMAP/HDBSCAN/AHC kNN). Behind setting
   `cluster_faiss_gpu` (default off until the gate). Equivalence: GPU k-means is **not** bit-identical
   to CPU (float reduction order). Gate proposal for the owner (O2): centroid-assignment agreement
   (2.1 b) >= 0.98, objective within 0.5 %, purity not lower by > 0.5 points, auto-promote outcome
   identical on >= 99 % of items. If the owner requires bit-identity, GPU stays off and only the
   scheduling of WP-1.5 ships.
3. Write-back cost: the stage table will show whether the 2000-doc painless bulk dominates (12.8k
   items, B8); if so, parallel slices for the write-back (same guarded script, same results).
4. Incremental: assign-only (`assign_only_residuals`, `orchestrator.py:301`) between full retrains is
   implemented; measure it and document when the daemon of WP-1.5 uses it.

Expected (**estimate**): train stage from tens of seconds to a few seconds on GPU (research note:
GPU k-means about 12 s at 50k); total B8 bounded by fetch and write-back, measured in step 1.
Rollback: setting off.

## 7. Wave 4: VLM and region consumers

### WP-4.1 Work crops at model size (#219). Tier: Sonnet, from parent plan Wave 3.

Parent plan Wave 3 unchanged (`crop_max_side`, `vlm_crop_max_side`, C7 first), scheduled here so the
consumers change after embeddings and clustering are stable (rule D1). Gates P-vlm, P-sam,
P-region, P-ocr need set B (owner run). Public acceptance on COCO is byte accounting and unit tests.

### WP-4.2 Wheel teacher-student example (#84)

Runs after WP-4.1 as the public, reproducible segmenter-versus-detector comparison; its own issue
body is the plan.

## 8. Wave 5: training/export check, packaging

- **WP-5.1 Export parity.** Run export of a pinned project before Wave 1 and after Wave 4; the
  manifest (class names, boxes within 1 px, item set) must be identical except for deliberate
  label changes. Class identity is by name at every boundary. Tier: Sonnet. Issue: track under #40.
- **WP-5.2 12 GB builds (#42).** Workspace becomes a knob in `_ms_pe_trtexec` (hardcoded `8G`) and
  the exporters (hardcoded 4G/8G); stamp records it; INT8 result of WP-1.6 decides whether small
  cards get INT8 PE. Live suite on the 12 GB card with the owner's yes.
- **WP-5.3 Image slimming (#41).** After the optimization waves (owner roadmap).

## 9. Parallel track V: vector store evaluation (#221, related #56)

Measurement only, independent of Waves 1-4 except that the 1M-vector set is cheapest after Wave 1
(faster PE). Protocol, candidates, datastore requirements, split-architecture costs and decision
criteria: [v060_vector_db_evaluation_plan.md](v060_vector_db_evaluation_plan.md). The #56
quantization benchmark is its OpenSearch-tuning arm.

## 10. Work packages and issues

| WP | Issue | Wave | Depends on | Tier (plan / implement) |
|---|---|---|---|---|
| WP-0.0 merge baseline suite | PR #206, #45 | 0 | none | owner |
| WP-0.1 parity tooling | #208 | 0 | WP-0.0 | Sonnet / Sonnet |
| WP-1.1 FP16 PE, build stamp | #209 | 1 | WP-0.1 | Sonnet / Sonnet |
| WP-1.2 API CPU attribution | #210 | 1 | WP-0.0 | Sonnet / Sonnet |
| WP-1.3 API CPU fixes | #211 | 1 | WP-1.2 | Sonnet / Sonnet |
| WP-1.4 VLM worker idle | #207 | 1 | none | Sonnet / Sonnet |
| WP-1.5 cluster scheduling | #212 | 1 | WP-0.1 | Opus / Sonnet |
| WP-1.6 INT8 evaluation | #213 | 1 | WP-1.1 | Opus / Sonnet |
| WP-1.7 response serialization | #153 | 1 | WP-0.0 | Sonnet / Sonnet |
| WP-2.1 uint8 inputs | #214 | 2 | Wave 1 | Opus review / Sonnet |
| WP-2.2 shared memory | #215 | 2 | WP-2.1 | Opus / Sonnet |
| WP-2.3 cross-image PE batching | #216 | 2 | WP-1.1 | Sonnet / Sonnet |
| WP-2.4 GPU decode, resident ingest | #217 (#40, #67) | 2 | WP-2.1-2.3, go/no-go | Opus / Sonnet |
| WP-2.5 faces/OCR plan | #220 | 2 | WP-2.4 decision | Opus |
| WP-3.1 clustering bench, GPU IVF | #218 | 3 | WP-1.5 | Opus / Sonnet |
| WP-4.1 work crops | #219 | 4 | Wave 3 | Sonnet / Sonnet |
| WP-4.2 wheel example | #84 | 4 | WP-4.1 | Sonnet |
| WP-5.1 export parity | #40 | 5 | Wave 4 | Sonnet |
| WP-5.2 12 GB builds | #42 | 5 | WP-1.1, WP-1.6 | Sonnet |
| WP-5.3 slim images | #41 | 5 | Waves 1-4 | Sonnet |
| WP-V vector store evaluation | #221 (#56) | parallel | WP-0.0 | Opus / Sonnet |
| WP-S1 Triton and client sweep (study X-1) | #226 | 1 | WP-1.1 | Sonnet / Sonnet |
| WP-S2 PE engine efficiency (X-2) | #227 | 1 | WP-1.1 | Sonnet, Opus review / Sonnet |
| WP-S3 data-parallel PE across GPUs (X-3) | #228 | 2 (first) | WP-S1, owner GPU window | Opus / Sonnet |
| WP-S4 bulk ingest runner (X-4) | #229 | 2 (last) | WP-2.1-2.3, WP-S3 | Opus / Sonnet |
| WP-S5 datastore write ceiling (X-5) | #230 | parallel (track V) | WP-0.0 | Sonnet / Sonnet |
| WP-S6 ortloom-serve head-to-head (X-6) | #231 | parallel | WP-S2 | Sonnet / Opus decides |
| WP-S7 scale ladder 10k, 100k, 1M (X-7) | #232 | each wave exit | shipped levers | Sonnet, owner schedules |

Parallelism: WP-1.2, WP-1.4, WP-1.7 and WP-V can start at once after WP-0.0; WP-1.1 needs
WP-0.1's capture of `ref-v050` first (the FP32-sized plan must be captured before it is rebuilt).

## 11. Risks

1. **FP16 is already in place** (B3 inferred from size): then Wave 1's headline win disappears;
   WP-1.1 step 1 answers it before any code.
2. **FP16 changes the stored space** for deployments embedded with the FP32 plan: P-pe gate; if it
   fails, stop (re-embed decision = parent plan Wave 10).
3. **INT8 accuracy** on a ViT encoder: measurement only, owner decision.
4. **Host becomes the limit** after FP16: WP-1.2/1.3 sit in the same wave on purpose.
5. **Shared memory and isolation**: region names per compose project, leak checks, default off.
6. **GPU k-means non-determinism**: owner-defined gate (O2), default off.
7. **COCO hides pixel costs**: Wave 2.4 and Wave 4 gates need set B; public docs carry pass/fail.
8. **Benchmark contamination** by the live stack on the same host: load-average guard, GPU 0 only,
   foreign-SM sampling (already in the suite).

## 12. Owner decisions (2026-10-10)

| ID | Decision |
|---|---|
| O1 | Existing installs are told in the release notes to rebuild models once the FP16 fix lands; with the build stamp the rebuild is automatic on the next install. |
| O2 | GPU k-means (WP-3.1) is gated by agreement (cluster purity and label agreement against the CPU result on the public set), not bit-identity. Cluster training keeps its full work; methods stay pluggable because the owner is comparing semi-supervised and self-clustering methods. |
| O3 | Benchmarks may use any GPU that is free, including GPU 1 (the 12 GB card) for WP-1.6 and WP-5.2. Check `nvidia-smi` first and coordinate with the other repositories' agents (cloud, transcribe); never touch other projects' containers. |
| O4 | INT8 ships, if it passes its gates, as a per-deployment option; FP16 stays the default. |
| O5 | The 200-crop retrain threshold stays during v0.6.0 and is revisited with WP-3.1 numbers. |
| O6 | A second datastore service is acceptable if it wins on the criteria of the vector store plan; Milvus is a candidate the owner favours. Metadata stays in OpenSearch either way. |
| O7 | Public runs use public datasets only (COCO, Open Images, ImageNet). The owner runs private-data benchmarks separately on his own non-public data; none of it is committed. |
| O8 | One million images is the destination, not the first step: the work builds up the optimisation levers in order, measuring how fast this server can process a 1M-image batch with industry-standard practice. Scale points (2k, 10k, 100k, then 1M) are reached as levers land. |
| O9 | Embed-all must be fast: some datasets (vehicles) need every crop embedded to cluster and label, so the selective policy stays configurable and is never the only way to be fast. |
| O10 | Bulk ingest is a first-class feature, delivered both as a CLI and as an API route that the API clients and Cropwright can trigger (start, progress, pause/resume, cancel), built on one shared runner so the two entry points cannot diverge (WP-S4, #229). The route is a curation job route with a served contract (`make contracts`), not a side door. |
| O11 | Triton is made to work properly first. ortloom / ortloom-serve is a later task: the head-to-head (#231) and any ortloom generic tensor backend wait until the Triton hardening waves have landed. |
| O12 | Target for 1M images at embed-everything is set by measurement as levers land; the study's 4-6 h figure is an estimate, not yet a commitment. |

## 13. Adjustments from the serving backend trade study (2026-10-10)

Source: [v060_serving_backend_trade_study.md](v060_serving_backend_trade_study.md) (roofline
ceilings, trade matrix, experiments X-1 to X-7). The owner decisions of section 12 are unchanged;
this section adds work packages and cross-references only.

Findings that change the ordering:

1. **PE is 99 % of the GPU arithmetic per image** (384 GFLOP per embedding, counted from the ONNX
   graph; the detector is 22 GFLOP). After FP16 (PR #225: 17.4 img/s, GPU 74 % busy) the PE engine
   runs at about 45 % of the A6000's FP16 peak, near the best published TensorRT ViT-L point on that
   GPU. The one-A6000 ceiling at policy `all` is about 23.5 img/s (estimate).
2. **The second A6000 is the largest single lever** (about 2x); every model is pinned to GPU 0
   today. It moves ahead of the transport packages for images/s; WP-2.1/2.2 remain for host CPU and
   latency.
3. **Triton stays the model server.** ortloom-serve serves YOLO detection only today and cannot beat
   Triton on a compute-bound ViT in the same TensorRT kernels; its JPEG-in decode pipeline is a
   candidate arm for WP-2.4 on high-resolution photos, decided by WP-S6.

Placement:

| Wave | Added | Note |
|---|---|---|
| 1 | WP-S1 (#226) Triton and client sweep, WP-S2 (#227) PE engine efficiency | after WP-1.1; WP-S1 tells whether WP-1.2/1.3 must land before more GPU work (GPU busy < 85 % at every arm = host binds); parent plan Wave 4 config items are executed inside WP-S1 |
| 2 | WP-S3 (#228) data-parallel PE across GPUs as the **first** Wave 2 item; WP-S4 (#229) bulk ingest runner as the **last** | WP-S3 needs the owner's GPU 2 window (study Q1); WP-S4 depends on WP-2.1 to 2.3 and WP-S3 |
| parallel (track V) | WP-S5 (#230) datastore write ceiling and cluster pass at scale | shares the recall harness of WP-V (#221); feeds WP-3.1 (#218) with the 100k cluster stage table |
| parallel | WP-S6 (#231) ortloom-serve head-to-head | after WP-S2 so Triton is measured tuned; decision rule in the study section 10 |
| each wave exit | WP-S7 (#232) scale ladder 10k, 100k, 1M | implements O8; 1M from Open Images |

Changes inside existing packages:

- **WP-1.6 (#213).** TensorRT 11 removed implicit INT8 calibration and the precision builder flags;
  use NVIDIA Model Optimizer explicit Q/DQ with FP16 as the high-precision type. A public A6000
  measurement on CLIP ViT-L shows 1.33x at batch 8 with correct Q/DQ placement and a 2x slowdown with
  Q/DQ on Transpose outputs and Add inputs (study [S18]); the variant list keeps (c) as the first
  candidate.
- **WP-2.2 (#215).** Stays system shared memory. Triton 26.06 and 26.09 list a known issue with
  `tritonclient` CUDA shared memory in multithreaded clients (study [S9]), which confirms the earlier
  exclusion of CUDA shared memory.
- **WP-2.4 (#217).** The bake-off adds an nvImageCodec batched-decode arm and ortloom's nvJPEG
  pipeline as an external reference (study X-8 note). GA102 has no hardware JPEG decoder; DALI's
  hybrid Huffman path for images above 1 MP keeps a CPU cost (study section 3.4).
- **Benchmark protocol (section 0 rule 3).** Every GPU run also records `nvidia-smi dmon -s pucv` so
  power-capped clocks (300 W per A6000) are visible; trtexec numbers from TensorRT 11 (CUDA graphs on,
  transfers off by default) are never compared with Triton end-to-end numbers.
