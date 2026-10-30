# Improvement backlog and lessons (state at the end of the 0.7.0 cycle)

Written 2026-10-30. One page: what to do next, why, where the evidence is, the issue that tracks it,
and a rough size and risk. Items marked **to file** have no issue yet. Nothing here is a commitment;
the research notes decide, the owner orders. Evidence links are repo paths; numbers are in the linked
notes, not repeated here.

Size: S is a day or less, M is a few days, L is a wave. Risk is the chance the item does not pay off
or breaks something that works.

## 1. Clustering and labeling

| Item | Why | Evidence | Issue | Size / risk |
|---|---|---|---|---|
| **Z5: detector-class roots with a hold-back guard in the auto-label loop** | The one real lead of the cluster program: class roots needed about a third of IVF's labelling decisions at the same label precision, and the label-free guard removed most of the open-set exposure. The open-set bar could not be read (2 of 20 seeds reached the registered minimum), so IVF stays until it is re-run. Repeat Z4 on a set with at least 1,200 open-set crops, then integrate into the loop only if a bar passes | [`cluster_config_z4.md`](../research/cluster_config_z4.md), [`cluster_program_outcome.md`](../research/cluster_program_outcome.md) section 6, `ship_track_plan.md` section 7 | to file (the loop driver is #286) | M for the experiment, M for the integration; risk medium (open-set animals the detector cannot name) |
| **Fine-grained make and model** | The judge is reliable at make level only; model level needs a person. The labels of Commons set C come from category names and were only machine-reviewed, so every model-level figure is provisional. Set C labels are weak. Needs a person-checked label set (FG-0) before any model-level claim or product work | [`fine_grained_benchmark_results.md`](../research/fine_grained_benchmark_results.md), [`visual_judge_e0.md`](../research/visual_judge_e0.md), [`cluster_config_decision.md`](../research/cluster_config_decision.md) 0A.3 | #385 (public proxies and licences); person-checked labels: to file | L; risk high until labels exist |
| **Faces plan** | Faces are a core-API index outside project isolation; union-find at similarity 0.7 chains and loses corrections; Immich's defaults (distance 0.5, minimum 3 faces) and weaknesses are written up, with a plan for prototypes, review bands, stable person ids and a purge | [`cluster_search_makemodel_plan.md`](cluster_search_makemodel_plan.md) section 3 (Immich comparison, scope decision O1) | #353 (FAISS indexes not reloaded after restart), #354 (singletons, destructive re-run, dense-matrix memory), #379 (embeddings-off and purge) | L; risk medium; do #353, #354, #379 first (S to M each) because they are bugs and privacy controls, not research. No face clustering experiment has been run |
| **Attribute signals** | Plates, burst time, near-duplicate groups and detector class decide cases embeddings cannot. Rule from the plan: no single attribute decides a merge or split alone | plan section 6.7 | to file | L; risk medium (cost per signal differs; EXIF time is a header read) |
| **Registration-plate identity text** | A generic "identity text" region (plate, serial number, bib) whose reconciled OCR and VLM text links same-object items. Plate region boxes and the readers exist; the key and the rules do not | plan section 6.8 (wave W12) | to file | L; risk medium; plate sheets never go to a hosted model |
| **Leiden and GPU clustering (parked)** | Works on synthetic data; on real crops only a strict filter beat the baseline on both axes. Calibration of gamma and the filter floors per embedding is the open problem; nothing above 20k on GPU, no 1M run | [`leiden_refinement_prototype.md`](../research/leiden_refinement_prototype.md), [`gpu_clustering.md`](../research/gpu_clustering.md), outcome section 7 | #342 to #346 | L; risk high; do not start without a held-out set and threshold calibration |

## 2. Embeddings and search

| Item | Why | Evidence | Issue | Size / risk |
|---|---|---|---|---|
| **PE-Core-L stays** | No challenger passed the registered Recall@1 gain on CARS196, CUB or SOP. PE-Core-G is best on all three but costs about eight times the encode time and five times the memory. The set C ranking (MobileCLIP2 first) did not hold on the standard benchmarks | [`wb2_standard_benchmarks.md`](../research/wb2_standard_benchmarks.md) | #347 (evaluation of a newer text-and-image embedding model), #365 (model swap with a dual-field migration) | Do not swap. Any new candidate needs a text tower, because search uses it (4 of the 5 encoders tested have one). The registered sanity check of WB2 was not met, so its comparisons are exploratory |
| **Semantic search over crops** | Item and box vectors exist. `GET {P}/search/text` queries item vectors only; `{P}/search/similar`, `{P}/search/image` and a `target=item|box|frame` option are proposed, none built. Cheapest first step is "more like this" from a stored vector (no encode) | plan section 4 | to file | M; risk low. Related: #384 (a `class_name` filter makes text search 4 to 5 times slower), #356 (about 1.5 s outlier on 1 in 10 queries) |
| **Search filter cost** | A class filter slows text search; the unfiltered path is fast | issue text | #384, #356 | S to M |

## 3. Ingest speed

| Item | Why | Evidence | Issue | Size / risk |
|---|---|---|---|---|
| **Merge the ingest-speed work** | The stage counters on main can fail on Triton 2.70 (see the caution on the stage timing page); the fix and the SP0 measurements are on `perf/v070-ingest-speed`. `run_baseline.py --clients` is also there | [`ingest-stage-timing`](../../docs-site/docs/operations/ingest-stage-timing.mdx), [`sp0_review.md`](../reviews/sp0_review.md) | #390, #392 | S to merge; do this before trusting any stage number |
| **Batched CUDA kernels (design C2)** | Launch-bound stages (letterbox, crop squares, device-to-host copies) are suspected to be host-side serialisation; the plan is one fused kernel and one copy per image. The design document `sp_gpu_prep_concurrency.md` is not on `main` (on branch) | issue #392 text; branch `perf/v070-ingest-speed` | #392 | M to L; risk medium; measure with `STAGE_SYNC=1` first |
| **CUDA MPS as an opt-in** | Two Triton instances on one GPU may be time-slicing without MPS; MPS needs root on the host and adds a daemon and failure coupling, so it can only be an opt-in with a documented setup | [`v060_serving_backend_trade_study.md`](v060_serving_backend_trade_study.md) (MPS entries); not isolated by any measurement | #392; opt-in switch: to file | M; risk medium (host root, GeForce support unconfirmed) |
| **Triton memory at start** | One install held 36.5 GB at start against 23.1 GB on another; this decides whether two `gpu_image_prep` instances fit a 48 GB card | issue text | #393 | S to investigate; blocks the installer's profile choice on 48 GB cards |
| **Geometry stage is the largest cluster-pass stage** | After the IVF fit was shown to be flat above the training cap, the geometry stage was not measured in the ladder. It is reported as now the largest stage; no note on `main` carries that number yet | [`v070_measurement_log.md`](../research/v070_measurement_log.md) "Open issues" (#337 asked for it) | #337 | M; risk low; write the measurement before optimising |
| **Remaining stage 2 items** | Set B photos, minimal and standard profiles, two-card sweeps, CUDA graphs, engine stamps | issue text | #268 | M each |

## 4. Product loose ends

| Item | Why | Evidence | Issue | Size / risk |
|---|---|---|---|---|
| **Label operations UI** | The API (list, dry-run undo, undo) is on `main`; Cropwright does not show operations yet | [`cq3_label_operations.md`](cq3_label_operations.md) section 10; [label operations guide](../../docs-site/docs/guides/label-operations-and-undo.mdx) | UI3 and UI4 of the ship-track plan; to file if not tracked | M; risk low |
| **Thumbnails return 404 on older projects** | On two older projects of one stack every crop thumbnail returned 404 because the sample-images mount no longer existed in the API container. The frontend now shows a dimmed empty tile; the files need restoring, or the project needs a regenerate-thumbnails path | [`cropwright_health_2026-10-29.md`](../research/cropwright_health_2026-10-29.md) | to file | S; risk low |
| **Live write-path suite failures on main** | Eight failures after a wipe | issue text | #397 | S to M |

## 5. Judge and contact-sheet protocol

The visual judge (Haiku) is research tooling only. Its "same" answer is reliable and its "different"
answer needs a person; on Open Images vehicles it failed the registered bars (pair kappa 0.611, flag
precision 0.454). The local VLM judged at near chance on COCO crops, so it may rank clusters for review
and nothing more. The layout calibration was vacuous (the judge said "different" to every sheet), so no
default sheet layout was chosen. Hosted judges are refused for face, identity-text and plate sheets
before any call. Next: the 5 percent human re-judge and the FG-0 label check, both for a person, then
a layout calibration with sheets that include known-same pairs. Evidence:
[`visual_judge_e0.md`](../research/visual_judge_e0.md), outcome section 8. Issue: to file.

## 6. Lessons learned

- **Synthetic data misleads.** Leiden recovered planted partitions and then barely beat the baseline on
  539 real crops; the k-means fit was 2.3 times slower on real vectors than on a synthetic mixture of the
  same size. A quality or speed claim needs real photos. The scorecard refuses quality fields on a
  synthetic dataset for this reason (`scripts/bench/scorecard.py`).
- **Weak labels mislead.** The encoder ranking from Commons set C (category-name labels) did not survive
  the author-labelled benchmarks. Check labels before comparing models on them.
- **Test on standard benchmarks.** CARS196, CUB-200-2011 and SOP have published numbers and were the only
  fair check of the encoder question. Even they were labelled exploratory here because the registered
  sanity check failed.
- **Register the bar before the score.** Bars were committed before scoring in W0e, Z4, WB and WB2, and
  none was moved. Z4's open-set bar was still unreadable: register a minimum sample size per bar too.
- **CI-only float noise.** Tolerances that pass on one machine can fail on another (TensorRT kernels,
  batching and concurrency change embeddings at the level of run-to-run noise even with the same inputs;
  see parity in the measurement log, E4). Pin the comparison to cosine and sorted outputs, not bit equality.
- **GPU sharing etiquette.** Several sessions share three GPUs. Check `nvidia-smi` first, use the
  documented lock, give the stack its own compose project name (`docker compose -p`), keep to the agreed
  cards, and never act on the live stack. A measurement taken on a busy host is an upper bound: record the
  load average and mark the row noisy (the harness does).
- **Wait-loop pitfalls.** Poll a health route with a timeout and a marker, never `pgrep -f` or `pkill -f`
  on a string that is also in your own command line (it matches itself), and never a blind `sleep` before
  a check. Time `/health` before a run: a slow health route is the first sign of a loaded stack.
- **Counters can crash the server.** Custom Triton Python-backend metrics must be kept alive, dropped in
  the right order and incremented from one thread (#390).
- **Fail closed beats a quiet empty result.** A partial cluster pass used to report success (#357); undo
  re-decides each item inside the write so a person's edit wins.
