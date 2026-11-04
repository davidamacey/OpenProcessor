# OpenProcessor: clustering research program, umbrella plan (issue #346)

**Status: PLAN ONLY, NOT IMPLEMENTED.** Nothing in this gist has been built or run. It is the
umbrella plan for the clustering research tracker #346 and its family (#398, #416, #406, #347,
#365, #400, #401, #435, #436).

- Written 2026-11-04 against branch `int/b1-correctness` at `81c4e4ef` (read-only worktree).
- Every `path:line` below was read on that commit.
- Reader: a fresh agent with no memory of this session. This file sets the order, the ownership,
  the windows and the landing rules for the research lines. The per-experiment design lives in
  the repo design doc this gist builds on: `docs/design/research_clustering_plan.md`.
- Past results are cited only through the public notes in `docs/research/` and the backing JSON
  in `docs/benchmarks/`.

---

## 0. TL;DR

### What I recommend

1. **Use the repo plan as the design source; do not write a second one.**
   `docs/design/research_clustering_plan.md` was committed in `c053d123` on `int/b1-correctness`.
   It is not on `main` yet. It already holds:
   - the research questions RQ-1 to RQ-10;
   - the arms, with library citations;
   - the datasets, splits, metrics and judge procedure;
   - the pass bars, the 12 experiments, the three GPU windows and the stop rules.

   I checked it against the code; most of it holds. This gist is the overlay a tracker needs:
   - who owns which question (section 3);
   - the ordered program and a reconciled GPU, CPU and person calendar (sections 5 and 6);
   - the corrections the plan and the issues need before work starts (section 9);
   - how a winner lands in the product after B2 (section 7);
   - the verification plan and the failure register (sections 11 and 12).
2. **The ship track comes first and stays small.** Run these first, because they are cheap and
   each closes an issue either way:
   - RQ-1 Z5: class roots with the hold-back guard, #398;
   - RQ-2: near-duplicate thresholds, #416;
   - RQ-8: judge calibration and the human re-judge, #406;
   - RQ-10: frame-vector check, #435.
3. **Re-size Z5 before fetching anything.** The quota of "1,200 open-set crops" in #398 and in the
   plan is too small. At that size the guarded arm cannot be read, and the guarded arm is the one
   that decides adoption (section 4.1, correction C5). Register the readability per arm instead,
   and count the Open Images train CSV before downloading any images.
4. **Fix the ID collision before any window is announced.** The clustering research plan and the
   GPU-pipeline research plan (`docs/design/research_gpu_pipeline_plan.md`) both name their
   experiments `E-1...` and their windows `RW-1..RW-3`. In this gist the clustering windows are
   **CRW-1..CRW-3** and the GPU-pipeline items carry a **GP-** prefix (J3).
5. **Add the missing viewpoint line, or park it with a reason.** The owner asked on 2026-10-29 for
   two things:
   - an identity-versus-view model;
   - viewpoint evaluation strata.

   The cross-domain note's E4 experiment covers this: frozen PE-Core across azimuth bins on public
   multi-view data. The repo plan's RQ list dropped it. I propose a small RQ-11 inside CRW-2 (J4).
6. **Heavy work in the product becomes a Celery job after B2.** It never runs in a request
   handler. Person grouping is already scheduled as B2 family 18 (task 2C-6), so the research
   plan's "new issue N-1" is done as planning (C4).

### What I am NOT deciding

These are owner calls, listed in section 10:
- the GPU window calendar against v0.8.0 and the GPU-pipeline research (J3);
- whether the viewpoint line runs (J4);
- whether #435 is measured or closed now (J5);
- the leidenalg licence (J6);
- face numbers from LFW-family sets (J7);
- the Haiku and person-time budgets (J8);
- whether a hosted judge needs a provenance allowlist (J9);
- the parked privacy lines (J10);
- the Z5 re-registration (J2).

---

## 1. Context: why the obvious approach is wrong here

The obvious approach is to pick the next promising clustering algorithm and run it at scale.
The 0.7.0 research shows why that fails here (`docs/research/cluster_program_outcome.md`
sections 1 to 7):

- **The method was rarely the limit.** The numbers come from the outcome note and its sources:
  - IVF at K = n // 6 forces pair recall near zero by configuration
    (`clustering_research_consolidated.md` section 1).
  - Above pair recall of about 0.3, the neighbourhoods themselves are the limit: P@10 is 0.82
    overall and 0.74 for elongated crops.
  - Fine-grained (make and model) grouping is an encoder problem.
- **Every challenger failed a pre-registered bar or could not be read:**
  - whitening, k-reciprocal and DBA;
  - the novelty flag;
  - the judge-gated merge;
  - split by descent;
  - within-class roots;
  - HDBSCAN and FINCH.
- **The one lead is in the gate, not the method.** Detector-class roots need about 0.35x IVF's
  labelling decisions at the same precision. The open-set read (Z5) is still pending
  (`docs/research/cluster_config_z4.md`, "Verdicts").
- **Synthetic wins did not carry over.** Leiden CPM went from 0.08 to 0.79-0.86 pair recall on
  planted partitions. That did not reproduce on 539 real COCO crops beyond a filter tuned in-sample
  (`docs/research/leiden_refinement_prototype.md`).
- **A held-out half wears out.** The 3,000-item held-out half has been read by three waves.

So the program is a sequence of **pre-registered, stop-ruled reads on fresh real data**. It
starts with the cheap, directly shippable questions. A winner reaches the product only through a
Celery job, an OCC activation and a human review.

---

## 2. What is there today (verified in code)

### 2.1 Product clustering and grouping paths

| Path | What runs | Where (verified) |
|---|---|---|
| Curation items, residual pool | FAISS IVF. `n_clusters = min(512, max(1, n // 6))`; training sample capped at 50,000; niter 50 | `src/services/curation/clustering/methods/ivf.py:62-77`, `:121` |
| Curation region boxes | MiniBatchKMeans with `k = min(max(8, round(n / 800)), n)` | `src/services/curation/clustering/region_box_clustering.py:63`, `:120-126` |
| Curation crop near-duplicates | Exact blocked cosine plus union-find, run per IVF bucket. Default 0.98 from `OP_CROP_DUP_THRESHOLD`. The docstring calls the threshold "a hypothesis, not a finding" | `src/services/curation/item_scores/near_dup.py:1-27`, `:53-57`; `src/services/detection/frame_dedup.py:34`, `:60` |
| Curation frame near-duplicates | Same primitive at 0.98 on the frame `pe_embedding` | `src/services/detection/frame_dedup.py:34` |
| Core ingest near-duplicates | MobileCLIP kNN at 0.99, the default on every core ingest route | `src/routers/ingest/single.py:78-83`, `src/routers/ingest/batch.py:75-76`, `src/routers/ingest/directory.py:76-77`, `src/services/visual_search/ingest_batch.py:93`, `:323` |
| Core person grouping | Exact FAISS range search (`IndexFlatIP`) with connected components at cosine 0.7 (single linkage), behind one interface. Docstring: O(n^2 d), about 4 min at 200k | `src/services/face_identity/person_grouping.py:13-66`; route `src/routers/persons.py:166-210` (threshold 0.5 to 0.9, default 0.7) |
| Core whole-image clusters | FAISS IVF over MobileCLIP 512-d (`ClusterIndex` configs, `n_clusters=512`) | `src/services/visual_search/clustering_models.py:23-25` |
| Curation semantic search | `target=item|box`; no frame target, no `group_by` | `src/routers/curation/search.py:52`, `:79`, `:167`, `:203`; `src/routers/curation/search_image.py:29` |
| Frame vector (images index) | `pe_embedding` on the project's images index | `src/clients/curation_opensearch/bodies_core.py:73-105` |

### 2.2 Encoder preprocessing, the RQ-3a input

| Path | Preprocessing | Where |
|---|---|---|
| Whole frame (CPU) | Shorter edge to 336 with Pillow BILINEAR, centre crop, CHW uint8; normalisation inside the engine | `src/utils/pe_preprocess.py:1-19`, `:33-43`, `:64` |
| Crops (CPU) | `resize_crop_rgb` (resize shorter edge, centre crop) per crop | `src/clients/pe_encoder.py:93`, `:125-131` |
| Crops and frame (GPU) | One square per detection, cut from the full-resolution frame, plain bilinear "as `resize_crop_rgb`", through PE by BLS | `models/gpu_image_prep/1/model.py:12-14`; geometry `models/gpu_image_prep/1/prep_geometry.py:167` |
| Encode entry point a research script can call | `PEEncoder.encode_images(chws)` takes preprocessed CHW squares | `src/clients/pe_encoder.py:211` |

The consequence for RQ-3a is in correction C10. A preprocessing variant is tested by cutting the
squares in a research script and calling `encode_images`. It is never tested through a product
flag.

### 2.3 Research harness that exists (reuse, do not rewrite)

| Piece | Where | Notes |
|---|---|---|
| Partition metrics | `scripts/bench/cluster_eval_metrics.py:43` pairwise, `:99` moved protected items, `:139` BCubed, `:184` AMI, `:205` risk-coverage | No V-measure or purity helper yet: the plan's E-0 adds them through scikit-learn |
| Scorecard | `scripts/bench/scorecard.py:30`, `:67-94` | Refuses quality fields, strata and acceptance blocks on a dataset with `synthetic: true`. This enforces the real-data rule mechanically |
| Loop simulator, Z4 bars | `scripts/research/cluster_diag/loop_sim.py`, `z4_run.py`, `z4_bars.py` | Tests in `tests/research_scripts/test_loop_sim.py`, `test_z4_bars.py`. Twelve mutations were applied in place; all twelve are killed by the current tests (Z4 note, "Tests") |
| Chaining, held-out, acceptance | `scripts/research/cluster_diag/chaining.py`, `heldout.py`, `acceptance.py`, `y_acceptance.py` | |
| Loop partition extension point | `src/services/curation/assist_loop/partition.py:108-115` (`PartitionReader` protocol), `:139` (`ClusterFieldPartition`, the only implementation and the only reader of `cluster_id` in the loop) | RQ-1 and RQ-5 land as implementations of this protocol, not as a new loop |
| Leiden and GPU prototypes | `scripts/research/leiden_refine_proto.py`, `gpu_methods.py`, `graph_lib.py`, `bench_driver.py` | |
| Retrieval harness | `scripts/bench/retrieval_eval.py`, `retrieval_eval_metrics.py`, `retrieval_eval_data.py` | Pin `scripts/datasets/manifests/coco_bench_2000.json` |
| Visual judge | `scripts/research/visual_judge.py`, `judge_*.py`; sheet builder `src/services/labeling/contact_sheet.py` | Hosted refusal `scripts/research/judge_protocol.py:23` (`face`, `identity_text`, `plate`) and `:55-58`; test `tests/research/test_judge_protocol.py` |
| Fetchers | `scripts/datasets/fetch_cluster_eval.py`, `fetch_vehicle_eval.py`, `fetch_face_proxy.py`, `fetch_openimages_plates.py`, `prepare_labelled_benchmark.py` | The Open Images URL maps hold `validation` and `test` only (`scripts/datasets/fetch_openimages_plates.py:74-81`; `proxy_openimages.py:40`). **There is no train split** (Z5 needs it) |
| Pins | `scripts/datasets/manifests/` (`cluster_eval_3000.json`, `cluster_eval_fresh.json`, `openset_test_z4.json`, `vehicle_eval_8000.json`, `coco_bench_2000.json`, `oi_plates_300.json`, `openimages_faces.pin.json`, `lfw.pin.json`, `splits/`) | Every new set adds a pin here |
| Research dependencies | `scripts/research/requirements.txt`, `fine_grained_requirements.txt` (igraph, leidenalg, torch, open_clip, timm, transformers) | Not product requirements; tests that need them skip when they are absent |
| Kept benchmark data | `docs/benchmarks/cluster_eval/` (with `SHA256SUMS`), `wb2_standard_benchmarks/`, `retrieval_baseline/`, `cluster_config_baseline_3000.json` | Read only. **Never pruned** |

### 2.4 Program state that constrains the research

- `VERSION` is `0.6.0`. No v0.7.0 tag exists (`git tag -l` shows v0.4.1, v0.5.0, v0.6.0).
- The research plan's header says research starts after the v0.7.0 tag. Only the bar registration
  (R0, docs and scripts) may start earlier.
- The 0.7.0 program puts #346 and the family outside its scope:
  - `docs/design/program/00_index.md:356-358`;
  - milestones in `docs/design/program/09_github_actions.md:633-652`.
- GitHub state today: milestones `v0.7.0` (#6) and `v0.8.0` (#7) only.
  - #347, #365, #398, #406, #416, #435 and #436 are on `v0.8.0`.
  - #346, #348, #400 and #401 have no milestone.
- B2 (Celery) is not merged. The task names in section 7 are B2's planned names:
  - `docs/design/program/03_b2_overview.md:160-205`;
  - `03b_b2_w2_job_families.md:111`;
  - `03c_b2_w3_beat_reaper_workers.md:33-40`.
- Program GPU budget: exactly three windows inside v0.7.0 (`00_index.md:289-303`), none for
  research.
- v0.8.0 has its own windows V0, V1, V-EM and V2 (`docs/design/v080_platform_breadth_plan.md`
  section 3.3).
- The GPU-pipeline research has three more windows of about 30 GPU hours
  (`research_gpu_pipeline_plan.md` section 4.3).

---

## 3. Ownership: which issue owns which question

Research questions and experiment numbers follow `research_clustering_plan.md` sections 2 and 5.
"Window" uses this gist's CRW names (J3).

| Issue | Question | Experiment | Window | Ship bar (plan 4.6) | Outcome either way |
|---|---|---|---|---|---|
| **#398** class roots + hold-back guard | RQ-1: does `key` or `key_guard` keep open-set propagated precision >= 0.90 with >= 30 propagated per run, plus U2 >= 0.95 and U1 <= 0.5x IVF? | E-1 (Z5), then Z5-I (integration) | CRW-1 + CPU slot | Z4 bars unchanged; adopt the cheaper arm that passes | PASS: Z5-I lands in the loop's partition reader. FAIL or INCONCLUSIVE at the re-sized quota: IVF stays, #398 closes with the note |
| **#416** near-duplicate threshold | RQ-2: per path (core MobileCLIP, curation frame PE, curation crop PE), which threshold and linkage catches re-encodes without grouping distinct photos? | E-2 | CRW-1 + CPU slot | False-group rate <= 0.5 % on distinct pairs; recall >= 0.90 on JPEG q60 and half size; >= 0.80 on 90 % crop | New per-path defaults (`OP_CROP_DUP_THRESHOLD` default, core route defaults), or "keep, documented trade-off" |
| **#406** judge calibration, 5 % re-judge, FG-0 label check | RQ-8: per kind, is the judge trustworthy, at which layout, and does it agree with a blind person? | E-10 | Haiku half in CRW-1; local half in CRW-3 | Pair kappa >= 0.7, flag precision >= 0.8, human-vs-judge kappa >= 0.7, non-vacuous layout calibration | Judge per kind: authority or "ranks only"; #406 closes |
| **#347** EmbeddingGemma 2 and reference encoders | RQ-3b: does any encoder pass the RQ-3a bar on real crops plus WB2 B1 on CUB and SOP, has a text tower, and is within 2x encode time? | E-4 (after E-3) | CRW-2 | Plan 4.6 RQ-3b | #347 closes with a go or no-go. Owner rule: if useful, it ships as an optional encoder, off by default, never replacing PE-Core (#347 comment of 2026-10-29) |
| **#365** encoder swap, dual field | Landing of RQ-3b only | none of its own | v0.8.0 V-EM | `v080_platform_breadth_plan.md` section 2.9 gate | Unblocks only on an E-4 PASS; otherwise parked (v0.8.0 section 6) |
| **#400** attribute signals | RQ-6: must-link or veto evidence at a fixed false-merge rate | E-8 (feasibility first) | CPU slot | Plan 4.6 RQ-6 | Stays parked unless E-8 passes; E-8 runs only if E-7's machinery exists |
| **#401** plate identity | RQ-7: plate key precision >= 0.99 with privacy modes | E-9 (read quality only) | CRW-3 | Plan 4.6 RQ-7 | Stays parked; E-9 needs the owner's privacy-design yes first (J10) |
| **#435** `target=frame` | RQ-10: is the PE frame vector within 0.02 recall@10 of MobileCLIP on the 2,000-image pin? | E-12 | CRW-1 | Plan 4.6 RQ-10 | Within: a frame route can be sized (v0.8.0 Q-6). Loses: close as "whole-image search stays in the core API" (J5) |
| **#436** `group_by=image` | **Not research.** It is product work in v0.8.0 F2 (`v080_platform_breadth_plan.md` section 2.2, task SR-1, spike SP-1) | none | none | n/a | Leaves the research family (C12) |
| **#346** tracker | RQ-4 (global refinement), RQ-5 (stable partitions), RQ-9 (faces) and RQ-11 (viewpoint, proposed) until they get their own issues | E-5, E-6, E-7, E-11, E-13 (proposed) | CRW-2, CRW-3, CPU | Plan 4.6 | One results row per experiment in the tracker body |

New issues the plan proposes (`research_clustering_plan.md` section 6.5), with their status today:

| ID | Title | Status |
|---|---|---|
| N-1 | person grouping as a Celery job | **Already scheduled**: B2 family 18, task 2C-6 (`03_b2_overview.md:176`, `03b_b2_w2_job_families.md:111`, test `tests/test_persons_cluster_job.py` named at `03b:121`). Do not file it again (C4) |
| N-2 | `research(clustering): register bars and extend fetchers for the research waves` | To file: E-0 |
| N-3 | `research(embed): crop preprocessing (letterbox, context margin) on COCO and Open Images crops` | To file: E-3 |
| N-4 | `research(clustering): global refinement on a fresh held-out real set, 3k to 860k, with an agreement-gated GPU path` | To file: E-5, E-6 |
| N-5 | `feat(clustering): versioned partitions with stable ids, activation and merge/split suggestions` | File **only** on an E-5 or E-6 PASS |
| N-6 | `research(faces): identity grouping, calibrated incremental assignment and scale on public faces` | To file: E-11 |
| N-7 (new here) | `research(clustering): viewpoint-stratified identity on public multi-view data` | To file only if J4 = yes: E-13 |
| N-8 (new here) | `fix(curation): fail closed on an invalid OP_CROP_DUP_THRESHOLD` | Small fix, not research. `near_dup.py:53-57` returns the default on `ValueError` (C13) |

---

## 4. Requirement-by-requirement design notes

This section covers only the places where this gist adds to or changes the repo plan. Everything
else stands as written in `research_clustering_plan.md` sections 3 to 5.

### 4.1 RQ-1 / #398: Z5 sizing and the fetcher

**The data problem.** Z4 held 395 open-set crops in the held-out half. The quoted per-run
propagation is from `cluster_config_z4.md`, "Verdicts":

| Arm | Open-set items propagated per run (395 held-out) | Rate per held-out open-set crop | Held-out open-set crops needed for >= 30 per run |
|---|---|---|---|
| `key` | 19.9 | about 0.050 | about 600 |
| `key_guard` | 9.2 | about 0.023 | **about 1,290** |

The adoption rule picks the cheaper of `key` and `key_guard` that passes
(`cluster_config_decision.md` 0A.2). `key` already shows open-set precision of 0.733 against a bar
of 0.90, so in practice the guarded arm decides. The guarded arm reaches 30 propagated per run
only at about 1,290 held-out open-set crops. With a 50 % image split, that is about 2,600 in total.

The note's own estimate of "about 1,200 open-set crops (about 600 per half)" (Z4 section "What
follows", and #398's comment) matches the `key` arm only. A Z5 fetched at that size would leave
`key_guard` INCONCLUSIVE a second time. The plan's stop rule would then close #398 on a data
defect, not on the method.

✅ **Recommended (J2).**
- Re-register Z5 as a new registration, as the Z4 note requires.
  - Readability per arm: each arm's open-set bar is read only when **that arm** propagates at
    least 30 open-set items per run on average.
  - Quota target: at least 1,500 held-out open-set crops (about 3,000 in total). That leaves a
    margin over the computed 1,290, because Z4's exposure was heavy-tailed (0 to 72 per seed for
    `key`).
- Keep the Z4 bars unchanged.
- The original decision note registered 2,000 open-set crops in total (`cluster_config_decision.md:90`).
  Z5 asks for more and says why.

❌ **Rejected: keeping 1,200.** It is computed for the arm that is unlikely to pass.

❌ **Rejected: lowering the 30-per-run read rule.** That moves a bar after a read.

**Fetcher work (E-0, CPU).**

1. Add a `train` entry to the Open Images URL maps in `scripts/datasets/fetch_openimages_plates.py:74-81`.
   This is the map that `fetch_vehicle_eval.py` and `fetch_cluster_eval.py` import. The `--split`
   choices come from `sorted(BBOX_URL)` (`fetch_vehicle_eval.py:184`), so the new key reaches the
   CLI with no other change. Documented URLs (Open Images V7 download page, read 2026-11-04):
   - boxes: `https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv`;
   - images with rotation: `https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv`;
   - page: https://storage.googleapis.com/openimages/web/download_v7.html
2. **Count before downloading images.** Stream the train box CSV, which is multi-GB. Apply the same
   licence, rotation, depiction, group-of and area filters as Z4 (`cluster_config_z4.md` lines
   18-31). Print the open-set crop count per class. Commit that count in the registration before
   any image is fetched.
3. Exclude every earlier pin, as Z4 did (`--exclude-manifest` for `cluster_eval_3000`,
   `cluster_eval_fresh`, `coco_va_200`, `splits/coco_sample_200`, `vehicle_eval_8000`,
   `openset_test_z4`). Keep `disjointness.json`.
4. The open-set definition is by class **name**: classes whose names are absent from the
   detector's label space. Commit a COCO-to-Open-Images name map as data. No raw class index
   crosses a boundary.

**Red test for the fetcher:** `tests/test_vehicle_eval_selection.py` gains a case where
`--split train` builds the train URL. On the current tree it fails with an argparse "invalid
choice" error. The paired control: `validation` and `test` produce byte-identical URLs as before.

### 4.2 RQ-2 / #416: three paths, three spaces

- The three paths use two encoders: MobileCLIP 512-d (core) and PE 1024-d (frame and crop).
  One threshold never transfers between them. The plan already says this; keep it explicit in
  the note's tables.
- Crop near-duplicates run **per IVF bucket**, so cross-bucket pairs are never found
  (`near_dup.py:19-26`). E-2 must report that miss rate separately. Do this by running the same
  threshold globally (exact FAISS range search on CPU) and counting the pairs the bucketed run
  misses. Otherwise a "recall 0.90" can come from a global run that the product never performs.
- Complete linkage inside a candidate group is the documented way to stop chaining at lower
  thresholds. Use scikit-learn `AgglomerativeClustering(linkage='complete', distance_threshold=...)`
  (https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html).
  It runs on each union-find component, not on the whole pool.
- Landing: a default change and a re-score through the existing item-scores job. No new task.

### 4.3 RQ-3a / RQ-3b: where variants are cut

See C10. The research script:
1. reads the stored crop rectangles;
2. cuts each variant square (letterbox, 10 % or 20 % context margin) with Pillow `ImageOps.pad`
   or explicit crop arithmetic;
3. calls `PEEncoder.encode_images` (`src/clients/pe_encoder.py:211`) on the B5 stack.

No product flag and no second preprocessing path in `src/`. Only a PASS triggers a product change:
one change to `pe_encoder._preprocess_crops` and to the GPU prep geometry, gated by the B5 parity
gates (embedding cosine >= 0.999).

RQ-3b arms follow #347's owner comments:
- EmbeddingGemma 2 text and vision;
- Nemotron 3.5 Embed 1B as the NVIDIA arm that could ship;
- `omni-embed-nemotron-3b` as a measurement-only arm (research or non-commercial licence);
- Qwen3-VL-Embedding as a reference;
- the LLM embedders through vLLM or transformers, not TensorRT.

All arms run on GPU 0 unless the owner states otherwise. The scenarios are #347's S1-S6, in
`docs/research/embedding_model_eval_plan.md`.

### 4.4 RQ-4 / RQ-5: graph refinement only on a fresh set

The plan's design stands: mutual-kNN graph (k 30); graph HAC with a dev-calibrated cut; Leiden CPM
with and without the filter; GPU kNN feeding the CPU methods; cuGraph ECG as a contrast. Documented
limits to carry into the lane brief:

- `cugraph.leiden` optimises modularity only and takes no initial partition
  (https://docs.nvidia.com/cugraph/latest/api_docs/api/cugraph/cugraph.leiden/). "CPM from the
  current partition" is CPU `leidenalg` only
  (https://leidenalg.readthedocs.io/en/stable/reference.html: `CPMVertexPartition(initial_membership=...)`,
  `Optimiser.optimise_partition(..., is_membership_fixed=...)`).
- FAISS GPU indexes cap `k` and `nprobe` at 2048, and the GPU wiki documents no range search
  (https://github.com/facebookresearch/faiss/wiki/Faiss-on-the-GPU). The GPU face and
  near-duplicate arms are kNN graph plus threshold, never a GPU range search.
- Sizing at COCO train2017 (about 860,000 crops x 1024 float32): about 3.5 GB of vectors. A k = 30
  graph holds at most about 26M directed edges before mutual pruning. Record peak RSS. The CPU slot
  cap of 8 threads applies.
- E-7 (RQ-5) runs only on an E-5 or E-6 PASS. A partition that loses review work is worse than
  IVF, so RQ-5's protected-moves-0 bar is a ship precondition, not a nice-to-have.

### 4.5 RQ-9: faces

- The grouping interface already exists for this: `similarity_components(embeddings, threshold)`
  (`person_grouping.py:13`). The docstring names the research as the replacement. Stable ids, locks
  and writes consume only its output (`person_management.py:138-192`).
- So a winning method replaces that body. A new route is not needed.
- Incremental assignment and abstention are new behaviour. They land as `op.jobs.persons_assign`
  after B2.
- Privacy: face sheets go only to the local VLM or a person (`judge_protocol.py:23`).
- Publication of LFW-family aggregates is J7.

### 4.6 RQ-11 (proposed): viewpoint-stratified identity

Source: `docs/research/cross_domain_clustering_lessons.md` section 7, E4. The owner's comment on
#346 of 2026-10-29 asks for an identity-versus-view model and viewpoint strata.

- **Question:** does frozen PE-Core (and MobileCLIP as a contrast) link same-object views across
  azimuth bins 0-30, 30-90, 90-150 and 150-180 degrees?
- **Data:** public multi-view sequences, CO3D v2 (car, motorcycle, bicycle) and Objectron (bike,
  shoe). Licences are non-commercial (CC BY-NC 4.0 and C-UDA 1.0 per the cross-domain note), so the
  data is local only and only aggregate numbers are published.
- **Arms:** centroid against nearest of P prototypes (P in {1, 2, 4, 8}), from the same note's E3.
- **Bar:** register it in R0. A suggested bar is same-identity pair recall in the 150-180 degree bin
  at a fixed false-merge rate against chance.
- **Stop rule:** near chance in the 150-180 degree bin means "no-go for frozen-embedding object
  identity", and class-level grouping stays the product scope. No re-ID dataset is used without the
  owner's licence decision (the cross-domain note's E5).
- **Cost:** GPU about 1 h (encode), CPU about 2 h. Runs in CRW-2 next to E-4, with the same encoder
  set and no extra window.

### 4.7 Visual verification: provenance guard (forward-looking)

- Today the hosted judge refuses by sheet **kind**. The kind comes from the run's arguments
  (`visual_judge.py` `--kind`, `judge_protocol.py:55-58`).
- Every research run in this program must also refuse a hosted call unless the sheet's dataset
  pool is a public pin on an allowlist. The kind should come from the pool's pin manifest, not
  only from a flag.
- This keeps the owner rule ("no private photo is ever sent to a hosted model") mechanical. It is
  E-0 work with a red test: a sheet from a pool without a public pin raises before any client call.
- See J9.

---

## 5. The ordered program

Order of operations (plan section 5.2, with this gist's changes in bold):

| Step | When | Contents | Gate to start | Output |
|---|---|---|---|---|
| R0 / E-0 | Any time; docs and research scripts only; CPU slot | Register bars (Z5 re-sized per 4.1; RQ-11 if J4). Open Images train split in the fetcher; near-duplicate transform builder. V-measure and purity helpers with hand-computed red/green tests. COCO-to-Open-Images name map. **Hosted provenance guard (4.7).** **Rename windows to CRW (J3)** | Owner go on J2, J3, J9 | One docs commit with the registrations, committed **before** any score; N-2 closes |
| CRW-1 (ship track) | After the v0.7.0 tag | E-1 ingest of the Z5 set; E-2 encodes on three paths; E-3 crop re-encodes; E-12 frame encodes. One stack `-p op-rs-crw1` | Tag exists; peers agree on the GPU; host `uptime` below 16 | Vectors stored once in `artifacts_local/` |
| CPU slot A | After CRW-1 | E-1 simulation (20 seeds); E-2 sweeps; E-3 scoring; E-12 scoring; E-10 Haiku half (Message Batches) | Vectors exist | Notes `cluster_config_z5.md`, `near_duplicate_threshold.md`, `crop_preprocessing_eval.md`, a retrieval baseline append; #398, #416, #435 verdicts |
| Person pass 1 | Inside slot A | 200 natural near-pairs (E-2); 5 % re-judge of E-10 sheets (E-10); FG-0 make/model label sample | Sheets built | Kappa tables; FG-0 result |
| CRW-2 | After slot A | E-4 encoder arms (one-day time box per arm) at E-3's winner; E-5 kNN graphs; E-6 GPU arms; **E-13 viewpoint if J4** | E-3 verdict exists | Vectors and graphs |
| CPU slot B | After CRW-2 | E-5 HAC and Leiden at 860k (12-16 CPU h); E-6 agreement; E-4 scoring | Graphs exist | `encoder_eval_curation.md`, `global_refinement_eval.md`; #347 verdict |
| CRW-3 | After slot B | E-10 local VLM half; E-11 faces; E-9 only on J10 = yes | | |
| CPU slot C | After CRW-3 | E-11 scoring; E-7 only if E-5 or E-6 passed; E-8 only if E-7's machinery exists | | `face_grouping_eval.md`, `partition_stability_eval.md` |
| Landing | Per PASS | Section 7 | PASS + visual check + human G3 review + owner's yes | Feature PRs |

Rules carried from the plan:
- One heavy run at a time.
- A window never overlaps a program or v0.8.0 window, or a live-harness run.
- Pause at once when a peer session asks.
- Never act on the default compose project; always pass `-p`.
- Waits use healthchecks or the Monitor tool. Never `pgrep -f` or sleep loops.

---

## 6. GPU, CPU and person calendar

### 6.1 Every window that wants the A6000s after the v0.7.0 tag

| Window | Owner plan | Card | GPU hours | Depends on |
|---|---|---|---:|---|
| V0 spikes | v0.8.0 | GPU 0 | at most 3 | v0.7.0 tag |
| V1 breadth | v0.8.0 W8.3 | GPU 0 | about 36 | W8.3 sweep |
| V-EM | v0.8.0 W8.4 (conditional) | GPU 0 | about 24 | **E-4 PASS** (this program) |
| V2 | v0.8.0 R8 | agreed | inside R8 | release |
| GP-RW-1 / GP-RW-2 / GP-RW-3 | GPU-pipeline research | GPU 0; GPU 2 optional; both for GP-RW-3 | about 9.5 / 11 / 10 | B5 records |
| CRW-1 / CRW-2 / CRW-3 | this program | GPU 0 or GPU 2 as agreed | about 4-5 / 12-14 / 6 | tag; previous CRW |

### 6.2 Recommended sequence (J3)

1. CRW-1 right after the tag, before V1. It is short, closes three issues, and its encodes run on
   the stack that just shipped.
2. GP-RW-1, which characterises only and changes no vectors, may run between CRW-1 and CRW-2.
3. CRW-2 before any GP-RW-2 adoption merges. A GP adoption that changes preprocessing (for example
   a native executor or letterbox in the engine) must pass the B5 embedding parity gate
   (cosine >= 0.999 against stored vectors). Otherwise every CRW vector becomes stale.
4. V-EM only after the E-4 verdict, which is what it depends on.
5. CRW-3 any time after CRW-2. It uses the local VLM and the face models, so on GPU 0 it must not
   overlap a V1 measurement.

### 6.3 GPU rules

- GPU 1 is never used unless its owning session frees it (host rules, `11_risks_cleanup_host.md`
  section 1).
- GPU 2 is used only after `nvidia-smi` and a peer reply.
- GPU-0 jobs are serialised under one shared lock file all jobs agree on
  (`docs/research/CLAUDE.md`, "GPU etiquette").

### 6.4 CPU slots

- Any period with no sweep, live run or GPU window.
- `nice`, at most 8 threads.
- `uptime` is recorded at start and end. A run that starts at a 1-minute load above 16 is marked
  noisy and repeated (plan 4.3 "Cost").
- Every 0.7.0 run had load 9-79, so the first quiet-host timings are themselves a deliverable
  (tracker "Not yet established").

### 6.5 Person time

About 8 hours over the program (plan section 5.1):
- 200 near-pairs: about 1 h;
- the 5 % re-judge and FG-0: about 3 h;
- the face chaining check: about 1.5 h;
- plate transcription, only if J10: about 1.5 h.

---

## 7. How winners land in the product (after B2)

Nothing lands without:
- a registered PASS;
- the visual check;
- a human G3 review in Cropwright;
- the owner's yes.

Each landing is one feature PR, with:
- red/green tests;
- `make contracts` in the same commit as any wire change;
- the docs listed;
- a CHANGELOG entry under `[Unreleased]`.

| Winner | Celery task (B2 names) | Queue | Lease | Product change | Dead code removed in the same PR |
|---|---|---|---|---|---|
| RQ-1 class roots + guard | existing auto-label / assist-loop task; `op.periodic.cluster_refresh` (`03_b2_overview.md:204`) | `autolabel` | `autolabel` per project | A second implementation of the loop's `PartitionReader` protocol (`src/services/curation/assist_loop/partition.py:108-115`; today `ClusterFieldPartition` at `:139` is the only one, built at `assist_loop/runtime.py:181` and `prepare.py:68`): roots are detector classes keyed by **class name**, plus the hold-back guard sending members beyond the 95th-percentile distance to review. The protocol was written for exactly this (module docstring `:1-10`), so driver, gate and writer do not change | `ClusterFieldPartition`, if the owner makes class roots the only partition (otherwise both stay, selected by a project setting) |
| RQ-2 thresholds | `op.jobs.item_scores` (`03_b2_overview.md:160`) | `jobs` | `item_scores` | Defaults per path (`DEFAULT_FRAME_DEDUP_THRESHOLD` `frame_dedup.py:34`, `OP_CROP_DUP_THRESHOLD` default `env.template:590`, core route defaults); optional complete linkage inside groups; re-score through the guarded writers | Superseded default constants; doc lines that call the threshold "a hypothesis" |
| RQ-3 preprocessing or encoder | Preprocessing: none (an engine and prep change under the B5 parity gate). Encoder: `op.jobs.embedding_migrate` (v0.8.0 F9) | `jobs` | `embedding_migrate` | v0.8.0 section 2.9 | The old crop path after the flip |
| RQ-4 global refinement | `op.jobs.cluster_refine` (new) | `autolabel` (GPU-capable worker) | `cluster_refine` per (project, space, kind) | Writes a **candidate** partition only. Activation is an OCC write of the active-partition pointer (RQ-5) | n/a |
| RQ-5 partitions | none: a request doing an OCC write on the project settings document | | | Partition record, stable ids by Jaccard >= 0.5, anchors (plan 6.2). Design doc N-5 first. The OpenSearch guard must cover the new record; project delete must remove it | |
| RQ-9 faces | `op.jobs.persons_cluster` (B2 2C-6, already planned); `op.jobs.persons_assign` (new) | `jobs` | `persons_cluster` (global) | Replace the body of `similarity_components` only | The old union-find body |

Standing rules for every landing:
- No in-process or file locks for shared state. Use leases, OCC and idempotency; about 30 uvicorn
  workers plus worker containers share the state.
- No asyncio fan-out in request handlers.
- Automated writers never overwrite human- or import-validated classes or boxes, or the frozen
  holdout (`src/clients/occ_locks.py`).
- Machine merges keep their own provenance and are never written as validated.
- Proposed routes are written without a leading slash until they exist (docs check):
  - `curation/projects/{project}/clusters/partitions`;
  - `.../partitions/{partition_id}/activate`;
  - `.../clusters/suggestions`.
- Library citations go into the PR: FAISS, cuVS, cuGraph, leidenalg, scikit-learn, statsmodels and
  SciPy, as listed in plan section 3.1.

---

## 8. Shared evaluation harness and datasets (summary)

Plan section 4 stands. These additions are binding for every experiment in this program:

- **Data tiers:**
  - published (COCO, Open Images, a pin per set);
  - local only (LFW, CALFW, CPLFW, CARS196, CUB, SOP, CO3D, Objectron: aggregates only, subject
    to J7);
  - owner-run (the owner's private photo set, repeated by the owner on the winners only, numbers
    only, never named, never sent to a hosted model).
- **Splits:**
  - by image, seeded, 50/50 dev and test;
  - near-duplicate groups at frame cosine >= 0.98 split as a unit;
  - open-set splits are class-disjoint;
  - a test half is read once per method family.
- **Metrics:** BCubed P/R/F is the headline. Also pairwise, AMI, V-measure, purity, risk-coverage,
  loop U1/U2, and stability (permutation ARI must be 1.000).
- **Strata:** low resolution, elongated (> 2:1), rare, similar classes, person, open-set, and
  viewpoint bin (RQ-11).
- **Spread:**
  - three seeds plus one permutation per headline row;
  - 20 seeds for loop simulations;
  - a gain counts only above twice the run-to-run spread with a bootstrap interval excluding 0.
- **Recording:** each run records git sha, pin hash, model and engine version, embedding field,
  seed, GPU id, an `nvidia-smi` snapshot, host load, wall time, peak RSS and VRAM.
- **Raw output** goes to `artifacts_local/` (gitignored). Validated artefacts move out of `/tmp`.
- **Backing JSON** goes to a **new** folder under `docs/benchmarks/<experiment>/` with
  `SHA256SUMS`. Existing benchmark folders are never edited or pruned.
- **Results row:** every experiment adds one to the #346 body table, with the note's path.

---

## 9. Corrections to the issues' and plans' premises

| # | Premise | What the tree says | Action |
|---|---|---|---|
| C1 | #346 body: "Product clustering stays IVF (k-means, K = min(512, n // 6))" | **Holds**: `ivf.py:121` (`min(self.n_clusters, max(1, n // IVF_MIN_VECTORS_PER_CLUSTER))`, cluster count 512, 6 per cluster) | none |
| C2 | Triage comments on #347, #365, #398, #406, #416: "Moved v0.8.0 -> v0.10.0" | The later program re-cut put them on `v0.8.0` (`09_github_actions.md:88-89`). Only milestones v0.7.0 and v0.8.0 exist. The comments are stale, not the state | A one-line comment on each when this program starts, or leave as is |
| C3 | #398, #400, #401, #406 bodies cite `docs/design/improvement_backlog.md` (branch `docs/v070-user-docs`) | The file was deleted in `1b0d4e29` (row 14 of `12_prior_plans_reconciliation.md`, "DELETE-AFTER-MOVE"). It does not exist on `main` or on this branch | Point the bodies at `docs/design/research_clustering_plan.md` (once on `main`) and say the backlog is in git history |
| C4 | Plan section 6.5 N-1 and OQ-5: "person grouping as a Celery job" is a new issue | Already in B2 as family 18 / task 2C-6 with a red test named (`03_b2_overview.md:176`, `03b_b2_w2_job_families.md:111`, `:121`) | Do not file N-1; mark OQ-5 answered |
| C5 | #398 and plan RQ-1: Z5 needs "at least 1,200 open-set crops". Plan 4.1 also says "1,200 **held-out**", and the two are inconsistent | 1,200 in total (600 held out) makes `key` readable but not `key_guard` (about 1,290 held out needed; section 4.1). The original registration was 2,000 in total (`cluster_config_decision.md:90`) | Re-register Z5 by per-arm readability, at least 1,500 held out (J2) |
| C6 | Plan 4.4 step 5: the hosted refusal is "already tested in `tests/test_contact_sheet.py`" | The refusal is `scripts/research/judge_protocol.py:55-58`; the test is `tests/research/test_judge_protocol.py`. `tests/test_contact_sheet.py` has no refusal test. The refusal keys on a sheet kind taken from run arguments | Fix the reference; add the provenance guard (4.7, J9) |
| C7 | The clustering and GPU-pipeline research plans each name their windows RW-1..RW-3 and their experiments E-1.. | Both files are on this branch (`research_clustering_plan.md:286-320`, `research_gpu_pipeline_plan.md:198-212`). The names collide in any peer announcement or ledger | Rename clustering windows CRW-n and prefix GPU-pipeline items GP- (J3), in a docs commit |
| C8 | Owner comment on #346 (2026-10-29): identity versus view, viewpoint strata | The cross-domain note's E4 covers it; `research_clustering_plan.md` has no viewpoint RQ or stratum | RQ-11 / E-13 (4.6) or an explicit park (J4) |
| C9 | `v080_platform_breadth_plan.md` section 6: #348 revisits after "the clustering research plan reports its place-recall results" | The clustering plan has no place-recall experiment. Same-place grouping is a different task (geometry, not class or identity) | Resolved in the #348 plan: its own first experiment, not a clustering RQ. Fix the trigger text |
| C10 | Plan RQ-3a arms: "`src/utils/pe_preprocess.py` variants behind a research flag" | `pe_preprocess.py` is the **whole-frame** path (`:1-19`). Crops go through `pe_encoder._preprocess_crops` (`pe_encoder.py:125-131`) on CPU and the GPU prep model (`models/gpu_image_prep/1/model.py:12-14`). A product research flag would be dead code | Cut the variants in the research script; call `encode_images` (`pe_encoder.py:211`); section 4.3 |
| C11 | #346 body: "Owner's pending research notes ... add links here as they land"; comments say branches "not pushed" | `cluster_types.md`, `cross_domain_clustering_lessons.md`, `labelled_benchmarks.md`, `embedding_model_eval_plan.md`, `clustering_research_consolidated.md` are all on `main`. `research_clustering_plan.md` is on `int/b1-correctness` only | Tick the box with links; link the plan when it merges |
| C12 | #436 is in the research family (plan header) | It is product work with a concrete design in v0.8.0 F2 (task SR-1, spike SP-1). The plan's RQ-10 decides #435 only | Drop #436 from the research family |
| C13 | Plan 6.6: the near-duplicate fail-open on a bad env value | **Holds**: `near_dup.py:53-57` catches `ValueError` and returns the default. It is not tracked in any program file (`rg` over `docs/design/`) | File N-8 (small fix, not research); the fix is a fail-closed startup error with a red test |
| C14 | Plan RQ-6 arm: "the existing EXIF reader if any" | None exists. No EXIF time, GPS or camera fields are captured; the images index has no capture-time field (`bodies_core.py:73-96`). EXIF is used only for orientation | E-8's feasibility step reads EXIF from source files with Pillow `Image.getexif()` in a research script; no product field before a PASS |
| C15 | #346 body "Related": #341 harness closed | Holds; the harness is in `scripts/bench/` (section 2.3) | none |

---

## 10. Judgment calls (owner)

**J1. Adopt `research_clustering_plan.md` as the design source, with this gist as the umbrella.**
- Recommendation: yes. Merge the plan to `main` with the C5-C10 fixes, then link it in the #346
  body.
- Trade-off: two documents to keep aligned. The gist is frozen (plan only); the repo doc is the
  one that changes.

**J2. Z5 quota.**
- Recommendation: a new registration at at least 1,500 held-out open-set crops (about 3,000 in
  total), with readability per arm, and the train CSV counted before any image download.
- Trade-off: a larger fetch, about 8,000-20,000 images (the plan's estimate), and a longer ingest
  in CRW-1. The alternative risks a second INCONCLUSIVE that closes #398 on data.

**J3. Window names and calendar.**
- Recommendation: rename to CRW-1..3 and GP- (C7). Sequence per section 6.2: CRW-1 before V1;
  CRW-2 before any GP-RW-2 adoption; V-EM after E-4.
- Trade-off: v0.8.0 V1 waits about 5 GPU hours for CRW-1.

**J4. Run RQ-11 (viewpoint, E-13) in CRW-2.**
- Recommendation: yes. It is small (about 1 GPU h, 2 CPU h), answers the owner's 2026-10-29
  requirement, and has a clean stop rule.
- Trade-off: non-commercial data (CO3D, Objectron), local only, aggregates only. If no, record
  "viewpoint not evaluated" in the outcome note.

**J5. #435: measure (E-12, about 1 GPU h in CRW-1) or close now per v0.8.0 Q-6.**
- Recommendation: measure if CRW-1 runs before W8.1 decides. Otherwise close per Q-6 and reopen
  only on an E-12 within-0.02 result.
- Trade-off: half a GPU hour against a decision made without data.

**J6. leidenalg (GPL-3.0) as a product dependency.**
- Recommendation: decide only if E-5 picks Leiden. The product fallback is scikit-learn average
  linkage on a sparse connectivity graph.
- Trade-off: GPLv3 section 13 allows combination with AGPLv3, but it needs an ATTRIBUTION row and
  the owner's yes (plan OQ-4).

**J7. Publish aggregate face numbers from LFW, CALFW and CPLFW.**
- Recommendation: aggregates with a licence caveat; nothing per image.
- Trade-off: published face claims could otherwise come only from person-checked Open Images
  faces, which are slow and small.

**J8. Budgets.**
- Recommendation: Haiku at most about 300 sheets for E-10 and 50 per other experiment, through
  Message Batches; about 8 hours of owner or delegate time spread over the windows.
- Trade-off: below that, the half-width 0.10 rule turns runs into "insufficient sample".

**J9. Hosted-judge provenance allowlist (4.7).**
- Recommendation: yes, in E-0, before any hosted call in this program.
- Trade-off: a small research-script change and one test.

**J10. Parked privacy lines (#401 plate identity; #400 colour and EXIF signals).**
- Recommendation: keep both parked. Run E-9 (read quality only) only after the owner approves the
  privacy design (hash-only default, purge with secret rotation, no plate text in logs, exports or
  committed files).
- Trade-off: public data cannot test plate *linking* at all; only the owner-run set can.

**J11. Keep #346 open as a standing tracker after this program.**
- Recommendation: yes, unmilestoned, one row per experiment. New datasets or notes reopen work
  here, as the body says.
- Trade-off: an open issue with no milestone, which the label scheme already allows (`parked`,
  `research`).

---

## 11. File-by-file change list (when the program is executed)

### 11.1 R0 / E-0: docs and research scripts, CPU, before any score

| File | Change |
|---|---|
| `docs/design/research_clustering_plan.md` | Apply C5 (Z5 quota and readability), C6 (test path), C7 (CRW names), C8 (RQ-11 or park note), C10 (variant mechanism), C12 (drop #436), C4 (N-1 done). One docs commit |
| `docs/research/cluster_config_z5_registration.md` (new) | Z5 registration: data, train-split counts (pasted from the fetcher's count step), arms, bars, per-arm readability, seeds; committed before the fetch |
| `docs/research/near_duplicate_registration.md` (new) | RQ-2 bars per path, transforms, the person-checked sample rule |
| `docs/research/viewpoint_registration.md` (new, J4) | RQ-11 bars and bins |
| `scripts/datasets/fetch_openimages_plates.py` | Add the `train` entries to `BBOX_URL` and `IMAGES_URL` (`:74-81`) |
| `scripts/datasets/fetch_vehicle_eval.py` | A `--count-only` mode that streams the CSV and prints per-class open-set counts after the Z4 filters, without downloading images |
| `scripts/datasets/near_dup_variants.py` (new) | Builds 8 transforms per original with Pillow (JPEG q60/q30, half and quarter size, 90 %/80 % centre crop, brightness x1.15, small rotation); writes a pin |
| `scripts/datasets/manifests/coco_oi_class_names.json` (new) | COCO-to-Open-Images name map, as data |
| `scripts/bench/cluster_eval_metrics.py` | `v_measure` (scikit-learn `homogeneity_completeness_v_measure`) and `purity` helpers |
| `scripts/research/judge_protocol.py` | Provenance allowlist check before any hosted call (J9) |
| `tests/bench/test_cluster_eval_metrics.py` | Hand-computed V-measure and purity cases, red on a wrong formula (swap homogeneity and completeness: the asymmetric case must fail) |
| `tests/test_vehicle_eval_selection.py` | The `--split train` URL case (red: argparse invalid choice); count-only on a tiny CSV fixture |
| `tests/research/test_judge_protocol.py` | A pool without a public pin raises before the client is called (red today: it is sent) |

### 11.2 Experiments: research scripts only, no `src/` change

| File | Experiment |
|---|---|
| `scripts/research/cluster_diag/z5_run.py` (new, a thin wrapper over `z4_run.py` with the new registration) | E-1 |
| `scripts/research/near_dup_sweep.py` (new) | E-2: exact global versus bucketed, single versus complete linkage, per path |
| `scripts/research/crop_variants_encode.py` (new) | E-3: cut variants, call `PEEncoder.encode_images` |
| `scripts/research/standard_benchmarks/` (extend) | E-4 arms per #347 |
| `scripts/research/leiden_refine_proto.py`, `gpu_methods.py` (extend) | E-5, E-6 at 860k |
| `scripts/research/face_grouping_eval.py` (new) | E-11: arms call through the `similarity_components` signature so a winner is a body swap |
| `scripts/research/viewpoint_eval.py` (new, J4) | E-13 |
| `docs/research/<note>.md` (new per experiment) and `docs/benchmarks/<experiment>/` with `SHA256SUMS` | Every experiment |
| `docs/research/README.md`, `docs/design/README.md` | One row per new note (`docs/research/CLAUDE.md`, "Adding a note") |

### 11.3 Landings (only on PASS; after B2)

See section 7. Typical files:
- `src/services/curation/item_scores/near_dup.py` and `src/services/detection/frame_dedup.py`
  (RQ-2);
- `src/services/curation/assist_loop/partition.py` (RQ-1: a new `PartitionReader`; RQ-5: the versioned partition implements the same protocol), `assist_loop/runtime.py:181`, `assist_loop/prepare.py:68`;
- `src/services/face_identity/person_grouping.py` (RQ-9);
- `src/clients/pe_encoder.py` and `models/gpu_image_prep/1/prep_geometry.py` (RQ-3a);
- `contracts/` and `docs/design/curation_api_contract.md` for any wire change;
- `CHANGELOG.md`;
- `ATTRIBUTION.md` for any new product dependency.

### 11.4 GitHub (on the owner's go; through the active program's GitHub-actions file)

- File N-2, N-3, N-4, N-6, N-8 (and N-7 if J4).
- Comment on #398 (J2 quota), #436 (leaves the research family), and #398/#400/#401/#406 (C3 dead
  link).
- Update the #346 body: tick the notes box (C11), link the plan, add the CRW calendar.

---

## 12. How we prove it: the verification plan

### 12.1 Static checks (whole-tree, before a window opens)

1. **Bars precede scores.** For every experiment, `git log --format='%H %ad' -- <registration file>`
   shows a commit older than the first commit of its results note and its `docs/benchmarks/`
   folder.
   - Red looks like this: a results commit with no earlier registration, or a registration edited
     after the results commit without a "new registration because..." paragraph.
2. **Benchmark data is only added.** `git diff --stat <base>..HEAD -- docs/benchmarks/` shows
   additions only. Any `D` or `M` line on an existing file fails.
3. **Real data only.** Every scorecard JSON passes `scripts/bench/scorecard.py`. Any quality field
   on a dataset with `synthetic: true` raises (`scorecard.py:87-94`).
4. **No private names.** `scripts/codegen/check_naming_leaks.py` is green on every committed note.
   The owner's private set appears only as aggregate numbers, described as the owner's private set.
5. **Docs check.** `.venv/bin/python scripts/docs/check_docs_vs_code.py` is green. Proposed routes
   are written without a leading slash.
6. **Metric helpers.** The new V-measure and purity tests are seen failing on a deliberately wrong
   formula before the right one, and the red output is pasted in the PR.

### 12.2 Dynamic checks (per window)

1. **Pin integrity.** `manifest_sha256` of every pin matches before encode. `disjointness.json`
   shows 0 shared names and 0 shared SHA-256 against every earlier pin.
2. **Vector provenance.** Each vector bundle records the git sha, engine version and B5 parity
   status.
   - Before CRW-2 and CRW-3, re-encode a seeded 1,000-crop sample and compare with the stored
     vectors: cosine >= 0.999 for every row.
   - Red looks like this: any row below the bar, meaning the stack changed under the research
     (plan risk "B5 crop path").
3. **Host record.** `uptime` at start and end and an `nvidia-smi` snapshot per run. A start above
   16 marks the row noisy, and the row is repeated.
4. **Stack isolation.** `docker compose -p op-rs-crw<n>` only. Snapshot `docker ps -a`, volumes,
   networks and images before and after, and diff them (host rule 4).

### 12.3 Empirical checks (per verdict)

1. **Machine bars** on the test half of a fresh set, three seeds, one permutation, a bootstrap
   interval. INCONCLUSIVE is reported as INCONCLUSIVE. It is never rounded to PASS.
2. **Visual check.** Contact sheets per plan 4.4:
   - per-kind agreement first;
   - planted controls about 10 %; a batch with failing controls is discarded;
   - position-bias re-render about 10 %;
   - Wilson intervals with a half-width of 0.10, or "insufficient sample".
3. **Human re-judge.** A blind, seeded 5 % (at least 20 per run). Judge-versus-human kappa below
   0.7 suspends the judge for that kind.
4. **Landing proof** (product PRs only):
   - red/green unit tests on the changed behaviour;
   - a live check on a throwaway COCO project through the API (`-p` name);
   - the Cropwright review page opened in a browser with full-page screenshots at desktop and
     430 px width, actually viewed.

---

## 13. Edge cases and traps (the failure register)

These are the ways the program can look done and not be.

1. **The guarded arm is silently unread.**
   - A Z5 that meets "1,200" passes the fetch step and `key` becomes readable.
   - `key_guard` stays at fewer than 30 per run, and the verdict table shows INCONCLUSIVE for the
     arm that matters.
   - Guard: per-arm readability in the registration (4.1).
2. **The train CSV is too large to load.** Loading the multi-GB box CSV into memory will exhaust
   the CPU slot. Stream it and filter per row.
3. **Licence filter drift.** The train split's rotation CSV must carry the CC BY 2.0 licence column
   that Z4 filtered on. If a column differs, the filter silently passes everything. The count step
   must print rows dropped per filter.
4. **Class index leakage.** Open-set membership computed from detector **indices** breaks when the
   registry order changes. Compute it from names against the name map only.
5. **Bucketed near-duplicates.** Crop near-duplicate recall measured on a global pass overstates
   the product, which runs per IVF bucket (`near_dup.py:19-26`). Report both.
6. **Mixed spaces.** A threshold or graph that mixes MobileCLIP and PE vectors, or PE vectors from
   two preprocessing variants, is meaningless. Each bundle carries one `embedding_field` plus a
   variant id. The scorer refuses a bundle with more than one.
7. **Worn held-out half.** The 3,000-item held-out half is regression-only (read by three waves).
   An acceptance claim on it is invalid. The fresh set's test half is read once per method family.
8. **Near-duplicates straddling dev and test.** Calibration leaks unless frame-cosine >= 0.98
   groups are split as a unit.
9. **Row order.** Before trusting any method, check permutation ARI = 1.000 on identical vectors.
   The 0.7.0 research had a row-order fix (`cluster_config_baseline.md`).
10. **Abstention scoring.** BCubed counts an abstained item as a singleton (harness rule). A method
    that abstains a lot can look precise. Always report the unassigned fraction and risk-coverage.
11. **Oracle labeller.** Loop simulations with an oracle hide VLM errors. Report the 10 % noisy
    labeller row beside every oracle row.
12. **Vacuous judge calibration.** A judge that says "different" to every sheet gives kappa 0 and
    an empty calibration. Use balanced sheets, half clean and half contaminated, and check the
    verdict mix before scoring.
13. **Hosted call by mislabel.** A face-bearing or private pool run with a non-face `--kind` would
    pass the kind check. The provenance allowlist (4.7) closes this for this program.
14. **cuGraph Leiden as a stand-in for CPM.** It optimises modularity and takes no initial
    partition. A "GPU Leiden" row is a contrast arm, never the CPM result.
15. **FAISS GPU caps.** `k` above 2048 fails or truncates. A k = 30 graph is fine; a "range" emulated
    with a large `k` is not.
16. **Partition ids renumbered.** Without the Jaccard >= 0.5 id carry-over, every re-run destroys
    review work while all quality metrics stay green. RQ-5's bars (protected moves 0, id stability
    >= 0.95) are the only check that sees it.
17. **Celery time limits.** An 860k HAC or Leiden run inside a `jobs` queue task can exceed the
    queue's `time_limit`. Research runs never run as product tasks. A landing of RQ-4 sizes its
    task limit from E-5's measured wall time.
18. **Synthetic numbers in a quality table.** The scorecard refuses them, but a hand-written
    markdown table would not. Every quality number in a note must link the scorecard JSON it came
    from.
19. **Benchmark pruning by cleanup.** A cleanup PR deleting "old" `docs/benchmarks/` JSON passes
    every test. Static check 12.1.2 is the guard. Some of that data backs a write-up.
20. **Loaded-host timings presented as absolute.** Only within-table comparisons hold unless the
    load record shows a quiet host.
21. **Plan-ID collision.** "RW-2 done" in a ledger is ambiguous between the two research plans
    (C7).
22. **Faces never hosted.** Face, plate and identity-text sheets go to the local VLM or a person
    only (`judge_protocol.py:23`). Also, no face crop may enter a generic "object" pool from COCO
    person boxes without the kind set to `face` where the crop is a face.

---

## 14. Suggested phasing

| Phase | Contents | Exit |
|---|---|---|
| P0 (now, docs only) | Owner answers J1-J11. Merge the plan to `main` with C4-C12 applied. Update the #346 body (C11). File N-2, N-3, N-4, N-6, N-8 (N-7 if J4) | Plan on `main`; issues filed |
| P1 (R0, CPU, any time) | E-0 per 11.1. Z5 count step run and pasted into the registration | Registrations committed; fetcher and metric tests red then green |
| P2 (after the tag) | CRW-1 plus CPU slot A plus person pass 1 | Verdicts on #398, #416, #435 and the Haiku half of #406 |
| P3 | CRW-2 plus CPU slot B | Verdicts on #347 (E-4) and RQ-4 (E-5, E-6); RQ-11 if J4 |
| P4 | CRW-3 plus CPU slot C | Faces verdict (E-11); #406 closes; E-7 only on an RQ-4 PASS |
| P5 (per PASS, after B2) | Landings per section 7 | Feature PRs merged with the owner's yes |
| Close-out | Update `docs/research/cluster_program_outcome.md` (append a section; never rewrite the 0.7.0 sections). One #346 row per experiment, including the parked and killed lines | The tracker reads complete without opening any gist |

---

*Plan only, not implemented. Sources: the repository files cited inline at `81c4e4ef`; library
documentation as cited (scikit-learn, leidenalg, cuGraph, FAISS wiki, Open Images V7 download
page), read 2026-11-04.*
