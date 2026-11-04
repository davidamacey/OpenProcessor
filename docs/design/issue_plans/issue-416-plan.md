# OpenProcessor — evaluate the near-duplicate grouping threshold (issue #416, default 0.99)

Status: **PLAN ONLY, not implemented.** Written 2026-11-04 against branch `int/b1-correctness` at
`81c4e4ef`. Every `path:line` below is at that commit. Nothing in the repository was changed by this
planning pass.

Reader: a research or implementation agent with no memory of the planning session. This gist is
the execution-ready, per-issue version of RQ-2 / E-2 in `docs/design/research_clustering_plan.md`
(sections 1.1, 2, 3.2, 4.1, 4.6, 5, 6.1, 7.1, OQ-9). Where this gist and that plan differ, the
difference is argued in section 9 ("Corrections").

---

## 0. TL;DR — what I recommend, and what I am NOT deciding

**The issue asks one question; the code has three different answers to it, on two different
scales, in two different embedding spaces.**

| Path | Grouping | Space | Threshold today | Scale of that number |
|---|---|---|---|---|
| **core** (`POST /ingest`, `/ingest/batch`, `/ingest/directory`) | online, greedy: join the first neighbour's group, else create a group with the best match | MobileCLIP 512-d, `visual_search_global.global_embedding`, OpenSearch Faiss HNSW `cosinesimil` | **0.99** | **OpenSearch score** = `(1 + cos) / 2`, so cosine **0.98** |
| **frame** (curation export `dedup_threshold`) | exact blocked cosine + union-find over the export's frames | PE-Core-L14-336 whole-frame `pe_embedding`, images index | off by default; UI default **0.98** | raw cosine |
| **crop** (curation `near_dup` scorer, `OP_CROP_DUP_THRESHOLD`) | exact cosine + union-find **inside IVF buckets** of the residual pool | PE-Core-L14-336 crop `pe_embedding`, items index | **0.98** | raw cosine |

So "0.99 vs 0.98" is not a disagreement about strictness: the core 0.99 and the curation 0.98 are
the **same cosine cut** (0.98) applied in different spaces, and the code comment that 0.99 "matches
Immich's maxDistance=0.01" is wrong twice (Immich's 0.01 is a cosine **distance**, i.e. cosine 0.99,
in a different CLIP model). Section 9, C1 and C2.

**Recommendation:**

1. **E-2 (research).** Build a pre-registered near-duplicate set from 5,000 public originals (COCO
   val2017 plus Open Images V7 validation, disjoint from earlier pins) with factor-isolated
   transforms; encode it once through each product path on a research stack; sweep the threshold
   **offline on the vectors** with the product's own grouping functions; read recall per transform
   and the false-group rate on true non-duplicates per path; visually check the boundary pairs with a
   blind human re-judge. Output: `docs/research/near_duplicate_threshold.md` and
   `docs/benchmarks/near_duplicate_threshold/` with `SHA256SUMS`.
2. **Landing (only on a per-path PASS and the owner's yes).** One named default per path, stated in
   cosine with the core score derived from it, the seven duplicated `0.99` literals collapsed into
   one constant, the crop scorer version bumped and its stale fields cleared on re-score, docs and
   CHANGELOG. Existing core groups are not silently re-grouped (J6).

**Stop rule (per path).** No threshold meets both bars on that path: keep today's default there,
document the trade-off in the note, and offer a project setting only if the owner asks for one
(research plan 7.1 RQ-2).

**What I am NOT deciding** (section 10): the scale exposed on the core wire (J1), whether
factor-isolated variants are saved as PNG or JPEG q95 (J2), the linkage arm adopted if complete
linkage wins (J3), one default per path versus a project setting (J4), the judge for the visual
check (J5), what happens to existing core groups after a default change (J6), whether the race and
stale-field defects found here are fixed inside #416's landing or as separate fixes (J7), and the
recall/precision trade-off if no threshold meets both bars but one comes close (J8).

---

## 1. Context — why the obvious approach is wrong here

- **"Sweep 0.90 to 0.99 through `POST /ingest` and pick the knee" conflates three things**: the
  threshold, the scale of the number, and the grouping algorithm. The retrieval baseline did exactly
  that (`scripts/bench/retrieval_eval_runs.py:574-640`, `eval_dedupe`) and its own note explains why
  "grouped with original" is lower than "grouped": variants join groups of their own earlier
  variants, i.e. the result depends on arrival order (`docs/research/retrieval_baseline.md`, section
  "Near-duplicate grouping at ingest"). A threshold study must separate the vector question (which
  pairs are above t) from the algorithm question (what groups the product builds from those pairs).
- **The scale differs by path.** Core compares the OpenSearch kNN `_score` against the threshold
  (`src/services/duplicate_detection/queries.py:44-56`). For the `cosinesimil` space the OpenSearch
  score is `(2 - d) / 2` with `d = 1 - cos` (OpenSearch docs, "Spaces",
  <https://docs.opensearch.org/latest/mappings/supported-field-types/knn-spaces/>, consulted
  2026-11-04), i.e. `(1 + cos) / 2`. The global index uses that space on the Faiss engine
  (`src/clients/opensearch/indexes.py:119-131`). Score 0.99 is cosine 0.98; score 0.95 is cosine
  0.90; the route's lower bound `ge=0.90` (`src/routers/ingest/batch.py:75-80`,
  `single.py:78-84`, `directory.py:76-80`) is cosine 0.80. Curation paths use raw numpy cosine
  (`src/services/detection/frame_dedup.py:60-90`).
- **A threshold is only meaningful in its space.** The research plan's lesson 1.3 item 4: the PE
  space is anisotropic (mean random-pair cosine 0.50 on COCO, effective rank 104 to 122 of 1,024),
  so a cosine cut does not transfer between encoders, between crops and frames, or between datasets.
  Each path is calibrated in its own space.
- **The baseline's transforms were compounded.** `make_variant` saves every variant as JPEG
  quality 90, so "half size", "90 % crop" and "brighter" are each also a recompression
  (`scripts/bench/retrieval_eval_data.py:77-97`). That hides which factor moves the similarity. The
  new set isolates factors (section 3.2, J2).
- **Crops have hard negatives frames do not.** Two different people in one photo, or two cars in one
  street scene, can sit very close in a crop space; the scorer's own docstring says crops "likely
  need a tighter cut than the whole-frame 0.98" (`src/services/curation/item_scores/near_dup.py:11-17`).
  The crop study must include same-image distinct-object pairs as negatives.
- **The crop path cannot find cross-bucket pairs by design.** Groups are computed only inside IVF
  buckets (`near_dup.py:19-27`, `:87-95`). A threshold study that runs global union-find on crops
  measures a method the product does not run. Both are reported; the product number is the bucketed
  one.
- **Changing a default does not change data already grouped.** Core groups are written at ingest and
  there is no re-grouping job (`src/services/duplicate_detection/service.py` has no scan function
  despite its docstring line 7, "Scan entire index to create duplicate groups"). The crop scorer
  writes fields only for grouped items, so a re-score at a new threshold leaves stale fields behind
  (section 2.4). Landing must handle both.

---

## 2. What is actually there today (measured, not estimated)

### 2.1 Core path

| Fact | Where |
|---|---|
| Default constant `DEFAULT_SIMILARITY_THRESHOLD = 0.99`, comment "matches Immich's maxDistance=0.01" and a 0.90/0.95/0.99 legend | `src/services/duplicate_detection/models.py:7-11` |
| The same `0.99` is repeated as a literal default in 7 more places | `src/services/visual_search/ingest_single.py:77`, `:276`; `src/services/visual_search/ingest_batch.py:81`, `:93`; `src/routers/ingest/single.py:78-84`; `src/routers/ingest/batch.py:75-80`; `src/routers/ingest/directory.py:76-80` |
| kNN body: `size = k`, `knn.global_embedding.k = k`, returns `_score` | `src/services/duplicate_detection/queries.py:26-41` |
| Matches = hits with `_score >= threshold` minus self, in reply order | `queries.py:44-66` |
| Assignment: search `max_results=10`; if any match has a group, join the **first** such group; else create a group with this image as primary and the best match as member | `src/services/visual_search/ingest_single.py:272-355` (`_assign_to_duplicate_group`) |
| Any exception in assignment is logged at warning and swallowed (ingest continues, the image is ungrouped) | `ingest_single.py:357-360` |
| Batch ingest: bulk index with `refresh=False`, then `asyncio.gather` over every document's assignment, all at once | `ingest_batch.py:250` (refresh), `:223-224`, `:323-345` (gather at `:339`) |
| Index: `knn_vector`, `hnsw`, `space_type: cosinesimil`, `engine: faiss` | `src/clients/opensearch/indexes.py:119-131` |
| Read routes: `GET /query/duplicates`, `/query/duplicates/stats`, `/query/duplicates/{group_id}` | `src/routers/query_duplicates.py:30,90,125` |
| The public API doc lists `near_duplicate_threshold` as a form field of `POST /ingest` | `docs-site/docs/api-reference/core.mdx:69` |

### 2.2 Frame path (curation export)

| Fact | Where |
|---|---|
| `DEFAULT_FRAME_DEDUP_THRESHOLD = 0.98`, bounds 0.0 to 1.0; docstring: at or above 0.98 "true duplicates", 0.96 to 0.97 "same subject, different angle" kept as variation | `src/services/detection/frame_dedup.py:15-38` |
| Exact blocked `X Xᵀ`, union-find, groups of size >= 2 | `frame_dedup.py:41-90` |
| Export dedup: off unless the request sets `dedup_threshold`; prefers a held-out frame as survivor so near-twins drop from train | `src/routers/curation/_common_models.py:280-286`; `frame_dedup.py:104-143`, `:172-200`; callers `src/services/curation/export_images.py:264-266`, `src/services/curation/export_single_class.py:206-208` |
| The UI default offered for `dedup_threshold` is the 0.98 constant | `src/services/curation/strategy_builders.py:220-226` |

### 2.3 Crop path (curation `near_dup` scorer)

| Fact | Where |
|---|---|
| `OP_CROP_DUP_THRESHOLD`, default = the frame constant 0.98; docstring "Threshold is a hypothesis, not a finding" | `src/services/curation/item_scores/near_dup.py:11-17`, `:53-57` |
| An invalid value is caught (`except ValueError`) and the default used silently (fail-open; already scheduled as B2 item 2C-7) | `near_dup.py:53-57`; `docs/design/program/03b_b2_w2_job_families.md:112-117` |
| Bucket scoping by nearest IVF centroid of the bound project's store; no store => `RuntimeError` | `near_dup.py:60-95`, `:117-134`; the store path is per project (`src/services/curation/clustering/methods/ivf_store.py:13-15,58-66`) |
| Writes `dup_group_id = f'dup_{gi}'` (positional per run), `dup_group_size`, `dup_is_representative`, `dup_threshold`, `dup_method`, `dup_scored_at`, only for grouped items | `near_dup.py:146-159` |
| Pool `residual` only: validated and confidently labelled crops are never grouped | `near_dup.py:111` |
| Write-back is a plain bulk partial update of the fields in the result | `src/services/curation/item_scores/job.py:200-228` |
| Consumer: curation search `hide_near_duplicates` keeps items with `dup_is_representative` true or missing | `src/services/curation/semantic_search.py:137-145`; routes `src/routers/curation/search.py:75`, `search_image.py:66` |
| Status: the scorer is not in `VALIDATED_SCORERS` (shadow); an earlier pre-screen swept 0.95 to 0.99 for class-coverage retention only, precision never executed | `src/services/curation/strategy_flags.py:117-121`; `docs/research/curation_scores.md` section 4 |

### 2.4 Consequences of 2.1-2.3 (read from the code, not measured)

- **Stale crop fields after a re-score.** An item grouped at 0.98 and not grouped at a new threshold
  keeps its old `dup_*` fields (the writer only writes grouped items, `job.py:213-221`); if its old
  `dup_is_representative` is false, `hide_near_duplicates` keeps hiding it. `dup_0` from one run and
  `dup_0` from the next are unrelated groups with the same id.
- **Core batch grouping is racy and order-dependent.** Documents are bulk-indexed with
  `refresh=False` and every assignment runs at once (`ingest_batch.py:250`, `:339-343`); a variant
  and its original in the same batch may not see each other (not yet refreshed), and two concurrent
  assignments can each create a group. With many API workers the same holds across requests. This
  is a data-quality defect (not a security one): groups depend on timing.
- **The holdout freeze does not consider near-duplicates.** `src/routers/curation/holdout_freeze.py`
  has no duplicate logic (`rg -n -i "dup" src/routers/curation/holdout_freeze.py` is empty); leakage
  protection exists only in export, only when `dedup_threshold` is set. The research plan's phrase
  "duplicate groups are ... the train/test split guard" describes the parked design
  (`docs/design/cluster_search_makemodel_plan.md:517`), not the code.
- **Near-duplicate groups do not feed the loop gate.** No `dup_` reference exists under
  `src/services/curation/assist_loop/`. "One vote per group in gates" is also the parked design.

### 2.5 What the baseline measured (quoted, `docs/benchmarks/retrieval_baseline/dedupe_r1.json`)

Core route, 100 originals per threshold, each variant ingested once; similarities are OpenSearch
scores; host load 25 to 39 (1-minute):

| Threshold (score) | cosine | q60 grouped (with original) | half size | 90 % crop | brighter x1.15 | median score of the matched pair (q60 / half / crop / brighter) |
|---|---|---|---|---|---|---|
| 0.99 | 0.98 | 34 (24) | 13 (9) | 36 (21) | 96 (87) | 0.9932 / 0.9920 / 0.9930 / 0.9962 |
| 0.95 | 0.90 | 100 (55) | 100 (61) | 99 (75) | 100 (87) | 0.9859 / 0.9801 / 0.9890 / 0.9963 |
| 0.90 | 0.80 | 100 (52) | 100 (52) | 100 (67) | 100 (71) | 0.9873 / 0.9811 / 0.9882 / 0.9961 |

Plus: 2,000 distinct originals produced one false group at 0.99 (score 0.9924), and exact
re-upload of 20 originals returned `duplicate` for all 20 (the imohash path, not this threshold).
Every variant was saved at JPEG quality 90 (section 1). At score 0.95 the smallest matched score for
q60 was 0.970 (cosine 0.94), for half size 0.964 (cosine 0.93): the recompression and resize pairs
sit between cosine 0.93 and 0.99 in MobileCLIP space, which is the range the sweep must resolve.

### 2.6 Reusable tooling

| Tool | Where |
|---|---|
| Variant maker (to extend, not to copy) | `scripts/bench/retrieval_eval_data.py:73-97` |
| Core route harness (for the end-to-end confirmation only) | `scripts/bench/retrieval_eval_runs.py:574-640` (`eval_dedupe`) |
| Retrieval pin (2,000 COCO val2017 images, ids and sha256) | `scripts/datasets/manifests/coco_bench_2000.json`, `docs/benchmarks/retrieval_baseline/sample_ids.json` |
| Grouping functions to import (never re-implement) | `frame_dedup.near_dup_groups`, `near_dup.compute_near_dup_groups`, `duplicate_detection.queries.matches_from_hits` |
| Vector export from a project | `scripts/research/cluster_diag/fetch_vectors.py`; `scripts/bench/export_project_embeddings.py` |
| Contact sheets, judge, privacy refusal | `src/services/labeling/contact_sheet.py`, `scripts/research/visual_judge.py`, `tests/test_contact_sheet.py` |

---

## 3. Requirement R1 — the near-duplicate set (E-0 builder, E-2 data)

### 3.1 Originals

- 5,000 distinct originals: 2,500 COCO val2017 and 2,500 Open Images V7 validation, seeded,
  **disjoint** by file name and SHA-256 from every earlier pin (`--exclude-manifest` for all of
  `scripts/datasets/manifests/*.json`, as the Z4 fetcher does), licence-filtered (COCO: Flickr terms
  as recorded; Open Images: CC BY 2.0, no stored rotation).
- Each original keeps its dataset boxes and categories (COCO instances, Open Images boxes), so crop
  pairs can be matched and face/plate content can be routed (section 6.4).
- A pin manifest `scripts/datasets/manifests/near_dup_5000.json` with counts, licences and a
  `manifest_sha256`; images in a scratch directory outside the repo.

### 3.2 Transforms (factor-isolated, plus compound rows)

| Id | Transform | Saved as | Purpose |
|---|---|---|---|
| T-q60 | JPEG re-encode at quality 60 of the decoded original | JPEG q60 | recompression alone |
| T-q30 | JPEG quality 30 | JPEG q30 | heavy recompression |
| T-half | 50 % resize (Lanczos) | J2 (PNG or JPEG q95) | resize alone |
| T-quarter | 25 % resize | J2 | strong resize |
| T-crop90 | centred 90 % crop | J2 | framing |
| T-crop80 | centred 80 % crop | J2 | framing, harder |
| T-bright | brightness x1.15 | J2 | photometric |
| T-rot | rotation by 3 degrees, expand false, centre crop to remove borders | J2 | small rotation |
| C-social (compound) | 50 % resize then JPEG q75 | JPEG q75 | what a messenger or social upload does |
| C-baseline (compound) | the baseline's four variants exactly (`make_variant`) | JPEG q90 | continuity with `dedupe_r1.json` |

Pillow only (`PIL.Image.resize`, `ImageEnhance.Brightness`, `Image.rotate`), seeded, deterministic,
each output hashed in the manifest. The builder extends `retrieval_eval_data.py` (a new function
beside `make_variant`, `VARIANTS` stays for continuity) rather than adding a parallel module.

Every decoder on every product path must accept the chosen format: check with one file of each kind
through core `POST /ingest` and curation upload before the run (J2; trap T5).

### 3.3 Negatives and hard negatives

| Kind | Source | Truth |
|---|---|---|
| distinct originals (all pairs) | the 5,000 originals | different by construction, **except** true duplicates already inside COCO or Open Images: every distinct pair above the lowest swept threshold on any path is person-checked (budget: at most 200 pairs; beyond that a seeded sample of 200 and the rate reported with a Wilson interval) |
| same-image distinct objects (crop path) | two crops of one original with box IoU < 0.1 | different objects |
| same-class different images (crop path) | crops of one detector class from two originals | different objects |
| natural near-pairs | pairs of distinct originals above cosine 0.95 found by search | person-checked, 200 at most; positives if the person says same photo or burst |

### 3.4 Crop-path ground truth

A variant crop is a positive of an original crop when the original's box, mapped through the known
transform (identity, scale, offset or rotation of the corners), has IoU >= 0.7 with the variant's
box and both carry the same `detector_class_name`. A crop truncated by more than 10 % of its area by
T-crop80 or T-rot is excluded from positives (reported as a count). Matching uses the detector's
boxes, not the dataset's, because the product groups detector crops; dataset boxes only route
faces and plates.

---

## 4. Requirement R2 — measurement method (vectors first, then the route)

### 4.1 Encode once per path on a research stack

- Stack `-p op-rs-rw1` (research window RW-1, shared with E-1/E-3/E-12 per the research plan 5.1),
  GPU agreed with peer sessions (GPU 2 or GPU 0; never GPU 1), `/health` ready, `uptime` recorded.
- **Core:** ingest originals and variants into the stack's own core index through `POST /ingest/batch`
  with `detect_near_duplicates=false` (vectors only, no grouping side effects), then export
  `global_embedding` for every id. Never the live stack's core index.
- **Frame and crop:** one throwaway curation project, `scripts/curation/ingest_upload.py --project
  <slug> --force-rehash`; export the images index `pe_embedding` (frame) and the items index
  `pe_embedding` (crop) read-only; run the clustering pipeline once so the project has an IVF store
  (the scorer needs it, `near_dup.py:123-132`), and export its centroids.
- If the run is on the B5 stack, record that; if not, record the image tags. Vectors from another
  preprocessing are a different space (trap T10).
- Estimated GPU time: 45,000 images through core plus the same through curation, about 1 to 3 hours
  (unmeasured; the research plan estimated 1 h); time box 4 hours.

### 4.2 Offline sweep (CPU slot, 8 threads, `nice`)

- All candidate pairs with cosine >= 0.85 per path by exact range search: FAISS `IndexFlatIP` on
  unit vectors with `range_search` (FAISS wiki, "Special operations on indexes", range search on CPU
  indexes; <https://github.com/facebookresearch/faiss/wiki/Special-operations-on-indexes>). Exact,
  deterministic, no ANN noise.
- Threshold grid in **cosine** 0.900 to 0.995 in steps of 0.005 (20 values), every row also printed
  on the OpenSearch score scale for core (`score = (1 + cos) / 2`).
- Grouping per path, by **importing the product functions**:
  - core: a faithful offline emulation of `_assign_to_duplicate_group` in a seeded arrival order
    (exact top-10 neighbours among already-inserted images, `matches_from_hits` semantics, first
    existing group wins), run for 3 orders; plus the order-free reading (connected components) to
    show how much the online rule changes the result;
  - frame: `frame_dedup.near_dup_groups` as is;
  - crop: `near_dup.compute_near_dup_groups` with the project's centroids (the product number) and
    without (global; the bucket loss is the difference).
- Linkage arms (research plan 3.2 RQ-2): single linkage (today) and complete linkage inside each
  single-linkage component, via scikit-learn `AgglomerativeClustering(metric='cosine',
  linkage='complete', distance_threshold=1 - t, n_clusters=None)` (scikit-learn docs, "Hierarchical
  clustering", <https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering>).
  Complete linkage stops chaining (A~B, B~C, A not ~ C) at lower thresholds, the known false-merge
  cause (`docs/research/cross_domain_clustering_lessons.md`).
- Split: by original, 50 % calibration and 50 % test, seeded; thresholds and the arm are chosen on
  calibration; every bar is read once on the test half.

### 4.3 End-to-end confirmation (one run, after the choice)

At the chosen core threshold only, ingest the test half's originals and then its variants through
`POST /ingest` (single route, concurrency 1, so the race of section 2.4 does not blur the read) with
the existing `eval_dedupe` harness extended to take the new set; the grouped/not-grouped decision
per variant must agree with the offline emulation for at least 98 % of variants. A lower agreement
means the emulation is wrong, and no number from it is published until it is fixed and re-run.
A second, unbarred row at the production batch route shows the race's effect.

---

## 5. Requirement R3 — metrics and bars (registered before any score)

`docs/research/near_duplicate_threshold.md`, registration section, committed before the set is
scored.

### 5.1 Metrics per path, per threshold, per arm

- **Recall per transform:** share of variants grouped **with their original** (the stricter of the
  baseline's two counts); for crops, share of positive crop pairs in one group.
- **False-group rate on distinct pairs:** share of groups that contain two different originals
  (frame, core) or two different objects (crop), after the person check corrects the ground truth;
  with a Wilson 95 % interval (`statsmodels.stats.proportion.proportion_confint`, `method='wilson'`).
- **Pair precision**, **group purity**, group size distribution, largest group (chaining shows up
  here first).
- **Crop only:** same-image distinct-object false-pair rate; bucket loss (global minus bucketed
  recall).
- **Order sensitivity (core):** spread of recall over the 3 arrival orders.
- **Stability across sources:** the best threshold chosen on COCO originals only and on Open Images
  originals only; their difference (feeds J4).

### 5.2 Bars (from research plan 4.6, RQ-2; per path, on the test half)

| Bar | Value |
|---|---|
| B-1 false groups | false-group rate on distinct pairs <= 0.5 % (at most 1 group in 200 mixes distinct photos or objects in the person-checked sample) |
| B-2 recall, re-encode and resize | >= 0.90 on T-q60 and on T-half |
| B-3 recall, framing | >= 0.80 on T-crop90 |
| B-4 no regression | on every other transform, recall not below today's default on the same test half |

Verdict per path: PASS if all bars pass at some threshold (and arm), FAIL if none does,
INCONCLUSIVE if the person-checked sample cannot bound B-1 (Wilson half-width above 0.25 %).
Choice rule when several thresholds pass: the **highest** cosine threshold that passes (the most
conservative), single linkage preferred over complete linkage at equal bars (no new code). The
bars are not moved after the read.

---

## 6. Requirement R4 — visual verification and the human re-judge

- **What is checked:** the boundary. Pair sheets of (a) variant/original pairs just below the chosen
  threshold (what would be missed), (b) distinct pairs just above it (what would be wrongly grouped),
  (c) the largest groups at the chosen threshold (chaining), (d) for crops, same-image pairs above
  it.
- **Sample** (owner decision O3): at most 10 pair sheets and 20 group sheets per path; half
  targeted (a to d), half a seeded uniform control; targeted sheets never pooled into estimates.
- **Judge agreement first** (research plan 4.4): the hosted judge has never been scored on the task
  "same photograph versus different photograph". Plant 40 sheets from the calibration half: 20
  known same (constructed variants, including hard ones such as T-quarter and T-crop80) and 20
  known different, half of them person-confirmed distinct pairs above cosine 0.95 (easy negatives
  make the calibration vacuous; the balanced-sheet lesson, research plan 1.3 item 8). Bars: pair
  kappa >= 0.7, flag precision >= 0.8. Below them the judge only ranks sheets for a person (J5).
- **Privacy routing (hard rule):** COCO and Open Images photos often contain people. A sheet goes to
  a hosted model only if none of its images has a COCO `person` box or an Open Images `Human face`,
  `Person`, `Man`, `Woman`, `Boy`, `Girl` or `Vehicle registration plate` box; every other sheet
  goes to the local VLM or a person (`src/services/labeling/contact_sheet.py` already refuses a
  hosted call for face and plate kinds, `tests/test_contact_sheet.py`; the routing here is by the
  dataset boxes because the core path has no face kind on a whole image).
- **Human re-judge:** blind, seeded 5 % of verdicts, at least 20, by the owner or a delegate;
  judge-vs-human kappa below 0.7 suspends the judge for this task.
- **Reporting:** beside the ground-truth metrics, never instead of them; "insufficient sample" when
  a Wilson half-width exceeds 0.10.

---

## 7. Requirement R5 — landing (only on a per-path PASS and the owner's yes)

### 7.1 Core

- One constant `DEFAULT_NEAR_DUP_COSINE` (the PASS value) and a derived
  `DEFAULT_NEAR_DUP_SCORE = (1 + DEFAULT_NEAR_DUP_COSINE) / 2` in
  `src/services/duplicate_detection/models.py`; delete `DEFAULT_SIMILARITY_THRESHOLD`.
- Replace the seven `0.99` literals (section 2.1) with the constant. Route `Form` defaults reference
  it so OpenAPI shows the new value.
- Rewrite the comments: state the scale ("OpenSearch `cosinesimil` score, `(1 + cos) / 2`") and
  remove the Immich claim, or restate it correctly (Immich `maxDistance` 0.01 is cosine distance
  0.01, i.e. cosine 0.99, in Immich's own CLIP model, so it is not comparable; Immich config docs,
  <https://docs.immich.app/install/config-file/>, `duplicateDetection.maxDistance` default 0.01).
- Route bounds: decide with J1 whether `ge=0.90` stays (cosine 0.80 groups unrelated images; the
  baseline's 0.90 row already mixes distinct images).
- J6 decides what happens to existing groups.

### 7.2 Frame

- `DEFAULT_FRAME_DEDUP_THRESHOLD` becomes the PASS value; its docstring (`frame_dedup.py:15-19`)
  is rewritten with the evidence note's numbers (the 0.96-0.97 "different angle" claim is checked,
  not repeated).

### 7.3 Crop

- A separate `DEFAULT_CROP_DUP_THRESHOLD` in `near_dup.py` (today it reuses the frame constant,
  `near_dup.py:38-41`, `:55`), with the docstring's "hypothesis" sentence replaced by the note's
  result.
- `OP_CROP_DUP_THRESHOLD` read in the startup config, invalid value refused (B2 2C-7; if 2C-7 has
  already landed, nothing to do).
- `NEAR_DUP_VERSION = 'v2'` and a re-score **clears** `dup_*` on every scored item before writing new
  groups, so no stale `dup_is_representative=false` hides an item (section 2.4). Group ids become
  stable per run (`dup_<run id>_<n>`) so ids from two runs cannot collide.
- `env.template:590` updated; promotion to a validated scorer (`strategy_flags.py`) is a separate
  owner decision and not part of this landing.

### 7.4 Docs, wire, tests

- `docs-site/docs/api-reference/core.mdx`, `docs/CURATION.md` (export dedup and scores),
  `env.template`, `CHANGELOG.md` ("Changed: default near-duplicate threshold ..." with the scale
  stated); `docs/research/retrieval_baseline.md` gets an appended dated line pointing at the new note
  (never rewritten).
- `contracts/` holds only the curation OpenAPI (`contracts/openapi/curation.json`); if the crop or
  frame default appears there (for example through `dedup_threshold_default` in the methods payload),
  run `make contracts` and commit in the same commit.
- Tests: section 8.2.

---

## 8. How we PROVE it — verification plan

### 8.1 E-2 (research)

| Claim | Proof | Red looks like |
|---|---|---|
| Transforms isolate factors | builder unit test: T-half output decoded equals the Lanczos resize of the decoded original (PNG) or is within the q95 error budget; T-q60 output dimensions equal the original's | a resized variant whose JPEG quality differs from the registration |
| Scale conversion right | a unit test on a tiny real OpenSearch-free fixture: for unit vectors with cosine c, the emulation's score is `(1 + c) / 2`; plus the end-to-end agreement of 4.3 (>= 98 %) | the emulation groups at cosine where the route groups at score |
| Product functions used | the sweep script imports `near_dup_groups`, `compute_near_dup_groups`, `matches_from_hits`; a test mutates `near_dup_groups` (monkeypatch to a no-op) and the sweep output changes | a copied union-find that passes while the product's differs |
| Bars fixed before scores | registration commit precedes results; results JSON stamps the registration sha | |
| Data kept | `docs/benchmarks/near_duplicate_threshold/SHA256SUMS` verifies; `docs/benchmarks/retrieval_baseline/` untouched (`git diff --stat` empty there) | a pruned or rewritten baseline file |
| Visual check honest | agreement table, Wilson intervals, re-judge kappa | a pass with a half-width above 0.10 |

### 8.2 Landing (product)

| Test | Proves | Red today |
|---|---|---|
| `tests/test_near_dup_defaults.py::test_single_core_constant` | no `0.99` literal remains as a near-duplicate default: an AST scan of the seven call sites (function defaults and `Form(...)` defaults named `*near_duplicate_threshold*` or `threshold` in the duplicate modules) resolves every default to the constant | seven literals |
| `...::test_score_scale` | `DEFAULT_NEAR_DUP_SCORE == (1 + DEFAULT_NEAR_DUP_COSINE) / 2` and the OpenAPI default of `/ingest` equals it (read from `app.openapi()`) | |
| `tests/curation/test_near_dup_rescore_clears.py` | a second score at a higher threshold clears `dup_*` on items no longer grouped (fake OpenSearch records the bulk body) | stale fields survive today |
| `...::test_group_ids_unique_across_runs` | two runs never share a group id | `dup_0` twice |
| `tests/curation/test_near_dup_threshold_config.py` | invalid `OP_CROP_DUP_THRESHOLD` refuses at config load; unset uses `DEFAULT_CROP_DUP_THRESHOLD` (2C-7's test, if not already landed) | silent default today |
| `tests/test_duplicate_detection_service.py` (extend) | `matches_from_hits` boundary at the new default (equal passes, just below fails) | |
| `tests/curation/test_frame_dedup.py` (extend) | default value and the holdout-preferring survivor unchanged | |
| core ingest goldens `tests/golden/ingest_single_golden.json`, `ingest_batch_golden.json` | updated only where the default appears, with the diff reviewed line by line | |

Every new test is watched failing first; the PR lists each red run. One full offline sweep on the
merged branch (`.venv/bin/python -m pytest tests/ -q --no-cov -m 'not live'`, `pre-commit run
--all-files`, `make contracts-check`, docs check), not per commit.

### 8.3 Empirical after landing

The 4.3 end-to-end confirmation at the new default is the acceptance record; no new GPU window is
opened for it.

---

## 9. Corrections to the issue's premises

| # | Issue or plan says | What is true | Consequence |
|---|---|---|---|
| C1 | "default 0.99" as a cosine (issue title; research plan 1.1 lists "core 0.99" beside "frame 0.98" as if comparable) | core compares an OpenSearch `cosinesimil` score, `(1 + cos) / 2`; 0.99 is cosine 0.98, the same cut as the curation paths, in another space | sweep in cosine, print core on both scales; J1 |
| C2 | "matches Immich's maxDistance=0.01" (`models.py:7`, `ingest_single.py:108,293`, `service.py:21`) | Immich's `maxDistance` is a cosine distance in its own CLIP model (default 0.01 per the Immich config docs), i.e. cosine 0.99, score 0.995; and the models differ, so no number transfers | remove or correct the comment at landing |
| C3 | "catches only 34/100 q60 JPEG recompressions and 13/100 half-size resizes" | true for "grouped"; "grouped with the original" was 24 and 9. All four baseline variants were also JPEG q90 re-encodes, so "half size" is resize plus recompression | the new set isolates factors and reports both counts |
| C4 | "Sweep the threshold ... recompressed and resized variants" (one threshold) | three paths with different spaces, scales and algorithms (section 0 table) | one verdict and one default per path |
| C5 | research plan RQ-2: duplicate groups "are must-links, one vote per group in gates, and the train/test split guard" | none of these is wired today: no `dup_` read in the loop, no duplicate logic in the holdout freeze; leakage protection only in export when `dedup_threshold` is set | the evidence note must not claim downstream effects that do not exist; the gaps are recorded for the coordinator (J7) |
| C6 | `docs/design/program/03b_b2_w2_job_families.md:117`, 2C-7 control: "unset gives the default and `0.97` is used" | the default is 0.98 (`near_dup.py:55` via `frame_dedup.py:34`); `0.97` presumably means "a valid set value is used" | the 2C-7 test should assert the constant, not a literal |
| C7 | "Report the precision / recall trade-off and the false-merge rate on true non-duplicates" | true non-duplicates cannot be assumed: COCO and Open Images contain real near-duplicates of their own | person-check every distinct pair above the lowest swept threshold (budgeted) |

---

## 10. Judgment calls (J1 … J8)

| J | Question | Recommendation | Trade-off |
|---|---|---|---|
| J1 | Scale on the core wire: keep `near_duplicate_threshold` as an OpenSearch score, or switch it to cosine | keep the score scale (it is what the field has always meant) and document it; change only the default; tighten `ge` to the score of the lowest cosine the study supports | a cosine field would match curation and be clearer, but it silently changes the meaning of every stored client call: a breaking change for no measured gain |
| J2 | Factor-isolated variants as PNG or JPEG q95 | PNG if every product decoder path accepts it (check first, trap T5), else JPEG q95 with its error budget stated | PNG isolates exactly; q95 matches what decoders always accept |
| J3 | If complete linkage passes where single linkage fails | adopt complete linkage only for the path where it is needed, inside the existing union-find component (`AgglomerativeClustering` per component), and only after its runtime at 350k crops is measured | new code in a scorer; single linkage stays wherever it passes |
| J4 | One default per path or a project setting | one default per path (research plan OQ-9); a project setting only if the COCO-only and Open Images-only best thresholds differ by more than 0.01 cosine | a setting nobody can calibrate is a trap; a fixed default may be wrong for very different data |
| J5 | Judge | the hosted judge only on person-free, plate-free sheets and only after the planted calibration passes; everything else local VLM ranks and a person decides | person time (about 1 hour per path) |
| J6 | Existing core groups after a default change | document that the default applies to new ingests; offer a re-group job only after B2 as a Celery task (`op.jobs.duplicate_regroup`, queue `jobs`, a lease, idempotent by image) if the owner wants it | without a job, a library holds groups made at two thresholds; with one, a new job family |
| J7 | The defects of section 2.4 (stale crop fields, racy batch grouping, no duplicate logic at holdout freeze, the swallowed exception) | stale-field clearing goes into this landing (the default change makes it bite); the batch race and the holdout leakage become their own tracked fixes (they exist at any threshold); the swallowed broad except is listed for the fail-open ratchet | bundling everything delays the default change; splitting needs two more issues |
| J8 | If no threshold meets both bars but one is close on recall | keep today's default on that path (stop rule), publish the curve; the owner may choose a recall-leaning default knowingly | a better-recall default with a measured false-group rate above 0.5 % versus the status quo |

Precondition on J6 (constraint on unwritten code, not a finding): a re-group job writes every
grouped document of the core index; it needs the same review of bulk-write scope and idempotency
that every bulk writer gets, and must never run in a request handler.

---

## 11. File-by-file change list

### 11.1 E-2 (research; docs and research scripts only)

| File | Change |
|---|---|
| `scripts/bench/retrieval_eval_data.py` | `make_isolated_variant(src, kind, fmt)` beside `make_variant`; the T-* and C-* kinds |
| `scripts/datasets/near_dup_set.py` (new) | select 5,000 originals from COCO val2017 and Open Images validation, exclude earlier pins, write the pin |
| `scripts/datasets/manifests/near_dup_5000.json` (new) | pin |
| `scripts/research/near_dup/encode_export.py` (new) | drive the research stack: core batch ingest without grouping, curation upload, vector export (reuses `fetch_vectors.py`) |
| `scripts/research/near_dup/sweep.py` (new) | range search, grid, grouping by import, metrics, both scales |
| `scripts/research/near_dup/core_emulation.py` (new) | arrival-order emulation of the core assignment |
| `scripts/research/near_dup/person_check.py` (new) | seeded sample, local review sheet, verdict file outside git |
| `scripts/bench/retrieval_eval_runs.py` | `eval_dedupe` accepts the new pin and a threshold list for 4.3 |
| `tests/research_scripts/test_near_dup_variants.py`, `test_near_dup_sweep.py`, `test_core_emulation.py` (new) | section 8.1 |
| `docs/research/near_duplicate_threshold.md` (new) | registration first, results later |
| `docs/benchmarks/near_duplicate_threshold/` (new) | backing JSON + `SHA256SUMS` |
| `docs/research/README.md`, `docs/design/README.md` | index rows |

### 11.2 Landing (product; per path, only on PASS)

| File | Change |
|---|---|
| `src/services/duplicate_detection/models.py` | `DEFAULT_NEAR_DUP_COSINE`, `DEFAULT_NEAR_DUP_SCORE`; old constant deleted; comments corrected |
| `src/services/duplicate_detection/service.py` | defaults reference the constant; docstring lines 7, 13-25 corrected (no "scan" that does not exist, no Immich equivalence) |
| `src/services/visual_search/ingest_single.py`, `ingest_batch.py` | literal defaults replaced (`:77`, `:276`, `:81`, `:93`); docstrings `:105-109`, `:293` |
| `src/routers/ingest/single.py`, `batch.py`, `directory.py` | `Form`/`Query` defaults and bounds (J1); descriptions state the scale |
| `src/services/detection/frame_dedup.py` | default and docstring |
| `src/services/curation/item_scores/near_dup.py` | `DEFAULT_CROP_DUP_THRESHOLD`, `NEAR_DUP_VERSION = 'v2'`, run-scoped group ids |
| `src/services/curation/item_scores/job.py` | clear `dup_*` for the scored pool before writing a near-dup result (the scorer declares its fields in `writes`, `near_dup.py:102-109`; clear exactly those) |
| `src/config/` | `OP_CROP_DUP_THRESHOLD` validated at startup (if 2C-7 has not landed) |
| `env.template` | line 590 |
| `docs-site/docs/api-reference/core.mdx`, `docs/CURATION.md`, `CHANGELOG.md`, `docs/research/retrieval_baseline.md` (append only) | docs |
| tests | section 8.2 |

---

## 12. Edge cases and traps (the failure register)

| # | Trap | How it silently looks done | Guard |
|---|---|---|---|
| T1 | Wrong scale | the sweep reports cosine 0.98 as "the same as today" for core and recommends 0.98 as the core default, which is score 0.98 = cosine 0.96 | every core row printed on both scales; the scale test of 8.2 |
| T2 | Measuring the route instead of the vectors | the order-dependent online rule and the batch race leak into the threshold curve | offline sweep first, route only for confirmation (4.3) |
| T3 | Re-implemented grouping | a research union-find that differs from the product's (e.g. no upper-triangle filter, or k=all instead of k=10) | import the product functions; the monkeypatch test |
| T4 | Compounded transforms | "resize" also recompresses, so resize looks worse than it is | factor isolation (3.2); the baseline variants kept as one compound row for continuity |
| T5 | A format the product cannot decode | PNG variants silently fail on a GPU JPEG decoder path and are dropped from the denominator | pre-flight one file per format per path; failed decodes are counted, never dropped |
| T6 | Global vs bucketed crops | the crop sweep runs global union-find and reports recall the product never achieves | report both; the bucketed one is the product number |
| T7 | No IVF store | the crop scorer raises without centroids; a research run quietly falls back to global | run the project's clustering once; refuse to score crops without exported centroids |
| T8 | Dataset duplicates counted as false groups | COCO's own near-duplicates inflate the false-group rate and push the threshold up | person check of every distinct pair above the lowest swept threshold (budgeted, sampled with an interval beyond 200) |
| T9 | Faces to a hosted judge | whole-image COCO sheets contain people | routing by dataset boxes (section 6); a test that a sheet with a `person` box is refused for the hosted sender |
| T10 | Vectors from another pipeline | a later B5c change of crop preprocessing changes the crop space and invalidates the crop default | record the image tags and preprocessing; re-run the crop path after B5c if it lands later (research plan 7.2) |
| T11 | Stale fields after a default change | the default changes, a re-score runs, `hide_near_duplicates` still hides items from the old grouping | clear-before-write (7.3) and its test |
| T12 | Existing core groups | the default changes and old groups keep the old cut; stats mix both | J6, stated in CHANGELOG |
| T13 | One threshold read on the calibration half | choosing and reading on the same half overstates recall | split by original; read the test half once |
| T14 | HNSW vs exact | the route uses approximate HNSW with k=10; the offline sweep is exact; a pair the index misses is counted as a recall the route cannot reach | the 4.3 agreement check; report how many offline pairs the route's top-10 did not return |
| T15 | Benchmark data edits | someone "refreshes" `dedupe_r1.json` | new folder only; `SHA256SUMS`; additions-only diff |
| T16 | Private data | the owner's photo bursts are the most natural near-duplicates, and tempting | owner-run option only, aggregates only, never named |

---

## 13. Suggested phasing

| Phase | Content | Gate | Where |
|---|---|---|---|
| P0 (E-0) | set builder, variant kinds, sweep and emulation scripts with unit tests | offline tests green | any time; research scripts only |
| P1 | registration committed (J2, J4 and the bars) | owner's yes on the registration text | any time |
| P2 (E-2) | encode on the research stack (RW-1), export vectors | vectors and decode counts recorded | RW-1, after the v0.7.0 tag |
| P3 | offline sweep, person checks, test-half read | registered rule applied per path | CPU slot; about 1 person hour per path |
| P4 | visual check and human re-judge | agreement bars | API budget plus local VLM in RW-1 or RW-3 |
| P5 | end-to-end confirmation on core (4.3) | >= 98 % agreement | RW-1 |
| P6 | decision per path | owner | |
| P7 | landing per passing path as one PR (core, frame, crop in separate commits), the J7 split decided | full offline sweep once on the merged branch | after B2 (2C-7 is in B2) |

If every path stops, #416 closes with the note and the curves; the defaults stay; the code comments
are still corrected (C2) because they are wrong at any threshold.

---

## 14. Libraries and documented patterns consulted

| Need | Library / document | What it says that this plan relies on |
|---|---|---|
| Score scale of the core path | OpenSearch "Spaces" (<https://docs.opensearch.org/latest/mappings/supported-field-types/knn-spaces/>) | `cosinesimil`: `d = 1 - cos`, score `(2 - d) / 2`; Faiss normalizes vectors for this space |
| Immich comparison | Immich config file docs (<https://docs.immich.app/install/config-file/>); the community probe tool that runs Immich's own query (<https://github.com/dhcgn/immich-clip-probe>) | `duplicateDetection.maxDistance` default 0.01, a pgvector `<=>` cosine distance on Immich's CLIP model |
| Exact range search | FAISS wiki, "Special operations on indexes" (<https://github.com/facebookresearch/faiss/wiki/Special-operations-on-indexes>) | `range_search` on CPU flat indexes; exact and deterministic |
| Complete linkage with a threshold | scikit-learn `AgglomerativeClustering` (<https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html>) | `metric='cosine'`, `linkage='complete'`, `distance_threshold` with `n_clusters=None` |
| Wilson intervals | statsmodels `proportion_confint` (<https://www.statsmodels.org/stable/generated/statsmodels.stats.proportion.proportion_confint.html>) | false-group rate and judge purity intervals |
| Image transforms | Pillow `Image.resize`, `Image.rotate`, `ImageEnhance` (<https://pillow.readthedocs.io/en/stable/>) | deterministic resampling and encoding |

No new grouping algorithm is written: the product's union-find and kNN assignment are imported, and
the only new method arm (complete linkage) is a scikit-learn call inside the existing components.
