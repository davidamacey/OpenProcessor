# OpenProcessor: attribute-signal layer for clustering (issue #400)

**Status: PLAN ONLY, NOT IMPLEMENTED.** The issue stays parked. This plan says what must be true
before any part of it runs, how each signal is tested on real data, the stop rule, and what lands in
the product if a signal passes.

- Written 2026-11-04 against branch `int/b1-correctness` at `81c4e4ef` (the newest code; B1 merged).
- Builds on: `docs/design/research_clustering_plan.md` (RQ-6 is this issue; experiment E-8),
  `docs/design/cluster_search_makemodel_plan.md` section 6.7 (the original design, parked),
  `docs/research/cluster_program_outcome.md`, `docs/research/cross_domain_clustering_lessons.md`
  (T11, section 6.2, experiment E8), `docs/design/program/00_index.md` (research is outside the
  0.7.0 program).
- Sibling: #401 (registration-plate identity text) has its own plan. It is one of the signals here,
  but its privacy design is separate and comes first.
- Audience: a fresh agent with no memory of this session. Symbols are cited as `path:LINE` at the
  commit above; re-find them with `rg` if lines move.

---

## 0. TL;DR

**What I recommend**

1. **Keep #400 parked.** It cannot start until three other lines have answered:
   - RQ-2 / #416 (the near-duplicate threshold);
   - RQ-1 / #398 (detector-class roots);
   - RQ-4 and RQ-5 (the graph method, and the constraint machinery that consumes a must-link or
     a veto).

   Today no clustering path in the product can consume a must-link or a cannot-link: IVF and the
   per-cluster AHC refine take only vectors.
2. **Split the issue's signal list in three.** The split follows what public COCO and Open Images
   can actually test, and the numbers here are measured, not assumed:
   - **Testable on public data now:** the near-duplicate group as a must-link (class level), and a
     zero-shot view tag (class level, as a prototype assigner and stratum, never a veto). Colour
     too, as a VLM pack attribute that already exists, but only as a class-level stratum; its
     identity-level veto role has no public ground truth.
   - **Already owned by another issue:** detector class as a partition or a veto is the RQ-1 /
     #398 experiment (Z5). #400 does not run it again; it consumes #398's verdict.
   - **Not testable on public COCO or Open Images:** EXIF capture time, bursts, camera id and GPS.
     A census of the local public samples found EXIF capture time on **0 of 2,000 COCO images**
     (`coco_bench_2000`) and on **1 of 300 Open Images images** (the plate pin, fetched from the
     dataset's S3 mirror). The mirror copies have their metadata stripped. Identity-level vetoes
     (colour, plate) also have no COCO or Open Images identity labels. These signals are
     **owner-run only on the private set**, or need a public set that carries EXIF, which is the
     owner's call (J3).
3. **Run E-8 as four registered sub-experiments.** These are E-8a feasibility, E-8b near-duplicate
   must-link, E-8c view tag, and E-8d colour stratum. Each has a bar written before any score.
   Each sub-experiment's stop rule is independent, so one failure does not hold up the others.
   E-8a (feasibility) needs no constraint machinery and can run in any CPU slot after the v0.7.0
   tag.
4. **The product lands only what passes, as Celery jobs after B2.** Each signal follows a repo
   pattern that already exists:
   - The view tag becomes a new `CropScorer` in the `item_scores` job family. It needs no new job
     kind (`src/services/curation/item_scores/base.py`).
   - EXIF fields are a header read inside the decode ingest already does
     (`src/services/curation/image_decode.py:80`), plus a header-only backfill job.
   - The mapping is one additive module on the `assist_fields.py` pattern.
   - The rules ("no attribute decides alone") are pure functions in `src/domain/`.

**What I am NOT deciding** (Judgment calls, section 9):

- whether a public EXIF-bearing set (outside COCO and Open Images) may back a published claim (J3);
- whether E-8 waits for E-7 (RQ-5 machinery), or runs its must-link and veto arms on the research
  harness's own constraint pre-processing (J1);
- whether GPS is stored at all in curation projects (J5).

---

## 1. Context: why the obvious approach is wrong here

The obvious approach is "add a `signals` object to items, compute colour, view, EXIF and plate, and
feed them into clustering as must-link and veto edges". The makemodel plan's 6.7 table reads that
way. It fails on four counts.

1. **There is nothing to feed them into.** Product clustering is FAISS IVF with K = min(512, n // 6)
   (`src/services/curation/clustering/methods/ivf.py`). On-demand refinement is complete-linkage
   AHC over a dense distance matrix (`src/services/curation/clustering/refine.py`). Neither takes a
   constraint. The constrained machinery (versioned partitions, `links[]` evidence, a merge
   validator) was the subject of #343. #343 is **closed**: it was folded into the #346 tracker and
   becomes N-5 in the research plan, built **only if RQ-4 wins** (`research_clustering_plan.md`
   sections 6.2 and 6.5). An attribute layer now would be a feature with no consumer.
2. **The measured limit is the encoder, not missing side signals.** The program outcome says
   fine-grained grouping "is an embedding problem, not a clustering problem"
   (`docs/research/cluster_program_outcome.md` section 1). Above pair recall of about 0.3, the
   neighbourhoods themselves are the limit (`research_clustering_plan.md` section 1.3, lesson 2).
   An attribute layer could help in narrow strata (a front and a rear that embeddings cannot link,
   frames of one burst). It does not move the headline numbers, so it must be measured where it
   can matter. Otherwise it reads as noise.
3. **Public data cannot test half of the list.** See TL;DR point 2: EXIF is effectively absent from
   the COCO and Open Images copies the repo fetches. The owner rule is "real data, public COCO or
   Open Images for claims; the private set is owner-run only". Under that rule, EXIF, burst and GPS
   claims can only be owner-run numbers.
4. **Several "new" signals already exist under other names, or belong to other issues:**
   - colour is a VLM pack attribute (`examples/prompt_packs/general_objects.json:62`);
   - near-duplicate groups are stored (`dup_group_id`; see section 2.2);
   - the detector-class partition is #398's Z5;
   - the plate key is #401.

   Re-deriving any of them in #400 would duplicate code and experiments.

---

## 2. What is actually there today (measured, not estimated)

### 2.1 Clustering and its inputs

| Thing | Where | Fact |
|---|---|---|
| Item clustering | `src/services/curation/clustering/methods/ivf.py:1` | FAISS IVF partitioning; a Voronoi cell is not a discovered boundary (module docstring, line 25) |
| Refine | `src/services/curation/clustering/refine.py:1` | sklearn AHC, complete linkage, cosine, `distance_threshold=0.25`, full pairwise matrix, skip above `OP_MAX_REFINE_MEMBERS` (default 8000) |
| Region-box clustering | `src/services/curation/clustering/region_box_clustering.py` | MiniBatchKMeans plus AHC refine, on demand |
| Methods registry | `src/services/curation/clustering/methods/` | `ahc.py`, `hdbscan.py`, `ivf.py`, `base.py`; none takes constraints |
| Cluster write guard | `src/services/curation/clustering/cluster_write_guard.py` | the only writer of cluster fields; every product landing goes through it |
| Research graph library | `scripts/research/graph_lib.py`, `scripts/research/leiden_refine_proto.py` | the #342 prototype: mutual-kNN plus Leiden CPM; research only |
| Research diagnostics | `scripts/research/cluster_diag/` (`partitions.py`, `loop_sim.py`, `merge_stage.py`, `heldout.py`, ...) | the harness E-8 runs on |
| Attribute pair metrics | `scripts/bench/cluster_eval_metrics.py:246` `attribute_pair_scores` | pair recall and false-merge rate for pairs sharing or differing in an attribute: **already the metric E-8 needs**; do not rewrite |

### 2.2 Signals that already exist

| Signal | Field / source | Where | Consumers today |
|---|---|---|---|
| Near-duplicate group (crop) | `dup_group_id`, `dup_group_size`, `dup_is_representative` | written by the `near_dup` scorer, `src/services/curation/item_scores/near_dup.py`; threshold `OP_CROP_DUP_THRESHOLD` (`near_dup.py:55`), default `DEFAULT_FRAME_DEDUP_THRESHOLD = 0.98` (`src/services/detection/frame_dedup.py:34`); the module docstring says the crop threshold "is a hypothesis, not a finding" | review requests and semantic search collapse to the representative (`src/services/curation/review_request.py:92`, `src/services/curation/semantic_search.py:142`). **Not** used by clustering or by the loop's gate (`rg -n dup_is_representative src scripts` lists no clustering or autolabel file) |
| Near-duplicate group (core API) | `duplicate_group_id` | `src/services/duplicate_detection/models.py:21` | core-only; a different name and a different subsystem from the curation field |
| Detector class | `detector_class_name`, `detector_confidence` on items (`src/services/curation/wire.py:348`) | ingest | #398 (Z5) tests it as the loop's root |
| VLM pack attributes (colour, material, brand, make, model) | `assist.attributes[]` nested (`key`, `value`, `confidence`) | `src/clients/curation_opensearch/assist_fields.py:22`; packs `examples/prompt_packs/general_objects.json` (colour vocabulary with 11 values) and `vehicle_wheel.json` (make, model) | shown in the UI as suggestions; "stored apart ... so no export, training or holdout path reads it" (`assist_fields.py:1`) |
| Item OCR text | `item_text_lines`, `item_text_tokens` | `src/domain/item_text.py:38` | `GET {P}/crops?item_text=` search |
| Region text | `region_boxes[].text*` | `src/domain/region_boxes.py:59` | review, search; see #401 |
| PE text tower | `PeEncoder.encode_text` | `src/clients/pe_encoder.py:504` | curation text search; **reusable for a zero-shot view tag with no pixels** |

### 2.3 Signals that do not exist

| Signal | Evidence of absence |
|---|---|
| EXIF capture time, sub-second, offset | `rg -n -i "DateTimeOriginal\|capture_time\|taken_at" src` returns nothing. The only EXIF read in ingest is orientation: `src/services/curation/image_decode.py:80` (`header.getexif().get(EXIF_ORIENTATION, 1)`) |
| GPS, camera model, body serial | same `rg`: nothing |
| Burst or session id | nothing; the word "burst" appears only in comments about near-duplicate photo bursts (`near_dup.py:12`) |
| View tag | nothing stored; `attribute_pair_scores` takes any per-item label, so research can compute one offline |
| A `signals` object on items or images | the images body (`src/clients/curation_opensearch/bodies_core.py:73`) has `width`, `height`, `imohash`, `phash`, `indexed_at`, embeddings: no capture metadata |

### 2.4 EXIF on the public samples (measured 2026-11-04)

Read-only census with Pillow 12.3.0 (`Image.getexif()`, `get_ifd(ExifTags.IFD.Exif)`,
`get_ifd(ExifTags.IFD.GPSInfo)`) over the local copies the repo fetchers produce:

| Sample | Files | Any EXIF | DateTimeOriginal | GPS | Camera model |
|---|---|---|---|---|---|
| `coco_bench_2000` (COCO val, the retrieval pin) | 2,000 | 0 | 0 | 0 | 0 |
| Open Images plate pin (`oi_plates_300.json`, S3 mirror `IMAGE_DOWNLOAD_TMPL` in `scripts/datasets/fetch_openimages_plates.py:82`) | 300 | 1 | 1 | 1 | 1 |

The Open Images `OriginalURL` column points at the original Flickr files, which may still carry
EXIF. Those are not the dataset files, and many are gone. **Conclusion: EXIF signals cannot
produce a public-data quality claim from COCO or the Open Images mirror.** The E-8a reproduce
command is in section 8.

### 2.5 Program position

- Research is outside the 0.7.0 program: #400 is listed among "Research and platform breadth are
  out of this program" in `docs/design/program/00_index.md` section 10 ("Scope boundaries"). It is mapped "defer, stays
  parked" in `docs/design/program/09_github_actions.md:643`.
- Research windows start after the v0.7.0 tag (`research_clustering_plan.md` section 5). Phase R0
  (registering bars) is docs-only and may happen any time.
- Every product change after B2 is a Celery job under a B1 lease (`research_clustering_plan.md`
  section 6.1; `docs/design/program/03_b2_overview.md` job inventory: `op.jobs.item_scores` is
  family 3).

---

## 3. Design principles for every signal

These are carried from the makemodel plan 6.7 and cross-domain T11. They are restated as testable
rules.

1. **No attribute decides a merge or a split alone.** A must-link also needs embedding similarity
   above a calibrated floor. A veto also needs weak embedding similarity. Only a human correction
   acts alone. Plates at identity level are the one stated exception in 6.8, and #401 owns it.
2. **Every signal has a type, a confidence and a role.** The role is one of:
   - `must_link`: contraction or a strong positive edge;
   - `veto`: a cannot-link, hard in the merge validator only when stated;
   - `soft`: an edge weight change;
   - `stratum`: reporting only;
   - `assigner`: chooses which prototype a member is compared to.

   The role is data in a registered table, not code branches.
3. **Level is explicit.** A signal is valid at `class` level, at `identity` level, or both. The
   graph builder refuses a signal at a level it is not registered for (fail closed).
4. **Calibrated per project and space.** No fixed cosine floor: the space is anisotropic (mean
   random-pair cosine 0.50 on COCO; `research_clustering_plan.md` 1.3 lesson 4). Every floor is
   calibrated on the dev half for its space and model, and carries a calibration id.
5. **Class identity by name.** A class-compatibility check compares class names, never ids. This
   covers the detector class, registry names and the coarse class in a veto.
6. **One vote per near-duplicate group.** When a near-duplicate group is contracted, it counts as
   one item in every statistic, gate and calibration. Otherwise a burst of 40 identical frames
   dominates a cut.
7. **Provenance.** A machine edge from an attribute is recorded as `attribute:<name>` evidence with
   its confidence and calibration id. It never sets `*_validated`. The lock rule
   (`src/clients/occ_locks.py`) stays the only gate for writes.
8. **Order independence.** Signal extraction and edge building sort by document id and seed from a
   stable key (#372 rule). Permutation ARI must stay 1.000.

---

## 4. Signal by signal

Each subsection says what the signal is, what exists, how it is computed (with the library), how it
is tested, and its stop rule.

### 4.1 Near-duplicate group as must-link (class level, and identity level by construction)

- **What:** items in one `dup_group_id` are contracted to one node before clustering. They then
  count as one vote.
- **Depends on #416 (RQ-2).** The crop threshold of 0.98 is an unvalidated hypothesis
  (`near_dup.py` docstring). A must-link built on a wrong threshold is a false merge factory.
  ✅ Recommended: E-8b runs only after E-2 (the #416 experiment) has fixed a per-path threshold.
  E-8b then uses that threshold, never 0.98 by default.
- **Library:** contraction is pre-processing around the clustering call. The research plan's
  section 3.2 explains why this is the documented way to impose must-links with scikit-learn
  `AgglomerativeClustering` (`connectivity`) or leidenalg (`is_membership_fixed`), since no
  maintained constrained-clustering library fits. Groups come from the stored field; no new math.
  - Contraction for vectors: the group's medoid (a real member). Never the mean, because of
    principle 4 of the makemodel plan (multi-prototype).
- **Bucket artefact:** near-duplicate groups are computed inside IVF buckets only (`near_dup.py`
  docstring, "Cross-bucket near-dups are consequently never found"). E-8b measures how many true
  near-duplicate pairs straddle buckets on the E-2 transform set. If more than 5 % straddle, the
  must-link under-reports. The finding is recorded, and #416 owns the fix.
- **Test:** BCubed recall at a fixed false-merge rate, with and without contraction. Also the
  loop simulator's U1 (decisions per item) with one vote per group, because counting duplicates
  as votes is a gate-statistics bug in its own right.
- ❌ Rejected: using near-duplicate groups as a soft edge weight. A group at a validated threshold
  is the same picture. Contraction is cheaper and cannot be outvoted.

### 4.2 Detector class (owned by #398)

- #398 (Z5) tests detector-class roots with and without the hold-back guard
  (`research_clustering_plan.md` RQ-1, E-1). That is the class-level partition use of this signal.
- What #400 adds is only the **veto across coarse classes when both detections are confident**.
  For example, never merge a confident "dog" with a confident "car" at identity level.
  ✅ Recommended: fold this into #398's verdict rather than a separate experiment. If Z5 adopts
  class roots, the cross-class veto is implied: groups never span roots. If Z5 fails, a
  cross-class veto alone is measured in E-8b as one arm (cheap: the class name is stored).
- Confidence: detector confidence is badly under-calibrated (makemodel plan 6.1, prior-art
  lesson). A "confident" threshold is calibrated with `sklearn.isotonic.IsotonicRegression` on the
  dev half against Open Images labels, as the research plan's section 3.1 lists.

### 4.3 View tag (zero-shot, from stored vectors; class level)

- **What:** score each crop's stored `pe_embedding` against PE text-tower embeddings of project
  view prompts. Examples: "a photo of the front of a {coarse class}", rear, side, front
  three-quarter, rear three-quarter. The prompts are project data in the region profile or the
  prompt pack. There is no image re-read and no new model.
- **Library and pattern:** CLIP-style zero-shot classification (Radford et al. 2021,
  https://arxiv.org/abs/2103.00020), using the encoder already loaded:
  `PeEncoder.encode_text` (`src/clients/pe_encoder.py:504`). Margin = top-1 minus top-2 cosine. It
  is calibrated to a confidence with isotonic regression on the dev half.
- **Role:** `assigner` (chooses the prototype a member is compared with) and `stratum` (reporting).
  **Never a veto.** Views differ inside one class and inside one physical object (makemodel plan
  6.2 point 2).
- **Ground truth problem:** COCO and Open Images have no view labels. Options:
  - (a) CompCars view labels, which are local-only (licence O2 in the makemodel plan);
  - (b) a person labels a seeded 300-crop sample of Open Images vehicles by view, stored as a
    manifest of content hashes plus labels in `artifacts_local/`, with aggregates only committed;
  - (c) the local VLM as labeller. Ruled out as ground truth: the local VLM judge was killed as an
    authority at pair accuracy 0.51 (`research_clustering_plan.md` section 1.2).

  ✅ Recommended: (b) for the published number, (a) as a local cross-check (J4).
- **Bar (from 6.2):** view-tag accuracy at least 0.8 on the five bins, else the tag stays off. Add
  a minimum sample of at least 60 crops per bin, with Wilson 95 % intervals through
  `statsmodels.stats.proportion.proportion_confint(method='wilson')`. A bin with fewer crops
  reports INCONCLUSIVE.
- **Effect test (only if accuracy passes):** per-view prototypes and cross-view linking. Measure
  identity-free class-level BCubed recall at fixed precision with and without the tag as an
  assigner. Also `attribute_pair_scores` stratified by same view, adjacent view and opposite view.
- **Cost:** one text encode per prompt per project, plus a (n x 5) dot product over stored vectors.
  This is CPU, seconds at 100k crops.

### 4.4 Colour (class-level stratum; identity-level veto is owner-run only)

- **What exists:** `colour` with an 11-value closed vocabulary is a pack attribute in
  `examples/prompt_packs/general_objects.json:62`. It lands in `assist.attributes[]`
  (`assist_fields.py:22`) when an assist run uses that pack. It is a VLM reading: it flips with
  light (cross-domain T11).
- ✅ Recommended order:
  1. **pack attribute first.** It needs no new code, and its cost is VLM time under the loop's
     budget (10 crops per second on one A6000, `docs/PERFORMANCE.md` "Measured VLM labeling
     speed");
  2. **pixel colour only if the pack attribute flips too often.** Use Pillow
     `Image.quantize(colors=k, method=Image.Quantize.MEDIANCUT)` on the crop already decoded at
     ingest (one decode, `docs/research/CLAUDE.md` "One decode per image"). The palette is mapped
     to the same 11 names through CIE Lab distance (`skimage.color.rgb2lab` from scikit-image,
     if it is already present in the image; otherwise the documented sRGB-to-Lab formulas with a
     unit test against published reference values).

  ❌ Rejected: a new colour model. There is no case for one until the cheap signals fail.
- **Role:** at class level a colour is never a veto (two cars of one model differ in colour). It is
  a `stratum` only. At identity level it is a soft veto with weak embedding similarity (T11). Its
  ground truth is identity labels, which COCO and Open Images do not have, so the veto arm is
  owner-run (identity groups from the private set) or uses CO3D v2 sequences if the owner allows a
  non-COCO/OI set (J3).
- **Flip-rate check (public, class level):** on Open Images vehicle crops, find pairs that are
  near-duplicate (same physical object, by construction, after #416) and record how often the pack
  colour differs. Bar: flip rate at most 10 % on near-duplicate pairs. If it is higher, the colour
  veto is not opened at any level.

### 4.5 EXIF capture time, sub-second, offset, camera id, burst (owner-run)

- **Extraction (cheap, and the code shape is known):**
  - Pillow's documented API: `Image.getexif()` then `exif.get_ifd(ExifTags.IFD.Exif)` for
    `ExifTags.Base.DateTimeOriginal`, `SubsecTimeOriginal`, `OffsetTimeOriginal`
    (https://pillow.readthedocs.io/en/stable/reference/ExifTags.html).
  - Camera id comes from `Model` plus `BodySerialNumber` when present. Store only a keyed hash of
    the serial, never the serial (J5 covers GPS; the serial is a device identifier).
  - The read rides on the `header.getexif()` call ingest already makes
    (`src/services/curation/image_decode.py:80`): no second open, no second decode.
- **Burst rule:** frames from one camera id within `burst_gap_s` (project data; start at 2 s) form
  a burst. Unsynced clocks across cameras mean bursts never span camera ids.
- **Role:** a must-link only with high embedding similarity or a #401 key match. A burst is a
  candidate set, not an answer: a passing vehicle and a parked one share a burst.
- **Why owner-run only:** section 2.4. The public copies carry no EXIF.
- **Feasibility first (E-8a):** on the private set, the share of images with
  `DateTimeOriginal`, the share with sub-second time, and the median burst size. Committed as
  aggregates only. If under 30 % of images carry usable time, the burst signal stops (stop rule
  section 7).
- **Spatial work (#348):** EXIF time and bursts are the candidate sets for #348's scene discovery.
  ✅ Recommended: store `capture_time`, `capture_offset`, `camera_key` (keyed hash) and `burst_id`
  in the form #348 needs, so it is designed once (section 5.2). #348 stays parked; this plan only
  avoids a second migration.

### 4.6 EXIF GPS (owner decision, privacy)

- The makemodel plan's O10 recommended "store GPS, project-scoped, never served by default, never
  exported by default". This plan does not decide it (J5). If stored:
  - it lives on the images doc only, in a field that `src/services/curation/wire.py` never
    serialises unless a project setting opts in;
  - no export path reads it (a test enumerates every export writer);
  - project delete removes it (it lives in the project's own index);
  - a purge sub-command on the pattern of `./openprocessor faces purge` (`scripts/purge_face_data.py`)
    removes it with counts-only audit.
- Role: feature for K1 events and places (#348), **never** a must-link or a veto for crops.

### 4.7 Scene and sign OCR text (K1 feature only)

- Exists as `item_text_lines` (curation) and the core `/ocr` index. For whole-image events and
  places only. Never a must-link alone. Not part of E-8: it has no clustering consumer until K1
  whole-image grouping exists (RQ-10, #435 and #436).

### 4.8 Registration-plate identity key (owned by #401)

- #401's plan owns the key, the tiers, the privacy modes and the purge. #400 owns only the generic
  edge interface (section 5.3) that a #401 key plugs into, so the two issues share one
  must-link/veto path and one test suite.

### 4.9 Face-identity link (core only)

- Faces are core-only (owner decision O1). A face link for curation images needs a project-scoped
  face kind, which does not exist. Out of scope; recorded so it is not re-proposed.

---

## 5. Product shape (only for signals that pass)

### 5.1 Where computation runs (after B2)

| Signal | Compute | Job | Lease | Pattern reused |
|---|---|---|---|---|
| View tag | dot product of stored `pe_embedding` against prompt text vectors | `op.jobs.item_scores` with a new scorer `view_tag` | `item_scores` (existing) | `CropScorer` protocol, `src/services/curation/item_scores/base.py:1`; registry `_SCORERS` in `item_scores/__init__.py:18`; `writes` ClassVar names the exact fields |
| Colour (pack) | assist run with a pack declaring `colour` | `op.jobs.assist_run` (existing) | existing | no code |
| Colour (pixels, only if 4.4 says so) | at ingest, on the crop already decoded | inline in ingest (cheap) plus a backfill | backfill under `item_scores` or `reprocess` | one decode per image |
| EXIF time, camera key | header read at ingest | inline in `image_decode` (header already open) | none | `image_decode.py:80` |
| EXIF backfill for existing images | header-only reads | new family `op.jobs.signals_backfill`, queue `jobs` | `signals_backfill` per project | B2 `JobTask` wrapper (bind project, lease, heartbeat, fence, release) |
| Burst ids | sort by (camera key, time), gap rule | inside the backfill job and incrementally at ingest | as above | pure function in `src/domain/` |
| Edges | built when a clustering job runs | inside the clustering job that consumes them (RQ-4/RQ-5's `op.jobs.cluster_refine`) | that job's lease | section 5.3 |

No `asyncio.gather` fan-out in request handlers, and no in-process or file locks for cross-process
state (program PQ-9). Any concurrency inside a job uses `asyncio.TaskGroup` with kept task
references (https://docs.python.org/3/library/asyncio-task.html#task-groups).

### 5.2 Storage

- One additive mapping module `src/clients/curation_opensearch/signals_fields.py` (proposed), on
  the `assist_fields.py` pattern: a `SIGNALS_ITEM_MAPPING` and a `SIGNALS_IMAGE_MAPPING` dict, plus
  an `ensure_signals_fields()` that PUTs them onto the bound project's index and tolerates the
  recoverable mapping conflict (`assist_fields.py:56`).
- Item fields (proposed): `signals.view` (keyword), `signals.view_margin` (float),
  `signals.view_calibration` (keyword), `signals.colour` (keyword), `signals.colour_source` (`vlm`
  or `pixels`).
- Image fields (proposed): `signals.capture_time` (date), `signals.capture_offset` (keyword),
  `signals.capture_subsec` (keyword), `signals.camera_key` (keyword: keyed hash),
  `signals.burst_id` (keyword). GPS only if J5 says yes: `signals.gps` (`geo_point`), never in the
  wire unless the project opts in.
- Wire: new fields go through `src/services/curation/wire.py` only. Run `make contracts` in the
  same commit, and update `docs/design/curation_api_contract.md`.
- The project guard (`src/services/projects/guard.py`) already covers the items and images indexes.
  No new index role is needed.

### 5.3 Edge interface (shared with #401)

A pure module `src/domain/attribute_edges.py` (proposed). It imports only `config` and `utils`,
per the layering rule for `src/domain/`.

```python
@dataclass(frozen=True)
class SignalSpec:
    name: str                 # 'near_dup', 'view', 'colour', 'burst', 'identity_text', ...
    role: Literal['must_link', 'veto', 'soft', 'stratum', 'assigner']
    levels: frozenset[Literal['class', 'identity']]
    embedding_floor: Literal['high', 'low', 'weak_only', 'none']  # which calibrated floor applies

@dataclass(frozen=True)
class AttributeEdge:
    a: str                    # item or box id, sorted so a < b
    b: str
    kind: Literal['must_link', 'cannot_link', 'review']
    signal: str
    confidence: float
    calibration_id: str

def build_edges(pairs, signals, *, level, floors, registry) -> list[AttributeEdge]: ...
```

Rules enforced in `build_edges`, each with a red/green test:

- a `must_link` is emitted only if the pair's cosine is at least the signal's calibrated floor;
- a `cannot_link` from a soft-veto signal is emitted only if the pair's cosine is below the weak
  floor;
- a signal is refused at a level it is not registered for (`ValueError`, fail closed);
- an unknown signal name is refused;
- coarse-class compatibility compares class names (casefolded), never ids;
- output is sorted (deterministic).

`build_edges` has no consumer until RQ-5 lands. It is written when the first signal passes its bar,
not before (no dead code, `docs/VISION_AND_GOALS.md`).

### 5.4 Review UI (Cropwright)

The backend serves the edge evidence (`attribute:<name>`, confidence) on merge and split
suggestions (the research plan's proposed `curation/projects/{project}/clusters/suggestions`,
which does not exist yet). The frontend renders it, per the owner rule "backend data, frontend
render". Every screen is checked in a browser, with full-page screenshots at desktop width and at
430 px.

---

## 6. Corrections to the issue's premises (and to the parked plan 6.7)

| # | Premise | Correction | Evidence |
|---|---|---|---|
| C1 | Signals enter "the graph and the merge validator" and are recorded on "the cluster link (`links[]`)" | No graph, merge validator or `links[]` record exists. #343, which would have built them, is closed into #346. The research plan builds them only as N-5, after RQ-4 passes | `methods/` has `ahc.py`, `hdbscan.py`, `ivf.py`; `gh issue view 343` is CLOSED; `research_clustering_plan.md` 6.5 N-5 |
| C2 | "EXIF time is a header read", implying cheap and broadly available | Cheap, yes. Available on the public data, no: 0 of 2,000 COCO and 1 of 300 Open Images mirror files carry capture time. EXIF signals cannot back a public claim | section 2.4 census |
| C3 | Dominant colour is "not stored" | Colour exists as a closed-vocabulary VLM pack attribute (`general_objects.json:62`), stored in `assist.attributes[]` | `assist_fields.py:22` |
| C4 | Near-duplicate group: "must-link; one vote per group in gates and statistics" (field `duplicate_group_id`) | The curation field is `dup_group_id`; `duplicate_group_id` is the core API's. Neither clustering nor the loop gate reads it. "One vote per group" is unimplemented. The crop threshold is an unvalidated hypothesis (#416) | `wire.py:400`, `near_dup.py` docstring, `rg` in 2.2 |
| C5 | Detector-class prior is a new #400 signal | The partition use is #398 (Z5), an open, registered experiment; #400 consumes its verdict | `research_clustering_plan.md` RQ-1 |
| C6 | The signal layer "lands with the W6 mapping change" | W6 (taxonomy as data) is parked with the rest of the makemodel waves; there is no W6 migration to ride. A standalone additive mapping (5.2) is the right shape | makemodel plan status line; `assist_fields.py` pattern |
| C7 | The makemodel plan's evaluation ground truth for bursts and same-vehicle groups ("session bursts linked by the plate") | That source is the private set only; it gives no public number. It also depends on #401's key, which is itself parked behind a privacy design | makemodel plan 5.4; #401 plan |
| C8 | The research plan's E-8 is "on the RQ-5 machinery" | Partly avoidable: contraction and a veto check are method-agnostic pre- and post-processing, so E-8b and E-8c can run on the research harness's existing arms without waiting for E-7. Whether to do so is J1 | `research_clustering_plan.md` 3.2 last paragraph |

---

## 7. Stop rules and revisit triggers

### 7.1 Stop rule per sub-experiment

| Sub | Stop when | Then |
|---|---|---|
| E-8a feasibility | a signal's coverage on its eligible data is below its floor: EXIF capture time below 30 % of private-set images (owner-run); view-tag accuracy below 0.8 on any of the five bins (or INCONCLUSIVE at n < 60 per bin); pack colour flip rate on near-duplicate pairs above 10 % | that signal is not opened; record "not done, why" in the outcome note |
| E-8b near-dup must-link | BCubed recall gain below +0.02 at a fixed false-merge rate on a stratum with at least 100 affected pairs, or any stratum with at least 100 crops loses more than 0.02 precision | no contraction in the product; one-vote-per-group in the loop's gate is still proposed separately if the loop simulator shows a U1 change (that is a statistics fix, not a signal) |
| E-8c view tag | no gain at matched precision in cross-view pair recall, or accuracy fails | the tag stays off; modes stay unnamed (6.2) |
| E-8d colour stratum or veto | veto: false merges blocked less than 3x true merges lost (RQ-6 bar); stratum: no reporting value, so no product change | colour stays a UI suggestion only |
| Whole line | E-8a shows too little public signal, or no sub-experiment passes | #400 stays parked with an evidence comment; the owner may run the EXIF arms on the private set; no product code |

### 7.2 Revisit triggers (what un-parks #400)

All three of the following, in order:

1. **#416 closed with a validated per-path near-duplicate threshold** (E-2 PASS). Without it, E-8b
   has no trustworthy must-link.
2. **#398 verdict recorded** (E-1 PASS or FAIL). This decides whether the class veto is implied or
   tested.
3. **A consumer exists or is approved.** Either E-5 or E-6 (RQ-4) passes and N-5 (versioned
   partitions with constraints) is approved, or the owner approves J1 (run E-8 on the harness
   without the product machinery).

Plus, for the EXIF arms only: the owner chooses to run E-8a on the private set (J3).

Any one of these makes #400 a no-go for the release cycle in which it happens:

- RQ-4 fails for good (E-5 FAIL on the fresh set);
- RQ-3 changes the encoder (vectors change, so every calibration restarts);
- the owner closes #346.

---

## 8. How we prove it: verification plan

### 8.1 Pre-registration (phase R0, docs only)

Before any score, commit to `docs/research/attribute_signals_eval.md` (proposed; new note):

- each sub-experiment's bar, minimum sample per bar and stratum list;
- the dataset manifests with `manifest_sha256` (section 8.2);
- the calibration split (by image, seeded, 50/50 dev/test; near-duplicates never straddle the
  split, per `research_clustering_plan.md` 4.2).

A bar moved after a read needs a new registration and a reason (research plan rule).

### 8.2 Data (real only)

| Use | Set | Ground truth | Published? |
|---|---|---|---|
| E-8b, E-8c effect | the fresh acceptance set of `research_clustering_plan.md` 4.1 (COCO val2017 plus an Open Images V7 validation sample, about 10,000 images) | human boxes, class names | yes |
| E-8b near-dup | the E-2 near-duplicate transform set (5,000 originals x 8 transforms) | positives by construction | yes |
| E-8c view labels | seeded 300-crop sample of Open Images vehicles (`openimages_vehicles.pin.json`), labelled by a person locally | person labels, aggregates only | yes (aggregates) |
| E-8c cross-check | CompCars view labels | dataset authors | local only (O2) |
| E-8d flip rate | Open Images vehicle crops in near-duplicate groups (after #416) | same object by construction | yes |
| E-8a EXIF, burst arms | the private set | validated items, frozen holdout | owner-run; numbers only |

Synthetic vectors are allowed only for runtime and memory. `scripts/bench/scorecard.py` already
refuses quality fields on a synthetic dataset.

### 8.3 Static proof (unit, offline, `-m 'not live'`)

| Requirement | Test (proposed file) | Red first by |
|---|---|---|
| No attribute decides alone | `tests/test_attribute_edges.py::test_must_link_needs_embedding_floor` | a pair with an equal signal and cosine below the floor emits an edge (stub `build_edges` that ignores the floor) |
| Veto needs weak similarity | `...::test_soft_veto_needs_weak_similarity` | high-similarity pair with a differing colour emits `cannot_link` |
| Level registry fails closed | `...::test_signal_refused_at_unregistered_level` | a view-tag veto at class level is accepted |
| Names, not ids | `...::test_class_compat_by_name` | two items with equal class ids and different names treated as compatible |
| Determinism | `...::test_edges_sorted_and_permutation_stable` | shuffle input, compare outputs |
| View scorer writes only its fields | extend `tests/curation/test_crop_scores.py::test_no_scorer_writes_cluster_fields` to include `view_tag` | add `cluster_id` to the scorer's `writes` |
| View zero-shot math | `tests/test_view_tag.py` with hand-built 4-d vectors and text vectors: argmax, margin, calibration lookup | swap prompt order and see the label move |
| EXIF parse | `tests/test_exif_signals.py` with a tiny JPEG built in the test by Pillow `Image.save(exif=...)` (no committed photo): `DateTimeOriginal`, sub-second, offset, missing fields → `None`, malformed date → `None` and counted, never a raised ingest failure | a malformed date crashes decode |
| Burst rule | `tests/test_burst_ids.py`: two cameras interleaved, a gap exactly at the limit, equal timestamps, sort stability | bursts spanning cameras |
| Camera serial never stored raw | test that the images-doc write body never contains the fixture serial string (canary) | write the raw serial |
| One vote per group | loop-simulator unit test: a group of 40 duplicates counts once in the gate sample size | count 40 |

### 8.4 Empirical proof (research, after the tag)

- Each sub-experiment's runner sits in `scripts/research/cluster_diag/` (extend `run.py`, or add
  `attribute_run.py`). It writes raw output to `artifacts_local/` and the small backing JSON to
  `docs/benchmarks/attribute_signals/` with `SHA256SUMS`.
- Reported metrics:
  - BCubed P/R/F, pair P/R, AMI;
  - `attribute_pair_scores` (`scripts/bench/cluster_eval_metrics.py:246`) by view stratum;
  - false merges blocked against true merges lost for every veto arm;
  - U1 and U2 from the loop simulator;
  - permutation ARI (must be 1.000).
- Spread: three seeds and one row permutation per headline row. A gain counts only if it exceeds
  twice the run-to-run spread and its 1,000-resample bootstrap interval excludes 0
  (`scipy.stats.bootstrap`).
- Visual check: the method that passes is checked on contact sheets by the judge (Haiku on public
  non-face, non-plate images, at most 20 cluster sheets and 10 pair sheets per sub-experiment, as
  owner decision O3 requires). A blind human then re-judges a seeded 5 %. Judge verdicts sit
  beside the ground-truth metrics, never instead of them.
- E-8a census reproduce command (the one used for section 2.4; read-only):

```bash
.venv/bin/python -I scripts/research/exif_census.py <images_dir>   # proposed; counts only
```

### 8.5 Product proof (only for a signal that passed)

- Red/green tests as 8.3; `make contracts` and `make contracts-check`; the docs check
  (`scripts/docs/check_docs_vs_code.py`).
- One live check on the dev stack under its own compose project name (`docker compose -p <name>`):
  ingest 200 COCO images, then run `item_scores` with `view_tag`. Verify the field count equals
  the crop count, and that a second run writes identical values (idempotent).
- The Cropwright evidence display is checked in a browser at desktop width and at 430 px.

---

## 9. Judgment calls

| J | Question | Recommendation | Trade-off |
|---|---|---|---|
| J1 | Must E-8b and E-8c wait for E-7 (RQ-5 machinery), as the research plan orders, or may they run on the harness's existing arms (IVF, graph HAC) with contraction and a veto check as pre- and post-processing? | **Run on the harness** once #416 and #398 have verdicts. Contraction and a veto check are method-agnostic, and waiting couples a cheap question to the most expensive line | A positive result has no product consumer until N-5; the research outcome is recorded either way |
| J2 | One vote per near-duplicate group in the auto-label loop's gate statistics: treat it as part of #400, or file it as a separate loop correctness item? | **Separate item** under #398 or the loop tracker, if the simulator shows a U1 or U2 change. It is a statistics bias, not a signal | One more issue; but it can ship without any of #400 |
| J3 | May a public set outside COCO and Open Images that carries EXIF or identity labels back a published claim? Candidates: CO3D v2 (multi-view identity, CC BY-NC 4.0), and a Flickr-sourced set with metadata (licence to verify) | **No for the EXIF arms** (owner-run on the private set, numbers only). **Local-only CO3D** for the identity-level colour veto, aggregates published only with the owner's approval | Without it, identity-level arms have no public evidence at all |
| J4 | View ground truth: person-labelled Open Images sample (300 crops, about 1 h of owner or delegate time) or CompCars only? | **Person-labelled Open Images** for the published number; CompCars as a local cross-check | Person time; CompCars is licence-restricted |
| J5 | Store EXIF GPS in curation projects (makemodel O10 said yes, project-scoped, never served or exported by default)? | **Not in #400.** Store it only when #348 is un-parked and needs it; until then extract nothing. Data minimisation: a field nobody reads is a liability | #348 would need a header-only backfill later (cheap) |
| J6 | Camera body serial: keyed hash with the per-deployment secret introduced by #401, or omit it and use `Model` plus `Make` only? | **Keyed hash**, reusing #401's secret mechanism (one secret pattern in the repo, not two). If #401 never lands, use `Make` plus `Model` only, and accept that two bodies of one model share a burst scope | Couples #400's burst arm to #401's secret design |
| J7 | Pixel colour (Pillow quantize plus Lab names) if the VLM pack colour flips too often? | **Only after E-8d shows the pack colour fails and the pixel version is measured to flip less.** Never both by default | More ingest CPU per crop |

---

## 10. File-by-file change list

Nothing below is written until its sub-experiment passes, except the research and docs items
marked R.

### Research and docs (R; after the tag, CPU slots)

| File | Change |
|---|---|
| `docs/research/attribute_signals_eval.md` (new) | R0 registration (bars, samples, manifests), then results; Setup, Method, Results, Findings, Open issues, Reproduce (per `docs/research/CLAUDE.md`) |
| `scripts/research/exif_census.py` (new) | read-only EXIF counts per directory (Pillow `getexif`, `get_ifd`); prints counts only, never values |
| `scripts/research/cluster_diag/attribute_run.py` (new) | E-8b to E-8d arms: contraction, veto check, view assigner; reuses `partitions.py`, `loop_sim.py`, `heldout.py` and `scripts/bench/cluster_eval_metrics.py` |
| `scripts/research/view_tag.py` (new) | zero-shot view labels from stored vectors and PE text vectors; isotonic calibration on dev |
| `scripts/datasets/manifests/` | a view-label manifest (content hashes plus labels) stays in `artifacts_local/`; only a counts manifest is committed |
| `docs/benchmarks/attribute_signals/` (new) | backing JSON plus `SHA256SUMS`; never pruned |
| `docs/design/research_clustering_plan.md` | RQ-6 row: link this plan's sub-experiments; E-8 split into a to d (docs commit) |

### Product (only for passing signals; after B2)

| Subsystem | File | Change |
|---|---|---|
| Domain | `src/domain/attribute_edges.py` (new) | `SignalSpec`, `AttributeEdge`, `build_edges`; pure |
| Domain | `src/domain/bursts.py` (new, EXIF arm only) | burst assignment by (camera key, time); pure |
| Mapping | `src/clients/curation_opensearch/signals_fields.py` (new) | `SIGNALS_ITEM_MAPPING`, `SIGNALS_IMAGE_MAPPING`, `ensure_signals_fields` |
| Mapping | `src/clients/curation_opensearch/bodies_core.py` | spread the new mappings into the items and images bodies, as `ASSIST_ITEM_MAPPING` is |
| Ingest | `src/services/curation/image_decode.py` | read EXIF time, offset, sub-second and camera fields from the header already opened at line 80; return them on `DecodedImage` (new optional field); no new open |
| Scores | `src/services/curation/item_scores/view_tag.py` (new) | `CropScorer` with `writes = ('signals.view', 'signals.view_margin', 'signals.view_calibration')`; register in `item_scores/__init__.py` |
| Jobs | the B2 task module for `item_scores` | no change beyond the scorer registry |
| Jobs | new `op.jobs.signals_backfill` (EXIF arm only) | header-only reads for existing images; lease `signals_backfill` per project; B2 `JobTask` wrapper |
| Config | region profile or prompt pack | `view_prompts` list (project data); validation in `src/services/config_store/profile_field_checks.py` or the pack validator |
| Wire | `src/services/curation/wire.py`, `src/routers/curation/_item_models.py` | serialise `signals.*` (never GPS unless opted in); `make contracts` |
| Docs | `docs/design/curation_api_contract.md`, `docs/CURATION.md`, docs-site guide | new fields and what they are for |
| Tests | section 8.3 files | red/green |
| Frontend | Cropwright suggestions and evidence display | render backend-provided evidence only |
| Changelog | `CHANGELOG.md` `[Unreleased]` | one entry per landed signal |

No env var is added for the view tag or the burst gap: they are project data. If an `OP_*` variable
is needed later, it goes into `env.template` and `tests/test_env_surface.py` in the same commit.

---

## 11. Edge cases and traps (the failure register)

1. **A must-link test that passes because the floor is never reached.** If the fixture's
   embeddings all sit above the floor, "needs embedding floor" is untested. The fixture must
   include a pair with an equal signal **below** the floor, and the test asserts no edge.
2. **Contracting near-duplicates with the unvalidated 0.98 threshold.** It looks like a clean gain
   on a set with many jittered copies. It also merges distinct objects in one frame (the
   `near_dup.py` docstring warns about this). Gate E-8b on #416.
3. **Duplicates that count as votes.** A calibration that counts every member of a 40-frame burst
   tunes the cut to that burst. Contract first, then calibrate.
4. **View tag "accuracy" on a set whose labels came from the same text prompts.** Any VLM- or
   CLIP-derived view label makes the accuracy circular. Ground truth is a person or the dataset
   authors.
5. **View tag used as a veto by accident.** A config that lists `view` with role `veto` must be
   refused by `SignalSpec` validation, not just discouraged in docs.
6. **Colour from `assist.attributes[]` treated as a label.** `assist` is a suggestion store that
   "no export, training or holdout path reads" (`assist_fields.py:1`). An edge builder reading it
   must not change that: edges are evidence, never labels, and never written to `class_*`.
7. **EXIF time without an offset.** `DateTimeOriginal` has no zone. Bursts compare times within one
   camera only, so the zone does not matter there. Any cross-camera comparison is wrong without
   `OffsetTimeOriginal` and must not be made.
8. **Camera clock resets and equal timestamps.** Sort by (camera key, time, image id) so equal
   times are stable. A clock that jumps backwards splits a burst, which is the safe failure.
9. **Images edited by software.** `DateTime` (IFD0) is the edit time. Only `DateTimeOriginal` is
   capture time. The census and the extractor read the Exif IFD, not IFD0.
10. **EXIF read failures that fail the ingest.** Malformed EXIF raises in Pillow; `crop_bytes.py`
    already lists `ValueError`, `TypeError` and `SyntaxError` on malformed EXIF. The signal read
    must catch exactly those, record `signals.exif_error`, and never fail the image (the
    orientation read today already has to survive them). This is not a broad `except`: the broad
    excepts ratchet (`scripts/codegen/`) must not grow.
11. **A second decode for colour.** The pixel colour arm must use the frame ingest already
    decoded. A crop re-read in a backfill is a separate, measured cost, never hidden.
12. **Edge lists that differ between runs.** A `set` iteration order in `build_edges` breaks
    permutation ARI. Sort edges, and test with a shuffled input.
13. **Class compatibility by id.** Two projects can map the same id to different names after a
    combine. Compare names (casefolded); the test uses equal ids with different names.
14. **A signal with no coverage that "does no harm".** A view tag that abstains on 95 % of crops
    passes every no-loss stratum bar and helps nothing. Report coverage beside every gain, and
    register a coverage floor (at least 50 % of the stratum) for a signal to count as passing.
15. **A held-out half that wears out.** E-8 reads the fresh set's test half once per signal
    family. A second read retires it for acceptance (`research_clustering_plan.md` 4.2).
16. **GPS sneaking into the wire.** A generic serialiser loop over `signals.*` would serve GPS.
    `wire.py` lists fields explicitly. A test asserts `signals.gps` is absent from every
    serialised item and image unless the project opted in.
17. **The scorer registry test that passes on an empty registry.** Extending
    `test_no_scorer_writes_cluster_fields` must assert that `view_tag` is in the registry it
    iterates. Otherwise an unregistered scorer is untested and the test stays green.
18. **Private-set numbers in a public file.** EXIF arms produce owner-run numbers. Commit
    aggregates only (counts, rates), never per-image rows, paths or names. The naming scan
    (`scripts/codegen/check_naming_leaks.py`) does not catch numbers that identify images.

---

## 12. Suggested phasing

| Phase | When | What | Exit |
|---|---|---|---|
| P0 | any time (docs only) | R0 registration in `docs/research/attribute_signals_eval.md`; the census script | bars committed before any score |
| P1 | after the v0.7.0 tag, CPU slot | E-8a public feasibility: view-tag accuracy on the person-labelled sample; pack-colour flip rate after #416 | per-signal go/no-go |
| P2 | after #416 and #398 verdicts (and J1) | E-8b near-dup contraction; E-8c view assigner; E-8d colour (stratum public, veto local per J3) | PASS/FAIL per sub-experiment; outcome note updated |
| P3 | owner's choice | EXIF and burst arms on the private set (owner-run) | aggregate numbers; go/no-go for the EXIF extractor |
| P4 | only for passing signals, after B2 and after N-5 is approved | product landing per section 10, one feature PR, targeted tests in lanes, one full sweep for the wave | `make contracts-check` clean; browser check; CHANGELOG |

If P1 and P2 both fail, #400 is closed or stays parked with an evidence comment. No product code
lands, and the outcome note records what was done and what was not.

---

## 13. Sources

- Pillow `ExifTags` and `Image.Exif.get_ifd`: https://pillow.readthedocs.io/en/stable/reference/ExifTags.html,
  https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Exif
- Pillow `Image.quantize`: https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.quantize
- CLIP zero-shot classification: Radford et al. 2021, https://arxiv.org/abs/2103.00020
- scikit-learn `AgglomerativeClustering` (`connectivity`) and `IsotonicRegression`:
  https://scikit-learn.org/stable/modules/clustering.html, https://scikit-learn.org/stable/modules/isotonic.html
- leidenalg fixed membership: https://leidenalg.readthedocs.io/en/stable/reference.html
- statsmodels Wilson intervals: https://www.statsmodels.org/stable/generated/statsmodels.stats.proportion.proportion_confint.html
- SciPy bootstrap: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html
- BCubed: Amigo et al. 2009, https://doi.org/10.1007/s10791-008-9066-8
- asyncio TaskGroup: https://docs.python.org/3/library/asyncio-task.html#task-groups
- EXIF 2.32 tag definitions (DateTimeOriginal, SubSecTimeOriginal, OffsetTimeOriginal,
  BodySerialNumber): CIPA DC-008, https://www.cipa.jp/std/documents/e/DC-008-Translation-2019-E.pdf
- CO3D v2 (if J3 allows, local only): https://github.com/facebookresearch/co3d
- Repo: `docs/design/research_clustering_plan.md` (RQ-6, E-8, sections 3, 4, 6, 7),
  `docs/design/cluster_search_makemodel_plan.md` 6.2 and 6.7,
  `docs/research/cross_domain_clustering_lessons.md` (T11, 6.2, E8),
  `docs/research/cluster_program_outcome.md` section 1.
