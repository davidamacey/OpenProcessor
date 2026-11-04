# OpenProcessor: spatial scene discovery and photogrammetry, feasibility study and staged plan (issue #348)

**Status: PLAN ONLY, NOT IMPLEMENTED. The issue is parked.** No experiment has run and no product
code exists. This gist is a feasibility study and a staged plan. It covers:

- the libraries to evaluate, each with its documentation and licence checked on 2026-11-04;
- the public multi-view data already catalogued;
- the prerequisites in the OpenProcessor stack;
- a go/no-go bar that is registered before any data is scored;
- a revisit trigger.

Grounding:
- Branch `int/b1-correctness` at `81c4e4ef` (read-only worktree).
- The existing research note `docs/research/OpenProcessor_Spatial_Memory_Photogrammetry_Research_README.md`
  (the "spatial note") and `docs/research/spatial_datasets.md`, both on `main`.

Reader: a fresh agent with no memory of this session.

---

## 0. TL;DR

### What I recommend

1. **Keep #348 parked.** Do not start product work. The useful next step is the issue's
   "one-weekend experiment", narrowed and registered in advance. It splits into four measured
   stages, and each stage has its own pass line:
   - **S1a, place retrieval.** Do the vectors OpenProcessor already stores find same-place photos?
     Compare the curation frame vector (PE-Core-L, 1024-d) and the core MobileCLIP vector (512-d)
     against one maintained place-recognition descriptor (MegaLoc, MIT).
   - **S1b, pair verification.** Do the retrieved pairs share real geometry? Use COLMAP's own
     two-view geometry through `pycolmap`, with SIFT or ALIKED features and the LightGlue matcher
     built into COLMAP 4.x.
   - **S1c, scene grouping.** Do connected components of verified edges give pure scene groups with
     no false merges between look-alike places?
   - **S1d, reconstruction.** Does the best *discovered* group (not a hand-picked one) reconstruct
     with COLMAP's global mapper (the former GLOMAP) as well as the dataset's reference model?
2. **Use maintained libraries end to end. Write nothing custom except glue and metrics:**
   - `pycolmap` (BSD-3) for features, matching, verification and mapping;
   - MegaLoc through `torch.hub` (MIT) for place retrieval;
   - Kornia (Apache-2.0) only as a cross-check for the matcher;
   - FAISS (already in the stack) for kNN;
   - the repo's existing metric helpers for BCubed and recall@K.

   Splatting (gsplat), dense MVS (OpenMVS), viewers and a timeline stay out of scope until S1
   passes.
3. **Use public data only, already catalogued:**
   - IMC PhotoTourism for discovery and reconstruction, with a COLMAP reference model per scene;
   - Cambridge Landmarks for poses;
   - Google Landmarks v2 subset and COCO val2017 as distractors;
   - Aachen Day-Night and RobotCar Seasons as a reported, not gated, cross-time stress row.

   Several of these sets are non-commercial. Keep them local and publish aggregates only
   (`docs/research/spatial_datasets.md`).
4. **The compute is small.** About 3-4 GPU hours and about 6-10 CPU hours in one research window,
   plus about 2 hours of person checks. It fits as a rider in a clustering research window (J5).
   It needs **no** product change and **no** B2 job, so the issue's "after 0.7.0" gate can relax
   to "after the v0.7.0 tag, in a research window" (J1).
5. **Fix the stale trigger.** The v0.8.0 plan's revisit trigger for #348 waits on "place-recall
   results" from the clustering research plan. That plan has no such experiment (C5). S1a is that
   experiment. It belongs to #348, not to the clustering tracker.

### What I am NOT deciding

These are owner calls, listed in section 9:
- when S1 runs (J1);
- which retrieval arms are allowed (J2);
- which matcher stack (J3);
- where a product version would live (J4);
- the window slot (J5);
- the GPS and EXIF privacy design (J6);
- the dense, splat and viewer stages (J7);
- whether to publish numbers from non-commercial sets (J8).

---

## 1. Context: why the obvious approach is wrong here

There are two obvious approaches, and both are wrong.

**Wrong approach 1: run COLMAP on the whole library.**
- Matching a mixed library is all-pairs work, and most pairs are unrelated.
- Repeated structures (two similar cathedral façades) create false edges that merge separate places
  (spatial note sections 5 and 13.3).
- COLMAP 4.2's multi-component global mapper reconstructs disconnected parts separately (COLMAP
  4.2.0 release notes). It does not *discover* places in an unrelated library. Discovery has to
  come first and be bounded.

**Wrong approach 2: treat "same place" as clustering on the existing vectors.**
- The vectors in this repo were built for class-level grouping and semantic search, not place
  identity.
- The clustering research found that even fine-grained *class* grouping is limited by the encoder
  (`docs/research/cluster_program_outcome.md` section 1).
- Place identity is a stricter relation: same physical geometry, not the same kind of thing.
- The spatial note says so itself (section 9.4): "Start with geometry-consistent subgraphs and use
  CLIP/HDBSCAN only for candidate indexing".

So the right first question is measurable and cheap:
- Can the stored vectors *propose* same-place pairs with high recall?
- Can a maintained geometric verifier *confirm* them with near-zero false edges?

If either answer is no, no amount of 3D or viewer work rescues the feature. That is the go/no-go.

---

## 2. What is there today (verified)

### 2.1 Research material already in the repository

| Item | Where | State |
|---|---|---|
| Spatial research note (tool survey, architecture, phases, recommended experiment) | `docs/research/OpenProcessor_Spatial_Memory_Photogrammetry_Research_README.md` (commit `521f6cbd`, on `main`) | Literature and vendor README review only. Nothing benchmarked (its own "Evidence standard" paragraph, section 1) |
| Public multi-view dataset catalogue | `docs/research/spatial_datasets.md` (commit `242c20fa`, on `main`) | 10 datasets, 158 GB, integrity-checked on 2026-10-29 (67 archives, 0 failures after a refetch) |
| Resumable fetcher with archive verification | `scripts/datasets/fetch_spatial_sets.py` (`main()` at `:230`; `verify --root ... --test-archives`, `fetch --only <set>`) | Present. Tanks and Temples needs a manual form |
| Research index row | `docs/research/README.md`, "Long-term" | Says "milestone v0.9.0" (stale, C3) |

### 2.2 What the stack offers a spatial pipeline

| Capability | Today | Where |
|---|---|---|
| Whole-image vector, curation | `pe_embedding` (PE-Core-L14-336, 1024-d) on the project's images index | `src/clients/curation_opensearch/bodies_core.py:73-105`; preprocessing `src/utils/pe_preprocess.py:1-19` (shorter edge to 336, centre crop) |
| Whole-image vector, core | `global_embedding` (MobileCLIP, 512-d) on the core global index | `src/clients/opensearch/indexes.py:96-119` |
| Image identity hashes | `imohash`, `phash` | `bodies_core.py:91-92` |
| Near-duplicate grouping | Frame 0.98 (PE), core 0.99 (MobileCLIP) | `src/services/detection/frame_dedup.py:34`; `src/routers/ingest/single.py:78-83` |
| Person and vehicle boxes (to mask dynamic objects) | Detector boxes per item | the curation region stage (`region_boxes`) |
| EXIF orientation | Applied at decode | `src/utils/pe_preprocess.py:16-18`; `src/services/curation/image_decode.py:24` |
| **EXIF time, GPS, camera make, model, focal length** | **Not captured anywhere.** No capture-time or geo field on the images index (`bodies_core.py:73-96`); EXIF is read only for orientation (`rg -n -i "exif\|gps" src`) | Missing (prerequisite P1) |
| Durable jobs | Not yet. B2 (Celery, leases, beat, reaper) is planned, not merged (`docs/design/program/03_b2_overview.md`) | Missing (P2) |
| Optional GPU side-car pattern | Compose profiles (`segmenter`, `vlm`, `evaluator`, `training`) with GPU placement from `.env` | `docker-compose.yml` profile blocks; root `CLAUDE.md` "Services and compose" |
| Project isolation of new indexes | OpenSearch guard refuses any index the bound project does not own | `src/services/projects/guard.py`; index names `{OP_PROJECT_INDEX_PREFIX}{slug}__{role}` |
| Evaluation helpers | BCubed, pairwise, AMI, risk-coverage (`scripts/bench/cluster_eval_metrics.py:43`, `:139`, `:184`, `:205`); retrieval recall@k, MRR, nDCG (`scripts/bench/retrieval_eval_metrics.py`) | Reusable as is |
| Scorecard that refuses synthetic quality claims | `scripts/bench/scorecard.py:30`, `:87-94` | Reusable |
| Hosted-judge refusal for face, plate and identity-text sheets | `scripts/research/judge_protocol.py:23`, `:55-58` | Tourist photos contain faces. Visual checks for S1 go to the local VLM or a person |

### 2.3 Program position

- `VERSION` is 0.6.0. There is no v0.7.0 tag.
- #348 is outside the 0.7.0 program (`docs/design/program/00_index.md:356`). It has no milestone
  (`09_github_actions.md:635`).
- The v0.8.0 plan parks it with a trigger (`docs/design/v080_platform_breadth_plan.md` section 6).
- GPU time after the tag is already contended:
  - v0.8.0 windows V0, V1, V-EM and V2;
  - GPU-pipeline research GP-RW-1..3;
  - clustering research CRW-1..3 (see the #346 plan, section 6).

---

## 3. Libraries to evaluate (documentation and licences, checked 2026-11-04)

The licence column is from each repository's licence metadata (`gh api repos/<r>`) or its LICENSE
file. The owner's licensing rule applies: the whole repo is AGPL-3.0, and any third-party weights
or data with their own terms are named separately.

| Library | Role in S1 | Licence | Activity / version | Documented facts relied on | Verdict for S1 |
|---|---|---|---|---|---|
| [COLMAP](https://colmap.github.io/) + [pycolmap](https://colmap.github.io/pycolmap/index.html) | Features, matching, two-view verification, global and incremental mapping | New BSD for COLMAP itself; dependencies are separately licensed (LICENSE file) | 4.2.1, 2026-09-29; repo active | `pip install pycolmap` (CPU wheels); `pycolmap-cuda12` for GPU, Linux only. `extract_features`, `match_exhaustive`, `match_vocabtree`, `match_spatial`, `match_image_pairs`, `incremental_mapping`, `global_mapping`. `FeatureExtractorType` includes `SIFT`, `ALIKED_N16ROT`, `ALIKED_N32`, `LOMA_B`; `FeatureMatcherType` includes `SIFT_LIGHTGLUE`, `ALIKED_LIGHTGLUE`. 4.2.0: the global mapper reconstructs every connected component by default (`GlobalMapper.multiple_models`) | **Primary** for S1b, S1c and S1d |
| [GLOMAP](https://github.com/colmap/glomap) | Global SfM | BSD-3 | **Archived.** README: "fully migrated to COLMAP, where GLOMAP functionality is exposed as the 'global' mapper" | Use `pycolmap.global_mapping`, never the archived repo | Evaluated only as COLMAP's global mapper (S1d arm) |
| [MegaLoc](https://github.com/gmberton/MegaLoc) | Place-recognition descriptor | MIT | Pushed 2026-08 | `torch.hub.load("gmberton/MegaLoc", "get_trained_model")`; 8448-d L2-normalised output; README claims state of the art "on most VPR datasets" (vendor claim, to be measured) | **S1a challenger arm** (shippable licence) |
| [SALAD](https://github.com/serizba/salad) | Place-recognition descriptor (DINOv2 backbone) | Code GPL-3.0; **pretrained weights have no stated licence** (Google Drive links) | Pushed 2025-10 | Descriptor sizes 8192+256, 2048+64, 512+32 | Measurement-only arm, if J2 allows; never shippable without a weights licence |
| [LightGlue](https://github.com/cvg/LightGlue) | Learned matcher | Apache-2.0 for code and weights; **SuperPoint follows a restrictive licence for its weights and inference file** | Pushed 2026-10 | Weights released for SuperPoint, DISK, ALIKED and SIFT | Used **through COLMAP's built-in matcher** with SIFT or ALIKED. SuperPoint is excluded |
| [Kornia](https://kornia.readthedocs.io/en/latest/feature.html) | Cross-check: ALIKED, DISK, LightGlue, LoFTR; RANSAC in `kornia.geometry.ransac` | Apache-2.0 | v0.9.0, 2026-10-23; active | `kornia.feature` lists ALIKED, DISK, XFeat, DeDoDe, LightGlue, LoFTR. The page does not state weight licences, so each pretrained model's terms must be checked before use | Optional cross-check arm (J3) |
| [hloc](https://github.com/cvg/Hierarchical-Localization) | Retrieval, matching and localisation orchestration on pycolmap | Apache-2.0 | Latest tagged release v1.4 (2023); repo pushed 2025-12 | Retrieval: NetVLAD, AP-GeM, OpenIBL, MegaLoc. Matchers: LightGlue, SuperGlue, NN. "pycolmap is enough" since v1.3 | Not needed for S1 (pycolmap plus MegaLoc covers it). Reconsider for **old-photo registration** against a reference scene (stage S3) |
| [OpenMVS](https://github.com/cdcseacave/openMVS) | Dense MVS, meshing, texturing | AGPL-3.0 | v2.4.0, 2026-01-20; active | Input is camera poses plus a sparse cloud | Out of S1. Stage S2 candidate. Licence-compatible with an AGPL repo, but bundling and modification obligations need the owner's review (J7) |
| [gsplat](https://github.com/nerfstudio-project/gsplat) | Gaussian splat training | Apache-2.0 | Pushed 2026-09 | CUDA rasteriser and training library, not a turnkey app (spatial note section 6) | Out of S1. Stage S2 candidate (J7) |
| [VGGT](https://github.com/facebookresearch/vggt) | Feed-forward poses and depth | Repo licence not asserted in metadata. The spatial note records the original checkpoint as **non-commercial**, plus a separate commercial checkpoint with use exclusions | Pushed 2026-05 | | Measurement-only, stage S2+, if ever |
| [GLUEMAP](https://github.com/colmap/gluemap) | Research SfM for low-overlap and repeated structure | Repo licence not asserted in metadata. The spatial note says BSD, with optional checkpoints under their own terms | Pushed 2026-06 | Multi-model dependency tree (spatial note section 5) | Out of S1. Try only if S1d fails on low-overlap groups |
| [FAISS](https://github.com/facebookresearch/faiss) | Exact kNN for retrieval candidates | MIT | already a product dependency | `IndexFlatIP` exact search at these sizes (tens of thousands of vectors) | Used |

Why this split:
- **pycolmap as the single geometry dependency** keeps verification identical to what
  reconstruction will use. A pair that passes S1b is a pair COLMAP will also accept in S1d. A
  second matcher stack would add a parity problem without a measured need.
- **MegaLoc over SALAD as the challenger.** It has a clear MIT licence and a documented one-line
  load. SALAD's weights carry no stated licence.

❌ **Rejected: hand-rolled RANSAC, essential-matrix or scene-graph code.**
- COLMAP's two-view geometry estimation already provides it, and so does Kornia's RANSAC.
- Connected components over verified edges come from SciPy's `scipy.sparse.csgraph.connected_components`.
  The repo already uses it at `src/services/face_identity/person_grouping.py:34`, `:61`.

❌ **Rejected: running COLMAP's exhaustive matcher on the whole library.** It is the "obvious
approach" of section 1. S1 bounds pairs by retrieval top-K.

---

## 4. Public multi-view data and the S1 evaluation set

From `docs/research/spatial_datasets.md`. The images are local only and never committed.

| Role in S1 | Dataset | Count | Ground truth used | Licence note |
|---|---|---|---|---|
| Discovery and reconstruction (main) | IMC PhotoTourism, 7 test scenes (british_museum 660, florence_cathedral_side 108, milan_cathedral 124, mount_rushmore 138, piazza_san_marco 249, sagrada_familia 401, st_pauls_cathedral 615) | 2,295 | Scene id; COLMAP sparse model per scene (co-visibility of image pairs; reference poses) | Flickr photos under individual licences, research use; do not redistribute |
| Calibration (dev) | IMC PhotoTourism, 4 train scenes (brandenburg_gate 1,363, reichstag 75, sacre_coeur 1,179, temple_nara_japan 904) | 3,521 | As above | As above |
| Outdoor poses, second domain | Cambridge Landmarks, 5 scenes | 7,295 | Scene id; per-image pose | No licence shipped; research use, cite Kendall et al. 2015 |
| Hard distractors | Google Landmarks v2 subset (6 shards) | 49,596 | Landmark id | CC BY or public domain per image; attribution CSV |
| Non-place negatives | COCO val2017 | 5,000 | none | Annotations CC BY 4.0; Flickr images |
| Cross-time stress (reported, not gated) | Aachen Day-Night v1.1; RobotCar Seasons v2 | 7,890; 32,796 | Database reference model; train poses | CC BY-NC-SA 4.0, non-commercial academic |
| Indoor sanity (optional) | 7-Scenes | 43,000 frames | Per-frame pose | Microsoft non-commercial research licence |

### 4.1 Ground truth for S1 (registered in R0)

- **Same-place pair:** two images of one IMC or Cambridge scene.
- **Co-visible pair:** an IMC pair that shares at least `T_cov` 3D points in the scene's reference
  COLMAP model. Read it with `pycolmap.Reconstruction(path)`: `points3D` hold tracks over image ids.
  `T_cov` is registered in R0, with a suggested value of 100.
  - Same-place but not co-visible pairs are reported separately. They are the "same place, no
    shared geometry" case that verification should reject, and must not be counted as a
    verification false negative.
- **Look-alike negatives:** cross-scene pairs among the test scenes (for example
  florence_cathedral_side against milan_cathedral against st_pauls_cathedral). Also the retrieval
  top-K neighbours drawn from GLDv2 and COCO.
- **Contamination rule:** GLDv2 may contain the IMC landmarks. Before scoring, drop GLDv2 images
  whose landmark category matches an IMC or Cambridge scene. Use the GLDv2 label-to-category
  metadata (Wikimedia category per landmark id). Then visually check a seeded 50-image sample of
  the top retrieved GLDv2 neighbours per scene, by a person. Record the count dropped.
- **Split:** the IMC train scenes are dev (thresholds, `K`, matcher options). The IMC test scenes
  plus Cambridge are test, read once.

### 4.2 Pin

Add a new manifest `scripts/datasets/manifests/spatial_s1.json`, in the same format as the other
pins (counts, licence per set, `manifest_sha256`). It holds file names and hashes only, never
images. It is built by a new `fetch_spatial_sets.py pin` subcommand over the already-fetched
tree.

---

## 5. Staged plan

### 5.1 S0: desk review (this gist; done except the items marked)

- Licences and versions: section 3. The open owner items are J2 (SALAD), J7 (OpenMVS) and J8
  (non-commercial numbers).
- Datasets: section 4, already fetched and verified on 2026-10-29. **Re-run
  `fetch_spatial_sets.py verify --test-archives` before S1.** Truncated IMC archives were found and
  resumed in the last pass.
- Prerequisites: section 6.

### 5.2 S1: the registered experiment (research scripts only)

| Stage | What | Arms | Compute (A6000, quiet host; estimates) | Metric |
|---|---|---|---|---|
| S1a retrieval | Encode every S1 image; exact kNN top-K (K in {10, 20, 50}) per image within the pooled test set plus distractors | (1) PE-Core-L frame vector, computed with the product preprocessing by calling `PEEncoder` on whole frames; (2) MobileCLIP core vector; (3) MegaLoc; (4) SALAD only if J2 | GPU about 2-3 h for about 64,000 images over 3-4 encoders; CPU minutes for kNN | Recall@K of co-visible pairs; recall@K of same-place pairs; precision of same-place among top-K; per-image encode cost |
| S1b verification | For the top-K pairs of the best S1a arm (plus the PE arm, if a different arm wins): extract features and verify two-view geometry | (1) pycolmap SIFT + `SIFT_LIGHTGLUE`; (2) pycolmap `ALIKED_N16ROT` + `ALIKED_LIGHTGLUE`; (3) pycolmap SIFT + NN ratio (classical baseline); (4) Kornia ALIKED + LightGlue + `kornia.geometry.ransac` only if J3 | about 46,000 to 115,000 pairs; GPU about 1 h with `pycolmap-cuda12`; CPU fallback recorded | Edge precision on look-alike and cross-scene pairs; edge recall on co-visible pairs; inlier count distribution; per-pair cost |
| S1c grouping | Graph of verified edges (inlier threshold calibrated on dev); connected components; also a 2-core variant to cut single-edge bridges (documented `networkx.k_core` or igraph) | components; 2-core components | CPU minutes | BCubed precision and recall against scene id; **cross-scene merges (count)**; fragmentation (components per scene) |
| S1d reconstruction | The largest discovered component per test scene into COLMAP | `pycolmap.global_mapping` (former GLOMAP); `pycolmap.incremental_mapping` | CPU about 2-6 h in all; GPU for features only | Registered-image share against the reference model; relative pose accuracy as mAA at the IMC thresholds (Jin et al., "Image Matching across Wide Baselines", IJCV 2021, arXiv:2003.01587); wall time; peak RSS |
| S1e cross-time (reported) | S1a and S1b on Aachen day against night, and RobotCar seasons | best arms only | GPU about 0.5 h | Recall@K and edge recall per condition. Not gated |

Rules for every stage:
- Visual check. A person (or the local VLM as a ranking aid) reviews a seeded sample of:
  - 50 verified edges with their match visualisations (pycolmap match drawing or OpenCV
    `drawMatches`);
  - 20 rejected co-visible pairs;
  - every cross-scene merge.

  These are tourist photos with faces, so they are **never sent to a hosted model**
  (`judge_protocol.py:23` covers face sheets; S1 adds the provenance rule from the #346 plan,
  section 4.7).
- One decode per image per pass: each encoder arm decodes once, and the feature extractor reads
  the file again. That is acceptable in research. Product implications are in P5.
- Spread: three seeds where any sampling happens (distractor sampling, visual-check sample).
  Exact kNN and COLMAP with a fixed seed otherwise. Repeat S1d once to measure run-to-run spread.
- Record per run: git sha, pin hash, library versions (`pycolmap.__version__`, the MegaLoc hub
  commit), GPU id, `nvidia-smi`, host load at start and end, wall time, peak RSS and VRAM.

### 5.3 Go/no-go bar (registered in R0 before any score)

All bars are read on the **test** scenes (IMC test plus Cambridge), once.

| ID | Bar | Why this level |
|---|---|---|
| G-1 retrieval | The best arm reaches recall@20 >= 0.90 on co-visible pairs with distractors present | Verification cannot recover pairs that retrieval never proposed. Below 0.9, discovery leaves large holes |
| G-2 reuse rule | If PE-Core-L (already stored) is within 0.05 recall@20 of the best arm, the product plan reuses PE and adds **no new encoder** | A new encoder costs ingest time, storage and a re-index on every project. It must earn that |
| G-3 verification precision | 0 verified edges between different scenes on the test set (look-alike cathedrals included), and edge precision >= 0.99 on all verified edges | A single false geometric edge merges two places. That is the failure the issue itself names ("without false geometry merges") |
| G-4 verification recall | Edge recall >= 0.80 on co-visible pairs among retrieved pairs | Enough edges to keep a scene connected |
| G-5 grouping | BCubed precision >= 0.98 and BCubed recall >= 0.80 against scene id, with 0 cross-scene merges | Scene groups a user would accept without manual splitting |
| G-6 reconstruction | On the discovered components: registered-image share >= 0.90 of the reference model's registered images, for at least 5 of the 7 IMC test scenes. mAA within 0.05 of the same mapper run on the reference image list | The discovered group must be as good as a curated one. That is the point of discovery |
| G-7 cost | Recorded, not gated: per-image descriptor cost, per-pair verification cost, mapper wall time per 100 images | Sizes a later job design |

Verdicts:
- **GO** (G-1, G-3, G-4, G-5, G-6 pass). Write a phase 0-1 product design doc in `docs/design/`.
  It covers P1-P6 in section 6, with the job shape and the privacy design. Only then are product
  issues filed. Stages S2 and S3 stay separate decisions.
- **NO-GO on discovery** (G-1, G-3, G-4 or G-5 fails). Record it, keep #348 parked, and name the
  failing stage. Try the alternatives of section 3 (GLUEMAP, hloc) only if the failure is specific
  to them, for example low-overlap scenes.
- **NO-GO on reconstruction only** (G-6 fails, discovery passes). Discovery alone ("we found these
  N photos of the same place") is still a product candidate (spatial note "MVP A"). Ask the owner
  (J7).
- **INCONCLUSIVE** (a bar unreadable, for example too few co-visible pairs after the GLDv2
  contamination filter). Report it as such. Never round it to a pass.

### 5.4 Stages after a GO (not planned in detail here)

| Stage | Contents | Gate |
|---|---|---|
| S2 dense and splat | OpenMVS or COLMAP dense on one discovered scene; gsplat training; held-out PSNR, SSIM and LPIPS; disk size | GO plus J7 yes; its own registration |
| S3 cross-time registration | Register old or cross-season photos to a reference scene (hloc localisation on pycolmap); share localised; anchor drift | GO plus an S1e signal |
| S4 product phase 0-1 | Design doc then issues: EXIF capture, descriptor field (if G-2 says new), pair-evidence store, scene groups as a job, review UI | GO plus the owner's yes, after B2 |

---

## 6. Prerequisites in the OpenProcessor stack (for a product, not for S1)

S1 needs none of these. A product version needs all of them. Each is phrased as a constraint on
code that is not written yet.

| ID | Prerequisite | Today | Constraint for the design |
|---|---|---|---|
| P1 | Capture EXIF time (with time zone and uncertainty), GPS (with accuracy), camera make and model, focal length | Not captured (section 2.2) | A new set of fields written once at ingest from the already-open image (Pillow `Image.getexif()` and the GPS IFD), with no second decode. GPS is personal location data, so the privacy design (J6) comes first: off by default per project, excluded from exports unless asked, removed by project delete. Never sent to a hosted model |
| P2 | Durable jobs | B2 planned (Celery, leases, beat, reaper) | Pair verification and mapping are heavy, so they run as Celery tasks on a GPU-capable queue under a lease per (project, scene), never inside a request handler. Activation of a scene version is an OCC write. No in-process or file locks: about 30 uvicorn workers plus worker containers share state |
| P3 | Optional GPU side-car | Compose profiles exist for other side-cars | A `spatial` profile, off by default, with its own image (multi-stage, `-slim` base, non-root user, Trivy before publish), GPU from `.env` (a new `SPATIAL_GPU_ID`), pinned library versions. Core and curation must work with it absent |
| P4 | Project isolation and storage | Guard and project-owned indexes exist | Any new index role (pair evidence, scenes) is project-owned and covered by the guard. Artifacts (COLMAP models) live in the project's state directory and are removed by project delete. Scenes never span projects |
| P5 | Decode once | Owner rule; B5 makes frames GPU-resident | Feature extraction runs at its own resolution, which may be the full frame, so it is a second read of the file. The design either reuses the resident decoded frame, or records a reasoned exception approved by the owner. It is never silent |
| P6 | Stable scene ids and corrections | Not present (the same open problem as #346 RQ-5) | Scene ids carry over by membership overlap. User corrections (split, merge, "not the same place") are anchors keyed to image ids and never lost on a re-run. Machine merges keep their own provenance |
| P7 | Dynamic-object masking | Detector person and vehicle boxes exist | Masking features inside person and vehicle boxes is an S2+ option, measured, not assumed |

---

## 7. Corrections to the issue's premises

| # | Premise | What the tree says | Action |
|---|---|---|---|
| C1 | "The research note (committed on branch `docs/spatial-memory-research`)" | The note is on `main` (`521f6cbd`), same path | Update the issue body |
| C2 | Scope: "Run the note's recommended one-weekend experiment on public or owned data" | The public datasets are already catalogued and fetched, with a verifying fetcher (`spatial_datasets.md`, `scripts/datasets/fetch_spatial_sets.py`). Only the pin and the scoring scripts are missing | Link both in the issue |
| C3 | `docs/research/README.md` lists the note under "milestone v0.9.0" | #348 has no milestone. v0.9.0 is to be closed once empty (`00_index.md` section 3) | Drop the milestone text in the next docs commit |
| C4 | Triage comment: "Reference to #346 will be stale once that epic is closed" | #346 stays open as the standing clustering research tracker (see the #346 plan, J11) | The reference stays valid |
| C5 | v0.8.0 plan section 6 trigger: "the clustering research plan reports its place-recall results" | `docs/design/research_clustering_plan.md` has no place-recall experiment. Same-place grouping is geometric, not class or identity grouping | S1a is the place-recall experiment and belongs to #348. Rewrite the trigger (section 8) |
| C6 | Spatial note section 12: "SALAD or alternate VPR descriptors" as the phase-one default | SALAD's code is GPL-3.0 and its weights have no stated licence | Measurement-only arm (J2). MegaLoc (MIT) is the shippable challenger |
| C7 | Spatial note sections 5 and 12 list GLOMAP-era tooling and "COLMAP 4.2 LoMa" | GLOMAP is archived and lives on as COLMAP's global mapper. COLMAP 4.2.0 added LoMa and multi-component global mapping; 4.2.1 (2026-09-29) restored the legacy BA behaviour of the global mapper | Use `pycolmap.global_mapping`, pinned at 4.2.1 or later |
| C8 | Spatial note section 9.1: "Existing OpenProcessor MobileCLIP global" as the semantic vector | The curation images index stores a PE-Core-L frame vector (`bodies_core.py:105`). MobileCLIP is the core API's index. Both exist | Both are S1a arms |
| C9 | Issue: "Reuses the clustering evaluation approach from #341" | Partly holds. BCubed and pairwise helpers fit grouping (S1c). Retrieval recall@K (S1a) comes from `scripts/bench/retrieval_eval_metrics.py`. Edge metrics and the co-visibility ground truth are new | New metric code is small and unit-tested (section 10) |
| C10 | Spatial note section 15: "COLMAP: Permissive BSD-family" | Confirmed: new BSD, with the file's own caveat that dependencies are separately licensed. The CUDA wheel `pycolmap-cuda12` brings CUDA runtime terms | Record it in ATTRIBUTION only if a product image ever ships it |

---

## 8. Revisit trigger (replaces the v0.8.0 plan's wording)

Run S1 when **all** of these hold:
1. The v0.7.0 tag exists.
2. A research GPU window has about 4 free GPU hours on an agreed card (GPU 0 or GPU 2, never GPU 1
   without its owner's yes). The owner agrees to spend them on S1 (J5).
3. A CPU slot of about 10 hours exists between program sweeps.

Re-open product planning (S4) only when:
- S1 is GO;
- B2 is merged;
- the owner states a product need for scene discovery in a curation project (J4).

Otherwise #348 stays parked. A new major library generation (for example a COLMAP release that
changes the global mapper, or a VPR model with a clear licence beating MegaLoc in its own
benchmarks) is a reason to re-run S1, not to skip it.

---

## 9. Judgment calls (owner)

**J1. When S1 runs.**
- Recommendation: after the v0.7.0 tag, as a rider in a clustering research window (CRW-1 or
  CRW-3 of the #346 plan). The issue's "after 0.8.0 clustering work is scheduled" condition is
  met by scheduling, not by finishing.
- Trade-off: about 4 GPU hours and 2 person hours taken from other research.

**J2. Allowed retrieval arms.**
- Recommendation: PE, MobileCLIP and MegaLoc. SALAD as a measurement-only arm only if the owner
  accepts unlicensed weights for local research.
- Trade-off: SALAD is the descriptor the spatial note names. Without it, a MegaLoc loss cannot be
  read as "VPR does not help".

**J3. Matcher stack.**
- Recommendation: pycolmap only (SIFT or ALIKED with COLMAP's LightGlue matchers). Add the Kornia
  cross-check only if pycolmap pair throughput is under about 20 pairs/s on the A6000.
- Trade-off: one dependency and verification identical to reconstruction, against a second view
  of matcher quality.

**J4. Where a product version would live.**
- Recommendation: curation, project-scoped (images index, project isolation and guard, delete
  semantics), not the core API.
- Trade-off: core users with a plain photo library would need a curation project to use it.

**J5. Window slot.**
- Recommendation: CRW-1 rider, because it shares the "encode once, store vectors" step for PE
  frame vectors with E-12 of the #346 plan.
- Trade-off: lengthens CRW-1 by about 3 h.

**J6. GPS and EXIF privacy design.**
- Recommendation: required before any product capture (P1): off by default, never exported unless
  asked, purged with the project, never sent to a hosted model.
- Trade-off: none for S1 (public data). It blocks S4.

**J7. Dense, splat and viewer (S2), and AGPL OpenMVS.**
- Recommendation: decide only after a GO. Prefer COLMAP dense plus gsplat (BSD, Apache). OpenMVS
  only if mesh quality needs it, with its AGPL obligations reviewed.
- Trade-off: deferral against early visual impact.

**J8. Publishing numbers from non-commercial sets (Aachen, RobotCar, 7-Scenes, IMC Flickr photos).**
- Recommendation: aggregate tables only; no images or derived images in the repo; GO decided on
  IMC and Cambridge only.
- Trade-off: the cross-time row stays "reported".

---

## 10. File-by-file change list (for S1; research only)

| File | Change |
|---|---|
| `docs/research/spatial_s1_registration.md` (new) | Data, ground-truth rules (4.1, including `T_cov` and the contamination rule), arms, bars G-1 to G-7, seeds. Committed before any score |
| `scripts/datasets/fetch_spatial_sets.py` | `pin` subcommand writing `scripts/datasets/manifests/spatial_s1.json` (names, hashes, counts, licences) |
| `scripts/research/spatial/covis.py` (new) | Co-visible pairs from a reference COLMAP model through `pycolmap.Reconstruction` |
| `scripts/research/spatial/retrieval.py` (new) | Encoders: PE through `src/clients/pe_encoder.py` whole-frame encode on the research stack; MobileCLIP through the core embed route on a throwaway stack; MegaLoc through `torch.hub`. FAISS exact top-K; recall@K through `scripts/bench/retrieval_eval_metrics.py` |
| `scripts/research/spatial/verify.py` (new) | pycolmap database per run; `extract_features` and `match_image_pairs` on the top-K pairs; edge table with inlier counts |
| `scripts/research/spatial/group.py` (new) | Components and 2-core components (SciPy or igraph); BCubed through `scripts/bench/cluster_eval_metrics.py:139` |
| `scripts/research/spatial/reconstruct.py` (new) | `pycolmap.global_mapping` and `incremental_mapping` on the discovered components; registration share; mAA |
| `scripts/research/requirements-spatial.txt` (new) | `pycolmap-cuda12` (or `pycolmap`), `kornia` (if J3), pinned. Never in the product requirements |
| `tests/research_scripts/test_spatial_covis.py` (new) | Co-visibility on a tiny hand-built `pycolmap.Reconstruction` (3 images, known shared points). Red: an off-by-one in the shared-point threshold |
| `tests/research_scripts/test_spatial_metrics.py` (new) | Edge precision and recall, and cross-scene merge count, hand-computed. Red: a cross-scene edge counted as correct |
| `docs/research/spatial_s1_results.md` (new) | Setup, method, results, findings, open issues, reproduce (`docs/research/CLAUDE.md` structure) |
| `docs/benchmarks/spatial_s1/` (new) with `SHA256SUMS` | Small backing JSON for every published number; never pruned |
| `docs/research/README.md`, `docs/design/README.md` | Rows for the two new notes; drop "milestone v0.9.0" (C3) |
| `docs/design/v080_platform_breadth_plan.md` section 6 | Trigger text per section 8 (C5), in a docs commit on the active integration branch |

---

## 11. How we prove it: the verification plan

### 11.1 Static checks

1. Registration precedes results: `git log` shows `spatial_s1_registration.md` committed before
   `spatial_s1_results.md` and before `docs/benchmarks/spatial_s1/`.
2. Scorecards pass `scripts/bench/scorecard.py`, with real data and `synthetic: false`. The unit
   tests' tiny hand-built reconstructions are plumbing only and never appear in a results table.
3. The naming-leak scan and the docs check are green (`scripts/codegen/check_naming_leaks.py`,
   `scripts/docs/check_docs_vs_code.py`).
4. `git diff --stat` under `docs/benchmarks/` shows additions only.

### 11.2 Dynamic checks

1. `fetch_spatial_sets.py verify --test-archives` is clean on the day of the run, and the pin hash
   matches.
2. Ground-truth sanity: the reference COLMAP model's registered-image count per IMC test scene
   matches the image count the catalogue lists, within the known unregistered images. Count
   co-visible pairs per scene; a scene with zero co-visible pairs fails the setup.
3. Contamination: the GLDv2 drop count is recorded, and the person checks a 50-image sample per
   scene.
4. Stack isolation: any compose stack uses `-p op-rs-spatial`. The default project is never
   touched. The GPU is agreed with the peers, `uptime` is recorded, and one heavy run runs at a
   time.

### 11.3 Empirical checks

1. Each bar G-1 to G-6 is read once on test. Each verdict is PASS, FAIL or INCONCLUSIVE.
2. Visual review of edges and merges (section 5.2) by a person, with the local VLM ranking only.
   No hosted model.
3. S1d is repeated once; the spread is reported next to the mAA.

What red looks like:
- G-3: any cross-scene verified edge (shown with its match drawing in the note);
- G-1: a recall@20 table below 0.90 for every arm;
- G-6: fewer than 5 of 7 scenes reaching 0.90 registration.

---

## 12. Edge cases and traps

These are the ways S1 can look done and not be.

1. **Curated groups instead of discovered groups.** Running COLMAP on the reference image list
   proves nothing about discovery. G-6 is read only on components that S1c produced.
2. **Distractor contamination.** GLDv2 holds famous landmarks. Without the category filter, "false
   positives" are true matches and recall looks worse while precision looks better.
3. **Co-visible versus same-place.** Counting non-co-visible same-scene pairs as verification
   misses punishes a correct verifier. Report both relations.
4. **Repeated structure.** Façades and arcades give many inliers with wrong geometry. The inlier
   count alone is not evidence. Use COLMAP's two-view geometry configuration (degenerate or planar
   flags) and check the bridges in the visual review.
5. **Single bridge edges.** One wrong edge merges two scenes in connected components. Always report
   the 2-core variant, and count merges, not only BCubed.
6. **SuperPoint weights slip in.** LightGlue's SuperPoint path is under a restrictive licence. The
   research requirements must not pull it. Use COLMAP's SIFT or ALIKED matchers only.
7. **Unlicensed weights as a product default.** SALAD weights (and any Kornia pretrained model
   whose terms are unverified) stay measurement-only.
8. **Archived GLOMAP.** Installing the archived repo gives an old, unmaintained mapper. Use
   `pycolmap.global_mapping`.
9. **CPU wheel surprise.** `pip install pycolmap` is a CPU build. Feature extraction then runs on
   CPU and the timing row is wrong. Record which package ran (`pycolmap-cuda12` for GPU).
10. **Loaded host.** The 0.7.0 runs had load 9 to 79. Timing rows from a loaded host are
    within-table only.
11. **Faces in tourist photos.** Visual checks never go to a hosted model. Derived images from
    non-commercial sets are never committed.
12. **Encoder preprocessing drift.** The PE arm must use the product whole-frame preprocessing
    (`pe_preprocess.py`). A research-only resize gives a number that does not describe the stored
    vectors.
13. **EXIF-free public data.** Public sets have little or no GPS. S1 measures vision-only discovery,
    and that is a strength of the test. Do not claim GPS-assisted results from it.
14. **Scope creep into a viewer.** S1 ships no UI and no route. Any proposed route in the notes is
    written without a leading slash, with a note that it does not exist.

---

## 13. Suggested phasing

| Phase | Contents | Exit |
|---|---|---|
| P0 (now) | Owner answers J1-J8. Update the issue body (C1, C2) and the v0.8.0 trigger text (C5) in a docs commit | Issue and plan text current |
| P1 (CPU, any time) | Registration, pin subcommand, co-visibility and metric scripts with red/green tests | Registration committed |
| P2 (research window after the tag) | S1a to S1e runs; visual review | `spatial_s1_results.md` with G-1 to G-6 verdicts |
| P3 | GO: product design doc (P1-P7) and the owner's decision on S2, S3, S4. NO-GO: issue comment with the failing stage; stay parked | Decision recorded on #348 |

---

*Plan only, not implemented. Sources: the repository files cited inline at `81c4e4ef`; library
documentation and licence metadata as cited (COLMAP and pycolmap docs and releases, GLOMAP
README, MegaLoc, SALAD, LightGlue, Kornia, hloc, OpenMVS, gsplat), read 2026-11-04.*
