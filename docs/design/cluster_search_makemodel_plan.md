# Clustering and semantic search for faces, objects, images and items; fine-grained grouping (make/model) for set P

Status: plan, nothing implemented. Written 2026-10-29 against `origin/main` at `5620f6f6`.
Audience: a fresh implementation agent with no conversation memory. Every wave in section 9 is
self-contained: it names its inputs, exact files, tests, verification commands, GPU use, model
tier and the issues it closes. Symbols are named instead of line numbers; re-find them with `rg`.

Notation: `{P}` = `/curation/projects/{project}`. Routes that do not exist yet are written without
a leading slash and marked "proposed" (`scripts/docs/check_docs_vs_code.py` rejects a doc that
names a route the app does not serve). **Set P** is the owner's private collection of about 21 MP
photos (about 1.93M files); it is described by numbers only. Its paths, names, thumbnails and
per-image results stay in `artifacts_local/` (gitignored) and its source volume is read-only to
every tool in this plan. A private sibling sorting application was reviewed as prior art; it is
described generically and nothing from it is copied.

Read first: `CLAUDE.md`, `docs/VISION_AND_GOALS.md`, `docs/research/CLAUDE.md`,
[cluster_types.md](../research/cluster_types.md), [cluster_eval_baseline.md](../research/cluster_eval_baseline.md),
[clustering_methods.md](../research/clustering_methods.md), [auto_label_loop_plan.md](auto_label_loop_plan.md).
Unmerged inputs, read with `git show <branch>:<path>`:

| Input | Where | What it gives this plan |
|---|---|---|
| Leiden prototype results (#342) | branch `research/v080-342-leiden-prototype`, `docs/research/leiden_refinement_prototype.md`, scripts under `scripts/research/` | Mutual-kNN + Leiden CPM + CHAMELEON filter: conditional go; CPM resolution must be calibrated per space; HDBSCAN, FINCH, consensus and modularity Leiden ruled out |
| Cross-domain clustering lessons | branch `research/v080-cross-domain-clustering`, `docs/research/cross_domain_clustering_lessons.md` | Level (class vs identity), multi-prototype clusters, same-image cannot-link, abstention, calibration, experiments E0 to E9 |
| 3,000-item evaluation baseline (#341) | branch `fix/v070-341-work`, `docs/research/cluster_eval_baseline.md` | Production IVF on 2,950 labelled crops: pair precision 0.696, false merge 0.304 (0.400 low-res), false split 0.978, permutation ARI 0.306 (#372) |
| Auto-label loop (#286) | branch `feat/v070-assist-loop` (S0 to S6a committed) | Propagated machine label kind, exact sampled-agreement gate, splitter, budgets, rollback |
| Retrieval baseline | uncommitted in worktree `OpenProcessor-wt-ret` (`scripts/bench/retrieval_eval*.py`, `docs/benchmarks/retrieval_baseline/`) | Text and image search quality and latency on 2,000 COCO val images (section 1.2) |
| Program plan | branch `docs/program-plan`, `docs/design/program_plan.md` | Phase C (clustering) items C0 to C6, model tiers, DoD, measurement rules |
| Clustering research of 9 October | `docs/cluster_research_9oct.md`, untracked in the main checkout (not in git) | The refinement-engine design that #346 to #345 were filed from |
| **Consolidated clustering research** (primary method reference) | branch `docs/cluster-methods-review`, `docs/research/clustering_research_consolidated.md` | Real-data diagnosis (section 4), recommended stack per kind (section 7), experiments E0 to E10 on real data with a held-out split (section 8), missing evidence (section 9). Where this plan and that note differ, the note's real-data findings win |

**Real-data facts this plan is held to** (consolidated note sections 1 and 4): production pair
precision/recall is 0.696 / 0.022 on the 3,000-item set and 0.905 / 0.088 on the 539-crop COCO
project; any "precision about 0.95" figure is synthetic. Most fragmentation is configuration (IVF
K = n // 6 gives 6 to 8 crops per cluster and row-order dependence); one global average-linkage
AHC on raw vectors reaches 0.92 / 0.20 on COCO, matching the in-sample tuned Leiden + CHAMELEON
(0.944 / 0.195, which leaves 9.8 % unassigned). Above recall about 0.3 the neighbourhoods are the
limit (P@10 0.82; 0.78 under 32 px; 0.74 for crops longer than 2:1). The PE crop preprocessing
resizes then centre-crops (`src/utils/pe_preprocess.py`), so elongated crops lose their ends. The
space is anisotropic (mean random-pair cosine 0.50 COCO, 0.63 set P; effective rank 104 to 122),
so a fixed cosine threshold means different things per project: every threshold is calibrated per
project and space.

### Prior research this plan builds on (not redone)

The clustering method question is already researched. This plan does **not** re-survey it; each
part cites the section it builds on. New research was done only for what those notes do not
cover: Immich face clustering (section 3.1), fine-grained make/model recognition and its public
data (sections 5.2, 6), identity-level ground truth (5.4) and the visual-verification protocol
(section 7).

| This plan | Builds on |
|---|---|
| Mutual-kNN graph, Leiden CPM, CHAMELEON split/merge, no fixed cluster count (principle 3, 6.1 fine groups) | `cluster_research_9oct.md` sections 3A, 3B and 5; Leiden prototype sections 5 and 6 (conditional go, CPM calibrated per space, consensus/HDBSCAN/FINCH/modularity not built) |
| Versioned cluster records, `level`, lineage, provenance (principles 2, 7, 8) | `cluster_research_9oct.md` section 6; #343; cross-domain lessons 6.1, 6.2 |
| Viewpoint, colour and similar attributes as filters or tags, not taxonomy branches (6.2, 6.4) | `cluster_research_9oct.md` section 6 (last paragraph) |
| Multi-prototype clusters, abstention, same-image cannot-link, calibration, quality-gated evidence (principles 4 to 6, 3.2) | cross-domain lessons T1, T2, T4, T5, T8, T9 (from the sibling speech-transcription project's voice clustering) and experiments E3, E6 |
| Tiered refresh: kNN assignment on ingest, local re-evaluation, scheduled recompute, full audit; microclusters | `cluster_research_9oct.md` section 9; #344 |
| No embedding retraining first; an adapter only on stable clusters and trusted labels (6.5) | `cluster_research_9oct.md` section 8; #345 |
| Why IVF is the current default, why HDBSCAN and UMAP are not used for clustering, AHC limits at scale | [clustering_methods.md](../research/clustering_methods.md) sections 2, 3, 8 and 9 |
| GPU placement and the GPU clustering overlay | [gpu_clustering.md](../research/gpu_clustering.md) sections 2, 3 and 6; Leiden prototype section 8 (cuGraph Leiden is modularity only) |
| Current state of the three subsystems and what must not be mixed (section 1, principle 1) | [cluster_types.md](../research/cluster_types.md) sections 1 to 6 |
| Evaluation metrics and strata (section 8) | `cluster_research_9oct.md` section 10; #341 harness; cross-domain T12 and E0 |
| GPU arms (#345), fixed on 2026-10-29: the earlier failures came from the bench driver's address-space cap, not CUDA | synthetic 20k, 128-d, GPU 2 (scale-only evidence): cuGraph ECG precision 0.953 / recall 0.899 in 7 s and 1.8 GB; GPU kNN + CPU CPM Leiden 0.925 / 0.863; cuGraph Leiden (modularity only) over-merges, 0.776 / 0.910; cuML HDBSCAN unusable. The 100k, 500k, 1M and 1024-d arms were still running when this plan was written (raw output under `$SCRATCH/cluster-research-data/gpufix/`). **ECG is a candidate to re-test on real held-out crops and faces (W5), not a result** |
| Real COCO run of the prototype | 539 labelled crops: Leiden 0.905 / 0.234 at one setting, and the best CHAMELEON floors, were tuned on the same labels they were scored on; no held-out set exists yet, so W0 builds one before any method is chosen |

## 0. Owner goals and the constraints that shape the design

1. OpenProcessor must cluster **four kinds** well: faces (people), detected objects (crops and
   region boxes), whole images, and curation items. It must give fast semantic search over whole
   images **and** over crops/items. Labeling and self-supervised clustering group and label
   objects of interest.
2. **The core is domain-agnostic.** Detection, embedding, quality gating, graph clustering,
   prototypes and label propagation must work for any object class, faces or any other cluster
   kind. Vehicle make/model in set P is one worked example, not a special case in code.
   Everything domain-specific (class names, taxonomy, viewpoint prompts, quality thresholds) is
   project data, not code.
3. **Labels are never assumed perfect.** VLM and human labels feed a statistical gate, an audit
   and constraints; they are not ground truth by assertion.
4. **Local VLMs may run in bulk** (captions, labels, summaries over many images), because they run
   on our GPUs; the plan states where a bulk pass helps and what it costs (section 6.6).
5. **Haiku is a visual judge for testing, not a labeler.** It checks whether cluster members,
   search results or image pairs are really similar the way a person would judge them, on public
   images by default and on a set P sample only with the owner's explicit approval (section 7).
6. **One measurement standard** for every wave: the same metrics, acceptance bars, test sets and
   protocol per cluster kind (section 8).
7. **Real data decides.** Synthetic vectors are allowed only for speed and scale tests (memory
   bounds, latency at 1M, runtime ladders), never for a quality bar or a method choice. Every
   acceptance criterion is measured on real images: COCO and Open Images crops, the public
   fine-grained and face sets, and real set P photos (the already-ingested private projects and
   the NVMe sample copies; numbers only in committed text), cross-checked with the visual
   verification protocol (section 7). Unit tests may use tiny hand-built vectors to prove logic;
   they prove code paths, not quality.
8. GPU: **GPU 2 only** for this plan (GPUs 0 and 1 belong to other workloads for now). One GPU
   job at a time under `flock -w 3600 "$SCRATCH/locks/gpu2.lock" <cmd>`; every stack runs under
   its own `docker compose -p <name>` with its own ports.

## 1. Current state (code-derived, 2026-10-29)

### 1.1 The four cluster kinds

The repo has three clustering subsystems that share the field names `cluster_id` and
`cluster_distance` but no code ([cluster_types.md](../research/cluster_types.md)). Mapped onto the
owner's four kinds:

| Owner kind | What exists | Space | Algorithm, trigger | Gaps |
|---|---|---|---|---|
| **K1 whole images** | Core: FAISS IVF over `visual_search_global` (`src/services/visual_search/clustering.py`, `clusters.py`, `src/routers/clusters.py`), albums (clusters of at least 5). Curation: a whole-frame PE vector `pe_embedding` on each images doc (`src/clients/curation_opensearch/bodies_core.py` `_images_body`, written by `src/services/curation/ingest_index.py` and `reprocess_embed.py` part `frame`) | MobileCLIP2-S2 512 (core); PE-Core 1024 (curation frame) | k-means cells, manual `POST /clusters/train/{index}`; nothing clusters the curation frame vector | Cells are a partition, not groups; ids renumber on retrain; trained index not reloaded after restart and lost on recreate (#353); no project scoping in core; the curation frame vector is never clustered or searched |
| **K2 items** (curation item = one primary-detection crop, `crop_id`) | Residual pool: FAISS IVF K = min(512, n // 6), AHC refine on demand, ingest-time nearest-centroid assignment (`src/services/curation/clustering/orchestrator.py`, `methods/ivf.py`, `ivf_ingest.py`, `refine.py`, `residual_fetch.py`), refresh worker `scripts/curation/cluster_refresh_daemon.py` | PE-Core 1024 `pe_embedding` | Auto-label job and refresh worker | Fragmented and impure: on the 3,000-item set precision 0.696, false merge 0.304, false split 0.978; row-order dependent (#372, ARI 0.306); always assigns, never abstains; single centroid per cluster |
| **K3 detected objects / region crops** | Region-box clustering: MiniBatchKMeans k = max(8, round(n / 800)), AHC refine, false-positive sub-centroids (`src/services/curation/clustering/region_box_clustering.py`, `region_cluster_jobs.py`, `src/routers/curation/regions_fp.py`); core per-crop indexes `visual_search_vehicles`, `visual_search_people` (IVF only) | PE-Core 1024 box vectors in nested `region_box_embeddings`; MobileCLIP 512 (core crops) | Manual background job | On demand only; no persisted centroids, so no incremental assignment; L2 `cluster_distance` not comparable with K2's cosine; no evaluation set of its own |
| **K4 faces** | Core only, global index `visual_search_faces`: (a) FAISS IVF cells; (b) `POST /persons/cluster` union-find over a dense cosine matrix (`src/services/face_identity/person_management.py` `auto_cluster_faces`), names, merge, delete (`src/routers/persons.py`) | ArcFace `arcface_w600k_r50` 512, detector `scrfd_10g_bnkps` (the same pair as InsightFace `buffalo_l`) | Manual | Single-linkage chaining; dense n x n (about 40 GB at the 100,000-face cap); singletons not written as documented; a re-run discards names and merges (#354); no incremental assignment of new faces; no merge suggestions or split; no face quality gate; no per-project scoping, opt-out or purge (#379); faces never reach curation projects; no offline test of `auto_cluster_faces` (#353) |

Further defects found while reading, to fix in the waves that touch the code:

- `person_management.py` builds painless scripts by f-string interpolation of `person_name` and
  ids (`rename_person`, `merge_persons`): a name containing a quote breaks the script, and the
  value should be a script `params` entry. Every method also catches `Exception` and returns a
  `status: error` dict (fail-open shape). Wave W2 fixes both.
- `src/services/visual_search/search.py` `search_vehicles`, `search_people` and `search_by_object`
  call the synchronous Triton client inside `async def`, blocking the event loop; and
  `search_by_object` searches a non-vehicle, non-person **crop** vector against the whole-image
  index `visual_search_global` (same model, wrong population). Wave W4 fixes both.
- `src/services/curation/semantic_search.py` docstring names a route that no longer exists
  (`/curation/search/text`); `SemanticSearchDisabledError` is documented as "for any future
  caller" and has none (dead code). Wave W4 removes it.

### 1.2 The search kinds

| Search | Route (exists) | Space, index | Measured (uncommitted retrieval baseline, 2,000 COCO val images, loaded host) | Gaps |
|---|---|---|---|---|
| text -> whole image (core) | `POST /search/text` | MobileCLIP 512, `visual_search_global` | recall@10 0.886; client p50 41 ms, p95 54 ms | no project scoping |
| image -> whole image (core) | `POST /search/image` | MobileCLIP 512, global | precision@10 0.476 (category-overlap proxy); p50 61 ms, p95 72 ms | proxy relevance only |
| image -> object crop (core) | `POST /search/object` | MobileCLIP 512, vehicles / people / global | not measured | blocking Triton call; other classes hit the image index |
| face -> face (core) | `POST /search/face`, `POST /faces/search`, `POST /faces/identify` | ArcFace 512, faces | not measured | global only |
| text -> item (curation) | `GET /curation/projects/{project}/search/text` | PE-Core text tower -> item `pe_embedding`, Lucene HNSW cosine | recall@10 0.733 (images ranked by their best crop); server p50 110 ms, p95 161 ms; filtered p50 84 ms against unfiltered 18 ms; a 1.5 s outlier on about 1 in 10 queries at 91k items (#356) | off by default (`OP_SEMANTIC_SEARCH_ENABLED`, #371) |
| crop -> crop, image -> crop, text -> region box, text/image -> whole frame (curation) | none | vectors exist (`pe_embedding` on items and images docs, nested `region_box_embeddings`) | n/a | **not built**: the only curation kNN query is text -> item (`rg "'knn'" src` finds one curation call site) |

The OpenSearch fields are already right for the missing searches: Lucene HNSW cosine
(`_knn_field` in `src/clients/curation_opensearch/base.py`), and the box vectors are a nested
`knn_vector`, which OpenSearch searches with a nested kNN query that returns the best box per
parent document, or all matching boxes with `expand_nested_docs`
([nested k-NN docs](https://docs.opensearch.org/latest/vector-search/specialized-operations/nested-search-knn/),
[multi-vector blog](https://opensearch.org/blog/enhanced-multi-vector-support-in-opensearch-knn/)).
Efficient filtering runs inside the HNSW search for Lucene since 2.4
([efficient filtering](https://docs.opensearch.org/latest/vector-search/filter-search-knn/efficient-knn-filtering/));
the compose default is OpenSearch 3.6.0. A filter combined with a nested kNN query is an open
upstream request ([k-NN #1768](https://github.com/opensearch-project/k-NN/issues/1768)): wave W4
verifies it on 3.6 before relying on it.

### 1.3 Labeling and training pieces that the make/model workflow reuses

| Piece | Where | Note |
|---|---|---|
| Store every detection, per-project embedding policy | `generic_detector_and_selective_embedding_plan.md` (implemented) | the prior-art sorter lost precision by keeping only the largest box per photo; we already keep all |
| Class registry by name, with a one-level `group` keyword | `_classes_body` in `src/clients/curation_opensearch/bodies_other.py` | no taxonomy deeper than one level |
| VLM attributes declared by a pack (`item` or `box`) | `src/services/labeling/vlm_attributes.py` | suggestions, never labels; a natural home for a VLM-reported view or make |
| VLM `vlm_raw_label` hierarchical clusters for the labeler | `GET /curation/projects/{project}/review/raw_label_clusters` (`src/routers/curation/review_terms.py`) | free-text label grouping, useful for naming |
| Auto-label loop (#286) | branch `feat/v070-assist-loop` | gate, propagate-or-split, `assist_propagated` (not validated), audit, rollback |
| Lock rule, OCC writes, history | `src/clients/occ_locks.py`, `class_write_guard.py`, `src/domain/history.py` | every automated write in this plan goes through them |
| Export | `src/services/curation/export.py` (`GenericYoloExportService`), `export_single_class.py` | **YOLO detection only**: there is no classification-crop export, which a make/model classifier needs |
| Bake-off, training, promotion | `src/services/training/`, `scripts/curation/bakeoff/` | trains YOLO26 detectors |

## 2. Design principles (apply to every kind)

1. **One graph per (space, kind, level).** Never mix MobileCLIP, PE-Core and ArcFace vectors, or
   item and box vectors, in one distance computation, graph, centroid set or threshold.
2. **Name the level.** `class` level groups objects of one category (K2, K3, make/model);
   `identity` level groups one physical person or object (K4 faces, a future same-vehicle kind).
   Identity graphs add the free cannot-link "two detections in one image are two identities";
   class graphs must never use it (cross-domain lessons, section 6.1).
3. **Fix the configuration and input first, then the algorithm** (consolidated note sections 4 and
   7). Baseline batch method: sparse mutual kNN graph (k 30) and **average-linkage graph HAC** with
   a cut calibrated on the dev split (largest recall whose dev precision meets the target). IVF
   stays only as the coarse index for streaming assignment and kNN candidates, with K chosen for the
   index, not for clusters. Comparison arms: Leiden CPM (Traag et al. 2019,
   [Leiden](https://www.nature.com/articles/s41598-019-41695-z); keeps more precision at high
   recall, 0.776 at 0.51 against 0.40 at 0.59), CHAMELEON validation only if it adds held-out
   precision, and cuGraph ECG on GPU 2 (synthetic evidence only so far). An arm replaces the
   baseline only if it wins on real held-out data. Single linkage (union-find) and modularity are
   banned for identity work; HDBSCAN, FINCH and UMAP are rejected on real data.
4. **Multi-prototype clusters.** A cluster keeps up to 8 prototypes: medoids (real members) of its
   sub-communities, each with count, dispersion and an optional mode tag (a view for vehicles, a
   pose for faces). Assignment, merge scores and naming samples use the best prototype pair,
   never the mean. This is the domain-agnostic answer to multiple viewpoints: the modes are found
   by clustering, and a viewpoint tag is optional metadata.
5. **Abstain.** Online assignment accepts a new member only above a calibrated score **and** a
   margin to the second-best cluster; the band below goes to review; below that the member
   starts a microcluster (Immich's "deferred" pass is the same idea; section 3.2).
6. **Quality-gated evidence.** Small, blurred, truncated or occluded crops (and low-score or
   extreme-pose faces) may join clusters but never form or bridge one, are never prototypes and
   never count as merge evidence. Thresholds are project data.
7. **Corrections persist.** Names, merges, splits and validated labels survive every re-run; ids
   are stable by membership overlap (#343 for curation, wave W2 for persons). A correction is tied
   to the object (crop id plus box geometry by IoU), not to a cluster id.
8. **Provenance.** Machine merges, propagations and assignments carry their own provenance and are
   never validated; the lock rule (`src/clients/occ_locks.py`) stays the only gate.
9. **Order independence.** Every fit sorts its input by document id and seeds every random choice
   from a stable key (#372).

## 3. Faces (K4): design against Immich

### 3.1 What Immich does (public sources)

- Defaults: `facialRecognition.enabled true`, `modelName buffalo_l`, `minScore 0.7` (detection),
  `maxDistance 0.5` (recognition distance; docs advise 0.3 to 0.7), `minFaces 3`
  ([config file docs](https://docs.immich.app/install/config-file/),
  [facial recognition docs](https://docs.immich.app/features/facial-recognition)).
- Algorithm, incremental and DBSCAN-like: for each new face, a vector search within `maxDistance`
  returning up to `minFaces` neighbours; a face with at least `minFaces` neighbours is a core
  point. A face that matches a face already in a person joins that person; otherwise a core face
  creates a new person; a non-core face is deferred and processed after all others, and if still
  unmatched it is an outlier. The docs state that clustering is incremental, larger batches give
  better results, and re-running on all assets behaves closer to standard DBSCAN
  ([docs](https://docs.immich.app/features/facial-recognition);
  code `handleRecognizeFaces` in
  [person.service.ts](https://raw.githubusercontent.com/immich-app/immich/main/server/src/services/person.service.ts),
  read through a summary: re-verify any detail before copying it).
- People management: merge, hide, feature photo, birth date; a nightly job processes new faces; a
  forced re-run unassigns machine-assigned faces and re-queues all of them.
- Store: 512-d `vector(512)` columns in Postgres with VectorChord (pgvecto.rs dropped)
  ([v1.133.0 release](https://github.com/immich-app/immich/releases/tag/v1.133.0),
  [discussion 29984](https://github.com/immich-app/immich/discussions/29984)).
- Known weaknesses from its tracker: look-alikes and siblings merge while ordinary people split
  ([discussion 3946](https://github.com/immich-app/immich/discussions/3946)); one global threshold
  plus chaining builds "worm" clusters across age
  ([discussion 32382](https://github.com/immich-app/immich/discussions/32382)); no split-person
  tool ([discussion 7672](https://github.com/immich-app/immich/discussions/7672)); a full rescan
  discards manual merges and names
  ([immich-face-fix README](https://github.com/pabera/immich-face-fix/tree/main)). No merge
  suggestions were found (not verified as absent).

Because our models are the same pair as `buffalo_l` (SCRFD-10G detector, ArcFace R50 trained on
WebFace600K), Immich's thresholds are a reasonable **starting** point here: `maxDistance 0.5`
cosine distance is cosine similarity 0.5, much looser than our union-find default of 0.7. Neither
is calibrated; wave W5 calibrates on labelled pairs.

### 3.2 What we build (keep, improve)

| Concern | Design | Improves on Immich by |
|---|---|---|
| Neighbour search | kNN (k = 30) from OpenSearch k-NN on `visual_search_faces.embedding` for incremental work; FAISS flat/IVF over a point-in-time export for batch, blocked so memory is O(n k), never n x n | same idea, bounded memory proven by test |
| Batch grouping | mutual-kNN graph (k-reciprocal weights as a WB option), edges above a calibrated similarity floor, average or complete graph HAC or Leiden CPM (whichever wins E6 held out), CHAMELEON merge test only if it adds precision; same-image cannot-link; quality-gated faces cannot bridge | Immich chains through any core point (single linkage); we refuse bridges and same-photo merges |
| Incremental assignment of a new face | nearest **prototype** of each person (up to 8 per person, by pose mode); accept if calibrated score >= t_accept and margin to the second person >= m; between t_review and t_accept -> `suggested_person_id` (review); below -> unassigned, re-tried by the next batch (Immich's deferred pass) | per-person multi-prototype instead of "any member within distance"; an explicit review band; never silently attaches a look-alike |
| Core point rule | a new person needs at least `min_faces` (default 3, as Immich) mutually linked faces of acceptable quality | same, plus quality |
| Stable ids, corrections | person records keep their id when membership overlap (Jaccard) with the previous run's person is at least 0.5; named and merged persons are anchors: their faces are must-link, cannot move by a machine run; a human split writes a cannot-link between the two sides | Immich loses corrections on a forced re-run |
| Merge suggestions | person pairs whose best prototype pair is in the calibrated "same" band and passes the CHAMELEON test, ranked by score, served as a list with evidence (the two prototype faces and the score); a person accepts or rejects; a rejection is a cannot-link | not offered by Immich |
| Split | a person whose prototypes do not link to each other above the band, or that holds a same-image pair, is offered as a split proposal; a person can also split by selecting faces | Immich has no split |
| Quality | detector score >= 0.7 (Immich `minScore`), face box side >= 40 px (project setting), yaw estimated from the five landmarks (eye-nose-mouth geometry) within +-60 degrees for prototypes, Laplacian blur ratio as for crops; ArcFace w600k is not MagFace, so the embedding norm is **not** used as quality until measured ([MagFace](https://arxiv.org/abs/2103.06627), [CR-FIQA](https://arxiv.org/abs/2112.06592)) | Immich gates on detection score only |
| Singletons (#354) | a face that is not in a person has `person_id` null and `person_state` `unassigned` or `suggested`; no synthetic `person_<face_id>` ids; the route description says so | fixes the documented-vs-actual bug |
| Deletion (#354) | `DELETE /persons/{person_id}` with `delete_faces=false` sets the faces to unassigned (not new fake persons) and keeps a cannot-link so the next run does not rebuild the same person | |
| Privacy and purge (#379) | a per-scope face setting (`faces_enabled`, default on for the core API, off by default for curation projects, see 3.3); when off, `/faces/*` writes, `/persons/*` and face ingest refuse with a typed error (fail closed); a purge operation deletes every face vector, face crop reference and person record in the scope, requires a confirmation token (the count it will delete, returned by a dry run first), writes an audit event, and is verified by a count query that must return 0 | Immich has no purge of biometrics short of deleting assets |
| Haiku | **never** sees a face (biometric data); face judging uses public identity labels, and face contact sheets go to the local VLM only (sections 7.2, 7.4) | |

Calibration of t_accept, t_review, m and the graph floor: isotonic fit of same-identity
probability against cosine on labelled pairs (section 5.1), chosen so that accepted assignments
have precision >= 0.99 on the calibration split; values are stored with the run record together
with the model version (a threshold is valid only for its space and model).

### 3.3 Scope decision for the owner (O1)

Faces live only in the core API's global `visual_search_faces` index, outside project isolation.
#379 asks for a per-project switch. Options: (a) keep faces core-only and add a deployment-wide
switch plus purge; (b) add a project-scoped `faces` role index (`{prefix}{slug}__faces`) so a
curation project can opt in, with one face-identity service parameterized by index (no duplicate
logic; the core routes keep using the global index). **Recommendation: (a) in wave W2/W3, (b) only
if the owner wants faces in curation projects.** Either way the purge and the fail-closed switch
ship with the first face change that grows identity data.

## 4. Crop, item and whole-image semantic search

### 4.1 Routes (curation, project-scoped; all proposed)

| Proposed route | Query | Target | Notes |
|---|---|---|---|
| `{P}/search/similar` (GET, `crop_id`, optional `box_id`) | stored vector of an item or a box | items (`pe_embedding`) or boxes (nested kNN) | no encode step: the cheapest search; "more like this" from any card in Cropwright |
| `{P}/search/image` (POST, multipart image, optional `box` normalized) | PE image tower on the uploaded image or the given box | items, boxes or frames (`target=item|box|frame`) | the uploaded image is decoded once and encoded once; no detector run unless `detect=true` |
| `{P}/search/text` (exists) gains `target=item|box|frame` | PE text tower | as above | one route, one filter builder (`_build_filter`), no parallel route |
| all of them | `group_by=image` option | | ranks images by their best crop, the way the retrieval baseline scores |

One service module (`src/services/curation/semantic_search.py`, split if it crosses the size
warn band) owns the query builder for the three targets; the router stays a thin adapter. The
response reuses the wire item serializer (`src/services/curation/wire.py`) plus `semantic_score`
and, for box targets, `matched_box_id`. `make contracts` in the same commit.

Core API: `POST /search/object` keeps its contract but stops searching crop vectors against the
whole-image index (fix in 1.1) and moves the Triton call off the event loop. No new core routes.

### 4.2 Indexes

No new index and no mapping change: items and images docs already carry Lucene HNSW cosine
`pe_embedding`, boxes carry the nested vector. Kept as is: Lucene engine (recall near 1.0 under
filters, see the `_knn_field` docstring). Two measured decisions in W4: (a) whether a filtered
nested kNN works on OpenSearch 3.6 (if not, a pre-filter on the parent with `k` over-fetch and a
documented recall check); (b) the `ef_search`/`k` over-fetch that keeps ANN recall@100 >= 0.98
against exact search at 100k and 1M items.

### 4.3 Latency targets (server time, warm, one A6000 for the encoder on GPU 2, OpenSearch on the host)

| Search | 100k items | 1M items | Basis |
|---|---|---|---|
| crop -> crop / box -> box (stored vector) | p95 <= 50 ms | p95 <= 120 ms | unfiltered curation kNN is 18 ms p50 today at 2k images |
| text -> item/box/frame, page size 50, filters on | p95 <= 120 ms | p95 <= 250 ms | today server p95 161 ms at page size 200 on 2k images; #356 outlier must be gone (p99 <= 2 x p95) |
| image -> item/box/frame (upload, decode + encode + kNN) | p95 <= 200 ms for a 1 MP upload | p95 <= 300 ms | encode cost from `pe_encoder_batching_benchmark.md` |
| ANN recall@100 vs exact | >= 0.98 | >= 0.98 | Lucene default kept unless this fails |

These are targets to measure against, not claims; a missed target is recorded with its cause.

### 4.4 Measurement plan

Extend the retrieval harness (`scripts/bench/retrieval_eval*.py`, currently uncommitted in
`OpenProcessor-wt-ret`; wave W4 starts only after its owner merges it) with three scenarios:

1. **text -> crop** on public crops with crop-level text: Open Images boxes of vehicle and animal
   classes (label = class name, CC BY 4.0 annotations) and the Stanford Cars / Commons vehicle set
   (label = "make model year"); queries are class or model names and referring captions produced
   by a bulk local-VLM pass and spot-checked (200 by a person, 300 by the Haiku judge, section 7).
   Metrics: recall@1/5/10, MRR@10, nDCG@10 with bootstrap CI95.
2. **crop -> crop** leave-one-out on the same sets: precision@10 and mAP@100 with same-class (and,
   on CompCars, same-model) relevance.
3. **image -> crop**: query with a held-out photo's crop, relevance by model label.

Latency at concurrency 1 and 8, filtered and unfiltered, at 100k and 1M (synthetic vectors from
`scripts/bench/gen_synthetic_embeddings.py` for the 1M latency row; quality rows only on real
data). Three repetitions of each headline row; load average recorded.

## 5. Public-data proxies and evaluation sets

All evaluation data is public. Licences noted; anything non-commercial or of unclear licence is
used locally only, never redistributed, and only aggregate numbers are committed **after the owner
approves publishing them** (O2).

### 5.1 Faces

| Set | Use | Licence status |
|---|---|---|
| LFW (13,233 images, 5,749 identities, 1,680 with two or more) | identity clustering and pair calibration (BCubed, pairwise F); the approximate rank-order paper reports F 0.87 on LFW ([Otto, Wang, Jain](https://arxiv.org/abs/1604.00989)) | research use per secondary sources; confirm on the UMass page before use |
| CPLFW or CFP-FP | pose-stratified pairs (frontal vs profile) for the yaw bins | to verify before use |
| Open Images V7 "Human face" boxes | detection quality, throughput, chaining tests on unlabelled faces (no identities) | annotations CC BY 4.0, images listed CC BY 2.0 ([Open Images](https://storage.googleapis.com/openimages/web/factsfigures_v7.html)) |
| Synthetic 100k faces (random unit vectors) | **scale only**: memory and runtime bound for #354, never a quality number | generated |

Face research benchmarks for orientation (not reproduced here): MS1M part1 584K split, pairwise F
of GCN(V+E) 87.93 and BCubed 86.09 ([Liu et al. ICCV 2021 table](https://openaccess.thecvf.com/content/ICCV2021/papers/Liu_Learn_To_Cluster_Faces_via_Pairwise_Classification_ICCV_2021_paper.pdf)),
Ada-NETS BCubed 91.40 ([Ada-NETS](https://arxiv.org/abs/2202.03800)). Learned GCN clusterers
need identity-labelled training graphs of the target distribution; they are out of scope until the
graph method plateaus.

### 5.2 Vehicles (make/model proxy for set P)

| Set | Content | Use | Licence status |
|---|---|---|---|
| Stanford Cars | 16,185 images, 196 classes "Make Model Body Year" | classic fine-grained benchmark; kNN accuracy at make and model level | original site down, torchvision download broken ([vision #7545](https://github.com/pytorch/vision/issues/7545)); Kaggle mirrors list the licence as unknown; local only |
| CompCars web-nature | 136,726 images, 163 makes, 1,716 models, year, five viewpoint labels ([Yang et al. 2015](https://arxiv.org/abs/1506.08959)) | **hierarchy and viewpoint**: make -> model -> year, view-stratified pairs, open-set splits (hold out models) | non-commercial research, no redistribution; local only, owner approval (O2) |
| VMMRdb | 291,752 images, 9,170 classes ([paper](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w9/papers/Tafazzoli_A_Large_and_CVPR_2017_paper.pdf)) | long tail and scale | no formal licence found; local only |
| Wikimedia Commons vehicle-model categories (new "set C") | proposed: about 60 models x 40 photos from per-model categories, licence recorded per file | the only make/model set whose thumbnails may appear in committed docs and screenshots | per-file CC BY / CC BY-SA / public domain; labels are category-derived and noisy, so a person checks a 10 % sample |
| Open Images V7 vehicles (Car, Truck, Bus, Motorcycle, Land vehicle; confirm exact class names in `oidv7-class-descriptions-boxable.csv`) | class-level grouping, detection, the 1M scale rows | CC BY 4.0 annotations |

Viewpoint bins for evaluation: CompCars labels (front, rear, side, front-side, rear-side) and,
for sets without labels, the zero-shot view tag of section 6.3 checked against a 300-crop human
sample.

### 5.3 Whole images and items

The #341 3,000-item set (COCO + Open Images, strata: low-res, mixed scene, near-duplicate, rare,
similar classes) for K2 and K3; COCO val captions and the 2,000-image retrieval pin for K1 and
text search. The planted synthetic partitions from #342 are used for runtime and memory only.

### 5.4 Identity-level ground truth (same person, same vehicle)

Class labels say "two sedans of one model"; identity labels say "the same physical car" or "the
same person". Sources of real identity labels:

| Identity | Source | Strength | Status |
|---|---|---|---|
| Same person | LFW and CPLFW / CFP-FP identity labels | public, labelled by the dataset authors; pose-stratified pairs from CPLFW / CFP-FP | licence check (O2) |
| Same person, unlabelled real faces (Open Images, COCO person crops) | judged by a person or by the **local** VLM only (faces never go to Haiku) | sample-scale check of chaining and false merges | |
| Same vehicle, set P | **session bursts linked by the plate**: photos of one session taken seconds apart (EXIF burst, 6.7) whose plates give the same `confident` key through the product's identity-text feature (6.8, wave W12) form same-vehicle groups; different confident keys in one burst give hard negatives. A person confirms a seeded 200-group sample. Until W12 lands, the same rule runs as a local research script over the existing region text fields | real, multi-view (a passing vehicle is shot front, side and rear in one burst), large and free | decided (O7, section 14): local only; plate text, crops and timestamps stay in `artifacts_local/`; only counts and rates are committed; never sent to a hosted model |
| Same vehicle, set P without a legible plate | burst adjacency plus embedding similarity, confirmed by a person on a sample | weaker, labelled "burst-linked" | |
| Same vehicle, public | CO3D v2 car and motorcycle sequences (identity = sequence, multi-view with camera poses, CC BY-NC 4.0), VeRi-776 (on request, non-commercial), VehicleID (academic agreement) | real multi-view identity with view angles (CO3D) or surveillance re-ID (VeRi) | as in cross-domain lessons E4 and E5; licences per O2 |
| Same image content | #341 near-duplicate stratum (jittered copies of real images) and `/query/duplicates` | weak: augmentation, not a second photo | exists |

Identity metrics use these labels only; class metrics never use identity labels and vice versa.

### 5.5 Set P as a real evaluation set (numbers only)

- **Already-ingested private projects**: the 91k set P vectors named in the program plan C1, and
  any other set P project on the dev stack (W0 lists them with counts only, in
  `artifacts_local/`; the about 350k-crop corpus of
  [clustering_methods.md](../research/clustering_methods.md) section 1 is used if the owner
  confirms it is set P). Their human- and import-validated items are real class labels at scale; a
  frozen holdout is drawn once (seeded by crop id) and never used for calibration.
- **NVMe sample copies** (P-5k, P-20k of the program plan 6.2) for runs that need pixels
  (encoding a new arm, plate OCR, the judge).
- Every K2, K3 and make/model bar in section 8 is reported on the public set **and** on set P;
  a method that passes on public data and fails on set P does not ship as the default.

## 6. Make/model workflow for set P (worked example of the generic fine-grained pipeline)

The pipeline below is generic. "Vehicle" is the coarse class, "make/model" the fine taxonomy,
"viewpoint" the mode. For wildlife it would be species with pose modes; for parts, part number
with orientation modes. Nothing in code says "vehicle".

### 6.1 Stages

```text
photos -> detection (all boxes) -> quality score -> PE crop vector (exists)
       -> coarse class (detector class / registry group)
       -> fine graph per coarse class (mutual kNN + graph HAC or Leiden, per W0b and W5) = fine groups
       -> prototypes per group (sub-community medoids, optional view tag)
       -> link groups across modes (set-to-set prototype score + verification) = entity candidates
       -> naming on samples (person; local VLM; loop gate) -> taxonomy labels (make > model > generation)
       -> label propagation as suggestions -> loop gate writes assist_propagated -> review/audit
       -> export: detector at coarse level + classification crops at fine level -> train
       -> classifier predictions become class_source model (calibrated) -> recluster residual only
```

| Stage | Reuses | New | Notes |
|---|---|---|---|
| Detection | primary detector, every box stored | none | low detector confidence is fine for embedding; gating happens later (prior-art lesson: raw detector confidence was badly under-calibrated) |
| Quality score | `blur_lap_ratio`, `crop_rank_in_image` | `crop_quality` in [0, 1] from box side, blur ratio, truncation (box touches the frame edge), occlusion (max IoU with another box); thresholds in the region profile | gates prototypes and merge evidence (principle 6) |
| Embedding | PE-Core 1024 `pe_embedding` | none at first | W5 measures whether frozen PE separates models; if not, the adapter of 6.5 |
| Fine groups | #342 graph library, #343 partition records when landed | a `fine_group` run per coarse class (level `class`) | the method W0b and W5 pick (average-linkage graph HAC baseline, Leiden CPM arm); cut or resolution calibrated per class on the dev split; CHAMELEON floors only if they add held-out precision (the 128-d floors were wrong at 1024-d, #342 section 4) |
| Prototypes | AHC `refine_members` / Leiden sub-communities | up to 8 medoids per group, with count and dispersion | each prototype is a real crop: inspectable and a natural VLM sample |
| Cross-mode linking | | set-to-set score over the top prototype pairs, cohort-normalized; merge proposal only when the CHAMELEON test passes and no cannot-link exists; at most one hop per run | front vs rear of one model may never link by embedding alone: accepted; a person or the naming step links them ("linked by a person" vs "linked by evidence") |
| Naming | Cropwright cluster page, `review/raw_label_clusters`, VLM attributes, loop (#286) | representatives per group: one per prototype plus typical members (TypiClust-style typicality, [Hacohen et al. 2022](https://arxiv.org/abs/2202.02794)) | a group is named, not each crop |
| Taxonomy labels | registry `group` | `taxonomy_path` (6.4) | class identity stays the leaf **name** |
| Propagation | loop gate and `assist_propagated` | graph label propagation ([Zhou et al. 2004](http://papers.nips.cc/paper/2506-learning-with-local-and-global-consistency.pdf), [Iscen et al. 2019](https://arxiv.org/abs/1904.04717)) from validated seeds as a ranking signal for review, not a write | writes only through the loop gate |
| Open set | VLM `__new__` proposals are reported, never created | per-crop novelty score = calibrated distance to the nearest known-class prototype; groups with no known prototype above the band go to a "new candidate" queue | new classes are created by a person only |
| Export and train | YOLO export, trainer, bake-off, promote | classification-crop export by class name at a chosen taxonomy level | a detector with thousands of fine classes is the wrong tool; detect coarse, classify fine |

### 6.2 Multi-viewpoint handling

1. Modes come from structure, not a viewpoint model: inside a fine group, sub-communities
   (Leiden at a higher resolution, or AHC) are the modes; each gets a prototype.
2. Optional view tag per crop, zero-shot from stored vectors: score the crop's `pe_embedding`
   against PE text-tower embeddings of the project's view prompts ("a photo of the front of a
   {coarse class}", "...rear...", "...side...", "...front three-quarter...", "...rear
   three-quarter..."). Prompts are project data (region profile or prompt pack); no new model, no
   image re-read. W5 measures its accuracy on CompCars view labels; it ships only if accuracy >=
   0.8 on the five bins (else the tag stays off and modes stay unnamed).
3. Prototype per (group, view) when tagged; the loop samples at least one crop per prototype and
   propagates only if every prototype passes its gate (cross-domain lessons 6.5), so a cluster that
   mixes a model's front with another model's rear cannot pass on one view.
4. Evaluation is stratified by view difference (same view, adjacent, opposite) and by crop size
   (section 8).

### 6.3 Open-set new models

- Known/unknown scoring per crop: distance to nearest known prototype (kNN-based OSR); report
  AUROC on CompCars with 20 % of models held out. Literature: a good closed-set classifier with
  max-logit scoring is a strong open-set baseline ([Vaze et al. ICLR 2022](https://arxiv.org/abs/2110.06207)).
- Generalized category discovery as the target behaviour for "find new models among the
  unlabelled": GCD reports All/Old/New 39.0/57.6/29.9 on Stanford Cars and SimGCD 53.8/71.9/45.0
  ([GCD](https://arxiv.org/abs/2201.02609), [SimGCD](https://arxiv.org/abs/2211.11727)). Our
  version is non-parametric: known-class prototypes are anchors (must-link inside a class,
  cannot-link across classes) in the fine graph; communities with no anchor are new candidates.
  Report All/Old/New accuracy on the proxy so the number is comparable.

### 6.4 Hierarchical labels: taxonomy as data

- Registry gains `taxonomy_path` (array of class **names**, root to leaf, e.g. `vehicle > car >
  make > model > generation`; the leaf is the class name). Parents are registry entries too, so a
  crop labelled only to make level is valid ("make known, model unknown"), which is the honest
  label when a VLM or person is sure of the make only.
- Every boundary carries names: import maps by name, combine resolves path conflicts by name,
  export takes `taxonomy_level` and collapses leaves to their ancestor **by name**, training and
  promotion record the level. No class index crosses a boundary.
- Mapping migration in `src/clients/curation_opensearch/` (registry index only), wire models,
  `make contracts`. Owner design review DR-M1 first (wave W6).

### 6.5 When frozen PE is not enough: a trained adapter

If W5 shows frozen PE kNN accuracy at model level is too low to cluster (decision rule: model-level
cluster purity < 0.85 at a usable cluster size on CompCars), train a small projection head on
frozen PE vectors (linear or 2-layer MLP, supervised contrastive or ArcFace-margin loss) from
**human- and import-validated** labels only, with a frozen holdout. It needs no image re-read and
trains in minutes on GPU 2. Its output is a second vector field per project (the dual-field
migration of #365), never mixed with `pe_embedding`. This is #345's "embedding adapter" with the
confirmation-bias guard: no propagated or machine label ever enters its training set. Reference
points: DINOv2 ViT-L/14 linear probe about 90 % on Stanford Cars, CLIP ViT-L/14 zero-shot about
77 % ([DINOv2](https://arxiv.org/abs/2304.07193), [CLIP](https://arxiv.org/abs/2103.00020);
numbers from the papers' tables as relayed by secondary sources, re-check before quoting).

### 6.6 Bulk local-VLM passes: where they help and what they cost

Measured local VLM speed: **10.0 crops/s** (`POST /vlm/label_batch`, 32 crops per request, 8
client threads, one A6000 shared with Triton) and 9.8 crops/s for the continuous worker
(`docs/PERFORMANCE.md`, "Measured VLM labeling speed"). At that rate one GPU does about 36,000
crops per hour, 860,000 per day. The winter set P subset (300k photos at about 2.3 crops per photo,
about 690k crops) is about 19 GPU-hours on GPU 2.

| Bulk pass | Helps | Cost on GPU 2 at 10 crops/s | Guard |
|---|---|---|---|
| Captions per crop (short, attribute-style) | text -> crop search ground truth on public sets; `review/raw_label_clusters`-style grouping of free text; second "view" for clustering experiments | 100k crops: 2.8 h | captions are features, never labels |
| Coarse attributes (pack attributes: view, colour, body type, make if legible) | view tag cross-check; soft attribute veto on merges (both signals required) | 100k crops: 2.8 h | attributes flip with light; veto only with weak embedding similarity too |
| Labels on every crop | **not** recommended at set P scale: the loop labels a cluster from a statistically sized sample (29 to 54 crops per node at theta 0.9) | | the loop gate is the write path |
| Re-labels after a split | targeted | small | through the loop |

Run bulk passes only when GPU 2 is free of the Triton ingest: they share the card, and the ingest
slowdown with a resident VLM has not been measured on GPU 2 (measure it in W5 item 7 before
overlapping the two).

### 6.7 Attribute signals layer (applies to every kind)

Embeddings are one signal. Other signals often decide what embeddings cannot (a plate links the
front and rear of one car; a burst links frames of one moment). Each signal has a type, a
confidence and a role, and enters the graph and the merge validator as **evidence with a
confidence**, recorded on the cluster link (`links[]` evidence type `attribute:<name>`,
cross-domain lessons 6.2). **Rule: no single attribute decides a merge or a split alone.** A
must-link also needs embedding similarity above a calibrated floor (the embedding does not
contradict it); a veto also needs weak embedding similarity (cross-domain T11), except where stated
for plates at identity level. Human corrections are the only signal that acts alone.

| Signal | Type | Stored today? | Confidence from | Role | Level | Cost |
|---|---|---|---|---|---|---|
| Detector class | category | yes (`class_name`, detector score) | detector score, calibrated | soft prior: partition within class (consolidated E5); veto across coarse classes only when both are confident | class | free |
| Near-duplicate group | group id | yes (`duplicate_group_id`, `dup_is_representative`) | dedup threshold | must-link; one vote per group in gates and statistics | both | free |
| Viewpoint tag | category | no; zero-shot from stored vectors (6.2) | text-prompt margin | assigns members to prototypes; strata; **never** a veto (views differ inside one class) | both | cheap, no pixels |
| Plate text | identifier | partly: plate region boxes already carry `text*` fields; the key and rules are new (6.8) | OCR and VLM agreement, per-character confidence, format gate | must-link when confident and equal; cannot-link when confident and different (identity level) | identity; class must-link only | region stage cost |
| Dominant colour | colour histogram / name | no; computed from the decoded crop at ingest (one decode) or a pack attribute | histogram peakedness | soft veto at identity level only, with weak embedding similarity (colour shifts with light) | identity | cheap at ingest; backfill re-reads pixels |
| Scene / sign OCR text | text | core OCR index only (`/ocr`); region text in curation | OCR confidence | feature for whole-image events and places; never a must-link alone | K1 | existing OCR pass |
| EXIF capture time | timestamp | no; header read at ingest | camera clock (unsynced clocks across cameras) | defines bursts with camera id: frames within a few seconds are same-object candidates; must-link only with high embedding similarity or a plate match | both | cheap (header only) |
| EXIF GPS | coordinates | no | EXIF presence and DOP | feature for K1 events and places; **never served by default** (privacy) | K1 | cheap |
| Camera / burst id | id (camera model + serial, hashed; sequence number) | no | EXIF presence | scopes bursts; same-image and same-burst rules | both | cheap |
| Face-identity link | person id | core faces index only | calibrated assignment score (3.2) | must-link for images or person crops holding the same named person; feature for events | identity, K1 | existing face pass |
| VLM pack attributes (make if legible, body type) | category | yes when a pack declares them | VLM confidence level | soft feature and soft veto, both-signals rule (cross-domain E8) | class | VLM time |

New extraction (EXIF time, GPS, camera id, colour, view tag, plate key) lands with the W6 mapping
change as a typed `signals` object on items and images docs (one migration, `make contracts`), so
it is designed once. **EXIF time and burst grouping also feed the spatial work (#348)**: date
sessions and same-place shoots are the candidate sets for scene discovery and photogrammetry, so
W6 stores them in the form #348 needs (session id per images doc, burst id, GPS only in the
project and never in exports by default).

### 6.8 Registration-plate reading as an identification signal (product feature)

Plates are one instance of a generic **identity text** region: a region whose text identifies one
physical object (plates; serial numbers, bib or jersey numbers in other domains). The code says
"identity text"; the example plate region profile turns it on for vehicles.

What exists: the example plate profile in `examples/region_profiles/` (SAM 3 text-prompted
segmenter plus an OCR text-hint re-pass, optional detector leg), the region text reader
(`src/services/detection/region_text.py`: tallest-line selection, charset, length, stopwords,
placeholders), OCR and VLM readers with reconciliation fields on each box (`text`, `text_ocr`,
`text_vlm`, `text_confidence`, `text_choice`, `text_disagreement`), PP-OCRv5 in Triton, the pinned
Open Images plate sample (`scripts/datasets/manifests/oi_plates_300.json`, CC BY 2.0 images), the
plate bake-off example under `examples/bakeoff/` and the false-positive region bucket
(`src/routers/curation/regions_fp.py`). No permissively licensed plate detector ships; an operator
may plug in their own through the profile's detector leg.

Lessons from the private prior plate-reading work (generic; no names or data copied): a small detector
trained on curated, deduplicated data clearly beat public baselines, while the text-prompted
segmenter was the throughput wall; detecting on the vehicle crop inherits the vehicle detector's
recall ceiling, and some public detectors change accuracy sharply between full-frame and crop
inputs; many vehicle photos show no plate at all (side and three-quarter views), so plates are a
**minority** signal and never a precondition; in-plane rotation (leaning motorcycles) hurts
detection and reading; OCR needs a small border inset, letterboxed (not stretched) recognition
input, and a confidence from the per-character maximum logit rather than a mean softmax; a
permanent false-positive bucket with sub-centroids keeps non-plates out; whole-frame near-duplicates
(cosine >= 0.98 on the frame vector) must not straddle train and test.

Design:

| Part | Rule |
|---|---|
| Key | normalized identity text: uppercase, profile charset, stopwords and placeholders removed; a profile **confusable map** (for plates O/0, I/1, B/8, S/5, Z/2) applied for matching only; a profile format gate (`identity_text_formats`: optional regex list, else length bounds) |
| Confidence tiers | `confident`: format passes and either OCR and VLM agree or the OCR minimum per-character confidence is above a calibrated `c_hi`; `partial`: short, occluded or low-confidence characters (stored as a pattern with unknown positions); `none` |
| Must-link | two `confident` equal keys **and** crop embedding similarity above a calibrated low floor **and** compatible coarse class: identity-level must-link; at class level the same edge joins the two crops' view prototypes (the front and rear of one vehicle end up in one make/model group, the link embeddings cannot make) |
| Veto | two `confident` keys at edit distance >= 2 (after confusable folding): identity-level cannot-link (graph penalty, hard in the merge validator); overridden only by a person. No veto at class level (two cars of one model are one class) |
| Near match | edit distance 1, or a `partial` key compatible with a `confident` one: a review suggestion, never an edge |
| Ambiguity | several plates inside one item: only the plate box inside the item and dominant by area counts, else `none`; the same confident key on two items of one image: flag for review (reflection or double detection); the same key on visually incompatible vehicles (similarity below the floor): no edge, flagged (dealer or moved plates, misreads) |
| Privacy | project setting `identity_text_mode`: `off` (no text read or stored for identity regions, edges refused), `hash_only` (default: matching uses `text_key` = HMAC-SHA256 of the key with a per-project secret held in the project state directory; raw text fields are not stored), `raw` (raw text stored for review, owner opt-in). Plate text never appears in logs (a redaction filter plus a test), never in exports unless the operator opts in, never in committed files (tests use made-up strings), never sent to a hosted model. Purge: dry run returns counts, then deletes every identity-text field and key in the project, rotates the secret (old keys become meaningless) and writes an audit event; a count query must return 0 |

Evaluation (real data only):

- Public: Open Images plate sample (detection recall and precision at IoU 0.5 per input regime and
  per rotation stratum; a person transcribes the 300 plates locally for reading accuracy, text not
  committed), the OpenALPR benchmark images ([openalpr/benchmarks](https://github.com/openalpr/benchmarks),
  licence to verify), UFPR-ALPR ([Laroca et al.](https://arxiv.org/abs/1802.09567), research use on
  request) and CCPD ([detectRecog/CCPD](https://github.com/detectRecog/CCPD), research use) as
  optional local reading sets.
- Set P (numbers only): a person transcribes a seeded 500-plate sample; 200 plate-linked pairs and
  100 vetoed pairs checked by a person or the local VLM.
- Metrics: exact-match read accuracy, character error rate, `confident`-key precision (target
  >= 0.99), coverage (share of vehicle crops with a confident key), plate-link precision (target
  >= 0.99), identity-level same-vehicle recall by view bin with and without plate edges at a fixed
  false-merge rate, class-level make/model purity and propagation precision with and without plate
  edges, false merges blocked by the veto against true merges lost, region-stage cost per crop.

## 7. Visual verification protocol (Haiku as judge)

Purpose: a check, in testing, that clusters, search results and pairs are similar the way a person
would judge them, where no ground truth exists or to cross-check where it does. Haiku never writes
labels into a project.

### 7.1 Model, client, cost

- Model `claude-haiku-5-5` (current Haiku; $0.10 per million input tokens and $0.50 per million
  output tokens for prompts up to 100k tokens). Images cost about width x height / 750 tokens
  ([vision docs](https://platform.claude.com/docs/en/build-with-claude/vision); verify with
  `messages.count_tokens` in W8). The judge sees **contact sheets** (7.4), not single images: a
  6 x 4 sheet of 160 px thumbnails is 960 x 640 px, about 820 tokens for 24 members (128 px: about
  520; 192 px: about 1,180), against about 200 tokens for one 384 px crop. Offline runs use the
  Message Batches API (50 % of the price).
- Client: the official `anthropic` Python SDK in a research script
  (`scripts/research/visual_judge.py`, proposed), installed into the throwaway research venv, not
  into the product's requirements. Not routed through the product's OpenAI-compatible VLM client.
  A product integration would be a separate owner decision (O4).
- Credentials from the environment (`ANTHROPIC_API_KEY` or an `ant auth login` profile); never
  logged.

### 7.2 Data rules

- **Spot checks only.** Haiku and the VLMs used for verification are development and spot-check
  tools on small, seeded samples. No verification run ever sends all images of a set or project.
  A bulk local-VLM pass (section 6.6) is a separate, explicit choice made per run by the operator,
  never part of verification.
- Public images only by default (COCO, Open Images, Commons sets B and C, the local-only vehicle
  sets).
- Set P: only a sample the owner approves explicitly, identified by a manifest of content hashes
  stored in `artifacts_local/`; committed text holds aggregate numbers only.
- **Faces are never sent** (biometric data). Face clusters are verified against identity labels
  only.
- **Plate crops and plate text are never sent** to a hosted model (section 6.8); plate
  verification uses the local VLM or a person.

### 7.3 Sampling

| Unit | Sample | Question |
|---|---|---|
| Cluster sheet (purity and naming) | seeded: clusters stratified by size decile; one sheet per cluster (7.4 member selection) | "Do all numbered cells show the same {level}? List the cells that do not. Name what the majority shows." |
| Pair sheet (merge/split audit, false-merge estimate) | positives and negatives in equal numbers (proposed merges, cross-cluster nearest pairs, within-cluster farthest pairs); 12 pairs per sheet, one pair per row, labelled A1/B1 ... A12/B12 | "For each row: same {level}? yes, no or unsure." |
| Search sheet | the query (text in the prompt, or the query image in cell 0) and the top 23 results; one sheet per query | "Which numbered cells match the query?" |

`{kind}` and `{level}` come from the project (e.g. "car model", "animal species", "scene"). The
prompt asks for a strict JSON reply (structured outputs), shows no labels, cluster ids or roles,
and the cell order is shuffled by the seed; seed and order are recorded in the sheet manifest.

### 7.4 Contact sheets

Every judge and every sheet-based labeling call (Haiku and the local VLMs) receives contact
sheets: grid montages of small thumbnails, one sheet per cluster, per pair batch or per query.

- **Layout.** Default grid 6 x 4 (24 cells) at 160 px per cell (range 128 to 192 px, chosen in
  calibration below); each thumbnail is letterboxed into its cell on neutral grey (no
  stretching), with its cell index (1 to 24) drawn in a corner box; a 4 px gutter. The default
  bitmap font is used so rendering needs no font files and is identical on every host.
- **Member selection** (seeded by `{seed}:{sheet_id}` over member ids sorted by id): all cluster
  prototypes (up to 8), then members nearest to the prototypes, then 2 to 3 far or outlier members
  (largest distance to their best prototype), then random members to fill the grid; a cluster
  with fewer members gets a smaller grid. Roles are recorded in the manifest, never drawn on the
  sheet.
- **Builder.** `src/services/labeling/contact_sheet.py` (pure: selection, layout and rendering
  with Pillow from crop bytes the caller supplies; no OpenSearch, no network), so the research
  judge and the W7 group-naming step use one implementation. Output: a PNG written without
  timestamps or other varying metadata, and a manifest JSON
  `{sheet_id, builder_version, seed, grid, thumb_px, cells: [{cell, item_id, box_id, bbox_norm,
  content_hash, role}], png_sha256}`. Verdicts carry `sheet_id` and cell numbers, which the
  manifest maps back to item and box ids.
- **Reproducibility test.** `tests/test_contact_sheet.py`: building twice from the same inputs
  gives byte-identical PNGs; rebuilding from a saved manifest reproduces `png_sha256`; a changed
  source `content_hash` makes the rebuild refuse; cell-to-item mapping round-trips; the selection
  is independent of input order; letterbox geometry of an elongated crop is correct.
- **One sheet, two answers.** The cluster prompt asks for the same/different verdict, the odd
  cells and a name for the majority (from the project's registry names at the requested level if
  one fits, else free text or `unknown`), with a confidence level. The name is a suggestion: from
  Haiku it is only a verification signal; from the local VLM it may feed the W7 naming step, which
  writes only through the loop gate (#286).
- **Calibration (part of E0 in W8).** The same clusters rendered at 3 x 3, 4 x 4 and 6 x 4 and at
  128 and 192 px: verdict stability across grid sizes (kappa on same/different, overlap of the
  flagged cells for the members shown in both); agreement between Haiku, each tested local catalog
  VLM and ground truth (pairwise Cohen's kappa, Fleiss' kappa across judges). The default is the
  smallest grid and thumbnail whose kappa with ground truth is within 0.05 of the best setting.
- **Privacy.** Face sheets and plate sheets are built only for the local VLM; the sender refuses
  to pass a sheet whose kind is `face` or identity text to a hosted model (tested).

### 7.5 Budget and recording

- Per-run hard caps (`max_calls`, `max_input_tokens`), reserved write-ahead before each call as
  the loop's budgets are. Per owner decision O3 (section 14) the samples are small and counted in
  sheets (one call per sheet): per wave at most 20 cluster sheets, 10 pair sheets (120 pairs) and
  20 search sheets for Haiku, 50 calls and about 50,000 input tokens at the default layout; the
  agreement run of W8 (E0) at most 100 cluster sheets and 25 pair sheets (300 pairs) per kind for
  Haiku, plus the grid-size calibration on 30 of those clusters at 5 layouts (150 sheets), with the
  local VLMs judging the full E0 sample (200 cluster sheets, 42 pair sheets). Larger budgets need
  the owner's yes.
- Raw JSON per verdict under `artifacts_local/bench/cluster_search/<wave>/judge/<run_id>.jsonl`:
  `{run_id, model, prompt_version, unit, sheet_id, manifest path, verdict, odd cells, name,
  unsure, input_tokens, output_tokens, latency_ms, request_id}`, plus a run header with the
  sampling seed, budget, git sha and dataset manifest hash.
- **Agreement first** (consolidated note E0): before the judge is trusted on any kind, it is run
  on a labelled public sample of that kind (the 3,000-item set for crops) and its agreement with
  ground truth is reported (accuracy, Cohen's kappa on pairs, precision and recall of the "does not
  belong" flags, false-"same" rate, cost per judgement). Proposed bars, to be agreed with the
  owner: pair kappa >= 0.7 and flag precision >= 0.8. Below them the judge only ranks clusters for
  human review and its verdicts carry no weight.
- Output: judge-estimated purity and false-merge rate with Wilson 95 % intervals, next to the
  ground-truth metrics where they exist; disagreements are listed for a person to look at (a
  20-item human spot check per run).

### 7.6 Best practices for efficient local checking (built into W8)

- **Which sheets.** Half of each budget goes to targeted strata: borderline merges, the largest
  clusters, clusters with many low-margin members, judge or method disagreements. The other half
  is a seeded uniform random control, so purity estimates stay unbiased (targeted sheets are
  reported separately, never pooled into the estimate).
- **Position bias.** LLM judges favour positions ([Zheng et al. 2023](https://arxiv.org/abs/2306.05685),
  [Wang et al. 2023](https://arxiv.org/abs/2305.17926)); multimodal judges show similar biases
  ([MLLM-as-a-Judge](https://arxiv.org/abs/2402.04788)). Cell order is shuffled per sheet by the
  seed, pair sides are swapped at random, and 10 % of sheets are re-rendered in a second order to
  measure the residual effect.
- **Planted controls.** Every batch carries known-same and known-different sheets from labelled
  public data (about 10 %), giving false-accept and false-reject rates per run; a batch whose
  controls fail is discarded. (Gold questions are standard crowdsourcing practice; no single source
  is cited here, unverified.)
- **Ensemble.** Haiku plus two local VLMs (local only for faces and plates); a panel of smaller
  judges can match one larger judge at lower cost ([Verga et al. 2024](https://arxiv.org/abs/2404.18796)).
  Unanimous verdicts are accepted; any disagreement goes into the owner's G3 review list.
- **Enough sheets.** Purity is reported with a Wilson interval
  ([Wilson 1927](https://doi.org/10.1080/01621459.1927.10502953)) and a cluster-level bootstrap,
  because members of one sheet are correlated: the effective sample is sheets, not cells. At a
  purity near 0.9, a half-width of 0.05 needs about 140 independent judgements and 0.10 about 35;
  the bars in 8.2 state the interval, and a run that cannot reach it within budget reports
  "insufficient sample", not a pass.
- **Cache.** Verdicts are keyed by `(png_sha256, model, prompt_version)`; a re-run with an
  unchanged sheet reuses the verdict at no cost, and a changed prompt or model invalidates it.
- **Human audit.** The owner or a delegate re-judges a seeded 5 % of verdicts (at least 20 per
  run) blind to the judge's answer; judge-vs-human kappa is tracked per kind and a drop below the
  7.5 bar suspends that judge for that kind.

## 8. Measurement standard (every wave reports against this)

### 8.1 Protocol

- Public data in committed text; set P numbers only. Three repetitions of each headline row
  (different seeds, and for clustering a row permutation), mean and spread; one factor changed at a
  time.
- Every run records: git sha, dataset manifest hash, model and engine versions, embedding field,
  seed, GPU id, `nvidia-smi` snapshot, host load average at start and end (a 1-minute load above
  16 marks a run noisy: repeat before using it), wall time, peak RSS and VRAM.
- Raw JSON under `artifacts_local/bench/cluster_search/<wave>/<task>/`; small public result files
  that back a published number go to `docs/benchmarks/cluster_search/`.
- One scorecard envelope for every scorer: `{schema_version, kind, level, dataset, run_meta,
  metrics, strata, acceptance: {bar, value, pass}}`, produced by the existing scorers
  (`scripts/bench/cluster_eval.py`, the retrieval harness) and a shared writer
  `scripts/bench/scorecard.py` (proposed, W0).
- Notes go to `docs/research/` (Setup, Method, Results, Findings, Open issues, Reproduce).
- **Real data for every quality bar** (principle 0.7): a scorecard whose `dataset` is synthetic
  may only carry runtime, memory and latency fields; the scorer refuses quality fields for it
  (W0 implements the refusal).

### 8.2 Metrics and acceptance bars per kind

Common clustering metrics (all kinds): BCubed precision, recall, F (headline; [Amigo et al.
2009](https://doi.org/10.1007/s10791-008-9066-8)); pairwise precision, recall, F; false-merge and
false-split rates; AMI; unassigned/abstained fraction; risk-coverage curve when abstention exists;
stability ARI across seeds and row permutations; protected items moved (must be 0); runtime and
peak memory. Strata: low resolution, near-duplicate, similar classes, view difference, crop size.

| Kind | Test sets | Kind-specific metrics | Acceptance bars (to pass the wave that changes the method) |
|---|---|---|---|
| K1 whole images | 2,000-image COCO pin, Open Images 100k, set P sample (P-5k, same-session photos as near-duplicate groups); synthetic 1M for latency only | album purity by category set; text and image search recall | BCubed precision not below the current IVF partition at equal or higher recall; stability ARI >= 0.95 across permutations; trained index survives restart (#353) |
| K2 items | #341 3,000-item set; vehicle proxy; set P projects (validated items, frozen holdout) | majority-class share; label-propagation precision | BCubed precision >= 0.90 overall and >= 0.85 on low-res, with pair recall at least 3x the current 0.022; permutation ARI = 1.0 on identical vectors (#372); 0 protected moves |
| K3 region boxes | a box-level set built like #341 on the public car-wheel example plus Open Images parts; set P region boxes with validated classes | as K2 per box | same bars as K2, scored per box |
| K4 faces | LFW, CPLFW / CFP-FP pose pairs (quality); Open Images faces checked by a person or the local VLM; synthetic 100k for the memory bound only | persons per true identity (fragmentation), identities per person (contamination), chaining incidents, incremental assignment precision at coverage, names/merges preserved | BCubed precision >= 0.97; accepted incremental assignments precision >= 0.99; 100 % of names, merges and cannot-links preserved across a re-run; peak RSS at 100k synthetic faces O(n k), under 2 GB; purge leaves 0 vectors |
| Make/model (fine-grained) | CompCars, Stanford Cars, Commons set C; set P projects (class level) and plate-linked bursts (identity level, 5.4) | kNN top-1/top-5 at make and model level; cluster purity at each taxonomy level; view-stratified same-model recall; open-set AUROC; GCD All/Old/New; labels per correctly labelled crop | measurement first (W5 sets the baseline); product bars: groups offered for naming have model-level purity >= 0.90 (judge and ground truth); propagated-label precision >= 0.95 (audit lower bound, matches the loop's theta 0.9 / alpha 0.05 gate) |
| Plate identity signal (6.8) | Open Images plate sample, OpenALPR benchmark (licence permitting); set P 500 transcribed plates and 300 checked pairs (numbers only) | exact-match read accuracy, character error rate, coverage, link precision, veto blocks vs losses, recall lift by view bin | confident-key precision >= 0.99; plate-link precision >= 0.99; identity-level same-vehicle recall at fixed false-merge rate higher with plate edges than without in every view bin that has plates; purge leaves 0 identity-text fields and keys; no plate text in logs (test) |
| Search (all) | section 4.4 | recall@1/5/10, MRR@10, nDCG@10 with CI95, precision@10, mAP@100; p50/p95/p99 client and server, QPS at concurrency 1 and 8, ANN recall vs exact | latency targets of 4.3; crop-target text search recall@10 on crop-level queries not below the item target's; ANN recall@100 >= 0.98 |

Visual-judge results (section 7) are reported beside, never instead of, ground-truth metrics.

### 8.3 Gate sequence: every wave ends with machine metrics, LLM verification, then a human review

A wave is **done only when G3 passes**. The gates run in this order; a failed gate sends the wave
back, it does not skip ahead.

| Gate | What | Who | Recorded in |
|---|---|---|---|
| G1 automated metrics | the section 8.2 metrics on the **held-out** half of each real set named for the wave (W0 split manifests), 3 reps, scorecard envelope with `acceptance.pass` | Sonnet runs, Haiku collects | `artifacts_local/bench/cluster_search/<wave>/g1/`; summary rows in the wave's research note |
| G2 LLM visual verification | section 7 protocol on samples of the wave's output: Haiku judge on public images (and an owner-approved set P sample), local VLM for bulk checks and for faces; agreement with ground truth first, then verdicts on unlabelled output | Sonnet runs the scripts; the judge model | raw verdict JSONL under `.../<wave>/g2/`; judge purity and false-merge estimates with Wilson intervals next to G1 |
| G3 human review | the owner works through the wave's **review package** in Cropwright and records a verdict | owner | `docs/reviews/cluster_search/<wave>.md` (public projects only); set P remarks in `artifacts_local/reviews/<wave>.md`; the verdict is also posted on the wave's issue |

**Review package** (one per wave, prepared by the implementing agent, file
`docs/reviews/cluster_search/<wave>.md`, created from the template below):

```text
# <wave> review package
Stack: docker compose -p op-cs-review ... (ports), Cropwright URL, git sha, how to start it
Projects to open: <public project slugs and what each holds>; set P project (local only)
Clusters / persons / queries to look at: explicit ids, chosen as: 5 largest, 5 random, 5 with the
  worst G1 or G2 score, 5 the judge disagreed on, plus every merge/split proposal of the run
Checklist (tick each, add a note):
  [ ] cluster browse: members look alike; prototypes show the modes (views/poses)
  [ ] merge/split: accept one suggestion, reject one, split one cluster; ids and names survive a re-run
  [ ] people view: persons, a merge suggestion, a split, a rename, purge dry run (faces waves)
  [ ] crop semantic search: 5 text queries, 5 "more like this", 2 image uploads; results relevant
  [ ] make/model groups: names at make and model level; a new-candidate group; view prototypes
  [ ] judge verdicts shown beside the members agree with what I see
Expected states: full-page screenshots, desktop and 430 px wide, captured with Playwright on the
  public projects (DISPLAY=:11 for headed runs), stored next to the package
G1 / G2 summary: the numbers and where the raw JSON is
Owner verdict: pass | pass with follow-ups | fail     Date:     Notes:
```

Per-wave content of G1 to G3:

| Wave | G1 held-out real sets | G2 units judged | G3 checklist focus |
|---|---|---|---|
| W0 | re-score of the current partitions on the held-out halves | none (the judge is built in W8, which runs next) | read the scorecard and the held-out re-score; confirm the bars of 8.2 |
| W1 | label derivation spot check (200 per set) | 300 crops per set: "does the label match the image" | browse each proxy set in a throwaway project |
| W2 | LFW / CPLFW held out; reload after restart | faces: local VLM or a person only | people view (W-UI item 3): persons, rename, merge, delete, re-run keeps names |
| W3 | purge counts; incremental assignment precision on held out | local VLM on merge suggestions | purge dry run and run on a test project; accept/reject suggestions; split |
| W4 | retrieval held-out queries, public and set P | top-10 panels for 300 queries | crop search, more like this, image upload, box and frame targets |
| W5 | all experiments on held-out halves | cluster panels per arm (ECG, CPM, AHC) | research note read-through; a sample of make/model groups per arm in a throwaway project |
| W6 | export by name equality, import refusal cases | none (no clustering output) | taxonomy tree, export at make and model level, re-import |
| W7 | fine groups on held out, public and set P | cluster panels and naming checks | make/model groups, prototypes, new-candidate queue, loop per-prototype sampling |
| W8 | judge agreement per kind (E0) on contact sheets: local VLMs on 200 cluster sheets and 42 pair sheets, Haiku on at most 100 cluster sheets and 25 pair sheets, plus the grid-size calibration (7.4) | the protocol itself | read disagreements; decide whether the judge is trusted per kind |
| W-UI | frontend `npm run check` and `npm run build`; backend routes' tests | none | every 8.4 screen opened in a browser on the public projects, desktop and 430 px |
| W0b | E1 raw arms and E5 on held-out halves; #372 permutation test | judge panels: current IVF vs the new default | the same 20 clusters before and after |
| WB | the WB decision rules per variable and kind, held out | old-vs-new variant on the same sampled panels | the results note; 20 clusters per kind, old and new variant side by side |
| W12 | plate metrics of 8.2 on public plates and the set P sample | local VLM (never Haiku) on plate-linked and vetoed pairs | plate-linked groups with their evidence, the near-match queue, a `hash_only` project shows no plate text, purge dry run and run on a test project |
| W9 | adapter lift on held out | panels before/after | compare groups before and after the adapter |
| W10 | set P pilot, numbers only | approved set P sample | the pilot project, local only |

### 8.4 Cropwright work the human gate needs

Backend serves the data and the decisions; Cropwright renders them (no client-side scoring).
Built in wave W-UI (section 9), in step with the backend waves:

1. **Judge verdicts beside cluster members**: on `/p/{project}/clusters/{id}`, the members grid
   with each member's judge verdict (odd-one-out flag, unsure) and the run's judge purity, read
   from a backend verification record (a proposed read-only route
   `{P}/clusters/{cluster_id}/verification`, fed by importing a judge run into the project state
   directory). The same panel for search results.
2. **Prototypes and modes**: prototype strip at the top of a cluster, members grouped by
   prototype, optional view tag.
3. **People view** (core faces have no UI today): persons grid, person detail with faces grouped
   by prototype, rename, merge suggestions with the two evidence faces and the score,
   accept/reject, split by selection, purge dry run with the count.
4. **Merge/split proposals for item and box clusters** (with #343): a dry-run view of moves,
   splits and merges before apply.
5. **Semantic search panel**: text box with a target switch (items, boxes, frames), "more like
   this" on every card, image upload, group-by-image.
6. **Make/model**: taxonomy tree in the class panel, group naming with level choice, the
   new-candidate queue.
7. **Link evidence** (6.7, 6.8): on a cluster or a pair, why members are linked ("by embedding",
   "by plate (confident)", "by burst", "by a person"), the near-match review queue, and the project's
   identity-text mode and purge dry run in project settings.
8. **Review mode**: a page that opens the clusters, persons and queries a review package lists,
   in order, from a list the backend serves (the package's ids imported into the project state
   directory); the owner's verdict stays in the review file.

## 9. Waves

Model tiers follow the program plan: **Opus** for design reviews before risky stages (DR) and for
review of data integrity, isolation, privacy and destructive operations; **Sonnet** implements,
tests, runs and does ordinary reviews; **Haiku** runs fixed commands, collects JSON, checks links,
contracts and naming. Every wave: branch from `main` (or the named predecessor), small
conventional commits, `.venv/bin/pre-commit run --all-files` before each commit, never
`--no-verify`, red-first tests for fixes, `make contracts` with any route or wire change in the
same commit, CHANGELOG `[Unreleased]` entry, no push without the owner's yes, merged `--no-ff`.
**Every wave ends with gates G1, G2 and G3 of section 8.3, in that order, using the row of the
8.3 table for that wave; it is not done until the owner's G3 verdict is recorded as a pass.**

The consolidated note's experiments map onto the waves (its split rules apply: split by image,
50 % dev / 50 % test, test read once per method family, 1,000-resample bootstrap intervals,
class-disjoint splits for open-set):

| Experiment (consolidated note section 8) | Wave |
|---|---|
| E0 judge calibration | W8 (runs right after W0, W0b and W1) |
| E1 reproduce at 3,000 items (re-ingest), raw-vector arms; E5 detector-class prior | W0b |
| E1 representation arms (whitened, k-reciprocal), E2 crop preprocessing, E3 frozen embedding alternatives | WB (embedding and processing benchmark) |
| E4 adapter, E8 viewpoint, E9 fine-grained, E10 scale on real vectors | W5 (E4 feeds W9) |
| E6 faces | W2 (method) and W5 (calibration) |
| E7 whole image | W5 |

Shorthands:

```bash
T0="pytest <touched tests> -q --no-cov && .venv/bin/python -c 'import src.main' && .venv/bin/pre-commit run --all-files"
T1="docker compose -p op-cs-live -f docker/test/compose.yml up -d --wait && .venv/bin/python -m pytest tests/live -q --no-cov -m live; docker compose -p op-cs-live -f docker/test/compose.yml down -v --remove-orphans"
GPU2='flock -w 3600 "$SCRATCH/locks/gpu2.lock"'
DOCS=".venv/bin/python scripts/docs/check_docs_vs_code.py"
```

GPU preflight for any GPU step: `nvidia-smi` (GPU 2 must not hold someone else's job), list other
containers, timed `curl` of `/health` of your own stack only. Stack env: `TRITON_GPU_ID=2`,
`API_GPU_ID=2`, `VLM_GPU_ID=2`, own ports, `-p op-cs-<wave>`.

### W0: scorecard and harness extension (CPU) [Sonnet; Haiku runs]

- Depends on: #341 harness on `main` (it is), the 3,000-item set branch `fix/v070-341-work`
  merged, the retrieval harness merged by its owner.
- Files: `scripts/bench/cluster_eval_metrics.py` (add BCubed P/R/F, AMI, risk-coverage for
  partitions with an `abstained` state, optional `identity` and `view` label fields with pair
  strata by view difference and crop size); `scripts/bench/scorecard.py` (new: envelope writer,
  run_meta capture including load average and `nvidia-smi`); `scripts/bench/cluster_eval.py` and
  the retrieval scorer call it.
- Tests: `tests/test_cluster_eval_metrics.py` extended with hand-computed BCubed cases (perfect,
  all-in-one, all-singletons, a published worked example), AMI against scikit-learn, abstention
  coverage; `tests/test_scorecard.py` (envelope schema, acceptance pass/fail).
- Held-out sets (none exists yet; the #342 COCO result was tuned in-sample): split every real
  evaluation set once, seeded and by source image (identity sets by identity), into
  `calibration` and `heldout` halves; write the split manifests under
  `scripts/datasets/manifests/splits/` (public) and `artifacts_local/splits/` (set P). Calibration
  (CPM gamma, CHAMELEON floors, face thresholds) uses only `calibration`; every G1 number in
  section 8.3 comes from `heldout`. The scorer refuses a run whose tuning manifest overlaps its
  scoring manifest.
- Synthetic refusal: `scorecard.py` refuses quality fields when `dataset.synthetic` is true.
- Set P inventory: list the set P projects on the dev stack with item, labelled and validated
  counts only, into `artifacts_local/cluster_search/w0/setp_inventory.json`.
- Verify: `T0`; re-score the existing 3,000-item partition and the #342 COCO partitions on the
  held-out half; the new BCubed rows are appended to `cluster_eval_baseline.md`.
- Issues: comment on #341 (extension, held-out split), #342 (re-score held out).

### W0b: configuration and input baseline before heavier methods (CPU, GPU 2 for one re-encode) [Sonnet; Opus review of the kill rules and the #372 fix]

Why first: the consolidated note shows most fragmentation is configuration and input (section
"Real-data facts" above). Heavier methods are judged against the fixed baseline, not against IVF
at K = n // 6.

- Land the diagnostics: the scratch scripts `fetch.py`, `diag.py`, `heldout.py` (currently only in
  `$SCRATCH/cm-review/`) move to `scripts/research/cluster_diag/` as `fetch_vectors.py`,
  `diagnostics.py` (P@k by stratum, effective rank, hubness skew, PCA whitening fit/apply,
  k-reciprocal re-ranking, linkage curves) and `heldout.py`, with
  `tests/research_scripts/test_cluster_diag.py` (hand-computed small cases for every metric and
  transform). Reproduce the consolidated note section 4 numbers within the stated folds; the note's
  Reproduce section then points at the repo scripts.
- E1: re-ingest the pinned 3,000-item set into a throw-away project (`-p op-cs-e1`, GPU 2) and score,
  on the held-out half: IVF (current), global average-linkage AHC, complete-linkage AHC, Leiden
  CPM, ECG once, on the stored raw vectors; plus the 2b MiniBatchKMeans partition. Kill rule from
  the note: Leiden is not the default unless it beats average linkage at the chosen recall.
  Representation arms (whitening, k-reciprocal, preprocessing, other models) belong to WB.
- E5: detector-class prior on the residual-like low-confidence subset.
- #372 (product, red first): sort the pool by document id before sampling and fitting, seed from a
  stable key; test that a permuted fetch gives the identical partition.
- Product default (only if E1 confirms): the residual partition becomes the sparse kNN
  average-linkage graph HAC with a dev-calibrated cut; IVF keeps streaming assignment with K sized
  for the index; the cut is calibrated per project (anisotropy differs per project). sklearn AHC does
  not scale (about an hour and 124 GB at 347k, [clustering_methods.md](../research/clustering_methods.md)
  2.2), so above about 100k the method runs per coarse class or as approximate graph HAC; W5 E10
  measures it on real vectors.
- Gates: G1 on the 3,000-item and COCO held-out halves and the set P inventory projects; G2 judge
  panels on the baseline vs the new default; G3 package: browse the same 20 clusters before and
  after in Cropwright.
- Issues: #372 (close), #342 and #341 (comment), new `research(clustering): configuration
  baseline (E1 raw arms, E5) with the diagnostics in the repo`.

### W1: public proxy datasets (CPU, network) [Sonnet; Haiku collects manifests]

- Files: `scripts/datasets/fetch_vehicle_proxy.py` (Stanford Cars from a local mirror path given
  by the operator, CompCars from the operator's licensed copy, Commons set C by category list,
  Open Images vehicles via the existing `fetch_openimages_bench.py` patterns),
  `scripts/datasets/fetch_face_proxy.py` (LFW, a pose set, Open Images faces);
  `scripts/datasets/manifests/` (sha256 pins, licence per source; Commons per-file licence);
  label JSONL with `class_name`, `taxonomy_path`, `view`, `identity`.
- Rules: nothing non-redistributable is committed; Commons set C manifest is committed with
  attributions.
- Tests: manifest verification and label derivation unit tests (`tests/test_vehicle_proxy_labels.py`:
  Stanford class string -> make/model/body/year; CompCars ids -> path).
- Verify: `T0`; fetch dry run prints counts per class and licence.
- Owner gate: O2 (licences, publication of aggregates).
- Issues: new `research(datasets): public face and vehicle make/model proxies with licences`.

### WB: embedding and processing benchmark (GPU 2) [Sonnet runs; Opus review of protocol and the go/no-go per variable]

Runs right after W0 (scorecard), W1 (datasets) and W8 (judge calibration). **Rule: no change to
production pre- or post-processing, or to the embedding model, without a recorded WB result that
shows the gain on held-out real data, and an owner G3 pass.** The text-retrieval side of model
choice reuses [embedding_model_eval_plan.md](../research/embedding_model_eval_plan.md) (#347); WB
covers neighbour and cluster quality.

Protocol (written at the top of the results note `docs/research/embedding_processing_benchmark.md`):

- One factor at a time against the production reference (PE-Core-L14-336, resize then centre-crop,
  L2, raw cosine; MobileCLIP2-S2 for whole images; SCRFD + ArcFace R50 for faces). Same held-out
  real data for every variant (W0 split manifests); fixed seeds; 3 reps for headline rows.
- Metrics per variant: P@1/P@10 and 10-NN vote by stratum (size, aspect, class, view); the W0
  scorecard (BCubed, pairwise, false-merge) for the baseline clustering method of W0b at its
  dev-calibrated cut; retrieval recall@10 where the kind has queries.
- Cost per variant: encode throughput (crops/s or images/s) and peak GPU memory on GPU 2 (A6000),
  vector size, and the ingest-path change: whether it keeps decode-once (a variant must consume the
  frame or crop the pass already decoded), whether stored vectors must be recomputed (a re-embed of
  every item through the #365 dual-field migration, with its time at 100k and 1M estimated from the
  measured throughput), and the stored-box rule (the vector is computed from exactly the stored,
  clamped box; a margin or letterbox variant stamps its policy with the vector so the staleness check
  on `bbox_norm` still holds).
- Visual cross-check: for each variant that wins G1, the judge (Haiku on public panels, local VLM
  for faces and plates) compares cluster purity of the old and the new variant **on the same
  sampled panels** (same seeds, both partitions), within the 7.5 budgets.
- Raw JSON per run under `artifacts_local/bench/cluster_search/wb/<kind>/<variable>/<variant>/`;
  the small public result files that back the note's tables go to
  `docs/benchmarks/embedding_processing/` for the paper.

Variables (each a row group in the results note; kinds: whole images, crops/items, vehicles, faces):

| Variable | Variants | Kinds |
|---|---|---|
| Crop preprocessing | centre-crop (current), letterbox/pad to square, aspect-bucketed resize | crops, vehicles |
| Context | 0 %, 10 %, 20 % margin around the stored box | crops, vehicles, faces (alignment margin) |
| Input resolution and minimum size | model native vs one step up where the model allows; size gate at 24 / 32 / 48 px (gated crops join but never form clusters) | all |
| Embedding model (frozen) | PE-Core (current), MobileCLIP2-S2, DINOv2-L, SigLIP 2 (L or So400m), the #347 candidates; PE-Core-G if it fits | images, crops, vehicles |
| Faces | 5-point alignment (current) vs alignment plus margin; quality gate: detector score 0.5 / 0.7, minimum face size, yaw bins from landmarks, blur | faces |
| Post-processing | L2 only (current), centring, PCA-256 whitening fitted on the project pool or a reference pool, k-reciprocal re-ranking (k1 20, k2 6, lambda 0.3), per-project calibrated thresholds | all |
| Multi-crop / test-time augmentation | single view (current), horizontal flip average, two-scale average | crops, vehicles |

Decision rules (from the consolidated note, made stricter where it says so): a preprocessing
variant ships only if elongated-crop P@10 rises by >= 0.03 with no loss elsewhere (E2); a second
embedding model only if it beats PE-Core by >= 0.03 P@10 on the low-res and person strata or on
model-level vehicle P@10 (E3); whitening only if held-out precision at matched recall rises by
>= 0.02 (E1); TTA only if the gain covers its encode cost; a face variant only if BCubed
precision at equal recall improves on LFW and CPLFW held out. Ties keep the current path.

- Files: `scripts/research/embed_bench/` (`variants.py` preprocessing and post-processing variants
  as pure functions over decoded crops or stored vectors, `encode_variant.py` re-encode driver
  against Triton or a research venv model, `run_bench.py` per variable), unit tests in
  `tests/research_scripts/test_embed_bench_variants.py` (letterbox geometry, margin clamping to
  the frame, whitening fit/apply round trip, k-reciprocal on a hand-computed case).
- GPU: GPU 2 under `$GPU2`; reference models other than the shipped encoders run in the research
  venv or Docker, never installed into the product.
- Gates: G1 the decision rules above on held-out data; G2 old-vs-new panels; G3 package: the
  results note, plus the same 20 clusters per kind browsed in two throw-away projects (old and new
  variant) side by side.
- Issues: new `research(embed): embedding and processing benchmark for clustering (preprocessing,
  models, faces, post-processing, TTA)`; comment on #347, #365.

### W2: persons fixes #354 and FAISS reload #353 [Opus DR-F0 short review of the id-stability rule; Sonnet implements; Opus review of delete/merge writes]

- Files: `src/services/face_identity/person_management.py` (replace `auto_cluster_faces` with
  blocked kNN neighbour search, mutual-kNN graph and Leiden CPM, or average-linkage AHC on the kNN
  graph if Leiden is not adopted; quality gate; same-image cannot-link; stable ids by Jaccard
  >= 0.5; named/merged persons as anchors; singleton rule; painless `params` instead of string
  interpolation; typed errors instead of `status: error` dicts), split into
  `face_identity/person_grouping.py` (pure, numpy) and `person_store.py` (OpenSearch writes) if the
  file grows; `src/routers/persons.py` (descriptions match behaviour; response models);
  `src/services/visual_search/clustering.py` (`is_trained` loads from disk lazily) and a volume for
  `faiss_indexes/` in `docker-compose.yml`.
- New dependency check: `leidenalg`/`igraph` licences (GPL-3.0 for leidenalg) against the AGPL
  repo; if not acceptable, use average-linkage AHC on the sparse kNN graph (scikit-learn,
  `connectivity=`) which also does not chain.
- Tests (red first; small hand-built vectors prove the logic, not the quality):
  `tests/test_person_grouping.py` (a bridge pair does not merge two groups; a same-image pair
  never merges; singleton faces get `person_id` null; re-run keeps ids, names and merges; delete
  leaves a cannot-link; scale: 100k synthetic faces, RSS sampled, below 2 GB);
  `tests/test_person_management_scripts.py` (a quote in a name round-trips);
  `tests/test_clustering_reload.py` (train, drop the in-memory cache, ingest assigns).
- Quality (G1, real data): W0 scorer on the LFW and CPLFW held-out halves against the old
  union-find at 0.7 and at Immich's 0.5, recorded in a note section of `cluster_types.md` 4.2.
- Verify: `T0`; `T1` with a live faces case.
- Issues: close #354 and #353 (DoD of the program plan).

### W3: faces privacy, incremental assignment, merge/split suggestions [Opus DR-F1 (scope O1, purge semantics); Sonnet; Opus review of purge]

- Files: setting and guard (`src/config/` setting, read once, fail closed), purge service
  (`src/services/face_identity/purge.py`), routes proposed `persons/merge_suggestions`,
  `persons/{person_id}/split`, `faces/purge` (core prefix, or project-scoped per O1), person
  prototypes stored on a person record (new small index or fields, per DR-F1), incremental
  assignment at face ingest (`src/services/visual_search/ingest_single.py` /
  `ingest_batch.py` face path) with the review band.
- Tests: purge leaves 0 vectors and 0 person ids (count queries); disabled setting makes every
  face route refuse; suggestions ranked and a rejection becomes a cannot-link; split keeps ids
  stable for the larger side; incremental assignment accepts above t_accept with margin and
  abstains otherwise.
- Docs: `SECURITY.md` privacy section, docs-site faces page, CHANGELOG with upgrade note.
- Verify: `T0`, `T1`, `make contracts`, `DOCS`.
- Issues: close #379; new `feat(faces): incremental assignment with abstention, merge suggestions
  and split`.

### W4: curation crop, box and frame search [Sonnet; Haiku runs latency sweeps]

- Depends on: the retrieval harness merged; #356 root cause known (fix it here if it is in the
  search path).
- Files: `src/services/curation/semantic_search.py` (query builders for `item`, `box` (nested
  kNN with `inner_hits`, `score_mode max`), `frame` (images index); stored-vector lookup for
  similar; remove dead `SemanticSearchDisabledError` and the stale route name);
  `src/routers/curation/search.py` (proposed `{P}/search/similar`, `{P}/search/image`,
  `target` and `group_by` on the existing text route); `src/schemas/curation/` wire models;
  `src/services/visual_search/search.py` (run Triton calls in the executor; `search_by_object`
  stops querying the global index with crop vectors); `contracts/`.
- Tests: query-shape unit tests per target in `tests/curation/test_semantic_search.py`; router
  tests in `test_search_router.py`; project guard test (a search can only touch the bound
  project's indexes); a test that a crop vector is never sent to the whole-image index; live test
  in `tests/live/` for similar and image search.
- Measure: section 4.4 quality on real crops (Open Images vehicles + COCO + the proxies, held-out
  queries) and on a set P project (crop -> crop with validated classes as relevance, numbers
  only); latency at 100k real items and 1M synthetic vectors (latency only), 3 reps, on GPU 2
  for the encoder: `$GPU2 .venv/bin/python scripts/bench/retrieval_eval.py <scenario> --api
  http://127.0.0.1:<port> --project <p> --out artifacts_local/bench/cluster_search/w4/...`.
- Bars: 4.3 targets; results to a new section of the retrieval note and `docs/PERFORMANCE.md`.
- Issues: new `feat(search): crop-to-crop, image-to-crop, box and frame search in curation
  projects`; update #356, #371.

### W5: fine-grained and face measurement (research, GPU 2) [Sonnet runs; Opus review of the math and the go/no-go]

- Data: W1 proxies (held-out halves for scoring), the #341 3,000-item set, the set P projects of
  the W0 inventory (vectors read through a point-in-time export, read only) and set P plate-linked
  bursts (5.4) built from the P-20k NVMe copy. Encode public sets with the product encoders
  through a throwaway project on a stack bound to GPU 2 (`$GPU2`), or with
  `scripts/bench/export_project_embeddings.py` from that project.
- Experiments (one factor at a time, 3 seeds):
  1. Representation: the WB result (PE-Core unless a WB variant passed G3); kNN top-1/top-5 at make
     and model level reported for it; E9 seeds (1, 3, 5 per class) and label propagation.
  2. Fine graph: mutual-kNN + Leiden CPM (+ CHAMELEON) vs current IVF+AHC vs average-linkage AHC
     vs **cuGraph ECG on GPU 2**, per coarse class, CPM gamma log-grid 0.0003 to 0.01; purity at
     make and model level. ECG enters as a candidate only: its synthetic 20k 128-d result (below)
     counts for nothing until it holds on real crops and faces.
  3. Multi-prototype (P in 1, 2, 4, 8) vs centroid assignment of a 20 % holdout (cross-domain E3).
  4. Zero-shot view tag accuracy on CompCars views; view-stratified same-model recall.
  5. Open set: 20 % of models held out; AUROC of the novelty score; GCD-style All/Old/New.
  6. Faces: union-find (old) vs the W2 method on LFW and pose pairs; chaining incidents;
     calibration of t_accept, t_review, m and the graph floor.
  7. Bulk local-VLM captions on 5,000 public crops: throughput on GPU 2 and caption usefulness for
     text -> crop recall; ingest slowdown with the VLM resident on the same card.
  8. Identity level on real data: same-vehicle recall by view difference on set P plate-linked
     bursts and CO3D sequences at a fixed false-merge rate (does frozen PE link a front view to a
     rear view at all; cross-domain E4), numbers only for set P.
  9. Whole images (E7): hierarchical k-means vs flat IVF vs average-linkage HAC on MobileCLIP,
     label-set Jaccard purity on COCO val and the Open Images val sample, judge-rated coherence.
  10. Scale on real vectors (E10): kNN (FAISS GPU or CAGRA, PCA 256 vs 1024), graph HAC and Leiden
     at the set P inventory size and on about 860k COCO train2017 objects; wall time, memory, kNN
     recall, precision on COCO train labels. This replaces the synthetic 1M runs for any quality
     claim.
- Files: `scripts/research/fine_grained_eval.py`, `scripts/research/view_tag_eval.py` (research
  scripts; unit tests in `tests/research_scripts/` for any metric code).
- Output: `docs/research/fine_grained_grouping_eval.md` with the decisions: adapter needed (6.5
  rule), view tag ships (6.2 rule), CPM gamma and CHAMELEON floors per coarse class, face
  thresholds.
- Issues: new `research(clustering): fine-grained make/model grouping on public proxies`; comment
  on #342, #345, #347 (the same scorecard frames the #347 embedding comparison).

### W6: taxonomy as data, quality score, classification export [Opus DR-M1; Sonnet; Opus review of the migration]

- Files: registry mapping (`taxonomy_path`), class registry service and routes (by-name parent
  validation; refuse cycles and unknown parents: fail closed), import/combine path handling,
  `crop_quality` computed at ingest from existing fields (`src/services/curation/ingest_index.py`)
  and a backfill script, `src/services/curation/export_classify.py` (new: crops per class name at
  a chosen taxonomy level, split by source image so near-duplicates never straddle train/val),
  wire and `contracts/`, Cropwright types regenerated.
- Tests: taxonomy validation; export collapses by name (two projects with different class orders
  export identical folders); import with a parent unknown in the target refuses; quality score
  bounds; holdout items stay in the test split.
- Verify: `T0`, `T1`, `make contracts`, `DOCS`.
- Issues: new `feat(curation): taxonomy as data and a classification-crop export by class name`.

### W7: fine groups, prototypes and the loop [Opus DR-M2 (record shape, shared with #343); Sonnet]

- Depends on: W5 go; #343 records if landed (else the fine-group run stores its partition in a
  job directory like the loop's, and #343 adopts it).
- Files: `src/services/curation/clustering/fine_groups.py` (per coarse class graph run, level
  `class`), `prototypes.py` (sub-community medoids, view tags), worker integration in
  `scripts/curation/cluster_refresh_daemon.py` (on demand first, scheduled after #344), the loop's
  `PartitionReader` seam reading fine groups and sampling one crop per prototype
  (`assist_loop/partition.py`, `evaluate.py` on the loop branch), Cropwright cluster page shows
  prototypes and the "new candidate" queue (backend data, thin frontend).
- Tests: order independence (permuted input, identical groups); lock rule per locked class;
  per-prototype gate (a mixed-view node fails); novelty queue membership.
- Verify: `T0`, `T1`; W5 proxy rerun through the product path reproduces the research numbers
  within CI.
- Issues: update #286 (per-prototype sampling), #343 (prototypes and level fields), #344
  (abstention and microclusters).

### W8: visual judge (research) [Sonnet; Haiku is the judge under test]

- Files: `src/services/labeling/contact_sheet.py` (the sheet builder of 7.4),
  `scripts/research/visual_judge.py` (cluster, pair and search sheets; Haiku through the
  `anthropic` SDK and the local VLMs through the existing OpenAI-compatible endpoint; budgets;
  JSONL recording; Message Batches), `scripts/research/judge_agreement.py` (kappa per judge, grid
  stability, Fleiss' kappa).
- Steps: build sheets and manifests; the 7.4 calibration (grid sizes, thumbnail sizes, Haiku vs
  each local VLM vs ground truth) on labelled public samples per kind; fix the default layout;
  then judge the W2, W4, W5 and W7 outputs; report beside the ground truth.
- Tests: `tests/test_contact_sheet.py` (7.4), budget never exceeded (unit, with a fake client),
  a face or identity-text sheet is refused for a hosted model, verdict JSON schema, verdicts map
  back to item ids through the manifest.
- Output: section in `docs/research/fine_grained_grouping_eval.md` and the retrieval note.
- Owner gates: O3 (API spend and set P sample approval).
- Issues: new `research(eval): visual verification protocol with a judge model`.

### W9: adapter and classifier (only if W5 says so) [Sonnet; Opus review of the leakage guard]

- Files: `scripts/curation/train_adapter.py` (validated labels only, frozen holdout, writes a
  versioned adapter), dual-field write through #365's migration path, classifier predictions as
  `class_source: model` with calibrated confidence (temperature scaling on the holdout).
- Tests: a propagated or machine label never enters the training set (fails if the filter is
  removed); holdout never trained on; calibration error reported.
- Measure: lift in model-level purity and kNN accuracy on the proxy; GPU 2 under `$GPU2`.
- Issues: update #345, #365.

### W10: set P pilot (private; numbers only) [Sonnet; owner-gated]

- 20k-crop pilot from an owner-approved seeded sample, read-only from the NAS through the
  existing bind mount, project `-p op-cs-p`, GPU 2: ingest, fine groups, naming on 50 groups by a
  person, loop on the rest with a small budget, judge on an approved sub-sample, export at make and
  model level, a classifier trained and evaluated on a frozen holdout.
- Committed: aggregates only (counts, purities, rates, timings) in the research note; everything
  else in `artifacts_local/`.
- Owner gates: O5.

### W11: audit [Opus, read-only]

- Completion matrix against section 8 bars, DoD of the program plan, naming scan, contracts,
  docs check; the audit note `docs/research/cluster_search_audit.md`.

### W-UI: Cropwright screens for the human gate [Sonnet; Opus design review DR-U1 of the verification record route]

- Depends on: the backend route of each screen (W2/W3 people view, W4 search panel, W7 prototypes
  and make/model, W8 verification records); built in slices alongside those waves.
- Files: `frontend/src/lib/api_*.ts` and `types_*.ts` from regenerated contracts, components
  under `frontend/src/lib/components/` (people, prototypes, verification panel, search panel,
  taxonomy tree, review mode), routes under `frontend/src/routes/p/`; backend read-only
  routes for verification records and review lists (proposed, section 8.4) with `make contracts`.
- Tests: `npm run check`, `npm run build`, component tests where the frontend has them; backend
  route tests; Playwright screenshots (desktop and 430 px, `DISPLAY=:11` for headed runs) that
  become the review packages' expected states.
- Issues: new `feat(cropwright): review screens for clustering and search (verification panel,
  people view, prototypes, search panel, taxonomy, review mode)`.

### W12: identity text (registration plates) as a clustering signal [Opus DR-L1 (key, tiers, privacy modes, edge rules); Sonnet; Opus review of privacy, purge and logging]

- Depends on: W6 (the `signals` mapping change and the DR for it), W7 (fine groups and the merge
  validator consume the edges). The reading-quality half (detection, OCR, key, tiers, privacy) can
  start after W6.
- Files: `src/services/detection/identity_text.py` (pure: normalization, confusable folding,
  format gate, tiers, edit distance, partial patterns; thresholds from the region profile: new
  profile fields `identity_text`, `identity_text_formats`, `identity_text_confusables`, with the
  example plate profile updated); `src/services/curation/identity_text_store.py`
  (per-project HMAC secret in the project state directory, `text_key` writes through the OCC
  helpers, purge with dry run and audit event); project setting `identity_text_mode`
  (`off`/`hash_only`/`raw`, fail closed when unset for an identity-text profile);
  `src/services/curation/clustering/attribute_edges.py` (must-link, cannot-link, near-match
  suggestions from 6.7 signals, embedding-floor rule, level rules); a log redaction filter in
  `src/core/logging.py` for identity-text fields; OCR checks from the prior work (letterboxed
  recognition input, border inset, per-character max-logit confidence) verified in the existing
  reader and fixed where they differ; proposed route `{P}/identity_text/purge`; wire models and
  `contracts/`; Cropwright link evidence and settings (8.4 item 7).
- Tests (made-up plate strings only): normalization and confusable table; tiering (confident,
  partial, none); a confident equal key with embedding similarity below the floor makes **no**
  edge; a veto applies at identity level and never at class level; edit distance 1 gives a review
  suggestion, not an edge; two plates in one item give `none`; `hash_only` stores no raw text;
  keys differ across projects; `off` refuses edge building; purge leaves 0 identity-text fields and
  keys and rotates the secret; caplog shows no fixture plate text in any log line of a full region
  pass.
- Measurement: 6.8 evaluation on GPU 2 (region stage), held-out halves, numbers only for set P;
  the plate-linked burst labels of 5.4 rebuilt through the product path and compared with the
  research-script version.
- Gates: the W12 rows of 8.2 and 8.3.
- Issues: new `feat(curation): identity text (registration plates) as a clustering signal with privacy
  modes and purge`; comment on #343 (link evidence), #348 (bursts), #379 (shared purge pattern).

Ordering and parallelism: W0 and W1 first (CPU, parallel), with W0b (configuration baseline)
right after W0. **W8 (the judge tool and its agreement runs) next, because every later G2 uses
it, then WB (embedding and processing benchmark), whose results fix the representation every
later wave uses.** W2 can run in parallel with WB (it fixes shipped bugs; faces have their own WB
rows). W4 and W5 after WB (W5 holds GPU 2 in blocks; W4's GPU need is short). W3 after W2 and
DR-F1. W6 after DR-M1, parallel with W5. W7 after W5 and W6. W12 after W6, its edge half after
W7. W-UI slices follow each backend wave so its G3 can be reviewed in Cropwright. W9 only on W5's
rule. W10 after W7 and W12. W11 last.

## 10. Issues to create or update

| Action | Issue |
|---|---|
| create | `research(datasets): public face and vehicle make/model proxies with licences` (W1) |
| create | `feat(faces): incremental assignment with abstention, merge suggestions and split` (W3) |
| create | `feat(search): crop-to-crop, image-to-crop, box and frame search in curation projects` (W4) |
| create | `research(clustering): fine-grained make/model grouping on public proxies` (W5) |
| create | `feat(curation): taxonomy as data and a classification-crop export by class name` (W6) |
| create | `feat(clustering): fine groups with multi-prototype clusters and a new-candidate queue` (W7) |
| create | `research(eval): visual verification protocol with a judge model` (W8) |
| create | `feat(cropwright): review screens for clustering and search (verification panel, people view, prototypes, search panel, taxonomy, link evidence, review mode)` (W-UI) |
| create | `research(clustering): configuration baseline (E1 raw arms, E5) with the diagnostics in the repo` (W0b) |
| create | `research(embed): embedding and processing benchmark for clustering (preprocessing, models, faces, post-processing, TTA)` (WB) |
| create | `feat(curation): identity text (registration plates) as a clustering signal with privacy modes and purge` (W12) |
| update | #348 (EXIF sessions, bursts and GPS stored in the form the spatial work needs, 6.7) |
| create | `fix(search): object search runs Triton on the event loop and queries crops against the whole-image index` (W4) |
| update | #341 (BCubed, AMI, abstention, identity/view strata), #342 (re-score, fine-grained calibration), #343 (level, prototypes, links), #344 (abstention, microclusters), #345 (adapter rule), #347 (scorecard frame), #353, #354 (design here), #356, #365, #371, #372 (order independence everywhere), #379, #286 (per-prototype sampling) |

## 11. Risks

| Risk | Mitigation |
|---|---|
| Frozen PE cannot separate models of one make | W5 measures first; adapter (6.5) trained on validated labels only; otherwise stay at make level and say so |
| Opposite views never link by embedding | accepted; linking by a person or by naming; the UI says which |
| Leiden calibration does not transfer from proxies to set P | calibrate per coarse class; re-check on the W10 pilot with a person's sample |
| `leidenalg` licence (GPL-3.0) | check against AGPL in W2; fallback is average-linkage AHC on the sparse kNN graph |
| Proxy licences (CompCars, Stanford Cars, VMMRdb, LFW) | local only; aggregates published only with owner approval; Commons set C for anything shown |
| Judge bias or drift | agreement measured per kind before use; verdicts never weigh as ground truth; prompt versioned |
| Biometric leakage | faces never sent to an external model; purge and fail-closed switch ship with the first face identity change |
| Plate text leakage | `hash_only` default, no plate text in logs, exports or committed files, never sent to a hosted model, purge with secret rotation, tests that fail on a leak |
| One strong attribute forces a wrong merge (shared or misread plate, burst with two vehicles) | the 6.7 rule: no attribute decides alone; embedding floor on every must-link; near matches only to review |
| A preprocessing or model change looks good on one stratum and hurts another | WB decision rules require no loss elsewhere; old-vs-new judge panels; owner G3 before any production change |
| GPU 2 contention (shared with ingest and the VLM) | one job at a time under the lock; bulk VLM passes only when ingest is idle |
| Scope creep into a vehicle-specific product | every vehicle term is project data; a test project with a non-vehicle taxonomy (Open Images animals) runs through W6/W7 |

## 12. Open questions for the owner

| ID | Question | Recommendation |
|---|---|---|
| O1 | Faces: core-only with a deployment switch, or also a project-scoped faces kind? | core-only now; project kind later on demand |
| O2 | May CompCars, Stanford Cars, VMMRdb and LFW be used locally, and may aggregates be published? | local yes; publish aggregates after review |
| O3 | Haiku judge budget per wave (default 3,000 calls) and approval of a set P sample for judging | yes for public data; set P sample after W7 |
| O4 | Should Haiku (or another hosted model) ever be a product-side verifier, behind the external-endpoint consent? | no for now; revisit after W8 |
| O5 | Set P pilot size and composition (20k crops, seeded across sessions) | yes |
| O6 | Make/model labels from the loop (`assist_propagated`) in the training export? | only after audit precision >= 0.95 on that class |
| O7 | May plate OCR on set P bursts be used locally to build same-vehicle labels (plate text never leaves `artifacts_local/`)? | **decided yes** (section 14). Extended: plates are also a product signal (6.8, W12) with `hash_only` as the default mode, a project switch and purge; the test labels of 5.4 are rebuilt through that product path |
| O9 | Default `identity_text_mode` for new projects with an identity-text profile: `hash_only` (recommended) or `off`? | `hash_only` |
| O10 | Store EXIF GPS in curation projects (never served by default, never exported by default) for the spatial work (#348)? | yes, project-scoped |
| O8 | Is the review-package format (8.3) and a G3 verdict per wave the right amount of owner time? | yes; the package lists at most about 25 clusters and 12 queries |

## 13. Sources

Immich: [config file](https://docs.immich.app/install/config-file/),
[facial recognition](https://docs.immich.app/features/facial-recognition),
[person.service.ts](https://raw.githubusercontent.com/immich-app/immich/main/server/src/services/person.service.ts),
[discussion 3946](https://github.com/immich-app/immich/discussions/3946),
[discussion 32382](https://github.com/immich-app/immich/discussions/32382),
[discussion 7672](https://github.com/immich-app/immich/discussions/7672),
[discussion 29984](https://github.com/immich-app/immich/discussions/29984),
[v1.133.0](https://github.com/immich-app/immich/releases/tag/v1.133.0),
[immich-face-fix](https://github.com/pabera/immich-face-fix/tree/main).
Clustering: [Leiden](https://www.nature.com/articles/s41598-019-41695-z),
[approximate rank-order](https://arxiv.org/abs/1604.00989),
[L-GCN](https://arxiv.org/abs/1903.11306), [GCN-V/E](http://personal.ie.cuhk.edu.hk/~ccloy/files/cvpr_2020_gcnface.pdf),
[STAR-FC](https://arxiv.org/abs/2103.13225), [Ada-NETS](https://arxiv.org/abs/2202.03800),
[pairwise-classification comparison table](https://openaccess.thecvf.com/content/ICCV2021/papers/Liu_Learn_To_Cluster_Faces_via_Pairwise_Classification_ICCV_2021_paper.pdf),
[FINCH](https://arxiv.org/abs/1902.11266), [TEMI](https://papers.bmvc2023.org/0297.pdf),
[k-reciprocal re-ranking](https://arxiv.org/abs/1701.08398),
[BCubed and extrinsic metrics](https://doi.org/10.1007/s10791-008-9066-8).
Face quality: [MagFace](https://arxiv.org/abs/2103.06627), [CR-FIQA](https://arxiv.org/abs/2112.06592),
[FIQA bias study](https://arxiv.org/abs/2211.15265).
Fine-grained and open set: [CompCars](https://arxiv.org/abs/1506.08959),
[VMMRdb](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w9/papers/Tafazzoli_A_Large_and_CVPR_2017_paper.pdf),
[BoxCars116k](https://arxiv.org/abs/1703.00686),
[Stanford Cars download issue](https://github.com/pytorch/vision/issues/7545),
[open-set recognition](https://arxiv.org/abs/2110.06207), [GCD](https://arxiv.org/abs/2201.02609),
[SimGCD](https://arxiv.org/abs/2211.11727), [DINOv2](https://arxiv.org/abs/2304.07193),
[CLIP](https://arxiv.org/abs/2103.00020).
Labeling: [label propagation, Zhou et al.](http://papers.nips.cc/paper/2506-learning-with-local-and-global-consistency.pdf),
[label propagation for deep SSL](https://arxiv.org/abs/1904.04717),
[TypiClust](https://arxiv.org/abs/2202.02794).
Search: [nested k-NN](https://docs.opensearch.org/latest/vector-search/specialized-operations/nested-search-knn/),
[multi-vector support](https://opensearch.org/blog/enhanced-multi-vector-support-in-opensearch-knn/),
[efficient filtering](https://docs.opensearch.org/latest/vector-search/filter-search-knn/efficient-knn-filtering/),
[k-NN #1768](https://github.com/opensearch-project/k-NN/issues/1768),
[RegionCLIP](https://arxiv.org/abs/2112.09106).
Data: [Open Images V7](https://storage.googleapis.com/openimages/web/factsfigures_v7.html).
Judge: [Claude vision docs](https://platform.claude.com/docs/en/build-with-claude/vision).
Plates: [OpenALPR benchmarks](https://github.com/openalpr/benchmarks),
[UFPR-ALPR](https://arxiv.org/abs/1802.09567), [CCPD](https://github.com/detectRecog/CCPD).
Methods consolidated (real-data evidence, recommended stack, E0 to E10): branch
`docs/cluster-methods-review`, `docs/research/clustering_research_consolidated.md`.

## 14. Owner decisions (2026-10-29)

- O3: Haiku may judge public-data clusters. The 3,000-call default is too many images per wave: judge
  small subsets (few clusters, few members per cluster, small pair samples) and size the budget per
  wave from that. A set P sample is asked again before use.
- O7: plate OCR on set P bursts may run locally to build same-vehicle labels. Plate text stays in the
  gitignored local artifacts folder and never reaches the repo or any hosted model.
- O2, O5, O6, O8 follow the recommendations in section 12; O1 and O4 stay as recommended (core-only
  faces, no product-side hosted verifier).
