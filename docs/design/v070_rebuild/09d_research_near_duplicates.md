# Near-duplicate detection and grouping: research and clean-rebuild plan

> Contracts (tables, fields, job kinds) live in 02d_research_seam_schema.md and the lane files; this file is the rationale and the prototype design.
> A coding lane reads 02d and its lane section, not this whole file.

| Field | Value |
|---|---|
| Superseded parts | the landing lanes, waves and file ownership of this plan's section 8, its GPU lock and its GPU-hour estimates are superseded by the program (review RR-16, owner answers of 2026-11-04): lanes in [03d_stage4_job_families.md](03d_stage4_job_families.md) (W2.3a and W2.3b), [03b_stage2_control_data.md](03b_stage2_control_data.md), [03f_stage6_scheduler_scaleout.md](03f_stage6_scheduler_scaleout.md), [05_b4_stateless_api.md](05_b4_stateless_api.md) and [06e_w5e_plate_text.md](06e_w5e_plate_text.md); one lock `gpu<N>.lock` per card shared by every topic ([traceability.md](reference/traceability.md) X-14); short tiered runs under a cap of about 10 GPU hours before the tag ([v070_rebuild/08_research_prototype_track.md.md](08_research_prototype_track.md) section 2.1). A fresh agent loads those files first |
| Status | **plan only**: nothing implemented, nothing run. Written 2026-11-04 on `research/near-dup` from `origin/main` at `497315954`; every `path:line` below is at that commit |
| Scope | whole images and crops; burst and sequence duplicates; exact, perceptual and embedding duplicates; the grouping thresholds; group storage and versions; re-score staleness; holdout leakage; dedup at ingest versus as a batch job |
| Issues | #416 (threshold research), #473 (stale fields, core batch race, holdout check), #346 (research tracker; its 2026-11 comment puts #416 on the ship track) |
| Absorbs | `docs/design/issue_plans/issue-416-plan.md` (deleted in this commit). Its transform set, bars, traps and corrections are carried into sections 2, 6 and 9 here; where they differ, this file wins |
| Supersedes for near-duplicates | RQ-2, E-2 and OQ-9 of [09b_research_clustering.md](09b_research_clustering.md); tasks 2D-4, 4A-6, 4A-7 and 4A-4b of the program files are re-cut in section 11 |
| Owner decision already applied | 2026-11-04 (MQ-5, [v070_rebuild/reference/decision_register.md.md](reference/decision_register.md) section 6): when a holdout is frozen, a held-out item's whole near-duplicate group goes into the holdout together |
| Owner direction applied | 2026-11-04: "We need near duplicate, and we can use Immich as a guide or other research papers"; a clean backend rewrite researched to best practice first; prototypes are local-venv GPU scripts (no Docker builds) on set P and public data; plan first, run nothing |
| Reader | an implementation or research agent with no memory of this session |

---

## 1. Plain summary

**What exists today.** Three unrelated near-duplicate paths, plus an exact-duplicate check:

| Path | What it groups | Space | Algorithm | Threshold today | Where |
|---|---|---|---|---|---|
| core `POST /ingest` (`/ingest/batch`, `/ingest/directory`) | whole images in the global index | MobileCLIP2-S2 512-d `global_embedding`, OpenSearch Faiss HNSW `cosinesimil` | online and greedy: kNN top 10, join the first neighbour's group, else make a group with the best match | **0.99 on the OpenSearch score scale**, which is `(1 + cos) / 2`, so **cosine 0.98** | `src/services/visual_search/ingest_single.py:272-360`, `src/services/duplicate_detection/` |
| curation export `dedup_threshold` | whole frames of one export | PE-Core-L14-336 1024-d `pe_embedding` | exact blocked `X Xᵀ` + union-find, computed per request | off unless the request sets it; UI default cosine 0.98 | `src/services/detection/frame_dedup.py:34,60-90,172` |
| curation `near_dup` scorer | crops of the residual pool only | PE `pe_embedding` on items | union-find **inside IVF buckets** (nearest centroid) | cosine 0.98 (`OP_CROP_DUP_THRESHOLD`, "a hypothesis, not a finding") | `src/services/curation/item_scores/near_dup.py` |
| exact duplicates (curation and core ingest) | byte-identical files | `imohash` (16 KiB samples of start, middle, end plus size) | lookup against the index | - | `src/services/curation/ingest_helpers.py:25-37` |

**What is wrong with it** (sections 2.3 and 2.4): the numbers live on two scales; the core path
misses most plain recompressions and resizes (34 and 13 of 100 at its default); groups are
order-dependent and racy; crop groups go stale after a re-score and reuse positional ids; **crop
groups never contain a validated or held-out item, so the holdout freeze cannot use them as the
program now plans (2D-4)**; the export recomputes groups per request instead of reading one stored
answer; `imohash` is a sampled hash, not a content identity; the curation images index maps a
`phash` field nothing writes.

**What to build in the v0.7.0 rebuild** (sections 5 and 7):

1. **One near-duplicate service with two levels**: `image` (whole frames) and `item` (crops),
   plus the core global index as a third, deployment-scoped level `core`. Item-level groups are
   **derived from image-level groups** (the same object across near-duplicate frames), which
   replaces the IVF bucket scoping.
2. **Evidence tiers as typed edges**: exact (SHA-256 of the bytes, enforced by a PostgreSQL unique
   constraint at ingest, so concurrent batches cannot store a file twice), embedding (cosine in a
   named space), and later perceptual (PDQ) and burst (EXIF time and camera) once the research
   shows they add recall at equal precision.
3. **Groups are built by a single-writer Celery job per project and level** (Immich serializes its
   duplicate job behind a database lock; we use the jobs table's exclusive group), with kNN
   candidates (k = 64, as Immich and DINOv2 use), a threshold, connected components and a group-size
   guard. Ingest only records "pending" rows; it never groups in the request.
4. **Runs are versioned in PostgreSQL** (runs, groups, members, edges, human decisions, pending
   rows); OpenSearch carries a search projection (`near_dup_*` fields) rewritten from the current
   run, so a re-score can never leave stale fields behind, and group ids are never reused.
5. **Two cuts per level**: a `dedup` cut (what review hides and export collapses) and a looser or
   equal `leak` cut (what the holdout pulls in). DINOv2 uses exactly this split: self-dedup at
   similarity 0.6, and relative dedup against evaluation sets at the lower threshold 0.45 (a looser match, so a stricter cleanup).
6. **Holdout rule** (the owner's decision): at freeze, every member of a held-out item's `leak`
   group goes into the holdout; after the freeze, every run moves new near-duplicates of held-out
   items into the holdout; a freeze on stale near-duplicate state refuses (fail closed).
7. **Thresholds are data**: a revisioned config kind `near_dup_profile` per project, seeded from
   code defaults that equal today's behaviour until the research note changes them.
   `OP_CROP_DUP_THRESHOLD` is retired.

**What to research** (section 6), with local-venv GPU scripts after the owner's go: the copy-level
threshold sweep (the old E-2), the tiers (SHA-256 vs `imohash`, PDQ and pHash, SSCD as a copy
specialist), burst and sequence grouping on California-ND, INRIA Holidays and set P, crop
derivation versus global crop search, holdout leakage, and throughput at high resolution.

---

## 2. Measured today (every number with its source)

### 2.1 Measurements

| What | Number | Source |
|---|---|---|
| Core grouping at score 0.99 (cosine 0.98), 100 originals per transform: grouped (grouped with the original) | JPEG q60 34 (24); half size 13 (9); 90 % centre crop 36 (21); brightness x1.15 96 (87) | `docs/research/retrieval_baseline.md`, "Near-duplicate grouping at ingest"; `docs/benchmarks/retrieval_baseline/dedupe_r1.json` |
| Same at score 0.95 (cosine 0.90) | 100 (55); 100 (61); 99 (75); 100 (87) | same |
| Same at score 0.90 (cosine 0.80) | 100 (52); 100 (52); 100 (67); 100 (71) | same |
| Median score of the matched pair at 0.99 (q60 / half / crop / brighter) | 0.9932 / 0.9920 / 0.9930 / 0.9962 | `dedupe_r1.json`, as quoted in the absorbed issue-416 plan section 2.5 |
| Smallest matched score at score 0.95 | q60 0.970 (cosine 0.94); half size 0.964 (cosine 0.93) | same |
| False groups among 2,000 distinct COCO originals at score 0.99 | 1 group (score 0.9924) | `retrieval_baseline.md` |
| Exact re-upload of 20 originals | 20 of 20 answered `duplicate` (the `imohash` path, not the threshold) | `retrieval_baseline.md` |
| Caveat on those variants | every variant was saved as JPEG q90, so "half size", "crop" and "brighter" were each also a recompression | `scripts/bench/retrieval_eval_data.py:77-97` |
| Crop scorer, production grouping over a 20,000-item sample, bucket-scoped: groups / items in groups at cosine 0.95, 0.96, 0.97, 0.98, 0.99 | 536 / 1,252; 326 / 698; 145 / 300; 48 / 99; 3 / 6 | `docs/research/curation_scores.md` section 4 |
| Same: class-coverage retention after collapsing groups | 100.00 % at every threshold; **pair precision never executed** (no human judge was available); verdict "stays shadow" | same |
| Clustering harness, near-duplicate stratum (jittered crops of one object, 512 IVF centroids over 3,961 crops) | pair recall 0.244, pair precision 0.896: three quarters of the pairs of crops of one object land in different clusters | `docs/research/cluster_eval_baseline.md` A.3 |
| Exact duplicates under concurrent batches (#358 fix): set P, 3,450 files = 3,000 contents + 450 byte copies, concurrency 8, 16, 32, 3 runs each | 3,000 stored in all 9 runs; before the fix a content was stored twice in 11 of 24 runs; 35.0 to 36.6 img/s at every concurrency | `docs/research/v070_measurement_log.md` E2 and "Bugs and suspected bugs" item 1 |
| PE space shape | mean random-pair cosine 0.50 on COCO; effective rank 104 to 122 of 1,024 (anisotropic: a cosine cut does not transfer between encoders, crops and frames, or datasets) | [09b_research_clustering.md](09b_research_clustering.md) section 1.3 item 4 |

Not measured anywhere: frame-path precision or recall, crop-path precision, any burst or sequence
grouping, any perceptual-hash tier, `imohash` collisions, the holdout leak rate, any throughput of a
near-duplicate step at high resolution.

### 2.2 Code facts (verified at `497315954`)

| Fact | Where |
|---|---|
| `DEFAULT_SIMILARITY_THRESHOLD = 0.99` with the comment "matches Immich's maxDistance=0.01" | `src/services/duplicate_detection/models.py:7-11` |
| The same 0.99 as a literal default in seven more places | `ingest_single.py:77,276`; `ingest_batch.py:81,93`; `src/routers/ingest/single.py:79`, `batch.py:76`, `directory.py:77` |
| Matches = hits with `_score >= threshold` in reply order | `src/services/duplicate_detection/queries.py:44-66` |
| Assignment: kNN `max_results=10`; join the first matching group, else create one with the best match | `ingest_single.py:300-355` |
| Any exception in the assignment is logged and swallowed (fail open: the image stays ungrouped silently) | `ingest_single.py:357-360` |
| Batch: bulk index with `refresh=False`, then every assignment at once through `asyncio.gather` | `ingest_batch.py:250,323-345` |
| Core index fields `duplicate_group_id`, `is_duplicate_primary`, `duplicate_score`, `imohash` | `src/clients/opensearch/indexes.py:138-145` |
| Crop scorer: `pool = 'residual'`, which excludes `class_validated`, every confident `class_source` and `class_excluded`; the job also excludes `test_holdout` | `near_dup.py:111`; `src/services/curation/item_scores/base.py:95-103`; `job.py:21-30` |
| Crop scorer ids `f'dup_{gi}'` (positional per run); writes only grouped items; plain bulk partial update | `near_dup.py:146-159`; `job.py:200-228` |
| Invalid `OP_CROP_DUP_THRESHOLD` silently falls back to the default (fail open) | `near_dup.py:53-57` |
| No IVF store means `RuntimeError` | `near_dup.py:123-132` |
| Review and search hide a crop when `dup_is_representative` is false | `src/services/curation/semantic_search.py:137-145` |
| Export dedup prefers a held-out frame as the survivor | `frame_dedup.py:104-143`, `:172-200` |
| The holdout cohort is `class_validated = true` and `class_source = 'human'` | `src/services/curation/holdout.py:231-252` |
| `phash` is mapped on the curation images index and written by nothing (`rg -n "'phash'" src scripts` finds only the mapping) | `src/clients/curation_opensearch/bodies_core.py:92` |
| `imohash` falls back to a SHA-256 prefix when the package is missing | `ingest_helpers.py:25-37` |
| OpenSearch image | `opensearchproject/opensearch:3.6.0` (`docker-compose.yml:1229`), which supports radial search |

### 2.3 Corrections to earlier premises

| # | Premise | What is true | Consequence |
|---|---|---|---|
| C1 | "default 0.99" read as a cosine (#416 title) | core compares an OpenSearch `cosinesimil` score, `(1 + cos) / 2` ([OpenSearch spaces](https://docs.opensearch.org/latest/mappings/supported-field-types/knn-spaces/)); 0.99 is cosine 0.98, the same cut the curation paths use, in another space | every threshold in this plan is a cosine in a named space; the core score is derived |
| C2 | "matches Immich's maxDistance=0.01" | Immich's `maxDistance` is a pgvector `<=>` cosine **distance** (0.01 = cosine 0.99) on ViT-B-32 OpenAI CLIP (section 3.1), so no number transfers | the comment goes at landing |
| C3 | the baseline's "half size" and "crop" rows measure resize and framing | each was also a JPEG q90 re-encode | factor-isolated transforms (6.3) |
| C4 | one threshold | three spaces, two scales, three algorithms | one profile per level |
| C5 | "duplicate groups are must-links, one vote per group in gates, and the train/test split guard" | none of these is wired: no `dup_` read in `assist_loop/`, no duplicate logic in the freeze | section 5.4 builds the guard |
| C6 | 03b 4A-6 control "unset gives 0.97" | the default is 0.98 | moot once the variable is retired (11) |
| C7 | the 2,000 "distinct" originals are true negatives | COCO and Open Images contain their own near-duplicates | person-check distinct pairs above the lowest swept cut (6.5) |
| **C8** | 2D-4: "the freeze checks near-duplicates of the held-out items (the items' near-duplicate group fields)" | the holdout cohort is human-validated, and the crop scorer never scores validated or held-out items (2.2), so **a held-out item never carries a group field**; a red test written against today's fields cannot turn green by reading them | 2D-4 moves to the near-duplicate lane, which groups every item (section 11) |
| **C9** | `near_dup.py` docstring: "already validated at threshold 0.98 for whole-frame" | no validation record exists; `curation_scores.md` says precision was never executed | the defaults are hypotheses until 6.2 P1 |
| **C10** | "near-duplicate photos of the same subject already land in the same or adjacent IVF bucket" (`near_dup.py:24-27`) | the clustering harness found 76 % of near-duplicate crop pairs split across clusters at about 8 crops per cluster (2.1) | the bucket scope is dropped (D3 in section 4) |
| **C11** | `imohash` is the exact-duplicate key | it hashes 16 KiB samples plus the size; its README says not to use it for "file verification" or where "fixed-size files are manipulated" ([py-imohash](https://github.com/kalafut/py-imohash)) | SHA-256 of the bytes becomes the key (J6) |

---

## 3. Literature and industry review

Every claim below carries its source; "verified" means read in the source on 2026-11-04.

### 3.1 Immich (the owner's named guide)

Read in the Immich repository (`server/src/services/duplicate.service.ts`,
`server/src/repositories/duplicate.repository.ts`, `server/src/dtos/system-config.dto.ts`, last change to
the service on 2026-10-28; <https://github.com/immich-app/immich>):

- **Exact duplicates** are a SHA-1 checksum of the file (`crypto.repository.ts` `hashFile`), separate
  from near-duplicates.
- **Near-duplicates** use the CLIP smart-search embedding (default model `ViT-B-32__openai`).
  Per asset: a kNN query ordered by `smart_search.embedding <=> :embedding` (cosine distance),
  `LIMIT 64` (`DUPLICATE_SEARCH_LIMIT`), then `distance <= maxDistance`. Default
  `duplicateDetection.maxDistance` is **0.01**; the admin bound is 0.001 to 0.1.
- **Grouping is transitive**: the asset joins its own group or the first neighbour's group, and every
  other neighbour group is **merged** into it (`updateDuplicates` then `merge` over `sourceIds`).
  That is incremental connected components with k = 64.
- **Single writer**: the whole per-asset job runs inside
  `databaseRepository.withLock(DatabaseLock.DuplicateDetection, ...)`, so two assets are never grouped
  concurrently. This is the pattern that removes our core batch race (#473 part b).
- **Staleness**: a per-asset `duplicatesDetectedAt` job-status column; the queue-all job selects
  assets where it is null, or all assets with `force`. A re-run that finds no neighbour clears the
  asset's `duplicateId`. Singleton groups are cleaned on read.
- **Exclusions**: stacked (burst-stacked), hidden and locked assets are skipped. Immich stacks bursts
  by metadata separately from duplicates.
- **Resolution** is a human step: keep and trash lists per group; the suggested keeper is the largest
  file, then the most EXIF fields (`server/src/utils/duplicate.ts`).
- Community discussion asks for a stricter review threshold after detection
  (<https://github.com/immich-app/immich/discussions/25831>); users report lowering `maxDistance` to
  0.001 to keep only true duplicates. That is the dedup-versus-review split of section 5.

What we take: exact key separate from near-duplicates; k = 64 kNN plus a threshold; transitive merge;
a single-writer job; a per-member "detected at" staleness marker; human resolution that automated
runs respect. What we do not take: one global threshold for every library, and the CLIP space (our
spaces are PE and MobileCLIP; numbers do not transfer, C2).

### 3.2 Perceptual hashing

- **ImageHash** (BSD-2-Clause, 4.3.2 on PyPI): average, perceptual (pHash), difference (dHash),
  wavelet, HSV colour and crop-resistant hashes; subtraction gives the Hamming distance
  (<https://github.com/JohannesBuchner/imagehash>). The README makes no robustness claim per method.
- **PDQ** (Meta, ThreatExchange): a DCT hash "heavily inspired by pHash" with Jarosz-filter
  downsampling to 64 x 64, a 16 x 16 DCT and a median threshold, giving **256 bits** and a **quality**
  score. The reference README recommends a match threshold of Hamming distance **<= 31** and
  discarding hashes with quality **<= 49**; it does not guarantee rotation invariance
  (<https://github.com/facebook/ThreatExchange/blob/main/pdq/README.md>). Unrelated images sit near
  128 bits apart, and all eight dihedral variants come from one DCT (third-party implementation docs,
  <https://lib.rs/crates/pdq-rs>). Python binding: `pdqhash` 0.2.8, MIT
  (<https://github.com/faustomorales/pdqhash-python>).
- **Limits** shown in the literature: hashes are brittle to crops and rotations and can be evaded
  adversarially ([Jain et al., USENIX Security 2022](https://www.usenix.org/system/files/sec22summer_jain.pdf);
  [arXiv 2406.00918](https://arxiv.org/pdf/2406.00918)). Adversarial evasion does not matter for
  curation; crops and rotations do.
- **What a hash is good for here**: it is model-independent (survives an encoder swap, #347, #365),
  near-zero false positives at small distances, and cheap once a small luminance tile exists. It is
  weak on crops, so it is never the only tier.

### 3.3 Copy detection: DISC21 and SSCD

- **DISC21** (Douze et al., "The 2021 Image Similarity Dataset and Challenge",
  [arXiv 2106.09672](https://arxiv.org/abs/2106.09672)): a 1M-image reference set, a training set and
  query sets of transformed copies (automatic, hand-crafted and learned edits), metric micro-AP; the
  NeurIPS'21 challenge results are in [arXiv 2202.04007](https://arxiv.org/pdf/2202.04007). The
  baseline code is CC BY-NC 4.0 (<https://github.com/facebookresearch/isc2021>); the dataset terms
  are on the DISC21 page and must be read before download (J11).
- **SSCD** (Pizzi et al., CVPR 2022, [arXiv 2202.10261](https://arxiv.org/abs/2202.10261)): SimCLR
  with an entropy regulariser, mixup-aware InfoNCE and inference-time score normalisation; ResNet-50
  512-d (and ResNeXt-101 1024-d). Code MIT; TorchScript models; the README recommends
  `sscd_disc_mixup`, inputs with the short side at 288 px (or a 320 x 320 square), and states that
  for that model **cosine > 0.75 marks a copy at about 90 % precision on DISC**
  (<https://github.com/facebookresearch/sscd-copy-detection>). The weights' licence is not stated in
  the README (J11).
- **DINOv2's data curation** (Oquab et al., [arXiv 2304.07193](https://arxiv.org/abs/2304.07193),
  appendix A.3, verified): SSCD embeddings; **k = 64** nearest neighbours by cosine; edges with
  similarity **> 0.6**; connected components; one representative kept per component
  (self-deduplication). Then **relative deduplication** against evaluation sets at the lower threshold
  **> 0.45** (a looser match, a stricter cleanup), discarding the whole matching component, so that evaluations stay clean. This is the
  holdout-leak pattern: the leak cut is looser than the dedup cut, and the whole component moves.

### 3.4 Embedding deduplication at scale

- **SemDeDup** (Abbas et al., [arXiv 2303.09540](https://arxiv.org/abs/2303.09540), verified):
  k-means over CLIP embeddings (k = 50,000 for LAION), pairwise cosine inside each cluster, pairs at
  or above `1 - ε` are duplicates, the member least similar to the centroid is kept. This is exactly
  today's IVF-bucketed crop scorer, and it shares its weakness: pairs that straddle clusters are
  never compared. SemDeDup accepts that for a training-set filter; a holdout guard cannot.
- **FiftyOne Brain** (Voxel51; <https://docs.voxel51.com/brain/index.html>, source
  <https://github.com/voxel51/fiftyone-brain>): `compute_exact_duplicates` groups by file hash;
  `compute_near_duplicates` defaults to `resnet18-imagenet-torch` embeddings, cosine distance
  (`metric="cosine"`), `threshold=0.2`, and keeps the first of each radius neighbourhood in index order
  (`_remove_duplicates_thresh`), a leader rule that does not chain but depends on order. It supports
  patches through `roi_field` (crop-level), and **`compute_leaky_splits`** finds near-duplicates across
  train and test splits (default threshold 0.2, "values in [0.1, 0.25] work well").
- **NVIDIA NeMo Curator** runs CLIP embedding plus semantic dedup for image curation (already noted in
  `docs/research/v060_serving_backend_trade_study.md`).

### 3.5 Near-duplicates in personal photo collections (bursts and sequences)

- **California-ND** (Jinda-Apiraksa, Vonikakis, Winkler, QoMEX 2013): 701 photos from one real
  holiday collection, 1024 x 768, original sequence kept, mostly consecutive and burst shots; 10
  annotators (including the photographer) each marked near-duplicates, shipped as ten 701 x 701
  matrices (<https://publish.illinois.edu/adsc-vintage/california-nd/>). A later benchmark counts
  4,609 pairs marked by at least one rater, raters disagreeing on 82 % of them, and 107 clusters of
  6.55 images on average ([Morra and Lamberti, arXiv 1907.02821](https://arxiv.org/pdf/1907.02821)).
  Creative Commons licence; the zip password is given by email (J11). This is the closest public
  stand-in for set P's bursts, and the rater disagreement says a burst threshold has no single truth.
- **INRIA Holidays** (Jégou, Douze, Schmid, ECCV 2008): 1,491 personal holiday photos in 500 groups of
  one scene or object, including deliberate rotation, viewpoint, illumination and blur changes;
  INRIA holds the copyright (<https://thoth.inrialpes.fr/~jegou/data.php.html>). **Copydays** is the
  matching copy-detection set from the same page (crops, JPEG and "strong" edits of personal photos).

### 3.6 Search and grouping building blocks

- **OpenSearch radial search**: a `knn` query takes exactly one of `k`, `max_distance` or `min_score`;
  Faiss and Lucene engines; filters and nested fields supported
  (<https://docs.opensearch.org/latest/vector-search/specialized-operations/radial-search-knn/>). With
  `cosinesimil`, `min_score = (1 + cos) / 2`. It returns every point above the cut, so a dense burst
  returns many hits: the plan uses `k = 64` plus a score cut for online candidates, and radial search
  only where a full neighbourhood is required (the leak closure check, 5.4).
- **FAISS**: exact `IndexFlatIP` on unit vectors; `range_search` on CPU indexes
  (<https://github.com/facebookresearch/faiss/wiki/Special-operations-on-indexes>); GPU flat kNN for
  batch builds (`faiss-gpu-cu12` is already in the API image).
- **Group-aware splits**: scikit-learn's grouped cross-validation iterators keep a group on one side
  of a split (<https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data>).
  The holdout closure is the same rule applied to a frozen test set.
- **Complete linkage with a distance cut**: scikit-learn `AgglomerativeClustering(metric='cosine',
  linkage='complete', distance_threshold=1 - t, n_clusters=None)`
  (<https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html>),
  the chaining guard inside an oversized component.
- **Wilson intervals**: statsmodels `proportion_confint(method='wilson')`
  (<https://www.statsmodels.org/stable/generated/statsmodels.stats.proportion.proportion_confint.html>).

### 3.7 What the review settles and what it leaves open

| Settled by the sources | Left for the prototypes |
|---|---|
| exact key separate from near-duplicates (Immich, FiftyOne) | whether SHA-256 costs anything measurable next to decode |
| kNN with k = 64, a threshold and connected components (Immich, DINOv2) | the cosine cuts in PE and MobileCLIP space, per level |
| one writer per scope (Immich's lock) | the incremental rescore cost at 100k and 1M |
| a looser cut and whole-component moves for leakage (DINOv2 relative dedup, FiftyOne leaky splits) | how much the leak cut inflates the holdout |
| bucketed dedup misses cross-bucket pairs (SemDeDup design, our C10) | how many item pairs the image-group derivation misses |
| hashes are brittle to crops and rotation | whether PDQ adds recall at equal precision on top of embeddings |
| copy-specialised descriptors exist (SSCD, cosine > 0.75 about 90 % precision on DISC) | whether SSCD beats PE enough to justify another model |
| burst near-duplicates are subjective (California-ND, 82 % rater disagreement) | the burst cut, and whether EXIF time helps |

---

## 4. Candidate designs

| # | Design | Verdict | Why |
|---|---|---|---|
| D1 | **Hash pre-filter plus embedding confirmation** (PDQ Hamming <= t gives candidates; an embedding cosine confirms) | 🟡 as a pre-filter ❌, as an extra evidence tier 🟡 | ANN kNN candidates are already cheap, so a pre-filter saves little and loses every crop and rotation pair the hash misses. As a second tier (an edge of kind `pdq` that joins two images the embedding scored just below the cut) it may add copy-level recall and survive encoder swaps. P2 decides; not in v0.7.0 unless P2 runs first |
| D2 | **Embedding kNN (k = 64) + cosine cut + connected components** | ✅ the base | Immich and DINOv2; deterministic given the vectors; incremental |
| D3 | IVF-bucketed exact grouping (today's crop scorer, SemDeDup) | ❌ | cross-bucket pairs are never found (C10); needs a centroid store; scores only the residual pool (C8) |
| D4 | Exact all-pairs blocked `X Xᵀ` (today's frame dedup) | 🟡 research oracle only | exact and simple; O(n²); in the product it would be a second grouping implementation next to D2. The prototypes keep it as ground truth for the ANN loss |
| D5 | Online greedy assignment in the ingest request (today's core path) | ❌ | order-dependent, racy across batches and replicas, never splits, swallows errors |
| D6 | **Single-writer job per project and level, fed by pending rows** (Immich's lock pattern on our jobs table) | ✅ | removes the race by construction; ingest stays fast; staleness is a row count |
| D7a | **Crop level derived from image-level groups**: compare items only across images in one image-level `leak` group, same detector class, never two items of one image, cosine cut on the crop vectors | ✅ default | the near-duplicate of a crop is the same object in a near-duplicate frame; cost is the sum over image groups of (images x items per image)², tiny; it covers validated and held-out items too |
| D7b | Crop level by global item kNN | 🟡 arm in P4 | catches the zoomed burst frame whose whole-frame vector fell below the cut; but it also matches two different cars of one model (identity, not duplication); P4 measures what D7a misses |
| D7c | Crop level independent per bucket (today) | ❌ | D3 |
| D8a | **Per-project thresholds as a revisioned config kind** (`near_dup_profile`, seeded from code defaults) | ✅ | the project's rule that domain behaviour is data, revisioned and revertible, like region profiles; an anisotropic space means a medical set and a street set will not share a cut |
| D8b | Thresholds in an env var (`OP_CROP_DUP_THRESHOLD`) | ❌ | deployment-wide, unrevisioned, fail-open today |
| D8c | Threshold per request (export `dedup_threshold`) | ❌ | results not reproducible from stored state; a second grouping implementation |
| D9 | Linkage: single linkage + a group-size guard; complete linkage inside an oversized component | ✅ single + guard; 🟡 complete as P1's arm | single linkage is what Immich and DINOv2 run; chaining shows first in the largest group, which the guard catches |
| D10 | A copy-specialised encoder (SSCD) as a third vector | 🟡 | best documented copy detector; but another Triton model, another decode-sized input (288 px), another vector field. Adopt only if P2 shows a large gain (stop rule 6.9) |
| D11 | Burst grouping from EXIF capture time and camera | 🟡 candidate signal only | must also pass an embedding floor (the rule in [cluster_search_makemodel_plan.md](../cluster_search_makemodel_plan.md): "no single attribute decides a merge"); P3 |
| D12 | Groups only as OpenSearch fields (today) | ❌ | no run, version, owner or history; stale fields; positional ids |
| D13 | **Groups in PostgreSQL runs, OpenSearch as a projection** | ✅ | the control-plane rule: anything with a lifecycle goes to PostgreSQL ([02_control_plane_database.md](02_control_plane_database.md) section 3.1); the filter fields stay searchable |
| D14 | **Two cuts per level (`dedup`, `leak`)** | ✅ | DINOv2's self versus relative dedup; FiftyOne's leaky splits |
| D15 | Exact key: SHA-256 with a PostgreSQL unique constraint | ✅ (J6) | race-free across workers and replicas, which an index lookup is not (#358); a real content identity (C11) |

---

## 5. Seams the v0.7.0 rebuild must include

**Owning plan file: this file.** Landing lanes are in section 8 (ND-A, ND-B in W2.3; ND-C in W4).
The tables follow [02a_postgres_schema.md](02a_postgres_schema.md) section 1: plural snake_case,
`uuidv7()` keys, `timestamptz`, closed sets as `text` + `CHECK`, `jsonb` only for pydantic-validated
bodies, row-level security on every `project_id` table, `NULL` project = deployment scope (allowed
only under `deployment` scope, as for `jobs`). One Alembic revision `near_duplicates`, owned by ND-A.

### 5.1 PostgreSQL tables

```sql
-- Exact duplicates: one row per stored content. The insert is the duplicate check.
CREATE TABLE image_contents (
  id              bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  project_id      uuid NULL REFERENCES projects(id) ON DELETE CASCADE,   -- NULL: the core global index
  content_sha256  text NOT NULL CHECK (content_sha256 ~ '^[0-9a-f]{64}$'),
  image_id        text NOT NULL,
  size_bytes      bigint NOT NULL CHECK (size_bytes >= 0),
  created_at      timestamptz NOT NULL DEFAULT now(),
  UNIQUE NULLS NOT DISTINCT (project_id, content_sha256),
  UNIQUE NULLS NOT DISTINCT (project_id, image_id)
);

-- One grouping run of one level. Exactly one 'current' run per (project, level).
CREATE TABLE near_dup_runs (
  id                 uuid PRIMARY KEY DEFAULT uuidv7(),
  project_id         uuid NULL REFERENCES projects(id) ON DELETE CASCADE,
  level              text NOT NULL CHECK (level IN ('image','item','core')),
  mode               text NOT NULL CHECK (mode IN ('full','incremental')),
  status             text NOT NULL CHECK (status IN ('building','current','superseded','failed')),
  job_id             uuid NULL REFERENCES jobs(id) ON DELETE SET NULL,
  parent_run_id      uuid NULL REFERENCES near_dup_runs(id) ON DELETE SET NULL,
  profile_config_id  uuid NULL,
  profile_revision   integer NULL,
  params             jsonb NOT NULL,           -- NearDupRunParams: cuts, k, linkage, max_group_size, scope
  space              text NOT NULL,            -- '<encoder>@<engine sha256[:12]>:<field>'
  algorithm_version  text NOT NULL,            -- src/domain/near_dup.py ALGORITHM_VERSION, 'nd-1'
  members_scored     integer NOT NULL DEFAULT 0,
  dedup_groups       integer NOT NULL DEFAULT 0,
  leak_groups        integer NOT NULL DEFAULT 0,
  grouped_members    integer NOT NULL DEFAULT 0,
  largest_group      integer NOT NULL DEFAULT 0,
  oversized_split    integer NOT NULL DEFAULT 0, -- components split by the size guard
  holdout_absorbed   integer NOT NULL DEFAULT 0,
  started_at         timestamptz NOT NULL DEFAULT now(),
  finished_at        timestamptz NULL,
  FOREIGN KEY (profile_config_id, profile_revision)
    REFERENCES config_revisions(config_id, revision) ON DELETE RESTRICT,
  CHECK ((level = 'core') = (project_id IS NULL)),
  CHECK ((level = 'core') = (profile_config_id IS NULL)),
  CHECK ((status = 'building') = (finished_at IS NULL))
);
CREATE UNIQUE INDEX uq_near_dup_runs_current  ON near_dup_runs (project_id, level) NULLS NOT DISTINCT WHERE status = 'current';
CREATE UNIQUE INDEX uq_near_dup_runs_building ON near_dup_runs (project_id, level) NULLS NOT DISTINCT WHERE status = 'building';
CREATE INDEX ix_near_dup_runs_recent ON near_dup_runs (project_id, level, started_at DESC);

CREATE TABLE near_dup_groups (
  id                 uuid PRIMARY KEY DEFAULT uuidv7(),   -- the group id on the wire and in OpenSearch; never reused
  run_id             uuid NOT NULL REFERENCES near_dup_runs(id) ON DELETE CASCADE,
  purpose            text NOT NULL CHECK (purpose IN ('dedup','leak')),
  lineage_id         uuid NOT NULL,     -- carried from the parent run's group with member Jaccard >= 0.5, else = id
  representative_id  text NOT NULL,
  member_count       integer NOT NULL CHECK (member_count >= 2),
  min_edge_cosine    real NOT NULL,
  min_cosine_to_rep  real NOT NULL,
  has_holdout        boolean NOT NULL DEFAULT false
);
CREATE INDEX ix_near_dup_groups_run     ON near_dup_groups (run_id, purpose, member_count DESC);
CREATE INDEX ix_near_dup_groups_lineage ON near_dup_groups (lineage_id);

CREATE TABLE near_dup_members (
  run_id             uuid NOT NULL REFERENCES near_dup_runs(id) ON DELETE CASCADE,
  purpose            text NOT NULL CHECK (purpose IN ('dedup','leak')),
  member_id          text NOT NULL,     -- image_id (levels image, core) or item_id (level item)
  group_id           uuid NOT NULL REFERENCES near_dup_groups(id) ON DELETE CASCADE,
  is_representative  boolean NOT NULL,
  cosine_to_rep      real NOT NULL,
  PRIMARY KEY (run_id, purpose, member_id)
);
CREATE INDEX ix_near_dup_members_group ON near_dup_members (group_id);
CREATE UNIQUE INDEX uq_near_dup_members_rep ON near_dup_members (group_id) WHERE is_representative;

-- Current candidate edges at or above the profile's lowest cut, one space at a time.
CREATE TABLE near_dup_edges (
  id           bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  project_id   uuid NULL REFERENCES projects(id) ON DELETE CASCADE,
  level        text NOT NULL CHECK (level IN ('image','item','core')),
  space        text NOT NULL,
  a_id         text NOT NULL,
  b_id         text NOT NULL,
  kind         text NOT NULL CHECK (kind IN ('embedding')),  -- 'pdq', 'burst' added by the lanes that land them
  score        real NOT NULL,
  computed_at  timestamptz NOT NULL DEFAULT now(),
  CHECK (a_id < b_id),
  UNIQUE NULLS NOT DISTINCT (project_id, level, space, kind, a_id, b_id)
);
CREATE INDEX ix_near_dup_edges_b ON near_dup_edges (project_id, level, space, b_id);

-- Human must-link / cannot-link. Automated runs apply them and never override them (lock rule).
CREATE TABLE near_dup_decisions (
  id                  uuid PRIMARY KEY DEFAULT uuidv7(),
  project_id          uuid NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  level               text NOT NULL CHECK (level IN ('image','item')),
  a_id                text NOT NULL,
  b_id                text NOT NULL,
  decision            text NOT NULL CHECK (decision IN ('same','different')),
  decided_by          text NOT NULL,
  decided_by_user_id  uuid NULL,
  decided_at          timestamptz NOT NULL DEFAULT now(),
  CHECK (a_id < b_id),
  UNIQUE (project_id, level, a_id, b_id)
);

-- Staleness: a member whose vector, boxes or existence changed since the current run.
CREATE TABLE near_dup_pending (
  id                 bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  project_id         uuid NULL REFERENCES projects(id) ON DELETE CASCADE,
  level              text NOT NULL CHECK (level IN ('image','item','core')),
  member_id          text NOT NULL,
  reason             text NOT NULL CHECK (reason IN ('added','vector_changed','boxes_changed','deleted')),
  enqueued_at        timestamptz NOT NULL DEFAULT now(),
  claimed_by_job_id  uuid NULL REFERENCES jobs(id) ON DELETE SET NULL,
  UNIQUE NULLS NOT DISTINCT (project_id, level, member_id)
);
CREATE INDEX ix_near_dup_pending_age ON near_dup_pending (project_id, level, enqueued_at);

-- Every item the near-duplicate rule moved into the holdout, and why.
CREATE TABLE holdout_near_dup_moves (
  id              bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  project_id      uuid NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  item_id         text NOT NULL,
  anchor_item_id  text NOT NULL,       -- the held-out item whose leak group pulled it in
  group_id        uuid NULL,           -- no FK: runs are pruned, the audit row stays
  freeze_id       uuid NULL REFERENCES holdout_freezes(id) ON DELETE CASCADE,   -- moved at a freeze
  run_id          uuid NULL REFERENCES near_dup_runs(id) ON DELETE SET NULL,    -- absorbed by a later run
  moved_at        timestamptz NOT NULL DEFAULT now(),
  UNIQUE (project_id, item_id)
);

ALTER TABLE holdout_freezes
  ADD COLUMN near_dup_image_run_id uuid NULL REFERENCES near_dup_runs(id) ON DELETE SET NULL,
  ADD COLUMN near_dup_item_run_id  uuid NULL REFERENCES near_dup_runs(id) ON DELETE SET NULL,
  ADD COLUMN moved_by_near_dup     integer NOT NULL DEFAULT 0;
```

Rules:

- **RLS**: `image_contents`, `near_dup_runs`, `near_dup_edges`, `near_dup_pending`,
  `near_dup_decisions`, `holdout_near_dup_moves` carry the policy of 02a section 1.4 (NULL project
  only in `deployment` scope); `near_dup_groups` and `near_dup_members` use the `EXISTS` policy through
  `near_dup_runs`. Tests extend `tests/db/test_row_level_security.py`.
- **Current flip** in one transaction: `UPDATE near_dup_runs SET status = 'superseded' ... WHERE
  status = 'current'`, then `SET status = 'current'` on the new run, insert the
  `search_projection_tasks` row (kind `near_dup_run`, payload `{run_id, level}`), and the `events` row
  (`near_dup.run`). The partial unique indexes make two current runs impossible.
- **Exact check at ingest**: `INSERT INTO image_contents ... ON CONFLICT DO NOTHING RETURNING id`; no
  row means the content exists, and the existing `duplicate` answer is returned. The image delete
  path deletes the row in the same code path that deletes the document. This replaces the
  `imohash` lookup, so the concurrency fix of #358 becomes a constraint instead of a batch lookup.
- **Pending rows are written before the OpenSearch write** they describe (an extra pending row is
  harmless: the rescore is idempotent per member; a missing one is a silent stale group). Writers:
  curation ingest (`added`), reprocess and re-embed (`vector_changed`), the region stage when it
  creates or deletes items or moves boxes (`boxes_changed`), image and item delete (`deleted`),
  dataset import and combine (`added`), core ingest (`added`, project NULL). A profile activation or
  an encoder-space change does not write rows: the status marks the level `rebuild_required`.
- **Edges** below the profile's `leak` cut are not stored; an edge whose member is deleted is deleted
  by the rescore that drains the `deleted` row.
- **Retention**: keep the current run and the two newest superseded runs per (project, level); a beat
  task prunes older runs (cascade deletes their groups and members). Row for 02a section 9.

### 5.2 Celery job kinds (rows for [02b_jobs_locking_events.md](02b_jobs_locking_events.md) section 2.1)

Module `src/config/job_kinds/near_dup.py`; tasks in `src/tasks/near_dup.py`; handlers in
`src/services/near_duplicates/`. Every kind shares the exclusive group `near_dup` with scope
`<level>`, so one writer per project and level (the Immich lock, as the `uq_jobs_one_active` index).

| Kind | Project | Queue | Exclusive group, scope | Auto-resume | Soft / hard limit | Max queue wait | What it does |
|---|---|---|---|---|---|---|---|
| `near_dup_score` | yes | `jobs` | `near_dup`, level | yes (idempotent per member) | 3 h / 3 h 15 m | 24 h | candidate edges for the given members (all, or the claimed pending rows): image level by exact FAISS GPU kNN (k = 64) over the project's PIT-sliced vectors for a full build, by OpenSearch kNN (k = 64, project index, `min_score` of the leak cut) for an incremental one; item level by D7a inside image `leak` groups; upserts `near_dup_edges` |
| `near_dup_group` | yes | `jobs` | `near_dup`, level | no (a re-run rebuilds from edges) | 3 h / 3 h 15 m | 24 h | connected components over edges at the `dedup` and `leak` cuts, decisions applied (`different` removes the edge, `same` adds one), size guard and complete-linkage split, representative = most central (`frame_dedup.most_central` moves to `src/domain/near_dup.py`), lineage by Jaccard, writes the run, flips current, then runs the holdout absorption (5.4) |
| `near_dup_rescore` | yes | `jobs` | `near_dup`, level | yes | 3 h / 3 h 15 m | 1 h | the incremental path: claim pending rows (`SKIP LOCKED`), score them, recompute only the components they touch (members of their current groups plus their new neighbours), write an `incremental` run that copies untouched groups; started by a beat tick per (project, level) when pending rows exist, idempotency key `near_dup_rescore:<project>:<level>:<slot>` |
| `near_dup_projection_sync` | yes | `jobs` | `near_dup_projection`, level | yes | 3 h / 3 h 15 m | 24 h | applies the current run to OpenSearch (5.3); created in the flip transaction and by the reaper for an orphaned `search_projection_tasks` row (the `class_projection_sync` pattern) |
| `near_dup_core_group` | NULL | `jobs` | `near_dup_core`, '' | yes | 3 h / 3 h 15 m | 1 h | the core global index: drains `core` pending rows, kNN on `global_embedding`, components, writes the `core` run and the core fields directly (deployment scope, `deployment_tx` ratchet entry) |

The full build is `near_dup_score` then `near_dup_group` chained by the task (one `jobs` row per
step, `parent_job_id` set). `item` level always runs after the `image` level of the same project
(the item scope reads image `leak` groups): the `image` group step enqueues the `item` rescore.

### 5.3 OpenSearch fields (search projection; the same names on both curation indexes)

| Field | Type | On | Meaning |
|---|---|---|---|
| `near_dup_group_id` | keyword | images, items | `dedup` group of the current run; absent = not grouped |
| `near_dup_leak_group_id` | keyword | images, items | `leak` group of the current run |
| `near_dup_is_representative` | boolean | images, items | true for the `dedup` representative |
| `near_dup_group_size` | integer | images, items | `dedup` member count |
| `near_dup_run_id` | keyword | images, items | the run that wrote the fields |
| `content_sha256` | keyword | images | the exact key (also in `image_contents`) |

Writes: bulk-set the members' fields with the new `near_dup_run_id`, then `update_by_query` removes
the five group fields from every document whose `near_dup_run_id` exists and differs from the current
run. A document between the two steps shows a stale group for seconds; `hide_near_duplicates` keeps
documents whose `near_dup_is_representative` is true or missing, so a stale non-representative is
the only exposure, and the second step ends it. Writes go through the guarded OCC writers for the
lock rule (`src/clients/occ_locks.py`): the near-duplicate fields are machine fields and never touch
a validated class or box.

Deleted with no replacement (no shims): items `dup_group_id`, `dup_group_size`,
`dup_is_representative`, `dup_threshold`, `dup_method`, `dup_scored_at`
(`bodies_core.py:408-413`, `ensure_overlay_fields.py:239-257`); images `phash` (no writer) and
`imohash` (J6). The core global index renames `duplicate_group_id`, `is_duplicate_primary`,
`duplicate_score` to `near_dup_group_id`, `near_dup_is_representative`, `near_dup_cosine_to_rep`
(J15); the `/query/duplicates` wire shapes stay. Existing 0.6.0 indexes keep the old mappings
(mappings cannot be dropped without a reindex); the 0.7.0 upgrade step clears their values with one
`update_by_query` per index, and a full `near_dup` build runs after the upgrade.

Later lanes add, each with its edge kind: `pdq_hash` (keyword, 64 hex) and `pdq_quality` (short) on
images (P2 pass, J9); `capture_time` (date) and `burst_key` (keyword) belong to the typed `signals`
object of [cluster_search_makemodel_plan.md](../cluster_search_makemodel_plan.md) and are not added here.

### 5.4 The holdout-freeze rule (MQ-5, applied)

1. **Precondition (fail closed)**: the freeze reads the `image` and `item` status. It refuses with 409
   `near_dup_stale` when either level has no current run, is `rebuild_required`, or has pending rows
   for the project (J4). The response names the level and the pending count, and the status route
   tells Cropwright which job to start.
2. **Closure at freeze**: `H0` = the selected cohort (unchanged `select_test_holdout`). `H` = `H0` plus
   every member of every current item-level **`leak`** group that contains an item of `H0`
   (transitive by construction: a group is a component). Items of the anchor's image that are other
   objects are not moved (today's per-crop holdout is unchanged; whole-frame closure is J2).
3. **Write**: `test_holdout = true` on `H` through the OCC writers; one `holdout_near_dup_moves` row per
   moved item with its anchor and group; `holdout_freezes.moved_by_near_dup`, `near_dup_image_run_id`,
   `near_dup_item_run_id` set; all in the freeze's existing fail-closed path (a database error refuses
   the freeze).
4. **After the freeze (absorption)**: every `near_dup_group` run at item level, before it flips
   current, finds `leak` groups that mix held-out and non-held-out items and moves the rest into the
   holdout the same way (`run_id` set on the move row, `near_dup_runs.holdout_absorbed` counted, an
   `events` row `holdout.near_dup_absorbed`). J5 asks whether to move or only to exclude from training.
5. **Training export gate**: an export or training cohort build refuses with 409 `near_dup_stale` while
   the project has a frozen holdout and pending near-duplicate rows (J4).
6. **Evaluation**: moved items that are not human-validated stay out of training and carry no label for
   evaluation; the bake-off and test metrics read only validated held-out items, as today.

Test (replaces 2D-4): `tests/near_duplicates/test_holdout_near_duplicates.py` (`db` + fake
OpenSearch): a validated item whose near-duplicate crop is in the cohort's leak group ends on the
holdout side and is counted; red today (C8). Controls: an item with no near-duplicate is unchanged;
a stale project refuses.

### 5.5 Config kind `near_dup_profile`

- `config_documents.kind` CHECK gains `'near_dup_profile'`; `config_activations.axis` gains
  `'near_dup'` with the pair `('near_dup','near_dup_profile')` (02a section 4).
- Body model `NearDupProfileBody` in `src/schemas/curation/near_duplicates.py`, validated on write and
  read:

```yaml
version: 1
image:
  field: pe_embedding          # images index vector
  dedup_cosine: 0.98           # today's frame value; hypothesis until P1
  leak_cosine: 0.98            # equal until P5 (J3)
  k: 64                        # Immich DUPLICATE_SEARCH_LIMIT, DINOv2 A.3
  max_group_size: 1000         # guard; above it the component is split by complete linkage
item:
  field: pe_embedding          # items index vector
  scope: image_leak_groups     # D7a
  same_detector_class: true
  dedup_cosine: 0.98           # today's crop value; hypothesis until P4
  leak_cosine: 0.98
  max_group_size: 1000
```

- Validation: `0.5 <= leak_cosine <= dedup_cosine <= 1.0`; `8 <= k <= 256`; `2 <= max_group_size`.
- Code defaults in `src/domain/near_dup.py` (`DEFAULT_IMAGE_DEDUP_COSINE`,
  `DEFAULT_ITEM_DEDUP_COSINE`, `DEFAULT_CORE_DEDUP_COSINE = 0.98` with
  `core_score(cos) = (1 + cos) / 2` for the core path, `DEFAULT_K = 64`, `ALGORITHM_VERSION`); project
  creation seeds revision 1 from them and activates it. The core path has no project and reads the
  code default only.
- A profile activation marks both levels `rebuild_required`; nothing regroups silently.

### 5.6 API and wire (proposed routes: none of these exist yet)

Curation (no `/v1` twin), router `src/routers/curation/near_duplicates.py`, wire models in
`src/schemas/curation/near_duplicates.py`, serialized through `src/services/curation/wire.py`:

| Route | Answer |
|---|---|
| `GET curation/projects/{project}/near_duplicates/status` | `NearDupStatus`: per level the current run, `pending_count`, `stale_reasons` (`pending`, `rebuild_required`, `no_run`), the active profile revision |
| `POST curation/projects/{project}/near_duplicates/runs` | body `{level, mode}`; 202 with the job id; 409 `job_busy` |
| `GET curation/projects/{project}/near_duplicates/runs`, `.../runs/{run_id}` | `NearDupRun` list and detail (counts, params, space, timings) |
| `GET curation/projects/{project}/near_duplicates/groups` | query `level`, `purpose`, `min_size`, `has_holdout`, cursor; page of `NearDupGroupSummary` |
| `GET curation/projects/{project}/near_duplicates/groups/{group_id}` | `NearDupGroupDetail`: members with `cosine_to_rep`, `test_holdout`, class name, thumbnails by the existing crop routes |
| `POST curation/projects/{project}/near_duplicates/decisions`, `DELETE .../decisions/{decision_id}` | `NearDupDecisionIn {level, a_id, b_id, decision}`; writes a pending row for both members |
| `near_dup_profiles` routes mirroring the `ingest_profiles` family (list, get, save, revisions, activate, rollback, validate, schema) | the existing config-store shapes |

Changed: `/curation/projects/{project}/test_holdout/freeze` answers `moved_by_near_dup` and the 409
`near_dup_stale`; review and search keep `hide_near_duplicates` and read the new fields; the export's
`dedup_threshold` request field becomes `collapse_near_duplicates: bool` (reads the current `dedup`
groups, J8); `/curation/projects/{project}/scores/compute` loses the `near_dup` scorer. Events:
`near_dup.run`, `holdout.near_dup_absorbed`. `make contracts` and the contracts commit in the same
commit as the routes.

Core: `POST /ingest` stops grouping in the request. Its response's near-duplicate block becomes the
kNN candidates above the core cut (`near_duplicate_candidates`: id and cosine), and the group appears
once `near_dup_core_group` has run (J7). `GET /query/duplicates` and its two siblings read the core
run unchanged in shape. The `near_duplicate_threshold` form field on the three ingest routes goes
(the cut is the code default; no per-request grouping), which [06d_b5_processing_plans.md](06d_b5_processing_plans.md)
Q1 already plans for the selection flags.

---

## 6. Prototype protocol

### 6.1 Code location: tracked `scripts/research/near_dup/` with exact pins (recommended)

| Option | For | Against |
|---|---|---|
| **Tracked `scripts/research/near_dup/` + `scripts/research/near_dup/requirements.txt` with `==` pins + unit tests in `tests/research_scripts/`** | `docs/research/CLAUDE.md` requires a reproduce command that works from a clean checkout; the scripts import the product's grouping code (parity, trap T3); reviewable; the naming-leak and docs hooks scan them | must hold no private path or name: every dataset root is an argument or env var |
| Gitignored scratch scripts | fast, no review | published numbers cannot be reproduced; the scripts drift from the product; violates the research rules |

Recommendation: tracked, exact pins. The two existing research requirements files are unpinned
today ([01_package_pins.md](01_package_pins.md) section 3.9 pins them); the new file starts pinned.
Raw outputs never go in git: `artifacts_local/near_dup/` (gitignored, `.gitignore:421`). Only small
JSON that backs a published public number goes to `docs/benchmarks/near_duplicate_detection/` with a
`SHA256SUMS`.

Files (P0 writes them, each with a docstring and a `--help`):

| File | Role |
|---|---|
| `requirements.txt` | exact pins (6.2) |
| `build_sets.py` | the constructed copy set (6.3) and readers for California-ND, Holidays, Copydays and the DISC21 sample from local roots given as arguments |
| `encode.py` | vectors through `scripts/research/fine_grained/models.py` (`s_pe_core`, `s_mobileclip2`, `s_dinov2`) plus an SSCD TorchScript loader; writes `.npy` + ids |
| `parity.py` | venv vectors against product vectors exported from a project (`scripts/research/fetch_project_vectors.py`) |
| `hashes.py` | SHA-256, `imohash`, PDQ (`pdqhash`), pHash and dHash (ImageHash), with timings |
| `sweep.py` | exact candidates (FAISS `IndexFlatIP` range search on CPU, or blocked torch matmul on GPU), the threshold grid, both scales for core |
| `group.py` | components with the size guard and the complete-linkage arm; imports `frame_dedup.near_dup_groups` and `near_dup.compute_near_dup_groups` to reproduce today's product numbers (so the baseline rows run on a commit where those modules still exist, recorded in the results JSON; after ND-A it imports `src/domain/near_dup.py`) |
| `crops.py` | D7a versus D7b versus today's bucketed scorer |
| `leakage.py` | the holdout simulation (P5) |
| `throughput.py` | high-resolution timings (P6) |
| `sheets.py` | pair and group sheets through `src/services/labeling/contact_sheet.py`, with the privacy routing of 6.6 |
| `report.py` | results JSON stamped with the registration commit |

### 6.2 The venv

- `python3.12 -m venv "$SCRATCH/venvs/op-near-dup"` (outside the repo; `$SCRATCH` is the operator's
  scratch root). Call its binaries directly; never `source`.
- Install order: `torch==2.15.0 torchvision==0.29.2` (PyPI wheels, CUDA 13.2 libraries, per 13
  section 2), then `pip install -r requirements-test.txt` (the product code the scripts import), then
  `pip install -r scripts/research/near_dup/requirements.txt`:
  `numpy==2.5.4`, `pillow==13.0.0`, `scipy==1.18.1`, `scikit-learn==1.9.1`, `faiss-cpu==1.15.1`,
  `open-clip-torch==3.3.0`, `timm==1.0.31`, `imohash==1.1.0`, `ImageHash==4.3.2`,
  `PyWavelets==1.10.0`, `pdqhash==0.2.8`, `statsmodels==0.15.0` (versions as listed on PyPI on
  2026-11-04 or in 13). P0 checks that `pdqhash` has a wheel for the interpreter; without one it needs
  a C++ toolchain, recorded, not improvised.
- SSCD: `sscd_disc_mixup.torchscript.pt` downloaded by URL with its SHA-256 recorded in the results
  JSON (weights licence: J11).
- `pip freeze` and `nvidia-smi` output saved next to every result.

### 6.3 Datasets and sampling

| Set | Use | Size and sampling | Terms |
|---|---|---|---|
| **Constructed copies** (the absorbed E-2 set) | copy-level cuts per level and space | 5,000 originals: 2,500 COCO val2017 + 2,500 Open Images V7 validation, seeded, disjoint by SHA-256 from every pin in `scripts/datasets/manifests/`; pin `near_dup_5000.json` | COCO Flickr terms per image; Open Images CC BY 2.0 |
| | transforms (factor-isolated, Pillow, seeded, hashed) | T-q60, T-q30 (JPEG); T-half, T-quarter (Lanczos); T-crop90, T-crop80 (centre); T-bright (x1.15); T-rot (3 degrees, border cropped); compounds C-social (half then JPEG q75) and C-baseline (the four `make_variant` outputs, for continuity with `dedupe_r1.json`). Non-JPEG transforms saved as PNG if every product decoder accepts it, else JPEG q95 (P0 checks one file per format) | |
| | hard negatives | all distinct-original pairs above the lowest swept cut, person-checked up to 200 (seeded sample beyond, rate with a Wilson interval); for crops, two boxes of one image with IoU < 0.1, and same-class crops from two originals | |
| **California-ND** | burst and sequence cuts | all 701 photos; truth = pairs marked by at least 5 of 10 raters (majority), plus the photographer's matrix reported separately | CC licence, password by email; local only, never committed (J11) |
| **INRIA Holidays** | scene-level near-duplicates with rotation and viewpoint change; a negative check for the copy cut | all 1,491 images, 500 groups | INRIA copyright; local only (J11) |
| **Copydays** | copy-level recall on crops, JPEG and strong edits | the full set | INRIA copyright; local only (J11) |
| **DISC21 sample** | copy-level benchmark against a published standard | 10,000 reference images, the dev queries whose ground-truth reference is in that sample, and an equal number of distractor queries, seeded | dataset terms on the DISC21 page; code CC BY-NC 4.0 (J11) |
| **High resolution** | throughput | `scripts/datasets/manifests/commons_hires_500.json` (500 Wikimedia Commons images, up to 5,100 x 3,996, free licences) | public |
| **Set P** | the owner's real target: bursts and sequences at high resolution | a seeded sample of 2,000 images forming consecutive capture sequences plus 2,000 random singles, copied to local NVMe scratch (originals never touched or deleted); a person labels a seeded 300 pairs (200 candidate pairs across the cut range, 100 random) | private: never named, never sent to a hosted model, numbers recorded privately (6.8) |

### 6.4 Experiments

Bars and the grid are committed in `docs/research/near_duplicate_detection.md` (registration section)
**before** any score is read; the results JSON stamps the registration commit.

| Id | Question | Method | Data | Cost (estimate, unmeasured) |
|---|---|---|---|---|
| P0 | harness, parity | builders and unit tests (red then green); encoder parity: 200 images encoded by the venv and by the product, median cosine >= 0.999 or the venv vectors are not used (trap T10) | public | CPU about 2 h; GPU 10 min |
| P1 | copy-level cuts per level and space (old E-2) | encode once (PE frame, PE crop, MobileCLIP frame); exact candidates at cosine >= 0.85; grid 0.900 to 0.995 step 0.005; single linkage, single + size guard, complete-linkage arm; today's three product functions run on the same vectors for the baseline; 50/50 calibration and test split by original, read once on test | constructed copies, Copydays, DISC21 sample | GPU about 1 to 2 h; CPU about 2 h |
| P2 | tiers | (a) `imohash` collisions against SHA-256 on every byte in every set (count, by file size); (b) PDQ (Hamming 0 to 64) and pHash and dHash alone and as an extra edge over embeddings at the P1 cut; (c) SSCD at its documented 0.75 and swept | as P1 | GPU 30 min; CPU 1 h |
| P3 | burst and sequence cuts | PE frame and MobileCLIP frame against California-ND majority truth; Holidays groups as a scene-level check; set P labelled pairs; EXIF time and camera as a candidate signal with an embedding floor (set P only: COCO and Open Images carry little EXIF) | California-ND, Holidays, set P | GPU 20 min; person about 1 h |
| P4 | crop level | D7a (derived, at the P1/P3 image leak cut) against D7b (global item kNN) and today's bucketed scorer (with the project's real centroids, the product number); same-image distinct objects as negatives; positives by the box mapped through the known transform (IoU >= 0.7, same detector class; boxes truncated over 10 % excluded and counted) | constructed copies (detector crops), set P | GPU 30 min (detection through the product only if a research stack is already up; else dataset boxes, recorded as such) |
| P5 | holdout leakage | simulate a freeze on validated-proxy items: leak rate = held-out items with a train item above the leak cut, before and after closure; holdout inflation per leak cut (0.98 down to 0.90); DINOv2's looser-cut rule tested | constructed copies, California-ND, set P | CPU 1 h |
| P6 | throughput at high resolution | per image: SHA-256 (MB/s), PDQ from the full decode and from a 512 px tile (bit agreement with the full-size hash), pHash; exact FAISS GPU kNN k = 64 at 10k and 100k real vectors (the Open Images 100k pin); the incremental rescore of 1,000 pending members against 100k; no 1M claim (1M is a goal, not a measurement) | `commons_hires_500`, set P, OI 100k vectors | GPU 30 min; CPU 1 h |
| P7 | visual check | 6.6 | boundary sheets from P1, P3, P4 | person about 2 h |

### 6.5 Metrics

- **Pairs**: precision and recall of duplicate pairs at each cut, per transform (copy sets), per rater
  majority (California-ND), per labelled pair (set P), with Wilson 95 % intervals.
- **Groups**: false-group rate (share of groups mixing distinct originals or objects), group purity
  and completeness (BCubed precision and recall), group-size distribution and the largest group
  (chaining shows there first), share of components the size guard split.
- **Product-path deltas**: bucket loss (global minus bucketed recall, crop scorer); ANN loss (exact
  versus HNSW top 64 recall of the pairs above the cut, through an existing research stack only if
  one is up); order sensitivity of today's core rule (3 arrival orders).
- **Leakage**: leak rate before and after closure; holdout inflation; validated share of moved items.
- **Throughput**: images per second per tier at high resolution; MB/s for SHA-256; kNN queries per
  second; rescore wall time; peak GPU memory; host load average next to every row.
- **Stability**: the best cut chosen on COCO originals only and on Open Images only; their difference
  feeds J10.

**Bars (registered before scoring; per level and space, on the test half).** Copy level, carried
from #416: false-group rate <= 0.5 %; recall >= 0.90 on T-q60 and T-half; >= 0.80 on T-crop90; no
transform below today's default on the same half. Burst level (proposed; the owner confirms in the
registration, J12): pair precision >= 0.90 against the California-ND majority at the chosen cut,
recall reported, no bar (rater disagreement makes recall ill-defined). Leakage: after closure, zero
held-out items with a train item above the leak cut (by construction) and holdout inflation reported.
Choice rule when several cuts pass: the highest cosine that passes; single linkage over complete at
equal bars.

### 6.6 Visual check (human and VLM)

- What: pair sheets just below the chosen cut (misses), just above it on distinct pairs (false
  merges), the largest groups (chaining), and for crops the same-image pairs above the cut.
- Sample: at most 10 pair sheets and 20 group sheets per level and set; half targeted, half a seeded
  uniform control; targeted sheets never pooled into estimates.
- Judge: the hosted judge only on public sheets with no person, face or plate box (COCO `person`; Open
  Images `Person`, `Man`, `Woman`, `Boy`, `Girl`, `Human face`, `Vehicle registration plate`); every
  other public sheet to the local VLM or a person. **Set P sheets go to a person or the local VLM
  only.** Calibration first: 40 planted sheets (20 known same, including T-quarter and T-crop80; 20
  known different, half of them person-confirmed distinct pairs above cosine 0.95); judge bars pair
  kappa >= 0.7 and flag precision >= 0.8, else the judge only ranks sheets for the person.
- Human re-judge: blind, seeded 5 % of verdicts (at least 20); kappa below 0.7 suspends the judge.
- Tools: `src/services/labeling/contact_sheet.py` (it already refuses hosted calls for face and
  plate kinds), `scripts/research/visual_judge.py`.

### 6.7 Repetition and GPU etiquette

- Grouping math on fixed vectors is deterministic: one run per configuration; sampling and splits use
  3 seeds and report the spread. Throughput rows run 3 times and report the median and range with the
  load average.
- **GPUs 0 and 2 only; never GPU 1.** One GPU job at a time: `CUDA_VISIBLE_DEVICES=0` (or 2) and
  `flock "$SCRATCH/locks/gpu0.lock" <command>` (or `gpu2.lock`); check `nvidia-smi` and `uptime`
  before each job, ask the peer session that owns a busy card and wait for its reply, pause at once
  when a peer asks. CPU steps under `nice`, 8 threads. No Docker build, no new compose stack; a
  product-vector export reads an already-running research stack with its own `-p` name, never the
  default project.
- No wait loops in agent briefs (no `until`/`sleep` polling, no `pgrep -f` on a string in the
  agent's own command line); long steps run in the background and report when they exit.

### 6.8 Stop rules

| Experiment | Stop or keep |
|---|---|
| P0 | parity median below 0.999: the venv vectors are not used; P1 waits for product vectors |
| P1 | per level and space: no cut meets every bar, keep today's 0.98 there, publish the curve; a close miss is the owner's call (J8 of #416, kept) |
| P2a | zero collisions in every set: SHA-256 still recommended on principle (C11), cost from P6 |
| P2b | PDQ adds less than 2 percentage points of copy recall at equal false-group rate over the P1 cut, or costs more than 5 % of the per-image ingest time at high resolution: no hash tier |
| P2c | SSCD below PE by any margin, or above it by less than 5 points of recall at equal false-group rate: no SSCD (another model is not justified) |
| P3 | no cut reaches precision 0.90 on the California-ND majority: bursts are not grouped automatically; review shows candidates only |
| P4 | D7a misses more than 10 % of the positive crop pairs D7b finds at equal precision: add D7b as a second item-level candidate source |
| P5 | holdout inflation above 2x the selected cohort at the chosen leak cut: the owner chooses the leak cut knowingly (J3) |
| any | time box twice the estimate; a peer's request for the GPU; a host load average above 32 during a timing row (the row is re-run, not published) |

### 6.9 Results recording

- Public: `docs/research/near_duplicate_detection.md` (Setup with date, commit, GPU, driver,
  versions and load; Method; Results; Findings; Open issues; Reproduce), rows in
  `docs/research/README.md` and `docs/design/README.md`, backing JSON in
  `docs/benchmarks/near_duplicate_detection/` with `SHA256SUMS`; append-only, never rewriting
  `docs/benchmarks/retrieval_baseline/`. Only public datasets appear by name; California-ND,
  Holidays, Copydays and DISC21 numbers are aggregates (no images, no file names).
- Set P: raw outputs and per-pair labels in `artifacts_local/near_dup/set_p/` (gitignored) and the
  owner's private notes. The public note may say "set P (private, n images)" with aggregate numbers
  only if the owner agrees (J12); otherwise it says the set P check was run and passed or failed.
- #416 gets a comment with the note's path and the per-level verdicts.

---

**PW-1 outcomes (2026-11-06; [results](../../research/near_duplicate_detection.md)).** About 0.5 GPU hours with
tier 2. P0: product-engine parity passes (PE median cosine 0.99992). P1 tier 1 chose PE 0.925 and core 0.965
(single linkage) on every split; tier 2 through the live core route confirms **core 0.965 (score 0.9825)** on
the point estimate (copy recall 0.905 against 0.563 at 0.98, no mixed group in 357) and does **not** confirm PE
0.925 (JPEG q60 recall 0.896, bar 0.90). Lane ND-A therefore seeds `near_dup_profile` with core 0.965 (labelled
"tier 2, point estimate") and keeps PE at 0.98; 0.92 for PE is a post-tag RW-4 run. Crop level, leak cut and `k`
stay at today's values. P2: no hash tier (PDQ costs 12.8 ms per 640 px image against a 1.4 ms allowance);
SHA-256 at about 319 MB/s single thread. Tier 2 also showed that today's core route groups copies far worse than
its own vectors allow (the stored group ties a q60 copy to its original 3 % of the time, exact grouping on the
same vectors 61 %): the per-batch approximate assignment over an unrefreshed batch, which ND-C's single-writer
grouping job replaces.

## 7. Production design

### 7.1 Flow

```
ingest (curation or core)
  -> SHA-256 of the bytes -> INSERT image_contents ON CONFLICT DO NOTHING   (exact: duplicate answered here)
  -> decode once -> vectors (PE frame, PE crops; MobileCLIP for core)
  -> INSERT near_dup_pending ('added')  -> OpenSearch document writes
beat tick (per project and level with pending rows)
  -> near_dup_rescore (single writer: exclusive group near_dup, scope level)
       claim pending -> kNN k=64 at the leak cut -> upsert edges
       -> recompute touched components (+ decisions, size guard) -> incremental run
       -> holdout absorption -> flip current (+ search_projection_tasks row, event)
  -> near_dup_projection_sync -> OpenSearch near_dup_* fields
  -> image level done -> item-level rescore over the touched image leak groups
review / search      : hide_near_duplicates reads near_dup_is_representative
export / training    : collapse_near_duplicates reads the current dedup groups; refuses when stale and a holdout exists
holdout freeze       : refuses when stale; closure over item leak groups; moves recorded
```

### 7.2 Module layout

| Path | Layer | Holds |
|---|---|---|
| `src/domain/near_dup.py` | domain | levels, purposes, reasons, defaults, `ALGORITHM_VERSION`, `core_score()`, pure components with the size guard, `most_central` (moved from `frame_dedup.py`), lineage Jaccard; imports only `config` and `utils` |
| `src/db/models/near_dup.py`, `src/db/repos/near_dup.py` | db | ORM models; every statement (current flip, pending claim with `SKIP LOCKED`, `COPY` of members) |
| `src/services/near_duplicates/` | services | `candidates.py` (FAISS GPU batch kNN, OpenSearch kNN), `build.py`, `rescore.py`, `items.py` (D7a), `projection.py`, `holdout.py`, `status.py`, `profiles.py`, `content_key.py` (exact) |
| `src/tasks/near_dup.py`; `src/config/job_kinds/near_dup.py`; `src/config/task_routes/near_dup.py` | tasks, config | the five task names and kind rows |
| `src/schemas/curation/near_duplicates.py` | schemas | the wire models and `NearDupProfileBody` |
| `src/routers/curation/near_duplicates.py`, `src/routers/curation/near_dup_profiles.py` | routers | thin adapters |

Deleted in the same change: `src/services/curation/item_scores/near_dup.py` and its registry entry;
`frame_dedup.py` (its export use replaced by the stored groups; `most_central` moved; the research
oracle in `scripts/research/near_dup/group.py` reimplements nothing, it imports the domain function);
`src/services/duplicate_detection/service.py::create_duplicate_group` and the in-request assignment
in `ingest_single.py` and `ingest_batch.py`; the seven 0.99 literals; `OP_CROP_DUP_THRESHOLD` (into
`src/config/retired_env.py`, `env.template:590` removed); `imohash` from `requirements.txt` and the
pins if J6 is accepted.

### 7.3 Behaviour that matters

- **Determinism**: a full build on the same vectors, profile and decisions yields the same groups
  (exact FAISS kNN, sorted edge processing, stable representative tie-break by member id). An
  incremental run equals a full build on the same state for the components it touched; a test
  asserts it on a fixture.
- **Single writer**: one `near_dup` job per (project, level) by the unique index; replicas only add
  capacity. No in-process locks (the multi-worker rule).
- **Fail closed**: a missing vector on a pending member fails that member into the run's error count
  and keeps its pending row; the run is `current` only if every claimed row was handled or kept; an
  unbound project refuses; a profile that fails validation refuses activation.
- **Lock rule**: near-duplicate writes are machine fields; they never alter a validated class or box.
  Human `near_dup_decisions` bind every later run. Holdout moves only add items to the holdout.
- **Region boxes**: inherit their item's group (J13); no region-level grouping.
- **Encoder change** (#347, #365): a new `space` string marks every level `rebuild_required`; edges of
  the old space are deleted after the rebuild flips current.

### 7.4 Tests (red first, per lane)

| Test | Proves | Red today |
|---|---|---|
| `tests/near_duplicates/test_content_key_race.py` (`db`) | two concurrent ingests of one content store it once | the index lookup lets in-flight copies through (#358 shape) |
| `tests/near_duplicates/test_rescore_clears_stale.py` | after a rescore that ungroups an item, no `near_dup_*` field remains on it and the id is new | stale `dup_*` and reused `dup_0` (#473 part a) |
| `tests/near_duplicates/test_single_writer.py` (`db`) | two concurrent starts give one `near_dup` job per (project, level) | - (new) |
| `tests/near_duplicates/test_incremental_equals_full.py` | incremental run equals a full build on the touched components | - |
| `tests/near_duplicates/test_items_include_validated.py` | validated and held-out items are grouped at item level | the residual pool excludes them (C8) |
| `tests/near_duplicates/test_holdout_near_duplicates.py` (`db`) | 5.4 closure, counts, stale refusal | the freeze ignores groups (#473 part c) |
| `tests/near_duplicates/test_holdout_absorption.py` | a later run moves a new near-duplicate of a held-out item into the holdout | - |
| `tests/near_duplicates/test_decisions_bind_runs.py` | `different` splits, `same` joins, in every later run | - |
| `tests/near_duplicates/test_profile_validation.py` | `leak > dedup`, `k = 0`, cosine outside range refuse | the env var falls back silently |
| `tests/visual_search/test_core_grouping_serialized.py` | two concurrent core batches with a shared pair give one group | racy today (#473 part b) |
| `tests/test_no_legacy_near_dup_fields.py` | no `dup_group_id`, `duplicate_group_id`, `phash`, `OP_CROP_DUP_THRESHOLD` or 0.99 near-duplicate literal remains in `src/` | they exist |

---

## 8. Landing lanes

| Lane | Wave | Branch | Owns | Cut after | Tasks |
|---|---|---|---|---|---|
| **ND-A** near-duplicate machinery | W2.3 (parallel with 4A-4F) | `b2/near-duplicates` | 7.2 paths; the `near_duplicates` Alembic revision; `src/clients/curation_opensearch/bodies_core.py` and `ensure_overlay_fields.py` (the near-duplicate mapping lines only); `semantic_search.py` and `review_request.py` (the filter fields only); `item_scores/near_dup.py` (deleted; taken from 4A); `frame_dedup.py`, `export_images.py`, `export_single_class.py`, `strategy_builders.py` (the dedup lines only); `env.template` and `retired_env.py` (append rules); `contracts/` | 2C (config store), 2D, 3A, 3B merged | ND-1 schema + RLS; ND-2 profile kind and seeding; ND-3 pending writers in curation ingest, reprocess, region stage, delete, import, combine; ND-4 score, group, rescore tasks; ND-5 projection sync and field deletion; ND-6 routes, wire, contracts; ND-7 export collapse (J8); ND-8 content key (J6) in the curation ingest; ND-9 retire `OP_CROP_DUP_THRESHOLD` (replaces 4A-6); ND-10 the upgrade-step field clear (handed to 2E's owner if 2E is still open, else in this lane) |
| **ND-B** holdout guard | W2.3, after ND-A merged | `b2/near-dup-holdout` | `src/services/near_duplicates/holdout.py`; the freeze call sites in `holdout_freeze.py` and `holdout.py` (2D merged by then); the training-export gate call | ND-A | ND-B1 closure at freeze (replaces 2D-4); ND-B2 absorption at run flip; ND-B3 export and cohort gate; ND-B4 status route `stale_reasons` read by Cropwright |
| **ND-C** core path | W4 (inside `b4/bulk-occ`, which rewrites `ingest_batch.py`) | the 4A lane of [05_b4_stateless_api.md](05_b4_stateless_api.md) | `src/services/visual_search/ingest_single.py`, `ingest_batch.py` (the grouping lines), `src/services/duplicate_detection/`, `src/routers/ingest/*.py` (the threshold fields), `src/routers/query_duplicates.py` (reads), the core index mapping | B3 head | ND-C1 core pending rows + `near_dup_core_group` (replaces 4A-4b's "batch-scoped step"); ND-C2 the ingest response candidates (J7); ND-C3 delete the literals, the Immich comment and the swallowed exception; ND-C4 core content key |
| **ND-D** research | after the owner's go (J1) | `research/near-dup-protocol` | `scripts/research/near_dup/**`, `tests/research_scripts/test_near_dup_*.py`, `scripts/datasets/manifests/near_dup_5000.json`, `docs/research/near_duplicate_detection.md`, `docs/benchmarks/near_duplicate_detection/` | any | P0-P7 |
| **ND-E** defaults and tiers from evidence | after ND-D results and the owner's yes | `feat/near-dup-defaults` | `src/domain/near_dup.py` defaults; the PDQ tier (edge kind, fields, ingest hash on the decoded frame, reusing the B5 resident frame) only on a P2 pass | ND-A, ND-D | change the seeded defaults (existing projects keep their active revision; a new revision is offered, not forced); add tiers |
| docs | with each lane | in-lane | `docs/CURATION.md`, `docs/ARCHITECTURE.md`, `docs/opensearch_schema_design.md`, `docs-site/docs/api-reference/core.mdx`, `CHANGELOG.md` | | each lane documents its routes and fields in the same PR |

Each lane: its own worktree and branch, targeted red-then-green tests and static checks on touched
files only; the wave's single full sweep runs on the integration branch.

---

## 9. Risks

| # | Risk | Guard |
|---|---|---|
| R1 | Wrong scale: a cosine read as an OpenSearch score or the reverse | every number named with its space; `core_score()` in one place; a unit test pins `(1 + cos) / 2` |
| R2 | Chaining: one giant leak group swallows a project and the holdout explodes | size guard + complete-linkage split; `largest_group` and `oversized_split` on every run; P5 inflation bar |
| R3 | Stale state used for a freeze or a training cohort | 409 `near_dup_stale` (fail closed); pending rows written before the vector write |
| R4 | The incremental run drifts from a full build | the equality test; a weekly full build per project by beat (cheap at image level) |
| R5 | ANN misses (HNSW top 64) at the online step | full builds use exact FAISS; P1 and the ANN-loss metric |
| R6 | Item level misses zoomed burst frames (D7a) | P4; D7b as a second source if the stop rule says so |
| R7 | Holdout absorption keeps moving items as data grows | every move audited in `holdout_near_dup_moves`; J5; the status shows the count |
| R8 | A new hash or SSCD model adds a second decode | the one-decode rule; PDQ from the resident frame's 512 px tile (P6 measures bit agreement) |
| R9 | Private data in public text or a hosted judge | set P aggregates only with J12; privacy routing; the naming-leak hook |
| R10 | Dataset terms (DISC21, INRIA sets, California-ND) | local use only, never committed, J11 before download |
| R11 | Old 0.6.0 documents carry stale `dup_*` and `duplicate_*` values | the upgrade clear plus a post-upgrade full build; `test_no_legacy_near_dup_fields.py` for the code |
| R12 | Scope creep in v0.7.0 | ND-A and ND-B are seams with today's defaults; thresholds and tiers change only through ND-E after evidence |
| R13 | Members table growth | full copy per run with `COPY`; retention of three runs; sizes recorded in 02a section 9 at landing |

---

## 10. Owner questions

| J | Question | Recommendation | Trade-off |
|---|---|---|---|
| J1 | Timing: build ND-A and ND-B in v0.7.0 (W2.3) with today's 0.98 defaults and research after the tag, or run P0-P2 (about 4 GPU hours, GPU 0 or 2) before W2.3 so the defaults ship from evidence? | build the seams in W2.3; run P0 and P1 before W2.3 if a free GPU window appears, otherwise after the tag | v0.7.0 otherwise ships the hypothesis defaults (made revertible by the profile) |
| J2 | Holdout closure scope: the held-out item's item-level leak group only, or every item on near-duplicate frames of the anchor's image? | item-level group only (today's per-crop holdout unchanged) | the whole-frame rule leaks less background but moves more unrelated objects |
| J3 | A separate looser `leak` cut (DINOv2 pattern) with its default equal to the dedup cut until P5? | yes | equal cuts are simpler but give the holdout no margin |
| J4 | Freeze and training export refuse on stale near-duplicate state (409), rather than the freeze starting a rescore itself? | refuse; Cropwright offers the rescore button | one extra click versus a request-bound job wait |
| J5 | After a freeze, move new near-duplicates of held-out items into the holdout automatically, or only exclude them from training? | move (audited) | the holdout keeps growing in burst-heavy projects; exclusion alone leaves items in limbo |
| J6 | SHA-256 replaces `imohash` as the exact key, with the PostgreSQL unique constraint | yes; P6 reports the cost | a full read of every file; P2a counts real collisions |
| J7 | Core `/ingest` groups asynchronously (the response lists candidates; the group appears after the core job) | yes | a wire change for core clients; synchronous grouping under a lock serializes ingest |
| J8 | Export: replace `dedup_threshold` with `collapse_near_duplicates` reading stored groups | yes | per-request cuts go away |
| J9 | Perceptual tier: PDQ (if P2 passes), ImageHash pHash, or none; SSCD only on a large gain | PDQ on a pass, else none; SSCD only past its stop rule | another field and ingest step versus model independence |
| J10 | Per-project thresholds through `near_dup_profile` (supersedes OQ-9's "one default per path") | yes, seeded defaults, an advanced setting in the UI | a setting few users can calibrate; the status and the research note give them a starting point |
| J11 | May the research download and use, locally and never committed, California-ND (password by email), INRIA Holidays and Copydays (INRIA copyright), a DISC21 sample (its terms), and SSCD weights (licence not stated in the README)? | yes, aggregates only in public text | without them, burst and copy-benchmark evidence rests on constructed sets and set P |
| J12 | May set P aggregate numbers appear in the public note, and are the burst bars of 6.5 right? | aggregates allowed, no images or names; bars as written | the set P result is the one the owner cares about; private-only keeps it out of the paper |
| J13 | Region boxes inherit their item's group, no region-level grouping | yes | a wheel of a near-duplicate car crop is grouped through its item |
| J14 | Retire `OP_CROP_DUP_THRESHOLD` instead of validating it (4A-6) | retire | 4A-6's small fix becomes a deletion |
| J15 | Rename the core index fields to `near_dup_*`, keeping the `/query/duplicates` wire shapes | yes | one vocabulary in storage; a mapping change on the global index |

---

## 11. Required changes to other plan files (for the integration agent)

This commit edits only: this file (new), `docs/design/issue_plans/issue-416-plan.md` (deleted),
`docs/design/README.md` (the index rows), and the one link in
[v070_rebuild/05_b4_stateless_api.md.md](05_b4_stateless_api.md) task 4A-4b that pointed at the
deleted file (retargeted here so the link check passes). Everything else below is for the
integration agent:

| File | Change |
|---|---|
| [v070_rebuild/00_master_plan.md.md](00_master_plan.md) W2.2, W2.3, W4 | W2.2: drop 2D-4 from lane 2D's fold-ins (moved, C8). W2.3: add lanes ND-A (`b2/near-duplicates`) and ND-B (`b2/near-dup-holdout`) with the ownership of section 8; `item_scores/near_dup.py` moves from 4A to ND-A; the #473 part a and part c fold-ins point at ND-A and ND-B. W4: 4A-4b becomes ND-C1-C4 inside `b4/bulk-occ`. Add J1 and J14 to the owner questions |
| [v070_rebuild/reference/decision_register.md.md](reference/decision_register.md) section 6 | register J1-J15 when answered; MQ-5 gains "applied by ND-B (09d_research_near_duplicates.md section 5.4)"; section 10 scope lists ND-D and ND-E as after the tag unless J1 says otherwise |
| [v070_rebuild/reference/traceability.md.md](reference/traceability.md) sections 2-3 | #416: "research plan `09d_research_near_duplicates.md` (ND-D, ND-E)"; #473: part a ND-A (ND-4, ND-5), part b ND-C1, part c ND-B1; F-5..F-8 rows re-pointed; a new F row for the content key (J6) as the structural fix of #358's batch lookup |
| [v070_rebuild/03b_stage2_control_data.md.md](03b_stage2_control_data.md) lane 2D | remove 2D-4 and its test (moved to ND-B1, reason C8); 2D-1 notes that `holdout_freezes` gains three columns in the ND-A revision |
| [v070_rebuild/03d_stage4_job_families.md.md](03d_stage4_job_families.md) lane 4A | remove `item_scores/near_dup.py` from 4A's ownership; 4A-6 becomes "retired by ND-9" (J14); 4A-7 becomes "superseded by ND-4/ND-5 (run-scoped groups, clear on projection)"; the `item_scores` kind loses the `near_dup` scorer |
| [v070_rebuild/05_b4_stateless_api.md.md](05_b4_stateless_api.md) 4A-4b | rewrite as ND-C1: core ingest writes `near_dup_pending` rows; grouping by the `near_dup_core_group` job (single writer); red test `tests/visual_search/test_core_grouping_serialized.py` |
| [v070_rebuild/02a_postgres_schema.md.md](02a_postgres_schema.md) | a new section "Near-duplicates" with the tables of 5.1 (or a link here); the `config_documents.kind` and `config_activations.axis` CHECK lists and the axis pair of 5.5; `search_projection_tasks.kind` gains `near_dup_run`; `holdout_freezes` columns; section 9 retention row |
| [v070_rebuild/02b_jobs_locking_events.md.md](02b_jobs_locking_events.md) section 2.1 | the five kind rows of 5.2; the beat schedule gains the `near_dup_rescore` tick, the weekly full build and the run pruning |
| [v070_rebuild/02_control_plane_database.md.md](02_control_plane_database.md) sections 3.1, 3.2, 4.7 | rows: near-duplicate runs, groups, members, edges, decisions, pending, holdout moves -> PostgreSQL, the `near_dup_*` fields as search projection; the exact key `image_contents`; images lose `imohash` and `phash` |
| [v070_rebuild/06d_b5_processing_plans.md.md](06d_b5_processing_plans.md) | the `near_duplicates` branch returns candidates, not groups (J7); Q1 includes the `near_duplicate_threshold` field |
| [v070_rebuild/01_package_pins.md.md](01_package_pins.md) section 3.9 | add `scripts/research/near_dup/requirements.txt` with the pins of 6.2; if J6 passes, `imohash` leaves the product manifests; if J9 passes, `pdqhash==0.2.8` (MIT, bundles ThreatExchange PDQ under BSD) joins them in ND-E |
| [09b_research_clustering.md](09b_research_clustering.md) | RQ-2, E-2, the RQ-2 rows of its method, data, bars and landing tables, and OQ-9 point here (absorbed as P1 and J10); RQ-6's near-duplicate must-link reads the `leak` groups |
| [cluster_search_makemodel_plan.md](../cluster_search_makemodel_plan.md) signals table | "Near-duplicate group" row: fields become `near_dup_group_id`, `near_dup_is_representative` |
| [v070_rebuild/reference/github_actions.md.md](reference/github_actions.md) | comment on #416 with this plan's path (the gist plan is superseded); comment on #473 with the re-cut placement |
| `docs/CURATION.md`, `docs/ARCHITECTURE.md`, `docs/opensearch_schema_design.md`, `docs-site/docs/api-reference/core.mdx` | at landing, by ND-A, ND-B and ND-C (not before: docs describe current behaviour) |
