# Cluster types: what is clustered, in which embedding space

Status: code-derived reference (2026-10), no new measurements. Every statement below was read from the code at the commit this note was written; the algorithm comparison and the measured numbers for the curation residual pool are in [clustering_methods.md](clustering_methods.md).

Implemented in the public repo: yes, all three kinds. They share the word "cluster" and the field names `cluster_id` and `cluster_distance`, but they are separate subsystems with separate indexes, embedding models, algorithms and triggers. They share no code (the IVF module in the curation package says so itself: `src/services/curation/clustering/methods/ivf.py`).

Two of the three kinds have two mechanisms, so the table has five rows:

| Kind | Mechanisms |
|---|---|
| 1. Whole-image (and object-crop) similarity | 1. FAISS IVF over the visual-search indexes |
| 2. Detection / region-crop clustering for curation | 2a. residual pool over item (crop) embeddings; 2b. region-box clustering over per-box embeddings |
| 3. Face clustering | 3a. FAISS IVF over the faces index; 3b. identity grouping into `person_id` |

## 1. Comparison

| | 1. Visual-search IVF | 2a. Curation item residual pool | 2b. Curation region-box clustering | 3a. Faces IVF | 3b. Person grouping |
|---|---|---|---|---|---|
| What is clustered | One vector per image (`global`), per detected vehicle crop (`vehicles`), per person crop (`people`) | One vector per item (a crop, `crop_id`) that is not confidently labeled | One vector per accepted region box (`crop_id` + `box_id`); false-positive boxes separately | One vector per detected face | One vector per detected face |
| Embedding model, dim | MobileCLIP2-S2, 512 | PE-Core-L14-336, 1024 (`pe_embedding`) | PE-Core, 1024 (`region_box_embeddings[].embedding`) | ArcFace, 512 | ArcFace, 512 |
| Vector index / field | `visual_search_global.global_embedding`; `visual_search_vehicles.embedding`; `visual_search_people.embedding` | project items index, field `pe_embedding` (override `OP_RESIDUAL_EMBEDDING_FIELD`) | sibling nested field `region_box_embeddings` of the project items index | `visual_search_faces.embedding` | `visual_search_faces.embedding` |
| Algorithm | FAISS `IndexIVFFlat`, inner product on L2-normalized vectors (cosine); k-means centroids | default `ivf`: FAISS k-means, 512 centroids; selectable `ahc`, `hdbscan`; AHC refine per bucket | scikit-learn `MiniBatchKMeans`, `k = max(8, round(n / 800))`; AHC refine per bucket; FP sub-typing by k-means | same as 1, `faces` config | union-find over pairwise cosine similarity, connected components |
| Thresholds | `n_clusters` default `global` 1024, `vehicles` 512, `people` 512 (`faces` 2048); rebalance threshold 0.5 | min 32 residuals; IVF K = min(512, n // 6); AHC distance 0.25 complete linkage; refine 4 to 8000 members | min 32 boxes; AHC distance 0.25; FP auto-pull L2 0.20; FP sub-type size 150 | as 1, `faces` 2048 clusters, `nprobe` 64 | `similarity_threshold` 0.5 to 0.9, default 0.7; `min_faces_per_person` default 2; `max_persons` default 1000; 100000-face read cap |
| Trigger | manual `POST /clusters/train/{index}`; ingest assigns the nearest centroid once an index is trained | auto-label job (`train_clusters`), the refresh worker, ingest-time assignment from persisted centroids | manual `POST .../regions/cluster` (background job), UI button; never part of the auto-label job or the refresh worker | manual, `index=faces`; face ingest assigns when trained | manual `POST /persons/cluster` |
| Worker | API process | `curation-auto-label-worker` runs the job; `curation-cluster-refresh` decides when | API process (single-flight background task, status in a file under the project state dir) | API process | API process |
| Stored fields | `cluster_id`, `cluster_distance` (1 - cosine), `clustered_at` on the document | item `cluster_id`, `cluster_subid`, `cluster_distance`; centroids in the project state dir (`ivf_residuals/`) | per box in `region_boxes[]`: `cluster_id`, `cluster_subid`, `cluster_distance` (L2); FP centroids in the FP store | as 1 | `person_id`, `cluster_assigned_at`, `person_name`, `merged_from` |
| API | `/clusters/*` (also `/v1/clusters/*`), no project prefix | `/curation/projects/{project}/clusters*`, `.../pipeline/auto_label/start` | `/curation/projects/{project}/regions/cluster*`, `.../regions/clusters*`, `.../regions/fp_centroids/*` | as 1 | `/persons/*` (also `/v1/persons/*`), no project prefix |
| UI surface | none in Cropwright | Cropwright `/p/{project}/clusters` list and `/clusters/{id}` | region gallery on the same `/clusters` page | none | none |
| Key files | `src/routers/clusters.py`, `src/services/visual_search/clusters.py`, `clustering.py`, `clustering_models.py`, `src/clients/opensearch/clusters.py` | `src/services/curation/clustering/orchestrator.py`, `methods/ivf.py`, `ivf_store.py`, `ivf_ingest.py`, `refine.py`, `residual_fetch.py`, `scripts/curation/cluster_refresh_daemon.py` | `src/services/curation/clustering/region_box_clustering.py`, `region_cluster_jobs.py`, `src/routers/curation/regions_fp.py` | as 1 | `src/routers/persons.py`, `src/services/face_identity/person_management.py` |

The curation routes are project-scoped (`/curation/projects/{project}/...`) and have no `/v1` twin. The `/clusters` and `/persons` routes are the core API: no project prefix, one global set of `visual_search_*` indexes, and no project isolation.

## 2. Kind 1: visual-search IVF (whole-image similarity)

Data flow:

1. `POST /clusters/train/{index}` (`index` is `global`, `vehicles`, `people` or `faces`) reads every embedding from the OpenSearch index (`max_samples` caps it), trains a FAISS IVF index on the GPU when one is available, assigns every document to its nearest centroid and writes `cluster_id` and `cluster_distance` back.
2. The trained index is saved atomically to `data/faiss_indexes/<index>.index` plus `<index>.meta.npy` (relative to the API working directory, `/app` in the container, where `./data` is a bind mount, so the indexes survive a container recreate; the directory is git-ignored).
3. At ingest, a document whose index is trained gets `cluster_id`, `cluster_distance` and `clustered_at` from a nearest-centroid lookup (`src/clients/opensearch/ingest.py`, `bulk.py`). A failing assigner is logged and skipped; ingest never fails on clustering.
4. `POST /clusters/assign/{index}` assigns documents that have no `cluster_id` yet. `GET /clusters/stats/{index}`, `GET /clusters/balance/{index}`, `POST /clusters/rebalance/{index}` (a retrain), `GET /clusters/{index}/{cluster_id}` (members) and `GET /clusters/albums` (global-index clusters of at least `min_size` images, default 5, as "albums") complete the surface.

Parameters: `n_clusters` (16 to 8192) and `max_samples` on train. When `n_clusters` is omitted, the configured default is used if it lies between `max(16, sqrt(n))` and `n / 10`, otherwise the nearest bound, and never more than `n`. A rebalance is recommended when the largest non-empty cluster is more than 10 times the smallest, more than 10% of clusters are empty, or the vectors added since training exceed 50% of the training set.

Known limits:

- This is a partition, not identity or near-duplicate detection. Near duplicates are `duplicate_group_id` and the `/query/duplicates` routes, a different mechanism.
- `ClusteringService` is built with a fixed 512 dimension for every index, so only 512-d vectors can be trained.
- Trained indexes are loaded from disk on first use, including by the ingest-time `is_trained` check, so ingest assigns clusters right after an API restart. A corrupt, untrained or wrong-dimension index file is logged and treated as untrained (fail closed) until the index is retrained.
- Cluster ids are cell numbers of a k-means partition; a retrain renumbers them.

Tests: `tests/test_clustering_singleton.py` (singleton and `load_index` branches), `tests/test_cluster_route_order.py` (route order). No offline test trains an index or exercises `check_balance`, and there is no offline test for the album listing. Those paths are covered only by the live stack.

## 3. Kind 2: curation clustering of detections and region crops

Two mechanisms write `cluster_id`, at different levels. Item-level clustering writes the top-level `cluster_id` of an item; region-box clustering writes `cluster_id` inside each `region_boxes[]` entry, and an item with several boxes can sit in several buckets (the item itself then has no cluster from this pass).

### 3.1 Item residual pool (2a)

Data flow:

1. Class clusters mirror the class: for a labeled item `cluster_id == class_id` (`id_normalize.py`). Ids `>= 10000` (`RESIDUAL_CLUSTER_ID_OFFSET`) are candidate clusters from the residual pass; negative ids are unassigned (`-1`), excluded (`-2`) or parked by the primary-subject gate (`-3`).
2. The residual pool is every embedded item that is not human-validated, not from a confident class source (model, VLM, human) and not excluded (`residual_fetch.py`). Confident items are never re-clustered.
3. The pool is fetched with a point-in-time view and sliced parallel readers, clustered by the chosen method (default `ivf`: FAISS k-means, K = min(512, n // 6), centroids persisted per project under `ivf_residuals/`), and `RESIDUAL_CLUSTER_ID_OFFSET + centroid index` is written back through a guarded bulk writer. Fewer than 32 residuals: no-op.
4. At ingest, an unconfident item is assigned the nearest persisted centroid immediately (`ivf_ingest.py`; no centroids yet leaves `cluster_id` null). `reassign_only=true` re-sorts the pool against the current centroids without a retrain.
5. `POST /curation/projects/{project}/clusters/refine/{cluster_id}` runs AHC (complete linkage, cosine, `distance_threshold` 0.25) inside one cluster and writes `cluster_subid` (`"47a"`, `"47b"`, ...). It skips clusters under 4 or over `OP_MAX_REFINE_MEMBERS` (default 8000) members.
6. `POST /curation/projects/{project}/clusters/auto_promote` validates classifier-labeled items that agree with a high-purity cluster, behind the accuracy-audit gate.

Trigger and worker: `POST /curation/projects/{project}/pipeline/auto_label/start` (`train_clusters`, `clustering_method`, `n_clusters`, `recluster_unvalidated`, `reassign_only`, `gate_max_rank`, `gate_min_blur_ratio`) runs the cluster stage in the auto-label worker. `scripts/curation/cluster_refresh_daemon.py` (service `curation-cluster-refresh`) polls each project and, once the item count has grown by `OP_CLUSTER_REFRESH_GROWTH_THRESHOLD` (default 200) since the last retrain, runs auto-promote and queues the full retrain as an auto-label job, waiting for ingest to go quiet for at most `OP_CLUSTER_REFRESH_MAX_DEFERRAL_S`. Other knobs: `OP_CLUSTER_REFRESH_INTERVAL_S`, `OP_CLUSTER_REFRESH_RETRY_BACKOFF_S`, `OP_CLUSTER_REFRESH_AUTO_LABEL`. The IVF retrain gates (`OP_IVF_RETRAIN_GROWTH`, `OP_IVF_RETRAIN_MIN_INTERVAL_S`, `OP_IVF_RETRAIN_CHECK_S`) are described in [clustering_methods.md](clustering_methods.md#4-streaming-architecture). GPU placement: [gpu_clustering.md](gpu_clustering.md), `OP_CLUSTER_GPU_DEVICE`.

UMAP is not part of the clustering path: the clusterer reads the raw L2-normalized 1024-d vectors, and UMAP exists only for the 2-d scatter (`POST /curation/projects/{project}/cluster/umap/rebuild`).

Known limits: IVF cells can split a tight group at a boundary (refine addresses it); AHC refine builds a full pairwise matrix (about 8 n^2 bytes), hence the member cap; candidate ids shift when centroids retrain.

Tests: `tests/curation/test_clustering_orchestrator.py`, `test_orchestrator_residuals_refine.py`, `test_clustering_fail_closed.py`, `test_ivf_single_bucket.py`, `test_ivf_idspace.py`, `test_park_gated_residuals.py`, `test_cluster_distance_written.py`, `test_cluster_refresh_schedule.py`, `test_cluster_refresh_daemon_glue.py`, `tests/projects/test_ivf_store_per_project.py`; live: `tests/live/test_live_clusters.py`.

### 3.2 Region-box clustering (2b)

Data flow:

1. Each accepted box with a vector in `region_box_embeddings` is a row. Vectors live in a sibling nested field so that item reads and OCC writes never carry (or accidentally delete) them; the entry stores `bbox_norm`, and a box moved since embedding has a stale vector and is skipped.
2. `POST /curation/projects/{project}/regions/cluster` starts a single-flight background job and returns at once; poll `GET .../regions/cluster/status` (state lives in a file so any API worker can answer). The job rebuilds false-positive sub-centroids, auto-pulls tight false-positive matches (`auto_fp_threshold`, default 0.20 L2, 0 disables), then re-partitions the good boxes (`max_rank` limits to the top-N largest crops). A fresh manual refine suppresses the re-partition unless `force_repartition=true`.
3. The partition is `MiniBatchKMeans` on unit-norm vectors, `k = max(8, round(n / 800))`, written as `cluster_id` 0..K-1 and `cluster_distance` (L2). It clears `cluster_subid`, respects human-final items and locked boxes, and has no class gate and no id offset.
4. `POST .../regions/clusters/refine/{cluster_id}` runs AHC on one bucket's boxes (fixed 0.25 threshold, same member cap) and writes `cluster_subid`. `GET .../regions/clusters` returns the cluster cards (`size` counts items, `box_count` counts boxes).
5. False-positive boxes live in a permanent bucket, `cluster_id = -100`, sub-typed by k-means (about 150 boxes per sub-type, `cluster_subid` like `-100a`) with the centroids persisted by `src/services/detection/fp_store.py`. They are never rows of the good partition. `POST .../regions/fp_centroids/build` and `GET .../regions/fp_centroids/status` run the sub-typing alone.

Known limits: on demand only, so a stream of new boxes is not clustered until someone starts the job; no persisted partition centroids for ingest-time assignment (unlike 2a); `cluster_distance` here is L2 on unit vectors, while 2a stores a cosine distance, so the two values are not comparable; the region refine route takes no threshold or member-cap parameter.

Tests: `tests/curation/test_region_box_clustering.py` (partition, refine, FP sub-typing and auto-pull, stale vectors, human and lock handling, write races); live: `tests/live/test_live_regions.py`, `tests/live/test_live_clusters.py`.

## 4. Kind 3: face clustering

Two independent mechanisms, writing different fields of `visual_search_faces`.

### 4.1 Faces IVF (3a)

The same machinery as kind 1 with `index=faces` (2048 clusters, `nprobe` 64 by default). Writes `cluster_id` and `cluster_distance`. These are cells of an embedding partition, not people: one person spans several cells and one cell holds many people. Use it as a coarse index at scale (the persons route itself points there for more than 100000 faces), not as identity.

### 4.2 Person grouping (3b)

Data flow: `POST /persons/cluster` pages through all faces (stop at 100000), stacks the 512-d ArcFace vectors, computes the full cosine similarity matrix, unions every pair with similarity at or above `similarity_threshold` (union-find) and treats each connected component as one person. A component with at least `min_faces_per_person` faces gets a new `person_id` (`person_<12 hex>`) and `cluster_assigned_at` written on each face. `GET /persons`, `GET /persons/{person_id}`, `PUT /persons/{person_id}/name` (writes `person_name` to all faces), `POST /persons/merge` (retargets `person_id`, records `merged_from`) and `DELETE /persons/{person_id}` (`delete_faces` true deletes the faces, false gives each face its own `person_<face_id>` and clears the name) complete the lifecycle. Names and merges are the only human input.

Known limits:

- Connected components are single linkage: one bridging pair joins two people (chaining). Raise the threshold to cut chains at the cost of splitting a person across pose and light.
- The similarity matrix is dense n x n float32. 100000 faces is about 40 GB, so the stated cap is not safe on a small host.
- A component below `min_faces_per_person` is not written. Its faces keep whatever `person_id` they had; the code does not assign a unique id as the route docstring says.
- A re-run issues fresh `person_id`s for every qualifying component, so earlier merges and names on re-grouped faces are replaced, not carried over.
- `max_persons` stops creating persons once exceeded, leaving later components unwritten.

Tests: `tests/test_person_queries.py` (list and detail queries use the keyword field), `tests/test_export_face_recognition.py` (export). No offline test covers `auto_cluster_faces` or the faces IVF path.

## 5. What must not be mixed

Never put vectors from different embedding models into one distance computation, one centroid set, one graph or one comparison of distances. The dimensions overlap by accident (512 for both MobileCLIP and ArcFace), so a shape check does not catch it.

| Space | Where | Used by |
|---|---|---|
| MobileCLIP2-S2, 512 | `visual_search_global`, `_vehicles`, `_people` | kind 1 |
| ArcFace, 512 | `visual_search_faces.embedding` | kind 3 |
| PE-Core-L14-336, 1024 | items-index `pe_embedding`; `region_box_embeddings[].embedding` | kind 2 (2a on the item vector, 2b on the box vector) |
| Detector backbone RoI-pool, 1024 | items-index `backbone_embedding` | no clustering path reads it |

Further rules:

- Within PE-Core, an item vector (2a) and a box vector (2b) describe different image regions. Do not pool them, and do not compare an item's `cluster_id` with a box's `cluster_id`; the namespaces differ (2a: class ids, then 10000+; 2b: 0..K-1 and -100).
- `cluster_distance` means cosine distance in kinds 1, 2a and 3a and L2 on unit vectors in 2b.
- `cluster_id` in `visual_search_faces` (a cell) and `person_id` (an identity group) answer different questions on the same document.
- Retrain, re-embed or switch `OP_RESIDUAL_EMBEDDING_FIELD` only with a full recluster: persisted IVF centroids belong to one field and are not valid for another.
- Class identity is the class name, never a cluster id or class index, across any import, combine, export or promote boundary.

## 6. How the graph refinement epic (#346) applies

The epic (children #341 evaluation set, #342 offline kNN + Leiden prototype, #343 constrained split/merge with versioned partitions, #344 continuous refinement in the refresh worker, #345 GPU and experimental algorithms) states one graph per embedding model and task, never mixed spaces. It is written for the curation subsystem: it names PE-Core vectors, the refresh worker, the OCC lock rule and Cropwright. As filed, it does not cover kinds 1 and 3, and it describes the current baseline as including "residual UMAP", while UMAP is visualization-only in the code (section 3.1).

| Kind | Starting partition the graph would reconsider | Graph nodes and space | Constraints it must honor | Fit |
|---|---|---|---|---|
| 2a item residual pool | IVF candidate ids plus AHC `cluster_subid` | items, PE-Core `pe_embedding`, residual pool only | the lock rule (`src/clients/occ_locks.py`): human- and import-validated and holdout items never move; confident-source items are outside the pool; class clusters (`cluster_id == class_id`) are trusted cannot-link anchors | primary target of #342 to #344; the worker that would run #344 is the one that retrains 2a |
| 2b region boxes | k-means buckets plus `cluster_subid`, and the `-100` FP bucket | boxes keyed `(crop_id, box_id)`, PE-Core box vectors; a graph over items would be wrong because one item holds several boxes | respect human-final items and locked boxes as the partition already does; keep the FP bucket out of the good graph; drop boxes with a stale `bbox_norm`; there is no persisted-centroid store for boxes, so no incremental assignment exists today | needs its own graph and its own evaluation set in #341; #344's "ingest-time kNN assignment" has no 2b counterpart yet because 2b runs on demand |
| 1 visual-search IVF | FAISS cell ids | images or crops, MobileCLIP 512, per index | no human labels or locks exist in these indexes, so no cannot-link source; cell ids are unstable across retrains, so "stable cluster identity" would be new | outside the filed scope; would be a separate graph and a separate decision |
| 3a faces IVF | FAISS cell ids | faces, ArcFace 512 | none | same as kind 1; a coarse index is not a refinement target |
| 3b person grouping | connected components | faces, ArcFace 512 | `person_name`, merges and deletes are trusted human constraints; a re-run today discards them | the nearest analogue of the epic's problem (single-linkage chaining, no confidence, no lineage); a mutual-kNN graph with community detection would address chaining, but nothing in #341 to #345 plans it |

Phase notes by kind: #341 must score each curation mechanism (2a, 2b) against its own public evaluation set and report false merges apart from false splits; #343's partition versioning would need a per-mechanism record because 2a and 2b store ids at different levels; #345 (GPU Leiden) shares the GPU overlay used by 2a (`docker-compose.gpu-clustering.yml`) and must coordinate devices through `OP_CLUSTER_GPU_DEVICE`.
