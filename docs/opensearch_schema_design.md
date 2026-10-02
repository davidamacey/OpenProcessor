# OpenSearch Schema Design

OpenSearch holds two families of indexes:

1. **Global visual-search indexes** written by the inference API
   (`/ingest`, `/search`, `/clusters`, `/query`).
2. **Per-project curation indexes** written by the curation subsystem
   (`/curation/projects/{project}/...`).

They never share an index. The project guard allows project-bound code to touch
only that project's indexes, and unbound code only indexes no project owns (see
[ARCHITECTURE.md](ARCHITECTURE.md#projects-and-isolation)).

Every index here is created with one shard and no replicas (single-node
default). k-NN fields use the FAISS HNSW engine with cosine similarity.

---

## 1. Global visual-search indexes

Names come from `IndexName` in `src/clients/opensearch.py`. The indexes are
created on first use by the API.

| Index | One document per | Embedding | HNSW (`ef_construction`, `m`) |
|---|---|---|---|
| `visual_search_global` | image | `global_embedding`, 512-d | 512, 16 |
| `visual_search_vehicles` | vehicle detection (car, truck, motorcycle, bus, boat) | `embedding`, 512-d | 256, 12 |
| `visual_search_people` | person detection | `embedding`, 512-d | 512, 16 |
| `visual_search_faces` | face | `embedding`, 512-d (ArcFace) | 1024, 32 |
| `visual_search_ocr` | text line | none (text search) | n/a |

### Fields

**`visual_search_global`**: `image_id` (keyword), `image_path` (keyword),
`global_embedding`, `cluster_id` (integer), `cluster_distance` (float), `width`,
`height`, `metadata` (object), `indexed_at`, `clustered_at` (dates), duplicate
detection fields `imohash` (keyword), `file_size_bytes` (long),
`duplicate_group_id` (keyword), `is_duplicate_primary` (boolean) and
`duplicate_score` (float).

**`visual_search_vehicles`**: `detection_id`, `image_id`, `image_path`,
`embedding`, `cluster_id`, `cluster_distance`, `box` (float, `[x1, y1, x2, y2]`
normalized), `class_id`, `class_name`, `confidence`, `metadata`, `indexed_at`,
`clustered_at`.

**`visual_search_people`**: as vehicles without `class_*`, plus `has_face`
(boolean) and `face_id` (keyword).

**`visual_search_faces`**: `face_id`, `image_id`, `image_path`,
`person_detection_id`, `embedding`, `cluster_id`, `cluster_distance`, `box`,
`landmarks` (object with `left_eye`, `right_eye`, `nose`, `left_mouth`,
`right_mouth`), `confidence`, `quality_score`, `person_id`, `person_name`,
`is_reference`, `metadata`, `indexed_at`, `clustered_at`, `thumbnail_b64`
(stored, not indexed).

**`visual_search_ocr`**: `ocr_id`, `image_id`, `image_path`, `text` (text with a
2-4 character n-gram analyzer for fuzzy matching, standard search analyzer),
`text_raw` (keyword), `box` (8 floats, the quadrilateral), `box_normalized`
(4 floats), `det_score`, `rec_score`, `language`, `metadata`, `indexed_at`.

### Ingest routing

`POST /ingest` writes the whole-image embedding to `visual_search_global`,
vehicle detections to `visual_search_vehicles`, person detections to
`visual_search_people`, faces to `visual_search_faces` and OCR lines to
`visual_search_ocr`. Detections of other classes are covered by the global
embedding only. `GET /query/stats` returns counts for every index and
`DELETE /query/image/{id}` removes an image from all of them.

### Search

| Route | Index searched |
|---|---|
| `POST /search/image`, `POST /search/text` | `visual_search_global` |
| `POST /search/object` | `visual_search_vehicles` or `visual_search_people`, falling back to `visual_search_global` |
| `POST /search/face`, `POST /faces/search`, `POST /faces/identify` | `visual_search_faces` |
| `POST /search/ocr` | `visual_search_ocr` |

### FAISS IVF clustering

The four embedding indexes (`global`, `vehicles`, `people`, `faces`) can be
clustered with FAISS IVF. Training learns centroids from a sample of stored
vectors; assignment of a new vector is then a nearest-centroid lookup, so
ingest does not need to re-cluster. Each document stores `cluster_id` and
`cluster_distance`. Clusters are the basis of the auto-generated albums.

Defaults (`src/services/clustering.py`):

| Index | `n_clusters` | `nprobe` | Rebalance threshold |
|---|---:|---:|---:|
| `global` | 1024 | 32 | 0.5 |
| `vehicles` | 512 | 24 | 0.5 |
| `people` | 512 | 24 | 0.5 |
| `faces` | 2048 | 64 | 0.5 |

When `n_clusters` is not given, the trainer picks the default if it lies between
`sqrt(n)` and `n/10` for `n` vectors (minimum 16), otherwise the nearest bound,
and never more than `n`.

| Route | Purpose |
|---|---|
| `POST /clusters/train/{index}` | Train an index (`n_clusters` 16-8192 and `max_samples` are optional query parameters) |
| `POST /clusters/assign/{index}` | Assign vectors to clusters |
| `GET /clusters/stats/{index}` | Sizes and counts |
| `GET /clusters/{index}/{cluster_id}` | Members of one cluster |
| `GET /clusters/balance/{index}` | Whether a rebalance is recommended |
| `POST /clusters/rebalance/{index}` | Retrain from current data |
| `GET /clusters/albums` | Clusters of the global index as albums (`min_size`, default 5) |

A rebalance is recommended when the largest non-empty cluster is more than 10
times the smallest, more than 10% of clusters are empty, or the vectors added
since training exceed the rebalance threshold fraction of the training set.
Face identities have their own routes under `/persons` (`POST /persons/cluster`,
`GET /persons`, `GET /persons/{person_id}`, `PUT /persons/{person_id}/name`,
`POST /persons/merge`, `DELETE /persons/{person_id}`).

Duplicate groups (`imohash` for exact, CLIP similarity for near duplicates) are
read with `GET /query/duplicates`, `GET /query/duplicates/stats` and
`GET /query/duplicates/{group_id}`.

---

## 2. Per-project curation indexes

Each project owns six indexes named `{OP_PROJECT_INDEX_PREFIX}{slug}__{role}`
(default prefix `op_prj_`). For project `cars`: `op_prj_cars__images`,
`op_prj_cars__items`, `op_prj_cars__labels_confirmed`,
`op_prj_cars__classes`, `op_prj_cars__umap_state`, `op_prj_cars__configs`. The
registry is the separate `op_projects` index (`OP_PROJECTS_INDEX`), and VLM
endpoints live in `op_global_configs` (`OP_GLOBAL_CONFIGS_INDEX`). Bodies are in
`src/clients/curation_opensearch.py`.

Embedding dimensions: `embedding` 512, `pe_embedding` 1024 (PE-Core),
`backbone_embedding` 1024. HNSW defaults `ef_construction` 512 and `m` 16
(`OP_HNSW_EF_CONSTRUCTION`, `OP_HNSW_M`).

### `images`

One document per ingested source image: `image_id`, `image_path`,
`source_identifier` (client name for an uploaded image), `ingest_run_id`,
`source`, `width`, `height`, `imohash`, `phash`, `indexed_at`,
`original_resolution`, `error_kind`, `embedding` (512-d) and `pe_embedding`
(1024-d, whole-frame). Import provenance: `dataset_split`, `import_ids`,
`import_source_stem`, `import_stratum`, `import_hard_negative`,
`import_label_state`, `negative_for`. Combine provenance: `origin_project`,
`origin_image_id`, `origin_split`.

### `items`

One document per crop. Field groups:

| Group | Fields |
|---|---|
| Identity and geometry | `crop_id`, `image_id`, `image_path`, `source`, `request_id`, `bbox_norm` (4 floats, the crop in the source image), `crop_area_norm`, `crop_rank_in_image`, blur fields |
| Class | `class_id`, `class_name`, `class_source`, `class_validated`, `class_detector`, `class_detector_version`, `class_labeler`, `class_labeled_at`, `confidence`, `proposal_name`, `label_source`, `class_id_history` (stored, not indexed) |
| VLM | `vlm_endpoint`, `vlm_model`, `vlm_prompt_pack`, `vlm_confidence`, `vlm_raw_class`, `vlm_proposed_class`, `needs_new_class`, `vlm_raw_label*`, label-cluster fields |
| Clusters | `cluster_id`, `cluster_subid`, `cluster_distance`, `cluster_nearest_id`, `cluster_auto_suggest` |
| Embeddings | `pe_embedding` (1024-d), `backbone_embedding` (1024-d) |
| Region summary | `region_status`, `region_reason`, `region_validated`, `region_auto_confirmed`, `region_verified`, `region_verifier`, `region_verifier_version`, `region_verified_at`, `region_visible`, `region_detector_chain`, `region_detected_at`, `region_profile`, `region_profile_revision`, `region_class_id`, `region_label_source`, `region_pairing`, `region_skip_verify`, `region_rejection_reason` |
| Region boxes | `region_boxes` (nested), `region_box_embeddings` (nested), `region_count`, `region_rejected_count`, `region_max_score`, `region_set_complete`, `region_revision`, `region_box_seq` |
| Item text | OCR lines read on the crop and their search tokens (`item_text_lines`) |
| Review state | `test_holdout`, `class_excluded` and `excluded_*`, `review_dismissed_*`, `vlm_dismissed_*`, `edit_history` (stored, not indexed) |
| Probe and scores | `probe_pred_*`, `probe_disagreement`, `probe_model_version`, `uniqueness_*`, `mistakenness_*`, `dup_*` |
| Import and combine | `dataset_split`, `import_ids`, `imported_at`, `import_dataset_name`, `import_dataset_sha`, `proposed_by_import`, `on_negative_frame`, `import_standalone_region`, `proposal_chain`, `origin_project`, `origin_item_id`, `origin_image_id`, `origin_split`, `combine_conflict`, `combine_conflict_origins`, `combine_merged_origins` |
| Timestamps | `created_at`, `updated_at` |

The `region_*` names are the default storage names. `RegionFields`
(`src/config/region_fields.py`) can override a storage name without a reindex;
the HTTP wire names do not change. The list of wire keys is
[`contracts/json/item_wire.json`](../contracts/json/item_wire.json).

#### `region_boxes` (nested)

`region_boxes` is a **nested** list, one element per region box. N=1 is a list
of one. It is nested so that a query like "a box whose detector is X and state
is accepted" matches one box, and every box filter of `GET /regions` is applied
to the same box. Element keys are fixed strings:

| Key | Type | Meaning |
|---|---|---|
| `box_id` | keyword | `b1`, `b2`, ...; never reused within an item |
| `bbox_norm` | float x4, not indexed | `[x1, y1, x2, y2]` in the item crop's frame |
| `state` | keyword | `proposed`, `accepted`, `rejected`, `false_positive` |
| `score` | float | detector score |
| `detector`, `detector_version`, `source` | keyword | who produced the box (a human edit sets the human detector name) |
| `bbox_correct` | boolean | a reviewer's geometry verdict |
| `confidence` | keyword | VLM confidence bucket |
| `rejection_reason` | keyword | why a verifier or human rejected it |
| `text`, `text_raw` | keyword | the read text, and the text before normalization |
| `text_source`, `text_engine_version`, `text_confidence` | keyword, keyword, float | where the text came from |
| `text_vlm`, `text_ocr`, `text_choice`, `text_vlm_invalid`, `text_disagreement` | keyword, keyword, keyword, keyword, boolean | per-reader outputs and which was chosen |
| `cluster_id`, `cluster_subid`, `cluster_distance` | integer, keyword, float | the box's own cluster placement |
| `detected_at` | date | when the box was written |

`region_box_embeddings` is a sibling nested field with `box_id`, `bbox_norm` (the
geometry the vector was computed from) and `embedding` (1024-d k-NN). It is kept
apart so that rewriting the box list cannot drop vectors, and every read that
feeds the wire excludes it. A vector whose `bbox_norm` no longer matches its box
is stale and is dropped on write.

Every k-NN index keeps OpenSearch's derived source (no `knn.derived_source.enabled`
override), so a vector is stored twice (HNSW graph and flat copy, about 8.4 KB at
1024-d) and not a third time as JSON text in `_source` (about 25 KB). Measured with
this mapping: 8.4 KB per vector for `pe_embedding` and for each nested box vector.
One trap: a search whose `_source` names the bare `region_box_embeddings` path gets
the number `1` for each vector; readers use
`region_box_embeddings.box_vector_source_includes` (the leaf paths), which returns
the real values. `_source` excludes, `GET`, `mget` and `_update` are unaffected.

`region_revision` is bumped by any write that changes a box's state, geometry or
text (a cluster-only write does not). Editors send it back as
`expected_region_revision` to detect a concurrent change.

Routes: `GET /curation/projects/{project}/regions` lists one row per box;
`PUT /curation/projects/{project}/crops/{crop_id}/regions` and
`PATCH /curation/projects/{project}/crops/{crop_id}/regions/{box_id}` edit them;
`POST /curation/projects/{project}/regions/cluster` clusters boxes by their own
embeddings.

### `labels_confirmed`

The mapping is kept (`label_id`, `crop_id`, `image_path`, `bbox_norm`,
`class_id`, `class_name`, `label_source`, `class_source`, `confirmed_at`,
mismatch fields) but nothing writes to it in this release. Dataset imports keep
their own ledger on disk.

### `classes`

The class registry mirror: `class_id` (integer), `class_name` (keyword),
`group` (keyword), `sample_count`, `validated_count`, `added_at`, `deprecated`,
`notes`. The registry file `class_registry.json` under the project's data root is
the source of record; `POST /curation/projects/{project}/classes/sync_to_opensearch`
refreshes the mirror. Class identity across datasets is the name, not
`class_id`.

### `umap_state`

The clustering reducer cache: `state_id`, `reducer_b64` (binary), `n_components`,
`metric`. Mapped with `dynamic: false`.

### `configs`

The config store, one document per row, discriminated by `doc_type`
(`config`, `revision`, `activation`, `activation_event`, `meta`, `runtime`):
`kind` (`prompt_pack`, `region_profile`, ...), `name`, `revision`, `body`
(stored, not indexed), `description`, `created_at`, `updated_at`, `updated_by`,
`cloned_from`, `axis`, `previous`, `config_revision`, `process`, `applied_at`.
The project's settings document (`defaults`, addressed by id `default`) and the
UMAP projection state (id `current`: `projection_version`, `scope`, `n_points`,
`fitted_at`, ...) are stored in this same index, by fixed id and never searched,
so a project needs six indexes rather than eight.

---

## 3. Operations

```bash
make opensearch-status          # cluster health
make opensearch-indices         # list indexes
```

`make opensearch-reset` sends a delete-all-indexes request straight to OpenSearch: it removes
**every** index, including every project's indexes and the project registry. It
bypasses the project guard because it does not go through the API. Delete a
single project with `DELETE /curation/projects/{project}` instead.

Shard budget: every project adds six single-shard indexes. The installer sets
the OpenSearch heap from host RAM and a soft shard budget of 20 shards per heap
GB (`OP_SHARDS_PER_HEAP_GB`); creating a project past the hard limit is refused
with `shard_budget_exceeded`. See
[INSTALLATION.md](../INSTALLATION.md#opensearch-heap-sizing).

`index.knn` is a final setting: turning it on for an existing index needs a
reindex.
