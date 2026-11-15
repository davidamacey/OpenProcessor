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

Names come from `IndexName` in `src/clients/opensearch/names.py`. The indexes are
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
`is_reference`, `metadata`, `indexed_at`, `clustered_at`, `named_at`,
`merged_at`, `merged_from`, `cluster_assigned_at`, `thumbnail_b64` (stored, not
indexed). A face with `named_at` or `merged_at` is locked: person clustering
never rewrites its `person_id` or `person_name`.

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
| `POST /search/object` | `visual_search_vehicles` or `visual_search_people`; a box of any other class is refused with `422 object_class_not_indexed` (the global index holds whole-image vectors) |
| `POST /search/face`, `POST /faces/search`, `POST /faces/identify` | `visual_search_faces` |
| `POST /search/ocr` | `visual_search_ocr` |

A class or cluster filter on a legacy visual-search query sits inside the `knn` clause
(`knn[field].filter`), so the k nearest are taken from the matching documents and a
restrictive filter still returns `k` hits. Wrapped in `bool.filter` beside the `knn` clause,
the filter runs after the k nearest are chosen and returns fewer
([filtering in k-NN search](https://docs.opensearch.org/latest/vector-search/filter-search-knn/)).

### FAISS IVF clustering

The four embedding indexes (`global`, `vehicles`, `people`, `faces`) can be
clustered with FAISS IVF. Training learns centroids from a sample of stored
vectors; assignment of a new vector is then a nearest-centroid lookup, so
ingest does not need to re-cluster. Each document stores `cluster_id` and
`cluster_distance`. Clusters are the basis of the auto-generated albums.

Defaults (`src/services/visual_search/clustering_models.py`):

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
With `OP_FACES_ENABLED=false` ingest writes no faces and the face, person and `faces` cluster routes answer `400 faces_disabled`. Face identities have their own routes under `/persons` (every face gets a person and re-runs keep ids stable; see [FACE_RECOGNITION_IMPLEMENTATION.md](FACE_RECOGNITION_IMPLEMENTATION.md)) (`POST /persons/cluster`,
`GET /persons`, `GET /persons/{person_id}`, `PUT /persons/{person_id}/name`,
`POST /persons/merge`, `DELETE /persons/{person_id}`).

Duplicate groups (`imohash` for exact, CLIP similarity for near duplicates) are
read with `GET /query/duplicates`, `GET /query/duplicates/stats` and
`GET /query/duplicates/{group_id}`.

---

## 2. Per-project curation indexes

Each project owns the indexes `{OP_PROJECT_INDEX_PREFIX}{slug}__{role}` (default prefix
`op_prj_`) for the roles `images` and `items` only. For project `cars`: `op_prj_cars__images` and
`op_prj_cars__items`. Every other control record (projects, classes, configs, settings, jobs,
embedding spaces, cluster partitions, near-duplicate groups) is in PostgreSQL. Bodies are in
`src/clients/curation_opensearch/` (`bodies_core.py` for images and items, with the field groups in
`cluster_fields.py`, `near_dup_fields.py`, `capture_fields.py`, `count_fields.py`, `assist_fields.py` and
`open_vocab_fields.py`). Indexes are created and migrated once by the `db-migrate` step
(`scripts/opensearch/migrate.py`); the API only checks the schema stamp of each mapping and refuses
to start with `opensearch_mapping_not_current`.

An index created before a field existed gets it from an additive `PUT _mapping`
migration (`src/services/curation/index_bootstrap.py`), run by `db-migrate`. After a full run every
index's mapping `_meta` holds `op_schema`, a fingerprint of the bodies and the migration list; the
API and the workers read the markers in one request and refuse to start unless all are current. A
new body field or migration changes the fingerprint, so every project migrates once more.

A removed field is dropped from the bodies and the writers only; OpenSearch cannot unmap a
field, so an index created earlier keeps the dead mapping (and any stored values) until it is
rebuilt. Nothing reads or writes it. `backbone_embedding` (retired in 0.7.0) is the example:
its mapping stays on older indexes, new indexes never get it, and `OP_BACKBONE_EMBEDDING_DIM`
is a retired env var.

Embedding dimensions: `embedding` 512, `pe_embedding` 1024 (PE-Core). HNSW on the Lucene engine (new projects; earlier projects keep faiss), defaults `ef_construction` 128 and `m` 16
(`OP_HNSW_EF_CONSTRUCTION`, `OP_HNSW_M`).

**Vector fields are named by the registry.** A vector field in `images` or `items` belongs to an
embedding **space** (`embedding_spaces` in PostgreSQL): the row stores the physical field name
(`vector_field`), the optional state field (`state_field`, the per-item `embedded` / `not_selected` /
`deferred` / `failed` keyword), the nested path for a box space and the k-NN method. Code takes the
names from a resolved space and never builds them. A project's default spaces keep the 0.6.0 names
(`pe_embedding`, `embedding_state`, `region_box_embeddings` with the leaf `embedding`), so nothing
moves on upgrade; a candidate space adds a field with a registry-generated name by an additive
`PUT _mapping`, and retiring it leaves the unmapped values until the index is rebuilt. At most
three live spaces per target.

### `images`

One document per ingested source image: `image_id`, `image_path`,
`source_identifier` (client name for an uploaded image), `ingest_run_id`, `decode_route` (keyword, written once at create: `gpu` or `cpu_<reason>`, see `src/domain/decode_route.py`),
`source`, `width`, `height`, `imohash`, `phash`, `indexed_at`, `capture` (object: `taken_local`, `taken_utc`, camera, `has_location`, `exif_status`, `parser`, and under the project's location policy the `geo_point` `location` with altitude, accuracy, heading and `location_precision`; items hold only `capture_taken_local` and `capture_taken_utc`),
`original_resolution`, `error_kind`, `embedding` (512-d) and `pe_embedding`
(1024-d, whole-frame). Import provenance: `dataset_split`, `import_ids`,
`import_source_stem`, `import_stratum`, `import_hard_negative`,
`import_label_state`, `negative_for`. Combine provenance: `origin_project`,
`origin_image_id`, `origin_split`.

Ingest and dataset import name the document by its content: `image_id` (also the
`_id`) is `sha256(imohash)[:32]` (the SHA-256 of the bytes instead, for a file whose
imohash matched a different stored file), written with `create`. The content lookup
(`image_contents`: imohash first, SHA-256 only to confirm a match) before
the write cannot see a copy another request wrote but the index has not yet
refreshed, so the `create` is what keeps one image per content: the copy that gets
409 is reported `duplicate` and stores no items or crops. The writer stamps each
images document with a fresh `ingest_write_id` (stored, not indexed): when a
transport retry re-sends a bulk whose first attempt was applied, the 409 finds the
writer's own write id and the image counts as written, so its items are written instead
of being dropped as a duplicate. Images written before this
rule keep their path-and-hash ids; the lookup still finds them, so they need no
migration. A combine target names its images by target path and hash instead.

A photo is stored whole or not at all. The images doc is written before its items, so it
lists them: `item_ids` (the unit's crop ids, sorted; stored, not indexed; `[]` for a photo with
no detections) is written with the doc, and a stored image is a `duplicate` only when every
listed item exists (one real-time `mget` on the items index, in the content lookup and in the
writer's answer to a 409). The list only names items that exist: the one place items are deleted
(`item_delete.delete_items`: a re-detection, an open-vocabulary run, an import undo) takes the
ids off it first, and a combine does not copy it. A doc without `item_ids` is legacy and counts
as whole, always (`scripts/curation/backfill_item_ids.py` lists the items of those docs; it
cannot tell an old broken photo from one with no detections and does not try). An incomplete
image is completed by the next ingest of the same bytes when no live writer can still be writing
it: the writer that saw its own write fail set `ingest_incomplete` (stored, not indexed), or the
doc is older than `REPAIR_AFTER_S` (`src/services/curation/ingest_writer.py`, derived from the
OpenSearch client's timeout and retries). The repair writes the items (the same OCC upsert, human
guards included), then sets `item_ids` and clears the mark in one OCC update; it never replaces
the images doc, and the stale items the old list named are deleted only when no person or import
validated them and they are not in the test holdout. A stored image that is incomplete and may
still be written is answered `failed` with `error_kind` `incomplete_write` (send it again later).
`scripts/bench/ingest_consistency.py` audits a project: every listed item exists, every item has
its images doc, every exact-content row has its images doc.

### `items`

One document per crop. Field groups:

| Group | Fields |
|---|---|
| Identity and geometry | `crop_id`, `image_id`, `image_path`, `source`, `request_id`, `bbox_norm` (4 floats, the crop in the source image), `crop_area_norm`, `crop_rank_in_image`, blur fields |
| Class | `class_id`, `class_name`, `class_source`, `class_validated`, `class_detector`, `class_detector_version`, `class_labeler`, `class_labeled_at`, `confidence`, `proposal_name`, `detector_class_name` (keyword), `detector_class_id` (integer), `detector_confidence` (float), `label_source`, `label_op_id` (keyword: the bulk label or move operation that wrote the current label), `class_id_history` (stored, not indexed; entries carry `op_id`) |
| VLM | `vlm_endpoint`, `vlm_model`, `vlm_prompt_pack`, `vlm_confidence`, `vlm_raw_class`, `vlm_proposed_class`, `needs_new_class`, `vlm_raw_label*`, label-cluster fields |
| Clusters (the served partition) | `cluster_partition_id` (keyword), `cluster_uid` (keyword), `cluster_state` (`assigned`, `review`, `unassigned`), `cluster_score`, `cluster_margin` (float), `cluster_assigned_by` (keyword: the run or `incremental`). One projection writer owns them; a box element carries the same keys |
| Near-duplicates (also on `images`) | `near_dup_group_id`, `near_dup_leak_group_id` (keyword), `near_dup_is_representative` (boolean), `near_dup_group_size` (integer), `near_dup_run_id` (keyword). `hide_near_duplicates` reads `near_dup_is_representative` and keeps an item without the field |
| Embeddings | `pe_embedding` (1024-d) and `embedding_state` for the default space, or the fields of each registered space (keyword states: `embedded`, `not_selected`, `deferred`, `failed`; absent on older items, so "has a vector" is always the `exists` test on the vector field) |
| Region summary | `region_status`, `region_reason`, `region_validated`, `region_auto_confirmed`, `region_verified`, `region_verifier`, `region_verifier_version`, `region_verified_at`, `region_visible`, `region_detector_chain`, `region_detected_at`, `region_profile`, `region_profile_revision`, `region_class_id`, `region_label_source`, `region_pairing`, `region_skip_verify`, `region_rejection_reason` |
| Region boxes | `region_boxes` (nested), `region_box_embeddings` (nested), `region_count`, `region_rejected_count`, `region_max_score`, `region_set_complete`, `region_revision`, `region_box_seq` |
| Item text | OCR lines read on the crop and their search tokens (`item_text_lines`) |
| Item geometry | `geometry_source` (keyword: `human` when a person drew or moved the box), `geometry_revision` (integer: the optimistic-concurrency token of a box edit, 0 or absent when never edited), `detector_bbox_norm` (float x4: the machine's box kept on the first human move) |
| Review state | `test_holdout`, `class_excluded` and `excluded_*`, `review_dismissed_*`, `vlm_dismissed_*`, `edit_history` (stored, not indexed) |
| Probe and scores | `probe_pred_*`, `probe_disagreement`, `probe_model_version`, `uniqueness_*`, `mistakenness_*` |
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
| `text_source`, `text_confidence`, `text_conf_char_min` | keyword, float, float | where the text came from, and its line and per-character confidence |
| `text_reader`, `text_reader_version` | keyword, keyword | the Triton model (or `vlm`) that read the text, and its version |
| `text_frame` | keyword | this frame's own reading |
| `text_norm`, `text_fold` | keyword with the `op_text_compact` normalizer; `text_fold.sub` is a `wildcard` | storage-only match keys: the compact upper-case text, and that after the profile's confusable map; exact and fuzzy search read `text_fold`, fragment search `text_fold.sub` |
| `text_vlm`, `text_ocr`, `text_choice`, `text_vlm_invalid`, `text_disagreement` | keyword, keyword, keyword, keyword, boolean | per-reader outputs and which was chosen |
| `cluster_partition_id`, `cluster_uid`, `cluster_state`, `cluster_score`, `cluster_margin`, `cluster_assigned_by` | keyword, keyword, keyword, float, float, keyword | the box's own served cluster placement |
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
the real values. `_source` excludes are unaffected.

#### Nested documents per item: the limit of 150

OpenSearch k-NN issue 3623 (<https://github.com/opensearch-project/k-NN/issues/3623>, fixed by
PR 3624 for 3.10, not released; every 3.x is affected, measured on 3.6.0 with this mapping): with
derived source for vectors, a parent document with 150 or more nested Lucene documents, counted
across ALL its nested fields, is read back with another document's nested vectors on search,
`GET` and `mget`, and any partial update, scripted update, `update_by_query` or reindex then
writes those wrong vectors to disk. At 74 boxes (148 nested documents) none of 200 items was
wrong; at 75 boxes (150) 96% were. The item's own `pe_embedding` is never wrong. Turning derived
source off avoids it at 20.5 KB stored per vector instead of 4.2 KB.

One item document holds these nested documents: `region_boxes` (one per box), every box-vector
list (`region_box_embeddings`, and one `boxvec_<key>` per further box space: one entry per
embedded box each), `assist.attributes` and `assist.boxes` with their `attributes` (the last are
nested inside a nested field; Lucene gives each its own document, the case was not measured
separately). With the default configuration each box takes two (the box and its vector), so one
item holds at most 74 boxes (148 documents); a second box space lowers that to 49, a third to 37.
The rule, `src/domain/nested_limit.py`:

- A write whose result holds 150 or more nested documents is refused (the edit routes answer 422
  `item_over_nested_limit` with the limit in the body; the region worker records the item as
  `detection_failed` with that reason; an import reports the image as failed; a combine does not
  copy the item and lists its id in `over_nested_limit_ids`). A box list is never truncated.
- The count assumes one vector per box in every box-vector list, because reads leave the vectors
  out; the limit therefore holds before the embed stage runs.
- An item that already holds 150 or more (written before the limit) accepts no write but one: a
  box-list write that brings it under the limit, which also empties its box vectors (they are
  read back wrong); the box embed stage and `backfill_region_embeddings.py` re-create them.
  `scripts/curation/find_items_over_nested_limit.py --project <slug> [--repair-plan]` lists such
  items and exits 1 when any exist.
- The cluster projection scripts and the embedding-state backfill leave such an item as it is and
  count it (`skipped_over_nested_limit`, `legacy_repair_skipped_over_nested_limit`).

These writers send partial updates to item documents outside the guarded paths. They are only
reachable for an item that already holds 150 or more nested documents (no guarded writer can
take an item there), and for such an item they persist wrong box vectors; repair those items first:

| Writer | File | What it writes |
|---|---|---|
| needs-new-class flag | `src/routers/curation/crops_write.py` | `needs_new_class*` |
| review dismissal | `src/services/curation/label_undo.py` (`review_dismiss`) | `review_dismissed_*` |
| probe predictions | `src/services/curation/probe_predictions.py` | `probe_pred_*` |
| scatter coordinates | `src/services/curation/embedding_viz.py` | `viz_x`, `viz_y` |
| scores | `src/services/curation/item_scores/job.py` | `uniqueness_*`, `mistakenness_*` |
| near-duplicate projection | `src/services/near_duplicates/projection.py` | `near_dup_*` (bulk, clear script, `update_by_query`) |
| source backfill | `src/services/source_backfill/capture.py`, `location_purge.py` | `capture*` (bulk, `update_by_query`) |
| embedding space re-index | `src/services/embedding_spaces/reindex_opensearch.py` | the new space's vectors (bulk, `update_by_query`) |
| 0.6.0 rollback | `src/services/upgrade/from_060/rollback.py` | restores exported documents |

`region_revision` is bumped by any write that changes a box's state, geometry or
text (a cluster-only write does not). Editors send it back as
`expected_region_revision` to detect a concurrent change.

Routes: `GET /curation/projects/{project}/regions` lists one row per box;
`PUT /curation/projects/{project}/crops/{crop_id}/regions` and
`PATCH /curation/projects/{project}/crops/{crop_id}/regions/{box_id}` edit them;
Clustering of boxes by their own embeddings runs through
`POST /curation/projects/{project}/clusters/runs`.

### Retired roles and fields

The `classes`, `configs`, `settings`, `umap_state`, `umap_viz_state`, `labels_confirmed` and `leases`
index roles, the `op_projects` and `op_global_configs` indexes are gone: their data is in PostgreSQL
(the class registry in `class_registries` and `project_classes`, the config store in
`config_documents`, `config_revisions` and `config_activations`, settings in `project_settings`,
jobs and their leases in `jobs`). A project index created by 0.6.0 keeps the old roles only until
`./openprocessor upgrade` has moved and verified the data, and then they are deleted. OpenSearch
cannot unmap a field, so an index created earlier keeps these field mappings (and any values) until
it is rebuilt, and nothing reads or writes them:

| Retired field | Replaced by |
|---|---|
| `cluster_id`, `cluster_subid`, `cluster_distance`, `cluster_nearest_id`, `cluster_auto_suggest` (items and boxes); the label-cluster fields | `cluster_partition_id`, `cluster_uid`, `cluster_state`, `cluster_score`, `cluster_margin`, `cluster_assigned_by` |
| `backbone_embedding` | nothing; `OP_BACKBONE_EMBEDDING_DIM` is a retired env var |
| `dup_group_id`, `dup_group_size`, `dup_is_representative`, `dup_threshold`, `dup_method`, `dup_scored_at` (items) | the `near_dup_*` projection; the upgrade clears them |

The class list route joins its counts from one terms aggregation on `items.class_id`. Class
identity across datasets is the name, not `class_id`.

#### The class registry

The class registry is two PostgreSQL tables per deployment, `class_registries` (one row per project:
the id counter and a revision) and `project_classes` (the classes). See
[02a](design/v070_rebuild/02a_postgres_schema.md) section 3. Ids come from `next_class_id` and are
never reused; `ensure_class` is idempotent by name and is what the automated writers (import,
combine, the VLM and open-vocabulary passes) call; `add_class` keeps the route's 409 on a duplicate
name. `class_registry.json` is only an export snapshot and the training pin.

---

## 3. Operations

```bash
make opensearch-status          # cluster health
make opensearch-indices         # list indexes
```

`make opensearch-reset` sends a delete-all-indexes request straight to OpenSearch: it removes
**every** index, including every project's indexes. It
bypasses the project guard because it does not go through the API. Delete a
single project with `DELETE /curation/projects/{project}` instead.

Shard budget: every project adds two indexes (`images` and `items`), each with one primary shard
and no replicas, so two shards. `active_shards` on `GET /curation/projects` counts every index in
the cluster (the other projects, the `default` project's indexes and the legacy search
indexes), not only the projects you made. The installer sets
the OpenSearch heap from host RAM and a soft shard budget of 20 shards per heap
GB (`OP_SHARDS_PER_HEAP_GB`); creating a project past the hard limit (`cluster.max_shards_per_node` times
the data nodes) is refused with `shard_budget_exceeded`; past the soft budget it succeeds with a
`shard_budget_high` warning. The `capacity` block (and that warning's `detail`) names the
binding limit (`limit_source`: `heap` or `cluster_max_shards_per_node`) and the total after a
create (`shards_after_create`). See
[INSTALLATION.md](../INSTALLATION.md#opensearch-heap-sizing).

`index.knn` is a final setting: turning it on for an existing index needs a
reindex.
