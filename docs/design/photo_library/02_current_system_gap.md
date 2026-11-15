# 02: Photo library: what the current system provides, and the gap

Status: analysis, 2026-11-13, written against branch `plan/photo-library` (same tree as
`int/b4-stateless`). Research only: nothing here is decided. Audience: the planner and the owner,
and a fresh agent with no conversation memory.

Notation: `{P}` = `/curation/projects/{project}`. Routes that do not exist yet are written without a
leading slash and marked **proposed**. File paths are relative to the repo root. "Library" means a
photo-library experience of the Apple Photos / Google Photos / Immich class inside Cropwright.

**The owner's vision, in one line.** OpenProcessor is a local ML engine that finds and records
faces, people, objects, places, time and text in a photo library with the originals left in place;
one processed library serves (1) labelling and training, (2) organising (timeline, albums, smart
albums, People, Places), (3) curating (find people or objects and exclude them from library views)
and (4) creating (calendar-aware smart albums, memories). Images only today; video later.

**Headline findings.**

1. The curation stack already provides most of the *engine* a library needs: content-addressed
   images with originals in place, EXIF capture time and GPS on every image, a server-aggregated map,
   item-level PE-Core vectors with text, image and "similar" search, near-duplicate groups with human
   decisions, stable cluster identities, an undoable bulk label-op log, a shared filter grammar and
   cursor paging everywhere.
2. **Faces are the biggest hole.** SCRFD + ArcFace, person grouping, naming, merge and purge exist,
   but only in the deployment-wide core index `visual_search_faces`, outside project isolation. The
   curation ingest refuses the `faces` branch (`src/domain/processing_plan.py:80`: "No curation store
   exists for faces, text or duplicate groups"). Cropwright has no People UI at all.
3. Everything in the curation model is organised around **items (detected objects)**; the library's
   unit is the **photo**. A predicate about items is resolved to photos through a capped join
   (`JOIN_MAX_IMAGES = 10_000`, `src/services/curation/image_join.py:30`), which caps "photos
   containing X" at 10,000 photos.
4. No album, favourite, hidden, trash, rating, memory or calendar concept exists anywhere
   (`rg -i 'album|favou?rite|trash|memor|calendar' src` finds only the core FAISS "albums" in
   `src/routers/clusters.py`).
5. The scale shape is not yet library-sized: one shard per project index
   (`src/clients/curation_opensearch/base.py:72`), thumbnails rendered on demand from the original
   with a per-process LRU of 2,000 entries (`src/services/curation/image_serving.py:278`), and
   infinite-scroll grids that keep every loaded card in the DOM (Detections reaches row 10,000 in
   about 100 s on an 18,675-item project, re-walk 2026-11-13).

---

## 1. Data model today

### 1.1 The project concept (per-project indexes and registries, and why)

| Aspect | Today | Files |
|---|---|---|
| What a project is | A named, isolated dataset workspace for one labelling job. Row in `projects` (slug, status `building/active/archived/deleting/failed`, `index_prefix`, `storage_prefix`, `model_prefix`, `mlflow_experiment`, `lock_key`, `revision`, `paused_at`); slugs are never reused (`project_slugs`, `409 slug_retired`) | `src/db/models/projects.py`, `src/services/projects/lifecycle.py`, `registry.py`, `records.py` |
| Indexes | Exactly two per project: `{OP_PROJECT_INDEX_PREFIX}{slug}__images` and `__items`, 1 shard, 0 replicas; names computed once and stored in `project_indexes` | `src/clients/curation_opensearch/base.py`, `bodies_core.py`, `src/db/models/projects.py` (`project_indexes`) |
| Control data | PostgreSQL rows carrying `project_id`, under forced row-level security; a transaction binds one project (`project_tx`) | `src/db/session.py:155`, `src/db/migrations/schema_0001/security.py:127` |
| Registries | Class registry per project (`class_registries`, `project_classes`, ids from `next_class_id`, never reused; `ensure_class` idempotent by name), keymap, settings, config-store activations (prompt packs, region/ingest profiles, cluster, near-dup, loop presets), embedding spaces, cluster partitions | `src/db/models/classes.py`, `config.py`, `embeddings.py`, `clustering.py` |
| Files | Exports and eval sets under `OP_PROJECTS_DATA_ROOT/<slug>/`; uploads under `OP_STATE_DIR/projects/<slug>/`; tmpfs crop cache `<crop_cache_dir>/<slug>/` | `src/services/curation/source_image_cache.py`, `src/services/projects/delete_paths.py` |
| The guard | Transport-level wrapper on every OpenSearch client: allowlisted request shapes only, every named index must belong to the bound project; wildcards, `_all`, aliases, `_reindex`, index-less searches raise `CrossProjectAccess`. Unbound code may touch only indexes no project owns (`visual_search_*`) | `src/services/projects/guard.py`, `guard_policy.py`, `guard_parse.py`, `src/config/project_context.py` |
| Cross-project paths | Combine copies 1-8 sources into a **new** project (class mapping by name, images merged by content hash); clone settings; share a trained model (`models/{model_name}/sharing`). Nothing reads two projects at once | `src/services/projects/combine/`, `clone*.py`, `src/routers/curation/_models_sharing.py` |
| Capacity | No project-count cap; shard and heap budget checked, create answers 409 past it | `src/services/projects/capacity.py` |
| **Why** | Fail-closed isolation is a v0.6.0 "done" criterion (`docs/VISION_AND_GOALS.md`): each labelling job has its own class scheme, holdout, models and indexes, and nothing can leak across a boundary | `docs/ARCHITECTURE.md` "Projects and isolation" |

Identity that crosses projects already: `image_id` is content-derived (`sha256(imohash)[:32]`,
`docs/opensearch_schema_design.md` "images"), and crop ids are content-derived too
(`source_image_cache.py` docstring). The same photo ingested into two projects gets the same
`image_id`. A library layer can join by it.

### 1.2 Images and items

| Store | Unit | Key fields (library-relevant) | Files |
|---|---|---|---|
| OpenSearch `__images` | One doc per source photo | `image_id` (=`_id`, content key), `image_path` (container path), `source`, `source_identifier` (upload's client name), `ingest_run_id`, `width`, `height`, `imohash`, `phash`, `indexed_at`, `capture.*` (1.3), `embedding` (512-d, empty on new docs per re-walk), `pe_embedding` (1024-d whole frame), `counts.*` + `counts_at` (stored per-image item counts), `near_dup_*` projection, import and combine provenance | `src/clients/curation_opensearch/bodies_core.py`, `capture_fields.py`, `count_fields.py`, `near_dup_fields.py` |
| OpenSearch `__items` | One doc per detected object (crop) | `crop_id`, `image_id`, `bbox_norm`, class (`class_id`, `class_name`, `class_source`, `class_validated`, `detector_class_name`), `capture_taken_local/utc` (no coordinates ever), cluster projection (`cluster_partition_id`, `cluster_uid`, `cluster_state`), `near_dup_*`, `pe_embedding` + `embedding_state`, `region_boxes` (nested) + `region_box_embeddings`, `item_text_lines/tokens`, review state (`class_excluded`, `excluded_*`, `review_dismissed_*`, `test_holdout`), `label_op_id`, `class_id_history`, `edit_history` | same; wire keys in `contracts/json/item_wire.json`; serializer `src/services/curation/wire.py` |
| PostgreSQL | Control rows | `image_contents` (per-project content key -> image), `near_dup_*`, `cluster_*`, `label_operations` + `label_operation_members`, `holdout_freezes`, `ingest_sources` + `ingest_source_files`, `jobs`, `events`, `privacy_purges` | `src/db/models/near_dup.py`, `clustering.py`, `datasets.py`, `jobs.py`, `ops.py` |

Density measured in the 2026-11-13 re-walk: 52 street photos produced 669 items (about 13 per
photo). A photo with no detection has an images doc and no items, so every item-driven view
(Detections, Clusters, Review, item filters, semantic search) never shows it; only Gallery, Table
(image rows) and Map list it.

### 1.3 Capture metadata (EXIF time, GPS, camera)

| Field / feature | Today | Files |
|---|---|---|
| Reader | ExifTool through one `-stay_open` process per worker (PyExifTool), timeout per read, restart on failure; missing tag never fails ingest; `exif_status unknown` when ExifTool is down, the `capture` step of `source_backfill` re-reads later | `src/clients/exiftool_client.py`, `src/domain/capture_metadata.py`, `src/services/curation/capture_read.py` |
| Time | `capture.taken_local` (floating wall-clock), `capture.taken_utc` only when the offset is known (`OffsetTimeOriginal` or GPS time), `offset_minutes`, `time_source`, `offset_source`, `time_precision`; `ModifyDate` is never capture time | `src/clients/curation_opensearch/capture_fields.py` |
| Camera | `camera_make`, `camera_model`, `lens_model`, `focal_length_mm`, `focal_length_35mm`, `orientation` | same |
| Position | `capture.location` (`geo_point`), `altitude_m`, `location_accuracy_m`, `heading_deg`, `location_precision`, `has_location`; **images only, never copied to an item, label, manifest, VLM request or log** | same; `src/domain/location_policy.py` |
| Location policy | Per project `store` / `store_coarse` (0.01 degree) / `drop`; tightening starts a `location_purge` job; read-time enforcement too | `{P}/location_policy`, `src/services/curation/location_policy_store.py`, `docs/CURATION.md` "Location policy" |
| Filters | Shared item filter: `taken_from`, `taken_to` (inclusive days), `geo_bbox`, `geo_polygon`, `has_location` | `src/routers/curation/_item_filter_params.py` |
| Sorts | `GET {P}/images`: `taken_at:desc/asc` ("Newest/Oldest captured", missing last), `indexed_at`, `file_name`, unlabelled-count | `src/services/curation/image_view.py:66` |
| Map | `GET {P}/map` (availability), `GET {P}/map/cells` (`geotile_grid` + `geo_centroid`, at most 5,000 cells, GeoJSON, `n_located_matching`/`n_unlocated_matching`) | `src/routers/curation/map.py`, `src/services/curation/map_cells.py` |
| Not present | No reverse geocoding (no place names), no time histogram (`date_histogram` is used only on `indexed_at` in `ingest_reads.py`), no time-zone inference beyond EXIF/GPS, no camera facet route | |

### 1.4 The faces pipeline

| Aspect | Today | Files |
|---|---|---|
| Models | SCRFD-10G (boxes + 5 landmarks), Umeyama alignment to 112x112, ArcFace w600k_r50 (512-d, L2-normalised), FP16 Triton engines | `docs/FACE_RECOGNITION_IMPLEMENTATION.md`, `models/scrfd_10g_bnkps/`, `models/arcface_w600k_r50/` |
| Where faces are stored | **Only** the deployment-wide core index `visual_search_faces` (written by core `POST /ingest`), outside project isolation | `src/services/face_identity/`, `docs/opensearch_schema_design.md` section 1 |
| Per face | `face_id`, `image_id`, `image_path`, `person_detection_id`, `embedding` (512-d), `box`, `landmarks`, `confidence`, `quality_score`, `person_id`, `person_name`, `is_reference`, `cluster_id`, `named_at`, `merged_at`, `merged_from`, `cluster_assigned_at`, `thumbnail_b64` (stored, not indexed) | same |
| Grouping | `similarity_components`: exact FAISS range search above a cosine threshold, connected components (the "union-find" grouping; recorded as a face partition with `method = 'union_find'`, `target = 'face'`, no project), O(n^2 d), about 4 min at 200k x 512 | `src/services/face_identity/person_grouping.py`, `person_partition.py` |
| Identity rules | Stable person ids across runs (`assign_person_ids`); a named or merged person's faces are locked (`is_person_locked`) and OCC-written | `src/domain/person_identity.py`, `person_management.py` |
| Routes (core, not `/curation`) | `POST /persons/cluster` (202 job), `GET /persons/cluster`, `GET /persons`, `GET /persons/{person_id}`, `PUT /persons/{person_id}/name`, `POST /persons/merge`, `DELETE /persons/{person_id}`; `POST /faces/detect`, `/faces/recognize`, `/faces/verify`, `/faces/search`, `/faces/identify` | `src/routers/persons.py`, `src/routers/faces.py` |
| Privacy | `OP_FACES_ENABLED` (fail-closed `400 faces_disabled`), counts-only purge audit (`privacy_purges`), purge of all face data | `src/config/faces.py`, `src/services/face_privacy.py`, `src/db/models/ops.py` |
| Curation | `curation_ingest` allows only `detect`, `embed_frame`, `embed_crops`, `region`; asking for `faces` is a 422 | `src/domain/processing_plan.py:64-81` |
| Not present | No split, no "not this person" (cannot-link), no hide-person, no merge suggestions, no person cover photo, no birthday; singleton handling issues are tracked (#354); Cropwright has no face or person screen (`rg persons frontend/src` finds nothing) | the designed improvements are in `docs/design/cluster_search_makemodel_plan.md` section 3 (prototypes, review band, cannot-links, split, quality gate, owner option O1: core-only vs a project `__faces` role) |

The `person` class from the COCO detector is a body box, not an identity: "People" in the library
sense needs faces.

### 1.5 Near-duplicates

| Level | Today | Files |
|---|---|---|
| Exact | `image_contents` keys each image on sampled `imohash`, SHA-256 only to confirm; a second copy of the same bytes is reported `duplicate` and stores nothing | `src/services/curation/ingest_dedup.py`, `src/db/models/near_dup.py` |
| Near | Runs (`full`/`incremental`, level `image` or `item`) find k-NN edges above a cosine cut, connected components with human decisions applied first, oversize groups split by complete linkage, the most central member is the representative, group identity carried from the parent run | `src/services/near_duplicates/`, `src/domain/near_dup.py` |
| Rows | `near_dup_runs`, `near_dup_groups`, `near_dup_members`, `near_dup_edges`, `near_dup_decisions` (`same`/`different`, survive re-runs), `near_dup_pending` (drained every 30 s), `holdout_near_dup_moves` | `src/db/models/near_dup.py` |
| Projection | `near_dup_group_id`, `near_dup_leak_group_id`, `near_dup_is_representative`, `near_dup_group_size`, `near_dup_run_id` on images and items | `src/clients/curation_opensearch/near_dup_fields.py` |
| Routes | `{P}/near_duplicates/status`, `/runs`, `/runs/{run_id}`, `/groups`, `/groups/{group_id}`, `/decisions` (GET, POST, DELETE); `{P}/near_dup_profiles*` (thresholds: dedup 0.98, leak 0.98, k 64 by default) | `src/routers/curation/near_duplicates.py`, `near_dup_gate.py` |
| UI | Gallery "Duplicates" control (`near_dup=collapsed|member|representative|none`), card badge "In a group of N", details panel scope to the group, group decision modal | `frontend/src/lib/components/workspace/DuplicatesControl.svelte`, `frontend/src/lib/nearDup/` |
| Purpose today | Training leakage (holdout rule, `409 near_dup_stale`) and export collapse. Rotated re-saves score 0.93-0.95 against the 0.98 cut (re-walk) | |

### 1.6 Embeddings and semantic search

| Aspect | Today | Files |
|---|---|---|
| Vectors | PE-Core L14 (`pe_core_l14_336_v1`), 1024-d: per item (`pe_embedding`), per region box (`region_box_embeddings`), whole frame on images (`pe_embedding`); HNSW (Lucene for new projects); about 8.4 KB stored per vector | `docs/opensearch_schema_design.md` section 2 |
| Space registry | Fields named by `embedding_spaces` rows (targets `item`, `box`, `frame`, `item_text`), up to three live spaces per target, build/promote/retire, evaluations | `src/domain/embedding_spaces.py:32`, `src/services/embedding_spaces/`, `{P}/embedding/spaces*` |
| Search routes | `GET {P}/search/text` (PE text tower, RRF fusion over spaces), `GET {P}/search/similar` (stored vector of a crop), `POST {P}/search/image` (uploaded picture); target `item` or `box` only (`SearchTarget`); result is one card per item | `src/services/curation/semantic_search*.py`, `src/schemas/curation/search_target.py` |
| Text on items | `item_text` filter over OCR lines read on item crops when the region worker has OCR configured | `src/domain/item_text.py`, `src/services/curation/item_filter.py:379` |
| Not present | Whole-photo text search (`frame` target exists in the registry but no search serves it; `docs/design/cluster_search_makemodel_plan.md` 4.1 proposes `target=frame` and `group_by=image`), whole-image OCR in curation, combined person + place + date + text query, ingest policy `embedding.mode` can turn crop vectors off | |

### 1.7 Thumbnails and image serving

| Route | Behaviour | Files |
|---|---|---|
| `GET {P}/images/{image_id}/file` | Clean source render, EXIF transpose, optional `max_dim` 128-8192; `Cache-Control: public, max-age=3600` | `src/routers/curation/images.py:103` |
| `GET {P}/images/{image_id}/thumbnail` | Whole frame shrunk to `size` 32-512 (default 256), computed from the original on first request, **per-process** LRU (`THUMBNAIL_CACHE_MAX_ENTRIES = 2000`, about 50 MB) | `images.py:134`, `src/services/curation/image_serving.py:278` |
| `GET {P}/crops/{crop_id}/thumbnail`, `/preview`, `/region_thumbnail` | Crop thumbnails; tmpfs crop cache written at ingest | `src/services/curation/crop_preview.py`, `source_image_cache.py` |
| Missing source | Typed `404 source_missing` (`alias`, `root_mounted`), shown on the tile | `docs/CURATION.md` "When the source images are not reachable" |
| Not present | No persistent thumbnail or preview tier (every API worker and replica re-decodes the original for its own cache; a 20 MP JPEG per tile on a cold timeline scroll), no sized variants at ingest, no blurhash/placeholder, no HEIC (bulk default extensions `.jpg .jpeg .png`, `src/services/curation/bulk_ingest/limits.py:22`), no RAW, no video | |

### 1.8 Originals in place

| Aspect | Today | Files |
|---|---|---|
| Roots | Deployment-wide `OP_SOURCE_ROOT` (container `/data/source`, mounted read-only into `api` and three workers) plus named `OP_SOURCE_PATH_ALIASES`; uploads land under `OP_STATE_DIR/projects/<slug>/` (copies) | `src/config/curation.py:383`, `docs/CURATION.md` "Mounting your image source" |
| Reference | Items and images store the **container path**; moving the host directory under the same container path needs no re-ingest; a different layout under the root does | same |
| Never moved | Curation never deletes in normal operation; the one delete path is import undo and the `detect` reprocess scope, both OCC-guarded (`item_delete.py`); a `drop` project's export strips GPS from the exported copies, never from the originals | `src/services/curation/item_delete.py`, `export_images.py` |
| Watching | `ingest_sources`: folder or S3 prefix with watch / schedule / webhook triggers, durable per-file queue (`ingest_source_files`), retry-dead | `src/routers/curation/ingest_sources.py`, `src/services/curation/bulk_ingest/` |
| Not present | Moved or renamed file detection (a re-ingest of moved bytes is a `duplicate` and the stored `image_path` stays stale), deleted-file reconciliation, sidecar (XMP) read or write | |

---

## 2. UI today (Cropwright Workspace)

One Workspace page (`frontend/src/routes/p/[project]/workspace/+page.svelte`) hosts seven views
from `frontend/src/lib/workspace/viewRegistry.ts`. The URL grammar (`workspaceUrl.ts`) holds
`view`, `queue`, `scope` (`cluster|near_dup|image`), `focus` (`item|image|cluster|cell`), per-view
`sort_*`, `rows`, `columns`, `thumb` (`s|m|l`), `q`, `similar_to`, `near_dup`, `map`. One shared
item filter (`frontend/src/lib/itemFilter/itemFilterState.svelte.ts`) feeds every view.

| View | Lists | Paging | Selection kind | Known performance |
|---|---|---|---|---|
| Clusters (`ClustersView.svelte`, `ClusterListView.svelte`, `ClusterScope.svelte`) | Cluster cards with representatives, facets, open cluster | `GET {P}/clusters` + representatives, 24 per page (`clusterListController.svelte.ts:46`) | `clusters` (filter + `cluster_uids`) | Counts honour queue and filter (re-walk check 16). Mixed-cluster warning missing (re-walk P2-B) |
| Gallery (`GalleryView.svelte`, `components/images/ImageGrid.svelte`, `ImageCard.svelte`) | Photos | `GET {P}/images`, cursor (`search_after`), 60 default, 200 max; accumulating pager | `images` (filter + `image_ids`) | An item predicate goes through the 10k image join (`join_truncated` banner); counts trail by the drain interval |
| Detections (`DetectionsView.svelte`, `CropResultGrid`) | Items as crops | `GET {P}/crops`, 200 per page (`MATCHING_PAGE_SIZE`), accumulating pager | `items` (`crop_ids`) | Re-walk P2-10: 18,675-item project, 2,200 rows at 2 s, 5,000 at 37 s, 10,000 at 100 s; about 6 s per 1,000 rows at the top, 14 s per 1,000 at 7,500-10,000. Cause class: no virtualisation, every card stays in the DOM (12f section 4.4 counts 1,970 controls at 477 cards) |
| Review (`ReviewView.svelte`, `lib/review/`) | One item at a time with filmstrip and source pane | `GET {P}/review/{tab}`, 30 per page, `/locate` deep links | single focus | Fine for labelling; empty block under the source image (re-walk P3-E) |
| Table (`TableView.svelte`, `lib/table/tableController.svelte.ts`) | Detection rows or image rows, served columns and sorts | `GET {P}/table`, 50 default, 200 max, cursor | `items` or `images` | Same accumulating pager as Detections |
| Map (`MapView.svelte`, `lib/map/`) | MapLibre GL 6.17 basemap, server cells, cell panel (12 cards), area tools (Shift+drag, lasso) | `GET {P}/map/cells` per viewport | `images` | Server-side aggregation; panel and "Show in Gallery" equal the API (re-walk check 15). Item predicates through the 10k join |
| Embedding (`EmbeddingView.svelte`) | UMAP projection, `viz_box` filter | `GET {P}/viz/projection` | items | Advanced; available when `/methods` serves it |

| Piece | Today |
|---|---|
| Selection model | `WorkspaceSelection` = `none | items | images | clusters | matching`; a bulk write sends an `ItemSelection` (filter alone, filter + ids, or crop ids), never an exclusion list (`frontend/src/lib/workspace/selection.ts`). Selection survives a view switch. Bulk writes cap at `OP_BATCH_SYNC_MAX` synchronously, then a job; label ops cap 20,000 items |
| Filters and chips | `ItemFilterBar.svelte` (class names include/exclude, confidence, area, rank, detector class, origin, embedding state, review status, import, audit outcome, capture dates, geo box/polygon, has location, near-dup group, cluster uids, viz box, image ids); `FilterBar.svelte` adds queue and scope chips; `HiddenItemsNote.svelte` explains hidden holdout and excluded counts; image-only: `source`, `in_holdout`, `near_dup`, `label_state`, `sort`, `include_test` |
| Queues | `WorkspaceRail.svelte`: served review tabs with counts (`GET {P}/review/tabs`), then classes (`RailClasses.svelte`) |
| Gallery card | 4:3 thumbnail with server-served boxes drawn client-side, chips "N detections, N labelled, N unlabelled", near-dup badge, file name, Details button, lightbox hover button (`ImageCard.svelte`); details panel shows file name, capture time, near-dup group (`workspace/details/ImageDetails.svelte`). No capture date on the card, no date grouping or headers |
| Lightbox | "View full size" dialog with previous/next, zoom; title and `focus=` do not follow a hop (re-walk P3-B) |
| Design direction | `docs/design/v070_rebuild/12f_ux_refinement_plan.md`: thin frontend, served copy and counts, lanes A-D; nothing in it targets a library |

---

## 3. The gap, feature by feature

Legend: **E** exists, **P** partial, **M** missing. "Needed" lists backend routes (proposed), storage
and UI pieces; every new route regenerates `contracts/` (`make contracts`).

| Feature | E / P / M | What exists | What is missing | Needed (routes, storage, UI) |
|---|---|---|---|---|
| Timeline by date with a scrubber | P | `taken_at:desc` sort on `GET {P}/images`; `taken_from/taken_to`; floating local time; cursor paging | Day/month/year buckets with counts, a jump-to-date cursor, date section headers, undated bucket, a virtualised grid that holds 100k+ photos without growing the DOM, persistent thumbnails | proposed `{P}/timeline/buckets` (`date_histogram` on `capture.taken_local`, `calendar_interval` day/month/year, filter-aware, `missing` bucket) and a locate-by-date cursor (`{P}/images/locate` already locates by id; add by date); storage: none new; UI: a virtualised windowed grid (evict off-screen pages; the pager today only appends), scrubber driven by bucket counts |
| Albums (manual) | M | `label_operations` show the undoable bulk-op pattern; selections by filter or ids | Album records, membership, order, cover, share/export | Postgres `albums` (+`album_members` image ids, position, added_by, added_at), RLS like every project table; proposed `{P}/albums` CRUD and `{P}/albums/{album_id}/members` with `expected_revision`; UI: album list, album view = Gallery scoped by album (new `scope` kind) |
| Smart albums (rules over time, place, person, object, text) | P | The shared `ItemFilter` is already a serialisable rule (classes, dates, geo box/polygon, cluster uids, near-dup); per-view counts | Saved rules, person and text predicates, image-level semantics (all items vs any item), place names, live counts | Postgres `smart_albums` holding a versioned filter document (or reuse the config store `config_documents` with a new kind); proposed `{P}/smart_albums` CRUD + evaluate; filter grammar extended with person id, text query, `taken_month_day` (for "on this day"); the 10k join cap must go (see 4) |
| People: named identities | P (core only) | Face detection, ArcFace, grouping, stable ids, rename, merge, delete, purge, lock on named/merged (core `/persons`, `visual_search_faces`) | Faces in a curation/library scope, Cropwright People screen, cover face, face thumbnails served as crops instead of `thumbnail_b64` | Decide the face scope (owner option O1 in `cluster_search_makemodel_plan.md` 3.3, and section 4 here); a faces store with `image_id` and the box; the `faces` branch allowed in the library ingest (`processing_plan.py:80`); person records in Postgres (today a person is only a `person_id` value on face docs) |
| People: merge / split | P | Merge exists (core); split does not | Split by selection, split proposals, merge suggestions with evidence | Designed in `cluster_search_makemodel_plan.md` 3.2 (prototypes, cannot-links, review band); proposed `persons/{person_id}/split`, `persons/suggestions` in the chosen scope |
| People: "not this person" | M | Lock rule protects named faces from machine moves | A per-face cannot-link that every later run respects | Postgres `face_person_constraints` (must-link, cannot-link) read by grouping before writing; route proposed `persons/{person_id}/faces/{face_id}/reject` |
| Hide a person everywhere | M | Item-level `class_excluded` hides items from browse lists (`visibility_clauses`, `item_filter.py:464`); `HiddenItemsNote` explains counts | Image-level hide keyed on a person, applied to every library view, search, map, memory | Person `hidden` flag in Postgres; projection onto images (e.g. `hidden_by` keyword list on the images doc, written by one projection writer) so every list adds one `must_not`; new photos inherit through face assignment (row "exclusions") |
| Places | P | GPS on images, map cells, geo box and polygon filters, location policy, area selection | Place names (city, country, POI), a Places grid, place facets and search | Offline reverse geocoder (a library, e.g. a GeoNames-based one; must stay local to honour the location policy and the "positions never leave the network" rule); storage `capture.place.*` keywords on images under the same policy (purged with `location_purge`); proposed `{P}/places` (terms aggregation) |
| Favourites | M | None | Per-photo flag, per user if multi-user | Images doc `favourite` (or Postgres `image_flags`); proposed `{P}/images/flags` bulk write with OCC; a `favourite` image filter; heart on `ImageCard` |
| Hidden | P | Item exclusion (`class_excluded`, reasons in `{P}/exclude_reasons`) | Photo-level hide with a "Hidden" view | Same flag mechanism as favourites; `visibility_clauses` gains an image-level clause |
| Trash | M | No delete of user photos by design (originals never moved) | A soft-delete view that hides photos from the library and can be emptied (removing index docs, never the original) | Images doc `trashed_at`; proposed `{P}/images/trash`, `/restore`, `/trash/empty` (job; reuses the OCC delete in `item_delete.py`); a guard that "empty trash" never touches the source file |
| Duplicates | E (training-oriented) | Exact and near duplicates, decisions, representatives, Gallery collapse | Library framing: "keep best" choice by quality, resolution and favourite; burst stacks; rotated copies are below the 0.98 cut | A representative rule that prefers favourites/sharpest; optional stack view; no new storage beyond a decision kind |
| Search by person + place + date + text | P | Text search over item crops; filters for date, geo, class, cluster; `item_text` | Whole-photo text search (`frame` target), person predicate, place-name predicate, one combined query ranked per photo | `GET {P}/search/text` with `target=frame` and `group_by=image` (already proposed in `cluster_search_makemodel_plan.md` 4.1) plus the shared filter carrying person, place and date; k-NN filter inside the `knn` clause (documented in `opensearch_schema_design.md`) |
| Memories | M | Near-dup representatives, quality fields on items (blur), capture time and place, clusters | Event segmentation (time and place gaps), photo quality/aesthetic score per photo, "best of" selection, generated story records | Celery job `memories_build` (beat), Postgres `memories` + `memory_members`; a photo-level quality score (new embedding-space or Triton model, measured first); proposed `{P}/memories` |
| Calendar import | M | None | iCalendar import (holidays, birthdays, trips), mapping events to time ranges | Postgres `calendar_events` per library; an ICS parser library (e.g. the `icalendar` package) rather than hand parsing; proposed `{P}/calendars` (upload, list, delete); smart-album rules gain `calendar_event` |
| Exclusions that new photos inherit | P | Ingest policy filters detections; incremental cluster assignment places new items into served clusters (`clustering.automatic`); `class_excluded` per item | Rules stored as data and applied at ingest and by a re-apply job; person-, class- and cluster-keyed hides that new photos pick up | Postgres `library_exclusion_rules` (kind: person, class name, cluster uid, smart-album filter), applied by the projection writer after face assignment and detection; a re-apply job when a rule changes; served "hidden by rule N" reason so the UI explains each hide |

Cross-cutting needs that several rows share:

| Need | Why | Today |
|---|---|---|
| Photo-level query path without the 10k join | Smart albums, person, hide and memories all ask "photos where some item/face matches" | `image_join.py:30` caps at 10,000; the images index already carries stored `counts.*` that avoid the join for label state |
| Photo-level projections | Hide, favourite, person ids, place names, trash must filter on the images index in one query | `count_fields.py` and `near_dup_fields.py` show the projection-writer pattern |
| Persistent thumbnails | Timeline scroll over 100k+ photos | On-demand per-process LRU only |
| Virtualised grids | 10k+ rows | Accumulating `createPager` (`frontend/src/lib/pager.svelte.ts`) |
| Users | Favourites, albums, hidden people are personal in Apple/Google/Immich | No authentication (`SECURITY.md`); `projects.created_by_user_id` exists, nothing else |

---

## 4. The key architecture question: library vs projects

The library spans the whole collection (500k to millions of photos, one root or a few). The system
is organised into isolated projects with two indexes each, one shard each. Three options (and a
hybrid) for the planner and the owner. No recommendation is made here.

### Option A: one library project

The library is an ordinary project (for example slug `library`) over the whole source root.
Training jobs either run in the same project or are spun out with `combine`.

### Option B: a library layer over many projects

Photos stay in projects (by folder, year or job). A new library layer reads across them: a
deployment-level `libraries` record lists member projects, and library routes fan in.

### Option C: a new library index (deployment-scoped role)

A third kind of storage beside projects: `op_lib_{library}__images`, `__faces` (and maybe
`__objects`), owned by a `libraries` row with its own guard binding. Projects for labelling
reference library photos by the content `image_id` (ingest from the library instead of from disk,
or a "send to project" that creates items there).

### Option D (hybrid): library as the source, projects as derived workspaces

C's library index is the only place photos, faces, places and time live; a training project is
created from a library selection (smart album or filter) and keeps today's isolation for labels,
holdout and models. Listed separately because it changes how ingest and combine relate.

| Criterion | A: one library project | B: layer over many projects | C: library index | D: hybrid |
|---|---|---|---|---|
| Isolation (fail-closed guard) | Unchanged: one bound project; no guard change | The guard refuses any multi-index request by design; a fan-in must bind each project in turn and merge in the API (no cross-index search, no global sort or `search_after` across projects), or the guard gains a "library binding" that owns several projects' indexes, a real change to the isolation model needing adversarial review | New binding kind with its own owned names; projects keep their guard; the boundary "library -> project" must be explicit (copy or reference by `image_id`), no index shared | As C; labelling isolation stays exactly as today |
| Scale 500k-millions | One `__items` shard holds every object: at about 13 items per photo, 1M photos is about 13M item docs and about 109 GB of 1024-d item vectors (8.4 KB each) plus 8.4 GB of frame vectors, in one shard (`base.py:72`); needs multi-shard indexes, ingest policy `embedding.mode=selected/lazy`, or both | Natural sharding by project, but every library query is N queries plus a merge; global timeline order and cursors across N projects are hard; the project capacity check (shards and heap) grows with N | Can be designed for size from the start: shard count, photo-first documents, faces as their own index, item vectors optional | As C; derived projects are small |
| Reuse of the curation pipeline | Full: ingest, capture, map, near-dup, clusters, search, label ops, review all work as is | Full inside each project; library features are new code over the fan-in | Ingest, capture, near-dup, embedding-space and map code reused only if parameterised by index role (they read names from the bound project today); risk of two parallel pipelines (forbidden by "no duplicate logic") | Same reuse question as C, plus a new library-to-project path that must map classes by name |
| Photos with no detections | Listed in Gallery/Map, invisible to item views | Same | Photo-first model can index them fully | Same as C |
| Faces | Needs the project `__faces` role (O1 option b) | `__faces` per project; one person split across projects needs a cross-project identity table | Faces live once, in the library | Same as C |
| Class scheme conflicts | Library labels and training labels share one registry and one holdout; a training job's class scheme pollutes library views | Per project, as today | Library holds machine labels and person names; training class schemes live in projects | Same as C |
| Lock rule and label ops | Unchanged | Unchanged per project | Library writes (names, hides, favourites) need their own human-lock rule; projects unchanged | Same as C |
| Migration of existing projects | None | Existing projects join a library | Existing projects keep working; a library is ingested fresh (content ids line up, so existing projects can be linked by `image_id`) | Same as C |
| Main risk | One giant shard, mixed concerns (library + training in one registry) | Weakens or complicates the isolation guarantee; slow fan-in | A second storage family with its own guard, migrations (`scripts/opensearch/migrate.py` schema stamp) and capacity accounting | Largest up-front design; ingest path split |

Facts that bear on every option: `image_id` and crop ids are content-derived (cross-scope joins by
id are possible without a mapping table); `image_contents` is per project (an exact-duplicate check
across scopes needs a library-level content table); OpenSearch has no unmap, so a new field family
is additive and a role change needs a rebuild; Postgres RLS is keyed on `project_id`, so a library
scope needs its own RLS key or a `library_id` column on the new tables.

---

## 5. Constraints any design must respect

| Constraint | Source | What it means for the library |
|---|---|---|
| Thin frontend: the backend serves data and decisions | `docs/design/v070_rebuild/12f_ux_refinement_plan.md` section 7, owner rule "no workarounds: add the route" | Timeline buckets, smart-album counts, memory selection, "hidden because" reasons, sort options and empty-state copy are served; Cropwright never filters, counts or ranks |
| Wire contract | `src/services/curation/wire.py` (one serializer), `make contracts`, `contracts/` committed in the same commit; pre-commit rejects stale contracts | Every new route and field ships its contract; wire names fixed independent of storage |
| Class identity by name | `docs/VISION_AND_GOALS.md`, engineering standards principle 6 | Library-to-project transfer, smart-album class rules and exclusion rules store class **names**, never `class_id` |
| Lock rule | `src/clients/occ_locks.py`, `docs/CURATION.md` "The lock rule" | Person names, merges, "not this person", favourites and hides are human decisions: machine regrouping, re-detection and reprocess never overwrite them; OCC read/merge/write helpers (`src/clients/occ.py`, `occ_bulk_rounds`) |
| No file locks; leases | `src/clients/job_lease.py`, `services/curation/lease_heartbeat.py`, `scripts/codegen/check_file_locks.py` ratchet | Singleton library jobs (face regroup, memories build, calendar refresh) claim an OpenSearch/Postgres job lease |
| Postgres for relational data, Celery for long jobs | Owner decision 2026-11-04 (all relational data in Postgres, all long jobs on Celery + Valkey, no job files) | Albums, smart albums, persons, constraints, calendar events, memories, exclusion rules are Postgres tables with RLS and Alembic revisions; grouping, memories, re-apply and purge are Celery tasks; handlers never fan out (engineering standards principle 9) |
| Stateless API, multi-user | `docs/ARCHITECTURE.md` "Per-process state", `scripts/codegen/check_process_state.py` | No in-process caches without revision checks; the thumbnail tier must be shared (disk or object store), not per-process |
| Engineering standards | `docs/design/engineering_standards.md` | Fail closed (an unbound library scope refuses; counts render unknown on failure); no shims or parallel routes; split by concern; evidence first (tests watched failing); baselines only shrink; review depth by risk (isolation, faces and purge get adversarial review) |
| Privacy | `src/services/face_privacy.py`, `privacy_purges`, location policy | Faces are biometric: per-scope switch, purge with dry-run count, counts-only audit; positions never copied to items, labels, logs or VLM requests; reverse geocoding must be offline; no face is sent to a remote model |
| Docs checked against code | `scripts/docs/check_docs_vs_code.py` | Plans may name proposed routes; user docs only routes that exist |
| Public data for tests | owner rule | COCO / Open Images / the Commons geotagged manifest (`scripts/datasets/manifests/commons_geotagged_60.json`) for tests and docs; faces tests on public identity sets only |

---

## 6. Questions for the planner and the owner

1. Which option in section 4 (A, B, C or D)? It decides where faces, person records and library
   flags live.
2. Are favourites, albums and hidden people per user? There is no authentication today.
3. Formats: HEIC and RAW in scope for the first library release, or JPEG/PNG only?
4. Reverse geocoding: which offline data set, and is a place name subject to the location policy
   (it is derived from the position)?
5. Faces default: on for a library scope, off for training projects (as the cluster plan proposes)?
6. "Hide a person everywhere": does it also remove the photos from training projects derived from
   the library, or only from library views?
7. Thumbnails: generate at ingest (disk cost per photo) or lazily into a shared disk cache?
