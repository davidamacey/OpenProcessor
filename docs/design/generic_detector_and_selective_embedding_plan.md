# Generic detector, store-everything ingest and selective embedding: design and implementation plan

Status: implemented (waves W0 to W9). Default embedding policy stays `all` (flip decision: keep). Deferred: arbitrary-filter clustering, a model-delete guard for a project's own detector, the optional `backfill_embedding_state.py`. Issues: #52 (default detector, store
every detection, embedding policy), #45 (public COCO baseline set, used for the
before/after numbers). Sibling plan: `docs/design/sam3_full_image_detection_plan.md`
(#30; its section 12 assumes this plan: a hit is always stored, whether it is
embedded follows the project embedding policy).

A fresh agent with no memory should be able to implement this from this file
alone. Paths are relative to the repo root. Line numbers are from main at
`13a38024` (source files unchanged since `837ca316`) and will drift; re-find by
symbol. Env var names written here all exist in code today; this plan adds no
env var (new tunables live in the per-project policy, section 5.2).

Notation for routes: existing routes are written `METHOD /path` and are checked
by `tests/test_docs_vs_code.py`. Proposed routes are written with a lowercase
method and a project-relative path, for example "put `/ingest/policy`
(proposed)", so the docs checker does not mistake them for shipped routes.
Project-relative means under `/curation/projects/{project}`.

## 1. Goal and decisions already made

OpenProcessor must be a generic inference and labeling engine, not a
vehicle-only tool. Owner direction (2026-10-03, recorded on #52):

D1. The default detector exposes the model's FULL class vocabulary (COCO 80 for
    the shipped `yolov11_small_trt_end2end`). The number of classes in the
    detector head adds essentially nothing to inference time.
D2. STORE every detection as an item row (metadata only, about 1.8 KB, no
    vector). Filtering is a later user choice, never forced.
D3. EMBEDDING is a separate, selectable step with a per-project policy:
    `all` | `selected` (classes by NAME, min size, min confidence) | `lazy`
    (embed on demand when the user filters, clusters or opens a class).
    Remaining detections can be embedded later with `POST
    /curation/projects/{project}/reprocess` scope `embed`. The lock rule is
    unchanged (embeddings are derived data, so locked items are included, see
    `src/services/curation/reprocess_embed.py` module docstring).
D4. Items without `pe_embedding` (and without region or backbone vectors) are a
    first-class state: clusters, search and exports ignore or flag them, and the
    state is visible on the wire and in the UI.
D5. An optional ingest-time class filter and per-image caps exist but are OFF by
    default.
D6. Class identity is by NAME everywhere in policy and seeding. Registry ids are
    per project and never assumed equal to a model's class ids
    (`src/utils/class_names.py` module docstring: "no fallback to another
    model's vocabulary").

Decisions this plan adds (owner can overturn; see section 12):

P1. Default embedding mode is `all` (section 5.5 for the justification and the
    rule that flips it).
P2. Items with a human or imported label are ALWAYS embedded regardless of mode
    (they are the training signal and cluster anchors).
P3. The set of embedded items is the project's "working set": the VLM stage,
    review queue, clustering and ordering views operate on it. Un-embedded
    detections are stored and browsable but wait for an embed step. This makes
    the embedding policy the single cost knob for embed AND VLM spend.
P4. GET requests never start GPU work. "Lazy" is triggered by an explicit embed
    step or by starting a pipeline run (section 5.7).
P5. Policy lives in the per-project settings document, not the versioned config
    store (section 5.2).

## 2. Current-state findings (evidence)

F1. Which detector ingest uses, and why it looked vehicle-only. Ingest has NO
    code default for the detector: `DetectionProfile.detector_model` defaults to
    `''` (`src/config/detection_profile.py`), so ingest returns 503 until
    `OP_INGEST_PRIMARY_DETECTOR_MODEL` is set (`src/routers/curation/ingest.py`
    `_get_ingest_service`, ~L90-L110). The profile is read from env on every
    call (`src/config/ingest_profiles.py:34` `ingest_primary_profile`), i.e. one
    detector for the whole deployment, shared by all projects, changed only by
    editing `.env` and recreating the `yolo-api` container. The core pipeline
    list loads `yolov11_small_trt_end2end` into Triton
    (`docker-compose.yml:40`), 80 classes (`models/yolov11_small_trt_end2end/labels.txt`),
    end2end four-tensor response with max 300 detections and the engine's baked
    confidence threshold 0.25 (`export/export_models.py:159`).
    The vehicle-only behaviour on the live pass came from the docs, not the
    code: `README.md:255` and `docs-site/docs/getting-started/quick-start.mdx:143`
    tell the user to set `OP_INGEST_PRIMARY_CLASS_IDS=2,3,5,7`; the live stack
    container has exactly that (`docker exec openprocessor-api env`). The code
    default (`class_ids` empty = every class,
    `src/services/curation/ingest_detect.py:221-231`) already keeps all 80.
F2. Detections are unlabeled proposals by default. `assigns_class` defaults
    false (`detection_profile.py`), so each detection becomes an item with
    `class_source=item_proposal`, no `class_id`/`class_name`, and
    `proposal_name` = the model's own label (`ingest_detect.py:244-257`,
    `proposer_label` at `ingest_profiles.py:54`, falling back to the model dir's
    `labels.txt` via `get_class_names`). The class registry starts EMPTY on a new
    project (quick-start step 3 creates classes by hand), so on a non-vehicle
    dataset nothing maps `person` to a class and the VLM has no vocabulary.
F3. `OP_INGEST_PRIMARY_CONFIDENCE_FLOOR` (docs: 0.4) is ignored for proposers:
    the floor is only applied in the `assigns_class` branch
    (`ingest_detect.py:260` `confident = conf >= floor`); the proposer branch
    at :244 appends every detection the engine emitted (>= 0.25). Docs
    (`docs-site/docs/configuration/advanced.mdx:122`) imply otherwise.
F4. Every detection is embedded unconditionally: `index_items`
    (`src/services/curation/ingest_index.py:114-124`) crops every item, calls
    `pe_encoder.embed_crops` for all, and on ANY exception only logs
    `ingest_embed_crops_failed` and writes the items with no vector and no
    marker (a silent fail-open: a vectorless item is indistinguishable from a
    deliberate one). The whole-frame vector (`images.pe_embedding`) is
    computed separately for every new image (:128-138). Every item also gets a
    crop JPEG written to the crop cache (:182 `write_crop_cache`, pruned at
    `crop_cache_max_bytes`, default 3 GiB, `src/config/curation.py:158`).
F5. Vector-less items already exist in the codebase (embedding exception
    above, project import without vectors `project_source.py:88`, combine
    dropping a wrong-dimension vector `combine/copy_docs.py:118`), but nothing
    names the state. The mapping needs no change to allow them: absent
    `knn_vector` fields are simply not indexed (`curation_opensearch.py:554`
    `pe_embedding`, :557 backbone; derived source stays on, :146-158).
F6. `reprocess` scope `embed` exists but is IMAGE-unit and always re-embeds
    everything: `IMAGE_SCOPES = ('detect','embed')` (`reprocess_images.py:32`),
    `_embed_targets` (:35) loads ALL items of the selected images, and
    `reembed_items(..., parts=ALL_PARTS)` (:105) rewrites crop, frame and region
    vectors with no only-missing option for the crop part
    (`reprocess_embed.py:104`; `only_missing` exists only for region boxes). The
    filter model (`reprocess_models.py:27` `ReprocessFilter`) has no selector for
    embedding state, class name, size or confidence.
F7. Residual (not-yet-labeled) clustering, IVF placement and most orderings
    already require the vector via `exists` filters (section 4 table). The ones
    that do NOT: the auto-label VLM selection, stats, wire. The VLM worker DOES
    require it (`scripts/curation/vlm_worker.py:172`), so a vectorless item is
    never labeled and nothing says why.
F8. Wire: vectors are excluded from every item read
    (`src/services/curation/wire.py:83` `_EMBEDDING_SOURCE_EXCLUDES`), so a
    client cannot tell an embedded item from a vectorless one. `ItemDoc`
    (`src/routers/curation/_item_models.py:25`) and `serialize_item`
    (`wire.py:214`) carry no embedding field.
F9. Filtering by detector class name is not exposed: `GET /crops` has no
    `proposal_name` filter (`src/routers/curation/crops.py:102-147`), though the
    field is mapped keyword (`curation_opensearch.py:504`). Region scope already
    matches by name case-insensitively against `class_name` OR `proposal_name`
    (`src/services/curation/region_scope.py:25-62`); that predicate is the model
    for name matching here.
F10. Registry names must match `^[a-z0-9_]+$`
    (`src/routers/curation/_class_models.py:67`) but 15 COCO labels contain
    spaces (`traffic light`, `hot dog`, `dining table`, ...). Seeding must
    slugify (`traffic_light`) while `proposal_name` keeps the raw label, so name
    matching must normalize both sides.
F11. Per-project settings document exists (index role `settings`,
    `curation_opensearch.py:627` `_settings_body`, read by `get_curation_settings`
    :794 with a 5 s cache, cloned by the `settings_defaults` axis,
    `_project_models.py:42` `CLONEABLE_AXES`). Its `defaults` map is axis ->
    single id (`strategy_defaults.py`) and cannot hold a structured policy. The
    config store (prompt packs, region profiles, VLM activation:
    `src/services/config_store/`, axes `prompt_pack | detection_profile | vlm`,
    `index.py:35`) is versioned+activated and heavy; it carries per-process
    snapshots and hot reload that a few small ingest knobs do not need.
F12. Measured on the live stack (read-only, 2026-10-03, project
    `sample-coco-2k-v2`, 2,000 images, vehicle-only detector): items 2,740 docs,
    25.8 MB primary store (9.4 KB per item, all with `pe_embedding`; includes
    123 deleted-doc leftovers); images 16.6 MB (8.3 KB per image, the frame
    vector). So 1.37 items per image. Proposal names present: car 1,770,
    motorcycle 453, truck 314, bus 203.

## 3. Consumer table: how each consumer treats a vector-less item today

"Required" assumes the target design (section 5). Verified means I read the
code path; "not verified" means no vector reference was found by search and the
claim should be re-checked when the wave touches it.

| # | Consumer | Current behaviour (file:line) | Required change |
|---|---|---|---|
| 1 | Ingest write path | Embeds every item (`ingest_index.py:114-124`); on exception logs and writes vectorless silently; `build_item_doc` writes the vector only if set (`item_doc.py:205`); IVF residual placement only when the vector exists (`ingest_index.py:172,186`); a class-labeled item still gets `cluster_id = class_id` without a vector (:184) | Gate embedding by policy; always write `embedding_state`; failure becomes state `failed` plus `IngestResult` counters; factor the IVF placement block into a function reused by the embed step |
| 2 | OpenSearch mapping | `pe_embedding` and backbone knn fields, derived source on (`curation_opensearch.py:146-158,554-557`); absent vectors are legal | Add explicit keyword `embedding_state` to the items mapping (tests: `tests/curation/test_items_mapping_explicit.py`); no knn change, so no index recreation for this alone |
| 3 | Residual clustering (UMAP/AHC/IVF) | All fetchers filter `exists RESIDUAL_EMBEDDING_FIELD` (`embedding_reduce.py:164,295`; `orchestrator.py:610,754,1139`; skip-None in scroll loops :278, :1169); `residual_gate_coverage` counts only embedded (:763) | Behaviour already correct (vectorless ignored). Needed: surface `n_unembedded` in run summaries; note `OP_RESIDUAL_EMBEDDING_FIELD` can name another field, so the shared predicate takes the field as a parameter |
| 4 | Per-cluster refine / centroid geometry / outliers | `_members_query` and `_cluster_ids` require the vector (`cluster_geometry.py:85-110`); `fetch_cluster_members` skips None (`orchestrator.py:278`) | Cluster card `count` (all members with that `cluster_id`) can exceed the geometry `n` for class clusters holding vectorless class-labeled items; report both (`n_embedded`) |
| 5 | Region clustering | Box vectors only: `has_vector_clause` (`region_box_rows.py:92`), `regions_fp.py:170` | None. Independent of the item vector; region stage runs on vectorless items (section 3, row 17) |
| 6 | kNN semantic search | `knn` query on `pe_embedding` (`semantic_search.py:35,144`); kNN returns only docs with the field, so vectorless items are silently absent; envelope is `{items,total,page,page_size}` | Add `unembedded_in_scope` (count of scope items without a vector) so "no results" is not mistaken for "no matches" |
| 7 | "Similar"/diverse selection | `select.py:256` filters `exists`; `crop_browse.py:127-150` `with_exists_filter`/`embedding_pool_query_and_count` narrow the pool; `crop_orders.py:50-110` returns `n_pool` | Add `n_unembedded` next to `n_pool` in the `crops_page` envelope |
| 8 | Label propagation | No kNN label spreading exists. The propagation-like path is `POST /curation/projects/{project}/clusters/auto_promote` (`routers/curation/clusters.py:495`, `clustering/auto_promote.py:170`), which works on `cluster_id` bucket aggregations and `class_source`, not vectors | None; add a test pinning that vectorless class-labeled members count toward purity as today |
| 9 | FP matcher | Region-box FP only (`regions_fp.py`, `region_box_clustering.py`), box vectors; no item-level FP matcher in this repo | None |
| 10 | Export | No item vectors read. Frame dedup uses the images index vector (`frame_dedup.py:169`, `export.py:337`), which stays always-on. BUT `unlabeled_items_on_exported_images` / `require_fully_labeled_images` (`export_images.py:20,174-187`) count every non-excluded unlabeled item, so full-vocabulary detections of classes the user does not want inflate it | Document; provide bulk exclude by filter (wave W6) so out-of-scope detections become `class_excluded` and stop counting |
| 11 | auto_promote | Bucket aggregation, `class_source` terms (`auto_promote.py:108,264`); vector-agnostic | None (test only) |
| 12 | Review queues | `all` tab's "no class" branch requires the vector (`review_queries.py:392`); empty-state reasons are computed from live state (`review_empty_reason.py:47`) | Keep the requirement through the shared predicate; add review/empty reason "N detections are not embedded (policy: ...)" and a way to list them (filter `embedding_state`) |
| 13 | Dashboards / stats | `GET .../stats` and `class_sources` aggs count all items (`routers/curation/stats.py:340-390`), project counts in `services/projects/stats.py:36`; no embedding breakdown. `unlabeled` currently means "no class", not "clusterable" | Add `embedding_states` terms agg (with `missing` bucket) to curation stats and `items_embedded` to project counts |
| 14 | Undo | Label/edit undo restore class/box fields from `class_id_history`/edit history without touching vectors (`edit_history.py:67`; `label_undo.py`); item delete has no vector logic (`item_delete.py`); dataset-import undo (`dataset_import/undo.py`) deletes imported items | None; add a test that exclude -> undo of a vectorless item leaves `embedding_state` and cluster fields coherent (exclusion stores prior cluster id, `exclusion.py:32-60`) |
| 15 | Dataset import | Import builds `DetectedItem`s and calls `index_items` (`dataset_import/chunk.py:330`), so imported items embed through the same code | Imported/labeled items are always embedded (P2); propose-mode machine items follow the policy; project-to-project import excludes vectors by default (`project_source.py:88`) and re-embeds via the same path |
| 16 | Combine projects | `copy_docs.py:100-135` deletes a vector the target cannot use (dimension) and keeps vectorless docs as is | When a vector is dropped set `embedding_state=deferred`; target project policy is not merged |
| 17 | Detection (region) worker | Selects by region status and `parent_classes` only (`scripts/curation/worker/cascade.py:55-100`); its box vectors are independent (`region_embed_stage.py`) | None for vectors. With full vocabulary, a region profile with EMPTY `parent_classes` now applies to every class: shipped profiles must keep `parent_classes` set (docs note) |
| 18 | VLM worker | Pending query requires the vector (`vlm_worker.py:172`, `_build_pending_query` :105); the auto-label VLM stage selection does NOT (`autolabel/selection.py:30`) | One shared predicate for both (P3); expose a "waiting for embedding" count |
| 19 | Auto-label worker | Stages `cluster_id_normalize, cluster_residuals, auto_promote, vlm, finalize` (`autolabel/job.py:70`); no embed stage | Add optional `embed_missing` stage for lazy mode (W7); verify the worker has Triton access first |
| 20 | Wire models / contracts | `_EMBEDDING_SOURCE_EXCLUDES` (`wire.py:83`); `ItemDoc` (`_item_models.py:25`); generated `contracts/ts/itemWire.ts`, `contracts/json/item_wire.json`, `contracts/openapi/curation.json` | Add `embedding_state` to `ItemDoc` and `serialize_item`; regenerate with `make contracts` (hooks `api-contracts-drift`) |
| 21 | Item scores / probe / viz | Scorers fetch embeddings with exists filters (`item_scores/base.py:96-112`); embedding viz uses exists (`embedding_viz.py:346,620`); probe scores crops through a model (no stored-vector reference found in `probe_*.py`, not verified) | None; report unembedded counts in viz; confirm probe by test |
| 22 | Reprocess `embed` | See F6 | Make it item-unit with selectors and only-missing (W5) |
| 23 | Reprocess `detect` | `redetect_image` merges with `remove_stale`, calls `index_items` (`reprocess_detect.py:132`) | Honors the same policy; a tightened detect filter removes UNLOCKED machine items outside it (document; dry run shows `removed`) |
| 24 | Re-ingest of an existing item | `occ_upsert_bulk` updates the stored doc (`clients/occ.py:392`); vectors excluded from the pre-fetch (:64-70) | Test: a re-index of an already embedded item under mode `selected` must not drop its stored vector (partial update); if it does, the update must omit `embedding_state` |

## 4. Ingest-time cost model

Per detection (one stored item):

| Part | Metadata-only item | Item with vector |
|---|---:|---:|
| Item document (labels, bbox, scores, provenance) | about 1.8 KB (owner figure; measured live store gives about 1 KB of non-vector overhead per item, F12) | same |
| `pe_embedding` (1024-d, HNSW + flat copy, derived source) | 0 | 8.4 KB (`docs/opensearch_schema_design.md` L202-204) |
| Crop JPEG in crop cache | written for every stored item (F4), bounded by the 3 GiB prune cap | same |
| GPU | detector only | + one `pe_image_encoder` crop forward (max batch 32, `models/pe_image_encoder/config.pbtxt`) |
| VLM | none (outside the working set, P3) | + one VLM call per working-set item that reaches the VLM stage (unlabeled, below classifier skip confidence) |
| Residual clustering | not in pool | UMAP and AHC input; AHC refine is capped at 8,000 members per cluster (`MAX_REFINE_MEMBERS`) so a 5x pool costs more than 5x time |

Per image: frame vector 8.4 KB and one frame forward, always (needed by frame
dedup and export dedup); detector forward once per image regardless of class
count. Per-image caps (section 5.2) bound the item count and so every row above.

Per 1,000 images (storage in MB; items use 1.8 KB metadata + 8.4 KB vector;
detections per image are the owner's COCO figure, to be re-measured in W0):

| Scenario | Items | Item store | Frame vectors | Total | vs vehicles | Crop forwards | VLM calls (upper bound) |
|---|---:|---:|---:|---:|---:|---:|---:|
| A. Vehicle-only detector, embed all (today) | 1,370 | 14.0 | 8.4 | 22.4 | 1.0x | 1,370 | 1,370 |
| B. Full COCO, embed all (7/img) | 7,000 | 71.4 | 8.4 | 79.8 | 3.6x | 7,000 | 7,000 |
| C. Full COCO, store all, embed none (lazy before any trigger) | 7,000 | 12.6 | 8.4 | 21.0 | 0.9x | 0 (+1,000 frame) | 0 |
| D. Full COCO, embed 4 vehicle classes only (`selected`) | 7,000 (1,370 embedded) | 24.1 | 8.4 | 32.5 | 1.45x | 1,370 | 1,370 |

Reading: keeping everything with a vector costs 3.6x storage and 5.1x crop
forwards and VLM calls per image; keeping everything WITHOUT a vector costs
about the same storage as today. Storage is the small number; GPU and VLM time
is the real cost (VLM throughput on this stack must be measured in W0, it is not
in the public docs). Throughput formulas for the PERFORMANCE.md table:
`images/s = 1 / (t_detect + t_frame + n_embedded_per_image * t_crop_embed + t_write)`,
`vlm_hours = n_vlm_items / vlm_items_per_s / 3600`.

## 5. Target design

### 5.1 Data model

Items index gets one explicit field (mapping at
`src/clients/curation_opensearch.py` next to `proposal_name`, ~L504):

`embedding_state` (keyword), values:
- `embedded`: `pe_embedding` is present. Written by ingest and by the embed step.
- `not_selected`: mode `selected` and the item did not match.
- `deferred`: mode `lazy`, or a vector dropped because the target project could
  not use it (combine).
- `failed`: the encoder raised for this item (today a silent case, F4).
- absent: a document written before this change. Treat as unknown: the
  authoritative test for "has a vector" is always the `exists` query; the state
  field only supplies the reason. Optional one-off script
  `scripts/curation/backfill_embedding_state.py` (two `update_by_query` calls
  with a constant `params` value, never reading vectors in painless: documents
  that match `exists pe_embedding` get `embedded`; documents without it and
  without a state get `failed`). Stacks are normally recreated, so this script
  is optional.

New module `src/services/curation/embedding_state.py` (pure, no OpenSearch
client): constants for the four values; `embedded_clause(field=ITEM_EMBEDDING_FIELD)`
returning `{'exists': {'field': field}}`; `not_embedded_clause(field)`;
`select_for_embedding(items, policy, image_w, image_h) -> list[EmbedDecision]`
(decision = `embed` or a state value); `normalize_class_name(text) -> str`
(strip, lowercase, spaces and hyphens to `_`, collapse repeats) and
`name_matches(names, *, class_name, proposal_name)`, modeled on
`region_scope.py` but normalizing to the registry slug form. Every consumer in
section 3 that needs "embedded" or "not embedded" imports from here; no
consumer writes its own exists clause after W3 (a repo test enforces this with a
grep, see W3).

`DetectedItem` (`item_doc.py:51`) gains `embedding_state: str | None`.
`build_item_doc` (`item_doc.py:151`) always writes it.

Wire: `ItemDoc.embedding_state: Literal['embedded','not_selected','deferred','failed'] | None`
(null = legacy), emitted by `serialize_item`.

### 5.2 Policy: where it lives, shape, validation

Stored in the per-project settings document (F11) under a new top-level field
`ingest_policy` (object, `enabled: false` in `_settings_body`, so it is stored
and never indexed; read by a new `get_ingest_policy(client)` that reuses the
settings document GET and its 5 s cache with immediate invalidation on write,
as `update_curation_settings` does). Reasons: it is per project, it needs no
versions or activation (a policy change affects only future ingests and
explicit embed runs, never past data), it is cloned by project clone, and it is
read by exactly one process family (the API; ingest runs through HTTP:
`scripts/curation/ingest_walker.py` posts to `/ingest/batch`). No env var is
added. The config store was rejected: it exists for artifacts that workers hot
reload and that carry revisions pinned onto results (prompt packs, region
profiles, VLM endpoints).

Typed model (pydantic, `extra='forbid'`, in `src/routers/curation/_ingest_policy_models.py`):

```
IngestPolicy:
  revision: int                       # server-managed, OCC token
  detect:   DetectFilter              # OFF by default: every field null/empty
  embedding: EmbeddingPolicy
DetectFilter:
  classes: list[str] | None           # allow-list by NAME (null = every class)
  exclude_classes: list[str]          # deny-list by NAME
  min_confidence: float | None        # 0..1, applies on top of the engine's 0.25
  min_box_area_frac: float | None     # box area / image area, 0..1
  max_per_image: int | None           # >= 1; keep highest score, ties by area
EmbeddingPolicy:
  mode: 'all' | 'selected' | 'lazy'   # default 'all'
  classes: list[str]                  # by NAME (mode selected)
  min_confidence: float | None
  min_box_area_frac: float | None
  max_per_image: int | None           # embed at most N per image, best score first
```
Rules: `selected` requires at least one criterion (else 422, an empty selection
embeds nothing); criteria combine with AND, class names OR within the list;
names are normalized with `normalize_class_name` and compared with the item's
`proposal_name` AND `class_name`; unknown names are accepted and returned as
`unknown_names` warnings (a model switch or a later class can make them valid);
items with a human or imported label (`item.label` set, or a locked
`class_source`) are always embedded (P2). No policy document = defaults =
exactly today's behaviour.

Where applied:
- Detect filter: inside `CurationIngestService.detect_items` (`ingest.py:410`),
  after the secondary pass, so the single-image path, the batch path
  (`prefilled_items`), dataset-import propose mode and reprocess `detect` all
  honor it. `detect_items` returns a small `DetectResult(items,
  secondary_detector_error, n_filtered)` instead of a tuple; its three callers
  (`ingest_one`, `dataset_import/proposals.py`, `reprocess_detect.py`) change
  with it. Filtered detections are not stored (that is the explicit user
  choice); the count is returned.
- Embedding gate: inside `index_items` (`ingest_index.py:114`), replacing the
  unconditional `embed_crops`. Only selected crops are sent to the encoder; all
  crops are still cut for the crop cache. A backbone vector from a secondary
  detector (`attach_backbone_embeddings`) follows the same decision (dropped
  for non-selected items so the storage promise holds).
- Frame vector: unchanged and always on.
- `OP_INGEST_PRIMARY_CLASS_IDS` stays a deployment-level hard drop by model id
  (it contradicts "store everything" if left in the quick start). W1 removes it
  from the quick start and README; W4 decides retirement (open question Q3).

Routes (proposed, project scoped, typed, fail closed through the existing
project guard like every scoped route):

| Method | Path | Body / response | Notes |
|---|---|---|---|
| get | `/ingest/policy` | `IngestPolicy` | defaults when never written |
| put | `/ingest/policy` | `IngestPolicyUpdate` (policy without server fields + `expected_revision`) -> `IngestPolicy` + `unknown_names` | 409 on revision mismatch; 422 on invalid |
| post | `/ingest/policy/preview` | candidate policy -> counts over STORED items: total, would embed, would not, by class name, estimated vector storage MB | read-only; drives the UI cost preview |
| post | `/classes/seed_from_detector` | `{names?: list[str], group?: str, dry_run: bool = true}` -> created/skipped/conflicts | append-only by name, idempotent |
| get | `/detections/summary` | counts by `proposal_name` x `embedding_state`, class name, plus `suggested_reprocess` (a ready-to-POST reprocess body) | W6 |

Existing `GET /curation/projects/{project}/ingest/config` (`ingest.py:313`,
`IngestConfigResponse` in `_common_models.py:382`) gains a read-only
`detector` block (model, version, input size, `assigns_class`, env-level
`class_ids` filter if set, label count, the label list with raw name and
registry slug, `confidence_floor_applies` = false for proposers per F3) and a
`policy` echo. Existing `POST /curation/projects/{project}/ingest/batch`
results gain `n_embedded`, `n_not_embedded`, `n_filtered` per image
(`IngestResult`, `ingest_models.py:44`) and the same sums in `IngestSummary`.
The ingest walker prints them.

### 5.3 Detector and registry

- Switching the detector (documented in W1): set
  `OP_INGEST_PRIMARY_DETECTOR_MODEL` (and `OP_INGEST_PRIMARY_LABELS_PATH` if the
  model dir has no `labels.txt`) in `.env`, recreate the `yolo-api` container;
  the model must serve the end2end four-tensor contract (`ingest_detect.py`
  docstring; a promoted YOLO26 fused output is NOT drop-in,
  `docs/CURATION.md` L826-L830). It is deployment-wide. Per-project detector
  selection is a later option (Q4), not needed for the minimum slice.
- Default stays `assigns_class=false`: detections are proposals carrying
  `proposal_name`. Registry ids are never trusted to equal model ids, which
  makes a detector switch safe (D6).
- `seed_from_detector`: reads the label list the same way ingest does
  (`proposer_label` / `get_class_names`), slugifies (F10), skips names already
  in the registry (by name, case-insensitive), appends the rest through
  `create_registry_class` (`routers/curation/classes.py:251` path), group
  defaults to `detector`, `dry_run` default true. It does not align ids, so it
  never conflicts with a hand-built registry. The CLI
  `scripts/curation/seed_class_registry.py` (id-aligned, for a model that
  assigns classes) is unchanged.
- Optional later (Q5/W8): `detect.class_resolution = by_name` makes ingest set
  `class_id`/`class_name` when a registry class equals the slugified proposal
  name, written with a classifier-style `class_source` so the VLM skips them.
  Not needed to remove the vehicle-only limitation and changes classification
  semantics, so it is deliberately outside the minimum slice.

### 5.4 Worker flow

Ingest (API process): decode, dedup -> detector (`run_primary_batch`, one
Triton call per `batch_limit` chunk) -> optional secondary -> detect filter ->
`select_for_embedding` -> `embed_crops` on selected only -> placement (class
cluster id, IVF residual placement for embedded unclassed items, parked
otherwise) -> `build_item_doc` with `embedding_state` -> bulk upsert.

Embed step (reprocess scope `embed`, W5): selector -> item-unit targets ->
`embed_crops` in batches of 32 -> bulk partial update of `pe_embedding` +
`embedding_state='embedded'` (+ `cluster_id`/`cluster_distance` through the
shared placement function when the item has no class and the IVF store
exists) -> job progress. Runs under the existing per-project singleton job
(`reprocess_job.create_job`), so it never races another reprocess.

VLM / review / clustering: operate on the working set via
`embedding_state.embedded_clause()` (P3).

### 5.5 Default embedding policy: `all`, justified

Recommend `all` as the stored default (absent policy = `all`).
1. It is today's behaviour: nothing changes for existing projects and every
   downstream feature (clusters, search, review ordering, VLM) works on first
   ingest.
2. Storage is the small cost (section 4: 8.4 KB per detection, 3.6x on full
   COCO); the real costs are GPU and VLM time, which the user controls with
   one call to put `/ingest/policy` BEFORE a big ingest or `selected`/`lazy`
   afterwards without losing any detection.
3. `lazy` as default would make a first-run user see an empty review queue,
   empty clusters and no search until they find an embed step; that is a worse
   first-run than a longer first ingest.
4. Flip rule (revisit after W0/W9): if the full-COCO baseline shows the crop
   embed plus VLM share of wall time above 40% and a typical first dataset
   above 10,000 images, make `selected` with an empty-class guard or `lazy` the
   default for NEW projects only (existing policies are never rewritten).
The ingest response always reports `n_embedded` / `n_not_embedded` so the
trade-off is visible, and `GET /ingest/config` echoes the policy.

### 5.6 Items without vectors: UX contract

- Every item read carries `embedding_state`.
- Lists: `GET /crops` gains `embedding_state` (repeatable), `proposal_name`
  (repeatable, normalized match) and `min_area` filters (W6).
- Ordered views (`order=outliers|diverse|core_first`) keep ranking only
  embedded items and return `n_unembedded` plus a `suggested_reprocess` body;
  they never embed on the fly (P4).
- Empty-state text: `compute_empty_reason` adds "N detections are not
  embedded (policy: selected)" when relevant.
- Stats: counts by `embedding_state`.

### 5.7 Lazy triggers

"Lazy" has three explicit triggers, no hidden GPU work in reads (P4):
1. The user asks: `POST /curation/projects/{project}/reprocess` with scope
   `embed` and a selector (class names, `embedding_state`, size, confidence,
   crop or image ids), usually from the `suggested_reprocess` the UI received.
2. Starting a pipeline run with an embed scope: new optional stage
   `embed_missing` (first in `STAGES`, `autolabel/job.py:70`), limited to the
   run's `class_id`/`cluster_id` scope when one is given.
3. Opening a class: the class page requests the summary; if unembedded items
   exist in that class name, the UI offers the embed action (frontend delta
   F7). No automatic embedding on view.

### 5.8 Embedding use cases (owner addition 2026-10-03)

Normal flow: embed everything at ingest (default `all`); filtering, selecting
and clustering are views over embedded items. Six ways a detection needs a
vector AFTER ingest, with the state of the code after W3 (the user-facing text
is in `docs/CURATION.md`, "Embedding use cases"):

| # | Case | Today (after W3) | Follow-up |
|---|---|---|---|
| 1 | New object or box (user, SAM 3, region profile) | Items are created only by ingest or dataset import, and both embed. The region worker embeds the region boxes it writes. A region box a person draws has no vector until an embed run. | W5/W7: embed a human-drawn box on write (embed in the box-edit route through the app encoder, or queue it for the embed-missing job). Decide with the owner; GPU work in a PUT is allowed by P4, a GET never. |
| 2 | Box moved or resized | An item's own box is never edited. A moved or deleted region box has its vector pruned at once (`prune_box_embeddings`), so it counts as missing. Nothing re-embeds it automatically. | Same follow-up as 1: re-embed the pruned box after the edit. |
| 3 | Embedding failed at ingest | W3: state `failed`, counted in `n_not_embedded`. Retry path: reprocess scope `embed` (now writes `embedding_state=embedded`). | W5: item-unit selector `embedding_state=failed` so a retry does not re-embed the whole image. |
| 4 | Skipped by an ingest policy | No policy yet (W4). The states `not_selected` and `deferred` exist. | W4 writes them; W5 embed-missing selects them. |
| 5 | Imported dataset | Import embeds through `index_items` (test: `test_import_run.py`), so items are `embedded` or `failed`. Project import without vectors and a combine that drops a vector leave items vectorless (`deferred` for the drop). | W5 embed-missing covers them; project import should set `deferred`. |
| 6 | Embedding model changed | Full re-embed with reprocess scope `embed` over every image (same dimension only: the mapping fixes it). Distinct from embed-missing. | W5: keep a full-rewrite option next to only-missing; document the dimension limit (done). |

## 6. Wave breakdown

Each wave is independently shippable and testable. Red-first: write the named
tests, watch them fail for the stated reason, then implement. "Mutation" lines
are deliberate breakages to prove the tests bite. Run the whole `tests/` plus
`tests/test_naming_leaks.py` and the docs checks per wave.

### W0. Baseline measurement (docs and script only) - v0.4.0
Files: `docs/PERFORMANCE.md` (new "Ingest cost per image" section), a small
script `scripts/bench/ingest_cost_probe.py` (reads item/image store sizes via
`_stats`, counts, wall time from the walker output). Dependencies: the public
COCO set (#45; `make sample-coco-readme` gives 200 images, the 4,000-image
manifest is the target).
Steps: in a throwaway project, ingest the 4,000 set with today's settings
(vehicle filter env set, then unset), record: images/s, detections/image,
item store bytes/item, frame bytes/image, crop forward rate, VLM items/s on a
200-image sample, crop cache growth. Fill section 4's table with measured
numbers.
Tests: the script has a unit test on its size/ratio arithmetic.
Acceptance: PERFORMANCE.md contains the before table for vehicle-only and
full-vocabulary, both with and without the vector (rows with vs without).

### W1. Generic default, docs and first run (no behaviour change) - v0.4.0
Files: `README.md` (~L240-L262), `docs-site/docs/getting-started/quick-start.mdx`
(L130-L180), `env.template` (L194-L197 quick block, L392-L408),
`docs/CURATION.md` (L866-L872, L980-L987), `docs-site/docs/configuration/basic.mdx:139`,
`docs-site/docs/configuration/advanced.mdx:122` (floor note, F3),
`docs-site/docs/guides/use-your-own-domain.mdx`, `CHANGELOG.md`.
Content: the default detector is `yolov11_small_trt_end2end` with all 80
classes; unset `OP_INGEST_PRIMARY_CLASS_IDS` and say it is an optional hard
drop (not recommended; the per-project filter replaces it in W4); explain why
the first-run COCO sample now yields person/animals/food items; how to switch
detectors (section 5.3); cost per image (link to W0 numbers); the
`confidence_floor` note; region profiles need `parent_classes` under a
full-vocabulary detector.
Tests (red-first): extend `tests/test_docs_vs_code.py` coverage by adding a
doc test `tests/test_generic_default_docs.py` asserting the quick start and
README no longer instruct `OP_INGEST_PRIMARY_CLASS_IDS=2,3,5,7` as a required
step and that `env.template` quick block leaves it unset (fails today).
Mutation: re-add the line to the quick start; the test must fail.
Acceptance: a fresh reader following the quick start on the COCO sample ends
with person, animal and food items (checked live in section 8).

### W2. Detector introspection and registry seeding by name - v0.4.0
Files: `src/routers/curation/_common_models.py` (`IngestConfigResponse` +
`IngestDetectorInfo`), `src/routers/curation/ingest.py:313`, new
`src/services/curation/detector_vocabulary.py` (labels, slug, seed plan),
`src/routers/curation/classes.py` (route `seed_from_detector`),
`src/routers/curation/_class_models.py` (request/response), contracts.
Tests (red-first): `tests/curation/test_detector_vocabulary.py` (slugify:
`traffic light` -> `traffic_light`, collisions, duplicates), 
`tests/curation/test_seed_from_detector.py` (dry-run default writes nothing;
idempotent second run creates 0; existing hand-made `car` is skipped; names
subset; 422 on unknown name; append-only ids from `max(id)+1`),
`tests/curation/test_ingest_config_detector.py`. Leak sweep: register the new
route in `tests/curation/test_cross_project_leak.py` (`route_params` L113,
`route_bodies` L135 with a valid body, `NO_WRITE` L317 only for the dry-run
form if the sweep body is dry-run) and confirm `tests/projects/test_route_scoping.py`
classifies it as scoped. Run `make contracts`.
Mutation: make the seed write on `dry_run=true`; make slugify keep spaces.
Acceptance: `POST seed_from_detector` on an empty project creates 80 classes
whose slugs round-trip to the detector's labels; the second call creates 0.
Minimum fallback if the release is too tight: ship only the `detector` block
in `ingest/config` plus a `--labels` option on the CLI seed script.

### W3. `embedding_state` and vector-less consumer fixes - v0.4.0 if time allows, else first in v0.4.x
Files: new `src/services/curation/embedding_state.py`; mapping
(`curation_opensearch.py`); `item_doc.py`; `ingest_index.py` (failure path
sets `failed`, counters); `ingest_models.py`; `wire.py`; `_item_models.py`;
`scripts/curation/vlm_worker.py:172` and `autolabel/selection.py:30`
(shared predicate); `review_queries.py:392`; `review_empty_reason.py`;
`routers/curation/stats.py` (+ `services/projects/stats.py`);
`semantic_search.py`/`search.py` (`unembedded_in_scope`); `crop_orders.py`
(`n_unembedded`); `combine/copy_docs.py:118`; optional backfill script;
contracts; table rows 1,2,6,7,12,13,16,18,20.
Under mode `all` behaviour is unchanged except failures become visible.
Tests (red-first):
- `tests/curation/test_embedding_state.py`: predicates and
  `normalize_class_name`; `name_matches` against proposal and class names.
- `tests/curation/test_ingest_embed_failure.py`: encoder raises -> items stored,
  `embedding_state='failed'`, `IngestResult.n_not_embedded == n`, no silent
  success (fails today).
- `tests/curation/test_embedding_state_consumers.py`: one parametrized fake-OS
  test (`tests/curation/query_fakes.py` `matches`) proving the VLM pending
  query, auto-label VLM selection, review `all` tab and residual fetchers all
  exclude a vectorless doc and include the embedded twin, and that no module
  outside `embedding_state.py` builds an exists clause on the item embedding
  field (grep-style test, fails today for `vlm_worker.py`, `review_queries.py`,
  `select.py`, `cluster_geometry.py`).
- `tests/curation/test_item_wire_embedding_state.py` plus the generated-contract
  tests (`tests/test_codegen_api_contracts.py`).
- `tests/curation/test_items_mapping_explicit.py`: add `embedding_state` keyword.
- `tests/curation/test_reingest_keeps_vector.py` (table row 24).
Mutation: drop the shared predicate from the VLM worker (the consumer test must
fail); write `embedded` on failure (the failure test must fail).
Acceptance: contracts regenerated and committed; `GET .../stats` shows the
breakdown; a vectorless item shows `failed` end to end.

### W4. Policy store, ingest gating, filter and caps - v0.5.0
Files: `src/routers/curation/_ingest_policy_models.py`, new route module
`src/routers/curation/ingest_policy.py` (imported by the curation router
package like the other routers), settings client functions in
`curation_opensearch.py` (`get_ingest_policy`, `put_ingest_policy` with
revision OCC), `_settings_body` mapping, `ingest.py:_get_ingest_service`
(load policy once per request), `ingest.py` service `detect_items`
(`DetectResult`) and its three callers, `ingest_index.py` (embedding gate),
`ingest_models.py`, `_project_models.py` + `projects/clone.py`
(`CLONEABLE_AXES += 'ingest_policy'`), `scripts/curation/ingest_walker.py`
(print counters), docs.
Tests (red-first):
- `tests/curation/test_ingest_policy_models.py`: validation (`selected` with
  no criterion 422; ranges; unknown names accepted with warnings).
- `tests/curation/test_embedding_selection.py` (pure): class by normalized name
  vs proposal and class names; min area; min confidence; `max_per_image`
  ordering and ties; labeled items always embed; AND/OR semantics.
- `tests/curation/test_detect_filter.py`: allow/deny lists, caps, counts, and
  that with the default policy output is byte-identical to today (fails if the
  filter ever drops by default).
- `tests/curation/test_ingest_policy_routes.py`: get defaults; put; stale
  `expected_revision` 409; preview counts match a seeded index.
- `tests/curation/test_ingest_selected_mode.py`: batch ingest of 3 fake images
  with `selected`: only chosen classes embedded (encoder fake records its
  inputs), others stored with `not_selected` and no vector, class-labeled
  cluster ids intact, IVF placement only for embedded.
- `tests/projects/test_clone_settings.py`: policy cloned; combine does not merge.
- Leak sweep: `tests/curation/test_cross_project_leak.py` entries; alpha policy
  never visible from beta (read and write).
Mutation: apply the filter when the policy is default; embed all in `selected`;
read the policy from the default project instead of the bound one.
Acceptance: policy round trip; ingest counters; contracts regenerated.

### W5. Embed-missing reprocess (item-unit, selective) - v0.5.0
Files: `reprocess_models.py` (`ReprocessFilter` += `embedding_state: list[str]`,
`proposal_name: list[str]`, `min_area_frac`, `min_confidence`; new
`EmbedOptions {only_missing: bool = True, parts}` on the request),
`reprocess_targets.py` (`item_filter_query`/`selector_clauses` clauses using
`embedding_state.py`), `reprocess_images.py` (`_embed_targets` keeps only the
selected crop ids; `parts` from options; `only_missing` for the crop part),
`reprocess_embed.py` (skip items that already have a vector when
`only_missing`; write `embedding_state`; shared placement function from
`ingest_index.py` so a newly embedded unclassed item gets its IVF cluster),
`reprocess.py` (`plan_reprocess` dry-run counts: `to_embed`, `already_embedded`,
estimated vector MB, `locked_skipped` stays 0 for embed because embeddings are
derived), `reprocess_job.py` progress, docs.
Tests (red-first): `tests/curation/test_reprocess_embed_missing.py` (selector
by class names; only-missing skips an embedded item's crop; locked item IS
embedded; state flips to `embedded`; placement assigns a residual cluster id
when a fake IVF store exists; job singleton: second concurrent start -> busy),
extend `tests/curation/test_lock_rule.py` / `test_reprocess_locks.py`
(embed never touches class fields), dry-run equals applied counts.
Mutation: ignore `only_missing`; let the embed write class fields.
Acceptance: after ingest under `selected`, embed-missing for a chosen class
produces vectors only for it, in one job, dry run exact.

### W6. Filtering over stored detections - v0.5.0
Files: `crops.py` (`proposal_name`, `embedding_state`, `min_area` params),
`crop_browse`/`review_request` plumbing, new route
`routers/curation/detections.py` (`get /detections/summary` with
`suggested_reprocess`), bulk exclude by filter (extend `POST
/curation/projects/{project}/crops/batch_exclude`, `crops.py:561`, with an
optional filter alternative guarded by dry run and a cap, or a reprocess-style
job), docs.
Tests (red-first): crops filter params (`tests/curation/test_crops_browse_params.py`
style), summary counts, bulk exclude dry run vs apply, exclusion keeps prior
cluster for undo (`test_crops_undo_exclude.py`).
Mutation: match proposal names case-sensitively.
Acceptance: "hide all `person`" is two calls, reversible, and exports stop
counting those items as unlabeled.

### W7. Lazy triggers - v0.5.0
Files: `autolabel/job.py` (`embed_missing` stage), `pipeline_params.py` /
`pipeline_start.py` (scope), the auto-label worker compose/service env check
(it needs Triton access; confirm before coding), docs.
Tests: stage ordering and cancel at boundary; stage skipped when nothing is
missing; scope limited to a class.
Acceptance: starting auto-label with an embed scope on a `lazy` project embeds
the in-scope items first, then clusters them.

### W8. Optional: by-name class resolution and per-project detector - after v0.5.0
`detect.class_resolution = by_name` (section 5.3) and per-project detector
selection (profile read from the policy, `_get_ingest_service` change; Triton
model existence check). Only if the owner wants them (Q4, Q5).

### W9. Close-out
Re-run the section 8 measurements, fill the "after" rows of PERFORMANCE.md,
update `docs/opensearch_schema_design.md` (`embedding_state`, the
with/without-vector rows), `docs-site` operations pages, `CHANGELOG.md`, the
frontend deltas doc, and decide the default flip (section 5.5 rule 4).

Ordering and dependencies: W0 and W1 are independent and first. W2 needs
nothing. W3 precedes W4 (the policy writes the state). W5 needs W3 and W4's
selector shapes. W6 needs W3. W7 needs W5. W8 last.

### Release placement
- v0.4.0 minimum slice that removes the vehicle-only limitation: W0 + W1 + W2
  (docs, measured cost table, detector info, one-call registry seed). With the
  quick start fixed, ingest stores and embeds all 80 classes (policy `all`
  semantics, which is today's code), the registry can be seeded from the
  detector in one call, and the env drop `OP_INGEST_PRIMARY_CLASS_IDS` remains
  the cost escape hatch until W4. No migration, no new stored field.
- v0.4.0 stretch: W3 (state + consumer fixes; it also fixes the silent embed
  failure). If the release is frozen, W3 is first in v0.4.x.
- v0.5.0: W4-W7. After: W8.

## 7. Risks

- Cost surprise: full vocabulary multiplies embed and VLM work 5x per image on
  COCO. Mitigated by W0 numbers in the docs, the response counters, the policy
  preview, and keeping the env hard drop documented until W4.
- Locks: embed writes only derived vector fields and `embedding_state`; it
  must never write class or region fields (test in W5). Reprocess `detect`
  with a tightened filter deletes unlocked machine items outside it by the
  existing `remove_stale` rule; dry run must show `removed`.
- Undo: exclusion stores prior cluster placement (`exclusion.py`); an item
  embedded after being excluded must not regain a residual cluster (placement
  skips `class_excluded`). Test in W5.
- Contracts: `embedding_state` on `ItemDoc` is an additive wire change; the
  Cropwright client must tolerate null (legacy). Regenerate contracts in the
  same commit (drift hooks).
- Mapping: adding `embedding_state` needs the items mapping to carry it; the
  repo does not migrate, stacks are recreated, so existing indexes keep a
  dynamic mapping (text + keyword subfield) for the field until recreated,
  which breaks terms aggregations on the bare name. Provide an additive
  `ensure_items_embedding_state_field` put-mapping call next to
  `ensure_items_embedding_fields` (`curation_opensearch.py:1352`) invoked from
  `_ensure_indexes` (`_common.py:288`), or document recreation. No knn field
  changes, so no kNN index recreation is required by this plan.
- Perf regressions: more stored items per image raise crop-cache writes and
  bulk sizes; `crop_rank_in_image` ranks among more items; the IVF ingest gate
  (`ingest_passes_gate`) parks low-rank items, which is helpful. Batch ingest
  concurrency (`OP_MAX_INGEST_CONCURRENCY`) is unchanged. Measure in W0/W9.
- Export honesty: unlabeled stored detections count against
  `require_fully_labeled_images`; ship W6 bulk exclude before telling users to
  keep everything for training exports.
- Residual field switch: `OP_RESIDUAL_EMBEDDING_FIELD` can select another
  field; predicates take the field as an argument; `embedding_state` tracks
  `pe_embedding` only (document).
- Silent incompleteness: kNN and ordered views drop vectorless items without an
  error; the added `unembedded_in_scope` / `n_unembedded` fields are the guard.
- Policy cache: 5 s settings cache means a policy change can apply up to 5 s
  late within one API process; the PUT invalidates the writing process
  immediately. Acceptable; documented.
- Isolation: policy and seed routes are project scoped; the leak sweep must
  prove alpha's policy and registry never reach beta (fail closed through the
  existing guard).

## 8. Verification plan (live stack)

Use only the public COCO sets; do not touch the running project's data; create a
throwaway project per run and delete it after. Do not use `git stash` or touch
the dirty runtime files in the repo (`models/yolov11_small_trt_end2end/config.pbtxt`,
`data/projects/`).

Baseline (W0, before any code change): with the current stack, throwaway
project `gen-base`, ingest the 4,000-image COCO set (manifest from #45; if only
200 are available use those and say so). Run once with the vehicle env set and
once with `OP_INGEST_PRIMARY_CLASS_IDS` unset (recreate `yolo-api`). Record per
run: wall time, images/s, detections/image, items store bytes (`GET
_stats` of the project items index, `size_in_bytes` / docs), images store
bytes/image, crop cache growth, embed forwards (Triton metrics), VLM items/s on
a 200-image sample, residual cluster run time.

After (W3-W7 on a branch image): same set, same machine, runs for: `all`;
`selected` with four vehicle classes; `lazy` then embed-missing for `person`;
detect filter on; per-image cap 5. Record the same columns plus
`n_embedded / n_not_embedded`, bytes per item with and without vector
(expect about 1 KB-1.8 KB vs about 9-10 KB), and per-image cost deltas against
section 4's model. Acceptance: measured storage per metadata-only item within
2x of 1.8 KB; `selected` run's images/s at least 1.5x the full `all` run;
embed-missing result count equals the dry-run count; vector counts equal
`embedded` counts.

Live tests to add under `tests/live/` (the existing pattern in
`tests/live/conftest.py`): `test_live_generic_ingest.py` (full vocabulary
appears: at least person and one animal class in `proposal_name` aggregation of
the sample), `test_live_embedding_policy.py` (policy put -> ingest 20 images ->
state counts -> embed-missing -> counts), `test_live_seed_from_detector.py`.
Visual check per convention: full-page screenshots, desktop and narrow,
opened and inspected, for the frontend surfaces in section 9.

## 9. Frontend deltas for Cropwright (numbered, as built)

All paths are project-relative (under `/curation/projects/{project}`). The
generated contracts (`contracts/openapi/curation.json`, `contracts/ts/*.ts`,
`contracts/json/*.json`) carry every shape; re-vendor them first.

1. `ItemWire.embedding_state`: `'embedded' | 'not_selected' | 'deferred' |
   'failed' | null` (null = written before the field). Badge for the three
   non-embedded states with a tooltip per state.
2. `GET /ingest/config`: `detector` block (model, version, input size,
   `assigns_class`, `n_labels`, `labels[{class_id, name, slug}]`,
   `confidence_floor_applies`; there is no env class-id filter any more) and a
   `policy` echo. Use it for the first-run screen instead of hardcoded
   vocabulary.
3. Ingest settings page over `GET/PUT /ingest/policy`. Shape: `{revision,
   detect: {classes|null, exclude_classes, min_confidence, min_box_area_frac,
   max_per_image, class_resolution: 'proposal'|'by_name'}, embedding: {mode:
   'all'|'selected'|'lazy', classes, min_confidence, min_box_area_frac,
   max_per_image}, detector: {model, version, input_size, labels_path}|null}`.
   `PUT` adds `expected_revision` (409 on stale: reload) and answers the policy
   plus `unknown_names` (warn, never block). 422: `selected` with no criterion;
   a `detector` that is not loaded on Triton or lacks the end2end outputs (a
   list of reasons); 503 when Triton cannot be asked. The detect filter and the
   detector override are collapsed "advanced" sections, off by default.
4. Cost preview: `POST /ingest/policy/preview` with the candidate policy body
   returns `{total_items, scanned, truncated, would_embed, would_not_embed,
   estimated_vector_mb, by_class[{name, would_embed, would_not_embed}]}`. Show
   "N of M stored detections would be embedded, about X MB".
5. Ingest results and batch summary: `n_embedded`, `n_not_embedded`,
   `n_filtered` (single-image response too). `failed` is a warning, not success.
6. First run: "Create classes from the detector" (`POST /classes/seed_from_detector`,
   dry run first).
7. One filter on every list: query parameters `class_name`, `exclude_class_name`
   (repeatable, by name), `conf_min`, `conf_max`, `min_area`, `max_area`,
   `max_rank` (N largest per image), `origin` (`detector|sam3|human|import`),
   `embedding_state`, `review_status` (`pending|validated|dismissed|excluded`)
   on `GET /crops`, `/review/{tab}` (+ `/locate`), `/search/text`,
   `/stats/classes`, `/stats/dataset`, `/clusters`, `/regions`,
   `/detections/summary`. `GET /review/tabs` lists them in every tab's `filters`.
   `GET /crops` also takes `open_vocab_set` and `source_prompt`. 400 on a
   malformed band.
8. `GET /detections/summary` (same filter): `{total, embedding{embedded,
   not_embedded, by_state}, by_label[{name, count, embedding}], labels_truncated,
   suggested_reprocess}`. Drives the "Detected as" facet and the class page
   "Embed N detections" action: POST `suggested_reprocess` to `/reprocess`
   (dry run first).
9. Filter, select, act: `POST /crops/batch_exclude`, `/crops/batch_unexclude`,
   `PUT /crops/batch_label`, `POST /crops/move` take `crop_ids` OR
   `selection: {filter, limit, sample: 'random'|'largest', seed, include_test,
   include_excluded}` and `dry_run`. A dry run answers `{dry_run: true,
   selected}`. 422 for both/neither forms, an empty filter without a limit, or
   more than 20000 items. Flow: show the count, confirm, run, undo toast.
10. `POST /reprocess`: `embed: {only_missing, parts}` and `targets.limit/sample/seed`
    (filter targets only); `ReprocessFilter` is the shared filter plus the
    reprocess selectors. Scope `embed` dry-run `detail`: `items`,
    `without_vector`, `to_embed`, `region_boxes_to_embed`, `estimated_vector_kb`.
11. Region box edits (`PUT /crops/{id}/regions`, `PUT /crops/batch_regions`,
    `PATCH /crops/{id}/regions/{box_id}`, `POST /regions/batch_box_state`,
    `PATCH /crops/{id}/region_meta`, `POST /regions/batch_status`) answer an
    extra `vector_refresh: {embedded, pending}`; `pending > 0` means a box has no
    vector yet (encoder down): offer the embed action.
12. `POST /pipeline/auto_label/start`: `embed_missing=true` and the shared filter
    as query parameters scope the embed and VLM stages; the job stage list gains
    `embed_missing` (first); `result.stages.embed_missing` is `{images,
    images_failed, embedded}` or `{skipped, reason}` or `{status: 'error'}`.
13. `POST /export/yolo`: optional `item_filter` (the filter as an object),
    recorded in the manifest as `item_filter`.
14. The regions review queue with the region profile off answers an empty queue with
    `empty_reason`; `/review/regions/locate` answers reason `region_profile_off`.
    Show the reason, not a blank grid.
15. Stats and empty states: `embedding_states` breakdown in curation stats,
    `items_embedded` in project counts, the review empty reason for unembedded
    detections; "unlabeled" is not "clusterable".
16. Ordered and semantic views: read `n_unembedded` / `unembedded_in_scope` and
    show a banner with the embed action.
17. `class_source` `<detector>_model` now also marks a by-name resolved class
    (`detect.class_resolution: by_name`): a machine label, not validated.
18. Contract refresh, then visual check: desktop and narrow full-page
    screenshots of the ingest settings page, the filter bar, the class page
    banner and the card badges.

Merged in from the SAM 3 full-image plan (its section 9, items 1 to 8, in
`docs/design/sam3_full_image_detection_plan.md`): the `/open_vocab` settings
page and target editor, the "test on an image" panel, the `open_vocab` reprocess
scope with `all_images` / `open_vocab_status`, item provenance fields, the
"skipped by gate" image state and the no-segmenter empty state. Items 7 and 8
above add `origin: sam3` (an item that carries an `open_vocab_set`) and the
embedding policy to those items.

## 10. Config, contracts and registration checklist (per wave)

- Typed request and response models for every new route (pydantic,
  `extra='forbid'` on requests), `response_model` set, no bare dicts.
- `make contracts` then commit generated files in the same commit (hooks
  `region-status-ts-drift`, `api-contracts-drift`); contract tests in
  `tests/test_codegen_api_contracts.py`.
- Leak sweep registration in `tests/curation/test_cross_project_leak.py`:
  `route_params` (L113), `route_bodies` (L135, valid body so no 422),
  `NO_WRITE` (L317, with a reason, only for routes that really do not write),
  `EXPECTED_5XX` untouched; `tests/projects/test_route_scoping.py` classifies
  scoped vs global.
- Items mapping explicitness: `tests/curation/test_items_mapping_explicit.py`.
- Naming guard `scripts/codegen/check_naming_leaks.py`: use only the public
  vocabulary (COCO classes, vehicle, person); no private names. Region field
  literals: new status names must come from the field config
  (`check_no_literal_region_fields.py`).
- File size hook (700 lines): keep modules split by concern (policy models,
  policy routes, selection, state, summary).
- Docs checks: any `METHOD /path` in docs must exist in the OpenAPI contract
  and any `OP_*` token must exist in code, `env.template`, compose or the
  installer scripts (`scripts/docs/check_docs_vs_code.py`); describe proposed
  routes with a lowercase method (as in this file) until they ship, then
  switch to the real form in the user docs.
- Pre-commit: `.venv/bin/pre-commit run --all-files` from the repo venv
  in the worktree; tests with the repo test environment.

## 11. Relationship to other plans

- Sibling #30 (`docs/design/sam3_full_image_detection_plan.md`): its hits are
  stored items; whether they are embedded follows this policy; targets already
  in the detector vocabulary should use the detector. Its dedup defaults
  should reuse `normalize_class_name` from `embedding_state.py` for the
  class-name equality warning.
- #45 supplies the COCO manifests used for every number here.
- #46 (gating): the shared gate for the segmenter reads `parent_classes`; with
  a full-vocabulary detector an empty `parent_classes` now means "every class",
  so shipped example profiles keep it set.

## 12. Open questions for the owner (with recommendations)

1. Default embedding mode: `all` (recommended, section 5.5) or `lazy`? Recommend
   `all` now, revisit after W9 with the flip rule.
2. Should the VLM stage be limited to the embedded working set (P3)? Recommend
   yes: one knob controls both embed and VLM cost, and it matches today's VLM
   worker behaviour. The alternative is a separate VLM scope knob.
3. Retire `OP_INGEST_PRIMARY_CLASS_IDS` through `src/config/retired_env.py`
   (fails loudly at startup) once the per-project filter exists? Recommend yes
   in W4, because a process-level drop by model id contradicts "store every
   detection" and ignores class identity by name; until then keep it documented
   as an optional hard drop.
4. Per-project detector selection (W8) in v0.5.0 or later? Recommend later:
   one deployment-wide detector is enough for the generic claim; revisit when a
   user needs two domains side by side.
5. By-name auto-assignment of detector labels to registry classes (W8)?
   Recommend off by default and later; today's proposal-only default is safer
   and the VLM or a human assigns classes.
6. Should the frame (whole-image) vector also be optional? Recommend no: one
   8.4 KB vector per image is small and frame dedup depends on it.
7. Is W3 allowed into v0.4.0, or must v0.4.0 be limited to W0-W2? Recommend
   W0-W2 for the release and W3 immediately after; W3 touches nine consumers
   and the wire contract.
