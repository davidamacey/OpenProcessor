# 15c: storage cut-over, the vertical slices

| Field | Value |
|---|---|
| Status | plan, revision 0 (2026-11-15): the cut-over of plan 15 section 10 steps 2 to 8 as vertical slices on `int/storage`, written from [15_storage_plan.md](15_storage_plan.md), [15a_storage_schema.md](15a_storage_schema.md) and the read map [15b_storage_reads.md](15b_storage_reads.md) |
| Code base read | `int/storage` at `7cadf2968`. Writer ids (I1..I11, T1..T21, G1..G9) and shape numbers (1..80) are those of the storage inventory (`artifacts_local/speed_1113/storage_inventory.md`, gitignored), sections 2 and 3; paths in it are relative to `src/services/curation/` unless they name another directory |
| Built already (step 1) | `src/domain/claim*.py`, `fold*.py`, `results.py`, `claim_match.py`; migration `0010_storage_record`; `src/db/repos/` (`photos`, `items`, `claims`, `results`, `state`, `fold_load`, `analysers`, `upserts`); `src/services/record/apply.py`; `src/db/vector_ddl.py`; `src/services/erasure/catalogue.py`. Nothing calls them yet |
| Data | wiped and re-ingested (owner, 2026-11-14): no slice migrates a search-index document |

## 0. Rules every slice follows

1. **The user's job first.** A person adds photos, a detector finds objects, the person labels their
   own classes cluster by cluster, edits boxes, freezes a holdout and trains; later the photos are a
   gallery. Each slice says which part of that job works again when it lands.
2. **Vertical.** A slice moves a feature family's writers and its readers together and deletes the
   old path in the same slice: no dual write, no compatibility shim, no dead code. Families not yet
   moved do not work on `int/storage` between slices (accepted). A shared old helper (for example
   `mget_crops`, `curation_opensearch/crops.py:79`) is deleted by the slice that moves its **last**
   caller, never earlier, so the offline suite always imports cleanly; each slice lists the helpers it
   deletes and the ones it leaves to a named later slice.
3. **Filters of families not moved.** The item filter translator (`src/db/queries/item_filter_sql.py`,
   15b P1) has one function per family; until a family's slice, its function raises
   `FilterNotMovedError` (400 with a served sentence naming the family). The family's slice replaces
   that function; nothing else is temporary.
4. **Lanes** (global rules, multi-agent lanes): one agent per slice in its own worktree and branch,
   one commit per concern, hooks on, never `--no-verify`, never push or merge. Lanes run unit and
   contract tests, lint, type checks and hooks on touched files only. Tests marked (DB), live and
   browser tests are written by the lane and run by the coordinator's wave sweep. A unit test is seen
   failing before its fix, and the commit message says how.
5. **Docs and contracts in the same commit.** A slice that deletes or adds a route, wire field or
   `OP_*` variable updates every doc that names it (`scripts/docs/check_docs_vs_code.py` runs in the
   hooks) and runs `make contracts`, committing `contracts/` with the change. A retired variable goes
   into `src/config/retired_env.py`. A vulture whitelist entry for a step-1 name
   (`scripts/codegen/vulture_whitelist.py`) is removed by the slice that wires the name.
6. **Package docs.** Every lane brief cites the pages read for what it touches: PostgreSQL 18 pages
   named in 15a and 15b, psycopg 3 (server-side cursors, `copy`), SQLAlchemy 2 asyncio, pgvector
   0.8.7, Alembic.
7. **Measurements.** A query 15b marks "to be measured" is timed in the first sweep after its slice
   (15b section 1.3). A miss is code work for the same slice's lane in the next wave.

## 1. Order, revisions and waves

| Slice | Plain name | Inventory proposal | Revision it writes | Depends on |
|---|---|---|---|---|
| S1a | Ingest writes the record | (1) | `0011_storage_indexes`, `0012_ingest_cutover` (drops `image_contents`) | step 1 |
| S1b | Browse: object and photo lists, detail, crops, locate, map | (1) | none | S1a's tables and indexes |
| S2 | Labelling, review queue, class counts | (2) | `0013_claims_cutover` (drops `label_operation_members`) | S1a, S1b |
| S3 | Vectors: index interface, search, vector jobs | (5) | `0014_vector_cutover` (15a 6.3 `embedding_spaces` change) | S1a, S1b |
| F | Files through the storage layer | (independent) | none | S1b (shares `image_serving.py`) |
| S4 | Region stage and box editing | (3) | none | S2 (human-act service), S3 (box vector writer) |
| S5 | Near-duplicates and the holdout | (6, first half) | `0015_near_dup_cutover` (bigint member keys; drops `near_dup_decisions`) | S2, S3 |
| S6 | Clustering and the cluster screens | (4) | `0016_cluster_cutover` (`cluster_memberships.subject_key`, X22, X23, cluster summaries) | S3, S4, S5 |
| S7 | Import, combine, export, training inputs | (6, second half) | `0017_import_cutover` (`dataset_import_images.photo_key`) | S2, S4, S5, F |
| S8 | Machine writers and re-runs | (7) | none | S2 to S6 |
| S9 | Deletions, docs, re-ingest, acceptance | (8) | `0018_reads_cutover` (drops `project_indexes`, `search_projection_tasks`) | all |

Changes to the proposed order, with reasons:
- **Slice 1 split into S1a and S1b**, two lanes in one wave and one sweep: the proposal's slice 1 is
  about 60 files (ingest, the wire serializer, every list and the cursor); one lane would exceed a
  lane's working size. The seam is the schema (`0010`, `0011`), fixed before either starts.
- **Vectors (S3) before clustering (S6) and regions (S4).** Clustering reads every vector of its pool
  (shape 59) and the region worker and box edits write box vectors (T3, T7, T12); building the vector
  writer and the streaming reader first means no later slice writes them, and clustering does not
  build a reader that S3 would replace.
- **Proposal slice 6 split into S5 (near-duplicates, holdout) and S7 (import, combine, export,
  training).** The holdout freeze needs the near-duplicate closure (`holdout_near_dup_moves`), and
  import and combine freeze holdouts and read boxes, so S7 must follow S4 and S5; the split keeps each
  lane at a normal size.
- **Revision ids.** 15a section 9's `0011` to `0014` are superseded by this table (15a 2.6). Two
  revision-writing lanes share wave B (S2, S3) and wave D (S6, S7): their `down_revision` chain is
  fixed above, so the coordinator merges S2 before S3 and S6 before S7.

```
step 1 ── S1a ─┬─ S2 ──┬─ S4 ──┬─ S6 ──┬─ S8 ── S9
         S1b ──┤       │       │       │
               ├─ S3 ──┼─ S5 ──┴─ S7 ──┘
               └─ F ───────────────┘(S7)
```

| Wave | Lanes (at most three) | Sweep covers |
|---|---|---|
| A | S1a, S1b | ingest, browse, summaries, 0011, 0012 |
| B | S2, S3, F | labelling, vectors, files |
| C | S4, S5 | regions, near-duplicates, holdout |
| D | S6, S7 | clustering, import, combine, export, training |
| E | S8 | machine writers |
| F | S9 | deletions, re-ingest, acceptance of plan 15 section 11 |

## 2. Contract diff addendum

15a section 10 lists the wire changes of the schema; these come from the read design (15b) and from
the writers. Each slice applies its rows; the acceptance checks the whole table.

| Wire field or route (model) | Change | Reason | Slice |
|---|---|---|---|
| `total_capped: bool` (`CropsPageResponse`, `TablePage`, `ImagePage`, `ReviewQueuePage`, `RegionRowPage`, `RegionFpPage`, `CropLocate`, `ImageLocate`, `ReviewLocateResponse`) | added; `total` is at most 10,000 when true | capped-count rule, 15b section 5 | S1b, S2 (review), S4 (regions) |
| `index_capped: bool` (the three locate models) | added | locate rank capped, 15b section 5 rule 3 | S1b, S2 |
| `capped: bool` (`HiddenCounts`, `frontend/src/lib/types.ts:295`) | added | 15b section 5 rule 4 | S1b |
| `join_truncated` (`ImagePage`, `ReviewQueuePage`) | deleted | the photo join is a semi-join with no cap (15b P6) | S1b, S2 |
| `rows_truncated` (`RegionRowBase`) | deleted | no inner-hits window (15b section 9) | S4 |
| `POST /image_counts/backfill` (`routers/curation/image_counts.py:24`), job kind `image_counts_backfill` (`config/job_kinds/image_counts.py:11`) | deleted | card counts are kept by the writer (15b section 6) | S1a |
| image list `sort_options` and the "not counted" refusal (`image_view.py:254-262`) | the `unlabeled_count` sorts are always offered; the refusal goes | the same | S1b |
| `crops_updated`, `crops_preserved_human`, `crops_final_conflicts` (ingest results, `ingest_index.py:607-610`) | deleted | a photo is stored whole or reported duplicate; no document is updated by ingest | S1a |
| ingest `error_kind` `incomplete_write` and the `held_back` answer (`ingest_index.py:565-570`) | deleted | no half-stored photo exists | S1a |
| order of equal sort values | under a `desc` sort ties come in descending `crop_id` (or `photo_key`, `item_key`) order | one direction per row comparison (15b section 4, M7) | S1b, S2, S4 |
| `GET /regions` item order | newest item first (was newest detected box) | 15b section 9 | S4 |
| text filters | a fragment under three characters answers 400 `text_too_short` | 15b section 8 | S4 |

The frontend reads of `total` that change (15b section 5): `pager.svelte.ts:123`, `:195`;
`MatchingItemsView.svelte:80`, `:138`; `ReviewHeader.svelte:79-82`; `reviewController.svelte.ts:69`,
`:77`, `:144`; `editorController.svelte.ts:169`, `:179`; `api_cropRead.ts:240`; `api_clusters.ts:147`.

## 3. The slices

Each slice: goal; writers (inventory id, today's function with file:line, what replaces it);
readers (shapes, forms in 15b section 3); deletions; wire; tests; keep-list; size and effort; what it
owes the wave. "Human-act service" is `src/services/record/human.py` (new in S2): open the operation
row (`records.create_label_operation`), build drafts with `src/domain/claim_builders*.py`, call
`apply()`, insert events, in chunks of 1,000 for bulk acts (15a section 7, bulk row).

### S1a. Ingest writes the record

**Goal.** A person adds photos (upload, batch, objects, sources, imports' image stage): each photo,
with its objects, detector classes, capture data, vectors and "created" events, is stored in one
transaction per commit group, or not at all; the same bytes again answer `duplicate`; a new project
gets its vector tables. Card counts follow every write.

| Id | Today | Replaced by |
|---|---|---|
| I1 | `IngestWriter._create_images` `ingest_writer.py:477` (bulk create `:491`), content claim `near_duplicates/ingest_claim.py` `claim_and_mark_in_session` | in `ingest_index.commit_group` (`:627`, rewritten): `photos.lock_imohashes` (moved from `repos/image_contents.py:71-89`), `photos.photo_keys_by_image_id`, `photos.insert_photos`; SHA-256 confirm by `photos.set_content_sha256`; `near_dup_pending` marks kept as today |
| T1 | `IngestWriter._write_items` `:572` (`occ_upsert_bulk` `:585`), `_create_missing` `:612` | same transaction: `items.insert_items`; `analysers.get_or_create_analyser` (detector, classifier profile, `capture_parser`, one encoder per space, region stage); `apply(session, project, results=[detections, class_opinions (classifier), item_region_results seed (region_seed_status, item_doc.py:37), photo_capture], item_keys=<new>, context, photo_scope=True)`; `vectors.insert_vectors` (new `src/db/repos/vectors.py`, one multi-row `INSERT` per group, table name `vector_ddl.qualified(lock_key, space_key)`) and `vector_gaps` rows; `analysers.record_ledger`; `events.insert_events_async` (`crop.created`, today `services/events/publish.py:184` after the write); `sum_photos` `+1` |
| I2 | `image_completeness.begin_repair` `:186`, `_release` `:217`, `forget_items` `:95` | deleted: a photo is stored whole by the transaction |
| I3 | `image_counts_refresh.recount_images` `:80` | deleted: `photo_counts` kept by `apply()` (15b 6.3) |
| I4 | `image_counts_dirty._clear_counts_at` `clients/image_counts_dirty.py:136` | deleted with the Valkey dirty set |
| I6 | `source_backfill/capture.py:79` `_write_images`; `location_purge.py:93` `purge_pass` | `results.upsert_results(session, project, [ResultWrite('photo_capture', rows)])`, `analysers.record_ledger` (kind `capture_parser`), the `sum_photos` located move |
| T20 | `source_backfill/capture._copy_to_items` `:91`; `location_purge.py:93` (items) | deleted: capture time is joined (15a 5.2) |

Parts of ids owned elsewhere, done here: T21's `scripts/curation/seed_live_harness.py` (`_bulk`
`:384`, `:522`, `:527`) seeds the record through the ingest repositories and `apply()` with drafts
from `claim_builders` for the human cohorts (every live test depends on it); I11's
`scripts/curation/backfill_item_ids.py:113` is deleted. Project creation (`projects/lifecycle.py:117`)
also creates the default spaces' vector tables through `src/services/embedding_spaces/ddl.py` (new,
the `op_vector` autocommit pool of 15a 6.1, `vector_ddl.create_statements`,
`vector_ddl.row_security_flags`); the approximate index and `ready` are S3's.

`apply()` gains the summary upkeep of 15b 6.3 (`src/db/repos/tallies.py`, `src/domain/tallies.py`,
`TALLY_NEUTRAL_TABLES`), with `sum_items`, `sum_region`, `sum_box`, `sum_photos`, `photo_counts` from
`0011`. Ingest placement in clusters (`ingest_index.place_prepared` `:188`, `item_doc.placement_fields`
`:101`) is deleted here with the documents it stamped; S6 places at ingest as membership rows.

**Readers.** 31, 34, 54, 55, 78, 80.

**Deletes.** `ingest_writer.py`; `image_completeness.py`; `repair_match.py`; `clients/image_counts_dirty.py`;
`image_counts_refresh.py`; `image_counts_backfill.py`; `tasks/image_counts.py`, the beat entry
`op.periodic.image_counts_drain` (`tasks/periodic.py:46`), `config/job_kinds/image_counts.py`,
`routers/curation/image_counts.py`; `item_doc.build_image_doc` (`:113`), `build_item_doc` (`:166`),
`placement_fields`; in `ingest_index.py` `build_documents`, `write_prepared`, `write_group`,
`place_prepared`, `doc_context` (rewritten as one commit step); `content_key.release_bound`;
`repos/image_contents.py` and its model (`db/models/near_dup.py:45`); `clients/curation_opensearch/count_fields.py`;
`IMAGES_INGEST_WRITE_MAPPING`; `scripts/curation/backfill_item_ids.py`. Tests: every test that imports
a deleted module (`rg -l 'ingest_writer|image_completeness|image_counts_|repair_match|backfill_item_ids' tests`),
replaced by the tests below.

**Wire.** Contract addendum rows for S1a; `image_path` in ingest results becomes the display path of
`photos.storage_key` (15a section 10).

**Tests.** Unit, red first: `tests/storage/test_tallies_offline.py` (`tally_rows` per table, delta
signs, a vanished and a new item), `tests/curation/test_ingest_commit_record.py` (the commit step's
drafts and result rows for a photo with two objects, a duplicate, a refused claim, an object without a
vector), `test_apply_offline.py` extended (every result table classified tally-neutral or not). (DB),
coordinator: `tests/db/test_ingest_record.py` (a group is stored whole; a crash between statements
leaves nothing; the same bytes twice give one photo and `duplicate`; two concurrent groups with the
same imohash give one photo), `tests/db/test_tallies_rebuild.py` (15b 6.4), `tests/db/test_vector_space_ddl.py`
(`op_app` cannot create in `vec`; the table has both row-security flags). Live, green again after
wave A: `tests/live/test_live_ingest.py`, `test_live_bulk_restart.py`, `test_live_whole_photo.py`
(rewritten: the transaction replaces the repair), `test_live_seed_from_detector.py`,
`test_class_registry_live.py`, `test_live_core_ingest.py` (global path: a red is a bug).

**Keep-list.** Duplicate bytes answer `duplicate` (`tests/curation/test_ingest_whole_photo.py`,
rewritten on the record); a photo is stored whole or not at all (`test_live_whole_photo.py`); the
per-photo flow of plan 13a with its 503 `database_busy` on a pool timeout (`tests/curation/test_ingest_finish_batch.py`);
`crop.created` once per created item (`tests/curation/test_ingest_batched_inference_and_events.py`);
the location policy drops positions (`tests/curation/test_capture_backfill.py`, `test_export_gps_by_policy.py`
later in S7); the embedding policy's skip states (`tests/curation/test_item_wire_embedding_states.py`).

**Size.** About 35 files, 2,500 lines changed, 2,000 deleted. Effort: high (transaction and lock
order, migration, concurrency of duplicate claims).

**Owes the wave.** The (DB) tests above; the plan 13 four-way ingest table on the public sets not
below its measured numbers, host CPU per photo recorded; `0011` and `0012` applied on the harness.

### S1b. Browse: object and photo lists, detail, crops, locate, map

**Goal.** The person sees their photos and objects: the object list with every sort and filter of
the moved families, the photo gallery with card counts, a photo's page, crops and thumbnails, the map,
and "where is this item in my list". Totals are exact for class and flag filters and "10,000+"
otherwise.

**Writers.** None (the family's writers are S1a's).

**Readers.** 1, 3, 4, 5, 7, 9, 24, 25, 26, 27, 28, 29, 30, 32, 33 (15b section 3), through
`src/db/queries/item_filter_sql.py` (P1 translator), `src/db/repos/item_reads.py` (pages, locate,
counts), `src/db/repos/item_records.py` (`load_item_records`, P2), `src/db/repos/photo_reads.py`
(P6, P15), `src/db/queries/totals.py` (`exact_total`) and `src/domain/list_totals.py` (`TOTAL_CAP`).
`list_cursor.py` keeps `encode_cursor`, `decode_cursor`, `cursor_carry` and replaces `page_body`,
`with_tiebreak`, `split_page` with `keyset_clause` and `split_rows` (15b section 4). `serialize_item`
(`wire.py:288`) and `serialize_items` (`:486`) take the typed record. Crop and thumbnail serving
(`image_serving.py`, routes `routers/curation/images.py:103`, `:134`, `:197`, `:242`, `:308`) read the
photo through `photos.storage_key` and the crop box from `item_state.bbox`; the crop cache keyed by
`crop_id` stays until F.

**Deletes.** `image_join.py` (`joined_image_ids`, `item_image_ids`); the OpenSearch bodies of
`crop_listing.py` (`crop_list_body` `:85`, `crops_by_ids` `:120`), `list_locate.py`,
`hidden_counts.py`, `image_view.py`, `image_counts.py` (`item_facts`), `image_detail.py`,
`map_cells.py`, `table_view.py`, `capture_wire.py:64`, `list_counts.page_counts` (OpenSearch hit
totals); `wire.item_source_excludes` (`:92`), `item_list_source_excludes` (`:99`),
`item_context_source_excludes` (`:114`); `source_image_path` (`wire.py:322-323`). Left to later
slices, with their last callers: `curation_opensearch/crops.py` `mget_crops` (S9), `region_to_wire`
storage-name mapping (`wire.py:121`, S4).

**Wire.** Addendum rows for S1b; 15a section 10: `image_path` display value, `source_image_path`
deleted, `class_unresolved` added. Frontend: `frontend/src/lib/types.ts`, the pager and the four
components named in section 2, generated types from `make contracts`.

**Tests.** Unit, red first: `tests/curation/test_list_cursor.py` (rewritten: ISO microsecond
timestamps round-trip; tie direction follows the sort; the missing-last segment resumes; a foreign
cursor is refused), `tests/storage/test_item_filter_sql.py` (each moved filter gives the SQL predicate
named in 15b P1; a family not moved raises `FilterNotMovedError`), `tests/storage/test_totals_rule.py`
(exact for summary keys, capped otherwise, the 10,001st row sets `total_capped`). (DB):
`tests/db/test_item_reads.py` (page, deep keyset, locate with cursors, capped count on 10,001 seeded
rows, a NULL sort value pages last in both directions, item filter on the photo list as a semi-join).
Live, green after wave A: `test_live_images.py`, `test_live_paging.py`, `test_live_capture_map.py`
(rewritten to ingest its photos instead of the 0.6.0 carry-in). Browser, green after wave A:
`frontend/e2e/live/test_map_renders.py`; the stubbed tier (`frontend/e2e/stubbed/test_gallery_view.py`,
`test_item_filter.py`, `test_map_view.py`) with the new fields. Red until later: the per-route cases
of `frontend/e2e/live/test_route_sweep.py` for routes of unmoved families (green at S8).

**Keep-list.** Cursor refused across sorts (`tests/curation/test_crops_cursor.py`); locate agrees
with the page that follows it (`tests/curation/test_crops_locate.py`, `test_images_locate.py`,
`test_live_paging.py`); hidden counts exclude holdout and excluded as today (`tests/curation/test_hidden_counts.py`);
card counts equal the items (`tests/curation/test_image_counts_view.py`, `test_live_images.py`); the
location policy on the map and the photo page (`test_live_capture_map.py`); the thumbnail `?v=`
changes with the revisions (`tests/curation/test_geometry_wire.py`).

**Size.** About 30 files, 2,500 lines changed, 1,500 deleted, plus the frontend (about 8 files).
Effort: high (cursor correctness, wire serializer).

**Owes the wave.** Timing of every new page and capped count of rows 1, 3 to 5, 7, 24, 25, 28 to 30
at 10 M and on set P (15b 1.3); the browser run of the map page at desktop and 430 px.

### S2. Labelling, review queue, class counts

**Goal.** The person labels: one item, a batch, a cluster move, accept machine labels, adopt the
detector's class, unlabel, discard, exclude and restore, dismiss, flag "needs a new class", resolve
proposals, draw an object, move its box, start an audit, undo any of it; the review queue, class
counts, dataset statistics and label-operation list read the result at once.

| Id | Today | Replaced by |
|---|---|---|
| T2 | `create_drawn_item` `image_item_write.py:125` (`index` `:198`) | human-act service: `claim_builders_region.draw_object`, `items.insert_items` under the photo lock (`photos.lock_photos`, rank 5), `apply(photo_scope=True)` |
| T5 | `recorded_relabel` `label_ops/record.py:105` → `crop_bulk.occ_bulk_human_relabel` `:54`; `run_relabel` `record.py:178`; `accept_merger` `label_ops/accept_merge.py:105`; adopt detector (`class_label.py:110-122`) | `claim_builders.label`, `move_to_cluster`, `accept_machine_eligible` + `accept_machine`; adoption writes `class_opinions` rows (`adopted_detector`, `op_id`) through `upsert_results` inside `apply()`; batches in chunks of 1,000 with `append_claims`' `ON CONFLICT DO NOTHING` (M4) |
| T6 | `label_ops/undo._restore` `:116`; `label_undo._occ_write` `:74`, `discard_many` `:171`, `review_dismiss` `:248`; `class_undo.py:90`; `edit_undo.py:82`; `routers/curation/edit_undo.py:77` | `claim_builders.unlabel`, `discard`, `review_dismiss`, `review_undismiss`, `vlm_dismiss`, `probe_dismiss`; undo: the undo job walks `claims.live_claims_of_operation`, writes `claim_builders.withdraw` drafts and folds per chunk in one transaction (15a 3.5); step-back undo of one item withdraws its newest live operation's claims; adoption undo deletes the operation's `class_opinions` rows (X17) |
| T8 | `exclusion._occ_bulk_human_write` `:106`; `item_geometry.edit_geometry` `:269`, `undo_geometry` `:302`, `rerank_image_items` `:219` | `claim_builders.exclude`, `restore`, `claim_builders_region.set_geometry`; ranks are `apply()`'s (geometry is a rank-changing kind) |
| T9 | `crops_write.flag_new_class` `:245`; `new_class_resolve.py:95`; `audit.start_audit` `:140` | `claim_builders.flag_new_class`, `resolve_new_class`, `audit_sample` |
| T16 | `classes/projection_sync.relabel_merged_items` `:191` and its `update_by_query` | nothing at merge time (readers use the class family, M16); the merge job walks X2 in keyset chunks and updates `item_state.class_id` with the summary deltas of 15b 6.2 |

**Readers.** 2, 15, 16, 19, 36, 38, 43, 46, 47, 66. Review tabs become SQL predicates generated from
`src/domain/review_tabs.py` (new) with the property test of 15b 6.2; tabs that need a family not
moved (cluster fit: S6; region text: S4; probe: S8) raise `FilterNotMovedError` until then.
`count_items(filter)` (row 38) is the one count function; callers outside labelling switch in their
slices.

**Deletes.** `label_ops/crop_bulk.py`; the document merges of `class_label.py`
(`class_label_update` `:208`, `candidate_move_update` `:239`); `domain/history.py` and
`services/curation/edit_history.py` (claims are the history); `label_undo.py` OpenSearch writes;
`class_undo.py`; `edit_undo.py`; the OpenSearch parts of `exclusion.py`, `item_geometry.py`,
`new_class_resolve.py`, `audit.py`, `accept_machine.py`, `detector_adoption.py`;
`classes/projection_sync.py` and the class rows of `search_projection_tasks`
(`repos/classes._queue_projection`); `clients/opensearch/scroll.py` `scroll_ids` (last caller);
`records.add_label_operation_members`, `_recount_members`, `label_operation_members` and the table
(`0013`). Left: `occ.py`, `occ_bulk.py`, `occ_upsert.py` (S9, last callers in S4, S7, S8).

**Wire.** 15a section 10: `conflicts`, `conflict_ids` deleted (`schemas/curation/label_ops.py:77-78`);
`kept_changed` semantics. Addendum: `total_capped`, `index_capped`, `join_truncated` on the review
models.

**Tests.** Unit, red first: `tests/curation/test_human_act.py` (drafts per route, chunking, a replayed
chunk appends nothing), `tests/domain/test_review_tabs_sql_python.py` (property: both encodings select
the same rows), `tests/curation/test_label_op_undo.py` rewritten on claims (outcomes `restored`,
`kept_changed`, `kept_holdout`, `already_undone`). (DB): `tests/db/test_label_writes.py` (two sessions
label the same item: no lost claim, state equals a fresh fold; undo killed after a chunk resumes; class
merge job keeps summaries equal to a rebuild), `tests/db/test_review_queue.py`. Live, green after wave
B: `test_live_labels.py`, `test_live_annotation.py`, `test_live_detector_class.py`,
`test_job_recovery_real_handlers.py`, `test_live_image_counts.py` (rewritten: counts follow a label in
the same request). Red until S5: `test_live_classes.py` (holdout freeze), `test_one_vs_n.py`
(near-duplicate runs). Browser: `frontend/e2e/stubbed/test_labeling_flow.py`, `test_label_ops.py`,
`test_cluster_bulk_undo.py` with new fields.

**Keep-list.** Step-back undo over label, batch label, move, resolve, discard and not over a bulk
accept (`tests/curation/test_label_op_undo.py`, 15a Q4); discard undo restores class and review state
together (M12); exclusion never leaves an item validated (`test_exclude_reasons.py`, M2); a person
may relabel a frozen holdout item (15a question 1); class identity by name across merge (`tests/integration/test_class_identity_e2e.py`);
the review sorts' clauses and coverage fallbacks (`test_review_sorts.py`, `test_review_sort_coverage.py`);
label-op current counts (`test_label_ops.py`).

**Size.** About 40 files, 3,000 lines changed, 3,500 deleted. Effort: high (locks, undo chunks,
concurrency, migration).

**Owes the wave.** (DB) tests; timing of rows 2 (each default tab sort), 15, 16 against the bars; the
label route's p99 with summary upkeep under the write-under-read load (15b 6.5).

### S3. Vectors: index interface, search, vector jobs

**Goal.** Semantic search over the person's objects works from the record: by text, by a similar
item, by an uploaded image, over item or box vectors; a new embedding space is filled and becomes
ready; the jobs that need every vector read them in one stream.

| Id | Today | Replaced by |
|---|---|---|
| I10 | `embedding_spaces/reindex_opensearch.py:236` `write`, `:285` `mark`, `:363` `clear` (frame vectors) | `vectors.replace_vectors` (staging `COPY` then `INSERT ... SELECT`, 15a section 1) and `vector_gaps` writes, in `src/services/embedding_spaces/reindex_vectors.py` (new) |
| T12 | `reprocess_embed._write_item_vectors` `:250`, `_bulk_update` `:231`; `reindex_opensearch.py` `:236`, `:285`, `:363`; `region_box_embeddings.write_box_embeddings` `:184` (via `refresh_box_embeddings`); `scripts/curation/backfill_region_embeddings.py` | `vectors.replace_vectors` for items and frames; `write_box_embeddings` rewritten to write the box space's table (its callers in T3 and T7 move with S4); the script deleted (the box-space re-index covers it) |

`0014_vector_cutover` applies 15a 6.3 (`vector_table`, `vector_index`; the OpenSearch name columns
dropped); `KnnMethodSpec`, `physical_names`, `legacy_names` (`domain/embedding_spaces.py:70-135`)
deleted with their last callers here. The index interface `src/services/vector_index/` (new): add or
replace, delete, search with a filter, rebuild (plan 15 section 7); the default implementation is
pgvector HNSW on `halfvec` built `CONCURRENTLY` after the fill, `indisvalid` checked before `ready`
(15a 6.3); the three search paths of 15a 6.4 with thresholds as settings whose first values are 20,000
(`hnsw.max_scan_tuples` default) and 200,000, replaced by gate B's measured values.
`src/db/repos/vector_stream.py` (new) is P11.

**Readers.** 14, 20, 21, 22, 23, 35, 37, 59, 64, 65.

**Deletes.** `reindex_opensearch.py`; `semantic_search_run._search_body`, `build_knn_query`,
`_count_body` and `msearch_checked` use for search; `semantic_search_box.build_box_knn_query`;
`reindex_encoder.py:76` mget; `domain/embedding_state.py` `embedded_clause`,
`not_embedded_clause`; `embedding_spaces/readers.py` OpenSearch field names. Left: `clustering/pit_pages.py`,
`clustering/vector_read.py` (S6), `near_duplicates/vectors.py` (S5).

**Wire.** None (search responses unchanged; `unembedded_in_scope` from P4, never capped above the
page's use: it is shown as a count, so it carries `total_capped` semantics only if over 10,000, noted
in its description).

**Tests.** Unit, red first: `tests/storage/test_vector_index_paths.py` (the path chosen per filtered
count), `tests/embedding_spaces/test_reindex_vectors.py`. (DB): `tests/db/test_vector_search.py`
(exact path equals brute force; a filtered search returns full depth; a killed index build is
dropped and rebuilt; `op_app` cannot create in `vec`), `tests/db/test_vector_stream.py` (key order,
binary round-trip of `halfvec`). Live, green after wave B: `test_live_search.py`,
`test_live_embedding_state.py`, `test_live_generic_detector.py`. Browser stubbed:
`test_embedding_spaces_page.py`, `test_embedding_state.py`, `test_image_search_popover_430.py`.

**Keep-list.** Fused search by reciprocal rank in Python (`tests/curation` semantic search tests, `rg
-l semantic_search tests`); items without a vector are counted, never silently missing
(`test_live_embedding_state.py`); the status machine `TRANSITIONS` (`domain/embedding_spaces.py:60-67`);
float16 storage (15a 6.2).

**Size.** About 25 files, 1,800 lines changed, 1,500 deleted. Effort: high (DDL role, concurrent
index build, migration).

**Owes the wave.** Recall and latency of P12 on set P vectors (exact against brute force); the
stream rate at 1 M; gate B continues separately.

### F. Files through the storage layer

**Goal.** Originals, uploads, crops, exports and training files are reached through `RootedStore`
by key (plan 15 section 8, R9), so a second store implementation is possible. Today's callers that
open paths directly: inventory section 7.

**Moves.** Originals by `photos.storage_key` (`image_serving.resolve_safe_path` `:196`,
`_render_source_image` `:455`, `resolve_crop_root` `:553`); uploads (`persist_uploaded_bytes` `:152`,
`_upload_dest` `:187`, `upload_ingest.py:89`); the crop and thumbnail cache keyed by photo, box and
size (`source_image_cache.crop_cache_file` `:69`, `write_crop_cache` `:82`, `crop_bytes.crop_jpeg_from_cache`
`:45`, `load_vlm_item_jpeg` `:159`, `ThumbnailCache` `image_serving.py:281`), with a size cap and no
`/dev/shm` assumption; exports (`export_support.allocate_export_dir` `:435`, symlinks inside the local
implementation); training (`services/training/jobs.py:130` passes a store key, the trainer resolves
it); dataset archives (`dataset_import/upload_job.py:46`, `:51`, `scan.py`, `image_load.py`,
`capped_io.py`); the false-positive store (`detection/fp_store.py:34-47`). The global FAISS files
(`visual_search/clustering_persist.py:36-67`) stay with section 1.2 until phase 3.

**Shapes and writers.** None of the 80 shapes or 41 writers. **Deletes.** `BlobStore.local_path` use
outside the local implementation (`clients/storage/base.py:42`); `domain/storage_keys.crop_key`
whitelist entry (now called).

**Tests.** Unit: `tests/storage/test_rooted_store_callers.py` (an `rg` ratchet: no `Image.open` or
`open()` on a stored path outside `clients/storage/`), crop cache key tests. Live, green after wave B:
`test_live_images.py` (file and thumbnail routes), `test_live_export.py` stays red until S7.
**Keep-list.** `tests/curation/test_crop_cache_project_scope.py`, `test_crop_cache_prune.py`,
`test_export_dir_allocation.py`, `test_export_retention.py`. **Size.** About 20 files, 1,200 lines.
Effort: medium. **Owes the wave.** The live file routes; a crop served after a cache purge.

### S4. Region stage and box editing

**Goal.** The region stage finds boxes inside objects and reads their text; the person creates,
moves, accepts, rejects and types boxes, closes a box set, gives a set verdict, undoes; box rows,
region drain and text search read it.

| Id | Today | Replaced by |
|---|---|---|
| T3 | `_bulk_update_one_project` `scripts/curation/worker/bulk_writer.py:111` (`occ_skip_on_conflict_bulk` `:269`); `_write_box_embeddings` `:329` → `region_box_embeddings.write_box_embeddings` `:184` | `src/db/repos/region_results.py` (new) `replace_region_boxes` (old ids kept by IoU through `claim_match.match_boxes`, new ids from `items.box_seq`, `items.region_seq` +1) and `upsert_results` for `item_region_results`, `box_texts`, `class_opinions` (the cascade's class), all inside `apply()`; box vectors by S3's `write_box_embeddings`. No lock check (R5): the closed set is the fold's (15a 4.3 layer 4) |
| T7 | `regions_edit._write_one` `:101`, `_batch_write` `:257`; `regions_boxes_edit._write_one_boxes` `:228` (`region_writes.human_box_write` `:68`, `human_status_box_write` `:114`) | human-act service with `claim_builders_region.put_regions`, `patch_box`, `batch_box_state`, `set_region_status`, `set_rejection_reason`; `expected_region_revision` checked against `item_state.region_revision` under the item lock (409 `region_revision_conflict`, 15a section 7); box vectors refreshed after the commit |

**Readers.** 6, 40, 42, 49 (the worker's poll; the VLM worker's variant switches in S8). The box
and text filter functions of the translator replace their `FilterNotMovedError`.

**Deletes.** `bulk_writer.py` OpenSearch parts; `scripts/curation/worker/crop_gate.py` lock checks;
the OpenSearch parts of `regions_edit.py`, `regions_boxes_edit.py`, `region_writes.py`,
`region_rows.py`, `region_drain.py`, `region_impact.py`, `cascade._fetch_pending`,
`region_text_search.py`; `curation_opensearch/ensure_overlay_fields.inner_result_window`;
`wire.region_to_wire` storage mapping and `RegionFields.from_env` with the `OP_REGION_FIELD_*`
variables (15a 5.5) into `retired_env.py`; the `edit_history` region kind (already gone in S2).
`tests/live/test_live_bulk_occ.py` is replaced by `tests/live/test_live_concurrent_claims.py`
(review resolve and region undo from two clients: no lost claim).

**Wire.** `rows_truncated` deleted; region rows order; `text_too_short`; `total_capped` on region
pages.

**Tests.** Unit, red first: `tests/curation/test_region_results_replace.py` (ids kept by IoU, a split
keeps the claimed box), `tests/curation/test_regions_boxes_edit.py` rewritten on claims. (DB):
`tests/db/test_region_writes.py` (a worker write racing a box edit: the person's box survives, the
machine box is stored and not shown in a closed set; `region_revision` grows), box-row pages and
`sum_box` totals. Live, green after wave C: `test_live_regions.py`, `test_live_region_drain_narrowed.py`,
`test_live_region_text.py`, `test_live_ocr.py`, `test_live_concurrent_claims.py`; browser
`frontend/e2e/live/test_data_agreement.py`, `test_deep_link.py`. Red until S8: `frontend/e2e/live/test_writes.py`
(reprocess).

**Keep-list.** Whole-set verdict semantics (`test_region_verdict_semantics.py`, 15a Q7); region undo
and its batch (`test_region_undo.py`, `test_region_undo_batch_bulk.py`); case-insensitive text search
(`test_regions_text_search_case_insensitive.py`); region paging (`test_regions_paging.py`); a human
accept leaves `locked` false on the wire (15a 4.4).

**Size.** About 30 files, 2,200 lines changed, 2,000 deleted. Effort: high (worker concurrency, OCC
replaced by locks).

**Owes the wave.** Timing of X6 pages and the trigram filter at 10 M (`pg_ext`), worker throughput on
set P against the plan 13 region numbers.

### S5. Near-duplicates and the holdout

**Goal.** Near-duplicate runs group the person's photos and objects, verdicts are recorded, and
freezing the test holdout pulls each group to one side so the test split does not leak.

| Id | Today | Replaced by |
|---|---|---|
| I5 | `near_duplicates/projection.py:93` `_bulk`, `apply_run` `:127` (images) | nothing for the project indexes: readers join `near_dup_members` of the current run (15a 2.3); the flip of the current run rebuilds `sum_cluster_dup` once S6 exists. The global path of `apply_run` (G6) stays |
| T11 | `near_duplicates/projection.py` `:93`, `:127` (items); `near_duplicates/holdout.flag_held_out` `:73` | the same for items; the freeze (`holdout_freeze.freeze_test_holdout` `:91`, C3 order kept: snapshot file, `records.freeze_holdout`, claims, `near_dup_holdout.record_moves`) writes `claim_builders.holdout_freeze` drafts as a bulk human act in chunks |

Also here, not an inventory writer id: near-duplicate verdicts (`repos/near_dup.upsert_decision`,
`delete_decision`) become `claim_builders.duplicate_verdict` claims and withdraws; the
`near_dup_pending` ids become keys. `0015_near_dup_cutover` turns `near_dup_members.member_id`,
`near_dup_edges.a_id`, `b_id`, `holdout_near_dup_moves.item_id` into `bigint` keys and drops
`near_dup_decisions`.

**Readers.** 45, 61, 63; call sites of 34 (`near_duplicates/holdout.py:170`) and 59
(`near_duplicates/vectors.py:87`, `:120`, `chain.py:43`) switch to S1a's and S3's functions.

**Deletes.** `near_duplicates/projection.py` project-index paths, `near_dup_fields.py`; the
OpenSearch reads of `near_duplicates/vectors.py`, `candidates.py`, `rescore.py`, `member_views.py`,
`status.py`, `holdout.py`; `repos/near_dup.upsert_decision`, `delete_decision`.

**Wire.** None.

**Tests.** Unit, red first: verdict claims and withdraws, freeze chunking. (DB):
`tests/db/test_holdout_freeze.py` (a freeze killed midway resumes; a frozen item keeps its frozen
class against later opinions, M1), `tests/db/test_near_dup_keys.py`. Live, green after wave C:
`test_live_classes.py`, `test_one_vs_n.py`; browser stubbed `test_near_duplicates.py`,
`test_export_split_counts_and_holdout_freeze.py`.

**Keep-list.** `tests/curation/test_holdout_freeze_errors.py`; split leakage
(`test_export_split_leakage.py`, checked again in S7); `tests/near_duplicates/` group semantics.

**Size.** About 18 files, 1,200 lines. Effort: high (migration, bulk human act).

**Owes the wave.** Near-duplicate run time on set P against today's; freeze time for a 10 % holdout
at 1 M.

### S6. Clustering and the cluster screens

**Goal.** Clustering runs group the person's objects; the cluster screens (cards, representatives,
facets, region clusters, ordered pages) read the serving partition; activation flips a pointer; a new
photo is placed in the serving partition at ingest.

| Id | Today | Replaced by |
|---|---|---|
| T10 | `clustering/projection._bulk` `:147`; `clustering/region_box_rows.write_box_edits` `:232` | nothing: readers join `cluster_memberships` of `cluster_activations.serving_partition_id` (15a 2.3, M6); activation (`services/clustering/activation.py`) also builds the partition's summaries (15b 6.4) in its transaction; box partitions are read by join |

Ingest placement: in S1a's commit transaction, `assign.place_new` results become `cluster_memberships`
rows of the serving partition before `apply()` (the 02d immutability trigger guards only `UPDATE OR
DELETE`, `db/migrations/schema_0002/security.py:152-153`). `services/clustering/partitions.py` writes
`subject_key`. `0016_cluster_cutover`: `cluster_memberships.item_key text` → `subject_key bigint`,
X22 replacing `ix_cluster_memberships_cluster`, X23, the four cluster summaries. The cluster filter of
the translator and the cluster-fit review tabs replace their `FilterNotMovedError`.

**Readers.** 8, 10, 11, 12, 13, 41, 60, 79; call sites of 59 for clustering, region false-positive
rows and the outlier and diverse orders switch to `vector_stream`.

**Deletes.** `clustering/projection.py`, `clients/curation_opensearch/cluster_fields.py`, the cluster
rows of `search_projection_tasks`; `clustering/pit_pages.py` and `clustering/vector_read.py` (last
callers); OpenSearch bodies of `cluster_cards.py`, `cluster_facets.py`, `clusters.py`,
`region_fp_clusters.py`, `crop_orders.py`, `residual_fetch.py`, `outliers.py`, `pool_snapshot.py`,
`periodic/cluster_refresh.py`.

**Wire.** None.

**Tests.** Unit, red first: cluster summary deltas for a move, an exclusion, a placement.
(DB): `tests/db/test_cluster_summaries.py` (cards from summaries equal the join aggregation after
random writes; activation of a retained partition rebuilds), `tests/db/test_cluster_placement.py`.
Live, green after wave D: `test_live_clusters.py`; browser stubbed `test_cluster_*.py`,
`test_clusters_empty_states.py`.

**Keep-list.** Card fields (`tests/curation/test_cluster_cards.py`, `test_cluster_card_near_dups.py`,
`test_cluster_card_dominant.py`, `test_cluster_hidden.py`); facets (`test_cluster_facets.py`); an old
move is not placed (`test_old_move_not_placed`, 15a 4.5).

**Size.** About 25 files, 1,800 lines changed, 1,500 deleted. Effort: high (migration, summary
rebuild in activation).

**Owes the wave.** Rows 10 to 13 at 10 M (with `pg_sum` and `pg_pl` numbers when the chain ends);
a clustering run's read time at 1 M vectors.

### S7. Import, combine, export, training inputs

**Goal.** The person imports a labelled dataset, combines projects, exports a YOLO dataset with a
leak-free split, and trains from it.

| Id | Today | Replaced by |
|---|---|---|
| I8 | `dataset_import/chunk.py:96` `_stamp_existing_image` | `photo_provenance` rows (kind `import`) |
| I9 | `projects/combine/image_copy.py:231` `_bulk` | photos copied through S1a's commit step with `photo_provenance` (kind `combine`) |
| T17 | `dataset_import/item_labels.py:179`, `:209`; `proposals.py:112`; `region_labels.py:146`; `undo.py:426`; items via I1/T1 | `claim_builders_import.import_label`, `import_item`, region claims with `actor_kind 'import'`; `class_opinions` `import_suggestion`; detections, `region_boxes`, `item_notes` with `import_id`; `claim_match` attach under the photo lock; `fold.conflicts` gives `locked_conflict`; undo withdraws the operation's claims and deletes the `import_id` rows (15a 3.5) |
| T18 | `combine/copy_docs.py:46`; `combine/holdout.py:91` | claims with `actor_kind 'combine'`, `combine_conflict` claims; holdout by S5's freeze |

`0017_import_cutover`: `dataset_import_images.photo_key`, `item_ids` dropped; `dataset_imports.dataset_name`
written. T14's call site in import undo (`item_delete._delete_round` `:145`) is replaced by attach;
the module goes in S8.

**Readers.** 39, 58, 67, 68; call sites of 38 in export readiness (`export_readiness.py:114`) and
training preflight (`training/preflight_checks.py:181`).

**Deletes.** OpenSearch parts of `dataset_import/` (`chunk.py`, `item_labels.py`, `proposals.py`,
`region_labels.py`, `undo.py`, `project_source.py`, `image_plan.py`, `reconcile.py`, `runner.py`),
`projects/combine/` (`image_copy.py`, `copy_docs.py`, `holdout.py`, `plan.py`, `warnings.py`),
`export_support.scroll_hits` and export bodies, `proposal_merge.py:109-170` (replaced by
`claim_match`, 15a 3.4).

**Wire.** 15a section 10: combine labels show `external_label`; `frame_mismatch` on import results.

**Tests.** Unit, red first: import drafts per label trust; combine conflicts. (DB):
`tests/db/test_import_record.py` (an import never overrides a person, B2; undo leaves items a person
claimed), `tests/db/test_export_snapshot.py` (an export during labelling sees one snapshot). Live, green
after wave D: `test_live_combine.py`, `test_live_export.py`, `test_live_train.py`,
`test_live_train_promote.py`, `test_live_models_promote.py`; browser stubbed `test_dataset_import.py`,
`test_combine_projects.py`, `test_export_yolo_synchronous.py`.

**Keep-list.** Export golden files (`tests/curation/test_export_dataset_golden.py`); split leakage
(`test_export_split_leakage.py`); class identity by name across import, combine, export, train
(`tests/integration/test_class_identity_e2e.py`); `tests/curation/dataset_import/`; GPS by policy in
exports (`test_export_gps_by_policy.py`).

**Size.** About 35 files, 2,500 lines changed, 2,500 deleted. Effort: high (attach under locks,
import undo).

**Owes the wave.** Export time on the public set against today's; the scripted labelling walk's
export and training step (section 4).

### S8. Machine writers and re-runs

**Goal.** Every machine helper works from the record: the VLM labels and verifies, auto-label and
the assist loop propose, the probe and the item scores rank, the projection map draws, open
vocabulary finds new objects, re-detect and requeue re-run the stage; none can undo a person.

| Id | Today | Replaced by |
|---|---|---|
| T4 | `vlm_batch._write_updates` `:221`; `vlm_class_batch.write_updates` `:203`; `assist/verify.py:110`; `autolabel` | `class_opinions` (`vlm`) through `apply()` with the M17 guard (`results._opinion_guard`); verify fields in `item_region_results`; attempts in the ledger (kind `vlm`) |
| T13 | `item_scores/job.py:116`; `probe_predictions.py:290`; `embedding_viz.py:315` | `upsert_results` for `item_scores`, `probe_predictions`, `item_projections` (tally-neutral, no fold) |
| T14 | `item_delete._delete_round` `:145` | attach (`claim_match`) under the photo lock removes items with neither a detection nor an asserting claim (15a 4.1) |
| T15 | `reprocess_detect._apply_plan` `:105`; `region_requeue.py:465`, `:545`; `registry_reclassify.py:281`; `reprocess_vlm.apply_vlm` `:86`; `repair_scan.py:106`; `empty_vlm_answer_repair.py:251`; `region_text_repair.py:281` | re-detect and open vocabulary: new detections and attach; requeue: `item_region_results` reset to pending under a new analyser and the ledger; reclassify: the class family walk of S2; VLM reprocess: the ledger work list; the three repairs are deleted (they fixed document defects the record cannot have; a re-run replaces them) |
| T19 | `assist/slot_write.write_slots` `:90`, `write_box_attributes` `:127` | `assist_answers` rows |
| I7 | `open_vocab_run.stamp_open_vocab_status` `:348`; `open_vocab_ingest.py:73` | `analysis_ledger` kind `open_vocab` (state, reason) |

T21's `scripts/curation/cluster_raw_labels.py:212` writes `raw_label_clusters`. Propagation
(`assist_loop/propagate.py`) writes `assist_propagated` opinions with `op_id` (15a section 7 row).

**Readers.** 17, 18, 44, 48, 50, 51, 52, 53, 56, 57, 76, 77; call sites of 49 (`vlm_pending.py:211`)
and 79 (`open_vocab_sweep.py:29`).

**Deletes.** `class_write_guard.py` checks, `scripts/curation/vlm_pending.py` lock clauses,
`combined_resolve.py` guard reads; `item_delete.py`; `repair_scan.py`, `empty_vlm_answer_repair.py`,
`region_text_repair.py`, `stale_label_source_repair.py`, `unmatched_class_repair.py` if their only
target is a document defect (the lane lists each with its `rg` evidence); `assist_fields.py`,
`open_vocab_fields.py`; the names without a writer (15a 4.2: `cluster_majority_agreement`,
`classifier_vlm_agreement`).

**Wire.** `frame_mismatch` on reprocess results (15a section 10).

**Tests.** Unit, red first: VLM empty answer keeps a class (M17); open-vocabulary target kept against
the VLM (15a 4.2 rank 1). (DB): a VLM batch racing a label: the label decides, the opinion is
stored; re-detect with jittered boxes keeps claimed items (`test_rerun_restores_state`, 15a 4.5).
Live, green after wave E: `test_live_vlm.py`, `test_live_open_vocab.py`, `test_live_assist_loop.py`,
`test_live_served_facts.py`; browser `frontend/e2e/live/test_writes.py`, `test_route_sweep.py` (all
routes).

**Keep-list.** `tests/curation/test_combined_resolve.py`, `test_combined_attributes.py`,
`test_assist_wire.py`, `test_probe_*`, `test_open_vocab_item_wire.py`; the probe's actionable flag
(`wire.py:253`).

**Size.** About 45 files, 3,000 lines changed, 4,000 deleted. Effort: high (attach, worker
concurrency).

**Owes the wave.** Throughput of the VLM and region workers on set P; the auto-label run end to end.

### S9. Deletions, docs, re-ingest, acceptance

**Goal.** Nothing of the curation search path is left; the docs describe the record; the public sets
and set P are re-ingested and plan 15 section 11 passes.

**Writers.** I11 (owner: `services/upgrade/from_060/admin_client.py:110`, `steps/legacy_repair.py:77`,
`steps/near_duplicates.py:78`), T21 (owner: the remaining `seed_*` scripts, `seed_docs_embeddings.py`,
`seed_docs_run_results.py`, upgrade steps), G1 to G9 kept: they are the global indexes of inventory
section 1.2, unchanged until phase 3. Index lifecycle (inventory 2.3: `lifecycle._create_one`
`clients/curation_opensearch/lifecycle.py:29`, `projects/delete.py:185`, `ensure_*.py`,
`services/curation/index_bootstrap.py`) deleted.

**Readers.** 62, 69 to 75 kept (global).

**Deletes.** `clients/occ.py`, `occ_bulk.py`, `occ_upsert.py`; `clients/occ_locks.py` after the fold
differential test's oracle (`tests/domain/test_fold_differential.py`) is frozen into recorded cases
(`tests/domain/world_allowed.py` keeps the allow-list); `clients/curation_opensearch/` except what
section 1.2 code imports; `curation_opensearch/crops.py`; `bodies_core.py`, `items_extra.py`,
`capture_fields.py` mapping parts; `projects.indexes` use, `project_indexes`, `search_projection_tasks`
(`0018`); `OP_PROJECT_INDEX_PREFIX` and the other curation index variables into `retired_env.py`;
`src/services/upgrade/from_060/` (question 1). A ratchet test fails when any module outside the
section 1.2 code calls `items_index()`, `images_index()` or the curation OpenSearch factory.

**Docs.** `docs/ARCHITECTURE.md`, `docs/CURATION.md`, `docs/opensearch_schema_design.md` (reduced to
the global indexes), `docs/design/curation_api_contract.md`, `CLAUDE.md` (layout, services), the
docs site pages for storage (concise user page, detailed developer page).

**Tests.** The full offline suite, (DB) tier, live harness and browser live tier on the merged tree;
the acceptance walk (section 4).

**Size.** About 60 files, 300 lines changed, 8,000 deleted. Effort: medium (deletions), high for the
acceptance runs.

**Owes the wave.** The acceptance record with every number of section 4.

## 4. The acceptance walk (plan 15 section 11)

1. Re-ingest COCO (public) and set P on the record; the plan 13 four-way ingest table not below the
   numbers measured before the change; host CPU per photo per process recorded.
2. Derived stores: drop and rebuild `item_state`/`box_state` (`fold.rebuild`), the summary tables
   (`rebuild_tallies`), each vector index (`VectorIndex.rebuild`): same answers (R7).
3. Empty every rebuildable result table, re-run the analysers from the ledger: every claim attaches,
   and every state row equals its value before (revisions aside).
4. The scripted labelling walk on the public set: label by cluster, undo, bulk accept, exclude,
   freeze the holdout, export, train; trained-model AP equal to before the change.
5. Two API processes and two workers writing the same objects under load: no lost claim, no state row
   that differs from a fresh fold, summaries equal to a rebuild.
6. Gate A bars at 10 M for every page and dashboard read with the final designs (15b section 1.3);
   gate B bars for the vector index.
7. The wire contract differs only by 15a section 10 and section 2 here (`make contracts-check`
   against the pre-cut-over contract).
8. The live harness and the browser live tier fully green; the route sweep at desktop and 430 px,
   screenshots viewed.

## 5. Risks and their early signals

| Risk | Early signal |
|---|---|
| Summary upkeep slows label writes or queues labellers on a hot row | label route p99 in the S2 sweep; `pg_stat_activity.wait_event` `tuple` or `transactionid` on `sum_items` during the write-under-read run |
| Non-HOT `item_state` updates bloat tables and indexes | `n_dead_tup` and index sizes between sweeps; `n_tup_hot_upd / n_tup_upd` near zero as predicted (15a 2.6) |
| A writer path bypasses the summaries (drift) | the S1a rebuild-compare test; a nightly `rebuild_tallies` dry run that reports differences |
| Top-N sorts or capped counts on broad or rare filters are slow | the S1b sweep's `EXPLAIN (ANALYZE, BUFFERS)` rows removed by filter; 15b question 1 |
| Long export snapshots hold back vacuum | `age(backend_xmin)` in `pg_stat_activity` during the S7 export test |
| Ingest throughput drops (fold and summaries inside the commit group) | the plan 13 four-way table in the wave A sweep |
| The vector index at 10 M (27 GB heap, HNSW build memory) | gate B; `maintenance_work_mem` warnings during the S3 build test |
| A lane runs out of working size (S1a, S2, S8 are largest) | commits stop arriving at the planned concern boundaries; the coordinator splits by concern |
| The fold's differential oracle disappears with `occ_locks.py` | S9 freezes recorded cases before the deletion; the differential test count must not drop |
| Frontend breaks on the wire changes | `make contracts-check` and the stubbed browser tier in every slice |

## 6. Questions for the coordinator

1. **Delete the 0.6.0 upgrade path** (`src/services/upgrade/from_060/`, I11 and T21 parts) in S9?
   Evidence: it carries OpenSearch documents into the 0.7 indexes; the data is wiped and re-ingested
   (owner, 2026-11-14), so it has no target. Recommendation: delete, and say in the changelog that
   0.6 data is re-ingested.
2. **Core API data stays on the global OpenSearch indexes until phase 3**, so the reserved core
   project of plan 15 section 14 answer 1 is not created in these slices. Evidence: the coordinator's
   decision keeps section 1.2 until phase 3; G1 to G9 write only those indexes. Recommendation:
   confirm; phase 3 creates the core project with its tables.
3. **Ingest placement removed in S1a, re-added as membership rows in S6** (no code written twice; the
   ingest call site is touched twice). Alternative: move the `cluster_memberships` column change into
   S1a, which breaks the clustering writers a wave earlier. Recommendation: as planned.
4. **Replace the OCC live test** (`tests/live/test_live_bulk_occ.py`) by a two-client claim test in
   S4: optimistic concurrency no longer exists to prove. Recommendation: replace.

## 7. Trace: every writer in exactly one slice

Owner = the slice whose exit removes every part of the id; parts done earlier are named.

| Id | Owner | Parts done by another slice |
|---|---|---|
| I1 | S1a | |
| I2 | S1a | |
| I3 | S1a | |
| I4 | S1a | |
| I5 | S5 | the global `apply_run` path stays with G6 (S9) |
| I6 | S1a | |
| I7 | S8 | |
| I8 | S7 | |
| I9 | S7 | |
| I10 | S3 | |
| I11 | S9 | `backfill_item_ids.py` deleted in S1a |
| T1 | S1a | |
| T2 | S2 | |
| T3 | S4 | |
| T4 | S8 | |
| T5 | S2 | |
| T6 | S2 | |
| T7 | S4 | |
| T8 | S2 | |
| T9 | S2 | |
| T10 | S6 | |
| T11 | S5 | |
| T12 | S3 | the T3 and T7 callers of `write_box_embeddings` move in S4 |
| T13 | S8 | |
| T14 | S8 | the import-undo call site is replaced in S7 |
| T15 | S8 | |
| T16 | S2 | |
| T17 | S7 | |
| T18 | S7 | |
| T19 | S8 | |
| T20 | S1a | |
| T21 | S9 | `seed_live_harness.py` in S1a, `cluster_raw_labels.py` in S8 |
| G1 to G9 | S9 (kept, global, phase 3) | |

11 + 21 + 9 = 41 writer ids, each with one owner. Shapes: 15b section 12 (80, each in one slice);
S1a owns 6, S1b 15, S2 10, S3 10, S4 4, S5 3, S6 8, S7 4, S8 12, S9 8 (kept).
