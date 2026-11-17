# 15c: storage cut-over, the vertical slices

| Field | Value |
|---|---|
| In force again | the Python slices of this plan ARE written (owner, 2026-11-16: hybrid; Rust owns the photo's bytes end to end, Python owns the routes and every database write). The Rust record slices of [18b](18b_backend_rust_port_later_slices.md) are parked. The ingest slice S1a-ii commits what the Rust photo path hands over (the photo-path plan is in writing and not yet in this tree) |
| Status | plan, revision 1 (2026-11-15): the cut-over of plan 15 section 10 steps 2 to 8 as vertical slices on `int/storage`, written from [15_storage_plan.md](15_storage_plan.md), [15a_storage_schema.md](15a_storage_schema.md) and the read map [15b_storage_reads.md](15b_storage_reads.md). Revision 1 folds in the independent review of 15b and 15c and the coordinator's decisions on it (appendix "Review trace"): slices split, entry conditions, a module deletion table and a pending-test list by test id |
| Code base read | `int/storage` at `9ce22b5f6`. Writer ids (I1..I11, T1..T21, G1..G9) and shape numbers (1..80) are those of the storage inventory (`artifacts_local/speed_1113/storage_inventory.md`, gitignored), sections 2 and 3; paths in it are relative to `src/services/curation/` unless they name another directory |
| Built already (step 1) | `src/domain/claim*.py`, `fold*.py`, `results.py`, `claim_match.py`; migration `0010_storage_record`; `src/db/repos/` (`photos`, `items`, `claims`, `results`, `state`, `fold_load`, `analysers`, `upserts`); `src/services/record/apply.py`; `src/db/vector_ddl.py`; `src/services/erasure/catalogue.py`. Nothing calls them yet |
| Data | wiped and re-ingested (owner, 2026-11-14): no slice migrates a search-index document |

## 0. Rules every slice follows

1. **The user's job first.** A person adds photos, a detector finds objects, the person labels their
   own classes cluster by cluster, edits boxes, freezes a holdout and trains; later the photos are a
   gallery. Each slice says which part of that job works again when it lands.
2. **Vertical.** A slice moves a feature family's writers and its readers together and deletes the
   old path in the same slice: no dual write, no compatibility shim, no dead code. Families not yet
   moved do not work on `int/storage` between slices (accepted). A module is deleted by the slice
   that removes its **last importer** (section 8 lists each with its importers); a slice that would
   break another slice's import cuts that import in the same commit. Before its last commit every
   lane runs `.venv/bin/python -c "import src.main"` and `.venv/bin/python -m pytest --collect-only
   -q tests/` (offline, no database), so the import graph is whole at every merge.
3. **Filters of families not moved.** The item filter translator is one package with a module per
   family (`src/db/queries/item_filter/`, 15b P1); an unregistered family raises
   `FilterNotMovedError` (400 with a served sentence naming the family). A slice registers its
   family's module; nothing else is temporary.
4. **Lanes** (global rules, multi-agent lanes): one agent per slice in its own worktree and branch,
   one commit per concern, hooks on, never `--no-verify`, never push or merge. Lanes run unit and
   contract tests, lint, type checks and hooks on touched files, and the check of rule 2. Tests
   marked (DB), live and browser tests are written by the lane and run by the coordinator's wave
   sweep. A unit test is seen failing before its fix, and the commit message says how.
5. **Docs and contracts in the same commit.** A slice that deletes or adds a route, wire field or
   `OP_*` variable updates every doc that names it (`scripts/docs/check_docs_vs_code.py` runs in the
   hooks) and runs `make contracts`, committing `contracts/` with the change. A retired variable goes
   into `src/config/retired_env.py`. A vulture whitelist entry for a step-1 name
   (`scripts/codegen/vulture_whitelist.py`) is removed by the slice that wires the name.
6. **Package docs.** Every lane brief cites the pages read for what it touches: PostgreSQL 18 pages
   named in 15a and 15b, psycopg 3 (server-side cursors, `copy`), SQLAlchemy 2 asyncio, pgvector
   0.8.7, Alembic.
7. **Entry conditions.** A slice starts only when the measurements it names exist (15b section 13,
   items M1 to M10, or gate B) and meet their bars, or the coordinator has decided on a miss. No lane
   builds on an unmeasured number; queries outside section 13 are timed in the first sweep after
   their slice (15b section 1.3), and a miss is code work for that lane in the next wave.
8. **Pending tests by id.** Section 9 is the one list of tests that are red until a named slice. A
   listed test runs as a strict expected failure (passing early is an error), a test not listed must
   pass, and each slice's merge removes its ids; the list is empty at S9.

## 1. Order, revisions and waves

| Slice | Plain name | Revision it writes | Depends on | Entry (15b section 13) |
|---|---|---|---|---|
| S1a-i | Summaries, indexes and the state writer | `0011_storage_indexes` | step 1 | M1, M2, M3, M4, M5 |
| S1a-ii | Ingest writes the record (and attach) | `0012_ingest_cutover` (fences `image_contents` to the core API's rows) | S1a-i | M1 (ingest variant) |
| S1b-i | Object lists: cursor, pages, locate, counts | none | S1a-i's DDL | M6, M7, M8 |
| S1b-ii | Photos, photo page, crops, map | none | S1a-i's DDL | M6 |
| S2a | Labelling acts and undo | `0013_claims_cutover` (drops `label_operation_members`) | S1a-i, S1b-i | M1 |
| F | Files through the storage layer | none | S1b-ii (shares `image_serving.py`) | none |
| S2b | Review queue, class counts, statistics | none | S2a | M6, M7 |
| S3 | Vectors: index interface, search, vector jobs | `0014_vector_cutover` (15a 6.3) | S1a-ii, S1b-i | gate B |
| S4 | Region stage and box editing | none | S2a, S3 | M9 |
| S5 | Near-duplicates and the holdout | `0015_near_dup_cutover` (bigint member keys; drops `near_dup_decisions`) | S2a, S3 | gate B |
| S6 | Clustering and the cluster screens | `0016_cluster_cutover` (`subject_key`, X22, X23, cluster summaries) | S3, S4, S5 | M10 |
| S7 | Import, combine, export, training inputs | `0017_import_cutover` (`dataset_import_images.photo_key`) | S1a-ii, S2a, S4, S5, F | none |
| S8a | VLM, auto-label, assist loop | none | S2b to S6 | none |
| S8b | Score and projection writers | none | S2b to S6 | none |
| S8c | Re-runs: re-detect, open vocabulary, requeue, repairs | none | S2b to S7 | none |
| S9 | Deletions, docs, re-ingest, acceptance | `0018_reads_cutover` (drops `project_indexes`, `search_projection_tasks`) | all | all recorded |

Changes to revision 0's order, with reasons:
- **Splits at the review's points** (M11): S1a into the summary machinery (S1a-i, merged first:
  every later slice writes through it) and the ingest cut-over (S1a-ii); S1b into object lists and
  photos; S2 into acts and undo (S2a) and the queue, counts and statistics (S2b); S8 into the
  machine helpers (S8a), the score writers (S8b) and the re-runs (S8c). Open vocabulary's status
  (I7) goes with the re-runs, not with the VLM: its stamp and its attach are one flow (T15).
- **Attach in S1a-ii** (decision on M8): ingest creates objects through it; S7 and S8c consume it.
- **Vectors (S3) before clustering (S6) and regions (S4)**, as in revision 0: clustering reads
  every vector of its pool and the region worker writes box vectors.
- **Revision ids.** 15a section 9's ids are superseded by this table (15a 2.6). The `down_revision`
  chain is fixed above, so the coordinator merges S1a-ii before S2a in wave B and S6 before S7 in
  wave E; every other wave writes at most one revision.

```
step 1 ── S1a-i ─┬─ S1a-ii ─┬─ S3 ────────┬─ S4 ─┬─ S6 ─┬─ S8a, S8b, S8c ── S9
                 ├─ S1b-i ──┼─ S2a ── S2b ┼─ S5 ─┘      │
                 └─ S1b-ii ─┴─ F ─────────┴──────── S7 ─┘
```

| Wave | Lanes (at most three) | Revisions, merge order | Sweep covers |
|---|---|---|---|
| A | S1a-i, S1b-i, S1b-ii | `0011` (S1a-i first) | summaries, indexes, lists, photos (DB tests on seeded rows) |
| B | S1a-ii, S2a, F | `0012` then `0013` | ingest, attach, labelling, files; the live tiers of wave A turn green |
| C | S2b, S3 | `0014` | review, counts, statistics, vectors |
| D | S4, S5 | `0015` | regions, near-duplicates, holdout |
| E | S6, S7 | `0016` then `0017` | clustering, import, combine, export, training |
| F | S8a, S8b, S8c | none | machine writers and re-runs |
| G | S9 | `0018` | deletions, re-ingest, acceptance of plan 15 section 11 |

Wave A's three lanes code against the `0011` DDL fixed in 15a 2.6 and 15b 6.1; S1b-i and S1b-ii
call no S1a-i function except the `_now` views. Measurements of 15b section 13 run before wave A
(M1 to M8), before wave D (M9) and before wave E (M10), while earlier waves code.

## 2. Contract diff addendum

15a section 10 lists the wire changes of the schema; these come from the read design (15b) and from
the writers. Each slice applies its rows; the acceptance checks the whole table.

| Wire field or route (model) | Change | Reason | Slice |
|---|---|---|---|
| `total_capped: bool`, `total_unknown: bool` (`CropsPageResponse`, `TablePage`, `ImagePage`, `ReviewQueuePage`, `RegionRowPage`, `RegionFpPage`, `CropLocate`, `ImageLocate`, `ReviewLocateResponse`) | added; `total` is at most the cap when capped, and unset meaning "many" when unknown | 15b section 5 rules 2 and 3 | S1b-i, S1b-ii (photos), S2b (review), S4 (regions) |
| `index_capped: bool` (the three locate models) | added | 15b section 5 rule 4 | S1b-i, S2b |
| `capped: bool` (`HiddenCounts`, `frontend/src/lib/types.ts:295`; field coverage `n_scored`) | added | 15b section 5 rules 5 and the cap table | S1b-i, S2b |
| `join_truncated` (`ImagePage`, `ReviewQueuePage`) | deleted | the photo join is a semi-join with no cap (15b P6) | S1b-ii, S2b |
| `rows_truncated` (`RegionRowBase`) and the three-box window | deleted; every matching box of a page item is returned | no inner-hits window (15b section 9) | S4 |
| `POST /image_counts/backfill` (`routers/curation/image_counts.py:24`), job kind `image_counts_backfill` (`config/job_kinds/image_counts.py:11`) | deleted | card counts follow every write (15b section 6) | S1a-ii |
| image list `sort_options` and the "not counted" refusal (`image_view.py:254-262`) | the `unlabeled_count` sorts are always offered; the refusal goes | the same | S1b-ii |
| `crops_updated`, `crops_preserved_human`, `crops_final_conflicts` (ingest results, `ingest_index.py:607-610`) | deleted | a photo is stored whole or reported duplicate | S1a-ii |
| ingest `error_kind` `incomplete_write` and the `held_back` answer (`ingest_index.py:565-570`) | deleted | no half-stored photo exists | S1a-ii |
| order of equal sort values | under a `desc` sort ties come in descending `crop_id` (or `photo_key`, `item_key`) order | one direction per row comparison (15b section 4) | S1b-i, S1b-ii, S2b, S4 |
| `updated_at` (items) | moves only with facts a person sees on the item; machine-only writes leave it (owner confirmation, 15b section 11 question 8) | 15a 2.4, review m6 | S1a-i |
| `GET /regions` item order | newest item first (was newest detected box) | 15b section 9 | S4 |
| text filters | a fragment under three characters answers 400 `text_too_short` | 15b section 8 | S4 |
| cluster card `review.all` | kept as P13 over the visible cards; dropped only if 15b section 11 question 6 is answered so | 15b 6.2 | S6 |

Frontend reads of totals, each with its change, are the interface table of 15b section 5; each row
goes with the slice of its screen: lists, table, detections, editor, bulk bar (S1b-i), gallery
(S1b-ii), review and scores (S2b), slot galleries (S4), cluster pager (S6); job counters change
nowhere.

## 3. The slices

Each slice: goal; entry conditions; writers (inventory id, today's function with file:line, what
replaces it); readers (shapes, forms in 15b section 3); deletions (section 8 holds the importers);
wire; tests; keep-list; size and effort; what it owes the wave. "Human-act service" is
`src/services/record/human.py` (new in S2a): open the operation row
(`records.create_label_operation`), build drafts with `src/domain/claim_builders*.py`, call
`apply()`, insert events, in chunks of 1,000 for bulk acts (15a section 7, bulk row).

### S1a-i. Summaries, indexes and the state writer

**Goal.** Every count the person sees (class counts, card numbers, dataset statistics, totals)
follows every write exactly, and 30 people labelling one class never wait on a counter.

**Entry.** 15b section 13 M1 to M5 recorded and meeting their bars; M3 gives the compactor batch,
M5 the fillfactors.

**Builds.** `0011_storage_indexes` (15a 2.6: indexes X1 to X21 and X24 to X32, the `items` and
`item_state` columns, the summary tables, `sum_deltas`, the `_now` views, statistics, fillfactors);
`state.store_states` (15b 6.2) replacing `upsert_item_states`, `upsert_box_states`,
`delete_states`, and `apply()` calling it; `photo_capture` out of `results.RESULT_SPECS`;
`photos.insert_photos` passing `photos_added`; `src/domain/tallies.py`; `src/db/repos/sum_compact.py`
(compactor, `rebuild_tallies`, the 6.7 wait helper), job kind `sum_compact`, its beat entry and the
nightly drift check; the fold outputs `detector_class_name`, `detector_confidence`,
`vlm_confidence`, `vlm_class_empty` and the `updated_at` rule (15a 2.4) in `src/domain/fold*.py`;
the declarative tab table `src/domain/review_tabs.py` with its `counted` flag and Python evaluator;
the pending-test mechanism of section 9 (`tests/pending_slices.txt` and the conftest hooks of
`tests/` and `frontend/e2e/`) with its first content.

**Writers and readers.** No inventory writer id and no shape: it is the machinery every later
writer and summary reader uses.

**Tests.** Unit, red first: `tests/storage/test_tallies_offline.py`, `test_apply_offline.py`
extended (every result table classified), `tests/storage/test_counted_writers.py` (ratchet, 15b
6.2), `tests/domain/test_review_tabs_counted.py` (a tab reading a result table is not counted),
fold tests for the new outputs and the `updated_at` rule. (DB): `tests/db/test_tallies_property.py`,
`tests/db/test_sum_compact.py`, `tests/db/test_storage_schema_matches_plan.py` extended (every
index, view and `security_invoker` flag).

**Keep-list.** `tests/domain/test_fold_differential.py` count does not drop; `apply()`'s outcome
fields; the lock order of 15a section 7.

**Size.** About 20 files, 1,600 lines. Effort: high (concurrency, migration). **Owes the wave.** The
(DB) tests above; the property test run 1,000 rounds.

### S1a-ii. Ingest writes the record

**Goal.** A person adds photos (upload, batch, objects, sources, imports' image stage): each photo,
with its objects, detector classes, capture data, vectors and "created" events, is stored in one
transaction per commit group, or not at all; the same bytes again answer `duplicate`; a new project
gets its vector tables; objects are created through attach.

**Entry.** S1a-i merged; M1's ingest variant met its bar.

| Id | Today | Replaced by |
|---|---|---|
| I1 | `IngestWriter._create_images` `ingest_writer.py:477` (bulk create `:491`), content claim `near_duplicates/ingest_claim.py` `claim_and_mark_in_session` | in `ingest_index.commit_group` (`:627`, rewritten): `photos.lock_imohashes` (the lock functions move from `repos/image_contents.py:71-89` to `photos.py`, and `image_contents` imports them back for the core path), `photos.photo_keys_by_image_id`, `photos.insert_photos`; SHA-256 confirm by `photos.set_content_sha256`; `near_dup_pending` marks kept |
| T1 | `IngestWriter._write_items` `:572` (`occ_upsert_bulk` `:585`), `_create_missing` `:612` | same transaction: `analysers.get_or_create_analyser` (detector, classifier profile, `capture_parser`, one encoder per space, region stage); `attach.attach(...)` creates the items and calls `apply(..., results=[detections, class_opinions, item_region_results seed (item_doc.py:37), photo_capture], photo_scope=True)`; `vectors.insert_vectors` (new `src/db/repos/vectors.py`, one multi-row `INSERT` per group, `vector_ddl.qualified(lock_key, space_key)`) and `vector_gaps`; `analysers.record_ledger`; `events.insert_events_async` (`crop.created`, today `services/events/publish.py:184`) |
| I2 | `image_completeness.begin_repair` `:186`, `_release` `:217`, `forget_items` `:95` | the ingest calls go: a photo is stored whole |
| I3 | `image_counts_refresh.recount_images` `:80` | the ingest call goes: `photo_counts` follows `store_states` |
| I4 | `image_counts_dirty._clear_counts_at` `clients/image_counts_dirty.py:136` | deleted with the Valkey dirty set (section 8) |
| I6 | `source_backfill/capture.py:79` `_write_images`; `location_purge.py:93` `purge_pass` | `state.store_states(capture=...)` (the `located` move, 15b 6.2), `analysers.record_ledger` (kind `capture_parser`) |
| T20 | `source_backfill/capture._copy_to_items` `:91`; `location_purge.py:93` (items) | deleted: capture time is joined (15a 5.2) |

**Attach** (`src/services/record/attach.py`, new; decision on M8): `attach(session, project,
photo_keys, detections, context)` under the photo lock (rank 5): `claim_match` groups each live
claim group onto the photo's detections (15a 3.4), keeps or inserts `items` rows, folds through
`apply(photo_scope=True)`, and deletes the identity rows of items `apply()` reports `vanished`. One
module, consumed by S7 (import reconcile, import undo) and S8c (re-detect, open vocabulary).

Also here: T21's `scripts/curation/seed_live_harness.py` (`_bulk` `:384`, `:522`, `:527`) seeds
through the ingest repositories, attach and `apply()` with `claim_builders` drafts; I11's
`scripts/curation/backfill_item_ids.py:113` is deleted. Project creation (`projects/lifecycle.py:117`)
creates the default spaces' vector tables through `src/services/embedding_spaces/ddl.py` (new, the
`op_vector` autocommit pool of 15a 6.1). Ingest placement in clusters (`ingest_index.place_prepared`
`:188`, `item_doc.placement_fields` `:101`) goes with the documents it stamped; S6 places as
membership rows. `image_contents` stays for the core API (B1): `0012_ingest_cutover` deletes rows
with a `project_id` and adds `CHECK (project_id IS NULL)`.

**Readers.** 31, 34, 54, 55, 78, 80.

**Deletes.** `ingest_writer.py`, `repair_match.py`, `image_counts_backfill.py`, `tasks/image_counts.py`,
the beat entry `op.periodic.image_counts_drain` (`tasks/periodic.py:46`) and
`periodic/image_counts_drain.py`, `config/job_kinds/image_counts.py`, `routers/curation/image_counts.py`,
`clients/image_counts_dirty.py` (its marks in `occ.py`, `occ_bulk.py`, `occ_upsert.py` cut);
`item_doc.build_image_doc` (`:113`), `build_item_doc` (`:166`), `placement_fields`; in
`ingest_index.py` `build_documents`, `write_prepared`, `write_group`, `place_prepared`,
`doc_context`; `content_key.claim_bound`, `resolve_many_bound`, `release_bound`;
`source_backfill/content.py`; `scripts/curation/backfill_item_ids.py`; `scripts/bench/ingest_write_path.py`
and `ingest_consistency.py` (they measure the OpenSearch write path). Left (section 8):
`image_completeness.py`, `image_counts_refresh.py`, `count_fields.py`, `IMAGES_INGEST_WRITE_MAPPING`
(S9), `repos/image_contents.py` (phase 3). Tests of deleted modules are replaced by those below.

**Wire.** Addendum rows for S1a-ii; `image_path` in ingest results becomes the display path of
`photos.storage_key` (15a section 10).

**Tests.** Unit, red first: `tests/curation/test_ingest_commit_record.py` (drafts and result rows
for a photo with two objects, a duplicate, a refused claim, an object without a vector),
`tests/curation/test_attach_offline.py` (groups to detections, a vanished item). (DB):
`tests/db/test_ingest_record.py` (a group is stored whole; a crash between statements leaves
nothing; the same bytes twice give one photo and `duplicate`; two concurrent groups with the same
imohash give one photo), `tests/db/test_attach.py` (claims attach to re-detected boxes; two attaches
of one photo serialise), `tests/db/test_vector_space_ddl.py` (`op_app` cannot create in `vec`; both
row-security flags). Live: section 9 ids of S1a-ii; `test_live_core_ingest.py` must stay green (the
core path keeps `image_contents`).

**Keep-list.** Duplicate bytes answer `duplicate` (`tests/curation/test_ingest_whole_photo.py`,
rewritten on the record); a photo is stored whole (`test_live_whole_photo.py`); the per-photo flow of
plan 13a with its 503 `database_busy` (`tests/curation/test_ingest_finish_batch.py`); `crop.created`
once per item (`tests/curation/test_ingest_batched_inference_and_events.py`); the location policy
drops positions (`tests/curation/test_capture_backfill.py`); the embedding policy's skip states
(`tests/curation/test_item_wire_embedding_states.py`).

**Size.** About 30 files, 2,200 lines changed, 2,000 deleted. Effort: high (transaction and lock
order, duplicate claims, attach). **Owes the wave.** The (DB) tests; the plan 13 four-way ingest
table on the public sets not below its measured numbers, host CPU per photo; `0012` on the harness.

### S1b-i. Object lists: cursor, pages, locate, counts

**Goal.** The person sees their objects: the object list with every sort and filter of the moved
families, the table view, "where is this item in my list". Totals are exact for summary keys and
"10,000+" (or "many") otherwise.

**Entry.** M6, M7, M8 recorded: `:count_budget_ms`, `T_sort`, the multi-class branch form.

**Readers.** 1, 3, 4, 5, 7, 9, 32, 33 through the `core` module of `src/db/queries/item_filter/`,
`src/db/repos/item_reads.py` (pages, locate, counts, the two-path sort rule), `src/db/repos/item_records.py`
(`load_item_records`, P2), `src/db/queries/totals.py` (`exact_total`, `count_items`) and
`src/domain/list_totals.py`. `list_cursor.py` keeps `encode_cursor`, `decode_cursor`, `cursor_carry`
and replaces `page_body`, `with_tiebreak`, `split_page` with `keyset_clause` (every sort of 15b
section 7, including the expanded mixed form) and `split_rows`. `serialize_item` (`wire.py:288`) and
`serialize_items` (`:486`) take the typed record.

**Deletes.** The OpenSearch bodies of `crop_listing.py` (`crop_list_body` `:85`, `crops_by_ids`
`:120`), `list_locate.py`, `hidden_counts.py`, `table_view.py`, `list_counts.page_counts`;
`wire.item_source_excludes` (`:92`), `item_list_source_excludes` (`:99`),
`item_context_source_excludes` (`:114`), `source_image_path` (`wire.py:322-323`). Left: `mget_crops`
(S9), `region_to_wire` storage mapping (S4).

**Wire.** Addendum rows for S1b-i; 15a section 10: `image_path`, `source_image_path`,
`class_unresolved`. Frontend: `types.ts`, the list rows of 15b section 5's interface table,
generated types.

**Tests.** Unit, red first: `tests/curation/test_list_cursor.py` (rewritten: ISO microsecond
timestamps; tie direction follows the sort; the missing-last segment resumes; a foreign cursor is
refused; every sort of 15b section 7 has a keyset form), `tests/storage/test_item_filter_sql.py`
(every `ItemFilter` field has a form or a `FilterNotMovedError` entry), `tests/storage/test_totals_rule.py`.
(DB): `tests/db/test_item_reads.py` (deep keyset; locate with cursors; a capped count on 10,001 rows;
a count over budget answers `total_unknown`; NULL sort values last in both directions; `EXPLAIN`
shows every `UNION ALL` branch on X2 and the review predicate using X4),
`tests/db/test_totals_differential.py`. Live and browser: section 9 ids of S1b-i.

**Keep-list.** Cursor refused across sorts (`tests/curation/test_crops_cursor.py`); locate agrees
with the page that follows (`tests/curation/test_crops_locate.py`, `test_live_paging.py`); hidden
counts exclude holdout and excluded (`tests/curation/test_hidden_counts.py`); class identity by name.

**Size.** About 22 files, 2,000 lines changed, 1,000 deleted, plus about 6 frontend files. Effort:
high (cursor correctness, wire serializer). **Owes the wave.** Timing of rows 1, 3 to 5, 7 at 10 M
and on set P (15b 1.3).

### S1b-ii. Photos, photo page, crops, map

**Goal.** The person sees their photos: the gallery with card counts, a photo's page, crops and
thumbnails, the map under the location policy.

**Entry.** M6 recorded (the photo list's capped counts use the same budget).

**Readers.** 24, 25, 26, 27, 28, 29, 30 through `src/db/repos/photo_reads.py` (P6, P15) and the
`photos` module of the filter package (photo predicates, `image_ids`, `geo_bbox`, `geo_polygon`).
Crop and thumbnail serving (`image_serving.py`, routes `routers/curation/images.py:103`, `:134`,
`:197`, `:242`, `:308`) reads `photos.storage_key` and `item_state.bbox`; the crop cache keyed by
`crop_id` stays until F.

**Deletes.** `image_join.py`; the OpenSearch parts of `image_view.py`, `image_counts.py`
(`item_facts`), `image_detail.py`, `map_cells.py` (its policy functions stay), `capture_wire.py:64`.

**Wire.** Addendum rows for S1b-ii. Frontend: the gallery and map rows of 15b section 5.

**Tests.** Unit, red first: P6 and P15 statement builders (the item filter inside the limited
subquery; the policy clamps). (DB): `tests/db/test_photo_reads.py` (a filtered page is never short;
a photo with no item reads `no_items`; card counts equal the items after compaction and before it;
`taken_at` orders by `taken_local`). Live and browser: section 9 ids of S1b-ii; stubbed
`frontend/e2e/stubbed/test_gallery_view.py`, `test_item_filter.py`, `test_map_view.py`.

**Keep-list.** `tests/curation/test_images_locate.py`; card counts equal the items
(`tests/curation/test_image_counts_view.py`); the location policy on the map and the photo page
(`test_live_capture_map.py`); the thumbnail `?v=` (`tests/curation/test_geometry_wire.py`).

**Size.** About 14 files, 1,200 lines changed, 900 deleted, 3 frontend files. Effort: medium.
**Owes the wave.** Rows 24, 25, 28 to 30 at 10 M; the map page in a browser at desktop and 430 px.

### S2a. Labelling acts and undo

**Goal.** The person labels: one item, a batch, a cluster move, accept machine labels, adopt the
detector's class, unlabel, discard, exclude and restore, dismiss, flag "needs a new class", resolve
proposals, draw an object, move its box, start an audit, merge classes, undo any of it.

**Entry.** M1: the label transaction's p99 with deltas at 30 writers met its bar.

| Id | Today | Replaced by |
|---|---|---|
| T2 | `create_drawn_item` `image_item_write.py:125` (`index` `:198`) | human-act service: `claim_builders_region.draw_object`, `items.insert_items` under the photo lock (`photos.lock_photos`, rank 5), `apply(photo_scope=True)` |
| T5 | `recorded_relabel` `label_ops/record.py:105` → `crop_bulk.occ_bulk_human_relabel` `:54`; `run_relabel` `record.py:178`; `accept_merger` `label_ops/accept_merge.py:105`; adopt detector (`class_label.py:110-122`) | `claim_builders.label`, `move_to_cluster`, `accept_machine_eligible` + `accept_machine`; adoption writes `class_opinions` rows (`adopted_detector`, `op_id`) through `upsert_results` inside `apply()`; chunks of 1,000 with `append_claims`' `ON CONFLICT DO NOTHING` (M4) |
| T6 | `label_ops/undo._restore` `:116`; `label_undo._occ_write` `:74`, `discard_many` `:171`, `review_dismiss` `:248`; `class_undo.py:90`; `edit_undo.py:82`; `routers/curation/edit_undo.py:77` | `claim_builders.unlabel`, `discard`, `review_dismiss`, `review_undismiss`, `vlm_dismiss`, `probe_dismiss`; undo walks `claims.live_claims_of_operation`, writes `withdraw` drafts and folds per chunk in one transaction (15a 3.5); step-back undo withdraws the item's newest live operation; adoption undo deletes the operation's `class_opinions` rows (X17) |
| T8 | `exclusion._occ_bulk_human_write` `:106`; `item_geometry.edit_geometry` `:269`, `undo_geometry` `:302`, `rerank_image_items` `:219` | `claim_builders.exclude`, `restore`, `claim_builders_region.set_geometry`; ranks are `apply()`'s |
| T9 | `crops_write.flag_new_class` `:245`; `new_class_resolve.py:95`; `audit.start_audit` `:140` | `claim_builders.flag_new_class`, `resolve_new_class`, `audit_sample` |
| T16 | `classes/projection_sync.relabel_merged_items` `:191` and its `update_by_query` | nothing at merge time (readers use the class family, M16); the merge job runs `apply()` over keyset chunks of X2 with no new input (15a section 7), then drops the source id from the family |

**Readers.** 66 (the merge walk).

**Deletes.** `label_ops/crop_bulk.py`; the document merges of `class_label.py`
(`class_label_update` `:208`, `candidate_move_update` `:239`); `class_undo.py`; `edit_undo.py`
(with its task in `tasks/b4_fanout.py:11`, `:29`); the OpenSearch writes of `label_undo.py`; the
OpenSearch parts of `exclusion.py`, `item_geometry.py`, `new_class_resolve.py`, `audit.py`;
`classes/projection_sync.py` and the class rows of `search_projection_tasks`
(`repos/classes._queue_projection`); `records.add_label_operation_members`, `_recount_members`,
`label_operation_members` and the table (`0013`). Left (section 8): `domain/history.py` (S9),
`edit_history.py` (S7), `clients/opensearch/scroll.py` (phase 3), `occ*.py` (S9).

**Wire.** 15a section 10: `conflicts`, `conflict_ids` deleted (`schemas/curation/label_ops.py:77-78`);
`kept_changed` semantics.

**Tests.** Unit, red first: `tests/curation/test_human_act.py` (drafts per route, chunking, a
replayed chunk appends nothing), `tests/curation/test_label_op_undo.py` rewritten on claims
(`restored`, `kept_changed`, `kept_holdout`, `already_undone`). (DB): `tests/db/test_label_writes.py`
(two sessions label one item: no lost claim, state equals a fresh fold; undo killed after a chunk
resumes; the merge job keeps every `_now` view equal to the `GROUP BY`). Live and browser: section 9
ids of S2a; stubbed `test_labeling_flow.py`, `test_label_ops.py`, `test_cluster_bulk_undo.py`.

**Keep-list.** Step-back undo over label, batch label, move, resolve, discard and not over a bulk
accept (`tests/curation/test_label_op_undo.py`, 15a Q4); discard undo restores class and review state
together (M12); exclusion never leaves an item validated (`test_exclude_reasons.py`); a person may
relabel a frozen holdout item (15a question 1); class identity by name across merge
(`tests/integration/test_class_identity_e2e.py`).

**Size.** About 26 files, 2,000 lines changed, 2,500 deleted. Effort: high (locks, undo chunks,
migration). **Owes the wave.** (DB) tests; the label route's p99 on set P against M1.

### F. Files through the storage layer

**Goal.** Originals, uploads, crops, exports and training files are reached through `RootedStore`
by key (plan 15 section 8, R9), so a second store implementation is possible. Today's callers that
open paths directly: inventory section 7. **Entry.** None (no new query shape).

**Moves.** Originals by `photos.storage_key` (`image_serving.resolve_safe_path` `:196`,
`_render_source_image` `:455`, `resolve_crop_root` `:553`); uploads (`persist_uploaded_bytes` `:152`,
`_upload_dest` `:187`, `upload_ingest.py:89`); the crop and thumbnail cache keyed by photo, box and
size (`source_image_cache.crop_cache_file` `:69`, `write_crop_cache` `:82`, `crop_bytes.crop_jpeg_from_cache`
`:45`, `load_vlm_item_jpeg` `:159`, `ThumbnailCache` `image_serving.py:281`), with a size cap and no
`/dev/shm` assumption; exports (`export_support.allocate_export_dir` `:435`); training
(`services/training/jobs.py:130` passes a store key); dataset archives (`dataset_import/upload_job.py:46`,
`:51`, `scan.py`, `image_load.py`, `capped_io.py`); the false-positive store
(`detection/fp_store.py:34-47`). The global FAISS files (`visual_search/clustering_persist.py:36-67`)
stay with section 1.2 until phase 3.

**Shapes and writers.** None. **Deletes.** `BlobStore.local_path` use outside the local
implementation (`clients/storage/base.py:42`); the `domain/storage_keys.crop_key` whitelist entry.
**Tests.** Unit: `tests/storage/test_rooted_store_callers.py` (an `rg` ratchet: no `Image.open` or
`open()` on a stored path outside `clients/storage/`), crop cache key tests. Live: the file and
thumbnail routes of `test_live_images.py`. **Keep-list.** `tests/curation/test_crop_cache_project_scope.py`,
`test_crop_cache_prune.py`, `test_export_dir_allocation.py`, `test_export_retention.py`. **Size.**
About 20 files, 1,200 lines. Effort: medium. **Owes the wave.** A crop served after a cache purge.

### S2b. Review queue, class counts, statistics

**Goal.** The review queue, class counts, dataset statistics, selection summary, field coverage and
label-operation counts read the record, exact where a summary key answers.

**Entry.** M6 (the drained queue through X2 and X4, the count budget) and M7 (`T_sort`).

**Readers.** 2, 15, 16, 19, 36, 38, 43, 46, 47. The `review` module of the filter package builds
each tab's SQL from `src/domain/review_tabs.py` (S1a-i), with the review partial predicate written
as a literal so X4 is usable ([partial indexes](https://www.postgresql.org/docs/18/indexes-partial.html):
a parameterised clause never implies the predicate); tabs of families not moved raise
`FilterNotMovedError` (cluster fit: S6; region text: S4; probe: S8b). `count_items(filter, cap=)`
callers in labelling get their caps (15b section 5 table); the selection summary guard counts to
20,001.

**Deletes.** The OpenSearch bodies of the review routes (`review_tab_clauses.py` clause builders,
`review_sorts.py` clauses), `class_counts.py`, `stats_dataset_query.py`, `stats_classes.py`,
`review_resolve.py`, `selection_summary.py`, `field_coverage.py`, `label_ops/views.py`, the census
of `accept_machine.py` and `detector_adoption.py`.

**Wire.** `total_capped`, `total_unknown`, `index_capped`, `join_truncated` on the review models;
`capped` on field coverage. Frontend: the review and scores rows of 15b section 5.

**Tests.** Unit, red first: `tests/domain/test_review_tabs_sql_python.py` (property: SQL and Python
readings select the same rows), the guard caps. (DB): `tests/db/test_review_queue.py` (each default
tab sort; a nearly drained queue reads X4; counted tab totals equal `count(*)`). Live and browser:
section 9 ids of S2b.

**Keep-list.** The review sorts' clauses and coverage fallbacks (`test_review_sorts.py`,
`test_review_sort_coverage.py`); label-op current counts (`test_label_ops.py`); the selection limit
refusal (`tests/curation/test_selection_summary.py`).

**Size.** About 18 files, 1,300 lines changed, 1,200 deleted, 5 frontend files. Effort: medium.
**Owes the wave.** Rows 2 (each default tab), 15, 16 at 10 M against their bars.

### S3. Vectors: index interface, search, vector jobs

**Goal.** Semantic search over the person's objects works from the record: by text, by a similar
item, by an uploaded image, over item or box vectors; a new embedding space is filled and becomes
ready; the jobs that need every vector read them in one stream.

**Entry.** Gate B (the vector index at 10 M, filtered search) recorded.

| Id | Today | Replaced by |
|---|---|---|
| I10 | `embedding_spaces/reindex_opensearch.py:236` `write`, `:285` `mark`, `:363` `clear` (frame vectors) | `vectors.replace_vectors` (staging `COPY` then `INSERT ... SELECT`, 15a section 1) and `vector_gaps` writes, in `src/services/embedding_spaces/reindex_vectors.py` (new) |
| T12 | `reprocess_embed._write_item_vectors` `:250`, `_bulk_update` `:231`; `reindex_opensearch.py` `:236`, `:285`, `:363`; `region_box_embeddings.write_box_embeddings` `:184`; `scripts/curation/backfill_region_embeddings.py` | `vectors.replace_vectors` for items and frames; `write_box_embeddings` writes the box space's table (its T3 and T7 callers move in S4); the script deleted |

`0014_vector_cutover` applies 15a 6.3; `KnnMethodSpec`, `physical_names`, `legacy_names`
(`domain/embedding_spaces.py:70-135`) go with their last callers (section 8: `from_060`). The index
interface `src/services/vector_index/` (new): add or replace, delete, search with a filter, rebuild
(plan 15 section 7); default pgvector HNSW on `halfvec` built `CONCURRENTLY` after the fill,
`indisvalid` checked before `ready` (15a 6.3); the search paths of 15a 6.4 with gate B's thresholds.
`src/db/repos/vector_stream.py` (new) is P11.

**Readers.** 14, 20, 21, 22, 23, 35, 37, 59, 64, 65.

**Deletes.** `reindex_opensearch.py`; `semantic_search_run._search_body`, `build_knn_query`,
`_count_body`; `semantic_search_box.build_box_knn_query`; `reindex_encoder.py:76` mget;
`domain/embedding_state.py` `embedded_clause`, `not_embedded_clause`; `embedding_spaces/readers.py`
field names. Left: `pit_pages.py`, `vector_read.py` (S6), `near_duplicates/vectors.py` reads (S5).

**Wire.** None (`unembedded_in_scope` described as "10,000+" when capped).

**Tests.** Unit, red first: `tests/storage/test_vector_index_paths.py`, `tests/embedding_spaces/test_reindex_vectors.py`.
(DB): `tests/db/test_vector_search.py` (exact path equals brute force; a filtered search returns full
depth; a killed index build is dropped and rebuilt), `tests/db/test_vector_stream.py` (key order,
binary round-trip; `EXPLAIN` walks the vector key and probes `item_state`, 15b section 10). Live and
browser: section 9 ids of S3; stubbed `test_embedding_spaces_page.py`, `test_embedding_state.py`,
`test_image_search_popover_430.py`.

**Keep-list.** Fused search by reciprocal rank in Python (`rg -l semantic_search tests`); items
without a vector are counted (`test_live_embedding_state.py`); `TRANSITIONS`
(`domain/embedding_spaces.py:60-67`); float16 storage (15a 6.2).

**Size.** About 25 files, 1,800 lines changed, 1,500 deleted. Effort: high. **Owes the wave.**
Recall and latency of P12 on set P; the stream rate at 1 M.

### S4. Region stage and box editing

**Goal.** The region stage finds boxes inside objects and reads their text; the person creates,
moves, accepts, rejects and types boxes, closes a box set, gives a set verdict, undoes; box rows,
region drain and text search read it.

**Entry.** M9: box-row pages on X6, X26, X27 and the text forms at 10 M meet 50 / 200 ms, and
`fastupdate` is chosen by insert p99 (written by `0015`, S5, the first revision after M9).

| Id | Today | Replaced by |
|---|---|---|
| T3 | `_bulk_update_one_project` `scripts/curation/worker/bulk_writer.py:111` (`occ_skip_on_conflict_bulk` `:269`); `_write_box_embeddings` `:329` | `src/db/repos/region_results.py` (new) `replace_region_boxes` (old ids kept by IoU through `claim_match.match_boxes`, new ids from `items.box_seq`, `items.region_seq` +1) and `upsert_results` for `item_region_results`, `box_texts`, `class_opinions`, all inside `apply()`; box vectors by S3's `write_box_embeddings`. No lock check (R5) |
| T7 | `regions_edit._write_one` `:101`, `_batch_write` `:257`; `regions_boxes_edit._write_one_boxes` `:228` (`region_writes.human_box_write` `:68`, `human_status_box_write` `:114`) | human-act service with `claim_builders_region.put_regions`, `patch_box`, `batch_box_state`, `set_region_status`, `set_rejection_reason`; `expected_region_revision` checked under the item lock (409 `region_revision_conflict`); box vectors refreshed after the commit |

**Readers.** 6, 40, 42, 49 (the VLM worker's variant switches in S8a); the `boxes` module of the
filter package (box predicates and text, with the length-derived fuzzy threshold of 15b section 8).

**Deletes.** `bulk_writer.py` OpenSearch parts; `scripts/curation/worker/crop_gate.py` lock checks;
the OpenSearch parts of `regions_edit.py`, `regions_boxes_edit.py`, `region_writes.py`,
`region_rows.py`, `region_drain.py`, `region_impact.py`, `cascade._fetch_pending`,
`region_text_search.py`; `ensure_overlay_fields.inner_result_window`; `wire.region_to_wire` storage
mapping and `RegionFields.from_env` with the `OP_REGION_FIELD_*` variables (15a 5.5) into
`retired_env.py`. `tests/live/test_live_bulk_occ.py` is replaced by `tests/live/test_live_concurrent_claims.py`
(review resolve and region undo from two clients: no lost claim).

**Wire.** `rows_truncated` deleted; region rows order; `text_too_short`; `total_capped` and
`total_unknown` on region pages; slot gallery frontend rows of 15b section 5.

**Tests.** Unit, red first: `tests/curation/test_region_results_replace.py` (ids kept by IoU, a split
keeps the claimed box), `tests/curation/test_regions_boxes_edit.py` rewritten on claims,
`tests/domain/test_fuzzy_threshold.py` (no plate at distance 1 is missed). (DB):
`tests/db/test_region_writes.py` (a worker write racing a box edit: the person's box survives;
`region_revision` grows), box-row pages (only matching boxes returned; each filter shape uses its
index) and `sum_box` totals. Live and browser: section 9 ids of S4.

**Keep-list.** Whole-set verdict semantics (`test_region_verdict_semantics.py`); region undo and its
batch (`test_region_undo.py`, `test_region_undo_batch_bulk.py`); case-insensitive text search
(`test_regions_text_search_case_insensitive.py`); region paging (`test_regions_paging.py`); a human
accept leaves `locked` false on the wire (15a 4.4).

**Size.** About 30 files, 2,200 lines changed, 2,000 deleted. Effort: high. **Owes the wave.**
Worker throughput on set P against the plan 13 region numbers.

### S5. Near-duplicates and the holdout

**Goal.** Near-duplicate runs group the person's photos and objects, verdicts are recorded, and
freezing the test holdout pulls each group to one side so the test split does not leak.

**Entry.** Gate B (the batched kNN of row 61). M9's `fastupdate` choice is written by `0015` as
`ALTER INDEX ix_box_state_text_trgm SET (fastupdate = ...)` and the same for X25, since `0015` is
the first revision after M9 (S4 writes none).

| Id | Today | Replaced by |
|---|---|---|
| I5 | `near_duplicates/projection.py:93` `_bulk`, `apply_run` `:127` (images) | nothing for the project indexes: readers join `near_dup_members` of the current run (15a 2.3); the global path of `apply_run` (G6) stays |
| T11 | `near_duplicates/projection.py` `:93`, `:127` (items); `near_duplicates/holdout.flag_held_out` `:73` | the same for items; the freeze (`holdout_freeze.freeze_test_holdout` `:91`, C3 order kept) writes `claim_builders.holdout_freeze` drafts as a bulk human act in chunks |

Also here: verdicts (`repos/near_dup.upsert_decision`, `delete_decision`) become
`claim_builders.duplicate_verdict` claims and withdraws; `near_dup_pending` ids become keys;
`0015_near_dup_cutover` turns `near_dup_members.member_id`, `near_dup_edges.a_id`, `b_id`,
`holdout_near_dup_moves.item_id` into `bigint` keys and drops `near_dup_decisions`.

**Readers.** 45, 61, 63; call sites of 34 (`near_duplicates/holdout.py:170`) and 59
(`near_duplicates/vectors.py:87`, `:120`, `chain.py:43`) switch to S1a-ii's and S3's functions.

**Deletes.** `near_duplicates/projection.py` project-index paths; the OpenSearch reads of
`near_duplicates/vectors.py`, `candidates.py`, `rescore.py`, `member_views.py`, `status.py`,
`holdout.py`; `repos/near_dup.upsert_decision`, `delete_decision`. Left: `near_dup_fields.py` (S9).

**Tests.** Unit, red first: verdict claims and withdraws, freeze chunking. (DB):
`tests/db/test_holdout_freeze.py` (a freeze killed midway resumes; a frozen item keeps its class
against later opinions), `tests/db/test_near_dup_keys.py`. Live and browser: section 9 ids of S5;
stubbed `test_near_duplicates.py`, `test_export_split_counts_and_holdout_freeze.py`.

**Keep-list.** `tests/curation/test_holdout_freeze_errors.py`; split leakage
(`test_export_split_leakage.py`); `tests/near_duplicates/` group semantics.

**Size.** About 18 files, 1,200 lines. Effort: high. **Owes the wave.** Run time on set P against
today's; freeze time for a 10 % holdout at 1 M.

### S6. Clustering and the cluster screens

**Goal.** Clustering runs group the person's objects; the cluster screens (cards, representatives,
facets, region clusters, ordered pages) read the serving partition; activation flips a pointer; a new
photo is placed in the serving partition at ingest.

**Entry.** M10: cards from the `_now` views, the placement joins, the `all` tab per card and the
filtered cards, against 200 / 800 ms.

| Id | Today | Replaced by |
|---|---|---|
| T10 | `clustering/projection._bulk` `:147`; `clustering/region_box_rows.write_box_edits` `:232` | nothing: readers join `cluster_memberships` of `serving_partition_id` (15a 2.3); the job fills the partition's summaries before activation (15b 6.7); box partitions are read by join |

Ingest placement: in S1a-ii's commit transaction, `assign.place_new` results become
`cluster_memberships` rows of the serving partition before `attach` (the 02d immutability trigger
guards only `UPDATE OR DELETE`, `db/migrations/schema_0002/security.py:152-153`), so the new item's
first state row already counts in its cluster. `0016_cluster_cutover`: `cluster_memberships.item_key
text` → `subject_key bigint`, X22 replacing `ix_cluster_memberships_cluster`, X23, the four cluster
summaries and their `_now` views, the foreign keys of `sum_deltas.partition_id` and `run_id`,
`near_dup_runs.summaries_named_at`. `store_states`' `tally_context` gains the memberships. The cluster
filter module and the cluster-fit review tabs replace their `FilterNotMovedError`.

**Readers.** 8, 10, 11, 12, 13, 41, 60, 79; call sites of 59 for clustering, region false-positive
rows and the outlier and diverse orders switch to `vector_stream`.

**Deletes.** `clustering/projection.py`, the cluster rows of `search_projection_tasks`,
`clustering/pit_pages.py`, `clustering/vector_read.py` (last importers, section 8); OpenSearch bodies
of `cluster_cards.py`, `cluster_facets.py`, `clusters.py`, `region_fp_clusters.py`, `crop_orders.py`,
`residual_fetch.py`, `outliers.py`, `pool_snapshot.py`, `periodic/cluster_refresh.py`. Left:
`cluster_fields.py` (S9).

**Tests.** Unit, red first: cluster deltas for a move, an exclusion, a placement. (DB):
`tests/db/test_cluster_summaries.py` (cards equal the join aggregation after random writes),
`tests/db/test_partition_fill.py` (a writer that read the activation before naming and commits after
the fill is counted), `tests/db/test_cluster_placement.py`. Live and browser: section 9 ids of S6;
stubbed `test_cluster_*.py`, `test_clusters_empty_states.py`.

**Keep-list.** Card fields (`tests/curation/test_cluster_cards.py`, `test_cluster_card_near_dups.py`,
`test_cluster_card_dominant.py`, `test_cluster_hidden.py`); facets (`test_cluster_facets.py`); an old
move is not placed (`test_old_move_not_placed`, 15a 4.5).

**Size.** About 25 files, 1,800 lines changed, 1,500 deleted. Effort: high. **Owes the wave.** A
clustering run's read time at 1 M vectors; activation time with the fill.

### S7. Import, combine, export, training inputs

**Goal.** The person imports a labelled dataset, combines projects, exports a YOLO dataset with a
leak-free split, and trains from it.

**Entry.** None new: its reads are keyset walks and the export manifest (15b section 10), timed in
its sweep.

| Id | Today | Replaced by |
|---|---|---|
| I8 | `dataset_import/chunk.py:96` `_stamp_existing_image` | `photo_provenance` rows (kind `import`) |
| I9 | `projects/combine/image_copy.py:231` `_bulk` | photos copied through S1a-ii's commit step with `photo_provenance` (kind `combine`) |
| T17 | `dataset_import/item_labels.py:179`, `:209`; `proposals.py:112`; `region_labels.py:146`; `undo.py:426`; items via I1/T1 | `claim_builders_import.import_label`, `import_item`, region claims with `actor_kind 'import'`; `class_opinions` `import_suggestion`; detections, `region_boxes`, `item_notes` with `import_id`; reconcile through S1a-ii's `attach`; `fold.conflicts` gives `locked_conflict`; undo withdraws the operation's claims, deletes the `import_id` rows and re-attaches (T14's call site `item_delete._delete_round` `:145` goes) |
| T18 | `combine/copy_docs.py:46`; `combine/holdout.py:91` | claims with `actor_kind 'combine'`, `combine_conflict` claims; holdout by S5's freeze |

`0017_import_cutover`: `dataset_import_images.photo_key`, `item_ids` dropped;
`dataset_imports.dataset_name` written.

**Readers.** 39, 58, 67, 68; call sites of 38 in `export_readiness.py:114` and
`training/preflight_checks.py:181` (P3 when keys, else the guard's cap).

**Deletes.** OpenSearch parts of `dataset_import/` (`chunk.py`, `item_labels.py`, `proposals.py`,
`region_labels.py`, `undo.py`, `project_source.py`, `image_plan.py`, `reconcile.py`, `runner.py`),
`projects/combine/` (`image_copy.py`, `copy_docs.py`, `holdout.py`, `plan.py`, `warnings.py`),
`export_support.scroll_hits` and export bodies, `proposal_merge.py:109-170`; `edit_history.py` (last
importers).

**Wire.** 15a section 10: combine labels show `external_label`; `frame_mismatch` on import results.

**Tests.** Unit, red first: import drafts per label trust; combine conflicts. (DB):
`tests/db/test_import_record.py` (an import never overrides a person; undo leaves items a person
claimed), `tests/db/test_export_manifest.py` (an export during labelling reflects one snapshot; no
transaction stays open during the file copy). Live and browser: section 9 ids of S7; stubbed
`test_dataset_import.py`, `test_combine_projects.py`, `test_export_yolo_synchronous.py`.

**Keep-list.** Export golden files (`tests/curation/test_export_dataset_golden.py`); split leakage;
class identity by name across import, combine, export, train; `tests/curation/dataset_import/`; GPS
by policy in exports (`test_export_gps_by_policy.py`).

**Size.** About 35 files, 2,500 lines changed, 2,500 deleted. Effort: high. **Owes the wave.**
Export time on the public set; the scripted walk's export and training step (section 4).

### S8a. VLM, auto-label, assist loop

**Goal.** The VLM labels and verifies, auto-label and the assist loop propose; none can undo a
person. **Entry.** None new.

| Id | Today | Replaced by |
|---|---|---|
| T4 | `vlm_batch._write_updates` `:221`; `vlm_class_batch.write_updates` `:203`; `assist/verify.py:110`; `autolabel` | `class_opinions` (`vlm`) through `apply()` with the M17 guard (`results._opinion_guard`); verify fields in `item_region_results`; attempts in the ledger (kind `vlm`) |
| T19 | `assist/slot_write.write_slots` `:90`, `write_box_attributes` `:127` | `assist_answers` rows |

T21's `scripts/curation/cluster_raw_labels.py:212` writes `raw_label_clusters`; propagation
(`assist_loop/propagate.py`) writes `assist_propagated` opinions with `op_id`.

**Readers.** 17, 18, 44, 48, 50, 77; call site of 49 (`vlm_pending.py:211`).

**Deletes.** `class_write_guard.py` checks, `scripts/curation/vlm_pending.py` lock clauses,
`combined_resolve.py` guard reads, `assist_fields.py`; `vlm_class_attempt.py` history writes; the
names without a writer (15a 4.2: `cluster_majority_agreement`, `classifier_vlm_agreement`).

**Tests.** Unit, red first: an empty VLM answer keeps a class (M17). (DB): a VLM batch racing a
label: the label decides, the opinion is stored. Live: section 9 ids of S8a. **Keep-list.**
`tests/curation/test_combined_resolve.py`, `test_combined_attributes.py`, `test_assist_wire.py`.
**Size.** About 18 files, 1,200 lines changed, 1,500 deleted. Effort: high. **Owes the wave.** VLM
worker throughput on set P; the auto-label run end to end.

### S8b. Score and projection writers

**Goal.** The probe and the item scores rank, the projection map draws. **Entry.** None new.

| Id | Today | Replaced by |
|---|---|---|
| T13 | `item_scores/job.py:116`; `probe_predictions.py:290`; `embedding_viz.py:315` | `upsert_results` for `item_scores`, `probe_predictions`, `item_projections` (no fold, no counted column) |

**Readers.** 56, 57, 76; the `projections` filter module and the probe review tabs.

**Deletes.** The OpenSearch parts of `item_scores/`, `probe_predictions.py`, `embedding_viz.py`,
`embedding_viz_read.py`. **Tests.** Unit: the weighted top N; (DB) score pages on X20 and X21. Live
and browser: section 9 ids of S8b. **Keep-list.** `test_probe_*`; the probe's actionable flag
(`wire.py:253`). **Size.** About 12 files, 800 lines changed, 900 deleted. Effort: medium.
**Owes the wave.** The probe run time on set P.

### S8c. Re-runs: re-detect, open vocabulary, requeue, repairs

**Goal.** Re-detect, requeue and open vocabulary re-run the stages; claimed items survive.
**Entry.** None new.

| Id | Today | Replaced by |
|---|---|---|
| T14 | `item_delete._delete_round` `:145` | S1a-ii's `attach` removes items with neither a detection nor an asserting claim (15a 4.1) |
| T15 | `reprocess_detect._apply_plan` `:105`; `region_requeue.py:465`, `:545`; `registry_reclassify.py:281`; `reprocess_vlm.apply_vlm` `:86`; `repair_scan.py:106`; `empty_vlm_answer_repair.py:251`; `region_text_repair.py:281` | re-detect and open vocabulary: new detections and attach; requeue: `item_region_results` reset to pending under a new analyser and the ledger; reclassify: the class family walk of S2a; VLM reprocess: the ledger work list; the three repairs are deleted (they fixed document defects the record cannot have) |
| I7 | `open_vocab_run.stamp_open_vocab_status` `:348`; `open_vocab_ingest.py:73` | `analysis_ledger` kind `open_vocab` (state, reason) |

**Readers.** 51, 52, 53; call site of 79 (`open_vocab_sweep.py:29`).

**Deletes.** `item_delete.py` (last importers); `repair_scan.py`, `empty_vlm_answer_repair.py`,
`region_text_repair.py`, `stale_label_source_repair.py`, `unmatched_class_repair.py` if their only
target is a document defect (the lane lists each with its `rg` evidence); `open_vocab_fields.py`.

**Wire.** `frame_mismatch` on reprocess results (15a section 10). **Tests.** Unit, red first: the
open-vocabulary target kept against the VLM (15a 4.2 rank 1). (DB): re-detect with jittered boxes
keeps claimed items (`test_rerun_restores_state`, 15a 4.5). Live and browser: section 9 ids of S8c.
**Keep-list.** `test_open_vocab_item_wire.py`. **Size.** About 18 files, 1,200 lines changed,
2,000 deleted. Effort: high (attach under load). **Owes the wave.** Region worker throughput on set
P after a requeue.

### S9. Deletions, docs, re-ingest, acceptance

**Goal.** Nothing of the curation search path is left; the docs describe the record; the public sets
and set P are re-ingested and plan 15 section 11 passes. **Entry.** Every section 13 item and gate B
recorded; the pending list holds only S9's ids.

**Writers.** I11 (owner: `services/upgrade/from_060/admin_client.py:110`, `steps/legacy_repair.py:77`,
`steps/near_duplicates.py:78`), T21 (the remaining `seed_*` scripts, `seed_docs_embeddings.py`,
`seed_docs_run_results.py`, upgrade steps), G1 to G9 kept: the global indexes of inventory section
1.2, unchanged until phase 3. Index lifecycle (`lifecycle._create_one`
`clients/curation_opensearch/lifecycle.py:29`, `projects/delete.py:185`, `ensure_*.py`,
`services/curation/index_bootstrap.py`) deleted.

**Readers.** 62, 69 to 75 kept (global).

**Deletes.** The S9 rows of section 8 (`occ.py`, `occ_bulk.py`, `occ_upsert.py` after the fold
differential oracle is frozen into recorded cases, `tests/domain/world_allowed.py` keeping the
allow-list; `domain/history.py`; `curation_opensearch/` except what section 1.2 imports;
`image_completeness.py`, `image_counts_refresh.py`, `count_fields.py`, `near_dup_fields.py`,
`cluster_fields.py`); `projects.indexes` use, `project_indexes`, `search_projection_tasks` (`0018`);
`OP_PROJECT_INDEX_PREFIX` and the other curation index variables into `retired_env.py`;
`src/services/upgrade/from_060/` if not gone in S3 (section 8). A ratchet test fails when a module
outside the section 1.2 code calls `items_index()`, `images_index()` or the curation OpenSearch
factory.

**Docs.** `docs/ARCHITECTURE.md`, `docs/CURATION.md`, `docs/opensearch_schema_design.md` (reduced to
the global indexes), `docs/design/curation_api_contract.md`, `CLAUDE.md`, the docs site pages for
storage (concise user page, detailed developer page).

**Tests.** The full offline suite, (DB) tier, live harness and browser live tier on the merged tree;
the acceptance walk (section 4). **Size.** About 60 files, 300 lines changed, 8,000 deleted. Effort:
medium, high for the acceptance runs. **Owes the wave.** The acceptance record of section 4.

## 4. The acceptance walk (plan 15 section 11)

1. Re-ingest COCO (public) and set P on the record; the plan 13 four-way ingest table not below the
   numbers measured before the change; host CPU per photo per process recorded.
2. Derived stores: drop and rebuild `item_state`/`box_state` (`fold.rebuild`) and each vector index
   (`VectorIndex.rebuild`): same answers; `rebuild_tallies` on the settled record inserts no
   correcting row (R7).
3. Empty every rebuildable result table, re-run the analysers from the ledger: every claim attaches,
   and every state row equals its value before (revisions aside).
4. The scripted labelling walk on the public set: label by cluster, undo, bulk accept, exclude,
   freeze the holdout, export, train; trained-model AP equal to before the change.
5. Two API processes and two workers writing the same objects under load: no lost claim, no state row
   that differs from a fresh fold, every `_now` view equal to the `GROUP BY`.
6. Gate A bars at 10 M for every page and dashboard read with the final designs (15b section 1.3);
   gate B bars for the vector index.
7. The wire contract differs only by 15a section 10 and section 2 here (`make contracts-check`
   against the pre-cut-over contract).
8. The live harness and the browser live tier fully green; the route sweep at desktop and 430 px,
   screenshots viewed.

## 5. Risks and their early signals

| Risk | Early signal |
|---|---|
| The compactor falls behind the writers and reads slow down | pending `sum_deltas` per project between sweeps against M2's point; M3's rate |
| A writer path changes a counted fact outside `store_states` (drift) | the ratchet `test_counted_writers.py`; the nightly drift check's correcting rows |
| Non-HOT `item_state` updates bloat tables and indexes | `n_dead_tup` and index sizes between sweeps; `n_tup_hot_upd / n_tup_upd` against M5 |
| Sorts or capped counts on broad or rare filters are slow | `total_unknown` rate in the logs; the S1b-i sweep's `EXPLAIN (ANALYZE, BUFFERS)` rows removed by filter |
| Long snapshots hold back vacuum | `age(backend_xmin)` during rebuilds, partition fills and export manifests |
| Ingest throughput drops (fold and deltas inside the commit group) | the plan 13 four-way table in the wave B sweep |
| The vector index at 10 M (27 GB heap, HNSW build memory) | gate B; `maintenance_work_mem` warnings during the S3 build test |
| A lane runs out of working size | commits stop arriving at the planned concern boundaries; the coordinator splits by concern |
| The fold's differential oracle disappears with `occ_locks.py` | S9 freezes recorded cases first; the differential test count must not drop |
| Frontend breaks on the wire changes | `make contracts-check` and the stubbed browser tier in every slice |

## 6. Questions for the coordinator

Revision 0's questions 1 to 4 are answered below ("Coordinator answers" 6 to 9). New in revision 1:

1. **The 0.6.0 upgrade path breaks in S3, not S9.** `from_060` imports `KnnMethodSpec`,
   `legacy_names` and `legacy_embedded_clause`, which S3 deletes, and the clustering helpers S6
   changes (section 8). Proposal: the owner confirms answer 6 before wave C and S3 deletes the
   package; otherwise S3 cuts its `embedding_spaces` and `legacy_repair` steps and S6 its
   `clustering` step.
2. **15a stays above 1,000 lines** (about 1,535): the edits of this revision were made in place and
   tightened, but its field map (section 5) and appendix alone are about 200 lines. Proposal: move
   15a section 5 (field map) and appendix A to `15d_storage_field_map.md` in the next plan pass.
3. **Open vocabulary (I7) is in S8c, not S8a** (review M11 grouped it with the VLM): its ledger stamp
   and its attach are one flow inside T15. Confirm.

## 7. Trace: every writer in exactly one slice

Owner = the slice whose exit removes every part of the id; parts done earlier are named.

| Id | Owner | Parts done by another slice |
|---|---|---|
| I1, I2, I3, I4, I6 | S1a-ii | |
| I5 | S5 | the global `apply_run` path stays with G6 (S9) |
| I7 | S8c | |
| I8, I9 | S7 | |
| I10 | S3 | |
| I11 | S9 | `backfill_item_ids.py` deleted in S1a-ii |
| T1, T20 | S1a-ii | |
| T2, T5, T6, T8, T9, T16 | S2a | |
| T3, T7 | S4 | |
| T4, T19 | S8a | |
| T10 | S6 | |
| T11 | S5 | |
| T12 | S3 | the T3 and T7 callers of `write_box_embeddings` move in S4 |
| T13 | S8b | |
| T14 | S8c | `attach` built in S1a-ii; the import-undo call site replaced in S7 |
| T15 | S8c | |
| T17, T18 | S7 | |
| T21 | S9 | `seed_live_harness.py` in S1a-ii, `cluster_raw_labels.py` in S8a |
| G1 to G9 | S9 (kept, global, phase 3) | |

I1 to I11 (11) + T1 to T21 (21) + G1 to G9 (9) = 41 writer ids, each with one owner; S1a-i, S1b-i,
S1b-ii, S2b and F own none. Shapes: 15b section 12 (80, each in one slice): S1a-ii 6, S1b-i 8,
S1b-ii 7, S2a 1, S2b 9, S3 10, S4 4, S5 3, S6 8, S7 4, S8a 6, S8b 3, S8c 3, S9 8.

## 8. Module deletion by last importer

Importers from `rg -l` on `int/storage` (2026-11-15). A row's slice deletes the module; earlier slices
remove their own uses. A module not listed here is deleted by the slice that names it in "Deletes",
after the rule 2 check passes.

| Module | Importers outside the deleting slice | Deleted in |
|---|---|---|
| `src/db/repos/image_contents.py`, model `db/models/near_dup.py:45` | core `near_duplicates/content_key.py` (`resolve_many_core`, `claim_many_core`, `release_core`), `core_content.py`, `visual_search/ingest_batch.py`, `ingest_single.py`; `upgrade/from_060/steps/near_duplicates.py` | phase 3 (B1) |
| `services/curation/ingest_writer.py`, `repair_match.py` | `ingest_dedup.py`, `ingest_index.py`, `ingest_items.py`, `ingest.py` (S1a-ii); `scripts/bench/ingest_write_path.py` (deleted with it) | S1a-ii |
| `clients/image_counts_dirty.py` | `occ.py`, `occ_bulk.py`, `occ_upsert.py` (S1a-ii cuts the marks), `image_counts_refresh.py` | S1a-ii |
| `services/curation/image_completeness.py` | ingest modules (S1a-ii), `combine/copy_docs.py` (S7), `item_delete.py` (S8c), `bodies_core.py` (S9) | S9 |
| `services/curation/image_counts_refresh.py` | `image_item_write.py` (S2a), `item_delete.py` (S8c), `count_fields.py` (S9) | S9 |
| `clients/curation_opensearch/count_fields.py`, `IMAGES_INGEST_WRITE_MAPPING` | `bodies_core.py`, `index_bootstrap.py`, `ensure_overlay_fields.py` | S9 |
| `services/curation/class_undo.py`, `edit_undo.py` | S2a modules; `tasks/b4_fanout.py:11` | S2a |
| `services/curation/edit_history.py` | S2a modules, `routers/curation/regions_edit.py` (S4), `dataset_import/undo.py`, `region_labels.py` (S7) | S7 |
| `src/domain/history.py` | S2a modules, `region_box_pass.py` (S4), `dataset_import/undo.py` (S7), `vlm_class_attempt.py` (S8a), `registry_reclassify.py`, `reprocess_vlm.py`, `stale_label_source_repair.py`, `unmatched_class_repair.py` (S8c), `clients/occ_upsert.py` (S9) | S9 |
| `clients/opensearch/scroll.py` | `classes/projection_sync.py` (S2a), `clients/opensearch/clusters.py` (global) | phase 3 |
| `near_duplicates/near_dup_fields.py` | `review_request.py` (S2b), `semantic_search.py` (S3), `bodies_core.py`, `ensure_overlay_fields.py` (S9) | S9 |
| `clients/curation_opensearch/cluster_fields.py` | `copy_docs.py` (S7), `raw_label_clusters.py`, `cluster_raw_labels.py` (S8a), `embedding_viz.py`, `item_scores/base.py` (S8b), `bodies_core.py`, `ensure_fields.py`, `index_bootstrap.py` (S9) | S9 |
| `clustering/pit_pages.py`, `clustering/vector_read.py` | `near_duplicates/chain.py`, `vectors.py` (S5), `selection/pool_fetch.py` (S3), `region_box_rows.py`, `residual_fetch.py`, `outliers.py`, `pool_snapshot.py` (S6) | S6 |
| `services/curation/item_delete.py` | `dataset_import/reconcile.py`, `undo.py` (S7), `reprocess_detect.py`, `open_vocab_run.py` (S8c) | S8c |
| `curation_opensearch/crops.py` (`mget_crops`), `clients/occ.py`, `occ_bulk.py`, `occ_upsert.py` | about 20 callers through S8c | S9 |
| `src/services/upgrade/from_060/` | uses `KnnMethodSpec`, `legacy_names`, `legacy_embedded_clause` (deleted in S3), `clustering.partitions`, `activation` (changed in S6), `image_contents` (kept) | S3 if the owner confirms before wave C (question 1), else S9 with S3 and S6 cutting their steps |

## 9. Pending tests by id

`tests/pending_slices.txt` (S1a-i) holds one line per pending test, `<node id> <slice>`. The
conftest of `tests/` and of `frontend/e2e/` marks each listed id
`pytest.mark.xfail(strict=True, reason="pending <slice>")` ([pytest xfail, strict](https://docs.pytest.org/en/stable/how-to/skipping.html#strict-parameter)),
so a listed test that passes fails the run. S1a-i writes the first content: the ids below, whole
files expanded with `pytest --collect-only -q <file>` (offline). Each slice's merge removes its ids
and nothing else; a listed test that passes at a wave sweep is removed by the coordinator; a test
not listed that fails is a red for a lane. A live test turns green no earlier than S1a-ii's merge,
when ingest writes the record, so wave A's live reads are listed under S1a-ii.

| Test ids | Green from |
|---|---|
| `tests/live/test_live_ingest.py`, `test_live_bulk_restart.py`, `test_live_whole_photo.py`, `test_live_seed_from_detector.py`, `test_class_registry_live.py` (all ids); `frontend/e2e/live/test_writes.py::test_ingest_lands_in_the_project_status`; `frontend/e2e/live/test_route_sweep.py::test_route_mounts_cleanly[/ingest]` | S1a-ii |
| `tests/live/test_live_paging.py` (all); `test_route_mounts_cleanly[/workspace?view=detections]` (reads of S1b-i) | S1a-ii |
| `tests/live/test_live_images.py`, `test_live_capture_map.py`, `frontend/e2e/live/test_map_renders.py` (all); `test_route_mounts_cleanly[/workspace?view=gallery]`, `[/map]` (reads of S1b-ii) | S1a-ii |
| `tests/live/test_live_labels.py`, `test_live_annotation.py`, `test_live_detector_class.py`, `test_job_recovery_real_handlers.py`, `test_live_image_counts.py` (all); `tests/live/test_live_classes.py::test_merge_success_relabels_items`, `::test_labeling_against_a_deprecated_class_is_refused` | S2a |
| `tests/live/test_live_classes.py::test_registry_lists_the_seeded_classes_with_live_counts`; `test_route_mounts_cleanly[/dashboard]`, `[/classes]` | S2b |
| `tests/live/test_live_search.py`, `test_live_embedding_state.py`, `test_live_generic_detector.py` (all) | S3 |
| `tests/live/test_live_regions.py`, `test_live_region_drain_narrowed.py`, `test_live_region_text.py`, `test_live_ocr.py`, `frontend/e2e/live/test_data_agreement.py`, `test_deep_link.py` (all); `frontend/e2e/live/test_writes.py::test_region_confirm_then_undo`, `::test_editor_draws_moves_deletes_and_undoes_in_the_browser`; `tests/live/test_live_bulk_occ.py` (all: deleted in S4) | S4 |
| `tests/live/test_live_classes.py::test_freeze_test_holdout_selects_a_deterministic_cohort`, `::test_merge_is_refused_when_it_would_relabel_frozen_holdout_rows`; `tests/live/test_one_vs_n.py` (`test_sequential_scenario_has_an_empty_diff`, `test_concurrent_scenario_keeps_its_invariants`, `test_pool_sizes_follow_the_worker_knob`) | S5 |
| `tests/live/test_live_clusters.py` (all); `test_route_mounts_cleanly[/workspace?view=clusters]`, `[/workspace?view=clusters&class_name={region_class}]` | S6 |
| `tests/live/test_live_combine.py`, `test_live_export.py`, `test_live_train.py`, `test_live_train_promote.py`, `test_live_models_promote.py` (all); `test_route_mounts_cleanly[/export]`, `[/train]` | S7 |
| `tests/live/test_live_vlm.py`, `test_live_assist_loop.py`, `test_live_served_facts.py` (all); `test_route_mounts_cleanly[/review?tab=new_class_proposals]` | S8a |
| `test_route_mounts_cleanly[/review?tab=all]`, `[/review?tab=uncertainty]`, `[/review?tab=model_disagreements]`, `[/review?tab=classifier_blind_spots]` (their tabs read probe results) | S8b |
| `tests/live/test_live_open_vocab.py` (all); `frontend/e2e/live/test_writes.py::test_draw_move_delete_survive_a_redetect` | S8c |

Never listed (must pass at every sweep): `test_live_core_ingest.py`, `test_control_plane_smoke.py`,
`test_job_load.py`, `test_queue_delivery.py`, `test_sse_across_replicas.py`, the offline suite, the
stubbed browser tier, and the route sweep's `/models`, `/bakeoff`, `/settings`, `/projects` and
`test_global_route_mounts_cleanly` cases.

## Coordinator answers (2026-11-15)

| # | Question | Answer |
|---|---|---|
| 1 | Seven object sorts have no index | No refusal by size. Every sort the interface offers gets an index on the table that holds its value (a score sorts on its score table, which does not count against the limit on the state table). The read takes one of two paths chosen by the capped count it already has: under the cap, sort the filtered rows; over it, walk the sort's index in order and apply the filter. Both paths are measured in slice S1b at 10 million rows. A sort that misses the page bar on both paths is listed for the owner as a candidate to remove; it is not answered with an error. Revision 1: placement per the decision on review M3 (15b section 7) |
| 2 | Box rows list the newest object first, not the newest detected box | Accepted; listed in the contract diff. |
| 3 | The region stage polls its result rows, not the analysis ledger | Accepted: the cascade is a state machine per object. The ledger still records the finished state. |
| 4 | Review-tab membership would be written twice, in SQL and in Python | One source: a declarative table of tabs and their clauses, read by the SQL builder and by the Python evaluator. A property test holds the two interpreters equal on generated objects. No clause is written by hand in two places. |
| 5 | Cluster cards filtered by something other than the detector class cannot use the summaries | Measure on the public sets and on set P before limiting any filter; report the numbers. |
| 6 | Delete the 0.6.0 upgrade path in slice S9 | Planned as proposed, and listed for the owner to confirm before S9 starts: a 0.6.0 installation then moves its work by exporting its labelled datasets and importing them, not by an in-place upgrade. Revision 1: the confirmation is needed before wave C (section 6 question 1) |
| 7 | Core API data stays on the global search indexes until phase 3 | Confirmed; the reserved core project waits with it. |
| 8 | Cluster placement at ingest is removed in S1a and returns in S6 as membership rows | Accepted. |
| 9 | Replace the live bulk conditional-write test with a two-client claim test in S4 | Accepted. |

## Appendix. Review trace

One row per finding of the review of 15b and 15c (2026-11-15) and the place that now answers it.

| Finding | Answered in |
|---|---|
| B1 `image_contents` still used by core ingest | S1a-ii (kept, `0012` fences it); section 8 rows `image_contents`, `from_060` |
| B2 exclusive rebuild lock | 15b 6.6, 6.7; S1a-i; entry M4 |
| B3 summary rows locked to commit | 15b 6.2, 6.3; S1a-i; entry M1 |
| M1 class family on X2 | 15b P1; S1b-i (DB `EXPLAIN` test); entry M8 |
| M2 drained queue | 15b P1, P8, section 5; 15a X2, X4; S1b-i, S2b entries M6 |
| M3 index per sort against five | 15b section 7; 15a 2.6 X28 to X32; S1a-i (`0011`); entry M5, M7 |
| M4 drift paths | 15b 6.2 callers, 6.6; S1a-i tests; 15a sections 7 and 8 |
| M5 capped counts as guards | 15b section 5 tables; section 2 rows; S2b (guard caps) |
| M6 box rows | 15b section 9; S4 entry M9 and tests |
| M7 dropped reader behaviours | 15b P1, P15, sections 7 and 8, rows 15, 24, 30, 43; section 2 here |
| M8 attach without an owner | S1a-ii (attach); S7, S8c consume it; section 7 T14 |
| M9 deletion before the last importer | rule 2; section 8 |
| M10 red by file | rule 8; section 9 |
| M11 slices too large | section 1 (splits); section 3; section 7; 15b section 12 |
| M12 round trips per label | 15b 6.9; S2a entry M1 |
| m1 P6 short page | 15b P6; S1b-ii tests |
| m2 X22 direction | 15a X22; 15b P8 |
| m3 upsert order | 15b 6.5 |
| m4 advisory key space | 15b 6.3; 15a section 7 |
| m5 P4 heap fetches | 15b section 5 rule 8; entry M6 |
| m6 machine churn non-HOT | 15a 2.4; section 2 (`updated_at` row); 15b section 11 question 8 |
| m7 GIN `fastupdate` | 15b section 8; S4 entry M9; S5 (`0015` writes the choice) |
| m8 one filter file for six slices | rule 3; 15b P1 |
| m9 `photo_counts` serialises a photo | 15b 6.1, 6.2 |
| m10 mixed-direction sorts | 15b sections 4 and 7; S1b-i (`keyset_clause`) |
| m11 capped-total classification | 15b section 5 rules 1 and 9; S1b-i tests |
| Jobs (the review's question 6) | 15b section 10; S3 and S7 tests |

## Coordinator answers after the review fix pass (2026-11-15)

| # | Question | Answer |
|---|---|---|
| 1 | The count of the `all` review tab on a cluster card | Computed per visible card, measured in round 3 (M10). If it misses the dashboard bar it leaves the cards and stays on the queue page. |
| 2 | Crop area and rank live on `items` | Confirmed: written only when geometry or membership changes; a label never touches them. |
| 3 | "Recently updated" moves only on facts a person sees | Decided: a label, a class change, a geometry or box change, a region verdict. Background score writes do not move it. Told to the owner as a behaviour note. |
| 4 | The 0.6.0 upgrade code breaks in slice S3, not S9 | The owner's confirmation of its deletion is an entry condition of wave C. Without it, S3 keeps the upgrade code compiling by cutting only its imports of the moved modules, and S9 decides. |
| 5 | 15a is over 1,000 lines | Its field map and review appendices move to a companion file in the docs commit of slice S1a-i; no content changes. |
| 6 | Open vocabulary in S8c | Confirmed. |

