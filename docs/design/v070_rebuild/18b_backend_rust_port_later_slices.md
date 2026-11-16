# 18b: the backend in Rust, the slices after stage 0, the storage mapping and traceability

| Field | Value |
|---|---|
| Status | plan, revision 1 (2026-11-15), brought in step on 2026-11-16 with plan 16a's changes (section 4 there), plan 15b's corrections (its section 16), decisions 14 to 20 and what stage 0 built (18a section 5), companion of [18](18_backend_rust_port_plan.md) and [18a](18a_backend_rust_port_slices.md). Every slice here uses 18a's template (section 1.1), workspace rules (section 2), test and parity rules (section 3) and fail-closed rules (section 4) |
| Reads with | plan 15c on `int/storage` (each record slice builds the design of the 15c slice it names; section 6 maps them), 15a (the schema, binding), 15b (the read shapes and the measurements written "15b M1" and so on) |

Each slice below names its entry, what it builds, replaces and deletes, its contract changes, its
tests and its bar. The lane copies its slice, expands the module list with `rg` at entry, and lists the
documentation pages it read. Selective checks are always 18a section 1.1's; "Owed" is what the full
sweep runs for it.

## 1. Stage A: jobs, events, the record and the ingest switch

Merge order: R3a, then R3b and R1b (either order), then R11a and R1c in the same large merge; R2a
whenever its entry holds.

### R3a. Periodic ticks in Rust, Python workers claim from `jobs`, Celery leaves

**Goal.** Jobs and housekeeping run without Celery or a broker; the Python workers that stay take their
jobs from the same table. **Measurement.** Needs none. **Entry.** R1j and R1a merged; and a decision on R1j's open finding: a
runner that registers one kind, behind a large backlog of queued rows of other kinds, reads about
80,500 pages per poll and no index tried is used by the planner (18a R1j, "Built"). The Python
kinds this slice moves make that case real (one claim loop per kind group), so the lane measures it
on the claim query it writes and chooses the index, the query form or a per-kind queue, with the
plan and the page count recorded.
**Builds.** `op_server` roles `worker` (`op_jobs::run_worker`) and `periodic` (`op_jobs::run_periodic`).
Ticks in Rust (the inventory's list, logic from `src/services/periodic/` and `src/tasks/periodic.py`):
`reaper` (all rules of `src/services/jobs/reaper.py` and `src/db/repos/jobs_reaper.py` except the
re-dispatch rule, which has no meaning without a broker), `ingest_dispatch`, `upload_sweep` (stale
landing files; R1c adds unreferenced objects), `job_stats`, `vlm_batch_staging_sweep`,
`serving_inventory`, `reload_promoted`, `cache_prune`, `resource_reachability`, `db_backup`,
`gpu_arbiter` (the arbiter's placement logic; the lane reads the Docker Engine API documentation and
picks a maintained Rust client, citing its page), and the ticks that start work as a queued job row:
`open_vocab_sweep`, `cluster_refresh` (today's rule as it is, `src/services/clustering/refresh_schedule.py:314-343`,
queuing the Python `cluster_run` until R8-1 switches it to the Rust kind), `near_dup_rescore`,
`near_dup_core_group`, `near_dup_full_build`. Python: `src/tasks/claim_loop.py` (18 section 2.5: the claim query for the
kinds the process registers, `LISTEN op_jobs`, one subprocess per job, kill at `hard_limit_s`);
`dispatch()` becomes "insert the row and `pg_notify('op_jobs', ...)` in the same transaction";
`run_job` loses `SoftTimeLimitExceeded`; the conformance scenarios of 18a section 3.5 replayed in the
Python database tier. Compose: the Celery worker services keep their names and run the claim loop
with their kind groups and concurrency; new services `worker` and `periodic` run `op-server`; the
trainer, export and evaluator images replace their Celery entry with the claim loop.
`CLAUDE.md`'s `curation-cluster-refresh` row is corrected to the `cluster_refresh` tick.
**Replaces.** Celery beat (`src/tasks/beat_schedule.py`, `beat_lock.py`), the Celery app and task
registrations, `src/tasks/periodic.py`, `trigger.py`, `reaper.py`, the ported services of
`src/services/periodic/`, `src/services/jobs/reaper.py`.
**Contract.** None. **Tests.** Tick unit tests with a fake clock; the single ticker under two
`periodic` processes (fake lock); the claim loop's unit tests (a child killed at its hard limit is
marked as `jobs` says; a second loop claims nothing claimed).
**Owed.** `test_job_load.py`, `test_queue_delivery.py`, `test_job_recovery_real_handlers.py` live;
the conformance suite on both runners; no tick missed over a 600 s run.
**Go bar.** Owed green. **Deletes.** `src/tasks/app.py` and the Celery task modules,
`src/config/queue.py`, `src/config/task_routes/`, `src/clients/task_queue.py`,
`src/tasks/runtime/redis_transport.py`, the ported periodic modules, `celery`, `kombu`, `flower` from
the requirements, the `jobs-beat`, `periodic-worker` and `flower` services; their compose, CLI,
installer and test entries.

### R3b. Live events from Rust; Valkey leaves

**Goal.** The progress a person watches streams from the Rust server, woken by PostgreSQL.
**Measurement.** M17. **Entry.** R3a merged; M17 meets its bar (else its coalesced variant, 18
section 2.5).
**Builds.** `op_api` module `events`: the three event-stream operations, `events/publish`,
`events/stats`, the pipeline-event and global-status routes; axum `Sse` with keep-alive, one `LISTEN
op_events` connection per process plus a 2 s tick, the committed-safe cursor `(xid, id)` of
`src/services/events/cursor.py` as `Last-Event-ID`, a slow client dropped with `reconnect` as today.
`op_store`-independent event inserts (`op_db` statements) with `NOTIFY op_events` in the inserting
transaction. Python producers: `src/services/events/publish.py` writes `pg_notify` in the transaction
instead of the Valkey publish after commit.
**Replaces.** `src/routers/curation/events.py`, `pipeline_events.py`, `global_status.py`,
`src/services/events/sse.py`, `listener.py`, `src/tasks/runtime/event_wakeup.py`; every remaining
Valkey use (reachability probes and the queued marker become rows or go).
**Contract.** Unchanged. **Tests.** Cursor goldens; stream tests with a fake listener.
**Owed.** `test_sse_across_replicas.py` live; the frontend SSE tests; differential cases of the
family. **Go bar.** Owed green; commit-to-client event latency p99 not above Python's.
**Deletes.** The replaced modules; the `valkey` service, its client library, its variables into the
retired list, its compose, CLI, installer and test entries.

### R1b. The record and the fold in Rust

**Goal.** Rust can store a photo's record and every summary a person sees, by the rules plan 15
fixed, checked against the frozen Python oracle.
**Measurement.** 15b M1 to M5 recorded and meeting their bars (the summary design's entry in 15c).
What 15b section 16 marks as waiting for round 4 or the owner is built only after its cell: both
text search copies, their functions, the SQL file, `store_states`' `put_box_text` call and the go
bar of 15b section 8 (text search): R4-3, and the posting-table fallback if the copies miss R4-3 or
that go bar; backup and restore on a managed service, and the bootstrap on a managed service:
C-1 (the check on a real managed instance, which waits for the owner's yes); `item_state` fillfactor
90 or 80: R4-7; the batch and beat constants: R4-10 (constants only). Everything else in 15b
section 16's R1b row is final and is built now.
**Entry.** R1j merged; R0x's output; the schema dump in place.
**Builds.** `src/db/migrations/sql/0010_storage_record.sql` (R0x's rendering) and
`0011_storage_indexes.sql` (15a 2.6 and 15b 6.1: indexes X1 to X6, X8 to X21, X24 and X26 to X33, the `txt` search
copies with X7 and X25 (project first, `fastupdate = on, gin_pending_list_limit = 512`) and their
seven functions, `op_bound_project_strict` and the tightened `op_scope_project` (15b section 8), the
`items` and
`item_state` columns, the summary tables, `sum_deltas`, the `_now` views, statistics, fillfactors from
15b M5), each run by a ten-line Alembic revision. `op_domain` (new crate): the fold, claim matching,
claim kinds, result classification, tallies, the `updated_at` rule, the outputs 15c S1a-i lists;
R0x's files under `native/op_domain/tests/goldens/`. `op_store` (new crate): `photos` (insert, the
imohash advisory locks moved from `src/db/repos/image_contents.py:71-89`, SHA-256 confirmation),
`items`, `results`, `state` (`store_states` with the delta emission of 15b 6.2), `sum` (the compactor
of 15b 6.5 as job kind `sum_compact` with its tick, and the nightly drift check by `rebuild_tallies`),
`vectors` (per space, `COPY` into an `ON COMMIT DROP` temporary table then `INSERT ... SELECT`),
`vector_ddl` (space tables through a separate `op_vector` pool), `analysers` and the ledger, `apply`
and `attach` (15c S1a-ii's design), `events` (insert with `NOTIFY op_events`).
**Replaces.** Nothing on the wire yet.
**Contract.** None. **Tests.** Oracle replay (every committed case equal); the fold property test
(proptest: any order of machine writes around a claim gives the same row); repository unit tests over
statement text; tallies against a fresh `GROUP BY` on generated writes.
**Owed.** The full oracle set from `artifacts_local/` once; the `db` feature tests in the shape of
15c's S1a-i and S1a-ii list (a group stored whole, a crash between statements leaves nothing, the same
bytes twice give one photo, two concurrent groups with one imohash give one photo, attach serialises
per photo, `op_app` cannot create in `vec`, the compactor under 30 writers within 15b M3's rate,
summaries equal to `GROUP BY` after compaction); the statement-prepare test.
**Go bar.** Owed green; the oracle has no divergence outside recorded Python defects listed with the
owner; and the go bar of 15b section 8 (text search: the two-project matrix over every function, the
scopes, the mutation run, the privileges and the catalogue test, the concurrency test) for the text
copies, once R4-3 has passed. **Deletes.** Nothing (the Python step 1 never reaches the integration
branch).

### R11a. Projects: create, list, delete

**Goal.** A person creates and deletes projects on the record, with their vector tables, files and
class registry. **Measurement.** Needs none. **Entry.** R1b merged. Merges in the same large merge as
R1c, never before it (a Rust-created project has no search indexes for the Python ingest).
**Builds.** `op_api` module `projects` (`GET /curation/projects`, create, delete, the project page
reads) and the job kinds `project_create` (registry row, default spaces' vector tables through
`vector_ddl`, directories, the MLflow experiment as today) and `project_delete` (record rows under
`op_project_delete_allows`, objects by prefix `projects/<project_id>/` through `op_storage::erase`,
vector tables).
**Replaces.** The project routes of `src/routers/curation/` and `src/services/projects/lifecycle.py`,
`delete.py` except the combine part (R9).
**Contract.** Unchanged. **Tests.** Handler and job unit tests. **Owed.** The project live tests;
differential cases of the family. **Go bar.** Owed green. **Deletes.** The replaced modules by last
importer (the OpenSearch index lifecycle stays until R2b if a core path imports it).

### R1c. Ingest on Rust, uploads included (the switch)

**Goal.** A person's photos, uploaded from the browser or an API client or read from a folder on the
server, are processed at the card's rate, stored once in the object store, and committed to the record
without the server holding or copying whole photos.
**Measurement.** M3 to M5, M10 and M18 recorded (M1 and M2 are not run: 18 section 8); D2 decided; M16 optional.
**Entry.** N3 merged (N1b and N1c for the benchmark plan); R1b, R1s, R1u, R1x, R1e, R3a, R3b merged;
R11a in the same merge.
**Builds.** `src/db/migrations/sql/0012_ingest_cutover.sql` (fences `image_contents` to the core API's
rows: deletes rows with a `project_id`, adds `CHECK (project_id IS NULL)`). `op_ingest` pipeline:
admission, `imohash` from the landing file, the content claim, the engine call (bytes read once from
the landing file; by key once plan 17's item lands), capture through the ExifTool pool by path (or
D2's branch), the mover before the commit (18 section 4.2), the commit group (attach, `apply()`,
`store_states`, vectors, ledger, `crop.created` events), then, after the group commits, the queued
`cluster_place` job (the line is R8-1's; no numeric code runs in the commit), the crop JPEGs into `derived/crops`, the
open-vocabulary job row after ingest (a Python kind until R10, taken by the claim loop), per-photo
result values, the 503 with `Retry-After` when the engine is down for every new photo. The upload
route of R1u mounted in `op_api`, with the pipeline as its `PartSink` (18a R1u, "Built"); `batch`, `objects`, `bulk`, `sources`, `path_lookup` and `status`
handlers (the lane lists the operation ids at entry; `config` and `policy` stay with R11, `region_drain`
with R7); job kinds `bulk_ingest_run`, `source_poll`, `source_backfill`; `upload_sweep` extended to
unreferenced objects (18 section 4.2 step 6, under the imohash lock rule of 4.5).
`tests/pending_rust.txt` gets the ids this switch turns red (15c section 9's lists are the starting
point). The seeding of the ingest live tests moves to `seed_http/ingest.py`.
**Replaces.** The ingest operations of `src/routers/curation/ingest.py`, `ingest_upload.py`,
`ingest_objects.py`, `ingest_bulk.py`, `ingest_sources.py`; `POST /image_counts/backfill` and the job
kind `image_counts_backfill` and tick `image_counts_drain` (deleted: counts follow every write).
**Contract.** 15c section 2's ingest rows (`crops_updated`, `crops_preserved_human`,
`crops_final_conflicts`, `incomplete_write`, `held_back` deleted; `image_path` shows the display path of
`photos.storage_key`). Frontend: `frontend/src/lib/api_ingest.ts` sends `source`, `image_paths` and
`run_id` before the `images` parts; `scripts/curation/ingest_upload.py` the same.
**Tests.** Pipeline unit tests with fake engine, store, storage and capture traits; the commit group
on a fake store; the reconcile rule.
**Owed.** Differential cases of the ingest family against the reference; `tests/live/test_live_ingest.py`,
`test_live_whole_photo.py`, `test_live_bulk_restart.py`; the four-way ingest rows and M10's upload rows
on the native engine (one GPU hold, `coco2k` and `setp200`, one card and two); the regression run of
18 section 5.4; a 600 s soak with failed 0; an S3-compatible store on loopback for one upload row.
**Go bar.** API host CPU per photo at most 0.5x the Python route's (the capacity rows of 18 section 1.1, M10) at 20 MP, server-side
and upload (18 section 3.4 says why small photos are reported, not barred, unless D2 or M16 removed
ExifTool); photos/s not below Python's at any client count by more than the spread, and above it by
more than the spread in the host-bound row (20 MP, plan F on one card or plan C on two); failed 0.
**Deletes.** By last importer, listed at entry: the ported routers; `ingest_batch.py`,
`ingest_finish.py`, `ingest_stage.py`, the commit part of `ingest_index.py`, `upload_ingest.py`,
`bulk_ingest/`, `src/utils/file_read.py`, `src/clients/frame_engine.py` and
`src/domain/frame_engine_wire.py` if no other importer, `capture_read.py` and
`src/clients/exiftool_client.py` once `source_backfill` is Rust, the image-counts modules of 15c's
S1a-ii list; bench scripts that import the Python ingest service are deleted when they measure the
Python path or moved to HTTP. Python modules still imported by import, reprocess or open vocabulary
stay unmounted until R9 and R10 remove those importers.

### R2a. Inference routes on the engine

**Goal.** `/detect`, `/faces`, `/embed`, `/ocr` and `/analyze` answer from the native engine with no
Python preprocessing. **Measurement.** M6. **Entry.** N3 merged; R0c, R1a, R1e merged; and a decided response shape for each
operation of this family that `contracts/openapi/core.json` leaves open (18a R0c, "Built": here the
free-form `FaceVerifyResponse`, `TestInferenceResponse` and the `failures` items of the three batch
answers), written into the file by a hand edit with the family's intended-change entry before a Rust
type is held to it.
**Builds.** `op_api` module `inference`; the `/v1` twins kept. Uploads stream as in R1u and go to the
engine as bytes (`IMAGE`), since these routes store nothing. Text tokens through tokenizers, checked
token for token.
**Replaces.** `src/routers/detect.py`, `analyze.py`, `ocr.py`, `embed.py`, the detect, recognize and
verify routes of `faces.py`, `models.py` (model test); `src/clients/triton_preprocess.py`,
`src/utils/image_processing.py`, `cpu_preprocess.py`, `face_align.py`, `scrfd_decode.py`,
`src/clients/pe_encoder.py`, `pe_text_backends.py`, the API's `transformers` in `src/utils/cache.py`
and torch in `src/services/detection/ensemble_nms.py` (M8 says whether they load today).
**Contract.** `contracts/openapi/core.json` shapes unchanged; intended: boxes, scores and vectors
follow the engine's preprocessing. **Tests.** Handler tests on fake engine answers; token goldens.
**Owed.** Ground truth, Python route against Rust route on the same photos: detector mAP on COCO val;
LFW, CALFW and CPLFW verification; COCO-Text F1; each not below Python by more than its two-run
difference; M6 rerun; differential cases (shapes) of the family.
**Go bar.** Those rows; API CPU per request below M6's. **Deletes.** The listed modules.

## 2. Stage B: the gallery, labelling, erasure and settings

### R4. Gallery reads: photos, objects, crops, thumbnails, map, table

**Goal.** The person sees their photos and objects again, on the record: lists with every sort and
filter, the photo page, crops, thumbnails, the map, the table. **Measurement.** 15b M6, M7, M8; M7. What 15b section 16 marks as waiting is built after
its cell: the sort threshold and the planner-estimate step of the path decision: R4-5; the band:
owner question 2, asked with R4-5's numbers; the detector-class count form: R4-4; the text predicate
inside P1, P4 and P8: R4-3; reads under writers: R4-1 (a miss goes back as code work).
**Entry.** R1c merged.
**Builds.** `op_api` module `gallery`; `op_store` reads of 15b (P1 to P7, P15, the keyset forms of 15b
section 7 including the mixed-direction form, the capped totals of 15b section 5); `op_domain` filter
grammar and cursor codec (proptest); crop and thumbnail serving from `derived/` through
`op_storage::derived` (generated on a cache miss in the rayon pool with turbojpeg as the engine uses).
**Replaces.** crops 33, images 8, map and map cells, table, viz 4 operations; 15c S1b-i, S1b-ii and
F's serving part.
**Contract.** 15c section 2 rows for S1b-i and S1b-ii (`total_capped`, `total_unknown`,
`index_capped`, `capped`, `join_truncated` deleted, tie order, `sort_options`), with their frontend
edits (15b section 5's interface table). **Tests.** Unit tests per statement builder; cursor
property tests. **Owed.** The family's differential cases; its live and browser tests (moved off the
pending list); 15b's 10 M timing rows for these shapes; the map page in a browser at desktop and 430 px.
**Go bar.** 15b's page bars (50 / 200 ms); p50 and p99 at 8 clients and API CPU per request not above
the reference. **Deletes.** The replaced routers and their services by last importer.

The gallery frontend update (the owner's step 3) is the photo-library plans' work and starts when R4
merges.

### R5. Labelling acts, undo, review, classes, counts

**Goal.** The person labels, undoes, reviews and sees class counts again, and no machine writer can
undo a person. **Measurement.** 15b M1 (label variant), M6, M7. Waiting, per 15b section 16: the routed review queue:
R4-6; more than one chunk job at a time (`K` above 1, today's exclusive group): R4-8; the "to review"
definition: owner question 1. **Entry.** R4 merged.
**Builds.** `src/db/migrations/sql/0013_claims_cutover.sql` (drops `label_operation_members`; also creates
`label_selections` and `label_selection_items`, plan 16b section 1.2);
`op_store::human` (the human-act service of 15c S2a: operation row, claim builders, `apply()`,
events, chunks of 1,000), the review tabs as one declarative table read by the SQL builder and the
Rust evaluator (15c answer 4), the readers of 15c S2b. Job kinds `label_op`, `label_undo`,
`class_merge`, `review_resolve_batch`, `edit_undo_batch` in Rust. Every label act carries the
precondition of plan 16b section 3.1: `class_version` per shown object, `selection_token` from the
dry run for filter acts, `expected_class_version` on the single label; the item wire gains
`class_version`; the frontend sends them (`api_cropWrites.ts`, `api_select.ts`,
`selectionActionController.svelte.ts`); the `db` tier's two-session label test also covers a stale
tab.
**Replaces.** classes 9, review 7, label_ops 4, select 4, scores 4, stats 3 operations; the
`interactive-worker` service goes.
**Contract.** 15c section 2 rows for S2a and S2b and 15a section 10's (`conflicts`, `conflict_ids`
deleted). **Owed.** Differential cases; live and browser tests of the family; the two-session label
test of 15c S2a in the `db` feature. **Go bar.** Owed green; the label route's p99 on set P within 15b
M1's bar. **Deletes.** The replaced routers and services by last importer.

### R5e. Erasure

**Goal.** A person can erase a photo, a person's faces or a project for good, with everything derived
from it, and the act is logged without its content. **Measurement.** Needs none. **Entry.** R5 merged.
**Builds.** The erasure job of 15a section 8 in `op_jobs`: the `erasure_jobs` guard, the catalogue
(15a's table, from `src/services/erasure/catalogue.py`'s design on `int/storage`), chunked deletes in
the lock order of 15a section 7, the file step of 18 section 4.5 (reference check by query and delete
under the imohash advisory lock), the `privacy_purges` log and the privacy audit read; the trash
emptying route. R8-1 adds the rows and the trigger change of plan 16b section 1.3, and B5-2 (if
built) the rows of 16b section 3.5, to the catalogue; the completeness test covers them.
**Replaces.** `src/routers/curation/privacy_purges.py` and `privacy_audit.py` behaviour (by
operation), `scripts/purge_*_data.py` for project data.
**Tests.** Unit tests of the catalogue walk. **Owed.** `test_erasure_catalogue_complete` and
`test_erasure_leaves_nothing` in the `db` feature; an upload of the same bytes racing an erasure leaves
the photo with its object. **Go bar.** Owed green. **Deletes.** The replaced modules.

### R11. Configuration documents, settings, VLM registry, models, health

**Goal.** Every settings and model screen works on Rust, with the same revision and `ETag` behaviour.
**Measurement.** Needs none. **Entry.** R1c merged; the core health and pool operations
(`/health`, `/health/pe_text`, `/ready`, `/connection_pool_info`) declare no response shape in
`contracts/openapi/core.json` (18a R0c, "Built"): their shapes are decided and written into the file
by a hand edit first.
**Builds.** `op_api` modules for region_profiles 17, prompt_packs 14, ingest_profiles 14,
cluster_profiles 7, near_dup_profiles 7, assist_loop_presets 7, serving_profiles 6, `/curation/vlm/*`
15, ingest `config` and `policy`, keymap, location_policy, settings, pause, bakeoff 9,
`/curation/health`, `/curation/embedding/models`; core `health.py` (`/ready`, `/health`, the pool and
metrics reads), `api_docs.py`, `models_lifecycle.py`, `models_export.py`; `src/services/config_store/`
ported with its revision rules; the keymap and class contracts' sources move to Rust (files stay
byte-identical); `bakeoff` and `model_export` stay Python kinds.
**Owed.** Differential cases; live tests of the family. **Go bar.** Owed green; p50 and p99 not above
the reference. **Deletes.** The replaced routers and `src/services/config_store/` by last importer.

## 3. Stage C: the database and the clustering

| Slice | Plain name | Revision | Entry (besides the previous slices) | Kinds and workers | Notes |
|---|---|---|---|---|---|
| R6a | Vectors and search | `0014_vector_cutover` | gate B recorded; R5 merged | `embedding_reindex` | the vector-index interface of 15c S3 in `op_store`; search by text (tokenizers), by similar item, by uploaded image; the `op_vector` pool; search 3, embedding 8 operations |
| R6b | Near-duplicates and the holdout | `0015_near_dup_cutover` | 15b M9 and M13 recorded; R6a merged | `near_dup_group`, `near_dup_rescore` (petgraph, pathfinding); `near_dup_score` per D5 | near_duplicates 9 operations; verdicts as claims; the freeze as a bulk human act |
| R7 | Regions, boxes and the region stage | none | 15b M9 and M9 recorded; R5 merged; waiting for round 4 (15b section 16): P9 with fragments and `fuzzy_items`' threshold, the pending-list setting under writers, and the `text_lines` copy writer with `line_text_has` and word search: R4-3 (then the wave sweep on set P) | `region_fp_centroids` (Rust); the `curation-detection-worker` becomes an `op-server` worker role calling Triton, the segmenter and the VLM over tonic and reqwest | regions 9, region_stage 3, ingest `region_drain` |
| R8-1 | Groups on the record: storage and jobs (plan 16a section 3; R8 split by concern) | `0016_cluster_cutover` (plan 16b section 1.1; also creates `ix_sum_deltas_cluster`, 15b 6.1) | 15b M10 and M12 recorded; R6b and R7 merged; L16-1 (the `op_numeric` library, plan 16a) merged; plan 18's M12 bar met (section 8 there: it decides CPU or GPU inside Rust); waiting for round 4 (15b section 16): P7 against its bar and the compactor's rate with the card refresh: R4-2; filtered cards: R4-9 and owner question 3; the "to review" count: owner question 1 | `cluster_run` in Rust (items and boxes; no Python fallback); new `cluster_place`, queued after the ingest commit (placement is no longer part of the commit) | the Python `cluster_run`, `cluster_projection_sync`, `src/services/clustering/` and the placement calls go; the `cluster-worker` service goes |
| R8-2 | The cluster routes and screens on Rust (plan 16a section 3) | none | R8-1 merged | none | clusters 8 operations; read-time group counts |
| R9 | Import, combine, export, training inputs | `0017_import_cutover` | R8-1 merged (export reads claims and state, never `head_proposals`) | `dataset_import`, `dataset_import_undo`, `dataset_upload_extract`, `combine` in Rust; `train`, `promote` stay Python | datasets 11, export 7, combine 5, train 18 operations; exports and training files under `derived/` |
| R10 | Machine helpers and re-runs | none | R9 merged; nothing of 15b section 16 waits for it unless the owner answers question 1 with (a) or (d): then the probe job and the embedding re-index report membership changes (through `store_states` for (a), per probe batch for (d)) | `reprocess`, `assist_run`, `selection`, `open_vocab_drain`, `vlm_label_batch`, `vlm_verify_region_batch`, `autolabel`, `assist_loop`, the VLM worker loop, `item_scores` per D5 in Rust; `embedding_viz` and `probe` stay Python with record readers and writers written once | assist 15, assist_loop 15, open_vocab 15, pipeline 8, reprocess 3, probe operations; the `open-vocab-worker`, `curation-vlm-worker`, `curation-auto-label-worker`, `curation-assist-loop-worker` services go; the assist loop keeps today's cluster-sampled roots on the serving partition (one partition per target, or per scope if plan 16's B4 is built) |
| R2b | Core data on the record; OpenSearch leaves | `0018_reads_cutover` | gates A and B recorded; D4 decided; R10 merged; a decided response shape for each core operation `contracts/openapi/core.json` leaves open (`POST /ingest`'s 200 duplicate answer; the free-form `JobView`, `SearchResult`, `ObjectSearchResult`, `OCRSearchResult`, `ListAlbumsResponse`, `ObjectSearchResponse`; 18a R0c, "Built"), written into the file by a hand edit | `directory_ingest` (R1c's pipeline bound to the reserved core project); `persons_cluster`, `near_dup_core_group` per D5 | core `ingest/`, `search/`, faces search and identify, `persons.py`, `clusters.py`, `query.py`, `query_duplicates.py`, `jobs.py`; the five global indexes become the reserved core project's tables (15 section 14 answer 1) |

Each row's slice follows 18a's template: the family's Rust handlers, repositories and kinds; the
contract rows of 15c section 2 for its 15c slice, with frontend edits; differential cases and the
family's live and browser tests (removing its pending ids); 15b's timing rows for its shapes; p50 and
p99 at 8 clients and API CPU per request not above the reference; deletion by last importer with
ledger rows. R2b, when D4 says no, also deletes the OpenSearch compose service and Dashboards, its
health, metrics and error mapping in `src/core/`, its install and heap steps in `scripts/lib/` and
`scripts/setup.sh`, the benchmark scripts that target it, its `OP_*` variables into the retired list,
`src/services/visual_search/`, the global-index parts of `src/services/face_identity/`, the rest of
`src/clients/opensearch/`, `src/db/repos/image_contents.py` and its model, the curation index
lifecycle, `project_indexes` and `search_projection_tasks` (`0018`), and the core ingest upload
routes' disk guard moves into Rust with them. At the end of stage C the acceptance walk of 15c section
4 runs once on the Rust stack (re-ingest, derived-store rebuilds, the scripted labelling walk, two API
processes and two workers under load, gate A and B bars, the contract diff, live and browser tiers).

## 4. Stage D: R12, the Python API removed

**Goal.** One backend language on every request path. **Entry.** Every slice above merged; the
pending list empty; every ledger closed.
**Deletes.** `src/main.py`, `src/app/`, the FastAPI parts of `src/core/`, `src/routers/`,
`src/schemas/` and the Pydantic settings only the API used, FastAPI and uvicorn from the requirements,
the `api-py` service and `OP_PY_API_URL`, the forwarder, the FastAPI contract generator (the frozen files stay), `scripts/curation/seed_live_harness.py`. The schema leaves
Alembic: a baseline SQL migration from the head schema, without the columns that lost their writer
(`jobs.dispatch_count`, `jobs.last_dispatched_at` and any other listed at entry), run by `op-server
migrate` in the `db-migrate` service; the migration library (refinery or another over tokio-postgres)
is chosen here from its documentation. The retired-variable list's source and the job-kinds source
move to Rust; `scripts/lib/retired_env.sh` and the claim loop read the generated files.
**What remains of `src/`, and its rule.** `src/db/` (models, session and repositories for the job
tables and the stay kinds' tables: training runs, models and engines, probe predictions, projections,
bake-off results), `src/config/` (the settings the stay workers read), `src/tasks/claim_loop.py` and
`src/tasks/runtime/`, `src/services/training/` and the promote and probe code, `embedding_viz` with its
record reader and writer, `export/`, `scripts/curation/bakeoff/`. A layering rule in
`check_layering.py` fails any import of FastAPI, an OpenSearch client, an HTTP server or the fold in
the surviving package. Scripts: every file under `scripts/curation`, `scripts/bench`,
`scripts/research` and `scripts/validation` that imports `src` (53, 24, 19 and the validation count,
listed by `rg -l` at entry) either calls the API over HTTP, imports only the surviving package, or is
deleted.
**Docs.** `CLAUDE.md`, `docs/ARCHITECTURE.md`, the docs site's architecture and deploy pages (concise
user page, detailed developer page).
**Owed.** The whole suite on Rust; the cloud benchmark's local rows (B6).

## 5. Stage E: the cloud benchmark

18 section 5 is the specification; the runs need R1c (B1 to B3) and R4 (B4), the deploy shape of 18
section 6, and `op-upload-loadgen` (R0g). No slice writes product code for it; a deploy plan writes
the compose overlay for an S3 store and a remote landing host if the first run needs one.

## 6. The storage slices of plan 15c, mapped to this plan (decision 1)

"Kept as design" means the 15c text is the specification a Rust slice builds; "kept as SQL" means the
migration's SQL is written once, as a file; "no longer written" means the Python code 15c planned is
never written.

| 15c slice | What 15c planned | Rust slice | Kept as design | Kept as SQL | No longer written |
|---|---|---|---|---|---|
| step 1 (done on `int/storage`) | claims, fold, results, repositories, `apply()`, `vector_ddl`, the erasure catalogue | R0x, R1b, R5e | the fold, claim kinds, truth table, catalogue | `0010` (rendered by R0x) | nothing new; frozen as the oracle, never merged into the integration branch |
| S1a-i | `0011`, `store_states`, tallies, `sum_compact`, review tabs, the pending-test mechanism | R1b (state, tallies, compactor, drift check), R5 (review tabs), R0d (pending list) | all | `0011` | the Python state writer, tallies, compactor, review-tab evaluator |
| S1a-ii | `0012`, `commit_group`, `attach`, `vectors`, embedding-space DDL, harness seeding | R1b (`attach`, vectors, DDL), R1c (commit group), R0e (seeding over HTTP) | all | `0012` | the Python commit path, `attach.py`, `vectors.py`, `ddl.py`, the repository-based seeder |
| S1b-i | item filter SQL, keyset, totals, `item_reads` | R4 | all (statement text is reused) | none (no revision) | the Python readers |
| S1b-ii | photo reads, crop and thumbnail serving | R4 | all | none | the Python readers |
| S2a | `0013`, the human-act service, claim builders, undo | R5 | all | `0013` | `human.py`, the claim builders in Python |
| F | `RootedStore` callers | R1s (layer), R4 (serving), R9 (exports, training, archives) | the key contract, as 18 section 4.3 extends it | none | every Python caller move |
| S2b | review queue, class counts, statistics | R5 | all | none | the Python readers |
| S3 | `0014`, the vector-index interface, search, vector jobs | R6a | all | `0014` | the Python interface and jobs |
| S4 | region stage writes and box edits | R7 | all | none | the Python region writers |
| S5 | `0015`, near-duplicate keys, verdicts, the holdout freeze | R6b | all | `0015` | the Python near-duplicate writers |
| S6 | `0016`, cluster summaries, placement | R8-1 (storage and jobs), R8-2 (routes) | all | `0016` (plan 16b section 1.1), placement as the `cluster_place` job, read-time group counts | the Python cluster readers and placement |
| S7 | `0017`, import, combine, export, training inputs | R9 | all | `0017` | the Python import and combine writers |
| S8a | VLM, auto-label, assist loop writers | R10 | all | none | the Python writers (they port to Rust, 18 section 2.3) |
| S8b | score and projection writers | R10 | all | none | Python score writers except `embedding_viz` (stays Python) and `item_scores` if D5 keeps it |
| S8c | re-runs, open vocabulary, requeue, repairs | R10 | all | none | the Python re-run code; the three repairs stay deleted |
| S9 | `0018`, deletions, docs, re-ingest, acceptance | R2b (`0018`, OpenSearch), R12 (Python API), each slice (its docs) | the acceptance walk (section 3 end) | `0018` | nothing: the deletions are the same work in either language |

15c's pending test list (its section 9) is the starting content of `tests/pending_rust.txt` (R1c), and
its module deletion table (its section 8) is the last-importer evidence each Rust slice re-checks with
`rg` at entry.

## 7. Traceability

### 7.1 Routes

| Area (inventory 1.2, 1.3) | Slice |
|---|---|
| Core: `detect.py`, `analyze.py`, `ocr.py`, `embed.py`, `models.py`, faces detect, recognize, verify | R2a |
| Core: `ingest/`, `search/`, faces search and identify, `persons.py`, `clusters.py`, `query.py`, `query_duplicates.py`, `jobs.py` | R2b |
| Core: `health.py` (`/live` in R1a, the rest in R11), `api_docs.py`, `models_lifecycle.py`, `models_export.py` | R11 |
| `/v1` twins | with each core route |
| Curation ingest (27): upload, batch, objects, bulk, sources, path lookup, status | R1c |
| Curation ingest: config, policy | R11; region_drain: R7 |
| Projects list, create, delete | R11a; combine: R9 |
| crops, images, map, table, viz | R4 |
| classes, review, label_ops, select, scores, stats | R5 |
| privacy purges and audit | R5e |
| search, embedding | R6a; near_duplicates: R6b |
| regions, region_stage | R7 |
| clusters | R8-1 (jobs), R8-2 (routes) |
| datasets, export, combine, train | R9 |
| assist, assist_loop, open_vocab, pipeline, reprocess, probe | R10 |
| configuration documents, VLM registry, keymap, location_policy, settings, pause, bakeoff, curation health, embedding models | R11 |
| events (3 streams, publish, stats), pipeline events, global status | R3b |
| `/curation/jobs/{job_id}` | R3a |
| `POST /image_counts/backfill` | deleted in R1c |

### 7.2 Job kinds (45)

| Kinds | Slice or owner |
|---|---|
| `noop`, `sleep_noop` | R1j |
| `bulk_ingest_run`, `source_poll`, `source_backfill` | R1c |
| `project_create`, `project_delete` | R11a |
| `directory_ingest` | R2b |
| `label_op`, `label_undo`, `class_merge`, `review_resolve_batch`, `edit_undo_batch` | R5 |
| `embedding_reindex` | R6a |
| `near_dup_group`, `near_dup_rescore` | R6b (Rust) |
| `near_dup_score` | R6b per D5 (M13) |
| `region_fp_centroids` | R7 (Rust) |
| `cluster_run` | R8-1 (Rust; M12 decides CPU or GPU; no Python fallback) |
| `cluster_place` (new, plan 16) | R8-1 |
| `head_fit` (plan 16 B5-2) | B5-2, only if B5 is built: it makes 46 |
| `dataset_import`, `dataset_import_undo`, `dataset_upload_extract`, `combine` | R9 |
| `reprocess`, `assist_run`, `selection`, `open_vocab_drain`, `vlm_label_batch`, `vlm_verify_region_batch`, `autolabel`, `assist_loop` | R10 (Rust) |
| `item_scores` | R10 per D5 (M13) |
| `persons_cluster`, `near_dup_core_group` | R2b per D5 (M13) |
| `image_counts_backfill` | deleted in R1c |
| `cluster_projection_sync`, `class_projection_sync`, `near_dup_projection_sync` | never ported; deleted with R8-1, R5 and R6b |
| `embedding_viz` | stays Python (UMAP; M14 informs the owner) |
| `train`, `model_export`, `bakeoff`, `probe`, `promote` | stay Python |
| `sum_compact` (new in plan 15's design) | R1b |
| the erasure job (plan 15's design) | R5e |

### 7.3 Ticks (17, plus two new)

`reaper`, `ingest_dispatch`, `open_vocab_sweep`, `upload_sweep`, `job_stats`,
`vlm_batch_staging_sweep`, `serving_inventory`, `reload_promoted`, `cache_prune`,
`resource_reachability`, `cluster_refresh`, `db_backup`, `gpu_arbiter`, `near_dup_rescore`,
`near_dup_core_group`, `near_dup_full_build`: R3a (`cluster_refresh` queues the Python `cluster_run` until R8-1 switches it to the Rust kinds). `image_counts_drain`: deleted in R1c. New:
`sum_compact` and the nightly drift check (R1b); the unreferenced-object part of `upload_sweep` (R1c).

### 7.4 Processes

| Service | Slice or owner |
|---|---|
| `api` | R1a (Rust front); the Python API as `api-py` until R12 |
| `db-migrate` | Alembic until R12, then `op-server migrate` |
| `frame-engine` | plan 17 |
| `worker`, `periodic` (new, `op-server`) | R3a |
| `jobs-worker`, `cluster-worker`, `model-export-worker`, `curation-trainer`, `curation-evaluator`, `gpu-arbiter-worker` | claim loop from R3a; kinds leave as their slices port them; the stay list remains |
| `interactive-worker` | claim loop in R3a; goes in R5 |
| `open-vocab-worker`, `curation-auto-label-worker`, `curation-assist-loop-worker` | claim loop in R3a; go in R10 |
| `curation-vlm-worker` (not Celery) | R10 (a Rust worker role) |
| `curation-detection-worker` (not Celery) | R7 (a Rust worker role) |
| `jobs-beat`, `periodic-worker`, `flower` | removed in R3a |
| `valkey` | removed in R3b |
| `opensearch` and Dashboards | R2b when D4 says no |
| `segmenter`, `vlm`, `triton-server`, `postgres`, monitoring | unchanged |

### 7.5 Mechanisms (inventory section 7)

| Mechanism | Slice |
|---|---|
| Pydantic models and FastAPI OpenAPI | hand-written serde types checked against the frozen contract file per family (contract-first, decision 14); the check pattern in R1a; the FastAPI generator stops writing a family when Rust owns it and is deleted in R12 |
| FastAPI dependencies | axum extractors (`ProjectTx`, settings, pools) from R1a and R1j |
| Middleware, CORS, correlation id, origin | R1a |
| Error handlers and the 503 mappings | R1a (module), each family for its errors |
| Thread pools and `contextvars` copying | tokio runtime and the rayon pool; the bound project travels in `ProjectTx` |
| asyncio structure (`run_all`, stage workers, committer) | R1c pipeline |
| Multi-process Prometheus | gone (one process); names kept from R1a |
| Structured logging | tracing, R1a |
| Retired-variable guard | R1a (generated list); source moves in R12 |
| Config store revisions and `ETag` | R11 |
| Project guard and binding | 18a section 4; R2b deletes the OpenSearch guard |
| OCC helpers (`clients/occ*.py`) | never ported; deleted by last importer (R2b at the latest) |
| Server-sent events | R3b |
| Upload limits and the disk guard | R1u (curation upload), R2b (core ingest uploads) |
| Wire serialisation (`WireRoute`, orjson, NaN as null) | each family; NaN as `null` checked by a test on serde_json's output |
| Job runtime (`src/tasks/runtime/`) | R1j (Rust core), R3a (claim loop, ticks, reaper) |
| uvicorn settings | R1a |

### 7.6 Items the review found unassigned, now assigned

| Item | Slice |
|---|---|
| Erasure job, `privacy_purges`, `privacy_audit` | R5e |
| `sum_compact`, its tick and the nightly drift check | R1b |
| `DISK_GUARD` and `src/core/disk_space.py` | R1u (curation upload), R2b (core ingest uploads) |
| Installer, `openprocessor` CLI, `scripts/lib/*.sh` health probes, `tests/installer`, `tests/compose` | the slice that changes a service: R1a, R3a, R3b, R2b, R12 (18a section 2.3) |
| `scripts/research`, `scripts/validation`, the `scripts/curation` and `scripts/bench` files importing `src` | the last-importer check of each slice; R12's list |
| The end state of `src/` and its layering rule | R12 |
| `/curation/projects` list for several users | non-goal, 18 section 6.1 row 2 |
| `curation-cluster-refresh` named in `CLAUDE.md` but absent from compose | R3a corrects `CLAUDE.md` |
