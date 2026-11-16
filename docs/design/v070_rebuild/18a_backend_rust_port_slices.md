# 18a: the backend in Rust, the slices

| Field | Value |
|---|---|
| Status | plan, revision 0 (2026-11-15), companion of [18](18_backend_rust_port_plan.md). Decisions, order, measurements and the cloud benchmark live there; this file is what a lane executes |
| Reads with | [17](17_native_engine_plan.md) sections 1.1 and 3.1 (the engine protocol and the Rust gates); 15c on `int/storage` (the storage slices each Rust slice follows, its section 8 "module deletion by last importer" method and its section 2 contract diff) |

## 1. How every slice runs

### 1.1 The slice template

Every slice below fills these rows. A lane brief copies its slice's rows, the gates of 18 section 7,
and the lane checklist of the project; it cites the documentation pages read for every crate it
touches (18 section 2.2 lists the first set).

| Row | Content |
|---|---|
| Goal | one sentence for the person who uses OpenProcessor |
| Entry | merged storage slices, engine phases and measurements, by id |
| Replaces | Python routes (by contract operation), modules, job kinds, processes |
| Contract | the routes kept byte for byte in behaviour; the intended changes, each with its frontend edit; the generated files touched |
| Selective checks (the lane runs them) | the slice's own Rust unit tests, each seen failing first; its case-contract replays; contract drift and oasdiff; `cargo` gates and hooks on touched files; `npm run check` when a frontend file changes. No database, compose, live, browser or GPU run |
| Owed to the wave sweep | the slice's database-tier tests, its parity replay, the live and browser tiers, one targeted stack or GPU run where the slice needs it (named), its measurement rows |
| Go bar | the measured bar that keeps the slice; a miss goes back to one fix lane with its number |
| Deletes | the Python modules whose last importer the slice removes (15c section 8 method: `rg -l` at entry, listed in the brief); tests of deleted modules with the Rust tests or cases that replace them |

The full sweep (offline suite, database tier, live tier, browser tier, `--all-files` hooks, the
benchmark regression run) runs once per wave of merged slices, never per slice (owner, 2026-11-15:
no long test runs for their own sake).

### 1.2 What happens to the Python tests

| Kind of test | Rule |
|---|---|
| Pure domain functions (`src/domain/`, the fold, geometry, wire rules, filter grammar) | kept as the reference while the Python module lives. The slice that ports a module adds a generator under `scripts/codegen/` that writes `contracts/json/<module>_cases.json` from the Python functions (as `frame_engine_plan_cases.json` is made today), with a drift hook; Rust tests replay it. When the Python module is deleted, its last generated file is frozen under `native/<crate>/tests/goldens/` and its Python tests go with it |
| Route tests with fakes (the `_curation_app.py` fixtures) | replaced by the family's parity pairs (section 1.3) and Rust handler tests; deleted with the Python route |
| Service tests with fakes | become Rust unit tests of the ported logic; deleted with the module |
| Database-tier tests of a Python repository | ported to Rust tests behind a cargo feature `db` (testcontainers-modules, PostgreSQL) by the slice that ports the repository; schema and migration tests stay Python until R12 (Alembic owns the schema until then) |
| Live tier (`tests/live/`) | unchanged; black-box over HTTP; must pass against the Rust front at every wave. A slice edits a live test only for an intended change it lists |
| Browser tier and vitest (`frontend/`) | unchanged except a listed frontend edit |
| Codegen and ratchet tests | extended to `native/` (layering by crate, file size, naming leaks) |
| Tests of Python workers that stay | unchanged |

The offline suite shrinks only by tests of deleted modules. Each slice's last commit lists the
removed test ids and what replaces each.

### 1.3 Parity before deletion

1. **Recorded pairs.** The Rust front (R1a) forwards every unported path to the Python API. Built
   with the cargo feature `parity-record`, it writes each forwarded request and response (method,
   path, query, body, status, chosen headers, response body) to `tests/parity/<family>/` while the
   live tier runs on a harness seeded by `scripts/curation/seed_live_harness.py`. The wave sweep
   before a family's slice records its pairs; the coordinator commits them; the lane gets them as
   files.
2. **Replay.** A Rust test (feature `parity`, owed to the wave) replays the pairs against the Rust
   router on a harness seeded the same way and compares parsed JSON after the family's
   normalisation list (generated ids, timestamps, durations, ordering of sets). An empty diff, or a
   diff inside the slice's intended changes, is required before the Python handler's deletion
   merges.
3. **Live and browser tiers** pass unchanged with Rust serving the family.
4. **Ground truth** where an output changes on purpose (R2a: the engine's preprocessing replaces
   the API's): the quality rows named in the slice, not bits.
5. **Cost.** p50 and p99 at 8 clients and API host CPU per request not above the Python route's
   (measured in the wave), and lower where the slice claims a gain.

Allowed differences, everywhere: the `msg` text of validation errors (not their `loc` or `type`,
and never "String should match pattern '...'"); the OpenAPI cosmetics of 18 section 2.4; float
spelling (parsed values compared); key order; header case; the `server` header.

### 1.4 Fail-closed rules that must survive

| Rule today | In Rust | Proof |
|---|---|---|
| Row-level security through the transaction (`src/db/session.py` `project_tx`, role `op_app` without `BYPASSRLS`) | `ProjectTx`: repositories of project tables take `&mut ProjectTx`, made only by `project_tx(pool, project)` | the cross-project leak cases of `tests/curation/cross_project_leak/` ported to the `db` feature; a repository call without a scope does not compile |
| The project guard on OpenSearch (`src/services/projects/guard*.py`) | deleted with the curation indexes (15c S9) and the core indexes (R2b); if D4 keeps OpenSearch, a guarded client type built only from a bound project | the guard's 24 test files' cases as Rust tests, or deleted with OpenSearch |
| The lock rule (a machine writer never overwrites a human or import decision) | claims are inserted only by `op_store::human` (module privacy); machine writers have no claim function; the fold decides state | `fold_cases.json`, the fold property test, the lock-rule cases of 15a section 6 |
| Unbound or unknown project | 404 as today, from the project extractor | parity pairs |
| Retired environment variables | start refusal from a generated `contracts/json/retired_env.json` (source `src/config/retired_env.py` until R12) in every role | one test per variable: the role exits non-zero naming it |
| The vector DDL role `op_vector` only in the embedding-space job | a separate pool used only by that job (R6) | the database test of 15c S1a-ii, ported |
| Upload and batch limits (`src/core/upload_limits.py`, `CurationConfig`) | the same served limits; tower-http body limit at the byte cap | request-edge tests |
| Paths outside the configured roots | storage keys resolved under cap-std root handles; `..`, absolute keys and unknown roots refused before any open | property test |

## 2. The slices

### R0. Measurements

**Goal.** The numbers that fix the order and the bars exist before code. **Entry.** None. **What.**
18 section 8: M1 to M5, M8, M10 now; M6, M7, M9, M11 before their slices. **Deletes.** Nothing.
**Owed.** Coordinator holds; no lane.

### RP. The engine reads the photo by key, Python API side (conditional)

**Goal.** Large photos stop being copied through the Python API: it sends the engine a storage key.
**Entry.** D1 branch A (18 section 3.4); N3 merged; 15c slice F merged (storage keys).
**Replaces.** In the engine (`native/frame_engine`): a new input `source` (BYTES [1], one JSON
object: root, key, size, imohash), exactly one of `image` and `source`; key resolution under the
storage roots slice F defines, through cap-std; new photo statuses `source_missing` and
`source_changed` (the file's size differs from the claim) and plan faults for an unknown root or an
invalid key, all written into the generator of `contracts/json/frame_engine_v1.json` and the plan
cases. In the API: `src/clients/frame_engine.py` sends `source`; `stage_engine` and `read_files`
stop reading whole files for server-side paths and uploads (the upload already persisted);
`content_imohash` reads the file; capture through `read_capture_file`. **Contract.** No curation
route changes; the engine contract changes as above. **Selective checks.** Rust front tests for the
new input and every refusal; the Python client tests (`tests/clients/test_frame_engine_*.py`)
updated; contract drift. **Owed.** The four-way ingest rows and M1's rows on the engine, one GPU
hold; the live ingest tier. **Go bar.** API CPU per photo at 20 MP down by at least M3's measured
share, photos/s not lower anywhere. **Deletes.** `src/utils/file_read.py` if its last importer
goes; the pipe feed of `src/clients/exiftool_client.py` if unused. R1 rewrites the Python half.

### R1a. The Rust server front

**Goal.** The Rust server answers on port 4603 and passes every path it does not yet own to the
Python API, unchanged for the person. **Entry.** R0's M8; merges in the same wave as R1b and R1c,
never alone. **Builds.** Crates `op_api`, `op_server`, `op_contracts` (18 section 7.1); the
`op-server api` role: axum router, tower-http request id on `x-request-id`, CORS from the settings
`src/core/cors.py` reads, the HTTP metrics names of `src/app/factory.py`, structured logs, a
concurrency limit with load shedding at today's 512 (uvicorn's `--limit-concurrency`) answering 503,
keep-alive 75 s, graceful shutdown 30 s, the retired-variable refusal, the error-shape module
(18 section 2.4). A `fallback` that forwards to the Python API over hyper-util with streaming bodies
both ways (server-sent events included), forwarding `x-request-id`, `x-forwarded-for`,
`x-forwarded-proto` and `host` so `src/core/request_origin.py` sees what it sees today (read at
entry). `/live` is served by Rust (container health); every other path forwards. Compose: service
`api` runs `op-server api`; the Python API becomes `api-py` on internal port 4613 (not published),
reached through the new variable `OP_PY_API_URL`. The contract merge in
`scripts/codegen/export_api_contracts.py`, the oasdiff hook and the intended-change file
`scripts/codegen/contract_intended_changes.txt`. **Selective checks.** Router tests with
`tower::ServiceExt::oneshot`: forwarding of method, body, headers and an event stream against a stub
upstream; error shapes; load shedding; retired-variable refusal; contract export. **Owed.** The
whole live and browser tiers through the front (they prove the forwarder); a p99 comparison of
forwarded routes against direct. **Go bar.** Live and browser tiers green through the front;
forwarded p99 at 8 clients within the spread of direct plus 1 ms. **Deletes.** Nothing (the
forwarder itself is deleted in R12).

### R1b. The record and the jobs core in Rust

**Goal.** Rust can write a photo's record and run jobs on the same tables the Python code uses.
**Entry.** Wave E merged (18 section 3.2); M8. **Builds.** `op_domain`: ports of
`src/domain/fold*.py`, `claim_match.py`, `claim*.py` as far as ingest and attach need, `results.py`,
and the tallies the summary writer emits, with the generator `scripts/codegen/fold_cases.py`
writing `contracts/json/fold_cases.json` (recorded differential cases plus property-generated
histories) and its drift hook; the fold property test in proptest. `op_store`: the pool (the same
`OP_*` database settings `src/db/engine.py` reads), `ProjectTx`, the statement registry, the
repositories ingest needs (`photos` with the imohash locks, `items`, `results`, `state` with the
delta emission of 15b section 6.2, `vectors` per space, `analysers` and the ledger, `events` insert
with the Valkey publish after commit on today's channel names) and `src/services/record/apply.py`
and `attach.py`. `op_jobs`: the claim, lease, heartbeat and terminal-state protocol of
`src/tasks/runtime/` on the `jobs` table; kinds from a generated `contracts/json/job_kinds.json`
(source `src/config/job_kinds/`), which gains a field `runner` (`python` or `rust`); wake-up by
`LISTEN` on a channel `op_jobs` plus a 2 s poll of queued Rust kinds; the role `op-server worker`;
kinds `noop` and `sleep_noop`. Python side: `dispatch()` sends no Celery message for a Rust kind;
the reaper's rule c (`src/db/repos/jobs_reaper.py:150`) skips Rust kinds; a new process
`jobs-dispatch` (`src/tasks/dispatcher.py`, compose service of the same name) listens on
`op_jobs_python` and calls `dispatch()` for each queued Python-kind row not yet dispatched; Rust
writes that `NOTIFY` in the transaction that queues the row. **Contract.** No route. **Selective
checks.** Fold case replay; property tests; repository unit tests over the statement text; runner
state-machine tests with a fake clock; job-kind and retired-variable generators' drift. **Owed.**
The `db` feature tests: ported `tests/db/test_ingest_record.py`, `test_attach.py`,
`test_vector_space_ddl.py`, the jobs claim and reaper tests, and one conformance suite run against
both runtimes (claim race, lease expiry, cancel, exclusive group, pause); the statement-prepare
test; `test_job_load.py` and `test_queue_delivery.py` live. **Go bar.** All owed tests green; a Rust
`noop` job and a Python job queued by Rust both finish in the live harness. **Deletes.** The Python
runtime for Rust kinds only (their Celery task registrations).

### R1c. Ingest on Rust, uploads included

**Goal.** A person's photos, uploaded from the browser or read from a folder on the server, are
processed at the card's rate without the API holding or copying whole photos. **Entry.** R1a, R1b;
N3 merged (N1b and N1c for the benchmark plan); M1 to M5 and M10 recorded; D1 and D2 decided.
**Replaces.** The ingest operations of the contract whose handlers are `src/routers/curation/
ingest.py` (`image`, `batch`), `ingest_upload.py` (`upload`), `ingest_objects.py` (`objects`),
`ingest_bulk.py`, `ingest_sources.py`, and the `path_lookup` and `status` reads (the lane lists the
operation ids from the contract at entry; `config`, `policy` and `region_drain` stay with R11 and
R7); job kinds `bulk_ingest_run`, `source_poll`, `source_backfill`. The pipeline of inventory
section 6 in `op_ingest`: admission (photos and bytes), `imohash` from the file, the content claim,
the engine call with `source` (the engine half of RP if RP did not run), capture through a Rust
ExifTool `-stay_open` pool by path (or D2's other branch), the commit group (attach, `apply()`,
`store_states`, vectors, ledger, `crop.created` events), the crop cache write, the open-vocabulary
job row after ingest (a Python kind, through `jobs-dispatch`), the per-photo result values and the
503 `Retry-After` when the engine is down for every new photo. Uploads as 18 section 4: streamed
parts to the landing root, processing overlapping the body, back pressure by not polling the body.
**Contract.** Unchanged routes and responses; intended: none on the wire. Frontend edit:
`frontend/src/lib/api_ingest.ts` appends `source` and `image_paths` before the `images` parts;
`scripts/curation/ingest_upload.py` the same. `import_labeled_dataset.py` and the walkers already
use HTTP. The dataset-import job's image stage (`src/services/curation/dataset_import/`) calls `POST
/curation/projects/{project}/ingest/batch` over HTTP until R9 ports it: the one stated piece of
Python this plan rewrites besides RP. **Selective checks.** Pipeline unit tests with fake engine,
store and capture traits; `imohash` goldens; the streaming upload under a slow body and a full
admission (in-process); the capture driver against a fake ExifTool; the frontend `npm run check`.
**Owed.** Parity replay of the ingest family; `tests/live/test_live_ingest.py`,
`test_live_whole_photo.py`, `test_live_bulk_restart.py` and the rest of 15c's S1a-ii list; the
four-way ingest rows and the M10 upload rows on the native engine (one GPU hold, `coco2k` and
`setp200`, one card and two); the benchmark regression run of 18 section 5.2; the 600 s soak with
failed 0. **Go bar.** API host CPU per photo at most 0.5x the Python route's (M1, M10) on both lists,
server-side and upload; photos/s not below Python's at any client count by more than the spread and
above it by more than the spread in the host-bound row (20 MP, plan F on one card or plan C on two);
failed 0 (roadmap 14 section 4: kept only if it wins by more than the spread). **Deletes.** By last
importer: the ported routers; `src/services/curation/ingest_batch.py`, `ingest_finish.py`,
`ingest_stage.py`, the commit part of `ingest_index.py`, `upload_ingest.py`, `bulk_ingest/`,
`capture_read.py` and `src/clients/exiftool_client.py` once `source_backfill` is Rust,
`src/utils/file_read.py`; bench scripts that import the Python ingest service are deleted when they
measure the Python path, or moved to HTTP when they drive the API. Modules still imported by
reprocess or open vocabulary (R10) stay until then.

### R2a. Inference routes on the engine

**Goal.** `/detect`, `/faces`, `/embed`, `/ocr` and `/analyze` answer from the native engine with
no Python preprocessing. **Entry.** N3 merged; M6. **Replaces.** `src/routers/detect.py`,
`analyze.py`, `ocr.py`, `embed.py`, the detect, recognize and verify routes of `faces.py`,
`models.py` (model test), and their `/v1` twins (kept as twins); `src/clients/triton_preprocess.py`,
`src/utils/image_processing.py`, `cpu_preprocess.py`, `face_align.py`, `scrfd_decode.py`, the text
tower client `src/clients/pe_encoder.py` and `pe_text_backends.py` (text tokens through tokenizers,
checked token for token), the API's use of `transformers` in `src/utils/cache.py` and of torch in
`src/services/detection/ensemble_nms.py` (M8 says whether they load today). Uploads stream as in
R1c and go to the engine as bytes (`image` input), since these routes store nothing.
**Contract.** Response shapes unchanged (CLAUDE.md "Core API shape"). Intended: boxes, scores and
vectors follow the engine's preprocessing instead of the API's cv2 path. **Selective checks.**
Handler tests on fake engine answers; token goldens. **Owed.** Ground truth on the same photos,
Python route against Rust route: detector mAP on COCO val, LFW, CALFW and CPLFW verification for
faces, COCO-Text F1 for text, each not below the Python route by more than its two-run difference;
M6 rerun. **Go bar.** Those rows, and API CPU per request below M6's. **Deletes.** The listed Python
modules.

### R2b. Core data on the record; OpenSearch leaves

**Goal.** The core library (search by picture or text, faces, persons, clusters, duplicates) lives
in the record like every project, and the stack has one database. **Entry.** 15c wave G merged; N3;
R1c; D4 decided. **Replaces.** Core `src/routers/ingest/` (batch, single, directory, status, cancel),
`src/routers/search/`, the search and identify routes of `faces.py`, `persons.py`, `clusters.py`,
`query.py`, `query_duplicates.py`, `jobs.py` (deployment jobs read); job `directory_ingest`; the
five global indexes of `src/clients/opensearch/names.py`, which become the reserved core project's
record tables (15 section 14 answer 1; faces and text lines as 15a designs them); core ingest runs
R1c's pipeline bound to the core project. `persons_cluster` and `near_dup_core_group` stay Python
and read the core project's tables. **Contract.** Response shapes unchanged; intended: search
scores and tie order follow the record's vector index. **Selective checks.** Handler and repository
unit tests; the erasure catalogue extended to the core tables. **Owed.** The core live tests
(`test_live_core_ingest.py` and the core search tests) unchanged; search recall at 10 against exact
search on the core data not below the OpenSearch route's; the face and text purge flows through
erasure. **Go bar.** Those rows; one store in the stack when D4 says no. **Deletes.**
`src/services/visual_search/`, the global-index parts of `src/services/face_identity/`, what remains
of `src/clients/opensearch/` (`scroll.py`, `clusters.py`, `names.py`), `src/db/repos/image_contents.py`
and its model (15c B1), `scripts/purge_*_data.py` (replaced by erasure), and when D4 says no: the
OpenSearch compose service and Dashboards, its health, metrics and error mapping in `src/core/`, its
install and heap steps in `scripts/lib/` and `scripts/setup.sh`, the benchmark scripts that target
it, and its `OP_*` variables into the retired list.

### R3a. Jobs and periodic ticks on Rust

**Goal.** Every Rust job kind and every tick runs in Rust; Python keeps only its own workers.
**Entry.** R1c merged. **Replaces.** `jobs-beat` (Celery beat, `src/tasks/beat_schedule.py`,
`beat_lock.py`) by `op-server periodic` with tokio intervals and the same PostgreSQL advisory lock
key; the reaper (all rules, for both runtimes; rule c sends `NOTIFY` for Python kinds); the ticks
`ingest_dispatch`, `open_vocab_sweep`, `upload_sweep`, `job_stats`, `vlm_batch_staging_sweep`,
`serving_inventory`, `reload_promoted`, `cache_prune`, `resource_reachability`, `cluster_refresh`,
`db_backup`, `near_dup_rescore`, `near_dup_core_group`, `near_dup_full_build` (those that start
Python work write a Python-kind row); `gpu_arbiter` becomes a Python job kind started by the tick,
consumed by the arbiter worker of the opt-in overlay; the route `GET /curation/jobs/{job_id}`.
**Selective checks.** Tick unit tests with a fake clock; the single-tick lock under two periodic
processes (fake). **Owed.** `test_job_load.py`, `test_queue_delivery.py`,
`test_job_recovery_real_handlers.py` live; the conformance suite. **Go bar.** Owed tests green; no
tick missed over a 600 s run. **Deletes.** `src/tasks/beat_schedule.py`, `beat_lock.py`,
`src/tasks/periodic.py`, `trigger.py`, `reaper.py` and the services of `src/services/periodic/` they
alone import; the `periodic-worker` and `jobs-beat` services.

### R3b. Live events from Rust

**Goal.** The progress a person watches streams from the Rust server. **Entry.** R1c merged.
**Replaces.** The three event-stream operations, `events/publish` and `events/stats`, the pipeline
event and global status routes (`src/routers/curation/events.py`, `pipeline_events.py`,
`global_status.py`), `src/services/events/sse.py` and `listener.py`: axum `Sse` with keep-alive,
one Valkey subscription per process for wake-ups plus a 2 s tick, the committed-safe cursor
`(xid, id)` of `src/services/events/cursor.py` as `Last-Event-ID`, a slow client dropped with
`reconnect` as today. Python producers keep `src/services/events/publish.py`. **Selective checks.**
Cursor codec goldens; stream tests with a fake bus. **Owed.** `test_sse_across_replicas.py` live; the
frontend SSE tests. **Go bar.** Owed green; event latency (commit to client) p99 not above Python's.
**Deletes.** The replaced routers and modules.

### R4 to R11. Route families after wave G

Entry for each: 15c wave G merged; R1a; R3a; the family's M7 row; its parity pairs recorded in the
previous wave. Default order below; M7 reorders by API CPU per request times request rate in the
acceptance walk. Each family slice may split into lanes by concern (at most three per wave). Each
deletes its routers, the services they alone import, and the schemas they alone use.

| Slice | Plain name | Contract operations (inventory section 1.3) | Storage slice it follows | Jobs and workers | Notes |
|---|---|---|---|---|---|
| R4 | Gallery reads: objects, photos, crops, thumbnails, map, table | crops 33, images 8, map and map cells, table, viz 4 | S1b-i, S1b-ii, F | none | thumbnails and crops served as files from the cache by key (15 section 8); any remaining encode in the rayon pool with turbojpeg as the engine uses; benchmark B4 reads |
| R5 | Labelling acts, review, classes, counts | classes 9, review 7, label_ops 4, select 4, scores 4, stats 3 | S2a, S2b | `label_op`, `label_undo`, `class_merge`, `review_resolve_batch`, `edit_undo_batch`; the `interactive-worker` service goes | the human-act service (`src/services/record/human.py`) ported; `class_projection_sync` ported here only if 15c did not delete it |
| R6 | Search, vectors, near-duplicates | search 3, embedding 8, near_duplicates 9 | S3, S5 | `embedding_reindex`; `near_dup_projection_sync` only if 15c kept it; the four `near_dup_*` scoring kinds stay Python | the vector-index interface of S3; the `op_vector` pool |
| R7 | Regions, boxes and the region stage | regions 9, region_stage 3, ingest `region_drain` | S4 | the `curation-detection-worker` (`scripts/curation/worker.py`) becomes an `op-server` worker role, calling Triton, the segmenter and the VLM over tonic and reqwest | M9 first |
| R8 | Cluster screens | clusters 8 | S6 | `cluster_run` stays Python | cards from the partition-keyed summaries |
| R9 | Import, combine, export, training | datasets 11, export 7, combine 5, train 18 | S7 | `dataset_import`, `dataset_import_undo`, `dataset_upload_extract`, `combine`; `train` and `promote` stay Python | export and training artifacts as streamed files; the HTTP call of R1c's dataset-import stage is replaced by an in-process call |
| R10 | Machine helpers and re-runs | assist 15, assist_loop 15, open_vocab 15, pipeline 8, reprocess 3, probe | S8a, S8b, S8c | `reprocess`, `assist_run`, `selection`, `open_vocab_drain`, `vlm_label_batch`, `vlm_verify_region_batch`, the VLM worker loop; `autolabel` and `assist_loop` by the rule of 18 section 2.3; `probe` stays Python; the `open-vocab-worker` and `curation-vlm-worker` services go | after this slice no Python code folds: the Python fold, `fold_cases.json`'s generator and the Python attach are deleted |
| R11 | Projects, configuration documents, settings, models and the rest | `/curation/projects` list, create, delete, combine; region_profiles 17, prompt_packs 14, ingest_profiles 14, cluster_profiles 7, near_dup_profiles 7, assist_loop_presets 7, serving_profiles 6; `/curation/vlm/*` 15; ingest `config` and `policy`; keymap, location_policy, settings, pause, bakeoff 9, `/curation/health`, `/curation/embedding/models`; core `health.py`, `api_docs.py`, `models_lifecycle.py`, `models_export.py` | S9 | `project_create`, `project_delete`; `bakeoff` and `model_export` stay Python | config documents keep their `ETag` headers; `src/services/config_store/` ported with its revision rules; the keymap and class contracts' sources move to Rust (files stay byte-identical) |

Go bar for each: parity replay empty outside intended changes; the family's live and browser tests
green; p50 and p99 at 8 clients and API CPU per request not above Python's. Selective checks: the
family's Rust handler tests, case replays and contract checks. Owed: parity replay, live and browser
tiers, the M7 row.

### R12. The Python API removed

**Goal.** One backend language on every request path. **Entry.** R1 to R11 merged. **Deletes.**
`src/main.py`, `src/app/`, the FastAPI parts of `src/core/`, `src/routers/`, `src/schemas/` and the
Pydantic settings only the API used, FastAPI and uvicorn from the requirements, the `api-py`
service and `OP_PY_API_URL`, the forwarder and the `parity-record` feature (the pairs stay as Rust
regression goldens), the Python half of the contract generator (`op_contracts` writes every file;
each file byte-identical to its last Python output except recorded intended changes). The schema
leaves Alembic: a baseline SQL migration taken from the head schema (wipe and re-ingest allowed) run
by `op-server migrate` in the `db-migrate` service; the migration library (refinery or sqlx's
migrator over tokio-postgres) is chosen in this slice from their documentation. The retired-variable
list's source moves to Rust; `scripts/lib/retired_env.sh` reads the generated file. The Python
worker image keeps only the workers of 18 section 2.3. The final review: Q2 (Celery for the Python
workers) and the numeric jobs, by measurement. Docs: `CLAUDE.md`, `docs/ARCHITECTURE.md`, the docs
site's architecture and deploy pages (concise user page, detailed developer page).

## 3. Traceability

Every area of the port inventory, to the slice that ports it, keeps it in Python, or the plan that
deletes it.

### 3.1 Routes

| Area (inventory 1.2, 1.3) | Slice |
|---|---|
| Core: `detect.py`, `analyze.py`, `ocr.py`, `embed.py`, `models.py`, faces detect, recognize, verify | R2a |
| Core: `ingest/`, `search/`, faces search and identify, `persons.py`, `clusters.py`, `query.py`, `query_duplicates.py`, `jobs.py` | R2b |
| Core: `health.py` (`/live` in R1a), `api_docs.py`, `models_lifecycle.py`, `models_export.py` | R11 |
| `/v1` twins (`src/routers/v1.py`) | with each core route |
| Curation ingest (27): hot routes, bulk, sources, path lookup, status | R1c |
| Curation ingest: config, policy | R11; region_drain: R7 |
| crops, images, map, table, viz | R4 |
| classes, review, label_ops, select, scores, stats | R5 |
| search, embedding, near_duplicates | R6 |
| regions, region_stage | R7 |
| clusters | R8 |
| datasets, export, combine, train | R9 |
| assist, assist_loop, open_vocab, pipeline, reprocess, probe | R10 |
| config documents (seven profile and preset families), projects, VLM registry, keymap, location_policy, settings, pause, bakeoff, curation health, embedding models | R11 |
| events (3 streams, publish, stats), pipeline events, global status | R3b |
| `/curation/jobs/{job_id}` | R3a |
| `POST /image_counts/backfill` | deleted by 15c S1a-ii |

### 3.2 Job kinds (44)

| Kinds | Slice or owner |
|---|---|
| `noop`, `sleep_noop` | R1b |
| `bulk_ingest_run`, `source_poll`, `source_backfill` | R1c |
| `directory_ingest` | R2b |
| `label_op`, `label_undo`, `class_merge`, `review_resolve_batch`, `edit_undo_batch` | R5 |
| `embedding_reindex` | R6 |
| `dataset_import`, `dataset_import_undo`, `dataset_upload_extract`, `combine` | R9 |
| `reprocess`, `assist_run`, `selection`, `open_vocab_drain`, `vlm_label_batch`, `vlm_verify_region_batch` | R10 |
| `autolabel`, `assist_loop` | R10 by the rule of 18 section 2.3 (Rust if equivalents exist, else Python writing through the server) |
| `project_create`, `project_delete` | R11 |
| `image_counts_backfill` | deleted by 15c S1a-ii |
| `cluster_projection_sync` | deleted by 15c S6 |
| `class_projection_sync`, `near_dup_projection_sync` | deleted by 15c (plan 15 section 9: the fenced projections go); if one survives wave G, R5 or R6 ports it |
| `cluster_run`, `item_scores`, `embedding_viz`, `region_fp_centroids`, `persons_cluster`, `near_dup_score`, `near_dup_group`, `near_dup_rescore`, `near_dup_core_group` | stay Python (plan 16 owns the logic) |
| `train`, `model_export`, `bakeoff`, `probe`, `promote` | stay Python |

Ticks (17): `image_counts_drain` deleted by 15c S1a-ii; the other 16 in R3a (`gpu_arbiter` as a
Python job kind started by the Rust tick).

### 3.3 Processes (inventory 3.1)

| Service | Slice or owner |
|---|---|
| `api` | R1a (Rust front); the Python API as `api-py` until R12 |
| `db-migrate` | R12 (`op-server migrate`) |
| `frame-engine` | plan 17 |
| `jobs-worker` | its Rust kinds move to `op-server worker` (R1b, R1c, R2b, R5, R6, R9, R10, R11); a Python worker keeps the kinds of 3.2 that stay |
| `interactive-worker` | R5 |
| `periodic-worker`, `jobs-beat` | R3a |
| `open-vocab-worker`, `curation-vlm-worker` | R10 |
| `curation-auto-label-worker`, `curation-assist-loop-worker` | R10 rule |
| `curation-detection-worker` | R7 |
| `cluster-worker`, `model-export-worker`, `curation-trainer`, `curation-evaluator`, `segmenter`, `vlm`, `gpu-arbiter-worker` | stay Python |
| `flower` | stays while Celery serves the Python workers (Q2) |
| `jobs-dispatch` (new) | R1b; reviewed in R12 |
| `opensearch` and Dashboards | R2b when D4 says no |
| `triton-server`, `postgres`, `valkey`, monitoring | unchanged |

### 3.4 Mechanisms (inventory section 7)

| Mechanism | Slice |
|---|---|
| Pydantic models and FastAPI OpenAPI | serde, garde and utoipa per family; generator merge in R1a; Python half deleted in R12 |
| FastAPI dependencies | axum extractors (`ProjectTx`, settings, pools) from R1a and R1b |
| Middleware, CORS, correlation id, origin | R1a |
| Error handlers and the 503 mappings | R1a (module), each family for its errors |
| Thread pools and `contextvars` copying | tokio runtime and the rayon pool (R1a); the bound project travels in `ProjectTx`, not a context variable |
| asyncio structure (`run_all`, stage workers, committer) | R1c pipeline |
| Multi-process Prometheus | gone (one process); names kept, R1a onward |
| Structured logging | tracing, R1a |
| Retired-variable guard | R1a (generated list); source flips in R12 |
| Config store revisions and `ETag` | R11 |
| Project guard and binding | 1.4 above; R2b and 15c S9 delete the OpenSearch guard |
| OCC helpers (`clients/occ*.py`) | deleted by 15c S9, never ported |
| Server-sent events | R3b |
| Upload limits | R1a (body limit), R1c (counts and bytes) |
| Wire serialisation (`WireRoute`, orjson, NaN as null) | each family; NaN as `null` checked by a test on serde_json's output |
| Job runtime (`src/tasks/runtime/`) | R1b (core), R3a (ticks, reaper) |
| uvicorn settings | R1a |
