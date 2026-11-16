# 18: the backend in Rust, ordered by measured gain

| Field | Value |
|---|---|
| Status | plan, revision 0 (2026-11-15), not reviewed. Covers roadmap gaps G5 (core routes on the engine and the record) and G14 (deploy, the parts this port changes) of [14](14_roadmap.md) |
| Companion | [18a_backend_rust_port_slices.md](18a_backend_rust_port_slices.md): the slice template, the test and parity rules, every slice, and the traceability table. This file holds the decisions, the order, the measurements, the cloud benchmark, deploy, gates, risks and questions |
| Code base read | `int/b4-stateless` at `7614a8648` (this worktree); `int/storage` at `0d9412024` (15b, 15c); `int/native` (the `native/` workspace of plan 17). Facts about the API come from the port inventory of 2026-11-15 (gitignored, `artifacts_local/speed_1113/api_port/inventory.md`); its "inferred" rows are used here only where re-read in code or marked as hypotheses |
| Numbers | every number names its source. "API capacity rows" = the partial grid of 2026-11-15 (gitignored `artifacts_local/speed_1113/api_capacity/results.md`): 8 API processes, the Rust engine as a stub returning 2 objects per photo, server-side path ingest. Estimates are marked "estimate" |
| Not in scope | the engine's internals (plan 17), the storage schema and its slices (plans 15, 15a, 15b, 15c), the cluster, label and train logic (plan 16), the gallery's design (photo-library plans), authentication |

## 0. The user's job, and what the port is for

A person points OpenProcessor at a folder of photos, or uploads them from the browser: thousands to
millions, small web photos up to 20 MP camera photos. The detector finds the objects; every photo
and object gets vectors, text and faces; the person finds photos, labels their own classes by
cluster and trains a detector; a gallery (timeline, map, people, events) sits on the same record.

The owner's objective: "the whole point is to process this image data set as fast as possible and
efficiently and make it work for other tasks as well for images". The whole backend ends in Rust,
so no interpreter lock and no per-process workarounds remain, deployable on this machine, other
machines and a cloud account. No release tag until the programme is done, so there is no
compatibility duty to a released version: wipe and re-ingest beats migration, no shims, no dead
code, and no step writes code a later step rewrites.

The measure is photos per second per core and per GPU from "photo in" to "stored, findable and on
the map", and the latency of that path for one photo.

## 1. What the measurements say today, and what follows

### 1.1 The API capacity rows (8 API processes, stub engine, 2 objects per photo)

| Clients | 1 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|
| Small photos (about 165 KB), photos/s | 56 | 212 | 329 | 461 | 495 |
| 20 MP photos (about 6.5 MB), photos/s | 18 | 56 | 80 | 91 | 95 |

| Cost per photo (host CPU) | Small | 20 MP |
|---|---|---|
| API container (8 uvicorn processes plus their ExifTool children) | 19 to 21 ms | 85 to 109 ms |
| PostgreSQL | 1.0 to 1.5 ms | 3.7 to 6.5 ms |
| OpenSearch | 6.5 to 14 ms | 7.4 to 18 ms |
| Stub engine (its own copy of the photo) | 0.5 ms | 9 to 12 ms |

At 32 small-photo clients the API used 10.7 cores, OpenSearch 4.5, PostgreSQL 0.65; no store queue
was seen. The API processes are the bound. Real photos give about 7 objects, not 2, so real API
cost is higher. Not measured yet: where the API's CPU goes, the 30-process and 1-process grids,
set P rows.

### 1.2 What a native engine needs from the host (plan 17 sections 0.1 and 0a)

| Fact | Number |
|---|---|
| Engine host CPU per photo, end to end | COCO 7.3 to 7.7 ms; 20 MP 92 to 103 ms (nvJPEG's host phase) |
| Card ceiling, plan F (detector + frame encoder) | COCO about 196 photos/s; 20 MP about 125 (138 measured at 128 in flight) |
| Card ceiling, plan C (ingest plan with crop vectors) | COCO 52 to 65 (arithmetic); set P 63 to 71 |
| Host | two Xeon E5-2680 v3, 24 physical cores |

### 1.3 The core budget (estimate, arithmetic on 1.1 and 1.2)

| Case | Engine cores | Python API cores (85 to 109 ms) | Stores | Sum against 24 |
|---|---|---|---|---|
| One card, 20 MP, plan C at 70/s | 6.4 to 7.2 | 6.0 to 7.6 | about 1 | 13 to 16: fits |
| One card, 20 MP, plan F at 125/s | 11.5 to 12.9 | 10.6 to 13.6 | about 1.5 | 24 to 28: host-bound |
| Two cards, 20 MP, plan C at 140/s | 12.9 to 14.4 | 11.9 to 15.3 | about 2 | 27 to 32: host-bound |
| Two cards, COCO, plan F at 390/s | 2.8 to 3.0 | 7.4 to 8.2 | about 4 | 14 to 15: fits |

Read plainly: on small photos and one card the Python API keeps up. On 20 MP photos, on two cards,
and on any cloud host with few cores (section 6), the API's CPU per photo decides whether the GPU
can be fed at all. Each millisecond of API CPU per photo costs 0.14 cores at 140 photos/s. So the
first steps of the port must cut API CPU per photo on the ingest path, uploads included, and the
size-dependent part first.

### 1.4 Where the size-dependent API cost comes from (read in code)

On the engine route the API never decodes (`ImageContext.pil` is None, `ingest_index.py:96`). It
touches the photo's bytes in these places, per photo:

1. `read_files` (`src/utils/file_read.py:56`) reads the whole file into one `bytes`.
2. `frame_engine.process` (`src/clients/frame_engine.py:308-350`) wraps it in a numpy object array;
   tritonclient's `serialize_byte_tensor` packs and joins it (a full copy), the request builder
   extends `raw_input_contents` (another), gRPC serialises the message on send (inferred: 3 to 4
   full copies; the first two on the event loop thread).
3. `read_capture` (`src/services/curation/capture_read.py:144`) feeds the bytes to ExifTool through
   a pipe in 1 MiB chunks; `read_capture_file(path)` (same file, line 155) already reads by path and
   is used by backfills.
4. The upload route also parses multipart into a spooled file, reads it into memory
   (`ingest_upload.py:136`) and writes it again (`persist_uploaded_bytes`).

Not size dependent: `imohash` (about 48 KiB sampled), documents, vectors, rows. The crop JPEGs the
engine returns grow with box size (no side cap by default): inferred, measured by M1.

## 2. Decisions

### 2.1 The target architecture

```text
 browser / scripts ──HTTP──► op-server "api" (Rust, axum) ──tonic──► op-frame-engine (Rust, one per card) ──► Triton
                               │  ingest pipeline in-process          ▲ reads the photo by storage key
                               │  ExifTool -stay_open children         │
                               ├──► PostgreSQL (record, jobs, events) ◄┤
                               ├──► storage layer (local dir or object store)
                               └──► Valkey (wake-ups; Celery broker for Python workers)
 op-server "worker" (Rust job runner)   op-server "periodic" (Rust ticks)
 Python workers (Celery): trainer, export, evaluator, probe, promote, numeric jobs (FAISS, scikit-learn, UMAP)
 jobs-dispatch (Python, small): LISTEN on PostgreSQL, sends the Celery message for a Python-kind job row
```

| Question | Decision | Why |
|---|---|---|
| One binary or several | one server binary `op-server` with roles as subcommands (`api`, `worker`, `periodic`, and `all` for a one-host install); the engine stays its own binary and process per card (plan 17 section 2.1); each role runs in its own container | one image and one code base for the CPU side; roles scale apart; the engine stays a pure function (photo in, results out) with no database, which keeps the fault rule of plan 17 section 1.3 and lets `gpu_frame` move to the owner's Rust serving project |
| Processes per host for the API | one process; a tokio multi-thread runtime over all cores; no `WEB_CONCURRENCY` | no interpreter lock; one connection pool, one ExifTool pool, one metrics registry; per-process singletons and the multi-process Prometheus directory disappear |
| Where ingest runs | in the `api` role, in-process (and in the `worker` role for bulk and source jobs, the same crate) | the measured cost is per photo in the API; a hop to a separate ingest service adds a copy and a process boundary for no gain |
| Inside the engine process | no; the engine never writes the record | keeps the engine stateless and replaceable; project, row-level security and claims stay in one place |
| What stays Python, and how Rust talks to it | trainer, model export, bake-off and probe (Ultralytics, PyTorch, TensorRT, ONNX Runtime); promote (the `onnx` head edit); the numeric jobs (FAISS, scikit-learn, SciPy, UMAP: `cluster_run`, `item_scores`, `embedding_viz`, `region_fp_centroids`, `persons_cluster`, the four `near_dup_*` scoring kinds); `autolabel` and `assist_loop` unless R10 finds Rust equivalents; the segmenter and vLLM servers. The VLM client jobs port (section 2.3). Rust writes a `jobs` row (the spec) and the Python worker writes its results to the record | section 2.3 |
| HTTP framework | axum 0.8.9 | section 2.2 |
| Database driver | tokio-postgres 0.7.18 with deadpool-postgres 0.14.2 and pgvector 0.4.2 (features `postgres`, `halfvec`) | section 2.2 |
| Queue | the existing `jobs` table and its claim protocol, run by a Rust runner for Rust kinds; Celery and Valkey stay for the Python workers only; Rust never writes a Celery message | section 2.3 |
| Search | SQL on the record (gate A) and the vector index gate B picks; if gate A or B keeps OpenSearch as a derived store, the official `opensearch` crate 2.4.0 | no OpenSearch code is ported before 15c S9 settles which shapes remain |
| Contract | code-first: utoipa 6.0.0 derives on the Rust wire structs; during the port the generator merges the Rust part and the FastAPI part into the one committed `contracts/openapi/curation.json`; equality is semantic (section 2.4) | section 2.4 |
| Metrics, logs | prometheus-client 0.26.0 (as the engine, plan 17 section 3), same metric names and labels as today; tracing 0.1.44 with tracing-subscriber 0.3.23 JSON output, same field names (`request_id`) | |
| Workspace | one Cargo workspace, `native/` (section 7.1) | |

### 2.2 Library choices, each against its alternative

Versions are the latest stable on 2026-11-15, read from the crates.io release records
(`https://crates.io/api/v1/crates/<name>`), pinned exactly in `native/Cargo.lock` as plan 17 does.

| Need | Choice | Alternatives and why not | Documentation read |
|---|---|---|---|
| HTTP server | axum 0.8.9, tower-http 0.7.1 (request id, CORS, trace, body limit, timeout), hyper-util 0.1.21 | actix-web 4.19.0 is mature and fast, but brings its own runtime model and middleware; axum "doesn't have its own middleware system" and uses `tower::Service`, so it shares middleware and the hyper and tokio stack with tonic, which the API already needs as the engine's client | https://docs.rs/axum/0.8.9/axum/ (tower, SSE under the default `tokio` feature, `Multipart` behind the `multipart` feature) |
| Streaming uploads | `axum::extract::Multipart` with `Field::chunk()` (a `Stream` of `Bytes`); `DefaultBodyLimit::disable()` on the upload route only, replaced by tower-http's `RequestBodyLimitLayer` at the served byte cap | buffering extractors stop at 2 MB by default and hold the body in memory | https://docs.rs/axum/0.8.9/axum/extract/struct.DefaultBodyLimit.html ; https://docs.rs/axum/0.8.9/axum/extract/multipart/struct.Field.html |
| CPU-bound work in the server | a dedicated rayon 1.12.0 pool, results back through a oneshot channel | `spawn_blocking`: tokio says "the upper limit on the number of blocking threads is very large" and recommends "a separate thread pool dedicated to CPU bound tasks", naming rayon | https://docs.rs/tokio/1.54.1/tokio/index.html ("CPU-bound tasks and blocking code") |
| PostgreSQL | tokio-postgres 0.7.18 (pipelining "automatically when futures are polled concurrently"; a `binary_copy` module for vector bulk loads), deadpool-postgres 0.14.2, pgvector 0.4.2 | sqlx 0.9.0: its checked `query!` macros need "`DATABASE_URL` ... set at build-time" or a committed `.sqlx` directory made by `cargo sqlx prepare` against a running database; lanes run no database (global lane rule), so every new query would wait for a wave. The check sqlx gives at compile time is replaced by a database-tier test that prepares every registered statement against the head schema (section 7.2) | https://docs.rs/tokio-postgres/0.7.18/tokio_postgres/ ; https://docs.rs/sqlx/0.9.0/sqlx/macro.query.html ; https://github.com/pgvector/pgvector-rust |
| Row-level security | a `ProjectTx` type: the only way to get a transaction for project data is `project_tx(pool, project)`, which opens the transaction and sets the project scope as `src/db/session.py` `project_tx` does; every repository function for project tables takes `&mut ProjectTx` | a runtime check per call can be forgotten; a type cannot | (pattern; PostgreSQL row security as 02a and 15 R10) |
| Job queue | the existing `jobs` table protocol (`src/tasks/runtime/`, 02b) ported as a Rust runner: claim by `FOR UPDATE SKIP LOCKED`, lease and heartbeat, terminal states, exclusive groups, pausable slices; wake-up by `LISTEN`/`NOTIFY` with a 2 s backstop poll | apalis 1.2.0 (maintained) brings its own schema and semantics: a second jobs table beside the one the Python workers, the reaper and `/curation/jobs/{id}` read. Custom code is justified: no library implements this table's protocol | https://crates.io/api/v1/crates/apalis ; https://www.postgresql.org/docs/18/sql-notify.html |
| Talking to Celery | none from Rust: a small Python `jobs-dispatch` process listens for `NOTIFY` and calls today's `dispatch()` (`src/services/jobs/dispatch.py`) | the Rust `celery` crate's last release is 0.5.5 of 2023-09-25 (unmaintained); Celery's protocol page leaves `embed` signatures unspecified and says the "Body is only for language specific data"; the Valkey transport envelope is kombu's code, not a document. PostgreSQL delivers `NOTIFY` "until and unless the transaction is committed", which is exactly the "send after commit" rule of `dispatch.py` | https://crates.io/api/v1/crates/celery ; https://docs.celeryq.dev/en/stable/internals/protocol.html (Celery 5.7) |
| Valkey (wake-ups, Celery broker reads) | redis 1.8.2 | fred 10.1.0 last released 2025-02 | https://crates.io/api/v1/crates/redis |
| OpenAPI | utoipa 6.0.0 with utoipa-axum 0.3.0: serde `rename`, `rename_all`, `tag`, `untagged`, `default`, `deny_unknown_fields`, `flatten` are read; doc comments become descriptions; `Option` is nullable | aide 0.15.1 (schemars based): fewer users, same merge problem; contract-first codegen from the frozen file: the file would become hand-edited, against `contracts/README.md` ("Do not edit by hand") | https://docs.rs/utoipa/6.0.0/utoipa/ ; https://docs.rs/utoipa/6.0.0/utoipa/derive.ToSchema.html |
| Contract comparison | oasdiff (Apache-2.0; OpenAPI 3.1 support documented in its `docs/OPENAPI-31.md`) for breaking changes, plus a normalised schema diff (section 2.4) | byte equality with Pydantic output (titles, `anyOf` for optional fields) would need a hand-written schema for each of 759 components | https://github.com/oasdiff/oasdiff |
| Request validation | serde with `serde_path_to_error` 0.1.20 (the failing path becomes `loc`) and garde 0.23.0 for field constraints (`ge`, `le`, lengths, patterns) | validator: not chosen without a reason beyond garde's recent release; the first lane reads both pages and records the choice | https://crates.io/api/v1/crates/garde |
| Files and object storage | cap-std 4.0.3 `Dir` handles for local roots (paths opened relative to an open directory); object_store 0.14.2 for S3, GCS, Azure behind the storage layer's key contract (15 section 8, slice F) | `std::fs` with string checks for `..` and symlinks: the lane proves cap-std's escape refusal with a test, since its front page does not state it | https://docs.rs/cap-std/4.0.3/cap_std/ ; https://crates.io/api/v1/crates/object_store |
| Content hash | `imohash` 0.1.2 if it matches goldens generated from the Python `imohash` on the standard lists; else a port of the small algorithm with those goldens | | https://crates.io/api/v1/crates/imohash |
| ExifTool | a driver for `exiftool -stay_open True -@ -` by path, written here (about 200 lines, estimate), as `src/clients/exiftool_client.py` is today | `exiftool` 0.3.1 and other wrappers have few users (under 10,000 downloads); `exiftool-rs` 1.0.0 (a Rust port of ExifTool 13.60, released 2026-10-30) is a later candidate only after a parity run against ExifTool on the format corpus, owned by the gallery metadata plan ([05](../photo_library/05_p2a0_metadata.md)) | https://crates.io/api/v1/crates?q=exiftool |
| Text tokens for text search | tokenizers 0.23.2 (the library under transformers' fast tokenizers), checked token for token against today's tokenizer | | https://crates.io/api/v1/crates/tokenizers |
| HTTP clients (segmenter, VLM, model-server HTTP) | reqwest 0.13.5 | | https://crates.io/api/v1/crates/reqwest |
| Tests | proptest 1.12.0; testcontainers-modules 0.15.0 (PostgreSQL) for the database tier | cargo-fuzz needs a nightly toolchain, against the pinned stable toolchain of plan 17 | https://crates.io/api/v1/crates/proptest |

Licences are recorded by `cargo deny` at selection with plan 17's allow list; none above is
outside it by its crates.io record, and the first lane's `cargo deny` run is the proof.

### 2.3 What stays Python, with the reason (the roadmap statement checked)

Roadmap 14 section 4 says "Training, export and the VLM stay Python: they are the GPU frameworks'
language". Checked against the inventory's import table:

| Group | Libraries | Verdict |
|---|---|---|
| `train`, `model_export`, `bakeoff`, `probe`, `promote` | Ultralytics, PyTorch, TensorRT and ModelOpt, ONNX Runtime, `onnx` | stay Python: no Rust equivalent of the training and export frameworks |
| Numeric jobs (8 kinds above) | FAISS, scikit-learn, SciPy, UMAP, cuML optional | stay Python in this plan (plan 16 owns their logic). One job per process, and these libraries run in C or C++ outside the interpreter lock, so the lock is not their bound. Reopened per kind only when a measured run at 1 million vectors shows the Python process (not the library) binding a wait a person feels |
| VLM client jobs and the VLM worker loop (`vlm_label_batch`, `vlm_verify_region_batch`, `scripts/curation/vlm_worker.py`) | httpx and tenacity | challenged and changed: the GPU framework is vLLM in its own server, not these clients. They are input-output bound, so speed is not the reason to move them; the fold is. They write class opinions through `apply()` (15c S8a), and a fold kept in Python for them would exist twice for good. They port in R10 |
| `autolabel`, `assist_loop` | scikit-learn (`raw_label_clusters.py`, `assist_loop/splitter.py`) and VLM calls | they also write opinions. At R10's entry the lane reads what each scikit-learn call does: a call with a maintained Rust equivalent ports with the job; otherwise the job stays Python and writes its opinions through the server (one internal write route of the `api` role, R10), never through a Python fold |
| Segmenter and vLLM servers | their own images | stay |

The rule behind the table: after R12 only Rust folds. A Python job that calls `apply()` after wave
G is either ported or writes through the server. The numeric jobs write run and result tables that
are joined, not folded (15 section 4.4); R10's entry confirms this per kind with `rg apply\(`.

The remaining Python processes are single-purpose workers whose cost sits in GPU or C libraries.
"No more GIL or workarounds" holds for every process on the request and ingest paths.

### 2.4 The contract and the frontend

The frontend imports `contracts/ts/itemWire.ts` and reads `contracts/openapi/curation.json` in its
contract tests key by key (for example `frontend/src/lib/itemFilter/filterKeys.contract.test.ts`).
The contract today is FastAPI's `app.openapi()` (OpenAPI 3.1.0), 2.3 MB, 759 components.

- The generator gains a Rust part: `op_contracts` (a binary in `native/`) prints the utoipa document
  of the ported routes; `scripts/codegen/export_api_contracts.py` merges it with the FastAPI part,
  sorts keys and writes the same file. The drift hooks run both.
- Equality is semantic, not byte for byte: (1) `oasdiff breaking` between the contract on the
  integration head and the merged one reports nothing outside the slice's intended-change list;
  (2) a normalised diff of each ported route's operation and its schemas (titles dropped, the two
  spellings of an optional field made one, keys sorted) is empty outside that list; (3) the
  frontend's contract tests, `npm run check` and `npm run build` pass unchanged. The first slice of
  each route family records the cosmetic regeneration (titles and optional spelling) as one
  intended change.
- Error bodies keep FastAPI's shapes, which `frontend/src/lib/api_errors.ts` reads: `{detail:
  "..."}`, `{detail: {error, message}}`, and 422 `{detail: [{loc, msg, type}]}`. The `msg` text of
  validation errors may differ (intended change), except "String should match pattern '...'", which
  the frontend rewrites and Rust emits word for word.
- `contracts/json/item_wire.json`, `itemWire.ts`, `regionStatus.ts`, `classSources.ts`,
  `classNames.ts`, `keymap_actions.json` and the frame-engine contracts stay generated from their
  Python sources until the slice that ports the source; then the source moves to Rust and the files
  stay byte-identical (a test compares).

## 3. The order of the work, by measured gain

### 3.1 The candidates (brief items a to d)

| Candidate | What it gains (estimate unless measured) | Cost | Waits for | Must be measured first |
|---|---|---|---|---|
| a. The engine reads the photo by storage key (server-side paths and uploads both, since an upload is written to storage first): the API sends a key and a content claim (size, imohash), not the bytes | removes items 1 to 3 of section 1.4 from the API; estimate: 20 MP API CPU from 85 to 109 ms toward 25 to 45 ms in Python (the small-photo cost plus crop handling); the engine reads the file once (it needs the bytes anyway), removing the stub's 9 to 12 ms message copy | engine: one new input `source` and key resolution under configured roots (Rust, permanent); API: about 150 lines (estimate) in `file_read.py`, `ingest_stage.py`, `frame_engine.py` client, capture by path | N3 (protocol version 1 is fixed until then and the Python engine takes no new feature, plan 17 sections 1.1 and 0); slice F (storage keys) | M1, M3 |
| b. The ingest writer in Rust, the API only accepting the job | removes the commit path's Python work (documents, rows, vectors, events); leaves the per-photo request work in Python; no gain on the size-dependent cost by itself | needs the fold, attach and summary writer in Rust (section 3.3); a second process boundary for every photo | S1a-ii; fold stable | M1 (the share of commit work) |
| c. Ingest routes first in a Rust server beside the Python one (the Rust server fronts port 4603 and forwards unported paths), then families one at a time | estimate: API CPU per photo 2 to 6 ms of server work at both sizes plus ExifTool's own CPU (unknown until M5), against 19 to 21 (small) and 85 to 109 (20 MP); uploads stream to storage without a whole-photo buffer | a server skeleton, the record and jobs core, the ingest pipeline; each later family is its own slice | N3; S1a-ii and the fold stable (section 3.2) | M1 to M5 |
| d. A full port in one cut | none until the end; 435 route decorators, 44 job kinds and 175,623 lines of `src/` in one change, parity proven all at once, while 15c rewrites the same modules | very large; a long-lived branch rewritten twice | everything | rejected |

Decision: c, with a's engine half inside its first slice. b is not a step of its own: a writer
service beside a Python API keeps the per-photo request cost in Python and adds a boundary, and the
Rust ingest pipeline of c contains the writer anyway. d is rejected.

a's Python half (sending keys from the Python API) would be rewritten by c's first slice. It is
planned as a conditional slice RP (18a) only under branch A of decision D1 below.

### 3.2 What each step waits for, so nothing is ported that is about to be deleted

| Rust slice (18a) | Waits for in storage (15c) | Waits for in the engine (17) | Why |
|---|---|---|---|
| RP (conditional) | slice F (keys) | N3 | protocol frozen before N3 |
| R1 ingest | wave E merged (S6 places new photos as membership rows; S7 changes import's image stage); S9 not needed | N3 (and N1b, N1c for faces and text in the benchmark plan) | S1a-ii rewrites the commit path; S6 puts placement back into it; porting earlier means a Rust commit path that two Python lanes must then edit |
| R2 core routes (G5) | wave G (S9): the curation search path gone; the reserved core project of 15 answer 1 | N3 (the engine branches `detect`, `faces`, `face_embed`, `ocr`, `embed_frame` native) | 15c answer 7 keeps core data on the global indexes until this track; doing it in Python first would be rewritten |
| R3 jobs, ticks, live events | R1 merged | none | the runner core arrives in R1; the beat and the server-sent events port after |
| R4 to R11 route families | wave G merged (every family on the record, the contract diff of 15c section 2 applied) | none | 15c section 2 changes wire fields of these families slice by slice until S9 |
| R12 Python API removed | all of the above | none | |

The fold in Rust: R1 needs `apply()`, `attach` and `store_states` (`src/domain/fold*.py`,
`claim_match.py`, `src/services/record/apply.py`, `attach.py`, `src/db/repos/state.py`: about 3,100
lines of Python domain and 1,300 of repositories on `int/storage`). From R1's merge until the last
Python caller of the fold is ported (R10 at the latest, section 2.3), the fold exists twice. The bound on that:
a generated case contract `contracts/json/fold_cases.json` (from the Python fold over the recorded
differential cases and property-generated histories), a drift hook, and the rule plan 17 section 7
already uses for the plan parser: the lane that changes the fold regenerates the cases and changes
the Rust fold in the same commit. The Python fold is deleted with its last caller.

### 3.3 The resulting order

```text
now ── R0 measurements ──┬────────────────────── (wave E, N3) ── R1 Rust ingest ── R3 jobs, ticks, events ─┐
                         └─ D1 branch A only: RP after N3 + F                                             │
                                          (wave G, N3) ── R2 core routes, OpenSearch leaves ──────────────┤
                                          (wave G) ── R4 .. R11 families, ordered by M7 ───────── R12 Python API removed
```

| Step | Gain (estimate unless marked) | First usable result |
|---|---|---|
| R0 | the numbers that fix D1 to D3 | none (measurement only) |
| R1 Rust ingest (uploads and server-side paths) | API CPU per photo from 19 to 21 ms (small) and 85 to 109 ms (20 MP) to 2 to 6 ms plus ExifTool (estimate); at 140 photos/s on two cards about 11 to 15 cores freed; uploads with no whole-photo buffer and back pressure to the browser | ingest at the card ceiling on 20 MP and two cards where the host allows; the cloud benchmark's ingest half |
| R2 core routes on the engine and the record | the core routes stop preprocessing in Python (cv2, numpy); the five global OpenSearch indexes and `image_contents` go; with 15c S9 and gate B in PostgreSQL, OpenSearch leaves the stack (its heap and its cores: 4.5 cores at 495 photos/s on the old curation path, measured) | one store; `/detect`, `/faces`, `/embed`, `/ocr`, `/analyze`, `/search` on the engine |
| R3 | small throughput gain; removes Celery beat and the Python job runtime for Rust kinds; live events from one process | the Python API no longer runs jobs |
| R4 to R11 | per family, measured by M7; the page bars of 15 section 6 already hold in SQL (15b gate A: deep list page 1.3 / 2 ms p50 / p99 in PostgreSQL), so these gains are API CPU per request and one language | each family is usable alone behind the same paths |
| R12 | the Python API, FastAPI, uvicorn, the forwarder and the Python contract generator deleted | one backend language on every request path |

### 3.4 Decisions the measurements settle

| # | Decision | Branches | Settled by |
|---|---|---|---|
| D1 | Does the Python API get the engine-by-key change (RP) before R1? | A: yes, when M3 shows the change removes at least 60% of the size-dependent API CPU at 20 MP and R1's entry (wave E) is expected more than four weeks after N3; RP is then the one planned piece of Python that R1 rewrites, stated as such. B: no; the engine's `source` input ships inside R1 | M3, and the coordinator's dates for N3 and wave E (question Q1) |
| D2 | Does ExifTool stay on the hot path? | keep (M5 under 3 ms CPU per photo at 20 MP, estimate threshold); else the hot path reads only capture time, orientation and GPS from EXIF in Rust and the full ExifTool read becomes a queued job, which needs the gallery metadata plan's agreement ([05](../photo_library/05_p2a0_metadata.md) owns the reader) | M5 |
| D3 | API worker count until R1 lands | the 30-process grid: if photos/s at 20 MP rises above the plan C ceiling with CPU per photo unchanged, set the compose default from the grid (configuration only) | M2 |
| D4 | Does OpenSearch survive as a derived store? | no (R2 deletes the client); yes (R2 and R6 use the `opensearch` crate behind the vector-index or search-copy interface) | gate A in S9, gate B round 2 |

## 4. The upload path is a hot path

The cloud picture (section 5) is the browser uploading through the existing route. There is no
second upload API.

Today, read in code: the frontend (`frontend/src/lib/api_ingest.ts:40-52`) sends `POST
{prefix}/ingest/upload` as multipart: one `images` part per file, then `image_paths` (a JSON list of
identifiers) and `source`. The planner (`frontend/src/lib/ingest/uploadPlanner.ts`) packs files up
to the served `upload_max_images_per_request` (128) and `upload_max_bytes_per_request` (512 MiB,
`src/config/curation.py:129-130`); `DEFAULT_UPLOAD_CONCURRENCY` is 2 requests in parallel
(`frontend/src/lib/ingest/ingestConfig.ts:35`). The route answers once every photo of the request
is done, with one result per photo. The live tier covers it (`tests/live/test_live_ingest.py`).

The Rust route keeps the path, the form fields, the limits and the response, and changes what
happens inside:

1. Each `images` part is streamed with `Field::chunk()` into the storage layer: a temporary key on
   the landing root, written as chunks arrive; no process holds a whole photo except the engine,
   which must for nvJPEG.
2. When a part ends: `imohash` from the stored file (three 16 KiB reads), the content claim, then
   the engine call with the storage key, and capture by path. The next part is read meanwhile, so
   engine work overlaps the rest of the upload.
3. Back pressure: the photos admitted to work and the bytes in flight are bounded (one semaphore
   each, sized from cores and the engine's in-flight cap). When either is full the route stops
   polling the body; hyper then stops reading the socket and TCP flow control slows the browser. A
   request that cannot start inside its admission bound answers 503 with `Retry-After`, as the
   engine-unavailable answer does today.
4. A duplicate's temporary file is deleted; a new photo's file is renamed or copied to its content
   key by the storage layer after the commit.

Contract: no wire change is needed for speed. One frontend edit is planned with R1: append
`source` and `image_paths` before the `images` parts (`api_ingest.ts:47-50`), so each photo can
commit as soon as it is processed; a client that sends them last still works and its commits wait
for the end of the body. `scripts/curation/ingest_upload.py` gets the same order. The default
concurrency of 2 is reset from measurement M10 (a frontend constant, not a contract).

## 5. The cloud benchmark

The north star (owner, 2026-11-15): a photo service in the cloud that takes uploaded photos,
processes them as fast as possible, stores them in PostgreSQL and shows them on a map and in time or
event groups as quickly as possible. The product stays a local application; this is the benchmark
to work toward, and it deploys as Docker. The Cropwright frontend runs in front of the Rust backend;
authentication and network security are assumed set up for such a deployment and are not built
here (section 6).

### 5.1 What is measured

Photos arrive as uploads through `POST /curation/projects/{project}/ingest/upload` (bytes, not
server-side paths), sent as the frontend sends them (section 4). Processing is the library plan:
detector, frame vector, crop vectors, faces, text lines, and capture time and place from EXIF.

| Id | Metric | How |
|---|---|---|
| B1 | Photos per second per GPU, end to end | photos committed / wall seconds of the steady window / GPUs; failed photos counted and must be 0 |
| B2 | Photos per second per CPU core | B1 x GPUs divided by (a) the cores busy, summed over every container's cgroup `cpu.stat` in the window, and (b) the cores provisioned |
| B3 | Time from the last byte of a photo's upload to "findable", p50 and p99 | the client stamps the last byte of each part; "findable" is the first `crop.created` event for the photo on the project's event stream, confirmed by the photo page route answering it and a search by its own picture returning it. No polling loop: the event is the signal, one read confirms |
| B4 | Time from the last byte to "on the map" and "in a time group", p50 and p99 | after the event, one read of `GET /curation/projects/{project}/map/cells` that counts the photo's cell, and one read of the time grouping (below) that counts its group |
| B5 | Cost per million photos | price per hour / (B1 x GPUs x 3600) x 1,000,000, compute only; storage, transfer and a managed database added as separate stated lines |
| B6 | The same on this machine | B1 to B4 on one A6000 and on two; B5 not computed (owned hardware) |

Named cloud host: AWS `g6.4xlarge` (one NVIDIA L4 with 24 GB, 16 vCPU, 64 GiB), $1.3232 per hour
on demand in us-east-1, Linux (price trackers of the AWS price list: https://ec2calc.cloud/instances/g6.4xlarge
and https://cloudprice.net/aws/ec2/instances/g6.4xlarge, read 2026-11-15). B5 = 367.6 / (photos/s)
dollars per million: 50 photos/s gives $7.35, 100 gives $3.68 (arithmetic; the L4's rate is not
measured). Two facts change the engine on that host: the L4 has 4 hardware JPEG decoders (NVIDIA's
L4 page, https://www.nvidia.com/en-us/data-center/l4/), which plan 17's GA102 lacks, and 16 vCPU
are 8 cores. At plan 17's 92 to 103 ms of host CPU per 20 MP photo, 8 cores cap the engine near 80
photos/s before the API takes any: the API's CPU per photo and the use of the hardware decoder
(nvJPEG's hardware backend, baseline single-scan JPEG only) decide B5. The hardware backend is a
plan 17 revision item (question Q5), not this plan's.

Time groups: today the route `GET /curation/projects/{project}/events` is the live job-event
stream, not photo events. Photo events and trips are planned in the gallery's create plan
([30](../photo_library/30_p2c_create.md), incremental re-segmentation of plus or minus 7 days after
ingest); the timeline buckets in the organise plan ([10](../photo_library/10_p2a1_organise.md)).
The benchmark measures B4 against the timeline buckets once they exist and against events once P2c
builds them; until then a SQL probe on the capture-time column of the record stands in and is
labelled so. This plan designs neither view. The gallery plan's existing target "New photo visible
after ingest: 30 s" ([00](../photo_library/00_photo_library_plan.md) section 10) is the starting bar
for B3 and B4; the owner sets the benchmark's own bars after the first baseline run.

### 5.2 Runs at product scale

| Run | Photos | Mix | Purpose | Expected wall time (estimate) |
|---|---|---|---|---|
| Regression | 2,000 per list: `op-bench-v1/coco2k` and `setp2k` (registry names) | one list per run | every slice that touches ingest; one targeted run | under 1 minute each at 40 to 120 photos/s |
| Headline | 100,000 | half set P (20 MP), half public small photos (COCO train2017, Open Images, Objects365 per the dataset registry) | B1 to B6 | 25 to 45 minutes at 40 to 70 photos/s |
| Scale | 1,000,000 | same mix | what shows only at scale: database growth, vector index build and size, memory, the map and time views filling while ingest runs, B3 and B4 drift over the run | 4 to 7 hours at 40 to 70 photos/s; under 3 hours at 120 |

Staging (this machine): the photos are read from NVMe, never from the NAS during a run. NVMe has
313 GB free on 2026-11-15. The 100,000 run needs about 340 GB of source photos at the
half-and-half mix (50,000 x 6.5 MB plus 50,000 x 0.2 MB) and as much again for the stored originals;
the 1,000,000 run about 3.4 TB of each. Neither fits whole. For both long runs the client reads a
rolling window of photos staged ahead from the NAS to NVMe, and the storage layer keeps the stored
originals on a NAS root while the landing root (where uploads are first written) stays on NVMe,
which is the cloud design (landing disk, then object storage). Measurement M11 records the NAS read and write rates first; if the link binds the
1,000,000 run, that run reports the scale effects and the link as its bound, and the headline rate
comes from the 100,000 run. Set P appears in results as numbers only; the public 1,000,000 part is
fetched and entered in the dataset registry before the run (a stewardship task, not a lane).

### 5.3 What each metric depends on

| Metric | Depends on |
|---|---|
| B1, B2 | R1 (Rust upload ingest); N3, N1b (faces), N1c (text) of plan 17; S1a-ii; D2 (ExifTool) |
| B3 | R1; S1a-ii (transactional commit); S3 and gate B (search in the record, visible at commit; an index outside PostgreSQL adds its lag) |
| B4 | S1b-ii (photos and map reads); P2a-1 timeline buckets or P2c events; R4 if the map route must be fast under ingest load |
| B5 | B1 on the named host; the L4 hardware decoder (Q5); section 6 images |
| B6 | all of the above on this machine; plan 17 N2's two-card row |

## 6. Deploy shape (gap G14, the parts this port changes)

| Part | After the port | Notes |
|---|---|---|
| CPU images | one `op-server` image (Rust binary, ExifTool and Perl, non-root, `-slim` base); one Python worker image for the numeric and VLM client jobs and `jobs-dispatch`, without FastAPI and uvicorn | sizes measured against today's API image (M8); a Trivy scan before publishing (global rule) |
| GPU images | the engine (plan 17), Triton, the trainer, the export image, the evaluator, segmenter and vLLM, unchanged by this plan | |
| CPU-only host | runs the server roles, PostgreSQL, Valkey and the Python CPU workers: browse, label, search, cluster | ingest needs a GPU host with the engine and Triton |
| Split hosts | the engine reads photos by storage key, so a GPU host needs read access to the storage layer (a shared volume or the object store), not a copy pushed by the API | the reason candidate a matters beyond speed |
| Managed PostgreSQL | must provide pgvector with `halfvec`, row-level security, and a login role `op_vector` created by the operator (15 section 14 answer 2); if gate B picks VectorChord, most managed services do not offer it (risk) | the server connects through deadpool with a pool sized per process, not per worker; PgBouncer becomes optional |
| Object storage | object_store behind the storage-layer key contract of slice F; uploads land on local disk first, then move | |
| Authentication | **entry condition for any deployment reachable from the internet**: the API has none by the owner's decision (14 section 9, answer 4: added later). Until it exists the stack stays on loopback, a LAN or a private network of the owner's choosing | not work in this plan |

Left to a later deploy plan: authentication and authorisation (plan 10 F5), TLS termination,
orchestration (Kubernetes or another), managed-database backups and point-in-time recovery, secret
handling, autoscaling, and the GPU arbiter overlay.

## 7. Gates for Rust code

### 7.1 One workspace

One Cargo workspace, `native/`, holds the engine crates and the server crates
(https://doc.rust-lang.org/cargo/reference/workspaces.html): one `Cargo.lock`, so tokio, tonic, hyper
and prost are one version on both sides of the engine protocol; one set of gates; the KServe protos
move from `frame_engine` into a shared `kserve_proto` crate used by the engine and the server.
`gpu_frame` leaves the workspace when it moves to the owner's Rust serving project (plan 17 N3), with
no effect on the server crates.

| Crate | Holds | May depend on |
|---|---|---|
| `op_domain` | pure ports: fold, claim matching, tallies, geometry, filter grammar, cursor codec, wire rules | nothing with input or output |
| `op_store` | the pool, `ProjectTx`, the statement registry, repositories | `op_domain` |
| `op_engine_client` | the engine and Triton clients | `kserve_proto` |
| `op_ingest` | the ingest pipeline, capture driver, storage keys | `op_store`, `op_engine_client` |
| `op_jobs` | the runner, kinds, ticks | `op_store`, `op_ingest` |
| `op_api` | axum routers, wire structs, errors, utoipa | `op_store`, `op_ingest`, `op_jobs` |
| `op_server` | the binary and its roles | all |
| `op_contracts` | the contract exporter and case generators' Rust side | `op_api` |

`scripts/codegen/check_layering.py` gains a rule over `cargo metadata` that holds this table.

### 7.2 The gates

Plan 17 section 3.1 applies unchanged (format, clippy with `-D warnings` and the workspace lints,
`cargo test` with no GPU and no network, `cargo deny` licences, bans, sources, advisories at
pre-push and CI, the `native` CI job, Dependabot, the file-size ratchet over `native/**/*.rs`). The
server adds:

| Gate | Rule |
|---|---|
| No unsafe | `#![forbid(unsafe_code)]` in every server crate (`op_domain` to `op_contracts`) |
| Contract drift | the drift hooks regenerate the merged contract (Rust part from `op_contracts`); stale fails |
| Breaking changes | `oasdiff breaking` against the integration head's contract; anything outside the slice's intended-change file fails |
| Case contracts | `fold_cases.json` and every other generated case file replayed by Rust tests; drift fails |
| Statements | a database-tier test prepares every statement of the registry against the head schema (wave sweep) |
| Property tests | proptest for the fold (any order of machine writes around a claim gives the same row, as the Python property test), the filter grammar, the cursor codec, storage-key resolution (no key escapes its root) |
| Request edges | tests for body limits, upload streaming under a slow reader, the 503 with `Retry-After` at admission, the error shapes of section 2.4 |
| Schema-driven requests | Schemathesis 4.33.4 (PyPI) against the live harness for families with wide query inputs (filters), checking no 5xx and conformance to the contract, at the wave sweep; the first lane confirms its OpenAPI 3.1 support on its documentation |

## 8. Measurements before coding (R0)

Each runs in the coordinator's hold, writes raw files under `artifacts_local/` and a summary row
under 17 section 5's evidence clause. Lists by registry name; set P as numbers only.

| Id | What | Script-level description | Bar or use | What each outcome changes |
|---|---|---|---|---|
| M1 | Where the API's CPU goes | the API capacity harness with its stdlib sampler on (`APICAP_PROF=1`), 8 processes, stub engine, `coco2k` and `setp200`; per-PID `/proc/<pid>/stat` deltas to split uvicorn from ExifTool; log crop JPEG bytes per photo | fact-finding | copies dominate: a and R1 carry the gain (both keep their place); ExifTool dominates: D2's other branch, and R1's estimate drops to ExifTool's cost; per-object Python work dominates: R1's gain grows with objects (M4) |
| M2 | 30-process and 1-process grids | `chain_all.sh` rows at W=30 and W=1, same lists | D3 | photos/s rises with CPU per photo flat: the bound was process count, D3 sets W, R1's gain is cores, not rate; CPU per photo grows with W: contention, R1's gain larger |
| M3 | Path-mode spike | a scratch copy (as the stub patch of 2026-11-15): the API sends the path, the stub engine reads the file, capture by `read_capture_file`, imohash from the file; same rows as M1; no product file touched | D1: 60% of the size-dependent cost | branch A or B of D1 |
| M4 | Real objects per photo | the grid with plan 17 N0's real pipeline, plan C, `coco2k` and `setp200` | slope of API CPU per object | the per-object part of R1's estimate |
| M5 | ExifTool's own cost | drive `exiftool -stay_open` over `coco2k`, `setp200`, `commons` and a format corpus (TIFF, PNG, HEIC, WebP), by path and by pipe; CPU per file and bytes read (`/proc/<pid>/io`) | D2: 3 ms at 20 MP (estimate threshold) | D2 |
| M6 | Core routes | `/detect`, `/embed/image`, `/faces/recognize` at 1, 8, 32 clients on `coco2k`, API CPU per request | R2's gain row | R2's place against R4 to R11 |
| M7 | Read routes per family (after wave G) | the 80 shapes of 15b through their HTTP routes at 8 clients: p50, p99, API CPU per request | the order of R4 to R11 | highest API CPU per request times request rate in the acceptance walk first |
| M8 | Static facts | import `create_app()` and count routes including `/v1`; `pytest --collect-only -m live`; `python -X importtime -c "import src.main"` for torch and transformers; today's API image size | none | slice sizes; the image-size baseline |
| M9 | Region-stage worker (after S4) | profile `scripts/curation/worker.py --continuous` on S4's code: CPU per item, one core saturated or not | R7's priority | |
| M10 | Upload through the route | the Python route today: browser-shaped multipart requests (128 photos, 512 MiB cap, concurrency 1, 2, 4, 8) on `coco2k` and `setp200`; photos/s, API CPU per photo, request latency | the baseline R1 must beat; the frontend default concurrency | R1's upload bar |
| M11 | NAS rates | sequential read and write MB/s to the NAS root from this host | the 1,000,000 run's staging | section 5.2 |

M1 to M5, M8 and M10 come before R1's first lane; M6 before R2; M7 before R4; M9 before R7; M11
before the first scale run.

## 9. Risks and their early signals

| Risk | Early signal | Answer |
|---|---|---|
| The contract cannot be kept equal | oasdiff or the normalised diff reports a change on the first ported route | the intended-change file and a frontend edit in the same slice, or a custom `ToSchema` impl for that type |
| The two folds drift | a fold case fails in Rust after a Python storage lane | the same-commit rule (3.2); the Python fold goes with its last caller |
| The two job runtimes drift | a conformance case (claim race, lease expiry, cancel, exclusive group) differs | one database-tier conformance suite run against both runtimes |
| The port is large (175,623 lines of `src/`, 852 Pydantic classes) | lanes stop at concern boundaries without finishing | families split by concern; one lane per area; at most three lanes per integration branch |
| ExifTool, not the API, is the cost | M5 over the D2 threshold | D2's branch |
| Storage waves slip | wave E later than N3 by months | D1 branch A (RP) |
| Forwarded routes add latency | p99 of unported routes rises through the front | measured in R1's sweep; the forwarder is deleted in R12 |
| Metric names change | the scripted metric-name diff fails | prometheus-client name and label checks per slice |
| The cloud host is CPU-bound | B2 on `g6.4xlarge` shows 8 cores busy below the card ceiling | the hardware JPEG backend (Q5); API CPU per photo |
| OpenAPI 3.1 tooling gaps | oasdiff or Schemathesis misreads a 3.1 construct | the normalised diff stays the gate; the tool row is advisory |
| A storage key escapes its root | the property test or a crafted key test | cap-std handles; refuse `..`, absolute keys and unknown roots before any open |

## 10. Not in this plan

Authentication and any public deployment; the gallery's views (timeline, events, people) and their
read shapes; the engine's internals, its hardware-decoder use and its branches; the storage schema,
its migrations and its slices; cluster, label and train logic (plan 16); training, export,
bake-off, VLM serving in Rust; the frontend beyond the edits named per slice; video; the desktop
application; orchestration.

## 11. Questions for the coordinator

| # | Question | Recommended answer |
|---|---|---|
| Q1 | Allow RP (the Python API sends keys to the engine) as the one planned piece of Python that R1 rewrites? | yes, only under D1 branch A: M3 shows at least 60% of the size-dependent cost removed and wave E is more than four weeks after N3 |
| Q2 | Keep Celery and Valkey for the Python workers, with `jobs-dispatch` as the bridge, so Rust never writes a Celery message? | yes (it keeps the owner's 2026-11-04 decision for Python long jobs); R12 reviews whether the Python workers should claim from the `jobs` table directly and leave Valkey for wake-ups only |
| Q3 | Accept semantic contract equality (section 2.4) instead of byte equality, with one recorded cosmetic regeneration per family? | yes |
| Q4 | The numeric jobs stay Python at the end of this plan, reopened only by measurement; the VLM client jobs port in R10 because they fold (section 2.3), which revises roadmap 14 section 4's "the VLM stays Python" to "the VLM server stays Python" | yes |
| Q5 | Name `g6.4xlarge` (one L4) as the cloud benchmark host, and add the nvJPEG hardware backend to a plan 17 revision | yes to both |
