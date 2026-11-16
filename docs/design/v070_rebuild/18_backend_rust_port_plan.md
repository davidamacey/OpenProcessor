# 18: the backend in Rust, in the owner's order

| Field | Value |
|---|---|
| Status | plan, revision 1 (2026-11-15). Revision 0 (`5763a2d04`) reworked after an independent review (4 blocker, 9 major, 9 minor findings) and the coordinator's 13 decisions; section 13 traces every finding and decision to the text that now answers it. Covers roadmap gaps G5 (core routes on the engine and the record) and G14 (deploy, the parts this port changes) of [14](14_roadmap.md) |
| Companions | [18a_backend_rust_port_slices.md](18a_backend_rust_port_slices.md): how a slice runs, the Cargo workspace and crates, the test and parity rules, and the slices that start now, each complete. [18b_backend_rust_port_later_slices.md](18b_backend_rust_port_later_slices.md): the slices after the record exists, the mapping from the storage slices of plan 15c to this plan, and the traceability tables |
| Code base read | `int/b4-stateless` at `7614a8648` (this worktree); `int/storage` at `0d9412024` (plans 15b, 15c, the Python step 1); `int/native` at `1a3311290` (the `native/` workspace of plan 17). API facts come from the port inventory of 2026-11-15 (gitignored, `artifacts_local/speed_1113/api_port/inventory.md`); its "inferred" rows are used only where re-read in code or marked as hypotheses |
| Numbers | every number names its source; estimates say "estimate". "Capacity rows" = the partial grid of 2026-11-15 (gitignored `artifacts_local/speed_1113/api_capacity/results.md`, `summary_w8_partial.txt`). "Ingest timeline" = gitignored `artifacts_local/speed_1113/ingest_timeline/timeline_clean.md`. Measurement ids M1 to M18 are this plan's (section 8); the storage measurements of plan 15b section 13 are written "15b M1" and so on |
| Not in scope | the engine's internals (plan 17), the storage schema and its design (plans 15, 15a, 15b, 15c: kept as design, section 3.2), the cluster, label and train logic (plan 16), the gallery's views (photo-library plans), authentication |

## 0. The person's job, the owner's objective and his order

A person points OpenProcessor at a folder of photos, or uploads them from the browser or an API
client: thousands to millions, small web photos up to 20 MP camera photos. The detector finds the
objects; every photo and object gets vectors, text and faces; the person finds photos, labels their
own classes by cluster and trains a detector; a gallery (timeline, map, people, events) sits on the
same record.

The owner's objective: "the whole point is to process this image data set as fast as possible and
efficiently and make it work for other tasks as well for images". The measure is photos per second
per core and per GPU from "photo in" to "stored, findable and on the map", and the latency of that
path for one photo.

The owner's order (2026-11-15), which this plan follows:

1. The Rust image engine (plan 17).
2. The API and the Python code replaced by Rust, with a robust upload endpoint for one or many files
   from the frontend or API clients; uploads stored in S3 in the cloud, organised and stored
   correctly. The upload path is the one the Cropwright frontend uses today: no second upload API.
3. The frontend gallery update.
4. The database and the clustering.
5. A cloud-like deployment: "a cloud version of our app for people to store and organize their
   photos for a fee, or for small businesses to run".

His standing rules: Rust writes to the database, is the API, does the image processing, the
housekeeping and admin work, the job queue, and the clustering where a Rust crate or our own code can
do it. Python is used only for training models (the exact list is section 2.3). No code that another
lane writes too, and none that a later step rewrites. Selective checks per slice; the full sweep once
per large merge. No authentication built now; a public deployment needs it first. No release tag
until the whole programme is done, so wipe and re-ingest beats migration, no shims, no dead code.
Benchmarks at product scale: 2,000, 100,000 and 1,000,000 photos, set P and the public sets.

## 1. What the measurements say today

### 1.1 The capacity rows (8 API processes, stub engine, 2 objects per photo)

Setup, from the capacity rows: host load 13.9 to 18.5 at the start of the rows (other projects'
work), 2 or 3 repetitions per row, server-side path ingest (`POST .../ingest/batch`), the Rust
engine as a stub returning 2 objects per photo. Small photos are the `coco2k` list (about 165 KB);
the 20 MP rows used the `mp20` list (about 6.5 MB), not set P. The 20 MP rows at 16 and 32 clients
lasted 2 to 3 seconds each, so their spread is wide.

| Clients | 1 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|
| Small, photos/s, median (min to max) | 56.5 (56.0 to 56.7) | 211.9 (172.8 to 218.4) | 328.9 (286.5 to 371.3) | 460.6 (456.1 to 465.2) | 495.2 (483.0 to 507.5) |
| 20 MP, photos/s, median (min to max) | 18.1 (17.7 to 18.5) | 55.9 (54.2 to 57.5) | 79.6 (74.4 to 84.8) | 91.3 (72.3 to 110.2) | 94.7 (86.3 to 103.0) |

| Host CPU per photo | Small | 20 MP |
|---|---|---|
| API container (8 uvicorn processes and their ExifTool children) | 18.6 to 21.3 ms | 84.9 to 109.1 ms |
| PostgreSQL | 1.0 to 1.5 ms | 3.7 to 6.5 ms |
| OpenSearch | 6.5 to 14.3 ms | 7.4 to 17.6 ms (44 at 1 client, noisy) |
| Stub engine (its own copy of the photo) | 0.5 to 0.6 ms | 9.1 to 11.7 ms |

At 32 small-photo clients the API used 10.7 cores, OpenSearch 4.5, PostgreSQL 0.65; no store queue
was seen. The API processes are the bound. Real photos give about 7 objects, not 2, so real API cost
is higher. Not measured yet: where the API's CPU goes (M1), the 30-process and 1-process grids (M2),
set P rows.

### 1.2 What the API's CPU is made of (what a Rust server can and cannot remove)

| Part | Number | Source |
|---|---|---|
| ExifTool execution per photo (capture time, place, orientation, camera) | 10.5 ms per COCO photo, 33.8 ms per 20 MP photo (wall time inside ExifTool per read, `-fast -G -n`) | ingest timeline section 5. Measured with the earlier feed by temporary file, whose write added 2.8 and 19.6 ms; the pipe feed replaced it |
| CPU outside the Python interpreters at small/32 | at least 2.7 cores of the 10.7 (8 interpreters use at most 8): at least 5.5 ms per photo at 495 photos/s | arithmetic on the capacity rows; it holds the ExifTool children, gRPC and orjson native threads |
| Per-photo byte touches in the API | the whole file read into one `bytes`; 3 to 4 full copies on the engine call (inferred); the ExifTool pipe feed; on uploads, multipart parse to a spooled file, a read into memory and a write again | inventory section 6.3, read in code |
| Not size dependent | `imohash` (about 48 KiB sampled), documents, vectors, rows | inventory section 6.3 |

Read plainly: dropping the interpreter lock removes the interpreter work and the copies. It does not
remove ExifTool's own cost, which is the largest single part at 20 MP. Section 3.4 states the gain
as an estimate with these parts.

### 1.3 What a native engine needs from the host (plan 17 sections 0.1 and 0a)

| Fact | Number |
|---|---|
| Engine host CPU per photo, end to end | COCO 7.3 to 7.7 ms; 20 MP 92 to 103 ms (nvJPEG's host phase) |
| Card ceiling, plan F (detector + frame encoder) | COCO about 196 photos/s; 20 MP about 125 (138 measured at 128 in flight) |
| Card ceiling, plan C (ingest plan with crop vectors) | COCO 52 to 65 (arithmetic); set P 63 to 71 |
| Host | two Xeon E5-2680 v3 (Haswell), 24 physical cores |

Each millisecond of API CPU per photo costs 0.14 cores at 140 photos/s (two cards, 20 MP, plan C).
On 20 MP photos, on two cards and on any cloud host with few cores (section 5), the API's CPU per
photo decides whether the GPU can be fed (revision 0's budget table, arithmetic on 1.1 and 1.3, kept:
one card plan C at 70/s fits in 24 cores; two cards plan C at 140/s needs 27 to 32 and is host-bound).

## 2. Decisions

### 2.1 The target architecture

```text
 browser / API clients ──HTTP──► op-server "api" (Rust, axum) ──tonic──► op-frame-engine (Rust, one per card) ──► Triton
                                  │  upload receiver ─► landing zone (local fast disk) ◄─ reads by key (plan 17 item)
                                  │  ingest pipeline, ExifTool -stay_open pool by path
                                  ├──► PostgreSQL: record, jobs, events (NOTIFY is the wake-up)
                                  └──► object store (object_store crate: local directory or S3-compatible)
 op-server "worker" (Rust job runner)        op-server "periodic" (Rust ticks, reaper)
 Python workers (one claim loop each, one subprocess per job): train, model_export, bakeoff, probe, promote, embedding_viz
```

| Question | Decision | Why |
|---|---|---|
| One binary or several | one server binary `op-server` with roles as subcommands (`api`, `worker`, `periodic`, `migrate` from R12, `all` for a one-host install); each role in its own container; the engine stays its own binary, one process per card | one image and one code base for the CPU side; roles scale apart; the engine stays a pure function with no database (plan 17 section 1.3) |
| Processes per host for the API | one process with a tokio multi-thread runtime; no `WEB_CONCURRENCY`. Two or more `api` replicas behind a load balancer are supported: every per-process pool (ExifTool, admission permits, database pool) is sized per process and no state is shared in memory | no interpreter lock; one pool of each kind; the failure domain is answered in section 4.6 |
| Where ingest runs | in the `api` role, in-process; in the `worker` role for bulk and source jobs (the same crate) | the cost is per photo in the API; a separate ingest service adds a copy and a boundary |
| The engine and storage | the engine reads local files only and never talks to S3; uploads land on local fast disk, the engine reads them there; a mover makes them durable in the object store (section 4) | plan 17's engine has no database and no credentials; reading a photo once from local disk is its cheapest input |
| Job queue | a Rust runner on the existing `jobs` and `job_attempts` tables (plan 02b semantics: claim, lease, fencing, exclusive groups, pause, yield, `auto_resume`, the reaper); the Python jobs that stay use a small claim loop over the same tables, one subprocess per job. No Celery, no broker, no `jobs-dispatch`, no `runner` field, no Flower | decision 2; section 2.5 |
| Valkey | leaves the stack in R3b (18b) unless M17 shows a need PostgreSQL cannot meet; events reach the browser from the event rows with `NOTIFY` as the wake-up | decision 2; section 2.5 |
| What stays Python | section 2.3 | decision 3 |
| HTTP framework | axum 0.8.9 with tower-http 0.7.1 | section 2.2 |
| Database driver | tokio-postgres 0.7.18, deadpool-postgres 0.14.2, pgvector 0.4.2 (features `postgres`, `halfvec`) | section 2.2 |
| Object storage | one interface: the `object_store` crate 0.14.2 (local directory and S3-compatible), no second mechanism beside it | decision 4; section 4 |
| Search | SQL on the record (gate A) and the vector index gate B picks; if gate A or B keeps OpenSearch as a derived store, the `opensearch` crate 2.4.0 | no OpenSearch code is ported before gates A and B decide (R2b, 18b) |
| Contract | decided by the contract spike M15 between code-first (utoipa) and contract-first (section 2.4) | decision 10 |
| Metrics, logs | prometheus-client 0.26.0 (as the engine), same metric names and labels as today; tracing 0.1.44 with tracing-subscriber 0.3.23 JSON output, same field names (`request_id`) | |
| Workspace | one Cargo workspace, `native/`, engine and server crates side by side; the crate list, owners and rules are in 18a section 2 | 18a section 2.1 gives the reasons |

### 2.2 Library choices, each against its alternative

Versions are the latest stable on 2026-11-15, read from the crates.io records
(`https://crates.io/api/v1/crates/<name>`) or the docs.rs page named, pinned exactly in
`native/Cargo.lock` as plan 17 does. `cargo deny` with plan 17's allow list (`native/deny.toml`) is
the licence proof at each lane's first commit.

| Need | Choice | Alternatives and why not | Documentation read |
|---|---|---|---|
| HTTP server | axum 0.8.9, tower-http 0.7.1 (request id, CORS, trace, body limit, timeouts), hyper-util 0.1.21 (already in `native/Cargo.lock`) | actix-web 4.19.0 brings its own runtime model and middleware; axum uses `tower::Service`, the stack tonic already uses for the engine client | https://docs.rs/axum/0.8.9/axum/ |
| Streaming uploads | `axum::extract::Multipart` (`next_field`, then `Field::chunk()` per part); `DefaultBodyLimit::disable()` on the upload route only, replaced by tower-http's `RequestBodyLimitLayer` at the served byte cap and its `RequestBodyTimeoutLayer` as the idle-body limit (its deadline resets per frame) | buffering extractors stop at 2 MB by default and hold the body in memory | https://docs.rs/axum/0.8.9/axum/extract/multipart/struct.Multipart.html ; https://docs.rs/tower-http/0.7.1/tower_http/timeout/index.html |
| CPU-bound work in the server | a dedicated rayon 1.12.0 pool, results back through a oneshot channel | `spawn_blocking`: tokio recommends "a separate thread pool dedicated to CPU bound tasks", naming rayon | https://docs.rs/tokio/1.54.1/tokio/index.html |
| PostgreSQL | tokio-postgres 0.7.18 (pipelining "when futures are polled concurrently"), deadpool-postgres 0.14.2, pgvector 0.4.2. Vector bulk loads: `COPY` into an `ON COMMIT DROP` temporary table, then one `INSERT ... SELECT`, because `COPY FROM` is refused on tables with row-level security (15a section 1) | sqlx 0.9.0's checked `query!` macros need a database or a committed `.sqlx` directory at build time; lanes run no database. A database-tier test prepares every registered statement instead (section 7) | https://docs.rs/tokio-postgres/0.7.18/tokio_postgres/ ; https://docs.rs/sqlx/0.9.0/sqlx/macro.query.html |
| Wake-ups | `LISTEN`/`NOTIFY`: one dedicated, reconnecting connection per process, outside the pool, since tokio-postgres delivers notifications only through the connection task the caller drives (`AsyncMessage::Notification`); it does not work through a transaction-mode pooler. Payload under 8,000 bytes; delivered only after commit; a 2 s poll covers a lost wake-up | Valkey pub/sub: a second store for a job PostgreSQL does | https://www.postgresql.org/docs/18/sql-notify.html ; https://docs.rs/tokio-postgres/0.7.18/tokio_postgres/enum.AsyncMessage.html |
| Row-level security | a `ProjectTx` type: repositories of project tables take `&mut ProjectTx`, made only by `project_tx(pool, project)`, which sets the scope as `src/db/session.py` does; the pool is private to `op_db` so a handler cannot open a bare transaction | a runtime check per call can be forgotten; a type cannot | (pattern; PostgreSQL row security as 02a and 15 R10) |
| Job queue | our own runner on the existing tables (section 2.5) | section 2.5's library table | https://www.postgresql.org/docs/18/sql-select.html (`SKIP LOCKED`) |
| Object storage | object_store 0.14.2 (MIT or Apache-2.0, 2026-09-15): `LocalFileSystem::new_with_prefix` and `AmazonS3Builder` (`with_endpoint` for S3-compatible servers, `with_allow_http` for a local test server, `with_url` for Cloudflare R2, `with_checksum_algorithm` for an integrity check on upload, `with_conditional_put`). "All operations are atomic, and readers cannot observe partial and/or failed writes"; `put_multipart_opts` writes "without buffering the entire object in memory"; `PutMode::Create` fails with `AlreadyExists` on an existing object; `Path::parse` refuses `.`, `..`, empty segments and control characters. `LocalFileSystem` follows symlinks that leave its root, so the key layout never creates one and local read paths are checked (section 4.4). The 0.13 upgrade guide renamed `copy`, `rename` and `delete` to `copy_opts`, `rename_opts` (default: copy, then delete) and `delete_stream`; the lane reads it | `aws-sdk-s3` 1.159.0 (Apache-2.0): S3 only, a second mechanism for the local store; cap-std 4.0.3: local only. Decision 4: one interface | https://docs.rs/object_store/0.14.2/object_store/ ; https://docs.rs/object_store/0.14.2/object_store/aws/struct.AmazonS3Builder.html ; https://docs.rs/object_store/0.14.2/object_store/local/struct.LocalFileSystem.html ; https://docs.rs/object_store/0.14.2/object_store/path/struct.Path.html ; https://docs.rs/object_store/0.14.2/object_store/enum.PutMode.html |
| Content hash on write | SHA-256 while streaming, `sha2` 0.11.1 (already pinned in `native/Cargo.lock`); the host's Haswell CPUs have no SHA instructions, so M18 measures the cost at 20 MP | BLAKE3 is faster in software but 15a's `photos.content_sha256` is SHA-256 | https://docs.rs/sha2/0.11.1/sha2/ |
| Duplicate probe | a port of py-imohash (about 30 lines: 16 KiB samples at start, middle and end, MurmurHash3 x64 128, size as a varint over the first bytes), checked against goldens from the Python `imohash` on the standard lists; `murmur3` 0.5.2 (MIT or Apache-2.0, last release 2022-10-10) for the hash, or a port of the reference function if the goldens disagree | the `imohash` crate 0.1.2 (last release 2024-08-28, about 10,000 downloads): the port is the default | https://crates.io/api/v1/crates/murmur3 ; https://crates.io/api/v1/crates/imohash |
| ExifTool | a driver for `exiftool -stay_open True -@ -` by path, written here (about 200 lines, estimate), as `src/clients/exiftool_client.py` is today | `exiftool` 0.3.1 and other wrappers have few users. `exiftool-rs` 1.0.0 (2026-10-30, GPL-3.0-or-later: compatible with AGPL, but outside `native/deny.toml`'s allow list, so an explicit `cargo deny` exception would be needed) is not chosen | https://crates.io/api/v1/crates?q=exiftool |
| Capture fast path (candidate, M16) | kamadak-exif 0.6.2 (BSD-2-Clause, 2026-11-05): reads Exif "directly from some image formats including TIFF, JPEG, HEIF, PNG, and WebP" through `Reader::read_from_container`; time, GPS, orientation and camera tags on JPEG. Accepted only if M16 shows equal values to ExifTool on the corpus; ExifTool stays for the formats and tags the fast path does not cover | nom-exif 3.8.0: licence "non-standard" on crates.io | https://docs.rs/kamadak-exif/0.6.2/exif/ |
| Text tokens for text search | tokenizers 0.23.2, checked token for token against today's tokenizer | | https://crates.io/api/v1/crates/tokenizers |
| HTTP clients (segmenter, VLM, model-server HTTP) | reqwest 0.13.5 | | https://crates.io/api/v1/crates/reqwest |
| Graph and assignment (near-duplicate groups, lineage) | petgraph 0.8.3 (MIT or Apache-2.0: connected components, union-find); pathfinding 4.16.0 (Apache-2.0 or MIT: `kuhn_munkres_min`, the Hungarian method; weights must be `Ord`, so float costs are scaled to integers, checked against SciPy's `linear_sum_assignment` on recorded inputs) | | https://crates.io/api/v1/crates/petgraph ; https://docs.rs/pathfinding/4.16.0/pathfinding/kuhn_munkres/fn.kuhn_munkres.html |
| k-means and kNN at scale | decided by M12 and M13: matrix-multiply kernels in our own GPU crate first; cuvs 26.10.0 (Apache-2.0) only if its Rust API is proven on the card; linfa-clustering 0.8.1 on CPU as the small-set path | section 2.3 | review evidence, section 2.3 |
| Tests | proptest 1.12.0; testcontainers-modules 0.15.0 (PostgreSQL) for the database tier | cargo-fuzz needs nightly, against the pinned stable toolchain | https://crates.io/api/v1/crates/proptest |

### 2.3 What stays Python (decision 3), and the rule for everything else

| Group | Verdict |
|---|---|
| `probe`, `train`, `model_export`, `bakeoff`, `promote` | stay Python: Ultralytics, PyTorch, TensorRT and ModelOpt, ONNX Runtime, `onnx` have no Rust equivalent |
| `embedding_viz` (UMAP) | stays Python until a product decision replaces the method: annembed 0.1.6 is "à la UMAP" and no maintained exact port exists; M14 measures map quality so the owner can decide |
| `cluster_run` (both methods), `item_scores` (uniqueness), `near_dup_score` and `near_dup_core_group` (exact all-pairs kNN), `persons_cluster` (face kNN) | Rust if M12 or M13 passes on 100,000 and 1,000,000 real crop vectors; else Python reading and writing its own result tables. The decision falls before R6 and R8 start, so neither language's version is written and then replaced |
| `score_rows`, `region_fp_centroids`, `selection` (k-center greedy), `near_dup_group`, `near_dup_rescore`, `group_lineage`, `raw_label_clusters` (`autolabel`), the `assist_loop` splitter, `embedding_reindex`, `reprocess`, the VLM client jobs and the VLM worker loop | Rust in their slice ("straightforward" in the review's table: a GEMM, a farthest-point loop, union-find, the Hungarian method, average or complete linkage over at most 2,000 members, HTTP and gRPC calls), each checked for equal output on recorded inputs |
| Segmenter and vLLM servers | stay, in their own images |

One rule, which replaces revision 0's two (review minor 9): every job not in the stay list above is
Rust; the jobs that stay never fold opinions, so no internal write route for Python is built. A
Python job reads and writes only the job tables and its own result tables (runs, scores,
projections), through `src/db`; `rg "apply\("` at each family's entry confirms it. Roadmap 14 section
4's "the VLM stays Python" becomes "the VLM server stays Python".

### 2.4 The contract and the frontend (decision 10)

The frontend imports `contracts/ts/itemWire.ts` and reads `contracts/openapi/curation.json` in 20
contract tests key by key; `frontend/e2e/harness/stub.py` builds stubs from it. The contract today is
FastAPI's `app.openapi()` filtered to `/curation` (OpenAPI 3.1.0, 759 components, 388 operations).
The 66 core route decorators have no contract; R0c (18a) writes `contracts/openapi/core.json` as
their baseline before R2a.

Two ways to keep the contract, decided by the spike M15 before R1a's contract part:

| | A. Code-first | B. Contract-first |
|---|---|---|
| Source of truth | Rust wire structs with utoipa 6.0.0 derives (utoipa-axum 0.3.0); `op_contracts` prints the document | the committed `contracts/openapi/*.json`; Rust wire types generated from it or checked against it by a test |
| During the port | `scripts/codegen/export_api_contracts.py` merges the Rust part and the FastAPI part into the one file | the file is frozen per family at its switch; the FastAPI generator stops writing a family when Rust owns it |
| Known limits | utoipa's serde support is partial; `discriminator` works only on untagged enums; `Option` becomes nullable (ToSchema docs). Whether it writes `prefixItems` (8 uses), `const` (42) and a discriminator with a mapping on a tagged enum (2) is unknown | `contracts/README.md` says the files are generated, never hand-edited; a change becomes an edit of the file plus regenerated types, which needs a rule and a tool |
| What it changes | a hook needs a built Rust binary: the `op_contracts` hook runs `cargo run --locked -p op_contracts` scoped to `native/**` and the contract files (cost measured in M15) | the drift hooks compare Rust types to the file; the generator of item wire and enum files moves to a Rust test |

M15 builds, in a scratch crate outside the merge, the two discriminator schemas, two tuple
(`prefixItems`) schemas and three `const` fields of `curation.json` both ways and compares them
with the normalised diff below. A passes when every one matches with no custom `ToSchema` impl or
with at most one documented impl per construct; else B.

Equality is by meaning, not bytes (decision 5, Q3 accepted with the review's conditions): (1) `oasdiff
breaking` (OpenAPI 3.1 support documented from v1.15.0) between the integration head and the new
file reports nothing outside the slice's intended-change list; (2) a normalised diff of each ported
operation and its schemas (titles dropped, the two spellings of an optional field made one, keys
sorted) is empty outside that list, and compares `prefixItems`, `const`, discriminators and nullable
forms directly, since oasdiff covers `prefixItems` only when added or removed
(https://github.com/oasdiff/oasdiff/blob/main/docs/OPENAPI-31.md); (3) the frontend's contract
tests, `npm run check` and `npm run build` pass unchanged. Error bodies keep FastAPI's shapes, which
`frontend/src/lib/api_errors.ts` reads (`{detail}`, `{detail: {error, message}}`, 422 `{detail:
[{loc, msg, type}]}`); the `msg` text of validation errors may differ, except "String should match
pattern '...'", which Rust emits word for word. The generated files `item_wire.json`, `itemWire.ts`,
`regionStatus.ts`, `classSources.ts`, `classNames.ts`, `keymap_actions.json` and the frame-engine
contracts stay generated from their Python sources until the slice that ports the source; then the
source moves to Rust and the files stay byte-identical (a test compares).

### 2.5 Jobs without Celery (decision 2)

The existing tables already hold the protocol: `jobs` (33 columns: status with 11 transitions guarded
by the trigger `jobs_status_transition_guard`, `fencing_token`, `owner`, `heartbeat_at`,
`lease_expires_at`, `retry_count`, `max_attempts`, `attempt`, `cancel_requested_at`,
`pause_requested_at`, `idempotency_key`, `parent_job_id`), `job_attempts`, `job_items`, and the partial
unique index `uq_jobs_one_active` for exclusive groups (02b section 1). `run_job`
(`src/tasks/runtime/job_task.py`) already claims a row by id in one statement, heartbeats and writes a
fenced terminal state; Celery only delivers the id and backstops time limits.

| Runner | Design |
|---|---|
| Rust (`op_jobs`, R1j in 18a) | claim: `SELECT ... FROM jobs WHERE status = 'queued' AND kind = ANY($kinds_this_process_handles) ... FOR UPDATE SKIP LOCKED LIMIT n` ("can be used to avoid lock contention with multiple consumers accessing a queue-like table", PostgreSQL docs); wake-up: `NOTIFY op_jobs` in the transaction that queues a row, `LISTEN` on the dedicated connection, 2 s poll backstop; lease and heartbeat with the fencing token; terminal states, cancel, pause, yield, `auto_resume`, exclusive groups as 02b; the reaper and ticks in the `periodic` role with the existing PostgreSQL advisory lock for a single ticker |
| Python (the stay list of 2.3, R3a in 18b) | one claim loop per worker image (`src/tasks/claim_loop.py`): the same claim query for the kinds that process registers, `LISTEN` for wake-ups, then `run_job(job_id)` in one subprocess per job (replacing Celery's `max-tasks-per-child` and `max-memory-per-child`); the loop kills the child at the kind's `hard_limit_s`; `SoftTimeLimitExceeded` leaves `job_task.py`. The trainer, export and evaluator images replace their Celery consumer entry with the same loop |
| Which runner takes a kind | the process that registers a handler for it; no `runner` column. A kind moves to Rust in the same slice as every route and tick that queues it |
| Removed (R3a) | Celery and kombu configuration, `src/tasks/app.py` and the task registrations, `jobs-dispatch` (never built), `dispatch()`'s broker send, `task_queue`, `redis_transport`, `event_wakeup`, the reaper's re-dispatch rule, `jobs-beat`, `periodic-worker`, `flower`. Columns `dispatch_count` and `last_dispatched_at` lose their writer and leave in R12's baseline schema |

Libraries checked (crates.io, 2026-11-15; the review's table): apalis 1.2.0 (apalis-postgres 1.1.0 shows
licence "non-standard" on crates.io while its README says Apache-2.0 or MIT; own `apalis` schema),
graphile_worker 0.14.2 (own schema), underway 0.2.0 (16 months since release, own tables), sqlxmq
0.6.0 (own tables, tied to sqlx), pgmq 0.33.7 (a queue table per name, no state machine), fang 0.11.0
(own tables), rusty-sidekiq 0.15.2 (Redis semantics), celery 0.5.5 (2023, unmaintained). None runs
the lease, fencing token, `interrupted` versus `lost`, exclusive groups, pause, yield and
`auto_resume` on the existing columns, which the API's job route and the Python claim loop read: our
own runner (about 1,500 lines, estimate) is justified.

Valkey after Celery: its remaining uses are the event wake-up (becomes `NOTIFY op_events`, R3b), the
image-counts dirty set (never ported: plan 15's design deletes it), reachability probes and the
reaper's queued marker (both become rows or disappear). PostgreSQL's documented limits for
`NOTIFY`: a payload under 8,000 bytes, one queue of 8 GB, delivery only after commit. Commit-time
serialisation of notifying transactions is reported for PostgreSQL's implementation but not stated
on the docs page; M17 measures it at three times the target commit rate. Valkey stays only if M17
fails and a coalesced wake-up (at most one `NOTIFY` per process per 50 ms; the event rows stay the
truth) also fails.

## 3. The order of the work

### 3.1 Stages, in the owner's order

| Stage | Owner's step | Slices (18a, 18b) | What the person gets |
|---|---|---|---|
| 0, now | (1) runs in plan 17; (2) starts | R0 measurements and tools (R0w, R0c, R0d, R0e, R0g, R0x, M15); R1s storage layer; R1a server skeleton and front; R1j job runner core; R1e engine client; R1x ExifTool pool and hashing; R1u upload receiver up to the landing zone | nothing changes for the person; the pieces of the Rust ingest are built and tested |
| A | (2) | R3a ticks in Rust, Python claim loop, Celery leaves; R3b live events from Rust, Valkey leaves; R1b the record and the fold in Rust; R11a projects on the record; R1c Rust ingest switches (uploads and folders; S3 storage); R2a inference routes on the engine | ingest on Rust at the card's rate; uploads stream to storage; `/detect`, `/faces`, `/embed`, `/ocr`, `/analyze` on the engine |
| B | (2), (3) | R4 gallery reads; R5 labelling acts, review, counts; R5e erasure; R11 configuration, settings, models | the gallery, labelling and settings work again on Rust; the gallery frontend update (photo-library plans) builds on R4 |
| C | (4) | R6 vectors, search, near-duplicates, holdout; R7 regions; R8 clustering; R9 import, combine, export, training inputs; R10 machine helpers and re-runs; R2b core data on the record, OpenSearch leaves | every family on the record; one database when gates A and B say so |
| D | (2) closes | R12 the Python API removed; the schema leaves Alembic | one backend language on every request path |
| E | (5) | the cloud benchmark (section 5) on the deploy shape (section 6) | a measured cloud deployment, behind authentication that a later plan builds |

R2b deletes OpenSearch only when gates A and B decide (decision D4 below), and it sits in stage C, so
the cloud north star's one database arrives in stage C, not earlier (review major 9). The cloud
benchmark's ingest half (B1, B2, B3) can run after stage A; B4 needs R4.

### 3.2 The storage plans (decision 1: every write path is built in Rust directly)

No new Python storage slice is written. Kept from plans 15, 15a, 15b and 15c, and binding: the schema
(15a); migrations 0010 to 0018 as SQL; the transaction and lock-order design (15a section 7); the read
shapes (15b); the claim kinds and the fold's truth table as data; the key layout (section 4.3 here,
from 15 section 8); every measurement (15b M1 to M10, gate A, gate B). The Python step 1 on
`int/storage` (fold, claims, repositories, `apply()`, the 45,000 differential sequences over 2,000
seeds of `tests/domain/test_fold_differential.py`; 3,137 lines of domain and 1,457 of repositories
and `apply.py`) is the oracle: R0x exports its cases as data, the Rust fold must pass them, and it is
not extended. The Python backend on `int/b4-stateless` stays as it is (its current store, its tests)
as the reference and the differential-parity partner until each family's Rust slice replaces it; it
gets fixes only. 18b section 6 maps every 15c slice to the Rust slice that now builds its design, and
says what of it is kept as design, kept as SQL, or no longer written.

Schema ownership until R12: Alembic stays the single migrator, so one tool writes the schema of one
database. Each of 0010 to 0018 is a committed SQL file under `src/db/migrations/sql/`, run by a
ten-line Alembic revision; R12 replaces the chain with one baseline SQL (wipe and re-ingest allowed).
Rust database tests get the schema from a committed dump (18a section 3.3).

### 3.3 Windows: what does not work when, and what the full sweep runs against

The Rust slices merge into one integration branch for the programme (question Q1 proposes
`int/rust`: `int/b4-stateless` with `int/native` merged). The reference stack is `int/b4-stateless`
at a pinned commit.

| Window | On the integration branch, works | On the integration branch, does not work | The full sweep (once per large merge) runs |
|---|---|---|---|
| W0: stage 0 (until R1c merges) | everything as today: Python serves every route, through the Rust front once R1a merges | nothing new is broken | the offline suite, the database tier, the live and browser tiers through the Rust front, `--all-files` hooks, the cargo gates and the Rust database tier (jobs on the schema dump); the Python suite of `int/b4-stateless` only when it takes a fix |
| W1: after R3a | as W0; jobs run on the Rust runner and the Python claim loop | nothing new is broken (every tick and kind keeps a runner) | as W0, plus the job conformance suite (18a section 3.5) and `test_job_load.py`, `test_queue_delivery.py`, `test_job_recovery_real_handlers.py` live |
| W2: after R1c (the record) | ingest (upload, batch, objects, bulk, sources) on Rust into the record; core routes; configuration documents, settings, models, VLM registry; jobs | every curation family that reads or writes the search store shows nothing for photos ingested after the switch: gallery, lists, labels, review, search, regions, clusters, near-duplicates, import, combine, export, training inputs, machine helpers. Accepted by decision 1, as plan 15c accepted "families not yet moved do not work between slices" | the switched families: their live tests through Rust, their differential cases against the reference stack (brought up once per sweep), their Rust database tier; the families not yet switched: their live tests listed in `tests/pending_rust.txt` (strict expected failure, 18a section 3.4), and their behaviour stays proven by the reference stack as the differential partner; unaffected tests must pass |
| W3 onward: each family slice | each switched family works again, on the record | the families still listed in `tests/pending_rust.txt` | as W2; each family slice removes its ids from the pending list and must leave it shorter |
| After R12 | everything, one language | nothing | the whole suite on Rust; the differential ledger closed |

### 3.4 The gain of the Rust ingest, as an estimate with its parts (decision 7)

| Part of API CPU per photo | Small (today 18.6 to 21.3 ms) | 20 MP (today 84.9 to 109.1 ms) | After R1c (estimate) |
|---|---|---|---|
| ExifTool execution | 10.5 ms (ingest timeline) | 33.8 ms | unchanged while ExifTool stays on the hot path; less if read by path costs less than the measured feed (M5); near 0 for JPEG if the fast path passes M16 |
| Interpreter work and photo copies | the rest: about 8 to 11 ms | the rest: about 51 to 75 ms | Rust server work 2 to 6 ms (estimate, revision 0's figure, unmeasured) |
| SHA-256 on write (uploads only) | about 165 KB per photo | 6.5 MB per photo | M18 measures; a software SHA-256 on this host is a material share at 20 MP (estimate) |
| Floor, ExifTool kept | | | small about 12 to 17 ms; 20 MP about 40 to 60 ms (estimate, upper end with SHA-256 on uploads) |

So the go bar of R1c, API host CPU per photo at most 0.5x Python's (18b R1c), is plausible at 20 MP
and doubtful on small photos unless M16's fast path or decision D2 takes ExifTool off the hot path.
At 140 photos/s on two cards at 20 MP the estimate frees about 3.5 to 9.7 cores (arithmetic on 25 to
69 ms saved per photo). M1 and M5 run first; M16 is the candidate that changes the small-photo floor.

### 3.5 Decisions the measurements settle

| # | Decision | Branches | Settled by |
|---|---|---|---|
| D2 | Does ExifTool stay on the hot path? | keep (M5 under 3 ms CPU per photo at 20 MP, an estimate threshold); else the hot path reads time, orientation, GPS and camera in Rust (M16's reader) and the full ExifTool read becomes a queued job, agreed with the gallery metadata plan ([05](../photo_library/05_p2a0_metadata.md), which owns the reader) | M5, M16 |
| D3 | API process count until R1c lands | if photos/s at 20 MP rises above the plan C ceiling with CPU per photo unchanged at 30 processes, set the compose default from the grid (configuration only) | M2 |
| D4 | Does OpenSearch survive as a derived store? | no (R2b deletes the client and the service); yes (R2b and R6 use the `opensearch` crate behind the vector-index or search-copy interface) | gate A, gate B |
| D5 | Numeric kinds in Rust or Python | per section 2.3 | M12, M13 (and M14 for the owner's product decision on the 2-D map) |
| D6 | Contract mechanism | A or B of section 2.4 | M15 |

Revision 0's D1 (a Python half of "the engine reads by key") is dropped (decision 6): no Python
change that R1c replaces is written. Reading by key is an engine protocol addition for plan 17
(section 12).

## 4. Storage and the upload path (decision 4)

### 4.1 Today, read in code

The frontend (`frontend/src/lib/api_ingest.ts:40-52`) sends `POST {prefix}/ingest/upload` as
multipart: one `images` part per file, then `image_paths` (a JSON list of identifiers), `source`,
and the optional `run_id`. The planner (`frontend/src/lib/ingest/uploadPlanner.ts`) packs files up to
the served `upload_max_images_per_request` (128) and `upload_max_bytes_per_request` (512 MiB,
`src/config/curation.py:129-130`); two requests run in parallel (`ingestConfig.ts:35`). Accepted
extensions are `.jpg`, `.jpeg`, `.png` (`src/config/curation.py:131`). The route answers once every
photo of the request is done, one result per photo. Uploads land at
`<upload_root>/<imohash[:2]>/<imohash><ext>` (`persist_uploaded_bytes`, `image_serving.py:152`), under
the project's upload root (`src/config/projects.py:137`), named by the sampled hash; the Python `BlobStore` protocol (`src/clients/storage/base.py`) writes whole bytes (`write_atomic`),
with no streaming and no hash check; `DISK_GUARD` (`src/core/disk_space.py`) refuses uploads when
free space is low.

### 4.2 The flow per photo in Rust (R1u, R1s, R1c)

1. **Receive.** `next_field()` yields one part at a time. The receiver writes the part's chunks to a
   new file in the landing zone (`landing/<upload_id>/<n>.part`, local fast disk), computing SHA-256
   as the bytes pass. A permit of the bytes-in-flight bound is taken when the part's first bytes
   arrive, not before, so a slow client holds no permit while idle; the idle-body limit is tower-http's
   `RequestBodyTimeoutLayer` (deadline reset per frame). `DISK_GUARD`'s thresholds apply to the
   landing volume before a part is accepted.
2. **Part complete.** The file is flushed; `imohash` is read from it (three 16 KiB reads); the file is
   renamed inside the landing zone to its content name `landing/<project_id>/<sha256>` (same
   filesystem, so a rename). The finished part goes to a spawned task through a bounded channel (its
   capacity is the photos-in-work bound); the receiver reads the next part meanwhile.
3. **Work, in parallel.** The content claim (duplicate check, record); the engine call (bytes read once
   from the landing file today; the landing key once plan 17 adds reading by key, section 12);
   capture through the ExifTool pool by path (or the fast path, M16); the mover (step 4).
4. **Make durable before the commit.** The mover writes the object to its content key in the object
   store with `PutMode::Create`: a local store renames within the same store when the landing zone and
   the store share a filesystem, else streams with `put_multipart_opts`; S3 checks integrity on upload
   with `with_checksum_algorithm`. `AlreadyExists` means the content is present: its size and hash
   are checked, never overwritten. The photo row is committed only after the object is durable, so
   the record never says "stored" before the bytes are (MAJOR 3: no rename or move after the commit).
5. **Commit.** One transaction per commit group (18b R1b, R1c). A duplicate's landing file is deleted
   and no object is written.
6. **After the commit.** The landing file is deleted. A crash between 4 and 5 leaves an object no row
   references; the reconcile tick (R3a's `upload_sweep`, extended) deletes such objects older than one
   hour and stale landing files, under the imohash lock rule of 4.5.

Back pressure: when the photos-in-work or bytes-in-flight bound is full the route stops polling the
body; hyper stops reading the socket and TCP flow control slows the client. A request that cannot
start inside its admission bound answers 503 with `Retry-After`, as the engine-unavailable answer does
today.

Server-side paths (`ingest/batch`, bulk and source jobs) skip steps 1, 2 and 4: the source root is read
in place and never written (plan 15 section 8); `photos.storage_key` is `source/<relative path>`.

### 4.3 The key layout (by project and content)

| Kind | Key | Notes |
|---|---|---|
| Original of an upload | `projects/<project_id>/originals/<sha256[0:2]>/<sha256>` | the project id, not the slug (a slug can be renamed); content addressed, so a retry or a second upload of the same bytes writes nothing |
| Other files of a photo (`photo_files`: motion, edited, raw, sidecar) | `projects/<project_id>/originals/<sha256[0:2]>/<sha256>` of that file | the same rule |
| Derived files | `projects/<project_id>/derived/<kind>/<photo_key>/...` with kinds `crops`, `thumbs`, `exports`, `training`, `snapshots` | derived files are separate from originals and deleted by prefix; crop and thumbnail entries are a cache that can be regenerated from the original |
| Source roots read in place | `source/<relative path>` | never written or erased (plan 15 section 8) |
| Models, bake-off results, caches | `models/`, `bakeoff_results/`, `cache/` as today's roots (`src/domain/storage_keys.py`) | unchanged roots; the `upgrade` root goes with the 0.6.0 upgrade code |

`photos.content_sha256` is filled for every upload (computed while streaming), which 15a allows (the
column is nullable and unique per project); `imohash` stays the cheap duplicate probe.

### 4.4 Local read paths

The engine and ExifTool need a file path. `op_storage` resolves two kinds of local key: landing keys
and `source/` keys. Each is parsed with `object_store::path::Path::parse` (which refuses `.`, `..`,
empty segments and control characters), joined to its configured root, canonicalised, and refused
unless the result is inside the canonical root (symlinks that leave a root are refused, since
`LocalFileSystem` would follow them). A property test generates keys and asserts no resolved path
leaves its root.

### 4.5 Delete, erase and shared files

- Delete of a photo (trash) is a claim (plan 15 answer 4); emptying the trash is erasure.
- Erasure of a photo (R5e in 18b, on 15a section 8's design): the record rows go under the
  `erasure_jobs` guard; then the files: the original if the project owns it (uploads; never a source
  root), `photo_files`, every derived prefix of the photo, exports and training copies listed by 15a
  section 8, the holdout snapshot rewritten without it. `op_storage` provides the primitives (R1s):
  delete by key, delete by prefix, both idempotent.
- Shared files: originals are keyed per project, so erasure in one project never touches another
  project's bytes. Inside a project, one stored object can be referenced by more than one row (a photo
  and a `photo_files` row with equal bytes, a holdout snapshot, an export). Before an object is
  deleted, the erasure job checks by query that no row in the catalogue's file-key columns still names
  it; the check and the delete run under the same `(project, imohash)` advisory lock that the ingest
  commit takes, and the ingest commit checks the object exists (a `head`) after taking that lock and
  writes it again from the landing file if not. So a concurrent upload of the same bytes and an
  erasure never leave a committed photo without its object. This is decision 4's "shared-file reference
  counts" as a reference check by query, not a stored counter (question Q2).

### 4.6 Failure domain and replicas

One `api` process per host means a crash or out-of-memory kill drops every in-flight request of every
project (today 8 processes share that load). The client retry is safe: content addressing makes a
repeated photo a `duplicate` and its object already present. Two or more `api` replicas may run behind
a load balancer; each sizes its own pools and permits. Risk row in section 9.

### 4.7 Wire, formats and the frontend edit

No wire change is needed for speed. One frontend edit ships with R1c: `api_ingest.ts` appends `source`,
`image_paths` and `run_id` before the `images` parts, so each photo can commit as soon as it is
processed; a client that sends them last still works and its commits wait for the end of the body.
`scripts/curation/ingest_upload.py` gets the same order. The accepted extensions do not widen in R1c:
HEIC, TIFF and WebP are added by changing the configured list when the engine's format corpus accepts
them (plan 17 N1a) and M5's format rows are known (configuration, no code). The default of two
parallel requests is reset from M10 (a frontend constant, not a contract).

## 5. The cloud benchmark (decision 8)

### 5.1 What is measured, as requests with expected results

Photos arrive as uploads through `POST /curation/projects/{project}/ingest/upload` as the frontend
sends them (section 4.7). Processing is the library plan: detector, frame vector, crop vectors, faces,
text lines, capture time and place.

| Id | Metric | How |
|---|---|---|
| B1 | Photos per second per GPU, end to end | photos committed / wall seconds of the steady window / GPUs; failed photos counted and must be 0 |
| B2 | Photos per second per CPU core | B1 x GPUs divided by (a) the cores busy, summed over every container's cgroup `cpu.stat` in the window, and (b) the cores provisioned |
| B3 | Time to "stored and findable", p50 and p99 | server stamps per photo: `part_complete` (step 2 of 4.2) and `committed` (the commit group's commit), logged with the `image_id`; reported as committed minus part_complete. The client stamps the last byte it wrote for each part and the response time of each request; both are reported, the client view as a second line. Every photo then passes, once, after the request answered: `GET /curation/projects/{project}/images/{image_id}` answers 200 with its stored fields, and `GET /curation/projects/{project}/images/locate` places it in the default photo list; a photo with at least one object also returns one of its crops in the top 10 of `POST /curation/projects/{project}/search/image` with its own bytes. Photos with no object pass the first two. The record is transactional, so a photo that passes is findable from its commit; a vector index outside PostgreSQL (if gate B picks one) adds its lag, measured as a third stamp |
| B4 | "On the map" and "in a time group" | on the map: for each photo with GPS allowed by the location policy, `GET /curation/projects/{project}/map/cells` over a box around its position counts it (photos without GPS are counted apart, not failed). In a time group: there is no route yet; until the timeline buckets of the organise plan ([10](../photo_library/10_p2a1_organise.md)) exist, a SQL probe on the record's capture time stands in and is reported as "not a system check" |
| B5 | Cost per million photos | price per hour / (B1 x GPUs x 3600) x 1,000,000, compute only; separate stated lines for storage, requests, transfer and the database (5.3) |
| B6 | The same on this machine | B1 to B4 on one A6000 and on two; B5 not computed (owned hardware) |

The gallery plan's target "new photo visible after ingest: 30 s"
([00](../photo_library/00_photo_library_plan.md) section 10) is the starting bar for B3 and B4; the
owner sets the benchmark's bars after the first baseline.

### 5.2 Who sends the uploads

A compiled closed-loop client, `op-upload-loadgen` (R0g in 18a), sends browser-shaped multipart: the
field order of 4.7, up to 128 photos or 512 MiB per request, N requests in parallel, stamps per part
and per request, one JSON line per row as plan 17's `op-fe-loadgen` does. A Python client cannot
produce 80 photos/s of 6.5 MB (about 520 MB/s). It runs on its own host in the same zone in the cloud,
and on cores pinned apart from the stack on this machine.

### 5.3 The host is a parameter; `g6.4xlarge` is the named example

| Fact for `g6.4xlarge` | Value and source |
|---|---|
| Price | $1.3232 per hour on demand in us-east-1, Linux (https://ec2calc.cloud/instances/g6.4xlarge and https://cloudprice.net/aws/ec2/instances/g6.4xlarge, read 2026-11-15; the review also read about $1.00 spot and about $0.861 for one year) |
| Compute | 16 vCPU (AMD EPYC 7R13, so 8 cores), 64 GiB, one NVIDIA L4 with 24 GB |
| Disk and network | 600 GiB local NVMe (instance store, lost on stop); EBS up to 1,000 MB/s; network up to 25 Gbps (burst figure) |
| JPEG hardware | NVIDIA lists 4 JPEG decoders on the L4 (https://www.nvidia.com/en-us/data-center/l4/); nvJPEG's hardware backend "supports baseline JPEG images with single scan with 1 or 3 channels" and not 4:1:0 or 4:1:1 (https://docs.nvidia.com/cuda/nvjpeg/index.html). The docs name architectures, not the L4: a test on the card decides (plan 17 item, section 12) |
| Engine host CPU cap | plan 17's 92 to 103 ms per 20 MP photo were measured on Haswell; on this CPU the number is unknown, so the engine's photos/s cap from 8 cores is a parameter of the run, not "about 80" |

What each run records, beside the metrics: the landing disk (the local NVMe, which holds only the
landing window: the 1,000,000 run's 3.4 TB of originals go to S3); the S3 bucket and region; where
PostgreSQL runs (on the host on an EBS volume, or a managed service with pgvector `halfvec` and
row-level security) and its volume; the load-generator host type and zone; and the cost lines: S3
storage (GB-month), S3 requests (PUT and multipart requests per photo times the price per 1,000),
transfer (inbound and any cross-zone), and the database, each with the price read from the AWS
pricing pages at run time.

B5's arithmetic for the example: 367.6 / (photos/s) dollars per million (50 photos/s gives $7.35, 100
gives $3.68; the L4's rate is not measured).

### 5.4 Runs at product scale

| Run | Photos | Mix | Purpose | Expected wall time (estimate) |
|---|---|---|---|---|
| Regression | 2,000 per list: `op-bench-v1/coco2k` and `setp2k` (registry names) | one list per run | every slice that touches ingest | under 1 minute each at 40 to 120 photos/s |
| Headline | 100,000 | half set P (20 MP), half public small photos (COCO train2017, Open Images, Objects365 per the dataset registry) | B1 to B6 | 25 to 45 minutes at 40 to 70 photos/s |
| Scale | 1,000,000 | same mix | database growth, vector index build and size, memory, map and time views filling during ingest, B3 and B4 drift | 4 to 7 hours at 40 to 70 photos/s; under 3 hours at 120 |

The 100,000 run needs 50,000 distinct set P photos (a repeated photo answers `duplicate` without engine
work), so it draws distinct photos beyond the 2,000 list; the 1,000,000 run needs 500,000. Staging on
this machine: photos are read from NVMe, never from the NAS during a run; NVMe had 313 GB free on
2026-11-15; the 100,000 run needs about 340 GB of sources (50,000 x 6.5 MB plus 50,000 x 0.2 MB) and
as much again for stored originals; the 1,000,000 run about 3.4 TB of each. For both long runs the
client reads a rolling window staged ahead from the NAS to NVMe, the object store keeps originals on
a NAS root, and the landing zone stays on NVMe, which is the cloud design. M11 records the NAS rates
first; if the link binds the 1,000,000 run, that run reports the scale effects and the link as its
bound. Set P appears in results as numbers only; the public 1,000,000 part is fetched and entered in
the dataset registry before the run (a stewardship task, not a lane).

## 6. Deploy shape (G14, the parts this port changes)

| Part | After the port | Notes |
|---|---|---|
| CPU images | one `op-server` image (Rust binary, ExifTool and Perl, non-root, `-slim` base, multi-stage); one Python worker image for the stay list of 2.3, without FastAPI and uvicorn | sizes against today's API image (M8); a Trivy scan before publishing |
| GPU images | the engine (plan 17), Triton, the trainer, export, evaluator, segmenter and vLLM; unchanged here except the Celery entry replaced by the claim loop (R3a) | |
| CPU-only host | runs the server roles, PostgreSQL and the Python CPU workers: browse, label, search, cluster | ingest needs a GPU host with the engine and Triton |
| Split hosts | the engine reads the landing zone, so a GPU host runs the `api` role (or a receiver) next to the engine, with the landing zone on its local disk; the object store is shared | the reason the landing zone is local to the engine |
| Managed PostgreSQL | pgvector with `halfvec`, row-level security, the operator-created login role `op_vector` (15 section 14 answer 2); VectorChord, if gate B picks it, is rarely offered (risk) | the pool is sized per process; PgBouncer optional, but the `LISTEN` connection must bypass a transaction-mode pooler |
| Object storage | `object_store`: a local directory on this machine, S3 (or an S3-compatible server) in the cloud; uploads land on local disk first (section 4) | |
| Authentication | **entry condition for any deployment reachable from the internet**: none is built (14 section 9 answer 4). Until it exists the stack stays on loopback, a LAN or a private network | |

Left to a later deploy plan: authentication and authorisation (plan 10 F5), TLS termination,
orchestration, managed-database backups and point-in-time recovery, secret handling, autoscaling, the
GPU arbiter overlay.

### 6.1 Multi-user seams (decision 9: non-goals, each with the place it would attach)

Already in place: every project table has `project_id` with forced row-level security (15a), the scope
is set inside the transaction (`src/db/session.py:51`), and `ProjectTx` carries it into Rust. Nothing
below is built by this plan.

| # | Seam | State | Where it would attach |
|---|---|---|---|
| 1 | Account or tenant above the project | `projects` has no owner; slugs are global | `projects.account_id`, a transaction setting `op.account` the policies also check, a `Principal` extractor in `op_api` that `project_tx` takes |
| 2 | Project list and the global event stream | `GET /curation/projects` lists all; `/curation/events` carries every project's events | filter by principal in the list route and the stream |
| 3 | Deployment-wide registries | the VLM endpoint registry is deployment wide and may hold keys | per-account registry, or admin-only reads |
| 4 | Core library | the global indexes become one reserved core project in R2b | a core project per account |
| 5 | Files | keys under `projects/<project_id>/` (section 4.3) | an account prefix above the project |
| 6 | Vector tables | one table per project and space (`vec.v<lock_key>_<space_key>`) | thousands of tenants mean thousands of tables and index builds: measure the catalogue size before a public deployment; quotas on the creation job |
| 7 | Queues | one `jobs` table for all projects; exclusive groups per project | per-project round robin in the claim query; a per-account concurrency cap |
| 8 | Upload limits and back pressure | deployment-wide limits; admission bounds per process | admission keyed by project or account beside the global one, in the same tower layer |
| 9 | Metrics and logs | project label, request id | an account label; per-account log filters |
| 10 | Rate and quota | none in code | a tower layer in `op_api` before the extractors |
| 11 | Authentication | not built (owner) | the same layer; section 6's entry condition |

## 7. Gates for Rust code

Plan 17 section 3.1 applies unchanged, as it stands on `int/native` today: `cargo fmt`, `cargo clippy
-D warnings` with the workspace lints of `native/Cargo.toml` (`missing_docs`, `unsafe_op_in_unsafe_fn`,
rustdoc broken and private intra-doc links, `allow_attributes` and `allow_attributes_without_reason`
denied so every suppression is an `#[expect(.., reason = "..")]`, `cognitive_complexity` at 15,
`too_many_lines` at 100, `too_many_arguments` at 7 from `native/clippy.toml`, `unwrap_used` denied
outside tests, `missing_errors_doc`, `missing_panics_doc`), `cargo doc` with warnings denied, `cargo
test` with no GPU and no network, `cargo deny` licences, bans, sources (advisories at pre-push), the
jscpd duplication check over `native/**/*.rs`, the file-size ratchet, Dependabot. The server adds:

| Gate | Rule |
|---|---|
| No unsafe | `#![forbid(unsafe_code)]` in every server crate |
| Layering | `scripts/codegen/check_layering.py` gains a rule over `cargo metadata` that holds 18a section 2.2's dependency table |
| Contract drift | the drift hooks regenerate or check the contract by D6's mechanism; stale fails |
| Breaking changes | `oasdiff breaking` and the normalised diff of 2.4 against the integration head; anything outside the slice's intended-change file fails |
| Case contracts | the fold cases, plan-build cases, capture goldens and every other generated case file replayed by Rust tests; drift fails |
| Statements | a database-tier test prepares every statement of the registry against the head schema (wave sweep) |
| Property tests | the fold (any order of machine writes around a claim gives the same row), the filter grammar, the cursor codec, local key resolution (no key leaves its root) |
| Request edges | body limits, idle body, a slow reader, the 503 with `Retry-After`, the error shapes of 2.4 |
| Differential parity | the rule of 18a section 3.2 for every family; the ledger check fails a slice that deletes a Python test without a replacement row |

## 8. Measurements

Each runs in the coordinator's hold (host load under 16 at the start, recorded), writes raw files
under `artifacts_local/` and a summary row under plan 17 section 5's evidence clause. Lists by
registry name; set P as numbers only.

| Id | What | Bar or use | Before |
|---|---|---|---|
| M1 | Where the API's CPU goes: the capacity harness with its stdlib sampler (`APICAP_PROF=1`), 8 processes, stub engine, `coco2k`, `mp20` and `setp200`; per-PID `/proc/<pid>/stat` deltas split uvicorn from ExifTool; crop JPEG bytes per photo logged | fact-finding: the parts of 3.4 | R1c |
| M2 | 30-process and 1-process grids, same lists | D3 | R1c |
| M3 | Path-mode spike in a scratch copy (no product file touched): the API sends a path, the stub engine reads the file, capture by `read_capture_file`, imohash from the file | the share of API CPU that reading by key removes: the priority of plan 17's by-key item (section 12) | that item |
| M4 | Real objects per photo: the grid with plan 17's real pipeline, plan C | slope of API CPU per object | R1c |
| M5 | ExifTool's own cost: `-stay_open` over `coco2k`, `setp200`, `commons` and a format corpus (TIFF, PNG, HEIC, WebP), by path and by pipe; CPU per file and bytes read (`/proc/<pid>/io`) | D2 (3 ms at 20 MP, estimate threshold) | R1c |
| M6 | Core routes: `/detect`, `/embed/image`, `/faces/recognize` at 1, 8, 32 clients on `coco2k`, API CPU per request | R2a's go row | R2a |
| M7 | Read routes per family: the 80 shapes of 15b through HTTP at 8 clients: p50, p99, API CPU per request (on the reference stack) | the order inside stage B and C | R4 |
| M8 | Static facts: routes including `/v1` from `create_app()`; `pytest --collect-only -m live`; `python -X importtime -c "import src.main"`; today's API image size | slice sizes; the image baseline | R1a |
| M9 | Region-stage worker profile (`scripts/curation/worker.py --continuous`) | R7's priority | R7 |
| M10 | The Python upload route today with `op-upload-loadgen`: 128 photos, 512 MiB cap, concurrency 1, 2, 4, 8 on `coco2k` and `setp200`; photos/s, API CPU per photo, request latency | the baseline R1c must beat; the frontend default | R1c |
| M11 | NAS sequential read and write MB/s from this host | section 5.4 staging | the first scale run |
| M12 | k-means on 100,000 and 1,000,000 x 1,024 real crop vectors, k about 1,000 to 4,000: wall time and objective against faiss CPU and GPU, assignment agreement; our GPU matrix-multiply kernel, cuvs only if its Rust API covers k-means on the card, linfa on CPU for small sets | D5 for `cluster_run`: Rust when the objective is within the two-run spread of faiss and the time is not above faiss GPU by more than the spread | R8 |
| M13 | Exact all-pairs top-k on the same vectors and on face vectors: time and recall against faiss `IndexFlatIP`; uniqueness score rank correlation against faiss IVF | D5 for `near_dup_score`, `near_dup_core_group`, `persons_cluster`, `item_scores` | R6 |
| M14 | 2-D map quality at 128,000 points: trustworthiness and kNN preservation of annembed and of PCA against umap-learn, and time | the owner's product decision for `embedding_viz` | none (informs) |
| M15 | Contract spike (section 2.4): both mechanisms on the discriminator, tuple and `const` schemas; the hook's cost per commit | D6 | R1a's contract part |
| M16 | Capture fast path: kamadak-exif against ExifTool on JPEG time, GPS, orientation and camera tags over `coco2k`, `setp2k`, `commons` and the format corpus: equal values per file, CPU per file | accept the fast path for the formats and tags where every value is equal | R1c (optional: R1c ships without it) |
| M17 | `NOTIFY` at ingest commit rate: commit groups with one `NOTIFY` each at 3x the target commit rate, with listeners in 4 processes; commit p99 with and without, event latency | Valkey leaves when commit p99 rises by at most the two-run spread and event p99 is under 100 ms (estimate threshold) | R3b |
| M18 | Storage write cost per photo on this host: SHA-256 while streaming, landing write and rename, a local object store, and an S3-compatible server on loopback, at both sizes | B1's budget and the floor of 3.4 | R1c |

M12, M13 and M14 share one build of the vectors with the vector-index round and plan 16: 100,000 and
1,000,000 real crop vectors from the engine's crop encoder over the dataset registry's lists (set P as
numbers only). The early lanes of 18a section 5 need none of M1 to M18 to start, except R1a's
contract part (M15).

## 9. Risks and their early signals

| Risk | Early signal | Answer |
|---|---|---|
| The contract cannot be kept equal | the normalised diff reports a change on the first ported route | the intended-change file and a frontend edit in the same slice; D6's other branch |
| The Rust fold disagrees with the oracle | a case of R0x's export fails | the oracle wins unless the case is a recorded Python defect, listed with the owner |
| Two job runtimes in one table drift | a conformance case (claim race, lease expiry, cancel, exclusive group, pause) differs between the Rust runner and the Python claim loop | one conformance suite run against both (18a section 3.5) |
| A family's window lasts long (W2) | the pending list does not shrink between sweeps | the coordinator orders stage B by the person's acceptance walk |
| The port is large (175,623 lines of `src/`, 852 Pydantic classes) | lanes stop at concern boundaries without finishing | families split by concern; at most three lanes per wave |
| ExifTool, not the API, is the cost | M5 over D2's threshold | D2's branch, M16 |
| SHA-256 on write costs too much on this host | M18 at 20 MP | the cost is B1 budget; question Q4 |
| One process per host fails | an out-of-memory kill drops every upload in flight | client retry (safe by content address); memory bounded by the bytes-in-flight permit; two replicas allowed |
| `NOTIFY` serialises commits | M17 | coalesced wake-up; Valkey stays only if that fails too |
| Feature unification in one workspace changes server behaviour | R0w's test (18a) | question Q3 |
| Forwarded routes add latency | p99 of forwarded routes rises through the front | R1a's go bar; the forwarder is deleted in R12 |
| Metric names change | the metric-name diff fails | name and label checks per slice |
| The cloud host is CPU-bound | B2 shows 8 cores busy below the card ceiling | the hardware JPEG backend; API CPU per photo |
| OpenAPI 3.1 tooling gaps | oasdiff or Schemathesis misreads a construct | the normalised diff stays the gate |
| A key leaves its root | the property test | section 4.4 |

## 10. Not in this plan

Authentication and any public deployment; the gallery's views and their read shapes; the engine's
internals, its hardware-decoder use and its branches; the storage schema's content (15a is binding);
cluster, label and train logic (plan 16); training, export, bake-off and VLM serving in Rust; the
frontend beyond the edits named per slice; video; the desktop application; orchestration.

## 11. Questions for the coordinator

| # | Question | Recommended answer |
|---|---|---|
| Q1 | The integration branch for the Rust slices | `int/rust`, made from `int/b4-stateless` with `int/native` merged (`--no-ff`); `int/native` keeps the engine lanes until N3 and is merged into `int/rust` at each engine wave; the reference stack is `int/b4-stateless` at a pinned commit |
| Q2 | Decision 4 asks for "shared-file reference counts"; 15a (binding) has no counter table, and per-project keys leave sharing only inside a project | a reference check by query over the erasure catalogue's file-key columns under the imohash advisory lock (section 4.5); a stored counter would be a 15a schema change |
| Q3 | `frame_engine` enables serde_json's `arbitrary_precision` and `preserve_order` features (`native/frame_engine/Cargo.toml`); Cargo unifies features across the packages of one build, so `cargo test --workspace` builds the server with them too | R0w adds a test that round-trips an internally tagged enum and a `flatten` struct with numbers. Result: it failed, so the engine's plan reader is its own and the features are gone; one workspace stands (18a section 2.1) |
| Q4 | SHA-256 of every upload on a host with no SHA instructions | keep it (decision 4) and record M18; if it exceeds the B1 budget at 20 MP, the owner chooses between a faster content hash (a 15a column change) and the cost |
| Q5 | The engine-client lane shares one BYTES-tensor codec with `frame_engine_loadgen` (an edit of `native/frame_engine_loadgen/src/client.rs`, plan 17's crate) | yes, in one commit, with plan 17 lanes holding that file for that wave (18a section 2.3); the engine's own decoder stays separate on purpose |

## 12. Items for plan 17 (decision 6: written here, plan 17 is not edited)

| Item | Content |
|---|---|
| Read the photo by key (phase N2) | an optional input beside `IMAGE` (as `BOXES` is optional in protocol version 1), or a protocol revision: plan 17 decides. It names a landing or source key and the content claim (size, imohash); the engine resolves it under configured read-only roots, answers new statuses for a missing or changed file and plan faults for an unknown root or bad key, in the generator of `contracts/json/frame_engine_v1.json` and the plan cases. Compose mounts the landing zone and source roots read-only into `frame-engine` (no mounts today). Priority from M3 |
| Hardware JPEG decode on the L4 | a test on the card (section 5.3) |
| The proto copy | if `gpu_frame` leaves the workspace, the one copy of the KServe protos (`native/gpu_frame/proto`, read by three crates' `build.rs`) moves to a neutral directory first |
| serde_json features | Q3 |

## 13. Review trace

| Finding | Resolved in |
|---|---|
| BLOCKER 1 order; write path written twice | 0 (owner's order); 3.1; 3.2 (no Python storage slice); 18b section 6 (15c mapping); R1u split from R1c (18a) |
| BLOCKER 2 parity by recorded traffic | 18a section 3.2 (differential from the contract, ledger, live tier); R0c core baseline; R0d |
| BLOCKER 3 `jobs-dispatch` and `runner` | 2.5; R3a (18b); 18a R1j |
| BLOCKER 4 no storage slice | 4; 18a R1s; 2.2 object storage row |
| MAJOR 1 gain estimate | 1.2; 3.4; M1, M5, M16, M18 |
| MAJOR 2 cloud checks | 5.1 (requests, stamps); 5.2 (load generator); 5.3 (host parameter, cost lines); 5.4 (distinct photos) |
| MAJOR 3 upload windows | 4.2 (durable before commit, permits, idle limit, spawned task); 4.6; 4.7 (`run_id`, formats) |
| MAJOR 4 test gaps | 18a section 3 (ledger, schema dump, seeding over HTTP, surviving `src/`); 18b R12 |
| MAJOR 5 unassigned items | 18b section 7.6 |
| MAJOR 6 contract tooling | 2.4; M15; R0s in 18a |
| MAJOR 7 plan 17 interaction | 3.5 (RP dropped); 12 |
| MAJOR 8 two folds and runtimes | 3.2 (no Python fold extended: one fold, Rust, checked against the frozen oracle); 2.5 and 18a 3.5 (two runners, one conformance suite) |
| MAJOR 9 R2b's store decision | 3.1 (stage C, D4) |

| Decision | Applied in |
|---|---|
| 1 order (ii-b) | 3.1, 3.2, 3.3; 18b sections 1 to 6 |
| 2 jobs without Celery | 2.1, 2.5; R1j (18a), R3a, R3b (18b); M17 |
| 3 what stays Python | 2.3; M12 to M14; R6, R8, R10 (18b) |
| 4 storage layer | 2.2, 4; R1s, R1u (18a); R1c, R5e (18b) |
| 5 parity | 2.4; 18a section 3.2; R0c, R0d |
| 6 RP dropped | 3.5; 12 |
| 7 gain estimate | 1.2, 3.4; M16 |
| 8 cloud benchmark | 5 |
| 9 multi-user seams | 6.1 |
| 10 contract tooling | 2.4; M15 |
| 11 early lanes | 18a sections 2 and 5 |
| 12 measurements | 8 |
| 13 the rest | this table; 18a sections 3 and 4; 18b section 7 |

Minor findings: all fixed (1: 1.1; 2: 2.2 PostgreSQL row; 3: 2.2 wake-ups row; 4: 2.2 duplicate probe;
5: 3.2 sizes, M1; 6: 2.2 ExifTool row; 7: 2.5; 8: 5.3; 9: 2.3).
