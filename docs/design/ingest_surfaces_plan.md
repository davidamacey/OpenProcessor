# Ingest surfaces: S3 source, triggers and the gRPC decision (#264, #263)

Status: plan; nothing in `src/` changed by this file. Issues #264 (S3 source with watch mode) and #263
(gRPC ingest channel, only if it wins), built on the bulk runner of #229. Written 2026-10-12 against
`origin/main` `300a321c`. A fresh implementation agent executes this file alone: section 9 gives the slices
in order with files, interfaces, tests and exit gates; section 10 the verification commands. File and
line references drift; re-find by symbol.

## 0. Summary

Every way of getting images in ends in one pipeline, `CurationIngestService.ingest_batch`:

| Surface | State | This plan |
|---|---|---|
| HTTP API requests (`/ingest`, `/ingest/batch`, `/ingest/upload`) | exists | unchanged |
| Bulk from a folder or manifest (`/ingest/bulk`, CLI, Cropwright panel) | exists | gains named sources (section 3) |
| Bulk from S3 or an S3-compatible store | new | a route that takes object references (fetch + stage + ingest), S3 listing, `s3` source kind |
| Triggers: S3 event webhook, cron schedule, poll-with-checkpoint watcher (S3 and local folder) | new | one dispatcher, one durable per-source queue |
| gRPC ingest channel | open question | built only if the measurement of section 8 shows a win beyond noise |

Core decisions (all justified in section 2):

1. **Stage the bytes, do not stream them past the disk.** An object is fetched into memory, written once
   content-addressed under `upload_root` (the `/ingest/upload` mechanism), and goes through the same
   `ingest_batch` as an upload. The stored `image_path` is the staged file, so thumbnails, crops, the
   region worker and the VLM keep working unchanged. The stored reference to the original is
   `source_identifier = s3://bucket/key#etag=...&version=...` (one keyword field, no mapping change).
2. **One queue per source decides what is ingested.** A watcher poll, a webhook event, a cron fire and a
   manual run all do the same thing: insert `(key, etag)` rows into the source's durable SQLite queue.
   A bulk run of kind `queue` drains the pending rows through the existing runner. Dedup by `(key, etag)`
   is the primary key.
3. **One dispatcher** (a periodic task started in the API lifespan, one leader per project through a file
   lock, like the open-vocabulary sweeper) turns due triggers into scans and runs. No new service, no new
   dependency for scheduling.
4. **gRPC is decided by measurement, bounded first by Amdahl.** Transport cost per image is measured on its
   own; if it is a small share of the 0.18 s (COCO) to 1.1 s (photo) of API CPU per image, nothing is built.

Non-goals: authentication or hardening of the API (the owner decision is "runs locally"), S3 Event
Notifications to SQS, bucket mutation, uploading to a bucket, deleting from a bucket, cross-region
replication logic.

## 1. What exists (read before changing)

- Runner: `src/services/curation/bulk_ingest/` (`runner.py` 443 lines, `source.py`, `store.py`, `sender.py`,
  `walk.py`, `limits.py`). The runner holds paths, never bytes; it posts each batch to the API's own
  `POST /ingest/batch` over loopback (`HttpBatchSender`) so batches spread over all API workers; the
  ledger `done.jsonl` (one `{p, o}` line per stored path) makes a run resumable; `claim_run` is the single
  per-project claim (409 `bulk_ingest_busy`), `exclusive_start_lock` the cross-process lock.
- Routes: `src/routers/curation/ingest_bulk.py` (+ `_ingest_bulk_models.py`), CLI `scripts/curation/bulk_ingest.py`
  (a client of the routes), Cropwright `frontend/src/lib/components/ingest/BulkIngestPanel.svelte` with
  `frontend/src/lib/ingest/bulkIngestController.svelte.ts`.
- Bytes in: `POST /ingest/upload` (`src/routers/curation/ingest_upload.py`): reads multipart bytes, `_imohash_bytes`,
  `persist_uploaded_bytes` (content-addressed `upload_root/<hh>/<hash><ext>`, atomic, dedup-by-existence),
  then `service.ingest_batch(data, paths, source, source_identifiers, ingest_run_id)`. `source_identifier`
  is an indexed keyword, already matched by `path_lookup`.
- Decode routing and GPU front end: inside `ingest_batch` (`ingest_batch.py`, `ingest_frontend.py`); photos at or
  above `ingest_gpu_min_megapixels` may use the opt-in GPU path, CPU below. **Not touched.**
- Admission: `src/utils/cpu_pool.py` `InflightGate` (FIFO, counts MiB, a request larger than capacity is clamped
  and runs alone); the process gate `ingest_gate()` counts decoded MiB (`OP_INGEST_INFLIGHT_MB`, 1024).
- Coalesced OpenSearch writer (`ingest_writer.py`), content-hash dedup (msearch of imohash), OCC and the lock rule:
  inherited because the new path calls `ingest_batch`.
- No scheduler exists in the repo (`cluster_maintenance.py` only mentions APScheduler in a docstring). The pattern
  for a periodic in-process job is `open_vocab_sweeper.py` (task per worker, `for_each_project`, non-blocking
  `exclusive_start_lock`).
- `boto3` is not a dependency yet. API runs `uvicorn --workers=32` (N processes): every shared fact lives on disk.

## 2. Design decisions

### D1. Staged, not streamed (settles "streaming vs staging")

Streaming bytes straight into decode and dropping them would break every later reader of `image_path`
(thumbnail and crop serving, the region worker, the VLM, reprocess, export), which all open a local file. The
bytes are in memory anyway (decode needs them), so staging costs one atomic write per **new** object (the
existing `persist` stage timer, content-addressed so duplicates write nothing). Measurement: the live check
reads `op_pipeline_stage_seconds{stage="persist"}` against the run's seconds per image and records the
share in `docs/PERFORMANCE.md`; staging is kept unless `persist` exceeds 5 percent of API CPU per image on
set A (it should not: it is an atomic `write_bytes`), in which case the decision and the numbers are recorded
and revisited. Disk use equals the bytes of the objects ingested; the originals stay in the bucket and are
reachable by the reference below and by a presigned URL (D6).

### D2. Image reference

`source_identifier` = `s3://<bucket>/<percent-encoded key>#etag=<etag>[&version=<version_id>]`
(`ObjectRef.to_uri()` / `ObjectRef.parse()` in `object_store.py`; the key is quoted with `safe='/'` so `#` and `?`
in a key survive). The ETag has its quotes stripped. This identifies bucket, key, etag and version in the one
indexed field that already exists; no mapping migration, no `ingest.py` signature change. A changed object
(new etag) is a new reference and a new image (its bytes hash differently); an unchanged one is skipped before
download (D4).

### D3. Where the fetch happens

The `objects` route (new, sibling of `/ingest/upload`) takes object
references, not bytes, and returns the same `BatchIngestResponse`. The server fetches the objects with the
configured client (bounded, D5), then calls one shared function that also serves `/ingest/upload`
(`stage_and_ingest(...)` extracted from `ingest_upload.py`: hash, `persist_uploaded_bytes`, `ingest_batch`,
open-vocabulary scheduling). Upload and objects are then two thin adapters over one implementation. The bulk
runner sends its batches to this route over loopback, exactly as it sends folder paths to `/ingest/batch`, so
the work spreads over all API workers and the runner still holds only references.

### D4. Idempotency, three layers

1. Queue primary key `(key, etag)`: replayed webhooks, repeated polls and overlapping scans insert nothing.
2. Before any GET, the route asks the images index for the batch's exact `source_identifier` values (one
   `terms` query through the factory client); a hit is answered `duplicate` without a download.
3. `ingest_batch` content-hash dedup (unchanged): the same bytes under another key are a `duplicate`.

### D5. Bounded in-flight bytes

A second `InflightGate` instance (the same class) in `cpu_pool.py` (`fetch_gate()`, capacity
`S3_INFLIGHT_MB`, default 256) counts **encoded** MiB held by fetches in a process, taken from the object
size the listing/event carried (HEAD when absent) and held from the GET until the bytes are handed to
`ingest_batch`. It is separate from `ingest_gate()` because that gate counts decoded pixels and `ingest_batch`
acquires it itself; holding it across a fetch would deadlock. Per-object cap `S3_MAX_OBJECT_MB` (default 256):
larger objects fail with `error_kind=too_large`. GETs run in threads (`boto3` is synchronous) with a bounded
pool (`S3_FETCH_WORKERS`, default 8 per process); the runner's concurrency (default 16) bounds batches.

### D6. Originals stay reachable

The staged file serves review and crops. The presign route (`/objects/presign`, POST) takes a stored `source_identifier`,
re-checks it against the allow-list and returns a time-limited presigned GET URL (`expires_in`, default 900 s).
The API never proxies bytes and never returns credentials.

### D7. Credentials and fail-closed configuration

Credentials come only from the boto3 default chain: environment (`AWS_ACCESS_KEY_ID` ...), a mounted shared
credentials file (`AWS_SHARED_CREDENTIALS_FILE`), or the instance/task role. Never from a request, the tree or
a stored document; logs never print them. Configuration (every variable is read with the `OP_` prefix; the table lists the part after it):

| Variable | Meaning | Default |
|---|---|---|
| `S3_ALLOWED` | comma list of `bucket` or `bucket/prefix` entries a source may use | empty: S3 is **off** and every S3 route answers 503 `s3_not_configured` |
| `S3_ENDPOINT_URL` | S3-compatible endpoint (MinIO) | AWS |
| `S3_REGION` | region | boto3 chain |
| `S3_PATH_STYLE` | path-style addressing (MinIO) | off |
| `S3_INFLIGHT_MB`, `S3_MAX_OBJECT_MB`, `S3_FETCH_WORKERS`, `S3_LIST_WORKERS` | bounds (D5, D8) | 256, 256, 8, 8 |
| `INGEST_DISPATCH_S` | dispatcher tick; 0 disables triggers | 5 |

A source or a fetch naming a bucket/prefix outside `S3_ALLOWED` is refused 422 `s3_not_allowed` (checked at
create, at run and again per object in `/ingest/objects`, so a stale source cannot widen access). Folder sources
keep the existing `OP_SOURCE_ROOT` rule.

### D8. Listing at scale

`s3_listing.py`: `ListObjectsV2` pages (1000 keys) with a continuation token. Parallel prefixes: one
delimiter listing (`Delimiter='/'`) under the source prefix yields child prefixes; each is listed by its own task
in a pool of `S3_LIST_WORKERS`, objects directly under the prefix are listed first. Filters by extension
(`.jpg .jpeg .png` default, as folders). A **scan pass** keeps durable progress in the queue's `meta` table:
`{pass_started_at, partitions: {prefix: last_key | "done"}}`, written after the page's rows are committed, so a
restart resumes at the last page of each unfinished partition instead of relisting. When a pass completes, the
watermark becomes `pass_started_at - 120 s` (clock skew and in-flight uploads); the next pass queues objects with
`LastModified >= watermark` that are not yet in the queue by `(key, etag)`. A pass with no watermark (first run,
or "rescan all") queues everything. Cost: `ceil(N / 1000)` LIST requests per pass, parallel over prefixes; for
buckets where that is too much the webhook trigger is the answer and the docs say so.

### D9. Failure and retry semantics per object

Queue row states: `pending`, `done`, `failed`, `dead`. A fetch error is classified in the route:

| Cause | error_kind | Row after |
|---|---|---|
| 404/NoSuchKey/NoSuchVersion | `gone` | `dead` (deleted before fetch; not retried) |
| 403/AccessDenied, other 4xx | `forbidden` | `failed` (retried: permissions can be fixed) |
| 5xx, throttling, timeout, connection | `fetch_error` | `failed` (boto3 retries first: `retries={'mode':'standard','max_attempts':5}`) |
| larger than `S3_MAX_OBJECT_MB` | `too_large` | `dead` |
| not an accepted extension / empty / undecodable (existing `error_kind` values) | as today | `dead` (deterministic) |

`failed` rows are retried by later runs up to 3 attempts (`attempts` column), then `dead`; the UI offers
"retry dead" (sets them back to `pending`). A whole-batch failure is the runner's existing rule (five in a row
stop the run). The run's `failures.jsonl` reports each failure with its reason, as for folders.

### D10. Triggers, one mechanism

All triggers write queue rows; the dispatcher starts runs.

| Trigger | Mechanism |
|---|---|
| Manual "run now", CLI | the run route of a source (`/sources/{id}/run`, POST): scan (if the backend lists) then drain |
| Poll-with-checkpoint watcher | per source `watch.interval_s` (min 10); due when `now - last_poll_started >= interval`; scan then drain only if rows are pending (no empty runs) |
| Cron schedule | per source `schedule.cron` (5 fields, UTC: `*`, lists, ranges, steps); due when the next fire after `last_fire_at` is in the past; missed fires collapse into one (no catch-up storm) |
| S3 event webhook | the events route of a source (`/sources/{id}/events`, POST) (below): inserts rows and returns; the dispatcher drains within one tick |

The local-folder watcher is the same code: its scan walks the folder (`walk.iter_image_paths`) and queues paths
whose `(path, "<mtime_ns>:<size>")` is new (watermark on mtime with the same overlap), so a copy still in
progress is picked up once stable (a file modified in the last 5 s is deferred to the next pass).

Webhook contract. Body: the S3 event notification JSON (`{"Records":[{"eventName":"ObjectCreated:Put",
"s3":{"bucket":{"name":...},"object":{"key":...,"size":...,"eTag":...,"versionId":...}}}]}`), as sent by S3/SNS
HTTP forwarding and by MinIO's webhook target. Rules: a body that is not that shape is 422
`invalid_s3_event` (with the path of the first bad field); records whose event is not `ObjectCreated:*` are
counted `ignored`; records for another bucket, outside the source prefix or with a non-image extension are
`ignored` (never an error: a bucket-wide notification feeds several sources); keys are URL-decoded exactly once
(S3 sends them form-encoded, `+` is a space); a record without an etag is queued with the etag looked up by one
HEAD at drain time (empty-string placeholder replaced). Response 202 `{received, queued, duplicates, ignored}`
within one SQLite transaction (target p99 under 50 ms at 1000 records); the dispatcher, not the request, does
all S3 and pipeline work. A webhook for a source without `triggers.webhook.enabled` is 409 `webhook_disabled`.
The receiver is per source, so the URL itself selects project and source; there is no shared secret in this
iteration (owner decision: local use), and the docs say the endpoint must stay on loopback/LAN like the rest
of the API.

### D11. Where the dispatcher lives

`bulk_ingest/dispatcher.py`: `start_ingest_dispatcher()` / `stop_ingest_dispatcher()` mirrored on
`start_open_vocab_sweeper`, started from `src/main.py` lifespan. Each tick, per project (`for_each_project`),
under `exclusive_start_lock(<project dir>/dispatch.lock)` (non-blocking: one worker acts per project per tick):
list sources, compute due actions with injected `now`, start scans/runs as `asyncio` tasks of this worker
(a worker that dies leaves a stale heartbeat; `repaired_state` marks the run `interrupted` and the next tick
resumes it through `claim_run`; a scan resumes from its checkpoint). The loopback address for the runner's
sender comes from `API_LOOPBACK_URL` (default `http://127.0.0.1:8000`, the container's own listen port; the
route-started runs keep using the request's socket). One live run per project is the existing rule; a due source
waits for the next tick while another run is live (sources of a project are served oldest-due first).

### D12. Runner changes (small, no rewrite)

- `BulkSource` gains kind `queue` with `path = source_id`; `iter_source_paths` yields the queue's pending
  references. Existing kinds are untouched.
- `BatchSender` gains `result_key(result) -> str | None` (folder sender: `image_path`; objects sender:
  `source_identifier`) so the runner's result matching works for both.
- `BulkIngestRun` takes an optional `listener` with `on_done(refs, outcome)` and `on_failed(rows)`; the queue
  uses it to mark rows. Folder runs pass none.
- The existing bulk start route accepts exactly one of `folder`, `manifest`, `s3` (`{bucket, prefix}`) or `source_id`.
  `s3` finds-or-creates a trigger-less source for the pair and runs it, so a one-shot bucket ingest and a watched
  source are the same mechanism.

## 3. Interfaces

New files, each under 700 lines (target under 300). All under `src/services/curation/bulk_ingest/` unless noted.

| File | Content |
|---|---|
| `object_store.py` | `ObjectRef` (`bucket,key,etag,version_id,size`; `to_uri`, `parse`); `ObjectStore` Protocol (`list_page(bucket, prefix, token, delimiter) -> ListPage`, `get(ref) -> bytes`, `head(ref)`, `presign(ref, expires)`); `Boto3ObjectStore` (built lazily from env, `asyncio.to_thread` at call sites); `S3Config.from_env()`, `allowed(bucket, key)`; typed errors `ObjectGone`, `ObjectForbidden`, `ObjectFetchError`, `S3NotConfigured`, `S3NotAllowed` |
| `s3_listing.py` | `scan_s3(store, queue, source, *, workers, now) -> ScanResult` (parallel prefixes, durable partition progress, watermark) |
| `folder_scan.py` | `scan_folder(queue, source, *, now)` (mtime watermark, stable-file deferral) |
| `object_queue.py` | `ObjectQueue(path)` over SQLite (WAL, `busy_timeout`): `enqueue(rows) -> (queued, duplicates)`, `iter_pending(limit)`, `mark(refs, state, error)`, `retry_dead()`, `counts()`, `meta_get/meta_set`, `events` counters |
| `s3_events.py` | `parse_s3_event(body) -> list[EventRecord]`, raising `InvalidS3Event(path, reason)`; `EventRecord.matches(source)` |
| `schedule.py` | `Cron.parse(expr)`, `Cron.next_after(dt)`; `interval_due(last, interval_s, now)` |
| `sources.py` | `IngestSourceDef` (dataclass, JSON), `SourceRegistry` (CRUD on `<runs_root>/sources/<id>/source.json`, runtime `state.json`, atomic writes), `source_status(...)` (counts, next poll/fire, last run summary: the data the UI renders) |
| `dispatcher.py` | `tick(now) -> list[Action]` (pure decision, unit-tested), `run_tick(...)` (executes), start/stop |
| `object_fetch.py` | `fetch_objects(store, refs, gate, pool) -> list[FetchOutcome]` (size admission, classification of D9) |
| `src/services/curation/upload_ingest.py` | `stage_and_ingest(...)` shared by `/ingest/upload` and `/ingest/objects` |
| `src/routers/curation/ingest_objects.py` | the `/objects` and `/objects/presign` routes |
| `src/routers/curation/ingest_sources.py`, `_ingest_sources_models.py` | sources CRUD, run-now, retry-dead, webhook `events`, status |

Routes (all under `/curation/projects/{project}/ingest/`, no `/v1` twin):

| Route | Body / result |
|---|---|
| `/sources` (GET, POST) | create: `{name, backend: 's3'\|'folder', s3:{bucket,prefix}\|folder, extensions, source, triggers:{watch:{enabled,interval_s}, schedule:{enabled,cron}, webhook:{enabled}}, batch_size, concurrency}`; list returns `SourceStatus[]` |
| `/sources/{id}` (GET, PATCH, DELETE) | `SourceStatus`: definition, queue counts (`pending, done, failed, dead`), `scan` (state, pages, listed, queued, last_error), `watch` (last_poll_at, next_poll_at, last_error), `schedule` (cron, last_fire_at, next_fire_at), `webhook` (path, received/queued/duplicates/ignored totals, last_event_at), `last_run` (id, status, counts), `can_run`, `can_retry_dead`, `blocked_reason` |
| `/sources/{id}/run`, `/sources/{id}/retry-dead` (POST) | 202 `SourceStatus`; 409 `bulk_ingest_busy` when a run is live |
| `/sources/{id}/events` (POST) | S3 event JSON in; 202 `{received, queued, duplicates, ignored}` |
| `/objects` (POST) | `{items:[{bucket,key,etag?,version_id?,size?}], source, ingest_run_id}` in; `BatchIngestResponse` out |
| `/objects/presign` (POST) | `{source_identifier, expires_in?}` in; `{url, expires_in}` out |

Wire models live in the router `_models` modules; storage-to-wire mapping goes through the single serializer
(`wire.py`) where an item field is touched (none is added here); `make contracts` runs in the same commit as each
route change.

On disk, per project, beside the bulk runs: `<OP_BULK_INGEST_DIR>/projects/<slug>/sources/<source_id>/`
`source.json`, `state.json`, `queue.sqlite` (tables `objects(key, etag, version, size, state, attempts, error,
queued_at, updated_at, PRIMARY KEY(key, etag))`, `meta(k, v)`, `event_totals`). `list_stores` ignores the
directory (run ids are matched by regex); project deletion removes it with the project's jobs directory.

## 4. Edge cases the code and tests must cover

- Key with space, `+`, `%`, `#`, `?`, non-ASCII, trailing slash ("folder" placeholder objects are skipped).
- Zero-byte, non-image and over-size objects: per-object failure, never a failed batch.
- Webhook records for versioned buckets (`versionId`), delete events (ignored), replayed deliveries, event for a key
  already queued with another etag (new row), burst of 1000 records.
- Restart mid-scan (resume at the partition's last key), restart mid-run (`interrupted`, resume skips the ledger),
  two workers racing the same tick (one acts), clock skew (overlap), objects uploaded during a pass.
- S3 unreachable or credentials missing: scan fails with a recorded `last_error`, rows stay pending, the
  dispatcher backs off (doubling to 10 min), the run is not started.
- `S3_ALLOWED` unset or a source outside it: refused, nothing listed, nothing fetched.

## 5. Cropwright

Thin frontend: the backend serves every decision (`can_run`, `blocked_reason`, `next_poll_at`, human-readable
state keys) and the frontend renders. New "Sources" section on `/p/[project]/ingest` next to the Bulk panel:

- `frontend/src/lib/components/ingest/IngestSourcesPanel.svelte` (list of source cards: backend and location,
  trigger badges, last/next poll and fire, queue counts, last run with link to the existing run view, failures
  drawer, "Run now", "Retry dead", enable toggles) and `SourceForm.svelte` (create/edit: S3 bucket/prefix or
  folder, interval, cron, webhook toggle; shows the webhook path and a ready-to-copy MinIO/AWS configuration hint).
- `frontend/src/lib/ingest/sourcesController.svelte.ts` (polling, mutations), `api_ingestSources.ts`,
  `types_ingestSources.ts` (checked against `contracts/` by a contract test as `bulkIngestContract.test.ts` does).
- Vitest for controller and API client; one stubbed e2e like `frontend/e2e/stubbed/test_bulk_ingest.py`.

## 6. Tests (few, meaningful, each seen failing first)

Offline (`tests/curation/`, a fake `ObjectStore` with an in-memory bucket, paging and injectable errors):

| File | Proves |
|---|---|
| `test_object_ref_and_config.py` | ref round trip for hostile keys; allow-list semantics; unconfigured is fail-closed |
| `test_s3_listing_scan.py` | paging across partitions, extension filter, watermark picks only new objects, resume after a simulated crash mid-pass lists no page twice, first pass queues all |
| `test_object_queue.py` | `(key, etag)` dedup, state transitions, attempts -> `dead`, `retry_dead`, concurrent writers (two threads) |
| `test_s3_events.py` | accepts S3 and MinIO samples; rejects malformed with the field path; URL-decoding; filters by bucket/prefix/extension/event name |
| `test_cron_schedule.py` | next fire for lists/ranges/steps, month and year rollover, day-of-week, invalid expressions, missed-fire collapse |
| `test_ingest_dispatcher.py` | `tick()` decisions with injected clock: poll due, cron due, pending-only drain, busy project waits, backoff after scan failure, no empty runs |
| `test_ingest_objects_route.py` | route with fake store and fake `ingest_batch`: not-allowed 422, already-stored skipped without GET, classification table of D9, bytes bound (the fetch gate never exceeds capacity), reference stored as `source_identifier` |
| `test_ingest_sources_routes.py` | CRUD, run-now to a finished run over the fake sender (queue rows marked), webhook idempotency and shape errors, `blocked_reason` |

Live (one MinIO, isolated compose project, section 7): create a bucket and upload a pinned public COCO subset
(200 images of `coco_bench_2000.json`) with a script that uses `boto3` against MinIO; the single end-to-end check
is MinIO webhook target -> `/events` -> dispatcher -> bulk run -> items in OpenSearch -> visible in Cropwright
(full-page screenshots at desktop width and 430 px, viewed). It also exercises the poll watcher and a one-shot
`--s3` CLI run on a second prefix, and records `persist` share (D1).

Hooks and gates: ruff, mypy, the file-size ratchet, `check_docs_vs_code.py`, `test_env_surface.py`,
`test_naming_leaks.py`, contract drift; `make contracts` output committed with its change.

## 7. Safety for the live checks

Compose project `-p isurf` only (the default project `openprocessor` is live and is never touched), its own ports on
127.0.0.1, local-only image tags, MinIO (`minio/minio` already pulled) as a service of `docker/test`-style compose file
`docker/test/compose.s3.yml` with fixed throwaway credentials that exist only in that file for that purpose (never
real credentials; the API container gets them through compose `environment:` literals of the test file, never
`.env`). GPU only for the one end-to-end check: GPU 2 when `nvidia-smi` shows it clean, `flock /tmp/ortloom-bench.lock`
around any timed command. Public data only (COCO / Open Images). Teardown (`compose -p isurf down -v`, remove the
worktree-local volumes) is proven in the final report with `docker ps`, `docker volume ls`, `docker network ls` filters.

## 8. gRPC decision protocol (#263)

Question: does a gRPC ingest service beat HTTP beyond noise on this pipeline?

Step 1, transport-only (no pipeline, no GPU, no OpenSearch). A harness `scripts/bench/ingest_transport_bench.py`
sends the real bytes of COCO set A (the pinned 2,000, 640 x 480, median 166 KB) and set B (the pinned Commons
photo set, 12 to 24 MP; if the 5 GB set cannot be fetched, the largest available subset with the count stated)
to a server that reads the bytes, computes the imohash (the work the real route does first) and returns a
fixed-size result of the same shape as a `BatchIngestResponse`. Contenders at **equal client concurrency**
(4, 16, 32), 3 repetitions each, median and range:

1. HTTP/1.1 multipart through the production stack (uvicorn, FastAPI `File` parsing), one connection per client;
2. HTTP/2 multipart (hypercorn if installable in the bench venv; otherwise stated as not measured and why);
3. gRPC `grpc.aio` client-streaming of image chunks with one response per batch (and unary with the whole batch as
   a second row).

Recorded per row: images/s, request p50/p99, **server CPU seconds per image** and **client CPU seconds per image**
(`psutil` process times of the server and the client), bytes on the wire per image, peak resident memory of the server.

Step 2, Amdahl bound. `transport_share = (server CPU per image of the HTTP row - of the gRPC row) / API CPU
per image of the full pipeline` using the measured 0.18 s (set A) and 1.10 s (set B) (PERFORMANCE.md), and the
images/s plateau of 21 (set A, GPU bound) and 15 (set B, CPU bound). The pipeline is GPU-bound on set A: a
transport change cannot raise images/s there, only lower API CPU.

Step 3, decision rule (written before the numbers):

- **Skip** (record the numbers, close #263 with the reason, build nothing) unless the transport-only saving is at
  least 10 percent of the pipeline's API CPU per image on set A **or** set B, with ranges that do not overlap
  across the 3 repetitions.
- If it passes: build a prototype `grpc.aio` servicer behind a flag that calls the shared `stage_and_ingest`
  function (one implementation, two transports), repeat the comparison through the full pipeline at equal
  concurrency on set A and set B (3 repetitions, policy `all`, the PERFORMANCE.md protocol: fresh project per
  repetition, warm-up first, GPU 2 clean, `flock`), and ship only if images/s or API CPU per image improves by
  at least 10 percent with non-overlapping ranges and p99 not worse than 1.1x; otherwise delete the prototype.
- Either outcome is written in `docs/PERFORMANCE.md` (section "Ingest transports") and on #263.

A gRPC service, if built, must use the same project binding (`project` field, fail closed when unbound), the same
lock rule/OCC (it calls `ingest_batch`), the decode routing, and a generated `.proto` checked by a contract hook.

## 9. Slices (each ends with passing targeted tests, ruff, mypy; one commit per coherent step)

Plan PR: this file only, branch `docs/ingest-surfaces-plan`. Implementation branch `feat/ingest-surfaces` off
`origin/main` after the plan PR is opened (merge `origin/main` before each PR).

| Slice | Content | Exit gate |
|---|---|---|
| S1 S3 source and checkpointed watcher | `boto3` into `requirements.txt` and `requirements-test.txt`; `object_store.py`, `object_fetch.py`, `upload_ingest.py` (extract from `/ingest/upload`), `ingest_objects.py` route, `object_queue.py`, `s3_listing.py`, `folder_scan.py`, `sources.py`, queue source kind and listener in the runner, bulk start `s3`/`source_id`, sources CRUD + run-now routes, CLI `--s3`/`--source-id`, env/compose/docs; `make contracts` | offline tests of those files pass; `/ingest/upload` tests unchanged and green |
| S2 Webhook | `s3_events.py`, `/sources/{id}/events`, counters, docs | event tests; replay is idempotent |
| S3 Schedules and dispatcher | `schedule.py`, `dispatcher.py`, lifespan hook, `status`/`blocked_reason`, backoff | cron and dispatcher tests |
| S4 Cropwright | panel, form, controller, API client, types, tests, stubbed e2e | `npm run check`, `npm run build`, vitest, e2e stub green |
| S5 gRPC measurement and decision | harness, numbers, decision, docs; prototype only if the rule passes | decision recorded in PERFORMANCE.md and on #263 |
| S6 Live | MinIO compose file, the one end-to-end check, screenshots, `persist` share, teardown proof | evidence listed in the PR body |

CHANGELOG `[Unreleased]` entries and doc sections are appended at the end of their parents; docs name only the
current version; docs name only routes in the contract and `OP_*` variables the code reads.

## 10. Verification commands

```bash
.venv/bin/python -m pytest tests/curation -q --no-cov -k "s3 or object or source or cron or dispatcher or bulk"
OMP_NUM_THREADS=2 .venv/bin/python -m pytest tests/ -q --no-cov -m 'not live'    # once, before the PR
.venv/bin/pre-commit run --all-files
make contracts && make contracts-check
.venv/bin/python scripts/docs/check_docs_vs_code.py
(cd frontend && npm run check && npm run build && npx vitest run)
```

## 11. Risks

| Risk | Handling |
|---|---|
| Disk use of staged originals | documented; content-addressed so duplicates cost nothing; the reference allows re-fetch |
| Poll cost on very large buckets | parallel prefixes, durable resume; webhook trigger recommended; documented numbers from the live check |
| Dispatcher in a worker that restarts (`--limit-max-requests` recycles workers) | all state on disk; stale-heartbeat repair; scans resume from checkpoint |
| SQLite on the shared volume | WAL, one container only, `busy_timeout`, tested with concurrent writers |
| `boto3` import cost in 32 workers | imported lazily on first S3 use |
