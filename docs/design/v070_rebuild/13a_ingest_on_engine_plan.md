# 13a: ingest on the engine, one photo at a time from the engine to the store

| Field | Value |
|---|---|
| Status | plan, revision 1; the coordinator answered section 12 on 2026-11-14; an independent review of the claim, placement and retry semantics runs before lanes L4 and L2 start |
| Code base read | `int/b4-stateless` at `1375ce707` (read only) |
| Refines | [13_backend_speed_plan.md](13_backend_speed_plan.md): A1 (curation ingest part), E4, P1 and D3; section 11 lists every place where this plan deviates from 13 and why |
| Not in scope | core `/ingest` and `/analyze` (A1 moves them to plans; the commit pattern here can be reused there), faces, OCR, the region-embed stage (E3), dataset import and the reprocess scopes (they keep their current path until E3 or A1 moves them) |

## 0. The user's job, and what this plan fixes

A person adds photos. A detector finds the objects. Every photo and every object crop gets a
vector. The person labels their own classes by cluster and trains a model. For that loop the
measure is **photos per second from "photo in" to "stored and findable"**, with the GPU busy.

The owner's direction: one photo in, decoded once, all ML on the GPU from that frame, everything
stored once, then findable.

What went wrong. A four-way measurement (photos per second, batches of 32):

| Row | old code, array | old code, NVMe | new code, array | new code, NVMe |
|---|---|---|---|---|
| 4 clients, 4,000-photo mix | 18.1 to 19.2 | **29.3 to 30.7** | 24.0 to 25.3 | 24.8 to 26.4 |
| 4 clients, set P (20 MP) | 18.2 | **35.5** | 30.3 | 29.1 |
| 4 clients, COCO 2,000 | 18.4 | **25.7** | 20.4 | 20.4 |
| 1 client, COCO | 7.0 | **12.4** | 9.7 | 9.9 |
| 1 client, 12 MP JPEG | 8.4 | 14.7 | 13.0 | **16.5** |
| 1 client, 20 MP | 6.0 | 12.4 | 14.3 | **14.4** |

Once storage is fast, the new code is slower than the old code on every row except large JPEGs
at one client. The GPU work per photo is the same in both (PE image encoder about 9 ms per item,
one instance, max batch 32, 15 ms queue delay). The difference is how often the GPU is fed:
throughput follows the GPU busy share within 1 to 10% on all three rows checked (old 92% busy
against new 75% on COCO at 4 clients).

The cause, from the per-request timeline (32 COCO photos, one client, 3.4 s): the new
`finish_photos` runs the batch as barrier phases. The GPU works only in one of them.

| Phase of one 32-photo request | 1 client: ms | GPU % | 4 concurrent: ms | GPU % |
|---|---|---|---|---|
| prefill_detections (CPU decode + batched detector) | 525 | 0 | 945 | 3 |
| p1 identity + ExifTool capture + detect filter | 90 | - | 269 | 13 |
| p2 resolve_classes | 185 | 0 | 862 | 20 |
| p3 PE embed of crops and whole frame | 1,323 | 88 | 3,636 | 100 |
| p4 claim + OpenSearch writes + crop cache + publish | 1,091 | 0 to 12 | 1,715 | 1 to 15 |
| request wall | 3,378 | 35 | 8,534 | 57 |

The old code ran each photo through its own pipeline, so one photo's embed overlapped another
photo's writes.

Proof (new code on NVMe, COCO 2,000, no code change, only more requests in flight so their
phases interleave): 4 x 32 photos 20.4 photos per second at 74% GPU; 8 x 32 25.6 at 88%;
16 x 32 26.8 at 96%; 16 x 8 **27.2 at 98%**, above the old code's 25.7, with a request p50 equal
to the old code's (4.8 s). So the gap is GPU idle time from the phase convoy, not slower work,
and COCO's GPU-bound ceiling on one card is about 27 to 28 photos per second. On 20 MP photos
(200 photos, 8 x 32, indicative only) the GPU was 70% busy with the PE queue at 20 ms: there the
frame engine's own queue (285 ms per photo) and decode (128 ms) were the first limit.

Ruled out by numbers: PostgreSQL pool waits (30 to 0.5 ms per photo after P0),
the PE engine build, ExifTool (persistent processes, 90 ms per batch), the frame engine itself
(0% SM during the COCO probe), API CPU (10% less per photo than the old code).

**This plan:** every photo of a curation ingest goes to the frame engine with one plan; each
photo moves on its own from the engine answer to its write; the database work stays a fixed
number of transactions per request; the publish of events becomes one statement. Expected: the
new code at or above the old code on NVMe on every row of the table, with the GPU busy share
reported for each.

## 1. Facts from the code that the design rests on

All paths are under the repository root. Line numbers are at `1375ce707`.

| # | Fact | Where |
|---|---|---|
| F1 | The batch decodes every CPU-route photo and runs the detector batched for all of them before any photo goes on | `src/services/curation/ingest_batch.py:151-178` (`_prefill_detections`), called at `:310` |
| F2 | `finish_photos` waits for all photos at each step: detect (`run_all` at `:176`), classes (`:182`), prepare/embed (`:190`), commit (`:198-199`) | `src/services/curation/ingest_finish.py:171-208` |
| F3 | The commit runs placement, claims, all writes and the publish once for the batch | `src/services/curation/ingest_index.py:535-585` |
| F4 | `apublish_many` inserts one event per statement: one `INSERT ... RETURNING` per crop through `run_sync`, so a 32-photo COCO batch is about 230 round trips | `src/services/events/publish.py:95-101`, `src/db/repos/events.py:80-114` |
| F5 | The Valkey wake-up is a synchronous redis-py `PUBLISH` in an `after_commit` listener, once per transaction, on the event-loop thread (socket timeout 1 s) | `src/services/events/publish.py:60-65`, `src/tasks/runtime/event_wakeup.py:62-67` |
| F6 | Embedding selection reads only the detector label (`proposal_name`), score and box area, not the resolved class: `name_matches` uses `class_name or proposal_name`, and before resolution `class_name` is empty | `src/services/curation/ingest_policy.py:140-212`, `src/services/curation/name_match.py:35-42` |
| F7 | `resolve_classes` is in-memory when the registry snapshot has every label; it opens a transaction (registry row `FOR UPDATE`) only for a label the project lacks | `src/services/curation/ingest_items.py:207-244`, `src/db/repos/classes.py:125` |
| F8 | The engine refuses PNG and other formats, progressive JPEG, CMYK, 12-bit, truncated files and photos below its size gate; the API gate (`OP_INGEST_GPU_MIN_MEGAPIXELS`, default 3.0) sends small photos to the CPU path first | `frame_engine/prep_geometry.py:120-146`, `src/services/curation/ingest_gpu_decode.py:165-181`, `src/config/curation.py:236-237` |
| F9 | Today every engine failure falls back to the API CPU path, photo by photo | `src/services/curation/ingest_batch.py:108-115`, `ingest_gpu_decode.py:184-204` |
| F10 | The engine plan for ingest embeds every detection (`embed_classes` unset); the embedding policy and the detect filter are applied in the API afterwards, so vectors of dropped crops are computed and thrown away | `src/services/curation/ingest_gpu_decode.py:215-219`, `ingest_index.py:210-231` |
| F11 | The engine holds a photo on one of 16 worker threads for its whole life, model-call waits included; 128 requests admitted at once, above that RESOURCE_EXHAUSTED | `frame_engine/settings.py:77-81`, `frame_engine/front.py:241-246` |
| F12 | The engine client retries an overload 3 times with a 120 s timeout per try | `src/clients/frame_engine.py:32-36` |
| F13 | The OpenSearch writer already coalesces documents across photos and requests (up to 1,000 docs per bulk, 4 flushes in flight, 20 ms linger), writes `refresh=False`, and answers each photo alone | `src/services/curation/ingest_writer.py:57-59, 287` |
| F14 | Every batch request ends with an explicit index refresh | `src/services/curation/ingest_batch.py:352`, `src/services/curation/ingest.py:405-414` |
| F15 | The P0 test asserts at most 5 sessions per batch, one connection at a time, and the same count for 8 and 32 photos | `tests/curation/test_ingest_finish_batch.py:225-237` |
| F16 | The bulk-ingest runner sends batches of 16 with 16 batches in flight by default; the one-client benchmark rows are a stricter case than the real bulk path | `src/services/curation/bulk_ingest/limits.py:15-16` |
| F17 | The engine image has no Pillow; `src/utils/image_decode.py` (Pillow plus simplejpeg, EXIF upright) imports only Pillow and numpy | `docker/frame-engine/requirements.txt`, `src/utils/image_decode.py:15-21, 83-113` |

## 2. Dependency graph of one photo and of one batch, as the code has it

### 2.1 One photo: what truly needs what

```
bytes ──> imohash ──> resolve_contents (batch query) ──> new or duplicate
  │
  ├──> engine call: decode -> detector -> boxes -> crop squares -> PE crops
  │                         \-> frame square -> PE frame;  blur sums;  crop JPEGs
  ├──> ExifTool capture (new photos only)                      [needs bytes only]
  │
  boxes ──> detect filter (score, area, label)                 [no registry]
        ──> embedding selection (label, score, area)           [no registry, F6]
        ──> crop ids = f(image_id, bbox_norm)
  items ──> class resolution (registry by name)                [needed only to build docs]
  vectors ──> placement (serving-partition prototypes, DB read) [needed only to build docs]
  image_id, bytes ──> content claim                            [before the docs]
  item ids ──> pending rows (near-duplicate rescore)           [before the docs]
  docs ──> OpenSearch write ──> created ids ──> crop cache, crop.created events
```

### 2.2 One batch: barriers and what stands behind them

| Barrier today | Code | A real dependency? |
|---|---|---|
| B1 all CPU-route photos decoded and detected before any finishes | `ingest_batch.py:310` | No. Batching detector calls is the model server's job (dynamic batcher); photos do not need each other |
| B2 all photos detected and captured before any class is resolved | `ingest_finish.py:176` | No |
| B3 all classes resolved before any embed | `ingest_finish.py:182` | No. Embedding needs labels, not classes (F6) |
| B4 all photos embedded before any commit | `ingest_finish.py:190-199` | No. Each photo's documents depend on its own vectors |
| B5 placement for all, then claims for all, then writes, then publish | `ingest_index.py:549-584` | Partly. Claim and pending rows must precede that photo's documents; events must follow them. Nothing ties one photo's write to another's |
| B6 explicit refresh at the end of each request | `ingest_batch.py:352` | No reader of the batch response needs it; the content lookup reads PostgreSQL and a realtime `mget` |
| B7 waves of one `imohash` (`_rounds`) | `ingest_batch.py:226-238, 330-339` | Yes: two different files with one `imohash` are decided one after the other. Kept |

## 3. Target shape: every photo goes to the engine

### 3.1 Options, with what each costs and deletes

| | (a) every photo to the engine | (b) keep an API path for some formats | (c) mix by size |
|---|---|---|---|
| What runs where | The engine decodes every image: nvJPEG for baseline JPEG at or above its gate, a CPU decode inside the engine (the same `decode_image` function) for everything else; then the same GPU kernels and model calls | Small and non-JPEG photos keep API decode, CPU letterbox, CPU PE preprocessing, API calls to the detector and PE | as (b), with the line drawn by megapixels only |
| API work per photo | orchestration and writes only | today's 25 to 28 ms of pixel work per small photo (decode, letterbox, PE preprocess, crop, blur, JPEG; refresh step 1 table) | same as (b) |
| Code deleted | the API prefill and batched detector path, the API size gate and `OP_INGEST_GPU_MIN_MEGAPIXELS`, the CPU fallback per photo, the CPU branch of `prepare_image` for ingest (it stays only for the callers listed as out of scope until their lanes) | nothing | the prefill only |
| Open cost | the engine's speed on small photos is **not measured** (no engine call took a COCO photo in any row); engine threads and admission must be sized for many small photos | two full pipelines kept; every later branch (faces, OCR, thumbnails) needs two implementations | as (b) |

**Decision: (a), gated by the measured spike S1 (section 8, lane L1).** It is the owner's shape
and section 6 of 13 already decided that small images are decoded on the CPU inside the engine.
Nothing in (a) is an interim rewrite: the per-photo flow of lane L2 is the final shape whichever
way S1 ends.

If S1 fails, the fallback is (c) with no new code: the API keeps `gpu_route`'s size gate, and the
small photos go through the host stage of section 4.2 (today's per-photo CPU functions). L2 is
started only after S1, so nothing is written twice.

### 3.2 What the engine needs for (a) (lane L1)

- **CPU decode inside the engine.** `route()` stops refusing a decodable file. It returns
  `('gpu', header)` for a baseline 8-bit JPEG at or above `OP_FRAME_ENGINE_MIN_MEGAPIXELS` and up
  to `OP_FRAME_ENGINE_MAX_MEGAPIXELS`, and `('cpu', reason)` for the rest (`small`, `not_jpeg`,
  `progressive`, `components`, `precision`, `truncated`). The CPU route calls
  `src/utils/image_decode.decode_image` on an engine thread (simplejpeg or Pillow, EXIF upright:
  the exact pixels the API decodes today), copies the frame to the device in the shared arena,
  and runs the same branches. `huge` stays a refusal. A file neither decoder reads answers
  `decoder`.
- The engine image adds `pillow==13.0.0` and `simplejpeg==1.9.0` (the API's pins) and copies
  `src/utils/image_decode.py`.
- `decode_route` in the answer: `gpu`, or `cpu_<reason>` (the names already in
  `src/domain/decode_route.py`; their meaning changes from "decoded in the API" to "decoded on
  the CPU inside the engine"). `cpu_gpu_off`, `cpu_call_failed`, `cpu_not_image` and `cpu_single`
  leave the vocabulary for engine projects when L2 merges; a host-stage photo (section 4.2) is
  `cpu_host`, to be added.
- The frame budget counts a CPU-decoded frame like a GPU one (`need_bytes`).
- Settings measured in S1, not guessed: `OP_FRAME_ENGINE_WORKERS` (16, 32, 64), and
  `OP_FRAME_ENGINE_MAX_IN_FLIGHT` with the engine's resident memory per queued photo.

Profiles (E4): with (a) a deployment whose engine is off cannot ingest on the engine. Until E4
proves the engine fits every profile, `OP_INGEST_GPU_DECODE=0` selects the host stage for the
whole deployment (section 4.2). E4 then retires that variable; that deletion is lane L6.

## 4. Flow and bounds of one ingest request

### 4.1 Shape

```
route reads files ─> imohash, resolve_contents (session 1) ─> waves by imohash (B7, kept)
  engine not ready and the project uses it ─> 503 engine_unavailable with Retry-After
  admission gate holds the encoded bytes (the API no longer decodes)
  one TaskGroup per wave:
    K stage workers  (K = OP_MAX_INGEST_CONCURRENCY, default 16) pull photos from a queue:
        engine call ║ ExifTool capture      (two tasks in a per-photo TaskGroup)
        -> detect filter -> embedding states -> blur, ids  = PreparedImage (no classes yet)
        -> put on the committer's inbox  (a failure is a result, never an exception)
    one committer: commit groups, at most 3 per wave, one at a time
        group tx (one session): placement read + content claims + pending rows
        then for each photo of the group, a write task (kept by the TaskGroup):
            build docs -> writer.write (coalescing, F13) -> crop cache
    after the TaskGroup: release failed claims (only if any), publish all crop.created (one tx)
  no index refresh (B6)
```

A commit group starts when at least `ceil(n_live / 3)` photos wait in the inbox, or when no
photo is still in its stage; it takes every photo waiting at that moment. The third group always
waits for the last photo. So photos 1 to n/3 are claimed and written while the GPU still works
on the rest, and the tail after the last GPU answer is one group's commit, not the whole batch's.

Expected effect, one client, 32 COCO photos (an estimate from the timeline, to be measured):
head before the GPU is busy about 0.15 s (the first engine answers; today 0.6 s of prefill and
p1), GPU work about 1.25 s (p3's 1.32 s at 88%, plus the detector), tail about 0.3 s (one group
of about 11 photos: claim, 90 documents written, crop cache, one publish statement). About 1.7 to
2.0 s per request, 16 to 19 photos per second, against 9.9 today and 12.4 for the old code. At
4 clients and more the GPU-bound ceiling (about 27 to 28 on COCO) is the target.

Class resolution happens in the commit group, just before the documents are built: in memory in
the steady state (F7), a separate transaction only when a label is new to the project. That
keeps today's fall-back for a failed class create untouched (`resolve_classes` is unchanged).

### 4.2 The two stage implementations

| Stage | Used for | Code |
|---|---|---|
| engine stage | every project whose plan the engine runs, when `OP_INGEST_GPU_DECODE=1` | `run_gpu_decode` renamed `run_engine_stage`, plus the capture task; `prepare_image`'s engine branch |
| host stage | a project with a secondary detector or a non-uniform letterbox fill (D2 keeps them on the API until A1 adds a second `detect` model); a deployment with `OP_INGEST_GPU_DECODE=0` until E4; small photos only if S1 fails | today's per-photo functions unchanged: `_decode_image` on `run_cpu`, `detect_items` (single-image detector calls; Triton's dynamic batcher groups concurrent photos), `prepare_image`'s CPU branch |

The host stage is not new code: it is what `ingest_one` runs today. The batched prefill (F1),
`run_primary_batch` and `run_secondary_raw_batch` are deleted.

### 4.3 Bounds and backpressure (no in-process locks for shared state)

| Level | Bound | Mechanism |
|---|---|---|
| per request | K photos in their stage (default 16) | K worker coroutines in the wave's `TaskGroup` pulling from an `asyncio.Queue`; no task per photo is created up front |
| per request, database | one session at a time; at most 5 in the steady state (section 5) | the committer runs one group at a time |
| per API process | engine calls in flight: the engine pool's `max_concurrent` (`triton_inflight()`, `src/config/api_pools.py:57-59`) | the pool's semaphore (per process, not shared state) |
| per process, memory | encoded megabytes held (`ingest_gate`, `src/utils/cpu_pool.py`) | unchanged gate; it now counts encoded bytes only |
| whole card | the engine's admission (`OP_FRAME_ENGINE_MAX_IN_FLIGHT`), frame budget and region waits | RESOURCE_EXHAUSTED; the client retries with jittered exponential backoff **until the photo's deadline**, not a fixed 3 tries (F12) |
| across requests | none needed in the API | requests no longer move in step: each photo's GPU work and writes interleave with other requests' photos |

Python docs ("Task Groups"): the first failing child cancels the others, and the group keeps a
strong reference to every task it creates
(<https://docs.python.org/3/library/asyncio-task.html#task-groups>). So:

- a per-photo failure is returned as a value (`call_frame_engine` already returns a reason
  string, `detect_for_ingest` returns the exception, `prepare_image` returns a failed
  `IndexOutcome`); no new broad `except`;
- a request-level failure (PostgreSQL pool timeout, a bug) raises out of the group, cancels the
  rest, and the route answers as today (503 `database_busy`, or 500);
- no `gather` fan-out is added.

### 4.4 Failure table (changes against `ingest_finish.py`'s table)

| Failure | Today | After |
|---|---|---|
| engine not ready at admission | every photo takes the CPU path | 503 `engine_unavailable`, `Retry-After`; the bulk sender already retries 503 with `Retry-After` |
| engine overloaded | CPU path after 3 tries | retried with backoff until the photo's deadline (120 s, `TIMEOUT_SECONDS`), then `failed`, `error_kind` `engine_unavailable` (to be added in `ingest_models.py`) |
| engine call fails, or a branch errors | CPU path | `failed`, `engine_unavailable`; the bulk run retries the file |
| file neither decoder reads | `failed`, `decode_failed` | same (engine status `decoder`) |
| class create, claim, write, crop cache, publish | table in `ingest_finish.py` | the same per photo; a claim transaction that fails as a whole falls back to per-photo claims for that group only |
| API worker killed mid-request | claims without documents are released by the next lookup (`ingest_dedup.py:74-77`) | same |

## 5. The commit phase, stage by stage

| Stage | Today (1 client / 4 concurrent, ms per 32-photo request) | Documented practice | Change | Expected | Measured by |
|---|---|---|---|---|---|
| resolve_classes | barrier B3; 185 / 862; the timeline used fresh projects, so this is mostly first-time class creation under the registry row lock | lock only what you write, in one order (PostgreSQL explicit locking, "Deadlocks": <https://www.postgresql.org/docs/18/explicit-locking.html>) | runs per commit group, in memory when every label is known; unchanged create path when one is new | off the critical path; warm project near 0 | `class_create` stage timer; warm-project row |
| placement | its own session per batch (`place_vectors`, `ingest_index.py:155-167`) | one transaction for the reads a write depends on | `place_new(session, ...)` inside the group transaction (it already takes a session) | one session less per request | session counter test |
| claim and pending rows | one transaction per batch, 112 / 175 | one transaction, advisory locks in sorted order (already so) | one transaction per commit group (at most 3), same function `claim_in_session` | the tail pays one third | `claim` stage timer |
| doc context | 6 / 0 (cached) | | unchanged, read once per group | none | |
| images and items writes | 647 / 697 at the end of the batch; server 1.22 ms per document, no rejections, 256 docs per flush | bulk of 5 to 15 MB, `refresh=false` (OpenSearch bulk API: <https://docs.opensearch.org/latest/api-reference/document-apis/bulk/>) | unchanged writer; photos arrive spread out, so writes overlap GPU work instead of following it | write time leaves the critical path except the last group | `opensearch_write` stage timer; GPU busy in the write window |
| items pre-read (`mget`) under a newly created images doc | one `mget` per flush, all misses | | **measure only (lane L5):** create first, the existing 409 path reads and merges; the lock rule is untouched because `create` never overwrites | a few ms per flush; adopt only if at least 5% of write time | `ingest_write_path.py` probe |
| crop cache | 140 / 136, after all writes | | per photo right after its own write (already per photo; no longer behind a barrier). All crops are engine JPEG bytes, so the API encodes none | overlapped | `crop_cache` timer |
| publish (`crop.created`) | 281 / 632: one `INSERT ... RETURNING` per event (F4) | SQLAlchemy "insertmanyvalues": an executemany INSERT with RETURNING is sent as multi-row statements, 1,000 rows per batch by default, "using much fewer database round trips" (<https://docs.sqlalchemy.org/en/21/core/connections.html#insert-many-values-behavior-for-insert-statements>) | `insert_events_async(session, project, rows)` sends one `insert(Event).returning(...)` with a list of parameter sets; `apublish_many` calls it once; one wake-up per transaction as now | 281 -> about 10 to 30 ms (about 230 round trips -> 1) | `publish` timer; statement count in a unit test |
| wake-up | sync redis-py `PUBLISH` on the loop thread per commit (F5) | redis.asyncio client for async code (<https://redis.readthedocs.io/en/stable/examples/asyncio_examples.html>) | unchanged in this plan: one call per request, under a millisecond when Valkey is up. Listed as a risk (a Valkey outage blocks the loop up to 1 s per transaction) | none | |
| refresh | explicit per request (F14); C's node did 2,272 refreshes (53 s) and 144 merges (53 s) over the refresh rows | "Refresh operations are resource intensive ... wait for OpenSearch's periodic refresh rather than performing explicit refreshes when possible"; default every 1 s while searched (<https://docs.opensearch.org/latest/api-reference/index-apis/refresh/>) | delete from `run_ingest_batch`; the bulk run refreshes once when it ends; `/ingest/image` and `/ingest/upload` keep one refresh (a person waits to see the photo) | fewer small segments and merges under load | OpenSearch node stats (refresh and merge totals) per row |

Sessions per wave, steady state: lookup 1 + commit groups at most 3 + publish 1 = **at most 5**,
one at a time (P0 kept). A label new to the project adds one class transaction per group that
meets it (at most 3, only while a project is cold); a failed write adds the release transaction
(as today).

## 6. Streaming the batch response (D3)

Do not stream the HTTP body in this plan.

- At one client the client sends its next batch only when the response ends, streamed or not.
  What fills the GPU at one client is the per-photo flow inside the request (section 4), not the
  body format.
- The bulk runner (the real bulk path) keeps 16 batches in flight (F16) and settles rows per
  batch; per-photo rows would change its ledger code for no GPU gain.
- A streamed body sends 200 before the work, so `503 database_busy` with `Retry-After`, which the
  sender retries today, would need an in-stream error record and a new client contract.
  FastAPI's `StreamingResponse` streams an async generator's chunks
  (<https://fastapi.tiangolo.com/advanced/custom-response/#streamingresponse>); nothing in it
  carries a late status.

Revisit when the upload UI needs per-photo progress: an NDJSON variant of the route, added with
its contract, is the documented way.

## 7. Seams left open, not built

| Seam | What stays open | Rule for the code of this plan |
|---|---|---|
| A second encoder for crops (a smaller crop encoder measured 8.5x faster at equal crop quality) | `EnginePlan.embed` names one model for frame and crops today | build the plan in one function, `ingest_engine_plan(service)` in `ingest_gpu_decode.py`, that reads frame and crop encoders separately (both PE today); the item space's vector field and dimension come from `item_write_fields()`, never from a PE constant in the commit path |
| faces, OCR, derivatives branches | the engine already returns per-branch status and outputs | the engine stage keeps the whole `EngineAnswer` on `PreparedImage` (one field), so a later branch adds a writer step, not a new call |
| a vector store other than OpenSearch | vectors travel inside documents built by `build_documents` | all writes go through `service._bulk_index` -> `IngestWriter.write`; nothing else in the flow calls OpenSearch for ingest writes |
| embedding selection inside the engine | F10: dropped crops are embedded anyway | send `embed_classes` (detector class ids whose labels pass the detect filter's and the policy's class lists, by name: D2 Q4). Score, area and per-image caps stay in the API; a plan field for them is added only if P6 makes such a policy the default |

## 8. Work breakdown

Each lane: its own worktree and branch off `int/b4-stateless`, light checks only, red test first,
one commit per concern, no push, no merge. Files owned by one lane only.

| Lane | Owns (files) | Starts | Depends on |
|---|---|---|---|
| L1 engine CPU decode + S1 probe | `frame_engine/prep_geometry.py`, `frame_engine/engine.py`, `frame_engine/host_decode.py` (new), `frame_engine/prep_budget.py`, `frame_engine/settings.py`, `docker/frame-engine/requirements.txt`, `docker/frame-engine/Dockerfile`, `src/domain/decode_route.py`, `src/domain/frame_engine_wire.py`, `scripts/bench/engine_direct_bench.py` (new, unless an existing direct harness is found with `rg -l frame_engine scripts/`), `tests/frame_engine_tests/` | wave 1 | none |
| L3 publish in one statement | `src/services/events/publish.py`, `src/db/repos/events.py`, `tests/events/` | wave 1 | none |
| L4 commit groups | `src/services/curation/ingest_index.py`, `src/services/near_duplicates/ingest_claim.py`, `tests/curation/test_ingest_finish_batch.py` (placement and claim tests only) | wave 1 | none |
| S1 measurement | coordinator, after wave 1 merges | wave 1 sweep | L1 |
| L2 per-photo flow | `src/services/curation/ingest_finish.py`, `ingest_batch.py`, `ingest.py`, `ingest_gpu_decode.py`, `ingest_models.py`, `ingest_detect.py`, `src/clients/frame_engine.py`, `src/config/curation.py`, `src/config/retired_env.py`, `env.template`, `docs/CURATION.md`, the ingest tests under `tests/curation/` not owned by L4 | wave 2 | L1, L3, L4 merged; S1 decided |
| L5 items create-first (measure, then maybe code) | `src/clients/occ_upsert.py`, `scripts/bench/ingest_write_path.py` | wave 2, measurement only | none |
| L6 delete the host stage | the host-stage code paths named in 4.2 | after E4 and A1 (second detector) land | not gated in this plan |

### L1: the engine decodes every image

1. Red: `tests/frame_engine_tests/test_prep_geometry.py`: `route()` of a PNG, a progressive
   JPEG, a CMYK JPEG and a JPEG below the gate answers `('cpu', <reason>)`; a corrupt file still
   answers `corrupt`; `huge` stays a refusal.
2. Red: `frame_engine/host_decode.py` returns, for the same bytes, exactly the pixels of
   `src.utils.image_decode.decode_image` (array equality on generated JPEG, PNG, WebP, GIF,
   progressive and EXIF-rotated files; offline, numpy only).
3. Red: `tests/frame_engine_tests/test_engine.py` with the existing fakes: a CPU-route photo runs
   every requested branch, answers `ok` with `decode_route` `cpu_<reason>`, and reserves its frame
   bytes from the budget; a file neither decoder reads answers `decoder`.
4. Delete in the same change: the refusal statuses that became CPU reasons, from `PREP_STATUSES`
   and the engine's `_failed` paths; nothing in the API maps them to a CPU fallback any more after
   L2 (L2 deletes that mapping in `ingest_batch.py`).
5. Add `scripts/bench/engine_direct_bench.py` if no direct harness exists: N photos to the
   engine at a fixed number in flight, photos per second, p50 and p99, engine CPU seconds per
   photo, GPU busy; numbers only, written under the run's output directory.

**S1 gate (decides 3.1).** On COCO 2,000, the 4,000-photo mix and 200 photos of set P, direct
to the engine, plan `detect`, `embed_frame`, `embed_crops`, `crops: inline`, engine workers 16,
32 and 64 (and nvJPEG lanes equal to workers):

| Measure | Bar |
|---|---|
| COCO photos per second at 32 in flight | at least 26 with the GPU at least 95% busy (the GPU-bound ceiling is about 27 to 28, proof row in section 0; the old code's row is 25.7) |
| set P photos per second at 32 in flight | at least 40 (E0 measured 44 to 47 direct; the old code's API row is 35.5); report the engine queue and decode ms per photo, the first limit in the 20 MP proof row |
| COCO p50 at one photo in flight | at most 150 ms |
| engine host CPU per COCO photo | at most 20 ms (the API spends 25 to 28 ms of pixel work per small photo today) |
| outputs | boxes, classes and blur sums equal to today's API CPU path for CPU-decoded photos; crop vectors cosine at least 0.9995 each and median at least 0.9999 against today's vectors, read against a control captured twice (the detector is not bit-stable) |

Pass: (a). Fail: (c), and L2 keeps the API size gate.

### L3: one statement per publish

1. Red: `tests/events/test_publish_transactional.py`: `apublish_many` of 200 events executes one
   INSERT statement (count `session.execute` calls on a recording session), the rows come back
   in order with their ids, and exactly one wake-up is pended per transaction.
2. Code: `insert_events_async(session, project, rows)` in `src/db/repos/events.py` with
   `_check_size` per payload and one `insert(Event).returning(Event.id, Event.project_id)` executed
   with the list of parameter sets (insertmanyvalues); `_write_async` uses it.
3. Delete: the per-event loop in `_write_async`.
4. Pending wave run: the database-tier event tests (`tests/events/`, db marker).

### L4: commit groups

1. Red: a group transaction places, claims and marks pending in one session
   (`commit_group(service, prepared) -> list[GroupOutcome]`, in `ingest_index.py`;
   `GroupOutcome` holds the claim outcome and the built documents).
2. Red: `place_prepared` takes the caller's session (no session of its own).
3. Keep: `claim_and_mark_each` as the fall-back when a group transaction fails as a whole; the
   pool-timeout rule (no per-photo retry) unchanged.
4. Split `commit_prepared` into `commit_group` (one transaction), `write_prepared` (per photo,
   unchanged) and `finish_commit(service, written, unclaimed)` (release, then one publish). No
   caller keeps the old function: `index_items` (a batch of one) calls the three in order.
5. Pending wave run: `tests/curation/ingest_db/` claim and pending tests.

### L2: the per-photo flow

1. Red, in `tests/curation/test_ingest_finish_batch.py` (L2 owns the flow tests there; L4 owns
   only the placement and claim tests): with an engine fake whose answers arrive one at a time,
   the first photo's documents are written before the last photo's engine answer arrives.
2. Red: with a slow-engine fake, the batch opens at most 5 sessions for 1, 8, 32 and 64 photos,
   one at a time. This replaces the assertion "same count for 8 and 32 photos" (F15): the count
   now depends on arrival order, while the bound stays.
3. Red: an engine fake that is never ready makes `ingest_batch` raise the 503
   `engine_unavailable` rejection with `Retry-After`; an engine call that fails after its retries
   gives that photo `failed` / `engine_unavailable` and the others succeed (replaces the CPU
   fallback tests in `test_ingest_gpu_decode.py` and `test_ingest_gate_gpu_route.py`, whose
   behaviour D1 removes).
4. Red: the embedding mask before and after `resolve_classes` is equal for every policy mode on
   the same items (guards F6, so classes can move after the embed).
5. Red: the engine plan carries `embed_classes` for a policy with classes and a detect filter
   with classes, by name through the detector's labels.
6. Red: `run_ingest_batch` no longer refreshes (replaces
   `test_a_batch_refreshes_once_covering_both_indexes`); `/ingest/image` and `/ingest/upload`
   still refresh once.
7. Code, in this order: `run_engine_stage` (engine call and capture in one per-photo
   `TaskGroup`); `finish_photos` as section 4.1 (K stage workers, one committer, write tasks
   kept by the wave's `TaskGroup`); `run_ingest_batch` without prefill or refresh; `ingest_one`
   through the same flow; overload retry until the photo's deadline in
   `src/clients/frame_engine.py`.
8. Delete in the same change: `_prefill_gpu_decode`, `_prefill_detections`,
   `WholeImageDetector.run_primary_batch` and `run_secondary_raw_batch` (if `rg` finds no other
   caller), `PhotoJob.prefilled_image`, `prefilled_items`, `prefilled_secondary`; when S1 passed:
   `gpu_route`, the API size gate, `OP_INGEST_GPU_MIN_MEGAPIXELS` (added to
   `src/config/retired_env.py`, removed from `env.template` and docs), the `cpu_*` routes that
   only the API fallback produced.
9. After a wire change: `make contracts`, and commit `contracts/` in the same commit.

Checks per lane (light only):

```bash
nice -n 19 .venv/bin/python -m pytest -q --no-cov tests/curation/test_ingest_finish_batch.py \
  tests/curation/test_ingest_gpu_decode.py tests/curation/test_ingest_read_your_writes.py
nice -n 19 .venv/bin/python -m pytest -q --no-cov tests/frame_engine_tests/ tests/events/
.venv/bin/pre-commit run --files <touched files>
.venv/bin/pre-commit run --hook-stage pre-push --files <touched files>
rg -n "_prefill_detections|run_primary_batch|run_secondary_raw_batch|prefilled_image" src tests   # empty after L2
rg -n "OP_INGEST_GPU_MIN_MEGAPIXELS" src tests docs env.template                                # only retired_env.py after L2
```

## 9. Gates (coordinator's wave runs)

| Gate | How | Bar |
|---|---|---|
| Offline suite and hooks | `.venv/bin/python -m pytest tests/ -q --no-cov -m 'not live and not families'`; `.venv/bin/pre-commit run --all-files` and the pre-push stage | green |
| Detection and vector parity | `scripts/bench/parity.py capture` on COCO and Objects365 (public), reference captured twice on the current head, then `compare` | GPU-route photos: as E1 (identical boxes, crop bytes, vectors at least 0.999889). CPU-decoded photos: S1's output bars. Same documents and item ids for the same input (fields other than `decode_route`, write ids and times) |
| Four-way rows | the refresh's row set on NVMe: mix, set P, COCO at 4 clients; COCO, 12 MP JPEG, 20 MP and 12 MP PNG at one client; plus 8 and 16 clients on COCO | **not below the old code on NVMe on any row**; GPU busy share (mean and p50) reported per row; failed 0, else the row is invalid |
| Stage timeline | the probe that produced section 0's table, on the new head | GPU busy outside the embed window; the tail after the last engine answer at most one group's commit |
| Sessions | P0 counter, plus PostgreSQL commits per request from the probe | at most 5 per request in the steady state |
| Restart and kill | 50 engine kills during a bulk run; one model-server restart under light load; one API worker killed mid-request | model server never restarts; every file stored after the bulk run's retry; no claim row without its images doc after the next lookup |

## 10. Risks and what would change the choice

| Risk | Sign | Response |
|---|---|---|
| Small photos are slower through the engine (thread per photo, a second gRPC hop, CPU decode under the engine's GIL) | S1 fails | (c), no new code; or raise engine workers if the gap is in the queue |
| The engine is the first limit on large photos (20 MP proof row: GPU 70%, engine queue 285 ms and decode 128 ms per photo, 16 workers each holding a photo through its model-call waits, F11) | set P row below 35.5 after L2 | engine workers and lanes from S1; if more threads do not help, a second engine process per card (D1 allows it as a measured setting) |
| The engine becomes the host-memory bound (queued requests hold encoded bytes) | S1's memory per queued photo times admission exceeds the container budget | size `OP_FRAME_ENGINE_MAX_IN_FLIGHT` from S1; overload is retried, never a fallback |
| Commit groups add PostgreSQL contention between concurrent requests of one project | claim time grows at 16 clients | groups of a third, never smaller; claims keep sorted lock order |
| No engine fallback turns an engine outage into failed photos | bulk runs stop after 5 failed batches | intended (D1); the engine restarts by itself (E1: ready 2.7 s after a restart) |
| Embedding before class resolution changes a mask | L2 step 4 test | the test fails closed; the fix is to resolve the labels first, in memory |
| Valkey outage blocks the loop up to 1 s per transaction (F5) | loop lag p99 | outside this plan; the wake-up moves to `redis.asyncio` when measured |
| OpenSearch 3.9.0 costs more per document than 3.6.0 | write window still large after the flow change | experiment 3 of the diagnosis (3.6.0 under the new code), only if the rows miss the bar |

## 11. Relation to 13 (deviations, with the reason)

| 13 says | This plan | Why |
|---|---|---|
| D3: "the batch route streams" | per-photo flow inside the request; the HTTP body is not streamed | section 6: no GPU gain at one client, breaks the 503 retry contract |
| D2: a project with a secondary detector keeps today's route until a second `detect` model lands | kept, but through the per-photo host stage; its batched detector calls are deleted | one flow instead of two; Triton's dynamic batcher groups per-photo calls (<https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html>) |
| D1: engine down means the image fails visibly | kept, plus a 503 at admission when the engine is not ready | the bulk sender already backs off on 503 |
| D1: "protocol version 1 does not change after E1" | additive changes only: `decode_route` values, `cpu_host` | the engine and the API ship together; earlier lanes already added fields (OCR, derivatives) |
| E4: engine sizes per profile | `OP_INGEST_GPU_DECODE=0` keeps the host stage until E4 lands | no deployment loses ingest before its engine fits |
| P0: at most 5 sessions per batch | at most 5 in the steady state; cold projects add class transactions, failures the release | the bound no longer depends on photos; groups need their own transactions |

## 12. Questions and the coordinator's answers (2026-11-14)

| # | Question | Answer |
|---|---|---|
| 1 | Secondary detector and non-uniform letterbox fill stay on the host stage until A1? | Yes. |
| 2 | If S1 misses a bar by 15% or less, tune engine workers and admission before falling back to (c)? | Yes, one extra S1 hold. A miss on the output bars is never tuned away. |
| 3 | The batch response stays unstreamed (section 6)? | Yes. Revisit with the upload UI. |
| 4 | The session bound restated as in section 11 (at most 5 in the steady state)? | Yes. The test names the two additions (a label new to the project, a failed write) so the bound cannot drift further. |
| 5 | `OP_INGEST_GPU_DECODE=0` keeps the host stage until E4, and `1` means the engine is required (no fallback)? | Yes. |
| 6 | Remove the per-request refresh from batch ingest, keep it on `/ingest/image` and `/ingest/upload`, one refresh at the end of a bulk run? | Yes, with one condition: L2 reads `tests/curation/test_ingest_read_your_writes.py` first and keeps every read-your-writes promise the routes document; a promise that needed the per-request refresh is listed in the lane report before it is changed. |

Order fixed by the coordinator: L3 starts at once. L1 starts after the colour and faces lanes
merge, because they change the engine's decode path and its branch list. L4 and L2 start after the
independent review.
