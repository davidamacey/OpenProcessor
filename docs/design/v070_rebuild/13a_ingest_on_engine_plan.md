# 13a: ingest on the engine, one photo at a time from the engine to the store

| Field | Value |
|---|---|
| Status | plan, revision 2: the independent review of claims, placement and retries is folded in (appendix A); the coordinator answered section 12 on 2026-11-14 |
| Code base read | `int/b4-stateless` at `b01dac70d` (read only; lane L3 is merged there) |
| Refines | [13_backend_speed_plan.md](13_backend_speed_plan.md): A1 (curation ingest part), E4, P1 and D3; section 11 lists every place where this plan deviates from 13 and why |
| Not in scope | core `/ingest` and `/analyze` (A1 moves them to plans; the commit pattern here can be reused there), faces, OCR, the region-embed stage (E3); dataset import, the reprocess scopes and the open-vocabulary pass keep their pixel path until E3 or A1 moves them, but they share the commit functions of L4 and the write rules of L0 |

## 0. The user's job, and what this plan fixes

A person adds photos. A detector finds the objects. Every photo and every object crop gets a
vector. The person labels their own classes by cluster and trains a model. For that loop the
measure is **photos per second from "photo in" to "stored and findable"**, with the GPU busy,
and **every photo stored whole or not at all**.

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

Ruled out by numbers: PostgreSQL pool waits (30 to 0.5 ms per photo after P0), the PE engine
build, ExifTool (persistent processes, 90 ms per batch), the frame engine itself (0% SM during
the COCO probe), API CPU (10% less per photo than the old code).

**This plan:** every photo of a curation ingest goes to the frame engine with one plan; each
photo moves on its own from the engine answer to its write; the database work stays a small,
stated number of transactions per request; a photo is stored whole or not at all, also across a
crash; a retryable failure never uses up a file's attempts. Expected: the new code at or above the
old code on NVMe on every row of the table, with the GPU busy share reported for each.

## 1. Facts from the code that the design rests on

Paths are under the repository root; line numbers at `b01dac70d`.

| # | Fact | Where |
|---|---|---|
| F1 | The batch decodes every CPU-route photo and runs the detector batched for all of them before any photo goes on; the engine-route photos also wait for each other (`run_all`) | `src/services/curation/ingest_batch.py:108` (engine), `:151-178` (CPU), called at `:310` |
| F2 | `finish_photos` waits for all photos at each step: detect (`run_all` at `:176`), classes (`:182`), prepare/embed (`:190`), commit (`:198-199`) | `src/services/curation/ingest_finish.py:171-208` |
| F3 | The commit runs placement, claims, all writes and the publish once for the batch; release and publish are in a `finally` | `src/services/curation/ingest_index.py:535-585` |
| F4 | Publishing a batch of events is one `INSERT ... RETURNING` with one project resolve (lane L3, merged) | `src/db/repos/events.py:117-138`, `src/services/events/publish.py:95-101` |
| F5 | The Valkey wake-up is a synchronous redis-py `PUBLISH` in an `after_commit` listener, once per transaction, on the event-loop thread (socket timeout 1 s) | `src/services/events/publish.py:60-65`, `src/tasks/runtime/event_wakeup.py:62-67` |
| F6 | Embedding selection reads the label (`class_name or proposal_name`, normalized), score and box area. Before resolution a `by_name` item has only `proposal_name`; resolution sets `class_name` to the registry spelling of the same normalized slug, so the mask is equal. A detector with `assigns_class` sets `class_id`/`class_name` itself and resolution leaves it alone. **Not equal:** `region_seed_status` compares `class_name` and `proposal_name` lower-cased but not slug-normalized against the region profile's parent classes, inside `build_documents`, so classes must be resolved before documents are built | `src/services/curation/ingest_policy.py:140-260`, `src/services/curation/name_match.py:35-42`, `src/services/curation/ingest_detect.py:228-243`, `src/services/curation/item_doc.py:45-56`, `src/services/curation/region_scope.py:34-43`, `ingest_index.py:448-452` |
| F7 | `resolve_classes` is in-memory when the registry snapshot has every label; it opens a transaction (registry row `FOR UPDATE`) only for a label the project lacks | `src/services/curation/ingest_items.py:207-244`, `src/db/repos/classes.py:125` |
| F8 | The engine refuses PNG and other formats, progressive JPEG, CMYK, 12-bit, truncated files and photos below its size gate; the API gate (`OP_INGEST_GPU_MIN_MEGAPIXELS`, default 3.0) sends small photos to the CPU path first | `frame_engine/prep_geometry.py:120-146`, `src/services/curation/ingest_gpu_decode.py:165-181`, `src/config/curation.py:236-237` |
| F9 | Today every engine failure falls back to the API CPU path, photo by photo | `ingest_batch.py:108-115`, `ingest_gpu_decode.py:184-204` |
| F10 | The engine plan for ingest embeds every detection (`embed_classes` unset); the policy and the detect filter drop crops afterwards | `ingest_gpu_decode.py:215-219`, `ingest_index.py:210-231` |
| F11 | The engine holds a photo on one of 16 worker threads for its whole life; 128 requests admitted at once, above that RESOURCE_EXHAUSTED | `frame_engine/settings.py:77-81`, `frame_engine/front.py:241-246` |
| F12 | The engine client retries an overload 3 times with a 120 s timeout per try | `src/clients/frame_engine.py:32-36` |
| F13 | The OpenSearch writer coalesces documents across photos and requests (up to 1,000 docs per bulk, 4 flushes, 20 ms linger), writes the images doc (`create`) and then the items, `refresh=False`; a cancelled caller cancels only its wait, the flush runs on | `src/services/curation/ingest_writer.py:57-59, 245-266, 287` |
| F14 | Every batch request ends with an explicit index refresh; `/ingest/image` refreshes again in its route; `/ingest/upload` and `/ingest/objects` rely on the one inside `ingest_batch` | `ingest_batch.py:352`, `src/routers/curation/ingest.py:103`, `src/services/curation/upload_ingest.py:98-109` |
| F15 | The P0 test counts calls to a replaced session factory with stubbed repositories: at most 5, one at a time, and the same count for 8 and 32 photos | `tests/curation/test_ingest_finish_batch.py:86-159, 225-237` |
| F16 | The bulk-ingest runner sends batches of 16 with 16 batches in flight by default | `src/services/curation/bulk_ingest/limits.py:15-16` |
| F17 | The engine image has no Pillow; `src/utils/image_decode.py` imports only Pillow and numpy | `docker/frame-engine/requirements.txt`, `src/utils/image_decode.py:15-21, 83-113` |
| F18 | **A photo can be stored without its items, today.** When the items step of a flush raises, every unit of the flush fails (`ingest_writer.py:252-258`), `write_prepared` releases the claim (`ingest_index.py:484-487, 575`) but the images doc stays. The retry finds no content row, claims again, its `create` answers 409 with another write id, and `_settle_conflicts` answers `duplicate` (`ingest_writer.py:314-337`): no items are ever written. A kill between the two bulks leaves the doc and the claim, and the next lookup answers `duplicate` | as listed |
| F19 | **A refused item is silent.** A document the store refuses is only logged; the photo counts `success` with `n_crops` of every built doc | `ingest_items.py:117-118`, `ingest_index.py:511-528` |
| F20 | `RequestRejectedError` answers `{"detail": ...}` with no headers; only `PoolTimeoutError` and connection failures answer 503 with `Retry-After` | `src/core/error_handlers.py:112-114`, `src/core/db_session.py:45-90` |
| F21 | A positive engine-readiness verdict is cached per process for good | `ingest_gpu_decode.py:105, 137, 150` |
| F22 | The bulk sender retries a 503 only when it carries `Retry-After` (6 attempts, jittered backoff up to 30 s). A failed file uses one of `MAX_ATTEMPTS = 3` unless its kind is terminal, and is re-claimed at once; a run stops after 5 batches in a row failed entirely | `src/services/curation/bulk_ingest/sender.py:32-66, 100-128`, `src/db/repos/ingest_sources.py:40, 255-316`, `bulk_ingest/source_queue.py:35-40`, `bulk_ingest/runner.py:286-340` |
| F23 | The open-vocabulary drain finds its images by a search on `open_vocab_status: pending`; the stamps are per-image updates written **after** the batch's refresh, so today's refresh never made them visible to the drain; the 120 s sweep (`OP_OPEN_VOCAB_SWEEP_S`) catches what a drain missed | `src/services/curation/open_vocab_ingest.py:60-132`, `src/services/curation/reprocess_targets.py:156-186`, `src/services/curation/open_vocab_run.py:342-361`, `src/tasks/beat_schedule.py:21, 44` |
| F24 | Sessions a batch really opens: the lookup opens 1 plus one locked transaction per imohash the project already stores (`content_key.py:169-183`), one more when it releases orphans (`ingest_dedup.py:74-77`); a later wave looks up again (`ingest_batch.py:336`); the location policy is a 5 s cache read per `doc_context` (`location_policy_store.py:23-47`) | as listed |
| F25 | `index_items` (the commit of one photo) is also called by `reprocess_detect.py:124`, `dataset_import/chunk.py:250` and `open_vocab_run.py:440` | as listed |
| F26 | `stage_timer` refuses a name outside `STAGES`; there is no `crop_cache` stage | `src/utils/stage_timing.py:31-62, 118-127` |

## 2. Dependency graph of one photo and of one batch, as the code has it

### 2.1 One photo: what truly needs what

```
bytes ──> imohash ──> resolve_contents (batch query) ──> new, duplicate, or (L0) incomplete
  │
  ├──> engine call: decode -> detector -> boxes -> crop squares -> PE crops
  │                         \-> frame square -> PE frame;  blur sums;  crop JPEGs
  ├──> ExifTool capture (new photos only)                      [needs bytes only]
  │
  boxes ──> detect filter (score, area, label)                 [no registry]
        ──> embedding selection (label, score, area)           [no registry, F6]
        ──> crop ids = f(image_id, bbox_norm)
  items ──> class resolution (registry by name)                [before the docs: region seeding, F6]
  vectors ──> placement (serving-partition prototypes, DB read) [before the docs]
  image_id, bytes ──> content claim                            [before the docs]
  item ids ──> pending rows (near-duplicate rescore)           [before the docs]
  docs ──> images doc, then items ──> created ids ──> crop cache, crop.created events
```

### 2.2 One batch: barriers and what stands behind them

| Barrier today | Code | A real dependency? |
|---|---|---|
| B1 all photos of a route decoded and detected before any finishes | `ingest_batch.py:108, 310` | No. Batching detector calls is the model server's job (dynamic batcher) |
| B2 all photos detected and captured before any class is resolved | `ingest_finish.py:176` | No |
| B3 all classes resolved before any embed | `ingest_finish.py:182` | No for the embed (F6). Yes for the documents: resolution stays before `build_documents` |
| B4 all photos embedded before any commit | `ingest_finish.py:190-199` | No |
| B5 placement for all, then claims for all, then writes, then publish | `ingest_index.py:549-584` | Partly: claim and pending rows precede that photo's documents; events follow them |
| B6 explicit refresh at the end of each request | `ingest_batch.py:352` | Not for the lookup (PostgreSQL plus a realtime `mget`), the dirty-counts drain (refreshes itself), the near-duplicate rescore (retries a member without a vector), the region drain (stable polls) or the live counter (counts events). Yes for the read-your-writes promise of `/ingest/image`, `/ingest/upload` and `/ingest/objects` (a person waits). Not effective for the open-vocabulary drain (F23). Section 5 says what each keeps |
| B7 waves of one `imohash` (`_rounds`) | `ingest_batch.py:226-238, 330-345` | Yes: two files with one `imohash` and other bytes are decided one after the other. Kept |

## 3. Target shape: every photo goes to the engine

### 3.1 Options, with what each costs and deletes

| | (a) every photo to the engine | (b) keep an API path for some formats | (c) mix by size |
|---|---|---|---|
| What runs where | The engine decodes every image: nvJPEG for baseline JPEG at or above its gate, a CPU decode inside the engine (the same `decode_image` function) for everything else; then the same GPU kernels and model calls | Small and non-JPEG photos keep API decode, CPU letterbox, CPU PE preprocessing, API calls to the detector and PE | as (b), with the line drawn by megapixels only |
| API work per photo | orchestration and writes only | today's 25 to 28 ms of pixel work per small photo | same as (b) |
| Code deleted | the API prefill and batched detector path, the API size gate and `OP_INGEST_GPU_MIN_MEGAPIXELS`, the CPU fallback per photo, the CPU branch of `prepare_image` for ingest (kept only for the out-of-scope callers until their lanes) | nothing | the prefill only |
| Open cost | the engine's speed on small photos is **not measured**; engine threads and admission must be sized for many small photos | two pipelines kept; every later branch needs two implementations | as (b) |

**Decision: (a), gated by the measured spike S1 (lane L1).** Section 6 of 13 already decided that
small images are decoded on the CPU inside the engine. The per-photo flow of L2 is the final shape
whichever way S1 ends. If S1 fails, the fallback is (c) with no new code: the API keeps
`gpu_route`'s size gate and the small photos take the host stage of 4.2. L2 starts only after S1.

### 3.2 What the engine needs for (a) (lane L1)

- **CPU decode inside the engine.** `route()` stops refusing a decodable file. It returns
  `('gpu', header)` for a baseline 8-bit JPEG at or above `OP_FRAME_ENGINE_MIN_MEGAPIXELS` and up
  to `OP_FRAME_ENGINE_MAX_MEGAPIXELS`, and `('cpu', reason)` for the rest (`small`, `not_jpeg`,
  `progressive`, `components`, `precision`, `truncated`). The CPU route calls
  `src/utils/image_decode.decode_image` on an engine thread (the exact pixels the API decodes
  today), copies the frame to the device in the shared arena, and runs the same branches. `huge`
  stays a refusal. A file neither decoder reads answers `decoder`.
- The engine image adds `pillow==13.0.0` and `simplejpeg==1.9.0` (the API's pins) and copies
  `src/utils/image_decode.py`.
- `decode_route` in the answer: `gpu`, or `cpu_<reason>` (names already in
  `src/domain/decode_route.py`; their meaning becomes "decoded on the CPU inside the engine").
  `cpu_gpu_off`, `cpu_call_failed`, `cpu_not_image` and `cpu_single` leave the vocabulary when L2
  merges; a host-stage photo (4.2) is `cpu_host`, to be added.
- The frame budget counts a CPU-decoded frame like a GPU one (`need_bytes`).
- Settings measured in S1, not guessed: `OP_FRAME_ENGINE_WORKERS` (16, 32, 64), and
  `OP_FRAME_ENGINE_MAX_IN_FLIGHT` with the engine's resident memory per queued photo.

Profiles (E4): until E4 proves the engine fits every profile, `OP_INGEST_GPU_DECODE=0` selects
the host stage for the whole deployment. E4 then retires that variable (lane L6).

## 4. Flow, bounds and answers of one ingest request

### 4.1 Shape

```
route reads files ─> imohash, resolve_contents ─> waves by imohash (B7)
  project uses the engine, at least one photo is new, engine not ready
      ─> 503 engine_unavailable, Retry-After 5
  admission gate holds the encoded bytes (the API no longer decodes)
  per wave, run_all([K stage workers, 1 committer])   (src/utils/task_group.py, 3.12 TaskGroup)
    producer: every new photo on an asyncio.Queue, then K end markers
    stage worker (K = OP_MAX_INGEST_CONCURRENCY, default 16), per photo:
        engine call ║ ExifTool capture        (one inner run_all of two coroutines)
        -> detect filter -> embedding states -> blur, ids = PreparedImage (no classes yet)
        -> committer inbox; a photo failure is a result value, never an exception
    committer, at most 3 commit groups, one at a time:
        1. resolve_classes for the group's items (in memory; a class transaction only for a
           label new to the project); a photo whose class create failed leaves the group
           as failed (detector_infer) and is never claimed
        2. group transaction (one session): placement read, content claims, pending rows
        3. build documents (classes and placement present), then run_all of the group's
           write_prepared calls (writer -> crop cache), awaited before the next group
  finally: release the claims that must be released (L0 rule), publish crop.created of every
           photo written (one transaction); neither masks the exception that ended the wave
  no index refresh in ingest_batch (section 5)
```

A commit group starts when at least `ceil(n_live / 3)` photos wait in the inbox, or when no photo
is left in a stage worker; it takes every photo waiting at that moment, and the third group waits
for the last photo. The committer knows "no photo left" from a counter of photos handed to the
stage workers and not yet in the inbox; the stage workers stop at their end marker (the pattern of
`bulk_ingest/runner.py:355-360`; `asyncio.Queue.shutdown` is Python 3.13 and the venv is 3.12.3).

Write tasks are not created dynamically: the committer awaits its group's writes through
`run_all` before it starts the next group, so `create_task` is never called on a group that is
shutting down. The stage workers keep the GPU busy while a group commits.

Expected effect, one client, 32 COCO photos (an estimate from the timeline, to be measured):
head before the GPU is busy about 0.15 s (today 0.6 s of prefill and p1), GPU work about 1.25 s,
tail about 0.3 s (one group of about 11 photos). About 1.7 to 2.0 s per request, 16 to 19 photos
per second, against 9.9 today and 12.4 for the old code. At 4 clients and more the target is the
GPU-bound ceiling (about 27 to 28 on COCO).

### 4.2 The two stage implementations

| Stage | Used for | Code |
|---|---|---|
| engine stage | every project whose plan the engine runs, when `OP_INGEST_GPU_DECODE=1` | `run_gpu_decode` renamed `run_engine_stage`, plus the capture coroutine; `prepare_image`'s engine branch |
| host stage | a project with a secondary detector or a non-uniform letterbox fill (until A1 adds a second `detect` model); a deployment with `OP_INGEST_GPU_DECODE=0` (until E4); small photos only if S1 fails | today's per-photo functions: `_decode_image` on `run_cpu`, `detect_items` (single-image detector calls; the model server's dynamic batcher groups concurrent photos), `prepare_image`'s CPU branch |

The batched prefill (F1), `run_primary_batch` and `run_secondary_raw_batch` are deleted.

### 4.3 Bounds, backpressure and exceptions

| Level | Bound | Mechanism |
|---|---|---|
| per request | K photos in their stage (default 16) | K worker coroutines pulling from one `asyncio.Queue` |
| per request, database | one session at a time; the count in section 5 | the committer runs one group at a time; `bound_project_id()` is read before a session opens; `doc_context` runs outside the group transaction |
| per API process | engine calls in flight: the engine pool's `max_concurrent` (`triton_inflight()`, `src/config/api_pools.py:57-59`) | the pool's semaphore (per process, not shared state) |
| per process, memory | encoded megabytes held (`ingest_gate`) | unchanged gate; it counts encoded bytes only |
| whole card | the engine's admission, frame budget and region waits | RESOURCE_EXHAUSTED; the client retries with jittered exponential backoff until the photo's deadline (120 s), not a fixed 3 tries (F12) |

Exceptions (Python 3.12 `asyncio.TaskGroup`: the first failing child cancels its siblings and the
body, and the failures leave as an `ExceptionGroup`;
<https://docs.python.org/3/library/asyncio-task.html#task-groups>):

- every task group of the flow goes through `src/utils/task_group.run_all`, which re-raises the
  first child's exception unchanged, so `PoolTimeoutError` and `OperationalError` still reach the
  handlers of `src/core/db_session.py` (503 with `Retry-After`) and nothing turns into a 500;
- a per-photo failure is a value (`call_frame_engine` returns a reason, `detect_for_ingest`
  returns the exception, `prepare_image` and `write_prepared` return a failed `IndexOutcome`); no
  new broad `except`;
- a database failure in group 2 or 3 does not raise: see 4.4;
- a write whose waiting coroutine is cancelled (a bug elsewhere ended the wave) keeps running
  inside the writer (F13). It has no result, no event and no release; its photo is either written
  whole (complete, the retry answers `duplicate`) or left incomplete and repaired by the retry
  (L0). No cancelled write can leave a photo half stored for good.

### 4.4 Answers: per photo, per request, and what the callers do

Result vocabulary. `status` stays `success`, `duplicate` or `failed`. New `error_kind` values,
all **transient** (retrying can fix them), added to `src/services/curation/ingest_models.py` by
L0: `engine_unavailable`, `database_busy`, `incomplete_write`.

| Situation | Photos | Request answer |
|---|---|---|
| engine not ready at admission, the project uses the engine and the wave has a new photo | none processed | 503 `engine_unavailable`, `Retry-After: 5` (a handler like `database_busy_response`, F20) |
| an all-duplicate batch | `duplicate` each | 200; readiness is not asked |
| every new photo of the request ended `engine_unavailable` and nothing was claimed | none stored | 503 `engine_unavailable`, `Retry-After: 5` (nothing to release; resending is safe) |
| some photos `engine_unavailable`, others stored | per photo | 200, `status: partial` |
| pool timeout or connection failure before any group transaction committed | none stored | 503 `database_busy` / `database_unavailable` as today |
| the same in group 2 or 3 | stored photos keep `success`; photos of that group and every photo not yet committed answer `failed`, `database_busy`; the stage workers stop taking photos | 200, `status: partial`; release and publish still run (`finally`) |
| a photo found incomplete and younger than the repair age (L0) | `failed`, `incomplete_write` | 200 |
| `/ingest/image` (one photo, a person waiting) when its engine call fails or the engine is not ready | none stored | 503 `engine_unavailable`, `Retry-After: 5` |

What the callers do:

- **Routes.** `schedule_open_vocab_after_ingest` keeps running for the `success` results of a
  partial answer (`routers/curation/ingest.py:195-198`, `upload_ingest.py:105-108`), so a stored
  photo never loses its open-vocabulary pass; its events are published by the wave's `finally`.
- **Bulk run (lane L7).** A transient kind does not use up an attempt; the row becomes claimable
  again after a pause (`retry_at`), not at once. The sender's 503 retries (6 attempts, up to about
  2.5 minutes of backoff) cover an engine restart (E1: ready again 2.7 s after an engine restart;
  a model-server restart 29 s plus the reconnect of R3). A batch whose 503 retries run out is
  recorded with the transient kind `busy`, so a longer outage stops the run after 5 such batches
  (unchanged) and leaves its files `failed`, never `dead`; a resume stores them.
- **Engine readiness.** The verdict expires after 5 s and is dropped on the first engine call
  that fails after its retries (F21), so a warm worker sees an engine that died later.
- **Person waiting** (`/ingest/image`, `/ingest/upload`, `/ingest/objects`): a 503 with
  `Retry-After` during an engine restart; lane L2 checks in a browser what Cropwright's upload
  shows for it and reports it.

## 5. The commit phase, stage by stage

| Stage | Today (1 client / 4 concurrent, ms per 32-photo request) | Documented practice | Change | Expected | Measured by |
|---|---|---|---|---|---|
| resolve_classes | barrier B3; 185 / 862 (fresh projects: mostly first-time class creation under the registry row lock) | lock only what you write, in one order (<https://www.postgresql.org/docs/18/explicit-locking.html>, "Deadlocks") | per commit group, before the group transaction and before the documents | off the critical path; warm project near 0 | `class_create` timer; a warm-project row |
| placement | its own session per batch (`ingest_index.py:155-167`) | one transaction for the reads a write depends on | `place_new(session, ...)` inside the group transaction (it already takes a session). Each group reads the active partition, so one request can carry two `cluster_partition_id` values when a partition is promoted between groups (accepted: prototypes are immutable per partition, `clustering/assign.py:95-122`) | one session less per request | session count |
| claim and pending rows | one transaction per batch, 112 / 175 | sorted advisory locks (already) | one transaction per group, same `claim_in_session` | the tail pays one third | `claim` timer |
| doc context | 6 / 0 (cached) | | read once per wave, outside any transaction (the 5 s location-policy cache, F24) | none | |
| images and items writes | 647 / 697 at the end of the batch; 1.22 ms per document on the server, no rejections | bulk of 5 to 15 MB, `refresh=false` (<https://docs.opensearch.org/latest/api-reference/document-apis/bulk/>) | unchanged writer, plus L0's rules; photos arrive spread out, so writes overlap GPU work | write time leaves the critical path except the last group | `opensearch_write` timer |
| items pre-read (`mget`) under a new images doc | one `mget` per flush, all misses | | measure only (L5) | a few ms per flush; adopt only at 5% of write time or more | `scripts/bench/ingest_write_path.py` |
| crop cache | 140 / 136, after all writes | | per photo right after its own write; best effort as today (`source_image_cache.py:80-112`); all crops are engine JPEG bytes | overlapped | a new `crop_cache` stage (added to `STAGES`, the ingest dashboard and `tests/test_monitoring_metrics.py`, L2) |
| publish (`crop.created`) | 281 / 632 before L3 | SQLAlchemy "insertmanyvalues" (<https://docs.sqlalchemy.org/en/21/core/connections.html#insert-many-values-behavior-for-insert-statements>) | **done (L3)**: one statement per publish | 281 -> about 10 to 30 ms (to be confirmed by the timeline) | `publish` timer |
| wake-up | sync `PUBLISH` on the loop thread per commit (F5) | `redis.asyncio` for async code (<https://redis.readthedocs.io/en/stable/examples/asyncio_examples.html>) | unchanged: one call per publish | none | risk, section 10 |
| refresh | explicit per request (F14); 2,272 refreshes (53 s) and 144 merges (53 s) over the refresh rows | "Refresh operations are resource intensive ... wait for OpenSearch's periodic refresh rather than performing explicit refreshes when possible"; every 1 s while searched, an idle shard refreshes at its first search (<https://docs.opensearch.org/latest/api-reference/index-apis/refresh/>) | see the table below | fewer small segments and merges under load | node stats per row |

Refresh after the change, by caller:

| Caller | After | Promise |
|---|---|---|
| `POST .../ingest/batch` (bulk run, API clients) | no refresh | the photo is visible to search within the refresh interval (1 s), or at the first search of an idle shard |
| `POST .../ingest/image` | one refresh in the route (`routers/curation/ingest.py:103`, unchanged) | visible when the answer arrives |
| `POST .../ingest/upload`, `POST .../ingest/objects` | one refresh in `stage_and_ingest` after `ingest_batch` (`upload_ingest.py`, L2) | visible when the answer arrives |
| the bulk run's end | no refresh: nothing in the job body reads the run's writes by search | as `/ingest/batch` |
| `schedule_open_vocab_after_ingest` | stamps all ids in one bulk update with `refresh=wait_for` ("wait for the changes to become visible before replying", bulk API page), then starts the drain | the drain's search sees every stamped image (fixes F23's miss today); the 120 s sweep stays as the safety net |

Sessions per wave, from the code (F24), counted from `run_ingest_batch`'s entry:

```
1 lookup + 1 per imohash the project already stores (re-scans, copies) + 1 if an orphan row is released
+ up to 3 class transactions (only for labels new to the project)
+ up to 3 group transactions + 1 location-policy read (only when its 5 s cache is cold)
+ 1 release (only when a write failed and its images doc was not created, L0) + 1 publish
```

Fresh photos in a warm project with a warm policy cache: **1 + 3 + 1 = 5 at most**, one at a
time. The test counts them two ways: the unit test keeps the counting factory but drives an engine
fake that answers one photo at a time (so 3 groups really form) for 1, 8, 32 and 64 photos, and
adds a duplicate-rich batch whose expected count is the formula; the database-tier test (pending
wave run) counts real pool checkouts with SQLAlchemy's `checkout` pool event
(<https://docs.sqlalchemy.org/en/21/core/events.html#sqlalchemy.events.PoolEvents.checkout>) and
asserts the formula and a peak of one.

## 6. Streaming the batch response (D3)

Do not stream the HTTP body in this plan.

- At one client the client sends its next batch only when the response ends, streamed or not.
  What fills the GPU at one client is the per-photo flow inside the request (section 4).
- The bulk runner keeps 16 batches in flight (F16) and settles rows per batch.
- A streamed body sends 200 before the work, so the 503 answers of 4.4, which the sender retries,
  would need an in-stream error record and a new client contract. FastAPI's `StreamingResponse`
  streams an async generator's chunks
  (<https://fastapi.tiangolo.com/advanced/custom-response/#streamingresponse>); nothing in it
  carries a late status.

Revisit when the upload UI needs per-photo progress: an NDJSON variant of the route, with its
contract.

## 7. Seams left open, not built

| Seam | What stays open | Rule for the code of this plan |
|---|---|---|
| A second encoder for crops (a smaller crop encoder measured 8.5x faster at equal crop quality) | `EnginePlan.embed` names one model for frame and crops today | build the plan in one function, `ingest_engine_plan(service)` in `ingest_gpu_decode.py`, that reads frame and crop encoders separately (both PE today); the item space's vector field and dimension come from `item_write_fields()` |
| faces, OCR, derivatives branches | the engine already returns per-branch status and outputs | the engine stage keeps the whole `EngineAnswer` on `PreparedImage` (one field) |
| a vector store other than OpenSearch | vectors travel inside documents built by `build_documents` | all ingest writes go through `service._bulk_index` -> `IngestWriter.write` |
| embedding selection inside the engine | F10 | send `embed_classes` (detector class ids whose labels pass the detect filter's and the policy's class lists, by name: D2 Q4); score, area and per-image caps stay in the API until P6 makes such a policy the default |

## 8. Work breakdown

Each lane: its own worktree and branch off `int/b4-stateless`, light checks only, red test first,
one commit per concern, no push, no merge. Files are owned by one lane at a time.

| Lane | Owns (files) | Starts | Depends on |
|---|---|---|---|
| L0 a photo is stored whole or not at all | `src/services/curation/ingest_writer.py`, `ingest_items.py` (`_bulk_index`), `ingest_dedup.py`, `ingest_models.py`, `ingest.py` (`ingest_image` only), `src/services/near_duplicates/content_key.py`, `src/clients/curation_opensearch/bodies_core.py` and the mapping-ensure path for the new field, `ingest_index.py` (only `write_prepared`, the `ContentClaim` line and `ImageContext`), `scripts/bench/ingest_consistency.py` (new), their tests | now | none |
| L3 publish in one statement | | **merged** (`1fa4f3c82`) | |
| L4 commit groups | `src/services/near_duplicates/ingest_claim.py`, `src/services/clustering/assign.py` (only if `place_new`'s callers need it), `ingest_index.py` (everything except L0's three places), the placement and claim tests of `tests/curation/test_ingest_finish_batch.py`, tests of `index_items`' other callers (F25) | now on `ingest_claim.py` and tests; `ingest_index.py` after L0 merges (the coordinator merges L0 first) | L0 for `ingest_index.py` |
| L7 transient failures in the bulk run | `src/services/curation/bulk_ingest/` (`sender.py`, `runner.py`, `ledger.py`, `source_queue.py`), `src/db/repos/ingest_sources.py`, `src/db/models/datasets.py` (the `IngestSourceFile` model), one new migration under `src/db/migrations/versions/`, their tests | after L0 merges (imports L0's kinds) | L0 |
| L1 engine CPU decode, then S1 | `frame_engine/prep_geometry.py`, `engine.py`, `host_decode.py` (new), `prep_budget.py`, `settings.py`, `docker/frame-engine/requirements.txt`, `docker/frame-engine/Dockerfile`, `src/domain/decode_route.py`, `src/domain/frame_engine_wire.py`, `scripts/bench/engine_direct_bench.py` (new, unless `rg -l frame_engine scripts/` finds a direct harness), `tests/frame_engine_tests/` | after the colour and faces lanes merge | none |
| L2 per-photo flow | `src/services/curation/ingest_finish.py`, `ingest_batch.py`, `ingest.py`, `ingest_gpu_decode.py`, `ingest_detect.py`, `upload_ingest.py`, `open_vocab_ingest.py`, `src/clients/frame_engine.py`, `src/core/error_handlers.py`, `src/utils/stage_timing.py`, `monitoring/dashboards/ingest-pipeline.json`, `src/config/curation.py`, `src/config/retired_env.py`, `env.template`, `docs/CURATION.md`, the ingest tests under `tests/curation/` not owned by L0 or L4, `tests/test_monitoring_metrics.py` | last | L0, L4, L7, L1 merged; S1 decided |
| L5 items create-first | `src/clients/occ_upsert.py`, `scripts/bench/ingest_write_path.py` | measurement in the wave after L2 | L2 |
| L6 delete the host stage | the host-stage paths of 4.2 | after E4 and A1's second detector | not gated here |

### L0: a photo is stored whole or not at all (independent of the engine)

The decision: **the images doc lists its items, and a duplicate is decided only after checking
that they exist.**

| Design | Covers the failed items step | Covers a kill between the bulks | Cost on the happy path | Why not chosen |
|---|---|---|---|---|
| Delete the own images doc before releasing the claim, plus a marker for the kill window (the review's proposal) | yes, if the delete succeeds | needs a second mechanism | none | the compensating delete can fail too, and the kill window still needs the second mechanism: two mechanisms for one rule |
| A completion flag on the images doc set by an update after the items | yes | yes | one more write per photo, which re-indexes the frame vector (D5) | slower, and a flag alone cannot tell a dead writer from one still writing |
| `stored_at` on the PostgreSQL content row | yes | yes | one more statement per request | the row and the index disagree after a kill anyway; still needs a liveness rule |
| **The images doc carries `item_ids`; `duplicate` requires every listed item to exist (realtime `mget`); an incomplete image is repaired by the next attempt** | yes | yes | none (one `mget` only when a lookup or a `create` finds a stored image) | chosen: one rule at the two places a duplicate is decided, no extra write |

Rules:

1. The writer adds `item_ids` (the unit's crop ids, sorted) to the images doc it creates; the
   field is mapped like `ingest_write_id` (`keyword`, `index: false`, `doc_values: false`,
   `bodies_core.py:84`) and added to existing indexes through the mapping-ensure path. Existing
   images docs get it from a backfill in the same lane (the item ids of each image, read by an
   items query); after the backfill a doc without the field counts as incomplete.
2. A unit fails as a whole when any of its item docs is refused (F19): `failed`, `bulk_index`.
3. A unit whose images doc this unit created keeps its content claim when it fails; only a unit
   whose images doc was not created releases it (`write_prepared`).
4. `resolve_contents` (`ingest_dedup.py`): for every holder it already reads, check `item_ids`
   with one realtime `mget` on the items index (`_source: false`) for the whole batch. All found:
   `duplicate` as today. One missing, and the images doc older than `REPAIR_AFTER_S`: `Resolved`
   marks the photo **repair** (the stored doc's `_seq_no`, `_primary_term` and `item_ids`). Younger:
   `failed`, `incomplete_write` (a writer may still be writing; the bulk run retries later).
   `REPAIR_AFTER_S` is above the longest a flush can take (the OpenSearch client timeout times its
   tries plus the 429 pauses, `ingest_writer.py:61`); the lane reads the client's timeout and
   writes the value with that derivation. The age is the images doc's `indexed_at`.
5. A repair photo is ingested normally; its claim is `ContentClaim(..., repair=True)` and
   `claim_in_session` treats the row of its own image id as its own (stored, not held). The
   writer writes its images doc with `index` and `if_seq_no`/`if_primary_term` (OpenSearch
   optimistic concurrency: the write succeeds only if the document has not changed since it was
   read, else 409;
   <https://docs.opensearch.org/latest/api-reference/document-apis/index-document/>), then its
   items as usual (OCC upsert with the human-field guards), then deletes the stale ids (old
   `item_ids` minus new) that carry no human or import-validated field (`class_write_locked`). A
   409 on the repair write: re-check; complete answers `duplicate`, otherwise `incomplete_write`.
6. `_settle_conflicts` (a 409 on a plain `create` with another write id): `duplicate` only when the
   stored image is complete; otherwise `failed`, `incomplete_write`.
7. A content row whose images doc is missing is still released by the lookup (orphan,
   `ingest_dedup.py:74-77`); the images-doc `create` stays the second guard.

Red-first tests (unit level, `tests/curation/ingest_fakes.py` fakes):

- the items step raises after the images doc was created: the claim is kept; the next ingest of
  the same bytes writes every item and answers `success` (today: `duplicate`, no items);
- the writer stops between the two bulks (fake kill: images doc, no items, claim held): the next
  ingest, past `REPAIR_AFTER_S`, repairs it; before it, answers `failed`/`incomplete_write`;
- one item doc refused: the photo answers `failed`, `bulk_index` (today `success`);
- a repair deletes a stale machine item and keeps a stale item with a human class;
- request A claims, request B looks up before A's images doc exists (B releases A's row as an
  orphan), then A writes: one images doc, its items complete, B answers `duplicate` or
  `incomplete_write`, never a second image;
- a complete stored image answers `duplicate` with one extra `mget` and no write.

Sweep additions (pending wave run): after every kill and outage test, run
`scripts/bench/ingest_consistency.py --project <slug>`: for every images doc, every `item_ids`
entry exists and no item of the project points to a missing images doc; every content row has its
images doc. Bar: 0 inconsistencies.

### L4: commit groups

1. Red: one group transaction places, claims and marks pending in one session
   (`commit_group(service, prepared) -> list[GroupOutcome]`; `GroupOutcome` holds the claim
   outcome); `place_prepared` takes the caller's session.
2. Keep `claim_and_mark_each` as the fall-back of a group transaction that fails as a whole; the
   pool timeout is re-raised to the caller, which decides by 4.4.
3. Split `commit_prepared` into `commit_group`, `write_prepared` (per photo, L0's rules) and
   `finish_commit(service, written, unclaimed)` (release, then one publish, both in `finally`).
4. `index_items` calls the three in order; red tests for its callers (F25): a reprocess and an
   import (`created` False, no claim) still write and announce as today.
5. Pending wave run: `tests/curation/ingest_db/` claim and pending tests.

### L7: transient failures in the bulk run

1. Red: a file failed with a transient kind (`engine_unavailable`, `database_busy`,
   `incomplete_write`, `busy`) keeps its attempt count and is not claimed again before its
   `retry_at`; a terminal kind still uses up the attempts; another kind uses one.
2. Red: a batch whose 503 retries ran out records `busy` for its files; 5 such batches in a row
   stop the run (unchanged) with no file `dead`.
3. Code: `retry_at` column (migration), `_MARK_FAILED` without the increment for transient kinds,
   the claim query skipping rows whose `retry_at` is in the future.
4. Pending wave run: the database-tier ledger tests.

### L1: the engine decodes every image

1. Red: `tests/frame_engine_tests/test_prep_geometry.py`: `route()` of a PNG, a progressive JPEG,
   a CMYK JPEG and a JPEG below the gate answers `('cpu', <reason>)`; a corrupt file answers
   `corrupt`; `huge` stays a refusal.
2. Red: `frame_engine/host_decode.py` returns exactly the pixels of
   `src.utils.image_decode.decode_image` (generated JPEG, PNG, WebP, GIF, progressive and
   EXIF-rotated files; numpy only).
3. Red: `tests/frame_engine_tests/test_engine.py` with the existing fakes: a CPU-route photo runs
   every branch, answers `ok` with `decode_route` `cpu_<reason>`, and reserves its frame bytes.
4. Delete in the same change: the refusal statuses that became CPU reasons.
5. `scripts/bench/engine_direct_bench.py` if no direct harness exists.

**S1 gate (decides 3.1).** COCO 2,000, the 4,000-photo mix and 200 photos of set P, direct to
the engine, plan `detect`, `embed_frame`, `embed_crops`, `crops: inline`, engine workers 16, 32
and 64 (nvJPEG lanes equal to workers):

| Measure | Bar |
|---|---|
| COCO photos per second at 32 in flight | at least 26 with the GPU at least 95% busy (ceiling about 27 to 28; the old code's row 25.7) |
| set P photos per second at 32 in flight | at least 40 (E0: 44 to 47 direct; the old code's API row 35.5); engine queue and decode ms per photo reported |
| COCO p50 at one photo in flight | at most 150 ms |
| engine host CPU per COCO photo | at most 20 ms |
| outputs for CPU-decoded photos, structure | every photo `ok`; classes equal; every box matched one to one by overlap with today's API CPU path (the share at IoU 0.99 and at 0.95 and the unmatched count are reported); blur sums equal; the cosine distribution of crop and frame vectors is reported |
| outputs, ground truth | COCO val2017 box AP and AP50 through the engine route within 0.5 points of the API CPU path; text-to-image recall at 1 and at 10 and one crop-level retrieval number within the harness's run-to-run floor |

The output bar was changed on 2026-11-14 after its first measurement, and the reason is recorded
here because a bar must not move quietly. Revision 2 asked for equal crop ids (the SHA-256 of the
box rounded to 6 decimals) and a crop-vector cosine median of at least 0.9999 against the API CPU
path. On 1,680 public photos the engine route answered every photo, with equal classes and equal
blur sums, but equal crop ids only where the detector letterbox is the identity (1,031 of 1,031
photos), and on about 7% of the photos whose letterbox is resampled; the crop cosine median was
0.99947. The decode is not the cause (PNG, WebP and BMP give the same numbers as JPEG): the engine's
resampler keeps a float intermediate where Pillow rounds to 8 bits after the first pass, so about
3% of letterbox pixels differ by one level. That holds for the GPU route merged in E1 as well, so no
engine route could meet the old bar, and matching Pillow's arithmetic bit for bit would bind the
engine to the path this plan deletes. The same bytes are never detected twice (the content claim
answers `duplicate`), and the two places that compare a new detection with stored items match by
overlap, not by id: `reprocess detect` (its merge plan, `src/services/curation/reprocess_detect.py`)
and the repair of L0. So the engine is judged by ground truth and by one-to-one box matches.

### L2: the per-photo flow

1. Red: with an engine fake answering one photo at a time, the first photo's documents are
   written before the last photo's engine answer.
2. Red: the session counts of section 5 (1, 8, 32, 64 photos; a duplicate-rich batch), peak one.
   This replaces "same count for 8 and 32 photos" (F15).
3. Red: the answers of 4.4, one test each: engine not ready (503 with `Retry-After`, and not for
   an all-duplicate batch), every photo `engine_unavailable` (503), a partial engine failure (200
   partial), `PoolTimeoutError` in group 1 (503 `database_busy` through `run_all`'s unwrap) and in
   group 2 (group 1 `success` with its events published and open-vocab scheduled, the rest
   `failed`/`database_busy`, 200), `/ingest/image` during an engine restart (503), the readiness
   verdict expiring after 5 s and dropped after a failed call. These replace the CPU-fallback tests
   of `test_ingest_gpu_decode.py` and `test_ingest_gate_gpu_route.py` (behaviour removed by D1).
4. Red: for `assigns_class` true and false and policy modes `all`, `selected` (classes), `lazy`
   and `none`, the embedding mask, the region seeding and the built documents are equal to
   resolving classes first (F6).
5. Red: the engine plan carries `embed_classes` for a policy or detect filter with classes.
6. Red: `run_ingest_batch` does not refresh; `stage_and_ingest` refreshes once;
   `schedule_open_vocab_after_ingest` stamps in one bulk with `refresh=wait_for` before it starts
   the drain. `tests/curation/test_ingest_read_your_writes.py` moves to the new owners of the
   refresh; every promise it documents is kept or listed in the lane report before it changes.
7. Code, in order: `run_engine_stage`; `finish_photos` per 4.1 (stage workers, committer, end
   markers, `run_all`); `run_ingest_batch` without prefill or refresh; `ingest_one` through the
   same flow; overload retry until the photo's deadline; the 503 handler; `crop_cache` in
   `STAGES` and the dashboard (the `gpu_decode` stage keeps its name for the engine call).
8. Delete in the same change: `_prefill_gpu_decode`, `_prefill_detections`,
   `WholeImageDetector.run_primary_batch` and `run_secondary_raw_batch` (if `rg` finds no other
   caller), `PhotoJob.prefilled_image`, `prefilled_items`, `prefilled_secondary`; when S1 passed:
   `gpu_route`, the API size gate, `OP_INGEST_GPU_MIN_MEGAPIXELS` (into `retired_env.py`, out of
   `env.template` and docs), the `cpu_*` routes only the API fallback produced.
9. After a wire change: `make contracts`, and commit `contracts/` in the same commit.
10. Open a page in a browser: an upload in Cropwright during an engine restart; report what it
    shows.

Checks per lane (light only):

```bash
nice -n 19 .venv/bin/python -m pytest -q --no-cov <the lane's test files>
.venv/bin/pre-commit run --files <touched files>
.venv/bin/pre-commit run --hook-stage pre-push --files <touched files>
rg -n "_prefill_detections|run_primary_batch|run_secondary_raw_batch|prefilled_image" src tests   # empty after L2
rg -n "OP_INGEST_GPU_MIN_MEGAPIXELS" src tests docs env.template                                # only retired_env.py after L2
```

## 9. Gates (coordinator's wave runs)

| Gate | How | Bar |
|---|---|---|
| Offline suite and hooks | `.venv/bin/python -m pytest tests/ -q --no-cov -m 'not live and not families'`; `.venv/bin/pre-commit run --all-files` and the pre-push stage | green |
| Database tier | the ingest, claim, ledger and event tests with the database marker, including the real session-checkout count | green; sessions per the formula of section 5 |
| Detection and vector parity | `scripts/bench/parity.py capture` on COCO and Objects365, reference captured twice on the current head, then `compare` | GPU-route photos as E1; CPU-decoded photos per S1's output bars; same documents and crop ids for the same input (except `decode_route`, write ids, times, `item_ids` ordering) |
| Four-way rows | the refresh's row set on NVMe: mix, set P, COCO at 4 clients; COCO, 12 MP JPEG, 20 MP and 12 MP PNG at one client; COCO at 8 and 16 clients | **not below the old code on NVMe on any row**; GPU busy (mean and p50) per row; failed 0, else invalid |
| Stage timeline | the probe of section 0 on the new head | GPU busy outside the embed window; the tail after the last engine answer at most one group's commit |
| Stored whole | `scripts/bench/ingest_consistency.py` after each test below | 0 inconsistencies |
| Kill and restart | 50 engine kills during a bulk run; one model-server restart under light load; one API worker killed mid-request; one OpenSearch items bulk failed by fault injection | model server never restarts; after the run and one resume every file is `done` and none is `dead`; the consistency check passes |
| Outage | the engine stopped for 60 s and for 5 minutes during a bulk run | 60 s: no file failed (the sender's 503 retries); 5 minutes: the run stops, no file `dead`, a resume stores every file |

## 10. Risks and what would change the choice

| Risk | Sign | Response |
|---|---|---|
| Small photos are slower through the engine | S1 fails | (c), no new code; or raise engine workers if the gap is in the queue |
| The engine is the first limit on large photos (20 MP proof row: GPU 70%, engine queue 285 ms and decode 128 ms per photo, F11) | set P row below 35.5 after L2 | engine workers and lanes from S1; a second engine process per card if threads do not help |
| The engine becomes the host-memory bound (queued requests hold encoded bytes) | S1's memory per queued photo | size `OP_FRAME_ENGINE_MAX_IN_FLIGHT` from S1 |
| Commit groups add contention between concurrent requests of one project | claim time grows at 16 clients | groups of a third, never smaller; sorted lock order |
| A repair races a live writer of the same content | two writers of one image within `REPAIR_AFTER_S` | the age rule; the OCC write of the images doc; the consistency check in every sweep |
| `REPAIR_AFTER_S` too short (a flush longer than the derived bound) | consistency check after a slow-store test | derive it from the client timeout; raise it, never lower it below the flush bound |
| Valkey outage blocks the loop up to 1 s per publish (F5) | loop lag p99 | outside this plan; `redis.asyncio` when measured |
| OpenSearch 3.9.0 costs more per document than 3.6.0 | write window still large after the flow change | the diagnosis's experiment 3, only if the rows miss the bar |

## 11. Relation to 13 (deviations, with the reason)

| 13 says | This plan | Why |
|---|---|---|
| D3: "the batch route streams" | per-photo flow inside the request; the HTTP body is not streamed | section 6 |
| D2: a project with a secondary detector keeps today's route until a second `detect` model lands | kept, but through the per-photo host stage; its batched detector calls are deleted | one flow instead of two; the dynamic batcher groups per-photo calls (<https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html>) |
| D1: engine down means the image fails visibly | kept, as a 503 with `Retry-After` when nothing was stored and transient per-photo failures otherwise | the sender and the bulk ledger retry them without using attempts (L7) |
| D1: "protocol version 1 does not change after E1" | additive changes only: `decode_route` values, `cpu_host` | the engine and the API ship together |
| E4: engine sizes per profile | `OP_INGEST_GPU_DECODE=0` keeps the host stage until E4 lands | no deployment loses ingest before its engine fits |
| P0: at most 5 sessions per batch | the formula of section 5; 5 at most for fresh photos in a warm project | the bound no longer depends on the photo count; the other terms exist today too |
| D3: "one explicit refresh when a run or request ends only where a caller reads its own writes" | the per-caller table of section 5; no refresh at a bulk run's end | nothing reads a run's writes by search at its end |

## 12. Questions and the coordinator's answers (2026-11-14)

| # | Question | Answer |
|---|---|---|
| 1 | Secondary detector and non-uniform letterbox fill stay on the host stage until A1? | Yes. |
| 2 | If S1 misses a bar by 15% or less, tune engine workers and admission before falling back to (c)? | Yes, one extra S1 hold. A miss on the output bars is never tuned away. |
| 3 | The batch response stays unstreamed (section 6)? | Yes. Revisit with the upload UI. |
| 4 | The session bound restated as in section 11 (at most 5 in the steady state)? | Yes. The test names the two additions (a label new to the project, a failed write) so the bound cannot drift further. |
| 5 | `OP_INGEST_GPU_DECODE=0` keeps the host stage until E4, and `1` means the engine is required (no fallback)? | Yes. |
| 6 | Remove the per-request refresh from batch ingest, keep it on `/ingest/image` and `/ingest/upload`, one refresh at the end of a bulk run? | Yes, with one condition: L2 reads `tests/curation/test_ingest_read_your_writes.py` first and keeps every read-your-writes promise the routes document; a promise that needed the per-request refresh is listed in the lane report before it is changed. |

Answers to revision 2 (after the independent review):

| # | Question | Answer |
|---|---|---|
| 7 | No refresh at the end of a bulk run (revision 2 found no reader of the run's writes by search in the job body)? | Yes; this replaces the last part of answer 6. |
| 8 | L7 adds a `retry_at` column through a migration? | Yes. |
| 9 | A repair may delete stale machine items, never an item with a human or import-validated field? | Yes. One more rule for L0: `item_ids` must stay true for every writer that adds, replaces or removes the items of an image (re-detection, dataset import, the open-vocabulary run, any delete). L0 lists those writers first and reports the rule for each before it codes; an image whose items were changed on purpose must never look incomplete. |
| 10 | The engine 503 carries `Retry-After: 5`? | Yes. |
| 11 | L0 and L4 both touch `ingest_index.py`? | One writer per file: L4 starts only after L0 is merged, on the whole of its file list. |

Order fixed by the coordinator: L3 is merged. L0 starts now. L4 and L7 start after L0 merges. L1
starts after the colour and faces lanes merge, because they change the engine's decode path and
its branch list. L2 is last.

## Appendix A. Review findings and how each was resolved

Review of revision 1 (data safety), checked against the code at `b01dac70d`.

| Finding | Verified | Resolution |
|---|---|---|
| B1 a late group failing turns the request into a 503 after earlier groups are stored | yes, for the revision-1 design (`ingest_claim.py:118-119` re-raises; revision 1 put release and publish after the group) | 4.4: a database failure in group 2 or 3 answers per-photo `database_busy` and 200 partial; release and publish in `finally`; L2 test 3 |
| B2 dropping the refresh breaks the open-vocab drain; the promised refreshes have no code or lane | partly. The drain searches the stamped field, and the stamps are written after today's refresh (`open_vocab_ingest.py:72-73` runs after `ingest_batch.py:352`), so today's refresh never served the drain: the miss exists today and the 120 s sweep catches it. The missing refreshes for `/ingest/upload` and `/ingest/objects` and the missing lane are right | section 5 refresh table: `stage_and_ingest` refreshes; the stamps go in one bulk with `refresh=wait_for` before the drain starts (fixes today's miss); no end-of-run refresh (nothing reads it); files added to L2 |
| B3 images doc without items after a failed items step or a kill; the retry answers `duplicate` | yes (F18), in today's code. Also found: a refused item is silent (F19) | lane L0 |
| M1 the admission 503 has no `Retry-After`; the readiness verdict never expires; outages use up attempts | yes (F20, F21, F22) | 4.4 and L7: handler with `Retry-After`, 5 s verdict, all-`engine_unavailable` answers 503, transient kinds do not use attempts, `busy` for exhausted 503 retries; the outage gate of section 9 |
| M2 `/ingest/image` during an engine restart | yes (`ingest.py:345-366` has no admission step) | 4.4: 503 `engine_unavailable` with `Retry-After` |
| M3 F6 incomplete: `assigns_class`, region seeding; resolve before the group transaction | yes (`item_doc.py:45-56`, `region_scope.py:34-43`, `ingest_detect.py:228-243`) | F6 rewritten; 4.1 step 1 resolves before the transaction and before the documents; L2 test 4 extended |
| M4 TaskGroup facts in Python 3.12 | yes (venv 3.12.3; `utils/task_group.py:22-29` unwraps) | 4.1 and 4.3: `run_all` everywhere, end markers, no dynamic write tasks, cancelled writes covered by L0 |
| M5 the session bound is not what the code opens | yes (F24); the P0 test counts stub calls (F15) | section 5 formula; unit test with a one-at-a-time engine fake plus a database-tier checkout count |
| M6 B6 incomplete | yes | B6 row rewritten |
| m1 F1 misses the engine-photo barrier; F4 slug lookup per event | F1 yes; F4 fixed by L3 (one resolve per publish, `events.py:133`) | F1 and F4 updated |
| m2 no `crop_cache` stage | yes (F26) | L2 adds it to `STAGES`, the dashboard and the metrics test |
| m3 two partition ids in one request | yes | accepted and stated in section 5 |
| m4 a document without its event after a kill | yes; harmless (the consumers count events) | no change |
| m5 "boxes equal" must mean equal crop ids | superseded: the S1 output bar is ground truth and one-to-one box matches (see S1) | S1 output bar |
| m6 a lookup can release another request's live claim | yes | kept (the images-doc `create` stays the guard); L0 test |
| m7, m8 | yes; m8's "crop cache fails: request fails" was stale in revision 1 | crop cache stated as best effort |
| m9 other callers of `index_items` | yes (F25) | L4 step 4 |
| Coordinator answer 6: "one refresh at the end of a bulk run" | the job body has no reader of the run's writes by search | not built; stated in section 11 for the coordinator |
