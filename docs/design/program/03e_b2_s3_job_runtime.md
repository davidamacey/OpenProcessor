# B2 stage 3: the job runtime, the reaper and the event feed

| Field | Value |
|---|---|
| Status | plan, not started |
| Depends on | stage 1 merged; 3B branches from the stage-1 merge and codes alongside stage 2; **3A branches after 2A merged** (it reads projects through 2A's repository, review fix R-18); both merge after stage 2 |
| Design | [02b](02b_jobs_locking_events.md) sections 1-4 and 6 |
| Lanes | 3A, 3B in parallel |
| Merge order | 3B, then 3A (both after stage 2's 2F). 3A's wrapper calls two 3B functions whose signatures 3B-1 fixes (`repos/events.py::insert_event(session, project, type, payload)` and `event_wakeup.wake(project)`); 3A codes against them and merges the integration head (with 3B) into its branch before its final test run (review R-18) |
| Compose project | none in the lanes |

Stage 3 builds the machinery every family uses in stages 4 and 5; it moves no family itself. The
B1 lease code stays until the last caller is converted (deleted in stage 6, lane 6A).

---

## Lane 3A `b2/job-runtime`

**Owns:** new `src/services/jobs/{__init__,service,reaper,inline,status}.py` (API side);
`src/db/repos/jobs.py`; new `src/tasks/runtime/{job_task,context,register,heartbeat}.py` (review fix
R-2: the heartbeat, the fenced progress and checkpoint writes and the terminal write live where the
GPU images can import them; the OpenSearch client factory is injected by the handler module, never
imported by `src/tasks/runtime/`); new
`src/tasks/reaper.py` (created empty in stage 1); `src/tasks/beat_schedule.py` and
`src/config/task_routes/periodic.py` (the reaper entries only; 6A adds the rest); new
`src/domain/job_timing.py`; `src/services/projects/busy.py`; new routers
`src/routers/curation/jobs.py` and `src/routers/jobs.py` and their mounting in `src/app/factory.py`;
new schemas `src/schemas/jobs.py`; the stream-worker project claims (3A-10): new
`src/db/repos/stream_claims.py`, the claim loop in `scripts/curation/worker/flow.py` and
`scripts/curation/vlm_worker.py`; tests under `tests/jobs/`, `tests/tasks/test_job_task_wrapper.py`,
`tests/tasks/test_reaper.py`, `tests/db/test_job_*.py`, `tests/workers/test_stream_claims.py`,
`tests/test_jobs_single_entry.py`.

| ID | Task | Docs | Red test | Control |
|---|---|---|---|---|
| 3A-1 | `src/domain/job_timing.py` (`LEASE_TTL_S = 120`, `HEARTBEAT_TICK_S = 15`, `FENCE_MARGIN_S = 30`, `PROGRESS_MIN_INTERVAL_S = 1`) | - | `tests/jobs/test_timing_constants.py`: `FENCE_MARGIN_S < LEASE_TTL_S`, `HEARTBEAT_TICK_S * 4 < LEASE_TTL_S` | - |
| 3A-2 | `src/db/repos/jobs.py`: `insert_queued`, `insert_running_inline`, `claim`, `heartbeat`, `progress`, `checkpoint`, `finish`, `interrupt`, `cancel`, `pause`, `resume`, `mark_dispatched`, the reaper selects ([02b](02b_jobs_locking_events.md) sections 2-4, the SQL there verbatim); `job_attempts` rows; every write fenced | [locking clause](https://www.postgresql.org/docs/current/sql-select.html#SQL-FOR-UPDATE-SHARE), [READ COMMITTED](https://www.postgresql.org/docs/current/transaction-iso.html#XACT-READ-COMMITTED) | `tests/db/test_job_start_race.py`, `test_claim_and_fencing.py` (`db`, [02c](02c_migration_ops_tests.md) section 7) | a single start, claim, finish |
| 3A-3 | `src/services/jobs/service.py`: `start_job` (insert inside a savepoint, so a `unique_violation` aborts only the savepoint and the 409 read runs in the same transaction, review fix R-5; commit, enqueue, marker, dispatched; 409 `job_busy` with the active id; 503 `database_unavailable` and `queue_unavailable`), the `Idempotency-Key` handling (D-58), `cancel_job`, `pause_job`, `resume_job`; `src/services/jobs/inline.py::start_inline_job` | - | `tests/jobs/test_start_order.py`: the commit happens before `send_task` (fake producer records the order; trap T-9); broker down -> row `failed`, 503; same idempotency key twice -> one job, two equal answers; different body same key -> 422 | the happy path returns 202 and the id |
| 3A-4 | `src/tasks/runtime/heartbeat.py` (R-2): a thread with its own sync connection (never the task's), renew every 15 s, `Lost` sets the stop event, cancel and pause requests set their events, `may_write()` self-fence | B1's `lease_heartbeat.py` shape | `tests/jobs/test_heartbeat_thread.py` (`db`): with the handler blocking its event loop for 3 x the (monkeypatched, 2 s) TTL the lease stays live; a row deleted under it gives `Lost` and stops writes | the thread stops on `stop()` |
| 3A-5 | `src/tasks/runtime/{job_task,context,register}.py`: the wrapper of [02b](02b_jobs_locking_events.md) section 3.1 exactly; `register_job(kind, handler)` reads queue, limits and auto-resume from the kind table; `JobContext` (`progress` throttled, `checkpoint`, `cancelled`, `paused`, `may_write`, client factory with a per-task `NullPool` engine and the OpenSearch factory **injected** by `register_job(kind, handler, client_factories=...)` from the image's handler module, R-2); importable without FastAPI, OpenSearch client classes or torch, and naming no `src.services` module, not even lazily (`check_layering.py` checks function-level imports) | [Celery tasks](https://docs.celeryq.dev/en/stable/userguide/tasks.html), [testing](https://docs.celeryq.dev/en/stable/userguide/testing.html) (unit tests with mocks, never `task_always_eager`) | `tests/tasks/test_job_task_wrapper.py`: claims and runs once; a second delivery while running is an ack and no-op; terminal -> no-op; `attempt` 3 -> `failed`; exception -> `failed` with the message; cancel -> `cancelled`; `SoftTimeLimitExceeded` -> `failed (time_limit)`; `SystemExit` -> `interrupted` with the kind's auto-resume and re-raise; a `Lost` fence writes nothing; a job whose project was archived or deleted while it was queued ends `failed (project_unavailable)` without running (carried from the original E11); `tests/tasks/test_runtime_import_minimal.py` (fresh interpreter: no `fastapi`, `opensearchpy`, `torch` after importing `src.tasks.runtime`); `tests/tasks/test_engine_per_task.py` (trap T-1) | a no-op handler completes |
| 3A-6 | progress throttle: at most one write per `PROGRESS_MIN_INTERVAL_S` and on each stage change; snapshot payloads | - | `tests/jobs/test_progress_throttle.py` (trap T-31) | a stage change always writes |
| 3A-7 | the reaper: `src/services/jobs/reaper.py` (rules a-g of [02b](02b_jobs_locking_events.md) section 4, each a short transaction, `SKIP LOCKED`, the tick's advisory lock, Valkey `Unknown` changes nothing; rule b excludes rows whose group is busy and runs each row in a savepoint, review fix R-5), `src/tasks/reaper.py` (`op.periodic.reaper`, queue `periodic`, soft limit 120 s, beat entry every 30 s with `expires` 30 s) | [periodic tasks](https://docs.celeryq.dev/en/stable/userguide/periodic-tasks.html) | `tests/tasks/test_reaper.py` and `tests/db/test_reaper_rules.py` (`db`): each rule acts; a fresh lease is untouched; two reapers at once never requeue one job twice; marker `Unknown` -> no dispatch; an interrupted job whose group is busy does not block the batch, and an exhausted one moves `interrupted -> failed` without the trigger refusing it (R-5) | a clean database gives no writes |
| 3A-8 | `projects/busy.py`: "is this project busy" is one query on active `jobs` rows (no state-file scan) | - | `tests/projects/test_busy_from_jobs.py` (`db`) | an idle project is not busy |
| 3A-9 | job listing routes: `GET /curation/projects/{project}/jobs` (filter by kind and status, keyset pagination by `created_at, id`) and `GET /jobs` (deployment-level kinds), `GET /curation/projects/{project}/jobs/{job_id}`; read only; per-family status routes stay the family views and read the same row | FastAPI response models | `tests/jobs/test_job_routes.py`: a project sees only its jobs (RLS); pagination stable under inserts; contracts regenerated | - |
| 3A-10 | stream-worker project claims (D-53; designed in [02b](02b_jobs_locking_events.md) section 5): every pass upserts the worker's `worker_capabilities` row, counts live replicas from those rows, renews its claims, releases the excess above `ceil(active / live)` oldest first at a project boundary, then claims up to the fair share; the fencing token moves only on takeover (review fix R-8); paused projects skipped; claims released on shutdown; reaper rule g deletes expired claims (3A-7) | [02b](02b_jobs_locking_events.md) section 5 | `tests/workers/test_stream_claims.py` (`db`): one worker with three projects, then a second worker -> both hold projects within two passes; each project held by one; a killed worker's projects move after the TTL; a renewal keeps the token, a takeover bumps it | one worker behaves as today |
| 3A-11 | `tests/test_jobs_single_entry.py` (named in 02b section 2 and 10; no lane created it): AST scan, no `jobs` insert outside `src/db/repos/jobs.py`, no `send_task` outside `src/clients/task_queue.py` | - | a planted fixture fails | the real tree passes |

**3A done when:** every 3A test green (`-m db` included); the reaper runs from beat on the merged
branch (stage exit bring-up).

## Lane 3B `b2/events-pg`

**Owns:** new `src/services/events/{__init__,publish,listener,cursor,sse}.py`;
`src/db/repos/events.py`; new `src/tasks/runtime/event_wakeup.py` (the Valkey wake-up every image
can import, review fix R-2 and the owner decision of 2026-11-04 on R-6); `src/services/curation/event_hub.py` (deleted);
`src/routers/curation/events.py`, `src/routers/curation/pipeline_events.py`,
`src/routers/curation/global_status.py` (the SSE parts); `src/app/lifespan.py` (the event tail
replaced by the Valkey subscriber); `src/services/curation/job_lock.py` only to remove the event hub's use
(the file is deleted by 6A when its last caller goes); `frontend/` is **not** touched (the browser
re-sends `Last-Event-ID` itself; 4E checks the pages); tests under `tests/events/`.

| ID | Task | Docs | Red test | Control |
|---|---|---|---|---|
| 3B-1 | `publish(event_type, payload, *, project=None, session=None)`: same name and argument order as today's event-hub publish so call sites do not change; with a session it writes the row (`repos/events.py::insert_event`) in the caller's transaction, without one it opens its own short transaction; after the commit, `event_wakeup.wake(project)` publishes `op:events:<project id or global>` on Valkey (owner decision 2026-11-04 after review R-6: no `pg_notify`, no `LISTEN`); a failed wake-up is logged and ignored; payload cap 7,000 bytes | [redis-py pub/sub](https://redis.readthedocs.io/en/stable/advanced_features.html#publish-subscribe), [Valkey PUBLISH](https://valkey.io/commands/publish/) | `tests/events/test_publish_transactional.py` (`db` + fake Valkey): a rolled-back transaction writes no row and sends no wake-up; a committed one does both, the wake-up after the commit | a publish without a session commits on its own |
| 3B-2 | the cursor: `(xid, id)`, the `pg_snapshot_xmin` filter ([02b](02b_jobs_locking_events.md) section 6.3) | [snapshot functions](https://www.postgresql.org/docs/current/functions-info.html#FUNCTIONS-PG-SNAPSHOT) | `tests/db/test_events_cursor.py` (`db`): two transactions commit out of id order; a reader polling between the commits emits every event once, in cursor order | a single writer is emitted in id order |
| 3B-3 | the listener: one Valkey pub/sub subscription per API process (redis-py async, channels `op:events:<project>` for the projects with local subscribers plus `op:events:global`), reconnect with backoff, 2 s safety tick, each catch-up query on a pooled connection returned at once, per-project fan-out to local subscribers (bounded queues; a slow client is dropped with a final `reconnect` event, never blocks the others) | [redis-py asyncio pub/sub](https://redis.readthedocs.io/en/stable/examples/asyncio_examples.html) | `tests/events/test_listener_fanout.py` (`db` + fake Valkey): 3 subscribers on 2 projects get only their project's events; with every wake-up dropped, the 2 s tick still delivers each event once; a dropped subscription recovers and the catch-up delivers the gap; `tests/events/test_publish_from_worker.py` (`db`): a publish from a Celery-task context (no FastAPI app) reaches an API-side subscriber (carried from the original B4 4C-1) | - |
| 3B-4 | SSE routes on `fastapi.sse.EventSourceResponse` with `ServerSentEvent(id="<xid>-<id>", event=type, data=payload)`; `Last-Event-ID` resume; a cursor older than the oldest retained row gets one `reset` event first (review fix R-34); the routes never declare `DbSessionDep` (R-1); the two hand-written framers deleted (LIB-3) | [FastAPI SSE](https://fastapi.tiangolo.com/tutorial/server-sent-events/#resuming-with-last-event-id) | `tests/events/test_sse_resume.py`: publish 100, disconnect at 40, reconnect with the 40th id -> 41..100 in order, no duplicates; frames carry `id`, `event`, `data` | the existing SSE route tests with ids added |
| 3B-5 | lifespan: the event-hub tail and its `to_thread` appends deleted; the Valkey subscriber started and stopped per process (a Valkey outage leaves the 2 s tick running); `OP_EVENT_BUS` and `OP_EVENT_LOG_MAX_BYTES` retired (ENV-16) | - | `tests/test_lifespan_no_event_hub.py` (AST: no `event_hub` import, no JSONL path); retired-env test | - |
| 3B-6 | retention: events older than 24 h deleted by reaper rule f (3A owns the rule; 3B provides `repos/events.py::delete_older_than`) | - | `tests/db/test_events_retention.py` | - |

**3B done when:** `rg -n "event_hub|events\.jsonl|OP_EVENT_BUS" src` is empty; the file-lock
baseline loses the `event_hub.py` line; tests green.

## Stage 3 exit criteria

- every lane's tests green on the merged branch (after stage 2), `-m db` green, contracts clean;
- a coordinator bring-up (`-p op-b2-s3check`, CPU, 20 min cap): `op.infra.noop` started through
  `start_job` ends `completed`; its `job.state` events arrive on the project SSE stream; killing the
  worker mid-noop (a sleeping noop variant in `tests/live/`) leaves `interrupted` within 150 s;
  `down -v`.

## Lane briefs

Use the common block of [03a](03a_b2_w1_queue_infra.md), with: "B2 stage-3 lane", "cut from
int/b2-control-plane at the stage-1 merge (3B) or after 2A merged (3A, R-18); you code alongside
stage 2 and the coordinator merges you after stage 2", the reading list `03e_b2_s3_job_runtime.md` (your lane) and
[02b](02b_jobs_locking_events.md) sections 1-4 and 6. Compose project: none. GPU: none.
