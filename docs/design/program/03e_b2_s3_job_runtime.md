# B2 stage 3: the job runtime, the reaper and the event feed

| Field | Value |
|---|---|
| Status | plan, not started |
| Depends on | stage 1 merged (lanes branch from the stage-1 merge and code alongside stage 2); merges after stage 2 |
| Design | [02b](02b_jobs_locking_events.md) sections 1-4 and 6 |
| Lanes | 3A, 3B in parallel |
| Merge order | 3A, 3B (both after stage 2's 2F) |
| Compose project | none in the lanes |

Stage 3 builds the machinery every family uses in stages 4 and 5; it moves no family itself. The
B1 lease code stays until the last caller is converted (deleted in stage 6, lane 6A).

---

## Lane 3A `b2/job-runtime`

**Owns:** new `src/services/jobs/{__init__,service,heartbeat,reaper,inline,status}.py`;
`src/db/repos/jobs.py`; new `src/tasks/runtime/{job_task,context,register}.py`; new
`src/tasks/reaper.py` (created empty in stage 1); `src/tasks/beat_schedule.py` and
`src/config/task_routes/periodic.py` (the reaper entries only; 6A adds the rest); new
`src/domain/job_timing.py`; `src/services/projects/busy.py`; new routers
`src/routers/curation/jobs.py` and `src/routers/jobs.py` and their mounting in `src/app/factory.py`;
new schemas `src/schemas/jobs.py`; tests under `tests/jobs/`, `tests/tasks/test_job_task_wrapper.py`,
`tests/tasks/test_reaper.py`, `tests/db/test_job_*.py`.

| ID | Task | Docs | Red test | Control |
|---|---|---|---|---|
| 3A-1 | `src/domain/job_timing.py` (`LEASE_TTL_S = 120`, `HEARTBEAT_TICK_S = 15`, `FENCE_MARGIN_S = 30`, `PROGRESS_MIN_INTERVAL_S = 1`) | - | `tests/jobs/test_timing_constants.py`: `FENCE_MARGIN_S < LEASE_TTL_S`, `HEARTBEAT_TICK_S * 4 < LEASE_TTL_S` | - |
| 3A-2 | `src/db/repos/jobs.py`: `insert_queued`, `insert_running_inline`, `claim`, `heartbeat`, `progress`, `checkpoint`, `finish`, `interrupt`, `cancel`, `pause`, `resume`, `mark_dispatched`, the reaper selects ([02b](02b_jobs_locking_events.md) sections 2-4, the SQL there verbatim); `job_attempts` rows; every write fenced | [locking clause](https://www.postgresql.org/docs/current/sql-select.html#SQL-FOR-UPDATE-SHARE), [READ COMMITTED](https://www.postgresql.org/docs/current/transaction-iso.html#XACT-READ-COMMITTED) | `tests/db/test_job_start_race.py`, `test_claim_and_fencing.py` (`db`, [02c](02c_migration_ops_tests.md) section 7) | a single start, claim, finish |
| 3A-3 | `src/services/jobs/service.py`: `start_job` (insert, commit, enqueue, marker, dispatched; 409 `job_busy` with the active id; 503 `database_unavailable` and `queue_unavailable`), the `Idempotency-Key` handling (D-58), `cancel_job`, `pause_job`, `resume_job`; `src/services/jobs/inline.py::start_inline_job` | - | `tests/jobs/test_start_order.py`: the commit happens before `send_task` (fake producer records the order; trap T-9); broker down -> row `failed`, 503; same idempotency key twice -> one job, two equal answers; different body same key -> 422 | the happy path returns 202 and the id |
| 3A-4 | `src/services/jobs/heartbeat.py`: a thread with its own sync connection (never the task's), renew every 15 s, `Lost` sets the stop event, cancel and pause requests set their events, `may_write()` self-fence | B1's `lease_heartbeat.py` shape | `tests/jobs/test_heartbeat_thread.py` (`db`): with the handler blocking its event loop for 3 x the (monkeypatched, 2 s) TTL the lease stays live; a row deleted under it gives `Lost` and stops writes | the thread stops on `stop()` |
| 3A-5 | `src/tasks/runtime/{job_task,context,register}.py`: the wrapper of [02b](02b_jobs_locking_events.md) section 3.1 exactly; `register_job(kind, handler)` reads queue, limits and auto-resume from the kind table; `JobContext` (`progress` throttled, `checkpoint`, `cancelled`, `paused`, `may_write`, client factory with a per-task `NullPool` engine and `make_script_opensearch`); importable without FastAPI, OpenSearch client classes or torch (only the factory imports them lazily) | [Celery tasks](https://docs.celeryq.dev/en/stable/userguide/tasks.html), [testing](https://docs.celeryq.dev/en/stable/userguide/testing.html) (unit tests with mocks, never `task_always_eager`) | `tests/tasks/test_job_task_wrapper.py`: claims and runs once; a second delivery while running is an ack and no-op; terminal -> no-op; `attempt` 3 -> `failed`; exception -> `failed` with the message; cancel -> `cancelled`; `SoftTimeLimitExceeded` -> `failed (time_limit)`; `SystemExit` -> `interrupted` with the kind's auto-resume and re-raise; a `Lost` fence writes nothing; `tests/tasks/test_runtime_import_minimal.py` (fresh interpreter: no `fastapi`, `opensearchpy`, `torch` after importing `src.tasks.runtime`); `tests/tasks/test_engine_per_task.py` (trap T-1) | a no-op handler completes |
| 3A-6 | progress throttle: at most one write per `PROGRESS_MIN_INTERVAL_S` and on each stage change; snapshot payloads | - | `tests/jobs/test_progress_throttle.py` (trap T-31) | a stage change always writes |
| 3A-7 | the reaper: `src/services/jobs/reaper.py` (rules a-g of [02b](02b_jobs_locking_events.md) section 4, each a short transaction, `SKIP LOCKED`, the tick's advisory lock, Valkey `Unknown` changes nothing), `src/tasks/reaper.py` (`op.periodic.reaper`, queue `periodic`, soft limit 120 s, beat entry every 30 s with `expires` 30 s) | [periodic tasks](https://docs.celeryq.dev/en/stable/userguide/periodic-tasks.html) | `tests/tasks/test_reaper.py` and `tests/db/test_reaper_rules.py` (`db`): each rule acts; a fresh lease is untouched; two reapers at once never requeue one job twice; marker `Unknown` -> no dispatch | a clean database gives no writes |
| 3A-8 | `projects/busy.py`: "is this project busy" is one query on active `jobs` rows (no state-file scan) | - | `tests/projects/test_busy_from_jobs.py` (`db`) | an idle project is not busy |
| 3A-9 | job listing routes: `GET /curation/projects/{project}/jobs` (filter by kind and status, keyset pagination by `created_at, id`) and `GET /jobs` (deployment-level kinds), `GET /curation/projects/{project}/jobs/{job_id}`; read only; per-family status routes stay the family views and read the same row | FastAPI response models | `tests/jobs/test_job_routes.py`: a project sees only its jobs (RLS); pagination stable under inserts; contracts regenerated | - |

**3A done when:** every 3A test green (`-m db` included); the reaper runs from beat on the merged
branch (stage exit bring-up).

## Lane 3B `b2/events-pg`

**Owns:** new `src/services/events/{__init__,publish,listener,cursor,sse}.py`;
`src/db/repos/events.py`; `src/services/curation/event_hub.py` (deleted);
`src/routers/curation/events.py`, `src/routers/curation/pipeline_events.py`,
`src/routers/curation/global_status.py` (the SSE parts); `src/app/lifespan.py` (the event tail
replaced by the listener); `src/services/curation/job_lock.py` only to remove the event hub's use
(the file is deleted by 6A when its last caller goes); `frontend/` is **not** touched (the browser
re-sends `Last-Event-ID` itself; 4E checks the pages); tests under `tests/events/`.

| ID | Task | Docs | Red test | Control |
|---|---|---|---|---|
| 3B-1 | `publish(event_type, payload, *, project=None, session=None)`: same name and argument order as today's event-hub publish so call sites do not change; with a session it writes in the caller's transaction, without one it opens its own short transaction; `pg_notify('op_events', <project id or 'global'>)` in that transaction; payload cap 7,000 bytes | [NOTIFY](https://www.postgresql.org/docs/current/sql-notify.html) | `tests/events/test_publish_transactional.py` (`db`): a rolled-back transaction emits no row and no notification; a committed one emits both | a publish without a session commits on its own |
| 3B-2 | the cursor: `(xid, id)`, the `pg_snapshot_xmin` filter ([02b](02b_jobs_locking_events.md) section 6.3) | [snapshot functions](https://www.postgresql.org/docs/current/functions-info.html#FUNCTIONS-PG-SNAPSHOT) | `tests/db/test_events_cursor.py` (`db`): two transactions commit out of id order; a reader polling between the commits emits every event once, in cursor order | a single writer is emitted in id order |
| 3B-3 | the listener: one `LISTEN op_events` connection per API process (psycopg async, autocommit, `OP_DB_DIRECT_HOST`), reconnect with backoff, 2 s safety tick, per-project fan-out to local subscribers (bounded queues; a slow client is dropped with a final `reconnect` event, never blocks the others) | [psycopg async](https://www.psycopg.org/psycopg3/docs/advanced/async.html) | `tests/events/test_listener_fanout.py` (`db`): 3 subscribers on 2 projects get only their project's events; a killed listener connection recovers and the catch-up delivers the gap | - |
| 3B-4 | SSE routes on `fastapi.sse.EventSourceResponse` with `ServerSentEvent(id="<xid>-<id>", event=type, data=payload)`; `Last-Event-ID` resume; the two hand-written framers deleted (LIB-3) | [FastAPI SSE](https://fastapi.tiangolo.com/tutorial/server-sent-events/#resuming-with-last-event-id) | `tests/events/test_sse_resume.py`: publish 100, disconnect at 40, reconnect with the 40th id -> 41..100 in order, no duplicates; frames carry `id`, `event`, `data` | the existing SSE route tests with ids added |
| 3B-5 | lifespan: the event-hub tail and its `to_thread` appends deleted; the listener started and stopped per process; `OP_EVENT_BUS` and `OP_EVENT_LOG_MAX_BYTES` retired (ENV-16) | - | `tests/test_lifespan_no_event_hub.py` (AST: no `event_hub` import, no JSONL path); retired-env test | - |
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
int/b2-control-plane at the stage-1 merge; you code alongside stage 2 and the coordinator merges
you after stage 2", the reading list `03e_b2_s3_job_runtime.md` (your lane) and
[02b](02b_jobs_locking_events.md) sections 1-4 and 6. Compose project: none. GPU: none.
