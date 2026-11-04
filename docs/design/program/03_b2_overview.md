# B2 overview: the control plane on PostgreSQL, Valkey and Celery

| Field | Value |
|---|---|
| Status | plan, not started |
| Release | part of the single v0.7.0 release (no intermediate tag; R-final in [06_release_acceptance.md](06_release_acceptance.md)) |
| Depends on | B1 merged into `main` (done); the owner's go for B2 |
| Design | [02_control_plane_database.md](02_control_plane_database.md), [02a](02a_postgres_schema.md) (schema), [02b](02b_jobs_locking_events.md) (jobs, events, queues, pools), [02c](02c_migration_ops_tests.md) (migrations, upgrade, services, storage, tests, traps). This file and the stage files say **who builds what, in which order, proven how** |
| Decisions applied | approved 2026-11-04: D-17 (Celery 5.6), D-18 (Valkey), D-31 (evaluator on `gpu_eval`), D-39..D-45 (the owner decisions). Also approved 2026-11-04 as a block: D-19, D-20, D-20a, D-20b, D-22a, D-22b, D-23, D-24, D-25, D-26, D-46..D-63 (D-48 and D-50 as amended), and after the review [00c](00c_plan_review.md): D-64 (inference without the control plane; replaced the same day by D-69: every service requires the control plane), D-65 (Valkey wake-up), D-66 (kombu and redis-py held), D-67 (pins part A first). Register: [00_index.md](00_index.md) section 6 |
| Packages | named here, pinned in `13_package_pins.md` (written on its own branch): SQLAlchemy (with the `asyncio` extra), Alembic, psycopg 3 (`psycopg[binary]`), Celery, kombu, billiard, redis-py, Flower, testcontainers (`testcontainers[postgres,redis]`); the Postgres and Valkey images and digests of `13_package_pins.md` section 4 (Postgres base: owner question MQ-2) |
| Source IDs | `task_queue_plan.md` Phases 2-5 and E1-E26 (task detail where still valid); audit CC-4, CC-5, CC-9, CC-11, OPS-2, SEC-4, SEC-5; ENV-4, ENV-6, ENV-14, ENV-18, ENV-20 of [08](08_environment_and_setup.md) |

Verified against `origin/main` at `7f3453de` on 2026-11-04 (read only).

---

## 1. Goal and done-when

**Goal.** PostgreSQL holds every control record; every long job runs on Celery (Valkey broker),
training and model export included; the API and every worker are stateless and can run as several
replicas; `FileJob`, every file protocol, the `/jobs` volume and every file lock are gone.

**Why (mission).** Jobs die when an API worker recycles (CC-4); periodic side effects run once per
API worker (32 times); claims hold only with one container; job files and OpenSearch control
documents cannot be queried, constrained or shared across hosts ([02](02_control_plane_database.md)
section 1).

**B2 is done when:**
- `rg -n "file_job|FileJob|job_lease|lease_heartbeat|job_lock|trigger\.json|running\.lock|cancel\.flag|claim\.g|\.job\.json|/jobs/" src scripts docker export` is empty (outside `src/services/upgrade/from_060/` and its tests, which read 0.6.0 files);
- `scripts/codegen/baselines/file_locks.txt` is empty and the ratchet asserts zero;
  `rg -n "fcntl" src` is empty (this was B4's done-when; B2 meets it);
- no job runs inside an API worker: `rg -n "create_task\(" src/services src/routers src/app` lists
  only the per-process helpers allowlisted in [02](02_control_plane_database.md) section 4.4;
- `src/app/lifespan.py` creates only per-process resources (pools, clients, caches, the registry and
  config polls, the Valkey event subscriber, the schema-head and mapping checks);
- OpenSearch holds only `images` and `items` per project and the `visual_search_*` indexes
  (`op_projects`, `op_global_configs` and the `configs`, `classes`, `umap_state`, `labels_confirmed`
  roles are gone from the code);
- no service in `docker-compose.yml` has a `container_name`; `docker compose up --scale
  jobs-worker=2 --scale api=2` starts (D-54);
- the B2 sweep (section 9) is green, including the one B2 live run;
- the closing PR moves the keep-list (section 11) into the permanent docs and deletes the B2 and 02
  plan files and their rows in `00_index.md` and `docs/design/README.md` ([00_index.md](00_index.md)
  section 8.2); merged with a merge commit; no tag.

## 2. Preconditions

| Check | Command (read only) | Expect |
|---|---|---|
| B1 on main | `git log --oneline origin/main \| rg -i "b1\|lease\|registry"` | the B1 merges |
| the code B2 replaces still exists | `rg -ln "class FileJob" src/services/curation/file_job.py`; `rg -n "def (acquire\|takeover_expired\|renew\|release)" src/clients/job_lease.py` | present |
| package pins | `13_package_pins.md` merged into the integration branch | exact versions for every package in the header |
| decisions | [00_index.md](00_index.md) section 6 | the "needed by B2" rows answered or the recommendation accepted |
| host | [11_risks_cleanup_host.md](11_risks_cleanup_host.md) section 1 | peers told before the spikes and the live run; no GPU needed except stage 5's optional smoke (none in lanes) |

If a symbol moved, re-find it by concept and record the new name in the lane's PR notes.

## 3. Stages

| Stage | Plain name | File | Lanes | Codes in parallel with | Merges after |
|---|---|---|---|---|---|
| 1 | Foundation: Postgres, Valkey, the Celery app, the schema, the storage seam, the test database | [03a_b2_w1_queue_infra.md](03a_b2_w1_queue_infra.md) | 1-0 (deps, coordinator), 1A, 1B, 1C, 1D, 1E | - | - |
| 2 | Control data moves to Postgres, and the 0.6.0 upgrade step | [03d_b2_s2_control_data.md](03d_b2_s2_control_data.md) | 2A, 2B, 2C, 2D, 2E, then 2F | stage 3 | stage 1 |
| 3 | Job runtime, the reaper and the event feed | [03e_b2_s3_job_runtime.md](03e_b2_s3_job_runtime.md) | 3A, 3B | stage 2 (codes from the stage-1 head) | stage 2 |
| 4 | The API-hosted job families move to Celery | [03b_b2_w2_job_families.md](03b_b2_w2_job_families.md) | 4A, 4B, 4C, 4D, then 4E (frontend) | stage 5 | stage 3 |
| 5 | GPU queues: training, model export, the evaluator, promotion and the model registry | [03f_b2_s5_gpu_queues.md](03f_b2_s5_gpu_queues.md) | 5A, 5B, 5C, 5D | stage 4 | stage 3 |
| 6 | Beat and the periodic tasks, the deletions, scale-out compose, hardening, the docs | [03c_b2_w3_beat_reaper_workers.md](03c_b2_w3_beat_reaper_workers.md) | 6A and 6B, then 6C, then 6D (docs) | - | stages 4 and 5 |

(The file names keep their earlier stage letters so links elsewhere stay valid; the table above is
the order.)

## 4. Branch, lanes and file ownership

- Integration branch `int/b2-control-plane`, cut from `main`. Each lane is one agent in its own
  worktree `<repo>-wt-b2-<lane>` on branch `b2/<lane>`, cut from the integration branch at the start
  of its stage (stage 3 lanes from the stage-1 merge).
- **Disjoint files.** Every lane owns a file list (in its stage file); no two lanes running at the
  same time edit the same file. Shared registries are split into per-lane modules created empty in
  stage 1, so later lanes only edit their own module:
  - job kinds: `src/config/job_kinds/{__init__,infra,families_a,families_b,families_c,trigger,gpu_train,gpu_export,gpu_eval,promote}.py`;
  - task routes: `src/config/task_routes/` with the same module names plus `periodic.py`;
  - task modules: `src/tasks/{infra,reaper,jobs_a,jobs_b,jobs_c,trigger,promote,periodic}.py` and
    `src/tasks/beat_schedule.py`, all listed in the
    Celery `include=` from stage 1; each GPU image builds its own app with
    `src/tasks/runtime/gpu_app.py::make_app(include=[...])` from its own handler module (stage 5).
- **One Alembic head.** The whole schema lands in `0001_control_plane` (lane 1A). A later schema
  change is a coordinator commit on the integration branch (one revision per stage, before the lanes
  that need it start); lanes never add revisions, and `tests/db/test_migrations.py` asserts a single
  head.
- **Append-only shared files.** `env.template`, `src/config/retired_env.py` and the baselines in
  `scripts/codegen/baselines/` are edited by many lanes because the env-surface test and the ratchets
  require the change in the same commit. Each lane adds or removes only its own lines; the
  coordinator resolves a textual conflict at merge by keeping both sides.
- **Compose after stage 1.** Every `docker-compose*.yml` and `docker/test/compose.yml` change of
  stages 4 and 5 is made by lane 6B (stage 6), from the facts the stage-4 and stage-5 lanes write in
  their PR notes; no stage-4 or stage-5 lane edits a compose file.
- **The one allowed overlap** is generated output: `contracts/`. A lane that changes a wire model
  runs `make contracts` in the same commit; if two merged lanes both regenerated it, the coordinator
  resolves the conflict by running `make contracts` on the merged tree inside the merge commit, then
  `make contracts-check`.
- **Shared docs** (`docs/ARCHITECTURE.md`, `INSTALLATION.md`, `docs/CURATION.md`, `SECURITY.md`,
  `CLAUDE.md`, root `CHANGELOG.md`, `ATTRIBUTION.md`, `docs-site/`, `docs/opensearch_schema_design.md`,
  `docs/design/curation_api_contract.md`, `README.md`) are written only by lane 6D. Other lanes write
  their facts in their PR notes for 6D. `frontend/CHANGELOG.md` belongs to 4E.
- Lanes run only their targeted red/green tests and static checks on touched files
  (`.venv/bin/python -m pytest <files> -q --no-cov`, `.venv/bin/pre-commit run --files <touched>`);
  `db`-marked tests run against the lane's own testcontainers database. **No full suite, no
  `--all-files`, no live harness in a lane.** One commit per task, hooks on, never `--no-verify`,
  never push, never merge, never another checkout, never `git stash`.
- The coordinator merges lanes `--no-ff`, one merge commit per lane, in the stage's order (a failing
  lane reverts alone with `git revert -m 1 <merge>`). After each stage merge: the touched lanes'
  tests, `make contracts-check`, and the `db` tier (`-m db`, minutes, no GPU; D-63). One full sweep
  for the batch (section 9).
- Opus plans; Sonnet implements and reviews. Every brief forbids `pgrep -f`, `pkill -f` and
  sleep/until loops, and names its compose project (`-p`), its GPU (none for every B2 lane), its
  worktree and its deadline. Spikes use their own `-p` and run `down -v` at the end.

## 5. Architecture after B2

```
browser --SSE--> api (N replicas x API_WORKERS)   --SQL (project_tx, RLS)--> postgres
                  | start: INSERT jobs (queued) -> COMMIT -> send_task(task_id=job_id) -> SET op:queued:<id>
                  | SUBSCRIBE op:events:<project> on Valkey (wake-up) + 2 s tick -> events rows -> SSE
                  v
               valkey (redis:// transport, AOF everysec, noeviction, password, no port)
                  |
  +---------+------------+-------------+-----------+--------------+-----------+-----------+-----------+
  v         v            v             v           v              v           v           v
 jobs    periodic    open_vocab    autolabel   assist_loop     gpu_train   gpu_export   gpu_eval
 worker  worker      worker        worker      worker          (trainer    (export      (evaluator
 (core)  (core)      (core)        (curation)  (curation)      image)      image)       image)
   | JobTask: claim UPDATE (fencing_token+1) -> heartbeat thread -> asyncio.run(handler)
   |          -> progress/checkpoint (fenced, 1/s) -> terminal UPDATE (fenced) -> events row + Valkey wake-up
 jobs-beat (one; session advisory lock): reaper, ingest dispatch, open-vocab sweep, cache prune,
   GPU arbiter (queue `arbiter`, consumed only by gpu-arbiter-worker, R-20), promoted reload,
   cluster refresh, IVF retrain check, retention

 stream workers (detection, VLM): claim projects in stream_worker_claims, item state machine in OpenSearch
 opensearch: images + items per project, visual_search_*; blobs: storage seam (local backend)
```

## 6. Inventory and destinations

The full inventory (every `FileJob` user, every other job mechanism, the B1 lease and registry code,
every loop, every file lock, every worker service, every index role) with its destination stage is
in [02_control_plane_database.md](02_control_plane_database.md) section 4. Each stage file lists
the items it moves.

## 7. Upgrade impact (0.6.0 -> 0.7.0 in one step)

The installer adds `postgres`, `valkey`, `db-migrate`, the core workers and beat, the secrets
(Postgres owner and app passwords, the Valkey password and config, Flower auth), and runs the
control-data step that moves every project, class, config, settings document and lasting job record
out of OpenSearch and the job files, verifies it, then deletes the sources
([02c](02c_migration_ops_tests.md) sections 1.2 and 2). In-flight 0.6.0 jobs end `interrupted`
(D-51). R-final proves it on a real 0.6.0 install with an in-flight job.

## 8. Rollback

Forward-only inside the program: before the release, revert the B2 merge commit on `main` (or a
lane's merge on the integration branch). After v0.7.0, the only earlier release is 0.6.0:
`openprocessor upgrade --rollback` restores the install backup and recreates the 0.6.0 control
indexes and files from the upgrade bundle ([02c](02c_migration_ops_tests.md) section 2.3); changes
made after the upgrade are not carried back.

## 9. The B2 sweep (once, after stage 6 merged)

Run by the coordinator on `int/b2-control-plane`, announced to the peer sessions; one heavy run at a
time; never overlapping B3's GPU window. Budget rules: [00_index.md](00_index.md) section 8.1.

**T1 offline (about 40 min):** `.venv/bin/pre-commit run --all-files` and the pre-push hooks; the
Docker-free suite `.venv/bin/python -m pytest tests/ -q --no-cov -m 'not live and not families and not db' -n 8`;
the database tier `.venv/bin/python -m pytest tests/ -q --no-cov -m db -n 4` (testcontainers, one
container per xdist worker); `make contracts-check`; `.venv/bin/python scripts/docs/check_docs_vs_code.py`;
the frontend full set (4E changed `frontend/`): `npm ci`, `npm run check`, lint, knip, `npm run test`,
`npm run build`, `npm run test:e2e`.

**T2 live harness, `-p op-b2-live`, budget 4 h 30 min, the one run that covers B1 and B2.** The harness gains
`postgres`, `valkey`, `db-migrate`, `jobs-worker` (scaled to 2), `periodic-worker`,
`open-vocab-worker`, `jobs-beat`, the two curation consumers and a fake GPU consumer per GPU queue
(no GPU: the fake handlers sleep and write progress). Wait with healthchecks or the Monitor tool; no
sleep loops.

| Step | Scenario | Pass | Budget |
|---|---|---|---|
| 1 | bring-up; `/health` shows `database: ok`, `queue: ok` | every healthcheck green; `db-migrate` exited 0 | 15 min |
| 2 | `tests/live/test_queue_infra.py` | a no-op job runs and its row ends `completed` | 5 min |
| 3 | race: 20 concurrent starts of one reprocess; two sends of one job id | one row, 19 x 409 `job_busy`; one run; the other delivery logs `claim_skipped` | 10 min |
| 4 | kill mid-job: reprocess of 2,000 images, `docker kill` one `jobs-worker` replica after progress > 0 | `interrupted` within 150 s, then `running` on the other replica with `attempt = 2`, completes; image count equals the total; no duplicate writes | 25 min |
| 5 | cold-shutdown requeue: `docker stop` one replica mid-job (`REMAP_SIGTERM`, soft shutdown) | `interrupted` at once, resumed by the reaper without waiting for the lease TTL | 15 min |
| 6 | cancel running; cancel queued (workers stopped, start, cancel, start workers) | `cancelled` within one progress step; the queued job never runs | 15 min |
| 7 | API recycle: `--limit-max-requests=50`, 500 requests during a reprocess | the job completes | 15 min |
| 8 | broker loss: stop Valkey, delete its volume, start it | `/ready` answers 503 naming `queue` during the outage and 200 after; queued jobs dispatched again by the reaper (rule c); running jobs keep their lease; SSE events arrive on the 2 s tick (owner decisions 2026-11-04, R-6, D-69) | 10 min |
| 9 | Postgres restart mid-job | the heartbeat reconnects (`pool_pre_ping`); the job completes or is resumed; during the outage starts answer 503, `/ready` answers 503 naming `database` and `/live` answers 200; after it `/ready` is 200 again (D-69) | 15 min |
| 10 | two beats (start a second `jobs-beat`) | the second exits with `beat_already_running`; periodic tasks run once per interval | 10 min |
| 11 | SSE: subscribe, kill and reconnect a client with `Last-Event-ID` mid-job; then resume with an id older than the retained minimum | no lost or duplicated event per project; terminal state delivered; the stale resume gets one `reset` event (R-34) | 15 min |
| 12 | GPU queues with fake handlers: train, export, bake-off start, progress, cancel | each completes or cancels; no file under the old job roots | 20 min |
| 13 | detection worker replicas: start **one** replica with three projects, wait for its claims, then start a second (review fix R-8) | both replicas hold projects within two TTLs (rebalance by release of the excess); each project is held by one replica; killing a replica moves its projects within 150 s | 20 min |
| 14 | per-family smoke (start, progress, completes) for every kind in [02b](02b_jobs_locking_events.md) section 2.1 | each completes | 25 min |
| 15 | the two B1 checks never scripted (OS-6 filtered-kNN hit count; a blocked event loop does not cost a job its heartbeat), written under `tests/live/` first | both pass; both OS-6 hit counts recorded in `docs/PERFORMANCE.md` and `docs/opensearch_schema_design.md` (carried from the original step 12) | 10 min |
| 16 | acceptance subset: `acceptance_run.sh --only ingest,cluster,holdout,label_ops,loop,export,train,bakeoff,promote` | PASS | 15 min |
| 17 | GPU arbiter with the overlay (carried from the original step 9): a training start with the arbiter overlay on `periodic-worker` | one stop per configured container in the Docker events log, not one per API worker | 10 min |
| 18 | auto-label consumers (carried from the original step 10): two `curation-auto-label-worker` replicas, one job; kill one mid-stage; then cancel a second job at a stage boundary | one run per job; the killed one `interrupted` and resumed or restarted per its kind's auto-resume rule; the cancelled one `cancelled` | 15 min |

A step that overruns stops the run; the coordinator reports instead of extending the budget.

**Images:** Trivy on `postgres`, `valkey`, the api, trainer, export and evaluator images; the
image-vs-git audit; `pip check` inside each Python image.

**Browser check:** jobs UI (the `queued` state, progress over SSE, cancel of a queued and a running
job) at 1440 and 430 px, full-page screenshots viewed.

Then one PR from `int/b2-control-plane` against `main`, local CI parity first, CI green, merge commit.

## 10. Risks

| # | Risk | Mitigation |
|---|---|---|
| R2-1 | the batch is large (six stages) | stages merge separately with the `db` tier each; lanes revert alone; stage 3 codes alongside stage 2 |
| R2-2 | the 0.6.0 upgrade loses or corrupts control data | bundle first, one-transaction load, verify before any delete, rollback from the bundle, the fixture tests, R-final on a real install ([02c](02c_migration_ops_tests.md) section 2) |
| R2-3 | a project row leaks to another project's request | RLS fails closed; `tests/db/test_row_level_security.py`; the OpenSearch guard unchanged |
| R2-4 | connection exhaustion with 32 API processes | the budget table and its test; `OP_PG_MAX_CONNECTIONS`; PgBouncer overlay for scale-out |
| R2-5 | a redelivery or duplicate runs a job twice | the claim statement; race tests; live steps 3-5 |
| R2-6 | the event feed stalls behind a long write transaction | rule 4; batching; trap T-13, T-14 |
| R2-7 | B3 edits `docker/trainer/` while 5A rewrites its entry point | 5A owns the job protocol and entry files; B3 owns the training code and the Ultralytics pin; the shared `docker/trainer/Dockerfile` is edited by 5A first, B3 rebases after B2 merges (as planned) |
| R2-8 | Cropwright misses `queued` or the SSE id format | contracts per family; 4E; the browser check |
| R2-9 | a secret file unreadable by a container user | spikes S0 and S1 record the ownership rule; installer tests |
| R2-10 | Celery and kombu versions drift | exact pins, `pip check`, spike S1 on the pins (trap T-17) |
| R2-11 | the live run overruns its 4 h 30 min budget | stop and report |
| R2-12 | projects starve each other on the `jobs` queue (carried from the original E13) | FIFO with `OP_JOBS_CONCURRENCY` (4) plus the per-project exclusive kinds; per-project routing is added only if the live run or a user report measures starvation |

## 11. What to keep after merge

| Fact | Target doc |
|---|---|
| data ownership, schema overview, job state machine, claim and fencing, reaper rules, events and the cursor, queues, pools, single beat (02, 02a, 02b) | `docs/ARCHITECTURE.md` (a new "Control plane" section; the schema DDL stays in the Alembic revision) |
| job states per family, `queued`, `interrupted` and auto-resume, the busy window (150 s), cancel and pause | `docs/CURATION.md`, `docs/design/curation_api_contract.md` |
| services, secrets, ownership rule, `max_connections`, backup and restore, the PgBouncer overlay, multi-host with a shared volume, upgrade and rollback | `INSTALLATION.md`, `SECURITY.md` (neutral preconditions only) |
| removed indexes, the two remaining roles | `docs/opensearch_schema_design.md` |
| live-run numbers (busy window, event delay, request latency under a running job) | `docs/PERFORMANCE.md` |
| Breaking entries | `CHANGELOG.md`, `frontend/CHANGELOG.md` |

## 12. Config handed to [08_environment_and_setup.md](08_environment_and_setup.md)

08 section 4 carries the table (its B2 rows); the names are defined in the stage files and
[02c](02c_migration_ops_tests.md) section 4.

## 13. Release checks handed to R-final

- `installer_upgrade` from a real 0.6.0 install (classes, configs, an in-flight job): counts and ids
  equal; the OpenSearch control indexes gone; the bundle present; jobs `interrupted`; rollback and
  re-upgrade work.
- Fresh install: `openprocessor status/health` lists the new services; `/health` reports `database`
  and `queue`; no service has a `container_name`.
- `openprocessor db backup` and `db restore` on the release stack.
- Kill a job worker mid-reprocess: resumes with `attempt = 2`.
- Trivy and the image-vs-git audit on every changed image; ATTRIBUTION rows present.
- Browser check of `queued`, progress and cancel at 1440 and 430 px.

## 14. Old task IDs used in other files

Files 09 and 12 still cite the task IDs of the former three-stage B2 and of B4's events track. They
map to the lanes of this plan as follows (the old content is replaced, not kept):

| Old ID | Now |
|---|---|
| W1-0 .. W1-14 (queue infrastructure) | stage 1 lanes 1B (Celery, queues, OPS-2, spike S1), 1C (compose, secrets, installer), 1D (health, CI) |
| 2A (families 1-6), 2B (families 7-12), 2C (families 13-18) | stage 4 lanes 4A, 4B, 4C (promote moved to 5D) |
| 2G (frontend `queued`) | stage 4 lane 4E |
| 2D (beat, reaper, OPS-2) | the reaper: stage 3 lane 3A; beat and the periodic tasks: stage 6 lane 6A |
| 2E, 2E-1, 2E-2 (auto-label and assist-loop consumers) | stage 4 lane 4D (4D-1, 4D-2) |
| 2F (hardening) | stage 6 lane 6C |
| B4 4C (events on Redis Streams) | stage 3 lane 3B (events in Postgres, D-48) |
| B4 4A-6 (evaluator) | stage 5 lane 5C |
