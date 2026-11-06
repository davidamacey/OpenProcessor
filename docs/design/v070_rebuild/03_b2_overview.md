# B2 overview: the control plane on PostgreSQL, Valkey and Celery

| Field | Value |
|---|---|
| Status | plan, not started |
| Release | part of the single v0.7.0 release (no intermediate tag; R-final in [07_release_acceptance.md](07_release_acceptance.md)); B2 itself merges to `main` as three PRs (section 1) |
| Depends on | B0 and B1 merged into `main` (done); the plan set, PR #477 (4 CI shards) and W2.0 (pins part A) on `main`; the owner's go for B2 |
| Design | [02_control_plane_database.md](02_control_plane_database.md), [02a](02a_postgres_schema.md) (schema), [02b](02b_jobs_locking_events.md) (jobs, events, queues, pools), [02c](02c_migration_ops_tests.md) (migrations, upgrade, services, storage, tests, traps). This file and the stage files say **who builds what, in which order, proven how** |
| Decisions applied | approved 2026-11-04: D-17 (Celery; the latest release at bump time, 5.7.0, since D-66 was reversed on 2026-11-05, D-70), D-18 (Valkey), D-31 (evaluator on `gpu_eval`), D-39..D-45 (the owner decisions). Also approved 2026-11-04 as a block: D-19, D-20, D-20a, D-20b, D-22a, D-22b, D-23, D-24, D-25, D-26, D-46..D-63 (D-48 and D-50 as amended), and after the review [review_1](reference/review_1_control_plane.md): D-64 (inference without the control plane; replaced the same day by D-69: every service requires the control plane), D-65 (Valkey wake-up), D-66 (kombu and redis-py held; reversed by the owner on 2026-11-05: the latest releases at bump time, Celery 5.7.0, kombu 5.7.0, redis-py 8.1.0), D-67 (pins part A first). Register: [decision_register.md](reference/decision_register.md) section 6 |
| Packages | named here, pinned in [01_package_pins.md](01_package_pins.md) (on `main`): SQLAlchemy (with the `asyncio` extra), Alembic, psycopg 3 (`psycopg[binary]`), Celery, kombu, billiard, redis-py, Flower, testcontainers (`testcontainers[postgres,redis]`); the Postgres and Valkey images and digests of `01_package_pins.md` section 4 (Postgres base `postgres:18.6-trixie`, MQ-2 answered) |
| Source IDs | `task_queue_plan.md` Phases 2-5 and E1-E26 (task detail where still valid); audit CC-4, CC-5, CC-9, CC-11, OPS-2, SEC-4, SEC-5; ENV-4, ENV-6, ENV-14, ENV-18, ENV-20 of [environment_and_setup](reference/environment_and_setup.md) |

Verified against `origin/main` at `7f3453de` on 2026-11-04 (read only). `main` has moved since (the plan set and the CI shards landed, 2026-11-04/05); a symbol or path that moved is re-found by concept at the cut.

---

## 1. Goal and done-when

**Goal.** PostgreSQL holds every control record; every long job runs on Celery (Valkey broker),
training, model export, label operations and their undo included; the API and every worker are stateless and can run as several
replicas; `FileJob`, every file protocol, the `/jobs` volume and every file lock are gone.

**Why (mission).** Jobs die when an API worker recycles (CC-4); periodic side effects run once per
API worker (32 times); claims hold only with one container; job files and OpenSearch control
documents cannot be queried, constrained or shared across hosts ([02](02_control_plane_database.md)
section 1).

**B2 is done when:**
- `rg -n "file_job|FileJob|job_lease|lease_heartbeat|job_lock|trigger\.json|running\.lock|cancel\.flag|claim\.g|\.job\.json|/jobs/(model_export|scores|select|viz|auto_label|imports|reprocess|assist|assist_loop|combine|region_drain|bulk_ingest|probe)\b|:/jobs\b" src scripts docker export docker-compose*.yml` is empty (outside `src/services/upgrade/from_060/` and its tests, which read 0.6.0 files). The `/jobs/<name>` alternatives are the thirteen directory names under today's shared `/jobs` volume and `:/jobs\b` its compose mounts (checked against the repo on 2026-11-05); a bare `/jobs/` would never pass, because the new `GET /jobs` routes (lane 3A-9: `src/routers/jobs.py`, `src/routers/curation/jobs.py`) and the surviving API paths `/reprocess/jobs/{job_id}` and `/promote/{job_id}/jobs/{promote_id}` contain it;
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
- the PR3 sweep (section 9) is green, including the one full B2 live run (run 2 of 3);
- the closing PR (PR3) moves the keep-list (section 11) into the permanent docs and deletes the B2 and 02
  plan files and their rows in `decision_register.md` and `docs/design/README.md` ([decision_register.md](reference/decision_register.md)
  section 8.2); merged with a merge commit; no tag.

**Delivery: three PRs to `main` (owner decision 2026-11-05, D-76, as amended by the no-rework rule D-84 the same day).** One integration branch, `int/b2-control-plane`, three PRs from its head, each leaving `main` bootable, no tag between them. After a merge the branch is an ancestor of `main`, so the next lanes merge without a rebase. PR1 = stages 1 and 2 and the job runtime of stage 3 (W2.1 + W2.2: database, control data, job runtime; job execution unchanged): part check plus a short live smoke (steps 1, 8, 9, 11). PR2 = W2.3a: stages 4 and 5 with every family written once in its final form, the research families included (4F, ND-A, ES-B, ND-B), and stage 6's 6A and 6B whole ([03f](03f_stage6_scheduler_scaleout.md) "Delivery"): part check plus the job-queue live smoke (steps 3-7 and 12). PR3 = W2.4: ES-C, Cropwright (4E, one lane), 6C's private items, docs, then the full sweep of section 9 with live run 2. Each PR: local CI parity first, the owner's yes before the push, merge commit.

## 2. Preconditions

| Check | Command (read only) | Expect |
|---|---|---|
| B1 on main | `git log --oneline origin/main \| rg -i "b1\|lease\|registry"` | the B1 merges |
| the code B2 replaces still exists | `rg -ln "class FileJob" src/services/curation/file_job.py`; `rg -n "def (acquire\|takeover_expired\|renew\|release)" src/clients/job_lease.py` | present |
| package pins | pins part A merged to `main` (W2.0), [01](01_package_pins.md) | exact versions for every package in the header |
| decisions | [decision_register.md](reference/decision_register.md) section 6 | the "needed by B2" rows answered or the recommendation accepted |
| host | [risks_cleanup_host.md](reference/risks_cleanup_host.md) section 1 | peers told before the spikes and the live run; no GPU needed except stage 5's optional smoke (none in lanes) |

If a symbol moved, re-find it by concept and record the new name in the lane's PR notes.

## 3. Stages

| Stage | Plain name | File | Lanes | Codes in parallel with | Merges after |
|---|---|---|---|---|---|
| 1 | Foundation: Postgres, Valkey, the Celery app, the schema, the storage seam, the test database | [03a_stage1_foundation.md](03a_stage1_foundation.md) | 1-0 (deps, coordinator), 1A, 1B, 1C, 1D, 1E, 1F (research-seam tables) | - | - |
| 2 | Control data moves to Postgres, and the 0.6.0 upgrade step | [03b_stage2_control_data.md](03b_stage2_control_data.md) | 2A, 2B, 2C, 2D, 2E, ES-A (embedding registry), then 2F | stage 3 | stage 1 |
| 3 | Job runtime, the reaper and the event feed | [03c_stage3_job_runtime.md](03c_stage3_job_runtime.md) | 3A, 3B | stage 2 (codes from the stage-1 head) | stage 2 |
| 4 | The API-hosted job families move to Celery, each in its final form (D-84) | [03d_stage4_job_families.md](03d_stage4_job_families.md) | W2.3a: 4A, 4B, 4C, 4D, 4F (clustering), ND-A (near-duplicates), ES-B (re-index), then ND-B (holdout guard). W2.4: 4E (frontend, after ES-C) | stage 5 | stage 3 |
| 5 | GPU queues: training, model export, the evaluator, promotion and the model registry | [03e_stage5_gpu_queues.md](03e_stage5_gpu_queues.md) | 5A, 5B, 5C, 5D, then 5E (W2.3a) | stage 4 | stage 3 |
| 6 | Beat and the periodic tasks, the deletions, scale-out compose, hardening, the docs | [03f_stage6_scheduler_scaleout.md](03f_stage6_scheduler_scaleout.md) | 6A and 6B, once each (close W2.3a, PR2); then ES-C (embedding readers) and 6C (private items), then 4E, then 6D (docs) (W2.4, PR3) | - | stages 4 and 5 |

(Stage n is the file with the n-th letter: stage 1 = 03a ... stage 6 = 03f. Older text elsewhere may
still use the earlier letters, see the "Old short names" note in [00_master_plan.md](00_master_plan.md).)

## 4. Branch, lanes and file ownership

- Integration branch `int/b2-control-plane`, cut from `main`. Each lane is one agent in its own
  worktree `<repo>-wt-b2-<lane>` on branch `b2/<lane>`, cut from the integration branch at the start
  of its stage (stage 3 lanes from the stage-1 merge).
- **Disjoint files.** Every lane owns a file list (in its stage file); no two lanes running at the
  same time edit the same file. Shared registries are split into per-lane modules created empty in
  stage 1, so later lanes only edit their own module:
  - job kinds: `src/config/job_kinds/{__init__,infra,families_a,families_b,families_c,trigger,gpu_train,gpu_export,gpu_eval,promote,b4_fanout,b4_project}.py` and the research modules `{clustering,b2_embedding,near_dup,near_dup_core,source_backfill,region_text}.py` (review RR-4: created empty in stage 1 by lane 1B, so no later lane edits an aggregator);
  - task routes: `src/config/task_routes/` with the same module names plus `periodic.py`;
  - task modules: `src/tasks/{infra,reaper,jobs_a,jobs_b,jobs_c,trigger,promote,periodic,b4_fanout,b4_project}.py`, the research
    modules `src/tasks/{clustering,embedding_reindex,near_dup,near_dup_core,source_backfill,region_text}.py`, and
    `src/tasks/beat_schedule.py`, all listed in the
    Celery `include=` from stage 1; each GPU image builds its own app with
    `src/tasks/runtime/gpu_app.py::make_app(include=[...])` from its own handler module (stage 5; the
    one signature, `make_app(include, device_uuid=None)`, is fixed in [03a](03a_stage1_foundation.md) 1B-8).
- **One Alembic head, four fixed revision ids.** The base schema, research seams included, lands in
  `0001_control_plane` (lane 1A). Each wave has at most one revision-writing lane, with a fixed id
  ([02d](02d_research_seam_schema.md) section 0): `0002_research_seams` (lane 1F, W2.1), `0003_near_duplicates`
  (lane ND-A, W2.3a), `0004_region_text` (W5e, after B2). Every other lane adds no revision; any other
  schema change is a coordinator commit on the integration branch before the lanes that need it start.
  `tests/db/test_migrations.py` asserts a single head.
- **Append-only shared files.** `env.template`, `src/config/retired_env.py`, the baselines in
  `scripts/codegen/baselines/` and `src/routers/curation/__init__.py` (the register-by-import list of the
  curation routers: a lane that adds a router adds one import line) are edited by many lanes because the env-surface test and the ratchets
  require the change in the same commit. Each lane adds or removes only its own lines; the
  coordinator resolves a textual conflict at merge by keeping both sides.
- **Compose after stage 1.** Every `docker-compose*.yml` and `docker/test/compose.yml` change of
  stages 4 and 5 is made by lane 6B (stage 6, at the close of W2.3a), from the facts the stage-4 and stage-5 lanes write in
  their PR notes; no stage-4 or stage-5 lane edits a compose file.
- **The one allowed overlap** is generated output: `contracts/`. A lane that changes a wire model
  runs `make contracts` in the same commit; if two merged lanes both regenerated it, the coordinator
  resolves the conflict by running `make contracts` on the merged tree inside the merge commit, then
  `make contracts-check`.
- **Shared docs** (`docs/ARCHITECTURE.md`, `INSTALLATION.md`, `docs/CURATION.md`, `SECURITY.md`,
  `CLAUDE.md`, root `CHANGELOG.md`, `ATTRIBUTION.md`, `docs-site/`, `docs/opensearch_schema_design.md`,
  `docs/design/curation_api_contract.md`, `README.md`) are written only by lane 6D. Other lanes write
  their facts in their PR notes for 6D. `frontend/CHANGELOG.md` belongs to 4E. The root `CHANGELOG.md`
  is the coordinator's: each PR's closing commit writes that PR's entries once (D-84; master section 7
  step 8).
- Lanes run only their targeted red/green tests and static checks on touched files
  (`.venv/bin/python -m pytest <files> -q --no-cov`, `.venv/bin/pre-commit run --files <touched>`);
  `db`-marked tests run against the lane's own testcontainers database. **No full suite, no
  `--all-files`, no live harness in a lane.** One commit per task, hooks on, never `--no-verify`,
  never push, never merge, never another checkout, never `git stash`.
- The coordinator merges lanes `--no-ff`, one merge commit per lane, in the stage's order (a failing
  lane reverts alone with `git revert -m 1 <merge>`). After each stage merge: the touched lanes'
  tests, `make contracts-check`, the `db` tier (`-m db`, minutes, no GPU; D-63) and the Docker-free
  offline suite in its 4 shards (owner decision 2026-11-04, MQ-3). One full sweep
  for the batch, in PR3 (section 9; PR1 and PR2 carry a part check and a short smoke).
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
  +---------+------------+-------------+-----------+--------------+-----------+-----------+-----------+-----------+
  v         v            v             v           v              v           v           v           v
 jobs    periodic    open_vocab    autolabel   assist_loop      cluster     gpu_train   gpu_export   gpu_eval
 worker  worker      worker        worker      worker           worker      (trainer    (export      (evaluator
 (core)  (core)      (core)        (curation)  (curation)       (curation,  image)      image)       image)
                                                                CPU)
 (the jobs worker also runs the research families: near-duplicate, source backfill, embedding
  re-index; the `arbiter` queue is consumed only by gpu-arbiter-worker, see below)
   | JobTask: claim UPDATE (fencing_token+1) -> heartbeat thread -> asyncio.run(handler)
   |          -> progress/checkpoint (fenced, 1/s) -> terminal UPDATE (fenced) -> events row + Valkey wake-up
 jobs-beat (one; session advisory lock): reaper, ingest dispatch, open-vocab sweep, cache prune,
   GPU arbiter (queue `arbiter`, consumed only by gpu-arbiter-worker, R-20), promoted reload,
   serving inventory, cluster refresh (starts `cluster_run`; there is no IVF tick), near-duplicate
   ticks, retention

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

## 9. The B2 checks: a smoke per early PR, one full sweep in PR3

Run by the coordinator on `int/b2-control-plane`, announced to the peer sessions; one heavy run at a
time; never overlapping B3's GPU window or a research prototype window. Budget rules: [decision_register.md](reference/decision_register.md) section 8.1.

**PR1 check (W2.1 + W2.2).** The part check of [00_master_plan.md](00_master_plan.md) W2.2 and a short live
smoke, `-p op-b2-smoke1`: steps 1, 8, 9 and 11 of the T2 table below, using the no-op job and the
sleeping no-op of `tests/live/` (no family has moved yet); budget 55 minutes (15 + 10 + 15 + 15).

**PR2 check (W2.3a).** The W2.3a part check and a live smoke of the job-queue steps only (owner
2026-11-05), `-p op-b2-smoke2`: steps 3-7 (80 minutes) and 12 (20 minutes) of the T2 table; budget
1 h 40 min. Steps 13-18 and the research steps run first in PR3's full run, which covers every step.

**PR3 sweep (the full B2 sweep, after every lane merged).** T1 and T2 below, with the image and browser
checks; PR1 and PR2 smokes are not counted as numbered live runs and every step runs again here.

**T1 offline (about 40 min):** `.venv/bin/pre-commit run --all-files` and the pre-push hooks; the
Docker-free suite `.venv/bin/python -m pytest tests/ -q --no-cov -m 'not live and not families and not db' -n 8`;
the database tier `.venv/bin/python -m pytest tests/ -q --no-cov -m db -n 4` (testcontainers, one
container per xdist worker); `make contracts-check`; `.venv/bin/python scripts/docs/check_docs_vs_code.py`;
the frontend full set (4E changed `frontend/`): `npm ci`, `npm run check`, lint, knip, `npm run test`,
`npm run build`, `npm run test:e2e`.

**T2 live harness, `-p op-b2-live`, budget 6 h 10 min = 370 minutes, the sum of the step budgets below (step 24, the load check, 20 minutes, owner D-83; steps 1-18: 265 minutes, 4 h 25 min; the research-seam steps 19-23, approved by the owner 2026-11-04, RD-1: 85 minutes), the one run that covers B1 and B2 (live run 2 of 3).** The harness gains
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
| 6 | cancel running; cancel queued (workers stopped, start, cancel, start workers) | `cancelled` within one progress step; the queued job never runs (the claim refuses the cancelled row; Celery `revoke` is not used) | 15 min |
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
| 17 | GPU arbiter with the overlay (carried from the original step 9): a training start with the arbiter overlay on `gpu-arbiter-worker` (RC-6) | one stop per configured container in the Docker events log, not one per API worker | 10 min |
| 18 | auto-label consumers (carried from the original step 10): two `curation-auto-label-worker` replicas, one job; kill one mid-stage; then cancel a second job at a stage boundary | one run per job; the killed one `interrupted` and resumed or restarted per its kind's auto-resume rule; the cancelled one `cancelled` | 15 min |
| 19 | the assist loop as an assistant (research auto-label 5.3, gaps G-1..G-3): start a loop on a public COCO project with a fake VLM; pause, resume three times; edit the budget down to the spent amount; start a second project's loop on the one consumer | `paused` within one request timeout plus 1 s each time; the fourth resume runs; the budget edit stops the run with `on_budget` pause; project B's first node starts within one slice; the loop record is in the `assist_loop_*` tables after the 24-hour event retention is simulated | 20 min |
| 20 | clustering on partitions (research clustering 5.1): `cluster_run` for items on a public project, activation, projection; a second run with the same vectors in another order | the served view equals the first run's; every `cluster_uid` survives; a stale activation is 409 | 15 min |
| 21 | near-duplicates and the holdout (research near-duplicates 5.4): a full `image` and `item` build on a constructed public copy set; freeze a holdout | every leak-group member of a held-out item is held out and counted; a freeze with pending rows answers 409 `near_dup_stale` | 15 min |
| 22 | the embedding re-index rehearsal in small (research embeddings 5.5): register a second public encoder as a candidate `item` space on 2,000 items, kill a worker mid-build, resume, promote, retire | resume re-encodes nothing finished; served search answers during the build; the flip is visible to every worker within one config poll | 15 min |
| 23 | the five stale Cropwright screenshots (owner-approved gap, 2026-11-04; [traceability](reference/traceability.md) section 1.8): with the docs overlay of `docs-site/docs/developer-guide/screenshots.mdx` on public data, capture only `projects-1600.png` (phone layout), `ingest-policy-1600.png`, `ingest-policy-preview-1600.png`, `dashboard-1600.png` (button label) and `clusters-1600.png` (filter bar and loop panel) with `frontend/scripts/capture_docs_screenshots.py` | each viewed full-page at desktop and 430 px before commit; the five files replace the stale ones under `docs-site/static/img/screenshots/cropwright/` in the B2 closing commit; `tests/test_docs_screenshot_seed.py` green | 20 min |
| 24 | load check (owner D-83): `tests/live/test_job_load.py` queues 1,000 short `op.infra.noop`-style jobs spread over 10 projects with 4 `jobs-worker` processes (2 replicas, concurrency 2) | all 1,000 end `completed`; no deadlock or `claim` error in the logs; every project's first job starts within 60 s (fair claiming); the peak of `pg_stat_activity` stays within the connection budget of [02b](02b_jobs_locking_events.md) section 8; median claim-to-start and heartbeat lag recorded in the PR | 20 min |

A step that overruns stops the run; the coordinator reports instead of extending the budget.

**Images:** Trivy on `postgres`, `valkey`, the api, trainer, export and evaluator images; the
image-vs-git audit; `pip check` inside each Python image.

**Browser check:** jobs UI (the `queued` state, progress over SSE, cancel of a queued and a running
job) at 1440 and 430 px, full-page screenshots viewed; and the research pages 4E changed (the loop's
pause, budget dialog and event log; the clusters pages; the near-duplicate status and the freeze
refusal).

Then PR3 from `int/b2-control-plane` against `main`, local CI parity first, the owner's yes, CI green, merge commit. The closing chores are the coordinator's ([00_master_plan.md](00_master_plan.md) section 7).

## 10. Risks

| # | Risk | Mitigation |
|---|---|---|
| R2-1 | the batch is large (six stages) | three PRs, each bootable with its own check; stages merge separately with the `db` tier each; lanes revert alone; stage 3 codes alongside stage 2 |
| R2-2 | the 0.6.0 upgrade loses or corrupts control data | bundle first, one-transaction load, verify before any delete, rollback from the bundle, the fixture tests, R-final on a real install ([02c](02c_migration_ops_tests.md) section 2) |
| R2-3 | a project row leaks to another project's request | RLS fails closed; `tests/db/test_row_level_security.py`; the OpenSearch guard unchanged |
| R2-4 | connection exhaustion with 32 API processes | the budget table and its test; `OP_PG_MAX_CONNECTIONS`; PgBouncer overlay for scale-out |
| R2-5 | a redelivery or duplicate runs a job twice | the claim statement; race tests; live steps 3-5 |
| R2-6 | the event feed stalls behind a long write transaction | rule 4; batching; trap T-13, T-14 |
| R2-7 | B3 edits `docker/trainer/` while 5A rewrites its entry point | 5A owns the job protocol and entry files; B3 owns the training code and the Ultralytics pin; the shared `docker/trainer/Dockerfile` is edited by 5A first (PR2), and B3's lanes are cut from `main` after PR2 (D-84) |
| R2-8 | Cropwright misses `queued` or the SSE id format | contracts per family; 4E; the browser check |
| R2-9 | a secret file unreadable by a container user | spikes S0 and S1 record the ownership rule; installer tests |
| R2-10 | Celery and kombu versions drift | exact pins, `pip check`, spike S1 on the pins (trap T-17) |
| R2-11 | the live run overruns its 6 h 10 min budget (the PR1 and PR2 smokes: 55 min and 1 h 40 min) | stop and report |
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

## 12. Config handed to [environment_and_setup.md](reference/environment_and_setup.md)

[environment_and_setup](reference/environment_and_setup.md) section 4 carries the table (its B2 rows); the names are defined in the stage files and
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

[github_actions.md](reference/github_actions.md) and [prior_plans_reconciliation.md](reference/prior_plans_reconciliation.md) still cite the task IDs of the former three-stage B2 and of B4's events track. They
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
