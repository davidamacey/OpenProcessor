# OpenProcessor 0.7.0 program: decision register, legends and reference

**Execution order lives in [00_master_plan.md](00_master_plan.md)** (the waves, their lanes, the
GPU windows, the owner questions). This file is the decision register (section 6), the plain-words
map (section 2.1), the wave exit gates (section 4), the production-ready definition (section 5),
the branch and test rules (sections 7-8) and the legends (section 11).

Status: plan, approved for execution step by step (owner decision 1 of 2026-11-03); B0 code is done.
Depends on: owner approval of the set; then each wave's preconditions ([00_master_plan.md](00_master_plan.md) section 4).
Decisions applied: owner decisions 1-9 of 2026-11-03, which override the former master plan
(rev 4, in git history; section 6 records every D-n with its status).
Source IDs: every ID in `docs/design/task_queue_plan.md` (Q-), `docs/design/codebase_audit_plan.md`
(BA-, OS-, GPU-, FE-, T-, OPS-, CC-, SEC-) and `docs/design/model_family_versions_plan.md` (U-),
plus the D-, N- and M- IDs this set owns (section 11).

Verified against `fix/411-r2-round` at `af62c947` on 2026-11-03 (and the pin commit `d1c6bab0`
that landed the same day).
Consolidated 2026-11-03 (owner decision D-37 = yes, the test and GPU budget, the deps lane).
Updated 2026-11-04 (owner decisions of that day, D-39..D-45): PostgreSQL holds all control data,
Valkey is the broker, every long job (training and model export included) runs on Celery, the
processes are stateless for scale-out; B2 is re-planned in six stages on that foundation
([02_control_plane_database.md](02_control_plane_database.md)).

This set replaces the former master plan (rev 4, in git history; the file is deleted in the
plan-set commit, [01_b0_close_out.md](01_b0_close_out.md) B0-1). Section 12 says where each of
its sections went. [12_prior_plans_reconciliation.md](12_prior_plans_reconciliation.md) reconciles
every older plan with this set.

## 0. How to use this set

A fresh agent executes **one file** (one batch, one lane, or one procedure) plus the source-plan
task text that file cites by ID. Where a source plan and this set disagree, this set wins.

| File | What it owns |
|---|---|
| [00_master_plan.md](00_master_plan.md) | **the execution order**: every wave to the tag, lanes and file ownership, folded-in decided items, sweeps and exit gates, the GPU-window budget, the wave procedure, the owner questions |
| [00b_traceability.md](00b_traceability.md) | every task, ID and issue of the original plans mapped to its wave and lane or retired with a reason; every open issue mapped; the fold-in record; the proposed GitHub actions |
| [00c_plan_review.md](00c_plan_review.md) | the independent review of the plan set (R-1..R-38) and, in its section 5, the planner's response to every finding |
| `00_index.md` (this) | mission, decision register, wave exit gates, production-ready definition, branch/lane model, test tiers, the test and GPU budget rules (section 8.1), legends, section map, superseded plans |
| [01_b0_close_out.md](01_b0_close_out.md) | B0: gate cycle, push, merge |
| B1 (merged; plan files removed, the lasting facts are in the permanent docs and `CHANGELOG.md`) | B1 correctness: leases, class registry, inference and search, API/frontend/tests, Ultralytics live bugs. **Done.** |
| [02_control_plane_database.md](02_control_plane_database.md), [02a](02a_postgres_schema.md), [02b](02b_jobs_locking_events.md), [02c](02c_migration_ops_tests.md) | the control-plane design B2 builds: data ownership, the Postgres schema, jobs and events, migrations and the 0.6.0 upgrade |
| [03_b2_overview.md](03_b2_overview.md), [03a](03a_b2_w1_queue_infra.md), [03d](03d_b2_s2_control_data.md), [03e](03e_b2_s3_job_runtime.md), [03b](03b_b2_w2_job_families.md), [03f](03f_b2_s5_gpu_queues.md), [03c](03c_b2_w3_beat_reaper_workers.md) | B2 the control plane on PostgreSQL, Valkey and Celery, six stages |
| [04_b3_ultralytics_one_version.md](04_b3_ultralytics_one_version.md) | B3 one Ultralytics version, one detection contract |
| [05_b4_stateless_api.md](05_b4_stateless_api.md) | B4 stateless API |
| [06_release_acceptance.md](06_release_acceptance.md) | release prep, R-final, the tag and publish procedure |
| [07_b5_overview.md](07_b5_overview.md), [07a](07a_b5_baseline.md), [07b](07b_b5_decode_once.md), [07c](07c_b5_crops_ocr_faces.md), [07d](07d_b5_processing_plans.md) | B5 speed (inside v0.7.0, D-37) |
| [12_prior_plans_reconciliation.md](12_prior_plans_reconciliation.md) | every older plan: what is unfinished, where it lives now, what was removed and when |
| [08_environment_and_setup.md](08_environment_and_setup.md) | dev and deployment setup; config added and removed per batch |
| [09_github_actions.md](09_github_actions.md) | new issues, re-scopes, milestones, closes, dependabot and the deps lane |
| [10_standards_and_libraries.md](10_standards_and_libraries.md) | code standards, the library-per-mechanism register with sources, the hand-roll audit |
| [11_risks_cleanup_host.md](11_risks_cleanup_host.md) | risk register, cleanup, host/GPU/peer rules, the #42 side track, "do not redo" |
| [13_package_pins.md](13_package_pins.md) | exact pins for every Python dependency and the control-plane images; the bump order (part A in W2.1, part B in W3) |

### 0.1 Standing procedure for a session resuming the program

1. Read [00_master_plan.md](00_master_plan.md) (the current wave's section), then the batch file it links.
2. Read the private ledger `artifacts_local/LEDGER.md`, latest "RESUME HERE" section: the live
   status, what is merged, what is waiting on whom.
3. Read the repository state: `git -C <repo> worktree list`, `git -C <repo> fetch origin`,
   `git -C <repo> log -1 --oneline origin/main`, the current integration branch head, and
   `gh pr list --state open`.
4. ListAgents: find the peer sessions (OpenTranscribe sessions included) and any lane agents
   still running. Message the peers before any heavy run or GPU use and wait for the reply.
5. Host load: `uptime` and `nvidia-smi`. Follow [11_risks_cleanup_host.md](11_risks_cleanup_host.md)
   section 1.
6. Check the decision register (section 6) for decisions the current wave needs; ask the owner
   for any that are open. Never start a batch without the owner's go.
7. After each step, update the ledger (private), not this public set; fix this set only when a
   plan changes, in a docs commit on the current integration branch.

## 1. Mission

OpenProcessor processes our images **fast and correctly** for the task at hand, for many users
and many processes at once: inference on Triton (detect, faces, embed, OCR, search) plus curation
(ingest, regions, VLM labeling, review, training). Correct comes first, under concurrency too.
Speed is measured once the stack it is measured on stops changing. See
[`docs/VISION_AND_GOALS.md`](../../VISION_AND_GOALS.md).

## 2. The program in one table

| Batch | Milestone | Size | Depends on | Scope | Correct | Fast | File |
|---|---|---|---|---|---|---|---|
| **B0** close-out | v0.7.0 | M | plan approval; host all-clear | CPU gate cycle, push, merge of the R2 round, the B0 small fixes and the exact Ultralytics pin (code done, `d1c6bab0`) | OCR boxes on the text and in the reported frame; real DB unclip; holdout freeze fails closed; create-only first config writes; status polls stop answering 500; outages stop reading as "not found"; Ultralytics stops floating | recycle jitter stops 32 workers restarting together | [01](01_b0_close_out.md) |
| **B1** correctness (done, merged) | v0.7.0 | XL (L, L, M, M, S-M, deps S) | B0 merged; D-8..D-12, D-15 | 1A leases, 1B class registry and upgrade migration, 1C inference/search hygiene, 1D API/frontend/tests and SEC-3/SEC-6, 1E Ultralytics live bugs, the deps lane | duplicate class ids end; one start per singleton; a sibling's live job is never flipped; a blocked loop keeps its lease; stale per-worker caches end; YOLO26 uploads build the right engine; class by name on the secondary | GPU-5/GPU-9 drop multi-GB allocations; GPU-8 stops retrying deterministic errors; OS-6 filtered kNN returns k hits | B1 (merged; see [CHANGELOG](../../../CHANGELOG.md)) |
| **B2** control plane | v0.7.0 | XXL (6 stages) | B1 merged; D-17..D-26 (D-17, D-18, D-31 approved), D-39..D-63 | PostgreSQL for every control record (projects, classes, config, jobs, runs, models) with Alembic and row-level security; Valkey and Celery for every long job, training, model export and the bake-off included; the event feed in Postgres; the 0.6.0 upgrade that moves the OpenSearch control data; beat and the reaper; scale-out compose; OPS-2; SEC-4, SEC-5 | jobs survive API recycling (CC-4); periodic side effects run once; claims hold with any number of workers and replicas; one owner per value | heavy work leaves the request processes; 6 OpenSearch indexes per project become 2 | [02](02_control_plane_database.md), [03](03_b2_overview.md) |
| **B3** one Ultralytics | v0.7.0 | L + about 1 GPU day | B1 merged; D-27, D-28 (D-38) | U-Step 4-9, trainer `nms` by capability, full U-6.1 matrix in CI, GPU-14 final, GPU-18 docs, U-C11, first #368 slice | one version, one fused `[b,300,6]` contract, no monkey-patched fork, no deprecated plugin | at least 95 % of today's detector throughput, or Stage A ships alone | [04](04_b3_ultralytics_one_version.md) |
| **B4** stateless API | v0.7.0 | M-L | B2, B3 merged; D-29, D-29a, D-32, D-32a, D-33 | Q-Phase 6-7 (bulk OCC, `API_WORKERS`, pools sized from totals), #439 (events and the last file lock moved into B2) | same answers with 1 and N workers and replicas | worker count is a tuning knob, benchmarked on the B3 engine | [05](05_b4_stateless_api.md) |
| **B5** speed | v0.7.0 (D-37) | XL, 4 PRs | B4 merged; D-35 | 5a #45 baseline, #268, GPU-6 recall; 5b GPU-3b then #40/#407 (+GPU-12); 5c #219, #220 (+GPU-7, GPU-19); 5d #67 | parity gates on every change | the GPU pipeline, crops at model size, OCR/faces on resident crops, processing plans | [07](07_b5_overview.md) |
| **R-final** | v0.7.0 | M (1-2 days) | B5d merged; D-16 | release prep (VERSION and CHANGELOG), one acceptance run (old R3 + R-B1 + R-0.9), tag on the owner's word | the production-ready definition (section 5) | no regression against the retaken baseline and the B5 numbers | [06](06_release_acceptance.md) |
| side #42 | v0.8.0 | S (GPU time) | GPU 1 free | 12 GB card acceptance; never blocks the tag | | | [07](07_b5_overview.md) section 11 |

### 2.1 The plan in plain words (read this before the codes)

The codes (B2, W1, D-18, CC-4, Q-Phase 3, ...) are cross-reference handles. The wording below
says what each piece is without them. Plan text names the thing first and puts the code after
it, for example "the lease reaper (Q-P3.2)".

| Batch | Plain name | What it changes, in one sentence | Status |
|---|---|---|---|
| B0 | Close out the 0.6 audit round | Merge the audit fixes, the docs scrub and the plan set to `main` | done, merged |
| B1 | Correctness under many workers | One owner per background job (job leases), one class list per project stored with conflict checks (class registry), plus inference, API and frontend bug fixes | done, merged |
| B2 | A real database for control data, and every long job on a queue | PostgreSQL becomes the record of projects, classes, configs, jobs, training runs and models; every long job (training and model export included) runs on Celery workers fed by Valkey; any process can run as several copies | next |
| B3 | One Ultralytics version | Every image and the trainer use one pinned Ultralytics release and one detector output format | after B2's coding starts (runs alongside) |
| B4 | Stateless API | The API's worker count becomes a plain tuning setting and request fan-out goes; the result is checked with 1 and many workers and replicas | after B2 and B3 |
| B5 | Speed | Measure first, then decode each image once on the GPU, crop at model size, run OCR and faces on those crops, and add per-project processing plans | after B4 |
| R-final | Release acceptance | One full acceptance run on the finished stack, then the v0.7.0 tag on the owner's word | last |

B2 has six stages; [02_control_plane_database.md](02_control_plane_database.md) is the design they build:

| Stage | Plain name | File |
|---|---|---|
| 1 | Foundation: the Postgres and Valkey services, the schema, the Celery app, the storage seam, the test database | [03a](03a_b2_w1_queue_infra.md) |
| 2 | Projects, classes, configs and other records move into Postgres; the one-step 0.6.0 upgrade | [03d](03d_b2_s2_control_data.md) |
| 3 | The job machinery (start, claim, heartbeat, cancel, the reaper) and the event feed to the browser | [03e](03e_b2_s3_job_runtime.md) |
| 4 | Each job type that ran inside the API moves onto the queue | [03b](03b_b2_w2_job_families.md) |
| 5 | Training, model export and the bake-off evaluator run as GPU queue workers; promotion and the model registry in Postgres | [03f](03f_b2_s5_gpu_queues.md) |
| 6 | Timed jobs run once from one scheduler; the old file machinery is deleted; every service can scale; hardening; docs | [03c](03c_b2_w3_beat_reaper_workers.md) |

The audit codes used most often in the table above:

| Code | Plain meaning |
|---|---|
| CC-4 | A running job dies when its API worker restarts (B2 fixes it) |
| RLS | Row-level security: Postgres itself refuses a row of another project |
| CC-2 | Concurrent class-list edits lose classes or duplicate ids (B1 fixed it with the class registry document; B2 moves the registry into Postgres tables) |
| OPS-2 | Metrics files left behind by dead API workers are never cleaned (measure first, then fix in B2) |
| SEC-n | A security hardening item (details stay off the public tree) |
| GPU-n | A GPU memory or throughput fix |
| OS-n | An OpenSearch query or index fix |
| U-Step n | A step of the Ultralytics upgrade |
| Q-Phase n | A phase of the job-queue move |
| W0..W6 | The program's waves: one integration branch and one full test sweep each; B2's wave W2 has five parts W2.1, W2.2, W2.3a, W2.3b and W2.4, each closed by a short part check ([00_master_plan.md](00_master_plan.md)) |

Approved by the owner on 2026-11-04: Celery 5.6 (D-17); Valkey as the broker (D-18); PostgreSQL
for all control data (D-39); OpenSearch for search data (D-40); every long job on Celery, training
and export included (D-41); stateless scale-out (D-42); the one-step upgrade that moves and then
deletes the OpenSearch control data (D-43); exact pins (D-44); plain names first (D-45); the
evaluator on a GPU queue (D-31).

The B2 decisions below were **approved by the owner on 2026-11-04 as a block** (section 6 records
each; D-48 and D-50 as amended after the review in [00c_plan_review.md](00c_plan_review.md), with
D-64..D-67 added, and D-64 replaced the same day by D-69). The questions still open are in [00_master_plan.md](00_master_plan.md) section 9:

| Decision | Question in plain words |
|---|---|
| D-19 | Do the new queue containers start by default (recommended), or only under a profile? |
| D-20 | A job is acknowledged only after it finishes and tried at most 3 times: agreed? |
| D-20a | May long job types (ingest, import, combine, reprocess) run up to 24 h instead of 3 h? |
| D-20b | Should `docker stop` interrupt and requeue a running job after 30 s? |
| D-22a | Does the old directory-ingest route become a background job (202) rather than being deleted? |
| D-22b | May the synchronous `wait=true` mode of promote be deleted (only tests use it)? |
| D-23 | How many jobs one general worker runs at once (4, tunable)? |
| D-24 | Is a broker that keeps about one second of data on a crash, with no backup, enough? |
| D-25 | Which job types restart by themselves after a crash? |
| D-26 | Keep the API worker recycling with jitter that landed in B0? |
| D-46 | Multi-host files: a shared volume now and an S3-compatible store in v0.8.0? |
| D-47 | Should items keep the class name next to the class id, kept correct by one job? |
| D-48 | Do job progress and events reach the browser through Postgres (a table plus a wake-up signal)? |
| D-49 | Should Celery keep no results of its own, the jobs table being the only record? |
| D-50 | Leave the connection pooler (PgBouncer) out of the default install, as an opt-in for many hosts? |
| D-51 | Jobs running during the 0.6.0 upgrade end as "interrupted" and are restarted by hand? |
| D-52 | Move only the lasting 0.6.0 job history (training, promotions, imports, exports, label operations, combine) into Postgres? |
| D-53 | Keep the detection and VLM workers as stream workers, with each copy claiming whole projects? |
| D-54 | Remove the fixed container name from every service? |
| D-55 | Should Postgres itself refuse another project's rows (row-level security)? |
| D-56 | Delete the four per-project OpenSearch indexes that held control data (6 indexes per project become 2)? |
| D-57 | Drop the 10,000-class limit? |
| D-58 | Accept an optional `Idempotency-Key` header on job-start routes? |
| D-59 | Store prompt packs, profiles and settings as validated JSON documents with relational keys? |
| D-60 | Enforce a single scheduler with a database lock? |
| D-61 | No scheduled database backup by default (a dump on every upgrade and on demand)? |
| D-62 | Real Postgres in tests through testcontainers locally and a CI service container, kept out of the Docker-free command? |
| D-63 | A quick database test run at each B2 stage merge, in addition to the one full sweep? |

## 3. Order and dependencies

Moved to [00_master_plan.md](00_master_plan.md) sections 2-3 (the waves, the dependency diagram and
the hard edges). The order decision itself is D-13 as amended by D-37 (section 6).

## 4. Wave exit gates

The waves, their lanes and what codes in parallel during each sweep are in
[00_master_plan.md](00_master_plan.md) section 4. The task list that used to be section 4.1 is mapped
task by task in [00b_traceability.md](00b_traceability.md) section 1.

Common exit criteria, required by every wave ("PQ" gates):

| ID | Gate | Check |
|---|---|---|
| PQ-1 | CI green locally, then on the PR | every job of `.github/workflows/ci.yml` (and `frontend.yml` when `frontend/` changed) reproduced locally, the changelog rules included; then `gh pr checks <n>`; the push verified with `git ls-remote` |
| PQ-2 | Ratchets only shrink | `git diff main...HEAD -- scripts/codegen/baselines/` shows only removed lines or lower counts; an oversize file is split by concern, never re-baselined |
| PQ-3 | No dead code, no shims | the vulture and dead-code hooks green; a superseded route, field, module or env var deleted in the same change (retired env vars added to `src/config/retired_env.py`); no compatibility read path |
| PQ-4 | Contracts regenerated | `make contracts` in each commit that changes a wire model; `make contracts-check` clean in the sweep |
| PQ-5 | Docs check clean | `.venv/bin/python scripts/docs/check_docs_vs_code.py`; the docs listed for the batch (section 9) updated; CHANGELOG entries under `[Unreleased]` (root and frontend) |
| PQ-6 | Red/green evidence | each task's red test seen failing on the base plus its paired control, with the red output in the PR description |
| PQ-7 | Hot-path numbers | where the wave touches the hot path, the performance and parity numbers against the stated reference, recorded in `docs/PERFORMANCE.md` (and `docs/research/` for studies) |
| PQ-8 | Image hygiene | when an image is built for a gate or a push: the image-vs-git audit and the Trivy gate |
| PQ-9 | Owner rules | no new in-process or file lock for cross-process state (file-lock ratchet from B1), no asyncio fan-out in request handlers, class identity by name, fail closed ([10_standards_and_libraries.md](10_standards_and_libraries.md)) |

## 5. Definition of production-ready for v0.7.0

R-final verifies every line; the RF row ids are in [06_release_acceptance.md](06_release_acceptance.md)
section 4.

| # | Production-ready means | Verified by |
|---|---|---|
| PR-01 | Cross-process control state is in PostgreSQL (transactions, constraints, fenced claims, advisory locks, idempotency keys), item state uses OpenSearch OCC, never a file or in-process lock: `rg -n "fcntl" src` is empty and the file-lock ratchet is at zero | release-prep precondition; B2 exit |
| PR-02 | A singleton job starts once under concurrent requests; a dead holder blocks a restart for at most 150 s (lease TTL plus one reaper tick) | RF-42 |
| PR-03 | Concurrent class writes never lose a class or duplicate an id | RF-43 |
| PR-04 | Jobs run in workers: they survive an API restart, a killed worker's resumable job is re-enqueued, periodic tasks run once | RF-52, RF-53, RF-55 |
| PR-05 | The API gives the same answers with 1 and 32 workers and with several replicas of the API and the job workers | the B4 sweep record |
| PR-06 | Class identity is the name across import, combine, export, train, promote and the secondary detector | B1 tests; RF-40, RF-43 |
| PR-07 | Fail closed: every process waits for PostgreSQL, Valkey and OpenSearch (and Triton where it calls it) and refuses to start naming the missing one (D-69); Postgres down at run time -> 503 on every curation write and job start, `/ready` 503 naming `database`; OpenSearch down -> 503 on data-plane writes and `review_dismiss`; a schema not at the code's head -> the process refuses to start; another project's rows -> refused by the database; a raw-head engine -> refused; Valkey down -> `/ready` 503 naming `queue` and 503 starts | RF-36, RF-41, RF-57, RF-61 |
| PR-08 | One exact Ultralytics version in every image; one fused detection contract; the B3 parity and throughput gate passed | RF-20, RF-57, the B3 sweep record |
| PR-09 | Fresh install, re-run, repair, `--skip-models`, engine adoption, uninstall work | RF-01, RF-02, RF-12, RF-23, RF-24, RF-25 |
| PR-10 | The supported upgrade, 0.6.0 -> 0.7.0, moves every project, class (ids kept), config and lasting job record into Postgres and deletes the OpenSearch copies after verifying them, adds the new services, rebuilds the default engine once; rollback to 0.6.0 recreates the 0.6.0 state from the upgrade bundle | RF-40, RF-41, RF-56, RF-58, RF-59, RF-62, RF-63 |
| PR-11 | Security: SEC-1..SEC-6 fixed (neutral wording); the broker and the database have no published port and required passwords from files; Trivy fixable CRITICAL = 0; every image passes the image-vs-git audit; no secret in any log or report | RF-08, RF-09, RF-10, RF-34, RF-50 |
| PR-12 | Quality: CI green; ratchet baselines smaller than or equal to their `5b95591f` values; no dead code or shims; contracts in sync; docs check clean | PQ-1..PQ-5 at every wave; the release commit's full gate |
| PR-13 | Speed: no regression against the 32-worker baseline retaken on the B3 engine at the shipped `API_WORKERS` default; numbers in `docs/PERFORMANCE.md` | the GPU window 2 record (5a baseline) |
| PR-14 | UI: every changed page checked at 1440 and 430 px with the screenshots viewed; the frontend live suite passes, region tests included | RF-04, RF-07, RF-33, RF-45, RF-54, RF-60 |
| PR-15 | Docs and release: INSTALLATION (services, secrets, backup and restore, upgrade, downgrade), ARCHITECTURE (the control plane, services, queues, events, model families), CURATION (job states, the `job_busy` window), SECURITY (broker and database preconditions) current; CHANGELOG consolidated with the breaking changes first; release notes; `/health` reports 0.7.0 | RP-1..RP-6, RF-26 |

## 6. Decision register

Statuses: **approved** (owner decision of 2026-11-03, or of 2026-11-04 where the row says so), **done**, **open** (with the plan's
recommendation), **dropped**, **replaced**. "Q-" = task-queue plan, "U-" = model-family plan.

| ID | Decision | Status | Recommendation or note | Needed by | Detail |
|---|---|---|---|---|---|
| D-1 | B0 scope | approved, minus release prep | the B0 small fixes, the uvicorn floor and the pin on `fix/411-r2-round`, one gate cycle | B0 | [01](01_b0_close_out.md) |
| D-2 | Ultralytics exact pin in B0 (U-J2) | approved, done (`d1c6bab0`) | `==8.4.197` in `requirements.txt`, `requirements-test.txt`, `pyproject.toml`; next move only in B3 | B0 | [01](01_b0_close_out.md) |
| D-3 | Gates, push, CI, merge commit | approved; R3 deferred into R-final | | B0 | [01](01_b0_close_out.md), [06](06_release_acceptance.md) |
| D-4 | Segmenter tier in acceptance | approved | HF token from the stack's `.env`, never printed; GPU 0 (GPU 2 if free and agreed) | R-final | [06](06_release_acceptance.md) RF-22 |
| D-5 | #42 timing | approved | runs when GPU 1 frees; never a tag blocker; release notes say so if unrun | side | [07](07_b5_overview.md) section 11 |
| D-6 | Known-issue lines for #447, CC-2, CC-4 | dropped | no release ships before B1-B4 fix them | - | |
| D-7 | Release train (early v0.8.0 after B1) | replaced | by owner decisions 2-3 and D-37: one release v0.7.0 after B5 and R-final | - | [06](06_release_acceptance.md) |
| D-8 | J-LEASE | approved | OpenSearch lease docs in the SETTINGS role, thread heartbeat 15 s, TTL 120 s, cooperative fencing | B1 | B1 1A (merged; see [ARCHITECTURE](../../ARCHITECTURE.md)) |
| D-9 | J-REG and its upgrade path | approved | registry doc with OCC (5 attempts, 10,000 classes), 409 `registry_not_migrated`, `upgrade` hook, `.migrated` rename, downgrade note. Only 0.6.0 -> 0.7.0 is tested (no 0.7.0 install exists) | B1 | B1 1B (merged; see [ARCHITECTURE](../../ARCHITECTURE.md)) |
| D-10 | B1-1E scope | approved | minimal in B1; trainer `nms`, export module and full matrix in B3 | B1 | B1 1E (merged; see [CHANGELOG](../../../CHANGELOG.md)) |
| D-11 | U-J3 YOLOv5 scope | approved, option b | bring-your-own original-YOLOv5 ONNX for probe and secondary; YOLOv5u for serving | B1 | B1 1E (merged; see [CHANGELOG](../../../CHANGELOG.md)) |
| D-12 | U-J4 `nms` by capability | approved | `nms=False` with `one2one_cv2`, else `nms=True`, class-agnostic | B1 (export), B3 (trainer) | B1 1E (merged; see [CHANGELOG](../../../CHANGELOG.md)), [04](04_b3_ultralytics_one_version.md) |
| D-13 | Order (Q-J8) | approved; amended by D-37 | B0, B1, B2, B3, B4, B5, then R-final last | - | section 3 |
| D-14 | Commit the plans | done | committed on `fix/411-r2-round` | - | |
| D-15 | Trainer claim (Q-P1.6) | approved | create-only generation files; API `training/jobs.py` `O_EXCL` create | B1 | B1 1A (merged; see [ARCHITECTURE](../../ARCHITECTURE.md)) |
| D-16 | Tag | replaced | one tag decision: v0.7.0 after R-final PASS (R-final runs after B5), on the owner's word (absorbs the old D-34) | R-final | [06](06_release_acceptance.md) section 7 |
| D-17 | Q-J1 Celery 5.6 | approved 2026-11-04 | Celery 5.6 with kombu's `redis://` transport and redis-py; exact pins in `13_package_pins.md` | B2 | [02b](02b_jobs_locking_events.md) section 7 |
| D-18 | Q-J2 broker | approved 2026-11-04: **Valkey** | Valkey 9 (BSD-3-Clause) by digest; kombu has no `valkey://` scheme (issue #2245), so the documented `redis://` transport and redis-py are used; no third-party transport | B2 | [02b](02b_jobs_locking_events.md) section 7.2 |
| D-19 | Q-J3 default install | approved 2026-11-04 (owner, the open B2 recommendations accepted as a block) | core: `postgres`, `valkey`, `db-migrate` (one-shot), `jobs-worker`, `periodic-worker` (so the reaper and the dispatch tick never wait behind long jobs), `open-vocab-worker`, `jobs-beat`; Flower under `monitoring` | B2 | [02c](02c_migration_ops_tests.md) section 4 |
| D-20 | Q-J4 delivery | approved 2026-11-04 (owner, the open B2 recommendations accepted as a block) | `acks_late` + `reject_on_worker_lost` + the claim `UPDATE` with a fencing token + attempt cap 3 + soft shutdown 30 s | B2 | [02b](02b_jobs_locking_events.md) section 3 |
| D-20a | Time limits for long job kinds | approved 2026-11-04 (owner, the open B2 recommendations accepted as a block) | default 3 h / 3 h 15 m; 24 h for the long kinds, 72 h for training, as the kind table says | B2 | [02b](02b_jobs_locking_events.md) section 2.1 |
| D-20b | `REMAP_SIGTERM=SIGQUIT` on worker services | approved 2026-11-04 (owner, the open B2 recommendations accepted as a block) | yes, so `docker stop` gets the soft shutdown; `stop_grace_period` above the soft-shutdown timeout | B2 | [02b](02b_jobs_locking_events.md) section 7.1 |
| D-21 | Q-J5 job-state store | replaced by D-39 (2026-11-04) | the `jobs` table in PostgreSQL is the store; `FileJob` and the leases are deleted | - | [02b](02b_jobs_locking_events.md) |
| D-22 | Q-J6 scope | approved 2026-11-04 (owner decision 4, D-41): scope now includes training, model export and the evaluator | every API-hosted family, the periodic loops, auto-label, assist loop, the trainer (`gpu_train`), model export (`gpu_export`), the evaluator (`gpu_eval`); the detection and VLM stream workers stay (D-53) | B2 | [02](02_control_plane_database.md) section 4 |
| D-22a | Directory ingest is not project-bound | approved 2026-11-04 (owner, the open B2 recommendations accepted as a block) | a deployment-level job row (`project_id` NULL), 202 + job id | B2 | [03b](03b_b2_w2_job_families.md) 4C-6 |
| D-22b | Promote `wait=true` | approved 2026-11-04 (owner, the open B2 recommendations accepted as a block) | delete it (only tests use it); no shim | B2 | [03f](03f_b2_s5_gpu_queues.md) 5D-3 |
| D-23 | Q-J9 `jobs` concurrency | approved 2026-11-04 (owner, the open B2 recommendations accepted as a block) | 4, tunable | B2 | [03a](03a_b2_w1_queue_infra.md) |
| D-24 | Q-J10 broker durability | approved 2026-11-04 (owner, the open B2 recommendations accepted as a block) | Valkey AOF everysec, no backup (the reaper rebuilds the queue from the `jobs` rows) | B2 | [02b](02b_jobs_locking_events.md) section 7.2 |
| D-25 | Q-J11 auto-resume | approved 2026-11-04 (owner, the open B2 recommendations accepted as a block) | reprocess, dataset import, bulk ingest, combine, assist loop, project delete, class projection sync | B2 | [02b](02b_jobs_locking_events.md) section 2.1 |
| D-26 | Q-J12 recycling | approved 2026-11-04 (owner, the open B2 recommendations accepted as a block) | keep `--limit-max-requests` with jitter (landed in B0) | B2 | [03](03_b2_overview.md) |
| D-27 | U-J1 staging + U-J5..J8 | open | one version; U-6.2 once for Stage A and B; J5 keep `conf=0.25`; J6 keep the Triton name; J7 export batch = `max_batch_size`; J8 trainer on the same release | B3 | [04](04_b3_ultralytics_one_version.md) |
| D-28 | B3 pin choice rule | open | newest 8.4.x passing U-6.1 at a freeze date set at B3 kick-off (at least 3 days old) | B3 | [04](04_b3_ultralytics_one_version.md) |
| D-29 | Q-J7 request fan-out | open | bulk OCC; the ingest data plane a documented `TaskGroup` exemption | B4 | [05](05_b4_stateless_api.md) |
| D-29a | VLM batch routes (`label_batch`, `verify_region_batch`) (new) | open | 202 + a Celery job; the bounded `TaskGroup` runs in the worker | B4 | [05](05_b4_stateless_api.md) |
| D-30 | Q-J13 event transport | replaced by D-48 (2026-11-04) | events in Postgres, built in B2 stage 3 | - | [02b](02b_jobs_locking_events.md) section 6 |
| D-31 | Q-J6 evaluator | approved 2026-11-04 (owner decision 4) | `-Q gpu_eval -c 1`, built in B2 stage 5 | B2 | [03f](03f_b2_s5_gpu_queues.md) |
| D-32 | #439 approach | open | measure first; then a `project_create` job (202) | B4 | [05](05_b4_stateless_api.md) |
| D-32a | Project status during creation (new) | open | reuse the existing `building` status (combine targets use it), not a new `creating` | B4 | [05](05_b4_stateless_api.md) |
| D-33 | Q-J14 API workers | open | `API_WORKERS` (`WEB_CONCURRENCY`), default `min(nproc, 32)`, confirmed by the benchmark; replicas need nothing more after B2 | B4 | [05](05_b4_stateless_api.md) |
| D-34 | Tag v0.9.0 | dropped | merged into D-16 | - | |
| D-35 | GPU-6 `/detect` upscaling | open | keep `scaleup=False` until 5a measures recall | B5a | [07a](07a_b5_baseline.md) |
| D-36 | GitHub actions | approved | executed only on the owner's go after this set is approved | before W1 | [09](09_github_actions.md) |
| D-37 | B5 before the first tag? (new) | approved: YES | owner, 2026-11-03: "not tagging until all of these plans are implemented and tested". v0.7.0 = B0..B5 + R-final; R-final runs last; milestones re-cut (section 3, [09](09_github_actions.md) section 7) | - | section 3 |
| D-38 | Dependabot #432 handling: Triton 26.09 and a newer Python base (new; the same question as Q-4 and ENV-10 in [08](08_environment_and_setup.md)) | open | close #432 and #422; do the move only inside B3 as optional task B3-A5, with the TensorRT pins and the `Dockerfile.triton` assertion moved together, behind the U-6.2 gate, if B3's GPU window allows; otherwise after the tag | B3 kick-off | [04](04_b3_ultralytics_one_version.md) B3-A5, [09](09_github_actions.md) section 8 |
| D-39 | PostgreSQL holds all relational and control data (projects, settings, classes, revisioned config, jobs and their locking, training runs, models, promotions; later users and tokens) | approved 2026-11-04 (owner decision 1; replaces D-21) | the schema of [02a](02a_postgres_schema.md) | B2 | [02](02_control_plane_database.md) |
| D-40 | OpenSearch keeps the search, text and vector data; each value has one owner | approved 2026-11-04 (owner decision 2) | the data-ownership table | B2 | [02](02_control_plane_database.md) section 3 |
| D-41 | Every long job runs on Celery, training and model export included; `FileJob`, the file protocols, `/jobs` and every file lock are deleted | approved 2026-11-04 (owner decision 4) | GPU queues `gpu_train`, `gpu_export`, `gpu_eval` | B2 | [03f](03f_b2_s5_gpu_queues.md) |
| D-42 | Cloud-ready scale-out: stateless API and workers, N replicas, no fixed `container_name` on scalable services, shared state only in Postgres, Valkey or OpenSearch | approved 2026-11-04 (owner decision 5) | the storage seam (D-46) for blobs | B2 | [02](02_control_plane_database.md) section 2 |
| D-43 | 0.6.0 -> 0.7.0 in one step: the OpenSearch control data moves into Postgres by a tested upgrade step, then the OpenSearch copies are deleted (no dual read) | approved 2026-11-04 (owner decision 6) | bundle, load, verify, delete; rollback from the bundle | B2 | [02c](02c_migration_ops_tests.md) section 2 |
| D-44 | Packages exact-pinned at the latest installable versions | approved 2026-11-04 (owner decision 7) | `13_package_pins.md` | B2 | - |
| D-45 | Plain names first, codes after, in every plan text | approved 2026-11-04 (owner decision 8) | each decision has a one-line plain question (section 2.1) | - | - |
| D-46 | Multi-host files: shared volume or object store? | approved 2026-11-04 (owner, the open B2 recommendations accepted as a block) | the storage seam with a local backend in 0.7.0 (a shared POSIX volume for several hosts); an S3-compatible backend in v0.8.0, planned as feature F10 of [v080_platform_breadth_plan.md](../v080_platform_breadth_plan.md) (owner-approved addition 2026-11-04; issue text in [00b](00b_traceability.md) section 4.6) | B2 stage 1 | [02c](02c_migration_ops_tests.md) section 5 |
| D-47 | Do items keep the class name? | approved 2026-11-04 (owner, the open B2 recommendations accepted as a block) | yes, as a projection owned by Postgres: one writer path, an outbox row in the same transaction, a sync job, a verifier | B2 stage 2 | [02](02_control_plane_database.md) section 3.3 |
| D-48 | How do progress and events reach the browser? | approved 2026-11-04 (owner), as amended after the review (00c R-6) | a Postgres `events` table is the durable record and the `(xid, id)` cursor is the SSE id; the wake-up goes through Valkey pub/sub (`op:events:<project>`), not a per-process Postgres `LISTEN` (replaces D-30) | B2 stage 3 | [02b](02b_jobs_locking_events.md) section 6 |
| D-49 | Does Celery keep results? | approved 2026-11-04 (owner, the open B2 recommendations accepted as a block) | no result backend; the `jobs` row is the only truth | B2 stage 1 | [02b](02b_jobs_locking_events.md) section 7.1 |
| D-50 | A connection pooler in the default install? | approved 2026-11-04 (owner), as amended after the review (00c R-6) | no pooler by default; the opt-in PgBouncer overlay becomes required when the connection formula of 02b section 8, with the planned replicas, exceeds `max_connections` | B2 stage 6 | [02b](02b_jobs_locking_events.md) section 8.1 |
| D-51 | Jobs in flight during the 0.6.0 upgrade? | approved 2026-11-04 (owner, the open B2 recommendations accepted as a block) | recorded `interrupted`, no automatic resume | B2 stage 2 | [02c](02c_migration_ops_tests.md) section 2 |
| D-52 | Which 0.6.0 job histories move? | approved 2026-11-04 (owner, the open B2 recommendations accepted as a block) | training runs, promotions, models, imports, exports, label operations, combine provenance; the transient ones stay in the bundle | B2 stage 2 | [02c](02c_migration_ops_tests.md) section 2.1 |
| D-53 | Detection and VLM workers on Celery? | approved 2026-11-04 (owner, the open B2 recommendations accepted as a block) | no: stream workers that claim whole projects through Postgres | B2 stage 3 | [02b](02b_jobs_locking_events.md) section 5 |
| D-54 | Which services lose `container_name`? | approved 2026-11-04 (owner, the open B2 recommendations accepted as a block) | all; lookups by Compose labels | B2 stage 6 | [03c](03c_b2_w3_beat_reaper_workers.md) 6B-2 |
| D-55 | Row-level security in Postgres? | approved 2026-11-04 (owner, the open B2 recommendations accepted as a block) | yes, `FORCE`d on every project-scoped table, with `project_tx` | B2 stage 1 | [02a](02a_postgres_schema.md) section 1.4 |
| D-56 | Delete the per-project `configs`, `classes`, `umap_state`, `labels_confirmed` indexes? | approved 2026-11-04 (owner, the open B2 recommendations accepted as a block) | yes (6 indexes per project become 2) | B2 stage 2 | [02](02_control_plane_database.md) section 4.7 |
| D-57 | Keep the 10,000-class cap? | approved 2026-11-04 (owner, the open B2 recommendations accepted as a block) | no; `registry_full` goes | B2 stage 2 | [02a](02a_postgres_schema.md) section 3 |
| D-58 | `Idempotency-Key` on job starts? | approved 2026-11-04 (owner, the open B2 recommendations accepted as a block) | yes, optional | B2 stage 3 | [02b](02b_jobs_locking_events.md) section 2.3 |
| D-59 | Config bodies as validated JSON documents? | approved 2026-11-04 (owner, the open B2 recommendations accepted as a block) | yes, `jsonb` validated by pydantic, identity and relations as columns | B2 stage 1 | [02a](02a_postgres_schema.md) section 4 |
| D-60 | One beat enforced how? | approved 2026-11-04 (owner, the open B2 recommendations accepted as a block) | a Postgres session advisory lock at beat start | B2 stage 6 | [02b](02b_jobs_locking_events.md) section 4.1 |
| D-61 | Scheduled database backups by default? | approved 2026-11-04 (owner, the open B2 recommendations accepted as a block) | no: a dump on every upgrade and on demand; an opt-in beat task | B2 stage 1 | [02c](02c_migration_ops_tests.md) section 6 |
| D-62 | How do tests get a real Postgres? | approved 2026-11-04 (owner, the open B2 recommendations accepted as a block) | testcontainers locally, a CI service container; the `db` marker outside the Docker-free command | B2 stage 1 | [02c](02c_migration_ops_tests.md) section 7 |
| D-63 | A database test run at each B2 stage merge? | approved 2026-11-04 (owner, the open B2 recommendations accepted as a block) | yes (`-m db`, minutes, no GPU) | B2 | [03](03_b2_overview.md) section 4 |
| D-64 | Does inference keep serving when the control plane is down? | **replaced 2026-11-04 by D-69** (the owner: the system only works when everything is up, so running the app without its databases makes no sense) | was: yes: `/detect`, `/faces`, `/embed`, `/ocr`, `/search` and `/health` keep serving when PostgreSQL or Valkey is down; only the curation and job routes need them and answer 503; `/health` and `/ready` say which dependency is down; workers and beat keep the hard dependency | B2 stage 1 | [02c](02c_migration_ops_tests.md) section 1.2 |
| D-69 | Must every service have PostgreSQL, Valkey and OpenSearch? | approved 2026-11-04 (owner correction, replaces D-64) | yes: the API and every worker require PostgreSQL, Valkey and OpenSearch, plus Triton for inference; no degraded mode and no flag for one; at start a process waits with bounded retries, then refuses with an error naming the missing dependency; `/live` says the process is up, `/ready` answers 503 and names each dependency that is down; compose `depends_on` with `service_healthy` on all of them and `service_completed_successfully` on `db-migrate`; the OpenSearch migrate step runs in `db-migrate` after Alembic (RC-1), and on an upgrade from 0.6.0 again after the projects are loaded | B2 stages 1 and 2 | [02c](02c_migration_ops_tests.md) section 1.2 |
| D-65 | How are browser events woken up? | approved 2026-11-04 (owner decision 2, after 00c R-6; amends D-48) | Valkey pub/sub (`op:events:<project>`), not a per-process Postgres `LISTEN`; the Postgres `events` table stays the durable record and the resume cursor | B2 stage 3 | [02b](02b_jobs_locking_events.md) section 6 |
| D-66 | Which kombu and redis-py? | approved 2026-11-04 (owner decision 3, after 00c R-10) | hold `kombu==5.6.2` and `redis==6.4.0` until a Celery release declares support for kombu 5.7 | B2 stage 1 | [13](13_package_pins.md) sections 1 and 4 |
| D-67 | Where does pins part A land? | approved 2026-11-04 (owner decision 4, after 00c R-16) | its own PR to `main` before W2.1 (wave W2.0); part B stays with the B3 GPU gate | W2.0 | [13](13_package_pins.md) section 7.1 |
| D-68 | The GitHub actions of the master-plan pass | approved 2026-11-04 (owner decision 11) | run by the coordinator after the review fix pass, with the exact text of [00b](00b_traceability.md) section 4 | after this pass | [00b](00b_traceability.md) section 4 |
| OI-1 | #469: which region kinds may reach external model endpoints? | approved 2026-11-04 (owner: keep today's behaviour) | when external use is allowed, external endpoints may receive any region kind; no code change; #469 closes as "decided: keep current behaviour"; the privacy lane adds a plain SECURITY.md statement | W4 (docs only) | [00b](00b_traceability.md) section 4 |
| OI-2 | #471: on what basis does the research judge refuse a hosted call? | approved 2026-11-04 (owner: no new restriction) | contact sheets and photos go to the hosted judge as today; the existing refusal by declared kind stays as the opt-out; no new check; #471 closes as "decided: keep current behaviour" | W4 (docs only) | [00b](00b_traceability.md) section 4 |
| MQ-1 | Accept every open B2 recommendation as written, naming only the exceptions? | approved 2026-11-04 (owner decision 5) | yes | before W2.1 | [00_master_plan.md](00_master_plan.md) section 9.1 |
| MQ-2 | Which Postgres image base? | approved 2026-11-04 (owner decision 6) | `postgres:18.6-trixie` by digest everywhere | before W2.1 | [13](13_package_pins.md) section 4 |
| MQ-3 | Also run the Docker-free offline suite (4 shards) at each B2 part check? | approved 2026-11-04 (owner) | yes: every B2 part check runs it (about 10 min since PR #477) | W2.1 on | [03](03_b2_overview.md) section 4 |
| MQ-4 | Pillow and simplejpeg ride the W3 GPU gate with OpenCV; the Hugging Face stack and MLflow ride there by the bisect order | approved 2026-11-04 (owner) | yes: all four are members of pins part B (the W3 GPU gate) | W3 | [13](13_package_pins.md) section 7.1 |
| MQ-5 | #473: what happens to near-duplicates of held-out items at a holdout freeze? | approved 2026-11-04 (owner decision 8) | the whole near-duplicate group goes into the holdout together | W2.2 (2D-4) | [03d](03d_b2_s2_control_data.md) 2D-4 |
| MQ-6 | #468-#472 (five neutral security items, in no plan yet): one W4 lane scoped from the private notes; #469 and #471 need the owner's decision text | approved 2026-11-04 (owner decision 7); #469 and #471 still need the owner's decision text (OI-1, OI-2; both decided 2026-11-04 as "keep current behaviour", so the lane does #468, #470, #472 and a SECURITY.md statement) | a W4 lane `b4/privacy-hardening` | before W4 | [00_master_plan.md](00_master_plan.md) W4 |
| MQ-7 | #465 typed response models (BA-6): CPU lanes during W5a-W5c, merged into the B5 sub-batch PRs | approved 2026-11-04 (owner decision 7) | CPU lanes during W5a-W5c | before W5a | [05](05_b4_stateless_api.md) section 13 |
| MQ-8 | Fold the GPU-15 leftover (refuse an undeclared region head at profile validation) into B3-D3 | approved 2026-11-04 (owner) | yes: part of B3-D3 | W3 | [04](04_b3_ultralytics_one_version.md) B3-D3 |
| Q-3 / ENV-18 | Flower: loopback port 4613, the gateway subpath, or both? | approved 2026-11-04 (owner) | both: port 4613 on loopback and the gateway subpath, only when the `monitoring` profile runs | B2 stage 1 | [08](08_environment_and_setup.md) ENV-18 |
| MQ-9 | The research plans' experiments run after the tag, except what B5 already carries | approved 2026-11-04 (owner decision 9); **amended 2026-11-04 by RD-2**: short decision-feeding prototype runs on idle GPU during W2-W4, about 10 GPU hours in total | after the tag for everything else, each post-tag window on the owner's OK; their already-decided items stay folded into the rewrite | before W5a | [00_master_plan.md](00_master_plan.md) section 10; [00e_research_track.md](00e_research_track.md) |
| RD-1 | Build every research seam into the rebuild now | **approved 2026-11-04** (owner), as recommended; the review's deferrals rejected (Q-A), the plate-text index normalizer included, its upgrade close and reopen made safe and bounded; W2.3 split into W2.3a and W2.3b "to do in the right order and waves" | every seam now | W2.1 | [00d](00d_owner_questions.md) section 0; [03b](03b_b2_w2_job_families.md) |
| RD-2 | Prototypes on idle GPU during W2-W4 | **approved 2026-11-04, modified**: "we want to implement what we can of the research and prototypes, but we do not want long GPU hours"; short tiered runs on small seeded samples with early stop, no long sweeps, today's approach the default, about 10 GPU hours in total before the tag, more only on the owner's OK | short runs only | W2.1 | [00e_research_track.md](00e_research_track.md) section 2.1 |
| RD-4 | Plate tables in their own revision | **approved 2026-11-04**: `0004_region_text` in W5e | as recommended | W2.1 | [03g](03g_w5e_plate_text.md) |
| RD-5 | Capture metadata at ingest in W4 | **approved 2026-11-04** with every seam (RD-1) | as recommended | W4 | [05](05_b4_stateless_api.md) section 6c |
| RD-7 | Prototype code tracked with exact pins and a lock file | **approved 2026-11-04** | as recommended | PW-1 | 00e |
| RD-8 | Set P local only, "met / not met" in public text | **approved 2026-11-04**; one shared seeded copy | as recommended | PW-1 | [11](11_risks_cleanup_host.md) rule 10 |
| RD-10 | The research datasets, local only, aggregates with citations | **approved 2026-11-04, but skip UFPR-ALPR and LPLC** (freely downloadable plate sets plus set P; no academic request, review Q-C) | the dataset register | PW-1 | 00e section 4.1 |
| RD-23 | Cluster memberships in PostgreSQL | **approved 2026-11-04** | as recommended | W2.1 | [02d](02d_research_seam_schema.md) section 2.2 |
| RD-25 | The `cluster` queue and `cluster-worker` | **decided 2026-11-04** (review Q-D) | CPU in v0.7.0 | W2.1 | [03a](03a_b2_w1_queue_infra.md) 1B-8 |
| RD-51 | GPS coordinates | **reversed 2026-11-04: store them now**, behind a per-project `location_policy` (default `store`) in the W2.1 schema | store, policy per project | W2.1 | [02a](02a_postgres_schema.md) section 4 |
| RD-54 | Exported copies and their GPS | **approved 2026-11-04: exports keep the embedded GPS, as today**, by default; a `drop`-policy project's exports strip it (CM-5, review RN-4) | keep | W4 | 05 CM-4, CM-5 |
| Q-B (research review) | PP-OCRv6 as a reader candidate | **approved 2026-11-04**: one short comparison against PP-OCRv5 in the plate-text and B5c OCR work | as recommended | PW-2 | [07c](07c_b5_crops_ocr_faces.md) |
| RD-3, RD-6, RD-9, RD-11..RD-80 (the rest) | The remaining research owner questions | open: 69 rows, none blocks coding | the plan follows each recommendation and marks the place "pending RD-n"; each answer becomes its own row here | at the wave named | [00d_owner_questions.md](00d_owner_questions.md) |
| MQ-10 | Move #42 to v0.8.0 if the 12 GB run has not happened by the tag (D-5) | approved 2026-11-04 (owner decision 10) | yes | W6 | [07](07_b5_overview.md) section 11 |
| OQ-B5-1..3 | B5 open questions (public replacement for private acceptance bars, the 1M run, GPU 2 for two-card rows) | open | see the owning file | before B5a | [07](07_b5_overview.md) section 9 |
| OC-1 | Shared OCR detector-size helper for `/ocr` and the cascade (open choice from the B0 fixes) | open | it changes `/ocr` behaviour; decide in B1-1C | B1 | B1 1C (merged; see [CHANGELOG](../../../CHANGELOG.md)) |
| OC-2 | pyclipper in the hardened Triton scan image (`docker/hardened/triton`) | open | decide with B2's image work (stage 6, lane 6C) | B2 | [03c](03c_b2_w3_beat_reaper_workers.md) |
| OC-3 | `autolabel/job.get_state` still repairs through `_fail_unclaimed` | decided: leave | B2 stage 4 (lane 4D, trigger workers) deletes it | B2 | [03c](03c_b2_w3_beat_reaper_workers.md) |

## 7. Branch, lane and wave model

- **B0** merges `fix/411-r2-round` to `main` first, with a merge commit.
- **Each batch** gets `int/bN-<name>` (`int/b1-correctness`, `int/b2-control-plane`,
  `int/b3-ultralytics`, `int/b4-stateless`, `int/b5a-baseline` and so on) cut from `main` after
  the previous batch merged. A lane may branch from the previous integration branch's frozen
  head while that batch's PR is open; B3 is cut from `main` after B1 and rebased after B2.
- **Lane-brief rules (include in every brief).** Never use `git stash`: the stash is shared by
  every worktree of a repository, so a lane commits work in progress to its own branch instead.
  The file-size guard (700 lines) is a guide: a file that is one cohesive concern may exceed it
  with a `cohesive: <reason>` entry in `scripts/codegen/baselines/file_size.txt`, reviewed at
  merge; split by concern, never to dodge the number; existing baselines otherwise only shrink.
- **Each lane** is one agent in its own worktree `<repo>-wt-bN-<lane>` on branch `bN/<lane>`.
  Never two writers in one worktree. Lanes run only the targeted red/green tests of their own
  changes plus static checks on the files they touch. No full suites, no `--all-files` in a lane.
- **The coordinator** merges lanes `--no-ff`, one merge commit per lane in the stated order (so a
  failing lane reverts alone with `git revert -m 1 <merge>`), then runs exactly **one full sweep
  per batch** (B2's internal stages get a targeted check at each stage merge, and one sweep at the
  end), then opens one PR against `main` and merges it with a merge commit on green CI.
- **Agents:** Opus plans; Sonnet implements and reviews. Every brief forbids `pgrep -f`,
  `pkill -f` and sleep/until loops, and names its compose project (`-p`), GPU, worktree and
  deadline.
- **Commits:** conventional commits, one per fix or concern; hooks on (never `--no-verify`); a
  wire change commits `contracts/` in the same commit; nothing is pushed or merged by a lane.
- **Accepted churn** (review item A.5 #3, not adopted): B1 edits the claim lines of the twelve
  start paths and B2 edits their launch lines. Doing both at once would put Celery into B1.
- **No hard-coded SHAs or counts** for moving branches; read the head with `git log -1`.
- **Releases:** no VERSION bump, release images or GitHub release between batches. Breaking
  changes accumulate in `CHANGELOG.md` `[Unreleased]`. Local-only image tags wherever a gate or
  harness needs images.

## 8. Test tiers per batch

| Tier | B0 | B1 | B2 | B3 | B4 | R-final | B5 (each sub-batch) |
|---|---|---|---|---|---|---|---|
| T0 red/green + control per task | done on the branch | every lane | every lane | every lane | every lane | each `fix/rfinal-findings` fix | every lane |
| T1 pre-commit `--all-files`, offline suite (`-n 8`, about 25-33 min), contracts-check, docs check, changelog rules; frontend full set when `frontend/` changed; from B2 also the `db` tier (`-m db`) | once | once | once (after stage 6; the `db` tier also at each stage merge, D-63) | once | once | release-prep gate; the release commit's full gate | once |
| T2 live harness (`docker/test/compose.yml`, own `-p`) | - | lease/registry smoke, cap 30 min | the single B1+B2 run (about 4 h) | - | 1-vs-N diff (workers and replicas) | RF-03 | - |
| T3 GPU gates | - (CPU gates only) | - | - | GPU window 1: U-6.2 once for Stage A and B (about 1 day) plus the B0 OCR before/after (about 2 h) | - | GPU window 3: inside the run (segmenter, OCR set) | GPU window 2: baseline once, then each sub-batch measurement in sequence |
| T4 acceptance | - | - | `--only` job phases | - | - | full (section 4 of 06) | - |
| T5 benchmarks | - | lease latency only | job vs request latency | throughput gate | 32-worker baseline retaken on the B3 engine, then the knob | - | 5a baseline once, then gains |
| CI | the PR | the PR + the `families` job (new) | the PR | the PR + the full U-6.1 matrix in `families` | the PR | the release-prep PR, the release commit | each sub-PR |

Rules: local CI parity before every push (never discover a failure in CI); verify pushes with
`git ls-remote` and the run; heavy tiers are announced to peers and run one at a time; the
image-vs-git audit and Trivy run whenever an image is built for a gate or a push (B2, B3,
R-final, the release). The budget below caps how often each tier may run.

### 8.1 Test and GPU budget

Owner rule: no long or GPU runs per branch or per commit; testing happens in waves; compute and
time are limited. Every batch file follows this section and does not restate it.

| Tier | Budget |
|---|---|
| T0, per task | a targeted red/green test (plus its paired control) and static checks (ruff, mypy, the hooks that apply) on the touched files only. No full suite, no `--all-files`, no live or GPU run in a lane or a commit |
| One offline sweep per wave | on the merged integration branch, never per lane: the full suite `-n 8`, `pre-commit run --all-files`, `make contracts-check`, the docs check, and the frontend set only if `frontend/` changed |
| Live harness (`docker/test/compose.yml`) | exactly three runs: (1) a short B1 lease/registry smoke check, cap 30 min (replaces the old 90-minute B1 run); (2) one run at the end of B2 covering B1 and B2 together; (3) one at B4 (1-vs-32 workers). R-final's RF-03 is part of the R-final run itself, not a per-batch run |
| GPU windows | exactly three: (1) the B3 gate (U-6.2 for Stage A and B) plus the B0 OCR before/after comparison; (2) the B5 window: the #45 baseline once, then each sub-batch's measurement in sequence (5b, 5c, 5d); (3) R-final. Each window is announced to the peer sessions and run on the agreed GPU ([11](11_risks_cleanup_host.md) section 1). The contents of each window, in order, are in [00_master_plan.md](00_master_plan.md) section 5 (window 1 also carries the pins part B image smokes and the B2 stage-5 GPU smoke) |

Consequences: the B0 merge needs CPU gates only; a lane that needs a GPU number waits for its
window instead of running its own; a failed window is re-run only after the fix is merged to the
integration branch, not per attempt.

### 8.2 Plan lifecycle

Plans are working papers, not archives. The closing PR of each batch (and of each B5
sub-batch) deletes that batch's `program/` files in the same PR, after moving the lasting
knowledge into the permanent docs:

| What | Moves to |
|---|---|
| as-built design | `docs/ARCHITECTURE.md` or `docs/CURATION.md` |
| measured numbers | `docs/PERFORMANCE.md` (studies in `docs/research/`) |
| user-facing changes | `CHANGELOG.md` |
| API semantics | `docs/design/curation_api_contract.md` |

The same PR removes the batch's rows from this file and from `docs/design/README.md`. Each batch
file's done-when includes "delete this batch's plan files after moving the keep-list". After
R-final and the v0.7.0 tag the whole `program/` folder is removed; GitHub milestones carry
anything still open and git history keeps every plan. Only the living reference docs stay:
`curation_api_contract.md`, `curation_design_rationale.md`, `engineering_standards.md`. The removal
waves R-2 and R-3 of [12_prior_plans_reconciliation.md](12_prior_plans_reconciliation.md) follow
the same rule. This file, 08, 09, 10, 11 and 12 are removed with the folder, after their lasting
content (register of libraries, standards, host rules) is in `engineering_standards.md` and the
permanent docs.

## 9. Docs touched per batch

| Batch | ARCHITECTURE | CURATION | opensearch_schema_design | INSTALLATION | curation_api_contract + contracts/ | PERFORMANCE | other | CHANGELOG |
|---|---|---|---|---|---|---|---|---|
| B0 | - | - | - | - | - | (jitter already) | the plan set, `docs/design/README.md` | entries present; frontend #440 present |
| B1 | leases, registry | `job_busy` window, `registry_not_migrated`, migrate command | lease and registry docs, mapping | upgrade, downgrade note | BA-2, BA-5, new error codes | - | `CLAUDE.md` lock rule; ATTRIBUTION (cachetools) | root (breaking), frontend |
| B2 | the control plane (data ownership, Postgres, jobs, events), services, queues | job states, `queued`, `cancelling`, `paused` | two roles per project; the removed indexes | Postgres, Valkey, secrets, backup and restore, upgrade and downgrade | `queued`, 202 routes, job listing routes, removed error codes | live-run numbers | `CLAUDE.md`, `SECURITY.md`, `CONTRIBUTING.md` (test tiers), docs-site, ATTRIBUTION (SQLAlchemy, Alembic, psycopg, Celery, kombu, billiard, redis-py, Flower, Valkey, PostgreSQL) | root (breaking), frontend |
| B3 | supported model families | - | - | engine rebuild | export formats | U-6.2 results | README, MIGRATION_TRITON_26, image_size_audit, export/README, docs-site | root (breaking) |
| B4 | workers | - | - | `API_WORKERS` | `conflicts`, `building` on create, VLM batch 202 | retaken baseline, knob | engineering_standards (J7 exemption) | root, frontend |
| R-final | - | - | - | release upgrade notes | - | - | `docs/releases/v0.7.0.md`, `RELEASE_CHECKLIST.md`, `docs/design/README.md` | consolidated, then `[0.7.0]` |
| B5 | GPU pipeline | - | - | - | per sub-batch | all gates | `docs/research/` | root |

Config added or removed per batch (env keys, compose services, installer steps) is owned by
[08_environment_and_setup.md](08_environment_and_setup.md).

## 10. Scope boundaries

**Deferred, with the reason:**
- Performance (#40, #407, #219, #220, #67, #268, #45) waits for B4: Celery moves load out of the
  API, `API_WORKERS` changes the per-process pools, and Stage B changes the detector engine, so
  an earlier baseline would be thrown away. None is a live correctness bug.
- GPU-6 (`/detect` upscaling) is an owner decision that needs a recall measurement (D-35, B5a).
- BA-6 (71 untyped operations): one router per PR after B4, so routers that B2 and B4 rewrite are
  typed once (N15).
- Evidence-first items act only on a measured defect: OS-8, GPU-19, FE-6, T-6; OPS-2 (measured
  before B2); CC-10 (measured before B4) (N14).
- #369 API authentication stays blocked; B2 adds no public surface (the broker has no published
  port).
- Research and platform breadth: the research **seams** are in this program (approved, RD-1:
  [02d](02d_research_seam_schema.md), the lanes listed in [00b](00b_traceability.md) section 5.2),
  registration-plate text lands as wave W5e (#401, pending RD-3; its revision approved, RD-4), and
  the decision-feeding prototypes run as short runs on idle GPU in W2-W4, about 10 GPU hours in
  total (approved as short runs, RD-2). The **experiments and their landings** are
  the post-tag research waves RW-1..RW-6 ([00e](00e_research_track.md)): #346, #347, #348, #365,
  #398, #400, #406, #416, #435, #436, #466 (remainder), #84, #231, #403. Still out of this program:
  #8 (parked), #366 (done), #367, #370, #402 (v0.8.0 features)
  ([09_github_actions.md](09_github_actions.md) section 7 and 00b section 5.3 give each one's place).
- Refuted and cut audit items stay refuted and cut ([11_risks_cleanup_host.md](11_risks_cleanup_host.md)
  section 4).

**Backlog after B4 (not batched yet):**
- BA-6, one router per PR (N15).
- OS-8, FE-6, T-6 evidence tasks (N14).
- #368 remainder: per-family training adapters and the generic single-class exporter, after B3's
  slice.
- GPU-15 leftover: refuse a model with an undeclared region head at profile validation instead
  of at decode time (audit plan 4.3 GPU-15, "Left over, optional ... Not scheduled"). The decode
  path already fails closed; low.

## 11. Legends

**ID prefixes**

| Prefix | Where defined | What |
|---|---|---|
| `Q-P1.x`, `Q-Phase N`, `Q-Jn`, `Q-Dn`, `Q-En` | `docs/design/task_queue_plan.md` | Celery phases, judgment calls, defects D1-D15, edge cases E1-E26 |
| `BA-`, `OS-`, `GPU-`, `FE-`, `T-`, `OPS-`, `CC-`, `SEC-` | `docs/design/codebase_audit_plan.md` | audit tasks (section 4 there: fix, traps, red test, control) |
| `U-Step n`, `U-Jn`, `U-Cn`, `U-Rn`, `U-6.1`, `U-6.2` | `docs/design/model_family_versions_plan.md` | Ultralytics migration steps, judgment calls, corrections, risks, test matrix |
| `D-n`, `OC-n` | this file, section 6 | owner decisions and open choices |
| `N-n` | [09](09_github_actions.md) section 4 | GitHub issues to create |
| `W-n`, `W2.n` | [00_master_plan.md](00_master_plan.md) section 2 | program waves; W2 (B2) has four parts W2.1-W2.4; B2's stages are numbered 1-6 (the file names 03a, 03b, 03c keep older letters) |
| `MQ-n` | this file, section 6; [00_master_plan.md](00_master_plan.md) section 9 | owner questions raised by the master-plan pass |
| `PQ-n`, `PR-nn` | this file, sections 4 and 5 | wave exit gates; production-ready items |
| `RF-nn`, `RP-n` | [06](06_release_acceptance.md) | R-final checks; release-prep tasks |
| `GH-n`, `C-<issue>` | [09](09_github_actions.md) | GitHub steps; re-scope comments |
| `M-n` | [11](11_risks_cleanup_host.md) section 3 | risks |
| `B0-n` | [01](01_b0_close_out.md) | B0 tasks |

**Paths** (the public tree carries no machine paths):
- `<repo>` is the OpenProcessor checkout; `<repo>-wt-<name>` is a sibling git worktree.
- `<scratch>` is the host's scratch directory for acceptance installs.
- `artifacts_local/` is the gitignored evidence directory on the owner's machine: the ledger,
  the audit findings and verdicts (security details stay there), the Ultralytics evidence
  (`artifacts_local/model_versions/`), the R1/R2 reports, and each batch's gate evidence.
- `<scratch repro>` marks a throwaway script an audit used; each task's red test recreates it.
- **Line references** (`file:line`) in the three source plans are against `main` at `5b95591f`.
  Recreate that tree with `git worktree add --detach .claude/worktrees/audit-r2 5b95591f` if
  needed; code the branches changed has moved, so find it by symbol.

## 12. Where each section of the former master plan went

The former master plan (rev 4) is deleted in the plan-set commit and stays in git history.

| Master section | Now in |
|---|---|
| Header, mission, "How to use", path legend, ID legend | this file, sections 0, 1, 11 |
| 0 "Branch facts" | [01](01_b0_close_out.md) section 3 (current); this file section 7 (no hard-coded SHAs) |
| 0 "Inputs read for revision 2" | this file section 15 (revision log) |
| 1 TL;DR, release train | this file sections 2-4 (re-cut to one release by owner decisions 2-3) |
| 2 Mission check | this file section 2 (Correct/Fast columns); detail in each batch file |
| 2 "Deferred, and why" | this file section 10 |
| 3 Owner rules applied | [10_standards_and_libraries.md](10_standards_and_libraries.md) |
| 3a Replacement for the Celery plan's Phase 1 | B1 1A (merged; see [ARCHITECTURE](../../ARCHITECTURE.md)) (B0/B1 rows), [03a](03a_b2_w1_queue_infra.md) (Celery exclusive starts row) |
| 3b Every `flock` site (19 files, 55 hits) | B1 1A (merged; see [ARCHITECTURE](../../ARCHITECTURE.md)) (B1 rows, the ratchet), [03c](03c_b2_w3_beat_reaper_workers.md) (dispatcher, sweeper, cache prune), [05](05_b4_stateless_api.md) (event hub, `job_lock.py`) |
| 3c Q-D1..D15 cross-reference | B1 (merged; see [CHANGELOG](../../../CHANGELOG.md)) (B1 rows) and [03](03_b2_overview.md) (B2 rows); D1, D10, D11, D13 are B0 fixes recorded in [01](01_b0_close_out.md) |
| 3d Superseded source-plan tests | B0 rows: [01](01_b0_close_out.md) section 3 and Corrections; B1 rows: B1 1A (merged; see [ARCHITECTURE](../../ARCHITECTURE.md)), B1 1B (merged; see [ARCHITECTURE](../../ARCHITECTURE.md)) |
| 4 Release strategy | replaced by owner decisions 2-3 and D-37; this file section 6 (D-7, D-16, D-34, D-37) and [06](06_release_acceptance.md) |
| 5 B0 (5.1 scope, 5.2 gate cycle) | [01](01_b0_close_out.md) |
| 5.3 R3 live acceptance | [06](06_release_acceptance.md) sections 4.1, 4.2, 5 |
| 5.4 Tag decision point | [06](06_release_acceptance.md) section 7 |
| 6.1 J-LEASE, 6.3 trainer claims | B1 1A (merged; see [ARCHITECTURE](../../ARCHITECTURE.md)) |
| 6.2 J-REG | B1 1B (merged; see [ARCHITECTURE](../../ARCHITECTURE.md)) |
| 7 Common rules | this file section 7 |
| 7.1 B1 | B1 (merged; see [CHANGELOG](../../../CHANGELOG.md)); "Release: R-B1" -> [06](06_release_acceptance.md) section 4.3 |
| 7.2 B2 | [03](03_b2_overview.md), 03a-03c |
| 7.3 B3 | [04](04_b3_ultralytics_one_version.md) |
| 7.4 B4 | [05](05_b4_stateless_api.md); R-0.9 -> [06](06_release_acceptance.md) section 4.4 |
| 7.5 B5 | [07](07_b5_overview.md), 07a-07d |
| 7.6 #42 side track | [07](07_b5_overview.md) section 11 (GPU 1 host rule in [11](11_risks_cleanup_host.md) section 1) |
| 7.7 Backlog | this file section 10 |
| 8 Decision register | this file section 6 |
| 9 GitHub issue actions | [09](09_github_actions.md) |
| 10 Issue map | [09](09_github_actions.md) section 7 |
| 11 Task list | this file section 4.1 |
| 12 Test strategy | this file section 8 and 8.1 (the test and GPU budget, new) |
| (new) reconciliation of older plans | [12_prior_plans_reconciliation.md](12_prior_plans_reconciliation.md) |
| (new) plan lifecycle | this file section 8.2 |
| 13 Host, GPUs and peers | [11](11_risks_cleanup_host.md) section 1 |
| 14 Cleanup | [11](11_risks_cleanup_host.md) section 5 |
| 15 Risks and "Unverified" | [11](11_risks_cleanup_host.md) sections 3, 3.1 |
| 16 Docs per batch | this file section 9 |
| 17 Sources | [10_standards_and_libraries.md](10_standards_and_libraries.md) (library register), with each batch file citing its own |
| 18 Revision log | this file section 15 |
| Appendix A (review dispositions) | this file section 14 |
| Appendix B (refuted and cut) | [11](11_risks_cleanup_host.md) section 4 |

## 13. Older plans this set supersedes

[12_prior_plans_reconciliation.md](12_prior_plans_reconciliation.md) gives the verdict for every
older plan. The finished ones are removed by removal wave R-1 (task B0-1b in
[01_b0_close_out.md](01_b0_close_out.md)), the rest at R-final (R-2) or after the named B5
sub-batch (R-3); the statuses of the survivors are set in `docs/design/README.md`.

| Plan | Assessment |
|---|---|
| `program_plan.md` (2026-10-28, `main` at `96886e0a`) | **superseded** for ordering, milestones and the release mapping. Its research phases (clustering, spatial memory, the white paper) are out of this program and stay as parked issues (#346, #348) |
| `ship_track_plan.md` (2026-10-29) | **superseded**. Its waves shipped on `main` (the earlier "batch" merges) or are re-planned here (speed items in B5) |
| `v070_wave_plan.md` (2026-10-28) | **superseded**. Its performance waves are B5 (#45, #40, #219, #268); the rest shipped |
| `v060_execution_path.md` (2026-10-14) | **historical** (0.6.0 shipped); not a plan for this program. Mark "implemented in 0.6.0" |
| `docs/handoff/0.7.0-next-steps.md` | superseded and removed: its steps are done (R1, R2) or re-planned here |
| the former master plan (rev 4) | **replaced** by this set and deleted in the plan-set commit (in git history) |

Not superseded, still used by reference: `task_queue_plan.md`, `codebase_audit_plan.md`,
`model_family_versions_plan.md` (task detail by ID), `release_acceptance_plan.md` (the method
R-final follows), `processing_plans_plan.md` (B5d design), `triton_pipeline_optimization_plan.md`,
`sp_gpu_prep_concurrency.md` and `gpu_decode_plan.md` (B5b inputs), `engineering_standards.md`.

## 14. Review dispositions carried (Appendix A of the former master plan)

The review of 2026-11-03 (private notes) was applied in master plan revision 2. Where each item
lives now:

| Review items | Disposition and owner file |
|---|---|
| M-1 (#8, #346, #348, #365, #400, #401) | every open issue mapped: [09](09_github_actions.md) section 7; #365 interplay in B1 1B (merged; see [ARCHITECTURE](../../ARCHITECTURE.md)) |
| M-2 (audit Appendix C/D) | [11](11_risks_cleanup_host.md) section 4 |
| M-3 (#17/#18 leftovers) | [07c](07c_b5_crops_ocr_faces.md) (#220 step 3), [07d](07d_b5_processing_plans.md) (#67 phase 2+) |
| M-4 (per-process limits) | BA-8 documented in B1 1D (merged; see [CHANGELOG](../../../CHANGELOG.md)); pools sized per `API_WORKERS` in [05](05_b4_stateless_api.md) |
| M-5 (CC-4 interim) | the jitter shipped in B0; the known-issue line is dropped (D-6); CC-4 is fixed in B2 |
| M-6 (PR-3/4/6 as 0.7.x) | replaced by owner decision 2 (one release) |
| M-7, M-8 (superseded tests, Q-D cross-reference) | [01](01_b0_close_out.md), B1 1A (merged; see [ARCHITECTURE](../../ARCHITECTURE.md)), B1 1B (merged; see [ARCHITECTURE](../../ARCHITECTURE.md)), B1 (merged; see [CHANGELOG](../../../CHANGELOG.md)), [03](03_b2_overview.md) |
| M-9 (installer/upgrade for B1) | B1 1B (merged; see [ARCHITECTURE](../../ARCHITECTURE.md)); live in [06](06_release_acceptance.md) RF-40..RF-41 |
| M-10 (docs per batch) | this file section 9 and each batch file |
| M-11 (installer YOLO11 config divergence, U-C11) | accepted until B3, which deletes the heredoc: [04](04_b3_ultralytics_one_version.md) |
| M-12, M-13 (baselines only shrink; image audit) | PQ-2, PQ-8 (section 4) |
| W-1 (head/commit counts) | no hard-coded SHAs (section 7) |
| W-2 (CHANGELOG step done) | [01](01_b0_close_out.md) section 3 |
| W-3, W-4, W-5, W-9 (lease index, O_EXCL "added", B1 done-when, flock count) | B1 1A (merged; see [ARCHITECTURE](../../ARCHITECTURE.md)), B1 (merged; see [CHANGELOG](../../../CHANGELOG.md)) |
| W-6 (8.4.197 shipped) | done: the pin commit `d1c6bab0` |
| W-7 (GPU-15 leftover) | review partly not adopted: kept in the backlog with its source (section 10) |
| W-8 (#447 premise) | [09](09_github_actions.md) C-447; B1 1E (merged; see [CHANGELOG](../../../CHANGELOG.md)) |
| C-1 (release gap) | replaced by owner decision 2 |
| C-2 (one large B1 PR) | one merge commit per lane, revert coupling (section 7, M21); the single PR is kept |
| C-3 (CC-1 and resume) | [01](01_b0_close_out.md) Corrections; B1 1A (merged; see [ARCHITECTURE](../../ARCHITECTURE.md)) |
| C-4, C-5 (TTL and heartbeat; stale reclaim race) | B1 1A (merged; see [ARCHITECTURE](../../ARCHITECTURE.md)) |
| C-6 (OpenTranscribe alignment) | [03a](03a_b2_w1_queue_infra.md) re-reads it; listed unverified in [11](11_risks_cleanup_host.md) 3.1 |
| C-7, C-8 (trainer untested on GPU; three versions) | trainer moves in B3 behind U-6.2 ([04](04_b3_ultralytics_one_version.md)); the pin is done |
| A.4 executability gaps 1-7 (CHANGELOG, red-test names, pyclipper pin, OCR rollback, OCR image set, heavy-run scheduling, CC-1 resume test) | [01](01_b0_close_out.md); [11](11_risks_cleanup_host.md) section 1 |
| A.4 gaps 8-11, 18 (lease location, kind/scope, busy window, no-doc jobs, live race budget) | B1 1A (merged; see [ARCHITECTURE](../../ARCHITECTURE.md)) |
| A.4 gaps 12-14 (registry upgrade, bounds, trainer ordering) | B1 1B (merged; see [ARCHITECTURE](../../ARCHITECTURE.md)) |
| A.4 gaps 15, 16 (cachetools licence, BA-5 contracts; `families` CI) | B1 1D (merged; see [CHANGELOG](../../../CHANGELOG.md)), B1 1E (merged; see [CHANGELOG](../../../CHANGELOG.md)) |
| A.4 gap 17 (who rebuilds images for the pin) | resolved: the pin is done; R-final builds the images |
| A.4 gaps 19-21 (B2 compose/env/upgrade, jobs over 6 h, forward-only) | [03](03_b2_overview.md) |
| A.4 gaps 22-24 (B3 pin rule, B4 spike, B5 thresholds) | [04](04_b3_ultralytics_one_version.md), [05](05_b4_stateless_api.md), [07](07_b5_overview.md) |
| A.4 cross-cutting (docs per batch, contracts, upgrades) | section 9; [06](06_release_acceptance.md) (0.6.0 -> 0.7.0) |
| A.5 churn 1, 2, 4, 6, 7 | pin once then B3 ([04](04_b3_ultralytics_one_version.md)); minimal 1E (B1 1E (merged; see [CHANGELOG](../../../CHANGELOG.md))); one heartbeat class (B1 1A (merged; see [ARCHITECTURE](../../ARCHITECTURE.md)), [03a](03a_b2_w1_queue_infra.md)); baseline retaken on the B3 engine ([05](05_b4_stateless_api.md)); #365 out of scope (B1 1B (merged; see [ARCHITECTURE](../../ARCHITECTURE.md))) |
| A.5 churn 3 | not adopted, accepted cost (section 7) |
| A.5 churn 5 (Cropwright three times) | moot: everything ships in one release (M24) |
| A.6 doc suggestions (soft shutdown, uvicorn floor, fencing, wall clock, `nms`-only tests) | [03a](03a_b2_w1_queue_infra.md); done (`6a53673c`); B1 1A (merged; see [ARCHITECTURE](../../ARCHITECTURE.md)); B1 1E (merged; see [CHANGELOG](../../../CHANGELOG.md)) |
| A.7 specific edits 1-18 | applied in revision 2; their content now lives where the rows above say |

## 15. Revision log

| Revision | Date | Changes |
|---|---|---|
| master rev 1 | 2026-11-03 | first master plan merging the task-queue, audit and model-family plans into B0..B5 |
| master rev 2 | 2026-11-03 | the fresh review applied in full (section 14); lease, registry upgrade, trainer claims, release train, Ultralytics pin, decision register D-1..D-36. Inputs read: the review, the ledger, `docs/VISION_AND_GOALS.md`, the three source plans, the audit files, the open issues and milestones, read-only code checks, package versions in the rc5 images, the uvicorn 0.41.0 notes, the Ultralytics export page |
| master rev 3 | 2026-11-03 | B0 small fixes merged; their open choices assigned; T-8 added to B1-1D |
| master rev 4 | 2026-11-03 | plans committed with neutral paths; path legend; line references pinned to `main` `5b95591f` |
| program set 1 | 2026-11-03 | the master plan split into this set; owner decisions 1-9 applied (one release v0.7.0 after B4 and R-final, B5 = v0.8.0, milestones re-cut, D-6 dropped, D-7/D-16/D-34 replaced, D-37 and D-38 added, the pin marked done); execution waves W0-W10 and the production-ready definition added |
| program set 2 (consolidation) | 2026-11-03 | owner decision D-37 = yes (B5 inside v0.7.0, R-final last; waves renumbered W0-W6 with W5a-W5d; the v0.8.0 acceptance removed; milestones re-cut); section 8.1 test and GPU budget (live harness three times, GPU windows three); section 8.2 plan lifecycle; the `b1/deps` lane; B0 facts (`d1c6bab0`, `ad553ecd`, `0cc2b3ac`); file 12 added; the master plan deleted; older-plan removal wave R-1 |
| program set 3 (control plane) | 2026-11-04 | owner decisions of 2026-11-04 applied (D-39..D-45; D-17, D-18, D-31 approved; D-21, D-30 replaced; D-22 widened); files 02, 02a, 02b, 02c added; B2 re-planned in six stages (03, 03a-03f); B4's events track moved into B2; 05, 08, 10, 11 updated; open decisions D-46..D-63 |
| program set 4 (master plan) | 2026-11-04 | `00_master_plan.md` (execution order, waves W2.1-W6, GPU-window budget, wave procedure, owner questions) and `00b_traceability.md` added; this file became the register and legends (sections 3-4 point to the master plan, the duplicated sections 3-6 and the stale D-18 row removed); MQ-1..MQ-10 added; decided items folded into their lanes (00b section 3) |
| program set 5 (review fixes) | 2026-11-04 | the independent review [00c_plan_review.md](00c_plan_review.md) applied (R-1..R-38; planner responses in its section 5); the owner's answers of 2026-11-04 recorded (D-64..D-68, the open B2 recommendations as a block, MQ-1, 2, 5, 6, 7, 9, 10); OI-1, OI-2 opened; pins part A became its own wave W2.0 |
| program set 6 (re-check fixes) | 2026-11-04 | the re-check of [00c_plan_review.md](00c_plan_review.md) section 6 applied (RC-1..RC-13; responses in its section 5.1); D-64 replaced by D-69 (every service requires PostgreSQL, Valkey and OpenSearch, no degraded mode); OI-1, OI-2, MQ-3, MQ-4, MQ-8 and Q-3 answered by the owner; the open list reduced to the B3, B4, B5 and tag kick-off questions |
| research integration | 2026-11-04 | the seven research plans folded into the program on `research/master`: [02d](02d_research_seam_schema.md) (research-seam tables), [03g](03g_w5e_plate_text.md) (wave W5e), [00d](00d_owner_questions.md) (80 owner questions), [00e](00e_research_track.md) (prototype track and research waves); new lanes 1F, ES-A, ES-B, ES-C, 4F, ND-A, ND-B, ND-C, `b4/capture-metadata`, `b4/identity-text-privacy`, `tools/make-gate`, `b2/frontend-spaces`; 2D-4 moved to ND-B; the `NO ACTION` foreign-key rule (02a 1.2); resolutions X-1..X-20 and the issue map in [00b](00b_traceability.md) section 5; F10 (S3 storage) in the v0.8.0 plan |
| research final pass | 2026-11-04 | the research review [00e_research_review.md](00e_research_review.md) merged and every finding answered in its Planner responses (RR-1..RR-24); the owner answers of 2026-11-04 recorded (section 6: RD-1, RD-2, RD-4, RD-5, RD-7, RD-8, RD-10, RD-23, RD-25, RD-51, RD-54, review Q-A..Q-D); W2.3 split into W2.3a and W2.3b; GPS stored per `location_policy`; prototypes cut to short runs, about 10 GPU hours before the tag ([00b](00b_traceability.md) section 5.6) |

## 16. Corrections found while splitting

- 2026-11-04: the old B2 premise "`FileJob` files and the OpenSearch lease stay the truth" and the B4
  Redis Streams plan are replaced; the premises found wrong are listed in
  [02_control_plane_database.md](02_control_plane_database.md) section 5.

- `docs/design/README.md` lists none of the four plans of 2026-11-03; the plan-set commit adds a
  "Program plan (current)" section and the missing rows.
- Section 7 of the former master plan said to commit the source plans as the first commit of
  `int/b1`; they are already committed (D-14). Only the header lines needed repointing (done in
  the plan-set commit).
- R-B1 named a `class` acceptance phase; `acceptance_run.sh` has no such phase (06 section 5.2).
- B0's OCR comparison cannot run on the live harness (no Triton); B0-3 uses a real stack.
- Owner decision 3 asks for a "Breaking" subsection; the repo's convention (`[0.6.0]`) is
  "**Breaking:**" bullets under "Upgrade notes / behaviour changes". RP-2 follows the repo
  convention; rename to a literal `### Breaking` heading at release prep if the owner prefers.
- The former master plan named a peer session by its session name; names change, so this set says "the
  OpenTranscribe release session" and finds it with ListAgents.
- Per-file corrections: [01](01_b0_close_out.md) section 12 (CC-1 test file, pin files), [09](09_github_actions.md) section 10 (milestone renames, `build` label,
  #366, close points).
