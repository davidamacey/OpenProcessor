# 00b: traceability, fold-ins and proposed GitHub actions

| Field | Value |
|---|---|
| Status | plan, written 2026-11-04 with [00_master_plan.md](00_master_plan.md) |
| What it maps | (1) every task, ID and issue of the **original** program plans (the files under `docs/design/program/` on `origin/main` at `4f3f4813`, read with `git show origin/main:<file>`) to where it lives now, or "retired: reason"; (2) every open GitHub issue (44 on 2026-11-04) to a wave and lane, or to v0.8.0 or parked with a reason; (3) the decided items folded into lanes in this pass; (4) the GitHub actions, approved by the owner on 2026-11-04 (D-68) and run by the coordinator |
| Method | old and new text of every file compared; files unchanged since `origin/main` (01, 04 before this pass, 07, 07a-07d) are mapped by ID range; the rewritten B2 files (03-03c) and 05 item by item. Counts are table lines, not unique IDs |
| Old-to-new ID table for B2 | [03_b2_overview.md](03_b2_overview.md) section 14 (W1-n, 2A-2G, 2D-2F, B4 4C, B4 4A-6). Warning: the old 2A..2F (job families, beat) and the new 03d lanes 2A..2F (stage 2) share letters and are unrelated |

## 1. Original program tasks and where they live now

Status words: **kept** (same ID, same file), **moved** (renamed or relocated), **restored** (was
missing from the re-plan, put back in this pass), **retired** (removed on purpose, with the reason).

### 1.1 Summary

| Count | Lines |
|---|---|
| kept | 69 |
| moved | 140 |
| restored in this pass (missing whole items) | 3 |
| restored in this pass (missing sub-items) | 11 |
| **mapped in total** | **223** |
| retired, with a reason (section 1.9) | 16 whole items, 2 sub-items |

### 1.2 `00_index.md` (original)

| Original | Now |
|---|---|
| B0-1..B0-5, B0-1b, B0-a..B0-i (the old section 4.1 task list) | kept in [01](01_b0_close_out.md); B0 merged; B0-3 (OCR before/after) is gate step G10 of [04](04_b3_ultralytics_one_version.md) in GPU window 1 (W3); B0-f release prep is RP-1..RP-9 (W6); B0-h the tag (W6) |
| "#28 rest", #25, #22, #27, GH-1 (#38), #31, #34, B1-0, B1-1A..1E, R-B1, "B2 stages 1-6", B3, B4-4A..4D, R-0.9, "#17/#18 leftovers", B5a..d, "#40 (ledger)" | the old section 4.1 rows: done (B0, B1, #31, #34, B1-0), dropped (R-B1, its checks in [06](06_release_acceptance.md) section 4.3), or the waves of [00_master_plan.md](00_master_plan.md) (B2 = W2.1-W2.4, B3 = W3, B4 = W4, R-0.9 = W6 section 4.4, #220 step 3 = W5c, #67 phase 2+ = W5d, #42 = side track, cleanup per [11](11_risks_cleanup_host.md) section 5) |
| W0..W6 wave table (old section 4) | moved to [00_master_plan.md](00_master_plan.md) section 2 (W2 split into W2.1-W2.4; W3-W6 keep their meaning) |
| PQ-1..PQ-9, PR-01..PR-15 | kept (00_index sections 4 and 5; PR-01, 02, 05, 07, 10, 11, 15 re-worded for Postgres in the re-plan) |
| D-1..D-16, D-27, D-28, D-34..D-38 | kept unchanged |
| D-17, D-18 | kept, approved 2026-11-04 (the stale second D-18 row, "open, Redis 8", removed in this pass) |
| D-19, D-20, D-20a, D-20b, D-22a, D-22b, D-23..D-26, D-29..D-33 | kept, re-pointed to 02b, 02c and the stage files |
| D-21, D-30 | retired: replaced by D-39 and D-48 |
| D-22, D-31 | kept: D-22 widened by D-41, D-31 approved |
| OQ-B5-1..3, OC-1..OC-3, R-1/R-2/R-3 | kept (OC-2, OC-3 re-pointed from "stage W3" to stage 6 lane 6C and stage 4 lane 4D) |

### 1.3 `03_b2_overview.md` (original; no task IDs, every section and table row mapped)

| Original | Now |
|---|---|
| s1 done-when, s2 preconditions, s3 lanes, s4 diagram | moved: [03](03_b2_overview.md) sections 1-5 (stricter done-when: file-lock baseline empty, `rg fcntl src` empty, no `FileJob` or `job_lease`) |
| s5.1 families 1-6, 7-12, 13-18 | moved: [03b](03b_b2_w2_job_families.md) 4A-1..4A-4, 4B-1..4B-4, 4C-2..4C-7; promote to [03f](03f_b2_s5_gpu_queues.md) 5D-1..5D-3. `dataset_import_resume` is no longer its own kind: 4B-3 resumes the same `dataset_import` job (the rescan moves into the task) |
| s5.1 "not moved" list | kept in [02](02_control_plane_database.md) section 4.4 (label ops become inline `jobs` rows, 4A-5; legacy repair becomes an upgrade step, 2E) |
| s5.2 periodic loops, s5.3 workers, s5.4 allowlist, s5.5 lifespan rule | moved: [03c](03c_b2_w3_beat_reaper_workers.md) 6A-1; 02 sections 4.4 and 4.6; 03 section 1; 6A-3 |
| s6 queues, s7 Celery config, s8 broker service, s9 job record, lease, marker, wrapper, reaper, cancel | moved: [02b](02b_jobs_locking_events.md) sections 1-4 and 7, 1B-3, 1C-2 (Valkey); the lease became the claim `UPDATE` with a fencing token; reaper rules a-d became a-g |
| E1-E12, E14, E17-E21, E23-E26 (edge cases) | moved: T-11 and 3A-5 (E1, E2, E6, E9), 3A-3 (E3), 3A-7 rule c (E4, E12), 6A-2 and 6B-3 (E5), 1B-7 (E7), 1B-1 and 1C-2 (E8, E25), 1B-3 (E10, E24), 02b section 3.1 and 4C-5 (E11), 4E (E14), 1C-2 and T-19 (E17), 6B-1 (E18), 6A-4 (E19), 4B-4 (E20), 1C-3 and 6D (E21), 1B-6 (E23), [05](05_b4_stateless_api.md) section 7 (E26) |
| E11 unit test | restored: 3A-5 now tests `project_unavailable` for a job whose project was archived or deleted while queued |
| E13 (projects starve each other on `jobs`) | restored: [03](03_b2_overview.md) risk R2-12 (FIFO plus per-project exclusive kinds; per-project routing only if measured) |
| Q-D1, D4-D9, D14, D15 | moved: stage 4 (D1), 3A-7 and 6A-3 (D4), 6A-1 (D5), 4C-4 (D6), 4D and 5A (D7), 4D-1 (D8), 3A-4 (D9), B4 4B (D14), 1B-0 and 6A-4 (D15) |
| D-18..D-26 rows | moved (see 1.2) |
| s13 1-vs-32 equivalence | moved: [05](05_b4_stateless_api.md) section 7 |
| s14 sweep T1 and steps 1-8, 11-13 | kept or moved: 03 section 9 T1 (adds `-m db`) and steps 1-8, 14, 15, 16 |
| s14 step 9 (GPU arbiter with the overlay: one stop per configured container) | restored: 03 section 9 step 17 |
| s14 step 10 (two auto-label consumers, kill mid-stage, cancel at a stage boundary) | restored: 03 section 9 step 18 (live budget 4 h to 4 h 30 min) |
| s14 OS-6 hit counts recorded in `docs/PERFORMANCE.md` and `docs/opensearch_schema_design.md` | restored: step 15's pass column |
| s15 upgrade impact, s16 rollback, s17 docs and keep-list, s18 config, s19 release checks | moved: 03 sections 7, 8, 11-13; 6D; [02c](02c_migration_ops_tests.md) section 2 |
| R2-1..R2-8, R2-10..R2-12 (risks) | moved: 03 section 10 (R2-5, R2-8, R2-9, R2-10, R2-11), 11 M11, the 6B-1 test, 1C-2 `periodic-worker`, the kind-table limits |
| K-2..K-9, K-11..K-13, K-16, K-18 (corrections) | moved: 1B-1, 6A-1, 4C-3, 1C-2, 4C-5, 1B-2, 1B-4, 02 section 4.4, 02b, 02c section 4, 4C-6, 6A-3, [05](05_b4_stateless_api.md) K-18 |
| s22 OpenTranscribe re-read, s24-25 sources | moved: [02](02_control_plane_database.md) section 8; citations across 02b and 02c |

### 1.4 `03a` (original stage W1: queue infrastructure)

| Original | Now |
|---|---|
| W1-0 OPS-2 evidence | moved: [03a](03a_b2_w1_queue_infra.md) 1B-0 |
| W1-1 spike S1 | moved: 1B-1 (on Valkey; adds the `REDISCLI_AUTH` check) |
| W1-2 deps, pins, licences | moved: lane 1-0 and [13](13_package_pins.md); ATTRIBUTION rows in 6D |
| W1-3 `src/config/queue.py` | moved: 1B-2 (queue wait and auto-resume are kind-table columns, 02b section 2.1) |
| W1-4 Celery app | moved: 1B-3 |
| W1-5 producer | moved: 1B-4 |
| W1-6 `JobTask` wrapper and lease extension | moved: 3A-2..3A-5; the lease part retired (section 1.9) |
| W1-7 queued marker | moved: 1B-4 |
| W1-8 compose, dev overlay, harness | moved: 1C-2, 1C-3, then 6B-1 and 6B-3 |
| W1-9 pool and beat healthchecks | moved: 1B-6 (pool); restored in this pass: `tests/compose/test_beat_healthcheck.py` and the "nothing acts on `unhealthy`" compose comment in 1C-2 |
| W1-10 installer, CLI, secret | moved: 1C-4..1C-6 |
| W1-11 `/health` queue | moved: 1D-1 (adds `database`) |
| W1-12 layering and no-terminate guards | moved: 1A-11, 1B-7; restored in this pass: the clauses "nothing under `scripts/` imports `src.tasks`" and "`src/tasks` imports no `src.routers`, `fastapi`, `starlette`" in 1A-11 |
| W1-13 route-complete, consumers-from-compose, live `test_queue_infra` | moved: 1B-5, 6B-1; restored in this pass: 1B-5 now creates `tests/live/test_queue_infra.py` (sweep step 2 runs it) |
| W1-14 docs | moved: 6D, 1D-4 |

### 1.5 `03b` and `03c` (original stages W2 and W3)

| Original | Now |
|---|---|
| 2A-1..2A-6 | moved: 4A-1 (reprocess), 4A-2 (assist run), 4A-3 (scores, probe, selection), 4A-4 (embedding viz) |
| 2B-1, 2B-3, 2B-4, 2B-5 | moved: 4B-1, 4B-3, 4B-2, 4B-4 |
| 2B-2 promote, D-22b | moved: 5D-1..5D-3 |
| 2C-1..2C-6 | moved: 4C-2..4C-7 |
| 2C-7 fail-closed `OP_CROP_DUP_THRESHOLD` | moved: was 4C-8, now 4A-6 (moved with `near_dup.py` in this pass) |
| 2G frontend `queued` | moved: 4E |
| 2D-1 periodic tasks and beat | moved: 6A-1 (`lease_reaper` became `op.periodic.reaper`, 3A-7; `db_backup` added) |
| 2D-2 reaper rules | moved: 3A-7 (rules a-g); its job-family registry, combine-directory scan and mid-upgrade rule retired (section 1.9) |
| 2D-3 deletes and keeps, arbiter overlay | moved: 6A-3, 6B-4; restored in this pass: the CHANGELOG "Breaking" line and INSTALLATION note that the arbiter overlay now targets `periodic-worker` (6D) |
| 2D-4 OPS-2 act, worker metrics | moved: 6A-4 |
| 2E-1 auto-label consumer | moved: 4D-1; restored in this pass: the leftover `trigger.json` / `running.lock` test, now in 2E-3 (`tests/upgrade/test_leftover_job_files_removed.py`) |
| 2E-2 assist-loop consumer, 2E-3 one liveness mechanism | moved: 4D-2, 4D-3 |
| 2F SEC-4, SEC-5 | moved: 6C |
| 6A-1 beat-schedule name list vs B4's `resource_reachability` | restored in this pass: 6A-3 and [05](05_b4_stateless_api.md) 4A-5 say B4 extends the schedule test |

### 1.6 `05_b4_stateless_api.md` (original)

| Original | Now |
|---|---|
| 4A-0..4A-5, 4B-1..4B-5, 4D-0 and the 4D track, D-29, D-29a, D-32, D-32a, D-33, R4-n, K-18..K-22 | kept (Redis wording became Valkey; the 202 paths use B2's `start_job`; the B4 kinds modules `src/config/job_kinds/b4_fanout.py` and `b4_project.py` (with their `task_routes` and `src/tasks` twins) are now created empty by 1B-2, restored in this pass and split per track by re-check RC-5) |
| 4A-6 evaluator | moved: [03f](03f_b2_s5_gpu_queues.md) 5C-1..5C-3 |
| 4C-1 events, the last `flock`, `job_lock.py` | moved: [03e](03e_b2_s3_job_runtime.md) 3B-1..3B-5 and 6A-3; restored in this pass: `tests/events/test_publish_from_worker.py` in 3B-3 |
| FE-7 SSE reconnect | moved: 3B-4 (`Last-Event-ID` resume) |
| D-30, 4C-0 | retired (section 1.9) |

### 1.7 Files unchanged in content since `origin/main`

| File | IDs | Status |
|---|---|---|
| [01](01_b0_close_out.md) | B0-1..B0-5, B0-1b | kept (B0 merged) |
| [04](04_b3_ultralytics_one_version.md) | U-Step 4-9, B3-A1..A5, B3-F1..F3, B3-G1, B3-E1..E7, B3-D1..D3, B3-W1, G0-G10, RF-B3-1..8, U-R1..R8 | kept; this pass adds track `deps` (pins part B), gate steps G11 and G12, and the MQ-8 note in B3-D3 |
| [07](07_b5_overview.md), [07a](07a_b5_baseline.md)-[07d](07d_b5_processing_plans.md) | M-1..M-10, the parity gates, 5a-S1..S8, 5b-D1, 5b-P1, 5b-G1..G4, 5b-W1, 5b-C1, 5c-K1..K4, 5c-O1..O3, 5c-N1, 5d P0-P6, Q1-Q9, OQ-B5-1..3 | kept |
| [06](06_release_acceptance.md) | RF-01..RF-76, RP-1..RP-9 | kept (RF-40, 41, 56, 62 and RP-4, RP-5 re-worded for the control-data move) |
| [08](08_environment_and_setup.md) | ENV-1..ENV-20, Q-1..Q-4 | kept (ENV-4 = 1C-0, ENV-6 = 1B-3, ENV-14 = 1C-4, ENV-16 = 3B-5, ENV-18 = MQ list, ENV-20 = 4D-3 and 5A-5C) |
| [09](09_github_actions.md) | GH-0..GH-12, N1-N16, C-n | kept; N11, N12, N16 rows re-pointed to the new stage lanes in this pass |
| [10](10_standards_and_libraries.md), [11](11_risks_cleanup_host.md), [12](12_prior_plans_reconciliation.md) | STD-n, LIB-1..LIB-3; M-1..M-46; the ADD lists | kept; stale old B2 IDs re-pointed (10, 11) or explained by a note (12 section 0) |

### 1.8 New-plan gaps found and closed in this pass

| Gap | Closed by |
|---|---|
| D-53's stream-worker project claims were designed (02b section 5, 02a `stream_worker_claims`, reaper rule g, sweep step 13) but no lane implemented them | [03e](03e_b2_s3_job_runtime.md) 3A-10 (owns the claim calls in the detection and VLM workers) |
| `tests/test_jobs_single_entry.py` (02b, 10) had no lane | 3A-11 |
| `tests/config/test_db_connection_budget.py` (02b, 10, 11 M36; 02c called it `tests/db/test_connection_budget.py`) had no lane | 1C-7 (the 02b name is used) |
| `tests/compose/test_beat_single_instance.py` still asserted a fixed `container_name` (D-54 removes them) | 1C-2 now asserts `deploy.replicas: 1` |
| the B4 batch kinds module was not created in stage 1 | 1B-2 creates `job_kinds/b4.py`, `task_routes/b4.py`, `src/tasks/b4.py` empty |
| two different Postgres and Valkey image tags (13 section 4 vs 02c, 03, 03a, 08) | every B2 file now points at 13 section 4; the base (trixie or alpine) is MQ-2 |
| `00_index.md` held sections 3-6 twice, a malformed second register, and a stale D-18 | de-duplicated; this file and the master plan took the wave and task-list content |
| RF-42 (R-final): 02b section 4 said it was "updated" to the 150 s busy window, but it still said 120 s and "kill the API container", which no longer kills a job once jobs run in `jobs-worker` (review 00c R-23, section 3.7: "does not hold") | rewritten in [06](06_release_acceptance.md): kill the `jobs-worker` replica, `interrupted` within 150 s, then resume or restart; 02b section 4 now points at the rewritten row |
| the review [00c](00c_plan_review.md) (R-1..R-38) | every finding applied in the plan files or answered in 00c section 5 "Planner responses" |

### 1.9 Retired, with the reason

| Item (original file) | Reason |
|---|---|
| E15 `flock` on non-local volumes (03) | no file locks remain after B2 (true once 6A-3 empties the file-lock baseline, review 00c section 3.7); multi-host files are D-46 |
| E16 class-registry lock blocks the loop (03) | the registry is rows in Postgres (2B) |
| E22 runner and status poll write one record (03) | writes are fenced SQL statements (02b section 3) |
| D-21 job state in files (00, 03) | replaced by D-39 (owner, 2026-11-04) |
| D-30 event transport on Redis Streams (00, 05) | replaced by D-48 |
| 4C-0 Redis Streams spike (05) | void with D-48 |
| R2-9 "twelve start paths edited twice" (03) | the B1 lease claim lines it counted are deleted by B2 |
| K-1 queued lease TTL vs 120 s (03) | there is no lease; the job row holds queue state |
| K-10 the 19-file, 55-line `flock` count (03) | the baseline goes to zero (6A-3) |
| K-14, K-15 (03) | already obsolete or resolved in the original text |
| K-17 the OS-4 fix (03) | void in the original; 4B-4 carries no OS-4 item |
| section 23 "what not to migrate" (03) | the trainer, export worker and evaluator now migrate (D-41); detection and VLM stay stream workers (D-53) |
| W1-6 `test_job_lease_queued_state.py` (03a) | the lease is deleted (6A-3) |
| 2D-2 job-family registry, combine-directory scan, mid-upgrade rule (03c) | one indexed query over `jobs` replaces directory scans (02 section 5) |
| RF-62's "Q-1B-1" reference (06) | B1's registry migration never shipped; the 0.6.0 file moves straight to Postgres |
| sub-item: `tests/tasks/test_job_families_registry.py` (03c) | the registry it tested is gone |
| sub-item: "same `visibility_timeout` in both places" (03a W1-4) | there is no result backend (D-49), so that pair is gone; the different "several places" the review found (the worker, producer and GPU app factories) is now enforced by 1B-3 (R-33) |

## 2. Every open GitHub issue (44, read 2026-11-04)

| Issue | Milestone | Wave and lane, or deferral |
|---|---|---|
| #8 video analytics epic | none, parked | parked: a separate product; v0.8.0 plan F7 takes the first video-file slice ([v080](../v080_platform_breadth_plan.md) section 6) |
| #40 decode once | v0.7.0 | W5b (`decode`, `prep`, `routes`, `codec`) |
| #42 12 GB card | v0.7.0 | side track on GPU 1; owner-approved judgment calls (2026-11-02) already shipped in #441; only the acceptance run remains; MQ-10 |
| #45 benchmark ladders | v0.7.0 | W5a `ladders` |
| #67 processing plans | v0.7.0 | W5d |
| #84 wheel example | v0.8.0 | v0.8.0 F8: after training is stable |
| #219 crops at model size | v0.7.0 | W5c `crops` |
| #220 OCR and faces | v0.7.0 | W5c `ocr` (faces alignment in W5d `gpu`) |
| #231 serving head-to-head | v0.7.0 | GPU window 2 after the 5b gate, under the owner gate of 2026-10-10, or recorded and closed ([07](07_b5_overview.md) section 10) |
| #268 Triton follow-ups | v0.7.0 | W5a `sweeps` |
| #346 clustering research tracker | none, parked | parked: research after the tag (MQ-9); none of its J1-J11 is approved |
| #347 embedding-model research | v0.8.0 | research after the tag; owner decisions of 2026-10-29 bind its design, none touches product code |
| #348 spatial research | none, parked | parked (its J1-J8 not approved) |
| #365 embedding swap | v0.8.0, blocked | v0.8.0 F9, conditional on the research naming an encoder |
| #367 second segmenter | v0.8.0 | v0.8.0 F6 |
| #368 model-family registry | v0.8.0 | first slice in W3 (B3-F1); the rest v0.8.0 F4 |
| #369 API authentication | v0.8.0, blocked | v0.8.0 F5 (re-base its storage on Postgres before starting) |
| #370 importers | v0.8.0 | v0.8.0 F1 (only COCO segmentation-only boxes remain) |
| #398 detector-class roots (Z5) | v0.8.0 | research gate first; not folded into 4D (its J1-J10 are not approved) |
| #400 attribute signals | none, parked | parked |
| #401 plate identity text | none, parked | parked (needs an approved privacy design) |
| #403 MPS overlay | v0.7.0, parked | W5b conditional row (E8, adopt only with a 10 % gain); else closed or moved at W6 |
| #406 judge calibration | v0.8.0 | research after the tag |
| #407 GPU-prep staged speed | v0.7.0 | W5b `profile` then `prep` |
| #411 0.7.0 close-out | v0.7.0 | W6, closes after the tag |
| #416 near-duplicate threshold | v0.8.0 | research; its stale-field and race parts became #473 |
| #435 frame-target search | v0.8.0 | v0.8.0 F2 (recommendation there: close) |
| #436 `group_by=image` | v0.8.0 | v0.8.0 F2 |
| #439 project create 13-20 s | v0.7.0 | W4 lane `b4/project-create` |
| #451 jobs epic (N1) | v0.7.0 | W2.1-W2.4 and W4; closes with the B4 PR |
| #453 job leases (N3) | v0.7.0 | B1 delivered the leases; B2 deletes them for the jobs table (6A-3); close with the B2 PR |
| #458 one Ultralytics version (N8) | v0.7.0 | W3 |
| #461 monitoring-service privileges (N11) | v0.7.0 | W2.4 lane 6C |
| #462 build input pin and hardening (N12) | v0.7.0 | W2.4 lane 6C |
| #463 `/detect` upscaling (N13) | v0.7.0 | W5a `recall`, D-35 |
| #464 evidence tasks (N14) | v0.7.0 | OPS-2 in W2.1 (1B-0) and W2.4 (6A-4); CC-10 in W4 (4A-0); GPU-19 in W5c (`ocrnorm`); OS-8, FE-6, T-6 to a v0.8.0 follow-up per the issue body |
| #465 typed response models (N15) | v0.7.0 | CPU lanes `api/typed-responses-<n>` in W5a-W5c, the 5d routers inside 5d (owner decision 2026-11-04); closes with the 5d PR |
| #466 assist-loop pause, events, budgets (N16) | v0.8.0 | v0.8.0 F3; B2 4D-2 already delivers pause and resume through the jobs table and events as rows, so its scope shrinks (section 4) |
| #468 labeler log contents | v0.7.0 | W4 lane `b4/privacy-hardening` (owner decision 2026-11-04) |
| #469 region kinds to external endpoints | v0.7.0 | W4 lane `b4/privacy-hardening`, after the owner's decision text (OI-1) |
| #470 stored recognized text access | v0.7.0 | W4 lane `b4/privacy-hardening` |
| #471 research judge refusal by provenance | v0.7.0 | W4 lane `b4/privacy-hardening`, after the owner's decision text (OI-2) |
| #472 core text index isolation and purge | v0.7.0 | W4 lane `b4/privacy-hardening` |
| #473 near-duplicate data quality | v0.7.0 | folded in: part a W2.3 4A-7, part b W4 4A-4b, part c W2.2 2D-4 (the group goes into the holdout together, owner decision 2026-11-04) |

Open issues without a plan home before the master-plan pass: #465, #468, #469, #470, #471, #472,
#473. After the owner's answers of 2026-11-04 every one has a wave and lane; #469 and #471 also need
the owner's decision text (OI-1, OI-2) before their part of the W4 lane runs.

## 3. Fold-in record (decided items moved into the lane that rewrites their code)

A decided item is an owner-approved D-number of [00_index.md](00_index.md) section 6, an item
marked approved in a research plan, or an issue-plan judgment call whose issue carries the owner's
approval comment. Only #42 (of the open issues) carries that comment ("Owner decision (2026-11-02):
all judgment calls in the plan gist are approved as recommended"), and its calls are already
shipped (#441). No research-plan hypothesis, experiment or v0.8.0 feature is recorded as approved.

| # | Item | Decision recorded in | Moved into | Edit made |
|---|---|---|---|---|
| F-1 | the pin series split: part A in W2.1, part B (the bumps that can change model output) with the B3 GPU gate | owner direction 3 of 2026-11-04; D-44 (approved 2026-11-04) | [13](13_package_pins.md) section 7.1; [03a](03a_b2_w1_queue_infra.md) header; [04](04_b3_ultralytics_one_version.md) track `deps` and brief | the series order kept inside each part; torch-pair CI commands moved from commit 2 to 9; a shrink-only allowlist in commit 1 |
| F-2 | Pillow and simplejpeg join part B (commit 7) | the owner's own rule ("bumps that can change model output") applied to two image decoders; D-67 (2026-11-04) placed part B with the gate; MQ-4 confirms the exact members | 13 section 7.1 | `simplejpeg` was missing from the bisect list; added to commit 7 |
| F-3 | the Postgres and Valkey images and the test extras of 13 section 4 | D-44; [02](02_control_plane_database.md) says 13 owns every pin; `postgres:18.6-trixie` approved 2026-11-04 (MQ-2) | [03](03_b2_overview.md) header, 03a 1-0, 1C-1, 1D-2, [02c](02c_migration_ops_tests.md) section 4, [08](08_environment_and_setup.md) section 4 | the files now cite 13 section 4; the base image (trixie or alpine) is MQ-2 |
| F-4 | the image smokes of 13 section 6 and the B2 stage-5 real-GPU smoke in GPU window 1 | D-44 (13 section 6: "GPU and image work is batched into one window"); [03f](03f_b2_s5_gpu_queues.md) "GPU check" | 04 gate steps G11, G12 | budget about 1.5 GPU days |
| F-5 | #473 part c: the holdout freeze checks near-duplicates | a v0.7.0-milestone bug ("Each needs a red test first. The fix belongs with the curation correctness work (v0.7.0)"); the behaviour approved by the owner on 2026-11-04 (the whole near-duplicate group goes into the holdout together) | [03d](03d_b2_s2_control_data.md) 2D-4 (lane 2D rewrites the freeze) | new task |
| F-6 | #473 part a: a re-score refreshes the crop `dup_*` fields | as F-5 | [03b](03b_b2_w2_job_families.md) 4A-7 (lane 4A rewrites `item_scores/`) | new task; `near_dup.py` and its fail-closed task (was 4C-8, now 4A-6) moved to 4A so one lane owns the file |
| F-7 | #473 part b: the core batch grouping race | as F-5 | [05](05_b4_stateless_api.md) 4A-4b (4A-4 rewrites `visual_search/ingest_batch.py`) | new task |
| F-8 | the fail-closed near-duplicate threshold (clustering plan N-8) | already a program task (03b 4C-8) | 4A-6 | moved with its file |

Checked and already scheduled (no move needed): the OCR and faces judgment calls J1-J5 of
`faces_ocr_resident_crops_plan.md` (pre-approved; step 1 done, J5 in W5d); `gpu_decode_plan.md`
D1-D3 (implemented); person grouping as a job (4C-7); the CV-CUDA and DALI evaluations (W5b); MPS
opt-in only (#403); the #347 owner decisions (research, no product code); the clustering
agreement gate (research, after the tag).

Not folded (not decided; the reason): #398 Z5 partition (gated on its experiment; J1-J10 not
approved); #416 threshold change (needs its measurement); #466 token budgets and the UI pause
control (v0.8.0 F3, no owner go); #401 privacy baseline (design not approved); an `actor` column for
v0.8.0 auth (Q-1..Q-4 of the v0.8.0 plan open; the `jobs` rows already record `requested_by`);
uint8 faces with in-graph decode and CUDA graphs for the other models (unmeasured research
hypotheses); 07d Q1-Q9 (asked at W5d kick-off); the GPU-15 leftover (MQ-8).

## 4. GitHub actions (approved by the owner on 2026-11-04, D-68; the coordinator runs them after the review fix pass)

Rules as in [09](09_github_actions.md) section 1: plain words, neutral security wording (no
mechanism detail), no private names or paths, fewer issues is better, edit and close rather than
delete. Post each text exactly as written below; create N19 and N20 first and replace those two
placeholders in the other texts with their issue numbers; record the created numbers in the private
ledger and then in this file.

### 4.1 New issues (2)

**N19.** Title: `feat(control-plane): PostgreSQL for all control data, with migrations, row-level security and the one-step 0.6.0 upgrade`.
Labels: enhancement, area:curation, effort:hard. Milestone: v0.7.0. Body:

> Projects, classes, revisioned configs, settings, holdout freezes, label-operation records, ingest
> sources, dataset imports, combine runs, training runs, models and promotions move from OpenSearch
> documents and job files into PostgreSQL, with one migration history and per-project row-level
> security. OpenSearch keeps only the search data: two indexes per project instead of six. The
> upgrade from 0.6.0 moves the control data once, verifies it, then deletes the old copies, and can be
> rolled back from its bundle. New commands: `openprocessor db backup`, `db restore`, `db verify`. The
> API and the workers require PostgreSQL, Valkey and OpenSearch: they wait for them at start and
> refuse to start, naming the missing one, and `/ready` names any that goes down later.
> Tests use a real database (a container locally, a service container in CI).
>
> Part of #451. Plan: `docs/design/program/02_control_plane_database.md`, `03a`, `03d` (waves W2.1
> and W2.2 of `00_master_plan.md`).
>
> Acceptance: the stage 1 and stage 2 exit criteria of the plan; the release checks RF-40, RF-41,
> RF-56, RF-58, RF-59, RF-62 and RF-63.

**N20.** Title: `build(deps): exact pins at the latest installable versions in every manifest and image`.
Labels: chore, effort:moderate. Milestone: v0.7.0. Body:

> Every requirements file, `pyproject.toml` and inline Dockerfile pin becomes an exact `==` pin at
> the newest version that installs and works, the same in every manifest that shares a package. The
> few held-back pins are listed with their reasons. Part A (tooling, pure-Python libraries, the web
> layer, the clients) ships first as its own PR. Part B (the bumps that can change model output:
> numpy and pandas with the numeric stack, OpenCV 5, Pillow, simplejpeg, the Hugging Face stack,
> torch, the ONNX, ONNX Runtime, ModelOpt and TensorRT set, CuPy, the image pins, then MLflow and the
> Dependabot rules) ships with the Ultralytics GPU gate (#458), so the accuracy and speed check runs
> once for both. Inline image pins move into requirements files that Dependabot can read.
>
> Plan: `docs/design/program/13_package_pins.md`.
>
> Acceptance: the pin test passes with an empty allowlist; `pip check` and the smoke imports pass in
> every image; the image scans pass.

### 4.2 Re-scope comments (9)

| Issue | Comment to post (exact text) |
|---|---|
| #451 | "Scope update (owner decisions of 2026-11-04): background jobs now run on Celery with Valkey as the broker, and PostgreSQL holds every job and control record; training, model export and the bake-off evaluator move onto the queue too. The inference API keeps serving when the control plane is down. Child issues: N19 (PostgreSQL control plane), N20 (exact pins), #453, #461, #462, #439, #473. Plan: `docs/design/program/00_master_plan.md`, waves W2.1-W2.4 and W4." |
| #453 | "The per-project job leases landed with B1 (#467). The control-plane rewrite (#451) replaces them with job rows in PostgreSQL and deletes the lease code, so this issue closes with that PR." |
| #458 | "The dependency bumps that can change model output (part B of N20) ride this issue's GPU gate, so one accuracy and speed run judges the Ultralytics version and those bumps together. The gate also runs the image smoke checks and a short GPU check of the new training, export and evaluator queues. Plan: `docs/design/program/04_b3_ultralytics_one_version.md`, wave W3." |
| #464 | "Placement: OPS-2 is measured in W2.1 and acted on in W2.4; CC-10 is measured in W4 before the fan-out change; GPU-19 is checked in W5c. As the body says, OS-8, FE-6 and T-6 move to a v0.8.0 follow-up." |
| #465 | "Scheduled (owner decision of 2026-11-04): one CPU lane per wave during W5a-W5c types a group of routers that the speed work does not touch; the routers that W5d rewrites are typed inside W5d. This issue closes with the W5d PR." |
| #466 | "The queue rewrite (#451) already delivers pause and resume through the job rows and the events as database rows. What remains here: the Cropwright pause control, token budgets read from the endpoint's usage reports, and the label-rate measurement. Milestone stays v0.8.0." |
| #468, #470, #472 | (the same text on each) "Scheduled in wave W4 (owner decision of 2026-11-04). Details stay in the maintainers' private notes." |
| #469, #471 | (the same text on each) "Scheduled in wave W4 (owner decision of 2026-11-04), after the owner records the decision this issue asks for. Details stay in the maintainers' private notes." |
| #473 | "Placement: the stale near-duplicate fields after a re-score are fixed in W2.3; the batch grouping race in W4; the holdout check in W2.2. Owner decision of 2026-11-04: when a holdout is frozen, a held-out item's whole near-duplicate group goes into the holdout together. Each fix gets a red test first." |

### 4.3 Milestone moves (at W6)

| Issue | Move |
|---|---|
| #42 | v0.7.0 -> v0.8.0 if the 12 GB run has not happened by the tag (owner decision of 2026-11-04, D-5), with the comment "Moved to v0.8.0: the 12 GB card run did not happen before the v0.7.0 tag; the release notes say the small profile is not yet verified on a 12 GB card." |
| #403 | if W5b gave it no trigger: close as not planned with "Not adopted: the measured gap that would justify it did not appear (see docs/PERFORMANCE.md)", or move to v0.8.0 if the measurement was not taken |
| #231 | if GPU window 2 had no room: move to v0.8.0 with "Moved: GPU window 2 had no room for this comparison"; otherwise close with a link to its recorded result |

### 4.4 Closes

| Item | When | Comment |
|---|---|---|
| PR #474 (frontend dev dependencies) | after W2.0 merges | "Re-applied by hand in `<commit>` (pins part A, W2.0), following the rule that dependency bumps land through a reviewed deps lane during the program." |
| PR #449 (Triton base 26.10) | now, unless the owner places B3-A5 in W3 (D-38) | "The Triton and Python base moves are an owner decision (D-38 in `docs/design/program/00_index.md`): if taken, inside the B3 engine rebuild with the TensorRT pins moved together. Closing; Dependabot opens a fresh PR when due." |
| #453 | with the B2 PR (closing keyword) | - |
| #461, #462 | with the B2 PR (closing keywords) | - |
| #451, #439, #473, #468, #470, #472 | with the B4 PR (closing keywords) | - |
| #469, #471 | with the B4 PR once their owner decision is implemented | - |
| #465 | with the W5d PR | - |
