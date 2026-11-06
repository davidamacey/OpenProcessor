# 00b: traceability, fold-ins and proposed GitHub actions

| Field | Value |
|---|---|
| Status | plan, written 2026-11-04 with [00_master_plan.md](../00_master_plan.md) |
| Research integration | section 5 (added 2026-11-04): the cross-topic resolutions X-1..X-20, every research seam mapped to its lane, every research issue mapped to a wave or lane, the research findings put into lanes with red tests, and the seams changed or rejected with the reason |
| What it maps | (1) every task, ID and issue of the **original** program plans (the files under `docs/design/program/` on `origin/main` at `4f3f4813`, read with `git show origin/main:<file>`) to where it lives now, or "retired: reason"; (2) every open GitHub issue (44 on 2026-11-04) to a wave and lane, or to v0.8.0 or parked with a reason; (3) the decided items folded into lanes in this pass; (4) the GitHub actions, approved by the owner on 2026-11-04 (D-68) and run by the coordinator |
| Method | old and new text of every file compared; files unchanged since `origin/main` (01, 04 before this pass, 07, 07a-07d) are mapped by ID range; the rewritten B2 files (03-03c) and 05 item by item. Counts are table lines, not unique IDs |
| Old-to-new ID table for B2 | [03_b2_overview.md](../03_b2_overview.md) section 14 (W1-n, 2A-2G, 2D-2F, B4 4C, B4 4A-6). Warning: the old 2A..2F (job families, beat) and the new 03d lanes 2A..2F (stage 2) share letters and are unrelated |

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

### 1.2 `decision_register.md` (original)

| Original | Now |
|---|---|
| B0-1..B0-5, B0-1b, B0-a..B0-i (the old section 4.1 task list) | kept in [13](b0_close_out_done.md); B0 merged; B0-3 (OCR before/after) is gate step G10 of [04](../04_b3_ultralytics_one_version.md) in GPU window 1 (W3); B0-f release prep is RP-1..RP-9 (W6); B0-h the tag (W6) |
| "#28 rest", #25, #22, #27, GH-1 (#38), #31, #34, B1-0, B1-1A..1E, R-B1, "B2 stages 1-6", B3, B4-4A..4D, R-0.9, "#17/#18 leftovers", B5a..d, "#40 (ledger)" | the old section 4.1 rows: done (B0, B1, #31, #34, B1-0), dropped (R-B1, its checks in [07](../07_release_acceptance.md) section 4.3), or the waves of [00_master_plan.md](../00_master_plan.md) (B2 = W2.1-W2.4, B3 = W3, B4 = W4, R-0.9 = W6 section 4.4, #220 step 3 = W5c, #67 phase 2+ = W5d, #42 = side track, cleanup per [11](risks_cleanup_host.md) section 5) |
| W0..W6 wave table (old section 4) | moved to [00_master_plan.md](../00_master_plan.md) section 2 (W2 split into W2.1-W2.4; W2.3b folded into W2.3a by D-84 on 2026-11-05; W3-W6 keep their meaning) |
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
| s1 done-when, s2 preconditions, s3 lanes, s4 diagram | moved: [03](../03_b2_overview.md) sections 1-5 (stricter done-when: file-lock baseline empty, `rg fcntl src` empty, no `FileJob` or `job_lease`) |
| s5.1 families 1-6, 7-12, 13-18 | moved: [03d](../03d_stage4_job_families.md) 4A-1..4A-4, 4B-1..4B-4, 4C-2..4C-6 (person grouping, was 4C-7, now 4F-9, D-84); promote to [03e](../03e_stage5_gpu_queues.md) 5D-1..5D-3. `dataset_import_resume` is no longer its own kind: 4B-3 resumes the same `dataset_import` job (the rescan moves into the task) |
| s5.1 "not moved" list | kept in [02](../02_control_plane_database.md) section 4.4 (label ops become Celery jobs like every other long job, 4A-5, D-75; legacy repair becomes an upgrade step, 2E) |
| s5.2 periodic loops, s5.3 workers, s5.4 allowlist, s5.5 lifespan rule | moved: [03f](../03f_stage6_scheduler_scaleout.md) 6A-1; 02 sections 4.4 and 4.6; 03 section 1; 6A-3 |
| s6 queues, s7 Celery config, s8 broker service, s9 job record, lease, marker, wrapper, reaper, cancel | moved: [02b](../02b_jobs_locking_events.md) sections 1-4 and 7, 1B-3, 1C-2 (Valkey); the lease became the claim `UPDATE` with a fencing token; reaper rules a-d became a-g |
| E1-E12, E14, E17-E21, E23-E26 (edge cases) | moved: T-11 and 3A-5 (E1, E2, E6, E9), 3A-3 (E3), 3A-7 rule c (E4, E12), 6A-2 and 6B-3 (E5), 1B-7 (E7), 1B-1 and 1C-2 (E8, E25), 1B-3 (E10, E24), 02b section 3.1 and 4C-5 (E11), 4E (E14), 1C-2 and T-19 (E17), 6B-1 (E18), 6A-4 (E19), 4B-4 (E20), 1C-3 and 6D (E21), 1B-6 (E23), [05](../05_b4_stateless_api.md) section 7 (E26) |
| E11 unit test | restored: 3A-5 now tests `project_unavailable` for a job whose project was archived or deleted while queued |
| E13 (projects starve each other on `jobs`) | restored: [03](../03_b2_overview.md) risk R2-12 (FIFO plus per-project exclusive kinds; per-project routing only if measured) |
| Q-D1, D4-D9, D14, D15 | moved: stage 4 (D1), 3A-7 and 6A-3 (D4), 6A-1 (D5), 4C-4 (D6), 4D and 5A (D7), 4D-1 (D8), 3A-4 (D9), B4 4B (D14), 1B-0 and 6A-4 (D15) |
| D-18..D-26 rows | moved (see 1.2) |
| s13 1-vs-32 equivalence | moved: [05](../05_b4_stateless_api.md) section 7 |
| s14 sweep T1 and steps 1-8, 11-13 | kept or moved: 03 section 9 T1 (adds `-m db`) and steps 1-8, 14, 15, 16 |
| s14 step 9 (GPU arbiter with the overlay: one stop per configured container) | restored: 03 section 9 step 17 |
| s14 step 10 (two auto-label consumers, kill mid-stage, cancel at a stage boundary) | restored: 03 section 9 step 18 (live budget 4 h to 4 h 30 min) |
| s14 OS-6 hit counts recorded in `docs/PERFORMANCE.md` and `docs/opensearch_schema_design.md` | restored: step 15's pass column |
| s15 upgrade impact, s16 rollback, s17 docs and keep-list, s18 config, s19 release checks | moved: 03 sections 7, 8, 11-13; 6D; [02c](../02c_migration_ops_tests.md) section 2 |
| R2-1..R2-8, R2-10..R2-12 (risks) | moved: 03 section 10 (R2-5, R2-8, R2-9, R2-10, R2-11), 11 M11, the 6B-1 test, 1C-2 `periodic-worker`, the kind-table limits |
| K-2..K-9, K-11..K-13, K-16, K-18 (corrections) | moved: 1B-1, 6A-1, 4C-3, 1C-2, 4C-5, 1B-2, 1B-4, 02 section 4.4, 02b, 02c section 4, 4C-6, 6A-3, [05](../05_b4_stateless_api.md) K-18 |
| s22 OpenTranscribe re-read, s24-25 sources | moved: [02](../02_control_plane_database.md) section 8; citations across 02b and 02c |

### 1.4 `03a` (original stage W1: queue infrastructure)

| Original | Now |
|---|---|
| W1-0 OPS-2 evidence | moved: [03a](../03a_stage1_foundation.md) 1B-0 |
| W1-1 spike S1 | moved: 1B-1 (on Valkey; adds the `REDISCLI_AUTH` check) |
| W1-2 deps, pins, licences | moved: lane 1-0 and [01](../01_package_pins.md); ATTRIBUTION rows in 6D |
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
| 2C-1..2C-6 | moved: 4C-2..4C-6 and 4F-9 (person grouping, D-84) |
| 2C-7 fail-closed `OP_CROP_DUP_THRESHOLD` | moved: was 4C-8, now 4A-6 (moved with `near_dup.py`; the scorer is deleted by 4A, not ported, D-84) |
| 2G frontend `queued` | moved: 4E |
| 2D-1 periodic tasks and beat | moved: 6A-1 (`lease_reaper` became `op.periodic.reaper`, 3A-7; `db_backup` added) |
| 2D-2 reaper rules | moved: 3A-7 (rules a-g); its job-family registry, combine-directory scan and mid-upgrade rule retired (section 1.9) |
| 2D-3 deletes and keeps, arbiter overlay | moved: 6A-3, 6B-4; restored in this pass: the CHANGELOG "Breaking" line and INSTALLATION note that the arbiter overlay now targets `periodic-worker` (6D) |
| 2D-4 OPS-2 act, worker metrics | moved: 6A-4 |
| 2E-1 auto-label consumer | moved: 4D-1; restored in this pass: the leftover `trigger.json` / `running.lock` test, now in 2E-3 (`tests/upgrade/test_leftover_job_files_removed.py`) |
| 2E-2 assist-loop consumer, 2E-3 one liveness mechanism | moved: 4D-2, 4D-3 |
| 2F SEC-4, SEC-5 | moved: 6C (W2.4, the private items only; the public `no-new-privileges` and `cap_drop` lines moved to 1C-9 and 6B-9, D-84) |
| 6A-1 beat-schedule name list vs B4's `resource_reachability` | restored in this pass: 6A-3 and [05](../05_b4_stateless_api.md) 4A-5 say B4 extends the schedule test |

### 1.6 `05_b4_stateless_api.md` (original)

| Original | Now |
|---|---|
| 4A-0..4A-5, 4B-1..4B-5, 4D-0 and the 4D track, D-29, D-29a, D-32, D-32a, D-33, R4-n, K-18..K-22 | kept, except (D-84) the `/health` part of 4A-5 (now B2 1D-1), the `_msearch` part of 4A-5 (now B2 ES-C4), 4B-7 (now B2 ES-C5), and 4A-0 and 4A-3 if D-29a is answered before W2.3a (then B2 4D-8) (Redis wording became Valkey; the 202 paths use B2's `start_job`; the B4 kinds modules `src/config/job_kinds/b4_fanout.py` and `b4_project.py` (with their `task_routes` and `src/tasks` twins) are now created empty by 1B-2, restored in this pass and split per track by re-check RC-5) |
| 4A-6 evaluator | moved: [03e](../03e_stage5_gpu_queues.md) 5C-1..5C-3 |
| 4C-1 events, the last `flock`, `job_lock.py` | moved: [03c](../03c_stage3_job_runtime.md) 3B-1..3B-5 and 6A-3; restored in this pass: `tests/events/test_publish_from_worker.py` in 3B-3 |
| FE-7 SSE reconnect | moved: 3B-4 (`Last-Event-ID` resume) |
| D-30, 4C-0 | retired (section 1.9) |

### 1.7 Files unchanged in content since `origin/main`

| File | IDs | Status |
|---|---|---|
| [13](b0_close_out_done.md) | B0-1..B0-5, B0-1b | kept (B0 merged) |
| [04](../04_b3_ultralytics_one_version.md) | U-Step 4-9, B3-A1..A5, B3-F1..F3, B3-G1, B3-E1..E7, B3-D1..D3, B3-W1, G0-G10, RF-B3-1..8, U-R1..R8 | kept; this pass adds track `deps` (pins part B), gate steps G11 and G12, and the GPU-15 leftover in B3-D3 (MQ-8, approved) |
| [06](../06_b5_overview.md), [06a](../06a_b5_baseline.md)-[06d](../06d_b5_processing_plans.md) | M-1..M-10, the parity gates, 5a-S1..S8, 5b-D1, 5b-P1, 5b-G1..G4, 5b-W1, 5b-C1, 5c-K1..K4, 5c-O1..O3, 5c-N1, 5d P0-P6, Q1-Q9, OQ-B5-1..3 | kept |
| [07](../07_release_acceptance.md) | RF-01..RF-76, RP-1..RP-9 | kept (RF-40, 41, 56, 62 and RP-4, RP-5 re-worded for the control-data move) |
| [08](environment_and_setup.md) | ENV-1..ENV-20, Q-1..Q-4 | kept (ENV-4 = 1C-0, ENV-6 = 1B-3, ENV-14 = 1C-4, ENV-16 = 3B-5, ENV-18 = Q-3, decided 2026-11-04, ENV-20 = 4D-3 and 5A-5C) |
| [09](github_actions.md) | GH-0..GH-12, N1-N16, C-n | kept; N11, N12, N16 rows re-pointed to the new stage lanes in this pass |
| [10](standards_and_libraries.md), [11](risks_cleanup_host.md), [12](prior_plans_reconciliation.md) | STD-n, LIB-1..LIB-3; M-1..M-46; the ADD lists | kept; stale old B2 IDs re-pointed (10, 11) or explained by a note (12 section 0) |

### 1.8 New-plan gaps found and closed in this pass

| Gap | Closed by |
|---|---|
| D-53's stream-worker project claims were designed (02b section 5, 02a `stream_worker_claims`, reaper rule g, sweep step 13) but no lane implemented them | [03c](../03c_stage3_job_runtime.md) 3A-10 (owns the claim calls in the detection and VLM workers) |
| `tests/test_jobs_single_entry.py` (02b, 10) had no lane | 3A-11 |
| `tests/config/test_db_connection_budget.py` (02b, 10, 11 M36; 02c called it `tests/db/test_connection_budget.py`) had no lane | 1C-7 (the 02b name is used) |
| `tests/compose/test_beat_single_instance.py` still asserted a fixed `container_name` (D-54 removes them) | 1C-2 now asserts `deploy.replicas: 1` |
| the B4 batch kinds module was not created in stage 1 | 1B-2 creates `job_kinds/b4.py`, `task_routes/b4.py`, `src/tasks/b4.py` empty |
| two different Postgres and Valkey image tags (13 section 4 vs 02c, 03, 03a, 08) | every B2 file now points at 13 section 4; the base (trixie or alpine) is MQ-2 |
| `decision_register.md` held sections 3-6 twice, a malformed second register, and a stale D-18 | de-duplicated; this file and the master plan took the wave and task-list content |
| RF-42 (R-final): 02b section 4 said it was "updated" to the 150 s busy window, but it still said 120 s and "kill the API container", which no longer kills a job once jobs run in `jobs-worker` (review 00c R-23, section 3.7: "does not hold") | rewritten in [07](../07_release_acceptance.md): kill the `jobs-worker` replica, `interrupted` within 150 s, then resume or restart; 02b section 4 now points at the rewritten row |
| the review [00c](review_1_control_plane.md) (R-1..R-38) | every finding applied in the plan files or answered in 00c section 5 "Planner responses" |

Added by the research integration pass (owner-approved gaps of 2026-11-04):

| Gap | Where it lives now |
|---|---|
| (a) one local gate command for the sweeps | W2.0 lane `tools/make-gate`: `make gate` and `scripts/dev/gate.sh`, two parallel tracks after the installs, every step logged with its exit code ([00_master_plan.md](../00_master_plan.md) section 4.0a) |
| (b) five stale Cropwright screenshots (`projects-1600.png`, `ingest-policy-1600.png`, `ingest-policy-preview-1600.png`, `dashboard-1600.png`, `clusters-1600.png`) | recaptured in the W2.4 (PR3) live run, step 23 of [03](../03_b2_overview.md) section 9, committed in the B2 closing commit |
| (c) an S3-compatible storage backend for several hosts (D-46) | v0.8.0 feature F10 of [10_v080_platform_breadth.md](../10_v080_platform_breadth.md); issue text N21 (#483) in section 4.6 |

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
| #8 video analytics epic | none, parked | parked: a separate product; v0.8.0 plan F7 takes the first video-file slice ([v080](../10_v080_platform_breadth.md) section 6) |
| #40 decode once | v0.7.0 | W5b (`decode`, `prep`, `routes`, `codec`); its choices read `gpu:PR-1`, `PR-2` (PW-1) first (`PR-3`, `PR-4` cut before the tag: today's instance shape and backend stay, 5b-G4 not opened); the export orientation fix L-X1 rides 5b-D1 |
| #42 12 GB card | v0.7.0 | side track on GPU 1; owner-approved judgment calls (2026-11-02) already shipped in #441; only the acceptance run remains; MQ-10; its build is a `model_engines` row with `build_profile = 'small'` (GPU S-3) |
| #45 benchmark ladders | v0.7.0 | W5a `ladders`; the ladders read `JobPerfSummary` from `jobs.result` (GPU S-8, lane 3A-15) and record DCGM profiling fields |
| #67 processing plans | v0.7.0 | W5d; the resolved plan in `jobs.params.plan` (GPU S-9) and `op_plan_branch_total` (S-21) |
| #84 wheel example | v0.8.0 | v0.8.0 F8: after training is stable; the loop side is `al:AL5` and `feat/loop-retrain` in RW-6 |
| #219 crops at model size | v0.7.0 | W5c `crops`; the size-variant crop key is lane 1E-4 (GPU S-10) |
| #220 OCR and faces | v0.7.0 | W5c `ocr` (with `rec_char_min` for W5e; reads `gpu:PR-6` first); faces alignment in W5d `gpu` (reads `gpu:PR-5`) |
| #231 serving head-to-head | v0.7.0 | GPU window 2 after the 5b gate, under the owner gate of 2026-10-10, or recorded and moved to RW-1 ([06](../06_b5_overview.md) section 10) |
| #268 Triton follow-ups | v0.7.0 | W5a `sweeps`; the two-card and priority rows are RW-1 (`gpu:PR-7`, `PR-8`) |
| #346 clustering research tracker | none, parked | seams in v0.7.0: the partition tables (1F) and the clustering family on partitions (W2.3a lane 4F, D-84); experiments `clu:CL-0..CL-9` and landings L3-L6 in RW-3 (approved, RD-1, RD-6) |
| #347 embedding-model research | v0.8.0 | research after the tag as `emb:EP-4` in RW-2 under the embeddings plan (RD-37); the registry it needs ships in v0.7.0 (ES-A..ES-C) |
| #348 spatial research | none, parked | the one v0.7.0 slice is capture metadata and GPS coordinates at ingest under the per-project location policy (W4 lane `b4/capture-metadata`; RD-5 approved, RD-51 reversed, exports keep GPS per RD-54); place discovery and reconstruction are `sp:P-4..P-9` in RW-5, product work only on an S1 GO |
| #365 embedding swap | v0.8.0, blocked | the machinery ships in v0.7.0 (ES-A with ES-A4 in W2.2, ES-B in W2.3a, ES-C in W2.4); v0.8.0 F9 shrinks to register, build, evaluate, flip, still conditional on RW-2 naming an encoder |
| #367 second segmenter | v0.8.0 | v0.8.0 F6 |
| #368 model-family registry | v0.8.0 | first slice in W3 (B3-F1); the rest v0.8.0 F4 |
| #369 API authentication | v0.8.0, blocked | v0.8.0 F5 (re-base its storage on Postgres before starting) |
| #370 importers | v0.8.0 | v0.8.0 F1 (only COCO segmentation-only boxes remain) |
| #398 detector-class roots (Z5) | v0.8.0 | research `al:AL6` in RW-6 (calls RD-80); its seams ship now: the loop reads `(partition_id, cluster_uid)` from `cluster_partitions` (4D, 4F) and `detector_class` is a partition method value |
| #400 attribute signals | none, parked | `sp:P-2`, `sp:P-3` in RW-5; lane L-A1 only on a PASS (reserved DDL in [02d](../02d_research_seam_schema.md) section 5) |
| #401 plate identity text | none, parked (to v0.7.0, `parked` removed, 2026-11-05) | wave W5e ([06e](../06e_w5e_plate_text.md); its revision in W5e approved, RD-4; RD-3 answered as recommended, RD-61 and RD-62 answered, D-79 and D-80); creation-time seams in W2.1 (1C-9) and W2.2 (2F-6, kept by the owner, Q-A), the OCR output in W5c (PP-OCRv6 compared once, Q-B), the privacy baseline in W4; prototypes `pt:P0..P5` as short runs in PW-2 (`pt:P6` cut; UFPR-ALPR and LPLC skipped, RD-10); milestone v0.7.0 on the owner's go |
| #403 MPS overlay | v0.7.0, parked | MPS stays off in v0.7.0 (`gpu:PR-4` cut before the tag, review RN-6); moved to RW-1 at W6, where `PR-4` may run on the owner's OK |
| #406 judge calibration | v0.8.0 | `clu:CL-1` in RW-3 (protocol file section 5) |
| #407 GPU-prep staged speed | v0.7.0 | W5b `profile` then `prep`; reads `gpu:PR-2` (CV-CUDA) first; 5b-G4 (the C++ bar) is not opened, `PR-4` is post-tag in RW-1 (review RN-6) |
| #411 0.7.0 close-out | v0.7.0 | W6, closes after the tag |
| #416 near-duplicate threshold | v0.8.0 | the near-duplicate service ships in v0.7.0 with today's 0.98 defaults (ND-A, ND-B in W2.3a; ND-C in W4); `nd:P0..P2` as short runs in PW-1 (P1 may seed the defaults; the SSCD arm cut); `nd:P3..P7` and ND-E in RW-4 on the owner's OK |
| #435 frame-target search | v0.8.0 | decided by `emb:EP-2` in RW-2 (RD-38: close if the bar fails) |
| #436 `group_by=image` | v0.8.0 | v0.8.0 F2, composing with the search `space` parameter of ES-C |
| #439 project create 13-20 s | v0.7.0 | W4 lane `b4/project-create` |
| #451 jobs epic (N1) | v0.7.0 | W2.1-W2.4 and W4; closes with the B4 PR |
| #453 job leases (N3) | v0.7.0 | B1 delivered the leases; B2 deletes them for the jobs table (6A-3); close with the B2 PR |
| #458 one Ultralytics version (N8) | v0.7.0 | W3 |
| #461 monitoring-service privileges (N11) | v0.7.0 | W2.4 lane 6C |
| #462 build input pin and hardening (N12) | v0.7.0 | W2.4 lane 6C |
| #463 `/detect` upscaling (N13) | v0.7.0 | W5a `recall`, D-35 |
| #464 evidence tasks (N14) | v0.7.0 | OPS-2 in W2.1 (1B-0) and W2.3a (6A-4); CC-10 in W4 (4A-0), or in W2.3a (4D-8) if D-29a is answered before it; GPU-19 in W5c (`ocrnorm`); OS-8, FE-6, T-6 to a v0.8.0 follow-up per the issue body |
| #465 typed response models (N15) | v0.7.0 | filler lanes `api/typed-responses-<n>` whenever an agent is free, from W2.4 on, only on routers no active or later wave rewrites; a rewriting wave (B4 4A and 4D, 5d, W5e) types its own routers (MQ-7 amended by D-84); closes with the 5d PR at the latest |
| #466 assist-loop pause, events, budgets (N16) | v0.8.0 | pause, resume, budgets with token reservations, usage capture, slices and the event log now ship in W2.3a (4D-2..4D-7) and W2.4 (4E; gaps G-1..G-3, approved, RD-1); what remains for v0.8.0 is the review queue and metrics routes (`feat/loop-review`, RW-6) |
| #468 labeler log contents | v0.7.0 | W4 lane `b4/privacy-hardening` (owner decision 2026-11-04) |
| #469 region kinds to external endpoints | v0.7.0 | decided 2026-11-04: keep current behaviour (OI-1); closes with no code change; the SECURITY.md statement is a task of W4 lane `b4/privacy-hardening` |
| #470 stored recognized text access | v0.7.0 | W4 lane `b4/privacy-hardening` |
| #471 research judge refusal by provenance | v0.7.0 | decided 2026-11-04: keep current behaviour (OI-2); closes with no code change; covered by the same SECURITY.md statement |
| #472 core text index isolation and purge | v0.7.0 | W4 lane `b4/privacy-hardening` |
| #473 near-duplicate data quality | v0.7.0 | part a W2.3a ND-A (run-scoped groups, the projection clears stale fields); part b W4 ND-C1 (single-writer core grouping); part c W2.3a ND-B1 (moved from 2D-4, finding C8; the group goes into the holdout together, MQ-5) |

Open issues without a plan home before the master-plan pass: #465, #468, #469, #470, #471, #472,
#473. After the owner's answers of 2026-11-04 every one has a wave and lane, except #469 and #471,
which the owner decided as "keep current behaviour" (OI-1, OI-2): they close with no code change.

## 3. Fold-in record (decided items moved into the lane that rewrites their code)

A decided item is an owner-approved D-number of [decision_register.md](decision_register.md) section 6, an item
marked approved in a research plan, or an issue-plan judgment call whose issue carries the owner's
approval comment. Only #42 (of the open issues) carries that comment ("Owner decision (2026-11-02):
all judgment calls in the plan gist are approved as recommended"), and its calls are already
shipped (#441). No research-plan hypothesis, experiment or v0.8.0 feature is recorded as approved.

| # | Item | Decision recorded in | Moved into | Edit made |
|---|---|---|---|---|
| F-1 | the pin series split: part A in W2.1, part B (the bumps that can change model output) with the B3 GPU gate | owner direction 3 of 2026-11-04; D-44 (approved 2026-11-04) | [01](../01_package_pins.md) section 7.1; [03a](../03a_stage1_foundation.md) header; [04](../04_b3_ultralytics_one_version.md) track `deps` and brief | the series order kept inside each part; torch-pair CI commands moved from commit 2 to 9; a shrink-only allowlist in commit 1 |
| F-2 | Pillow and simplejpeg join part B (commit 7) | the owner's own rule ("bumps that can change model output") applied to two image decoders; D-67 (2026-11-04) placed part B with the gate; MQ-4 (approved 2026-11-04) confirms the exact members | 13 section 7.1 | `simplejpeg` was missing from the bisect list; added to commit 7 |
| F-3 | the Postgres and Valkey images and the test extras of 13 section 4 | D-44; [02](../02_control_plane_database.md) says 13 owns every pin; `postgres:18.6-trixie` approved 2026-11-04 (MQ-2) | [03](../03_b2_overview.md) header, 03a 1-0, 1C-1, 1D-2, [02c](../02c_migration_ops_tests.md) section 4, [08](environment_and_setup.md) section 4 | the files now cite 13 section 4; the base image (trixie or alpine) is MQ-2 |
| F-4 | the image smokes of 13 section 6 and the B2 stage-5 real-GPU smoke in GPU window 1 | D-44 (13 section 6: "GPU and image work is batched into one window"); [03e](../03e_stage5_gpu_queues.md) "GPU check" | 04 gate steps G11, G12 | budget about 1.5 GPU days |
| F-5 | #473 part c: the holdout freeze checks near-duplicates | a v0.7.0-milestone bug ("Each needs a red test first. The fix belongs with the curation correctness work (v0.7.0)"); the behaviour approved by the owner on 2026-11-04 (the whole near-duplicate group goes into the holdout together) | [03b](../03b_stage2_control_data.md) 2D-4 (lane 2D rewrites the freeze) | new task |
| F-6 | #473 part a: a re-score refreshes the crop `dup_*` fields | as F-5 | [03d](../03d_stage4_job_families.md) 4A-7 (lane 4A rewrites `item_scores/`) | new task; `near_dup.py` and its fail-closed task (was 4C-8, now 4A-6) moved to 4A so one lane owns the file; 4A deletes it, never ports it (D-84) |
| F-7 | #473 part b: the core batch grouping race | as F-5 | [05](../05_b4_stateless_api.md) 4A-4b (4A-4 rewrites `visual_search/ingest_batch.py`) | new task |
| F-8 | the fail-closed near-duplicate threshold (clustering plan N-8) | already a program task (03b 4C-8) | 4A-6 | moved with its file |

Checked and already scheduled (no move needed): the OCR and faces judgment calls J1-J5 of
`faces_ocr_resident_crops_plan.md` (pre-approved; step 1 done, J5 in W5d); `gpu_decode_plan.md`
D1-D3 (implemented); person grouping as a job (4C-7); the CV-CUDA and DALI evaluations (W5b); MPS
opt-in only (#403); the #347 owner decisions (research, no product code); the clustering
agreement gate (research, after the tag).

Not folded (not decided; the reason), as of the master-plan pass: #398 Z5 partition (gated on its
experiment; J1-J10 not approved); #416 threshold change (needs its measurement); #466 token budgets
and the UI pause control (v0.8.0 F3, no owner go); #401 privacy baseline (design not approved). The
research integration pass of section 5 now folds the **seams** of #398, #416, #466 and #401 into
lanes, each marked pending its RD question; the threshold change and Z5 itself stay research; an `actor` column for
v0.8.0 auth (Q-1..Q-4 of the v0.8.0 plan open; the `jobs` rows already record `requested_by`);
uint8 faces with in-graph decode and CUDA graphs for the other models (unmeasured research
hypotheses); 07d Q1-Q9 (asked at W5d kick-off); the GPU-15 leftover, folded into B3-D3 (MQ-8, approved 2026-11-04).

## 4. GitHub actions (approved by the owner on 2026-11-04, D-68; the coordinator runs them after the review fix pass)

Rules as in [09](github_actions.md) section 1: plain words, neutral security wording (no
mechanism detail), no private names or paths, fewer issues is better, edit and close rather than
delete. Post each text exactly as written below (N19 is #479 and N20 is #480, created 2026-11-04, and the
placeholders in the other texts are replaced). The texts of N19 and N20 in 4.1 are the 2026-11-04
wording and predate the 2026-11-05 decisions (latest pins D-70, no `revoke` D-71); the coordinator
rewrites those two issue bodies (section 4.7).

### 4.1 New issues (2)

**Done 2026-11-04:** N19 is #479 and N20 is #480; the 12 re-scope comments in 4.2 are posted (with #479/#480 in place of the placeholders); #469 and #471 are closed as decided; PR #449 is closed (PR #474 was closed 2026-11-04 too). Still to do at their points: the 4.3 moves at W6, and the closes that ride with their PRs.

**N19 (#479).** Title: `feat(control-plane): PostgreSQL for all control data, with migrations, row-level security and the one-step 0.6.0 upgrade`.
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
> Part of #451. Plan: `docs/design/v070_rebuild/02_control_plane_database.md`, `03a`, `03d` (waves W2.1
> and W2.2 of `00_master_plan.md`).
>
> Acceptance: the stage 1 and stage 2 exit criteria of the plan; the release checks RF-40, RF-41,
> RF-56, RF-58, RF-59, RF-62 and RF-63.

**N20 (#480).** Title: `build(deps): exact pins at the latest installable versions in every manifest and image`.
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
> Plan: `docs/design/v070_rebuild/01_package_pins.md`.
>
> Acceptance: the pin test passes with an empty allowlist; `pip check` and the smoke imports pass in
> every image; the image scans pass.

### 4.2 Re-scope comments (9)

| Issue | Comment to post (exact text) |
|---|---|
| #451 | "Scope update (owner decisions of 2026-11-04): background jobs now run on Celery with Valkey as the broker, and PostgreSQL holds every job and control record; training, model export and the bake-off evaluator move onto the queue too. Every service requires PostgreSQL, Valkey and OpenSearch: there is no degraded mode, and a missing dependency stops start-up with a clear message. Child issues: N19 (#479) (PostgreSQL control plane), N20 (#480) (exact pins), #453, #461, #462, #439, #473. Plan: `docs/design/v070_rebuild/00_master_plan.md`, waves W2.1-W2.4 and W4." |
| #453 | "The per-project job leases landed with B1 (#467). The control-plane rewrite (#451) replaces them with job rows in PostgreSQL and deletes the lease code, so this issue closes with that PR." |
| #458 | "The dependency bumps that can change model output (part B of N20 (#480)) ride this issue's GPU gate, so one accuracy and speed run judges the Ultralytics version and those bumps together. The gate also runs the image smoke checks and a short GPU check of the new training, export and evaluator queues. Plan: `docs/design/v070_rebuild/04_b3_ultralytics_one_version.md`, wave W3." |
| #464 | "Placement: OPS-2 is measured in W2.1 and acted on in W2.3a; CC-10 is measured in W4 before the fan-out change; GPU-19 is checked in W5c. As the body says, OS-8, FE-6 and T-6 move to a v0.8.0 follow-up." |
| #465 | "Scheduled (owner decision of 2026-11-04): each router is typed once, after (or by) the last wave that rewrites it: filler lanes take the routers no later wave rewrites whenever an agent is free, and the waves that rewrite a router (the stateless-API wave, the processing-plans wave W5d, the plate-text wave) type it in the same change. This issue closes with the W5d PR at the latest." |
| #466 | "The queue rewrite (#451) already delivers pause and resume through the job rows and the events as database rows. What remains here: the Cropwright pause control, token budgets read from the endpoint's usage reports, and the label-rate measurement. Milestone stays v0.8.0." |
| #468, #470, #472 | (the same text on each) "Scheduled in wave W4 (owner decision of 2026-11-04). Details stay in the maintainers' private notes." |
| #469, #471 | (the same text on each, posted with the close) "Decided by the owner on 2026-11-04: keep the current behaviour. When external use is enabled, this data may be sent as it is today, and the existing opt-out stays. No code change; the security policy will say this plainly in 0.7.0." |
| #473 | "Placement: the stale near-duplicate fields after a re-score are fixed in W2.3a; the batch grouping race in W4; the holdout check in W2.3a. Owner decision of 2026-11-04: when a holdout is frozen, a held-out item's whole near-duplicate group goes into the holdout together. Each fix gets a red test first." |

### 4.3 Milestone moves (at W6)

| Issue | Move |
|---|---|
| #42 | v0.7.0 -> v0.8.0 if the 12 GB run has not happened by the tag (owner decision of 2026-11-04, D-5), with the comment "Moved to v0.8.0: the 12 GB card run did not happen before the v0.7.0 tag; the release notes say the small profile is not yet verified on a 12 GB card." |
| #403 | if W5b gave it no trigger: close as not planned with "Not adopted: the measured gap that would justify it did not appear (see docs/PERFORMANCE.md)", or move to v0.8.0 if the measurement was not taken |
| #231 | if GPU window 2 had no room: move to v0.8.0 with "Moved: GPU window 2 had no room for this comparison"; otherwise close with a link to its recorded result |

### 4.4 Closes

| Item | When | Comment |
|---|---|---|
| PR #474 (frontend dev dependencies) | closed 2026-11-04 (done; was: after W2.0 merges) | "Re-applied by hand in `<commit>` (pins part A, W2.0), following the rule that dependency bumps land through a reviewed deps lane during the program." |
| PR #449 (Triton base 26.10) | closed 2026-11-04 (done; was: now, unless the owner places B3-A5 in W3 (D-38)) | "The Triton and Python base moves are an owner decision (D-38 in `docs/design/v070_rebuild/reference/decision_register.md`): if taken, inside the B3 engine rebuild with the TensorRT pins moved together. Closing; Dependabot opens a fresh PR when due." |
| #453 | with the B2 PR (closing keyword) | - |
| #461, #462 | with the B2 PR (closing keywords) | - |
| #451, #439, #473, #468, #470, #472 | with the B4 PR (closing keywords) | - |
| #469, #471 | closed 2026-11-04 (done), as "decided: keep current behaviour", with the comment above (owner decision 2026-11-04, OI-1 and OI-2) | - |
| #465 | with the W5d PR | - |
| #401 | with the W5e PR (closing keyword), on the owner's go (RD-3) | - |

### 4.5 Research comments (drafted 2026-11-04; RD-1 and RD-2 answered the same day; **not yet posted** as of 2026-11-05: post on the owner's go)

Same rules as above. One comment per issue; the plan path is the only link.

| Issue | Comment to post (exact text) |
|---|---|
| #346 | "Plan: `docs/design/v070_rebuild/09b_research_clustering.md` and its protocol file. The rebuild records every clustering run as a versioned partition with stable cluster identities and a served pointer, so a better method later is a configuration change. The experiments run after the 0.7.0 tag with registered bars and a blind human check." |
| #347, #365, #435 | "Plan: `docs/design/v070_rebuild/09a_research_embeddings_search.md`. The 0.7.0 rebuild adds an embedding registry (models, per-project spaces, a re-index job with pause and resume, a fenced switch), so a later encoder change is a data operation. The encoder evaluation and the whole-frame search question are decided by the experiments after the tag." |
| #348, #400 | "Plan: `docs/design/v070_rebuild/09f_research_spatial_attributes.md`. 0.7.0 stores capture time, camera, lens and GPS coordinates at ingest, under a per-project location policy (default keep). Colour and view tags, place grouping and reconstruction are research after the tag, with product work only on a go." |
| #401 | "Plan: `docs/design/v070_rebuild/09h_research_plate_text.md`. Proposed for 0.7.0 as its own wave after the speed work: per-box readings with a calibrated confidence, burst consensus, exact, fuzzy and fragment search, and privacy modes with a purge." |
| #406 | "Plan: section 5 of `docs/design/v070_rebuild/09c_research_clustering_protocol.md`. Runs after the 0.7.0 tag." |
| #416, #473 | "Plan: `docs/design/v070_rebuild/09d_research_near_duplicates.md`. 0.7.0 replaces the three separate near-duplicate paths with one service (whole images and crops), run-scoped groups in the database, an exact-content key, a single writer per project, and the holdout rule (a held-out item's whole group goes into the holdout). Thresholds stay at today's values until the measurements; they become a per-project setting. #473: part a and c are fixed by the new service in the queue wave, part b by the single-writer core grouping in the stateless-API wave." |
| #466, #398 | "Plan: `docs/design/v070_rebuild/09e_research_auto_label_assist.md`. The queue rewrite gives the assist loop its own record in the database, pause and resume that never use up retries, time slices so one project cannot starve another, budgets with token reservations, and an event log. What stays for 0.8.0: the review queue and the measured label rate and precision; the detector-class roots (#398) after their experiment." |

### 4.6 New issue for the coordinator to file: N21 (#483), the S3-compatible storage backend (approved addition 2026-11-04)

Title: `feat(storage): an S3-compatible storage backend for multi-host installs`.
Labels: enhancement, effort:hard, plus the existing area label that fits. Milestone: v0.8.0. Body:

> 0.7.0 stores every blob (uploads, crops, exports, training artifacts, model files, the upgrade
> bundle) behind one storage interface with a local-disk backend; several hosts share one POSIX
> volume. This adds a second backend for an S3-compatible object store, selected per deployment, so
> API and worker replicas on different hosts need no shared file system. Code that needs a real file
> (training, engine builds, index files) gets a job-local copy that is written back atomically;
> Triton reads the model repository from the store directly. Originals stay separate from
> regenerable derivatives. A migration command copies an existing install's blobs and verifies every
> key; database rows keep relative keys, so no row changes.
>
> Plan: feature F10 in `docs/design/v070_rebuild/10_v080_platform_breadth.md`.
>
> Acceptance: the storage contract tests pass on both backends; a two-host install with the object
> store ingests, trains, exports and promotes; the migration copies and verifies a 0.7.0 install;
> the release checks of F10.


### 4.7 Issue rewrites and milestone moves approved on 2026-11-05 (the coordinator does them)

| What | Detail |
|---|---|
| Issue bodies to rewrite | #451, #453, #479, #480, #401 and #466: bring each body to the plan as it now stands (Celery 5.7.0 and the latest pins, D-70; cancel as a database write, D-71; the three-PR B2, D-76; plate text as W5e) |
| Milestone moves | only #401 to v0.7.0 (approved 2026-11-05; it has none today, remove its `parked` label). #348, #365, #416 and #466 keep their milestones; their v0.7.0 slices are listed in the #479 and #466 bodies. #401's draft text no longer mentions privacy modes (D-79) |
| Where recorded | [decision_register.md](decision_register.md) D-70..D-78; [github_actions.md](github_actions.md) status line |

## 5. Research integration (2026-11-04)

The seven research plans (`docs/design/research_*_plan.md`) each listed the seams the rebuild must
include (their section 5), landing lanes (8), owner questions (10) and edits for other plan files
(11). This section records how they were folded: one consistent design where two plans touch the
same thing (5.1), every seam with its lane (5.2), every research issue with its wave (5.3), the
findings turned into lanes with red tests (5.4), the seams changed or rejected (5.5), and the
review and owner-answer changes of the final pass (5.6). Every seam was approved by the owner on
2026-11-04 (RD-1 of [00d](owner_questions.md), every seam now; review Q-A); a row that names
another RD-n follows that open question's recommendation.

### 5.1 Cross-topic resolutions

| # | Overlap | Resolution | Reason |
|---|---|---|---|
| X-1 | grouping records: clustering partitions, near-duplicate runs, plate consensus groups, auto-label selections (and later scenes) | **distinct tables, one shared pattern.** Each is a run record (params validated by pydantic, the profile pinned as `(config_id, revision)` referencing `config_revisions` with `NO ACTION`, `job_id ... ON DELETE SET NULL`), membership bulk-written with `COPY`, one served or current pointer guarded by a partial unique index or an OCC activation, and an OpenSearch projection through the `search_projection_tasks` outbox (kinds `cluster_partition`, `near_dup_run`, later `scene_membership`) with the reaper re-creating a lost sync job (rule i). Large membership tables (`cluster_memberships`, `partition_clusters`, `near_dup_groups`, `near_dup_members`) carry a denormalized `project_id` with a composite foreign key **and** the single-column one (a composite key is not checked when `project_id` is NULL); budget-bounded children (the loop tables) keep the `EXISTS` policy. One Jaccard lineage helper `src/domain/group_lineage.py` (4F) serves clustering carry-over and near-duplicate lineage. Near-duplicate `leak` groups are must-link input to clustering (S-later `cluster_constraints.source = 'near_dup'`); plate identity links enter as `source = 'identity_text'`; plate consensus may read the near-duplicate image groups as its basis | the lifecycles differ (human-gated activation; a current flip with holdout absorption; replace-in-place with no history; a per-run audit with 90-day retention), so one table would need a type column and nullable fields for each; the shared pattern keeps each simple and the same to review |
| X-1b | `ON DELETE RESTRICT` between two rows that one project delete cascades to | **`NO ACTION` everywhere for such pairs** (02a section 1.2; also fixes the existing `config_activations`, `promotions` and `training_runs` references) | the documented difference is that `NO ACTION` can be deferred and `RESTRICT` cannot (review RR-6 corrected the earlier wording); the project delete does not rely on one cascade (the batched per-table delete of 4C-5, RR-7), the catalog-enumerated test decides, and a pair that still fails is declared `DEFERRABLE INITIALLY DEFERRED`; `tests/db/test_profile_fk_cascade.py` proves it |
| X-2 | config kinds | v0.7.0 adds `serving_profile` (deployment), `cluster_profile` (project; a deployment variant only with the faces lane L5), `assist_loop_preset` (project), `near_dup_profile` (project, ND-A's revision). `attribute_profile` and `spatial_profile` are two distinct kinds, both reserved (02d section 5). The plate fields are `region_profile` body fields owned by W5e (not a new kind; the privacy mode is a table, not a document, because it holds a salt). The deployment serving profile is activated through `deployment_settings('active_serving_profile')`, not `config_activations` | `config_activations` is project-scoped by design; a deployment document needs a deployment pointer, like `local_vlm_desired` |
| X-3 | queues | v0.7.0: `cluster` (W2.1, `cluster-worker`, CPU) and `text_read` (W5e, `text-read-worker`, no GPU); no GPU-prep queue (GPU S-11); device-scoped queues only when a family declares itself so (RW-1); `gpu_spatial` reserved | each new queue exists because a long job must not hold a `jobs` slot that request-started work needs; a queue is declared only with its consumer (no dead route) |
| X-4 | OpenSearch fields and index-creation settings | families each in their own mapping module registered additively by 2F's registry: `cluster_*` (4F), `near_dup_*` and `content_sha256` (ND-A), `vec_<space_key>` and `boxvec_<space_key>` (ES), `capture.*` on images and `capture_taken_*` on items (W4), the plate `text_*` keys of a region box (W5e); `attr_*`, `scene_ids`, `place_embedding`, `capture.location` reserved. Creation-time settings: the plate normalizer `op_text_compact` in the items body, added to **upgraded** indexes by close, settings, open inside the upgrade window (2F-6); `index.knn = true` on images guarded by a door test (2F-7); k-NN method per field from `KnnMethodSpec` (ES-A3). `imohash` and the writer-less `phash` leave images (RD-43, answered as recommended) | analysis settings are static, so a later addition would need a close and reopen or a reindex of every project; one module per family keeps the W2.3a research lanes file-disjoint |
| X-5 | the model registry: the GPU plan's engine contracts vs the embeddings plan's `embedding_model_versions` | **two grains, one contract.** `model_engines` is one built Triton engine (physical facts, `input_contract`, `output_contract`, build profile, measured memory); `embedding_model_versions` is the logical encoder catalog (weights, modalities, dimensions, text prefixes, licence). `EncoderInputSpec.image` **is** an `EngineInputContract` (one pydantic class in `src/schemas/models.py`) and must equal the ready engine's contract (ES-A1 test). Clustering and near-duplicate runs record `embedding_model = space_key` and `preprocess_id = input_spec_sha256`, not free text. The W3 gate G7 writes `engine_parity_min_cos`; a lower parity makes a new model version | one definition of every pixel input for the CPU route, the GPU route, the registry and the research; a contract diff is a gated change, not a hidden edit in two files |
| X-6 | `gpu_devices` vs GPU reservations | `gpu_devices` (inventory by NVML UUID, written by the GPU workers at start and on every heartbeat, by `serving_inventory` and by the reservation insert; review RR-9) is referenced by `gpu_reservations(gpu_uuid)`; `training_runs.gpu_uuids text[]`; serving-profile servers name UUIDs; research scripts set `CUDA_DEVICE_ORDER=PCI_BUS_ID` | CUDA numbers cards fastest first, so an index is ambiguous across processes (the measured 12 GB-card trap) |
| X-7 | job control across topics | pausable kinds: `assist_loop`, `autolabel` (RD-69, answered as recommended), `embedding_reindex`, `source_backfill`; `slice_s` for `assist_loop`, `embedding_reindex` and `source_backfill` (review RR-20); pause never revokes the task id (review RR-1); `retry_count` and cancel-is-terminal apply to every kind | G-1 is a generic claim bug, not a loop bug; pause means the same thing everywhere (RD-70) |
| X-8 | holdout | one rule source (`occ_locks.py`): the loop never selects held-out items; clustering pools exclude them; ND-B owns the near-duplicate closure (replacing 2D-4) | one owner per rule |
| X-9 | durable records vs events | every topic's record lives in its own tables; the 24-hour `events` rows are the UI feed only (G-3) | a 24-hour table cannot be an audit log |
| X-10 | the 0.6.0 upgrade for research data | a step registry (2E-8): W2.3a lanes add `clustering.py` and `near_duplicates.py` as new files; the `source_backfill` job (ND-A, capture step W4) back-fills the exact key and the capture data of upgraded images (review RR-10) | file-disjoint lanes across waves |
| X-11 | `project_state` keys the clustering plan removes | `region_cluster_*` markers and `last_clustering` are never ported (they become partitions); **`cluster_refresh` stays**, redefined by 4F as the beat's scheduling clock (deferral start, last attempt, backoff) with the trained count read from the serving partition | overrides one line of the clustering plan's section 11: the refresh rule's deferral and backoff need a durable per-project clock that no partition row holds, and 2F deletes the settings index that holds it today in W2.2 |
| X-12 | prototype IDs (three plans use `P0`) | topic prefixes in program text: `gpu:`, `clu:`, `emb:`, `nd:`, `sp:`, `pt:`, `al:` ([00f](../08_research_prototype_track.md) section 1) | unambiguous references across files |
| X-13 | set P | one rule (RD-8, approved): one shared seeded local copy under one manifest, removed after PW-3; public text "met / not met"; aggregates only with the owner's yes per note; never to a hosted model (RD-9) | the plans differed (met/not met, aggregates allowed, per note); the strictest common rule wins until the owner says otherwise |
| X-14 | GPU windows and locks | short prototype runs in PW-1..PW-3 on idle GPU during W2-W4, about 10 GPU hours in total (RD-2 as answered), research windows RG-1..RG-6 after the tag only on the owner's OK, never overlapping a program window or a live run; one lock file per card `$SCRATCH/locks/gpu<N>.lock` shared by every topic | the plans named three different lock files, which would let two topics share a card |
| X-15 | research pins | each topic's `scripts/research/<topic>/requirements*.txt` with a lock, never the product manifests; GPL packages never in a product manifest ([01](../01_package_pins.md) section 3.9) | reproducibility without changing the product |
| X-16 | capture time | the spatial plan's `capture.*` (W4) is the one capture time; the plate plan's `captured_at` (its J13) is replaced by `capture_taken_utc` on items (RD-68); the clustering signals table's `capture_time` and `burst_key` are not added | one owner per value |
| X-17 | privacy text across W4 lanes | `b4/privacy-hardening` owns `SECURITY.md` in W4 and writes the three statements (external endpoints; location stored per `location_policy` and exports keeping the embedded GPS; identity text); the capture and identity-text lanes hand their text in the plan | one writer per file per wave |
| X-18 | VLM usage capture across W2.3a lanes | 4D owns `vlm_client.py` and makes `usage_source` a required keyword; 4D merges before the other VLM callers, which add the keyword in their own files against the fixed signature | every VLM call is counted (`vlm_usage_hourly`) with no shared file |
| X-19 | the IVF retrain | deleted with the auto-label worker by 4D (W2.3a) and never ported (D-84), with no `IVF_RETRAIN` lock and no `ivf_retrain_check` beat task; it is a `cluster_run(method='ivf')`; `cluster_refresh` starts `cluster_run` | one mechanism for every clustering run |
| X-20 | the embedding re-index decode | ES-B reuses 4A's reprocess decode path (4A merged in W2.3a) and is `gpu_affinity = 'any_card'` | one decode per image (the B5 rule) |

### 5.2 Every research seam and its lane

Status words: **folded** (in the lane named), **moved** (to another wave than the plan proposed,
with the reason), **reserved** (DDL kept, created by a later lane), **changed** (folded with a
change, 5.5).

**GPU pipeline** ([09g_research_gpu_pipeline.md](../09g_research_gpu_pipeline.md) section 5; 23 seams, 22 folded, 1 changed):

| Seam | Lane |
|---|---|
| S-1..S-4 engine contracts, build profile and stamp, measured cost | 1A-13 (columns), 1F-3 (pydantic contracts), 5B-4 (writer), 5D-5 (builtin rows) |
| S-5 serving profile | 1A-13 (kind and `deployment_settings` key), 2C-7 (body and activation), 2E-8 (generated default) |
| S-6, S-7 GPU inventory and reservations by UUID | 1A-13 (tables), 1B-8 (writer at `worker_ready`), 5A-6 (training) |
| S-8 `JobPerfSummary`; S-9 the resolved plan | 3A-15; 02b section 2.2; filled by 4A, 4C and 5d |
| S-10 crop size-variant keys | 1E-4; used by 5c-K1 |
| S-11 no GPU-prep queue, `gpu_affinity` | 1B-8 (field), 3A-15 (wrapper) |
| S-12 GPU worker shape | **changed**: 1B-8 (`make_app(include, device_uuid=None)`, CUDA only after fork); queue derivation deferred to RW-1 (5.5) |
| S-13 export records and refuses on mismatch | 5B-4 |
| S-14 serving inventory | 1F-1 (table), 6A (beat task) |
| S-15 router; S-23 its gauge | B4 4B-6 |
| S-16 Triton per card in compose | 6B-7 |
| S-17 one geometry module | 5b-G1 (07b amendment) |
| S-18..S-20 metrics | 5b-D1, 5b (07b amendment) |
| S-21 plan branch counter | 5d-P2 (07d amendment) |
| S-22 DCGM profiling fields | 6B-8 (RD-18, answered as recommended) |

**Clustering** ([09b_research_clustering.md](../09b_research_clustering.md) section 5.6; 11 seams, 9 folded, 1 changed, 1 reserved):

| Seam | Lane |
|---|---|
| S-core tables and retention | 1F-1 ([02d](../02d_research_seam_schema.md) 2.2), 02a section 9 |
| `project_state` keys removed | **changed**: 1A-13 drops the region markers and `last_clustering`; `cluster_refresh` kept (X-11) |
| queue `cluster`, `cluster-worker` | 1B-8, 1C-8 |
| kinds `cluster_run`, `cluster_projection_sync`; `region_cluster` removed; IVF retrain out of `autolabel` | 4F-2, 4F-3; 4D-1 |
| persons as a partition | 4F-9, was 4C-7 (D-84; the plan's smaller option: `persons_cluster` keeps its name and records a `union_find` partition) |
| beat `cluster_refresh` starts `cluster_run` | 6A |
| OpenSearch fields, projection, ratchet | 4F-3, 4F-6; `tests/test_no_cluster_id_namespace.py` |
| `cluster_profile` kind and axes | 1A-13, 2C-7, 4F-7 |
| routes and wire (S-core) | 4F-6, contracts; 4E renders |
| S-later tables and routes | **reserved** (02d section 5), RW-3 lanes L4-L6 |
| the loop's partition reader | 4D-7 |

**Embeddings and search** ([09a_research_embeddings_search.md](../09a_research_embeddings_search.md) section 5; 8 seams, all folded):

| Seam | Lane |
|---|---|
| 5.1 vocabulary | ES-A (`src/domain/embedding_spaces.py`) |
| 5.2 tables and default rows | 1F-1; 2A-7 (create), 2E-8 (upgrade), ES-A1 (catalog) |
| 5.3 field naming and versioning | ES-A2, ES-A3 |
| 5.4 readers resolve their field | ES-A4 `embedding_space()` (W2.2); each W2.3a lane in its own files; ES-C1, ES-C2 (W2.4) for the rest; 4E (W2.4, the former `b2/frontend-spaces`) |
| 5.5 the re-index job and the flip | ES-B1, ES-B3; ES-A2 (flip) |
| 5.6 `project_settings.search` and the catalog | 1A-13; ES-A1 |
| 5.7 routes and search wire | ES-B2, ES-C3 |
| ES-E in-process text encoders deleted | B2 ES-C5 (W2.4; was B4 4B-7, D-84) |

**Near-duplicates** ([09d_research_near_duplicates.md](../09d_research_near_duplicates.md) section 5; 7 seams, all folded, 2 with changes):

| Seam | Lane |
|---|---|
| 5.1 tables | ND-A (`0003_near_duplicates`, **changed**: denormalized project on groups and members, `space_key` values, X-1) |
| 5.2 job kinds | ND-A (ND-4, ND-5); ND-C1 (`near_dup_core_group`, W4); 6A (beat ticks) |
| 5.3 OpenSearch fields | ND-A (ND-5, via 2F-8's `near_dup_fields.py`); ND-B5, ND-B6 (the item wire and the hide filter, after 4F); ND-C3 (core rename) |
| 5.4 the holdout rule | ND-B1..ND-B3 (replaces 2D-4) |
| 5.5 `near_dup_profile` | ND-A (ND-2) |
| 5.6 routes and wire; core ingest candidates | ND-A (ND-6, ND-7); ND-C2; 07d amendment |
| 7.2 exact key | ND-A (ND-8), ND-C4 (RD-43, answered as recommended) |

**Spatial attributes** ([09f_research_spatial_attributes.md](../09f_research_spatial_attributes.md) section 5; 13 seams: 11 folded, 2 reserved; `location_policy` conditional on RD-51):

| Seam | Lane |
|---|---|
| (A) capture metadata at ingest, and the coordinates under `location_policy` (RD-51 reversed) | W4 `b4/capture-metadata` CM-1..CM-4 |
| (B) `jobs.kind` without a `CHECK` | already in 02a (kept) |
| (B) images `index.knn = true` door | 2F-7 |
| (B) delete removes an unlisted area | 1E-5 |
| (B) `op_scope_allows` for child tables | already in 02a (kept) |
| (B) additive, versioned mapping registry | 2F-8 (its first post-baseline user is `capture_fields.py`) |
| (B) fixed region-box keys in one dataclass | kept; `tests/test_no_legacy_region_scalars.py` |
| (B) explicit wire fields; coordinates only in the image `capture` block under the location policy | CM-2 (the location canary), CM-4 |
| (B) queue consumer test covers profiles | already in 02b section 7.3 (kept) |
| L-X1 export orientation | 5b-D1 (07b amendment) |
| L-X2 export GPS | exports keep the embedded GPS by default (owner, RD-54); a `drop`-policy project's exports strip it (CM-5, review RN-4) |
| (C) attribute signals | **reserved** (02d section 5), RW-5 L-A1 |
| (C) scenes and reconstructions | **reserved** (02d section 5), RW-5 L-SP1 |

**Registration-plate text** ([09h_research_plate_text.md](../09h_research_plate_text.md) section 5; 11 seams: 10 folded, 1 moved):

| Seam | Lane |
|---|---|
| 5.1 per-box fields | W5e `w5e/region-text-fields` PT-F1, PT-F4 |
| 5.1 index analysis settings (creation time) | 2F-6 (with the upgraded-index close and open, X-4) |
| 5.2 tables | W5e store lane PT-S1 (`0004_region_text`, approved, RD-4) |
| 5.3 job kinds and the `text_read` queue | W5e jobs lane PT-J1..PT-J3 |
| 5.4 queries | PT-F3 |
| 5.5 `region_profile` fields | **moved** from 2C to the W5e fields lane (5.5) |
| 5.5 environment | dropped 2026-11-05 (D-79; was B4 `b4/identity-text-privacy`) |
| 5.5 secrets (creation time) | 1C-9 |
| 5.6 routes and wire | PT-F4, PT-J1, PT-M1 |
| 5.7 `rec_char_min` | W5c track `ocr` (07c amendment) |
| 5.8 privacy baseline | dropped 2026-11-05 (D-79; was `b4/identity-text-privacy`); `SECURITY.md` by `b4/privacy-hardening` (X-17) |

**The auto-label assistant** ([09e_research_auto_label_assist.md](../09e_research_auto_label_assist.md) section 5; 9 seams, all folded):

| Seam | Lane |
|---|---|
| 5.1 loop tables | 1F-1 (02d 2.4); retention rows in 02a section 9 |
| 5.2 `retry_count`, `pause_reason`, `yielded`, transitions | 1A-13 (columns and the transition table), 3A-12..3A-14 |
| 5.2 kind fields (`pausable`, `slice_s`, limits) | 1B-8 (fields), 4D (values) |
| 5.3 control API and handler semantics | 4D-4, 4D-6 |
| 5.4 event types | 4D-7; 02b section 6.2 |
| 5.5 budgets and `vlm_usage_hourly` | 4D-5; 4A-8 and the other VLM callers (X-18) |
| 5.6 lock rule and label-operation kinds | 2D-5; 4D red tests |
| 5.7 `assist_loop_preset` | 1A-13, 2C-7 |
| 5.8 the UI wire and Cropwright | 4D-4 (`AssistLoopControl`), 4E |

Total: 82 seams; 76 folded (two of the near-duplicate ones with the X-1 table changes), 3 changed or moved (5.5), 3 reserved by design.

### 5.3 Research issues: where each lands

The rows of section 2 above are updated for #40, #42, #45, #67, #84, #219, #220, #231, #268, #346,
#347, #348, #365, #398, #400, #401, #403, #406, #407, #416, #435, #436, #466 and #473. In one table:

| Issue | v0.7.0 (wave and lane) | After the tag |
|---|---|---|
| #346 clustering | 1F (tables), W2.3a 4F | RW-3 (`clu:CL-0..CL-9`, L3-L6) |
| #347 encoder evaluation | ES-A..ES-C (the registry it lands on) | RW-2 `emb:EP-4` |
| #348 spatial | W4 `b4/capture-metadata` | RW-5 (`sp:P-4..P-9`, L-SP1 on a GO) |
| #365 embedding swap | ES-A (W2.2), ES-B (W2.3a), ES-C (W2.4) | RW-2, v0.8.0 F9 reduced |
| #398 Z5 roots | the partition seam (1F, 4F) and the loop's partition reader (4D) | RW-6 `al:AL6`, `feat/loop-z5i` |
| #400 attribute signals | none (reserved DDL) | RW-5 L-A1 |
| #401 plate text | 1C-9, 2F-6, W5c `rec_char_min`, **W5e** | - |
| #406 judge calibration | none | RW-3 `clu:CL-1` |
| #416 near-duplicates | ND-A, ND-B (W2.3a), ND-C (W4); `nd:P0..P2` in PW-1 | RW-4 (`nd:P3..P7`, ND-E) |
| #435 frame target | none | RW-2 `emb:EP-2` decides |
| #436 `group_by=image` | the `space` parameter it composes with (ES-C3) | v0.8.0 F2 |
| #466 loop pause, events, budgets | W2.3a 4D-2..4D-7, W2.4 4E | RW-6, v0.8.0 `feat/loop-review` |
| #473 near-duplicate data quality | part a ND-A, part b ND-C1, part c ND-B1 | - |
| #84 wheel example | none | RW-6 `al:AL5`, v0.8.0 F8 |
| #40 decode once | W5b (reads `gpu:PR-1`, `PR-2`; L-X1 in 5b-D1) | RW-1 (`PR-3`, `PR-4`) |
| #407 GPU-prep speed | W5b (reads `gpu:PR-2`; 5b-G4 not opened) | RW-1 (`PR-4`) |
| #219 crops at model size | W5c (S-10 key from 1E-4) | - |
| #220 OCR and faces | W5c (`gpu:PR-6`, `rec_char_min`), W5d (`gpu:PR-5`) | - |
| #268 Triton follow-ups | W5a | RW-1 remainder |
| #403 MPS | off in v0.7.0 | RW-1 (`PR-4`) |
| #231 serving head-to-head | GPU window 2, its own rule | RW-1 if no room |
| #42 12 GB card | side track; `build_profile = 'small'` | - |
| #45 ladders | W5a (reads `JobPerfSummary`) | - |
| #67 processing plans | W5d (S-9, S-21) | - |

### 5.4 Research findings put into lanes (each with a red test)

| Finding | Source | Lane and red test |
|---|---|---|
| lane 2D-4 cannot work: today's crop groups never contain a validated or held-out item, so a freeze that reads the group fields moves nothing | near-duplicates plan C8 | moved to ND-B1 (W2.3a since D-84): `tests/near_duplicates/test_holdout_near_duplicates.py` (a validated item whose near-duplicate crop is in the cohort's leak group ends held out and counted) |
| G-1: pause and resume spend attempts (the claim refuses at `attempt >= 3`) | auto-label plan section 2 | 3A-12: `tests/db/test_pause_resume_cycles.py`; end to end `tests/curation/test_assist_loop_pause_cycles.py` (4D) |
| G-2: one long loop holds the only `assist_loop` consumer and starves other projects | auto-label plan section 2 | 3A-13 (`Yield`, slices): `tests/tasks/test_two_project_slices.py`; 4D-6: `tests/curation/test_assist_loop_two_projects.py` |
| G-3: the loop's durable record cannot live in 24-hour events or in `jobs.params` | auto-label plan section 2 | 1F-1 (the loop tables), 4D-2: `tests/curation/test_assist_loop_record_in_tables.py`, `test_assist_loop_budget_sql.py` |
| exports ignore EXIF orientation 2-8 on the resize and crop path | spatial plan section 2.1 | 5b-D1 (L-X1): `tests/curation/test_export_orientation.py` |
| byte-copy exports keep embedded GPS | spatial plan section 9 | kept as today by default (RD-54, 2026-11-04); stripped only for `drop`-policy projects, W4 CM-5 (review RN-4): `tests/curation/test_export_gps_by_policy.py`; RF-84 checks both |
| the in-process text encoders are dead code under D-69 | embeddings plan ES-E | B2 ES-C5 (was B4 4B-7, D-84): `tests/curation/test_search_requires_triton_text.py`; retired variables refuse startup |
| `cluster_id` is overloaded (index assignment, class id below 10000, residual offset, review state) | clustering plan section 1 | 4F-6 splits it into the assigner, the versioned partition and the review state: `tests/test_no_cluster_id_namespace.py`; 4F-2 `tests/clustering/test_carry_over.py` (ARI 1.000 on permuted input, 0.306 today) |
| a `RESTRICT` foreign key inside one project's cascade can fail the project delete | found by this integration pass | 02a section 1.2: `tests/db/test_profile_fk_cascade.py` (1F-4) |
| the plate normalizer must exist on upgraded indexes too (analysis settings are static) | found by this integration pass | 2F-6, kept by the owner (Q-A) and made safe and bounded (review RR-8: per index, dry run, progress, `try/finally` reopen, stated downtime): `tests/opensearch/test_items_text_normalizer.py`, `tests/opensearch/test_analysis_settings_failure.py` |
| a deployment-wide serving profile cannot use the project-scoped activation table | found by this integration pass | 2C-7: `deployment_settings('active_serving_profile')` with `expected_revision`; `tests/config_store/test_research_kinds.py` |
| a composite foreign key with a NULL `project_id` checks nothing, so core-face memberships could dangle | found by this integration pass | 1F-4: `tests/db/test_cluster_memberships_fk_null_project.py` |

### 5.5 Seams changed, moved or not taken (with the reason)

| Seam | Decision | Reason |
|---|---|---|
| GPU S-12: device-scoped queue names derived from `gpu_devices` | not in 0.7.0; `make_app` takes the UUID and records it | no family is device-scoped in 0.7.0, so derivation code would have no reader (no dead code); RW-1's two-card lane adds it |
| GPU S-6: `openprocessor gpu plan` as a second writer of `gpu_devices` (also review RR-9's suggestion) | not taken; the writers are the GPU workers (every NVML-visible card at start, `seen_at` on every heartbeat), `op.periodic.serving_inventory` (every card Triton reports) and the reservation insert (its own card, same transaction); `gpu plan` reads | the CLI is bash and would need its own database client; the three Python writers cover RR-9's two gaps (a long-lived worker, a card with no worker) |
| plate 5.5: "room in the `region_profile` body model in 2C now" | moved to the W5e fields lane | fields with no reader until W5e would be dead code in W2.2; W5e owns the model then |
| plate J13: a separate `captured_at` field | replaced by the W4 capture fields (X-16, RD-68) | one owner per value |
| clustering 11: drop `project_state('cluster_refresh')` | kept as the beat's clock (X-11) | deferral and backoff need a per-project clock no partition row holds |
| near-duplicates 5.1: `EXISTS` RLS on groups and members | denormalized `project_id` (X-1) | members grow with every image and item per run; the constant-InitPlan policy keeps a 1M-row `COPY` cheap |
| auto-label 5.1 and plate 5.2: profile references `ON DELETE SET NULL` to `config_documents` | composite reference to `config_revisions`, `NO ACTION` (X-1, X-1b) | a run must name the exact revision that drove it, and one project delete must pass |

### 5.6 The final pass: the research review and the owner answers (2026-11-04)

The independent review [review_2_research.md](review_2_research.md) found RR-1..RR-24; every
finding is answered in its "Planner responses" section (fixed with the file, lane and test, or
rebutted with the reason). The owner's answers of the same day changed the plan as follows.

| Change | Where | Test or check |
|---|---|---|
| W2.3 split into W2.3a (5D, 4D, 4A, 4B, 4C, 5A, 5B, 5C: ports of today's behaviour) and W2.3b (4F, ND-A, ES-B, then ND-B, then 4E) (RR-13, Q-A) | [03d](../03d_stage4_job_families.md) header and sections 2-8; [00_master_plan.md](../00_master_plan.md) W2.3a and W2.3b | each part closes with its own part check; every W2.3b cross-lane call is against a signature fixed in 03b. **Superseded on 2026-11-05 by D-84** (section 5.7): W2.3b folded into W2.3a |
| the research pieces W2.3a lanes used to call moved to W2.3b: the `near_dup` scorer deletion and every `enqueue_pending` call (ND-A, was 4A-6, 4B-4, 4C), person grouping as a partition (4F-9, was 4C-7), the IVF retrain and the loop's cluster reader (4F-10, was 4D-1, 4D-7) | 03b | `tests/curation/test_no_near_dup_scorer.py`, `tests/test_persons_cluster_partition.py`, `tests/curation/test_assist_loop_reads_partition.py`. **Superseded on 2026-11-05 by D-84** (section 5.7): nothing is ported for a later lane to replace |
| every seam kept, the plate normalizer included; its upgrade step safe and bounded (Q-A, RR-8, RR-12 rebutted) | [02c](../02c_migration_ops_tests.md) section 2.2; [03b](../03b_stage2_control_data.md) 2F-6 | `tests/opensearch/test_analysis_settings_failure.py` |
| GPS coordinates stored now under `project_settings.location_policy` (default `store`; RD-51 reversed); exports keep the embedded GPS (RD-54) except for `drop`-policy projects (CM-5, review RN-4, ExifTool in the jobs image for that path only) | [02a](../02a_postgres_schema.md) section 4; [02d](../02d_research_seam_schema.md) section 1; [05](../05_b4_stateless_api.md) CM-2..CM-4; [07](../07_release_acceptance.md) RF-84; [08](environment_and_setup.md); [10](standards_and_libraries.md) | `tests/curation/test_location_policy.py`; the location canary |
| short tiered prototype runs, about 10 GPU hours before the tag (RD-2, RD-7); cuts: `gpu:PR-3`, `PR-4` (5b-G4 not opened), `PR-7..PR-9`, `pt:P6`, the SSCD arm of `nd:P2`, `nd:P3..P7`, every `emb:`/`clu:` GPU run, `sp:P-2..P-9`, `al:AL1..AL6` | [08_research_prototype_track.md](../08_research_prototype_track.md) sections 2-4; [06b](../06b_b5_decode_once.md) 5b-G4; [11](risks_cleanup_host.md) rules 9, 10 | the coordinator's GPU ledger |
| datasets: one register; UFPR-ALPR, LPLC and RodoSol skipped (RD-10, Q-C, RR-11) | 00e section 4.1 | each prototype's registration commit records the terms |
| PP-OCRv6 as one short reader comparison against PP-OCRv5 (Q-B, RR-5) | 00e `pt:P1`; [06c](../06c_b5_crops_ocr_faces.md); [06e](../06e_w5e_plate_text.md); 10 | `pt:P1`'s registered bar |
| the `cluster` queue and worker decided (RD-25, Q-D) | [03a](../03a_stage1_foundation.md) 1B-8, 1C-8 | `tests/tasks/test_planned_modules_included.py` |
| owner answers recorded | [00d](owner_questions.md) section 0; [decision_register.md](decision_register.md) section 6 | 69 questions stay open; none blocks coding |

### 5.7 The no-rework pass (2026-11-05, owner rule D-84)

The owner's rule: code is not written and then rewritten by a later wave. The full inventory with each
resolution is [00_master_plan.md](../00_master_plan.md) section 3a; the lane and wave changes it made:

| Change | Was | Now | Test or check |
|---|---|---|---|
| W2.3b folded into W2.3a | 4F, ND-A, ES-B, ND-B in W2.3b; 4E in W2.3b | 4F, ND-A, ES-B, ND-B in W2.3a; 4E in W2.4 | the W2.3a part check |
| the `near_dup` scorer | ported by 4A, deleted by ND-A (ND-9) | deleted by 4A (4A-6), never ported | `tests/curation/test_no_near_dup_scorer.py` |
| the IVF retrain check | ported by 4D into `ivf_check.py`, deleted by 4F-10 | deleted with the worker by 4D | the 4D done-when `rg` |
| the loop's cluster reader | ported by 4D, replaced by 4F-10 | written partition-backed by 4F-10 in W2.3a | `tests/curation/test_assist_loop_reads_partition.py` |
| person grouping | 4C-7, then 4F-9 | 4F-9 only | `tests/test_persons_cluster_job.py`, `tests/test_persons_cluster_partition.py` |
| `enqueue_pending` calls in 4A's and 4B's files | ND-A (ND-3) a wave later | 4A (4A-9) and 4B (4B-3, 4B-4), ND-A merging first | `tests/curation/test_reprocess_enqueues_near_dup.py` |
| `embedding_space()` | ES-C (W2.4) | ES-A4 (W2.2) | `tests/embedding_spaces/test_resolve.py` |
| stage 6 | 6A and 6B each in two parts (PR2, PR3) | 6A and 6B once each in W2.3a (PR2) | `tests/periodic/test_beat_schedule.py` lists every tick at PR2 |
| public container hardening | 6C (W2.4) | 1C-9 (W2.1) and 6B-9 (W2.3a) | `tests/compose/test_new_service_hardening.py` |
| Cropwright in B2 | 4E and `b2/frontend-spaces` | 4E only (W2.4) | the 4E unit tests |
| `/health` sequential | B4 4A-5 | B2 1D-1 | `tests/test_health_no_fanout.py` |
| semantic search `_msearch` | B4 4A-5 | B2 ES-C4 | `tests/curation/test_semantic_search_msearch.py` |
| in-process text encoders | B4 4B-7 | B2 ES-C5 | `tests/curation/test_search_requires_triton_text.py` |
| the API database pool rule | B2 1A-6, folded into `api_pools.py` by B4 4B-2 | B2 1A-6 in `api_pools.py`; 4B-2 adds the other totals | `tests/config/test_api_pools.py` |
| the upgrade load on the repositories | 2E on the ORM, switched by 2F-5 | 2E on the repositories; 2F-5 gone | the 2E AST test |
| the bulk-ingest launch | 4C inside the loop, 6A replaced the loop | 4C `dispatch_once` (4C-1a), called by 6A's task | `tests/curation/test_dispatcher_starts_poll_job.py` |
| CHANGELOG entries | each PR, then rewritten by 6D in PR3 | each PR once; 6D owns no root CHANGELOG | CI's changelog job per PR |
| VLM batch routes (D-29a) | B4 4A-0, 4A-3 | B2 4D-8 if D-29a is answered before W2.3a; else B4 | `tests/curation/test_vlm_batch_routes_enqueue.py` |
| B3 timing | after PR3 | lanes and GPU window 1 after PR2; the PR may merge after PR2 | B3 P1 |
| B4 track 4A timing | after B2 and B3 | codes from `main` after PR2 | master section 4.0b |
| typed responses (#465) | lanes in W5a-W5c | filler lanes on routers no later wave rewrites | the per-router schema tests |
| PR2 live smoke | steps 3-7 and 12-18 (3 h 15 min) | steps 3-7 and 12 (1 h 40 min); PR3 runs every step | 03 section 9 |
