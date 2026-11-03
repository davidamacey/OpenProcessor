# OpenProcessor master plan: finish 0.7.0, then correctness, jobs, models, speed

Status: plan only, revision 4 (2026-11-03). Committed with its three source plans on the owner's request. Nothing in
it is implemented yet; batches start only on the owner's go. No test or stack was run for the plan itself and no issue was edited.
Revision 2 applies the fresh review `artifacts_local/master_plan_review.md` in full. Appendix A
records what happened to each review item, and section 18 is the revision log.

**Mission (every batch is checked against it):** OpenProcessor processes our images **fast and
correctly** for the task at hand, for many users and many processes at once. That means
inference on Triton (detect, faces, embed, OCR, search) plus curation (ingest, regions, VLM
labeling, review, training). Correct comes first, under concurrency too. Speed is measured
once the stack it is measured on stops changing.

## 0. How to use this file

A fresh implementation agent executes **one batch** from this file **plus** the task text in
the source plans. Task IDs below point into those plans:

| Prefix | Source plan | What the IDs are |
|---|---|---|
| `Q-P1.x`, `Q-Phase N`, `Q-Jn`, `Q-Dn`, `Q-En` | `docs/design/task_queue_plan.md` | Celery phases, judgment calls, defects (D1-D15), edge cases (E1-E26) |
| `BA-`, `OS-`, `GPU-`, `FE-`, `T-`, `OPS-`, `CC-`, `SEC-` | `docs/design/codebase_audit_plan.md` | audit tasks (section 4 there has the fix, the traps, the red test and the control) |
| `U-Step n`, `U-Jn`, `U-Cn`, `U-Rn`, `U-6.1`, `U-6.2` | `docs/design/model_family_versions_plan.md` | Ultralytics migration steps, judgment calls, corrections, risks, test matrix |
| `D-n` | this file, section 8 | owner decisions (renumbered in revision 2) |
| `N-n` | this file, section 9 | GitHub issues to create (titles and briefs) |

**Path legend (the public tree carries no machine paths).**
- `<repo>` is the OpenProcessor checkout. `<repo>-wt-<name>` is a sibling git worktree.
- `<scratch>` is the host's scratch directory for acceptance installs.
- `artifacts_local/` is the gitignored evidence directory, on the owner's machine only. It holds the ledger, the audit
  findings and verdicts (the security details stay there), the Ultralytics evidence (`artifacts_local/model_versions/`:
  matrix, run logs and the probe scripts; regenerate the ONNX files with those scripts) and the R1/R2 reports.
- `<scratch repro>` marks a throwaway script an audit or review used. It is not kept; each task's red test recreates the scenario.
- **Line references** (`file:line`) in the three source plans are against `main` at `5b95591f`. The pinned audit tree
  was removed, so recreate it with `git worktree add --detach .claude/worktrees/audit-r2 5b95591f`. Code that the
  integration branch changed has moved; find it by the symbol name, not the line.

**Where a source plan and this file disagree, this file wins.** This file replaces the Celery
plan's Phase 1 locks (3a), the B1 split of the Ultralytics steps (D-10), the release train
(section 4) and the lease and registry designs (section 6).

**Branch facts (do not hard-code SHAs or commit counts; they move).**
- The integration branch is `fix/411-r2-round` (worktree `<repo>-wt-r2round`,
  based on main `5b95591f`, not pushed). Refer to it as "**fix/411-r2-round after B0 small
  fixes**". Read its head with `git -C <repo>-wt-r2round log -1 --oneline`.
- **B0 small fixes are in progress now.** GPU-1, GPU-3a/13 and GPU-2 are on `b0/ocr-fixes`, which
  is already merged into the integration branch. OS-2, OS-1 and OPS-1 are committed on
  `b0/data-ops-fixes`. CC-1 and BA-7 are still being implemented on `b0/data-ops-fixes`.
- The stale "bootstrap lock ... kept per loop" CHANGELOG bullet is **already removed** (commit
  "docs: drop the superseded per-loop lock changelog entry"). There is no step for it.
- **VERSION is still `0.6.0`.** It is bumped in the B0 release-prep commit (4.1), not in the fix
  branches.

**Inputs read for revision 2:**
- the review, `artifacts_local/LEDGER.md`, `docs/VISION_AND_GOALS.md`, the three source plans,
  `artifacts_local/audit/*`;
- `gh issue list --state open --limit 100` (37 open issues) and the milestones;
- code checks, read only, on the branch worktree:
  - `file_job.py`, `config/projects.py` (`resources_for_new`), `bodies_other.py` (`_configs_body`,
    `_settings_body`), `guard.py`, `occ.py`;
  - every `exclusive_start_lock`/`flock` call site, the `openprocessor upgrade` path, the trainer
    Dockerfile pin;
- package versions read from the local `0.7.0-rc5` images with `importlib.metadata`: API and export
  images `ultralytics 8.4.197` and `uvicorn 0.54.0`, `/opt/venv-y11` `ultralytics 8.3.253`;
- uvicorn release 0.41.0 notes ("Add `--limit-max-requests-jitter` to stagger worker restarts
  (#2707)", read through `gh api`);
- the Ultralytics export page (`nms` semantics, and the maintainer comment that `end2end` is
  deprecated and mapped to `nms`).

---

## 1. TL;DR

| Batch | Name | Release | Size | Depends on | One-line scope |
|---|---|---|---|---|---|
| **B0** | Finish 0.7.0 | **v0.7.0** | M | owner yes; host all-clear | finish the in-progress small fixes on `fix/411-r2-round` (GPU-1, GPU-3a+13, GPU-2, OS-2, OS-1, OPS-1 with a uvicorn floor, CC-1, BA-7); add the exact Ultralytics pin of what the images resolve today (`==8.4.197`, D-2); release prep (VERSION, CHANGELOG); one gate cycle; push, CI, merge; **R3** on merged main; tag on the owner's word |
| **B1** | Correctness foundation (5 tracks, 1 PR) | **v0.8.0** (early, D-7) | XL (tracks L, L, M, M, S) | B0 merged; D-8..D-12 | **1A** OpenSearch job leases (thread heartbeat, TTL 120 s, cooperative fencing) replace every start lock; trainer claims by generation; OS-3. **1B** class registry document with OCC, plus an upgrade migration that fails closed. **1C** inference and search hygiene. **1D** API, frontend and test integrity, SEC-3, SEC-6. **1E** minimal live Ultralytics fixes (#447 export arg and head check, upload routing by head kind, class by name on the secondary detector, YOLOv5 ONNX probe input) |
| **R-B1** | 0.8.0 release acceptance | v0.8.0 | M | B1 merged | acceptance subset including `installer_upgrade` from 0.6.0 and 0.7.0 with the registry migration; tag on the owner's word |
| **B2** | Jobs off the API (Celery) | v0.9.0 | XL (3 waves) | B1 (lease); D-17..D-26 | `Q-Phase 2` infra, `Q-Phase 3` all 11 families, `Q-Phase 4` beat and the lease reaper, `Q-Phase 5` trigger workers; OPS-2; SEC-4, SEC-5 |
| **B3** | One Ultralytics version, one detection contract | v0.9.0 | L + about 1 GPU day | B1 merged; D-27, D-28 | U-Step 4-9 (Stage A and B judged in one GPU gate), the trainer's `nms` by capability and its pin, the full U-6.1 matrix in CI, GPU-14 final, GPU-18 docs, U-C11, the first #368 slice |
| **B4** | Stateless API | v0.9.0 | L | B2, B3 | `Q-Phase 6` (bulk OCC, J7 exemption, CC-7, CC-10), `Q-Phase 7` (`API_WORKERS`, benchmarked against a 32-worker baseline **retaken on the B3 engine**), Redis Streams for events after a spike, #439, the last file lock deleted; then **R-0.9** full release acceptance |
| **B5** | Speed (mission) | v0.10.0 | XL, 4 sub-PRs | B4 merged | 5a #45 baseline + #268 sweeps + GPU-6 recall; 5b GPU-3b one decode, then #40/#407 (+GPU-12); 5c #219 + #220 (+GPU-7, GPU-19); 5d #67 processing plans |
| side | #42 small-GPU acceptance | note in v0.7.0 / v0.10.0 | S (GPU time) | GPU 1 free | run whenever the OpenTranscribe session frees GPU 1; never blocks a batch |

**Release train (D-7, recommended):**
- v0.7.0 = B0.
- **v0.8.0 = B1**, tagged early so the registry race fix, the leases and about 30 LIVE audit fixes
  reach users without waiting for Celery.
- v0.9.0 = B2 + B3 + B4.
- v0.10.0 = B5.
- Existing milestones are renamed to fit (section 9.3); the renames carry their issues.

There is one full test cycle per batch (B5: one per sub-PR, because each sub-PR is gated on its
own measurement). Release acceptance runs three times: **R3** (0.7.0), **R-B1** (0.8.0, a subset
plus the upgrade path) and **R-0.9** (full).

---

## 2. Mission check

| Batch | Correct | Fast | Why now |
|---|---|---|---|
| B0 | `/ocr` boxes land on the text (GPU-1, high) and in the reported frame (GPU-3a, high); DB unclip is real (GPU-2); a holdout freeze never claims an unflagged item (OS-2, protects the lock rule); status polls stop answering 500 (CC-1); concurrent first config writes conflict instead of both winning (OS-1); an outage stops reading as "not found" (BA-7); the Ultralytics version stops floating between builds (exact pin) | the jitter (OPS-1) stops 32 workers recycling in lockstep under ingest load | high-severity live output bugs would otherwise ship in 0.7.0; every item is size S |
| B1 | duplicate class ids and lost classes end (CC-2); two POSTs can no longer start one singleton twice (Q-D2); a sibling's live job is never flipped (Q-D4/CC-3); a blocked loop no longer costs a job its liveness (Q-D9, thread heartbeat); per-worker caches stop serving stale vocabularies (BA-1); a YOLO26 upload stops building a wrong engine (U-C2, #447); class by name on the secondary detector (U-C10) | inference hygiene removes multi-GB allocations (GPU-5, GPU-9) and retries of deterministic errors (GPU-8); the kNN filter moves inside the ANN query (OS-6), so filtered search returns `k` hits | the lease is the claim the Celery wrapper needs (no churn); the registry race is the biggest open multi-user bug; shipping it as 0.8.0 gets it to users months earlier |
| B2 | jobs survive API worker recycling (CC-4, reproduced); periodic side effects run once, not 32 times (Q-D5); claims hold with more than one worker (Q-D7/D8) | heavy work leaves the request processes, so request latency stops competing with jobs | the largest live correctness gap after the registry |
| B3 | "the right model with the right code": one Ultralytics version, one fused `[b,300,6]` contract, no monkey-patched fork, no deprecated TensorRT plugin | must hold at least 95 % of the current detector throughput (U-6.2 step 4), or Stage A ships alone | after B1 fixed the live bugs; before any speed baseline, because Stage B changes the default detector engine |
| B4 | the API gives the same answers with 1 worker and with 32 (Q-section 4e); request fan-out becomes bulk OCC | the worker count becomes a tuning knob sized to cores, benchmarked on the final engine | needs the jobs moved (B2) and the final engine (B3), so it is benchmarked once |
| B5 | parity gates on every change | the GPU pipeline, crops at model size, OCR/faces on resident crops, processing plans | measured only on the final stack, so no baseline is taken twice |

**Deferred, and why:**
- **Performance** (#40, #407, #219, #220, #67, #268, #45) waits for B4. Celery moves load out of
  the API, `API_WORKERS` changes the per-process pools, and Stage B changes the detector engine,
  so an earlier baseline would be thrown away. The standing rule is "#45 baseline before any
  optimization code". None of these is a live correctness bug.
- **GPU-6** (`/detect` upscaling) is an owner decision that needs a recall measurement; it goes
  with the B5 measurement tooling.
- **BA-6** (71 untyped operations): one router per PR after B4, so the routers that B2 and B4
  rewrite are typed once.
- **Evidence-first items** act only on a measured defect, as the audit plan says:
  - OS-8, GPU-19, FE-6 and T-6;
  - OPS-2, measured before B2;
  - CC-10, measured before B4.
- **Refuted and cut audit items stay refuted and cut** (Appendix B lists them: OS-4, GPU-11,
  FE-4, the TESTS-* cuts, and the asyncio R6/R9/R10 claims, with every Appendix D proposal).
  Do not redo them.
- **#369 API authentication** stays blocked. B2 adds no public surface, because the broker has no
  published port.
- **Research and platform-breadth issues** are out of scope for this plan; section 10 says why for
  each:
  - #8 video, #346 clustering research, #348 spatial;
  - #365 embedding swap, #400 attribute signals, #401 plate identity;
  - #347, #398, #406, #416, #366, #367, #370, #84, #402, #435, #436, #231, #403.

---

## 3. Owner rules applied

1. **Docs and maintained libraries first.** Every mechanism cites its doc (section 17):
   - Celery for transport and scheduling;
   - OpenSearch OCC for claims and the registry;
   - Python `tempfile`/`os.replace` for atomic files;
   - `asyncio.TaskGroup` for bounded concurrency;
   - Ultralytics' documented `nms` export argument instead of the vendored patch;
   - Triton's `async_exec` instead of hand batching;
   - the existing `?ids=` route instead of a new one.

   The lease TTL and heartbeat numbers follow OpenTranscribe's measured thread-heartbeat lease
   (about 90 s TTL, `task_queue_plan.md` section 4d), rounded up to 120 s (6.1).
2. **No in-process locks and no file locks.**
   - The Celery plan's Phase 1 prescribed `flock` in three places; 3a replaces each one.
   - Today's tree has **18 files with 51 `flock`-family hits**. Section 3b schedules every one for
     removal.
   - A shrink-only ratchet stops new ones.
   - The trainer claim uses create-only generation files (6.3), never unlink-then-create.
3. **No asyncio fan-out in request handlers.**
   - Heavy work goes to Celery (B2).
   - The two handler fan-outs become one `mget` plus one bulk with per-item OCC (B4).
   - The ingest data plane keeps request-scoped parallelism only as a documented `TaskGroup`
     exemption (Q-J7, D-29).
   - The API ends stateless (B4).
4. **Ultralytics.**
   - B0 pins exactly what today's images resolve (`8.4.197`), so R3 tests the pin and nothing
     floats.
   - B1 fixes the live bugs on that version.
   - B3 moves to one newer version once, after U-6.1 (CPU) and U-6.2 (GPU) pass.
   - Original YOLOv5 is bring-your-own ONNX (U-J3 option b), with real-ONNX tests.
5. **No churn.** Nothing in the request path is fixed if B2 moves it, unless it is a live bug that
   ships before B2:
   - CC-3's mtime heuristic is cut; the lease fixes it properly in B1.
   - P1.6's auto-label and assist-loop `claim_trigger` rewrite is cut; B2 Phase 5 deletes those
     files.
   - GPU-14's interim clip is cut; B3 deletes the heuristic.
   - The lease and its **thread heartbeat** built in B1 are the same objects the B2 task wrapper
     uses.
   - B1 makes only the minimal export-code change that B3's single export module keeps (D-10).
   - The trainer changes only once, in B3.
   - GPU-3b lands as the decode helper #40 builds on.
6. **Tests.** Every fix has a red test seen failing on the unchanged base, plus a paired control
   that passes before and after (the audit plan gives both per task). PR descriptions record the
   red output.
7. **Security.** SEC items appear as neutral one-liners, here and in every public issue. Details
   stay in `artifacts_local/audit/security*.md`.
8. **Host:** see section 13.
9. **Release.** No tag without the owner's explicit word (v0.7.0, v0.8.0, v0.9.0, v0.10.0).
10. **Mission first.** Correctness under concurrency, then speed on a stable stack.

### 3a. Compliant replacement for the Celery plan's Phase 1

| Celery plan item | What it used | Replacement in this plan | Batch |
|---|---|---|---|
| Q-P1.0 atomic writes + `FileJob.write` under `_state_lock()` | `flock` | **CC-1**: `src/utils/atomic_write.py` (`NamedTemporaryFile(dir=parent, delete=False)` + `flush` + `fsync` + `os.replace`, temp unlinked on failure). **Readers stop writing in B0** (`get_state` of item scores, embedding viz and selection return the derived `interrupted` view). `repair_if_stale` stays the one persister until B1 (see the CC-1 row in 4.1). B1 makes persisting the job of the lease holder or of whoever takes over an expired lease (6.1), and B2 the job of the reaper. `_state_lock` is deleted in B1-1A | B0, B1-1A |
| Q-P1.1 `FileJob.claim` on `exclusive_start_lock` | `flock` | **Job lease document in OpenSearch** (J-LEASE, D-8, section 6.1): claim = `op_type=create`, or `index` with `if_seq_no`/`if_primary_term` over an expired or released lease; a 409 means `Busy` | B1-1A |
| Q-P1.2 unlocked starts (auto-label, selection, embedding viz, region clustering) | `flock` | the same lease, acquired before the job record is written; region clustering moves to `FileJob` and loses the 1800 s flag (Q-D12) | B1-1A |
| Q-P1.3 class registry under `exclusive_file_lock` | `flock` | **J-REG** (D-9, section 6.2): one registry document per project with OCC; `class_registry.json` becomes an export snapshot only | B1-1B |
| Q-P1.4 OS-1 | (compliant) | unchanged | B0 |
| Q-P1.5 reconcile treats a missing heartbeat as stale only if old | mtime heuristic | **cut**. A reconcile must take over the expired lease with OCC before writing, so a sibling's fresh lease is never touched. It is deleted with the reaper in B2 | B1-1A, B2 |
| Q-P1.6 worker claims | `O_EXCL`, `rename` | **Added, not kept: these claims do not exist today** (`rg O_EXCL docker/trainer src/services/training` finds nothing). Trainer job claim and API `training/jobs.py` create: create-only files (`O_CREAT|O_EXCL`). **A stale claim is reclaimed only by a create-only takeover of the next generation (6.3), never by unlink + create** (that is a check-then-act race between two reclaimers). `export_worker.reconcile_claimed`: fail only claimed jobs whose task heartbeat is stale. The auto-label/assist-loop `claim_trigger` rewrite is **cut** (B2 Phase 5 deletes it) | B1-1A |
| Q-P1.7 OS-3 | (compliant) | unchanged; uses the OS-2 helper from B0 | B1-1A |
| Celery exclusive starts (Q-section 5 rule 3, "under the job's start lock") | `flock` | the task wrapper **transfers the API-acquired lease** to itself (OCC) and keeps the same heartbeat thread class. A redelivered message finds a live lease and acks without running. Global serialization (open-vocab drain, auto-label, assist loop) is Celery single-consumer routing (`-c 1` queue); the lease still guards per-project exclusivity | B2 |

### 3b. Every `flock` site and when it goes

`rg -n "exclusive_start_lock|exclusive_file_lock|fcntl\.flock" src` on the branch finds **18
files, 51 hits** (including definitions and docstrings). B1-1A adds
`scripts/codegen/check_file_locks.py` with a shrink-only baseline in `scripts/codegen/baselines/`
(the same pattern as the existing ratchets; CONTRIBUTING.md). The baseline reaches zero in B4,
when `src/services/curation/job_lock.py` is deleted.

| Site | Purpose | Replacement | Batch |
|---|---|---|---|
| `reprocess_job.py`, `assist/job.py`, `assist_loop/job.py` (2), `dataset_import/runner.py` (3), `item_scores/job.py`, `probe_job.py`, `training/promote_job.py`, `projects/combine/service.py`, `bulk_ingest/runner.py`, `projects/legacy_repair.py`, `bulk_ingest/source_scan.py` | per-project singleton start | job lease (kind and scope table in 6.1) | B1-1A |
| `label_ops/store.py` `write_lock`, `undo_lock`, `_lock_is_held` (callers `label_ops/record.py`, `label_ops/undo.py`) | one writer per op; one undo per project | leases `label_op:<op_id>` and `label_undo` | B1-1A |
| `training/promote_json.py` (`exclusive_file_lock`, 2 calls) | read-modify-write of a JSON file | single writer: only the holder of the `promote` lease writes it, via `atomic_write_json`; the calls are deleted (the definition stays in `job_lock.py` until B4) | B1-1A |
| `file_job.py` `_state_lock` | `update()` and `repair_if_stale` read-modify-write | single-writer rule under the lease; deleted | B1-1A |
| `bulk_ingest/dispatcher.py` `dispatch.lock` | one dispatcher tick per project | Celery beat (exactly one scheduler) + a per-project lease for the tick | B2 (Q-Phase 4) |
| `open_vocab_sweeper.py` | one sweep at a time | beat task + `open_vocab` queue with `-c 1` | B2 (Q-Phase 4) |
| `source_image_cache.py` `.prune.lock` | one prune at a time | beat periodic task; the prune is idempotent (unlink tolerates a missing file) | B2 (Q-Phase 4) |
| `event_hub.py` append + rotate | JSONL event log shared by processes | Redis Streams (`XADD ... MAXLEN ~`, `XREAD BLOCK`) after the B4 spike (D-30) | B4 |
| `job_lock.py` (definitions) | the helpers themselves | deleted when the last caller goes | B4 |

### 3c. Celery-plan defects (Q-D1..D15) cross-reference

| Q-D | Defect | Where fixed |
|---|---|---|
| D1 | heavy jobs inside recyclable request workers | B2 (CC-4); interim: OPS-1 jitter (B0) + known-issue line (D-6) |
| D2 | singleton starts without a cross-process claim | B1-1A |
| D3 | class-registry lost update | B1-1B (CC-2) |
| D4 | startup reconcile assumes it is alone | B1-1A (lease takeover), deleted in B2 |
| D5 | GPU-arbiter loop runs in 32 processes | B2 W3 (2D) |
| D6 | serialization is per process (`_PASS_LOCK`) | B2 W2 (2C, `open_vocab` `-c 1`) |
| D7 | worker-container claims correct only with one replica | trainer and training jobs: B1-1A (6.3); auto-label/assist loop: B2 W3 (2E) |
| D8 | at-most-once auto-label claim | B2 W3 (2E) |
| D9 | a blocked event loop falsifies liveness | **B1-1A** (thread heartbeat, TTL 120 s) |
| D10 | job success states lie on bulk item errors | B0 (OS-2), B1-1A (OS-3) |
| D11 | config first-create race | B0 (OS-1) |
| D12 | region-cluster stale window | B1-1A |
| D13 | job-state writers race on one temp file | B0 (CC-1) |
| D14 | per-process caches and caps behave as global | B1-1D (BA-1, BA-3, BA-4, BA-8 documented); B4-4B (executor and Triton channel pools sized per `API_WORKERS`) |
| D15 | Prometheus multiprocess files accumulate | B2 W3 (OPS-2, evidence first) |

### 3d. Source-plan tests that are superseded

| Source-plan test | Replaced by | Batch |
|---|---|---|
| `tests/test_file_job_concurrent_write.py` (Q-P1.0) | `tests/test_atomic_write_concurrent.py` (CC-1) | B0 |
| `tests/test_job_claim_cross_process.py` (Q-P1.1/1.2) | `tests/test_job_lease_cross_process.py` + `tests/live/test_job_lease_live.py` | B1-1A |
| `tests/test_reconcile_sibling_job.py` (Q-P1.5) | `tests/test_job_lease_takeover.py` (fresh lease untouched; expired lease taken over) | B1-1A |
| `tests/test_class_registry_cross_process.py` (Q-P1.3, file lock) | same name, rewritten against the OCC document and the Manager-backed fake (CC-2) | B1-1B |
| `tests/test_config_store_first_create.py` (Q-P1.4) | `tests/curation/test_config_store_first_create.py` (OS-1, on `b0/data-ops-fixes`) | B0 |
| Trainer claim test (Q-P1.6) | `tests/trainer/test_job_claim_generations.py` (6.3) | B1-1A |

---

## 4. Release strategy (D-7)

**Problem.** The first plan tagged nothing between v0.7.0 (B0) and the end of B4. B1 carries the
registry race, the lease starts and about 30 LIVE audit fixes. The audit plan meant PR-3, PR-4 and
PR-6 to ship as 0.7.x.

**Options:**

| Option | What ships when | Cost |
|---|---|---|
| **A (recommended): early v0.8.0 after B1** | 0.7.0 = B0; **0.8.0 = B1** after R-B1; 0.9.0 = B2+B3+B4 after R-0.9; 0.10.0 = B5 | one extra acceptance subset (R-B1, about half a day); milestone renames (9.3) |
| B: v0.7.1 after B1 | same timing as A | B1 removes wire fields (BA-2 timing keys, BA-5 `error_type`) and needs a data migration (registry), which is not a patch release; rejected |
| C: one v0.8.0 after B4 | B1..B4 together | 0.7.x users get no audit fixes for the duration of B1..B4 (an XL Celery batch included): duplicate class ids, stale class names per worker, sliver-crop GB allocations and the kNN post-filter stay live |

**Recommendation: A.** Breaking changes are allowed in 0.x and the owner has said no
backwards-compatibility shims, so B1's wire changes and the registry migration go in 0.8.0 with
a CHANGELOG "Breaking" section and the upgrade hook (6.2). Releases after A:

| Release | Content | Acceptance |
|---|---|---|
| v0.7.0 | B0 | R3 (full) |
| v0.8.0 | B1 | **R-B1**: acceptance subset + upgrade path (section 7.1, "Release") |
| v0.9.0 | B2 + B3 + B4 | R-0.9 (full; Cropwright changes of B2 and B4 ship once) |
| v0.10.0 | B5 | per sub-batch gates + one release acceptance |

**Direct upgrades are supported.** 0.6.0 -> 0.8.0 (skipping 0.7.0) and 0.7.0 -> 0.8.0 are both
supported upgrades; R-B1 tests both (`installer_upgrade` from each). 0.8.0 -> 0.9.0 adds Redis
and the worker services. R-0.9 tests 0.7.0 -> 0.9.0 and 0.8.0 -> 0.9.0.

---

## 5. Batch 0: finish 0.7.0

**Branch:** `fix/411-r2-round` (worktree `<repo>-wt-r2round`). The B0 fix
branches are cut from it and merge back with `--no-ff`:
- `b0/ocr-fixes`: GPU-1, GPU-3a+13, GPU-2; already merged.
- `b0/data-ops-fixes`: OS-2, OS-1 and OPS-1 committed; CC-1 and BA-7 in progress.

Everything below lands **before pushing**: one gate cycle instead of two. The audit plan's
"PR-1" is absorbed into this branch.

### 5.1 Scope

> **Status 2026-11-03 (revision 3):** all eight B0 small fixes are now MERGED into `fix/411-r2-round`.
> - **OCR:** GPU-1 `a9ac8ed8`, GPU-3a/13 `a6571415`, GPU-2 `0b722ddc`.
> - **Data/ops:** OS-2 `16c3924f`, OS-1 `b76f744f`, OPS-1 `3b5a3ddd` + uvicorn floor `6a53673c`, BA-7 `1ff98a77`, CC-1 `55219ec5`.
> - **Test fake:** `713d50e9`, the worker-leak fake returns `hits.total`.
> - **Verification:** a targeted run of the B0 test files on the merged branch gave 86 passed. The full gates are deferred.
> - **Remaining B0 steps:** D-2 (the `ultralytics==8.4.197` pin), release prep (VERSION), the full gate cycle, push, merge, R3 and the tag.
> - **Open choices raised by the fixes:**
>   - a shared OCR detector-size helper for `/ocr` and the cascade; it changes `/ocr` behaviour, so decide in B1-1C;
>   - pyclipper in the hardened Triton scan image (`docker/hardened/triton`), decide in B2 with the image work;
>   - `autolabel/job.get_state` still repairs through `_fail_unclaimed`; Celery wave 3 removes it, so leave it.

Already on the branch (ledger section 3):
- installer engine adoption and `--skip-models`;
- assist-loop admission and `reason_message`;
- #437, #440, #442, #443, #445;
- `test_full_system` fails loudly;
- validator bypass sets;
- the #439 revert;
- removal of module-level asyncio locks;
- SEC-1, SEC-2;
- the ultralytics floor `>=8.4.174`;
- GPU-15 (region head from the declared output).

**CHANGELOG entries already present** (do not duplicate them):
- "No in-process lock stands in for cross-process safety";
- the #437, #442 and #443 entries;
- the OPS-1 jitter entry from `b0/data-ops-fixes`.

Check with `rg -n "jitter|holdout freeze|create-only|pyclipper|EXIF" CHANGELOG.md` before you add
an entry.

One commit per task (conventional commits, red test first, control paired). Each task's fix,
traps and tests are in audit plan section 4.

| Task | Status | Why in 0.7.0 | Files | Red test file | Size |
|---|---|---|---|---|---|
| **GPU-1** | merged (`b0/ocr-fixes`) | high, live on every `/ocr` call | `src/clients/triton_ocr.py`, `src/services/detection/cascade_detect/ocr_recognizer.py` (one `ocr_det_input_size`) | `tests/test_ocr_client_frame.py` (on the branch) | S |
| **GPU-3a + GPU-13** | merged | high, live for rotated photos; GPU-13 is the same function | `src/services/ocr_service.py`, `src/clients/triton_ocr.py` (`_uint8_bgr` float64 path deleted) | `tests/test_ocr_orientation.py` (on the branch) | S |
| **GPU-2** | merged | medium, live; R3 rebuilds every image anyway, so the Triton rebuild costs nothing extra | `Dockerfile.triton`, the hardened Triton Dockerfile, `models/ocr_pipeline/1/ocr_postprocess.py`, `requirements-test.txt`, `ATTRIBUTION.md` | `tests/test_ocr_unclip_pyclipper.py` (on the branch) | S |
| **OS-2** | committed (`b0/data-ops-fixes`) | medium-high; protects the frozen holdout (lock rule) | `src/routers/curation/holdout_freeze.py`; public helper in `src/clients/opensearch/bulk.py` | `tests/curation/test_test_holdout_freeze.py` (on `b0/data-ops-fixes`) | S |
| **OS-1** | committed | live (narrow), compliant (`op_type='create'`) | `src/services/config_store/index.py`, `src/services/curation/keymap.py` | `tests/curation/test_config_store_first_create.py` (on `b0/data-ops-fixes`) | S |
| **OPS-1** | committed; **add the uvicorn floor** | docs already claim the jitter; the interim mitigation for CC-4 until B2 | `docker-compose.yml` api command (done); check the `Dockerfile` CMD; **`requirements.txt` and `requirements-test.txt`: `uvicorn[standard]>=0.41.0`** (0.41.0 added `--limit-max-requests-jitter`; images resolve 0.54.0 today) | `tests/compose/test_worker_recycle_jitter.py` (exists) + a requirements assertion that the uvicorn floor is `>=0.41.0` | S |
| **CC-1** | in progress | reproduced 500s on status polls; removes CC-2's 500 mode; every later batch writes through it | `src/utils/atomic_write.py` (new); `file_job.py`, `job_reconcile.py`, `holdout.py`, `routers/curation/bakeoff.py`, `dataset_import/store.py`, `label_ops/store.py`, `clients/curation_opensearch/registry.py`; `get_state` in `item_scores/job.py`, `embedding_viz.py`, `selection/job.py` stop writing | `tests/test_atomic_write_concurrent.py` + `tests/test_job_state_read_only.py` (below) | S |
| **BA-7** | in progress | an outage must not read as "not found" | `src/routers/curation/crops_exclude.py` | the implementer's red test (audit BA-7; `tests/curation/test_crop_discard.py` is the nearest existing file) | S |
| **Ultralytics exact pin** (new in revision 2, D-2) | to do | R3 must test the version that ships; today `>=8.4.174,<8.5` floats per build day | `requirements.txt`, `requirements-test.txt`: `ultralytics==8.4.197` (what the rc5 API and export images resolve); comment cites #447 and B3; `/opt/venv-y11` (8.3.253, exact already) unchanged | `tests/test_ultralytics_pin.py`: every requirements file that names ultralytics uses `==` and the same version (U-6.1 item 7, CPU, no network) | S |
| **Release prep** | after the fixes | R2 finding 1 | `VERSION` -> `0.7.0` (in this commit, not in the fix branches); CHANGELOG entries for every fix not yet covered. The OCR geometry and orientation changes are **behaviour changes**: say so. Known issues: #447 line (below); the CC-2 line already says "tracked separately", so link N2; CC-4: "a background job can end `interrupted` when an API worker recycles; resume it (fixed when jobs move to workers)" | `tests/test_version_matches_changelog.py` if one exists, else the changelog-rules check | S |

**CC-1 and resume (review C-3).** Today `is_live` is already derived (a stale heartbeat reads as
not live, without a write), and `repair_if_stale` persists `interrupted` under `_state_lock`. B0
keeps that split so neither singleton starts nor resume wedge:
- **Start paths** keep using `is_live`. A dead holder's `running` record with a stale heartbeat
  is not busy, so a new start proceeds and nothing is written on the read.
- **Resume paths** (dataset import, reprocess, assist, assist loop, bulk ingest, combine,
  promote) keep calling `repair_if_stale` first. It persists `interrupted`, so the
  `RESUMABLE_STATUSES` check passes. This is the one persister until B1 replaces it with lease
  takeover.
- **`get_state` readers** that wrote a repaired state today (item scores, embedding viz,
  selection) return the derived view and write nothing. None of the three is resumable.
- Red tests in `tests/test_job_state_read_only.py`:
  - a status poll on a stale job leaves the `state.json` mtime unchanged (fails today for item
    scores);
  - with a dead holder (`running`, heartbeat older than `HEARTBEAT_STALE_S`), a new
    `selection.start_job` is not refused as busy;
  - a stale dataset import still resumes after one `repair_if_stale` call.
- Control: a fresh `running` job is reported `running` and a second start is refused.

**GPU-2 details** (already merged; for the release notes):
- the commit pins `pyclipper` to an exact version in the Triton Dockerfiles and in
  `requirements-test.txt`;
- `ATTRIBUTION.md` gains the MIT row;
- the gate checks `find_spec('pyclipper')` in the built image.

The executing agent records the pinned version in the PR description.

Judged and **left out** of 0.7.0:
- **#447 / U-Step 1-3.**
  - YOLO26 export is not in the default install (`rg -i yolo26 scripts/lib` finds nothing); it is
    reached only by `POST /models/upload` with the export worker and by manual export.
  - The trainer's promote path is correct on its 8.4.48 pin.
  - Fixed in B1-1E.
  - CHANGELOG "Known issues": "uploading a YOLO26 checkpoint for export can build an engine with
    the wrong output head; use the trainer's promote path until the fix in 0.8.0."
- **CC-2** (registry race): size L and needs J-REG. Known issue (above). Fixed in B1-1B.
- **CC-3**: cut (3a).
- **U-C11** (the installer's heredoc YOLO11 config is FP32 while the committed one is uint8/FP16):
  accepted until B3. Both serve correct boxes, and B3 deletes the heredoc.

**Rollback.** Each OCR task is its own commit. If the OCR behaviour change is rejected after the
push, revert the three OCR commits only (GPU-1, GPU-3a+13, GPU-2) on a short branch; the rest of
B0 stands.

### 5.2 Gate cycle (once)

Wait for the host all-clear first. Heavy runs are paused until the transcribe-app-1d session
messages "done" (ledger section 8). Then, in the branch worktree:

1. Targeted red/green per task: `.venv/bin/python -m pytest <files> -q --no-cov`. Record each red
   output for the PR description.
2. `.venv/bin/pre-commit run --all-files`.
3. Offline suite once: `.venv/bin/python -m pytest tests/ -q --no-cov -m 'not live' -n auto`.
   This is a heavy run, so announce it to peer sessions.
4. `make contracts-check`; `.venv/bin/python scripts/docs/check_docs_vs_code.py`.
5. Frontend full set (the branch touches `frontend/src` for #440):
   - real `npm ci` in the worktree's `frontend/` (not a symlinked `node_modules`);
   - then `npm run check`, `npm run lint`, `npx knip`, `npm test`, `npm run build`;
   - then `npm run test:e2e`, Playwright, which is a heavy run: schedule it with the offline
     suite, never in parallel with another heavy run;
   - `frontend/CHANGELOG.md` has the #440 entry.
6. Changelog rules, as CI runs them (root and frontend).
7. **OCR comparison** (GPU-1, GPU-2 and GPU-3a together, once):
   - live harness under `docker compose -p op-ocr-b0 ...` on GPU 0, with the Triton image rebuilt
     under a **local-only tag** (watchtower clobber hazard);
   - **image set:** 50 COCO images chosen by a stated rule: the first 50 image ids, in ascending
     order, from the public COCO-Text v2 annotations with at least 3 legible machine-printed words.
     COCO-Text annotates COCO train2014 images, so it stays within the public-data rule.
   - add EXIF orientation 6 and 8 copies of 10 of them, and 10 RGBA PNG conversions;
   - record the ids in `artifacts_local/b0/ocr/ids.txt`;
   - record before/after box geometry and text in `artifacts_local/b0/ocr/`;
   - check that `find_spec('pyclipper')` is True in the image.
   - If the COCO-Text licence or download is a problem on the day, use COCO val2017 images whose
     captions contain one of `sign`, `text`, `written`, `menu`, `banner`
     (first 50 ids) and record the rule.
8. Push only when all of the above are green. Verify the push with `git ls-remote` and the CI run,
   never a piped exit code. Open the PR against the branch's upstream
   (`git rev-parse --abbrev-ref @{u}`; main if unset). Merge commit, never squash, only on the
   owner's yes.

### 5.3 R3 live acceptance (after the merge, on the merged main SHA)

Same procedure as R2 (`artifacts_local/r2/REPORT.md`):
- its own compose project (`op-r3`);
- its own install dir under `<scratch>/r3`;
- a port base distinct from the peers';
- images built by `build_and_publish.sh --dry-run --version v0.7.0 --local-tag-suffix -rc6`
  (local tags only);
- `scripts/release/acceptance_run.sh` (all phases), `tests/live` (`-p op-r3-live`), frontend
  `npm run test:live`, `tests/test_full_system.py`, `tests/validate_visual_results.py`;
- image audit for every image, Trivy, uninstall, the token-leak grep;
- before/after snapshots proving no peer stack was touched.

R3 must **also** verify:

| Check | How | Pass |
|---|---|---|
| **Ultralytics pin** | `python -c "import importlib.metadata as m; print(m.version('ultralytics'))"` in the API and export images | prints `8.4.197` in both; `/opt/venv-y11` prints `8.3.253` |
| **uvicorn floor and jitter** | the same for `uvicorn`; `docker inspect` of the api command | version >= 0.41.0; the command carries `--limit-max-requests-jitter` |
| Segmenter tier (ledger task #22) | install with the segmenter tier; HF token from the stack's `.env` (never printed); segmenter on GPU 0 (GPU 2 if free and agreed) | segmenter healthy; a region profile that uses it runs on a COCO subset; the acceptance phases that use regions pass; the 8 frontend live tests that skipped for lack of region data now run |
| `--skip-models` live | fresh install with `--skip-models` | exit code 0, OpenSearch healthy, the message says models were skipped; then `models install --yes` and `start` reach ready |
| Engine adoption | second fresh install dir seeded with R3's engines and `.install/precision.tsv` | the log shows every group adopted or "up to date, skipped"; no rebuild (R2 rebuilt all 8 in about 30 min) |
| Triton down vs engine invalid | stop Triton during `models install` | a clear "Triton unreachable" stop, no rebuild |
| VERSION | `/health` | reports `0.7.0` |
| OCR fixes | `/ocr` on the B0 OCR set inside R3 | boxes align with the text; rotated frames match `image_size` |
| Holdout freeze | acceptance `holdout` phase | passes (the fail-closed path is covered offline) |
| Cleanup cancels jobs (R2 fix (e), unverified) | start a job that outlives the cleanup window (a dataset import of the 2k COCO set, or a train with more epochs), then run cleanup | the job ends `cancelled` and the project deletes |
| Assist-loop admission (R2 oddity: budget spent, nothing visited) | UI loop with `min_root_size 10`, budget 120 | every spent crop belongs to a node with a decision; unvisited nodes carry `reason_message` |
| #437 | the promote phase yields a YOLO26 model; point a region profile at it | region candidates decode (`[B,N,6]`) |
| #442, #443 | activate a VLM endpoint right after creating it; ingest while the cluster-refresh daemon runs | 200 on activate; the log shows the `recent_ingest` hold |
| #440 | browser at 1440 and 430 px, full-page screenshots actually viewed | the profile list and activate work, the refusal report shows, the cards are readable at 430 |
| SEC-1, SEC-2 | the branch tests named in the private notes, plus the positive control (a legitimate model validates and exports) | pass |
| Status polls under load | poll a running job's status from a script while ingest runs | no 500 (CC-1); `state.json` mtime unchanged by polls of a stale job |
| BA-7 | stop OpenSearch, call `review_dismiss` | 503, not 404 |

R3 failures are fixed on a short branch `fix/r3-findings`: targeted tests only, one PR, and only
the failed acceptance phases re-run.

### 5.4 Tag decision point

When R3 passes, report to the owner with the evidence path (`artifacts_local/r3/REPORT.md`) and
wait. Only on an explicit yes:
1. release commit (CHANGELOG `[Unreleased]` -> `[0.7.0]`);
2. annotated tag `git tag -a v0.7.0`;
3. image build, image-vs-git audit, Trivy;
4. publish per `RELEASE_CHECKLIST`;
5. GitHub release, published, not a draft;
6. verify with `git ls-remote --tags` and the release page;
7. execute the B0 rows of section 9.4 (issue closes).

**Done when (B0):**
- the branch is merged with green CI;
- the R3 report says PASS;
- the owner's decision is recorded.

**Size:** M (10 small commits, one gate cycle, one acceptance run).

**Upgrade impact:**
- none beyond a normal upgrade: the images are rebuilt, and the default engines are unchanged;
- the OCR output changes (a behaviour change, in the CHANGELOG);
- the Ultralytics pin equals what 0.7.0-rc images already ship.

**Docs, contracts, CHANGELOG:**
- `docs/PERFORMANCE.md` (the jitter now matches);
- `ATTRIBUTION.md` (pyclipper);
- `CHANGELOG.md`: behaviour changes, known issues for #447, CC-2 and CC-4;
- `frontend/CHANGELOG.md` (#440);
- no wire change in the new B0 commits (the existing `reason_message` contracts are already on the
  branch).

**Risks:**
- the OCR behaviour change surprises a consumer (mitigation: the CHANGELOG and before/after
  evidence; rollback above);
- host contention delays R3 (section 13);
- R3 finds a regression in the large branch (`fix/r3-findings`).

---

## 6. Designs this plan introduces (B1)

### 6.1 J-LEASE: the job lease document

**Why OpenSearch:**
- The claim must hold across processes, and later across workers, before Redis exists (B1
  precedes B2).
- It is per project, so the isolation guard covers it.
- OpenSearch OCC is the documented mechanism (`op_type=create`, `if_seq_no`/`if_primary_term` ->
  409 [O1]), and OpenSearch's own Job Scheduler takes a lock document with `seq_no`/`primary_term`
  the same way [O2].
- Redis's docs discourage the simple `SET NX PX` lock in favour of Redlock [V5], and Redis would
  exist only from B2.
- Filesystem generation files are compliant but single-host; they are used only for the
  trainer, which has no OpenSearch client (6.3).

**Where (corrected in revision 2):** in the project's **SETTINGS role**, resolved through the
project resources with `index_name(cfg, IndexRole.SETTINGS)`, never by a built name:
- **folded layout** (every project created by `resources_for_new`, `default` included): SETTINGS
  resolves to the `configs` index (`{prefix}{slug}__configs`). That index has
  `mappings.dynamic: false`, and its folded docs are read by fixed id only and carry no
  `doc_type`;
- **legacy layout** (projects whose stored resources predate the shard fold): a separate
  settings index whose mapping is dynamic.

**Document shape: one top-level object key per feature, so nothing collides with mapped fields.**
- The configs index maps `kind`, `name` and `revision` (integer) for config-store rows, so lease
  and registry fields must not use those names at the top level.
- Doc id `lease:<kind>` or `lease:<kind>:<scope>`; source:
  `{"lease": {kind, scope, job_id, owner, generation, acquired_at, heartbeat_at, ttl_s, state}}`.
  No `doc_type`, so a `doc_type`-filtered config-store query never sees it.
- Mapping: add `"lease": {"type": "object", "enabled": false}` to `_configs_body()` **and** to
  `_settings_body()`.
- An additive mapping put through the existing bootstrap (which already applies additive mapping
  puts) brings existing folded and legacy indexes up to date. On the dynamic legacy index this
  stops OpenSearch from mapping lease fields on its own.
- Tests:
  - `tests/projects/test_configs_mapping_union.py` stays conflict-free;
  - `test_folded_roles_by_id_only.py` gains the `lease:*` and `class_registry` ids;
  - new `tests/test_lease_folded_and_legacy_index.py` acquires, renews and releases a lease on a
    folded project and on a legacy-layout project, and asserts the index each one used.

**Code:**
- pure rules in `src/domain/job_lease.py`: expiry `heartbeat_at + ttl_s < now`, generation bump,
  the kind/scope table below;
- OpenSearch operations in `src/clients/job_lease.py`, built on the existing helpers in
  `src/clients/occ.py` (`_fetch_with_version` pattern), not a new retry loop;
- the heartbeat thread in `src/services/curation/lease_heartbeat.py`, used unchanged by the B2
  task wrapper.

| Operation | Calls | Outcome |
|---|---|---|
| `acquire(kind, scope, job_id, owner, ttl_s)` | realtime `get`; absent -> `index(op_type='create')`; present and (`released` or expired) -> `index(if_seq_no, if_primary_term)` with `generation+1` | `Acquired(generation, seq_no, primary_term)`; any 409, or a live holder -> `Busy(job_id, heartbeat_at)` |
| `renew(lease)` | `index` with the lease's own `if_seq_no` | `Renewed` (new seq_no), or `Lost` on 409 |
| `fence(lease)` | `renew` used as a check immediately before a terminal or progress write | proceed only on `Renewed` |
| `transfer(lease, new_owner)` | as renew, changing `owner` | the B2 task wrapper takes over the API-acquired lease |
| `release(lease)` | `index` `state=released` with `if_seq_no` | a 409 means someone took over: log it and do not overwrite |
| `takeover_expired(kind, scope)` | as acquire, only when expired | used by reconcile and `repair_if_stale` now, by the B2 reaper later |

**Kind and scope per start site** (the lease doc lives in the project's index, so the project is
implicit):

| Kind | Scope | Start sites (re-find with `rg -n "exclusive_start_lock\(" src`) |
|---|---|---|
| `reprocess` | none (project singleton) | `reprocess_job.py` |
| `assist_run` | none | `assist/job.py` |
| `assist_loop` | none | `assist_loop/job.py` (start and resume) |
| `dataset_import` | none | `dataset_import/runner.py` (start, resume, undo) |
| `item_scores` | none | `item_scores/job.py` |
| `probe` | none | `probe_job.py` |
| `promote` | none | `training/promote_job.py` (also the single writer of `promote_json.py`) |
| `combine` | none (target project) | `projects/combine/service.py` |
| `bulk_ingest` | none | `bulk_ingest/runner.py` |
| `source_scan` | `<source_id>` | `bulk_ingest/source_scan.py` |
| `legacy_repair` | none | `projects/legacy_repair.py` |
| `label_op` | `<op_id>` | `label_ops/store.py` `write_lock` |
| `label_undo` | none | `label_ops/store.py` `undo_lock` |
| `autolabel`, `selection`, `embedding_viz`, `region_cluster`, `region_fp_centroids` | none | the four unlocked starts (Q-D2) |

**Rules:**
- The API acquires the lease **before** writing the job record (`queued`/`running`) and answers
  409 `job_busy` on `Busy` (the existing error code per family).
- **Heartbeat in a thread from B1** (Q-D9).
  - It is a daemon thread started next to the job. It runs its own event loop with its own
    guarded client from `make_script_opensearch`, so the isolation guard applies, and it uses no
    new sync client factory.
  - Every tick `HEARTBEAT_TICK_S = 15` s it touches the file heartbeat (UI display) and renews
    the lease.
  - The asyncio `heartbeat_ticker` is deleted.
  - Reason: a blocked loop must not cost a live job its lease. Until B2, CC-5 upload extract and
    directory ingest still block the loop.
- **TTL: `LEASE_TTL_S = 120` s**, at least 8 missed ticks. `HEARTBEAT_STALE_S` becomes the same
  constant, so the file and the lease agree on staleness. OpenTranscribe's thread-heartbeat
  lease uses about 90 s (task plan 4d).
- **Busy window after a crash**: a dead holder blocks a new start for at most `LEASE_TTL_S` (120
  s) after its last renew. The next start after that takes the lease over (409 `job_busy` until
  then). Document this in `docs/CURATION.md`.
- **Fencing is cooperative.**
  - On `Lost` the heartbeat thread sets a stop flag. The job checks it before each progress
    write and stops.
  - Every terminal write and every `state.json` progress write is preceded by `fence(lease)`, and
    the record carries `lease_generation`.
  - Residual risk: a holder that stalls longer than the TTL between `fence` and the write can
    still write once. With TTL 120 s and a write that takes milliseconds this is accepted and
    documented. A taker-over always writes a higher `lease_generation`, so the audit trail shows
    a superseded writer.
- Only the lease holder (or a taker-over) writes `state.json`. Readers never write (CC-1).
  `repair_if_stale` becomes `takeover_expired` + write, so `_state_lock` goes.
- **Mid-upgrade jobs.** Upgrading recreates the API container, so an in-API job dies with it.
  - A `running` record with **no lease doc** and a stale heartbeat can be taken over: `acquire`
    sees the doc absent, creates it, and the record is repaired to `interrupted` (resumable
    families) by the acquirer.
  - A `running` record with no lease doc and a **fresh** heartbeat is treated as live until the
    heartbeat ages past `LEASE_TTL_S`. This window exists only in the first two minutes after an
    upgrade.
- Cancel stays cooperative (`cancel.flag`, a separate file); it does not need the lease.
- OpenSearch unreachable -> the start is refused with 503 (fail closed).
- A project delete removes the lease docs with the index. A running job then gets `Lost` on its
  next renew and stops. Delete already waits for or cancels jobs.
- Clocks: on one host they agree. Multi-host would need the TTL to exceed the clock skew; record
  this in INSTALLATION if multi-host is ever supported.

**Traps:**
- `get` is realtime, so no refresh is needed between `get` and `index`; never read a lease with
  `search`.
- Do not name top-level lease fields `kind`, `name` or `revision` (mapped on configs).
- The heartbeat thread must not share the request loop's client.
- A `fork`ed child must not inherit a running heartbeat thread. Use `spawn` in tests.

**Tests (B1-1A):**
- Red: `tests/test_job_lease_cross_process.py`.
  - 8 `spawn` processes acquire one lease at a `multiprocessing.Barrier`, against a fake
    OpenSearch whose state lives in a `multiprocessing.Manager`. The fake's per-document atomicity
    emulates the server; it is test scaffolding, not production locking.
  - Pass: exactly one `Acquired` and 7 `Busy`.
  - Repeat for `autolabel.start_job`, `selection.start_job`, `embedding_viz.start_job` and region
    clustering (today more than one starts).
- Red: `tests/test_job_lease_takeover.py`.
  - A renew after a takeover returns `Lost`, and the old holder writes nothing more (its `fence`
    fails).
  - Reconcile on a fresh lease leaves it alone; on an expired lease it takes over and marks the
    job `interrupted`.
- Red: `tests/test_lease_heartbeat_thread.py`. A handler that blocks the event loop for longer
  than `LEASE_TTL_S` (monkeypatched to 2 s) keeps its lease. This fails with the asyncio ticker.
- `tests/test_lease_folded_and_legacy_index.py` (above).
- `tests/test_lease_upgrade_no_doc.py`: a `running` record with no lease doc and a stale
  heartbeat can be acquired and is repaired; with a fresh heartbeat it is busy.
- Control: a sequential acquire, release, acquire gives generations 1 and 2.
- Live: `tests/live/test_job_lease_live.py` (live harness, its own `-p`) runs the same 8-process
  race against real OpenSearch, 20 repetitions, with a **budget of 10 minutes**. Record p50/p99
  acquire latency.

### 6.2 J-REG: class registry in OpenSearch, with an upgrade path

This follows audit CC-2's recommended design. This section fixes the open choices.

**Authority:**
- one document per project, id `class_registry`, in the SETTINGS role index (resolved as in 6.1);
- source `{"class_registry": {classes: [{id, name, deprecated, hotkey, ...}], next_class_id,
  registry_revision}}`;
- the key is mapped `{"type": "object", "enabled": false}` in both bodies.

**Size bound:**
- `MAX_REGISTRY_CLASSES = 10_000` per project (a module constant). That is about 1-2 MB of
  `_source`, far below the 100 MB default `http.max_content_length`.
- `add_class` past the bound returns 409 `registry_full`.
- A 100k-class import is out of scope; it would need per-class docs (the `classes` mirror index
  already exists) and is recorded as a risk, not built.

**Mutators** (`add_class`, `rename_class`, `merge_class`, `set_deprecated`, hotkey set):
- realtime `get`, apply in memory, `index(if_seq_no, if_primary_term)`;
- on 409, re-read and re-apply, **at most 5 attempts**, then 409 `registry_conflict` to the
  caller. Use the OCC helpers.
- `add_class` is idempotent by name: an existing non-deprecated class with that name returns its
  id.
- The follow-up hotkey write and the rename-then-write in `routers/curation/classes_write.py`
  each become one mutation.

**Readers:**
- one async store with a per-process snapshot keyed by the doc's `_seq_no`, checked with a cheap
  realtime `get`;
- this is the project-registry `ensure_fresh` pattern that the audit's Appendix B found correct,
  and the "revision stamp" that fixes the class-name staleness class (BA-1).

**The file:**
- `class_registry.json` stops being an authority (CC-2 option (a)). It is written only as a
  snapshot inside exports and training datasets (`export_writers.py`, `dataset_import/op_export.py`).
- **The trainer is unchanged.** It already reads the export's own `class_registry.json`
  (`docker/trainer/dataset_prep.py`, `subset_dataset.py`) and has no OpenSearch client. So there
  is no ordering problem between the trainer image and the API: the API writes the snapshot from
  the doc.
- Every other reader of the project-level file is deleted. That is about 75 references; find them
  with `rg -n "class_registry_path|class_registry\.json" src scripts docker`, and keep the trainer
  and export-snapshot hits.
- `scripts/curation/seed_class_registry.py` and `repair_merged_active_classes.py` move to the doc,
  or are deleted if obsolete.

**Upgrade path (new in revision 2; D-9):**
1. **Fail closed.**
   - When a project is bound and its `class_registry.json` exists while the `class_registry` doc
     is missing, every class-dependent route of that project answers **409
     `registry_not_migrated`**. The message names the command:
     `openprocessor curation migrate-registry --project <slug>`.
   - Inference routes and other projects are unaffected.
   - There is no runtime read-shim (no back-compat rule).
2. **The migration CLI**, `scripts/curation/migrate_class_registry.py`:
   - flags `--project <slug> | --all`, `--dry-run`;
   - seeds the doc from the file with `op_type='create'` (idempotent: an existing doc that equals
     the file is a no-op; one that differs is an error);
   - preserves ids exactly;
   - then **renames** the file to `class_registry.json.migrated` (it never deletes it);
   - writes its dry-run and real-run reports to the project state dir.
3. **CLI subcommand**: `./openprocessor curation migrate-registry [--project <slug>|--all]
   [--dry-run]` runs the script in the `api` container.
4. **Installer hook**:
   - `openprocessor upgrade` (and `repair`), after the stack is healthy, runs the migration with
     `--all --dry-run`, saves the output under `.install/`, then runs it for real;
   - a failure stops the upgrade with the report path and leaves the stack in its fail-closed
     409 state;
   - fresh installs create the doc at project creation.
5. **Downgrade note** (INSTALLATION.md and the CHANGELOG "Breaking" section): to go back to 0.7.x,
   rename `class_registry.json.migrated` to `class_registry.json` in each project's data dir.
   Classes added after the migration exist only in the doc; export them first (export writes the
   snapshot).
6. **Tests:**
   - `tests/test_registry_not_migrated.py`: file present and doc absent gives 409, and other
     projects keep working;
   - `tests/test_migrate_class_registry.py`: dry-run writes nothing; the real run creates the doc
     and renames the file; a rerun is a no-op; a conflicting doc is an error;
   - `tests/installer/test_upgrade_registry_migration.py`: the upgrade path calls the migration
     and stops on failure;
   - the acceptance phase `installer_upgrade` (`scripts/release/acceptance_phases.sh`) gains a
     project with classes created before the upgrade, and asserts that the ids are equal after it
     (R-B1).

**Mirror:** `sync_to_opensearch` (the per-class `classes` index) is written from the doc after each
successful OCC write.

**Async:** combine, import and `class_ensure` call the store with `await`; no `to_thread`.

**Tests:**
- CC-2's tests: 8 spawned `add_class` calls at a barrier, against the Manager-backed fake that
  enforces `if_seq_no`;
- 100 runs give N classes and N distinct ids;
- control: sequential ids 0..N-1;
- the live two-worker scenario;
- red against today's file registry. It is intermittent, so repeat it 20x, as CC-2 says.

**Interplay with #365 (embedding swap):** none in B1. The registry doc is independent of embedding
fields. When #365 is scheduled, its dual-field migration should reuse this upgrade-hook pattern
(fail-closed state, `upgrade` hook, rename-not-delete).

### 6.3 Trainer and training-job claims (no OpenSearch): create-only generations

The trainer container has no OpenSearch client. Q-D7 is latent (one trainer replica), but the
API's `training/jobs.py` `exists()`-then-write is live. Design:
- **Claim:**
  - the claim file `<job_id>.claim.g<N>` is created with `os.open(..., O_CREAT|O_EXCL)`;
  - the first claim is `g1`;
  - the holder is the highest generation present;
  - the claimer writes `{owner, generation, claimed_at}` into it and touches the job heartbeat.
- **Stale reclaim:**
  - when the holder's heartbeat is older than the stale window, a reclaimer creates
    `<job_id>.claim.g<N+1>` with `O_EXCL`;
  - exactly one reclaimer wins; the others get `FileExistsError` and back off;
  - the winner then removes generations at or below N;
  - this is the file analogue of OCC. **No unlink-then-create**, and nothing is renamed over a
    live claim.
- **Fencing:** before each status write, the holder checks that no `g>N` exists. If one does, it
  stops (cooperative, as in 6.1).
- **API create** (`src/services/training/jobs.py`): write the job spec with `O_EXCL` instead of
  `exists()` then write; a collision returns 409.
- Tests: `tests/trainer/test_job_claim_generations.py`.
  - 4 spawned reclaimers on one stale claim: exactly one wins `g2`.
  - The stale holder's fence fails.
  - Two `process_one` calls in two processes: one runs.
  - The API create collision gives 409.

---

## 7. Batches after 0.7.0

**Common rules for every batch:**
- **Branch model:**
  - one integration branch per batch, `int/bN-<name>`, cut from main after the previous batch
    merged (stack on the previous integration branch only while its PR is open);
  - each track gets its own branch `bN/<track>` and worktree
    `<repo>-wt-bN-<track>`;
  - tracks merge into the integration branch with `--no-ff`, **one merge commit per track**, in
    the stated order, so a failing track can be reverted alone (`git revert -m 1 <merge>`);
  - then one gate cycle, one PR, a merge commit.
- **Agents.** Opus plans; Sonnet implements and reviews. Every agent brief:
  - forbids `pgrep -f`, `pkill -f` and sleep/until loops;
  - names its compose project (`-p`), its GPU, its worktree and the PR deadline.
- **Each task:**
  - a red test seen failing on the base, plus a paired control;
  - `.venv/bin/pre-commit run --all-files` before each commit;
  - `make contracts` in the same commit as any wire change.
- **Line numbers** in the source plans are from `5b95591f`; re-find every site with the `rg`
  pattern the plan gives.
- **Ratchet baselines only shrink.** If a track grows a file past the `check_file_size` baseline,
  split the file by concern; never raise a baseline.
- **Docs gate.** Every batch lists the docs it touches; `check_docs_vs_code.py` runs in the gate.
- **Commit the source plans** (`task_queue_plan.md`, `codebase_audit_plan.md`,
  `model_family_versions_plan.md`) as the first commit of `int/b1-correctness`, after the
  naming-leak scan and the docs check pass on them, with the stale "at `4103656b`" header line of
  `model_family_versions_plan.md` corrected to "fix/411-r2-round after B0 small fixes" in the
  same commit. All four plans are committed (D-14, done).

### 7.1 B1: correctness foundation (v0.8.0)

**Dependencies:**
- B0 merged and the owner's B0 decision recorded;
- decisions D-7 (release train), D-8 (J-LEASE), D-9 (J-REG and upgrade), D-10 (1E scope), D-11
  (YOLOv5 scope), D-12 (`nms` by capability), D-15 (trainer claim).

Track 1B may slip into its own follow-up PR before the 0.8.0 tag if D-9 is late; no other track
waits on it.

| Track | Scope | Files (main ones) | Size |
|---|---|---|---|
| **1A leases** | section 6.1 (thread heartbeat, TTL 120 s, fencing, both index layouts); convert every start-lock site in 3b's B1 rows using the kind/scope table; the four unlocked starts (Q-D2: `autolabel/job.py`, `selection/job.py`, `embedding_viz.py`, `clustering/region_cluster_jobs.py` incl. FP centroids, now `FileJob` with the heartbeat thread, `_JOB_STALE_S` deleted, `job.json` wire shape kept or contracts + Cropwright updated in the same commit); `FileJob._state_lock` deleted, `repair_if_stale` becomes takeover + write; reconcile takes over expired leases (replaces Q-P1.5/CC-3); `promote_json.py` single writer; section 6.3 (trainer `docker/trainer/job_protocol.py`, `trainer.py`; API `training/jobs.py`); `export_worker.reconcile_claimed` stale-only; OS-3 at the three sites with the B0 helper; the file-lock ratchet | `src/domain/job_lease.py`, `src/clients/job_lease.py`, `src/services/curation/{file_job,job_lock,job_reconcile,lease_heartbeat}.py`, `bodies_other.py`, the sites listed, `scripts/codegen/check_file_locks.py` | L |
| **1B registry** | section 6.2 in full, including the upgrade path | `src/clients/curation_opensearch/registry.py` and `__init__.py`, `routers/curation/classes_*.py`, `class_ensure.py`, `dataset_import/prepare.py`, `combine/execute.py`, export writers, the migration script, `openprocessor` + `scripts/lib/cli_*.sh`, `setup-openprocessor.sh` upgrade step, `scripts/release/acceptance_phases.sh` | L |
| **1C inference and search hygiene** | GPU-4, GPU-5, GPU-8, GPU-9, GPU-10, GPU-16, GPU-17 (with API-12); OS-5, OS-6, OS-7, OS-9, OS-10; CC-6; CC-8 | `src/clients/triton_*.py`, `src/utils/{image_processing,cpu_preprocess,retry}.py`, `src/clients/model_residency.py`, `src/clients/opensearch/search.py`, `routers/curation/{audit,clusters}.py`, `clustering/pit_pages.py`, `residual_fetch.py`, `scripts/curation/worker/runner.py` | M |
| **1D API, frontend, tests** | BA-1, BA-2, BA-3 (adds `cachetools`: MIT licence row in ATTRIBUTION, added to `requirements.txt` and `requirements-test.txt`, try-import deleted), BA-4, BA-5 (removes `error_type` from 500 bodies: `make contracts` in the same commit), BA-8 (document the per-process probe cap; Appendix D cut a cluster-wide lease); CC-5 thumbnail and render handlers (`run_cpu` or `asyncio.to_thread`, one offload per request); FE-1, FE-2, FE-3, FE-5, FE-7; T-1, T-2, T-3, T-4, T-7; T-8 (new, pre-existing on main): test-order pollution, where `tests/curation/cross_project_leak/test_keymap_and_ingest_policy_isolation.py` fails (3 tests) after `tests/projects/test_clone_settings.py` + `test_clone_activations.py` in one process. No single test pollutes it alone, and xdist hides it. Find the shared process state and reset it in a fixture, and add a no-xdist ordered run of those files as the red test; SEC-3 and SEC-6 (neutral; details private) | `src/utils/class_names.py`, `src/app/factory.py`, `src/core/error_handlers.py`, `settings_doc.py`, `diverse_order.py`, `outliers.py`, `eval_datasets.py`, `vlm_endpoint_probe.py`, `routers/curation/images.py`, `image_serving.py`, `frontend/src/...`, `tests/...` | M |
| **1E Ultralytics live bugs (minimal, D-10)** | (1) **#447**: `export/export_yolo26.py::export_fused_onnx` passes `nms=False`; `inspect_fused_output` refuses any output whose per-item dims are not `[300, 6]` (this function moves unchanged into B3's single module); `yolo_triton_config.py` comment fixed and the promote check refuses a non-`[300,6]` ONNX. (2) **U-C2 routing**: `model_export.validate_pytorch_model` records `head_kind` (`one2one` if `hasattr(model.model[-1], 'one2one_cv2')`, else `nms`) and `ultralytics_version`; `export_worker.build_command`/`engines_to_load` switch on `head_kind` instead of `end2end` (a condition change; B3 collapses the two commands). (3) **U-Step 3**: secondary detector class by name (U-C10); the YOLOv5 objectness probe reads its input size from the ONNX and refuses a wrong output shape. (4) Tests through `nms` only (the `end2end` argument is deprecated and mapped to `nms` per the Ultralytics maintainers). **Not in B1** (moved to B3): the trainer's `_export_onnx` `nms` by capability, `half=` changes, the new export module, the full U-6.1 matrix | `export/export_yolo26.py`, `src/services/training/yolo_triton_config.py`, `src/services/model_export.py`, `export/export_worker.py`, `src/services/curation/{ingest_detect,probe_models}.py`, `.github/workflows/*` (job `families`), `tests/export/` | S-M |

**1E tests and CI:**
- `tests/test_export_yolo26_head.py` (marker `export`):
  - exports `yolo26n.pt` with the production function and asserts `[batch, 300, 6]`;
  - negative: a raw-head ONNX is refused (red today).
- `tests/test_validate_pytorch_model_families.py` (yolo26n, yolo11n, yolov10n, yolov5nu, original
  yolov5n): asserts the head kind; the original YOLOv5 `.pt` is refused with a message naming
  YOLOv5u.
- `tests/test_secondary_class_by_name.py`: a registry with a gap, red today.
- `tests/test_yolov5_onnx_probe.py`: a real 640 static official export, red today.
- **Fixtures, sources and licences:**
  - Ultralytics weights from the official GitHub release assets
    (`github.com/ultralytics/assets/releases`, AGPL-3.0);
  - the original-YOLOv5 ONNX produced by the official `ultralytics/yolov5` repo's `export.py` from
    `yolov5n.pt` (AGPL-3.0), once, in CI;
  - none is committed. Fetched in a session fixture into `~/.cache/op-test-weights`.
- **CI job `families`:**
  - its own job, with network;
  - `actions/cache` keyed on the Ultralytics pin and the fixture list;
  - runtime budget 15 min, CPU;
  - skips with an explicit reason only when offline locally; it never skips in CI.
- B3 extends this job to the full U-6.1 matrix; it is not rewritten.

**Merge order into `int/b1-correctness`:** 1E, 1C, 1D, 1A, 1B.
- 1E, 1C and 1D are independent and small.
- 1A and 1B both add a top-level object to the settings/configs mappings, and both touch the
  class/registry callers in combine and import, so 1B rebases on 1A.
- **Revert coupling:** 1B depends on 1A (mapping and bootstrap); 1C, 1D and 1E revert
  independently.

**Test cycle (once):**
- Targeted red/green per task. Run the spawn races (lease, registry, CC-1, trainer claim) 10x
  locally.
- Full offline suite, pre-commit, contracts-check, docs check.
- Frontend full set (1D touches `frontend/src`). Browser check of `/p/<slug>/export` and the
  `/review` diverse page at 1440 and 430 px, with screenshots viewed.
- Live harness, one run, `-p op-b1-live`, GPU 0, with a budget of 90 minutes:
  - `tests/live`;
  - the new lease (20 repetitions, 10 min) and registry live tests;
  - OS-6's filtered-kNN hit count;
  - one blocked-loop check: a dataset upload extract while a job runs keeps its lease.
- The `families` CI job runs in CI, and once locally in the export image (CPU, `--cpus 8`).
- No GPU gate.

**Done when:**
- every task's done-when in its source plan holds;
- the file-lock ratchet baseline lists only the B2/B4 rows of 3b and `job_lock.py`;
- **no caller of `exclusive_file_lock` or `exclusive_start_lock` remains outside `job_lock.py`
  except the dispatcher, sweeper, cache-prune and event-hub sites** (`rg -n
  "exclusive_(file|start)_lock\(" src | rg -v job_lock.py` lists only those);
- no `class_registry.json` read outside export snapshots and the trainer;
- `registry_not_migrated` is tested;
- CI green.

**Size:** XL overall (L, L, M, M, S-M) across 5 parallel agents.

**Rollback:**
- revert a track's merge commit on the integration branch before the PR merges;
- after the merge, revert the PR's merge commit;
- 1B after users migrated: follow the downgrade note (6.2), since the doc stays and the file is
  restored from `.migrated`.

**Upgrade impact:**
- the registry migration runs from `openprocessor upgrade` (fail closed until it runs);
- the new mapping objects are applied additively at bootstrap;
- lease docs appear on first start;
- in-flight in-API jobs die with the container recreate, as on any upgrade, and resumable ones
  show `interrupted`.

**Docs, contracts, CHANGELOG:**
- `docs/ARCHITECTURE.md` (leases, registry doc);
- `docs/CURATION.md` (409 `job_busy` window, `registry_not_migrated`, the migration command);
- `docs/opensearch_schema_design.md` (lease and registry docs and their mapping);
- `INSTALLATION.md` (upgrade, downgrade note);
- `docs/design/curation_api_contract.md`, `contracts/` (BA-2, BA-5, new error codes, region
  cluster job shape if changed);
- `CLAUDE.md` (lock rule now names the lease);
- `CHANGELOG.md` with a "Breaking" section (wire fields removed, registry migration);
- `frontend/CHANGELOG.md`.

**Risks:**
- 1B migration of the owner's real projects (mitigations: the CLI is idempotent, dry-run first,
  the file is renamed not deleted, the owner runs it);
- a lease round trip per start and per 15 s tick (negligible next to the job; measured live);
- 1D's BA-5 removes `error_type` from 500 bodies (check `rg error_type tests frontend/src
  contracts` first);
- T-3 may fail on a fresh checkout's required env vars (run the three files on the CI runner
  config before merging).

**Release: R-B1, then v0.8.0 (D-7).** On merged main, run an acceptance subset in its own project
(`-p op-r-b1`, install dir `<scratch>/r-b1`, local-only image tags):
- `acceptance_run.sh --only installer_upgrade` twice: from a 0.6.0 install and from a 0.7.0
  install, each with projects that have classes, so the registry migration runs and the ids are
  equal afterwards;
- `--only` the class, holdout, label_ops, ingest, cluster, export, train and promote phases;
- `tests/live`, frontend `npm run test:live`, image-vs-git audit, Trivy, screenshots at 1440 and
  430 px of the pages 1D touched.

Tag v0.8.0 only on the owner's word (D-16).

### 7.2 B2: jobs off the API on Celery (v0.9.0)

**Dependencies:**
- B1 merged (the lease is the claim);
- decisions D-17..D-26;
- the OpenTranscribe alignment is the file list in `task_queue_plan.md` section 4d and the Phase 2
  header (`backend/app/core/celery.py`, `backend/app/core/constants.py`,
  `backend/scripts/celery_pool_healthcheck.py`, `docker-compose.yml` worker/beat blocks,
  `app/tasks/CLAUDE.md`). The W1 agent re-reads them read-only and cites the lines in the PR.

**Waves on `int/b2-celery`:**

| Wave | Track | Scope | Size |
|---|---|---|---|
| W1 (one agent) | 2-infra | Q-Phase 2 in full. **Spike S1 first**: record the versions; verify SIGKILL redelivery, the cold-shutdown ack and revoke of a queued task. **Then:** a Redis 8 service (pinned digest, AOF everysec, `noeviction`, required password from an installer-generated mode-600 file, no `ports:`); `src/tasks/{app,queues,job_task}.py`; `src/clients/task_queue.py` (enqueue by name, `task_id = job_id`); `src/config/queue.py`; compose `jobs-worker`, `open-vocab-worker`, `jobs-beat`; Flower under `monitoring`; the pool healthcheck ported from OpenTranscribe; installer and CLI rows; the layering ratchet gains the `tasks` layer; docs. **Celery config** per task plan section 5 rule 4, **including `worker_soft_shutdown_timeout=30` and `worker_enable_soft_shutdown_on_idle=True`** [C8], `visibility_timeout=21600`, `task_soft_time_limit=10800`, `task_time_limit=11700`, `prefetch_multiplier=1`, `acks_late`, `reject_on_worker_lost`. **The wrapper's claim is `job_lease.transfer` of the API-acquired lease, and it reuses the B1 heartbeat thread (`lease_heartbeat.py`) unchanged** | L |
| W2 (three agents) | 2A | Q-Phase 3 families 1-3: `reprocess`, `assist_run`, `item_scores`, `probe`, `selection` (k-center off the loop), `embedding_viz` | M |
| | 2B | families 4-7: `region_cluster`, `region_fp_centroids`, `promote` (delete `run_detached`), `dataset_import` + undo + resume (the resume rescan moves into the task), `dataset_upload_extract` (CC-5 upload), `combine` | M |
| | 2C | families 8-11: `bulk_ingest_run` (`OP_API_LOOPBACK_URL`), `open_vocab_drain` (queue `open_vocab`; delete `_PASS_LOCK`, the last CC-11 site, and its allowlist entry in `tests/test_no_module_level_async_locks.py`), `project_delete_finish`, `directory_ingest` (CC-5, 202 + job id, contracts); CC-9 TaskGroup + Semaphore bounds where each family moves | M |
| W2 (fourth agent, after the contracts settle) | 2G | Cropwright renders the new `queued` status (Q-E14) and the directory-ingest 202; `frontend/CHANGELOG.md` | S |
| W3 (two agents) | 2D | Q-Phase 4: `beat_schedule`; the lease reaper (stale lease -> takeover -> `interrupted` + re-enqueue if resumable; `queued` older than 120 s re-enqueued); dispatcher, sweeper and cache prune moved to beat (3b's flock rows removed); GPU arbiter + promoted reload once (Q-D5; the arbiter overlay mounts the socket into `jobs-worker`); cluster-refresh daemon and IVF retrain check into beat; delete `startup_reconcile.py`, every `reconcile_orphaned_jobs`, `background_jobs.py` and the lifespan loops; OPS-2 measured first, then `mark_process_dead` and a separate metrics dir for Celery children (Q-E19) | M |
| | 2E | Q-Phase 5: auto-label and assist loop become `-Q autolabel -c 1` / `-Q assist_loop -c 1` consumers; delete the trigger files, poll loops, `running.lock` and `QUEUE_MAX_AGE_S`; pick one liveness mechanism and delete the other | M |
| W3 (small) | 2F | SEC-4 and SEC-5 (neutral: an opt-in monitoring service's privileges, a build-input pin and container hardening), done here because B2 already edits compose and adds images | S |

Each family follows Q-Phase 3's five steps; Q-Phase 3's table gives the files. B1 already
replaced every family's start lock with the lease. B2 replaces `asyncio.create_task(run_x)` with
`task_queue.enqueue`, and deletes the family's `_tasks` dict and its reconcile entry. These are
different lines from B1's claim edit (churn item 3, accepted, Appendix A).

**Jobs longer than the visibility timeout (6 h):**
- the heartbeat thread keeps the lease;
- at 6 h Redis redelivers the message to another worker, which finds a live lease and acks it as
  a no-op;
- the original delivery keeps running and acks on completion (`acks_late`);
- tested in `test_job_task_wrapper.py` with a lowered timeout (Q-E1).

**Test cycle (once, after W3):**
- Unit:
  - the Q-Phase 2 tests (`test_job_task_wrapper.py`, `test_task_routes_complete.py`,
    `test_queue_consumers_from_compose.py`, `test_celery_config.py`, the last asserting the
    soft-shutdown settings);
  - per family, "enqueues once with `task_id=job_id`", and broker-down gives 503 with the record
    `failed`;
  - the Q-Phase 4 schedule tests, and the compose check that `jobs-beat` cannot scale.
- Full offline suite, pre-commit, contracts-check, docs check, frontend full set (2G).
- Live harness once, `-p op-b2-live`, GPU 0, budget 3 hours. Compose gains `redis`,
  `jobs-worker-a`, `jobs-worker-b` and `jobs-beat`. Run:
  - Q-Phase 3's race;
  - kill mid-job (the reaper resumes on b with `attempt=2`);
  - cancel running, cancel queued;
  - an API recycle at `--limit-max-requests=50`;
  - Q-Phase 4's arbiter check (one stop per container);
  - Q-Phase 5's two auto-label workers.
- Acceptance subset: `acceptance_run.sh --only` the job phases (ingest, cluster, holdout,
  label_ops, loop, export, train, bakeoff, promote) on the B2 stack.
- Trivy and the image-vs-git audit for the images B2 changes. Add the Redis image licence row
  to ATTRIBUTION (AGPLv3 option).

**Done when:**
- no job family runs as a task inside an API worker (CC-4 done-when);
- lifespan only creates per-process resources;
- the file-lock ratchet lists only the event-hub row and `job_lock.py`;
- CI green.

**Size:** XL (3 waves, up to 4 agents per wave).

**Rollback.** **B2 is forward-only.** Once the families' in-API runners are deleted, there is no
switch back. The rollback is reverting the B2 PR merge commit before v0.9.0 is tagged. After the
tag, users downgrade to 0.8.0 (Redis and the worker services are simply not started; job records
are compatible).

**Upgrade impact (0.8.0 -> 0.9.0):**
- `openprocessor upgrade` and `repair` create `secrets/queue/redis.conf` when it is missing (not
  only on fresh installs);
- they add the new services and env keys from `env.template`;
- **in-flight jobs**: the installer stops the API and lets it drain (`stop_grace_period`). In-API
  jobs end `interrupted`, and the resumable ones (D-25 list) are re-enqueued by the reaper on
  first start;
- the installer prints which jobs were interrupted;
- tested in `installer_upgrade` from 0.8.0 (R-0.9).

**Docs, contracts, CHANGELOG:**
- `docs/ARCHITECTURE.md` (services, queues table);
- `CLAUDE.md` (services table, Celery conventions);
- `INSTALLATION.md`, `docs/CURATION.md`, `SECURITY.md` (broker preconditions), `docs-site`;
- `contracts/` (`queued`, directory-ingest 202);
- `CHANGELOG.md` (root, "Breaking": new services) and `frontend/CHANGELOG.md`.

**Risks:**
- Q-E1..E26 (task plan section 8). The biggest are:
  - a redelivery storm (E2: attempt cap);
  - a cold-shutdown ack losing a message (E25: the reaper recovers from the record; the soft
    shutdown reduces it);
  - a route into a queue nothing drains (E18: compose consumer test);
  - Cropwright missing the `queued` state (E14: browser check).

### 7.3 B3: one Ultralytics version and one detection contract (v0.9.0)

**Dependencies:**
- B1 merged; decisions D-27 (U-J1, J5-J8 bundle) and D-28 (pin choice rule).
- B3 is independent of B2 in files (the export worker and trainer are not migrated). So its CPU
  work can run in parallel with B2 on `int/b3-ultralytics`, cut from main after B1. It rebases
  onto main after B2 merges, and merges second.
- The GPU gate needs a scheduled GPU window (section 13). It never overlaps B2's live harness.

**Scope:**
- **U-Step 4**: one export module, `export/ultralytics_export.py`.
  - It absorbs `inspect_fused_output` and the `head_kind` routing from B1-1E unchanged.
  - The export worker becomes one command; installer scripts, the Makefile and the CLI call it.
  - The committed YOLO11 config becomes `output0 [300,6]` FP32.
  - Export formats are renamed; run `make contracts`.
- **U-Step 5**: delete `End2EndNMSAdapter`, `END2END_OUTPUTS` and the `max() > 1.0` heuristic
  (GPU-14's final fix); the ingest primary takes the fused contract (U-C9).
- **U-Step 6**: the region detector drops `[B,5,N]` and the greedy NMS once region models are
  re-exported; fail closed otherwise.
- **U-Step 7**:
  - pin `ultralytics==<X>` (D-28) in every requirements file;
  - delete `/opt/venv-y11`, `requirements-export-y11.txt`, the re-exec, `src/ultralytics_patches/`
    and their licence lines;
  - **the trainer moves in this batch only**: `ULTRALYTICS_SHA` = the SHA of tag `v<X>` (U-J8),
    `trainer_fit.py::_export_onnx` passes `nms` by capability (D-12) and drops `half=False`;
    `half=` -> `quantize=` or none (U-C6);
  - delete `src/utils/pytorch_utils.py` if the dead-code ratchet agrees.
- **U-Step 8** docs, including GPU-18's EfficientNMS deprecation note in
  `docs/MIGRATION_TRITON_26.md`.
- **U-Step 9**: `src/domain/model_families.py`, the first slice of #368.
- **U-C11**: delete the installer's heredoc YOLO11 config.
- **U-J7**: the batch guard in the config writer.
- The engine stamp changes, so `repair` rebuilds (U-R3).
- **The `families` CI job** grows to the full U-6.1 matrix at the new pin.

**Test cycle (once):**
- the U-6.1 matrix at the new pin; full offline suite, pre-commit, contracts-check, docs check;
- **the U-6.2 GPU gate, run once for both the Stage A and Stage B engines** (steps 1-9, GPU 0, its
  own `-p op-b3-gpu`, budget one GPU day):
  - parity: |delta mAP50-95| <= 0.3, and at least 99 % detection agreement;
  - throughput at least 95 % at batch 1/8/32/64, with p99 within +10 %;
  - YOLO26/v10/v5u/v8/v9 subset mAP;
  - the promote path at the new pin (1 epoch, served `[300,6]`), which is the first GPU check of
    the trainer change;
  - embedding cosine >= 0.999 (the export image's pip set changes);
  - the original-YOLOv5 ONNX as the ingest secondary;
  - export image size about -0.9 GB, `pip check`, Trivy, image-vs-git audit.

**Fallback:**
- If Stage B misses the throughput gate, ship Stage A only: the patch kept on the single version,
  the venv deleted. Take the measurement to the owner (U-J1).
- Record the TensorRT plugin deprecation in the release notes.

**Done when:**
- one Ultralytics version in every image and requirements file;
- the pin test passes in CI and in each image;
- U-6.2 recorded in `docs/PERFORMANCE.md` and `docs/research/`;
- the chosen X and the matrix output recorded in the PR.

**Size:** L plus about one GPU day.

**Rollback:** revert the PR. Deployments that already rebuilt engines rebuild again with `repair`;
the engine stamp forces it.

**Upgrade impact:**
- deployments rebuild the default detector engine once (`openprocessor upgrade` triggers it
  through the stamp);
- region profiles that point at a raw-head engine fail closed until re-exported (a CHANGELOG
  "Breaking" entry with the command).

**Docs, contracts, CHANGELOG:**
- `README.md`, `docs/MIGRATION_TRITON_26.md`, `docs/image_size_audit.md`, `export/README.md`,
  `docs-site` training/advanced pages;
- a "Supported model families" table in `docs/ARCHITECTURE.md`;
- `contracts/` (export formats);
- `ATTRIBUTION.md` (vendored fork removed);
- `CHANGELOG.md` "Breaking".

**Risks:**
- U-R1, unrolled NMS throughput;
- U-R2, several releases a day (handled by the D-28 freeze rule);
- U-R3, engine rebuilds;
- U-R4, the FP16 class column;
- U-R8, probe vs served head for YOLO26 (decided in #368).

### 7.4 B4: stateless API (v0.9.0), then R-0.9

**Dependencies:** B2 and B3 merged. Decisions D-29 (Q-J7), D-30 (Q-J13 after the spike), D-31
(evaluator), D-32 (#439), D-33 (Q-J14).

| Track | Scope | Size |
|---|---|---|
| 4A fan-out | Q-Phase 6: `review_resolve` and `edit_undo` become one `mget` + one bulk of `update` actions with per-item `if_seq_no`/`if_primary_term` through the OCC helpers; a 409 per item -> `conflicts`; above `OP_BATCH_SYNC_MAX` -> 202 + job; the CC-10 evidence task first (peak in-flight with a stub VLM), then `label_batch` moves to a worker job or gets TaskGroup + Semaphore bounds in the worker; CC-7: the ingest data plane written as `asyncio.TaskGroup` under the J7 exemption, documented in `docs/design/engineering_standards.md`; the evaluator moves to `-Q gpu_eval -c 1` only if D-31 says so | M |
| 4B worker knob | Q-Phase 7: drop `--workers` from compose and the Dockerfile CMD (uvicorn reads `WEB_CONCURRENCY` when `--workers` is absent [U1]); `API_WORKERS` written by the installer (default per D-33); executor and Triton channel pools sized from totals / `API_WORKERS` (Q-D14's remaining part); ratchet: no `asyncio.create_task` and no new module-level lock/semaphore/mutable dict outside an allowlist; offline compose test | M |
| 4C events | **Step 0, the spike** (gate for the rest): Redis Streams in the B2 stack; 10,000 events/s for 60 s from 4 producers; 50 SSE consumers with `XREAD BLOCK`; kill and reconnect with `Last-Event-ID`. **Pass:** no lost or reordered event per stream, resume correct, p99 delivery < 250 ms, Redis memory bounded by `MAXLEN ~`. The owner decides D-30 on the result before the JSONL log is deleted. Then: `XADD` with approximate `MAXLEN` trimming, consumers `XREAD BLOCK` per SSE connection, event ids give SSE `id:` for resume (also FE-7's root cause); delete the JSONL log, its rotation and the last `flock`; delete `src/services/curation/job_lock.py`; ratchet baseline at zero | M |
| 4D #439 | measure the stages of `POST /curation/projects` with the #414 stage-timer pattern and post the numbers; then the D-32 option (recommended: creation becomes a `project_create` Celery job returning 202 with the record in `creating`, like combine; no request-path fan-out) | S-M |

**Test cycle (once):**
- full offline suite, pre-commit, contracts-check, docs check;
- frontend full set (4A conflict display, 4D `creating` state);
- live harness: the Q-section 4e diff test (the same scripted scenario with `API_WORKERS=1` and
  `=32`, outcomes diffed);
- **benchmarks:**
  - first, **retake the 32-worker ingest and search baseline on the B3 engine**. The baseline in
    `docs/PERFORMANCE.md` was taken on the 0.6.0 engine, so comparing against it would mix
    engines;
  - then the installer default, compared against that retaken baseline;
  - this validates the knob; it is not the B5 baseline.

**Then R-0.9 release acceptance** on merged main. It runs the full R3 procedure:
- the acceptance phases;
- the installer lifecycle (fresh, re-run, upgrade from 0.7.0 and from 0.8.0, rollback,
  uninstall);
- `tests/live`, frontend live, full system, visual results;
- all image audits, Trivy, screenshots at 1440 and 430 px.

Tag v0.9.0 only on the owner's word (D-34).

**Done when:**
- `rg "fcntl" src` is empty;
- no lifespan side effects;
- the 1-vs-32 diff is empty;
- R-0.9 PASS.

**Size:** L.

**Rollback:** revert the PR before the tag. 4C is the risky track: it reverts alone, and the
JSONL log comes back with its allowlisted `flock`.

**Upgrade impact:**
- `API_WORKERS` is written into `.env` on upgrade (default D-33);
- SSE clients reconnect once;
- the event history in the old JSONL log is not migrated (it is transient).

**Docs, contracts, CHANGELOG:**
- `docs/ARCHITECTURE.md` (events, workers), `docs/PERFORMANCE.md` (retaken baseline, knob),
  `INSTALLATION.md` (`API_WORKERS`), `docs/design/engineering_standards.md` (J7 exemption);
- `contracts/` (4A `conflicts`, 4D `creating`);
- both CHANGELOGs.

**Risks:**
- a lower default worker count reduces throughput (the benchmark decides the default);
- SSE rewrite regressions (browser check of every live page);
- Redis becomes load-bearing for the UI: `/health` shows `queue: down` as degraded, and the UI
  shows the outage (a refetch fallback only if the owner wants it).

### 7.5 B5: speed (v0.10.0), one PR per sub-batch

**Dependencies:** B4 merged (final engine, final worker model). Each sub-batch is its own PR,
because each is gated on its own measurement ("short tests when the gain is known"; one full
confirmation per bundle). #45's 5a baseline owns the reference numbers for every later gate.

| Sub-batch | Scope | Gate (thresholds) |
|---|---|---|
| 5a baseline | #45 public ladders: Open Images category subsets of 1k/10k/100k images and full COCO val+train; 1M by extrapolation first, a 1M run only on the owner's word. #268 sweeps: set B, profiles, two-card, CUDA graphs, `priority_levels`. GPU-6 recall on COCO val2017 (5k) with `scaleup` on and off, for D-35 | numbers recorded with host load; no default changes unless the gain beats 2x the run-to-run spread and parity passes |
| 5b decode once | GPU-3b (one oriented decode helper in `src/utils/`, all decoders adapted; behaviour change in the CHANGELOG) as step 0, then #40's resident-frame waves and #407 in its stated order (profile first; the C2 premise is unproven), GPU-12 inside #407's byte-parity gate | #40/#407 acceptance; parity: identical boxes within 1 px at the original resolution, embedding cosine >= 0.999 |
| 5c crops and OCR/faces | #219 `crop_max_side` (off first) with the P-vlm/P-sam/P-region/P-ocr gates; #220: uint8 OCR input, GPU-7 via `async_exec` in the OCR BLS (Triton's dynamic batcher batches; per-crop widths kept, so there is no bucket padding: the #220 comment showed padding fails P-ocr), padding-aware CTC or argmax in the graph; face alignment on resident frames; the GPU-19 evidence task folded into P-ocr; the #220 step 3 research (ledger #17/#18 leftover) | #219/#220 acceptance criteria |
| 5d processing plans | #67 phase 2+ (plan doc: `docs/design/processing_plans_plan.md`; ledger #17/#18 leftover); per-stage "not run" tests | #67 acceptance |

**Test cycle per sub-batch:**
- targeted tests and one full offline suite;
- the sub-batch's benchmark and parity gate on GPU 0 (and GPU 2 when agreed);
- one confirmation run;
- numbers in `docs/PERFORMANCE.md`.

**Size:** XL over 4 PRs.

**Rollback:** each sub-PR reverts alone. Defaults change only behind a passed gate.

**Upgrade impact:**
- engine rebuilds only if 5b or 5c change a model graph (the stamp handles it);
- otherwise none.

**Docs:** `docs/PERFORMANCE.md`, `docs/research/`, `docs/ARCHITECTURE.md` (GPU pipeline), both
CHANGELOGs.

**Risks:**
- host load from peers skews the numbers (record host load with every run; repeat only the
  disputed point);
- parity regressions (every gate compares to the 5a reference).

### 7.6 Side track: #42 small-GPU acceptance

- **Worktree:** create `<repo>-wt-fix42` from `main` when #42 runs (the earlier empty one was removed; branch `perf/42-small-gpu-acceptance`, no
  commits, at main).
- **When:** run it when the transcribe-app session frees GPU 1 (3080 Ti, 12 GB). Rebase it on main
  first.
- **Pass:**
  - `small` profile engines built with the small target;
  - Triton READY with peak memory < 11 GB;
  - the live suite passes;
  - set A throughput recorded;
  - embedding cosine p01 >= 0.995.
- **Results** go to `docs/PERFORMANCE.md` and #42.
- **It never blocks a batch.** If it has not run by the 0.7.0 tag, the release notes say the `small`
  profile is not yet verified on a 12 GB card (D-5).

### 7.7 Backlog (after B4; not batched yet)

- BA-6: one router per PR, typed responses, contracts (N15).
- Evidence tasks: OS-8, FE-6, T-6 (act only on a measured defect; N14).
- #368 remainder (per-family training adapters, generic single-class exporter) after B3's slice.
- **GPU-15 leftover:** "refuse such a model at profile validation instead of at decode time". The
  source is audit plan section 4.3, GPU-15 "Left over, optional ... Not scheduled". The decode
  path already fails closed, so this is backlog, low.
- **Out of scope for this plan, unchanged milestones** (section 10 gives the reasons): #416, #406,
  #402, #398, #370, #369, #367, #366, #365, #347, #84, #435, #436, #231, #403; parked research or
  epics #8, #346, #348, #400, #401.

---

## 8. Owner decision register (renumbered, deduplicated, ordered by when it is needed)

"Q-" = task queue plan, "U-" = Ultralytics plan. Each source judgment call appears once.

| ID | Decision | Recommendation | Consequence | Needed by |
|---|---|---|---|---|
| **D-1** | B0 scope | finish GPU-1, GPU-3a+13, GPU-2, OS-2, OS-1, OPS-1, CC-1 and BA-7 (in progress), add the uvicorn floor and the Ultralytics exact pin, then release prep on `fix/411-r2-round` before pushing | one gate cycle; OCR output changes (documented) | **B0, now** |
| **D-2** | Ultralytics pin in B0 (U-J2) | `ultralytics==8.4.197` in both requirements files now (what the rc images resolve); exact `==` everywhere from then on; the next move is B3 | R3 tests the pin that ships; one later bump instead of two | **B0, now** |
| **D-3** | Merge, then R3 | gates, push, CI, merge, then R3 on the merged main SHA (as R2) | what is tagged is what R3 tested; R3 fixes come as a small PR | B0 |
| **D-4** | Segmenter in R3 | exercise the segmenter tier with the stack's HF token on GPU 0 (GPU 2 if free and agreed) | closes R2's gap; about one hour of GPU | R3 |
| **D-5** | #42 timing | run when GPU 1 frees; not a tag blocker; the release notes state it if unrun | no wait on the peer session | R3 |
| **D-6** | Known issues in 0.7.0 | ship #447, CC-2 and CC-4 as CHANGELOG known issues (CC-4 mitigated by the jitter) | users see the gaps and the fix releases | B0 release prep |
| **D-7** | **Release train** (new) | option A (section 4): v0.8.0 = B1 after R-B1; v0.9.0 = B2+B3+B4; v0.10.0 = B5; rename milestones as in 9.3 | about 30 LIVE fixes and the registry race reach users months earlier; one extra acceptance subset | **before B1 starts** (milestone edits) |
| **D-8** | J-LEASE | OpenSearch lease docs in the SETTINGS role (folded or legacy), thread heartbeat every 15 s, TTL 120 s, cooperative fencing with `fence()` before writes (6.1). Alternatives: Redis `SET NX PX` (only after B2; Redis docs prefer Redlock); `O_EXCL` files (single host) | one OpenSearch round trip per start and per tick; a dead holder blocks restarts for at most 120 s; works before Celery; the same lease and thread serve the Celery wrapper | B1 |
| **D-9** | J-REG and upgrade | registry doc with OCC (max 5 attempts, cap 10,000 classes); fail closed with 409 `registry_not_migrated`; `openprocessor upgrade` runs the migration; the file is renamed `.migrated`; a downgrade note (6.2) | about 75 references change; your live projects migrate on upgrade (dry-run saved first) | B1 |
| **D-10** | B1-1E scope | minimal in B1 (the #447 `nms=False` + head check, `head_kind` routing, class by name, the YOLOv5 probe input); the trainer `nms`, the export module and the full matrix in B3 | no double rewrite; the trainer changes once, behind the GPU gate | B1 |
| **D-11** | U-J3 YOLOv5 scope | option b: bring-your-own original-YOLOv5 ONNX for the probe and secondary only, with real-ONNX tests; YOLOv5u for GPU serving. Option a (drop it) costs one PR deleting three modules | no `/detect` serving of original YOLOv5 | B1 |
| **D-12** | U-J4 `nms` by capability | `nms=False` when the head has `one2one_cv2` (YOLO26, v10), `nms=True` otherwise; class-agnostic, as today | the export path in B1, the trainer in B3 | B1 (export), B3 (trainer) |
| **D-13** | Q-J8 order | B1 leases and registry, B2 Celery, B3 Ultralytics, B4 stateless API, B5 speed | as planned | B1 |
| **D-14** | Commit the plans | DONE 2026-11-03 at the owner's request: all four plans committed on `fix/411-r2-round`, after the naming check (neutral paths) | public plan docs | done |
| **D-15** | Trainer claim (Q-P1.6) | create-only generation files (6.3) in B1; the API `training/jobs.py` `O_EXCL` create | the latent multi-replica bug closed with no file lock; S | B1 |
| **D-16** | Tag v0.8.0 | after R-B1 PASS and your yes | per the rule | end of B1 |
| **D-17** | Q-J1 Celery 5.6 | confirmed (already decided) | none | B2 |
| **D-18** | Q-J2 broker | Redis 8 (AGPLv3 option); Valkey only if a redistributor objects | one more image | B2 |
| **D-19** | Q-J3 default install | `redis`, `jobs-worker`, `open-vocab-worker`, `jobs-beat` in the core; Flower under `monitoring` | four more containers and one secret; curation routes work by default | B2 |
| **D-20** | Q-J4 delivery | `acks_late` + `reject_on_worker_lost` + lease + attempt cap 3 + soft shutdown 30 s | at-least-once delivery, made idempotent by the lease | B2 |
| **D-21** | Q-J5 job-state store (revised) | status/progress/cancel stay in `FileJob` files; exclusivity is the OpenSearch lease | single host for `/jobs` until a multi-host store is wanted | B2 |
| **D-22** | Q-J6 scope | migrate all API-hosted jobs, the periodic loops, auto-label and assist loop; keep the detection/VLM workers, trainer and export worker; the evaluator is D-31 | as planned | B2 |
| **D-23** | Q-J9 `jobs` concurrency | 4, tunable (`OP_JOBS_CONCURRENCY`) | | B2 |
| **D-24** | Q-J10 Redis durability | AOF everysec, no backup | an hour of queue loss at worst is recovered from the records by the reaper | B2 |
| **D-25** | Q-J11 auto-resume | only reprocess, dataset import, bulk ingest, combine, assist loop | the others end `interrupted` for a manual resume | B2 |
| **D-26** | Q-J12 recycling | keep `--limit-max-requests` with jitter (landed in B0) | | B2 |
| **D-27** | U-J1 staging + U-J5..J8 | one version; U-6.2 once for Stage A and B; ship B if it passes, else A. J5: keep `conf=0.25` baked for `nms=True`. J6: keep the Triton name `yolov11_small_trt_end2end`. J7: export batch = Triton `max_batch_size` = profile max, the writer refuses a mismatch. J8: trainer on the same release in the same batch | deployments rebuild the default detector engine once | B3 |
| **D-28** | B3 pin choice rule | X = the newest 8.4.x release that passes U-6.1 as of a **freeze date set at B3 kick-off** (the newest release at least 3 days old on that date); recorded in the PR; any later bump reruns U-6.1 and U-6.2 | deterministic for an agent | B3 |
| **D-29** | Q-J7 request fan-out | handler fan-outs become bulk OCC; the ingest data plane is a documented `TaskGroup` exemption | `/ingest` keeps its latency | B4 |
| **D-30** | Q-J13 event transport | Redis Streams, **decided after the B4-4C spike passes**; alternative: keep the JSONL log with an allowlisted `flock` (violates the rule; not recommended) | the last file lock goes | B4 (after the spike) |
| **D-31** | Q-J6 evaluator | move the bake-off evaluator to `-Q gpu_eval -c 1` in B4; alternative: keep the file protocol | removes its at-least-once re-run | B4 |
| **D-32** | #439 approach | measure first; then project creation as a Celery job (202, `creating`); alternative: keep it synchronous if one slow step explains the time | | B4 |
| **D-33** | Q-J14 API workers | one knob `API_WORKERS` (`WEB_CONCURRENCY`), installer default `min(nproc, 32)`, confirmed by the B4 benchmark against the retaken baseline; `API_WORKERS=1` + replicas needs a multi-host job store first | | B4 |
| **D-34** | Tag v0.9.0 | after R-0.9 PASS and your yes | | end of B4 |
| **D-35** | GPU-6 `/detect` upscaling | keep `scaleup=False` until 5a measures recall with and without it on COCO val | decided on numbers | B5 |
| **D-36** | GitHub actions (section 9) | approve the new issues, re-scopes, milestone renames and closes as listed | the coordinator executes them | before B1 (B0 closes after the merge) |

---

## 9. GitHub issue actions (for the coordinator, after the owner's review; nothing is executed by this plan)

### 9.1 New issues to create

Titles are final. Bodies are given in brief; expand them with the source-plan section as the
"Plan" link. Security issues stay neutral: no mechanism, no path, no exploit wording. Fewer issues
is better: bugs fixed inside a batch are referenced in the PR, not filed (memory: "fix bugs as you
go").

| # | Title | Body in brief | Labels | Milestone (after 9.3) | Batch |
|---|---|---|---|---|---|
| N1 | `epic(jobs): run background jobs on Celery workers and make the API stateless` | Jobs started by a request run inside API worker processes and die when a worker recycles; periodic loops run once per worker. Move them to Celery 5.6 on Redis 8 (job record and OpenSearch lease are the truth, the broker is transport), one beat, then make the worker count a tuning knob. Plan: `docs/design/task_queue_plan.md`. Children: B2 waves, B4 tracks. Done when no job runs in an API worker and the 1-vs-32 worker diff is empty | epic, area:curation, effort:hard | v0.9.0 | B2-B4 |
| N2 | `fix(curation): class registry moves to an OpenSearch document with optimistic concurrency` | Two concurrent class writes (two users, or a user plus an import or combine) can lose a class or give two classes one id, because the registry file is a read-modify-write across processes. Move the authority to one document per project written with `if_seq_no`/`if_primary_term`; the file becomes an export snapshot. Includes the upgrade migration (`openprocessor upgrade`, fail-closed 409 `registry_not_migrated` until migrated, file kept as `.migrated`). Plan: `codebase_audit_plan.md` CC-2 | bug, area:curation, effort:hard | v0.8.0 | B1-1B |
| N3 | `fix(curation): per-project job leases in OpenSearch replace start file locks` | Singleton job starts are guarded by file locks (and four starts have no guard), and stale-job repair can flip a sibling's live job. Replace them with lease documents (create-only / OCC, thread heartbeat, TTL 120 s, cooperative fencing), add a shrink-only ratchet on file locks that reaches zero when the event log moves (B4) | bug, area:curation, effort:hard | v0.8.0 | B1-1A (ratchet to B4) |
| N4 | `fix(api): per-process caches, response bodies and blocking image work across API workers` | Class-name, settings, query and eval-dataset caches are per worker and can serve stale or cached-error results; curation bodies carry undeclared timing keys; 500 bodies echo an exception class; thumbnail and render handlers block the event loop. Audit BA-1..5, BA-8, CC-5 (handlers) | bug, effort:moderate | v0.8.0 | B1-1D |
| N5 | `fix(inference): memory, retry, timeout and readiness hygiene in the Triton clients and search` | Sliver crops allocate GBs, decode happens before the size check, deterministic errors are retried, sync calls lack timeouts, residency never learns of an unload; the kNN filter is a post-filter, paging is unbounded, scroll contexts leak on error, PIT slices orphan. Audit GPU-4/5/8/9/10/16/17, OS-3/5/6/7/9/10, CC-6, CC-8 | bug, area:gpu-serving, area:search, effort:moderate | v0.8.0 | B1-1C (OS-3 in 1A) |
| N6 | `fix(frontend): diverse review uses one batch read; export stats refresh; SSE resync; tests that cannot fail` | Diverse review fans out 30 crop fetches (use the existing `?ids=` route); the export page's stats never refresh; preload binds the project in `load`; SSE reconnect does not resync keymap/config; some tests cannot fail or skip on the failure they guard; a test dependency is missing. Audit FE-1/2/3/5/7, T-1/2/3/4/7 | bug, area:frontend, effort:moderate | v0.8.0 | B1-1D |
| N7 | `fix(models): secondary detector class by name, and original YOLOv5 ONNX for the probe and secondary` | The ingest secondary detector stores a raw model index as the class id (breaks class identity by name); the YOLOv5 objectness probe hard-codes a 1280 input. Fix both; document bring-your-own original-YOLOv5 ONNX (exported by the official repo) for the probe and secondary, tested on real exports. Plan: `model_family_versions_plan.md` Step 3, J3 | bug, area:ingest, effort:moderate | v0.8.0 | B1-1E |
| N8 | `build(ultralytics): one exact Ultralytics version and one fused [batch,300,6] detection contract` | Three Ultralytics versions ship (API, an isolated YOLO11 venv with a vendored patch, the trainer at a git SHA). Move every image to one exact release, export every family through the documented `nms` argument to one fused head, delete the venv and the patch behind a GPU gate (Stage A/B), trainer on the same release. Plan: `model_family_versions_plan.md` Steps 4-9 | build, area:gpu-serving, area:training, effort:hard | v0.9.0 | B3 |
| N9 | `security: harden persisted clustering state` | Neutral. Details in the maintainers' private notes | security, effort:quick | v0.8.0 | B1-1D (SEC-3) |
| N10 | `security: widen the CLI secret redaction key list` | Neutral | security, effort:quick | v0.8.0 | B1-1D (SEC-6) |
| N11 | `security: reduce the privileges of an opt-in monitoring service` | Neutral | security, effort:quick | v0.9.0 | B2 (SEC-4) |
| N12 | `security: pin a build input and review per-service container hardening` | Neutral | security, effort:moderate | v0.9.0 | B2 (SEC-5) |
| N13 | `question(detect): should /detect upscale small images like ingest does` | `/detect` letterboxes without upscaling (`scaleup=False`, Ultralytics validation behaviour) while ingest upscales. Measure recall with and without on COCO val2017, then decide. Audit GPU-6 | question, area:gpu-serving | v0.10.0 | B5a |
| N14 | `chore(audit): evidence tasks before acting` | Measure first, act only on a defect: delete-by-query failures (OS-8), OCR detector normalisation (GPU-19), pager memory (FE-6), FCFS test timing (T-6), Prometheus files of recycled workers (OPS-2, before B2), VLM chunk fan-out peak (CC-10, before B4). One checkbox each with its batch | chore, effort:moderate | v0.10.0 (items close in their batches) | backlog / B2 / B4 / B5 |
| N15 | `chore(api): typed response models, one router per PR` | 71 operations have no response model. One router per PR after B4 so rewritten routers are typed once; contracts regenerate per PR. Audit BA-6 | chore, effort:hard | v0.10.0 | backlog |

Dropped from the first plan's list (Appendix A explains each):
- N4 "ratchet" (folded into N3);
- N10b (folded into the #447 re-scope);
- N14 decode (folded into the #40 re-scope);
- N15 OCR concurrency (folded into the #220 re-scope);
- N20 B0 traceability (B0 fixes are referenced in the PR and the CHANGELOG instead).

### 9.2 Re-scopes (corrected premises; post as a comment and edit the body's scope)

| Issue | Corrected premise and new scope |
|---|---|
| #447 | Root cause: in Ultralytics 8.4, `nms` unset (`None`) exports the raw one-to-many head ("`None` exports raw one-to-many predictions for external NMS", export docs). The checkpoint from the 8.4.48 trainer still gave `[300,6]`; 8.4.174+ do not. `end2end` is deprecated and mapped to `nms` (maintainer comment on the export page), which explains the `end2end=True` discrepancy in the version runs. Fix: `nms=False` plus a head check that refuses anything but `[300,6]`. The row "`yolo26n.pt` + `nms=False` -> raw head" did not reproduce (U-C1). **Also in scope:** a YOLO26 upload is routed to the YOLO11 toolchain (U-C2), fixed by routing on the head kind. Milestone v0.8.0 (B1-1E). The trainer's export moves with N8 in B3 |
| #439 | The concurrent index creation was reverted; the 13-20 s is not attributed yet. Next: stage timings with the #414 timer; then creation as a job returning 202 (`creating`), not request-path fan-out. Milestone v0.9.0 (B4-4D) |
| #42 | The `small` profile shipped (#441, CHANGELOG "Added"). Only the 12 GB-card acceptance remains, waiting for GPU 1. Milestone stays perf (v0.10.0 after the renames); the 0.7.0 notes say unverified if it has not run |
| #368 | B3 delivers the family registry module (`src/domain/model_families.py`) used by the probe, validator and exporter. The issue keeps the per-family training adapters and the single-class exporter. Add U-R8 (probe head vs served head for YOLO26). Milestone becomes v0.11.0 by the rename |
| #220 | Width-bucket padding fails P-ocr (comment of 2026-11). Next: GPU-7 `async_exec` in the OCR BLS (no padding) or a padding-aware decode / argmax in the graph; uint8 input; face alignment on resident frames; the step 3 research. Absorbs the former "OCR crops concurrently" issue. B5c |
| #407 | C2's launch-overhead premise is unsupported; profile first. Add GPU-12 (half-pixel letterbox undo) to its parity gate. B5b |
| #40 | The child list gains GPU-3b (one oriented decode helper for `/embed`, `/analyze` and legacy ingest) as step 0. B5b |
| #411 | Checklist updated: add the B0 small fixes, the Ultralytics pin, R3's extra checks (5.3), the tag decision; closes after the tag |
| #366 | Re-check after B0 merges: #440 delivers choosing and activating an ingest profile in Cropwright on the existing config store. If the stored per-project `ingest_profile` kind already exists, close #366 as done by #440; else keep it with the remainder |

### 9.3 Milestones (titles and descriptions)

Apply in this order (renames carry their issues, so no issue is edited for the move):

1. Rename `v0.10.0` to **`v0.11.0`**. Description: "Platform breadth: video analytics, model
   families (training adapters), second segmenter, importers, full API authentication, embedding
   swap."
2. Rename `v0.9.0` to **`v0.10.0`**. Description: "Performance and the GPU pipeline on the final
   0.9 stack: #45 baseline first, then decode once (#40/#407), crops at model size (#219), OCR and
   faces on resident crops (#220), processing plans (#67), Triton follow-ups (#268), 12 GB GPUs
   (#42)."
3. Edit `v0.8.0`. Description: "Correctness foundation: OpenSearch job leases, class registry
   with OCC and upgrade migration, inference/search/API/frontend/test fixes from the 0.7.0 audit,
   Ultralytics live bugs (#447)." Replace the stale "Graph-based global cluster refinement"
   text; that clustering work is tracked in #346 (parked).
4. Create **`v0.9.0`**. Description: "Jobs on Celery workers, one Ultralytics version and one
   detection contract, stateless API (worker count is a tuning knob), events on Redis Streams."
5. Edit `v0.7.0`. Description: "Release close-out: R2 round fixes, OCR geometry/orientation/unclip,
   holdout freeze fails closed, create-only config writes, unique temp files, recycle jitter,
   exact Ultralytics pin; R3 acceptance; tag on the owner's word." Replace the stale perf ordering
   text, which now lives in v0.10.0.

Then set the milestones of the new issues (9.1), and move #447 to v0.8.0 and #439 to v0.9.0.

### 9.4 Close

| When | Issues | How |
|---|---|---|
| B0 PR merges | #437, #440, #442, #443, #445 | "Closes #n" lines in the PR body; verify each closed |
| v0.7.0 tagged | #411 | comment with the release link and the R3 report path |
| v0.8.0 tagged | #447, N2..N7, N9, N10 | PR closing keywords; verify |
| v0.9.0 tagged | N1, N8, N11, N12, #439 | |
| B5 sub-PRs | #45, #268, #40, #407, #219, #220, #67, N13 | as each acceptance is met |
| side | #42 | after the run |
| obsolete now | none. Re-check after B3: if Stage B ships, any request for EfficientNMS-specific work is obsolete. Re-check #366 after B0 (9.2) | |

---

## 10. Issue map: every open issue (37) to a batch, a deferral or a cut

| Issue | Milestone today | Disposition | Reason |
|---|---|---|---|
| #8 epic(video) | none (parked) | **defer**, out of scope | platform breadth; needs the stable job system (B2) and the GPU pipeline (B5) first. Set milestone v0.11.0 or keep it parked |
| #40 decode once (epic) | v0.9.0 -> v0.10.0 | B5b | speed; re-scoped (9.2) |
| #42 small GPU | v0.9.0 -> v0.10.0 | side track | waits for GPU 1 |
| #45 benchmark ladders | v0.9.0 -> v0.10.0 | B5a | baseline before optimization |
| #67 processing plans | v0.9.0 -> v0.10.0 | B5d | ledger #17/#18 "phase 2+" leftover |
| #84 wheel example | v0.10.0 -> v0.11.0 | defer | example; after training stabilises (B3) |
| #219 crops at model size | v0.9.0 -> v0.10.0 | B5c | |
| #220 OCR/faces | v0.9.0 -> v0.10.0 | B5c | re-scoped; step 3 research included |
| #231 ortloom vs Triton | v0.9.0 -> v0.10.0 | defer (owner gate: after #40 lands) | owner decision 2026-10-10 |
| #268 Triton follow-ups | v0.9.0 -> v0.10.0 | B5a | |
| #346 clustering research tracker | none (parked) | **defer**, out of scope | research tracker; graph clustering is not in this plan; per memory, never name the private runner in it |
| #347 embedding-model research | v0.10.0 -> v0.11.0 | defer | research |
| #348 spatial research | none (parked) | **defer**, out of scope | after 0.7.0 and the clustering work; long-term |
| #365 embedding swap | v0.10.0 -> v0.11.0 (blocked) | **defer** | interplay with B1 noted (6.2): it reuses the upgrade-hook pattern; no B1 work |
| #366 ingest_profile kind | v0.10.0 -> v0.11.0 | re-check after B0 (9.2) | #440 may already cover it |
| #367 second segmenter | v0.10.0 -> v0.11.0 | defer | platform breadth |
| #368 model-family registry | v0.10.0 -> v0.11.0 | first slice in B3; remainder deferred | re-scoped |
| #369 API auth | v0.10.0 -> v0.11.0 (blocked) | defer | blocked; B2 adds no public surface |
| #370 importers | v0.10.0 -> v0.11.0 | defer | platform breadth; uses the B1 registry (class by name) when built |
| #398 detector-class roots | v0.10.0 -> v0.11.0 | defer | research |
| #400 attribute signals | none (parked) | **defer**, out of scope | parked clustering research (#346 family) |
| #401 plate identity | none (parked) | **defer**, out of scope | parked; privacy design needed |
| #402 similar-crop search | v0.10.0 -> v0.11.0 | defer | feature; after B4 |
| #403 MPS overlay | v0.9.0 -> v0.10.0 (parked) | defer | parked; adopt only with >= 10 % gain, measured in B5 |
| #406 judge calibration | v0.10.0 -> v0.11.0 | defer | research |
| #407 GPU-prep speed | v0.9.0 -> v0.10.0 | B5b | re-scoped |
| #411 0.7.0 close-out | v0.7.0 | B0 | closes after the tag |
| #416 near-dup threshold | v0.10.0 -> v0.11.0 | defer | research |
| #435 frame-target search | none | defer, set v0.11.0 | follow-up to #402; decide if it belongs in curation search |
| #436 group_by=image | none | defer, set v0.11.0 | not needed by the current UI |
| #437 YOLO26 region decode | none | B0 (on the branch) | closes on merge |
| #439 project create 13-20 s | none | B4-4D | re-scoped; set v0.9.0 |
| #440 Cropwright ingest profile | none | B0 (on the branch) | closes on merge |
| #442 VLM activate not_found | none | B0 (on the branch) | closes on merge |
| #443 cluster refresh mid-ingest | none | B0 (on the branch) | closes on merge |
| #445 COCO test images | none | B0 (on the branch) | closes on merge |
| #447 YOLO26 raw head | none | B1-1E | re-scoped; set v0.8.0 |

Cut outright: none. Every open issue keeps a home or a stated deferral.

---

## 11. Task list (open tasks mapped to batches)

The first column is the ledger/TaskList number where one exists.

| Task | What | Batch | Status | Blocked on |
|---|---|---|---|---|
| #28 | B0: combine the fixes, gates, push, merge | B0 | in progress (fix branches) | host all-clear for heavy gates; owner yes to push/merge |
| B0-a | GPU-1, GPU-3a+13, GPU-2 (`b0/ocr-fixes`) | B0 | merged into the integration branch | - |
| B0-b | OS-2, OS-1, OPS-1 (`b0/data-ops-fixes`) | B0 | committed | - |
| B0-c | CC-1, BA-7 (`b0/data-ops-fixes`) | B0 | being implemented | - |
| B0-d | uvicorn `>=0.41.0` floor + test (OPS-1 follow-up) | B0 | to do | D-1 |
| B0-e | `ultralytics==8.4.197` exact pin + `tests/test_ultralytics_pin.py` | B0 | to do | D-2 |
| B0-f | release prep: VERSION 0.7.0, CHANGELOG entries and known issues | B0 | to do | B0-c..e |
| B0-g | OCR before/after comparison (`artifacts_local/b0/ocr/`) | B0 gate | to do | host all-clear, GPU 0 |
| #25 | R3 live acceptance (5.3) | B0 | to do | B0 merged |
| #22 | segmenter tier with HF token in R3 | B0 (R3) | to do | R3 |
| B0-h | tag v0.7.0 + GitHub release | B0 | to do | R3 PASS, owner yes (D-6 is release prep, D-5 notes) |
| #27 | #42 small-GPU acceptance on GPU 1 | side | waiting | transcribe-app-1d frees GPU 1 |
| GH-1 | execute section 9 (issues, re-scopes, milestones) | before B1 | to do | owner review (D-36) |
| #31 | audit plan | done (doc) | committed at the start of B1 (D-14) | - |
| #34 | Ultralytics plan | done (doc) | committed at the start of B1; header SHA line corrected | - |
| B1-0 | commit the source plans on `int/b1-correctness` | B1 | to do | B0 merged |
| B1-1A..1E | tracks as in 7.1 | B1 | to do | D-8..D-12, D-15 |
| R-B1 | 0.8.0 acceptance subset + upgrade path | B1 | to do | B1 merged |
| B2-W1..W3 | as in 7.2 | B2 | to do | B1 merged; D-17..D-26 |
| B3 | as in 7.3 | B3 | to do | B1 merged; D-27, D-28; GPU window |
| B4-4A..4D, R-0.9 | as in 7.4 | B4 | to do | B2, B3 |
| #17/#18 leftovers | #67 phase 2+ (B5d), #220 step 3 research (B5c) | B5 | to do | B4 |
| B5a..d | as in 7.5 | B5 | to do | B4 |
| CL-* | cleanup rows (section 14) | as listed | to do | per row |

---

## 12. Test strategy across batches (no cycle repeated needlessly)

| Tier | What | Where it runs | Never repeated because |
|---|---|---|---|
| T0 | a red test seen failing + a paired control, per task | every task, in its worktree | it is the task's own evidence |
| T1 | pre-commit, full offline suite (25-33 min), contracts-check, docs check, changelog rules; the frontend full set only when `frontend/` changed | once per batch, on the integration branch before the push | tracks merge first, then one run |
| T2 | live harness (`docker/test/compose.yml`, its own `-p`), with a time budget per batch | B1 (lease/registry live, 90 min), B2 (jobs cross-process, kill/cancel/recycle, 3 h), B4 (1-vs-32 diff) | only batches that change cross-process behaviour or topology |
| T3 | GPU gates | B0 (one OCR before/after), B3 (U-6.2 once for both stages, about 1 day), B5 (per sub-batch) | B3 runs the Stage A and B engines in one window; the embedding cosine gate runs only in B3 (the export pip set changes) |
| T4 | release acceptance | **R3** after B0 (full), **R-B1** after B1 (subset + `installer_upgrade` from 0.6.0 and 0.7.0), **R-0.9** after B4 (full) | B2 runs only an `--only` subset of the job phases |
| T5 | benchmarks | B3 throughput gate, B4 worker knob (32-worker baseline retaken on the B3 engine first), B5 baselines and gains | B5a's baseline is taken once on the final stack |
| CI | GitHub Actions on the pushed PR, including the `families` job from B1 | every batch PR | local parity first (memory: never discover failures in CI); verify with `git ls-remote` and the run |

**Image hygiene.** The image-vs-git audit and Trivy run whenever an image is built (B0/R3,
R-B1, B2, B3, R-0.9) and are never skipped before a push (memory: the 0.6.0 image leak).

**Scheduling.** Heavy tiers (T1 with `-n auto`, Playwright e2e, T2, T3, T4, T5) are announced to
peer sessions first and run one at a time (section 13).

---

## 13. Host, GPUs and peer sessions

- **GPU 0** (A6000): this repo's live and GPU runs.
- **GPU 2** (A6000): this repo's by the standing owner decision. Check `nvidia-smi` and ask peer
  sessions (ListAgents) before using it; another session has asked for it.
- **GPU 1** (3080 Ti): only when the OpenTranscribe session (transcribe-app-1d) says it is free;
  used only for #42.
- **Heavy runs** (full offline suite with `-n auto`, Playwright e2e, image builds, acceptance,
  GPU gates, benchmarks):
  - announce them to peer sessions first;
  - today they are paused until transcribe-app-1d messages "done" for the OpenTranscribe 0.6.1
    release.
- Never touch another session's containers, volumes or networks. Snapshot before and after every
  acceptance (R2's method).
- Always run `docker compose -p <name>`: the default project name is the live stack. From a
  worktree, always pass `-p`.
- Waiting: use healthchecks or the Monitor tool. No `pgrep -f`, `pkill -f` or sleep loops (and
  say so in every agent brief).
- One heavy run at a time from this repo. B3's GPU gate and B2's live harness must not overlap.

---

## 14. Cleanup plan (verify contents first; ask before deleting anything not listed as safe)

| Item | State (checked 2026-11-03) | Action | When |
|---|---|---|---|
| `<repo>-wt-r2round` + branch `fix/411-r2-round` | integration branch, unpushed | `git worktree remove`; delete the branch after the PR merges | after the B0 merge |
| `b0/ocr-fixes`, `b0/data-ops-fixes` branches and their `.claude/worktrees/agent-*` worktrees (one is **locked**) | B0 fix branches | unlock, remove the worktrees, delete the branches once merged into the integration branch | after the B0 merge |
| `.claude/worktrees/agent-a0103bac0b3dc70fe` (`fix/model-route-hardening`, **locked**), `agent-a2bacbb8a217a8320` (`review/backend`, **locked**), `agent-a11c580f104ff4b6a` (`review/installer`), `agent-a21206080a3c73a2c` (`review/loop`), `agent-aa442585ee8216ee1` (`review/frontend`) | all contained in `fix/411-r2-round` | `git worktree unlock` (locked ones), `git worktree remove`, delete the branches incl. `worktree-agent-a0103bac0b3dc70fe` (already in main). Re-list with `git worktree list` first; some may be gone | after the B0 merge |
| `.claude/worktrees/audit-r2` (detached at `5b95591f`) | line references of the source plans | keep until B1 has re-found its sites, then remove | end of B1 |
| `<repo>-wt-b7` (`chore/v080-batch7`) | merged into main | DONE 2026-11-03: worktree and branch removed | done |
| `<repo>-wt-fix42` (`perf/42-small-gpu-acceptance`, no commits) | DONE 2026-11-03: removed (no commits); recreate from `main` when #42 runs | done |
| branch `perf/220-width-buckets` (1 commit, not merged) | evidence cited in #220 | keep until B5c decides | B5c |
| `<scratch>/b8/deploy` (detached worktree at `04536b3c`) | old deploy | check for local changes, then remove | now, after a look |
| `<scratch>/fixa`, `<scratch>/r1`, `<scratch>/r2` (incl. `r2/install`, `r2/release`) | acceptance scratch; evidence already in `artifacts_local/r1`, `r2` | list the contents, confirm the evidence is copied, then remove; keep `r2/release` until R3 has its own bundle | after R3 |
| `op-r1` network `op-r1_triton_net` and `op-r1_*` volumes (10) | R1 leftovers | confirm no container uses them, then `docker network rm` / `docker volume rm` | now, after the check |
| root-owned empty dirs (`secrets/vlm` remnants, `<scratch>/op-install-2`) | need sudo | ask the owner to run the removal | owner |
| images `openprocessor-local/*:0.7.0-rc3`, `latest-rc3`, `:b8` | older builds | remove after R3's rc6 images exist | after R3 |
| images `openprocessor-local/*:0.7.0-rc5` and `latest-rc5` | R2 build; source of the 8.4.197 / uvicorn 0.54.0 reading | keep until R3's rc6 images exist; then remove | after R3 |
| `artifacts_local/model_versions` (Ultralytics matrix scratch) | evidence copied to `artifacts_local/model_versions/` per the ledger | verify the copy, then remove | now, after the check |
| R-B1 and R-0.9 scratch (`<scratch>/r-b1`, `r-0.9`) and their `op-r-*` projects | future | same rule as R3 | after each tag |
| bake-off outputs | some back the paper | never bulk-delete | n/a |

Never delete the archived repos (listed in the owner's private repo map).

---

## 15. Risks and edge cases

| # | Risk / edge case | Mitigation |
|---|---|---|
| M1 | Host load from peer sessions delays the B0 gates and R3 | wait for the all-clear; announce heavy runs; one heavy run at a time |
| M2 | R3 finds a regression in the large integration branch | `fix/r3-findings`, targeted tests, re-run only the failed phases |
| M3 | OCR changes (GPU-1/2/3a) alter downstream text consumers | before/after on public images in `artifacts_local/b0/ocr/`; the CHANGELOG states the behaviour change; revert the three OCR commits alone if rejected |
| M4 | A lease holder pauses longer than the TTL (GC, a blocked thread) and a reaper takes over | the thread heartbeat survives a blocked loop; TTL 120 s; `fence()` before every write; residual one-write window documented; records carry `lease_generation` |
| M5 | OpenSearch outage blocks every job start | fail closed with 503 (rule); `/health` already reports OpenSearch |
| M6 | Lease doc lost with a project delete while a job runs | the job's next renew is `Lost`, and it stops; delete already waits for or cancels jobs |
| M7 | Registry migration on the owner's live projects | idempotent CLI, dry run saved, the file renamed not deleted, ids preserved, fail-closed 409 until it runs, a downgrade note |
| M8 | A reader still reads `class_registry.json` after B1 | the B1 done-when greps for it outside export snapshots and the trainer; the dead-code and layering ratchets |
| M9 | Celery redelivery, cold-shutdown ack, dead queues, wedged pools | Q-E1, E25, E18, E23 mitigations (lease, reaper from the records, soft shutdown, compose consumer test, pool healthcheck) |
| M10 | `queued` status breaks Cropwright | contracts + 2G + browser check |
| M11 | An install upgraded to B2 lacks the Redis secret | the installer creates it on `upgrade` and `repair`; tested in `installer_upgrade` |
| M12 | Ultralytics ships a breaking patch between U-6.1 and the image build | exact pin from B0; the pin test in CI and in each image; the D-28 freeze rule |
| M13 | Stage B throughput miss (unrolled NMS at batch 64) | Stage A fallback; the owner sees the measurement |
| M14 | The deployed default detector engine becomes incompatible after B3 | the adapter fails closed; the engine stamp changes so `repair` rebuilds; CHANGELOG "Breaking" |
| M15 | The `API_WORKERS` default lowers throughput | the B4 benchmark against the 32-worker baseline retaken on the B3 engine decides the default |
| M16 | Redis Streams change SSE ordering or resume | the 4C spike gates it; event ids give ordered resume per stream; browser check of every live page |
| M17 | A security item's detail leaks into a public file or issue | neutral titles and bodies only; naming-leak scan and a manual read of each diff and issue that touches SEC |
| M18 | An image ships untracked files | image-vs-git audit before every push of an image |
| M19 | Contract drift between the tracks of one batch | `make contracts` per commit; contracts-check once on the integration branch; tracks that change wire models merge last in their wave |
| M20 | Line numbers in the source plans have moved | always re-find with the given `rg` pattern |
| M21 | A large single B1 PR fails the live harness in one track | one merge commit per track; revert that track's merge and ship the rest (1B is coupled to 1A) |
| M22 | A project with more than 10,000 classes | `registry_full` 409; per-class docs would be a separate design (not planned) |
| M23 | Users stay on 0.7.0 through B1 | mitigated by the early 0.8.0 (D-7) |
| M24 | Cropwright touched in B1, B2 and B4 | B1 ships in 0.8.0; the B2 and B4 wire changes ship together in 0.9.0 (one frontend release) |
| M25 | 0.6.0 -> 0.8.0 direct upgrade misses a 0.7.0 step | R-B1 runs `installer_upgrade` from 0.6.0 as well |

**Unverified (be honest):**
- Nothing in this plan was run (no tests, stacks, engines, npm or browsers) except reading package
  versions in the local rc5 images with `importlib.metadata`.
- The lease and registry designs are designs. Their cost per heartbeat and their behaviour under
  OpenSearch load are argued, not measured; the B1 live tests measure them.
- The branch's own tests were not run by this plan. "On the branch" means the commit is present.
- R2's suspected loop-accounting oddity is believed fixed by `898d7075` (admission by full claim);
  R3 verifies it.
- The Redis Streams event log (D-30) has no spike yet; B4-4C starts with one.
- U-6.2 was never run: TRT builds, Triton serving, embedded-NMS throughput at batch 64, and
  checkpoints from the 8.4.48 trainer.
- The YOLO26 `end2end=True` + `simplify` discrepancy is explained only by the maintainer's
  "`end2end` deprecated, mapped to `nms`" comment. Checking the head after every export stays the
  safeguard.
- #439's cause is not attributed.
- The allocation of GPU 2 is a standing decision with an open request from another session.
- The OpenTranscribe repo was not re-read for this revision. The alignment rests on the
  task-plan section 4d citations, and B2-W1 re-reads them.
- COCO-Text v2's current download location and licence were not re-checked (5.2 step 7 has a
  fallback rule).
- The R3 images do not exist yet. 8.4.197 is what the rc5 images resolved; the B0 pin makes R3
  resolve the same version by construction.

---

## 16. Docs touched per batch (summary)

| Batch | ARCHITECTURE | CURATION | opensearch_schema_design | INSTALLATION | curation_api_contract + contracts/ | PERFORMANCE | other | CHANGELOG |
|---|---|---|---|---|---|---|---|---|
| B0 | - | - | - | - | - | jitter | ATTRIBUTION (pyclipper) | root (behaviour changes, known issues), frontend (#440) |
| B1 | leases, registry | `job_busy` window, `registry_not_migrated`, migrate command | lease + registry docs, mapping | upgrade, downgrade note | BA-2, BA-5, new error codes | - | CLAUDE.md lock rule, ATTRIBUTION (cachetools) | root ("Breaking"), frontend |
| B2 | services, queues | job states, `queued` | - | Redis, new services | `queued`, directory-ingest 202 | - | CLAUDE.md, SECURITY.md, docs-site, ATTRIBUTION (Celery, Redis) | root ("Breaking"), frontend |
| B3 | supported model families | - | - | engine rebuild | export formats | U-6.2 results | README, MIGRATION_TRITON_26, image_size_audit, export/README, docs-site | root ("Breaking") |
| B4 | events, workers | - | - | `API_WORKERS` | `conflicts`, `creating` | retaken baseline, knob | engineering_standards (J7) | root, frontend |
| B5 | GPU pipeline | - | - | - | per sub-batch | all gates | docs/research | root |

`check_docs_vs_code.py` runs in every batch gate.

---

## 17. Sources (official docs; fetched 2026-11-03 unless a source plan cites them)

- [O1] OpenSearch index document API: `op_type=create` "indexes a document only if it does not
  already exist"; an `if_seq_no`/`if_primary_term` mismatch returns 409:
  https://docs.opensearch.org/latest/api-reference/document-apis/index-document/
- [O2] OpenSearch Job Scheduler locks (a lock document with seq_no/primary_term):
  https://docs.opensearch.org/latest/monitoring-your-cluster/job-scheduler/locks/
- [V5] Redis `SET` (NX, PX; "The following pattern is discouraged in favor of the Redlock
  algorithm"): https://redis.io/docs/latest/commands/set/
- Redis Streams (`XADD` trimming with `MAXLEN ~`, `XREAD BLOCK`):
  https://redis.io/docs/latest/develop/data-types/streams/
- [U1] Uvicorn settings: `--workers` "Defaults to the `$WEB_CONCURRENCY` environment variable if
  available, or 1"; `--limit-max-requests-jitter` staggers restarts:
  https://github.com/encode/uvicorn/blob/master/docs/settings.md
- [U2] Uvicorn 0.41.0 release notes: "Add `--limit-max-requests-jitter` to stagger worker restarts
  (#2707)": https://github.com/encode/uvicorn/releases/tag/0.41.0
- Python `asyncio` Task Groups and `asyncio.to_thread`: https://docs.python.org/3/library/asyncio-task.html
- Python `tempfile.NamedTemporaryFile`, `os.replace`, `os.open` with `O_EXCL`:
  https://docs.python.org/3/library/tempfile.html , https://docs.python.org/3/library/os.html
- FastAPI background tasks, server workers, containers, concurrency:
  https://fastapi.tiangolo.com/tutorial/background-tasks/ ,
  https://fastapi.tiangolo.com/deployment/server-workers/ ,
  https://fastapi.tiangolo.com/deployment/docker/ , https://fastapi.tiangolo.com/async/
- Celery 5.6 (Redis transport and visibility timeout, tasks and `acks_late`, workers, soft
  shutdown [C8], revoke, periodic tasks, routing, security, testing): links [C0]-[C9] in
  `task_queue_plan.md` section 9; the Redis backend page:
  https://docs.celeryq.dev/en/stable/getting-started/backends-and-brokers/redis.html
- [UL1] Ultralytics export: `nms` "`None` exports raw one-to-many predictions for external NMS";
  "`True` embeds NMS where supported; `False` selects the NMS-free head when available"; formats
  without `batch` export batch 1; maintainer comment (2026-04-27) that `end2end` is deprecated and
  mapped to `nms`: https://docs.ultralytics.com/modes/export/ . Other Ultralytics pages: links in
  `model_family_versions_plan.md` section 10.
- NVIDIA EfficientNMS deprecation and INMSLayer; onnx-tensorrt NonMaxSuppression support: links in
  `model_family_versions_plan.md` section 10.
- Triton python backend `async_exec` (GPU-7): https://github.com/triton-inference-server/python_backend
- OpenSearch k-NN efficient filtering (OS-6), bulk API item errors (OS-2/OS-3), scroll (OS-9):
  cited per task in `codebase_audit_plan.md` section 4.
- Svelte/SvelteKit `data-sveltekit-preload-data` (FE-3) and `$derived` (FE-1):
  https://svelte.dev/docs/kit/link-options , https://svelte.dev/docs/svelte/$derived
- OpenTranscribe patterns (read only): file list in `task_queue_plan.md` section 4d and the
  Phase 2 header.

---

## Appendix A: Disposition of every review item

`R` = the review (`artifacts_local/master_plan_review.md`).

### A.1 Missing items (R 1.2)

| ID | Disposition |
|---|---|
| M-1 (#8, #346, #348, #365, #400, #401) | adopted: section 10 (every open issue), 2 "Deferred", 6.2 (#365 interplay) |
| M-2 (Appendix C/D) | adopted: Appendix B of this file, 2 "Deferred" |
| M-3 (#17/#18 leftovers) | adopted: 7.5 (5c #220 step 3, 5d #67 phase 2+), 11 |
| M-4 (per-process limits) | adopted: 3c Q-D14 row (BA-8 documented, pools in B4-4B) |
| M-5 (CC-4 interim) | adopted: OPS-1 jitter + CHANGELOG known issue (D-6); early 0.8.0 does not fix CC-4, which waits for B2 by design |
| M-6 (PR-3/4/6 as 0.7.x) | adopted: section 4 (early 0.8.0, D-7) |
| M-7 (superseded test files) | adopted: 3d |
| M-8 (Q-D cross-reference) | adopted: 3c |
| M-9 (installer/upgrade for B1) | adopted: 6.2 upgrade path, 7.1 upgrade impact and R-B1 |
| M-10 (docs per batch) | adopted: a "Docs, contracts, CHANGELOG" line per batch, plus section 16 |
| M-11 (installer YOLO11 config divergence) | adopted: 5.1 "left out" (accepted until B3) |
| M-12 (baselines only shrink) | adopted: 7 common rules |
| M-13 (image audit) | already present; kept in 12 |

### A.2 Wrong or stale references (R 2)

| ID | Disposition |
|---|---|
| W-1 head/commit count | adopted, and **not** replaced by new hard-coded values (the head moved again during this revision): section 0 "Branch facts". The model plan header line is corrected when it is committed (7, common rules) |
| W-2 CHANGELOG step done | adopted: removed; section 0 and 5.1 list the existing entries |
| W-3 lease index | adopted: 6.1 "Where" (SETTINGS role through the resources, folded and legacy, one top-level object key, mapping on both bodies, test) |
| W-4 O_EXCL "kept" | adopted: 3a says "added"; 6.3 |
| W-5 B1 done-when | adopted: "no caller outside `job_lock.py`" except the B2/B4 sites (7.1) |
| W-6 "currently shipped 8.4.197" | adopted with evidence: the rc5 images resolve 8.4.197 (read 2026-11-03); pinned in B0 so R3 tests it (D-2) |
| W-7 GPU-15 leftover | **review partly not adopted**: the leftover has a source (audit plan 4.3 GPU-15, "Left over, optional ... Not scheduled"). Kept in the backlog with the citation (7.7) |
| W-8 #447 premise | adopted: 9.2 #447 re-scope cites the export page and the `end2end` -> `nms` mapping; 1E tests go through `nms` only |
| W-9 flock count | adopted: 18 files, 51 hits (3, 3b) |

### A.3 Contradictions (R 3)

| ID | Disposition |
|---|---|
| C-1 release gap | adopted: section 4, D-7 |
| C-2 one large B1 PR | adopted: one merge commit per track, revert coupling (7.1, M21). The single PR is kept (owner rule: few PRs, one cycle) |
| C-3 CC-1 and resume/start | adopted: 5.1 "CC-1 and resume" with red tests; B0 keeps `repair_if_stale` as the only persister, starts use the derived `is_live` |
| C-4 TTL/heartbeat | adopted: thread heartbeat from B1, TTL 120 s (6.1) |
| C-5 stale reclaim race | adopted: 6.3 create-only generation takeover |
| C-6 OpenTranscribe alignment unproven | adopted as far as possible without reading that repo: cites task-plan 4d and the Phase 2 file list; B2-W1 re-reads and cites the lines (7.2); listed as unverified (15) |
| C-7 trainer change untested on GPU | adopted: the trainer change moves to B3, behind U-6.2's promote check (D-10) |
| C-8 three Ultralytics versions | adopted: pin 8.4.197 in B0 (R3 tests it), one bump in B3 (D-2, D-28) |

### A.4 Executability gaps (R 4)

| # | Disposition |
|---|---|
| 1 CHANGELOG step / duplicates | adopted (0, 5.1) |
| 2 per-task red-test names | adopted (5.1 table column) |
| 3 pyclipper pin/licence | adopted (5.1 "GPU-2 details"; the version is recorded from the merged commit) |
| 4 OCR rollback | adopted (5.1 "Rollback") |
| 5 OCR image set | adopted (5.2 step 7: COCO-Text v2 rule, fallback rule) |
| 6 heavy-run scheduling | adopted (5.2 steps 3 and 5, section 12 "Scheduling") |
| 7 CC-1 resume test | adopted (5.1) |
| 8 lease index location + test | adopted (6.1) |
| 9 kind -> scope table | adopted (6.1) |
| 10 busy window after a crash | adopted (6.1: at most 120 s, documented in CURATION.md) |
| 11 running jobs without a lease doc | adopted (6.1 "Mid-upgrade jobs", test) |
| 12 registry upgrade/fail-closed/rollback | adopted (6.2) |
| 13 registry size bound and retry cap | adopted (6.2: 10,000 classes, 5 attempts) |
| 14 trainer vs API ordering | adopted (6.2: the trainer is unchanged and reads the export snapshot) |
| 15 cachetools licence; BA-5 contracts | adopted (7.1 1D) |
| 16 `families` CI cache, budget, fixture licences | adopted (7.1 "1E tests and CI") |
| 17 who rebuilds images for the pin | adopted: the pin moves to B0, and R3 builds the images (5.3) |
| 18 live lease race budget | adopted (6.1: 10 min; B1 live 90 min) |
| 19 B2 compose/env/CLI and in-flight jobs on upgrade | adopted (7.2 "Upgrade impact") |
| 20 jobs longer than 6 h | adopted (7.2) |
| 21 B2 forward-only | adopted (7.2 "Rollback") |
| 22 B3 pin non-deterministic | adopted (D-28 freeze rule; recorded in the PR) |
| 23 B4 Redis Streams spike | adopted (7.4 4C step 0 with pass criteria, D-30 after it) |
| 24 B5 thresholds | adopted where the plan can fix them (7.5 gate column: ladder sizes, spread rule, parity numbers); the per-issue acceptance stays in #45/#40/#219/#220/#67 |
| cross-cutting: docs per batch, contracts in B1, 0.6.0 -> 0.8.0 upgrades | adopted (16, 7.1 docs line, section 4 "Direct upgrades", R-B1) |

### A.5 Churn risks (R 5)

| # | Disposition |
|---|---|
| 1 Ultralytics pin three times | adopted: B0 pin = today's resolved version, then B3 once |
| 2 B1-1E vs B3 export rewrite | adopted: D-10 minimal 1E (one argument, a head check that moves unchanged, one routing condition); the module, the trainer and the matrix go in B3 |
| 3 twelve start paths edited twice | **not adopted** (accepted cost): B1 changes the claim lines and B2 the launch lines. Doing both together would put Celery in B1 and delay the registry fix and the 0.8.0 release by the XL B2. The second edit is mechanical (`create_task` -> `enqueue`) |
| 4 heartbeat moved twice | adopted: thread heartbeat in B1, reused by B2 |
| 5 Cropwright touched three times | adopted in part: B2 and B4 wire changes ship together in 0.9.0 (one frontend release); B1's ship in 0.8.0 (M24) |
| 6 B4 baseline on a different engine | adopted: retake the 32-worker baseline on the B3 engine (7.4) |
| 7 registry vs #365 | adopted: out of scope, the interplay stated (6.2) |

### A.6 Doc suggestions (R 6)

| Suggestion | Disposition |
|---|---|
| `worker_soft_shutdown_timeout` | adopted: 7.2 W1 config line (it is already in task plan section 5 rule 4; now explicit here and asserted in `test_celery_config.py`) |
| uvicorn minimum version for the jitter flag | adopted: `uvicorn[standard]>=0.41.0` in B0 (0.41.0 release notes [U2]) |
| cooperative fencing and a generation check before the terminal write | adopted: `fence()` (6.1) |
| wall-clock dependence | adopted: noted in 6.1 rules |
| test through `nms` only (U-C1/U-6.1) | adopted: 1E tests (7.1) |

### A.7 Specific edits (R 7)

| Edit | Disposition |
|---|---|
| 1 header head/count | replaced with non-hard-coded wording (section 0) |
| 2 CHANGELOG release-prep row | done (5.1) |
| 3 Q-P1.6 row | done (3a) |
| 4 lease "Where" | done (6.1) |
| 5 heartbeat thread + TTL + fencing | done (6.1) |
| 6 done-when wording | done (7.1) |
| 7 registry upgrade | done (6.2) |
| 8 1E/D8 pin and trainer move | done (5.1, D-2, D-10) |
| 9 release strategy | done (section 4, D-7) |
| 10 B4 baseline retake | done (7.4) |
| 11 issue rows + Appendix C/D line | done (10, Appendix B) |
| 12 GPU-15 leftover | sourced, not removed (A.2 W-7) |
| 13 CC-1 row | done (5.1) |
| 14 OPS-1 uvicorn pin | done (5.1) |
| 15 YOLOv5 fixture source/licence; CI budget | done (7.1) |
| 16 B2 soft shutdown, forward-only, in-flight jobs | done (7.2) |
| 17 docs touched | done (per batch + 16) |
| 18 OCR image set | done (5.2) |

---

## Appendix B: Audit Appendix C and D carried over (stay refuted or cut; do not redo)

**Refuted (audit Appendix C):**
- OS-4: combine holdout recompute;
- GPU-11: ArcFace std;
- asyncio R9: periodic loops die;
- asyncio R10: `append_done` interleave;
- the `health.py` readiness `TimeoutError` claim;
- `get_event_loop()` in coroutines;
- asyncio R6 (part): `append_nowait` drops tasks;
- FE-4: state-writing `$effect`s;
- the FE-2 premise "no batch read-by-ids route";
- TESTS-5: `xfail_strict`;
- TESTS-8: `/detect` 503 `!= 500`;
- TESTS-9: script-style tests;
- API-3 "wire contract violated" wording;
- TESTS-7 (part): process-cache leak.

**Cut as churn (audit Appendix D), by topic:**

| Area | Cut proposals |
|---|---|
| asyncio | ruff `ASYNC` rules; a `spawn` helper + `create_task` ratchet (asyncio R5); a `gather` ratchet; the bounded `event_hub` writer |
| tests | `check_test_integrity.py`; `xfail_strict`; `register_process_cache`; the expected-skip CI gate and research deps; renaming script tests; rewriting other sleeps; replacing the 7 frontend source-text tests |
| security | URL-userinfo redaction (SEC-6 extension); the larger SEC-3..5 options (reasons private) |
| API | pure-ASGI middlewares; a pure-ASGI 500 catch-all; dropping the startup reconcile early (B2 deletes it); API-10 to zero in one pass; the capacity-cache change; a cluster-wide probe-cap lease |
| locks | `flock` in the registry and in `FileJob.write` |
| OpenSearch | `search_after` for `audit/queue`; composite aggregations; PIT for scroll; rewriting `combine/holdout.py`; OS-7 extras |
| GPU/inference | GPU-4 TTL re-validation; an EfficientNMS CI check; region unit-guess removal before a box-format parameter (GPU-15); GPU-12 standalone; GPU-6 before the decision; tenacity for GPU-8; `OPENCV_IO_MAX_IMAGE_PIXELS` as the GPU-5 fix; changing the ArcFace constant |
| frontend | refactoring the `$effect`s and an effect lint; pager windowing; moving `select()` out of `load`; a new `crops/get_batch` route |

---

## 18. Changelog of plan revisions

| Revision | Date | Changes |
|---|---|---|
| 1 | 2026-11-03 | First master plan merging the task-queue, audit and model-family plans into B0..B5 with 30 decisions |
| 2 | 2026-11-03 | Applied the fresh review in full (Appendix A). **Branch facts:** no hard-coded head or count; the B0 small fixes in progress on `b0/ocr-fixes` (merged) and `b0/data-ops-fixes`; the done CHANGELOG step removed; VERSION bumped at release prep. **Lease:** SETTINGS role through the project resources (folded configs index or legacy settings index), one top-level object key mapped on both bodies, thread heartbeat from B1, TTL 120 s, cooperative fencing, kind/scope table, crash busy window, mid-upgrade rule. **Trainer claims:** create-only generation takeover, Q-P1.6 wording "added". **B1 done-when** for `exclusive_file_lock` fixed. **Registry upgrade:** 409 `registry_not_migrated`, `openprocessor upgrade` hook, `.migrated` rename, downgrade note, installer and acceptance tests, size and retry bounds. **Release train:** early v0.8.0 after B1 (R-B1), v0.9.0 = B2-B4, v0.10.0 = B5, milestone renames. **Ultralytics:** B0 exact pin `8.4.197` (verified in the rc5 images), the trainer `nms` moved to B3, the B1-1E/B3 double rewrite removed, B3 pin freeze rule. **OPS-1:** uvicorn `>=0.41.0`. **CC-1:** red tests for the derived `interrupted` view without wedging starts or resume. **Coverage:** every open issue mapped (section 10), audit Appendix C/D carried (Appendix B), Q-D cross-reference, superseded tests, B4 retakes the 32-worker baseline on the B3 engine, the B4 Redis Streams spike with pass criteria, B5 thresholds, docs per batch. **New sections:** release strategy (4), GitHub issue actions (9), task list (11), docs per batch (16), Appendices A and B. The decision register is renumbered D-1..D-36 with "needed by" |
| 3 | 2026-11-03 | B0 small fixes merged (status note in 5.1, with SHAs); open choices from those fixes assigned to batches; T-8 test-order pollution added to B1-1D |
| 4 | 2026-11-03 | Plans committed to the public tree with neutral paths. Path legend in section 0. `/tmp` evidence references re-pointed to `artifacts_local/model_versions` or marked `<scratch repro>`. Line references pinned to `main` 5b95591f. Cleanup table updated (b7 and fix42 worktrees removed) |
