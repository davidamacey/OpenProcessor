# OpenProcessor 0.7.0 program: index (read first)

Status: plan, approved for execution step by step (owner decision 1 of 2026-11-03); B0 code is done.
Depends on: owner approval of the set; then each wave's preconditions (section 4).
Decisions applied: owner decisions 1-9 of 2026-11-03, which override the former master plan
(rev 4, in git history; section 6 records every D-n with its status).
Source IDs: every ID in `docs/design/task_queue_plan.md` (Q-), `docs/design/codebase_audit_plan.md`
(BA-, OS-, GPU-, FE-, T-, OPS-, CC-, SEC-) and `docs/design/model_family_versions_plan.md` (U-),
plus the D-, N- and M- IDs this set owns (section 11).

Verified against `fix/411-r2-round` at `af62c947` on 2026-11-03 (and the pin commit `d1c6bab0`
that landed the same day).
Consolidated 2026-11-03 (owner decision D-37 = yes, the test and GPU budget, the deps lane).

This set replaces the former master plan (rev 4, in git history; the file is deleted in the
plan-set commit, [01_b0_close_out.md](01_b0_close_out.md) B0-1). Section 12 says where each of
its sections went. [12_prior_plans_reconciliation.md](12_prior_plans_reconciliation.md) reconciles
every older plan with this set.

## 0. How to use this set

A fresh agent executes **one file** (one batch, one lane, or one procedure) plus the source-plan
task text that file cites by ID. Where a source plan and this set disagree, this set wins.

| File | What it owns |
|---|---|
| `00_index.md` (this) | mission, schedule (waves), decision register, branch/lane model, test tiers, the test and GPU budget (section 8.1), legends, section map, superseded plans |
| [01_b0_close_out.md](01_b0_close_out.md) | B0: gate cycle, push, merge |
| B1 (merged; plan files removed, the lasting facts are in the permanent docs and `CHANGELOG.md`) | B1 correctness: leases, class registry, inference and search, API/frontend/tests, Ultralytics live bugs. **Done.** |
| [03_b2_overview.md](03_b2_overview.md), [03a](03a_b2_w1_queue_infra.md), [03b](03b_b2_w2_job_families.md), [03c](03c_b2_w3_beat_reaper_workers.md) | B2 jobs on Celery, three internal stages |
| [04_b3_ultralytics_one_version.md](04_b3_ultralytics_one_version.md) | B3 one Ultralytics version, one detection contract |
| [05_b4_stateless_api.md](05_b4_stateless_api.md) | B4 stateless API |
| [06_release_acceptance.md](06_release_acceptance.md) | release prep, R-final, the tag and publish procedure |
| [07_b5_overview.md](07_b5_overview.md), [07a](07a_b5_baseline.md), [07b](07b_b5_decode_once.md), [07c](07c_b5_crops_ocr_faces.md), [07d](07d_b5_processing_plans.md) | B5 speed (inside v0.7.0, D-37) |
| [12_prior_plans_reconciliation.md](12_prior_plans_reconciliation.md) | every older plan: what is unfinished, where it lives now, what was removed and when |
| [08_environment_and_setup.md](08_environment_and_setup.md) | dev and deployment setup; config added and removed per batch |
| [09_github_actions.md](09_github_actions.md) | new issues, re-scopes, milestones, closes, dependabot and the deps lane |
| [10_standards_and_libraries.md](10_standards_and_libraries.md) | code standards, the library-per-mechanism register with sources, the hand-roll audit |
| [11_risks_cleanup_host.md](11_risks_cleanup_host.md) | risk register, cleanup, host/GPU/peer rules, the #42 side track, "do not redo" |

### 0.1 Standing procedure for a session resuming the program

1. Read this file, then the file of the current wave (section 4).
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
| **B2** jobs on Celery | v0.7.0 | XL (3 stages) | B1 merged; D-17..D-26 | Q-Phase 2-5: Redis and five core containers, every API-hosted job family (17, listed in [03](03_b2_overview.md)), beat, the periodic worker and the lease reaper, trigger workers; OPS-2; SEC-4, SEC-5 | jobs survive API recycling (CC-4); periodic side effects run once; claims hold with several workers | heavy work leaves the request processes | [03](03_b2_overview.md) |
| **B3** one Ultralytics | v0.7.0 | L + about 1 GPU day | B1 merged; D-27, D-28 (D-38) | U-Step 4-9, trainer `nms` by capability, full U-6.1 matrix in CI, GPU-14 final, GPU-18 docs, U-C11, first #368 slice | one version, one fused `[b,300,6]` contract, no monkey-patched fork, no deprecated plugin | at least 95 % of today's detector throughput, or Stage A ships alone | [04](04_b3_ultralytics_one_version.md) |
| **B4** stateless API | v0.7.0 | L | B2, B3 merged; D-29..D-33 | Q-Phase 6-7 (bulk OCC, `API_WORKERS`), Redis Streams events after a spike, #439, the last file lock deleted | same answers with 1 and 32 workers | worker count is a tuning knob, benchmarked on the B3 engine | [05](05_b4_stateless_api.md) |
| **B5** speed | v0.7.0 (D-37) | XL, 4 PRs | B4 merged; D-35 | 5a #45 baseline, #268, GPU-6 recall; 5b GPU-3b then #40/#407 (+GPU-12); 5c #219, #220 (+GPU-7, GPU-19); 5d #67 | parity gates on every change | the GPU pipeline, crops at model size, OCR/faces on resident crops, processing plans | [07](07_b5_overview.md) |
| **R-final** | v0.7.0 | M (1-2 days) | B5d merged; D-16 | release prep (VERSION and CHANGELOG), one acceptance run (old R3 + R-B1 + R-0.9), tag on the owner's word | the production-ready definition (section 5) | no regression against the retaken baseline and the B5 numbers | [06](06_release_acceptance.md) |
| side #42 | v0.8.0 | S (GPU time) | GPU 1 free | 12 GB card acceptance; never blocks the tag | | | [07](07_b5_overview.md) section 11 |

## 3. Order and dependencies

```
B0 --merge--> B1 --merge--> B2 ----------merge--> B4 --merge--> B5a -> B5b -> B5c -> B5d --> release prep --> R-final --tag v0.7.0-->
               \                         /
                +--> B3 (CPU lanes in parallel with B2; rebase on main after B2; GPU gate; merge second)
side: #42 whenever GPU 1 frees (never blocks)
```

- Order (D-13, D-37): B0, B1, B2, B3, B4, B5, then release prep and R-final last.
- B3 depends only on B1 in files, so its CPU work runs during B2 on its own integration branch;
  it merges after B2. Its GPU gate never overlaps B2's live harness.
- B4 needs the jobs moved (B2) and the final engine (B3), so it is benchmarked once.
- B5 measures only the final stack (#45 baseline before any optimization code).
- **Decided (D-37 = yes):** the owner said "not tagging until all of these plans are implemented
  and tested". B5 is inside the release, R-final runs last (after B5d), there is no separate
  v0.8.0 acceptance, and the B5 issues sit in milestone v0.7.0. Milestones: `v0.7.0` holds B0-B5
  and R-final; `v0.8.0` is platform breadth ([09_github_actions.md](09_github_actions.md)
  section 7); the old `v0.9.0` and `v0.10.0` are closed once empty, never deleted.

## 4. Execution waves (whole program)

A **wave** is the unit that one full sweep closes. Each lane is one agent in its own worktree
`<repo>-wt-<lane>` on its own branch; the coordinator merges finished lanes into the integration
branch with `--no-ff` and runs exactly one sweep per wave. **While one wave's sweep runs, the
next wave's lanes are already coding** (they branch from the frozen head of the previous
integration branch and merge into the next one without a rebase, because that head becomes an
ancestor of `main`). Branch names below are the defaults; the owning file may refine them.

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

| Wave | Batch | Lanes (branch -> brief in) | Codes in parallel / overlaps | Sweep that closes the wave | Exit criteria (in addition to PQ-1..PQ-9) |
|---|---|---|---|---|---|
| **W0** | B0 on `fix/411-r2-round` | none: the code is done; the coordinator commits the plan set (B0-1) | GH-0..GH-12 after the owner's D-36 go; W1 lanes may start coding from the frozen B0 head once the plan set is approved | B0 gate cycle: pre-commit `--all-files`, offline suite `-n 8`, contracts-check, docs check, frontend full set with e2e, changelog rules: CPU gates only, no GPU run (the OCR before/after comparison moved to GPU window 1, the B3 gate; [01](01_b0_close_out.md) B0-2, B0-3) | merge commit on `main`; `main` fast-forwarded; #437, #440, #442, #443, #445 and #366 closed; VERSION still 0.6.0; no tag |
| **W1** | B1 on `int/b1-correctness` | `b1/deps` -> [09](09_github_actions.md) section 8; `b1/ultralytics` (merged); `b1/inference` (merged); `b1/api-frontend` (merged); `b1/leases` (merged); `b1/registry` (merged). Merge order: deps, 1E, 1C, 1D, 1A, 1B (B1 (merged; see [CHANGELOG](../../../CHANGELOG.md))) | all six code in parallel; 1B rebases on 1A before its merge; during W1's sweep, B2 stage W1 (`b2/queue-infra`, spike S1 first) and the B3 CPU lanes start from the frozen `int/b1` head | B1 sweep: T1 + frontend full set + browser check of the export and diverse review pages + a short lease/registry smoke on the live harness (`-p op-b1-live`, cap 30 min; the full live run is the single B1+B2 run at the end of B2) + the `families` CI job; spawn races run 10x locally | file-lock ratchet lists only the B2/B4 sites and `job_lock.py`; no `class_registry.json` read outside export snapshots and the trainer; `registry_not_migrated` tested; lease p50/p99 acquire latency recorded; N2-N7, N9, N10, #447 closed on merge. PQ-7: no GPU gate in B1 by design; the lease round-trip cost is the recorded number |
| **W2** | B2 on `int/b2-celery` | stage B2-W1: `b2/queue-infra` -> [03a](03a_b2_w1_queue_infra.md); stage B2-W2: `b2/families-a`, `b2/families-b`, `b2/families-c`, then `b2/frontend-queued` -> [03b](03b_b2_w2_job_families.md); stage B2-W3: `b2/beat-reaper`, `b2/trigger-workers`, `b2/hardening` -> [03c](03c_b2_w3_beat_reaper_workers.md) | stages run in order; the coordinator merges each stage and runs targeted tests and static checks on it (not a full sweep); B3 lanes code throughout W2 on `int/b3-ultralytics`; 4C's spike plan and 4D's timing script may be drafted | B2 sweep after stage W3: T1 + frontend full set + the one live run covering B1 and B2 together (`-p op-b2-live`: B1 lease/registry checks plus race, kill mid-job, cancel, API recycle, arbiter, two auto-label workers; about 3 h) + `acceptance_run.sh --only` the job phases + image audit and Trivy for changed images | no job family runs in an API worker; lifespan creates per-process resources only; file-lock ratchet lists only the event-hub site and `job_lock.py`; OPS-2 measured first; Redis licence row; N11, N12 closed. PQ-7: request latency of `/detect` and `/ingest` under a running job recorded against the B1 numbers |
| **W3** | B3 on `int/b3-ultralytics` | `b3/stagea` first, then `b3/families`, `b3/frame`, `b3/export`, `b3/decoders`, `b3/docs` as [04](04_b3_ultralytics_one_version.md) orders them | codes during all of W2 (CPU only); rebases on `main` after B2 merges; during W3's sweep, B4's 4C spike (on the merged B2 stack) and 4D's measurement start | B3 sweep: T1 + U-6.1 matrix at the pin X + **GPU window 1**: the U-6.2 gate once for Stage A and Stage B plus the B0 OCR before/after comparison (`-p op-b3-gpu`, GPU 0, one GPU day; never overlapping a B2 live run) + `pip check`, image audit, Trivy | one Ultralytics version in every image and requirements file; pin test green in CI and in each image; N8 closed. PQ-7: |delta mAP50-95| <= 0.3, >= 99 % detection agreement, throughput >= 95 % at batch 1/8/32/64 with p99 within +10 %, embedding cosine >= 0.999, export image about -0.9 GB, all recorded |
| **W4** | B4 on `int/b4-stateless` | `b4/bulk-occ` (4A), `b4/worker-knob` (4B), `b4/events` (4C, spike first, D-30 after it), `b4/project-create` (4D) -> [05](05_b4_stateless_api.md) | 4C spike and 4D timing during W3's sweep; release-prep drafting (release notes) may start during W4's sweep | B4 sweep: T1 + frontend full set + the third and last live run (1-vs-32 worker diff) (no GPU in B4: the installer default `API_WORKERS` is provisionally `min(nproc, 32)`; the T5 32-worker ingest and search benchmark on the B3 engine runs in GPU window 2, the B5 window, per 05 step 4B-5, and 5a retakes the same baseline) | `rg -n "fcntl" src` empty; file-lock ratchet baseline zero; the 1-vs-32 diff empty; no lifespan side effects; N1, #439 closed. PQ-7: the `API_WORKERS` default is confirmed or corrected by the GPU window 2 benchmark before R-final |
| **W5a** | B5 baseline ([07a](07a_b5_baseline.md)) on `int/b5a-baseline` | per 07a | no optimization code starts before the 5a reference numbers exist; the B5b/5c/5d lanes may code while earlier sub-batches are measured | 5a sweep: T1 + **GPU window 2** opens: the #45 baseline once (ladders, #268 sweeps, GPU-6 recall) | numbers recorded with host load; no default changes unless the gain beats 2x the run-to-run spread and parity passes; D-35 decided (N13) |
| **W5b** | B5b decode once ([07b](07b_b5_decode_once.md)) | per 07b (GPU-3b first) | 5c lanes code during W5b's sweep | 5b sweep: T1 + the #40/#407 measurement and parity (boxes within 1 px at the original resolution, embedding cosine >= 0.999), in sequence inside GPU window 2 | #40/#407 acceptance met; numbers in `docs/PERFORMANCE.md` |
| **W5c** | B5c crops, OCR, faces ([07c](07c_b5_crops_ocr_faces.md)) | per 07c | 5d lanes code during W5c's sweep | 5c sweep: T1 + P-vlm, P-sam, P-region, P-ocr gates, GPU window 2 | #219/#220 acceptance met |
| **W5d** | B5d processing plans ([07d](07d_b5_processing_plans.md)) | per 07d | release-prep drafting may start | 5d sweep: T1 + #67 acceptance with per-stage "not run" tests, GPU window 2 closes | #67 acceptance met |
| **W6** | release prep + R-final ([06](06_release_acceptance.md)) | `chore/release-0.7.0-prep` (RP-1..RP-9: VERSION 0.6.0 -> 0.7.0 and CHANGELOG `[Unreleased]` -> `[0.7.0]` first); `fix/rfinal-findings` only if a row fails | nothing else merges to `main` | R-final itself (every RF row), **GPU window 3** | every RF row PASS; the production-ready definition (section 5) ticked; report to the owner; tag v0.7.0 only on the owner's word (D-16); #411 closed after the tag |
| side | #42 ([07](07_b5_overview.md) section 11) | `perf/42-small-gpu-acceptance` | any time GPU 1 is free; never blocks | its own run on GPU 1 | the #42 pass list; results in `docs/PERFORMANCE.md` and #42 |

### 4.1 Task list (carried from section 11 of the former master plan)

| Task | What | Wave | Status |
|---|---|---|---|
| #28 / B0-a..c | GPU-1, GPU-3a+13, GPU-2; OS-2, OS-1, OPS-1; CC-1, BA-7 | W0 | merged into `fix/411-r2-round` |
| B0-d | uvicorn `>=0.41.0` floor and test | W0 | done (`6a53673c`) |
| B0-e | `ultralytics==8.4.197` exact pin and test | W0 | done (`d1c6bab0`) |
| B0-f | release prep | W6 | moved (owner decision 3) |
| B0-g | OCR before/after comparison | W3 (GPU window 1) | moved out of B0 (budget, section 8.1) |
| B0-i | docs fixes `ad553ecd` (stale "patch breaks on 8.4" comments), `0cc2b3ac` (docs checker skips route/env checks in `docs/design/`) | W0 | done |
| #28 rest | plan commit (with removal wave R-1), CPU gate cycle, push, PR, CI, merge, fast-forward | W0 | to do |
| #25, #22 | R3 live acceptance with the segmenter tier | W6 | moved into R-final (D-3, D-4) |
| B0-h | tag v0.7.0 and the GitHub release | W6 | after R-final, owner's word |
| #27 | #42 small-GPU acceptance on GPU 1 | side | waiting for GPU 1 |
| GH-1 (#38) | GitHub actions | before W1 | to do, on the owner's go (D-36) |
| #31, #34 | audit plan, Ultralytics plan | - | done (committed, D-14) |
| B1-0 | commit the source plans | - | done (D-14); the header pointers of the three source plans are repointed to this set in the plan-set commit |
| B1-1A..1E, deps | B1 lanes | W1 | to do |
| R-B1 | 0.8.0 acceptance subset | - | dropped (owner decision 3); its checks are in R-final section 4.3 |
| B2-W1..W3 | B2 stages | W2 | to do |
| B3 | B3 | W3 | to do |
| B4-4A..4D, R-0.9 | B4; R-0.9 merged into R-final | W4, W6 | to do |
| #17/#18 leftovers | #220 step 3 research (B5c), #67 phase 2+ (B5d) | W5c, W5d | to do |
| B5a..d | B5 sub-batches | W5a..W5d | to do |
| #40 (ledger) | leftover cleanup | per [11](11_risks_cleanup_host.md) section 5 | owner yes per row |

## 5. Definition of production-ready for v0.7.0

R-final verifies every line; the RF row ids are in [06_release_acceptance.md](06_release_acceptance.md)
section 4.

| # | Production-ready means | Verified by |
|---|---|---|
| PR-01 | Cross-process state uses idempotency, OpenSearch OCC, leases or Celery, never a file or in-process lock: `rg -n "fcntl" src` is empty and the file-lock ratchet is at zero | release-prep precondition; B4 exit |
| PR-02 | A singleton job starts once under concurrent requests; a dead holder blocks a restart for at most 120 s | RF-42 |
| PR-03 | Concurrent class writes never lose a class or duplicate an id | RF-43 |
| PR-04 | Jobs run in workers: they survive an API restart, a killed worker's resumable job is re-enqueued, periodic tasks run once | RF-52, RF-53, RF-55 |
| PR-05 | The API gives the same answers with 1 and 32 workers | the B4 sweep record |
| PR-06 | Class identity is the name across import, combine, export, train, promote and the secondary detector | B1 tests; RF-40, RF-43 |
| PR-07 | Fail closed: OpenSearch down -> 503 on starts and `review_dismiss`; unmigrated registry -> 409; a raw-head engine -> refused; Redis down -> degraded health and 503 starts | RF-36, RF-41, RF-57, RF-61 |
| PR-08 | One exact Ultralytics version in every image; one fused detection contract; the B3 parity and throughput gate passed | RF-20, RF-57, the B3 sweep record |
| PR-09 | Fresh install, re-run, repair, `--skip-models`, engine adoption, uninstall work | RF-01, RF-02, RF-12, RF-23, RF-24, RF-25 |
| PR-10 | The supported upgrade, 0.6.0 -> 0.7.0, keeps every class id, adds the queue and the new services, rebuilds the default engine once; rollback to 0.6.0 works with the documented downgrade step | RF-40, RF-41, RF-56, RF-58, RF-59, RF-62, RF-63 |
| PR-11 | Security: SEC-1..SEC-6 fixed (neutral wording); the broker has no published port; Trivy fixable CRITICAL = 0; every image passes the image-vs-git audit; no secret in any log or report | RF-08, RF-09, RF-10, RF-34, RF-50 |
| PR-12 | Quality: CI green; ratchet baselines smaller than or equal to their `5b95591f` values; no dead code or shims; contracts in sync; docs check clean | PQ-1..PQ-5 at every wave; the release commit's full gate |
| PR-13 | Speed: no regression against the 32-worker baseline retaken on the B3 engine at the shipped `API_WORKERS` default; numbers in `docs/PERFORMANCE.md` | the GPU window 2 record (5a baseline) |
| PR-14 | UI: every changed page checked at 1440 and 430 px with the screenshots viewed; the frontend live suite passes, region tests included | RF-04, RF-07, RF-33, RF-45, RF-54, RF-60 |
| PR-15 | Docs and release: INSTALLATION (upgrade, downgrade), ARCHITECTURE (leases, registry, services, events, model families), CURATION (`job_busy` window, `registry_not_migrated`), SECURITY (broker) current; CHANGELOG consolidated with the breaking changes first; release notes; `/health` reports 0.7.0 | RP-1..RP-6, RF-26 |

## 6. Decision register

Statuses: **approved** (owner decision of 2026-11-03), **done**, **open** (with the plan's
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
| D-8 | J-LEASE | approved | OpenSearch lease docs in the SETTINGS role, thread heartbeat 15 s, TTL 120 s, cooperative fencing | B1 | B1 1A (merged; see [ARCHITECTURE](../../../ARCHITECTURE.md)) |
| D-9 | J-REG and its upgrade path | approved | registry doc with OCC (5 attempts, 10,000 classes), 409 `registry_not_migrated`, `upgrade` hook, `.migrated` rename, downgrade note. Only 0.6.0 -> 0.7.0 is tested (no 0.7.0 install exists) | B1 | B1 1B (merged; see [ARCHITECTURE](../../../ARCHITECTURE.md)) |
| D-10 | B1-1E scope | approved | minimal in B1; trainer `nms`, export module and full matrix in B3 | B1 | B1 1E (merged; see [CHANGELOG](../../../CHANGELOG.md)) |
| D-11 | U-J3 YOLOv5 scope | approved, option b | bring-your-own original-YOLOv5 ONNX for probe and secondary; YOLOv5u for serving | B1 | B1 1E (merged; see [CHANGELOG](../../../CHANGELOG.md)) |
| D-12 | U-J4 `nms` by capability | approved | `nms=False` with `one2one_cv2`, else `nms=True`, class-agnostic | B1 (export), B3 (trainer) | B1 1E (merged; see [CHANGELOG](../../../CHANGELOG.md)), [04](04_b3_ultralytics_one_version.md) |
| D-13 | Order (Q-J8) | approved; amended by D-37 | B0, B1, B2, B3, B4, B5, then R-final last | - | section 3 |
| D-14 | Commit the plans | done | committed on `fix/411-r2-round` | - | |
| D-15 | Trainer claim (Q-P1.6) | approved | create-only generation files; API `training/jobs.py` `O_EXCL` create | B1 | B1 1A (merged; see [ARCHITECTURE](../../../ARCHITECTURE.md)) |
| D-16 | Tag | replaced | one tag decision: v0.7.0 after R-final PASS (R-final runs after B5), on the owner's word (absorbs the old D-34) | R-final | [06](06_release_acceptance.md) section 7 |
| D-17 | Q-J1 Celery 5.6 | open per decision 6 | the owner already chose Celery (task plan J1); confirm the 5.6 line at B2 kick-off | B2 | [03](03_b2_overview.md) |
| D-18 | Q-J2 broker | open | Redis 8 (AGPLv3 option); Valkey only if a redistributor objects | B2 | [03a](03a_b2_w1_queue_infra.md) |
| D-19 | Q-J3 default install | open | five core containers: `redis`, `jobs-worker`, `open-vocab-worker`, a `periodic-worker` for the `periodic` queue (so the reaper and the dispatch tick never wait behind long jobs), `jobs-beat`; Flower under monitoring | B2 | [03a](03a_b2_w1_queue_infra.md) |
| D-20 | Q-J4 delivery | open | `acks_late` + `reject_on_worker_lost` + lease + attempt cap 3 + soft shutdown 30 s | B2 | [03a](03a_b2_w1_queue_infra.md) |
| D-20a | Time limits for long job kinds (new) | open | default 3 h / 3 h 15 m; 24 h / 24 h 15 m for the long kinds named in 03 | B2 | [03](03_b2_overview.md) |
| D-20b | `REMAP_SIGTERM=SIGQUIT` on worker services (new) | open | yes, so `docker stop` gets the soft shutdown; `stop_grace_period` above the soft-shutdown timeout | B2 | [03](03_b2_overview.md) |
| D-21 | Q-J5 job-state store | open | status/progress/cancel stay in `FileJob` files; exclusivity is the lease | B2 | [03](03_b2_overview.md) |
| D-22 | Q-J6 scope | open | all 17 API-hosted job families (the dispatcher source poll and the cache prune included), the periodic loops, auto-label, assist loop; keep detection/VLM workers, trainer, export worker; evaluator per D-31 | B2 | [03](03_b2_overview.md) |
| D-22a | Directory ingest is not project-bound (new) | open | a plain task with a deployment-level job record, no lease | B2 | [03](03_b2_overview.md) |
| D-22b | Promote `wait=true` (new) | open | delete it (only tests use it); no shim | B2 | [03b](03b_b2_w2_job_families.md) |
| D-23 | Q-J9 `jobs` concurrency | open | 4, tunable | B2 | [03a](03a_b2_w1_queue_infra.md) |
| D-24 | Q-J10 Redis durability | open | AOF everysec, no backup | B2 | [03a](03a_b2_w1_queue_infra.md) |
| D-25 | Q-J11 auto-resume | open | reprocess, dataset import, bulk ingest, combine, assist loop | B2 | [03c](03c_b2_w3_beat_reaper_workers.md) |
| D-26 | Q-J12 recycling | open | keep `--limit-max-requests` with jitter (landed in B0) | B2 | [03](03_b2_overview.md) |
| D-27 | U-J1 staging + U-J5..J8 | open | one version; U-6.2 once for Stage A and B; J5 keep `conf=0.25`; J6 keep the Triton name; J7 export batch = `max_batch_size`; J8 trainer on the same release | B3 | [04](04_b3_ultralytics_one_version.md) |
| D-28 | B3 pin choice rule | open | newest 8.4.x passing U-6.1 at a freeze date set at B3 kick-off (at least 3 days old) | B3 | [04](04_b3_ultralytics_one_version.md) |
| D-29 | Q-J7 request fan-out | open | bulk OCC; the ingest data plane a documented `TaskGroup` exemption | B4 | [05](05_b4_stateless_api.md) |
| D-29a | VLM batch routes (`label_batch`, `verify_region_batch`) (new) | open | 202 + a Celery job; the bounded `TaskGroup` runs in the worker | B4 | [05](05_b4_stateless_api.md) |
| D-30 | Q-J13 event transport | open | Redis Streams, decided after the 4C spike passes | B4 | [05](05_b4_stateless_api.md) |
| D-31 | Q-J6 evaluator | open | `-Q gpu_eval -c 1` in B4 | B4 | [05](05_b4_stateless_api.md) |
| D-32 | #439 approach | open | measure first; then a `project_create` job (202) | B4 | [05](05_b4_stateless_api.md) |
| D-32a | Project status during creation (new) | open | reuse the existing `building` status (combine targets use it), not a new `creating` | B4 | [05](05_b4_stateless_api.md) |
| D-33 | Q-J14 API workers | open | `API_WORKERS` (`WEB_CONCURRENCY`), default `min(nproc, 32)`, confirmed by the benchmark | B4 | [05](05_b4_stateless_api.md) |
| D-34 | Tag v0.9.0 | dropped | merged into D-16 | - | |
| D-35 | GPU-6 `/detect` upscaling | open | keep `scaleup=False` until 5a measures recall | B5a | [07a](07a_b5_baseline.md) |
| D-36 | GitHub actions | approved | executed only on the owner's go after this set is approved | before W1 | [09](09_github_actions.md) |
| D-37 | B5 before the first tag? (new) | approved: YES | owner, 2026-11-03: "not tagging until all of these plans are implemented and tested". v0.7.0 = B0..B5 + R-final; R-final runs last; milestones re-cut (section 3, [09](09_github_actions.md) section 7) | - | section 3 |
| D-38 | Dependabot #432 handling: Triton 26.09 and a newer Python base (new; the same question as Q-4 and ENV-10 in [08](08_environment_and_setup.md)) | open | close #432 and #422; do the move only inside B3 as optional task B3-A5, with the TensorRT pins and the `Dockerfile.triton` assertion moved together, behind the U-6.2 gate, if B3's GPU window allows; otherwise after the tag | B3 kick-off | [04](04_b3_ultralytics_one_version.md) B3-A5, [09](09_github_actions.md) section 8 |
| OQ-B5-1..3 | B5 open questions (public replacement for private acceptance bars, the 1M run, GPU 2 for two-card rows) | open | see the owning file | before B5a | [07](07_b5_overview.md) section 9 |
| OC-1 | Shared OCR detector-size helper for `/ocr` and the cascade (open choice from the B0 fixes) | open | it changes `/ocr` behaviour; decide in B1-1C | B1 | B1 1C (merged; see [CHANGELOG](../../../CHANGELOG.md)) |
| OC-2 | pyclipper in the hardened Triton scan image (`docker/hardened/triton`) | open | decide with B2's image work (stage W3, hardening lane) | B2 | [03c](03c_b2_w3_beat_reaper_workers.md) |
| OC-3 | `autolabel/job.get_state` still repairs through `_fail_unclaimed` | decided: leave | B2 stage W3 (trigger workers) deletes it | B2 | [03c](03c_b2_w3_beat_reaper_workers.md) |

## 7. Branch, lane and wave model

- **B0** merges `fix/411-r2-round` to `main` first, with a merge commit.
- **Each batch** gets `int/bN-<name>` (`int/b1-correctness`, `int/b2-celery`,
  `int/b3-ultralytics`, `int/b4-stateless`, `int/b5a-baseline` and so on) cut from `main` after
  the previous batch merged. A lane may branch from the previous integration branch's frozen
  head while that batch's PR is open; B3 is cut from `main` after B1 and rebased after B2.
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
| T1 pre-commit `--all-files`, offline suite (`-n 8`, about 25-33 min), contracts-check, docs check, changelog rules; frontend full set when `frontend/` changed | once | once | once (after stage W3) | once | once | release-prep gate; the release commit's full gate | once |
| T2 live harness (`docker/test/compose.yml`, own `-p`) | - | lease/registry smoke, cap 30 min | the single B1+B2 run (about 3 h) | - | 1-vs-32 diff | RF-03 | - |
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
| GPU windows | exactly three: (1) the B3 gate (U-6.2 for Stage A and B) plus the B0 OCR before/after comparison; (2) the B5 window: the #45 baseline once, then each sub-batch's measurement in sequence (5b, 5c, 5d); (3) R-final. Each window is announced to the peer sessions and run on the agreed GPU ([11](11_risks_cleanup_host.md) section 1) |

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
| B2 | services, queues | job states, `queued` | - | Redis, new services | `queued`, directory-ingest 202 | - | `CLAUDE.md`, `SECURITY.md`, docs-site, ATTRIBUTION (Celery, Redis) | root (breaking), frontend |
| B3 | supported model families | - | - | engine rebuild | export formats | U-6.2 results | README, MIGRATION_TRITON_26, image_size_audit, export/README, docs-site | root (breaking) |
| B4 | events, workers | - | - | `API_WORKERS` | `conflicts`, `building` on create, VLM batch 202 | retaken baseline, knob | engineering_standards (J7 exemption) | root, frontend |
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
- Research and platform breadth are out of this program: #8, #346, #348 (parked); #365, #400,
  #401; #347, #398, #406, #416, #366 (done), #367, #370, #84, #402, #435, #436, #231, #403
  ([09_github_actions.md](09_github_actions.md) section 7 gives each one's milestone and reason).
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
| `W-n` | this file, section 4 | program waves (B2's internal stages are written `B2-W1..W3`) |
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
| 3a Replacement for the Celery plan's Phase 1 | B1 1A (merged; see [ARCHITECTURE](../../../ARCHITECTURE.md)) (B0/B1 rows), [03a](03a_b2_w1_queue_infra.md) (Celery exclusive starts row) |
| 3b Every `flock` site (19 files, 55 hits) | B1 1A (merged; see [ARCHITECTURE](../../../ARCHITECTURE.md)) (B1 rows, the ratchet), [03c](03c_b2_w3_beat_reaper_workers.md) (dispatcher, sweeper, cache prune), [05](05_b4_stateless_api.md) (event hub, `job_lock.py`) |
| 3c Q-D1..D15 cross-reference | B1 (merged; see [CHANGELOG](../../../CHANGELOG.md)) (B1 rows) and [03](03_b2_overview.md) (B2 rows); D1, D10, D11, D13 are B0 fixes recorded in [01](01_b0_close_out.md) |
| 3d Superseded source-plan tests | B0 rows: [01](01_b0_close_out.md) section 3 and Corrections; B1 rows: B1 1A (merged; see [ARCHITECTURE](../../../ARCHITECTURE.md)), B1 1B (merged; see [ARCHITECTURE](../../../ARCHITECTURE.md)) |
| 4 Release strategy | replaced by owner decisions 2-3 and D-37; this file section 6 (D-7, D-16, D-34, D-37) and [06](06_release_acceptance.md) |
| 5 B0 (5.1 scope, 5.2 gate cycle) | [01](01_b0_close_out.md) |
| 5.3 R3 live acceptance | [06](06_release_acceptance.md) sections 4.1, 4.2, 5 |
| 5.4 Tag decision point | [06](06_release_acceptance.md) section 7 |
| 6.1 J-LEASE, 6.3 trainer claims | B1 1A (merged; see [ARCHITECTURE](../../../ARCHITECTURE.md)) |
| 6.2 J-REG | B1 1B (merged; see [ARCHITECTURE](../../../ARCHITECTURE.md)) |
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
| M-1 (#8, #346, #348, #365, #400, #401) | every open issue mapped: [09](09_github_actions.md) section 7; #365 interplay in B1 1B (merged; see [ARCHITECTURE](../../../ARCHITECTURE.md)) |
| M-2 (audit Appendix C/D) | [11](11_risks_cleanup_host.md) section 4 |
| M-3 (#17/#18 leftovers) | [07c](07c_b5_crops_ocr_faces.md) (#220 step 3), [07d](07d_b5_processing_plans.md) (#67 phase 2+) |
| M-4 (per-process limits) | BA-8 documented in B1 1D (merged; see [CHANGELOG](../../../CHANGELOG.md)); pools sized per `API_WORKERS` in [05](05_b4_stateless_api.md) |
| M-5 (CC-4 interim) | the jitter shipped in B0; the known-issue line is dropped (D-6); CC-4 is fixed in B2 |
| M-6 (PR-3/4/6 as 0.7.x) | replaced by owner decision 2 (one release) |
| M-7, M-8 (superseded tests, Q-D cross-reference) | [01](01_b0_close_out.md), B1 1A (merged; see [ARCHITECTURE](../../../ARCHITECTURE.md)), B1 1B (merged; see [ARCHITECTURE](../../../ARCHITECTURE.md)), B1 (merged; see [CHANGELOG](../../../CHANGELOG.md)), [03](03_b2_overview.md) |
| M-9 (installer/upgrade for B1) | B1 1B (merged; see [ARCHITECTURE](../../../ARCHITECTURE.md)); live in [06](06_release_acceptance.md) RF-40..RF-41 |
| M-10 (docs per batch) | this file section 9 and each batch file |
| M-11 (installer YOLO11 config divergence, U-C11) | accepted until B3, which deletes the heredoc: [04](04_b3_ultralytics_one_version.md) |
| M-12, M-13 (baselines only shrink; image audit) | PQ-2, PQ-8 (section 4) |
| W-1 (head/commit counts) | no hard-coded SHAs (section 7) |
| W-2 (CHANGELOG step done) | [01](01_b0_close_out.md) section 3 |
| W-3, W-4, W-5, W-9 (lease index, O_EXCL "added", B1 done-when, flock count) | B1 1A (merged; see [ARCHITECTURE](../../../ARCHITECTURE.md)), B1 (merged; see [CHANGELOG](../../../CHANGELOG.md)) |
| W-6 (8.4.197 shipped) | done: the pin commit `d1c6bab0` |
| W-7 (GPU-15 leftover) | review partly not adopted: kept in the backlog with its source (section 10) |
| W-8 (#447 premise) | [09](09_github_actions.md) C-447; B1 1E (merged; see [CHANGELOG](../../../CHANGELOG.md)) |
| C-1 (release gap) | replaced by owner decision 2 |
| C-2 (one large B1 PR) | one merge commit per lane, revert coupling (section 7, M21); the single PR is kept |
| C-3 (CC-1 and resume) | [01](01_b0_close_out.md) Corrections; B1 1A (merged; see [ARCHITECTURE](../../../ARCHITECTURE.md)) |
| C-4, C-5 (TTL and heartbeat; stale reclaim race) | B1 1A (merged; see [ARCHITECTURE](../../../ARCHITECTURE.md)) |
| C-6 (OpenTranscribe alignment) | [03a](03a_b2_w1_queue_infra.md) re-reads it; listed unverified in [11](11_risks_cleanup_host.md) 3.1 |
| C-7, C-8 (trainer untested on GPU; three versions) | trainer moves in B3 behind U-6.2 ([04](04_b3_ultralytics_one_version.md)); the pin is done |
| A.4 executability gaps 1-7 (CHANGELOG, red-test names, pyclipper pin, OCR rollback, OCR image set, heavy-run scheduling, CC-1 resume test) | [01](01_b0_close_out.md); [11](11_risks_cleanup_host.md) section 1 |
| A.4 gaps 8-11, 18 (lease location, kind/scope, busy window, no-doc jobs, live race budget) | B1 1A (merged; see [ARCHITECTURE](../../../ARCHITECTURE.md)) |
| A.4 gaps 12-14 (registry upgrade, bounds, trainer ordering) | B1 1B (merged; see [ARCHITECTURE](../../../ARCHITECTURE.md)) |
| A.4 gaps 15, 16 (cachetools licence, BA-5 contracts; `families` CI) | B1 1D (merged; see [CHANGELOG](../../../CHANGELOG.md)), B1 1E (merged; see [CHANGELOG](../../../CHANGELOG.md)) |
| A.4 gap 17 (who rebuilds images for the pin) | resolved: the pin is done; R-final builds the images |
| A.4 gaps 19-21 (B2 compose/env/upgrade, jobs over 6 h, forward-only) | [03](03_b2_overview.md) |
| A.4 gaps 22-24 (B3 pin rule, B4 spike, B5 thresholds) | [04](04_b3_ultralytics_one_version.md), [05](05_b4_stateless_api.md), [07](07_b5_overview.md) |
| A.4 cross-cutting (docs per batch, contracts, upgrades) | section 9; [06](06_release_acceptance.md) (0.6.0 -> 0.7.0) |
| A.5 churn 1, 2, 4, 6, 7 | pin once then B3 ([04](04_b3_ultralytics_one_version.md)); minimal 1E (B1 1E (merged; see [CHANGELOG](../../../CHANGELOG.md))); one heartbeat class (B1 1A (merged; see [ARCHITECTURE](../../../ARCHITECTURE.md)), [03a](03a_b2_w1_queue_infra.md)); baseline retaken on the B3 engine ([05](05_b4_stateless_api.md)); #365 out of scope (B1 1B (merged; see [ARCHITECTURE](../../../ARCHITECTURE.md))) |
| A.5 churn 3 | not adopted, accepted cost (section 7) |
| A.5 churn 5 (Cropwright three times) | moot: everything ships in one release (M24) |
| A.6 doc suggestions (soft shutdown, uvicorn floor, fencing, wall clock, `nms`-only tests) | [03a](03a_b2_w1_queue_infra.md); done (`6a53673c`); B1 1A (merged; see [ARCHITECTURE](../../../ARCHITECTURE.md)); B1 1E (merged; see [CHANGELOG](../../../CHANGELOG.md)) |
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

## 16. Corrections found while splitting

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
