# GitHub actions: issues, re-scopes, milestones, closes and dependabot

Status: plan, not started. Nothing in this file has been executed.
Depends on: the owner's approval of this plan set, then the owner's go for D-36 (owner decision 4).
Decisions applied: owner decisions 2, 3, 5 and 9 of 2026-11-03 and D-37 = yes (one release
v0.7.0 = B0-B5 + R-final, tagged only when every plan is implemented and tested; v0.8.0 =
platform breadth; v0.9.0 and v0.10.0 closed once empty; every new issue on v0.7.0; parked
unmilestoned research stays parked; one batched deps step that never floats the exact
Ultralytics pin); D-36. The register is in [00a_decision_register.md](00a_decision_register.md) section 6.
Source IDs: the former master plan (rev 4, in git history) sections 9 and 10 (N1-N16,
re-scopes, milestones, close table, issue map); audit IDs per issue; ledger section 7 (private).

What to keep after merge: the milestone descriptions and issue bodies are the permanent record
(they carry what is open); the issue map (section 7) is not copied anywhere. The plan files go
as described in [00a_decision_register.md](00a_decision_register.md) "Plan lifecycle".

Verified against `fix/411-r2-round` at `af62c947` on 2026-11-03; GitHub state read the same day
with `gh api repos/davidamacey/OpenProcessor/milestones?state=all`, `gh issue list --state open
--limit 100 --json number,title,milestone,labels` and `gh pr list --state open`.
Consolidated 2026-11-03.

## 1. Rules for whoever executes this

- Execute only after the owner's go. Run the steps in section 3 in order; each is safe to repeat.
- Security issues stay neutral: no mechanism, no path, no exploit wording. Details live in the
  maintainers' private notes. Read every SEC-related body and comment once more before posting.
- No private names, machine paths or other private repositories in any title, body or comment.
- Fewer issues is better: bugs fixed inside a batch are referenced in the PR, not filed.
- Prefer edit, move and close over delete. Nothing in this plan deletes a milestone or an issue.
- Record what was created (N-id to issue number) in `artifacts_local/gh/created.tsv`, then add the
  numbers to this file in the next docs commit (a "Created as" column in section 4).

## 2. State on 2026-11-03

| Milestone (number) | State | Open | Closed | Description today |
|---|---|---|---|---|
| v0.7.0 (6) | open | 1 (#411) | 35 | a stale performance ordering ("Suggested order: 1 #337 + #268 + #259 ...") |
| v0.8.0 (7) | open | 0 | 19 (15 issues, 4 PRs, all closed 2026-10-31 to 2026-11-02) | stale: "Graph-based global cluster refinement ..." |
| v0.9.0 (8) | open | 10 (the perf issues) | 2 (#429, #414, closed 2026-11-02) | stale: "Long-term: spatial memory and photogrammetry ..." |
| v0.10.0 (9) | open | 12 (platform breadth) | 1 (#399, closed 2026-10-31) | "Platform breadth: ..." |

Every closed item on v0.8.0, v0.9.0 and v0.10.0 was closed after the v0.6.0 release
(2026-10-16), so it ships in 0.7.0. Moving those items to v0.7.0 makes the milestones tell the
truth; it is why the plan moves rather than renames.

Open issues: 37 (section 7 maps every one). Open PRs: five dependabot PRs, #422, #423, #432,
#433, #434 (section 8). Labels: there is no `build` label (the old N8 row used it); this file
uses existing labels only.

## 3. Steps, in order

Variables used below: `R=repos/davidamacey/OpenProcessor`. Re-read the milestone numbers first;
the commands assume v0.7.0=6, v0.8.0=7, v0.9.0=8, v0.10.0=9. Order matters: v0.7.0 first
(GH-1, GH-3); v0.8.0 is repurposed only after GH-1 has emptied it (GH-4).

**GH-0 Snapshot.**
```bash
mkdir -p artifacts_local/gh
gh api "$R/milestones?state=all&per_page=100" > artifacts_local/gh/milestones_before.json
gh issue list --state all --limit 500 --json number,title,state,milestone,labels > artifacts_local/gh/issues_before.json
gh api "$R/milestones?state=all&per_page=100" --jq '.[] | [.number,.title,.state,.open_issues,.closed_issues] | @tsv'
```
Stop if the numbers or counts differ from section 2; re-plan the moves from the new state.

**GH-1 Move the closed items of v0.8.0, v0.9.0 and v0.10.0 to v0.7.0.** The issues API edits
issues and PRs alike.
```bash
for m in 7 8 9; do gh api "$R/issues?milestone=$m&state=closed&per_page=100" --jq '.[].number'; done | sort -n > artifacts_local/gh/closed_to_move.txt
# expected: 41 101 351 353 354 379 384 385 399 404 405 410 412 414 415 417 419 426 427 428 429 430
while read -r n; do gh api -X PATCH "$R/issues/$n" -F milestone=6 --jq '[.number,.milestone.title]|@tsv'; done < artifacts_local/gh/closed_to_move.txt
```

**GH-2 Edit the v0.7.0 description.**
```bash
gh api -X PATCH "$R/milestones/6" -f description='One release for the 0.7.0 program, tagged only when every plan is implemented and tested: B0 close-out (R2 round fixes, OCR geometry, orientation and unclip, holdout freeze fails closed, create-only config writes, atomic state writes, recycle jitter, exact Ultralytics pin), B1 correctness (job leases, class registry with OCC and an upgrade migration, audit fixes, Ultralytics live bugs), B2 jobs on Celery workers, B3 one Ultralytics version and one detection contract, B4 stateless API, B5 speed (#45 baseline, decode once, crops at model size, OCR and faces on resident crops, processing plans), then one release acceptance (R-final). Tagged on the owner word. Plan: docs/design/program/00a_decision_register.md'
```

**GH-3 Move the B5 perf issues from v0.9.0 to v0.7.0** (D-37: B5 is inside the release; #42 is
a non-blocking side track inside it, 07 section 11).
```bash
for n in 40 42 45 67 219 220 231 268 403 407; do gh issue edit "$n" --milestone v0.7.0; done
```

**GH-4 Repurpose v0.8.0 for platform breadth and move those issues in.** v0.8.0 is empty after
GH-1 (its closed items moved to v0.7.0); check before renaming nothing else.
```bash
gh api "$R/milestones/7" --jq '[.open_issues,.closed_issues]|@tsv'   # must print 0<TAB>0
gh api -X PATCH "$R/milestones/7" -f description='Platform breadth: model-family training adapters, second segmenter, importers, full API authentication, embedding swap, search follow-ups and research items.'
for n in 84 347 365 367 368 369 370 398 406 416 435 436; do gh issue edit "$n" --milestone v0.8.0; done
```
#402 is closed in GH-8 (C-402), so it is not moved.

**GH-5 Put the B0-B4 issues on v0.7.0.**
```bash
for n in 437 439 440 442 443 445 447; do gh issue edit "$n" --milestone v0.7.0; done
```

**GH-6 Close #366 as done** (section 5, comment C-366).
```bash
gh issue edit 366 --milestone v0.7.0
gh issue close 366 --reason completed --comment "$(cat <scratch>/gh/C-366.md)"
```

**GH-7 Create N1-N16** (section 4). Write each body to `<scratch>/gh/Nn.md` from section 4, then:
```bash
gh issue create --title "<title>" --label "<labels>" --milestone "<milestone>" --body-file <scratch>/gh/Nn.md
```
Append `Nn<TAB>#<number>` to `artifacts_local/gh/created.tsv` after each one. Then replace the
`N2`..`N7` placeholders in N1's and N14's bodies with the real numbers (`gh issue edit <n> --body-file`).

**GH-8 Post the re-scope comments** (section 5) and edit each body's scope line where the comment
says so: `gh issue comment <n> --body-file <scratch>/gh/C-<n>.md`. C-370 also moves #370 to
v0.8.0 and edits its scope line. C-402 first moves #402 to v0.7.0 (`gh issue edit 402
--milestone v0.7.0`), then closes it (`gh issue close 402 --reason completed --comment ...`).
The moved-doc comments (section 5) run when the removal wave R-1 lands, not here.

**GH-9 Retitle #447** to the corrected premise:
```bash
gh issue edit 447 --title "fix(export): YOLO26 exports the raw one-to-many head unless nms=False, and YOLO26 uploads route to the YOLO11 toolchain"
```

**GH-10 Dependabot** (section 8): close #422 and #432 with their comments now; close #423, #433
and #434 with theirs once the B1 deps lane has committed the re-applied changes.

**GH-11 Close v0.9.0 and v0.10.0** once each is empty (never delete them):
```bash
for m in 8 9; do
  gh api "$R/milestones/$m" --jq '[.open_issues,.closed_issues]|@tsv'   # must print 0<TAB>0
  gh api -X PATCH "$R/milestones/$m" -f state=closed
done
```

**GH-12 Verify** (section 9) and save `milestones_after.json` and `issues_after.json` next to the
"before" files.

## 4. New issues N1-N16

Every body ends with the same footer: "Plan: `docs/design/program/<file>`. Task detail by ID in
`<source plan>`. Tracked in the 0.7.0 program (`docs/design/program/00a_decision_register.md`)."

| N | Title | Labels | Milestone | Batch | Closes when |
|---|---|---|---|---|---|
| N1 | `epic(jobs): run background jobs on Celery workers and make the API stateless` | epic, area:curation, effort:hard | v0.7.0 | B2-B4 | B4 merges |
| N2 | `fix(curation): class registry moves to an OpenSearch document with optimistic concurrency` | bug, area:curation, effort:hard | v0.7.0 | B1-1B | B1 merges |
| N3 | `fix(curation): per-project job leases in OpenSearch replace start file locks` | bug, area:curation, effort:hard | v0.7.0 | B1-1A | B1 merges (the ratchet's last step is tracked in N1) |
| N4 | `fix(api): per-process caches, response bodies and blocking image work across API workers` | bug, effort:moderate | v0.7.0 | B1-1D | B1 merges |
| N5 | `fix(inference): memory, retry, timeout and readiness hygiene in the Triton clients and search` | bug, area:gpu-serving, area:search, effort:moderate | v0.7.0 | B1-1C (OS-3 in B1-1A) | B1 merges |
| N6 | `fix(frontend): diverse review uses one batch read; export stats refresh; SSE resync; tests that cannot fail` | bug, area:frontend, effort:moderate | v0.7.0 | B1-1D | B1 merges |
| N7 | `fix(models): secondary detector class by name, and original YOLOv5 ONNX for the probe and secondary` | bug, area:ingest, effort:moderate | v0.7.0 | B1-1E | B1 merges |
| N8 | `build(ultralytics): one exact Ultralytics version and one fused [batch,300,6] detection contract` | chore, area:gpu-serving, area:training, effort:hard | v0.7.0 | B3 | B3 merges |
| N9 | `security: harden persisted clustering state` | security, effort:quick | v0.7.0 | B1-1D (SEC-3) | B1 merges |
| N10 | `security: widen the CLI secret redaction key list` | security, effort:quick | v0.7.0 | B1-1D (SEC-6) | B1 merges |
| N11 | `security: reduce the privileges of an opt-in monitoring service` | security, effort:quick | v0.7.0 | B2 stage 6, lane 6C (SEC-4) | B2 merges |
| N12 | `security: pin a build input and review per-service container hardening` | security, effort:moderate | v0.7.0 | B2 stage 6, lane 6C (SEC-5) | B2 merges |
| N13 | `question(detect): should /detect upscale small images like ingest does` | question, area:gpu-serving | v0.7.0 | B5a (D-35) | B5a decides |
| N14 | `chore(audit): evidence tasks before acting` | chore, effort:moderate | v0.7.0 | B2, B4, B5c, backlog | every item checked |
| N15 | `chore(api): typed response models, one router per PR` | chore, effort:hard | v0.7.0 | backlog after B4 | every router typed (a long tail; an open remainder moves to a v0.8.0 issue at the tag) |
| N16 | `feat(loop): pause control, event stream and token budgets for the assist loop` | enhancement, area:curation, effort:moderate | v0.8.0 | after B2 stage 4 lane 4D-2 (the assist loop on Celery, pause and resume through the jobs table) and stage 3 lane 3B (events in Postgres) | pause route, event stream and `usage` budgets work |

Dropped earlier and not to be filed: the separate ratchet issue (folded into N3), a separate
#447 follow-up (folded into the #447 re-scope), a separate decode issue (folded into the #40
re-scope), a separate OCR-concurrency issue (folded into the #220 re-scope), a B0 traceability
issue (B0 is referenced by its PR and the CHANGELOG).

### N1 body

```markdown
## Problem
Jobs started by a request (17 families: reprocess, assist, item scores, probe, selection,
embedding viz, region clustering, promote, dataset import, combine, bulk ingest, open-vocab
drain, project delete, directory ingest and the rest listed in the plan) run inside API worker
processes. A worker recycle kills them
mid-run, and the periodic loops (dispatcher, sweeper, cache prune, cluster refresh, GPU arbiter)
run once per API worker instead of once.

## Scope
- PostgreSQL holds every control record (projects, classes, configs, jobs, training runs, models);
  the `jobs` row is the truth and Celery 5.6 on Valkey 9 is the transport; `task_id = job_id`; a
  claim `UPDATE` with a fencing token admits one delivery ([02](02_control_plane_database.md)).
- Postgres and Valkey with no published port, generated password files, Valkey AOF everysec and
  `noeviction`; one-shot `db-migrate`; core workers `jobs-worker`, `periodic-worker`,
  `open-vocab-worker`, `jobs-beat` (one scheduler, enforced by a database lock); Flower under the
  monitoring profile.
- Every family moves off the API, training, model export and the bake-off on GPU queues; the
  periodic loops move to beat; the reaper re-queues resumable jobs; the job files, the file
  protocols and every file lock are deleted; events reach the browser from Postgres.
- Then the API is stateless (B4): request fan-out becomes one read plus one bulk write with
  per-item optimistic concurrency, and the worker count is one `API_WORKERS` knob.
- Children: N3 (leases), N11, N12, #439.

## Acceptance
- No job runs as a task inside an API worker; lifespan only creates per-process resources.
- The same scripted scenario gives the same outcomes with `API_WORKERS=1` and `=32`.
- `rg -n "fcntl" src` is empty; the file-lock ratchet baseline is zero.
- Jobs survive an API restart; a killed worker's resumable job is re-enqueued and completes.

Plan: docs/design/program/03_b2_overview.md and 05_b4_stateless_api.md. Task detail by ID in
docs/design/task_queue_plan.md (Q-Phase 2-7, Q-D1..D15, Q-E1..E26).
```

### N2 body

```markdown
## Problem
Two concurrent class writes (two users, or a user plus an import or a combine) can lose a class
or give two classes one id, because the project's class registry is a JSON file that every API
worker reads, changes and writes back.

## Scope
- One registry document per project in the project's settings role index, written with
  `if_seq_no`/`if_primary_term`; at most 5 attempts, then 409 `registry_conflict`; cap 10,000
  classes (409 `registry_full`); `add_class` idempotent by name.
- Readers keep a per-process snapshot keyed by the document's `_seq_no` (fixes stale class names
  per worker).
- `class_registry.json` becomes an export snapshot only; the trainer keeps reading the export's
  own snapshot.
- Upgrade: a project whose file exists but whose document does not answers 409
  `registry_not_migrated` on class routes (fail closed). `openprocessor curation migrate-registry
  [--project <slug>|--all] [--dry-run]` seeds the document, keeps ids, renames the file to
  `class_registry.json.migrated`. `openprocessor upgrade` runs a dry run, saves it, then
  migrates; a failure stops the upgrade. A downgrade note is in INSTALLATION.md.

## Acceptance
- 8 spawned `add_class` calls at a barrier against a fake that enforces `if_seq_no`: N classes,
  N distinct ids, 100 runs; live two-worker check.
- Migration: dry run writes nothing; real run creates the document and renames the file; rerun
  is a no-op; a conflicting document is an error.
- Release acceptance upgrades a real 0.6.0 install with classes and finds the ids equal.

Plan: B1 1B (merged; see ARCHITECTURE.md). Task detail: docs/design/codebase_audit_plan.md CC-2.
```

### N3 body

```markdown
## Problem
Singleton job starts are guarded by file locks, four starts (auto-label, selection, embedding
viz, region clustering) have no cross-process guard, stale-job repair can flip a sibling's live
job, and a blocked event loop can cost a live job its liveness.

## Scope
- A lease document per job kind and scope in the project's settings role index: create-only
  acquire, `if_seq_no`/`if_primary_term` renew, takeover only of an expired or released lease.
- Heartbeat in a thread (15 s tick, TTL 120 s), so a blocked loop does not lose the lease.
- Cooperative fencing: a renew check before every progress and terminal write; records carry
  `lease_generation`.
- Every start-lock site converted; the four unguarded starts gain the lease; reconcile takes
  over only expired leases.
- Trainer and training-job claims use create-only generation files (no OpenSearch client there).
- A shrink-only ratchet on file locks; it reaches zero when the event log moves (N1).

## Acceptance
- 8 spawned processes acquire one lease at a barrier: exactly one wins, 20 live repetitions.
- A handler that blocks the loop longer than the TTL keeps its lease.
- A renew after a takeover returns lost; the old holder writes nothing more.
- No caller of the file-lock helpers remains outside the module except the sites scheduled for
  the Celery and event-log work.

Plan: B1 1A (merged; see ARCHITECTURE.md). Task detail: docs/design/task_queue_plan.md
Q-D2, Q-D4, Q-D7, Q-D9, Q-D12; docs/design/codebase_audit_plan.md OS-3.
```

### N4 body

```markdown
## Problem
With many API workers, per-process caches behave as if they were global: class names, settings,
queries and eval datasets can be stale or can cache an error. Curation responses carry timing
keys that are not in the contract, 500 bodies name an exception class, and thumbnail and render
handlers block the event loop.

## Scope
Audit BA-1 (class-name cache keyed by the registry revision), BA-2 (undeclared timing keys
removed), BA-3 (`cachetools` for the bounded caches, licence row added), BA-4, BA-5 (`error_type`
removed from 500 bodies; contracts regenerated), BA-8 (the per-process probe cap documented),
CC-5 (thumbnail and render handlers offload their blocking work once per request).

## Acceptance
Each item's red test fails on the base and passes after, with a paired control; contracts and
the API contract doc updated in the same commits.

Plan: B1 1D (merged; see CHANGELOG.md). Task detail: docs/design/codebase_audit_plan.md section 4.
```

### N5 body

```markdown
## Problem
Sliver crops allocate gigabytes, images are decoded before the size check, deterministic errors
are retried, some synchronous calls have no timeout, model residency never learns of an unload;
the kNN class filter is a post-filter (filtered search returns fewer than k hits), paging is
unbounded, scroll contexts leak on error, and PIT slices can be orphaned.

## Scope
Audit GPU-4, GPU-5, GPU-8, GPU-9, GPU-10, GPU-16, GPU-17 (with API-12); OS-3 (job success
states honour bulk item errors, done with the lease work), OS-5, OS-6 (filter inside the ANN
query), OS-7, OS-9, OS-10; CC-6; CC-8.

## Acceptance
Each item's red test and control from the audit plan; a live check that a filtered search
returns k hits.

Plan: B1 1C (merged; see CHANGELOG.md) (OS-3 in B1 1A (merged; see ARCHITECTURE.md)). Task
detail: docs/design/codebase_audit_plan.md section 4.
```

### N6 body

```markdown
## Problem
Diverse review fans out 30 crop fetches although a batch read-by-ids route exists; the export
page's stats never refresh; link preload binds the project during `load`; an SSE reconnect does
not resync keymap and config. Some tests cannot fail or skip on the very failure they guard, a
test dependency is missing, and one test file fails when run after two others in one process
(order pollution hidden by xdist).

## Scope
Audit FE-1, FE-2 (use the existing `?ids=` route), FE-3, FE-5, FE-7; T-1, T-2, T-3, T-4, T-7;
T-8 (find the shared process state between the project clone tests and the cross-project
keymap/ingest-policy isolation test and reset it in a fixture; add an ordered no-xdist run as
the red test).

## Acceptance
Each item's red test and control; browser check of the export and diverse review pages at 1440
and 430 px with the screenshots viewed.

Plan: B1 1D (merged; see CHANGELOG.md). Task detail: docs/design/codebase_audit_plan.md section 4.
```

### N7 body

```markdown
## Problem
The ingest secondary detector stores a raw model output index as the class id, which breaks
class identity by name. The YOLOv5 objectness probe hard-codes a 1280 input, so any other
static export fails.

## Scope
- Map the secondary detector's index through the model's own names to a class name, then
  resolve the registry id by name (the probe already does this).
- The probe reads its input size from the ONNX and refuses a wrong output shape.
- Original YOLOv5 is supported as bring-your-own ONNX exported by the official `ultralytics/yolov5`
  `export.py`, for the probe and the secondary only; YOLOv5u is the path for GPU serving. The
  original `.pt` is refused with a message naming YOLOv5u.
- Tests on real exports (fetched in a fixture, never committed), in the CI `families` job.

## Acceptance
A registry with a gap gives the right class by name (red today); a real 640 static official
export runs through the probe (red today).

Plan: B1 1E (merged; see CHANGELOG.md). Task detail:
docs/design/model_family_versions_plan.md Step 3, C8, C10, J3.
```

### N8 body

```markdown
## Problem
Three Ultralytics versions ship: the API and export images (8.4.197, exact since the 0.7.0
close-out), an isolated YOLO11 venv at 8.3.253 with a vendored patch that targets a deprecated
TensorRT plugin, and the trainer at a git SHA of 8.4.48. Export defaults change inside 8.4.x, so
the same model can get a different head depending on which code exports it.

## Scope
- One exact release X in every image and requirements file (X chosen by the freeze-date rule
  recorded in the PR); the trainer moves to the SHA of tag vX in the same batch.
- One export module; every family exports through Ultralytics' documented `nms` argument to one
  fused `[batch, 300, 6]` head (`nms=False` for heads with the one-to-one branch, `nms=True`
  otherwise); export batch equals Triton `max_batch_size`, guarded.
- Delete the YOLO11 venv, its requirements file, the re-exec and the vendored patch (Stage A);
  replace the patch with `nms=True` if the GPU gate passes (Stage B).
- Delete the adapter heuristics the fused contract makes dead; the installer's hand-written
  YOLO11 config goes; the first slice of #368 (a model-family module).
- The CI `families` job runs the full CPU matrix at the pin.

## Acceptance
- CPU matrix green at X in CI and in each image (`importlib.metadata` check).
- GPU gate once for Stage A and Stage B: |delta mAP50-95| <= 0.3 and >= 99 % detection agreement
  on COCO val2017; throughput >= 95 % at batch 1/8/32/64 with p99 within +10 %; promote path at
  the new pin serves `[300,6]`; embedding cosine >= 0.999.
- If Stage B misses throughput, Stage A ships and the measurement goes to the owner.

Plan: docs/design/program/04_b3_ultralytics_one_version.md. Task detail:
docs/design/model_family_versions_plan.md Steps 4-9, sections 6.1 and 6.2.
```

### N9-N12 bodies (security, neutral)

```markdown
Hardening item from the 0.7.0 audit. Details are in the maintainers' private notes, not in this
issue. It is fixed in batch <B1 | B2> of the 0.7.0 program, and the PR references this issue.

Plan: docs/design/program/<B1 1D (merged; see CHANGELOG.md) | 03f_stage6_scheduler_scaleout.md>.
```

N9 and N10 use 02d (B1); N11 and N12 use 03c (B2).

### N13 body

```markdown
## Question
`/detect` letterboxes without upscaling small images (`scaleup=False`, the Ultralytics
validation behaviour), while ingest upscales. Should `/detect` upscale too?

## How it is decided
Measure recall on COCO val2017 (5k images) with and without upscaling, on the final engine,
with the B5a measurement tooling. Keep `scaleup=False` until then (D-35). The owner decides on
the numbers.

Plan: docs/design/program/06a_b5_baseline.md. Task detail: docs/design/codebase_audit_plan.md GPU-6.
```

### N16 body

```markdown
## Problem
The assist loop can start, watch, cancel, resume and roll back, but has no pause control and no
event log, because the API serves neither (`LoopStore` already has a `pause.flag` and the state
has `pause_reason`). Remote VLM endpoints have no token budget because the client does not read
the `usage` object. The label rate that sizes the budgets was never measured (S0).

## Scope
- Measure the label rate first (S0 design in `docs/design/auto_label_loop_plan.md` section 13).
- A pause route on the B2 consumer (`03b` 4D-2).
- Events from the Postgres event feed (B2 stage 3, `03e`).
- `usage` tokens read from the OpenAI-compatible response.
- Cropwright pause button and event log.

Depends on: B2 stage 4 lane 4D-2 and stage 3 lane 3B (the former 2E-2 and B4 4C).
Plan: docs/design/program/03f_stage6_scheduler_scaleout.md and 05_b4_stateless_api.md.
```

### N14 body

```markdown
Measure first, act only on a measured defect. Each item closes in its batch.

- [ ] OPS-2: Prometheus multiprocess files of recycled workers (measure before the Celery batch; B2 stage W3)
- [ ] CC-10: peak in-flight VLM chunk fan-out with a stub VLM (measure before B4 track 4A)
- [ ] GPU-19: OCR detector normalisation (folded into the P-ocr gate of B5c)
- [ ] OS-8: delete-by-query failures (backlog)
- [ ] FE-6: pager memory (backlog)
- [ ] T-6: FCFS test timing (backlog)

Items still open when v0.7.0 is tagged move to a follow-up issue on v0.8.0 (or this issue moves).

Plan: docs/design/program/00a_decision_register.md (backlog). Task detail: docs/design/codebase_audit_plan.md section 4.
```

### N15 body

```markdown
## Problem
71 operations have no declared response model, so the contract does not describe their bodies.

## Scope
One router per PR, after the API-shaping batches (B2, B4) so each router is typed once;
contracts regenerate in the same PR. Audit BA-6.

## Acceptance
Every operation has a response model; `make contracts-check` clean; no behaviour change.

Plan: docs/design/program/00a_decision_register.md (backlog). Task detail: docs/design/codebase_audit_plan.md BA-6.
```

## 5. Re-scope comments (ready to post)

Post each as a comment (`gh issue comment <n> --body-file`) and edit the body's scope line where
it says so.

**C-447** (also GH-9 retitle):
```markdown
Re-scope after the version runs. Root cause: in Ultralytics 8.4, an unset `nms` (`None`) exports
the raw one-to-many head ("`None` exports raw one-to-many predictions for external NMS", export
docs). The 8.4.48 trainer's checkpoint still exported `[300,6]` by default; 8.4.174 and later do
not. `end2end` is deprecated and mapped to `nms` (maintainer comment on the export page), which
explains the `end2end=True` discrepancy. The row "`yolo26n.pt` + `nms=False` -> raw head" did not
reproduce. Also in scope: a YOLO26 upload is routed to the YOLO11 toolchain, because
`model.end2end` is not a stable flag across 8.4 releases.
Fix (B1, track 1E of the 0.7.0 program): export with `nms=False` and refuse any output that is
not `[300,6]`; route uploads by head kind (the one-to-one branch), not by `end2end`. The
trainer's export moves with the single-version work (B3). Milestone v0.7.0.
Plan: B1 1E (merged; see CHANGELOG.md).
```

**C-439:**
```markdown
The concurrent index creation was reverted on the R2 branch; the 13-20 s is not attributed yet.
Next (B4, track 4D of the 0.7.0 program): stage timings with the #414 timer pattern, numbers
posted here; then project creation as a job that returns 202 with the project in `building`,
like a combine target, not request-path fan-out (D-32, D-32a, open). Milestone v0.7.0.
Plan: docs/design/program/05_b4_stateless_api.md.
```

**C-42:**
```markdown
The `small` profile shipped (#441, CHANGELOG "Added"). Only the acceptance on a 12 GB card
remains: small-target engines built, Triton READY with peak memory under 11 GB, the live suite
passing, set A throughput recorded, embedding cosine p01 >= 0.995. It runs when GPU 1 is free and
never blocks a batch; if it has not run by the v0.7.0 tag, the release notes say the profile is
not yet verified on a 12 GB card. Milestone v0.7.0 (side track).
Plan: docs/design/program/06_b5_overview.md section 11 (side track).
```

**C-368:**
```markdown
The single-version work (B3 of the 0.7.0 program) delivers the first slice: a model-family
module (`src/domain/model_families.py`) used by the probe, the validator and the exporter. This
issue keeps the per-family training adapters and the generic single-class exporter. Added risk:
the probe's head versus the served head for YOLO26 must agree (model-family plan U-R8).
Milestone v0.8.0.
Plan: docs/design/program/04_b3_ultralytics_one_version.md.
```

**C-220:**
```markdown
Width-bucket padding fails the P-ocr gate (comment of 2026-11). Next, in B5c: GPU-7, `async_exec`
in the OCR BLS so Triton's dynamic batcher batches per-crop widths (no padding), or a
padding-aware CTC/argmax in the graph; uint8 input; face alignment on resident frames; the step 3
research. This absorbs the former "OCR crops concurrently" item. Milestone v0.7.0.
Plan: docs/design/program/06c_b5_crops_ocr_faces.md.
```

**C-407:**
```markdown
C2's launch-overhead premise is not supported by a measurement yet: profile first. GPU-12
(half-pixel letterbox undo) joins this issue's byte-parity gate. B5b, milestone v0.7.0.
Plan: docs/design/program/06b_b5_decode_once.md.
```

**C-40:**
```markdown
Child list update: GPU-3b (one EXIF-oriented decode helper in `src/utils/` for `/embed`,
`/analyze` and legacy ingest; a behaviour change, in the CHANGELOG) is step 0, before the
resident-frame waves. The #45 baseline (B5a) comes first. B5b, milestone v0.7.0.
Plan: docs/design/program/06b_b5_decode_once.md.
```

**C-411** (edit the body's checklist too):
```markdown
Checklist update for the single 0.7.0 release (D-37: nothing is tagged until every plan is
implemented and tested):
- [ ] B0: R2 round, B0 small fixes and the exact Ultralytics pin merged to main
- [ ] B1 correctness, B2 jobs on Celery, B3 one Ultralytics version, B4 stateless API, B5 speed merged
- [ ] Release prep (VERSION 0.6.0 -> 0.7.0, CHANGELOG `[Unreleased]` -> `[0.7.0]`)
- [ ] R-final release acceptance PASS (it includes every check the old R3 listed, the segmenter
      tier, and the upgrade from a real 0.6.0 install), run last, after B5
- [ ] Tag v0.7.0 on the owner's word; images, release and post-release checks
There is no intermediate release or tag. This issue closes after the tag.
Plan: docs/design/program/00a_decision_register.md, 13_b0_close_out.md, 07_release_acceptance.md.
```

**C-366** (then close, GH-6):
```markdown
Done. Ingest profiles are a revisioned config kind next to prompt packs and region profiles
(`/curation/projects/{project}/ingest_profiles`: list, create, get, revisions, validate,
activate, rollback, deactivate, clone), selectable per project at runtime; CHANGELOG
`[Unreleased]` "Ingest profiles (#366)". Merged to main in PR #438. The Cropwright picker is
#440 (merged with the 0.7.0 close-out). Closing as completed.
```

**C-402** (move #402 to v0.7.0 first, because v0.10.0 must be empty to close; then close; GH-8):
```markdown
Phases A and B shipped: `/curation/projects/{project}/search/similar`,
`/curation/projects/{project}/search/image` and the `target` option
(`src/routers/curation/search.py`, `search_image.py`). What remains is tracked in #435
(`target=frame`) and #436 (`group_by=image`). Closing as completed.
```

**C-370** (re-scope; GH-8):
```markdown
Every format on the 2026-10-31 list shipped (PR #426 and `195a7697`: VOC, Open Images, CVAT,
Label Studio, KITTI, CreateML, LabelMe, YOLO names files, YOLO-seg and OBB as boxes). Only COCO
segmentation-only annotations as boxes remain. This issue is re-scoped to that one item
(milestone v0.8.0). It will use the B1 class registry by name. Format guide:
`docs-site/docs/guides/dataset-import.mdx`.
```

**Moved-doc comments.** Run when the removal wave R-1 lands ([13_b0_close_out.md](13_b0_close_out.md)
B0-1b; target list in [12_prior_plans_reconciliation.md](12_prior_plans_reconciliation.md) section
6). Post one comment per open issue whose body cites a moved or deleted design doc, giving the new
path: "The design doc `<old>` was retired; the content now lives in `<new>`."

| Issue | Old doc | New place |
|---|---|---|
| #40, #67, #219, #231, #268 | `v060_wave_plan.md` | `docs/PERFORMANCE.md` section "Wave 0 parity and FP16 gates" |
| #398, #400, #401, #402 (closed by C-402), #403, #406 | `improvement_backlog.md` | #407 and the issue itself; lessons in `docs/design/engineering_standards.md` section 5 |
| #365-#370 | `program_plan.md` | `docs/design/program/00a_decision_register.md` (this program) |
| #40 | `ingest_write_path_investigation.md`, `v060_vector_db_evaluation_plan.md`, `v060_serving_backend_trade_study.md` | the same names under `docs/research/` |
| #45 | `storage_sizing_and_ingest_baselines.md` | `docs/research/storage_sizing_and_ingest_baselines.md` |
| #67 | `v060_vector_db_evaluation_plan.md`, `v060_serving_backend_trade_study.md` | `docs/research/` |
| #231 | `v060_serving_backend_trade_study.md` | `docs/research/v060_serving_backend_trade_study.md` |

Docs that stay need no comment now (their R-3 removals comment later): `triton_pipeline_optimization_plan.md`,
`sp_gpu_prep_concurrency.md`, `faces_ocr_resident_crops_plan.md`, `gpu_decode_plan.md`,
`cluster_search_makemodel_plan.md`, `selectable_assist_plan.md`,
`generic_detector_and_selective_embedding_plan.md`.

## 6. Close table

| When | Issues | How |
|---|---|---|
| GH-6 (with D-36) | #366 | comment C-366, close as completed |
| GH-8 | #402 | comment C-402, close as completed |
| B0 PR merges | #437, #440, #442, #443, #445 | "Closes #n" lines in the B0 PR body ([13_b0_close_out.md](13_b0_close_out.md) B0-4); verify each `CLOSED` |
| B1 PR merges | #447, N2, N3, N4, N5, N6, N7, N9, N10 | closing keywords in the B1 PR body; verify |
| B2 PR merges | N11, N12 | closing keywords |
| B3 PR merges | N8 | closing keyword |
| B4 PR merges | N1, #439 | closing keywords |
| N14 items | N14 | each box checked in its batch PR; the issue closes when all are checked |
| B5 sub-PRs | #45, #268 (B5a), N13 (B5a decision), #40, #407 (B5b), #219, #220 (B5c), #67 (B5d) | closing keywords in each sub-PR as its acceptance is met |
| v0.7.0 tagged (last) | #411; then milestone v0.7.0 closed when it has no open issue | comment with the release link and the report location ([07_release_acceptance.md](07_release_acceptance.md) section 7) |
| Side track | #42 | after the 12 GB run |
| Re-check | after B3: if Stage B ships, any request for EfficientNMS-specific work is obsolete | comment and close |

Closing keywords close an issue when the PR merges into the default branch (`main`); with no
intermediate tags, "fixed on main" is the closing point and the milestone records the release.

## 7. Issue map: every open issue (37) after the moves

| Issue | Milestone now -> after | Disposition |
|---|---|---|
| #8 epic(video) | none (parked) | defer, stays parked; needs the job system (B2) and the GPU pipeline (B5) |
| #40 decode once | v0.9.0 -> v0.7.0 | B5b (C-40) |
| #42 small GPU | v0.9.0 -> v0.7.0 | side track on GPU 1, does not block the tag (C-42) |
| #45 benchmark ladders | v0.9.0 -> v0.7.0 | B5a, baseline before any optimization |
| #67 processing plans | v0.9.0 -> v0.7.0 | B5d (phase 2+, ledger leftover) |
| #84 wheel example | v0.10.0 -> v0.8.0 | defer, after training stabilises (B3) |
| #219 crops at model size | v0.9.0 -> v0.7.0 | B5c |
| #220 OCR and faces | v0.9.0 -> v0.7.0 | B5c (C-220), includes the step 3 research |
| #231 serving head-to-head research | v0.9.0 -> v0.7.0 | B5 deferred row: owner gate, after #40 lands ([06](06_b5_overview.md) section 10) |
| #268 Triton follow-ups | v0.9.0 -> v0.7.0 | B5a |
| #346 clustering research tracker | none (parked) | defer, stays parked; graph clustering is not in this program |
| #347 embedding-model research | v0.10.0 -> v0.8.0 | defer |
| #348 spatial research | none (parked) | defer, stays parked |
| #365 embedding swap | v0.10.0 -> v0.8.0 (blocked) | defer; reuses the registry upgrade-hook pattern when built |
| #366 ingest_profile kind | v0.10.0 -> v0.7.0, closed | done on main (C-366) |
| #367 second segmenter | v0.10.0 -> v0.8.0 | defer |
| #368 model-family registry | v0.10.0 -> v0.8.0 | first slice in B3 (C-368); remainder deferred |
| #369 API auth | v0.10.0 -> v0.8.0 (blocked) | defer; B2 adds no public surface (the broker has no published port) |
| #370 importers | v0.10.0 -> v0.8.0 | re-scoped to COCO segmentation-only annotations as boxes (C-370); every other format shipped; uses the B1 registry by name |
| #398 detector-class roots | v0.10.0 -> v0.8.0 | defer (research) |
| #400 attribute signals | none (parked) | defer, stays parked |
| #401 plate identity | none (parked) | defer, stays parked; needs a privacy design |
| #402 similar-crop search | v0.10.0 -> v0.7.0, closed | phases A and B shipped; closed with C-402 (the rest is #435, #436) |
| #403 MPS overlay | v0.9.0 -> v0.7.0 (parked) | B5 deferred row; adopt only with >= 10 % gain measured in B5 |
| #406 judge calibration | v0.10.0 -> v0.8.0 | defer (research) |
| #407 GPU-prep speed | v0.9.0 -> v0.7.0 | B5b (C-407) |
| #411 0.7.0 close-out | v0.7.0 | closes after the tag (C-411) |
| #416 near-dup threshold | v0.10.0 -> v0.8.0 | defer (research) |
| #435 frame-target search | none -> v0.8.0 | defer; decide whether it belongs in curation search |
| #436 group_by=image | none -> v0.8.0 | defer; not needed by the current UI |
| #437 YOLO26 region decode | none -> v0.7.0 | B0, closes on merge |
| #439 project create 13-20 s | none -> v0.7.0 | B4-4D (C-439) |
| #440 Cropwright ingest profile | none -> v0.7.0 | B0, closes on merge |
| #442 VLM activate not_found | none -> v0.7.0 | B0, closes on merge |
| #443 cluster refresh mid-ingest | none -> v0.7.0 | B0, closes on merge |
| #445 COCO test images | none -> v0.7.0 | B0, closes on merge |
| #447 YOLO26 raw head | none -> v0.7.0 | B1-1E (C-447, retitled) |

New: N16 (v0.8.0, section 4). Cut outright: none. Every open issue has a home or a stated deferral.

## 8. Dependabot PRs and the deps lane

Rules (owner decision 9):
- No dependabot PR is merged on GitHub straight into `main` during the program: it would move
  `main` under the open integration branches without a batch sweep.
- Bumps land through **the separate deps lane `b1/deps`** (brief in
  B1 (merged; see [CHANGELOG](../../../CHANGELOG.md)) section 4.1; a later batch gets one only when PRs are
  open): the lane re-applies each PR's manifest change on its own branch, one commit per PR
  (`build(deps): ...`), runs targeted checks, and the coordinator merges it into the batch's
  integration branch like any lane, so the batch sweep tests it. Then the PR is closed with a
  comment naming the commit.
- **Ultralytics never floats.** It stays in the dependabot `ignore` list; the exact pin moves only
  in B3 by the D-28 rule. The TensorRT pins move only together with the Triton image.
- A security-group dependabot PR is taken in the current batch's deps lane, not held.

The per-package rules (which pins are exact, which caps stay, the new `ignore` entries) are
owned by [08_environment_and_setup.md](08_environment_and_setup.md) section 7.2 and ENV-12; this
section owns when the PRs are taken and the GitHub side. Where the two disagree, 08 wins on
package content and this file on sequencing.

| PR | What it bumps | Decision | Where |
|---|---|---|---|
| #434 | `onnxruntime-gpu` cap `<1.25` -> `<1.32`, `fastapi` 0.143.3 -> 0.162.0, `pydantic` 2.14.2, `ruff` 0.17.3, `mypy` 2.4.1, `pytest` 9.2.0 (`frontend/e2e`) | re-apply by hand, then close | B1 deps lane, one commit each: ruff together with the `ruff-pre-commit` hook rev (venv and hook must match); mypy; pydantic; pytest; **fastapi as its own commit** with `make contracts` and the contract tests (`test_the_image_pins_fastapi_as_the_tests_do` keeps image and tests equal). **Reject** the `onnxruntime-gpu` cap widening (`nvidia-modelopt` constrains it; 08 ENV-12 adds the ignore). New ruff or mypy findings are fixed in code; baselines never grow |
| #433 | frontend dev deps: `@sveltejs/adapter-static` 4.0.1, `@sveltejs/kit` 3.1.1, `@types/node` 26.6.6, `knip` 6.44.0 | re-apply, then close | B1 deps lane; frontend full set in the lane's targeted checks |
| #423 | `node` and `nginx-unprivileged` digest-only updates (frontend and docs-site images) | re-apply, then close | B1 deps lane; Trivy of the Cropwright and docs-site images at R-final |
| #432 | Python bases -> 3.14 (root, test, trainer), `nvidia/tritonserver` 26.06 -> 26.09 | close (D-38) | Triton 26.09 moves TensorRT, which must equal `ARG TRT_VERSION` in `Dockerfile.triton` and `tensorrt-cu13` in both export requirements files (`test_tensorrt_pin_matches_the_triton_image_everywhere`), so the PR fails as it stands and every engine rebuilds. If the owner wants the move (D-38), it is done inside B3 as its own commit, behind the U-6.2 gate; otherwise after the tag |
| #422 | Python bases -> 3.15, the same Triton bump, the `alpine/git` digest; it also edits `docker/evaluator/Dockerfile`, which no longer exists on `main` | close (D-38) | stale and superseded; take the `alpine/git` digest by hand in the deps lane only if Trivy asks |

Close comments (GH-10, after the lane commit exists for the first three):
- #434, #433, #423: "Re-applied by hand in `<commit>` on the B1 integration branch (one batched
  deps step, docs/design/program/09_github_actions.md section 8); it reaches `main` with the B1
  merge."
- #432, #422: "Interpreter and Triton base moves are an owner decision (D-38 in
  docs/design/program/00a_decision_register.md): if taken, inside the B3 engine rebuild with the TensorRT pins
  moved together. Closing; dependabot opens a fresh PR when due."

The deps lane also lands ENV-12 (08): the new `ignore` entries and the corrected reason on the
`ultralytics` ignore (today it says the vendored patch breaks on 8.4 and that `requirements.txt`
carries the 8.4 range; both are false since U-C4 and `d1c6bab0`). Keep the `ultralytics` ignore:
the exact pin moves only by the single-version rule (D-28).

### The deps lane brief

The lane `b1/deps` is its own lane in B1; its brief, branch, checks and merge position live in
B1 (merged; see [CHANGELOG](../../../CHANGELOG.md)) section 4.1 (this file keeps only the per-PR decisions above
and the GitHub side). The lane runs only the targeted checks of the
[Test and GPU budget](00a_decision_register.md); the full suite runs once in the B1 sweep.

## 9. Verification after GH-12

```bash
gh api "$R/milestones?state=all&per_page=100" --jq '.[] | select(.title|test("^v0\\.(7|8|9|10)\\.0$")) | [.title,.state,.open_issues,.closed_issues] | @tsv'
gh issue list --state open --limit 100 --json number,milestone --jq '.[] | [.number,(.milestone.title // "-")] | @tsv' | sort -n
gh issue list --state open --label parked --json number,milestone --jq '.[] | select(.milestone==null) | .number'
```

Expected, if the B0 PR has not merged yet:

| Milestone | State | Open | Closed |
|---|---|---|---|
| v0.7.0 | open | 33: #411, #437, #439, #440, #442, #443, #445, #447, N1-N15, #40, #42, #45, #67, #219, #220, #231, #268, #403, #407 | 59: the 35 of today, the 22 moved closed items, #366, #402 |
| v0.8.0 | open | 13: #84, #347, #365, #367, #368, #369, #370, #398, #406, #416, #435, #436, N16 | 0 |
| v0.9.0 | closed | 0 | 0 |
| v0.10.0 | closed | 0 | 0 |

After the B0 merge, five of the v0.7.0 open issues move to closed (28 open, 64 closed). The
unmilestoned open issues are exactly the parked ones: #8, #346, #348, #400, #401.

## 10. Corrections found while splitting

- The former master plan (rev 4, in git history) 9.3 renamed milestones (v0.10.0 -> v0.11.0,
  v0.9.0 -> v0.10.0, a new v0.9.0). Superseded by owner decision 5 and D-37: titles stay,
  contents move, v0.9.0 and v0.10.0 close once empty, v0.8.0 is repurposed.
- #370 was called "importers, defer" and #402 "defer after B4"; both are almost fully shipped
  (C-370, C-402).
- The old N8 row used a `build` label that does not exist; N8 uses `chore`.
- The old #366 re-scope said "re-check after B0". #366 is already done on `main` (PR #438;
  CHANGELOG "Ingest profiles (#366)"); it closes now.
- The old close table closed B1 issues "when v0.8.0 is tagged". With no intermediate release,
  issues close when their batch PR merges to `main`.
- N13 and N15 were on v0.8.0 (speed); with D-37 they are on v0.7.0.
- The old issue map named #231 by a product name; it is "serving head-to-head research" here.

## Research integration (2026-11-04)

The research plans asked for comments on #346, #347, #348, #365, #398, #400, #401, #406, #416,
#435, #466 and #473 and for one new issue (N21, the S3-compatible storage backend). Their exact text
is in [00b_traceability.md](00b_traceability.md) sections 4.5 and 4.6; the comments wait for the
owner's answers to RD-1 and RD-2 ([00d](00d_owner_questions.md)); N21 is approved (2026-11-04) and
filed by the coordinator.
