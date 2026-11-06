# v0.7.0 master plan: the waves from now to the tag (the one file a coding session follows)

## Start here: the files in this folder, in the order they are used

This folder holds every plan for the v0.7.0 backend rebuild. Read this file first; it gives the
order. Agents execute the numbered files in wave order; `reference/` holds the decisions, reviews,
questions and standards they consult.

| Order | File | Used in |
|---|---|---|
| 0 | `00_master_plan.md` (this file) | every wave: the order, lanes, gates |
| 1 | [01_package_pins.md](01_package_pins.md) | W2.0 (part A), W3 (part B) |
| 2 | [02_control_plane_database.md](02_control_plane_database.md), [02a](02a_postgres_schema.md), [02b](02b_jobs_locking_events.md), [02c](02c_migration_ops_tests.md), [02d](02d_research_seam_schema.md) | W2.1-W2.4: the PostgreSQL design, jobs and locking, migration and ops, research tables |
| 3 | [03_b2_overview.md](03_b2_overview.md) | W2.1-W2.4: the job-queue move, overview |
| 3a | [03a_stage1_foundation.md](03a_stage1_foundation.md) | W2.1 |
| 3b | [03b_stage2_control_data.md](03b_stage2_control_data.md) | W2.2 |
| 3c | [03c_stage3_job_runtime.md](03c_stage3_job_runtime.md) | W2.2 |
| 3d | [03d_stage4_job_families.md](03d_stage4_job_families.md) | W2.3a, W2.3b |
| 3e | [03e_stage5_gpu_queues.md](03e_stage5_gpu_queues.md) | W2.3a |
| 3f | [03f_stage6_scheduler_scaleout.md](03f_stage6_scheduler_scaleout.md) | W2.4 |
| 4 | [04_b3_ultralytics_one_version.md](04_b3_ultralytics_one_version.md) | W3 (GPU window 1) |
| 5 | [05_b4_stateless_api.md](05_b4_stateless_api.md) | W4 |
| 6 | [06_b5_overview.md](06_b5_overview.md), [06a](06a_b5_baseline.md), [06b](06b_b5_decode_once.md), [06c](06c_b5_crops_ocr_faces.md), [06d](06d_b5_processing_plans.md) | W5a-W5d (GPU window 2) |
| 6e | [06e_w5e_plate_text.md](06e_w5e_plate_text.md) | W5e |
| 7 | [07_release_acceptance.md](07_release_acceptance.md) | W6: release prep, acceptance, the tag |
| 8 | [08_research_prototype_track.md](08_research_prototype_track.md) | short prototype runs during W2-W4 (about 10 GPU hours) |
| 9a-9h | [09a embeddings](09a_research_embeddings_search.md), [09b clustering](09b_research_clustering.md), [09c clustering protocol](09c_research_clustering_protocol.md), [09d near-duplicates](09d_research_near_duplicates.md), [09e auto-label](09e_research_auto_label_assist.md), [09f spatial](09f_research_spatial_attributes.md), [09g GPU pipeline](09g_research_gpu_pipeline.md), [09h plate text](09h_research_plate_text.md) | the research designs behind the seams and lanes above |
| 10 | [10_v080_platform_breadth.md](10_v080_platform_breadth.md) | after the tag: v0.8.0 |
| 11 | [11_side_track_issue_42_small_gpu.md](11_side_track_issue_42_small_gpu.md) | side track #42 (GPU 1, never blocks) |
| ref | [reference/](reference/): [decision_register](reference/decision_register.md), [owner_questions](reference/owner_questions.md), [traceability](reference/traceability.md), [review_1](reference/review_1_control_plane.md), [review_2](reference/review_2_research.md), [environment_and_setup](reference/environment_and_setup.md), [standards_and_libraries](reference/standards_and_libraries.md), [risks_cleanup_host](reference/risks_cleanup_host.md), [github_actions](reference/github_actions.md), [prior_plans_reconciliation](reference/prior_plans_reconciliation.md), [b0_close_out_done](reference/b0_close_out_done.md) | consulted by every wave |

**Old short names.** Older text in the other files of this folder may still call a file by its earlier number. Read
them as: `00`/`00_index` = reference/decision_register; `00b` = reference/traceability; `00c` =
reference/review_1; `00d` = reference/owner_questions; `00e` = reference/review_2 (or, for the
prototype windows, `08`); `00f` = 08; `01` (B0) = reference/b0_close_out_done; `13` (pins) = 01;
`03a` = 03a; old `03d` = 03b; old `03e` = 03c; old `03b` = 03d; old `03f` = 03e; old `03c` = 03f;
`03g` = 06e; old `06` = 07; old `07`, `07a`-`07d` = 06, 06a-06d; old `08` = reference/environment_and_setup;
`09` = reference/github_actions; `10` = reference/standards_and_libraries; `11` =
reference/risks_cleanup_host; `12` = reference/prior_plans_reconciliation; the `research_*_plan`
files = 09a-09h; `v080_platform_breadth_plan` = 10; `issue-42-plan` = 11. This file and 03 to 03f use only
the new names; the list stays for the files still to be converted.

| Field | Value |
|---|---|
| Status | plan, written 2026-11-04, status as of 2026-11-05: the whole plan set is on `main` (PRs #478, #481, #482, #485). Executes wave by wave on the owner's go |
| What this file is | the **execution order**: every wave from now to the v0.7.0 tag, its lanes, file ownership, sources, issues, folded-in decided items, tests, sweep, exit gate, GPU window, and what codes in parallel. It links to the batch files for task detail and never restates it |
| What it is not | the decision register and the legends: those are [decision_register.md](reference/decision_register.md) (section 6 is the register, section 11 the ID legend, section 2.1 the plain-words map). Where this file and a batch file disagree on order, this file wins; on task content, the batch file wins |
| Traceability | [traceability.md](reference/traceability.md): every original task, ID and open issue mapped to its wave and lane, or retired with a reason; the fold-in record; the GitHub actions (approved 2026-11-04, D-68) |
| Review | [review_1_control_plane.md](reference/review_1_control_plane.md): the independent review (R-1..R-38) and, in its section 5, the planner's response to each finding. Every finding is applied in the plan files; this file carries the wave and lane consequences |
| Owner answers applied | 2026-11-04, after the review: every service requires PostgreSQL, Valkey and OpenSearch, no degraded mode (D-69, which replaced D-64 the same day); event wake-ups over Valkey pub/sub (D-65); kombu and redis-py held (D-66; reversed on 2026-11-05: the latest releases at bump time, Celery 5.7.0, kombu 5.7.0, redis-py 8.1.0); pins part A as its own PR first (D-67, wave W2.0); every open B2 recommendation accepted as a block; `postgres:18.6-trixie`; the security lane in W4 and the typed-response lanes in W5a-W5c; near-duplicate groups held out together; research experiments after the tag; #42 to v0.8.0 if unrun; the GitHub actions of the traceability file (D-68) |
| Research integration | 2026-11-04 on `research/master` (from `research/integration` = `main` + the seven research plans): every research seam is folded into the lane that owns its file (tables in [02d](02d_research_seam_schema.md); the seam-to-lane map, the cross-topic resolutions X-1..X-20 and the issue map in [traceability](reference/traceability.md) section 5); new lanes 1F, ES-A, ES-B, ES-C, 4F, ND-A, ND-B, ND-C, `b4/capture-metadata`, a W2.0 gate lane; a new wave W5e for registration-plate text ([06e](06e_w5e_plate_text.md)); the research prototype track and the post-tag research waves RW-1..RW-6 ([08](08_research_prototype_track.md)); the 80 de-duplicated research owner questions ([owner_questions](reference/owner_questions.md)). Final pass the same day: the research review ([review_2_research.md](reference/review_2_research.md), every finding answered in its Planner responses) and the owner's answers (11 rows approved, among them every seam now, W2.3 split in two, GPS stored per project policy, short prototype runs capped at about 10 GPU hours); 69 questions stayed open; on 2026-11-05 the owner answered 62 as recommended and the plate-text and plate-detector questions, 7 stay open, none before W4 (section 9.3) |
| Owner direction applied | 2026-11-04: foundation first (PostgreSQL for control data, Celery and Valkey for every long job, a stateless API that scales to N replicas); then the speed work on it; all package upgrades with the bumps that can change model output riding the B3 GPU gate; no task dropped except with a reason; decided items folded into the lane that already rewrites their code; plain names first; waves with disjoint lanes and one sweep each; CI path-scoped and sharded |

Plain-words rule: every heading and row names the thing first and puts its code after it, for
example "the job runtime (stage 3, lane 3A)". The batch codes are B2 (control plane), B3 (one
Ultralytics version), B4 (stateless API), B5 (speed), R-final (release acceptance).

---

## 1. Where things stand (2026-11-05)

| Item | State |
|---|---|
| B0 (close out the 0.6 audit round) and B1 (correctness under many workers) | merged on `main` (B1 is PR #467) |
| The plan set | on `main` (PRs #478, #481, #482, #485): the PostgreSQL design (files 02-02d), B2 re-planned in six stages (03-03f), exact pins (01), the research plans, this master plan |
| CI | path-scoped checks (PR #476) and the 4-shard offline suite (PR #477, merged 2026-11-04) are on `main`; nothing to merge before W2.0 |
| Dependabot PRs | #474 (frontend dev group) and #449 (Triton base 26.10) are **closed**. W2.0 regenerates the frontend dev-group bump itself (`npm update` of that group plus the lock) instead of re-applying #474; #449 needs no action (its Python-base question stays D-38 in section 9.2) |
| Decisions | the B2 set approved 2026-11-04 as a block; MQ-3, MQ-4, MQ-8, the Flower port and OI-1 / OI-2 answered 2026-11-04. Owner decisions of 2026-11-05: B2 ships as three PRs (section 2); label operations and their undo run on Celery like every other long job (no inline job path); Celery `revoke` is never used (cancel is a database write; the claim refuses a cancelled, paused or terminal row); packages move to the latest releases at bump time (D-66 reversed); 62 of the 69 open research questions answered as recommended, 7 stay open and none before W4 (section 9.3); registration-plate text is ordinary searchable label data (no privacy modes) and plate boxes come from a detector the user brings (section 9.3) |
| Research plans | seven plans (GPU pipeline, clustering with its protocol file, embeddings and search, near-duplicates, spatial attributes, registration-plate text, the auto-label assistant), each with seams, prototypes, landing lanes and owner questions; integrated 2026-11-04; their questions: section 9.3 |

## 2. The waves in one table

A wave closes with exactly one full sweep on its integration branch. B2 is one integration branch
(`int/b2-control-plane`) in five parts (W2.3 split into W2.3a and W2.3b, review RR-13, owner Q-A of
2026-11-04) that ships as **three PRs to `main`** (owner decision 2026-11-05), each leaving `main`
bootable, no tag between them: **PR1** = W2.1 + W2.2 (database, control data, job runtime; job
execution unchanged); **PR2** = W2.3a plus the two closing stage-6 lanes it makes possible (6A and 6B
part 1: the beat tasks, the deletions of `FileJob`, the leases and the file locks, and the compose for
the ported consumers); **PR3** = W2.3b, the rest of stage 6 (including ES-C and the frontend lanes), the
docs, then the full B2 sweep with live run 2. Each part ends with a short part check (D-63); PR1 and
PR2 add a short live smoke, and only PR3 carries the full sweep. Each PR: local CI parity first, the
owner's yes before the push, merge commit.

| Wave | Plain name | Batch and files | Integration branch | GPU | Closing check | Issues it closes |
|---|---|---|---|---|---|---|
| **W2.0** | Dependency pins part A as their own PR, and the local `make gate` | [01](01_package_pins.md) section 7.1, part A (D-67); the gate lane of section 4 | `deps/pins-part-a`, PR to `main` | none | one offline sweep (CPU), run with the new `make gate` | - (N20 (#480) stays open for part B) |
| **W2.1** | Foundation: PostgreSQL, Valkey, the Celery app, the schema, the storage seam, the test database; the research-seam tables (every seam now, RD-1) | B2 stage 1 ([03a](03a_stage1_foundation.md); 1F and [02d](02d_research_seam_schema.md)) | `int/b2-control-plane` | none; **prototype window PW-1** on idle GPU (short runs, RD-2) | part check | - |
| **W2.2** | Control data moves into PostgreSQL; the job machinery (pause, resume and time slices that never spend attempts) and the event feed; the embedding registry | B2 stages 2 and 3 ([03b](03b_stage2_control_data.md), [03c](03c_stage3_job_runtime.md)) | same | none; PW-1 continues | part check, then **PR1** to `main`: part check plus live smoke (steps 1, 8, 9, 11) | - |
| **W2.3a** | Every long job onto the queue as a port of today's behaviour, training, model export, the evaluator, label operations and their undo included; the assist loop's own record, pause and budgets; then the closing stage-6 lanes (beat tasks and the old job machinery deleted, compose for the consumers) | B2 stage 4 lanes 4A-4D ([03d](03d_stage4_job_families.md)), stage 5 ([03e](03e_stage5_gpu_queues.md)), stage 6 part 1 (6A, 6B; [03f](03f_stage6_scheduler_scaleout.md)) | same | none (real-GPU smoke deferred to GPU window 1); **PW-2** on idle GPU | part check, then **PR2** to `main`: part check plus live smoke (steps 3-7 and 12-18) | #466 (pause, events, budgets), #451, #453 |
| **W2.3b** | The research families on the queue: clustering on versioned partitions; near-duplicate runs, the exact key and its backfill, the holdout guard; the embedding re-index job; Cropwright follows every new state | B2 stage 4 lanes 4F, ND-A, ES-B, ND-B, 4E ([03d](03d_stage4_job_families.md)) | same | none; PW-2 continues | part check (PR3 follows W2.4) | #473 parts a and c |
| **W2.4** | The rest of stage 6: the research ticks on the one beat, monitoring and the PgBouncer overlay, readers on the embedding registry, hardening, docs; **the B2 sweep** with the five stale screenshots recaptured | B2 stage 6 part 2 ([03f](03f_stage6_scheduler_scaleout.md)); sweep [03](03_b2_overview.md) section 9 | same, then **PR3** to `main` | none; PW-2 continues outside live run 2 | **PR3: full B2 sweep + live run 2 of 3** | #461, #462 |
| **W3** | One Ultralytics version and the dependency bumps that can change model output | B3 ([04](04_b3_ultralytics_one_version.md)); pins part B ([01](01_package_pins.md) section 7) | `int/b3-ultralytics` | **GPU window 1** (about 1.5 days, GPU 0) | full B3 sweep + the GPU gate | #458 |
| **W4** | Stateless API: request fan-out out, the worker count a tuning knob, project creation; the privacy and security items; capture metadata and GPS coordinates at ingest under the project's location policy; core near-duplicate grouping off the request path | B4 ([05](05_b4_stateless_api.md)); lanes `b4/privacy-hardening`, `b4/capture-metadata` | `int/b4-stateless` | none (its benchmark runs first in window 2); **PW-3** on idle GPU outside live run 3 | full B4 sweep + live run 3 of 3 | #451, #439, #473, #468, #470, #472 (and #469, #471 once OI-1 and OI-2 are answered) |
| **W5a** | Public baseline, Triton sweeps, the `/detect` upscaling question; typed responses part 1 | B5a ([06a](06a_b5_baseline.md)); typed-response lane | `int/b5a-baseline` | **GPU window 2 opens** | full sweep + 5a runs | #45, #268, #463 |
| **W5b** | Decode each image once; typed responses part 2 | B5b ([06b](06b_b5_decode_once.md)); typed-response lane | `int/b5b-decode-once` | window 2 | full sweep + 5b gate | #40, #407 |
| **W5c** | Crops at model size; OCR on the GPU; typed responses part 3 | B5c ([06c](06c_b5_crops_ocr_faces.md)); typed-response lane | `int/b5c-crops-ocr-faces` | window 2 | full sweep + 5c gate | #219, #220 |
| **W5d** | Request-driven processing plans; faces on resident frames (the routers 5d rewrites are typed here) | B5d ([06d](06d_b5_processing_plans.md)) | `int/b5d-processing-plans` | window 2 | full sweep + 5d gate | #67, #464 (remaining boxes moved), #465 |
| **W5e** | Registration-plate text: per-box readings with a real confidence, burst consensus, exact, fuzzy and fragment search, stored as ordinary readable label text (its own revision, RD-4) | [06e](06e_w5e_plate_text.md) | `int/b5e-plate-text` | **window 2 closes** (a short W5e gate) | full sweep + W5e gate | #401 |
| **W6** | Release prep, the one acceptance run, the tag on the owner's word | R-final ([07](07_release_acceptance.md)) | `chore/release-0.7.0-prep` | **GPU window 3** | R-final (every RF row) | #411 after the tag |
| side | 12 GB card acceptance | [06](06_b5_overview.md) section 11 | `perf/42-small-gpu-acceptance` | GPU 1 when its owner frees it | its own pass list | #42 |
| side | Research prototypes that feed a v0.7.0 decision: short tiered runs, about 10 GPU hours in total (RD-2, RD-7) | [08](08_research_prototype_track.md) sections 2-4 | `research/<topic>-proto` per topic (scripts and notes only) | PW-1 (W2.1-W2.2), PW-2 (W2.3a-W2.4), PW-3 (W4), idle GPU only | each prototype's registered bar | - |
| after the tag | Research waves RW-1..RW-6 (GPU serving breadth, embeddings, clustering, near-duplicate evidence, spatial, the auto-label assistant measured) | [08](08_research_prototype_track.md) section 5 | `int/rw<n>-<topic>` | research GPU windows RG-1..RG-6, not scheduled: each on the owner's OK | per wave | #403, #231, #347, #365, #435, #436, #346, #406, #416, #348, #400, #466, #398, #84 |

The wave codes W2..W6 keep the meaning they had in [decision_register.md](reference/decision_register.md) before this file
(W0 = B0, W1 = B1, both done); W2 gains the five parts and the pins PR W2.0 before them, and ships as
three PRs (D-76); W5e is new (the plate-text wave).

## 3. Dependency diagram

```
main (B0, B1 merged, the plan set and the CI shards on it)
  |
  v
W2.0 pins part A (own PR to main)
  |
  v
W2.1 foundation ---> W2.2 control data + job runtime ==> PR1 to main (part check + live smoke 1, 8, 9, 11)
  |
  v
W2.3a jobs onto the queue (ports) + stage 6 part 1 ==> PR2 to main (part check + live smoke 3-7, 12-18)
  |
  v
W2.3b research families ---> W2.4 rest of stage 6, docs, sweep ==> PR3 to main (full B2 sweep + live run 2 of 3)
  |                                                      |  (B2 done; B3 needs PR3, below)
  |                                    W3 lanes start from the frozen int/b2 head --> W3 B3 + pins part B --[GPU window 1]--> B3 PR
  |                                                                                                          |
  |                                                        W4 lanes start from the frozen int/b3 head --> W4 stateless API --> B4 PR
  |                                                                                                          |
  |                                                    [GPU window 2: 4B-5 benchmark -> 5a -> 5b -> (#231) -> 5c -> 5d]
  |                                                                                                          v
  |                                                                         W5a -> W5b -> W5c -> W5d -> W5e (one PR each)
  |                                                                                                          |
  |                                                                         W6 release prep -> R-final [GPU window 3] -> tag v0.7.0 (owner's word)
  |                                                                                                          |
  |                                                             research waves RW-1..RW-6 [RG-1..RG-6] and v0.8.0
side: #42 on GPU 1 whenever its owner frees it (never blocks)
side (RD-2): short prototype runs on idle GPU, never inside a program window or live run; about 10 GPU h in total
   PW-1 during W2.1-W2.2:   gpu:PR-0..2, nd:P0..2 (3.4 h) --> W5a/W5b decoder choice; nd:P1 seeds ND-A's defaults (W2.3b)
   PW-2 during W2.3a-W2.4:  pt:P1..5 on PP-OCRv6 (1.9 h) --> W5e reader, tier, consensus (W5c takes PP-OCRv6, D-82)
   PW-3 during W4:          gpu:PR-5..6, al:AL0 (1.5 h)    --> W5c, W5d; budget defaults later
   reserve:                 about 2.2 h, tier-2 confirmations only; beyond the cap only on the owner's OK
   CPU slots any time:      clu:CL-0, emb:EP-0, sp:P-0..1 (sp:P-1 gates the W4 capture lane), pt:P0
```

Hard edges: B3 needs B2 merged, that is PR3 (P1 of [04](04_b3_ultralytics_one_version.md) section 2 reads "the B2 PR merge commit is present"; B3's files depend on the PR2 content, the export worker, trainer and evaluator ports, and its pins on PR1, but its GPU gate judges the whole merged B2 stack, so the B3 PR waits for PR3; B3 lanes still code from the frozen integration head during PR3's sweep); B4 needs
B2 and B3 (it benchmarks once on the final engine); B5 measures only the final B4 stack, and no
optimization code merges before the 5a numbers exist; W5e needs W5c (the `rec_char_min` output) and
W4 (the capture time); W2.3b needs the W2.3a part check (it calls the
families W2.3a ported and edits their files); R-final runs after W5e. A prototype never blocks a
wave: a decision whose prototype has not run is taken on today's behaviour and the lane says so.

## 4. The waves in detail

Every lane: one agent in its own worktree `<repo>-wt-<lane>` on its own branch, cut from the
integration branch at the point the wave says; it edits only the files it owns; it runs only its
red-then-green tests and static checks on the files it touches. Lane rules and the common brief
block: [03a](03a_stage1_foundation.md) "Lane brief" (B2), [04](04_b3_ultralytics_one_version.md)
section 17 (B3), [05](05_b4_stateless_api.md) section 17 (B4), each 07 file's "Lane briefs" (B5).

Append-only shared files (`env.template`, `src/config/retired_env.py`, the baselines in
`scripts/codegen/baselines/`, `src/routers/curation/__init__.py`, the register-by-import list: one import line per router) and generated `contracts/` follow the merge rules of
[03](03_b2_overview.md) section 4 in every wave.

### W2.0 Dependency pins part A, as its own PR (D-67)

**Goal.** The packages that cannot change model output move to exact `==` pins at their newest
versions in a small PR of their own, so a dependency regression never blocks or muddies the B2 PRs. Versions are the latest releases at bump time (D-70).

| Order | Lane | Branch | Owns | Brief | Issues |
|---|---|---|---|---|---|
| 1 | Pins part A: tooling, pure-Python libraries, web layer, clients | `deps/pins-part-a` (from `main`) | `requirements.txt`, `requirements-test.txt`, `.pre-commit-config.yaml` (the ruff `rev` moves to v0.17.4), `pyproject.toml` (existing dependency lines only), `tests/test_requirements_contract.py`, `frontend/package.json` and its lock (the dev-group bump of PR #474, which is closed, regenerated with `npm update` of that group plus the lock) | [01](01_package_pins.md) section 7.1, part A | N20 (#480) |
| 1 | The local gate: `make gate` (owner-approved gap, 2026-11-04) | `tools/make-gate` (from `main`, merged into `deps/pins-part-a` before its sweep) | `Makefile` (the `gate` target only), new `scripts/dev/gate.sh`, new `tests/dev/test_gate_script.py`, `CONTRIBUTING.md` (one paragraph) | section 4.0a below | - |

**Sweep and exit.** One CPU offline sweep (the 4 CI shards, run through the new `make gate`, which
is its own first user), pre-commit `--all-files`,
`make contracts-check`, the frontend set (the dev-group bump touches `frontend/`); the CPU image
builds and `pip check` because requirements changed (the CI image scans run once, here); PR to
`main`, merge commit. PRs #474 and #449 are already closed: no close-out action ([traceability](reference/traceability.md) section 4 lists the same). **Cut point (re-check RC-10):** the W2.1
lanes are cut from `main` after this PR merges, so they build on the part A pins (1A uses the
pinned SQLAlchemy, psycopg and Alembic); W2.0 is small, and the W2.1 briefs are written while it
runs.

### 4.0a The local gate lane (`tools/make-gate`, W2.0)

**Why.** The W2.0, W2.4, W3, W4 and W5x sweeps each run the same two tracks by hand; the owner
approved a script (2026-11-04). **What.** `scripts/dev/gate.sh [py|fe|all]` and `make gate`
(`all` runs both tracks in parallel, as the owner's two-track rule says), in this order:

1. **Installs first and alone**, logged: `.venv/bin/pip install -r requirements-test.txt` and
   `.venv/bin/pip check`; `npm ci` in `frontend/` and in `docs-site/`. No other step starts until
   they end.
2. **Python track** (sequential steps): `.venv/bin/pre-commit run --all-files`; the offline suite
   sharded locally with xdist (`.venv/bin/python -m pytest tests/ -q --no-cov -m '<the default
   expression>' -n ${GATE_XDIST:-8}`; the expression is read from `pyproject.toml` `addopts`, so it
   follows lane 1-0's `not db` change without an edit); `make contracts-check`;
   `.venv/bin/python scripts/docs/check_docs_vs_code.py`.
3. **Frontend track** (sequential steps): `npm run check`, `npm run lint`, `npx knip`, `npm test`,
   `npm run build`, `npm run test:e2e` in `frontend/`; `npm run build` in `docs-site/`.
4. Every step writes `<out>/<step>.log` and one summary line `== <step> rc=<exit code> <start>
   <end> load=<1-min load>` to `<out>/summary_<track>.txt`; a failed step is recorded and the track
   goes on (no `set -e`); the script exits non-zero if any step failed and prints the summary.
5. When `GATE_NICE=1` (default when another session shares the host) every step runs under
   `nice -n 10 ionice -c3`, and a new step waits while the 1-minute load is above
   `GATE_MAX_LOAD` (default 35), recording the hold in the summary.
6. The output directory is an argument or `GATE_OUT` (default `artifacts_local/gate/<utc>/`, which
   is gitignored); no machine path in the tracked script.

Red test: `tests/dev/test_gate_script.py` runs the script with fake step commands (a `PATH` shim)
and asserts the installs ran before every other step, the two tracks ran in parallel, a failing step
is recorded with its code and the next step still runs, and the summary lists every step. Control:
`shellcheck` clean; `make help` lists `gate`. The private reference implementation used for the B0
gate is the model; nothing from it is copied that names a machine path.

### W2.1 Foundation (B2 stage 1)

**Goal.** The stack starts with PostgreSQL at the schema head, Valkey, idle Celery workers and
beat; the API and every worker wait for PostgreSQL, Valkey and OpenSearch (and Triton where they
call it) and refuse to start, naming the missing one, or on a schema at the wrong head (D-69);
`/ready` names any dependency that is down; tests get a real database, and the default
offline command deselects the database tests (R-4). No job family moves yet.

**Operations additions (owner D-83).** 1B-9 request and job ids in every log line (`asgi-correlation-id` plus structlog contextvars); 1D-7 low-disk reporting in `/ready` and a 507 refusal at critical. Their metrics, dashboard and alerts land in W2.4 (6A-4, 6B-6) and the load check in the PR3 sweep (step 24).

**Preconditions.** W2.0 merged; `int/b2-control-plane` cut from `main`. The CI shards (PR #477, merged 2026-11-04) and this plan set are already on `main`.

| Order | Lane (plain name) | Branch | Owns | Brief | Issues |
|---|---|---|---|---|---|
| 0 | Control-plane package pins, the `db` marker and its deselection in `addopts` (coordinator commit) | on `int/b2-control-plane` | `requirements.txt`, `requirements-test.txt`, `pyproject.toml` (the new packages at the latest releases at bump time: Celery 5.7.0, kombu 5.7.0, redis-py 8.1.0, D-70 which reversed D-66 on 2026-11-05; the `db` marker, the `addopts` expression), `tests/test_control_plane_pins.py` | [03a](03a_stage1_foundation.md) lane 1-0 | #451 |
| 1 | The schema and the database layer | `b2/db-core` | [03a](03a_stage1_foundation.md) lane 1A (minus the `pyproject.toml` marker, now in lane 0) | 03a lane 1A | N19 (#479) |
| 1 | The Celery app, queues, the producer | `b2/queue-core` | 03a lane 1B | 03a lane 1B | #451, #464 (OPS-2 box) |
| 1 | Services, secrets, installer, CLI | `b2/compose-installer` | 03a lane 1C | 03a lane 1C | N19 (#479) |
| 1 | The storage seam | `b2/storage-seam` | 03a lane 1E | 03a lane 1E | N19 (#479) |
| 2 | Health, readiness, the CI database job, the offline commands, the worker-start schema check, the connection budget | `b2/health-ci` (after 1A and 1B merged) | 03a lane 1D (adds `Makefile`, the `ci.yml` and `CONTRIBUTING.md` marker commands, `src/tasks/app.py`'s worker-start call, `tests/config/test_db_connection_budget.py`) | 03a lane 1D | N19 (#479) |
| 2 | The research-seam tables: serving reports, clustering partitions, the embedding registry, the assist-loop record (approved, RD-1) | `b2/db-research-seams` (after 1A merged) | 03a lane 1F (revision `0002_research_seams`, its model files, `src/domain/clustering_vocab.py`, the contract models in `src/schemas/models.py`) | 03a lane 1F; [02d](02d_research_seam_schema.md) | #398, #346, #365, #466 (seams only) |

Merge order: lane 0, 1A, 1E, 1B, 1C, 1F, then 1D. The pieces that need 1A's code while 1A is still
coding moved into 1D (R-18) and 1F. Research seams inside existing lanes: 1A-13 (02d section 1),
1B-8 (`cluster` queue, the job-kind fields, the GPU inventory), 1C-8 (`cluster-worker`), 1E-4 and 1E-5 (crop size variants, the unknown-area delete).

**Folded in (decided).** `postgres:18.6-trixie` and the Valkey image by the digests of 01 section 4
(D-44, owner decision 6); Celery, kombu and redis-py at the latest releases at bump time (D-70, replacing the hold of D-66); every Python service waits on
`db-migrate` and on healthy `postgres`, `valkey` and `opensearch` (D-69); Valkey `maxmemory` and memory limits on the new services (R-19);
the `arbiter` queue declared (R-20); OPS-2 is measured first (1B-0, #464).

**Red-then-green tests.** Each lane's table in 03a, including the review cases: a failed commit is a
5xx (R-1), an unscoped statement raises (R-13), the default selection holds no `db` test (R-4), a
runtime module importing `src.services` fails the layering check (R-2), an unreachable database
or broker refuses the start naming it, and `/ready` answers 503 naming `database` (D-69).

**Part check.** 03a "Stage 1 exit criteria": the lanes' tests on the merged head, `-m db`,
`make contracts-check`, the 15-minute CPU bring-up (`-p op-b2-s1check`); plus the Docker-free
suite in 4 shards (MQ-3, approved).

**GPU window.** None. Spikes S0 and S1 are CPU-only with their own `-p` names.

**Codes in parallel during the part check.** The W2.2 lanes, cut from the merged W2.1 head.

### W2.2 Control data into PostgreSQL, and the job machinery (B2 stages 2 and 3)

**Goal.** Every project, class, config, setting and lasting record is read and written in
PostgreSQL; OpenSearch keeps two indexes per project; the one-step 0.6.0 upgrade exists and is
tested; the job start, claim, heartbeat, cancel, reaper and the event feed exist (no family moves
yet).

| Order | Lane | Branch | Owns | Brief | Issues |
|---|---|---|---|---|---|
| 1 | Projects, slugs, index ownership, pause | `b2/projects-pg` | [03b](03b_stage2_control_data.md) lane 2A | 03b 2A | N19 (#479) |
| 1 | The class registry | `b2/classes-pg` | 03b lane 2B | 03b 2B | N19 (#479) |
| 1 | Config store, VLM endpoints, settings, keymap, view state | `b2/config-pg` | 03b lane 2C | 03b 2C | N19 (#479) |
| 1 | Holdout freezes, label-operation records, project markers | `b2/records-pg` | 03b lane 2D | 03b 2D | N19 (#479), #473 (part c) |
| 1 | The one-step 0.6.0 upgrade and its rollback | `b2/upgrade-060` | 03b lane 2E | 03b 2E | N19 (#479) |
| 1 | The job runtime and the reaper | `b2/job-runtime` | [03c](03c_stage3_job_runtime.md) lane 3A | 03c 3A | #451 |
| 1 | Events in PostgreSQL and resumable SSE | `b2/events-pg` | 03c lane 3B | 03c 3B | #451 |
| 1 | The embedding catalog and spaces (approved, RD-1) | `b2/embedding-registry` | 03b lane ES-A | 03b ES-A | #365 (seams) |
| 2 | Remove the OpenSearch control indexes | `b2/opensearch-control-removal` (after 2A-2E and ES-A merged) | 03b lane 2F | 03b 2F | N19 (#479) |

Cuts: 2A-2E, ES-A and 3B from the merged W2.1 head; **3A after 2A merged** (it reads projects through
2A's repository, R-18); 2F after 2A-2E and ES-A merged. Merge order: 2A, 2B, 2C, 2D, 2E, ES-A, 2F, then 3B, then 3A
(3A codes against the two 3B signatures fixed in 03c 3B-1 and merges the integration head before its
final test run).
Ownership notes from the review: 2B also owns the three registry scripts no lane owned (R-18); 2E
owns `src/db/schema_check.py` for the upgrade gate (R-9); 2F owns the OpenSearch bootstrap
(`src/core/dependencies.py`, new `scripts/opensearch/migrate.py`, its call in `scripts/db/migrate.py`)
and, once 2E merged, the switch of 2E's load to the repositories (R-7, R-18); 3A owns the heartbeat
in `src/tasks/runtime/` and the stream-worker claims (R-2, R-8); 3B owns the Valkey wake-up module
`src/tasks/runtime/event_wakeup.py` (D-65). No two of these lanes own one file.

**Folded in.** #473 part c **moved to W2.3b lane ND-B** (research finding C8: today's crop groups never
contain a validated or held-out item, so 2D-4 could move nothing; the owner decision MQ-5 is kept);
the research seams 2A-7 (three default spaces), 2C-7 (the serving, cluster and loop-preset kinds),
2D-5 (the loop's label-operation kinds), 2E-8 (the upgrade step registry), 2F-6..2F-8 (the plate
normalizer at index creation and on upgraded indexes, the kNN door test, the mapping split), 3A-12..
3A-15 (gaps G-1 and G-2, the reaper's system resume, the GPU affinity and performance summary); the review fixes R-3 (class names by `name_key`),
R-5 (savepoints in the start and the reaper), R-7 (OpenSearch bootstrap once, in `db-migrate`),
R-8 (claims rebalance), R-9 (the upgrade gate, `--reset`, rollback order), R-13 (RLS roles and
policies), R-25 (`COPY` load), R-34 (`reset` event), and D-65 (the Valkey wake-up).

**Part check.** 03b and 03c exit criteria: `-m db`, `make contracts-check`, the 20-minute CPU
bring-up (`-p op-b2-s3check`); the Docker-free suite in 4 shards (MQ-3).

**PR1 (W2.1 + W2.2; owner decision 2026-11-05, D-76).** `main` stays bootable: the stack gains PostgreSQL,
Valkey, the Celery workers and beat, control data in PostgreSQL, the job runtime and the event feed, and the
0.6.0 upgrade; **job execution is unchanged** (every family still runs as today). Checks, in order: the part
check above on the merged W2.2 head; a short live smoke, `-p op-b2-smoke1`, steps 1, 8, 9 and 11 of
[03](03_b2_overview.md) section 9 on the no-op and sleeping no-op jobs (55 minutes, announced to the
peers); `.test_durations` refreshed; local CI parity (every job of `ci.yml`, the new `db` job included);
the owner's yes; push; one PR to `main`; merge commit; no tag. The image scans run here (the new base
images). The W2.3a lanes are cut from the merged W2.2 head and may start before PR1 merges.

**Codes in parallel.** The W2.3a lanes, cut from the merged W2.2 head.

### W2.3a Every long job onto the queue, as ports (B2 stage 4 part 1 and stage 5)

**Goal.** Every job that ran inside an API worker, plus training, model export and the bake-off
evaluator, runs on Celery workers with today's behaviour (label operations and their undo too: owner
decision 2026-11-05, D-75, no job runs inline in the API process); the trigger-file workers become queue
consumers; the assist loop gets its own record, pause and budgets. No W2.3a lane calls a research
lane (review RR-13): the `near_dup` scorer, the IVF retrain check and the loop's cluster reader are
ported unchanged, in one module each, for W2.3b to replace.

| Order | Lane | Branch | Owns | Brief | Issues |
|---|---|---|---|---|---|
| 1 | Reprocess, assist runs, scores (the `near_dup` scorer ported unchanged), probe, selection, viz, label operations and their undo as queued jobs (D-75) | `b2/families-a` | [03d](03d_stage4_job_families.md) lane 4A | 03d 4A | #451 |
| 1 | Dataset upload and import, combine, class merge and the name projection | `b2/families-b` | 03d lane 4B (region clustering moved to 4F) | 03d 4B | #451 |
| 1 | Bulk ingest, source polls, open-vocabulary drain, project delete (batched, RR-7), directory ingest, person grouping as a plain job | `b2/families-c` | 03d lane 4C | 03d 4C | #451 |
| 1 | Auto-label and the assist loop as queue consumers; the loop's own record, pause, budgets, slices and usage (gaps G-2, G-3) | `b2/trigger-workers` | 03d lane 4D | 03d 4D | #451, #466 (pause, events, budgets) |
| 1 | Training as a GPU queue consumer | `b2/gpu-train` | [03e](03e_stage5_gpu_queues.md) lane 5A | 03e 5A | #451 |
| 1 | Model upload and export as GPU queue jobs | `b2/gpu-export` | 03e lane 5B | 03e 5B | #451 |
| 1 | The bake-off evaluator as a GPU queue consumer | `b2/gpu-eval` | 03e lane 5C | 03e 5C | #451 |
| 2 | Remove the shared per-project job-directory plumbing once every caller is gone (the expand/contract "contract" step) | `b2/job-dirs-cleanup` (after every other W2.3a lane merged) | 03e lane 5E | 03e 5E | #451 |
| 1 | Promotion and the model registry | `b2/promote-models` | 03e lane 5D | 03e 5D | #451 |
| 2 | Periodic work to beat, part 1; the old job infrastructure deleted | `b2/beat-periodic` (cut after the eight lanes merged) | [03f](03f_stage6_scheduler_scaleout.md) lane 6A part 1 | 03f 6A | #451, #453, #464 (OPS-2 act part) |
| 2 | Compose for the ported consumers, no `container_name`, scale, the arbiter worker (part 1) | `b2/scale-out-compose` (cut after the eight lanes merged) | 03f lane 6B part 1 | 03f 6B | #451 |

Merge order: 5D, 4D, 4A, 4B, 4C, 5A, 5B, 5C, 5E (03d header gives the reasons), then 6A part 1, then 6B part 1. Size: 9 lanes, one
cross-lane signature (4D's VLM `usage_source` keyword), then 2.
Why stage 6 part 1 is here: `main` must boot after PR2, so the ported consumers need their compose services, and the
PR2 smoke runs on that compose; once every family is ported the old job paths, leases and file locks have no caller
and are deleted in the same PR (rule: no dead code). The research ticks and the monitoring work wait for W2.3b
([03f](03f_stage6_scheduler_scaleout.md) "Delivery in two parts").

**Folded in.** The research seams 5A-6 (reservations by card UUID, the reservation upserts its card,
RR-9), 5B-4 (the export writes the engine contracts and refuses a mismatch), 5D-5 (registry rows
with contracts); the batched project delete (RR-7); the OpenSearch factory injected into job
handlers (R-2); the fake-based family tests rewritten (R-14); `psycopg` C builds in the three GPU
images (R-27).

**Part check.** 03d section 7 and 03e exit criteria: `-m db` and `make contracts-check` after each
merge; the Docker-free suite in 4 shards (MQ-3).

**PR2 (W2.3a plus stage 6 part 1; D-76).** `main` stays bootable with every family ported. Checks, in order:
the part check above; the done-when `rg` lines of [03](03_b2_overview.md) section 1 that hold at PR2 (the old job
paths and mounts, `FileJob` and the leases, `file_locks.txt` empty, no `fcntl`); a live smoke, `-p op-b2-smoke2`,
steps 3-7 and 12-18 of 03 section 9 on the kinds and queues that exist at PR2 (3 h 15 min: 80 + 115 minutes; fake
GPU handlers, no GPU); `.test_durations` refreshed; the image scans if a GPU image or requirements file changed
(5A and 5B add `psycopg` C builds); local CI parity; the owner's yes; push; one PR to `main`; merge commit.

**GPU window.** None in the lanes; stage 5's real-GPU smoke runs in GPU window 1 (W3).

**Codes in parallel.** The W2.3b lanes 4F, ND-A and ES-B, cut from the W2.3a head after its part check (they may start before PR2 merges, on the coordinator's say-so).

### W2.3b The research families on the queue (B2 stage 4 part 2)

**Goal.** Clustering runs on versioned partitions, near-duplicates are run-scoped groups with an
exact key and a backfill, the holdout pulls in near-duplicates, the embedding re-index job exists,
and Cropwright follows every new state. Approved by the owner (RD-1, Q-A: every seam now).

| Order | Lane | Branch | Owns | Brief | Issues |
|---|---|---|---|---|---|
| 1 | Clustering on versioned partitions: `cluster_run`, activation, projection, the `cluster_id` split; person grouping and the auto-label consumers moved onto partitions | `b2/clustering` | 03d lane 4F | 03d 4F | #346 (S-core) |
| 1 | Near-duplicates: one service, two levels, run-scoped groups, the exact key, the `source_backfill` job, the `near_dup` scorer deleted | `b2/near-duplicates` | 03d lane ND-A (revision `0003_near_duplicates`) | 03d ND-A | #416, #473 (part a) |
| 1 | The embedding re-index job (time-sliced, RR-20) and the space routes | `b2/embedding-reindex` | 03d lane ES-B | 03d ES-B | #365 (seams) |
| 2 | The near-duplicate holdout guard (replaces 2D-4) | `b2/near-dup-holdout` (after 4F and ND-A merged) | 03d lane ND-B | 03d ND-B | #473 (part c) |
| 3 | Cropwright follows the queued states and every new control | `b2/frontend-queued` (after every other W2.3a and W2.3b lane merged and `make contracts-check` is green) | 03d lane 4E | 03d 4E | #451 |

Merge order: 4F, ND-A, ES-B, ND-B, 4E. Size: 3 lanes at once, then 2 in sequence; every cross-lane
call (`record_partition`, `group_lineage`, `assign.place_new`) is against a signature fixed in 03d.

**Folded in.** #473 part a (stale crop near-duplicate fields) is ND-A's run-scoped projection
(4A-7 superseded); the threshold fail-closed check is retired with `OP_CROP_DUP_THRESHOLD` (ND-9
deletes the scorer); the exact-key backfill for upgraded images and the stored-bytes check (RR-10);
the 4C-7 person grouping recorded as a partition (4F-9).

**Part check.** As W2.3a, after the W2.3b merges (no PR here: PR3 follows W2.4).

**GPU window.** None.

**Codes in parallel.** The W2.4 lanes 6A and 6B, cut from the merged W2.3b head. The B3 `frame`
track (no B2 file) may start from `main` on the coordinator's say-so.

### W2.4 The rest of stage 6, hardening, docs, and the B2 sweep (B2 stage 6 part 2, PR3)

**Goal.** The research ticks (cluster refresh, near-duplicate re-score, full build and prune) run once from
the one beat and the cluster-refresh service goes; monitoring for replicas, the PgBouncer overlay and the
per-card Triton shape; the new services are hardened; the permanent docs describe the control plane. PR2
already moved the other periodic work to beat and deleted `FileJob`, the job files, the leases and every
file lock, and gave the consumers their compose services. Then the one B2 sweep.

| Order | Lane | Branch | Owns | Brief | Issues |
|---|---|---|---|---|---|
| 1 | Periodic work to beat, part 2: the cluster-refresh and near-duplicate ticks; the cluster-refresh daemon deleted | `b2/beat-research-ticks` | [03f](03f_stage6_scheduler_scaleout.md) lane 6A part 2 | 03f 6A | #451 |
| 1 | Scale-out part 2: monitoring (with `monitoring/alloy-config.alloy` and the dashboards, R-12), the PgBouncer overlay, the per-card Triton shape, the DCGM fields, the `curation-cluster-refresh` service deleted | `b2/scale-out-monitoring` | 03f lane 6B part 2 (6B-5..6B-8) | 03f 6B | #451 |
| 1 | Readers resolve their vector field through the embedding registry (approved, RD-1) | `b2/embedding-readers` | 03f lane ES-C (about 40 files; the coordinator checks them against 6A and 6B at the cut) | 03f ES-C | #365 (seams) |
| 2 | Cropwright reads `embedding_states` | `b2/frontend-spaces` (after ES-C merged) | 03f lane `b2/frontend-spaces` | 03f | - |
| 2 | Hardening (after 6B part 2 merged) | `b2/hardening` | 03f lane 6C | 03f 6C | #461, #462 |
| 3 | Docs (last) | `b2/docs` | 03f lane 6D | 03f 6D | N19 (#479) |

**Sweep (PR3, the one full B2 sweep).** Exactly [03](03_b2_overview.md) section 9 (its T1 offline set, the
**live harness run 2 of 3** with the steps and budget stated there, now including the research steps
19-22 and step 23, the five stale Cropwright screenshots recaptured on the docs overlay and viewed
at desktop and 430 px before commit, the image audit, Trivy and `pip check`, the browser check);
this file does not restate the step count or the budget (R-23). The PR1 and PR2 smokes are not counted
as live runs and every step runs again here. Then local CI parity, the owner's yes, PR3 to `main`, merge commit.

**Exit gate.** The done-when of [03](03_b2_overview.md) section 1 by `rg`; PQ-1..PQ-9 of
[decision_register.md](reference/decision_register.md) section 4; the closing chores of section 7 step 10,
done by the coordinator in PR3.

**Codes in parallel during the sweep.** W3 lanes (pins part B and `b3/frame`, then the rest in
order) cut from the frozen `int/b2-control-plane` head; B4's 4A-0 evidence script (offline).

### W3 One Ultralytics version and the bumps that can change model output (B3, pins part B, GPU window 1)

**Goal.** One exact Ultralytics release and one fused `[batch, 300, 6]` detection contract in every
image; and, judged by the same GPU gate, every dependency bump that can change model output
(torch and torchvision, OpenCV 5, the onnx, onnxruntime, modelopt and TensorRT set, numpy and
pandas with the numeric stack, CuPy 14, the Hugging Face stack, Pillow and simplejpeg), then the
MLflow and Dependabot tail of the pin series.

| Order | Lane | Branch | Owns | Brief | Issues |
|---|---|---|---|---|---|
| 1 | Pins part B: the model-output set and the image pins (D-67) | `deps/pins-part-b` | the three manifests, `requirements-export.txt` and the `export` extra (versions only), `docker/trainer/requirements.txt`, `scripts/curation/bakeoff/requirements.txt`, `Dockerfile.triton`, `docker/hardened/triton/Dockerfile`, `docker/segmenter/Dockerfile`, new `docker/triton/requirements.txt` and `docker/segmenter/requirements.txt`, the root `Dockerfile` (the evaluator's inline `mlflow`), `docker-compose.yml` (`MLFLOW_IMAGE` only), `CONTRIBUTING.md` and `.github/workflows/ci.yml` (the CPU torch-pair commands only), `.github/dependabot.yml`, `tests/test_requirements_contract.py` | [01](01_package_pins.md) section 7, part B; [04](04_b3_ultralytics_one_version.md) track `deps` | N20 (#480), #458 |
| 1 | Each adapter declares its box frame (GPU-14 final; the only B3 track that touches no B2 file, R-18) | `b3/frame` | 04 track `frame` | 04 B3-G1 | #458 |
| 2 | Stage A: the pin, the second environment deleted, the trainer on the pin | `b3/stagea` (from the pins part B merge) | 04 track `stagea` | 04 B3-A1..A4 (A5 only if D-38 yes) | #458 |
| 3 | The family registry and the full CPU matrix | `b3/families` (from `b3/stagea`) | 04 track `families` | 04 B3-F1..F3 | #458, #368 (first slice) |
| - | Stage A point recorded | coordinator | | 04 section 4 | |
| 4 | One export module (Stage B) | `b3/export` | 04 track `export` | 04 B3-E1..E7 | #458 |
| 4 | Fused-only decoders (Stage B) | `b3/decoders` | 04 track `decoders` | 04 B3-D1..D3 | #458 |
| 5 | Docs for the stage that ships (after the gate verdict) | `b3/docs` | 04 track `docs` | 04 B3-W1 | #458 |

Merge order: pins part B, `frame`, `stagea`, `families`, Stage A point, `export`, `decoders`, the
GPU gate, `docs`. `requirements-export-y11.txt` is not bumped by pins part B: B3-A2 deletes it.

**Folded in.** The pin-series bumps that can change model output (owner direction 3, D-67); the
real-GPU smoke of B2 stage 5 (03e "GPU check", gate step G12); the GPU part of the image smokes of
01 section 6 (G11), with `cv2.dnn.blobFromImage` in the OpenCV 5 parity set (R-17). Every B3 track
re-verifies its files and symbols on the post-B2 head first (R-18). The GPU-15 leftover (refuse an
undeclared region head when a profile is saved) is part of B3-D3 (MQ-8, approved 2026-11-04).

**Sweep.** [04](04_b3_ultralytics_one_version.md) section 7: T1 offline, the CI `families` matrix
at the pin, the CPU part of G11 (image builds, `pip check`, imports, weight downloads: R-17), image
audit and Trivy (the changed base images make the image scans run once here).

**GPU window 1** (GPU 0, GPU 2 only if agreed; `-p op-b3-gpu`; announced to peers; about 1.5 GPU
days): 04 section 6 G0-G11 (G10 is the B0 OCR before/after comparison; G11 the 01 section 6 image
smokes), plus the B2 stage-5 real-GPU smoke (one short training run, one export, one 20-image
bake-off). Engine EA (Stage A point) now carries the pin-series bumps and the Ultralytics pin
together, so G3 "EA vs E0" judges every version change at once and G3 "EB vs EA" the contract
change. The window has a written cap (its estimate plus one day, R-17): on an EA failure the first
action is the named fallback pins of 01 section 8 for the failing step, then a re-run of that step
only; a bisect over the part B commits comes second and only inside the cap; past the cap, report to
the owner ([04](04_b3_ultralytics_one_version.md) section 6, "Contingency").

**Codes in parallel during the sweep and the gate.** W4 lanes cut from the frozen
`int/b3-ultralytics` head (B4 touches no B3 file); 4D's stage timer; the privacy lane's file list
checked against 4A, 4B and 4D.

### W4 Stateless API (B4)

**Goal.** No request handler fans out; the worker count is one knob (`API_WORKERS`); pools are
sized per process from totals; project creation leaves the request; the same answers with 1 and N
workers and replicas.

| Order | Lane | Branch | Owns | Brief | Issues |
|---|---|---|---|---|---|
| 1 | Request fan-out becomes bulk writes; the ingest data plane as task groups | `b4/bulk-occ` | [05](05_b4_stateless_api.md) track 4A | 05 section 4 | #451, #464 (CC-10 box), #473 (part b) |
| 2 | Project creation (#439) | `b4/project-create` | 05 track 4D | 05 section 6 | #439 |
| 3 | The worker knob and the statelessness ratchet | `b4/worker-knob` | 05 track 4B | 05 section 5 | #451 |
| 2 | Capture metadata and GPS coordinates at ingest under the project's `location_policy` (default `store`; RD-51 reversed), the location purge; exports keep the embedded GPS (RD-54) unless the project's policy is `drop` (CM-5) (research spatial L-C1; RD-5 approved) | `b4/capture-metadata` (after 4A, 4D and the privacy lane merged) | [05](05_b4_stateless_api.md) section 6c | 05 section 6c (CM-1..CM-4) | #348 (the one v0.7.0 slice) |
| 1 | Privacy and security items (owner decision 7 of 2026-11-04) | `b4/privacy-hardening` | the files named in the maintainers' private notes; the coordinator checks they are disjoint from 4A, 4B and 4D before the cut and moves any shared file's task into the owning track | the private notes; public text stays neutral (one line per issue); plus one docs task, `SECURITY.md`: a plain statement that, when external use is enabled, external VLM endpoints and the hosted research judge may receive any region kind and contact-sheet image (OI-1, OI-2; the matching `docs/CURATION.md` sentence is in track 4D, which owns that file in W4) | #468, #470, #472 (#469 and #471 close with no code change, traceability section 4) |

Merge order: 4A, 4D, privacy lane, capture lane, 4B (its ratchet allowlist must
see the others' final code). The privacy lane owns `SECURITY.md` in W4 and writes both
statements (OI-1 and OI-2 on external endpoints; the location rules: coordinates stored per `location_policy`, exports keep the embedded GPS unless the project's policy is `drop`). Registration-plate text needs no privacy statement or machinery: it is ordinary label data (D-79).

**Folded in.** #473 part b, rewritten as ND-C1..ND-C4 (core ingest writes pending rows and the
single-writer `near_dup_core_group` job groups them; task 4A-4b in 05), in the file 4A-4 already
rewrites; the research seams 4A-5's space-aware `_msearch`, 4B-6 (the Triton router over the serving
profile) and 4B-7 (the in-process text encoders deleted: dead code under D-69); `project_create` as a `jobs` kind (R-21); the API pool as a real
deployment total over workers and replicas (R-6): 4 connections per API worker, a total of 128 (`OP_DB_API_CONNECTIONS_TOTAL`) and Postgres `max_connections` 400 (D-74).

**Sweep.** 05 section 7: T1 offline, frontend full set, the browser check of every streaming page,
**live harness run 3 of 3** (`-p op-b4-live`, budget 2 h): the 1-vs-N check in the three
configurations of 05 section 7, a sequential scenario with an exact diff and a concurrent one judged
by invariants (R-15), and the 4D stage timings.

**GPU window.** None: the 4B-5 benchmark is the first job of GPU window 2.

**Codes in parallel during the sweep.** W5a `ladders` (A1) and `recall` (R1) code; release-notes
drafting (RP-4); the first typed-response lane.

### Typed response models during W5a-W5c (#465, owner decision 7 of 2026-11-04)

One CPU lane per wave, `api/typed-responses-<n>`, each owning one group of routers that no lane of
that B5 sub-batch touches (the coordinator lists the sub-batch's owned files at the cut and picks a
disjoint group: curation review and classes routers first, then the dataset and export routers, then
the core routers outside `/detect`, `/embed`, `/analyze` and `/ingest`). Task per router: model
today's exact response shape, set `response_model=`, `make contracts` in the same commit, a
per-router test that its operations have non-empty schemas ([05](05_b4_stateless_api.md) section 13,
audit BA-6). It merges into that sub-batch's integration branch and its sweep. The routers B5d
rewrites (`/analyze`, `/ingest`) are typed inside 5d's own lanes; #465 closes with the 5d PR.

### W5a Public baseline (B5a; GPU window 2 opens)

**Goal and lanes.** [06a](06a_b5_baseline.md): tracks `ref`, `ladders`, `sweeps`, `recall`; first
the 4B-5 benchmark of [05](05_b4_stateless_api.md) (the 32-worker baseline retaken on the B3
engine, then the `API_WORKERS` default). Owner answers OQ-B5-1..3 before it; D-35 is decided on
the recall numbers at its end (#463). No optimization code merges before the 5a numbers exist.

**Sweep.** T1 offline; the 5a runs in window 2. **Codes in parallel:** the W5b lanes and the second
typed-response lane.

### W5b Decode once (B5b)

[06b](06b_b5_decode_once.md): tracks `decode` (GPU-3b first), `profile`, `prep` (#407 staged,
GPU-12), `routes` (#40 Wave 6), `codec`. Gate in window 2. The serving head-to-head (#231) may run
after the 5b gate in the same window, under its own rule ([06](06_b5_overview.md) section 10), or
is recorded and closed. **Codes in parallel:** the W5c lanes and the third typed-response lane.

### W5c Crops at model size, OCR on the GPU (B5c)

[06c](06c_b5_crops_ocr_faces.md): tracks `crops` (#219), `ocr` (#220, GPU-7), `ocrnorm` (GPU-19
evidence, #464 box). One GPU day in window 2 for P-vlm, P-sam, P-region, P-ocr. **Codes in
parallel:** the W5d lanes.

Research amendments for W5a-W5d (each 07 file's last section): the decisions fed by the GPU
prototypes (`gpu:PR-0..PR-6`, read before the lane codes), the export orientation fix L-X1 in 5b-D1,
the one geometry module and the metric rename in 5b, `rec_char_min` in 5c's track `ocr`, the
near-duplicate candidates in 5d.

### W5d Request-driven processing plans (B5d)

[06d](06d_b5_processing_plans.md): tracks `p0`, `analyze`, `curation`, `gpu` (faces on resident
frames), `docs`. Owner answers 07d Q1-Q9 at kick-off. **Codes in parallel:** the W5e lanes (cut from
the frozen W5d head).

### W5e Registration-plate text (GPU window 2 closes; RD-3 answered as recommended, D-78)

[06e](06e_w5e_plate_text.md): lanes `w5e/region-text-store` (revision `0004_region_text`), then
`w5e/region-text-fields` and `w5e/region-text-jobs` in parallel; the
reader lane and a frontend lane only on their conditions. Plate text is ordinary searchable label data stored as
real readable text (exact, fuzzy and fragment search; no privacy modes, no master secret, no separate purge:
project delete removes it like any data; D-79, D-80). Preconditions: W5c and W4 merged; the
plate prototypes `pt:P0..P5` read (short runs in PW-2; `pt:P6` cut). Plate boxes come from a detector the user brings (for example from Hugging Face)
through the existing model upload path, or from the segmenter with the text hint; the owner's private plate detector
stays in their own deployments and no private model ever ships in the repo or the images (RD-67 answered yes, D-80). One full sweep and a short GPU gate (about 2 h) at
the end of window 2. **Codes in parallel:** release-prep drafting (moved here from W5d).

### W6 Release prep, R-final, the tag (GPU window 3)

[07](07_release_acceptance.md): RP-1..RP-9 (VERSION and CHANGELOG first), then the one R-final run
(every RF row including the research rows RF-80..RF-86 of 06 section 4.7, the segmenter tier, the 0.6.0 upgrade with an in-flight job), the fix loop only for
a failed row, the tag and release only on the owner's word (D-16). Nothing else merges to `main`.

### Side track: the 12 GB card (#42)

[06](06_b5_overview.md) section 11. Runs on GPU 1 only after its owner session confirms the card is
free; never blocks a wave.

### 4.10 The research prototype track (side; approved, RD-2 and RD-7)

[08](08_research_prototype_track.md) holds every prototype with its data, hours, dependency, window,
decision and landing wave. In short, per the owner (2026-11-04): working, known code first;
prototypes are simple local-venv GPU scripts (no Docker builds) on set P and freely downloadable
public sets (the dataset register, 08 section 4.1), tracked under `scripts/research/<topic>/` with
exact pins and a committed lock file (RD-7); every run is a short tiered run on a small seeded sample
with early stop (08 section 2.1); the default is today's approach, replaced only on a clear
short-run win. PW-1 (W2.1-W2.2, about 3.4 GPU h), PW-2 (W2.3a-W2.4, about 1.9) and PW-3 (W4, about
1.5) plus a 2.2 h reserve make **about 10 GPU hours before the tag**; anything beyond needs the
owner's OK. Cut before the tag: `gpu:PR-3`, `PR-4` (5b-G4 not opened), `PR-7..PR-9`, `pt:P6`, the SSCD
arm of `nd:P2`, `nd:P3..P7`, every `emb:` and `clu:` GPU run, `sp:P-2..P-9`, `al:AL1..AL6`. The
post-tag windows RG-1..RG-6 are not scheduled; each needs the owner's OK. A prototype lane owns only its topic's `scripts/research/<topic>/`,
`tests/research_scripts/` or `tests/research/` files, its registration and result notes under
`docs/research/` and `docs/benchmarks/<exp>/`; it never edits a product file, so it never conflicts
with a program lane. Registration (the bars) is committed before any score.

### 4.11 The research waves after the tag (RW-1..RW-6)

| Wave | Prototypes | Landing (only on a PASS) | Issues |
|---|---|---|---|
| RW-1 GPU serving breadth | `gpu:PR-3`, `PR-4`, `PR-7`, `PR-8`, `PR-9` | two cards, interactive priorities, INT8 opt-in, CUDA graphs on small engines, MPS, a C++ backend (5b-G4) and the serving head-to-head on their rules | #403, #231, #268 remainder |
| RW-2 Embeddings and search | `emb:EP-0..EP-7` | the v0.8.0 swap reduced to register, build, evaluate, flip; `group_by` with `space`; optional spaces; OpenSearch 3.9 | #347, #365, #435, #436 |
| RW-3 Clustering and the visual judge | `clu:CL-0..CL-9` | a method value, suggestions and constraints, faces on partitions, verification with the review screens | #346, #406 |
| RW-4 Near-duplicate evidence | `nd:P3..P7`, the SSCD arm of `nd:P2` | seeded defaults from evidence, a PDQ tier | #416 |
| RW-5 Spatial and attributes | `sp:P-2..P-9` | attribute signals, the timeline (v0.8.0), scenes after a GO | #348, #400 |
| RW-6 The auto-label assistant, measured | `al:AL1..AL6` (`al:AL0` ran short in PW-3) | the review queue and metrics, strategies, Z5 roots, retraining | #466, #398, #84 |

Lanes, gates and windows: [08](08_research_prototype_track.md) section 5; their v0.8.0 landings:
[10_v080_platform_breadth.md](10_v080_platform_breadth.md) section 2.11.

## 5. GPU-window budget

Exactly three program windows, each announced to the peer sessions, on GPU 0 (GPU 2 only when
agreed); GPU 1 only for #42 with its owner's consent ([risks_cleanup_host](reference/risks_cleanup_host.md) section 1).
The research prototype windows are separate, run only on idle GPU time and never overlap a program
window or a live-harness run (rows PW-1..PW-3; RD-2: short runs, about 10 GPU hours in total).

| Window | Wave | Contents, in order | Estimate |
|---|---|---|---|
| 1 | W3 | 04 G0 (reference images, an 8.4.48-trainer checkpoint), G1-G9 (engines, Triton contract, COCO parity, throughput, families, promote, embeddings with the OpenCV 5 face path, YOLOv5 secondary, image audit), G10 (B0 OCR before/after), G11 (the GPU part of the pins part B smokes: the CUDA provider, both CuPy wheels, one ModelOpt export, the GPU decode tests, segmenter start-up), G12 (the B2 stage-5 smoke: one short training run, one export, one 20-image bake-off; MLflow client and server together). The CPU parts of G11 run in the W3 sweep (R-17) | about 1.5 GPU days, cap plus one day (R-17) |
| PW-1 | W2.1-W2.2 (research) | `gpu:PR-0`, `PR-1`, `PR-2`; `nd:P0`, `P1`, `P2` | about 3.4 GPU hours |
| PW-2 | W2.3a-W2.4 (research), outside live run 2 | `pt:P1` (PP-OCRv6, D-82), `P2`, `P3`, `P4`, `P5` | about 1.9 GPU hours |
| PW-3 | W4 (research), outside live run 3 and before the 4B-5 benchmark | `gpu:PR-5`, `PR-6`; `al:AL0` (shortened) | about 1.5 GPU hours |
| reserve | any PW window | tier-2 confirmations only | about 2.2 GPU hours |
| 2 | W5a-W5e | the 4B-5 benchmark; 5a `ref`, then `ladders`, `sweeps` (tiered), `recall`; 5b `profile`, then the 5b gate; the #231 head-to-head (optional, after 5b); the 5c gate (one day); 5d `p0` and the 5d gate; the W5e gate (about 2 h) | several GPU days spread over the B5 waves, one run at a time |
| 3 | W6 | the R-final run (segmenter tier, OCR set, B5 confirmation rows) | 1-2 days |
| side | any | #42 on GPU 1 | about half a day |
| RG-1..RG-6 | after the tag (research waves) | [08](08_research_prototype_track.md) section 2.2; not scheduled, each on the owner's OK | the plans' upper estimates sum to about 70 GPU hours; tier 1 at most 2 h per prototype |

Research GPU total before the tag: **about 10 hours**, the owner's cap (was 31 before and 62 after).

A failed window step is re-run only after its fix is merged to the integration branch, never per
attempt. A lane that needs a GPU number waits for its window.

## 5a. Rough size per wave (owner D-83; a guide, not a promise)

Estimates for planning around the shared host and peer sessions. "Coding" is wall-clock with the
wave's lanes running in parallel (one Sonnet agent per lane; a lane is typically 1-4 agent hours);
"checks" is the part check, smoke or sweep after the merges. Re-estimate after W2.1 from its measured
lane times and record the actuals in the ledger.

| Wave | Lanes | Coding (wall-clock) | Checks | Ends with |
|---|---|---|---|---|
| W2.0 pins part A + `make gate` | 2 | half a day | about 1 h (CPU sweep) | its own PR |
| W2.1 foundation | 7 | 1-2 days (1D after 1A and 1B) | about 1 h part check | - |
| W2.2 control data + job runtime | 9 | 2-3 days (3A after 2A; 2F last) | about 1 h part check + 55 min smoke | **B2 PR1** |
| W2.3a every job onto the queue | 9 (+ stage-6 part 1) | 2-3 days | about 1 h part check + 3 h 15 min smoke | **B2 PR2** |
| W2.3b research families | 5 | 1-2 days | about 1 h part check | - |
| W2.4 stage-6 part 2, docs, sweep | 6 | 1-2 days | about 40 min T1 + 6 h 10 min live run | **B2 PR3** |
| W3 Ultralytics + pins part B | 7 | 2 days | sweep + GPU window 1 (about 1.5 GPU days) | B3 PR |
| W4 stateless API | 5 | 1-2 days | sweep + live run 3 (2 h) | B4 PR |
| W5a-W5e speed and plate text | 3-6 each | 1-2 days each | sweep + GPU window 2 gates | one PR each |
| W6 release | prep lanes | 1 day | R-final, GPU window 3 (1-2 days) | the tag |

Total, roughly 4-6 weeks of calendar time at one wave in flight with the next coding in parallel,
dominated by the B2 waves and the GPU windows.

## 6. Test budget per wave

| Tier | Where it runs |
|---|---|
| T0 red-then-green + paired control, static checks on touched files | every lane, every task |
| Part check (D-63) | W2.1, W2.2, W2.3a, W2.3b: `-m db`, contracts, the merged lanes' tests, a short CPU bring-up |
| Short live smoke | PR1 (W2.2 close: steps 1, 8, 9, 11, 55 min) and PR2 (W2.3a close: steps 3-7 and 12-18, 3 h 15 min); subsets of the run-2 steps, not numbered runs |
| T1 full offline sweep | once per wave on the merged head: W2.0 (CPU only), W2.4 (PR3), W3, W4, W5a-W5d, the release-prep gate (PR1 and PR2 run the part check and the CI jobs only) |
| Live harness | run 2 in W2.4 (PR3), 6 h 10 min = 370 minutes, the sum of steps 1-24 (step 24 the load check, D-83) ([03](03_b2_overview.md) section 9: steps 1-18 4 h 25 min, the research steps 19-23 1 h 25 min); run 3 in W4 (2 h); run 1 was B1's smoke (done) |
| GPU | section 5 |

Lanes never run a full suite, `--all-files`, the live harness or a GPU.

## 7. How a coding session runs one wave

1. **Preflight.** Read this file's wave section, then the batch file sections it links; read the
   private ledger's latest "RESUME HERE"; `git fetch origin`, `git worktree list`,
   `gh pr list --state open`; `uptime` and `nvidia-smi`; find the peer sessions and tell them what
   heavy work the wave will run. Check section 9 for the questions the wave needs: never start a
   wave without the owner's go and those answers.
2. **Integration branch.** Cut `int/<batch>` from `main` (W2.1, W3, W4, W5x) or reuse it (W2.2-W2.4; B2's one branch carries three PRs, each opened from its head when its last wave closes).
   A wave that starts during the previous wave's sweep cuts its lanes from the frozen previous
   integration head, which becomes an ancestor of `main`, so the lanes merge without a rebase.
3. **Briefs.** For each lane: the batch file's common brief block plus the lane's section, with the
   worktree, branch, compose project (`-p`, or none), GPU (none in every lane), and a deadline.
   Every brief forbids `pgrep -f`, `pkill -f`, sleep and until loops, `git stash`, `--no-verify`,
   pushing, merging and touching another checkout, and says to read each cited official doc before
   coding and cite it in the commit. Sonnet implements; reviews are Sonnet too.
4. **Worktrees.** `git worktree add <repo>-wt-<lane> -b <branch> int/<batch>` per lane; one agent per
   worktree; never two writers in one.
5. **Lane done.** The report lists commits, each red output and green output, files touched, and
   anything not done. The coordinator re-runs the lane's targeted tests in the lane worktree before
   merging (claims are checked, not trusted).
6. **Merges.** `git merge --no-ff` in the wave's order, one merge commit per lane; generated
   `contracts/` conflicts are resolved by `make contracts` on the merged tree inside the merge
   commit; append-only files keep both sides. A failing lane reverts alone (`git revert -m 1`).
7. **Checks.** Part checks or the wave sweep exactly as section 4 says; run the Python and frontend
   tracks in parallel, xdist capped by the host load; heavy and live runs one at a time,
   announced, with their own `-p`. The coordinator never extends a budget; a step that overruns
   stops and is reported.
8. **PR.** The coordinator's closing commit adds the PR's entries under `## [Unreleased]` in the root `CHANGELOG.md` (and `frontend/CHANGELOG.md` when `frontend/` changed), Keep a Changelog sections, because CI's "CHANGELOG entry for backend changes" job fails a code PR without one; this applies to W2.0, each of the three B2 PRs and every later wave PR (lane 6D still writes the B2 summary and breaking-change notes in PR3). Local CI parity first (every job of `ci.yml`, and `frontend.yml` when `frontend/`
   changed, the changelog rules included). Ask the owner before pushing. Push the integration
   branch, open one PR against `main` (`gh pr create`; B2 opens three, PR1 after W2.2, PR2 after W2.3a and PR3 after W2.4, each with its own checks from section 4, none tagged, each leaving `main` bootable), check `gh pr checks`, verify the push with
   `git ls-remote`. Merge with a merge commit on green CI and the owner's yes.
9. **Close-out.** Verify each issue the PR should close is closed; post the prepared comments of
   [traceability](reference/traceability.md) section 4 (approved 2026-11-04, D-68); remove merged worktrees
   and branches; update the ledger.
10. **Closing-PR chores (the coordinator's, no lane owns them).** In the last PR of a batch (PR3 for B2; the batch PR
    of W3, W4, W5x) the coordinator, not a lane, does: the plan-lifecycle deletions (delete the batch's plan files after
    moving their keep-lists, and their rows in `decision_register.md` and `docs/design/README.md`, per
    [decision_register.md](reference/decision_register.md) section 8.2); the `docs/design/README.md` rows for any plan
    file added or removed; and the `.test_durations` refresh (also in PR1 and PR2, which add many tests). These edits are
    made on the integration branch after the last lane merged and before the sweep's final run, so the sweep sees them.

## 8. CI and test speed

- **The 4-shard offline suite is on `main`** (PR #477, merged 2026-11-04): about 10 minutes
  instead of 35, which is what makes the per-part offline suite (MQ-3, approved) affordable. Its
  shard command gains `and not db` in lane 1D (R-4).
- **The `db` marker is deselected by default** from W2.1 on: `pyproject.toml` `addopts` (lane 1-0),
  both `ci.yml` commands and `CONTRIBUTING.md` (1D), `CLAUDE.md` (6D); a test asserts the default
  selection holds no `db` test (R-4).
- **The `db` job (1D-2).** Add it behind the existing `changes` path filter in `ci.yml` (code paths
  only), so docs-only PRs stay fast; it needs a Postgres service container, not Docker-in-Docker. It lands in PR1 (lane 1D), so
  PR2, PR3 and every later PR run it.
- **Refresh `.test_durations`** in every B2 PR (PR1, PR2, PR3: each adds many tests) and in the
  closing PR of W3 and W4: an outdated durations file unbalances the shards. The coordinator does it
  (section 7 step 10).
- **One image-scan run per wave.** The image scans run only when a base image or a requirements
  file changes; land each wave's dependency edits in that wave's single PR (W2.0, PR1, which adds the
  PostgreSQL and Valkey images, PR2, whose GPU images gain `psycopg`, and W3), so the scans run on
  those PRs only, not per bump; PR3 runs the image audit and Trivy once in its sweep.
- **Docs-only PRs** (this plan, plan-lifecycle deletions) carry `[skip-changelog]` and touch no
  code path, so they run only the docs jobs.
- **Locally**, the sweep runs as two parallel tracks (Python, frontend) after the installs, with
  xdist sized to the host load; the live harness reuses locally tagged images between W2.4 and W4
  (never `:latest`, which Watchtower re-pulls).

## 9. Owner questions

### 9.1 Answered on 2026-11-04 (recorded in the register, [decision_register.md](reference/decision_register.md) section 6)

Every open B2 recommendation as a block (D-19..D-26, D-46..D-63, with D-48 and D-50 as amended);
every service requires the control plane, no degraded mode (D-69, replacing D-64); Valkey
pub/sub wake-ups (D-65); kombu
and redis-py held (D-66; reversed on 2026-11-05, D-70, see below); pins part A as its own PR (D-67); the GitHub actions of the traceability file
(D-68); `postgres:18.6-trixie` (MQ-2); near-duplicate groups into the holdout together (MQ-5); the
security lane in W4 (MQ-6); typed-response lanes in W5a-W5c (MQ-7); research experiments after the
tag with their decided items kept in the rewrite (MQ-9); #42 to v0.8.0 if unrun at the tag (MQ-10).
Later the same day: every service requires PostgreSQL, Valkey and OpenSearch with no degraded mode
(D-69, replacing D-64); the Docker-free suite in 4 shards at every B2 part check (MQ-3); Pillow,
simplejpeg, the Hugging Face stack and MLflow in pins part B (MQ-4); the GPU-15 leftover in B3-D3
(MQ-8); Flower on loopback port 4613 and the gateway subpath, only with `monitoring` (Q-3,
ENV-18); #469 and #471 kept as they are, closed with no code change, with a SECURITY.md statement
in the privacy lane (OI-1, OI-2).

### 9.1a Answered on 2026-11-05 (recorded in the register as D-70..D-81)

All package pins move to the latest releases at bump time, Celery 5.7.0, kombu 5.7.0, redis-py 8.1.0 (D-70,
reversing the D-66 hold; [01](01_package_pins.md) owns the table); Celery `revoke` is never used, cancel is a
database write only (D-71); label operations and their undo run on Celery like every other long job, no inline job
path (D-75); B2 ships as three PRs to `main` (D-76); the open research questions are presented in small batches (D-77)
and 62 of the 69 were answered as recommended (D-78, section 9.3); registration-plate text is ordinary searchable label
data with no privacy machinery (D-79); RD-61, RD-62 and RD-67 answered (D-80); PRs #474 and #449 are closed.

### 9.2 Still open, with the plan's recommendation

Only the kick-off questions of B3, B4, B5 and the tag remain; each is asked at its wave's
kick-off, not before. The planner first answers whatever the official documentation and best practice
already settle; the questions that remain are presented to the owner in small batches, each with its
recommendation (D-77).

| ID | Question in plain words | Needed before | Recommendation |
|---|---|---|---|
| D-27, D-28 | Ultralytics staging and the pin rule | W3 | as in [04](04_b3_ultralytics_one_version.md) section 3 |
| D-38 / Q-4, PR #449 | Triton 26.09 or 26.10 and a newer Python base inside B3? | W3 | only if window 1 has room (B3-A5); otherwise after the tag. PR #449 is already closed, so there is no action now |
| D-29, D-29a, D-32, D-32a, D-33, ENV-17 | B4 fan-out, VLM batch routes, project creation, the worker default, pool names | W4 | as in [05](05_b4_stateless_api.md) section 9 |
| OQ-B5-1..3, D-35 | public replacement bars, the 1M run, GPU 2 rows; `/detect` upscaling after the recall numbers | W5a | as in [06](06_b5_overview.md) section 9 |
| 07d Q1-Q9 | processing-plan behaviour | W5d | as in [06d](06d_b5_processing_plans.md) section 3 |
| Q-1, Q-2 | a Python 3.13 CI job; image lock files | W6 | one 3.13 `pytest` job; `pip freeze` records now, lock files later ([environment_and_setup](reference/environment_and_setup.md) section 8) |
| D-16 | the tag | after R-final | on your word after R-final passes |

### 9.3 The research questions (80, de-duplicated from 112; 73 answered, 7 open)

The full list, grouped by theme with one recommendation each, is [owner_questions.md](reference/owner_questions.md).

**Answered by the owner on 2026-11-04** ([owner_questions](reference/owner_questions.md) section 0; register rows in [decision_register.md](reference/decision_register.md)
section 6): RD-1 every seam now, the review's deferrals rejected (Q-A), so W2.3 is split into W2.3a
and W2.3b; RD-2 short tiered prototype runs only, about 10 GPU hours before the tag, more only on the
owner's OK; RD-4 the plate tables in their own W5e revision; RD-5 capture metadata at ingest; RD-7
tracked prototype code with pins and a lock file; RD-8 set P local only, one shared copy; RD-10 the
datasets, without UFPR-ALPR and LPLC; RD-23 cluster memberships in PostgreSQL; RD-25 the `cluster`
queue (Q-D); RD-51 reversed, GPS coordinates stored now under a per-project `location_policy`
(default keep); RD-54 exports keep the embedded GPS; review Q-B, PP-OCRv6 as one short reader (superseded 2026-11-05 by D-82: the latest models, PP-OCRv6 and the latest Ultralytics, adopted with no comparison run)
comparison.

**Answered by the owner on 2026-11-05:** every other open question, as recommended (D-78): the six that the
W2.1 and W2.2 lanes need (RD-30, RD-34, RD-35, RD-36, RD-70, RD-75) and the rest except the seven below. Among
them, RD-61 and RD-62 (registration-plate text is stored as real readable text, searchable by exact, fuzzy and
fragment match; no keyed or off modes, no deployment ceiling, no master secret or salt, no separate purge: project
delete removes it like any data; D-79, D-80) and RD-67 (yes: the owner's private plate detector stays in their own
deployments, other users bring their own detector through the existing model upload path, and no private model ever
ships in the repo or the images; D-80). The rows are recorded in [owner_questions.md](reference/owner_questions.md).

**Still open: 7 rows** (RD-9, RD-11, RD-12, RD-13, RD-20, RD-26, RD-78), asked in one batch (batch 3) after the tag;
none is needed before W4 and none blocks coding. Each has a recommendation the plan already follows.

MQ-9 ("research experiments after the tag") stays as answered for everything RD-2 does not name;
RD-2 amends it for the short PW runs only.

## 10. After the tag

The research waves RW-1..RW-6 (section 4.11, [08](08_research_prototype_track.md) section 5) run after the
tag in RG-1..RG-6, each window only on the owner's OK and as short tiered runs; the prototypes RD-2
brings forward ran short in PW-1..PW-3. The seams they land on are
already in the rebuild (traceability section 5.2). v0.8.0 ([10_v080_platform_breadth.md](10_v080_platform_breadth.md))
follows, with F3 and F9 shrunk by the seams and F10 (an S3-compatible storage backend for several
hosts, D-46) added. Before their waves
start, re-base the v0.8.0 designs that assumed Redis, OpenSearch leases or an OpenSearch auth index
(F3, F5, F7, F9) on PostgreSQL and Valkey. The program folder is deleted after the tag
([decision_register.md](reference/decision_register.md) section 8.2).

## 11. Fold-in record

Every decided item moved into a lane, with the place its decision is recorded, is listed in
[traceability.md](reference/traceability.md) section 3; every research seam with its lane, the
cross-topic resolutions and the research issue map are in its section 5.
