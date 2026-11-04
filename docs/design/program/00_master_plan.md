# v0.7.0 master plan: the waves from now to the tag (the one file a coding session follows)

| Field | Value |
|---|---|
| Status | plan, written 2026-11-04 on `plan/master` (from `plan/integration` = `origin/main` + the control-plane re-plan + the package pins). Executes wave by wave on the owner's go |
| What this file is | the **execution order**: every wave from now to the v0.7.0 tag, its lanes, file ownership, sources, issues, folded-in decided items, tests, sweep, exit gate, GPU window, and what codes in parallel. It links to the batch files for task detail and never restates it |
| What it is not | the decision register and the legends: those are [00a_decision_register.md](00a_decision_register.md) (section 6 is the register, section 11 the ID legend, section 2.1 the plain-words map). Where this file and a batch file disagree on order, this file wins; on task content, the batch file wins |
| Traceability | [00b_traceability.md](00b_traceability.md): every original task, ID and open issue mapped to its wave and lane, or retired with a reason; the fold-in record; the GitHub actions (approved 2026-11-04, D-68) |
| Review | [00c_plan_review.md](00c_plan_review.md): the independent review (R-1..R-38) and, in its section 5, the planner's response to each finding. Every finding is applied in the plan files; this file carries the wave and lane consequences |
| Owner answers applied | 2026-11-04, after the review: every service requires PostgreSQL, Valkey and OpenSearch, no degraded mode (D-69, which replaced D-64 the same day); event wake-ups over Valkey pub/sub (D-65); kombu and redis-py held (D-66); pins part A as its own PR first (D-67, wave W2.0); every open B2 recommendation accepted as a block; `postgres:18.6-trixie`; the security lane in W4 and the typed-response lanes in W5a-W5c; near-duplicate groups held out together; research experiments after the tag; #42 to v0.8.0 if unrun; the GitHub actions of 00b (D-68) |
| Research integration | 2026-11-04 on `research/master` (from `research/integration` = `main` + the seven research plans): every research seam is folded into the lane that owns its file (tables in [02d](02d_research_seam_schema.md); the seam-to-lane map, the cross-topic resolutions X-1..X-20 and the issue map in [00b](00b_traceability.md) section 5); new lanes 1F, ES-A, ES-B, ES-C, 4F, ND-A, ND-B, ND-C, `b4/capture-metadata`, `b4/identity-text-privacy`, a W2.0 gate lane; a new wave W5e for registration-plate text ([06e](06e_w5e_plate_text.md)); the research prototype track and the post-tag research waves RW-1..RW-6 ([00f](00f_research_track.md)); the 80 de-duplicated research owner questions ([00d](00d_owner_questions.md)). Final pass the same day: the research review ([00e_research_review.md](00e_research_review.md), every finding answered in its Planner responses) and the owner's answers (11 rows approved, among them every seam now, W2.3 split in two, GPS stored per project policy, short prototype runs capped at about 10 GPU hours); 69 questions stay open, none blocks coding, each marked "pending RD-n" where it sits |
| Owner direction applied | 2026-11-04: foundation first (PostgreSQL for control data, Celery and Valkey for every long job, a stateless API that scales to N replicas); then the speed work on it; all package upgrades with the bumps that can change model output riding the B3 GPU gate; no task dropped except with a reason; decided items folded into the lane that already rewrites their code; plain names first; waves with disjoint lanes and one sweep each; CI path-scoped and sharded |

Plain-words rule: every heading and row names the thing first and puts its code after it, for
example "the job runtime (stage 3, lane 3A)". The batch codes are B2 (control plane), B3 (one
Ultralytics version), B4 (stateless API), B5 (speed), R-final (release acceptance).

---

## 1. Where things stand (2026-11-04)

| Item | State |
|---|---|
| B0 (close out the 0.6 audit round) and B1 (correctness under many workers) | merged on `main` (B1 is PR #467) |
| The plan set | `plan/integration` (this branch's base): the PostgreSQL design (files 02-02c), B2 re-planned in six stages (03-03f), exact pins (13). Not on `main` yet: it lands with this plan in one docs PR |
| CI | path-scoped checks merged (PR #476); the 4-shard offline suite is PR #477, **open**: merge it before wave W2.0 (section 8) |
| Open dependabot PRs | #474 (frontend dev group), #449 (Triton base 26.10). Handling in [00b_traceability.md](00b_traceability.md) section 4 |
| Decisions | the B2 set is approved (2026-11-04, as a block); MQ-3, MQ-4, MQ-8, the Flower port and OI-1 / OI-2 (#469, #471 kept as they are) answered 2026-11-04; still open: section 9.2, only the B3, B4, B5 and release questions at their kick-offs, and the 69 open research questions of section 9.3 (none blocks coding) |
| Research plans | seven plans on `research/integration` (GPU pipeline, clustering with its protocol file, embeddings and search, near-duplicates, spatial attributes, registration-plate text, the auto-label assistant), each with seams, prototypes, landing lanes and owner questions; integrated here 2026-11-04; 11 of their questions answered the same day, 69 open (section 9.3) |

## 2. The waves in one table

A wave closes with exactly one full sweep on its integration branch. B2 is one wave (one
integration branch, one PR, one full sweep) in five parts (W2.3 split into W2.3a and W2.3b, review
RR-13, owner Q-A of 2026-11-04); each part ends with a short part check (D-63), not a full sweep.

| Wave | Plain name | Batch and files | Integration branch | GPU | Closing check | Issues it closes |
|---|---|---|---|---|---|---|
| **W2.0** | Dependency pins part A as their own PR, and the local `make gate` | [01](01_package_pins.md) section 7.1, part A (D-67); the gate lane of section 4 | `deps/pins-part-a`, PR to `main` | none | one offline sweep (CPU), run with the new `make gate` | - (N20 (#480) stays open for part B) |
| **W2.1** | Foundation: PostgreSQL, Valkey, the Celery app, the schema, the storage seam, the test database; the research-seam tables (every seam now, RD-1) | B2 stage 1 ([03a](03a_stage1_foundation.md); 1F and [02d](02d_research_seam_schema.md)) | `int/b2-control-plane` | none; **prototype window PW-1** on idle GPU (short runs, RD-2) | part check | - |
| **W2.2** | Control data moves into PostgreSQL; the job machinery (pause, resume and time slices that never spend attempts) and the event feed; the embedding registry | B2 stages 2 and 3 ([03b](03b_stage2_control_data.md), [03c](03c_stage3_job_runtime.md)) | same | none; PW-1 continues | part check | - |
| **W2.3a** | Every long job onto the queue as a port of today's behaviour, training, model export and the evaluator included; the assist loop's own record, pause and budgets | B2 stage 4 lanes 4A-4D ([03d](03d_stage4_job_families.md)) and stage 5 ([03e](03e_stage5_gpu_queues.md)) | same | none (real-GPU smoke deferred to GPU window 1); **PW-2** on idle GPU | part check | #466 (pause, events, budgets) |
| **W2.3b** | The research families on the queue: clustering on versioned partitions; near-duplicate runs, the exact key and its backfill, the holdout guard; the embedding re-index job; Cropwright follows every new state | B2 stage 4 lanes 4F, ND-A, ES-B, ND-B, 4E ([03d](03d_stage4_job_families.md)) | same | none; PW-2 continues | part check | #473 parts a and c |
| **W2.4** | One scheduler; the old job machinery deleted; scale-out compose; hardening; readers on the embedding registry; docs; **the B2 sweep** with the five stale screenshots recaptured | B2 stage 6 ([03f](03f_stage6_scheduler_scaleout.md)); sweep [03](03_b2_overview.md) section 9 | same, then PR to `main` | none; PW-2 continues outside live run 2 | **full B2 sweep + live run 2 of 3** | #461, #462, #453 |
| **W3** | One Ultralytics version and the dependency bumps that can change model output | B3 ([04](04_b3_ultralytics_one_version.md)); pins part B ([01](01_package_pins.md) section 7) | `int/b3-ultralytics` | **GPU window 1** (about 1.5 days, GPU 0) | full B3 sweep + the GPU gate | #458 |
| **W4** | Stateless API: request fan-out out, the worker count a tuning knob, project creation; the privacy and security items; capture metadata and GPS coordinates at ingest under the project's location policy; the identity-text privacy baseline; core near-duplicate grouping off the request path | B4 ([05](05_b4_stateless_api.md)); lanes `b4/privacy-hardening`, `b4/capture-metadata`, `b4/identity-text-privacy` | `int/b4-stateless` | none (its benchmark runs first in window 2); **PW-3** on idle GPU outside live run 3 | full B4 sweep + live run 3 of 3 | #451, #439, #473, #468, #470, #472 (and #469, #471 once OI-1 and OI-2 are answered) |
| **W5a** | Public baseline, Triton sweeps, the `/detect` upscaling question; typed responses part 1 | B5a ([06a](06a_b5_baseline.md)); typed-response lane | `int/b5a-baseline` | **GPU window 2 opens** | full sweep + 5a runs | #45, #268, #463 |
| **W5b** | Decode each image once; typed responses part 2 | B5b ([06b](06b_b5_decode_once.md)); typed-response lane | `int/b5b-decode-once` | window 2 | full sweep + 5b gate | #40, #407 |
| **W5c** | Crops at model size; OCR on the GPU; typed responses part 3 | B5c ([06c](06c_b5_crops_ocr_faces.md)); typed-response lane | `int/b5c-crops-ocr-faces` | window 2 | full sweep + 5c gate | #219, #220 |
| **W5d** | Request-driven processing plans; faces on resident frames (the routers 5d rewrites are typed here) | B5d ([06d](06d_b5_processing_plans.md)) | `int/b5d-processing-plans` | window 2 | full sweep + 5d gate | #67, #464 (remaining boxes moved), #465 |
| **W5e** | Registration-plate text: per-box readings with a real confidence, burst consensus, exact, fuzzy and fragment search, privacy modes (its own revision, RD-4) | [06e](06e_w5e_plate_text.md) | `int/b5e-plate-text` | **window 2 closes** (a short W5e gate) | full sweep + W5e gate | #401 |
| **W6** | Release prep, the one acceptance run, the tag on the owner's word | R-final ([07](07_release_acceptance.md)) | `chore/release-0.7.0-prep` | **GPU window 3** | R-final (every RF row) | #411 after the tag |
| side | 12 GB card acceptance | [06](06_b5_overview.md) section 11 | `perf/42-small-gpu-acceptance` | GPU 1 when its owner frees it | its own pass list | #42 |
| side | Research prototypes that feed a v0.7.0 decision: short tiered runs, about 10 GPU hours in total (RD-2, RD-7) | [00f](00f_research_track.md) sections 2-4 | `research/<topic>-proto` per topic (scripts and notes only) | PW-1 (W2.1-W2.2), PW-2 (W2.3a-W2.4), PW-3 (W4), idle GPU only | each prototype's registered bar | - |
| after the tag | Research waves RW-1..RW-6 (GPU serving breadth, embeddings, clustering, near-duplicate evidence, spatial, the auto-label assistant measured) | [00f](00f_research_track.md) section 5 | `int/rw<n>-<topic>` | research GPU windows RG-1..RG-6, not scheduled: each on the owner's OK | per wave | #403, #231, #347, #365, #435, #436, #346, #406, #416, #348, #400, #466, #398, #84 |

The wave codes W2..W6 keep the meaning they had in [00a_decision_register.md](00a_decision_register.md) before this file
(W0 = B0, W1 = B1, both done); W2 gains the five parts and the pins PR W2.0 before them; W5e is new
(the plate-text wave).

## 3. Dependency diagram

```
main (B0, B1 merged) + this plan set (docs PR) + PR #477 (CI shards)
  |
  v
W2.0 pins part A (own PR to main)
  |
  v
W2.1 foundation ---> W2.2 control data + job runtime ---> W2.3a jobs onto the queue (ports) ---> W2.3b research families ---> W2.4 scheduler, deletions, scale-out, sweep
  |                                                                                                                          |  B2 PR merges to main
  |                                                                                       v
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
   PW-2 during W2.3a-W2.4:  pt:P1..5 with PP-OCRv6 (2.9 h) --> W5c OCR model, W5e reader, tier, consensus
   PW-3 during W4:          gpu:PR-5..6, al:AL0 (1.5 h)    --> W5c, W5d; budget defaults later
   reserve:                 about 2.2 h, tier-2 confirmations only; beyond the cap only on the owner's OK
   CPU slots any time:      clu:CL-0, emb:EP-0, sp:P-0..1 (sp:P-1 gates the W4 capture lane), pt:P0
```

Hard edges: B3 needs B2 merged (P1 of [04](04_b3_ultralytics_one_version.md) section 2); B4 needs
B2 and B3 (it benchmarks once on the final engine); B5 measures only the final B4 stack, and no
optimization code merges before the 5a numbers exist; W5e needs W5c (the `rec_char_min` output) and
W4 (the capture time, the redaction processor); W2.3b needs the W2.3a part check (it calls the
families W2.3a ported and edits their files); R-final runs after W5e. A prototype never blocks a
wave: a decision whose prototype has not run is taken on today's behaviour and the lane says so.

## 4. The waves in detail

Every lane: one agent in its own worktree `<repo>-wt-<lane>` on its own branch, cut from the
integration branch at the point the wave says; it edits only the files it owns; it runs only its
red-then-green tests and static checks on the files it touches. Lane rules and the common brief
block: [03a](03a_stage1_foundation.md) "Lane brief" (B2), [04](04_b3_ultralytics_one_version.md)
section 17 (B3), [05](05_b4_stateless_api.md) section 17 (B4), each 07 file's "Lane briefs" (B5).

Append-only shared files (`env.template`, `src/config/retired_env.py`, the baselines in
`scripts/codegen/baselines/`) and generated `contracts/` follow the merge rules of
[03](03_b2_overview.md) section 4 in every wave.

### W2.0 Dependency pins part A, as its own PR (D-67)

**Goal.** The packages that cannot change model output move to exact `==` pins at their newest
versions in a small PR of their own, so a dependency regression never blocks or muddies the B2 PR.

| Order | Lane | Branch | Owns | Brief | Issues |
|---|---|---|---|---|---|
| 1 | Pins part A: tooling, pure-Python libraries, web layer, clients | `deps/pins-part-a` (from `main`) | `requirements.txt`, `requirements-test.txt`, `pyproject.toml` (existing dependency lines only), `tests/test_requirements_contract.py`, `frontend/package.json` and its lock (re-applies PR #474) | [01](01_package_pins.md) section 7.1, part A | N20 (#480) |
| 1 | The local gate: `make gate` (owner-approved gap, 2026-11-04) | `tools/make-gate` (from `main`, merged into `deps/pins-part-a` before its sweep) | `Makefile` (the `gate` target only), new `scripts/dev/gate.sh`, new `tests/dev/test_gate_script.py`, `CONTRIBUTING.md` (one paragraph) | section 4.0a below | - |

**Sweep and exit.** One CPU offline sweep (the 4 CI shards, run through the new `make gate`, which
is its own first user), pre-commit `--all-files`,
`make contracts-check`, the frontend set (the #474 re-apply touches `frontend/`); the CPU image
builds and `pip check` because requirements changed (the CI image scans run once, here); PR to
`main`, merge commit. Then close PR #474 (00b section 4). **Cut point (re-check RC-10):** the W2.1
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

**Preconditions.** PR #477 (CI shards) and W2.0 merged; this plan set merged to `main`;
`int/b2-control-plane` cut from `main`.

| Order | Lane (plain name) | Branch | Owns | Brief | Issues |
|---|---|---|---|---|---|
| 0 | Control-plane package pins, the `db` marker and its deselection in `addopts` (coordinator commit) | on `int/b2-control-plane` | `requirements.txt`, `requirements-test.txt`, `pyproject.toml` (the new packages with kombu 5.6.2 and redis-py 6.4.0, the `db` marker, the `addopts` expression), `tests/test_control_plane_pins.py` | [03a](03a_stage1_foundation.md) lane 1-0 | #451 |
| 1 | The schema and the database layer | `b2/db-core` | [03a](03a_stage1_foundation.md) lane 1A (minus the `pyproject.toml` marker, now in lane 0) | 03a lane 1A | N19 (#479) |
| 1 | The Celery app, queues, the producer | `b2/queue-core` | 03a lane 1B | 03a lane 1B | #451, #464 (OPS-2 box) |
| 1 | Services, secrets, installer, CLI | `b2/compose-installer` | 03a lane 1C | 03a lane 1C | N19 (#479) |
| 1 | The storage seam | `b2/storage-seam` | 03a lane 1E | 03a lane 1E | N19 (#479) |
| 2 | Health, readiness, the CI database job, the offline commands, the worker-start schema check, the connection budget | `b2/health-ci` (after 1A and 1B merged) | 03a lane 1D (adds `Makefile`, the `ci.yml` and `CONTRIBUTING.md` marker commands, `src/tasks/app.py`'s worker-start call, `tests/config/test_db_connection_budget.py`) | 03a lane 1D | N19 (#479) |
| 2 | The research-seam tables: serving reports, clustering partitions, the embedding registry, the assist-loop record (approved, RD-1) | `b2/db-research-seams` (after 1A merged) | 03a lane 1F (revision `0002_research_seams`, its model files, `src/domain/clustering_vocab.py`, the contract models in `src/schemas/models.py`) | 03a lane 1F; [02d](02d_research_seam_schema.md) | #398, #346, #365, #466 (seams only) |

Merge order: lane 0, 1A, 1E, 1B, 1C, 1F, then 1D. The pieces that need 1A's code while 1A is still
coding moved into 1D (R-18) and 1F. Research seams inside existing lanes: 1A-13 (02d section 1),
1B-8 (`cluster` queue, the job-kind fields, the GPU inventory), 1C-8 (`cluster-worker`), 1C-9 (the
identity-text secret), 1E-4 and 1E-5 (crop size variants, the unknown-area delete).

**Folded in (decided).** `postgres:18.6-trixie` and the Valkey image by the digests of 13 section 4
(D-44, owner decision 6); kombu 5.6.2 and redis-py 6.4.0 (D-66); every Python service waits on
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
| 1 | Projects, slugs, index ownership, pause | `b2/projects-pg` | [03b](03b_stage2_control_data.md) lane 2A | 03d 2A | N19 (#479) |
| 1 | The class registry | `b2/classes-pg` | 03d lane 2B | 03d 2B | N19 (#479) |
| 1 | Config store, VLM endpoints, settings, keymap, view state | `b2/config-pg` | 03d lane 2C | 03d 2C | N19 (#479) |
| 1 | Holdout freezes, label-operation records, project markers | `b2/records-pg` | 03d lane 2D | 03d 2D | N19 (#479), #473 (part c) |
| 1 | The one-step 0.6.0 upgrade and its rollback | `b2/upgrade-060` | 03d lane 2E | 03d 2E | N19 (#479) |
| 1 | The job runtime and the reaper | `b2/job-runtime` | [03c](03c_stage3_job_runtime.md) lane 3A | 03e 3A | #451 |
| 1 | Events in PostgreSQL and resumable SSE | `b2/events-pg` | 03e lane 3B | 03e 3B | #451 |
| 1 | The embedding catalog and spaces (approved, RD-1) | `b2/embedding-registry` | 03d lane ES-A | 03d ES-A | #365 (seams) |
| 2 | Remove the OpenSearch control indexes | `b2/opensearch-control-removal` (after 2A-2E and ES-A merged) | 03d lane 2F | 03d 2F | N19 (#479) |

Cuts: 2A-2E, ES-A and 3B from the merged W2.1 head; **3A after 2A merged** (it reads projects through
2A's repository, R-18); 2F after 2A-2E and ES-A merged. Merge order: 2A, 2B, 2C, 2D, 2E, ES-A, 2F, then 3B, then 3A
(3A codes against the two 3B signatures fixed in 03e 3B-1 and merges the integration head before its
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

**Part check.** 03d and 03e exit criteria: `-m db`, `make contracts-check`, the 20-minute CPU
bring-up (`-p op-b2-s3check`); the Docker-free suite in 4 shards (MQ-3).

**Codes in parallel.** The W2.3a lanes, cut from the merged W2.2 head.

### W2.3a Every long job onto the queue, as ports (B2 stage 4 part 1 and stage 5)

**Goal.** Every job that ran inside an API worker, plus training, model export and the bake-off
evaluator, runs on Celery workers with today's behaviour; the trigger-file workers become queue
consumers; the assist loop gets its own record, pause and budgets. No W2.3a lane calls a research
lane (review RR-13): the `near_dup` scorer, the IVF retrain check and the loop's cluster reader are
ported unchanged, in one module each, for W2.3b to replace.

| Order | Lane | Branch | Owns | Brief | Issues |
|---|---|---|---|---|---|
| 1 | Reprocess, assist runs, scores (the `near_dup` scorer ported unchanged), probe, selection, viz, label ops | `b2/families-a` | [03d](03d_stage4_job_families.md) lane 4A | 03b 4A | #451 |
| 1 | Dataset upload and import, combine, class merge and the name projection | `b2/families-b` | 03b lane 4B (region clustering moved to 4F) | 03b 4B | #451 |
| 1 | Bulk ingest, source polls, open-vocabulary drain, project delete (batched, RR-7), directory ingest, person grouping as a plain job | `b2/families-c` | 03b lane 4C | 03b 4C | #451 |
| 1 | Auto-label and the assist loop as queue consumers; the loop's own record, pause, budgets, slices and usage (gaps G-2, G-3) | `b2/trigger-workers` | 03b lane 4D | 03b 4D | #451, #466 (pause, events, budgets) |
| 1 | Training as a GPU queue consumer | `b2/gpu-train` | [03e](03e_stage5_gpu_queues.md) lane 5A | 03f 5A | #451 |
| 1 | Model upload and export as GPU queue jobs | `b2/gpu-export` | 03f lane 5B | 03f 5B | #451 |
| 1 | The bake-off evaluator as a GPU queue consumer | `b2/gpu-eval` | 03f lane 5C | 03f 5C | #451 |
| 1 | Promotion and the model registry | `b2/promote-models` | 03f lane 5D | 03f 5D | #451 |

Merge order: 5D, 4D, 4A, 4B, 4C, 5A, 5B, 5C (03b header gives the reasons). Size: 8 lanes, one
cross-lane signature (4D's VLM `usage_source` keyword).

**Folded in.** The research seams 5A-6 (reservations by card UUID, the reservation upserts its card,
RR-9), 5B-4 (the export writes the engine contracts and refuses a mismatch), 5D-5 (registry rows
with contracts); the batched project delete (RR-7); the OpenSearch factory injected into job
handlers (R-2); the fake-based family tests rewritten (R-14); `psycopg` C builds in the three GPU
images (R-27).

**Part check.** 03b section 7 and 03f exit criteria: `-m db` and `make contracts-check` after each
merge; the Docker-free suite in 4 shards (MQ-3).

**GPU window.** None in the lanes; stage 5's real-GPU smoke runs in GPU window 1 (W3).

**Codes in parallel.** The W2.3b lanes 4F, ND-A and ES-B, cut from the W2.3a head after its part check.

### W2.3b The research families on the queue (B2 stage 4 part 2)

**Goal.** Clustering runs on versioned partitions, near-duplicates are run-scoped groups with an
exact key and a backfill, the holdout pulls in near-duplicates, the embedding re-index job exists,
and Cropwright follows every new state. Approved by the owner (RD-1, Q-A: every seam now).

| Order | Lane | Branch | Owns | Brief | Issues |
|---|---|---|---|---|---|
| 1 | Clustering on versioned partitions: `cluster_run`, activation, projection, the `cluster_id` split; person grouping and the auto-label consumers moved onto partitions | `b2/clustering` | 03b lane 4F | 03b 4F | #346 (S-core) |
| 1 | Near-duplicates: one service, two levels, run-scoped groups, the exact key, the `source_backfill` job, the `near_dup` scorer deleted | `b2/near-duplicates` | 03b lane ND-A (revision `0003_near_duplicates`) | 03b ND-A | #416, #473 (part a) |
| 1 | The embedding re-index job (time-sliced, RR-20) and the space routes | `b2/embedding-reindex` | 03b lane ES-B | 03b ES-B | #365 (seams) |
| 2 | The near-duplicate holdout guard (replaces 2D-4) | `b2/near-dup-holdout` (after 4F and ND-A merged) | 03b lane ND-B | 03b ND-B | #473 (part c) |
| 3 | Cropwright follows the queued states and every new control | `b2/frontend-queued` (after every other W2.3a and W2.3b lane merged and `make contracts-check` is green) | 03b lane 4E | 03b 4E | #451 |

Merge order: 4F, ND-A, ES-B, ND-B, 4E. Size: 3 lanes at once, then 2 in sequence; every cross-lane
call (`record_partition`, `group_lineage`, `assign.place_new`) is against a signature fixed in 03b.

**Folded in.** #473 part a (stale crop near-duplicate fields) is ND-A's run-scoped projection
(4A-7 superseded); the threshold fail-closed check is retired with `OP_CROP_DUP_THRESHOLD` (ND-9
deletes the scorer); the exact-key backfill for upgraded images and the stored-bytes check (RR-10);
the 4C-7 person grouping recorded as a partition (4F-9).

**Part check.** As W2.3a, after the W2.3b merges.

**GPU window.** None.

**Codes in parallel.** The W2.4 lanes 6A and 6B, cut from the merged W2.3b head. The B3 `frame`
track (no B2 file) may start from `main` on the coordinator's say-so.

### W2.4 One scheduler, the deletions, scale-out, hardening, docs, and the B2 sweep (B2 stage 6)

**Goal.** Timed jobs run once from one beat; `FileJob`, the job files, the leases and every file
lock are deleted; no service has a fixed container name and `--scale` works; the new services are
hardened; the permanent docs describe the control plane. Then the one B2 sweep.

| Order | Lane | Branch | Owns | Brief | Issues |
|---|---|---|---|---|---|
| 1 | Periodic work to beat; old infrastructure deleted | `b2/beat-periodic` | [03f](03f_stage6_scheduler_scaleout.md) lane 6A | 03c 6A | #451, #453, #464 (OPS-2 act part) |
| 1 | Scale-out compose and monitoring (with `monitoring/alloy-config.alloy` and the dashboards, R-12; the `gpu-arbiter-worker` in the arbiter overlay, R-20) | `b2/scale-out-compose` | 03c lane 6B | 03c 6B | #451 |
| 1 | Readers resolve their vector field through the embedding registry (approved, RD-1) | `b2/embedding-readers` | 03c lane ES-C (about 40 files; the coordinator checks them against 6A and 6B at the cut) | 03c ES-C | #365 (seams) |
| 2 | Cropwright reads `embedding_states` | `b2/frontend-spaces` (after ES-C merged) | 03c lane `b2/frontend-spaces` | 03c | - |
| 2 | Hardening (after 6B merged) | `b2/hardening` | 03c lane 6C | 03c 6C | #461, #462 |
| 3 | Docs (last) | `b2/docs` | 03c lane 6D | 03c 6D | N19 (#479) |

**Sweep (the one B2 sweep).** Exactly [03](03_b2_overview.md) section 9 (its T1 offline set, the
**live harness run 2 of 3** with the steps and budget stated there, now including the research steps
19-22 and step 23, the five stale Cropwright screenshots recaptured on the docs overlay and viewed
at desktop and 430 px before commit, the image audit, Trivy and `pip check`, the browser check);
this file does not restate the step count or the budget (R-23).
Then local CI parity, the owner's yes, one PR to `main`, merge commit.

**Exit gate.** The done-when of [03](03_b2_overview.md) section 1 by `rg`; PQ-1..PQ-9 of
[00a_decision_register.md](00a_decision_register.md) section 4; the plan lifecycle deletions for B2 and 02-02c in the
closing PR ([00a_decision_register.md](00a_decision_register.md) section 8.2).

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
real-GPU smoke of B2 stage 5 (03f "GPU check", gate step G12); the GPU part of the image smokes of
13 section 6 (G11), with `cv2.dnn.blobFromImage` in the OpenCV 5 parity set (R-17). Every B3 track
re-verifies its files and symbols on the post-B2 head first (R-18). The GPU-15 leftover (refuse an
undeclared region head when a profile is saved) is part of B3-D3 (MQ-8, approved 2026-11-04).

**Sweep.** [04](04_b3_ultralytics_one_version.md) section 7: T1 offline, the CI `families` matrix
at the pin, the CPU part of G11 (image builds, `pip check`, imports, weight downloads: R-17), image
audit and Trivy (the changed base images make the image scans run once here).

**GPU window 1** (GPU 0, GPU 2 only if agreed; `-p op-b3-gpu`; announced to peers; about 1.5 GPU
days): 04 section 6 G0-G11 (G10 is the B0 OCR before/after comparison; G11 the 13 section 6 image
smokes), plus the B2 stage-5 real-GPU smoke (one short training run, one export, one 20-image
bake-off). Engine EA (Stage A point) now carries the pin-series bumps and the Ultralytics pin
together, so G3 "EA vs E0" judges every version change at once and G3 "EB vs EA" the contract
change. The window has a written cap (its estimate plus one day, R-17): on an EA failure the first
action is the named fallback pins of 13 section 8 for the failing step, then a re-run of that step
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
| 3 | The identity-text privacy baseline (research plate 5.8) | `b4/identity-text-privacy` (after the capture lane merged) | 05 section 6c | 05 section 6c | #401 (privacy baseline) |
| 1 | Privacy and security items (owner decision 7 of 2026-11-04) | `b4/privacy-hardening` | the files named in the maintainers' private notes; the coordinator checks they are disjoint from 4A, 4B and 4D before the cut and moves any shared file's task into the owning track | the private notes; public text stays neutral (one line per issue); plus one docs task, `SECURITY.md`: a plain statement that, when external use is enabled, external VLM endpoints and the hosted research judge may receive any region kind and contact-sheet image (OI-1, OI-2; the matching `docs/CURATION.md` sentence is in track 4D, which owns that file in W4) | #468, #470, #472 (#469 and #471 close with no code change, 00b section 4) |

Merge order: 4A, 4D, privacy lane, capture lane, identity-text lane, 4B (its ratchet allowlist must
see the others' final code). The privacy lane owns `SECURITY.md` in W4 and writes all three
statements (OI-1 and OI-2; the location rules: coordinates stored per `location_policy`, exports keep the embedded GPS unless the project's policy is `drop`; the identity-text section).

**Folded in.** #473 part b, rewritten as ND-C1..ND-C4 (core ingest writes pending rows and the
single-writer `near_dup_core_group` job groups them; task 4A-4b in 05), in the file 4A-4 already
rewrites; the research seams 4A-5's space-aware `_msearch`, 4B-6 (the Triton router over the serving
profile) and 4B-7 (the in-process text encoders deleted: dead code under D-69); `project_create` as a `jobs` kind (R-21); the API pool as a real
deployment total over workers and replicas (R-6).

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

### W5e Registration-plate text (GPU window 2 closes; pending RD-3)

[06e](06e_w5e_plate_text.md): lanes `w5e/region-text-store` (revision `0004_region_text`), then
`w5e/region-text-fields` and `w5e/region-text-jobs` in parallel, then `w5e/identity-text-modes`; the
reader lane and a frontend lane only on their conditions. Preconditions: W5c and W4 merged; the
plate prototypes `pt:P0..P5` read (short runs in PW-2; `pt:P6` cut, so plate boxes come from the segmenter with the text hint). One full sweep and a short GPU gate (about 2 h) at
the end of window 2. **Codes in parallel:** release-prep drafting (moved here from W5d).

### W6 Release prep, R-final, the tag (GPU window 3)

[07](07_release_acceptance.md): RP-1..RP-9 (VERSION and CHANGELOG first), then the one R-final run
(every RF row including the research rows RF-80..RF-86 of 06 section 4.7, the segmenter tier, the 0.6.0 upgrade with an in-flight job), the fix loop only for
a failed row, the tag and release only on the owner's word (D-16). Nothing else merges to `main`.

### Side track: the 12 GB card (#42)

[06](06_b5_overview.md) section 11. Runs on GPU 1 only after its owner session confirms the card is
free; never blocks a wave.

### 4.10 The research prototype track (side; approved, RD-2 and RD-7)

[00f](00f_research_track.md) holds every prototype with its data, hours, dependency, window,
decision and landing wave. In short, per the owner (2026-11-04): working, known code first;
prototypes are simple local-venv GPU scripts (no Docker builds) on set P and freely downloadable
public sets (the dataset register, 00e section 4.1), tracked under `scripts/research/<topic>/` with
exact pins and a committed lock file (RD-7); every run is a short tiered run on a small seeded sample
with early stop (00e section 2.1); the default is today's approach, replaced only on a clear
short-run win. PW-1 (W2.1-W2.2, about 3.4 GPU h), PW-2 (W2.3a-W2.4, about 2.9) and PW-3 (W4, about
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

Lanes, gates and windows: [00f](00f_research_track.md) section 5; their v0.8.0 landings:
[v080_platform_breadth_plan.md](../v080_platform_breadth_plan.md) section 2.11.

## 5. GPU-window budget

Exactly three program windows, each announced to the peer sessions, on GPU 0 (GPU 2 only when
agreed); GPU 1 only for #42 with its owner's consent ([11](11_risks_cleanup_host.md) section 1).
The research prototype windows are separate, run only on idle GPU time and never overlap a program
window or a live-harness run (rows PW-1..PW-3; RD-2: short runs, about 10 GPU hours in total).

| Window | Wave | Contents, in order | Estimate |
|---|---|---|---|
| 1 | W3 | 04 G0 (reference images, an 8.4.48-trainer checkpoint), G1-G9 (engines, Triton contract, COCO parity, throughput, families, promote, embeddings with the OpenCV 5 face path, YOLOv5 secondary, image audit), G10 (B0 OCR before/after), G11 (the GPU part of the pins part B smokes: the CUDA provider, both CuPy wheels, one ModelOpt export, the GPU decode tests, segmenter start-up), G12 (the B2 stage-5 smoke: one short training run, one export, one 20-image bake-off; MLflow client and server together). The CPU parts of G11 run in the W3 sweep (R-17) | about 1.5 GPU days, cap plus one day (R-17) |
| PW-1 | W2.1-W2.2 (research) | `gpu:PR-0`, `PR-1`, `PR-2`; `nd:P0`, `P1`, `P2` | about 3.4 GPU hours |
| PW-2 | W2.3a-W2.4 (research), outside live run 2 | `pt:P1` (with PP-OCRv6), `P2`, `P3`, `P4`, `P5` | about 2.9 GPU hours |
| PW-3 | W4 (research), outside live run 3 and before the 4B-5 benchmark | `gpu:PR-5`, `PR-6`; `al:AL0` (shortened) | about 1.5 GPU hours |
| reserve | any PW window | tier-2 confirmations only | about 2.2 GPU hours |
| 2 | W5a-W5e | the 4B-5 benchmark; 5a `ref`, then `ladders`, `sweeps` (tiered), `recall`; 5b `profile`, then the 5b gate; the #231 head-to-head (optional, after 5b); the 5c gate (one day); 5d `p0` and the 5d gate; the W5e gate (about 2 h) | several GPU days spread over the B5 waves, one run at a time |
| 3 | W6 | the R-final run (segmenter tier, OCR set, B5 confirmation rows) | 1-2 days |
| side | any | #42 on GPU 1 | about half a day |
| RG-1..RG-6 | after the tag (research waves) | [00f](00f_research_track.md) section 2.2; not scheduled, each on the owner's OK | the plans' upper estimates sum to about 70 GPU hours; tier 1 at most 2 h per prototype |

Research GPU total before the tag: **about 10 hours**, the owner's cap (was 31 before and 62 after).

A failed window step is re-run only after its fix is merged to the integration branch, never per
attempt. A lane that needs a GPU number waits for its window.

## 6. Test budget per wave

| Tier | Where it runs |
|---|---|
| T0 red-then-green + paired control, static checks on touched files | every lane, every task |
| Part check (D-63) | W2.1, W2.2, W2.3a, W2.3b: `-m db`, contracts, the merged lanes' tests, a short CPU bring-up |
| T1 full offline sweep | once per wave on the merged head: W2.0 (CPU only), W2.4, W3, W4, W5a-W5d, the release-prep gate |
| Live harness | run 2 in W2.4 (5 h 30 min with the research steps 19-23), run 3 in W4 (2 h); run 1 was B1's smoke (done) |
| GPU | section 5 |

Lanes never run a full suite, `--all-files`, the live harness or a GPU.

## 7. How a coding session runs one wave

1. **Preflight.** Read this file's wave section, then the batch file sections it links; read the
   private ledger's latest "RESUME HERE"; `git fetch origin`, `git worktree list`,
   `gh pr list --state open`; `uptime` and `nvidia-smi`; find the peer sessions and tell them what
   heavy work the wave will run. Check section 9 for the questions the wave needs: never start a
   wave without the owner's go and those answers.
2. **Integration branch.** Cut `int/<batch>` from `main` (W2.1, W3, W4, W5x) or reuse it (W2.2-W2.4).
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
8. **PR.** Local CI parity first (every job of `ci.yml`, and `frontend.yml` when `frontend/`
   changed, the changelog rules included). Ask the owner before pushing. Push the integration
   branch, open one PR against `main` (`gh pr create`), check `gh pr checks`, verify the push with
   `git ls-remote`. Merge with a merge commit on green CI and the owner's yes.
9. **Close-out.** Verify each issue the PR should close is closed; post the prepared comments of
   [00b](00b_traceability.md) section 4 (approved 2026-11-04, D-68); delete the wave's plan files
   after moving their keep-lists (plan lifecycle); remove merged worktrees and branches; update
   the ledger.

## 8. CI and test speed

- **Merge PR #477 first.** The offline suite then runs in 4 balanced shards (about 10 minutes
  instead of 35), which is what makes the per-part offline suite (MQ-3, approved) affordable. Its
  shard command gains `and not db` in lane 1D (R-4).
- **The `db` marker is deselected by default** from W2.1 on: `pyproject.toml` `addopts` (lane 1-0),
  both `ci.yml` commands and `CONTRIBUTING.md` (1D), `CLAUDE.md` (6D); a test asserts the default
  selection holds no `db` test (R-4).
- **The `db` job (1D-2).** Add it behind the existing `changes` path filter in `ci.yml` (code paths
  only), so docs-only PRs stay fast; it needs a Postgres service container, not Docker-in-Docker.
- **Refresh `.test_durations`** in the closing PR of each wave that adds many tests (W2.4, W3,
  W4): an outdated durations file unbalances the shards.
- **One image-scan run per wave.** The image scans run only when a base image or a requirements
  file changes; land each wave's dependency edits in that wave's single PR (W2.0, W2.1 and W3), so
  the scans run three times in the program instead of per bump.
- **Docs-only PRs** (this plan, plan-lifecycle deletions) carry `[skip-changelog]` and touch no
  code path, so they run only the docs jobs.
- **Locally**, the sweep runs as two parallel tracks (Python, frontend) after the installs, with
  xdist sized to the host load; the live harness reuses locally tagged images between W2.4 and W4
  (never `:latest`, which Watchtower re-pulls).

## 9. Owner questions

### 9.1 Answered on 2026-11-04 (recorded in the register, [00a_decision_register.md](00a_decision_register.md) section 6)

Every open B2 recommendation as a block (D-19..D-26, D-46..D-63, with D-48 and D-50 as amended);
every service requires the control plane, no degraded mode (D-69, replacing D-64); Valkey
pub/sub wake-ups (D-65); kombu
5.6.2 and redis-py 6.4.0 held (D-66); pins part A as its own PR (D-67); the GitHub actions of 00b
(D-68); `postgres:18.6-trixie` (MQ-2); near-duplicate groups into the holdout together (MQ-5); the
security lane in W4 (MQ-6); typed-response lanes in W5a-W5c (MQ-7); research experiments after the
tag with their decided items kept in the rewrite (MQ-9); #42 to v0.8.0 if unrun at the tag (MQ-10).
Later the same day: every service requires PostgreSQL, Valkey and OpenSearch with no degraded mode
(D-69, replacing D-64); the Docker-free suite in 4 shards at every B2 part check (MQ-3); Pillow,
simplejpeg, the Hugging Face stack and MLflow in pins part B (MQ-4); the GPU-15 leftover in B3-D3
(MQ-8); Flower on loopback port 4613 and the gateway subpath, only with `monitoring` (Q-3,
ENV-18); #469 and #471 kept as they are, closed with no code change, with a SECURITY.md statement
in the privacy lane (OI-1, OI-2).

### 9.2 Still open, with the plan's recommendation

Only the kick-off questions of B3, B4, B5 and the tag remain; each is asked at its wave's
kick-off, not before.

| ID | Question in plain words | Needed before | Recommendation |
|---|---|---|---|
| D-27, D-28 | Ultralytics staging and the pin rule | W3 | as in [04](04_b3_ultralytics_one_version.md) section 3 |
| D-38 / Q-4, PR #449 | Triton 26.09 or 26.10 and a newer Python base inside B3? | W3 | only if window 1 has room (B3-A5); otherwise after the tag, and close PR #449 now |
| D-29, D-29a, D-32, D-32a, D-33, ENV-17 | B4 fan-out, VLM batch routes, project creation, the worker default, pool names | W4 | as in [05](05_b4_stateless_api.md) section 9 |
| OQ-B5-1..3, D-35 | public replacement bars, the 1M run, GPU 2 rows; `/detect` upscaling after the recall numbers | W5a | as in [06](06_b5_overview.md) section 9 |
| 07d Q1-Q9 | processing-plan behaviour | W5d | as in [06d](06d_b5_processing_plans.md) section 3 |
| Q-1, Q-2 | a Python 3.13 CI job; image lock files | W6 | one 3.13 `pytest` job; `pip freeze` records now, lock files later ([08](08_environment_and_setup.md) section 8) |
| D-16 | the tag | after R-final | on your word after R-final passes |

### 9.3 The research questions (80, de-duplicated from 112; 11 answered, 69 open)

The full list, grouped by theme with one recommendation each, is [00d_owner_questions.md](00d_owner_questions.md).

**Answered by the owner on 2026-11-04** (00d section 0; register rows in [00a_decision_register.md](00a_decision_register.md)
section 6): RD-1 every seam now, the review's deferrals rejected (Q-A), so W2.3 is split into W2.3a
and W2.3b; RD-2 short tiered prototype runs only, about 10 GPU hours before the tag, more only on the
owner's OK; RD-4 the plate tables in their own W5e revision; RD-5 capture metadata at ingest; RD-7
tracked prototype code with pins and a lock file; RD-8 set P local only, one shared copy; RD-10 the
datasets, without UFPR-ALPR and LPLC; RD-23 cluster memberships in PostgreSQL; RD-25 the `cluster`
queue (Q-D); RD-51 reversed, GPS coordinates stored now under a per-project `location_policy`
(default keep); RD-54 exports keep the embedded GPS; review Q-B, PP-OCRv6 as one short reader
comparison.

**Still open: 69 rows, none blocks coding.** Each has a recommendation the plan already follows and
is asked only before the wave named in 00d (most at W2.3a, W2.3b, W4, W5b, W5e or after the tag).

MQ-9 ("research experiments after the tag") stays as answered for everything RD-2 does not name;
RD-2 amends it for the short PW runs only.

## 10. After the tag

The research waves RW-1..RW-6 (section 4.11, [00f](00f_research_track.md) section 5) run after the
tag in RG-1..RG-6, each window only on the owner's OK and as short tiered runs; the prototypes RD-2
brings forward ran short in PW-1..PW-3. The seams they land on are
already in the rebuild (00b section 5.2). v0.8.0 ([v080_platform_breadth_plan.md](../v080_platform_breadth_plan.md))
follows, with F3 and F9 shrunk by the seams and F10 (an S3-compatible storage backend for several
hosts, D-46) added. Before their waves
start, re-base the v0.8.0 designs that assumed Redis, OpenSearch leases or an OpenSearch auth index
(F3, F5, F7, F9) on PostgreSQL and Valkey. The program folder is deleted after the tag
([00a_decision_register.md](00a_decision_register.md) section 8.2).

## 11. Fold-in record

Every decided item moved into a lane, with the place its decision is recorded, is listed in
[00b_traceability.md](00b_traceability.md) section 3; every research seam with its lane, the
cross-topic resolutions and the research issue map are in its section 5.
