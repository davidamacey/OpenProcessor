# v0.7.0 master plan: the waves from now to the tag (the one file a coding session follows)

| Field | Value |
|---|---|
| Status | plan, written 2026-11-04 on `plan/master` (from `plan/integration` = `origin/main` + the control-plane re-plan + the package pins). Executes wave by wave on the owner's go |
| What this file is | the **execution order**: every wave from now to the v0.7.0 tag, its lanes, file ownership, sources, issues, folded-in decided items, tests, sweep, exit gate, GPU window, and what codes in parallel. It links to the batch files for task detail and never restates it |
| What it is not | the decision register and the legends: those are [00_index.md](00_index.md) (section 6 is the register, section 11 the ID legend, section 2.1 the plain-words map). Where this file and a batch file disagree on order, this file wins; on task content, the batch file wins |
| Traceability | [00b_traceability.md](00b_traceability.md): every original task, ID and open issue mapped to its wave and lane, or retired with a reason; the fold-in record; the proposed GitHub actions |
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
| CI | path-scoped checks merged (PR #476); the 4-shard offline suite is PR #477, **open**: merge it before wave W2.1 (section 8) |
| Open dependabot PRs | #474 (frontend dev group), #449 (Triton base 26.10). Handling in [00b_traceability.md](00b_traceability.md) section 4 |
| Open decisions | the B2 set (D-19..D-26, D-46..D-63), B3 (D-27, D-28, D-38), B4 (D-29..D-33), B5 (D-35, OQ-B5-1..3, 07d Q1-Q9), and the new questions of this pass: section 9 |

## 2. The waves in one table

A wave closes with exactly one full sweep on its integration branch. B2 is one wave (one
integration branch, one PR, one full sweep) in four parts; each part ends with a short part check
(D-63), not a full sweep.

| Wave | Plain name | Batch and files | Integration branch | GPU | Closing check | Issues it closes |
|---|---|---|---|---|---|---|
| **W2.1** | Foundation: PostgreSQL, Valkey, the Celery app, the schema, the storage seam, the test database; first dependency pins | B2 stage 1 ([03a](03a_b2_w1_queue_infra.md)); pins part A ([13](13_package_pins.md) section 7) | `int/b2-control-plane` | none | part check | - |
| **W2.2** | Control data moves into PostgreSQL; the job machinery and the event feed | B2 stages 2 and 3 ([03d](03d_b2_s2_control_data.md), [03e](03e_b2_s3_job_runtime.md)) | same | none | part check | - |
| **W2.3** | Every long job onto the queue, training, model export and the evaluator included | B2 stages 4 and 5 ([03b](03b_b2_w2_job_families.md), [03f](03f_b2_s5_gpu_queues.md)) | same | none (real-GPU smoke deferred to GPU window 1) | part check | - |
| **W2.4** | One scheduler; the old job machinery deleted; scale-out compose; hardening; docs; **the B2 sweep** | B2 stage 6 ([03c](03c_b2_w3_beat_reaper_workers.md)); sweep [03](03_b2_overview.md) section 9 | same, then PR to `main` | none | **full B2 sweep + live run 2 of 3** | #461, #462, #453 |
| **W3** | One Ultralytics version and the dependency bumps that can change model output | B3 ([04](04_b3_ultralytics_one_version.md)); pins part B ([13](13_package_pins.md) section 7) | `int/b3-ultralytics` | **GPU window 1** (about 1.5 days, GPU 0) | full B3 sweep + the GPU gate | #458 |
| **W4** | Stateless API: request fan-out out, the worker count a tuning knob, project creation | B4 ([05](05_b4_stateless_api.md)) | `int/b4-stateless` | none (its benchmark runs first in window 2) | full B4 sweep + live run 3 of 3 | #451, #439, #473 |
| **W5a** | Public baseline, Triton sweeps, the `/detect` upscaling question | B5a ([07a](07a_b5_baseline.md)) | `int/b5a-baseline` | **GPU window 2 opens** | full sweep + 5a runs | #45, #268, #463 |
| **W5b** | Decode each image once | B5b ([07b](07b_b5_decode_once.md)) | `int/b5b-decode-once` | window 2 | full sweep + 5b gate | #40, #407 |
| **W5c** | Crops at model size; OCR on the GPU | B5c ([07c](07c_b5_crops_ocr_faces.md)) | `int/b5c-crops-ocr-faces` | window 2 | full sweep + 5c gate | #219, #220 |
| **W5d** | Request-driven processing plans; faces on resident frames | B5d ([07d](07d_b5_processing_plans.md)) | `int/b5d-processing-plans` | **window 2 closes** | full sweep + 5d gate | #67, #464 (remaining boxes moved) |
| **W6** | Release prep, the one acceptance run, the tag on the owner's word | R-final ([06](06_release_acceptance.md)) | `chore/release-0.7.0-prep` | **GPU window 3** | R-final (every RF row) | #411 after the tag |
| side | 12 GB card acceptance | [07](07_b5_overview.md) section 11 | `perf/42-small-gpu-acceptance` | GPU 1 when its owner frees it | its own pass list | #42 |

The wave codes W2..W6 keep the meaning they had in [00_index.md](00_index.md) before this file
(W0 = B0, W1 = B1, both done); W2 gains the four parts.

## 3. Dependency diagram

```
main (B0, B1 merged) + this plan set (docs PR) + PR #477 (CI shards)
  |
  v
W2.1 foundation ---> W2.2 control data + job runtime ---> W2.3 jobs onto the queue ---> W2.4 scheduler, deletions, scale-out, sweep
  |  (pins part A rides here)                                                             |  B2 PR merges to main
  |                                                                                       v
  |                                    W3 lanes start from the frozen int/b2 head --> W3 B3 + pins part B --[GPU window 1]--> B3 PR
  |                                                                                                          |
  |                                                        W4 lanes start from the frozen int/b3 head --> W4 stateless API --> B4 PR
  |                                                                                                          |
  |                                                    [GPU window 2: 4B-5 benchmark -> 5a -> 5b -> (#231) -> 5c -> 5d]
  |                                                                                                          v
  |                                                                         W5a -> W5b -> W5c -> W5d (one PR each)
  |                                                                                                          |
  |                                                                         W6 release prep -> R-final [GPU window 3] -> tag v0.7.0 (owner's word)
side: #42 on GPU 1 whenever its owner frees it (never blocks)
```

Hard edges: B3 needs B2 merged (P1 of [04](04_b3_ultralytics_one_version.md) section 2); B4 needs
B2 and B3 (it benchmarks once on the final engine); B5 measures only the final B4 stack, and no
optimization code merges before the 5a numbers exist; R-final runs after 5d.

## 4. The waves in detail

Every lane: one agent in its own worktree `<repo>-wt-<lane>` on its own branch, cut from the
integration branch at the point the wave says; it edits only the files it owns; it runs only its
red-then-green tests and static checks on the files it touches. Lane rules and the common brief
block: [03a](03a_b2_w1_queue_infra.md) "Lane brief" (B2), [04](04_b3_ultralytics_one_version.md)
section 17 (B3), [05](05_b4_stateless_api.md) section 17 (B4), each 07 file's "Lane briefs" (B5).

Append-only shared files (`env.template`, `src/config/retired_env.py`, the baselines in
`scripts/codegen/baselines/`) and generated `contracts/` follow the merge rules of
[03](03_b2_overview.md) section 4 in every wave.

### W2.1 Foundation (B2 stage 1, plus pins part A)

**Goal.** The stack starts with PostgreSQL at the schema head, Valkey, idle Celery workers and
beat; the API refuses to start on a stale schema; tests get a real database; the packages that
cannot change model output move to exact pins. No job family moves yet.

**Preconditions.** PR #477 (CI shards) merged; this plan set merged to `main`; the owner's answers
to the "before W2.1" questions (section 9.1); `int/b2-control-plane` cut from `main`.

| Order | Lane (plain name) | Branch | Owns | Brief | Issues |
|---|---|---|---|---|---|
| 0 | Control-plane package pins and the `db` test marker (coordinator commit) | on `int/b2-control-plane` | `requirements.txt`, `requirements-test.txt`, `pyproject.toml` (the new packages and the `db` marker), `tests/test_control_plane_pins.py` | [03a](03a_b2_w1_queue_infra.md) lane 1-0 | #451 |
| 1 | Pins part A: tooling, pure-Python libraries, web layer, clients | `deps/pins-part-a` | `requirements.txt`, `requirements-test.txt`, `pyproject.toml` (existing lines only), `tests/test_requirements_contract.py`, `frontend/package.json` and its lock (re-applies PR #474) | [13](13_package_pins.md) section 7, part A | N20 (new) |
| 1 | The schema and the database layer | `b2/db-core` | [03a](03a_b2_w1_queue_infra.md) lane 1A (minus the `pyproject.toml` marker, now in lane 0) | 03a lane 1A | N19 (new) |
| 1 | The Celery app, queues, the producer | `b2/queue-core` | 03a lane 1B | 03a lane 1B | #451, #464 (OPS-2 box) |
| 1 | Services, secrets, installer, CLI | `b2/compose-installer` | 03a lane 1C | 03a lane 1C | N19 |
| 1 | The storage seam | `b2/storage-seam` | 03a lane 1E | 03a lane 1E | N19 |
| 2 | Health and the CI database job | `b2/health-ci` (after 1A and 1B merged) | 03a lane 1D | 03a lane 1D | N19 |

Merge order: lane 0, pins part A, 1A, 1E, 1B, 1C, then 1D. Pins part A edits the three manifests
after lane 0 is committed, so the two never run at once on those files.

**Folded in (decided).** The images and test extras follow [13](13_package_pins.md) section 4
(D-44); the Postgres image choice is question MQ-2. OPS-2 is measured first (1B-0, #464).

**Red-then-green tests.** Each lane's table in 03a; pins part A: the generalized pin test is red on
today's floors, with a shrink-only allowlist for the packages that move in W3 (13 section 7).

**Part check.** 03a "Stage 1 exit criteria": the lanes' tests on the merged head, `-m db`,
`make contracts-check`, the 15-minute CPU bring-up (`-p op-b2-s1check`); plus the Docker-free
suite in 4 shards if MQ-3 is approved.

**GPU window.** None. Spikes S0 and S1 are CPU-only with their own `-p` names.

**Codes in parallel during the part check.** The W2.2 lanes, cut from the merged W2.1 head.

### W2.2 Control data into PostgreSQL, and the job machinery (B2 stages 2 and 3)

**Goal.** Every project, class, config, setting and lasting record is read and written in
PostgreSQL; OpenSearch keeps two indexes per project; the one-step 0.6.0 upgrade exists and is
tested; the job start, claim, heartbeat, cancel, reaper and the event feed exist (no family moves
yet).

| Order | Lane | Branch | Owns | Brief | Issues |
|---|---|---|---|---|---|
| 1 | Projects, slugs, index ownership, pause | `b2/projects-pg` | [03d](03d_b2_s2_control_data.md) lane 2A | 03d 2A | N19 |
| 1 | The class registry | `b2/classes-pg` | 03d lane 2B | 03d 2B | N19 |
| 1 | Config store, VLM endpoints, settings, keymap, view state | `b2/config-pg` | 03d lane 2C | 03d 2C | N19 |
| 1 | Holdout freezes, label-operation records, project markers | `b2/records-pg` | 03d lane 2D | 03d 2D | N19, #473 (part c) |
| 1 | The one-step 0.6.0 upgrade and its rollback | `b2/upgrade-060` | 03d lane 2E | 03d 2E | N19 |
| 1 | The job runtime and the reaper | `b2/job-runtime` | [03e](03e_b2_s3_job_runtime.md) lane 3A | 03e 3A | #451 |
| 1 | Events in PostgreSQL and resumable SSE | `b2/events-pg` | 03e lane 3B | 03e 3B | #451 |
| 2 | Remove the OpenSearch control indexes | `b2/opensearch-control-removal` (after 2A-2E merged) | 03d lane 2F | 03d 2F | N19 |

Merge order: 2A, 2B, 2C, 2D, 2E, 2F, then 3A, 3B (stage 3 codes from the W2.1 head alongside
stage 2 and merges after 2F).

**Folded in.** #473 part c (the holdout freeze checks near-duplicates; task 2D-4 in 03d; the
behaviour is question MQ-5).

**Part check.** 03d and 03e exit criteria: `-m db`, `make contracts-check`, the 20-minute CPU
bring-up (`-p op-b2-s3check`); the Docker-free suite in 4 shards if MQ-3.

**Codes in parallel.** The W2.3 lanes, cut from the merged W2.2 head.

### W2.3 Every long job onto the queue (B2 stages 4 and 5)

**Goal.** Every job that ran inside an API worker, plus training, model export and the bake-off
evaluator, runs on Celery workers; the trigger-file workers become queue consumers; Cropwright
shows `queued`, `cancelling` and `paused`.

| Order | Lane | Branch | Owns | Brief | Issues |
|---|---|---|---|---|---|
| 1 | Reprocess, assist runs, scores, probe, selection, viz, label ops, near-duplicate scoring | `b2/families-a` | [03b](03b_b2_w2_job_families.md) lane 4A (now with `item_scores/near_dup.py`) | 03b 4A | #451, #473 (part a) |
| 1 | Region clusters, dataset upload and import, combine, class merge and the name projection | `b2/families-b` | 03b lane 4B | 03b 4B | #451 |
| 1 | Bulk ingest, source polls, open-vocabulary drain, project delete, directory ingest, person grouping | `b2/families-c` | 03b lane 4C (without `near_dup.py`) | 03b 4C | #451 |
| 1 | Auto-label and the assist loop as queue consumers | `b2/trigger-workers` | 03b lane 4D | 03b 4D | #451, #466 (pause and events transport) |
| 1 | Training as a GPU queue consumer | `b2/gpu-train` | [03f](03f_b2_s5_gpu_queues.md) lane 5A | 03f 5A | #451 |
| 1 | Model upload and export as GPU queue jobs | `b2/gpu-export` | 03f lane 5B | 03f 5B | #451 |
| 1 | The bake-off evaluator as a GPU queue consumer | `b2/gpu-eval` | 03f lane 5C | 03f 5C | #451 |
| 1 | Promotion and the model registry | `b2/promote-models` | 03f lane 5D | 03f 5D | #451 |
| 2 | Cropwright follows the queued states | `b2/frontend-queued` (after the eight above merged and `make contracts-check` is green) | 03b lane 4E | 03b 4E | #451 |

Merge order: 5D, 4A, 4B, 4C, 4D, 5A, 5B, 5C, then 4E.

**Folded in.** #473 part a (a re-score recomputes the crop near-duplicate fields; 03b task 4A-7);
the near-duplicate threshold fail-closed check moves with its file from 4C to 4A (03b task 4A-6,
was 4C-8; the clustering plan's N-8 is the same fix).

**Part check.** 03b section 7 and 03f exit criteria: `-m db` and `make contracts-check` after each
merge; the Docker-free suite in 4 shards if MQ-3.

**GPU window.** None in the lanes; stage 5's real-GPU smoke runs in GPU window 1 (W3).

**Codes in parallel.** The W2.4 lanes 6A and 6B, cut from the merged W2.3 head. The B3 `frame`
track (no B2 file) may start from `main` on the coordinator's say-so.

### W2.4 One scheduler, the deletions, scale-out, hardening, docs, and the B2 sweep (B2 stage 6)

**Goal.** Timed jobs run once from one beat; `FileJob`, the job files, the leases and every file
lock are deleted; no service has a fixed container name and `--scale` works; the new services are
hardened; the permanent docs describe the control plane. Then the one B2 sweep.

| Order | Lane | Branch | Owns | Brief | Issues |
|---|---|---|---|---|---|
| 1 | Periodic work to beat; old infrastructure deleted | `b2/beat-periodic` | [03c](03c_b2_w3_beat_reaper_workers.md) lane 6A | 03c 6A | #451, #453, #464 (OPS-2 act part) |
| 1 | Scale-out compose | `b2/scale-out-compose` | 03c lane 6B | 03c 6B | #451 |
| 2 | Hardening (after 6B merged) | `b2/hardening` | 03c lane 6C | 03c 6C | #461, #462 |
| 3 | Docs (last) | `b2/docs` | 03c lane 6D | 03c 6D | N19 |

**Sweep (the one B2 sweep).** [03](03_b2_overview.md) section 9: T1 offline (pre-commit
`--all-files` and pre-push, the Docker-free suite, `-m db`, contracts, docs check, the frontend full
set), **live harness run 2 of 3** (`-p op-b2-live`, budget 4 h, the 16 steps), image audit and
Trivy for the changed images, `pip check` per Python image, the browser check at 1440 and 430 px.
Then local CI parity, the owner's yes, one PR to `main`, merge commit.

**Exit gate.** The done-when of [03](03_b2_overview.md) section 1 by `rg`; PQ-1..PQ-9 of
[00_index.md](00_index.md) section 4; the plan lifecycle deletions for B2 and 02-02c in the
closing PR ([00_index.md](00_index.md) section 8.2).

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
| 1 | Pins part B: the model-output set and the image pins | `deps/pins-part-b` | the three manifests, `requirements-export.txt` and the `export` extra (versions only), `docker/trainer/requirements.txt`, `scripts/curation/bakeoff/requirements.txt`, `Dockerfile.triton`, `docker/hardened/triton/Dockerfile`, `docker/segmenter/Dockerfile`, new `docker/triton/requirements.txt` and `docker/segmenter/requirements.txt`, the root `Dockerfile` (the evaluator's inline `mlflow`), `docker-compose.yml` (`MLFLOW_IMAGE` only), `CONTRIBUTING.md` and `.github/workflows/ci.yml` (the CPU torch-pair commands only), `.github/dependabot.yml`, `tests/test_requirements_contract.py` | [13](13_package_pins.md) section 7, part B; [04](04_b3_ultralytics_one_version.md) track `deps` | N20, #458 |
| 1 | Each adapter declares its box frame (GPU-14 final) | `b3/frame` | 04 track `frame` | 04 B3-G1 | #458 |
| 2 | Stage A: the pin, the second environment deleted, the trainer on the pin | `b3/stagea` (from the pins part B merge) | 04 track `stagea` | 04 B3-A1..A4 (A5 only if D-38 yes) | #458 |
| 3 | The family registry and the full CPU matrix | `b3/families` (from `b3/stagea`) | 04 track `families` | 04 B3-F1..F3 | #458, #368 (first slice) |
| - | Stage A point recorded | coordinator | | 04 section 4 | |
| 4 | One export module (Stage B) | `b3/export` | 04 track `export` | 04 B3-E1..E7 | #458 |
| 4 | Fused-only decoders (Stage B) | `b3/decoders` | 04 track `decoders` | 04 B3-D1..D3 | #458 |
| 5 | Docs for the stage that ships (after the gate verdict) | `b3/docs` | 04 track `docs` | 04 B3-W1 | #458 |

Merge order: pins part B, `frame`, `stagea`, `families`, Stage A point, `export`, `decoders`, the
GPU gate, `docs`. `requirements-export-y11.txt` is not bumped by pins part B: B3-A2 deletes it.

**Folded in.** The pin-series bumps that can change model output (owner direction 3); the
real-GPU smoke of B2 stage 5 (03f "GPU check"); the image smoke checks of 13 section 6 (gate step
G11 in 04). Question MQ-8 asks whether the GPU-15 leftover joins B3-D3.

**Sweep.** [04](04_b3_ultralytics_one_version.md) section 7: T1 offline, the CI `families` matrix
at the pin, image audit and Trivy (the changed base images make the image scans run once here).

**GPU window 1** (GPU 0, GPU 2 only if agreed; `-p op-b3-gpu`; announced to peers; about 1.5 GPU
days): 04 section 6 G0-G11 (G10 is the B0 OCR before/after comparison; G11 the 13 section 6 image
smokes), plus the B2 stage-5 real-GPU smoke (one short training run, one export, one 20-image
bake-off). Engine EA (Stage A point) now carries the pin-series bumps and the Ultralytics pin
together, so G3 "EA vs E0" judges every version change at once and G3 "EB vs EA" the contract
change. If EA fails, the coordinator bisects by reverting pins part B commits on a scratch
branch and re-runs only the failing steps after the fix merges (budget rule).

**Codes in parallel during the sweep and the gate.** W4 lanes cut from the frozen
`int/b3-ultralytics` head (B4 touches no B3 file); 4D's stage timer.

### W4 Stateless API (B4)

**Goal.** No request handler fans out; the worker count is one knob (`API_WORKERS`); pools are
sized per process from totals; project creation leaves the request; the same answers with 1 and N
workers and replicas.

| Order | Lane | Branch | Owns | Brief | Issues |
|---|---|---|---|---|---|
| 1 | Request fan-out becomes bulk writes; the ingest data plane as task groups | `b4/bulk-occ` | [05](05_b4_stateless_api.md) track 4A | 05 section 4 | #451, #464 (CC-10 box), #473 (part b) |
| 2 | Project creation (#439) | `b4/project-create` | 05 track 4D | 05 section 6 | #439 |
| 3 | The worker knob and the statelessness ratchet | `b4/worker-knob` | 05 track 4B | 05 section 5 | #451 |
| 1 | Privacy and security items (only if MQ-6 yes) | `b4/privacy-hardening` | the files named in the private notes; the coordinator checks they are disjoint from 4A, 4B, 4D before the cut | MQ-6 | #468, #469, #470, #471, #472 |

Merge order: 4A, 4D, privacy lane, 4B (its ratchet allowlist must see the others' final code).

**Folded in.** #473 part b (the core batch near-duplicate grouping race; task 4A-4b in 05), in the
file 4A-4 already rewrites.

**Sweep.** 05 section 7: T1 offline, frontend full set, the browser check of every streaming page,
**live harness run 3 of 3** (`-p op-b4-live`, budget 2 h): the 1-vs-N diff (1 and 32 workers; two
`api` and two `jobs-worker` replicas) and the 4D stage timings.

**GPU window.** None: the 4B-5 benchmark is the first job of GPU window 2.

**Codes in parallel during the sweep.** W5a `ladders` (A1) and `recall` (R1) code; release-notes
drafting (RP-4); the typed-response lanes if MQ-7 is approved.

### W5a Public baseline (B5a; GPU window 2 opens)

**Goal and lanes.** [07a](07a_b5_baseline.md): tracks `ref`, `ladders`, `sweeps`, `recall`; first
the 4B-5 benchmark of [05](05_b4_stateless_api.md) (the 32-worker baseline retaken on the B3
engine, then the `API_WORKERS` default). Owner answers OQ-B5-1..3 before it; D-35 is decided on
the recall numbers at its end (#463). No optimization code merges before the 5a numbers exist.

**Sweep.** T1 offline; the 5a runs in window 2. **Codes in parallel:** the W5b lanes.

### W5b Decode once (B5b)

[07b](07b_b5_decode_once.md): tracks `decode` (GPU-3b first), `profile`, `prep` (#407 staged,
GPU-12), `routes` (#40 Wave 6), `codec`. Gate in window 2. The serving head-to-head (#231) may run
after the 5b gate in the same window, under its own rule ([07](07_b5_overview.md) section 10), or
is recorded and closed. **Codes in parallel:** the W5c lanes.

### W5c Crops at model size, OCR on the GPU (B5c)

[07c](07c_b5_crops_ocr_faces.md): tracks `crops` (#219), `ocr` (#220, GPU-7), `ocrnorm` (GPU-19
evidence, #464 box). One GPU day in window 2 for P-vlm, P-sam, P-region, P-ocr. **Codes in
parallel:** the W5d lanes.

### W5d Request-driven processing plans (B5d; GPU window 2 closes)

[07d](07d_b5_processing_plans.md): tracks `p0`, `analyze`, `curation`, `gpu` (faces on resident
frames), `docs`. Owner answers 07d Q1-Q9 at kick-off. **Codes in parallel:** release-prep
drafting.

### W6 Release prep, R-final, the tag (GPU window 3)

[06](06_release_acceptance.md): RP-1..RP-9 (VERSION and CHANGELOG first), then the one R-final run
(every RF row, the segmenter tier, the 0.6.0 upgrade with an in-flight job), the fix loop only for
a failed row, the tag and release only on the owner's word (D-16). Nothing else merges to `main`.

### Side track: the 12 GB card (#42)

[07](07_b5_overview.md) section 11. Runs on GPU 1 only after its owner session confirms the card is
free; never blocks a wave.

## 5. GPU-window budget

Exactly three windows, each announced to the peer sessions, on GPU 0 (GPU 2 only when agreed);
GPU 1 only for #42 with its owner's consent ([11](11_risks_cleanup_host.md) section 1).

| Window | Wave | Contents, in order | Estimate |
|---|---|---|---|
| 1 | W3 | 04 G0 (reference images, an 8.4.48-trainer checkpoint), G1-G9 (engines, Triton contract, COCO parity, throughput, families, promote, embeddings, YOLOv5 secondary, image audit), G10 (B0 OCR before/after), G11 (pins part B image smokes: `pip check`, imports, the CUDA provider, both CuPy wheels, one ModelOpt export, Triton numpy, the GPU decode tests, segmenter start-up and the weight download), the B2 stage-5 smoke (train 1 short run, export, 20-image bake-off; MLflow client and server together) | about 1.5 GPU days |
| 2 | W5a-W5d | the 4B-5 benchmark; 5a `ref`, then `ladders`, `sweeps` (tiered), `recall`; 5b `profile`, then the 5b gate; the #231 head-to-head (optional, after 5b); the 5c gate (one day); 5d `p0` and the 5d gate | several GPU days spread over the B5 waves, one run at a time |
| 3 | W6 | the R-final run (segmenter tier, OCR set, B5 confirmation rows) | 1-2 days |
| side | any | #42 on GPU 1 | about half a day |

A failed window step is re-run only after its fix is merged to the integration branch, never per
attempt. A lane that needs a GPU number waits for its window.

## 6. Test budget per wave

| Tier | Where it runs |
|---|---|
| T0 red-then-green + paired control, static checks on touched files | every lane, every task |
| Part check (D-63) | W2.1, W2.2, W2.3: `-m db`, contracts, the merged lanes' tests, a short CPU bring-up |
| T1 full offline sweep | once per wave on the merged head: W2.4, W3, W4, W5a-W5d, the release-prep gate |
| Live harness | run 2 in W2.4 (4 h), run 3 in W4 (2 h); run 1 was B1's smoke (done) |
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
9. **Close-out.** Verify each issue the PR should close is closed; post the prepared comments (only
   those the owner approved in [00b](00b_traceability.md) section 4); delete the wave's plan files
   after moving their keep-lists (plan lifecycle); remove merged worktrees and branches; update
   the ledger.

## 8. CI and test speed

- **Merge PR #477 first.** The offline suite then runs in 4 balanced shards (about 10 minutes
  instead of 35), which is what makes the optional per-part offline suite (MQ-3) affordable.
- **The `db` job (1D-2).** Add it behind the existing `changes` path filter in `ci.yml` (code paths
  only), so docs-only PRs stay fast; it needs a Postgres service container, not Docker-in-Docker.
- **Refresh `.test_durations`** in the closing PR of each wave that adds many tests (W2.4, W3,
  W4): an outdated durations file unbalances the shards.
- **One image-scan run per wave.** The image scans run only when a base image or a requirements
  file changes; land each wave's dependency edits in that wave's single PR (W2.1 and W3), so the
  scans run twice in the program instead of per bump.
- **Docs-only PRs** (this plan, plan-lifecycle deletions) carry `[skip-changelog]` and touch no
  code path, so they run only the docs jobs.
- **Locally**, the sweep runs as two parallel tracks (Python, frontend) after the installs, with
  xdist sized to the host load; the live harness reuses locally tagged images between W2.4 and W4
  (never `:latest`, which Watchtower re-pulls).

## 9. Owner questions

Each has the plan's recommendation. The register in [00_index.md](00_index.md) section 6 keeps
the status of every D-number; the MQ items are new in this pass.

### 9.1 Before W2.1

| ID | Question in plain words | Recommendation |
|---|---|---|
| MQ-1 | Accept every open B2 recommendation as written (D-19, D-20, D-20a, D-20b, D-22a, D-22b, D-23, D-24, D-25, D-26, D-46..D-63), naming only the exceptions? | yes |
| MQ-2 | The Postgres image: the pin file's `postgres:18.6-trixie` (glibc, the best-supported text sort order) or the B2 files' `18-alpine`? | trixie, by digest, as [13](13_package_pins.md) section 4 says |
| MQ-3 | Also run the Docker-free offline suite (4 shards, about 10 min) at each B2 part check, besides the database tier? | yes, cheap since PR #477 |
| ENV-18 / Q-3 | Flower: port 4613 on loopback, or only behind the gateway subpath? | both off unless `monitoring` runs; loopback port 4613 plus the subpath |
| MQ-4 | Pillow and simplejpeg decode images, so by your criterion they ride the W3 GPU gate with OpenCV; the Hugging Face stack and MLflow sit between and after the gate-set commits in the bisect order, so they ride there too. Agreed? | yes |

### 9.2 Before W2.2 and W2.3

| ID | Question | Recommendation |
|---|---|---|
| MQ-5 | #473: when a holdout is frozen, what happens to near-duplicates of held-out items? | the whole near-duplicate group goes into the holdout (group-aware split), and the freeze report lists how many moved |
| MQ-6 | #468-#472 (five neutral security items in v0.7.0, not in any plan yet): one W4 lane, scoped from the private notes? #469 and #471 also need your decision text | yes, W4 lane `b4/privacy-hardening` |

### 9.3 Before W3, W4 and B5

| ID | Question | Recommendation |
|---|---|---|
| D-27, D-28 | Ultralytics staging and the pin rule | as in [04](04_b3_ultralytics_one_version.md) section 3 |
| D-38 / Q-4, PR #449 | Triton 26.09 or 26.10 and a newer Python base inside B3? | only if window 1 has room (B3-A5); otherwise after the tag, and close PR #449 |
| MQ-8 | Fold the GPU-15 leftover (refuse an undeclared region head at profile validation) into B3-D3, which rewrites that path? | yes |
| D-29, D-29a, D-32, D-32a, D-33, ENV-17 | B4 fan-out, VLM batch routes, project creation, the worker default, pool names | as in [05](05_b4_stateless_api.md) section 9 |
| MQ-7 | #465 typed response models (BA-6, in v0.7.0 but not in any wave): CPU lanes during W5a-W5c, one lane per router group, merged into the B5 sub-batch PRs? | yes; the routers B5d rewrites are typed in W5d |
| OQ-B5-1..3, D-35 | public replacement bars, the 1M run, GPU 2 rows; `/detect` upscaling after the recall numbers | as in [07](07_b5_overview.md) section 9 |
| 07d Q1-Q9 | processing-plan behaviour | as in [07d](07d_b5_processing_plans.md) section 3 |

### 9.4 Before W6 and after

| ID | Question | Recommendation |
|---|---|---|
| Q-1, Q-2 | a Python 3.13 CI job; image lock files | one 3.13 `pytest` job; `pip freeze` records now, lock files later ([08](08_environment_and_setup.md) section 8) |
| MQ-9 | The research plans (GPU pipeline, clustering): their experiments run after the tag, as they say, except what B5 already carries? | yes |
| MQ-10 | Move #42 to v0.8.0 if the 12 GB run has not happened by the tag (D-5)? | yes |
| D-16 | the tag | on your word after R-final passes |

## 10. After the tag

The research plans ([research_gpu_pipeline_plan.md](../research_gpu_pipeline_plan.md),
[research_clustering_plan.md](../research_clustering_plan.md)) and v0.8.0
([v080_platform_breadth_plan.md](../v080_platform_breadth_plan.md)) follow. Before their waves
start, re-base the v0.8.0 designs that assumed Redis, OpenSearch leases or an OpenSearch auth index
(F3, F5, F7, F9) on PostgreSQL and Valkey. The program folder is deleted after the tag
([00_index.md](00_index.md) section 8.2).

## 11. Fold-in record

Every decided item moved into a lane, with the place its decision is recorded, is listed in
[00b_traceability.md](00b_traceability.md) section 3.
