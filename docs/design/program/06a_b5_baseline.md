# B5a: the public baseline, the Triton sweeps and the `/detect` recall question

| Field | Value |
|---|---|
| Status | plan, not started |
| Release | v0.7.0 (inside the release; R-final runs after B5; see `06_b5_overview.md` section 2) |
| Depends on | B4 merged (the final stack); OQ-B5-1 and OQ-B5-2 answered; the B5 GPU window (`00a_decision_register.md` section 8.1) |
| Blocks | 5b, 5c, 5d: no optimization code merges before 5a's reference exists |
| Decisions applied | owner rule "#45 baseline before any optimization code"; public data only; short tests when the gain is known |
| Open decisions | **D-35** (GPU-6 `/detect` upscaling), decided on this file's recall numbers |
| Source IDs | former master plan (rev 4, in git history) 7.5 row 5a, 8 (D-35), 9.1 (N13), 10 (#45, #268), 12 (T5); #45, #268; GPU-6 (`codebase_audit_plan.md` 4.3); `docs/research/storage_sizing_and_ingest_baselines.md` 6 |
| Verified | Verified against `fix/411-r2-round` at `af62c947` on 2026-11-03 (read-only `rg`, `gh issue view`; nothing run) |
| Consolidated | Consolidated 2026-11-03 |

## 1. Goal

One reproducible public baseline on the final stack, owned by 5a, that every later gate compares
to; the open #268 sweeps on it; and the recall numbers the owner needs for D-35. 5a changes a
shipped default only when a sweep's gain beats 2x the run-to-run spread and parity passes.

## 2. Preconditions

| # | Check |
|---|---|
| P1 | the final B4 stack is on main; record its SHA and image digests (`docker image inspect`) |
| P2 | OQ-B5-1 (public replacement for private-set bars) and OQ-B5-2 (1M run) answered |
| P3 | GPU 0 free of foreign processes or the peer session agrees; heavy runs announced; one heavy run at a time |
| P4 | public data fetched and verified against the pinned manifests (`--verify-only`) |

## 3. Branch, worktrees, tracks

Integration branch `int/b5a-baseline` from main. Tracks, each `b5a/<track>` in `<repo>-wt-b5a-<track>`:

| Order | Track | Kind | Scope | Size |
|---|---|---|---|---|
| 1 | `ref` | runs (coordinator or gate agent, GPU 0) | headline rows and the parity reference `ref-b5a` | M (GPU half day) |
| 2 | `ladders` | code + runs | #45: Open Images category subsets, full COCO, 1M extrapolation, protocol text fix | M |
| 3 | `sweeps` | runs + config only if a gate passes | #268: set B one and two cards, profile sweeps, two-card factors, `gpu_image_prep` instances and 2-instance parity, detector CUDA graphs, `priority_levels` | L (GPU days, tiered) |
| 4 | `recall` | small bench code + runs | GPU-6: COCO val2017 recall with `scaleup` on and off on `/detect` | S |

`ref` must finish first; the rest compare to it. `ladders` code (A1) and `recall` code (R1) can be
written while `ref` runs; their GPU runs wait for it.

## 4. Datasets (public only; sizes)

| Set | Images | Size | Licence | Pin |
|---|---|---|---|---|
| A | 2,000 COCO val/train 2017 | about 333 MB | CC BY 4.0 annotations; per-image Flickr licences recorded in `<manifest>.licenses.csv` | `scripts/datasets/manifests/coco_bench_2000.json` (seed 20261009) |
| B | 500 Wikimedia Commons photos, 12-24 MP | about 5.2 GB | CC0, PD, CC BY, CC BY-SA per image | `commons_hires_500.json` (seed 20261010); 2,000-photo extension per OQ-B5-1 |
| COCO val2017 | 5,000 | about 1 GB | as set A | official COCO download |
| COCO full | train2017 118,287 + val2017 5,000 = 123,287 (with test2017 40,670: about 164k) | about 19 GB (about 25 GB with test) | as set A | new manifest (A2) |
| Open Images v7 categories | 1k / 10k / 100k each of vehicles, people, animals | record at fetch (images are at most 1,024 px, about 0.7 MP) | images CC BY 2.0; annotations CC BY 4.0 | new manifests (A1) |
| Open Images v7 100k | 100,000 | record | as above | `openimages_bench_100000.sha256` (exists) |

## 5. Tasks

### Track `ref`

**5a-R0 Baseline headline rows (no code).**
- Stack per `06_b5_overview.md` M-2; protocol M-3..M-10.
- Rows (each 1 warm-up + 3 counted repetitions, median and min-max, spread):

| Row | Route and load | Profile |
|---|---|---|
| A-upload | set A, `POST .../ingest/upload`, 16 clients | default load list |
| A-batch | set A, `.../ingest/batch`, 32 images per request, 16 clients | default |
| B-4 / B-8 | set B, `.../ingest/batch`, 8 per request, `run_baseline.py --clients 4` and `8` | default, and `OP_LOAD_PROFILE=ingest` with 2 `gpu_image_prep` instances |
| A-detect-only | set A, policy `lazy` | default |
| endpoints | `/detect`, `/faces/recognize`, `/analyze`, `/ocr`, `/embed/image` on set A at 1 and 16 clients | default |
| perf_analyzer | detector and PE engines at batch 1, 8, 32, 64 | default |

- Comparison points (read, do not re-measure): 0.6.0 set A one card 28.81-29.07 img/s, two cards
  55.08; set B 35.0 img/s; 0.7.0 stage profile set B 26.64 at 4 batches (shared host, noisy). Any
  row more than 2x spread below its comparison point is flagged and its binding stage named.
- Output: `docs/PERFORMANCE.md` section "0.8.0 baseline (5a)"; raw JSON per M-8.

**5a-R1 Parity reference `ref-b5a`.**
- `scripts/bench/parity.py capture --tag ref-b5a` on set A and on set B with the MobileCLIP, face
  and OCR probes (`--probe-images 200`) and, if a VLM is configured, `--vlm-crops`;
  `gpu_prep_parity.py capture` on set B for the byte-identity gates of 5b.
- Done when: captures exist under `artifacts_local/bench/parity/ref-b5a/` and the noise floor is
  recorded (a second capture of the same stack compared to the first).

### Track `ladders` (#45)

**5a-A1 Open Images category subsets (code).** Size S.
- File: `scripts/datasets/fetch_openimages_bench.py` (flags today: `--out --manifest --n --seed
  --workers --cache-dir --verify-only`; no category option).
- Steps: add `--category vehicles|people|animals` resolving class names to MIDs through the
  official class-descriptions CSV (by name, never a raw index), selecting by the official
  image-level or box labels; seeded; manifest with count, bytes and sha256; per-image licence
  recorded (the existing pattern for the 100k pin).
- Red: `tests/datasets/test_fetch_openimages_category.py` on a tiny fixture CSV: the category
  filter returns only images carrying a MID of that category, deterministic for a seed; red today
  (no option). Control: without `--category` the selection equals the existing seeded selection.

**5a-A2 Full COCO manifest (code).** Size S.
- File: `scripts/datasets/fetch_coco_subset.py` (flags include `--bench-set N`); add a full mode
  (`--bench-set all`, train2017 + val2017, optional `--with-test`), manifest pinned.
- Red: manifest of the full mode on the annotation fixture lists every image id once; control:
  `--bench-set 2000` reproduces `coco_bench_2000.json`'s sha256.

**5a-A3 Ladder runs.**
- `scripts/bench/scale_run.py` (bulk ingest, one forced cancel at 50 %, resume, 15 s polls, 30 s
  resource samples) for: each Open Images category at 1k, 10k, 100k; full COCO (detect + embed +
  store + cluster, no segmentation, owner decision 2026-10-03).
- Pass (#45 acceptance, public): every row reproducible from a pinned manifest; full COCO within
  10 % of the 5a-R0 A-batch throughput or the gap explained; zero failed images; the stop/resume
  check passes; throughput per quarter, index bytes per image and per crop, and cluster-pass time
  recorded (0.6.0 comparison: 95.2 KB per input image, 19.4 KB per crop).
- 1M: extrapolate from the measured per-image cost and index growth first; a 1M run only on the
  owner's word (OQ-B5-2).

**5a-A4 Protocol text.** `docs/PERFORMANCE.md` "Baseline protocol" still calls set B private; make
it the public Commons set and point at `06_b5_overview.md` M-1..M-10.

**5a-A5 OpenSearch merge counters per ladder leg (XS).**
- Files: `scripts/bench/scale_run.py` (the resource sampler), tests beside it in `tests/`.
- Pattern: read the OpenSearch nodes stats API (`GET _nodes/stats/indices`: `merges`, `refresh`,
  `flush` totals; OpenSearch docs "Nodes stats") before and after each ladder leg, written next to
  the `host_pressure.py` IO samples. Never force-merge during a leg. If a leg shows a merge-driven
  throughput dip, run one extra arm with `POST <index>/_forcemerge` (OpenSearch docs "Force merge")
  between legs only.
- Red: a unit test with a stubbed stats response asserts the per-leg delta (merge count, merge
  time, flush total) appears in the leg record; fails today (no such field). Control: legs without
  the stats endpoint record "unavailable", never zero.
- Done when: every 5a-A3 leg record carries the three deltas; a dip, if any, is explained or its
  forced-merge arm is recorded.
- Origin: `ingest_write_path_investigation.md` section 7 items 2 and 4 (moved to
  `docs/research/`). Size XS.

### Track `sweeps` (#268)

Rules: one factor at a time, tiered (M-6), each point against its own control, 3 repetitions only
for the candidate and control. Config changes merge only if the gain beats 2x the spread and
parity passes (byte-identity for config-only changes where outputs must not move).

| ID | Sweep | Points | Pass to change a default |
|---|---|---|---|
| 5a-S1 | set B one card and two cards | ingest profile and default list; 4 and 8 batches | recorded (no default) |
| 5a-S2 | `minimal` and `standard` profile sweeps (`config_templates/profiles/*.json`) | instance counts per profile | gain > 2x spread, memory fits the profile's card |
| 5a-S3 | two-card factors: instance counts, `max_queue_delay_microseconds`, API worker count | one factor at a time, then one confirmation | beat 55.08 img/s (0.6.0 two-card) by > 2x spread |
| 5a-S4 | `gpu_image_prep` instance count and nvJPEG throughput at 20 MP (`op_gpu_prep_stage_seconds` decode stage) | 1, 2, 3 per profile | and the **2-instance output parity** check left open on #268: `gpu_prep_parity.py` byte-identical, 1 vs 2 instances |
| 5a-S5 | detector CUDA graphs: `optimization { cuda { graphs: true } }` (TensorRT backend only, `model_config.proto`) | batch 1, 8, 32, 64 with `graph_spec` for the profile batches | perf_analyzer gain > 2x spread, p99 not worse, boxes byte-identical |
| 5a-S6 | `priority_levels` / `default_priority_level` in the dynamic batcher (Triton batcher docs) for interactive versus bulk callers | interactive `/detect` p99 under bulk ingest load | interactive p99 better, bulk throughput not lower than 2x spread |
| 5a-S7 | RTX 3080 Ti rows | only with the GPU 1 owner's consent; otherwise left to #42 | recorded |

**5a-S8 Triton warmup and memory pools (S).**
- Files: `models/*/config.pbtxt` (detector, PE and OCR models), the Triton launch arguments in
  `docker-compose.yml` / `docker-compose.dev.yml` and `env.template`, with `tests/test_env_surface.py`
  in the same commit if a setting is added.
- Pattern (from `triton_pipeline_optimization_plan.md` Wave 4; not covered by S1-S7): `model_warmup`
  at batch 1 and at the profile's largest batch (Triton model configuration docs, "Model Warmup":
  READY is withheld until warmup finishes), and explicit `--cuda-memory-pool-byte-size` and
  `--pinned-memory-pool-byte-size` sized from the measured use (`tritonserver --help`).
- Measure (tier 1, then tier 2 per M-6): time from start to READY, first-request latency after a
  model load, p99 at 32 clients, GPU memory per card. Control: the same stack without warmup and
  with default pools. Red: a config-parse test that every changed `config.pbtxt` still loads and
  has the warmup block (fails before the edit).
- Pass: the 5a rule (gain beyond 2x the spread, outputs byte-identical); otherwise record and do not
  change the default. Done when: numbers recorded in `docs/PERFORMANCE.md`. Size S.

Not in 5a: engine build stamps for yolo, mobileclip, faces, ocr (moved to #415).

### Track `recall` (GPU-6, N13)

**5a-C1 Capture arm (bench code only; no product change before D-35).** Size S.
- Fact: `/detect` letterboxes with Ultralytics `LetterBox(new_shape=(size, size), auto=False,
  scaleup=False)` in `src/clients/triton_client.py` (Ultralytics validation behaviour), while
  ingest upscales (`letterbox_to_square`). `scripts/bench/detector_capture.py` today captures only
  the ingest letterbox.
- Steps: add a `detect` preprocessing arm to `detector_capture.py` that uses the `/detect`
  letterbox with `scaleup` False and True (same cv2 interpolation, so only `scaleup` differs; do
  not touch the documented PIL versus cv2 split).
- Red: a 320x240 synthetic image gives a 640-scale content box with `scaleup=True` and the
  original size with `False`; red today (no arm). Control: the existing ingest arm output unchanged.

**5a-C2 Recall run.** COCO val2017 (5,000), the B3 default detector engine, `/detect`'s default
confidence: AP, AP50, recall and precision overall and per COCO size bucket (small < 32^2, medium,
large) via `scripts/bench/detector_quality.py`, for both arms; plus latency per image at 1 and 16
clients. Raw results to `docs/benchmarks/v080/5a/gpu6/`.

**D-35 (open; present with these numbers).**

| Recommendation | Consequence | Needed by |
|---|---|---|
| keep `scaleup=False` unless `scaleup=True` raises small-object recall by more than 2x the noise floor without lowering AP; if it does, switch `/detect` to `scaleup=True` in 5b (after GPU-3b, so the change lands once) with a CHANGELOG behaviour-change line | `/detect` keeps matching Ultralytics validation by default; a switch changes boxes for every small image | end of 5a (before 5b starts) |

N13 (`question(detect): should /detect upscale small images like ingest does`) closes with the
decision and the numbers.

## 6. Gate (5a PR)

- T0 (targeted red/green plus static checks on the touched files) for A1, A2, A5, C1, S8; one
  offline sweep on the merged integration head (pre-commit `--all-files`, suite `-n 8`,
  contracts-check, docs check, CHANGELOG rules; frontend only if `frontend/` changed). The
  baseline is measured once, in the B5 GPU window (`00a_decision_register.md` section 8.1); no per-branch runs.
- Every row recorded with host load; noisy rows repeated; no default changed without its pass line.
- `docs/PERFORMANCE.md` sections written; `docs/benchmarks/v080/5a/` with `MANIFEST.sha256`.

## 7. Done when

- `ref-b5a` captures and the 5a-R0 table exist (every later gate's reference);
- #45's public rows are recorded (or explicitly deferred by the owner: 1M);
- #268's sweeps are recorded with the winning settings, and only gated defaults changed;
- D-35 is decided and N13 answered;
- CI green; the PR merged with a merge commit;
- this file is deleted in the closing PR after the keep-list below is moved (`00a_decision_register.md` "Plan
  lifecycle"), and its row removed from `00a_decision_register.md` and `docs/design/README.md`.

### What to keep after merge

| Fact | Target |
|---|---|
| the 5a-R0 headline table, ladder rows, sweep winners, GPU-6 recall table and the D-35 outcome | `docs/PERFORMANCE.md` (with `docs/benchmarks/v080/5a/MANIFEST.sha256`) |
| the public-data and protocol text (5a-A4) | `docs/PERFORMANCE.md` "Baseline protocol" |
| any changed profile default | `docs-site/docs/deployment/gpu-sizing.mdx`, CHANGELOG "Changed" |
| the method note | `docs/research/` |

## 8. Docs, CHANGELOG, rollback, upgrade impact

- Docs: `docs/PERFORMANCE.md` (0.8.0 baseline, ladders, sweeps, GPU-6), `docs/research/` note for
  the method, `docs-site/docs/deployment/gpu-sizing.mdx` only if a profile default changes.
- CHANGELOG: "Changed" for any default moved by a sweep; "Added" for the dataset options.
- Rollback: config changes revert alone (reload the model); no data changes.
- Upgrade impact: a profile default change is written by the installer on upgrade; no engine
  rebuild unless CUDA graphs change the build (they do not: it is a serving option).

## 9. Risks

| Risk | Mitigation |
|---|---|
| shared host makes 3-repetition medians wide | M-3; repeat only the disputed point; report the spread honestly |
| Commons rate limits (HTTP 429 seen on #268) | fetch slowly with the script's workers lowered; pin what was fetched; record shortfalls |
| a 100k or full-COCO run occupies the GPU for hours | announce; run at agreed times; stop/resume is part of the check |
| CUDA graphs with dynamic shapes capture only listed shapes | `graph_spec` per profile batch; unlisted shapes run uncaptured (documented in `model_config.proto`) |

## 10. Lane briefs (ready to paste)

Common block: as in `04_b3_ultralytics_one_version.md` section 17, with worktree
`<repo>-wt-b5a-<track>`, branch `b5a/<track>`, file `docs/design/program/06a_b5_baseline.md`, and
these additions: "Benchmarks follow `docs/design/program/06_b5_overview.md` M-1..M-10. Stack:
`docker compose -p op-b5a-bench` only, GPU 0 only (GPU 2 only if the coordinator says the peer
agreed). Announce heavy runs through the coordinator. Project slugs start with `b5a-`; never touch
`baseline-*` projects. Public data only."

- **`ref`**: "Run 5a-R0 and 5a-R1. No code. Report every row: median, min-max, spread, host load
  start/end, foreign GPU load, binding stage."
- **`ladders`**: "Do 5a-A1 and 5a-A2 (red tests first), then 5a-A3 runs in the order the
  coordinator schedules, then 5a-A4 and 5a-A5. Do not start a 1M run. Final step: delete this sub-batch's plan files only in
  the closing PR, per `00a_decision_register.md` Plan lifecycle."
- **`sweeps`**: "Run 5a-S1..S6 and S8 tiered (M-6); propose config changes only with their pass line met;
  each change is one commit with its numbers in the body. 5a-S7 only with written consent from the
  GPU 1 owner session."
- **`recall`**: "Do 5a-C1 (red test first), then 5a-C2. Do not change `src/clients/triton_client.py`.
  Report the D-35 table."

## Research amendments (2026-11-04, the research integration pass)

| Where | Change |
|---|---|
| 5a-R0 | read the `gpu:PR-0` and `gpu:PR-1` records first (honest GPU number, decoder matrix); the 5a rows carry SM active and tensor active from DCGM |
| 5a-S4 | the nvJPEG check at 20 MP is `gpu:PR-1`'s set-B row when the prototype window ran; otherwise run as written |
| 5a sweeps | the Triton router of B4 4B-6 and `serving_reports` are in place; a two-server row is research wave RW-1 (`gpu:PR-7`), not 5a |
