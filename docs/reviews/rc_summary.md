# Release candidate 0.7.0: summary of the combined branch

Branch `release/0.7.0-rc`, built on `origin/main` at `0bb5fed7` (merge of #394) by merging five wave
branches with `--no-ff`, in the order below, followed by the reconciliation commits. Written 2026-10-30.
Per-wave detail is in the review documents named in each row; this page only lists what each wave
delivered, what was checked, and what was not.

## Waves

| # | Branch | Commits (own, base to tip) | Delivers | Review |
|---|---|---|---|---|
| 1 | `perf/v070-cluster-pass` | 5, `1df27b53..2f9b31d6` | IVF fit with one k-means restart behind an agreement gate (#337); a pass that did not cover its pool is `partial` (#357); the cluster ladder times geometry, assign-only and reset | [`sp1_review.md`](sp1_review.md) |
| 2 | `perf/v070-ingest-speed` | 14, `1df27b53..db94c872` | Stage counters that work on Triton 2.70 (#390); device-memory budget in `gpu_image_prep` (#392); opt-in ingest load profile and on-demand model loading (#393); `/health` readiness (#391); `full` profile runs one instance; `run_baseline.py --clients` | [`sp0_review.md`](sp0_review.md), [`sp_fixes_review.md`](sp_fixes_review.md) |
| 3 | `feat/v070-assist-loop` | 23, `1df27b53..70af3457` | Auto-label loop (#286): routes, worker, audit, rollback, retrain hold, `/clusters` panel, fake-VLM modes and a live test | [`../design/auto_label_loop_plan.md`](../design/auto_label_loop_plan.md) |
| 4 | `chore/v070-acceptance-phases` | 2, `0bb5fed7..9debb1d8` | Acceptance phases for health, clusters, label operations, GPU route and loop | [`../design/release_acceptance_plan.md`](../design/release_acceptance_plan.md) |
| 5 | `docs/v070-user-docs` | 3, `0bb5fed7..0f6b0ab2` | README "What is new", guides for label operations and cluster filters, stage timing page, improvement backlog | [`../design/improvement_backlog.md`](../design/improvement_backlog.md) |

Conflicts were limited to `CHANGELOG.md` (kept both sides each time). Contracts needed no regeneration:
`contracts/` from the loop branch already matches the merged code (`make contracts-check` passes).

## What was verified, and where

| Wave | Verified | Evidence |
|---|---|---|
| Cluster pass | Fit time and cluster quality on real vectors (91,184 crops; public COCO, 7 seeds); timed pass at 86k, 100k and 250k rows, 3 reps, under host load | `sp1_review.md`, measurement log section C and the SP1 rows |
| Ingest speed | Stage counters on Triton 2.70 (toy-model reproduction of the three faults); fail-first unit tests; live parity of `gpu_image_prep` old against new on 200 public images; throughput rows for the budget, the load profile and the 24 GiB card | `sp_fixes_review.md`, measurement log SPF.1 to SPF.4 |
| Auto-label loop | Engine, store, budgets, audit, rollback and routes by unit and route tests; cross-project isolation tests; one live test against the harness with the fake VLM | `tests/curation/test_assist_loop_*`, `tests/live/test_live_assist_loop.py` |
| Acceptance phases | Phase scripts against a stubbed API in `tests/release/`; the `loop` phase routes and request body rechecked against the merged router and schemas (`GET` and `POST /assist/loops`, `loops` array, `loop_id` null on a dry run): they match | `scripts/release/acceptance_phases_review.sh` |
| User docs | `scripts/docs/check_docs_vs_code.py` and the docs-site build | see the gate results below |

## What was NOT verified live

- The loop with a real VLM on a real project: only the fake VLM of the harness ran. Quality of the labels it propagates is unmeasured.
- The loop panel against a real backend in a browser: the frontend tests use the stub server.
- The `loop` and other new acceptance phases against a running stack: the release run does this.
- The live write-path suite on the merged branch (#397 is about its failures on `main`).
- Ingest throughput on the merged branch: all numbers were measured on the wave branch, one RTX A6000, private photo set, host load 13 to 35 or higher (shared host). The public set A and B rows were not re-run for the memory budget.
- CUDA MPS (not run), batched kernels (C2, deferred: the bar is met), two GPUs for one model, a 24 GiB card measured by emulation (ballast), not a real 24 GiB card.
- The 2,042 s pass at 491k rows under sustained ingest (needs the GPU ingest stack); 100k and 250k pass timings use synthetic vectors (time and memory scaling only).
- Geometry stage speed-up: measured as the largest stage, not changed.

## Headline numbers

| Item | Number | Where |
|---|---|---|
| IVF fit, one restart against five | 50 s against 388 s on 91,184 real crops (about 8 times); purity 0.7962 against 0.7963 | CHANGELOG, `sp1_review.md` |
| Whole cluster pass | 1.7 to 2.4 ms per crop (0.6.0: 4.2 to 7.6); bar 2.5 | `sp1_review.md` |
| Geometry stage | 70 to 85 s at 86 to 100k rows, now the largest stage | `sp1_review.md` |
| Ingest, load profile, 2 instances, 4 batches | 48.3 img/s, GPU 94 percent, 0 fallbacks | measurement log SPF.4 |
| Ingest, default list, 1 instance | 38.9 img/s, GPU 83 percent, 0 fallbacks | SPF.4 |
| Ingest, default list, 2 instances | 35.9 img/s, 0.6 to 5.6 percent CPU fallbacks, card full (47.4 GiB) | SPF.4 |
| Ingest, 24 GiB card emulation, profile, 1 instance | 38.1 img/s | SPF.4 |
| Models loaded by default that ingest does not use | 27.7 GB (face, MobileCLIP, OCR) | `ingest-load-profile.mdx` |
| Stage counters overhead | none measurable (ratio 0.977 to 0.978) | measurement log SP0.3 |

## Owner decisions applied

- CQ3 defaults: the recommendations D-CQ3-1 to D-CQ3-8 of [`../design/cq3_label_operations.md`](../design/cq3_label_operations.md) section 12 (undo restores per item and keeps later edits; bulk undo never touches the holdout; undo is a dry run by default; synchronous undo; ops as files, newest 200 per project).
- `full` ships one `gpu_image_prep` instance (38.9 img/s, bar met); two instances stay the documented option with the ingest profile.
- On-demand model loading stays in the API clients (`src/clients/model_residency.py`), not in Triton.
- Research parked: Leiden and GPU clustering, average linkage, detector-class roots (#342 to #346 and the outcome note); IVF and PE-Core-L stay ([`../research/cluster_program_outcome.md`](../research/cluster_program_outcome.md)).
- Deferred: batched CUDA kernels (C2, #392 and the staged ideas in #407), MPS.

## Gate results on this branch

Run once on the merged branch, with the main checkout's virtualenv, no GPU, on a shared host (load 7 to 120).

| Gate | Result |
|---|---|
| `pre-commit run --all-files` | 35 hooks passed, 0 failed (the rest skipped: no matching files) |
| `scripts/docs/check_docs_vs_code.py` | 0 problems in all ten categories |
| `make contracts-check` | in sync |
| docs-site build | success |
| frontend `check`, `lint`, `knip`, `vitest`, `build` | 0 errors; 478 files, 4,998 tests passed; build ok |
| offline pytest (`-m 'not live'`) | 10,717 passed, 26 skipped, 104 deselected, 10 failed (75 minutes); the 10 failures had one cause and are fixed (below) |
| stubbed e2e | 307 passed, 12 failed under host load (45 s click timeouts); the four files re-run on their own: all pass |

The 10 pytest failures were an integration defect: `docker-compose.ingest-focus.yml` (wave 2) was never committed
because `.gitignore` excludes `docker-compose.*.yml`. The compose, overlay-render and installer manifest tests
failed on the missing file. The file is now tracked and un-ignored; `tests/compose` and `tests/installer/test_static.py`
pass (100 passed).
