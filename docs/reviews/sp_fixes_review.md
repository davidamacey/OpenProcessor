# SPF: ingest-speed fixes from the SP0.7 measurements (#392, #393)

Branch `perf/v070-ingest-speed`, merged with `origin/main` (PR #389 included). Evidence: log sections SP0.7 and
SPF in [`../research/v070_measurement_log.md`](../research/v070_measurement_log.md), design
`docs/design/sp_gpu_prep_concurrency.md`. **Status: code, tests and the GPU 0
acceptance run are done; nothing is pushed.**

## What changed

| Commit | Subject |
|---|---|
| `a9727b06` | perf(gpu-prep): wait for device memory instead of failing into the CPU fallback (C1) |
| `6807755a` | feat(triton): opt-in ingest load profile with on-demand model loading |
| `efe067ca` | fix(triton): load the OCR detector and recognizer before ocr_pipeline on demand |
| `eb75fe37` | perf(gpu-prep): 16 workers and 32 in flight by default |
| this commit | docs: SPF results, review, changelog, raw rows |

- **C1.** `prep_budget.py`: a FIFO byte budget (`DEVICE_BUDGET_MIB` 2048, `BUDGET_WAIT_SECONDS` 20). A photo reserves its decoded frame
  (twice that when EXIF rotates it) before decoding. Frames and orientation copies live on one allocation stream so blocks are reusable
  across workers; every path that drops a frame has waited for its worker stream. New statuses `gpu_memory` (allocation failed, wait timed
  out) and `over_budget` (photo larger than the budget) are answered, not raised; the API already counts any status as `cpu_<status>`.
  A `budget` stage counter shows the wait. The budget is reserved at load; a card that cannot hold it fails the load naming
  `DEVICE_BUDGET_MIB`. `DEVICE_BUDGET_MIB` above `POOL_MIB` is refused.
- **Config.** `WORKERS` 32 to 16, `MAX_IN_FLIGHT` 96 to 32.
- **Load profile (opt in).** `OP_LOAD_PROFILE=ingest` plus `docker-compose.ingest-focus.yml` (the CLI adds it): Triton preloads the
  detector and the PE encoders; `src/clients/model_residency.py` loads any other model on first use at the pool, the sync client, the
  face client and the embedding service, with `ocr_pipeline`'s two OCR models first. Default `full` is untouched.
- **Tool.** `scripts/bench/gpu_prep_parity.py` (capture and compare model answers).

## Measurement results

See log section SPF. Headlines (set P, 3 reps, A6000, host load 13 to 35 in the acceptance rows):

| Stack | 4 clients | 8 clients | Fallbacks | API CPU-s/img |
|---|---|---|---|---|
| ingest profile, 2 instances | 48.3 img/s, GPU 94 % | 48.3 | 0 | 0.052 / 0.056 |
| default list, 1 instance | 38.9, GPU 83 % | 38.6 | 0 | 0.060 |
| default list, 2 instances (what `full` ships) | 35.9, GPU 72 % | n/a (23.5 at 8 with the old config) | 0.6 to 23 % | 0.081 |
| 24 GiB emulation, profile, 1 instance | 38.1, GPU 84 % | | 0 | 0.059 |

Per-model memory (answers #393): scrfd 6,846 MiB, arcface 2,754, mobileclip image 3,454, mobileclip text 1,430, OCR detector 1,642, OCR
recognizer 11,572, ocr_pipeline 0; total 27,698 MiB. Every public endpoint answered the same on the profile and on the default list
(38 routes); the one difference is that `/ocr/predict` fails with a CUDA out of memory on the default list with two instances.

## How it was tested

- Fail-first: 18 of the 27 stage-metrics tests failed against the previous model (new statuses, budget, load refusal); the budget unit tests
  failed on import, and with the FIFO check or the accounting removed 1 and 5 of 16 failed. The residency wiring tests failed with each
  `ensure` call removed (3 of 13). The CLI overlay test failed without the CLI change. The API route-count tests for the two new statuses
  pass without an API change (the counter is generic); they are regression guards, not fail-first.
- Live parity of the model on 200 public images, old against new: identical wherever the detector boxes are identical (log SPF.1).
- Offline suites run: tests/models, tests/test_model_residency.py, tests/compose, the CLI compose tests, env and docs checks.

## Risks and what was not done

- **The shipped `full` profile still fills a 48 GB card** (default list, two instances: 47.4 GiB, up to 23 % fallbacks, 5.6 % in the
  last run). C1 makes the failures counted but cannot make 36.5 GB + 2 instances fit. The reservation covers frames only; nvJPEG's own
  buffers grow outside it, so both instances load and the failure comes later as `cpu_gpu_memory`. Not changed: a decision for the owner.
- WORKERS 12 and 24 could not be ranked: the host load swamped the differences (see SPF.2).
- C2 (batched kernels) deferred: the bar is met without it.
- CUDA MPS not tried. The set B public rows in PERFORMANCE were not re-run.
- A first call under the profile pays the model load once (2 to 18 s); a load can fail if the card has filled meanwhile.
- `GET /models/` lists never-loaded models as `UNKNOWN` on the profile.

## Decisions needed from the owner

1. Decided by the main session after this review: `full` ships one `gpu_image_prep` instance (38.9 img/s, bar met); two instances stay the documented option with the ingest profile (48.3 img/s). On-demand loading stays in the API clients.
2. Push and PR once the full suite has been run.

## Five-minute check

```bash
cd "$WT"
$VPY -m pytest tests/models tests/test_model_residency.py tests/compose/test_ingest_focus_overlay.py -q --no-cov
git show a9727b06 --stat && sed -n 1,60p models/gpu_image_prep/1/prep_budget.py
```
