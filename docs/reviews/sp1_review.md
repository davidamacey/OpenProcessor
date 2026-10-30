# SP1: cluster pass (#337) and #357 close-out

Branch `perf/v070-cluster-pass`, merged with `origin/main` after PR #389. CPU only, no GPU.
Plan: ship-track plan section 5 (SP1) and section 3. Tier note: the fit's numerics changed, so this
file is the place for the written verdict of the reviewer the plan names.

## What changed

| Commit | Subject |
|---|---|
| `f364ad4a` | fix(clustering): an assign-only pass that did not cover its pool is partial, not success (#357) |
| `08a83143` | perf(clustering): one k-means restart behind an agreement gate (#337); fixes the `IVF_MAX_TRAIN_SAMPLE` comment |
| next | perf(bench): geometry, assign-only and reset timed in `cluster_ladder.py` |
| this commit | docs: this review, changelog |

- `tests/curation/test_cluster_pass_partial_daemon.py` (4 tests, seen failing: `KeyError expected_pool`;
  each of the two checks was disabled in turn and its test failed). The assign-only pass now counts its
  pool at the start (a failed count raises), compares what it read with `POOL_COVERAGE_MIN` (0.99),
  reports `partial` with reason `short_read` or `write_failures`, and records no last run.
  This pass streams writes as it reads, so a partial result means the writes that happened stay; the
  next pass finishes the rest. The full pass still stops before the fit.
- `tests/curation/test_ivf_restarts.py` (2 tests; with `IVF_DEFAULT_NREDO = 5` both fail).
- `scripts/bench/ivf_nredo_agreement.py` and `tests/test_ivf_nredo_agreement.py` (7 tests; relaxing the
  purity bound made one fail).

## The agreement gate (nredo 5 to 1)

Production fit (spherical FAISS k-means, 512 centroids, 50 iterations, 50,000-vector sample, nearest
centroid assignment of the whole pool) on real vectors. Raw JSON:
`artifacts_local/bench/v070/sp1/agreement/` (main checkout, gitignored). Reproduce:
`scripts/bench/ivf_nredo_agreement.py --vectors V.npy --meta M.jsonl --labels L.jsonl --candidates 1,2,3 --seeds 7,13,21,5,9,11`.

Gate (my reading of the plan's wording, which gives no pair-agreement number): purity not lower by more
than 0.5 point, false-merge rate not higher by more than 0.5 point, ARI with the incumbent not below
the seed-to-seed noise floor by more than 0.02. Evaluated on the mean over seeds, because two seeds of
the same nredo=5 fit already differ by 2 to 3 points of purity and false-merge rate at k=512; a per-seed
paired check fails on noise in both directions (nredo=1 failed 2 of 7 seeds, nredo=2 failed 2 of 7 on a
different pair). If the reviewer wants a stricter bar, say so.

| Data | nredo | seed-mean purity | seed-mean false-merge | mean cosine to centroid | ARI vs incumbent | fit s (load about 100) | gate |
|---|---|---|---|---|---|---|---|
| Public COCO, 3,961 crops, 2,950 labelled, 7 seeds | 5 | 0.7963 | 0.3009 | 0.88628 | 1 (seed 42) | 8.1 | incumbent |
| same | 1 | 0.7962 | 0.3008 | 0.88578 | 0.331 (floor 0.311 to 0.344) | 2.1 | pass |
| same | 2 | 0.7964 | 0.3065 | 0.88592 | 0.330 | 3.0 | fail (+0.56 point false merge) |
| same | 3 | 0.7954 | 0.3081 | 0.88596 | 0.333 | 6.9 | fail (+0.72 point) |
| Set P, 91,184 crops, 3 seeds (numbers only, no labels) | 5 | n/a | n/a | 0.90719 | 1 | 388 | incumbent |
| same | 1 | n/a | n/a | 0.90693 | 0.396 (floor 0.393) | 50 | pass |

Reading: restarts do not improve the partition more than a change of seed does (ARI between nredo=5 and
nredo=1 equals the ARI between two nredo=5 seeds), and the compactness gain is 0.0005 on COCO and 0.0003
on set P. nredo=2 and 3 failing while 1 passes is noise of the same size as the gate, which is why the
gate is on the means and why nothing is claimed beyond "one restart is not worse". Set P has no ground
truth in the export, so its row checks agreement and compactness only; the label gate rests on COCO
(3,961 crops is small for 512 clusters, about 8 per cluster). A larger labelled public set would be the
stronger evidence; not done here.

## Timed pass at the ladder (`cluster_ladder.py --with-geometry --assign-only --topup 5000 --reset`)

Production `cluster_residuals` (nredo=1), then `cluster_geometry_stage`, then 5,000 more rows and
`assign_only_residuals` over the whole pool (it reads every row, not the top-up; unchanged rows are not
rewritten). Own throw-away OpenSearch (16 GB heap) on port 5617, fresh project per repetition, 3 reps,
median (range). **Every run is noisy by the protocol (1-minute load 24 to 107, other sessions share the
host); the numbers are upper bounds and are not a quiet-host result.** Raw JSON and per-rep load:
`artifacts_local/bench/v070/sp1/ladder/`.

| Rows | Data | pass s | fetch / fit / write-back s | geometry s | assign-only s | whole pass ms per crop |
|---|---|---|---|---|---|---|
| 86,000 | set P real (numbers only) | 88.8 (83.0 to 114.4) | 21.2 / 42.6 / 25.9 | 74.6 (70.2 to 85.1) | 25.2 (22.0 to 28.5) | 2.35 (2.09 to 2.44) |
| 100,000 | synthetic (time only) | 76.2 (75.6 to 77.3) | 18.6 / 24.9 / 32.2 | 73.9 (70.7 to 86.8) | 22.9 (22.4 to 23.4) | 1.72 (1.70 to 1.87) |
| 250,000 | synthetic (time only) | 149.1 (137.1 to 330.8) | 52.0 / 17.2 / 88.3 | 192.5 (173.6 to 357.7) | 62.3 (54.4 to 110.5) | 1.58 (1.49 to 3.20) |

Peak resident set 2.8 GB (86k), 3.5 GB (100k), 6.3 GB (250k).

Bars (plan SP1, at most 2.5 ms per crop for the whole pass including geometry and assign-only at 100k):
met on all three rows at the median even under load (the worst 250k repetition, 3.2, ran at load 107).
Set P is the high-resolution class; its 2.35 is the nearest to the bar. The 0.6.0 figure (4.2 to 7.6
ms/crop) is down to 1.7 to 2.4 here, of which the fit change is about 0.5 ms/crop at 86k (fit 182 s to
about 43 s on the same size).
Stage that is now largest: **geometry** (70 to 85 s at 86 to 100k, linear: it reads every clustered
vector again and rewrites every item, 0.8 ms/crop). It is the next target if the bar tightens; not
changed here (output-identical changes only, and it was untimed before this).

Not done: the "2,042 s at 491k" explanation under sustained ingest with merge pressure (needs the GPU
ingest stack; only the stage costs in isolation are measured: all linear except the fit, which the
50,000 cap flattens). Public Open Images 100k vectors were not available, so 100k and 250k are
synthetic (time and memory scaling only, as the brief allows); no quality claim rests on them.

## #357

1. Partial pass: the full pass reported `partial` already (a9c75abf); this branch adds the assign-only
   pass the refresh daemon runs. `auto_label_run` already turns a `partial` stage into a failed job.
2. Whole-index reset: already a polled task with a post-check (a9c75abf). Measured here at 250,000
   rows, 3 reps: 592 to 799 s (median 707), completed, no error. It is slow because each document is
   rewritten with its vector (graph re-insert); the limit is documented in the ladder docstring, and the
   ladder avoids the reset by using a fresh project per repetition.

Can close: #357. #337: acceptance items met except the sustained-ingest explanation (see above); close
or split that remainder at the owner's call.

## Verdict requested

Opus review of the fit change (gate definition above, small labelled set) and of the assign-only
write-back change.
