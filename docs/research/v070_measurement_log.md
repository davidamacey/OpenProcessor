# v0.7.0 measurement log: cluster pass size ladder

Measured 2026-10-28 (one session, 06:30 to 10:30 local). Groundwork for #337 (super-linear cluster
pass) and the scale rows of #45. Every number below is a median of 3 repetitions unless a range is
shown; raw per-repetition rows are in the JSON files the tools write (kept outside the repository).
Only the IVF cluster pass (`cluster_residuals`, method `ivf`) was measured. The geometry stage, AHC
refinement, UMAP and the refresh daemon's assign-only pass were not.

## Setup

| Item | Value |
|---|---|
| Host | 2 x Intel Xeon E5-2680 v3 (Haswell, AVX2, no AVX-512), 48 hardware threads, 503 GB RAM |
| Storage | Samsung 970 EVO Plus 2 TB NVMe (OpenSearch data and the project directories) |
| Not quiet | The host also ran other users' jobs during the session: 1-minute load average 8 to 16 at the start of most passes (the bench's own FAISS threads included), 24 to 33 in a few. Spread between repetitions is reported so this is visible. |
| OpenSearch | 3.6.0, one node, 30 GB heap, Lucene HNSW (`pe_embedding`, 1,024 dimensions, cosine, `m` 16, `ef_construction` 128), the product's real items mapping created through the project factory |
| Cluster pass runs in | the bench process on the host (CPU FAISS 1.15.1, Python 3.12.3, NumPy 2.5.2), against the stack's OpenSearch over loopback HTTP. No GPU was used by the cluster pass. |
| Method under test | `IVFMethod`: spherical k-means, 512 centroids, 50 iterations, 5 restarts, trained on at most 50,000 randomly sampled vectors (`IVF_MAX_TRAIN_SAMPLE`), then every vector assigned to its nearest centroid |
| Commit | base `9a42180c` (branch `perf/v070-cluster-and-embed-pass`); the four bench scripts added with this log are the only difference |
| Stack | a running test stack that also hosted a refresh daemon; it was not stopped and the existing projects on it were not touched |

Data:

* **Set P**: 91,184 real PE-Core-L14 crop embeddings (1,024 dimensions, L2-normalised) of private
  high-resolution photographs, stored in a product project, read out unchanged, shuffled once with a
  fixed seed so that every size is a random subsample. 37 distinct detector class names; the class
  used for the filtered k-NN probe covers about 3 percent of the rows. The photographs and their
  paths are not part of this log.
* **Synthetic (S)**: 1,000,000 vectors, seeded (`--seed 7`). 1,500 mixture components whose centroids
  are 1,500 distinct rows sampled from set P; component weights log-normal(0, 1) so the classes are
  unbalanced; each vector is its centroid plus isotropic gaussian noise of norm 0.6 (cosine to its
  centroid about 0.86), re-normalised. Sizes are nested prefixes. The filtered probe class covers
  about 0.75 percent of the rows.

In both cases the throw-away project's documents carry only the vector and a class name (the
loader of `cluster_plumbing_bench.py`), about as large as the real documents of set P: the real
project's items index was 409 MB for the same 91,184 rows, the throw-away copy 439 MB right
after the load.

## Method

* One rung = one size = one process (`cluster_ladder.py`), so the peak resident set is that size's.
* Every repetition gets a **fresh throw-away project** (created and deleted through the product
  lifecycle): bulk load (6 concurrent bulks of 1,000), refresh, then the timed pass. A fresh load has
  no cluster fields, so no reset step is needed. (An earlier version reset the cluster fields of one
  project in place with `update_by_query`; see Open issues for why that was dropped. Its 10,000 and
  50,000 rows agreed with the fresh-project rows within 3 percent: wall 22.7 s and 205.0 s against
  22.0 s and 203.2 s.)
* Wall time and the stage split come from `cluster_plumbing_bench.one_run`, which wraps the
  production functions from outside: `fetch_s` is `fetch_residual_embeddings_parallel`, `fit_s` is
  the method's `fit_predict` (train plus the full-pool assign), `write_s` is the bulk write-back of
  the assignments, `refresh_s` and `tail_s` are the index refresh and the bookkeeping after it. The
  five add up to `wall_s` within 10 ms in every row.
* A pass that did not cluster the whole pool (a refresh daemon assigning items first) would be
  discarded and repeated. None was discarded in the reported runs.
* Peak resident set: `VmHWM` of the bench process, reset (`/proc/self/clear_refs`) just before the
  pass. The input dump stays resident, so the table gives the resident set before the pass
  (`base`) and the peak; the pass's own cost is the difference.
* Index size: `_stats` store bytes of the items index, read right after the load and refresh
  (before) and right after the pass (after). Both are snapshots, not settled sizes: the first is
  taken while background merges are still running (so the small sizes read low), the second includes
  the superseded document versions of the write-back that have not been merged away yet.
* k-NN latency: 200 queries (after 20 warm-up queries), `size` 10, `k` 10, vectors of randomly chosen
  loaded rows, client-side wall time over loopback. Filtered = a `term` filter on one class inside
  the k-NN clause (Lucene efficient filter). Measured on the freshly loaded project before the pass.
* Fit isolation: `ivf_fit_scaling.py` runs the same `IVFCentroidStore.train` call on 10,000, 25,000
  and 50,000 vectors without OpenSearch, twice each, on both datasets.

## Results

### Cluster pass wall time and stage split (IVF, 3 repetitions, median; range of wall in brackets)

| Data | Crops | Wall s | Fetch s | Fit s | Write-back s | Refresh + tail s | Wall ms/crop | Fetch ms/crop | Fit ms/crop | Write ms/crop |
|---|---|---|---|---|---|---|---|---|---|---|
| P | 10,000 | 22.0 (22.0-24.9) | 2.5 | 15.5 | 4.2 | 0.1 | 2.20 | 0.248 | 1.547 | 0.421 |
| P | 25,000 | 80.6 (78.8-83.1) | 6.3 | 67.0 | 8.5 | 0.1 | 3.23 | 0.252 | 2.679 | 0.339 |
| P | 50,000 | 203.2 (201.9-205.3) | 10.2 | 177.5 | 15.4 | 0.2 | 4.06 | 0.204 | 3.549 | 0.308 |
| P | 91,184 | 231.2 (226.6-235.8) | 17.4 | 181.6 | 30.4 | 0.2 | 2.54 | 0.191 | 1.991 | 0.334 |
| S | 250,000 | 203.5 (199.5-211.3) | 40.1 | 84.6 | 79.1 | 0.2 | 0.81 | 0.160 | 0.338 | 0.316 |
| S | 500,000 | 322.6 (307.4-378.8) | 78.8 | 88.7 | 155.7 | 0.3 | 0.65 | 0.158 | 0.177 | 0.311 |
| S | 1,000,000 | 596.0 (586.2-649.4) | 167.9 | 99.9 | 336.2 | 0.8 | 0.60 | 0.168 | 0.100 | 0.336 |

The 500,000 spread (378.8 s) is one repetition at a 1-minute load average of 33 (fetch 128.9 s
against 76 to 79 s in the other two). All sizes completed; none was stopped for time (the longest
pass was 11 minutes) or disk (the whole session used 4.4 GB of result files plus the temporary
OpenSearch indexes, which peaked near 14 GB for the 1,000,000 project and were deleted after each
repetition).

### Memory and index size

| Data | Crops | Resident before pass MB | Peak resident MB | Pass overhead MB | Index after load MB | Index after pass MB | Bulk load s |
|---|---|---|---|---|---|---|---|
| P | 10,000 | 980 | 1,165 | about 190 | 54 | 40 | 4.8 |
| P | 25,000 | 1,179 | 1,640 | about 460 | 99 | 211 | 9.4 |
| P | 50,000 | 1,445 | 2,123 | about 680 | 98 | 784 | 18.3 |
| P | 91,184 | 2,084 | 3,069 | about 985 | 439 | 1,494 | 32.4 |
| S | 250,000 | 4,372 | 6,389 | about 2,020 | 2,094 | 4,536 | 89.1 |
| S | 500,000 | 7,376 | 11,284 | about 3,910 | 2,783 | 7,826 | 199.5 |
| S | 1,000,000 | 13,241 | 21,137 | about 7,900 | 7,899 | 14,028 | 390.2 |

The resident set before the pass is the bench's own copy of the vectors plus Python; the pass
overhead (peak minus before) is about 8 KB per crop at every size from 91,184 up: linear. The index
sizes are snapshots (see Method); only the order of magnitude, about 5 to 8 KB per document after the
load and roughly double after the write-back, should be read from them.

### k-NN query latency (OpenSearch Lucene HNSW, ms, 200 queries)

| Data | Crops | Unfiltered p50 | p95 | p99 | Filtered p50 | p95 | p99 | Filter selectivity |
|---|---|---|---|---|---|---|---|---|
| P | 10,000 | 5.09 | 6.15 | 6.77 | 4.13 | 5.17 | 5.35 | 2.9 % |
| P | 25,000 | 5.60 | 6.53 | 8.23 | 4.80 | 5.42 | 6.19 | 2.9 % |
| P | 50,000 | 6.28 | 7.06 | 8.80 | 4.79 | 5.56 | 5.99 | 3.0 % |
| P | 91,184 | 5.85 | 6.56 | 7.47 | 4.80 | 5.22 | 5.73 | 3.0 % |
| S | 250,000 | 6.91 | 7.85 | 8.38 | 6.58 | 7.45 | 8.34 | 0.77 % |
| S | 500,000 | 7.18 | 8.00 | 9.82 | 7.47 | 8.32 | 9.21 | 0.75 % |
| S | 1,000,000 | 6.86 | 9.86 | 10.59 | 7.69 | 10.93 | 12.20 | 0.74 % |

### Isolated k-means train time (same call as the fit stage, no OpenSearch, 2 repetitions)

| Training vectors | Set P train s | Set P GFLOP/s | Synthetic train s | Synthetic GFLOP/s |
|---|---|---|---|---|
| 10,000 | 14.5 to 14.7 | 179 to 181 | 9.6 to 9.7 | 270 to 273 |
| 25,000 | 63.4 to 64.1 | 102 to 103 | 30.2 to 30.4 | 216 to 217 |
| 50,000 | 175.3 to 177.2 | 74 to 75 | 76.0 to 77.8 | 169 to 172 |

GFLOP/s is `2 * n * 512 * 1024 * 50 * 5 / train time`, the arithmetic the k-means must do. The
assign of the whole prefix to the 512 centroids took 0.14 s, 0.35 s and 0.71 s: negligible.

## Findings

1. **The pass is linear in the crop count except for the k-means fit, and the fit stops growing at
   the training cap.** Fetch (0.16 to 0.25 ms/crop) and write-back (0.31 to 0.42 ms/crop) are flat
   per crop from 10,000 to 1,000,000: 1,000,000 crops cost 168 s and 336 s, 100 times the 10,000
   figures for 100 times the crops. Refresh and bookkeeping stay under 1 s. Nothing in the OpenSearch
   path (fetch, write-back, refresh) is super-linear.
2. **The super-linear step is the IVF fit, below the 50,000-vector training cap.** On set P the fit
   takes 15.5 s, 67.0 s and 177.5 s at 10,000, 25,000 and 50,000 crops: 4.3x time for 2.5x crops
   (exponent 1.6), then 2.6x for 2x (exponent 1.4); wall time per crop goes 2.20, 3.23, 4.06 ms. The
   fit is 70 percent of the wall at 10,000 crops and 87 percent at 50,000. Above the cap
   (`IVF_MAX_TRAIN_SAMPLE` = 50,000) the train cost is constant and only the assign grows (about
   0.02 ms/crop), so the fit is flat: 181.6 s at 91,184 crops on set P and 85 to 100 s at 250,000 to
   1,000,000 on the synthetic data. That is why wall time per crop falls again (2.54 ms at 91,184 and
   0.60 ms at 1,000,000): from the cap on, the pass is linear with a constant fit term, and the
   per-crop cost approaches fetch plus write-back, about 0.5 ms.
3. **The super-linearity is in the train arithmetic rate, not in the work.** The k-means does
   exactly `n * 512 * 1024 * 50 * 5` multiply-adds (linear in n), yet the machine's rate falls as n
   grows, on real vectors from 180 to 74 GFLOP/s (10,000 to 50,000) and on the synthetic ones from
   270 to 170. The same call on the same training size is 2.3x slower on real set P than on the
   synthetic mixture (176 s against 77 s at 50,000), so part of the cost depends on the data. This
   log does not isolate the cause (candidates not tested: memory and cache behaviour of the
   assignment blocks, empty-cluster splitting on real data, host contention); the isolated probe
   reproduces the in-pass fit times, so OpenSearch, the write-back and the event loop are ruled out
   as causes.
4. **The earlier winter-scale estimate does not reproduce for this stage.** The wave plan quotes 4.2
   ms/crop (public set) and 7.6 ms/crop (set P) for the cluster pass and projects 85 to 155 minutes at
   about 1.2 million crops. Measured here, the IVF pass costs 2.5 ms/crop at 91,184 crops and 0.60
   ms/crop at 1,000,000. Extrapolating the measured stages to 1.2 million crops on set P (fetch 0.17
   and write-back 0.33 ms/crop, fit about 200 s) gives about 14 minutes. The comparison is
   incomplete: the quoted figures came from end-to-end runs that may include the geometry stage, the
   refresh daemon and a busier host, none of which are in this ladder.
5. **Peak memory is linear and modest**: about 8 KB per crop above the input copy (7.9 GB over the
   input at 1,000,000 crops, 21 GB peak including the 13 GB input dump this tool keeps resident).
6. **k-NN latency is nearly flat in size.** Unfiltered p50 goes from 5.1 ms (10,000) to 6.9 ms
   (1,000,000) and p99 from 6.8 to 10.6 ms; filtered p99 reaches 12.2 ms at 1,000,000 (a 0.74
   percent filter). No outlier near a second appeared in any of 2,800 queries.
7. **Fixed 512 clusters.** The partition is 512 clusters at every size, so the average cluster grows
   from 20 crops (10,000) to about 1,950 (1,000,000). The speed result above comes from that choice;
   it also means cluster granularity is not scaled with the pool. This log does not measure the
   quality effect (the #341 harness does).

## Open issues

* **#337** (super-linear cluster pass): the data above point the investigation at the k-means fit
  (the stage that grows faster than the crop count, only below the 50,000 cap) and away from fetch,
  write-back and refresh. Next steps for the wave: find the cause of the falling GFLOP/s and the
  real-versus-synthetic gap, and measure the geometry stage and the refresh daemon's assign-only
  pass at the same sizes, which this ladder did not cover.
* **#356** (text search constant 1.5 s outlier on about 1 in 10 queries at 91k items): the k-NN
  probe here goes straight to OpenSearch with a stored vector, 200 queries per size and none
  slower than 13 ms, so the outlier is not in the index's k-NN path at 91,184 rows; it must be in
  the text-to-vector or API part of the request.
* **#45** (scale rows): the synthetic ladder to 1,000,000 fits in 11 minutes per pass on this host
  and 14 GB of index, so the 1M row of the cluster stage is cheap; the real-crop 1M row still
  needs real data (synthetic planted structure has a different fit time, finding 3).
* Measurement tooling:
  * `cluster_plumbing_bench.reset_clusters` (a whole-index `update_by_query` that rewrites every
    document with its vector) ran past 30 minutes at 250,000 rows and then failed with
    `ConnectionError(400, 'Too many headers received')` from the async client; the cause was not
    investigated, the ladder avoids it with a fresh project per repetition.
  * A pass can report `status: success` with `n_residuals` smaller than the pool when the refresh
    daemon assigns a new project's items first (seen as 5,117 of 25,000 and 62,760 of 91,184 in
    the first, discarded version of the ladder, which reset in place). The benches check
    `n_residuals`; a caller that does not would take a partial pass for a full one.

## Reproduce

`$OS` is the stack's OpenSearch URL (loopback), `$OUT` a directory outside the repository with
about 60 GB free, `PY` the project interpreter (`.venv/bin/python`). Run from the repository root.

```bash
# 1. Read-only export of a project's stored crop vectors (set P) to the vecdb npz.
$PY scripts/bench/export_project_embeddings.py --opensearch-url $OS \
    --index op_prj_<slug>__items --out $OUT/setP.npz --seed 20261028

# 2. Seeded synthetic scale-up, 1,000,000 x 1,024, centroids sampled from set P.
$PY scripts/bench/gen_synthetic_embeddings.py --base $OUT/setP.npz \
    --out $OUT/synth1m.npz --n 1000000 --components 1500 --sigma 0.6 --seed 7

# 3. The ladder, one process per size, 3 repetitions each.
for n in 10000 25000 50000 91184; do
  $PY scripts/bench/cluster_ladder.py --opensearch-url $OS --data $OUT/setP.npz \
      --size $n --reps 3 --label setP --out $OUT/results --state-dir $OUT/state
done
for n in 250000 500000 1000000; do
  $PY scripts/bench/cluster_ladder.py --opensearch-url $OS --data $OUT/synth1m.npz \
      --size $n --reps 3 --label synth --out $OUT/results --state-dir $OUT/state
done

# 4. The isolated fit, on both datasets.
$PY scripts/bench/ivf_fit_scaling.py --data $OUT/setP.npz --sizes 10000,25000,50000 \
    --reps 2 --out $OUT/results/fit_scaling_setP.json
$PY scripts/bench/ivf_fit_scaling.py --data $OUT/synth1m.npz --sizes 10000,25000,50000 \
    --reps 2 --out $OUT/results/fit_scaling_synth.json
```

`cluster_ladder.py` writes `<label>_<size>.json` with every repetition's stage split, resident set,
index sizes and the k-NN percentiles. The older single-tool ladder
(`scripts/bench/cluster_bench.py --source vecdb`) reads the same npz and gives wall time only.
