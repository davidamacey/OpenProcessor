# v0.7.0 measurement log: cluster pass size ladder

Measured 2026-10-28 (one session, 06:30 to 10:30 local). Groundwork for #337 (super-linear cluster
pass) and the scale rows of #45. Every number below is a median of 3 repetitions unless a range is
shown; raw per-repetition rows are in the JSON files the tools write (the small ones are in
`docs/benchmarks/v070/`, see the Raw-data index below; the rest is kept outside the repository).
Only the IVF cluster pass (`cluster_residuals`, method `ivf`) was measured. The geometry stage, AHC
refinement, UMAP and the refresh daemon's assign-only pass were not.

## Raw-data index

Small raw files are committed under [`docs/benchmarks/v070/`](../benchmarks/v070/README.md) (paths below
are relative to it). Everything else stays in the artifact root `$SCRATCH` (the host's scratch volume, outside
the repository); `MANIFEST.sha256` there lists the checksums of the vector dumps. Use this table to find the
file behind any number in this log.

| Log section | Committed (`docs/benchmarks/v070/`) | Artifact root only (`$SCRATCH/...`) |
|---|---|---|
| Ladder: wall and stage split, memory, k-NN, isolated fit | `cluster_ladder/{setP,synth}_*.json` (3 repetitions each, `rss_mb`, `index_*`, `knn`), `cluster_ladder/fit_scaling_*.json` | `op-data/cluster_ladder/setP.npz`, `synth1m.npz` (inputs, checksums in the manifest), `logs_*.txt`, `results_v1_reset/` |
| A and B (source device, concurrency, batch) | `ingest_campaign/results.jsonl` (labels `A_*`, `B_*`: img/s, stage sums, `gpu2_util_mean`, `triton_cpu_pct_mean`, load) | `op-data/lm/B_*.out`, `A_*`; 5 s samples `op-data/lm/samples_A_*.csv`, `samples_B_*.csv`; `triton_top_ingest_alone.txt` |
| C (cluster pass against an ingest) | `ingest_campaign/cluster_vs_ingest/C_*_91184.json`, `ingest_campaign/results.jsonl` (labels `C_ingest_*`) | `op-data/lm/C_ingest_*.out`, `C_ladder_*.log`, `samples_C_*.csv` |
| D (text search latency) | `ingest_campaign/D_summary.jsonl` (one summary per row) | per-request lists `op-data/lm/q_D_*.jsonl` (the 400-request new-connection row is `q_D_idle_nokeep_unique.jsonl`, the second project is `q_D_nokeep_otherproject_cocotest.jsonl`) |
| E1 (embed pass, 0.6.0 against branch) | `live_e/exp1.jsonl` | `lv/out/samples_exp1_*.csv`, `vec_a_*.npz`, `vec_p_ingest.npz` |
| E2 (#358 duplicates) | `live_e/exp2.jsonl`, `live_e/exp2_capped_probe.jsonl` | `lv/out/samples_lv-358*.csv`. The distinct-`imohash` and orphan counts of the table have no retained file: those projects were deleted after the counts were read. |
| E3 (#356 restart and recycle) | `live_e/exp3.jsonl` | per-request lists `lv/out/q_restart_*.jsonl`, `q_recycle_*.jsonl`, `q_probe*_cocotest*.jsonl`. **No raw data** for the `ensure_indexes` timing (380 to 470 ms cold, 0.01 ms warm) or the cProfile figures: they were read from a shell in the API container and the output was not kept. Re-time with `ensure_indexes(client, project)` twice in a fresh worker before relying on them. |
| E4 (#268 instances) | `live_e/exp4.jsonl` | `lv/out/samples_lv-268-*.csv` |
| F1 to F4 | `live_f/exp1.jsonl`, `live_f/gpu500.log` | `lw/out/samples_exp1_*.csv`, `vec_*.npz`, `reg_cpu.npz`, `reg_gpu.npz` (F2 region parity), `m_*.txt` (route counters) |
| G1 (region parity) | `live_g/exp1.jsonl` (labels `gpuR*`, `cpuR*`, `gpuW*`, `cpuW`) | `lx/out/reg_{cpu,gpu}*.npz` (region vectors per pass, checksums in the manifest), compared with `lx/cos.py`; dumped by `lx/dumpreg.py` |
| G2 (worker region-stage cost) | none | **No raw data.** The stdout of the two stage runs (86 against 220 ms per item, CPU 0.077 against 0.212 s) was not saved; only the GPU sample file `lx/out/samples_wk_lx-p2_branch.csv` and the input list `lx/wk_in.json` remain. The CPU ratio is 2.7; the wall ratio is 2.57 (220 / 85.6). Re-run `lx/wk.py` against the stage before quoting these as a baseline for #362. |
| G3 to G5 (tuning, final configuration) | `live_g/exp1.jsonl` (`gpu500_mean` is the 500 ms mean), `live_g/final.log`, `live_g/g500_*.log` (500 ms samples, one value per line), `live_g/tune*.log`, `live_g/cpu_run.log` | `lx/out/samples_exp1_*.csv` |
| SP0.7 (why throughput fell past 4 batches) | `sp0/gpu_prep_concurrency/<row>/` (per-repetition reports with the route counters and stage counters, summaries), `*.call_failed.txt`, `*.triton_errors.txt`, `e1_per_process_peaks.md`, `threads/`; tables `sp0/sp0_tables_headroom.md` | 500 ms per-process memory and route-counter series `artifacts_local/bench/v070/sp0/e1/*.apps.log`, `*.routes.log`; set B rows `artifacts_local/bench/v070/sp0/exp/` (main checkout) |
| SPF (ingest-speed fixes: memory budget, config, load profile, acceptance) | `spf/<row>/` per-repetition reports (`metrics.api.counters` cut to `op_ingest_*`), `spf/<row>.stages.prom`, `*.call_failed.txt`, `*.triton_errors.txt`, `spf/model_memory.txt`, `spf/parity.txt`, `spf/endpoints_*.txt` | 500 ms per-process memory and route series `artifacts_local/bench/v070/sp0/spf/*.apps.log`, `*.routes.log`; parity captures `par_*.npz` (same directory, main checkout) |

Where a table in this log disagrees with the file named here, the file is right; the corrections made on
2026-10-28 (memory table statistic, GPU and Triton CPU columns of A to C, the section D count) are noted at
their tables.

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

### Memory and index size (median of 3 repetitions; range in brackets; recomputed from the per-repetition rows)

Memory is `VmHWM`/`VmRSS` in MiB as the bench wrote it; index sizes are `_stats` store bytes in MiB.

| Data | Crops | Resident before pass | Peak resident | Pass overhead (peak minus before, per repetition) | Index after load | Index after pass | Bulk load s |
|---|---|---|---|---|---|---|---|
| P | 10,000 | 980 | 1,043 (1,015-1,165) | 114 | 63 (54-98) | 40 (40-40) | 4.8 |
| P | 25,000 | 1,179 | 1,554 (1,511-1,640) | 453 | 99 (84-163) | 211 (162-234) | 9.5 |
| P | 50,000 | 1,445 | 2,057 (1,937-2,123) | 612 | 247 (98-268) | 784 (690-908) | 18.3 |
| P | 91,184 | 2,084 | 3,002 (2,974-3,069) | 950 | 456 (439-458) | 1,447 (1,426-1,494) | 33.1 |
| S | 250,000 | 4,372 | 6,382 (6,341-6,389) | 2,017 | 2,099 (2,094-2,193) | 4,315 (4,174-4,536) | 89.6 |
| S | 500,000 | 7,376 | 11,278 (11,172-11,284) | 3,908 | 2,783 (2,618-2,865) | 7,701 (7,691-7,826) | 199.5 |
| S | 1,000,000 | 13,241 | 21,120 (21,001-21,137) | 7,882 | 5,182 (4,979-7,899) | 14,133 (14,028-19,479) | 390.2 |

An earlier version of this table was headed "median" but showed the maximum peak of the three
repetitions and the index sizes of repetition 0 (for example 7,899 MiB after load at 1,000,000, which
is the high outlier of the three; the other two read about 5,000 MiB). The repetition-to-repetition
spread of the small sizes' "before" column is large because the first repetition of a rung started
with a smaller resident base (612 MiB at 10,000).

The resident set before the pass is the bench's own copy of the vectors plus Python; the pass
overhead is about 8 KB per crop from 250,000 up (2,017 MiB, 3,908 MiB and 7,882 MiB; 10 KB at 91,184):
linear. The index sizes are snapshots (see Method); only the order of magnitude, about 5 to 8 KB per
document after the load and roughly double after the write-back, should be read from them.

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
* Measurement tooling (both fixed in #357):
  * `cluster_plumbing_bench.reset_clusters` ran one `update_by_query` for the whole reset and
    failed after about 30 minutes at 250,000 rows with `ConnectionError(400, 'Too many headers
    received')`. The client refuses a response with more than 128 headers; the node's `Warning`
    headers accumulate over a request that long (the header flood is reproduced offline against
    a stand-in server, the node's side of it is not). The reset is now a polled node task over
    documents that hold a cluster field, and fails unless none is left.
  * A pass could report `status: success` with `n_residuals` smaller than the pool when the
    refresh daemon assigned a new project's items first (5,117 of 25,000 and 62,760 of 91,184
    in the first, discarded version of the ladder). `cluster_residuals` now returns
    `status: partial` when it read less than 99 percent of the pool it expected (or a write
    failed), without fitting or writing; the auto-label job fails on it.

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

# Live ingest, contention and query-latency campaign

Measured 2026-10-28 (afternoon, one session, about 4 hours) on a running test stack (compose project
`optest1`: Triton on one RTX A6000, OpenSearch 3.6.0 single node on local NVMe, API with 32 uvicorn
workers, the stack's refresh daemon and curation workers running). Not a quiet machine: other users'
jobs kept the 1-minute load average between 4 and 31 (typically 9 to 17); the load during each group is
in the tables. Every group is 3 repetitions unless stated, each in a fresh throw-away project that was
deleted afterwards (no existing project was changed). The driver scripts and the raw per-run JSON and
5-second GPU/CPU samples are kept outside the repository.

Data: **set P** photographs (private, 16 to 24 MP baseline JPEGs, GPU decode on). Two inputs:

* **Folder F**: 3,275 files (20.1 GiB), 3,184 distinct contents (88 groups of byte-identical files).
  One byte-identical copy lived on the bulk HDD RAID (ext4, local mount) and one on NVMe (copy verified by
  SHA-256 of every file).
* **5k**: 5,000 files, 4,997 distinct contents, 2.35 crops per image (F: 1.53).

Method notes: before every ingest the input files' page-cache pages were dropped
(`posix_fadvise(DONTNEED)`; verified: a cold sequential read of 100 files ran at 185 MB/s, the second read
at 2,500 MB/s), so every run reads from the device. GPU utilisation is `nvidia-smi` every 5 s on the
Triton GPU, the mean of the samples taken during the ingest command (`gpu2_util_mean` in
`results.jsonl`, 18 to 22 samples per run, so the idle samples at the start and end of the command are
included). Triton CPU is `docker stats` (100 percent = one core), the mean of the same samples
(`triton_cpu_pct_mean`); the table gives the mean of the three runs. An earlier version of these
columns was 3 to 7 points (GPU) and 70 to 100 points (Triton CPU) off these values and did not state
its method; they were recomputed from `results.jsonl`. **Stage seconds are summed over all batches in flight, not wall time** (concurrency x
batch images are in flight at once, so `gpu_decode` is mostly queueing time inside Triton); they are
compared across runs, not added up to the wall. Default ingest settings are batch size 16, concurrency
16. Reference: 17.9 MP per image on folder F, so 37.5 img/s is about 670 MP/s.

## A. Source device: HDD RAID against NVMe, same files (folder F, 2 x 3 runs)

| Source | img/s (3 runs) | Run s | GPU util % (3 runs) | Triton CPU % | `file_read` stage s (sum) | `opensearch_write` stage s (sum) | Load avg during |
|---|---|---|---|---|---|---|---|
| HDD RAID (cold cache) | 37.2, 38.3, 38.9 | 88, 86, 84 | 52, 52, 58 | 603 | 84, 97, 110 | 1,477 to 1,978 | 10.7 to 20.5 |
| NVMe (cold cache) | 37.3, 33.9, 37.2 | 88, 96, 88 | 56, 63, 70 | 563 | 14 to 17 | 873 to 1,088 | 14.4 to 25.8 |

The device does not matter: the median is 38.3 img/s from HDD and 37.2 from NVMe (the NVMe low run, 33.9,
ran at a load average of 25.8). The read stage is 5 to 7 times faster on NVMe (about 5 ms against 30 ms per
image summed over workers) but it is not on the critical path at 37 img/s (a 6.6 MB file at 37 img/s is
245 MB/s, within what the RAID delivers).

## B. Concurrency and batch size, NVMe, folder F (one factor at a time, 3 runs each)

| Setting | img/s (3 runs) | Median | GPU util % (3 runs) | Triton CPU % | `gpu_decode` stage s (sum, first run) | Load avg during |
|---|---|---|---|---|---|---|
| concurrency 4 (batch 16) | 33.0, 33.1, 33.9 | 33.1 | 59, 59, 52 | 510 | 2,964 | 9.4 to 11.4 |
| concurrency 8 | 37.2, 38.0, 37.6 | 37.6 | 62, 60, 58 | 606 | 6,586 | 11.6 to 13.0 |
| concurrency 16 (default, from A) | 37.3, 33.9, 37.2 | 37.2 | 56, 63, 70 | 563 | 16,278 | 14.4 to 25.8 |
| concurrency 32 (maximum) | 37.7, 37.0, 37.7 | 37.7 | 57, 56, 61 | 610 | 33,702 | 13.7 to 14.8 |
| batch 8 (concurrency 16) | 38.6, 39.0, 40.9 | 39.0 | 68, 57, 67 | 649 | 6,962 | 14.2 to 16.9 |
| batch 16 (default, from A) | 37.3, 33.9, 37.2 | 37.2 | 56, 63, 70 | 563 | 16,278 | 14.4 to 25.8 |
| batch 32 | 38.0, 39.2, 39.1 | 39.1 | 55, 68, 60 | 613 | 28,714 | 15.0 to 17.9 |
| batch 64 (maximum) | 37.3, 36.4, 36.8 | 36.8 | 51, 52, 62 | 611 | 41,382 | 13.2 to 17.1 |

Throughput saturates at about 37 to 39 img/s from concurrency 8 on. Four batches in flight (64 images) cost
12 percent; 8, 16 and 32 batches in flight, and batches from 8 to 64 images, are all within the spread of
a loaded host (36.8 to 39.1 medians). What grows with in-flight images is latency: the `gpu_decode`
stage sum rises 5.7 times from concurrency 4 to 32 (and 5.8 times from batch 8 to 64) at the same
throughput, i.e. the extra images wait in Triton's queue. At saturation the GPU is **not** busy: mean
utilisation is 51 to 70 percent (single samples reach 100 and drop to 0 between bursts), while Triton
uses about 6 cores (about 160 CPU-ms per photo) and the 48-thread host is 70 percent idle. Triton's
`top -H` during a 5k ingest shows about 30 `triton_python_b` threads at 5 to 60 percent each and no
single thread pegged, so the limit is not one saturated thread. The `gpu_image_prep` model has
`instance_group count: 1` (one Python-backend process owns the decoder); the data are consistent with
that one process being the ceiling (its CPU-side work: nvJPEG Huffman stage, crop JPEG encoding,
Python between GPU calls) but this campaign did not prove it, see the next measurements below.

**Same-content count is not stable (bug, see Bugs).** Folder F holds 3,184 distinct contents, yet runs
reported 3,184 to 3,189 images ingested (11 of the 24 runs in A and B stored more than 3,184).

## C. Cluster pass against a concurrent 5k ingest (#337 contention)

Ingest of the 5k set (GPU decode, cold cache, default settings) with and without a concurrent
`cluster_ladder.py` pass at 91,184 real vectors (set P export, IVF, the same call as the winter log
above). The ingest was started when the ladder's items index was loaded (50 to 58 s after the ladder
started), so the whole ingest ran inside the pass's fetch and fit stages. 3 runs each.

| Run | Wall s | Fetch s | Fit s | Write-back s | Refresh + tail s | Load avg at start |
|---|---|---|---|---|---|---|
| Cluster pass alone | 224.2, 224.6, 237.4 | 16.7, 16.5, 23.7 | 178.3, 176.1, 183.4 | 27.8, 31.6, 30.0 | 1.5, 0.6, 0.6 | 19.1, 9.8, 16.1 |
| Cluster pass during the ingest | 241.5, 238.1, 242.7 | 17.9, 16.9, 17.1 | 192.5, 191.5, 197.5 | 30.8, 29.1, 27.3 | 0.6, 0.8, 0.9 | 9.5, 20.1, 9.2 |
| Winter log (91,184 crops, 3 runs) | 231.2 (226.6 to 235.8) | 17.4 | 181.6 | 30.4 | 0.2 | 8 to 16 |

| Ingest of the 5k set | img/s (3 runs) | Run s | GPU util % | Triton CPU % | Crops stored | Load avg during |
|---|---|---|---|---|---|---|
| Alone | 28.0, 27.6, 27.7 | 179 to 181 | 56, 59, 61 | 573 | 11,723 / 11,724 / 11,723 | 14.2 to 17.4 |
| During the cluster pass | 28.8, 28.6, 28.2 | 174 to 177 | 67, 60, 65 | 590 | 11,723 / 11,724 / 11,724 | 13.5 to 17.1 |

The cluster pass alone today reproduces the winter log (median wall 224.6 s against 231.2 s; fit 178 s
against 182 s). **A concurrent ingest slows the cluster pass by about 8 percent** (median wall 241.5 s,
+17 s; fit +14 s, 192.5 s against 178.3 s; fetch and write-back unchanged), and **the cluster pass does
not slow the ingest** (28.6 against 27.7 img/s, +3 percent, inside the spread). The fit is CPU-bound
(FAISS threads) and the ingest's CPU work is in Triton and the API, so they compete mildly. The ingest
of the 5k set runs at 28 img/s against 37 for folder F because it has 2.35 crops per image against 1.53
(more crops to embed and encode per photo). Every pass clustered the whole pool (`n_residuals` 91,184;
none discarded) although the stack's refresh daemon was running.

## D. Text search latency (GET `/curation/projects/{project}/search/text` on the 91,184 items, 200 requests per row)

Sequential requests, 0.25 s apart, `page_size` 30, one client. "unique" = 200 different phrases;
"repeat" = 10 phrases cycling. Client time in ms; `total_time_ms` is the server's own figure.

| Condition | n | p50 | p95 | p99 | max | > 500 ms | > 1 s | Server p50 | Server p99 | Load avg |
|---|---|---|---|---|---|---|---|---|---|---|
| Idle, unique (run 1, first after restart) | 200 | 66.9 | 73.8 | 93.3 | 611.8 | 1 | 0 | 61.7 | 83.7 | 9.6 |
| Idle, unique (run 2) | 200 | 69.4 | 75.9 | 88.4 | 101.5 | 0 | 0 | 64.0 | 83.6 | 9.8 |
| Idle, unique (run 3) | 200 | 35.8 | 42.8 | 48.7 | 70.3 | 0 | 0 | 29.1 | 41.0 | 5.0 |
| Idle, unique (run 4) | 200 | 68.4 | 81.6 | 101.0 | 508.8 | 1 | 0 | 62.8 | 95.8 | 4.4 |
| Idle, repeat (run 1) | 200 | 34.2 | 38.4 | 42.8 | 53.4 | 0 | 0 | 27.9 | 33.6 | 9.0 |
| Idle, repeat (run 2) | 200 | 35.2 | 63.3 | 78.5 | 546.6 | 1 | 0 | 28.6 | 72.9 | 6.2 |
| During a 5k ingest, unique (run 1) | 200 | 36.0 | 50.3 | 110.6 | 114.2 | 0 | 0 | 29.4 | 99.4 | 6.9 to 11.3 |
| During a 5k ingest, unique (run 2) | 200 | 79.1 | 152.2 | 459.3 | 977.5 | 2 | 0 | 73.0 | 274.2 | 17.5 to 20.9 |
| During a 5k ingest, repeat | 200 | 36.5 | 164.1 | 353.7 | 747.8 | 1 | 0 | 29.6 | 193.9 | 12.2 to 13.3 |
| Idle, **new connection per request**, unique | 400 | 64.3 | 80.1 | 533.0 | **1,382.7** | 7 | **1** | 58.5 | 518.8 | 10.9 |
| Idle, new connection, same, after 600 warm-up requests | 400 | 65.6 | 74.5 | 88.1 | 257.8 | 0 | 0 | 59.8 | 81.7 | 12.5 |
| Idle, new connection, second project (1,364 items), unique | 300 | 60.0 | 86.5 | 484.7 | 514.6 | 1 | 0 | 54.7 | 479.0 | 8.9 |

(The three rows "during a 5k ingest" ran while the ingest was in progress: 26.7 to 27.6 img/s, about
the same as the ingest alone. The ingest and the queries are two clients of the same 32-worker API.)

Findings:

* Idle and with a persistent connection the endpoint is steady: p50 35 to 69 ms, p99 under 110 ms, no
  request over 1 s in 1,200. The two p50 levels (about 35 and about 67 ms) are two different workers:
  the 200 phrases are the same every run, a connection stays on one worker, and a worker that already
  encoded them answers from its text-encoder cache (about 29 ms: kNN plus response) while one that has
  not pays about 32 ms more for the text encode (Triton).
* **During an ingest the tail grows**: p95 150 to 165 ms, p99 350 to 460 ms, maximum 748 to 978 ms (idle:
  p99 under 110 ms apart from the cold-start rows). No request reached 1 s.
* **The ~0.5 s and ~1.4 s outliers are a per-worker, per-project first-request cost.** With a new
  connection per request (which lands on any of the 32 uvicorn workers) 9 of the first 400 requests
  took 445 to 586 ms (6 of them over 500 ms) and one took 1,383 ms; server time was within 10 ms of client time, so the time is
  spent inside the handler. After 600 more requests (every worker touched) the same 400 requests had a
  maximum of 258 ms. On a second project the 13 slow requests (445 to 515 ms) were all among the
  first 120, a constant about 480 ms, although every worker was warm for the first project. Timing the
  call directly in the API container: `ensure_indexes` (the first-request bootstrap of
  `src/services/curation/index_bootstrap.py`: index-exists checks plus 22 mapping migrations) takes
  **380 to 470 ms cold and 0.01 ms warm**, for both projects. (One timing run in the API container; its output was not kept, so there is no raw file. Unverified; see the raw-data index.) That is the constant outlier: it is paid
  once per (uvicorn worker, project), and again after a worker is recycled (`--limit-max-requests=10000`
  in the image command). The 1.38 s request fell on a worker with 27 earlier queries in its cache
  statistics, i.e. a barely used worker; its extra 0.9 s was not decomposed (host load 10.9, other
  first-use work in the worker, or the bootstrap against a busier OpenSearch).
* The text-encoder LRU is per worker (256 entries x 32 workers) and `GET /health/pe_text` reports the
  cache of whichever worker answers: it showed `hits 0 / misses 0` after hundreds of searches, then
  `hits 600 / misses 201` and `hits 27 / misses 201` on later reads. Across connections the cache hit
  rate is therefore far below what the 256-entry figure suggests.
* OpenSearch k-NN statistics (`GET _plugins/_knn/stats`, read before and after every run and right
  after each request over 1 s): `knn_query_requests` grows by the number of queries, `graph_memory_usage`
  0, `hit_count` / `miss_count` / `load_exception_count` 0, circuit breaker not triggered. The index is
  on the Lucene engine, so there is no native graph cache to go cold; nothing in these statistics changes
  at an outlier. The earlier cluster ladder already showed k-NN itself under 13 ms.
* The first text search after the API was restarted took 8.8 s (one observation, `total_time_ms` 8,843:
  a cold worker plus a cold first query; not repeated, since the stack was not restarted again).

## Bugs and suspected bugs found

1. **Bulk ingest stores identical files twice when copies are processed in concurrent batches.** Folder F
   has 3,184 distinct contents (SHA-256 and the product's `imohash` agree). Runs reported
   3,184 to 3,189 images ingested; the image index of one run held 3,189 documents with 3,184 distinct
   `imohash` values (5 contents stored twice, 4 in another run, 0 in a run that reported 3,184). It
   happened in 11 of 24 runs, never in the 3 NVMe default runs and the 3 concurrency-4 runs, and not in
   the 5k set's 9 runs (3 duplicate files, always 4,997 stored). The duplicate check in
   `ingest_batch` is a batch lookup against the index (`check_duplicates_by_hash_batch`) before the
   write, so a copy in another batch still in flight (or written but not yet refreshed) is not seen; a
   hypothesis from the code and the counts, not instrumented here. The summary counters then disagree with
   the folder's distinct-content count run to run, and the index holds duplicate items and crops.
2. **Per-worker first-request cost of about 0.5 s (up to 1.4 s) on every project and worker**, the #356
   outlier: see D. A fix direction is to bootstrap the indexes once at project bind (or at worker start
   for known projects) rather than on the first request, or to remember the bootstrapped set in shared
   state. The exact 1.5 s of the issue (on a larger deployment) was not reproduced; the closest was
   1.38 s.
3. **`GET /health/pe_text` cache figures are one worker's** (32 workers), so they cannot be used to judge
   the cache hit rate.

## What this changes in the plan, and what to measure next

* The ingest ceiling on this card is about 37 to 39 img/s for 18 MP photos (28 img/s at 2.35 crops per
  image), with the GPU about 40 percent idle and Triton at about 6 cores. The 34 img/s floor of the rehearsal
  table (stage 1) is for 0.7 MP images and is not comparable; the stage 2 floor for set P of 27 img/s is
  met by this measurement only at 2.35 crops per image (27.6 to 28.8), without margin.
* More concurrency or bigger batches will not raise throughput; do not spend sweep time there. The
  lever is inside `gpu_image_prep` (one instance): measure with `instance_group count: 2` on the same
  card (the model config records 8.2 GB measured per instance; the A6000 has 48 GB), and profile the model's time
  between GPU calls (Python/GIL time against nvJPEG Huffman and crop encoding).
* Measure a second decode route: a 640-pixel-reduced decode (DCT scaling) for the detector input against
  the full-frame decode, since 18 MP decode dominates.
* Repeat D under the cluster pass and the ingest together, and with 4 to 8 parallel search clients (what
  a review UI does), after the bootstrap fix, to see the real tail.
* Re-run A with the ingest at higher crops per image, and with a spinning-disk-bound (cold, 2 x
  larger) set, to find when the read stage reaches the critical path.

# Live verification of release/0.7.0 (#259, #356, #358, #268)

Measured 2026-10-28 (evening) on the same test stack (compose project `optest1`, Triton on one RTX A6000,
OpenSearch on local NVMe, 32 API workers). The stack ran the v0.6.0 images with the release branch bind-mounted
over `src/` and `scripts/` of the API and the four curation workers (read-only), and the branch's two changed
`gpu_image_prep` files copied into the Triton model repository; the model was reloaded through Triton model
control. Live check: the API container holds `PEEncoder.prime_text_backend`, `/health/pe_text` reports `pid` and
`cache_scope`, and the startup log carries one `curation_indexes_bootstrapped` line per project per worker. After
the campaign the original files and compose were restored and the stack was back on the v0.6.0 images. The host was
not quiet (1-minute load average 6 to 16 in most runs, up to 22 once; the figure for each group is in the
tables). Every group is 3 repetitions unless stated, each in a fresh throw-away project that was deleted
afterwards. Input data: **set P** photographs (private; 16 to 24 MP baseline JPEGs) and 1,500 public COCO val2017
images (the 200-image README sample plus 1,300 more at random, seed 5). The crop cache was set unbounded
(`OP_CROP_CACHE_MAX_BYTES=0`) for the runs below, see the #358 note.

## E1. #259 embed reprocess pass (scope `embed`, parts crop, frame and region; all vectors rewritten)

Project holding 1,500 COCO images (9,666 crops, 6.4 per image) and, separately, 300 set-P photographs (675
crops). Baseline = the same pass on the same project with the API container switched back to the v0.6.0 image
code (only `src/` differs; Triton unchanged). API CPU = cgroup `usage_usec` of the API container over the pass
divided by images; GPU util = `nvidia-smi` mean of the Triton card over the pass.

| Set | Code | img/s (3 runs) | Median | API CPU-s / image | GPU util % (mean of runs) | Load avg during |
|---|---|---|---|---|---|---|
| COCO 1,500 | 0.6.0 | 15.64, 15.68, 15.71 | 15.68 | 0.0516, 0.0539, 0.0527 | 51 | 7.5 to 14.4 |
| COCO 1,500 | release/0.7.0 | 16.58, 16.29, 16.69 | 16.58 | 0.0660, 0.0658, 0.0649 | 52 | 6.5 to 9.5 |
| Set P 300 | 0.6.0 | 10.19, 10.87, 10.48 | 10.48 | 0.522, 0.520, 0.517 | 16 | 6.0 to 8.4 |
| Set P 300 | release/0.7.0 | 10.76, 10.73, 11.86 | 10.76 | 0.529, 0.524, 0.506 | 10 | 6.6 to 11.1 |

* The branch is 6 percent faster on COCO and unchanged on set P. The acceptance bar of #259 (at least 20 img/s,
  API CPU at most 0.047 CPU-s per image, encoder above 80 percent busy) is **not met** on this set: 16.6
  img/s, 0.065 CPU-s per image (25 percent more CPU per image than 0.6.0: the work moved to pool threads, it did
  not shrink), GPU about half idle. Set P is bound by CPU decode of 20 MP frames in the API (0.52 CPU-s per image).
  Triton's per-model compute counters sum over concurrent requests (they read 400 percent), so they are not a busy
  fraction and the card-level utilisation is used instead.
* **Vectors** (ingest vector against the vector after each pass, cosine; edge = box touching the right or bottom
  edge, 1,193 of 9,666 crops):

| Comparison | Crops | Median | p01 | Min | Below 0.999 |
|---|---|---|---|---|---|
| Branch ingest vs branch pass, interior | 8,473 | 0.999997 | 0.99996 | 0.9609 | 24 |
| Branch ingest vs branch pass, edge | 1,193 | 0.999997 | 0.99996 | 0.9993 | 0 |
| Branch ingest vs 0.6.0 pass (after the baseline runs), edge | 1,193 | 0.999997 | 0.99997 | 0.9993 | 0 |
| Frame vector, branch ingest vs branch pass | 1,500 | 0.999997 | 0.99992 | 0.9996 | 0 |
| 0.6.0 ingest (an older project of the same 200 images) vs branch ingest, interior | 1,185 | 0.999997 | 0.99999 | 0.99995 | 0 |
| 0.6.0 ingest vs branch ingest, edge | 169 | 0.999997 | 0.9885 | 0.9334 | 7 |

  The median is 0.999997, not exactly 1.0 (GPU float noise; repeated passes agree to the same level). Edge
  crops now agree between ingest and the pass (minimum 0.9993; 0.6.0 documented a minimum of 0.93 to 0.97),
  and interior crops of old and new ingest are identical. What remains is 24 small interior crops (median
  area 1.1 percent of the frame against 2.0 percent for all crops, minimum cosine 0.961) that differ between
  ingest and pass before and after the change; the likely cause is the stored box being rounded to
  normalized coordinates (not isolated here). The 7 edge crops that moved between 0.6.0 and the branch ingest are
  the intended change.

## E2. #358 duplicate content in concurrent batches

Folder of 3,450 files on NVMe: 3,000 distinct contents (set P) plus 450 byte-identical copies under other
names in 5 subfolders. Page cache of the input dropped before each run, fresh project per run, crop cache emptied.

| Concurrency | img/s (3 runs) | Ingested (3 runs) | `imohash` cardinality = images count | Items | Item image ids in images | Crop-cache files | GPU util % | Triton CPU % | Load avg during |
|---|---|---|---|---|---|---|---|---|---|
| 8 | 35.51, 36.36, 36.07 | 3,000 x 3 (450 duplicates) | 3,000 x 3 | 7,097 x 3 | yes x 3 | 7,097 x 3 | 61, 71, 68 | 584 to 610 | 10.0 to 12.7 |
| 16 | 34.96, 36.00, 36.57 | 3,000 x 3 | 3,000 x 3 | 7,097 x 3 | yes x 3 | 7,097 x 3 | 64, 65, 73 | 595 to 623 | 10.3 to 13.3 |
| 32 | 35.22, 36.04, 35.78 | 3,000 x 3 | 3,000 x 3 | 7,097 x 3 | yes x 3 | 7,097 x 3 | 73, 60, 66 | 603 to 608 | 12.3 to 13.7 |

All 9 runs: ingested equals the distinct count, no content stored twice, no orphan item, crop-cache file count
equals the item count. The campaign before the fix stored a content twice in 11 of 24 runs. Throughput does not
depend on concurrency (35.0 to 36.6). Notes: (1) with the default crop-cache cap of 3 GiB a 7,097-crop ingest
prunes the cache (one probe run held 5,347 files for 7,097 items), so the file-count check is only valid with
the cap off; (2) deleting a project does not remove its crop-cache files (7,097 orphan files remained after
all runs, 27 after a 10-image test).

## E3. #356 text search after an API restart and after a worker recycle

`GET /curation/projects/{project}/search/text` (a project of 91,184 items), 300 sequential requests 0.25 s apart, a new
connection per request (so any of the 32 workers answers), `page_size` 30, client time in ms. "Recycle" = all 32
worker processes sent SIGTERM and respawned by the supervisor, searches started about 50 s later.

| Condition | Phrases | First | p50 | p95 | p99 | Max | > 300 ms | > 1 s | Load avg |
|---|---|---|---|---|---|---|---|---|---|
| After `restart api` | 300 distinct | 919 | 71.5 | 891.9 | 969.7 | 1,119 | 19 | 1 | 11.1 |
| After `restart api` | 10 repeated | 68 | 39.1 | 76.0 | 79.8 | 877 | 1 | 0 | 5.9 |
| After worker recycle | 300 distinct | 945 | 71.9 | 84.0 | 966.1 | 997 | 14 | 0 | 6.8 |
| After worker recycle | 10 repeated | 68 | 39.6 | 77.0 | 86.6 | 102 | 0 | 0 | 10.1 |
| After a second recycle, other project (1,364 items) | 200 distinct | 938 | 65.6 | 826.4 | 937.9 | 988 | 13 | 0 | 22.2 |
| Earlier log (0.6.0, new connection, unique) | 400 | n/a | 64.3 | 80.1 | 533.0 | 1,383 | 7 (> 500 ms) | 1 | 10.9 |

* **The bootstrap part of the fix works and the outlier is not gone.** The per-worker `ensure_indexes` cost
  (about 0.45 s) is now paid at startup, but the first search a worker answers still takes 0.8 to 1.1 s (a
  worker answers its first search in 0.9 s whatever the project, then 30 to 70 ms). 14 to 19 of the first
  about 150 requests are slow, one per worker; after that no request is above 130 ms. The largest value and
  the count over 300 ms are not better than the 0.6.0 log; the count is higher because the 0.6.0 run had more
  requests after warm-up.
* **Root cause of what remains** (profiled in the API container with cProfile on the first request of a fresh
  app, single worker reproduces 0.90 s then 0.05 s): FastAPI 0.143.2 builds its route state lazily on the first
  request that reaches a router (`effective_candidates` / `_populate_api_route_state`, 321 routes, 1.5 s under
  the profiler). A text encode (0.03 s with the primed backend), the kNN query and the bootstrap are not
  involved. A throw-away request through the curation router at startup (in-process ASGI call, run with the
  lifespan) moves the cost out: first search 0.87 s on the warm-up request, 0.048 s after. A warm-up request
  to `/health/pe_text` does not help (a different router). This is a proposed fix, not part of the branch.
* `/health/pe_text` now reports `pid` and `cache_scope: "process"`; 12 reads returned 3 to 7 distinct pids, so
  a reader sees that the cache figures are one worker's.

## E4. #268 `gpu_image_prep` instance count (3,000 distinct NVMe photos, concurrency 16, batch 16)

Same card, KIND_GPU, the model unloaded and loaded through Triton model control after each config change.

| Instances | img/s (3 runs) | Median | GPU util % (3 runs) | Triton CPU % (3 runs) | GPU memory used after, MiB | `gpu_decode` stage s (sum) | Load avg during |
|---|---|---|---|---|---|---|---|
| 1 (shipped) | 31.75, 31.96, 31.63 | 31.75 | 71, 75, 70 | 600, 611, 634 | 30.4 to 30.7 k | 16.4 to 17.3 k | 8.5 to 11.9 |
| 2 | 47.06, 49.16, 48.01 | 48.01 | 87, 79, 78 | 875, 852, 864 | 34.6 to 37.9 k | 10.5 to 10.8 k | 12.3 to 16.2 |
| 3 | 48.28, 48.48, 48.04 | 48.28 | 79, 80, 80 | 914, 877, 810 | 43.5 to 44.9 k | 9.5 to 10.2 k | 12.4 to 14.4 |

Two instances raise ingest by 51 percent (31.8 to 48.0 img/s) for about 7 GB more card memory and 2.5 more Triton
cores; a third instance adds 0.6 percent (inside the spread) for another 6 to 7 GB. Recommended count: 2.
(The 31.8 img/s baseline is the 3,000 distinct photos, 2.3 crops per image; folder F of the earlier campaign
had 1.5 crops per image and reached 37.) Output parity with two instances: see below.

Parity (throw-away Triton 26.06 on GPU 0 only, detector, PE encoder and `gpu_image_prep`, the 200 public COCO
images, 8 concurrent streams, every output tensor kept): count 1 against count 2 gave identical status, size,
letterbox, det_scores and det_classes; embeddings, boxes, blur sums and crop JPEGs differ at the level of run-to-run
noise already present between two runs at count 1 (TensorRT kernels and the concurrent batching, not the instance
count). Max abs diff and minimum cosine, count 1 vs count 1 / count 1 vs count 2: det_boxes 2.9e-3 / 3.4e-3
(the 0.1975 raw diff in one image is row order; sorted by score it is 3.4e-3), whole_embedding 1.9e-3 and cosine
0.99992 / 2.2e-3 and 0.99992, crop_embeddings 4.1e-2 and 0.9885 / 5.7e-2 and 0.9905, blur_sums 3.04e6 in both
(max over 10 to 12 of 200 images; cosine 0.999988 in both), crop JPEG bytes differing 13 / 12 of 1,365. Memory after
the run was 9.6 GB for one instance and 9.8 GB for two (the CuPy pool grows with use on 0.3 MP frames).

# Live verification 2 of release/0.7.0: #259 embed pass on the GPU decode (2026-10-28, late)

Same test stack as above (compose project `optest1`, v0.6.0 images, Triton on one RTX A6000, 32 API workers).
Branch `src/` and `scripts/` of release/0.7.0 (3c8d0d38) bind-mounted read-only over the API and the four
curation workers; the three changed `gpu_image_prep` files (`config.pbtxt`, `1/model.py`, `1/prep_geometry.py`)
copied into the model repository and the model reloaded through Triton model control. After the reload
`/v2/models/gpu_image_prep` listed the inputs `jpeg`, `embed_crops`, `boxes`, and the instance count was 2 (as
committed). The photos were hard links of 1,500 set-P photographs (private; 16 to 24 MP) on NVMe and 1,500 public
COCO images (the first 1,500 of the pinned `coco_bench_2000` manifest, SHA-256 checked), read through the source
alias overlay. Fresh throw-away projects (`lw-` prefix) were used and deleted at the end. Host load average during
the runs was 5 to 10. The GPU-route runs and the CPU baseline differ only in the API environment
(`OP_INGEST_GPU_DECODE=0` through a second overlay for the baseline). API CPU = cgroup `usage_usec` of the API
container over the pass divided by images; GPU util = mean of 5 s `nvidia-smi` samples of the Triton card.

## F1. Throughput (set P, 1,500 photos, 3,451 crops, parts crop, frame and region; every vector rewritten)

Ingest of the set into the fresh project ran at 45.2 img/s (1,500 ingested, 0 failed). The photos carry no region
boxes (as in E1), so the region part had nothing to embed in these runs.

| Route | img/s (3 runs) | Median | API CPU-s / image | GPU util % (5 s mean) | `op_reprocess_decode_total` per pass |
|---|---|---|---|---|---|
| GPU route (branch) | 28.06, 29.30, 28.11 | 28.11 | 0.0294, 0.0313, 0.0316 | 33.3, 31.5, 34.9 | gpu +1,497, cpu_small +3, no other `cpu_*` |
| CPU route (`OP_INGEST_GPU_DECODE=0`, 1 run) | 12.23 | 12.23 | 0.497 | 27.5 | counter absent (route not usable) |
| E1 release/0.7.0 before the route (for reference) | 10.76, 10.73, 11.86 | 10.76 | 0.529, 0.524, 0.506 | 10 | n/a |

A fourth GPU-route pass in a second fresh project with `nvidia-smi` every 500 ms: 28.64 img/s, 0.0302 CPU-s per
image, GPU utilization mean 34.3 percent, 34 percent of the 148 samples above 50 percent (max 100).

| Target | Result | Met |
|---|---|---|
| at least 20 img/s | 28.1 median (2.6x the 10.8 of E1; 2.3x the CPU run of this campaign) | yes |
| at most 0.047 API CPU-s per image | 0.029 to 0.032 (0.497 on the CPU route) | yes |
| GPU more than 50 percent busy | 31.5 to 34.9 percent (5 s samples), 34.3 percent (500 ms samples) | **no** |
| `route="gpu"` equals the photo count | 1,497 of 1,500 per pass; the other 3 photos are under 3.0 MP and counted `cpu_small` (the gate, by design) | yes, with that reading |

* The GPU does not reach 50 percent although the pass is 2.3x faster than the CPU route; the pass runs at 28 img/s
  while ingest of the same photos on this stack runs at 45 to 49 img/s. The pass is therefore limited upstream of
  Triton: the candidates named in the procedure are the 16-photo in-flight bound (`_IMAGES_IN_FLIGHT`) and the
  64-photo chunk barrier. Neither was varied here (source not edited).
* With 5,177 synthetic region boxes added (about 5.7 vectors per photo, see F2) the GPU pass ran at 12.92 img/s,
  0.0482 CPU-s per image, GPU 52 percent busy (1 run); the CPU route on the same boxes ran at 4.44 img/s, 0.753
  CPU-s per image. Vectors per second are about the same as without regions (about 90 to 95 per second), so the
  encoder side is what bounds a box-heavy pass. The 0.047 CPU-s bar is missed by 0.001 in that one run.

## F2. Vector parity

Crop and frame vectors of the GPU pass against the vectors written by ingest (3 passes; cosine):

| Pass | Part | Vectors | Min | p01 | Median | Below 0.999 |
|---|---|---|---|---|---|---|
| 1 | crop | 3,451 | 0.99984 | 0.99993 | 0.999997 | 0 |
| 1 | frame | 1,500 | 0.99986 | 0.99994 | 0.999996 | 0 |
| 2 | crop | 3,451 | 0.99986 | 0.99993 | 0.999997 | 0 |
| 2 | frame | 1,500 | 0.99989 | 0.99995 | 0.999996 | 0 |
| 3 | crop | 3,451 | 0.99984 | 0.99993 | 0.999997 | 0 |
| 3 | frame | 1,500 | 0.99991 | 0.99995 | 0.999996 | 0 |

Gate (p01 at least 0.995): met by a wide margin, and repeat passes agree. For comparison, the CPU pass against
ingest on the same photos: crop min 0.9936, p01 0.9975, median 0.9994 (559 below 0.999); frame min 0.9974, p01
0.9991, median 0.9997. That is the nvJPEG against libjpeg difference ingest already has.

**Region vectors, GPU route against the CPU route: gate not met.** The photos had no region boxes, so for this
check two accepted boxes were written straight into the throw-away project's items (the first inside each item box
at 70 percent of its size, on all 3,451 items; the second at the top-left half of the item box on every second
item): 5,177 boxes, identical stored geometry on both runs. One CPU-route pass and one GPU-route pass, vectors
compared by cosine: n 5,177, **min 0.842, p01 0.886, median 0.965**, 4,226 below 0.995 (required: p01 at least
0.99). The cosine depends on the box size (medians: area under 2 percent of the frame 0.933, 2 to 5 percent
0.937, 5 to 10 percent 0.981, 10 to 30 percent 0.995, over 30 percent 0.997).

Cause (probed in the API container on 40 items, 55 boxes, crops of 800 to 3,800 px per side): the CPU region path
embeds the crop after a JPEG round trip at quality 92 (as designed and as the docs say). The same crop embedded
without the round trip against the GPU vector: median cosine 0.9993 and 0.9994. The CPU crop with the round trip
against the one without: median 0.934 for boxes under 5 percent of the frame, 0.993 above. So the GPU vector
agrees with the raw-pixel CPU embedding to the level of the crop part; the divergence is the CPU path's JPEG
round trip on multi-megapixel crops (the encoder client then downsamples them by a large factor, which turns the
JPEG noise into a large vector change). The "expected above 0.99" statement in PERFORMANCE.md is wrong for
21 MP photos. The GPU route is the closer one to the pixels; the fix, if cross-route agreement is wanted, is on
the CPU path (drop the round trip), which changes stored region vectors of existing CPU-embedded projects.

## F3. COCO (1,500 public images, ingest 29.1 img/s), release/0.7.0, GPU route enabled

| Run | img/s | API CPU-s / image | GPU util % (5 s) | `op_reprocess_decode_total` growth |
|---|---|---|---|---|
| 1 | 16.27 | 0.0640 | 43 | cpu_small +1,500 |
| 2 | 16.20 | 0.0642 | 42 | cpu_small +1,500 |
| 3 | 16.19 | 0.0643 | 54 | cpu_small +1,500 |

Reproduces the E1 release/0.7.0 row (16.58, 16.29, 16.69 img/s; 0.0660, 0.0658, 0.0649 CPU-s per image). Every
photo is counted `cpu_small`, no `gpu` growth, nothing else: the gate keeps COCO on the CPU path unchanged.

## F4. Zero-box request (parts `["frame"]`, set P, 1,500 photos)

No crop and no region box is sent for any photo (an empty `boxes` input). Completed, 0 failed, 1,500 frames
written, 47.6 img/s. Counter growth: gpu +1,497, cpu_small +3, no other `cpu_*`: no fallback storm, the model
accepts a `[0, 4]` boxes tensor.

## F5. Bugs and findings

1. Region parity gate as written in PERFORMANCE.md fails (F2); cause is the CPU path's JPEG round trip, not the
   GPU route. The documented expectation and the gate need to be rewritten or the CPU path changed.
2. GPU busy stays at about a third on set P (F1); the pass is not Triton bound.
3. Documentation points to correct: `route=gpu` equals the photo count only for photos at or above 3.0 MP (3 of
   the 1,500 set-P photos are below).
4. The route counter does not exist while the route is unusable (`OP_INGEST_GPU_DECODE=0`): nothing to read.

After the runs the three model files were restored (diff against the backup identical), the model reloaded with
the original two inputs, the overlays removed, the API and four workers recreated on the v0.6.0 images with no
branch mounts, and the three `lw-` projects deleted (default, cocotest, hiresp and nas20k untouched).

# Live tuning and parity of the embed pass, region vectors and the worker region stage (2026-10-28, night)

Same test stack (compose project `optest1`, v0.6.0 images, Triton on one RTX A6000, 32 API workers). Branch
`perf/v070-259c-tuning` (release/0.7.0 plus the change below) bind-mounted read-only over the API and the four
curation workers; the three `gpu_image_prep` files of the branch copied into the model repository (two instances,
inputs `jpeg`, `embed_crops`, `boxes`) and the model reloaded through Triton model control. Photos: set P (private;
16 to 26 MP baseline JPEGs), hard links on NVMe, read through the source alias. Fresh throw-away projects with the
prefix `lx-`, deleted at the end. GPU util = `nvidia-smi` of the Triton card every 500 ms over the pass (the 5 s
samples of section F read 14 points lower for the same pass, they alias the pass). API CPU = cgroup `usage_usec` of
the API container over the pass divided by images. The host was shared: the 1-minute load average was 4 to 10 in
every run below except the final group (G5), which ran at 20 to 80 because other users' jobs started.

## G1. #360 live parity: region vectors, GPU route against CPU route (project of 1,499 photos, 3,559 crops, 5,339 synthetic accepted boxes)

The synthetic boxes are those of F2 (the first inset 15 percent in the item box, the second at the top-left half on
every second item), written into the items; the box geometry is identical for both routes. CPU route =
`OP_INGEST_GPU_DECODE=0` on the API. 3 passes per route, every vector rewritten (parts crop, frame, region).

| Route | img/s (3 runs) | API CPU-s / image | Load avg during |
|---|---|---|---|
| GPU | 14.61, 12.91, 12.47 | 0.0453, 0.0482, 0.0493 | 5.5 to 6.4 |
| CPU, no round trip (branch) | 7.65, 7.67, 7.71 | 0.571, 0.574, 0.575 | 7.2 to 8.2 |

Region vector cosine, GPU pass against CPU pass (pass k against pass k; the 3 pairs agree to 5 digits):

| Set | n | Min | p01 | Median |
|---|---|---|---|---|
| All boxes (pass 1 / 2 / 3) | 5,339 | 0.98345 / 0.98348 / 0.98344 | 0.99699 / 0.99698 / 0.99698 | 0.999275 / 0.999274 / 0.999274 |
| Area under 2 percent of the frame | 1,235 | 0.98345 | 0.99540 | 0.999044 |
| 2 to 5 percent | 1,235 | 0.99540 | 0.99736 | 0.999245 |
| 5 to 10 percent | 1,352 | 0.99329 | 0.99802 | 0.999359 |
| 10 to 30 percent | 1,469 | 0.99601 | 0.99788 | 0.999366 |
| Over 30 percent | 48 | 0.99816 | 0.99834 | 0.999286 |
| GPU pass 1 against GPU pass 2 (repeat) | 5,339 | 0.99986 | 0.99997 | 0.999997 |
| CPU pass 1 against CPU pass 2 (repeat) | 5,339 | 0.99964 | 0.99991 | 0.999997 |

Gate of #360 (p01 at least 0.99): **met**, p01 0.9970 overall and at least 0.9954 in every area bucket (F2 before the
fix: p01 0.886, median 0.965). Two boxes are under 0.99 (min 0.9835, both under 2 percent of the frame).

**Worker-written vector against the CPU pass** (the detection worker's region stage, `embed_written_regions`, run
in a worker container on the 1,208 items of 500 photos with 2 accepted boxes each, 2,416 boxes, the same box
geometry stored in the project; then one CPU-route pass and one GPU-route pass over the project). Branch worker
against CPU pass: n 2,416, min 0.99937, p01 0.99992, median 0.999997 (3 worker runs agree). Branch worker against
GPU pass: min 0.99110, p01 0.99649, median 0.999229. v0.6.0 worker (item-crop JPEG round trip) against the CPU
pass: min 0.811, p01 0.872, median 0.940. The branch worker and the CPU pass cut and embed the same pixels; the
v0.6.0 worker is where the old divergence lived.

## G2. Worker region-stage cost (#360)

Not a region-profile ingest: the stack has no segmenter and the one detector leg that loads here (the demo
car model) finds no box inside the set-P item crops (profile `detector_model` tried: the YOLO end-to-end model is
rejected by the region detector for its output names, the non-end-to-end one has no engine in the model
repository, the demo model timed out and found nothing; every item ended `no_region_box`). The stage is therefore
measured directly: `embed_written_regions` (branch) and its v0.6.0 version on 1,208 items of 500 set-P photos with
2 accepted boxes each, in flushes of 32 items as the worker does, a worker container each, the crop JPEGs from the
crop cache as the worker has them, one warm-up flush excluded, 3 runs each (load average 4 to 6). This is the
embed stage alone; the detector, the verify call and the write are not in it.

| Code | Wall s / 1,208 items (3 runs) | Wall ms / item | Worker CPU-s / item |
|---|---|---|---|
| v0.6.0 (box cut from the item-crop JPEG) | 103.1, 103.9, 103.4 | 85.4, 86.0, 85.6 | 0.0770, 0.0776, 0.0773 |
| branch (box cut from the decoded source) | 266.2, 266.4, 263.0 | 220.4, 220.5, 217.7 | 0.2121, 0.2122, 0.2094 |

The stdout of these runs was not kept (see the raw-data index), so the figures cannot be re-derived from retained data. The branch costs 0.135 CPU-s and 133 ms more per item (2.7 times in CPU; 2.57 times in wall): each item decodes its full 16 to 26 MP source
once (about 0.13 CPU-s), and a photo with 2.4 items is decoded 2.4 times. The stage decodes the items of a flush
one after another (`_task_crops` awaits each in turn), so a flush of 32 items takes about 7 s of wall in that
stage. Whether that is a problem depends on the rest of the worker: the stage runs once per flush after the box
list is final, and in this stack's profile runs the worker's measured end-to-end rate was about 5 items/s
(`session_avg_cps` 4.9 in the logs of the failed detector attempt), the same order as the stage alone (4.5
items/s). On a worker that is bound by the segmenter and the VLM verify call the stage adds a fraction; on a
detector-only profile on large photos it becomes the limit. Not changed here; options if it matters: decode the
items of a flush concurrently on the CPU pool with a bound of a few photos (the decoded 21 MP frame is 63 MB), or
decode each photo once for all its items.

## G3. #259 tuning: in-flight bound and chunk barrier (project of 1,499 photos, 3,559 crops, no region boxes; parts crop, frame and region; GPU route)

Each configuration: API recreated, one warm-up pass, then 3 passes. `_IMAGES_IN_FLIGHT` is the photos embedded at
once, `CHUNK` the photos per read/embed/write unit, window = chunks embedded at once (the change below).

| Configuration | img/s (3 runs) | Median | API CPU-s / image | GPU util % (500 ms mean) | Load avg during |
|---|---|---|---|---|---|
| in flight 16, chunk 64 (shipped) | 27.66, 27.96, 27.63 | 27.66 | 0.0325, 0.0318, 0.0320 | 48.3 to 49.1 | 6.0 to 7.8 |
| in flight 16, chunk 64, window 1 (after the refactor) | 27.87, 27.84, 28.03 | 27.87 | 0.0319, 0.0320, 0.0309 | 47.4 to 50.0 | 7.2 to 8.0 |
| in flight 24 | 27.06, 27.84, 27.55 | 27.55 | 0.0319, 0.0320, 0.0318 | 46.5 to 47.0 | 5.5 to 6.5 |
| in flight 32 | 27.24, 27.62, 26.56 | 27.24 | 0.0328, 0.0321, 0.0344 | 45.8 to 46.2 | 5.4 to 7.2 |
| in flight 48 | 25.49, 26.43, 25.91 | 25.91 | 0.0355, 0.0342, 0.0336 | 42.2 to 44.8 | 5.6 to 6.2 |
| in flight 16, chunk 256 | 32.91, 33.19, 33.10 | 33.10 | 0.0293, 0.0295, 0.0300 | 55.3 to 59.0 | 6.3 to 7.2 |
| in flight 16, one chunk of all 1,499 (no barrier) | 34.59, 33.23, 34.85 | 34.59 | 0.0295, 0.0294, 0.0289 | 58.1 to 60.9 | 6.7 to 8.7 |
| in flight 16, chunk 64, **window 2** | 51.12, 48.16, 51.84 | 51.12 | 0.0250, 0.0260, 0.0259 | 84.5 to 92.1 | 6.8 to 7.7 |
| in flight 16, chunk 64, window 3 | 51.47, 51.57, 51.19 | 51.47 | 0.0265, 0.0251, 0.0258 | 89.8 to 91.2 | 8.9 to 10.0 |
| in flight 24, chunk 64, window 2 | 51.01, 50.48, 50.36 | 50.48 | 0.0264, 0.0272, 0.0263 | 83.8 to 85.1 | 8.5 |

More photos in flight do not help and 48 hurts (a bigger tail at every chunk end); the limit is the barrier. The
single event-loop thread of the job was 42 percent busy during a pass (per-thread CPU from `/proc`, every other
thread about 1 percent), so it is not CPU bound. Overlapping two chunks gives 1.85 times the images per second,
more than removing the barrier by one big chunk (1.25 times), because the next chunk's reads and the previous
chunk's bulk writes now also overlap with the photos being decoded. A third chunk adds nothing (51.5 against 51.1).

**Change kept** (#259): `EMBED_CHUNKS_IN_FLIGHT = 2` in `src/services/curation/reprocess_images.py`: a pure
`embed` pass runs two chunks at once under one shared in-flight bound of 16 photos (`new_image_gate`,
`reembed_items(..., gate=)`). Progress is reported in chunk order, a failed chunk fails only its images, a cancel
waits for the chunks already started, a cancelled job cancels the chunks in flight, and a pass that also runs
`detect` or `open_vocab` is unchanged (one chunk at a time). Gain 27.7 to 51.1 img/s (+85 percent, bar +10 percent),
API CPU per image 0.032 to 0.025 (bar: not above 0.047). Tests (failing before the change): the next chunk embeds
while the previous one is held, a cancelled job leaves no chunk running, two calls share one in-flight bound.

## G4. Results identical

Same 1,499 photos, crop, frame and region vectors, the window-1 code against the window-2 code (3 passes) and the
region vectors of the earlier window-1 passes of G1 against 3 window-2 passes: the per-pass counters (`items`,
`crop_written`, `frame_written`, `region_written`, `queued`, `failed`) are identical string for string; crop vectors
n 3,559: min 0.99963 to 0.99976, p01 0.99993, median 0.9999965; frame n 1,499: min 0.99987, median 0.9999964; region
vectors n 5,339: min 0.99985, p01 0.99997, median 0.9999967 (the repeat-pass noise of the GPU route, G1). Every
image and item id appears in the same order in the dumps.

## G5. Final configuration, set P 1,499 photos, with and without region boxes (host load 20 to 80, other users)

| Project | img/s (3 runs) | API CPU-s / image | GPU util % (500 ms) | `op_reprocess_decode_total` |
|---|---|---|---|---|
| Crop + frame + region, no region boxes | 48.05, 47.84, 47.98 | 0.0230, 0.0244, 0.0231 | 81.6 to 82.8 | gpu +1,497 per pass, cpu_small +3 |
| Same with 5,339 region boxes (3.6 vectors per photo more) | 22.68, 21.88, 20.45 | 0.0355, 0.0357, 0.0377 | 68.4 to 77.5 | (all gpu) |

The box-heavy pass was 12.5 to 14.6 img/s before (G1) and is 1.6 times faster now; 0.0355 to 0.0377 CPU-s per image
is under the 0.047 bar (F1 measured 0.0482 for the box-heavy pass before the change). The runs of this group are
under heavy foreign load and are lower bounds. A COCO re-run was not made (all COCO photos take the unchanged CPU
route below the 3 MP gate, but the chunk window also applies to them; not measured).

After the helper-class refactor that the complexity ratchet asked for, the final code was re-run on the same project (load
average 13 to 18, other users): 42.12, 47.74, 51.48 img/s, API CPU 0.0262, 0.0258, 0.0246 CPU-s per image, GPU util 74.8,
82.0, 88.6 percent.

## G6. Findings

1. The 24 "missing" region vectors seen in one first GPU dump were a read before the index refresh (a second pass
   and dump with a refresh found 2,416 of 2,416); not a defect.
2. Per-pass GPU utilisation from 5 s `nvidia-smi` samples (F1) under-reads by about 14 points; use 500 ms samples.
3. The worker's per-item region embedding now decodes the full source per item (G2); see the options there.
4. The region detector rejects a Triton model whose output is not named `output0` (the YOLO end-to-end models),
   and a profile pointing at it activates without error; every item then ends `no_region_box` after one logged
   exception per item. Not changed here.

#259 acceptance bars on this stack (set P, 21 MP): at least 20 img/s (51, met), API CPU at most 0.047 (0.025,
met), vectors against ingest p01 at least 0.995 (0.9999, F2) and the encoder busy above 80 percent (85 to 92
percent, met). #360: region parity p01 0.997, worker and CPU pass agree at 0.999997.

## SP0: stage profile (ship-track plan section 5; `perf/v070-ingest-speed`)

Measured 2026-10-30 at commit `6224897f` (branch `perf/v070-ingest-speed`, merged with `origin/main` at `d46dadc2`),
one RTX A6000 (GPU 0, PCI 03:00) shared by Triton and the API, driver 615.71.10 (CUDA 13.4), Triton 2.70.0
(release 26.06), TensorRT 11.1.0.106, release 0.6.0 images with this worktree's `src/`, `scripts/` and `gpu_image_prep`
mounted over them, `gpu_image_prep` with two instances (`count: 2`, the full profile), `OP_CROP_CACHE_MAX_BYTES=0`,
policy `all`, stage `ingest`, 100 warm-up images per repetition excluded, input page cache dropped before every
repetition, GPU sampled every 500 ms, 3 repetitions per row, batches of 32 photos through `POST .../ingest/batch`.
Host: 48 logical CPUs shared with other sessions; the 1-minute load average at the start of each repetition is in the
JSON (above 16 is marked noisy; most rows contain noisy repetitions, see SP0.5). Sets: A = 1,946 of the pinned 2,000
COCO images (the selector keeps 1,946; 1,846 measured), B = 500 public high-resolution photos (400 measured, 10.35 MB
each), P = 3,000 seeded NVMe copies of the private photo set (2,900 measured, 6.71 MB each; numbers only).
Reproduce: `docs/reviews/sp0_review.md`, "Runbook: SP0 measurement", plus `--clients N` for the concurrent rows.
Raw per-repetition JSON: `artifacts_local/bench/v070/sp0/<row>/` in the main checkout; the same files and the rendered
tables are committed in `docs/benchmarks/v070/sp0/`.

**Driver finding that shapes everything below.** `run_baseline.py` posted its batches one after another, so the API never
had more than its per-batch limit of 16 photos in flight (`OP_MAX_INGEST_CONCURRENCY`). The rows named "1 batch" are
that driver; the rows named "4 batches" and "8 batches" use the new `--clients N` (N batches at once). The 1-batch rows
describe per-image latency; they under-drive the stack (GPU 15 to 35 percent) and must not be quoted as throughput.

### SP0.1 Headline

| Row | img/s median (min to max) | Crops / image | GPU 0 util mean / p50 % | Triton cores | API CPU-s / image | Load at start (max) | Noisy reps |
|---|---|---|---|---|---|---|---|
| A, COCO, 1 batch (CPU route) | 13.14 (13.12 to 13.47) | 6.42 | 45 / 30 | 0.23 | 0.078 | 27.0 | 1, 2 |
| B, 1 batch (GPU route) | 9.88 (9.63 to 9.96) | 1.65 | 15 / 0 | 1.90 | 0.262 | 10.2 | none |
| P, 1 batch (GPU route) | 14.80 (11.92 to 16.47) | 2.41 | 35 / 0 | 2.53 | 0.078 | 20.2 | 2, 3 |
| B, 4 batches | 26.64 (25.21 to 29.00) | 1.65 | 49 / 43 | 2.54 | 0.303 | 21.0 | 2, 3 |
| P, 4 batches | 24.98 (23.50 to 33.08) | 2.41 | 58 / 60 | 4.36 | 0.200 | 26.4 | 1 |
| P, 8 batches | 20.13 (19.20 to 21.91) | 2.41 | 39 / 19 | 2.87 | 0.701 | 23.0 | 2, 3 |
| B, 1 batch, STAGE_SYNC set to 1 (attribution) | 10.61 (9.76 to 10.86) | 1.65 | 21 / 0 | 1.83 | 0.252 | 17.3 | 3 |
| P, 1 batch, STAGE_SYNC set to 1 (attribution) | 15.47 (14.82 to 16.15) | 2.41 | 35 / 0 | 2.92 | 0.079 | 17.9 | 1, 2 |

Route counts (`op_ingest_decode_total`, one repetition): set A 1,846 `cpu_small` (all below the 3 MP gate); set B 351
`gpu`, 47 `cpu_progressive`, 2 `cpu_truncated`; set P 2,882 `gpu`, 10 `cpu_decoder`, 6 `cpu_small`. The 49 CPU-decoded
photos of set B explain its API CPU of 0.25 to 0.30 CPU-s per image against 0.07 to 0.08 for set P at one batch.
No row had a failed image.

### SP0.2 Per-stage tables

Stage times are sums over concurrent work divided by images, compared across rows, never added to wall time. The share
column is the median of the per-repetition shares, so a column of shares need not add to exactly 100.

`gpu_image_prep` internal stages, host wall ms per image that ran the stage:

| Stage | B, 1 batch | P, 1 batch | B, 4 | P, 4 | P, 8 | P, 1 batch, SYNC |
|---|---|---|---|---|---|---|
| decode (nvJPEG, EXIF) | 86.8 (39.5%) | 89.9 (19.3%) | 121.2 (26.2%) | 159.7 (10.8%) | 469.5 (30.5%) | 91.8 (19.2%) |
| embed_pe (wait for the PE calls) | 56.4 (22.3%) | 126.2 (27.6%) | 79.2 (15.4%) | 203.2 (11.3%) | 196.1 (4.3%) | 125.3 (26.1%) |
| letterbox (detector canvas) | 5.4 (2.2%) | 21.5 (4.5%) | 97.6 (17.9%) | 183.5 (11.2%) | 475.2 (14.3%) | 24.2 (5.1%) |
| embed_crops (crop squares) | 4.9 (2.0%) | 53.3 (12.0%) | 29.4 (5.0%) | 325.2 (21.3%) | 683.0 (14.2%) | 61.4 (12.9%) |
| crop_copy (device to host) | 6.5 (2.6%) | 28.8 (5.8%) | 24.7 (4.4%) | 186.5 (10.8%) | 382.2 (9.1%) | 28.9 (6.1%) |
| crop_jpeg (sum over the crops) | 29.7 (12.0%) | 70.6 (15.8%) | 31.1 (5.1%) | 81.0 (4.8%) | 94.7 (2.0%) | 74.1 (15.5%) |
| detect (detector call) | 39.7 (15.5%) | 52.1 (11.5%) | 50.4 (8.9%) | 68.3 (4.2%) | 111.5 (2.6%) | 50.7 (10.6%) |
| embed_frame (whole-frame square) | 5.2 (2.1%) | 6.8 (1.4%) | 45.7 (7.8%) | 96.3 (5.7%) | 227.7 (4.9%) | 9.1 (1.9%) |
| blur | 4.6 (1.9%) | 10.6 (2.2%) | 20.4 (3.6%) | 49.6 (3.0%) | 122.3 (2.5%) | 11.3 (2.4%) |
| queue (wait for a slot and a thread) | 0.35 (0.2%) | 0.80 (0.2%) | 21.8 (4.3%) | 180.8 (14.0%) | 328.7 (18.6%) | 0.71 (0.1%) |

API stages (`op_pipeline_stage_seconds`, ms per image, share of the API stage total):

| Stage | A, 1 batch | B, 1 batch | P, 1 batch | B, 4 | P, 4 |
|---|---|---|---|---|---|
| embed (PE crop embedding, CPU route) | 434.3 (54.5%) | 6.7 (0.5%) | 0.2 (0.0%) | 16.8 (1.0%) | 29.6 (1.3%) |
| gpu_decode (the call into `gpu_image_prep`) | 0 | 929.9 (70.0%) | 818.0 (72.1%) | 1150.0 (70.4%) | 1757.8 (77.4%) |
| opensearch_write | 335.3 (42.1%) | 276.4 (20.8%) | 312.3 (27.5%) | 269.2 (16.5%) | 332.6 (14.7%) |
| decode (CPU decode, fallback or small) | 4.7 (0.6%) | 70.9 (5.3%) | 0.4 (0.0%) | 105.4 (6.5%) | 49.5 (2.2%) |
| everything else | 2.8% | 3.4% | 0.4% | 5.6% | 4.4% |

Triton per model, ms per request (queue / compute): set A `pe_image_encoder` 16.79 / 17.96, detector 0.03 / 0.62;
B, 1 batch: PE 4.83 / 11.85, detector 16.43 / 12.02; P, 1 batch: PE 7.39 / 16.98, detector 17.35 / 17.62; P, 4 batches:
PE 17.37 / 21.80, detector 15.35 / 16.11, `gpu_image_prep` queue 74.9; P, 8 batches: PE 38.80 / 35.75, detector
7.30 / 10.85, `gpu_image_prep` queue 1079.8. The `gpu_image_prep` compute column (6 to 34 ms) covers only the hand-off:
the model is decoupled and works after `execute` returns.

### SP0.3 Findings

1. **The one-batch driver under-drives the stack.** At one batch the API holds on average 12 (set P) and 9 (set B)
   photos in flight (rate times `gpu_decode` wait: 14.8 x 0.818 and 9.9 x 0.930) out of 16. Posting 4 batches at once
   raises set B from 9.9 to 26.6 img/s and set P from 14.8 to 25.0 img/s (GPU 15 to 49 and 35 to 58 percent). The 100 MB/s
   input rate that both one-batch rows share (99.7 to 103.1 on B, 80.0 to 110.6 on P) was therefore the driver's, not a
   stack ceiling; at 4 batches it is 261 to 300 (B) and 158 to 222 (P).
2. **Past about 4 batches the stack gets slower.** Set P at 8 batches: 20.1 img/s (19.2 to 21.9), GPU 39 percent,
   `gpu_image_prep` queue 1,080 ms per request, API CPU 0.70 CPU-s per image (0.07 to 0.25 at 4 batches). The stages that
   only issue CUDA launches and copies from the model's worker threads grow with the load: letterbox 21 / 183 / 475 ms,
   embed_crops 53 / 325 / 683, crop_copy 29 / 186 / 382, embed_frame 7 / 96 / 228, blur 11 / 50 / 122 (set P at 1 / 4 / 8
   batches), 11 to 33 times for the same work; nvJPEG decode grows 5 times (90 / 160 / 470). The stages that wait on another
   model or encode JPEG on host threads grow 1.3 to 2 times (embed_pe 126 / 203 / 196, detect 52 / 68 / 111, crop_jpeg
   71 / 81 / 95). STAGE_SYNC set to 1 moved only 1 to 8 ms per image into the launch stages, so those stages are host-side
   launch and serialisation time, not kernel time. Candidate causes, not isolated here: the GIL and per-call overhead of
   32 worker threads in one Python process, or two instances time-slicing one GPU without MPS.
3. **Bottleneck on the GPU route (ranked by evidence).** (1) The Python process hosting `gpu_image_prep` under concurrency:
   CUDA-call stages (letterbox, embed_crops, embed_frame, crop_copy, blur) are 52 percent of the model's stage time at 4
   batches on set P (66 with the queue), against 26 percent at one batch, and the GPU is at 39 to 58 percent. (2) nvJPEG decode: 10.8 to 39.5 percent of the stage time in every GPU row, part of it host-side work inside nvJPEG; it grows 5 times
   at 8 batches. (3) The PE embedding wait `embed_pe`: 4 to 28 percent, a hop into `pe_image_encoder` (17 to 36 ms compute and
   7 to 39 ms queue per request). (4) Crop JPEG encode and copy: 14.6 to 21.6 percent at one batch, 9.5 to 15.6 percent at 4
   batches on set P. OpenSearch writes (14.7 to 27.5 percent of the API stage time) overlap the model's work. At one batch
   44 percent (P) to 74 percent (B) of the API's `gpu_decode` wait lies outside the model's stage clock (stages add to 461 and 240
   ms against 818 and 930 ms); the clock starts at `execute`, and Triton's request duration for the model was not captured.
4. **CPU route (set A).** The PE crop embedding is first (54.5 percent of the API stage time at 6.4 crops per image, PE
   queue 16.8 ms plus compute 18.0 ms per request), OpenSearch writes second (42.1 percent). 13.1 img/s at one batch with the
   GPU at 45 percent and 0.23 Triton cores; not re-run at 4 batches.
5. **The counters cost nothing measurable.** Set B at one batch: 9.88 img/s (9.63 to 9.96) with, 10.10 (10.06 to 10.59)
   without (merge-base `model.py`), median ratio 0.978. At 4 batches: with 28.5 (25.2 to 31.8, nine runs in three groups) against
   without 29.2 (25.6 to 30.3, three runs), ratio 0.977, well inside the run-to-run spread (host load 10 to 59 across
   these runs). Keep the counters. STAGE_SYNC set to 1 did not lower throughput either (B 10.61 against 9.88, P 15.47
   against 14.80, one batch).
6. **Not reproduced: the earlier set P rates.** Log E4 and F1 recorded 45 to 49 img/s of set P ingest with two instances
   (E4: concurrency 16, batch 16; GPU 78 to 87 percent, Triton CPU about 8.5 cores). The best repetition here is 33.1 img/s
   (4 batches, first repetition), the median 25.0, GPU 73 and 58 percent, Triton 4.4 cores, under host load of 14 to 26. The
   repetitions fall within a session (33.1, 25.0, 23.5; API CPU 0.07, 0.20, 0.25), so a state that builds up in the API or
   the shared host is possible; it was not isolated. Compare stage shares across rows, not these img/s with the old log.
   Set A (13.1 img/s at one batch) is likewise below log F3 (16.2) and E1's 16.6 for the same reason (one batch, load 27).

### SP0.4 Ranking for the ingest lane (SP1 and later)

**Superseded by SP0.7:** the fall past 4 batches was device-memory exhaustion (CuPy and nvJPEG out-of-memory, images falling back to the CPU decode), not CUDA-call overhead. The ranking below is kept as measured; read it together with SP0.7.

The data ranks, by measured share on the GPU route: (1) host-side CUDA-call overhead and serialisation inside
`gpu_image_prep` under concurrency (SP-new, below); (2) nvJPEG decode; (3) the PE embedding wait; (4) crop JPEG encode and copy
(the item SP3 targets, whose planned gain is bounded by its 9.5 to 21.6 percent of the model's stage time); then OpenSearch writes.
On the CPU route: the PE crop embedding, then OpenSearch writes. SP1 (cluster pass) does not touch ingest throughput; SP2 (region
stage and embed-pass measurements) and SP3 follow the data only partly. The plan's order is not contradicted for SP3, but a
new first item ahead of it is: cut the number of small CUDA calls per image in `gpu_image_prep` (one fused letterbox and
square kernel set, one device-to-host copy for all of an image's crops) and measure CUDA MPS and the instance count at 4
batches, gated by the SP4 bar (set P at least 29 img/s, GPU above 75 percent) at `--clients 4`. See `docs/reviews/sp0_review.md`.

### SP0.5 What the numbers do not say

- Every row ran under foreign load. Noisy repetitions (load above 16 at the start) were not repeated: set A repetitions 1
  and 2, set P (1 batch) 2 and 3, set B (4 batches) 2 and 3, set P (4 batches) 1, set P (8 batches) 2 and 3. The conclusions use
  shares, the 4-versus-8 batch reversal and the counters on/off ratio, which hold across quiet and noisy repetitions; the
  absolute img/s are not release numbers.
- Four batches is one setting; 2, 3 and 6 batches were not measured, so the peak is somewhere between 4 and 8.
- The STAGE_SYNC rows are attribution only, at one batch only. Sync rows at 4 batches and the CPU route at 4 batches were not run.
- Set A selects 1,946 of 2,000 pinned images (`select_baseline_set.py local` keeps what decodes above its minimum size).

### SP0.6 Bugs found by this run (all in the SP0 code, fixed on the branch except the two issues)

1. The stage counters never reached Triton's metrics port and could stall the model: issue #390 (three Triton 2.70
   Python-backend facts, each reproduced on a toy model; fix `77500a12`).
2. `ingest_profile.py table` took the median GPU utilisation over every sampled GPU and missed Triton's `_total` series
   names (GPU row 0, empty Triton table); the row names made project slugs the API refuses; `run_baseline.py` posted
   batches one after another (fixes `553ade23`, `6224897f`).
3. The reused install had an empty `pe_image_encoder/1/` and `/health` still answered `ready`: issue #391.
4. The throughput fall past 4 batches and the API CPU growth: issue #392.

### SP0.7 Why throughput fell past 4 batches: device memory, not CUDA calls (issue #392; run 2026-10-30, GPU 0)

Same stack and protocol as SP0 (`optest2`, GPU 0 only, release 0.6.0 images, this branch's `gpu_image_prep`, set P 3,000 sample
copies, 3 repetitions, cache drop, 500 ms GPU samples; row names below are the project-slug rows). Pre-registered in
`docs/benchmarks/v070/sp0/gpu_prep_concurrency_matrix.md` (commit `dc522051`) before any run. GPU placement was checked at runtime
(`nvidia-smi --query-compute-apps` against the container pids): the Triton and API processes sat on GPU 0 only. Raw JSON, error
strings and per-process memory: `docs/benchmarks/v070/sp0/gpu_prep_concurrency/` (set P rows; the `metrics.api.counters` of
each report is cut to the three `op_ingest_*` series, everything else is as written), tables
`docs/benchmarks/v070/sp0/sp0_tables_headroom.md`. Host load was 5 to 147 (other sessions); noisy repetitions are marked.

**Deviations from the pre-registered matrix.** (1) Order changed after the design review found fallbacks in the SP0 counters:
E1 (headroom A/B, fresh stack, 8 batches first) ran first; the variants then ran on set P at 4 batches, not on set B, on the
headroom stack. The set B matrix ran only for the first rows (C0, instance count 1 and 3, MAX_IN_FLIGHT 32 and 16 at 4
batches) on the full-model stack, where it is confounded by memory (below), and was stopped. (2) E-b (CUDA MPS) not run: it needs a
root MPS daemon, owner decision pending. (3) E-d used per-thread CPU from `/proc/<pid>/task/*/stat` (6 s windows), not py-spy.
(4) `Dynamic batching` is not available for this model (`max_batch_size: 0`, decoupled).

#### SP0.7.1 E1: headroom against full model set (fresh Triton per phase, 8 batches first)

"Headroom" = the eight models ingest does not call unloaded (`arcface`, both MobileCLIP towers, `scrfd`, the three OCR models, one
demo model); "full" = the default load list. Fallback = `cpu_decoder` + `cpu_call_failed` over photos sent to the model.

| Row | img/s median (min to max) | GPU util mean % | Fallback % per repetition | API CPU-s / image | Card peak (MiB) |
|---|---|---|---|---|---|
| 8 batches, headroom | 48.5 (47.2 to 48.7) | 96 | 0 / 0 / 0 | 0.062 | 26,372 |
| 4 batches, headroom | 38.5 (36.0 to 39.1) | 80 | 0 / 0 / 0 | 0.059 | 26,404 |
| 8 batches, full | 40.4 (29.8 to 43.0) | 83 | 0.6 / 3.1 / 17.3 | 0.128 | 48,558 |
| 4 batches, full | 30.7 (22.5 to 32.1) | 63 | 1.4 / 6.6 / 22.9 | 0.154 | 48,558 |

- Card memory after startup: `tritonserver` 36,486 MiB with the default list, 8,864 to 8,896 MiB with the eight models unloaded
  (27.6 GB freed in total; per model not measured). Each `gpu_image_prep` stub holds 268 MiB idle and peaks at 8.5 to 9.1 GB under load
  with headroom (the config budgets 8,704 MiB), 6.9 to 7.5 GB when the card is full. Sum of peaks: 26.5 GB with headroom, 50.3 to 51.2 GB
  (over the 49.1 GB card) with the full set. Table: `e1_per_process_peaks.md`.
- Error strings, full set, 8 batches: 208 `cpu_call_failed`, all CuPy `Out of memory allocating N bytes (allocated so far ~2 GB, limit
  set to 4 GB)`, that is a device OOM below the pool limit; 320 decode exceptions `OutOfMemoryError` and 82 `NvJpegError: TransferToDevice
  returned nvjpegStatus 5` (nvJPEG's device allocation failing). 4 batches, full: 415 call failures, 383 + 93 decode ones, and four
  Triton `BackendMemory::Create ... failed to allocate GPU memory: out of memory` lines. With headroom: none of these, in either row.
- Session age: on the full set the fallback share grows within one fresh stack (0.6, 3.1, 17.3 percent; 1.4, 6.6, 22.9) and the rate
  falls with it (43 to 30, 32 to 23 img/s). With headroom the repetitions are flat (47.3, 48.5, 48.7).
- The "fall past 4 batches" of SP0 is the session-age effect plus the full card, not a property of 8 batches: with headroom 8 batches is
  faster than 4 (48.5 against 38.5 to 41.6), GPU 96 percent. API CPU is 0.06 CPU-s per image with no fallbacks, and rises with the
  fallback fraction (0.88 at 70 percent): the API-side growth of #392 is the CPU decode of the fallen-back photos, nothing else.

Verdict: the memory hypothesis is confirmed. Set P at `--clients 4` with headroom meets the SP4 bar (29 img/s, GPU above 75
percent, fallbacks 0): see the controls below (39.6 to 41.6 img/s, GPU 83 to 90 percent). On the full default stack it does not.

#### SP0.7.2 Controls and load scaling (headroom stack, `count: 2`, defaults)

| Row | img/s median (min to max) | GPU util mean % | Notes |
|---|---|---|---|
| 4 batches, control, four fresh-stack runs | 41.6 (40.7 to 41.9), 39.7 (33.8 to 41.6), 39.9 (38.9 to 40.9), 39.6 (37.6 to 43.0) | 83 to 88 | fallbacks 0 |
| 1 batch at a time (1 repetition) | 16.9 | 35 | |
| 2 batches | 23.0 | 51 | |
| 8 batches | 48.5 (47.2 to 48.7) and 48.2 (46.2 to 48.3) | 96 | |
| 12 batches | 48.4 | 97 | GPU-bound ceiling about 48 img/s |

Per-image `gpu_image_prep` stage time with headroom at 8 batches: letterbox 22 ms, embed_crops 89, crop_copy 35, against 21, 53, 29 at one
batch in SP0: the "5 to 33 times" growth of SP0.3 is gone, except `queue` (53 percent of stage time: callers outrun the GPU). At 4 batches the
inflation is smaller than in SP0 but still there (letterbox 41 to 51, embed_crops 173 to 205 ms), see SP0.7.4.

#### SP0.7.3 E-a: instance count and in-flight limits

Set P, headroom stack, 3 repetitions. Count 1 and 3 and the MAX_IN_FLIGHT, WORKERS rows of the first pass ran after one stub that the
reload left resident (below), so their card peak is up to 8 GB too high; none of their repetitions fell back.

| Row | img/s median (min to max) | GPU util mean % | Card peak GiB | Reps noisy |
|---|---|---|---|---|
| count 1, 4 batches | 32.1 (30.9 to 32.4) | 74 | 18.1 | 2 |
| count 1, 8 batches | 31.5 (30.3 to 31.7) | 75 | 19.1 | 2 |
| count 2 (control), 4 / 8 batches | 39.6 to 41.6 / 48.2 to 48.5 | 83 to 88 / 96 | 25.5 | 0 to 3 |
| count 3, 4 batches | 38.7 (37.5 to 43.5) | 82 | 33.0 | 1 |
| count 3, 8 batches | 46.2 (45.6 to 47.1) | 95 | 35.4 | 3 |
| MAX_IN_FLIGHT 32, 4 batches | 39.2 (38.5 to 40.9) | 83 | 25.7 | 3 |
| MAX_IN_FLIGHT 16, 4 batches | 45.1 (44.7 to 45.8) and 46.8 (46.4 to 47.1) | 93 to 95 | 25.6 | 3 |
| MAX_IN_FLIGHT 16, 8 batches | 48.0 (47.8 to 48.2) | 97 | 25.5 | 0 |

- A single instance caps near 32 img/s with the GPU 75 percent busy; two reach the GPU ceiling (about 48) at 8 batches; a third adds nothing
  (46.2 against 48.2) and costs 8 GB. Count 1 still passes the plan's bar (29 img/s) with a 18 to 19 GiB card peak, which is the
  24 GB card option (with the unused models unloaded).
- MAX_IN_FLIGHT 16 beats the default 96 at 4 batches by 13 percent (45.1 against 39.6 to 41.6; the ranges do not overlap with the
  three controls of the same session) and ties at 8 batches. The pre-registered "clear win" test is met.

#### SP0.7.4 E-c: other Python-backend settings (set P, 4 batches, headroom)

| Row | img/s median (min to max) | GPU util mean % | Card peak GiB | Reps noisy |
|---|---|---|---|---|
| WORKERS 16 | 46.8 (46.0 to 46.9) and 46.3 (45.5 to 47.5) | 94 to 95 | 25.8 | 0 to 1 |
| WORKERS 8 | 44.3 (41.3 to 46.2) | 95 | 33.3 (stale stub) | 3 |
| LANES 8 | 39.2 (38.1 to 42.6) | 85 | 22.5 | 0 |
| ENCODE_THREADS 16 | 39.9 (38.1 to 41.0) | 86 | 25.6 | 0 |
| Python backend shm 64 MiB default and growth | 40.2 (38.9 to 40.8) | 82 | 25.6 | 3 |

WORKERS 16 gives the same gain as MAX_IN_FLIGHT 16 (+13 to +17 percent at 4 batches, GPU 94 to 95 percent); LANES, ENCODE_THREADS and the shm
sizes do nothing. Stage times with WORKERS 16 against the control at 4 batches: letterbox 5 against 41 ms, embed_crops 51 against 173, crop_copy 24
against 57 (`queue` 325 against 86 ms, the wait moved in front of the workers). So at 4 batches the earlier inflation of the "CUDA-call"
stages comes from more concurrent images than the GPU and the allocator can interleave; limiting the images in flight inside the process to
16 removes it. At 8 batches the GPU is the limit either way.

#### SP0.7.5 E-d: threads of the Triton Python stubs (set P, 4 batches, headroom, 19 windows of 6 s)

While a repetition ran, each of the two `gpu_image_prep` stub processes used 3.0 to 4.5 cores (about 40 threads each), the Triton server process
about 1.0 core, and the busiest single thread 20 to 25 percent of a core. No thread is saturated and a stub is not confined to one core, so
the Python process is not GIL-bound at this load. (py-spy was not used.) Raw samples: `gpu_prep_concurrency/threads/`.

#### SP0.7.6 E-e: API CPU per image against clients

0.088, 0.084, 0.064, 0.062, 0.066 CPU-s per image at 1, 2, 4, 8 and 12 batches with headroom (no fallbacks): flat, so the API is not the limit.
With fallbacks it is 0.06 + about 0.9 times the fallback fraction (set B count 3, 8 batches: 70 percent fallbacks, 0.88).

#### SP0.7.7 Set B, full-model stack (first rows only)

Set B (400 photos measured, 47 of them progressive and 2 truncated, which take the CPU decode by design, so the GPU-route denominator is 351):
control 32.4 (29.2 to 33.1) at 4 and 33.5 (29.8 to 35.5) at 8 batches with 1.7 to 10.8 percent fallbacks; count 1: 28.8 and 34.0 with 0
fallbacks (card peak 43.8 GiB); count 3: 23.1 and 23.0 img/s with 29 to 76 percent fallbacks (peak 47.4 GiB, API CPU 0.51 to 0.88);
MAX_IN_FLIGHT 32: 29.2 and 35.0 (2 to 5 percent); MAX_IN_FLIGHT 16: 27.7 at 4 batches. These are the fallback-confounded numbers the
set B matrix would have produced; they confirm that adding a third instance on a full card makes things worse by exactly the
fallback fraction. Raw: `artifacts_local/bench/v070/sp0/exp/` (main checkout).

#### SP0.7.8 What it means, and what to change

Supported: memory exhaustion (device OOM in CuPy and nvJPEG, silent fallback to the API's CPU decode, growing with session age).
Not supported as the cause of the collapse: GIL (no saturated thread, 3 to 4.5 cores per stub), API-side limits (0.06 CPU-s per image without
fallbacks), instance count above 2. Real but secondary: too many images in flight inside one instance (letterbox and crop stages inflate
4 to 10 times at 4 batches with the default 32 workers and 96 in flight). Untested: CUDA MPS (E-b), combinations of the settings.

Changes, with memory cost on this 48 GB card (A6000):

1. Acceptance stack: do not keep models ingest does not call resident next to `gpu_image_prep`. Unloading eight of them frees 27.6 GB
   (`tritonserver` 36.5 to 8.9 GB) and moves set P at 4 batches from 22.5 to 32.1 img/s with 1 to 23 percent fallbacks to about 40 with none,
   and at 8 batches from 29.8 to 43.0 to 48.5. No config.pbtxt change; this is a load-list or profile decision (issue #393).
2. `models/gpu_image_prep/config.pbtxt`: `WORKERS` 32 to 16 (or `MAX_IN_FLIGHT` 96 to 16): +13 to +17 percent at 4 batches (about 46 against 40),
   no change at 8 batches, no change in memory (card peak 25.5 to 25.8 GiB, stubs 8.5 to 9.2 GB each). Not tested together or at other values
   between 8 and 32 with quiet host load; confirm WORKERS 16 against 8 and 24 on a quiet host before committing.
3. Keep `count: 2` on a 48 GB card. On 24 GB: count 1 (about 32 img/s, GPU 75 percent, card peak 18 to 19 GiB with the unused models
   unloaded) meets the 29 img/s bar; two instances need about 26 GiB. A 12 GB card (#42) cannot hold the detector, the PE encoder and one stub
   (about 9 GB): GPU decode stays off there.
4. Fail closed (design C1): the 8,704 MiB budget per instance is exceeded by up to 5 percent (9,144 MiB seen); a memory-pressure wait
   or a load-time reservation would turn the silent fallback into a counted wait.

SP4 bar on set P at `--clients 4`: met on the headroom stack (39.6 to 41.6 img/s, GPU 83 to 90 percent, 0 fallbacks; 45 to 47 img/s, 93 to 95 percent
with `MAX_IN_FLIGHT` or `WORKERS` 16); not met on the full default stack (30.7, GPU 63 percent, fallbacks up to 23 percent).

#### SP0.7.9 Harness notes

- `unload` then `load` of `gpu_image_prep` once left the previous load's stub process resident (8,024 MiB for the next five configurations,
  card peak up to 33.5 GiB); recreating the Triton container cleared it. Rows affected: MAX_IN_FLIGHT 16 (first pass), WORKERS 16 and 8 (first pass), LANES 8
  (first pass) and ENCODE_THREADS 16 (first pass). Their repetitions had 8 GB less headroom and still had no fallback.
- The first-pass LANES 8 (29.4) and ENCODE_THREADS 16 (26.6) rows ran at host load 45 to 147 and are not usable; the repeats in the second session (load 14 to 17)
  gave 39.2 and 39.9.
- All Triton restarts used `docker compose -p optest2 restart` or `up -d` (no `pkill`).



### SPF Ingest-speed fixes: memory budget, config, load profile (issues #392, #393; run 2026-10-30, GPU 0)

Same stack and protocol as SP0.7 (`optest2`, GPU 0 only, release 0.6.0 images, this branch's `gpu_image_prep` and API code mounted,
set P 3,000 sample copies, 3 repetitions unless noted, page cache dropped, 500 ms GPU samples, load average recorded). The host was
never quiet: the 1-minute load was 13 to 138 (other sessions' test suites), so every repetition is marked noisy by the harness's
load-16 rule and the same configuration varied by up to 40 percent between sessions (32/96: 38.2, 35.6 and 21.2 img/s at load 43, 73 and
138). Rows are therefore compared only inside one session. Fallback % = `cpu_decoder` + `cpu_call_failed` + `cpu_gpu_memory` +
`cpu_over_budget` over the photos sent to the model. Raw files: `docs/benchmarks/v070/spf/`.

#### SPF.1 C1: byte budget, new statuses, fail at load

Offline tests (fail first against the SP0.7 model: 18 of 27 stage-metrics tests failed): budget FIFO and no deadlock with 64 threads,
`gpu_memory` at every allocating stage including stored-box mode, `over_budget`, the wait and its timeout, no output change under a tight
budget, reservation at load and the load refusal. Live parity of the whole model (200 public images, set A and set B, direct
`gpu_image_prep` captures, `scripts/bench/gpu_prep_parity.py`): old model against old model at concurrency 1 differs in 8 of 193 images
(the detector's run-to-run variation), old against new differs in 11; in every one of those images the detector boxes differ, and where the
boxes are equal every output (sizes, letterbox, classes, blur sums, crop JPEG bytes) is byte-identical. Whole-frame embedding cosine at least
0.99999999 at concurrency 1. Concurrency 8: old against old 11 differing images, old against new 8. The arithmetic is unchanged
(`spf/parity.txt`).

Default model set (36.5 GB resident), shipped config of SP0.7 (32 workers, 96 in flight), two instances:

| Row | img/s median (min to max) | GPU % | Fallback % per rep | API CPU-s/img | Card peak GiB |
|---|---|---|---|---|---|
| 8 batches | 23.5 (23.2 to 24.0) | 59 | 2.5 / 8.2 / 9.4 | 0.133 | 47.4 |
| 4 batches | 25.5 (18.5 to 27.6) | 57 | 1.9 / 6.5 / 23.2 | 0.159 | 47.4 |
| 4 batches, one instance | 32.8 (30.8 to 34.4) | 75 | 0 / 0 / 0 | 0.058 | 45.0 |

On the default stack the budget does not remove the fallbacks, it names them: the card has 12.6 GB left after Triton's 36.5 GB, two
instances reserve 2 x 2.3 GB at load and need about 6 to 7 GB more each under load (nvJPEG buffers, CuPy scratch, BLS tensors), which is
more than is left. The failures now arrive as `cpu_gpu_memory` (2.5 to 9 percent) instead of `cpu_decoder` and `cpu_call_failed`, still
decaying with session age. Both instances loaded, so the load-time refusal did not fire: the reservation covers the frames, not the
nvJPEG buffers. One instance fits (32.8 img/s, GPU 75 percent, no fallback).

Ingest load profile (SPF.3), two instances, same config:

| Row | img/s median (min to max) | GPU % | Fallback % | API CPU-s/img | Card peak GiB |
|---|---|---|---|---|---|
| 8 batches | 47.9 (47.5 to 48.4) | 96 | 0 | 0.053 | 25.6 |
| 4 batches | 42.0 (41.3 to 42.7) | 88 | 0 | 0.057 | 25.7 |

The wait for the budget is not binding there: `op_gpu_prep_stage_seconds{stage="budget"}` is 0.2 s over 8,976 photos at 16 workers
(13.8 s over 8,978 at 32 workers on the full card).

#### SPF.2 Config: WORKERS and MAX_IN_FLIGHT (ingest profile, set P, 4 batches)

| Session | Row (workers/in flight) | img/s median (min to max) | GPU % | Load max |
|---|---|---|---|---|
| screening, 2 reps | 32/96 | 38.2 (37.9 to 38.5) | 80 | 43 |
| | 24/96 | 40.5 (40.4 to 40.6) | 83 | 53 |
| | 16/96 | 40.4 (37.5 to 43.3) | 86 | 62 |
| | 12/96 | 42.0 (41.1 to 43.0) | 86 | 54 |
| | 16/32 | 41.8 (38.4 to 45.3) | 87 | 59 |
| | 16/16 | 37.5 (36.4 to 38.6) | 80 | 52 |
| | 24/24 | 43.5 (42.8 to 44.3) | 89 | 99 |
| | 12/12 (1 rep) | 44.2 | 90 | 28 |
| confirmation, 3 reps | 32/96 | 35.6 (34.6 to 38.2) | 80 | 73 |
| | 16/32 | 40.5 (37.0 to 41.3) | 86 | 63 |
| | 24/24, 12/12, 32/96 repeat | 31.9, 30.7, 21.2 | 70, 70, 49 | 120, 110, 138 (unusable) |

16/32 beat 32/96 in all three comparisons that ran at similar load (+9, +14 percent here, +13 to +17 percent for 16/96 in SP0.7.4); the
lower values (12/12, 24/24) sometimes looked better and sometimes collapsed with the host load, so no ranking among them is claimed.
Default set to 16 workers, 32 in flight (no memory change: card peak 23.4 to 25.8 GiB in every row).

#### SPF.3 The load list (issue #393): memory per model and the profile

GPU 0 memory after loading each model on top of the three-model ingest profile (8,813 MiB; unloading each returned it to within 18 MiB):

| Model | Used by | Delta MiB |
|---|---|---|
| `scrfd_10g_bnkps` | /faces, /analyze, core /ingest, /search/face | 6,846 |
| `arcface_w600k_r50` | same | 2,754 |
| `mobileclip2_s2_image_encoder` | /embed image routes, /search image and object, /analyze, core /ingest | 3,454 |
| `mobileclip2_s2_text_encoder` | /embed/text, /search/text | 1,430 |
| `paddleocr_det_trt` | /ocr, /search/ocr, /analyze, curation text reading | 1,642 |
| `paddleocr_rec_trt` | same | 11,572 |
| `ocr_pipeline` | /ocr (calls the two above) | 0 |
| total | | 27,698 |

The 36.5 GB of the default list is therefore the three needed models (8.8 GB) plus 27.7 GB, of which the OCR recognizer alone is 11.6 GB
(its dynamic-shape engine). This answers #393's 36.5 against 23.1 GB: optest1 holds fewer of these.

Profile check: every public endpoint of the core API (38 routes with a body or no path parameter, `spf/endpoints_*.txt`) was called with
three public COCO images on the default list and on the profile. The status codes are identical (200, except the face routes' 400 "no face
detected" on photos without faces) with one difference, and it is the card, not the profile: on the default list with two instances the
card is full and `/ocr/predict` fails with "Failed to open the cudaIpcHandle: out of memory"; on the profile it answers 200. The first run
of the sweep found a real bug in this work (`ocr_pipeline` loaded without the OCR models it calls, `/ocr/predict` 502), fixed in
`efe067ca`. First calls on the profile cost 2.3 s (`/embed/text`) and 17.8 s (`/analyze`, five models) once.

#### SPF.4 Acceptance, set P (config 16/32, `--clients` 4 and 8, 3 reps, host load 13 to 35 during these rows)

| Stack | Clients | img/s median (min to max) | GPU % | Fallback % per rep | API CPU-s/img | Card peak GiB |
|---|---|---|---|---|---|---|
| ingest profile, 2 instances | 4 | 48.3 (45.3 to 49.0) | 94 | 0 / 0 / 0 | 0.052 | 25.4 |
| ingest profile, 2 instances | 8 | 48.3 (40.8 to 48.8) | 93 | 0 / 0 / 0 | 0.056 | 25.8 |
| default list, 1 instance | 4 | 38.9 (32.9 to 39.4) | 83 | 0 / 0 / 0 | 0.060 | 45.3 |
| default list, 1 instance | 8 | 38.6 (38.4 to 40.2) | 86 | 0 / 0 / 0 | 0.059 | 44.9 |
| default list, 2 instances | 4 | 35.9 (32.4 to 39.3) | 72 | 0.6 / 2.2 / 5.6 | 0.081 | 47.4 |
| 24 GiB emulation (ballast 24.8 GiB), ingest profile, 1 instance | 4 | 38.1 (35.1 to 39.1) | 84 | 0 / 0 / 0 | 0.059 | 42.1 total with ballast |

Bars (SP4: at least 29 img/s with GPU above 75 percent at 4 clients; fallbacks at most 0.5 percent; API CPU at most 0.08; no drop at 8):
met on the ingest profile with two instances, and on the default list with one instance. Not met on the default list with two instances
(fallbacks up to 5.6 percent, GPU 72, card at 47.4 of 48.0 GiB, i.e. full): that is the shipped `full` profile, see the review. The 24 GB case
passes with the ingest profile and one instance. C2 (batched kernels) is deferred: the bar is met without it.
