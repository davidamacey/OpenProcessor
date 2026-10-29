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
