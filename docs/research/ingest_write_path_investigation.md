# Ingest write path investigation (bulk ingest looked write- and queue-bound)

Branch `perf/ingest-write-path` (from `origin/main` c5c1597e). Written 2026-10-16 so a fresh agent can
continue. Everything below was measured on the shared dev host (load average 21 to 36 from other
projects), against throwaway OpenSearch containers named `optest-opensearch`, image
`opensearchproject/opensearch:3.6.0` (the image opmono runs), 2 GB heap (opmono's setting), on
127.0.0.1:4981. The running `opmono` stack was only read (settings, stats, 500 public COCO templates).
Both throwaway data directories and the container were removed afterwards.

## 1. The observations

The scale-ladder runs (section "Scale ladder: 10k and 100k" of `docs/PERFORMANCE.md` on branch
`release-0.6.0`; samples in `artifacts_local/scale/*.jsonl`): 100,000 public Open Images at 36 images/s, 100,000
private 20 MP photos at 21 images/s, one A6000, VLM off, OpenSearch and the source images on the NAS. The
run record's `stage_s` read `opensearch_write` about 50 percent, `batch_requests` about 35 percent and
`admission_wait` about 14 percent (20 MP run), with `embed` and `decode` near zero.

## 2. Verdict

1. **`stage_s` is not a time budget and says nothing about OpenSearch.** Each stage is wall time summed
   over concurrent callers: `batch_requests` is the whole request (it contains the other stages) and, with
   16 requests always in flight, equals 15.6 x elapsed by Little's law whatever the bottleneck is. In the
   20 MP run `opensearch_write` is the time an image waits for a shared coalesced flush plus the time the
   event loop needs to get back to it, and the loop was blocked by file reads (4). The GPU-decode path was
   not timed at all (`decode` 15 s, `embed` 35 s over 4,761 s), so the wait landed in the neighbouring stage.
2. **OpenSearch is not the limit at these rates, on NVMe or on the NAS.** The runs need about 200 documents/s
   (36 images/s x 4.9 crops + images) and about 150 (21 x 2.4 + images). One node takes 2,500 to 3,300
   documents/s with 4 writers on NVMe, 2,500 to 2,800 on the NAS (isolated), 1,100 on the NAS while
   other readers share the array. That is 5 to 20 times headroom. The small-image run is GPU-embed bound
   (94 percent mean GPU utilisation, 177 to 187 crops/s), which is the ceiling PERFORMANCE.md already states.
3. **The NAS matters for the tail, not the mean.** Isolated, NAS-hosted data costs 18 percent of write
   speed (4 writers) and 34 percent (8 writers). Under mixed load (two readers streaming 167 MB/s of source photos
   from the same array) NAS-hosted indexing falls to 1,106 docs/s with bulk p95 14.7 s (NVMe: 2,511 docs/s,
   p95 2.6 s) and flush time per run 26 s (NVMe 4 s). The leg 1 collapse of the 20 MP run (23 to 9 images/s over
   the last 10 minutes) coincides with a merge storm (section 5), a plausible NAS contention episode.
4. **The 20 MP run was limited upstream of the GPU by code, and two defects are fixed here.**
   (a) The admission gate charged a GPU-decoded photo its *decoded* size, although the GPU path never decodes
   it in the API process (`ingest_gpu_decode.py`: only vectors, boxes and JPEG crops come back). A request of 16
   photos of 20 MP was charged about 915 MiB of the 1,024 MiB per-process cap while holding about 110 MiB of
   encoded bytes. A second request landing on the same worker waited for the first to finish: that is
   `admission_wait` (13,919 s of 33,999 s request time, 41 percent). Connections stay on the worker that
   accepted them, so about 12 of 24 workers carry 16 connections. The queue model
   (`scripts/bench/gate_admission_sim.py`, 24 processes, 16 requests in flight) gives 30 percent wait and
   8.6 images/s against 12.2 with no gate: the same order as the observed 41 percent. It is a model, not
   a measurement of the stack.
   (b) `/ingest/batch` read its files with `Path.read_bytes()` one after another **on the event loop thread**.
   16 photos of 5.5 MB from the NAS cost 0.6 s of blocked loop per request (one reader: 155 MB/s, 28 files/s;
   16 readers: 700 MB/s, 126 files/s, measured on 120 sampled files of the private tree, aggregates only). A blocked loop delays
   every other request on that worker and every awaited OpenSearch or Triton reply, which inflates exactly
   the stages that were read as "OpenSearch time".
5. **Not proven, needs the live stack.** That fixing (4a) and (4b) lifts the 20 MP rate: the effect is
   predicted by the model and by GPU utilisation of 51 percent with API CPU at only 100 to 200 percent (one to two cores out of 48:
   the API was waiting, not computing), but the stack was off limits. Re-run `hires-100k` to confirm.

## 3. Method and raw numbers

Driver: `scripts/bench/os_write_ceiling.py`, extended here with `--source-url` (templates and mappings read
from another instance), `--repo-mappings` (the repo's `_items_body`/`_images_body`, Lucene HNSW m=16
ef_construction=128, one shard, no replica), per-arm index `settings`, and per-run deltas of node counters
(merge, flush, refresh, index time, GC, throttle) and host IO stall (`/proc/pressure/io`). Documents: 60,000 crop
documents plus 9,375 image documents per run (6.4 crops per image), real templates from the public
COCO bench project with new random 1024-d (items) and 1024/512-d (images) vectors built by the script's noise
model around pool vectors. 2 repetitions per arm, means below. Raw files:
`docs/benchmarks/write_path_nvme_2g.json`, `write_path_nas_2g.json`,
`write_path_nas_with_nas_reader.json`, `write_path_nvme_with_nas_reader.json`.

### 3.1 Data directory NVMe against NAS, and the bulk-ingest levers

| Arm | NVMe docs/s | NVMe bulk p95 s | NVMe flush s/run | NAS docs/s | NAS bulk p95 s | NAS flush s/run |
|---|---:|---:|---:|---:|---:|---:|
| shipped (1000 x 4 writers, refresh 1 s) | 3,199 | 1.58 | 3.4 | 2,631 | 2.00 | 14.8 |
| `translog.durability` async (30 s sync) | 3,302 | 1.46 | 2.5 | 3,009 | 1.63 | 15.0 |
| 8 writers | 5,334 | 1.78 | 2.8 | 3,542 | 3.27 | 14.0 |
| 1 writer | 645 | 2.58 | 1.3 | 575 | 2.90 | 4.9 |
| `refresh_interval` 30 s | 2,312 | 3.29 | 1.1 | not run | | |
| refresh -1 plus translog async (bulk mode) | 2,283 | 3.21 | 1.3 | not run | | |
| merge scheduler 1 thread | 2,981 | 1.80 | 5.4 | not run | | |
| merge scheduler 4 threads | 2,969 | 2.09 | 4.8 | not run | | |

GC per run 0.3 to 0.5 s of a 20 to 30 s run on every arm: the 2 GB heap was not binding here (the live node
peaked at 79 percent during a 100k run, 30 percent now). `node_merge_throttled` 0 on all arms.

### 3.2 Mixed load: source images read from the same NAS while indexing

Two reader threads read random 5.5 MB JPEGs from the NAS tree with the page cache dropped per file
(`posix_fadvise DONTNEED`), 167 MB/s, while the shipped arm runs (one repetition each):

| Data directory | docs/s | bulk p50 / p95 s | flush s/run | host IO full-stall s |
|---|---:|---:|---:|---:|
| NVMe | 2,511 | 1.43 / 2.59 | 4.0 | 0.9 |
| NAS | 1,106 | 1.84 / 14.74 | 26.1 | 4.9 |

Single observations on a shared host; read them as an effect size, not a constant.

### 3.3 What the levers say

- Refresh off or 30 s is **slower** (-28 percent), same as the earlier probe 5: on an HNSW index the graph for a segment is built
  when the segment is written, and 1 s refreshes spread that over many threads and small graphs; one big flush
  builds one graph serially. A "bulk mode" with refresh -1 is therefore a loss and was not implemented.
- Translog async: +3 percent (NVMe), +14 percent (NAS), within noise on this host, and it gives up the per-request
  crash guarantee. Not implemented. The measured cost of request durability is small.
- Merge scheduler threads: no effect on the load itself (merges were 3 s of CPU in 21 s).
- More writers help up to the node's CPU (8 writers: +67 percent on NVMe, +35 percent on NAS), but the product
  needs 5 to 10 percent of one writer's rate; the 4-writer default stays.
- Bulk size and `ef_construction`: already measured in probe 5 (bulk 500/1000/5000 within 25 percent; 128 vs 512
  -38 percent node CPU); the repo default is already 128 on the Lucene engine.

### 3.4 The live node's own counters (read-only, opmono, after the scale runs)

4.24 M documents indexed, indexing time 8,451 s; **merges: 4,918 totaling 68.2 GB of merged segments for 4.24 M indexed documents**, 5,715 s of merge time of which 927 s throttled by the auto-throttle; **flush: 451 flushes taking 1,753 s (3.9 s
each)**; refresh 49,517; heap 30 percent now; write thread pool 0 rejected, queue 0. Index settings carry
`merge.policy.floor_segment` 2 MB and `max_merge_at_once` 10 (set by the k-NN plugin, not by this repo). A 3.9 s
flush average is the fsync cost of the NAS (NVMe in section 3.1: 0.8 to 1.1 s per flush).

## 4. Read path

`curation_ingest_batch` (`src/routers/curation/ingest.py`) read every file with `Path(item.path).read_bytes()`
sequentially inside the coroutine. Measured on the private photo tree (120 files, mean 5.5 MB, cache dropped per file):

| Readers | files/s | MB/s |
|---:|---:|---:|
| 1 | 27.9 | 155 |
| 4 | 78.8 | 437 |
| 16 | 126.0 | 699 |

The 20 MP run moved 143 MB/s in total, i.e. the single-reader rate of the NAS per worker is about what the whole run
needed, and while a worker read, its loop did nothing else. Pillow decode of a 20 MP JPEG on the CPU is 0.2 to 0.4 s
per image per core (synthetic 20 MP image, 19 MB, 433 ms on one thread, 11 images/s with 8 threads), which is why
the 3 MP GPU gate exists; the host does not decode these photos when the GPU route is available.

## 5. Throughput collapse at the end of leg 1 (20 MP run)

From `hires-100k.jsonl`, 5-minute buckets (images/s, GPU2 util, API CPU, OpenSearch CPU, segments, merges in flight):

| Minute | img/s | GPU % | API CPU % | OS CPU % | segments | merges |
|---:|---:|---:|---:|---:|---:|---:|
| 20 | 18.6 | 57 | 138 | 50 | 44 | 0 |
| 25 | 18.5 | 66 | 119 | 99 | 57 | 0.1 |
| 30 | 16.5 | 46 | 119 | **422** | 128 | **1.0** |
| 35 | **8.3** | 75 | 97 | 150 | 52 | 0.2 |
| 40 | 18.1 | 22 | 101 | 49 | 53 | 0 |

The slowest bucket follows a merge that took the segment count from 128 to 52 (OpenSearch at 4 to 6 cores). Leg 2 had one
similar merge at minute 30 (625 percent) and dipped to 20 img/s for one bucket, then 31. API memory rose to 50 GiB in
the slow bucket. Together with 3.2 this supports "a merge storm on a NAS shared with the reads", not proven (no
IO counters were sampled). A restart did not need to cure it; the merge simply finished.
API CPU is only 1 to 2 cores out of 48 throughout, and the host never exceeded 69 percent: the 20 MP run was waiting, not computing.

## 6. What was implemented (commits on this branch)

- Admission gate charges a GPU-decoded photo its encoded size (`gpu_route` in `ingest_gpu_decode.py`, used by
  `ingest_batch.py`). Test: `tests/test_ingest_gate_gpu_route.py` (failed before: 12 MiB charged instead of 2).
- Files of a batch are read in parallel on a per-process pool (`src/utils/file_read.py`, `OP_INGEST_READ_THREADS`,
  default 8), off the loop, for `/ingest/batch` and `/ingest/image`. Test: `tests/test_ingest_file_read.py`.
- Two stage timers: `file_read` and `gpu_decode`, so the next `stage_s` covers the work that was invisible.
- `OP_OPENSEARCH_DATA` in `docker-compose.yml` (volume name or host path; default unchanged).
- `scripts/bench/os_write_ceiling.py` extended (above); `scripts/bench/gate_admission_sim.py`.
- Docs: `docs-site/docs/operations/storage.mdx` (NVMe recommendation, numbers, sizing, move steps), CHANGELOG.

## 7. What is not fixable here, and what to do next

- **Not done on purpose:** a bulk-ingest mode (refresh -1, translog async): refresh -1 measured 28 percent slower on this
  index type, async translog is within noise and trades crash safety. Revisit only with a non-HNSW-bound index (e.g. detection-only ingest).
- **Needs the live stack (do not do from a worktree):** move the opmono volume to NVMe (steps in the storage page: stop, copy, chown 1000, set
  `OP_OPENSEARCH_DATA`, start). Expected: tail latency and the leg 1 style dips go away; mean ingest rate does not change at 36 images/s.
- **Verify on the stack:** re-run `hires-100k` (or 10k of it) with this branch: expect `admission_wait` near zero, `file_read` visible,
  GPU util above 51 percent. If the rate is still ~21 images/s, the next suspect is the Triton `gpu_image_prep` instance count
  or the nvJPEG throughput (now timed by `gpu_decode`).
- **0.7.0:** (1) put source images and the OpenSearch volume on different devices; (2) sample `/proc/pressure/io` and OpenSearch
  node merge/flush counters in the scale driver so a collapse can be attributed; (3) investigate the cluster pass cost per crop
  (7.6 ms at 20 MP vs 4.2 ms at OI, 2,042 s at 491k crops vs 269 s at 64k; not investigated here: super-linear in crops suggests
  a quadratic step, check `cluster_residuals`); (4) consider scheduling forced merges (`_forcemerge`) between legs rather than during ingest;
  (5) make the gate charge a per-image size when requests mix GPU and CPU routes if a GPU call falls back to CPU often (a fallback
  decodes up to one request of photos on the host outside the charge; counted in `op_ingest_decode_total{route="cpu_call_failed"}`).
- Caveats: one dev host with 21 to 36 load average, 2 repetitions (1 for the mixed case), random vectors in real-shaped documents,
  2 GB heap, the NAS is a local md array mounted on this host (not a remote share in the mount table) so a true remote share is likely worse.

## After the fixes: 20 MP re-check (2026-10-16)

The admission-gate and parallel-read fixes (07cb77fa, c2292728) were run on the dev stack and compared with the earlier 100k
run of the same kind of photos (a private collection of 20 MP camera-original JPEGs, aggregates only).

| Run | Images | Throughput | Wall | Failures | Peak GPU util |
|---|---|---|---|---|---|
| Before (100k, two legs) | 100,000 | 21.0 img/s (72% of the 29.0 baseline) | 4,761 s | 0 | 100% / 26 GiB (mean 51%) |
| After (20k, forced stop at 50% and resume) | 20,000 | 27.18 img/s (94% of baseline) | 736 s (421 s + 315 s) | 0 | 100% / 36 GiB |

About 29% faster, with the stop and resume intact. Caveats: the after run is a 20k slice of the same collection, not the same
100k files, and a single observation on a shared host; the cumulative `admission_wait` fell from a large share of request time
to roughly a sixth of `opensearch_write` early in the run. The mean GPU utilisation of the after run was not sampled over a
full 100k, so a repeat at 100k is a 0.7.0 item. The OpenSearch node was not moved to NVMe for this check.
