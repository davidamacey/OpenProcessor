# SP0: stage-by-stage ingest profile (code half) and the measurement runbook

Branch `perf/v070-ingest-speed` (worktree `OpenProcessor-wt-sp`), merge base `0e2d3fe7` (`origin/main`),
merged with `origin/main` at `d46dadc2`. Plan: ship-track plan sections 3 (protocol) and 5 (SP0).
**Status: the code half and the measurement half have both run** (2026-10-30, GPU 0). The numbers, the per-stage
tables and the findings are in "Measurement results" below and in
[`../research/v070_measurement_log.md`](../research/v070_measurement_log.md), section "SP0: stage profile". The text
under "What changed" to "Risks" was written before the run; where the run changed it, "Measurement results" says so.

## What changed

| Commit | Subject |
|---|---|
| `1421df53` | perf(gpu-prep): time every gpu_image_prep stage as Triton counters |
| `aff19e82` | feat(bench): repetitions, cache drop, load average and gpu_image_prep stages in run_baseline |
| `016e47b6` | feat(bench): ingest profile driver, sample copy and per-stage table |
| `fee457e5` | docs(reviews): SP0 review and measurement runbook, changelog entry |
| `77500a12` | fix(gpu-prep): make the stage counters work on Triton 2.70 without crashing it (found by the run) |
| `553ade23` | fix(bench): lower-case, short project slugs for the SP0 rows |
| `6224897f` | fix(bench): stage tables read the stack GPU and Triton `_total` series, add `--clients` |
| this commit | docs: SP0 results (log section, PERFORMANCE heading, committed rows) |

Design points:

- **Metrics.** `models/gpu_image_prep/1/stage_metrics.py` keeps one `StageClock` per request and
  publishes it once when the request ends: two Triton counters labelled `stage`,
  `op_gpu_prep_stage_seconds` (summed host wall seconds) and `op_gpu_prep_stage_observations`
  (images that ran the stage). Cost per image: about ten `perf_counter` calls and one counter
  increment per stage that ran. Outputs are unchanged (test below).
- **Stages.** The plan's seven (`decode`, `letterbox`, `detect`, `embed_frame`, `embed_crops`,
  `crop_jpeg`, `blur`) plus three that the brief asked for and the plan's list lacked: `queue`
  (wait for a free slot and a worker thread, the per-instance queueing), `embed_pe` (the PE calls,
  split from the square preparation so `embed_frame` and `embed_crops` mean preparation only) and
  `crop_copy` (device-to-host copy of the crop pixels). `crop_jpeg` is the encode time summed over
  one image's crops (host threads, concurrent), not a wait.
- **Asynchronous kernels.** CUDA launches return early, so by default a kernel stage ends when its
  launches are queued and the GPU time lands in the next stage that waits (`detect`, `embed_pe`).
  STAGE_SYNC set to 1 (model parameter, default `"0"`) waits for the stream at the end of `letterbox`,
  `embed_frame`, `embed_crops` and `blur` so the time is attributed where the work ran. It removes
  overlap inside one image, so those runs are reported as attribution, never as throughput.
- **Stored-box mode and CPU fallbacks** publish only the stages they ran (`queue`, `decode`,
  `embed_*` for the embed pass; `queue`, `decode` for an image that falls back to the CPU decode).
- **Harness.** `run_baseline.py` gains `--reps N` (fresh project per repetition, files
  `<out>-rep<i>.json`, `<out>-summary.json` with median and range), `--drop-cache` (fadvise
  `DONTNEED` over the manifest once per repetition), `host.loadavg_start/end/noisy` (noisy when the
  1-minute average at the start exceeds 16), `gpu_prep_stages` (per stage: seconds, images, mean,
  share, from the Triton counter delta), and a GPU utilisation median next to the mean.
  `ingest_profile.py run` fixes the protocol values (3 reps, cache drop, 0.5 s GPU samples, policy
  `all`, stage `ingest`) and adds the API container's cgroup CPU seconds and the Triton container's
  `docker stats` CPU. `ingest_profile.py table` renders the per-row table. `sample_copy.py` copies
  a seeded sample to NVMe under neutral names with a SHA-256 check (never move, never link).
- Prometheus parsing: `prometheus_client` appends `_total` to counter names that Triton emits
  without it; `gpu_prep_stages` accepts both. A test with a realistic exposition text pins this.

## Measurement results

Full detail: log section "SP0: stage profile" (tables SP0.1 and SP0.2, findings SP0.3, limits SP0.5). Raw JSON:
`artifacts_local/bench/v070/sp0/<row>/` in the main checkout, committed copies in `docs/benchmarks/v070/sp0/`.
Stack: GPU 0, Triton 2.70.0, release 0.6.0 images, two `gpu_image_prep` instances, host load 10 to 59 (most rows
contain noisy repetitions, not repeated).

| Row | img/s median (min to max) | GPU 0 util mean % | Binding stage |
|---|---|---|---|
| A, COCO 1,846 images, CPU route, 1 batch | 13.14 (13.12 to 13.47) | 45 | PE crop embed 54.5% of API stage time, OpenSearch write 42.1% |
| B, 400 photos, GPU route, 1 batch | 9.88 (9.63 to 9.96) | 15 | `decode` 39.5% of the model's stage time |
| P, 2,900 photos, GPU route, 1 batch | 14.80 (11.92 to 16.47) | 35 | `embed_pe` 27.6%, `decode` 19.3% |
| B, 4 batches at once | 26.64 (25.21 to 29.00) | 49 | CUDA-call stages (`letterbox`, `embed_*`, `crop_copy`) |
| P, 4 batches at once | 24.98 (23.50 to 33.08) | 58 | `embed_crops` 21.3%, `queue` 14.0%, `letterbox`, `crop_copy` 11% each |
| P, 8 batches at once | 20.13 (19.20 to 21.91) | 39 | same, grown 5 to 33 times; queue 1,080 ms |

Acceptance:

| Bar | Measured | Met | Raw data |
|---|---|---|---|
| Three tables (set A, set B, set P 3,000), every stage named with its share | `docs/benchmarks/v070/sp0/sp0_tables.md` (1 batch), `sp0_tables_clients.md` (4 and 8 batches), `sp0_tables_sync.md` | yes | `docs/benchmarks/v070/sp0/<row>/` |
| The bounding stage named per route | CPU route: PE crop embed (then OpenSearch write). GPU route: at one batch `decode` (B) and `embed_pe` (P); at saturation the CUDA-call stages inside `gpu_image_prep` | yes; the GPU-route answer depends on the load (see below) | same |
| Stage timing adds no per-image cost that matters | counters on/off, set B: 1 batch ratio 0.978, 4 batches 0.977 (9 runs with, 3 without); inside the run-to-run spread | yes, loosely: the 1-batch ranges do not overlap (9.63 to 9.96 against 10.06 to 10.59) and 2 of the 3 "without" runs were noisy | `setB`, `setB-nometrics`, `setB-c4*` |

What the run changed:

- **Counters did not work as committed.** They never appeared on the metrics port (runbook step 1 check failed) and,
  once visible, crashed Triton 2.70. Fixed in `77500a12`; issue #390. The runbook check passes after the fix: all ten
  stages are listed at 0 after the load and count up under traffic.
- **The driver under-drove the stack.** `run_baseline.py` sent batches one after another (at most 16 photos in
  flight). The headline throughput rows therefore use `--clients 4`; the one-batch rows are the latency profile. SP4 and SP5
  must use `--clients` (or `scale_run.py`) for any bar.
- **Ranking.** GPU route: (1) CUDA-call and serialisation overhead inside the `gpu_image_prep` process under concurrency
  (issue #392), (2) nvJPEG decode, (3) PE embedding wait, (4) crop JPEG encode and copy (SP3's target: 9.5 to 21.6 percent of the
  model's stage time, so SP3's gain is bounded to about that). CPU route: PE crop embed, then OpenSearch writes.
- **Does the ranking agree with SP1, SP2, SP3?** Partly. SP3 is supported (crop stages above the plan's 10 percent
  threshold at one batch, 9.5 to 15.6 percent at 4). SP1 (cluster pass) does not touch ingest. SP2 is a measurement. The data
  adds an item ahead of SP3: reduce small CUDA calls per image in `gpu_image_prep` and test CUDA MPS and the instance
  count at 4 batches, then re-run set P at `--clients 4` against the SP4 bar (29 img/s, GPU above 75 percent).
  The owner decides the order (gate: the SP review package).
- **Not reproduced:** the earlier set P rates (45 to 49 img/s, E4 and F1). Best repetition here 33.1, median 25.0 at 4 batches.

Issues filed (milestone v0.7.0): #390 (counter constraints, fixed on the branch), #391 (`/health` ready with a startup
model failed to load), #392 (throughput falls past 4 batches, API CPU growth).

## How it was tested

Red first (each test file was run before its implementation or its implementation was reverted, and
seen failing):

- `tests/models/test_gpu_image_prep_stage_metrics.py` (8 tests; the model code runs against stand-ins
  for CuPy, nvJPEG, the CUDA kernels and Triton's Python backend; control flow, counters and response
  tensors are the real code). Revert checks: removing `publish` fails 4 tests, removing the
  `crop_jpeg` accounting fails `test_every_stage_is_observed_once_per_image`. Covered: every stage once
  per image in the full mode, only the ran stages in stored-box and CPU-fallback modes, the queue
  stage under a `MAX_IN_FLIGHT` of 1, outputs equal with metrics, without metrics and with STAGE_SYNC set to 1,
  no stream wait by default, an unknown stage name refused, STAGE_SYNC declared in `config.pbtxt`.
- `tests/bench/test_run_baseline_reps.py` (12), `tests/bench/test_baseline_report.py` (GPU median),
  `tests/bench/test_ingest_profile_report.py` (6), `tests/bench/test_ingest_profile.py` (5),
  `tests/bench/test_sample_copy.py` (5; reverting the copy to a hard link fails 2). Revert check on the
  `_total` handling fails `test_gpu_prep_stage_block_from_the_triton_metrics_delta`.
- `tests/bench` and `tests/models` together: 399 passed. `.venv/bin/pre-commit run --all-files`
  clean at each commit (no `--no-verify`).
- Not done on the CPU side: the real `pb_utils.MetricFamily`, the real CUDA stream and the real
  nvJPEG lanes. The live run exercised them and found the counter defects (#390); the stand-in now models
  Triton's family lifetime, its deletion order and its one-thread-at-a-time rule, and three new tests fail without the
  fix (the model hung or `increment` raised).
- After the run: `tests/bench` and `tests/models` pass; each fix commit ran `pre-commit` on its files.

## Risks and what was not done

- **Run:** every GPU step of the runbook, plus the `--clients` rows. See "Measurement results".
- **Not done: the `scale_run.py` latency probe and its 500 ms GPU series** (the third SP0 code item).
  The brief for this pass named the stage profile and the driver, and the probe belongs to SP6 (#356):
  it needs a decision on which k-NN route a "stored-vector k-NN" calls (the curation OpenAPI has
  `GET {P}/search/text` and the clusters list, no vector query route) and the plan's `GET /search/text`
  is a `POST` on the core API. Open as part of SP5/SP6, not forgotten.
- STAGE_SYNC runs change the overlap inside one image; the runbook keeps them in separate rows.
- The Triton metrics port serves the counters only for models that loaded: the model is loaded on
  first use, so the warm-up images of every row (default 100) also load it.
- The runbook reuses the stopped agent's install (`optest2`, ports 5700 to 5712). Its `.env` still
  says `TRITON_GPU_ID=2` and `API_GPU_ID=2`: step 1 changes both to 0 before anything starts.
  Starting it unchanged would put the stack on the owner's GPU 2.

## Decisions needed from the owner (gate ids)

None for the code. The 20,000 photo sample (plan section 3 item 1) is SP4 and needs the owner's yes;
this runbook uses the 3,000 sample only.

## Five-minute check for the owner

```bash
cd "$WT"   # the worktree root; MAIN, WT and VPY are defined in the runbook below
$VPY -m pytest tests/models/test_gpu_image_prep_stage_metrics.py \
  tests/bench/test_run_baseline_reps.py tests/bench/test_ingest_profile.py \
  tests/bench/test_ingest_profile_report.py tests/bench/test_sample_copy.py -q --no-cov
```

Expected: all pass. Then read `models/gpu_image_prep/1/stage_metrics.py` (the stage list and the
STAGE_SYNC note) and the table below it.

---

# Runbook: SP0 measurement (needs GPU 0 for about 1.5 to 2 hours under the lock)

Corrections found by running it on 2026-10-30 (the steps below are otherwise unchanged):

- Run a 30-photo probe first and check `curl .../metrics | rg op_gpu_prep` after traffic, not only after the load.
- Set P 3,000 is 20 GB (6.7 MB per photo), not 28 GB. Set A keeps 1,946 of the 2,000 pinned images. Set B's 500 downloads
  hit the host's rate limit (HTTP 429) and take about 25 minutes.
- Row names become project slugs (`sp0-<row lower case>-r<i>-<stamp>`, at most 32 characters): keep `setB-nometrics` as the
  longest. Add `--clients 4` (and `--clients 8` for the saturation row) to every throughput row; without it only 16
  photos are ever in flight. `ingest_profile.py table` reads GPU 0 unless `--gpu N` says otherwise.
- The `optest2` install lacked `models/pe_image_encoder/1/model.plan`; copy it from another install of the same release
  (read-only source) and check `curl .../v2/models/pe_image_encoder/ready` before the first row (issue #391).
- A Triton restart needs about 60 s before `/v2/health/ready` and the API's `/health` agree; wait for both.
- Never `pkill -x tritonserver` inside the Triton container (it is also the server): kill a recorded pid.

Expected duration: stack start and model load 5 to 10 min; warm-up plus 3 repetitions of set A (2,000
COCO images) about 5 min, set B (500 photos) about 3 min, set P 3,000 about 12 min (cold page cache, 28 GB
read); the STAGE_SYNC rows about 15 min; data preparation (CPU and NVMe only, no lock) about 20 min. Image
rates are from the existing log (one instance: 37 to 39 img/s; two: 32 to 48), so a repetition of
3,000 photos takes about 1 to 2 minutes plus cold reads.

Preconditions, checked first and reported if they fail (a failed check stops the step):

```bash
nvidia-smi --query-gpu=index,pci.bus_id,utilization.gpu,memory.used --format=csv   # GPU 0 (03:00) idle
nvidia-smi --query-compute-apps=gpu_bus_id,pid,used_memory --format=csv            # nothing on 03:00
docker ps --filter name=optest2                                                      # empty: the SP stack is down
ss -ltn | rg ':57(0[0-9]|1[0-2])\b'                                                  # ports 5700 to 5712 free
```

Shell names (from the plan's section 0):

```bash
MAIN=$(git worktree list | head -n1 | cut -d' ' -f1)   # the main checkout
WT=$MAIN/../OpenProcessor-wt-sp
SCRATCH=$(dirname "$MAIN")/../scratch                  # host scratch; locks in $SCRATCH/locks
INST=$SCRATCH/op-install-2/openprocessor          # compose project optest2, ports 5700 to 5712
VPY=$MAIN/.venv/bin/python
ART=$MAIN/artifacts_local/bench/v070/sp0          # raw JSON, gitignored, survives worktree removal
export CUDA_DEVICE_ORDER=PCI_BUS_ID
cd "$WT" && mkdir -p "$ART" && "$VPY" --version
```

## 0. Data (no GPU, no lock)

```bash
mkdir -p "$ART" "$SCRATCH/sp"
# Set A: 2,000 COCO images, pinned
$VPY scripts/datasets/fetch_coco_subset.py --out $SCRATCH/sp/src_coco2000 --bench-set 2000 \
  --seed 20261009 --manifest scripts/datasets/manifests/coco_bench_2000.json
$VPY scripts/bench/select_baseline_set.py local --root $SCRATCH/sp/src_coco2000 --count 2000 \
  --seed 20261009 --out $ART/setA.txt
# Set B: 500 high-resolution public photos, pinned
$VPY scripts/datasets/fetch_hires_set.py --out $SCRATCH/sp/src_hires500 --n 500 \
  --manifest scripts/datasets/manifests/commons_hires_500.json
$VPY scripts/bench/select_baseline_set.py local --root $SCRATCH/sp/src_hires500 --count 500 \
  --seed 20261010 --out $ART/setB.txt
# Set P: 3,000 seeded photos copied to NVMe (SETP_ROOT = the owner's photo root; never write it into a committed file)
$VPY scripts/bench/select_baseline_set.py local --root "$SETP_ROOT" --count 3000 --seed 20261029 \
  --out $SCRATCH/sp/setp3000_src.txt
$VPY scripts/bench/sample_copy.py --manifest $SCRATCH/sp/setp3000_src.txt \
  --dest $SCRATCH/sp/setp3000 --out $ART/setP3000.txt --seed 20261029
df -h "$SCRATCH" | tail -1                      # about 28 GB are written
```

`fetch_coco_subset.py --verify-only` (see its docstring) re-checks the COCO hashes offline. The
1,500-image directories `src_coco` and `src_hires` that the earlier SP attempt left in `$SCRATCH/sp` hold
fewer images than the plan's sets; do not use them for these rows.

## 1. Stack on GPU 0 (isolated project `optest2`)

The overlay `$SCRATCH/sp/docker-compose.sp.yml` bind-mounts this worktree's `src/` and `scripts/` read-only
over the API and workers and `$SCRATCH/sp` at `/data/nvme`. Add the crop-cache switch of protocol item 3
to its `api` service (`environment:` list) before the first start:
`- OP_CROP_CACHE_MAX_BYTES=0`.

```bash
# the install's .env still points at GPU 2: move Triton and the API to GPU 0 (edit with an editor, do not print the file)
#   TRITON_GPU_ID=0     API_GPU_ID=0
# install this branch's gpu_image_prep (config, model.py, stage_metrics.py) into the model repository
cp -a $INST/models/gpu_image_prep $SCRATCH/sp/bak/gpu_image_prep_before_sp0       # new name, keeps the v0.6.0 copy
rsync -a --exclude __pycache__ $WT/models/gpu_image_prep/ $INST/models/gpu_image_prep/
rg -n "count:" $INST/models/gpu_image_prep/config.pbtxt                         # 2 on the full profile; 1 on minimal and standard
```

Decision to record in each row's heading: the instance count (the committed `count: 2` is the full
profile and the one the log's two-instance numbers use; keep 2 for SP0).

```bash
COMPOSE="docker compose -p optest2 --project-directory $INST -f $INST/docker-compose.yml -f $SCRATCH/sp/docker-compose.sp.yml"
flock -w 3600 $SCRATCH/locks/gpu0.lock bash $SCRATCH/sp/sp0_session.sh        # the whole session, step 2 to 4, holds the lock
```

Put steps 2 to 4 in `$SCRATCH/sp/sp0_session.sh` (a scratch file, not in the repository) so one `flock`
covers the session; its first lines start the stack:

```bash
$COMPOSE up -d triton-server api opensearch
# wait for health through the published ports (marker loop with a timeout, no pgrep -f)
for i in $(seq 1 120); do curl -sf -m 5 http://127.0.0.1:5703/health >/dev/null && break; sleep 5; done
time curl -s -m 20 http://127.0.0.1:5703/health | head -c 300            # preflight: timed /health
# explicit model control: reload the model so the new files are used
curl -s -X POST http://127.0.0.1:5700/v2/repository/models/gpu_image_prep/unload
curl -s -X POST http://127.0.0.1:5700/v2/repository/models/gpu_image_prep/load
curl -s http://127.0.0.1:5702/metrics | rg 'op_gpu_prep_stage_(seconds|observations)' | head -4   # the 10 stages are listed at 0
```

If the health check or the metrics line does not appear, stop and report; do not work around it.
Note the commit, image tags, driver, Triton and TensorRT versions in the first row's heading
(`git rev-parse HEAD`, `docker compose -p optest2 images`, `nvidia-smi | head -4`, the Triton
`/v2` server metadata).

## 2. The three rows (one factor at a time, 3 repetitions each)

```bash
for ROW in setA setB setP3000; do
  $VPY scripts/bench/ingest_profile.py run $ROW $ART/$ROW.txt \
    --api-url http://127.0.0.1:5703 --triton-metrics-url http://127.0.0.1:5702/metrics \
    --api-container optest2-api --triton-container optest2-triton \
    --path-map $SCRATCH/sp=/data/nvme --out-root $ART
done
```

(`setA` is the CPU route, 640 x 480 frames; `setB` and `setP3000` take the GPU route.) Each repetition
creates a project `sp0-<row>-r<i>-<stamp>`, drops the manifest files from the page cache first, samples
the GPU every 500 ms, reads the load average at start and end, and diffs the Triton and API metrics.
The first 100 images of each repetition are warm-up and excluded. A repetition marked noisy in the
table (1-minute load above 16 at the start) is repeated before it decides anything:
`ingest_profile.py run <row> ... --reps 1` into a fresh `--out-root`.

## 3. Attribution rows (STAGE_SYNC=1; not throughput)

```bash
sed -i 's/key: "STAGE_SYNC" value: { string_value: "0" }/key: "STAGE_SYNC" value: { string_value: "1" }/' $INST/models/gpu_image_prep/config.pbtxt
curl -s -X POST http://127.0.0.1:5700/v2/repository/models/gpu_image_prep/unload
curl -s -X POST http://127.0.0.1:5700/v2/repository/models/gpu_image_prep/load
for ROW in setB setP3000; do
  $VPY scripts/bench/ingest_profile.py run ${ROW}-sync $ART/$ROW.txt --api-url http://127.0.0.1:5703 \
    --triton-metrics-url http://127.0.0.1:5702/metrics --api-container optest2-api \
    --triton-container optest2-triton --path-map $SCRATCH/sp=/data/nvme --out-root $ART
done
# restore the default and reload
sed -i 's/key: "STAGE_SYNC" value: { string_value: "1" }/key: "STAGE_SYNC" value: { string_value: "0" }/' $INST/models/gpu_image_prep/config.pbtxt
curl -s -X POST http://127.0.0.1:5700/v2/repository/models/gpu_image_prep/unload
curl -s -X POST http://127.0.0.1:5700/v2/repository/models/gpu_image_prep/load
```

## 4. Overhead check of the counters themselves (set B, 3 reps)

The question is whether the counters cost throughput. Put the merge base's `model.py` (no timing) next
to this branch's other model files, reload, run the row, then restore the branch files and reload:

```bash
git -C $WT show 0e2d3fe7:models/gpu_image_prep/1/model.py > $INST/models/gpu_image_prep/1/model.py
curl -s -X POST http://127.0.0.1:5700/v2/repository/models/gpu_image_prep/unload
curl -s -X POST http://127.0.0.1:5700/v2/repository/models/gpu_image_prep/load
$VPY scripts/bench/ingest_profile.py run setB-nometrics $ART/setB.txt --api-url http://127.0.0.1:5703 \
  --triton-container optest2-triton --api-container optest2-api --path-map $SCRATCH/sp=/data/nvme --out-root $ART
rsync -a --exclude __pycache__ $WT/models/gpu_image_prep/ $INST/models/gpu_image_prep/
curl -s -X POST http://127.0.0.1:5700/v2/repository/models/gpu_image_prep/unload
curl -s -X POST http://127.0.0.1:5700/v2/repository/models/gpu_image_prep/load
```

Same config, same data, same instance count, so only the timing differs. Report the median ratio of
images/s (`setB` over `setB-nometrics`) and the range; the bar is "within the repetition spread".

## 5. Tables, notes and teardown

```bash
$VPY scripts/bench/ingest_profile.py table "set A (COCO 2,000)=$ART/setA" "set B (500 photos)=$ART/setB" \
  "set P (3,000 photos)=$ART/setP3000" > $ART/sp0_tables.md
$VPY scripts/bench/ingest_profile.py table "set B, STAGE_SYNC=1=$ART/setB-sync" \
  "set P, STAGE_SYNC=1=$ART/setP3000-sync" > $ART/sp0_tables_sync.md
```

- Public rows (set A, set B) go to `docs/PERFORMANCE.md` under a new heading "0.7.0 stage profile", with
  the reproduce commands above and the raw-data path. Small result files that back a number go to
  `docs/benchmarks/v070/sp0/` with `MANIFEST.sha256` updated.
- Set P goes to a new section "SP0: stage profile" of `docs/research/v070_measurement_log.md`:
  aggregates, ratios and the table only; no path, folder or file name (the naming scan enforces it).
- Name the stage that bounds each route (expected from the log: inside `gpu_image_prep` on the GPU route,
  with `crop_jpeg` growing with crops per image; the PE crop embed on the CPU route) and say whether the
  ranking agrees with the order SP1, SP2, SP3. If it does not, say so in the next review package.
- Stage seconds are sums over concurrent work: compare them across rows, never add them to wall time.
  The `queue` stage includes the wait for a free slot (MAX_IN_FLIGHT 96), so a large `queue` share means
  the callers outrun the model, not that the model is slow.
- Teardown: `$COMPOSE down` (no `-v`); delete only the throw-away projects named `sp0-*` through
  `DELETE {P}` for each slug (never a bulk delete, never another prefix); keep the raw JSON; release the lock by
  ending the session script. Restore the install's model files from the backup if the stack is to be reused
  by another lane.
