# SPD: `gpu_image_prep` under concurrency, design review (diagnosis first, then code)

Status: **design review only, nothing implemented.** Branch `docs/sp-gpu-prep-design` (from
`perf/v070-ingest-speed` at `2d231652`), 2026-10-30. Issues: #392 (the symptom), #40 (epic, SP4 bar),
#337 (admission charge on fallbacks), #219 (SP3, crop size), #42 (12 GB cards), #391 (health).

Inputs read: [`../reviews/sp0_review.md`](../reviews/sp0_review.md) "Measurement results", the SP0
tables and the 36 per-repetition JSON files in [`../benchmarks/v070/sp0/`](../benchmarks/v070/sp0/sp0_tables_clients.md),
[`../research/v070_measurement_log.md`](../research/v070_measurement_log.md) sections B, E4, F1, G3, G5 and
"SP0: stage profile", [`../PERFORMANCE.md`](../PERFORMANCE.md) "0.7.0 stage profile", the ship-track plan
section 5 (`docs/design/ship_track_plan.md` on branch `docs/ship-track-plan`), the model
([`model.py`](../../models/gpu_image_prep/1/model.py), [`prep_kernels.py`](../../models/gpu_image_prep/1/prep_kernels.py),
[`prep_geometry.py`](../../models/gpu_image_prep/1/prep_geometry.py), [`nvjpeg_lanes.py`](../../models/gpu_image_prep/1/nvjpeg_lanes.py),
[`stage_metrics.py`](../../models/gpu_image_prep/1/stage_metrics.py), [`config.pbtxt`](../../models/gpu_image_prep/config.pbtxt)),
and the callers ([`ingest_batch.py`](../../src/services/curation/ingest_batch.py),
[`ingest_gpu_decode.py`](../../src/services/curation/ingest_gpu_decode.py),
[`reprocess_gpu_embed.py`](../../src/services/curation/reprocess_gpu_embed.py)). Set P appears as numbers only.

## 1. Summary

1. The SP0 JSON already holds the answer SP0 did not read: under load most of the throughput loss is
   **images falling back from the GPU route to the API's CPU decode**, not slow CUDA launches.
   Set P at 8 batches: 60 to 72 percent of the photos that should have taken the GPU route fell back
   (`op_ingest_decode_total` `cpu_decoder` + `cpu_call_failed`); at 4 batches 2.3, 18.1, 24.4 percent
   (repetitions 1, 2, 3); at 1 batch 0.03 to 0.35 percent.
2. Every `cpu_call_failed` is an exception raised inside the model, and only at the stages that
   allocate GPU memory (letterbox, crop_copy, embed_frame, embed_crops), never in detect or embed_pe;
   every `cpu_decoder` is `_decode` swallowing an exception. GPU 0 peaked at 48,558 of 49,140 MiB in
   every GPU row. Reading: **device memory exhaustion**, surfacing as silent CPU fallbacks.
3. API CPU per image is explained by those fallbacks: 0.059 + 0.91 x (fallback fraction), r = 0.997
   over nine repetitions. #392's "cause not found" for the API CPU growth is this.
4. Fallbacks grow with **time since the stack started**, not only with load: within one stack session
   set P at 4 batches went 2.3, 18.1, 24.4 percent, then set B and set P at 8 batches; set B at 4
   batches on freshly started stacks fell back 0 to 2.8 percent. The 8-batch row never ran on a fresh
   stack, so "throughput falls past 4 batches" is confounded by session age and is not established.
5. The 45 to 49 img/s of log E4 ran the same in-flight count (16 batches of 16 = 256 photos; SP0 at
   8 batches of 32 = 256), the same engines, model configs and `POOL_MIB`, but on a card with about
   11 to 14 GB more free memory (E4 "used after" 34.6 to 37.9 GB; SP0 card full). The SP stack's
   Triton holds 36.5 GB resident at start; the `optest1` stack's 23.1 GB. That difference, not the
   driver, is the most likely reason SP0 did not reproduce E4. Not proven until E1 runs.
6. Secondary hypotheses stay real but rank lower: about 25 hidden stream synchronisations and 20
   device allocations per image (pageable weight uploads that miss the cache for every crop, one
   pageable device-to-host copy per crop), GIL convoy across 32 threads, two contexts without MPS.
7. Section 4 gives eleven cheap experiments with exact commands and expected signatures; E1
   (unload the models ingest does not use, run 8 batches first on a fresh stack) decides H1 in about
   20 minutes of GPU time with no code.
8. Section 5 gives the code if confirmed: a byte-budget admission inside the model so memory
   pressure waits instead of falling back, a distinct counted status for an allocation failure,
   a reservation at load so a card that cannot hold the instances fails at load (fail closed), then
   fewer syncs and allocations per image (packed pinned uploads, batched resample launches, one
   device-to-host copy per image), then concurrency shape (threads, instances, MPS) only on evidence.
9. Bars (section 7): set P at `--clients 4` at least 29 img/s with GPU above 75 percent on the
   default stack, no fall at `--clients 8`, fallbacks at most 0.5 percent, API CPU at most 0.08
   CPU-s per image, card never full, outputs bit-identical where the math did not change.
10. Owner decisions: the acceptance stack (all default models loaded, recommended), MPS (opt-in
    overlay only if measured), the order of this item ahead of SP3, and the Triton resident-memory
    gap (36.5 against 23.1 GB) as its own issue.

## 2. What the SP0 data shows when read per repetition

### 2.1 Route counters, set P (from the committed JSON, `metrics.api.counters`)

Fallback = `cpu_decoder` + `cpu_call_failed`, over the photos the API sent to the model
(`gpu` + fallbacks; `cpu_small` excluded).

| Row, repetition (session order) | img/s | GPU util mean % | Fallbacks | API CPU-s / image |
|---|---|---|---|---|
| 1 batch, r1 | 11.9 | 26 | 2 / 2,892 (0.07%) | 0.078 |
| 1 batch, r2 | 14.8 | 35 | 10 / 2,892 (0.35%) | 0.080 |
| 1 batch, r3 | 16.5 | 37 | 1 / 2,892 (0.03%) | 0.066 |
| 4 batches, r1 (first row of a fresh stack) | 33.1 | 73 | 67 / 2,892 (2.3%) | 0.071 |
| 4 batches, r2 | 25.0 | 58 | 524 / 2,892 (18.1%) | 0.200 |
| 4 batches, r3 | 23.5 | 50 | 705 / 2,892 (24.4%) | 0.249 |
| 8 batches, r1 (same stack, after set B) | 19.2 | 40 | 1,740 / 2,892 (60.2%) | 0.641 |
| 8 batches, r2 | 20.1 | 38 | 2,088 / 2,892 (72.2%) | 0.706 |
| 8 batches, r3 | 21.9 | 39 | 2,022 / 2,893 (69.9%) | 0.701 |

Set B at 4 batches, same counters: in the stack session that had just run set P at 4 batches,
5.7, 11.4, 9.1 percent (29.0, 26.6, 25.2 img/s); on freshly started stacks (sessions D and E of the
run) 2.0, 0.0, 2.0 / 2.8, 1.1, 2.3 / 0.0, 0.6, 0.9 percent (26.2 to 31.8 img/s). The first set P
repetition at 4 batches, on a fresh stack with 2.3 percent fallbacks, already ran at 33.1 img/s with
the GPU 73 percent busy: within a few points of the SP4 bar.

### 2.2 Where the model fails

The stage counters count a stage when it is entered (the clock adds in `finally`), so the drop from
one stage's observations to the next is the number of requests that raised in that stage. Set P,
8 batches, repetition 1: `decode` 2,894, `letterbox` 1,546 (1,348 returned status `decoder`; the API
counted 1,348 `cpu_decoder`), `detect` 1,308 (238 raised in letterbox), `crop_copy` 1,308 (0 raised in
detect), `embed_frame` 1,206 (102 raised in crop_copy), `embed_crops` 1,169 (37 in embed_frame),
`embed_pe` 1,154 (15 in embed_crops), `blur` 1,154 (0 in embed_pe). Sum 392 = the API's
`cpu_call_failed` 392. The identity holds exactly in all six set P repetitions at 4 and 8 batches and
in set B: **every failed call is an exception inside the model**, raised only where CuPy allocates
(`cp.full` and the resample scratch in letterbox, `cp.ascontiguousarray(...).get()` in
`_crop_pixels`, `cp.empty` and the weight uploads in `_embed`), never in the BLS stages (detect,
embed_pe). The `decoder` statuses are `_decode`'s `except Exception` around `NvJpegLanes.decode`,
which allocates the full frame (`cp.empty` in `_decode_on`) and nvJPEG's own device buffers.

### 2.3 Memory

`gpu.0.mem_peak_mb` is 48,442 to 48,558 MiB in every GPU-route repetition (card 49,140 MiB), and
37,060 MiB in the CPU-route row (set A, `gpu_image_prep` loaded but idle). Measured today on the
freshly started `optest2` stack (same install, same models): `tritonserver` 36,488 MiB and each idle
`gpu_image_prep` stub 268 MiB. So two instances had about 12 GB for two CuPy pools capped at
`POOL_MIB` 4,096 each, two nvJPEG lane sets (16 lanes, buffers grown to the largest photo seen),
two CUDA contexts and the BLS tensors. `config.pbtxt` budgets 8,704 MiB per instance
(`GPU_MIB_PER_INSTANCE`): 36.5 + 2 x 8.5 = 53.5 GB does not fit a 48 GB card.

### 2.4 Corrections to SP0's reading

- "At most 16 photos in flight" holds for `ingest_one` (`OP_MAX_INGEST_CONCURRENCY`), not for the GPU
  decode: `_prefill_gpu_decode` in `ingest_batch.py` gathers every photo of the batch the route chose,
  so one batch of 32 puts up to 32 photos into `gpu_image_prep` at once (4 batches 128, 8 batches 256).
  SP0's Little's-law average (12 in flight at one batch) stands.
- The launch-stage growth (letterbox 21 / 183 / 475 ms and so on) is measured over requests that
  include the failing ones and the allocator's retry path, so part of it is memory pressure, not
  launch cost. The first repetition at 4 batches (2.3 percent fallbacks) still shows letterbox 105,
  embed_crops 311, crop_copy 117 ms per image against 21, 53, 29 at one batch: a real 4 to 6 times
  growth remains to explain once the memory effect is removed (H2 to H5).
- The "4 against 8 batches" reversal is confounded by session age (2.1). It must be re-measured with
  a fresh stack per row and the 8-batch row first (E1).

## 3. Reconciling 45 to 49 img/s (E4, F1, G5) with 25 img/s (SP0)

| Factor | E4 (#268, 48.0 img/s at 2 instances) | SP0 set P, 4 and 8 batches | Same? |
|---|---|---|---|
| Photos in flight into the model | 16 batches x 16 = up to 256 (log section B: "concurrency x batch images are in flight") | 4 x 32 = 128, 8 x 32 = 256 | 8 batches: yes |
| Model code | release/0.7.0 files | same plus the stage counters, the `boxes` input and `crop_rect` clamping; counters cost nothing measurable (SP0.3 item 5) | yes in the hot path |
| Engines and model configs | `op-install-1` | `op-install-2`: plan sizes within 0.3 percent, `pe_image_encoder`, detector and text encoders' configs identical, same `POOL_MIB`, `WORKERS`, `MAX_IN_FLIGHT` (checked 2026-10-30) | yes |
| Triton command, image, environment | 26.06, 2.70.0 | identical (diffed) | yes |
| Data | 3,000 distinct set P photos, 2.3 crops per image | 3,000 seeded set P copies, 2.41 crops per image | close |
| Card memory | "used after" 34.6 to 37.9 GB; that stack's `tritonserver` holds 23.1 GB today | peak 48.56 of 49.14 GB in every row; `tritonserver` 36.5 GB at start | **no: about 11 to 14 GB less headroom** |
| Model state at start | model unloaded and loaded before each configuration | one load per stack session, rows run back to back | partly |
| Host load (1-minute) | 12.3 to 16.2 | 14 to 26 | worse |
| GPU sampling | 5 s (reads about 14 points low, log G6) | 500 ms | E4's GPU figures are low, not high |
| Route counters | not recorded in the E4 table (F1, F4, G5 on the same stack recorded no fallbacks) | 2.3 to 72 percent fallbacks | unknown for E4 |

Verdict, stated as strength of evidence: the earlier rate is plausible and was most likely real on a
stack with headroom; SP0's stack ran out of device memory, the GPU route failed over to the CPU
route photo by photo, and the failures accumulated with session time. Because `POOL_MIB` was 4,096 on
both stacks and E4 ran the same in-flight count without a visible collapse, the binding limit was the
card (cudaMalloc failing below the pool cap), not the pool cap. What is not known: why the SP stack's
`tritonserver` holds 13 GB more than `optest1`'s with identical engines and configs (E0b), and whether
E4 had fallbacks it did not record. Until E1 runs, quote neither 48 nor 25 img/s as the stack's
capacity; quote "33.1 img/s on a fresh stack with 2.3 percent fallbacks" as the best clean SP0 point.

## 4. Hypotheses, ranked, and the experiment that separates each

| Rank | Hypothesis | For | Against / open | Decided by |
|---|---|---|---|---|
| H1 | **Device memory exhausted** (card full from the start; pool and nvJPEG growth over the session fail in cudaMalloc), failures fall back to the CPU decode in the API | 2.1 to 2.3: failures only at allocation sites, card at 48.5 GB, API CPU linear in fallbacks, decay with session age, E4 headroom | exception text not captured (`_decode` swallows it; Triton log not kept) | E1, E2, E3 |
| H1b | **Allocator churn below failure**: CuPy's pool keeps one arena per stream (32 worker streams + 16 lane streams per instance), so cached blocks of one stream cannot serve another; at the `POOL_MIB` cap or on a failed cudaMalloc it frees every cached block (cudaFree synchronises the device) and runs `gc.collect()` before retrying (CuPy 13.6 in the image; behaviour per CuPy's `SingleDeviceMemoryPool`, to confirm by E5) | the stages that grow 11 to 33 times are exactly the allocating ones; embed_pe, detect and crop_jpeg (no device allocation) grow 1.3 to 2 times | E4 ran the same cap without collapse | E4b, E5 |
| H2 | **Hidden stream syncs in the "launch" stages**: `PrepKernels._weights` misses for every crop (its key is the crop's own size) and uploads 6 small arrays from pageable memory per crop (a pageable host-to-device copy waits for the stream's earlier work); `_crop_pixels` does one pageable `.get()` per crop; `lap_sums` uploads the rectangles; the cache is cleared at 4,096 keys (about every 850 images), so the letterbox and frame keys miss again | embed_crops (most syncs per image, about 6 per crop) grows most; STAGE_SYNC at one batch moved only 1 to 8 ms, so the time is waiting and host work, not kernel time | at one batch letterbox is already 21 ms per image | E5, E9 |
| H3 | **GIL convoy**: 32 worker threads per process; about 40 GIL release and reacquire points per image (nvJPEG ctypes calls, CUDA syncs, BLS `exec`, `cv2.imencode`); a thread reacquiring the GIL behind busy threads can wait up to the 5 ms switch interval (`sys.getswitchinterval()` is 0.005 in the image) | Triton CPU fell while stage times rose (5.8 cores at 4 batches r1, 2.9 at 8 batches): threads wait, they do not compute | the log's section B `top -H`: about 30 stub threads at 5 to 60 percent, none pegged | E6, E7 |
| H4 | **Contexts time-slicing without MPS**: the server (TensorRT models) and two stubs are three CUDA contexts on one card; kernels of different contexts do not overlap | GPU 39 to 58 percent busy while work queues | low GPU util says the host side is idle-waiting, which time-slicing alone does not produce; E4's count 2 gained 51 percent over count 1 without MPS | E7 (count 1/2/3), E8 |
| H5 | **Per-call CuPy launch overhead**: about 60 CUDA API calls per image (2 x (2 + crops) kernel launches, about 20 allocations, uploads, copies) at 10 to 50 microseconds each in Python | adds a few ms per image | a few ms per image cannot make 300 to 700 ms per image alone; it matters only multiplied by H3 | E5 (counts), E7 |
| H6 | **Python backend IPC**: each request's 6.7 MB JPEG is copied into the stub's shared memory, responses back; BLS GPU tensors go through Triton's CUDA memory pool (default 64 MB) and may bounce through host memory when it runs out | `queue` and `gpu_decode` wait outside the stage clock (SP0.3 item 3) | detect and embed_pe, the BLS stages, grow least | E5 (server-side memcpy), E10 |
| H7 | **Pinned against pageable host memory**: every device-to-host crop copy and every weight upload uses pageable numpy memory (bounce buffers, synchronous) | part of H2 | | E5 |
| H8 | **Default-stream serialisation** | | the model runs on per-thread `cp.cuda.Stream(non_blocking=True)` and per-lane non-blocking streams; no work is issued on the legacy default stream in the hot path | E5 (kernels on the default stream: expected none) |
| H9 | **API-side limits** (per-batch gather of 32 photos, `OP_MAX_INGEST_CONCURRENCY` 16 for `ingest_one`, JSON and crop bytes, OpenSearch bulk) | `opensearch_write` 270 to 400 ms per image summed | with fallbacks near 0 the API used 0.07 CPU-s per image; API stages other than `gpu_decode` and `opensearch_write` are under 5 percent | E11, after H1 is removed |
| H10 | Foreign host load (confounder, not a cause to fix) | load 14 to 59 | rule: load above 16 marks a run noisy | every row records load |

### 4.0 Common shell names (as the SP0 runbook; the `optest2` stack, ports 5700 to 5712)

```bash
MAIN=$(git worktree list | head -n1 | cut -d' ' -f1)
WT=$MAIN/../OpenProcessor-wt-sp
SCRATCH=$(dirname "$MAIN")/../scratch
INST=$SCRATCH/op-install-2/openprocessor
VPY=$MAIN/.venv/bin/python
ART=$MAIN/artifacts_local/bench/v070/spd
COMPOSE="docker compose -p optest2 --project-directory $INST -f $INST/docker-compose.yml -f $SCRATCH/sp/docker-compose.sp.yml"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
mkdir -p "$ART"
row() {  # row name (at most 14 characters, lower case), clients, reps
  "$VPY" scripts/bench/ingest_profile.py run "$1" "$MAIN/artifacts_local/bench/v070/sp0/setP3000.txt" \
    --api-url http://127.0.0.1:5703 --triton-metrics-url http://127.0.0.1:5702/metrics \
    --api-container optest2-api --triton-container optest2-triton \
    --path-map "$SCRATCH/sp=/data/nvme" --out-root "$ART" --clients "$2" --reps "${3:-3}"
}
reload_prep() {  # fresh stub process: CuPy pool, nvJPEG lanes and streams start empty
  curl -s -X POST http://127.0.0.1:5700/v2/repository/models/gpu_image_prep/unload; sleep 10
  curl -s -X POST http://127.0.0.1:5700/v2/repository/models/gpu_image_prep/load
}
```

Every GPU step runs in a session script under `flock -w 3600 $SCRATCH/locks/gpu0.lock`, with the
preflight of the SP0 runbook (GPU 0 idle, timed `/health`, the ten stage counters listed). Row names
become project slugs `sp0-<row>-r<i>-<stamp>`, so a row name has at most 14 lower-case characters.
Screening rows use `--reps 1`; a row that decides something uses 3. Delete only the `sp0-*`
throw-away projects, one `DELETE` per slug, as the runbook does.

### 4.1 Samplers attached to every row (E2 and E3; no code)

```bash
START=$(date -Is)
( while :; do echo "t $(date +%s.%N)"; nvidia-smi --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader; sleep 0.5; done ) > "$ART/$ROWNAME.apps.log" & APPS=$!
( while :; do printf '%s ' "$(date +%s)"; curl -s -m 2 http://127.0.0.1:5703/metrics \
    | rg '^op_ingest_decode_total' | tr '\n' ' '; echo; sleep 1; done ) > "$ART/$ROWNAME.routes.log" & ROUTES=$!
row "$ROWNAME" 4 3
kill $APPS $ROUTES
docker logs --since "$START" optest2-api 2>&1 | rg -o 'gpu_image_prep_call_failed.{0,400}' \
  | rg -o 'error[=:"][^,}]{0,200}' | sort | uniq -c | sort -rn > "$ART/$ROWNAME.call_failed.txt"
docker logs --since "$START" optest2-triton 2>&1 \
  | rg -i 'out of memory|outofmemory|cudaErrorMemoryAllocation|nvjpeg|gpu_image_prep' | cut -c1-240 \
  | sort | uniq -c | sort -rn | head -50 > "$ART/$ROWNAME.triton_errors.txt"
```

The `decoder` statuses carry no text. For E2 only, a scratch edit of the **installed** copy
(`$INST/models/gpu_image_prep/1/model.py`, never the repository) adds one line in `_decode`'s
`except Exception as exc:` branch, `sys.stderr.write(f'gpu_image_prep decode: {type(exc).__name__}: {exc}\n')`
(and `import sys`), followed by `reload_prep`; the branch files are restored with
`rsync -a --exclude __pycache__ $WT/models/gpu_image_prep/ $INST/models/gpu_image_prep/` and a reload.

Signatures. H1: `call_failed.txt` and the decode lines name `OutOfMemoryError`,
`cudaErrorMemoryAllocation` or nvJPEG status 5 (allocator failure); in `apps.log` the card total
reaches about 48.4 GB and a stub's memory stops growing at the moment `routes.log` shows
`cpu_decoder` and `cpu_call_failed` starting to climb; the stub that grows over the session is the
one to fix (if `tritonserver` itself grows, H6 and the server side move up). Not H1: other exception
types (an nvJPEG parse error, a timeout from the 120 s `stream_timeout` in `stream_one`).

### 4.2 The experiments

Cost is GPU-locked minutes, including restarts. "Sweep" marks the config-only experiments the
running sweep agent can do; the others need a scratch overlay or the scratch edit above.

**E0b. Resident-memory census (5 min, sweep).** After `$COMPOSE up -d triton-server api opensearch`
and health: `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv` once; then
unload each model in turn through Triton model control and record the drop
(`curl -s -X POST http://127.0.0.1:5700/v2/repository/models/<name>/unload`, then load it back).
Output: MiB per model on this stack. Explains (or not) the 36.5 against 23.1 GB gap.

**E1. Headroom A/B, fresh stack, 8 batches first (20 to 25 min, sweep). Decides H1.**

```bash
$COMPOSE restart triton-server   # fresh server and stubs; wait for /v2/health/ready and /health (about 60 s)
curl -s -X POST http://127.0.0.1:5700/v2/repository/index | "$VPY" -c "import json,sys; keep={'yolov11_small_trt_end2end','pe_image_encoder','pe_text_encoder','gpu_image_prep'}; print(' '.join(m['name'] for m in json.load(sys.stdin) if m.get('state')=='READY' and m['name'] not in keep))" > "$ART/unloaded.txt"
for m in $(cat "$ART/unloaded.txt"); do curl -s -X POST http://127.0.0.1:5700/v2/repository/models/$m/unload; done
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$ART/e1_apps_start.csv"
reload_prep
ROWNAME=pc8hr; row pc8hr 8 3      # with the 4.1 samplers around it
reload_prep
ROWNAME=pc4hr; row pc4hr 4 3
for m in $(cat "$ART/unloaded.txt"); do curl -s -X POST http://127.0.0.1:5700/v2/repository/models/$m/load; done
```

Then the control on the full default model set, also fresh and 8 batches first:
`$COMPOSE restart triton-server`, `reload_prep`, rows `pc8full` and `pc4full`.
Check first that `/health` and the ingest route work with those models unloaded (the ingest path
calls the detector, PE and `gpu_image_prep` only); if `/health` refuses, stop and report.

Signatures. H1 confirmed: in `pc8hr` and `pc4hr` fallbacks at most 0.5 percent, API CPU about 0.07,
img/s at 8 batches not below 4 batches, the launch stages fall to at most about twice their one-batch
values; `pc8full` reproduces the collapse even on a fresh stack. H1 refuted: fallbacks stay above a
few percent with the headroom, or the collapse persists with fallbacks near 0 (then H2 to H5 lead).
Session-age check: if `pc4full` repetitions still decay (rep 1 low, rep 3 high), the accumulation is
in the stub (pool fragmentation, nvJPEG buffers); E3 names the process.

**E2, E3.** The samplers of 4.1 on every row of E1 and E4.

**E4. Pool size (15 min per value, sweep).** In `$INST/models/gpu_image_prep/config.pbtxt` set
`POOL_MIB` to 2048, 4096, 8192 (one value per load, `reload_prep`), on the headroom stack of E1,
`row pc4p<value> 4 1`. E4b: the same three values on the full default stack. Signatures: H1b
(arena churn at the cap): with headroom, launch stages and fallbacks fall as the pool grows and rise
at 2048; card-bound H1: on the full stack 8192 makes fallbacks worse (the pool then competes with the
card), with headroom it changes little.

**E5. CUDA API census with Nsight Systems (40 min; scratch overlay).** The host has
`/opt/nvidia/nsight-systems/2026.3.2`. A scratch overlay `$SCRATCH/sp/docker-compose.nsys.yml` (never
committed) for `triton-server`: volumes `/opt/nvidia/nsight-systems/2026.3.2:/opt/nsys:ro` and
`$SCRATCH/sp/nsys:/nsys` (absolute paths written out), and
`entrypoint: ["/opt/nsys/bin/nsys", "launch", "--session-new=spd", "--trace=cuda,nvtx,osrt", "--sample=none", "--cpuctxsw=none", "--trace-fork-before-exec=true", "--cuda-memory-usage=true", "--wait=all"]`
(the compose `command` keeps `tritonserver ...`). Recreate Triton with that overlay added to
`$COMPOSE`; during a 1-batch row and during a 4-batch row, 60 s into a repetition:

```bash
curl -s http://127.0.0.1:5702/metrics | rg 'op_gpu_prep_stage_observations\{stage="decode"' > "$ART/nsys_c4_before.txt"
docker exec optest2-triton /opt/nsys/bin/nsys start --session=spd --output=/nsys/c4 --force-overwrite=true
# 30 s later (inside the session script)
docker exec optest2-triton /opt/nsys/bin/nsys stop --session=spd
curl -s http://127.0.0.1:5702/metrics | rg 'op_gpu_prep_stage_observations\{stage="decode"' > "$ART/nsys_c4_after.txt"
/opt/nvidia/nsight-systems/2026.3.2/bin/nsys stats --format csv --output "$ART/nsys_c4" \
  --report cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_size_sum,cuda_gpu_mem_time_sum,osrt_sum "$SCRATCH/sp/nsys/c4.nsys-rep"
```

Normalise by images (decode observations after minus before). Signatures: H1b: `cudaFree` calls in
steady state (each an implicit device sync) and `cudaMalloc` well above 0.1 per image, growing from
1 to 4 batches; H2 and H7: about 6 pageable host-to-device copies per crop plus one pageable
device-to-host copy per crop, and `cudaStreamSynchronize` or implicit-sync time per image that grows
with load; H5: kernel launches per image times the mean API call time under 5 ms per image (then H5
alone is negligible); H8: any prep kernel or copy on the legacy default stream; H6: host-to-device and
device-to-host copies of 1.2 MB (canvas) and 0.34 MB (square) sizes in the server process (BLS bounced
through host memory). Remove the overlay and recreate Triton afterwards.

**E6. py-spy on the stubs (20 min; no code).** `py-spy` goes into the main venv
(`$MAIN/.venv/bin/pip install py-spy`), its binary is copied into the container, the stub pids come
from `docker exec optest2-triton ps -eo pid,args` (never `pgrep -f`):

```bash
docker cp "$MAIN/.venv/bin/py-spy" optest2-triton:/tmp/py-spy
docker exec optest2-triton ps -eo pid,args | rg 'triton_python_backend_stub' | rg gpu_image_prep
# for each stub PID, during a 4-batch repetition and during a 1-batch repetition:
docker exec -u 0 --privileged optest2-triton /tmp/py-spy record --native --idle --gil -r 100 -d 30 \
  -p "$PID" -f speedscope -o /tmp/spd_c4_gil_$PID.json
docker exec -u 0 --privileged optest2-triton /tmp/py-spy record --native --idle -r 100 -d 30 \
  -p "$PID" -f speedscope -o /tmp/spd_c4_all_$PID.json
docker cp optest2-triton:/tmp/spd_c4_gil_$PID.json "$ART/"; docker cp optest2-triton:/tmp/spd_c4_all_$PID.json "$ART/"
```

Signatures: H3: in the `--idle --native` profile a large share of `fe-image` thread samples in
`take_gil` / `PyEval_RestoreThread`, and the `--gil` profile near 100 percent of the wall (some
thread always holds the GIL), its time in CuPy launch and array-construction frames. H1b and H2:
worker samples in `cuMemFree`, `cudaStreamSynchronize`, `cuMemcpyHtoD` or `cuMemcpyDtoH` under
`_weights`, `_crop_pixels`, `lap_sums`. Waiting on other models: samples in the BLS `exec`.

**E7. Thread and process shape (10 min per row, sweep).** On the headroom stack, one factor at a
time, `row <name> 4 1`: `WORKERS` 8, 16, 32 at `count: 2`; then at about equal total threads
`count: 1` with 32 workers, `count: 2` with 16, `count: 3` with 12, `count: 4` with 8 (each a config
edit and `reload_prep`; the instance count needs the memory: skip a count that would put the card
above 47 GB at start). Signatures: H3: fewer threads per process lowers the launch stages strongly
at equal or better img/s, and more processes at equal threads win; H4: more processes lose or gain
nothing (then E8).

**E8. MPS on and off (30 min; host daemon and a scratch overlay; only if E7 points at H4 or H3).**
The container user is uid 1000, the same as the host user, so a host MPS daemon accepts it:

```bash
MPS=$SCRATCH/sp/mps; mkdir -p "$MPS/pipe" "$MPS/log"
export CUDA_VISIBLE_DEVICES=0 CUDA_MPS_PIPE_DIRECTORY=$MPS/pipe CUDA_MPS_LOG_DIRECTORY=$MPS/log
nvidia-cuda-mps-control -d
# scratch overlay docker-compose.mps.yml for triton-server: ipc: host, volume $MPS/pipe:/tmp/nvidia-mps,
# environment CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps; recreate triton-server with it added to $COMPOSE
nvidia-smi   # expect nvidia-cuda-mps-server, and the Triton processes listed as M+C
# rows pc4mps (count 2) and pc4mps1 (count 1), 3 reps each
# after: recreate triton-server without the overlay, then stop the daemon
echo quit | CUDA_MPS_PIPE_DIRECTORY=$MPS/pipe nvidia-cuda-mps-control
```

Signatures: H4: at count 2 MPS raises img/s by at least 10 percent and GPU util, at count 1 it
changes little. No change: drop MPS.

**E9. STAGE_SYNC at 4 batches (10 min, sweep).** `STAGE_SYNC` "1", headroom stack, `row pc4sync 4 1`.
Attribution only. Signature for GPU contention inside the stages (time-slicing, H4): the launch
stages' time moves into kernel time (the stage grows with sync and its host part shrinks); H2 and H3:
the stages stay host-side as at one batch.

**E10. Triton Python backend shared memory and CUDA pool (20 min; compose args in a scratch overlay).**
Add to the server command `--backend-config=python,shm-default-byte-size=67108864`,
`--backend-config=python,shm-growth-byte-size=67108864` and `--cuda-memory-pool-byte-size=0:536870912`
(check the 26.06 Python backend README for the key names first), recreate, `row pc4shm 4 3`.
Signature for H6: queue and the detect and embed_pe stages fall, and E5's server-side bounce copies
disappear. Nothing: reject H6.

**E11. API side after H1 is removed (15 min, sweep; env only).** On the headroom stack, `row pc8hr`
against the same with `OP_MAX_INGEST_CONCURRENCY=32` on the API (overlay environment, API recreated).
Signature for H9: `opensearch_write` or `admission_wait` grow with clients while the model's `queue`
stays low; img/s moves with the API setting.

Order: E1 with E2 and E3 (decides H1), E0b, E4, E6 and E5 (they read the same rows), E7, then E8 only
on evidence, E9, E10, E11. About 3 to 4 hours of GPU 0 in total; E1 alone about 25 minutes.

## 5. Code changes if the hypotheses are confirmed

Each change is one commit on `perf/v070-ingest-speed`, red-first tests, `pre-commit` clean, no push.
Rollback of any step is `git revert` of its commit, then
`rsync -a --exclude __pycache__ $WT/models/gpu_image_prep/ $INST/models/gpu_image_prep/` and a model
reload through Triton model control (the model files are hot-swappable; nothing persists). No runtime
switch keeps the old path (project rule: no shims); the parity tests in section 6 are the safety net.

### C1. Memory pressure waits, is counted, and cannot exist at load (if H1 or H1b holds; first)

Files: `models/gpu_image_prep/1/model.py`, new `models/gpu_image_prep/1/prep_budget.py`,
`models/gpu_image_prep/1/nvjpeg_lanes.py`, `models/gpu_image_prep/config.pbtxt`.

1. **Byte-budget admission.** `prep_budget.py` (host only, testable offline):
   `need_bytes(header, net, pe_size, crop_allowance) -> int` from the JPEG header that `route()`
   already parses: frame `w*h*3`, plus a second frame when the EXIF orientation is not 1 (the
   transposed copy in `_orient`), plus the largest resample scratch (`rows x window_w x 3 x 4` for
   the letterbox, the frame square), canvas and squares, plus a crop allowance (a parameter, fraction
   of the frame). `ByteBudget(total)` with `acquire(n)` (FIFO tickets on a `threading.Condition`,
   so a large photo is not starved) and `release(n)`. In `TritonPythonModel._run`, the header is
   parsed first (`route()` moves out of `_decode` into `_process`), then `budget.acquire(need)`
   before the decode and `release` in `finally`. A photo whose `need` exceeds the whole budget gets a
   new status `over_budget` (CPU path, counted `cpu_over_budget` by the existing
   `count_route(f'cpu_{outcome}')`); the API needs no change for this. Each request acquires once
   (no second acquisition mid-request), so the budget cannot deadlock. Crops beyond the allowance are
   processed in chunks inside it.
2. **A failed allocation is a status, not a decoder failure.** In `_decode`, `cp.cuda.memory.OutOfMemoryError`
   and a `CUDARuntimeError` with `cudaErrorMemoryAllocation` (and nvJPEG status 5) return status
   `gpu_memory`; other exceptions stay `decoder`. In `_run`, the same classes caught around
   `_process` also answer status `gpu_memory` (an `InferenceResponse` with the status tensor) instead
   of an error response, so the API counts `cpu_gpu_memory` rather than `cpu_call_failed`. One log
   line per 100 such events (the exception class and the pool's `used_bytes`/`total_bytes`). With the
   budget this status should not occur; when it does, the counter says what happened.
3. **One arena for the large buffers.** Frames, orientation copies and resample scratch are allocated
   under one dedicated allocation stream (`self.alloc_stream`, used for no work), so a block freed by
   one worker is reusable by any other; `NvJpegLanes.decode` takes the stream to allocate `out` on.
   Invariant (documented in the module docstring, asserted by the offline test's stand-in): a buffer
   from that arena is dropped only after its worker stream was synchronised (today the last
   `cp.asnumpy` in `_blur_sums` and the BLS calls already guarantee it; stored-box mode gets an
   explicit `stream.synchronize()` before `im.frame = None`).
4. **Reserve at load, fail at load.** `initialize` allocates the budget once through the pool and
   frees it back (the pool keeps it cached), after the nvJPEG lanes are created. A card that cannot
   hold it makes the load fail with a message naming `DEVICE_BUDGET_MIB` and the free memory; the
   model is then UNAVAILABLE and `gpu_decode_ready` keeps every photo on the CPU path, visibly
   (#391 covers making `/health` say so). This replaces runtime degradation with a load-time refusal.
5. Parameters: `DEVICE_BUDGET_MIB` (new; default 3072, 75 percent of `POOL_MIB` so fragmentation has
   slack) and `CROP_ALLOWANCE` (fraction of the frame). `GPU_MIB_PER_INSTANCE` stays 8,704 and
   becomes a bound the code enforces rather than an observation; `scripts/lib/cli_gpu.sh` and the
   installer read it unchanged. `MAX_IN_FLIGHT` stays as the cap on requests held in `execute`.

Risk: medium (concurrency code; a budget too small lowers throughput, too large brings the
failures back). Tests: section 6.1 and 6.2. Expected effect: fallbacks to at most 0.5 percent on the
default stack, API CPU back to about 0.07, and the session-age decay gone; img/s near the clean
33 img/s point or above.

### C2. Fewer syncs and allocations per image (if H2, H5 or H7 holds; second)

Files: `models/gpu_image_prep/1/prep_kernels.py`, new `models/gpu_image_prep/1/prep_plan.py`
(host-only job planner), `models/gpu_image_prep/1/model.py` (`_detect`, `_embed`, `_crop_pixels`,
`_blur_sums`), new per-thread scratch in `model.py` (`self.local`).

1. **One packed upload per phase.** `prep_plan.plan_jobs(frame_size, jobs) -> (host_buffer, offsets)`
   computes every filter weight table of a phase with `prep_geometry.filter_weights` and packs them,
   with an int64 job table, into one per-thread pinned host buffer (`cp.cuda.alloc_pinned_memory`,
   grown geometrically, never shrunk) copied with one `cudaMemcpyAsync` (pinned, so no stream wait).
   Phase 1 (before the detector): the letterbox job and the whole-frame square job. Phase 2 (after
   the detector): every crop square, the crop gather table and the blur rectangles. `_weights` and
   its 4,096-key cache are deleted.
2. **Batched resample kernels.** `resample_h_jobs` and `resample_v_jobs` take the job table
   (`grid.z` = job): per output pixel the same expression in the same order as `resample_h` and
   `resample_v` today, compiled by the same `cp.RawModule` with the same options, so the output bytes
   are identical (the parity test proves it, 6.3). The canvas fill becomes one `cudaMemsetAsync`
   with 114 on a per-thread canvas instead of `cp.full`. Launches per image: 2 + 2 instead of
   2 x (2 + crops).
3. **One device-to-host copy for all crops.** A `gather_crops_bgr` kernel copies every detection's
   `crop_pil` rectangle, channels reversed as `[..., ::-1]` does today, into one device staging buffer;
   one `cudaMemcpyAsync` moves it with the blur sums into the per-thread pinned buffer; numpy views per
   crop feed `cv2.imencode` unchanged. Replaces 2.41 `ascontiguousarray` plus 2.41 pageable `.get()`
   per image.
4. **Syncs per image** after C2: the nvJPEG lane sync, the detector BLS, one sync before the host reads
   the crops and sums, the PE BLS (about 4 to 5, from about 25). Allocations per image in steady
   state: the frame (and an orientation copy) only.

Risk: medium to high (new kernels; the bit-identity test is the gate). Rollback as above.

### C3. Concurrency shape (only on E6 and E7 evidence; third)

- If H3: lower `WORKERS` and raise the instance `count` to the E7 optimum (config, per profile in
  `config_templates/profiles/*.json` as `gpu_image_prep_instances` already is), with the memory
  budget of C1 making each instance's footprint a known constant. If py-spy shows a convoy with few
  threads, `sys.setswitchinterval` at 1 ms in `initialize`, behind a `SWITCH_INTERVAL_MS` parameter
  with a comment giving the measured reason.
- If H4: an opt-in `docker-compose.mps.yml` overlay (MPS control daemon in a side-car with
  `ipc: host` and the pipe directory, Triton joined to it), documented next to
  `docker-compose.gpu-arbiter.yml`. Owner decision (section 9): MPS shares one server per card and a
  fault in one client ends every client on that card.
- A C++ or native backend stays v0.8.0 (#40 "profile Python/GIL overhead per stage; native only if it
  dominates after the data-volume work").

### C4. JPEG encode off the worker's critical path (if the worker pool is the limit after C1 and C2)

`_process` returns after the GPU part; the crop encodes are submitted to `encode_pool` with a done
callback that, when the last crop of the image finishes, builds the response, publishes the stage
clock and calls `response_sender.send` (decoupled models may send from any thread), then releases
`in_flight`. The budget and the worker thread are released when the GPU buffers are, not after the
encodes. Risk: low to medium (error paths: an encode failure still answers an error for that image;
tested with the stand-ins). SP3 (#219, `crop_max_side`) shrinks the same encode work and stays as
planned.

### C5. Triton flags (only if E10 shows a gain)

`--backend-config=python,shm-default-byte-size=...`, `shm-growth-byte-size=...` and
`--cuda-memory-pool-byte-size=0:...` in `docker-compose.yml`'s `triton-server` command with a comment
giving the E10 numbers. Deployment-wide; low risk.

### C6. API side (none required by H1)

`count_route(f'cpu_{outcome}')` already counts any new status. Document the two new labels
(`cpu_gpu_memory`, `cpu_over_budget`) where the route labels are described (`docs/CURATION.md` metrics
list and the PERFORMANCE "0.7.0 stage profile" notes). #337's item (charge the admission gate for
fallbacks) becomes unnecessary for ingest if C1 holds the fallback rate near 0; say so on #337.

## 6. Tests (red first; outputs bit-identical or within a stated tolerance)

### 6.1 Offline (no GPU; the suite's stand-ins for CuPy, nvJPEG and `pb_utils`)

- `tests/models/test_gpu_image_prep_budget.py` (new): `need_bytes` against hand-computed values for
  the orientations and sizes of `test_gpu_image_prep_geometry.py`; `ByteBudget` FIFO order, no
  deadlock with 64 threads and random sizes, `over_budget` for a need above the total, release in
  `finally` on an exception.
- `tests/models/test_gpu_image_prep_stage_metrics.py` (extend): an `OutOfMemoryError` raised by the
  stand-in at each allocating stage answers status `gpu_memory` (not an error response, not
  `decoder`); a non-memory decode exception still answers `decoder`; every stage still observed once
  per image; outputs equal with and without the budget for a sequence of images.
- `tests/models/test_gpu_image_prep_plan.py` (new, C2): the packed tables reproduce, per job,
  `filter_weights` sliced as `_weights` sliced it today (first tap, count, weights) for letterbox,
  frame square and crop squares at the sizes of the geometry tests; the job table's offsets and
  strides address the same destination windows as the per-call path; `_resample` (the float64 host
  model of the kernels in `test_gpu_image_prep_geometry.py`) run per job equals the per-call result.
- `tests/curation/test_ingest_gpu_decode.py` and `tests/curation/test_reprocess_gpu_embed.py`: a
  `gpu_memory` and an `over_budget` status are counted as `cpu_gpu_memory` and `cpu_over_budget` and
  take the CPU path (fail closed), in ingest and in the embed pass.
- Unchanged and still green: `tests/models/test_gpu_image_prep_geometry.py`,
  `tests/models/test_gpu_image_prep_config.py`, `tests/curation/test_crop_is_stored_box.py`,
  `tests/test_gpu_image_prep_instances.py`, `tests/test_triton_instance_budget.py`.

### 6.2 Live output parity of the whole model (new `scripts/bench/gpu_prep_parity.py`, GPU 0 under the lock)

`capture --url 127.0.0.1:5701 --images <list> --concurrency N --out cap.npz` streams each JPEG to
`gpu_image_prep` directly (the model's `MIN_MEGAPIXELS` is 0, so the 200 pinned COCO images and the
400 set B photos both take the GPU path) and keeps every output tensor; `compare a.npz b.npz` writes
the table. Captures: the current branch model and the changed model, each at concurrency 1 (twice,
to measure the floor) and 8.

- C1 and C4 (no arithmetic change): at concurrency 1, `status`, `size`, `letterbox`, `det_*`,
  `blur_sums` and `crop_jpegs` bytes identical; embeddings identical or within the current model's
  own concurrency-1 floor. At concurrency 8: within E4's run-to-run noise (det_boxes max abs diff at
  most 3.4e-3 after sorting by score, whole_embedding cosine at least 0.9999, crop_embeddings minimum
  cosine at least 0.988, at most 1 percent of crop JPEGs differing).
- Stored-box mode: the same capture with `boxes` rows (the embed pass): `crop_embeddings` row count and
  order unchanged; vectors within the floor.

### 6.3 Kernel bit-identity (C2; new `scripts/bench/gpu_prep_kernel_parity.py`, in the Triton image, GPU 0)

Decodes each image once with `NvJpegLanes`, then builds the letterbox canvas, the frame square, the
crop squares, the gathered crop pixels and the lap sums twice, with the old `PrepKernels` (from the
merge base via `git show`) and the new batched path, and compares bytes: **all identical** on the 200
COCO images, the 400 set B photos and every EXIF orientation fixture. Any differing byte fails C2.

### 6.4 Rules that must keep holding

- Decode once: `op_gpu_prep_stage_observations{stage="decode"}` equals the requests;
  `op_source_image_decode_count_total` delta 0 during ingest; `gpu` plus every `cpu_*` route equals
  the photos (no photo decoded on both paths).
- Stored box: `stored_box_rects` unchanged; `tests/curation/test_crop_is_stored_box.py` green; the
  embed pass (log G4 harness, 1,499 photos) gives crop, frame and region vectors with minimum cosine
  at least 0.9996 against the current code.
- Fail closed: a photo is either fully processed on the GPU or answered with a counted status and
  taken by the CPU path; never a partial output. A load that cannot reserve its budget fails the load.
- #40 parity gates against the 0.6.0 reference (`scripts/bench/parity.py`): P-box at least 0.98,
  P-item at least 0.99, P-pe crop cosine p01 at least 0.995.

## 7. Acceptance bars

Measured with the protocol of the ship-track plan section 3 (3 repetitions, cache drop, 500 ms GPU
samples, load above 16 marks a run noisy), **on the default stack** (all default models loaded, the
condition that failed), fresh stack per row, the 8-batch row first.

| Bar | Value |
|---|---|
| SP4 (#40) | set P at `--clients 4`: median at least 29 img/s and GPU util mean above 75 percent |
| No collapse | set P at `--clients 8`: median at least the `--clients 4` median minus 5 percent |
| No session decay | repetition 3 within 10 percent of repetition 1 in each row |
| Fallbacks | `cpu_decoder` + `cpu_call_failed` + `cpu_gpu_memory` at most 0.5 percent of GPU-eligible photos at 4 and 8 batches |
| API CPU | at most 0.08 CPU-s per image (set P, 4 and 8 batches) |
| 48 GB card | peak card memory at most 47.0 GiB; each `gpu_image_prep` stub at most `GPU_MIB_PER_INSTANCE` (8,704 MiB) in the 500 ms per-pid samples |
| 24 GB card | emulated by a ballast process holding all but 24 GiB of GPU 0 (`docker run --rm -d --name spd-ballast --gpus '"device=0"' davidamacey/openprocessor-triton:0.6.0 python3 -c "import cupy,time; x=cupy.empty(24564*2**20, cupy.uint8); time.sleep(10**9)"`), the ingest models only (detector, PE, PE text, `gpu_image_prep` count 1): set P at 4 batches completes with fallbacks at most 0.5 percent and no `gpu_memory` status; img/s recorded, no bar |
| 12 GB card (#42) | no regression: GPU decode stays off by default on the `standard` profile; with a ballast leaving 12 GiB, count 1 with `DEVICE_BUDGET_MIB` 1536 either loads next to the detector and PE in at most 11 GB or refuses at load (never fails at runtime); recorded |
| Quality | 6.2 to 6.4 green: bit-identical where the arithmetic did not change; #40 parity gates green |
| Set B (public) | `--clients 4` and `--clients 8` rows in `docs/PERFORMANCE.md` under "0.7.0 stage profile" with fallbacks reported |

A bar not met is reported as not met with the binding stage named (protocol item 8).

## 8. Briefs

### 8.1 What the running config-sweep agent should hand over

It changes only configuration (`config.pbtxt` values, instance count, Triton model control), on GPU 0
under the lock. Ask it to hand over, per row: the row name, the exact config diff against the branch
`config.pbtxt`, whether the stack or only the model was restarted before the row and what ran before
it in the same session, the per-repetition JSON (already contains the route counters and the stage
observations), the fallback fraction per repetition computed as in 2.1, the stage-observation drops
per stage as in 2.2, `nvidia-smi --query-compute-apps` at start and the per-pid 500 ms memory log
during the row (4.1), the API `gpu_image_prep_call_failed` error strings with counts and the Triton
log error lines (4.1), the 1-minute load at start and end, and the Triton resident memory after
startup. Rows wanted if not yet run, in this order: E1 (`pc8hr`, `pc4hr`, `pc8full`, `pc4full`,
fresh stack, 8 batches first), E0b, E4 and E4b, E7, E9, E11. It should not conclude a ranking from
img/s alone while fallbacks differ between rows, and should restore the branch `config.pbtxt`, reload
the unloaded models and run `$COMPOSE down` (no `-v`) at the end.

### 8.2 Brief for a fresh Sonnet agent (four steps; stop at a failed gate and report)

Context to read first: this file, `docs/reviews/sp0_review.md` (runbook and pitfalls),
`models/gpu_image_prep/1/*.py`, `tests/models/test_gpu_image_prep_stage_metrics.py`, the sweep
agent's handover. Work in `$WT` (`perf/v070-ingest-speed`); venv binaries from `$MAIN/.venv/bin`;
every GPU step in a session script under `flock -w 3600 $SCRATCH/locks/gpu0.lock`; `docker compose
-p optest2` only; never touch `optest1` or GPU 1 and 2; conventional commits, `pre-commit` on every
commit, never `--no-verify`, no push. Set P numbers only, in `docs/research/v070_measurement_log.md`.

1. **Diagnose (GPU, no product code; about 3 hours).** Run what the handover lacks of E1, E2, E3,
   E5, E6 (then E7 and E8 only if E6 or E7 point at H3 or H4; E10 only if E5 shows server-side
   bounce copies). Write a log section "SPD: concurrency diagnosis" with a verdict table: for H1 to
   H9, the signature predicted in section 4, the one observed, confirmed or rejected. Commit raw
   public files under `docs/benchmarks/v070/spd/` with `MANIFEST.sha256`. Gate: H1 confirmed or
   rejected by E1 with 3 repetitions per row. If H1 is rejected, stop after this step and report.
2. **C1 (memory-safe admission).** Tests of 6.1 first (seen failing), then `prep_budget.py`, the
   `gpu_memory` and `over_budget` statuses, the shared arena, the load-time reservation, the two
   parameters, the label docs (C6). Live: 6.2 captures (bit-identical at concurrency 1), then set P
   rows on the default stack at 8 and 4 batches, fresh stack each, 3 repetitions. Gate: fallbacks at
   most 0.5 percent and no session decay. Commit `perf(gpu-prep): ...`.
3. **C2 (fewer syncs and allocations), only if step 1 confirmed H2, H5 or H7.** `prep_plan.py` and its
   tests first, then the batched kernels, the memset canvas, the gather kernel and the pinned
   buffers. Gate: 6.3 all bytes identical, 6.2 green, set P at 4 batches not slower than after step 2
   and the launch stages lower. Commit separately.
4. **Concurrency shape and acceptance.** C3 per the E6 and E7 evidence (MPS only with the owner's yes),
   C4 if the worker pool is still the limit, C5 if E10 showed a gain. Then the section 7 rows, the
   24 GB and 12 GB emulations, the embed pass regression (log G4 harness), the #40 parity gates.
   Write `docs/reviews/spd_review.md` in the template of the ship-track plan section 12, the public
   set B rows in `docs/PERFORMANCE.md`, a CHANGELOG entry, and a status update for this file's row
   in `docs/design/README.md`. Report the SP4 bar as met or not met with the binding stage.

## 9. Decisions needed from the owner

1. **Acceptance stack.** Recommended: SP4 is measured on the default stack (all default models loaded,
   36.5 GB resident on the SP install), because that is what users run; a headroom stack only as the
   diagnostic row. Alternative: measure on an ingest-only model set and document it.
2. **MPS.** Recommended: only as an opt-in overlay, and only if E8 shows at least 10 percent; never
   on by default (fault isolation, one server per card).
3. **Order.** Recommended: this item (C1 at least) before SP3; SP3's gain is bounded by its 9.5 to
   21.6 percent stage share, C1's by the 2.3 to 72 percent fallbacks.
4. **The resident-memory gap** (`tritonserver` 36.5 GB on the SP install against 23.1 GB on `optest1`,
   same engines and configs): file as its own issue after E0b, since it decides whether the full
   profile's two instances fit a 48 GB card at all.
5. **Load-time refusal** (C1 item 4): an instance that cannot reserve its budget fails to load and
   ingest runs on the CPU path, visibly. Recommended yes (fail closed; #391 makes it visible in
   `/health`).

## 10. What this review does not claim

- That H1 is proven: the exception text was never captured. The route counters, the stage-observation
  identity, the memory peaks and the session-age pattern make it the leading explanation; E1 and E2
  decide it.
- Any img/s gain from C1 to C4; the figures in section 1 item 9 are bars, not predictions. The one
  clean data point (33.1 img/s with 2.3 percent fallbacks) suggests C1 alone brings the stack close
  to the SP4 bar; that is an inference from one repetition.
- That CuPy 13.6 frees all cached blocks and runs `gc.collect()` on a failed allocation: stated from
  CuPy's pool design, to be confirmed by E5's `cudaFree` counts.
- Anything about the CPU route (set A), which SP0 did not run at 4 batches.
