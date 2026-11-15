---
sidebar_position: 3
title: Running the native engine comparison (N0)
---

# Running the native engine comparison (N0)

This page is for the person who runs the first measurement of the native frame engine
against the Python one ([plan 17](https://github.com/davidamacey/OpenProcessor/blob/main/docs/design/v070_rebuild/17_native_engine_plan.md),
phase N0). It needs a GPU and Docker; the tools it uses are built and tested without
a GPU. Every number the plan's bars name comes out of one of the holds below, with the evidence a
paper table needs beside it.

The measure is photos per second from "photo in" to "answered", judged two ways: the **engine's
capacity** (the same transport with stand-in models that cost the card nothing) and **the card's
ceiling** (the models alone, with their inputs already resident). One client, `op-fe-loadgen`,
measures both engines, so a difference is the engine's.

## What is in the test bed

| Part | Where |
|---|---|
| The measuring client: `engine` (a closed loop of photos in flight against any engine at an address), `z` (the models alone), `ready`, `regions` | `native/frame_engine_loadgen/` |
| The photo lists, built from header reads | `scripts/bench/frame_engine_sets.py` |
| The stand-in models: null (cost nothing, answer constants) and capture (record every input row) | `scripts/bench/frame_engine_stand_ins.py` |
| The plans each photo carries, the parity rows and the comparison of the capture models' inputs | `scripts/bench/gpu_prep_parity.py` |
| One hold (S0, H1, H2, H3) and the fault holds | `scripts/bench/frame_engine_hold.sh`, `frame_engine_faults.sh` |
| The model server, the native engine and the Python engine on one card, project `op-n0` | `docker/frame-engine-native/` |

The plan text of every row is written by the API's own builder (`gpu_prep_parity.py plan`) and
sent byte for byte by the load generator, so the rule that builds a plan exists once.

## Prepare (once)

```bash
# the load generator and the engine (a build, no GPU)
cargo build --release --manifest-path native/Cargo.toml -p frame_engine -p frame_engine_loadgen

# the two photo sets, then the lists (headers only; the report holds each list's SHA-256)
python scripts/datasets/fetch_coco_subset.py --out data/samples/coco_bench_2000 --bench-set 2000 \
  --seed 20261009 --manifest scripts/datasets/manifests/coco_bench_2000.json
python scripts/datasets/fetch_hires_set.py --out data/samples/hires_500 --n 500 --seed 20261010 \
  --manifest scripts/datasets/manifests/commons_hires_500.json
python scripts/bench/frame_engine_sets.py --preset coco \
  --manifest scripts/datasets/manifests/coco_bench_2000.json \
  --images-dir data/samples/coco_bench_2000/images --out-dir artifacts_local/native_engine/n0/lists
python scripts/bench/frame_engine_sets.py --preset hires \
  --manifest scripts/datasets/manifests/commons_hires_500.json \
  --images-dir data/samples/hires_500/images --out-dir artifacts_local/native_engine/n0/lists
# set P (private; only counts and rates ever leave the host): the same header rule on your own list
python scripts/bench/frame_engine_sets.py --preset custom --name p --seed 1 --n 200 \
  --from-list <your list> --out-dir artifacts_local/native_engine/n0/lists

# the stand-in models: configs, the capture code and the ONNX graphs, then the TensorRT plans
python scripts/bench/frame_engine_stand_ins.py generate --onnx \
  --out artifacts_local/native_engine/n0/stand_in
python scripts/bench/frame_engine_stand_ins.py plan-commands --run

# the three images (the Python engine's is the frozen reference)
docker compose -p op-n0 -f docker/frame-engine-native/compose.n0.yml build
docker compose -p op-n0 -f docker/frame-engine-native/compose.n0.yml up -d
```

The pinned public sets are kept on disk under the data directory you name with `--out` and are
verified against the pin, not downloaded again, when they are already there; only absent or
changed files are fetched. Public files do disappear: a file the server answers 404 for is
recorded in `missing_upstream.txt` beside the images, the rest still download, and the end report
lists the missing ones. The fetch fails (exit 1) on a corrupt file, on any missing file with
`--strict`, or when more than `--max-missing-share` (default 0.10) of the pinned files is missing.
`--verify-only` hashes what is on disk without the network and reports the present, missing
upstream, not downloaded and corrupt counts apart. A list built over an incomplete set is not the
pinned list: the report of the run says so.

Always pass `-p op-n0`: the default project name is the live stack.

## Holds

Each hold is exclusive: it takes the hold file you give it with `flock -o` (waiting at most a
minute), checks that the card at `--gpu-index` has the bus id you expect before anything runs,
refuses a card another tenant is using, waits for the 1-minute load to be under 16, and runs every
command under `timeout --kill-after`. GPU 2 here is `00000000:84:00.0`; check with
`nvidia-smi --query-gpu=index,pci.bus_id --format=csv`.

```bash
HOLD=/path/the/coordinator/gives            # the lock every job on the card agrees on
BED="--hold-file $HOLD --gpu-index 2 --gpu-bus-id 00000000:84:00.0 \
     --lists artifacts_local/native_engine/n0/lists --capture-dir artifacts_local/native_engine/n0/capture"

# S0 smoke (about 90 s): region registered, plan F on 3 photos, equal model inputs, 65 s idle
timeout 600 scripts/bench/frame_engine_hold.sh $BED --rows s0 --sets coco \
  --out artifacts_local/native_engine/n0/s0.jsonl

# H1 parity: both engines, plans F and C, 1 and 32 in flight, the Python engine against itself
timeout 7200 scripts/bench/frame_engine_hold.sh $BED --rows h1 --out artifacts_local/native_engine/n0/h1.jsonl

# H2 rate and ceiling: plans F and C at 1, 8, 32, 64, 128 in flight, both engines, two runs per
# arm alternating; the Z rows at 32 and 64; the native input-only row for the ceiling's d
timeout 14400 scripts/bench/frame_engine_hold.sh $BED --rows h2 --out artifacts_local/native_engine/n0/h2.jsonl

# H3 capacity: the null models, plans F and C, COCO and the 20 MP set, 32, 64, 128 in flight; Z0 rows
timeout 7200 scripts/bench/frame_engine_hold.sh $BED --rows h3 --out artifacts_local/native_engine/n0/h3.jsonl

# H6 faults: 10 kills at 32 in flight, then a model-server restart with the engine up
scripts/bench/frame_engine_faults.sh --mode kill --hold-file $HOLD --gpu-index 2 \
  --gpu-bus-id 00000000:84:00.0 --list artifacts_local/native_engine/n0/lists/coco.txt \
  --plan-file artifacts_local/native_engine/n0/h2.jsonl.d/plans/frame_real.json \
  --out artifacts_local/native_engine/n0/h6.jsonl
```

`--dry-run` on either script prints every command and starts nothing. The runs of H4 (lane counts,
the NUMA arm, the receive windows) and H5 (memory, image size, start time) use the same client
with the engine's own `OP_FRAME_ENGINE_*` settings; they have no script of their own.

A single row by hand:

```bash
native/target/release/op-fe-loadgen engine --url 127.0.0.1:4621 \
  --list artifacts_local/native_engine/n0/lists/coco.txt \
  --plan-file artifacts_local/native_engine/n0/h2.jsonl.d/plans/ingest_real.json \
  --in-flight 32 --warmup-s 5 --window-s 20 --stats-url 127.0.0.1:4620 \
  --gpu-bus-id 00000000:84:00.0 --cpu-stat <the container's cpu.stat> --label try
```

## Reading a row

One JSON line per row (schema `op-fe-loadgen/1`). The rate is the photos answered `ok` inside the
window over the window's length; a photo counts by when it finished. Failures are counted by key:
`grpc:<CODE>`, `status:<photo status>`, `branch:<name>:<reason>` and `protocol:<what>`; a row with
any failure exits 3 and the hold stops.

| Field | What it is |
|---|---|
| `photos_per_s`, `p50_ms`, `p99_ms`, `mean_ms`, `max_ms` | the measured window (percentiles by numpy's linear rule) |
| `failed`, `failures_by_status`, `warmup_failed`, `after_window` | what went wrong, and what finished after the window |
| `cpu` | the measured process's `cpu.stat` over the window: `cpu_ms_per_photo`, `cores_busy`, throttling; memory at the end |
| `gpu` | the card's name, bus id, driver, PCIe link; mean busy share from NVML every 50 ms; memory idle and peak; peak per process, with each process's command and cgroup |
| `model_server` | per model, over the window: mean batch (items per execution), queue time per request, compute time per execution |
| `list`, `plan` | the list's path and SHA-256, the plan's text and SHA-256 |
| `engine`, `z` | the address and channels; for the models alone, the crop counts replayed and the region |
| `host`, `command`, `loadgen`, `started_at`, `ended_at` | CPU model and physical cores, the 1-minute load at start and end, the exact command line, the build |

`<out>.env.json` beside each hold records the commit, the image ids and start times, the SHA-256 of
every `.plan` and the driver. Raw files stay under `artifacts_local/native_engine/n0/` and are
never bulk-deleted.

## Notes on what each row measures

- **Z rows** make the model calls the engine makes for the plan (the detector and the frame
  encoder together, then the crop encoder in chunks of the plan's batch, all chunks in flight)
  with inputs already resident in a registered region. For the ingest plan each photo replays the
  crop count a Python parity run recorded (`gpu_prep_parity.py capture --crop-counts-out`, made by
  the H2 hold), refused if it was recorded for another list.
- **The input-only row** (G1's `d`) is the native engine on plan F with the null models at 32 in
  flight: protocol version 1 has no plan without models, and this is what is left of the work.
- **Stand-ins that bound the row**: when a null model's own alone row (Z0) is under twice the
  engine's rate, regenerate the stand-ins with more instances (`--detector-instances`,
  `--encoder-instances`) and run the row again; it is not judged.
- **Parity**: model outputs that depend on the server's batch composition do not qualify even for
  an engine against itself, so the vector rows pass on a floor, the Python engine against itself
  in the same hold (`gpu_prep_parity.py compare --floor-a ... --floor-b ...`).
