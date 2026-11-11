# GPU pipeline prototypes PR-0, PR-1, PR-2 (research window PW-1)

| Field | Value |
|---|---|
| Status | registered 2026-11-05 before any score (sections 1); tier 0 and tier 1 run 2026-11-05/06 (sections 2-6); no tier 2 was run |
| Plans | [08 section 2.1 and 3](../design/v070_rebuild/08_research_prototype_track.md), [09g section 6](../design/v070_rebuild/09g_research_gpu_pipeline.md) |
| Hardware | one RTX A6000 (GPU 0, 48 GB, no hardware JPEG engine), driver 620.70.06, CUDA 13.x; a shared host: the 1-minute load average is recorded per run and was 7 to 11 in the low-load reruns and 16 to 88 in the first runs |
| Measured at | harness commits `e61d453f4` to `13dd96e59` and the fixes of the commit that adds this section; Python 3.12.3; CuPy 13.6.0, TensorRT 11.1.0.106, DALI 2.4.0, nvImageCodec 0.9.0.20, CV-CUDA 0.19.0, OpenCV 4.13.0.92, Pillow 13.0.0, simplejpeg 1.9.0 (lock file beside the requirements) |
| Code | `scripts/research/gpu_pipeline/`, venv outside the repository (`requirements-gpu-research.txt`, lock beside it) |
| Protocol | tier 0 CPU smoke on 50 items, tier 1 short seeded run with today's approach as the control arm, no tier 2 without the coordinator (08 section 2.1) |

## 1. Registration

Everything in this section was fixed before any score was taken.

### 1.1 Samples, seed, order

| Set | Content | Tier-0 | Tier-1 | Order |
|---|---|---|---|---|
| A | `coco_bench_2000` pin, 640 px class | first 50 | first 200 | pin order |
| B | `commons_hires_500` pin, 12-24 MP | first 50 | first 200 | pin order |
| P | seeded sample of the owner's private photos, one shared local copy, content-hash names | first 50 | first 200 | order of the seeded draw (seed 20261104) |
| OI-1024 | 200 Open Images images whose longest side is 1,024 px, drawn with seed 20261104 | - | 200 (PR-1 only) | draw order |
| EXIF fixtures | about 40 files generated from public images: orientation tags 1-8, progressive, truncated, CMYK, 4:4:4, 4:2:2, 4:2:0, restart markers | all | all (PR-1 only) | file name |

Set P is reported in tracked text only as whether a registered bar was met (09g section 6.8); no set P figure is tracked. Every dataset path is an argument.

### 1.2 PR-0, the harness and the honest GPU number

Feeds 5a-R0 and the programme stop rule: **stop optimizing when SM active is above 85 % with PE at
least 70 % of GPU ms** at the bulk point.

No arms (instrumentation). It runs the shipped algorithm out of Triton: nvJPEG GPU-Huffman lanes
(`models/gpu_image_prep/1/nvjpeg_lanes.py` imported by path), the shipped CuPy kernels, the shipped
detector and PE engines (read-only, from a checkout's model repository, loaded with the venv's
TensorRT) and the host JPEG encode of the crops, on a worker-thread pool like the shipped model.

| Check | Bar |
|---|---|
| Kernel output | the harness's letterbox, whole-frame square and crop squares are byte-identical between two runs (determinism) and pass the tensor gate against a Pillow reference: mean abs diff < 0.5 grey, max <= 8 |
| Stage split | per stage GPU ms per image (CUDA events) and wall ms; three counted repetitions after one warm-up; a stage is reported "stable" when (max - min) / median over repetitions is at most 0.25 |
| Honest GPU number | `nsys` GPU metrics sampling (SM Active, Tensor Active) and NVML utilization side by side when the counters are available; where the driver refuses profiling counters to this user (`ERR_NVGPUCTRPERM`), the SM-active leg is reported as not measurable, with the substitute measured instead: the share of wall time with at least one kernel running (union of CUDA kernel intervals from the `nsys` trace) and per-stage GPU kernel ms attributed by NVTX ranges |
| Verdict | the stop rule evaluated on the measured numbers, with the number of repetitions behind it |

### 1.3 PR-1, the decoder matrix

Feeds 5a-S4: the decoder per size class (RD-14, RD-17). Control arm: D-a, the shipped ctypes nvJPEG
`GPU_HYBRID` lanes (`nvjpeg_lanes.py`, 16 lanes).

| Arm | Setup |
|---|---|
| D-b | nvImageCodec 0.9.0.20 (the newest that DALI 2.4.0 allows: `nvidia-dali-cuda130` requires `nvidia-nvimgcodec-cu13` below 0.10, while 0.10.0.9 is the newest on PyPI; both libraries must share one venv), `HYBRID_CPU_GPU` backend, `fancy_upsampling=1`, batch 1, 8 and 64 (the best batch is the arm's value) |
| D-c | DALI `mixed` decoder, `jpeg_fancy_upsampling=True`, `hybrid_huffman_threshold` 0 and the default (the better is the arm's value) |
| coverage probe, not an arm | nvImageCodec `GPU_ONLY` on 20 images per class: the card has no hardware JPEG engine, expected to decline |
| CPU reference columns, not arms | simplejpeg (libjpeg-turbo) and Pillow with `draft` (the 5c-K4 column) |

That is two arms plus the control (the limit is three). Size classes: A (about 0.3 MP), OI-1024
(about 0.7 MP), B and P (high-resolution photos), and the EXIF fixtures for coverage only.

Common conditions: 16 worker threads (or the library's own 16 CPU threads) for every arm, input
cache dropped per repetition, one untimed warm-up block of 64 images per arm, three counted
repetitions with the arm order alternating each repetition. Throughput is taken per **block of 64
files, each block timed on its own call** (200 files give 3 full blocks, the last 8 are dropped, so
9 paired blocks per arm and class; pairs share the same files). The interval is a 95 % paired
bootstrap over those pairs (5,000 draws, seed 20261104) on the ratio of medians. GPU ms per photo
comes from an `nsys` CUDA kernel trace of one repetition per arm and class. The shipped router
(`prep_geometry.route`) runs in front of every arm, so the declined set is the router's and equal by
construction (margin item 4); the raw behaviour of each arm on the declined fixtures (truncated,
progressive, CMYK) is reported separately as coverage.

**Registered margin.** An arm is a tier-2 candidate for a size class only if all hold:

1. photos/s ratio to control at least 1.15 with the lower end of the 95 % interval above 1.0 and GPU ms per photo not higher, **or** GPU ms per photo at most 0.70 of control (upper interval end below 1.0) at photos/s of at least 0.97 of control;
2. tensor gate against the Pillow decode (mean abs diff < 0.5 grey, max <= 8) on every decoded file;
3. EXIF orientation applied equal on tags 1-8 (checked on the oriented output);
4. the same set of declined files as the control (no silent acceptance).

**Early stop.** An arm stops for a class once its interval upper end on the photos/s ratio is below 1.0 after one full repetition, or it fails a gate.

### 1.4 PR-2, resize and crop operations

Feeds 5b-G2: a library that passes replaces the custom kernels. Control arm: K-a, the shipped CuPy
kernels (`prep_kernels.py`). Inputs: decoded frames (the control decoder), the four operations of
the shipped model: detector letterbox (640, Pillow bilinear with antialias), whole-frame PE square
(shorter edge 336 and centre crop), per-detection PE squares (plain bilinear, no antialias, crop
rectangles taken from the shipped detector's boxes), and the blur sums.

| Arm | Setup |
|---|---|
| K-b | CV-CUDA (0.19.0 or newest): `pillowresize` for the letterbox and the whole-frame square, `resize` (linear) for the crop squares, `laplacian` plus a gray conversion for the blur sums |
| K-c | DALI `fn.resize` with the three parity settings (bilinear with antialias, bilinear without antialias, and Pillow-compatible cropping) |

**Registered margin (5b-G2 rule).** An arm replaces a custom kernel only if the operation is
byte-identical to the control, or passes the tensor gate against the Pillow or OpenCV reference
(mean abs diff < 0.5 grey, max <= 8) and its GPU ms per photo is not higher than the control's (the
ratio's 95 % upper end at most 1.0). Box parity and PE parity (P-box, P-pe) need the engines and are
computed from the detector and PE outputs of the PR-0 harness on the arm's tensors: P-box at least
0.98 and PE cosine median at least 0.999 with p01 at least 0.995. The blur sums are exact integers
in the control; an arm passes only if its Laplacian variance equals the control's to 1e-6 relative.

### 1.5 Decisions the rows feed, and the default

The default in every row is today's behaviour: keep the shipped decoder lanes and the shipped
kernels, keep the programme stop rule as written. A prototype that does not run, stops early or
overruns is reported so and the decision is taken on today's behaviour. An arm that passes is a
"tier-2 candidate" for the coordinator; nothing is adopted here.

### 1.6 Budgets

| Prototype | Tier-1 GPU h | Stop |
|---|---:|---|
| PR-0 | 0.5 | at the budget |
| PR-1 | 1.5 | at the budget |
| PR-2 | 0.5 | at the budget |

GPU hours are wall hours of the `flock`-held commands on GPU 0, summed from the run logs.

### 1.7 Amendments made before any score

1. **PR-2 GPU ms interval.** GPU ms per photo per operation comes from `nsys` traces of two full
   passes (200 images each, NVTX range per arm and operation). "Not higher than the control" means
   the arm's median of the two passes is at most the control's, and a "lower" claim needs the
   difference to exceed the larger relative difference between that arm's own two passes. Wall ms
   per operation (median over 3 repetitions of 200 images, p90 beside it) is reported next to it as
   context: it includes host launch cost and is not the registered comparison.
2. **PR-2 fused op.** `resize_crop_convert_reformat` (CV-CUDA) is tried for the whole-frame square
   only, as a within-K-b variant (it resizes without Pillow antialiasing, so it is expected to fail
   the tensor gate on large downscales); it is reported as its own row and does not add an arm.
3. **PR-2 reference noise floor.** The detector plan is not bit-stable between two calls on the same
   canvas (observed in the tier-1 smoke before scoring), so P-box and P-item are reported for the
   control against itself (the floor) and every arm is judged against `min(0.98, floor)` for P-box.
4. **PR-0 determinism.** "Byte-identical between two runs" is evaluated on images whose detector
   rectangles are equal in both runs (the kernels get the same inputs); images where the detector
   moved a rectangle are counted and reported separately.
5. **PR-0 stage GPU ms.** The per-stage GPU ms come from the NVTX ranges of one traced repetition;
   TensorRT per-layer ranges are switched off in the harness so kernels fall into the stage that
   called them.

## 2. Tier 0 (CPU smoke, 50 items per set)

Run before any GPU command, `CUDA_VISIBLE_DEVICES` empty, sets A, B and P.

| Prototype | Result |
|---|---|
| PR-0 | letterbox with the shipped resampling weights on the CPU against the Pillow reference: 50/50 pass the tensor gate on each of A and B (worst mean abs diff 0.17 / 0.07 grey, worst max 1); set P: bar met; two runs byte-identical |
| PR-1 | router verdicts and CPU decode columns end to end: A 50 passed, B 47 passed (3 declined); set P: ran; simplejpeg and Pillow `draft` columns run |
| PR-2 | the four reference operations run on 50 images per set: letterbox gate 50/50 on each set, squares 3 x 336 x 336, deterministic |

## 3. gpu:PR-0, the harness and the honest GPU number

**What ran.** The shipped algorithm (nvJPEG GPU-Huffman lanes, shipped CuPy kernels, shipped detector
and PE plans through the venv's TensorRT, host crop JPEG encode) on 16 worker threads, no Triton, first
200 images of each set, 40 warm-up images, 3 counted repetitions, page cache dropped per repetition;
then one `nsys` traced repetition per set (NVTX range per stage, TensorRT per-layer ranges off).
Images per second below are the counted images per wall second of each repetition.

**Low-load rerun (primary, host load 7 to 11):**

| set | images/s per rep | spread | declined | load at start | NVML util % | kernel-busy share | PE share of GPU ms | GPU kernel ms/image |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 29.4 / 32.0 / 31.7 | 0.08 | 0 | 7 / 8 / 8 | 91 / 100 / 98 | 0.83 | 0.91 | 41.4 |
| B | 35.1 / 52.0 / 54.9 | 0.38 | 12 | 8 / 9 / 9 | 59 / 76 / 78 | 0.48 | 0.72 | 18.9 |

GPU kernel ms per image by stage (nsys, NVTX ranges):

| set | read | decode | letterbox | detect | embed_frame | embed_crops | embed_pe | blur | crop_copy | crop_jpeg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 0.00 | 0.22 | 0.04 | 3.40 | 0.02 | 0.12 | 37.57 | 0.04 | 0.03 | 0.00 |
| B | 0.00 | 1.11 | 0.34 | 2.42 | 0.22 | 0.04 | 13.60 | 1.12 | 0.07 | 0.00 |

**First run (host load 16 to 88):**

| set | images/s per rep | spread | declined | load at start | NVML util % | kernel-busy share | PE share of GPU ms | GPU kernel ms/image |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 28.4 / 30.4 / 30.1 | 0.07 | 0 | 16 / 17 / 17 | 93 / 100 / 97 | 0.87 | 0.90 | 42.2 |
| B | 37.6 / 48.1 / 51.9 | 0.30 | 13 | 23 / 25 / 27 | 64 / 74 / 80 | 0.28 | 0.74 | 16.1 |

GPU kernel ms per image by stage (nsys, NVTX ranges):

| set | read | decode | letterbox | detect | embed_frame | embed_crops | embed_pe | blur | crop_copy | crop_jpeg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 0.00 | 0.23 | 0.04 | 3.62 | 0.02 | 0.12 | 38.08 | 0.04 | 0.03 | 0.00 |
| B | 0.00 | 1.03 | 0.23 | 1.80 | 0.16 | 0.03 | 11.83 | 0.92 | 0.07 | 0.00 |

Bars (section 1.2):

| Bar | Result |
|---|---|
| kernel output deterministic and inside the tensor gate | yes on every set: every checked image passes the gate against the Pillow/OpenCV reference (letterbox, whole-frame square, crop squares; worst figures on A and B: mean abs diff 0.21 grey, max 3; set P met the gate too); with equal detector rectangles two runs are byte-identical. The detector (not the kernels) moved a crop rectangle between two runs on 1 of 40 images (A) and 1 of 38 (B); set P: kernels met the bar |
| stage split stable (spread of repetitions at most 0.25) | A yes (0.08); B no (0.38 in the rerun, 0.30 in the first run); set P met in the rerun and not met in the first run: the first repetition of B is consistently 22 to 36 % slower than the next two in both runs, cause not found |
| honest GPU number | SM Active is **not measurable** here: `nsys --gpu-metrics-devices` and DCGM need profiling rights this host denies (`ERR_NVGPUCTRPERM`, `dcgmi` is absent). Substitute: share of the traced window with at least one kernel resident (an upper bound on SM active) and NVML utilization |

**Findings.**

1. PE is most of the GPU kernel time but not equally: 91 % on set A (6 detections a photo), 72 % on B (under 2 detections a photo); on P it met the 70 % line. Whole-frame blur sums are the
   second largest stage on those (about 1 GPU ms a photo on B), then decode and detect.
2. The GPU is not saturated on photos. Kernel-busy share (an upper bound on SM active) is 0.48 on B in the rerun (0.28 in the first run)
   against 0.83 on A (0.87 first run); on set P it was below the 0.85 line in both runs. NVML
   utilization reads higher than the kernel-busy share on the photo sets, so the NVML number
   overstates how busy the SMs are. Wall time per photo is spent on the host: decode takes 87 ms of wall per B photo across 16 workers (the GPU part of it is about 1 ms).
3. Peak device memory (NVML, one process with 16 lanes and 4 TensorRT contexts per engine): 10,432 MiB (A) to 16,310 MiB (B).

**Decision it feeds (5a-R0, programme stop rule: SM active above 85 % with PE at least 70 % of GPU ms).**
The PE leg holds on every set (72 to 91 % on A and B; met on P). The SM-active leg cannot be measured, but its upper
bound is below 85 % on B in both runs (0.28 to 0.48), on set P (below the line in both runs) and on A in the rerun (0.83), so the rule
**does not stop optimization work** on photos; on set A it is undetermined in the first run (0.87) and not met
in the rerun. Taking the decision on today's behaviour (no stop) is therefore safe. A direct SM-active
reading needs the host's profiling counters enabled by the owner (NVIDIA driver parameter
`NVreg_RestrictProfilerToAdminUsers=0`, a root action); I did not ask for it.

## 4. gpu:PR-1, the decoder matrix

**What ran.** Control D-a (shipped ctypes nvJPEG `GPU_HYBRID` lanes) against nvImageCodec `HYBRID_CPU_GPU`
at batch 1, 8 and 64 (`fancy_upsampling=1`) and DALI `mixed` at batch 64 with
`hybrid_huffman_threshold` 0 and the default, on the first 200 files of A, OI-1024, B and P (the
router passes 200, 200 and 185 of the first three; set B has 14 progressive files and one without an end
marker). Three counted repetitions, blocks of 64 files, paired bootstrap over 9 blocks. GPU ms per
photo is kernel plus copy time from one `nsys` traced repetition. Host load at the runs: A 19 to 20,
OI 35 to 36 (above the 35 hold line, noisy), B 20 to 22; the reruns of all four ran at 8 to 16.

**First run:**

Set A: 200 files, 0 declined by the router {'gpu': 200}

| arm | images/s (median of blocks) | ratio to control [95 % interval] | rep spread | host CPU s/image | GPU ms/image (kernel + copy) | gate pass/fail |
| --- | --- | --- | --- | --- | --- | --- |
| control | 1829.2 | 1.00 (control) | 0.14 | 0.0031 | 0.13 | 200/0 |
| dali:0:64 | 2800.9 | 1.53 [1.38, 1.73] | 0.11 | 0.0027 | 0.14 | 200/0 |
| dali:default:64 | 2086.2 | 1.14 [0.95, 1.28] | 0.10 | 0.0043 | 0.15 | 200/0 |
| nvimgcodec:hybrid:1 | 316.5 | 0.17 [0.15, 0.20] | 0.14 | 0.0032 | 0.15 | 200/0 |
| nvimgcodec:hybrid:64 | 1861.4 | 1.02 [0.78, 1.21] | 0.36 | 0.0047 | 0.15 | 200/0 |
| nvimgcodec:hybrid:8 | 812.5 | 0.44 [0.37, 0.52] | 0.30 | 0.0046 | 0.15 | 200/0 |

CPU columns: pillow_draft640 1406.9 images/s; simplejpeg 2683.0 images/s

Set OI: 200 files, 0 declined by the router {'gpu': 200}

| arm | images/s (median of blocks) | ratio to control [95 % interval] | rep spread | host CPU s/image | GPU ms/image (kernel + copy) | gate pass/fail |
| --- | --- | --- | --- | --- | --- | --- |
| control | 1569.7 | 1.00 (control) | 0.07 | 0.0046 | 0.16 | 200/0 |
| dali:0:64 | 2061.3 | 1.31 [1.01, 1.88] | 0.09 | 0.0050 | 0.16 | 200/0 |
| dali:default:64 | 1443.7 | 0.92 [0.74, 1.43] | 0.20 | 0.0086 | 0.27 | 200/0 |
| nvimgcodec:hybrid:1 | 202.2 | 0.13 [0.10, 0.20] | 0.05 | 0.0053 | 0.27 | 200/0 |
| nvimgcodec:hybrid:64 | 1013.2 | 0.65 [0.46, 1.00] | 0.67 | 0.0099 | 0.28 | 200/0 |
| nvimgcodec:hybrid:8 | 329.6 | 0.21 [0.17, 0.35] | 0.10 | 0.0098 | 0.28 | 200/0 |

CPU columns: pillow_draft640 318.8 images/s; simplejpeg 1077.6 images/s

Set B: 200 files, 15 declined by the router {'gpu': 185, 'progressive': 14, 'truncated': 1}

| arm | images/s (median of blocks) | ratio to control [95 % interval] | rep spread | host CPU s/image | GPU ms/image (kernel + copy) | gate pass/fail |
| --- | --- | --- | --- | --- | --- | --- |
| control | 188.4 | 1.00 (control) | 0.23 | 0.0632 | 2.87 | 185/0 |
| dali:0:64 | 138.2 | 0.60 [0.33, 0.92] | 0.70 | 0.0750 | 2.31 | 185/0 |
| dali:default:64 | 156.0 | 0.81 [0.65, 0.91] | 0.20 | 0.0688 | 2.32 | 185/0 |
| nvimgcodec:hybrid:1 | 19.5 | 0.10 [0.09, 0.13] | 0.13 | 0.0503 | 2.56 | 185/0 |
| nvimgcodec:hybrid:64 | 209.4 | 1.10 [1.02, 1.15] | 0.23 | 0.0623 | 3.08 | 185/0 |
| nvimgcodec:hybrid:8 | 65.5 | 0.35 [0.32, 0.41] | 0.16 | 0.0635 | 3.16 | 185/0 |

CPU columns: pillow_draft640 50.4 images/s; simplejpeg 53.7 images/s

Set P: summarized in section 4.1 (bar status only).

**Set P (4.1, bar status only, figures not tracked):** the margin is a near miss for nvImageCodec batch 64
(throughput clause met in both runs, GPU-ms clause not met); not met for DALI and for the other nvImageCodec batches.

**Rerun at lower load (throughput only; GPU ms is not repeated):**

Set A: 200 files, 0 declined by the router {'gpu': 200}

| arm | images/s (median of blocks) | ratio to control [95 % interval] | rep spread | host CPU s/image | GPU ms/image (kernel + copy) | gate pass/fail |
| --- | --- | --- | --- | --- | --- | --- |
| control | 1597.0 | 1.00 (control) | 0.34 | 0.0039 | n/a | 200/0 |
| dali:0:64 | 3264.7 | 2.04 [1.51, 2.69] | 0.25 | 0.0030 | n/a | 200/0 |
| dali:default:64 | 2479.4 | 1.55 [1.05, 2.16] | 0.30 | 0.0046 | n/a | 200/0 |
| nvimgcodec:hybrid:1 | 309.4 | 0.19 [0.17, 0.25] | 0.11 | 0.0032 | n/a | 200/0 |
| nvimgcodec:hybrid:64 | 1786.4 | 1.12 [0.83, 1.47] | 0.37 | 0.0052 | n/a | 200/0 |
| nvimgcodec:hybrid:8 | 637.8 | 0.40 [0.34, 0.59] | 0.14 | 0.0057 | n/a | 200/0 |

CPU columns: pillow_draft640 1389.3 images/s; simplejpeg 2700.7 images/s

Set OI: 200 files, 0 declined by the router {'gpu': 200}

| arm | images/s (median of blocks) | ratio to control [95 % interval] | rep spread | host CPU s/image | GPU ms/image (kernel + copy) | gate pass/fail |
| --- | --- | --- | --- | --- | --- | --- |
| control | 1472.7 | 1.00 (control) | 0.65 | 0.0045 | n/a | 200/0 |
| dali:0:64 | 2085.3 | 1.42 [0.87, 1.75] | 0.11 | 0.0048 | n/a | 200/0 |
| dali:default:64 | 1474.5 | 1.00 [0.66, 1.29] | 0.05 | 0.0080 | n/a | 200/0 |
| nvimgcodec:hybrid:1 | 204.5 | 0.14 [0.09, 0.18] | 0.11 | 0.0052 | n/a | 200/0 |
| nvimgcodec:hybrid:64 | 980.4 | 0.67 [0.46, 0.87] | 0.56 | 0.0109 | n/a | 200/0 |
| nvimgcodec:hybrid:8 | 346.7 | 0.24 [0.15, 0.30] | 0.08 | 0.0103 | n/a | 200/0 |

CPU columns: pillow_draft640 310.4 images/s; simplejpeg 1004.3 images/s

Set B: 200 files, 15 declined by the router {'gpu': 185, 'progressive': 14, 'truncated': 1}

| arm | images/s (median of blocks) | ratio to control [95 % interval] | rep spread | host CPU s/image | GPU ms/image (kernel + copy) | gate pass/fail |
| --- | --- | --- | --- | --- | --- | --- |
| control | 201.2 | 1.00 (control) | 0.19 | 0.0606 | n/a | 185/0 |
| dali:0:64 | 130.8 | 0.63 [0.34, 0.91] | 0.56 | 0.0706 | n/a | 185/0 |
| dali:default:64 | 161.3 | 0.75 [0.60, 0.97] | 0.47 | 0.0662 | n/a | 185/0 |
| nvimgcodec:hybrid:1 | 20.9 | 0.11 [0.10, 0.12] | 0.07 | 0.0497 | n/a | 185/0 |
| nvimgcodec:hybrid:64 | 205.8 | 1.02 [0.98, 1.12] | 0.25 | 0.0638 | n/a | 185/0 |
| nvimgcodec:hybrid:8 | 66.0 | 0.34 [0.29, 0.39] | 0.11 | 0.0638 | n/a | 185/0 |

CPU columns: pillow_draft640 73.4 images/s; simplejpeg 59.8 images/s

Set P: summarized in section 4.1 (bar status only).

Coverage on 51 fixtures (orientation tags 1-8, progressive, truncated, CMYK, grey, 4:4:4, 4:2:2, 4:2:0,
restart markers; three source images) and on the 15 files of set B the router declines:

| Arm | Orientation 1-8 equal to Pillow | Progressive | CMYK | Truncated (cut at 60 %) | 15 router-declined files of B |
|---|---|---|---|---|---|
| D-a shipped lanes | yes, all 24 | decodes within the gate | raises (nvJPEG), so declined | decodes without an error | 14 progressive and 1 missing-end-marker file all pass the gate |
| D-b nvImageCodec hybrid | yes | yes | yes, within the gate | decodes without an error | all pass |
| D-c DALI mixed | yes | yes | yes, within the gate | decodes without an error | all pass |
| nvImageCodec `GPU_ONLY` | returns nothing for all 51 files: the card has no hardware JPEG engine, as expected | | | | |

**Registered margin, per size class** (ratio of images/s with the 95 % interval, then the GPU-ms clause):

| Class | Best arm | Result | Margin met |
|---|---|---|---|
| A, 0.3 MP | DALI hht 0, batch 64 | 1.53 [1.38, 1.73], rerun 2.04 [1.51, 2.69]; GPU ms/photo 0.135 vs 0.133 (+1.2 %, one trace, resolution 0.002 ms) | yes on throughput in both runs; GPU-ms clause met within resolution |
| OI-1024, 0.8 MP | DALI hht 0, batch 64 | 1.31 [1.01, 1.88], rerun 1.42 [0.87, 1.75]; GPU ms 0.162 vs 0.163 | no: the rerun interval includes 1.0 and control spread was 0.65 |
| B, 12-24 MP | nvImageCodec batch 64 | 1.10 [1.02, 1.15], rerun 1.02 [0.98, 1.12]; GPU ms 3.08 vs 2.87 (+7 %) | no |
| P, high-resolution photos | nvImageCodec batch 64 | figures not tracked (09g 6.8) | near miss: throughput clause met in both runs, GPU-ms clause not met (higher) |

DALI is below control on B (0.60 to 0.81) and does not meet the margin on P (intervals include 1.0).
nvImageCodec at batch 1 and 8 is far below control (0.10 to 0.47) because one caller thread feeds it.
CPU columns (16 threads): simplejpeg 2,683 / 2,701 images/s on A (the control GPU lanes 1,829 / 1,597),
1,078 / 1,004 on OI (control 1,570 / 1,473), 53.7 on B; Pillow `draft(640)` 1,407 on A, 319 on OI, 50 on B.

**Findings.**

1. Large photos (B, P): keep the shipped lanes. No arm passes the registered margin; the nearest is nvImageCodec at
   batch 64 on P (a near miss: faster, but the GPU-ms clause fails) which is not a win on B and is
   only reachable through a batch API with one calling thread.
2. Sub-megapixel images (A, and OI less clearly): DALI `mixed` with hht 0 at batch 64 is about 1.5 to 2 times
   the shipped lanes with the same GPU ms, but the CPU libjpeg-turbo route already decodes A at the same speed
   (2.7 k images/s) with no GPU time, and a photo's PE time (12 to 38 GPU ms) dwarfs the 0.1 to 0.2 ms
   decode. The small-image decode is a host-throughput question for the batched-decode bake-off (5b-W1), not a GPU-cost question.
3. The router declines 14 of the 200 B photos as progressive (7 %) and sends them to the CPU path, although the shipped
   lanes decode all 14 within the tensor gate (and the three progressive fixtures). That is a possible routing
   change for 5a-S4; the registered rule keeps the decline (D3, no silent acceptance), which still guards the
   truncated case: every GPU decoder here accepts a truncated file without an error.
4. Orientation 1-8 is correct in every arm; CMYK needs the router with the shipped lanes but is handled by D-b and D-c.

**Decision it feeds (5a-S4, the decoder per size class).** Today's decoder (shipped lanes) stays for the
12-24 MP classes (B, P). For sub-megapixel images DALI mixed hht 0 is the only tier-2 candidate (A only; OI
did not hold in the rerun) and the CPU route is the real competitor. Nothing is adopted here.

## 5. gpu:PR-2, resize and crop operations

**What ran.** Arms K-a (shipped CuPy kernels, control), K-b (CV-CUDA 0.19.0: `pillowresize`, `resize`
linear, `cvtcolor`, `convertto`, `laplacian`), the within-K-b fused variant for the whole-frame square
(`resize_crop_convert_reformat`) and K-c (DALI `fn.resize`, antialias on for the letterbox and the whole
square, off for the crops, one-image pipelines). Frames decoded by the shipped lanes, crop rectangles
from the shipped detector plan (first 200 images of A and B, a shorter sample of P), per operation wall time over
3 repetitions (fewer on P), tensor gates against the Pillow / OpenCV references, box and PE
parity against the control arm, GPU ms from `nsys` NVTX ranges (A: two passes, B and P: one pass, because the
passes cost 6 minutes each and the row budget was spent). Host load 13 to 17 (A) and 13 (B) at the end of the runs.

Set A: errors {'K-c': 'blur_sums:K-c has no Laplacian operator; the blur sums stay on K-a;'}

| arm | operation | wall ms median / p90 | GPU kernel ms/image (two passes) | gate pass/n | worst mean / max abs |
| --- | --- | --- | --- | --- | --- |
| K-a | blur_sums | 0.25 / 0.30 | 0.018 / 0.018 | - | - |
| K-a | crop_squares | 1.31 / 5.81 | 0.048 / 0.047 | 1273/1273 | 0.20 / 4 |
| K-a | letterbox | 0.35 / 0.47 | 0.026 / 0.025 | 200/200 | 0.19 / 3 |
| K-a | whole_square | 0.23 / 0.74 | 0.011 / 0.011 | 200/200 | 0.22 / 3 |
| K-b | blur_sums | 3.25 / 9.97 | 0.162 / 0.160 | - | - |
| K-b | crop_squares | 1.21 / 4.25 | 0.057 / 0.056 | 1273/1273 | 0.20 / 4 |
| K-b | letterbox | 0.41 / 0.49 | 0.030 / 0.029 | 200/200 | 0.06 / 3 |
| K-b | whole_square | 0.26 / 0.31 | 0.018 / 0.017 | 200/200 | 0.11 / 3 |
| K-b-fused | blur_sums | 3.31 / 9.93 | 0.158 / 0.156 | - | - |
| K-b-fused | crop_squares | 1.22 / 4.13 | 0.055 / 0.055 | 1273/1273 | 0.20 / 4 |
| K-b-fused | letterbox | 0.36 / 0.47 | 0.029 / 0.029 | 200/200 | 0.06 / 3 |
| K-b-fused | whole_square | 0.15 / 0.18 | 0.003 / 0.003 | 9/200 | 6.87 / 104 |
| K-c | crop_squares | 4.02 / 13.99 | 0.022 / 0.021 | 115/1273 | 3.22 / 112 |
| K-c | letterbox | 1.26 / 1.46 | 0.015 / 0.015 | 200/200 | 0.19 / 3 |
| K-c | whole_square | 0.96 / 1.17 | 0.004 / 0.004 | 196/200 | 0.22 / 12 |

| arm | P-box | P-item | PE cosine median / p01 | blur max relative diff |
| --- | --- | --- | --- | --- |
| K-a | 0.9984 | 1.000 | 1.00000 / 1.0000 | 0.0 |
| K-b | 0.9921 | 0.985 | 0.99997 / 0.9986 | 0.0 |
| K-b-fused | 0.9898 | 0.980 | 0.99997 / 0.9819 | 0.0 |
| K-c | 0.9969 | 0.995 | 0.99822 / 0.9412 | None |

Set B: errors {'K-c': 'blur_sums:K-c has no Laplacian operator; the blur sums stay on K-a;'}

| arm | operation | wall ms median / p90 | GPU kernel ms/image (two passes) | gate pass/n | worst mean / max abs |
| --- | --- | --- | --- | --- | --- |
| K-a | blur_sums | 1.17 / 1.49 | 0.878 | - | - |
| K-a | crop_squares | 0.43 / 2.41 | 0.027 | 326/326 | 0.16 / 3 |
| K-a | letterbox | 0.66 / 0.77 | 0.212 | 200/200 | 0.08 / 2 |
| K-a | whole_square | 0.48 / 1.78 | 0.127 | 200/200 | 0.09 / 2 |
| K-b | blur_sums | 2.90 / 6.03 | 1.843 | - | - |
| K-b | crop_squares | 0.38 / 1.84 | 0.077 | 326/326 | 0.16 / 3 |
| K-b | letterbox | 0.74 / 0.94 | 0.197 | 200/200 | 0.04 / 2 |
| K-b | whole_square | 0.54 / 0.70 | 0.199 | 200/200 | 0.10 / 2 |
| K-b-fused | blur_sums | 2.77 / 6.10 | 1.843 | - | - |
| K-b-fused | crop_squares | 0.37 / 1.82 | 0.077 | 326/326 | 0.16 / 3 |
| K-b-fused | letterbox | 0.65 / 0.96 | 0.197 | 200/200 | 0.04 / 2 |
| K-b-fused | whole_square | 0.22 / 0.33 | 0.013 | 0/200 | 32.11 / 217 |
| K-c | crop_squares | 1.14 / 5.39 | 0.006 | 197/326 | 0.33 / 88 |
| K-c | letterbox | 1.85 / 2.30 | 0.014 | 91/200 | 0.10 / 27 |
| K-c | whole_square | 1.40 / 1.85 | 0.004 | 120/200 | 0.12 / 28 |

| arm | P-box | P-item | PE cosine median / p01 | blur max relative diff |
| --- | --- | --- | --- | --- |
| K-a | 0.9908 | 1.000 | 1.00000 / 1.0000 | 0.0 |
| K-b | 0.6288 | 0.975 | 0.99996 / 0.9991 | 0.0 |
| K-b-fused | 0.6258 | 0.975 | 0.99996 / 0.9123 | 0.0 |
| K-c | 0.7086 | 0.995 | 0.99996 / 0.9929 | None |

Set P: summarized in section 5.1 (bar status only).

**Set P (5.1, bar status only, figures not tracked):** K-b met the tensor gate on every image, did not meet the
GPU-ms clause and did not meet the P-box bar (the control against itself was already at or below the bar); the fused
variant did not meet the tensor gate; K-c met the gate only in part. K-c GPU ms is not valid.

**Caveats on the numbers above.**

- K-c GPU ms is **not valid**: DALI launches its kernels from its own executor threads, so they fall outside the calling thread's NVTX range. Only K-c's wall time and gates count.
- P-box compares detector boxes at the registered 1 px tolerance in frame pixels. At a 6 to 8 times downscale one pixel of the 640 canvas is 6 to 8 frame pixels and the detector moves boxes by that much on tiny input differences: the control compared with itself is 0.9984 (A), 0.9908 (B) and, on set P, also at or below the 0.98 bar, because the plan is not bit-stable between two calls. On B (K-b 0.63) and P every non-identical arm misses the bar for that reason as well as because it is a different resampler; PE vectors stay within the bar for K-b (median 0.99996 or better on A and B, p01 0.9986 to 0.9991).
- The blur sums of K-b equal K-a's Laplacian variances to the last printed digit (relative difference 0), but cost 9 times (A) and 2.1 times (B) the GPU ms, and more on P, and 3 to 13 times the wall ms.

**Findings.**

1. CV-CUDA reproduces the Pillow letterbox, the Pillow whole-frame square and the OpenCV crop squares within the tensor gate on
   every image of every set (K-b letterbox is a little closer to Pillow than K-a: 0.02 to 0.06 vs 0.06 to 0.19 mean abs diff).
2. K-b is not cheaper in GPU time: letterbox -7 % on B (one pass, inside the noise), +15 % on A; whole-frame square +57 % to +64 %; crop squares +19 % (A), +185 % (B); blur sums 2 to 9 times; set P: GPU-ms clause not met.
   Summed over the three resize operations K-b costs 24 to 29 % more GPU ms than K-a on A and B; with blur 1.9 to 2.6 times. Together the shipped kernels
   are 0.25 % (A) and 6.6 % (B) of a photo's GPU kernel ms (PR-0 rerun), blur sums being most of it on photos.
3. The fused whole-frame square is 4 to 11 times cheaper in GPU ms than K-a, but it resizes without antialiasing and fails the tensor gate (0 of 200 on B, 9 of 200 on A, and on set P; max abs diff 104 to 217 on A and B).
4. DALI `fn.resize` passes the letterbox gate on A and fails it on B (91 of 200) and on set P; the whole-frame square passes 196/200 (A) and 120/200 (B); the non-antialiased crops pass 115 of 1,273 (A) and 197 of 326 (B). Its letterbox and square take 2 to 4 times the wall time of K-a (each call is a pipeline run; the crops 0.8 to 3 times), and it has no Laplacian operator.

**Decision it feeds (5b-G2, a library that passes replaces the custom kernels).** None passes: K-b meets the
tensor gate but its GPU ms is higher, K-b's boxes miss the P-box bar on photos, the fused variant and K-c fail the tensor gate.
The shipped CuPy kernels stay. The coordinator may still weigh that the replaceable part (letterbox, squares, crops) is 0.25 to a few per cent of a photo's GPU ms,
so a CV-CUDA adoption for maintainability would not move throughput, but it fails the registered rule, and the blur sums have no library equivalent that keeps exact integers.

## 6. Summary of verdicts

| Prototype | Feeds | Verdict |
|---|---|---|
| PR-0 | 5a-R0 programme stop rule | the rule is not met on photos (SM active upper bound below 0.85 on B and P); PE leg met on every set; SM active itself unmeasurable (counters denied); decision: no stop, continue |
| PR-1 | 5a-S4 decoder per size class | keep the shipped lanes for 12-24 MP; DALI mixed hht 0 batch 64 is a tier-2 candidate for sub-megapixel images on set A only, with the CPU route as the comparison; nvImageCodec batch 64 on set P is a near miss (GPU-ms clause) |
| PR-2 | 5b-G2 CV-CUDA vs custom kernels | the custom kernels stay; CV-CUDA passes correctness but costs more GPU ms and misses P-box on photos |

## 7. GPU hours

Wall seconds of GPU 0 commands held under the lock (including smoke runs and runs stopped for a harness
bug; the stopped runs are in the totals): PR-0 about 0.15 h of 0.5; PR-1 about 0.51 h of 1.5; PR-2 about 0.53 h of 0.5
(6 % over: one run was stopped and repeated after fixing a PE batch-size bug in the harness; the second profile pass of B was dropped and P ran on a shorter sample to stay near the budget). Total about 1.2 GPU h.
Per-command seconds are kept in local ledgers next to the raw outputs (not committed); small result files are in
`docs/benchmarks/gpu_pipeline_pr0_pr2/` with `SHA256SUMS` (public sets only; set P files stay local).

## 8. Open issues

- SM active is not measured (profiling counters denied). A direct reading would turn the PR-0 bound into a number.
- The host stayed busy (load above 16 for most first runs). The reruns (load 7 to 16) agree with the first runs on every verdict above except the OI-1024 interval of DALI (kept as "no").
- PR-1 GPU ms comes from one traced pass without an interval; the "GPU ms not higher" clause is therefore judged on a point estimate (A +1.2 %, B +7 %; set P: clause not met).
- Detector plan run-to-run variation (box moves of a pixel or more between two calls on the same canvas) is a floor for every box-parity bar.
- B repetition 1 is consistently slower than repetitions 2 and 3 in PR-0; not explained.
- The progressive-file routing finding (section 4, finding 3) used 14 files of set B and three fixtures; it is not a registered result.
- No tier 2 was run (not allowed without the coordinator); none is proposed except the two named in section 6.

## 9. Reproduce

```bash
python3.12 -m venv /path/to/op-gpu-research
/path/to/op-gpu-research/bin/pip install -r scripts/research/gpu_pipeline/requirements-gpu-research.txt
# data: set A/B per scripts/datasets (coco_bench_2000, commons_hires_500 pins), OI sample:
#   scripts/datasets/fetch_openimages_bench.py --n 200 --seed 20261104
# fixtures: scripts/research/gpu_pipeline/exif_fixtures.py; set P: set_p_sample.py (private source, not in this repository)
export PY=... LOCK=... GPU=0 SET_A=... SET_B=... SET_P=... SET_P_MANIFEST=... OI_ROOT=... OI_PIN=... \
       FIXTURES=... DET_PLAN=<yolov11_small_trt_end2end model.plan> PE_PLAN=<pe_image_encoder model.plan> OUT=...
bash scripts/research/gpu_pipeline/run_tier1.sh pr0   # then pr1, pr2
scripts/research/gpu_pipeline/summarize.py pr0|pr1|pr2 "$OUT/<pr>"   # the tables above
```

Plans: the two TensorRT plans are the built engines of a local stack (read-only); they print a TensorRT
warning that they were built on another device model. Tests: `pytest tests/research_scripts/test_gpu_pipeline_harness.py`.
