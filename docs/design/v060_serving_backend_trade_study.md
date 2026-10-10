# v0.6.0 serving backend trade study: maximum GPU throughput for bulk ingest

Status: study and plan (nothing implemented). Milestone v0.6.0. Umbrella: #40 (pipeline), #67
(request-driven plans). Sequencing home: [v060_wave_plan.md](v060_wave_plan.md) section 13. New
work packages WP-S1 to WP-S7 (issues in section 11). Written 2026-10-10 against `origin/main`
`52981772` plus the FP16 measurement of PR #225.

Question answered: which serving and pipeline architecture gets OpenProcessor's **full ingest
pipeline** (decode, detect, crop, embed, write, cluster) to the highest images/s on this server's
NVIDIA GPUs, on the way to a 1M-image batch (owner decision O8), with industry-standard Triton
practice, and whether the owner's own Rust serving project, ortloom (`ortloom-serve`), should take
over model serving or a part of it.

Conventions. **Measured** numbers come from this repository's runs (IDs B1-B12 from the wave plan
section 1, F1-F8 below) or from a cited source; **estimate** marks anything derived. Sources are
`[Sn]` (section 14, URL and access date). Experiments are `X-n` and map one-to-one onto work
packages `WP-Sn`. A fresh agent can run any experiment from this file plus the wave plan.

## 0. Summary

- **Where the work is.** Per ingested image at policy `all`, 99.2 % of the GPU arithmetic is the PE
  encoder: 7.39 embeddings x 384 GFLOP (counted from the ONNX graph), against 22 GFLOP for the
  detector. Throughput is PE tensor compute, the number of embeddings per image, and the host that
  feeds it. The serving layer does not change the PE kernels.
- **Where it stands.** FP16 PE: 17.4 img/s, 5.5 ms per embedding = about 70 TFLOP/s, 45 % of the
  A6000's 154.8 TFLOP/s dense FP16 peak, close to the best published TensorRT ViT-L point on the same
  GPU (about 50 %, [S18]). The GPU is 74 % busy, so on one A6000 there is about 35 % left (23.5 img/s
  ceiling), and 10-25 % more from engine tuning (estimates).
- **The big levers, in order.** (1) the second A6000, about 2x (every model is pinned to GPU 0
  today); (2) filling the GPU (batching and concurrency, host CPU fixes already planned); (3) INT8
  PE at about 1.33x if its accuracy gates pass (measured for CLIP ViT-L on an A6000, [S18]); (4)
  engine tuning; (5) the 3080 Ti, about +20 %. Transport (uint8, shared memory) and GPU decode save
  host CPU and latency; they do not raise images/s at policy `all` on COCO-sized images.
- **Ceilings (estimate, policy `all`).** One A6000 23.5 img/s; two 47; two plus the 3080 Ti 55-58;
  with INT8 or a tuned engine 61-76. 1M images: 16 h today, 4-6 h at the target.
- **Next binders after the GPUs.** Host CPU (about 55 img/s needs 12-15 of 24 cores at today's
  0.22 CPU-s per image) and the single OpenSearch node with CPU HNSW builds (unmeasured; X-5).
- **Recommendation.** Keep Triton (BSD-3-Clause, matches every model and the installer) and harden
  it (B), then add a bulk ingest runner that reuses the ingest service code, shards across one
  Triton per GPU and resumes (C). Do not switch to ortloom-serve now: it serves YOLO only, lacks
  multi-GPU, plan loading and model control, and cannot beat Triton on a compute-bound ViT in the
  same TensorRT kernels; a head-to-head (X-6, #231) decides whether to extend it, most likely as a
  GPU decode-and-crop engine for 20 MP photos, not as the model server. Reject a separate offline
  pipeline (E) unless C misses its gate by more than 15 %.
- **New work packages:** WP-S1 to WP-S7, issues #226 to #232; wave placement in the wave plan
  section 13.

## 1. Inputs: what is measured

### 1.1 Pipeline facts (this repository)

| ID | Fact | Number | Where |
|---|---|---|---|
| F1 | Ingest, policy `all`, FP16 PE, one A6000 | 17.4 (16.4-18.0) img/s upload, 18.8 (17.4-19.5) batch | PR #225, `docs/PERFORMANCE.md` "v0.6.0 Wave 1" |
| F2 | PE FP16 cost | 5.5 ms per embedding (182 emb/s) in the pipeline, mean batch 13.5, 7.39 embeddings per image (6.39 crops + 1 whole frame) | PR #225 |
| F3 | GPU busy after FP16 | 74 % (`nvidia-smi`), own SM 71 % (`pmon`): the GPU is no longer the only limit | PR #225 |
| F4 | API CPU after FP16 | 0.188 CPU-s/image upload, 0.177 batch; Triton 0.035 | PR #225 |
| F5 | Wire to Triton | 14.9 MB FP32 per image (PE 10.0 MB, detector 4.9 MB) inline in gRPC protobuf, no shared memory | B7, `src/clients/triton_pool.py`, `src/clients/pe_encoder.py:352` |
| F6 | Client shape | `tritonclient.grpc.aio`, 32 uvicorn workers, 2.07 PE requests per image (crops of one image, then the whole frame), harness 4 threads x 32 images | `docker-compose.yml:247`, `src/services/curation/ingest_index.py:118`, B1 |
| F7 | Model placement | every GPU model pinned to `gpus: [0]`, one instance (detector 2), PE `max_queue_delay` 15 ms, `preferred_batch_size [4, 8, 16, 32]`, engine profile opt 8 max 32; CUDA graphs off (MobileCLIP explicitly) | `models/*/config.pbtxt`, `scripts/lib/model_setup.sh` `_ms_pe_trtexec` |
| F8 | Cluster training (CPU IVF) after ingest | 78-159 s per 12.8k items (30-55 % of ingest wall) | B8, PR #225 |

Earlier findings that still hold (`docs/research/ingest_concurrency_tuning.md`): with Triton dynamic
batching, many small in-flight requests (batch 1-4, concurrency about 128) beat few large ones
(batch 32-64) by up to 3.4x on a 12 GB card, because large client batches serialize the bulk writes;
the per-model queue/compute ratio in Triton statistics names the binding model.

### 1.2 Host facts (read on 2026-10-10, no stack running)

| Item | Value | Consequence |
|---|---|---|
| CPU | 2 x Xeon E5-2680 v3 (Haswell), 12 cores / 24 threads per socket, 2 NUMA nodes | about 24 physical cores for API, Triton, OpenSearch, clients together |
| PCIe | every GPU link reports `pcie.link.gen.max = 3`, x16 (the Haswell root complex is PCIe 3.0) | about 15.8 GB/s per direction on paper per GPU, not the 31.5 GB/s of the A6000's PCIe 4.0 interface [S1] |
| Topology | GPU0 (A6000) and GPU1 (3080 Ti) on NUMA node 0 (CPUs 0-11, 24-35, `PHB`); GPU2 (A6000) on NUMA node 1 (CPUs 12-23, 36-47); GPU0-GPU2 is `SYS` (crosses QPI) | one Triton process spanning GPU0 and GPU2 makes every GPU2 copy cross the socket unless pinned; per-GPU processes can be NUMA-pinned |
| Persistence mode | enabled on all three | nothing to gain from turning it on |
| Power limit | 300 W per A6000, 350 W on the 3080 Ti | sustained tensor clocks sit below boost; record clocks and power in every run |

### 1.3 Model compute, counted from the ONNX graphs

Counted offline (2026-10-10) from the exported ONNX in the model build directory: multiply-accumulates
of every `Conv`, `MatMul` and `Gemm` at batch 1 with shapes taken from an ONNX Runtime CPU run (the
counter is a throwaway script; the method is reproducible with `onnx` plus `onnxruntime` from the
repo venv). FLOPs = 2 x MACs. Elementwise, softmax and normalization are excluded (small next to
the GEMMs for these models).

| Model (input) | Params | GMAC/inference | GFLOP/inference | Note |
|---|---:|---:|---:|---|
| PE-Core-L14 image encoder (336 x 336) | 317 M | 192.2 | 384.4 | 191.8 GMAC in MatMul (577 tokens, 24 blocks, width 1024); analytic ViT-L/14 count gives 191 GMAC |
| YOLO11s (640 x 640) | 9.4 M | 10.8 | 21.6 | matches the 21.5 GFLOPs that Ultralytics publishes [S20] |
| SCRFD-10G (640 x 640) | 4.2 M | 13.3 | 26.7 | "10G" is named at 640 x 480 |
| ArcFace R50 (112 x 112) | 43.6 M | 6.3 | 12.6 | |
| MobileCLIP2-S2 image (256 x 256) | 35.7 M | 7.8 | 15.7 | |

Per ingested image at policy `all` (7.39 PE + 1 detector): **2,862 GFLOP, 99.2 % of it PE.** The
detector is noise; every throughput question on this workload is a question about the PE encoder,
the number of embeddings per image, and the host that feeds it.

## 2. Ceiling analysis (roofline style)

### 2.1 Peak numbers used

| GPU | FP16 tensor, FP32 accumulate, dense | INT8 tensor, dense | Memory BW | Source |
|---|---:|---:|---:|---|
| RTX A6000 (GA102, SM86, 84 SMs, 300 W) | **154.8 TFLOP/s** (same rate with FP16 accumulate) | 309.7 TOP/s | 768 GB/s | [S1], [S2] |
| RTX 3080 Ti (GA102, SM86, 80 SMs, 350 W) | **68.2 TFLOP/s** (half rate; 136.4 with FP16 accumulate) | 272.8 TOP/s | 912 GB/s | [S3] |

All are vendor peak figures at the boost clock (claims, not measurements). The A6000 datasheet's
single "309.7 TFLOPS" tensor figure is the 2:4 sparse rate [S1]. GeForce GA102 runs FP16 with FP32
accumulate at half rate; professional GA102 runs it at full rate [S2], [S3]. Which accumulate mode
TensorRT picks for PE on the 3080 Ti is not verified: plan on 68 TFLOP/s and measure. FP8 and FP4 need
compute capability 8.9 or later; SM86 has FP16, BF16, TF32 and INT8 only [S4].

### 2.2 Achieved fraction today

F2 gives 182 PE embeddings/s x 384.4 GFLOP = **70 TFLOP/s achieved, about 45 % of the A6000's
dense FP16 peak** (measured numerator, spec denominator). That is already a good fraction for an
inference engine with batch 13.5 on average. PE's arithmetic intensity at batch 8 or more is in the
hundreds of FLOP/byte (weights 0.63 GB FP16 are read once per batch; activations of 577 x 1024 per
layer stay in L2 and on-chip), far above the A6000 ridge point (about 200 FLOP/byte at 155 TFLOP/s
and 768 GB/s), so PE is **compute bound**, as the flat perf_analyzer curve over batch 8-32 already
showed (B2). Memory bandwidth is not the lever; tensor-core efficiency and precision are.

Published reference points for "what fraction is achievable":

| Reference | Number | Implied fraction of FP16 peak | Kind |
|---|---|---|---|
| CLIP ViT-L/14 at 224, TensorRT 10.9 FP16, **RTX A6000**, batch 8, trtexec (NVIDIA maintainer) [S18] | 16.86 ms per batch of 8 = 474 img/s x 162 GFLOP = 77 TFLOP/s | about 50 % | measured |
| Same, INT8 explicit Q/DQ (ModelOpt) [S18] | 12.57-12.68 ms per batch of 8 | **1.33x** over FP16 | measured |
| timm PyTorch AMP eager, ViT-L/14-CLIP-336 vs PE-Core-L14-336, same GPU (RTX Pro 6000 Max-Q) [S22] | 487.6 vs 315.0 img/s at equal FLOPs | PE runs at **0.65x** of CLIP-L in eager mode (rotary embedding and attention pooling overhead, inference) | measured |
| clip-retrieval, ViT-L/14 at 224, one A100 [S23] | 312 samples/s per GPU, 2,500/s on 8 GPUs (linear) | GPU bound | measured (end to end, including loading) |

Reading: OpenProcessor's PE engine at 45 % of peak (trtexec-free, inside Triton, with transfers) is
close to the best published TensorRT ViT-L point on the same GPU (50 %, trtexec, no transfers). The
headroom from engine tuning alone is likely **10-25 %** (estimate), not 2x. PE-specific operators may
cost more than CLIP's; X-2 checks whether TensorRT fused them.

### 2.3 Ceilings per stage (images/s at policy `all`, 7.39 embeddings per image)

All rows are **estimates** derived from the measured costs above. Each row is the rate that stage
alone would allow if nothing else ran; the pipeline ceiling is the minimum over the rows, since the
stages overlap in a pipelined design.

| Stage (resource) | Cost basis | 1 x A6000 | 2 x A6000 | 2 x A6000 + 3080 Ti | Binds when |
|---|---|---:|---:|---:|---|
| PE FP16 at today's 45 % efficiency | 7.39 x 5.5 ms + detector 2 ms = 42.6 ms GPU per image | **23.5** | 47 | 55-58 (3080 Ti at about 44 % of an A6000 by spec, 2.1) | now (F3: 74 % busy at 17.4) |
| PE FP16 at 50-55 % efficiency (tuned engine, X-2) | 384.4 GFLOP x 7.39 at 77-85 TFLOP/s | 26-29 | 52-58 | 61-68 | after X-2 if it lands |
| PE INT8 (variant c of WP-1.6), 1.33x on PE as measured for CLIP ViT-L on an A6000 [S18] | 42.6 ms / 1.33 | 31 | 62 | 72-76 | only if WP-1.6 accuracy gates pass |
| PCIe 3.0 x16 H2D, FP32 wire | 14.9 MB per image at about 12 GB/s achievable | about 800 | per GPU | | never at policy `all` |
| PCIe, uint8 wire (WP-2.1) | 3.7 MB per image | about 3,200 | | | never |
| Host CPU, API + Triton | 0.22 CPU-s per image (F4), about 24-30 usable core-equivalents shared with OpenSearch and clients | 110-135 | same host | same host | at policy `selected`/`lazy` and after 2-3 GPUs; earlier if OpenSearch takes half the cores |
| Host CPU after WP-1.3 target (-40 %) | 0.13 CPU-s per image | 185-230 | | | |
| One API event loop | about 0.19 s of Python per image on one core per worker (F4 / 32 workers) | not binding with 32 workers; binds a single-process worker at about 5 img/s per process | | | design rule for X-4: a bulk worker needs process parallelism or the GIL-free parts off the loop |
| OpenSearch writes (one node, one shard per index, HNSW faiss on CPU, RAID data root) | 7.39 vectors + 7.4 docs per image; graph build on CPU | **unknown: measure (X-5)** | same node | same node | candidate binder at 50+ img/s (370+ vectors/s with HNSW insert) |
| Cluster training after ingest | CPU IVF, 50k sample cap, assign and write-back scale with items | not in img/s; adds wall: at 1M images, 6.4M items to fetch (26 GB of FP32 vectors) and rewrite | | | wall-clock at the end of the 1M job (X-5, WP-3.1) |

Binding order on this host at policy `all` (estimate): **(1) PE tensor compute** on each GPU, (2)
host CPU once two or three GPUs feed PE (about 55 img/s needs about 12-15 cores at today's CPU per
image, plus OpenSearch), (3) OpenSearch vector insert and graph build (unmeasured; X-5 decides
whether it is 2 or 3), (4) cluster write-back at the end of the job. PCIe is never binding at policy
`all`; the API event loop binds only a single-process design.

### 2.4 Policy and the embed-all rule (O9)

Embeddings per image set the PE bill directly. Policy `all` must be fast (O9), so the plan does not
lean on policy to hit the number, but the ceiling moves as follows (estimate, one A6000, FP16 at
today's efficiency, PE only):

| Policy | PE per image | PE-bound img/s, 1 x A6000 | 2 x A6000 | Next binder |
|---|---:|---:|---:|---|
| `all` (COCO default vocabulary) | 7.39 | 23.5 | 47 | host CPU at about 55 |
| `selected` (about 1.4 crops + whole frame, the narrow-vocabulary row of `docs/PERFORMANCE.md`) | 2.4 | 70 | 140 | host CPU (110-135 today) |
| `lazy` (whole frame only) | 1.0 | 160 | 320 | host CPU, then detector at about 500 per GPU (B: perf_analyzer 502 inf/s best) |

### 2.5 What 1M images costs in wall time (estimate)

| Configuration (policy `all`) | img/s | 1M images |
|---|---:|---:|
| v0.5.0 (FP32 PE) | 8.9 | 31 h |
| v0.6.0 Wave 1 FP16, one A6000 (measured F1) | 17.4 | 16 h |
| one A6000 at the PE ceiling (host fixed, batching tuned) | 23.5 | 11.8 h |
| two A6000 data-parallel | 47 | 5.9 h |
| two A6000 + 3080 Ti | 55 | 5.1 h |
| plus tuned PE engine (X-2) or INT8 (WP-1.6) | 61-76 | 3.7-4.6 h |
| plus cluster training and final merge at 1M (X-5, WP-3.1) | | + 0.5-1.5 h (unmeasured) |

The 1M job is a 4-6 hour batch on this server at policy `all` if PE runs on both A6000s and the host
and datastore keep up. Without the second GPU it stays above 9 hours whatever else is done (one A6000
with INT8: 31 img/s). For scale: LAION computed ViT-L/14 embeddings for 5B images in about a week
on 32 A100s [S46]; clip-retrieval measured 312 ViT-L/14 (224 px) samples/s per A100 end to end
[S23]. Per GPU, at 336 px (2.4x the FLOPs of 224 px), OpenProcessor's 182 PE embeddings/s on an
A6000 is in the same class.

## 3. External evidence

Collected 2026-10-10. This section adds to, and does not repeat,
`docs/research/triton_deep_research_2026-09.md` (Triton architecture, case studies, catalog) and
`docs/research/dali_gpu_decode_lessons.md` (DALI traps and measured Triton numbers on this GPU).
"Claim" = vendor statement without a published method; "measured" = a published run with hardware
and versions. Numbers measured on other GPUs are context, not predictions for the A6000.

### 3.1 Versions that matter

| Fact | Source |
|---|---|
| The Triton 26.06 base ships TensorRT 11.0 and CUDA 13.3; `Dockerfile.triton` upgrades TensorRT to 11.1.0.106 and asserts lockstep. 26.09 ships TensorRT 11.3, CUDA 13.4.1 | [S9], `Dockerfile.triton` |
| TensorRT 11.0 removed all weak-typing APIs: per-layer precision, the FP16/BF16/INT8/FP8 builder flags, implicit quantization and the INT8 calibrators. Precision lives in the ONNX graph (FP16 casts via Model Optimizer AutoCast or a bake as in PR #225; INT8 via explicit Q/DQ) | [S5], [S19] |
| TensorRT 11 `trtexec` turns CUDA graphs on and data transfers off by default (`--useCudaGraph`, `--noDataTransfers` deprecated as defaults) | [S5] |
| TensorRT 11.0 lists 5-10 % regressions on some CNNs (EfficientNet, RegNet on Blackwell) and a refit regression | [S5] |
| TensorRT 11.4 still supports SM 7.5 and later, including SM86 | [S4] |
| Triton 26.06 and 26.09 known issue: do not use `tritonclient` CUDA shared memory APIs in multithreaded clients (CUDA 13 device API through CuPy) | [S9] |
| `pytorch-quantization` is deprecated in favour of NVIDIA Model Optimizer (`nvidia-modelopt`, Apache-2.0) | [S19] |
| Model Analyzer is deprecated and excluded from Triton since 25.05; GenAI-Perf is being replaced; perf_analyzer is maintained | [S14] |

### 3.2 (a) Triton configuration best practice

| Topic | Guidance and evidence | Source | Applies to OpenProcessor |
|---|---|---|---|
| Dynamic batcher | Start with `dynamic_batching {}`, measure with perf_analyzer, raise `max_queue_delay_microseconds` only within the latency budget; `preferred_batch_size` "should not be used for most models" (exception: TensorRT engines with several profiles) | [S6] | X-1 removes `preferred_batch_size` and tries 2 ms delays (PE and detector are fed by batched callers) |
| Priority and queue policy | `priority_levels`, `default_priority_level`, per-priority `max_queue_size` and timeouts | [S6] | interactive routes over bulk ingest on one server (X-1) |
| Concurrency rule of thumb | for throughput, client concurrency about 2 x max batch x instances; a second instance added little once batching was on (official example) | [S16] | the harness's 4 x 32 is below that for PE; X-1 sweeps it |
| Instances and execution policy | each instance has its own context and stream (copy/compute overlap); TensorRT backend `execution-policy` `BLOCKING` vs `DEVICE_BLOCKING` | [S16], [S37] | PE instance 1 vs 2 in X-1 |
| CUDA graphs | `optimization { cuda { graphs: true, graph_spec [...] } }`; used only when the request shape matches a captured spec; reported crash with `output_copy_stream`; graphs help models with many small kernels (5-15 us launch cost each) | [S7], [S17] | detector at fixed batch sizes; PE (large kernels) gains little |
| Warmup, response cache, rate limiter | warmup delays READY; response cache serves identical requests only; rate limiter weights instances by resources and priority | [S8] | warmup yes (first-request latency); cache no; rate limiter not needed with per-GPU servers |
| Builder options | optimization level 0-5, default 3; timing cache is per device, CUDA and TensorRT version (treat as a build input); workspace via `--memPoolSize`; `--maxAuxStreams` trades memory for parallel layers | [S17], [S38] | X-2 variants; the WP-1.1 build stamp records them |
| Optimization profiles | switching profiles and batch sizes slowed one model 2x in a public report; one profile with `opt` at the common batch avoids switching | [S7] (issue 8171) | PE profile `opt` is 8 while ingest averages 13.5: X-2 |
| 2:4 sparsity | up to about 20 % (A100, ResNeXt-101 FP16) and 1.21-1.40x (A40 = GA102, ResNet-34 INT8, TensorRT 8.6); needs sparse fine-tuning | [S11] | rejected for PE: retraining changes the embedding space |
| Refit, version-compatible engines | lean-runtime engines load across TensorRT versions; load trusted engines only | [S37] | not needed: the installer rebuilds per TensorRT version |

### 3.3 (b) Data movement between client and server

| Topic | Evidence | Source | Reading |
|---|---|---|---|
| uint8 input with in-graph normalization | TensorRT `Cast` handles uint8 to float; Triton `TYPE_UINT8`; 4x fewer bytes by construction | [S8] | WP-2.1 as planned |
| CUDA shared memory vs gRPC | 555 vs 184 infer/s at concurrency 1 in one perf_analyzer report (Triton 24.01), not reproduced by the reporter's own Python client; CUDA vs system shared memory 303 vs 87 infer/s for a 2048 x 2048 model | [S34] | large payloads only; our client is Python and multithreaded (known issue [S9]); system shared memory is the safe path |
| Python client overhead | `tritonclient.grpc` 35-40 ms vs perf_analyzer 15-20 ms for the same YOLOv5 TensorRT request on an A100 | [S35] | perf_analyzer numbers overstate what the API achieves; X-1 reports both |
| Server memory pools | defaults 64 MB CUDA pool per GPU, 256 MB pinned; exhaustion logs a fallback to pinned host memory | [S36] | size the pools only when GPU tensors cross a BLS or DALI boundary (parent plan) |
| BLS vs ensemble vs Python backend | a face pipeline: BLS about 35 tps at 50 % GPU vs ensemble about 60 tps at 95 %; several reports of ensembles slower than separate calls; Python-backend string handling up to 30x slower than a C++ rewrite | [S10] | user reports, causes unconfirmed: confirms the parent plan's "few, large BLS calls" and the WP-2.4 15 % stop rule |
| DLPack in the Python backend | zero-copy since 21.09; inputs forced to CPU unless `FORCE_CPU_ONLY_INPUT_TENSORS` is `no`; importing torch adds about 890 MB GPU memory per stub | [S10] | budget for WP-2.4 |

### 3.4 (c) GPU decode and preprocessing

| Topic | Evidence | Source |
|---|---|---|
| Hardware JPEG engine | nvJPEG lists hardware decode for A100, A30, Hopper, Ada, Blackwell, Jetson Thor; GA102 (RTX A6000, RTX 3080 Ti) is not listed, and the A6000 datasheet names only NVENC/NVDEC video engines: **no hardware JPEG decoder on this server**; `hw_decoder_load` does nothing here | [S24], [S1] |
| Hybrid Huffman | DALI's `mixed` decoder sends images above `hybrid_huffman_threshold` (default 1,000,000 px) to a hybrid decoder that "still largely uses the CPU": 12-20 MP photos keep a CPU cost on GPU decode; GPU Huffman applies in `nvjpegDecodeBatched` for baseline JPEGs at batches above 50 | [S25], [S24] |
| nvImageCodec | 13.6 MP baseline JPEG: 24 ms CPU (libjpeg-turbo) vs 8 ms GPU single, 3 ms per image batched at 16, RTX 3090 (SM86), max pixel difference 1 | [S26] (measured) |
| torchvision `decode_jpeg(device='cuda')` | batched list input; deprecated since torchvision 0.29 in favour of TorchCodec | [S27] |
| CV-CUDA | pipeline-level claims (5-48x vs OpenCV CPU on T4/L4 video segmentation), no per-operator table; u8 resize up to 4x faster in v0.9 | [S28] (claim) |
| Letterbox parity | DALI resize must be `antialias=False` to match `cv2.INTER_LINEAR`; PIL antialiases downscales; torchvision tensor resize antialiases by default since 0.17 but is not identical to PIL; set `jpeg_fancy_upsampling=True` to match libjpeg-turbo chroma | [S29], [S27], [S25], `docs/research/dali_gpu_decode_lessons.md` |

### 3.5 (d) Precision

| Topic | Evidence | Source |
|---|---|---|
| INT8 on SM86 for ViT-L | CLIP ViT-L/14, A6000, TensorRT 10.9: 1.33x at batch 8 with explicit Q/DQ from Model Optimizer; a naive placement ran 2x **slower** than FP16 (Q/DQ on Transpose outputs and on Add inputs; FP16 must be requested as `high_precision_dtype` in Model Optimizer because builder flags are ignored for strongly typed networks) | [S18] (measured, NVIDIA maintainer) |
| ViT PTQ accuracy | W8A8 PTQ on ImageNet: ViT-B/384 86.00 to 85.82, DeiT-B 81.80 to 81.48 (PTQ4ViT); larger ViTs less sensitive. No published W8A8 result for ViT-L retrieval or PE: WP-1.6 must measure | [S40] |
| YOLO INT8 | YOLOv8n TensorRT: mAP50-95 0.37 to 0.33 for a small latency gain; NVIDIA YOLOv7 QAT keeps mAP (0.5113 vs 0.5124) at INT8 speed. Detector is 0.8 % of FLOPs here: no reason to quantize it | [S20], [S41] |
| FP8 | needs SM 8.9 or later | [S4] |

### 3.6 (e) Scheduling and multi-GPU

| Topic | Evidence | Source |
|---|---|---|
| Triton across GPUs | `instance_group { count, gpus }` places instances per listed GPU; default is one instance per visible GPU; load balancing does not weight by GPU speed (A6000 and 3080 Ti differ about 2.3x on FP16 with FP32 accumulate) | [S8], [S47] |
| MPS | useful when kernels underfill the GPU (an ASR case cut GPU count 75 % on AWS); one MPS server per user; GeForce support not confirmed by NVIDIA | [S13] |
| Clocks | `nvidia-smi -ac` deprecated; `-lgc`/`-lmc` lock clocks but the power limit still caps them; persistence through the daemon | [S39] |
| NUMA | pin the server and the decode workers to the GPU's NUMA node from `nvidia-smi topo -m` (guidance; this host: GPU0 node 0, GPU2 node 1) | [S39], section 1.2 |

### 3.7 (f) Serving alternatives for the same models

| Option | Evidence for vision/embedding serving | Status, licence | Verdict here |
|---|---|---|---|
| Triton (TensorRT backend) | server overhead tens of microseconds per request in public traces; queueing and transfer dominate large tensors | active, BSD-3-Clause [S44] | keep |
| ONNX Runtime GPU + TensorRT EP + I/O binding | same TensorRT kernels; graph partitioning can leave subgraphs on CUDA/CPU; without I/O binding inputs and outputs go through the host; engine cache needed | active, MIT [S38] | equal at best; it is ortloom's engine path |
| Rust `ort` | wraps the same ONNX Runtime; speed claims are marketing | active, MIT OR Apache-2.0 [S45] | engine parity |
| Plain TensorRT + CUDA streams in a custom service | no published comparison; NVIDIA's guidance is one context and stream per worker | TensorRT runtime under the NVIDIA licence | architecture E |
| TorchServe | archived 2025-08-07, no security patches | Apache-2.0 [S15] | no |
| BentoML, Ray Serve, KServe | no GPU vision throughput numbers found; Python orchestration over an engine | Apache-2.0 | no gain over Triton for TensorRT plans |
| NVIDIA NIM (NV-CLIP) | ViT-H/14 1024-d CLIP endpoint, base64 images over an OpenAI-style API; no YOLO, SCRFD or PE NIM | proprietary terms [S12] | does not serve our models |
| DeepStream, Holoscan | video and sensor pipelines | DeepStream source CC-BY-4.0 AND Apache-2.0, binaries under the NVIDIA SDK licence; Holoscan Apache-2.0 [S43] | out of scope (still images; `epic:video` later) |

### 3.8 (g) Bulk and offline patterns at 1M scale

| Pattern | Evidence | Source | Take-away |
|---|---|---|---|
| NVIDIA NeMo Curator image curation | WebDataset tar input, DALI GPU reader, CLIP ViT-L/14 embeddings, filters, semantic dedup, Parquet out, on Ray; embedder defaults 0.25 GPU per worker, batch 32; no images/s published | [S30] | the same shape as architecture C: sharded input, GPU decode, batched encoder, columnar output, resumable stages |
| clip-retrieval (LAION) | ViT-L/14 GPU bound and linear to 8 GPUs; ViT-B/32 resize bound | [S23] | big encoders scale linearly with GPUs when the host keeps up: supports X-3 |
| Ray Data batch inference | 800k ImageNet images, ResNet-18: time fell from 456 s to 111 s as CPUs per worker rose from 4 to 32: CPU decode bound | [S31] | for small models the host binds; for PE it binds only after 2-3 GPUs (section 2.3) |
| OpenSearch bulk load | during load: `refresh_interval -1`, replicas 0, more k-NN index threads; one-time loads may defer graph build with `index.knn.advanced.approximate_threshold` then force-merge; OpenSearch 3.0+ can build Faiss HNSW graphs remotely on a GPU (CAGRA via cuVS) for segments of 50 MB or more; bulk requests of about 5-15 MiB | [S32] | X-5 arms; the remote GPU build is a WP-V candidate |
| Milvus bulk import | BulkWriter writes Parquet, `bulk_import` loads it without the proxy path; no rows/s published | [S33] | WP-V (#221) |

### 3.9 (h) Throughput references for the models used

| Model | Published | Source |
|---|---|---|
| ViT-L/14-336 (CLIP) | 381.9 GFLOPs, 304 M params (image tower) | [S21] |
| PE-Core-L14-336 | 192.3 GMAC (about 385 GFLOP), 317 M params; matches the count in section 1.3 | [S22], [S42] |
| ViT-L/14-336 throughput | RTX 3090, PyTorch AMP eager: 157.7 img/s at batch 768; PE-Core-L14-336 runs at 0.65x CLIP-L on the same GPU in eager mode | [S22] |
| YOLO11s | 21.6 GFLOPs, 2.5 ms TensorRT FP16 on a T4 (batch 1 assumed) | [S20] |
| SCRFD-10G | 10 GFLOPs at 640 x 480 | [S48] |
| ArcFace R50 | 6.3 GFLOPs as named by insightface (that is MACs; 12.6 GFLOP by the 2 x MAC convention) | [S49] |
| MobileCLIP2-S2 | 35.7 M image params; latency published for a phone only | [S50] |

## 4. Candidate architectures

| ID | Architecture | Model server | Who orchestrates ingest | Data path |
|---|---|---|---|---|
| A | Today (v0.5.0 + FP16 PE) | Triton 26.06, TensorRT plans, one GPU | FastAPI workers (`/ingest/upload`, `/ingest/batch`), per-image PE calls | CPU decode, CPU letterbox and crop resize, FP32 inline gRPC |
| B | Triton hardened | Triton, same plans plus uint8 versions; tuned batching, instances, warmup, CUDA graphs where shapes are fixed, priority levels, explicit CUDA pool; optional DALI ensembles for stateless routes and the `ingest_pipeline` BLS of WP-2.4 | FastAPI workers | uint8 tensors, system shared memory, cross-image PE batching; GPU decode only if WP-2.4's go/no-go passes |
| C | B + a bulk ingest runner | Triton as in B, one Triton per GPU (or instance groups across GPUs, X-3) | a dedicated bulk process (the existing ingest service code run outside the API request path) for jobs of 10k-1M images; the API keeps interactive routes | manifest-driven prefetch (bounded queues, process-parallel decode and resize), cross-image PE batches, async gRPC with shared memory to each GPU's Triton, checkpoint by content hash, bulk writes tuned for load |
| D | ortloom-serve as the model server | ortloom-serve extended with a generic tensor backend, multi-GPU, model control, Triton metric names | FastAPI (unchanged client: KServe v2 gRPC) | as B, plus ortloom's nvJPEG decode pipeline where JPEG-in fits |
| E | Custom offline pipeline for bulk only | none for bulk (TensorRT Python/C++ runtime with CUDA streams, DALI or nvImageCodec decode, CV-CUDA resize in-process); Triton keeps the online routes | an offline job binary | GPU decode, GPU crops, in-process TensorRT, no IPC |

Notes that hold for every option:

- **The engine is the same.** All five run the same TensorRT kernels for PE (TensorRT plan through
  Triton's TensorRT backend, through the ORT TensorRT execution provider, or in-process). Per
  embedding GPU time is set by the engine build (precision, tactics, fused attention), not by the
  server. A serving layer can only change how full the GPU is kept and what the host pays per image.
- **The GPU is about 74 % busy at F1.** The serving-side prize on one A6000 is the remaining 26 %
  (17.4 to about 23.5 img/s); the larger prizes are the second A6000 (2x) and the engine itself
  (X-2, INT8).
- **Parity.** Any path that changes decode, resize or crop numerics (DALI, nvJPEG, CV-CUDA, GPU
  interpolation) falls under rule D2 of the wave plan and the WP-2.4 gates; a second preprocessing
  implementation for bulk only (E) must match the online one per gate, forever.

## 5. ortloom-serve: what it is and what it lacks for this workload

Read from the ortloom repository (owner's project, `MIT OR Apache-2.0`, version 0.1.1, marked
pre-release in its README; paths below are relative to that repository, read on 2026-10-10 at
`v0.1.1-5-g7c7bf43`).

### 5.1 What it is

A Rust server on the `ort` crate (`=2.0.0-rc.12`, ONNX Runtime 1.24) that speaks the KServe v2
protocol and is built around **Ultralytics YOLO detection**: it refuses a model whose input is not
FP32 square NCHW and parses outputs through YOLO "contracts" (`src/engine.rs:1557-1575`,
`src/contract/`). Its own documentation says to use Triton for more than YOLO detection
(`README.md:209`, `docs/comparison.md:70-73`). Its distinctive strength is a JPEG-in pipeline:
nvJPEG decode lanes with their own CUDA streams and pinned buffers, a CUDA letterbox kernel that
matches the CPU filter bit for bit, written straight into the batch slot, one batching stage, and
decode of newly admitted requests overlapped with inference of the current batch
(`src/decode/nvjpeg.rs:42-68`, `src/instance.rs:524-576`, `docs/batching.md:10-15`,
`docs/benchmarks.md:277-284`).

### 5.2 Measured against Triton (its own benchmarks)

One RTX A6000 on this host class (48-thread Xeon), shared and loaded host, Rust closed-loop load
generator, YOLO26n / YOLO11n at 640, 20 MP JPEGs where JPEG-in, max batch 8, 2 ms queue delay on
both servers. Engines differ: ortloom used the ORT TensorRT execution provider on TensorRT 10.15.1,
Triton a native plan on TensorRT 11.2.1 (`docs/benchmarks.md:76-97`, whitepaper
`whitepaper/ortloom_serve.tex:673-699` lists the threats to validity). img/s at 1 / 8 / 32 / 128
clients:

| Scenario | ortloom | Triton | Reading |
|---|---|---|---|
| Raw FP32 tensors, YOLO26n, TensorRT FP16 | 70 / 256 / 360 / 462 | **76.5 / 324 / 479 / 813** | Triton wins on raw tensors: pinned staging and overlapped copies; ortloom copies pageable memory on one thread (`docs/benchmarks.md:164-169, 289-293`) |
| JPEG in, GPU decode, tuned (16 nvJPEG lanes vs 4 DALI instances) | 8/32/128: **136 / 312 / 340** | 88 / 214 / 321 | ortloom wins at 8-32 clients by 1.0-1.5x; p99 lower at 8-32 (`docs/benchmarks.md:256-260`) |
| JPEG in, CPU decode | **36 / 91 / 97** | 20 / 48 / 54 | ortloom about 1.8x: decode overlapped with inference instead of DALI CPU decode serialized per batch |
| GPU memory, JPEG in on GPU, 32 clients | 4.1 GB | 9.3 GB | DALI instances are memory-hungry (`docs/benchmarks.md:307-312`) |

The same comparison is summarized publicly in `docs/research/dali_gpu_decode_lessons.md` (section
5), where an untuned DALI grid ran 1.5-3x behind the nvJPEG pipeline. The advantage is **server-side
decode scheduling** (one batching stage, decode overlapped with inference), not the protocol and
not the inference engine; on raw tensors Triton is faster.

### 5.3 Gaps against what OpenProcessor uses from Triton

| OpenProcessor uses | ortloom-serve today | Effort to close (S/M/L) |
|---|---|---|
| Arbitrary models: ViT encoder (PE), MobileCLIP, ArcFace, SCRFD, PaddleOCR det/rec, text encoders | YOLO detection only (FP32 square NCHW input, YOLO output contracts) | **L**: a generic tensor backend (any input/output names, dtypes including UINT8/INT64/BYTES, non-square and dynamic shapes) |
| Native TensorRT `.plan` files built by `export/` and `scripts/lib/model_setup.sh` (TensorRT 11.1 in the Triton image) | ONNX only; TensorRT through the ORT TensorRT execution provider (engine cache, EPContext `model_ctx.onnx`; `src/tensorrt.rs:4-32, 238-273`); fp32/fp16 only, no INT8 | **M per model** plus a parity pass: a second build path for every model, and the ORT EP may partition graphs |
| Triton `config.pbtxt` model repository, `version_policy` (rollback of uint8 versions in WP-2.1) | `model.toml` with `max_batch`, `max_queue_delay_ms`, `instances`; highest version wins (`ortloom-serve/src/repository.rs:1-56, 241-270`) | S-M |
| `instance_group.gpus`, multi-GPU placement | single CUDA device, default 0, never set by the server (`src/engine.rs:102, 138, 186`) | M |
| Explicit model control (`--model-control-mode=explicit`, load/unload/index used by `./openprocessor models` and the API's model status) | load at startup only; repository API absent (deferred, `docs/design/serving_plan.md:246, 503-504`) | M |
| Python BLS (`ocr_pipeline`), ensembles | none (`docs/comparison.md:53-58`) | L (or keep OCR on Triton) |
| ONNX Runtime CPU model (`pe_text_encoder`) | CPU EP exists, YOLO only | part of the generic backend |
| `/v2/health/*`, `/v2/models/stats`, KServe gRPC `ModelInfer` (what `tritonclient.grpc.aio` calls), `ModelStreamInfer` | present (`ortloom-serve/src/grpc/mod.rs:97-226`) | none |
| Prometheus `nv_inference_*` names on :8002 (Grafana dashboards, `baseline_suite.py` Triton phase) | `ortloom_*` names, seconds histograms, API key required (`ortloom-serve/src/metrics.rs:14-19`) | S |
| System / CUDA shared memory (WP-2.2 plans system shared memory) | absent (`docs/server-api.md:220`) | M |
| Pinned staging and overlapped H2D on raw tensors | absent: pageable copy on one worker thread (`docs/benchmarks.md:289-293`) | M |
| GPU crop fan-out (decode once, detect, cut N crops, resize to 336, one PE batch) | absent; its pipeline is decode then letterbox then YOLO then CPU NMS | L, and it is the only feature that would beat Triton for this workload (section 10) |

## 6. Trade matrix

Scores 1 (worst) to 5 (best). Ceilings are section 2 estimates at policy `all`; "1 GPU / 3 GPUs"
means one A6000 / two A6000 plus the 3080 Ti.

| Criterion | A today | B Triton hardened | C B + bulk runner | D ortloom-serve | E custom offline |
|---|---|---|---|---|---|
| Throughput ceiling, img/s (1 GPU / 3 GPUs) | 17-19 / n.a. (everything pinned to GPU 0) | 22-24 / 45-55 with instance groups across GPUs | 22-24 / 50-58 with per-GPU Triton and process-parallel host work | same PE engine: 22-24 / 50-58 once multi-GPU exists; better on 20 MP JPEG-in decode | 23-25 / 52-60 (saves IPC and copies, not PE time) |
| Score | 2 | 4 | **5** | 3 | 4 |
| Effort and risk | 5 (nothing) | 4 (config, uint8 re-export, shared memory: already WP-2.1/2.2) | 3 (a bulk process, job state, resume) | 1 (L gaps: generic backend, plan loading or re-export of every model, model control, multi-GPU, OCR BLS) | 2 (new decode, resize, crop, TensorRT runtime and writer glue; parity for all of it) |
| Maintainability | 4 | 4 | 4 (reuses the ingest service code; one more entry point) | 2 (two model servers while OCR and text encoders stay on Triton, or a large port) | 1 (a second preprocessing implementation that must stay bit-compatible with the online one) |
| Operability (installer, health, metrics, model control) | 5 | 5 | 4 (one more service; job status) | 2 (no repository API, different metric names, own engine cache) | 2 (no server, no metrics unless added) |
| Compatibility (API contract, `tritonclient`, curation datastore, wire) | 5 | 5 (transport-only changes are P-json gated) | 5 (same writers, same OCC helpers) | 3 (KServe gRPC works; repository, exporters, installer change) | 3 (must call the same writers; preprocessing duplicate) |
| Multi-GPU scaling | 1 | 3 (one process spans GPUs; cross-socket copies for GPU 2) | **5** (one Triton per GPU, NUMA-pinned, client shards) | 1 today (device 0 only) | 4 |
| Licence (repo is AGPL-3.0-or-later) | Triton BSD-3-Clause | + DALI Apache-2.0 | same | MIT OR Apache-2.0 (compatible) | + CV-CUDA, nvImageCodec Apache-2.0; TensorRT runtime under the NVIDIA licence (already the case) |
| Fit with "no shims, no duplicate logic, no parallel routes" | 5 | 4 (a GPU backend setting next to the CPU path is an owner decision in the parent plan, section 1 item 4) | 4 (must reuse, not copy, the ingest functions) | 2 (two servers in parallel, or a replacement that still needs Triton for OCR) | 1 (two pipelines doing the same job) |
| **Total (of 35, licence not scored)** | 27 | 29 | **30** | 14 | 17 |

Reading: B and C are the same stack at two levels of ambition; C is B plus what a 1M job needs
(multi-GPU sharding, prefetch, resume, load-time datastore settings). D does not raise the ceiling
for this workload (PE is compute bound in the same TensorRT kernels) and costs the most. E buys a
few percent over C at the price of a permanent second pipeline.

## 7. Recommendation

1. **Keep Triton as the model server** for every route and for bulk ingest. Harden it (B) along the
   wave plan's existing packages, and add only what the measurements of section 2 say is missing:
   batching and concurrency tuning on the FP16 stack (X-1), an engine efficiency pass on PE (X-2),
   and **data-parallel PE on both A6000s** (X-3). These three are where the images/s are.
2. **Build the bulk ingest runner (C)** after WP-2.1 to WP-2.3 and X-3: the existing ingest service
   functions driven by a manifest, with process-parallel prefetch, cross-image PE batches, one Triton
   endpoint per GPU, resume by content hash and load-time datastore settings (X-4, X-5). It is the
   vehicle for the scale ladder 10k, 100k, 1M (X-7, O8).
3. **Do not switch to ortloom-serve in v0.6.0.** For this workload 99 % of the FLOPs are PE in the
   same TensorRT kernels whichever server runs them; ortloom's measured wins are on JPEG-in YOLO
   pipelines, and it lacks a generic tensor backend, multi-GPU, plan loading and model control.
   Run the head-to-head X-6 to settle it with numbers; extend ortloom only on its go criterion
   (section 10).
4. **Do not build a separate offline pipeline (E)** unless C, after X-1 to X-4, stays more than 15 %
   below the per-GPU PE ceiling with the GPU idle for host reasons that C cannot remove.
5. **Precision is the second multiplier after GPUs:** the INT8 evaluation (#213) stays as planned,
   with explicit Q/DQ through NVIDIA's Model Optimizer because TensorRT 11 removed implicit INT8
   calibration [S5].
6. **GPU decode (#217)** keeps its go/no-go; add the arms listed in section 8 (X-8 note) so the
   decision also covers nvImageCodec and ortloom's nvJPEG pipeline on the high-resolution set.

Expected path (estimate, policy `all`): 17.4 img/s now; about 21-23 after X-1 and the Wave 1 host
fixes; 45-55 with both A6000s (X-3/X-4) and the 3080 Ti; 61-76 if X-2 or INT8 land. A 1M-image job
then takes about 4-6 hours, plus the final cluster pass.

## 8. Experiment plan

Protocol for every experiment (wave plan section 0 rule 3): isolated compose project (`-p <name>`),
the pinned COCO 2,000-image manifest `scripts/datasets/manifests/coco_bench_2000.json`, harness
`scripts/bench/baseline_suite.py` (PR #206), 100-image warm-up, **3 repetitions, median (min-max)**,
policy `all` headline row, host load average recorded and runs above 8 discarded, raw JSON under
`artifacts_local/bench/v060/<wp>/<arm>/`. Additionally for every GPU run: `nvidia-smi dmon -s pucv
-d 1` (power, SM clock, utilisation) saved with the JSON, so power-capped clocks are visible.
Reference state: the FP16 stack of PR #225 ("ref-fp16"). Unless a row says otherwise, a candidate
is adopted when its median img/s is at least 5 % higher with p99 request latency not worse by more
than 10 % (wave plan / parent plan benchmark gate); knobs in a tuning sweep need 3 %.

| ID / WP | Question | Arms | Go / no-go (measured) |
|---|---|---|---|
| **X-1 / WP-S1 (#226)** Triton and client sweep on the FP16 stack (no rebuild) | Why is the GPU 26 % idle at F1, and which knobs fill it? | Harness `--concurrency` 4, 8, 16 x `--batch-size` 8, 32; PE `instance_group.count` 1, 2; PE and detector `max_queue_delay_microseconds` 15000 vs 2000; `preferred_batch_size` removed vs today [S6]; `--log-verbose=0`; `model_warmup` at batch 1 and 32; detector CUDA graphs with one `graph_spec` per batch size used (16, 32) [S7]; `priority_levels: 2` with the interactive routes at priority 1 (measure `/detect` p99 while ingest runs) | Adopt each knob by the 3 % rule. **Target: GPU busy >= 90 % and >= 21 img/s (90 % of the 23.5 ceiling).** If no arm passes 85 % busy, the host binds: stop, run WP-1.2/1.3 first, then re-run X-1 |
| **X-2 / WP-S2 (#227)** PE engine efficiency | Is 45 % of FP16 peak the best TensorRT 11.1 does for PE on SM86? | Layer dump of the FP16 plan (`--dumpLayerInfo --profilingVerbosity=detailed`): attention fused into one kernel per block, MatMuls FP16, where the 127 inserted casts sit; then rebuild variants: optimization profile `opt` 16 and 32 (today 8, while ingest batches average 13.5), `--builderOptimizationLevel=5`, a persisted timing cache, `--maxAuxStreams` default vs 0. Measure perf_analyzer through Triton at batch 8/16/32 (trtexec in TensorRT 11 runs CUDA graphs and skips transfers by default [S5], so its numbers are an upper bound only) | Adopt a variant at **>= 10 % PE embeddings/s** at batch 16 through Triton, with P-pe p01 >= 0.9999 against ref-fp16 (same precision, only tactic rounding moves). If the plan already runs fused attention and reaches >= 55 % of peak, close as measured |
| **X-3 / WP-S3 (#228)** Data-parallel PE across GPUs | How close to 2x do two A6000s get, and in which process layout? | (a) one Triton, PE and detector `instance_group { gpus: [0, 2] }` (Triton places one instance set per listed GPU [S8]); (b) one Triton per GPU (two services, each `CUDA_VISIBLE_DEVICES` one GPU, `cpuset` on that GPU's NUMA node from section 1.2), client sends round-robin or least-outstanding across the two gRPC endpoints; (c) arm (b) plus the 3080 Ti as a third endpoint with a PE profile sized for 12 GB. Run the 2,000 pin **and** the 10k pin (the 2k pin lasts under a minute at 45 img/s) | **Two-A6000 scaling >= 1.8x** the one-A6000 median img/s at policy `all`; pick the arm by img/s, then p99, then operability. Below 1.6x: the host binds; publish the stage table before adding GPUs. GPU 2 needs the owner's go for that run window (open question Q1) |
| **X-4 / WP-S4 (#229)** Bulk ingest runner (architecture C) | Does a dedicated process beat `/ingest/batch` at the same GPUs, and does it resume? | Runner reusing `src/services/curation/ingest*` (no copied logic): manifest in, N decode/resize processes feeding bounded queues, cross-image PE chunks of 32, uint8 inputs and system shared memory (after WP-2.1/2.2), one Triton endpoint per GPU (X-3 arm b), bulk writes through the existing OCC helpers. Arms: the runner vs `/ingest/batch` with the X-1 best client settings | **>= 1.25x img/s** over `/ingest/batch` on the same GPUs, P-json identical stored documents (minus ids and timestamps), kill at 50 % and restart gives no duplicate and the same final document count, interactive `/detect` p99 during the run <= 2x its idle p99. Below 1.25x: keep `/ingest/batch` plus the ingest walker as the bulk path and fold the prefetch changes into the service |
| **X-5 / WP-S5 (#230)** Datastore write ceiling at scale | Can one OpenSearch node absorb 50-100 img/s of item and image documents with 7.4 HNSW vectors per image, and what does the final cluster pass cost at 100k and 1M items? | Replay documents captured from a 10k ingest into fresh project indexes at offered loads of 25, 50, 100, 200 img/s; arms: today's settings (1 shard, 0 replicas, `hnsw_ef_construction` 512, m 16, faiss); `refresh_interval` -1 during load then restore; `hnsw_ef_construction` 128 (recall checked with the WP-V recall harness, #221); bulk size 500 vs 2,000 docs; client threads 4 vs 16; plus the cluster pass (fetch, train, assign, write-back stage times from WP-3.1) at 100k items | **Sustain >= 100 img/s equivalent (about 740 vectors/s plus 840 documents/s)** with search-ready time after load reported. If today's settings sustain less than the X-3 img/s, adopt the cheapest arm that does, gated by WP-V's recall@10 >= 0.95 against exact search; any index-setting change that alters recall is the owner's call |
| **X-6 / WP-S6 (#231)** ortloom-serve head-to-head | Does ortloom-serve beat Triton on our ingest-path models at realistic request shapes? | (a) detector raw tensors from the Python `tritonclient.grpc.aio` client at batch 1, 8, 16 and concurrency 8, 32, 128 (works today: YOLO contract); (b) JPEG in, decode plus letterbox plus detect, COCO pin and the high-resolution set B (owner run), vs the Triton DALI ensemble and vs the CPU client path; (c) PE raw tensors at batch 8, 16, 32 once ortloom has a minimal generic tensor passthrough (feature 1 in section 10; built in the ortloom repository, not here). Same host, same GPU, same ONNX source; TensorRT versions recorded (ORT EP vs Triton plan) | **Extend for OpenProcessor only if** (c) PE embeddings/s >= 1.10x Triton at equal or better p99, **or** (b) on set B >= 1.3x the best Triton arm with the P-box/P-item gates passing **and** WP-2.4 is a go. Otherwise record and close; ortloom stays a separate YOLO server |
| **X-7 / WP-S7 (#232)** Scale ladder 10k, 100k, 1M | Does the rate hold as indexes grow, and does a 1M job finish? | The shipped levers at each wave exit; 10k and 100k from COCO train2017 pins (`fetch_coco_subset.py --bench-set`), 1M from Open Images (O7 allows it; manifest committed, images not) | 100k at >= 90 % of the 2k-pin img/s; 1M completes with zero failed items and resumes after a forced stop; publish img/s over time, OpenSearch merge and heap curves, total wall including the cluster pass |

X-8 (note on #217, not a new package): when WP-2.4's go/no-go runs, its bake-off adds two arms to
the parent plan's A/B/C: nvImageCodec batched decode inside the BLS model (no DALI pipeline
serialization) and ortloom's nvJPEG pipeline as an external reference (X-6 b). Decode on COCO-sized
images is 4.6 ms (B6) and stays on the CPU unless set B says otherwise.

## 9. What not to do

| Do not | Why (evidence) |
|---|---|
| Replace Triton for the online routes in v0.6.0 | No measured gain for compute-bound PE; ortloom lacks generic models, plan loading, model control, multi-GPU (section 5.3) |
| Serve GPU models through Triton's ONNX Runtime backend | cuDNN plan rebuilds per new batch shape collapsed it to 11-15 img/s vs about 300 for the TensorRT plan on this GPU (`docs/research/dali_gpu_decode_lessons.md` section 6) |
| Use CUDA shared memory between the API and Triton | Triton 26.06 and 26.09 list a known issue with `tritonclient` CUDA shared memory in multithreaded clients [S9]; the API container would need a GPU and the same device as Triton. System shared memory (WP-2.2) is the planned path |
| Call BLS per crop, or chain ensembles for variable-count work | BLS and ensemble overheads reported at 2x or worse in public issues [S10]; parent plan decision 5 already says few large calls |
| Use DALI CPU decode inside Triton | slowest measured arm, about 28 img/s cap (`docs/research/dali_gpu_decode_lessons.md` section 5) |
| Chase transport (shared memory, gRPC tuning) for img/s at policy `all` before X-1 | PCIe and wire are not the binder (section 2.3); WP-2.2 is a CPU saving, measured as such |
| Use FP8 or FP4 | not supported on SM86 (Ampere) [S2], [S5] |
| Use 2:4 structured sparsity on PE | needs sparse fine-tuning, which changes the embedding space; measured gains on Ampere are 1.2-1.4x on CNNs and none is published for a ViT-L [S11] |
| Enable the response cache for image models | it only helps identical requests [S8]; ingest images are unique by content hash |
| Use MPS with one Triton per GPU | Triton already multiplexes instances with CUDA streams inside one process; MPS adds a daemon and failure coupling for no measured need on this host [S13] |
| Depend on Model Analyzer or GenAI-Perf in the harness | Model Analyzer is deprecated and excluded from Triton since 25.05; GenAI-Perf is being replaced [S14]; perf_analyzer stays |
| Adopt TorchServe or a Python-native server for TensorRT models | TorchServe is archived [S15]; Python servers add host overhead without a faster engine |
| Compare trtexec numbers from TensorRT 11 with Triton end-to-end numbers | TensorRT 11 trtexec enables CUDA graphs and skips data transfers by default [S5] |
| Change PE preprocessing (squash, full-resolution whole frame) for speed | a new embedding space: parent plan Wave 10 only |

## 10. Should ortloom-serve be extended?

**Not as OpenProcessor's model server in v0.6.0; possibly as its GPU decode-and-crop engine later,
decided by X-6.** The evidence:

- What decides throughput here is PE tensor compute (99 % of FLOPs, section 1.3). ortloom-serve would
  run the same TensorRT kernels through the ORT execution provider; the best it can do on PE is
  parity, and its raw-tensor path is measured behind Triton today (section 5.2).
- Where ortloom-serve wins (JPEG in, decode overlapped with inference, one batching stage, lower GPU
  memory than DALI) is exactly the part of OpenProcessor's plan that is conditional (WP-2.4) and
  matters on 12-20 MP photos, not on COCO.
- The decisive measurement is X-6 (c) and (b): PE at ingest shapes from the Python client, and the
  JPEG-in path on set B against Triton's best arm.

If X-6 passes, the first features, in order (each is a change in the ortloom repository, owned there):

| # | Feature | Why it is first | Gap row (5.3) |
|---|---|---|---|
| 1 | Generic tensor backend: arbitrary input/output names and dtypes (UINT8, INT64, BYTES), non-square and dynamic shapes, no YOLO contract | without it no OpenProcessor model except the detector can be served; it is also what X-6 (c) needs | arbitrary models |
| 2 | Pinned staging and overlapped H2D/D2H on the raw-tensor path (two buffers per instance) | its raw path loses to Triton at 32-128 clients because of pageable single-thread copies | pinned staging |
| 3 | Multi-GPU placement (instances per device, device per model) | the second A6000 is the largest lever in this study | multi-GPU |
| 4 | Decode-once crop fan-out: JPEG in, detector, N crops cut and resized on the GPU to 336, one PE batch, vectors out | the only feature that would beat Triton for this workload (it is WP-2.4's BLS done without Python stubs) | GPU crop fan-out |
| 5 | Triton-compatible operations: `nv_inference_*` metric names, repository index/load/unload, version selection | lets `./openprocessor models`, the API's model status and the benchmark harness work unchanged | model control, metrics |

Native TensorRT plan loading (or an export path into ortloom's engine format with the build stamp of
WP-1.1) is required before any production use, but it does not decide the experiment.

## 11. Work packages and issues

| WP | Issue | Experiment | Wave (wave plan section 13) | Depends on | Tier (plan / implement) |
|---|---|---|---|---|---|
| WP-S1 Triton and client sweep | #226 | X-1 | 1 (after WP-1.1) | WP-1.1 (#209); WP-1.2/1.3 if the host binds | Sonnet / Sonnet |
| WP-S2 PE engine efficiency | #227 | X-2 | 1 | WP-1.1 | Sonnet, Opus reviews the layer dump |
| WP-S3 data-parallel PE across GPUs | #228 | X-3 | 2 (first item) | WP-S1; owner's GPU window | Opus sub-plan / Sonnet |
| WP-S4 bulk ingest runner | #229 | X-4 | 2 (last item) | WP-2.1, 2.2, 2.3, WP-S3 | Opus / Sonnet |
| WP-S5 datastore write ceiling | #230 | X-5 | parallel track V | WP-0.0; shares harness with WP-V (#221) | Sonnet / Sonnet |
| WP-S6 ortloom-serve head-to-head | #231 | X-6 | parallel, after WP-S2 | WP-S2 (tuned Triton baseline); arm (c) needs ortloom feature 1 | Sonnet / Opus decides |
| WP-S7 scale ladder 10k, 100k, 1M | #232 | X-7 | at each wave exit | the levers shipped so far | Sonnet; owner schedules |

Changes to existing packages (recorded in the wave plan, no new issue): #213 (INT8) uses NVIDIA Model
Optimizer explicit Q/DQ with `high_precision_dtype` FP16 and the Q/DQ placement lessons of [S18];
#217 (GPU decode) adds the nvImageCodec and ortloom nvJPEG arms (X-8 note in section 8).

## 12. Open questions for the owner

| ID | Question | Recommendation |
|---|---|---|
| Q1 | May X-3 and X-7 use GPU 2 (and GPU 1) in scheduled windows? The wave plan's ground rule 1 reserves GPU 2 for another project; O3 allows any free GPU. | Yes for scheduled windows; the 1M target depends on it (section 2.5) |
| Q2 | Is 4-6 hours for 1M images at policy `all` the goal, or is a lower wall time wanted (which then needs INT8, or `selected` policy for some datasets)? | Accept 4-6 h as the v0.6.0 target; revisit after #213 |
| Q3 | Should ortloom gain a generic tensor backend now (feature 1, in the ortloom repository) so X-6 arm (c) can run, or wait for arms (a) and (b)? | Wait for (a) and (b); build feature 1 only if (b) on set B shows the decode pipeline is worth integrating |
| Q4 | For the bulk runner (WP-S4): a long-running service with a job route (needs `make contracts`), or a CLI script bound to a project like the other scripts? | CLI script first (no contract change); a route only if Cropwright needs to start bulk jobs |
| Q5 | 1M public images come from Open Images (O7). Is local use of the full Open Images train set acceptable for aggregate published numbers? | Yes, with the manifest committed and images never committed |
| Q6 | OpenSearch index settings that change recall (`hnsw_ef_construction` 512 to 128, deferred graph build): owner call after X-5 and WP-V numbers | decide on measured recall@10 |

## 13. Licences of recommended dependencies

The repository is AGPL-3.0-or-later. Everything recommended is already a dependency or is
permissively licensed.

| Component | Licence | Status here |
|---|---|---|
| NVIDIA Triton Inference Server | BSD-3-Clause [S44] | in use |
| TensorRT runtime | NVIDIA software licence (proprietary runtime, open-source parsers and plugins Apache-2.0) | in use inside the Triton image; not redistributed differently by any recommendation |
| NVIDIA DALI (Triton DALI backend) | Apache-2.0 | conditional (WP-2.4) |
| nvImageCodec, CV-CUDA | Apache-2.0 | candidates in WP-2.4 only |
| NVIDIA Model Optimizer (`nvidia-modelopt`) | Apache-2.0 [S19] | export-time only (#213) |
| ONNX Runtime | MIT [S38] | in use (text encoder) |
| ortloom / ortloom-serve | MIT OR Apache-2.0 | candidate, decided by X-6 |
| OpenSearch | Apache-2.0 | in use |

## 14. Sources

All accessed 2026-10-10. Access: F = page fetched and read; S = content taken from a search-result
summary of that page (re-check before quoting a number elsewhere). Repository-internal evidence is
cited inline by path.

| ID | Source | Access |
|---|---|---|
| S1 | NVIDIA RTX A6000 datasheet: https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/proviz-print-nvidia-rtx-a6000-datasheet-us-nvidia-1454980-r9-web%20(1).pdf | F |
| S2 | NVIDIA Ampere GA102 GPU architecture whitepaper v2.1 (Table 3): https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.1.pdf | F |
| S3 | NVIDIA Ada GPU architecture whitepaper (appendix table with the RTX 3080 Ti): https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf | F |
| S4 | TensorRT support matrix (latest, 11.4): https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html ; TensorRT 10.9 support matrix (FP8 from CC 8.9): https://docs.nvidia.com/deeplearning/tensorrt/10.9.0/getting-started/support-matrix.html | F |
| S5 | TensorRT 11.0.0 release notes: https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/release-notes-11/11.0.0.html ; TensorRT 10.x to 11.x migration guide: https://docs.nvidia.com/deeplearning/tensorrt/latest/api/migration/tensorrt-10x-to-11x.html | F (notes), S (guide) |
| S6 | Triton dynamic batcher: https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html | F |
| S7 | Triton `model_config.proto` (CUDA graphs, `graph_spec`): https://raw.githubusercontent.com/triton-inference-server/common/main/protobuf/model_config.proto ; issues https://github.com/triton-inference-server/server/issues/7150 , https://github.com/triton-inference-server/server/issues/5789 , https://github.com/triton-inference-server/server/issues/8171 | F (proto), S (issues) |
| S8 | Triton model configuration (instance groups, warmup, response cache, rate limiter, datatypes): https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_configuration.html | F |
| S9 | Triton release notes 26.06 and 26.09: https://docs.nvidia.com/deeplearning/triton-inference-server/release-notes/rel-26-06.html , https://docs.nvidia.com/deeplearning/triton-inference-server/release-notes/rel-26-09.html | F |
| S10 | BLS vs ensemble and Python backend reports: https://github.com/triton-inference-server/server/issues/4619 , https://github.com/triton-inference-server/server/issues/7214 , https://github.com/triton-inference-server/server/issues/8348 ; Python backend README (DLPack, `FORCE_CPU_ONLY_INPUT_TENSORS`): https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/python_backend/README.html | S |
| S11 | Sparsity on Ampere: https://developer.nvidia.com/blog/accelerating-inference-with-sparsity-using-ampere-and-tensorrt/ ; INT8 sparsity on A40: https://developer.nvidia.com/blog/sparsity-in-int8-training-workflow-and-best-practices-for-tensorrt-acceleration/ | S, F |
| S12 | NVIDIA NIM NV-CLIP: https://docs.nvidia.com/nim/nvclip/latest/getting-started.html | S |
| S13 | CUDA MPS: https://docs.nvidia.com/deploy/mps/when-to-use-mps.html ; Triton with MPS on AWS: https://aws.amazon.com/blogs/machine-learning/reduce-asr-inference-costs-by-75-with-nvidia-mps-on-amazon-ec2/ | F |
| S14 | Model Analyzer deprecation: https://docs.nvidia.com/deeplearning/triton-inference-server/archives/triton-inference-server-2591/user-guide/docs/model_analyzer/README.html ; perf_analyzer and GenAI-Perf status: https://github.com/triton-inference-server/perf_analyzer | S |
| S15 | TorchServe (archived): https://github.com/pytorch/serve | S |
| S16 | Triton optimization guide: https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/optimization.html | F |
| S17 | TensorRT performance best practices (CUDA graphs, timing cache, builder level, aux streams): https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/optimization.html | F |
| S18 | CLIP ViT-L/14 INT8 vs FP16 on an RTX A6000, TensorRT 10.9, Model Optimizer: https://github.com/NVIDIA/TensorRT-Model-Optimizer/issues/167 | F |
| S19 | pytorch-quantization deprecation: https://github.com/NVIDIA/TensorRT/blob/main/tools/pytorch-quantization/README.md ; NVIDIA Model Optimizer: https://github.com/NVIDIA/Model-Optimizer | F |
| S20 | Ultralytics YOLO11 (21.6 GFLOPs, T4 TensorRT latency): https://docs.ultralytics.com/models/yolo11 ; TensorRT INT8 export numbers: https://docs.ultralytics.com/integrations/tensorrt | F |
| S21 | open_clip model profile (ViT-L-14-336 381.92 GFLOPs): https://raw.githubusercontent.com/mlfoundations/open_clip/main/docs/model_profile.csv | F |
| S22 | timm benchmark results (inference img/s per GPU, PE-Core and CLIP ViT-L): https://github.com/huggingface/pytorch-image-models/tree/main/results | F |
| S23 | clip-retrieval distributed inference numbers: https://github.com/rom1504/clip-retrieval/blob/main/docs/distributed_clip_inference.md | F |
| S24 | nvJPEG documentation (hardware decode list, backends): https://docs.nvidia.com/cuda/nvjpeg/index.html ; A100 hardware JPEG decoder: https://developer.nvidia.com/blog/leveraging-hardware-jpeg-decoder-and-nvjpeg-on-a100/ | F |
| S25 | DALI `fn.decoders.image`: https://docs.nvidia.com/deeplearning/dali/user-guide/docs/operations/nvidia.dali.fn.decoders.image.html | F |
| S26 | nvImageCodec vs libjpeg-turbo on an RTX 3090: https://github.com/medcognetics/dicom-preprocessing/issues/133 | F |
| S27 | torchvision `decode_jpeg`: https://docs.pytorch.org/vision/main/generated/torchvision.io.decode_jpeg.html ; `Resize` antialias: https://docs.pytorch.org/vision/main/generated/torchvision.transforms.Resize.html | F |
| S28 | CV-CUDA throughput blog: https://developer.nvidia.com/blog/increasing-throughput-and-reducing-costs-for-computer-vision-with-cv-cuda/ ; resize speedup: https://github.com/CVCUDA/CV-CUDA/discussions/207 | F, S |
| S29 | Ultralytics DALI guide (GPU letterbox, Triton ensemble): https://docs.ultralytics.com/guides/nvidia-dali | F |
| S30 | NVIDIA NeMo Curator image curation: https://docs.nvidia.com/nemo/curator/latest/curate-images ; release notes 26.02: https://docs.nvidia.com/nemo/curator/v26.02/about/release-notes | F |
| S31 | Ray Data batch inference benchmark: https://docs.ray.io/en/latest/data/benchmark.html | F |
| S32 | OpenSearch k-NN tuning: https://opensearch.org/docs/1.3/search-plugins/knn/performance-tuning/ ; approximate k-NN and `approximate_threshold`: https://docs.opensearch.org/latest/vector-search/vector-search-techniques/approximate-knn/ ; remote GPU index build: https://docs.opensearch.org/latest/vector-search/remote-index-build/ | F |
| S33 | Milvus bulk import: https://milvus.io/docs/import-data.md | F |
| S34 | Triton shared memory measurements: https://github.com/triton-inference-server/server/issues/7126 , https://github.com/triton-inference-server/server/issues/6978 | F, S |
| S35 | Python gRPC client vs perf_analyzer latency: https://github.com/triton-inference-server/server/issues/6464 | S |
| S36 | Triton CUDA memory pool fallback: https://github.com/triton-inference-server/server/issues/6954 , https://github.com/triton-inference-server/server/issues/8177 | S |
| S37 | Triton TensorRT backend (execution policy, version-compatible engines): https://github.com/triton-inference-server/tensorrt_backend | F |
| S38 | ONNX Runtime TensorRT execution provider: https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html ; licence: https://github.com/microsoft/onnxruntime/blob/main/LICENSE | S |
| S39 | `nvidia-smi` manual (clocks, persistence): https://man.archlinux.org/man/nvidia-smi.1.en ; NUMA binding guidance: https://docs.coreweave.com/products/sunk/optimize_workloads/cpu-binding-numa-affinity | F |
| S40 | PTQ4ViT: https://github.com/hahnyuan/PTQ4ViT (paper https://arxiv.org/abs/2111.12293) | F |
| S41 | NVIDIA YOLOv7 QAT: https://github.com/NVIDIA-AI-IOT/yolo_deepstream/blob/main/yolov7_qat/README.md | F |
| S42 | Meta Perception Encoder: https://github.com/facebookresearch/perception_models/blob/main/apps/pe/README.md | F |
| S43 | DeepStream: https://github.com/nvidia/deepstream ; Holoscan SDK: https://github.com/nvidia-holoscan/holoscan-sdk | S |
| S44 | Triton licence: https://docs.nvidia.com/deeplearning/triton-inference-server/bsd/index.html | S |
| S45 | Rust `ort` crate: https://ort.pyke.io/ | S |
| S46 | LAION-5B paper: https://arxiv.org/abs/2210.08402 | S |
| S47 | Triton FAQ: https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/faq.html | F |
| S48 | SCRFD: https://github.com/deepinsight/insightface/tree/master/detection/scrfd | F |
| S49 | ArcFace (arcface_torch): https://github.com/deepinsight/insightface/blob/master/recognition/arcface_torch/README.md | S |
| S50 | Apple MobileCLIP: https://github.com/apple/ml-mobileclip | F |

Sources that decided the recommendation: S18 (INT8 and achievable FP16 fraction for ViT-L on this
exact GPU), S2/S1 (peaks), S22 (PE vs CLIP-L cost), S5 (TensorRT 11 typing and trtexec defaults),
S9 (CUDA shared memory known issue), S6/S16 (batching guidance), S24/S25 (no hardware JPEG engine
on GA102, hybrid Huffman on the CPU), S23/S31 (bulk embedding scales with GPUs until the host binds),
and the ortloom repository's own benchmarks (section 5.2).
