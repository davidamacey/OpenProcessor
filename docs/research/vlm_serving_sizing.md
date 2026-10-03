# VLM serving sizing (vLLM)

Status: research snapshot (2026-05, vLLM with the default VLM); measured on RTX A6000 48 GB (VLM) with a segmenter on a second card and Triton CNN detectors on an RTX 3080 Ti 12 GB; numbers will drift.

This note explains how to size `--max-model-len` and concurrency for the in-compose VLM (the `vlm` service), which vLLM flags are worth sweeping, what a measured sweep looked like, and why the VLM, not the segmenter, was the pipeline bottleneck.

## Contents

1. Sizing `max_model_len` and concurrency from measured token usage
2. Flag recon: what to sweep and what to skip
3. Measured sweep
4. Finding: the VLM is the wall, not the segmenter
5. How to repeat

## 1. Sizing `max_model_len` and concurrency

The pipeline calls the VLM in two shapes:

1. Class labeling: a batch of up to 4 crops, each classified into one of roughly 80 registry classes.
2. Verification: a single crop, "is this a real small-region?" after a detector or segmenter proposed a box.

A large context window makes vLLM reserve KV cache for sequences that never use it, which caps concurrency. KV memory per sequence scales linearly with `max-model-len`, so right-sizing the context frees budget that can be re-spent on `max-num-seqs`.

### Measured token usage

Measured against a live vLLM endpoint with real crops (5472x3648 source images, the default VLM in bf16):

| Workload | Prompt tokens | Completion tokens | Total | Latency |
|---|---:|---:|---:|---:|
| Verify (1 image, ~150-token prompt) | 378 | 255 | **633** | 4.8 s |
| Label (4 images, 81-class registry) | 1306 | 529 | **1835** | 9.4 s |

- An image costs about 256-300 tokens per crop; the rest is system/user prompt and JSON output.
- The VLM is a reasoning model: the completion is mostly chain-of-thought followed by the JSON answer. The verify call's `max_tokens` had to be raised from 128 to 384, otherwise the model spends its budget on reasoning before it emits the JSON object.

### Sizing arithmetic

Budget headroom for registry growth (about 500 prompt tokens), richer structured output (about 500 completion tokens), and a 10 % buffer:

```
1835 (measured worst case) + 500 (registry) + 500 (output) = 2835 tokens
2835 x 1.1                                                 = 3120 tokens
```

Round to a power-of-two-friendly boundary: `--max-model-len 8192`. That is a 16x shrink versus a 131072 default, with roughly 2.6x margin over the 3120-token worst case.

The earlier deployment ran `--max-model-len 131072 --max-num-seqs 8`. The sizing recommendation was `--max-model-len 8192 --max-num-seqs 32`, which at the same VRAM budget gives:

- about 4x more concurrent sequences in flight;
- an expected throughput gain of about 4x on the verify path (the rate limiter for the async region worker drain), not measured in isolation;
- slightly better time-to-first-token under load because the scheduler has more sequences to interleave.

`--limit-mm-per-prompt` is a per-request image cap, not a concurrency knob; leave it matched to the client's images-per-call setting.

Why not 4096: it doubles concurrency again but leaves zero margin. If the registry grows to about 200 classes (registry text roughly doubles to about 600 tokens) or the system prompt grows, requests fail on `max_tokens` truncation. A later analysis with 6 images per prompt at a reduced vision-token budget (`max_soft_tokens=70`, about 9x expansion per soft token) gave roughly 3780 vision tokens + 600 prompt + 384 decode = about 4760 tokens, which confirms 4096 is too tight and suggests 5120-6144 as the practical floor for multi-image prompts.

### Quick check after changing flags

1. Restart vLLM with the new flags.
2. Replay a fixed payload and record prompt/completion token counts and latency (the vLLM `usage` field in the response gives both).
3. Watch `nvidia-smi`. VRAM should stay roughly flat; if it dropped a lot, raise `--max-num-seqs`.
4. Drain a fixed image set through the region worker and compare crops/s before and after.

## 2. Flag recon: what to sweep and what to skip

Reference config used as the baseline (the default VLM, AWQ INT4, single A6000, SM 8.6):

```
--dtype float16  --quantization awq
--max-model-len 8192  --gpu-memory-utilization 0.95
--async-scheduling  --enable-prefix-caching
--limit-mm-per-prompt '{"image":6,"audio":0}'
--mm-processor-kwargs '{"max_soft_tokens":70}'
--max-num-seqs 768  --max-num-batched-tokens 49152
--reasoning-parser <model-parser>  --enable-auto-tool-choice --tool-call-parser <model-parser>
```

The recon audited about 40 throughput-relevant knobs against the upstream vLLM docs (engine args, optimization guide, V1 user guide, quantization and multimodal pages). Only about 8 had been swept at the time. The items below are expectations from documentation, not measurements; verify each on your own hardware.

### Candidate knobs (expected impact, unmeasured)

| Knob | Why it might help | Expected effect |
|---|---|---|
| `--max-model-len` 8192 to 6144 | Caps per-request KV reservation, more sequences fit | 5-10 % |
| `--max-num-partial-prefills 4 --long-prefill-token-threshold 2048 --max-long-partial-prefills 2` | Image-heavy prompts are all long prefills; default of 1 serializes them | 10-20 % |
| `--quantization awq_marlin` (optionally `--dtype bfloat16`) | Faster INT4 GEMM on Ampere; may unlock bf16 | 5-15 %, may be rejected by the checkpoint validator |
| `--compilation-config '{"cudagraph_mm_encoder": true}'` | Same vision-encoder shape runs thousands of times per minute | 5-15 % encoder TTFT; longer startup, extra VRAM |
| Client `max_tokens` 1024 to 384 | Frees decode slots sooner; outputs rarely exceed about 200 tokens at low vision-token budgets | 5-10 %; watch parse-failure count |
| `--block-size 32` | Long uniform vision-token runs pack better | 2-5 % |
| `--max-num-batched-tokens` 49152 to 32768 or 16384 | Fewer prefills interrupting decodes, better inter-token latency | neutral to positive |
| `--mm-processor-cache-gb 0` | Crops are unique, so cache hit rate is about 0 and 4 GiB RAM per process is wasted | about 0 % throughput, saves RAM |
| Structured output backend (xgrammar) | Removes malformed-JSON parse failures by construction | 0-3 % |
| Fan-out concurrency about 0.5 x `max-num-seqs` | Leaves KV headroom | situational |

### Knobs to skip under vLLM V1

- `--num-scheduler-steps`, `--multi-step-stream-outputs`: V0-only; async scheduling replaces them.
- `--preemption-mode swap`, `--swap-space`: V1 removed GPU-to-CPU KV swapping; only recompute works.
- `--scheduling-policy priority`: no priority signal in a single-tenant pipeline.
- `--enable-chunked-prefill`, `--enable-prefix-caching`, `--async-scheduling`: default-on in V1; setting them explicitly is harmless.
- `--mm-processor-cache-type`, `--mm-encoder-tp-mode`: no-ops with a single process and TP=1.

### FP8 KV cache on Ampere (SM 8.6)

- `fp8_e4m3` cannot run on SM 8.6 (Inductor reports `type fp8e4nv not supported in this architecture`); that is a hardware floor.
- `fp8_e5m2` could run in principle, but an upstream validator gate rejected it for any quantized (AWQ/GPTQ/INT4) checkpoint regardless of whether the path would work (vLLM issue 39137).
- Recommendation: keep KV at fp16 until the gate fix ships in a tagged release; time-box any retry (about 10 minutes) and record the verbatim error.

### `awq` vs `awq_marlin`

`awq_marlin` is the documented faster INT4 path on Turing and newer and is the route default-VLM AWQ checkpoints take to unlock bf16. Validator regressions have been reported for some the VLM 4 AWQ checkpoints and vLLM tags, so capture the verbatim startup error if it fails and fall back to `awq`.

### Suggested compound candidate

For a single highest-information run against the baseline: `awq_marlin`, `max-model-len 6144`, `max-num-seqs 512`, `max-num-batched-tokens 32768`, the three partial-prefill flags above, `block-size 32`, `mm-processor-cache-gb 0`, and `cudagraph_mm_encoder`, with client `max_tokens=384`. Expected aggregate was 20-35 % (not additive) if `awq_marlin` loads, 12-22 % if not. Decompose stage by stage if you need to attribute a win.

## 3. Measured sweep

Method: reset the indexes, ingest a fixed set of images, let the full cascade drain (detectors, segmenter, VLM verification), and record per profile: wall time, images/s, vLLM end-to-end mean latency, VLM verify mean latency, segmenter in-flight mean, and Triton per-model mean latency and counts. Each profile restarts vLLM with different flags; `ready_in_s` is the vLLM startup time.

### Profiles (about 1000 images each)

| Profile | Description | ingest_wall_s | imgs/s | vllm_e2e_mean_s | vlm_verify_mean_s | seg_inflight_mean_s | det_a mean_ms | det_b mean_ms | region det mean_ms | ready_in_s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P0 | baseline (production flags) | 246.4 | 4.058 | 41.634 | 66.445 | 14.013 | 135.850 | 38.726 | 35.380 | 211.6 |
| P1c | compound with awq, no `cudagraph_mm_encoder` | 437.8 | 2.284 | 34.175 | 68.597 | 16.946 | 301.158 | 52.179 | 51.534 | 196.5 |
| P9 | baseline + `image:1` + ipc=1 + client conc 768 | 559.1 | 1.789 | 10.592 | 31.895 | 23.681 | 287.270 | 49.206 | 45.741 | 206.6 |
| P10 | baseline + client conc 768 | 455.5 | 2.195 | 36.764 | 72.103 | 19.089 | 140.242 | 38.937 | 35.890 | 206.5 |
| P11 | max-num-seqs 1024 + client 768 | 267.1 | 3.744 | 42.314 | 61.149 | 16.036 | 226.420 | 44.381 | 42.683 | 221.7 |
| P12 | max-num-seqs 1536 + client 1024 | 459.1 | 2.178 | 40.494 | 72.984 | 19.006 | 131.018 | 39.191 | 34.653 | 297.3 |
| P13 | baseline vLLM + ipc=6 (max batch) | 331.6 | 3.016 | 32.761 | 63.749 | 18.366 | 132.901 | 39.529 | 35.800 | 206.5 |
| P14 | max concurrency + ipc=6 | 358.2 | 2.792 | 40.008 | 69.979 | 19.663 | 292.514 | 53.405 | 44.157 | 247.0 |

Column notes: `det_a` is the custom vehicle detector, `det_b` the COCO YOLO11 detector, and the last latency column is the small-region detector, all served by Triton. Detector call counts were 994-996 per profile (small-region detector 2322-2375) with no errors.

### Scaling the baseline with image count

| Profile | N | ingest_wall_s | imgs/s | vllm_e2e_mean_s | vlm_verify_mean_s | seg_inflight_mean_s | ready_in_s |
|---|---:|---:|---:|---:|---:|---:|---:|
| N500 | 500 | 166.7 | 2.999 | 28.996 | 49.737 | 9.807 | 201.5 |
| N2000 | 2000 | 600.2 | 3.332 | 51.894 | 110.547 | 17.217 | 206.6 |
| N5000 | 5000 | 1143.6 | 4.372 | 36.563 | 70.715 | 14.314 | 206.6 |

### Reading the sweep

- No flag combination beat the production baseline (P0, 4.06 images/s) by a clear margin; most variants were slower. Throughput at the same flags varied 2.2 to 4.1 images/s between runs, so run-to-run noise is large. Repeat each profile before drawing conclusions.
- Raising client concurrency or `max-num-seqs` far past the point where KV cache is the cap (P10, P12) did not help and often hurt: vLLM only admits as many sequences as KV allows, so extra client fan-out just queues.
- VLM verify mean latency (30-110 s) dwarfs detector latency (tens to hundreds of ms) in every profile.
- Detector latencies rise (for example the custom detector from about 135 to about 290 ms) in profiles where the segmenter and VLM stages are busier, which indicates shared-GPU or client-side contention rather than a change in the detector itself.
- Throughput rose with N (3.0 to 4.4 images/s from 500 to 5000) as pipeline fill amortized.

## 4. Finding: the VLM is the wall, not the segmenter

Question: was the region worker's fan-out into the segmenter (96 client consumers into 64 segmenter instances) the real bottleneck, or the VLM?

Evidence:

- Per-stage telemetry showed the VLM stage at a p50 of about 46 s per crop, far larger than any segmenter or detector stage.
- Segmenter request overhead was about 50-150 ms of HTTP/JSON/base64 framing per crop (median crop 30-80 KB base64) on top of about 60 ms of inference, so roughly 110-160 ms per crop end to end.
- Sweeping segmenter instances and VLM knobs (section 3) moved throughput far less than VLM state did.

Conclusion: the VLM is the dominant wall-clock cost in the cascade; the segmenter is not the bottleneck. The two costs are additive and can be optimized independently, but improving the segmenter alone will not move end-to-end throughput.

Segmenter-side telemetry to confirm this on your own stack (decompose each segmenter call):

| Phase | Meaning |
|---|---|
| wait | client queue wait before the HTTP call starts (connection-pool wait) |
| inflight | HTTP round trip: network + server queue + GPU inference + response transfer |
| response | client-side JSON parse |

Decision rule:

- If wait p95 exceeds inflight p50 at high client concurrency and is notably smaller at lower concurrency, the client is over-subscribing: set concurrency equal to the segmenter instance count.
- If inflight p95 / inflight p50 is above 3 regardless of wait, the segmenter server queue is the limit: keep concurrency and add instances if VRAM allows.
- Otherwise leave the defaults.

Note that an HTTP client does not expose pre-connection wait as an event; wait is approximated as the time from function entry to the `.post()` call and is about 0 with a generously sized connection pool. Queueing then shows up inside inflight instead.

Proposal (not implemented): a batched segmenter endpoint accepting a list of crops would amortize five of the six per-request overheads (framing, JSON, headers, parse, base64 decode, serialization) for an estimated 3-5x reduction in framing overhead, saving roughly 50-100 ms per crop.

## 5. How to repeat

The in-compose VLM is the `vlm` service (profile `vlm`) in `docker-compose.yml`; its sizing knobs are environment variables in `env.template`:

```bash
# in .env
VLM_MAX_MODEL_LEN=8192
VLM_GPU_MEMORY_UTILIZATION=0.4
VLM_LIMIT_MM_IMAGES=8          # keep equal to OP_VLM_MAX_IMAGES_PER_CALL
VLM_EXTRA_ARGS="--max-num-seqs 32"   # any extra vLLM flag from section 2

docker compose --profile vlm up -d vlm
```

- Client fan-out is controlled by `OP_VLM_MAX_IMAGES_PER_CALL`, `OP_VLM_HTTPX_MAX_CONNECTIONS`, and `OP_REGION_WORKER_VLM_CONCURRENCY` / `OP_REGION_WORKER_VLM_VISIBLE_CONCURRENCY`.
- To measure token usage, send a representative request to the vLLM OpenAI-compatible endpoint and read `usage.prompt_tokens` and `usage.completion_tokens` from the response.
- The default `VLM_MAX_MODEL_LEN` and `max-num-seqs` in the public compose file differ from the values measured here; re-run the sizing arithmetic for your registry size and images per call.
- The per-profile restart-and-drain harness and the segmenter telemetry histograms described above are not part of the public repo; treat them as a proposal if you want to reproduce section 3 or section 4 end to end.
