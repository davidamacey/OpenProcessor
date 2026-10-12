# 0.6.0 execution path: status and the order of the remaining work

Companion to [v060_wave_plan.md](v060_wave_plan.md) (the work packages and their gates). That plan says
what each package is; this file says where the milestone stands, in what order the rest is done, and the
working rules that keep it fast. Update the status table when a package merges or is closed.

The product goal behind every package: detect objects, embed the crops, then cluster and label like
crops so a dataset is labelled faster. Speed work is judged by images per second through that path
(detector, whole-frame and crop embeddings, storage) and by the time from ingest-complete to clusters
and labels being available.

## 1. Status

| Area | Package (issue) | State | Result |
|---|---|---|---|
| Measurement | Parity tooling, throughput suite (#208, #45) | merged | every change is gated; references captured twice |
| Precision | PE FP16 engine (#209 partly, #225) | merged | 13.7 to 5.5 ms per embedding |
| Precision | MobileCLIP FP16 (#224) | merged | 1.5x to 2.1x engine-only |
| Precision | INT8 PE (#213) | closed, not shipped | faster but fails the accuracy gates; weight-only gains nothing |
| Host CPU | API CPU attribution and fixes (#210, #211, #241) | merged | photos 5 to 25 images/s, no collapse past 32 clients |
| Host CPU | uint8 inputs, in-graph normalize (#214) | merged | Triton bytes per image -75%, API CPU per image -47% in total |
| Host CPU | Shared memory API to Triton (#215) | measured, not adopted | below 5% of API CPU |
| GPU | Data-parallel PE over several cards (#228) | merged | 2 x A6000 = 1.91x |
| GPU | PE engine pass, half-split rotary (#227) | merged | engine-only +24%, through Triton +24% |
| GPU | Cross-image batching (#216) | merged (embed pass only) | ingest is engine-bound and Triton already packs; embed pass 6.3 to 14 images/s |
| Services | OCR instances (#240), cluster scheduling (#212), serializer (#153), VLM idle (#207) | merged | |
| Ingest | Bulk ingest runner, CLI, route, Cropwright panel (#229) | merged | one runner; speed follows the data path |
| Storage | Vector store evaluation (#221) | merged | stay on tuned OpenSearch; follow-ups #254, #248, #230 |

## 2. Order of the remaining work

The order follows data dependencies: a stage is started when what it measures against is stable.

1. **Stage 1, design before build (running).** GPU decode, GPU resize and batching design (#217, plan
   file `gpu_decode_plan.md`). No code is built before the measurements and the parity/AP effect are in.
2. **Stage 2, tune the passes on the final engines, as one combined pass.** Detector engine pass
   (profile shapes, layer cost), Triton and client sweep (#226), embed-pass limiter (#259) and engine
   build stamps (#209). These are built and tested together (see section 4), not one long run each.
3. **Stage 3, build the data path from the Stage 1 plan.** GPU decode with a measured size gate;
   resize on the GPU in the order described in section 3; work crops (#219); faces and OCR on the same
   resident crops (#220), using the lessons below.
4. **Stage 4, storage and search.** Lucene engine and `ef_construction` 128 for new indexes after the
   product parity gate (#254), write coalescing to about 1000 documents per bulk (#248), write ceiling
   at scale (#230).
5. **Stage 5, prove it end to end.** A clean full-pipeline re-baseline with nothing else on the host;
   scale ladder 10k, 100k, 1M on public data (#232); then the ortloom head-to-head (#231), 12 GB card
   builds (#42), image slimming (#41). Small fixes ride along (#235, #236, #249).
6. **Clustering (#218) is deliberately after the ingest speed work.** See section 5.

## 3. GPU data path requirements (inputs to Stage 1 and Stage 3)

- **Decode once on the GPU and keep the pixels resident.** Models are fed from the largest required
  size down: full image, then the largest model input, then the next size down, in the order the
  models run (detector, PE whole-frame, PE crops cut from the resident full-resolution image using the
  detector boxes, then blur, thumbnails and the other consumers). The plan states which tensor each
  resize reads, the interpolation per step, and the measured parity and AP effect of resizing from an
  already-resized tensor instead of the original.
- **Zero host round-trips where possible,** and every remaining copy is counted.
- **Size gate.** Small images stay on the CPU path and large ones go to the GPU, at a cutoff measured
  on this hardware (the A6000 has no hardware JPEG engine, so GPU decode shares CUDA cores with the
  embedding model; the cost is quantified). Images the GPU decoder cannot handle fall back to the CPU
  explicitly and are counted.
- **Embedding batching.** Several crops per image go to Triton as properly sized batches. Measured: the
  dynamic batcher already packs crops from different images (mean batch 27.7 at 16 clients) and a
  client-side coalescer gave no gain, so batching settings (max batch, preferred sizes, queue delay,
  instances per GPU, profile opt shape) change only on measurement.

## 4. How the tests are kept short

- A change is proven cheap first (engine-only `trtexec`, unit tests, one parity capture). Features that
  have proven out are then combined into one stack and tested once, with one parity capture and one
  throughput run, instead of one long run per feature.
- Engine variants are built once and compared in one session on a quiet GPU.
- Every timed run records foreign GPU load, host load and clocks, is repeated three times and reports
  ranges; a rep with foreign GPU load is discarded.
- Parity gates use the reference captured for the current tree (`docs/benchmarks/parity_references.md`).

## 5. Clustering policy

The clustering methods (AHC and its refinement, and the other research methods) are not rewritten by
this milestone. Work on clustering is limited to scheduling, acceleration behind the agreement gate,
and measurement, and starts after the ingest speed work. Training keeps its full work.

## 6. Faces and OCR

They are not part of default ingest and are not dropped. They get the same treatment as the detector
and embedding passes: uint8 inputs, FP16 where the accuracy gates allow, instance counts sized on
measurement (OCR is already at two), and GPU-resident crops, so images can be searched and recalled
quickly later (#220).

## 7. Working rules

- One pull request merges at a time; the remote is checked after each merge.
- Each issue closes with its measured result, including negative results.
- Where packages overlap they are completed together.
- Nothing is shipped as dead code; a measured negative result is documented instead.
