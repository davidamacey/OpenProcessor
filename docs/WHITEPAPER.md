# OpenProcessor 0.7.0: a systems whitepaper

Status: written 2026-10-31 from the 0.7.0 measurement log, research notes and review documents, after the
0.7.0 release candidate was merged (#408, #409). Issue #412. This paper reports nothing new: every number
below is copied from the document cited next to it, which holds the method, the raw data and the caveats.
Where the source and this paper disagree, the source is right.

**Data rule.** Every figure quoted here was measured on public data: COCO, Open Images V7, the pinned
Wikimedia Commons high-resolution set ("set B"), CARS196, CUB-200-2011 and Stanford Online Products. Several
0.7.0 throughput rows were measured only on a private photo collection (called "set P" in the logs). Those
rows are left out, and the paper says where. Model memory figures (gigabytes a loaded engine holds on the
card) do not depend on any image set; they are quoted as such.

## Abstract

OpenProcessor is an open, domain-agnostic backend for computer-vision dataset curation and training. It
sits on a fast inference API served by NVIDIA Triton. Cropwright, a separate web frontend, renders what the
backend decides. Release 0.7.0 brought three main pieces of work, and this paper covers each. (1) **GPU
image preparation.** A decode-once Triton model, `gpu_image_prep`, lost throughput under load. The cause
turned out to be device-memory exhaustion: photos silently fell back to CPU decoding. It was not CUDA-call
overhead or the Python GIL, as first suspected. The fix is a memory budget reserved when the model loads,
counted fallback routes, and an opt-in load profile that frees 27.7 GB on a 48 GB card. (2) **The
clustering research program.** Eleven waves tested alternatives to the production IVF k-means partition
and the PE-Core-L14-336 embedding against bars registered in advance. None was adopted. The one lead,
detector-class roots with a hold-back guard, needs about a third of IVF's labelling decisions, but its
open-set bar could not be read. (3) **An auto-label loop.** It samples each cluster, asks a vision-language
model (VLM), and propagates a class only when an exact hypergeometric bound certifies that the cluster is
homogeneous. Machine labels stay separate from human ones and are never written over a human, validated or
held-out item. The loop's label quality with a real VLM was not measured in 0.7.0. A visual-judge study
found that a hosted small model's "same" verdicts are reliable on COCO crops, its "different" verdicts need
a person, and a small local VLM is near chance on pairs. The paper also covers data isolation and the
fail-closed rules, lists what was not established, and links the open issues.

## 1. System overview

Sources: [ARCHITECTURE.md](ARCHITECTURE.md), [VISION_AND_GOALS.md](VISION_AND_GOALS.md),
[design/curation_design_rationale.md](design/curation_design_rationale.md).

### 1.1 Inference API on Triton

A FastAPI process (port 4603) serves the inference routes `/detect`, `/faces`, `/embed`, `/search`,
`/ingest`, `/ocr`, `/analyze`, `/clusters`, `/query`, `/models` and `/health`, also mounted under `/v1`.
The models run as TensorRT engines in Triton with dynamic batching:

- detection: YOLO11 and YOLO26;
- faces: SCRFD and ArcFace;
- whole-image embeddings: MobileCLIP2-S2;
- OCR: PP-OCRv5;
- curation embeddings: the PE-Core image encoder, plus a CPU text encoder.

Each process shares one gRPC client pool, because opening a connection per request defeats Triton's
dynamic batching ([ARCHITECTURE.md](ARCHITECTURE.md), "Inference path"). OpenSearch is both the k-NN store
and the curation datastore.

### 1.2 Curation subsystem

The `/curation` surface is a generic active-learning stack. A domain is configured with data, not code:
a class registry, a region profile, a prompt pack and a VLM endpoint, each stored per project and revisioned.
The pipeline runs ingest, primary detection, a region stage (zero, one or many boxes per item), embedding
and clustering, human and AI review, import of labelled datasets, export, YOLO26 training, bake-off and
promotion back into serving ([VISION_AND_GOALS.md](VISION_AND_GOALS.md)). Long work runs in workers under
the `curation` compose profile. Release 0.7.0 adds a dedicated `curation-assist-loop-worker` for the
auto-label loop ([../CHANGELOG.md](../CHANGELOG.md), Unreleased).

### 1.3 Cropwright

Cropwright (`frontend/`, SvelteKit) is a pure frontend. It has no database and no offline mode, and every
datum it shows comes from the backend over one HTTP prefix
([../docs-site/docs/cropwright/getting-started/introduction.md](../docs-site/docs/cropwright/getting-started/introduction.md)).
The design rule is that the backend serves data and decisions and the frontend renders them. In 0.7.0,
cluster sorting, filtering, facets and the loop plan are all computed on the server.

## 2. The GPU image-prep and ingest pipeline

Sources: [research/v070_measurement_log.md](research/v070_measurement_log.md) (sections SP0, SP0.7, SPF,
E1, E4, F3), [design/sp_gpu_prep_concurrency.md](design/sp_gpu_prep_concurrency.md),
[reviews/sp0_review.md](reviews/sp0_review.md), [reviews/sp_fixes_review.md](reviews/sp_fixes_review.md),
[PERFORMANCE.md](PERFORMANCE.md) ("0.7.0 stage profile", "GPU memory and the ingest load profile").

### 2.1 Design

Photos of at least 3 MP take the GPU route ([research/v070_measurement_log.md](research/v070_measurement_log.md),
F1 and F5). The API sends the JPEG bytes to `gpu_image_prep`, a decoupled Triton Python-backend model. It
decodes once with nvJPEG, then reuses the decoded frame on the card for the detector letterbox, the
whole-frame embedding square, the crop squares for the PE encoder, the crop JPEGs and a blur score. Smaller
images, and photos the GPU decoder cannot take (progressive or truncated JPEGs), go through the CPU decode in
the API. Every route is counted in `op_ingest_decode_total`. Release 0.6.0 built this path; on set B
(12 to 24 MP Commons photos) it moved detect-plus-embed ingest from 27.0 to 35.0 images/s and host CPU per
photo from 1.12 to 0.34 CPU-s ([PERFORMANCE.md](PERFORMANCE.md), summary table).

Release 0.7.0 added stage timing inside the model. The Triton counters `op_gpu_prep_stage_seconds` and
`op_gpu_prep_stage_observations` cover the queue, decode, letterbox, detect, embedding, copy, JPEG and blur
stages. `STAGE_SYNC=1` attributes asynchronous kernel time during profiling runs
([../CHANGELOG.md](../CHANGELOG.md), Added). On set B the counters cost nothing measurable: the throughput
ratio with and without them was 0.978 at one batch and 0.977 at four batches, inside the run-to-run spread
([research/v070_measurement_log.md](research/v070_measurement_log.md), SP0.3 finding 5).

### 2.2 Stage profile on public data

One RTX A6000 ran Triton and the API on a shared host, with load average 10 to 59. The figures are therefore
stage shares, not release throughput ([PERFORMANCE.md](PERFORMANCE.md), "0.7.0 stage profile").

| Row | img/s median (min to max) | GPU util mean | Largest stages (share) |
|---|---|---|---|
| Set A (COCO, 1,846 measured), 1 batch, CPU route | 13.14 (13.12 to 13.47) | 45 % | PE crop embed 54.5 %, OpenSearch write 42.1 % of API stage time |
| Set B (400 Commons photos measured), 1 batch | 9.88 (9.63 to 9.96) | 15 % | decode 39.5 %, PE wait 22.3 %, detect 15.5 %, crop JPEG 12.0 % |
| Set B, 4 batches at once | 26.64 (25.21 to 29.00) | 49 % | decode 26.2 %, letterbox 17.9 %, PE wait 15.4 %, detect 8.9 % |

The first lesson concerned the driver. `run_baseline.py` used to post its batches one after another, so the
API never held more than 16 photos in flight. Posting four batches at once raised set B from 9.9 to
26.6 img/s ([research/v070_measurement_log.md](research/v070_measurement_log.md), SP0.3 finding 1).
One-batch rows describe latency and must not be quoted as throughput. On the CPU route (set A) the PE crop
embedding comes first and OpenSearch writes second (SP0.3 finding 4).

### 2.3 The GPU-memory-exhaustion finding

The first reading of the SP0 profile was that throughput fell past about four concurrent batches because
of host-side CUDA-call overhead. The image-handling stages grew 11 to 33 times for the same work
([research/v070_measurement_log.md](research/v070_measurement_log.md), SP0.3 finding 2). The design review
re-read the per-repetition route counters and found something else
([design/sp_gpu_prep_concurrency.md](design/sp_gpu_prep_concurrency.md), section 1):

- Every failed call was an exception raised inside the model, and only at the stages that allocate GPU
  memory (letterbox, crop copy, frame and crop embedding squares), never in detect or the PE call. The
  error strings were CuPy and nvJPEG out-of-memory errors (log SP0.7.1).
- Failed calls fell back silently to the API's CPU decode. API CPU per image tracked the fallback fraction
  almost exactly, and the fallback share grew with the time since the stack started, not only with load.
- On set B, freshly started stacks fell back 0 to 2.8 percent at four batches
  ([design/sp_gpu_prep_concurrency.md](design/sp_gpu_prep_concurrency.md), section 1, item 4).

**Where the memory goes.** These figures are measured model residency, independent of any image set. With
the default load list, Triton held 36.5 GB of a 48 GB A6000 before any photo arrived. Unloading the models
ingest never calls freed 27.7 GB
([research/v070_measurement_log.md](research/v070_measurement_log.md), SPF.3):

| Model | Delta on the card |
|---|---|
| `scrfd_10g_bnkps` | 6,846 MiB |
| `arcface_w600k_r50` | 2,754 MiB |
| `mobileclip2_s2_image_encoder` | 3,454 MiB |
| `mobileclip2_s2_text_encoder` | 1,430 MiB |
| `paddleocr_det_trt` | 1,642 MiB |
| `paddleocr_rec_trt` (dynamic-shape engine) | 11,572 MiB |
| total | 27,698 MiB |

The three models ingest needs (the detector and the PE encoders) take 8.8 GB. One `gpu_image_prep`
instance peaks at about 8.5 to 9.1 GB under load (log SP0.7.1). Two instances next to the default list do
not fit the card.

On public set B with every model loaded, adding instances made things worse by exactly the fallback
fraction. Four and eight batches gave the following (log SP0.7.7):

| Instances | img/s, 4 / 8 batches | Fallbacks |
|---|---|---|
| 1 | 28.8 / 34.0 | 0 |
| 2 | 32.4 / 33.5 | 1.7 to 10.8 % |
| 3 | 23.1 / 23.0 | 29 to 76 % |

The log calls these "fallback-confounded" numbers, and the set B matrix was stopped there. The tests that
separated the hypotheses (headroom against a full card, instance count, in-flight limits, thread sampling of
the Triton Python stubs) ran on set P, so their throughput is not quoted. Their conclusions do not depend on
the image set: memory exhaustion was confirmed. No thread was saturated, so the GIL was not the cause. API
CPU was flat when nothing fell back. A third instance added nothing (log SP0.7.8).

**What shipped** ([../CHANGELOG.md](../CHANGELOG.md), Unreleased; [reviews/sp_fixes_review.md](reviews/sp_fixes_review.md)):

1. A device-memory budget inside `gpu_image_prep` (#392). Each photo reserves its decoded frame from
   `DEVICE_BUDGET_MIB` (default 2048) and waits at most `BUDGET_WAIT_SECONDS` (20). The budget is reserved
   when the model loads, so an instance that cannot hold it fails to load instead of degrading later. A
   failed allocation, a timed-out wait and an oversized photo are now counted routes (`cpu_gpu_memory`,
   `cpu_over_budget`). On a full card the budget does not remove the fallbacks; it names them. The
   reservation covers decoded frames, not nvJPEG buffers (log SPF.1).
2. An opt-in ingest load profile (#393). `OP_LOAD_PROFILE=ingest` preloads only the detector and the PE
   encoders; the face, MobileCLIP and OCR models load on their first call, so every endpoint still answers.
   The endpoint check called 38 core routes with three public COCO images on both load lists. Status codes
   matched, with one exception caused by the full card: `/ocr/predict` failed on the default list with two
   instances and answered on the profile. On the profile, the first call cost 2.3 s for `/embed/text` and
   17.8 s for `/analyze` (five models) (log SPF.3).
3. The `full` installer profile ships one `gpu_image_prep` instance. Two instances are documented as an
   option together with the ingest profile ([reviews/rc_summary.md](reviews/rc_summary.md), "Owner
   decisions applied").
4. `/health` and `/ready` answer 503 when a required Triton model failed to load (#391). Before this fix, a
   missing PE engine left `/health` at `ready`.
5. The defaults inside the model moved from 32 workers and 96 in flight to 16 workers and 32 in flight
   (log SPF.2).

**Parity.** On 200 public images (set A and set B), the old and new models were compared using direct
captures (`scripts/bench/gpu_prep_parity.py`). At concurrency 1, old against old differed in 8 of 193
images and old against new in 11. In each of those images the detector boxes differed (the detector's
run-to-run variation). Where the boxes were equal, every output was byte-identical, and whole-frame
embedding cosine was at least 0.99999999 (log SPF.1). An earlier check on 200 public COCO images
compared one instance with two. Differences stayed at the level of the run-to-run noise between two
single-instance runs: whole-frame embedding minimum cosine was 0.99992 in both comparisons, and crop
embedding minimum cosine was 0.9885 (one against one) and 0.9905 (one against two) (log E4).

**Left out.** The acceptance throughput of the shipped configurations was measured only on set P. That
covers the ingest profile with two instances, the default list with one instance, and the 24 GiB card
emulation. [PERFORMANCE.md](PERFORMANCE.md) shows those rows labelled as private; this paper does not
repeat them. The public set A and set B rows were not re-run after the memory budget
([reviews/rc_summary.md](reviews/rc_summary.md), "What was NOT verified live").

### 2.4 The embed reprocess pass and vector consistency

Release 0.7.0 changed how stored vectors are recomputed (#259, #360). Crops that touch the right or bottom
image edge used to be embedded from a zero-padded crop. Every route now embeds the stored, image-clamped
box. On 1,500 public COCO images (9,666 crops, 1,193 of them on an edge), ingest and the reprocess pass now
agree on edge crops with minimum cosine 0.9993. The 0.6.0 documentation recorded 0.93 to 0.97. Interior
crops of old and new ingest are identical
([research/v070_measurement_log.md](research/v070_measurement_log.md), E1).

On the same COCO set the pass ran at 16.58 img/s against 15.68 on 0.6.0, but used more API CPU (0.066
against 0.053 CPU-s per image). The work moved to pool threads; it did not shrink. The #259 acceptance bar
(at least 20 img/s, at most 0.047 CPU-s per image) is **not met** on COCO (log E1). COCO images are under
the 3 MP gate, so all of them stay on the CPU route; a later run reproduced 16.2 img/s (log F3). The GPU
reprocess route and the region-vector parity were measured on set P and are not quoted.

### 2.5 Ingest tuning lessons carried forward

Two older research notes give method lessons only. Their numbers come from a non-public detector and
non-public data, so none are quoted:

- Small client batches at high concurrency feed Triton's dynamic batcher better than large client batches.
  Per-model queue-to-compute ratios in Triton's metrics show the real bottleneck
  ([research/ingest_concurrency_tuning.md](research/ingest_concurrency_tuning.md)).
- A model with a very large raw output can lose throughput as batch size grows at low concurrency, because
  the response payload dominates. A batch sweep at fixed concurrency catches this before deployment
  ([research/triton_perf_crossvalidation.md](research/triton_perf_crossvalidation.md)).

## 3. The embedding choice: PE-Core-L

Sources: [research/wb2_standard_benchmarks.md](research/wb2_standard_benchmarks.md) (section 9),
[research/labelled_benchmarks.md](research/labelled_benchmarks.md),
[research/cluster_program_outcome.md](research/cluster_program_outcome.md) (sections 4 and 5).

Production embeds every crop and frame with PE-Core-L14-336 (1,024 dimensions, an FP16 TensorRT engine).
An earlier wave (WB) ran on a Commons car set whose make and model labels came from category names and were
only machine-reviewed. On that set, smaller encoders ranked ahead of PE-Core-L. Because the labels were weak,
that result is provisional and its numbers are not quoted here
([research/cluster_program_outcome.md](research/cluster_program_outcome.md), section 5). Wave WB2 re-ran the
question on three benchmarks labelled by their authors, with bars registered before scoring.

**Protocol.** Whole images, zero-shot deep-metric-learning splits: CARS196 test classes (8,131 images), CUB
classes 101 to 200 (5,924), and the SOP test split (60,502). Recall@K uses the query-excluded convention.
k-means NMI is computed at the true number of classes. Five encoders were each run with centre-crop and
letterbox preprocessing; the variant was chosen on calibration classes
([research/wb2_standard_benchmarks.md](research/wb2_standard_benchmarks.md), sections 1 to 3).

| Encoder (variant) | CARS196 R@1 | CUB R@1 | SOP R@1 | Encode img/s | Peak MiB | Dim |
|---|---|---|---|---|---|---|
| PE-Core-L (letterbox), production | 0.961 | 0.867 | 0.796 | 133 | 1,835 | 1,024 |
| PE-Core-G (letterbox) | 0.969 | 0.907 | 0.820 | 17 | 9,170 | 1,280 |
| MobileCLIP2-S2 (centre) | 0.964 | 0.752 | 0.674 | 815 | 629 | 512 |
| SigLIP 2 (letterbox) | 0.975 | 0.744 | 0.790 | 314 | 1,982 | 1,152 |
| DINOv2-L (centre) | 0.825 | 0.892 | 0.576 | 373 | 1,451 | 1,024 |

Recall from section 9.2, cost from section 9.5. Cost was measured encoder-only at batch 32 in FP16 on a
shared GPU, so it is indicative.

**Decision.** No challenger passed bar B1, a registered Recall@1 gain over PE-Core-L. PE-Core-G is better
on all three sets (+0.008 on CARS196, +0.040 on CUB, +0.024 on SOP), but its CARS196 gain is confidently
below the 0.03 margin, and it costs about eight times the encode time and five times the memory.
MobileCLIP2-S2 is 0.115 below PE-Core-L on CUB and 0.122 below on SOP. DINOv2-L has no text tower, and the
product needs one for text search (bar B4). Letterbox preprocessing gave no gain of 0.03 on CARS196 for any
encoder (bar B2), so it is not a lever for whole images (section 9.3). PE-Core-L stays.

**Caveats the note itself records** (sections 9.1 and 9.6):

- The registered sanity check was not met. CARS196 Recall@1 for four models was above the 0.95 plausibility
  ceiling, and the shuffled-label NMI bar had been set below its chance floor. Every WB2 comparison is
  therefore labelled exploratory.
- CARS196 is near its ceiling, and contamination from pre-training data cannot be ruled out.
- All three sets are whole images, not detector crops of street photos, and text-search quality was not
  measured.

**Grouping on top of the embedding.** Average linkage with a cut chosen on calibration classes keeps test
BCubed precision at 0.90 or more on 2 of 3 sets for PE-Core-L (it fails on CUB at 0.84 to 0.87) and on 3 of
3 for PE-Core-G, at pair recall 0.05 to 0.38. At the true number of clusters it is clearly worse than
k-means (section 9.3, bar B3). The program-level reading: fine-grained grouping (make and model) is an
embedding problem more than a clustering problem
([research/cluster_program_outcome.md](research/cluster_program_outcome.md), section 1).

Face sets CALFW and CPLFW were downloaded and indexed but not scored, so no face-embedding figure exists for
0.7.0 ([research/labelled_benchmarks.md](research/labelled_benchmarks.md), "Status of downloads").

## 4. The clustering program and its outcome

Source: [research/cluster_program_outcome.md](research/cluster_program_outcome.md) and the wave notes it
links. All data are public COCO and Open Images crops. Bars were committed before the scores in W0e, Z4, WB
and WB2, and none was moved afterwards (outcome, section 2).

### 4.1 The incumbent

Production clustering is IVF: spherical k-means with K = min(512, n // 6) centres over the PE-Core-L
vectors, then nearest-centre assignment. On the 3,000-item public set (944 images, 3,961 crops, 2,950
labelled in 86 classes) its clusters are mostly right but numerous:

| Metric | Value |
|---|---|
| Held-out pair precision | 0.716 |
| BCubed precision | 0.784 |
| Pair recall | 0.027 |
| BCubed recall | 0.186 |

(outcome, section 4; [research/cluster_eval_baseline.md](research/cluster_eval_baseline.md)). Average
linkage reaches higher recall only by giving up precision. At pair recall 0.126 its held-out precision is
0.330, against IVF's 0.734 at 0.029 ([research/cluster_config_baseline.md](research/cluster_config_baseline.md)).

### 4.2 The one product change: one k-means restart (#337)

The IVF fit ran five k-means restarts. On the public COCO set (3,961 crops, 2,950 labelled, 7 seeds), one
restart passed an agreement gate. Fit time is on a host at load about 100
([reviews/sp1_review.md](reviews/sp1_review.md)):

| Restarts | Seed-mean purity | Seed-mean false-merge | ARI with incumbent | Fit s |
|---|---|---|---|---|
| 5 (incumbent) | 0.7963 | 0.3009 | 1 | 8.1 |
| 1 (shipped) | 0.7962 | 0.3008 | 0.331 (seed-to-seed floor 0.311 to 0.344) | 2.1 |

Restarts do not improve the partition more than a change of seed does. The gate is evaluated on seed means,
because two seeds of the five-restart fit already differ by 2 to 3 points. The review says plainly that
3,961 crops is a small labelled set for 512 clusters. The headline speed-up quoted in the changelog
(91,184 crops) and the whole-pass timings per crop were measured on set P and on synthetic vectors, so they
are left out here. The same review made a related fix: a cluster pass that read less than 99 percent of
its pool now reports `partial` instead of `success` (#357).

### 4.3 Decision per arm

From the outcome note, section 3. "Parked" means the arm may return with new evidence; "killed" means a
registered bar failed and no follow-up is planned.

| Arm | Decision | Evidence (public sets) |
|---|---|---|
| IVF k-means | Stays | Cheapest; best on `rare` items |
| Average linkage inside the detector class | Parked | Fresh set: BCubed recall 0.288 against bar 0.329; per-class cut moves between sets |
| Label-free per-class cut rule | Killed | Fails recall bars on the held-out half and the fresh set |
| Whitening, k-reciprocal re-ranking, DBA | Killed | Precision gain at matched recall negative (-0.065 to -0.011) |
| Novelty flag by distance | Killed | AUROC 0.838 and 0.851 against bar 0.90 |
| Judge-gated merge of small neighbours (Z3) | Killed | Adds 2.2 % effort instead of saving 10 % |
| Split by descent | Killed | 71 % and 9 % of impure clusters made pure; bar 80 % |
| Within-class roots for the loop | Killed | 0.526 decisions per item against limit 0.417 |
| Detector-class roots (with or without guard) | Parked as the lead | Passes precision and effort in Z4; open-set bar unreadable |
| Leiden CPM, CHAMELEON-style filter | Parked (#342 to #346) | Real-crop gain only with a filter tuned on the same sample |
| Consensus Leiden, HDBSCAN leaf, FINCH | Killed | 5x time for equal quality; about 80 % noise; no granularity control |
| GPU Leiden/ECG (cuGraph) | Parked | One 20k synthetic run only |
| Another encoder | Not adopted | Section 3 |
| Visual judge (hosted) | Research tooling only | Section 5.3 |
| Local VLM judge | Killed as an authority | Pair accuracy 0.51 |

### 4.4 The lead: detector-class roots and the hold-back guard

The auto-label loop needs a first grouping, its "roots". Wave W0e simulated the loop's gate with an oracle
labeller on a vehicle-heavy Open Images validation set (33,580 crops, 20 seeds)
([research/cluster_config_loop_cost.md](research/cluster_config_loop_cost.md); outcome, section 6):

| Roots | Decisions per item | Propagated precision |
|---|---|---|
| IVF (incumbent) | 0.776 | 0.988 |
| One cluster per detector class | 0.374 | 0.989 |
| Average linkage inside the class | 0.326 | 0.996 |

Z4 repeated the comparison on an Open Images test-split set built to hold open-set animals the detector
cannot name. It had 7,209 labelled held-out crops, 395 of them open-set
([research/cluster_config_z4.md](research/cluster_config_z4.md)):

| Roots | Decisions per item | Precision | Open-set items wrongly propagated per run |
|---|---|---|---|
| `key` (class roots) | 0.293 | 0.982 | 9.05 (0 to 72 by seed) |
| `key_guard` (plus hold-back) | 0.377 | 0.988 | 0.25 |
| `within_key` | 0.526 | 0.990 | 0.95 |
| IVF | 0.834 | 0.979 | not comparable |

**Verdict: inconclusive, IVF stays.** The registered rule needed at least 30 open-set items propagated per
run to read the open-set bar. Only 2 of 20 seeds reached that, because the whole test split could not
supply the registered 2,000 open-set crops. The direction is clear, though. Class roots cost 0.35 times
IVF's effort, and the guard adds 29 percent effort while removing 97 percent of the open-set exposure
(outcome, section 6). Z5 would repeat the test on the Open Images train split with at least 1,200 open-set
crops before any integration (#398). Inside a single detector class such as `car`, the class gives no
help, so the lead does not carry over to make and model
([research/cluster_config_decision.md](research/cluster_config_decision.md), 0A.3).

### 4.5 Graph refinement (parked)

The Leiden prototype (#342) ran on 539 real labelled COCO crops
([research/leiden_refinement_prototype.md](research/leiden_refinement_prototype.md); outcome, section 7).
The unfiltered pass traded precision for recall. Only a stricter CHAMELEON-style filter, tuned on the same
sample, beat the baseline on both axes: precision 0.944 and recall 0.195, against 0.905 and 0.088. The
authors call it a conditional go with a weak case. The method looks strong on synthetic planted partitions,
but the program's rule is that synthetic structure does not count as quality evidence, so those figures are
not quoted. Calibrating the resolution and filter floors per embedding distribution is the open problem.
GPU runs reached only 20k synthetic points, and 1M was not run.

## 5. The auto-label loop and the judge findings

Sources: [design/auto_label_loop_plan.md](design/auto_label_loop_plan.md),
[../docs-site/docs/guides/auto-label-loop.mdx](../docs-site/docs/guides/auto-label-loop.mdx),
[reviews/l1_review.md](reviews/l1_review.md), [research/visual_judge_e0.md](research/visual_judge_e0.md).

### 5.1 What the loop does (#286)

`POST /curation/projects/{project}/assist/loops` returns a dry-run plan by default. With `dry_run: false`
it queues a loop for the `curation-assist-loop-worker`. For each pinned cluster (a root node), the loop:

1. draws a seeded sample of the unlocked members;
2. asks the project's VLM through the assist-run core;
3. propagates the top class to the node's other free members when a statistical gate passes;
4. otherwise splits the node by embedding and recurses on each part.

Anything still doubtful is left for review with its votes. The loop stops on depth, minimum node size,
budget (VLM crops, requests, wall time), cancel, or a failing audit
([design/auto_label_loop_plan.md](design/auto_label_loop_plan.md), section 0).

### 5.2 The gate

The existing assist gate ("3 of 3 agree") shows homogeneity above only 0.37 at 95 percent confidence. The
loop instead uses a one-sided exact hypergeometric bound. A node with N eligible members passes for class c
only if, were the share of members the VLM would call c just below theta, seeing at least the observed
number of agreeing answers would have probability at most alpha. With no dissent and large N this reduces to
theta^n <= alpha. Required sample sizes are computed exactly, not measured
([design/auto_label_loop_plan.md](design/auto_label_loop_plan.md), section 6):

| theta | alpha | 0 dissent | 1 dissent | 2 dissent |
|---|---|---|---|---|
| 0.80 | 0.05 | 14 | 22 | 30 |
| 0.90 | 0.05 | 29 | 46 | 61 |
| 0.95 | 0.05 | 59 | 93 | 124 |

At the defaults, a 300-member cluster needs 32 clean answers and a 10,000-member cluster needs 35
([../docs-site/docs/guides/auto-label-loop.mdx](../docs-site/docs/guides/auto-label-loop.mdx)). The plan
also records what the bound does not prove. It certifies agreement with the VLM, not correctness, so a
systematic confusion between two similar classes passes it. That is why the loop has a validated-member
veto and a second-model audit: after each batch of propagated items it re-asks a sample and pauses when
the Wilson lower bound of agreement drops below a floor (default 0.85) (plan, sections 6 and 7).

### 5.3 Safety of what it writes

- Propagated items carry `class_source: "assist_propagated"`. They are not validated, not locked, and not
  exported for training until a person reviews them. A loop never impersonates a human label (plan,
  decision D1).
- The loop never writes a human or validated class, an excluded item, or an item in the frozen test
  holdout. Every write goes through optimistic concurrency, and a write that races a person loses.
- Budgets are reserved before each request, so a crash can waste what was in flight but never exceed a
  cap.
- `POST /curation/projects/{project}/assist/loops/{loop_id}/rollback` restores items only where the state
  still equals what the loop wrote, and keeps anything changed since.
- A running loop holds the cluster retrain: `409 assist_loop_running`
  ([../CHANGELOG.md](../CHANGELOG.md), Unreleased).

The live test drove the harness with a fake VLM ([reviews/l1_review.md](reviews/l1_review.md)). Budgets
held. Locked and held-out items stayed unchanged through the loop and the rollback. In one case a person
edited a propagated item before rollback, and rollback restored 14 items and kept the edited one. A cluster
whose answers dissented was split and then left for review. These are behaviour checks: the fake answers
by rule.

**Not measured.** The loop has not run with a real VLM on a real project, so the precision, coverage and
cost of its propagated labels are unknown ([reviews/rc_summary.md](reviews/rc_summary.md), "What was NOT
verified live").

### 5.4 Visual judges on contact sheets

Wave W8 built a contact-sheet harness (`src/services/labeling/contact_sheet.py`,
`scripts/research/visual_judge.py`). It measured whether a VLM can judge cluster purity from one image
grid. Hosted judges are refused for face, identity-text and plate sheets before any call
([research/visual_judge_e0.md](research/visual_judge_e0.md)). The hosted judge was Claude Haiku, run on 55
cluster sheets and 12 pair sheets per set:

| Haiku | COCO crops | Open Images vehicles |
|---|---|---|
| Pair kappa / flag precision (bars 0.7 / 0.8) | 0.821 / 0.827 (pass) | 0.611 / 0.454 (fail) |
| Clean sheets wrongly rejected | 0.217 | 0.765 |
| Contaminated sheets accepted | 0.094 | 0.000 |
| Purity estimate vs truth | 0.886 vs 0.888 | 0.763 vs 0.890 |

Haiku's "same" verdict is trustworthy, but its "different" needs a person. On vehicle types it rejected 13
of 17 clean sheets, partly because car, van and truck overlap as labels. The small local model
(Qwen3-VL-4B, served by the stack's vLLM) scored near chance on pairs: accuracy 0.51 and kappa 0.031 on COCO
crops. It over-flagged cells, estimating purity at 0.62 against a true 0.91, so it may rank clusters for
review but carries no weight as a verdict. Layout calibration on the vehicle set was vacuous, because both
judges said "different" for every sheet, so no default layout was chosen. A second local catalog model
returned truncated replies because of a 600-token output cap. The local default is now 2,048 tokens, and
that model still needs a rerun. The 5 percent human re-judge has not been done (#406).

## 6. Data isolation and safety

Sources: [ARCHITECTURE.md](ARCHITECTURE.md) ("Projects and isolation", "Concurrency and the lock rule"),
[VISION_AND_GOALS.md](VISION_AND_GOALS.md), [../CHANGELOG.md](../CHANGELOG.md).

**Projects and the guard.** A project owns its OpenSearch indexes (`{prefix}{slug}__{role}`), its
directories, its class registry, its MLflow experiment and its promoted models. Names are computed when the
project is created and stored in its record, so a later env change never remaps a live project. A
transport-level guard (`src/services/projects/guard.py`) wraps every OpenSearch client. It allows only the
request shapes the code sends, aimed at the bound project's indexes. Wildcards, `_all`, aliases,
`_reindex`, index-less searches and any request that names another project's index raise
`CrossProjectAccess`. The loop's unit and route tests include cross-project isolation tests
([reviews/rc_summary.md](reviews/rc_summary.md)).

**The lock rule.** Ingest, the workers, auto-label, clustering, import, reprocess and the new loop all write
through optimistic concurrency: read with the sequence number, write conditionally, re-merge on conflict.
No automated writer changes a class a human set or confirmed, a validated imported label, an item in the
test holdout, or a box a human created, judged or transcribed (`src/clients/occ_locks.py`). The 0.7.0
label-operation undo (`POST /curation/projects/{project}/label_ops/{op_id}/undo`) obeys the same rule. It
is a dry run by default, it undoes item by item, and an item that changed later, was excluded, or sits in
the holdout is kept ([design/cq3_label_operations.md](design/cq3_label_operations.md);
[reviews/cq3_review.md](reviews/cq3_review.md)).

**Class identity by name.** Classes cross every boundary by name, never by raw index: import, combine,
export, train, promotion and model sharing. A dataset's class order is never trusted
([VISION_AND_GOALS.md](VISION_AND_GOALS.md)).

**Fail closed.** Several 0.7.0 changes replace a silent degradation with a visible refusal or a counted
state:

| Change | Behaviour now |
|---|---|
| Required Triton model failed to load (#391) | `/health` and `/ready` answer 503 and name the model |
| `gpu_image_prep` cannot hold its budget (#392) | The instance fails at load, and ingest falls back visibly |
| GPU memory fallbacks (#392) | Counted as `cpu_gpu_memory` and `cpu_over_budget` |
| Cluster pass that missed part of its pool (#357) | Reports `partial` and records no last run; an auto-label job fails on it |
| Region profile with a detector that has no `output0` (#362) | Activation refused with `detector_model_unusable` |
| Retired env vars, including `OP_BACKBONE_EMBEDDING_DIM` (#410) | Rejected at startup |

Sources: [../CHANGELOG.md](../CHANGELOG.md), Unreleased.

**Privacy in research tooling.** Hosted judges refuse face, identity-text and plate sheets
([research/visual_judge_e0.md](research/visual_judge_e0.md)). A remote audit VLM for the loop needs the
deployment's external policy, the endpoint's `allow_external` flag and an explicit acknowledgement in the
request ([design/auto_label_loop_plan.md](design/auto_label_loop_plan.md), section 7). The API itself has no
authentication and must not be exposed ([../SECURITY.md](../SECURITY.md)).

## 7. Limitations and what was not established

- **Host noise.** Every 0.7.0 measurement ran on a shared host, with load averages from about 5 to over 100.
  Only comparisons within one table and one session hold; absolute rates are not release figures
  ([research/v070_measurement_log.md](research/v070_measurement_log.md), SP0.5 and SPF;
  [research/cluster_program_outcome.md](research/cluster_program_outcome.md), section 10).
- **One card type.** Ingest numbers come from one RTX A6000. The 24 GiB case was emulated with a memory
  ballast, not run on a real 24 GiB card. CUDA MPS and batched kernels were not measured
  ([reviews/rc_summary.md](reviews/rc_summary.md)).
- **Public throughput after the fix.** Set A and set B were not re-run with the memory budget and the load
  profile (rc summary).
- **The auto-label loop.** Label quality with a real VLM is unmeasured. The Cropwright loop panel was
  tested against a stub server only (rc summary).
- **Detector-class roots.** The open-set bar is unreadable on the available test split (Z5 not run).
- **Embedding.** WB2's sanity check was not met. Every comparison is exploratory, the inputs are whole
  images rather than detector crops, and text-search quality was not measured.
- **Make and model.** No verdict rests on person-checked labels, street photos or the 13,301-row scale.
- **Faces.** No face clustering experiment was run, and CALFW and CPLFW were not scored.
- **Scale.** No real-crop cluster pass at 1M. GPU graph methods above 20k points were not run. The cluster
  ladder above 100k rows used synthetic vectors (timing only).
- **Embed pass bar.** The #259 bar is not met on COCO.
- **Text-search latency.** The first-search cost per worker (#356) was profiled to the index bootstrap
  and to lazy route-state building in FastAPI. Both now run at API worker start
  ([../CHANGELOG.md](../CHANGELOG.md), Unreleased), but every latency figure behind the issue came from a
  non-public project. No public-data latency after the change is recorded, and #356 stays open.
- **Judges.** No 5 percent human re-judge, a second local model not scored, no default sheet layout.

## 8. Future work

Each item is an open issue in the tracker.

| Issue | Work |
|---|---|
| #398 | Z5: detector-class roots with a hold-back guard as the loop partition, on the Open Images train split |
| #342 to #346 | Graph-based refinement (Leiden, constrained split and merge, refresh-worker refinement, GPU methods) |
| #406 | Finish the judge calibration, the human re-judge and the label check |
| #407 | Staged `gpu_image_prep` speed ideas: batched kernels, a DALI front end, a native backend |
| #403 | Optional CUDA MPS overlay, adopted only on a measured gain of at least 10 percent |
| #40 | Epic: keep decoded frames on the GPU across models |
| #219 | Work crops at model size for the VLM and region consumers |
| #220 | Faces and OCR on resident GPU frames |
| #268 | Stage-2 follow-ups: set B photos, minimal and standard profiles, two-card sweeps |
| #42, #41 | Smaller-workspace engine builds and 12 GB guidance; slimmer images |
| #45 | Public benchmark ladders: Open Images subsets, full COCO, 1M images |
| #356, #414, #384 | Text-search first-request outlier, per-stage search timing at 100k and 1M, filtered-search cost |
| #365, #347 | Embedding swap with a dual-field migration; evaluate another text and similarity embedding |
| #400, #416 | Attribute signals for clustering; near-duplicate threshold evaluation |
| #67 | Request-driven processing plans |
| #397 | Live write-path suite failures |
| #411 | 0.7.0 close-out: live acceptance run and tag |

## 9. Reproducibility

Every source note ends with a Reproduce section that has the exact commands. Use an isolated compose
project (`docker compose -p <name>`), never the default one. The entry points:

| To reproduce | Start from |
|---|---|
| Public ingest sets | `scripts/datasets/fetch_coco_subset.py --bench-set 2000 --manifest scripts/datasets/manifests/coco_bench_2000.json`; `scripts/datasets/fetch_hires_set.py --n 500 --manifest scripts/datasets/manifests/commons_hires_500.json`; `scripts/bench/select_baseline_set.py` |
| Stage profile and throughput | `scripts/bench/ingest_profile.py run ... --clients 4`, then `ingest_profile.py table`; `scripts/bench/run_baseline.py --reps N --drop-cache --clients N` ([PERFORMANCE.md](PERFORMANCE.md), "0.7.0 stage profile"; [reviews/sp0_review.md](reviews/sp0_review.md), runbook) |
| `gpu_image_prep` parity | `scripts/bench/gpu_prep_parity.py` (capture, then `compare a.npz b.npz`) |
| Per-model memory and the load profile | [../docs-site/docs/operations/ingest-load-profile.mdx](../docs-site/docs/operations/ingest-load-profile.mdx); raw rows in `docs/benchmarks/v070/spf/` |
| IVF restart gate | `scripts/bench/ivf_nredo_agreement.py --vectors V.npy --meta M.jsonl --labels L.jsonl --candidates 1,2,3 --seeds 7,13,21,5,9,11` |
| Cluster quality | `scripts/bench/cluster_eval.py`; baseline summary `docs/benchmarks/cluster_config_baseline_3000.json` |
| Loop simulation and Z4 | `scripts/research/cluster_diag/loop_sim.py`, `scripts/research/cluster_diag/z4_run.py`; pin `scripts/datasets/manifests/openset_test_z4.json` |
| Labelled benchmarks and WB2 | `scripts/datasets/prepare_labelled_benchmark.py`; `scripts/research/standard_benchmarks/` (`embed_run`, `score_run score`, `score_run decide`); raw JSON `docs/benchmarks/wb2_standard_benchmarks/` |
| Leiden prototype | `scripts/research/leiden_refine_proto.py` |
| Visual judge | `scripts/research/visual_judge.py` |
| The loop, live | `tests/live/test_live_assist_loop.py` on the harness in `docker/test/compose.yml` |

The offline tests for the research tooling are
`pytest tests/research tests/research_scripts tests/bench tests/test_contact_sheet.py -q --no-cov`
([research/cluster_program_outcome.md](research/cluster_program_outcome.md), section 11). Raw run output
that is too large to commit stays in gitignored `artifacts_local/`. The notes name the paths and the
commands that regenerate them.

Hardware for the 0.7.0 rows: one RTX A6000 (48 GB) for Triton and the API, Triton 2.70.0 (release 26.06),
TensorRT 11.1, OpenSearch 3.6.0 on local NVMe, and a 48-thread dual Xeon E5-2680 v3 host shared with other
jobs ([research/v070_measurement_log.md](research/v070_measurement_log.md), Setup and SP0).
