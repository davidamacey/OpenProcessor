# Fine-grained grouping benchmark (wave WB): results

Measured 2026-10-29 and 2026-10-30 on branch `feat/cluster-wb-fine-grained`. Plan and bars:
[fine_grained_benchmark_plan.md](fine_grained_benchmark_plan.md), registered before any data was
read and not changed. Every bar below is the registered one. Hardware: one 48 GB RTX A6000 (GPU 2)
shared with an unrelated inference server, behind a lock, plus CPU. The host was busy throughout
(1-minute load average 9 to 78, noted per run in the raw JSON). Raw JSON (per run, with git sha and
load average) is kept in a scratch directory outside the repository; the note's Reproduce section
says how to regenerate it. No product code changed. No private data, no set P crop and no
hosted-model call on a car crop from set P: set P was not touched in this note.

## What was run

| Step | State |
|---|---|
| F0 FG-0 label check of set C (294-row review sample) | done as **machine review** (Haiku subagents on contact sheets); no person has checked it |
| F1/F2 embeddings of set C plus 10,358 Z2 car-key distractors, 5 models | done: 13,301 crops, one decode per image |
| F3 preprocessing variants on the top models | done for MobileCLIP2, SigLIP2 and PE-Core-G (letterbox, 10 % margin, whole image) |
| F4 FG-2, FG-4, FG-5 (CPU) | average linkage, k-reciprocal and DBA done on 13,301 rows; Leiden and ECG on set C alone (2,943 rows, see FG-4) |
| F5 FG-7 judge agreement at make and model level | Haiku half done; local-VLM half **not run** (GPU 0 is taken by another session) |
| F6 FG-3 on set C-large | **not run**: set C-large not fetched; reported unreadable |
| F7 FG-6 class-disjoint novelty queue | done |
| F8 set P sizes and local judge | **not run** (needs the local VLM on GPU 0 and an owner-approved sample) |

## Setup and method

- **Data.** Set C: 2,943 images, 26 makes, 50 models, 74 generations (Wikimedia Commons, licence
  per file). Distractors: the 10,358 crops of the Z2 vehicle set the product detector filed under
  `car`, added as unlabelled members, never scored. Total key: 13,301 rows.
- **Splits.** Set C is split 50/50 by image, stratified by generation, seed 20261105
  (`setc.split_halves`). Fold 1 calibrates on one half and is read once on the other; fold 2 swaps
  them. The class-disjoint split for FG-6 holds out half the models of each make with two or more
  models (`setc.split_new_models`).
- **Crops.** The labelled crop of a set C image is the largest car or truck box of a YOLO11s
  detector (confidence 0.25). 137 of 2,943 images (4.7 %) had no detection and use the whole
  image. This replaces the plan's throwaway product project: no ingest stack was started, so GPU 2
  was used only for the encoders. Distractors use their stored product box. Each image is decoded
  once in a loader worker; every (model, variant) square is cut from that frame.
- **Encoders** (public weights, research venv, fp16 autocast, L2-normalised):
  PE-Core-L14-336 (production model, `timm` weights, repo preprocessing `src/utils/pe_preprocess.py`
  reproduced exactly: a test pins the centre variant to it; cosine to the stored production
  vectors on 40 Z2 crops: mean 0.985, min 0.920); MobileCLIP2-S2 (the repo's reparameterised
  export weights); DINOv2-L/14 (224 px, CLS; matches the Hugging Face reference at cosine 0.998 on
  50 crops); SigLIP 2 So400m/16 at 256 px; PE-Core-G14-448 (fits GPU 2: 9.2 GB peak).
- **Preprocessing variants** (`preprocess.py`): centre (production), letterbox to a square on
  mid-grey, 10 % context margin, whole image (set C only).
- **Metrics.** Ranking: P@1, P@10, mAP@R, 10-NN vote at make, model and generation level among
  the labelled crops of the scored half, and P@10 again with every distractor in the gallery as a
  wrong answer. Partition: BCubed at model and make level, pair recall, within-make false-merge
  share, R1 (review cost, oracle flagger), U1 to U3 of the #286 gate at model and make level (20
  seeds, theta 0.9, alpha 0.05, max dissent 1), permutation ARI. Clusters are formed over all
  13,301 rows and scored on the labelled crops of one half; for R1 and U1 to U3 the pool is that
  half's labelled crops (as in Z2).
- **Operating point.** Section 6's rule at model level, on the calibration half only: the largest
  target pair recall in {0.05, 0.1, 0.2, 0.3} whose most precise cut keeps BCubed model precision
  at or above 0.90 (`metrics.choose_operating_point`).

## FG-0: set C labels (machine review only)

Sheets: one per claimed model (50), the model's review-sample images plus two planted wrong images
(another model of the same make where the make has one, another make). The reviewer is told the
claimed make and model and answers yes, no or unsure per cell for "is this model" and "is this
make". Verdicts are kept as raw JSON.

| | Count | Rate (Wilson 95 %) |
|---|---|---|
| Sample rows | 294 | |
| Unsure (interiors, badges, detail shots, heavy occlusion) | 87 | 0.296 |
| Model confirmed, among decided rows (201 of 207) | | **0.971 (0.938 to 0.987)** |
| Model confirmed, unsure counted as not confirmed (201 of 294) | | 0.684 (0.628 to 0.734) |
| Make confirmed, among decided rows (206 of 207) | | 0.995 (0.973 to 0.999) |
| Planted same-make wrong model rejected at model level (25 of 34) | | 0.735 (0.569 to 0.854) |
| Planted other-make image rejected at make level (29 of 50) | | 0.580 (0.442 to 0.706) |

**FG-0: INCONCLUSIVE.** The registered bar is 0.95 *checked by a person*. This reviewer is a small
vision model: it accepts a quarter of the planted wrong models and 42 % of the planted wrong-make
images (including "unsure" as not rejecting), so it cannot certify model-level labels, and its
decided-row accuracy of 0.971 is not evidence of 0.95. Make-level labels look sound (0.995 on 207
decided rows). A person should spot check: the 294 rows are in `review_sample.csv` next to set C,
and the per-cell verdicts and plant roles are in the raw FG-0 report. A real finding independent of
the reviewer: about 30 % of sampled Commons images are not usable exterior shots (an unsure cell is
an interior, a badge, a screen), so set C carries label-irrelevant images that count against every
model-level score.

**Consequence for the bars.** Per the plan, if FG-0 does not hold only make-level bars are read.
FG-0 has not failed, it is unread. This note therefore reports model-level numbers, marks them
**provisional until a person fills FG-0**, and treats make-level numbers as firm.

## FG-1: representation screen (centre crop; P@10 at model level, held-out half)

Fold 1 is the read; fold 2 is reported. All values are means over the queries of the half
(about 1,470 queries; 20 to 30 same-model images per query).

| Model | Dim | Make P@10 | Model P@10 f1 (f2) | Model mAP@R | Generation P@10 | Model P@10 with distractors | FG-1 vs PE-Core (+0.03) |
|---|---|---|---|---|---|---|---|
| PE-Core-L14-336 (production) | 1024 | 0.748 | 0.650 (0.642) | 0.325 | 0.557 | 0.597 | baseline |
| MobileCLIP2-S2 | 512 | 0.826 | 0.759 (0.751) | 0.469 | 0.666 | 0.643 | **PASS** (+0.109, +0.109) |
| SigLIP 2 So400m/16 | 1152 | 0.796 | 0.728 (0.716) | 0.392 | 0.611 | 0.625 | **PASS** (+0.078, +0.074) |
| DINOv2-L/14 | 1024 | 0.328 | 0.282 (0.256) | 0.121 | 0.245 | 0.218 | FAIL (-0.368, -0.387) |
| PE-Core-G14-448 | 1280 | 0.849 | 0.795 (0.782) | 0.442 | 0.681 | 0.749 | **PASS** (+0.145, +0.140) |

The ranking is MobileCLIP2 and PE-Core-G first, SigLIP 2 third, PE-Core-L fourth, DINOv2 far
behind. DINOv2's CLS vector groups by scene, not by vehicle model (it matches the reference
implementation, so this is the model, not the pipeline). **The production encoder is not the best
encoder for this question, and the best small one (MobileCLIP2-S2) is already served by Triton.**
PE-Core-G needs 9.2 GB and 17 images per second, see the cost table.

### Preprocessing variants (FG-1, second half)

Run on the models that led the screen: MobileCLIP2, PE-Core-G and, because it was third by a small
margin, SigLIP 2. Model P@10, fold 1 (fold 2); change against the same model at centre.

| Model | Letterbox | 10 % margin | Whole image (set C diagnostic) |
|---|---|---|---|
| MobileCLIP2-S2 | 0.799 (0.791), **+0.040**, make +0.029 | 0.760 (0.755), +0.001 | 0.700 (0.677) |
| SigLIP 2 | 0.842 (0.833), **+0.114**, make +0.083 | 0.737 (0.732), +0.009 | 0.682 (0.670) |
| PE-Core-G | 0.821 (0.810), +0.026, make +0.020 | 0.789 (0.777), -0.006 | 0.748 (0.738) |

Class-level P@10 (the shared crop path check) is measured on the labelled crops of the Z2 vehicle
set (14,961 crops, 24 classes, same-photo neighbours excluded), not on the plan's 3,000-item
project, which was not available as crops. Letterbox against centre: MobileCLIP2 0.933 to 0.954,
SigLIP 2 0.913 to 0.942. No class-level loss.

| Variant | FG-1 |
|---|---|
| MobileCLIP2 letterbox | **PASS** in both folds |
| SigLIP 2 letterbox | **PASS** in both folds |
| PE-Core-G letterbox | FAIL (+0.026 < +0.03) |
| 10 % margin, every model | FAIL |
| Whole image | diagnostic only; lowest everywhere, as the plan expected |

Letterbox keeps the whole car in view and helps because the centre crop of a wide car box cuts off
the front and rear. Context margin does not help. The gain on the production PE-Core was not
measured: the plan runs variants on the top models only.

## FG-2: average linkage inside the car key

Operating point chosen on calibration, read once on the held-out half. Pools are the labelled
crops of the half (about 1,470). BCubed at model and make level, R1 and ARI from the table. The
within-make column is the share of model-level false merges that still share a make.

| Representation | Variant | Target recall | Model BCubed P (R) f1 | Make BCubed P f1 | R1 f1 | Clusters f1 | Within-make false merges f1 | Model P f2 | Perm. ARI | FG-2 |
|---|---|---|---|---|---|---|---|---|---|---|
| PE-Core-L | centre | 0.1 | 0.930 (0.157) | 0.965 | 0.539 | 709 | 0.54 | 0.959 | 1.000 | **FAIL** (R1 > 0.5) |
| MobileCLIP2-S2 | centre | 0.3 | 0.918 (0.413) | 0.960 | 0.319 | 365 | 0.75 | 0.931 | 1.000 | **PASS** |
| MobileCLIP2-S2 | letterbox | 0.3 | 0.941 (0.402) | 0.970 | 0.300 | 364 | 0.76 | 0.956 | 1.000 | **PASS** |
| SigLIP 2 | centre | 0.2 | 0.932 (0.258) | 0.957 | 0.448 | 574 | 0.46 | 0.939 | 1.000 | **PASS** |
| SigLIP 2 | letterbox | 0.3 | 0.960 (0.352) | 0.977 | 0.365 | 482 | 0.68 | 0.967 | 1.000 | **PASS** |
| PE-Core-G | centre | 0.3 | 0.912 (0.354) | 0.972 | 0.404 | 472 | 0.91 | 0.913 | 1.000 | **PASS** |
| DINOv2-L | centre | none | no target reaches 0.90 on calibration | | | | | | | **FAIL** |

No fold-2 model or make precision misses its bar by more than 0.03, so no result reopens.

What the pass means. Clusters hold 0.92 to 0.96 BCubed model precision at 0.25 to 0.41 BCubed
recall. Most remaining model errors stay inside the make (0.54 to 0.91 for the passing
representations): a wrongly merged cluster is mostly still the right make. Gate statistics at
model level (U1 decisions per item, 20 seeds): 0.887 to 0.973; U2 0.99 and above. That is, on set C
the gate propagates labels for only 3 to 11 % of items (U3 0.03 to 0.11) because 31 to 40 images per
generation make most clusters smaller than the sample the gate needs. This is the plan's known
limit, and FG-3 needs set C-large.

## FG-4: other grouping methods against average linkage (MobileCLIP2-S2, centre)

Replace average linkage only if precision at matched BCubed recall (within 0.02) rises by 0.02 in
both folds with no make loss above 0.01.

| Method | Fold 1 precision gain, make change | Fold 2 precision gain, make change | FG-4 |
|---|---|---|---|
| k-reciprocal re-ranking (20, 6, 0.3) | -0.008, -0.016 | -0.027, -0.024 | **FAIL** |
| DBA (k 5, alpha 3) | -0.026, -0.025 | -0.023, -0.022 | **FAIL** |
| Leiden CPM (mutual 30-NN graph, calibrated fraction) | -0.037, -0.025 | -0.053, -0.051 | **FAIL**; also seed ARI 0.79 and permutation ARI 0.82, below the 0.95 stability bar |
| ECG (32-run ensemble, mutual 30-NN graph) | no operating point | no operating point | **FAIL**: no cut reaches 0.90 BCubed model precision on calibration at any target recall |

Leiden and ECG ran on set C alone (2,943 rows, no distractors): a single Leiden partition of the
13,301-row key did not finish in over 40 minutes on this busy host, a run of 12 resolutions would
have taken hours, so the graph methods are compared with average linkage on the same 2,943 rows
(average linkage there: model BCubed precision 0.924 and 0.926, make 0.961 and 0.959, R1 0.325 and
0.360 in folds 1 and 2). The k-reciprocal and DBA rows above use the full 13,301-row key. ECG is
implemented from the paper's definition (igraph's Python binding has no ECG) with an ensemble of
32 first-level Louvain runs; it has no resolution knob, so it is read at its single partition.

## FG-5: a second vector

A second embedding vector means a re-embed of every item. At FG-2's precision bars, R1 against
PE-Core's 0.539 (fold 1): MobileCLIP2-S2 0.319 (0.59 x), SigLIP 2 0.448 (0.83 x). Both pass FG-1
and the 0.90 factor, so FG-5 reads **PASS** for both. This is a *replacement* case in practice:
the production vector loses on this benchmark, so the finding is that PE-Core should not be the only
vector for make and model grouping. Which vector and whether it is adopted is a decision for the
owner (and for the paper's own MobileCLIP2-versus-PE comparison on class-level tasks, which this
benchmark does not cover). MobileCLIP2 is already served; its 512-dimension vector is half the
size.

## FG-6: a queue of new models

Class-disjoint split, novelty against the known examples of the same make, per make with held-out
models (10 makes). Fold 1 (fold 2). AUROC bar 0.90 in each make; flag precision bar 0.80.

| Representation | Makes with AUROC at least 0.90 | Median AUROC | Pooled flag precision | FG-6 |
|---|---|---|---|---|
| PE-Core-L | 1 of 10 (0 of 10) | 0.755 (0.735) | 0.753 | **FAIL** |
| MobileCLIP2-S2 | 3 of 10 (2 of 10) | 0.860 (0.814) | 0.814 (0.720) | **FAIL** |
| SigLIP 2 | 1 of 10 (0 of 10) | 0.779 (0.777) | 0.714 | **FAIL** |
| PE-Core-G | 0 of 10 (1 of 10) | 0.799 (0.802) | 0.750 | **FAIL** |

**New models are found by a person naming groups**, as the plan says on failure.

## FG-7: judge agreement at make and model level (Haiku, 6 x 4 contact sheets of set C crops)

55 cluster sheets at make level and 53 at model level (50 base sheets plus re-ordered twins) and 12 pair sheets per level; synthetic clusters
of 24 with 0 to 8 contaminants drawn from other makes or models. Bars: pair kappa at least 0.7 and
cell-flag precision at least 0.8.

| Level | Pair kappa | Cell-flag precision (recall) | Cluster same/different accuracy | Pair unsure rate | FG-7 |
|---|---|---|---|---|---|
| Make | 0.796 | 0.819 (0.840) | 0.855 | 0.389 | **PASS** |
| Model | 0.455 | 0.676 (0.898) | 0.660 | 0.424 | **FAIL** |

The judge may be used for make-level purity checks and may not be used for model-level ones. Its
"same" answers are reliable (false-same rate 0 at both levels in the cluster sheets); its
"different" answers over-flag at model level (false-different rate 0.78). Limits: one Haiku run
per sheet, 55 or 53 cluster sheets and 12 pair sheets (4 make-level cluster sheets beyond the 55 limit were not judged), and the
local-VLM half of E0 was **not run** (GPU 0 is in use elsewhere), so the plan's "local VLM judges
the full sample" and the set P step stay open.

## Cost per variant

Encoder throughput and peak GPU memory, fp16 autocast, batch 32, synthetic input, 3 repetitions
(preprocessing and image decode excluded). Measured on a shared GPU with the host load average
between 9 and 78, so the rates are lower bounds. These are PyTorch rates, not the Triton TensorRT
rates the product uses; they are comparable among rows.

| Model | Dim | Images/s (3 runs) | Peak memory | Re-embed 100k crops | Re-embed 1M crops |
|---|---|---|---|---|---|
| MobileCLIP2-S2 | 512 | 807, 873, 860 | 0.63 GB | 2 min | 20 min |
| DINOv2-L | 1024 | 375, 374, 373 | 1.45 GB | 4.5 min | 45 min |
| SigLIP 2 So400m | 1152 | 313, 313, 312 | 1.98 GB | 5.3 min | 53 min |
| PE-Core-L (production) | 1024 | 134, 134, 134 | 1.84 GB | 12.4 min | 124 min |
| PE-Core-G | 1280 | 17.6, 17.2, 17.3 | 9.17 GB | 96 min | 16.0 h |

MobileCLIP2 is the cheapest and, here, the best small model. PE-Core-G is 7.7x slower than PE-Core-L
and needs 5x the memory.

## Bar table

| Bar | Result | Note |
|---|---|---|
| FG-0 | **INCONCLUSIVE** | machine review cannot read model level; make level looks sound; a person must check |
| FG-1 | MobileCLIP2, SigLIP 2, PE-Core-G **PASS** as challengers; letterbox **PASS** for MobileCLIP2 and SigLIP 2 | model-level, provisional on FG-0; margin fails |
| FG-2 | **PASS** for MobileCLIP2 (centre and letterbox), SigLIP 2, PE-Core-G; **FAIL** for PE-Core-L (R1) and DINOv2 | model-level provisional; make-level precision passes for all passing rows |
| FG-3 | **INCONCLUSIVE (unreadable)** | set C-large not fetched |
| FG-4 | k-reciprocal, DBA, Leiden, ECG all **FAIL** | average linkage stays; Leiden and ECG also fail the stability bar or have no operating point |
| FG-5 | **PASS** for MobileCLIP2 and SigLIP 2 | |
| FG-6 | **FAIL** | person names new models |
| FG-7 | make **PASS**, model **FAIL** | Haiku half only |

## Findings

- At make and model level the representation matters far more than the grouping method. Replacing
  the production vector by MobileCLIP2-S2 raises model P@10 by 0.11 and cuts review cost R1 from
  0.54 to 0.32 at the same registered precision bars; letterbox preprocessing adds 0.04 to 0.11
  more for MobileCLIP2 and SigLIP 2.
- Average linkage inside the key already meets FG-2 for three representations. k-reciprocal
  re-ranking, DBA, Leiden and ECG lose to it.
- The class-level lesson holds one level down: the review gain comes from the encoder, not from a
  cleverer grouping.
- Make-level naming is judgeable by the Haiku judge; model-level is not. Treat model-level names as
  a person's job.
- New-model detection by distance to known examples is not good enough.

## Limits

- Set C is curated Commons photography, with 4.7 % crops that fell back to the whole image and about
  30 % non-exterior shots in the review sample. It is easier than street photos, and set P is not
  measured here.
- The labels are unverified by a person (FG-0). Model-level numbers are provisional.
- Crops come from a YOLO11s detector, not the throwaway-project product path.
- One data split seed per fold pair; no confidence intervals on the partition metrics. Fold 2
  differences of 0.01 to 0.03 are visible in the tables above.
- Letterbox was not run for PE-Core-L. Leiden and ECG were not run on the 13,301-row key.
- Viewpoint effects are not measured (no view labels, no CompCars).
- FG-3 and set P sizes are open.

## Open issues

- A person fills the 294 FG-0 rows.
- Fetch set C-large and read FG-3; run the local-VLM half of FG-7 when a GPU is free; set P sizes
  and local-judge purity at make level.
- Re-run the variant check on the plan's 3,000-item project once it is built as crops.

## Reproduce

Research-only dependencies: `scripts/research/fine_grained_requirements.txt`, installed into a
venv outside the repository. Required environment variables: `FG_SETC_ROOT` (set C directory with
`labels.jsonl` and `review_sample.csv`), `FG_Z2_BUNDLE` (the Z2 vector bundle: `vecs.meta.jsonl`
and `eval/labels.jsonl`), `FG_DETECTOR` (YOLO11s weights), `FG_MOBILECLIP_CKPT` (MobileCLIP2-S2
checkpoint).

```bash
export PYTHONPATH=. FG_SETC_ROOT=... FG_Z2_BUNDLE=... FG_DETECTOR=... FG_MOBILECLIP_CKPT=...
G="flock $LOCKS/gpu2.lock nice -n 10 env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 $VENV/bin/python -m scripts.research.fine_grained"
$G.embed_run extract --out $EMB --models s_pe_core s_mobileclip2 s_dinov2 s_siglip2 --subset all
$G.embed_run extract --out $EMB --models s_mobileclip2 s_siglip2 --variants letterbox margin10 --subset all
$G.embed_run extract --out $EMB --models s_mobileclip2 s_siglip2 --variants whole --subset setc
$G.embed_run extract --out $EMB --models s_pe_core_g --variants centre letterbox margin10 --subset all
$G.embed_run extract --out $EMB --models s_mobileclip2 s_siglip2 --variants centre letterbox --subset z2
$G.bench --out $RES/bench.json --models s_pe_core s_mobileclip2 s_dinov2 s_siglip2 s_pe_core_g
# CPU, any venv with scipy, igraph and leidenalg
$VENV/bin/python -m scripts.research.fine_grained.screen_run --emb $EMB --out $RES/screen_centre.json --models pe_core_l14_336 mobileclip2_s2 dinov2_l14 siglip2_so400m_p16_256
$VENV/bin/python -m scripts.research.fine_grained.group_run --emb $EMB --out $RES/group.json --model mobileclip2_s2 --methods ahc kr dba
$VENV/bin/python -m scripts.research.fine_grained.group_run --emb $EMB --out $RES/graph.json --model mobileclip2_s2 --methods ahc leiden ecg --distractors 0
$VENV/bin/python -m scripts.research.fine_grained.fg6_run --emb $EMB --model mobileclip2_s2 --out $RES/fg6.json
$VENV/bin/python -m scripts.research.fine_grained.class_p10 --emb $EMB --out $RES/class_p10.json --keys mobileclip2_s2__centre mobileclip2_s2__letterbox
$VENV/bin/python -m scripts.research.fine_grained.summary_run --res $RES --out $RES/summary.json
# FG-0 and FG-7 (contact sheets; the Haiku step is the Agent tool, see visual_judge_e0.md)
$VENV/bin/python -m scripts.research.fine_grained.fg0_run prepare --out $FG0   # then collect
$VENV/bin/python -m scripts.research.fine_grained.fg7_pool --boxes $EMB/boxes_setc.json --out $FG7/pool
```

Tests: `.venv/bin/python -m pytest tests/research/test_fine_grained_data.py
tests/research/test_fine_grained_metrics.py tests/research/test_fine_grained_novelty.py -q --no-cov`.
Twelve mutations of the logic (padding offset, margin size, within-make test, review-cost sheet
count, operating-point precision test, plant threshold, decided-row filter, FG-1 and FG-2
thresholds, ECG weight formula, split size, mAP@R) each made a test fail.
