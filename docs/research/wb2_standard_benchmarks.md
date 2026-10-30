# WB2: embedding and clustering on standard labelled benchmarks

Status: **registration**, written 2026-10-29 before any embedding of these sets was scored. Sections 1
to 8 are fixed by this commit. A bar changed after a read needs a new registration and a statement of
what was read first. Results are appended in section 9 by a later commit; sections 1 to 8 are not
edited then.

Why this exists: the Commons set C result of wave WB ([fine_grained_benchmark_plan.md](fine_grained_benchmark_plan.md))
rests on labels taken from category names whose error rate is unknown
([labelled_benchmarks.md](labelled_benchmarks.md), "Why"). The embedding decision (#347, #365) is
therefore re-run on sets whose labels the dataset authors wrote and checked.

What was seen before registering (disclosure): the WB set C screen. Model-level P@10 on the held-out
half, centre-crop: PE-Core-G 0.795, MobileCLIP2-S2 0.759, SigLIP 2 0.728, PE-Core-L 0.650, DINOv2-L
0.282; with letterbox, SigLIP 2 0.842 and MobileCLIP2-S2 0.799. Of the grouping methods, average
linkage passed its bar and k-reciprocal re-ranking and DBA failed theirs (raw WB JSON, outside the
repo). No result on CARS196, CUB-200-2011 or SOP had been computed.

## 1. Data and splits

Data is read from `$WB2_DATA_ROOT` (the `clustering/` directory of the local dataset store; each set
has `manifest.json` and `labels.jsonl`, read-only). Nothing from the sets is committed. All three are
non-commercial research or marketplace photos: local evaluation only.

| Set | Evaluated ("test") | Calibration ("train") | Expected counts, checked against the manifest |
|---|---|---|---|
| CARS196 (Stanford Cars, community parquet mirror) | zero-shot protocol: the 98 held-out classes (`dml_split == test`), train and test image splits together | the other 98 classes | 8,131 test images; 8,054 calibration; 16,185 total |
| CUB-200-2011 | classes 101 to 200 (`dml_split == test`) | classes 1 to 100 | 5,924 test images; 5,864 calibration; 11,788 total |
| Stanford Online Products (SOP) | the official test split: 11,316 products | a seeded sample of 3,000 train products (all their images) | 60,502 test images; calibration size recorded |

The calibration sets exist only to choose a preprocessing variant and a clustering cut, never to
train. VMMRdb is not run: its labels come from advertisement titles (noisy, no published protocol), so
it cannot arbitrate a ranking. Faces are out of scope here.

Query set = gallery = evaluated set, the query itself excluded (the DML protocol).

## 2. Models and preprocessing

Encoders are the five of the WB harness (`scripts/research/fine_grained/models.py`, same weights,
fp16 autocast, L2-normalised): PE-Core-L14-336 (the production model), PE-Core-G14-448,
MobileCLIP2-S2, SigLIP 2 So400m/16-256, DINOv2-L/14 (224). New heavy dependencies stay in the research
venv; nothing is installed into the product.

Preprocessing (whole image, no detector, since the protocols score whole images), each image decoded
once:

- `centre`: shorter edge to the model size, centre square (the production rule);
- `letterbox`: longer edge to the model size, grey padding, nothing cropped.

Both use Pillow BILINEAR, as WB. Per model the variant that goes into the cross-model comparison is
the one with the higher Recall@1 on that dataset family's **calibration** set (CARS196 calibration
decides for all three sets, to keep one rule per model; CUB and SOP read both variants as well).

## 3. Metrics

- **Retrieval:** Recall@K, a query scores 1 if any of its K nearest neighbours (cosine, itself excluded,
  ties by index) shares its class. K = 1, 2, 4, 8 on all sets; also 10, 100, 1000 on SOP.
- **k-means NMI:** k-means on the evaluated embeddings with K = number of classes (98, 100, 11,316), NMI
  against the labels (arithmetic normalisation, as scikit-learn). Own torch implementation (k-means++
  start, at most 300 Lloyd steps, relative inertia tolerance 1e-4) so K = 11,316 is feasible; it is
  unit-tested against scikit-learn. Seeds 0, 1, 2; the mean and the spread over the three seeds are
  reported.
- **Our pipeline:** average-linkage agglomerative clustering on cosine distance, as the product
  cluster pass uses inside a key (`scripts/research/cluster_diag/diagnostics.py`, `linkage_of`). It is
  the only grouping method WB kept; k-reciprocal re-ranking and DBA failed their bars there and are
  not carried over. Two readings, each scored with ARI, NMI and BCubed precision, recall and F:
  1. **Operating cut:** WB's rule moved to classes. On the calibration set, take the largest target
     pair recall T in {0.05, 0.1, 0.2, 0.3} whose calibration BCubed precision is at least 0.90 (as
     `choose_operating_point`); apply that dendrogram height to the evaluated set. No qualifying T
     means the rule fails for that model and set.
  2. **Cut at K:** the dendrogram cut into exactly K clusters (K = number of classes). Oracle K, so
     reported next to k-means NMI, not as a deployable setting.
- **SOP clustering subset:** average linkage is quadratic in memory; on SOP it runs on a seeded sample
  of 3,000 test products with all their images (the calibration uses the 3,000-product train sample).
  k-means and Recall use the full test split.
- **Uncertainty of a Recall@1 difference:** paired bootstrap over queries (2,000 resamples, seed 0) of
  the per-query hit vectors, 95 % percentile interval. Embeddings are deterministic, so repeated runs
  of the same model give the same value; the interval is the sampling uncertainty.
- **Cost:** encode throughput and peak GPU memory per model from `scripts/research/fine_grained/bench.py`
  (3 repetitions), vector size, re-embed time at 100k and 1M. k-means and linkage wall time over the three
  repetitions, with the load average.

Every raw result file records the git sha and the load average at run time.

## 4. Sanity check: what "pipeline sane" means

The pipeline is declared sane only if **all** hold; if any fails, no model comparison is read and the
cause is fixed first.

1. **Metric code:** the unit tests pass (Recall against a brute-force loop, NMI against scikit-learn,
   k-means against scikit-learn on separable blobs, bootstrap on a known case).
2. **Row alignment, negative control:** with the labels randomly permuted (seed 0), every model's
   Recall@1 on CARS196 is below 0.02 and k-means NMI below 0.05. A pass here shows labels and vectors
   are aligned and the score is not trivially high.
3. **Internal consistency:** for every model and set, Recall@1 <= @2 <= @4 <= @8; NMI in (0, 1); the
   number of rows equals the registered count in section 1.
4. **Plausible range:** every model's CARS196 Recall@1 is at least 0.20 (chance is about 0.01) and no
   model exceeds 0.95 on any set (above that would indicate leakage, for example a duplicate image
   counted as a neighbour). Fine-tuned ViT-S/16 Recall@1 on these sets is 78.7 to 90.1 per the survey
   note; a frozen encoder is not expected to beat it clearly, so a frozen value above the fine-tuned
   reference by more than 0.05 on the same set is flagged for inspection, not accepted silently.
5. **Weak published cross-check, not a gate:** MIEB (linear probe, classification) orders Cars as
   SigLIP > CLIP > DINOv2 ([labelled_benchmarks.md](labelled_benchmarks.md), section 2). A probe is not
   retrieval, so the registered reading is: SigLIP 2 Recall@1 on CARS196 >= DINOv2-L's. If it holds,
   noted as consistent; if it does not, it is reported as a contradiction with its size and the
   preprocessing is checked, but it does not by itself fail the pipeline (items 1 to 4 do).

## 5. Bars

Embeddings are deterministic, so a bar is a comparison of measured values with the uncertainty of
section 3. Verdicts: **PASS**, **FAIL**, **INCONCLUSIVE** (the evidence does not settle the bar);
every verdict is reported with its limits.

Let delta_d(M) be Recall@1 of model M minus Recall@1 of PE-Core-L (the production model) on dataset
d, each at its calibration-selected variant, with the bootstrap interval [lo, hi].

| ID | Bar | PASS | FAIL | Otherwise |
|---|---|---|---|---|
| B1 | A challenger model M replaces PE-Core-L as the image embedding: margin 0.03 as WB's FG-1 | delta_CARS >= 0.03 with lo > 0, and on CUB and SOP delta >= -0.01 | delta_CARS: hi < 0.03; or on CUB or SOP: hi < -0.01 | INCONCLUSIVE |
| B2 | A preprocessing variant (letterbox over centre) goes forward for model M | CARS196 Recall@1 gain >= 0.03 with lo > 0, and on CUB and SOP no loss beyond 0.01 | gain: hi < 0.03; or on CUB or SOP loss with hi < -0.01 | INCONCLUSIVE |
| B3 | Our pipeline transfers a calibration-chosen cut | test BCubed precision >= 0.90 at the cut of section 3 (reading 1) | no qualifying cut on calibration, or test precision < 0.90 | not applicable |
| B4 | A model may be recommended for production only with a text tower (semantic search, `/search/text`) | a text tower exists for the same image weights and can be exported or served | none | |

B3 is read per model, per set. B4 is a requirement, not a score: DINOv2-L has no text tower, so it
cannot be the production image embedding whatever its numbers; it can enter only as a second vector,
under WB's FG-5 rule (a second vector means re-embedding every item, #365). The text towers of the
other four are available as open_clip configurations (PE-Core-L-14-336, PE-Core-bigG-14-448,
MobileCLIP2-S2, ViT-SO400M-16-SigLIP2-256); that the timm image weights used here are the image tower
of those same checkpoints is assumed and is listed as unverified until measured.

## 6. Comparison with the literature and with set C

- Published numbers are quoted only from [labelled_benchmarks.md](labelled_benchmarks.md) and only where
  I re-open the source in section 9; otherwise the survey's figure is marked "as stated in the survey,
  not re-checked".
- Set C ordering for comparison, from WB (model-level P@10, centre-crop, held-out): PE-Core-G, MobileCLIP2-S2,
  SigLIP 2, PE-Core-L, DINOv2-L. A contradiction is reported when two models' order on CARS196
  Recall@1 is reversed relative to set C **and** the bootstrap interval of their CARS196 difference
  excludes 0. The implication is stated in section 9.

## 7. Compute and ordering

Embeddings on GPU 2 only, under `flock $SCRATCH/locks/gpu2.lock` with
`CUDA_DEVICE_ORDER=PCI_BUS_ID`, `nice -n 10`; GPUs 0 and 1 are not used. Each image is decoded once; all
(model, variant) squares come from that frame. Embeddings are cached outside the repo, one file per
(set, model, variant). Order: CARS196, CUB, SOP; the sanity check (section 4) is read after CARS196 and
before the others are scored.

## 8. Known limits, before any run

- Frozen encoders on whole images; set C and set P hold detector crops, so the preprocessing finding
  transfers only in direction.
- CARS196 is a mirror with no stated licence; labels are the dataset authors', but its own noise is
  not measured here.
- SOP products are marketplace photos with white backgrounds; they say little about street photos.
- Recall@1 on one set ranks retrieval, not the quality of a clustering cut; B3 covers the cut.
- The tests of MIEB-style generic clustering (ImageNet, CIFAR) are not reproduced.

## 9. Results (measured 2026-10-29 to 2026-10-30)

Setup: embeddings on one RTX A6000 (GPU 2, shared with another job that held about 29 GB; my
process peaked near 14 GB), torch 2.15.0+cu132, fp16 autocast, `nice -n 10`, under the GPU lock. Host
load average 13 to 63 during extraction (CARS196 43 min, CUB 37 min, SOP 209 min wall), 28 to 77 during
scoring, 79 during the throughput run, so every timing below is indicative only. Code: embeddings of
CARS196 ran from the working tree of commit `0de07359` (uncommitted, identical to the
committed `c387d18c` apart from formatting), CUB from `c387d18c`, SOP and its scoring from
`3f13fb68`. Raw JSON (git sha and load average in each): `docs/benchmarks/wb2_standard_benchmarks/`
(`scores_<set>.json`, `ahc_sop.json`, `decisions.json`, `bench.json`); per-query hit vectors and the
cached embeddings are in the local scratch cache (`$WB2_CACHE`, `res_a/`), not in the repository.
k-means and Recall for CARS196 and CUB ran on CPU, for SOP on the GPU; the numbers do not depend on it.

### 9.1 Sanity check (section 4): NOT satisfied as registered

| Item | Result | Reading |
|---|---|---|
| 1 Metric tests | pass (24 tests, mutation-checked) | ok |
| 2 Row-alignment control | Recall@1 on shuffled labels 0.008 to 0.011 for every model (bar < 0.02): **pass**. k-means NMI against shuffled labels 0.14 (CARS196), 0.18 (CUB), 0.79 (SOP) against the bar < 0.05: **fail** | The NMI bar was set wrongly. NMI of two independent random labelings of 8,131 rows into 98 classes is 0.145 (5,924 rows, 100 classes: 0.190), so 0.14 is the chance floor at this size, and with about 5 images per SOP product the floor is high. Post-hoc reading, not a registered change: alignment holds (Recall@1 at chance), NMI must be read above its floor |
| 3 Consistency | Recall@1 <= @2 <= @4 <= @8, NMI in (0,1), row counts equal 8,131 / 5,924 / 60,502: **pass** | ok |
| 4 Plausible range | CARS196 Recall@1 of MobileCLIP2, PE-Core-L, PE-Core-G and SigLIP 2 is 0.954 to 0.975, above the 0.95 ceiling: **fail** for those four. DINOv2-L (0.825) is inside | Flagged for inspection, as registered. Byte-identical files (9 groups, 18 of 8,131 files) explain at most 0.001 (Recall@1 without them is 0.001 higher, not lower). Only 1 to 3 % of queries have a neighbour with cosine above 0.98 (median nearest-neighbour cosine 0.89 to 0.94). So the high values are not duplicates. I cannot rule out that web-scale pre-training data contains Stanford Cars images (unknown, not tested). The frozen values also exceed the fine-tuned ViT-S/16 reference of the survey (86.6 to 89.2, as stated there, not re-checked) by more than 0.05 on CARS196, and PE-Core-G on CUB (0.907 against 0.856) |
| 5 Weak MIEB cross-check | SigLIP 2 0.975 >= DINOv2-L 0.825 on CARS196: consistent | The probe ordering is not reproduced on the other sets (below) |

Consequence: the registered sanity condition is not met, so every comparison below is **exploratory**.
The pipeline is not shown to be wrong (alignment control and consistency pass), but CARS196 is near its
ceiling for the CLIP-type models, which limits what it can separate.

### 9.2 Retrieval and k-means (test classes; Recall@K as fractions; k-means NMI mean of seeds 0 to 2, sd at most 0.007)

Variant shown is the one chosen on the CARS196 calibration classes (letterbox for PE-Core-L, PE-Core-G
and SigLIP 2, centre for MobileCLIP2 and DINOv2-L).

| Model (variant) | CARS196 R@1 / 2 / 4 / 8 | k-means NMI | CUB R@1 / 2 / 4 / 8 | k-means NMI | SOP R@1 / 10 / 100 / 1000 | k-means NMI |
|---|---|---|---|---|---|---|
| PE-Core-L (letterbox) | 0.961 / 0.986 / 0.994 / 0.997 | 0.939 | 0.867 / 0.924 / 0.953 / 0.971 | 0.874 | 0.796 / 0.922 / 0.976 / 0.995 | 0.919 |
| PE-Core-G (letterbox) | 0.969 / 0.986 / 0.994 / 0.996 | 0.954 | 0.907 / 0.944 / 0.958 / 0.968 | 0.916 | 0.820 / 0.939 / 0.983 / 0.997 | 0.927 |
| MobileCLIP2-S2 (centre) | 0.964 / 0.984 / 0.993 / 0.996 | 0.938 | 0.752 / 0.847 / 0.911 / 0.952 | 0.771 | 0.674 / 0.828 / 0.923 / 0.980 | 0.885 |
| SigLIP 2 (letterbox) | 0.975 / 0.987 / 0.994 / 0.996 | 0.962 | 0.744 / 0.843 / 0.916 / 0.958 | 0.735 | 0.790 / 0.921 / 0.974 / 0.995 | 0.918 |
| DINOv2-L (centre) | 0.825 / 0.877 / 0.919 / 0.949 | 0.608 | 0.892 / 0.934 / 0.954 / 0.966 | 0.881 | 0.576 / 0.732 / 0.851 / 0.956 | 0.862 |

All ten (model, variant) rows, the SOP Recall@2/4/8, the calibration Recall@1 and the k-means seeds
are in the JSON. SOP Recall@1 without byte-identical files falls by 1 to 2 points (for example
PE-Core-L letterbox 0.796 to 0.784); the registered (unmasked) values are shown. k-means NMI floors
(shuffled labels) are given in 9.1.

### 9.3 Bars

Recall@1 differences against PE-Core-L (its own calibration-chosen variant), paired bootstrap 95 %
interval in brackets.

| Challenger | CARS196 | CUB | SOP | B1 |
|---|---|---|---|---|
| PE-Core-G | +0.008 [0.005, 0.012] | +0.040 [0.032, 0.047] | +0.024 [0.021, 0.026] | **FAIL** (CARS196 gain confidently below 0.03) |
| MobileCLIP2-S2 | +0.003 [-0.001, 0.008] | -0.115 [-0.126, -0.104] | -0.122 [-0.125, -0.119] | **FAIL** |
| SigLIP 2 | +0.015 [0.011, 0.018] | -0.123 [-0.134, -0.112] | -0.006 [-0.009, -0.004] | **FAIL** |
| DINOv2-L | -0.135 [-0.143, -0.127] | +0.025 [0.017, 0.034] | -0.220 [-0.224, -0.217] | **FAIL**; also no text tower (B4) |

B2 (letterbox over centre, Recall@1 gain): PE-Core-L +0.006 / -0.004 / +0.011 (CARS196 / CUB / SOP),
PE-Core-G +0.013 / 0.000 / +0.007, SigLIP 2 +0.008 / -0.029 / +0.019, MobileCLIP2 -0.005 / -0.045 /
+0.005, DINOv2-L -0.030 / -0.013 / +0.007. **FAIL for all five** (no gain reaches 0.03 on CARS196).
Letterbox is therefore not a lever for whole images, unlike on the detector crops of set C.

B3 (calibration-chosen cut, test BCubed precision >= 0.90), average linkage:

| Set | PASS | FAIL |
|---|---|---|
| CARS196 | 9 of 10 rows (test precision 0.96 to 0.98, pair recall 0.045 to 0.206) | DINOv2-L both variants (no calibration cut reaches 0.90) |
| CUB | PE-Core-G both variants (0.94 / pair recall 0.30), DINOv2-L both (0.93 / 0.13) | MobileCLIP2 both, SigLIP 2 both, PE-Core-L both (calibration chose a cut, test precision 0.84 to 0.87) |
| SOP (3,000-product subset) | PE-Core-G both (0.92 / 0.37 to 0.38), PE-Core-L both (0.91 to 0.92 / 0.36), SigLIP 2 letterbox (0.91 / 0.36) | MobileCLIP2 both, DINOv2-L both (no cut), SigLIP 2 centre (0.895) |

PE-Core-L, the production model, fails B3 on CUB. The cut transfers on 2 of 3 sets for PE-Core-L
and 3 of 3 for PE-Core-G. Reading with K clusters (oracle K): ARI 0.05 to 0.39, NMI 0.49 to 0.91 (best
CARS196 SigLIP 2 letterbox ARI 0.385, NMI 0.850, BCubed F 0.593; k-means NMI on the same embeddings
0.962). Average linkage cut at K is clearly worse than k-means NMI on every row; its use is the
high-precision, low-recall regime of the operating cut.

B4: text tower available for PE-Core-L, PE-Core-G, MobileCLIP2 and SigLIP 2; none for DINOv2-L.

### 9.4 Against Commons set C (WB)

Registered contradiction rule (CARS196 only): SigLIP 2 is above MobileCLIP2 (+0.011 [0.007, 0.015])
and above PE-Core-G (+0.006 [0.003, 0.010]) while set C ranks both above SigLIP 2. Both differences are
about one point at the ceiling. Reading across all three sets, which is not registered but is what the
decision needs:

- Set C put DINOv2-L last by a wide margin (model-level P@10 0.28 against 0.65 to 0.80). Here it is
  0.825 / 0.892 / 0.576 Recall@1, below the CLIP-type models on CARS196 and SOP, **above PE-Core-L,
  SigLIP 2 and MobileCLIP2 on CUB**. The collapse on set C is not a property of DINOv2-L; it
  comes from the crops, the labels or the 224-pixel input of that setup.
- Set C had MobileCLIP2 and SigLIP 2 0.08 to 0.11 P@10 above PE-Core-L. Here they are level with PE-Core-L
  on CARS196 and 0.12 Recall@1 **below** it on CUB (both) and, for MobileCLIP2, on SOP. B1 fails for both.
- Set C had PE-Core-G first (0.795), PE-Core-L fourth. Here PE-Core-G is first on CUB and SOP and
  equal on CARS196, with a smaller margin over PE-Core-L (+0.01 to +0.04).
- Letterbox gave +0.04 to +0.11 P@10 on set C crops; on whole images it gives at most +0.019.

Implication: the set C ranking (small encoders ahead of PE-Core-L) does not reproduce on labels written
by the dataset authors, so it should not drive the embedding decision.

### 9.5 Cost (3 repetitions each, `bench.json`; encoder only, batch 32, fp16, shared GPU at load 79)

| Model | Images/s | Peak memory MiB | Dim | Re-embed 1M (h) |
|---|---|---|---|---|
| MobileCLIP2-S2 | 815 | 629 | 512 | 0.34 |
| DINOv2-L | 373 | 1,451 | 1024 | 0.74 |
| SigLIP 2 | 314 | 1,982 | 1152 | 0.88 |
| PE-Core-L | 133 | 1,835 | 1024 | 2.1 |
| PE-Core-G | 17 | 9,170 | 1280 | 15.9 |

Pipeline timing (3 repetitions, PE-Core-L, indicative): k-means 0.4 to 0.9 s on CARS196 and CUB (CPU), 18 to
22 s on SOP (GPU, K = 11,316); average-linkage dendrogram 3.2 to 3.5 s (8,131 rows), 2.0 s (5,924), 14 to
18 s (SOP subset).

### 9.6 Findings and what is still unknown

1. No challenger passes B1. PE-Core-L stays the production embedding on this evidence. PE-Core-G is
   better on all three sets (+0.01 to +0.04 Recall@1), below the 0.03 margin on CARS196, at about
   8 times the encode cost and 5 times the memory.
2. The set C ranking does not hold on standard benchmarks (9.4).
3. Preprocessing: letterbox is not worth a change for whole images.
4. Our pipeline: average linkage keeps precision 0.9 and above at pair recall 0.05 to 0.38 where a cut
   transfers; it does not recover the classes well at K clusters.
5. Not established: the registered sanity check was not met (9.1), CARS196 is near its ceiling and may
   be contaminated by pre-training data; the timm image weights used here are assumed to be the image
   tower of the checkpoints whose text towers exist, and text-search quality was not measured; all
   sets are whole images, none are detector crops of cars from street photos; VMMRdb and faces were not run;
   MIEB per-dataset retrieval figures and the Hyp-ViT table were not re-opened at the source (the
   arXiv listing gave no table and the PDF exceeded the fetch limit), so the literature references stay
   "as stated in the survey"; whether a second DINOv2-L vector (FG-5) pays was not evaluated.

### Reproduce

```bash
# 1. data check (read-only on the dataset store)
python -I scripts/datasets/prepare_labelled_benchmark.py verify "$WB2_DATA_ROOT/stanford_cars"
# 2. embeddings, research venv, GPU 2 (set WB2_DATA_ROOT, WB2_CACHE, FG_MOBILECLIP_CKPT)
flock $SCRATCH/locks/gpu2.lock nice -n 10 env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
  $VENV/bin/python -m scripts.research.standard_benchmarks.embed_run --dataset cars196 \
  --models s_pe_core s_pe_core_g s_mobileclip2 s_siglip2 s_dinov2 --variants centre letterbox
# 3. scoring, then the bars
$VENV/bin/python -m scripts.research.standard_benchmarks.score_run score --dataset cars196 --out RES --device cpu
$VENV/bin/python -m scripts.research.standard_benchmarks.score_run ahc --dataset sop --out RES   # SOP: score with --skip-ahc
$VENV/bin/python -m scripts.research.standard_benchmarks.score_run decide --out RES
# 4. throughput
$VENV/bin/python -m scripts.research.fine_grained.bench --out bench.json --models s_pe_core s_pe_core_g s_mobileclip2 s_siglip2 s_dinov2
```
