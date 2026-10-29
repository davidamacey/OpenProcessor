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
