# Class-partitioned clustering: results of the CPU experiments X0 to X7 and X9

Measured 2026-10-29 on branch `feat/cluster-w0c-experiments`, code at commit `41396e0b` (the cut-transfer
run at `4135401a`, which differs only in how two cut heights are compared). CPU only, 48 cores, shared
host: the one-minute load average was 15 to 75 across the runs, and the load at the start and end of every
run is stored in its JSON (`load_avg_start`, `load_avg_end`). Nothing here is a timing except X7, and its
load is stated there.

These are the experiments of [cluster_config_decision.md](cluster_config_decision.md) section 6, run as
specified on the pinned 3,000-item set (944 images, 3,961 crops, 2,950 labelled, 86 classes; public COCO and
Open Images data; the vectors, labels and stored partition exported by the earlier baseline run, see
[cluster_config_baseline.md](cluster_config_baseline.md)). The cut is always chosen on the manifest's
`calibration` half and scored on `heldout` (fold 1, the acceptance direction), then the reverse (fold 2).
X8 is not run (it needs the judge). No product code was changed.

## Verdicts

| ID | Bar (decision note) | Measured | Verdict |
|---|---|---|---|
| X0 | Agreement reproduces 0.820 / 0.810 / 0.812 | 0.8203 (all 2,950), 0.8097 (confidence under 0.7), 0.8119 (under 0.5) | **PASS** |
| X1 | Committed arms reproduce section 2 to three decimals | 34 of 37 quoted cells match; 3 non-person cells differ by 0.001 to 0.003, explained below | **PASS**, with a defect in the note's table |
| X2 | All product bars at the target-0.1 cut, both folds | BCubed recall 0.357 (fold 1) and 0.352 (fold 2) against 0.37; fold 2 also has stratum `rare` 0.0235 below IVF; every other bar passes | **FAIL** |
| X3 | Gain >= 0.02 BCubed precision at matched recall, both folds, no stratum losing over 0.02 | whitening -0.033 / -0.065, k-reciprocal -0.014 (fold 2 unmatched, best case -0.034), DBA -0.011 (fold 2 unmatched, best case -0.007) | **FAIL**, all three arms: closed for crops |
| X4 | Novelty AUROC >= 0.90 on every judged key; flag precision >= 0.80; suggestion precision >= 0.90 | sheep AUROC 0.838 / 0.851; no calibration quantile reaches flag precision 0.80 (best 0.733 / 0.761); top-5 / 10 / 20 suggestion precision 0.8 / 0.5 / 0.35 and 0.6 / 0.3 / 0.15 | **FAIL**: neither the state nor the stage is justified by these bars |
| X5 transfer | COCO held-out BCubed precision >= 0.90 at the transferred cut | 0.935 (n = 233); 0.934 without images of the tuning half (n = 229) | **PASS**, weak (the bar is also met by IVF, 0.931, and nearly by the detector alone, 0.897) |
| X5 sample size | Smallest per-key count whose cut is within 0.02 of the full cut in 9 of 10 draws | 0 of 10 at every count (5, 10, 20, 50 per key); the cuts sit at 0.20 to 0.26 against 0.35 | **FAIL** as specified; a random draw of 400 labelled items gives 10 of 10 (supplementary) |
| X6 | Accepted assignments' label precision >= 0.95 | pinned seed 0.946 / 0.948 (centroid), 0.949 / no margin reaches 0.95 (best member); median over 5 seeds x 2 folds 0.946 | **FAIL**, marginal (within sampling noise; 0.97 to 0.98 on labels the detector can name) |
| X7 | Largest key: dense AHC in under 30 min and 32 GB, else sparse within 0.01 BCubed | dense, 40,016-crop key: 172 s, 12.7 GiB peak; sparse variant as specified: BCubed precision -0.064, recall +0.293 against dense | dense fits: **PASS**; sparse variant as specified **FAIL** (not needed) |
| X9 | X2 bars hold on the pinned split; spread reported; reopen if any split fails a bar by more than 0.03 | X2 does not hold on the pinned split (X2 above). Four splits, fold 1: no bar missed by more than 0.03. Fold 2: two cells miss by more than 0.03 | **FAIL** on the strict reading (reopen rule fires on fold 2); not on fold 1 only |

What to take from the table:

- **The method is sound on precision and short on recall at the specified cut.** At the target-0.1 cut the
  within-class arm clears every precision bar (BCubed precision 0.943 overall, 0.925 non-person, 0.968 low
  resolution) at 1.9 times IVF's BCubed recall. It misses the 0.37 recall bar by 0.013, as the decision note
  itself predicted. The cut chosen for target recall 0.2 (0.381) clears the recall bar and fails only the
  strictest stratum rule (see X2). The decision note's bar set is not met as written; whether to move the
  recall bar or the operating point is the owner's call, and this note does not change the plan.
- **X3 closes the representation arms for crops. X4 does not justify the novelty state or the suggestion stage
  at these thresholds.**
- **The default cut does not transfer to set P** (X7 side result): at 0.346 the largest key holds 77 % of its
  crops in one cluster.

## X0: names compared with the product rule

`run.py` compared the detector's registry slug with the COCO-form label as raw strings. It now goes through
`normalize_class_name` (`src/utils/class_names.py`, the product's one name rule) via two helpers in
`scripts/research/cluster_diag/dataset.py` (`detector_keys`, `class_agreement`); the partition key in E5 uses
the same names. The test fails on the old code (agreement 0.0 for `traffic_light` against `traffic light`).
Re-run on the bundle: 0.8203 / 0.8097 / 0.8119; raw output `artifacts_local/cluster_search/w0c/x0/e5.json`. The
restated figures are in [cluster_config_baseline.md](cluster_config_baseline.md). The COCO sample's 11 class
names are single words, so its figures are unaffected.

## X1: the committed arms and the section 2 table

`export_partition.py` has two new arms (`detector_class`, `ahc_average_within_class`); `class_prior_run.py
partition` computes every row of section 2 in process. Scored with `scripts/bench/cluster_eval.py` on the
exported partitions (three repetitions, identical; the same numbers as the in-process scorer):

| Held-out half, fold 1 | Pair P / R | BCubed P / R / F | AMI | Non-person BCubed P / R | Low-res BCubed P / R | Clusters |
|---|---|---|---|---|---|---|
| IVF now (refit) | 0.734 / 0.029 | 0.792 / 0.187 / 0.302 | 0.416 | 0.814 / 0.252 | 0.803 / 0.209 | 374 |
| Detector class only (no clustering) | 0.906 / 0.960 | 0.786 / 0.870 / 0.826 | 0.864 | 0.711 / 0.825 | 0.957 / 0.981 | 69 |
| All crops alone (trivial) | - / 0.000 | 1.000 / 0.052 / 0.098 | 0.000 | 1.000 / 0.070 | 1.000 / 0.096 | 1,494 |
| One cluster (trivial) | 0.092 / 1.000 | 0.093 / 1.000 / 0.170 | 0.000 | 0.033 / 1.000 | 0.157 / 1.000 | 1 |
| Global average linkage, cut 0.307 | 0.399 / 0.073 | 0.741 / 0.285 / 0.412 | 0.456 | 0.763 / 0.380 | 0.527 / 0.396 | 347 |
| Within class, cut 0.346 (targets 0.05 and 0.1) | 0.924 / 0.129 | 0.943 / 0.357 / 0.518 | 0.622 | 0.925 / 0.461 | 0.968 / 0.606 | 359 |
| Within class, cut 0.381 (target 0.2) | 0.919 / 0.241 | 0.920 / 0.457 / 0.611 | 0.678 | 0.895 / 0.561 | 0.965 / 0.849 | 285 |
| Within class, cut 0.425 (target 0.3) | 0.906 / 0.312 | 0.889 / 0.533 / 0.667 | 0.725 | 0.852 / 0.641 | 0.965 / 0.959 | 213 |

Person crops carry 81.4 % of the held-out same-label pairs (81.6 % in fold 2). The detector-only row is the
person-dominance check: it has the best pair scores and BCubed recall and the worst non-person BCubed
precision (0.711), so pair metrics alone would reward it; BCubed precision on non-person crops is where the
clustering shows (0.925 against 0.711 at 1.8 times IVF's non-person recall). The all-alone and one-cluster rows
bound every metric.

**Defect in the decision note's section 2 table.** Three non-person cells (detector-only recall, global
average-linkage precision and recall) were computed on crops that are neither labelled nor detected as
person; the other non-person cells use labelled non-person crops only. Under the second definition, used
here for every row, they are 0.825 (note 0.828), 0.763 (0.762) and 0.380 (0.381). Computed under the first
definition the detector-only recall is 0.828 again. Nothing else in the table moves; the three cells are the
whole of the 34-of-37 count.

## X2: the product bars at the target-0.1 cut

| Bar | Fold 1 (cut 0.346) | Fold 2 (cut 0.361) |
|---|---|---|
| BCubed precision >= 0.90, all | 0.943 pass | 0.953 pass |
| BCubed precision >= 0.88, non-person | 0.925 pass | 0.946 pass |
| BCubed precision >= 0.85, low resolution | 0.968 pass | 0.966 pass |
| BCubed recall >= 0.37 (twice IVF's 0.187) | **0.357 fail** (IVF 0.187) | **0.352 fail** (IVF 0.165) |
| Pair recall >= 3 x 0.022 = 0.066 | 0.129 pass | 0.181 pass |
| No stratum's BCubed precision below IVF's | pass (worst gap 0.000, stratum `other`) | **fail** (`rare` 0.958 against IVF 0.981, gap -0.0235) |
| Permutation ARI 1.000 | 1.000 pass | 1.000 pass |
| Protected items moved 0 | 0 of 0 | 0 of 0 |

Per stratum at the fold-1 cut (within class / IVF BCubed precision): low resolution 0.968 / 0.803, mixed
scene 0.978 / 0.835, near-duplicate 0.972 / 0.935, rare 0.986 / 0.956, similar classes 0.934 / 0.888,
elongated over 2:1 0.978 / 0.895, side under 32 px 0.968 / 0.801. At the target-0.2 cut (0.381) fold 1 clears
the recall bar (0.457) and fails only the stratum rule (`rare` 0.0315 below IVF); at target 0.3 (0.425)
BCubed precision falls to 0.889 and 0.852 non-person, below two bars.

Notes on the bars:

- **Pair recall "3 x 0.022".** The held-out IVF pair recall is 0.029 (refit) and 0.027 (stored partition);
  0.022 matches neither, so the literal 0.066 is used. Three times 0.029 is 0.087, which 0.129 also clears.
- **Protected items.** The labels carry no protected item (`protected` is false on all 2,950), so "0 moved" is
  vacuously true here. The lock rule is a property of the product writers (wave WC), not of this harness.
- **The stratum rule is the strictest bar.** Strata hold 8 to 27 crops (`other`) up to 500; a rule of "never
  below IVF" fires on sampling noise as well as on real loss. It is reported as written.

## X3: representation arms inside the class

Each arm replaces the cosine distance of the per-class dendrograms (whitening fitted on 40,000 set P
vectors, used only for the fit; the re-ranking arms computed inside each class). Cuts: raw at the
calibration-chosen height, the arm at the height whose calibration BCubed recall is closest to raw's, scored
on the other half. At the target-0.1 operating point:

| Arm | Fold 1: raw P / R, arm P / R, gain | Fold 2: raw P / R, arm P / R, gain | Worst stratum loss (fold 1 / 2) | Verdict |
|---|---|---|---|---|
| PCA whitening 256 | 0.943 / 0.357, 0.911 / 0.366, **-0.033** | 0.952 / 0.362, 0.887 / 0.381, **-0.065** | 0.133 / 0.240 (`rare`) | FAIL |
| k-reciprocal | 0.943 / 0.357, 0.930 / 0.354, **-0.014** | 0.952 / 0.362, 0.891 / 0.389, -0.061 (unmatched) | 0.077 / 0.182 | FAIL |
| DBA | 0.943 / 0.357, 0.932 / 0.356, **-0.011** | 0.952 / 0.362, 0.936 / 0.388, -0.015 (unmatched) | 0.031 / 0.056 | FAIL |

An unmatched cell (held-out recall more than 0.02 from raw's) is not read as a pass. The best case over
every cut of the arm, at the held-out recall nearest raw's, is also negative in every cell of the table
above and in 23 of the 24 cells overall (the exception, +0.012, is DBA at target 0.3 in fold 2, still under the 0.02 bar). No arm gains precision at
matched recall, so all three are closed for crops, as the note's rule says. The oracle gain that
k-reciprocal and DBA showed on the global frontier (baseline note, oracle ceilings) does not carry inside the
class.

## X4: novelty and cross-class suggestions

Setup (the note leaves details open): a crop's novelty is 1 minus its mean cosine to its 10 nearest validated
examples of the class it is filed under (the fitting half's labelled crops whose label equals the class,
other source images only); a cluster's novelty is the mean over its members; the flag threshold per class
is a quantile of the novelty of its own validated examples, the lowest quantile whose flag precision on the
fitting half reaches 0.80. "Novel" is an open-set label (one the detector has no class for). Cut: the X2 cut.

| | Fold 1 | Fold 2 |
|---|---|---|
| Judged keys (>= 20 examples, >= 10 open-set crops), cluster-score AUROC | bird 0.997, cow 0.977, **sheep 0.838** | cat 0.959, horse 1.000, **sheep 0.851** |
| Same keys, crop-score AUROC (the note's section 2c definition) | bird 0.986, cow 0.941, sheep 0.919 | cat 0.955, horse 0.986, sheep 0.887 |
| Flag precision on the fitting half at q = 0.5 / 0.75 / 0.9 / 0.95 / 0.99 | 0.19 / 0.29 / 0.43 / 0.57 / 0.73 | 0.19 / 0.30 / 0.46 / 0.58 / 0.76 |
| Flag precision / recall on the other half at those quantiles (thresholds from the fit half) | 0.24 / 0.36 / 0.56 / 0.66 / 0.87 and recall 0.55 / 0.55 / 0.43 / 0.42 / 0.38 | 0.16 / 0.24 / 0.32 / 0.45 / 0.57 and recall 0.37 / 0.37 / 0.32 / 0.32 / 0.25 |
| Cross-key suggestions, precision of the top 5 / 10 / 20 | 0.80 / 0.50 / 0.35 (109 evaluable) | 0.60 / 0.30 / 0.15 (70 evaluable) |

No quantile reaches flag precision 0.80 on the fitting half, so no threshold is chosen on calibration and the
bar fails. At q = 0.99 fold 1 scores 0.87 on the other half, fold 2 only 0.57: not stable. The suggestions
were evaluated at q = 0.99 as an exploratory row (no calibrated threshold exists); top-ranked precision is
far under 0.90 and no suggestion threshold reaches 0.90 on the fitting half, so the merge effect on open-set
pair recall was not computed. The note's per-class AUROCs (bird 0.986, cow 0.941, sheep 0.919) reproduce at
crop level. A cluster-level score is worse than the crop-level one for sheep, because sheep clusters mix
true sheep with the animals the detector files under sheep. Samples are small (9 to 55 open-set crops per
key); read the direction, not the digits.

## X5: cut transfer and sample size

**Transfer.** The cut chosen on the 3,000-item calibration half (0.346 at target 0.1) applied unchanged to
the 200-image COCO project (same embedding model), held-out half of `splits/coco_sample_200.json`:

| Cut (target) | BCubed P / R | Pair P / R | n | Without the 3 images that are also in the tuning half |
|---|---|---|---|---|
| 0.346 (0.05 and 0.1) | **0.935** / 0.410 | 0.918 / 0.361 | 233 | 0.934 / 0.414 (229) |
| 0.381 (0.2) | 0.933 / 0.593 | 0.916 / 0.620 | 233 | 0.932 / 0.602 |
| 0.425 (0.3) | 0.929 / 0.730 | 0.916 / 0.770 | 233 | 0.928 / 0.734 |

Beside them on the same crops: IVF refit 0.931 / 0.222 BCubed, detector class only 0.897 / 0.874. The bar
(0.90) is met, but the COCO sample (11 single-word classes, no person class) is the set whose precision
misled the earlier global-linkage conclusion, and IVF clears the same bar. The informative figure is recall:
1.8 times IVF's at the same precision. The manifest lists 179 of the project's 200 images; the 21 others (87
labelled crops) are clustered but not scored. 116 of the 179 listed images are also in the 3,000-item set;
three held-out images are in its calibration half, hence the second column.

**Sample size, as specified.** Subsampling the calibration half to 5, 10, 20 and 50 labelled crops per
detector class and re-choosing the cut at target recall 0.1: the full-calibration cut is 0.350 (dense grid);
the subsample cuts are 0.20 to 0.41 at 5 per key, 0.205 to 0.26 at 10, 0.235 to 0.26 at 20 and exactly 0.235
at 50. **0 of 10 draws are within 0.02 at every count.** The cause is the protocol, not the sample size: capping
each class removes the person class's 81 % share of the pairs, so "pair recall 0.1" on the sample is a different
target from "pair recall 0.1" on the half. Per-key capping therefore cannot calibrate this cut.

**Sample size, supplementary (not in the note).** Drawing labelled crops at random from the calibration half
(keeping the class mix):

| Labelled crops drawn | 100 | 200 | 400 | 800 |
|---|---|---|---|---|
| Draws within 0.02 of 0.350 | 4 of 10 | 8 of 10 | **10 of 10** | 10 of 10 |
| Range of the chosen cut | 0.30 to 0.595 | 0.325 to 0.45 | 0.345 to 0.37 | 0.35 to 0.37 |

So about 400 validated items drawn from the project's real class mix reproduce the cut here (27 % of the
calibration half). A project that cannot supply the mix would need a different calibration target.

## X6: late arrivals

20 % of the images (seeded by a hash in its own namespace, so it is independent of the evaluation split that
ranks the same images by the same seed) are held back; the rest are clustered inside each class at the X2
cut. Each arrival goes to the nearest cluster of its own key and abstains beyond the cut or inside a margin
(0, 0.01, 0.02, 0.05, 0.1), chosen on the arrivals of one image half and scored on the other. Label
precision = the arrival's label equals the plurality label of the labelled base members of the cluster it
joins. Pinned seed (849 arrivals), margin chosen on one half, scored on the other:

| Method | Fold 1 margin: accepted, precision (nameable labels), abstain | Fold 2 margin: accepted, precision (nameable), abstain |
|---|---|---|
| Centroid | 0: 307, **0.946** (0.977), 10 % | 0.05: 177, **0.948** (0.984), 41 % |
| Best member | 0: 303, **0.949** (0.977), 11 % | none reaches 0.95 on the fitting half |
| Mean member (extra) | 0.1: 84, 0.947 (0.966), 77 % | 0: 244, 0.926 (0.944), 20 % |

Over 5 seeds and 2 folds the median precision of the chosen setting is 0.946 (centroid, 5 of 10 settings
exist), 0.95 (best member, 4 of 10) and 0.92 (mean member, 6 of 10); the bar is met in 0, 2 and 1 of those
cells. The shortfall at the pinned seed is 0.001 to 0.004 on about 300 judged arrivals (a standard error of
about 0.013), so the honest reading is "at the bar, not clearly above". An arrival whose label is open-set
cannot match a cluster the detector class decided, so the figure on nameable labels (0.97 to 0.98) is the
cleaner measure of the assignment rule; the bar as written uses all labels.

ARI of the incremental partition against a full refit (reported, no bar): 0.54 to 0.66 over all crops. The
same refit differs from the base clustering of the surviving 80 % of the crops with ARI **0.647** on those
crops alone at the pinned seed (0.65 to 0.70 over the seeds), before any arrival is assigned: a fixed-cut average-linkage partition moves when 20 % of the
data is removed, so a refit does not reproduce an earlier partition and an incremental run cannot be
expected to.

## X7: scale on real vectors (set P, numbers only)

Set P project, 91,184 crops, 37 detector keys; the largest keys are 40,016 (person), 32,054 (motorcycle),
14,009 (car) and 2,769 (truck) crops; the other 33 keys hold 2,336 together. Read-only export, vectors kept in
`artifacts_local/` only.

| Run (single process; host load at start and end in the note below) | Fit time | Peak RSS (GiB) | Clusters at cut 0.346 | Largest cluster |
|---|---|---|---|---|
| Dense average linkage, person key (40,016 crops) | **172 s** | **12.7** | 136 | 76.9 % of the key |
| Dense average linkage, motorcycle key (32,054 crops) | 68 s | 8.4 | 24 | 95.3 % of the key |
| Sparse (mutual 30-NN, scikit-learn), person key | 574 s (earlier run: 692 s) | 14.2 | 62 | 99.3 % of the key |

Host load average at the start and end of each run: person dense 42 and 43, motorcycle dense 40 and 20,
sparse 37 and 11 (one minute, 48 cores). Each is a single run, so the times carry that noise. The dense
method passes its own bar by a wide margin (30 min and 32 GB). The sparse variant as specified
does not reproduce the 3,000-item row: at the fold-1 cut, mutual 30-NN gives BCubed precision 0.879 against
0.943 and recall 0.650 against 0.357 (mutual 10-NN: 0.831 and 0.701); ARI against the dense partition 0.27. The
scikit-learn connectivity-constrained average linkage averages only the edges that exist in the graph, so
missing pairs count as close and clusters over-merge. A sparse method that is a true average linkage needs
missing pairs imputed (for example at the cut) and is not what was measured. It is not needed at this size.

**Side result: the public-set cut does not transfer to set P.** On the same dendrograms, the share of the key
in its largest cluster at other cut heights:

| Cut | 0.346 | 0.30 | 0.25 | 0.20 | 0.15 | 0.10 | 0.05 |
|---|---|---|---|---|---|---|---|
| Person: clusters / largest share | 136 / 76.9 % | 334 / 47.5 % | 961 / 21.1 % | 2,617 / 7.7 % | 6,961 / 2.0 % | 16,739 / 0.2 % | 33,329 / 0.0 % |
| Motorcycle: clusters / largest share | 24 / 95.3 % | 55 / 49.9 % | 182 / 41.6 % | 606 / 16.8 % | 2,253 / 6.5 % | 7,874 / 0.6 % | 22,766 / 0.04 % |

Set P's vectors are tighter (mean random-pair cosine 0.63 against 0.46 on this set), as the decision note
warns. A default cut calibrated here would violate X8's "largest cluster at most 10 % of its key" rule on
the two largest keys unless the judge rates those clusters pure; the person key would need a cut near 0.20
and the motorcycle key near 0.15 to 0.20. This is a size statistic, not a quality statement.

## X9: further splits

Three more seeded splits of the pin by source image (`eval_split.py make-pin` logic, seeds 20261030,
20261031, 20261101; the pinned split stays the acceptance split), the cut re-chosen on each calibration half
at target 0.1. Fold 1 (cut chosen on calibration, scored on the held-out half) and fold 2 (reverse):

| Split | Fold | Cut | BCubed P | Non-person P | Low-res P | BCubed R (IVF R) | Pair R | Worst stratum gap to IVF | Bars missed |
|---|---|---|---|---|---|---|---|---|---|
| pinned | 1 | 0.346 | 0.943 | 0.925 | 0.968 | 0.357 (0.187) | 0.129 | 0.0000 | BCubed recall |
| pinned | 2 | 0.361 | 0.953 | 0.946 | 0.966 | 0.352 (0.165) | 0.181 | -0.0235 (`rare`) | BCubed recall, stratum |
| 20261030 | 1 | 0.361 | 0.952 | 0.942 | 0.967 | 0.382 (0.176) | 0.201 | 0.0000 | none |
| 20261030 | 2 | 0.361 | 0.934 | 0.914 | 0.974 | 0.377 (0.174) | 0.183 | -0.0232 (`rare`) | stratum |
| 20261031 | 1 | 0.361 | 0.946 | 0.935 | 0.987 | 0.377 (0.180) | 0.189 | -0.0004 (`rare`) | stratum |
| 20261031 | 2 | 0.361 | 0.939 | 0.921 | 0.950 | 0.378 (0.168) | 0.196 | **-0.0509** (`near_duplicate`) | stratum |
| 20261101 | 1 | 0.361 | 0.947 | 0.932 | 0.965 | 0.381 (0.177) | 0.190 | -0.0004 (`rare`) | stratum |
| 20261101 | 2 | 0.331 | 0.948 | 0.938 | 0.966 | **0.304** (0.164) | 0.103 | 0.0000 | BCubed recall |

Spread over the four splits, fold 1: BCubed precision 0.943 to 0.952, non-person 0.925 to 0.942, low
resolution 0.965 to 0.987, BCubed recall 0.357 to 0.382, pair recall 0.129 to 0.201. Fold 2: BCubed
precision 0.934 to 0.953, BCubed recall 0.304 to 0.378. The precision bars hold on every split and fold by
0.034 or more; permutation ARI is 1.000 everywhere. The recall bar is met on three of four splits in fold 1
(pinned misses by 0.013) and the stratum rule fails by 0.0004 to 0.05 on `rare` and `near_duplicate`.

Reopen rule ("any split fails a bar by more than 0.03"): in fold 1, the acceptance direction, no split does.
In fold 2, two cells do: stratum `near_duplicate` on seed 20261031 (0.051 below IVF) and BCubed recall on
seed 20261101 (0.304, the cut chosen on that half was 0.331 where the others were 0.361). The note does not
say whether fold 2 counts; on the strict reading the rule fires.

## Problems with the note, and what was unclear

1. Section 2's non-person cells mix two definitions (X1).
2. X2's pair-recall bar "3 x 0.022" matches no measured IVF value (X2).
3. X2's BCubed recall bar was already known to fail at the cut it names (0.357 against 0.37); X9 then says the
   X2 bars hold on the pinned split, which they do not.
4. The per-key subsampling protocol of X5 changes the target it calibrates (X5); the 200-image COCO manifest
   omits 21 images and shares most of its images with the 3,000-item set.
5. X7's sparse variant is under-specified: scikit-learn's connectivity-constrained average linkage is not
   average linkage over the graph's missing pairs (X7).
6. "No stratum below IVF" is exact-comparison on strata of 8 to 500 crops; two near-zero gaps (-0.0004) are
   float-level ties in effect.
7. X4 leaves the unit of the novelty score (crop or cluster), the flag quantile and the meaning of "novel"
   open; the choices are stated in X4.
8. Seeding the late arrivals of X6 with the evaluation split's seed puts all arrivals in one half; fixed here
   by a separate hash namespace.
9. The note's X8 bar and the 10 % rule need a per-key cut on set P (X7 side result), not a global default.

## Limits

One public set, one pinned split plus three seeded ones (they share images, so they are not independent), 1,494
held-out labelled crops. Person crops are 28 % of the labelled crops and 81 % of the same-label pairs.
The detector is trained on COCO's label space. The X4 samples are tens of crops per key. The COCO transfer set
has 11 classes and overlaps the 3,000-item set. X7 timings come from a shared host at load 40 to 45 and are
single runs. Unlabelled crops (1,011) are clustered but never scored. Set P figures are counts and cluster
sizes only. Not run: X8 (needs the judge). Not measured: faces, whole images, region boxes, fine-grained
groups.

## Reproduce

All commands run from the repository root with the research environment (`PYTHONPATH` must reach `igraph`
and `leidenalg` only for the older E1 arms; none of these commands needs them). `<out>` is the exported
bundle of [cluster_config_baseline.md](cluster_config_baseline.md) (`vecs.npy`, `vecs.meta.jsonl`,
`eval/labels.jsonl`, `eval/partition.jsonl`).

```bash
C="--bundle <out>/vecs --labels <out>/eval/labels.jsonl --coco-gt <dir>/coco_gt.json \
   --split-manifest scripts/datasets/manifests/splits/cluster_eval_3000.json"
R=".venv/bin/python scripts/research/cluster_diag/class_prior_run.py"
.venv/bin/python scripts/research/cluster_diag/run.py e5 --bundle <out>/vecs --labels <out>/eval/labels.jsonl \
    --coco-gt <dir>/coco_gt.json --split-manifest scripts/datasets/manifests/splits/cluster_eval_3000.json \
    --confidence-cuts 1.01 0.7 0.5 --out x0/e5.json                                           # X0
$R partition $C --stored-partition <out>/eval/partition.jsonl --out partition.json           # X1, X2
$R splits $C --stored-partition <out>/eval/partition.jsonl \
    --pin scripts/datasets/manifests/cluster_eval_3000.json --out splits.json                # X9
$R representations $C --fit-pool <independent 40,000-crop vectors>.npy --out repr.json       # X3
$R novelty $C --out novelty.json                                                             # X4
$R transfer $C --second-bundle <coco200>/vecs --second-labels <coco200>/labels.jsonl \
    --second-coco-gt <coco200>/coco_gt.json \
    --second-split-manifest scripts/datasets/manifests/splits/coco_sample_200.json --out transfer.json   # X5
$R streaming $C --out streaming.json                                                         # X6
$R scale-check $C --out scale_check.json                                                     # X7, sparse against dense
$R scale-time --bundle <set P export>/vecs --key person --mode dense --cut 0.346 \
    --sweep 0.30 0.25 0.20 0.15 0.10 0.05 --out person_dense.json                            # X7, cost
.venv/bin/python scripts/research/cluster_diag/export_partition.py --bundle <out>/vecs \
    --arm ahc_average_within_class --param 0.3462035340847838 --out within.partition.jsonl
.venv/bin/python scripts/bench/cluster_eval.py score --labels <out>/eval/labels.jsonl \
    --partition within.partition.jsonl --split-manifest \
    scripts/datasets/manifests/splits/cluster_eval_3000.json --split heldout --out within.json   # W0 scorecard
.venv/bin/python -m pytest tests/research_scripts -q --no-cov                                # 114 tests
```

Raw outputs, one JSON per run with the commit and the host load: `artifacts_local/cluster_search/w0c/`
(gitignored): `x0/`, `partition/`, `splits/`, `representations/`, `novelty/`, `transfer/`, `streaming/`,
`scale-check/`, `scale_time/`, `x1_cli/` (scorecards, three repetitions each) and `old_runs/` (runs superseded
by code fixes during the wave, kept). The bundle inputs were copied from the earlier baseline run's
`artifacts_local` (read-only there).

## Tests

`tests/research_scripts/test_class_prior.py` and `test_class_prior_experiments.py` hold hand-computed cases
(the arithmetic is in the comment beside each expected value). Eleven mutations were applied to the
committed code in a scratch checkout, each run against the tests that cover it: sorting by id removed from the
permutation ARI, the BCubed recall bar lowered, the same-image exclusion removed from the novelty score,
raw-string name comparison, the farthest instead of the nearest recall match, the inverted margin test,
the distance floor removed, one extra crop per key in the subsample, a tie counted as a full win in the
AUROC, the key ignored in the within-key linkage, and the sign of the stratum gap flipped. Ten were killed by
the first run; the distance-floor mutation survived (the test used rows that do not round above 1) and the
test was then rewritten with rows that do, after which it fails on the mutation.
