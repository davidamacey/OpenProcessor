# Cluster evaluation baselines

Two baselines of the cluster-quality harness (`scripts/bench/cluster_eval.py`,
`scripts/bench/cluster_eval_metrics.py`, `scripts/bench/cluster_eval_set.py`), both on the 0.6.0
image (#341). Each scores the partition the production pipeline (FAISS IVF, run by the refresh
worker) already produced; neither changed any clustering.

| | Baseline A: pinned 3,000-item set | Baseline B: 200-image COCO sample |
|---|---|---|
| Measured | 2026-10-28 and 2026-10-29 | 2026-10-28 |
| Images, crops | 944 images, 3,961 crops, 2,950 labelled (86 classes) | 200 images, 1,364 crops, 539 labelled (10 classes) |
| Source | COCO val2017 and Open Images v7, pinned by manifest | COCO val2017 public sample |
| Reproduce | fetcher plus a running stack | committed files, offline |

The two are not comparable: the image sets, class mixes (86 classes against 10), crop counts and
cluster sizes (about 8 crops per cluster against about 4) differ, and so do the strata. Compare a
future run only with the baseline that has the same set and pipeline.

## Baseline A: the pinned 3,000-item set

The evaluation set is pinned by `scripts/datasets/manifests/cluster_eval_3000.json` and fetched by
`scripts/datasets/fetch_cluster_eval.py`.

### A.1 The evaluation set

Public images only. 744 base images plus 200 near-duplicate variants, seed 20261028:

| Source | Images | Licence |
|---|---|---|
| COCO val2017 (Flickr) | 384 | CC BY (Attribution) 254, CC BY-SA 129, no known copyright restrictions 1 |
| Open Images v7 validation and test | 360 | CC BY 2.0 |

The per-image licence, author, source URL and SHA-256 are in the manifest, and the fetcher writes
`ATTRIBUTION.csv` next to the images. The images themselves are not redistributed.

Image groups, each drawn with a seeded sample: visually similar classes (largest box is car,
truck, bus, train; bicycle, motorcycle; cat, dog; horse, cow, sheep), rare classes (the 12 COCO
classes with the fewest train2017 instances plus 12 Open Images animals the COCO label set does
not have), many small boxes, and multi-class scenes. Quotas are in the manifest; several groups
came in below quota because the licence-filtered val2017 pool is small (for example hair drier
has no eligible image).

Near-duplicates: 100 source images (at most three annotated boxes each, so the stratum stays
near 13 percent), each given two seeded variants (crop margins up to 6 percent per side, JPEG
quality 55 to 90). The variants' boxes are the source boxes moved into the cropped frame, boxes
less than 60 percent kept are dropped. The source image and its variants are all flagged
`near_duplicate`.

Ground truth is independent of the pipeline: the datasets' own boxes (COCO instances, Open Images
confident non-group non-depiction boxes) written as `coco_gt.json`. A crop of the product's
detector gets the class of the annotation it overlaps most (IoU at least 0.5, crowd ignored);
a crop that matches no annotation is not labelled. Class identity is the name (Open Images
Cattle is `cow`). Strata, first match wins: near-duplicate, low resolution (box side under 32 px),
rare (the manifest's pinned list, rare in COCO rather than in this sample), similar classes,
mixed scene, other.

Reproduce (the images and `artifacts_local/` stay out of git):

```bash
.venv/bin/python scripts/datasets/fetch_cluster_eval.py --out <dir> --cache-dir <cache> \
    --manifest scripts/datasets/manifests/cluster_eval_3000.json      # or --verify-only
# ingest <dir>/images into a throw-away project through the product (policy all), let the refresh
# worker cluster it, then
.venv/bin/python scripts/bench/cluster_eval_set.py build --api http://127.0.0.1:5603 \
    --project <slug> --coco-gt <dir>/coco_gt.json \
    --eval-manifest scripts/datasets/manifests/cluster_eval_3000.json --out-dir artifacts_local/cluster_eval/<slug>
.venv/bin/python scripts/bench/cluster_eval.py score --labels .../labels.jsonl \
    --partition .../partition.jsonl --out .../score.json
```

With an existing manifest the fetcher recomputes the selection and refuses to continue if it no
longer equals the pin, and it checks every downloaded base image and the annotations against the
pinned hashes.

### A.2 What was measured

Project `ev-clu3k-d` on the test stack: 944 images ingested through
`POST /ingest/bulk` with the default policy (all crops embedded), 3,961 crops, 3,961 embedded.
The refresh worker then ran its retrain job. That job is the IVF residual pass: spherical FAISS
k-means, 512 centroids, 50 iterations, 5 restarts, seed 42, CPU, then nearest-centroid
assignment of every crop; auto-promote and the VLM stage are off by default and AHC refine is an
on-demand route, so neither ran. 2,950 crops match an annotation and are labelled (86 classes,
485 clusters over them); the other 1,011 crops (objects of unannotated classes) are left out.
No crop is protected, so the moved-items count is not exercised.

### A.3 Results (2,950 labelled crops)

| Metric | All | Low res | Mixed scene | Near-duplicate | Rare | Similar classes | Other |
|---|---|---|---|---|---|---|---|
| Items | 2950 | 764 | 889 | 377 | 358 | 527 | 35 |
| Share of items | 100% | 25.9% | 30.1% | 12.8% | 12.1% | 17.9% | 1.2% |
| Clusters touched | 485 | 168 | 240 | 104 | 101 | 143 | 14 |
| Pair precision | 0.696 | 0.601 | 0.723 | 0.896 | 0.944 | 0.882 | 1.000 |
| False-merge rate | 0.304 | 0.400 | 0.277 | 0.104 | 0.056 | 0.118 | 0.000 |
| Pair recall | 0.022 | 0.042 | 0.013 | 0.244 | 0.278 | 0.094 | 0.282 |
| False-split rate | 0.978 | 0.958 | 0.987 | 0.756 | 0.723 | 0.906 | 0.718 |
| Correct-split recovery | 0.9990 | 0.9951 | 0.9974 | 0.9989 | 0.9991 | 0.9981 | 1.0000 |
| Items outside their cluster's majority class | 0.208 | 0.251 | 0.169 | 0.053 | 0.036 | 0.074 | 0.000 |
| Mixed clusters | 248 | 67 | 81 | 8 | 11 | 31 | 0 |
| Unassigned fraction | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Reading it: with 512 centroids over 3,961 crops the clusters are small (about 8 crops), so
same-class pairs are mostly split (false-split rate 0.98), the cheap error for curation. The
costly error, a false merge, is 30 percent of pairs overall and 40 percent among low-resolution
crops, and 21 percent of labelled crops sit in a cluster whose majority class is another class
(25 percent for low resolution). Near-duplicates and rare classes are the cleanest strata; the
near-duplicate pair recall is only 0.24, so about three quarters of the pairs of crops of one
object in jittered copies are still split. Strata are scored on their own items and the clusters
as they are, so they do not add up to the whole.

### A.4 Repeatability and run-to-run spread

Identical vectors, same seed (`scripts/bench/ivf_repeat.py`, read only on the project's stored
vectors, the production `IVFMethod`, centroids in a temporary directory, 3,961 crops): three fits
in the same row order gave identical partitions (ARI 1.0, 4.6 to 4.7 s each). The same fit on a
seeded permutation of the rows gave ARI 0.306 against the first, so the IVF pass is repeatable
on identical input in identical order but the grouping depends strongly on row order (k-means
initialisation draws from the rows). The product's job fetches the rows in an order the tool does not reproduce: its partition against the tool's sorted-order fit has ARI
0.353 over the 2,950 labelled crops.

End to end: the same 944 images were ingested again into a second throw-away project
(`ev-clu3k-e`) and clustered by the same worker. Of 3,961 crops 3,913 carry the same crop id in
both; their vectors agree to cosine 0.99986 or better (668 are bit-identical, largest absolute
component difference 0.0025, GPU encoder nondeterminism), and the ARI of the two partitions over the 2,913 labelled
crops present in both is 0.356, not an identical grouping.
Re-running `pipeline/auto_label/start` on a clustered project is a no-op (`no_residuals`, all
3,961 unchanged), so it cannot be used for this check. AHC is not part of the scheduled pass, so
its run-to-run spread is not measured here.

### A.5 Caveats

- Only labelled crops are scored (2,950 of 3,961). The 1,011 unlabelled crops share clusters
  with labelled ones, so a merge between an unlabelled object and a labelled one is invisible.
- Rare is defined by COCO frequency, plus 12 Open Images animals. The detector is the COCO
  class set, so an Open Images animal it has no class for may get a box of another class or none:
  doubling the Open Images rare group from 130 to 260 images (a different seeded draw) raised the
  crop count only from 3,764 to 3,961. Rare-stratum numbers describe the rare objects the
  detector boxes, not all of them.
- Image groups are balanced by image, not by crop; the crop strata come out at 12 to 30
  percent except `other` (1.2 percent, not a required stratum). Person is 34 percent of the
  low-resolution stratum and 58 percent of the mixed-scene stratum (28 percent of all labelled
  crops), so those two strata mostly measure how person crops cluster.
- The set is built from the datasets' ground truth, whose Open Images boxes are only complete
  for verified classes; an unannotated object that the detector boxes is unlabelled, not wrong.
- Crop ids are not bit-stable across ingests (3,913 of 3,961 shared), so cross-run comparisons
  use the shared labelled subset.
- Near-duplicate variants depend on the Pillow and libjpeg build; the manifest records their bytes
  but `--verify-only` checks only their presence. Base images are verified by SHA-256.

## Baseline B: the 200-image COCO sample (re-scored)

### B.1 What was measured

Project: 200 COCO val images (the public sample set, 20 images per class for 10 vehicle and
animal classes), ingested with the product's own detector and PE embeddings into a test stack,
1,364 crops, clustered by the refresh worker. Annotations: the COCO ground truth of the same
images (CC BY 4.0).

Ground truth is independent of the pipeline. A crop gets the class of the COCO annotation it
overlaps most (IoU at least 0.5, crowd boxes ignored). 539 of the 1,364 crops match; the other
825 are objects of classes the sample annotations do not carry (people and so on) and are left
out of the labels rather than guessed. Strata (first match wins): low resolution (box side under
32 px), rare class (least frequent third of the annotated classes), similar classes (car, truck,
bus, train; bicycle, motorcycle; cat, dog; horse, cow, sheep), mixed scene, other.

The scored partition, the labels and the second project's partition are committed in
`docs/benchmarks/cluster_eval/` (checksums in `SHA256SUMS`), so the numbers below reproduce from a clean
checkout with no stack. They were exported from the stack by `cluster_eval_set.py build`:

```bash
# Export (needs the stack that held the project; already done, the outputs are committed):
.venv/bin/python scripts/bench/cluster_eval_set.py build --api http://127.0.0.1:5603 \
    --project cocotest --coco-gt <sample dir>/coco_gt.json --out-dir <out>/cocotest
# Reproduce the table below from the committed files (offline, under a second):
cd docs/benchmarks/cluster_eval && sha256sum -c SHA256SUMS && cd -
.venv/bin/python scripts/bench/cluster_eval.py score \
    --labels docs/benchmarks/cluster_eval/cocotest/labels.jsonl \
    --partition docs/benchmarks/cluster_eval/cocotest/partition.jsonl \
    --second-partition docs/benchmarks/cluster_eval/default/partition.jsonl \
    --out /tmp/score.json     # same numbers as docs/benchmarks/cluster_eval/score.json
```

An earlier version of this note gave 134 clusters, pair precision 0.893, recall 0.079, low-res
precision 0.846, 32 mixed clusters and an ARI of 0.41. Those came from a scoring run on a partition
that the stack's refresh daemon changed afterwards; that run's partition and score file were not
kept, so they could not be reproduced and are withdrawn. The table below is re-scored from the
partition and labels committed here, which a second run of the same command reproduces exactly.

### B.2 Results (539 labelled crops, 128 clusters over them, 10 classes)

| Metric | All | Low res | Rare | Similar classes |
|---|---|---|---|---|
| Items | 539 | 112 | 113 | 314 |
| Pair precision | 0.905 | 0.945 | 0.924 | 0.971 |
| False-merge rate | 0.095 | 0.055 | 0.076 | 0.029 |
| Pair recall | 0.088 | 0.136 | 0.139 | 0.107 |
| False-split rate | 0.912 | 0.864 | 0.861 | 0.893 |
| Correct-split recovery | 0.9987 | 0.9965 | 0.9962 | 0.9993 |
| Items outside their cluster's majority class | 0.065 | 0.062 | 0.035 | 0.019 |
| Mixed clusters | 28 | 7 | 4 | 6 |
| Unassigned fraction | 0 | 0 | 0 | 0 |

Protected items (holdout, locked, validated): 0, so the moved-items count is not exercised on
this data.

Reading it: the clusters are small and fine grained (about 4 labelled crops each), so same-class
pairs are mostly split across clusters (false-split rate 0.91). That is the cheap error for
curation, because a split cluster only means more review. The costly error is the false merge:
about 6.5 percent of assigned crops sit in a cluster whose majority class is a different one (6.2
percent among low-resolution crops, 3.5 among rare classes, 1.9 among similar classes). A label
propagated through a cluster would be wrong for those. (The earlier version of this note called low
resolution the clear weak stratum, from the withdrawn numbers; on this re-scored partition its pair
precision is the second highest, so that reading is withdrawn here. Baseline A, a larger and
harder set, does show low resolution as its weakest stratum for false merges.)

Stability: the same images were clustered independently in two projects (`cocotest` and
`default`). The ARI between the two partitions over the 531 shared labelled crops is 0.37, and
the grouping is not identical. The two projects ran the unmodified pipeline and detector, so the
crop sets differ slightly (1,350 of 1,364 crop ids are shared) and part of the spread is that,
not clustering alone. This is a spread to record, not the "same seed, same vectors" comparison:
the plan's check that the IVF stage is identical across two runs on identical vectors still needs
a run that reuses stored vectors.

### B.3 Caveats

- Small and easy: 539 crops from 200 images, all ten classes are common COCO classes. It
  separates the metrics and is not a final evaluation set; the near-duplicate and mixed-scene
  strata are empty here (every labelled class is in a similar-class group, so mixed scenes fall
  into that stratum first). Baseline A is the larger set with those strata.
- Only labelled crops are scored. The 825 unlabelled crops share clusters with labelled ones, so
  a merge between an unlabelled object and a labelled one is invisible here.
- Pair metrics inside a stratum use only that stratum's items and the clusters as they are, so
  the strata do not add up to the whole.

## Cost of the harness itself

Cost of the harness itself (synthetic planted partitions, 200 classes, pairwise scoring /
stability / protected-moved, seconds on one CPU core):

| Items | Pairwise | Stability | Moved items |
|---|---|---|---|
| 20,000 | 0.005 | 0.008 | 0.018 |
| 100,000 | 0.03 | 0.02 | 0.09 |
| 1,000,000 | 0.15 | 0.14 | 1.05 |


## Caveats for both

- The stack's detector and PE encoder are the production ones at 0.6.0; a different engine or
  vector space gives different numbers. Test-only: no GPU benchmark was run for this note.

## Held-out scoring and the BCubed rows

The scorer now also reports BCubed precision, recall and F, AMI, risk-coverage (abstentions
lower coverage, never risk) and, when the labels carry `view` or `size_bin`, same-class pair
recall and false-merge rate for pairs that share or differ in that value. `identity` labels are
scored as a separate block. Unassigned items count as singleton clusters for BCubed and AMI.

The 3,000-item pin has a seeded split by source image (near-duplicate variants follow their
parent), balanced within each pin group:
`scripts/datasets/manifests/splits/cluster_eval_3000.json` (372 calibration and 372 heldout
source images). Anything tuned uses `calibration` only; the acceptance numbers come from
`heldout`:

```bash
.venv/bin/python scripts/bench/cluster_eval.py score --labels <labels.jsonl> --partition <partition.jsonl> \
    --split-manifest scripts/datasets/manifests/splits/cluster_eval_3000.json --split heldout \
    --scorecard-out <card.json> --kind K2 --level items --dataset-name cluster_eval_3000 --seed 1
```

The rows above were measured on all labelled crops, before any split existed and with the
COCO-run settings chosen on the same labels. They are not held-out numbers. The held-out BCubed
rows for the 200-image COCO partitions and the 3,000-item partition are at the end of this note.

### BCubed rows on the 200-image COCO sample (measured 2026-10-29)

Project `cocotest` (1,364 crops, production pipeline, nothing re-clustered), labelled from the
full COCO val2017 instances file rather than the 10-class sample file used above, so it labels
1,005 crops in 57 classes (people and other classes included) and is not comparable row for row
with the 539-crop table. Split: `splits/coco_sample_200.json`, seed 20261029, by source image
(88 calibration, 91 heldout images). Nothing was tuned on this set, so the heldout half is
clean for any later tuning on `calibration`. Three scoring repetitions are identical (the scorer is
deterministic; 0.03 to 0.05 s each). Host load average was 41 to 62 during the run, so only the
metrics, not the timings, are meaningful.

| Metric | All (1,005) | Heldout (519) |
|---|---|---|
| Pair precision | 0.576 | 0.627 |
| Pair recall | 0.029 | 0.042 |
| BCubed precision | 0.677 | 0.732 |
| BCubed recall | 0.148 | 0.203 |
| BCubed F | 0.242 | 0.318 |
| AMI | 0.336 | 0.344 |
| Risk (accepted items outside their cluster majority) | 0.256 | 0.222 |
| ARI against the `default` project's independent run | 0.33 | 0.41 |

Heldout strata (pair precision): low_res 0.62 (152), mixed_scene 0.72 (151), similar_classes 0.94
(189), rare 0.17 (15, too few to read). Raw JSON and scorecards:
`artifacts_local/cluster_search/w0/rescore/cocotest_{full,heldout}_r{1,2,3}.json` and `.card.json`
(gitignored). The 3,000-item partition is scored in the next section.

### BCubed rows on the 3,000-item set (measured 2026-10-29)

Project of 944 images (744 base plus 200 near-duplicate variants; public COCO and Open Images),
3,961 crops, 2,950 labelled in 86 classes, uploaded through `ingest_upload.py` into a throw-away
project, clustered by the production refresh pass (512 clusters), labels from
`scripts/research/cluster_diag/build_labels.py` (same matching rules as `cluster_eval_set.py build`;
an uploaded image is named by its client path, not its hash file name). Split:
`splits/cluster_eval_3000.json` (calibration 1,456, heldout 1,494 labelled crops). Nothing was tuned
on the heldout half. Scoring is deterministic: three repetitions per row are identical (0.06 to 0.2 s
each; host load average 5 to 22, so the times are not meaningful).

| Metric | All (2,950) | Calibration (1,456) | Heldout (1,494) |
|---|---|---|---|
| Clusters | 485 | 364 | 377 |
| Pair precision | 0.677 | 0.676 | 0.716 |
| Pair recall | 0.021 | 0.024 | 0.027 |
| BCubed precision | 0.732 | 0.767 | 0.784 |
| BCubed recall | 0.141 | 0.166 | 0.186 |
| BCubed F | 0.236 | 0.273 | 0.301 |
| AMI | 0.411 | 0.383 | 0.403 |
| Items outside their cluster's majority class | 0.208 | 0.183 | 0.175 |
| Mixed clusters | 228 | 129 | 133 |

Heldout pair precision by stratum: low_res 0.685 (397 items), mixed_scene 0.645 (391),
near_duplicate 0.884 (221), rare 0.938 (187), similar_classes 0.875 (271), other 1.000 (27). The all-items
row differs from the 944-image table above (0.696) because this project's refresh pass trained its
k-means on the 3,772 crops present when it fired during the upload and assigned the other 189 by
nearest centroid, so it is a different partition of the same images. BCubed and AMI rows of the
alternative methods (average linkage, Leiden, ECG) are in `cluster_config_baseline.md`.
Raw JSON and scorecards: `artifacts_local/cluster_search/w0b/e1_3000/score/` (gitignored); summary
`docs/benchmarks/cluster_config_baseline_3000.json`.
