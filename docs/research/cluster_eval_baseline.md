# Cluster evaluation baseline

Measured 2026-10-28 and 2026-10-29 on the 0.6.0 image (#341). The cluster-quality harness is
`scripts/bench/cluster_eval.py`, `scripts/bench/cluster_eval_metrics.py` and
`scripts/bench/cluster_eval_set.py`; the evaluation set is pinned by
`scripts/datasets/manifests/cluster_eval_3000.json` and fetched by
`scripts/datasets/fetch_cluster_eval.py`. The baseline scores the partition the production
pipeline already produced (below); it did not change any clustering.

## The evaluation set

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

## What was measured

Project `ev-clu3k-d` on the test stack: 944 images ingested through
`POST /ingest/bulk` with the default policy (all crops embedded), 3,961 crops, 3,961 embedded.
The refresh worker then ran its retrain job. That job is the IVF residual pass: spherical FAISS
k-means, 512 centroids, 50 iterations, 5 restarts, seed 42, CPU, then nearest-centroid
assignment of every crop; auto-promote and the VLM stage are off by default and AHC refine is an
on-demand route, so neither ran. 2,950 crops match an annotation and are labelled (86 classes,
485 clusters over them); the other 1,011 crops (objects of unannotated classes) are left out.
No crop is protected, so the moved-items count is not exercised.

## Results (2,950 labelled crops)

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

## Repeatability and run-to-run spread

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

Cost of the harness itself (synthetic planted partitions, 200 classes, pairwise scoring /
stability / protected-moved, seconds on one CPU core):

| Items | Pairwise | Stability | Moved items |
|---|---|---|---|
| 20,000 | 0.005 | 0.008 | 0.018 |
| 100,000 | 0.03 | 0.02 | 0.09 |
| 1,000,000 | 0.15 | 0.14 | 1.05 |

## Caveats

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
