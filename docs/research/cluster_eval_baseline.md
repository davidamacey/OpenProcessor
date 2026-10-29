# Cluster evaluation baseline

Measured 2026-10-28. This is the first baseline of the cluster-quality harness
(`scripts/bench/cluster_eval.py`, `scripts/bench/cluster_eval_metrics.py`,
`scripts/bench/cluster_eval_set.py`). It scores the partition the production pipeline (FAISS
IVF + AHC + residual, run by the refresh worker) already produced; it did not run or change any
clustering.

## What was measured

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

## Results (539 labelled crops, 128 clusters over them, 10 classes)

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
resolution the clear weak stratum, from the withdrawn numbers; on the re-scored partition its pair
precision is the second highest, so that reading is withdrawn until a larger set is scored.)

Stability: the same images were clustered independently in two projects (`cocotest` and
`default`). The ARI between the two partitions over the 531 shared labelled crops is 0.37, and
the grouping is not identical. The two projects ran the unmodified pipeline and detector, so the
crop sets differ slightly (1,350 of 1,364 crop ids are shared) and part of the spread is that,
not clustering alone. This is a spread to record, not the "same seed, same vectors" comparison:
the plan's check that the IVF stage is identical across two runs on identical vectors still needs
a run that reuses stored vectors.

Cost of the harness itself (synthetic planted partitions, 200 classes, pairwise scoring /
stability / protected-moved, seconds on one CPU core):

| Items | Pairwise | Stability | Moved items |
|---|---|---|---|
| 20,000 | 0.005 | 0.008 | 0.018 |
| 100,000 | 0.03 | 0.02 | 0.09 |
| 1,000,000 | 0.15 | 0.14 | 1.05 |

## Caveats

- Small and easy: 539 crops from 200 images, all ten classes are common COCO classes. It
  separates the metrics; it does not show a clearly weak stratum yet and is not a final
  evaluation set. The planned 3,000-item set with near-duplicate and Open Images strata is not
  built yet; the near-duplicate and mixed-scene strata are empty here (every labelled class is in
  a similar-class group, so mixed scenes fall into that stratum first).
- Only labelled crops are scored. The 825 unlabelled crops share clusters with labelled ones, so
  a merge between an unlabelled object and a labelled one is invisible here.
- Pair metrics inside a stratum use only that stratum's items and the clusters as they are, so
  the strata do not add up to the whole.
- The stack's detector and PE encoder are the production ones at 0.6.0; a different engine or
  vector space gives different numbers.
