# Cluster configuration baseline (E1 raw arms, E5) and the diagnostics in the repo

Measured 2026-10-29 on the stored PE-Core vectors of the public COCO 200-image project (1,364 crops,
544 labelled, 11 classes), CPU only, 6 to 8 threads, host load average 60 to 85 (shared host; nothing
here is a timing). Scripts as committed in `scripts/research/cluster_diag/` (the runs
were made from the working tree before lint-only edits and the removal of one unused helper; the
residual-pool fix is commit `0d650a15`). Raw outputs: `artifacts_local/cluster_search/w0b/` (`diag.json`,
`e1.json`, `e5.json`, logs; the reference numbers of the earlier scratch run are kept beside them).

The 3,000-item set was **not** re-ingested for this note (it needs the throw-away stack and GPU 2); every
number below is on the COCO project. The row-order finding of the consolidated note (ARI 0.306 on the
3,000-item set) reproduces on the COCO vectors (0.308) and is fixed.

## Reproduce

```bash
# vectors and metadata of one project, read only (sorted by crop id)
.venv/bin/python scripts/research/cluster_diag/fetch_vectors.py \
    --opensearch-url http://127.0.0.1:<port> --project <slug> --out artifacts_local/cluster_search/w0b/coco
# diagnostics, E1, E5 (labels.jsonl: item_id, label, stratum; GT: the public COCO annotation file)
for c in diag e1 e5; do
  .venv/bin/python scripts/research/cluster_diag/run.py $c --bundle artifacts_local/cluster_search/w0b/coco \
      --labels <labels.jsonl> --coco-gt <coco_gt.json> --out artifacts_local/cluster_search/w0b/$c.json \
      $([ $c = diag ] && echo --fit-pool <independent vectors>.npy)
done
.venv/bin/python -m pytest tests/research_scripts -q --no-cov        # 38 hand-computed cases
```

The Leiden and ECG arms need `igraph` and `leidenalg`, which `requirements-test.txt` does not carry;
those tests skip without them. The fit pool is an independent set of 40,000 set P crop vectors (used
only to fit the centring, whitening and CSLS statistics, never scored).

## The diagnostics reproduce the consolidated note

`run.py diag` recomputes section 4 of the consolidated clustering note. Compared with the earlier scratch
run on the same inputs, every geometry figure (effective rank 122 and 104, hub skew 2.57 and 2.42, mean
random cosine 0.50 and 0.63), every neighbour-purity cell (P@1, P@5, P@10, vote accuracy and the
stratum columns for all nine similarities) and every average- and complete-linkage oracle-curve cell
agree to three decimals. The Leiden CPM oracle cells agree except eight cells that differ by 0.01 to 0.07
(for example k-reciprocal at recall 0.2: 0.805 here, 0.873 there). Leiden's quality moves with the seed
and with the edge order of the graph (see the seed spread below), so those cells are within the method's
own noise; they do not change a ranking in the note.

## E1: raw-vector arms on the held-out half

All 1,364 crops are clustered; the 544 labelled crops are scored. The labelled crops are split by source
image into two halves (262 and 282 crops, seed 20261029, sha256 ranking of the image id). For each
target recall the cut with the highest precision on one half that still reaches the recall is chosen,
and its precision / recall on the **other** half is reported; both directions are shown. This is a
direction, not a value: with 260 to 280 labelled crops per half the two folds disagree by up to 0.2 in
recall.

| Arm (raw vectors) | Target recall on the calibration half | Fold 1 held-out P / R | Fold 2 held-out P / R | Unassigned |
|---|---|---|---|---|
| IVF now (K = 227) | none | 0.882 / 0.094 | 0.898 / 0.130 | 0 |
| MiniBatchKMeans 2b (K = 8) | none | 0.359 / 0.555 | 0.310 / 0.612 | 0 |
| ECG, once (mutual 30-NN) | none | 0.740 / 0.539 | 0.687 / 0.638 | 0.8 to 1.1 % |
| Average-linkage AHC | >= 0.15 | 0.940 / 0.100 | 0.948 / 0.294 | 0 |
| | >= 0.20 | 0.873 / 0.137 | 0.907 / 0.379 | 0 |
| | >= 0.30 | 0.850 / 0.195 | 0.907 / 0.379 | 0 |
| | >= 0.50 | 0.710 / 0.427 | 0.398 / 0.603 | 0 |
| Complete-linkage AHC | >= 0.15 | 0.881 / 0.104 | 0.934 / 0.249 | 0 |
| | >= 0.20 | 0.849 / 0.197 | 0.883 / 0.307 | 0 |
| | >= 0.30 | 0.847 / 0.223 | 0.786 / 0.386 | 0 |
| | >= 0.50 | 0.330 / 0.535 | 0.296 / 0.517 | 0 |
| Leiden CPM (mutual 30-NN) | >= 0.15 and 0.20 | 0.893 / 0.132 | 0.858 / 0.511 | 2 to 8 % |
| | >= 0.30 | 0.909 / 0.240 | 0.858 / 0.511 | 2 to 3 % |
| | >= 0.50 | 0.923 / 0.378 | 0.741 / 0.568 | 2 % |

Row-order and seed stability (ARI between the exported order and a seeded permutation of the rows):

| Arm | ARI |
|---|---|
| IVF now, rows as fetched | **0.308** (the 3,000-item set gave 0.306) |
| IVF after the id sort (#372) | **1.000** |
| Average-linkage AHC at a fixed cut | 1.000 |
| MiniBatchKMeans 2b | 0.495 |
| Leiden CPM (gamma 0.3 of the mean weight) | 0.739; across four seeds 0.749 to 0.761 |
| ECG | 0.723 |

What the table says, against the kill rules of the consolidated note:

- **The current configuration is the limit, not the embeddings.** IVF at K = n // 6 scores precision
  0.88 to 0.90 at recall 0.09 to 0.13. Average-linkage AHC on the same vectors holds the same precision
  at a recall of 0.14 to 0.38 in the target 0.2 and 0.3 rows (0.873 / 0.137 to 0.907 / 0.379). The cut was
  chosen on the other half, so this is not an oracle threshold.
- **Average linkage dominates complete linkage** at every target from 0.15 up; at 0.5 both collapse
  (complete 0.30 to 0.33 precision, average 0.40 to 0.71).
- **Leiden does not beat average linkage at low recall** (0.15 to 0.3: 0.89 / 0.13 against 0.87 / 0.14
  in fold 1, and the fold-2 recall is not matched) and **does at 0.4 and above** (0.92 / 0.38 against
  0.71 / 0.43 in fold 1; 0.74 / 0.57 against 0.40 / 0.60 in fold 2). The kill rule ("not the default
  unless it beats average linkage at the chosen recall") therefore depends on the operating recall the
  product picks; it is not met in the 0.15 to 0.3 range. Leiden also leaves 2 to 8 % of crops
  unassigned and is not order- or seed-stable (ARI 0.74 to 0.76), where average linkage is exactly
  repeatable.
- **ECG, once** sits with modularity Leiden of the earlier note (precision 0.69 to 0.74, recall 0.54 to
  0.64); it does not reach the average-linkage frontier at recall 0.4. 2b MiniBatchKMeans at K = 8 is a
  bucket index, not a cluster set (precision 0.31 to 0.36).
- Per stratum (average linkage, target 0.2, held-out half 1): short side under 32 px 0.72 / 0.22,
  elongated boxes 0.64 / 0.10, boxes at least 96 px 0.98 / 0.17, rare classes 0.92 / 0.04. Small and
  elongated crops stay the weak spot, as the note found.
- Whitening and k-reciprocal arms are not scored here; they belong to the embedding and processing
  benchmark.

## E5: detector class as a prior on the low-confidence subset

The subset is the crops whose detector confidence is below a cut; the detector's class is a hard
partition key (average-linkage AHC inside each detector class, one distance cut for all classes)
against the same method over the whole subset. Same split and selection rule as E1.

| Subset | Crops (labelled) | Detector class = label | Method, target recall >= 0.3 | Fold 1 held-out P / R | Fold 2 held-out P / R |
|---|---|---|---|---|---|
| confidence < 0.5 | 432 (101) | 87.1 % | global | 0.461 / 0.454 | 0.885 / 0.174 |
| | | | within detector class | **0.937 / 0.525** | **0.973 / 0.545** |
| confidence < 0.7 | 730 (213) | 87.3 % | global | 0.696 / 0.452 | 0.889 / 0.176 |
| | | | within detector class | **0.966 / 0.524** | **0.960 / 0.228** |

Over the 16 cells (two subsets, four targets, two folds) partitioning inside the detector class has the
higher held-out precision in 14 and ties in two (both 1.00, at the two lowest targets of the 0.5 subset in
fold 2, where its recall is 0.167 against 0.174). Its recall is at least the global recall in 11 cells and
lower in five (by 0.003 to 0.007 in four, by 0.05 in one).
Adoption rule of the note
("precision rises at equal recall on the residual-like subset"): **met**. Caveats that keep this a
direction: 101 and 213 labelled crops, 11 labelled classes, a detector trained on the same label
space (the class agrees with the label for 87 % of the subset, which will be lower on the 86-class set
and in a residual pool of unknown classes), and an unconfident detector class is a soft prior, not a
veto. It also caps recall for crops whose detector class is wrong.

## The #372 fix

Both residual-pool fetch variants now return the pool sorted by document id
(`src/services/curation/clustering/residual_fetch.py`). The k-means seeds were already constants (42), so
the fit is a function of the document set. `tests/curation/test_residual_fetch_order_independence.py`
fetches the same documents in two shard orders and asserts identical pools and an identical IVF
partition; it first asserts that unsorted rows really give a different partition (ARI under 0.95), so the
test cannot pass by accident, and it fails on the previous code (5 of 5). Region-box clustering
(`region_box_clustering.py`) reads its rows by a separate scroll and was not changed; the Leiden, ECG
and 2b rows above show order and seed dependence that any new method must remove the same way.
