# Cluster configuration baseline (E1 raw arms, E5) and the diagnostics in the repo

Measured 2026-10-29 on the stored PE-Core vectors of the public COCO 200-image project (1,364 crops,
544 labelled, 11 classes), CPU only, 6 to 8 threads, host load average 60 to 85 (shared host; nothing
here is a timing). Scripts as committed in `scripts/research/cluster_diag/` (the runs
were made from the working tree before lint-only edits and the removal of one unused helper; the
residual-pool fix is commit `0d650a15`). Raw outputs: `artifacts_local/cluster_search/w0b/` (`diag.json`,
`e1.json`, `e5.json`, logs; the reference numbers of the earlier scratch run are kept beside them).

The first sections below are on the COCO project; the 3,000-item set (86 classes) is measured in the last
sections, and its result differs from the COCO-sample direction. The row-order finding of the consolidated note (ARI 0.306 on the
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

## E1 and E5 on the 3,000-item set (measured 2026-10-29)

The pinned set (`scripts/datasets/manifests/cluster_eval_3000.json`, 744 base images plus 200
near-duplicate variants, COCO val2017 and Open Images, public data) was fetched, verified against the
pin and uploaded (`scripts/curation/ingest_upload.py`, 17 images/s) into a throw-away project of the
test stack (Triton on GPU 2; GPU 0 not needed). 944 images, 3,961 crops, all embedded, 512 clusters
from the product's own refresh pass; 2,950 crops match an annotation (86 classes; strata counts equal
the earlier 944-image project). The project was deleted afterwards; the exported vectors, labels and
scores are in `artifacts_local/cluster_search/w0b/e1_3000/` (`vecs.npy`, `vecs.meta.jsonl`,
`eval/labels.jsonl`, `e1.json`, `e5.json`, `arms/`, `arm_scores/`, `score/`). Summary committed:
`docs/benchmarks/cluster_config_baseline_3000.json`. The cut of every arm is chosen on the split
manifest's **calibration** half (1,456 labelled crops) and scored on its **heldout** half (1,494);
the numbers below are that direction only. The refresh pass here trained its k-means on the 3,772
crops present when it fired mid-upload and assigned the rest by nearest centroid, so the production
partition differs slightly from the earlier 944-image project (pair precision 0.677 against 0.696 on
all labelled crops); the partition is the one scored. Host load average was 5 to 44 (shared host), so
nothing here is a timing; the scorer is deterministic (three repetitions identical).

Reproduce (public data; the project slug is yours):

```bash
export OPENSEARCH_URL=http://127.0.0.1:<port>   # script client binds the project from the registry
.venv/bin/python scripts/curation/ingest_upload.py --root <dir>/images --project <slug> \
    --api-base http://127.0.0.1:<api>/curation --force-rehash
.venv/bin/python scripts/research/cluster_diag/fetch_vectors.py --opensearch-url $OPENSEARCH_URL \
    --project <slug> --out <out>/vecs
.venv/bin/python scripts/research/cluster_diag/build_labels.py --bundle <out>/vecs \
    --coco-gt <dir>/coco_gt.json --eval-manifest scripts/datasets/manifests/cluster_eval_3000.json --out-dir <out>/eval
for c in e1 e5; do .venv/bin/python scripts/research/cluster_diag/run.py $c --bundle <out>/vecs \
    --labels <out>/eval/labels.jsonl --coco-gt <dir>/coco_gt.json --out <out>/$c.json \
    --split-manifest scripts/datasets/manifests/splits/cluster_eval_3000.json; done
.venv/bin/python scripts/research/cluster_diag/export_partition.py --bundle <out>/vecs --arm ahc_average --param 0.41 --out <arm>.jsonl
.venv/bin/python scripts/bench/cluster_eval.py score --labels <out>/eval/labels.jsonl --partition <arm>.jsonl \
    --split-manifest scripts/datasets/manifests/splits/cluster_eval_3000.json --split heldout
```

(`run.py e1` took 54 minutes of one core on 3,961 crops; almost all of it is the 14 Leiden fits at 2 to
3 minutes each, so run it with the research environment's `igraph` and `leidenalg` on the path.)

### E1 on the heldout half

Cut chosen on calibration at the target recall, scored on heldout (precision / recall; unassigned share
in brackets where non-zero). The target-0.05 and 0.1 rows are new here: the 86-class set sits at pair
recall 0.03 to 0.1, below the 0.15 floor used on the COCO sample.

| Arm (raw vectors) | Target recall | Heldout P / R |
|---|---|---|
| IVF now (K = 512, the production configuration) | none | 0.734 / 0.029 |
| MiniBatchKMeans 2b (K = 8) | none | 0.187 / 0.273 |
| ECG, once | none | 0.376 / 0.200 (1.1 %) |
| Average-linkage AHC | >= 0.05 | 0.399 / 0.073 |
| | >= 0.10 | 0.330 / 0.126 |
| | >= 0.15 | 0.266 / 0.151 |
| | >= 0.20 | 0.258 / 0.260 |
| | >= 0.30 | 0.201 / 0.387 |
| | >= 0.50 | 0.168 / 0.521 |
| Complete-linkage AHC | >= 0.05 | 0.395 / 0.075 |
| | >= 0.10 | 0.310 / 0.128 |
| | >= 0.15 and 0.20 | 0.199 / 0.271 |
| | >= 0.30 and 0.50 | no cut reaches the recall |
| Leiden CPM (mutual 30-NN) | >= 0.05 | 0.637 / 0.066 (3.7 %) |
| | >= 0.10 and above | no cut reaches the recall (the best is 0.087 at gamma 0.02 of the mean weight) |

Heldout precision against recall, read along each method's whole curve (`e1.json`, `curves`; these
points are not calibration-chosen, they show the frontier):

| Heldout recall about | IVF now | Average-linkage AHC | Leiden CPM (unassigned) |
|---|---|---|---|
| 0.03 | 0.734 (0.029) | 0.62 (0.033) | 0.79 (0.036, 11 %) |
| 0.05 | none | 0.40 (0.050) | 0.71 (0.054, 4.7 %) |
| 0.07 | none | 0.40 (0.073) | 0.64 (0.066, 3.7 %) |
| 0.10 | none | 0.34 (0.103) | none (gamma 0.02: 0.52 at 0.087) |
| 0.26 | none | 0.26 (0.260) | none |

Row-order and seed stability: IVF now, rows as fetched, ARI 0.324 (the earlier note gave 0.306 on this
set and 0.308 on the COCO sample); after the id sort (#372) 1.000; average-linkage at a fixed cut 1.000;
MiniBatchKMeans 0.553; Leiden CPM 0.675 (three further seeds 0.642 to 0.688); ECG 0.641.

BCubed and AMI on the heldout half, scored with `scripts/bench/cluster_eval.py` on each arm's
partition (cut taken from the table above; `arm_scores/`; the unassigned count as singletons):

| Partition | Clusters | Pair P / R | BCubed P / R / F | AMI | Unassigned | Items outside their cluster majority |
|---|---|---|---|---|---|---|
| Production (stored `cluster_id`) | 377 | 0.716 / 0.027 | 0.784 / 0.186 / 0.301 | 0.403 | 0 | 0.175 |
| IVF refit by the diagnostics | 374 | 0.734 / 0.029 | 0.792 / 0.187 / 0.302 | 0.416 | 0 | 0.166 |
| Average linkage, cut 0.307 (target 0.05) | 347 | 0.399 / 0.073 | 0.741 / 0.285 / 0.412 | 0.456 | 0 | 0.202 |
| Average linkage, cut 0.41 (target 0.20) | 141 | 0.258 / 0.260 | 0.520 / 0.486 / 0.503 | 0.500 | 0 | 0.359 |
| Complete linkage, cut 0.762 | 21 | 0.199 / 0.271 | 0.273 / 0.538 / 0.362 | 0.428 | 0 | 0.580 |
| Leiden CPM, gamma 0.05 | 164 | 0.637 / 0.066 | 0.696 / 0.354 / 0.469 | 0.517 | 3.7 % | 0.243 |
| Leiden CPM, gamma 0.3 | 314 | 0.790 / 0.036 | 0.840 / 0.251 / 0.386 | 0.430 | 11 % | 0.146 |
| ECG, once | 26 | 0.376 / 0.200 | 0.416 / 0.519 / 0.462 | 0.547 | 1.1 % | 0.471 |
| MiniBatchKMeans 2b | 8 | 0.187 / 0.273 | 0.203 / 0.512 / 0.290 | 0.347 | 0 | 0.681 |

Production partition on all 2,950 labelled crops: pair P / R 0.677 / 0.021, BCubed 0.732 / 0.141 / F
0.236, AMI 0.411.

Per stratum (heldout, IVF now / average linkage at target 0.2 / Leiden at target 0.05, precision and
recall): short side under 32 px 0.75 / 0.057, 0.18 / 0.702, 0.64 / 0.099; mixed scene 0.76 / 0.018,
0.43 / 0.104, 0.60 / 0.034; near-duplicates 0.91 / 0.341, 0.74 / 0.584, 0.87 / 0.591; rare 0.92 / 0.339,
0.76 / 0.713, 0.86 / 0.615; similar classes 0.85 / 0.106, 0.64 / 0.457, 0.90 / 0.301.

What this says against the COCO-sample direction:

- **The claim that average linkage keeps the IVF precision at 1.5 to 3 times the recall does not hold on
  the 86-class set.** On the COCO sample (11 classes) average linkage held 0.87 to 0.91 at recall 0.14
  to 0.38. Here IVF is 0.73 at recall 0.03 and average linkage is already 0.62 at the same recall, 0.40
  at 1.7 times it and 0.26 at 9 times it. No average-linkage cut reaches IVF's precision at any higher
  recall. Complete linkage is no better, and the one-cut AHC has no precision anywhere near the COCO
  figures. The order of the two methods flips.
- **Leiden CPM is better than average linkage at low recall, the reverse of the COCO finding.**
  At recall about 0.05 to 0.07 it holds 0.64 to 0.71 against 0.40, at the price of 4 to 5 percent of
  crops left unassigned and (as before) an unstable fit (ARI 0.64 to 0.69 across seeds and row orders).
  It cannot reach recall 0.1 on the calibration half, so no operating point above 0.09 exists for it.
- **The coarser BCubed and AMI view is kinder to the alternatives**: average linkage at cut 0.307 and
  Leiden at gamma 0.05 raise BCubed F from 0.30 to 0.41 and 0.47 and AMI from 0.40 to 0.46 and 0.52, with
  BCubed precision down 0.04 to 0.09 and the outside-the-majority rate up from 0.17 to 0.20 and 0.24.
  That is a trade (more recall, more false merges), not a free gain, and the false merge is the costly
  error for propagating labels. ECG has the best AMI (0.55) and the worst pair precision of the useful
  arms.
- **E5 (below) is confirmed, strongly**, and it is the stronger lever on this set.

Likely reasons the COCO sample misled: 11 classes against 86 (an AHC cut that separates 11 well-spread
classes has far more near neighbours of other classes at 86), person crops are 28 percent of the labelled
crops and dominate the positive pairs, a third of the set is low-resolution crops (AHC precision 0.18 at
the target-0.2 cut), and the set adds near-duplicates and Open Images animals. The caveat that the
labelled crops are 74 percent of the project (unlabelled objects are invisible) applies as before.

### E5 on the 3,000-item set

Correction (wave W0c, X0): the first run of this section compared the detector's registry-slug class
name (`traffic_light`) with the ground-truth label in COCO form (`traffic light`) as raw strings, so
the agreement figures were low (72.4 %, 72.9 % and 75 %). Names now compare through
`normalize_class_name`, the product's one name rule. Re-run on the same bundle: 0.812 (confidence
under 0.5), 0.810 (under 0.7) and 0.820 (all labelled crops); raw output in
`artifacts_local/cluster_search/w0c/x0/e5.json`. The partitions and every precision and recall below
are unchanged, because the partition key is the slug and was consistent. The COCO sample's 11 class
names are single words, so its 87 % figures are unaffected.

Same method and split rule as above (average-linkage AHC inside each detector class against the same
method over the whole subset; one distance cut for all classes; fold 1 = cut chosen on calibration,
scored on heldout).

| Subset | Crops (labelled) | Detector class = label | Method, target recall >= 0.3 | Heldout P / R |
|---|---|---|---|---|
| confidence < 0.5 | 1,351 (707) | 81.2 % | global | 0.161 / 0.308 |
| | | | within detector class | **0.966 / 0.421** |
| confidence < 0.7 | 2,136 (1,282) | 81.0 % | global | 0.210 / 0.260 |
| | | | within detector class | **0.969 / 0.346** |

Across the eight cells of fold 1 (two subsets, four targets) partitioning inside the detector class has
heldout precision 0.956 to 0.969 where the global cut has 0.138 to 0.234, and its recall is at or above the
global recall in six cells; it is lower in two (0.445 against 0.463 at target 0.5 on the 0.5 subset, and
0.188 against 0.219 at target 0.15 on the 0.7 subset). Fold 2 (cut chosen on heldout, scored on
calibration) agrees: 0.885 to 0.932 against 0.121 to 0.253 (`e5.json`). Adoption rule (precision rises at equal recall on the residual-like subset): **met**, by a far
larger margin than on the COCO sample, because the global cut collapses at 86 classes. The caveats stand:
the detector class is a soft prior that agrees with the label for 81 percent of the subset, it caps
recall for crops whose detector class is wrong, and the detector is trained on the COCO label space (the
class agrees less for the Open Images animals it has no class for).

### Diagnostics (`run.py diag`) on the 3,000-item set

3,961 crops, 2,950 labelled, statistics fitted on the same independent 40,000-crop pool as the COCO
run. Raw vector geometry: mean random-pair cosine 0.464, effective rank 157, hub skew 2.78 (the COCO
sample gave 122, 2.57 and 0.50). Neighbour purity over the labelled crops (precision of the 10 nearest
labelled neighbours, P@10): raw 0.679, centred 0.682, PCA-whitened 256 0.686 (128: 0.646), CSLS k = 10
0.689 (centred 0.690), k-reciprocal 0.691, DBA 0.688; 1-NN precision 0.77 to 0.79 for all. Every
re-representation moves P@10 by under 0.015, and the strata with the lowest purity do not change (P@10
raw: low resolution 0.607, mixed scene 0.606, elongated boxes 0.645, short side under 32 px 0.604;
near-duplicates 0.807, rare 0.777). Hub skew falls from 2.78 to 0.75 (k-reciprocal) and 1.4 to 1.5
(CSLS, DBA) without a matching gain in purity, as on the COCO sample. The detector class equals the
label for 82 % of the labelled crops (88 % on the COCO sample).

Oracle ceilings (cut chosen on the scored labels, so an upper bound and not a held-out figure), best
pair precision at recall at least 0.1 / 0.2 / 0.3 / 0.5: raw average linkage 0.331 / 0.253 / 0.198 /
0.163; k-reciprocal 0.425 / 0.367 / 0.145 / none; DBA 0.402 / 0.215 / 0.189 / 0.159; whitened,
centred and CSLS variants are at or below the raw figures; complete linkage is below average linkage in
every cell. Leiden CPM reaches recall 0.1 in none of the nine similarity variants. Only k-reciprocal and
DBA lift the low-recall ceiling (0.40 to 0.43 against 0.33 at recall 0.1), on a precision scale that is
a fraction of the 0.87 to 0.91 the COCO sample showed at recall 0.14 to 0.38: the embeddings, not only the
algorithm, are the limit at 86 classes with 28 percent person crops and a quarter of the crops under
32 px. The embedding and processing benchmark should therefore be run on this set, not on the COCO
sample. Raw: `artifacts_local/cluster_search/w0b/e1_3000/diag.json` (it took 4 h 8 min of one core;
the Leiden curves dominate).
