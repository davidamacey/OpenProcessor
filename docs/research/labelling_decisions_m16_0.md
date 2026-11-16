# M16-0: labelling decisions per object, under two cost models (2026-11-16)

| Field | Value |
|---|---|
| Date | runs 2026-11-15 to 2026-11-16; this summary 2026-11-16 |
| Code | crop vectors from the product's crop encoder (`pe_core_l14_crop224`, 1,024 dimensions, the TensorRT plan of the engine track) through the export image built at commit `ae96d42a9`; crops by the product's letterbox rule (Pillow bilinear, fill 128) |
| Hardware | embedding: one RTX A6000, 108 s for 50,496 crops; simulation: one niced CPU thread per job on a shared host (load 15 to 25), so every time below is indicative |
| Data | public sets only: Stanford Cars (test split), CUB-200-2011, FGVC-Aircraft, COCO val2017. Set P was not used |
| Raw | per-run rows, logs and the harness are local artifacts (gitignored) under `artifacts_local/speed_1113/cluster_rule/`; this note carries every number plan 16 cites |
| Used by | [plan 16](../design/v070_rebuild/16_label_train_loop_plan.md) and its companions 16a, 16b |

Evidence and conditions only. The design that reads it is plan 16.

## 1. The person's job and the unit

A person points OpenProcessor at photos; the detector finds objects (car, bird, airplane); the
person labels their own fine classes (car models, bird species) and trains a model on them. The
unit here is **decisions per object**: the person's answers divided by the number of objects in the
library, until the library's labels reach a target accuracy. It is the cost to label a pool, not the
cost to train a good model: how many labels a good model needs was not measured (section 10).

## 2. Data

| Pool | Objects | Fine classes | Detector class |
|---|---|---|---|
| Stanford Cars, test split | 8,041 | 196 (about 41 per class) | from the product's detector box: car 6,064, truck 1,918, bus 31, none 28 |
| CUB-200-2011 | 11,788 | 200 (about 59 per class) | bird |
| FGVC-Aircraft | 10,000 | 100 (100 per class) | airplane |
| Three fine sets together | 10,000 sample and 29,829 | 496 | as above |
| Mixed: three fine sets plus COCO val2017 objects (sides at least 40 px; COCO car, truck, bus, bird and airplane left out) | 12,000 and 40,000 samples | 496 fine plus 75 COCO | fine sets as above; **COCO's detector class is simulated**: equal to the true class with probability 0.921, else another class (the 92.1 % of an earlier detector study, not a detector run) |

Sanity: leave-one-out recall at 1 is 0.912 (cars), 0.869 (CUB), 0.726 (aircraft), 0.709 (COCO).
Smaller pools (1,000 and 3,000) are seeded samples of each set.

## 3. The simulated person and the two cost models

One oracle labeller for every arm: it knows the true class, answers exactly, never tires. A class
is discovered only when an object of it is asked. Five seeds per cell; cells are mean ± standard
deviation over the seeds.

| Model | What the person does | What "98 %" means | Bias |
|---|---|---|---|
| **Gate** | answers a sample of a group (or of a band of a classifier's proposals); when the sample proves purity at a gate level (one-sided, alpha 0.025, at most one dissent), the unseen rest of the group take the class, errors included; a failing group is split into 4 (depth at most 3) and small groups are reviewed one by one | final label accuracy of the whole library at least 98 %, against the truth | labels reach objects the person never saw |
| **Glance, seen only** | sees pages of 30 objects (Cropwright's review page size) of one group or one proposed class; pays one decision per page plus one per wrong object | every label is on a page the person saw, so the final accuracy is 100 % by construction; "98 %" does not exist here | optimistic: it assumes every wrong object is spotted at a glance and that a page of 6 costs what a page of 30 costs |
| **Glance, unseen accepted** | as above, but proposals at or above a confidence are accepted without being shown | final accuracy at least 98 % | mixes both biases |

Arms: **A** clusters only (spherical k-means as the product fits it: faiss, 50 iterations; rules
K = min(512, n // 6) as shipped, n // 10, 20, 40, 80, 160, over all objects or per detector class);
**B** a classifier on the person's labels (the "head"): class mean (each class the mean of its
labelled unit vectors, cosine margin), logistic regression (C from {1, 10, 100} by 3-fold CV), kNN
(k = 10); cold start 100 decisions (random, or one per cluster of K = n // 40, largest first), then
rounds of 50 decisions growing 15 % (25 % for logistic regression) of the labelled set, least
certain first (`unc`), base order (`order`) or pages of the most confident proposals (`pages`);
confidence calibrated by isotonic regression on 5-fold out-of-fold predictions of the labelled
part; **B per detector class**: one independent head per detector class; **C** clusters (K = n // 40)
until 5 or 10 % of decisions are spent, then B with the propagated labels.

Three kinds of setting, named in every table:

- **Oracle-tuned**: the cheapest fixed setting of the grid whose mean accuracy over the 5 seeds
  reaches the target. The truth picks the setting, so a product cannot choose it this way.
- **Held-out**: the setting picked on seeds 0 to 2, scored on seeds 3 and 4 (winner's-curse check).
- **Truth-free**: knobs a product can fix in advance (gate level = target; the head stops when its
  own estimate reaches the target).

## 4. The three fine sets at full size

Decisions per object. Gate at 98 %, oracle-tuned and held-out (picked / scored, held-out accuracy);
glance seen only (5-seed mean, and held-out picked / scored); glance with unseen accepted at 98 %
(with the realised accuracy). The class-mean glance cell on aircraft is 0.31, not the 0.46 the
first write-up printed (section 9).

| Set | Arm | Gate 98 %, oracle-tuned | Gate 98 %, held-out | Glance, seen only | Glance, seen only, held-out | Glance, unseen accepted |
|---|---|---|---|---|---|---|
| Cars 8,041 | A shipped K = min(512, n // 6) | 0.51 ± 0.01 | 0.52 / 0.50 (0.982) | **0.20 ± 0.01** | 0.21 / 0.20 | 0.20 |
| | A K = n // 40 | 0.54 ± 0.02 | 0.54 / 0.53 (0.985) | 0.32 ± 0.02 | 0.32 / 0.31 | 0.32 |
| | A best K (n // 20 gate; shipped glance) | 0.49 ± 0.01 | 0.49 / 0.48 (0.979) | 0.20 ± 0.01 | 0.21 / 0.20 | 0.20 |
| | A K = n // 40 per detector class | 0.61 ± 0.01 | 0.61 / 0.61 (0.987) | 0.39 ± 0.01 | 0.39 / 0.40 | 0.39 |
| | B class mean | **0.26 ± 0.04** | 0.23 / 0.23 (0.976) | 0.23 ± 0.00 | 0.23 / 0.24 | 0.21 (0.986) |
| | B logistic regression | 0.44 ± 0.07 | 0.49 / 0.36 (0.988) | 0.29 ± 0.01 | 0.29 / 0.29 | 0.26 (0.987) |
| | B kNN | 0.35 ± 0.05 | 0.34 / 0.36 (0.983) | 0.29 ± 0.01 | 0.29 / 0.29 | 0.26 (0.982) |
| | C clusters then class mean | 0.33 ± 0.02 | 0.34 / 0.33 (0.986) | 0.30 ± 0.01 | 0.29 / 0.30 | 0.27 (0.992) |
| CUB 11,788 | A shipped | 0.54 ± 0.01 | 0.54 / 0.54 (0.982) | **0.20 ± 0.01** | 0.20 / 0.21 | 0.20 |
| | A K = n // 40 | 0.60 ± 0.01 | 0.60 / 0.59 (0.986) | 0.26 ± 0.01 | 0.26 / 0.26 | 0.26 |
| | B class mean | **0.38 ± 0.11** | 0.35 / 0.28 (0.976) | 0.23 ± 0.00 | 0.23 / 0.22 | 0.23 |
| | B logistic regression | 0.42 ± 0.05 | 0.42 / 0.43 (0.985) | 0.30 ± 0.01 | 0.30 / 0.30 | 0.28 (0.986) |
| | B kNN | 0.41 ± 0.04 | 0.47 / 0.39 (0.980) | 0.33 ± 0.01 | 0.33 / 0.34 | 0.31 (0.985) |
| | C clusters then class mean | 0.40 ± 0.02 | 0.40 / 0.38 (0.986) | 0.28 ± 0.00 | 0.27 / 0.28 | 0.28 |
| Aircraft 10,000 | A shipped | 0.76 ± 0.01 | 0.75 / 0.76 (0.989) | 0.37 ± 0.01 | 0.37 / 0.38 | 0.37 |
| | A K = n // 40 | 0.71 ± 0.01 | 0.70 / 0.72 (0.985) | 0.39 ± 0.01 | 0.39 / 0.40 | 0.39 |
| | A best K (n // 80 gate; shipped glance) | 0.68 ± 0.01 | 0.68 / 0.68 (0.981) | 0.37 ± 0.01 | 0.37 / 0.38 | 0.37 |
| | B class mean | **0.57 ± 0.11** | 0.57 / 0.43 (0.972) | **0.31 ± 0.00** | 0.31 / 0.31 | 0.31 |
| | B logistic regression | 0.62 ± 0.07 | 0.49 / 0.45 (0.977) | 0.32 ± 0.01 | 0.32 / 0.33 | 0.32 |
| | B kNN | 0.65 ± 0.00 | 0.65 / 0.65 (0.989) | 0.44 ± 0.01 | 0.43 / 0.44 | 0.44 |
| | C clusters then class mean | 0.54 ± 0.01 | 0.54 / 0.53 (0.979) | 0.35 ± 0.01 | 0.36 / 0.35 | 0.35 |

Truth-free settings (gate level = target, head stops at its own estimate = target), cost and
realised accuracy:

| Set | Target | A shipped | A K = n // 40 | B class mean, least certain first | C class mean |
|---|---|---|---|---|---|
| Cars | 95 % | 0.94 (0.999) | 0.91 (0.999) | 0.33 ± 0.02 (0.993) | 0.35 ± 0.03 (0.991) |
| | 98 % | 1.00 | 1.00 | 0.56 ± 0.13 (0.998) | 0.56 ± 0.14 (0.998) |
| CUB | 95 % | 0.93 (0.999) | 0.90 (0.998) | 0.57 ± 0.10 (0.994) | 0.60 ± 0.11 (0.994) |
| | 98 % | 0.99 | 0.98 | 0.82 ± 0.06 (0.999) | 0.78 ± 0.03 (0.998) |
| Aircraft | 95 % | 0.96 (0.999) | 0.93 (0.999) | 0.69 ± 0.05 (0.995) | 0.77 ± 0.05 (0.998) |
| | 98 % | 0.99 | 0.98 | 0.84 ± 0.10 (0.999) | 0.84 ± 0.03 (0.999) |

What the table supports:

- Under the gate, oracle-tuned, the class-mean head is the cheapest arm on all three sets and the
  order holds on held-out seeds (cars 0.23 against clusters 0.48, CUB 0.28 against 0.54, aircraft
  0.43 against 0.68). The cheapest cluster settings at 98 % used gate levels 0.6 to 0.7, whose proof
  is weak: they reach 98 % because the groups happen to be pure.
- Truth-free, clusters save nothing at 98 % (0.98 to 1.00) and the head costs 0.56 to 0.84 with seed
  spread up to 0.13.
- Under the glance with every label seen, clusters with the shipped rule are cheapest on cars and
  CUB (0.20 against the head's 0.23), the head is cheaper on aircraft (0.31 against 0.37). Accepting
  unseen proposals above a confidence saves at most 0.02 (cars, at accuracy 0.986).
- The class mean is the cheapest head in every model on all three sets; logistic regression ties it
  only on aircraft under the glance (0.32 against 0.31).
- Glance cells have small spread (at most 0.02) and hold on held-out seeds; gate cells of the head
  have spread 0.04 to 0.11.

## 5. Library size

Best of each arm at 98 % (gate, oracle-tuned) and seen-only glance; mean over 5 seeds; the head is
the class mean except where noted.

| Pool | n | Gate: A | Gate: B | Gate: C | Glance: A | Glance: B class mean | Glance: C |
|---|---|---|---|---|---|---|---|
| Cars | 1,000 | 0.91 | 0.64 | 0.74 | **0.54** | 0.70 | 0.73 |
| Cars | 3,000 | 0.73 | 0.40 | 0.47 | **0.34** | 0.38 | 0.41 |
| Cars | 8,041 | 0.49 | 0.26 | 0.33 | **0.20** | 0.23 | 0.30 |
| CUB | 1,000 | 0.90 | 0.72 | 0.77 | **0.55** | 0.73 | 0.78 |
| CUB | 3,000 | 0.73 | 0.50 | 0.58 | **0.35** | 0.45 | 0.48 |
| CUB | 11,788 | 0.54 | 0.38 | 0.40 | **0.20** | 0.23 | 0.28 |
| Aircraft | 1,000 | 0.92 | 0.84 | 0.85 | **0.56** | 0.67 | 0.72 |
| Aircraft | 3,000 | 0.78 | 0.62 | 0.65 | 0.45 | 0.45 | 0.51 |
| Aircraft | 10,000 | 0.68 | 0.57 | 0.54 | 0.37 | **0.31** | 0.35 |
| Three fine sets | 10,000 | 0.75 | 0.54 | 0.55 | **0.36** | 0.44 | 0.48 |
| Three fine sets | 29,829 | 0.57 | 0.38 | 0.39 | 0.27 | 0.27 | 0.32 |
| Mixed (COCO simulated) | 12,000 | 0.65 (per detector class) | 0.69 (kNN) | 0.70 (kNN) | **0.32** (per detector class) | 0.48 | 0.50 |
| Mixed (COCO simulated) | 40,000 | 0.49 (per detector class) | 0.65 (kNN) | 0.60 (kNN) | **0.23** (per detector class) | 0.33 | 0.36 |

Standard deviation of every cell at most 0.04, except B under the gate (up to 0.11) and C at 1,000
(0.08). Under the gate every cost falls as the library grows. Under the seen-only glance clusters
are cheapest or tied in 12 of 13 rows; the head is cheaper only on aircraft at 10,000 and ties at
29,829. There is no size above which the head wins under the glance.

## 6. Mixed pools and per-detector-class scopes

"Per detector class" = one k-means fit (or one head) per detector class; "all" = one over the pool.

| Pool | Arm | Gate 98 %, oracle-tuned | Gate, held-out | Glance, seen only |
|---|---|---|---|---|
| Three fine sets 29,829 | A best K, all | 0.57 | 0.57 / 0.58 | 0.27 |
| | A best K, per detector class | 0.59 | 0.59 / 0.60 | 0.27 |
| | B class mean, one head | 0.38 ± 0.00 | 0.38 / 0.38 | 0.27 |
| | B class mean, per detector class | 0.41 ± 0.01 | 0.41 / 0.40 | 0.34 ± 0.01 |
| Mixed 12,000 | A best K, all | 0.83 | 0.82 / 0.86 | 0.40 |
| | A best K, per detector class | 0.65 | 0.66 / 0.65 | 0.32 |
| | B class mean, one head | 0.72 ± 0.08 | 0.74 / 0.70 | 0.48 |
| | B kNN, one head | 0.69 ± 0.08 | 0.66 / 0.72 | 0.53 |
| | B class mean, per detector class | 0.67 ± 0.07 | 0.67 / 0.68 | 0.69 ± 0.04 |
| Mixed 40,000 | A shipped, all | 0.71 | 0.71 / 0.70 | 0.45 |
| | A best K, all | 0.66 | 0.66 / 0.67 | 0.29 |
| | A best K, per detector class | 0.49 | 0.48 / 0.49 | 0.23 |
| | B class mean, one head | 0.69 ± 0.05 | 0.64 / 0.90 | 0.33 |
| | B kNN, one head | 0.65 ± 0.11 | 0.65 / 0.67 | 0.39 |
| | B class mean, per detector class | 0.43 ± 0.00 | 0.43 / 0.43 | 0.41 |

- Clusters per detector class help on the mixed pools in both models (40,000: gate 0.66 to 0.49,
  glance 0.29 to 0.23), and change nothing or cost a little on the fine sets (cars split into car,
  truck, bus: gate 0.49 to 0.60, glance 0.20 to 0.25; CUB and aircraft have one detector class).
- The mixed-pool gain rests on the simulated COCO detector class, which is the truth 92.1 % of the
  time: a split by a label that is mostly the answer. It is not evidence about a real detector
  splitting a coarse class; the fine sets, which hold the person's case (one coarse class holding
  many fine classes), show no gain.
- For the head the split helps only under the gate at 40,000 (0.69 to 0.43 for the class mean) and
  costs under the glance on every pool (0.27 to 0.34; 0.48 to 0.69; 0.33 to 0.41).
- One head over a mixed pool is unstable across seeds under the gate (40,000, kNN: accuracy 0.985 /
  0.958 / 0.990 / 0.990 / 0.989 at costs 0.65 / 0.44 / 0.72 / 0.72 / 0.72; the class mean picked at
  0.64 scores 0.90 on held-out seeds). Over all 296 held-out comparisons, 41 differ by more than 0.03.

## 7. How many clusters

Cost of clusters only (one fit over all objects) per K rule, gate at a level fixed in advance (0.9,
realised accuracy 0.994 to 0.999) and seen-only glance:

| Pool | Model | shipped | n // 10 | n // 20 | n // 40 | n // 80 | n // 160 |
|---|---|---|---|---|---|---|---|
| Cars 1,000 | glance | **0.54** | 0.61 | 0.73 | 0.85 | 0.94 | 0.98 |
| Cars 3,000 | gate / glance | 0.97 / **0.34** | 0.95 / **0.34** | 0.96 / 0.45 | 0.96 / 0.64 | 0.97 / 0.81 | 0.97 / 0.92 |
| Cars 8,041 | gate / glance | 0.84 / **0.20** | 0.89 / 0.22 | 0.82 / 0.21 | **0.79** / 0.32 | 0.80 / 0.56 | 0.83 / 0.78 |
| CUB 11,788 | gate / glance | 0.81 / **0.20** | 0.90 / 0.23 | 0.82 / **0.20** | 0.77 / 0.26 | **0.76** / 0.44 | 0.78 / 0.69 |
| Aircraft 10,000 | gate / glance | 0.90 / **0.37** | 0.94 / 0.39 | 0.90 / **0.37** | 0.86 / 0.39 | **0.84** / 0.44 | 0.85 / 0.57 |
| Three fine sets 29,829 | gate / glance | **0.80** / 0.39 | 0.91 / 0.28 | 0.85 / **0.27** | **0.80** / 0.32 | **0.80** / 0.48 | 0.81 / 0.68 |
| Mixed 40,000 | gate / glance | **0.84** / 0.45 | 0.92 / 0.30 | 0.88 / **0.29** | 0.85 / 0.33 | **0.84** / 0.45 | 0.85 / 0.58 |

"shipped" is K = min(512, n // 6): clusters of 6 up to 3,072 objects, then 512 clusters (16 at
8,041, 58 at 29,829, 78 at 40,000). Standard deviation at most 0.02. The two models disagree: the
gate is flat from n // 40 to n // 160 and punishes clusters of 10; the glance rewards small, pure
clusters (one page each) and punishes n // 40 and coarser at every size. n // 20 sits within 0.05
of each model's best on the pools of 8,000 objects and more, and is worse than the shipped rule
under the glance at 1,000 and 3,000. No rule was measured above 40,000 objects.

## 8. Calibration, finding the classes, rare classes

Calibration at the snapshot where the head's own estimate first reached 0.9 (random start, then
least certain first; 5 seeds):

| Set, head | Estimate | True accuracy on the unlabelled part | Accepted at confidence ≥ 0.90 / 0.95 / 0.98 | Wrong share of accepted | Per proposed class at 0.95: mean / max wrong share |
|---|---|---|---|---|---|
| Cars, class mean | 0.912 | 0.974 | 66 % / 50 % / 14 % | 1.6 % / 1.2 % / 1.0 % | 0.8 % / 45 % |
| Cars, logistic regression | 0.917 | 0.910 | 75 % / 51 % / 21 % | 4.5 % / 2.9 % / 1.1 % | 2.6 % / 63 % |
| CUB, class mean | 0.915 | 0.962 | 57 % / 39 % / 16 % | 1.8 % / 1.5 % / 1.0 % | 1.2 % / 37 % |
| CUB, logistic regression | 0.921 | 0.938 | 72 % / 48 % / 35 % | 2.6 % / 1.6 % / 1.2 % | 2.8 % / 70 % |
| Three fine sets 29,829, class mean | 0.909 | 0.971 | 53 % / 22 % / 11 % | 1.6 % / 1.0 % / 1.0 % | 0.6 % / 51 % |

- The class mean's global estimate is conservative under this labelling order; per proposed class
  the worst class still has 37 % to 70 % wrong above 0.95. No calibration row exists for aircraft,
  for the mixed pools, or for a labelled set built by naming groups (typical objects first).

Cost of finding the classes (random order until every class has k examples; class mean; 5 seeds):

| Set | k = 1 / 5 / 10 |
|---|---|
| Cars | 0.14 / 0.34 / 0.48 |
| CUB | 0.09 / 0.21 / 0.33 |
| Aircraft | 0.05 / 0.12 / 0.18 |

A cold start of one object per cluster (K = n // 40, largest first, each a single decision) beat
random order on CUB and aircraft; random won on cars, within the spread. Labelling a whole group
page in one decision as the cold start was not measured. Farthest-point order as a start was run only at
3,000 objects or fewer and was never the best there.

Rare classes: on the three fine sets every class has at least 25 objects. In the 10,000-object
three-set pool, classes under 10 objects (148 objects) ended at 0.91 (head) and 0.92 (clusters then
head) against 0.97 (clusters only) at the 98 % setting; 11 such objects in the 40,000 mixed pool.
Too few objects to carry a rule.

## 9. Run time and memory (indicative: shared host, one niced thread)

faiss k-means K = 512: 2.3 s (8,041), 3.6 s (11,788), 18.8 s (29,829); per detector class K = 4,174:
7.6 s (40,000). Class-mean fit 0.14 s (8,041 labelled, 196 classes) to 0.71 s (40,000, 571
classes); out-of-fold calibration 0.8 to 4.2 s; kNN 1.9 to 45 s; logistic regression 2.7 to 3.2 s
per fit. Peak resident memory of one simulation job 0.5 GB (1,000 objects) to 3.8 GB (40,000), of
which about 0.45 GB is the loaded library.

Corrections to the first write-up of these rows (`results.md` of the raw directory):

1. The class mean under the glance on aircraft is 0.31 ± 0.00 (all 5 seeds 0.31 to 0.32), not 0.46:
   the first table dropped one seed whose `pages` run stopped one object short of the end and then
   picked another setting. The held-out sub-table (0.313) was right.
2. The first table's "V 98 %" cells for the head include proposals accepted unseen (accuracy 0.986
   on cars); the seen-only cells are the ones above.
3. "Clusters cheapest below about 8,000 objects, the head ties or wins from about 8,000" holds only
   for aircraft 10,000 and the 29,829 pool; section 5 has the full curve.
4. The one-head entries on the mixed pools under the gate were kNN; the class mean is 0.72 (12,000)
   and 0.69 (40,000), in section 6.

## 10. Not measured

- How a real person reviews a page: misses at a glance, seconds per page against seconds per single
  answer, whether a page of 6 costs what a page of 30 costs. Both models are bounds around it.
- A noisy labeller (09c's 10 % row) or one who stops early.
- A real detector on COCO and on the fine sets' mixed library; set P.
- Pools above 40,000 objects; any pool of 100,000 or 1,000,000 in decisions.
- How many labels, in which order, give a trained model of a given accuracy (the learning curve of
  the trained model), and whether a test holdout drawn from labels made least-certain-first is
  biased.
- Calibration under group-naming labels; per-class calibration.
- A head refreshed late (the simulation refits after every round of 50 decisions).
- Placement of new objects, refits during a session, renames and merges of classes.
- Logistic regression and kNN on the 29,829 and 40,000 pools with the `pages` policy.

## 11. Reproduce

The harness is local (gitignored), under `artifacts_local/speed_1113/cluster_rule/scripts/`; the
rows are under `rows/` beside it. Owed: commit the harness under `scripts/research/` so a clean
checkout can rerun it (plan 16, coordinator question).

```bash
# crops and vectors (once): prep_crops.py through prep_run.sh in the export image; embed_crops.py
# through embed_hold.sh on one GPU
scripts/prep_run.sh <set>
scripts/embed_hold.sh <set list>
# job lists and runs (5 seeds; arms A, B, C; stages s1 small pools, s2 full sets, s3 mixed)
python mkjobs.py s1 > ../jobs_s1.txt     # also s2, s3
scripts/run_jobs.sh jobs_s1.txt <parallel jobs> <timeout s>
# one job by hand
python run_exp.py --arm B --kind cars --size 0 --seed 0 --clf mean \
  --policies order:rand,unc:rand,pages:rand --out ../rows/B_cars_0_s0_mean.jsonl
# tables
python report.py ../rows ../out/report_full.md      # every setting, both models, held-out
python compact.py ../rows sets                       # also: curve, mixed
python report_extra.py ../rows                       # curve, tails, calibration, cost rows
python old_head.py cars ../rows_old_head_cars.json   # cost of finding the classes
python retab_m16_0.py ../rows                        # this note's tables, both models side by side
```

Library: `crlib.py` (pools, k-means, the gate and glance cost functions, heads), `crhead.py` (the
head loop and its endgames). Runs: numpy, scipy, scikit-learn and faiss in a local virtual
environment; one thread per job (`OMP_NUM_THREADS=1`).
